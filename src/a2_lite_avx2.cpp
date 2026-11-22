// AVX2+FMA A2-Lite (3 channels).
//
// Three channels would use 3 of the 8 lanes of a register, so this engine uses a different layout from the A2-Full
// one: every register holds 8 CONSECUTIVE FRAMES of ONE channel. All lanes do useful work, a weight is a broadcast
// scalar, and a tap's history for 8 frames is one unaligned load.
//
// A block of n frames is processed layer by layer, in groups of up to 3 vectors of 8 frames ("tiles"). One tile does
// the layer's convolution, the activation, the head sum and the 1x1 + residual update together, so nothing but
// the residual stream goes through memory between them. Each layer keeps its input history per channel in a
// power-of-two ring with a mirrored tail (as the A2-Full engine does) so a window never has to wrap.
//
// A block's last vector may be partly empty: its spare lanes compute garbage from stale ring data that nothing
// reads (a lane only ever reads positions at or before its own), and only n values are ever copied out or stored
// into a ring. Every lane runs the same arithmetic sequence whatever its position in a vector, so the output is
// bit-identical for any split of the input into blocks.
#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "a2_avx2.h"
#include "engine_util.h"
#include "cpu_features.h"

#ifdef NAMTURBO_PROFILE
  #include <cstdio>
  #if defined(_MSC_VER) && !defined(__clang__)
    #include <intrin.h>
  #else
    #include <x86intrin.h>
  #endif
#endif

namespace namturbo
{
#ifdef NAMTURBO_PROFILE
// Opt-in build (cmake -DNAMTURBO_PROFILE=ON): per-section time stamp counter totals, to see where a block's time goes.
namespace
{
enum { kProfInput, kProfRing, kProfTiles, kProfHead, kProfTotal, kProfBlocks, kProfCount };
uint64_t g_prof[kProfCount];
struct ProfScope
{
  explicit ProfScope(int i) : id(i), t0(__rdtsc()) {}
  ~ProfScope() { g_prof[id] += __rdtsc() - t0; }
  int id;
  uint64_t t0;
};
} // namespace
  #define NAMTURBO_CAT2(a, b) a##b
  #define NAMTURBO_CAT(a, b) NAMTURBO_CAT2(a, b)
  #define NAMTURBO_PROF(id) ProfScope NAMTURBO_CAT(prof_scope_, __LINE__)(id)
void a2_lite_avx2_profile_dump()
{
  const double b = double(g_prof[kProfBlocks]);
  if (b == 0)
    return;
  const char* names[] = {"input scaling", "ring writes", "layer tiles (conv etc.)", "head ring + conv", "TOTAL"};
  std::printf("   A2-Lite profile (TSC cycles per block, %.0f blocks):\n", b);
  for (int i = 0; i < kProfTotal + 1; ++i)
    std::printf("     %-26s %9.0f  (%4.1f%%)\n", names[i], double(g_prof[i]) / b, 100.0 * double(g_prof[i]) / double(g_prof[kProfTotal]));
}
#else
  #define NAMTURBO_PROF(id) ((void)0)
#endif

namespace
{
using engine_util::AlignedFloats;
using engine_util::next_pow2;
using engine_util::ScopedNoDenormals;

constexpr int C = 3;
constexpr int kMaxTaps = 15;
constexpr float kLeakySlope = 0.01f;

struct Layer
{
  int K = 0, D = 0, lookback = 0, R = 0, stride = 0; // stride: floats between one channel's ring and the next
  uint32_t mask = 0;
  float W[kMaxTaps * C * C] = {}; // [tap][in][out]
  float conv_b[C] = {}, mixin[C] = {}, l1x1[C * C] = {}, l1x1_b[C] = {}; // l1x1: [in][out]
  AlignedFloats ring;             // C channels of `stride` floats: R slots, then M mirrored, then 8 spare
};

inline int round_up8(int v) { return (v + 7) & ~7; }

// dst[0..count) = src[0..count), exactly count floats (the caller's buffers may end right there).
inline void copy_floats(float* dst, const float* src, int count)
{
  int i = 0;
  for (; i + 8 <= count; i += 8)
    _mm256_storeu_ps(dst + i, _mm256_loadu_ps(src + i));
  for (; i < count; ++i)
    dst[i] = src[i];
}

// Write n values from src at the ring slot of logical index `pos`, then refresh the mirror: slots [R, R+M) copy
// slots [0, M), so any window of up to M slots starting inside the ring is contiguous.
void ring_write(float* ring, int R, uint32_t mask, int M, uint32_t pos, const float* src, int n)
{
  const int wp = static_cast<int>(pos & mask);
  const int first = std::min(n, R - wp);
  copy_floats(ring + wp, src, first);
  if (first < n)
    copy_floats(ring, src + first, n - first);
  if (wp < M)
  {
    const int end = std::min(wp + first, M);
    copy_floats(ring + R + wp, ring + wp, end - wp);
  }
  if (first < n)
    copy_floats(ring + R, ring, n - first);
}

#if defined(_MSC_VER) && !defined(__clang__)
  #define NAMTURBO_FORCEINLINE __forceinline
#else
  #define NAMTURBO_FORCEINLINE inline __attribute__((always_inline))
#endif

// Activation, head sum and 1x1 + residual for one vector of 8 frames, given the three convolution outputs.
// LeakyReLU as max(z, slope*z): exact for slope in (0,1) and, unlike a branch, takes the same time whatever
// the data; a NaN stays a NaN. `vec` is the vector's index in the block; channel o of X and HS is o*Mp floats on.
NAMTURBO_FORCEINLINE void finish_vector(const Layer& L, __m256 z0, __m256 z1, __m256 z2, float* X, float* HS, int Mp, int vec,
                                   bool first, bool last)
{
  const __m256 slope = _mm256_set1_ps(kLeakySlope);
  const __m256 a0 = _mm256_max_ps(z0, _mm256_mul_ps(slope, z0));
  const __m256 a1 = _mm256_max_ps(z1, _mm256_mul_ps(slope, z1));
  const __m256 a2 = _mm256_max_ps(z2, _mm256_mul_ps(slope, z2));
  float* hs = HS + vec * 8;
  _mm256_storeu_ps(hs, first ? a0 : _mm256_add_ps(_mm256_loadu_ps(hs), a0));
  _mm256_storeu_ps(hs + Mp, first ? a1 : _mm256_add_ps(_mm256_loadu_ps(hs + Mp), a1));
  _mm256_storeu_ps(hs + 2 * Mp, first ? a2 : _mm256_add_ps(_mm256_loadu_ps(hs + 2 * Mp), a2));
  if (last)
    return; // the last layer's residual feeds nothing: the head reads only the activations
  float* x = X + vec * 8;
#define NAMTURBO_RESIDUAL(o)                                                                    {                                                                                          __m256 r = _mm256_set1_ps(L.l1x1_b[o]);                                                  r = _mm256_fmadd_ps(_mm256_set1_ps(L.l1x1[0 * C + (o)]), a0, r);                         r = _mm256_fmadd_ps(_mm256_set1_ps(L.l1x1[1 * C + (o)]), a1, r);                         r = _mm256_fmadd_ps(_mm256_set1_ps(L.l1x1[2 * C + (o)]), a2, r);                         _mm256_storeu_ps(x + (o) * Mp, _mm256_add_ps(_mm256_loadu_ps(x + (o) * Mp), r));       }
  NAMTURBO_RESIDUAL(0) NAMTURBO_RESIDUAL(1) NAMTURBO_RESIDUAL(2)
#undef NAMTURBO_RESIDUAL
}

// One tile of FV vectors (8*FV frames) of one layer, starting at vector v0 of the block.
// base[k] is the ring slot (channel 0) that frame 0 of the block reads for tap k; channel j is stride floats on.
//
// The 3 x FV accumulators are separate named variables, and the tap body is a macro, on purpose: written with
// small arrays and loops, MSVC kept the accumulators on the stack (every FMA became load, FMA, store: twice as
// slow as clang's version of the same source). Naming them leaves nothing for a compiler to demote to memory.
#define NAMTURBO_LITE_STEP(j)                                                                  \
  {                                                                                       \
    const __m256 h0 = _mm256_loadu_ps(h + (j) * stride);                                  \
    __m256 h1 = h0, h2 = h0;                                                              \
    if constexpr (FV > 1) h1 = _mm256_loadu_ps(h + (j) * stride + 8);                     \
    if constexpr (FV > 2) h2 = _mm256_loadu_ps(h + (j) * stride + 16);                    \
    __m256 wb = _mm256_broadcast_ss(w + (j) * C + 0);                                     \
    a00 = _mm256_fmadd_ps(wb, h0, a00);                                                   \
    if constexpr (FV > 1) a01 = _mm256_fmadd_ps(wb, h1, a01);                             \
    if constexpr (FV > 2) a02 = _mm256_fmadd_ps(wb, h2, a02);                             \
    wb = _mm256_broadcast_ss(w + (j) * C + 1);                                            \
    a10 = _mm256_fmadd_ps(wb, h0, a10);                                                   \
    if constexpr (FV > 1) a11 = _mm256_fmadd_ps(wb, h1, a11);                             \
    if constexpr (FV > 2) a12 = _mm256_fmadd_ps(wb, h2, a12);                             \
    wb = _mm256_broadcast_ss(w + (j) * C + 2);                                            \
    a20 = _mm256_fmadd_ps(wb, h0, a20);                                                   \
    if constexpr (FV > 1) a21 = _mm256_fmadd_ps(wb, h1, a21);                             \
    if constexpr (FV > 2) a22 = _mm256_fmadd_ps(wb, h2, a22);                             \
  }

template <int FV, int K>
inline void layer_tile(const Layer& L, const float* const* base, int v0, const float* IN, float* X, float* HS, int Mp,
                       bool first, bool last)
{
  const int stride = L.stride;
  const __m256 mx0 = _mm256_set1_ps(L.mixin[0]), mx1 = _mm256_set1_ps(L.mixin[1]), mx2 = _mm256_set1_ps(L.mixin[2]);
  const __m256 b0 = _mm256_set1_ps(L.conv_b[0]), b1 = _mm256_set1_ps(L.conv_b[1]), b2 = _mm256_set1_ps(L.conv_b[2]);
  const __m256 in0 = _mm256_loadu_ps(IN + v0 * 8);
  __m256 in1 = in0, in2 = in0;
  if constexpr (FV > 1) in1 = _mm256_loadu_ps(IN + (v0 + 1) * 8);
  if constexpr (FV > 2) in2 = _mm256_loadu_ps(IN + (v0 + 2) * 8);
  // aOF: output channel O, frame vector F of the tile.
  __m256 a00 = _mm256_fmadd_ps(mx0, in0, b0), a10 = _mm256_fmadd_ps(mx1, in0, b1), a20 = _mm256_fmadd_ps(mx2, in0, b2);
  __m256 a01 = a00, a11 = a10, a21 = a20, a02 = a00, a12 = a10, a22 = a20;
  if constexpr (FV > 1)
  {
    a01 = _mm256_fmadd_ps(mx0, in1, b0);
    a11 = _mm256_fmadd_ps(mx1, in1, b1);
    a21 = _mm256_fmadd_ps(mx2, in1, b2);
  }
  if constexpr (FV > 2)
  {
    a02 = _mm256_fmadd_ps(mx0, in2, b0);
    a12 = _mm256_fmadd_ps(mx1, in2, b1);
    a22 = _mm256_fmadd_ps(mx2, in2, b2);
  }
  for (int k = 0; k < K; ++k)
  {
    const float* h = base[k] + v0 * 8;
    const float* w = L.W + k * C * C;
    // The 3 input channels are a rolled loop on purpose: unrolled, MSVC hoisted all 18 loads of a tap ahead of the
    // 27 FMAs, needed about 27 registers, and spilled every accumulator to the stack on every tap.
    for (int j = 0; j < C; ++j, h += stride, w += C)
    {
      NAMTURBO_LITE_STEP(0)
    }
  }
  finish_vector(L, a00, a10, a20, X, HS, Mp, v0, first, last);
  if constexpr (FV > 1)
    finish_vector(L, a01, a11, a21, X, HS, Mp, v0 + 1, first, last);
  if constexpr (FV > 2)
    finish_vector(L, a02, a12, a22, X, HS, Mp, v0 + 2, first, last);
}
#undef NAMTURBO_LITE_STEP

class A2LiteAvx2 final : public A2Engine
{
public:
  explicit A2LiteAvx2(const A2Weights& w, uint32_t initial_position = 0) : initial_pos_(initial_position)
  {
    if (w.channels != C)
      throw std::invalid_argument("make_a2_lite_avx2: A2-Lite (3 channels) only");
    if (!cpu_supports_avx2_fma())
      throw std::runtime_error("make_a2_lite_avx2: this CPU or OS does not support AVX2+FMA");
    std::copy(w.rechannel.begin(), w.rechannel.begin() + C, rechannel_);
    for (int l = 0; l < kLayers; ++l)
    {
      const A2Layer& src = w.layers[l];
      Layer& L = layers_[l];
      if (src.k > kMaxTaps || src.conv.size() != size_t(src.k) * C * C)
        throw std::invalid_argument("make_a2_lite_avx2: unexpected layer shape");
      L.K = src.k;
      L.D = src.d;
      L.lookback = (src.k - 1) * src.d;
      std::copy(src.conv.begin(), src.conv.end(), L.W);
      std::copy(src.conv_b.begin(), src.conv_b.begin() + C, L.conv_b);
      std::copy(src.mixin.begin(), src.mixin.begin() + C, L.mixin);
      std::copy(src.l1x1.begin(), src.l1x1.begin() + C * C, L.l1x1);
      std::copy(src.l1x1_b.begin(), src.l1x1_b.begin() + C, L.l1x1_b);
    }
    for (int k = 0; k < kHeadK; ++k)
      std::copy(w.head_w[k].begin(), w.head_w[k].begin() + C, head_w_[k]);
    head_b_ = w.head_b;
    head_scale_ = w.head_scale;
  }

  void reset(int max_block) override
  {
    if (max_block < 1 || max_block > kMaxEngineBlock)
      throw std::invalid_argument("A2LiteAvx2::reset: max_block must be in 1..kMaxEngineBlock");
    M_ = max_block;
    Mp_ = round_up8(M_);
    pos_ = initial_pos_;
    for (Layer& L : layers_)
    {
      L.R = next_pow2(L.lookback + M_);
      L.mask = static_cast<uint32_t>(L.R - 1);
      L.stride = round_up8(L.R + M_ + 8);
      L.ring.assign_zero(size_t(C) * L.stride);
    }
    head_R_ = next_pow2(kHeadK - 1 + M_);
    head_mask_ = static_cast<uint32_t>(head_R_ - 1);
    head_stride_ = round_up8(head_R_ + M_ + 8);
    head_ring_.assign_zero(size_t(C) * head_stride_);
    X_.assign_zero(size_t(C) * Mp_);
    HS_.assign_zero(size_t(C) * Mp_);
    IN_.assign_zero(size_t(Mp_));
    Y_.assign_zero(size_t(Mp_));

    std::vector<float> zeros(static_cast<size_t>(M_), 0.0f), sink(static_cast<size_t>(M_));
    for (int left = a2_receptive_field(); left > 0;)
    {
      const int n = std::min(left, M_);
      process(zeros.data(), sink.data(), n);
      left -= n;
    }
  }

  void process(const float* in, float* out, int n) override
  {
    if (n < 1 || n > M_)
      throw std::invalid_argument("A2LiteAvx2::process: n outside 1..max_block (was reset() called?)");
    ScopedNoDenormals no_denormals;
    NAMTURBO_PROF(kProfTotal);
#ifdef NAMTURBO_PROFILE
    ++g_prof[kProfBlocks];
#endif
    const int nv = (n + 7) / 8; // vectors of 8 frames, the last possibly partly empty
    float* IN = IN_.data();
    float* X = X_.data();
    {
      NAMTURBO_PROF(kProfInput);
      copy_floats(IN, in, n); // `out` may be the same buffer as `in`: it is only written at the end
      for (int i = n; i < nv * 8; ++i)
        IN[i] = 0.0f;
      for (int c = 0; c < C; ++c)
      {
        const __m256 r = _mm256_set1_ps(rechannel_[c]);
        for (int v = 0; v < nv; ++v)
          _mm256_storeu_ps(X + c * Mp_ + v * 8, _mm256_mul_ps(r, _mm256_loadu_ps(IN + v * 8)));
      }
    }
    for (int l = 0; l < kLayers; ++l)
      run_layer(layers_[l], n, nv, /*first=*/l == 0, /*last=*/l == kLayers - 1);
    run_head(out, n, nv);
    pos_ += static_cast<uint32_t>(n);
  }

  int prewarm_samples() const override { return a2_receptive_field(); }

private:
  void run_layer(Layer& L, int n, int nv, bool first, bool last)
  {
    {
      NAMTURBO_PROF(kProfRing);
      for (int c = 0; c < C; ++c)
        ring_write(L.ring.data() + c * L.stride, L.R, L.mask, M_, pos_, X_.data() + c * Mp_, n);
    }
    NAMTURBO_PROF(kProfTiles);
    if (L.K == 6)
      layer_tiles<6>(L, nv, first, last);
    else
      layer_tiles<15>(L, nv, first, last); // load_a2_weights guarantees kernel sizes of 6 or 15
  }

  // The window for tap k starts at logical index pos_ - (K-1-k)*D. K is a compile-time parameter.
  template <int K>
  void layer_tiles(const Layer& L, int nv, bool first, bool last)
  {
    const float* base[K];
    for (int k = 0; k < K; ++k)
      base[k] = L.ring.data() + ((pos_ - uint32_t((K - 1 - k) * L.D)) & L.mask);
    int v = 0;
    for (; v + 3 <= nv; v += 3)
      layer_tile<3, K>(L, base, v, IN_.data(), X_.data(), HS_.data(), Mp_, first, last);
    switch (nv - v)
    {
      case 2: layer_tile<2, K>(L, base, v, IN_.data(), X_.data(), HS_.data(), Mp_, first, last); break;
      case 1: layer_tile<1, K>(L, base, v, IN_.data(), X_.data(), HS_.data(), Mp_, first, last); break;
      default: break;
    }
  }

  // Head: y = head_b + sum over 16 taps and 3 channels of w * head_sum, then scaled. One accumulator per vector,
  // up to 8 vectors at a time, each summed in the same order as the scalar reference.
  template <int T>
  void head_tile(const float* const* hb, int v0)
  {
    __m256 acc[T];
    for (int u = 0; u < T; ++u)
      acc[u] = _mm256_set1_ps(head_b_);
    for (int k = 0; k < kHeadK; ++k)
      for (int j = 0; j < C; ++j)
      {
        const __m256 wb = _mm256_set1_ps(head_w_[k][j]);
        for (int u = 0; u < T; ++u)
          acc[u] = _mm256_fmadd_ps(wb, _mm256_loadu_ps(hb[k] + j * head_stride_ + (v0 + u) * 8), acc[u]);
      }
    const __m256 scale = _mm256_set1_ps(head_scale_);
    for (int u = 0; u < T; ++u)
      _mm256_storeu_ps(Y_.data() + (v0 + u) * 8, _mm256_mul_ps(scale, acc[u]));
  }

  void run_head(float* out, int n, int nv)
  {
    NAMTURBO_PROF(kProfHead);
    for (int c = 0; c < C; ++c)
      ring_write(head_ring_.data() + c * head_stride_, head_R_, head_mask_, M_, pos_, HS_.data() + c * Mp_, n);
    const float* hb[kHeadK];
    for (int k = 0; k < kHeadK; ++k)
      hb[k] = head_ring_.data() + ((pos_ - uint32_t(kHeadK - 1 - k)) & head_mask_);
    int v = 0;
    for (; v + 8 <= nv; v += 8)
      head_tile<8>(hb, v);
    switch (nv - v)
    {
      case 7: head_tile<7>(hb, v); break;
      case 6: head_tile<6>(hb, v); break;
      case 5: head_tile<5>(hb, v); break;
      case 4: head_tile<4>(hb, v); break;
      case 3: head_tile<3>(hb, v); break;
      case 2: head_tile<2>(hb, v); break;
      case 1: head_tile<1>(hb, v); break;
      default: break;
    }
    copy_floats(out, Y_.data(), n);
  }

  float rechannel_[C] = {};
  Layer layers_[kLayers];
  float head_w_[kHeadK][C] = {};
  float head_b_ = 0.0f, head_scale_ = 0.0f;

  int M_ = 0, Mp_ = 0;
  uint32_t initial_pos_ = 0;
  uint32_t pos_ = 0; // logical index of the first frame of the next block; wraps harmlessly (rings are powers of two)
  int head_R_ = 0, head_stride_ = 0;
  uint32_t head_mask_ = 0;
  AlignedFloats head_ring_, X_, HS_, IN_, Y_;
};
} // namespace

std::unique_ptr<A2Engine> make_a2_lite_avx2(const A2Weights& weights)
{
  return std::make_unique<A2LiteAvx2>(weights);
}

std::unique_ptr<A2Engine> make_a2_lite_avx2_at_position(const A2Weights& weights, uint32_t initial_position)
{
  return std::make_unique<A2LiteAvx2>(weights, initial_position);
}
} // namespace namturbo
