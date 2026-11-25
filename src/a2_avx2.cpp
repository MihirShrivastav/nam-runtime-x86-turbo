// AVX2+FMA A2-Full (8 channels).
//
// Layout: one frame = 8 floats = one 256-bit register. A block of n frames is processed layer by layer: the
// dilated convolution in tiles of up to kConvTile (6) frames, then the activation, the head accumulation and the 1x1 + residual
// update, and at the end the head convolution. In the convolution the 8 products of one tap are summed on their own and then added to
// the running sum (NAMTURBO_TWO_LEVEL), which rounds less than one long chain of fused multiply-adds.
// Each layer keeps its input history in a power-of-two ring of 8-float columns with a mirrored tail
// (columns [R, R+M) copy columns [0, M)), so the n-column window every tap reads is contiguous even when
// it wraps. Each output frame's arithmetic is independent of the tile it falls in, so the output is
// bit-identical for any split of the input into blocks.
#include "a2_avx2.h"

#include <immintrin.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <new>
#include <stdexcept>
#include <vector>

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
enum { kProfRing, kProfConv, kProfAct, kProfResidual, kProfHead, kProfInput, kProfTotal, kProfBlocks, kProfCount };
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
void a2_avx2_profile_dump()
{
  const double b = double(g_prof[kProfBlocks]);
  const char* names[] = {"ring writes", "dilated conv", "activation + head sum", "1x1 + residual", "head ring + conv", "input scaling", "TOTAL"};
  std::printf("   profile (TSC cycles per block, %.0f blocks):\n", b);
  for (int i = 0; i < kProfTotal + 1; ++i)
    std::printf("     %-24s %9.0f  (%4.1f%%)\n", names[i], double(g_prof[i]) / b, 100.0 * double(g_prof[i]) / double(g_prof[kProfTotal]));
}
#else
  #define NAMTURBO_PROF(id) ((void)0)
#endif

namespace
{
constexpr int C = 8;
constexpr float kLeakySlope = 0.01f;

using engine_util::AlignedFloats;
using engine_util::next_pow2;

struct LayerState
{
  int R = 0;
  uint32_t mask = 0;
  AlignedFloats ring; // (R + M) columns of 8
};

// Write n columns from src at the ring position of logical index `pos`, then refresh the mirror.
void ring_write(float* ring, int R, uint32_t mask, int M, uint32_t pos, const float* src, int n)
{
  const int wp = static_cast<int>(pos & mask);
  const int first = std::min(n, R - wp);
  std::memcpy(ring + size_t(wp) * C, src, size_t(first) * C * sizeof(float));
  if (first < n)
    std::memcpy(ring, src + size_t(first) * C, size_t(n - first) * C * sizeof(float));
  if (wp < M) // columns [wp, min(wp+first, M)) are inside the mirrored zone
  {
    const int end = std::min(wp + first, M);
    std::memcpy(ring + size_t(R + wp) * C, ring + size_t(wp) * C, size_t(end - wp) * C * sizeof(float));
  }
  if (first < n) // wrapped columns [0, n-first), always inside the mirrored zone
    std::memcpy(ring + size_t(R) * C, ring, size_t(n - first) * C * sizeof(float));
}

#ifndef NAMTURBO_TWO_LEVEL
  #define NAMTURBO_TWO_LEVEL 1
#endif
// Frames per tile of the convolution: the two-level sum needs registers for the partial sums as well.
constexpr int kConvTile = NAMTURBO_TWO_LEVEL ? 6 : 8;

// One tile of T frames of the dilated convolution for one layer. H[k] points at tap k's contiguous
// window; frame t reads H[k] + t*8. Bias and input mixer start the accumulator.
template <int T, int K>
inline void conv_tile(const PackedA2FullLayer& L, const float* const* H, int t0, const float* in, float* Z, __m256 bias,
                      __m256 mixin)
{
  __m256 acc[T];
  for (int u = 0; u < T; ++u)
    acc[u] = _mm256_fmadd_ps(mixin, _mm256_set1_ps(in[t0 + u]), bias);
#if NAMTURBO_TWO_LEVEL
  // The 8 products of one tap are summed on their own and then added to the accumulator: the long sum is rounded K times at the
  // size of the result instead of 8 K times.
  for (int k = 0; k < K; ++k)
  {
    const float* h = H[k] + size_t(t0) * C;
    const float* w = L.W.data() + size_t(k) * 64;
    __m256 part[T];
    const __m256 w0 = _mm256_loadu_ps(w);
    for (int u = 0; u < T; ++u)
      part[u] = _mm256_mul_ps(w0, _mm256_broadcast_ss(h + u * C));
    for (int j = 1; j < C; ++j)
    {
      const __m256 wj = _mm256_loadu_ps(w + j * C);
      for (int u = 0; u < T; ++u)
        part[u] = _mm256_fmadd_ps(wj, _mm256_broadcast_ss(h + u * C + j), part[u]);
    }
    for (int u = 0; u < T; ++u)
      acc[u] = _mm256_add_ps(acc[u], part[u]);
  }
#else
  for (int k = 0; k < K; ++k)
  {
    const float* h = H[k] + size_t(t0) * C;
    const float* w = L.W.data() + size_t(k) * 64;
    for (int j = 0; j < C; ++j)
    {
      const __m256 wj = _mm256_loadu_ps(w + j * C); // the 8 outputs for input channel j
      for (int u = 0; u < T; ++u)
        acc[u] = _mm256_fmadd_ps(wj, _mm256_broadcast_ss(h + u * C + j), acc[u]);
    }
  }
#endif
  for (int u = 0; u < T; ++u)
    _mm256_storeu_ps(Z + size_t(t0 + u) * C, acc[u]);
}

// Sum of the 8 lanes.
inline float hsum(__m256 v)
{
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_shuffle_ps(s, s, 0x55));
  return _mm_cvtss_f32(s);
}

// Head convolution for T frames: per frame, 16 taps of (head weights . head_sum frame), accumulated
// lane-wise, then reduced. T independent accumulators hide the fused-multiply-add latency.
template <int T>
inline void head_tile(const __m256* hw, const float* const* Hh, int t0, float* Y)
{
  __m256 acc[T];
  for (int u = 0; u < T; ++u)
    acc[u] = _mm256_setzero_ps();
  for (int k = 0; k < kHeadK; ++k)
    for (int u = 0; u < T; ++u)
      acc[u] = _mm256_fmadd_ps(hw[k], _mm256_loadu_ps(Hh[k] + size_t(t0 + u) * C), acc[u]);
  for (int u = 0; u < T; ++u)
    Y[t0 + u] = hsum(acc[u]);
}

using engine_util::ScopedNoDenormals;

// True iff every float in in[0..n) has exact bit pattern +0.0f (0x00000000).
inline bool is_all_pos_zero(const float* in, int n)
{
  int i = 0;
  __m256i acc = _mm256_setzero_si256();
  for (; i + 8 <= n; i += 8)
    acc = _mm256_or_si256(acc, _mm256_loadu_si256(reinterpret_cast<const __m256i*>(in + i)));
  if (!_mm256_testz_si256(acc, acc))
    return false;
  uint32_t tail = 0;
  for (; i < n; ++i)
  {
    uint32_t bits = 0;
    std::memcpy(&bits, in + i, sizeof(bits));
    tail |= bits;
  }
  return tail == 0;
}

inline int trailing_pos_zeros(const float* in, int n)
{
  int z = 0;
  for (int i = n - 1; i >= 0; --i)
  {
    uint32_t bits = 0;
    std::memcpy(&bits, in + i, sizeof(bits));
    if (bits != 0)
      break;
    ++z;
  }
  return z;
}

class A2Avx2 final : public A2Engine
{
public:
  explicit A2Avx2(std::shared_ptr<const PackedA2Full> packed, uint32_t initial_position = 0)
    : packed_(std::move(packed))
    , initial_pos_(initial_position)
  {
    if (!packed_)
      throw std::invalid_argument("make_a2_avx2: packed weights are null");
    if (!cpu_supports_avx2_fma())
      throw std::runtime_error("make_a2_avx2: this CPU or OS does not support AVX2+FMA");
  }

  void reset(int max_block) override { reset(max_block, true); }

  void reset(int max_block, bool prewarm) override
  {
    if (max_block < 1 || max_block > kMaxEngineBlock)
      throw std::invalid_argument("A2Avx2::reset: max_block must be in 1..kMaxEngineBlock");
    M_ = max_block;
    pos_ = initial_pos_;
    silent_count_ = 0;
    idle_ready_ = false;
    idle_output_ = 0.0f;
    for (int l = 0; l < kLayers; ++l)
    {
      const PackedA2FullLayer& L = packed_->layers[l];
      LayerState& S = states_[l];
      S.R = next_pow2(L.lookback + M_);
      S.mask = static_cast<uint32_t>(S.R - 1);
      S.ring.assign_zero(size_t(S.R + M_) * C);
    }
    head_R_ = next_pow2(kHeadK - 1 + M_);
    head_mask_ = static_cast<uint32_t>(head_R_ - 1);
    head_ring_.assign_zero(size_t(head_R_ + M_) * C);
    X_.assign_zero(size_t(M_) * C);
    Z_.assign_zero(size_t(M_) * C);
    HS_.assign_zero(size_t(M_) * C);
    A_.assign_zero(size_t(M_) * C);
    Y_.assign_zero(size_t(M_));

    if (prewarm)
    {
      std::vector<float> zeros(static_cast<size_t>(M_), 0.0f), sink(static_cast<size_t>(M_));
      for (int left = a2_receptive_field(); left > 0;)
      {
        const int n = std::min(left, M_);
        process(zeros.data(), sink.data(), n);
        left -= n;
      }
    }
  }

  void process(const float* in, float* out, int n) override
  {
    if (n < 1 || n > M_)
      throw std::invalid_argument("A2Avx2::process: n outside 1..max_block (was reset() called?)");
    ScopedNoDenormals no_denormals;
    NAMTURBO_PROF(kProfTotal);
#ifdef NAMTURBO_PROFILE
    ++g_prof[kProfBlocks];
#endif
    const bool all_zero = is_all_pos_zero(in, n);
    if (all_zero && idle_ready_)
    {
      std::fill(out, out + n, idle_output_);
      return;
    }
    const int tz = all_zero ? n : trailing_pos_zeros(in, n);

    const PackedA2Full& P = *packed_;
    float* X = X_.data();
    const __m256 rechannel = _mm256_loadu_ps(P.rechannel);
    {
      NAMTURBO_PROF(kProfInput);
      for (int t = 0; t < n; ++t)
        _mm256_storeu_ps(X + size_t(t) * C, _mm256_mul_ps(rechannel, _mm256_set1_ps(in[t])));
    }
    for (int l = 0; l < kLayers; ++l)
      run_layer(P.layers[l], states_[l], in, n, /*first=*/l == 0, /*last=*/l == kLayers - 1);
    run_head(P, out, n);
    pos_ += static_cast<uint32_t>(n);

    constexpr int kRF = a2_receptive_field();
    silent_count_ = all_zero ? std::min(silent_count_ + n, kRF) : std::min(tz, kRF);
    idle_ready_ = (silent_count_ >= kRF);
    if (idle_ready_)
      idle_output_ = out[n - 1];
  }

  int prewarm_samples() const override { return a2_receptive_field(); }

private:
  void run_layer(const PackedA2FullLayer& L, LayerState& S, const float* in, int n, bool first, bool last)
  {
    float* X = X_.data();
    {
      NAMTURBO_PROF(kProfRing);
      ring_write(S.ring.data(), S.R, S.mask, M_, pos_, X, n);
    }
    {
      NAMTURBO_PROF(kProfConv);
      if (L.K == 6)
        conv_block<6>(L, S, in, n);
      else
        conv_block<15>(L, S, in, n); // load_a2_weights guarantees kernel sizes of 6 or 15
    }

    layer_rest(L, n, first, last);
  }

  // Dilated convolution for a block (AVX2): the window for tap k starts at logical index pos_ - (K-1-k)*D.
  // K is a compile-time parameter so the tap loop has a constant trip count (whether it is unrolled depends on the compiler).
  template <int K>
  void conv_block(const PackedA2FullLayer& L, const LayerState& S, const float* in, int n)
  {
    float* Z = Z_.data();
    const float* H[K];
    for (int k = 0; k < K; ++k)
      H[k] = S.ring.data() + size_t((pos_ - uint32_t((K - 1 - k) * L.D)) & S.mask) * C;
    const __m256 bias = _mm256_loadu_ps(L.conv_b), mixin = _mm256_loadu_ps(L.mixin);
    int t = 0;
    for (; t + kConvTile <= n; t += kConvTile)
      conv_tile<kConvTile, K>(L, H, t, in, Z, bias, mixin);
    switch (n - t)
    {
      case 7: conv_tile<7, K>(L, H, t, in, Z, bias, mixin); break;
      case 6: conv_tile<6, K>(L, H, t, in, Z, bias, mixin); break;
      case 5: conv_tile<5, K>(L, H, t, in, Z, bias, mixin); break;
      case 4: conv_tile<4, K>(L, H, t, in, Z, bias, mixin); break;
      case 3: conv_tile<3, K>(L, H, t, in, Z, bias, mixin); break;
      case 2: conv_tile<2, K>(L, H, t, in, Z, bias, mixin); break;
      case 1: conv_tile<1, K>(L, H, t, in, Z, bias, mixin); break;
      default: break;
    }
  }

  // Residual update for T frames: x += l1x1_b + sum_i W1_i * a_i (W1_i = the 8 outputs for input channel i).
  template <int T>
  static inline void residual_tile(const __m256* w1, __m256 bias, const float* A, float* X, int t0)
  {
    __m256 acc[T];
    for (int u = 0; u < T; ++u)
      acc[u] = bias;
    for (int i = 0; i < C; ++i)
      for (int u = 0; u < T; ++u)
        acc[u] = _mm256_fmadd_ps(w1[i], _mm256_broadcast_ss(A + size_t(t0 + u) * C + i), acc[u]);
    for (int u = 0; u < T; ++u)
    {
      float* x = X + size_t(t0 + u) * C;
      _mm256_storeu_ps(x, _mm256_add_ps(_mm256_loadu_ps(x), acc[u]));
    }
  }

  // Activation, head accumulation, 1x1 and residual for a block whose convolution output is in Z_.
  void layer_rest(const PackedA2FullLayer& L, int n, bool first, bool last)
  {
    const float* Z = Z_.data();
    float* A = A_.data();
    float* HS = HS_.data();
    float* X = X_.data();
    // LeakyReLU as max(z, slope*z): exact for slope in (0,1) (z>0 gives z, z<0 gives slope*z) and, unlike a
    // branch, takes the same time whatever the data; a NaN stays a NaN.
    const __m256 slope = _mm256_set1_ps(kLeakySlope);
    {
    NAMTURBO_PROF(kProfAct);
    for (int f = 0; f < n; ++f)
    {
      const __m256 z = _mm256_loadu_ps(Z + size_t(f) * C);
      const __m256 a = _mm256_max_ps(z, _mm256_mul_ps(slope, z));
      _mm256_storeu_ps(A + size_t(f) * C, a);
      float* hs = HS + size_t(f) * C;
      _mm256_storeu_ps(hs, first ? a : _mm256_add_ps(_mm256_loadu_ps(hs), a));
    }
    }
    if (last)
      return; // the last layer's residual feeds nothing: the head reads only the activations
    NAMTURBO_PROF(kProfResidual);
    __m256 w1[C];
    for (int i = 0; i < C; ++i)
      w1[i] = _mm256_loadu_ps(L.l1x1 + i * C);
    const __m256 bias = _mm256_loadu_ps(L.l1x1_b);
    int t = 0;
    for (; t + 8 <= n; t += 8)
      residual_tile<8>(w1, bias, A, X, t);
    switch (n - t)
    {
      case 7: residual_tile<7>(w1, bias, A, X, t); break;
      case 6: residual_tile<6>(w1, bias, A, X, t); break;
      case 5: residual_tile<5>(w1, bias, A, X, t); break;
      case 4: residual_tile<4>(w1, bias, A, X, t); break;
      case 3: residual_tile<3>(w1, bias, A, X, t); break;
      case 2: residual_tile<2>(w1, bias, A, X, t); break;
      case 1: residual_tile<1>(w1, bias, A, X, t); break;
      default: break;
    }
  }

  void run_head(const PackedA2Full& P, float* out, int n)
  {
    NAMTURBO_PROF(kProfHead);
    ring_write(head_ring_.data(), head_R_, head_mask_, M_, pos_, HS_.data(), n);
    const float* Hh[kHeadK];
    __m256 hw[kHeadK];
    for (int k = 0; k < kHeadK; ++k)
    {
      Hh[k] = head_ring_.data() + size_t((pos_ - uint32_t(kHeadK - 1 - k)) & head_mask_) * C;
      hw[k] = _mm256_loadu_ps(P.head_w[k]);
    }
    float* Y = Y_.data();
    int t = 0;
    for (; t + 8 <= n; t += 8)
      head_tile<8>(hw, Hh, t, Y);
    switch (n - t)
    {
      case 7: head_tile<7>(hw, Hh, t, Y); break;
      case 6: head_tile<6>(hw, Hh, t, Y); break;
      case 5: head_tile<5>(hw, Hh, t, Y); break;
      case 4: head_tile<4>(hw, Hh, t, Y); break;
      case 3: head_tile<3>(hw, Hh, t, Y); break;
      case 2: head_tile<2>(hw, Hh, t, Y); break;
      case 1: head_tile<1>(hw, Hh, t, Y); break;
      default: break;
    }
    for (int i = 0; i < n; ++i)
      out[i] = P.head_scale * (Y[i] + P.head_b);
  }

  std::shared_ptr<const PackedA2Full> packed_;
  LayerState states_[kLayers];

  int M_ = 0;
  uint32_t initial_pos_ = 0;
  uint32_t pos_ = 0; // logical index of the first frame of the next block; wraps harmlessly (rings are powers of two)
  int silent_count_ = 0;
  bool idle_ready_ = false;
  float idle_output_ = 0.0f;
  int head_R_ = 0;
  uint32_t head_mask_ = 0;
  AlignedFloats head_ring_, X_, Z_, HS_, A_, Y_;
};
} // namespace

std::unique_ptr<A2Engine> make_a2_avx2_packed(std::shared_ptr<const PackedA2Full> packed, uint32_t initial_position)
{
  return std::make_unique<A2Avx2>(std::move(packed), initial_position);
}

std::unique_ptr<A2Engine> make_a2_avx2(const A2Weights& weights)
{
  return make_a2_avx2_packed(pack_a2_full(weights), 0);
}

std::unique_ptr<A2Engine> make_a2_avx2_at_position(const A2Weights& weights, uint32_t initial_position)
{
  return make_a2_avx2_packed(pack_a2_full(weights), initial_position);
}
} // namespace namturbo
