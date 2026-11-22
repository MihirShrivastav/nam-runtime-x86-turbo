#include "a2_ref.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace namturbo
{
namespace
{
int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
}

// The model, per output sample (condition = the raw input sample):
//   x      = rechannel * in
//   per layer:  z = conv_b + mixin*in + sum_tap W_tap * x_history[t - (K-1-tap)*d]
//               a = LeakyReLU(z);  head_sum += a;  x += l1x1_b + l1x1 * a
//   out    = head_scale * (head_b + sum_tap head_w[tap] . head_sum_history[t - (15-tap)])
// Each layer keeps its own input history in a power-of-two ring indexed by the running sample count.
// T is the arithmetic type: float is the plain reference; double is the best available stand-in for the exact
// answer (same function: the LeakyReLU slope is the float 0.01f in both, as in the model files).
template <class T>
class A2RefT final : public A2Engine
{
public:
  explicit A2RefT(const A2Weights& w) : w_(w), C_(w.channels)
  {
    for (int l = 0; l < kLayers; ++l)
    {
      const int lookback = (w_.layers[l].k - 1) * w_.layers[l].d;
      mask_[l] = static_cast<uint32_t>(next_pow2(lookback + 1) - 1);
      hist_[l].assign(size_t(mask_[l] + 1) * C_, T(0));
    }
    head_mask_ = static_cast<uint32_t>(next_pow2(kHeadK) - 1);
    head_hist_.assign(size_t(head_mask_ + 1) * C_, T(0));
  }

  void reset(int max_block) override
  {
    if (max_block < 1 || max_block > kMaxEngineBlock)
      throw std::invalid_argument("A2Ref::reset: max_block must be in 1..kMaxEngineBlock");
    max_block_ = max_block;
    pos_ = 0;
    for (auto& h : hist_)
      std::fill(h.begin(), h.end(), T(0));
    std::fill(head_hist_.begin(), head_hist_.end(), T(0));

    std::vector<float> zeros(static_cast<size_t>(max_block), 0.0f);
    std::vector<float> sink(static_cast<size_t>(max_block));
    for (int left = a2_receptive_field(); left > 0;)
    {
      const int n = std::min(left, max_block);
      process(zeros.data(), sink.data(), n);
      left -= n;
    }
  }

  void process(const float* in, float* out, int n) override
  {
    if (n < 1 || n > max_block_)
      throw std::invalid_argument("A2Ref::process: n outside 1..max_block (was reset() called?)");
    for (int t = 0; t < n; ++t)
      out[t] = static_cast<float>(step(static_cast<T>(in[t])));
  }

  int prewarm_samples() const override { return a2_receptive_field(); }

private:
  T step(T in)
  {
    const int C = C_;
    const T slope = static_cast<T>(0.01f);
    T x[8], a[8], head_sum[8] = {};
    for (int c = 0; c < C; ++c)
      x[c] = static_cast<T>(w_.rechannel[c]) * in;

    for (int l = 0; l < kLayers; ++l)
    {
      const A2Layer& L = w_.layers[l];
      T* ring = hist_[l].data();
      const uint32_t mask = mask_[l];
      std::memcpy(ring + size_t(pos_ & mask) * C, x, size_t(C) * sizeof(T));

      for (int o = 0; o < C; ++o)
      {
        T z = static_cast<T>(L.conv_b[o]) + static_cast<T>(L.mixin[o]) * in;
        for (int k = 0; k < L.k; ++k)
        {
          const T* h = ring + size_t((pos_ - uint32_t((L.k - 1 - k) * L.d)) & mask) * C;
          const float* W = &L.conv[size_t(k) * C * C];
          for (int i = 0; i < C; ++i)
            z += static_cast<T>(W[i * C + o]) * h[i];
        }
        a[o] = z > T(0) ? z : slope * z;
        head_sum[o] += a[o];
      }
      for (int o = 0; o < C; ++o)
      {
        T r = static_cast<T>(L.l1x1_b[o]);
        for (int i = 0; i < C; ++i)
          r += static_cast<T>(L.l1x1[i * C + o]) * a[i];
        x[o] += r;
      }
    }

    std::memcpy(head_hist_.data() + size_t(pos_ & head_mask_) * C, head_sum, size_t(C) * sizeof(T));
    T y = static_cast<T>(w_.head_b);
    for (int k = 0; k < kHeadK; ++k)
    {
      const T* h = head_hist_.data() + size_t((pos_ - uint32_t(kHeadK - 1 - k)) & head_mask_) * C;
      for (int j = 0; j < C; ++j)
        y += static_cast<T>(w_.head_w[k][j]) * h[j];
    }
    ++pos_;
    return static_cast<T>(w_.head_scale) * y;
  }

  A2Weights w_;
  int C_;
  int max_block_ = 0;
  uint32_t pos_ = 0; // running sample count; wraps harmlessly because rings are powers of two
  std::vector<T> hist_[kLayers];
  uint32_t mask_[kLayers] = {};
  std::vector<T> head_hist_;
  uint32_t head_mask_ = 0;
};
} // namespace

std::unique_ptr<A2Engine> make_a2_ref(const A2Weights& weights)
{
  return std::make_unique<A2RefT<float>>(weights);
}

std::unique_ptr<A2Engine> make_a2_ref_f64(const A2Weights& weights)
{
  return std::make_unique<A2RefT<double>>(weights);
}
} // namespace namturbo
