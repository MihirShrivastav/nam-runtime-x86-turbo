#include "resampler.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "namturbo/error.h"

namespace namturbo
{
namespace
{
constexpr double kPi = 3.14159265358979323846;
constexpr int kMaxRatioFactor = 2000; // the reduced ratio of the two rates must have a numerator and a denominator within this

// Modified Bessel function of the first kind, order 0 (power series).
double bessel_i0(double x)
{
  double sum = 1.0, term = 1.0;
  const double q = 0.25 * x * x;
  for (int k = 1; k < 500; ++k)
  {
    term *= q / (double(k) * double(k));
    sum += term;
    if (term < 1e-18 * sum)
      break;
  }
  return sum;
}

double kaiser_beta(double a)
{
  if (a > 50.0)
    return 0.1102 * (a - 8.7);
  if (a >= 21.0)
    return 0.5842 * std::pow(a - 21.0, 0.4) + 0.07886 * (a - 21.0);
  return 0.0;
}

int round_up(int v, int multiple)
{
  return (v + multiple - 1) / multiple * multiple;
}

// Writes n floats into a power-of-two ring of length R with a mirrored tail of `mirror` floats ([R, R + mirror) mirrors [0, mirror)).
inline void mirrored_ring_write(float* ring, int R, std::uint32_t mask, int mirror, std::uint32_t pos, const float* src, int n)
{
  const int wp = static_cast<int>(pos & mask);
  const int first = std::min(n, R - wp);
  std::memcpy(ring + wp, src, static_cast<std::size_t>(first) * sizeof(float));
  if (first < n)
    std::memcpy(ring, src + first, static_cast<std::size_t>(n - first) * sizeof(float));
  if (wp < mirror)
  {
    const int end = std::min(wp + first, mirror);
    std::memcpy(ring + R + wp, ring + wp, static_cast<std::size_t>(end - wp) * sizeof(float));
  }
  if (first < n)
  {
    const int tail = std::min(n - first, mirror);
    std::memcpy(ring + R, ring, static_cast<std::size_t>(tail) * sizeof(float));
  }
}

// Sum of x[i] * c[i] for i < n (n a multiple of 8; c aligned to 32 bytes). Four accumulators of 8 lanes for multiples of 32,
// then 8-wide tail. Fixed summation order for a given n, so the result is bit-identical across block splits.
inline float dot_product(const float* x, const float* c, int n)
{
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  int t = 0;
  for (; t + 32 <= n; t += 32)
  {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t), _mm256_load_ps(c + t), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 8), _mm256_load_ps(c + t + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 16), _mm256_load_ps(c + t + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 24), _mm256_load_ps(c + t + 24), a3);
  }
  if (t + 8 <= n)
  {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t), _mm256_load_ps(c + t), a0);
    t += 8;
  }
  if (t + 8 <= n)
  {
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t), _mm256_load_ps(c + t), a1);
    t += 8;
  }
  if (t + 8 <= n)
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t), _mm256_load_ps(c + t), a2);
  const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  __m128 q = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  q = _mm_add_ps(q, _mm_movehl_ps(q, q));
  q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 0x55));
  return _mm_cvtss_f32(q);
}

struct Preset
{
  double attenuation_db;
  double pass_fraction;       // of the lower Nyquist frequency
  double outer_pass_fraction; // for the outer 4x/2x cascade stage (relaxed transition band)
};

Preset preset_of(ResamplerQuality quality)
{
  return quality == ResamplerQuality::LowLatency ? Preset{72.0, 0.80, 0.45} : Preset{96.0, 0.90, 0.50};
}
} // namespace

// ---- filter design ----------------------------------------------------------------------------------------------------

int kaiser_length(double attenuation_db, double transition_cycles_per_sample)
{
  const double dw = 2.0 * kPi * transition_cycles_per_sample;
  int n = static_cast<int>(std::ceil((attenuation_db - 7.95) / (2.285 * dw))) + 1;
  if (n % 2 == 0)
    ++n;
  return std::max(n, 3);
}

std::vector<double> design_lowpass(int n, double cutoff_cycles_per_sample, double attenuation_db, double dc_gain)
{
  if (n < 3 || n % 2 == 0)
    throw std::invalid_argument("design_lowpass: the length must be odd and at least 3");
  const int c = (n - 1) / 2;
  const double beta = kaiser_beta(attenuation_db);
  const double i0_beta = bessel_i0(beta);
  std::vector<double> h(static_cast<std::size_t>(n));
  double sum = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const double t = double(i - c);
    const double sinc = (i == c) ? 2.0 * cutoff_cycles_per_sample : std::sin(2.0 * kPi * cutoff_cycles_per_sample * t) / (kPi * t);
    const double r = t / double(c);
    const double window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
    h[static_cast<std::size_t>(i)] = sinc * window;
    sum += h[static_cast<std::size_t>(i)];
  }
  const double scale = dc_gain / sum;
  for (double& v : h)
    v *= scale;
  return h;
}

std::vector<double> design_halfband(double attenuation_db, double pass_fraction)
{
  if (!(pass_fraction > 0.0 && pass_fraction < 1.0))
    throw std::invalid_argument("design_halfband: pass_fraction must be in (0, 1)");
  // Transition band at the 2x rate: pass = 0.25 * pass_fraction, stop = 0.5 - pass => width = 0.5 * (1 - pass_fraction).
  int n = kaiser_length(attenuation_db, 0.5 * (1.0 - pass_fraction));
  if (((n - 1) / 2) % 2 == 0)
    n += 2; // require n = 4*M - 1 so the center index c = 2*M - 1 is odd and endpoints are non-zero
  if (n < 47)
    n += 4; // short Kaiser half-band filters need one extra pair to meet their stop-band target cleanly
  const int c = (n - 1) / 2;
  const double beta = kaiser_beta(attenuation_db);
  const double i0_beta = bessel_i0(beta);
  std::vector<double> h(static_cast<std::size_t>(n), 0.0);
  h[static_cast<std::size_t>(c)] = 0.5;
  double odd_sum = 0.0;
  for (int i = 0; i < n; ++i)
  {
    const int d = i - c;
    if ((d & 1) == 0)
      continue;
    const double t = double(d);
    const double sinc = std::sin(0.5 * kPi * t) / (kPi * t);
    const double r = t / double(c);
    const double window = bessel_i0(beta * std::sqrt(std::max(0.0, 1.0 - r * r))) / i0_beta;
    const double v = sinc * window;
    h[static_cast<std::size_t>(i)] = v;
    odd_sum += v;
  }
  const double scale = 0.5 / odd_sum;
  for (int i = 0; i < n; ++i)
  {
    if (((i - c) & 1) != 0)
      h[static_cast<std::size_t>(i)] *= scale;
  }
  return h;
}

// ---- half-band 2x upsampler -------------------------------------------------------------------------------------------

HalfBandUp2x::HalfBandUp2x(const std::vector<double>& taps, int max_input_block)
  : max_block_(max_input_block)
{
  const int n = static_cast<int>(taps.size());
  if (n < 3 || (n + 1) % 4 != 0 || max_input_block < 1)
    throw std::invalid_argument("HalfBandUp2x: taps length must be 4*M - 1 and max_input_block >= 1");
  M_ = (n + 1) / 4;
  const int c = 2 * M_ - 1;
  coeffs2_.assign_zero(static_cast<std::size_t>(M_));
  float* c2 = coeffs2_.data();
  for (int k = 0; k < M_; ++k)
    c2[k] = static_cast<float>(2.0 * taps[static_cast<std::size_t>(c - (2 * k + 1))]);
  const int history = 2 * M_ - 1;
  mirror_ = history + max_block_ + 8;
  R_ = engine_util::next_pow2(mirror_);
  mask_ = static_cast<std::uint32_t>(R_ - 1);
  ring_.assign_zero(static_cast<std::size_t>(R_ + mirror_));
}

void HalfBandUp2x::reset()
{
  std::memset(ring_.data(), 0, static_cast<std::size_t>(R_ + mirror_) * sizeof(float));
  pos_ = 0;
}

void HalfBandUp2x::write_ring(const float* src, int n)
{
  mirrored_ring_write(ring_.data(), R_, mask_, mirror_, pos_, src, n);
}

void HalfBandUp2x::process_split_piece(const float* in, int n, float* even_out, float* odd_out)
{
  const std::uint32_t block_start = pos_;
  write_ring(in, n);
  pos_ += static_cast<std::uint32_t>(n);

  const int history = 2 * M_ - 1;
  const std::uint32_t oldest = (block_start - static_cast<std::uint32_t>(history)) & mask_;
  const float* base = ring_.data() + oldest + history;
  const float* c2 = coeffs2_.data();

  std::memcpy(odd_out, base - (M_ - 1), static_cast<std::size_t>(n) * sizeof(float));

  int t = 0;
  for (; t + 8 <= n; t += 8)
  {
    __m256 acc = _mm256_setzero_ps();
    const float* b = base + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b - (M_ - 1 - k)), _mm256_loadu_ps(b - (M_ + k)));
      acc = _mm256_fmadd_ps(_mm256_set1_ps(c2[k]), sum, acc);
    }
    _mm256_storeu_ps(even_out + t, acc);
  }
  if (t < n)
  {
    __m256 acc = _mm256_setzero_ps();
    const float* b = base + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b - (M_ - 1 - k)), _mm256_loadu_ps(b - (M_ + k)));
      acc = _mm256_fmadd_ps(_mm256_set1_ps(c2[k]), sum, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    std::memcpy(even_out + t, tmp, static_cast<std::size_t>(n - t) * sizeof(float));
  }
}

void HalfBandUp2x::process_split(const float* in, int n, float* even_out, float* odd_out)
{
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_block_);
    process_split_piece(in + done, piece, even_out + done, odd_out + done);
    done += piece;
  }
}

void HalfBandUp2x::process_interleaved_piece(const float* in, int n, float* out2x)
{
  const std::uint32_t block_start = pos_;
  write_ring(in, n);
  pos_ += static_cast<std::uint32_t>(n);

  const int history = 2 * M_ - 1;
  const std::uint32_t oldest = (block_start - static_cast<std::uint32_t>(history)) & mask_;
  const float* base = ring_.data() + oldest + history;
  const float* c2 = coeffs2_.data();

  int t = 0;
  for (; t + 8 <= n; t += 8)
  {
    __m256 ev = _mm256_setzero_ps();
    const float* b = base + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b - (M_ - 1 - k)), _mm256_loadu_ps(b - (M_ + k)));
      ev = _mm256_fmadd_ps(_mm256_set1_ps(c2[k]), sum, ev);
    }
    const __m256 od = _mm256_loadu_ps(b - (M_ - 1));
    const __m256 lo = _mm256_unpacklo_ps(ev, od);
    const __m256 hi = _mm256_unpackhi_ps(ev, od);
    _mm256_storeu_ps(out2x + 2 * t, _mm256_permute2f128_ps(lo, hi, 0x20));
    _mm256_storeu_ps(out2x + 2 * t + 8, _mm256_permute2f128_ps(lo, hi, 0x31));
  }
  if (t < n)
  {
    __m256 ev = _mm256_setzero_ps();
    const float* b = base + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b - (M_ - 1 - k)), _mm256_loadu_ps(b - (M_ + k)));
      ev = _mm256_fmadd_ps(_mm256_set1_ps(c2[k]), sum, ev);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, ev);
    const float* od = b - (M_ - 1);
    for (int i = 0; i < n - t; ++i)
    {
      out2x[2 * (t + i)] = tmp[i];
      out2x[2 * (t + i) + 1] = od[i];
    }
  }
}

void HalfBandUp2x::process_interleaved(const float* in, int n, float* out2x)
{
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_block_);
    process_interleaved_piece(in + done, piece, out2x + 2 * done);
    done += piece;
  }
}

// ---- half-band 2x downsampler -----------------------------------------------------------------------------------------

HalfBandDown2x::HalfBandDown2x(const std::vector<double>& taps, int max_output_block)
  : max_out_block_(max_output_block)
{
  const int n = static_cast<int>(taps.size());
  if (n < 3 || (n + 1) % 4 != 0 || max_output_block < 1)
    throw std::invalid_argument("HalfBandDown2x: taps length must be 4*M - 1 and max_output_block >= 1");
  M_ = (n + 1) / 4;
  const int c = 2 * M_ - 1;
  coeffs_.assign_zero(static_cast<std::size_t>(M_));
  float* cf = coeffs_.data();
  for (int k = 0; k < M_; ++k)
    cf[k] = static_cast<float>(taps[static_cast<std::size_t>(c - (2 * k + 1))]);

  mirror0_ = (2 * M_ - 1) + max_out_block_ + 8;
  R0_ = engine_util::next_pow2(mirror0_);
  mask0_ = static_cast<std::uint32_t>(R0_ - 1);
  ring0_.assign_zero(static_cast<std::size_t>(R0_ + mirror0_));

  mirror1_ = M_ + max_out_block_ + 8;
  R1_ = engine_util::next_pow2(mirror1_);
  mask1_ = static_cast<std::uint32_t>(R1_ - 1);
  ring1_.assign_zero(static_cast<std::size_t>(R1_ + mirror1_));

  tmp_p0_.resize(static_cast<std::size_t>(max_out_block_));
  tmp_p1_.resize(static_cast<std::size_t>(max_out_block_));
}

void HalfBandDown2x::reset()
{
  std::memset(ring0_.data(), 0, static_cast<std::size_t>(R0_ + mirror0_) * sizeof(float));
  std::memset(ring1_.data(), 0, static_cast<std::size_t>(R1_ + mirror1_) * sizeof(float));
  pos0_ = 0;
  pos1_ = 0;
  phase_ = 0;
}

void HalfBandDown2x::write_ring(float* ring, int R, std::uint32_t mask, int mirror, std::uint32_t pos, const float* src, int n)
{
  mirrored_ring_write(ring, R, mask, mirror, pos, src, n);
}

void HalfBandDown2x::process_split_piece(const float* even_in, const float* odd_in, int n, float* out)
{
  const std::uint32_t start0 = pos0_;
  write_ring(ring0_.data(), R0_, mask0_, mirror0_, pos0_, even_in, n);
  pos0_ += static_cast<std::uint32_t>(n);
  write_ring(ring1_.data(), R1_, mask1_, mirror1_, pos1_, odd_in, n);
  pos1_ += static_cast<std::uint32_t>(n);

  const int h0 = 2 * M_ - 1;
  const std::uint32_t oldest0 = (start0 - static_cast<std::uint32_t>(h0)) & mask0_;
  const std::uint32_t oldest1 = (start0 - static_cast<std::uint32_t>(M_)) & mask1_;
  const float* base0 = ring0_.data() + oldest0 + h0;
  const float* base1 = ring1_.data() + oldest1; // logical index start0 - M_
  const float* cf = coeffs_.data();
  const __m256 half = _mm256_set1_ps(0.5f);

  int t = 0;
  for (; t + 8 <= n; t += 8)
  {
    __m256 acc = _mm256_mul_ps(half, _mm256_loadu_ps(base1 + t));
    const float* b0 = base0 + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b0 - (M_ - 1 - k)), _mm256_loadu_ps(b0 - (M_ + k)));
      acc = _mm256_fmadd_ps(_mm256_set1_ps(cf[k]), sum, acc);
    }
    _mm256_storeu_ps(out + t, acc);
  }
  if (t < n)
  {
    __m256 acc = _mm256_mul_ps(half, _mm256_loadu_ps(base1 + t));
    const float* b0 = base0 + t;
    for (int k = 0; k < M_; ++k)
    {
      const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b0 - (M_ - 1 - k)), _mm256_loadu_ps(b0 - (M_ + k)));
      acc = _mm256_fmadd_ps(_mm256_set1_ps(cf[k]), sum, acc);
    }
    alignas(32) float tmp[8];
    _mm256_store_ps(tmp, acc);
    std::memcpy(out + t, tmp, static_cast<std::size_t>(n - t) * sizeof(float));
  }
}

void HalfBandDown2x::process_split(const float* even_in, const float* odd_in, int n, float* out)
{
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_out_block_);
    process_split_piece(even_in + done, odd_in + done, piece, out + done);
    done += piece;
  }
}

int HalfBandDown2x::process_interleaved(const float* in2x, int n_in, float* out)
{
  int produced = 0;
  const int max_in_piece = 2 * max_out_block_ - 1; // guarantees n0 <= max_out_block_ and n1 <= max_out_block_
  for (int done = 0; done < n_in;)
  {
    const int chunk = std::min(n_in - done, max_in_piece);
    const float* src = in2x + done;
    int n0 = 0, n1 = 0;
    int ph = phase_;
    for (int i = 0; i < chunk; ++i)
    {
      if (ph == 0)
        tmp_p0_[static_cast<std::size_t>(n0++)] = src[i];
      else
        tmp_p1_[static_cast<std::size_t>(n1++)] = src[i];
      ph ^= 1;
    }
    phase_ = ph;

    const std::uint32_t start0 = pos0_;
    if (n0 > 0)
    {
      write_ring(ring0_.data(), R0_, mask0_, mirror0_, pos0_, tmp_p0_.data(), n0);
      pos0_ += static_cast<std::uint32_t>(n0);
    }
    if (n1 > 0)
    {
      write_ring(ring1_.data(), R1_, mask1_, mirror1_, pos1_, tmp_p1_.data(), n1);
      pos1_ += static_cast<std::uint32_t>(n1);
    }
    if (n0 > 0)
    {
      const int h0 = 2 * M_ - 1;
      const std::uint32_t oldest0 = (start0 - static_cast<std::uint32_t>(h0)) & mask0_;
      const std::uint32_t oldest1 = (start0 - static_cast<std::uint32_t>(M_)) & mask1_;
      const float* base0 = ring0_.data() + oldest0 + h0;
      const float* base1 = ring1_.data() + oldest1;
      const float* cf = coeffs_.data();
      const __m256 half = _mm256_set1_ps(0.5f);
      float* dst = out + produced;

      int t = 0;
      for (; t + 8 <= n0; t += 8)
      {
        __m256 acc = _mm256_mul_ps(half, _mm256_loadu_ps(base1 + t));
        const float* b0 = base0 + t;
        for (int k = 0; k < M_; ++k)
        {
          const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b0 - (M_ - 1 - k)), _mm256_loadu_ps(b0 - (M_ + k)));
          acc = _mm256_fmadd_ps(_mm256_set1_ps(cf[k]), sum, acc);
        }
        _mm256_storeu_ps(dst + t, acc);
      }
      if (t < n0)
      {
        __m256 acc = _mm256_mul_ps(half, _mm256_loadu_ps(base1 + t));
        const float* b0 = base0 + t;
        for (int k = 0; k < M_; ++k)
        {
          const __m256 sum = _mm256_add_ps(_mm256_loadu_ps(b0 - (M_ - 1 - k)), _mm256_loadu_ps(b0 - (M_ + k)));
          acc = _mm256_fmadd_ps(_mm256_set1_ps(cf[k]), sum, acc);
        }
        alignas(32) float tmp[8];
        _mm256_store_ps(tmp, acc);
        std::memcpy(dst + t, tmp, static_cast<std::size_t>(n0 - t) * sizeof(float));
      }
      produced += n0;
    }
    done += chunk;
  }
  return produced;
}

// ---- one rational polyphase stage -------------------------------------------------------------------------------------

PolyphaseStage::PolyphaseStage(int up, int down, const std::vector<double>& taps, int max_input_block)
  : up_(up)
  , down_(down)
  , max_input_block_(max_input_block)
{
  if (up < 1 || down < 1 || taps.empty() || max_input_block < 1)
    throw std::invalid_argument("PolyphaseStage: bad parameters");
  step_whole_ = down / up;
  step_fraction_ = down % up;
  const int n = static_cast<int>(taps.size());
  taps_per_phase_ = round_up((n + up - 1) / up, 8);
  coefficients_.assign_zero(static_cast<std::size_t>(up) * static_cast<std::size_t>(taps_per_phase_));
  float* c = coefficients_.data();
  // Row `phase` holds the taps phase, phase + up, phase + 2 up, ... in the order of the window (oldest input sample first).
  for (int phase = 0; phase < up; ++phase)
    for (int t = 0; t < taps_per_phase_; ++t)
    {
      const std::int64_t index = phase + static_cast<std::int64_t>(taps_per_phase_ - 1 - t) * up;
      c[static_cast<std::size_t>(phase) * taps_per_phase_ + t] = index < n ? static_cast<float>(taps[static_cast<std::size_t>(index)]) : 0.0f;
    }
  mirror_ = taps_per_phase_;
  R_ = engine_util::next_pow2(taps_per_phase_ - 1 + max_input_block_);
  mask_ = static_cast<std::uint32_t>(R_ - 1);
  ring_.assign_zero(static_cast<std::size_t>(R_ + mirror_));
}

void PolyphaseStage::reset()
{
  std::memset(ring_.data(), 0, static_cast<std::size_t>(R_ + mirror_) * sizeof(float));
  pos_ = 0;
  next_input_ = 0;
  phase_ = 0;
}

int PolyphaseStage::process_piece(const float* in, int n, float* out)
{
  const int history = taps_per_phase_ - 1;
  const std::uint32_t block_start = pos_;
  float* ring = ring_.data();
  mirrored_ring_write(ring, R_, mask_, mirror_, pos_, in, n);
  pos_ += static_cast<std::uint32_t>(n);

  const float* coefficients = coefficients_.data();
  int produced = 0;
  while (next_input_ < n)
  {
    const std::uint32_t read_idx = (block_start + static_cast<std::uint32_t>(next_input_) - static_cast<std::uint32_t>(history)) & mask_;
    out[produced++] = dot_product(ring + read_idx, coefficients + static_cast<std::size_t>(phase_) * taps_per_phase_, taps_per_phase_);
    next_input_ += step_whole_;
    phase_ += step_fraction_;
    if (phase_ >= up_)
    {
      phase_ -= up_;
      ++next_input_;
    }
  }
  next_input_ -= n;
  return produced;
}

int PolyphaseStage::process(const float* in, int n, float* out)
{
  int produced = 0;
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_input_block_);
    produced += process_piece(in + done, piece, out + produced);
    done += piece;
  }
  return produced;
}

// ---- 2x / 4x phase-interleaved oversampler ----------------------------------------------------------------------------

Oversampler::Oversampler(int factor, int max_block, ResamplerQuality quality)
  : factor_(factor)
  , max_block_(max_block)
{
  if ((factor != 2 && factor != 4) || max_block < 1)
    throw std::invalid_argument("Oversampler: factor must be 2 or 4 and max_block >= 1");
  const Preset preset = preset_of(quality);
  const std::vector<double> h1 = design_halfband(preset.attenuation_db, preset.pass_fraction);
  up1_ = std::make_unique<HalfBandUp2x>(h1, max_block);
  down1_ = std::make_unique<HalfBandDown2x>(h1, max_block);
  const int c1 = up1_->delay_2x_samples(); // 2*M1 - 1

  if (factor == 2)
  {
    latency_ = c1; // exact integer delay in 1x samples
  }
  else
  {
    const std::vector<double> h2 = design_halfband(preset.attenuation_db, preset.outer_pass_fraction);
    up2_ = std::make_unique<HalfBandUp2x>(h2, 2 * max_block);
    down2_ = std::make_unique<HalfBandDown2x>(h2, 2 * max_block);
    const int m2 = up2_->half_taps();
    // Stage 1 delay = c1 (1x), stage 2 delay = c2 = 2*m2 - 1 (at 2x) plus 1 sample at 2x = 2*m2 (at 2x) = m2 (at 1x).
    latency_ = c1 + m2;
    mid0_.resize(static_cast<std::size_t>(max_block));
    mid1_.resize(static_cast<std::size_t>(max_block));
    mid96_.resize(static_cast<std::size_t>(2 * max_block));
    mid192_e_.resize(static_cast<std::size_t>(2 * max_block));
    mid192_o_.resize(static_cast<std::size_t>(2 * max_block));
    down96_.resize(static_cast<std::size_t>(2 * max_block));
  }
  reset();
}

void Oversampler::reset()
{
  up1_->reset();
  down1_->reset();
  if (up2_)
    up2_->reset();
  if (down2_)
    down2_->reset();
  delay96_ = 0.0f;
}

void Oversampler::upsample(const float* in, int n, float* const* phases)
{
  if (n <= 0)
    return;
  if (factor_ == 2)
  {
    up1_->process_split(in, n, phases[0], phases[1]);
    return;
  }
  // 4x: 1x -> 2x interleaved (2*n) -> 4x split into 4 phase streams of length n
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_block_);
    up1_->process_interleaved(in + done, piece, mid96_.data());
    up2_->process_split(mid96_.data(), 2 * piece, mid192_e_.data(), mid192_o_.data());
    // y4[2*k] = mid192_e_[k], y4[2*k + 1] = mid192_o_[k]
    // For k = 2*i: y4[4*i] = e[2*i] (phase 0), y4[4*i+1] = o[2*i] (phase 1)
    // For k = 2*i+1: y4[4*i+2] = e[2*i+1] (phase 2), y4[4*i+3] = o[2*i+1] (phase 3)
    float* p0 = phases[0] + done;
    float* p1 = phases[1] + done;
    float* p2 = phases[2] + done;
    float* p3 = phases[3] + done;
    const float* e = mid192_e_.data();
    const float* o = mid192_o_.data();
    for (int i = 0; i < piece; ++i)
    {
      p0[i] = e[2 * i];
      p1[i] = o[2 * i];
      p2[i] = e[2 * i + 1];
      p3[i] = o[2 * i + 1];
    }
    done += piece;
  }
}

void Oversampler::downsample(const float* const* phases, int n, float* out)
{
  if (n <= 0)
    return;
  if (factor_ == 2)
  {
    down1_->process_split(phases[0], phases[1], n, out);
    return;
  }
  // 4x: 4 phases of length n -> down2 (2*n samples at 2x) -> 1-sample alignment delay at 2x -> down1 (n samples at 1x)
  for (int done = 0; done < n;)
  {
    const int piece = std::min(n - done, max_block_);
    const float* p0 = phases[0] + done;
    const float* p1 = phases[1] + done;
    const float* p2 = phases[2] + done;
    const float* p3 = phases[3] + done;
    float* e = mid192_e_.data();
    float* o = mid192_o_.data();
    for (int i = 0; i < piece; ++i)
    {
      e[2 * i] = p0[i];
      o[2 * i] = p1[i];
      e[2 * i + 1] = p2[i];
      o[2 * i + 1] = p3[i];
    }
    down2_->process_split(e, o, 2 * piece, down96_.data());
    // Apply 1-sample delay at 2x rate and de-interleave into mid0_ (even) and mid1_ (odd) for down1_
    float prev = delay96_;
    const float* d = down96_.data();
    float* m0 = mid0_.data();
    float* m1 = mid1_.data();
    for (int i = 0; i < piece; ++i)
    {
      m0[i] = prev;
      m1[i] = d[2 * i];
      prev = d[2 * i + 1];
    }
    delay96_ = prev;
    down1_->process_split(m0, m1, piece, out + done);
    done += piece;
  }
}

// ---- the pair of rate-conversion stages -------------------------------------------------------------------------------

RateConverter::RateConverter(int host_rate, int model_rate, int max_block, ResamplerQuality quality)
{
  if (host_rate < 1 || model_rate < 1 || max_block < 1)
    throw Error(ErrorCode::InvalidArgument, "the sample rates and the block size must be positive");
  const int g = std::gcd(host_rate, model_rate);
  up_ = model_rate / g;
  down_ = host_rate / g;
  if (up_ > kMaxRatioFactor || down_ > kMaxRatioFactor)
    throw Error(ErrorCode::Unsupported, "the ratio of the sample rates (" + std::to_string(host_rate) + " Hz and " + std::to_string(model_rate) +
                                          " Hz) is too complex to convert; use a standard sample rate such as 44100, 48000, 88200 or 96000 Hz");

  const Preset preset = preset_of(quality);
  int stage2_max = 0;

  // For 4x host rates (e.g. 192 kHz -> 48 kHz), cascade a short symmetric half-band outer stage (192 <-> 96 kHz,
  // transition band 24..72 kHz) with a steep polyphase inner stage (96 <-> 48 kHz, stop-band at 24 kHz).
  // All other ratios use the upgraded polyphase FIR (8-wide padding, mirrored ring, block FIFO) so stop-band
  // attenuation is reached right at the lower Nyquist frequency.
  if (up_ == 1 && down_ == 4)
  {
    mode_ = Mode::HybridDown4;
    // Outer 4x <-> 2x half-band has a wide transition band, so +24 dB extra attenuation costs only 2 pairs and keeps pass-band
    // ripple below -130 dB.
    const std::vector<double> h_outer = design_halfband(preset.attenuation_db + 24.0, preset.outer_pass_fraction);
    const int max_mid = (max_block + 1) / 2;
    hb_down2_ = std::make_unique<HalfBandDown2x>(h_outer, max_mid); // 4x -> 2x
    const int c_outer = hb_down2_->delay_2x_samples();

    const int mid_rate = 2 * model_rate;
    const double grid = static_cast<double>(mid_rate); // up_inner = 1, down_inner = 2
    const double f_stop = 0.5 * static_cast<double>(model_rate);
    const double f_pass = preset.pass_fraction * f_stop;
    const int n1 = kaiser_length(preset.attenuation_db, (f_stop - f_pass) / grid);
    const int c1 = (n1 - 1) / 2;
    const double cutoff = 0.5 * (f_pass + f_stop) / grid;

    max_model_block_ = (max_mid + 1) / 2 + 1;
    stage1_ = std::make_unique<PolyphaseStage>(1, 2, design_lowpass(n1, cutoff, preset.attenuation_db, 1.0), max_mid);
    stage2_ = std::make_unique<PolyphaseStage>(2, 1, design_lowpass(n1, cutoff, preset.attenuation_db, 2.0), max_model_block_);
    const int mid_out_max = stage2_->max_output(max_model_block_);
    hb_up2_ = std::make_unique<HalfBandUp2x>(h_outer, mid_out_max); // 2x -> 4x
    hb_mid_.resize(static_cast<std::size_t>(std::max(max_mid, mid_out_max)));

    // Exact delays at 4x (host) rate: outer stage contributes 2*c_outer; inner stage (at 2x) contributes 2*c1 each -> 4*c1 at 4x.
    filter_delay_ = 4 * c1 + 2 * c_outer;
    fifo_start_ = 0;
    stage1_delay_model_ = 0.5 * static_cast<double>(c1) + 0.25 * static_cast<double>(c_outer);
    taps_stage1_ = stage1_->taps_per_phase() + hb_down2_->half_taps();
    taps_stage2_ = stage2_->taps_per_phase() + hb_up2_->half_taps();
    stage2_max = 2 * mid_out_max;
  }
  else
  {
    mode_ = Mode::Polyphase;
    const double grid = static_cast<double>(up_) * static_cast<double>(host_rate); // rate at which both filters are designed
    const double f_stop = 0.5 * std::min(host_rate, model_rate);
    const double f_pass = preset.pass_fraction * f_stop;
    const int n1 = kaiser_length(preset.attenuation_db, (f_stop - f_pass) / grid);
    const int c1 = (n1 - 1) / 2;
    stage1_delay_model_ = static_cast<double>(c1) / down_;
    // The delay of stage 1 is c1 / grid seconds and that of stage 2 is c2 / grid seconds; with grid = up_ * host rate, the sum is
    // (c1 + c2) / up_ host samples. Stage 2 gets the centre that makes this a whole number.
    const int c2 = c1 + static_cast<int>((((-2LL * c1) % up_) + up_) % up_);
    const int n2 = 2 * c2 + 1;
    filter_delay_ = (c1 + c2) / up_;
    const double cutoff = 0.5 * (f_pass + f_stop) / grid;

    max_model_block_ = static_cast<int>((static_cast<std::int64_t>(max_block) * up_ + down_ - 1) / down_) + 1;
    stage1_ = std::make_unique<PolyphaseStage>(up_, down_, design_lowpass(n1, cutoff, preset.attenuation_db, up_), max_block);
    stage2_ = std::make_unique<PolyphaseStage>(down_, up_, design_lowpass(n2, cutoff, preset.attenuation_db, down_), max_model_block_);
    taps_stage1_ = stage1_->taps_per_phase();
    taps_stage2_ = stage2_->taps_per_phase();

    std::int64_t worst = 0;
    for (std::int64_t n = 1; n <= down_; ++n)
    {
      const std::int64_t o1 = (static_cast<std::int64_t>(up_) * n - 1) / down_ + 1;
      const std::int64_t o2 = (static_cast<std::int64_t>(down_) * o1 - 1) / up_ + 1;
      worst = std::max(worst, n - o2);
    }
    fifo_start_ = static_cast<int>(worst);
    stage2_max = stage2_->max_output(max_model_block_);
  }

  stage2_out_.assign(static_cast<std::size_t>(stage2_max), 0.0f);
  std::size_t capacity = 1;
  while (capacity < static_cast<std::size_t>(max_block) + static_cast<std::size_t>(fifo_start_) + static_cast<std::size_t>(stage2_max) + 8)
    capacity <<= 1;
  fifo_.assign(capacity, 0.0f);
  fifo_mask_ = capacity - 1;
  reset();
}

void RateConverter::reset()
{
  if (stage1_)
    stage1_->reset();
  if (stage2_)
    stage2_->reset();
  if (hb_down1_)
    hb_down1_->reset();
  if (hb_down2_)
    hb_down2_->reset();
  if (hb_up1_)
    hb_up1_->reset();
  if (hb_up2_)
    hb_up2_->reset();
  std::fill(fifo_.begin(), fifo_.end(), 0.0f);
  fifo_read_ = 0;
  fifo_count_ = static_cast<std::size_t>(fifo_start_);
}

int RateConverter::to_model(const float* host, int n, float* model)
{
  if (n <= 0)
    return 0;
  if (mode_ == Mode::HybridDown4)
  {
    const int mid = hb_down2_->process_interleaved(host, n, hb_mid_.data());
    return mid > 0 ? stage1_->process(hb_mid_.data(), mid, model) : 0;
  }
  return stage1_->process(host, n, model);
}

void RateConverter::to_host(const float* model, int m, float* host, int n)
{
  if (m > 0)
  {
    int produced = 0;
    if (mode_ == Mode::HybridDown4)
    {
      const int mid = stage2_->process(model, m, hb_mid_.data());
      if (mid > 0)
        hb_up2_->process_interleaved(hb_mid_.data(), mid, stage2_out_.data());
      produced = 2 * mid;
    }
    else
      produced = stage2_->process(model, m, stage2_out_.data());
    const std::size_t cap = fifo_mask_ + 1;
    const std::size_t room = cap - fifo_count_;
    const std::size_t count = std::min(static_cast<std::size_t>(produced), room);
    if (count > 0)
    {
      const std::size_t wp = (fifo_read_ + fifo_count_) & fifo_mask_;
      const std::size_t first = std::min(count, cap - wp);
      std::memcpy(fifo_.data() + wp, stage2_out_.data(), first * sizeof(float));
      if (first < count)
        std::memcpy(fifo_.data(), stage2_out_.data() + first, (count - first) * sizeof(float));
      fifo_count_ += count;
    }
  }
  if (n > 0)
  {
    const std::size_t cap = fifo_mask_ + 1;
    const std::size_t take = std::min(static_cast<std::size_t>(n), fifo_count_);
    if (take > 0)
    {
      const std::size_t rp = fifo_read_ & fifo_mask_;
      const std::size_t first = std::min(take, cap - rp);
      std::memcpy(host, fifo_.data() + rp, first * sizeof(float));
      if (first < take)
        std::memcpy(host + first, fifo_.data(), (take - first) * sizeof(float));
      fifo_read_ += take;
      fifo_count_ -= take;
    }
    if (take < static_cast<std::size_t>(n))
      std::memset(host + take, 0, (static_cast<std::size_t>(n) - take) * sizeof(float));
  }
}
} // namespace namturbo
