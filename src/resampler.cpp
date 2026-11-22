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

// Sum of x[i] * c[i] for i < n (n a multiple of 32; c aligned to 32 bytes). Four accumulators of 8 lanes: the order of the
// additions is fixed, so the result for a given input does not depend on anything else.
inline float dot_product(const float* x, const float* c, int n)
{
  __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
  for (int t = 0; t < n; t += 32)
  {
    a0 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t), _mm256_load_ps(c + t), a0);
    a1 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 8), _mm256_load_ps(c + t + 8), a1);
    a2 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 16), _mm256_load_ps(c + t + 16), a2);
    a3 = _mm256_fmadd_ps(_mm256_loadu_ps(x + t + 24), _mm256_load_ps(c + t + 24), a3);
  }
  const __m256 s = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
  __m128 q = _mm_add_ps(_mm256_castps256_ps128(s), _mm256_extractf128_ps(s, 1));
  q = _mm_add_ps(q, _mm_movehl_ps(q, q));
  q = _mm_add_ss(q, _mm_shuffle_ps(q, q, 0x55));
  return _mm_cvtss_f32(q);
}

struct Preset
{
  double attenuation_db;
  double pass_fraction; // of the lower Nyquist frequency
};

Preset preset_of(ResamplerQuality quality)
{
  return quality == ResamplerQuality::LowLatency ? Preset{72.0, 0.80} : Preset{96.0, 0.90};
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

// ---- one stage --------------------------------------------------------------------------------------------------------

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
  taps_per_phase_ = round_up((n + up - 1) / up, 32);
  coefficients_.assign_zero(static_cast<std::size_t>(up) * static_cast<std::size_t>(taps_per_phase_));
  float* c = coefficients_.data();
  // Row `phase` holds the taps phase, phase + up, phase + 2 up, ... in the order of the window (oldest input sample first).
  for (int phase = 0; phase < up; ++phase)
    for (int t = 0; t < taps_per_phase_; ++t)
    {
      const std::int64_t index = phase + static_cast<std::int64_t>(taps_per_phase_ - 1 - t) * up;
      c[static_cast<std::size_t>(phase) * taps_per_phase_ + t] = index < n ? static_cast<float>(taps[static_cast<std::size_t>(index)]) : 0.0f;
    }
  window_.assign(static_cast<std::size_t>(taps_per_phase_ - 1 + max_input_block), 0.0f);
}

void PolyphaseStage::reset()
{
  std::fill(window_.begin(), window_.end(), 0.0f);
  next_input_ = 0;
  phase_ = 0;
}

int PolyphaseStage::process_piece(const float* in, int n, float* out)
{
  const int history = taps_per_phase_ - 1;
  float* window = window_.data();
  std::memcpy(window + history, in, static_cast<std::size_t>(n) * sizeof(float));
  const float* coefficients = coefficients_.data();
  int produced = 0;
  // The output whose newest input sample has index next_input_ (counted from the start of this block) can be computed once that
  // sample has arrived.
  while (next_input_ < n)
  {
    out[produced++] = dot_product(window + next_input_, coefficients + static_cast<std::size_t>(phase_) * taps_per_phase_, taps_per_phase_);
    // advance by down_ / up_ input samples without dividing: the whole part, then the fraction in units of 1 / up_
    next_input_ += step_whole_;
    phase_ += step_fraction_;
    if (phase_ >= up_)
    {
      phase_ -= up_;
      ++next_input_;
    }
  }
  next_input_ -= n;
  std::memmove(window, window + n, static_cast<std::size_t>(history) * sizeof(float));
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

// ---- the pair of stages -----------------------------------------------------------------------------------------------

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
  const double grid = static_cast<double>(up_) * static_cast<double>(host_rate); // rate at which both filters are designed
  const double f_stop = 0.5 * std::min(host_rate, model_rate);
  const double f_pass = preset.pass_fraction * f_stop;
  const int n1 = kaiser_length(preset.attenuation_db, (f_stop - f_pass) / grid);
  const int c1 = (n1 - 1) / 2;
  center1_ = c1;
  // The delay of stage 1 is c1 / grid seconds and that of stage 2 is c2 / grid seconds; with grid = up_ * host rate, the sum is
  // (c1 + c2) / up_ host samples. Stage 2 gets the centre that makes this a whole number.
  const int c2 = c1 + static_cast<int>((((-2LL * c1) % up_) + up_) % up_);
  const int n2 = 2 * c2 + 1;
  filter_delay_ = (c1 + c2) / up_;
  const double cutoff = 0.5 * (f_pass + f_stop) / grid;

  max_model_block_ = static_cast<int>((static_cast<std::int64_t>(max_block) * up_ + down_ - 1) / down_) + 1;
  stage1_ = std::make_unique<PolyphaseStage>(up_, down_, design_lowpass(n1, cutoff, preset.attenuation_db, up_), max_block);
  stage2_ = std::make_unique<PolyphaseStage>(down_, up_, design_lowpass(n2, cutoff, preset.attenuation_db, down_), max_model_block_);

  // Number of output samples after N input samples: stage 1 has produced floor((up N - 1) / down) + 1, stage 2 then
  // floor((down o1 - 1) / up) + 1. The function repeats with period `down` (adding `down` to N adds `down` to the result), so
  // the largest shortfall N - out(N) is found by trying N = 1 .. down. The FIFO starts with that many zeros, so it never runs dry.
  std::int64_t worst = 0;
  for (std::int64_t n = 1; n <= down_; ++n)
  {
    const std::int64_t o1 = (static_cast<std::int64_t>(up_) * n - 1) / down_ + 1;
    const std::int64_t o2 = (static_cast<std::int64_t>(down_) * o1 - 1) / up_ + 1;
    worst = std::max(worst, n - o2);
  }
  fifo_start_ = static_cast<int>(worst);

  const int stage2_max = stage2_->max_output(max_model_block_);
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
  stage1_->reset();
  stage2_->reset();
  std::fill(fifo_.begin(), fifo_.end(), 0.0f);
  fifo_read_ = 0;
  fifo_count_ = static_cast<std::size_t>(fifo_start_);
}

int RateConverter::to_model(const float* host, int n, float* model)
{
  if (n <= 0)
    return 0;
  return stage1_->process(host, n, model);
}

void RateConverter::to_host(const float* model, int m, float* host, int n)
{
  if (m > 0)
  {
    const int produced = stage2_->process(model, m, stage2_out_.data());
    std::size_t write = fifo_read_ + fifo_count_;
    for (int i = 0; i < produced && fifo_count_ <= fifo_mask_; ++i, ++write, ++fifo_count_)
      fifo_[write & fifo_mask_] = stage2_out_[static_cast<std::size_t>(i)];
  }
  for (int i = 0; i < n; ++i)
  {
    if (fifo_count_ > 0)
    {
      host[i] = fifo_[fifo_read_ & fifo_mask_];
      ++fifo_read_;
      --fifo_count_;
    }
    else
      host[i] = 0.0f; // not reachable: the FIFO start is chosen so that it never runs dry
  }
}
} // namespace namturbo
