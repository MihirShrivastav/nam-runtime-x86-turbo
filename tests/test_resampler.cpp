// Tests of the sample-rate conversion (src/resampler.*). Usage: test_resampler
// Exit codes: 0 pass, 1 failure, 77 skipped (the CPU lacks AVX2 and FMA).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "namturbo/cpu.h"
#include "namturbo/error.h"
#include "resampler.h"

using namespace namturbo;

static int g_failures = 0;
#define CHECK(cond)                                                         \
  do                                                                        \
  {                                                                         \
    if (!(cond))                                                            \
    {                                                                       \
      std::printf("  CHECK FAILED %s:%d  %s\n", __FILE__, __LINE__, #cond); \
      ++g_failures;                                                         \
    }                                                                       \
  } while (0)

constexpr double kPi = 3.14159265358979323846;

static double db(double ratio) { return 20.0 * std::log10(ratio); }

// Amplitude of the sine of frequency f (cycles per sample) that best fits x[begin, end), and the RMS of what is left, relative to
// that amplitude's RMS (so a clean sine gives a very small number).
struct Fit
{
  double amplitude, residual_rel;
};
static Fit sine_fit(const std::vector<float>& x, size_t begin, size_t end, double f)
{
  double scc = 0, sss = 0, scs = 0, sxc = 0, sxs = 0, sxx = 0;
  for (size_t i = begin; i < end; ++i)
  {
    const double ph = 2.0 * kPi * f * double(i), c = std::cos(ph), s = std::sin(ph), v = x[i];
    scc += c * c;
    sss += s * s;
    scs += c * s;
    sxc += v * c;
    sxs += v * s;
    sxx += v * v;
  }
  const double det = scc * sss - scs * scs;
  const double a = (sxc * sss - sxs * scs) / det, b = (sxs * scc - sxc * scs) / det;
  const double amp = std::sqrt(a * a + b * b);
  const double fitted = a * sxc + b * sxs; // energy of the fit
  const double resid = std::max(0.0, sxx - fitted);
  const double n = double(end - begin);
  return {amp, std::sqrt(resid / n) / (amp / std::sqrt(2.0))};
}

static std::vector<float> sine(size_t n, double f_hz, double rate, double amp)
{
  std::vector<float> x(n);
  for (size_t i = 0; i < n; ++i)
    x[i] = float(amp * std::sin(2.0 * kPi * f_hz * double(i) / rate));
  return x;
}

// Passes a signal through the converter with an identity engine, in blocks of random size 1 .. max_block.
static std::vector<float> round_trip(RateConverter& rc, const std::vector<float>& in, int max_block, unsigned seed = 1)
{
  std::vector<float> out(in.size()), model(static_cast<size_t>(rc.max_model_block()));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, max_block);
  for (size_t pos = 0; pos < in.size();)
  {
    const int n = int(std::min<size_t>(size_t(dist(rng)), in.size() - pos));
    const int m = rc.to_model(in.data() + pos, n, model.data());
    rc.to_host(model.data(), m, out.data() + pos, n);
    pos += size_t(n);
  }
  return out;
}

// Frequency response (linear amplitude) of taps h, linear phase about its centre, at f cycles per sample.
static double response(const std::vector<double>& h, double f)
{
  const int c = int(h.size() - 1) / 2;
  double re = 0.0;
  for (int i = 0; i < int(h.size()); ++i)
    re += h[size_t(i)] * std::cos(2.0 * kPi * f * double(i - c)); // symmetric taps: the imaginary part is zero
  return std::fabs(re);
}

struct Case
{
  int host, model;
  ResamplerQuality quality;
  const char* name;
};

static const char* quality_name(ResamplerQuality q) { return q == ResamplerQuality::Standard ? "Standard" : "LowLatency"; }

static void filter_design_meets_its_targets()
{
  std::printf("filter_design_meets_its_targets\n");
  struct Spec
  {
    double attenuation, pass;
  };
  for (Spec spec : {Spec{96.0, 0.90}, Spec{72.0, 0.80}})
  {
    const double grid = 160.0 * 44100.0, f_stop = 22050.0, f_pass = spec.pass * f_stop;
    const int n = kaiser_length(spec.attenuation, (f_stop - f_pass) / grid);
    const std::vector<double> h = design_lowpass(n, 0.5 * (f_pass + f_stop) / grid, spec.attenuation, 160.0);
    double sum = std::accumulate(h.begin(), h.end(), 0.0);
    CHECK(std::fabs(sum - 160.0) < 1e-9);
    bool symmetric = true;
    for (size_t i = 0; i < h.size(); ++i)
      symmetric = symmetric && h[i] == h[h.size() - 1 - i];
    CHECK(symmetric && n % 2 == 1);
    double worst_ripple = 0.0, worst_stop = 0.0;
    for (int i = 0; i <= 300; ++i) // pass band
      worst_ripple = std::max(worst_ripple, std::fabs(db(response(h, f_pass * i / 300.0 / grid) / 160.0)));
    for (int i = 0; i <= 600; ++i) // stop band: from the stop edge up to 3 times it (the rest is far from the transition)
      worst_stop = std::max(worst_stop, response(h, (f_stop + 2.0 * f_stop * i / 600.0) / grid) / 160.0);
    std::printf("  design %.0f dB, pass %.2f: %d taps, pass-band ripple %.5f dB, stop-band %.1f dB\n", spec.attenuation, spec.pass, n, worst_ripple, db(worst_stop));
    CHECK(worst_ripple < 0.01);
    CHECK(db(worst_stop) < -(spec.attenuation - 3.0));
  }
}

static void polyphase_equals_direct_convolution()
{
  std::printf("polyphase_equals_direct_convolution\n");
  struct Ratio
  {
    int up, down;
  };
  for (Ratio r : {Ratio{160, 147}, Ratio{147, 160}, Ratio{1, 2}, Ratio{2, 1}, Ratio{3, 2}, Ratio{2, 3}, Ratio{80, 147}, Ratio{1, 4}, Ratio{6, 1}})
  {
    const double grid = double(r.up) * 1000.0;
    const int n_taps = kaiser_length(60.0, 0.1 * 0.5 / double(std::max(r.up, r.down)) * 2.0) | 1; // any odd length will do
    (void)grid;
    const std::vector<double> taps = design_lowpass(n_taps, 0.45 / double(std::max(r.up, r.down)), 60.0, r.up);
    PolyphaseStage stage(r.up, r.down, taps, 700);

    std::mt19937 rng(5);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    std::vector<float> in(4000);
    for (float& v : in)
      v = d(rng);
    // the stage, fed in random pieces
    std::vector<float> out(static_cast<size_t>(stage.max_output(int(in.size()))) + 16);
    size_t produced = 0;
    std::uniform_int_distribution<int> piece(0, 900);
    for (size_t pos = 0; pos < in.size();)
    {
      const int k = int(std::min<size_t>(size_t(piece(rng)), in.size() - pos));
      produced += size_t(stage.process(in.data() + pos, k, out.data() + produced));
      pos += size_t(k);
    }
    // the reference: y[m] = sum_i x[i] h[m down - i up], in double precision
    const size_t expected = size_t((std::int64_t(r.up) * std::int64_t(in.size()) - 1) / r.down) + 1;
    CHECK(produced == expected);
    double err = 0.0, pow = 0.0;
    for (size_t m = 0; m < std::min(produced, expected); ++m)
    {
      const std::int64_t pos = std::int64_t(m) * r.down;
      double y = 0.0;
      const std::int64_t i_hi = pos / r.up;
      for (std::int64_t i = i_hi; i >= 0 && pos - i * r.up < std::int64_t(taps.size()); --i)
        y += double(in[size_t(i)]) * taps[size_t(pos - i * r.up)];
      err += (double(out[m]) - y) * (double(out[m]) - y);
      pow += y * y;
    }
    const double esr = 10.0 * std::log10(err / pow);
    std::printf("  up %3d down %3d: %zu outputs, error against the direct double-precision convolution %.1f dB\n", r.up, r.down, produced, esr);
    CHECK(esr < -120.0);
  }
}

static void round_trip_latency_is_exact_and_flat()
{
  std::printf("round_trip_latency_is_exact_and_flat\n");
  const int model_rate = 48000;
  for (ResamplerQuality q : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
    for (int host : {44100, 88200, 96000, 176400, 192000, 32000, 22050, 8000, 47952, 48048, 352800, 384000})
    {
      RateConverter rc(host, model_rate, 512, q);
      const double lower_nyquist = 0.5 * std::min(host, model_rate);
      const double pass = (q == ResamplerQuality::Standard ? 0.90 : 0.80) * lower_nyquist;
      // band-limited noise: 200 sines with random phases between 50 Hz and 85% of the pass band
      std::mt19937 rng(9);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      std::vector<float> in(60000, 0.0f);
      std::vector<double> acc(in.size(), 0.0);
      for (int k = 0; k < 200; ++k)
      {
        const double f = 50.0 + (0.85 * pass - 50.0) * u(rng), ph = 2.0 * kPi * u(rng), a = 0.002;
        for (size_t i = 0; i < in.size(); ++i)
          acc[i] += a * std::sin(2.0 * kPi * f * double(i) / host + ph);
      }
      for (size_t i = 0; i < in.size(); ++i)
        in[i] = float(acc[i]);
      const std::vector<float> out = round_trip(rc, in, 512);
      const int D = rc.latency_samples();
      // out[k + D] should equal in[k]; compare after the start-up of the filters and before the end
      const size_t from = size_t(D) * 2 + 2000, to = in.size() - size_t(D) - 100;
      double err = 0.0, pow = 0.0;
      for (size_t k = from; k < to; ++k)
      {
        const double e = double(out[k + size_t(D)]) - double(in[k]);
        err += e * e;
        pow += double(in[k]) * double(in[k]);
      }
      const double esr = 10.0 * std::log10(err / pow);
      std::printf("  %-10s host %6d Hz: latency %4d samples (%.2f ms), up %4d down %4d, %3d + %3d taps per phase, round-trip error %.1f dB\n", quality_name(q), host,
                  D, 1000.0 * D / host, rc.up(), rc.down(), rc.taps_per_phase_stage1(), rc.taps_per_phase_stage2(), esr);
      CHECK(esr < (q == ResamplerQuality::Standard ? -90.0 : -65.0));
    }
}

static void tones_pass_and_alias_images_are_rejected()
{
  std::printf("tones_pass_and_alias_images_are_rejected\n");
  for (ResamplerQuality q : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
  {
    const double attenuation = q == ResamplerQuality::Standard ? 96.0 : 72.0;
    const double pass_fraction = q == ResamplerQuality::Standard ? 0.90 : 0.80;
    for (int host : {44100, 96000, 192000, 32000})
    {
      RateConverter rc(host, 48000, 512, q);
      const double lower_nyquist = 0.5 * std::min(host, 48000);

      // 1. in the pass band, the whole round trip changes the level by less than 0.01 dB and adds nothing but a trace
      double worst_gain = 0.0, worst_spur = -300.0;
      for (double fraction : {0.05, 0.2, 0.5, 0.75, pass_fraction - 0.02})
      {
        rc.reset();
        const double f = fraction * lower_nyquist;
        const std::vector<float> in = sine(80000, f, host, 0.5);
        const std::vector<float> out = round_trip(rc, in, 512);
        const size_t from = size_t(rc.latency_samples()) * 2 + 4000;
        const Fit fit = sine_fit(out, from, out.size(), f / host);
        worst_gain = std::max(worst_gain, std::fabs(db(fit.amplitude / 0.5)));
        worst_spur = std::max(worst_spur, db(fit.residual_rel));
      }
      std::printf("  %-10s host %6d Hz: pass band: level error at most %.4f dB, spurious at most %.1f dB\n", quality_name(q), host, worst_gain, worst_spur);
      CHECK(worst_gain < 0.01);
      CHECK(worst_spur < -(attenuation - 6.0));

      // 2. above the lower Nyquist frequency, a tone must not get through (when downsampling, it would fold into the band)
      if (host > 48000)
      {
        double worst = -300.0;
        for (double fraction : {1.03, 1.1, 1.3, 1.6})
        {
          const double f = fraction * lower_nyquist;
          if (f >= 0.5 * host)
            continue;
          rc.reset();
          const std::vector<float> in = sine(80000, f, host, 0.5);
          const std::vector<float> out = round_trip(rc, in, 512);
          double e = 0.0;
          for (size_t i = size_t(rc.latency_samples()) * 2 + 4000; i < out.size(); ++i)
            e += double(out[i]) * double(out[i]);
          const double rms = std::sqrt(e / double(out.size() - size_t(rc.latency_samples()) * 2 - 4000));
          worst = std::max(worst, db(rms / (0.5 / std::sqrt(2.0))));
        }
        std::printf("  %-10s host %6d Hz: tones above the model Nyquist frequency come out at most %.1f dB\n", quality_name(q), host, worst);
        CHECK(worst < -(attenuation - 6.0));
      }

      // 3. when upsampling, the images that the zero-stuffing makes must be removed by stage 1 (seen at the model rate)
      if (host < 48000)
      {
        double worst = -300.0;
        for (double fraction : {0.1, 0.4, 0.8})
        {
          rc.reset();
          const double f = fraction * lower_nyquist;
          const std::vector<float> in = sine(60000, f, host, 0.5);
          std::vector<float> model(static_cast<size_t>(rc.stage1().max_output(int(in.size()))) + 8);
          const int m = rc.stage1().process(in.data(), int(in.size()), model.data());
          model.resize(size_t(m));
          const Fit fit = sine_fit(model, 6000, model.size(), f / 48000.0);
          worst = std::max(worst, db(fit.residual_rel));
        }
        std::printf("  %-10s host %6d Hz: images at the model rate at most %.1f dB\n", quality_name(q), host, worst);
        CHECK(worst < -(attenuation - 6.0));
      }
    }
  }
}

static void halfband_design_and_accuracy()
{
  std::printf("halfband_design_and_accuracy\n");
  struct Spec
  {
    double attenuation, pass;
  };
  for (Spec spec : {Spec{96.0, 0.90}, Spec{72.0, 0.80}, Spec{96.0, 0.50}})
  {
    const std::vector<double> h = design_halfband(spec.attenuation, spec.pass);
    const int n = int(h.size());
    const int c = (n - 1) / 2;
    CHECK((n + 1) % 4 == 0 && c % 2 == 1);
    CHECK(std::fabs(h[size_t(c)] - 0.5) < 1e-12);
    double sum = 0.0;
    bool symmetric = true, even_zero = true;
    for (int i = 0; i < n; ++i)
    {
      sum += h[size_t(i)];
      symmetric = symmetric && (h[size_t(i)] == h[size_t(n - 1 - i)]);
      if (i != c && ((i - c) & 1) == 0)
        even_zero = even_zero && (h[size_t(i)] == 0.0);
    }
    CHECK(std::fabs(sum - 1.0) < 1e-12 && symmetric && even_zero);

    const double f_pass = 0.25 * spec.pass;
    const double f_stop = 0.5 - f_pass;
    double worst_ripple = 0.0, worst_stop = 0.0;
    for (int i = 0; i <= 200; ++i)
      worst_ripple = std::max(worst_ripple, std::fabs(db(response(h, f_pass * i / 200.0))));
    for (int i = 0; i <= 200; ++i)
      worst_stop = std::max(worst_stop, response(h, f_stop + (0.5 - f_stop) * i / 200.0));
    std::printf("  halfband %.0f dB, pass %.2f: %d taps (%d pairs), ripple %.5f dB, stop %.1f dB\n", spec.attenuation, spec.pass, n, (n + 1) / 4,
                worst_ripple, db(worst_stop));
    CHECK(worst_ripple < 0.01);
    CHECK(db(worst_stop) < -(spec.attenuation - 3.0));

    // Verify HalfBandUp2x and HalfBandDown2x against direct double-precision convolution with random block splits.
    std::mt19937 rng(11);
    std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    std::vector<float> in(2048);
    for (float& v : in)
      v = dist(rng);

    HalfBandUp2x up(h, 256);
    std::vector<float> up_out(in.size() * 2);
    std::uniform_int_distribution<int> bdist(1, 300);
    for (size_t pos = 0; pos < in.size();)
    {
      const int k = int(std::min<size_t>(size_t(bdist(rng)), in.size() - pos));
      up.process_interleaved(in.data() + pos, k, up_out.data() + 2 * pos);
      pos += size_t(k);
    }
    double err_u = 0.0, pow_u = 0.0;
    for (size_t m = 0; m < up_out.size(); ++m)
    {
      double y = 0.0;
      for (int j = 0; j < n && int(m) - j >= 0; ++j)
      {
        const int idx = int(m) - j;
        if ((idx & 1) == 0)
          y += 2.0 * h[size_t(j)] * double(in[size_t(idx / 2)]);
      }
      err_u += (double(up_out[m]) - y) * (double(up_out[m]) - y);
      pow_u += y * y;
    }
    CHECK(10.0 * std::log10(err_u / pow_u) < -120.0);

    HalfBandDown2x dn(h, 256);
    std::vector<float> dn_out(up_out.size());
    size_t produced = 0;
    for (size_t pos = 0; pos < up_out.size();)
    {
      const int k = int(std::min<size_t>(size_t(bdist(rng)), up_out.size() - pos));
      produced += size_t(dn.process_interleaved(up_out.data() + pos, k, dn_out.data() + produced));
      pos += size_t(k);
    }
    CHECK(produced == in.size());
    double err_d = 0.0, pow_d = 0.0;
    for (size_t m = 0; m < produced; ++m)
    {
      double y = 0.0;
      for (int j = 0; j < n && int(2 * m) - j >= 0; ++j)
        y += h[size_t(j)] * double(up_out[size_t(2 * m - size_t(j))]);
      err_d += (double(dn_out[m]) - y) * (double(dn_out[m]) - y);
      pow_d += y * y;
    }
    CHECK(10.0 * std::log10(err_d / pow_d) < -120.0);
  }
}

static void oversampler_round_trip_and_block_independence()
{
  std::printf("oversampler_round_trip_and_block_independence\n");
  for (ResamplerQuality q : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
    for (int factor : {2, 4})
    {
      const double pass = (q == ResamplerQuality::Standard ? 0.90 : 0.80) * 24000.0;
      std::mt19937 rng(19);
      std::uniform_real_distribution<double> u(0.0, 1.0);
      std::vector<float> in(32000, 0.0f);
      std::vector<double> acc(in.size(), 0.0);
      for (int k = 0; k < 120; ++k)
      {
        const double f = 50.0 + (0.85 * pass - 50.0) * u(rng), ph = 2.0 * kPi * u(rng);
        for (size_t i = 0; i < in.size(); ++i)
          acc[i] += 0.002 * std::sin(2.0 * kPi * f * double(i) / 48000.0 + ph);
      }
      for (size_t i = 0; i < in.size(); ++i)
        in[i] = float(acc[i]);

      auto run = [&](int max_block, bool random_blocks) {
        Oversampler os(factor, max_block, q);
        std::vector<float> out(in.size());
        std::vector<float> ph[4];
        float* ptrs[4] = {};
        const float* cptrs[4] = {};
        for (int p = 0; p < factor; ++p)
        {
          ph[p].resize(size_t(max_block));
          ptrs[p] = ph[p].data();
          cptrs[p] = ph[p].data();
        }
        std::mt19937 brng(31);
        std::uniform_int_distribution<int> bdist(1, max_block);
        for (size_t pos = 0; pos < in.size();)
        {
          const int k = int(std::min<size_t>(size_t(random_blocks ? bdist(brng) : max_block), in.size() - pos));
          os.upsample(in.data() + pos, k, ptrs);
          os.downsample(cptrs, k, out.data() + pos);
          pos += size_t(k);
        }
        return std::make_pair(out, os.latency_samples());
      };

      const auto [out_whole, D] = run(512, false);
      const auto [out_rand, D2] = run(256, true);
      const auto [out_one, D3] = run(1, false);
      CHECK(D == D2 && D == D3);
      CHECK(out_whole == out_rand && out_whole == out_one);

      const size_t from = size_t(D) * 2 + 1000, to = in.size() - size_t(D) - 100;
      double err = 0.0, pow = 0.0;
      for (size_t k = from; k < to; ++k)
      {
        const double e = double(out_whole[k + size_t(D)]) - double(in[k]);
        err += e * e;
        pow += double(in[k]) * double(in[k]);
      }
      const double esr = 10.0 * std::log10(err / pow);
      std::printf("  %-10s %dx: latency %3d samples (%.2f ms), round-trip error %.1f dB, block-split bit-identical\n", quality_name(q), factor, D,
                  1000.0 * D / 48000.0, esr);
      CHECK(esr < (q == ResamplerQuality::Standard ? -90.0 : -65.0));
    }
}

static void unsupported_ratios_and_bad_arguments()
{
  std::printf("unsupported_ratios_and_bad_arguments\n");
  auto code = [](auto f) -> int {
    try
    {
      f();
    }
    catch (const Error& e)
    {
      return int(e.code());
    }
    return -1;
  };
  // 44056 Hz against 48000 Hz reduces to 6000 : 5507, which is beyond the limit
  CHECK(code([] { RateConverter rc(44056, 48000, 64, ResamplerQuality::Standard); }) == int(ErrorCode::Unsupported));
  CHECK(code([] { RateConverter rc(0, 48000, 64, ResamplerQuality::Standard); }) == int(ErrorCode::InvalidArgument));
  CHECK(code([] { RateConverter rc(44100, 48000, 0, ResamplerQuality::Standard); }) == int(ErrorCode::InvalidArgument));
  CHECK(code([] { RateConverter rc(47952, 48000, 64, ResamplerQuality::Standard); }) == -1); // 1000 : 999 is within the limit
}

static void guarded(const char* name, void (*fn)())
{
  try
  {
    fn();
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION in %s: %s\n", name, e.what());
    ++g_failures;
  }
}

int main()
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (!check_cpu().supported)
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2 and FMA\n");
    return 77;
  }
  guarded("filter_design_meets_its_targets", filter_design_meets_its_targets);
  guarded("halfband_design_and_accuracy", halfband_design_and_accuracy);
  guarded("oversampler_round_trip_and_block_independence", oversampler_round_trip_and_block_independence);
  guarded("polyphase_equals_direct_convolution", polyphase_equals_direct_convolution);
  guarded("round_trip_latency_is_exact_and_flat", round_trip_latency_is_exact_and_flat);
  guarded("tones_pass_and_alias_images_are_rejected", tones_pass_and_alias_images_are_rejected);
  guarded("unsupported_ratios_and_bad_arguments", unsupported_ratios_and_bad_arguments);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
