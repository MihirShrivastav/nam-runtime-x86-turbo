// Tests of 2x and 4x phase-interleaved oversampling in Processor.
// Usage: test_oversampling <slimmable A2 model.nam> <data_dir>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing, or the CPU lacks AVX2 and FMA).
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "namturbo/namturbo.h"

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

static bool bit_equal(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static std::vector<float> test_signal(size_t n, double rate)
{
  std::vector<float> x(n);
  std::mt19937 rng(42);
  for (size_t i = 0; i < n; ++i)
  {
    const double t = double(i) / rate;
    const double noise = (double(rng() >> 8) / 16777216.0 - 0.5) * 0.04;
    x[i] = float(0.12 * std::sin(2.0 * kPi * 440.0 * t) + 0.08 * std::sin(2.0 * kPi * 1750.0 * t) + noise);
  }
  return x;
}

static void latency_and_block_independence(const std::shared_ptr<const Model>& model)
{
  std::printf("latency_and_block_independence\n");
  for (Oversampling os : {Oversampling::X2, Oversampling::X4})
    for (ResamplerQuality q : {ResamplerQuality::Standard, ResamplerQuality::LowLatency})
      for (double rate : {48000.0, 44100.0, 96000.0})
      {
        const std::vector<float> in = test_signal(12000, rate);
        auto make = [&](Processor& p, int max_block) {
          ProcessorConfig c;
          c.sample_rate = rate;
          c.max_block = max_block;
          c.resampler = q;
          c.oversampling = os;
          c.variant = Variant::Lite; // fast to run all combinations
          p.prepare(c);
        };
        Processor whole(model), split(model), inplace(model), dbl(model);
        make(whole, 256);
        make(split, 256);
        make(inplace, 256);
        make(dbl, 256);

        CHECK(whole.latency_samples() > 0);
        if (rate == 48000.0 && q == ResamplerQuality::Standard)
          CHECK(whole.latency_samples() == (os == Oversampling::X2 ? 63 : 71));
        if (rate == 48000.0 && q == ResamplerQuality::LowLatency)
          CHECK(whole.latency_samples() == (os == Oversampling::X2 ? 23 : 29));

        std::vector<float> a(in.size()), b(in.size()), c = in;
        whole.process(in.data(), a.data(), int(in.size()));

        std::mt19937 rng(7);
        std::uniform_int_distribution<int> dist(1, 500);
        for (size_t pos = 0; pos < in.size();)
        {
          const int k = int(std::min<size_t>(size_t(dist(rng)), in.size() - pos));
          split.process(in.data() + pos, b.data() + pos, k);
          pos += size_t(k);
        }
        inplace.process(c.data(), c.data(), int(c.size()));

        std::vector<double> din(in.begin(), in.end()), dout(in.size());
        dbl.process(din.data(), dout.data(), int(din.size()));
        bool dsame = true;
        for (size_t i = 0; i < in.size(); ++i)
          dsame = dsame && (dout[i] == double(a[i]));

        CHECK(bit_equal(a, b) && bit_equal(a, c) && dsame);

        if (rate == 48000.0 && q == ResamplerQuality::Standard)
        {
          Processor single(model);
          make(single, 64);
          std::vector<float> s(in.size());
          for (size_t i = 0; i < in.size(); ++i)
            single.process(in.data() + i, s.data() + i, 1);
          CHECK(bit_equal(a, s));
        }

        // reset() restores start state
        whole.reset();
        std::vector<float> after_reset(in.size());
        whole.process(in.data(), after_reset.data(), int(in.size()));
        CHECK(bit_equal(a, after_reset));

        std::printf("  %dx %-10s %6.0f Hz: latency %3d samples (%.2f ms), bit-identical\n", int(os),
                    q == ResamplerQuality::Standard ? "Standard" : "LowLatency", rate, whole.latency_samples(),
                    1000.0 * whole.latency_samples() / rate);
      }
}

// Coherent DFT at a single integer bin k over N samples.
static double bin_power(const float* x, size_t n, size_t k)
{
  double re = 0.0, im = 0.0;
  const double w = 2.0 * kPi * double(k) / double(n);
  for (size_t i = 0; i < n; ++i)
  {
    const double ph = w * double(i);
    re += double(x[i]) * std::cos(ph);
    im -= double(x[i]) * std::sin(ph);
  }
  const double scale = 2.0 / double(n);
  return (re * re + im * im) * scale * scale;
}

static void aliasing_suppression_and_tone_preservation(const std::shared_ptr<const Model>& model)
{
  std::printf("aliasing_suppression_and_tone_preservation\n");
  const size_t N = 8192;
  const size_t warmup = 16384;
  const Oversampling modes[3] = {Oversampling::Off, Oversampling::X2, Oversampling::X4};

  // Case 1: 5 kHz sine (bin 853, 4998.05 Hz): harmonics 5..9 are 25..45 kHz -> 2x suppresses them heavily.
  // Case 2: 8 kHz sine (bin 1367, 8009.77 Hz): harmonics 7..11 are 56..88 kHz -> above 2x Nyquist, so 4x beats 2x.
  struct ToneCase
  {
    size_t bin;
    size_t h_lo, h_hi;
    bool check_4x_over_2x;
  };
  for (ToneCase tc : {ToneCase{853, 5, 12, false}, ToneCase{1367, 4, 15, true}})
  {
    const double f0 = double(tc.bin) * 48000.0 / double(N);
    std::vector<float> in(warmup + N + 256);
    for (size_t i = 0; i < in.size(); ++i)
      in[i] = float(0.25 * std::sin(2.0 * kPi * f0 * double(i) / 48000.0));

    double fund_db[3] = {}, alias_db[3] = {};
    for (int m = 0; m < 3; ++m)
    {
      Processor p(model);
      ProcessorConfig c;
      c.sample_rate = 48000.0;
      c.max_block = 256;
      c.variant = Variant::Full;
      c.oversampling = modes[m];
      p.prepare(c);
      std::vector<float> out(in.size());
      p.process(in.data(), out.data(), int(in.size()));

      const float* seg = out.data() + warmup;
      const double p_fund = bin_power(seg, N, tc.bin);
      double p_alias = 0.0;
      for (size_t h = tc.h_lo; h <= tc.h_hi; ++h)
      {
        const size_t raw = (h * tc.bin) % N;
        const size_t folded = raw <= N / 2 ? raw : N - raw;
        if (folded != tc.bin && folded != 2 * tc.bin && folded != 3 * tc.bin && folded != 4 * tc.bin)
          p_alias += bin_power(seg, N, folded);
      }
      fund_db[m] = 10.0 * std::log10(std::max(p_fund, 1e-30));
      alias_db[m] = 10.0 * std::log10(std::max(p_alias, 1e-30) / std::max(p_fund, 1e-30));
      std::printf("  %6.1f Hz %dx: fundamental %.3f dB, aliased harmonics %.2f dB rel fund\n", f0, int(modes[m]), fund_db[m], alias_db[m]);
    }
    CHECK(std::fabs(fund_db[1] - fund_db[0]) < 0.01);
    CHECK(std::fabs(fund_db[2] - fund_db[0]) < 0.01);
    if (!tc.check_4x_over_2x)
      CHECK(alias_db[1] < alias_db[0] - 20.0); // 5 kHz: 2x removes harmonics 5..9 heavily
    else
    {
      CHECK(alias_db[1] < alias_db[0] - 3.0);  // 8 kHz: 2x removes harmonics 4..5
      CHECK(alias_db[2] < alias_db[1] - 20.0); // 8 kHz: 4x removes harmonics 6..11 above 2x Nyquist
    }
  }
}

static void swap_under_oversampling(const std::shared_ptr<const Model>& model)
{
  std::printf("swap_under_oversampling\n");
  for (Oversampling os : {Oversampling::X2, Oversampling::X4})
  {
    const std::vector<float> in = test_signal(24000, 48000.0);
    Processor ref(model), sw(model);
    ProcessorConfig c;
    c.sample_rate = 48000.0;
    c.max_block = 128;
    c.variant = Variant::Lite;
    c.oversampling = os;
    c.swap_fade_ms = 10.0;
    ref.prepare(c);
    sw.prepare(c);

    std::vector<float> out_ref(in.size()), out_sw(in.size());
    const size_t swap_at = 10240;
    for (size_t pos = 0; pos < in.size(); pos += 128)
    {
      const int k = int(std::min<size_t>(128, in.size() - pos));
      if (pos == swap_at)
      {
        SwapTicket t = sw.prepare_swap(model, Variant::Lite);
        CHECK(sw.commit_swap(std::move(t)));
      }
      ref.process(in.data() + pos, out_ref.data() + pos, k);
      sw.process(in.data() + pos, out_sw.data() + pos, k);
      sw.service();
    }
    CHECK(sw.swaps_completed() == 1 && sw.swaps_dropped() == 0);
    // After fade (480 samples) + oversampler filter length (~200 samples), same-model swap must be bit-identical
    const size_t settle = swap_at + 1024;
    bool identical = true;
    for (size_t i = settle; i < in.size(); ++i)
      identical = identical && (out_ref[i] == out_sw[i]);
    std::printf("  %dx same-model swap: bit-identical after fade: %s\n", int(os), identical ? "yes" : "NO");
    CHECK(identical);
  }
}

static void invalid_oversampling_rejected(const std::shared_ptr<const Model>& model)
{
  std::printf("invalid_oversampling_rejected\n");
  Processor p(model);
  ProcessorConfig c;
  c.oversampling = static_cast<Oversampling>(3);
  bool caught = false;
  try
  {
    p.prepare(c);
  }
  catch (const Error& e)
  {
    caught = (e.code() == ErrorCode::InvalidArgument);
  }
  CHECK(caught);
}

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2 || !std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found\n");
    return 77;
  }
  if (!check_cpu().supported)
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2 and FMA\n");
    return 77;
  }
  try
  {
    auto model = Model::load(argv[1]);
    latency_and_block_independence(model);
    aliasing_suppression_and_tone_preservation(model);
    swap_under_oversampling(model);
    invalid_oversampling_rejected(model);
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION: %s\n", e.what());
    ++g_failures;
  }
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
