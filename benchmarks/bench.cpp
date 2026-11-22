// bench: time an engine per audio block, optionally beside NAM Core's own fast and generic paths.
//
//   bench --model <nam> --slim <0..1> --backend scalar [--buffer 64] [--seconds 2] [--iters 10] [--nam]
//   bench ... --backend avx2:0 --ab avx2:1     drift-proof A/B of two engines (see run_ab)
//   bench ... --core N                         pin the thread to logical core N
//   bench ... --host-rate 44100 [--low-latency]  time the Processor with sample-rate conversion (the model is at 48 kHz)
//   bench ... --monitor                        time the Processor with its own performance monitoring switched on
//
// Input is two sines (as NAM Core's bench_a2_fast uses). Each block is timed separately. Our engine's
// timing includes the double -> float -> double conversion NAM Core's interface requires. With --nam,
// NAM Core's A2 fast path and generic path are timed in the same process on the same input.
// Exit code 77 if the model file is missing.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "a2_dispatch.h"
#include "a2_weights.h"
#include "cpu_features.h"
#include "engines.h"
#include "namturbo/namturbo.h"
#ifdef NAMTURBO_WITH_NAMCORE
  #include "nam_core.h"
#endif
#include "pin.h"

using namespace namturbo;
using Clock = std::chrono::steady_clock;

constexpr double kPi = 3.14159265358979323846;

namespace
{
struct Stats { double min, p50, p99, p999, max, mean; };

double percentile(const std::vector<double>& sorted, double q)
{
  const double idx = q * (sorted.size() - 1);
  const size_t i = static_cast<size_t>(idx);
  if (i + 1 >= sorted.size())
    return sorted[i];
  const double frac = idx - double(i);
  return sorted[i] * (1.0 - frac) + sorted[i + 1] * frac;
}

Stats compute(std::vector<double> xs)
{
  std::sort(xs.begin(), xs.end());
  double sum = 0;
  for (double x : xs)
    sum += x;
  return {xs.front(), percentile(xs, 0.5), percentile(xs, 0.99), percentile(xs, 0.999), xs.back(), sum / double(xs.size())};
}

// Time `block_fn(in, out, n)` for every block of every iteration; one untimed warm-up pass first.
template <class Fn>
std::vector<double> time_blocks(Fn&& block_fn, const std::vector<double>& in, std::vector<double>& out, int buffer, int iters)
{
  auto pass = [&](std::vector<double>* times) {
    for (size_t pos = 0; pos < in.size(); pos += buffer)
    {
      const int n = static_cast<int>(std::min<size_t>(buffer, in.size() - pos));
      const auto t0 = Clock::now();
      block_fn(in.data() + pos, out.data() + pos, n);
      const auto t1 = Clock::now();
      if (times)
        times->push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
    }
  };
  pass(nullptr);
  std::vector<double> times;
  for (int i = 0; i < iters; ++i)
    pass(&times);
  return times;
}

// Drift-proof comparison of two engines: every block is run through both, back to back, the order alternating
// block by block so neither gets the warm cache. Clock speed changes hit both alike, so the per-block ratio
// is trustworthy even when the absolute times are not. Prints A's and B's median time and the distribution of
// the per-block speed-up of B over A (A time / B time).
void run_ab(const A2Weights& w, const std::string& name_a, const std::string& name_b, const std::vector<double>& in,
            int buffer, int iters)
{
  auto ea = make_engine(name_a, w), eb = make_engine(name_b, w);
  ea->reset(buffer);
  eb->reset(buffer);
  std::vector<float> fin(buffer), fa(buffer), fb(buffer);
  std::vector<double> ta, tb, ratio;
  bool a_first = true;
  for (int it = 0; it <= iters; ++it) // iteration 0 is an untimed warm-up
    for (size_t pos = 0; pos + buffer <= in.size(); pos += buffer)
    {
      for (int k = 0; k < buffer; ++k)
        fin[k] = static_cast<float>(in[pos + k]);
      double t[2];
      for (int turn = 0; turn < 2; ++turn)
      {
        const bool run_a = (turn == 0) == a_first;
        const auto t0 = Clock::now();
        (run_a ? ea : eb)->process(fin.data(), run_a ? fa.data() : fb.data(), buffer);
        const auto t1 = Clock::now();
        t[run_a ? 0 : 1] = std::chrono::duration<double, std::micro>(t1 - t0).count();
      }
      a_first = !a_first;
      if (it > 0)
      {
        ta.push_back(t[0]);
        tb.push_back(t[1]);
        ratio.push_back(t[0] / t[1]);
      }
    }
  const Stats sa = compute(ta), sb = compute(tb), sr = compute(ratio);
  size_t b_wins = 0;
  for (double r : ratio)
    b_wins += r > 1.0;
  std::printf("   A %-10s p50 %7.2f us (min %6.2f)\n   B %-10s p50 %7.2f us (min %6.2f)\n", name_a.c_str(), sa.p50, sa.min,
              name_b.c_str(), sb.p50, sb.min);
  std::printf("   speed-up of B over A, per block: median %.3fx  p10 %.3fx  p90 %.3fx   B faster in %.1f%% of %zu blocks\n",
              sr.p50, percentile([&] { auto v = ratio; std::sort(v.begin(), v.end()); return v; }(), 0.1),
              percentile([&] { auto v = ratio; std::sort(v.begin(), v.end()); return v; }(), 0.9),
              100.0 * double(b_wins) / double(ratio.size()), ratio.size());
}

void report(const std::string& label, const Stats& s, double block_us)
{
  std::printf("   %-14s min %7.2f  p50 %7.2f  p99 %7.2f  p99.9 %8.2f  max %9.2f  mean %7.2f us   RTF(p50) %6.2fx\n",
              label.c_str(), s.min, s.p50, s.p99, s.p999, s.max, s.mean, block_us / s.p50);
}
} // namespace

int main(int argc, char** argv)
{
  std::string model, backend = "scalar", ab_other;
  double slim = 1.0, seconds = 2.0;
  int buffer = 64, iters = 10, core = -1;
  double host_rate = 48000.0;
  bool low_latency = false, monitor = false;
  bool with_nam = false;
  for (int i = 1; i < argc; ++i)
  {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "bench: missing value after " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--model") model = next();
    else if (a == "--slim") slim = std::atof(next().c_str());
    else if (a == "--backend") backend = next();
    else if (a == "--buffer") buffer = std::atoi(next().c_str());
    else if (a == "--seconds") seconds = std::atof(next().c_str());
    else if (a == "--iters") iters = std::atoi(next().c_str());
    else if (a == "--nam") with_nam = true;
    else if (a == "--ab") ab_other = next();
    else if (a == "--core") core = std::atoi(next().c_str());
    else if (a == "--host-rate") host_rate = std::atof(next().c_str());
    else if (a == "--low-latency") low_latency = true;
    else if (a == "--monitor") monitor = true;
    else
    {
      std::cerr << "bench: unknown argument " << a << "\n";
      return 2;
    }
  }
  if (model.empty() || buffer < 1 || iters < 1)
  {
    std::cerr << "bench: --model required; --buffer and --iters must be positive\n";
    return 2;
  }
  if (!std::ifstream(model).good())
  {
    std::printf("SKIP: model file not found: %s\n", model.c_str());
    return 77;
  }
  if (core >= 0)
  {
    if (!pin_to_core(core))
    {
      std::cerr << "bench: cannot pin to core " << core << "\n";
      return 2;
    }
    std::printf("pinned to core %d\n", core);
  }

  try
  {
    const A2Weights w = load_a2_weights(model, slim);
    const double sr = 48000.0, block_us = 1e6 * buffer / sr;
    const size_t total = static_cast<size_t>(seconds * sr);
    std::vector<double> in(total), out(total, 0.0);
    for (size_t i = 0; i < total; ++i)
    {
      const double t = double(i) / sr;
      in[i] = 0.25 * std::sin(2.0 * kPi * 220.0 * t) + 0.10 * std::sin(2.0 * kPi * 1230.0 * t);
    }

    std::printf("== %s (%d channels), buffer %d (%.1f us per block), %.1f s of audio, %d iterations\n", model.c_str(),
                w.channels, buffer, block_us, seconds, iters);

    if (!ab_other.empty())
    {
      run_ab(w, backend, ab_other, in, buffer, iters);
      return 0;
    }
    auto engine = make_engine(backend, w);
    engine->reset(buffer);
    std::vector<float> fin(buffer), fout(buffer);
    report(backend, compute(time_blocks(
                      [&](const double* i, double* o, int n) {
                        for (int k = 0; k < n; ++k)
                          fin[k] = static_cast<float>(i[k]);
                        engine->process(fin.data(), fout.data(), n);
                        for (int k = 0; k < n; ++k)
                          o[k] = fout[k];
                      },
                      in, out, buffer, iters)),
           block_us);

#ifdef NAMTURBO_PROFILE
    if (backend == "avx2")
    {
      a2_avx2_profile_dump();
      a2_lite_avx2_profile_dump();
    }
#endif
    if (check_cpu().supported)
    {
      // The public API, as a plug-in uses it. The double version is timed because the interface of NAM Core uses double: the
      // time includes the conversion to float and back.
      Processor processor(Model::load(model));
      ProcessorConfig config;
      config.sample_rate = host_rate;
      config.max_block = buffer;
      config.variant = slim < 0.5 ? Variant::Lite : Variant::Full;
      config.resampler = low_latency ? ResamplerQuality::LowLatency : ResamplerQuality::Standard;
      config.monitoring = monitor;
      processor.prepare(config);
      char label[64];
      if (host_rate == 48000.0)
        std::snprintf(label, sizeof label, "ours (Processor)");
      else
        std::snprintf(label, sizeof label, "ours (Processor, %.0f Hz%s)", host_rate, low_latency ? ", low latency" : "");
      // The float version, with the signal already in float: the difference to the double row is the cost of the conversion.
      std::vector<float> fin_all(total), fout_all(total);
      for (size_t i = 0; i < total; ++i)
        fin_all[i] = static_cast<float>(in[i]);
      char flabel[80];
      std::snprintf(flabel, sizeof flabel, "%s, float", label);
      report(flabel,
             compute(time_blocks(
                 [&](const double* i, double* o, int n) {
                   const size_t at = static_cast<size_t>(i - in.data());
                   processor.process(fin_all.data() + at, fout_all.data() + at, n);
                   (void)o;
                 },
                 in, out, buffer, iters)),
             1e6 * buffer / host_rate);
      // At another host rate the block lasts buffer / host rate, and the input is the same signal sampled at the host rate.
      report(label,
             compute(time_blocks([&](const double* i, double* o, int n) { processor.process(i, o, n); }, in, out, buffer, iters)),
             1e6 * buffer / host_rate);
    }
#ifndef NAMTURBO_WITH_NAMCORE
    if (with_nam)
    {
      std::cerr << "bench: --nam needs a build with -DNAMTURBO_BASELINE=ON (NAM Core)\n";
      return 2;
    }
#else
    if (with_nam)
      for (const char* which : {"fast", "generic"})
      {
        NamCoreModel m = make_nam_core_dsp(model, slim, which);
        m.dsp->Reset(m.sample_rate, buffer);
        report(std::string("NAM ") + which,
               compute(time_blocks(
                   [&](const double* i, double* o, int n) {
                     NAM_SAMPLE* ip[] = {const_cast<NAM_SAMPLE*>(i)};
                     NAM_SAMPLE* op[] = {o};
                     m.dsp->process(ip, op, n);
                   },
                   in, out, buffer, iters)),
               block_us);
      }
#endif
    return 0;
  }
  catch (const std::exception& e)
  {
    std::cerr << "bench: " << e.what() << "\n";
    return 2;
  }
}
