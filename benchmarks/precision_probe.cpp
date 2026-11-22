// precision_probe: how far from the exact answer is each float engine, on inputs that stress rounding?
//
//   precision_probe <model.nam> <slim> [data_dir] [--check]
//
// For each stimulus, the error of NAM Core's A2 fast path, our scalar float reference and our AVX2 engine,
// each against our double-precision reference (the best available stand-in for the exact answer). An engine is
// as good as NAM Core if its error is no larger than NAM Core's own; an absolute limit is meaningless for A2-Lite,
// which amplifies rounding by an input-dependent amount.
// Error is ESR in dB, or max |error| in dBFS where the output is quiet (ESR divides by almost nothing there).
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "a2_dispatch.h"
#include "a2_ref.h"
#include "a2_weights.h"
#include "common.h"
#include "nam_core.h"

using namespace namturbo;

static std::vector<float> run_engine(A2Engine& e, const std::vector<float>& in, int block)
{
  e.reset(block);
  std::vector<float> out(in.size());
  for (size_t p = 0; p < in.size(); p += block)
    e.process(in.data() + p, out.data() + p, int(std::min<size_t>(block, in.size() - p)));
  return out;
}

static std::vector<float> run_nam(nam::DSP& d, const std::vector<float>& in, int block)
{
  d.Reset(48000.0, block);
  std::vector<double> x(in.begin(), in.end()), y(in.size());
  for (size_t p = 0; p < in.size(); p += block)
  {
    const int n = int(std::min<size_t>(block, in.size() - p));
    NAM_SAMPLE* ip[] = {x.data() + p};
    NAM_SAMPLE* op[] = {y.data() + p};
    d.process(ip, op, n);
  }
  std::vector<float> out(y.size());
  for (size_t i = 0; i < y.size(); ++i)
    out[i] = static_cast<float>(y[i]);
  return out;
}

static double score(const std::vector<float>& a, const std::vector<float>& truth, bool* quiet_out)
{
  double pow = 0;
  for (float t : truth)
    pow += double(t) * double(t);
  const bool quiet = std::sqrt(pow / double(truth.size())) < 0.01;
  *quiet_out = quiet;
  if (!quiet)
    return esr_db(a, truth);
  double worst = 0;
  for (size_t i = 0; i < a.size(); ++i)
    worst = std::max(worst, std::fabs(double(a[i]) - double(truth[i])));
  return 20.0 * std::log10(std::max(worst, 1e-30));
}

int main(int argc, char** argv)
{
  if (argc < 3)
  {
    std::printf("usage: precision_probe <model.nam> <slim> [data_dir]\n");
    return 2;
  }
  const std::string model = argv[1];
  const double slim = std::atof(argv[2]);
  bool check = false; // --check: exit 1 unless our engine is as precise as NAM Core on average over DC levels
  for (int i = 3; i < argc; ++i)
    check = check || std::string(argv[i]) == "--check";
  const A2Weights w = load_a2_weights(model, slim);
  const int N = 48000;
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  struct Stim { std::string name; std::vector<float> x; };
  std::vector<Stim> stims;
  auto add = [&](const char* name, auto gen) {
    Stim s{name, std::vector<float>(N)};
    for (int i = 0; i < N; ++i)
      s.x[i] = gen(i);
    stims.push_back(std::move(s));
  };
  add("dc +1", [](int) { return 1.0f; });
  add("dc -1", [](int) { return -1.0f; });
  add("dc 0.5", [](int) { return 0.5f; });
  add("dc 0.1", [](int) { return 0.1f; });
  add("nyquist +-1", [](int i) { return (i & 1) ? -1.0f : 1.0f; });
  add("square 100 Hz", [](int i) { return ((i / 240) & 1) ? -1.0f : 1.0f; });
  add("white full scale", [&](int) { return u(rng); });
  add("white 0.1", [&](int) { return 0.1f * u(rng); });
  add("chirp 20-20k", [](int i) {
    const double t = double(i) / 48000.0;
    return float(0.9 * std::sin(2.0 * 3.14159265358979 * (20.0 * t + 0.5 * 19980.0 * t * t)));
  });
  if (argc >= 4 && std::string(argv[3]) != "--check")
  {
    const std::string p = std::string(argv[3]) + "/sweep10s.raw";
    if (std::ifstream(p).good())
    {
      std::vector<float> s = read_raw(p);
      s.resize(std::min<size_t>(s.size(), 96000));
      stims.push_back({"hard sweep (2 s)", s});
    }
  }

  auto truth_engine = make_a2_ref_f64(w);
  auto scalar_engine = make_a2_ref(w);
  auto fast_engine = make_a2_engine(w, Backend::Auto);
  NamCoreModel nam = make_nam_core_dsp(model, slim, "fast");

  std::printf("%d channels. Error vs float64 reference (ESR dB, or max|err| dBFS for quiet outputs)\n", w.channels);
  std::printf("   %-18s %10s %10s %10s   %s\n", "stimulus", "NAM fast", "scalar", "ours", "ours vs NAM (positive = worse)");
  for (const Stim& s : stims)
  {
    const auto truth = run_engine(*truth_engine, s.x, 64);
    bool q1, q2, q3;
    const double e_nam = score(run_nam(*nam.dsp, s.x, 64), truth, &q1);
    const double e_scalar = score(run_engine(*scalar_engine, s.x, 64), truth, &q2);
    const double e_ours = score(run_engine(*fast_engine, s.x, 64), truth, &q3);
    std::printf("   %-18s %10.1f %10.1f %10.1f   %+6.1f dB %s\n", s.name.c_str(), e_nam, e_scalar, e_ours, e_ours - e_nam,
                q1 ? "(quiet output)" : "");
  }

  // DC makes the network a fixed point, so each engine's error is one deterministic rounding offset that moves by a
  // few dB with the operation order. One level says little; compare the average over many levels.
  std::printf("\nDC levels -1.0 .. +1.0 in steps of 0.1 (not 0): mean error power over the levels, and per-level win count\n");
  double sum_nam = 0, sum_scalar = 0, sum_ours = 0;
  int levels = 0, worse_than_nam_by_3db = 0, better_than_nam = 0;
  for (int i = -10; i <= 10; ++i)
  {
    if (i == 0)
      continue;
    std::vector<float> x(24000, 0.1f * float(i));
    const auto truth = run_engine(*truth_engine, x, 64);
    bool q;
    const double e_nam = score(run_nam(*nam.dsp, x, 64), truth, &q);
    const double e_scalar = score(run_engine(*scalar_engine, x, 64), truth, &q);
    const double e_ours = score(run_engine(*fast_engine, x, 64), truth, &q);
    sum_nam += std::pow(10.0, e_nam / 10);
    sum_scalar += std::pow(10.0, e_scalar / 10);
    sum_ours += std::pow(10.0, e_ours / 10);
    worse_than_nam_by_3db += (e_ours - e_nam) > 3.0;
    better_than_nam += e_ours < e_nam;
    ++levels;
  }
  const double mean_nam = 10 * std::log10(sum_nam / levels), mean_ours = 10 * std::log10(sum_ours / levels);
  std::printf("   mean ESR  NAM fast %.1f dB   scalar %.1f dB   ours %.1f dB   (ours better than NAM at %d of %d levels, "
              "worse by more than 3 dB at %d)\n",
              mean_nam, 10 * std::log10(sum_scalar / levels), mean_ours, better_than_nam, levels, worse_than_nam_by_3db);
  if (check)
  {
    // Not systematically less precise than NAM Core on the ill-conditioned fixed-point inputs.
    const bool ok = mean_ours <= mean_nam + 1.5;
    std::printf("%s: mean DC error %.1f dB vs NAM Core %.1f dB (allowed: at most 1.5 dB worse)\n", ok ? "PASSED" : "FAILED",
                mean_ours, mean_nam);
    return ok ? 0 : 1;
  }
  return 0;
}
