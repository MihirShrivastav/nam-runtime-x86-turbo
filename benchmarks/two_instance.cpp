// two_instance: two models running at once on one thread, like two plug-ins (a pedal capture and an amp capture, say) in one host.
//
//   two_instance --model <nam> [--a full|lite] [--b full|lite] [--seconds 60] [--buffer 64] [--data <dir>]
//                [--nam] [--free-run] [--spin] [--priority] [--core N] [--control-us N]
//
// Both instances process the same number of blocks, alternating A then B every block. Per block it measures the
// combined compute time of the pair and reports p50/p99/p99.9/max and the share of the block's deadline
// (buffer / 48 kHz). By default the loop is PACED to real time (it sleeps until each block's deadline with a
// high-resolution timer), the way an audio thread runs: caches and clocks go cold between blocks, which a loop
// running flat out hides. --free-run turns pacing off. In paced mode the wake-up lateness (what the operating
// system adds) is reported separately from the compute time.
//
// With --nam the same run is repeated with NAM Core's own A2 fast path for both instances, for comparison.
// --spin paces by busy-waiting instead of sleeping, so the core never goes idle between blocks.
// --priority raises the process and thread priority (Windows), as a real-time audio thread has.
// --core N pins the thread to logical core N (on a processor with performance and efficiency cores, to measure one kind of core).
// --control-us N adds two control runs without any audio work: one that spins on the clock for N microseconds (what
// it measures beyond N is the operating system interrupting the thread) and one that does about N microseconds of fixed
// L1-resident arithmetic (what it measures beyond N is the CPU's clock and power state after each sleep).
// Exit codes: 0 pass (no block's compute time exceeded its deadline, all output finite), 1 fail, 2 error, 77 skipped.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
  #ifndef NOMINMAX
    #define NOMINMAX
  #endif
  #ifndef WIN32_LEAN_AND_MEAN
    #define WIN32_LEAN_AND_MEAN
  #endif
  #include <windows.h>
#endif

#include "a2_dispatch.h"
#include "a2_weights.h"
#include "common.h"
#include "namturbo/namturbo.h"
#ifdef NAMTURBO_WITH_NAMCORE
  #include "nam_core.h"
#endif
#include "pin.h"

using namespace namturbo;
using Clock = std::chrono::steady_clock;

namespace
{
struct Instance
{
  std::function<void(const double*, double*, int)> run; // processes one block
  std::vector<double> source; // looped
  size_t pos = 0;
  std::vector<double> in, out;
};

struct Result
{
  std::vector<double> compute_us, lateness_us;
  size_t over_deadline = 0;
  bool finite = true;
};

double percentile(std::vector<double> v, double q)
{
  if (v.empty())
    return 0;
  std::sort(v.begin(), v.end());
  const double idx = q * double(v.size() - 1);
  const size_t i = static_cast<size_t>(idx);
  return i + 1 >= v.size() ? v[i] : v[i] + (v[i + 1] - v[i]) * (idx - double(i));
}

bool g_spin_pacing = false; // wait for each deadline by busy-waiting: the core never idles

// Sleep until `deadline` (accurate to well under a millisecond on Windows 10+), or spin if g_spin_pacing.
void sleep_until_precise(Clock::time_point deadline)
{
  if (g_spin_pacing)
  {
    while (Clock::now() < deadline) {}
    return;
  }
#ifdef _WIN32
  static HANDLE timer = CreateWaitableTimerExW(nullptr, nullptr, CREATE_WAITABLE_TIMER_HIGH_RESOLUTION, TIMER_ALL_ACCESS);
  const auto left = deadline - Clock::now();
  if (timer && left > std::chrono::microseconds(50))
  {
    LARGE_INTEGER due;
    due.QuadPart = -static_cast<LONGLONG>(std::chrono::duration_cast<std::chrono::nanoseconds>(left).count() / 100);
    if (SetWaitableTimer(timer, &due, 0, nullptr, nullptr, FALSE))
    {
      WaitForSingleObject(timer, INFINITE);
      return;
    }
  }
  while (Clock::now() < deadline) {}
#else
  std::this_thread::sleep_until(deadline);
#endif
}

// Fixed compute work that stays in L1: eight independent multiply-add chains. Unlike spinning on the clock, its
// duration depends on the CPU's clock speed and power state, which is the point of the work control.
float g_burn_sink = 0;
void burn(long iters)
{
  float x[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  for (long i = 0; i < iters; ++i)
    for (int k = 0; k < 8; ++k)
      x[k] = x[k] * 0.999999f + 0.5f;
  g_burn_sink += x[0] + x[7];
}

// Iterations of burn() that take about `us` microseconds on a warm core.
long calibrate_burn(double us)
{
  long iters = 1000;
  for (int round = 0; round < 6; ++round)
  {
    double best = 1e30;
    for (int rep = 0; rep < 20; ++rep)
    {
      const auto t0 = Clock::now();
      burn(iters);
      best = std::min(best, std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
    iters = std::max(1L, static_cast<long>(double(iters) * us / best));
  }
  return iters;
}

Result run(std::vector<Instance>& inst, int buffer, double seconds, bool paced, double control_us = 0, long burn_iters = 0)
{
  const double sr = 48000.0;
  const double deadline_us = 1e6 * buffer / sr;
  const size_t blocks = static_cast<size_t>(seconds * sr / buffer);
  const size_t warmup = static_cast<size_t>(sr / buffer); // 1 s, untimed and unrecorded
  Result r;
  r.compute_us.reserve(blocks);
  r.lateness_us.reserve(blocks);
  const auto period = std::chrono::nanoseconds(static_cast<long long>(1e9 * buffer / sr));
  auto next = Clock::now() + period;
  for (size_t b = 0; b < blocks + warmup; ++b)
  {
    for (Instance& s : inst) // input prepared outside the timed region
      for (int i = 0; i < buffer; ++i)
      {
        s.in[i] = s.source[s.pos];
        s.pos = (s.pos + 1) % s.source.size();
      }
    if (paced)
    {
      sleep_until_precise(next);
      next += period;
    }
    const auto t0 = Clock::now();
    if (burn_iters > 0) // work control: fixed compute, sensitive to clock speed and power state
      burn(burn_iters);
    else if (control_us > 0) // clock control: no audio work, just wait on the clock; any excess is the operating system's
      while (std::chrono::duration<double, std::micro>(Clock::now() - t0).count() < control_us) {}
    else
      for (Instance& s : inst)
      {
        s.run(s.in.data(), s.out.data(), buffer);
      }
    const auto t1 = Clock::now();
    for (Instance& s : inst)
      for (int i = 0; i < buffer; ++i)
        r.finite = r.finite && std::isfinite(s.out[i]);
    if (b >= warmup)
    {
      const double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      r.compute_us.push_back(us);
      r.over_deadline += us > deadline_us;
      if (paced)
        r.lateness_us.push_back(std::max(0.0, std::chrono::duration<double, std::micro>(t0 - (next - period)).count()));
    }
  }
  return r;
}

void report(const char* label, const Result& r, int buffer, bool paced)
{
  const double deadline_us = 1e6 * buffer / 48000.0;
  const double p50 = percentile(r.compute_us, 0.5), p99 = percentile(r.compute_us, 0.99),
               p999 = percentile(r.compute_us, 0.999), mx = *std::max_element(r.compute_us.begin(), r.compute_us.end());
  std::printf("   %-26s compute per block (pair): p50 %7.1f us  p99 %7.1f  p99.9 %7.1f  max %8.1f   = %4.1f%% / %4.1f%% / %5.1f%% of the %.0f us deadline\n",
              label, p50, p99, p999, mx, 100 * p50 / deadline_us, 100 * p99 / deadline_us, 100 * mx / deadline_us, deadline_us);
  std::printf("   %-26s blocks over the deadline: %zu of %zu   output finite: %s\n", "", r.over_deadline, r.compute_us.size(),
              r.finite ? "yes" : "NO");
  if (paced)
    std::printf("   %-26s wake-up lateness (operating system): p50 %.0f us  p99 %.0f  max %.0f\n", "",
                percentile(r.lateness_us, 0.5), percentile(r.lateness_us, 0.99),
                *std::max_element(r.lateness_us.begin(), r.lateness_us.end()));
}

std::vector<double> load_source(const std::string& data, const char* name, unsigned seed)
{
  std::vector<double> v;
  const std::string p = data + "/" + name;
  if (std::ifstream(p).good())
  {
    const std::vector<float> f = read_raw(p);
    v.assign(f.begin(), f.end());
  }
  if (v.empty()) // no data directory: a plucked-string-like mixture and noise
  {
    std::mt19937 g(seed);
    std::uniform_real_distribution<double> u(-0.2, 0.2);
    v.resize(48000 * 4);
    for (size_t i = 0; i < v.size(); ++i)
      v[i] = 0.3 * std::sin(2 * 3.14159265358979 * (110.0 + 40.0 * seed) * double(i) / 48000.0) + u(g);
  }
  return v;
}
} // namespace

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::string model, a = "full", b = "full", data = "data";
  double seconds = 60.0;
  int buffer = 64;
  bool with_nam = false, paced = true, priority = false;
  int core = -1;
  double control_us = 0; // >0: also run the control (spin for this long instead of processing)
  for (int i = 1; i < argc; ++i)
  {
    const std::string arg = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc)
      {
        std::cerr << "two_instance: missing value after " << arg << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (arg == "--model") model = next();
    else if (arg == "--a") a = next();
    else if (arg == "--b") b = next();
    else if (arg == "--seconds") seconds = std::atof(next().c_str());
    else if (arg == "--buffer") buffer = std::atoi(next().c_str());
    else if (arg == "--data") data = next();
    else if (arg == "--nam") with_nam = true;
    else if (arg == "--free-run") paced = false;
    else if (arg == "--priority") priority = true;
    else if (arg == "--core") core = std::atoi(next().c_str());
    else if (arg == "--spin") g_spin_pacing = true;
    else if (arg == "--control-us") control_us = std::atof(next().c_str());
    else
    {
      std::cerr << "two_instance: unknown argument " << arg << "\n";
      return 2;
    }
  }
  if (model.empty() || buffer < 1 || seconds <= 0 || (a != "full" && a != "lite") || (b != "full" && b != "lite"))
  {
    std::cerr << "two_instance: --model required; --a/--b are full or lite; --buffer and --seconds must be positive\n";
    return 2;
  }
  if (!std::ifstream(model).good())
  {
    std::printf("SKIP: model file not found: %s\n", model.c_str());
    return 77;
  }
  if (core >= 0 && !pin_to_core(core))
  {
    std::cerr << "two_instance: cannot pin to core " << core << "\n";
    return 2;
  }
#ifdef _WIN32
  if (priority)
  {
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
  }
#else
  (void)priority;
#endif

  try
  {
    const std::vector<double> src_a = load_source(data, "sweep10s.raw", 1), src_b = load_source(data, "synth.raw", 2);
    auto make = [&](bool ours, const std::string& kind, const std::vector<double>& source) {
      Instance s;
      const double slim = kind == "full" ? 1.0 : 0.0;
      if (ours)
      {
        auto processor = std::make_shared<Processor>(Model::load(model));
        ProcessorConfig config;
        config.max_block = buffer;
        config.variant = kind == "full" ? Variant::Full : Variant::Lite;
        processor->prepare(config);
        s.run = [processor](const double* in, double* out, int n) { processor->process(in, out, n); };
      }
      else
      {
#ifdef NAMTURBO_WITH_NAMCORE
        auto nam = std::make_shared<NamCoreModel>(make_nam_core_dsp(model, slim, "fast"));
        nam->dsp->Reset(48000.0, buffer);
        s.run = [nam](const double* in, double* out, int n) {
          NAM_SAMPLE* ip[] = {const_cast<NAM_SAMPLE*>(in)};
          NAM_SAMPLE* op[] = {out};
          nam->dsp->process(ip, op, n);
        };
#else
        (void)slim;
        throw std::runtime_error("--nam needs a build with -DNAMTURBO_BASELINE=ON (NAM Core)");
#endif
      }
      s.source = source;
      s.in.assign(buffer, 0.0);
      s.out.assign(buffer, 0.0);
      return s;
    };
    std::printf("== two instances (A2-%s + A2-%s) on one thread, %.0f s of audio, buffer %d, %s%s\n", a.c_str(), b.c_str(),
                seconds, buffer, paced ? (g_spin_pacing ? "paced to real time (busy-wait)" : "paced to real time (sleeping)") : "free-running", priority ? ", raised priority" : "");
    std::vector<Instance> ours;
    ours.push_back(make(true, a, src_a));
    ours.push_back(make(true, b, src_b));
    const Result r_ours = run(ours, buffer, seconds, paced);
    report("ours (AVX2 engines)", r_ours, buffer, paced);
    bool ok = r_ours.over_deadline == 0 && r_ours.finite;
    if (with_nam)
    {
      std::vector<Instance> nam;
      nam.push_back(make(false, a, src_a));
      nam.push_back(make(false, b, src_b));
      const Result r_nam = run(nam, buffer, seconds, paced);
      report("NAM Core (fast path)", r_nam, buffer, paced);
      std::printf("   median pair time: ours %.1f us, NAM Core %.1f us (%.2fx)\n", percentile(r_ours.compute_us, 0.5),
                  percentile(r_nam.compute_us, 0.5), percentile(r_nam.compute_us, 0.5) / percentile(r_ours.compute_us, 0.5));
    }
    if (control_us > 0)
    {
      std::vector<Instance> none;
      const Result r_ctl = run(none, buffer, seconds, paced, control_us);
      char label[64];
      std::snprintf(label, sizeof label, "control (spin %.0f us)", control_us);
      report(label, r_ctl, buffer, paced);
      const long iters = calibrate_burn(control_us);
      const Result r_work = run(none, buffer, seconds, paced, control_us, iters);
      std::snprintf(label, sizeof label, "control (%.0f us of FMA work)", control_us);
      report(label, r_work, buffer, paced);
    }
    std::printf("%s: %s\n", ok ? "PASSED" : "FAILED",
                ok ? "no block of the pair exceeded its deadline" : "a block exceeded its deadline, or the output was not finite");
    return ok ? 0 : 1;
  }
  catch (const std::exception& e)
  {
    std::cerr << "two_instance: " << e.what() << "\n";
    return 2;
  }
}
