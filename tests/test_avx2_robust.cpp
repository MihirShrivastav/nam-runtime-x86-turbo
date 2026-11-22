// Adversarial and abnormal scenarios for the AVX2 engines. Usage: test_avx2_robust <model.nam> <data_dir> [slim]
// slim 1.0 (default) tests the A2-Full engine, 0.0 the A2-Lite engine.
// Exit codes: 0 pass, 1 failure, 77 skipped (model/golden files missing, or CPU without AVX2+FMA).
//
// "Robust" here means: no crash, no hang, no memory overrun, defined behaviour on misuse, correct output
// on hostile but legal audio, timing that does not depend on what the audio is, and the caller's CPU
// state left as it was found.
#include <xmmintrin.h>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "a2_avx2.h"
#include "a2_ref.h"
#include "a2_weights.h"
#include "common.h"
#include "cpu_features.h"
#include "parity_limits.h"

using namespace namturbo;
using Clock = std::chrono::steady_clock;

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

static std::string g_model, g_data;
static double g_slim = 1.0; // 1.0: the A2-Full engine, 0.0: the A2-Lite engine
static bool exists(const std::string& p) { return std::ifstream(p).good(); }

// The AVX2 engine for the model under test.
static std::unique_ptr<A2Engine> make_engine(const A2Weights& w)
{
  return w.channels == 8 ? make_a2_avx2(w) : make_a2_lite_avx2(w);
}
static std::unique_ptr<A2Engine> make_engine_at(const A2Weights& w, uint32_t position)
{
  return w.channels == 8 ? make_a2_avx2_at_position(w, position) : make_a2_lite_avx2_at_position(w, position);
}
static const A2Weights& weights()
{
  static const A2Weights w = load_a2_weights(g_model, g_slim);
  return w;
}

static std::vector<float> run(A2Engine& e, const std::vector<float>& in, int block, int max_block)
{
  e.reset(max_block);
  std::vector<float> out(in.size());
  for (size_t pos = 0; pos < in.size(); pos += block)
  {
    const int n = int(std::min<size_t>(block, in.size() - pos));
    e.process(in.data() + pos, out.data() + pos, n);
  }
  return out;
}

static bool bit_equal(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static bool all_finite(const std::vector<float>& v, size_t from = 0)
{
  for (size_t i = from; i < v.size(); ++i)
    if (!std::isfinite(v[i]))
      return false;
  return true;
}

// Loud outputs: ESR within the parity limit. Quiet outputs (rms below 0.01): ESR is meaningless there,
// because it divides by almost nothing while every implementation, NAM Core's included, carries about
// 1e-6 of absolute error from the float32 resolution of the large DC state inside the network
// (measured against a float64 truth: max error 3e-8 to 1.3e-6 on silence, impulse, Nyquist and
// 1e-6 noise). So quiet outputs must agree to 5e-6 absolute (-106 dBFS), below 16-bit quantisation.
static bool agrees(const std::vector<float>& a, const std::vector<float>& ref, double* esr_out = nullptr)
{
  double pow = 0;
  for (float r : ref)
    pow += double(r) * double(r);
  if (std::sqrt(pow / double(ref.size())) < 0.01)
  {
    double worst = 0;
    for (size_t i = 0; i < a.size(); ++i)
      worst = std::max(worst, std::fabs(double(a[i]) - double(ref[i])));
    if (esr_out)
      *esr_out = 20.0 * std::log10(std::max(worst, 1e-30)); // reported as max |error| in dBFS
    return worst < 5e-6;
  }
  const double e = esr_db(a, ref);
  if (esr_out)
    *esr_out = e;
  return e <= truth_limit_db(weights().channels);
}

static bool quiet(const std::vector<float>& ref)
{
  double p = 0;
  for (float r : ref) p += double(r) * double(r);
  return std::sqrt(p / double(ref.size())) < 0.01;
}

struct Stim { std::string name; std::vector<float> x; };

static std::vector<Stim> stimuli()
{
  const int N = 48000; // 1 s
  std::mt19937 rng(3);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  std::vector<Stim> s;
  auto add = [&](const char* name, auto gen) {
    Stim st{name, std::vector<float>(N)};
    for (int i = 0; i < N; ++i)
      st.x[i] = gen(i);
    s.push_back(std::move(st));
  };
  add("silence", [](int) { return 0.0f; });
  add("impulse", [](int i) { return i == 1000 ? 1.0f : 0.0f; });
  add("dc +1", [](int) { return 1.0f; });
  add("dc -1", [](int) { return -1.0f; });
  add("dc 0.5", [](int) { return 0.5f; });
  add("nyquist +-1", [](int i) { return (i & 1) ? -1.0f : 1.0f; });
  add("square 100 Hz", [](int i) { return ((i / 240) & 1) ? -1.0f : 1.0f; });
  add("white full scale", [&](int) { return u(rng); });
  add("white 1e-6", [&](int) { return 1e-6f * u(rng); });
  add("denormal-range noise", [&](int) { return 1e-39f * u(rng); });
  add("over range x4", [&](int) { return 4.0f * u(rng); });
  add("chirp 20-20k", [](int i) {
    const double t = double(i) / 48000.0, f0 = 20.0, f1 = 20000.0;
    return float(0.9 * std::sin(2.0 * 3.14159265358979 * (f0 * t + 0.5 * (f1 - f0) * t * t)));
  });
  add("burst then silence", [&](int i) { return i < 14400 ? 0.5f * u(rng) : 0.0f; });
  return s;
}

static void hostile_but_legal_audio_matches_reference()
{
  std::printf("hostile_but_legal_audio_matches_reference\n");
  auto avx = make_engine(weights());
  auto ref = make_a2_ref_f64(weights()); // the exact answer, not another float engine's rounding
  for (const Stim& st : stimuli())
  {
    const std::vector<float> a = run(*avx, st.x, 64, 64), r = run(*ref, st.x, 64, 64);
    double e = 0;
    const bool ok = all_finite(a) && agrees(a, r, &e);
    std::printf("  %-22s %7.1f dB (%s)  %s\n", st.name.c_str(), e, quiet(r) ? "max |err|, dBFS" : "ESR", ok ? "ok" : "FAIL");
    CHECK(ok);
  }
}

static void non_finite_input_recovers()
{
  std::printf("non_finite_input_recovers\n");
  const std::vector<float> base = [] { std::vector<float> v(30000); std::mt19937 g(5); std::uniform_real_distribution<float> d(-0.3f, 0.3f); for (float& x : v) x = d(g); return v; }();
  struct Case { const char* name; size_t at, len; float value; };
  const float inf = std::numeric_limits<float>::infinity();
  const Case cases[] = {{"+Inf single", 1000, 1, inf},           {"-Inf single", 1000, 1, -inf},
                        {"NaN burst of 100", 1000, 100, std::nanf("")}, {"3e38 burst of 50 (overflows)", 1000, 50, 3e38f},
                        {"-FLT_MAX burst of 10", 1000, 10, -std::numeric_limits<float>::max()}};
  auto avx = make_engine(weights());
  for (const Case& c : cases)
  {
    std::vector<float> in = base;
    for (size_t i = 0; i < c.len; ++i)
      in[c.at + i] = c.value;
    const std::vector<float> out = run(*avx, in, 64, 64);
    const size_t clean = c.at + c.len + a2_receptive_field() + 64;
    const bool saw_damage = !all_finite(out, c.at); // sanity: the hostile input did reach the output
    const bool ok = all_finite(out, clean);
    std::printf("  %-22s damage seen: %s, finite again from sample %zu: %s\n", c.name, saw_damage ? "yes" : "no", clean, ok ? "yes" : "NO");
    CHECK(ok);
  }
  // The engine is still correct afterwards (state fully recovered, not just finite).
  const std::vector<float> after = run(*avx, base, 64, 64), fresh = run(*make_engine(weights()), base, 64, 64);
  CHECK(bit_equal(after, fresh));
}

static void api_misuse_is_defined()
{
  std::printf("api_misuse_is_defined\n");
  auto expect_invalid = [](const char* label, auto fn) {
    bool ok = false;
    try { fn(); } catch (const std::invalid_argument&) { ok = true; } catch (...) {}
    std::printf("  %-34s %s\n", label, ok ? "std::invalid_argument" : "WRONG (no throw or other exception)");
    CHECK(ok);
  };
  float in[512] = {}, out[512] = {};
  auto e = make_engine(weights());
  expect_invalid("process before reset", [&] { e->process(in, out, 64); });
  expect_invalid("reset(0)", [&] { e->reset(0); });
  expect_invalid("reset(-5)", [&] { e->reset(-5); });
  expect_invalid("reset(INT_MAX)", [&] { e->reset(INT_MAX); });
  expect_invalid("reset(65537) above the cap", [&] { e->reset(65537); });
  e->reset(64);
  expect_invalid("process n=0", [&] { e->process(in, out, 0); });
  expect_invalid("process n=-1", [&] { e->process(in, out, -1); });
  expect_invalid("process n=max+1", [&] { e->process(in, out, 65); });
  // After misuse the engine is still usable and still correct.
  std::vector<float> x(4000);
  std::mt19937 g(2);
  std::uniform_real_distribution<float> d(-0.3f, 0.3f);
  for (float& v : x) v = d(g);
  const bool same = bit_equal(run(*e, x, 64, 64), run(*make_engine(weights()), x, 64, 64));
  std::printf("  usable and correct after misuse: %s\n", same ? "yes" : "NO");
  CHECK(same);
  // The largest allowed block works and matches.
  auto big = make_engine(weights());
  std::vector<float> y(65536 + 100);
  for (float& v : y) v = d(g);
  const bool big_ok = all_finite(run(*big, y, 65536, 65536));
  std::printf("  max_block = 65536 runs: %s\n", big_ok ? "yes" : "NO");
  CHECK(big_ok);
}

static void buffer_guard_bands_are_untouched()
{
  std::printf("buffer_guard_bands_are_untouched\n");
  auto e = make_engine(weights());
  e->reset(64);
  const uint32_t pattern = 0xDEADBEEFu;
  float canary;
  std::memcpy(&canary, &pattern, 4);
  int bad = 0;
  for (int n : {1, 2, 3, 7, 8, 9, 31, 63, 64})
  {
    const int G = 64;
    std::vector<float> in(G + n + G, canary), out(G + n + G, canary);
    for (int i = 0; i < n; ++i)
      in[G + i] = 0.1f * float(i % 7);
    e->process(in.data() + G, out.data() + G, n);
    for (int i = 0; i < G; ++i)
      if (std::memcmp(&out[i], &pattern, 4) || std::memcmp(&out[G + n + i], &pattern, 4))
        ++bad;
    // Reading past n would pull canary (a NaN pattern) into the output.
    for (int i = 0; i < n; ++i)
      if (!std::isfinite(out[G + i]))
        ++bad;
  }
  std::printf("  guard words disturbed or NaN pulled in: %d\n", bad);
  CHECK(bad == 0);
}

static void in_place_processing_matches()
{
  std::printf("in_place_processing_matches\n");
  std::vector<float> x(20000);
  std::mt19937 g(9);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (float& v : x) v = d(g);
  auto e = make_engine(weights());
  const std::vector<float> separate = run(*e, x, 64, 64);
  std::vector<float> inplace = x;
  e->reset(64);
  for (size_t pos = 0; pos < inplace.size(); pos += 64)
    e->process(inplace.data() + pos, inplace.data() + pos, int(std::min<size_t>(64, inplace.size() - pos)));
  const bool same = bit_equal(separate, inplace);
  std::printf("  in == out gives the same output as separate buffers: %s\n", same ? "yes" : "NO");
  CHECK(same);
}

static void sample_counter_wraparound()
{
  std::printf("sample_counter_wraparound\n");
  std::vector<float> x(20000);
  std::mt19937 g(4);
  std::uniform_real_distribution<float> d(-0.4f, 0.4f);
  for (float& v : x) v = d(g);
  auto normal = make_engine(weights());
  const std::vector<float> base = run(*normal, x, 64, 512);
  for (uint32_t start : {0xFFFFFFF0u, 0xFFFFFF00u, 0xFFFFE000u, 0xFFFFF000u})
  {
    auto wrapped = make_engine_at(weights(), start);
    const std::vector<float> out = run(*wrapped, x, 64, 512);
    const bool same = bit_equal(base, out);
    std::printf("  counter starts at 0x%08X (wraps during prewarm or audio): %s\n", start, same ? "identical" : "DIFFERENT");
    CHECK(same);
  }
}

static void pathological_block_patterns()
{
  std::printf("pathological_block_patterns\n");
  std::vector<float> x(40000);
  std::mt19937 g(6);
  std::uniform_real_distribution<float> d(-0.4f, 0.4f);
  for (float& v : x) v = d(g);
  auto e = make_engine(weights());
  const std::vector<float> whole = run(*e, x, int(x.size()), int(x.size()));

  struct Pattern { const char* name; std::vector<int> sizes; };
  std::vector<int> ramp_up, ramp_down, all_small;
  for (int i = 1; i <= 64; ++i) { ramp_up.push_back(i); ramp_down.push_back(65 - i); }
  for (int i = 1; i <= 17; ++i) all_small.push_back(i);
  const Pattern pats[] = {{"alternating 1 and 512", {1, 512}}, {"ramp 1..64", ramp_up}, {"ramp 64..1", ramp_down},
                          {"1..17 repeating", all_small},     {"always 1", {1}},        {"511, 1, 512, 2", {511, 1, 512, 2}}};
  for (const Pattern& p : pats)
  {
    e->reset(512);
    std::vector<float> out(x.size());
    size_t pos = 0, k = 0;
    while (pos < x.size())
    {
      const int n = int(std::min<size_t>(p.sizes[k++ % p.sizes.size()], x.size() - pos));
      e->process(x.data() + pos, out.data() + pos, n);
      pos += n;
    }
    const bool same = bit_equal(whole, out);
    std::printf("  %-24s %s\n", p.name, same ? "bit-identical" : "DIFFERENT");
    CHECK(same);
  }
}

static void pathological_models_do_not_crash()
{
  std::printf("pathological_models_do_not_crash\n");
  std::vector<float> x(8000);
  std::mt19937 g(8);
  std::uniform_real_distribution<float> d(-0.4f, 0.4f);
  for (float& v : x) v = d(g);

  auto with = [&](const char* label, auto mutate, auto check) {
    A2Weights w = weights();
    mutate(w);
    auto e = make_engine(w);
    const std::vector<float> out = run(*e, x, 64, 64);
    const bool ok = check(out);
    std::printf("  %-34s %s\n", label, ok ? "ok" : "FAIL");
    CHECK(ok);
  };
  const float nan = std::nanf(""), big = 1e30f;
  with("all weights NaN", [&](A2Weights& w) { for (auto& L : w.layers) { std::fill(L.conv.begin(), L.conv.end(), nan); } w.head_b = nan; },
       [](const std::vector<float>& o) { return !o.empty(); }); // must simply return
  with("conv weights 1e30 (overflow)", [&](A2Weights& w) { for (auto& L : w.layers) std::fill(L.conv.begin(), L.conv.end(), big); },
       [](const std::vector<float>& o) { return !o.empty(); });
  with("all weights zero: constant output", [](A2Weights& w) {
         w.rechannel.fill(0); for (auto& L : w.layers) { std::fill(L.conv.begin(), L.conv.end(), 0.0f); L.conv_b.fill(0); L.mixin.fill(0); L.l1x1.fill(0); L.l1x1_b.fill(0); }
         for (auto& h : w.head_w)
           h.fill(0);
         w.head_b = 0.25f;
         w.head_scale = 2.0f;
       },
       [](const std::vector<float>& o) {
         for (float v : o)
         {
           if (v != 0.5f)
             return false;
         }
         return true;
       });
}

static void soak_mixed_content_random_blocks()
{
  std::printf("soak_mixed_content_random_blocks\n");
  const int N = 48000 * 20; // 20 s
  std::vector<float> x(N);
  std::mt19937 g(12);
  std::uniform_real_distribution<float> u(-1.0f, 1.0f);
  for (int i = 0; i < N; ++i)
  {
    const int seg = (i / 24000) % 8; // half-second segments of different content
    const double t = double(i) / 48000.0;
    switch (seg)
    {
      case 0: x[i] = 0.5f * u(g); break;
      case 1: x[i] = 0.0f; break;
      case 2: x[i] = float(0.9 * std::sin(2 * 3.14159265358979 * 110.0 * t)); break;
      case 3: x[i] = (i & 1) ? -0.8f : 0.8f; break;
      case 4: x[i] = (i % 4800 == 0) ? 1.0f : 0.0f; break;
      case 5: x[i] = 4.0f * u(g); break;
      case 6: x[i] = 1e-39f * u(g); break;
      default: x[i] = 0.7f; break;
    }
  }
  auto e = make_engine(weights());
  const std::vector<float> fixed = run(*e, x, 64, 512);
  e->reset(512);
  std::vector<float> varied(N);
  std::mt19937 bg(13);
  std::uniform_int_distribution<int> bd(1, 512);
  for (int pos = 0; pos < N;)
  {
    const int n = std::min(bd(bg), N - pos);
    e->process(x.data() + pos, varied.data() + pos, n);
    pos += n;
  }
  float peak = 0;
  for (float v : fixed)
    peak = std::max(peak, std::fabs(v));
  const bool ok_same = bit_equal(fixed, varied), ok_fin = all_finite(fixed), ok_bound = peak < 16.0f;
  std::printf("  20 s mixed content: finite %s, peak %.2f, block 64 vs random blocks: %s\n", ok_fin ? "yes" : "NO", peak,
              ok_same ? "bit-identical" : "DIFFERENT");
  CHECK(ok_fin);
  CHECK(ok_bound);
  CHECK(ok_same);
}

static double median_block_us(A2Engine& e, const std::vector<float>& x, int block, int passes)
{
  std::vector<float> out(block);
  std::vector<double> times;
  for (int p = 0; p < passes; ++p)
  {
    e.reset(block);
    for (size_t pos = 0; pos + block <= x.size(); pos += block)
    {
      const auto t0 = Clock::now();
      e.process(x.data() + pos, out.data(), block);
      times.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
    }
  }
  std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
  return times[times.size() / 2];
}

static void timing_does_not_depend_on_the_audio()
{
  std::printf("timing_does_not_depend_on_the_audio\n");
  auto e = make_engine(weights());
  double lo = 1e30, hi = 0;
  const Stim* slow = nullptr;
  const std::vector<Stim> st = stimuli();
  for (const Stim& s : st)
  {
    const double m = median_block_us(*e, s.x, 64, 3);
    std::printf("  %-22s median %7.2f us per 64-sample block\n", s.name.c_str(), m);
    if (m < lo) lo = m;
    if (m > hi) { hi = m; slow = &s; }
  }
  std::printf("  slowest / fastest = %.2fx (slowest: %s)\n", hi / lo, slow ? slow->name.c_str() : "-");
  CHECK(hi / lo < 3.0); // loose on purpose: this machine's timing is noisy; the printed ratio is the datum
}

static void denormal_input_is_flushed()
{
  std::printf("denormal_input_is_flushed\n");
  const size_t N = 12000;
  const std::vector<float> silence(N, 0.0f);
  std::vector<float> tiny(N);
  std::mt19937 g(14);
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (float& v : tiny)
    v = 1e-39f * d(g); // every sample is a denormal float
  auto e = make_engine(weights());
  const bool same = bit_equal(run(*e, silence, 64, 64), run(*e, tiny, 64, 64));
  std::printf("  denormal-level input gives output bit-identical to silence: %s\n", same ? "yes" : "NO");
  CHECK(same);
}

static void caller_cpu_state_is_restored()
{
  std::printf("caller_cpu_state_is_restored\n");
  std::vector<float> x(2000, 0.1f);
  std::vector<float> out(64);
  auto e = make_engine(weights());
  e->reset(64);
  const unsigned saved = _mm_getcsr();
  bool ok = true;
  for (unsigned csr : {0x1F80u /* default */, 0x1F80u | 0x8040u /* FTZ+DAZ already on */, 0x1F80u | 0x2000u /* round toward zero bits */})
  {
    _mm_setcsr(csr);
    e->process(x.data(), out.data(), 64);
    const unsigned after = _mm_getcsr();
    const bool same = (after & ~0x3Fu) == (csr & ~0x3Fu); // ignore the sticky exception flags
    std::printf("  MXCSR 0x%04X before, 0x%04X after: %s\n", csr, after, same ? "restored" : "CHANGED");
    ok = ok && same;
  }
  _mm_setcsr(saved);
  CHECK(ok);
}

static void tiny_max_block_matches_normal()
{
  std::printf("tiny_max_block_matches_normal\n");
  // The smallest buffers stress the ring sizing: rings must hold the history plus whatever the spare lanes of a
  // partly empty vector write past the end of the block.
  std::vector<float> x(6000);
  std::mt19937 g(21);
  std::uniform_real_distribution<float> d(-0.5f, 0.5f);
  for (float& v : x) v = d(g);
  auto base_engine = make_engine(weights());
  const std::vector<float> expect = run(*base_engine, x, 64, 64);
  for (int m : {1, 2, 3, 5, 7, 8, 9, 15, 17})
  {
    auto e = make_engine(weights());
    const bool same = bit_equal(run(*e, x, m, m), expect);
    std::printf("  max_block %2d, blocks of %2d: %s\n", m, m, same ? "bit-identical to block 64" : "DIFFERENT");
    CHECK(same);
  }
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
  catch (...)
  {
    std::printf("  UNKNOWN EXCEPTION in %s\n", name);
    ++g_failures;
  }
}

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0); // a crash must not swallow the output that led to it
  if (argc >= 4)
    g_slim = std::atof(argv[3]);
  if (argc < 3 || !exists(argv[1]) || !exists(std::string(argv[2]) + "/golden_" + (g_slim < 0.5 ? "lite" : "full") + "_synth.raw"))
  {
    std::printf("SKIP: model or golden files not found\n");
    return 77;
  }
  if (!cpu_supports_avx2_fma())
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2+FMA\n");
    return 77;
  }
  g_model = argv[1];
  g_data = argv[2];
  std::printf("testing the %d-channel engine\n", weights().channels);
  guarded("hostile_but_legal_audio_matches_reference", hostile_but_legal_audio_matches_reference);
  guarded("non_finite_input_recovers", non_finite_input_recovers);
  guarded("api_misuse_is_defined", api_misuse_is_defined);
  guarded("buffer_guard_bands_are_untouched", buffer_guard_bands_are_untouched);
  guarded("in_place_processing_matches", in_place_processing_matches);
  guarded("sample_counter_wraparound", sample_counter_wraparound);
  guarded("pathological_block_patterns", pathological_block_patterns);
  guarded("tiny_max_block_matches_normal", tiny_max_block_matches_normal);
  guarded("pathological_models_do_not_crash", pathological_models_do_not_crash);
  guarded("soak_mixed_content_random_blocks", soak_mixed_content_random_blocks);
  guarded("timing_does_not_depend_on_the_audio", timing_does_not_depend_on_the_audio);
  guarded("denormal_input_is_flushed", denormal_input_is_flushed);
  guarded("caller_cpu_state_is_restored", caller_cpu_state_is_restored);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
