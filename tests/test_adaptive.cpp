// Tests of the adaptive quality mode (AdaptiveConfig, Processor::set_adaptive, service(), smoothed_load()).
// Usage: test_adaptive <slimmable A2 model.nam> <data_dir>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing, or the CPU lacks AVX2 and FMA).
//
// The real load of this machine is small and not under the test's control, so the tests set thresholds that force each decision:
// a high threshold below any real load means "always too slow", a low threshold above any load means "always room for A2-Full".
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "json.hpp"
#include "namturbo/namturbo.h"

using namespace namturbo;
using json = nlohmann::json;

// A sanitizer slows the engine down by a factor of 10 or more, so the load that the test below expects at the default thresholds (well under 30%)
// need not hold. In such a build the absolute-load checks are skipped; the tests that force their decisions with extreme thresholds are unaffected.
#ifndef __has_feature
  #define __has_feature(x) 0
#endif
#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__) || __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
constexpr bool kSanitized = true;
#else
constexpr bool kSanitized = false;
#endif

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

static std::string g_model;
static bool exists(const std::string& p) { return std::ifstream(p).good(); }

static std::vector<float> make_signal(size_t n)
{
  std::vector<float> x(n);
  for (size_t i = 0; i < n; ++i)
    x[i] = float(0.1 * std::sin(6.283185307179586 * 440.0 * double(i) / 48000.0) + 0.05 * std::sin(6.283185307179586 * 2300.0 * double(i) / 48000.0));
  return x;
}

template <class F>
static std::optional<ErrorCode> error_of(F&& f)
{
  try
  {
    f();
  }
  catch (const Error& e)
  {
    return e.code();
  }
  catch (...)
  {
  }
  return std::nullopt;
}

// A processor and an audio loop that runs on the same thread as service(), so that swaps are never out of date.
struct Rig
{
  std::shared_ptr<const Model> model;
  Processor processor;
  std::vector<float> in, out;
  size_t pos = 0;
  std::vector<Variant> events;

  explicit Rig(std::shared_ptr<const Model> m) : model(std::move(m)), processor(model), in(make_signal(48000)), out(in.size()) {}

  AdaptiveConfig config(double high, double low, double down_ms = 0.0, double up_ms = 0.0)
  {
    AdaptiveConfig a;
    a.enabled = true;
    a.high_load = high;
    a.low_load = low;
    a.smoothing_ms = 20.0;
    a.down_after_ms = down_ms;
    a.up_after_ms = up_ms;
    a.on_switch = [this](Variant v) { events.push_back(v); };
    return a;
  }

  void prepare(const AdaptiveConfig& a, Variant v = Variant::Full)
  {
    ProcessorConfig c;
    c.max_block = 256;
    c.variant = v;
    c.adaptive = a;
    processor.prepare(c);
  }

  // Runs blocks of 64 samples, calling service() every 8 blocks, until `done()` or `max_blocks`.
  template <class Done>
  void run(Done done, int max_blocks)
  {
    for (int b = 0; b < max_blocks && !done(); ++b)
    {
      if (pos + 64 > in.size())
        pos = 0;
      processor.process(in.data() + pos, out.data() + pos, 64);
      pos += 64;
      if (b % 8 == 7)
        processor.service();
    }
  }
  void run_blocks(int n) { run([] { return false; }, n); }
};

static void goes_down_when_too_slow_and_back_up_when_there_is_room()
{
  std::printf("goes_down_when_too_slow_and_back_up_when_there_is_room\n");
  Rig r(Model::load(g_model));
  r.prepare(r.config(1e-6, 1e-9)); // always above the high threshold, never below the low one
  CHECK(r.processor.variant() == Variant::Full);
  r.run([&] { return r.processor.variant() == Variant::Lite; }, 2000);
  CHECK(r.processor.variant() == Variant::Lite);
  CHECK(r.events.size() == 1 && r.events[0] == Variant::Lite);
  CHECK(r.processor.smoothed_load() > 0.0);

  // no flapping: A2-Full is not expected to be cheaper than the (tiny) low threshold, so it stays on A2-Lite
  r.run_blocks(1500);
  CHECK(r.processor.variant() == Variant::Lite && r.events.size() == 1);

  // room again: thresholds under which A2-Full is expected to fit
  r.processor.set_adaptive(r.config(1e9, 1e8));
  r.run([&] { return r.processor.variant() == Variant::Full; }, 2000);
  CHECK(r.processor.variant() == Variant::Full);
  CHECK(r.events.size() == 2 && r.events[1] == Variant::Full);
  r.run_blocks(1500);
  CHECK(r.events.size() == 2); // and it stays there
  r.processor.service();
  std::printf("  down to A2-Lite and up to A2-Full once each, no switching in between: events %zu, swaps finished %llu\n", r.events.size(),
              static_cast<unsigned long long>(r.processor.swaps_completed()));
  CHECK(r.processor.swaps_completed() == 2);
}

static void the_expected_load_of_the_full_variant_decides_the_switch_up()
{
  std::printf("the_expected_load_of_the_full_variant_decides_the_switch_up\n");
  Rig r(Model::load(g_model));
  r.prepare(r.config(1e-6, 1e-9));
  r.run([&] { return r.processor.variant() == Variant::Lite; }, 2000);
  CHECK(r.processor.variant() == Variant::Lite);
  // The load of A2-Lite is taken as the lowest average seen while it settles: a busy machine only ever adds to it.
  double lite_load = 1e9;
  for (int i = 0; i < 30; ++i)
  {
    r.run_blocks(50);
    lite_load = std::min(lite_load, r.processor.smoothed_load());
  }
  // A2-Full costs several times as much as A2-Lite. With a low threshold only just above the load of A2-Lite there is no room for it ...
  AdaptiveConfig tight = r.config(1e9, std::max(lite_load * 1.3, 1e-9));
  r.processor.set_adaptive(tight);
  r.run_blocks(2500);
  CHECK(r.processor.variant() == Variant::Lite && r.events.size() == 1);
  // ... and with a threshold well above the expected load of A2-Full there is
  r.processor.set_adaptive(r.config(1e9, lite_load * 40.0));
  r.run([&] { return r.processor.variant() == Variant::Full; }, 3000);
  CHECK(r.processor.variant() == Variant::Full && r.events.size() == 2);
  std::printf("  load on A2-Lite %.2f%%: no switch up with the low threshold at %.2f%%, a switch up with it at %.1f%%\n", 100.0 * lite_load,
              100.0 * lite_load * 1.3, 100.0 * lite_load * 40.0);
}

static void holds_still_when_the_load_is_low()
{
  std::printf("holds_still_when_the_load_is_low\n");
  Rig r(Model::load(g_model));
  AdaptiveConfig a;
  a.enabled = true;
  a.on_switch = [&](Variant v) { r.events.push_back(v); };
  r.prepare(a); // the default thresholds: 70% and 30%
  r.run_blocks(3000);
  CHECK(kSanitized || (r.processor.variant() == Variant::Full && r.events.empty() && r.processor.swaps_completed() == 0));
  std::printf("  3000 blocks at a load of %.2f%% (high threshold 70%%): no switch\n", 100.0 * r.processor.smoothed_load());
  CHECK(r.processor.smoothed_load() > 0.0 && (kSanitized || r.processor.smoothed_load() < 0.3));
}

static void a_choice_of_the_program_is_respected()
{
  std::printf("a_choice_of_the_program_is_respected\n");
  Rig r(Model::load(g_model));
  r.prepare(r.config(1e9, 1e8)); // would switch up at once if the policy had switched down
  r.run_blocks(100);
  CHECK(r.processor.commit_swap(r.processor.prepare_swap(r.model, Variant::Lite))); // the program chooses A2-Lite
  r.run_blocks(2000);
  CHECK(r.processor.variant() == Variant::Lite && r.events.empty());
  std::printf("  A2-Lite chosen by the program stays, although the thresholds allow A2-Full\n");
}

static void needs_both_variants()
{
  std::printf("needs_both_variants\n");
  std::ifstream is(g_model);
  const json file = json::parse(is);
  const std::string text = file.at("config").at("submodels")[0].at("model").dump();
  auto lite_only = Model::load(text.data(), text.size());
  Rig r(lite_only);
  r.prepare(r.config(1e-6, 1e-9), Variant::Lite);
  r.run_blocks(1500);
  CHECK(r.processor.variant() == Variant::Lite && r.events.empty() && r.processor.swaps_completed() == 0);
  r.processor.set_adaptive(r.config(1e-6, 1e-9)); // turning it on for a model with one variant is harmless
  r.run_blocks(500);
  CHECK(r.events.empty());
}

static void validation_and_smoothed_load()
{
  std::printf("validation_and_smoothed_load\n");
  auto model = Model::load(g_model);
  Processor p(model);
  AdaptiveConfig bad;
  bad.enabled = true;
  CHECK(error_of([&] { p.set_adaptive(bad); }) == ErrorCode::InvalidArgument); // before prepare()
  ProcessorConfig c;
  c.max_block = 256;
  bad.low_load = 0.8;
  bad.high_load = 0.7;
  c.adaptive = bad;
  CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument); // low above high
  bad.low_load = 0.0;
  bad.high_load = 0.7;
  c.adaptive = bad;
  CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument);
  bad.low_load = 0.3;
  bad.smoothing_ms = 0.0;
  c.adaptive = bad;
  CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument);
  bad.smoothing_ms = 100.0;
  bad.up_after_ms = -1.0;
  c.adaptive = bad;
  CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument);

  // neither monitoring nor adaptive mode: no load is kept
  ProcessorConfig plain;
  plain.max_block = 256;
  p.prepare(plain);
  CHECK(error_of([&] { p.set_adaptive(AdaptiveConfig{}); }) == std::nullopt); // switching it off is fine
  std::vector<float> in = make_signal(48000), out(in.size());
  for (size_t pos = 0; pos + 64 <= in.size(); pos += 64)
    p.process(in.data() + pos, out.data() + pos, 64);
  CHECK(p.smoothed_load() == 0.0);

  // monitoring: the smoothed load follows the measured load
  plain.monitoring = true;
  plain.adaptive.smoothing_ms = 200.0;
  p.prepare(plain);
  for (size_t pos = 0; pos + 64 <= in.size(); pos += 64)
    p.process(in.data() + pos, out.data() + pos, 64);
  const double smoothed = p.smoothed_load(), mean = p.stats().mean_load;
  std::printf("  smoothed load %.3f%%, mean load %.3f%% over the same calls\n", 100.0 * smoothed, 100.0 * mean);
  CHECK(smoothed > 0.4 * mean && smoothed < 2.5 * mean);
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

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2 || !exists(argv[1]))
  {
    std::printf("SKIP: model file not found\n");
    return 77;
  }
  if (!check_cpu().supported)
  {
    std::printf("SKIP: this CPU (or OS) lacks AVX2 and FMA\n");
    return 77;
  }
  g_model = argv[1];
  guarded("goes_down_when_too_slow_and_back_up_when_there_is_room", goes_down_when_too_slow_and_back_up_when_there_is_room);
  guarded("the_expected_load_of_the_full_variant_decides_the_switch_up", the_expected_load_of_the_full_variant_decides_the_switch_up);
  guarded("holds_still_when_the_load_is_low", holds_still_when_the_load_is_low);
  guarded("a_choice_of_the_program_is_respected", a_choice_of_the_program_is_respected);
  guarded("needs_both_variants", needs_both_variants);
  guarded("validation_and_smoothed_load", validation_and_smoothed_load);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
