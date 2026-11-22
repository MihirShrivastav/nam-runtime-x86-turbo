// What happens on a processor without AVX2 or FMA: a clear error, never a crash. The CPU features are overridden for the test
// (src/cpu_features.h), so it runs on any machine; the parts that need a working engine are skipped on a machine that really
// lacks the instructions.
// Usage: test_cpu_failure <A2 model.nam>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing).
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "a2_dispatch.h"
#include "a2_weights.h"
#include "cpu_features.h"
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

template <class F>
static std::optional<ErrorCode> error_of(F&& f, std::string* what = nullptr)
{
  try
  {
    f();
  }
  catch (const Error& e)
  {
    if (what)
      *what = e.what();
    return e.code();
  }
  catch (...)
  {
  }
  return std::nullopt;
}

// Restores the real CPU features when a case ends, even by an exception.
struct Override
{
  explicit Override(bool avx2, bool fma)
  {
    CpuFeatures f;
    f.avx2 = avx2;
    f.fma = fma;
    set_cpu_features_for_tests(f);
  }
  ~Override() { clear_cpu_features_for_tests(); }
};

static bool contains(const std::string& text, const char* word) { return text.find(word) != std::string::npos; }

static void check_cpu_names_what_is_missing()
{
  std::printf("check_cpu_names_what_is_missing\n");
  {
    Override o(false, false);
    const CpuSupport c = check_cpu();
    CHECK(!c.supported && contains(c.detail, "AVX2") && contains(c.detail, "FMA"));
  }
  {
    Override o(false, true);
    const CpuSupport c = check_cpu();
    CHECK(!c.supported && contains(c.detail, "AVX2") && !contains(c.detail, "FMA"));
  }
  {
    Override o(true, false);
    const CpuSupport c = check_cpu();
    CHECK(!c.supported && contains(c.detail, "FMA") && !contains(c.detail, "AVX2"));
  }
  {
    Override o(true, true);
    CHECK(check_cpu().supported && check_cpu().detail.empty());
  }
}

static void loading_works_and_prepare_fails_cleanly(const std::string& model_path, bool real_cpu_ok)
{
  std::printf("loading_works_and_prepare_fails_cleanly\n");
  for (const auto& missing : {std::pair<bool, bool>{false, false}, {false, true}, {true, false}})
  {
    Override o(missing.first, missing.second);
    // Reading a model needs no special instruction.
    std::shared_ptr<const Model> model;
    try
    {
      model = Model::load(model_path);
    }
    catch (const std::exception& e)
    {
      std::printf("  EXCEPTION loading: %s\n", e.what());
      ++g_failures;
      return;
    }
    Processor p(model);
    ProcessorConfig c;
    c.max_block = 64;
    std::string what;
    CHECK(error_of([&] { p.prepare(c); }, &what) == ErrorCode::CpuUnsupported);
    CHECK(contains(what, missing.first ? "FMA" : "AVX2"));
    std::printf("  missing %s%s: \"%s\"\n", missing.first ? "" : "AVX2 ", missing.second ? "" : "FMA", what.c_str());

    // The processor stays usable: process() before a successful prepare() gives silence, and prepare_swap() says it is not prepared.
    std::vector<float> in(100, 0.5f), out(100, 9.0f);
    p.process(in.data(), out.data(), 100);
    bool silent = true;
    for (float v : out)
      silent = silent && v == 0.0f;
    CHECK(silent);
    CHECK(error_of([&] { p.prepare_swap(model, Variant::Full); }) == ErrorCode::InvalidArgument);
    CHECK(error_of([&] { p.set_adaptive(AdaptiveConfig{}); }) == ErrorCode::InvalidArgument);
    CHECK(p.latency_samples() == 0);
  }
  if (real_cpu_ok)
  {
    // The same processor prepares and runs once the (pretended) limitation is gone.
    auto model = Model::load(model_path);
    Processor p(model);
    ProcessorConfig c;
    c.max_block = 64;
    {
      Override o(false, false);
      CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::CpuUnsupported);
    }
    CHECK(error_of([&] { p.prepare(c); }) == std::nullopt);
    std::vector<float> in(64, 0.1f), out(64, 0.0f);
    p.process(in.data(), out.data(), 64);
    CHECK(true);
  }
}

static void a_failed_prepare_leaves_a_running_processor_running(const std::string& model_path)
{
  std::printf("a_failed_prepare_leaves_a_running_processor_running\n");
  auto model = Model::load(model_path);
  Processor p(model);
  ProcessorConfig c;
  c.max_block = 64;
  p.prepare(c);
  std::vector<float> in(64, 0.2f), a(64), b(64);
  p.process(in.data(), a.data(), 64);
  {
    Override o(false, false);
    CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::CpuUnsupported);
  }
  // If prepare() throws, the processor is left as it was: it carries on with the state it had (the second block continues the first).
  p.process(in.data(), b.data(), 64);
  Processor q(model);
  q.prepare(c);
  std::vector<float> qa(64), qb(64);
  q.process(in.data(), qa.data(), 64);
  q.process(in.data(), qb.data(), 64);
  CHECK(a == qa && b == qb);
}

static void the_engine_dispatcher_refuses_and_falls_back(const std::string& model_path)
{
  std::printf("the_engine_dispatcher_refuses_and_falls_back\n");
  const A2Weights w = load_a2_weights(model_path, 1.0);
  Override o(false, false);
  bool threw = false;
  try
  {
    make_a2_engine(w, Backend::Avx2, nullptr);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  CHECK(threw); // an explicit request for AVX2 is refused ...
  Backend chosen = Backend::Avx2;
  auto engine = make_a2_engine(w, Backend::Auto, &chosen);
  CHECK(engine && chosen == Backend::Scalar); // ... and an automatic choice falls back to the reference engine (used by tests and tools only)
}

int main(int argc, char** argv)
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  if (argc < 2 || !std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found\n");
    return 77;
  }
  const std::string model = argv[1];
  const bool real_cpu_ok = check_cpu().supported;
  try
  {
    check_cpu_names_what_is_missing();
    loading_works_and_prepare_fails_cleanly(model, real_cpu_ok);
    if (real_cpu_ok)
      a_failed_prepare_leaves_a_running_processor_running(model);
    the_engine_dispatcher_refuses_and_falls_back(model);
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION: %s\n", e.what());
    ++g_failures;
  }
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
