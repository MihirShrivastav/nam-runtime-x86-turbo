// Dispatcher tests. Usage: test_dispatch <model.nam>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing).
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "a2_avx2.h"
#include "a2_dispatch.h"
#include "a2_weights.h"
#include "cpu_features.h"

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
static bool throws(F&& f)
{
  try
  {
    f();
  }
  catch (const std::exception&)
  {
    return true;
  }
  return false;
}

static void test_choose_backend_is_pure_and_conservative()
{
  std::printf("choose_backend\n");
  const CpuFeatures none{false, false}, avx2_only{true, false}, fma_only{false, true}, both{true, true};
  CHECK(choose_backend(both, Backend::Auto) == Backend::Avx2);
  CHECK(choose_backend(none, Backend::Auto) == Backend::Scalar);
  CHECK(choose_backend(avx2_only, Backend::Auto) == Backend::Scalar); // the kernels use FMA too
  CHECK(choose_backend(fma_only, Backend::Auto) == Backend::Scalar);
  CHECK(choose_backend(none, Backend::Scalar) == Backend::Scalar);
  CHECK(choose_backend(both, Backend::Scalar) == Backend::Scalar);
  CHECK(choose_backend(both, Backend::Avx2) == Backend::Avx2);
  CHECK(throws([&] { choose_backend(none, Backend::Avx2); }));
  CHECK(throws([&] { choose_backend(avx2_only, Backend::Avx2); }));
  CHECK(throws([&] { choose_backend(fma_only, Backend::Avx2); }));
}

static void test_is_accelerated()
{
  std::printf("is_accelerated\n");
  CHECK(is_accelerated(Backend::Avx2));
  CHECK(!is_accelerated(Backend::Scalar)); // the reference: correct but not for production
}

static void test_make_engine(const std::string& model)
{
  std::printf("make_a2_engine\n");
  const A2Weights full = load_a2_weights(model, 1.0);
  const A2Weights lite = load_a2_weights(model, 0.0);
  CHECK(full.channels == 8 && lite.channels == 3);

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> u(-0.5f, 0.5f);
  std::vector<float> in(3000), a(in.size()), b(in.size());
  for (auto& s : in)
    s = u(rng);

  // Auto on this machine gives exactly what the AVX2 engine gives, when the CPU supports it.
  if (cpu_supports_avx2_fma())
  {
    auto via_dispatch = make_a2_engine(full, Backend::Auto);
    auto direct = make_a2_avx2(full);
    via_dispatch->reset(64);
    direct->reset(64);
    for (size_t p = 0; p < in.size(); p += 64)
    {
      const int n = int(std::min<size_t>(64, in.size() - p));
      via_dispatch->process(in.data() + p, a.data() + p, n);
      direct->process(in.data() + p, b.data() + p, n);
    }
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    CHECK(make_a2_engine(full, Backend::Avx2) != nullptr);
  }

  // A2-Lite has its own AVX2 engine: Auto and an explicit Avx2 request both give exactly what it gives.
  if (cpu_supports_avx2_fma())
  {
    Backend chosen = Backend::Scalar;
    auto lite_dispatch = make_a2_engine(lite, Backend::Auto, &chosen);
    CHECK(chosen == Backend::Avx2);
    auto lite_direct = make_a2_lite_avx2(lite);
    lite_dispatch->reset(64);
    lite_direct->reset(64);
    for (size_t p = 0; p < in.size(); p += 64)
    {
      const int n = int(std::min<size_t>(64, in.size() - p));
      lite_dispatch->process(in.data() + p, a.data() + p, n);
      lite_direct->process(in.data() + p, b.data() + p, n);
    }
    CHECK(std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0);
    CHECK(make_a2_engine(lite, Backend::Avx2) != nullptr);
  }

  CHECK(make_a2_engine(full, Backend::Scalar) != nullptr);
  CHECK(make_a2_engine(lite, Backend::Scalar) != nullptr);
}

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::printf("usage: test_dispatch <model.nam>\n");
    return 2;
  }
  setvbuf(stdout, nullptr, _IONBF, 0);
  if (!std::ifstream(argv[1]).good())
  {
    std::printf("SKIP: model file not found: %s\n", argv[1]);
    return 77;
  }
  try
  {
    test_choose_backend_is_pure_and_conservative();
    test_is_accelerated();
    test_make_engine(argv[1]);
  }
  catch (const std::exception& e)
  {
    std::printf("EXCEPTION: %s\n", e.what());
    return 1;
  }
  std::printf(g_failures ? "FAILED (%d)\n" : "PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
