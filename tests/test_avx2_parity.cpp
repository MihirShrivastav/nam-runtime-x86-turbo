// Tests for the AVX2 engines. Usage: test_avx2_parity <model.nam> <data_dir> [slim]
// slim 1.0 (default) tests the A2-Full engine, 0.0 the A2-Lite engine.
// Exit codes: 0 pass, 1 failure, 77 skipped (model/golden files missing, or CPU without AVX2+FMA).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <cstring>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "a2_avx2.h"
#include "a2_ref.h"
#include "a2_weights.h"
#include "common.h"
#include "cpu_features.h"
#include "parity_limits.h"

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

static std::string g_model, g_data, g_tag = "full"; // g_tag names the golden files: golden_<tag>_<input>.raw
static double g_slim = 1.0;
static int g_channels = 8;
static bool exists(const std::string& p) { return std::ifstream(p).good(); }

// The AVX2 engine for the model under test.
static std::unique_ptr<A2Engine> make_engine(const A2Weights& w)
{
  return w.channels == 8 ? make_a2_avx2(w) : make_a2_lite_avx2(w);
}

// Process `in` in chunks of `block` after reset(max_block).
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

static bool all_finite(const std::vector<float>& v, size_t from)
{
  for (size_t i = from; i < v.size(); ++i)
    if (!std::isfinite(v[i]))
      return false;
  return true;
}

static void avx2_matches_golden_all_blocks()
{
  std::printf("avx2_matches_golden_all_blocks\n");
  auto engine = make_engine(load_a2_weights(g_model, g_slim));
  const std::vector<int> blocks = {1, 7, 8, 9, 16, 31, 48, 64, 65, 128, 256, 512};
  for (const char* input : {"synth", "sweep10s"})
  {
    const std::string in_path = g_data + "/" + input + ".raw", g_path = g_data + "/golden_" + g_tag + "_" + input + ".raw";
    if (!exists(in_path) || !exists(g_path))
      continue;
    std::vector<float> in = read_raw(in_path), golden = read_raw(g_path);
    const bool heavy = std::string(input) == "sweep10s"; // the whole 10 s signal: fewer block sizes
    for (int b : heavy ? std::vector<int>{1, 64, 65} : blocks)
    {
      const double e = esr_db(run(*engine, in, b, 512), golden);
      const bool ok = e <= parity_limit_db(g_channels);
      std::printf("  %-8s block %3d: ESR %.1f dB (limit %.0f) %s\n", input, b, e, parity_limit_db(g_channels), ok ? "ok" : "FAIL");
      CHECK(ok);
    }
  }
}

static void avx2_block_split_invariance()
{
  std::printf("avx2_block_split_invariance\n");
  std::vector<float> in = read_raw(g_data + "/synth.raw");
  in.resize(std::min<size_t>(in.size(), kMaxEngineBlock)); // "one block" must respect the engine cap
  auto engine = make_engine(load_a2_weights(g_model, g_slim));
  const std::vector<float> whole = run(*engine, in, int(in.size()), int(in.size()));
  engine->reset(512);
  std::vector<float> split(in.size());
  std::mt19937 rng(11);
  std::uniform_int_distribution<int> dist(1, 512);
  for (size_t pos = 0; pos < in.size();)
  {
    const int n = int(std::min<size_t>(dist(rng), in.size() - pos));
    engine->process(in.data() + pos, split.data() + pos, n);
    pos += n;
  }
  const bool same = bit_equal(whole, split);
  std::printf("  random block sizes vs one block: %s\n", same ? "bit-identical" : "DIFFERENT");
  CHECK(same);
}

static void avx2_unaligned_pointers()
{
  std::printf("avx2_unaligned_pointers\n");
  std::vector<float> in = read_raw(g_data + "/synth.raw");
  in.resize(24000);
  auto engine = make_engine(load_a2_weights(g_model, g_slim));
  const std::vector<float> base = run(*engine, in, 64, 64);

  for (int off : {1, 2, 3, 5, 7})
  {
    // Buffers whose usable start is `off` floats past a 32-byte boundary.
    std::vector<float> raw_in(in.size() + 16), raw_out(in.size() + 16);
    auto aligned_start = [](std::vector<float>& v) {
      float* p = v.data();
      while (reinterpret_cast<uintptr_t>(p) % 32 != 0)
        ++p;
      return p;
    };
    float* pin = aligned_start(raw_in) + off;
    float* pout = aligned_start(raw_out) + off;
    std::copy(in.begin(), in.end(), pin);
    engine->reset(64);
    for (size_t pos = 0; pos < in.size(); pos += 64)
    {
      const int n = int(std::min<size_t>(64, in.size() - pos));
      engine->process(pin + pos, pout + pos, n);
    }
    const bool same = std::memcmp(pout, base.data(), base.size() * sizeof(float)) == 0;
    std::printf("  offset %d floats: %s\n", off, same ? "identical to aligned run" : "DIFFERENT");
    CHECK(same);
  }
}

static void avx2_nan_recovery()
{
  std::printf("avx2_nan_recovery\n");
  std::vector<float> in = read_raw(g_data + "/synth.raw");
  const size_t at = 1000;
  in[at] = std::nanf("");
  auto engine = make_engine(load_a2_weights(g_model, g_slim));
  const std::vector<float> out = run(*engine, in, 64, 64);
  CHECK(!std::isfinite(out[at])); // the test is not vacuous: the NaN did get through
  const size_t clean = at + a2_receptive_field() + 512;
  const bool ok = all_finite(out, clean);
  std::printf("  NaN at sample %zu: output finite again from sample %zu: %s\n", at, clean, ok ? "yes" : "NO");
  CHECK(ok);
}

static void avx2_extreme_level()
{
  std::printf("avx2_extreme_level\n");
  std::vector<float> in = read_raw(g_data + "/synth.raw");
  for (float& x : in)
    x *= 8.0f; // peak about 3.2, far above full scale
  const A2Weights w = load_a2_weights(g_model, g_slim);
  auto ref = make_a2_ref(w);
  auto avx = make_engine(w);
  const std::vector<float> r = run(*ref, in, 64, 64), a = run(*avx, in, 64, 64);
  const double e = esr_db(a, r);
  std::printf("  input x8: ESR vs scalar reference %.1f dB, finite: %s\n", e, all_finite(a, 0) ? "yes" : "NO");
  CHECK(all_finite(a, 0));
  CHECK(e <= parity_limit_db(g_channels));
}

static void avx2_rejects_wrong_model_and_repeats()
{
  std::printf("avx2_rejects_wrong_model_and_repeats\n");
  auto rejected = [](auto make, const A2Weights& w) {
    try
    {
      make(w);
    }
    catch (const std::invalid_argument&)
    {
      return true;
    }
    return false;
  };
  const bool full_rejects_lite = rejected(make_a2_avx2, load_a2_weights(g_model, 0.0));
  const bool lite_rejects_full = rejected(make_a2_lite_avx2, load_a2_weights(g_model, 1.0));
  std::printf("  A2-Full engine rejects 3 channels: %s; A2-Lite engine rejects 8 channels: %s\n",
              full_rejects_lite ? "yes" : "NO", lite_rejects_full ? "yes" : "NO");
  CHECK(full_rejects_lite);
  CHECK(lite_rejects_full);

  std::vector<float> in = read_raw(g_data + "/synth.raw");
  in.resize(24000);
  auto engine = make_engine(load_a2_weights(g_model, g_slim));
  const bool same = bit_equal(run(*engine, in, 64, 64), run(*engine, in, 64, 64));
  std::printf("  reset() twice gives identical output: %s\n", same ? "yes" : "NO");
  CHECK(same);
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
  g_tag = g_slim < 0.5 ? "lite" : "full";
  if (argc < 3 || !exists(argv[1]) || !exists(std::string(argv[2]) + "/golden_" + g_tag + "_synth.raw"))
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
  g_channels = load_a2_weights(g_model, g_slim).channels;
  std::printf("testing the %s engine (%d channels)\n", g_tag.c_str(), g_channels);
  guarded("avx2_matches_golden_all_blocks", avx2_matches_golden_all_blocks);
  guarded("avx2_block_split_invariance", avx2_block_split_invariance);
  guarded("avx2_unaligned_pointers", avx2_unaligned_pointers);
  guarded("avx2_nan_recovery", avx2_nan_recovery);
  guarded("avx2_extreme_level", avx2_extreme_level);
  guarded("avx2_rejects_wrong_model_and_repeats", avx2_rejects_wrong_model_and_repeats);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
