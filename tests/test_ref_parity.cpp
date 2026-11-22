// Tests for the scalar reference engine against NAM Core's golden output.
// Usage: test_ref_parity <model.nam> <data_dir>   (data_dir holds golden_*.raw and *.raw inputs)
// Exit codes: 0 pass, 1 failure, 77 skipped (model or golden files missing).
#include <algorithm>
#include <cstdio>
#include <exception>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include "a2_ref.h"
#include "a2_weights.h"
#include "common.h"
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

struct Model { const char* name; double slim; const char* tag; };
static const Model kModels[] = {{"A2-Full", 1.0, "full"}, {"A2-Lite", 0.0, "lite"}};

static std::string g_model, g_data;

// Process `in` in chunks of `block`, after reset(max_block). Returns the output.
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

static bool exists(const std::string& p) { return std::ifstream(p).good(); }

static void ref_matches_golden()
{
  std::printf("ref_matches_golden\n");
  for (const Model& m : kModels)
  {
    const A2Weights w = load_a2_weights(g_model, m.slim);
    auto engine = make_a2_ref(w);
    CHECK(engine->prewarm_samples() == 6347);
    for (const char* input : {"synth", "sweep10s"})
    {
      const std::string in_path = g_data + "/" + input + ".raw";
      const std::string g_path = g_data + "/golden_" + m.tag + "_" + input + ".raw";
      if (!exists(in_path) || !exists(g_path))
        continue; // sweep10s needs the v3 WAV; synth is always present
      std::vector<float> in = read_raw(in_path), golden = read_raw(g_path);
      const bool heavy = std::string(input) == "sweep10s";
      if (heavy) // keep the slow reference quick: the first 2 s, which are the chirps
      {
        in.resize(96000);
        golden.resize(96000);
      }
      const std::vector<int> blocks = heavy ? std::vector<int>{1, 64} : std::vector<int>{1, 16, 32, 48, 64, 128, 256, 512};
      for (int b : blocks)
      {
        const double e = esr_db(run(*engine, in, b, 512), golden);
        const bool ok = e <= parity_limit_db(w.channels);
        std::printf("  %s %-8s block %3d: ESR %.1f dB (limit %.0f) %s\n", m.name, input, b, e, parity_limit_db(w.channels),
                    ok ? "ok" : "FAIL");
        CHECK(ok);
      }
    }
  }
}

static void ref_block_split_invariance()
{
  std::printf("ref_block_split_invariance\n");
  std::vector<float> in = read_raw(g_data + "/synth.raw");
  in.resize(std::min<size_t>(in.size(), kMaxEngineBlock)); // "one block" must respect the engine cap
  for (const Model& m : kModels)
  {
    auto engine = make_a2_ref(load_a2_weights(g_model, m.slim));
    const std::vector<float> whole = run(*engine, in, int(in.size()), int(in.size()));

    engine->reset(512);
    std::vector<float> split(in.size());
    std::mt19937 rng(5);
    std::uniform_int_distribution<int> dist(1, 512);
    for (size_t pos = 0; pos < in.size();)
    {
      const int n = int(std::min<size_t>(dist(rng), in.size() - pos));
      engine->process(in.data() + pos, split.data() + pos, n);
      pos += n;
    }
    const bool same = std::memcmp(whole.data(), split.data(), whole.size() * sizeof(float)) == 0;
    std::printf("  %s: random block sizes vs one block: %s\n", m.name, same ? "bit-identical" : "DIFFERENT");
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
  if (argc < 3 || !exists(argv[1]) || !exists(std::string(argv[2]) + "/golden_full_synth.raw"))
  {
    std::printf("SKIP: model or golden files not found\n");
    return 77;
  }
  g_model = argv[1];
  g_data = argv[2];
  guarded("ref_matches_golden", ref_matches_golden);
  guarded("ref_block_split_invariance", ref_block_split_invariance);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
