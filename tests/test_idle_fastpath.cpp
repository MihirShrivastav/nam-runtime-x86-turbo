// Tests of the idle-silence fast path (F8) and no-prewarm swap optimization (OPT-SWAP).
// Usage: test_idle_fastpath <slimmable A2 model.nam> <data_dir>
// Exit codes: 0 pass, 1 failure, 77 skipped (model missing, or the CPU lacks AVX2 and FMA).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fstream>
#include <random>
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

static bool bit_equal(const std::vector<float>& a, const std::vector<float>& b)
{
  return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) == 0;
}

static std::vector<float> make_gated_signal()
{
  // Pattern: 8000 zeros (enters idle) -> 2000 active -> 3000 zeros (< 6347) -> 1500 active -> 12000 zeros (> 6347, enters idle)
  // -> 3000 active (wake-up) -> 8000 zeros -> 500 active.
  std::vector<float> x;
  auto push_zeros = [&](size_t n) { x.insert(x.end(), n, 0.0f); };
  auto push_active = [&](size_t n, unsigned seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> d(-0.25f, 0.25f);
    for (size_t i = 0; i < n; ++i)
      x.push_back(d(rng));
  };
  push_zeros(8000);
  push_active(2000, 1);
  push_zeros(3000);
  push_active(1500, 2);
  push_zeros(12000);
  push_active(3000, 3);
  push_zeros(8000);
  push_active(500, 4);
  return x;
}

static void bit_exact_across_silence_and_wakeup(const std::shared_ptr<const Model>& model)
{
  std::printf("bit_exact_across_silence_and_wakeup\n");
  const std::vector<float> in = make_gated_signal();
  for (Variant v : {Variant::Full, Variant::Lite})
  {
    // Reference: single huge block (contains non-zero samples, so the idle skip never triggers on this call).
    Processor ref(model);
    ProcessorConfig cr;
    cr.sample_rate = 48000.0;
    cr.max_block = int(in.size());
    cr.variant = v;
    ref.prepare(cr);
    std::vector<float> out_ref(in.size());
    ref.process(in.data(), out_ref.data(), int(in.size()));

    // Small fixed blocks (64 samples): triggers idle skip repeatedly during every long zero run.
    Processor fixed(model);
    ProcessorConfig cf;
    cf.sample_rate = 48000.0;
    cf.max_block = 64;
    cf.variant = v;
    fixed.prepare(cf);
    std::vector<float> out_fixed(in.size());
    for (size_t pos = 0; pos < in.size(); pos += 64)
      fixed.process(in.data() + pos, out_fixed.data() + pos, int(std::min<size_t>(64, in.size() - pos)));

    // Random block splits (1..512) and in-place processing.
    Processor rnd(model), inplace(model);
    cf.max_block = 512;
    rnd.prepare(cf);
    inplace.prepare(cf);
    std::vector<float> out_rnd(in.size()), out_ip = in;
    std::mt19937 rng(99);
    std::uniform_int_distribution<int> bdist(1, 512);
    for (size_t pos = 0; pos < in.size();)
    {
      const int k = int(std::min<size_t>(size_t(bdist(rng)), in.size() - pos));
      rnd.process(in.data() + pos, out_rnd.data() + pos, k);
      inplace.process(out_ip.data() + pos, out_ip.data() + pos, k);
      pos += size_t(k);
    }

    const bool ok = bit_equal(out_ref, out_fixed) && bit_equal(out_ref, out_rnd) && bit_equal(out_ref, out_ip);
    std::printf("  %s: 64-sample, random splits and in-place bit-identical to unskipped reference: %s\n",
                v == Variant::Full ? "A2-Full" : "A2-Lite", ok ? "yes" : "NO");
    CHECK(ok);
  }
}

static void idle_blocks_are_much_faster(const std::shared_ptr<const Model>& model)
{
  std::printf("idle_blocks_are_much_faster\n");
  for (Variant v : {Variant::Full, Variant::Lite})
  {
    Processor p(model);
    ProcessorConfig c;
    c.sample_rate = 48000.0;
    c.max_block = 64;
    c.variant = v;
    p.prepare(c);

    std::vector<float> zeros(64, 0.0f), active(64, 0.1f), out(64);
    // Warm up into active state
    for (int i = 0; i < 10; ++i)
      p.process(active.data(), out.data(), 64);

    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2000; ++i)
      p.process(active.data(), out.data(), 64);
    const double active_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / 2000.0;

    // Settle into idle (> 6347 zeros)
    for (int i = 0; i < 120; ++i)
      p.process(zeros.data(), out.data(), 64);

    const auto t1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 2000; ++i)
      p.process(zeros.data(), out.data(), 64);
    const double idle_us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t1).count() / 2000.0;

    std::printf("  %s: active %.2f us/block, idle %.3f us/block (%.1fx faster)\n", v == Variant::Full ? "A2-Full" : "A2-Lite", active_us, idle_us,
                active_us / std::max(idle_us, 1e-6));
    CHECK(idle_us < active_us * 0.25);
  }
}

static void swap_early_and_late_bit_exact(const std::shared_ptr<const Model>& model)
{
  std::printf("swap_early_and_late_bit_exact\n");
  const std::vector<float> in = make_gated_signal();
  // Test both swap during startup window (< 6347 samples: prewarm required) and after (> 6347 samples: prewarm skipped).
  for (size_t swap_at : {size_t(1024), size_t(10240)})
  {
    Processor ref(model), sw(model);
    ProcessorConfig c;
    c.sample_rate = 48000.0;
    c.max_block = 128;
    c.variant = Variant::Full;
    c.swap_fade_ms = 5.0;
    ref.prepare(c);
    sw.prepare(c);

    std::vector<float> out_ref(in.size()), out_sw(in.size());
    for (size_t pos = 0; pos < in.size(); pos += 128)
    {
      const int k = int(std::min<size_t>(128, in.size() - pos));
      if (pos == swap_at)
      {
        SwapTicket t = sw.prepare_swap(model, Variant::Full);
        CHECK(sw.commit_swap(std::move(t)));
      }
      ref.process(in.data() + pos, out_ref.data() + pos, k);
      sw.process(in.data() + pos, out_sw.data() + pos, k);
      sw.service();
    }
    CHECK(sw.swaps_completed() == 1 && sw.swaps_dropped() == 0);
    const size_t settle = swap_at + 512;
    bool same = true;
    for (size_t i = settle; i < in.size(); ++i)
      same = same && (out_ref[i] == out_sw[i]);
    std::printf("  same-model swap at sample %zu (%s): bit-identical after fade: %s\n", swap_at,
                swap_at < 6347 ? "early, prewarmed" : "late, no-prewarm", same ? "yes" : "NO");
    CHECK(same);
  }
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
    bit_exact_across_silence_and_wakeup(model);
    idle_blocks_are_much_faster(model);
    swap_early_and_late_bit_exact(model);
  }
  catch (const std::exception& e)
  {
    std::printf("  EXCEPTION: %s\n", e.what());
    ++g_failures;
  }
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
