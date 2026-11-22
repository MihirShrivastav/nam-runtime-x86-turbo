// Tests of swapping the model or the variant of a running processor (Processor::prepare_swap, commit_swap, service).
// Usage: test_swap <slimmable A2 model.nam> <data_dir>
// Exit codes: 0 pass, 1 failure, 77 skipped (model or data missing, or the CPU lacks AVX2 and FMA).
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <exception>
#include <fstream>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <vector>

#include "json.hpp"
#include "namturbo/namturbo.h"

using namespace namturbo;
using json = nlohmann::json;

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

static std::vector<float> make_signal(double rate, double seconds)
{
  const size_t n = size_t(seconds * rate);
  std::vector<float> x(n);
  std::mt19937 rng(21);
  for (size_t i = 0; i < n; ++i)
  {
    const double t = double(i) / rate;
    const double noise = (double(rng() >> 8) / 16777216.0 - 0.5) * 0.05;
    x[i] = float(0.12 * std::sin(6.283185307179586 * 330.0 * t) + 0.08 * std::sin(6.283185307179586 * 1700.0 * t) + noise);
  }
  return x;
}

static std::vector<int> make_blocks(size_t total, unsigned seed, int max_size = 300)
{
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> dist(1, max_size);
  std::vector<int> blocks;
  for (size_t done = 0; done < total;)
  {
    const int n = int(std::min<size_t>(size_t(dist(rng)), total - done));
    blocks.push_back(n);
    done += size_t(n);
  }
  return blocks;
}

static void prepare_with(Processor& p, double rate, Variant v, double fade_ms = 10.0, ResamplerQuality q = ResamplerQuality::Standard)
{
  ProcessorConfig c;
  c.sample_rate = rate;
  c.max_block = 512;
  c.variant = v;
  c.swap_fade_ms = fade_ms;
  c.resampler = q;
  p.prepare(c);
}

// Processes blocks [from, to) of the list; `pos` is the sample position of block `from` and is advanced.
static void run_blocks(Processor& p, const std::vector<float>& in, std::vector<float>& out, const std::vector<int>& blocks, size_t from, size_t to, size_t& pos)
{
  for (size_t b = from; b < to; ++b)
  {
    p.process(in.data() + pos, out.data() + pos, blocks[b]);
    pos += size_t(blocks[b]);
  }
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

// Index of the first block that starts at or after the sample position `at`.
static size_t block_at(const std::vector<int>& blocks, size_t at, size_t& position)
{
  size_t pos = 0, b = 0;
  while (b < blocks.size() && pos < at)
    pos += size_t(blocks[b++]);
  position = pos;
  return b;
}

static float fade_gain(int i, int fade_len) { return static_cast<float>(0.5 * (1.0 - std::cos(3.14159265358979323846 * (i + 0.5) / fade_len))); }

static void same_model_swap_changes_nothing()
{
  std::printf("same_model_swap_changes_nothing\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 2.5);
  const std::vector<int> blocks = make_blocks(in.size(), 5);
  for (double fade_ms : {10.0, 0.0})
    for (Variant v : {Variant::Full, Variant::Lite})
    {
      Processor swapped(model), plain(model);
      prepare_with(swapped, 48000.0, v, fade_ms);
      prepare_with(plain, 48000.0, v, fade_ms);
      std::vector<float> a(in.size()), b(in.size());
      size_t pa = 0, pb = 0, S = 0;
      const size_t swap_block = block_at(blocks, 60000, S);
      run_blocks(swapped, in, a, blocks, 0, swap_block, pa);
      SwapTicket ticket = swapped.prepare_swap(Model::load(g_model), v); // another Model object with the same contents
      CHECK(static_cast<bool>(ticket));
      CHECK(swapped.commit_swap(std::move(ticket)));
      CHECK(!ticket);
      CHECK(swapped.swap_busy());
      run_blocks(swapped, in, a, blocks, swap_block, blocks.size(), pa);
      run_blocks(plain, in, b, blocks, 0, blocks.size(), pb);
      CHECK(!swapped.swap_busy() && swapped.swaps_completed() == 1 && swapped.swaps_dropped() == 0);

      const int fade_len = int(std::lround(fade_ms * 1e-3 * 48000.0));
      double worst_inside = 0.0;
      bool identical_outside = true;
      for (size_t i = 0; i < in.size(); ++i)
      {
        const bool inside = i >= S && i < S + size_t(fade_len) + 300;
        if (inside)
          worst_inside = std::max(worst_inside, std::fabs(double(a[i]) - double(b[i])));
        else
          identical_outside = identical_outside && a[i] == b[i];
      }
      std::printf("  %s, fade %.0f ms: identical outside the fade, differs by at most %.1e inside it\n", v == Variant::Full ? "A2-Full" : "A2-Lite", fade_ms, worst_inside);
      CHECK(identical_outside);
      CHECK(worst_inside < 1e-6);
    }
}

static void variant_swap_equals_the_references()
{
  std::printf("variant_swap_equals_the_references\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 2.5);
  const std::vector<int> blocks = make_blocks(in.size(), 9);
  for (Variant from : {Variant::Full, Variant::Lite})
    for (double fade_ms : {10.0, 25.0, 0.0})
    {
      const Variant to = from == Variant::Full ? Variant::Lite : Variant::Full;
      Processor swapped(model), ref_from(model), ref_to(model);
      prepare_with(swapped, 48000.0, from, fade_ms);
      prepare_with(ref_from, 48000.0, from, fade_ms);
      prepare_with(ref_to, 48000.0, to, fade_ms);
      std::vector<float> a(in.size()), f(in.size()), t(in.size());
      size_t pa = 0, pf = 0, pt = 0, S = 0;
      const size_t swap_block = block_at(blocks, 70000, S);
      run_blocks(swapped, in, a, blocks, 0, swap_block, pa);
      CHECK(swapped.commit_swap(swapped.prepare_swap(model, to)));
      run_blocks(swapped, in, a, blocks, swap_block, blocks.size(), pa);
      run_blocks(ref_from, in, f, blocks, 0, blocks.size(), pf);
      run_blocks(ref_to, in, t, blocks, 0, blocks.size(), pt);
      CHECK(swapped.variant() == to && swapped.swaps_completed() == 1);

      const int fade_len = int(std::lround(fade_ms * 1e-3 * 48000.0));
      bool before = true, after = true;
      double worst_mix = 0.0;
      for (size_t i = 0; i < in.size(); ++i)
      {
        if (i < S)
          before = before && a[i] == f[i];
        else if (i < S + size_t(fade_len))
        {
          const float g = fade_gain(int(i - S), fade_len);
          worst_mix = std::max(worst_mix, std::fabs(double(a[i]) - double((1.0f - g) * f[i] + g * t[i])));
        }
        else
          after = after && a[i] == t[i];
      }
      // the step between neighbouring samples during the fade: no larger than the two signals have themselves
      double step_swapped = 0.0, step_signals = 0.0;
      for (size_t i = S + 1; i < S + size_t(std::max(fade_len, 1)) + 1 && i < in.size(); ++i)
      {
        step_swapped = std::max(step_swapped, std::fabs(double(a[i]) - double(a[i - 1])));
        step_signals = std::max({step_signals, std::fabs(double(f[i]) - double(f[i - 1])), std::fabs(double(t[i]) - double(t[i - 1]))});
      }
      std::printf("  %s to %s, fade %2.0f ms: before the swap identical to the first, after the fade identical to the second, fade equals the mix to %.1e; largest step %.4f (signals %.4f)\n",
                  from == Variant::Full ? "Full" : "Lite", to == Variant::Full ? "Full" : "Lite", fade_ms, worst_mix, step_swapped, step_signals);
      CHECK(before);
      CHECK(after);
      CHECK(worst_mix < 1e-6);
      if (fade_len > 0)
        CHECK(step_swapped <= 1.25 * step_signals);
    }
}

static void swap_with_a_resampler()
{
  std::printf("swap_with_a_resampler\n");
  auto model = Model::load(g_model);
  for (double rate : {44100.0, 96000.0})
  {
    const std::vector<float> in = make_signal(rate, 2.5);
    const std::vector<int> blocks = make_blocks(in.size(), 13);
    Processor swapped(model), ref_from(model), ref_to(model);
    prepare_with(swapped, rate, Variant::Full);
    prepare_with(ref_from, rate, Variant::Full);
    prepare_with(ref_to, rate, Variant::Lite);
    std::vector<float> a(in.size()), f(in.size()), t(in.size());
    size_t pa = 0, pf = 0, pt = 0, S = 0;
    const size_t swap_block = block_at(blocks, size_t(rate * 1.2), S);
    run_blocks(swapped, in, a, blocks, 0, swap_block, pa);
    CHECK(swapped.commit_swap(swapped.prepare_swap(model, Variant::Lite)));
    run_blocks(swapped, in, a, blocks, swap_block, blocks.size(), pa);
    run_blocks(ref_from, in, f, blocks, 0, blocks.size(), pf);
    run_blocks(ref_to, in, t, blocks, 0, blocks.size(), pt);
    // The swap happens at the model rate, so it reaches the output a little after host sample S, and the fade is smeared by the filter of
    // the second stage: before S the output is the first reference's, long after S the second's.
    const size_t settled = S + size_t(0.010 * rate) + size_t(swapped.latency_samples()) * 2 + 1500;
    bool before = true, after = true;
    for (size_t i = 0; i < in.size(); ++i)
    {
      if (i < S)
        before = before && a[i] == f[i];
      else if (i >= settled)
        after = after && a[i] == t[i];
    }
    std::printf("  %6.0f Hz: identical to the Full reference before the swap, identical to the Lite reference from %zu samples after it\n", rate, settled - S);
    CHECK(before);
    CHECK(after);
    CHECK(swapped.swaps_completed() == 1);
  }
}

static void swap_right_after_prepare_and_with_a_fallback()
{
  std::printf("swap_right_after_prepare_and_with_a_fallback\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 1.0);
  const std::vector<int> blocks = make_blocks(in.size(), 3);
  { // before any audio: the history is empty and the new engine starts from silence like the running one
    Processor swapped(model), ref(model);
    prepare_with(swapped, 48000.0, Variant::Full, 0.0);
    prepare_with(ref, 48000.0, Variant::Lite, 0.0);
    CHECK(swapped.commit_swap(swapped.prepare_swap(model, Variant::Lite)));
    std::vector<float> a(in.size()), t(in.size());
    size_t pa = 0, pt = 0;
    run_blocks(swapped, in, a, blocks, 0, blocks.size(), pa);
    run_blocks(ref, in, t, blocks, 0, blocks.size(), pt);
    CHECK(std::equal(a.begin(), a.end(), t.begin()));
  }
  { // a model that has only A2-Lite, asked for Full: the ticket uses Lite
    std::ifstream is(g_model);
    const json file = json::parse(is);
    const std::string text = file.at("config").at("submodels")[0].at("model").dump();
    auto lite_only = Model::load(text.data(), text.size());
    Processor p(model);
    prepare_with(p, 48000.0, Variant::Full, 0.0);
    CHECK(p.commit_swap(p.prepare_swap(lite_only, Variant::Full)));
    std::vector<float> a(in.size());
    size_t pa = 0;
    run_blocks(p, in, a, blocks, 0, blocks.size(), pa);
    CHECK(p.variant() == Variant::Lite && p.swaps_completed() == 1);
  }
}

static void stale_foreign_and_replaced_tickets()
{
  std::printf("stale_foreign_and_replaced_tickets\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 2.5);
  const std::vector<int> blocks = make_blocks(in.size(), 17, 64);
  Processor p(model), ref(model);
  prepare_with(p, 48000.0, Variant::Full);
  prepare_with(ref, 48000.0, Variant::Full);
  std::vector<float> a(in.size()), r(in.size());
  size_t pa = 0, pr = 0;
  run_blocks(p, in, a, blocks, 0, 400, pa);
  run_blocks(ref, in, r, blocks, 0, 400, pr);

  { // an empty ticket
    SwapTicket empty;
    CHECK(!empty && !p.commit_swap(std::move(empty)));
  }
  { // too old: the audio has moved on by more than 1024 samples since the ticket was made
    SwapTicket old = p.prepare_swap(model, Variant::Lite);
    run_blocks(p, in, a, blocks, 400, 460, pa);
    run_blocks(ref, in, r, blocks, 400, 460, pr);
    CHECK(!p.commit_swap(std::move(old)));
    CHECK(static_cast<bool>(old)); // not consumed
    CHECK(!p.swap_busy());
  }
  { // made for another processor
    Processor other(model);
    prepare_with(other, 48000.0, Variant::Full);
    SwapTicket foreign = other.prepare_swap(model, Variant::Lite);
    CHECK(!p.commit_swap(std::move(foreign)));
  }
  { // made before prepare() or reset()
    Processor q(model);
    prepare_with(q, 48000.0, Variant::Full);
    SwapTicket t1 = q.prepare_swap(model, Variant::Lite);
    prepare_with(q, 48000.0, Variant::Full);
    CHECK(!q.commit_swap(std::move(t1)));
    SwapTicket t2 = q.prepare_swap(model, Variant::Lite);
    q.reset();
    CHECK(!q.commit_swap(std::move(t2)));
  }
  // none of this changed the sound
  {
    run_blocks(p, in, a, blocks, 460, blocks.size(), pa);
    run_blocks(ref, in, r, blocks, 460, blocks.size(), pr);
    CHECK(std::equal(a.begin(), a.end(), r.begin()) && p.swaps_completed() == 0);
  }

  // two tickets committed before the audio runs: the second replaces the first, and the first is freed at once
  {
    Processor q(model);
    prepare_with(q, 48000.0, Variant::Full);
    std::weak_ptr<const Model> first;
    {
      auto m1 = Model::load(g_model);
      first = m1;
      CHECK(q.commit_swap(q.prepare_swap(m1, Variant::Lite)));
    }
    CHECK(!first.expired()); // the ticket in the processor holds it
    CHECK(q.commit_swap(q.prepare_swap(model, Variant::Lite)));
    CHECK(first.expired());  // replaced, freed by commit_swap
    std::vector<float> out(in.size());
    size_t pos = 0;
    run_blocks(q, in, out, blocks, 0, blocks.size(), pos);
    CHECK(q.swaps_completed() == 1 && q.swaps_dropped() == 0 && q.variant() == Variant::Lite);
  }
  // a ticket committed during a fade waits for the fade to end
  {
    Processor q(model), ref_full(model);
    prepare_with(q, 48000.0, Variant::Full);
    prepare_with(ref_full, 48000.0, Variant::Full);
    std::vector<float> out(in.size()), rf(in.size());
    size_t pos = 0, pr2 = 0;
    run_blocks(q, in, out, blocks, 0, 300, pos);
    CHECK(q.commit_swap(q.prepare_swap(model, Variant::Lite)));
    run_blocks(q, in, out, blocks, 300, 305, pos); // inside the 480-sample fade (blocks of 1 to 64 samples)
    CHECK(q.swap_busy() && q.swaps_completed() == 0);
    CHECK(q.commit_swap(q.prepare_swap(model, Variant::Full))); // waits
    run_blocks(q, in, out, blocks, 305, blocks.size(), pos);
    run_blocks(ref_full, in, rf, blocks, 0, blocks.size(), pr2);
    CHECK(!q.swap_busy() && q.swaps_completed() == 2 && q.variant() == Variant::Full);
    bool tail_identical = true;
    for (size_t i = in.size() - 20000; i < in.size(); ++i)
      tail_identical = tail_identical && out[i] == rf[i];
    CHECK(tail_identical); // Full, then Lite, then Full again: the end is the plain Full signal
  }
}

static void retired_engines_are_freed_by_service()
{
  std::printf("retired_engines_are_freed_by_service\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 4.0);
  const std::vector<int> blocks = make_blocks(in.size(), 23, 128);
  Processor p(model);
  prepare_with(p, 48000.0, Variant::Full);
  std::vector<float> out(in.size());
  size_t pos = 0, next_block = 0;
  auto run_some = [&](size_t count) {
    const size_t end = std::min(blocks.size(), next_block + count);
    run_blocks(p, in, out, blocks, next_block, end, pos);
    next_block = end;
  };
  run_some(200);

  std::weak_ptr<const Model> w1, w2;
  {
    auto m1 = Model::load(g_model);
    w1 = m1;
    CHECK(p.commit_swap(p.prepare_swap(m1, Variant::Lite)));
  }
  run_some(30); // m1 is now running
  CHECK(p.swaps_completed() == 1 && !w1.expired());
  {
    auto m2 = Model::load(g_model);
    w2 = m2;
    CHECK(p.commit_swap(p.prepare_swap(m2, Variant::Full)));
  }
  run_some(30); // m2 is running; the engine of m1 has been taken out of use but only service() may free it
  CHECK(p.swaps_completed() == 2);
  CHECK(!w1.expired()); // the audio thread freed nothing
  p.service();
  CHECK(w1.expired() && !w2.expired());
  std::printf("  the model of the retired engine stays alive until service() is called, then it is freed\n");

  // service() is never called: after 8 retired engines the queue is full and the next swap waits
  for (int i = 0; i < 8; ++i)
  {
    CHECK(p.commit_swap(p.prepare_swap(model, i % 2 == 0 ? Variant::Lite : Variant::Full)));
    run_some(25);
  }
  CHECK(p.swaps_completed() == 10 && !p.swap_busy()); // 2 before, 8 now: the queue holds 8
  CHECK(p.commit_swap(p.prepare_swap(model, Variant::Lite)));
  run_some(40);
  CHECK(p.swap_busy() && p.swaps_completed() == 10); // waiting: no room to retire the engine it would replace
  p.service();
  run_some(40);
  CHECK(!p.swap_busy() && p.swaps_completed() == 11);
  std::printf("  with the retire queue full a swap waits, and goes through after service()\n");
}

static void errors()
{
  std::printf("errors\n");
  auto model = Model::load(g_model);
  {
    Processor p(model);
    CHECK(error_of([&] { p.prepare_swap(model, Variant::Full); }) == ErrorCode::InvalidArgument); // before prepare()
    prepare_with(p, 48000.0, Variant::Full);
    CHECK(error_of([&] { p.prepare_swap(nullptr, Variant::Full); }) == ErrorCode::InvalidArgument);
    std::ifstream is(g_model);
    json file = json::parse(is);
    file["sample_rate"] = 44100.0;
    const std::string text = file.dump();
    auto other_rate = Model::load(text.data(), text.size());
    CHECK(error_of([&] { p.prepare_swap(other_rate, Variant::Full); }) == ErrorCode::Unsupported);
    try
    {
      p.prepare_swap(other_rate, Variant::Full);
    }
    catch (const Error& e)
    {
      std::printf("  example message: %s\n", e.what());
    }
  }
  {
    Processor p(model);
    ProcessorConfig c;
    c.swap_fade_ms = -1.0;
    CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument);
    c.swap_fade_ms = 100.5;
    CHECK(error_of([&] { p.prepare(c); }) == ErrorCode::InvalidArgument);
    c.swap_fade_ms = 100.0;
    CHECK(!error_of([&] { p.prepare(c); }));
  }
}

// The audio thread runs at `speed` times real time (it waits for each block's turn by spinning), so that the swap behaves as it would in a
// plug-in: preparing a swap takes some milliseconds while the audio goes on.
static void audio_and_control_threads_together()
{
  std::printf("audio_and_control_threads_together\n");
  auto model = Model::load(g_model);
  struct Scenario
  {
    double rate, speed;
  };
  for (Scenario sc : {Scenario{48000.0, 8.0}, Scenario{44100.0, 8.0}, Scenario{48000.0, 20.0}})
  {
    const std::vector<float> signal = make_signal(sc.rate, 6.0);
    Processor p(model);
    ProcessorConfig c;
    c.sample_rate = sc.rate;
    c.max_block = 128;
    c.monitoring = true;
    p.prepare(c);
    std::atomic<bool> audio_done{false};
    std::vector<float> out(signal.size());
    std::thread audio([&] {
      const auto begin = std::chrono::steady_clock::now();
      const int rounds = sc.speed > 10.0 ? 2 : 1;
      for (int round = 0; round < rounds; ++round)
        for (size_t pos = 0; pos + 64 <= signal.size(); pos += 64)
        {
          p.process(signal.data() + pos, out.data() + pos, 64);
          {
            const auto due = begin + std::chrono::nanoseconds(static_cast<long long>(double(pos + 64) / sc.rate * 1e9 / sc.speed));
            while (std::chrono::steady_clock::now() < due) {}
          }
        }
      audio_done.store(true);
    });
    int committed = 0, refused = 0;
    bool flip = false;
    while (!audio_done.load())
    {
      SwapTicket ticket = p.prepare_swap(model, flip ? Variant::Full : Variant::Lite);
      if (p.commit_swap(std::move(ticket)))
      {
        ++committed;
        flip = !flip;
      }
      else
        ++refused;
      p.service();
      std::this_thread::sleep_for(std::chrono::milliseconds(3));
    }
    audio.join();
    p.service();
    bool finite = true;
    double peak = 0.0;
    for (float v : out)
    {
      finite = finite && std::isfinite(v);
      peak = std::max(peak, double(std::fabs(v)));
    }
    std::printf("  %6.0f Hz, audio at %s: %d swaps committed (%d refused as out of date), %llu finished, %llu dropped, p99 load %.1f%%, max load %.1f%%; output finite, peak %.2f\n",
                sc.rate, sc.speed < 10.0 ? "8x real time" : "20x real time", committed, refused, static_cast<unsigned long long>(p.swaps_completed()),
                static_cast<unsigned long long>(p.swaps_dropped()), 100.0 * p.stats().p99_load, 100.0 * p.stats().max_load, peak);
    CHECK(finite && peak < 2.0);
    CHECK(committed >= (sc.speed < 10.0 ? 4 : 2));
    CHECK(refused <= 2 * committed + 2); // out-of-date tickets are refused cleanly; on a busy machine they are more frequent
    CHECK(p.swaps_completed() + p.swaps_dropped() <= static_cast<std::uint64_t>(committed));
    CHECK(p.swaps_completed() >= static_cast<std::uint64_t>(committed) / 2);
  }
}

// What an offline renderer does: it prepares and commits between two blocks on the thread that renders, at full speed. The ticket is never
// out of date, whatever the speed.
static void swapping_between_blocks_on_the_rendering_thread()
{
  std::printf("swapping_between_blocks_on_the_rendering_thread\n");
  auto model = Model::load(g_model);
  const std::vector<float> in = make_signal(48000.0, 6.0);
  Processor p(model);
  prepare_with(p, 48000.0, Variant::Full);
  std::vector<float> out(in.size());
  int swaps = 0;
  bool all_committed = true;
  Variant next = Variant::Lite;
  for (size_t pos = 0, block = 0; pos + 64 <= in.size(); pos += 64, ++block)
  {
    if (block % 40 == 20) // every 40 blocks, 8 of which are a 10 ms fade
    {
      all_committed = p.commit_swap(p.prepare_swap(model, next)) && all_committed;
      next = next == Variant::Lite ? Variant::Full : Variant::Lite;
      ++swaps;
    }
    p.process(in.data() + pos, out.data() + pos, 64);
    if (block % 40 == 39)
      p.service();
  }
  bool finite = true;
  for (float v : out)
    finite = finite && std::isfinite(v);
  std::printf("  %d swaps, all committed: %s, all finished: %s, output finite: %s\n", swaps, all_committed ? "yes" : "NO",
              p.swaps_completed() == static_cast<std::uint64_t>(swaps) ? "yes" : "NO", finite ? "yes" : "NO");
  CHECK(all_committed && finite);
  CHECK(p.swaps_completed() == static_cast<std::uint64_t>(swaps) && p.swaps_dropped() == 0);
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
  guarded("same_model_swap_changes_nothing", same_model_swap_changes_nothing);
  guarded("variant_swap_equals_the_references", variant_swap_equals_the_references);
  guarded("swap_with_a_resampler", swap_with_a_resampler);
  guarded("swap_right_after_prepare_and_with_a_fallback", swap_right_after_prepare_and_with_a_fallback);
  guarded("stale_foreign_and_replaced_tickets", stale_foreign_and_replaced_tickets);
  guarded("retired_engines_are_freed_by_service", retired_engines_are_freed_by_service);
  guarded("errors", errors);
  guarded("audio_and_control_threads_together", audio_and_control_threads_together);
  guarded("swapping_between_blocks_on_the_rendering_thread", swapping_between_blocks_on_the_rendering_thread);
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
