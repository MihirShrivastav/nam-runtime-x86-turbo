// Tests of the performance recorder (src/monitor.h). Usage: test_monitor
// Exit codes: 0 pass, 1 failure.
#include <atomic>
#include <cmath>
#include <cstdio>
#include <thread>

#include "monitor.h"

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

static bool near(double a, double b, double relative) { return std::fabs(a - b) <= relative * std::fabs(b); }

static void exact_statistics()
{
  std::printf("exact_statistics\n");
  Monitor m;
  CHECK(m.snapshot().blocks == 0 && m.snapshot().mean_load == 0.0 && m.snapshot().p99_load == 0.0);
  // blocks of 1 ms: 990 calls at load 0.1, 9 at 0.5 and one at 2.0 (the last)
  for (int i = 0; i < 990; ++i)
    m.record(100000, 1000000);
  for (int i = 0; i < 9; ++i)
    m.record(500000, 1000000);
  m.record(2000000, 1000000);
  const PerfStats s = m.snapshot();
  CHECK(s.blocks == 1000);
  CHECK(near(s.mean_load, (990 * 0.1 + 9 * 0.5 + 2.0) / 1000.0, 1e-9));
  CHECK(near(s.mean_us, (990 * 100.0 + 9 * 500.0 + 2000.0) / 1000.0, 1e-9));
  CHECK(s.last_us == 2000.0 && s.max_us == 2000.0);
  CHECK(near(s.max_load, 2.0, 1e-9));
  CHECK(s.over_deadline == 1);
  // 99% of the calls (990 of 1000) had load 0.1: the percentile is that value to within the 5% bucket width
  CHECK(s.p99_load >= 0.1 * 0.999 && s.p99_load <= 0.1 * 1.0501);
  std::printf("  mean load %.4f, p99 load %.4f (exact 0.1), max load %.2f, over the deadline %llu\n", s.mean_load, s.p99_load, s.max_load,
              static_cast<unsigned long long>(s.over_deadline));

  // with 2% of the calls at load 0.5, the 99th percentile is 0.5
  Monitor m2;
  for (int i = 0; i < 980; ++i)
    m2.record(100000, 1000000);
  for (int i = 0; i < 20; ++i)
    m2.record(500000, 1000000);
  const PerfStats s2 = m2.snapshot();
  CHECK(s2.p99_load >= 0.5 * 0.999 && s2.p99_load <= 0.5 * 1.0501);
  CHECK(s2.over_deadline == 0);

  // blocks of different lengths: the mean load is total time over total duration
  Monitor m3;
  m3.record(50000, 1000000);   // load 0.05 for 1 ms
  m3.record(1500000, 3000000); // load 0.5 for 3 ms
  CHECK(near(m3.snapshot().mean_load, 1550000.0 / 4000000.0, 1e-9));
  CHECK(m3.snapshot().over_deadline == 0);
  CHECK(m3.snapshot().mean_us == 775.0);
}

static void extremes_and_reset()
{
  std::printf("extremes_and_reset\n");
  Monitor m;
  m.record(0, 1000000);          // took no time
  m.record(5, 0);                // a block of no duration: no load, not counted as over
  m.record(1000000000000ULL, 1000000); // load 1e6
  const PerfStats s = m.snapshot();
  CHECK(s.blocks == 3);
  CHECK(s.max_load >= 1e5 && std::isfinite(s.max_load) && std::isfinite(s.p99_load) && std::isfinite(s.mean_load));
  CHECK(s.over_deadline == 2); // 5 ns against a zero deadline is above it, 1e12 ns above 1 ms
  m.reset();
  const PerfStats r = m.snapshot();
  CHECK(r.blocks == 0 && r.over_deadline == 0 && r.max_us == 0.0 && r.last_us == 0.0 && r.mean_load == 0.0 && r.p99_load == 0.0 && r.max_load == 0.0);
  m.record(200000, 1000000);
  CHECK(m.snapshot().blocks == 1 && near(m.snapshot().mean_load, 0.2, 1e-9));
}

static void concurrent_reader_and_writer()
{
  std::printf("concurrent_reader_and_writer\n");
  Monitor m;
  constexpr int kRecords = 2000000;
  std::atomic<bool> done{false};
  bool monotonic = true, finite = true;
  std::thread reader([&] {
    std::uint64_t previous = 0;
    while (!done.load())
    {
      const PerfStats s = m.snapshot();
      monotonic = monotonic && s.blocks >= previous;
      finite = finite && std::isfinite(s.mean_us) && std::isfinite(s.mean_load) && std::isfinite(s.p99_load) && s.mean_load >= 0.0;
      previous = s.blocks;
    }
  });
  for (int i = 0; i < kRecords; ++i)
    m.record(static_cast<std::uint64_t>(50000 + (i % 7) * 10000), 1000000);
  done.store(true);
  reader.join();
  const PerfStats s = m.snapshot();
  CHECK(monotonic && finite);
  CHECK(s.blocks == static_cast<std::uint64_t>(kRecords));
  std::printf("  %d records while another thread read the statistics: counts only grew, all values finite; final mean load %.4f\n", kRecords, s.mean_load);
}

int main()
{
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  exact_statistics();
  extremes_and_reset();
  concurrent_reader_and_writer();
  std::printf(g_failures ? "FAILED: %d check(s)\n" : "ALL PASSED\n", g_failures);
  return g_failures ? 1 : 0;
}
