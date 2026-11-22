// Performance statistics of a processor, written by the audio thread and read by any thread. No locks, no allocation: every field is a
// relaxed atomic and only one thread writes. A snapshot read while the audio thread is writing can be a few calls out of date and
// not perfectly consistent between its fields; each field is itself always a value that was recorded. Internal.
#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>

#include "namturbo/processor.h"

namespace namturbo
{
class Monitor
{
public:
  // Records one call that took `elapsed_ns` for a block that lasts `deadline_ns`. Audio thread only.
  void record(std::uint64_t elapsed_ns, std::uint64_t deadline_ns) noexcept
  {
    constexpr auto relaxed = std::memory_order_relaxed;
    blocks_.store(blocks_.load(relaxed) + 1, relaxed);
    sum_ns_.store(sum_ns_.load(relaxed) + elapsed_ns, relaxed);
    sum_deadline_ns_.store(sum_deadline_ns_.load(relaxed) + deadline_ns, relaxed);
    last_ns_.store(elapsed_ns, relaxed);
    if (elapsed_ns > max_ns_.load(relaxed))
      max_ns_.store(elapsed_ns, relaxed);
    const double load = deadline_ns > 0 ? static_cast<double>(elapsed_ns) / static_cast<double>(deadline_ns) : 0.0;
    const std::uint64_t load_ppb = static_cast<std::uint64_t>(std::min(load, 1e9) * 1e9); // load in parts per billion
    if (load_ppb > max_load_ppb_.load(relaxed))
      max_load_ppb_.store(load_ppb, relaxed);
    if (elapsed_ns > deadline_ns)
      over_.store(over_.load(relaxed) + 1, relaxed);
    const int bucket = bucket_of(load);
    histogram_[bucket].store(histogram_[bucket].load(relaxed) + 1, relaxed);
  }

  PerfStats snapshot() const noexcept
  {
    constexpr auto relaxed = std::memory_order_relaxed;
    PerfStats s;
    s.blocks = blocks_.load(relaxed);
    if (s.blocks == 0)
      return s;
    s.last_us = static_cast<double>(last_ns_.load(relaxed)) * 1e-3;
    s.max_us = static_cast<double>(max_ns_.load(relaxed)) * 1e-3;
    s.mean_us = static_cast<double>(sum_ns_.load(relaxed)) * 1e-3 / static_cast<double>(s.blocks);
    const double deadline = static_cast<double>(sum_deadline_ns_.load(relaxed));
    s.mean_load = deadline > 0.0 ? static_cast<double>(sum_ns_.load(relaxed)) / deadline : 0.0;
    s.max_load = static_cast<double>(max_load_ppb_.load(relaxed)) * 1e-9;
    s.over_deadline = over_.load(relaxed);
    // 99th percentile of the load, from the histogram: the upper edge of the bucket in which the cumulative count reaches 99%.
    std::uint64_t total = 0;
    for (const auto& h : histogram_)
      total += h.load(relaxed);
    if (total > 0)
    {
      const std::uint64_t wanted = (total * 99 + 99) / 100;
      std::uint64_t cumulative = 0;
      for (int i = 0; i < kBuckets; ++i)
      {
        cumulative += histogram_[i].load(relaxed);
        if (cumulative >= wanted)
        {
          s.p99_load = std::min(kLowest * std::pow(kRatio, i + 1), s.max_load); // never above the largest value seen
          break;
        }
      }
    }
    return s;
  }

  // Any thread. A call that the audio thread is making at that moment may be partly kept.
  void reset() noexcept
  {
    constexpr auto relaxed = std::memory_order_relaxed;
    blocks_.store(0, relaxed);
    sum_ns_.store(0, relaxed);
    sum_deadline_ns_.store(0, relaxed);
    last_ns_.store(0, relaxed);
    max_ns_.store(0, relaxed);
    max_load_ppb_.store(0, relaxed);
    over_.store(0, relaxed);
    for (auto& h : histogram_)
      h.store(0, relaxed);
  }

private:
  // Load buckets: from 1e-4 upwards, each 5% wider than the one below; the last collects everything above.
  static constexpr double kLowest = 1e-4;
  static constexpr double kRatio = 1.05;
  static constexpr int kBuckets = 400; // reaches 1e-4 * 1.05^400 = 3e4

  static int bucket_of(double load) noexcept
  {
    if (!(load > kLowest))
      return 0;
    const int b = static_cast<int>(std::log(load / kLowest) / std::log(kRatio));
    return std::min(b, kBuckets - 1);
  }

  std::atomic<std::uint64_t> blocks_{0}, sum_ns_{0}, sum_deadline_ns_{0}, last_ns_{0}, max_ns_{0}, max_load_ppb_{0}, over_{0};
  std::atomic<std::uint32_t> histogram_[kBuckets] = {};
};
} // namespace namturbo
