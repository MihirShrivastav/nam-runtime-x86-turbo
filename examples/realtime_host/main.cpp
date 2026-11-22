// A program shaped like an audio plug-in host, to show which thread does what.
//
//   main thread     loads the model, prepares the processor, and later swaps the variant and calls service()
//   audio thread    calls process() for a block every block-length of real time
//
// The audio thread here is a loop that waits for the time of each block. In a plug-in the host's audio callback does that.
//
// Usage: realtime_host <model.nam> [seconds, default 4]
// Exit code: 0 if everything worked, 1 if a swap never finished or the output was not finite, 2 on a usage or setup error.
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <thread>
#include <vector>

#include "namturbo/namturbo.h"

int main(int argc, char** argv)
{
  if (argc < 2)
  {
    std::fprintf(stderr, "usage: realtime_host <model.nam> [seconds]\n");
    return 2;
  }
  const double seconds = argc > 2 ? std::atof(argv[2]) : 4.0;
  constexpr double kRate = 44100.0; // a host rate that is not the rate of the model, so that the conversion is used
  constexpr int kBlock = 128;

  try
  {
    // 1. Check the processor before anything else, so that the user can be told.
    const namturbo::CpuSupport cpu = namturbo::check_cpu();
    if (!cpu.supported)
    {
      std::fprintf(stderr, "%s\n", cpu.detail.c_str());
      return 2;
    }

    // 2. Load the model. This reads a file and allocates memory: never do it on the audio thread.
    std::shared_ptr<const namturbo::Model> model = namturbo::Model::load(argv[1]);

    // 3. Make the processor and prepare it. This also allocates memory, and it takes a few milliseconds: do it where a plug-in
    //    prepares to play, not in the audio callback.
    namturbo::Processor processor(model);
    namturbo::ProcessorConfig config;
    config.sample_rate = kRate;
    config.max_block = kBlock;
    config.monitoring = true; // measure the time of every process() call
    processor.prepare(config);
    std::printf("host %.0f Hz, block %d samples, latency %d samples (%.2f ms)\n", kRate, kBlock, processor.latency_samples(),
                1000.0 * processor.latency_samples() / kRate);

    // 4. The audio thread. process() never allocates, locks or throws.
    std::atomic<bool> stop{false};
    std::atomic<bool> finite{true};
    std::thread audio([&] {
      std::vector<float> in(kBlock), out(kBlock);
      const auto block_time = std::chrono::duration<double>(kBlock / kRate);
      auto next = std::chrono::steady_clock::now();
      long long position = 0;
      while (!stop.load(std::memory_order_relaxed))
      {
        for (int i = 0; i < kBlock; ++i)
          in[i] = static_cast<float>(0.2 * std::sin(6.283185307179586 * 196.0 * double(position + i) / kRate));
        position += kBlock;
        processor.process(in.data(), out.data(), kBlock);
        for (float v : out)
          if (!std::isfinite(v))
            finite.store(false);
        next += std::chrono::duration_cast<std::chrono::steady_clock::duration>(block_time);
        std::this_thread::sleep_until(next);
      }
    });

    // 5. The control thread (this one) changes the variant every 400 ms while the audio runs.
    //    prepare_swap() builds the new engine and brings it up to date; commit_swap() hands it to the audio thread, which fades to it.
    //    service() frees the engines that are no longer used. Call it regularly: every 10 to 50 ms is enough.
    const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(seconds);
    int committed = 0, refused = 0;
    bool want_lite = true;
    auto next_swap = std::chrono::steady_clock::now() + std::chrono::milliseconds(400);
    while (std::chrono::steady_clock::now() < end)
    {
      if (std::chrono::steady_clock::now() >= next_swap && model->has(namturbo::Variant::Lite) && model->has(namturbo::Variant::Full))
      {
        namturbo::SwapTicket ticket = processor.prepare_swap(model, want_lite ? namturbo::Variant::Lite : namturbo::Variant::Full);
        if (processor.commit_swap(std::move(ticket)))
        {
          ++committed;
          want_lite = !want_lite;
        }
        else
          ++refused; // the ticket was out of date (the audio ran on while it was prepared): prepare a new one next time
        next_swap += std::chrono::milliseconds(400);
      }
      processor.service();
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    stop = true;
    audio.join();
    processor.service();

    // 6. Read the statistics (any thread can do this, also while audio runs).
    const namturbo::PerfStats stats = processor.stats();
    std::printf("%llu blocks: mean %.1f us, p99 load %.2f%%, max load %.2f%%, blocks over the deadline %llu\n",
                static_cast<unsigned long long>(stats.blocks), stats.mean_us, 100.0 * stats.p99_load, 100.0 * stats.max_load,
                static_cast<unsigned long long>(stats.over_deadline));
    std::printf("swaps: %d committed, %d refused, %llu finished, %llu dropped; now running %s\n", committed, refused,
                static_cast<unsigned long long>(processor.swaps_completed()), static_cast<unsigned long long>(processor.swaps_dropped()),
                processor.variant() == namturbo::Variant::Full ? "A2-Full" : "A2-Lite");

    const bool both = model->has(namturbo::Variant::Lite) && model->has(namturbo::Variant::Full);
    if (!finite.load())
    {
      std::fprintf(stderr, "the output was not finite\n");
      return 1;
    }
    if (both && (committed < 3 || processor.swaps_completed() < 3))
    {
      std::fprintf(stderr, "too few swaps finished\n");
      return 1;
    }
    return 0;
  }
  catch (const namturbo::Error& e)
  {
    std::fprintf(stderr, "namturbo error: %s\n", e.what());
    return 2;
  }
}
