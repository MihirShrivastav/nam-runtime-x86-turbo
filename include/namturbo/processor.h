// A processor runs an A2 model on one mono audio stream.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>

#include "namturbo/model.h"

namespace namturbo
{
// How the signal is converted when the host rate differs from the rate of the model (normally 48 kHz). The attenuation is the
// suppression of images and aliases, and the pass band is the part of the lower Nyquist frequency that passes unchanged.
// Below 44.1 kHz the transition band is narrower in Hz, so the delay is longer (3.8 ms at 32 kHz, 5.6 ms at 22.05 kHz, for Standard).
enum class ResamplerQuality
{
  Standard,   // 96 dB attenuation, flat up to 90% of the lower Nyquist frequency; the delay is 2.6 to 2.8 ms at host rates of 44.1 kHz and above
  LowLatency  // 72 dB attenuation, flat up to 80% of the lower Nyquist frequency; the delay is 0.9 to 1.0 ms at host rates of 44.1 kHz and above
};

// Phase-interleaved oversampling around the A2 network to suppress nonlinear aliasing while preserving the model's in-band
// frequency response. Uses linear-phase half-band FIR filters configured by ProcessorConfig::resampler.
enum class Oversampling
{
  Off = 1, // 1x: no oversampling (default)
  X2 = 2,  // 2x: two interleaved phases (about 2x engine cost; 63 or 23 model-rate samples of latency)
  X4 = 4   // 4x: four interleaved phases (about 4x engine cost; 71 or 29 model-rate samples of latency)
};

// How long process() takes, measured by the processor itself since prepare() or the last reset_stats(). A "call" is one call of
// process(); its "block" is the n samples of that call, which last n / sample rate seconds. The load of a call is its time divided
// by the duration of its block: 1.0 means that it took all the time it had.
struct PerfStats
{
  std::uint64_t blocks = 0;        // calls measured
  double last_us = 0.0;            // the most recent call, in microseconds
  double mean_us = 0.0;            // the average call
  double max_us = 0.0;             // the slowest call
  double mean_load = 0.0;          // total processing time divided by the total duration of the audio processed
  double p99_load = 0.0;           // 99% of the calls had a load no larger than this (accurate to 5%)
  double max_load = 0.0;           // the largest load of a call
  std::uint64_t over_deadline = 0; // calls with a load above 1.0
};

// Switching between A2-Full and A2-Lite of the same model when the processor is too slow, and back when there is room again.
// A2-Lite sounds different from A2-Full (it is a smaller network), so this is off by default and is meant for machines that
// cannot keep up. The decision is made in Processor::service(), which a program must call every 10 to 50 ms; the switch itself is
// a swap (prepare_swap and commit_swap), so it is click-free.
struct AdaptiveConfig
{
  bool enabled = false;
  double high_load = 0.70;        // switch to A2-Lite when the smoothed load stays above this
  double low_load = 0.30;         // switch back to A2-Full when the load that A2-Full is expected to cause stays below this
  double smoothing_ms = 300.0;    // time constant of the load average
  double down_after_ms = 200.0;   // how long the load must stay above high_load before the switch down
  double up_after_ms = 4000.0;    // how long the expected load must stay below low_load before the switch up
  std::function<void(Variant)> on_switch; // called from service() (never from the audio thread) after a switch has been committed
};

struct ProcessorConfig
{
  double sample_rate = 48000.0;      // the sample rate of the host: a whole number of Hz
  int max_block = 1024;              // the largest block the host passes to process(), 1 to 65536
  Variant variant = Variant::Full;   // the variant to run; if the model does not contain it, the other one is used
  ResamplerQuality resampler = ResamplerQuality::Standard; // quality preset for sample-rate conversion and oversampling
  Oversampling oversampling = Oversampling::Off;           // optional 2x or 4x phase-interleaved oversampling around the network
  bool monitoring = false;           // measure the time of every process() call (two clock reads per call); see stats()
  double swap_fade_ms = 10.0;        // length of the cross-fade when the model or variant is swapped (0 to 100 ms; 0 switches at once)
  AdaptiveConfig adaptive;           // see AdaptiveConfig; can also be changed later with set_adaptive()
};

// A new engine that is ready to take over from the running one, made by Processor::prepare_swap() and handed to
// Processor::commit_swap(). Move-only; an unused ticket is simply destroyed.
class SwapTicket
{
public:
  SwapTicket();
  SwapTicket(SwapTicket&&) noexcept;
  SwapTicket& operator=(SwapTicket&&) noexcept;
  ~SwapTicket();
  SwapTicket(const SwapTicket&) = delete;
  SwapTicket& operator=(const SwapTicket&) = delete;

  // True if the ticket holds an engine (it is empty after it was committed or moved from).
  explicit operator bool() const noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
  friend class Processor;
};

// Threads. prepare() and reset() are for a control thread (they allocate and take a few milliseconds) and must not run at
// the same time as process(), as with the prepareToPlay() of a plug-in. process() is for the audio thread, one thread at a
// time for one processor: it never allocates, locks, frees memory, throws or blocks. prepare_swap(), commit_swap() and service() are for control
// threads (not the audio thread) and can run while process() runs; they must not run at the same time as prepare() or reset(), and
// prepare_swap(), commit_swap(), service() and set_adaptive() must not run at the same time as each other for one processor. The
// functions that only read a value can be called from any thread. A stereo signal is two processors, which can share one Model.
class Processor
{
public:
  // Throws Error(InvalidArgument) if the model is null.
  explicit Processor(std::shared_ptr<const Model> model);
  ~Processor();
  Processor(const Processor&) = delete;
  Processor& operator=(const Processor&) = delete;

  // Builds the engine for the configuration and warms it up, so that the first block is processed like any other. Can be called again
  // with another configuration; the state is lost, and so is a swap that was waiting. Throws Error: CpuUnsupported (no AVX2 or FMA),
  // InvalidArgument (a block size outside 1 to 65536, a sample rate outside 1 kHz to 2 MHz, or a fade length outside 0 to 100 ms) or Unsupported
  // (a sample rate that is not a whole number of Hz, or whose ratio to the rate of the model is too complex: the reduced ratio must have a
  // numerator and a denominator of at most 2000). If prepare() throws, the processor is left as it was.
  void prepare(const ProcessorConfig& config);

  // Clears the state and warms the engine up again. Does nothing before prepare(). A swap that was waiting is dropped.
  void reset();

  // Processes n samples. n can be any value from 0: a block longer than max_block is processed in pieces, and n == 0 does
  // nothing. in and out can be the same buffer, but must not overlap otherwise. Before prepare(), out is filled with zeros.
  // The result does not depend on how a signal is divided into blocks. The double version gives the same values as the float
  // version (it converts through float).
  void process(const float* in, float* out, int n) noexcept;
  void process(const double* in, double* out, int n) noexcept;

  // The delay, in samples of the host, that the processor adds to the signal. It is 0 if the host rate equals the rate of the model
  // and oversampling is Off. Otherwise it is the delay of sample-rate conversion and oversampling (exact integer when the host rate
  // is a multiple of the model rate or oversampling is Off; rounded to the nearest host sample otherwise). Fixed after prepare().
  int latency_samples() const noexcept;

  // The variant that is running now (the one that prepare() selected, or the one of the last swap that has finished its fade).
  Variant variant() const noexcept;

  // The information of the model that the processor was constructed with.
  const ModelInfo& model_info() const noexcept;

  // ---- changing the model or the variant while the audio runs ---------------------------------------------------------------

  // Builds an engine for `model` and `variant` (the other variant if the model has not got that one) and brings it to the state of the
  // running engine by replaying the recent input, which takes a few milliseconds. The audio thread is not disturbed. The model must have the
  // same sample rate as the model that the processor was constructed with. Throws Error: InvalidArgument (before prepare(), or a null
  // model) or Unsupported (another sample rate). Control thread.
  SwapTicket prepare_swap(std::shared_ptr<const Model> model, Variant variant);

  // Hands the ticket to the audio thread, which takes it at the start of its next block and cross-fades to it over
  // ProcessorConfig::swap_fade_ms. Returns false, and leaves the ticket usable only for a new prepare_swap(), if the ticket is out of date:
  // prepare() was called after it was made, or the audio has moved on by more than 1024 samples of the model rate (about 21 ms) since. A ticket that was
  // committed earlier and has not been taken yet is replaced. While a fade is running, a committed ticket waits until it has ended.
  // The audio thread runs a short catch-up when it takes the ticket (at most a few hundred samples of engine work). Preparing a ticket takes
  // a few milliseconds and replays input faster than the audio produces it, which works while the audio runs up to about 40 times real time. A host
  // that renders offline as fast as the engine can run has no such margin: it should prepare and commit between two blocks on the thread that
  // renders, where the ticket is never out of date.
  // Not for the audio thread (it can free the ticket it replaces).
  bool commit_swap(SwapTicket&& ticket);

  // Frees the engines that finished their fade and were taken out of use, and runs the adaptive policy (see AdaptiveConfig). The audio
  // thread never frees anything, so a program that swaps or uses the adaptive mode calls this every 10 to 50 ms; the destructor frees
  // what is left. Control thread.
  void service();

  // Turns the adaptive mode on or off, or changes its thresholds, while the audio runs. Turning it on measures how much more A2-Full
  // costs than A2-Lite (a few milliseconds on this thread). Does nothing useful if the model has only one variant. Throws Error:
  // InvalidArgument (before prepare(), or thresholds that are not 0 < low_load < high_load) . Control thread.
  void set_adaptive(const AdaptiveConfig& config);

  // The load of process(), smoothed with AdaptiveConfig::smoothing_ms; 0 unless the adaptive mode or monitoring is on. Any thread.
  double smoothed_load() const noexcept;

  // True while a committed swap has not finished: it is waiting for the audio thread, or it is being faded in.
  bool swap_busy() const noexcept;

  // How many swaps have finished their fade since prepare(), and how many tickets the audio thread had to drop because they were
  // out of date when it took them (a program can try again).
  std::uint64_t swaps_completed() const noexcept;
  std::uint64_t swaps_dropped() const noexcept;

  // ---- monitoring -----------------------------------------------------------------------------------------------------------

  // Statistics of process(), if ProcessorConfig::monitoring was set (all zero otherwise). Lock-free and allocation-free: stats() can be
  // called from any thread while process() runs, and then returns the values of a moment a few calls ago, which need not agree perfectly
  // with each other. The totals count from prepare() or the last reset_stats(); a program that wants "the last second" calls
  // reset_stats() every second.
  PerfStats stats() const noexcept;
  void reset_stats() noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};
} // namespace namturbo
