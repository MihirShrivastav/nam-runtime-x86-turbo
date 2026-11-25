# API reference

All names are in the namespace `namturbo`. Include everything with `#include "namturbo/namturbo.h"`, or include a single header:

| Header | Contents |
|---|---|
| `namturbo/error.h` | `ErrorCode`, `Error` |
| `namturbo/cpu.h` | `CpuSupport`, `check_cpu` |
| `namturbo/model.h` | `Variant`, `ModelInfo`, `Model`, `db_to_linear`, `linear_to_db`, `input_calibration_db`, `output_normalization_db`, `output_calibration_db` |
| `namturbo/processor.h` | `ResamplerQuality`, `Oversampling`, `PerfStats`, `AdaptiveConfig`, `ProcessorConfig`, `SwapTicket`, `Processor` |
| `namturbo/version.h` | `Version`, `kVersion`, `kPreRelease`, `version`, `version_string` |

For an explanation of how to use these in a plug-in, read the [integration guide](integration-guide.md) first.

## Threads and real-time rules

Three kinds of thread are meant:

- The **audio thread** calls `Processor::process` and nothing else.
- A **control thread** is any other thread of the program that sets things up or changes them: it loads models, prepares processors, swaps
  models, and calls `service`. It may allocate memory and may take milliseconds.
- **Any thread** may call the functions that only read a value.

| Function | Thread | Allocates, locks or can throw? |
|---|---|---|
| `check_cpu`, `version`, `version_string` | any | does not throw |
| `Model::load` | control | allocates; throws `Error` |
| `Model::info`, `Model::has` | any | no |
| `Processor` constructor and destructor | control | allocates or frees |
| `Processor::prepare`, `reset` | control, not while `process` runs | allocates; `prepare` throws `Error` |
| **`Processor::process`** | **audio** | **never allocates, frees, locks, waits or throws** |
| `Processor::prepare_swap` | control | allocates; throws `Error` |
| `Processor::commit_swap` | control | may free a replaced ticket |
| `Processor::service` | control | frees engines |
| `Processor::set_adaptive` | control | allocates; throws `Error` |
| `latency_samples`, `variant`, `model_info`, `smoothed_load`, `swap_busy`, `swaps_completed`, `swaps_dropped`, `stats`, `reset_stats` | any | no (lock-free) |

Rules for one `Processor`:

- `prepare` and `reset` must not run at the same time as `process`, or as each other, or as `prepare_swap`, `commit_swap`, `service` or
  `set_adaptive`.
- `prepare_swap`, `commit_swap`, `service` and `set_adaptive` must not run at the same time as each other. They can run while `process`
  runs.
- `process` is called by one thread at a time.
- A stereo signal is two processors. They can share one `Model`.
- A `Model` is immutable after loading and can be used from any number of threads.

A test (`tests/test_realtime.cpp`) checks that `process` allocates and frees no memory, including while a swap is taken, faded and retired.

---

## Errors

### `enum class ErrorCode`

| Value | Meaning |
|---|---|
| `Io` | The file cannot be opened or read. |
| `InvalidFile` | The data is not valid JSON, or the model data is malformed (wrong length, a number that is not finite, a file larger than 64 MB, an empty file). |
| `NotA2` | A valid NAM file, but not an A2 model. |
| `UnsupportedShape` | An A2 model of a size that is not A2-Lite (3 channels) or A2-Full (8 channels). |
| `CpuUnsupported` | The processor or the operating system lacks AVX2 and FMA. |
| `InvalidArgument` | A parameter has a value that cannot be used. |
| `Unsupported` | A valid request that this version does not implement, for example a sample rate whose ratio to the model rate is too complex. |

### `class Error : public std::runtime_error`

`ErrorCode code() const noexcept` gives the code. `what()` is a complete sentence that a program can show to its user. Only functions
that set things up throw. Processing audio never throws.

---

## CPU check

### `struct CpuSupport`

| Member | Meaning |
|---|---|
| `bool supported` | `true` if the engines can run on this processor and operating system. |
| `std::string detail` | When not supported, a sentence that names the missing feature. |

### `CpuSupport check_cpu()`

Tests for AVX2 and FMA, and for operating-system support of the 256-bit registers. Does not throw. Safe to call on any x86-64 processor.

---

## Model

### `enum class Variant { Lite, Full }`

A2-Lite has 3 channels and A2-Full has 8. A slimmable model file contains both. A plain file contains one.

### `struct ModelInfo`

| Member | Meaning |
|---|---|
| `double sample_rate` | The rate the model was trained at. 48000 if the file does not state one. |
| `bool sample_rate_stated` | `false` if the file has no sample rate. |
| `std::optional<double> loudness_db` | From the metadata of the file, if present. |
| `std::optional<double> input_level_dbu` | Idem. |
| `std::optional<double> output_level_dbu` | Idem. |
| `bool has_lite`, `bool has_full` | Which variants the file contains. |

The library reports these values and does not apply them.

### `class Model`

Immutable after loading. All member functions can be called from any thread at the same time. Not copyable. Created only by `load`.

| Function | Description |
|---|---|
| `static std::shared_ptr<const Model> load(const std::filesystem::path& file)` | Reads a `.nam` file. Throws `Error`: `Io`, `InvalidFile`, `NotA2`, `UnsupportedShape`. |
| `static std::shared_ptr<const Model> load(const void* data, std::size_t size)` | The same for a file that is already in memory. A null pointer with a non-zero size throws `InvalidArgument`. |
| `const ModelInfo& info() const noexcept` | Information about the model. |
| `bool has(Variant variant) const noexcept` | Whether the model contains the variant. |

Both `load` functions check the file completely. They reject files larger than 64 MB, data that is not valid JSON, models of another
architecture, wrong weight counts, and weights that are not finite numbers (including numbers too large for a 32-bit float). They
never crash on damaged input; this is tested with generated damaged files (`tests/test_fuzz_loader.cpp`).

### Gain-staging helpers

Pure, allocation-free, `noexcept` functions in `namturbo/model.h`, safe to call from any thread:

| Function | Meaning |
|---|---|
| `double db_to_linear(double db) noexcept` | `10^(db/20)`, or `0` if `db` is not finite. |
| `double linear_to_db(double linear) noexcept` | `20*log10(linear)`, or `-infinity` if `linear <= 0` or not finite. |
| `std::optional<double> input_calibration_db(const ModelInfo& info, double interface_input_dbu) noexcept` | `interface_input_dbu - *info.input_level_dbu`, or empty if absent or non-finite. |
| `std::optional<double> output_normalization_db(const ModelInfo& info, double target_loudness_db = -18.0) noexcept` | `target_loudness_db - *info.loudness_db`, or empty if absent or non-finite. |
| `std::optional<double> output_calibration_db(const ModelInfo& info, double interface_output_dbu) noexcept` | `*info.output_level_dbu - interface_output_dbu`, or empty if absent or non-finite. |

---

## Processor

### `enum class ResamplerQuality { Standard, LowLatency }`

How the signal is converted when the host rate differs from the rate of the model.

| Value | Attenuation of images and aliases | Flat up to | Delay at host rates of 44.1 kHz and above |
|---|---|---|---|
| `Standard` | 96 dB | 90% of the lower Nyquist frequency | 2.6 to 2.8 ms |
| `LowLatency` | 72 dB | 80% of the lower Nyquist frequency | 0.9 to 1.0 ms |

Below 44.1 kHz the delay is longer in milliseconds (3.8 ms at 32 kHz and 5.6 ms at 22.05 kHz for `Standard`).

### `enum class Oversampling`

Optional phase-interleaved oversampling around the A2 network, using linear-phase half-band FIR filters with the quality preset of `ProcessorConfig::resampler`.

| Value | Factor | Engine cost | Latency at 48 kHz (`Standard` / `LowLatency`) |
|---|---|---|---|
| `Off` | 1x | 1x | 0 / 0 samples |
| `X2` | 2x | ~2x | 63 / 23 samples (1.31 / 0.48 ms) |
| `X4` | 4x | ~4x | 71 / 29 samples (1.48 / 0.60 ms) |

### `struct PerfStats`

The time that `process` takes, measured by the processor itself since `prepare` or the last `reset_stats`. A "call" is one call of
`process`. The load of a call is its time divided by the duration of its block (the number of samples divided by the sample rate).

| Member | Meaning |
|---|---|
| `std::uint64_t blocks` | Calls measured. |
| `double last_us` | The most recent call, in microseconds. |
| `double mean_us` | The average call. |
| `double max_us` | The slowest call. |
| `double mean_load` | Total processing time divided by the total duration of the audio processed. |
| `double p99_load` | 99% of the calls had a load no larger than this (accurate to 5%). |
| `double max_load` | The largest load of a call. |
| `std::uint64_t over_deadline` | Calls with a load above 1.0. |

### `struct AdaptiveConfig`

Automatic switching between A2-Full and A2-Lite of the same model. See [integration guide, section 9](integration-guide.md#9-adaptive-quality).

| Member | Default | Meaning |
|---|---|---|
| `bool enabled` | `false` | Turns the mode on. |
| `double high_load` | 0.70 | Switch to A2-Lite when the smoothed load stays above this. |
| `double low_load` | 0.30 | Switch back to A2-Full when the load that A2-Full is expected to cause stays below this. |
| `double smoothing_ms` | 300 | Time constant of the load average. |
| `double down_after_ms` | 200 | How long the load must stay above `high_load` before the switch down. |
| `double up_after_ms` | 4000 | How long the expected load must stay below `low_load` before the switch up. |
| `std::function<void(Variant)> on_switch` | empty | Called from `service()` (never from the audio thread) after a switch was committed. |

The thresholds must satisfy `0 < low_load < high_load`, `smoothing_ms > 0`, and the two delays must not be negative; otherwise
`prepare` or `set_adaptive` throws `InvalidArgument`.

### `struct ProcessorConfig`

| Member | Default | Meaning |
|---|---|---|
| `double sample_rate` | 48000 | Sample rate of the host: a whole number of Hz, 1000 to 2,000,000. |
| `int max_block` | 1024 | Largest block passed to `process`, 1 to 65536. |
| `Variant variant` | `Full` | The variant to run. If the model does not contain it, the other one is used. |
| `ResamplerQuality resampler` | `Standard` | Quality preset for sample-rate conversion and oversampling. |
| `Oversampling oversampling` | `Off` | Optional 2x or 4x phase-interleaved oversampling around the network. |
| `bool monitoring` | `false` | Measure the time of every `process` call (two clock reads per call). |
| `double swap_fade_ms` | 10 | Length of the cross-fade when the model or variant is swapped, 0 to 100 ms. 0 switches at once. |
| `AdaptiveConfig adaptive` | off | See above. Can also be changed later with `set_adaptive`. |

### `class SwapTicket`

A new engine that is ready to take over from the running one. Made by `Processor::prepare_swap` and handed to `Processor::commit_swap`.
Move-only. An unused ticket is destroyed normally. `explicit operator bool() const noexcept` is `true` if the ticket holds an engine (it
is empty after it was committed or moved from).

### `class Processor`

Runs one model on one mono audio stream. Not copyable.

#### Construction and preparation

| Function | Description |
|---|---|
| `explicit Processor(std::shared_ptr<const Model> model)` | Throws `Error(InvalidArgument)` if the model is null. The processor cannot process before `prepare`. |
| `void prepare(const ProcessorConfig& config)` | Builds the engine for the configuration and warms it up, so that the first block is processed like any other. Can be called again with another configuration; the state is lost, and so is a swap that was waiting. Throws `Error`: `CpuUnsupported` (no AVX2 or FMA); `InvalidArgument` (a block size outside 1 to 65536, a sample rate outside 1 kHz to 2 MHz, a fade length outside 0 to 100 ms, or bad adaptive thresholds); `Unsupported` (a sample rate that is not a whole number of Hz, or whose ratio to the rate of the model is too complex: the reduced ratio must have a numerator and a denominator of at most 2000). If `prepare` throws, the processor is left as it was. |
| `void reset()` | Clears the state and warms the engine up again. Does nothing before `prepare`. A swap that was waiting is dropped. |

#### Processing (audio thread)

| Function | Description |
|---|---|
| `void process(const float* in, float* out, int n) noexcept` | Processes `n` samples. `n` can be any value from 0; a block longer than `max_block` is processed in pieces. `in` and `out` can be the same buffer, but must not overlap otherwise. Before `prepare`, `out` is filled with zeros. The result does not depend on how a signal is divided into blocks. |
| `void process(const double* in, double* out, int n) noexcept` | The same for double buffers. It gives the same values as the float version, because it converts through float. |

#### Information (any thread)

| Function | Description |
|---|---|
| `int latency_samples() const noexcept` | The delay, in samples of the host, that the processor adds to the signal. 0 if the host rate equals the rate of the model. Otherwise the exact delay of the sample-rate conversion, a whole number of host samples. Fixed after `prepare`. |
| `Variant variant() const noexcept` | The variant that is running now: the one that `prepare` selected, or the one of the last swap that finished its fade. |
| `const ModelInfo& model_info() const noexcept` | The information of the model that the processor was constructed with. |
| `double smoothed_load() const noexcept` | The load of `process`, smoothed with `AdaptiveConfig::smoothing_ms`. 0 unless the adaptive mode or monitoring is on. |
| `bool swap_busy() const noexcept` | `true` while a committed swap has not finished: it is waiting for the audio thread, or it is being faded in. |
| `std::uint64_t swaps_completed() const noexcept` | How many swaps have finished their fade since `prepare`. |
| `std::uint64_t swaps_dropped() const noexcept` | How many tickets the audio thread had to drop because they were out of date when it took them. A program can try again. |

#### Changing the model or the variant while audio runs (control threads)

| Function | Description |
|---|---|
| `SwapTicket prepare_swap(std::shared_ptr<const Model> model, Variant variant)` | Builds an engine for `model` and `variant` (the other variant if the model does not have that one) and brings it to the state of the running engine by replaying the recent input. This takes a few milliseconds and does not disturb the audio thread. The model must have the same sample rate as the model that the processor was constructed with. Throws `Error`: `InvalidArgument` (before `prepare`, or a null model) or `Unsupported` (another sample rate). |
| `bool commit_swap(SwapTicket&& ticket)` | Hands the ticket to the audio thread, which takes it at the start of its next block and cross-fades to it over `swap_fade_ms`. Returns `false` if the ticket is out of date: `prepare` was called after it was made, or the audio has moved on by more than 1024 samples of the model rate (about 21 ms) since. A ticket that was committed earlier and has not been taken yet is replaced. While a fade is running, a committed ticket waits until it has ended. Not for the audio thread (it can free the ticket that it replaces). |
| `void service()` | Frees the engines that finished their fade and were taken out of use, and runs the adaptive policy. The audio thread never frees anything, so a program that swaps or uses the adaptive mode calls this every 10 to 50 ms. The destructor frees what is left. |
| `void set_adaptive(const AdaptiveConfig& config)` | Turns the adaptive mode on or off, or changes its thresholds, while the audio runs. Turning it on measures how much more A2-Full costs than A2-Lite (a few milliseconds on this thread). Does nothing useful if the model has only one variant. Throws `Error(InvalidArgument)` before `prepare` or for thresholds that are not `0 < low_load < high_load`. |

A swap that the audio thread takes runs a short catch-up first (at most a few hundred samples of engine work), then runs both engines
during the fade. The largest block load seen in the stress test was about a third of the block time.

#### Monitoring (any thread)

| Function | Description |
|---|---|
| `PerfStats stats() const noexcept` | Statistics of `process`, if `ProcessorConfig::monitoring` was set (all zero otherwise). Lock-free and allocation-free: it can be called from any thread while `process` runs, and then returns the values of a moment a few calls ago, which need not agree perfectly with each other. The totals count from `prepare` or the last `reset_stats`. |
| `void reset_stats() noexcept` | Starts a new measurement. |

---

## Version

### `struct Version { int major, minor, patch; }`

`constexpr Version kVersion` and `constexpr Version version() noexcept` give the version of the library. `const char* version_string() noexcept`
gives it as `"major.minor.patch"`. The version follows semantic versioning; before 1.0.0 the interface can change between minor
versions, and every such change is listed in [CHANGELOG.md](../CHANGELOG.md).
