# Integration guide

Use of namturbo in an audio plug-in or another real-time audio program: thread roles, call order, and model changes during playback.
Every function is specified in the [API reference](api-reference.md).

Reference programs: [examples/realtime_host/main.cpp](../examples/realtime_host/main.cpp) (threads, swap, monitoring; built and run by the
test suite) and [examples/consumer/main.cpp](../examples/consumer/main.cpp) (minimal).

## 1. Objects

| Object | Content | Created | Thread |
|---|---|---|---|
| `namturbo::Model` | Weights of one model file. Immutable. | Once per file, with `Model::load`. | Not the audio thread. |
| `namturbo::Processor` | State and engine for one mono audio stream. | Once per stream. | Created and prepared on a control thread; `process` on the audio thread. |

A `Model` can be shared by any number of processors and threads. It is held in a `std::shared_ptr<const Model>`; a processor keeps a
reference to its model.

## 2. Build integration

As a subdirectory (or with FetchContent):

    add_subdirectory(external/namturbo)
    target_link_libraries(my_plugin PRIVATE namturbo::namturbo)

or from an installed copy:

    cmake -S namturbo -B build-namturbo
    cmake --build build-namturbo --config Release
    cmake --install build-namturbo --config Release --prefix C:/libs/namturbo

    list(APPEND CMAKE_PREFIX_PATH C:/libs/namturbo)
    find_package(namturbo CONFIG REQUIRED)
    target_link_libraries(my_plugin PRIVATE namturbo::namturbo)

The library is static, requires C++20 (set by the CMake target) and has no other dependency. As a subdirectory, only the library is built
and the settings of the parent project are not modified. The CMake options are listed in [CONTRIBUTING.md](../CONTRIBUTING.md).

```cpp
#include "namturbo/namturbo.h"
```

## 3. CPU check

The engines require AVX2 and FMA. `namturbo::check_cpu()` reports whether the processor and the operating system provide them.
`Processor::prepare` throws the same message on failure; `check_cpu` allows the check before a model is loaded.

```cpp
const namturbo::CpuSupport cpu = namturbo::check_cpu();
if (!cpu.supported)
    show_message(cpu.detail);   // for example: "this processor, or its operating system, does not support AVX2 and FMA, which namturbo requires"
```

## 4. Loading a model

`Model::load` reads a file or a memory block and validates it. It allocates and performs file I/O and must not run on the audio thread.

```cpp
std::shared_ptr<const namturbo::Model> model;
try
{
    model = namturbo::Model::load(path);
}
catch (const namturbo::Error& e)
{
    show_message(e.what());     // a complete sentence, for example "this model is not an A2 model: ..."
}
```

`namturbo::Error::code()` returns one of `Io`, `InvalidFile`, `NotA2`, `UnsupportedShape`, `CpuUnsupported`, `InvalidArgument`,
`Unsupported`.

`model->info()` gives the training sample rate, the available variants and, if present in the file, the loudness and level values. See
[model compatibility](model-compatibility.md).

## 5. Creating a processor

```cpp
namturbo::Processor processor(model);

namturbo::ProcessorConfig config;
config.sample_rate = host_sample_rate;   // a whole number of Hz
config.max_block   = host_max_block;     // the largest block that the host will pass to process()
processor.prepare(config);
```

`prepare` builds the engine and warms it up, so that the first block is processed like any other. It allocates and takes a few
milliseconds. It belongs where a plug-in prepares to play (`prepareToPlay` in JUCE, `setupProcessing` in VST3), not in the audio callback.
It can be called again when the sample rate or the block size changes; the state is lost.

`prepare` and `reset` must not run while another thread is inside `process`.

| `ProcessorConfig` field | Meaning | Default |
|---|---|---|
| `sample_rate` | Host sample rate in Hz, a whole number from 1000 to 2,000,000. | 48000 |
| `max_block` | Largest block passed to `process`, 1 to 65536. | 1024 |
| `variant` | `Variant::Full` or `Variant::Lite`. If the model does not contain it, the other variant is used. | Full |
| `resampler` | `Standard` (about 2.7 ms delay) or `LowLatency` (about 1 ms, lower attenuation). Used only if the host rate differs from the model rate. | Standard |
| `monitoring` | Measure the time of every `process` call. | off |
| `swap_fade_ms` | Cross-fade length for swaps, 0 to 100 ms. | 10 |
| `adaptive` | Automatic variant switching. See section 9. | off |

`max_block` sizes the internal buffers. It is not a limit: a longer block is processed in pieces with the same result.

## 6. Processing audio

On the audio thread, for every block:

```cpp
processor.process(input, output, num_samples);
```

- `input` and `output` are `float*` or `double*`. The double overload converts through float and gives the same values as the float overload.
- In-place processing is allowed. Other overlap is not.
- `num_samples` can be any value from 0. A block larger than `max_block` is processed in pieces.
- The output does not depend on the partitioning of the signal into blocks.
- `process` does not allocate, free, lock, wait or throw. Before a successful `prepare` it writes zeros.
- One processor is used by one audio thread at a time.

### Latency

The latency is 0 if the host rate equals the model rate (normally 48 kHz). Otherwise `processor.latency_samples()` returns the exact delay of
the sample-rate conversion in host samples:

```cpp
setLatencySamples(processor.latency_samples());   // for example, in JUCE
```

| Host rate | `Standard` | `LowLatency` |
|---|---|---|
| 44.1 kHz | 123 samples (2.79 ms) | 45 samples (1.02 ms) |
| 88.2 kHz | 226 samples (2.56 ms) | 82 samples (0.93 ms) |
| 96 kHz | 246 samples (2.56 ms) | 90 samples (0.94 ms) |
| 192 kHz | 492 samples (2.56 ms) | 180 samples (0.94 ms) |

### Stereo

A processor handles one mono signal. A stereo signal uses two processors, which can share one `Model`.

### Reset

`processor.reset()` clears the state and warms the engine up again, for example after a transport jump. It takes a few milliseconds and must
not run at the same time as `process`.

## 7. Gain staging

The library does not change the signal level. A NAM model expects an input level close to the capture level. When present in the file,
`model->info()` provides `loudness_db`, `input_level_dbu` and `output_level_dbu`. Their use (output normalization, input gain) is left
to the plug-in.

## 8. Swapping models and variants

```cpp
// On a control thread (never the audio thread):
namturbo::SwapTicket ticket = processor.prepare_swap(new_model, namturbo::Variant::Full);
if (!processor.commit_swap(std::move(ticket)))
{
    // The ticket was out of date, because the audio ran on while it was prepared. Prepare a new one.
}

// Also on a control thread, regularly (every 10 to 50 ms, from a timer):
processor.service();
```

Sequence:

1. `prepare_swap` builds the new engine and runs it over the most recent input (about 0.13 s), so that its state equals the state of the running
   engine. It takes a few milliseconds and does not disturb the audio thread.
2. `commit_swap` passes the engine to the audio thread. The audio thread takes it at the start of its next block and cross-fades from the
   old engine to the new one over `swap_fade_ms`.
3. After the fade, the old engine is queued for deletion. `service()` deletes it. The audio thread never frees memory.

Rules:

- The new model must have the sample rate of the model that the processor was created with; otherwise `prepare_swap` throws `Error`
  with code `Unsupported`.
- `prepare_swap`, `commit_swap`, `service` and `set_adaptive` are control-thread functions. For one processor they must not run
  concurrently with each other, with `prepare` or with `reset`. They can run concurrently with `process`.
- `commit_swap` returns `false` if the audio advanced by more than about 21 ms while the ticket was prepared, or if `prepare` was called after
  the ticket was created. A new ticket must be prepared.
- A ticket committed during a fade waits until the fade ends. A committed ticket that has not been taken yet is replaced by a newer one.
- Without calls to `service()`, retired engines are kept until the processor is destroyed.
- `swap_busy()`, `swaps_completed()` and `swaps_dropped()` report the state of the swap machinery.
- Offline rendering faster than real time should prepare and commit between two blocks on the rendering thread. The ticket is then never out of date.

### Loading a new model during playback

1. Worker thread: `auto next = Model::load(path);`
2. Same worker thread: `prepare_swap(next, variant)` and `commit_swap(...)`.
3. Timer on a control thread: `service()`.

## 9. Adaptive quality

A2-Lite needs about one fifth of the processing time of A2-Full and has a different waveform. The adaptive mode starts with A2-Full and
switches to A2-Lite when the smoothed load stays above a threshold:

```cpp
namturbo::AdaptiveConfig adaptive;
adaptive.enabled        = true;
adaptive.high_load      = 0.70;    // go down when the smoothed load stays above 70% of the block time ...
adaptive.down_after_ms  = 200;     // ... for 200 ms
adaptive.low_load       = 0.30;    // go back up when A2-Full is expected to use less than 30% ...
adaptive.up_after_ms    = 4000;    // ... for 4 seconds
adaptive.on_switch      = [](namturbo::Variant v) { /* called from service(); update the display */ };

processor.set_adaptive(adaptive);      // or set config.adaptive before prepare()
```

The decision is made in `service()`, which must be called regularly. The switch is a normal swap. When the mode is enabled, the processor
measures the cost ratio of A2-Full to A2-Lite and uses it to decide whether a return to A2-Full is safe. A variant selected with `prepare_swap`
is never overridden. The mode is disabled by default. It has no effect on models that contain only one variant.

## 10. Performance monitoring

With `config.monitoring = true`, the statistics can be read from any thread:

```cpp
namturbo::PerfStats s = processor.stats();
// s.blocks, s.mean_us, s.max_us, s.mean_load, s.p99_load, s.max_load, s.over_deadline
```

The load of a block is its processing time divided by its duration; 1.0 means that the block used all of its time. `over_deadline` counts
blocks with a load above 1.0. `reset_stats()` starts a new measurement interval. Monitoring costs two clock reads per call (at most 0.2 us) and
does not allocate.

## 11. Plug-in outline

| Plug-in event | Action |
|---|---|
| Plug-in created | `check_cpu()`. One `Processor` per channel; a shared `std::shared_ptr<const Model>`. |
| Model file selected | Worker thread: `Model::load`. Audio stopped: create processors and `prepare`. Audio running: `prepare_swap` and `commit_swap`. |
| Host prepares to play (`prepareToPlay`) | `prepare` on each processor with host sample rate and largest block. Report `latency_samples()`. |
| Audio callback (`processBlock`) | Input gain, `process` per channel, output gain. |
| Timer, 20 to 50 ms, not on the audio thread | `service()` for each processor. |
| Host stops or resets | `reset()` on each processor, from the thread that prepared it. |
| Plug-in destroyed | Stop audio, destroy the processors, then release the model. |

## 12. Real-time safety checklist

- The audio callback calls only `process`. Loading, preparing, swapping and deleting happen elsewhere.
- `process` does not run concurrently with `prepare` or `reset` of the same processor.
- Each processor is used by one audio thread.
- `service()` is called periodically if swaps or the adaptive mode are used.
- The latency is reported to the host.
- Errors from `Model::load`, `prepare` and `prepare_swap` are caught.

`tests/test_realtime.cpp` replaces the global memory allocator and fails if `process` allocates or frees memory, in normal processing and
during swaps. Locks cannot be detected by this test; the code takes none.
