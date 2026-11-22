# namturbo

Runtime for Neural Amp Modeler (NAM) A2 models on x86-64 CPUs with AVX2 and FMA.

## Overview

namturbo executes NAM A2 models (A2-Full and A2-Lite) in real time. Inference is 2.4x to 7x faster than the NAM Core A2 fast path on the
tested CPUs (see [Results](#results)). The library is written in C++20, has no external dependencies at run time, and is built with CMake.

The interface consists of `Model` (immutable weights loaded from a `.nam` file) and `Processor` (state and engine for one mono stream).
`Processor::process` is real-time safe.

## Features

- A2-Full (8 channels) and A2-Lite (3 channels), from plain or slimmable model files.
- Float and double buffers. Block sizes from 1 sample. The output does not depend on the block partitioning.
- Built-in sample-rate conversion (polyphase FIR, two quality settings) for host rates other than the model rate. The latency is exact and
  reported in samples.
- Model and variant swap during processing, with a cross-fade. Preparation takes place on a control thread.
- Optional automatic switch between A2-Full and A2-Lite, driven by the measured load.
- Per-block timing statistics (mean, 99th percentile, maximum, missed deadlines), readable from any thread.
- No allocation, deallocation, locking or exceptions on the audio thread. The test suite checks allocation and deallocation, including during swaps.
- Typed errors for invalid files, non-A2 architectures and CPUs without AVX2 or FMA. The loader is tested with damaged and hostile input.

## Optimizations

- **A2-Full engine.** One time frame of 8 channels is one 256-bit register. Convolutions use fused multiply-add. Each loaded weight vector is
  applied to a tile of 6 frames.
- **A2-Lite engine.** A register holds 8 consecutive frames of one channel, so that all lanes are used with 3 channels. Convolution,
  activation, head accumulation and residual update are computed per tile, and intermediate values stay in registers.
- **History rings.** Per-layer power-of-two rings with a mirrored tail. Every tap reads a contiguous window; the inner loops contain no wrap
  branches.
- **Two-level accumulation.** The products of one tap are summed first and then added to the running sum. Cost: about 3% of the block time.
  Effect: the rounding error is 7 dB lower (see [Accuracy](#accuracy)).
- **Denormals.** FTZ and DAZ are set during `process` and the previous state is restored on return. The time per block does not depend on the signal.
- **Resampler.** Rational polyphase filters evaluate only the retained output samples, with AVX2 dot products. Both stages share one design
  grid, so that the total delay is an integer number of host samples.
- **Audio thread.** Swaps are prepared on a control thread and published with a single atomic exchange. Retired engines are freed by
  `Processor::service`.

See [docs/technique-notes.md](docs/technique-notes.md).

## Results

<!-- results:start -->

Speed-up over the NAM Core A2 fast path, 48 kHz, model `test-amp-and-cab.nam`, median of 5 runs. A speed-up of 3x means one third of the processing time. Further rates and details: [docs/RESULTS.md](docs/RESULTS.md).

**A2-Full** (buffer size in samples)

| Machine and compiler | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| AMD Ryzen 5 7535HS, clang-cl 19.1.5 | 2.8x | 2.5x | 2.4x | 2.4x |
| AMD Ryzen 5 7535HS, MSVC 19.44 | 3.5x | 3.1x | 3.0x | 3.1x |
| Intel Core Ultra 5 245K, MSVC 19.44 | 3.2x | 3.0x | 2.8x | 2.8x |

**A2-Lite** (buffer size in samples)

| Machine and compiler | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|
| AMD Ryzen 5 7535HS, clang-cl 19.1.5 | 4.6x | 5.1x | 5.4x | 5.2x |
| AMD Ryzen 5 7535HS, MSVC 19.44 | 5.6x | 6.4x | 6.9x | 7.1x |
| Intel Core Ultra 5 245K, MSVC 19.44 | 5.4x | 5.8x | 6.4x | 6.5x |

<!-- results:end -->

Two instances on one audio thread (A2-Full with A2-Full, and A2-Full with A2-Lite), paced to real time, buffer sizes 64, 128 and 256, 60 s
per setting: 0 of 157,500 blocks exceeded the deadline on each machine. The slowest block used 32% of the deadline on the AMD machine
and 13% on the Intel machine (A2-Full with A2-Full, buffer size 64).

Method, load at 44.1 and 96 kHz, the two-instance test and the second model: [docs/RESULTS.md](docs/RESULTS.md). The results apply to the machines
and compilers listed there.

## Quick start

Build and test:

    cmake -S . -B build
    cmake --build build --config Release
    ctest --test-dir build -C Release

CMake integration, as a subdirectory or with FetchContent:

    add_subdirectory(namturbo)
    target_link_libraries(my_plugin PRIVATE namturbo::namturbo)

or from an installed copy:

    cmake --install build --config Release --prefix <folder>

    find_package(namturbo CONFIG REQUIRED)
    target_link_libraries(my_plugin PRIVATE namturbo::namturbo)

Minimal program (from [examples/consumer/main.cpp](examples/consumer/main.cpp)):

```cpp
#include "namturbo/namturbo.h"

auto model = namturbo::Model::load("amp.nam");        // control thread

namturbo::Processor processor(model);                 // one processor per mono stream
namturbo::ProcessorConfig config;
config.sample_rate = 48000.0;                         // host sample rate
config.max_block = 256;                               // largest block passed to process()
processor.prepare(config);                            // control thread, before playback

processor.process(in, out, n);                        // audio thread, every block
```

`Model::load` and `Processor::prepare` throw `namturbo::Error`; `what()` is a complete sentence. `Processor::process` does not throw.

Threading, latency, swaps, monitoring and stereo: [docs/integration-guide.md](docs/integration-guide.md).

## Requirements

| | |
|---|---|
| Processor | x86-64 with AVX2 and FMA (Intel Haswell or newer, AMD Excavator or Zen and newer) |
| Compiler | C++20: MSVC (Visual Studio 2022), clang-cl, GCC 13 or Clang 18 |
| Build system | CMake 3.16 or newer |
| Models | NAM A2 models (A2-Full, A2-Lite), plain or slimmable. Other architectures are not supported. |

### Test results

| Platform | Status |
|---|---|
| Windows 11, AMD Ryzen 5 7535HS, MSVC 19.44 and clang-cl 19.1.5 | Tests and benchmarks passed |
| Windows 11, Intel Core Ultra 5 245K, MSVC 19.44 | Tests and benchmarks passed |
| GitHub Actions: Windows Server 2022 (MSVC, clang-cl) and Linux, Ubuntu 24.04 (GCC 13.3, Clang 18.1) | Build with warnings as errors and tests passed; no benchmarks |
| macOS (Intel) | Not tested |
| ARM | Not supported (AVX2 required) |

## Accuracy

Reference implementations: the NAM Core A2 fast path and a float64 implementation of the network. The error is the energy of the difference
divided by the energy of the signal, in dB.

- Six A2 captures and five guitar recordings, 60 combinations: error against the NAM Core A2 fast path of -103.5 dB or lower; typical values
  are around -125 dB. The largest errors occur on the high-gain capture.
- Against the float64 implementation, A2-Lite has the same error as the NAM Core A2 fast path; the error of A2-Full is about 2 dB higher.

See [docs/accuracy.md](docs/accuracy.md).

## Limitations

- A2 models only.
- x86-64 with AVX2 and FMA only.
- One processor handles one mono stream. A stereo signal uses two processors, which can share one `Model`.
- A2-Lite has the level of A2-Full but a different waveform: for the model of this repository the error of A2-Lite against A2-Full is -12 dB
  to -17 dB (see [docs/accuracy.md](docs/accuracy.md)). Automatic switching to A2-Lite is disabled by default.
- The sample-rate converter is linear-phase. Its delay is 2.6 ms to 2.8 ms (Standard) or about 1 ms (LowLatency) at host rates of 44.1 kHz and
  above. Oversampling around the network is not implemented.
- The network runs at the sample rate of the model (normally 48 kHz). Other host rates are converted before and after the network.
- Loudness and level metadata of a model file are reported in `ModelInfo` and are not applied to the signal.

## Documentation

| | |
|---|---|
| [docs/integration-guide.md](docs/integration-guide.md) | Use in a plug-in or another real-time program |
| [docs/api-reference.md](docs/api-reference.md) | Classes and functions, real-time and thread-safety rules |
| [docs/model-compatibility.md](docs/model-compatibility.md) | Accepted and rejected model files |
| [docs/accuracy.md](docs/accuracy.md) | Accuracy and method |
| [docs/RESULTS.md](docs/RESULTS.md) | Benchmark results and method |
| [docs/technique-notes.md](docs/technique-notes.md) | Engine implementation |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Build, test, change requirements |
| [CHANGELOG.md](CHANGELOG.md) | Version history |

## Acknowledgements

namturbo implements the A2 network and reads the model file format of [Neural Amp Modeler](https://github.com/sdatkinson/neural-amp-modeler),
created by Steven Atkinson. [NAM Core](https://github.com/sdatkinson/NeuralAmpModelerCore) is the reference implementation of the network and the
baseline for the accuracy tests and benchmarks of this project. Thanks to Steven Atkinson and to all contributors to Neural Amp Modeler and NAM Core
for the architecture, the open file format, and the reference code, and for the community of capture makers whose work these models come from.

## Licence

MIT, see [LICENSE](LICENSE). `tests/data/test-amp-and-cab.nam` was made by the author of this repository and has the same licence.
Third-party material: [NOTICE](NOTICE).
