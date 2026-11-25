# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
The version is defined in `include/namturbo/version.h`.

## [Unreleased]

## [1.2.0] - 2026-11-25

### Added

- Idle-silence fast path (`src/a2_avx2.cpp`, `src/a2_lite_avx2.cpp`): after 6347 consecutive exact `+0.0f` input samples, all-zero blocks
  bypass the 23-layer forward pass and emit the exact steady-state silence output with bit-identical wake-up.
- Pure, `noexcept` gain-staging helpers in `include/namturbo/model.h`: `db_to_linear`, `linear_to_db`, `input_calibration_db`,
  `output_normalization_db`, and `output_calibration_db`.
- `tests/test_idle_fastpath.cpp` verifying bit-exact silence/wake-up transitions, idle speedup, and early/late swaps.

### Changed

- Prepacked weights (`src/a2_packed.h`) are now built once in `Model::load` and shared read-only across all processors and 2x/4x
  oversampling phases, reducing weight memory by 4x to 8x in stereo/oversampled setups and improving 2x/4x cache hit rates.
- `Processor::prepare_swap` skips the redundant 6347-sample silence prewarm whenever sufficient input history is already recorded,
  roughly halving swap preparation time on the control thread.

## [1.1.0] - 2026-11-23

### Added

- Optional 2x and 4x phase-interleaved oversampling (`Oversampling::Off`, `Oversampling::X2`, `Oversampling::X4`) in `ProcessorConfig`,
  using linear-phase symmetric half-band AVX2 FIR filters with exact latency reporting and click-free swap support.
- `--oversampling 1|2|4` and `--resampler standard|low-latency` options in `nam_bench`.

### Changed

- Sample-rate converter (`src/resampler.cpp`) now uses a hybrid half-band + polyphase cascade for 4x host-to-model ratios (192 kHz),
  mirrored power-of-two input rings instead of per-block `memmove`, 8-wide phase padding in rational polyphase stages, and block `memcpy`
  in the output FIFO.

## [1.0.0] - 2026-11-22

First stable release. It has the content of the release candidate 1.0.0-rc.1; only the version number differs.

### Added

- Runtime for NAM A2 models (A2-Full with 8 channels and A2-Lite with 3 channels) for x86-64 processors with AVX2 and FMA.
- `Model`: loading from a file or from memory, with validation of the architecture, the weights and the file size.
- `Processor`: block processing with float and double buffers, any block size from 0 samples; the output does not depend on the block partitioning.
- Sample-rate conversion between the host rate and the model rate (`Standard` and `LowLatency` quality) with exact latency (`latency_samples()`).
- Model and variant swap during processing with a cross-fade (`prepare_swap`, `commit_swap`, `service`).
- Automatic switching between A2-Full and A2-Lite based on the measured load (`AdaptiveConfig`), disabled by default.
- Per-block timing statistics (`PerfStats`, `stats()`).
- `CpuUnsupported` error for processors without AVX2 or FMA (`check_cpu()`).
- CMake package (`find_package(namturbo CONFIG REQUIRED)`, target `namturbo::namturbo`); CMake options `NAMTURBO_BUILD_TESTS`, `NAMTURBO_INSTALL`,
  `NAMTURBO_WERROR`, `NAMTURBO_BASELINE`, `NAMTURBO_BASELINE_AVX2`, `NAMTURBO_PROFILE`.
- Test suite, benchmark program (`nam_bench`), accuracy tools (`accuracy`, `variant_compare`, `rate_check.py`), two-instance real-time test
  (`two_instance`), and examples (`consumer`, `realtime_host`).
- Documentation: README, integration guide, API reference, model compatibility, accuracy, results, technique notes, contributing guide, security policy.

### Notes

- Tests and benchmarks on Windows 11 with MSVC 19.44 and clang-cl 19.1.5 (AMD Ryzen 5 7535HS) and with MSVC 19.44 (Intel Core Ultra 5 245K).
- Continuous integration: Windows (MSVC, clang-cl) and Linux (Ubuntu 24.04, GCC 13.3, Clang 18.1) build with warnings as errors and pass the tests.
  Sanitizer builds (AddressSanitizer, UndefinedBehaviorSanitizer, ThreadSanitizer) pass the tests. Benchmarks were run on Windows only. macOS (Intel)
  has not been tested.

## [1.0.0-rc.1] - 2026-11-22

First release candidate. See 1.0.0 for the content.
