# Contributing

Build and test instructions, code layout, and the requirements for changes.

## Build and test

Required: CMake 3.16 or newer, a C++20 compiler (MSVC from Visual Studio 2022, or clang-cl) and a processor with AVX2 and FMA.

    cmake -S . -B build
    cmake --build build --config Release
    ctest --test-dir build -C Release

The tests need no other library and no Python. They use the files in `tests/data/`.

### CMake options

| Option | Default | Meaning |
|---|---|---|
| `NAMTURBO_BUILD_TESTS` | on if namturbo is the main project | Build the tests, tools, benchmarks and examples. |
| `NAMTURBO_INSTALL` | on if namturbo is the main project | Generate the install rules. |
| `NAMTURBO_WERROR` | off | Treat warnings as errors in every target of the project. The project must build without warnings. Use it before submitting a change. |
| `NAMTURBO_BASELINE` | off | Build the comparison with NAM Core: baseline columns in the benchmarks, the accuracy tool, the precision tests, and the tools that make the reference outputs. Needs the NAM Core submodule and Python 3 with NumPy. |
| `NAMTURBO_BASELINE_AVX2` | on | Build NAM Core (only the baseline) with AVX2 and FMA code generation. |
| `NAMTURBO_PROFILE` | off | Per-section cycle counters in the engines (the `bench` program prints them). |
| `NAMTURBO_TEST_MODEL`, `NAMTURBO_TEST_DATA` | files in `tests/data` | The model and the data folder that the tests use. |

### Baseline build

    git submodule update --init --recursive
    cmake -S . -B build-baseline -DNAMTURBO_BASELINE=ON -DNAMTURBO_WERROR=ON
    cmake --build build-baseline --config Release
    ctest --test-dir build-baseline -C Release

This adds the tests that compare against NAM Core (`precision_vs_nam_full`, `precision_vs_nam_lite`, `golden_selfcheck`) and the benchmarks
with a baseline column. The pinned NAM Core commit is the one that the submodule points to.

## Layout

| Folder | Contents |
|---|---|
| `include/namturbo/` | The public headers. This is the interface of the library. |
| `src/` | The implementation. `a2_avx2.cpp` and `a2_lite_avx2.cpp` are the engines, `resampler.cpp` the sample-rate conversion, `processor.cpp` the processor with swapping and the adaptive mode, `a2_weights.cpp` the model loader. |
| `tests/` | The tests. `tests/data/` has the models, the input signals and the reference outputs; its README explains how they were made. |
| `tools/`, `benchmarks/` | Small programs: parity checks, `process_file` (run a recording through a processor), `nam_bench` (the benchmark), `accuracy` and `variant_compare` (error measurements), `two_instance` (two processors at once, in real time), and report scripts (`make_report.py`, `run_report.py`, `rate_check.py`). |
| `examples/` | `consumer` (the smallest program) and `realtime_host` (which thread does what). |
| `docs/` | The documentation. `docs/results/` holds the benchmark data. |
| `third_party/` | The JSON library (nlohmann/json), single header, MIT licence. |
| `external/` | The NAM Core submodule, for the baseline only. |

## Requirements for changes

1. **The library has no dependency on NAM Core.** NAM Core may be used by the baseline, the benchmarks and the tools, never by `src/`
   or `include/`.
2. **The audio thread does not allocate, free, lock, wait or throw.** `Processor::process` and everything it calls must keep to this.
   `tests/test_realtime.cpp` checks allocation and freeing, including during swaps. Locks cannot be detected by a test and are checked by
   review.
3. **The output does not depend on how the signal is cut into blocks.** The arithmetic for one output sample must not depend on its
   position in a block or a tile. The tests compare block divisions bit by bit.
4. **Accuracy.** An engine change must keep the output close to the NAM Core fast path (the parity tests) and must not make it less precise
   against the float64 reference (`precision_vs_nam_*`). Judge precision on real, high-gain material as well as on the test signals: a
   high-gain model amplifies rounding differences, and the generated test signals did not show a loss of 7 dB that real material showed.
5. **Speed.** A change in `src/a2_avx2.cpp`, `src/a2_lite_avx2.cpp` or `src/resampler.cpp` needs before and after timings (`nam_bench`), on a quiet
   machine with the thread pinned. State the processor and the compiler.
6. **Warnings.** The project builds without warnings with `NAMTURBO_WERROR=ON` on MSVC and clang-cl.
7. **Tests.** A new behaviour has a test. A fixed defect has a test that fails without the fix. Tests must not depend on timing in a way
   that fails on a busy machine; the tests that run threads or deadlines run serially (`RUN_SERIAL`).
8. **Platform.** The library is tested on Windows with MSVC and clang-cl. Other platforms have not been tested. A change that adds support
   for another platform needs the test results of that platform.

## Style

The existing code is the guide: 2 spaces, braces on their own lines, `snake_case` for functions and variables, `PascalCase` for types,
`kName` for constants, and comments that say why (the arithmetic and the reasoning), not what the next line does. Public headers need
documentation of every function: what it does, which thread may call it, what it throws.

Public headers must compile on their own with only `include/` on the include path; the build checks this.

## Test data

`tests/data/README.md` describes every file. The reference outputs (`golden_*.raw`) belong to the input files and to
`a2-example.nam`. To regenerate them, build with `NAMTURBO_BASELINE=ON` and build the target `regenerate_test_data`. Regenerate the inputs
and all four reference outputs together; the random generator of NumPy is not guaranteed to give the same numbers in every version.

## Releases

Versions follow [Semantic Versioning 2.0.0](https://semver.org/spec/v2.0.0.html). The version is defined in `include/namturbo/version.h`
(`kVersion` and `kPreRelease`; CMake reads it from there). A release candidate has a pre-release identifier (`1.0.0-rc.1`, tag `v1.0.0-rc.1`);
a final release has none (`1.0.0`, tag `v1.0.0`). The CMake package is version-compatible within one major version.

Steps for a release:

1. Update `version.h` and add the section for the version to `CHANGELOG.md` ([Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format).
2. Run all tests with and without the baseline (`NAMTURBO_WERROR=ON`).
3. Rerun the benchmarks on the test machines (`benchmarks/run_report.py`, `two_instance`), add the data to `docs/results/`, and regenerate
   the documentation with `python benchmarks/make_docs.py`. CI fails if the generated files differ from the committed files.
4. Tag the commit `vMAJOR.MINOR.PATCH[-PRERELEASE]` and publish the release (a pre-release for release candidates).

## Licence

Contributions are released under the MIT licence of this repository.
