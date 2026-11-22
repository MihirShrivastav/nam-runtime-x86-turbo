# Test data

All files are mono, 48 kHz. The `.raw` files are little-endian 32-bit floats.

| File | What it is |
|---|---|
| `a2-example.nam` | The example A2 slimmable model of NAM Core (MIT licence, see `NOTICE`). It contains A2-Lite (3 channels) and A2-Full (8 channels). |
| `test-amp-and-cab.nam` | A2 slimmable model (A2-Lite and A2-Full) of a Fender Tweed Deluxe 5E3 amplifier and cabinet, made by the author of this repository with TONE3000 and released under the licence of this repository (MIT). The benchmarks use it, together with generated signals. The reference outputs below do not belong to it. |
| `synth.raw` | 2 s test input: 0.5 s of silence, 1 s of noise and two sine waves, 0.5 s of silence. Made by `tools/make_synth.py`. |
| `sweep10s.raw` | 10 s test input: noise ramp, sweeps, 64 tones, noise bursts, plucked strings, FM and AM tone, impulses. Made by `tools/make_sweep.py`; its description is in that file. |
| `golden_full_synth.raw`, `golden_full_sweep10s.raw` | Output of NAM Core's A2 fast path for A2-Full on the two inputs. |
| `golden_lite_synth.raw`, `golden_lite_sweep10s.raw` | The same for A2-Lite. |

The tests read these files; they do not need NAM Core or Python.

## How the reference outputs were made

NAM Core commit `d3189ed482ded880c128bf312941266c4fa5ceec`, built with `NAM_ENABLE_A2_FAST`, with MSVC 19.44 and default options.
Each input was processed in blocks of 64 samples after the model was reset the way a host resets it (the model warms up with silence).
The command is:

    cmake -S . -B build-baseline -DNAMTURBO_BASELINE=ON
    cmake --build build-baseline --config Release --target regenerate_test_data

It needs the NAM Core submodule (`git submodule update --init --recursive`), Python 3 and NumPy.

## Why the inputs are stored as files

The inputs can be regenerated with the two scripts above, but they are stored because the random number streams of NumPy are not
guaranteed to stay the same between versions, and the reference outputs belong to exactly these inputs. They were made with NumPy 2.x.
If the data is regenerated, all four reference outputs and the inputs must be regenerated together.

## How exact the reference outputs are

A different compiler builds NAM Core slightly differently, so its output changes in the last bits. The tests compare with a limit of
-110 dB error-to-signal ratio (`src/parity_limits.h`), far above that difference, so they pass with reference outputs made by
another compiler. The reference outputs are not a statement of the exact answer; the precision tests in `benchmarks/` compare against a
float64 implementation.
