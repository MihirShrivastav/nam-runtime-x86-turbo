# Accuracy

Error of the namturbo output against NAM Core and against a float64 reference, and the measurement method.

## Error measure

The error is the energy of the difference between two signals divided by the energy of the reference signal, in dB (error-to-signal ratio,
ESR). For orientation: the noise floor of a 16-bit recording is about -96 dB and that of a 24-bit recording about -144 dB. A second
figure, the largest single-sample difference relative to the largest sample of the signal, is printed by the `accuracy` tool; it is -87 dB or
lower over all 60 combinations below.

References:

- **NAM Core A2 fast path** (commit `d3189ed`): the reference implementation of the A2 network.
- **NAM Core generic WaveNet path**: a second implementation in NAM Core. The distance between the generic and the fast path indicates the
  spread between two correct 32-bit floating-point implementations.
- **float64 forward pass** (`benchmarks/f64_truth.py`): independent 64-bit implementation using the weights of the file. It stands for the exact result.

## Results against the NAM Core fast path

Models: the model of this repository and five other A2 captures (clean to high gain). Signals: five direct-input guitar recordings of 24 s to
55 s (chords, single notes, palm-muted riffs, loud passages). Both variants, blocks of 64 samples, 48 kHz: 12 model-variant pairs with 5
signals each, 60 combinations. The five other captures and the recordings are not part of the repository.

Error over the five signals, worst to best, in dB. "NAM generic" is the error of the NAM Core generic path against its own fast path.

| Model | A2-Full: namturbo | A2-Full: NAM generic | A2-Lite: namturbo | A2-Lite: NAM generic |
|---|---:|---:|---:|---:|
| Clean amplifier 1 | -124.1 to -128.4 | -125.4 to -129.1 | -123.8 to -130.7 | -123.1 to -129.8 |
| Clean amplifier 2 | -123.1 to -128.3 | -129.9 to -132.2 | -121.6 to -127.2 | -117.0 to -124.6 |
| Crunch amplifier | -120.1 to -124.4 | -125.0 to -127.8 | -120.3 to -124.3 | -115.9 to -120.8 |
| Lead amplifier | -128.8 to -129.9 | -130.7 to -132.2 | -125.1 to -127.5 | -121.5 to -124.7 |
| High-gain amplifier | -103.7 to -108.6 | -111.3 to -116.0 | -103.5 to -108.3 | -98.7 to -103.7 |
| Model of this repository | -120.1 to -126.7 | -125.2 to -129.1 | -116.3 to -124.1 | -110.2 to -119.0 |

With the generated test signals, the error of namturbo on the model of this repository is -126.8 dB (A2-Full) and -126.0 dB (A2-Lite).

The worst of the 60 combinations is -103.5 dB (high-gain capture). The high-gain capture has the largest errors for NAM Core as well. A
high-gain network amplifies differences between floating-point implementations, and its output is small relative to that amplification, which
raises the relative error.

## Results against the exact result

Error against the float64 forward pass, two captures, both variants, one 28 s recording:

| Capture | Variant | namturbo | NAM Core fast path |
|---|---|---:|---:|
| High gain | A2-Full | -105.4 dB | -107.3 dB |
| High gain | A2-Lite | -99.3 dB | -99.1 dB |
| Crunch | A2-Full | -122.0 dB | -123.9 dB |
| Crunch | A2-Lite | -116.8 dB | -116.6 dB |

A2-Lite matches the NAM Core fast path. The A2-Full error is about 2 dB higher.

The convolution sums the products in two levels: the eight products of one tap first, then the taps. A single chain of fused multiply-adds
produced an A2-Full error 7 dB higher on the high-gain capture. The difference was not visible with generated test signals and appeared
with recordings. See [technique notes](technique-notes.md).

## Block size independence

A signal processed in one call, in blocks of 1 to 700 samples, or in place gives identical samples (bit-exact). This holds for both
variants, with and without sample-rate conversion. The tests verify it.

## Sample-rate conversion

`tests/test_api.cpp` compares the converted path with the direct path for a strongly distorting model. The comparison requires the input to be
sampled at the same instants, which the test arranges with the delay of the first stage. Test tone, `Standard` quality:

| Host rate | Fundamental differs by | Harmonics 2 to 5, relative to the fundamental |
|---|---|---|
| 44.1 kHz | 0.0000 dB | -131.6 dB or lower |
| 96 kHz | 0.0000 dB | -129.0 dB or lower |
| 192 kHz | 0.0000 dB | -129.7 dB or lower |
| 32 kHz | 0.0000 dB | -122.2 dB or lower |

`LowLatency` (72 dB attenuation instead of 96 dB): the same harmonics are at -84 dB to -93 dB or lower. The passband is flat up to 90%
(`Standard`) or 80% (`LowLatency`) of the lower Nyquist frequency.

### Real recording

Model of this repository, host rate 44.1 kHz, 20 s of a direct-input guitar recording (24-bit, chords and single notes). The reference
(`benchmarks/rate_check.py`) is built as follows: the recording is converted to 48 kHz with an ideal Fourier resampler, shifted so that the
network sees the sampling instants that it sees inside the processor; NAM Core processes it; the output is converted back to 44.1 kHz with the
same resampler and delayed by the latency of the processor. The error is measured against that reference.

| Quality | A2-Full | A2-Lite | Recording limited to the flat band (A2-Full, A2-Lite) |
|---|---:|---:|---:|
| `Standard` | -98.4 dB | -95.0 dB | -104.9 dB, -104.4 dB |
| `LowLatency` | -81.7 dB | -81.8 dB | -81.9 dB, -82.2 dB |

Control without conversion (48 kHz): -126.8 dB.

The converters remove content above the flat band (90% of Nyquist for `Standard`, 80% for `LowLatency`); the ideal reference keeps it.
With the recording limited to the flat band (88% and 78% of Nyquist), the `Standard` error is lower by 6 dB to 9 dB, at about -105 dB. The remaining
`Standard` error and the whole `LowLatency` error come from the finite attenuation of the filters (96 dB and 72 dB). All errors are below
-80 dB; the default quality is at -95 dB or lower.

The low harmonics of a strongly distorting network depend on the instants at which the input is sampled, because the network aliases. Two
inputs that represent the same band-limited signal, sampled at different instants, give outputs that differ by about -50 dB in these
harmonics with any implementation. This property of the model is not caused by the converter.

## A2-Lite compared with A2-Full

A2-Lite is a smaller network. Both networks approximate the same amplifier, and each has its own error with respect to it. Their outputs
have similar level and character and different waveforms. `benchmarks/variant_compare.cpp` measures the error of A2-Lite against A2-Full.

Model of this repository:

| Signal | A2-Lite against A2-Full | After level matching | Level of A2-Lite minus A2-Full |
|---|---:|---:|---:|
| `tests/data/sweep10s.raw` (generated: sweeps, noise, tones, plucked strings) | -16.5 dB | -16.5 dB | -0.1 dB |
| `tests/data/synth.raw` (generated: noise and two sine waves) | -11.2 dB | -11.2 dB | -0.4 dB |
| Five guitar recordings (direct input) | -11.8 dB to -17.2 dB | | -0.2 dB to 0.0 dB |

The levels agree within 0.5 dB. The waveform error of -12 dB to -17 dB corresponds to a difference signal of about 14% to 25% of the signal
amplitude. A2-Lite is therefore not a transparent replacement for A2-Full. These are measurements of the signals, not listening tests. The
automatic switch to A2-Lite is disabled by default.

The difference depends on the capture. Five other captures (clean to high gain; not part of the repository, no figures given) had a level
within 0.25 dB of A2-Full. Their waveform differences varied by a factor of several in amplitude. The model of this repository and the
high-gain capture showed the largest differences; the clean and lead captures followed A2-Full more closely.

## Repeating the measurements

Required: a baseline build ([CONTRIBUTING.md](../CONTRIBUTING.md)), models, and recordings as raw 32-bit float mono files at the model
sample rate (`tools/wav_to_raw.py` converts a mono WAV file).

    build/Release/accuracy --model amp1.nam --model amp2.nam --input di1.raw --input di2.raw --csv result.csv

One line is printed per model, variant and recording. The exit code is non-zero if the error against the NAM Core fast path exceeds `--limit`
(default -100 dB). `--dump folder` writes the outputs of namturbo and of the fast path; `benchmarks/f64_truth.py` computes the float64 forward
pass for the comparison with the exact result.

A2-Lite against A2-Full (no baseline needed):

    build/Release/variant_compare --model amp.nam --input di.raw

Sample-rate conversion on a recording at the host rate (baseline build, NumPy, `process_file` and `golden`):

    python benchmarks/rate_check.py --model amp.nam --input di_44100.raw --host-rate 44100 \
        --process-file build-baseline/Release/process_file --golden build-baseline/Release/golden
