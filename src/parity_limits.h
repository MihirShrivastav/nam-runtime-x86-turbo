// ESR limits used by the tests.
#pragma once

namespace namturbo
{
// ESR limit of an engine's output against NAM Core's fast-path output (the "golden") on real audio.
//
// Measured, not chosen (first 2 s of the test sweep, tests/data/a2-example.nam, benchmarks/f64_truth.py):
// distance from the exact (float64) answer is -125.5 dB for NAM's A2-Full fast path and -117.8 dB for
// our plain scalar float engine; for A2-Lite it is -114.4 dB for both NAM's fast path and our scalar engine
// (the model itself amplifies rounding). NAM Core's fast and generic paths differ from each other by
// -126.8 dB (A2-Full) and -113.2 dB (A2-Lite). Correct float32 engines therefore differ from the golden by
// -113 to -127 dB depending on model, input and accumulation order. -110 dB is above all of that and
// still far below what a real bug produces (typically -80 dB or worse).
inline double parity_limit_db(int /*channels*/)
{
  return -110.0;
}

// ESR limit of an engine's output against the exact (float64) answer on HOSTILE inputs: full-scale DC,
// Nyquist, square waves, over-range noise. A2-Lite amplifies float rounding by an input-dependent amount, so
// no fixed limit near -110 dB can hold for it in general. On tests/data/a2-example.nam the worst stimulus of
// benchmarks/precision_probe scores -107.2 dB for our A2-Lite engine (DC -1) and -112.6 dB for NAM Core's A2-Lite
// fast path (DC 0.5); a different A2-Lite capture reached -98.8 dB for our engine on DC +1. Averaged over 20 DC
// levels the two implementations are equal (ours -115.0 dB, NAM Core -114.7 dB), which the test
// precision_vs_nam_* checks directly. The Lite limit is therefore set a few dB past the worst value either
// reaches, and still 15 dB short of what a real bug produces (-80 dB or worse).
inline double truth_limit_db(int channels)
{
  return channels == 3 ? -95.0 : -110.0;
}
} // namespace namturbo
