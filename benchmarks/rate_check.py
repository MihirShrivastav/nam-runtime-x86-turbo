#!/usr/bin/env python3
"""How accurate is the sample-rate conversion of the Processor on a real recording?

    python benchmarks/rate_check.py --model amp.nam --input host.raw --host-rate 44100 \
        --process-file build/Release/process_file.exe --golden build-baseline/Release/golden.exe [--variant full|lite] [--resampler standard|low-latency]

The input is a mono recording at the HOST rate (raw little-endian float32). The check builds the ideal result and compares it with the
Processor:

 1. The recording is converted to the model rate with an ideal (Fourier) resampler, shifted in time by the delay of the first conversion
    stage of the Processor, so that the network sees the samples at the same instants as it does inside the Processor. This matters:
    a distorting network aliases, and its low harmonics depend on the instants at which its input is sampled.
 2. NAM Core (A2 fast path, `golden`) runs that signal at the model rate.
 3. Its output is converted back to the host rate with the ideal resampler, advanced by the same delay, and delayed by the latency of the
    Processor.
 4. The Processor runs the original recording at the host rate (`process_file`).
 5. The error-to-signal ratio of (4) against (3) is printed, after leaving out the silence added around the recording.

Needs NumPy and a build with the baseline (for `golden`). The model must run at 48 kHz.
"""
import argparse
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

MODEL_RATE = 48000


def resample_shift(x, new_len, host_rate, rate_in, shift_s):
    """Fourier resampling of x (sampled at rate_in) to new_len samples, with the signal delayed by shift_s seconds (negative: advanced)."""
    n = len(x)
    spec = np.fft.rfft(x)
    m_bins = new_len // 2 + 1
    out = np.zeros(m_bins, dtype=complex)
    k = min(len(spec), m_bins)
    out[:k] = spec[:k] * (new_len / n)
    freqs = np.arange(m_bins) * rate_in / n
    out *= np.exp(-2j * np.pi * freqs * shift_s)
    return np.fft.irfft(out, new_len)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--host-rate", type=int, required=True)
    ap.add_argument("--process-file", required=True)
    ap.add_argument("--golden", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--resampler", default="standard")
    ap.add_argument("--seconds", type=float, default=0.0, help="use only the first seconds of the recording")
    ap.add_argument("--band-limit", type=float, default=0.0,
                    help="remove everything above this fraction of the Nyquist frequency from the recording first (for example 0.9), to separate the effect "
                         "of the band that the converter removes by design from its other errors")
    a = ap.parse_args()
    for name in ("model", "input", "process_file", "golden"):
        setattr(a, name, str(Path(getattr(a, name)).resolve()))

    g = math.gcd(a.host_rate, MODEL_RATE)
    host_per, model_per = a.host_rate // g, MODEL_RATE // g
    x = np.fromfile(a.input, dtype="<f4").astype(np.float64)
    if a.seconds > 0:
        x = x[: int(a.seconds * a.host_rate)]
    if a.band_limit > 0:
        spec = np.fft.rfft(x)
        spec[int(len(spec) * a.band_limit):] = 0
        x = np.fft.irfft(spec, len(x))
    pad = 2 * a.host_rate
    n = len(x) + 2 * pad
    n += (-n) % host_per                      # a whole number of periods, so that the model-rate length is whole
    xp = np.zeros(n)
    xp[pad : pad + len(x)] = x
    n_model = n // host_per * model_per

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "host.raw").write_bytes(xp.astype("<f4").tobytes())
        info = subprocess.run([a.process_file, "--model", a.model, "--input", str(tmp / "host.raw"), "--output", str(tmp / "ours.raw"),
                               "--host-rate", str(a.host_rate), "--variant", a.variant, "--resampler", a.resampler],
                              capture_output=True, text=True, check=True).stdout
        vals = dict(line.split("=") for line in info.split())
        latency = int(vals["latency_samples"])
        delay_model = float(vals["stage1_delay_model_samples"])
        tau = delay_model / MODEL_RATE

        r48 = resample_shift(xp, n_model, a.host_rate, a.host_rate, tau)      # the network input, at the instants of the Processor
        (tmp / "r48.raw").write_bytes(r48.astype("<f4").tobytes())
        slim = "0.0" if a.variant == "lite" else "1.0"
        subprocess.run([a.golden, "--model", a.model, "--slim", slim, "--input", str(tmp / "r48.raw"), "--out", str(tmp / "y48.raw"),
                        "--block", "64", "--path", "fast"], check=True, capture_output=True)
        y48 = np.fromfile(tmp / "y48.raw", dtype="<f4").astype(np.float64)
        z = resample_shift(y48, n, MODEL_RATE, MODEL_RATE, -tau)               # back to the host rate, advanced by the first-stage delay
        ideal = np.zeros(n)
        ideal[latency:] = z[: n - latency]                                      # then delayed by the latency of the Processor
        ours = np.fromfile(tmp / "ours.raw", dtype="<f4").astype(np.float64)

    start, end = pad, pad + len(x)
    d = ours[start:end] - ideal[start:end]
    esr = 10 * math.log10(np.sum(d * d) / np.sum(ideal[start:end] ** 2))
    peak = 20 * math.log10(np.max(np.abs(d)) / np.max(np.abs(ideal[start:end])))
    print(f"host rate {a.host_rate} Hz, {a.variant}, {a.resampler}: latency {latency} samples, first-stage delay {delay_model:.3f} model samples, "
          f"{len(x) / a.host_rate:.1f} s of audio")
    print(f"error of the Processor against the ideal result: ESR {esr:.1f} dB, largest difference {peak:.1f} dB relative to the peak")
    return 0


if __name__ == "__main__":
    sys.exit(main())
