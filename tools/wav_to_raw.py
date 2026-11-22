"""Convert a mono 16/24/32-bit PCM WAV (or a slice of it) to raw little-endian float32.

    python tools/wav_to_raw.py in.wav out.raw [--start-s S] [--dur-s D]
"""

import argparse
import wave
from pathlib import Path

import numpy as np


def read_wav(path: str):
    with wave.open(path) as w:
        assert w.getnchannels() == 1, "mono only"
        width, sr, n = w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        x = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        x = np.where(x >= 0x800000, x - 0x1000000, x)
        return x.astype(np.float32) / 2**23, sr
    if width == 2:
        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / 2**15, sr
    if width == 4:
        return np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2**31, sr
    raise ValueError(f"unsupported sample width {width}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("out")
    ap.add_argument("--start-s", type=float, default=0.0)
    ap.add_argument("--dur-s", type=float, default=None)
    a = ap.parse_args()
    x, sr = read_wav(a.wav)
    i0 = int(a.start_s * sr)
    i1 = len(x) if a.dur_s is None else min(len(x), i0 + int(a.dur_s * sr))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    x[i0:i1].astype("<f4").tofile(a.out)
    print(f"wrote {a.out}: {i1 - i0} samples at {sr} Hz")


if __name__ == "__main__":
    main()
