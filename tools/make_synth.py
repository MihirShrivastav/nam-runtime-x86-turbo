"""Write the synthetic golden input: data/synth.raw (mono float32, 48 kHz, 2 s).

0.5 s silence, 1.0 s of noise plus 220 Hz and 1230 Hz sines (peak about 0.4, roughly -8 dBFS),
0.5 s silence. The trailing silence also exercises state decay after a loud burst. Fixed seed.

    python tools/make_synth.py [out.raw]
"""

import sys
from pathlib import Path

import numpy as np

SR = 48_000


def make_synth() -> np.ndarray:
    rng = np.random.default_rng(1)
    t = np.arange(SR) / SR
    burst = 0.10 * rng.uniform(-1, 1, SR) + 0.15 * np.sin(2 * np.pi * 220 * t) + 0.08 * np.sin(2 * np.pi * 1230 * t)
    x = np.concatenate([np.zeros(SR // 2), burst, np.zeros(SR // 2)]).astype("<f4")
    assert len(x) == 2 * SR
    return x


if __name__ == "__main__":
    out = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parent.parent / "tests" / "data" / "synth.raw")
    out.parent.mkdir(parents=True, exist_ok=True)
    make_synth().tofile(out)
    print(f"wrote {out} ({out.stat().st_size} bytes)")
