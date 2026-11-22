"""Golden input material, shared by the golden self-check and the golden-file generator.

    data/synth.raw     2 s synthetic burst with silence                           (tools/make_synth.py)
    data/sweep10s.raw  10 s of sweeps, noise, chirps and a quiet tail: the hardest material the tests use (tools/make_sweep.py)
"""

from pathlib import Path

from make_sweep import make_sweep
from make_synth import make_synth


def make_inputs(data_dir: Path) -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    synth = data_dir / "synth.raw"
    make_synth().tofile(synth)
    sweep = data_dir / "sweep10s.raw"
    make_sweep().tofile(sweep)
    return {"synth": synth, "sweep10s": sweep}
