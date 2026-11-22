"""Generate golden outputs from NAM Core's A2 fast path (block 64):

    data/golden_full_<input>.raw   (A2-Full, slim 1.0)
    data/golden_lite_<input>.raw   (A2-Lite, slim 0.0)

    python benchmarks/make_golden.py <golden.exe> <model.nam> <data_dir>

Exit codes: 0 done, 1 failure, 77 skipped (model file missing).
"""

import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
from inputs import make_inputs  # noqa: E402


def main() -> int:
    golden, model, data_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if not model.exists():
        print(f"SKIP: model file not found: {model}")
        return 77
    inputs = make_inputs(data_dir)
    for slim, tag in ((1.0, "full"), (0.0, "lite")):
        for label, inp in inputs.items():
            out = data_dir / f"golden_{tag}_{label}.raw"
            r = subprocess.run(
                [golden, "--model", str(model), "--slim", str(slim), "--input", str(inp), "--out", str(out),
                 "--block", "64", "--path", "fast"],
                capture_output=True, text=True,
            )
            if r.returncode != 0:
                print(f"golden failed for {tag} {label}: {r.stderr.strip()[:300]}")
                return 1
            print(f"wrote {out.name} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
