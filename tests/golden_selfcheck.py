"""Self-check of the golden tool: NAM Core's fast and generic paths must agree, and the fast path
must not depend on the block size.

    python tests/golden_selfcheck.py <golden.exe> <model.nam> <data_dir>

If the golden output itself is unreliable, nothing downstream can be trusted. Exit codes: 0 pass,
1 failure, 77 skipped (model file missing).

Measured with tests/data/a2-example.nam (fast vs generic ESR): A2-Full -124.2 dB (synthetic), -128.4 dB (sweep);
A2-Lite -121.9 dB (synthetic), -117.5 dB (sweep). Two correct float32 implementations differ by
about that much, so the limits below are per model and sit with margin below the measured values.
"""

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

LIMIT_DB = {"A2-Full": -120.0, "A2-Lite": -112.0}
MODELS = ((1.0, "A2-Full"), (0.0, "A2-Lite"))


def esr_db(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(np.float64), b.astype(np.float64)
    return float(10 * np.log10(np.mean((a - b) ** 2) / np.mean(b**2)))


def run_golden(golden, model, slim, inp, out, block, path):
    r = subprocess.run(
        [golden, "--model", str(model), "--slim", str(slim), "--input", str(inp), "--out", str(out),
         "--block", str(block), "--path", path],
        capture_output=True, text=True,
    )
    if r.returncode != 0 or not out.exists():
        raise RuntimeError(f"golden failed (exit {r.returncode}): {r.stderr.strip()[:300]}")
    return np.fromfile(out, dtype="<f4")


def main() -> int:
    golden, model, data_dir = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    if not model.exists():
        print(f"SKIP: model file not found: {model}")
        return 77
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
    from inputs import make_inputs

    inputs = make_inputs(data_dir)
    synth = inputs["synth"]

    failures = 0
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for slim, name in MODELS:
            for label, inp in inputs.items():
                n_in = np.fromfile(inp, dtype="<f4").size
                try:
                    fast = run_golden(golden, model, slim, inp, tmp / "fast.raw", 64, "fast")
                    gen = run_golden(golden, model, slim, inp, tmp / "gen.raw", 64, "generic")
                except RuntimeError as e:
                    print(f"{name} {label}: {e}")
                    failures += 1
                    continue
                for path, y in (("fast", fast), ("generic", gen)):
                    if y.size != n_in or not np.isfinite(y).all() or np.mean(y.astype(np.float64) ** 2) <= 0:
                        print(f"{name} {label} {path}: bad output (len {y.size} vs {n_in})")
                        failures += 1
                e = esr_db(fast, gen)
                ok = e < LIMIT_DB[name]
                print(f"{name} {label}: ESR fast vs generic = {e:.1f} dB (limit {LIMIT_DB[name]}: {'ok' if ok else 'ABOVE'})")
                failures += 0 if ok else 1

            # Block-size consistency of NAM Core's fast path. On the MSVC default build the output is bit-identical at any
            # block size; with FMA or clang (measured: /arch:AVX2 and clang-cl) the single-frame path rounds differently,
            # at -126..-129 dB. So the requirement is "bit-identical or within rounding", and which one is reported.
            ref = run_golden(golden, model, slim, synth, tmp / "b64.raw", 64, "fast")
            for blk in (1, 512):
                y = run_golden(golden, model, slim, synth, tmp / f"b{blk}.raw", blk, "fast")
                same = np.array_equal(y, ref)
                e = -300.0 if same else esr_db(y, ref)
                ok = same or e < -120.0
                print(f"{name} fast path block {blk} vs 64: {'bit-identical' if same else f'ESR {e:.1f} dB'} ({'ok' if ok else 'ABOVE -120'})")
                failures += 0 if ok else 1
    print("ALL PASSED" if failures == 0 else f"FAILED: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
