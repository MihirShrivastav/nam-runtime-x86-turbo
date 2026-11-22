#!/usr/bin/env python3
"""Regenerates the generated parts of the documentation from the data in docs/results/.

    python benchmarks/make_docs.py

Writes docs/RESULTS.md and replaces the speed-up table between the markers in README.md. Input: the JSON files written by `nam_bench --json`
(`<machine>_<compiler>_<rate>.json`), the two-instance logs (`two-instance_<machine>_<compiler>.log`), and the text files INTRO.md and SOAK.md in
the same folder. CI runs this script and fails if the result differs from the committed files.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "docs" / "results"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_report import compiler_label, cpu_label  # noqa: E402


def soak_label(log: Path) -> str:
    """Machine and compiler of a two-instance log, taken from the benchmark file of the same machine and compiler."""
    key = log.stem.removeprefix("two-instance_")
    reference = sorted(RESULTS.glob(f"{key}_48000.json"))
    if not reference:
        raise SystemExit(f"{log.name}: no matching benchmark file {key}_48000.json")
    data = json.loads(reference[0].read_text(encoding="utf-8"))
    core = data.get("pinned_core", -1)
    pin = f", logical core {core}" if core >= 0 else ""
    return f"{cpu_label(data['cpu'])}, {compiler_label(data['compiler'])}{pin}"


def run(args):
    result = subprocess.run([sys.executable, str(ROOT / "benchmarks" / "make_report.py"), *args], capture_output=True, text=True)
    if result.returncode != 0:
        sys.stderr.write(result.stderr)
        raise SystemExit(result.returncode)
    return result.stdout


def main() -> int:
    jsons = sorted(RESULTS.glob("*.json"))
    if not jsons:
        raise SystemExit("no benchmark files in docs/results")
    args = [str(p) for p in jsons] + ["--intro", str(RESULTS / "INTRO.md"), "--soak-intro", str(RESULTS / "SOAK.md")]
    for log in sorted(RESULTS.glob("two-instance_*.log")):
        args += ["--soak", f"{soak_label(log)}={log}"]
    (ROOT / "docs" / "RESULTS.md").write_text(run(args), encoding="utf-8", newline="\n")
    run([str(p) for p in jsons if p.stem.endswith("_48000")] + ["--readme", str(ROOT / "README.md")])
    return 0


if __name__ == "__main__":
    sys.exit(main())
