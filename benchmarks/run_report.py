#!/usr/bin/env python3
"""One command for the benchmark table of one build.

Usage: run_report.py <nam_bench executable> <output folder> [--model file.nam ...] [--runs 3] [--seconds 2] [--core N] [--tag name] [--buffers 64,128] [--host-rate 44100]

Runs nam_bench for the buffer sizes (default 16 to 512) and both variants of every model at the host rate (default 48000 Hz), writes <tag>.json (every number) and <tag>.md (the tables) into
the output folder. The tag defaults to the compiler name found in the JSON file. The baseline column needs a nam_bench that was built with
-DNAMTURBO_BASELINE=ON. The default models are the two in tests/data: test-amp-and-cab.nam and NAM Core's example.
"""
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exe")
    ap.add_argument("out")
    ap.add_argument("--model", action="append")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--seconds", type=float, default=2.0)
    ap.add_argument("--core", type=int, default=-1)
    ap.add_argument("--tag")
    ap.add_argument("--buffers", default="16,32,64,128,256,512")
    ap.add_argument("--host-rate", type=float, default=48000.0)
    a = ap.parse_args()

    data = HERE.parent / "tests" / "data"
    models = a.model or [str(data / "test-amp-and-cab.nam"), str(data / "a2-example.nam")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "_running.json"
    cmd = [str(Path(a.exe).resolve()), "--buffers", a.buffers, "--host-rate", str(a.host_rate), "--runs", str(a.runs), "--seconds", str(a.seconds), "--json", str(tmp)]
    for m in models:
        cmd += ["--model", m]
    if a.core >= 0:
        cmd += ["--core", str(a.core)]
    result = subprocess.run(cmd)
    if result.returncode != 0:
        print(f"nam_bench failed with code {result.returncode}", file=sys.stderr)
        return result.returncode
    data = json.loads(tmp.read_text(encoding="utf-8"))
    tag = a.tag or re.sub(r"[^A-Za-z0-9._-]+", "_", data["compiler"].split("(")[0].strip())[:40]
    json_path = out / f"{tag}.json"
    tmp.replace(json_path)
    report = subprocess.run([sys.executable, str(HERE / "make_report.py"), str(json_path)], capture_output=True, text=True, check=True)
    (out / f"{tag}.md").write_text(report.stdout, encoding="utf-8", newline="\n")
    print(f"written: {json_path} and {out / (tag + '.md')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
