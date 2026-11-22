"""Turn the output of benchmarks/two_instance (several runs concatenated in one log) into a markdown table.

    python benchmarks/soak_table.py soak.log

Each run in the log starts with a line "== two instances (A2-full + A2-lite) on one thread, ..." and contains the compute
time of the pair for the engines of this library and (with --nam) for NAM Core's fast path. Runs with --control-us also contain
two control lines, which are printed below the table.
"""

import re
import sys

RUN = re.compile(r"^== two instances \(A2-(\w+) \+ A2-(\w+)\).*?buffer (\d+), (.*?)(?:, raised priority)?$")
COMPUTE = re.compile(
    r"compute per block \(pair\): p50\s+([\d.]+) us\s+p99\s+([\d.]+)\s+p99\.9\s+([\d.]+)\s+max\s+([\d.]+)\s+=\s+([\d.]+)%\s+/\s+([\d.]+)%\s+/\s+([\d.]+)%"
)
OVER = re.compile(r"blocks over the deadline:\s+(\d+) of (\d+)")
LATE = re.compile(r"wake-up lateness \(operating system\): p50\s+(\d+) us\s+p99\s+(\d+)\s+max\s+(\d+)")


def pacing_name(text: str) -> str:
    if "sleeping" in text:
        return "sleeping between blocks"
    if "busy-wait" in text:
        return "busy-waiting between blocks"
    return "free-running"


def parse(path: str):
    runs, controls, cur, last, key_now = [], [], None, None, None
    for line in open(path, encoding="utf-8", errors="replace"):
        m = RUN.match(line.strip())
        if m:
            cur = {"pair": f"A2-{m.group(1).capitalize()} + A2-{m.group(2).capitalize()}", "buffer": m.group(3), "pacing": pacing_name(m.group(4)), "who": {}}
            runs.append(cur)
            continue
        if cur is None:
            continue
        c = COMPUTE.search(line)
        if c:
            label = line.split("compute per block")[0].strip()
            vals = [float(v) for v in c.groups()]
            if label.startswith("control"):
                controls.append((label, vals))
                last = None
            else:
                key = "ours" if label.startswith("ours") else "nam"
                key_now = key
                cur["who"][key] = vals
                last = (cur["who"], key)
            continue
        w = LATE.search(line)
        if w and key_now is not None:
            cur["who"][key_now + "_late"] = tuple(int(v) for v in w.groups())
            continue
        o = OVER.search(line)
        if o and last is not None:
            last[0][last[1] + "_over"] = (int(o.group(1)), int(o.group(2)))
            last = None
    return runs, controls


def table_lines(runs):
    out = ["| Pair, buffer, pacing | namturbo: p50 / p99 / max | Share of the deadline (p50 / max) | Blocks over the deadline | "
           "NAM Core fast path: p50 / p99 / max | Blocks over the deadline | Pair speed-up (p50) | Wake-up delay of the system: p99 / max |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in runs:
        o, n = r["who"].get("ours"), r["who"].get("nam")
        oo, no = r["who"].get("ours_over"), r["who"].get("nam_over")
        lt = r["who"].get("ours_late")
        if not o:
            continue
        nam_txt = f"{n[0]:.1f} / {n[1]:.0f} / {n[3]:.0f} us" if n else "not run"
        nam_over = f"{no[0]}" if no else "-"
        speed = f"{n[0] / o[0]:.2f}x" if n else "-"
        late = f"{lt[1] / 1000:.2f} / {lt[2] / 1000:.1f} ms" if lt else "-"
        out.append(f"| {r['pair']}, buffer {r['buffer']}, {r['pacing']} | {o[0]:.1f} / {o[1]:.0f} / {o[3]:.0f} us | {o[4]:.1f}% / {o[6]:.1f}% | "
                   f"**{oo[0]} of {oo[1]:,}** | {nam_txt} | {nam_over} | {speed} | {late} |")
    return out


def main() -> int:
    runs, controls = parse(sys.argv[1])
    if not runs:
        print("no runs found", file=sys.stderr)
        return 1
    print("\n".join(table_lines(runs)))
    for label, v in controls:
        print(f"\n{label[0].upper() + label[1:]}: p50 {v[0]:.1f} us, p99 {v[1]:.1f} us, max {v[3]:.1f} us")
    return 0


if __name__ == "__main__":
    sys.exit(main())
