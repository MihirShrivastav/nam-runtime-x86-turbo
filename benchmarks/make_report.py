#!/usr/bin/env python3
"""Turns the JSON files written by `nam_bench --json` into Markdown tables.

Usage:
  make_report.py <results.json> [<results.json> ...] [--intro file.md] [--model name] [--soak "Machine=log" ...] [--soak-intro file.md] > report.md
  make_report.py <results.json> [...] --readme README.md      (replaces the table between the markers in the README)

Each JSON file is one run of nam_bench on one machine with one compiler at one host rate. The report has a summary (the speed-up
over the NAM Core fast path at 48 kHz, and the share of the real-time budget that the processor uses at every host rate), followed by
detail tables with the time of one block, ours and NAM Core. The summary uses one model, by default the first model in the files.
Nothing in the report is typed by hand. The text of --intro is placed before the tables.
"""
import argparse
import json
import re
import sys
from pathlib import Path
from collections import OrderedDict

START = "<!-- results:start -->"
END = "<!-- results:end -->"


def load(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    data["path"] = path
    return data


def cpu_label(name):
    name = re.sub(r"\((R|TM)\)", "", name)
    name = re.sub(r"\bwith Radeon Graphics\b", "", name)
    return re.sub(r"\s+", " ", name).strip()


def compiler_label(name):
    m = re.fullmatch(r"MSVC (\d{2})(\d{2})", name.strip())
    if m:
        return f"MSVC {m.group(1)}.{m.group(2)}"
    m = re.match(r"(clang-cl|clang) (\d+\.\d+\.\d+)", name)
    if m:
        return f"{m.group(1)} {m.group(2)}"
    return name.split("(")[0].strip()


def label(data):
    return f'{cpu_label(data["cpu"])}, {compiler_label(data["compiler"])}'


def has_baseline(data):
    return bool(data.get("baseline"))


def os_factor(data):
    return int(data.get("oversampling", 1))


def variant_name(variant):
    return "A2-Full" if variant == "full" else "A2-Lite"


def machine_order(files):
    """The machine and compiler combinations in the order of the files."""
    seen = OrderedDict()
    for d in files:
        seen.setdefault(label(d), True)
    return list(seen)


def speedup_tables(files, model, rate=48000):
    based = [d for d in files if has_baseline(d) and int(d["host_rate"]) == rate and os_factor(d) == 1]
    if not based:
        return []
    buffers = sorted({r["buffer"] for d in based for r in d["rows"] if r["model"] == model})
    out = []
    for variant in ("full", "lite"):
        out.append(f"**{variant_name(variant)}** (buffer size in samples)")
        out.append("")
        out.append("| Machine and compiler | " + " | ".join(str(b) for b in buffers) + " |")
        out.append("|---|" + "---:|" * len(buffers))
        for d in based:
            by = {r["buffer"]: r for r in d["rows"] if r["model"] == model and r["variant"] == variant}
            out.append(f"| {label(d)} | " + " | ".join(f'{by[b]["speedup"]:.1f}x' if b in by else "" for b in buffers) + " |")
        out.append("")
    return out


def load_tables(files, model):
    plain = [d for d in files if os_factor(d) == 1]
    rates = sorted({int(d["host_rate"]) for d in plain})
    buffers = sorted({r["buffer"] for d in plain for r in d["rows"] if r["model"] == model})
    out = []
    for variant in ("full", "lite"):
        out.append(f"**{variant_name(variant)}**: load at the median block time (buffer size in samples)")
        out.append("")
        out.append("| Machine and compiler | Host rate | " + " | ".join(str(b) for b in buffers) + " |")
        out.append("|---|---:|" + "---:|" * len(buffers))
        for lab in machine_order(plain):
            for rate in rates:
                for d in plain:
                    if label(d) != lab or int(d["host_rate"]) != rate:
                        continue
                    by = {r["buffer"]: r for r in d["rows"] if r["model"] == model and r["variant"] == variant}
                    cells = [f'{100.0 * by[b]["ours_p50_us"] / by[b]["block_us"]:.2f}%' if b in by else "" for b in buffers]
                    out.append(f"| {lab} | {rate / 1000:g} kHz | " + " | ".join(cells) + " |")
        out.append("")
    return out


def oversampling_tables(files, model):
    oversampled = [d for d in files if os_factor(d) > 1]
    if not oversampled:
        return []
    factors = sorted({os_factor(d) for d in oversampled})
    buffers = sorted({r["buffer"] for d in oversampled for r in d["rows"] if r["model"] == model})
    out = []
    for variant in ("full", "lite"):
        out.append(f"**{variant_name(variant)}**: load at the median block time with oversampling at 48 kHz (buffer size in samples)")
        out.append("")
        out.append("| Machine and compiler | Oversampling | " + " | ".join(str(b) for b in buffers) + " |")
        out.append("|---|---:|" + "---:|" * len(buffers))
        for lab in machine_order(oversampled):
            for factor in factors:
                for d in oversampled:
                    if label(d) != lab or os_factor(d) != factor:
                        continue
                    by = {r["buffer"]: r for r in d["rows"] if r["model"] == model and r["variant"] == variant}
                    cells = [f'{100.0 * by[b]["ours_p50_us"] / by[b]["block_us"]:.2f}%' if b in by else "" for b in buffers]
                    out.append(f"| {lab} | {factor}x | " + " | ".join(cells) + " |")
        out.append("")
    return out


def detail_section(data):
    os_suffix = f", {os_factor(data)}x oversampling" if os_factor(data) > 1 else ""
    out = [f"### {label(data)}, host rate {data['host_rate'] / 1000:g} kHz{os_suffix}", ""]
    pin = f'thread pinned to logical core {data["pinned_core"]}' if data.get("pinned_core", -1) >= 0 else "thread not pinned"
    out.append(f'{data["seconds"]:.0f} s of audio per run, median of {data["runs"]} runs; {pin}.')
    out.append("")
    groups = OrderedDict()
    for r in data["rows"]:
        groups.setdefault((r["model"], r["variant"]), []).append(r)
    for (model, variant), rows in groups.items():
        out.append(f"**{model}, {variant_name(variant)}**")
        out.append("")
        if has_baseline(data):
            out.append("| Buffer | Ours p50 (us) | Ours p99 (us) | NAM Core p50 (us) | NAM Core p99 (us) | Speed-up (p50) |")
            out.append("|---:|---:|---:|---:|---:|---:|")
            for r in rows:
                out.append(f'| {r["buffer"]} | {r["ours_p50_us"]:.2f} | {r["ours_p99_us"]:.2f} | {r["base_p50_us"]:.2f} | '
                           f'{r["base_p99_us"]:.2f} | {r["speedup"]:.2f}x |')
        else:
            out.append("| Buffer | p50 (us) | p99 (us) | Load at p50 |")
            out.append("|---:|---:|---:|---:|")
            for r in rows:
                out.append(f'| {r["buffer"]} | {r["ours_p50_us"]:.2f} | {r["ours_p99_us"]:.2f} | '
                           f'{100.0 * r["ours_p50_us"] / r["block_us"]:.2f}% |')
        out.append("")
    if data.get("esr_db"):
        out.append(f'Error against the NAM Core fast path (energy of the difference relative to the energy of the signal): '
                   f'{max(data["esr_db"]):.1f} dB or lower.')
        out.append("")
    return out


def readme_fragment(files, model):
    out = [START, "", f"Speed-up over the NAM Core A2 fast path, 48 kHz, model `{model}`, median of 5 runs. A speed-up of 3x means one third of the "
           "processing time. Further rates and details: [docs/RESULTS.md](docs/RESULTS.md).", ""]
    out += speedup_tables(files, model)
    out.append(END)
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+")
    ap.add_argument("--intro")
    ap.add_argument("--model")
    ap.add_argument("--readme")
    ap.add_argument("--soak", action="append", default=[], help="Machine label=log written by two_instance; the table of the two-instance test is added")
    ap.add_argument("--soak-intro")
    a = ap.parse_args()

    files = [load(p) for p in a.files]
    model = a.model or files[0]["rows"][0]["model"]

    if a.readme:
        text = open(a.readme, encoding="utf-8").read()
        if START not in text or END not in text:
            print(f"{a.readme}: the markers {START} and {END} are missing", file=sys.stderr)
            return 2
        head, rest = text.split(START, 1)
        _, tail = rest.split(END, 1)
        with open(a.readme, "w", encoding="utf-8", newline="\n") as f:
            f.write(head + readme_fragment(files, model) + tail)
        return 0

    out = []
    if a.intro:
        out.append(open(a.intro, encoding="utf-8").read().rstrip("\n"))
        out.append("")
    out += ["## Speed-up over the NAM Core fast path at 48 kHz", "", f"Model: `{model}`.", ""]
    out += speedup_tables(files, model)
    out += ["## Share of the real-time budget used by namturbo", "",
            "Load is the time to process a block divided by the duration of the block. 1% means that the processor is busy for 1% of the time "
            "that the audio takes. NAM Core has no sample-rate conversion, so there is no comparison at 44.1, 96 and 192 kHz; the load there includes "
            "the conversion.", ""]
    out += load_tables(files, model)
    os_section = oversampling_tables(files, model)
    if os_section:
        out += ["## Oversampling at 48 kHz", "",
                "Load with 2x and 4x phase-interleaved oversampling (`Standard` half-band filters, 48 kHz host rate).", ""]
        out += os_section
    if a.soak:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from soak_table import parse, table_lines
        out += [open(a.soak_intro, encoding="utf-8").read().rstrip("\n"), ""] if a.soak_intro else ["## Two instances in real time", ""]
        for spec in a.soak:
            label_text, path = spec.split("=", 1)
            runs, _ = parse(path)
            out += [f"**{label_text}**", ""] + table_lines(runs) + [""]
    out += ["## Details", ""]
    for d in files:
        out += detail_section(d)
    print("\n".join(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
