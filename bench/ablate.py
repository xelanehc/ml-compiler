"""Attribute the compiler's speed and memory to individual optimization passes.

Builds the model several times with different passes switched off
(`mlc compile --disable ...`), runs each build with `mlc exec`, and writes
bench/results/<name>.md.

Two ways of slicing it:

  cumulative     start with everything off and switch passes on one at a time,
                 in pipeline order. Each row shows what one more pass adds.
  leave-one-out  everything on except one pass. Each row shows what is lost
                 without that pass. This is the ablation STEERING.md asks for.

Usage:
  python bench/ablate.py                                   # cumulative, T = 16 64 256
  python bench/ablate.py --mode leave-one-out --seq-lens 64
  python bench/ablate.py --seq-lens 1024 --configs none all --runs 2
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLC = ROOT / "build" / "release" / "mlc"
GRAPHS = ROOT / "artifacts" / "graphs"
GOLDEN = ROOT / "artifacts" / "golden"
PASSES = ["simplify", "fold", "cse", "gelu", "attention", "memory"]  # pipeline order


def configurations(mode):
    """(label, passes to disable) for each build."""
    if mode == "cumulative":
        rows = [("none", list(PASSES))]
        for i, name in enumerate(PASSES):
            rows.append((f"+ {name}", PASSES[i + 1:]))
        return rows
    return [("all", [])] + [(f"all but {name}", [name]) for name in PASSES]


def measure(seq_len, disabled, runs):
    """Compile one configuration, run it, and return its numbers."""
    out_dir = ROOT / "build" / "release" / "gen" / f"ablate_T{seq_len}"
    cmd = [str(MLC), "compile", str(GRAPHS / f"gpt2_T{seq_len}.json"), "-o", str(out_dir)]
    if disabled:
        cmd += ["--disable", ",".join(disabled)]
    text = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    nodes, kernels, arena = re.search(r"(\d+) graph nodes -> (\d+) kernels, ([\d.]+) MB arena", text).groups()

    # The first run pays for page faults on the fresh arena; mlc exec reports
    # it separately, and the min and median are over the runs after it.
    text = subprocess.run(
        [str(MLC), "exec", str(out_dir / "model.so"), "--weights", str(GRAPHS / "gpt2_weights.bin"),
         "--input", str(GOLDEN / f"T{seq_len}_ids.npy"), "--expect", str(GOLDEN / f"T{seq_len}_logits.npy"),
         "--runs", str(runs + 1)],
        capture_output=True, text=True,
    ).stdout
    if "PASS" not in text:
        raise SystemExit(f"T={seq_len} with {disabled} disabled does not match the golden logits:\n{text}")
    first, best, median = map(float, re.search(r"first run ([\d.]+) s, then min ([\d.]+) s, median ([\d.]+) s", text).groups())
    return {"nodes": int(nodes), "kernels": int(kernels), "arena_mb": float(arena), "first_s": first, "min_s": best, "median_s": median}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cumulative", "leave-one-out"], default="cumulative")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--runs", type=int, default=0, help="timed runs per build; 0 picks a count from the sequence length")
    parser.add_argument("--configs", nargs="+", help="only these rows, by label (e.g. none all)")
    parser.add_argument("--name", default=None, help="basename of the output file")
    args = parser.parse_args()
    if not MLC.exists():
        raise SystemExit("build/release/mlc not found: build the compiler first (see README)")

    rows = configurations(args.mode)
    if args.configs:
        wanted = set(args.configs)
        rows = [r for r in rows if r[0] in wanted or (r[0] == "+ memory" and "all" in wanted)]
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    power = "AC" if "AC Power" in power else "battery" if "Battery Power" in power else "unknown"

    out = [
        f"# Pass ablation ({args.mode})",
        "",
        "GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.",
        "Every build was checked against the golden logits before it was timed.",
        f"Measured on {power} power. Times are the minimum over the timed runs, with the median in parentheses;",
        "the first run of each build (page faults on a fresh arena) is listed separately.",
        "",
    ]
    for seq_len in args.seq_lens:
        runs = args.runs or {16: 10, 64: 6, 256: 4}.get(seq_len, 2)
        out += [f"## T={seq_len} ({runs} timed runs per build)", "",
                "| passes | nodes | kernels | arena MB | first run s | time s | vs first row |", "|---|---:|---:|---:|---:|---:|---:|"]
        base = None
        for label, disabled in rows:
            r = measure(seq_len, disabled, runs)
            base = base or r["min_s"]
            out.append(f"| {label} | {r['nodes']} | {r['kernels']} | {r['arena_mb']:.1f} | {r['first_s']:.2f} | "
                       f"{r['min_s']:.3f} ({r['median_s']:.3f}) | {base / r['min_s']:.2f}x |")
            print(f"T={seq_len:5d}  {label:16s} {r['min_s']:8.3f} s  arena {r['arena_mb']:8.1f} MB  {r['kernels']} kernels", flush=True)
        out.append("")
    out += [f"Reproduce with `python bench/ablate.py --mode {args.mode} --seq-lens {' '.join(map(str, args.seq_lens))}`.", ""]

    name = args.name or f"m3_ablation_{args.mode.replace('-', '_')}"
    path = ROOT / "bench" / "results" / f"{name}.md"
    path.write_text("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
