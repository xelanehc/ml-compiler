"""Attribute the compiler's speed and memory to individual optimizations.

Builds the model several times with different optimizations switched off
(`mlc compile --disable ...`), runs each build with `mlc exec`, and writes
bench/results/<name>.md.

Two stages of the compiler can be taken apart:

  --stage m4  (default) the loop optimizations, on top of all graph passes
  --stage m3  the graph passes, with every loop optimization off

and two ways of slicing each:

  cumulative     start with the stage's optimizations off and switch them on
                 one at a time. Each row shows what one more adds.
  leave-one-out  everything on except one. Each row shows what is lost
                 without it. This is the ablation STEERING.md asks for.

Usage:
  python bench/ablate.py                                   # m4, cumulative, T = 16 64 256
  python bench/ablate.py --mode leave-one-out --seq-lens 64
  python bench/ablate.py --stage m3 --seq-lens 1024 --configs none "+ memory" --runs 2
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
GRAPH_PASSES = ["simplify", "fold", "cse", "gelu", "attention", "memory"]  # pipeline order
# The order in which the loop optimizations build on each other: `layout`
# rearranges the matrix product for the vectorizer, `hoist` removes the cost
# layout introduced, the two tilings shape the loops around it, and threads
# come last.
LOOP_OPTS = ["noalias", "vectorize", "vmath", "layout", "hoist", "regtile", "cachetile", "parallel"]
THREADS = 4


def configurations(stage, mode):
    """(label, optimizations to disable, threads) for each build."""
    if stage == "m3":
        names, always_off, start = GRAPH_PASSES, LOOP_OPTS, "none"
    else:
        names, always_off, start = LOOP_OPTS, [], "graph passes only (M3)"
    if mode == "cumulative":
        rows = [(start, always_off + names, 1)]
        for i, name in enumerate(names):
            rows.append((f"+ {name}", always_off + names[i + 1:], 1))
        if stage == "m4":  # `parallel` only shows with more than one thread
            rows[-1] = (f"+ parallel, 1 thread", [], 1)
            rows.append((f"+ parallel, {THREADS} threads", [], THREADS))
        return rows
    threads = THREADS if stage == "m4" else 1
    return [("all", always_off, threads)] + [(f"all but {name}", always_off + [name], threads) for name in names]


def measure(seq_len, disabled, threads, runs):
    """Compile one configuration, run it, and return its numbers."""
    out_dir = ROOT / "build" / "release" / "gen" / f"ablate_T{seq_len}"
    cmd = [str(MLC), "compile", str(GRAPHS / f"gpt2_T{seq_len}.json"), "-o", str(out_dir)]
    if disabled:
        cmd += ["--disable", ",".join(disabled)]
    text = subprocess.run(cmd, check=True, capture_output=True, text=True).stdout
    nodes, kernels, arena, persistent = re.search(
        r"(\d+) graph nodes -> (\d+) kernels \+ \d+ at load time, ([\d.]+) MB arena .*?, ([\d.]+) MB persistent", text).groups()

    # The first run pays for page faults on the fresh arena; mlc exec reports
    # it separately, and the min and median are over the runs after it.
    text = subprocess.run(
        [str(MLC), "exec", str(out_dir / "model.so"), "--weights", str(GRAPHS / "gpt2_weights.bin"),
         "--input", str(GOLDEN / f"T{seq_len}_ids.npy"), "--expect", str(GOLDEN / f"T{seq_len}_logits.npy"),
         "--runs", str(runs + 1), "--threads", str(threads)],
        capture_output=True, text=True,
    ).stdout
    if "PASS" not in text:
        raise SystemExit(f"T={seq_len} with {disabled} disabled does not match the golden logits:\n{text}")
    first, best, median = map(float, re.search(r"first run ([\d.]+) s, then min ([\d.]+) s, median ([\d.]+) s", text).groups())
    return {"nodes": int(nodes), "kernels": int(kernels), "arena_mb": float(arena), "persistent_mb": float(persistent),
            "first_s": first, "min_s": best, "median_s": median}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["m3", "m4"], default="m4")
    parser.add_argument("--mode", choices=["cumulative", "leave-one-out"], default="cumulative")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 64, 256])
    parser.add_argument("--runs", type=int, default=0, help="timed runs per build; 0 picks a count from the sequence length")
    parser.add_argument("--configs", nargs="+", help="only these rows, by label (e.g. none all)")
    parser.add_argument("--name", default=None, help="basename of the output file")
    args = parser.parse_args()
    if not MLC.exists():
        raise SystemExit("build/release/mlc not found: build the compiler first (see README)")

    rows = configurations(args.stage, args.mode)
    if args.configs:
        wanted = set(args.configs)
        rows = [r for r in rows if r[0] in wanted]
    power = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    power = "AC" if "AC Power" in power else "battery" if "Battery Power" in power else "unknown"
    if "lowpowermode         1" in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout:
        power += " power with Low Power Mode on (about half the normal single-core speed)"
    else:
        power += " power"

    what = "graph passes, every loop optimization off" if args.stage == "m3" else "loop optimizations, on top of all graph passes"
    threads_note = "one thread" if args.stage == "m3" else f"one thread unless the row says otherwise ({THREADS} for leave-one-out)"
    out = [
        f"# Ablation: {what} ({args.mode})",
        "",
        f"GPT-2 small, full forward pass, {threads_note}, compiled with `clang++ -O3 -mcpu=native`.",
        "Every build was checked against the golden logits before it was timed.",
        f"Measured on {power}. Times are the minimum over the timed runs, with the median in parentheses;",
        "the first run of each build (page faults on a fresh arena) is listed separately.",
        "",
    ]
    for seq_len in args.seq_lens:
        runs = args.runs or {16: 10, 64: 6, 256: 4}.get(seq_len, 2)
        out += [f"## T={seq_len} ({runs} timed runs per build)", "",
                "| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"]
        base = None
        for label, disabled, threads in rows:
            r = measure(seq_len, disabled, threads, runs)
            base = base or r["min_s"]
            out.append(f"| {label} | {r['nodes']} | {r['kernels']} | {r['arena_mb']:.1f} | {r['persistent_mb']:.1f} | {r['first_s']:.3f} | "
                       f"{r['min_s']:.3f} ({r['median_s']:.3f}) | {base / r['min_s']:.2f}x |")
            print(f"T={seq_len:5d}  {label:26s} {r['min_s']:8.3f} s  arena {r['arena_mb']:8.1f} MB  {r['kernels']} kernels", flush=True)
        out.append("")
    out += [f"Reproduce with `python bench/ablate.py --stage {args.stage} --mode {args.mode} --seq-lens {' '.join(map(str, args.seq_lens))}`.", ""]

    name = args.name or f"{args.stage}_ablation_{args.mode.replace('-', '_')}"
    path = ROOT / "bench" / "results" / f"{name}.md"
    path.write_text("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
