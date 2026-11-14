"""Run the benchmarks and write a results table.

Launches bench/worker.py once per (system, thread count), one at a time, and
collects the results into:

  bench/results/baselines.json   every raw sample, plus the environment
  bench/results/baselines.md     the summary table (committed)

Usage:
  python bench/run.py                         # everything
  python bench/run.py --seq-lens 16 64        # a quick subset
  python bench/run.py --systems ort --threads 4
  python bench/run.py --systems mlc --threads 1 --name mlc_m2   # our compiler; writes mlc_m2.md

The baselines are measured by default. Our compiler ("mlc") is opt-in, and
its results go to a separate file via --name so the baseline table is not
overwritten. mlc is single-threaded for now and is only run at 1 thread; while
it is slow, its measuring time is capped (see --mlc-budget), so its cells may
have fewer than 30 runs. The run count is printed and saved with each result.
"""

import argparse
import datetime
import json
import platform
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))
from model import SEQ_LENS  # noqa: E402

RESULTS = ROOT / "bench" / "results"
BASELINES = ("eager", "compile", "ort")
SYSTEMS = BASELINES + ("mlc",)
LABELS = {"eager": "PyTorch eager", "compile": "torch.compile", "ort": "ONNX Runtime", "mlc": "mlc (ours)"}


def shell(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def environment():
    import numpy
    import onnxruntime
    import torch
    import transformers

    blas = re.search(r"BLAS_INFO=(\w+)", torch.__config__.show())
    power = shell("pmset", "-g", "batt")
    return {
        "date": datetime.date.today().isoformat(),
        "cpu": shell("sysctl", "-n", "machdep.cpu.brand_string"),
        "performance_cores": shell("sysctl", "-n", "hw.perflevel0.physicalcpu"),
        "efficiency_cores": shell("sysctl", "-n", "hw.perflevel1.physicalcpu"),
        "os": f"{platform.system()} {platform.release()} {platform.machine()}",
        "power": "AC" if "AC Power" in power else "battery" if "Battery Power" in power else "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_blas": blas.group(1) if blas else "unknown",
        "transformers": transformers.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": numpy.__version__,
    }


def render(env, rows, seq_lens, name):
    by_key = {(r["system"], r["threads"], r["seq_len"]): r for r in rows}
    systems = [s for s in SYSTEMS if any(r["system"] == s for r in rows)]
    thread_counts = sorted({r["threads"] for r in rows})

    out = [
        "# Baseline results" if name == "baselines" else f"# Results: {name}",
        "",
        "Full forward pass of GPT-2 small, token ids to logits, batch 1, fp32.",
        "Each cell is **median** latency in milliseconds, with the minimum in parentheses.",
        "",
    ]
    for threads in thread_counts:
        out += [
            f"## {threads} thread{'s' if threads != 1 else ''}",
            "",
            "| system | " + " | ".join(f"T={t}" for t in seq_lens) + " |",
            "|---|" + "---:|" * len(seq_lens),
        ]
        for system in systems:
            cells = []
            for t in seq_lens:
                r = by_key.get((system, threads, t))
                cells.append(f"{r['median_ms']:.1f} ({r['min_ms']:.1f})" if r else "n/a")
            out.append(f"| {LABELS[system]} | " + " | ".join(cells) + " |")
        out.append("")

    worst = max(rows, key=lambda r: r["max_abs_diff"])
    out += [
        "## Correctness",
        "",
        f"Every configuration was checked against the golden logits. "
        f"Largest max-abs difference: {worst['max_abs_diff']:.1e} "
        f"({LABELS[worst['system']]}, T={worst['seq_len']}). "
        f"Argmax matched at every position: {'yes' if all(r['argmax_match'] for r in rows) else 'NO'}.",
        "",
        "## Environment",
        "",
        *[f"- {key}: {value}" for key, value in env.items()],
        "",
        "Reproduce with `python export/export_gpt2.py && python bench/run.py`.",
        "",
    ]
    short = [r for r in rows if r["runs"] < 30]
    if short:
        cells = ", ".join(f"{LABELS[r['system']]} T={r['seq_len']} ({r['runs']} runs)" for r in short)
        out += [f"Cells measured with fewer than 30 runs because of the time cap: {cells}.", ""]
    if env["power"] != "AC":
        out.insert(2, "> **Measured on battery power.** Numbers may be throttled and noisy; rerun on AC before quoting them.\n")
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(BASELINES))
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--seq-lens", type=int, nargs="+", default=list(SEQ_LENS))
    parser.add_argument("--name", default="baselines", help="basename for the output files")
    parser.add_argument("--mlc-budget", type=float, default=120.0,
                        help="seconds of measuring allowed per sequence length for mlc")
    args = parser.parse_args()

    env = environment()
    print(f"power: {env['power']}   torch BLAS: {env['torch_blas']}")

    rows = []
    for system in args.systems:
        for threads in args.threads:
            if system == "mlc" and threads != 1:
                print("  mlc is single-threaded for now; skipping threads=" + str(threads))
                continue
            budget = args.mlc_budget if system == "mlc" else 0.0
            proc = subprocess.run(
                [sys.executable, str(ROOT / "bench" / "worker.py"),
                 "--system", system, "--threads", str(threads),
                 "--budget-seconds", str(budget),
                 "--seq-lens", *map(str, args.seq_lens)],
                stdout=subprocess.PIPE, text=True,
            )
            if proc.returncode != 0:
                print(f"  {system} threads={threads} FAILED (exit {proc.returncode})")
                continue
            rows += json.loads(proc.stdout.strip().splitlines()[-1])

    if not rows:
        raise SystemExit("no results")

    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{args.name}.json").write_text(json.dumps({"environment": env, "results": rows}, indent=1))
    table = render(env, rows, args.seq_lens, args.name)
    (RESULTS / f"{args.name}.md").write_text(table)
    print()
    print(table)


if __name__ == "__main__":
    main()
