"""Where the time goes: seconds per kind of operation, for every sequence length.

Compiles the model, runs it with `mlc exec --profile` (each kernel's best time
over several runs, summed by op), and writes bench/results/per_op.md.

Usage:
  python bench/per_op.py                  # T = 16 64 256 1024, 1 and 4 threads
  python bench/per_op.py --tuned          # with the schedules in tuning/cache.json
"""

import argparse
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLC = ROOT / "build" / "release" / "mlc"
GRAPHS = ROOT / "artifacts" / "graphs"
GOLDEN = ROOT / "artifacts" / "golden"


def profile(seq_len, threads, tuned):
    out_dir = ROOT / "build" / "release" / "gen" / f"profile_T{seq_len}"
    cmd = [str(MLC), "compile", str(GRAPHS / f"gpt2_T{seq_len}.json"), "-o", str(out_dir)]
    if tuned:
        cmd += ["--tuning", str(ROOT / "tuning" / "cache.json"), "--tuned-threads", str(threads)]
    subprocess.run(cmd, check=True, capture_output=True)
    runs = {16: 40, 64: 20, 256: 10}.get(seq_len, 4)
    text = subprocess.run(
        [str(MLC), "exec", str(out_dir / "model.so"), "--weights", str(GRAPHS / "gpt2_weights.bin"),
         "--input", str(GOLDEN / f"T{seq_len}_ids.npy"), "--expect", str(GOLDEN / f"T{seq_len}_logits.npy"),
         "--runs", str(runs), "--threads", str(threads), "--profile"],
        capture_output=True, text=True).stdout
    if "PASS" not in text:
        raise SystemExit(f"T={seq_len} does not match the golden logits:\n{text}")
    ops = {m.group(1): (int(m.group(2)), float(m.group(3))) for m in re.finditer(r"^(\w+)\s+(\d+)\s+([\d.]+)\s+[\d.]+%$", text, re.M)}
    whole = float(re.search(r"then min ([\d.]+) s", text).group(1))
    return ops, whole


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[16, 64, 256, 1024])
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--tuned", action="store_true")
    args = parser.parse_args()

    battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    power = ("AC" if "AC Power" in battery else "battery") + " power"
    if "lowpowermode         1" in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout:
        power += " with Low Power Mode on"

    out = ["# Time per operation", "",
           "GPT-2 small compiled by mlc" + (", autotuned schedules" if args.tuned else ", default schedules") + ". Each kernel's best time over several runs,",
           "summed by kind of operation, in milliseconds, with the share of the total in parentheses.",
           "`gemm` is the 48 dense layers, `matmul` the 24 attention products and the output projection.",
           f"The last row is the whole call, timed separately. Measured on {power}.", ""]
    for threads in args.threads:
        results = {t: profile(t, threads, args.tuned) for t in args.seq_lens}
        names = sorted({op for ops, _ in results.values() for op in ops}, key=lambda op: -results[args.seq_lens[-1]][0].get(op, (0, 0))[1])
        out += [f"## {threads} thread{'s' if threads != 1 else ''}", "",
                "| op | kernels | " + " | ".join(f"T={t}" for t in args.seq_lens) + " |", "|---|---:|" + "---:|" * len(args.seq_lens)]
        for op in names:
            cells = []
            for t in args.seq_lens:
                ops, _ = results[t]
                total = sum(seconds for _, seconds in ops.values())
                cells.append(f"{ops[op][1] * 1e3:.3f} ({100 * ops[op][1] / total:.1f}%)" if op in ops else "")
            out.append(f"| {op} | {results[args.seq_lens[-1]][0][op][0]} | " + " | ".join(cells) + " |")
        out.append("| **sum of kernels** | | " + " | ".join(f"**{sum(s for _, s in results[t][0].values()) * 1e3:.2f}**" for t in args.seq_lens) + " |")
        out.append("| whole call (min) | | " + " | ".join(f"{results[t][1] * 1e3:.2f}" for t in args.seq_lens) + " |")
        out.append("")
    out += ["Reproduce with `python bench/per_op.py" + (" --tuned" if args.tuned else "") + "`.", ""]
    path = ROOT / "bench" / "results" / ("per_op_tuned.md" if args.tuned else "per_op.md")
    path.write_text("\n".join(out))
    print("\n".join(out))


if __name__ == "__main__":
    main()
