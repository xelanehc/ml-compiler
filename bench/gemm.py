"""Matrix-product throughput in isolation: our generated kernels against
Apple Accelerate (reached through PyTorch) and ONNX Runtime.

`mlc gemm` compiles every distinct matrix product of the model as a kernel of
its own and times it. This script runs it, then times the same products, with
the same shapes, in the two libraries, and writes bench/results/gemm_T<T>.md.

Each library gets the operand layout it prefers: ours reads the second operand
transposed (that is what the compiler arranges for, at no run-time cost);
PyTorch and ONNX Runtime get it as a plain [k, n] matrix. A product that has a
bias in the model is timed without it everywhere except in our kernel, where
adding it is part of the kernel; it is one add per k multiplies.

Usage:
  python bench/gemm.py                 # T=1024
  python bench/gemm.py --seq-len 16
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MLC = ROOT / "build" / "release" / "mlc"
SECONDS = 1.0  # measuring time per product, per library


def best_time(fn):
    """Minimum wall time of fn over at least three calls and about SECONDS of measuring."""
    fn()
    best, spent, calls = float("inf"), 0.0, 0
    while calls < 3 or spent < SECONDS:
        start = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - start
        best, spent, calls = min(best, elapsed), spent + elapsed, calls + 1
    return best


def worker(args):
    """Times every product in gemm.json with PyTorch and ONNX Runtime at one thread count."""
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)  # Apple Accelerate
    import numpy as np
    import onnxruntime as ort
    import torch

    torch.set_num_threads(args.threads)
    onnx_dir = ROOT / "artifacts" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)

    class MatMul(torch.nn.Module):
        def forward(self, a, b):
            return torch.matmul(a, b)

    rows = []
    for product in json.loads(Path(args.products).read_text())["products"]:
        a_shape = product["a"]
        b_shape = product["b"][:-2] + [product["b"][-1], product["b"][-2]]  # back to [.., k, n]
        rng = np.random.default_rng(0)
        a = rng.standard_normal(a_shape, dtype=np.float32)
        b = rng.standard_normal(b_shape, dtype=np.float32)
        ta, tb = torch.from_numpy(a), torch.from_numpy(b)

        def run_torch():
            with torch.inference_mode():
                return torch.matmul(ta, tb)

        name = "matmul_" + "x".join(map(str, a_shape)) + "_" + "x".join(map(str, b_shape)) + ".onnx"
        path = onnx_dir / name
        if not path.exists():
            torch.onnx.export(MatMul(), (ta, tb), dynamo=True).save(str(path))
        options = ort.SessionOptions()
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = 1
        session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
        feed = {session.get_inputs()[0].name: a, session.get_inputs()[1].name: b}

        rows.append({"torch_s": best_time(run_torch), "ort_s": best_time(lambda: session.run(None, feed))})
        print(f"  threads={args.threads} {a_shape} x {b_shape}: torch {product['flops'] / rows[-1]['torch_s'] / 1e9:7.1f}  "
              f"ort {product['flops'] / rows[-1]['ort_s'] / 1e9:7.1f} GFLOP/s", file=sys.stderr, flush=True)
    print(json.dumps(rows))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--products", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        args.threads = args.threads[0]
        return worker(args)
    if not MLC.exists():
        raise SystemExit("build/release/mlc not found: build the compiler first (see README)")

    out_dir = ROOT / "build" / "release" / "gen" / f"gemm_T{args.seq_len}"
    subprocess.run([str(MLC), "gemm", str(ROOT / "artifacts" / "graphs" / f"gpt2_T{args.seq_len}.json"), "-o", str(out_dir),
                    "--threads", ",".join(map(str, args.threads)), "--seconds", str(SECONDS)], check=True)
    ours = json.loads((out_dir / "gemm.json").read_text())

    # A fresh process per thread count: the libraries size their pools at startup.
    theirs = {}
    for threads in args.threads:
        proc = subprocess.run([sys.executable, __file__, "--worker", "--threads", str(threads), "--products", str(out_dir / "gemm.json")],
                              stdout=subprocess.PIPE, text=True, check=True)
        theirs[threads] = json.loads(proc.stdout.strip().splitlines()[-1])

    pmset = subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout
    battery = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout
    power = ("AC" if "AC Power" in battery else "battery") + " power"
    if "lowpowermode         1" in pmset:
        power += " with Low Power Mode on (about half the normal single-core speed)"

    out = [
        f"# Matrix products alone, T={args.seq_len}",
        "",
        "Every distinct matrix product in GPT-2 small at this sequence length, timed on its own, in GFLOP/s",
        "(2 x rows x inner x columns floating-point operations per product; best of several runs).",
        "`mlc` is this compiler's generated kernel. PyTorch calls Apple Accelerate, which is not limited to",
        "the CPU's SIMD units and does not follow the thread setting (STEERING.md section 7).",
        f"Measured on {power}.",
        "",
    ]
    for t, threads in enumerate(args.threads):
        out += [f"## {threads} thread{'s' if threads != 1 else ''}", "",
                "| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |", "|---|---:|---:|---:|---:|---:|"]
        total = {"flops": 0.0, "mlc": 0.0, "ort": 0.0, "torch": 0.0}
        for product, other in zip(ours["products"], theirs[threads]):
            b = product["b"][:-2] + [product["b"][-1], product["b"][-2]]
            flops, count = product["flops"], product["count"]
            times = {"mlc": product["seconds"][t], "ort": other["ort_s"], "torch": other["torch_s"]}
            out.append(f"| {product['a']} x {b} | {count} | {flops * count / 1e9:.2f} | " +
                       " | ".join(f"{flops / times[k] / 1e9:.1f}" for k in ("mlc", "ort", "torch")) + " |")
            total["flops"] += flops * count
            for k in times:
                total[k] += times[k] * count
        out.append(f"| **all, weighted by use** | | {total['flops'] / 1e9:.2f} | " +
                   " | ".join(f"**{total['flops'] / total[k] / 1e9:.1f}**" for k in ("mlc", "ort", "torch")) + " |")
        out.append("")
    out += [f"Reproduce with `python bench/gemm.py --seq-len {args.seq_len}`.", ""]

    path = ROOT / "bench" / "results" / f"gemm_T{args.seq_len}.md"
    path.write_text("\n".join(out))
    print("\n".join(out))
    print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
