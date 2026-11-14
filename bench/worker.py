"""Benchmark one system at one thread count, across sequence lengths.

run.py launches this in a fresh process for every (system, threads) pair. A
fresh process matters: thread pools are sized once at startup, and
torch.compile bakes the thread count into the code it generates, so changing
thread counts inside a live process does not reliably take effect.

Prints one JSON object on the last line of stdout.
"""

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))

WARMUP_RUNS = 5
MIN_RUNS = 30
MIN_SECONDS = 2.0  # fast configs keep sampling until this much time is measured
MAX_RUNS = 500


def time_it(fn):
    """Return per-call wall times in milliseconds.

    Warmup calls are discarded: the first few calls pay for one-time work
    (lazy allocation, page faults on fresh memory, thread pool startup, cold
    caches) that is not part of steady-state inference.
    """
    for _ in range(WARMUP_RUNS):
        fn()
    samples = []
    total = 0.0
    while len(samples) < MIN_RUNS or (total < MIN_SECONDS and len(samples) < MAX_RUNS):
        start = time.perf_counter_ns()
        fn()
        elapsed = (time.perf_counter_ns() - start) / 1e6
        samples.append(elapsed)
        total += elapsed / 1e3
    return samples


def build_eager(model, ids_np, threads):
    import torch

    ids = torch.from_numpy(ids_np)

    def run():
        with torch.inference_mode():
            return model(ids)

    return run, lambda out: out.numpy()


def build_compile(model, ids_np, threads):
    import torch

    # Forget anything compiled for a previous sequence length, and forbid
    # dynamic shapes, so each T gets code specialized to that exact shape.
    # That is the same deal our compiler gets.
    torch._dynamo.reset()
    compiled = torch.compile(model, dynamic=False)
    ids = torch.from_numpy(ids_np)

    def run():
        with torch.inference_mode():
            return compiled(ids)

    return run, lambda out: out.numpy()


def build_ort(model, ids_np, threads):
    import onnxruntime as ort
    import torch

    seq_len = ids_np.shape[1]
    onnx_dir = ROOT / "artifacts" / "onnx"
    onnx_dir.mkdir(parents=True, exist_ok=True)
    path = onnx_dir / f"gpt2_T{seq_len}.onnx"
    if not path.exists():
        program = torch.onnx.export(model, (torch.from_numpy(ids_np),), dynamo=True)
        program.save(str(path))

    options = ort.SessionOptions()
    options.intra_op_num_threads = threads  # threads used inside one operator
    options.inter_op_num_threads = 1        # no running several operators at once
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session = ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])
    feed = {session.get_inputs()[0].name: ids_np}

    def run():
        return session.run(None, feed)

    return run, lambda out: out[0]


BUILDERS = {"eager": build_eager, "compile": build_compile, "ort": build_ort}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=BUILDERS, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", required=True)
    args = parser.parse_args()

    # Must be set before torch is imported: OpenMP reads it once at startup.
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.threads)  # Apple Accelerate

    import numpy as np
    import torch
    from model import load_golden, load_model

    torch.set_num_threads(args.threads)
    model = load_model()

    results = []
    for seq_len in args.seq_lens:
        ids_np, golden = load_golden(seq_len)
        build_start = time.perf_counter()
        run, to_numpy = BUILDERS[args.system](model, ids_np, args.threads)
        logits = to_numpy(run())  # first call: includes any compilation
        build_seconds = time.perf_counter() - build_start

        max_diff = float(np.abs(logits - golden).max())
        argmax_match = bool((logits.argmax(-1) == golden.argmax(-1)).all())

        samples = time_it(run)
        results.append(
            {
                "system": args.system,
                "threads": args.threads,
                "seq_len": seq_len,
                "median_ms": statistics.median(samples),
                "min_ms": min(samples),
                "runs": len(samples),
                "build_seconds": build_seconds,
                "max_abs_diff": max_diff,
                "argmax_match": argmax_match,
                "samples_ms": samples,
            }
        )
        print(
            f"  {args.system:8s} threads={args.threads} T={seq_len:5d}  "
            f"median {statistics.median(samples):9.2f} ms  min {min(samples):9.2f} ms  "
            f"({len(samples)} runs, max|diff| {max_diff:.1e})",
            file=sys.stderr,
            flush=True,
        )

    print(json.dumps(results))


if __name__ == "__main__":
    main()
