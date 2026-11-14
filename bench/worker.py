"""Benchmark one system at one thread count, across sequence lengths.

run.py launches this in a fresh process for every (system, threads) pair. A
fresh process matters: thread pools are sized once at startup, and
torch.compile bakes the thread count into the code it generates, so changing
thread counts inside a live process does not reliably take effect.

Prints one JSON object on the last line of stdout.
"""

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "export"))

WARMUP_RUNS = 5
MIN_RUNS = 30
MIN_SECONDS = 2.0  # fast configs keep sampling until this much time is measured
MAX_RUNS = 500


def time_it(fn, budget_seconds=0.0):
    """Return per-call wall times in milliseconds.

    Warmup calls are discarded: the first few calls pay for one-time work
    (lazy allocation, page faults on fresh memory, thread pool startup, cold
    caches) that is not part of steady-state inference.

    A nonzero budget_seconds caps the time spent on a slow system: warmup stops
    once a fifth of the budget is gone, and sampling stops once the budget is
    spent, as long as at least three samples exist. The result then has fewer
    than MIN_RUNS samples, and the "runs" field says how many.
    """
    start_all = time.perf_counter()
    for i in range(WARMUP_RUNS):
        fn()
        if budget_seconds and time.perf_counter() - start_all > budget_seconds / 5:
            break
    samples = []
    total = 0.0
    while len(samples) < MIN_RUNS or (total < MIN_SECONDS and len(samples) < MAX_RUNS):
        if budget_seconds and len(samples) >= 3 and total > budget_seconds:
            break
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


def build_mlc(model, ids_np, threads):
    """Our compiler: build the shared library for this T, then call it through ctypes.

    The library is rebuilt every time, so the numbers always describe the
    compiler as it is now. The build is not part of the timing.
    """
    import numpy as np

    seq_len = ids_np.shape[1]
    mlc = ROOT / "build" / "release" / "mlc"
    if not mlc.exists():
        raise SystemExit("build/release/mlc not found: build the compiler first (see README)")
    graphs = ROOT / "artifacts" / "graphs"
    out_dir = ROOT / "build" / "release" / "gen" / f"gpt2_T{seq_len}"
    subprocess.run(
        [str(mlc), "compile", str(graphs / f"gpt2_T{seq_len}.json"), "-o", str(out_dir)],
        check=True, stdout=subprocess.DEVNULL,
    )

    lib = ctypes.CDLL(str(out_dir / "model.so"))
    lib.mlc_open.argtypes = [ctypes.c_char_p]
    lib.mlc_last_error.restype = ctypes.c_char_p
    if lib.mlc_open(str(graphs / "gpt2_weights.bin").encode()) != 0:
        raise SystemExit(f"mlc_open failed: {lib.mlc_last_error().decode()}")

    # mlc_run takes an array of input pointers and an array of output
    # pointers. The arrays are built once; each call then just runs the model,
    # writing logits straight into `logits`.
    ids = np.ascontiguousarray(ids_np, dtype=np.int64)
    logits = np.empty((1, seq_len, 50257), dtype=np.float32)
    lib.mlc_output_bytes.restype = ctypes.c_int64
    assert lib.mlc_output_bytes(0) == logits.nbytes
    inputs = (ctypes.c_void_p * 1)(ids.ctypes.data)
    outputs = (ctypes.c_void_p * 1)(logits.ctypes.data)
    lib.mlc_run.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_void_p)]
    lib.mlc_run.restype = None

    def run():
        lib.mlc_run(inputs, outputs)
        return logits

    return run, lambda out: out


BUILDERS = {"eager": build_eager, "compile": build_compile, "ort": build_ort, "mlc": build_mlc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=BUILDERS, required=True)
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--seq-lens", type=int, nargs="+", required=True)
    parser.add_argument("--budget-seconds", type=float, default=0.0,
                        help="cap on measuring time per sequence length; 0 means no cap")
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

        samples = time_it(run, args.budget_seconds)
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
