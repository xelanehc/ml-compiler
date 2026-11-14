# Build log

## M0: baselines and harness (2026-11-11)

**Built**

- `export/model.py`: the shared model wrapper (ids to logits, cache off) and the fixed inputs.
- `export/export_gpt2.py`: saves golden ids and logits per `T`, runs `torch.export` plus decomposition, checks the exported program against eager, writes `docs/export_ops.md`.
- `bench/worker.py` and `bench/run.py`: time PyTorch eager, `torch.compile`, and ONNX Runtime in separate processes; write `bench/results/baselines.md`.

**Numbers** (median ms, measured on battery, so provisional)

| system | threads | T=16 | T=64 | T=256 | T=1024 |
|---|---|---:|---:|---:|---:|
| PyTorch eager | 1 | 38.4 | 109.2 | 457.7 | 1121.7 |
| torch.compile | 1 | 35.4 | 93.7 | 267.1 | 1190.1 |
| ONNX Runtime | 1 | 70.8 | 281.0 | 1171.4 | 5343.1 |
| PyTorch eager | 4 | 38.8 | 84.5 | 189.3 | 783.4 |
| torch.compile | 4 | 36.5 | 79.9 | 163.4 | 785.2 |
| ONNX Runtime | 4 | 21.2 | 137.5 | 572.0 | 2950.8 |

Export: 517 nodes as traced, 852 after decomposition to core ATen, 37 distinct ops, identical across all four `T`. Exported program matches eager to about 1e-4.

**Surprises**

- PyTorch's matrix multiply runs at about 710 GFLOP/s on one thread (1024x768 times 768x3072, CPU time / wall time = 1.00) and 768 with four. One core's NEON units cannot do that, so Accelerate is using hardware outside ordinary SIMD, and the thread setting does not control it.
- Even so, PyTorch's whole-model rate at `T=1024` is only around 260 GFLOP/s. Most of its time at long sequences is not GEMM.
- ONNX Runtime beats PyTorch at `T=16` with 4 threads (21 ms against 38 ms) and loses badly at long `T`.
- `torch.compile` is barely faster than eager here. Both end up in the same Accelerate call.
- Timing noise on battery is large: several configurations have a p10 to p90 spread above 20% of the median.

**Open**

- Rerun `bench/run.py` on AC power and replace the provisional table.
