# Results: mlc_m2

Full forward pass of GPT-2 small, token ids to logits, batch 1, fp32.
Each cell is **median** latency in milliseconds, with the minimum in parentheses.

## 1 thread

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| mlc (ours) | 649.7 (633.8) | 2319.3 (2301.6) | 8968.8 (8818.1) | 38107.5 (37899.8) |

## Correctness

Every configuration was checked against the golden logits. Largest max-abs difference: 2.1e-04 (mlc (ours), T=256). Argmax matched at every position: yes.

## Environment

- date: 2026-11-12
- cpu: Apple M2
- performance_cores: 4
- efficiency_cores: 4
- os: Darwin 27.0.0 arm64
- power: AC
- python: 3.13.12
- torch: 2.15.0
- torch_blas: accelerate
- transformers: 5.22.2
- onnxruntime: 1.31.1
- numpy: 2.5.4

Reproduce with `python export/export_gpt2.py && python bench/run.py`.

Cells measured with fewer than 30 runs because of the time cap: mlc (ours) T=256 (14 runs), mlc (ours) T=1024 (4 runs).
