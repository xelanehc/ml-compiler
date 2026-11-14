# Baseline results

> **Measured on battery power.** Numbers may be throttled and noisy; rerun on AC before quoting them.

Full forward pass of GPT-2 small, token ids to logits, batch 1, fp32.
Each cell is **median** latency in milliseconds, with the minimum in parentheses.

## 1 thread

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 1 | 38.4 | 109.2 | 457.7 | 1121.7 |
| torch.compile | 1 | 35.4 | 93.7 | 267.1 | 1190.1 |
| ONNX Runtime | 1 | 70.8 | 281.0 | 1171.4 | 5343.1 |

## 4 threads

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 4 | 38.8 | 84.5 | 189.3 | 783.4 |
| torch.compile | 4 | 36.5 | 79.9 | 163.4 | 785.2 |
| ONNX Runtime | 4 | 21.2 | 137.5 | 572.0 | 2950.8 |

## Correctness

Every configuration was checked against the golden logits. Largest max-abs difference: 4.9e-04 (ONNX Runtime, T=256). Argmax matched at every position: yes.

## Environment

- date: 2026-11-12
- cpu: Apple M2
- performance_cores: 4
- efficiency_cores: 4
- os: Darwin 27.0.0 arm64
- power: battery
- python: 3.13.12
- torch: 2.15.0
- torch_blas: accelerate
- transformers: 5.22.2
- onnxruntime: 1.31.1
- numpy: 2.5.4

Reproduce with `python export/export_gpt2.py && python bench/run.py`.
