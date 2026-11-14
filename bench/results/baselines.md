# Baseline results

> **Measured on battery power.** Numbers may be throttled and noisy; rerun on AC before quoting them.

Full forward pass of GPT-2 small, token ids to logits, batch 1, fp32.
Each cell is **median** latency in milliseconds, with the minimum in parentheses.

## 1 thread

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 36.8 (32.5) | 96.1 (83.9) | 282.6 (263.4) | 1071.0 (1048.5) |
| torch.compile | 41.5 (33.1) | 94.4 (78.2) | 267.5 (242.2) | 1165.7 (1139.3) |
| ONNX Runtime | 70.8 (70.6) | 281.2 (278.3) | 1170.3 (1147.9) | 5335.9 (5317.7) |

## 4 threads

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 39.0 (33.8) | 103.5 (81.9) | 250.7 (200.2) | 836.8 (740.6) |
| torch.compile | 34.9 (30.9) | 79.0 (66.2) | 182.3 (159.7) | 804.4 (759.5) |
| ONNX Runtime | 21.2 (20.3) | 81.2 (73.6) | 552.4 (466.9) | 3027.7 (2848.1) |

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
