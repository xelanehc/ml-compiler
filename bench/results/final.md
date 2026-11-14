# Results: final

Full forward pass of GPT-2 small, token ids to logits, batch 1, fp32.
Each cell is **median** latency in milliseconds, with the minimum in parentheses.

## 1 thread

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 27.8 (27.3) | 63.8 (61.1) | 183.2 (174.6) | 705.5 (679.7) |
| torch.compile | 27.0 (26.0) | 59.8 (58.3) | 160.3 (154.7) | 749.4 (731.0) |
| ONNX Runtime | 43.2 (41.8) | 166.9 (163.2) | 700.8 (687.0) | 3219.4 (3202.5) |
| mlc (ours) | 50.3 (45.9) | 180.1 (173.7) | 708.7 (688.3) | 3202.0 (3177.8) |
| mlc, autotuned (ours) | 45.8 (44.9) | 177.5 (172.4) | 707.4 (692.7) | 3191.8 (3162.3) |

## 4 threads

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 29.6 (28.7) | 61.5 (59.0) | 144.8 (137.4) | 474.8 (450.7) |
| torch.compile | 27.4 (26.9) | 55.2 (53.0) | 121.8 (118.1) | 473.3 (451.1) |
| ONNX Runtime | 14.1 (13.9) | 47.0 (46.2) | 194.8 (185.0) | 1027.3 (994.1) |
| mlc (ours) | 22.2 (18.5) | 68.5 (56.4) | 218.0 (202.1) | 1032.7 (997.9) |
| mlc, autotuned (ours) | 23.1 (18.6) | 66.0 (55.9) | 239.1 (212.3) | 1032.6 (994.5) |

## Correctness

Every configuration was checked against the golden logits. Largest max-abs difference: 4.9e-04 (ONNX Runtime, T=256). Argmax matched at every position: yes.

## Environment

- date: 2026-11-14
- cpu: Apple M2
- performance_cores: 4
- efficiency_cores: 4
- os: Darwin 27.0.0 arm64
- power: AC
- low_power_mode: off
- python: 3.13.12
- torch: 2.15.0
- torch_blas: accelerate
- transformers: 5.22.2
- onnxruntime: 1.31.1
- numpy: 2.5.4

Reproduce with `python export/export_gpt2.py && python bench/run.py`.
