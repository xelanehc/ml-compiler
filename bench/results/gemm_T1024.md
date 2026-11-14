# Matrix products alone, T=1024

Every distinct matrix product in GPT-2 small at this sequence length, timed on its own, in GFLOP/s
(2 x rows x inner x columns floating-point operations per product; best of several runs).
`mlc` is this compiler's generated kernel. PyTorch calls Apple Accelerate, which is not limited to
the CPU's SIMD units and does not follow the thread setting (STEERING.md section 7).
Measured on battery power with Low Power Mode on (about half the normal single-core speed).

## 1 thread

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [1024, 768] x [768, 2304] | 12 | 43.49 | 58.5 | 25.6 | 770.4 |
| [12, 1024, 64] x [12, 64, 1024] | 12 | 19.33 | 49.3 | 59.5 | 239.3 |
| [12, 1024, 1024] x [12, 1024, 64] | 12 | 19.33 | 57.6 | 58.2 | 368.2 |
| [1024, 768] x [768, 768] | 12 | 14.50 | 58.0 | 61.1 | 836.4 |
| [1024, 768] x [768, 3072] | 12 | 57.98 | 58.3 | 60.6 | 739.3 |
| [1024, 3072] x [3072, 768] | 12 | 57.98 | 55.1 | 60.3 | 730.3 |
| [1024, 768] x [768, 50257] | 1 | 79.05 | 58.4 | 59.8 | 593.9 |
| **all, weighted by use** | | 291.65 | **56.9** | **50.0** | **585.7** |

## 4 threads

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [1024, 768] x [768, 2304] | 12 | 43.49 | 228.3 | 243.0 | 930.2 |
| [12, 1024, 64] x [12, 64, 1024] | 12 | 19.33 | 193.3 | 235.7 | 257.2 |
| [12, 1024, 1024] x [12, 1024, 64] | 12 | 19.33 | 227.7 | 196.2 | 374.9 |
| [1024, 768] x [768, 768] | 12 | 14.50 | 225.1 | 122.1 | 818.5 |
| [1024, 768] x [768, 3072] | 12 | 57.98 | 228.8 | 183.7 | 796.0 |
| [1024, 3072] x [3072, 768] | 12 | 57.98 | 201.2 | 193.9 | 700.4 |
| [1024, 768] x [768, 50257] | 1 | 79.05 | 234.5 | 116.7 | 627.8 |
| **all, weighted by use** | | 291.65 | **221.2** | **164.7** | **617.0** |

Reproduce with `python bench/gemm.py --seq-len 1024`.
