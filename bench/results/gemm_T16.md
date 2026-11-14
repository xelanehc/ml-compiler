# Matrix products alone, T=16

Every distinct matrix product in GPT-2 small at this sequence length, timed on its own, in GFLOP/s
(2 x rows x inner x columns floating-point operations per product; best of several runs).
`mlc` is this compiler's generated kernel. PyTorch calls Apple Accelerate, which is not limited to
the CPU's SIMD units and does not follow the thread setting (STEERING.md section 7).
Measured on battery power with Low Power Mode on (about half the normal single-core speed).

## 1 thread

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [16, 768] x [768, 2304] | 12 | 0.68 | 56.1 | 47.4 | 760.0 |
| [12, 16, 64] x [12, 64, 16] | 12 | 0.00 | 47.9 | 26.1 | 23.9 |
| [12, 16, 16] x [12, 16, 64] | 12 | 0.00 | 34.1 | 25.0 | 28.9 |
| [16, 768] x [768, 768] | 12 | 0.23 | 55.9 | 47.1 | 797.5 |
| [16, 768] x [768, 3072] | 12 | 0.91 | 55.3 | 47.5 | 632.2 |
| [16, 3072] x [3072, 768] | 12 | 0.91 | 51.5 | 47.4 | 833.8 |
| [16, 768] x [768, 50257] | 1 | 1.24 | 55.7 | 37.5 | 119.1 |
| **all, weighted by use** | | 3.96 | **54.6** | **43.7** | **274.5** |

## 4 threads

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [16, 768] x [768, 2304] | 12 | 0.68 | 198.3 | 183.2 | 757.5 |
| [12, 16, 64] x [12, 64, 16] | 12 | 0.00 | 111.0 | 40.2 | 23.7 |
| [12, 16, 16] x [12, 16, 64] | 12 | 0.00 | 98.3 | 40.3 | 28.8 |
| [16, 768] x [768, 768] | 12 | 0.23 | 192.3 | 178.3 | 793.3 |
| [16, 768] x [768, 3072] | 12 | 0.91 | 191.4 | 90.8 | 646.2 |
| [16, 3072] x [3072, 768] | 12 | 0.91 | 167.4 | 152.2 | 813.3 |
| [16, 768] x [768, 50257] | 1 | 1.24 | 200.8 | 67.5 | 137.0 |
| **all, weighted by use** | | 3.96 | **188.8** | **100.5** | **303.0** |

Reproduce with `python bench/gemm.py --seq-len 16`.
