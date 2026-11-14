# Matrix products alone, T=16

Every distinct matrix product in GPT-2 small at this sequence length, timed on its own, in GFLOP/s
(2 x rows x inner x columns floating-point operations per product; best of several runs).
`mlc` is this compiler's generated kernel. PyTorch calls Apple Accelerate, which is not limited to
the CPU's SIMD units and does not follow the thread setting (STEERING.md section 7).
Measured on AC power.

## 1 thread

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [16, 768] x [768, 2304] | 12 | 0.68 | 93.8 | 82.0 | 1257.1 |
| [12, 16, 64] x [12, 64, 16] | 12 | 0.00 | 82.8 | 45.6 | 40.2 |
| [12, 16, 16] x [12, 16, 64] | 12 | 0.00 | 58.3 | 43.9 | 49.4 |
| [16, 768] x [768, 768] | 12 | 0.23 | 93.8 | 81.7 | 1286.9 |
| [16, 768] x [768, 3072] | 12 | 0.91 | 92.2 | 81.8 | 1024.9 |
| [16, 3072] x [3072, 768] | 12 | 0.91 | 82.8 | 81.5 | 1151.2 |
| [16, 768] x [768, 50257] | 1 | 1.24 | 92.1 | 56.4 | 157.1 |
| **all, weighted by use** | | 3.96 | **90.1** | **71.6** | **378.7** |

## 4 threads

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [16, 768] x [768, 2304] | 12 | 0.68 | 276.7 | 299.4 | 1263.0 |
| [12, 16, 64] x [12, 64, 16] | 12 | 0.00 | 147.5 | 68.4 | 40.3 |
| [12, 16, 16] x [12, 16, 64] | 12 | 0.00 | 143.0 | 66.0 | 48.9 |
| [16, 768] x [768, 768] | 12 | 0.23 | 273.5 | 290.9 | 1286.9 |
| [16, 768] x [768, 3072] | 12 | 0.91 | 275.1 | 299.0 | 1051.6 |
| [16, 3072] x [3072, 768] | 12 | 0.91 | 238.5 | 304.6 | 1230.9 |
| [16, 768] x [768, 50257] | 1 | 1.24 | 286.5 | 165.5 | 165.0 |
| **all, weighted by use** | | 3.96 | **268.6** | **237.9** | **395.8** |

Reproduce with `python bench/gemm.py --seq-len 16`.
