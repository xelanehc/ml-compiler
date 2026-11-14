# Matrix products alone, T=1024

Every distinct matrix product in GPT-2 small at this sequence length, timed on its own, in GFLOP/s
(2 x rows x inner x columns floating-point operations per product; best of several runs).
`mlc` is this compiler's generated kernel. PyTorch calls Apple Accelerate, which is not limited to
the CPU's SIMD units and does not follow the thread setting (STEERING.md section 7).
Measured on AC power.

## 1 thread

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [1024, 768] x [768, 2304] | 12 | 43.49 | 101.0 | 105.1 | 1462.6 |
| [12, 1024, 64] x [12, 64, 1024] | 12 | 19.33 | 83.9 | 100.5 | 387.5 |
| [12, 1024, 1024] x [12, 1024, 64] | 12 | 19.33 | 91.5 | 96.6 | 470.2 |
| [1024, 768] x [768, 768] | 12 | 14.50 | 94.4 | 105.4 | 1341.1 |
| [1024, 768] x [768, 3072] | 12 | 57.98 | 95.0 | 104.3 | 1196.0 |
| [1024, 3072] x [3072, 768] | 12 | 57.98 | 89.3 | 104.5 | 1128.9 |
| [1024, 768] x [768, 50257] | 1 | 79.05 | 99.8 | 102.1 | 944.3 |
| **all, weighted by use** | | 291.65 | **94.8** | **103.1** | **925.6** |

## 4 threads

| product (A x B) | uses | GFLOP | mlc | ONNX Runtime | PyTorch (Accelerate) |
|---|---:|---:|---:|---:|---:|
| [1024, 768] x [768, 2304] | 12 | 43.49 | 370.3 | 395.3 | 1478.8 |
| [12, 1024, 64] x [12, 64, 1024] | 12 | 19.33 | 308.7 | 361.4 | 397.2 |
| [12, 1024, 1024] x [12, 1024, 64] | 12 | 19.33 | 346.3 | 359.7 | 474.8 |
| [1024, 768] x [768, 768] | 12 | 14.50 | 364.5 | 370.9 | 1313.4 |
| [1024, 768] x [768, 3072] | 12 | 57.98 | 360.4 | 392.1 | 1288.7 |
| [1024, 3072] x [3072, 768] | 12 | 57.98 | 289.2 | 358.5 | 1128.2 |
| [1024, 768] x [768, 50257] | 1 | 79.05 | 368.3 | 377.1 | 947.4 |
| **all, weighted by use** | | 291.65 | **342.5** | **376.1** | **941.9** |

Reproduce with `python bench/gemm.py --seq-len 1024`.
