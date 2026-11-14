# Ablation: loop optimizations, on top of all graph passes (cumulative)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power with Low Power Mode on (about half the normal single-core speed). Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=16 (10 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 154.4 | 0.0 | 1.563 | 1.050 (1.151) | 1.00x |
| + noalias | 615 | 258 | 154.4 | 0.0 | 1.103 | 1.047 (1.075) | 1.00x |
| + vectorize | 615 | 258 | 154.4 | 0.0 | 0.494 | 0.433 (0.442) | 2.42x |
| + vmath | 615 | 258 | 154.4 | 0.0 | 0.487 | 0.434 (0.438) | 2.42x |
| + layout | 686 | 329 | 11.9 | 0.0 | 0.393 | 0.356 (0.362) | 2.95x |
| + hoist | 686 | 280 | 0.3 | 339.8 | 0.269 | 0.247 (0.248) | 4.25x |
| + regtile | 686 | 280 | 0.3 | 339.8 | 0.092 | 0.075 (0.078) | 14.00x |
| + cachetile | 686 | 280 | 0.3 | 339.8 | 0.094 | 0.075 (0.078) | 14.00x |
| + parallel, 1 thread | 686 | 280 | 0.3 | 339.8 | 0.092 | 0.075 (0.076) | 14.00x |
| + parallel, 4 threads | 686 | 280 | 0.3 | 339.8 | 0.034 | 0.027 (0.070) | 38.89x |

## T=64 (6 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 154.6 | 0.0 | 3.997 | 3.876 (3.918) | 1.00x |
| + noalias | 615 | 258 | 154.6 | 0.0 | 3.984 | 3.884 (3.904) | 1.00x |
| + vectorize | 615 | 258 | 154.6 | 0.0 | 1.504 | 1.435 (1.447) | 2.70x |
| + vmath | 615 | 258 | 154.6 | 0.0 | 1.485 | 1.406 (1.412) | 2.76x |
| + layout | 686 | 329 | 12.4 | 0.0 | 1.165 | 1.111 (1.117) | 3.49x |
| + hoist | 686 | 280 | 1.4 | 339.9 | 1.017 | 0.996 (0.998) | 3.89x |
| + regtile | 686 | 280 | 1.4 | 339.9 | 0.318 | 0.296 (0.305) | 13.09x |
| + cachetile | 686 | 280 | 1.4 | 339.9 | 0.309 | 0.292 (0.295) | 13.27x |
| + parallel, 1 thread | 686 | 280 | 1.4 | 339.9 | 0.308 | 0.294 (0.298) | 13.18x |
| + parallel, 4 threads | 686 | 280 | 1.4 | 339.9 | 0.190 | 0.086 (0.104) | 45.07x |

## T=256 (4 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 155.2 | 0.0 | 15.954 | 15.719 (16.843) | 1.00x |
| + noalias | 615 | 258 | 155.2 | 0.0 | 15.708 | 15.656 (15.676) | 1.00x |
| + vectorize | 615 | 258 | 155.2 | 0.0 | 5.598 | 5.552 (5.560) | 2.83x |
| + vmath | 615 | 258 | 155.2 | 0.0 | 5.503 | 5.608 (6.259) | 2.80x |
| + layout | 686 | 329 | 14.2 | 0.0 | 4.219 | 4.183 (4.186) | 3.76x |
| + hoist | 686 | 280 | 8.7 | 340.5 | 4.296 | 4.106 (4.194) | 3.83x |
| + regtile | 686 | 280 | 8.7 | 340.5 | 1.257 | 1.221 (1.230) | 12.87x |
| + cachetile | 686 | 280 | 8.7 | 340.5 | 1.210 | 1.178 (1.182) | 13.34x |
| + parallel, 1 thread | 686 | 280 | 8.7 | 340.5 | 1.226 | 1.185 (1.202) | 13.26x |
| + parallel, 4 threads | 686 | 280 | 8.7 | 340.5 | 0.467 | 0.345 (0.444) | 45.56x |

Reproduce with `python bench/ablate.py --stage m4 --mode cumulative --seq-lens 16 64 256`.
