# Ablation: loop optimizations, on top of all graph passes (cumulative)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on AC power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=16 (10 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 154.4 | 0.0 | 0.685 | 0.638 (0.651) | 1.00x |
| + noalias | 615 | 258 | 154.4 | 0.0 | 0.698 | 0.653 (0.687) | 0.98x |
| + vectorize | 615 | 258 | 154.4 | 0.0 | 0.358 | 0.280 (0.290) | 2.28x |
| + vmath | 615 | 258 | 154.4 | 0.0 | 0.341 | 0.281 (0.292) | 2.27x |
| + layout | 686 | 329 | 11.9 | 0.0 | 0.242 | 0.211 (0.219) | 3.02x |
| + hoist | 686 | 280 | 0.3 | 339.8 | 0.159 | 0.145 (0.153) | 4.38x |
| + regtile | 686 | 280 | 0.3 | 339.8 | 0.063 | 0.046 (0.047) | 13.84x |
| + cachetile | 686 | 280 | 0.3 | 339.8 | 0.060 | 0.046 (0.046) | 13.84x |
| + parallel, 1 thread | 686 | 280 | 0.3 | 339.8 | 0.064 | 0.046 (0.049) | 13.90x |
| + parallel, 4 threads | 686 | 280 | 0.3 | 339.8 | 0.024 | 0.020 (0.040) | 31.57x |

## T=64 (6 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 154.6 | 0.0 | 2.407 | 2.310 (2.315) | 1.00x |
| + noalias | 615 | 258 | 154.6 | 0.0 | 2.336 | 2.309 (2.320) | 1.00x |
| + vectorize | 615 | 258 | 154.6 | 0.0 | 0.919 | 0.872 (0.879) | 2.65x |
| + vmath | 615 | 258 | 154.6 | 0.0 | 0.893 | 0.863 (0.874) | 2.68x |
| + layout | 686 | 329 | 12.4 | 0.0 | 0.692 | 0.658 (0.665) | 3.51x |
| + hoist | 686 | 280 | 1.4 | 339.9 | 0.626 | 0.586 (0.594) | 3.95x |
| + regtile | 686 | 280 | 1.4 | 339.9 | 0.190 | 0.175 (0.182) | 13.18x |
| + cachetile | 686 | 280 | 1.4 | 339.9 | 0.193 | 0.174 (0.181) | 13.31x |
| + parallel, 1 thread | 686 | 280 | 1.4 | 339.9 | 0.194 | 0.173 (0.177) | 13.32x |
| + parallel, 4 threads | 686 | 280 | 1.4 | 339.9 | 0.062 | 0.058 (0.112) | 39.90x |

## T=256 (4 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 155.2 | 0.0 | 9.254 | 9.209 (9.236) | 1.00x |
| + noalias | 615 | 258 | 155.2 | 0.0 | 9.321 | 9.198 (9.245) | 1.00x |
| + vectorize | 615 | 258 | 155.2 | 0.0 | 3.331 | 3.297 (3.320) | 2.79x |
| + vmath | 615 | 258 | 155.2 | 0.0 | 3.957 | 3.396 (3.514) | 2.71x |
| + layout | 686 | 329 | 14.2 | 0.0 | 2.523 | 2.495 (2.535) | 3.69x |
| + hoist | 686 | 280 | 8.7 | 340.5 | 2.490 | 2.427 (2.434) | 3.79x |
| + regtile | 686 | 280 | 8.7 | 340.5 | 0.741 | 0.711 (0.726) | 12.96x |
| + cachetile | 686 | 280 | 8.7 | 340.5 | 0.720 | 0.688 (0.708) | 13.39x |
| + parallel, 1 thread | 686 | 280 | 8.7 | 340.5 | 0.729 | 0.689 (0.700) | 13.38x |
| + parallel, 4 threads | 686 | 280 | 8.7 | 340.5 | 0.250 | 0.212 (0.246) | 43.40x |

Reproduce with `python bench/ablate.py --stage m4 --mode cumulative --seq-lens 16 64 256`.
