# Ablation: graph passes, every loop optimization off (cumulative)

GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on AC power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=16 (10 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 191.6 | 0.0 | 1.031 | 0.634 (0.665) | 1.00x |
| + simplify | 951 | 453 | 187.9 | 0.0 | 0.695 | 0.636 (0.638) | 1.00x |
| + fold | 902 | 426 | 187.9 | 0.0 | 0.710 | 0.643 (0.645) | 0.99x |
| + cse | 780 | 415 | 187.7 | 0.0 | 0.698 | 0.633 (0.636) | 1.00x |
| + gelu | 691 | 331 | 171.2 | 0.0 | 0.689 | 0.642 (0.646) | 0.99x |
| + attention | 615 | 258 | 170.8 | 0.0 | 0.697 | 0.633 (0.636) | 1.00x |
| + memory | 615 | 258 | 154.4 | 0.0 | 0.683 | 0.637 (0.641) | 0.99x |

## T=64 (6 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 314.8 | 0.0 | 2.348 | 2.284 (2.288) | 1.00x |
| + simplify | 951 | 453 | 298.2 | 0.0 | 2.381 | 2.320 (2.324) | 0.98x |
| + fold | 902 | 426 | 298.0 | 0.0 | 2.342 | 2.286 (2.292) | 1.00x |
| + cse | 780 | 415 | 295.9 | 0.0 | 2.353 | 2.282 (2.292) | 1.00x |
| + gelu | 691 | 331 | 229.8 | 0.0 | 2.360 | 2.308 (2.314) | 0.99x |
| + attention | 615 | 258 | 223.7 | 0.0 | 2.335 | 2.279 (2.284) | 1.00x |
| + memory | 615 | 258 | 154.6 | 0.0 | 2.325 | 2.280 (2.285) | 1.00x |

## T=256 (4 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 982.5 | 0.0 | 9.011 | 8.832 (9.402) | 1.00x |
| + simplify | 951 | 453 | 888.0 | 0.0 | 9.612 | 8.846 (8.919) | 1.00x |
| + fold | 902 | 426 | 884.6 | 0.0 | 9.164 | 9.017 (9.067) | 0.98x |
| + cse | 780 | 415 | 850.0 | 0.0 | 9.323 | 9.195 (9.207) | 0.96x |
| + gelu | 691 | 331 | 585.8 | 0.0 | 9.049 | 8.961 (8.980) | 0.99x |
| + attention | 615 | 258 | 488.2 | 0.0 | 9.262 | 9.125 (9.144) | 0.97x |
| + memory | 615 | 258 | 155.2 | 0.0 | 9.140 | 9.096 (9.100) | 0.97x |

Reproduce with `python bench/ablate.py --stage m3 --mode cumulative --seq-lens 16 64 256`.
