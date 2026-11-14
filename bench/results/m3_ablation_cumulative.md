# Pass ablation (cumulative)

GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=16 (10 timed runs per build)

| passes | nodes | kernels | arena MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 191.6 | 1.10 | 1.044 (1.050) | 1.00x |
| + simplify | 951 | 453 | 187.9 | 1.11 | 1.044 (1.057) | 1.00x |
| + fold | 902 | 426 | 187.9 | 1.14 | 1.052 (1.065) | 0.99x |
| + cse | 780 | 415 | 187.7 | 1.13 | 1.049 (1.054) | 1.00x |
| + gelu | 691 | 331 | 171.2 | 1.12 | 1.040 (1.051) | 1.00x |
| + attention | 615 | 258 | 170.8 | 1.17 | 1.050 (1.057) | 0.99x |
| + memory | 615 | 258 | 154.4 | 1.13 | 1.049 (1.064) | 1.00x |

## T=64 (6 timed runs per build)

| passes | nodes | kernels | arena MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 314.8 | 4.07 | 3.937 (3.968) | 1.00x |
| + simplify | 951 | 453 | 298.2 | 4.03 | 3.916 (3.940) | 1.01x |
| + fold | 902 | 426 | 298.0 | 4.01 | 3.925 (3.941) | 1.00x |
| + cse | 780 | 415 | 295.9 | 4.01 | 3.892 (3.932) | 1.01x |
| + gelu | 691 | 331 | 229.8 | 4.18 | 4.048 (4.096) | 0.97x |
| + attention | 615 | 258 | 223.7 | 4.07 | 3.949 (4.004) | 1.00x |
| + memory | 615 | 258 | 154.6 | 4.04 | 3.978 (3.992) | 0.99x |

## T=256 (4 timed runs per build)

| passes | nodes | kernels | arena MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 982.5 | 15.48 | 15.247 (15.316) | 1.00x |
| + simplify | 951 | 453 | 888.0 | 15.48 | 15.290 (15.406) | 1.00x |
| + fold | 902 | 426 | 884.6 | 15.77 | 15.520 (15.592) | 0.98x |
| + cse | 780 | 415 | 850.0 | 15.80 | 15.613 (15.638) | 0.98x |
| + gelu | 691 | 331 | 585.8 | 15.65 | 15.470 (15.630) | 0.99x |
| + attention | 615 | 258 | 488.2 | 15.63 | 15.645 (16.016) | 0.97x |
| + memory | 615 | 258 | 155.2 | 15.56 | 15.460 (15.746) | 0.99x |

Reproduce with `python bench/ablate.py --mode cumulative --seq-lens 16 64 256`.
