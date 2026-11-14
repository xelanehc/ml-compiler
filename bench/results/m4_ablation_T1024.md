# Ablation: loop optimizations, on top of all graph passes (cumulative)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on AC power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=1024 (3 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 157.5 | 0.0 | 37.938 | 37.401 (37.404) | 1.00x |
| + cachetile | 686 | 280 | 72.4 | 342.9 | 3.240 | 3.165 (3.168) | 11.82x |
| + parallel, 1 thread | 686 | 280 | 72.4 | 342.9 | 3.222 | 3.168 (3.185) | 11.81x |
| + parallel, 4 threads | 686 | 280 | 72.4 | 342.9 | 1.243 | 1.337 (1.548) | 27.98x |

Reproduce with `python bench/ablate.py --stage m4 --mode cumulative --seq-lens 1024`.
