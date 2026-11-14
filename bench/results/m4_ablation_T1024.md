# Ablation: loop optimizations, on top of all graph passes (cumulative)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power with Low Power Mode on (about half the normal single-core speed). Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=1024 (2 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| graph passes only (M3) | 615 | 258 | 157.5 | 0.0 | 63.836 | 63.582 (63.694) | 1.00x |
| + cachetile | 686 | 280 | 72.4 | 342.9 | 5.410 | 5.372 (5.380) | 11.84x |
| + parallel, 1 thread | 686 | 280 | 72.4 | 342.9 | 5.418 | 5.370 (5.692) | 11.84x |
| + parallel, 4 threads | 686 | 280 | 72.4 | 342.9 | 1.635 | 1.547 (4.053) | 41.10x |

Reproduce with `python bench/ablate.py --stage m4 --mode cumulative --seq-lens 1024`.
