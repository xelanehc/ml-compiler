# Pass ablation (cumulative)

GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=1024 (2 timed runs per build)

| passes | nodes | kernels | arena MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 6452.9 | 67.51 | 65.211 (65.412) | 1.00x |
| + memory | 615 | 258 | 157.5 | 63.95 | 63.982 (64.527) | 1.02x |

Reproduce with `python bench/ablate.py --mode cumulative --seq-lens 1024`.
