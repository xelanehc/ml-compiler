# Ablation: graph passes, every loop optimization off (cumulative)

GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on AC power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=1024 (2 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| none | 1062 | 541 | 6452.9 | 0.0 | 38.452 | 38.394 (38.506) | 1.00x |
| + memory | 615 | 258 | 157.5 | 0.0 | 37.756 | 37.689 (37.696) | 1.02x |

Reproduce with `python bench/ablate.py --stage m3 --mode cumulative --seq-lens 1024`.
