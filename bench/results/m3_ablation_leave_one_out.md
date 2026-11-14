# Pass ablation (leave-one-out)

GPT-2 small, full forward pass, one thread, compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=64 (6 timed runs per build)

| passes | nodes | kernels | arena MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|
| all | 615 | 258 | 154.6 | 4.07 | 3.946 (4.010) | 1.00x |
| all but simplify | 713 | 342 | 154.6 | 4.03 | 3.966 (3.983) | 0.99x |
| all but fold | 616 | 258 | 154.6 | 4.04 | 3.966 (4.003) | 0.99x |
| all but cse | 638 | 258 | 154.6 | 4.04 | 3.973 (4.009) | 0.99x |
| all but gelu | 704 | 342 | 154.6 | 4.03 | 3.896 (3.931) | 1.01x |
| all but attention | 691 | 331 | 154.8 | 4.02 | 3.889 (3.918) | 1.01x |
| all but memory | 615 | 258 | 223.7 | 4.10 | 3.953 (3.991) | 1.00x |

Reproduce with `python bench/ablate.py --mode leave-one-out --seq-lens 64`.
