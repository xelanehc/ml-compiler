# Ablation: loop optimizations, on top of all graph passes (leave-one-out)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on AC power. Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=64 (6 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 686 | 280 | 1.4 | 339.9 | 0.105 | 0.073 (0.082) | 1.00x |
| all but noalias | 686 | 280 | 1.4 | 339.9 | 0.087 | 0.058 (0.067) | 1.26x |
| all but vectorize | 686 | 280 | 1.4 | 339.9 | 0.990 | 0.975 (0.990) | 0.07x |
| all but vmath | 686 | 280 | 1.4 | 339.9 | 0.096 | 0.062 (0.070) | 1.17x |
| all but layout | 615 | 256 | 1.4 | 154.6 | 0.324 | 0.301 (0.337) | 0.24x |
| all but hoist | 686 | 329 | 12.4 | 0.0 | 0.132 | 0.104 (0.119) | 0.70x |
| all but regtile | 686 | 280 | 1.4 | 339.9 | 0.173 | 0.146 (0.161) | 0.50x |
| all but cachetile | 686 | 280 | 1.4 | 339.9 | 0.066 | 0.072 (0.079) | 1.01x |
| all but parallel | 686 | 280 | 1.4 | 339.9 | 0.191 | 0.173 (0.179) | 0.42x |

## T=256 (4 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 686 | 280 | 8.7 | 340.5 | 0.246 | 0.231 (0.246) | 1.00x |
| all but noalias | 686 | 280 | 8.7 | 340.5 | 0.229 | 0.240 (0.286) | 0.96x |
| all but vectorize | 686 | 280 | 8.7 | 340.5 | 4.030 | 3.994 (4.033) | 0.06x |
| all but vmath | 686 | 280 | 8.7 | 340.5 | 0.286 | 0.255 (0.336) | 0.90x |
| all but layout | 615 | 256 | 8.7 | 155.2 | 1.231 | 1.158 (1.188) | 0.20x |
| all but hoist | 686 | 329 | 14.2 | 0.0 | 0.382 | 0.261 (0.361) | 0.88x |
| all but regtile | 686 | 280 | 8.7 | 340.5 | 0.638 | 0.638 (0.646) | 0.36x |
| all but cachetile | 686 | 280 | 8.7 | 340.5 | 0.375 | 0.257 (0.319) | 0.90x |
| all but parallel | 686 | 280 | 8.7 | 340.5 | 0.726 | 0.690 (0.700) | 0.33x |

Reproduce with `python bench/ablate.py --stage m4 --mode leave-one-out --seq-lens 64 256`.
