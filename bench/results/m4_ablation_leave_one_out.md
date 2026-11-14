# Ablation: loop optimizations, on top of all graph passes (leave-one-out)

GPT-2 small, full forward pass, one thread unless the row says otherwise (4 for leave-one-out), compiled with `clang++ -O3 -mcpu=native`.
Every build was checked against the golden logits before it was timed.
Measured on battery power with Low Power Mode on (about half the normal single-core speed). Times are the minimum over the timed runs, with the median in parentheses;
the first run of each build (page faults on a fresh arena) is listed separately.

## T=64 (6 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 686 | 280 | 1.4 | 339.9 | 0.208 | 0.089 (0.104) | 1.00x |
| all but noalias | 686 | 280 | 1.4 | 339.9 | 0.118 | 0.106 (0.291) | 0.84x |
| all but vectorize | 686 | 280 | 1.4 | 339.9 | 1.780 | 1.978 (2.189) | 0.04x |
| all but vmath | 686 | 280 | 1.4 | 339.9 | 0.112 | 0.170 (0.206) | 0.52x |
| all but layout | 615 | 256 | 1.4 | 154.6 | 0.454 | 0.384 (0.557) | 0.23x |
| all but hoist | 686 | 329 | 12.4 | 0.0 | 0.248 | 0.220 (0.269) | 0.40x |
| all but regtile | 686 | 280 | 1.4 | 339.9 | 0.328 | 0.347 (0.383) | 0.26x |
| all but cachetile | 686 | 280 | 1.4 | 339.9 | 0.114 | 0.149 (0.205) | 0.60x |
| all but parallel | 686 | 280 | 1.4 | 339.9 | 0.326 | 0.292 (0.293) | 0.30x |

## T=256 (4 timed runs per build)

| optimizations | nodes | kernels | arena MB | persistent MB | first run s | time s | vs first row |
|---|---:|---:|---:|---:|---:|---:|---:|
| all | 686 | 280 | 8.7 | 340.5 | 0.364 | 0.361 (0.482) | 1.00x |
| all but noalias | 686 | 280 | 8.7 | 340.5 | 0.494 | 0.345 (0.397) | 1.05x |
| all but vectorize | 686 | 280 | 8.7 | 340.5 | 6.459 | 7.483 (8.820) | 0.05x |
| all but vmath | 686 | 280 | 8.7 | 340.5 | 0.559 | 0.381 (0.467) | 0.95x |
| all but layout | 615 | 256 | 8.7 | 155.2 | 1.766 | 2.917 (3.068) | 0.12x |
| all but hoist | 686 | 329 | 14.2 | 0.0 | 0.611 | 0.391 (0.556) | 0.92x |
| all but regtile | 686 | 280 | 8.7 | 340.5 | 1.103 | 1.040 (1.068) | 0.35x |
| all but cachetile | 686 | 280 | 8.7 | 340.5 | 0.518 | 0.359 (0.436) | 1.01x |
| all but parallel | 686 | 280 | 8.7 | 340.5 | 1.211 | 1.184 (1.202) | 0.30x |

Reproduce with `python bench/ablate.py --stage m4 --mode leave-one-out --seq-lens 64 256`.
