# Results and what they mean

GPT-2 small (124M parameters), full forward pass from token ids to logits, batch 1, fp32, on an Apple M2 (4 performance cores), AC power, Low Power Mode off. Every system reproduces PyTorch's logits within 1e-3 with the same argmax at every position. The tables come from the files under `bench/results/`; each names the command that regenerates it.

## The comparison

Median latency in ms (`bench/results/final.md`, all systems in one session):

| 1 thread | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 27.8 | 63.8 | 183.2 | 705.5 |
| torch.compile | 27.0 | 59.8 | 160.3 | 749.4 |
| ONNX Runtime | 43.2 | 166.9 | 700.8 | 3219.4 |
| mlc | 50.3 | 180.1 | 708.7 | 3202.0 |
| mlc, autotuned | 45.8 | 177.5 | 707.4 | 3191.8 |

| 4 threads | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 29.6 | 61.5 | 144.8 | 474.8 |
| torch.compile | 27.4 | 55.2 | 121.8 | 473.3 |
| ONNX Runtime | 14.1 | 47.0 | 194.8 | 1027.3 |
| mlc | 22.2 | 68.5 | 218.0 | 1032.7 |
| mlc, autotuned | 23.1 | 66.0 | 239.1 | 1032.6 |

Read plainly:

- **Against ONNX Runtime, the peer** (its kernels use the same NEON units ours do). One thread: 1% ahead at T=1024 and 1 to 6% behind at the shorter lengths, taking the better of our two builds. Four threads: level at T=1024, and 12%, 40%, and 57% behind at T=256, 64, and 16. The goal was to beat it at every length and thread count. We match it at the longest length and lose elsewhere.
- **Against PyTorch.** PyTorch's matrix products go through Apple Accelerate, which is not limited to the NEON units and ignores the thread setting. It is 1.6 to 4.5 times faster than us on one thread. On four threads we are ahead at T=16 (22 against 28 ms) and behind from T=64 up.
- **Against where we started.** The first correct build (M2) took 38 s at T=1024 on one thread. This one takes 3.2 s on one thread and 1.0 s on four.
- **Autotuned against default.** Within noise in most cells; the differences go both ways.

## Where the time goes

Seconds per kind of operation (`bench/results/per_op.md`), one thread:

| | T=16 | T=1024 |
|---|---:|---:|
| dense layers (`gemm`, 48) | 68.2% | 56.5% |
| attention products and output projection (`matmul`, 25) | 29.9% | 39.5% |
| softmax | 0.1% | 1.7% |
| GELU | 1.1% | 1.1% |
| permutes, slices, layer norms, adds, scalings | 0.7% | 1.2% |

Matrix products are 96 to 98% of every run. The comparison with ONNX Runtime is a comparison of matrix-product kernels and of how fast each system gets weights out of memory.

## The matrix product alone

GFLOP/s, each distinct product of the model timed on its own and weighted by use (`bench/results/gemm_T16.md`, `gemm_T1024.md`):

| | T=16, 1 thr | T=16, 4 thr | T=1024, 1 thr | T=1024, 4 thr |
|---|---:|---:|---:|---:|
| mlc | 90.1 | 268.6 | 94.8 | 342.5 |
| ONNX Runtime | 71.6 | 237.9 | 103.1 | 376.1 |
| PyTorch (Accelerate) | 378.7 | 395.8 | 925.6 | 941.9 |

- At T=1024 ONNX Runtime's kernel is 9% faster than ours on its best shapes (about 105 against 95 to 101 GFLOP/s). An M4 experiment predicted this: an inner loop in outer-product form, which needs packed operands, peaks about 5% above our dot-product form. We chose dot form because it needs no packing; that choice costs a few percent on large products.
- In the whole model at T=1024 we are level with ONNX Runtime anyway, so it loses elsewhere what it gains in the kernel.
- At T=16 our kernel is the faster one alone, yet the whole model is slower. Alone, the weight being multiplied stays in cache between runs. In the model, every call reads all 494 MB of weights from memory once and uses each value only 16 times, so the run is limited by memory, not arithmetic. Our dense layers reach 82 GFLOP/s there on one thread and about 260 on four, against 100 and 370 alone. We did not find what ONNX Runtime does better here. Software prefetch, the obvious candidate, made a hand-written version 15 to 30% slower.

## What each optimization bought

Loop optimizations switched on one at a time, seconds on one thread until the last row (`m4_ablation_cumulative.md`, `m4_ablation_T1024.md`):

| | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| graph passes only | 0.638 | 2.310 | 9.209 | 37.40 |
| + `noalias` | 0.653 | 2.309 | 9.198 | |
| + `vectorize` | 0.280 | 0.872 | 3.297 | |
| + `vmath` | 0.281 | 0.863 | 3.396 | |
| + `layout` | 0.211 | 0.658 | 2.495 | |
| + `hoist` | 0.145 | 0.586 | 2.427 | |
| + `regtile` | 0.046 | 0.175 | 0.711 | |
| + `cachetile` | 0.046 | 0.174 | 0.688 | 3.165 |
| + `parallel`, 4 threads | 0.020 | 0.058 | 0.212 | 1.337 |

- Vectorizing (2.3 to 2.8x), putting products in dot form (1.3x), and register tiling (3.2 to 3.5x) account for nearly all of the 13x on one thread. Threads add 2.3 to 3.2x.
- `noalias`, `vmath`, and `cachetile` each change the total by 3% or less. `vmath` matters to the ops it touches (softmax and GELU), which are 3% of the run. `cachetile` matters little because the kernel is limited by arithmetic.
- Leave-one-out on four threads (`m4_ablation_leave_one_out.md`) agrees on the big three and on `parallel`. Its rows for `noalias`, `vmath`, and `cachetile` at T=64 differ from the full build by less than the full build differs from itself between sessions, so they say nothing.
- The 1.337 s at T=1024 on four threads is slower than the 1.03 s in the comparison table: the ablation measures three runs straight after a build, the comparison thirty after warm-up.

Graph passes (`m3_ablation_cumulative.md`, `m3_ablation_T1024.md`) change speed by nothing measurable, on AC as before: 0.634 to 0.637 s at T=16, 38.4 to 37.7 s at T=1024. They cut the graph from 1062 nodes to 615 and, with memory planning, the arena at T=1024 from 6453 MB to 158 MB. After M4 the arena is 72 MB at T=1024 and 0.3 MB at T=16, plus 340 MB of weights transposed at load time.

## What I would do next

1. Find what limits the dense layers at small `T` on four threads, starting from a per-kernel comparison against ONNX Runtime's profiler.
2. Try the packed outer-product kernel for the large products, where its 5 to 9% is on offer, and keep dot form where packing costs more than it saves.
3. Store the dense-layer weights transposed in the weights file, which removes the 340 MB of persistent memory.
4. Make the autotuner measure with cold operands. It currently times a kernel whose weight is already in cache, which is not the situation in the model at small `T`.
