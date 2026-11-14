# ml-compiler

A small ahead-of-time ML compiler that runs GPT-2 small on a CPU.

It takes the real PyTorch model through `torch.export`, lowers it into its own graph IR, optimizes the graph, lowers again to a loop-level IR, applies tiling, vectorization, and multi-threading, and emits C++ with SIMD intrinsics. An autotuner picks kernel parameters by measurement. The compiler itself is written in C++, and the code it generates calls no BLAS library.

**Status:** the CPU compiler is complete (M0 to M6). GPT-2 compiles end to end and reproduces PyTorch's logits at every sequence length. Against ONNX Runtime, whose kernels use the same SIMD units, it is level at T=1024 on one and four threads, 1 to 6% behind at shorter lengths on one thread, and 12 to 57% behind on four. It does not approach PyTorch, whose matrix products run on Apple Accelerate. The results and what explains them are in [docs/writeup.md](docs/writeup.md); the design is in [STEERING.md](STEERING.md) and the history in [docs/log.md](docs/log.md). The Triton GPU backend (M7) is not started.

## Pipeline

```
GPT-2 (PyTorch)
   |  torch.export
   v
Graph IR      constant folding, operator fusion, layout, memory planning
   |
   v
Loop IR       tiling, vectorization, multi-threading
   |
   v
C++ + NEON intrinsics  ->  clang++  ->  shared library
   ^
   |  autotuner: tile sizes, micro-kernel shape, unroll, threads
```

A GPU backend that emits Triton kernels from the same graph IR comes last.

## Roadmap

| # | Milestone | Status |
|---|---|---|
| M0 | Baselines and benchmark harness | done |
| M1 | Frontend, graph IR, reference interpreter | done |
| M2 | Naive end-to-end scalar codegen | done |
| M3 | Graph optimizations | done |
| M4 | Loop optimizations and SIMD kernels | done |
| M5 | Autotuner | done |
| M6 | Final CPU benchmark and writeup | done |
| M7 | Triton GPU backend | not started |

## Benchmarks

Compared end to end against PyTorch eager, `torch.compile` on CPU, and ONNX Runtime: same machine, same inputs, fp32, batch 1, equal thread counts. Methodology is in [STEERING.md](STEERING.md#7-benchmarking-rules).

Apple M2, AC power, median latency in ms, all systems in one session ([bench/results/final.md](bench/results/final.md), `python bench/run.py --systems eager compile ort mlc mlc-tuned --name final`):

| 1 thread | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager (Accelerate) | 27.8 | 63.8 | 183.2 | 705.5 |
| torch.compile | 27.0 | 59.8 | 160.3 | 749.4 |
| ONNX Runtime | 43.2 | 166.9 | 700.8 | 3219.4 |
| mlc (ours) | 50.3 | 180.1 | 708.7 | 3202.0 |
| mlc, autotuned (ours) | 45.8 | 177.5 | 707.4 | 3191.8 |

| 4 threads | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager (Accelerate) | 29.6 | 61.5 | 144.8 | 474.8 |
| torch.compile | 27.4 | 55.2 | 121.8 | 473.3 |
| ONNX Runtime | 14.1 | 47.0 | 194.8 | 1027.3 |
| mlc (ours) | 22.2 | 68.5 | 218.0 | 1032.7 |
| mlc, autotuned (ours) | 23.1 | 66.0 | 239.1 | 1032.6 |

[docs/writeup.md](docs/writeup.md) explains these numbers. The tables behind it:

- Time per operation: [per_op.md](bench/results/per_op.md) (`python bench/per_op.py`). Matrix products are 96 to 98% of every run.
- Matrix products alone against ONNX Runtime and Accelerate: [gemm_T16.md](bench/results/gemm_T16.md), [gemm_T1024.md](bench/results/gemm_T1024.md) (`python bench/gemm.py --seq-len 1024`).
- What each loop optimization contributes: [m4_ablation_cumulative.md](bench/results/m4_ablation_cumulative.md), [m4_ablation_leave_one_out.md](bench/results/m4_ablation_leave_one_out.md), [m4_ablation_T1024.md](bench/results/m4_ablation_T1024.md) (`python bench/ablate.py`).
- What each graph pass contributes: [m3_ablation_cumulative.md](bench/results/m3_ablation_cumulative.md), [m3_ablation_T1024.md](bench/results/m3_ablation_T1024.md) (`python bench/ablate.py --stage m3`).
- Earlier sessions, kept as history and not comparable with the above: `baselines.md`, `mlc_m2.md`, `m4.md`, and `m3_ablation_leave_one_out.md` (battery, Low Power Mode), `m5.md` (AC).

## Scope

- Inference only, fp32, static shapes (one compile per sequence length).
- Primary target is Apple Silicon (arm64, NEON). The codegen keeps the instruction set behind a thin layer so x86 AVX2 can be added.
- GPT-2 small is the only model the compiler is required to handle.

## Requirements

- macOS on Apple Silicon, or Linux
- A C++20 compiler (`clang++`) and CMake
- Python 3.11 or newer, for the model exporter and the benchmarks only
- `torch`, `transformers`, `onnxruntime`, `numpy`, pinned in `requirements.txt`

## Setup

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

```sh
# Export GPT-2, verify the exported graph, save golden logits, graphs and weights (about 760 MB under artifacts/)
.venv/bin/python export/export_gpt2.py

# Benchmark the baselines; writes bench/results/baselines.md
.venv/bin/python bench/run.py

# Build the compiler (optimized) and run its tests
cmake -S . -B build/release
cmake --build build/release -j
ctest --test-dir build/release --output-on-failure            # add -LE slow to skip T=256 and T=1024

# Print the graph IR for one sequence length
build/release/mlc dump artifacts/graphs/gpt2_T16.json | less

# Run the reference interpreter against PyTorch's golden logits, with a per-op time breakdown
build/release/mlc run artifacts/graphs/gpt2_T16.json \
    --input artifacts/golden/T16_ids.npy --expect artifacts/golden/T16_logits.npy --profile

# See what each optimization pass removes, and the arena before and after memory planning
build/release/mlc opt artifacts/graphs/gpt2_T16.json
build/release/mlc dump artifacts/graphs/gpt2_T16.json --opt | less        # the optimized graph

# Print the loop IR: each graph node as loops, after scheduling (tiled, vectorized, parallel)
build/release/mlc loops artifacts/graphs/gpt2_T16.json | less
build/release/mlc loops artifacts/graphs/gpt2_T16.json --disable vectorize,regtile,cachetile,parallel | less   # the naive nests

# Compile to C++ and a shared library, then run it against the golden logits
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/gpt2_T16
build/release/mlc exec build/release/gen/gpt2_T16/model.so --weights artifacts/graphs/gpt2_weights.bin \
    --input artifacts/golden/T16_ids.npy --expect artifacts/golden/T16_logits.npy --runs 20 --threads 4 --profile

# Benchmark the baselines and the compiled model in one session; writes bench/results/final.md
.venv/bin/python bench/run.py --systems eager compile ort mlc mlc-tuned --name final

# Time the model's matrix products alone, against ONNX Runtime and Accelerate; writes bench/results/gemm_T1024.md
build/release/mlc gemm artifacts/graphs/gpt2_T1024.json -o build/release/gen/gemm_T1024    # ours only
.venv/bin/python bench/gemm.py --seq-len 1024

# Autotune: search each matrix product's schedule by measurement (about 25 minutes for all four lengths)
build/release/mlc tune artifacts/graphs/gpt2_T16.json --cache tuning/cache.json -o build/release/gen/tune --threads 1,4
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/gpt2_T16_tuned --tuning tuning/cache.json --tuned-threads 4
.venv/bin/python bench/run.py --systems ort mlc mlc-tuned --name m5      # default against tuned; writes bench/results/m5.md

# Time per kind of operation; writes bench/results/per_op.md
.venv/bin/python bench/per_op.py

# Attribute speed and memory to individual optimizations; writes bench/results/m4_ablation_*.md
.venv/bin/python bench/ablate.py                    # loop optimizations, one switched on at a time
.venv/bin/python bench/ablate.py --stage m3         # graph passes (the M3 tables)
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/no_tile --disable regtile,parallel   # any subset by hand
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/tile44 --tile 4x4 --block-kb 32      # schedule parameters

# The same tests under AddressSanitizer and UndefinedBehaviorSanitizer (slow: debug build)
cmake -S . -B build/asan -DCMAKE_BUILD_TYPE=Debug -DMLC_SANITIZE=ON
cmake --build build/asan -j
ctest --test-dir build/asan -R "unit|T16" --output-on-failure
```

The first run downloads the GPT-2 weights from Hugging Face. The full benchmark takes several minutes; add `--seq-lens 16 64` for a quick pass.

## Layout

```
export/       Python: torch.export -> graph.json + weights.bin
src/          the compiler (C++): frontend, graph, loop, codegen, tune, support; later gpu
tuning/       cache.json: schedules measured by the autotuner on the development machine
runtime/      handwritten C++ linked into generated code: arena, thread pool, weight loader, NEON exp and tanh
bench/        Python benchmark harness and baselines
tests/
docs/         writeup.md (results), log.md (what each milestone built and found)
```
