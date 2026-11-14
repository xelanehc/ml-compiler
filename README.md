# ml-compiler

A small ahead-of-time ML compiler that runs GPT-2 small on a CPU.

It takes the real PyTorch model through `torch.export`, lowers it into its own graph IR, optimizes the graph, lowers again to a loop-level IR, applies tiling, vectorization, and multi-threading, and emits C++ with SIMD intrinsics. An autotuner picks kernel parameters by measurement. The compiler itself is written in C++, and the code it generates calls no BLAS library.

**Status:** M4 complete. GPT-2 compiles end to end and reproduces PyTorch's logits at every sequence length. Loop optimizations (a transposed-operand layout for matrix products, register and cache tiling, NEON vectorization, load-time staging, threads) took one-thread time at T=1024 from 63.6 s to 5.4 s, and four threads bring it to between 1.5 and 3.2 s depending on throttling. That is within 1 to 7% of ONNX Runtime on one thread, and not yet ahead of it. All current numbers were taken on battery in Low Power Mode and need re-measuring on AC. See [STEERING.md](STEERING.md) for the design, milestones, and exit criteria, and [docs/log.md](docs/log.md) for what each milestone found.

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
| M5 | Autotuner | not started |
| M6 | Final CPU benchmark and writeup | not started |
| M7 | Triton GPU backend | not started |

## Benchmarks

Compared end to end against PyTorch eager, `torch.compile` on CPU, and ONNX Runtime: same machine, same inputs, fp32, batch 1, equal thread counts. Methodology is in [STEERING.md](STEERING.md#7-benchmarking-rules).

Current results, all four systems measured in one session: [bench/results/m4.md](bench/results/m4.md) (`python bench/run.py --systems eager compile ort mlc --name m4`). Median latency in ms on one thread, on battery with Low Power Mode on:

| system | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager (Accelerate) | 37.2 | 98.2 | 302.8 | 1077.6 |
| ONNX Runtime | 70.8 | 279.3 | 1166.3 | 5339.3 |
| mlc (ours) | 75.6 | 292.9 | 1189.3 | 5381.2 |

Supporting tables:

- Matrix products alone against ONNX Runtime and Accelerate: [gemm_T16.md](bench/results/gemm_T16.md), [gemm_T1024.md](bench/results/gemm_T1024.md) (`python bench/gemm.py --seq-len 1024`).
- What each loop optimization contributes: [m4_ablation_cumulative.md](bench/results/m4_ablation_cumulative.md), [m4_ablation_leave_one_out.md](bench/results/m4_ablation_leave_one_out.md), [m4_ablation_T1024.md](bench/results/m4_ablation_T1024.md) (`python bench/ablate.py`).
- Earlier stages: [baselines.md](bench/results/baselines.md), [mlc_m2.md](bench/results/mlc_m2.md), and the graph-pass ablations `m3_ablation_*.md`.

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

# Benchmark the baselines and the compiled model in one session; writes bench/results/m4.md
.venv/bin/python bench/run.py --systems eager compile ort mlc --name m4

# Time the model's matrix products alone, against ONNX Runtime and Accelerate; writes bench/results/gemm_T1024.md
build/release/mlc gemm artifacts/graphs/gpt2_T1024.json -o build/release/gen/gemm_T1024    # ours only
.venv/bin/python bench/gemm.py --seq-len 1024

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
src/          the compiler (C++): frontend, graph, loop, codegen, support; later tune, gpu
runtime/      handwritten C++ linked into generated code: arena, thread pool, weight loader, NEON exp and tanh
bench/        Python benchmark harness and baselines
tests/
docs/         build log and notes
```
