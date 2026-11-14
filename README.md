# ml-compiler

A small ahead-of-time ML compiler that runs GPT-2 small on a CPU.

It takes the real PyTorch model through `torch.export`, lowers it into its own graph IR, optimizes the graph, lowers again to a loop-level IR, applies tiling, vectorization, and multi-threading, and emits C++ with SIMD intrinsics. An autotuner picks kernel parameters by measurement. The compiler itself is written in C++, and the code it generates calls no BLAS library.

**Status:** M3 complete. GPT-2 compiles end to end and reproduces PyTorch's logits at every sequence length. Graph optimizations (simplification, constant folding, CSE, GELU and attention fusion) cut the graph from 1062 nodes to 615, and memory planning cuts the arena at T=1024 from 6.5 GB to 158 MB. The generated loops are still naive, so it remains far slower than the baselines; loop optimization is next. See [STEERING.md](STEERING.md) for the design, milestones, and exit criteria.

## Pipeline

```
GPT-2 (PyTorch)
   |  torch.export
   v
Graph IR      constant folding, operator fusion, memory planning
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
| M4 | Loop optimizations and SIMD kernels | not started |
| M5 | Autotuner | not started |
| M6 | Final CPU benchmark and writeup | not started |
| M7 | Triton GPU backend | not started |

## Benchmarks

Compared end to end against PyTorch eager, `torch.compile` on CPU, and ONNX Runtime: same machine, same inputs, fp32, batch 1, equal thread counts. Methodology is in [STEERING.md](STEERING.md#7-benchmarking-rules).

Baseline numbers are in [bench/results/baselines.md](bench/results/baselines.md). The compiler's unoptimized starting point is in [bench/results/mlc_m2.md](bench/results/mlc_m2.md), produced by `python bench/run.py --systems mlc --threads 1 --name mlc_m2`. What each graph optimization pass contributes is in [bench/results/m3_ablation_cumulative.md](bench/results/m3_ablation_cumulative.md). A side-by-side table will be added here once the compiler is worth comparing.

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

# Print the loop IR (what each graph node becomes as loops)
build/release/mlc loops artifacts/graphs/gpt2_T16.json | less

# Compile to C++ and a shared library, then run it against the golden logits
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/gpt2_T16
build/release/mlc exec build/release/gen/gpt2_T16/model.so --weights artifacts/graphs/gpt2_weights.bin \
    --input artifacts/golden/T16_ids.npy --expect artifacts/golden/T16_logits.npy --runs 5 --profile

# Benchmark the compiled model from Python (compiles each T first); writes bench/results/mlc_m2.md
.venv/bin/python bench/run.py --systems mlc --threads 1 --name mlc_m2

# Attribute speed and memory to individual passes; writes bench/results/m3_ablation_*.md
.venv/bin/python bench/ablate.py
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/no_gelu --disable gelu,memory   # any subset by hand

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
runtime/      handwritten C++ linked into generated code: arena, thread pool, weight loader
bench/        Python benchmark harness and baselines
tests/
docs/         build log and notes
```
