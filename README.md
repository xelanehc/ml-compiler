# ml-compiler

A small ahead-of-time ML compiler that runs GPT-2 small on a CPU.

It takes the real PyTorch model through `torch.export`, lowers it into its own graph IR, optimizes the graph, lowers again to a loop-level IR, applies tiling, vectorization, and multi-threading, and emits C++ with SIMD intrinsics. An autotuner picks kernel parameters by measurement. The compiler itself is written in C++, and the code it generates calls no BLAS library.

**Status:** M0 complete (baselines measured, golden outputs saved). The compiler itself is not written yet. See [STEERING.md](STEERING.md) for the design, milestones, and exit criteria.

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
| M1 | Frontend, graph IR, reference interpreter | not started |
| M2 | Naive end-to-end scalar codegen | not started |
| M3 | Graph optimizations | not started |
| M4 | Loop optimizations and SIMD kernels | not started |
| M5 | Autotuner | not started |
| M6 | Final CPU benchmark and writeup | not started |
| M7 | Triton GPU backend | not started |

## Benchmarks

Compared end to end against PyTorch eager, `torch.compile` on CPU, and ONNX Runtime: same machine, same inputs, fp32, batch 1, equal thread counts. Methodology is in [STEERING.md](STEERING.md#7-benchmarking-rules).

Baseline numbers are in [bench/results/baselines.md](bench/results/baselines.md). Compiler results will be added here once they exist, each with the command that reproduces it.

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
# Export GPT-2, verify the exported graph, save golden inputs and logits (about 260 MB under artifacts/)
.venv/bin/python export/export_gpt2.py

# Benchmark the baselines; writes bench/results/baselines.md
.venv/bin/python bench/run.py
```

The first run downloads the GPT-2 weights from Hugging Face. The full benchmark takes several minutes; add `--seq-lens 16 64` for a quick pass.

## Layout

```
export/       Python: torch.export -> graph.json + weights.bin
src/          the compiler (C++): frontend, graph, loop, codegen, tune, gpu
runtime/      handwritten C++ linked into generated code: arena, thread pool, weight loader
bench/        Python benchmark harness and baselines
tests/
docs/         build log and notes
```
