# The ml-compiler guide

This guide explains the whole project from the beginning: what it does, every concept it uses, how each part is implemented, and why each design choice was made. It assumes you can read C++ and Python and know what a matrix is. It does not assume you know anything about compilers, about how GPT-2 works inside, or about CPU performance.

It is long on purpose. Read it in order the first time; every chapter uses words the earlier ones define. After that, use it as a reference next to the source.

## Chapters

| # | Chapter | What you will understand afterwards |
|---|---|---|
| 1 | [The big picture](01-big-picture.md) | What a compiler for a neural network is, the pipeline end to end, and what the project achieved |
| 2 | [GPT-2, the program being compiled](02-gpt2.md) | Tensors, shapes, and every operation in one forward pass, with the math |
| 3 | [Getting the model out of PyTorch](03-export.md) | `torch.export`, ATen operators, the `graph.json` and weights file formats, golden outputs |
| 4 | [The graph IR and the importer](04-graph-ir.md) | SSA form, static types, the 27 operators, shape inference, how 37 PyTorch operators map onto them |
| 5 | [The reference interpreter and the idea of an oracle](05-interpreter.md) | How the graph is executed directly, strides and broadcasting, and how correctness is established |
| 6 | [Graph optimizations](06-graph-passes.md) | Rewriting, dead code elimination, constant folding, CSE, fusion, the layout pass |
| 7 | [The loop IR, lowering, and memory planning](07-loop-ir.md) | How "matmul" becomes loops over memory, where every value lives, liveness, staging |
| 8 | [Code generation and the runtime](08-codegen-runtime.md) | How loops become C++, how that becomes a library, how the library is loaded and called |
| 9 | [What makes code fast on a CPU](09-cpu-performance.md) | Caches, SIMD, pipelines, latency and throughput, aliasing, floating point, threads |
| 10 | [Loop transformations](10-loop-transformations.md) | Dependence analysis, the five rewrites, how the fast matrix product is derived step by step |
| 11 | [The autotuner](11-autotuner.md) | Searching for schedules by measurement, and why it found little |
| 12 | [Testing, benchmarking, and the results](12-testing-results.md) | Every kind of test, how the numbers were measured, what they mean |
| 13 | [Design decisions, open problems, further reading, glossary](13-reference.md) | Every choice in one table, what is unfinished, sources for each topic, definitions |

## How to read alongside the code

Each chapter has a "Where in the code" list. The most useful commands while reading are the ones that show you an intermediate form of the real model:

```sh
build/release/mlc dump  artifacts/graphs/gpt2_T16.json | less          # graph IR, as imported
build/release/mlc dump  artifacts/graphs/gpt2_T16.json --opt | less    # graph IR, after the passes
build/release/mlc opt   artifacts/graphs/gpt2_T16.json                 # what each pass removed
build/release/mlc loops artifacts/graphs/gpt2_T16.json --disable vectorize,regtile,cachetile,parallel | less   # naive loops
build/release/mlc loops artifacts/graphs/gpt2_T16.json | less          # scheduled loops
less build/release/gen/gpt2_T16/model.cpp                              # generated C++ (after `mlc compile`)
```

`T` is the sequence length, the number of tokens fed to the model. The project compiles four versions, for T = 16, 64, 256, and 1024. T=16 is the one to look at: everything is small enough to read.

## Other documents

- [`STEERING.md`](../../STEERING.md): the plan. What is being built, in what order, and how a stage is judged done. Short and prescriptive.
- [`docs/log.md`](../log.md): the history. One entry per milestone: what was built, the numbers, what was surprising.
- [`docs/writeup.md`](../writeup.md): the final results and their interpretation.
- This guide: the explanation. It repeats material from the three above where that helps.

## The milestones, for orientation

| | Milestone | Guide chapters |
|---|---|---|
| M0 | Baselines and benchmark harness | 3, 12 |
| M1 | Frontend, graph IR, reference interpreter | 3, 4, 5 |
| M2 | Naive end to end: loops, scalar C++, runtime | 7, 8 |
| M3 | Graph optimizations and memory planning | 6, 7 |
| M4 | Loop optimizations, SIMD, threads | 9, 10 |
| M5 | Autotuner | 11 |
| M6 | Final benchmark and write-up | 12 |
| M7 | Triton GPU backend | not started |
