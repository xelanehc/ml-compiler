# Build log

## M0: baselines and harness (2026-11-11)

**Built**

- `export/model.py`: the shared model wrapper (ids to logits, cache off) and the fixed inputs.
- `export/export_gpt2.py`: saves golden ids and logits per `T`, runs `torch.export` plus decomposition, checks the exported program against eager, writes `docs/export_ops.md`.
- `bench/worker.py` and `bench/run.py`: time PyTorch eager, `torch.compile`, and ONNX Runtime in separate processes; write `bench/results/baselines.md`.

**Numbers** (median ms, measured on battery, so provisional)

| system | threads | T=16 | T=64 | T=256 | T=1024 |
|---|---|---:|---:|---:|---:|
| PyTorch eager | 1 | 38.4 | 109.2 | 457.7 | 1121.7 |
| torch.compile | 1 | 35.4 | 93.7 | 267.1 | 1190.1 |
| ONNX Runtime | 1 | 70.8 | 281.0 | 1171.4 | 5343.1 |
| PyTorch eager | 4 | 38.8 | 84.5 | 189.3 | 783.4 |
| torch.compile | 4 | 36.5 | 79.9 | 163.4 | 785.2 |
| ONNX Runtime | 4 | 21.2 | 137.5 | 572.0 | 2950.8 |

Export: 517 nodes as traced, 852 after decomposition to core ATen, 37 distinct ops, identical across all four `T`. Exported program matches eager to about 1e-4.

**Surprises**

- PyTorch's matrix multiply runs at about 710 GFLOP/s on one thread (1024x768 times 768x3072, CPU time / wall time = 1.00) and 768 with four. One core's NEON units cannot do that, so Accelerate is using hardware outside ordinary SIMD, and the thread setting does not control it.
- Even so, PyTorch's whole-model rate at `T=1024` is only around 260 GFLOP/s. Most of its time at long sequences is not GEMM.
- ONNX Runtime beats PyTorch at `T=16` with 4 threads (21 ms against 38 ms) and loses badly at long `T`.
- `torch.compile` is barely faster than eager here. Both end up in the same Accelerate call.
- Timing noise on battery is large: several configurations have a p10 to p90 spread above 20% of the median.

**Open**

- Rerun `bench/run.py` on AC power and replace the provisional table.

## M1: frontend, graph IR, interpreter (2026-11-12)

**Built**

- `export/serialize.py`: writes each exported program as `artifacts/graphs/gpt2_T<T>.json` plus one shared `gpt2_weights.bin` (498 MB, tensors 64-byte aligned, tied weights stored once). It records ATen ops and arguments as they are and interprets nothing.
- `src/support/`: a strict JSON reader that keeps ints and floats apart, `mmap` for the weights, a `.npy` reader.
- `src/graph/ir.*`: the graph IR. 26 ops, one value per node, static dtype and shape. `infer_type` is the single definition of every op's typing rule.
- `src/graph/printer.cpp`, `verifier.cpp`: text dump; structural and type checks.
- `src/graph/interpreter.*`: reference interpreter. Frees values after their last use; reports per-op time and peak live memory.
- `src/frontend/importer.*`: lowers all 37 ATen ops. Rejects unknown ops and unread arguments; checks every inferred type against the exported one.
- `src/main.cpp`: `mlc dump | stats | run`.
- 40 unit tests and four end-to-end golden tests under `ctest`; an ASan + UBSan configuration.

**Numbers** (release build, one thread, on battery)

| T | max abs logit diff vs PyTorch | argmax mismatches | interpreter time | PyTorch eager, 1 thread (M0) | peak live memory | allocated in total |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 1.45e-04 | 0 of 16 | 0.56 s | 0.038 s | 158 MB | 195 MB |
| 64 | 1.14e-04 | 0 of 64 | 1.55 s | 0.109 s | 168 MB | 328 MB |
| 256 | 2.14e-04 | 0 of 256 | 5.86 s | 0.458 s | 207 MB | 1034 MB |
| 1024 | 1.45e-04 | 0 of 1024 | 27.4 s | 1.12 s | 363 MB | 6657 MB |

Tolerance is 1e-3. The differences are the same size as PyTorch's own export-versus-eager difference (about 1e-4), so they are fp32 summation order, not a bug.

Graph size: 852 ATen nodes become 1062 IR nodes. Of those, 290 are constants (148 weight tensors, the rest scalars and index tables), 230 reshapes, 85 permutes, 61 broadcasts; 73 are matrix products (48 `gemm`, 25 `matmul`).

Reproduce: `ctest --test-dir build/release --output-on-failure`, or `build/release/mlc run artifacts/graphs/gpt2_T<T>.json --input artifacts/golden/T<T>_ids.npy --expect artifacts/golden/T<T>_logits.npy --profile`.

**Surprises**

- The graph grew on import (852 to 1062). ATen passes Python scalars as arguments; in our IR every operand is a value, so each `0.5` becomes a constant node. `clone`, `alias`, `getitem`, and the metadata assertions go the other way and produce no node.
- The token embedding and the output projection are one tensor. The export looks it up as rows for the embedding and then, at the very end, transposes all of it (`permute` of `[50257, 768]`, 154 MB) for the final `mm`. That one node was 37% of a profiled T=16 run and sets peak memory at every `T` below 1024. It has no dependence on the input, so it is the first thing constant folding should remove, or better, the final product should read the weight transposed in place.
- About 40 nodes at the top of the graph rebuild the causal mask from `arange`, `cumsum`, and comparisons on every call. None of it depends on the input.
- Every attention layer carries a guard for rows that are masked everywhere (`eq -inf`, `not`, `reduce_any`, `not`, `where`): several full `[1,12,T,T]` passes per layer that can never fire with a causal mask, since position `i` can always attend to itself. At T=1024, `where` and `broadcast` alone are 1.7 s.
- All 49 `expand` nodes are no-ops (target shape equals the input shape), and the head merge is three permutes in a row that compose to one.
- GELU arrives as eight primitive nodes per layer, each a full pass over `[1, T, 3072]`.
- With the naive `i, p, j` loop the two matrix-product ops are 79% of the T=1024 run (21.6 s). Everything else is the remaining 5.8 s, which is roughly five times PyTorch's entire forward pass. Both halves need work, and the non-GEMM half is not small.

**Open**

- Baseline and interpreter timings above were taken on battery. Rerun on AC before quoting them.

## M2: naive end to end (2026-11-12)

**Built**

- `src/loop/ir.*`: the loop IR. Three statements (`For`, `Store`, `Block`), expressions with multi-dimensional `Load`, buffers as typed views of storage (arena, weights, input, output, inline). A printer (`mlc loops`) and a verifier.
- `src/loop/lower.cpp`: every graph op to a naive loop nest, one kernel per node. Each computed value gets its own arena slice; `Reshape` becomes a second view of the same bytes and no code.
- `src/loop/interpreter.cpp`: tree-walking reference interpreter for loop programs, bounds-checking every index on every axis.
- `src/codegen/emit_cpp.cpp`: loop IR to scalar C++17, one function per kernel plus a kernel table.
- `src/codegen/build.cpp`: runs `clang++ -O3 -mcpu=native -shared`, and loads the result with `dlopen`.
- `runtime/mlc_runtime.*`: maps the weights, allocates the arena, walks the kernel table, times kernels. C API for `ctypes`.
- `mlc loops | compile | exec`; `bench/worker.py` gained the `mlc` system.
- 18 new unit tests (58 total): every op's lowering checked differentially against the graph interpreter, a toy transformer through graph interpreter, loop interpreter, and compiled code, verifier negatives. `ctest` now compiles and runs the real model per `T`. In the sanitizer configuration the generated code is built with ASan + UBSan as well.

**Numbers** (AC power, one thread)

| T | max abs logit diff vs PyTorch | argmax mismatches | compiled, median | graph interpreter (M1) | PyTorch eager, 1 thread (M0, on battery) | arena |
|---|---:|---:|---:|---:|---:|---:|
| 16 | 1.4e-04 | 0 | 0.65 s | 0.41 s | 0.038 s | 192 MB |
| 64 | 1.2e-04 | 0 | 2.32 s | 1.00 s | 0.109 s | 315 MB |
| 256 | 2.1e-04 | 0 | 8.97 s | 3.54 s | 0.458 s | 983 MB |
| 1024 | 1.8e-04 | 0 | 38.1 s | 16.3 s | 1.12 s | 6453 MB |

This is the "unoptimized" row. Compiled times are from `python bench/run.py --systems mlc --threads 1 --name mlc_m2` (`bench/results/mlc_m2.md`); T=256 has 14 timed runs and T=1024 has 4, because of the harness's 120 s cap for slow systems. Interpreter times are single runs of `mlc run` on the same power state. The PyTorch column was measured on battery and is not directly comparable until M0 is rerun on AC.

1062 graph nodes become 541 kernels and 8626 lines of C++; `clang++` takes 10 to 17 s per sequence length.

Profile at T=1024 (`mlc exec ... --profile`, one run): `gemm` 30.5 s (77.5%), `matmul` 6.3 s (16.1%), `softmax` 0.9 s, everything else under 0.3 s each. The final `mm` alone is 5.0 s.

**Surprises**

- The compiled model is 1.6 to 2.5 times slower than the interpreter it was derived from. Both use the same `i, p, j` matmul loop. In the interpreter that loop is handwritten C++ with `A[i][p]` held in a local; in generated code every operand is a load through a pointer computed from `arena`. clang's vectorizer reports "loop not vectorized: unsafe dependent memory operations" for all 48 `gemm` kernels: it cannot prove the store to `C` leaves `A` unchanged, because both are offsets into the same arena. So `gemm` runs scalar and reloads `A[i][p]` on every iteration.
- The final `mm` is the same loop and *is* vectorized (width 4, interleave 4). Its output is the caller's pointer, not the arena, and clang is willing to guard that with a run-time overlap check. Per multiply-add it runs nearly three times faster than `gemm`. The only difference is where the buffers live.
- So the compiler knows something clang does not: its buffers never overlap. Telling clang (`restrict`, or loading invariants into locals) is the cheapest speedup available and belongs in M4, measured.
- The first call at T=1024 took 82 s against 38 to 41 s afterwards: 6.4 GB of fresh arena is 400 thousand page faults. Without memory reuse the arena is as large as every intermediate added together.
- The non-GEMM overhead that hurt the interpreter mostly disappeared: `where` + `broadcast` went from 1.7 s to 0.37 s at T=1024, because a broadcast is now a missing loop variable rather than an odometer.
- All four sequence lengths matched PyTorch on the first compiled run.

**Open**

- Rerun the M0 baselines on AC so the tables share a power state.
