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

## M3: graph optimizations (2026-11-13)

**Built**

- `src/graph/passes.*`: a rewriting engine (`rewrite`: walk the old graph in order, build a new one, each rule sees operands that are already rewritten) and the passes on top of it: `simplify`, `fold`, `cse`, `gelu`, `attention`, plus dead code elimination after each. `evaluate_constant` runs any input-free subgraph in the interpreter at compile time.
- Graph IR: a `Gelu` op and a causal form of `Softmax` (`attrs.causal`), both produced only by fusion.
- `src/loop/lower.cpp`: memory planning. Liveness per value, in-place reuse, best-fit offsets with a coalescing free list. Lowerings for `Gelu` and causal softmax.
- `mlc opt` (per-pass report), `--opt` on `dump` / `stats` / `run`, `--disable <passes>` wherever the graph is optimized. `compile` and `loops` optimize by default.
- `bench/ablate.py`: cumulative and leave-one-out ablations.
- 25 new unit tests (83 total): each pass fires where it should and holds back on lookalikes, each has a differential test, the planner is checked for reuse, in-place safety, and bit-identical results with planning on and off. `ctest` gained `optimized_T<T>` (the interpreter on the optimized real graph against golden logits) and an unoptimized compile-and-run at T=16.

**Numbers**

Graph size, identical at every `T` (`mlc opt artifacts/graphs/gpt2_T16.json`):

| pass | nodes before | nodes after | removed |
|---|---:|---:|---:|
| simplify | 1062 | 951 | 111 |
| fold | 951 | 902 | 49 |
| cse | 902 | 780 | 122 |
| gelu | 780 | 691 | 89 |
| attention | 691 | 615 | 76 |

Kernels go from 541 to 258; generated C++ compiles in 3 to 5 s instead of 10 to 17 s. What is left: 73 matrix products, 49 permutes, 36 slices, 25 layer norms, 25 adds, 24 multiplies, 12 GELUs, 12 softmaxes, 2 gathers, and 206 reshapes that cost nothing.

Arena (MB), cumulative, from `bench/results/m3_ablation_cumulative.md` and `m3_ablation_T1024.md`:

| passes | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| none | 191.6 | 314.8 | 982.5 | 6452.9 |
| + simplify | 187.9 | 298.2 | 888.0 | |
| + fold | 187.9 | 298.0 | 884.6 | |
| + cse | 187.7 | 295.9 | 850.0 | |
| + gelu | 171.2 | 229.8 | 585.8 | |
| + attention | 170.8 | 223.7 | 488.2 | |
| + memory | 154.4 | 154.6 | 155.2 | 157.5 |

Time (s, minimum over timed runs, one thread, on battery):

| passes | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| none | 1.044 | 3.937 | 15.247 | 65.21 |
| + simplify | 1.044 | 3.916 | 15.290 | |
| + fold | 1.052 | 3.925 | 15.520 | |
| + cse | 1.049 | 3.892 | 15.613 | |
| + gelu | 1.040 | 4.048 | 15.470 | |
| + attention | 1.050 | 3.949 | 15.645 | |
| + memory | 1.049 | 3.978 | 15.460 | 63.98 |

T=1024 was measured at the two ends only (2 timed runs each). Leave-one-out at T=64 (`m3_ablation_leave_one_out.md`) agrees: every row is within 2% of the full build.

Single profiled runs at T=256 (`mlc exec --profile`), all passes off against all on: time outside matrix products went from 0.476 s to 0.364 s; matrix products were 14.8 s and 16.4 s, which is run-to-run noise on battery.

All four sequence lengths match PyTorch after optimization, in the interpreter and compiled (max abs diff 1.4e-4 to 2.4e-4).

**Surprises**

- No pass produces a measurable speedup. The differences in the time table are within noise (up to 3%, in both directions). Matrix products are 97% of the run, and the graph passes do not touch them. The passes removed about a quarter of everything else, and a quarter of 3% does not show. Per-pass attribution of speed is therefore "none, yet"; what the passes bought is graph size, memory, and compile time. Their effect on speed has to be re-measured after M4, when the matrix products stop hiding everything else.
- These numbers are slower than the M2 entry (15.2 s against 9.0 s at T=256 for the same unoptimized build) because this session ran on battery and M2's benchmark ran on AC. Within this entry everything shares a power state.
- Memory planning is the large effect: 41x at T=1024 (6453 MB to 157.5 MB). The first call at T=1024 no longer pays for faulting in 6 GB.
- After planning, the arena is almost the same size at every `T`, about 154 MB, because one buffer sets it: the transposed copy of the output projection weight (50257 x 768 floats = 154.4 MB), which exists only for the final `mm`. All twelve layers together fit in the few MB above that at T=1024. Removing that transpose is now the single biggest memory item, and it needs a matmul that reads its operand transposed (M4).
- Fusing GELU did not make GELU faster in any way that matters: nine nodes took about 0.2 s at T=256 and the one fused node takes 0.13 s, because `tanh` is most of the cost either way. Its real effect is memory: seven fewer `[1, T, 3072]` buffers per layer, visible as the biggest single drop in the arena column before planning.
- CSE removed 122 nodes but only 11 kernels: nearly all of what it merged was duplicate scalar constants, which never had code.
- The constant-folding size cap behaves as intended at T=1024: the 4 MB mask is not folded (24 nodes removed instead of 49), the attention fusion still recognizes it by evaluating it on the side, and dead code elimination then removes the unfolded mask computation. The final graph is the same 615 nodes at every `T`.
- A unit test "failed" because folding evaluated a whole attention layer whose scores were a constant in the test. The pass was right and the test was wrong.

**Open**

- Rerun the baselines and the ablation on AC power.
- Re-run the ablation after M4.

## M4: loop optimizations (2026-11-13)

**Built**

- Loop IR: loops carry a kind (serial, vectorized, parallel) and an interleave count; extents may be expressions of enclosing variables; a `Persistent` storage class and a list of init kernels run once at load.
- `src/loop/schedule.*`: a dependence test (each iteration owns a slice of every buffer it writes), and the rewrites `split`, `interchange`, `unroll_and_jam`, `vectorize`, `parallelize`, each of which proves itself legal or throws. A default schedule applies them.
- Graph: `trans_b` on `matmul` and `gemm`, and the `layout` pass that puts every product in dot form and cancels existing transposes.
- Lowering: staging (`hoist`), the dot-form matrix product, and a causal softmax whose loops stop at column `i`.
- Codegen: `__restrict` parameters from the memory plan, NEON loops with register accumulators and a scalar leftover loop, fused multiply-add, parallel loops outlined into functions. `src/codegen/isa.h` is the only place that names an intrinsic. `runtime/mlc_neon.h` has vector `exp` and `tanh`.
- Runtime: a `std::thread` pool with static partitioning and spin-then-sleep workers, persistent memory, `mlc_set_threads`.
- Tools: `mlc gemm` (each distinct matrix product compiled and timed alone), `bench/gemm.py` (the same products in ONNX Runtime and Accelerate), `bench/ablate.py --stage m4`, and `bench/run.py` now records Low Power Mode.
- Tests: 98 unit tests (15 new), green in release and under ASan + UBSan, where the generated code is sanitized too and the real model was run at T=16 and T=64 with threads. T=256 and T=1024 were checked in release only.

**Numbers**

Everything below was measured on battery with Low Power Mode on. In that state one core peaks at about 59 GFLOP/s on an in-cache FMA loop (4 FMAs per cycle at about 1.85 GHz); on AC it should be nearly twice that. The tables are consistent with each other and with nothing else.

Whole model, median ms (`bench/results/m4.md`, all four systems in one session):

| 1 thread | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager (Accelerate) | 37.2 | 98.2 | 302.8 | 1077.6 |
| ONNX Runtime | 70.8 | 279.3 | 1166.3 | 5339.3 |
| mlc after M3 (`m4_ablation_*`) | 1050 | 3876 | 15719 | 63582 |
| mlc | 75.6 | 292.9 | 1189.3 | 5381.2 |

| 4 threads, median (min) | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| PyTorch eager | 38.0 (34.0) | 84.3 (75.2) | 188.7 (181.0) | 754.6 (632.1) |
| ONNX Runtime | 44.2 (25.8) | 140.7 (106.8) | 568.2 (450.2) | 3063.0 (2864.0) |
| mlc | 51.5 (33.5) | 182.2 (111.1) | 673.9 (360.8) | 3187.2 (2975.7) |

Matrix products alone, GFLOP/s weighted by the model's use (`bench/results/gemm_T16.md`, `gemm_T1024.md`):

| | T=16, 1 thr | T=16, 4 thr | T=1024, 1 thr | T=1024, 4 thr |
|---|---:|---:|---:|---:|
| mlc | 54.6 | 188.8 | 56.9 | 221.2 |
| ONNX Runtime `MatMul` | 43.7 | 100.5 | 50.0 | 164.7 |
| PyTorch (Accelerate) | 274.5 | 303.0 | 585.7 | 617.0 |

Attribution, one optimization switched on at a time, seconds, one thread until the last row (`m4_ablation_cumulative.md`, `m4_ablation_T1024.md`):

| | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| graph passes only (M3) | 1.050 | 3.876 | 15.719 | 63.58 |
| + noalias | 1.047 | 3.884 | 15.656 | |
| + vectorize | 0.433 | 1.435 | 5.552 | |
| + vmath | 0.434 | 1.406 | 5.608 | |
| + layout | 0.356 | 1.111 | 4.183 | |
| + hoist | 0.247 | 0.996 | 4.106 | |
| + regtile | 0.075 | 0.296 | 1.221 | |
| + cachetile | 0.075 | 0.292 | 1.178 | 5.372 |
| + parallel, 4 threads | 0.027 | 0.086 | 0.345 | 1.547 |

Memory at T=1024: arena 157.5 MB to 72.4 MB, plus 342.9 MB persistent. At T=16: arena 154.4 MB to 0.3 MB, plus 339.8 MB persistent. Load takes about 0.09 s once the weights are in the page cache.

**Surprises**

- We do not beat ONNX Runtime, which was the target. On one thread we are 1 to 7% slower at every `T`. On four threads the medians are 4 to 30% slower and the minimums are mixed (ours is better at T=256, worse elsewhere); four-thread medians in this power state swing by a factor of two between runs, so that comparison is not settled.
- Our matrix products alone are faster than ONNX Runtime's `MatMul` alone, yet the whole model is slower. Two reasons. ONNX Runtime inside the model does better than its standalone `MatMul` (its whole-model rate at T=1024 is 54.6 GFLOP/s, above the 50.0 it shows alone, so the standalone number undersells it, and one shape there measured an implausible 25.6). And our time outside matrix products is about 0.2 s at T=1024, 4% of the run: permutes, slices, softmax, GELU.
- Both ONNX Runtime and we are at the core's limit. Its best shapes reach 60 to 61 GFLOP/s where ours reach 58, against a 59 to 62 in-cache peak. A spike showed the outer-product (BLIS-style) inner loop peaks about 5% above the dot-product one (62.4 against 59.4). That 5% is the price of the no-packing design, in this power state.
- `noalias` alone did nothing (1.050 to 1.047 s), although M2 found that aliasing was why clang would not vectorize `gemm`. With `__restrict` clang's report changes, but the old `i, p, j` nest still runs at the same speed. I did not find out why. Leave-one-out is also inconclusive on it (slower without at T=64, faster without at T=256).
- `layout` on its own is a large slowdown: with the vectorizer off, the dot-form model takes 2.6 s at T=16 against 1.05 s for the M3 build. A dot product is a float reduction, which the C++ compiler may not reorder and therefore cannot vectorize. The layout only pays once our own vectorizer, which is allowed to reorder, is on.
- `regtile` is the largest single step (3.3x). One dot product has one accumulator, and the next add must wait three cycles for the previous one; twenty accumulators keep all four FMA units busy.
- `cachetile` adds 0 to 4% on one thread, though leave-one-out at T=64 on four threads shows 0.089 against 0.149 s. In Low Power Mode the core is slow relative to memory, so cache misses cost less. Expect it to matter more on AC.
- The first version of the vectorizer refused the tiled loop: it counted the other 19 accumulators in the same buffer as conflicts. The model was correct and ran at 1.47 s at T=16, slower than M3. The check now reasons per element. A wrong legality test that errs towards "no" costs speed silently, so the tests now assert which loops are vectorized.
- Leftover rows matter more than expected. A 64-column product with a 5-wide tile leaves 4 columns, 6% of the work, and done one column at a time they ran at a third of the speed. Leftovers are now jammed into a smaller tile and the tile shape is chosen per product to leave little over (49 to 57 GFLOP/s on that shape).
- Four-thread runs at T=1024 are throttled: the first run of a session takes 1.55 s and later ones about 3.2 s. The 1.547 s in the attribution table and the 3187 ms in the comparison table are the same build.
- Low Power Mode was on for this whole session, and probably for the earlier "battery" sessions. It explains why M3's numbers were 1.7x slower than M2's.

**Deviations from the plan**

- No operand packing and no micro-kernel template (STEERING.md section 4.3 has the reasoning). The cost is the 5% above and 340 MB of transposed weights.
- Rewrites not built because nothing needed them: `fuse`, `unroll` on its own, a general `reorder`.
- Cache sizes are a hand-picked constant, not read from `sysctl`. That moves to M5.

**Open**

- Re-measure everything on AC with Low Power Mode off: `bench/run.py --systems eager compile ort mlc --name m4`, `bench/gemm.py`, `bench/ablate.py`.
- The remaining gap to ONNX Runtime: the non-matmul 4%, the small-`k` product (`q @ k^T`, 49 against its 59 GFLOP/s), and four-thread behaviour at small `T`.
- Why `noalias` does not help the unvectorized nest.
- 340 MB of persistent memory for transposed weights.

## M5: autotuner (2026-11-14)

**Built**

- The matrix product's schedule is now data: `MatmulParams` (register tile, which operand is outside, block rows). `default_matmul_params` computes M4's hand-picked choice; `schedule_kernel` applies any choice with the same rewrites. A kernel carries a signature (op and operand shapes) and `ScheduleOptions::tuned` maps signatures to parameters.
- `src/tune/`: machine detection (`sysctl` for the CPU name, L1 size, performance cores, power state), the search space, batch measurement, the search, and a JSON cache.
- `mlc tune <graph> --cache tuning/cache.json -o <dir> [--threads 1,4] [--tune-threshold]`, and `--tuning <cache> [--tuned-threads N]` on `compile`, `loops`, and `gemm`.
- `tuning/cache.json`: 54 entries (7 products, fewer at T=64 where two share a shape, x 4 sequence lengths x 2 thread counts).
- `bench/run.py --systems ... mlc-tuned`, and a ctest pair that builds from the cache and checks the logits.
- Tests: 103 unit tests (5 new). One runs every point of a product's search space, over 100 distinct schedules, through the loop interpreter and requires bit-identical results. The sanitizer build was not rerun for this milestone.

**Numbers**

Measured on AC power, Low Power Mode off: the first session in that state since M2. One core now runs the large products at about 100 GFLOP/s (59 yesterday).

The search tried 33 of 462 points per product and thread count. It replaced the default in 12 of 54 entries. Largest gains in the final round: the T=1024 projection `[1024,3072] x [768,3072]^T` on four threads, 287 to 332 GFLOP/s (5x4 tile, A outside, blocks of 5), and the T=16 output projection on four threads, 296 to 348.

Whole model, default against tuned schedules in alternating runs (`mlc tune`), median seconds:

| | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| 1 thread, default | 0.0473 | 0.1806 | 0.7038 | 3.1761 |
| 1 thread, tuned | 0.0466 | 0.1814 | 0.6998 | 3.1672 |
| 4 threads, default | 0.0228 | 0.0628 | 0.2274 | 0.9868 |
| 4 threads, tuned | 0.0222 | 0.0634 | 0.2300 | 0.9518 |

Separate session through the Python harness (`bench/results/m5.md`), median ms with the minimum in parentheses:

| 1 thread | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| ONNX Runtime | 43.8 (42.1) | 167.5 (161.9) | 685.9 (675.0) | 3208.3 (3187.1) |
| mlc, default | 47.6 (45.9) | 183.0 (174.7) | 709.3 (689.7) | 3203.3 (3185.7) |
| mlc, tuned | 47.0 (44.9) | 179.6 (173.8) | 701.7 (690.3) | 3235.8 (3169.8) |

| 4 threads | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| ONNX Runtime | 14.7 (14.0) | 46.3 (46.1) | 189.3 (184.3) | 1020.1 (1005.4) |
| mlc, default | 22.4 (17.5) | 55.4 (53.6) | 244.6 (212.2) | 1087.7 (1034.9) |
| mlc, tuned | 23.6 (18.3) | 62.4 (55.3) | 219.4 (206.9) | 1049.5 (997.2) |

The thread threshold (`--tune-threshold`, T=64, four threads) kept the default 65536; the five candidates gave 0.056 to 0.076 s in no order, which says the measurement is noisier than the effect.

**Surprises**

- The exit criterion, "tuned schedules beat hand-picked defaults on the model", is met only weakly. Tuned is ahead by 1.5 to 3.7% at T=16 and at T=1024 on four threads, and within about 1% either way everywhere else, which is noise. In the harness session the tuned build is slower than the default in three of eight cells. The honest summary: on this model the defaults were already at the limit, and the tuner mostly confirmed them. Reproducibility from the cache does hold: the cached build compiles without measuring and passes the golden test.
- Why so little: for the large products every sensible schedule runs at 97 to 101 GFLOP/s on one core. The kernel is bound by the FMA units, not by the cache, so there is nothing for a cache-shape search to find. The products that did improve are the memory-bound ones (the 154 MB output projection, the 3072-wide projection) on four threads.
- Mutation found almost nothing that random sampling had not. In most products the best point after two generations is the best random point.
- A real bug on the way: candidate libraries were rebuilt at the same path, and the tuner reported nonsense (a "winner" 13% slower than the search's best). macOS does not unload a library that has thread-local state, so `dlopen` of a path seen before returned the old code. Every candidate library now gets a fresh path.
- With a 1% acceptance margin the tuner "found" improvements that a second run did not reproduce. The margin is 2% now. A tuner that fits noise produces a cache that is worse than the defaults on average.
- On AC the picture against ONNX Runtime is clearer than it was in Low Power Mode. One thread: level at T=1024, 3 to 9% behind below that. Four threads: 3 to 50% behind, worst at T=16 (22.4 against 14.7 ms). The gap is no longer in the matrix products. It is everything else, and thread start-up at small sizes.

**Deviations from the plan**

- Only matrix products are tuned per kernel. The plan listed unroll factors and the parallel loop choice as well; neither showed any measurable effect when tried by hand.
- The cache is keyed by CPU name, not by a fuller description (cache sizes, power state). The power state is recorded but not checked.

**Open**

- The remaining gap to ONNX Runtime at four threads and small `T`: profile where the 22 ms at T=16 goes (the matrix products alone account for about 11 ms at 370 GFLOP/s).
- Re-run `bench/run.py`, `bench/gemm.py`, and `bench/ablate.py` on AC so M4's tables match this session's power state.
- Sanitizer build for M5.

## M6: final CPU benchmark and writeup (2026-11-14)

**Built**

- `bench/per_op.py`: time per kind of operation at every `T`, from `mlc exec --profile`, which now takes each kernel's best time over all runs instead of one run's.
- `docs/writeup.md`: the comparison, the per-op breakdown, the matrix product alone, both ablations, and what they mean.
- Every result table re-measured in one afternoon on AC with Low Power Mode off: `final.md`, `gemm_T16.md`, `gemm_T1024.md`, `per_op.md`, `per_op_tuned.md`, `m4_ablation_*.md`, `m3_ablation_cumulative.md`, `m3_ablation_T1024.md`.
- Checked: 103 unit tests in release and under ASan + UBSan; sanitized end-to-end runs at T=16; the full release `ctest`, including T=256 and T=1024, before the last two source changes (the profiling minimum and a warning fix in `lower.cpp`), after which the unit tests and the benchmark's own golden checks at every `T` passed.

**Numbers**

In `docs/writeup.md`. The headline: against ONNX Runtime, level at T=1024 on one and four threads; 1 to 6% behind below that on one thread; 12, 40, and 57% behind at T=256, 64, and 16 on four.

**Surprises**

- The kernel comparison reversed with the power state. In Low Power Mode our matrix products alone beat ONNX Runtime's at T=1024 (56.9 against 50.0 GFLOP/s). On AC it wins, 103.1 against 94.8. A slower clock hides memory and load-port effects, and a result from a throttled machine does not carry over.
- M5's note that the small-`T` gap "is no longer in the matrix products" was wrong. The per-op table shows they are 97% of the four-thread run at T=16. What is different in the model is that the weights are cold: 494 MB read once per call. Alone, at 370 GFLOP/s, the dense layers would take 7 ms; in the model they take 10.5.
- Software prefetch for that case, tried in a hand-written copy of the kernel over 48 cold matrices, was 15 to 30% slower on one thread and 13 to 30% slower on four. Not added.
- `bench/profile.py`, as the per-op script was first called, shadowed Python's `profile` module and made `bench/run.py` fail inside `torch`. The first full benchmark run silently produced no table because the log was filtered. Renamed to `per_op.py`.
- Graph passes still buy no measurable speed after M4. M3's log said to re-check once matrix products stopped hiding everything else; they never did.

**Open, if the CPU side is taken further**

- The four items at the end of `docs/writeup.md`.
- M7 (Triton backend) needs a Linux machine with an NVIDIA GPU, which this project does not have yet.

