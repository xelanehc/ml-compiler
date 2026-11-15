# 12. Testing, benchmarking, and the results

Two questions remain about any compiler: is its output right, and how fast is it? This chapter covers how each was answered, and then goes through the final numbers.

## 12.1 The kinds of test

**Unit tests** (`tests/*.cpp`, 103 of them, run by the `mlc_tests` program). The test "framework" is one header, `tests/check.h`, about 70 lines:

- `TEST(name) { ... }` defines a function and a static object whose constructor adds the function to a global list before `main` runs.
- `CHECK(cond)` records a failure and keeps going, so one run reports every problem.
- `CHECK_NEAR(a, b, tol)` and `CHECK_THROWS(expr, ExceptionType)` do what they say.
- `mlc_tests gelu` runs only the tests whose name contains `gelu`.

**Differential tests.** Most unit tests do not compare against hand-computed answers. They compute the same thing two ways and compare. `tests/helpers.h` has the building blocks:

- `lowering_matches(graph, outputs)`: run the graph interpreter; lower the graph and run the loop interpreter; compare every output. Done twice, with staging off and on.
- `same_after(...)` in the schedule tests: apply a loop rewrite and require the loop interpreter to give **bit-identical** results. This is possible because the loop interpreter runs every loop serially in order, and none of the rewrites changes the order in which any one element's terms are added.
- `tiny_transformer(g, ids)`: one attention block and one MLP block at toy size (5 tokens, width 8, 2 heads), built from the same operators in the same arrangement as the real export, including the nine-node GELU and the mask guard. It goes through every stage: graph interpreter, optimized graph, loop interpreter, compiled code.

**Negative tests.** Things that must be *rejected*: a GELU lookalike with a wrong constant must not be fused; a loop with a carried dependence must not be vectorized or parallelized; a program that stores to read-only memory must fail verification; an out-of-range gather index must be caught. An optimizer that fires when it should not is worse than one that never fires.

**Every-point tests.** The autotuner can pick any schedule in its space, so one test applies every distinct point of a small product's space and checks each.

**Compiled-kernel tests at awkward sizes.** `compiled_kernels_match_the_graph_interpreter_at_awkward_sizes` builds real shared libraries for matrix products with 7 rows against tiles of 2, 11 columns against tiles of 3, and 13- and 9-element rows against 4 SIMD lanes, so that every leftover path in the generated code is exercised. It also checks that one thread and three threads give identical bits.

**End-to-end tests** (`ctest`, defined in `CMakeLists.txt`). For each `T`, on the real model against PyTorch's golden logits:

| Test | Checks |
|---|---|
| `golden_T*` | graph interpreter on the imported graph |
| `optimized_T*` | graph interpreter on the optimized graph |
| `compile_T*`, `compiled_T*` | the full pipeline to a shared library, run with the default thread count |
| `compiled_T16_one_thread` | the same library on one thread |
| `compile_tuned_T16`, `compiled_tuned_T16` | built from the autotuner's cache |
| `compile_unoptimized_T16`, `compiled_unoptimized_T16` | every optimization off (the baseline for ablations) |

T=256 and T=1024 are labelled `slow`; `ctest -LE slow` skips them.

**Sanitizers.** A second build configuration (`build/asan`, `-DMLC_SANITIZE=ON`) compiles everything with AddressSanitizer and UndefinedBehaviorSanitizer, which instrument the code to detect out-of-bounds accesses, use after free, and undefined arithmetic at run time. In that configuration the *generated* code is built with the sanitizers too, so a kernel that reads one element past a buffer is caught.

**Self-checks inside the compiler.** These run on every compile, not only in tests: the graph verifier after import and after every pass; the importer's comparison of every inferred type with PyTorch's; the loop verifier after lowering and after scheduling; legality checks in every rewrite.

## 12.2 The correctness criterion

From STEERING.md section 6, implemented in `compare_logits` (`src/main.cpp`): on the golden input, the maximum absolute difference in any logit is at most 1e-3, no logit is NaN or infinite, and the largest logit is at the same index at every position.

Observed across all builds: 1.3e-4 to 4.9e-4. ONNX Runtime's own difference from PyTorch is also 4.9e-4 at T=256.

## 12.3 How the benchmarks work

`bench/run.py` times whole forward passes. Rules (STEERING.md section 7):

- **Same model, same input.** Every system imports `export/model.py`.
- **A fresh process per (system, thread count).** Libraries size their thread pools at startup, and `torch.compile` bakes the thread count into generated code. `bench/worker.py` is that process.
- **Warm-up excluded.** Five untimed runs first. The first calls pay for one-time work: memory being faulted in, thread pools starting, `torch.compile` and ONNX Runtime doing their own compilation.
- **At least 30 timed runs**, and at least 2 seconds of measuring. Median and minimum are reported; raw samples go to a JSON file.
- **Thread counts set explicitly and equally**: 1 and 4.
- **Every system's output is checked** against the golden logits in the same run.
- **The environment is recorded**: library versions, which BLAS PyTorch uses, power state.

The four systems:

| System | What it is |
|---|---|
| PyTorch eager | the model run normally, under `torch.inference_mode()` |
| `torch.compile` | PyTorch's JIT compiler, with dynamic shapes disabled so it specializes per `T` as this project does |
| ONNX Runtime | the model exported to ONNX and run by Microsoft's inference engine, all graph optimizations on |
| mlc | this project, loaded through `ctypes`; `mlc-tuned` uses the autotuner's cache |

Supporting scripts:

| Script | Produces |
|---|---|
| `bench/gemm.py` | each distinct matrix product timed alone, in mlc, ONNX Runtime, and PyTorch |
| `bench/per_op.py` | time per kind of operation, from `mlc exec --profile` |
| `bench/ablate.py` | the model rebuilt with optimizations switched on one at a time (cumulative) or off one at a time (leave-one-out) |

An **ablation** answers "what is this optimization worth?" by building the same thing with and without it. Cumulative ablations show what each adds on top of the previous ones; leave-one-out shows what is lost without each. They can disagree when optimizations interact, which is why both exist.

## 12.4 Reading performance numbers honestly

Lessons this project paid for:

- **Compare only within one session.** The laptop's power state changed results by a factor of two (section 9.9).
- **Minimum for "what can the code do", median for "what will you see".** They diverge when something throttles.
- **A standalone kernel benchmark flatters.** It reuses warm data (section 12.6).
- **Differences under about 2% between sessions are noise** on this machine.
- **A result measured on a throttled machine can reverse** at full speed.

## 12.5 The final comparison

Apple M2, AC power, median milliseconds, all systems in one session (`bench/results/final.md`).

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

**Against ONNX Runtime** (the stated target was to beat it everywhere):

- One thread: 1% ahead at T=1024; 1 to 6% behind at shorter lengths.
- Four threads: level at T=1024; 12% behind at T=256, 40% at T=64, 57% at T=16.
- The target was not met. The compiler matches a mature engine at the longest length and loses elsewhere.

**Against PyTorch:** 1.6 to 4.5 times slower on one thread, because of Accelerate (section 9.10). Ahead only at T=16 on four threads.

**Against itself:** 38 s to 3.2 s at T=1024 on one thread since the first working build.

## 12.6 Explaining the numbers

**Where the time goes** (`bench/results/per_op.md`, one thread):

| | T=16 | T=1024 |
|---|---:|---:|
| dense layers (`gemm`) | 68.2% | 56.5% |
| attention products and output projection (`matmul`) | 29.9% | 39.5% |
| softmax | 0.1% | 1.7% |
| GELU | 1.1% | 1.1% |
| everything else | 0.7% | 1.2% |

Matrix products are 96 to 98% of every run. Everything about the comparison is about them.

**The matrix products alone** (`bench/results/gemm_T*.md`, GFLOP/s, weighted by the model's use):

| | T=16, 1 thr | T=16, 4 thr | T=1024, 1 thr | T=1024, 4 thr |
|---|---:|---:|---:|---:|
| mlc | 90.1 | 268.6 | 94.8 | 342.5 |
| ONNX Runtime | 71.6 | 237.9 | 103.1 | 376.1 |
| PyTorch (Accelerate) | 378.7 | 395.8 | 925.6 | 941.9 |

- **At T=1024, ONNX Runtime's kernel is about 9% faster** than ours on large products. This is the cost of the dot-product design without packing (section 10.4). In the whole model we are level anyway, so ONNX Runtime loses elsewhere what it gains here.
- **At T=16, our kernel is the faster one alone, yet the whole model is slower.** The standalone benchmark multiplies by the same weight matrix over and over, so the matrix stays in cache. In the model, each call reads all 494 MB of weights from main memory exactly once, and at T=16 uses each value only 16 times. The run is limited by memory bandwidth (section 9.4), not arithmetic. In the model our dense layers reach about 82 GFLOP/s on one thread and about 260 on four, against 100 and 370 alone.
- What ONNX Runtime does better in that situation was **not found**. Software prefetch (asking the CPU to fetch upcoming data early) was tried in a hand-written copy of the kernel over 48 cold matrices and made it 13 to 30% slower.

**Graph passes** (`m3_ablation_*.md`): no measurable effect on time (0.634 s to 0.637 s at T=16; 38.4 s to 37.7 s at T=1024). They are 3% of the story at most because they cannot touch matrix products. Their effect is on memory and size (section 6.11).

**Loop optimizations:** section 10.10.

## 12.7 Memory

| | T=16 | T=1024 |
|---|---:|---:|
| Weights file (mapped, shared with the system's file cache) | 498 MB | 498 MB |
| Arena, no reuse (M2) | 192 MB | 6453 MB |
| Arena, planned (M3) | 154 MB | 158 MB |
| Arena, final | 0.3 MB | 72 MB |
| Persistent (transposed weights), final | 340 MB | 343 MB |

The final design trades the 154 MB per-call transposed embedding for 340 MB of load-time transposed dense-layer weights. That is an open cost (section 13.2).

## 12.8 Where in the code

| File | Contents |
|---|---|
| `tests/check.h`, `test_main.cpp` | the test macros and runner |
| `tests/helpers.h` | graph builders, `lowering_matches`, `tiny_transformer` |
| `tests/test_*.cpp` | one file per component |
| `CMakeLists.txt` | build targets, `ctest` definitions, the sanitizer option |
| `bench/run.py`, `worker.py` | whole-model benchmark |
| `bench/gemm.py`, `per_op.py`, `ablate.py` | supporting measurements |
| `bench/results/` | every table, each with the command that regenerates it |
| `docs/writeup.md` | the results and interpretation in one place |

## 12.9 Further reading

- McKeeman, "Differential Testing for Software" (1998).
- Serebryany et al., "AddressSanitizer: A Fast Address Sanity Checker" (USENIX ATC 2012).
- Georges, Buytaert, and Eeckhout, "Statistically Rigorous Java Performance Evaluation" (OOPSLA 2007). Method for benchmarks with warm-up and noise; the advice applies to any runtime.
- Hoefler and Belli, "Scientific Benchmarking of Parallel Computing Systems" (SC 2015). Twelve rules for reporting performance results.
- Gregg, *Systems Performance* (2nd ed.), chapter 2 ("Methodologies") and chapter 12 ("Benchmarking").
- The ONNX Runtime documentation on "Thread management" and "Graph optimizations", to understand the peer.
