# Steering document

This is the source of truth for what we are building, in what order, and how we decide a stage is done. When the code and this document disagree, fix one of them in the same change.

## 1. Goal

Build a small ahead-of-time ML compiler that takes GPT-2 small (124M parameters) from PyTorch to a native CPU binary, and measure it honestly against PyTorch eager, `torch.compile` (CPU), and ONNX Runtime. Then add a GPU backend that emits Triton kernels.

The project exists to learn how ML compilers work and to get hands-on performance engineering experience. When a choice is between the quicker route and the one that teaches more about compilers or performance, take the second, and write down what was learned.

```
PyTorch model
  -> torch.export (ATen graph)      Python exporter: graph.json + weights.bin
  -> Graph IR            constant folding, fusion, layout, memory planning
  -> Loop IR             tiling, vectorization, multi-threading
  -> C++ with SIMD intrinsics  -> clang++ -> shared library
  -> autotuner picks kernel parameters by measurement
```

### Non-goals

- Training, autograd, or dynamic control flow.
- Arbitrary models. GPT-2 small is the target; generality is welcome only where it is free.
- Quantization or mixed precision. Everything is fp32.
- Calling a BLAS library from generated code. The kernels are ours; that is the point.

## 2. Target machine and what it implies

Development machine: Apple M2, arm64, 4 performance + 4 efficiency cores, 24 GB, Apple clang 21.

| Fact | Consequence |
|---|---|
| arm64, not x86 | SIMD target is **NEON** (128-bit, 4 x fp32, FMA via `vfmaq_f32`). No AVX. Codegen goes through a thin ISA layer so AVX2 can be added later without touching the loop IR. |
| Apple clang ships no OpenMP | Threading uses our own small `std::thread` pool with a static `parallel_for`. No `libomp` dependency. |
| 4P + 4E cores, no thread affinity API on macOS | Default to one thread per performance core and request high QoS so work lands on them. Thread count is a tunable and is reported with every number. |
| A laptop: battery power and Low Power Mode change the clock | Low Power Mode roughly halves single-core speed (measured in M4: 59 against an expected 110 GFLOP/s peak) and throttles sustained multi-core load. Every result file records the power state; numbers are compared only within one session. |
| PyTorch and ONNX Runtime on macOS can route GEMM through Apple Accelerate | Accelerate may use hardware that NEON intrinsics cannot reach. Matching it on large GEMM may not be possible. See section 7 for how we report this. |
| No NVIDIA GPU | Triton does not run here. The GPU milestone needs a Linux + CUDA machine (cloud or Colab). |

Cache sizes for tiling come from `sysctl hw.perflevel0.*` at tune time (M5). Until then the default schedule carries one hand-picked constant, a 64 KB working set against the 128 KB L1 data cache, chosen by measurement in M4.

## 3. Scope decisions

These are fixed unless this section is edited.

1. **Compiler in C++20, output in C++17.** Both IRs, all passes, lowering, codegen, and the autotuner are C++, built with CMake. The compiler is a command-line tool, `mlc`, that reads an exported graph, emits C++ text, and builds it with `clang++ -O3 -mcpu=native` into a shared library.
2. **Python only at the edges.** `torch.export` exists only in Python, so a thin exporter script writes the ATen graph as JSON and the weights as a flat binary. The baselines and the benchmark harness are also Python, and load our `.so` through `ctypes`. No compiler logic lives in Python.
3. **Static shapes.** Batch 1, fixed sequence length `T`. Each `T` is a separate compile. This is what makes constant folding and memory planning effective.
4. **Workload.** Primary: full forward pass, token ids to logits, at `T` in {16, 64, 256, 1024}. Stretch: a KV-cached single-token decode step, reported as tokens per second.
5. **Weights live outside the code.** One flat binary file, memory-mapped at load. Generated C++ contains offsets, never weight literals.
6. **Model source.** Hugging Face `GPT2LMHeadModel` with `use_cache=False`, wrapped to return logits only, exported with `torch.export` and decomposed to core ATen ops.
7. **End-to-end first.** A slow, correct scalar pipeline exists before any optimization work starts. Every later change is measured against it.

## 4. Architecture

### 4.1 Frontend

Two halves. The Python exporter walks the `ExportedProgram` and serializes it without interpreting it: nodes, ATen op names, attributes, shapes, dtypes, and weight offsets. The C++ importer reads that file and lowers ATen ops to the graph IR. An unsupported op is a hard error naming the op; no silent fallbacks. The export uses 37 distinct ATen operators (listed in `docs/export_ops.md`); the importer lowers all of them, and the table of lowerings in `src/frontend/importer.cpp` is the complete accepted surface. Arguments are held to the same rule as operators: an argument or keyword the lowering does not read is an error. Every lowered node's inferred type is compared with the type `torch.export` recorded, so our shape inference is checked against PyTorch's on the real model at every import.

### 4.2 Graph IR

- SSA values with static shape and dtype. Nodes are pure. Constants are first-class.
- One node produces one value, so a value is named by its node's index. Tuple-returning ATen ops are resolved in the importer.
- 27 ops (`src/graph/ir.h`). Composite ops (`layer_norm`, `softmax`, `gemm` with bias) are kept whole. `gelu`, the causal form of `softmax`, and the transposed-operand form of `matmul` and `gemm` (`trans_b`) are never imported; passes produce them.
- One function, `infer_type`, defines every op's typing rule. The builder uses it to type new nodes and the verifier uses it to re-check existing ones.
- A text printer (`mlc dump`) and a verifier (shapes, dtypes, use-before-def). The verifier runs after import and before every interpreter run, and will run after every pass.
- A **reference interpreter** in plain, unoptimized C++ executes any graph (`mlc run`). It is checked against PyTorch's golden logits and is then the oracle for every later stage. It frees each value after its last use and reports peak live bytes, which is the baseline memory planning has to beat.

Passes (`src/graph/passes.*`), in pipeline order. Each is a function from graph to graph built on one engine, `rewrite`, which walks the old graph in order and lets a rule replace each node while looking at operands that have already been rewritten. Dead code elimination runs after every pass, and so does the verifier.

| Pass | What it does for GPT-2 |
|---|---|
| `simplify` | Removes no-op `expand`, reshape, permute, slice; merges reshape chains; composes permute chains (the head merge is three permutes that become one); `x + 0`, `x * 1`, `not(not(x))`. |
| `fold` | Evaluates nodes whose operands are all inline constants: position ids, the causal mask. Never folds external weights (their bytes stay in the weights file) or `Broadcast` (it would inflate a scalar), and skips results over 1 MB. |
| `cse` | Merges identical nodes and identical constants. Each layer rebuilds the same mask and the same scalars. |
| `gelu` | Recognizes the nine-node tanh GELU expansion and emits one `Gelu`. |
| `attention` | Recognizes mask + softmax + all-masked-row guard and emits one causal `Softmax` on the raw scores. The mask is recognized by evaluating it at compile time, whatever computes it. |
| `layout` | Rewrites every matrix product `a @ b` to take `b` transposed (`trans_b`), so each output element is a dot product of two contiguous rows. Transposes already in the graph cancel (the output projection, `q @ k^T`); a transpose of a weight is left for `hoist`. |
| `memory` | Applied at lowering (`src/loop/lower.cpp`). Liveness per value, in-place reuse for ops that read each element before writing it, then best-fit offset assignment with a coalescing free list. Reports arena bytes against the no-reuse total. |
| `hoist` | Applied at lowering. Staging: a value that depends only on weights and constants is computed once when the model is loaded, by an init kernel writing persistent memory, instead of on every call. It is constant folding for values whose bytes the compiler does not have. |

Every optimization at every level can be switched off by name (`--disable`), which is how ablations are built (`bench/ablate.py`). The loop-level names are `noalias`, `vectorize`, `vmath`, `regtile`, `cachetile`, `parallel` (sections 4.3 and 4.4).

Fusions from the original plan that turned out not to be needed, or not yet:

- **matmul + bias to `gemm`**: ATen already exports `addmm`, which imports as `gemm`.
- **Residual add into the producer**: in-place reuse already makes the add write over its operand, so it costs no memory, and after M4 all 25 adds together are 0.2% of the run. Not done.
- **General elementwise chains**: after `gelu` and `attention`, the only elementwise ops left in GPT-2 are the 25 residual and embedding adds and 24 multiplies by the attention scale. There is no chain left to fuse.
- **The transposed output projection**: the export transposes the tied embedding (154 MB) on every call before the final `mm`. Resolved in M4 by `layout`: the product reads its second operand transposed, so the two transposes cancel and the embedding table is used as stored.

### 4.3 Loop IR

- Statements: `For(var, extent, kind)`, `Store`, `Block`. Expressions: immediates, loop variables, `Load`, arithmetic and comparisons, `Select`, `Cast`, intrinsic calls (`exp`, `tanh`, `pow`, `sqrt`).
- A loop's extent is a constant or an expression of enclosing loop variables (the causal softmax loops over columns `0..i`). Its `kind` is serial, vectorized, or parallel. The kind never changes what the loop computes: it is a promise, checked when made, that the faster execution gives the same values up to the order of float additions.
- Accesses are multi-dimensional (`buf[i, j, k]`); flattening to one offset happens at codegen.
- There are no scalar variables besides loop counters. Reductions accumulate through memory and temporaries are small buffers, so every dependence is a visible load or store. That is what the legality analysis reads. Registers appear only in generated code: a vectorized reduction keeps its accumulator in vector registers.
- A buffer is a typed, shaped view of storage: the arena, the weights file, persistent memory (filled at load time), a caller's input or output, or inline constant data. A graph `Reshape` lowers to a second view of the same storage and no code.
- Storage and stage are decided at lowering. With memory planning off, every computed value gets its own slice of the arena; with it on (the default), slices are reused once dead. A program has two kernel lists: init kernels, run once at load, and kernels, run per call.
- Each graph op lowers to a naive loop nest (`src/loop/lower.cpp`), one kernel per node. Optimizations are explicit rewrites on that nest (`src/loop/schedule.*`): `split`, `interchange`, `unroll_and_jam`, `vectorize`, `parallelize`. Cache tiling is `split` followed by `interchange`. A rewrite names its loop by the loop variable and either proves itself legal or throws and changes nothing.
- **Legality is dependence analysis.** A loop's iterations are independent when, for every buffer the body writes, there is an axis along which every access is at `c * v + r` with the same stride `c` and `r` ranging over less than `c`: each iteration owns a slice. `interchange`, `unroll_and_jam`, and `parallelize` require it. `vectorize` requires each store to be either elementwise over consecutive addresses or a sum or maximum into one element that nothing else in the loop touches. Two buffers that share bytes under different shapes are treated as dependent.
- A **schedule** is which rewrites to apply and with what parameters. M4 has one default schedule (`schedule_program`, parameters in `ScheduleOptions`): vectorize every innermost loop that can be, run the outermost loop of every large kernel on threads, and tile matrix products as below. The autotuner (M5) searches over the parameters; it never edits IR directly.
- A loop IR interpreter (slow, tree-walking, bounds-checking every index) is the reference for what a loop program means. It runs every loop serially whatever its kind, so a scheduled nest must compute bit for bit what the naive nest did, and the tests check exactly that. It is far too slow for the full model, which is checked in compiled form against the golden logits.

**The matrix product.** GEMM is 97% of the unoptimized run (768x768, 768x3072, and the 768x50257 output projection). The original plan was the BLIS design: pack both operands and call a micro-kernel generated from a template. What was built instead needs no packing and no template:

- `layout` puts every product in dot form, `C[i,j] = sum_p A[i,p] * Bt[j,p]`, where both operands are read along contiguous rows.
- `unroll_and_jam` on `i` and `j` turns the dot-product loop into a tile of `rows x cols` dot products that share their loads (4x5 by default: 20 accumulators, 9 loads, 29 of the 32 vector registers). `vectorize` on `p` keeps each accumulator in a register. That loop is the micro-kernel, and it came out of two general rewrites.
- Cache tiling blocks the operand with more rows so that a block, plus the rows it is multiplied against, stays in L1. The other operand streams from L2.
- Leftover rows and columns are jammed too, into a smaller tile, because a leftover handled one row at a time has too few accumulators to keep the FMA units busy.

Measured in isolation (`mlc gemm`, `bench/gemm.py`) these kernels reach 92 to 97% of the same loop's in-cache rate. Why not BLIS: packing a weight costs as much as multiplying by it at `T=16`, and the alternative of packing at load time costs the same memory as transposing at load time, which dot form needs for 48 of the 73 products and not for the largest one.

### 4.4 Codegen and runtime

- Loop IR to C++ (`mlc compile`). A serial loop is transcribed. A vectorized loop becomes a loop of SIMD intrinsics, with `interleave` independent accumulator sets for reductions, followed by a scalar loop for leftover iterations; `a + b * c` is emitted as one fused multiply-add. A parallel loop becomes a function of an index range, passed to the runtime's thread pool.
- The instruction set is one table (`src/codegen/isa.h`): the name of each vector operation. NEON is the only entry. Vector `exp` and `tanh` are handwritten polynomials in `runtime/mlc_neon.h`, accurate to 2.5e-7 and tested against the C library.
- Each kernel's buffers reach the loops as `__restrict` parameters wherever the memory plan shows they do not overlap (`noalias`). In M2 the C++ compiler refused to vectorize 48 of 49 matrix products because it could not prove that; the planner can.
- The runtime linked into generated code (`runtime/mlc_runtime.*`) is handwritten and does not grow kernels: arena and persistent allocation, weight mmap, the init kernels at load, per-kernel timing, and a thread pool (`std::thread`, one contiguous piece of the range per thread, workers spin briefly before sleeping). It exposes a small C API (`mlc_open`, `mlc_set_threads`, `mlc_run`, `mlc_run_profiled`, `mlc_close`) that Python calls through `ctypes` and `mlc exec` calls through `dlopen`.
- Generated code is kept on disk and readable. Being able to read the output is a debugging feature.

### 4.5 Autotuner

`src/tune/`, driven by `mlc tune`. It chooses schedule parameters by timing candidates, and records the result so a later compile needs no measuring.

- **What is tuned:** the schedule of each distinct matrix product (`MatmulParams`): the register tile, which operand is blocked, and the block size. Optionally one global parameter, the size below which a kernel stays on one thread (`--tune-threshold`). Nothing else: every other schedule choice is forced by legality or does not show in a measurement.
- **The space** is discrete and small (a few hundred points per product): every tile that fits the 32 vector registers with at least 12 accumulators, block sizes as fractions of the L1 data cache (read from `sysctl hw.perflevel0.l1dcachesize`), and either operand outside. Every point is a legal schedule; a test runs all of them through the loop interpreter.
- **A kernel instance** is identified by its signature, the op and operand shapes (`gemm [16,768] x [2304,768]^T`). Kernels with the same signature are the same loops over different memory, so one measurement serves all twelve layers.
- **Measuring:** a batch of candidates for one product is compiled into one shared library, the same product several times under different schedules, and timed in-process through `dlopen` with the runtime's per-kernel timer. All candidates in a batch run in turn within each pass, and the minimum over passes is kept.
- **Search:** the hand-picked default plus random points, then generations of single-step mutations of the best three. Then a final: the best three and the default measured again for three times as long. A candidate replaces the default only if it wins the final by more than 2%.
- **Validation:** the whole model is built with default and with tuned schedules and the two are timed in alternating runs.
- **Cache:** `tuning/cache.json`, keyed by CPU, thread count, and signature, committed. `mlc compile --tuning tuning/cache.json --tuned-threads N` reproduces the tuned build. A cache measured on another CPU is ignored with a warning.
- A learned cost model is out of scope.

### 4.6 GPU backend (last)

Reuses the frontend and graph IR unchanged. Lowers fused graph regions to Triton kernels instead of loop IR; `mlc` emits the Triton Python source as text. Requires Linux + CUDA. Compared against PyTorch eager and `torch.compile` on the same GPU.

## 5. Milestones

Each milestone ends with a short entry in `docs/log.md`: what was built, the numbers, what surprised us.

| # | Milestone | Exit criteria |
|---|---|---|
| M0 | Baselines and harness | Environment pinned. GPT-2 exports. Golden inputs and logits saved. Benchmark harness produces a table for all three baselines at every `T`. |
| M1 | Frontend, graph IR, interpreter | Export lowers with no unsupported ops. Interpreter logits match PyTorch within tolerance at every `T`. |
| M2 | Naive end to end | Graph to loop IR to scalar C++ to `.so`. Logits correct. Speed recorded as the "unoptimized" row. |
| M3 | Graph optimizations | Folding, fusion, memory planning. Node count and peak memory before and after. Still correct. Speedup attributed per pass. |
| M4 | Loop optimizations | Tiling, NEON vectorization, threading. Standalone GEMM GFLOP/s measured for 1 and 4 threads. Model speedup attributed per transformation. |
| M5 | Autotuner | Tuned schedules beat hand-picked defaults on the model, and the result is reproducible from the cache. |
| M6 | Final CPU benchmark and writeup | Full comparison table, ablation table, and a per-op time breakdown. README results filled in. |
| M7 | Triton backend | Correct logits on GPU. Comparison against PyTorch GPU baselines. |
| Stretch | KV-cached decode | Tokens per second against the same baselines running cached generation. |

## 6. Correctness

- **Oracle chain:** PyTorch -> graph interpreter -> loop interpreter -> compiled binary. Each stage is tested against the one before it.
- **Tolerance:** max absolute logit difference at or below `1e-3` and identical argmax at every position, on fixed golden inputs. Reordered fp32 summation makes bitwise equality impossible. If a change needs a looser bound, that is a finding to investigate, not a constant to bump.
- **Per pass:** every graph pass and every loop rewrite has a differential test (interpret before, interpret after, compare).
- **Per kernel:** each generated kernel is tested standalone against a naive scalar reference on random inputs, including sizes that are not multiples of the vector width or tile size.
- Build both the compiler and the generated code with `-fsanitize=address,undefined` in the test configuration.
- `-ffast-math` is off by default. If it is ever enabled it is a named, benchmarked option. The C++ compiler therefore never reorders float arithmetic. Our own vectorizer does, in one place: a vectorized sum adds its terms in lanes. That is a decision made in the loop IR, visible in `mlc loops`, and it is what the tolerance above is for.

## 7. Benchmarking rules

- Same machine, on AC power with Low Power Mode off, nothing else running. Same inputs, fp32, batch 1. Systems are compared within one run of the harness, never across sessions.
- Baselines run under `torch.inference_mode()`. `torch.compile` and ONNX Runtime get their warmup and compile excluded, as do we.
- Thread count is set explicitly and equally for every system. Report 1-thread and 4-thread results separately. Note that Accelerate's GEMM does not respond to this setting, so PyTorch's "1 thread" row is not a single-core NEON number.
- At least 5 warmup runs, then at least 30 timed runs. Report median and min. Keep the raw samples.
- Record library versions, the BLAS backend each baseline is actually using, and compiler flags next to every table.
- Every optimization claim has an ablation: the same build with that one thing turned off.

**What M0 measured (on battery; to be confirmed on AC).** PyTorch's GEMM goes through Accelerate and reaches roughly 700 GFLOP/s from a single thread with CPU time equal to wall time. That is several times what one core's NEON units can do, so it is not ordinary SIMD and the thread count setting does not govern it. ONNX Runtime uses its own CPU kernels and lands near 55 GFLOP/s on one thread at `T=1024`, which is what a good NEON kernel looks like. This gives the project two reference points:

- **ONNX Runtime is the peer.** Same class of hardware path as our generated code. Target: beat it at every `T`, at 1 and 4 threads.
- **PyTorch on Accelerate is the ceiling on GEMM.** We do not expect to match its matrix multiply. At small `T` it is beatable, because ONNX Runtime already beats it at `T=16` with 4 threads. At large `T` even PyTorch spends most of its time outside GEMM (its whole-model rate is far below its GEMM rate), so fused attention, softmax, and GELU are where we can close distance.

**Setting expectations.** The realistic places to win are per-op overhead, fusion, memory reuse, and shape specialization, which matter most at small `T`. At large `T` the result is decided by GEMM throughput, where the baselines sit on a vendor-tuned library. We set the numeric target after M0 measures the baselines, and we report losses as plainly as wins, with the per-op breakdown that explains them. If an x86 Linux machine is available, repeating the comparison there (AVX2 against MKL or OpenBLAS) is a fairer kernel-to-kernel contest and is worth doing.

## 8. Repository layout

```
export/         Python: torch.export -> graph.json + weights.bin, golden logits
src/
  frontend/     graph.json -> graph IR
  graph/        IR, verifier, printer, interpreter, passes/
  loop/         IR, interpreter, lowering, schedule rewrites
  codegen/      C++ emitter, ISA layer (neon), build driver
  tune/         the autotuner: search space, measurement, search, cache
  gpu/          Triton emitter (M7)
  main.cpp      the `mlc` driver
runtime/        handwritten C++ linked into generated code: arena, thread pool, weight loader, NEON vector math
tuning/         cache.json: schedules measured by `mlc tune` on the development machine
bench/          Python: harness, baselines, result tables
  support/      JSON reader, mmap, .npy reader
tests/          C++ unit and differential tests; end-to-end checks against golden logits (ctest)
docs/           log.md, notes
build/          CMake build trees (`build/release`, `build/asan`); generated C++ and libraries under `build/release/gen/` (gitignored)
```

The compiler depends on the C++ standard library and nothing else. The JSON reader (`src/support/json.*`) and the test macros (`tests/check.h`) are our own, a few hundred lines together; vendoring a library for either would cost more in build time than it saves. No LLVM, no MLIR.

## 9. Risks and open questions

| Item | Plan |
|---|---|
| The output projection's weight is transposed at run time | Resolved in M4 (`layout`, section 4.2). |
| GEMM cannot match Accelerate on M-series | Confirmed in M0 and measured in M4 (`bench/results/gemm_T*.md`): Accelerate is several times faster than any NEON kernel on large products. We report the gap. |
| Transposed weights double the weights' memory | Open. 340 MB of persistent memory holds transposed copies of the 48 dense-layer weights. The originals stay in the mapped file and can be paged out, but the address space and load time (about 0.1 s) are paid. Alternatives for M5: store the weights transposed in the weights file, or transpose per call in cache-sized blocks. |
| Numbers so far were taken on battery in Low Power Mode | Open. All of M4's tables are internally consistent but about half speed, and sustained four-thread runs at `T=1024` are throttled. Re-measure on AC before M6. |
| `torch.compile` CPU backend on macOS arm64 may be fragile | Resolved in M0: it runs and matches golden logits. |
| Export produces ops or patterns we did not expect | Resolved in M1: all 37 ops lower. The surprises (tied embedding transposed at run time, a mask computed from constants on every call, no-op expands, permute chains) are recorded in `docs/log.md` and are M3's work list. |
| C++ slows iteration on pattern-matching passes | Build a small pattern-matching helper and the IR printer early, in M1. Keep debug builds fast. |
| Fusion patterns are brittle against export changes | Pin `torch` and `transformers` versions. Pattern-match on our canonicalized IR, not raw ATen. |
| Autotuning noise on a laptop | Min-of-N timing with candidates interleaved in one process, a longer final round against the default, and a 2% margin before the default is replaced. |
| Where M7 runs | Open. Needs a Linux + CUDA machine. Decide before M6 ends. |
| Is the KV-cache decode step in scope | Open. It needs a second export and state passed between calls. Decide after M4. |

## 10. Working rules

- Small commits, each leaving tests green.
- No optimization lands without a before and after number.
- No number goes in the README without the command that reproduces it.
- Version-specific facts about PyTorch, ONNX Runtime, Triton, or Apple hardware are checked against current documentation or measured, not recalled.
