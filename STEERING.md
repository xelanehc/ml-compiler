# Steering document

This is the source of truth for what we are building, in what order, and how we decide a stage is done. When the code and this document disagree, fix one of them in the same change.

## 1. Goal

Build a small ahead-of-time ML compiler that takes GPT-2 small (124M parameters) from PyTorch to a native CPU binary, and measure it honestly against PyTorch eager, `torch.compile` (CPU), and ONNX Runtime. Then add a GPU backend that emits Triton kernels.

The project exists to learn how ML compilers work and to get hands-on performance engineering experience. When a choice is between the quicker route and the one that teaches more about compilers or performance, take the second, and write down what was learned.

```
PyTorch model
  -> torch.export (ATen graph)      Python exporter: graph.json + weights.bin
  -> Graph IR            constant folding, fusion, memory planning
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
| 4P + 4E cores, no thread affinity API on macOS | Default to 4 threads and request high QoS so work lands on P-cores. Thread count is a tunable and is reported with every number. |
| PyTorch and ONNX Runtime on macOS can route GEMM through Apple Accelerate | Accelerate may use hardware that NEON intrinsics cannot reach. Matching it on large GEMM may not be possible. See section 7 for how we report this. |
| No NVIDIA GPU | Triton does not run here. The GPU milestone needs a Linux + CUDA machine (cloud or Colab). |

Cache sizes for tiling come from `sysctl hw.perflevel0.*` at tune time, not from constants in the source.

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

Two halves. The Python exporter walks the `ExportedProgram` and serializes it without interpreting it: nodes, ATen op names, attributes, shapes, dtypes, and weight offsets. The C++ importer reads that file and lowers ATen ops to the graph IR. An unsupported op is a hard error naming the op; no silent fallbacks. The set GPT-2 needs is small (embedding, linear/addmm, matmul, layer norm, softmax, the tanh-form GELU, reshapes, permutes, split, masking, elementwise arithmetic). The exact list is established by dumping the export in M1, not assumed.

### 4.2 Graph IR

- SSA values with static shape and dtype. Nodes are pure. Constants are first-class.
- Roughly 15 to 20 ops. Composite ops (`gelu`, `layer_norm`, `softmax`, `gemm` with bias) exist so that fusion has something to produce.
- A text printer and a verifier (shapes, dtypes, use-before-def) run after every pass in debug mode.
- A **reference interpreter** in plain, unoptimized C++ executes any graph. It is checked against PyTorch's golden logits and is then the oracle for every later stage.

Passes, in order:

| Pass | What it does for GPT-2 |
|---|---|
| Canonicalize, DCE, CSE | Normalize export noise. |
| Constant folding | Position ids, the causal mask, attention scale factors, shape arithmetic. |
| Layout simplification | Cancel and merge reshape/permute chains around attention head split and merge. |
| Operator fusion | Recognize GELU from its primitive expansion; matmul + bias to `gemm`; scale + mask + softmax in attention; chains of elementwise ops into one loop; residual add into the producer where legal. |
| Memory planning | Liveness analysis, in-place reuse, then offset assignment into a single arena. Report peak arena bytes against the no-reuse total. |

### 4.3 Loop IR

- Statements: `For(var, extent, kind)`, `Store`, `Alloc`, `Block`, `Call`. Expressions: `Load`, arithmetic, intrinsic calls. `kind` is serial, parallel, or vectorized.
- Each graph op lowers to a naive loop nest. Optimizations are explicit, composable rewrites on that nest: `split`, `tile`, `reorder`, `fuse`, `unroll`, `vectorize`, `parallelize`.
- A **schedule** is a list of those rewrites plus parameters. The autotuner searches over schedules; it never edits IR directly.
- A loop IR interpreter (slow, tree-walking) checks that a scheduled nest still computes what the naive nest did.

GEMM dominates runtime (768x768, 768x3072, and the 768x50257 output projection), so it gets the most structure: cache-level tiling, operand packing, and a register-tiled micro-kernel with an unrolled FMA body. Outer loops come from the loop IR; the micro-kernel is generated from a parameterized template (`MR`, `NR`, unroll factor).

### 4.4 Codegen and runtime

- Loop IR to C++ with NEON intrinsics. Vectorized loops emit a vector body plus a scalar tail.
- The runtime linked into generated code is a few hundred lines: arena allocation, weight mmap, thread pool, timing. It is handwritten and does not grow kernels.
- Generated code is kept on disk and readable. Being able to read the output is a debugging feature.

### 4.5 Autotuner

- Tunables: tile sizes per cache level, `MR` x `NR`, unroll factor, pack or no pack, parallel loop choice and grain, thread count.
- Tunes each distinct kernel instance (op + shapes) in isolation on a micro-benchmark, then validates the composed model.
- Search: random search first, then an evolutionary refinement if random leaves measurable gains. A learned cost model is out of scope.
- Candidate kernels are compiled to small shared libraries and timed in-process through `dlopen`.
- Results are cached in JSON keyed by kernel signature and CPU, so a rebuild does not retune.

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
- `-ffast-math` is off by default. If it is ever enabled it is a named, benchmarked option.

## 7. Benchmarking rules

- Same machine, on power, nothing else running. Same inputs, fp32, batch 1.
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
  tune/         search, measurement, cache
  gpu/          Triton emitter (M7)
  main.cpp      the `mlc` driver
runtime/        handwritten C++ linked into generated code: arena, thread pool, weight loader
bench/          Python: harness, baselines, result tables
tests/          C++ unit and differential tests; end-to-end checks against golden logits
third_party/    vendored single-header libraries (JSON, test framework)
docs/           log.md, notes
build/          generated C++ and binaries (gitignored)
```

Dependencies in the compiler stay minimal: the standard library, one JSON header, one test header. No LLVM, no MLIR.

## 9. Risks and open questions

| Item | Plan |
|---|---|
| GEMM cannot match Accelerate on M-series | Confirmed in M0 (see section 7). Measure our kernel standalone in M4 against both Accelerate and ONNX Runtime. Report the gap; do not hide it inside a whole-model number. |
| `torch.compile` CPU backend on macOS arm64 may be fragile | Resolved in M0: it runs and matches golden logits. |
| Export produces ops or patterns we did not expect | M1 starts by dumping and reading the exported graph before writing any lowering. |
| C++ slows iteration on pattern-matching passes | Build a small pattern-matching helper and the IR printer early, in M1. Keep debug builds fast. |
| Fusion patterns are brittle against export changes | Pin `torch` and `transformers` versions. Pattern-match on our canonicalized IR, not raw ATen. |
| Autotuning noise on a laptop | Min-of-N timing, re-measure the top candidates, reject results with high variance. |
| Where M7 runs | Open. Needs a Linux + CUDA machine. Decide before M6 ends. |
| Is the KV-cache decode step in scope | Open. It needs a second export and state passed between calls. Decide after M4. |

## 10. Working rules

- Small commits, each leaving tests green.
- No optimization lands without a before and after number.
- No number goes in the README without the command that reproduces it.
- Version-specific facts about PyTorch, ONNX Runtime, Triton, or Apple hardware are checked against current documentation or measured, not recalled.
