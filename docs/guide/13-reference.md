# 13. Design decisions, open problems, further reading, glossary

## 13.1 Every significant design decision

Each row: what was chosen, what the alternative was, and why. The "chapter" column says where it is explained.

### Scope

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| Compile one model, GPT-2 small | A general model compiler | A fixed, real target keeps every decision concrete and measurable | 1 |
| Ahead-of-time compilation to a shared library | A JIT inside Python | Simpler to inspect; the output is a file you can read | 1 |
| Static shapes, one build per `T` | Dynamic shapes | Enables constant folding of the mask, memory planning, exact loop bounds | 1 |
| Batch 1, fp32, inference only | Batching, quantization, training | Out of scope | 1 |
| Compiler in C++20, edges in Python | All Python | The learning goal was a compiler in a systems language; `torch.export` exists only in Python | 3 |
| No LLVM, no BLAS, no vendored libraries | Use them | Writing the kernels and the IRs was the point | 1 |
| Output is C++ text compiled by `clang++` | Emit machine code | Reuses instruction selection and register allocation; output is readable | 8 |

### Frontend

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| `torch.export` + core ATen decomposition | ONNX as the interchange format | One operator vocabulary instead of two; 37 operators to support | 3 |
| The exporter interprets nothing | Simplify the graph in Python | All semantics in one tested place | 3 |
| Weights in a separate file, addressed by offset | Embed in the graph or the code | 498 MB cannot go in source; the file can be memory-mapped | 3, 8 |
| Unknown operators and unread arguments are errors | Skip or approximate | A silently wrong model is the worst outcome | 4 |
| Check every inferred type against PyTorch's | Trust our shape inference | Free, strong evidence on every import | 4 |
| Own JSON reader (360 lines) | Vendor a library | Small job; keeps ints and floats apart; fast builds | 4 |

### Graph IR

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| One value per node, named by index (SSA) | Named mutable variables | Unambiguous definitions; rewriting is always safe | 4 |
| Pure nodes | In-place operations | Lets passes delete, merge, and reorder freely | 4 |
| 27 operators, composites kept whole | Only primitives | A loop generator does better with "layer norm" than with its pieces | 4 |
| One `Attrs` struct for all operators | A struct per operator | Equality (for CSE) and printing are trivial | 4 |
| One `infer_type` used to build and to verify | Separate logic | One definition cannot disagree with itself | 4 |
| Passes build a new graph through one engine | Edit in place | Operands are already rewritten when a rule sees them; no iterator invalidation | 6 |
| DCE and the verifier after every pass | Once at the end | Each pass is charged for what it enables; a bad pass is caught immediately | 6 |
| Recognize the causal mask by its value | Match the nodes that build it | Works whatever computes it, including when folding is capped | 6 |
| Never fold weights; cap folded results at 1 MB | Fold everything constant | Weight bytes stay out of the compiler; avoid bloating the code | 6 |

### Loop IR and memory

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| A second IR for loops | Generate C++ from the graph | Loop order, tiling, and vectorization need loops to talk about | 1, 7 |
| Three statements, no scalar variables | A fuller language | Every dependence is a visible load or store, which the legality analysis needs | 7 |
| Multi-dimensional indices until codegen | Flat offsets | The analysis reasons per axis | 7, 10 |
| `Reshape` is a second view, no code | Copy | It is the same bytes | 7 |
| A broadcast is a missing loop variable | Materialize | Free at run time | 7 |
| Memory planned at compile time into one arena | `malloc` per tensor at run time | No allocation while running; 41x less memory at T=1024 | 7 |
| Best-fit free list with coalescing | An optimal packing algorithm | Simple and already within a few MB of the largest single buffer | 7 |
| In-place reuse for elementwise kernels | Always fresh memory | Removes most allocations outright | 7 |
| Staging: load-time values in persistent memory | Recompute per call, or fold in the compiler | The compiler lacks weight bytes; the runtime has them | 7 |
| A loop interpreter as the semantic reference | Trust codegen | Defines what a scheduled program must still compute | 5, 7 |

### Performance

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| Schedule separate from program; rewrites prove legality or refuse | Hand-write fast loops in the lowering | Any schedule is safe to try; enables the autotuner | 10 |
| Dependence test: "each iteration owns a slice" | Full polyhedral analysis | Sound, small, and sufficient for these loops | 10 |
| Dot-product matrix kernel, operand transposed | BLIS: packing plus outer-product micro-kernel | No packing; best at small `T`; the embedding table is already in that layout. Costs 5 to 9% on large products | 10 |
| Micro-kernel derived by unroll-and-jam + vectorize | A hand-written template | More general; the rewrites are reusable and tested | 10 |
| Emit NEON intrinsics ourselves | Rely on clang's auto-vectorizer | clang cannot reorder float sums and did not vectorize the loops | 9, 10 |
| The compiler reorders float sums; clang gets no `-ffast-math` | `-ffast-math` everywhere | Reordering is confined to marked loops and visible in the IR | 9 |
| Handwritten vector `exp` and `tanh` | Call the C library per lane | A per-lane call defeats vectorization | 10 |
| `__restrict` parameters from the memory plan | Leave aliasing to clang | The planner knows; clang cannot | 8, 9 |
| Own thread pool, static partition, spin then sleep | OpenMP | Apple's clang ships no OpenMP; small kernels need cheap wake-up | 10 |
| Threading decided per kernel at compile time | A run-time check | Shapes are static | 10 |
| An instruction-set table | Intrinsic names inside the emitter | Another instruction set is one more table | 10 |
| Autotune only the matrix-product schedule | Tune everything | Nothing else showed a measurable effect | 11 |
| Final round against the default with a 2% margin | Take the best sample | Avoids fitting noise | 11 |
| Cache keyed by CPU, threads, signature | One global answer | Measurements do not transfer | 11 |

### Method

| Decision | Alternative | Why | Ch. |
|---|---|---|---|
| A slow correct pipeline before any optimization | Optimize as you go | Every later change has something to be measured and checked against | 5 |
| Tolerance 1e-3 plus identical argmax | Bit equality | Float reordering makes equality impossible | 5, 9 |
| ONNX Runtime as the peer, Accelerate as the ceiling | "Beat PyTorch" | Accelerate uses hardware NEON code cannot reach | 9, 12 |
| Every optimization has an off switch | Always on | Ablations | 6, 12 |
| Losses reported as plainly as wins | | The point was to learn what is true | 12 |

## 13.2 What is unfinished or unexplained

Collected from `docs/log.md` and `docs/writeup.md`.

**Open performance problems**

1. **Small `T` on four threads.** 57% behind ONNX Runtime at T=16. The dense layers are limited by reading cold weights from memory. What ONNX Runtime does better was not found.
2. **Large products are 9% behind ONNX Runtime's kernel.** A packed outer-product kernel would likely recover it for large shapes.
3. **The autotuner measures warm operands**, which is not the situation in the model at small `T`.
4. **The small-inner-dimension product** (`q @ k^T`, inner size 64) runs at about 85% of the others' rate because per-tile overhead is a larger share.

**Open costs**

5. **340 MB of persistent memory** for transposed weights. Storing the weights transposed in the weights file would remove it.
6. **Permutes and slices are real copies.** They could be views (strided buffers), which the loop IR does not have.

**Never explained**

7. **`noalias` on its own changes nothing**, though aliasing was identified in M2 as what stopped clang's vectorizer.

**Not built**

8. Rewrites `fuse`, standalone `unroll`, general `reorder`: nothing needed them.
9. A second instruction set (AVX2).
10. **M7: the Triton GPU backend.** Needs a Linux machine with an NVIDIA GPU.
11. **Stretch: KV-cached decoding.** Generating one token at a time while reusing earlier attention keys and values. Needs a second export and state carried between calls.

## 13.3 Further reading, by topic

Each chapter ends with its own list. This is the consolidated version, ordered roughly by how much each would add.

**If you read only five things**

1. Karpathy, "Let's build GPT: from scratch, in code, spelled out" (video) and `nanoGPT`. To know the source program.
2. Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008). To know why fast matrix products look the way they do.
3. Ragan-Kelley et al., "Halide" (PLDI 2013). The algorithm/schedule separation.
4. Chen et al., "TVM" (OSDI 2018). The closest full system to this project.
5. Drepper, "What Every Programmer Should Know About Memory" (2007).

**Compilers in general**

- Cooper and Torczon, *Engineering a Compiler*.
- Muchnick, *Advanced Compiler Design and Implementation*.
- Allen and Kennedy, *Optimizing Compilers for Modern Architectures* (dependence analysis and loop transformations).
- Nystrom, *Crafting Interpreters* (free online; parsers and tree-walking interpreters).
- Cytron et al., "Efficiently Computing Static Single Assignment Form..." (1991).

**Machine-learning compilers**

- Lattner et al., "MLIR" (2021).
- Li et al., "The Deep Learning Compiler: A Comprehensive Survey" (2020). An overview of the field.
- Jia et al., "TASO" (2019), on graph rewriting.
- Ikarashi et al., "Exocompilation..." (2022), on user-scheduled kernels.
- Pisarchyk and Lee, "Efficient Memory Management for Deep Neural Net Inference" (2020).
- PyTorch documentation: `torch.export`, Core ATen IR.
- The ONNX Runtime documentation, and the source of its `MLAS` library for how its kernels are built.

**The model**

- Vaswani et al., "Attention Is All You Need" (2017).
- Radford et al., "Language Models are Unsupervised Multitask Learners" (2019).
- Alammar, "The Illustrated GPT-2".
- Hendrycks and Gimpel, "Gaussian Error Linear Units" (2016); Ba et al., "Layer Normalization" (2016).

**Hardware and performance**

- Hennessy and Patterson, *Computer Architecture: A Quantitative Approach*.
- Agner Fog's optimization manuals.
- Arm's NEON intrinsics reference and programmer's guide.
- Dougall Johnson's notes on Apple's Firestorm core.
- Van Zee and van de Geijn, "BLIS" (2015).
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991).

**Autotuning**

- Whaley et al., ATLAS (2001); Frigo and Johnson, FFTW3 (2005).
- Chen et al., "Learning to Optimize Tensor Programs" (2018); Zheng et al., "Ansor" (2020).

**Measurement and testing**

- McKeeman, "Differential Testing for Software" (1998).
- Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (2009).
- Hoefler and Belli, "Scientific Benchmarking of Parallel Computing Systems" (2015).

## 13.4 Glossary

**Ablation.** Measuring what one optimization contributes by building with and without it.

**Accumulator.** A variable (in generated code, a register) holding a running sum or maximum.

**Affine expression.** A sum of variables times constants, plus a constant. All array indices in the loop IR are affine.

**Aliasing.** Two names referring to the same memory.

**AOT (ahead-of-time) compilation.** Compiling before the program runs, to a file.

**Arena.** One block of scratch memory holding every intermediate tensor at a compiler-chosen offset.

**ATen.** PyTorch's C++ tensor library; its operators are what `torch.export` records.

**Attribute.** A non-tensor parameter of a graph node (an axis, a shape).

**Best fit.** Allocation policy: use the smallest free block that is large enough.

**Binding time.** When a value becomes known: compile time, load time, or run time.

**BLAS.** The standard interface for linear algebra routines; Accelerate, OpenBLAS, and BLIS implement it.

**Blocking / tiling.** Splitting a loop so that work is done on cache-sized pieces.

**Broadcasting.** Treating a size-1 axis as repeated to match another tensor.

**Cache.** Small fast memory holding recently used data. L1 is the smallest and fastest.

**Cache line.** The 64-byte unit caches work in.

**Causal mask.** The rule that position `i` may attend only to positions `0..i`.

**Coalescing.** Merging adjacent free memory blocks.

**CSE (common subexpression elimination).** Merging nodes that compute the same value.

**DCE (dead code elimination).** Removing nodes no output depends on.

**Dependence.** One loop iteration reading or writing what another writes.

**Dependency chain.** Instructions each needing the previous one's result.

**Differential test.** A test that compares two implementations of the same function.

**Dot product.** `sum of a[p] * b[p]`.

**dtype.** The element type of a tensor: `f32`, `i64`, or `bool`.

**Elementwise.** An operation applied independently to each element.

**Extent.** A loop's trip count.

**Flop.** One floating-point operation. **GFLOP/s**: billions per second.

**FMA.** Fused multiply-add: `a + b*c` as one instruction.

**Folding (constant folding).** Evaluating at compile time what does not depend on input.

**Fusion.** Replacing several operations by one that does the same work in one pass.

**Gather.** Looking up rows of a table by index.

**GEMM.** General matrix multiply. In this project `Gemm` is `a @ b + bias`.

**Golden output.** PyTorch's saved result on a fixed input; the reference answer.

**Graph IR.** The compiler's representation of the model as operations on whole tensors.

**Hoisting.** Moving a computation to where it runs less often.

**Init kernel.** A kernel run once at load time.

**Interchange.** Swapping two nested loops.

**Interleave.** The number of independent accumulator sets in a vectorized reduction.

**Intrinsic.** A C++ function that compiles to one specific instruction.

**IR (intermediate representation).** A compiler's internal form of a program.

**JIT (just-in-time) compilation.** Compiling during execution.

**Kernel.** The loop nest for one graph node; one function in generated code.

**Lane.** One of the four float slots in a NEON register.

**Latency.** Cycles from an instruction's inputs being ready to its result.

**Liveness.** A value is live from its definition to its last use.

**Logits.** The model's output scores, one per vocabulary token per position.

**Loop-carried dependence.** An iteration depending on an earlier iteration of the same loop.

**Lowering.** Translating to a more concrete representation.

**Memory planning.** Deciding at compile time where each tensor lives.

**Micro-kernel.** The innermost loop of a matrix product, operating on a register-sized tile.

**mmap.** Mapping a file into memory so its bytes appear at addresses, loaded on first touch.

**NEON.** The SIMD instruction set of arm64 CPUs.

**Oracle.** Whatever supplies the right answer in a test.

**Outlining.** Moving a loop body into its own function (for threads).

**Packing.** Copying a matrix into a contiguous buffer arranged for a micro-kernel. Not used here.

**Pass.** A function from a program to an equivalent, better program.

**Persistent memory.** Memory filled by init kernels at load and read on every call.

**Prefetcher.** CPU logic that fetches memory ahead of a detected access pattern.

**Pure.** Having no side effects; the result depends only on the inputs.

**Rank.** The number of axes of a tensor.

**Reduction.** Combining many values into one (a sum, a maximum).

**Register tiling.** Unroll-and-jam of two loops so a tile of results is computed at once in registers.

**Residual connection.** `x + f(x)`.

**Row-major.** Memory layout in which the last axis varies fastest.

**Schedule.** The choice of loop transformations and their parameters.

**Signature.** A kernel's operator and operand shapes; kernels with the same signature share a tuned schedule.

**SIMD.** Single instruction, multiple data.

**Softmax.** `exp(x_j) / sum of exp(x_i)`: turns scores into probabilities.

**Sound (analysis).** Never claims a property that does not hold.

**SSA (static single assignment).** Every value is defined exactly once.

**Staging.** Sorting computations by binding time.

**Stride.** How far in memory one step along an axis moves.

**Tensor.** An array of numbers with a shape and a dtype.

**Throughput.** How many of an instruction can start per cycle.

**Tracing.** Recording the operations a program performs on an example input.

**Unroll-and-jam.** Unrolling an outer loop and fusing the copies into the inner loop.

**Value numbering.** Giving each distinct computation a number so duplicates can be found; how CSE is implemented.

**Vectorize.** Rewrite a loop to use SIMD instructions.

**Weight tying.** Using the same parameter tensor for the input embedding and the output projection.
