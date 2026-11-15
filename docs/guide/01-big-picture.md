# 1. The big picture

## 1.1 What the project is

`mlc` is a compiler. Its input is a neural network, GPT-2 small, as PyTorch defines it. Its output is a native shared library (a `.so` file of machine code) with one job: given token ids, compute the same logits PyTorch would, on a CPU, as fast as it can.

The project then measures that library against three existing ways of running the same model: PyTorch itself, `torch.compile`, and ONNX Runtime.

It exists to learn two things by doing them: how compilers for machine learning work, and how to make numerical code fast. That goal explains several choices that would be wrong in a product. Everything is written from scratch: no LLVM, no BLAS library, no JSON library. Where there was an easy route and an instructive one, the instructive one was taken.

## 1.2 What "compiling a neural network" means

A trained neural network is two things:

1. **A fixed computation.** For GPT-2: look up embeddings, then twelve identical blocks of layer norm, attention, and a small feed-forward network, then one last matrix product. Nothing about which operations run depends on the input. There are no `if` statements on data.
2. **Numbers.** 124 million floating-point parameters (weights) that the computation uses.

PyTorch runs this by *interpreting* it. Python code calls `torch.matmul`, which is a C++ function that inspects its arguments, picks an implementation, allocates memory for the result, computes, and returns to Python. Then the next line runs. Each operation is fast inside, but nothing looks across operations, and every call pays for the inspection and allocation.

A compiler does the looking-across ahead of time. Because the computation is fixed, it can be studied once, before any input arrives:

- Parts that do not depend on the input can be computed once and stored.
- Sequences of operations can be recognized and replaced by something cheaper.
- All memory can be laid out in advance, so nothing is allocated while running.
- The loops for each operation can be generated for the exact sizes involved, with no checks left in them.

The result is a program that does nothing but arithmetic.

This is called **ahead-of-time (AOT) compilation**: the compiler runs first and produces a file; later, a separate program loads that file and runs it. The opposite is **just-in-time (JIT) compilation**, where the compiler runs inside the program the first time a piece of code is needed. `torch.compile` is a JIT.

## 1.3 Why there are several stages

A compiler is almost always built as a chain of small translations rather than one big one. Each stage has its own way of writing down the program, called an **intermediate representation (IR)**. An IR is just a data structure, chosen so that a particular kind of reasoning is easy.

This project has two IRs because two different kinds of question need answering:

- "Is this multiplication by 1 pointless? Are these nine operations really a GELU?" These are questions about *what is computed*. They are easy when the program is a graph of whole-tensor operations and hard when it is a million lines of loops.
- "Should this loop be split in blocks of 20? Can these iterations run on different threads?" These are questions about *how it is computed*. They cannot even be asked of a graph, which has no loops.

So the program is first held as a **graph IR** (operations on whole tensors) and then **lowered** to a **loop IR** (loops reading and writing memory). "Lowering" is the standard word for translating from a more abstract form to a more concrete one.

## 1.4 The pipeline

```
 PyTorch model (Python)
        │   torch.export + decomposition          [Python, export/]
        ▼
 graph.json + weights.bin                          files on disk
        │   importer                               [src/frontend/]
        ▼
 Graph IR  ── passes: simplify, fold, cse,         [src/graph/]
        │            gelu, attention, layout
        │   lowering + memory planning + staging   [src/loop/lower.cpp]
        ▼
 Loop IR   ── schedule: tile, vectorize,           [src/loop/schedule.cpp]
        │               parallelize                [src/tune/ picks parameters]
        │   code generation                        [src/codegen/]
        ▼
 model.cpp (C++ with NEON intrinsics)
        │   clang++ -O3, linked with the runtime   [runtime/]
        ▼
 model.so  ──  mlc_open(weights) ; mlc_run(inputs, outputs)
```

One sentence per stage. Each has its own chapter.

- **Export (chapter 3).** A Python script asks PyTorch to trace the model into a list of primitive operations and writes that list as JSON, with the weights in a separate binary file. This is the only place PyTorch's own machinery is used to understand the model.
- **Import (chapter 4).** The C++ compiler reads the JSON and builds its graph IR: 1062 nodes, each producing one tensor of known type and shape.
- **Graph passes (chapter 6).** Six rewrites shrink the graph to 615 nodes and then rearrange the matrix products.
- **Lowering (chapter 7).** Every graph node becomes a nest of loops (a **kernel**). At the same time the compiler decides where in memory every tensor will live, and which values can be computed once at load time.
- **Scheduling (chapters 9, 10, 11).** The loops are rewritten to run fast: reordered, split into blocks, marked for SIMD and for threads.
- **Code generation (chapter 8).** The loops are printed as C++ text.
- **Build and run (chapter 8).** `clang++` turns the C++ into machine code. A small handwritten runtime, linked into the same library, maps the weights file into memory and calls the kernels in order.

## 1.5 Two ideas that run through everything

**Static shapes.** The compiler is told the sequence length `T` up front and produces a library for exactly that `T`. Every tensor's shape is therefore a compile-time constant. This is what makes most of the optimizations possible: you cannot plan memory, or fold the attention mask into a constant, or generate a loop "from 0 to 768" if you do not know the sizes. The cost is one compile per `T`.

**Correctness by comparison (chapter 5).** How do you know a compiler is right? Here, by a chain of comparisons. PyTorch's output on a fixed input is saved once (the **golden** output). A deliberately slow and simple interpreter of the graph IR is checked against it. From then on, everything else is checked against whatever came before it in the chain. The standard is: no logit differs from PyTorch's by more than 0.001, and the most likely next token is the same at every position.

## 1.6 What it achieved

Measured on an Apple M2 laptop, on AC power. Times are for one forward pass.

| | T=16 | T=1024 |
|---|---:|---:|
| First correct compiled build (M2), 1 thread | 650 ms | 38 100 ms |
| Final build, 1 thread | 46 ms | 3 200 ms |
| Final build, 4 threads | 22 ms | 1 030 ms |
| ONNX Runtime, 1 thread | 43 ms | 3 220 ms |
| ONNX Runtime, 4 threads | 14 ms | 1 030 ms |
| PyTorch, 1 thread | 28 ms | 706 ms |

- The compiler made its own output 12 to 14 times faster on one thread between the first working version and the last.
- **ONNX Runtime** is the fair comparison: a mature inference engine whose kernels run on the same CPU instructions ours do. The final build matches it at T=1024 and is behind at smaller sizes, by a few percent on one thread and up to 57% on four.
- **PyTorch** is much faster on this machine for a reason outside the project's reach: its matrix products go through Apple's Accelerate library, which uses hardware that ordinary CPU instructions cannot.

Chapter 12 goes through these numbers properly, including the ones that are losses and why.

## 1.7 Where in the code

| Path | What |
|---|---|
| `export/` | Python: model definition, exporter, serializer |
| `src/frontend/` | graph.json to graph IR |
| `src/graph/` | graph IR, printer, verifier, interpreter, passes |
| `src/loop/` | loop IR, lowering, loop interpreter, schedule |
| `src/codegen/` | C++ emitter, instruction-set table, build driver |
| `src/tune/` | autotuner |
| `src/support/` | JSON reader, file mapping, `.npy` reader |
| `src/main.cpp` | the `mlc` command-line tool |
| `runtime/` | code linked into every compiled model |
| `tests/` | unit and differential tests |
| `bench/` | Python benchmark scripts and result tables |

About 6 700 lines of C++ in the compiler and runtime, 2 000 of tests, and 1 100 of Python.

## 1.8 Further reading

- Cooper and Torczon, *Engineering a Compiler* (3rd ed.), chapters 1 and 4. The standard introduction to why compilers are built as pipelines of IRs.
- Chen et al., "TVM: An Automated End-to-End Optimizing Compiler for Deep Learning" (OSDI 2018). The best-known system with the same overall shape as this project: graph level, loop level, schedules, autotuning.
- Ragan-Kelley et al., "Halide: A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines" (PLDI 2013). The paper that introduced separating *what* is computed from the *schedule* for computing it, which chapters 10 and 11 rely on.
