# 5. The reference interpreter and the idea of an oracle

## 5.1 Why build an interpreter inside a compiler

Before generating any code, the project built a program that *executes the graph IR directly*: walk the nodes in order, and for each one, run a plain C++ function that computes it. This is the **reference interpreter** (`src/graph/interpreter.cpp`, `mlc run`).

It is deliberately simple and slow. Its purpose is to be obviously correct, so that it can serve as a trustworthy answer key:

- It proves the importer understood the model. If the interpreter's logits match PyTorch's, then the graph IR means what PyTorch meant.
- It defines what each IR operator means, in runnable form.
- Every later stage is tested against it. When an optimization pass rewrites the graph, run the interpreter on the graph before and after: the outputs must agree.
- Constant folding (chapter 6) uses it as a calculator: to evaluate a piece of the graph at compile time, just interpret that piece.

## 5.2 The oracle chain

An **oracle** in testing is whatever tells you the right answer. This project has a chain of them, each checked against the one before:

```
PyTorch (golden logits)
   └─ graph interpreter, on the imported graph          [ctest: golden_T*]
        └─ graph interpreter, on the optimized graph    [ctest: optimized_T*]
        └─ loop interpreter, on small programs          [unit tests]
             └─ compiled code                           [unit tests, ctest: compiled_T*]
```

Each link compares two things that should compute the same function in different ways. A bug has to be present in both sides in exactly the same form to slip through.

**The standard of correctness** (STEERING.md section 6): on the fixed golden input, the largest absolute difference in any logit is at most `1e-3`, and at every position the largest logit (the predicted next token) is the same one. Exact equality is impossible, because floating-point addition gives slightly different results depending on the order of operations (section 9.7), and every optimization changes that order. Observed differences are 1e-4 to 5e-4. For scale, logits are numbers roughly between -300 and 0.

## 5.3 Tensors at run time

```cpp
struct Tensor {
  Type type;
  std::shared_ptr<std::byte[]> owner;  // keeps the memory alive; null if borrowed
  const std::byte* ptr;                // where the data is
};
```

A tensor either **owns** its memory (the interpreter allocated it) or **borrows** it (it points into the weights file, into an inline constant, or at the caller's input). `shared_ptr` is a reference-counted pointer: the memory is freed when the last tensor pointing at it goes away. This lets `Reshape` cost nothing: the result is a new `Tensor` with a different `type` and the same `owner` and `ptr`.

## 5.4 One loop to rule the elementwise and layout operators

Most operators are some version of "for every element of the output, read one element from each input". What differs is *which* input element. The interpreter has one generic loop for all of them, `strided_loop`, built on strides (section 2.1).

**Broadcasting as stride 0.** To read input `in` as if it had the larger output shape, give it a stride of 0 along every axis where it has size 1 (or no axis). Stepping along that output axis then does not move through the input at all, so the same element is read again. That is all broadcasting is.

**The odometer.** `strided_loop` visits every index of the output shape in row-major order, like a car's odometer: bump the last digit; when it overflows, reset it and carry into the next digit. Alongside the index it maintains, for each input, the current linear offset under that input's strides. Each step calls `body(output_position, input_offsets)`.

With that:

| Operator | Input strides |
|---|---|
| `Add`, `Mul`, `Eq`, `Where`, ... | each input's broadcast strides |
| `Broadcast` | the input's broadcast strides |
| `Permute` | the input's strides, reordered by the permutation |
| `Slice` | the input's own strides, starting from an offset |

`Permute` is the instructive one. If output axis `d` is input axis `dims[d]`, then one step along output axis `d` should move through the input by the stride of input axis `dims[d]`. So the permuted strides are just the input strides shuffled.

A helper, `by_dtype`, instantiates the same generic code for `float`, `int64_t`, and `uint8_t` (bools are stored as one byte), so each operator is written once.

## 5.5 The remaining operators

- **`Gather`** (`x[i0, i1, ...]`): for each position of the (broadcast) index tensors, compute an offset into `x` from the loaded indices and copy the remaining axes of `x` as one block. An index out of range is an error, not a wild read.
- **`Concat`**: copy runs of each input in turn with `memcpy`.
- **`CumSum`**: each element is its input plus the output one step back along the axis.
- **`ReduceAny`**: OR along one axis.
- **`MatMul` / `Gemm`**: see below.
- **`LayerNorm`**: the formula of section 2.3, per row. The sums are accumulated in `double` so the reference's own rounding error is far below the tolerance it is used to check.
- **`Softmax`**: computed as `exp(x_j - max) / sum(exp(x_i - max))`. Subtracting the row maximum first changes nothing mathematically (the factor cancels) but keeps every exponent at or below zero, so `exp` cannot overflow. A row that is entirely `-inf` gives 0/0 = NaN, the same as PyTorch.
- **`Gelu`**, causal `Softmax`, and the `trans_b` products were added in M3 and M4 when passes began producing them.

### The first loop interchange of the project

The definition of a matrix product, `C[i][j] = sum over p of A[i][p] * B[p][j]`, suggests three loops in the order `i, j, p`. The interpreter uses `i, p, j`:

```cpp
for (i ...) for (p ...) { float a = A[i*k + p]; for (j ...) C[i*n + j] += a * B[p*n + j]; }
```

With `j` innermost, the inner loop walks along a row of `B` and a row of `C`, both consecutive in memory, and `A[i][p]` is a constant. In the order `i, j, p` the inner loop would walk down a *column* of `B`, jumping `n` elements per step and touching a new cache line each time. Same arithmetic, several times faster. Chapter 10 is this idea taken to its conclusion.

## 5.6 Freeing values: liveness

At T=1024 the model's intermediate tensors add up to 6.7 GB. They are not all needed at once. A value is **live** from when it is computed until the last node that reads it; after that it is **dead** and its memory can be reused.

The interpreter does the simplest version: before running, count how many times each value will be read (`uses_left`). After each node runs, decrement the counts of its inputs; when a count reaches zero, drop the tensor. With that, peak memory at T=1024 is 363 MB instead of 6.7 GB.

The interpreter reports both numbers (`mlc run --profile`). The peak figure became the target that the compiler's memory planner (chapter 7) had to beat.

## 5.7 What the interpreter showed (M1)

Running it on the real model gave the first correct end-to-end result and the first profile:

| T | max logit difference vs PyTorch | interpreter time | PyTorch, 1 thread |
|---|---:|---:|---:|
| 16 | 1.45e-04 | 0.56 s | 0.038 s |
| 1024 | 1.45e-04 | 27.4 s | 1.12 s |

(Battery power; see section 12.4 on power states.)

The profile was a to-do list for the optimizer. From `docs/log.md`, M1:

- The transposed embedding table (a `permute` of `[50257, 768]`, 154 MB) was recomputed every call and was 37% of the T=16 run.
- About forty nodes rebuilt the causal mask from scratch every call.
- Every attention layer carried a guard for a case that cannot happen with a causal mask.
- All 49 `expand` nodes did nothing.
- GELU was eight separate full passes over a `[1, T, 3072]` tensor.
- The two matrix-product operators were 79% of the T=1024 run.

## 5.8 The second interpreter

There is also an interpreter for the loop IR (`src/loop/interpreter.cpp`, chapter 7). It plays the same role one level down: it defines what a loop program means, runs every loop in order whatever optimizations have been marked on it, and checks every single array index against its bounds. It is roughly a hundred times slower than compiled code, so it is used on small test graphs, never the full model.

## 5.9 Where in the code

| File | Contents |
|---|---|
| `src/graph/interpreter.h`, `interpreter.cpp` | `Tensor`, `RunStats`, `run_graph`, one function per operator |
| `src/main.cpp` | `mlc run`, including `compare_logits` (the correctness criterion) |
| `tests/test_interpreter.cpp` | each operator against hand-computed answers |
| `CMakeLists.txt` | the `golden_T*` and `optimized_T*` end-to-end tests |

## 5.10 Further reading

- McKeeman, "Differential Testing for Software" (1998). The idea of testing by comparing independent implementations, which is this project's whole correctness strategy.
- Yang et al., "Finding and Understanding Bugs in C Compilers" (PLDI 2011), the Csmith paper. What differential testing finds in real compilers.
- The NumPy documentation section "Internal memory layout of an ndarray" (strides). The same stride mechanism the interpreter uses.
