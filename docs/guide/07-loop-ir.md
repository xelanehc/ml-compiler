# 7. The loop IR, lowering, and memory planning

The graph IR says `%29 = matmul(%27, %28)`. To run that, something has to say: these are the loops, this is the order they nest in, and these are the memory addresses they read and write. That is the **loop IR** (`src/loop/ir.h`). Translating the graph into it is **lowering** (`src/loop/lower.cpp`).

See it on the real model:

```sh
build/release/mlc loops artifacts/graphs/gpt2_T16.json --disable vectorize,regtile,cachetile,parallel | less
```

## 7.1 The whole language

Three kinds of statement:

| Statement | Meaning |
|---|---|
| `For(var, extent, body)` | run `body` with `var = 0, 1, ..., extent-1` |
| `Store(buffer, indices, value)` | `buffer[indices] = value` |
| `Block(stmts)` | run statements in order |

Nine kinds of expression:

| Expression | Meaning |
|---|---|
| `IntImm`, `FloatImm` | a constant |
| `Var` | a loop variable |
| `Load(buffer, indices)` | read `buffer[indices]` |
| `Binary(op, a, b)` | `+ - * / max == != <= & \|` |
| `Not(a)` | logical not |
| `Call(fn, args)` | `exp`, `tanh`, `pow`, `sqrt` |
| `Select(c, a, b)` | `c ? a : b` |
| `Cast(type, a)` | convert between float, int, bool |

That is all. A matrix product in this language, printed by `mlc loops`:

```
kernel matmul  # %2
  for i0 in 0..2:
    for i1 in 0..3:
      v2[i0, i1] = 0
      for i2 in 0..4:
        v2[i0, i1] = (v2[i0, i1] + (v0[i0, i2] * v1[i1, i2]))
```

### Design choices in the language

**No scalar variables.** There is no way to write `sum = 0; ... sum += x`. A running sum has to live in memory: load it, add, store it back (the last line above). This looks wasteful, and in generated code it would be if left that way. The reason is analysis: with no hidden variables, *every* way one iteration can affect another is a visible load or store of a named buffer. The legality checks in chapter 10 depend on that. Registers come back at the very end, in generated code.

**Multi-dimensional indices.** Accesses are written `v2[i0, i1]`, one index per axis, and only flattened to a single offset when C++ is printed. Keeping the indices separate is what lets the analysis say "this loop variable appears only in axis 1".

**Loop extents are constants, or simple expressions of outer loop variables.** Shapes are static, so nearly every loop has a constant trip count. The one exception is the causal softmax, whose inner loops run to `i + 1`.

**Loops carry a "kind".** Serial, vectorized, or parallel. A fresh lowering is all serial; chapter 10 changes kinds. The kind never changes what a loop computes.

**Expressions are immutable and shared.** An `Expr` is held by `shared_ptr<const Expr>`, so a subexpression can appear in many places without copying, and no transformation can corrupt another's tree.

**Constructors check and tidy.** `binary(...)`, `load(...)` and friends throw if operand types do not match (adding an int to a float, a non-integer index). They also fold integer constants (`x + 0`, `x * 1`, `2 * 3`) so that index arithmetic stays readable.

## 7.2 Buffers and storage

A **buffer** is a named, typed, shaped view of some memory:

```cpp
struct Buffer { std::string name; DType dtype; Shape shape; Storage storage; int64_t offset; int32_t index; };
```

`storage` says where the bytes live at run time:

| Storage | Meaning | Writable |
|---|---|---|
| `Arena` | scratch memory owned by the runtime, at byte `offset` | yes |
| `Weights` | the memory-mapped weights file, at byte `offset` | no |
| `Persistent` | memory filled once when the model loads, at byte `offset` | by init kernels only |
| `Input` | the caller's input number `index` | no |
| `Output` | the caller's output number `index` | yes |
| `Inline` | a small constant table carried in the generated code | no |

The **arena** is one big block of scratch memory. Every intermediate tensor is a slice of it at a fixed offset decided by the compiler. At run time there is no `malloc`: the runtime allocates the arena once and kernels use offsets into it.

Two buffers may view the same bytes. That is how `Reshape` costs nothing: the reshaped value gets a second `Buffer` with a different shape and the same storage and offset, and no code is generated for it at all.

A **kernel** is the loop nest for one graph node, with a name for profiles. A **program** is the list of buffers, the kernels in order, and the sizes of the arena and persistent regions.

## 7.3 Lowering each operator

One function per kind of operator in `lower.cpp`. They share a few helpers:

- `nest(shape, body)`: one loop per axis of `shape`, outermost first. **An axis of size 1 gets the constant index 0 and no loop**, so a `[1, 16, 768]` tensor produces two loops, not three.
- `read(v, idx)`: a load of value `v`. A rank-0 inline constant is emitted as a literal, so the code says `x * 0.5`, not "load from a one-element table".
- `read_broadcast(v, out_idx)`: read `v` at an index of a larger broadcast shape. Where `v` has size 1 the index is the constant 0. **This is the interpreter's stride-0 trick done at compile time: the loop variable simply does not appear in the access.** A broadcast therefore costs nothing in the loop.

| Operator | Loop nest |
|---|---|
| `Add`, `Mul`, `Tanh`, `Where`, `Cast`, ... | `out[i...] = f(a[i...], b[i...])` with broadcasting |
| `Gelu` | the same, with the GELU formula as one expression; `x^3` is two multiplies, not a call to `pow` |
| `Permute` | `out[i...] = in[the indices shuffled back]` |
| `Slice` | `out[i...] = in[i... with start added on the axis]` |
| `Concat` | one nest per input, each writing its own stretch of the axis |
| `Gather` | `out[b..., t...] = x[idx[b...], t...]`: the index into `x` is itself loaded |
| `CumSum` | `out[0] = in[0]`, then `out[j+1] = out[j] + in[j+1]` |
| `ReduceAny` | `out = false`, then `out = out \| in[j]` |
| `MatMul`, `Gemm` | below |
| `LayerNorm` | per row: a sum loop for the mean, a sum loop for the variance, a loop that writes the output; `mean` and `rstd` are small temporary buffers with one element per row |
| `Softmax` | per row: a max loop, a loop that writes `exp(x - max)` and sums it, a loop that divides |
| `Input`, `Const`, `Reshape` | nothing: the value already exists in memory |

`CumSum` is worth a look because it shows the IR making a property visible. Its second loop reads `out[j]`, which the previous iteration wrote. That is a **loop-carried dependence**: the iterations must run in order. Chapter 10's analysis will find it by looking at exactly this load and store.

### The matrix products

There are two lowerings, depending on whether the `layout` pass (section 6.9) ran.

**Without `layout`** (operand `B` is `[k, n]`), the loop order is `i, p, j`, the same one the interpreter uses:

```
for i: { for j: C[i,j] = init
         for p: for j: C[i,j] = C[i,j] + A[i,p] * B[p,j] }
```

**With `layout`** (operand `Bt` is `[n, k]`), each element is a dot product:

```
for i: for j: { C[i,j] = init
                for p: C[i,j] = C[i,j] + A[i,p] * Bt[j,p] }
```

`init` is 0 for `MatMul` and `bias[j]` for `Gemm`. In the second form the lowering also records in the kernel which loop variable is `i`, which is `j`, and which is `p`, and a **signature** string such as `gemm [16,768] x [2304,768]^T`. The scheduler and autotuner use both.

### The causal softmax

For row `i`, only columns `0..i` take part. The loops' extent is the expression `i + 1`, and a final loop writes zeros to the columns after `i`. Half the exponentials of the full square are never computed. (In M3 this was written with a `Select` inside a full-length loop, because the IR did not yet have non-constant extents. M4 added them.)

## 7.4 Memory planning

Where should each value's buffer go in the arena? The naive answer (M2) gives every value its own slice. At T=1024 that is 6453 MB, and the first call spends 40 seconds just having the operating system hand over fresh pages.

Memory planning (M3) reuses slices. It is done during lowering by three mechanisms.

### Liveness

Kernels run in node order, so a value's lifetime is an interval: from its own node index to the index of the last node that reads it. Two values can share memory exactly when their intervals do not overlap.

`analyze()` computes `last_use` for every value in one pass over the graph. One subtlety: a `Reshape` does not own bytes. It is the same bytes as its operand. So each value has a **root**, the value that actually owns its memory, and a read of the reshaped value counts as a read of the root.

### In-place reuse

An elementwise kernel reads `in[i]` and then writes `out[i]`, and never looks at `in[i]` again. If this kernel is the *last* reader of that input, the output can simply take over the input's memory: by the time `out[i]` is stored, `in[i]` has been used.

`in_place_operand` checks the conditions:

- the operator is one of those that read each element before writing the same element (`Add`, `Sub`, `Mul`, `Pow`, `Tanh`, `Gelu`, `And`, `Not`, `Where`, `LayerNorm`, `Softmax`);
- this node is the operand's last reader;
- the operand has exactly the node's type (same shape and dtype);
- no other operand views the same bytes under a different shape (a broadcast or reshaped second view would read elements already overwritten).

When it applies, no new memory is allocated at all. A residual `add` writes its result over one of its inputs. A test checks that the model's output is bit-for-bit identical with planning on and off.

### The arena allocator

For values that need fresh memory, `ArenaPlanner` hands out byte ranges at compile time:

- It keeps a **free list**: ranges that were released, ordered by offset.
- `alloc(bytes)` uses **best fit**: the smallest free range that is large enough. Taking a snug block leaves the big ones for big tensors still to come. Whatever is left of the range goes back on the list.
- If nothing fits and the last free range touches the end of the arena, the arena grows by only the difference.
- `release` puts a range back and **coalesces** it with free neighbours, so two small adjacent holes can later serve one large request.
- Every range is rounded up to 64 bytes, so every buffer starts on a cache-line boundary.

After each node is lowered, `release_after` frees its temporaries and every operand it was the last reader of.

The planner reports two numbers: the arena size reached with reuse, and the total it would have been without.

| | T=16 | T=1024 |
|---|---:|---:|
| No reuse (M2) | 192 MB | 6453 MB |
| With planning (M3) | 154 MB | 158 MB |
| With planning, after M4 | 0.3 MB | 72 MB |

After M3 the arena was almost the same at every `T` because one buffer set it: the 154 MB transposed copy of the embedding table. M4's `layout` pass removed that buffer.

### Outputs

A value the caller wants back is computed straight into the caller's buffer (`Storage::Output`) when possible. When it is not (the output is a graph input, a constant, or the same value twice), a small copy kernel is added at the end.

## 7.5 Staging: compute it once, at load time

Some values depend only on the weights and on constants, not on the caller's input. They are the same on every call. Computing them on every call is waste.

Constant folding (section 6.5) cannot handle them: the compiler does not have the weight bytes. But the *runtime* does, once it has opened the weights file.

So lowering sorts every value by **when it becomes known**:

| Stage | What | Where it is computed |
|---|---|---|
| compile time | inline constants | by the compiler (`fold`) |
| load time | weights, and anything computed only from weights and constants | once, when the model is opened |
| run time | anything downstream of a graph input | on every call |

The rule (`stage()`): an `Input` is run-time; any other node is load-time if all of its inputs are load-time. A load-time node's kernel goes into a separate list, the **init kernels**, and its result goes into **persistent memory**, which is never freed or reused.

This is sometimes called **binding-time analysis**, or just hoisting. It is constant folding performed by the runtime.

In GPT-2 it applies to 49 values: the 48 transposed dense-layer weights that `layout` introduced, and the position-embedding lookup (`wpe[0..T-1]`, which depends only on `T`). They cost 340 MB of persistent memory and about 0.1 s at load.

## 7.6 The loop interpreter and verifier

`run_program` (`src/loop/interpreter.cpp`) executes a program by walking its tree. It runs the init kernels, then the kernels. Every loop runs serially in order, whatever its kind. Every index is checked against the size of its own axis, which is stricter than checking the flat offset: an index that runs off one axis but lands inside the buffer is still a bug, and is caught.

`verify_program` (`src/loop/ir.cpp`) checks structure: each access has as many indices as the buffer has axes; indices are integers; a stored value has the buffer's dtype; nothing stores to read-only storage; only init kernels write persistent memory and they never read caller inputs; loop variables are used only inside their loops and not bound twice in a nest; every buffer fits inside its region.

## 7.7 Where in the code

| File | Contents |
|---|---|
| `src/loop/ir.h` | the language: `Buffer`, `Expr`, `Stmt`, `Kernel`, `Program`, constructors |
| `src/loop/ir.cpp` | constructors with type checks and folding, `flat_index`, the printer, the verifier |
| `src/loop/lower.cpp` | `ArenaPlanner`; `Lowerer`: `stage`, `analyze`, `place`, `in_place_operand`, `release_after`, one function per operator |
| `src/loop/interpreter.cpp` | `run_program` |
| `tests/test_loop.cpp` | every lowering checked against the graph interpreter |
| `tests/test_passes.cpp` (end) | memory planning tests |

## 7.8 Further reading

- Poletto and Sarkar, "Linear Scan Register Allocation" (1999). Assigning storage from live intervals; memory planning here is the same idea with byte ranges instead of registers.
- Wilson et al., "Dynamic Storage Allocation: A Survey and Critical Review" (1995). Best fit, first fit, coalescing, and why they behave as they do.
- Pisarchyk and Lee, "Efficient Memory Management for Deep Neural Net Inference" (2020). Memory planning for exactly this setting (a static graph of tensors), with better algorithms than best fit.
- Jones, Gomard, and Sestoft, *Partial Evaluation and Automatic Program Generation* (1993), chapter 1, for binding times (section 7.5).
- The Halide and TVM papers cited in chapter 1 for loop-level IRs in production systems.
