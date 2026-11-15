# 10. Loop transformations

This is the chapter where the model goes from 37 seconds to 3 at T=1024. It covers `src/loop/schedule.cpp`, the vector and parallel parts of `src/codegen/emit_cpp.cpp`, `runtime/mlc_neon.h`, and the thread pool.

To watch it happen on the real model:

```sh
build/release/mlc loops artifacts/graphs/gpt2_T16.json --disable vectorize,regtile,cachetile,parallel | less   # before
build/release/mlc loops artifacts/graphs/gpt2_T16.json | less                                                  # after
```

## 10.1 Two questions, kept apart

Every loop transformation raises two separate questions.

**Legality:** does the transformed loop compute the same values? This has a definite answer, and the compiler must be able to prove it. A rewrite that cannot prove it refuses (throws `ScheduleError`) and changes nothing.

**Profitability:** is it faster? This depends on the machine and the sizes, and is answered by measurement. The rewrites never ask it. The caller, either the default schedule or the autotuner, decides what to apply.

A **schedule** is a choice of which rewrites to apply, to which loops, with which parameters. The program's meaning is fixed by the naive loop nest; the schedule only changes how it runs. This separation, introduced by the Halide system, is what makes an autotuner possible: it can try any schedule without any risk to correctness.

## 10.2 Dependence analysis

Legality comes down to one question about a loop: **are its iterations independent?** That is, does any iteration read or write an element that a *different* iteration writes? If not, the iterations can run in any order, interleaved, or simultaneously.

Recall that the loop IR has no scalar variables (section 7.1). Every effect of an iteration is a `Store` to a buffer and every input is a `Load`. So the question is entirely about array indices.

### Index expressions are affine

Every index the lowering produces is a sum of loop variables times constants plus a constant, for example `i3 * 20 + i7 * 5 + 2`. Such an expression is called **affine**. `affine()` converts an expression into that form: a map from variable to coefficient, and a constant. Anything else (an index that is itself loaded from memory, as in `Gather`) is "not affine" and is treated conservatively.

Each loop variable has a known range, `0` to `extent - 1`, so the smallest and largest value of an affine expression can be computed.

### The proof the analysis looks for

For a loop over `v`, and for every buffer the loop body writes, the analysis looks for **one axis** along which **every** access to that buffer in the body (loads and stores alike) has an index of the form

```
c * v  +  r  +  (terms in outer loop variables)
```

where

- `c >= 1` is the same constant in all of them,
- the outer-variable part is identical in all of them,
- `r` (inner loop variables and constants) varies over a range narrower than `c`.

If such an axis exists, each iteration of `v` touches its own slice of the buffer along that axis, slices of different iterations cannot overlap, and the iterations are independent. If two iterations `v1 != v2` collided, their `r` values would have to differ by a multiple of `c`, which a spread smaller than `c` rules out.

Anything the analysis cannot prove counts as dependent. It is **sound** (it never says "independent" wrongly) but not complete (it sometimes says "dependent" when a cleverer analysis could prove otherwise).

### Examples

The naive matrix product:

```
for i: for j: { C[i,j] = 0
                for p: C[i,j] = C[i,j] + A[i,p] * Bt[j,p] }
```

- Loop `i`: `C` is the only buffer written. Along axis 0 every access is at `i`. Independent.
- Loop `j`: along axis 1 every access is at `j`. Independent.
- Loop `p`: no index of `C` contains `p` at all. Every iteration writes the same element. Dependent. (It is a sum, a special case handled by `vectorize` below.)

Cumulative sum, `out[j+1] = out[j] + in[j+1]`: along the axis, one access is at `j + 1` and another at `j`. The spread of `r` is 1, not less than `c = 1`. Dependent, correctly: each step needs the previous one.

After blocking, an index looks like `jc * 20 + jj` with `jj` in `0..19`. For the loop over `jc`: `c = 20`, `r = jj` with spread 19, which is less than 20. Independent. This is why the proof is phrased with a stride and a range rather than "the index is exactly `v`".

### Buffers that share memory

The analysis treats two buffers as the same when they are the same bytes under the same shape (an in-place kernel's input and output). If two buffers overlap under *different* shapes, the indices cannot be lined up, and the loop is declared dependent.

## 10.3 The rewrites

Each takes a kernel body and the variable of the loop to act on, and returns a new body. Loops are named by their variable. Some rewrites copy a nest (a main part and a leftover part), which leaves several loops binding the same variable in sibling positions; a variable then names all of them and the next rewrite applies to each.

### `split(v, factor)`

```
for v in 0..N: S(v)
        becomes
for outer in 0..N/f:  for inner in 0..f:  S(outer*f + inner)
for tail  in 0..N%f:                      S((N/f)*f + tail)
```

Iterations run in exactly the same order, so it is always legal and on its own changes nothing. It exists to create the loops that `interchange` moves.

### `interchange(v)`

```
for v: for w: S     becomes     for w: for v: S
```

Requires `v`'s body to be exactly one loop, and `v`'s iterations to be independent. Why independence is the condition: afterwards, iteration `(v, w)` can run before `(v', w')` with `v' < v`. That reorders things across different values of `v`, which is harmless exactly when different values of `v` never depend on one another.

### `unroll_and_jam(v, factor)`

**Unrolling** a loop by 4 means writing the body four times per step (for `v`, `v+1`, `v+2`, `v+3`) and stepping by 4. **Jamming** means fusing those four copies all the way down: rather than four copies of the inner loops one after another, there is one copy of the inner loops whose innermost statements are each repeated four times.

```
for i: for p: C[i] += A[i,p] * b[p]
        becomes (factor 2)
for io: for p: { C[2*io]   += A[2*io,  p] * b[p]
                 C[2*io+1] += A[2*io+1,p] * b[p] }
```

The effect: one pass over `p` now computes two elements of `C`. They share the load of `b[p]`, and each has its own running sum, an independent dependency chain (section 9.3).

Requires `v`'s iterations to be independent (the copies' statements now run interleaved; each copy still sees its own statements in order). Leftover iterations, when `N` is not a multiple of the factor, are jammed too, all at once, into a smaller group. Left as a one-at-a-time loop they would have too few accumulators; this was measured to matter (a 64-column product with a 5-wide group left 4 columns, 6% of the work, running at a third of the speed).

### `vectorize(v, interleave)`

Marks a loop to run in SIMD lanes. It is legal when the body is a list of stores and each store is one of two kinds:

- **Elementwise:** iteration `v` writes element `v` of a contiguous run (the index along the last axis is `v` plus something constant in this loop). Anything else in the body that touches the same buffer must touch exactly the same element, or a different row entirely. Four iterations then become one vector store.
- **Reduction:** every iteration combines into the same element, as `t = t + e` or `t = max(t, e)`, and nothing else in the loop touches that element. The vector loop keeps the running value in a register and reorders the additions, which is the one place the compiler changes floating-point order (section 9.7).

The values involved must be something a vector can hold: 32-bit float arithmetic on things that are either the same for the whole loop (broadcast into all lanes) or read from consecutive addresses. That rules out `Select`, integer and boolean values, `pow`, and reads that stride across rows (as in `Permute`).

`plan_vector_loop` makes this decision and reports, for each store, which kind it is. The code generator calls the same function, so the two cannot disagree.

`interleave` is the number of independent accumulator sets for a reduction. A lone sum (layer norm's mean) gets 4, so that four dependency chains run at once. A loop that already has many accumulators gets 1.

### `parallelize(v)`

Marks a loop to run on several threads. Requires independence.

## 10.4 Deriving the fast matrix product

This is the central piece of the project. Start from the naive nest, with `A` of shape `[m, k]` and `Bt` of shape `[n, k]` (section 7.3):

```
for i in 0..m:
  for j in 0..n:
    C[i,j] = init
    for p in 0..k:
      C[i,j] = C[i,j] + A[i,p] * Bt[j,p]
```

Each element of `C` is a dot product of a row of `A` and a row of `Bt`. Both rows are contiguous in memory. That is why the `layout` pass (section 6.9) put the product in this form.

As written it has two problems. There is one accumulator, so it runs at a fraction of the FMA units' capacity (section 9.3). And it is not vectorized.

### Step 1: register tiling

Apply `unroll_and_jam` to `i` with factor 4 and to `j` with factor 5. The innermost loop becomes:

```
for p in 0..k:
  C[i+0,j+0] += A[i+0,p] * Bt[j+0,p]
  C[i+0,j+1] += A[i+0,p] * Bt[j+1,p]
  ...                                      20 statements: a 4 x 5 tile of C
  C[i+3,j+4] += A[i+3,p] * Bt[j+4,p]
```

One pass over `p` now computes twenty elements of `C`. It reads 4 rows of `A` and 5 rows of `Bt`: 9 loads feeding 20 multiply-adds, because each loaded value is used several times.

### Step 2: vectorize `p`

Each of the twenty statements is a reduction over `p`. Vectorizing keeps each in its own NEON register and processes four values of `p` per step:

- 20 accumulator registers,
- 4 registers for the loaded `A` values, 5 for `Bt`,
- 29 of the 32 registers in use.

Every step does 9 vector loads and 20 FMAs, and there are 20 independent dependency chains, enough to keep all four FMA units busy on every cycle. This loop is what high-performance libraries call the **micro-kernel**. Here it was not written by hand: it fell out of two general rewrites.

The tile shape is a trade-off inside a fixed budget: `rows * cols + rows + cols <= 32` registers. 4x5 has 20 accumulators. 4x6 would need 34 registers and spill to memory, which measured at half speed.

### Step 3: cache tiling

The register tile fixes the innermost loop. The loops around it decide what is in cache.

Whichever operand has more rows is put on the outside (if that is `Bt`, an `interchange` swaps `i` and `j`). Call it the *big* operand. It is then `split` into blocks of rows, and the block's inner loop is `interchange`d inside the loop over the other operand:

```
for block of the big operand's rows:        the block is loaded into L1 once
  for each tile of the small operand:
    for each tile of rows in the block:
      (the micro-kernel)
```

Every row of the small operand meets a block that is already in L1. The big operand is streamed through memory exactly once.

The block size is chosen so that the block, plus what it is multiplied against, fits comfortably in the 128 KB L1 cache (the default aims for about 64 to 72 KB in total).

### Step 4: threads

The outermost loop is marked parallel. Each thread gets a contiguous range of blocks, and therefore its own rows or columns of `C`.

### The result

For the output projection at T=16, `mlc loops` prints (abbreviated):

```
kernel matmul  # %684 mm
  parallel for i3 in 0..2512:               blocks of 20 rows of the embedding table
    for i6 in 0..4:                         tiles of 4 rows of A
      for i7 in 0..4:                       tiles of 5 rows within the block
        v684[...] = 0   (20 of these)
        vector for i2 in 0..768:
          v684[(i6*4), (i3*20 + i7*5)] = (v684[...] + (v683[(i6*4), i2] * v1[(i3*20 + i7*5), i2]))
          ... 19 more
```

### What this design is, and is not

The standard design for a fast matrix product (GotoBLAS, BLIS) is different: it **packs** both operands, copying them into contiguous buffers arranged for the micro-kernel, and its micro-kernel is in "outer product" form (each step multiplies a short column of `A` by a short row of `B`).

This project uses the "dot product" form on unpacked operands instead. The reasoning:

- At T=16 the operand `A` has only 16 rows. Packing a weight matrix costs as much as multiplying by it.
- Packing the weights once at load time is possible, but costs the same memory as transposing them at load time, which is what dot form needs.
- The largest operand of all, the 154 MB embedding table, is *already* in the layout dot form wants.

The cost, measured: the outer-product micro-kernel peaks about 5% higher than the dot-product one, and at T=1024 on AC power ONNX Runtime's kernel (which is of that kind) is about 9% faster than ours on large products. Section 12.6 has the numbers.

## 10.5 The default schedule

`schedule_program` applies, to every kernel:

1. If it is a dot-form matrix product: the steps of section 10.4, with parameters from `default_matmul_params` (or from the autotuner's cache). The tile is 4x5, swapped to 5x4 or 4x4 if that leaves clearly fewer leftover rows.
2. `vectorize` every innermost loop that `plan_vector_loop` accepts.
3. `parallelize` each outermost loop if its iterations are independent and the kernel is big enough to repay waking threads. "Big enough" is an estimate of the work (stores executed, with a store that calls `exp` or `tanh` counted as twenty) compared against a threshold. Because shapes are static, this decision is made at compile time, per kernel.

What ends up vectorized in GPT-2: all matrix products, layer norm (three loops), softmax (three loops plus the zero fill), GELU, the additions and scalings, and the embedding lookup (a contiguous copy). What stays scalar: permutes (strided reads) and slices' outer structure. They are under 1% of the run.

## 10.6 Vector loops in generated C++

For a loop over `v` with 4 lanes and interleave `U`, the emitter writes:

```
accumulators = identity                         one per reduction and per interleave slot (0 for a sum, -inf for a max)
for (v = 0; v + 4*U <= N; v += 4*U)  body, U vectors at a time
for (;      v + 4   <= N; v += 4)    body, one vector            (only if U > 1)
scalars = target (+) lanes of accumulators      fold each reduction into one float
for (;      v < N; ++v)              scalar body                  (the leftover iterations)
target = scalars
```

The mean of a layer norm, as generated:

```cpp
float32x4_t acc0_0 = vdupq_n_f32(0.0f);       // four independent accumulators
float32x4_t acc0_1 = vdupq_n_f32(0.0f);
float32x4_t acc0_2 = vdupq_n_f32(0.0f);
float32x4_t acc0_3 = vdupq_n_f32(0.0f);
int64_t i1 = 0;
for (; i1 + 16 <= 768; i1 += 16) {
  const float32x4_t x0 = vld1q_f32(v6 + ((i0 * 768) + i1));
  acc0_0 = vaddq_f32(acc0_0, x0);
  const float32x4_t x1 = vld1q_f32(v6 + ((i0 * 768) + (i1 + 4)));
  acc0_1 = vaddq_f32(acc0_1, x1);
  ...
}
float r0 = t9_mean[i0] + vaddvq_f32(vaddq_f32(vaddq_f32(vaddq_f32(acc0_0, acc0_1), acc0_2), acc0_3));
t9_mean[i0] = r0;
```

Details of the emitter:

- **Loop-invariant values** (a scalar constant, the row's mean) are broadcast into a vector once, before the loop.
- **Each distinct memory operand is loaded once per iteration**, however many statements use it. In the matrix product, each of the 9 rows is loaded once and used in 4 or 5 FMAs.
- **A value just stored is reused from its register** if a later statement reads it (softmax writes `exp(...)` and immediately adds it to the sum).
- **`a + b * c` is emitted as one FMA.**
- **Leftover loops are omitted** when the trip count is a known multiple of the vector width.

### The instruction-set table

`src/codegen/isa.h` is the only place that names an intrinsic: a struct with one string per operation (`add = "vaddq_f32"`, `fma = "vfmaq_f32"`, ...). The emitter asks the table. Supporting another instruction set (AVX2 on x86) means adding a second table and a second math header; the loop IR and emitter stay as they are.

## 10.7 Vector `exp` and `tanh` (`runtime/mlc_neon.h`)

The C library computes `exp` and `tanh` one float at a time. A vectorized loop that had to call it once per lane would gain nothing. So there are two handwritten vector functions.

**`exp`.** Write `x = n * ln(2) + r` with `n` an integer and `|r| <= 0.35`. Then `exp(x) = 2^n * exp(r)`.

- `n = round(x / ln 2)`.
- `r = x - n * ln 2`, computed with `ln 2` split into two parts so the first subtraction is exact.
- `exp(r)` is approximated by a degree-6 polynomial, accurate to float precision on that short interval. The coefficients are from the Cephes math library.
- `2^n` is built directly by writing `n + 127` into the exponent bits of a float.

The input is first clamped to `[-87.3, 88.0]` so the result is never zero, subnormal, or infinite. Measured relative error over the whole range: below 2.5e-7.

**`tanh`.** `tanh(x) = 1 - 2 / (exp(2x) + 1)`, with `x` clamped to `[-10, 10]`, beyond which `tanh` is 1 or -1 to float precision. Absolute error below 2.5e-7. Relative error is poor for `|x|` below about 0.001, where the subtraction cancels; GELU multiplies by `x` again, so it does not matter here. A general-purpose `tanh` would switch to a polynomial near zero.

Both are tested against the C library over a fine grid of inputs (`tests/test_schedule.cpp`).

## 10.8 Parallel loops in generated C++

A parallel loop is **outlined**: its body becomes a separate function of an index range, and the loop is replaced by a call that hands that function to the runtime.

```cpp
static void k7_gemm_par0(<the kernel's pointers>, int64_t begin, int64_t end) {
  for (int64_t i3 = begin; i3 < end; ++i3) { ... }
}
...
auto range = [=](int64_t begin, int64_t end) { k7_gemm_par0(v1, v683, v684, begin, end); };
mlc_parallel_for(2512, [](void* f, int64_t b, int64_t e) { (*static_cast<decltype(range)*>(f))(b, e); }, &range);
```

The outlined function again takes the buffers as `__restrict` parameters.

## 10.9 The thread pool (`runtime/mlc_runtime.cpp`)

`mlc_parallel_for(n, fn, context)` runs `fn(context, begin, end)` over `0..n`, cut into one contiguous range per thread, and returns when all are done.

- **A fixed pool.** `threads - 1` worker threads are created once and live as long as the model is open. The calling thread takes a range itself.
- **Static scheduling.** The range is cut into as many pieces as there are threads, decided up front. Nothing is handed out on demand, which costs nothing per iteration and keeps each thread on its own stretch of memory.
- **Spin, then sleep.** An idle worker watches for the next job in a tight loop for a short while (tens of microseconds) and only then goes to sleep on a condition variable. Kernels at small `T` last tens of microseconds and follow one another closely; workers that slept between them would spend longer waking than working.
- **A job lives on the caller's stack.** It is published through an atomic pointer. Before returning, the caller un-publishes it and waits for every worker that might still hold the pointer to let go, tracked with a counter that workers increment *before* reading the pointer.
- **No nesting.** A parallel loop reached from inside a parallel range runs on the calling thread.
- **Quality of service.** Workers request the class that makes macOS place them on performance cores.
- **Thread count.** `mlc_set_threads(n)`, or the `MLC_THREADS` environment variable, or by default the number of performance cores.

## 10.10 What each step was worth

One optimization switched on at a time, T=256, seconds, AC power (`bench/results/m4_ablation_cumulative.md`):

| Step | Time | Factor |
|---|---:|---:|
| graph passes only | 9.209 | |
| + `noalias` | 9.198 | 1.00 |
| + `vectorize` | 3.297 | 2.79 |
| + `vmath` | 3.396 | 0.97 |
| + `layout` | 2.495 | 1.36 |
| + `hoist` | 2.427 | 1.03 |
| + `regtile` | 0.711 | 3.41 |
| + `cachetile` | 0.688 | 1.03 |
| + `parallel`, 4 threads | 0.212 | 3.25 |

Reading it:

- **`vectorize`**, the dot-form **`layout`**, and **`regtile`** produce nearly all of the 13x on one thread.
- **`noalias` alone did nothing**, although M2 had identified aliasing as the reason clang would not vectorize. With the promise in place clang's report changes, but the old loop still runs at the same speed. This was never explained. It stopped mattering once the compiler emitted SIMD itself.
- **`vmath`** affects only softmax and GELU, about 3% of the run.
- **`hoist`** matters at small `T`: at T=16 it is 0.211 to 0.145 s, because without it 340 MB of weights are transposed on every call.
- **`cachetile` adds 3%.** The kernel is limited by arithmetic, not by the cache, on one thread.

## 10.11 Where in the code

| File | Contents |
|---|---|
| `src/loop/schedule.h` | the documented interface: analysis, rewrites, `MatmulParams`, `ScheduleOptions` |
| `src/loop/schedule.cpp` | `affine`, `independent`, `plan_vector_loop`, the five rewrites, `apply_matmul`, `default_matmul_params`, `schedule_kernel`, `schedule_program` |
| `src/graph/passes.cpp` | `transpose_matmul_operands` (the `layout` pass) |
| `src/codegen/emit_cpp.cpp` | `vector_loop`, `vector_body`, `lanes`, `parallel_loop`, `plan_pointers` |
| `src/codegen/isa.h` | the NEON table |
| `runtime/mlc_neon.h` | vector `exp` and `tanh` |
| `runtime/mlc_runtime.cpp` | `Pool`, `mlc_parallel_for` |
| `tests/test_schedule.cpp` | the analysis, each rewrite against the loop interpreter, compiled kernels at awkward sizes, math accuracy |

## 10.12 Further reading

- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (2008), and Van Zee and van de Geijn, "BLIS: A Framework for Rapidly Instantiating BLAS Functionality" (2015). The standard design that section 10.4 deviates from, and the vocabulary (micro-kernel, packing) it borrows.
- Allen and Kennedy, *Optimizing Compilers for Modern Architectures* (2001). The reference on dependence analysis and loop transformations. Chapters 2, 3, and 5 cover everything in sections 10.2 and 10.3, in far more generality.
- Wolfe, *High Performance Compilers for Parallel Computing* (1996), for the same material from another author; and Carr and Kennedy, "Improving the Ratio of Memory Operations to Floating-Point Operations in Loops" (1994), on unroll-and-jam specifically.
- Ragan-Kelley et al., the Halide paper (chapter 1), for schedules as a separate object.
- Ikarashi et al., "Exocompilation for Productive Programming of Hardware Accelerators" (PLDI 2022). A system where a BLIS-quality kernel is derived by user-applied, checked rewrites, the same idea as section 10.4 carried much further.
- The Cephes Mathematical Library (Stephen Moshier), `expf.c`, for the polynomial in section 10.7; and Muller, *Elementary Functions: Algorithms and Implementation*, for how such approximations are designed.
- Herlihy and Shavit, *The Art of Multiprocessor Programming*, chapters on thread pools and work distribution, for section 10.9; and the OpenMP specification's description of `schedule(static)`.
