# 9. What makes code fast on a CPU

Chapter 10 transforms loops to make them fast. Every transformation there exists because of a specific property of the hardware. This chapter is those properties. Nothing here is specific to the project; it is the background that the rest assumes.

The machine: an Apple M2. Four **performance cores** (fast) and four **efficiency cores** (slow, low power). The project targets the performance cores.

## 9.1 The unit of measurement

A matrix product of `[m, k]` by `[k, n]` does `2 m k n` floating-point operations (flops). **GFLOP/s** is billions of flops per second. It measures a matrix-product routine independently of the matrix sizes.

Reference points on this machine (AC power, one performance core):

| What | GFLOP/s |
|---|---:|
| The naive loop from M2, in the model | about 8 |
| This project's final kernel, in the model | 80 to 100 |
| The same kernel on data that fits in cache | about 100 |
| Theoretical limit of one core's SIMD units | about 110 |
| Apple Accelerate (uses other hardware) | 400 to 1400 |

## 9.2 SIMD

**SIMD** (single instruction, multiple data) instructions operate on several numbers at once. On this CPU the SIMD instruction set is **NEON**. A NEON register is 128 bits wide and holds four 32-bit floats. One instruction adds, multiplies, or loads all four **lanes** together.

In C++ these instructions are reachable as **intrinsics**: functions that the compiler turns into exactly one instruction.

```cpp
float32x4_t a = vld1q_f32(p);          // load 4 floats from address p
float32x4_t b = vld1q_f32(q);
float32x4_t c = vfmaq_f32(acc, a, b);  // acc + a*b, on all 4 lanes
vst1q_f32(r, c);                       // store 4 floats
float s = vaddvq_f32(c);               // add the 4 lanes into one float
```

There are 32 NEON registers. That number will matter.

**Vectorizing** a loop means rewriting it to process four iterations per step using these instructions, plus a scalar loop for the leftover iterations when the trip count is not a multiple of four.

**FMA** (fused multiply-add) computes `a + b*c` as one instruction with one rounding at the end. It is both faster and slightly more accurate than a multiply followed by an add. A matrix product is nothing but FMAs.

## 9.3 Pipelines: latency and throughput

A modern core does not execute one instruction at a time. It has several execution units and runs instructions as soon as their inputs are ready, out of order. Two numbers describe an instruction:

- **Latency:** how long from inputs ready to result ready. For an FMA on this core, about 3 to 4 clock cycles.
- **Throughput:** how many can be *started* per cycle. This core has four SIMD units, so four FMAs per cycle.

The gap between those two numbers is the key to the fast matrix product.

Consider a dot product with one accumulator:

```
acc = acc + a[p] * b[p]      for p = 0, 1, 2, ...
```

Each FMA needs the previous `acc` as input. They form a **dependency chain**: the next cannot start until the last finishes. The loop runs at one FMA per latency, about one per 3 cycles, while the hardware could start four per cycle. Eleven twelfths of the capacity is idle.

The fix is to run many *independent* chains at once. With 16 or more separate accumulators, each waiting on its own previous value, there is always something ready for every unit on every cycle.

This was measured directly in a hand-written experiment (M4): a kernel with 1 accumulator ran at 4.8 GFLOP/s, with 4 at 17.5, with 8 at 32, and with 16 or 20 at 59, where it stopped improving. (Those absolute numbers were taken in Low Power Mode, section 9.9. The ratios are what matter.)

The theoretical peak follows: 4 FMAs per cycle, each doing 4 multiplies and 4 adds, at about 3.5 billion cycles per second, is `4 * 8 * 3.5 = 112` GFLOP/s.

## 9.4 Caches

Main memory is slow relative to the core: a read from it takes on the order of a hundred cycles. So the CPU keeps recently used data in small fast memories called **caches**.

| Level | Size (performance core) | Roughly how long a read takes |
|---|---:|---|
| L1 data cache | 128 KB | 3 to 4 cycles |
| L2 cache | 16 MB (shared by the four performance cores) | about 15 cycles |
| Main memory | gigabytes | 100 cycles or more |

Caches work in **lines** of 64 bytes (16 floats). Touching one byte brings in its whole line.

Three consequences drive everything in chapter 10:

1. **Sequential access is cheap.** Reading memory in order touches each line once, and the CPU's **prefetcher** notices the pattern and fetches lines ahead of time. Jumping around defeats both.
2. **Reuse must happen while data is still in cache.** If a loop needs the same data many times, arrange for those uses to be close together in time, so the data is still in L1. This is **locality**.
3. **Memory bandwidth is finite.** Even perfectly sequential reads from main memory arrive at a limited rate, a few tens of gigabytes per second for the whole machine. A computation that does little arithmetic per byte it reads is limited by this, not by the core. Section 12.6 is an example.

A cache is also divided into **sets**, and each address can only live in one set, which has a handful of slots (for a 128 KB cache with 8 slots per set, addresses a multiple of 16 KB apart compete for the same 8 slots; the exact arrangement on this core was not measured). Accessing a matrix column whose elements are spaced by a "round" number of bytes can therefore thrash a cache that is nowhere near full. This is one of the classical reasons high-performance libraries copy data into contiguous buffers ("packing") before multiplying, and one reason this project's kernel is designed to read only along rows.

## 9.5 Why loop order and blocking matter

Take `C[i][j] += A[i][p] * B[p][j]` with matrices too large for the cache.

- With `p` innermost, the loop walks down a column of `B`, one element per row, jumping `n` floats each step. Every step is a new cache line.
- With `j` innermost, it walks along a row of `B` and a row of `C`. Sequential.

That is **loop interchange** (section 5.5 used it already).

**Blocking** (also called **tiling**) goes further. If a loop sweeps over a large array many times, split the array into blocks that fit in cache and finish all the work on one block before moving to the next. Each block is then loaded once and used many times while it is in L1, instead of being reloaded on every sweep.

## 9.6 Aliasing

Two pointers **alias** if they can refer to the same memory. Consider

```cpp
for (j ...) C[j] = C[j] + a[0] * B[j];
```

Can the compiler load `a[0]` once before the loop? Only if no store to `C[j]` can change `a[0]`. If `C` and `a` are just two pointers into unknown memory, it cannot know, and must reload `a[0]` on every iteration. Worse, it cannot vectorize: processing four iterations at once changes the result if the store in iteration 1 was supposed to affect the load in iteration 2.

`restrict` (spelled `__restrict` in C++) on a pointer parameter is the programmer's promise that, during the function, memory written through that pointer is not reached through any other. With it the compiler can assume no overlap.

This project's compiler knows its buffers do not overlap, because its own memory planner placed them. So it emits `__restrict` (the `noalias` switch). It has to be on function parameters because clang ignores `restrict` on local variables.

## 9.7 Floating point is not associative

A 32-bit float holds about seven significant decimal digits. Every operation rounds its result to the nearest representable value. Because of that rounding,

```
(a + b) + c    and    a + (b + c)
```

can differ in the last digit. Adding 768 numbers in a different order gives a slightly different sum.

Two consequences:

**The C++ compiler may not reorder float additions.** The language requires the result you would get by doing the operations in the written order. A vectorized sum adds lanes separately and combines them at the end, which is a different order. So `clang` will not vectorize a loop like `sum += x[i]` unless you pass `-ffast-math`, which permits reordering everywhere. This is why "transposed layout without our own vectorizer" was a large *slowdown* in the M4 ablation: a dot product is a float sum, and clang ran it one element at a time.

**The project's vectorizer reorders deliberately.** It is allowed to because the correctness criterion is a tolerance (1e-3), not bit equality. The reordering is a decision made in the loop IR (the loop is marked vectorized), visible in `mlc loops`, and limited to sums and maxima. The differences it causes are around 1e-4 on logits of magnitude 100, which is the expected size for seven-digit arithmetic.

Other facts used in the project: infinity and NaN ("not a number", the result of 0/0) are ordinary float values; `exp(-inf) = 0`; any comparison with NaN is false.

## 9.8 Threads

Four performance cores can run four pieces of work at once. To use them, a loop whose iterations are independent is split into four ranges, one per thread. Things that limit the gain:

- **Starting and waking threads costs time.** Waking a sleeping thread through the operating system takes tens of microseconds, comparable to an entire small kernel at T=16. The runtime keeps its threads alive and lets idle ones spin briefly before sleeping (section 10.9).
- **Shared resources.** The four cores share the L2 cache and the memory bus. A computation limited by memory bandwidth does not speed up four times.
- **No thread affinity on macOS.** A program cannot pin a thread to a core. It can only request a high quality-of-service class, which makes the scheduler prefer performance cores.
- **Amdahl's law.** If a fraction of the work is not parallel, it bounds the speedup.

Measured: four threads give 3.0 to 3.6 times the speed on the large matrix products in isolation, and 2.1 to 3.3 times on the whole model.

## 9.9 The power state of a laptop

This cost a full day of misleading numbers, so it has its own section.

On battery with **Low Power Mode** on, macOS caps the performance cores at roughly half their clock speed and throttles sustained multi-core load. In that state:

- One core peaked at 59 GFLOP/s instead of about 100.
- Four-thread runs at T=1024 took 1.6 s the first time and 3.2 s on later runs, as throttling set in.
- Because the core was slow relative to memory, memory effects were hidden. In Low Power Mode this project's kernel measured *faster* than ONNX Runtime's on large products; at full speed it is 9% *slower*. The conclusion reversed.

Every result file in `bench/results/` records the power state. Numbers are only compared within one session.

## 9.10 Why PyTorch is so much faster on this machine

PyTorch on macOS sends matrix products to Apple's **Accelerate** library. In M0 it measured about 700 GFLOP/s from a single thread, with CPU time equal to wall time. One core's NEON units cannot exceed about 110. So Accelerate is using hardware other than the ordinary SIMD units (Apple's matrix coprocessor), and the thread-count setting does not govern it.

No code built from NEON intrinsics can match that, which is why the project's stated peer is **ONNX Runtime**, whose CPU kernels use the same instructions this compiler emits.

## 9.11 Further reading

- Hennessy and Patterson, *Computer Architecture: A Quantitative Approach*, chapter 2 (memory hierarchy) and chapter 3 (out-of-order execution). The standard textbook for sections 9.3 and 9.4.
- Drepper, "What Every Programmer Should Know About Memory" (2007). Long, free, and the best practical account of caches. Sections 3 and 6 are the relevant ones.
- Goldberg, "What Every Computer Scientist Should Know About Floating-Point Arithmetic" (1991). For section 9.7.
- Agner Fog, "Optimizing software in C++" and "The microarchitecture of Intel, AMD and VIA CPUs". Written for x86 but the concepts (latency, throughput, dependency chains) are identical.
- Arm, "Neon Intrinsics Reference" (developer.arm.com) for every intrinsic by name; and Arm's "Learn the architecture: Neon programmer's guide".
- Dougall Johnson's Firestorm microarchitecture notes (dougallj.github.io/applecpu/firestorm.html) for measured latencies and unit counts on Apple performance cores. The M2's cores are a successor of the one documented there.
- Goto and van de Geijn, "Anatomy of High-Performance Matrix Multiplication" (ACM TOMS, 2008). The paper that explains how caches dictate the structure of a fast matrix product. Read this before chapter 10 if you read only one thing.
