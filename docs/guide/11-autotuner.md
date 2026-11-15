# 11. The autotuner

Chapter 10 ended with a default schedule whose parameters (a 4x5 tile, a block of about 64 KB) were chosen by hand from a few measurements. The autotuner (`src/tune/`, `mlc tune`) replaces hand-picking with a search: try many schedules, time each, keep the best, and write the answers to a file so that later compiles need no measuring.

It works, it is tested, and on this model it found very little. Both halves of that sentence are worth understanding.

## 11.1 Why tune by measurement

Which tile shape or block size is fastest depends on details that are hard to predict from first principles: how many registers the C++ compiler really manages to use, how the prefetcher reacts to a particular access pattern, how the caches are shared. Timing a candidate takes a fraction of a second and is always right about the machine it ran on. So production systems (ATLAS for linear algebra, FFTW for Fourier transforms, TVM for neural networks) search.

The separation of chapter 10 is what makes it safe. A schedule cannot change what is computed, so any point the search tries is a correct program.

## 11.2 What is tuned

The schedule of each distinct matrix product, as a small struct:

```cpp
struct MatmulParams {
  int tile_rows = 4, tile_cols = 5;  // the register tile
  bool cols_outside = false;         // which operand is blocked and streamed once
  int64_t block_rows = 0;            // rows per cache block; 0 = no cache tiling
};
```

Applying any value of this struct is the same sequence of rewrites as section 10.4 (`apply_matmul`). The hand-picked default is just one value, computed by `default_matmul_params`.

Optionally one global parameter: the work threshold below which a kernel stays on one thread (`--tune-threshold`).

Nothing else is tuned. Every other choice in the schedule is either forced by legality or made no measurable difference when tried by hand.

## 11.3 Kernel signatures

GPT-2 has 73 matrix products but only 7 distinct ones at a given `T`: the same shapes repeat in all twelve layers. Each dot-form kernel carries a **signature**, the operator and operand shapes:

```
gemm [16,768] x [2304,768]^T
matmul [12,16,64] x [12,16,64]^T
matmul [16,768] x [50257,768]^T
```

Two kernels with the same signature are the same loops over different memory, so a schedule measured for one is valid for all. The tuner works per signature.

## 11.4 The search space (`Space`)

A candidate is a point with three coordinates:

- **tile:** an index into the list of all tiles `(r, c)` with `r * c + r + c <= 32` (they fit the registers), `r * c >= 12` (enough accumulators to matter), and `r`, `c` no larger than the matrix. About 21 tiles for a large product.
- **block:** an index into a list of block sizes expressed as fractions of the L1 cache (1/16, 1/8, 3/16, 1/4, 3/8, 1/2, 3/4, 1, 2, 4), or -1 for "no cache tiling". The L1 size is read from the operating system (`sysctl hw.perflevel0.l1dcachesize`), so the same fractions mean the right thing on another machine. A block is rounded to a whole number of tiles.
- **which operand is outside:** two choices.

That is `21 * 11 * 2 = 462` points for a large product.

The coordinates are arranged so that neighbouring values are similar schedules, which is what makes "change one coordinate a little" a meaningful move.

A test (`tune_every_point_of_the_space_is_a_correct_schedule`) applies every distinct point of a small product's space, over a hundred schedules, and requires bit-identical results from the loop interpreter. The tuner may pick any point, so every point has to be right.

## 11.5 Measuring a candidate

Timing a schedule requires generating code for it and running it. Doing that one candidate at a time would mean one C++ compiler run each. Instead:

1. Build a small graph containing **the same product several times** (up to eight), all reading the same two operands.
2. Lower it. The results are never read, so the memory planner gives every copy the same piece of the arena. The copies differ in schedule and in nothing else.
3. Schedule each copy with a different candidate.
4. Generate C++, build one shared library, load it.
5. Run it repeatedly with the runtime's per-kernel timer. In each pass all candidates run in turn.
6. For each candidate keep the **minimum** time over all passes.

Two points of method:

- **Interleaving.** Because every candidate runs within each pass, anything that changes the machine's speed over time (heat, another process) affects all of them about equally.
- **Minimum, not mean.** On a quiet machine, noise only ever *adds* time: an interruption, a cache miss caused by something else. The smallest sample is the best estimate of what the code can do.

## 11.6 The search

For one product, with the default settings:

1. **Generation 0:** the default schedule plus 16 random points.
2. **Two more generations:** take the three best so far and **mutate** them. A mutation changes one coordinate by a small step: a nearby tile, the next block size up or down, or the other operand outside. Eight new candidates per generation. This is a simple evolutionary search.
3. **The final:** the three best candidates and the default, measured again side by side for three times as long.
4. **The decision:** a candidate replaces the default only if it wins the final by more than 2%.

So 33 of 462 points are tried per product.

Step 3 exists because of a statistical trap. If you measure 33 schedules that are all really the same speed, with a little noise each, the best-looking one will appear 1 to 2% faster than the default by luck alone. Picking it would be fitting noise. Re-measuring the leaders against the default, for longer, and demanding a margin, guards against that. An early version with a 1% margin produced "improvements" that a second run did not reproduce.

## 11.7 The cache (`tuning/cache.json`)

Results are saved as JSON:

```json
{
  "cpu": "Apple M2",
  "l1d_bytes": 131072,
  "performance_cores": 4,
  "power": "AC",
  "parallel_work": {"4": 65536},
  "products": [
    {"threads": 4, "signature": "gemm [1024,3072] x [768,3072]^T",
     "tile_rows": 5, "tile_cols": 4, "cols_outside": false, "block_rows": 5,
     "gflops": 331.8, "default_gflops": 287.2},
    ...
  ]
}
```

It is keyed by **CPU, thread count, and signature**. A schedule is only as good as the measurement behind it, so:

- the best schedule for one thread is not necessarily the best for four (with four, the cores compete for memory), hence the thread count in the key;
- a cache measured on a different CPU is ignored, with a warning.

The file is written after every product, so an interrupted run keeps what it learned.

Using it:

```sh
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o out --tuning tuning/cache.json --tuned-threads 4
```

That is deterministic and measures nothing: the tuned build is reproducible from the cache. A `ctest` pair builds T=16 from the committed cache and checks the logits.

## 11.8 Validation on the whole model

After tuning the products for a thread count, `mlc tune` builds the whole model twice, once with default schedules and once with tuned ones, and times them in **alternating** runs (default, tuned, default, tuned, ...), again so that drift affects both alike.

## 11.9 What it found

54 entries (7 products, 4 sequence lengths, 2 thread counts, minus two duplicates). The default was replaced in 12.

Whole model, default against tuned, from the tuner's own alternating runs (median seconds):

| | T=16 | T=64 | T=256 | T=1024 |
|---|---:|---:|---:|---:|
| 1 thread, default | 0.0473 | 0.1806 | 0.7038 | 3.1761 |
| 1 thread, tuned | 0.0466 | 0.1814 | 0.6998 | 3.1672 |
| 4 threads, default | 0.0228 | 0.0628 | 0.2274 | 0.9868 |
| 4 threads, tuned | 0.0222 | 0.0634 | 0.2300 | 0.9518 |

Tuned is ahead by 1.5 to 3.7% in three cells and within about 1% either way in the rest, which is noise. In the final benchmark session the two builds trade places from cell to cell.

**Why so little.** On one core, every sensible schedule runs the large products at 97 to 101 GFLOP/s. The kernel is limited by how fast the FMA units can go (section 9.3), not by the cache. A search over cache-related parameters has nothing to find. The products that did improve are the memory-bound ones on four threads: the 3072-wide projection at T=1024 went from 287 to 332 GFLOP/s, and the T=16 output projection from 296 to 348.

**Random search was enough.** In most products the best point after two generations of mutation was the best random point.

**The tuner optimizes the wrong thing at small `T`.** It times a kernel whose weight matrix stays in cache between runs. In the real model at T=16, every call reads each weight once from main memory (section 12.6). A schedule that is best with a warm operand is not necessarily best with a cold one. Making the measurement cold is listed as future work.

So the milestone's exit criterion, "tuned schedules beat hand-picked defaults on the model", is met only weakly. The honest conclusion is that the defaults from M4 were already at this machine's limit for this model, and the tuner mostly confirmed them.

## 11.10 A bug worth remembering

The first version rebuilt candidate libraries at the same file path. Results were nonsense: a "winner" 13% slower than the search's best. On macOS a library with thread-local state is never really unloaded, so loading the same path again returned the *old* code, and the tuner was timing stale candidates under new labels. Every candidate library now gets a fresh path and is deleted after use.

The general lesson: a measurement harness needs to be tested as carefully as the thing it measures.

## 11.11 Where in the code

| File | Contents |
|---|---|
| `src/tune/tuner.h` | the documented interface: `Machine`, `Cache`, `Space`, `Point`, `Product`, `TuneOptions` |
| `src/tune/tuner.cpp` | `detect_machine`, `load_cache` / `save_cache` / `apply_cache`, `Space`, `products_of`, `measure`, `tune_product` |
| `src/loop/schedule.*` | `MatmulParams`, `default_matmul_params`, `schedule_kernel`, the `tuned` table |
| `src/main.cpp` | `tune_command`: the driver, threshold tuning, model validation |
| `tuning/cache.json` | the committed results for the development machine |
| `tests/test_tune.cpp` | the space, the cache, signatures, and a real (very short) search |

## 11.12 Further reading

- Whaley, Petitet, and Dongarra, "Automated Empirical Optimizations of Software and the ATLAS Project" (2001). The original autotuned linear algebra library.
- Frigo and Johnson, "The Design and Implementation of FFTW3" (2005). Autotuning by composing and timing plans.
- Chen et al., "Learning to Optimize Tensor Programs" (NeurIPS 2018), and Zheng et al., "Ansor: Generating High-Performance Tensor Programs for Deep Learning" (OSDI 2020). TVM's autotuners, which add a learned model that predicts which candidates are worth timing. This project explicitly left that out.
- Ansel et al., "OpenTuner: An Extensible Framework for Program Autotuning" (2014), for search strategies in general.
- Mytkowicz et al., "Producing Wrong Data Without Doing Anything Obviously Wrong!" (ASPLOS 2009). On how easy it is for a performance measurement to be biased. Relevant to sections 11.5, 11.6, and 11.10.
