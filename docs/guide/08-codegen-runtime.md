# 8. Code generation and the runtime

The loop IR is still a data structure inside the compiler. This chapter covers the last steps: printing it as C++, compiling that into a shared library, and the small handwritten runtime that the library contains and that actually runs the model.

To follow along:

```sh
build/release/mlc compile artifacts/graphs/gpt2_T16.json -o build/release/gen/gpt2_T16
less build/release/gen/gpt2_T16/model.cpp
```

## 8.1 Why emit C++ and not machine code

A compiler normally ends by producing machine instructions. This one ends by producing C++ source text and then running `clang++` on it. Reasons:

- Turning arithmetic into good machine code (choosing instructions, assigning registers) is a large, well-solved problem. Using an existing compiler for it leaves the project free to spend its effort on the parts that are specific to neural networks.
- The output is readable. When something is slow or wrong you can open `model.cpp` and look.
- It is portable to any machine with a C++ compiler.

The division of labour is: this compiler makes every decision that requires knowing the program is a neural network (what to fuse, where memory goes, which loops to tile or vectorize). `clang++` handles the last mile. Chapter 10 shows where that line had to be moved: clang would not vectorize the loops on its own, so the project emits SIMD instructions explicitly.

## 8.2 What the generated file contains

For a program with *N* kernels, `model.cpp` has:

1. Includes, and one helper, `mlc_max`.
2. **Inline constants** as static arrays, for example `const int64_t inline_0[16] = {0, 1, ..., 15};`. Only small tables. Weights are never written into the code.
3. **One pair of functions per kernel** (below).
4. **Two tables**: the init kernels and the per-call kernels, each entry holding the operator name, a label, and a function pointer.
5. **`mlc_gen_model`**, a struct describing the model: arena size, expected weights file size, persistent memory size, the sizes of each input and output, and the two tables. This struct is the entire interface between generated code and the runtime.

## 8.3 One kernel in C++

A serial kernel is a direct transcription. An `Add` of two `[1, 16, 768]` tensors:

```cpp
// %6 add_1
static void k1_add_body(const float* __restrict v2, const float* __restrict v5, float* __restrict v6) {
  for (int64_t i0 = 0; i0 < 16; ++i0) {
    for (int64_t i1 = 0; i1 < 768; ++i1) {
      v6[((i0 * 768) + i1)] = (v2[((i0 * 768) + i1)] + v5[((i0 * 768) + i1)]);
    }
  }
}

void k1_add(char* arena, const char* weights, char* persist,
            const void* const* inputs, void* const* outputs) {
  k1_add_body((const float*)(arena + 0), (const float*)(persist + 0), (float*)(arena + 49152));
}
```

(Simplified for illustration: the offsets are made up, and in the real file the inner loop is vectorized, section 10.6.)

**Flat indexing.** The IR's `v6[i0, i1]` becomes one subscript by Horner's rule (section 2.1): `((i0 * 768) + i1)`. `flat_index` builds that expression.

**Float literals** are printed with nine significant digits (`0.100000001f`), which is always enough for a 32-bit float to read back as exactly the same value. Infinity is printed as `INFINITY`.

**Why two functions.** The outer function has the fixed signature the runtime calls: pointers to the three memory regions and to the caller's arrays. It turns offsets into typed pointers. The inner function receives those pointers as *parameters* marked `__restrict`. The reason is in section 9.6; briefly, `__restrict` is a promise to the C++ compiler that these pointers do not overlap, and C++ compilers only honour that promise on function parameters.

**Pointer grouping.** Buffers that are the same bytes (a value and its reshaped view; an in-place kernel's input and output) share one pointer. A pointer is marked `__restrict` only if no *other* pointer of the kernel reaches memory that is written through it. The memory plan from chapter 7 is what makes this decidable: the compiler knows every buffer's exact byte range.

## 8.4 Building the library (`src/codegen/build.cpp`)

`build_shared_library` runs:

```
clang++ -std=c++17 -O3 -mcpu=native -pthread -shared -fPIC -fvisibility=hidden \
        -I runtime/  model.cpp  runtime/mlc_runtime.cpp  -o model.so
```

| Flag | Meaning |
|---|---|
| `-O3` | full optimization |
| `-mcpu=native` | use every instruction this machine's CPU has |
| `-shared -fPIC` | produce a library that can be loaded at any address |
| `-fvisibility=hidden` | export only the `mlc_*` API, not every kernel |
| `-pthread` | the runtime uses threads |

Note that `-ffast-math` is **not** used. That flag lets the C++ compiler reorder floating-point arithmetic freely. Without it, clang never changes the order of additions, which keeps results reproducible, and means any reordering is a decision this compiler made visibly in its own IR (section 9.7).

The runtime's source is compiled *together with* the generated code into the same `.so`. The result is self-contained: one file plus the weights file.

## 8.5 The runtime (`runtime/mlc_runtime.{h,cpp}`)

About 280 lines of handwritten C++. It owns resources and drives kernels; it contains no numerical code. Its interface is plain C, so that Python can call it directly:

```c
int  mlc_open(const char* weights_path);   // once; returns 0 on success
void mlc_set_threads(int32_t n);           // optional
void mlc_run(const void* const* inputs, void* const* outputs);
void mlc_run_profiled(inputs, outputs, double* seconds);   // adds each kernel's time
void mlc_close(void);
const char* mlc_last_error(void);
// plus queries: mlc_arena_bytes, mlc_num_kernels, mlc_kernel_op, mlc_input_bytes, ...
```

### `mlc_open`

1. **Map the weights.** `mmap` asks the operating system to make a range of addresses stand for the file's contents. Nothing is read at that moment. Pages are fetched from disk the first time they are touched, the memory is shared with the system's file cache rather than copied, and a weight that is never read never costs anything. The file's size is checked against what the model was compiled for.
2. **Allocate the arena** with an anonymous `mmap` rather than `malloc`. The system hands out zeroed pages lazily, so arena the model never touches costs no physical memory, and the start is page-aligned.
3. **Allocate persistent memory** the same way.
4. **Run the init kernels** (section 7.5), which fill persistent memory from the weights.

### `mlc_run`

A loop over the kernel table, calling each function in order with the three region pointers and the caller's input and output arrays. That is the whole inference "engine": an array of function pointers.

`mlc_run_profiled` does the same and times each call. `mlc exec --profile` uses it to print time per operator, taking each kernel's best time over several runs.

### The thread pool

Section 10.9, after the concepts it needs.

## 8.6 Loading and calling a compiled model

**From C++** (`CompiledModel` in `build.cpp`, used by `mlc exec`, by tests, and by the autotuner): `dlopen` loads the library into the running process, `dlsym` looks up each `mlc_*` function by name, and the functions are then called through pointers.

**From Python** (`bench/worker.py`): the same, via `ctypes`:

```python
lib = ctypes.CDLL("build/release/gen/gpt2_T16/model.so")
lib.mlc_open(b"artifacts/graphs/gpt2_weights.bin")
lib.mlc_set_threads(4)
inputs  = (ctypes.c_void_p * 1)(ids.ctypes.data)       # ids: a NumPy int64 array
outputs = (ctypes.c_void_p * 1)(logits.ctypes.data)    # logits: a NumPy float32 array
lib.mlc_run(inputs, outputs)                           # writes straight into `logits`
```

No data is copied in either direction: the model reads the caller's array and writes the caller's array.

A practical hazard, found in M5: on macOS a library that uses thread-local variables is never really unloaded. Loading a *rebuilt* library from the same file path returns the old code. Anything that builds and loads repeatedly in one process must use a fresh path each time.

## 8.7 The command-line tool (`src/main.cpp`)

| Command | Does |
|---|---|
| `mlc dump <graph.json> [--opt]` | print the graph IR |
| `mlc stats <graph.json> [--opt]` | node counts per operator |
| `mlc opt <graph.json>` | run the passes and report what each did |
| `mlc run <graph.json> --input ... [--expect ...]` | reference interpreter |
| `mlc loops <graph.json>` | print the (scheduled) loop IR |
| `mlc compile <graph.json> -o <dir>` | write `model.cpp`, build `model.so` |
| `mlc exec <model.so> --weights ... --input ... [--expect ...] [--threads N] [--profile]` | load and run a compiled model |
| `mlc gemm <graph.json> -o <dir>` | time each matrix product of the model alone |
| `mlc tune <graph.json> --cache ... -o <dir>` | the autotuner (chapter 11) |

Options shared by the commands that optimize: `--disable name,name,...`, `--tile RxC`, `--block-kb N`, `--parallel-work N`, `--tuning <cache.json>`.

With `--expect`, `run` and `exec` compare the output to golden logits by the project's criterion (section 5.2) and exit with status 1 on a mismatch, which is what makes them usable as tests.

## 8.8 What the first compiled version showed (M2)

The first end-to-end build generated only serial loops. It was correct at every `T` on the first run, and slow:

| T | compiled | graph interpreter | arena |
|---|---:|---:|---:|
| 16 | 0.65 s | 0.41 s | 192 MB |
| 1024 | 38.1 s | 16.3 s | 6453 MB |

**The compiled model was slower than the interpreter it was derived from**, although both ran the same `i, p, j` loop. In the interpreter that loop is handwritten with `A[i][p]` held in a local variable. In generated code every operand was a load through a pointer computed from the arena. clang's vectorizer reported "loop not vectorized: unsafe dependent memory operations" for all 48 dense layers: it could not prove that storing to `C` leaves `A` unchanged, because both were offsets into the same arena. The one product whose output was the caller's buffer *was* vectorized, and ran three times faster per multiply-add.

That observation, that the compiler knows its buffers never overlap and clang does not, set up M4.

## 8.9 Where in the code

| File | Contents |
|---|---|
| `src/codegen/emit_cpp.cpp` | `Emitter`: constants, pointers, serial, vector, and parallel loops, kernel pairs, tables |
| `src/codegen/isa.h` | the names of the SIMD intrinsics (section 10.7) |
| `src/codegen/build.cpp` | `build_shared_library`, `CompiledModel` |
| `runtime/mlc_runtime.h` | the C API and the `mlc_model_desc` struct |
| `runtime/mlc_runtime.cpp` | `mlc_open`, `mlc_run`, the thread pool |
| `runtime/mlc_neon.h` | vector `exp` and `tanh` (section 10.8) |
| `src/main.cpp` | the driver |
| `tests/test_codegen.cpp` | emitted text for small cases; a toy transformer through build, load, run |

## 8.10 Further reading

- `man mmap`, `man dlopen`, and the Python `ctypes` documentation. All three are short.
- Drepper, "How To Write Shared Libraries" (2011). What `-fPIC` and symbol visibility are for.
- Levine, *Linkers and Loaders* (2000), chapters 1 and 10, for what loading a shared library involves.
- The C standard's description of `restrict` (C11 section 6.7.3.1), and the Clang documentation's note that `__restrict` is the C++ spelling.
