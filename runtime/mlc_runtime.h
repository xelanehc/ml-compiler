// The runtime linked into every compiled model. It owns the resources (the
// mapped weights, the scratch arena, the persistent memory, and the worker
// threads) and drives the kernels; the generated code contains only the
// kernels and a table describing them.
//
// The interface is plain C so that Python can call it through ctypes:
//
//   mlc_open("gpt2_weights.bin");        once; also runs the init kernels
//   mlc_set_threads(4);                  optional
//   mlc_run(inputs, outputs);            as many times as you like
//   mlc_close();
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Filled in by the generated code ----

// Every kernel has this signature. `arena`, `weights`, and `persist` are the
// memory regions the compiler assigned offsets into; `inputs` and `outputs`
// are the caller's tensors (null for init kernels, which run before any call).
typedef void (*mlc_kernel_fn)(char* arena, const char* weights, char* persist, const void* const* inputs, void* const* outputs);

typedef struct {
  const char* op;     // "matmul", "layer_norm", ...
  const char* label;  // "%75 bmm"
  mlc_kernel_fn fn;
} mlc_kernel;

typedef struct {
  int64_t arena_bytes;
  int64_t weights_bytes;
  int64_t persistent_bytes;
  int32_t num_inputs;
  int32_t num_outputs;
  const int64_t* input_bytes;   // size of each input tensor
  const int64_t* output_bytes;  // size of each output tensor
  int32_t num_init_kernels;
  const mlc_kernel* init_kernels;  // run once by mlc_open; they fill the persistent memory
  int32_t num_kernels;
  const mlc_kernel* kernels;    // in execution order
} mlc_model_desc;

extern const mlc_model_desc mlc_gen_model;

// ---- Called by the generated code ----

// Runs fn(context, begin, end) over [0, n), cut into one contiguous range per
// thread, and returns when all of them are done. The calling thread takes a
// range itself. Runs everything on the calling thread if there is one thread,
// or if called from inside another parallel range.
void mlc_parallel_for(int64_t n, void (*fn)(void* context, int64_t begin, int64_t end), void* context);

// ---- Called by the user ----

#define MLC_API __attribute__((visibility("default")))

// Maps the weights file, allocates the arena and the persistent memory, and
// runs the init kernels. Returns 0 on success; on failure returns nonzero and
// mlc_last_error() says why.
MLC_API int mlc_open(const char* weights_path);
MLC_API void mlc_close(void);
MLC_API const char* mlc_last_error(void);

// The number of threads parallel loops use, counting the caller. The default
// is the MLC_THREADS environment variable if set, else the number of
// performance cores. Values below 1 restore the default.
MLC_API void mlc_set_threads(int32_t n);
MLC_API int32_t mlc_threads(void);

// Runs the model. inputs[i] and outputs[i] point to caller-owned memory of
// mlc_input_bytes(i) and mlc_output_bytes(i) bytes. The tensors must not
// overlap one another: the kernels are compiled assuming they do not.
MLC_API void mlc_run(const void* const* inputs, void* const* outputs);

// Same, and adds each kernel's wall time in seconds to seconds[k].
MLC_API void mlc_run_profiled(const void* const* inputs, void* const* outputs, double* seconds);

MLC_API int64_t mlc_arena_bytes(void);
MLC_API int64_t mlc_persistent_bytes(void);
MLC_API int32_t mlc_num_inputs(void);
MLC_API int32_t mlc_num_outputs(void);
MLC_API int64_t mlc_input_bytes(int32_t i);
MLC_API int64_t mlc_output_bytes(int32_t i);
MLC_API int32_t mlc_num_kernels(void);
MLC_API const char* mlc_kernel_op(int32_t k);
MLC_API const char* mlc_kernel_label(int32_t k);

#ifdef __cplusplus
}
#endif
