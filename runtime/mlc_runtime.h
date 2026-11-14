// The runtime linked into every compiled model. It owns the resources (the
// mapped weights and the scratch arena) and drives the kernels; the generated
// code contains only the kernels and a table describing them.
//
// The interface is plain C so that Python can call it through ctypes:
//
//   mlc_open("gpt2_weights.bin");        once
//   mlc_run(inputs, outputs);            as many times as you like
//   mlc_close();
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// ---- Filled in by the generated code ----

// Every kernel has this signature. `arena` and `weights` are the two memory
// regions the compiler assigned offsets into; `inputs` and `outputs` are the
// caller's tensors.
typedef void (*mlc_kernel_fn)(char* arena, const char* weights, const void* const* inputs, void* const* outputs);

typedef struct {
  const char* op;     // "matmul", "layer_norm", ...
  const char* label;  // "%75 bmm"
  mlc_kernel_fn fn;
} mlc_kernel;

typedef struct {
  int64_t arena_bytes;
  int64_t weights_bytes;
  int32_t num_inputs;
  int32_t num_outputs;
  const int64_t* input_bytes;   // size of each input tensor
  const int64_t* output_bytes;  // size of each output tensor
  int32_t num_kernels;
  const mlc_kernel* kernels;    // in execution order
} mlc_model_desc;

extern const mlc_model_desc mlc_gen_model;

// ---- Called by the user ----

#define MLC_API __attribute__((visibility("default")))

// Maps the weights file and allocates the arena. Returns 0 on success; on
// failure returns nonzero and mlc_last_error() says why.
MLC_API int mlc_open(const char* weights_path);
MLC_API void mlc_close(void);
MLC_API const char* mlc_last_error(void);

// Runs the model. inputs[i] and outputs[i] point to caller-owned memory of
// mlc_input_bytes(i) and mlc_output_bytes(i) bytes.
MLC_API void mlc_run(const void* const* inputs, void* const* outputs);

// Same, and adds each kernel's wall time in seconds to seconds[k].
MLC_API void mlc_run_profiled(const void* const* inputs, void* const* outputs, double* seconds);

MLC_API int64_t mlc_arena_bytes(void);
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
