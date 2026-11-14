#include "mlc_runtime.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <chrono>
#include <string>

namespace {

char* g_arena = nullptr;
const char* g_weights = nullptr;
size_t g_weights_size = 0;
std::string g_error;

int fail(const std::string& what) {
  g_error = what;
  mlc_close();
  return 1;
}

}  // namespace

int mlc_open(const char* weights_path) {
  mlc_close();
  g_error.clear();
  const mlc_model_desc& m = mlc_gen_model;

  if (m.weights_bytes > 0) {
    if (!weights_path) return fail("this model needs a weights file");
    int fd = ::open(weights_path, O_RDONLY);
    if (fd < 0) return fail(std::string("cannot open ") + weights_path);
    struct stat st {};
    ::fstat(fd, &st);
    if (st.st_size != m.weights_bytes) {
      ::close(fd);
      return fail(std::string(weights_path) + " is not the size this model was compiled for");
    }
    void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
    ::close(fd);
    if (p == MAP_FAILED) return fail(std::string("cannot mmap ") + weights_path);
    g_weights = static_cast<const char*>(p);
    g_weights_size = static_cast<size_t>(st.st_size);
  }

  if (m.arena_bytes > 0) {
    // An anonymous mapping, not malloc: the kernel hands out zero pages
    // lazily, so arena the model never touches costs no physical memory, and
    // the start is page-aligned, which satisfies any alignment the compiler
    // assumed for offsets within it.
    void* p = ::mmap(nullptr, static_cast<size_t>(m.arena_bytes), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
    if (p == MAP_FAILED) return fail("cannot allocate the arena");
    g_arena = static_cast<char*>(p);
  }
  return 0;
}

void mlc_close(void) {
  if (g_arena) ::munmap(g_arena, static_cast<size_t>(mlc_gen_model.arena_bytes));
  if (g_weights) ::munmap(const_cast<char*>(g_weights), g_weights_size);
  g_arena = nullptr;
  g_weights = nullptr;
  g_weights_size = 0;
}

const char* mlc_last_error(void) { return g_error.c_str(); }

void mlc_run(const void* const* inputs, void* const* outputs) {
  const mlc_model_desc& m = mlc_gen_model;
  for (int32_t k = 0; k < m.num_kernels; ++k) m.kernels[k].fn(g_arena, g_weights, inputs, outputs);
}

void mlc_run_profiled(const void* const* inputs, void* const* outputs, double* seconds) {
  const mlc_model_desc& m = mlc_gen_model;
  using clock = std::chrono::steady_clock;
  for (int32_t k = 0; k < m.num_kernels; ++k) {
    clock::time_point t0 = clock::now();
    m.kernels[k].fn(g_arena, g_weights, inputs, outputs);
    seconds[k] += std::chrono::duration<double>(clock::now() - t0).count();
  }
}

int64_t mlc_arena_bytes(void) { return mlc_gen_model.arena_bytes; }
int32_t mlc_num_inputs(void) { return mlc_gen_model.num_inputs; }
int32_t mlc_num_outputs(void) { return mlc_gen_model.num_outputs; }
int64_t mlc_input_bytes(int32_t i) { return mlc_gen_model.input_bytes[i]; }
int64_t mlc_output_bytes(int32_t i) { return mlc_gen_model.output_bytes[i]; }
int32_t mlc_num_kernels(void) { return mlc_gen_model.num_kernels; }
const char* mlc_kernel_op(int32_t k) { return mlc_gen_model.kernels[k].op; }
const char* mlc_kernel_label(int32_t k) { return mlc_gen_model.kernels[k].label; }
