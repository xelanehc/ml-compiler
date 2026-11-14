// Loop IR -> C++ source -> shared library, and a loader for the result.
#pragma once

#include <string>
#include <vector>

#include "loop/ir.h"

namespace mlc::codegen {

struct CodegenError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// The C++17 source for a program: one function per kernel plus the table the
// runtime reads. Scalar code only; no intrinsics, no threads.
std::string emit_cpp(const loop::Program& p);

struct BuildOptions {
  std::string compiler = "clang++";  // overridden by the CXX environment variable
  std::string flags = "-O3 -mcpu=native";
  std::string runtime_dir;           // directory holding mlc_runtime.{h,cpp}; defaults to the one mlc was built from
};

// Compiles `cpp_path` together with the runtime into the shared library
// `so_path`. Returns the command line that was run. Throws CodegenError with
// the compiler's output if the build fails.
std::string build_shared_library(const std::string& cpp_path, const std::string& so_path, const BuildOptions& options = {});

// A compiled model loaded into this process with dlopen.
class CompiledModel {
 public:
  // `weights_path` may be empty for a model with no external weights.
  CompiledModel(const std::string& so_path, const std::string& weights_path);
  ~CompiledModel();
  CompiledModel(const CompiledModel&) = delete;
  CompiledModel& operator=(const CompiledModel&) = delete;

  void run(const void* const* inputs, void* const* outputs) const { run_(inputs, outputs); }

  struct KernelTime {
    std::string op, label;
    double seconds;
  };
  // Runs once and reports the time spent in each kernel.
  std::vector<KernelTime> run_profiled(const void* const* inputs, void* const* outputs) const;

  int64_t arena_bytes() const;
  int64_t input_bytes(int i) const;
  int64_t output_bytes(int i) const;
  int num_inputs() const;
  int num_outputs() const;
  int num_kernels() const;

 private:
  void* handle_ = nullptr;
  void (*run_)(const void* const*, void* const*) = nullptr;
  void* symbol(const char* name) const;
};

}  // namespace mlc::codegen
