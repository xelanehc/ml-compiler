// Runs the C++ compiler on generated code, and loads the result.
#include <dlfcn.h>

#include <cstdlib>
#include <fstream>
#include <sstream>

#include "codegen/codegen.h"

#ifndef MLC_RUNTIME_DIR
#define MLC_RUNTIME_DIR "runtime"
#endif

namespace mlc::codegen {

namespace {

// Single-quotes a path for the shell.
std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) out += c == '\'' ? std::string("'\\''") : std::string(1, c);
  return out + "'";
}

}  // namespace

std::string build_shared_library(const std::string& cpp_path, const std::string& so_path, const BuildOptions& options) {
  const char* env_cxx = std::getenv("CXX");
  std::string compiler = env_cxx && *env_cxx ? env_cxx : options.compiler;
  std::string runtime = options.runtime_dir.empty() ? MLC_RUNTIME_DIR : options.runtime_dir;
  std::string log = so_path + ".log";

  // -shared -fPIC           build a library that can be loaded at any address
  // -fvisibility=hidden     export only the mlc_* API, not every kernel
  // -std=c++17              the generated code's language level (STEERING.md section 3)
  // -pthread               the runtime's thread pool
  std::string command = compiler + " -std=c++17 " + options.flags + " -pthread -shared -fPIC -fvisibility=hidden -I" + shell_quote(runtime) + " " +
                        shell_quote(cpp_path) + " " + shell_quote(runtime + "/mlc_runtime.cpp") + " -o " + shell_quote(so_path);
  if (std::system((command + " > " + shell_quote(log) + " 2>&1").c_str()) != 0) {
    std::ifstream in(log);
    std::ostringstream text;
    text << in.rdbuf();
    throw CodegenError("compiling generated code failed:\n  " + command + "\n" + text.str().substr(0, 4000));
  }
  return command;
}

void* CompiledModel::symbol(const char* name) const {
  void* s = ::dlsym(handle_, name);
  if (!s) throw CodegenError(std::string("compiled model has no symbol ") + name);
  return s;
}

CompiledModel::CompiledModel(const std::string& so_path, const std::string& weights_path) {
  // dlopen searches system directories for a bare file name, so make the path explicit.
  std::string path = so_path.find('/') == std::string::npos ? "./" + so_path : so_path;
  handle_ = ::dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle_) throw CodegenError(std::string("cannot load ") + so_path + ": " + ::dlerror());
  auto open = reinterpret_cast<int (*)(const char*)>(symbol("mlc_open"));
  if (open(weights_path.empty() ? nullptr : weights_path.c_str()) != 0) {
    std::string why = reinterpret_cast<const char* (*)()>(symbol("mlc_last_error"))();
    ::dlclose(handle_);
    throw CodegenError("cannot open compiled model: " + why);
  }
  run_ = reinterpret_cast<void (*)(const void* const*, void* const*)>(symbol("mlc_run"));
}

CompiledModel::~CompiledModel() {
  if (!handle_) return;
  reinterpret_cast<void (*)()>(symbol("mlc_close"))();
  ::dlclose(handle_);
}

std::vector<CompiledModel::KernelTime> CompiledModel::run_profiled(const void* const* inputs, void* const* outputs) const {
  int n = num_kernels();
  std::vector<double> seconds(static_cast<size_t>(n), 0.0);
  reinterpret_cast<void (*)(const void* const*, void* const*, double*)>(symbol("mlc_run_profiled"))(inputs, outputs, seconds.data());
  auto op = reinterpret_cast<const char* (*)(int32_t)>(symbol("mlc_kernel_op"));
  auto label = reinterpret_cast<const char* (*)(int32_t)>(symbol("mlc_kernel_label"));
  std::vector<KernelTime> out;
  for (int k = 0; k < n; ++k) out.push_back({op(k), label(k), seconds[static_cast<size_t>(k)]});
  return out;
}

int64_t CompiledModel::arena_bytes() const { return reinterpret_cast<int64_t (*)()>(symbol("mlc_arena_bytes"))(); }
int64_t CompiledModel::persistent_bytes() const { return reinterpret_cast<int64_t (*)()>(symbol("mlc_persistent_bytes"))(); }
void CompiledModel::set_threads(int n) const { reinterpret_cast<void (*)(int32_t)>(symbol("mlc_set_threads"))(n); }
int CompiledModel::threads() const { return reinterpret_cast<int32_t (*)()>(symbol("mlc_threads"))(); }
int64_t CompiledModel::input_bytes(int i) const { return reinterpret_cast<int64_t (*)(int32_t)>(symbol("mlc_input_bytes"))(i); }
int64_t CompiledModel::output_bytes(int i) const { return reinterpret_cast<int64_t (*)(int32_t)>(symbol("mlc_output_bytes"))(i); }
int CompiledModel::num_inputs() const { return reinterpret_cast<int32_t (*)()>(symbol("mlc_num_inputs"))(); }
int CompiledModel::num_outputs() const { return reinterpret_cast<int32_t (*)()>(symbol("mlc_num_outputs"))(); }
int CompiledModel::num_kernels() const { return reinterpret_cast<int32_t (*)()>(symbol("mlc_num_kernels"))(); }

}  // namespace mlc::codegen
