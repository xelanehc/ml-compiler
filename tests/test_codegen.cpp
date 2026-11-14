// Codegen: emitted text for a small case, and a full build-load-run of a
// model through the real C++ compiler, compared with the graph interpreter.
#include <unistd.h>

#include <filesystem>
#include <fstream>

#include "check.h"
#include "codegen/codegen.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;

namespace {

std::string scratch_dir(const char* name) {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / ("mlc_test_" + std::to_string(::getpid()) + "_" + name);
  std::filesystem::create_directories(dir);
  return dir.string();
}

}  // namespace

TEST(codegen_emits_flat_indexing_and_literals) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {2, 3}}, "x");
  ValueId mask = g.add(Op::Where, {g.add(Op::Eq, {x, g.add_scalar(DType::F32, 0.1)}), g.add_scalar(DType::F32, -INFINITY), x});
  g.outputs = {g.add(Op::Permute, {mask}, dims({1, 0}))};
  std::string cpp = codegen::emit_cpp(loop::lower_graph(g, 0));
  CHECK(cpp.find("(const float*)inputs[0]") != std::string::npos);
  CHECK(cpp.find("(float*)outputs[0]") != std::string::npos);
  CHECK(cpp.find("for (int64_t i0 = 0; i0 < 2; ++i0) {") != std::string::npos);
  CHECK(cpp.find("0.100000001f") != std::string::npos);       // the f32 nearest to 0.1, spelled so it reads back exactly
  CHECK(cpp.find("(-INFINITY)") != std::string::npos);
  CHECK(cpp.find("v5[((i0 * 2) + i1)] = v4[((i1 * 3) + i0)];") != std::string::npos);  // permute: indices swapped on the read
  CHECK(cpp.find("mlc_gen_model") != std::string::npos);
}

TEST(codegen_build_reports_compiler_errors) {
  std::string dir = scratch_dir("bad");
  std::ofstream(dir + "/model.cpp") << "this is not C++\n";
  CHECK_THROWS(codegen::build_shared_library(dir + "/model.cpp", dir + "/model.so"), codegen::CodegenError);
  std::filesystem::remove_all(dir);
}

TEST(codegen_compiled_model_matches_graph_interpreter) {
  // The toy transformer with token ids as a real input, through every stage:
  // graph -> loops -> C++ -> clang++ -> dlopen -> run.
  Graph g;
  ValueId ids = g.add_input({DType::I64, {1, 5}}, "ids");
  g.outputs = {tiny_transformer(g, ids)};
  loop::Program program = loop::lower_graph(g, 0);

  std::string dir = scratch_dir("tiny");
  std::ofstream(dir + "/model.cpp") << codegen::emit_cpp(program);
  codegen::build_shared_library(dir + "/model.cpp", dir + "/model.so");
  codegen::CompiledModel model(dir + "/model.so", "");
  CHECK(model.num_inputs() == 1 && model.num_outputs() == 1);
  CHECK(model.input_bytes(0) == 5 * 8);
  CHECK(model.output_bytes(0) == 5 * 11 * 4);
  CHECK(model.num_kernels() == static_cast<int>(program.kernels.size()));

  for (std::vector<int64_t> tokens : {std::vector<int64_t>{3, 10, 0, 7, 3}, std::vector<int64_t>{0, 0, 0, 0, 0}}) {
    Tensor in = Tensor::borrow({DType::I64, {1, 5}}, reinterpret_cast<const std::byte*>(tokens.data()));
    Tensor want = run_graph(g, {&in, 1}, {}).at(0);

    std::vector<std::byte> got(want.type.byte_size());
    const void* inputs[] = {tokens.data()};
    void* outputs[] = {got.data()};
    model.run(inputs, outputs);
    CHECK(same_values(want.type, want.ptr, got.data(), 1e-4));

    // The loop interpreter must agree with both.
    std::vector<std::byte> via_loops(want.type.byte_size());
    void* loop_outputs[] = {via_loops.data()};
    loop::run_program(program, {}, inputs, loop_outputs);
    CHECK(same_values(want.type, want.ptr, via_loops.data(), 1e-4));
  }

  // Profiling runs the same kernels and accounts for each of them.
  std::vector<int64_t> tokens = {1, 2, 3, 4, 5};
  std::vector<std::byte> out(static_cast<size_t>(model.output_bytes(0)));
  const void* inputs[] = {tokens.data()};
  void* outputs[] = {out.data()};
  CHECK(model.run_profiled(inputs, outputs).size() == program.kernels.size());
  std::filesystem::remove_all(dir);
}
