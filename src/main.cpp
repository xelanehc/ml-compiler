// mlc: the compiler driver.
//
//   mlc dump    <graph.json> [--opt]    import and print the graph IR
//   mlc stats   <graph.json> [--opt]    import and print node counts
//   mlc opt     <graph.json>            run the optimization passes and report what each one did
//   mlc run     <graph.json> --input ids.npy [--expect logits.npy] [--tol 1e-3] [--profile] [--opt]
//                                       run the reference interpreter on the graph
//   mlc loops   <graph.json>            lower to the loop IR and print it
//   mlc compile <graph.json> -o <dir> [--flags "-O3 -mcpu=native"] [--emit-only]
//                                       write <dir>/model.cpp and build <dir>/model.so
//   mlc exec    <model.so> --weights w.bin --input ids.npy [--expect logits.npy]
//                                       [--tol 1e-3] [--runs N] [--profile]
//                                       load a compiled model and run it
//
// loops and compile optimize the graph first; dump, stats, and run do so with
// --opt. Wherever the graph is optimized, --disable a,b,... turns passes off
// (simplify, fold, cse, gelu, attention, memory, or all).
// With --expect, run and exec compare against golden logits and exit 1 on mismatch.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include "codegen/codegen.h"
#include "frontend/importer.h"
#include "graph/interpreter.h"
#include "graph/ir.h"
#include "graph/passes.h"
#include "loop/ir.h"
#include "support/files.h"

using namespace mlc;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  mlc dump    <graph.json> [--opt]\n"
               "  mlc stats   <graph.json> [--opt]\n"
               "  mlc opt     <graph.json>\n"
               "  mlc run     <graph.json> --input <ids.npy> [--expect <logits.npy>] [--tol <abs>] [--profile] [--opt]\n"
               "  mlc loops   <graph.json>\n"
               "  mlc compile <graph.json> -o <dir> [--flags \"<compiler flags>\"] [--emit-only]\n"
               "  mlc exec    <model.so> --weights <weights.bin> --input <ids.npy> [--expect <logits.npy>]\n"
               "              [--tol <abs>] [--runs <n>] [--profile]\n"
               "wherever the graph is optimized: --disable <pass,pass,...>   passes: %s\n",
               OptOptions::pass_names());
  return 2;
}

// Command-line options after the positional argument: `--name value` pairs
// and bare `--flag`s, looked up by name.
struct Options {
  std::vector<std::string> args;
  std::vector<bool> used;

  Options(int argc, char** argv) : args(argv, argv + argc), used(static_cast<size_t>(argc), false) {}

  std::string value(const std::string& name, const std::string& fallback = "") {
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] != name) continue;
      if (i + 1 >= args.size()) throw std::runtime_error(name + " needs a value");
      used[i] = used[i + 1] = true;
      return args[i + 1];
    }
    return fallback;
  }

  bool flag(const std::string& name) {
    for (size_t i = 0; i < args.size(); ++i) {
      if (args[i] != name) continue;
      used[i] = true;
      return true;
    }
    return false;
  }

  void done() const {
    for (size_t i = 0; i < args.size(); ++i)
      if (!used[i]) throw std::runtime_error("unknown option " + args[i]);
  }
};

// Reads --disable into a set of pass switches.
OptOptions opt_options(Options& opts) {
  OptOptions o;
  o.disable(opts.value("--disable"));
  return o;
}

// Imports a graph and, if asked, optimizes it, printing one line per pass.
ImportedModel load(const std::string& graph_path, bool optimized, const OptOptions& o, bool report = false) {
  ImportedModel model = import_graph_file(graph_path);
  if (!optimized) return model;
  std::vector<PassStat> stats;
  model.graph = optimize(model.graph, o, &stats);
  if (report) {
    std::printf("%-10s %7s %7s %8s\n", "pass", "before", "after", "removed");
    for (const PassStat& s : stats)
      std::printf("%-10s %7zu %7zu %8zd\n", s.name.c_str(), s.nodes_before, s.nodes_after,
                  static_cast<ssize_t>(s.nodes_before) - static_cast<ssize_t>(s.nodes_after));
  }
  return model;
}

DType dtype_of_npy(const NpyArray& a) {
  if (a.descr == "<f4") return DType::F32;
  if (a.descr == "<i8") return DType::I64;
  return DType::Bool;
}

Tensor tensor_of_npy(const NpyArray& a) { return Tensor::borrow({dtype_of_npy(a), a.shape}, a.data.data()); }

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// The project's correctness criterion (STEERING.md section 6): the largest
// absolute difference in any logit is within `tol`, and at every position the
// most likely next token is the same one.
bool compare_logits(const Tensor& got, const Tensor& want, double tol) {
  if (got.type != want.type) {
    std::printf("FAIL  output is %s, golden is %s\n", type_string(got.type).c_str(), type_string(want.type).c_str());
    return false;
  }
  const float* g = got.data<float>();
  const float* w = want.data<float>();
  const int64_t vocab = got.type.shape.back();
  const int64_t rows = got.size() / vocab;

  double max_diff = 0;
  int64_t non_finite = 0, argmax_mismatches = 0;
  for (int64_t r = 0; r < rows; ++r) {
    int64_t best_g = 0, best_w = 0;
    for (int64_t j = 0; j < vocab; ++j) {
      float a = g[r * vocab + j], b = w[r * vocab + j];
      if (!std::isfinite(a)) ++non_finite;
      max_diff = std::max(max_diff, static_cast<double>(std::fabs(a - b)));
      if (a > g[r * vocab + best_g]) best_g = j;
      if (b > w[r * vocab + best_w]) best_w = j;
    }
    argmax_mismatches += best_g != best_w;
  }

  bool ok = non_finite == 0 && max_diff <= tol && argmax_mismatches == 0;
  std::printf("%s  max|diff| = %.3e (tolerance %.0e)   argmax mismatches = %lld of %lld positions", ok ? "PASS" : "FAIL", max_diff, tol,
              static_cast<long long>(argmax_mismatches), static_cast<long long>(rows));
  if (non_finite) std::printf("   non-finite logits = %lld", static_cast<long long>(non_finite));
  std::printf("\n");
  return ok;
}

// Time per op kind, largest first. Shared by the graph interpreter and compiled models.
void print_op_times(const std::vector<std::pair<std::string, std::pair<int, double>>>& rows, double total) {
  std::printf("\n%-12s %6s %10s %7s\n", "op", "calls", "seconds", "share");
  for (const auto& [op, stat] : rows) std::printf("%-12s %6d %10.4f %6.1f%%\n", op.c_str(), stat.first, stat.second, 100.0 * stat.second / total);
}

int run(const std::string& graph_path, Options opts) {
  std::string input_path = opts.value("--input"), expect_path = opts.value("--expect");
  double tol = std::stod(opts.value("--tol", "1e-3"));
  bool profile = opts.flag("--profile");
  bool optimized = opts.flag("--opt");
  OptOptions o = opt_options(opts);
  opts.done();
  if (input_path.empty()) throw std::runtime_error("run needs --input");

  ImportedModel model = load(graph_path, optimized, o);
  std::filesystem::path weights_path = std::filesystem::path(graph_path).parent_path() / model.weights_file;
  MappedFile weights(weights_path.string());
  if (static_cast<int64_t>(weights.size()) != model.weights_bytes)
    throw std::runtime_error(weights_path.string() + " is not the size graph.json expects; re-run the exporter");

  NpyArray ids = read_npy(input_path);
  std::vector<Tensor> inputs = {tensor_of_npy(ids)};

  RunStats stats;
  auto t0 = std::chrono::steady_clock::now();
  std::vector<Tensor> outputs = run_graph(model.graph, inputs, {weights.data(), weights.size()}, &stats);
  double elapsed = seconds_since(t0);
  std::printf("%s: %zu nodes interpreted in %.2f s -> %s\n", std::filesystem::path(graph_path).filename().c_str(), model.graph.nodes.size(),
              elapsed, type_string(outputs.at(0).type).c_str());
  if (profile) {
    std::vector<std::pair<std::string, std::pair<int, double>>> rows;
    for (int i = 0; i < kNumOps; ++i)
      if (stats.calls[i]) rows.push_back({op_name(static_cast<Op>(i)), {stats.calls[i], stats.seconds[i]}});
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    print_op_times(rows, elapsed);
    std::printf("\nintermediate memory: %.1f MB allocated in total, %.1f MB live at the peak\n", stats.allocated_bytes / 1e6, stats.peak_bytes / 1e6);
  }

  if (expect_path.empty()) return 0;
  NpyArray golden = read_npy(expect_path);
  return compare_logits(outputs.at(0), tensor_of_npy(golden), tol) ? 0 : 1;
}

// graph.json -> graph IR -> loop IR -> model.cpp -> model.so
int compile(const std::string& graph_path, Options opts) {
  std::string out_dir = opts.value("-o");
  codegen::BuildOptions build;
  build.flags = opts.value("--flags", build.flags);
  bool emit_only = opts.flag("--emit-only");
  OptOptions o = opt_options(opts);
  opts.done();
  if (out_dir.empty()) throw std::runtime_error("compile needs -o <dir>");

  ImportedModel model = load(graph_path, true, o);
  loop::Program program = loop::lower_graph(model.graph, model.weights_bytes, {o.memory});
  std::string source = codegen::emit_cpp(program);

  std::filesystem::create_directories(out_dir);
  std::string cpp_path = out_dir + "/model.cpp", so_path = out_dir + "/model.so";
  std::ofstream(cpp_path) << source;
  std::printf("%zu graph nodes -> %zu kernels, %.1f MB arena (%.1f MB without reuse)\n", model.graph.nodes.size(), program.kernels.size(),
              program.arena_bytes / 1e6, program.unplanned_arena_bytes / 1e6);
  std::printf("wrote %s (%lld lines)\n", cpp_path.c_str(), static_cast<long long>(std::count(source.begin(), source.end(), '\n')));
  if (emit_only) return 0;

  auto t0 = std::chrono::steady_clock::now();
  std::string command = codegen::build_shared_library(cpp_path, so_path, build);
  std::printf("built %s in %.1f s\n  %s\n", so_path.c_str(), seconds_since(t0), command.c_str());
  return 0;
}

int exec(const std::string& so_path, Options opts) {
  std::string weights_path = opts.value("--weights"), input_path = opts.value("--input"), expect_path = opts.value("--expect");
  double tol = std::stod(opts.value("--tol", "1e-3"));
  int runs = std::stoi(opts.value("--runs", "1"));
  bool profile = opts.flag("--profile");
  opts.done();
  if (input_path.empty()) throw std::runtime_error("exec needs --input");

  codegen::CompiledModel model(so_path, weights_path);
  NpyArray ids = read_npy(input_path);
  if (model.num_inputs() != 1 || model.num_outputs() != 1) throw std::runtime_error("exec handles models with one input and one output");
  if (static_cast<int64_t>(ids.data.size()) != model.input_bytes(0)) throw std::runtime_error("input is not the size the model was compiled for");

  std::vector<std::byte> out(static_cast<size_t>(model.output_bytes(0)));
  const void* inputs[] = {ids.data.data()};
  void* outputs[] = {out.data()};

  // The first run pays for page faults on the fresh arena and on weights not
  // yet read from disk; it is reported separately from the rest.
  std::vector<double> times;
  for (int r = 0; r < std::max(runs, 1); ++r) {
    auto t0 = std::chrono::steady_clock::now();
    model.run(inputs, outputs);
    times.push_back(seconds_since(t0));
  }
  std::printf("%s: %d kernels, %.1f MB arena, first run %.3f s", std::filesystem::path(so_path).parent_path().filename().c_str(),
              model.num_kernels(), model.arena_bytes() / 1e6, times[0]);
  if (times.size() > 1) {
    std::vector<double> rest(times.begin() + 1, times.end());
    std::sort(rest.begin(), rest.end());
    std::printf(", then min %.3f s, median %.3f s over %zu runs", rest.front(), rest[rest.size() / 2], rest.size());
  }
  std::printf("\n");

  if (profile) {
    std::vector<codegen::CompiledModel::KernelTime> kernels = model.run_profiled(inputs, outputs);
    std::vector<std::pair<std::string, std::pair<int, double>>> rows;
    double total = 0;
    for (const auto& k : kernels) {
      total += k.seconds;
      auto it = std::find_if(rows.begin(), rows.end(), [&](const auto& row) { return row.first == k.op; });
      if (it == rows.end()) it = rows.insert(rows.end(), {k.op, {0, 0.0}});
      ++it->second.first;
      it->second.second += k.seconds;
    }
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    print_op_times(rows, total);
    std::sort(kernels.begin(), kernels.end(), [](const auto& a, const auto& b) { return a.seconds > b.seconds; });
    std::printf("\nslowest kernels:\n");
    for (size_t i = 0; i < std::min<size_t>(kernels.size(), 5); ++i)
      std::printf("  %8.4f s  %-10s %s\n", kernels[i].seconds, kernels[i].op.c_str(), kernels[i].label.c_str());
  }

  if (expect_path.empty()) return 0;
  NpyArray golden = read_npy(expect_path);
  if (golden.data.size() != out.size()) throw std::runtime_error("golden output is not the size the model produces");
  return compare_logits(Tensor::borrow({dtype_of_npy(golden), golden.shape}, out.data()), tensor_of_npy(golden), tol) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  std::string command = argv[1], path = argv[2];
  Options opts(argc - 3, argv + 3);
  try {
    if (command == "dump" || command == "stats") {
      bool optimized = opts.flag("--opt");
      OptOptions o = opt_options(opts);
      opts.done();
      ImportedModel model = load(path, optimized, o);
      if (command == "dump") std::cout << print_graph(model.graph);
      else std::printf("%d ATen nodes -> %s", model.aten_nodes, summarize_graph(model.graph).c_str());
      return 0;
    }
    if (command == "opt") {
      OptOptions o = opt_options(opts);
      opts.done();
      ImportedModel before = import_graph_file(path);
      ImportedModel after = load(path, true, o, /*report=*/true);
      // Memory is planned at lowering, so lower both ways to report it.
      loop::Program planned = loop::lower_graph(after.graph, after.weights_bytes, {o.memory});
      loop::Program naive = loop::lower_graph(before.graph, before.weights_bytes, {false});
      std::printf("\nnodes:   %zu -> %zu\n", before.graph.nodes.size(), after.graph.nodes.size());
      std::printf("kernels: %zu -> %zu\n", naive.kernels.size(), planned.kernels.size());
      std::printf("arena:   %.1f MB -> %.1f MB\n\n", naive.arena_bytes / 1e6, planned.arena_bytes / 1e6);
      std::printf("%s", summarize_graph(after.graph).c_str());
      return 0;
    }
    if (command == "loops") {
      OptOptions o = opt_options(opts);
      opts.done();
      ImportedModel model = load(path, true, o);
      std::cout << loop::print_program(loop::lower_graph(model.graph, model.weights_bytes, {o.memory}));
      return 0;
    }
    if (command == "run") return run(path, opts);
    if (command == "compile") return compile(path, opts);
    if (command == "exec") return exec(path, opts);
    return usage();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlc: %s\n", e.what());
    return 1;
  }
}
