// mlc: the compiler driver.
//
//   mlc dump  <graph.json>     import and print the graph IR
//   mlc stats <graph.json>     import and print node counts
//   mlc run   <graph.json> --input ids.npy [--expect logits.npy] [--tol 1e-3] [--profile]
//                              run the reference interpreter; with --expect,
//                              compare against golden logits and exit 1 on mismatch
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "frontend/importer.h"
#include "graph/interpreter.h"
#include "graph/ir.h"
#include "support/files.h"

using namespace mlc;

namespace {

int usage() {
  std::fprintf(stderr,
               "usage:\n"
               "  mlc dump  <graph.json>\n"
               "  mlc stats <graph.json>\n"
               "  mlc run   <graph.json> --input <ids.npy> [--expect <logits.npy>] [--tol <abs>] [--profile]\n");
  return 2;
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

void print_profile(const RunStats& s, double total) {
  std::printf("\n%-12s %6s %10s %7s\n", "op", "calls", "seconds", "share");
  std::vector<int> order;
  for (int i = 0; i < kNumOps; ++i)
    if (s.calls[i]) order.push_back(i);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return s.seconds[a] > s.seconds[b]; });
  for (int i : order)
    std::printf("%-12s %6d %10.4f %6.1f%%\n", op_name(static_cast<Op>(i)), s.calls[i], s.seconds[i], 100.0 * s.seconds[i] / total);
  std::printf("\nintermediate memory: %.1f MB allocated in total, %.1f MB live at the peak\n", s.allocated_bytes / 1e6, s.peak_bytes / 1e6);
}

int run(const std::string& graph_path, int argc, char** argv) {
  std::string input_path, expect_path;
  double tol = 1e-3;
  bool profile = false;
  for (int i = 0; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (a == "--input") input_path = next();
    else if (a == "--expect") expect_path = next();
    else if (a == "--tol") tol = std::stod(next());
    else if (a == "--profile") profile = true;
    else throw std::runtime_error("unknown option " + a);
  }
  if (input_path.empty()) throw std::runtime_error("run needs --input");

  ImportedModel model = import_graph_file(graph_path);
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
  if (profile) print_profile(stats, elapsed);

  if (expect_path.empty()) return 0;
  NpyArray golden = read_npy(expect_path);
  return compare_logits(outputs.at(0), tensor_of_npy(golden), tol) ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  std::string command = argv[1], graph_path = argv[2];
  try {
    if (command == "dump") {
      std::cout << print_graph(import_graph_file(graph_path).graph);
      return 0;
    }
    if (command == "stats") {
      ImportedModel model = import_graph_file(graph_path);
      std::printf("%d ATen nodes lowered to %s", model.aten_nodes, summarize_graph(model.graph).c_str());
      return 0;
    }
    if (command == "run") return run(graph_path, argc - 3, argv + 3);
    return usage();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "mlc: %s\n", e.what());
    return 1;
  }
}
