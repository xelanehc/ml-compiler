// The importer on small hand-written graph.json documents. The full GPT-2
// export is covered by the golden_T* tests.
#include <limits>

#include "check.h"
#include "frontend/importer.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;

namespace {

// Wraps a list of nodes in the rest of a graph.json. The single input is
// `x: float32[2, 3]`; there is one parameter `w: float32[3]` at offset 0.
ImportedModel import_nodes(const std::string& nodes, const std::string& output) {
  std::string doc = R"({"format": 1, "producer": "test", "seq_len": 0, "weights_file": "w.bin", "weights_bytes": 12,
    "inputs": [{"name": "x", "dtype": "float32", "shape": [2, 3]}],
    "params": [{"name": "w", "dtype": "float32", "shape": [3], "offset": 0, "nbytes": 12}],
    "nodes": [)" + nodes + R"(], "outputs": [")" + output + R"("]})";
  return import_graph(json::parse(doc));
}

std::vector<float> run_on(const ImportedModel& m, const std::vector<float>& x, const std::vector<float>& w = {0, 0, 0}) {
  Tensor in = Tensor::borrow({DType::F32, {2, 3}}, reinterpret_cast<const std::byte*>(x.data()));
  Tensor out = run_graph(m.graph, {&in, 1}, {reinterpret_cast<const std::byte*>(w.data()), w.size() * 4}).at(0);
  return std::vector<float>(out.data<float>(), out.data<float>() + out.size());
}

}  // namespace

TEST(import_scalar_operand_takes_the_tensor_dtype) {
  ImportedModel m = import_nodes(
      R"({"name": "y", "op": "aten.mul.Tensor", "args": [{"ref": "x"}, 0.5], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})", "y");
  CHECK(m.graph.nodes.back().op == Op::Mul);
  CHECK(run_on(m, {2, 4, 6, 8, 10, 12}) == (std::vector<float>{1, 2, 3, 4, 5, 6}));
}

TEST(import_view_resolves_minus_one_and_param_is_lazy) {
  ImportedModel m = import_nodes(
      R"({"name": "v", "op": "aten.view.default", "args": [{"ref": "x"}, [-1, 2]], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [3, 2]}]})", "v");
  CHECK(m.graph.type_of(m.graph.outputs[0]).shape == (Shape{3, 2}));
  CHECK(m.graph.constants.empty());  // `w` was never referenced, so no Const node exists for it
}

TEST(import_param_and_tuple_outputs) {
  // layer_norm returns a 3-tuple; getitem 0 picks the tensor we compute.
  ImportedModel m = import_nodes(
      R"({"name": "ln", "op": "aten.native_layer_norm.default", "args": [{"ref": "x"}, [3], {"ref": "w"}, {"ref": "w"}, 1e-05], "kwargs": {},
          "outputs": [{"dtype": "float32", "shape": [2, 3]}, {"dtype": "float32", "shape": [2, 1]}, {"dtype": "float32", "shape": [2, 1]}]},
         {"name": "g", "op": "getitem", "args": [{"ref": "ln"}, 0], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})",
      "g");
  CHECK(m.graph.nodes.back().op == Op::LayerNorm);
  CHECK(m.graph.constants.size() == 1 && m.graph.constants[0].is_external());
  // Asking for the mean (tuple slot 1) must fail loudly.
  CHECK_THROWS(import_nodes(
      R"({"name": "ln", "op": "aten.native_layer_norm.default", "args": [{"ref": "x"}, [3], {"ref": "w"}, {"ref": "w"}, 1e-05], "kwargs": {},
          "outputs": [{"dtype": "float32", "shape": [2, 3]}, {"dtype": "float32", "shape": [2, 1]}, {"dtype": "float32", "shape": [2, 1]}]},
         {"name": "g", "op": "getitem", "args": [{"ref": "ln"}, 1], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 1]}]})",
      "g"), ImportError);
}

TEST(import_split_becomes_slices) {
  ImportedModel m = import_nodes(
      R"({"name": "s", "op": "aten.split_with_sizes.default", "args": [{"ref": "x"}, [1, 2], 1], "kwargs": {},
          "outputs": [{"dtype": "float32", "shape": [2, 1]}, {"dtype": "float32", "shape": [2, 2]}]},
         {"name": "b", "op": "getitem", "args": [{"ref": "s"}, 1], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 2]}]})",
      "b");
  CHECK(run_on(m, {1, 2, 3, 4, 5, 6}) == (std::vector<float>{2, 3, 5, 6}));
}

TEST(import_slice_with_python_bounds) {
  // x[:, -2:9223372036854775807]: negative start, end far past the axis.
  ImportedModel m = import_nodes(
      R"({"name": "s", "op": "aten.slice.Tensor", "args": [{"ref": "x"}, 1, -2, 9223372036854775807], "kwargs": {},
          "outputs": [{"dtype": "float32", "shape": [2, 2]}]})", "s");
  CHECK(run_on(m, {1, 2, 3, 4, 5, 6}) == (std::vector<float>{2, 3, 5, 6}));
}

TEST(import_infinity_and_full_like) {
  // where(x == -inf, full_like(x, 0), x)
  ImportedModel m = import_nodes(
      R"({"name": "e", "op": "aten.eq.Scalar", "args": [{"ref": "x"}, {"float": "-inf"}], "kwargs": {}, "outputs": [{"dtype": "bool", "shape": [2, 3]}]},
         {"name": "z", "op": "aten.full_like.default", "args": [{"ref": "x"}, 0], "kwargs": {"pin_memory": false}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]},
         {"name": "w2", "op": "aten.where.self", "args": [{"ref": "e"}, {"ref": "z"}, {"ref": "x"}], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})",
      "w2");
  float inf = std::numeric_limits<float>::infinity();
  CHECK(run_on(m, {1, -inf, 3, -inf, 5, 6}) == (std::vector<float>{1, 0, 3, 0, 5, 6}));
}

TEST(import_clone_makes_no_node) {
  ImportedModel m = import_nodes(
      R"({"name": "c", "op": "aten.clone.default", "args": [{"ref": "x"}], "kwargs": {"memory_format": {"torch": "torch.contiguous_format"}},
          "outputs": [{"dtype": "float32", "shape": [2, 3]}]})", "c");
  CHECK(m.graph.nodes.size() == 1);            // just the input
  CHECK(m.graph.outputs[0] == m.graph.inputs[0]);
}

TEST(import_rejects_what_it_does_not_understand) {
  // An operator with no lowering.
  CHECK_THROWS(import_nodes(
      R"({"name": "y", "op": "aten.erf.default", "args": [{"ref": "x"}], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})", "y"), ImportError);
  // A known operator with an argument the lowering ignores (add's alpha).
  CHECK_THROWS(import_nodes(
      R"({"name": "y", "op": "aten.add.Tensor", "args": [{"ref": "x"}, {"ref": "x"}], "kwargs": {"alpha": 2}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})", "y"), ImportError);
  // Our inferred shape disagrees with what the export recorded.
  CHECK_THROWS(import_nodes(
      R"({"name": "y", "op": "aten.tanh.default", "args": [{"ref": "x"}], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [3, 2]}]})", "y"), ImportError);
  // A reference to a value that was never defined.
  CHECK_THROWS(import_nodes(
      R"({"name": "y", "op": "aten.tanh.default", "args": [{"ref": "nope"}], "kwargs": {}, "outputs": [{"dtype": "float32", "shape": [2, 3]}]})", "y"), ImportError);
}
