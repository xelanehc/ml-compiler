#include "check.h"
#include "graph/ir.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;

namespace {
Type f(Shape s) { return {DType::F32, std::move(s)}; }
Type i(Shape s) { return {DType::I64, std::move(s)}; }
Type b(Shape s) { return {DType::Bool, std::move(s)}; }

Type infer(Op op, std::vector<Type> in, Attrs a = {}) { return infer_type(op, in, a); }
}  // namespace

TEST(broadcast_shapes_rules) {
  CHECK(broadcast_shapes({1, 12, 16, 16}, {1, 1, 16, 16}) == (Shape{1, 12, 16, 16}));
  CHECK(broadcast_shapes({16, 1}, {16}) == (Shape{16, 16}));
  CHECK(broadcast_shapes({}, {3, 4}) == (Shape{3, 4}));  // a scalar broadcasts with anything
  CHECK(broadcast_shapes({1, 1, 1, 16}, {1, 1, 16, 1}) == (Shape{1, 1, 16, 16}));
  CHECK_THROWS(broadcast_shapes({3}, {4}), GraphError);
}

TEST(infer_elementwise) {
  CHECK(infer(Op::Add, {f({2, 3}), f({3})}) == f({2, 3}));
  CHECK(infer(Op::Mul, {i({4}), i({})}) == i({4}));
  CHECK(infer(Op::Le, {i({1, 1, 1, 4}), i({1, 1, 4, 1})}) == b({1, 1, 4, 4}));
  CHECK(infer(Op::Where, {b({1, 1, 4, 4}), f({}), f({})}) == f({1, 1, 4, 4}));
  CHECK_THROWS(infer(Op::Add, {f({2}), i({2})}), GraphError);   // mixed dtypes
  CHECK_THROWS(infer(Op::Add, {b({2}), b({2})}), GraphError);   // arithmetic on bool
  CHECK_THROWS(infer(Op::And, {f({2}), f({2})}), GraphError);
  CHECK_THROWS(infer(Op::Where, {f({2}), f({2}), f({2})}), GraphError);
  CHECK_THROWS(infer(Op::Tanh, {f({2}), f({2})}), GraphError);  // arity
}

TEST(infer_layout) {
  CHECK(infer(Op::Reshape, {f({1, 16, 768})}, dims({1, 16, 12, 64})) == f({1, 16, 12, 64}));
  CHECK_THROWS(infer(Op::Reshape, {f({2, 3})}, dims({4})), GraphError);

  CHECK(infer(Op::Permute, {f({1, 16, 12, 64})}, dims({0, 2, 1, 3})) == f({1, 12, 16, 64}));
  CHECK_THROWS(infer(Op::Permute, {f({2, 3})}, dims({0, 0})), GraphError);
  CHECK_THROWS(infer(Op::Permute, {f({2, 3})}, dims({0})), GraphError);

  CHECK(infer(Op::Broadcast, {f({})}, dims({2, 3})) == f({2, 3}));
  CHECK(infer(Op::Broadcast, {f({1, 3})}, dims({2, 3})) == f({2, 3}));
  CHECK_THROWS(infer(Op::Broadcast, {f({2, 3})}, dims({1, 3})), GraphError);  // cannot shrink

  Attrs s = axis(2);
  s.start = 768;
  s.end = 1536;
  CHECK(infer(Op::Slice, {f({1, 16, 2304})}, s) == f({1, 16, 768}));
  s.end = 9999;
  CHECK_THROWS(infer(Op::Slice, {f({1, 16, 2304})}, s), GraphError);

  CHECK(infer(Op::Concat, {i({1, 1}), i({1, 16})}, axis(1)) == i({1, 17}));
  CHECK_THROWS(infer(Op::Concat, {i({1, 1}), i({2, 16})}, axis(1)), GraphError);
}

TEST(infer_gather) {
  // Embedding lookup: one index tensor, the table's row axis is replaced by the index shape.
  CHECK(infer(Op::Gather, {f({50257, 768}), i({1, 16})}) == f({1, 16, 768}));
  // Two index tensors that broadcast against each other.
  CHECK(infer(Op::Gather, {i({1, 16}), i({1, 1, 1, 1}), i({1, 1, 16, 1})}) == i({1, 1, 16, 1}));
  CHECK_THROWS(infer(Op::Gather, {f({4, 4}), f({2})}), GraphError);            // float index
  CHECK_THROWS(infer(Op::Gather, {f({4}), i({2}), i({2})}), GraphError);       // more indices than axes
}

TEST(infer_linear_algebra) {
  CHECK(infer(Op::MatMul, {f({16, 768}), f({768, 50257})}) == f({16, 50257}));
  CHECK(infer(Op::MatMul, {f({12, 16, 64}), f({12, 64, 16})}) == f({12, 16, 16}));
  CHECK_THROWS(infer(Op::MatMul, {f({16, 768}), f({767, 3})}), GraphError);
  CHECK_THROWS(infer(Op::MatMul, {f({12, 16, 64}), f({11, 64, 16})}), GraphError);  // batch mismatch
  CHECK_THROWS(infer(Op::MatMul, {f({768}), f({768})}), GraphError);                // rank 1

  CHECK(infer(Op::Gemm, {f({16, 768}), f({768, 2304}), f({2304})}) == f({16, 2304}));
  CHECK_THROWS(infer(Op::Gemm, {f({16, 768}), f({768, 2304}), f({768})}), GraphError);

  CHECK(infer(Op::LayerNorm, {f({1, 16, 768}), f({768}), f({768})}) == f({1, 16, 768}));
  CHECK_THROWS(infer(Op::LayerNorm, {f({1, 16, 768}), f({16}), f({768})}), GraphError);
}

TEST(infer_reductions) {
  Attrs a = axis(3);
  CHECK(infer(Op::ReduceAny, {b({1, 12, 16, 16})}, a) == b({1, 12, 16}));
  a.keepdims = true;
  CHECK(infer(Op::ReduceAny, {b({1, 12, 16, 16})}, a) == b({1, 12, 16, 1}));
  CHECK_THROWS(infer(Op::CumSum, {b({4})}, axis(0)), GraphError);
  CHECK_THROWS(infer(Op::Softmax, {f({4})}, axis(1)), GraphError);
}

TEST(builder_rejects_ill_typed_nodes) {
  Graph g;
  ValueId x = g.add_input(f({2, 3}), "x");
  ValueId y = g.add_input(f({4}), "y");
  CHECK_THROWS(g.add(Op::Add, {x, y}), GraphError);
  CHECK_THROWS(g.add(Op::Tanh, {ValueId{99}}), GraphError);  // refers to a value that does not exist
  CHECK(g.nodes.size() == 2);                                // a failed add leaves the graph untouched
}

TEST(verifier_accepts_a_valid_graph) {
  Graph g;
  ValueId x = g.add_input(f({2, 3}), "x");
  ValueId h = g.add(Op::Tanh, {x});
  g.outputs = {g.add(Op::Add, {h, g.add_scalar(DType::F32, 1.0)})};
  CHECK(verify_graph(g).empty());
}

TEST(verifier_catches_corruption) {
  // Build a valid graph, then damage it the way a buggy pass might.
  auto make = [] {
    Graph g;
    ValueId x = g.add_input(f({2, 3}), "x");
    ValueId h = g.add(Op::Tanh, {x});
    g.outputs = {g.add(Op::Add, {h, x})};
    return g;
  };

  Graph forward = make();
  forward.nodes[1].inputs[0] = 2;  // tanh now reads a value defined after it
  CHECK(verify_graph(forward).size() == 1);

  Graph stale = make();
  stale.nodes[2].type.shape = {3, 2};  // stored type no longer follows from the inputs
  CHECK(verify_graph(stale).size() == 1);

  Graph wrong_dtype = make();
  wrong_dtype.nodes[0].type.dtype = DType::I64;  // tanh of an int: two nodes downstream are now ill-typed
  CHECK(!verify_graph(wrong_dtype).empty());

  Graph no_out = make();
  no_out.outputs = {7};
  CHECK(verify_graph(no_out).size() == 1);
  CHECK_THROWS(check_graph(no_out), GraphError);
}

TEST(printer_shows_ops_types_and_attrs) {
  Graph g;
  ValueId x = g.add_input(f({1, 16, 768}), "x");
  ValueId r = g.add(Op::Reshape, {x}, dims({1, 16, 12, 64}), "view_3");
  g.outputs = {g.add(Op::Mul, {r, g.add_scalar(DType::F32, 0.5)})};
  std::string text = print_graph(g);
  CHECK(text.find("%1: f32[1,16,12,64] = reshape(%0, dims=[1,16,12,64])") != std::string::npos);
  CHECK(text.find("# view_3") != std::string::npos);
  CHECK(text.find("const(0.5)") != std::string::npos);
  CHECK(text.find("return %3") != std::string::npos);
}
