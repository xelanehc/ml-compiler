// Each op against values worked out by hand (or by a formula simple enough to
// trust). The real model is the end-to-end test; these pin down which op is
// wrong when that one fails.
#include <cmath>
#include <limits>

#include "check.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;

namespace {
template <class T>
bool same(const std::vector<T>& a, const std::vector<T>& b) { return a == b; }

bool close(const std::vector<float>& a, const std::vector<float>& b, double tol = 1e-6) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!(std::fabs(a[i] - b[i]) <= tol)) return false;
  return true;
}
}  // namespace

TEST(interp_arithmetic_with_broadcasting) {
  Graph g;
  ValueId a = f32(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  ValueId row = f32(g, {3}, {10, 20, 30});
  ValueId col = f32(g, {2, 1}, {100, 200});
  CHECK(same(eval<float>(g, g.add(Op::Add, {a, row})), {11, 22, 33, 14, 25, 36}));
  CHECK(same(eval<float>(g, g.add(Op::Sub, {a, col})), {-99, -98, -97, -196, -195, -194}));
  CHECK(same(eval<float>(g, g.add(Op::Mul, {a, g.add_scalar(DType::F32, 0.5)})), {0.5, 1, 1.5, 2, 2.5, 3}));
  // Both operands broadcast: a column against a row gives the outer sum.
  CHECK(same(eval<float>(g, g.add(Op::Add, {col, row})), {110, 120, 130, 210, 220, 230}));
}

TEST(interp_int_arithmetic_and_comparisons) {
  Graph g;
  ValueId a = i64(g, {4}, {0, 1, 2, 3});
  CHECK(same(eval<int64_t>(g, g.add(Op::Sub, {a, g.add_scalar(DType::I64, 1)})), {-1, 0, 1, 2}));
  // The causal mask: column index <= row index.
  ValueId rows = g.add(Op::Reshape, {a}, dims({4, 1}));
  ValueId cols = g.add(Op::Reshape, {a}, dims({1, 4}));
  CHECK(same(eval<uint8_t>(g, g.add(Op::Le, {cols, rows})), {1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1}));
  CHECK(same(eval<uint8_t>(g, g.add(Op::Eq, {a, g.add_scalar(DType::I64, 2)})), {0, 0, 1, 0}));
  CHECK(same(eval<uint8_t>(g, g.add(Op::Ne, {a, g.add_scalar(DType::I64, 2)})), {1, 1, 0, 1}));
}

TEST(interp_float_comparison_with_infinity) {
  Graph g;
  float inf = std::numeric_limits<float>::infinity();
  ValueId a = f32(g, {3}, {-inf, 0, 1});
  CHECK(same(eval<uint8_t>(g, g.add(Op::Eq, {a, g.add_scalar(DType::F32, -inf)})), {1, 0, 0}));
}

TEST(interp_logic_where_cast) {
  Graph g;
  ValueId p = boolean(g, {4}, {1, 1, 0, 0});
  ValueId q = boolean(g, {4}, {1, 0, 1, 0});
  CHECK(same(eval<uint8_t>(g, g.add(Op::And, {p, q})), {1, 0, 0, 0}));
  CHECK(same(eval<uint8_t>(g, g.add(Op::Not, {p})), {0, 0, 1, 1}));
  ValueId zero = g.add_scalar(DType::F32, 0.0);
  ValueId ninf = g.add_scalar(DType::F32, -std::numeric_limits<double>::infinity());
  std::vector<float> w = eval<float>(g, g.add(Op::Where, {p, zero, ninf}));
  CHECK(w[0] == 0 && w[1] == 0 && std::isinf(w[2]) && w[2] < 0 && std::isinf(w[3]));
  Attrs to_int;
  to_int.to = DType::I64;
  CHECK(same(eval<int64_t>(g, g.add(Op::Cast, {p}, to_int)), {1, 1, 0, 0}));
}

TEST(interp_tanh_and_pow) {
  Graph g;
  ValueId a = f32(g, {3}, {-1, 0, 2});
  CHECK(close(eval<float>(g, g.add(Op::Tanh, {a})), {std::tanh(-1.0f), 0, std::tanh(2.0f)}));
  CHECK(close(eval<float>(g, g.add(Op::Pow, {a, g.add_scalar(DType::F32, 3.0)})), {-1, 0, 8}));
}

TEST(interp_reshape_shares_data_and_permute_moves_it) {
  Graph g;
  ValueId a = f32(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK(same(eval<float>(g, g.add(Op::Reshape, {a}, dims({3, 2}))), {1, 2, 3, 4, 5, 6}));  // same order
  CHECK(same(eval<float>(g, g.add(Op::Permute, {a}, dims({1, 0}))), {1, 4, 2, 5, 3, 6}));  // transposed
  // A 3-axis permutation: out[i][j][k] = in[k][i][j] for dims = {2, 0, 1} applied to shape [2,1,3] ... check by shape and content.
  ValueId b = f32(g, {2, 1, 3}, {1, 2, 3, 4, 5, 6});
  ValueId p = g.add(Op::Permute, {b}, dims({2, 0, 1}));
  CHECK(g.type_of(p).shape == (Shape{3, 2, 1}));
  CHECK(same(eval<float>(g, p), {1, 4, 2, 5, 3, 6}));
}

TEST(interp_broadcast_slice_concat) {
  Graph g;
  ValueId col = f32(g, {2, 1}, {1, 2});
  CHECK(same(eval<float>(g, g.add(Op::Broadcast, {col}, dims({2, 3}))), {1, 1, 1, 2, 2, 2}));
  CHECK(same(eval<float>(g, g.add(Op::Broadcast, {g.add_scalar(DType::F32, 7)}, dims({2, 2}))), {7, 7, 7, 7}));

  ValueId a = f32(g, {2, 4}, {1, 2, 3, 4, 5, 6, 7, 8});
  Attrs s = axis(1);
  s.start = 1;
  s.end = 3;
  CHECK(same(eval<float>(g, g.add(Op::Slice, {a}, s)), {2, 3, 6, 7}));
  Attrs rows = axis(0);
  rows.start = 1;
  rows.end = 2;
  CHECK(same(eval<float>(g, g.add(Op::Slice, {a}, rows)), {5, 6, 7, 8}));

  ValueId b = f32(g, {2, 1}, {-1, -2});
  CHECK(same(eval<float>(g, g.add(Op::Concat, {b, a}, axis(1))), {-1, 1, 2, 3, 4, -2, 5, 6, 7, 8}));
  CHECK(same(eval<float>(g, g.add(Op::Concat, {a, a}, axis(0))), {1, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3, 4, 5, 6, 7, 8}));
}

TEST(interp_gather) {
  Graph g;
  ValueId table = f32(g, {3, 2}, {10, 11, 20, 21, 30, 31});
  // Embedding lookup: each index selects a whole row.
  CHECK(same(eval<float>(g, g.add(Op::Gather, {table, i64(g, {1, 3}, {2, 0, 2})})), {30, 31, 10, 11, 30, 31}));
  // Two indices address single elements, broadcasting a column of row ids against a row of column ids.
  ValueId r = i64(g, {2, 1}, {0, 2});
  ValueId c = i64(g, {1, 2}, {1, 0});
  CHECK(same(eval<float>(g, g.add(Op::Gather, {table, r, c})), {11, 10, 31, 30}));
  // Out-of-range indices are an error, not a read of someone else's memory.
  Graph bad;
  ValueId t = f32(bad, {2}, {1, 2});
  ValueId out = bad.add(Op::Gather, {t, i64(bad, {1}, {5})});
  CHECK_THROWS(eval<float>(bad, out), GraphError);
}

TEST(interp_cumsum_and_any) {
  Graph g;
  ValueId a = i64(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  CHECK(same(eval<int64_t>(g, g.add(Op::CumSum, {a}, axis(1))), {1, 3, 6, 4, 9, 15}));
  CHECK(same(eval<int64_t>(g, g.add(Op::CumSum, {a}, axis(0))), {1, 2, 3, 5, 7, 9}));

  ValueId m = boolean(g, {2, 3}, {0, 0, 0, 0, 1, 0});
  Attrs keep = axis(1);
  keep.keepdims = true;
  ValueId any = g.add(Op::ReduceAny, {m}, keep);
  CHECK(g.type_of(any).shape == (Shape{2, 1}));
  CHECK(same(eval<uint8_t>(g, any), {0, 1}));
  CHECK(same(eval<uint8_t>(g, g.add(Op::ReduceAny, {m}, axis(0))), {0, 1, 0}));
}

TEST(interp_matmul) {
  Graph g;
  ValueId a = f32(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  ValueId b = f32(g, {3, 2}, {7, 8, 9, 10, 11, 12});
  // [1 2 3; 4 5 6] * [7 8; 9 10; 11 12] = [58 64; 139 154]
  CHECK(same(eval<float>(g, g.add(Op::MatMul, {a, b})), {58, 64, 139, 154}));
}

TEST(interp_batched_matmul) {
  Graph g;
  // Two independent 1x2 * 2x1 products in one call.
  ValueId a = f32(g, {2, 1, 2}, {1, 2, 3, 4});
  ValueId b = f32(g, {2, 2, 1}, {5, 6, 7, 8});
  CHECK(same(eval<float>(g, g.add(Op::MatMul, {a, b})), {1 * 5 + 2 * 6, 3 * 7 + 4 * 8}));
}

TEST(interp_gemm_adds_bias_to_every_row) {
  Graph g;
  ValueId a = f32(g, {2, 2}, {1, 0, 0, 1});  // identity
  ValueId b = f32(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  ValueId bias = f32(g, {3}, {100, 200, 300});
  CHECK(same(eval<float>(g, g.add(Op::Gemm, {a, b, bias})), {101, 202, 303, 104, 205, 306}));
}

TEST(interp_layer_norm) {
  Graph g;
  // x = [1, 2, 3, 4]: mean 2.5, variance ((1.5)^2 + (0.5)^2) * 2 / 4 = 1.25.
  ValueId x = f32(g, {1, 4}, {1, 2, 3, 4});
  ValueId w = f32(g, {4}, {1, 1, 2, 2});
  ValueId b = f32(g, {4}, {0, 10, 0, 10});
  Attrs a;
  a.eps = 0.0;
  float s = std::sqrt(1.25f);
  CHECK(close(eval<float>(g, g.add(Op::LayerNorm, {x, w, b}, a)), {-1.5f / s, -0.5f / s + 10, 2 * 0.5f / s, 2 * 1.5f / s + 10}));
  // eps sits inside the square root: a constant vector has variance 0 and must not divide by zero.
  ValueId flat = f32(g, {1, 4}, {5, 5, 5, 5});
  a.eps = 1e-5;
  CHECK(close(eval<float>(g, g.add(Op::LayerNorm, {flat, w, b}, a)), {0, 10, 0, 10}));
}

TEST(interp_softmax) {
  Graph g;
  float inf = std::numeric_limits<float>::infinity();
  ValueId x = f32(g, {2, 3}, {0, 0, 0, 1, 2, -inf});
  std::vector<float> y = eval<float>(g, g.add(Op::Softmax, {x}, axis(1)));
  float e = std::exp(1.0f);
  CHECK(close(y, {1 / 3.0f, 1 / 3.0f, 1 / 3.0f, 1 / (1 + e), e / (1 + e), 0}));
  // Large inputs would overflow exp() without the max subtraction.
  ValueId big = f32(g, {2}, {1000, 1000});
  CHECK(close(eval<float>(g, g.add(Op::Softmax, {big}, axis(0))), {0.5, 0.5}));
  // Along the first axis instead of the last.
  ValueId cols = f32(g, {2, 2}, {0, 5, 0, 5});
  CHECK(close(eval<float>(g, g.add(Op::Softmax, {cols}, axis(0))), {0.5, 0.5, 0.5, 0.5}));
}

TEST(interp_checks_inputs) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {2}}, "x");
  g.outputs = {g.add(Op::Tanh, {x})};
  CHECK_THROWS(run_graph(g, {}, {}), GraphError);  // missing input
  std::vector<int64_t> wrong = {1, 2};
  Tensor t = Tensor::borrow({DType::I64, {2}}, reinterpret_cast<const std::byte*>(wrong.data()));
  CHECK_THROWS(run_graph(g, {&t, 1}, {}), GraphError);  // wrong dtype
}

TEST(interp_frees_dead_values) {
  // A chain of 50 adds on a 1000-element tensor. Each step's input dies as
  // soon as the step has run, so the peak is two buffers, not fifty.
  Graph g;
  ValueId v = g.add(Op::Broadcast, {g.add_scalar(DType::F32, 1.0)}, dims({1000}));
  ValueId one = g.add_scalar(DType::F32, 1.0);
  for (int k = 0; k < 50; ++k) v = g.add(Op::Add, {v, one});
  g.outputs = {v};
  RunStats stats;
  Tensor out = run_graph(g, {}, {}, &stats).at(0);
  CHECK(out.data<float>()[999] == 51.0f);
  CHECK(stats.allocated_bytes == 51 * 4000);
  CHECK(stats.peak_bytes == 2 * 4000);
}
