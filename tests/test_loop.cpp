// Loop IR: lowering is checked differentially (graph interpreter against loop
// interpreter on the same graph), and the verifier is checked against
// deliberately broken programs.
#include <limits>

#include "check.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;
using namespace mlc::loop;

TEST(loop_index_arithmetic_folds_constants) {
  Program p;
  ExprPtr i = var(0);
  CHECK(add(i, int_imm(0)) == i);
  CHECK(mul(i, int_imm(1)) == i);
  CHECK(mul(i, int_imm(0))->kind == ExprKind::IntImm);
  CHECK(add(int_imm(2), int_imm(3))->int_value == 5);
  // Flattening [i, 0, j] into a buffer of shape [4, 1, 6]: the size-1 axis vanishes.
  Buffer b{"b", DType::F32, {4, 1, 6}, Storage::Arena, 0, -1};
  CHECK(print_expr(p, flat_index(b, {var(0), int_imm(0), var(1)})) == "((i0 * 6) + i1)");
}

TEST(loop_constructors_check_types) {
  CHECK_THROWS(add(int_imm(1), float_imm(1)), LoopError);
  CHECK_THROWS(binary(BinOp::Div, int_imm(1), int_imm(2)), LoopError);
  CHECK_THROWS(binary(BinOp::And, int_imm(1), int_imm(2)), LoopError);
  CHECK_THROWS(select(int_imm(1), float_imm(1), float_imm(2)), LoopError);
  CHECK_THROWS(load(0, DType::F32, {float_imm(0)}), LoopError);
  CHECK_THROWS(call(Intrinsic::Exp, {int_imm(1)}), LoopError);
  CHECK(binary(BinOp::Le, float_imm(1), float_imm(2))->type == DType::Bool);
}

TEST(lower_elementwise_and_broadcast) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1), row = random_f32(g, {3}, 2), col = random_f32(g, {2, 1}, 3);
  ValueId half = g.add_scalar(DType::F32, 0.5);
  CHECK(lowering_matches(g, {g.add(Op::Add, {a, row}), g.add(Op::Sub, {a, col}), g.add(Op::Mul, {col, row}), g.add(Op::Mul, {a, half}),
                             g.add(Op::Tanh, {a}), g.add(Op::Pow, {a, g.add_scalar(DType::F32, 3.0)}),
                             g.add(Op::Broadcast, {col}, dims({4, 2, 3})), g.add(Op::Broadcast, {half}, dims({2, 2}))}));
}

TEST(lower_comparisons_logic_where_cast) {
  Graph g;
  float inf = std::numeric_limits<float>::infinity();
  ValueId a = f32(g, {2, 3}, {-inf, 0, 1, 2, -inf, 3});
  ValueId is_inf = g.add(Op::Eq, {a, g.add_scalar(DType::F32, -inf)});
  ValueId n = i64(g, {3}, {0, 1, 2});
  ValueId le = g.add(Op::Le, {g.add(Op::Reshape, {n}, dims({1, 3})), g.add(Op::Reshape, {n}, dims({3, 1}))});
  ValueId p = boolean(g, {3}, {1, 0, 1});
  Attrs to_int, to_float;
  to_int.to = DType::I64;
  to_float.to = DType::F32;
  CHECK(lowering_matches(g, {is_inf, g.add(Op::Not, {is_inf}), le, g.add(Op::And, {le, p}), g.add(Op::Ne, {n, g.add_scalar(DType::I64, 1)}),
                             g.add(Op::Where, {is_inf, g.add_scalar(DType::F32, 0.0), a}), g.add(Op::Cast, {p}, to_int),
                             g.add(Op::Cast, {n}, to_float), g.add(Op::Sub, {n, g.add_scalar(DType::I64, 1)})}));
}

TEST(lower_layout_ops) {
  Graph g;
  ValueId a = random_f32(g, {2, 3, 4}, 1);
  Attrs s = axis(2);
  s.start = 1;
  s.end = 3;
  ValueId b = random_f32(g, {2, 1, 4}, 2);
  CHECK(lowering_matches(g, {g.add(Op::Permute, {a}, dims({2, 0, 1})), g.add(Op::Permute, {a}, dims({0, 2, 1})), g.add(Op::Slice, {a}, s),
                             g.add(Op::Concat, {a, b, a}, axis(1)), g.add(Op::Reshape, {g.add(Op::Tanh, {a})}, dims({6, 4}))}));
}

TEST(lower_gather) {
  Graph g;
  ValueId table = random_f32(g, {5, 3}, 1);
  ValueId rows = i64(g, {2, 1}, {4, 0}), cols = i64(g, {1, 3}, {2, 0, 1});
  CHECK(lowering_matches(g, {g.add(Op::Gather, {table, i64(g, {1, 4}, {3, 3, 0, 4})}), g.add(Op::Gather, {table, rows, cols})}));
}

TEST(lower_gather_out_of_range_is_caught_by_the_loop_interpreter) {
  Graph g;
  ValueId table = random_f32(g, {5, 3}, 1);
  g.outputs = {g.add(Op::Gather, {table, i64(g, {2}, {1, 5})})};
  Program p = lower_graph(g, 0);
  std::vector<std::byte> out(p.outputs[0].byte_size());
  void* ptrs[] = {out.data()};
  CHECK_THROWS(run_program(p, {}, {}, ptrs), LoopError);
}

TEST(lower_cumsum_and_reduce_any) {
  Graph g;
  ValueId a = i64(g, {2, 3}, {1, 2, 3, 4, 5, 6});
  ValueId m = boolean(g, {2, 3}, {0, 0, 0, 0, 1, 0});
  Attrs keep = axis(1);
  keep.keepdims = true;
  CHECK(lowering_matches(g, {g.add(Op::CumSum, {a}, axis(0)), g.add(Op::CumSum, {a}, axis(1)), g.add(Op::CumSum, {random_f32(g, {4}, 1)}, axis(0)),
                             g.add(Op::ReduceAny, {m}, axis(0)), g.add(Op::ReduceAny, {m}, keep)}));
}

TEST(lower_matmul_and_gemm) {
  Graph g;
  ValueId a = random_f32(g, {3, 4}, 1), b = random_f32(g, {4, 5}, 2), bias = random_f32(g, {5}, 3);
  ValueId ba = random_f32(g, {2, 3, 4}, 4), bb = random_f32(g, {2, 4, 2}, 5);
  CHECK(lowering_matches(g, {g.add(Op::MatMul, {a, b}), g.add(Op::Gemm, {a, b, bias}), g.add(Op::MatMul, {ba, bb}),
                             g.add(Op::MatMul, {random_f32(g, {1, 1}, 6), random_f32(g, {1, 1}, 7)})}));
}

TEST(lower_layer_norm_and_softmax) {
  Graph g;
  float inf = std::numeric_limits<float>::infinity();
  ValueId x = random_f32(g, {2, 3, 4}, 1);
  Attrs ln;
  ln.eps = 1e-5;
  ValueId masked = f32(g, {2, 3}, {0.5f, -inf, 1, -inf, -inf, -inf});  // second row is all -inf: NaN, as in PyTorch
  CHECK(lowering_matches(g, {g.add(Op::LayerNorm, {x, random_f32(g, {4}, 2), random_f32(g, {4}, 3)}, ln), g.add(Op::Softmax, {x}, axis(2)),
                             g.add(Op::Softmax, {x}, axis(0)), g.add(Op::Softmax, {masked}, axis(1))}));
}

TEST(lower_output_that_is_a_constant_or_shared_gets_a_copy) {
  Graph g;
  ValueId c = random_f32(g, {2, 2}, 1);
  ValueId t = g.add(Op::Tanh, {c});
  CHECK(lowering_matches(g, {c, t, t, g.add(Op::Reshape, {t}, dims({4}))}));
}

TEST(lower_tiny_transformer) {
  Graph g;
  ValueId out = tiny_transformer(g, i64(g, {1, 5}, {3, 10, 0, 7, 3}));
  CHECK(lowering_matches(g, {out}, 1e-4));
}

TEST(lower_gives_reshape_no_code_and_no_storage) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1);
  ValueId t = g.add(Op::Tanh, {a});
  ValueId r = g.add(Op::Reshape, {t}, dims({3, 2}));
  g.outputs = {g.add(Op::Tanh, {r})};
  Program p = lower_graph(g, 0);
  CHECK(p.kernels.size() == 2);  // the two tanh nodes; the constant and the reshape need none
  const Buffer& bt = p.buffers[t];
  const Buffer& br = p.buffers[r];
  CHECK(bt.storage == Storage::Arena && br.storage == Storage::Arena && bt.offset == br.offset);
  CHECK(br.shape == (Shape{3, 2}));
  CHECK(p.arena_bytes == 64);  // one 24-byte buffer, rounded up to the 64-byte alignment
}

TEST(loop_printer) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1), b = random_f32(g, {3, 2}, 2);
  g.outputs = {g.add(Op::MatMul, {a, b})};
  std::string text = print_program(lower_graph(g, 0));
  CHECK(text.find("kernel matmul") != std::string::npos);
  CHECK(text.find("for i0 in 0..2:") != std::string::npos);
  CHECK(text.find("v2[i0, i3] = (v2[i0, i3] + (v0[i0, i2] * v1[i2, i3]))") != std::string::npos);
}

TEST(loop_verifier_catches_broken_programs) {
  auto make = [] {
    Graph g;
    ValueId a = random_f32(g, {2, 3}, 1);
    g.outputs = {g.add(Op::Tanh, {g.add(Op::Tanh, {a})})};
    return lower_graph(g, 0);
  };
  CHECK(verify_program(make()).empty());

  Program rank = make();  // access a rank-2 buffer with one index
  rank.kernels[0].body = make_store(1, {int_imm(0)}, float_imm(1));
  CHECK(!verify_program(rank).empty());

  Program readonly = make();  // store into an inline constant
  readonly.kernels[0].body = make_store(0, {int_imm(0), int_imm(0)}, float_imm(1));
  CHECK(!verify_program(readonly).empty());

  Program dtype = make();  // store an int into a float buffer
  dtype.kernels[0].body = make_store(1, {int_imm(0), int_imm(0)}, int_imm(1));
  CHECK(!verify_program(dtype).empty());

  Program scope = make();  // use a loop variable with no enclosing loop
  scope.kernels[0].body = make_store(1, {var(7), int_imm(0)}, float_imm(1));
  CHECK(!verify_program(scope).empty());

  Program arena = make();  // a buffer that runs past the end of the arena
  arena.arena_bytes = 8;
  CHECK(!verify_program(arena).empty());
  CHECK_THROWS(check_program(arena), LoopError);
}
