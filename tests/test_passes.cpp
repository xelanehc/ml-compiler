// Graph passes. Every pass gets a differential test (interpret before,
// interpret after, compare) and tests that it fires where it should and
// holds back where it must.
#include <limits>

#include "check.h"
#include "graph/passes.h"
#include "helpers.h"

using namespace mlc;
using namespace mlc::test;

namespace {

int count(const Graph& g, Op op) {
  int n = 0;
  for (const Node& node : g.nodes) n += node.op == op;
  return n;
}

// Runs both graphs (no weights; `inputs` if they take any) and compares every output.
bool same_results(const Graph& before, const Graph& after, double tol = 0, std::span<const Tensor> inputs = {}) {
  std::vector<Tensor> a = run_graph(before, inputs, {}), b = run_graph(after, inputs, {});
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (a[i].type != b[i].type || !same_values(a[i].type, a[i].ptr, b[i].ptr, tol)) return false;
  return true;
}

ValueId scalar(Graph& g, double v) { return g.add_scalar(DType::F32, v); }

// GELU exactly as torch.export spells it: nine nodes.
ValueId exported_gelu(Graph& g, ValueId h, double tanh_scale = 0.7978845608028654) {
  ValueId half = g.add(Op::Mul, {h, scalar(g, 0.5)});
  ValueId cube = g.add(Op::Pow, {h, scalar(g, 3.0)});
  ValueId inner = g.add(Op::Mul, {g.add(Op::Add, {h, g.add(Op::Mul, {cube, scalar(g, 0.044715)})}), scalar(g, tanh_scale)});
  return g.add(Op::Mul, {half, g.add(Op::Add, {g.add(Op::Tanh, {inner}), scalar(g, 1.0)})});
}

// The additive attention mask as the export builds it: compare positions,
// then where(allowed, 0, -inf). `causal` false gives a mask that allows everything.
ValueId exported_mask(Graph& g, int64_t t, bool causal = true) {
  std::vector<int64_t> positions(static_cast<size_t>(t));
  for (int64_t i = 0; i < t; ++i) positions[static_cast<size_t>(i)] = causal ? i : 0;
  ValueId pos = i64(g, {t}, positions);
  ValueId allowed = g.add(Op::Le, {g.add(Op::Reshape, {pos}, dims({1, 1, 1, t})), g.add(Op::Reshape, {pos}, dims({1, 1, t, 1}))});
  return g.add(Op::Where, {allowed, scalar(g, 0.0), scalar(g, -INFINITY)});
}

// Masked softmax with the all-masked-row guard, as the export spells it.
ValueId exported_attention_probs(Graph& g, ValueId scores, ValueId mask) {
  const Shape shape = g.type_of(scores).shape;
  ValueId s = g.add(Op::Add, {scores, mask});
  ValueId probs = g.add(Op::Softmax, {s}, axis(3));
  Attrs keep = axis(3);
  keep.keepdims = true;
  ValueId not_inf = g.add(Op::Not, {g.add(Op::Eq, {s, scalar(g, -INFINITY)})});
  ValueId dead = g.add(Op::Not, {g.add(Op::ReduceAny, {not_inf}, keep)});
  ValueId zeros = g.add(Op::Broadcast, {scalar(g, 0.0)}, dims(shape));
  return g.add(Op::Where, {dead, zeros, probs});
}

}  // namespace

TEST(rewrite_with_no_rule_copies_the_graph) {
  Graph g;
  g.outputs = {tiny_transformer(g, i64(g, {1, 5}, {3, 10, 0, 7, 3}))};
  Graph copy = rewrite(g, [](Graph&, const Node&, const Constant*) { return kNoValue; });
  CHECK(print_graph(copy) == print_graph(g));
  CHECK(same_results(g, copy));
}

TEST(rewrite_rejects_a_rule_that_changes_a_type) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1);
  g.outputs = {g.add(Op::Tanh, {a})};
  CHECK_THROWS(rewrite(g, [](Graph& out, const Node& n, const Constant*) -> ValueId {
    return n.op == Op::Tanh ? out.add_scalar(DType::F32, 0.0) : kNoValue;  // a scalar where a [2,3] tensor was
  }), GraphError);
}

TEST(dce_removes_unused_nodes_and_keeps_inputs) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {2}}, "x");
  ValueId unused_input = g.add_input({DType::F32, {2}}, "y");
  (void)unused_input;
  g.add(Op::Tanh, {g.add(Op::Tanh, {x})});  // a dead chain
  g.add_scalar(DType::F32, 7.0);            // a dead constant
  g.outputs = {g.add(Op::Add, {x, x})};
  Graph out = eliminate_dead_code(g);
  CHECK(out.nodes.size() == 3);  // x, y, add
  CHECK(out.inputs.size() == 2 && out.constants.empty());
  CHECK(verify_graph(out).empty());
}

TEST(simplify_removes_no_ops) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1);
  Attrs whole = axis(1);
  whole.end = 3;
  ValueId v = g.add(Op::Broadcast, {a}, dims({2, 3}));
  v = g.add(Op::Reshape, {v}, dims({2, 3}));
  v = g.add(Op::Permute, {v}, dims({0, 1}));
  v = g.add(Op::Slice, {v}, whole);
  v = g.add(Op::Concat, {v}, axis(0));
  v = g.add(Op::Add, {v, scalar(g, 0.0)});
  v = g.add(Op::Mul, {scalar(g, 1.0), v});
  g.outputs = {v};
  Graph out = eliminate_dead_code(simplify(g));
  CHECK(out.nodes.size() == 1);  // only the constant is left
  CHECK(same_results(g, out));
}

TEST(simplify_keeps_ops_that_do_something) {
  Graph g;
  ValueId a = random_f32(g, {2, 1}, 1);
  // x + 0 where the zero broadcasts x to a bigger shape is not a no-op.
  ValueId zeros = g.add(Op::Broadcast, {scalar(g, 0.0)}, dims({2, 3}));
  g.outputs = {g.add(Op::Add, {a, zeros}), g.add(Op::Broadcast, {a}, dims({2, 3})), g.add(Op::Mul, {a, scalar(g, 2.0)})};
  Graph out = eliminate_dead_code(simplify(g));
  CHECK(count(out, Op::Add) == 1 && count(out, Op::Broadcast) == 2 && count(out, Op::Mul) == 1);
  CHECK(same_results(g, out));
}

TEST(simplify_collapses_reshape_and_permute_chains) {
  Graph g;
  ValueId a = random_f32(g, {2, 3, 4}, 1);
  ValueId r = g.add(Op::Reshape, {g.add(Op::Reshape, {g.add(Op::Reshape, {a}, dims({6, 4}))}, dims({4, 6}))}, dims({24}));
  // GPT-2's head merge: [2,0,1,3]-style round trip followed by a real permute.
  ValueId p = g.add(Op::Permute, {g.add(Op::Permute, {a}, dims({2, 0, 1}))}, dims({1, 2, 0}));  // composes to the identity
  ValueId q = g.add(Op::Permute, {p}, dims({0, 2, 1}));
  ValueId back = g.add(Op::Reshape, {g.add(Op::Reshape, {a}, dims({24}))}, dims({2, 3, 4}));    // reshape back to the original shape
  ValueId nn = g.add(Op::Not, {g.add(Op::Not, {boolean(g, {2}, {1, 0})})});
  g.outputs = {r, q, back, nn};
  Graph out = eliminate_dead_code(simplify(g));
  CHECK(count(out, Op::Reshape) == 1);
  CHECK(count(out, Op::Permute) == 1);
  CHECK(count(out, Op::Not) == 0);
  CHECK(out.outputs[2] == 0);  // `back` is the constant itself
  CHECK(same_results(g, out));
}

TEST(cse_merges_identical_nodes_and_constants) {
  Graph g;
  ValueId a = random_f32(g, {2, 3}, 1);
  ValueId x = g.add(Op::Mul, {g.add(Op::Tanh, {a}), scalar(g, 0.5)});
  ValueId y = g.add(Op::Mul, {g.add(Op::Tanh, {a}), scalar(g, 0.5)});  // the same computation, built again
  ValueId z = g.add(Op::Mul, {g.add(Op::Tanh, {a}), scalar(g, 0.25)}); // a different constant: must stay
  Attrs s1 = axis(1), s2 = axis(1);
  s1.end = 2;
  s2.end = 1;
  g.outputs = {g.add(Op::Add, {x, y}), z, g.add(Op::Slice, {a}, s1), g.add(Op::Slice, {a}, s2)};
  Graph out = eliminate_dead_code(eliminate_common_subexpressions(g));
  CHECK(count(out, Op::Tanh) == 1);
  CHECK(count(out, Op::Mul) == 2);
  CHECK(count(out, Op::Slice) == 2);   // same op and operand, different attributes
  CHECK(out.constants.size() == 3);    // a, 0.5, 0.25
  const Node& add = out.nodes[out.outputs[0]];
  CHECK(add.inputs[0] == add.inputs[1]);
  CHECK(same_results(g, out));
}

TEST(cse_does_not_merge_constants_of_different_types) {
  Graph g;
  ValueId f = g.add_scalar(DType::F32, 0.0), i = g.add_scalar(DType::I64, 0);
  g.outputs = {g.add(Op::Add, {random_f32(g, {2}, 1), f}), g.add(Op::Add, {i64(g, {2}, {1, 2}), i})};
  Graph out = eliminate_common_subexpressions(g);
  CHECK(out.constants.size() == 4);
  CHECK(same_results(g, out));
}

TEST(fold_evaluates_constant_subgraphs) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {1, 1, 4, 4}}, "x");
  g.outputs = {g.add(Op::Add, {x, exported_mask(g, 4)})};
  Graph out = eliminate_dead_code(fold_constants(g));
  CHECK(out.nodes.size() == 3);  // x, the folded mask, add
  const Constant& mask = out.constants[0];
  CHECK(mask.type == (Type{DType::F32, {1, 1, 4, 4}}));
  const float* m = reinterpret_cast<const float*>(mask.inline_data.data());
  CHECK(m[0] == 0 && std::isinf(m[1]) && m[4] == 0 && m[5] == 0 && m[15] == 0);
}

TEST(fold_leaves_weights_broadcasts_and_big_results_alone) {
  Graph g;
  Constant weight;
  weight.type = {DType::F32, {4}};
  weight.weights_offset = 0;
  ValueId w = g.add_const(weight, "w");
  ValueId fill = g.add(Op::Broadcast, {scalar(g, 0.0)}, dims({8, 8}));
  ValueId big = g.add(Op::Tanh, {random_f32(g, {64}, 1)});
  g.outputs = {g.add(Op::Tanh, {w}), fill, big};
  Graph out = fold_constants(g, /*max_bytes=*/64);
  CHECK(count(out, Op::Tanh) == 2);       // one on a weight, one over the size limit
  CHECK(count(out, Op::Broadcast) == 1);  // folding it would turn 4 bytes into 256
  Graph roomy = fold_constants(g, 1024);
  CHECK(count(roomy, Op::Tanh) == 1);     // now the 256-byte result is allowed
}

TEST(evaluate_constant_refuses_inputs_and_weights) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {2}}, "x");
  ValueId c = g.add(Op::Tanh, {f32(g, {2}, {0, 1})});
  ValueId dep = g.add(Op::Add, {x, c});
  CHECK(!evaluate_constant(g, dep, 1 << 20));
  CHECK(!evaluate_constant(g, c, 4));  // over the size limit
  std::optional<Tensor> value = evaluate_constant(g, c, 1 << 20);
  CHECK(value && value->data<float>()[0] == 0.0f);
  CHECK_NEAR(value->data<float>()[1], std::tanh(1.0f), 1e-7);
  // A value that is itself a constant must come back owning its bytes.
  std::optional<Tensor> leaf = evaluate_constant(g, 1, 1 << 20);
  CHECK(leaf && leaf->owner && leaf->data<float>()[1] == 1.0f);
}

TEST(gelu_interpreter_matches_its_definition) {
  Graph g;
  ValueId x = f32(g, {4}, {-3, -0.5, 0, 2});
  std::vector<float> y = eval<float>(g, g.add(Op::Gelu, {x}));
  for (size_t i = 0; i < 4; ++i) {
    double v = std::vector<double>{-3, -0.5, 0, 2}[i];
    CHECK_NEAR(y[i], 0.5 * v * (1 + std::tanh(std::sqrt(2 / M_PI) * (v + 0.044715 * v * v * v))), 1e-6);
  }
}

TEST(fuse_gelu_recognizes_the_exported_form) {
  Graph g;
  ValueId h = random_f32(g, {2, 5}, 1);
  g.outputs = {exported_gelu(g, h)};
  Graph out = eliminate_dead_code(fuse_gelu(g));
  CHECK(count(out, Op::Gelu) == 1);
  CHECK(out.nodes.size() == 2);  // h and gelu: all nine nodes and their five constants are gone
  CHECK(same_results(g, out, 1e-6));
}

TEST(fuse_gelu_leaves_lookalikes_alone) {
  Graph wrong_constant;
  wrong_constant.outputs = {exported_gelu(wrong_constant, random_f32(wrong_constant, {4}, 1), /*tanh_scale=*/0.6)};
  CHECK(count(fuse_gelu(wrong_constant), Op::Gelu) == 0);

  // Same shape of computation, but the cube is of a different tensor.
  Graph g;
  ValueId h = random_f32(g, {4}, 1), other = random_f32(g, {4}, 2);
  ValueId half = g.add(Op::Mul, {h, scalar(g, 0.5)});
  ValueId cube = g.add(Op::Pow, {other, scalar(g, 3.0)});
  ValueId inner = g.add(Op::Mul, {g.add(Op::Add, {h, g.add(Op::Mul, {cube, scalar(g, 0.044715)})}), scalar(g, 0.7978845608028654)});
  g.outputs = {g.add(Op::Mul, {half, g.add(Op::Add, {g.add(Op::Tanh, {inner}), scalar(g, 1.0)})})};
  CHECK(count(fuse_gelu(g), Op::Gelu) == 0);
}

TEST(causal_softmax_interpreter) {
  Graph g;
  ValueId x = f32(g, {1, 3, 3}, {5, 100, 100, 0, 0, 100, 0, 0, 0});
  Attrs a = axis(2);
  a.causal = true;
  std::vector<float> y = eval<float>(g, g.add(Op::Softmax, {x}, a));
  // Row 0 sees one column, row 1 two, row 2 three; the 100s above the diagonal are ignored.
  std::vector<float> want = {1, 0, 0, 0.5, 0.5, 0, 1 / 3.0f, 1 / 3.0f, 1 / 3.0f};
  for (size_t i = 0; i < 9; ++i) CHECK_NEAR(y[i], want[i], 1e-6);
  // Not along the last axis, or not square: ill-typed.
  Attrs bad = axis(1);
  bad.causal = true;
  CHECK_THROWS(g.add(Op::Softmax, {x}, bad), GraphError);
  CHECK_THROWS(g.add(Op::Softmax, {f32(g, {2, 3}, {0, 0, 0, 0, 0, 0})}, a), GraphError);
}

TEST(fuse_causal_softmax_recognizes_the_exported_form) {
  Graph g;
  ValueId scores = random_f32(g, {1, 2, 4, 4}, 1);
  g.outputs = {exported_attention_probs(g, scores, exported_mask(g, 4))};
  Graph out = eliminate_dead_code(fuse_causal_softmax(g));
  CHECK(out.nodes.size() == 2);  // scores and one softmax; the mask is not needed at run time
  CHECK(out.nodes[1].op == Op::Softmax && out.nodes[1].attrs.causal);
  CHECK(same_results(g, out, 1e-6));
}

TEST(fuse_causal_softmax_requires_a_causal_mask) {
  // A mask that allows every position: the guard is still dead, but this is
  // not the causal pattern, and the rule must not fire.
  Graph g;
  ValueId scores = random_f32(g, {1, 2, 4, 4}, 1);
  g.outputs = {exported_attention_probs(g, scores, exported_mask(g, 4, /*causal=*/false))};
  CHECK(count(fuse_causal_softmax(g), Op::Where) == 2);

  // A mask that depends on a graph input cannot be evaluated at compile time.
  Graph dynamic;
  ValueId s = random_f32(dynamic, {1, 2, 4, 4}, 1);
  ValueId m = dynamic.add_input({DType::F32, {1, 1, 4, 4}}, "mask");
  dynamic.outputs = {exported_attention_probs(dynamic, s, m)};
  CHECK(count(fuse_causal_softmax(dynamic), Op::Where) == 1);
}

TEST(optimize_pipeline_on_exported_patterns) {
  // A block with every pattern the passes target, as the export would write it.
  // The scores are a graph input; as a constant, folding would simply
  // evaluate the whole thing.
  Graph g;
  ValueId scores = g.add_input({DType::F32, {1, 2, 4, 4}}, "scores");
  ValueId probs = g.add(Op::Broadcast, {exported_attention_probs(g, scores, exported_mask(g, 4))}, dims({1, 2, 4, 4}));
  ValueId probs2 = exported_attention_probs(g, g.add(Op::Tanh, {probs}), exported_mask(g, 4));  // a second layer, its own copy of the mask
  ValueId merged = g.add(Op::Permute, {g.add(Op::Permute, {g.add(Op::Permute, {probs2}, dims({2, 0, 1, 3}))}, dims({1, 2, 0, 3}))}, dims({0, 2, 1, 3}));
  g.outputs = {exported_gelu(g, g.add(Op::Reshape, {merged}, dims({4, 8})))};

  std::vector<PassStat> stats;
  Graph out = optimize(g, {}, &stats);
  std::vector<float> data = noise(32, 9);
  Tensor in = Tensor::borrow({DType::F32, {1, 2, 4, 4}}, reinterpret_cast<const std::byte*>(data.data()));
  CHECK(same_results(g, out, 1e-6, {&in, 1}));
  CHECK(count(out, Op::Softmax) == 2 && count(out, Op::Gelu) == 1 && count(out, Op::Permute) == 1);
  CHECK(count(out, Op::Where) == 0 && count(out, Op::Broadcast) == 0 && count(out, Op::Pow) == 0);
  CHECK(out.nodes.size() == 7);  // scores, softmax, tanh, softmax, permute, reshape, gelu
  CHECK(stats.size() == 6);
  for (const PassStat& s : stats) CHECK(s.nodes_after <= s.nodes_before);
  CHECK(stats.back().nodes_after == out.nodes.size());

  // With everything off the graph comes back as it went in.
  OptOptions off;
  off.disable("all");
  CHECK(optimize(g, off).nodes.size() == g.nodes.size());
}

TEST(optimize_options_parse_pass_names) {
  OptOptions o;
  o.disable("gelu,memory");
  CHECK(!o.gelu && !o.memory && o.simplify && o.fold && o.cse && o.attention);
  o.disable("");
  CHECK(o.simplify);
  CHECK_THROWS(o.disable("fusion"), GraphError);
}

TEST(optimized_tiny_transformer_matches_through_every_stage) {
  Graph g;
  g.outputs = {tiny_transformer(g, i64(g, {1, 5}, {3, 10, 0, 7, 3}))};
  Graph out = optimize(g);
  CHECK(out.nodes.size() < g.nodes.size());
  CHECK(same_results(g, out, 1e-5));
  CHECK(lowering_matches(out, out.outputs, 1e-4));  // and its loops agree with its graph
}

// ---- Memory planning ----

TEST(memory_planning_reuses_dead_buffers) {
  // A chain of permutes cannot run in place, so each needs a fresh buffer,
  // but only two are ever alive at once.
  Graph g;
  ValueId v = random_f32(g, {8, 8}, 1);
  for (int k = 0; k < 10; ++k) v = g.add(Op::Permute, {v}, dims({1, 0}));
  g.outputs = {g.add(Op::Tanh, {v})};
  loop::Program planned = loop::lower_graph(g, 0);
  loop::Program naive = loop::lower_graph(g, 0, {false});
  CHECK(naive.arena_bytes == 10 * 256);
  CHECK(planned.arena_bytes == 2 * 256);
  CHECK(planned.unplanned_arena_bytes == naive.arena_bytes);
  CHECK(lowering_matches(g, g.outputs));
}

TEST(memory_planning_runs_elementwise_ops_in_place) {
  Graph g;
  ValueId v = g.add(Op::Permute, {random_f32(g, {8, 8}, 1)}, dims({1, 0}));
  for (int k = 0; k < 10; ++k) v = g.add(Op::Tanh, {g.add(Op::Add, {v, scalar(g, 0.1)})});
  g.outputs = {g.add(Op::Permute, {v}, dims({1, 0}))};
  CHECK(loop::lower_graph(g, 0).arena_bytes == 256);  // twenty ops, one buffer
  CHECK(lowering_matches(g, g.outputs));
}

TEST(memory_planning_keeps_values_that_are_read_later) {
  Graph g;
  ValueId x = g.add(Op::Tanh, {random_f32(g, {4, 4}, 1)});
  ValueId y = g.add(Op::Add, {x, scalar(g, 1.0)});       // x is read again below, so this must not overwrite it
  ValueId z = g.add(Op::Mul, {y, x});
  ValueId r = g.add(Op::Reshape, {z}, dims({16}));
  ValueId w = g.add(Op::Tanh, {g.add(Op::Permute, {y}, dims({1, 0}))});
  g.outputs = {g.add(Op::Add, {g.add(Op::Reshape, {w}, dims({16})), r})};  // z is kept alive through its reshape
  CHECK(lowering_matches(g, g.outputs));
}

TEST(memory_planning_in_place_with_broadcast_and_shared_operands) {
  Graph g;
  ValueId x = g.add(Op::Tanh, {random_f32(g, {1, 4}, 1)});
  ValueId col = g.add(Op::Tanh, {random_f32(g, {4, 1}, 2)});
  ValueId sq = g.add(Op::Mul, {x, x});                             // both operands are the dying value
  ValueId flat = g.add(Op::Add, {sq, g.add(Op::Reshape, {sq}, dims({4}))});  // the same bytes under two shapes
  ValueId outer = g.add(Op::Add, {col, flat});                     // [4,1] + [1,4] -> [4,4]: neither operand can be reused
  Attrs ln;
  ln.eps = 1e-5;
  ValueId normed = g.add(Op::LayerNorm, {outer, random_f32(g, {4}, 3), random_f32(g, {4}, 4)}, ln);
  g.outputs = {g.add(Op::Permute, {g.add(Op::Softmax, {normed}, axis(1))}, dims({1, 0}))};
  CHECK(lowering_matches(g, g.outputs));
}

TEST(memory_planning_matches_no_planning_on_the_toy_transformer) {
  Graph g;
  g.outputs = {tiny_transformer(g, i64(g, {1, 5}, {3, 10, 0, 7, 3}))};
  loop::Program planned = loop::lower_graph(g, 0), naive = loop::lower_graph(g, 0, {false});
  CHECK(planned.arena_bytes < naive.arena_bytes / 3);
  std::vector<std::byte> a(planned.outputs[0].byte_size()), b(a.size());
  void* pa[] = {a.data()};
  void* pb[] = {b.data()};
  loop::run_program(planned, {}, {}, pa);
  loop::run_program(naive, {}, {}, pb);
  CHECK(a == b);  // same kernels, different addresses: bit-identical results
}
