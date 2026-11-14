// Loop transformations: the dependence analysis, each rewrite checked against
// the loop interpreter, and the scheduled program compiled and compared with
// the graph interpreter at sizes that are not multiples of anything.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <memory>

#include "check.h"
#include "codegen/codegen.h"
#include "graph/passes.h"
#include "helpers.h"
#include "loop/schedule.h"

#if defined(__aarch64__)
#include "../runtime/mlc_neon.h"
#endif

using namespace mlc;
using namespace mlc::loop;
using namespace mlc::test;

namespace {

Attrs trans_b() {
  Attrs a;
  a.trans_b = true;
  return a;
}

// C = A @ Bt^T with constant operands, lowered to the naive i, j, p nest.
Program dot_product_program(int64_t m, int64_t k, int64_t n) {
  Graph g;
  g.outputs = {g.add(Op::MatMul, {random_f32(g, {m, k}, 1), random_f32(g, {n, k}, 2)}, trans_b())};
  return lower_graph(g, 0, kAtRunTime);
}

// Replaces kernel 0's body and reports whether the program still computes
// exactly what `before` did. The loop interpreter runs every loop in order,
// and none of the rewrites changes the order in which one element's terms are
// added, so the comparison is bit for bit.
bool same_after(const Program& before, StmtPtr body) {
  Program after = before;
  after.kernels[0].body = std::move(body);
  check_program(after);
  return run_loops(before) == run_loops(after);
}

size_t count(const std::string& text, const std::string& what) {
  size_t n = 0;
  for (size_t at = text.find(what); at != std::string::npos; at = text.find(what, at + 1)) ++n;
  return n;
}

std::string scratch_dir(const char* name) {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / ("mlc_test_" + std::to_string(::getpid()) + "_" + name);
  std::filesystem::create_directories(dir);
  return dir.string();
}

// Compiles a program and runs it once.
std::vector<std::vector<std::byte>> run_compiled(const Program& p, std::span<const void* const> inputs, int threads, const char* name) {
  std::string dir = scratch_dir(name);
  std::ofstream(dir + "/model.cpp") << codegen::emit_cpp(p);
  codegen::build_shared_library(dir + "/model.cpp", dir + "/model.so");
  std::vector<std::vector<std::byte>> got;
  std::vector<void*> ptrs;
  for (const Type& t : p.outputs) got.emplace_back(t.byte_size());
  for (auto& bytes : got) ptrs.push_back(bytes.data());
  {
    codegen::CompiledModel model(dir + "/model.so", "");
    model.set_threads(threads);
    model.run(inputs.data(), ptrs.data());
  }
  std::filesystem::remove_all(dir);
  return got;
}

}  // namespace

// ---- Dependence analysis ----

TEST(independence_of_matmul_loops) {
  Program p = dot_product_program(4, 6, 5);
  const Kernel& k = p.kernels[0];
  CHECK(is_independent(p, k.body, k.mm_i));  // each i writes its own row of C
  CHECK(is_independent(p, k.body, k.mm_j));  // each j its own column
  std::string why;
  CHECK(!is_independent(p, k.body, k.mm_p, &why));  // every p adds into the same element
  CHECK(why.find("disjoint") != std::string::npos);
}

TEST(independence_sees_a_running_sum) {
  Graph g;
  g.outputs = {g.add(Op::CumSum, {random_f32(g, {3, 8}, 1)}, axis(1))};
  Program p = lower_graph(g, 0, kAtRunTime);
  // for i0 (rows): out[i0,0] = ...; for i1: out[i0, i1+1] = out[i0, i1] + in[i0, i1+1]
  CHECK(is_independent(p, p.kernels[0].body, 0));   // rows do not interact
  CHECK(!is_independent(p, p.kernels[0].body, 1));  // each step reads what the previous one wrote
}

TEST(independence_with_in_place_and_overlapping_buffers) {
  // An op running in place reads and writes one buffer at the same index: fine.
  Graph g;
  ValueId x = g.add(Op::Permute, {random_f32(g, {4, 6}, 1)}, dims({1, 0}));
  g.outputs = {g.add(Op::Permute, {g.add(Op::Tanh, {x})}, dims({1, 0}))};
  Program p = lower_graph(g, 0, kAtRunTime);
  const Kernel& tanh = p.kernels[1];
  CHECK(p.buffers[1].offset == p.buffers[2].offset);  // in place
  CHECK(is_independent(p, tanh.body, 0) && is_independent(p, tanh.body, 1));

  // The same bytes under two shapes: the analysis cannot line the indices up, and says no.
  Program q = p;
  q.buffers[2].shape = {4, 6};
  q.kernels[1].body = make_for(0, 4, make_for(1, 6, make_store(2, {var(0), var(1)}, load(1, DType::F32, {var(1), var(0)}))));
  CHECK(!is_independent(q, q.kernels[1].body, 0));
}

// ---- Rewrites ----

TEST(split_keeps_the_iteration_order) {
  Program p = dot_product_program(7, 6, 5);
  const Kernel& k = p.kernels[0];
  SplitVars v;
  StmtPtr even = split(k.body, k.mm_p, 3, &v);  // 6 = 2 * 3
  CHECK(v.outer >= 0 && v.inner >= 0 && v.tail < 0);
  CHECK(same_after(p, even));
  StmtPtr ragged = split(k.body, k.mm_i, 3, &v);  // 7 = 2 * 3 + 1
  CHECK(v.outer >= 0 && v.tail >= 0);
  CHECK(same_after(p, ragged));
  StmtPtr one_chunk = split(k.body, k.mm_j, 4, &v);  // 5 = 1 * 4 + 1: no outer loop needed
  CHECK(v.outer < 0 && v.inner >= 0 && v.tail >= 0);
  CHECK(same_after(p, one_chunk));
  StmtPtr short_loop = split(k.body, k.mm_j, 9, &v);  // longer than the loop: only renamed
  CHECK(v.outer < 0 && v.tail < 0);
  CHECK(same_after(p, short_loop));
  CHECK_THROWS(split(k.body, 99, 2), ScheduleError);
}

TEST(interchange_swaps_independent_loops_only) {
  Program p = dot_product_program(4, 6, 5);
  const Kernel& k = p.kernels[0];
  StmtPtr swapped = interchange(p, k.body, k.mm_i);
  CHECK(swapped->var == k.mm_j && swapped->body->var == k.mm_i);
  CHECK(same_after(p, swapped));
  // j's body is two statements, not one loop.
  CHECK_THROWS(interchange(p, k.body, k.mm_j), ScheduleError);

  // out[i + j] += 1: iterations of i overlap, so the loops may not be swapped.
  Program q = p;
  q.buffers[2].shape = {20};
  StmtPtr store = make_store(2, {add(var(0), var(1))}, add(load(2, DType::F32, {add(var(0), var(1))}), float_imm(1)));
  q.kernels[0].body = make_for(0, 4, make_for(1, 5, store));
  CHECK_THROWS(interchange(q, q.kernels[0].body, 0), ScheduleError);
}

TEST(unroll_and_jam_builds_a_tile_of_accumulators) {
  Program p = dot_product_program(7, 6, 5);
  const Kernel& k = p.kernels[0];
  StmtPtr rows = unroll_and_jam(p, k.body, k.mm_i, 3);  // 7 rows: two tiles of 3, then 1
  CHECK(same_after(p, rows));
  StmtPtr tile = unroll_and_jam(p, rows, k.mm_j, 2);    // 5 columns: two tiles of 2, then 1
  CHECK(same_after(p, tile));

  Program after = p;
  after.kernels[0].body = tile;
  std::string text = print_kernel(after, after.kernels[0]);
  // Four combinations of (full or leftover rows) x (full or leftover columns),
  // each one loop over p. The stores inside add up to every element once:
  // per pair of row tiles (3 + 3 rows share a loop) and so on.
  CHECK(count(text, "for i" + std::to_string(k.mm_p) + " in 0..6:") == 4);
  CHECK(count(text, " = (v2[") == 3 * 2 + 3 * 1 + 1 * 2 + 1 * 1);
  // The dot-product loop accumulates, so it cannot be jammed into.
  CHECK_THROWS(unroll_and_jam(p, k.body, k.mm_p, 2), ScheduleError);
}

TEST(vectorize_accepts_elementwise_loops_and_reductions) {
  Graph g;
  ValueId x = g.add(Op::Permute, {random_f32(g, {9, 4}, 1)}, dims({1, 0}));  // [4, 9]
  Attrs ln;
  ln.eps = 1e-5;
  g.outputs = {g.add(Op::Gelu, {x}), g.add(Op::LayerNorm, {x, random_f32(g, {9}, 2), random_f32(g, {9}, 3)}, ln),
               g.add(Op::Softmax, {x}, axis(1))};
  Program p = lower_graph(g, 0, kAtRunTime);
  ScheduleOptions only_vectors;
  only_vectors.parallel = false;
  Program s = schedule_program(p, only_vectors);
  std::string text = print_program(s);
  CHECK(count(text, "vector for") == 3);      // gelu; layer norm's last loop; softmax's divide
  CHECK(count(text, "vector(x4) for") == 4);  // one accumulator each: layer norm's mean and variance; softmax's maximum, and its exp-and-sum
  CHECK(run_loops(p) == run_loops(s));            // the interpreter runs a vector loop serially: same bits

  // The plan says which stores are which.
  const Stmt& exp_loop = *s.kernels[3].body->body->stmts[3];
  std::optional<VectorPlan> plan = plan_vector_loop(s, exp_loop);
  CHECK(plan && plan->stores.size() == 2 && !plan->stores[0].reduction && plan->stores[1].reduction);
  // Without vector math the loop that calls exp stays scalar.
  std::string why;
  CHECK(!plan_vector_loop(s, exp_loop, /*vector_math=*/false, &why) && why.find("exp") != std::string::npos);
}

TEST(vectorize_rejects_what_lanes_cannot_do) {
  Graph g;
  ValueId x = random_f32(g, {3, 8}, 1);
  ValueId where = g.add(Op::Where, {g.add(Op::Le, {x, g.add_scalar(DType::F32, 0.0)}), x, g.add(Op::Tanh, {x})});
  g.outputs = {g.add(Op::CumSum, {x}, axis(1)), g.add(Op::Permute, {x}, dims({1, 0})), where, g.add(Op::Pow, {x, g.add_scalar(DType::F32, 2.0)})};
  Program p = lower_graph(g, 0, kAtRunTime);
  auto reason = [&](size_t kernel, VarId v) {
    try {
      vectorize(p, p.kernels[kernel].body, v);
    } catch (const ScheduleError& e) {
      return std::string(e.what());
    }
    return std::string("accepted");
  };
  CHECK(reason(0, 1).find("non-float") != std::string::npos);           // le: produces bools
  CHECK(reason(1, 1) == "accepted");                                    // tanh
  CHECK(reason(2, 1).find("select") != std::string::npos);              // where
  CHECK(reason(3, 1).find("another one writes") != std::string::npos);  // cumsum: a loop-carried dependence
  CHECK(reason(4, 1).find("not consecutive") != std::string::npos);     // permute: the read strides across rows
  CHECK(reason(5, 1).find("pow") != std::string::npos);
  // Nothing was vectorized by a failed attempt, and the default schedule leaves these loops alone.
  Program s = schedule_program(p);
  CHECK(count(print_program(s), "vector") == 1);
}

TEST(parallelize_needs_independent_iterations) {
  Graph g;
  g.outputs = {g.add(Op::CumSum, {random_f32(g, {3, 8}, 1)}, axis(1))};
  Program p = lower_graph(g, 0, kAtRunTime);
  StmtPtr rows = parallelize(p, p.kernels[0].body, 0);
  CHECK(rows->loop_kind == LoopKind::Parallel);
  CHECK_THROWS(parallelize(p, p.kernels[0].body, 1), ScheduleError);
  CHECK_THROWS(parallelize(p, rows, 0), ScheduleError);  // already scheduled
}

TEST(default_schedule_tiles_the_matrix_product) {
  Program p = dot_product_program(9, 16, 23);
  ScheduleOptions o;
  o.tile_rows = 2;
  o.tile_cols = 3;
  o.choose_tile = false;
  o.block_bytes = 512;   // 16 floats per row: blocks of a few rows
  o.parallel_work = 1;   // parallel even at this size
  Program s = schedule_program(p, o);
  std::string text = print_kernel(s, s.kernels[0]);
  CHECK(text.find("parallel for") != std::string::npos);
  CHECK(text.find("vector for") != std::string::npos);
  // A full tile carries six accumulators and is not interleaved. The corner
  // left over by both tiles (1 row by 2 columns) has only two, so it is.
  CHECK(count(text, "vector for") >= 4 && count(text, "vector(x4) for") == 1);
  CHECK(run_loops(p) == run_loops(s));

  // Each switch off in turn still gives the same answer.
  for (bool ScheduleOptions::*flag : {&ScheduleOptions::vectorize, &ScheduleOptions::regtile, &ScheduleOptions::cachetile, &ScheduleOptions::parallel}) {
    ScheduleOptions off = o;
    off.*flag = false;
    CHECK(run_loops(p) == run_loops(schedule_program(p, off)));
  }
  // The tile is chosen to leave little over: 4x5 leaves 3 of 23 columns, 5x4 leaves 3; 9 rows fit neither.
  ScheduleOptions choose;
  choose.parallel = false;
  CHECK(run_loops(p) == run_loops(schedule_program(p, choose)));
}

// ---- Layout and staging ----

TEST(layout_pass_cancels_transposes) {
  Graph g;
  ValueId q = g.add_input({DType::F32, {2, 5, 4}}, "q");
  ValueId k = g.add_input({DType::F32, {2, 5, 4}}, "k");
  ValueId scale = g.add_scalar(DType::F32, 0.5);
  ValueId kt = g.add(Op::Mul, {g.add(Op::Permute, {k}, dims({0, 2, 1})), scale});  // transpose, then scale, as GPT-2 exports it
  ValueId scores = g.add(Op::MatMul, {q, kt});                                     // [2,5,5]
  ValueId table = random_f32(g, {7, 4}, 1);
  ValueId flat = g.add(Op::Reshape, {q}, dims({10, 4}));
  ValueId logits = g.add(Op::MatMul, {flat, g.add(Op::Permute, {table}, dims({1, 0}))});  // the tied output projection
  ValueId dense = g.add(Op::Gemm, {flat, random_f32(g, {4, 3}, 2), random_f32(g, {3}, 3)});
  g.outputs = {scores, logits, dense};

  Graph out = eliminate_dead_code(transpose_matmul_operands(g));
  check_graph(out);
  auto count_op = [&](Op op) {
    size_t n = 0;
    for (const Node& node : out.nodes) n += node.op == op;
    return n;
  };
  // Both existing transposes are gone; the only one left is of the dense layer's weight.
  CHECK(count_op(Op::Permute) == 1);
  for (const Node& node : out.nodes)
    if (node.op == Op::MatMul || node.op == Op::Gemm) CHECK(node.attrs.trans_b);

  std::vector<float> qd = noise(40, 5), kd = noise(40, 6);
  std::vector<Tensor> in = {Tensor::borrow({DType::F32, {2, 5, 4}}, reinterpret_cast<const std::byte*>(qd.data())),
                            Tensor::borrow({DType::F32, {2, 5, 4}}, reinterpret_cast<const std::byte*>(kd.data()))};
  std::vector<Tensor> want = run_graph(g, in, {}), got = run_graph(out, in, {});
  for (size_t i = 0; i < want.size(); ++i) CHECK(same_values(want[i].type, want[i].ptr, got[i].ptr, 1e-5));

  // The weight's transpose depends on no input, so it is computed at load time.
  Program p = lower_graph(out, 0);
  CHECK(p.init_kernels.size() == 1 && p.init_kernels[0].op == "permute");
  CHECK(p.persistent_bytes == 64);
  CHECK(lower_graph(out, 0, kAtRunTime).init_kernels.empty());
  const void* inputs[] = {qd.data(), kd.data()};
  std::vector<std::vector<std::byte>> via_loops = run_loops(schedule_program(p), inputs);
  for (size_t i = 0; i < want.size(); ++i) CHECK(same_values(want[i].type, want[i].ptr, via_loops[i].data(), 1e-5));
}

TEST(staging_keeps_load_time_values_out_of_the_callers_memory) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {2, 3}}, "x");
  ValueId table = g.add(Op::Tanh, {random_f32(g, {2, 3}, 1)});  // known at load time
  g.outputs = {g.add(Op::Add, {x, table}), table};               // and also returned as is
  Program p = lower_graph(g, 0);
  CHECK(p.init_kernels.size() == 1);
  CHECK(p.kernels.size() == 2 && p.kernels[1].op == "copy");  // the load-time output is copied out on every call
  CHECK(verify_program(p).empty());

  // A per-call kernel may not write persistent memory, and an init kernel may not read an input.
  Program bad = p;
  std::swap(bad.init_kernels[0], bad.kernels[0]);
  CHECK(verify_program(bad).size() >= 2);
}

// ---- Generated code ----

TEST(codegen_spells_out_vector_and_parallel_loops) {
  Program p = dot_product_program(8, 16, 10);
  ScheduleOptions o;
  o.parallel_work = 1;
  std::string cpp = codegen::emit_cpp(schedule_program(p, o));
  CHECK(cpp.find("#include <arm_neon.h>") != std::string::npos);
  CHECK(cpp.find("float* __restrict v2") != std::string::npos);
  CHECK(cpp.find("mlc_parallel_for(") != std::string::npos);
  CHECK(cpp.find("vfmaq_f32(acc") != std::string::npos);
  CHECK(cpp.find("vaddvq_f32(acc") != std::string::npos);
  // 16 is a multiple of the vector width: no leftover loop is emitted.
  CHECK(cpp.find("; ++i2)") == std::string::npos);
  CHECK(codegen::emit_cpp(schedule_program(p, o), {/*noalias=*/false}).find("__restrict") == std::string::npos);
  // Unscheduled code has neither intrinsics nor threads.
  std::string plain = codegen::emit_cpp(p);
  CHECK(plain.find("arm_neon") == std::string::npos && plain.find("mlc_parallel_for(") == std::string::npos);
}

TEST(compiled_kernels_match_the_graph_interpreter_at_awkward_sizes) {
  // Every size here is chosen to leave something over: 7 rows against tiles
  // of 2, 11 columns against tiles of 3, 13 and 9 elements against 4 lanes.
  Graph g;
  ValueId a = g.add_input({DType::F32, {7, 13}}, "a");
  ValueId b = g.add_input({DType::F32, {13, 11}}, "b");
  ValueId ba = g.add_input({DType::F32, {3, 9, 9}}, "ba");
  ValueId bb = g.add_input({DType::F32, {3, 9, 6}}, "bb");
  Attrs ln;
  ln.eps = 1e-5;
  Attrs causal = axis(2);
  causal.causal = true;
  ValueId wide = g.add(Op::Mul, {a, g.add_scalar(DType::F32, 6.0)});  // values out to +-6, where tanh saturates
  g.outputs = {
      g.add(Op::MatMul, {a, b}),                                                          // [7,11]
      g.add(Op::MatMul, {ba, bb}),                                                        // batched, [3,9,6]
      g.add(Op::Gemm, {a, random_f32(g, {13, 5}, 1), random_f32(g, {5}, 2)}),             // constant weight: transposed at load time
      g.add(Op::LayerNorm, {a, random_f32(g, {13}, 3), random_f32(g, {13}, 4)}, ln),
      g.add(Op::Softmax, {ba}, axis(2)),
      g.add(Op::Softmax, {ba}, causal),
      g.add(Op::Gelu, {wide}),
      g.add(Op::Tanh, {wide}),
      g.add(Op::Sub, {g.add(Op::Add, {a, wide}), g.add(Op::Mul, {a, a})}),
  };
  Graph laid_out = eliminate_dead_code(transpose_matmul_operands(g));
  Program naive = lower_graph(laid_out, 0);
  CHECK(naive.init_kernels.size() == 1);

  std::vector<std::vector<float>> data = {noise(7 * 13, 1), noise(13 * 11, 2), noise(3 * 9 * 9, 3), noise(3 * 9 * 6, 4)};
  std::vector<const void*> inputs;
  std::vector<Tensor> tensors;
  for (size_t i = 0; i < data.size(); ++i) {
    inputs.push_back(data[i].data());
    tensors.push_back(Tensor::borrow(g.type_of(g.inputs[i]), reinterpret_cast<const std::byte*>(data[i].data())));
  }
  std::vector<Tensor> want = run_graph(g, tensors, {});

  ScheduleOptions o;
  o.tile_rows = 2;
  o.tile_cols = 3;
  o.choose_tile = false;
  o.block_bytes = 256;
  o.parallel_work = 1;
  Program scheduled = schedule_program(naive, o);
  std::vector<std::vector<std::byte>> one = run_compiled(scheduled, inputs, 1, "zoo1"), three = run_compiled(scheduled, inputs, 3, "zoo3");
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(same_values(want[i].type, want[i].ptr, one[i].data(), 2e-5));
    // Threads only divide the work; each element is computed by the same instructions.
    CHECK(one[i] == three[i]);
  }
  // The default parameters, and the unscheduled program, agree as well.
  std::vector<std::vector<std::byte>> defaults = run_compiled(schedule_program(naive), inputs, 2, "zoo_default");
  std::vector<std::vector<std::byte>> plain = run_compiled(naive, inputs, 1, "zoo_plain");
  for (size_t i = 0; i < want.size(); ++i) {
    CHECK(same_values(want[i].type, want[i].ptr, defaults[i].data(), 2e-5));
    CHECK(same_values(want[i].type, want[i].ptr, plain[i].data(), 2e-5));
  }
}

#if defined(__aarch64__)
TEST(vector_exp_and_tanh_accuracy) {
  auto exp4 = [](float x) { return vgetq_lane_f32(mlc_exp_f32x4(vdupq_n_f32(x)), 2); };
  auto tanh4 = [](float x) { return vgetq_lane_f32(mlc_tanh_f32x4(vdupq_n_f32(x)), 1); };
  double worst_exp = 0, worst_tanh = 0;
  for (int i = -87000; i <= 88000; ++i) {
    float x = static_cast<float>(i) / 1000.0f;
    worst_exp = std::max(worst_exp, std::fabs(static_cast<double>(exp4(x)) / std::exp(static_cast<double>(x)) - 1.0));
  }
  for (int i = -120000; i <= 120000; ++i) {
    float x = static_cast<float>(i) / 10000.0f;
    worst_tanh = std::max(worst_tanh, std::fabs(static_cast<double>(tanh4(x)) - std::tanh(static_cast<double>(x))));
  }
  CHECK(worst_exp < 2.5e-7);   // relative
  CHECK(worst_tanh < 2.5e-7);  // absolute
  // The documented behaviour at the ends of the range.
  CHECK(exp4(0.0f) == 1.0f);
  CHECK(exp4(-1000.0f) > 0.0f && exp4(-1000.0f) < 2e-38f);
  CHECK(std::isfinite(exp4(1000.0f)) && exp4(1000.0f) > 1e38f);
  CHECK(std::isnan(exp4(NAN)));
  CHECK(tanh4(50.0f) == 1.0f && tanh4(-50.0f) == -1.0f && tanh4(0.0f) == 0.0f);
}
#endif
