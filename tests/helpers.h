// Shorthand for building small graphs with inline constants and running them.
#pragma once

#include <cmath>
#include <cstring>
#include <vector>

#include "graph/interpreter.h"
#include "graph/ir.h"
#include "loop/ir.h"

namespace mlc::test {

template <class T>
ValueId constant(Graph& g, DType dtype, Shape shape, const std::vector<T>& values) {
  Constant c;
  c.type = {dtype, std::move(shape)};
  c.inline_data.resize(values.size() * sizeof(T));
  std::memcpy(c.inline_data.data(), values.data(), c.inline_data.size());
  return g.add_const(std::move(c));
}

inline ValueId f32(Graph& g, Shape shape, const std::vector<float>& v) { return constant<float>(g, DType::F32, std::move(shape), v); }
inline ValueId i64(Graph& g, Shape shape, const std::vector<int64_t>& v) { return constant<int64_t>(g, DType::I64, std::move(shape), v); }
inline ValueId boolean(Graph& g, Shape shape, const std::vector<uint8_t>& v) { return constant<uint8_t>(g, DType::Bool, std::move(shape), v); }

// Runs a graph that needs no inputs or weights and returns `out` as a flat vector.
template <class T>
std::vector<T> eval(Graph& g, ValueId out) {
  g.outputs = {out};
  Tensor t = run_graph(g, {}, {}).at(0);
  return std::vector<T>(t.data<T>(), t.data<T>() + t.size());
}

inline Attrs axis(int64_t a) {
  Attrs attrs;
  attrs.axis = a;
  return attrs;
}

inline Attrs dims(std::vector<int64_t> d) {
  Attrs attrs;
  attrs.dims = std::move(d);
  return attrs;
}

// Deterministic pseudo-random floats in [-1, 1), so tests need no seed plumbing.
inline std::vector<float> noise(size_t n, uint32_t seed = 1) {
  std::vector<float> out(n);
  uint32_t state = seed * 2654435761u + 12345u;
  for (float& v : out) {
    state = state * 1664525u + 1013904223u;  // a linear congruential generator
    v = static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
  }
  return out;
}

inline ValueId random_f32(Graph& g, Shape shape, uint32_t seed) {
  size_t n = static_cast<size_t>(num_elements(shape));
  return f32(g, std::move(shape), noise(n, seed));
}

// True when two buffers of the given type hold the same values: exactly for
// ints and bools, within `tol` for floats (NaN matches NaN, inf matches inf).
inline bool same_values(const Type& type, const std::byte* a, const std::byte* b, double tol) {
  if (type.dtype != DType::F32) return std::memcmp(a, b, type.byte_size()) == 0;
  const float* fa = reinterpret_cast<const float*>(a);
  const float* fb = reinterpret_cast<const float*>(b);
  for (int64_t i = 0, n = num_elements(type.shape); i < n; ++i) {
    if (std::isnan(fa[i]) && std::isnan(fb[i])) continue;
    if (fa[i] == fb[i]) continue;
    if (!(std::fabs(fa[i] - fb[i]) <= tol)) return false;
  }
  return true;
}

// These tests feed kernels from inline constants, where a real model has
// inputs. With staging on, a value computed from constants alone moves to
// load time; this option keeps it in the per-call kernels, where the test
// can look at it.
inline constexpr loop::LowerOptions kAtRunTime{true, false};

// Runs a lowered program in the loop interpreter and returns its outputs.
inline std::vector<std::vector<std::byte>> run_loops(const loop::Program& p, std::span<const void* const> inputs = {}) {
  std::vector<std::vector<std::byte>> got;
  std::vector<void*> ptrs;
  for (const Type& t : p.outputs) got.emplace_back(t.byte_size());
  for (auto& bytes : got) ptrs.push_back(bytes.data());
  loop::run_program(p, {}, inputs, ptrs);
  return got;
}

// The differential test for lowering: run the graph in the graph interpreter,
// lower it and run the loops in the loop interpreter, and compare every output.
// The graph must take no inputs and no external weights. Checked twice: with
// everything computed per call, and with staging moving it to load time.
inline bool lowering_matches(Graph& g, std::vector<ValueId> outs, double tol = 1e-5) {
  g.outputs = std::move(outs);
  std::vector<Tensor> want = run_graph(g, {}, {});
  for (bool hoist : {false, true}) {
    std::vector<std::vector<std::byte>> got = run_loops(loop::lower_graph(g, 0, {true, hoist}));
    for (size_t i = 0; i < want.size(); ++i)
      if (!same_values(want[i].type, want[i].ptr, got[i].data(), tol)) return false;
  }
  return true;
}

// One attention block and one MLP block at toy size (T=5, width 8, 2 heads),
// built from the same ops in the same arrangement as the GPT-2 export:
// embedding, layer norm, gemm, split, head reshape/permute, scaled batched
// matmul, causal mask, softmax, head merge, residual, tanh GELU, projection.
// `ids` is an i64[1,5] value: a constant or a graph input.
inline ValueId tiny_transformer(Graph& g, ValueId ids) {
  const int64_t T = 5, C = 8, H = 2, D = C / H, V = 11;
  auto slice = [&](ValueId x, int64_t ax, int64_t start, int64_t end) {
    Attrs a = axis(ax);
    a.start = start;
    a.end = end;
    return g.add(Op::Slice, {x}, a);
  };
  auto heads = [&](ValueId x) {  // [1,T,C] -> [H,T,D]
    ValueId split = g.add(Op::Reshape, {x}, dims({1, T, H, D}));
    return g.add(Op::Reshape, {g.add(Op::Permute, {split}, dims({0, 2, 1, 3}))}, dims({H, T, D}));
  };
  auto norm = [&](ValueId x, uint32_t seed) {
    Attrs a;
    a.eps = 1e-5;
    return g.add(Op::LayerNorm, {x, random_f32(g, {C}, seed), random_f32(g, {C}, seed + 1)}, a);
  };
  auto linear = [&](ValueId x, int64_t in, int64_t out, uint32_t seed) {  // [1,T,in] -> [1,T,out]
    ValueId flat = g.add(Op::Reshape, {x}, dims({T, in}));
    ValueId y = g.add(Op::Gemm, {flat, random_f32(g, {in, out}, seed), random_f32(g, {out}, seed + 1)});
    return g.add(Op::Reshape, {y}, dims({1, T, out}));
  };

  ValueId table = random_f32(g, {V, C}, 1);
  ValueId pos = i64(g, {T}, {0, 1, 2, 3, 4});
  ValueId x = g.add(Op::Add, {g.add(Op::Gather, {table, ids}), g.add(Op::Gather, {random_f32(g, {T, C}, 2), pos})});

  // Attention.
  ValueId qkv = linear(norm(x, 10), C, 3 * C, 20);
  ValueId q = heads(slice(qkv, 2, 0, C)), k = heads(slice(qkv, 2, C, 2 * C)), v = heads(slice(qkv, 2, 2 * C, 3 * C));
  ValueId scale = g.add_scalar(DType::F32, 1.0 / std::sqrt(std::sqrt(static_cast<double>(D))));
  ValueId kt = g.add(Op::Permute, {k}, dims({0, 2, 1}));
  ValueId scores = g.add(Op::MatMul, {g.add(Op::Mul, {q, scale}), g.add(Op::Mul, {kt, scale})});  // [H,T,T]
  ValueId allowed = g.add(Op::Le, {g.add(Op::Reshape, {pos}, dims({1, T})), g.add(Op::Reshape, {pos}, dims({T, 1}))});
  ValueId mask = g.add(Op::Where, {allowed, g.add_scalar(DType::F32, 0.0), g.add_scalar(DType::F32, -INFINITY)});
  ValueId probs = g.add(Op::Softmax, {g.add(Op::Add, {scores, mask})}, axis(2));
  ValueId mixed = g.add(Op::MatMul, {probs, v});  // [H,T,D]
  ValueId merged = g.add(Op::Permute, {g.add(Op::Reshape, {mixed}, dims({1, H, T, D}))}, dims({0, 2, 1, 3}));
  x = g.add(Op::Add, {x, linear(g.add(Op::Reshape, {merged}, dims({1, T, C})), C, C, 30)});

  // MLP with the tanh form of GELU: 0.5 * h * (1 + tanh(sqrt(2/pi) * (h + 0.044715 * h^3))).
  ValueId h = linear(norm(x, 40), C, 4 * C, 50);
  ValueId cube = g.add(Op::Pow, {h, g.add_scalar(DType::F32, 3.0)});
  ValueId inner = g.add(Op::Mul, {g.add(Op::Add, {h, g.add(Op::Mul, {cube, g.add_scalar(DType::F32, 0.044715)})}),
                                  g.add_scalar(DType::F32, 0.7978845608028654)});
  ValueId gelu = g.add(Op::Mul, {g.add(Op::Mul, {h, g.add_scalar(DType::F32, 0.5)}),
                                 g.add(Op::Add, {g.add(Op::Tanh, {inner}), g.add_scalar(DType::F32, 1.0)})});
  x = g.add(Op::Add, {x, linear(gelu, 4 * C, C, 60)});

  // Output projection through the transposed embedding table, as GPT-2 ties them.
  ValueId flat = g.add(Op::Reshape, {norm(x, 70)}, dims({T, C}));
  return g.add(Op::Reshape, {g.add(Op::MatMul, {flat, g.add(Op::Permute, {table}, dims({1, 0}))})}, dims({1, T, V}));
}

}  // namespace mlc::test
