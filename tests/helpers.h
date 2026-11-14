// Shorthand for building small graphs with inline constants and running them.
#pragma once

#include <cstring>
#include <vector>

#include "graph/interpreter.h"
#include "graph/ir.h"

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

}  // namespace mlc::test
