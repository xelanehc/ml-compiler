#include "graph/interpreter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>

namespace mlc {

namespace {

// ---- Memory -------------------------------------------------------------------

// Counts live bytes so a run can report its peak. Each buffer's deleter
// subtracts its size, so "live" is exact even when tensors share a buffer.
struct MemCounter {
  size_t live = 0, peak = 0, total = 0;
};
using Mem = std::shared_ptr<MemCounter>;

Tensor allocate(const Type& type, const Mem& mem) {
  size_t n = type.byte_size();
  mem->live += n;
  mem->total += n;
  mem->peak = std::max(mem->peak, mem->live);
  std::shared_ptr<std::byte[]> buf(new std::byte[std::max<size_t>(n, 1)], [mem, n](std::byte* p) {
    delete[] p;
    mem->live -= n;
  });
  const std::byte* ptr = buf.get();
  return Tensor{type, std::move(buf), ptr};
}

// Write access to a tensor we just allocated.
template <class T>
T* out_data(Tensor& t) { return reinterpret_cast<T*>(t.owner.get()); }

// Calls f with a value whose C++ type is the storage type for `d`, so one
// generic lambda can be instantiated for float, int64_t, and bool (uint8_t).
template <class F>
auto by_dtype(DType d, F&& f) {
  switch (d) {
    case DType::F32: return f(float{});
    case DType::I64: return f(int64_t{});
    case DType::Bool: return f(uint8_t{});
  }
  throw GraphError("unknown dtype");
}

// ---- Indexing -----------------------------------------------------------------
//
// A tensor is stored row-major: element [i0, i1, ..., in] lives at linear
// position i0*s0 + i1*s1 + ... + in*sn, where the stride s_d is the product
// of the sizes of all axes after d. The last axis has stride 1.

std::vector<int64_t> strides_of(const Shape& shape) {
  std::vector<int64_t> s(shape.size());
  int64_t step = 1;
  for (size_t d = shape.size(); d-- > 0;) {
    s[d] = step;
    step *= shape[d];
  }
  return s;
}

// Strides for reading `in` as though it had the broadcast shape `out`. Along
// an axis where `in` has size 1 (or no axis at all) the stride is 0, so every
// step along that output axis re-reads the same input element. That is all
// broadcasting is.
std::vector<int64_t> broadcast_strides(const Shape& in, const Shape& out) {
  std::vector<int64_t> own = strides_of(in);
  std::vector<int64_t> s(out.size(), 0);
  size_t shift = out.size() - in.size();
  for (size_t d = 0; d < in.size(); ++d)
    if (in[d] != 1) s[d + shift] = own[d];
  return s;
}

// Visits every index of `shape` in row-major order, like an odometer. For
// each of several operands it maintains the linear offset that the current
// index corresponds to under that operand's strides, and calls
// body(output_position, offsets). Every elementwise and layout op below is
// this loop with a different body.
template <class Body>
void strided_loop(const Shape& shape, const std::vector<std::vector<int64_t>>& strides, Body&& body) {
  const size_t rank = shape.size();
  const size_t n = strides.size();
  const int64_t total = num_elements(shape);
  std::vector<int64_t> index(rank, 0);
  std::vector<int64_t> offset(n, 0);
  for (int64_t pos = 0; pos < total; ++pos) {
    body(pos, offset.data());
    // Advance the odometer: bump the last axis; on overflow reset it and carry.
    for (size_t d = rank; d-- > 0;) {
      for (size_t k = 0; k < n; ++k) offset[k] += strides[k][d];
      if (++index[d] < shape[d]) break;
      for (size_t k = 0; k < n; ++k) offset[k] -= strides[k][d] * shape[d];
      index[d] = 0;
    }
  }
}

// ---- Elementwise ----------------------------------------------------------------

template <class In, class Out, class F>
Tensor binary(const Tensor& a, const Tensor& b, const Type& type, const Mem& mem, F f) {
  Tensor out = allocate(type, mem);
  const In* pa = a.data<In>();
  const In* pb = b.data<In>();
  Out* po = out_data<Out>(out);
  strided_loop(type.shape, {broadcast_strides(a.type.shape, type.shape), broadcast_strides(b.type.shape, type.shape)},
               [&](int64_t pos, const int64_t* off) { po[pos] = f(pa[off[0]], pb[off[1]]); });
  return out;
}

template <class In, class Out, class F>
Tensor unary(const Tensor& a, const Type& type, const Mem& mem, F f) {
  Tensor out = allocate(type, mem);
  const In* pa = a.data<In>();
  Out* po = out_data<Out>(out);
  for (int64_t i = 0, n = a.size(); i < n; ++i) po[i] = f(pa[i]);
  return out;
}

// Add/Sub/Mul: the result has the operand dtype.
template <class F>
Tensor arithmetic(const Tensor& a, const Tensor& b, const Type& type, const Mem& mem, F f) {
  return by_dtype(a.type.dtype, [&](auto tag) {
    using T = decltype(tag);
    return binary<T, T>(a, b, type, mem, f);
  });
}

// Eq/Ne/Le: the result is bool whatever the operand dtype.
template <class F>
Tensor comparison(const Tensor& a, const Tensor& b, const Type& type, const Mem& mem, F f) {
  return by_dtype(a.type.dtype, [&](auto tag) {
    using T = decltype(tag);
    return binary<T, uint8_t>(a, b, type, mem, [&](T x, T y) { return static_cast<uint8_t>(f(x, y)); });
  });
}

Tensor where(const Tensor& cond, const Tensor& a, const Tensor& b, const Type& type, const Mem& mem) {
  return by_dtype(type.dtype, [&](auto tag) {
    using T = decltype(tag);
    Tensor out = allocate(type, mem);
    const uint8_t* pc = cond.data<uint8_t>();
    const T* pa = a.data<T>();
    const T* pb = b.data<T>();
    T* po = out_data<T>(out);
    strided_loop(type.shape,
                 {broadcast_strides(cond.type.shape, type.shape), broadcast_strides(a.type.shape, type.shape),
                  broadcast_strides(b.type.shape, type.shape)},
                 [&](int64_t pos, const int64_t* off) { po[pos] = pc[off[0]] ? pa[off[1]] : pb[off[2]]; });
    return out;
  });
}

Tensor cast(const Tensor& a, const Type& type, const Mem& mem) {
  return by_dtype(a.type.dtype, [&](auto from) {
    return by_dtype(type.dtype, [&](auto to) {
      using S = decltype(from);
      using D = decltype(to);
      // uint8_t only ever stands for bool here, and casting to bool means "is nonzero".
      return unary<S, D>(a, type, mem, [](S x) { return std::is_same_v<D, uint8_t> ? static_cast<D>(x != 0) : static_cast<D>(x); });
    });
  });
}

// ---- Layout -----------------------------------------------------------------------

// out[pos] = in[base + offset(pos)]: permute, broadcast, and slice are all
// this, differing only in which strides they walk the input with.
Tensor copy_strided(const Tensor& a, const Type& type, const Mem& mem, std::vector<int64_t> strides, int64_t base) {
  return by_dtype(type.dtype, [&](auto tag) {
    using T = decltype(tag);
    Tensor out = allocate(type, mem);
    const T* pa = a.data<T>() + base;
    T* po = out_data<T>(out);
    strided_loop(type.shape, {strides}, [&](int64_t pos, const int64_t* off) { po[pos] = pa[off[0]]; });
    return out;
  });
}

Tensor permute(const Tensor& a, const Node& n, const Mem& mem) {
  // Output axis d is input axis dims[d], so stepping along output axis d
  // moves through the input by that input axis's stride.
  std::vector<int64_t> in_strides = strides_of(a.type.shape);
  std::vector<int64_t> strides(n.attrs.dims.size());
  for (size_t d = 0; d < strides.size(); ++d) strides[d] = in_strides[n.attrs.dims[d]];
  return copy_strided(a, n.type, mem, std::move(strides), 0);
}

Tensor slice(const Tensor& a, const Node& n, const Mem& mem) {
  std::vector<int64_t> strides = strides_of(a.type.shape);
  return copy_strided(a, n.type, mem, strides, n.attrs.start * strides[n.attrs.axis]);
}

Tensor concat(std::span<const Tensor> ins, const Node& n, const Mem& mem) {
  // Split the index space at the concat axis: `outer` counts the positions
  // before it, and each input contributes one contiguous run per position.
  Tensor out = allocate(n.type, mem);
  const size_t item = dtype_size(n.type.dtype);
  const Shape& shape = n.type.shape;
  int64_t outer = 1, inner = 1;
  for (int64_t d = 0; d < n.attrs.axis; ++d) outer *= shape[d];
  for (size_t d = n.attrs.axis + 1; d < shape.size(); ++d) inner *= shape[d];
  std::byte* dst = out.owner.get();
  for (int64_t o = 0; o < outer; ++o) {
    for (const Tensor& t : ins) {
      size_t run = static_cast<size_t>(t.type.shape[n.attrs.axis] * inner) * item;
      std::memcpy(dst, t.ptr + static_cast<size_t>(o) * run, run);
      dst += run;
    }
  }
  return out;
}

Tensor gather(std::span<const Tensor> ins, const Node& n, const Mem& mem) {
  const Tensor& x = ins[0];
  const size_t k = ins.size() - 1;  // number of indexed axes
  const Shape& xs = x.type.shape;
  std::vector<int64_t> x_strides = strides_of(xs);

  // The output shape is B (the broadcast of the index shapes) followed by the
  // un-indexed tail of x. Each position in B selects one whole tail.
  Shape b_shape(n.type.shape.begin(), n.type.shape.end() - static_cast<std::ptrdiff_t>(xs.size() - k));
  int64_t tail = 1;
  for (size_t d = k; d < xs.size(); ++d) tail *= xs[d];
  const size_t tail_bytes = static_cast<size_t>(tail) * dtype_size(x.type.dtype);

  std::vector<std::vector<int64_t>> strides;
  for (size_t i = 0; i < k; ++i) strides.push_back(broadcast_strides(ins[i + 1].type.shape, b_shape));

  Tensor out = allocate(n.type, mem);
  std::byte* dst = out.owner.get();
  strided_loop(b_shape, strides, [&](int64_t pos, const int64_t* off) {
    int64_t src = 0;
    for (size_t i = 0; i < k; ++i) {
      int64_t idx = ins[i + 1].data<int64_t>()[off[i]];
      if (idx < 0 || idx >= xs[i])
        throw GraphError("gather: index " + std::to_string(idx) + " out of range for axis of size " + std::to_string(xs[i]));
      src += idx * x_strides[i];
    }
    std::memcpy(dst + static_cast<size_t>(pos) * tail_bytes, x.ptr + static_cast<size_t>(src) * dtype_size(x.type.dtype), tail_bytes);
  });
  return out;
}

// ---- Reductions and scans ------------------------------------------------------------

Tensor cumsum(const Tensor& a, const Node& n, const Mem& mem) {
  return by_dtype(n.type.dtype, [&](auto tag) {
    using T = decltype(tag);
    Tensor out = allocate(n.type, mem);
    const T* pa = a.data<T>();
    T* po = out_data<T>(out);
    // `step` is the distance between neighbours along the axis; a `block` is
    // one full extent of the axis. Within a block, each element adds the
    // running total sitting one step behind it.
    const int64_t step = strides_of(n.type.shape)[n.attrs.axis];
    const int64_t block = step * n.type.shape[n.attrs.axis];
    for (int64_t base = 0, total = a.size(); base < total; base += block)
      for (int64_t i = 0; i < block; ++i) po[base + i] = pa[base + i] + (i >= step ? po[base + i - step] : T{});
    return out;
  });
}

Tensor reduce_any(const Tensor& a, const Node& n, const Mem& mem) {
  Tensor out = allocate(n.type, mem);
  const uint8_t* pa = a.data<uint8_t>();
  uint8_t* po = out_data<uint8_t>(out);
  const Shape& shape = a.type.shape;
  const int64_t dim = shape[n.attrs.axis];
  const int64_t inner = strides_of(shape)[n.attrs.axis];
  const int64_t outer = a.size() / std::max<int64_t>(dim * inner, 1);
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      uint8_t any = 0;
      for (int64_t j = 0; j < dim; ++j) any |= pa[(o * dim + j) * inner + i];
      po[o * inner + i] = any;
    }
  }
  return out;
}

// ---- Linear algebra -----------------------------------------------------------------

// C += A * B for row-major A (m x k), B (k x n), C (m x n).
//
// The definition is C[i][j] = sum over p of A[i][p] * B[p][j], which suggests
// loops in the order i, j, p. We use i, p, j instead. With j innermost, the
// loop reads a row of B and updates a row of C, both contiguous in memory, and
// A[i][p] is a constant for the whole inner loop. The i, j, p order would walk
// down a *column* of B, touching a new cache line on every step. Same
// arithmetic, same result up to rounding, several times faster, and this is
// the first loop interchange of the project. The real GEMM work is M4.
void matmul_accumulate(const float* A, const float* B, float* C, int64_t m, int64_t k, int64_t n) {
  for (int64_t i = 0; i < m; ++i) {
    float* c = C + i * n;
    for (int64_t p = 0; p < k; ++p) {
      const float a = A[i * k + p];
      const float* b = B + p * n;
      for (int64_t j = 0; j < n; ++j) c[j] += a * b[j];
    }
  }
}

Tensor matmul(const Tensor& a, const Tensor& b, const Node& node, const Mem& mem) {
  Tensor out = allocate(node.type, mem);
  const Shape& as = a.type.shape;
  const Shape& bs = b.type.shape;
  const int64_t m = as[as.size() - 2], k = as.back(), n = bs.back();
  const int64_t batch = a.size() / std::max<int64_t>(m * k, 1);
  float* po = out_data<float>(out);
  std::fill(po, po + out.size(), 0.0f);
  for (int64_t i = 0; i < batch; ++i)
    matmul_accumulate(a.data<float>() + i * m * k, b.data<float>() + i * k * n, po + i * m * n, m, k, n);
  return out;
}

Tensor gemm(const Tensor& a, const Tensor& b, const Tensor& bias, const Node& node, const Mem& mem) {
  Tensor out = allocate(node.type, mem);
  const int64_t m = a.type.shape[0], k = a.type.shape[1], n = b.type.shape[1];
  float* po = out_data<float>(out);
  // Start every output row as a copy of the bias, then accumulate the product on top.
  for (int64_t i = 0; i < m; ++i) std::copy_n(bias.data<float>(), n, po + i * n);
  matmul_accumulate(a.data<float>(), b.data<float>(), po, m, k, n);
  return out;
}

// ---- Composites ----------------------------------------------------------------------

// For each vector x along the last axis (length d):
//   mean = (1/d) sum x_i          var = (1/d) sum (x_i - mean)^2
//   y_i  = (x_i - mean) / sqrt(var + eps) * weight_i + bias_i
// The variance divides by d, not d - 1. Sums are accumulated in double so the
// oracle's own rounding error stays well below the tolerance it is used to check.
Tensor layer_norm(const Tensor& x, const Tensor& weight, const Tensor& bias, const Node& n, const Mem& mem) {
  Tensor out = allocate(n.type, mem);
  const int64_t d = x.type.shape.back();
  const int64_t rows = d ? x.size() / d : 0;
  const float* px = x.data<float>();
  const float* pw = weight.data<float>();
  const float* pb = bias.data<float>();
  float* po = out_data<float>(out);
  for (int64_t r = 0; r < rows; ++r) {
    const float* v = px + r * d;
    double sum = 0;
    for (int64_t i = 0; i < d; ++i) sum += v[i];
    const double mean = sum / static_cast<double>(d);
    double sq = 0;
    for (int64_t i = 0; i < d; ++i) sq += (v[i] - mean) * (v[i] - mean);
    const double inv_std = 1.0 / std::sqrt(sq / static_cast<double>(d) + n.attrs.eps);
    for (int64_t i = 0; i < d; ++i) po[r * d + i] = static_cast<float>((v[i] - mean) * inv_std) * pw[i] + pb[i];
  }
  return out;
}

// softmax(x)_j = exp(x_j) / sum_i exp(x_i), computed as
//   exp(x_j - max) / sum_i exp(x_i - max).
// Subtracting the maximum changes nothing mathematically (the factor exp(-max)
// cancels) but keeps every exponent at or below 0, so exp cannot overflow.
// A masked position holds -inf and contributes exp(-inf) = 0, which is how the
// causal mask removes future tokens. A row that is entirely -inf produces NaN
// (0/0), the same as PyTorch; the exported graph guards that case with a Where.
Tensor softmax(const Tensor& x, const Node& n, const Mem& mem) {
  Tensor out = allocate(n.type, mem);
  const Shape& shape = x.type.shape;
  const int64_t dim = shape[n.attrs.axis];
  const int64_t inner = strides_of(shape)[n.attrs.axis];
  const int64_t outer = x.size() / std::max<int64_t>(dim * inner, 1);
  const float* px = x.data<float>();
  float* po = out_data<float>(out);
  for (int64_t o = 0; o < outer; ++o) {
    for (int64_t i = 0; i < inner; ++i) {
      const int64_t base = o * dim * inner + i;
      // Causal: the axis is the last one (inner == 1), so `o` counts rows of
      // every [T, T] matrix in turn, and o % dim is the row's position i.
      // Row i takes part in columns 0..i; the rest are defined to be 0.
      const int64_t live = n.attrs.causal ? o % dim + 1 : dim;
      float max = -std::numeric_limits<float>::infinity();
      for (int64_t j = 0; j < live; ++j) max = std::max(max, px[base + j * inner]);
      double sum = 0;
      for (int64_t j = 0; j < live; ++j) {
        float e = std::exp(px[base + j * inner] - max);
        po[base + j * inner] = e;
        sum += e;
      }
      for (int64_t j = 0; j < live; ++j) po[base + j * inner] = static_cast<float>(po[base + j * inner] / sum);
      for (int64_t j = live; j < dim; ++j) po[base + j * inner] = 0.0f;
    }
  }
  return out;
}

// ---- Dispatch --------------------------------------------------------------------------

Tensor execute(const Graph& g, const Node& n, std::span<const Tensor> in, std::span<const Tensor> graph_inputs,
               std::span<const std::byte> weights, const Mem& mem) {
  switch (n.op) {
    case Op::Input:
      return graph_inputs[n.attrs.index];

    case Op::Const: {
      const Constant& c = g.constants[n.attrs.index];
      if (!c.is_external()) return Tensor::borrow(c.type, c.inline_data.data());
      if (static_cast<size_t>(c.weights_offset) + c.type.byte_size() > weights.size())
        throw GraphError("constant " + n.name + " extends past the end of the weights file");
      return Tensor::borrow(c.type, weights.data() + c.weights_offset);
    }

    case Op::Add: return arithmetic(in[0], in[1], n.type, mem, [](auto x, auto y) { return x + y; });
    case Op::Sub: return arithmetic(in[0], in[1], n.type, mem, [](auto x, auto y) { return x - y; });
    case Op::Mul: return arithmetic(in[0], in[1], n.type, mem, [](auto x, auto y) { return x * y; });
    case Op::Pow: return binary<float, float>(in[0], in[1], n.type, mem, [](float x, float y) { return std::pow(x, y); });
    case Op::Tanh: return unary<float, float>(in[0], n.type, mem, [](float x) { return std::tanh(x); });
    case Op::Gelu:
      return unary<float, float>(in[0], n.type, mem, [](float x) {
        return 0.5f * x * (1.0f + std::tanh(0.7978845608028654f * (x + 0.044715f * x * x * x)));
      });
    case Op::Eq: return comparison(in[0], in[1], n.type, mem, [](auto x, auto y) { return x == y; });
    case Op::Ne: return comparison(in[0], in[1], n.type, mem, [](auto x, auto y) { return x != y; });
    case Op::Le: return comparison(in[0], in[1], n.type, mem, [](auto x, auto y) { return x <= y; });
    case Op::And:
      return binary<uint8_t, uint8_t>(in[0], in[1], n.type, mem, [](uint8_t x, uint8_t y) { return static_cast<uint8_t>(x & y); });
    case Op::Not: return unary<uint8_t, uint8_t>(in[0], n.type, mem, [](uint8_t x) { return static_cast<uint8_t>(!x); });
    case Op::Where: return where(in[0], in[1], in[2], n.type, mem);
    case Op::Cast: return cast(in[0], n.type, mem);

    case Op::Reshape:
      // Same bytes, new shape: share the buffer instead of copying it.
      return Tensor{n.type, in[0].owner, in[0].ptr};
    case Op::Permute: return permute(in[0], n, mem);
    case Op::Broadcast: return copy_strided(in[0], n.type, mem, broadcast_strides(in[0].type.shape, n.type.shape), 0);
    case Op::Slice: return slice(in[0], n, mem);
    case Op::Concat: return concat(in, n, mem);
    case Op::Gather: return gather(in, n, mem);

    case Op::CumSum: return cumsum(in[0], n, mem);
    case Op::ReduceAny: return reduce_any(in[0], n, mem);

    case Op::MatMul: return matmul(in[0], in[1], n, mem);
    case Op::Gemm: return gemm(in[0], in[1], in[2], n, mem);
    case Op::LayerNorm: return layer_norm(in[0], in[1], in[2], n, mem);
    case Op::Softmax: return softmax(in[0], n, mem);
  }
  throw GraphError("interpreter: unknown op");
}

}  // namespace

std::vector<Tensor> run_graph(const Graph& g, std::span<const Tensor> inputs, std::span<const std::byte> weights, RunStats* stats) {
  check_graph(g);
  if (inputs.size() != g.inputs.size())
    throw GraphError("graph takes " + std::to_string(g.inputs.size()) + " inputs, got " + std::to_string(inputs.size()));
  for (size_t i = 0; i < inputs.size(); ++i)
    if (inputs[i].type != g.type_of(g.inputs[i]))
      throw GraphError("input " + std::to_string(i) + " has type " + type_string(inputs[i].type) + ", graph expects " +
                       type_string(g.type_of(g.inputs[i])));

  // Liveness, the simple way: count how many times each value will be read.
  // When the count reaches zero nothing later can need the value, so its
  // buffer is released at once. Without this, T=1024 would hold every
  // intermediate of all twelve layers at the same time.
  std::vector<int> uses_left(g.nodes.size(), 0);
  for (const Node& n : g.nodes)
    for (ValueId in : n.inputs) ++uses_left[in];
  for (ValueId out : g.outputs) ++uses_left[out];

  Mem mem = std::make_shared<MemCounter>();
  std::vector<Tensor> values(g.nodes.size());
  std::vector<Tensor> args;

  for (ValueId v = 0; v < g.nodes.size(); ++v) {
    const Node& n = g.nodes[v];
    args.clear();
    for (ValueId in : n.inputs) args.push_back(values[in]);

    auto t0 = std::chrono::steady_clock::now();
    values[v] = execute(g, n, args, inputs, weights, mem);
    if (stats) {
      stats->seconds[static_cast<int>(n.op)] += std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      ++stats->calls[static_cast<int>(n.op)];
    }

    args.clear();  // drop our extra references before deciding what is dead
    for (ValueId in : n.inputs)
      if (--uses_left[in] == 0) values[in] = Tensor{};
    if (uses_left[v] == 0) values[v] = Tensor{};  // computed but never read
  }

  std::vector<Tensor> outputs;
  for (ValueId out : g.outputs) outputs.push_back(values[out]);
  if (stats) {
    stats->allocated_bytes = mem->total;
    stats->peak_bytes = mem->peak;
  }
  return outputs;
}

}  // namespace mlc
