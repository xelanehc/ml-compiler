// Graph IR -> loop IR. One naive loop nest per node, one buffer per value.
#include <cstring>
#include <functional>
#include <limits>

#include "loop/ir.h"

namespace mlc::loop {

namespace {

using Idx = std::vector<ExprPtr>;

constexpr int64_t kAlign = 64;  // every arena buffer starts on a cache-line boundary

class Lowerer {
 public:
  Lowerer(const Graph& g, int64_t weights_bytes) : g_(g) { p_.weights_bytes = weights_bytes; }

  Program run() {
    check_graph(g_);
    for (ValueId in : g_.inputs) p_.inputs.push_back(g_.type_of(in));
    for (ValueId out : g_.outputs) p_.outputs.push_back(g_.type_of(out));
    assign_storage();
    for (ValueId v = 0; v < g_.nodes.size(); ++v) lower_node(v);
    copy_unowned_outputs();
    check_program(p_);
    return std::move(p_);
  }

 private:
  const Graph& g_;
  Program p_;
  std::vector<BufferId> buffer_of_;       // graph value -> its buffer
  std::vector<int> copy_outputs_;         // outputs that need an explicit copy at the end
  VarId next_var_ = 0;

  // ---- Storage ----

  int64_t arena_alloc(const Type& type) {
    int64_t offset = p_.arena_bytes;
    int64_t bytes = static_cast<int64_t>(type.byte_size());
    p_.arena_bytes = (offset + bytes + kAlign - 1) / kAlign * kAlign;
    return offset;
  }

  BufferId new_buffer(std::string name, const Type& type, Storage storage, int64_t offset, int32_t index) {
    p_.buffers.push_back(Buffer{std::move(name), type.dtype, type.shape, storage, offset, index});
    return static_cast<BufferId>(p_.buffers.size() - 1);
  }

  BufferId arena_buffer(std::string name, const Type& type) {
    int64_t offset = arena_alloc(type);
    return new_buffer(std::move(name), type, Storage::Arena, offset, -1);
  }

  // Decides where every value lives. A Reshape does not get storage of its
  // own: it is the same bytes as its operand under a different shape, so it
  // shares the operand's storage and costs nothing at run time. `root` follows
  // such chains back to the value that actually owns the bytes.
  //
  // Everything else is as simple as it can be: each computed value gets a
  // fresh slice of the arena that is never reused. The arena is therefore as
  // large as the sum of all intermediates. Memory planning (M3) shrinks it.
  void assign_storage() {
    const size_t n = g_.nodes.size();
    std::vector<ValueId> root(n);
    for (ValueId v = 0; v < n; ++v) root[v] = g_.nodes[v].op == Op::Reshape ? root[g_.nodes[v].inputs[0]] : v;

    // A value the caller wants back is computed straight into the caller's
    // buffer when possible. It is not possible when the value is an input or
    // a constant, or when two outputs are the same value; those get a copy.
    std::vector<int> output_of(n, -1);
    for (size_t k = 0; k < g_.outputs.size(); ++k) {
      ValueId r = root[g_.outputs[k]];
      Op op = g_.nodes[r].op;
      if (op != Op::Input && op != Op::Const && output_of[r] < 0) output_of[r] = static_cast<int>(k);
      else copy_outputs_.push_back(static_cast<int>(k));
    }

    struct Place {
      Storage storage;
      int64_t offset;
      int32_t index;
    };
    std::vector<Place> place(n);
    buffer_of_.resize(n);
    for (ValueId v = 0; v < n; ++v) {
      const Node& node = g_.nodes[v];
      if (root[v] != v) {
        place[v] = place[root[v]];
      } else if (node.op == Op::Input) {
        place[v] = {Storage::Input, 0, node.attrs.index};
      } else if (node.op == Op::Const) {
        const Constant& c = g_.constants[node.attrs.index];
        if (c.is_external()) {
          place[v] = {Storage::Weights, c.weights_offset, -1};
        } else {
          p_.inline_data.push_back(c.inline_data);
          place[v] = {Storage::Inline, 0, static_cast<int32_t>(p_.inline_data.size() - 1)};
        }
      } else if (output_of[v] >= 0) {
        place[v] = {Storage::Output, 0, output_of[v]};
      } else {
        place[v] = {Storage::Arena, arena_alloc(node.type), -1};
      }
      buffer_of_[v] = new_buffer("v" + std::to_string(v), node.type, place[v].storage, place[v].offset, place[v].index);
    }
  }

  // ---- Building blocks ----

  // One loop per axis of `shape`, outermost first, around whatever `body`
  // builds from the loop indices. An axis of size 1 gets the constant index 0
  // and no loop.
  StmtPtr nest(const Shape& shape, const std::function<StmtPtr(const Idx&)>& body) {
    Idx idx;
    std::vector<VarId> vars(shape.size(), -1);
    for (size_t d = 0; d < shape.size(); ++d) {
      if (shape[d] == 1) {
        idx.push_back(int_imm(0));
      } else {
        vars[d] = next_var_++;
        idx.push_back(var(vars[d]));
      }
    }
    StmtPtr s = body(idx);
    for (size_t d = shape.size(); d-- > 0;)
      if (vars[d] >= 0) s = make_for(vars[d], shape[d], s);
    return s;
  }

  StmtPtr loop(int64_t extent, const std::function<StmtPtr(const ExprPtr&)>& body) {
    if (extent == 1) return body(int_imm(0));
    VarId v = next_var_++;
    return make_for(v, extent, body(var(v)));
  }

  // Reads graph value `v` at a full index. A rank-0 inline constant is emitted
  // as a literal: `x * 0.5` rather than a load from a one-element table.
  ExprPtr read(ValueId v, const Idx& idx) {
    const Node& n = g_.nodes[v];
    if (n.op == Op::Const && n.type.shape.empty()) {
      const Constant& c = g_.constants[n.attrs.index];
      if (!c.is_external()) {
        switch (c.type.dtype) {
          case DType::F32: { float x; std::memcpy(&x, c.inline_data.data(), sizeof x); return float_imm(x); }
          case DType::I64: { int64_t x; std::memcpy(&x, c.inline_data.data(), sizeof x); return int_imm(x); }
          case DType::Bool: return bool_imm(c.inline_data[0] != std::byte{0});
        }
      }
    }
    return load(buffer_of_[v], n.type.dtype, idx);
  }

  // Reads `v` at an index of a larger, broadcast shape. The value's axes line
  // up with the last axes of the index; where the value has size 1 it is read
  // at 0 whatever the loop index is. This is the stride-0 trick from the graph
  // interpreter, done at compile time: the loop variable simply does not
  // appear in the access.
  ExprPtr read_broadcast(ValueId v, const Idx& out_idx) {
    const Shape& shape = g_.type_of(v).shape;
    size_t shift = out_idx.size() - shape.size();
    Idx idx;
    for (size_t d = 0; d < shape.size(); ++d) idx.push_back(shape[d] == 1 ? int_imm(0) : out_idx[d + shift]);
    return read(v, idx);
  }

  StmtPtr write(ValueId v, const Idx& idx, ExprPtr value) { return make_store(buffer_of_[v], idx, std::move(value)); }

  // A scratch buffer for an op that needs a temporary (a running maximum, a mean).
  BufferId temp(ValueId v, const char* what, DType dtype, const Shape& shape) {
    return arena_buffer("t" + std::to_string(v) + "_" + what, {dtype, shape});
  }

  static Idx concat_idx(const Idx& a, const ExprPtr& b) {
    Idx out = a;
    out.push_back(b);
    return out;
  }

  static Idx insert_at(const Idx& idx, size_t at, const ExprPtr& e) {
    Idx out = idx;
    out.insert(out.begin() + static_cast<std::ptrdiff_t>(at), e);
    return out;
  }

  // ---- Per-op lowering ----

  void emit(ValueId v, StmtPtr body) {
    const Node& n = g_.nodes[v];
    std::string label = "%" + std::to_string(v);
    if (!n.name.empty()) label += " " + n.name;
    p_.kernels.push_back(Kernel{op_name(n.op), std::move(label), std::move(body)});
  }

  // out[i...] = f(a[i...], b[i...]) with broadcasting.
  StmtPtr elementwise(ValueId v, const std::function<ExprPtr(const std::vector<ExprPtr>&)>& f) {
    const Node& n = g_.nodes[v];
    return nest(n.type.shape, [&](const Idx& idx) {
      std::vector<ExprPtr> operands;
      for (ValueId in : n.inputs) operands.push_back(read_broadcast(in, idx));
      return write(v, idx, f(operands));
    });
  }

  StmtPtr binary_op(ValueId v, BinOp op) {
    return elementwise(v, [op](const std::vector<ExprPtr>& x) { return binary(op, x[0], x[1]); });
  }

  StmtPtr permute(ValueId v) {
    const Node& n = g_.nodes[v];
    return nest(n.type.shape, [&](const Idx& idx) {
      // Output axis d is input axis dims[d], so the input is read at the
      // loop indices shuffled back into the input's axis order.
      Idx src(idx.size());
      for (size_t d = 0; d < idx.size(); ++d) src[n.attrs.dims[d]] = idx[d];
      return write(v, idx, read(n.inputs[0], src));
    });
  }

  StmtPtr slice(ValueId v) {
    const Node& n = g_.nodes[v];
    return nest(n.type.shape, [&](const Idx& idx) {
      Idx src = idx;
      src[n.attrs.axis] = add(src[n.attrs.axis], int_imm(n.attrs.start));
      return write(v, idx, read(n.inputs[0], src));
    });
  }

  // One loop nest per input, each writing its own stretch of the concat axis.
  StmtPtr concat(ValueId v) {
    const Node& n = g_.nodes[v];
    std::vector<StmtPtr> parts;
    int64_t offset = 0;
    for (ValueId in : n.inputs) {
      const Shape& shape = g_.type_of(in).shape;
      parts.push_back(nest(shape, [&](const Idx& idx) {
        Idx dst = idx;
        dst[n.attrs.axis] = add(dst[n.attrs.axis], int_imm(offset));
        return write(v, dst, read(in, idx));
      }));
      offset += shape[n.attrs.axis];
    }
    return make_block(std::move(parts));
  }

  // out[b..., t...] = x[i0[b...], ..., ik-1[b...], t...]. The index into x is
  // itself loaded from memory: an indirect access, which is what makes gather
  // different from every other layout op.
  StmtPtr gather(ValueId v) {
    const Node& n = g_.nodes[v];
    const size_t k = n.inputs.size() - 1;
    const size_t tail = g_.type_of(n.inputs[0]).shape.size() - k;
    const size_t b_rank = n.type.shape.size() - tail;
    return nest(n.type.shape, [&](const Idx& idx) {
      Idx b(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(b_rank));
      Idx src;
      for (size_t i = 0; i < k; ++i) src.push_back(read_broadcast(n.inputs[i + 1], b));
      src.insert(src.end(), idx.begin() + static_cast<std::ptrdiff_t>(b_rank), idx.end());
      return write(v, idx, read(n.inputs[0], src));
    });
  }

  // out[.., 0, ..] = in[.., 0, ..];  out[.., j+1, ..] = out[.., j, ..] + in[.., j+1, ..]
  // The second loop reads what its previous iteration wrote. That is a
  // loop-carried dependence: its iterations cannot be reordered or run in
  // parallel, and the loop IR shows why (a load and a store of the same buffer).
  StmtPtr cumsum(ValueId v) {
    const Node& n = g_.nodes[v];
    const size_t axis = static_cast<size_t>(n.attrs.axis);
    Shape rest = n.type.shape;
    const int64_t dim = rest[axis];
    rest.erase(rest.begin() + n.attrs.axis);
    ValueId in = n.inputs[0];
    return nest(rest, [&](const Idx& idx) {
      Idx first = insert_at(idx, axis, int_imm(0));
      std::vector<StmtPtr> body = {write(v, first, read(in, first))};
      if (dim > 1) {
        body.push_back(loop(dim - 1, [&](const ExprPtr& j) {
          Idx prev = insert_at(idx, axis, j);
          Idx cur = insert_at(idx, axis, add(j, int_imm(1)));
          return write(v, cur, add(load(buffer_of_[v], n.type.dtype, prev), read(in, cur)));
        }));
      }
      return make_block(std::move(body));
    });
  }

  // out[o] = false;  for j: out[o] = out[o] | in[o with j on the axis]
  StmtPtr reduce_any(ValueId v) {
    const Node& n = g_.nodes[v];
    const size_t axis = static_cast<size_t>(n.attrs.axis);
    const Shape& in_shape = g_.type_of(n.inputs[0]).shape;
    Shape rest = in_shape;
    rest.erase(rest.begin() + n.attrs.axis);
    return nest(rest, [&](const Idx& idx) {
      Idx out = n.attrs.keepdims ? insert_at(idx, axis, int_imm(0)) : idx;
      return make_block({
          write(v, out, bool_imm(false)),
          loop(in_shape[axis], [&](const ExprPtr& j) {
            return write(v, out, binary(BinOp::Or, load(buffer_of_[v], DType::Bool, out), read(n.inputs[0], insert_at(idx, axis, j))));
          }),
      });
    });
  }

  // C[b.., i, j] = init;  then  C[b.., i, j] += A[b.., i, p] * B[b.., p, j].
  //
  // Loop order is i, p, j, the same order the graph interpreter uses: the
  // inner loop walks a row of B and a row of C, both contiguous. The textbook
  // order i, j, p would be just as correct and far slower. Choosing an order
  // is unavoidable (a loop nest has to have one); improving on it is M4.
  StmtPtr matmul_like(ValueId v, const std::function<ExprPtr(const ExprPtr& j)>& init) {
    const Node& n = g_.nodes[v];
    ValueId a = n.inputs[0], b = n.inputs[1];
    const Shape& as = g_.type_of(a).shape;
    const int64_t m = as[as.size() - 2], k = as.back(), cols = n.type.shape.back();
    Shape batch(as.begin(), as.end() - 2);
    return nest(batch, [&](const Idx& bi) {
      return loop(m, [&](const ExprPtr& i) {
        Idx row = concat_idx(bi, i);
        return make_block({
            loop(cols, [&](const ExprPtr& j) { return write(v, concat_idx(row, j), init(j)); }),
            loop(k, [&](const ExprPtr& p) {
              return loop(cols, [&](const ExprPtr& j) {
                Idx c = concat_idx(row, j);
                ExprPtr product = mul(read(a, concat_idx(row, p)), read(b, concat_idx(concat_idx(bi, p), j)));
                return write(v, c, add(load(buffer_of_[v], DType::F32, c), product));
              });
            }),
        });
      });
    });
  }

  // Per row r of the input (all axes but the last), with d = row length:
  //   mean[r] = sum_i x[r,i] / d
  //   rstd[r] = 1 / sqrt(sum_i (x[r,i] - mean[r])^2 / d + eps)
  //   out[r,i] = (x[r,i] - mean[r]) * rstd[r] * weight[i] + bias[i]
  // mean and rstd are temporaries, one element per row.
  StmtPtr layer_norm(ValueId v) {
    const Node& n = g_.nodes[v];
    ValueId x = n.inputs[0], weight = n.inputs[1], bias = n.inputs[2];
    Shape rows(n.type.shape.begin(), n.type.shape.end() - 1);
    const int64_t d = n.type.shape.back();
    BufferId mean = temp(v, "mean", DType::F32, rows);
    BufferId rstd = temp(v, "rstd", DType::F32, rows);
    ExprPtr inv_d = float_imm(1.0 / static_cast<double>(d));
    return nest(rows, [&](const Idx& r) {
      ExprPtr mean_r = load(mean, DType::F32, r);
      ExprPtr rstd_r = load(rstd, DType::F32, r);
      auto centered = [&](const ExprPtr& i) { return sub(read(x, concat_idx(r, i)), mean_r); };
      return make_block({
          make_store(mean, r, float_imm(0)),
          loop(d, [&](const ExprPtr& i) { return make_store(mean, r, add(mean_r, read(x, concat_idx(r, i)))); }),
          make_store(mean, r, mul(mean_r, inv_d)),
          make_store(rstd, r, float_imm(0)),
          loop(d, [&](const ExprPtr& i) { return make_store(rstd, r, add(rstd_r, mul(centered(i), centered(i)))); }),
          make_store(rstd, r, div(float_imm(1), call(Intrinsic::Sqrt, {add(mul(rstd_r, inv_d), float_imm(n.attrs.eps))}))),
          loop(d, [&](const ExprPtr& i) {
            ExprPtr wi = read(weight, {i}), bi = read(bias, {i});
            return write(v, concat_idx(r, i), add(mul(mul(centered(i), rstd_r), wi), bi));
          }),
      });
    });
  }

  // Per vector along the axis:
  //   max = max_j x[j];  out[j] = exp(x[j] - max);  sum = sum_j out[j];  out[j] = out[j] / sum
  StmtPtr softmax(ValueId v) {
    const Node& n = g_.nodes[v];
    const size_t axis = static_cast<size_t>(n.attrs.axis);
    Shape rest = n.type.shape;
    const int64_t dim = rest[axis];
    rest.erase(rest.begin() + n.attrs.axis);
    ValueId x = n.inputs[0];
    BufferId max = temp(v, "max", DType::F32, rest);
    BufferId sum = temp(v, "sum", DType::F32, rest);
    return nest(rest, [&](const Idx& r) {
      ExprPtr max_r = load(max, DType::F32, r);
      ExprPtr sum_r = load(sum, DType::F32, r);
      auto at = [&](const ExprPtr& j) { return insert_at(r, axis, j); };
      auto out = [&](const ExprPtr& j) { return load(buffer_of_[v], DType::F32, at(j)); };
      return make_block({
          make_store(max, r, float_imm(-std::numeric_limits<double>::infinity())),
          loop(dim, [&](const ExprPtr& j) { return make_store(max, r, binary(BinOp::Max, max_r, read(x, at(j)))); }),
          make_store(sum, r, float_imm(0)),
          loop(dim, [&](const ExprPtr& j) {
            return make_block({
                write(v, at(j), call(Intrinsic::Exp, {sub(read(x, at(j)), max_r)})),
                make_store(sum, r, add(sum_r, out(j))),
            });
          }),
          loop(dim, [&](const ExprPtr& j) { return write(v, at(j), div(out(j), sum_r)); }),
      });
    });
  }

  void lower_node(ValueId v) {
    const Node& n = g_.nodes[v];
    next_var_ = 0;  // loop variable names restart in every kernel
    using X = const std::vector<ExprPtr>&;
    switch (n.op) {
      case Op::Input:
      case Op::Const:
      case Op::Reshape:
        return;  // no code: the value already exists in memory

      case Op::Add: return emit(v, binary_op(v, BinOp::Add));
      case Op::Sub: return emit(v, binary_op(v, BinOp::Sub));
      case Op::Mul: return emit(v, binary_op(v, BinOp::Mul));
      case Op::Eq: return emit(v, binary_op(v, BinOp::Eq));
      case Op::Ne: return emit(v, binary_op(v, BinOp::Ne));
      case Op::Le: return emit(v, binary_op(v, BinOp::Le));
      case Op::And: return emit(v, binary_op(v, BinOp::And));
      case Op::Pow: return emit(v, elementwise(v, [](X x) { return call(Intrinsic::Pow, {x[0], x[1]}); }));
      case Op::Tanh: return emit(v, elementwise(v, [](X x) { return call(Intrinsic::Tanh, {x[0]}); }));
      case Op::Not: return emit(v, elementwise(v, [](X x) { return logical_not(x[0]); }));
      case Op::Where: return emit(v, elementwise(v, [](X x) { return select(x[0], x[1], x[2]); }));
      case Op::Cast: return emit(v, elementwise(v, [&](X x) { return cast(n.attrs.to, x[0]); }));
      case Op::Broadcast: return emit(v, elementwise(v, [](X x) { return x[0]; }));

      case Op::Permute: return emit(v, permute(v));
      case Op::Slice: return emit(v, slice(v));
      case Op::Concat: return emit(v, concat(v));
      case Op::Gather: return emit(v, gather(v));
      case Op::CumSum: return emit(v, cumsum(v));
      case Op::ReduceAny: return emit(v, reduce_any(v));

      case Op::MatMul: return emit(v, matmul_like(v, [](const ExprPtr&) { return float_imm(0); }));
      case Op::Gemm: return emit(v, matmul_like(v, [&](const ExprPtr& j) { return read(n.inputs[2], {j}); }));
      case Op::LayerNorm: return emit(v, layer_norm(v));
      case Op::Softmax: return emit(v, softmax(v));
    }
    throw LoopError(std::string("no lowering for op ") + op_name(n.op));
  }

  void copy_unowned_outputs() {
    for (int k : copy_outputs_) {
      ValueId v = g_.outputs[k];
      const Type& type = g_.type_of(v);
      BufferId out = new_buffer("out" + std::to_string(k), type, Storage::Output, 0, k);
      next_var_ = 0;
      StmtPtr body = nest(type.shape, [&](const Idx& idx) { return make_store(out, idx, read(v, idx)); });
      p_.kernels.push_back(Kernel{"copy", "output " + std::to_string(k), std::move(body)});
    }
  }
};

}  // namespace

Program lower_graph(const Graph& g, int64_t weights_bytes) { return Lowerer(g, weights_bytes).run(); }

}  // namespace mlc::loop
