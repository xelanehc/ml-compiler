#include "graph/ir.h"

#include <algorithm>
#include <cstring>

namespace mlc {

size_t dtype_size(DType d) {
  switch (d) {
    case DType::F32: return 4;
    case DType::I64: return 8;
    case DType::Bool: return 1;
  }
  return 0;
}

const char* dtype_name(DType d) {
  switch (d) {
    case DType::F32: return "f32";
    case DType::I64: return "i64";
    case DType::Bool: return "bool";
  }
  return "?";
}

int64_t num_elements(const Shape& s) {
  int64_t n = 1;  // the empty product: a scalar has one element
  for (int64_t d : s) n *= d;
  return n;
}

std::string shape_string(const Shape& s) {
  std::string out = "[";
  for (size_t i = 0; i < s.size(); ++i) {
    if (i) out += ",";
    out += std::to_string(s[i]);
  }
  return out + "]";
}

std::string type_string(const Type& t) { return dtype_name(t.dtype) + shape_string(t.shape); }

Shape broadcast_shapes(const Shape& a, const Shape& b) {
  size_t rank = std::max(a.size(), b.size());
  Shape out(rank);
  for (size_t i = 0; i < rank; ++i) {
    // Walk both shapes from the right; a shape that runs out counts as 1.
    int64_t da = i < a.size() ? a[a.size() - 1 - i] : 1;
    int64_t db = i < b.size() ? b[b.size() - 1 - i] : 1;
    if (da != db && da != 1 && db != 1)
      throw GraphError("cannot broadcast " + shape_string(a) + " with " + shape_string(b));
    out[rank - 1 - i] = da == 1 ? db : da;
  }
  return out;
}

const char* op_name(Op op) {
  switch (op) {
    case Op::Input: return "input";
    case Op::Const: return "const";
    case Op::Add: return "add";
    case Op::Sub: return "sub";
    case Op::Mul: return "mul";
    case Op::Pow: return "pow";
    case Op::Tanh: return "tanh";
    case Op::Eq: return "eq";
    case Op::Ne: return "ne";
    case Op::Le: return "le";
    case Op::And: return "and";
    case Op::Not: return "not";
    case Op::Where: return "where";
    case Op::Cast: return "cast";
    case Op::Reshape: return "reshape";
    case Op::Permute: return "permute";
    case Op::Broadcast: return "broadcast";
    case Op::Slice: return "slice";
    case Op::Concat: return "concat";
    case Op::Gather: return "gather";
    case Op::CumSum: return "cumsum";
    case Op::ReduceAny: return "reduce_any";
    case Op::MatMul: return "matmul";
    case Op::Gemm: return "gemm";
    case Op::LayerNorm: return "layer_norm";
    case Op::Softmax: return "softmax";
  }
  return "?";
}

// ---- Type inference ---------------------------------------------------------

namespace {

// Small helpers so each rule below reads as a list of requirements.
struct Rule {
  Op op;
  std::span<const Type> in;

  [[noreturn]] void fail(const std::string& why) const {
    std::string msg = std::string(op_name(op)) + "(";
    for (size_t i = 0; i < in.size(); ++i) msg += (i ? ", " : "") + type_string(in[i]);
    throw GraphError(msg + "): " + why);
  }
  void require(bool ok, const char* why) const {
    if (!ok) fail(why);
  }
  void arity(size_t n) const {
    if (in.size() != n) fail("expected " + std::to_string(n) + " inputs");
  }
  void axis_in_range(int64_t axis, size_t rank) const {
    if (axis < 0 || axis >= static_cast<int64_t>(rank))
      fail("axis " + std::to_string(axis) + " out of range for rank " + std::to_string(rank));
  }
  Shape broadcast(const Shape& a, const Shape& b) const {
    try {
      return broadcast_shapes(a, b);
    } catch (const GraphError& e) {
      fail(e.what());
    }
  }
};

}  // namespace

Type infer_type(Op op, std::span<const Type> in, const Attrs& attrs) {
  Rule r{op, in};
  switch (op) {
    case Op::Input:
    case Op::Const:
      throw GraphError(std::string(op_name(op)) + " has no inferable type; use add_input / add_const");

    case Op::Add:
    case Op::Sub:
    case Op::Mul:
      r.arity(2);
      r.require(in[0].dtype == in[1].dtype, "operands must have the same dtype");
      r.require(in[0].dtype != DType::Bool, "arithmetic on bool is not defined");
      return {in[0].dtype, r.broadcast(in[0].shape, in[1].shape)};

    case Op::Pow:
      r.arity(2);
      r.require(in[0].dtype == DType::F32 && in[1].dtype == DType::F32, "operands must be f32");
      return {DType::F32, r.broadcast(in[0].shape, in[1].shape)};

    case Op::Tanh:
      r.arity(1);
      r.require(in[0].dtype == DType::F32, "operand must be f32");
      return in[0];

    case Op::Eq:
    case Op::Ne:
    case Op::Le:
      r.arity(2);
      r.require(in[0].dtype == in[1].dtype, "operands must have the same dtype");
      return {DType::Bool, r.broadcast(in[0].shape, in[1].shape)};

    case Op::And:
      r.arity(2);
      r.require(in[0].dtype == DType::Bool && in[1].dtype == DType::Bool, "operands must be bool");
      return {DType::Bool, r.broadcast(in[0].shape, in[1].shape)};

    case Op::Not:
      r.arity(1);
      r.require(in[0].dtype == DType::Bool, "operand must be bool");
      return in[0];

    case Op::Where:
      r.arity(3);
      r.require(in[0].dtype == DType::Bool, "condition must be bool");
      r.require(in[1].dtype == in[2].dtype, "branches must have the same dtype");
      return {in[1].dtype, r.broadcast(in[0].shape, r.broadcast(in[1].shape, in[2].shape))};

    case Op::Cast:
      r.arity(1);
      return {attrs.to, in[0].shape};

    case Op::Reshape: {
      r.arity(1);
      for (int64_t d : attrs.dims) r.require(d >= 0, "target dims must be non-negative");
      if (num_elements(attrs.dims) != num_elements(in[0].shape))
        r.fail("cannot reshape to " + shape_string(attrs.dims) + ": element counts differ");
      return {in[0].dtype, attrs.dims};
    }

    case Op::Permute: {
      r.arity(1);
      size_t rank = in[0].shape.size();
      r.require(attrs.dims.size() == rank, "permutation length must equal the rank");
      std::vector<bool> seen(rank, false);
      Shape out(rank);
      for (size_t i = 0; i < rank; ++i) {
        int64_t src = attrs.dims[i];
        r.axis_in_range(src, rank);
        r.require(!seen[src], "permutation repeats an axis");
        seen[src] = true;
        out[i] = in[0].shape[src];
      }
      return {in[0].dtype, out};
    }

    case Op::Broadcast: {
      r.arity(1);
      // Legal exactly when broadcasting the input against the target gives the target.
      if (r.broadcast(in[0].shape, attrs.dims) != attrs.dims)
        r.fail("cannot broadcast to " + shape_string(attrs.dims));
      return {in[0].dtype, attrs.dims};
    }

    case Op::Slice: {
      r.arity(1);
      r.axis_in_range(attrs.axis, in[0].shape.size());
      int64_t dim = in[0].shape[attrs.axis];
      r.require(0 <= attrs.start && attrs.start <= attrs.end && attrs.end <= dim, "slice bounds out of range");
      Shape out = in[0].shape;
      out[attrs.axis] = attrs.end - attrs.start;
      return {in[0].dtype, out};
    }

    case Op::Concat: {
      r.require(!in.empty(), "expected at least one input");
      size_t rank = in[0].shape.size();
      r.axis_in_range(attrs.axis, rank);
      Shape out = in[0].shape;
      out[attrs.axis] = 0;
      for (const Type& t : in) {
        r.require(t.dtype == in[0].dtype, "inputs must have the same dtype");
        r.require(t.shape.size() == rank, "inputs must have the same rank");
        for (size_t d = 0; d < rank; ++d)
          if (static_cast<int64_t>(d) != attrs.axis)
            r.require(t.shape[d] == in[0].shape[d], "inputs must agree on every axis but the concat axis");
        out[attrs.axis] += t.shape[attrs.axis];
      }
      return {in[0].dtype, out};
    }

    case Op::Gather: {
      // x[i0, ..., ik-1]: k index tensors address the first k axes of x. They
      // broadcast together to a shape B, and the result is B followed by the
      // axes of x that were not indexed. One index tensor is an embedding lookup.
      r.require(in.size() >= 2, "expected a tensor and at least one index");
      size_t k = in.size() - 1;
      r.require(k <= in[0].shape.size(), "more indices than axes");
      Shape out;
      for (size_t i = 1; i <= k; ++i) {
        r.require(in[i].dtype == DType::I64, "indices must be i64");
        out = r.broadcast(out, in[i].shape);
      }
      out.insert(out.end(), in[0].shape.begin() + static_cast<std::ptrdiff_t>(k), in[0].shape.end());
      return {in[0].dtype, out};
    }

    case Op::CumSum:
      r.arity(1);
      r.require(in[0].dtype != DType::Bool, "operand must be numeric; cast bool first");
      r.axis_in_range(attrs.axis, in[0].shape.size());
      return in[0];

    case Op::ReduceAny: {
      r.arity(1);
      r.require(in[0].dtype == DType::Bool, "operand must be bool");
      r.axis_in_range(attrs.axis, in[0].shape.size());
      Shape out = in[0].shape;
      if (attrs.keepdims) out[attrs.axis] = 1;
      else out.erase(out.begin() + attrs.axis);
      return {DType::Bool, out};
    }

    case Op::MatMul: {
      r.arity(2);
      const Shape& a = in[0].shape;
      const Shape& b = in[1].shape;
      r.require(in[0].dtype == DType::F32 && in[1].dtype == DType::F32, "operands must be f32");
      r.require(a.size() >= 2 && a.size() == b.size(), "operands must have equal rank of at least 2");
      size_t n = a.size();
      r.require(std::equal(a.begin(), a.end() - 2, b.begin()), "batch axes must match");
      r.require(a[n - 1] == b[n - 2], "inner dimensions must match");
      Shape out(a.begin(), a.end() - 1);  // batch..., rows of a
      out.push_back(b[n - 1]);            // columns of b
      return {DType::F32, out};
    }

    case Op::Gemm: {
      r.arity(3);
      for (const Type& t : in) r.require(t.dtype == DType::F32, "operands must be f32");
      const Shape& a = in[0].shape;
      const Shape& b = in[1].shape;
      r.require(a.size() == 2 && b.size() == 2, "matrices must be rank 2");
      r.require(a[1] == b[0], "inner dimensions must match");
      r.require(in[2].shape == Shape{b[1]}, "bias must be a vector with one entry per output column");
      return {DType::F32, {a[0], b[1]}};
    }

    case Op::LayerNorm: {
      r.arity(3);
      for (const Type& t : in) r.require(t.dtype == DType::F32, "operands must be f32");
      r.require(!in[0].shape.empty(), "input must have at least one axis");
      Shape last = {in[0].shape.back()};
      r.require(in[1].shape == last && in[2].shape == last, "weight and bias must match the last axis");
      return in[0];
    }

    case Op::Softmax:
      r.arity(1);
      r.require(in[0].dtype == DType::F32, "operand must be f32");
      r.axis_in_range(attrs.axis, in[0].shape.size());
      return in[0];
  }
  throw GraphError("infer_type: unknown op");
}

// ---- Graph builders ----------------------------------------------------------

ValueId Graph::add(Op op, std::vector<ValueId> ins, Attrs attrs, std::string name) {
  std::vector<Type> types;
  types.reserve(ins.size());
  for (ValueId v : ins) {
    if (v >= nodes.size()) throw GraphError(std::string(op_name(op)) + ": input refers to a value that does not exist yet");
    types.push_back(nodes[v].type);
  }
  Type type = infer_type(op, types, attrs);
  nodes.push_back(Node{op, std::move(type), std::move(ins), std::move(attrs), std::move(name)});
  return static_cast<ValueId>(nodes.size() - 1);
}

ValueId Graph::add_input(Type type, std::string name) {
  Attrs attrs;
  attrs.index = static_cast<int32_t>(inputs.size());
  nodes.push_back(Node{Op::Input, std::move(type), {}, attrs, std::move(name)});
  inputs.push_back(static_cast<ValueId>(nodes.size() - 1));
  return inputs.back();
}

ValueId Graph::add_const(Constant constant, std::string name) {
  if (!constant.is_external() && constant.inline_data.size() != constant.type.byte_size())
    throw GraphError("constant " + name + ": data size does not match its type " + type_string(constant.type));
  Attrs attrs;
  attrs.index = static_cast<int32_t>(constants.size());
  nodes.push_back(Node{Op::Const, constant.type, {}, attrs, std::move(name)});
  constants.push_back(std::move(constant));
  return static_cast<ValueId>(nodes.size() - 1);
}

ValueId Graph::add_scalar(DType dtype, double value, std::string name) {
  Constant c;
  c.type = {dtype, {}};
  c.inline_data.resize(dtype_size(dtype));
  switch (dtype) {
    case DType::F32: { float v = static_cast<float>(value); std::memcpy(c.inline_data.data(), &v, sizeof v); break; }
    case DType::I64: { int64_t v = static_cast<int64_t>(value); std::memcpy(c.inline_data.data(), &v, sizeof v); break; }
    case DType::Bool: { uint8_t v = value != 0; std::memcpy(c.inline_data.data(), &v, sizeof v); break; }
  }
  return add_const(std::move(c), std::move(name));
}

}  // namespace mlc
