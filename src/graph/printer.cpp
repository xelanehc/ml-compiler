#include <array>
#include <cstdio>
#include <cstring>

#include "graph/ir.h"

namespace mlc {

namespace {

std::string ints_string(const std::vector<int64_t>& v) { return shape_string(v); }

std::string format_double(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

// How a constant shows up in the listing: scalars by value, everything else
// by where its bytes live.
std::string const_string(const Constant& c) {
  if (c.is_external()) return "weights+" + std::to_string(c.weights_offset);
  if (c.type.shape.empty()) {
    switch (c.type.dtype) {
      case DType::F32: { float v; std::memcpy(&v, c.inline_data.data(), sizeof v); return format_double(v); }
      case DType::I64: { int64_t v; std::memcpy(&v, c.inline_data.data(), sizeof v); return std::to_string(v); }
      case DType::Bool: return c.inline_data[0] != std::byte{0} ? "true" : "false";
    }
  }
  return "inline";
}

// The attributes an op actually reads, as ", name=value" pairs.
std::string attrs_string(const Node& n) {
  const Attrs& a = n.attrs;
  switch (n.op) {
    case Op::Reshape:
    case Op::Permute:
    case Op::Broadcast: return ", dims=" + ints_string(a.dims);
    case Op::Slice:
      return ", axis=" + std::to_string(a.axis) + ", start=" + std::to_string(a.start) + ", end=" + std::to_string(a.end);
    case Op::Concat:
    case Op::CumSum: return ", axis=" + std::to_string(a.axis);
    case Op::Softmax: return ", axis=" + std::to_string(a.axis) + (a.causal ? ", causal=true" : "");
    case Op::ReduceAny: return ", axis=" + std::to_string(a.axis) + ", keepdims=" + (a.keepdims ? "true" : "false");
    case Op::LayerNorm: return ", eps=" + format_double(a.eps);
    case Op::Cast: return std::string(", to=") + dtype_name(a.to);
    case Op::MatMul:
    case Op::Gemm: return a.trans_b ? ", trans_b=true" : "";
    default: return "";
  }
}

}  // namespace

std::string print_node(const Graph& g, ValueId v) {
  const Node& n = g.nodes.at(v);
  std::string line = "%" + std::to_string(v) + ": " + type_string(n.type) + " = " + op_name(n.op) + "(";
  if (n.op == Op::Const) {
    line += const_string(g.constants.at(n.attrs.index));
  } else if (n.op == Op::Input) {
    line += std::to_string(n.attrs.index);
  } else {
    for (size_t i = 0; i < n.inputs.size(); ++i) line += (i ? ", %" : "%") + std::to_string(n.inputs[i]);
    line += attrs_string(n);
  }
  line += ")";
  if (!n.name.empty()) {
    if (line.size() < 64) line.resize(64, ' ');
    line += "  # " + n.name;
  }
  return line;
}

std::string print_graph(const Graph& g) {
  std::string out;
  for (ValueId v = 0; v < g.nodes.size(); ++v) out += print_node(g, v) + "\n";
  out += "return";
  for (size_t i = 0; i < g.outputs.size(); ++i) out += (i ? ", %" : " %") + std::to_string(g.outputs[i]);
  return out + "\n";
}

std::string summarize_graph(const Graph& g) {
  std::array<int, kNumOps> counts{};
  for (const Node& n : g.nodes) ++counts[static_cast<int>(n.op)];
  std::string out = std::to_string(g.nodes.size()) + " nodes\n";
  for (int i = 0; i < kNumOps; ++i) {
    if (!counts[i]) continue;
    char buf[64];
    std::snprintf(buf, sizeof buf, "  %-12s %5d\n", op_name(static_cast<Op>(i)), counts[i]);
    out += buf;
  }
  return out;
}

}  // namespace mlc
