#include "graph/ir.h"

namespace mlc {

// The verifier assumes nothing. Graph::add cannot build an ill-typed node, but
// passes will edit nodes in place, and a pass with a bug is exactly the thing
// this exists to catch. So every rule is re-derived from scratch here.
std::vector<std::string> verify_graph(const Graph& g) {
  std::vector<std::string> errors;
  auto report = [&](ValueId v, const std::string& what) {
    errors.push_back("%" + std::to_string(v) + " (" + op_name(g.nodes[v].op) + "): " + what);
  };

  for (ValueId v = 0; v < g.nodes.size(); ++v) {
    const Node& n = g.nodes[v];

    // Use before def. Because the node list is the execution order, "defined
    // earlier" is simply "has a smaller index". This one comparison also rules
    // out cycles: a cycle would need some edge to point forward.
    bool inputs_ok = true;
    for (ValueId in : n.inputs) {
      if (in >= v) {
        report(v, "uses %" + std::to_string(in) + ", which is not defined before it");
        inputs_ok = false;
      }
    }
    if (!inputs_ok) continue;

    if (n.op == Op::Input) {
      if (!n.inputs.empty()) report(v, "an input takes no operands");
      if (n.attrs.index < 0 || static_cast<size_t>(n.attrs.index) >= g.inputs.size() || g.inputs[n.attrs.index] != v)
        report(v, "is not registered in the graph's input list");
      continue;
    }

    if (n.op == Op::Const) {
      if (!n.inputs.empty()) report(v, "a constant takes no operands");
      if (n.attrs.index < 0 || static_cast<size_t>(n.attrs.index) >= g.constants.size()) {
        report(v, "constant index out of range");
        continue;
      }
      const Constant& c = g.constants[n.attrs.index];
      if (c.type != n.type) report(v, "node type " + type_string(n.type) + " differs from its constant's " + type_string(c.type));
      if (!c.is_external() && c.inline_data.size() != c.type.byte_size()) report(v, "inline data size does not match the type");
      continue;
    }

    std::vector<Type> in_types;
    for (ValueId in : n.inputs) in_types.push_back(g.nodes[in].type);
    try {
      Type expect = infer_type(n.op, in_types, n.attrs);
      if (expect != n.type) report(v, "stored type " + type_string(n.type) + " but its inputs imply " + type_string(expect));
    } catch (const GraphError& e) {
      report(v, e.what());
    }
  }

  for (size_t i = 0; i < g.inputs.size(); ++i) {
    ValueId v = g.inputs[i];
    if (v >= g.nodes.size() || g.nodes[v].op != Op::Input) errors.push_back("graph input " + std::to_string(i) + " is not an input node");
  }
  if (g.outputs.empty()) errors.push_back("graph has no outputs");
  for (ValueId v : g.outputs)
    if (v >= g.nodes.size()) errors.push_back("graph output %" + std::to_string(v) + " does not exist");

  return errors;
}

void check_graph(const Graph& g) {
  std::vector<std::string> errors = verify_graph(g);
  if (errors.empty()) return;
  std::string msg = "graph verification failed:";
  for (const std::string& e : errors) msg += "\n  " + e;
  throw GraphError(msg);
}

}  // namespace mlc
