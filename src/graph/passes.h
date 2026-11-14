// Graph optimization passes.
//
// Every pass is a function Graph -> Graph. Nothing is edited in place: a pass
// walks the old graph in order and builds a new one, deciding for each node
// whether to copy it, replace it with something cheaper, or point its users
// at a value that already exists. Because the graph is in SSA form and nodes
// are pure, "replace node v by value w" is always legal when v and w compute
// the same thing, with no question of what else might observe the change.
#pragma once

#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "graph/interpreter.h"
#include "graph/ir.h"

namespace mlc {

// ---- The rewriting engine ----------------------------------------------------

// Called once per node of the old graph, in order. `n` is that node with its
// inputs already translated to values of the new graph `out`, so the rule can
// inspect what its operands became. `constant` is set for Const nodes.
// The rule returns the value in `out` that should stand for this node, or
// kNoValue to have the node copied unchanged.
using RewriteRule = std::function<ValueId(Graph& out, const Node& n, const Constant* constant)>;

// Applies `rule` to every node. Because each node sees operands that have
// already been rewritten, a chain such as reshape(reshape(reshape(x)))
// collapses in a single sweep.
Graph rewrite(const Graph& g, const RewriteRule& rule);

// Copies `n` into `out` unchanged. What `rewrite` does when a rule declines.
ValueId copy_node(Graph& out, const Node& n, const Constant* constant);

// Evaluates `v` at compile time if it depends on no graph input and no
// external weight, and no value on the way is larger than `max_bytes`.
// The result owns its data.
std::optional<Tensor> evaluate_constant(const Graph& g, ValueId v, size_t max_bytes);

// ---- The passes ----------------------------------------------------------------

// Dead code elimination: drops every node no output depends on. Graph inputs
// are kept even when unused, because they are part of the calling convention.
Graph eliminate_dead_code(const Graph& g);

// Local cleanups that need no analysis: a broadcast, reshape, slice, or
// permute that changes nothing is removed; reshape-of-reshape and
// permute-of-permute collapse to one; x + 0, x * 1, and not(not(x)) become x.
Graph simplify(const Graph& g);

// Common subexpression elimination: two nodes with the same op, operands, and
// attributes compute the same value, so the second is replaced by the first.
// Constants with identical contents are merged the same way.
Graph eliminate_common_subexpressions(const Graph& g);

// Constant folding: a node whose operands are all inline constants is
// evaluated now and replaced by its result, provided the result is at most
// `max_bytes`. External weights are never folded (their data must stay in
// the weights file), and neither is Broadcast (folding it would turn four
// bytes into a full tensor).
Graph fold_constants(const Graph& g, size_t max_bytes = size_t{1} << 20);

// Recognizes the nine-node tanh GELU expansion and replaces it with one Gelu.
Graph fuse_gelu(const Graph& g);

// Recognizes "add a causal mask, softmax, zero the rows that were masked
// everywhere" and replaces it with one causal Softmax on the unmasked scores.
Graph fuse_causal_softmax(const Graph& g);

// Rewrites every matrix product to take its second operand transposed
// (attrs.trans_b), cancelling the transposes that were already there.
Graph transpose_matmul_operands(const Graph& g);

// ---- The pipeline ----------------------------------------------------------------

// One switch per optimization, at every level, so that one flag set describes
// a build and an ablation is a list of names.
struct OptOptions {
  // Graph passes.
  bool simplify = true;
  bool fold = true;
  bool cse = true;
  bool gelu = true;
  bool attention = true;
  bool layout = true;     // transpose_matmul_operands
  // Applied at lowering (src/loop/lower.cpp).
  bool memory = true;     // memory planning
  bool hoist = true;      // compute values that depend only on weights once, at load time
  // Loop schedule (src/loop/schedule.cpp).
  bool vectorize = true;
  bool vmath = true;      // vector exp and tanh
  bool regtile = true;
  bool cachetile = true;
  bool parallel = true;
  // Code generation.
  bool noalias = true;    // tell the C++ compiler that a kernel's buffers do not overlap

  // Turns passes off by name, from a comma-separated list. "all" turns every
  // pass off. Throws GraphError for a name that is not a pass.
  void disable(const std::string& names);
  static const char* pass_names() {
    return "simplify, fold, cse, gelu, attention, memory, noalias, vectorize, vmath, layout, hoist, regtile, cachetile, parallel, all";
  }
};

struct PassStat {
  std::string name;
  size_t nodes_before = 0;
  size_t nodes_after = 0;
};

// Runs the enabled passes in order, each followed by dead code elimination,
// and verifies the graph after every one. `stats` receives a row per pass.
Graph optimize(const Graph& g, const OptOptions& options = {}, std::vector<PassStat>* stats = nullptr);

}  // namespace mlc
