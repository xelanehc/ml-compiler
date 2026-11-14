#include "graph/passes.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <unordered_map>

namespace mlc {

// ---- The rewriting engine -------------------------------------------------------

ValueId copy_node(Graph& out, const Node& n, const Constant* constant) {
  if (n.op == Op::Input) return out.add_input(n.type, n.name);
  if (n.op == Op::Const) return out.add_const(*constant, n.name);
  return out.add(n.op, n.inputs, n.attrs, n.name);
}

Graph rewrite(const Graph& g, const RewriteRule& rule) {
  Graph out;
  // old value -> new value. Filled in node order, so by the time a node is
  // visited all of its operands already have an entry.
  std::vector<ValueId> remap(g.nodes.size(), kNoValue);
  for (ValueId v = 0; v < g.nodes.size(); ++v) {
    Node n = g.nodes[v];
    for (ValueId& in : n.inputs) in = remap[in];
    const Constant* constant = n.op == Op::Const ? &g.constants[n.attrs.index] : nullptr;
    // Inputs are never rewritten: their order is the calling convention.
    ValueId replacement = n.op == Op::Input ? kNoValue : rule(out, n, constant);
    remap[v] = replacement != kNoValue ? replacement : copy_node(out, n, constant);
    if (out.type_of(remap[v]) != g.nodes[v].type)
      throw GraphError("rewrite changed the type of %" + std::to_string(v) + " (" + op_name(n.op) + ") from " + type_string(g.nodes[v].type) +
                       " to " + type_string(out.type_of(remap[v])));
  }
  for (ValueId o : g.outputs) out.outputs.push_back(remap[o]);
  return out;
}

Graph eliminate_dead_code(const Graph& g) {
  // Liveness flows backwards: an output is live, and a live node makes its
  // operands live. Walking the nodes in reverse visits every user before the
  // values it uses, so one pass is enough.
  std::vector<bool> live(g.nodes.size(), false);
  for (ValueId o : g.outputs) live[o] = true;
  for (ValueId v = static_cast<ValueId>(g.nodes.size()); v-- > 0;) {
    if (g.nodes[v].op == Op::Input) live[v] = true;
    if (live[v])
      for (ValueId in : g.nodes[v].inputs) live[in] = true;
  }

  Graph out;
  std::vector<ValueId> remap(g.nodes.size(), kNoValue);
  for (ValueId v = 0; v < g.nodes.size(); ++v) {
    if (!live[v]) continue;
    Node n = g.nodes[v];
    for (ValueId& in : n.inputs) in = remap[in];
    remap[v] = copy_node(out, n, n.op == Op::Const ? &g.constants[n.attrs.index] : nullptr);
  }
  for (ValueId o : g.outputs) out.outputs.push_back(remap[o]);
  return out;
}

std::optional<Tensor> evaluate_constant(const Graph& g, ValueId v, size_t max_bytes) {
  // Find everything `v` depends on, refusing as soon as it reaches something
  // not known at compile time.
  std::vector<bool> needed(g.nodes.size(), false);
  std::vector<ValueId> stack = {v};
  while (!stack.empty()) {
    ValueId cur = stack.back();
    stack.pop_back();
    if (needed[cur]) continue;
    needed[cur] = true;
    const Node& n = g.nodes[cur];
    if (n.op == Op::Input) return std::nullopt;
    if (n.op == Op::Const && g.constants[n.attrs.index].is_external()) return std::nullopt;
    if (n.type.byte_size() > max_bytes) return std::nullopt;
    for (ValueId in : n.inputs) stack.push_back(in);
  }

  // Copy just that slice of the graph and run it in the interpreter.
  Graph sub;
  std::vector<ValueId> remap(g.nodes.size(), kNoValue);
  for (ValueId cur = 0; cur <= v; ++cur) {
    if (!needed[cur]) continue;
    Node n = g.nodes[cur];
    for (ValueId& in : n.inputs) in = remap[in];
    remap[cur] = copy_node(sub, n, n.op == Op::Const ? &g.constants[n.attrs.index] : nullptr);
  }
  sub.outputs = {remap[v]};
  Tensor result = run_graph(sub, {}, {}).at(0);
  if (!result.owner) {  // the value is itself an inline constant of `sub`, which is about to go away
    std::shared_ptr<std::byte[]> copy(new std::byte[std::max<size_t>(result.type.byte_size(), 1)]);
    std::memcpy(copy.get(), result.ptr, result.type.byte_size());
    result.ptr = copy.get();
    result.owner = std::move(copy);
  }
  return result;
}

// ---- Pattern-matching helpers -----------------------------------------------------

namespace {

// The node defining `v` if it has the given op, else nullptr.
const Node* as(const Graph& g, ValueId v, Op op) { return g.nodes[v].op == op ? &g.nodes[v] : nullptr; }

// The value of a rank-0 inline constant, as a double.
std::optional<double> scalar_value(const Graph& g, ValueId v) {
  const Node& n = g.nodes[v];
  if (n.op != Op::Const || !n.type.shape.empty()) return std::nullopt;
  const Constant& c = g.constants[n.attrs.index];
  if (c.is_external()) return std::nullopt;
  switch (c.type.dtype) {
    case DType::F32: { float x; std::memcpy(&x, c.inline_data.data(), sizeof x); return x; }
    case DType::I64: { int64_t x; std::memcpy(&x, c.inline_data.data(), sizeof x); return static_cast<double>(x); }
    case DType::Bool: return c.inline_data[0] != std::byte{0} ? 1.0 : 0.0;
  }
  return std::nullopt;
}

// True if `v` is the scalar constant `want`. Floats in the graph are f32, so
// a constant written as 0.7978845608028654 arrives rounded; compare loosely.
bool is_scalar(const Graph& g, ValueId v, double want) {
  std::optional<double> got = scalar_value(g, v);
  if (!got) return false;
  if (std::isinf(want) || std::isinf(*got)) return *got == want;
  return std::fabs(*got - want) <= 1e-6 * std::max(1.0, std::fabs(want));
}

// Matches `inputs` as (x, scalar c) in either order, for a commutative op.
bool operand_and_scalar(const Graph& g, const std::vector<ValueId>& inputs, double c, ValueId* x) {
  if (inputs.size() != 2) return false;
  if (is_scalar(g, inputs[1], c)) { *x = inputs[0]; return true; }
  if (is_scalar(g, inputs[0], c)) { *x = inputs[1]; return true; }
  return false;
}

// Matches value `v` as op(x, c) or op(c, x).
bool op_with_scalar(const Graph& g, ValueId v, Op op, double c, ValueId* x) {
  const Node* n = as(g, v, op);
  return n && operand_and_scalar(g, n->inputs, c, x);
}

// True if every element of `v` is known to be zero without evaluating anything big.
bool is_zero_fill(const Graph& g, ValueId v) {
  if (const Node* b = as(g, v, Op::Broadcast)) return is_zero_fill(g, b->inputs[0]);
  if (is_scalar(g, v, 0.0)) return true;
  const Node& n = g.nodes[v];
  if (n.op != Op::Const) return false;
  const Constant& c = g.constants[n.attrs.index];
  if (c.is_external()) return false;
  for (std::byte b : c.inline_data)
    if (b != std::byte{0}) return false;  // all-zero bytes are 0, 0.0f, and false alike
  return true;
}

bool is_identity_permutation(const std::vector<int64_t>& dims) {
  for (size_t i = 0; i < dims.size(); ++i)
    if (dims[i] != static_cast<int64_t>(i)) return false;
  return true;
}

}  // namespace

// ---- simplify ----------------------------------------------------------------------

Graph simplify(const Graph& g) {
  return rewrite(g, [](Graph& out, const Node& n, const Constant*) -> ValueId {
    auto in = [&](size_t i) { return n.inputs[i]; };
    switch (n.op) {
      case Op::Broadcast:
        // PyTorch's expand() to a tensor's own shape. All 49 in GPT-2 are this.
        if (out.type_of(in(0)).shape == n.attrs.dims) return in(0);
        break;

      case Op::Reshape:
        if (out.type_of(in(0)).shape == n.attrs.dims) return in(0);
        // Only the final shape matters: reshape(reshape(x, a), b) = reshape(x, b).
        if (const Node* inner = as(out, in(0), Op::Reshape)) {
          if (out.type_of(inner->inputs[0]).shape == n.attrs.dims) return inner->inputs[0];
          return out.add(Op::Reshape, {inner->inputs[0]}, n.attrs, n.name);
        }
        break;

      case Op::Permute:
        if (is_identity_permutation(n.attrs.dims)) return in(0);
        // Two permutations compose into one. The outer one says "my axis i is
        // the middle value's axis dims[i]", and the inner one says "my axis k
        // is x's axis inner[k]", so the result's axis i is x's axis inner[dims[i]].
        if (const Node* inner = as(out, in(0), Op::Permute)) {
          Attrs composed;
          for (int64_t d : n.attrs.dims) composed.dims.push_back(inner->attrs.dims[d]);
          if (is_identity_permutation(composed.dims)) return inner->inputs[0];
          return out.add(Op::Permute, {inner->inputs[0]}, composed, n.name);
        }
        break;

      case Op::Slice:
        if (n.attrs.start == 0 && n.attrs.end == out.type_of(in(0)).shape[n.attrs.axis]) return in(0);
        break;

      case Op::Concat:
        if (n.inputs.size() == 1) return in(0);
        break;

      case Op::Add:
      case Op::Mul: {
        // x + 0 and x * 1, when the other operand does not change x's shape.
        ValueId x;
        if (operand_and_scalar(out, n.inputs, n.op == Op::Add ? 0.0 : 1.0, &x) && out.type_of(x) == n.type) return x;
        break;
      }

      case Op::Not:
        if (const Node* inner = as(out, in(0), Op::Not)) return inner->inputs[0];
        break;

      default: break;
    }
    return kNoValue;
  });
}

// ---- CSE ---------------------------------------------------------------------------

Graph eliminate_common_subexpressions(const Graph& g) {
  // A node is identified by everything that determines its value: op,
  // operands, attributes. Operands are compared as value ids of the new
  // graph, which have themselves been deduplicated already; that is why
  // equal subtrees end up with equal keys (this is "value numbering").
  std::unordered_map<std::string, ValueId> seen;
  auto append = [](std::string& key, const void* data, size_t size) { key.append(static_cast<const char*>(data), size); };

  return rewrite(g, [&](Graph& out, const Node& n, const Constant* constant) -> ValueId {
    std::string key;
    key.push_back(static_cast<char>(n.op));
    key.push_back(static_cast<char>(n.type.dtype));
    for (int64_t d : n.type.shape) append(key, &d, sizeof d);
    key.push_back('|');
    if (n.op == Op::Const) {
      if (constant->is_external()) append(key, &constant->weights_offset, sizeof constant->weights_offset);
      else append(key, constant->inline_data.data(), constant->inline_data.size());
    } else {
      for (ValueId in : n.inputs) append(key, &in, sizeof in);
      key.push_back('|');
      const Attrs& a = n.attrs;
      for (int64_t x : {a.axis, a.start, a.end, static_cast<int64_t>(a.keepdims), static_cast<int64_t>(a.to), static_cast<int64_t>(a.causal),
                        static_cast<int64_t>(a.trans_b)})
        append(key, &x, sizeof x);
      append(key, &a.eps, sizeof a.eps);
      for (int64_t d : a.dims) append(key, &d, sizeof d);
    }

    auto [it, inserted] = seen.try_emplace(std::move(key), kNoValue);
    if (inserted) it->second = copy_node(out, n, constant);
    return it->second;
  });
}

// ---- Constant folding ----------------------------------------------------------------

Graph fold_constants(const Graph& g, size_t max_bytes) {
  return rewrite(g, [max_bytes](Graph& out, const Node& n, const Constant*) -> ValueId {
    if (n.op == Op::Const || n.op == Op::Broadcast || n.inputs.empty()) return kNoValue;
    if (n.type.byte_size() > max_bytes) return kNoValue;
    for (ValueId in : n.inputs) {
      const Node& operand = out.nodes[in];
      if (operand.op != Op::Const || out.constants[operand.attrs.index].is_external()) return kNoValue;
    }

    // Build a graph of just this node and its constant operands, and run it.
    // The operands were themselves folded earlier in this same sweep, so a
    // whole constant subgraph collapses one node at a time from the leaves up.
    Graph one;
    std::vector<ValueId> operands;
    for (ValueId in : n.inputs) operands.push_back(one.add_const(out.constants[out.nodes[in].attrs.index]));
    one.outputs = {one.add(n.op, operands, n.attrs)};
    Tensor value = run_graph(one, {}, {}).at(0);

    Constant folded;
    folded.type = value.type;
    folded.inline_data.assign(value.ptr, value.ptr + value.type.byte_size());
    return out.add_const(std::move(folded), n.name);
  });
}

// ---- GELU ----------------------------------------------------------------------------

// The export spells GELU as nine nodes:
//
//   half  = mul(h, 0.5)
//   cube  = pow(h, 3.0)
//   inner = mul(add(h, mul(cube, 0.044715)), 0.7978845608)     0.79788... is sqrt(2/pi)
//   out   = mul(half, add(tanh(inner), 1.0))
//
// The rule fires on the final mul and walks up through its operands, checking
// that the constants are the right ones and that every path ends at the same
// h. The eight nodes above it are left behind with no users, and dead code
// elimination removes them.
Graph fuse_gelu(const Graph& g) {
  return rewrite(g, [](Graph& out, const Node& n, const Constant*) -> ValueId {
    if (n.op != Op::Mul) return kNoValue;
    for (int swap = 0; swap < 2; ++swap) {  // mul commutes: try both operand orders
      ValueId half = n.inputs[swap], rest = n.inputs[1 - swap];
      ValueId h, tanh_value, scaled, cube;
      if (!op_with_scalar(out, half, Op::Mul, 0.5, &h)) continue;
      if (!op_with_scalar(out, rest, Op::Add, 1.0, &tanh_value)) continue;
      const Node* tanh = as(out, tanh_value, Op::Tanh);
      if (!tanh || !op_with_scalar(out, tanh->inputs[0], Op::Mul, 0.7978845608028654, &scaled)) continue;
      const Node* sum = as(out, scaled, Op::Add);
      if (!sum) continue;
      ValueId correction = sum->inputs[0] == h ? sum->inputs[1] : sum->inputs[0];
      if (sum->inputs[0] != h && sum->inputs[1] != h) continue;
      if (!op_with_scalar(out, correction, Op::Mul, 0.044715, &cube)) continue;
      const Node* pow = as(out, cube, Op::Pow);
      if (!pow || pow->inputs[0] != h || !is_scalar(out, pow->inputs[1], 3.0)) continue;
      if (out.type_of(h) != n.type) continue;
      return out.add(Op::Gelu, {h}, {}, n.name);
    }
    return kNoValue;
  });
}

// ---- Causal softmax ----------------------------------------------------------------------

namespace {

// True if `m` is an additive causal mask: shape [1, ..., 1, T, T] with 0 on
// and below the diagonal and -inf above it.
bool is_causal_mask(const Tensor& m) {
  const Shape& s = m.type.shape;
  if (m.type.dtype != DType::F32 || s.size() < 2 || s[s.size() - 1] != s[s.size() - 2]) return false;
  for (size_t d = 0; d + 2 < s.size(); ++d)
    if (s[d] != 1) return false;
  const int64_t t = s.back();
  const float* p = m.data<float>();
  for (int64_t i = 0; i < t; ++i)
    for (int64_t j = 0; j < t; ++j)
      if (p[i * t + j] != (j <= i ? 0.0f : -std::numeric_limits<float>::infinity())) return false;
  return true;
}

}  // namespace

// Each attention layer exports as:
//
//   s     = add(scores, mask)                    mask: 0 where j <= i, -inf where j > i
//   p     = softmax(s, last axis)
//   dead  = not(reduce_any(not(eq(s, -inf))))    "this whole row is -inf"
//   out   = where(dead, zeros, p)                such a row would be NaN; make it 0
//
// With a causal mask, row i always keeps column i, so no row is ever entirely
// masked and `dead` is always false. (This assumes the scores themselves are
// finite, which they are for any input that has not already overflowed.) And
// adding -inf before exp is a roundabout way to say "leave these columns out".
// So the whole group equals one causal softmax of the raw scores, which does
// half the exponentials and needs no mask in memory at all.
//
// The mask is recognized by value: whatever computes it, if it evaluates at
// compile time to the causal pattern, the rule fires.
Graph fuse_causal_softmax(const Graph& g) {
  std::map<ValueId, bool> mask_cache;  // every layer shares one mask after CSE
  return rewrite(g, [&](Graph& out, const Node& n, const Constant*) -> ValueId {
    if (n.op != Op::Where) return kNoValue;
    ValueId dead = n.inputs[0], zeros = n.inputs[1], probs = n.inputs[2];

    const Node* softmax = as(out, probs, Op::Softmax);
    const int64_t last = static_cast<int64_t>(n.type.shape.size()) - 1;
    if (!softmax || softmax->attrs.causal || softmax->attrs.axis != last || last < 1) return kNoValue;
    ValueId s = softmax->inputs[0];
    if (!is_zero_fill(out, zeros)) return kNoValue;

    // dead = not(reduce_any(not(eq(s, -inf)), last axis, keepdims))
    const Node* outer_not = as(out, dead, Op::Not);
    const Node* any = outer_not ? as(out, outer_not->inputs[0], Op::ReduceAny) : nullptr;
    const Node* inner_not = any ? as(out, any->inputs[0], Op::Not) : nullptr;
    if (!inner_not || any->attrs.axis != last || !any->attrs.keepdims) return kNoValue;
    ValueId compared;
    if (!op_with_scalar(out, inner_not->inputs[0], Op::Eq, -std::numeric_limits<double>::infinity(), &compared) || compared != s) return kNoValue;

    // s = add(scores, mask), in either order, with a causal mask.
    const Node* sum = as(out, s, Op::Add);
    if (!sum) return kNoValue;
    for (int swap = 0; swap < 2; ++swap) {
      ValueId scores = sum->inputs[swap], mask = sum->inputs[1 - swap];
      if (out.type_of(scores) != n.type) continue;
      auto cached = mask_cache.find(mask);
      if (cached == mask_cache.end()) {
        std::optional<Tensor> value = evaluate_constant(out, mask, size_t{64} << 20);
        cached = mask_cache.emplace(mask, value && is_causal_mask(*value)).first;
      }
      if (!cached->second) continue;
      Attrs attrs;
      attrs.axis = last;
      attrs.causal = true;
      return out.add(Op::Softmax, {scores}, attrs, n.name);
    }
    return kNoValue;
  });
}

// ---- Layout --------------------------------------------------------------------------------

namespace {

// A value equal to `v` with its last two axes swapped, built with as little
// run-time work as possible.
ValueId transposed(Graph& out, ValueId v) {
  const Node n = out.nodes[v];  // a copy: adding nodes below moves the originals
  const size_t rank = n.type.shape.size();
  std::vector<int64_t> swap(rank);
  for (size_t d = 0; d < rank; ++d) swap[d] = static_cast<int64_t>(d);
  std::swap(swap[rank - 1], swap[rank - 2]);

  // Already the transpose of something: use that something.
  if (n.op == Op::Permute && n.attrs.dims == swap) return n.inputs[0];
  // x * scalar, transposed, is transpose(x) * scalar. Moving the transpose
  // inwards lets it meet, and cancel, a transpose applied before the scaling.
  if (n.op == Op::Mul || n.op == Op::Add || n.op == Op::Sub) {
    for (int side : {0, 1}) {
      if (!out.type_of(n.inputs[1 - side]).shape.empty() || out.type_of(n.inputs[side]).shape != n.type.shape) continue;
      std::vector<ValueId> inputs = n.inputs;
      inputs[side] = transposed(out, inputs[side]);
      return out.add(n.op, inputs, n.attrs, n.name);
    }
  }
  Attrs attrs;
  attrs.dims = swap;
  return out.add(Op::Permute, {v}, attrs);
}

}  // namespace

// Every matrix product a @ b becomes a @ transpose(bt) with bt = transpose(b)
// stored row by row. Why: element (i, j) of the product is then the dot
// product of row i of a and row j of bt, two contiguous runs of memory, which
// is the shape the vectorized kernel wants (see schedule.cpp). The transposes
// this introduces cost nothing where it matters:
//   - the output projection's operand was transpose(embedding table) to begin
//     with, so the two transposes cancel and the table is used as stored;
//   - a transposed weight depends on nothing but the weights file, so it is
//     computed once at load time (staging, in lower.cpp);
//   - in attention, k was transposed for q @ k^T and that cancels too. Only
//     the value matrix gets a real transpose per call, and it is small.
Graph transpose_matmul_operands(const Graph& g) {
  return rewrite(g, [](Graph& out, const Node& n, const Constant*) -> ValueId {
    if ((n.op != Op::MatMul && n.op != Op::Gemm) || n.attrs.trans_b) return kNoValue;
    std::vector<ValueId> inputs = n.inputs;
    inputs[1] = transposed(out, inputs[1]);
    Attrs attrs = n.attrs;
    attrs.trans_b = true;
    return out.add(n.op, inputs, attrs, n.name);
  });
}

// ---- The pipeline ------------------------------------------------------------------------

void OptOptions::disable(const std::string& names) {
  size_t start = 0;
  while (start <= names.size()) {
    size_t comma = names.find(',', start);
    std::string name = names.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    start = comma == std::string::npos ? names.size() + 1 : comma + 1;
    if (name.empty()) continue;
    bool all = name == "all";
    bool known = all;
    for (auto [pass, flag] : {std::pair{"simplify", &simplify}, {"fold", &fold}, {"cse", &cse}, {"gelu", &gelu},
                              {"attention", &attention}, {"memory", &memory}, {"noalias", &noalias}, {"vectorize", &vectorize},
                              {"vmath", &vmath}, {"layout", &layout}, {"hoist", &hoist}, {"regtile", &regtile},
                              {"cachetile", &cachetile}, {"parallel", &parallel}}) {
      if (all || name == pass) {
        *flag = false;
        known = true;
      }
    }
    if (!known) throw GraphError("unknown pass '" + name + "'; passes are: " + pass_names());
  }
}

Graph optimize(const Graph& g, const OptOptions& options, std::vector<PassStat>* stats) {
  check_graph(g);
  Graph current = eliminate_dead_code(g);
  if (stats) stats->push_back({"dce", g.nodes.size(), current.nodes.size()});

  auto run = [&](bool enabled, const char* name, const std::function<Graph(const Graph&)>& pass) {
    if (!enabled) return;
    // A pass leaves the nodes it replaced behind with no users; DCE sweeps
    // them up, so each pass is charged for everything it made removable.
    Graph next = eliminate_dead_code(pass(current));
    try {
      check_graph(next);
    } catch (const GraphError& e) {
      throw GraphError(std::string("after pass '") + name + "': " + e.what());
    }
    if (stats) stats->push_back({name, current.nodes.size(), next.nodes.size()});
    current = std::move(next);
  };

  run(options.simplify, "simplify", simplify);
  run(options.fold, "fold", [](const Graph& x) { return fold_constants(x); });
  run(options.cse, "cse", eliminate_common_subexpressions);
  run(options.gelu, "gelu", fuse_gelu);
  run(options.attention, "attention", fuse_causal_softmax);
  run(options.layout, "layout", transpose_matmul_operands);
  return current;
}

}  // namespace mlc
