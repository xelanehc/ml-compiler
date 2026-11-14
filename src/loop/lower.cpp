// Graph IR -> loop IR: one naive loop nest per node, plus the decisions of
// where every value lives (memory planning) and when it is computed (staging).
#include <cstring>
#include <functional>
#include <limits>
#include <map>

#include "loop/ir.h"

namespace mlc::loop {

namespace {

using Idx = std::vector<ExprPtr>;

constexpr int64_t kAlign = 64;  // every arena buffer starts on a cache-line boundary

// Hands out byte ranges of the arena. With reuse on, a freed range goes on a
// free list and later requests are served from it when they fit; the arena
// only grows when nothing on the list is big enough.
class ArenaPlanner {
 public:
  explicit ArenaPlanner(bool reuse) : reuse_(reuse) {}

  int64_t alloc(int64_t bytes) {
    const int64_t size = round_up(bytes);
    total_ += size;
    if (reuse_) {
      // Best fit: the smallest free block that is large enough. Taking a
      // snug block leaves the big ones for the big tensors still to come.
      auto best = free_.end();
      for (auto it = free_.begin(); it != free_.end(); ++it)
        if (it->second >= size && (best == free_.end() || it->second < best->second)) best = it;
      if (best != free_.end()) {
        int64_t offset = best->first, left = best->second - size;
        free_.erase(best);
        if (left > 0) free_[offset + size] = left;
        return offset;
      }
      // Nothing fits. If the last free block touches the end of the arena,
      // grow the arena by only the difference.
      if (!free_.empty()) {
        auto last = std::prev(free_.end());
        if (last->first + last->second == end_) {
          int64_t offset = last->first;
          free_.erase(last);
          end_ = offset + size;
          return offset;
        }
      }
    }
    int64_t offset = end_;
    end_ += size;
    return offset;
  }

  void release(int64_t offset, int64_t bytes) {
    const int64_t size = round_up(bytes);
    if (!reuse_ || size == 0) return;
    auto it = free_.emplace(offset, size).first;
    // Merge with the neighbours so that two small holes side by side can
    // serve one large request.
    auto next = std::next(it);
    if (next != free_.end() && it->first + it->second == next->first) {
      it->second += next->second;
      free_.erase(next);
    }
    if (it != free_.begin()) {
      auto prev = std::prev(it);
      if (prev->first + prev->second == it->first) {
        prev->second += it->second;
        free_.erase(it);
      }
    }
  }

  int64_t peak_bytes() const { return end_; }    // the arena size: the furthest any allocation reached
  int64_t total_bytes() const { return total_; } // the sum of every allocation: the size with no reuse

 private:
  static int64_t round_up(int64_t bytes) { return (bytes + kAlign - 1) / kAlign * kAlign; }
  bool reuse_;
  int64_t end_ = 0, total_ = 0;
  std::map<int64_t, int64_t> free_;  // offset -> size, ordered by offset
};

class Lowerer {
 public:
  Lowerer(const Graph& g, int64_t weights_bytes, const LowerOptions& options) : g_(g), options_(options), arena_(options.plan_memory) {
    p_.weights_bytes = weights_bytes;
  }

  Program run() {
    check_graph(g_);
    if (g_.nodes.size() > 0 && options_.hoist) stage();
    for (ValueId in : g_.inputs) p_.inputs.push_back(g_.type_of(in));
    for (ValueId out : g_.outputs) p_.outputs.push_back(g_.type_of(out));
    analyze();
    for (ValueId v = 0; v < g_.nodes.size(); ++v) {
      place(v);
      lower_node(v);
      release_after(v);
    }
    copy_unowned_outputs();
    p_.arena_bytes = arena_.peak_bytes();
    p_.persistent_bytes = persistent_end_;
    p_.unplanned_arena_bytes = arena_.total_bytes();
    check_program(p_);
    return std::move(p_);
  }

 private:
  struct Block {
    int64_t offset = 0, bytes = 0;
  };

  const Graph& g_;
  LowerOptions options_;
  Program p_;
  ArenaPlanner arena_;
  std::vector<BufferId> buffer_of_;       // graph value -> its buffer
  std::vector<ValueId> root_;             // graph value -> the value that owns its bytes (itself, unless a Reshape)
  std::vector<int> output_of_;            // root -> index of the graph output computed into it, or -1
  std::vector<int64_t> last_use_;         // root -> index of the last node that reads it, or -1 if none
  std::vector<bool> owns_arena_;          // root -> holds an arena block right now
  std::vector<Block> block_;              // root -> that block
  std::vector<Block> temps_;              // scratch blocks of the node being lowered
  std::vector<int> copy_outputs_;         // outputs that need an explicit copy at the end
  std::vector<bool> at_load_;             // graph value -> known once the weights are (see stage())
  int64_t persistent_end_ = 0;
  VarId next_var_ = 0;

  // Staging, or binding-time analysis: sort values by when they become known.
  //   compile time   inline constants. Folding (a graph pass) already evaluated
  //                  what it could of these.
  //   load time      weights, and anything computed from weights and constants
  //                  alone. Nothing here depends on the caller's input, so it
  //                  is the same on every call.
  //   run time       everything downstream of a graph input.
  // A load-time value is computed by an init kernel into persistent memory,
  // once, when the model is opened. It is constant folding for values whose
  // bytes are not available to the compiler: the fold happens in the runtime.
  void stage() {
    at_load_.assign(g_.nodes.size(), false);
    for (ValueId v = 0; v < g_.nodes.size(); ++v) {
      const Node& n = g_.nodes[v];
      if (n.op == Op::Input) continue;
      bool known = true;
      for (ValueId in : n.inputs) known = known && at_load_[in];
      at_load_[v] = known;  // a Const has no inputs and is known
    }
  }

  bool at_load(ValueId v) const { return !at_load_.empty() && at_load_[v]; }

  // ---- Storage ----

  BufferId new_buffer(std::string name, const Type& type, Storage storage, int64_t offset, int32_t index) {
    p_.buffers.push_back(Buffer{std::move(name), type.dtype, type.shape, storage, offset, index});
    return static_cast<BufferId>(p_.buffers.size() - 1);
  }

  // Liveness analysis. For every value that owns bytes, find the last node
  // that reads it. Kernels run in node order, so a value's lifetime is the
  // interval from its own index to that last reader's index, and two values
  // may share memory exactly when their intervals do not overlap.
  //
  // A Reshape does not own bytes: it is the same bytes as its operand under a
  // different shape. `root_` follows such chains back to the owner, and a read
  // of the reshaped value counts as a read of the owner.
  void analyze() {
    const size_t n = g_.nodes.size();
    root_.resize(n);
    for (ValueId v = 0; v < n; ++v) root_[v] = g_.nodes[v].op == Op::Reshape ? root_[g_.nodes[v].inputs[0]] : v;

    last_use_.assign(n, -1);
    for (ValueId v = 0; v < n; ++v)
      for (ValueId in : g_.nodes[v].inputs) last_use_[root_[in]] = v;

    // A value the caller wants back is computed straight into the caller's
    // buffer when possible. It is not possible when the value is an input or
    // a constant, or when two outputs are the same value; those get a copy.
    output_of_.assign(n, -1);
    for (size_t k = 0; k < g_.outputs.size(); ++k) {
      ValueId r = root_[g_.outputs[k]];
      Op op = g_.nodes[r].op;
      // (Nor when it is computed at load time, before there is a caller.)
      if (op != Op::Input && op != Op::Const && !at_load(r) && output_of_[r] < 0) output_of_[r] = static_cast<int>(k);
      else copy_outputs_.push_back(static_cast<int>(k));
    }

    buffer_of_.assign(n, -1);
    owns_arena_.assign(n, false);
    block_.assign(n, {});
  }

  // Ops whose kernel reads each operand element at the same index it then
  // writes, and never reads it again afterwards. Such an op can write its
  // result over an operand: by the time out[i] is stored, in[i] has been used.
  static bool can_run_in_place(Op op) {
    switch (op) {
      case Op::Add: case Op::Sub: case Op::Mul: case Op::Pow: case Op::Tanh: case Op::Gelu:
      case Op::And: case Op::Not: case Op::Where: case Op::LayerNorm: case Op::Softmax:
        return true;
      default:
        return false;
    }
  }

  // In-place reuse: if this node is the last reader of an operand that has
  // exactly the node's own type, the result takes over the operand's memory.
  // Returns the operand's root, or kNoValue.
  ValueId in_place_operand(ValueId v) const {
    const Node& n = g_.nodes[v];
    if (!options_.plan_memory || !can_run_in_place(n.op)) return kNoValue;
    for (ValueId in : n.inputs) {
      ValueId r = root_[in];
      if (!owns_arena_[r] || last_use_[r] != static_cast<int64_t>(v) || g_.type_of(in) != n.type) continue;
      // Every operand that views the same bytes must be read at the output's
      // own index. A broadcast or reshaped second view would read elements
      // the kernel has already overwritten.
      bool safe = true;
      for (ValueId other : n.inputs)
        if (root_[other] == r && g_.type_of(other).shape != n.type.shape) safe = false;
      if (safe) return r;
    }
    return kNoValue;
  }

  // Decides where value `v` lives and creates its buffer.
  void place(ValueId v) {
    const Node& node = g_.nodes[v];
    Storage storage = Storage::Arena;
    int64_t offset = 0;
    int32_t index = -1;
    if (root_[v] != v) {
      const Buffer& owner = p_.buffers[buffer_of_[root_[v]]];
      storage = owner.storage;
      offset = owner.offset;
      index = owner.index;
    } else if (node.op == Op::Input) {
      storage = Storage::Input;
      index = node.attrs.index;
    } else if (node.op == Op::Const) {
      const Constant& c = g_.constants[node.attrs.index];
      if (c.is_external()) {
        storage = Storage::Weights;
        offset = c.weights_offset;
      } else {
        p_.inline_data.push_back(c.inline_data);
        storage = Storage::Inline;
        index = static_cast<int32_t>(p_.inline_data.size() - 1);
      }
    } else if (at_load(v)) {
      storage = Storage::Persistent;
      offset = persistent_end_;
      persistent_end_ += (static_cast<int64_t>(node.type.byte_size()) + kAlign - 1) / kAlign * kAlign;
    } else if (output_of_[v] >= 0) {
      storage = Storage::Output;
      index = output_of_[v];
    } else {
      ValueId donor = in_place_operand(v);
      if (donor != kNoValue) {
        block_[v] = block_[donor];
        owns_arena_[donor] = false;  // ownership moves; the donor must not free it
      } else {
        block_[v] = {arena_.alloc(static_cast<int64_t>(node.type.byte_size())), static_cast<int64_t>(node.type.byte_size())};
      }
      owns_arena_[v] = true;
      offset = block_[v].offset;
    }
    buffer_of_[v] = new_buffer("v" + std::to_string(v), node.type, storage, offset, index);
  }

  // After node `v` has been lowered: its scratch space is free again, and so
  // is every operand that `v` was the last reader of.
  void release_after(ValueId v) {
    for (const Block& t : temps_) arena_.release(t.offset, t.bytes);
    temps_.clear();
    auto drop = [&](ValueId r) {
      if (!owns_arena_[r]) return;
      arena_.release(block_[r].offset, block_[r].bytes);
      owns_arena_[r] = false;
    };
    for (ValueId in : g_.nodes[v].inputs)
      if (last_use_[root_[in]] == static_cast<int64_t>(v)) drop(root_[in]);
    if (root_[v] == v && last_use_[v] < 0) drop(v);  // computed but never read
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

  // A loop of `extent` iterations; one iteration gets the index 0 and no
  // loop. `id` receives the loop variable, or -1 if there is no loop.
  StmtPtr loop(int64_t extent, const std::function<StmtPtr(const ExprPtr&)>& body, VarId* id = nullptr) {
    if (id) *id = -1;
    if (extent == 1) return body(int_imm(0));
    VarId v = next_var_++;
    if (id) *id = v;
    return make_for(v, extent, body(var(v)));
  }

  // A loop whose extent depends on enclosing loop variables.
  StmtPtr loop(const ExprPtr& extent, const std::function<StmtPtr(const ExprPtr&)>& body) {
    if (extent->kind == ExprKind::IntImm) return extent->int_value == 0 ? make_block({}) : loop(extent->int_value, body);
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

  // A scratch buffer for an op that needs a temporary (a running maximum, a
  // mean). It lives only while its own kernel runs.
  BufferId temp(ValueId v, const char* what, DType dtype, const Shape& shape) {
    Type type{dtype, shape};
    Block b{arena_.alloc(static_cast<int64_t>(type.byte_size())), static_cast<int64_t>(type.byte_size())};
    temps_.push_back(b);
    return new_buffer("t" + std::to_string(v) + "_" + what, type, Storage::Arena, b.offset, -1);
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

  Kernel& emit(ValueId v, StmtPtr body) {
    const Node& n = g_.nodes[v];
    std::string label = "%" + std::to_string(v);
    if (!n.name.empty()) label += " " + n.name;
    std::vector<Kernel>& list = at_load(v) ? p_.init_kernels : p_.kernels;
    list.push_back(Kernel{op_name(n.op), std::move(label), std::move(body)});
    return list.back();
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

  // The same product with the second operand stored transposed, Bt[b.., j, p]:
  //
  //   C[b.., i, j] = init;  then  C[b.., i, j] += A[b.., i, p] * Bt[b.., j, p]
  //
  // Each element of C is a dot product of a row of A with a row of Bt, both
  // contiguous in memory. This is the naive nest; the schedule turns it into
  // the fast one, and needs to know which loop is which.
  void matmul_dot(ValueId v, const std::function<ExprPtr(const ExprPtr& j)>& init) {
    const Node& n = g_.nodes[v];
    ValueId a = n.inputs[0], b = n.inputs[1];
    const Shape& as = g_.type_of(a).shape;
    const int64_t m = as[as.size() - 2], k = as.back(), cols = n.type.shape.back();
    Shape batch(as.begin(), as.end() - 2);
    VarId vi = -1, vj = -1, vp = -1;
    StmtPtr body = nest(batch, [&](const Idx& bi) {
      return loop(m, [&](const ExprPtr& i) {
        return loop(cols, [&](const ExprPtr& j) {
          Idx c = concat_idx(concat_idx(bi, i), j);
          return make_block({
              write(v, c, init(j)),
              loop(k, [&](const ExprPtr& p) {
                ExprPtr product = mul(read(a, concat_idx(concat_idx(bi, i), p)), read(b, concat_idx(concat_idx(bi, j), p)));
                return write(v, c, add(load(buffer_of_[v], DType::F32, c), product));
              }, &vp),
          });
        }, &vj);
      }, &vi);
    });
    Kernel& kernel = emit(v, std::move(body));
    kernel.mm_i = vi;
    kernel.mm_j = vj;
    kernel.mm_p = vp;
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
    const ExprPtr minus_inf = float_imm(-std::numeric_limits<double>::infinity());
    return nest(rest, [&](const Idx& r) {
      ExprPtr max_r = load(max, DType::F32, r);
      ExprPtr sum_r = load(sum, DType::F32, r);
      auto at = [&](const ExprPtr& j) { return insert_at(r, axis, j); };
      auto out = [&](const ExprPtr& j) { return load(buffer_of_[v], DType::F32, at(j)); };
      // Causal: the axis is the last one, so the last loop index in `r` is the
      // row i, and only columns 0..i take part. The loops simply stop there:
      // their extent is i + 1, an expression of the enclosing loop variable.
      // Half the exponentials of the full square are never computed. The
      // columns after i are set to zero by a loop of their own.
      const ExprPtr live = n.attrs.causal ? add(r.back(), int_imm(1)) : int_imm(dim);
      std::vector<StmtPtr> body = {
          make_store(max, r, minus_inf),
          loop(live, [&](const ExprPtr& j) { return make_store(max, r, binary(BinOp::Max, max_r, read(x, at(j)))); }),
          make_store(sum, r, float_imm(0)),
          loop(live, [&](const ExprPtr& j) {
            return make_block({
                write(v, at(j), call(Intrinsic::Exp, {sub(read(x, at(j)), max_r)})),
                make_store(sum, r, add(sum_r, out(j))),
            });
          }),
          loop(live, [&](const ExprPtr& j) { return write(v, at(j), div(out(j), sum_r)); }),
      };
      if (n.attrs.causal)
        body.push_back(loop(sub(int_imm(dim), live), [&](const ExprPtr& z) { return write(v, at(add(live, z)), float_imm(0)); }));
      return make_block(std::move(body));
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

      case Op::Add: return (void)emit(v, binary_op(v, BinOp::Add));
      case Op::Sub: return (void)emit(v, binary_op(v, BinOp::Sub));
      case Op::Mul: return (void)emit(v, binary_op(v, BinOp::Mul));
      case Op::Eq: return (void)emit(v, binary_op(v, BinOp::Eq));
      case Op::Ne: return (void)emit(v, binary_op(v, BinOp::Ne));
      case Op::Le: return (void)emit(v, binary_op(v, BinOp::Le));
      case Op::And: return (void)emit(v, binary_op(v, BinOp::And));
      case Op::Pow: return (void)emit(v, elementwise(v, [](X x) { return call(Intrinsic::Pow, {x[0], x[1]}); }));
      case Op::Tanh: return (void)emit(v, elementwise(v, [](X x) { return call(Intrinsic::Tanh, {x[0]}); }));
      case Op::Gelu:
        // 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))) in one pass, with
        // x^3 as two multiplies rather than a call to pow.
        return (void)emit(v, elementwise(v, [](X x) {
          ExprPtr cube = mul(mul(x[0], x[0]), x[0]);
          ExprPtr inner = mul(add(x[0], mul(cube, float_imm(0.044715))), float_imm(0.7978845608028654));
          return mul(mul(x[0], float_imm(0.5)), add(call(Intrinsic::Tanh, {inner}), float_imm(1)));
        }));
      case Op::Not: return (void)emit(v, elementwise(v, [](X x) { return logical_not(x[0]); }));
      case Op::Where: return (void)emit(v, elementwise(v, [](X x) { return select(x[0], x[1], x[2]); }));
      case Op::Cast: return (void)emit(v, elementwise(v, [&](X x) { return cast(n.attrs.to, x[0]); }));
      case Op::Broadcast: return (void)emit(v, elementwise(v, [](X x) { return x[0]; }));

      case Op::Permute: return (void)emit(v, permute(v));
      case Op::Slice: return (void)emit(v, slice(v));
      case Op::Concat: return (void)emit(v, concat(v));
      case Op::Gather: return (void)emit(v, gather(v));
      case Op::CumSum: return (void)emit(v, cumsum(v));
      case Op::ReduceAny: return (void)emit(v, reduce_any(v));

      case Op::MatMul: {
        auto zero = [](const ExprPtr&) { return float_imm(0); };
        if (n.attrs.trans_b) return matmul_dot(v, zero);
        emit(v, matmul_like(v, zero));
        return;
      }
      case Op::Gemm: {
        auto bias = [&](const ExprPtr& j) { return read(n.inputs[2], {j}); };
        if (n.attrs.trans_b) return matmul_dot(v, bias);
        emit(v, matmul_like(v, bias));
        return;
      }
      case Op::LayerNorm: return (void)emit(v, layer_norm(v));
      case Op::Softmax: return (void)emit(v, softmax(v));
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

Program lower_graph(const Graph& g, int64_t weights_bytes, const LowerOptions& options) { return Lowerer(g, weights_bytes, options).run(); }

}  // namespace mlc::loop
