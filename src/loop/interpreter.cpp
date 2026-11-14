// Executes a loop program by walking its tree. Roughly a hundred times slower
// than compiled code, which is fine: its job is to be the reference for what a
// loop program means, on inputs small enough not to care.
#include <cmath>
#include <cstring>

#include "loop/ir.h"

namespace mlc::loop {

namespace {

class Machine {
 public:
  Machine(const Program& p, std::span<const std::byte> weights, std::span<const void* const> inputs, std::span<void* const> outputs)
      : p_(p), arena_(static_cast<size_t>(p.arena_bytes)) {
    if (inputs.size() != p.inputs.size() || outputs.size() != p.outputs.size()) throw LoopError("wrong number of inputs or outputs");
    if (static_cast<int64_t>(weights.size()) < p.weights_bytes) throw LoopError("weights are smaller than the program expects");
    for (const Buffer& b : p.buffers) {
      switch (b.storage) {
        case Storage::Arena: base_.push_back(arena_.data() + b.offset); break;
        case Storage::Weights: base_.push_back(const_cast<std::byte*>(weights.data()) + b.offset); break;
        case Storage::Input: base_.push_back(static_cast<std::byte*>(const_cast<void*>(inputs[b.index]))); break;
        case Storage::Output: base_.push_back(static_cast<std::byte*>(outputs[b.index])); break;
        case Storage::Inline: base_.push_back(const_cast<std::byte*>(p.inline_data[b.index].data())); break;
      }
    }
  }

  void run() {
    for (const Kernel& k : p_.kernels) {
      vars_.clear();
      exec(k.body);
    }
  }

 private:
  const Program& p_;
  std::vector<std::byte> arena_;
  std::vector<std::byte*> base_;   // start of each buffer's data
  std::vector<int64_t> vars_;      // current value of each loop variable, by VarId

  // The address of buffer[indices]. Each index is checked against its own
  // axis, which is stricter than checking the flattened offset: an index that
  // runs off one axis but lands inside the buffer is still a bug.
  std::byte* address(BufferId id, const std::vector<ExprPtr>& indices) {
    const Buffer& b = p_.buffers[id];
    int64_t flat = 0;
    for (size_t d = 0; d < indices.size(); ++d) {
      int64_t i = eval_int(indices[d]);
      if (i < 0 || i >= b.shape[d])
        throw LoopError("index " + std::to_string(i) + " out of range on axis " + std::to_string(d) + " of " + b.name + " " +
                        shape_string(b.shape));
      flat = flat * b.shape[d] + i;
    }
    return base_[id] + flat * static_cast<int64_t>(dtype_size(b.dtype));
  }

  // Expressions come in two flavours, so there are two evaluators: one for
  // f32 values and one for i64 and bool (bool is 0 or 1).
  float eval_float(const ExprPtr& e) {
    switch (e->kind) {
      case ExprKind::FloatImm: return static_cast<float>(e->float_value);
      case ExprKind::Load: {
        float v;
        std::memcpy(&v, address(e->buffer, e->args), sizeof v);
        return v;
      }
      case ExprKind::Binary: {
        float a = eval_float(e->args[0]), b = eval_float(e->args[1]);
        switch (e->binop) {
          case BinOp::Add: return a + b;
          case BinOp::Sub: return a - b;
          case BinOp::Mul: return a * b;
          case BinOp::Div: return a / b;
          case BinOp::Max: return a > b ? a : b;
          default: break;
        }
        break;
      }
      case ExprKind::Call: {
        float a = eval_float(e->args[0]);
        switch (e->intrinsic) {
          case Intrinsic::Exp: return std::exp(a);
          case Intrinsic::Tanh: return std::tanh(a);
          case Intrinsic::Sqrt: return std::sqrt(a);
          case Intrinsic::Pow: return std::pow(a, eval_float(e->args[1]));
        }
        break;
      }
      case ExprKind::Select: return eval_int(e->args[0]) ? eval_float(e->args[1]) : eval_float(e->args[2]);
      case ExprKind::Cast: return static_cast<float>(eval_int(e->args[0]));
      default: break;
    }
    throw LoopError("malformed f32 expression");
  }

  int64_t eval_int(const ExprPtr& e) {
    switch (e->kind) {
      case ExprKind::IntImm: return e->int_value;
      case ExprKind::Var: return vars_[e->var];
      case ExprKind::Load: {
        std::byte* at = address(e->buffer, e->args);
        if (e->type == DType::Bool) return at[0] != std::byte{0};
        int64_t v;
        std::memcpy(&v, at, sizeof v);
        return v;
      }
      case ExprKind::Binary: {
        const ExprPtr& lhs = e->args[0];
        const ExprPtr& rhs = e->args[1];
        if (lhs->type == DType::F32) {  // a comparison of floats yields a bool
          float a = eval_float(lhs), b = eval_float(rhs);
          switch (e->binop) {
            case BinOp::Eq: return a == b;
            case BinOp::Ne: return a != b;
            case BinOp::Le: return a <= b;
            default: break;
          }
          break;
        }
        int64_t a = eval_int(lhs), b = eval_int(rhs);
        switch (e->binop) {
          case BinOp::Add: return a + b;
          case BinOp::Sub: return a - b;
          case BinOp::Mul: return a * b;
          case BinOp::Eq: return a == b;
          case BinOp::Ne: return a != b;
          case BinOp::Le: return a <= b;
          case BinOp::And: return a & b;
          case BinOp::Or: return a | b;
          default: break;
        }
        break;
      }
      case ExprKind::Not: return !eval_int(e->args[0]);
      case ExprKind::Select: return eval_int(e->args[0]) ? eval_int(e->args[1]) : eval_int(e->args[2]);
      case ExprKind::Cast: {
        const ExprPtr& a = e->args[0];
        if (a->type == DType::F32) {
          float v = eval_float(a);
          return e->type == DType::Bool ? v != 0 : static_cast<int64_t>(v);
        }
        int64_t v = eval_int(a);
        return e->type == DType::Bool ? v != 0 : v;
      }
      default: break;
    }
    throw LoopError("malformed integer expression");
  }

  void exec(const StmtPtr& s) {
    switch (s->kind) {
      case StmtKind::For: {
        if (static_cast<size_t>(s->var) >= vars_.size()) vars_.resize(s->var + 1);
        for (int64_t i = 0; i < s->extent; ++i) {
          vars_[s->var] = i;
          exec(s->body);
        }
        return;
      }
      case StmtKind::Store: {
        const Buffer& b = p_.buffers[s->buffer];
        if (b.dtype == DType::F32) {
          float v = eval_float(s->value);
          std::memcpy(address(s->buffer, s->indices), &v, sizeof v);
        } else if (b.dtype == DType::I64) {
          int64_t v = eval_int(s->value);
          std::memcpy(address(s->buffer, s->indices), &v, sizeof v);
        } else {
          *address(s->buffer, s->indices) = std::byte{static_cast<unsigned char>(eval_int(s->value) != 0)};
        }
        return;
      }
      case StmtKind::Block:
        for (const StmtPtr& child : s->stmts) exec(child);
        return;
    }
  }
};

}  // namespace

void run_program(const Program& p, std::span<const std::byte> weights, std::span<const void* const> inputs, std::span<void* const> outputs) {
  check_program(p);
  Machine(p, weights, inputs, outputs).run();
}

}  // namespace mlc::loop
