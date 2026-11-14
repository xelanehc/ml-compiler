#include "loop/ir.h"

#include <cmath>
#include <cstdio>
#include <set>

namespace mlc::loop {

// ---- Expression constructors -----------------------------------------------------

namespace {

ExprPtr make(Expr e) { return std::make_shared<const Expr>(std::move(e)); }

Expr new_expr(ExprKind kind, DType type) {
  Expr e;
  e.kind = kind;
  e.type = type;
  return e;
}

Stmt new_stmt(StmtKind kind) {
  Stmt s;
  s.kind = kind;
  return s;
}

bool is_int_imm(const ExprPtr& e, int64_t v) { return e->kind == ExprKind::IntImm && e->type == DType::I64 && e->int_value == v; }
bool is_int_const(const ExprPtr& e) { return e->kind == ExprKind::IntImm && e->type == DType::I64; }

const char* binop_name(BinOp op) {
  switch (op) {
    case BinOp::Add: return "+";
    case BinOp::Sub: return "-";
    case BinOp::Mul: return "*";
    case BinOp::Div: return "/";
    case BinOp::Max: return "max";
    case BinOp::Eq: return "==";
    case BinOp::Ne: return "!=";
    case BinOp::Le: return "<=";
    case BinOp::And: return "&";
    case BinOp::Or: return "|";
  }
  return "?";
}

const char* intrinsic_name(Intrinsic fn) {
  switch (fn) {
    case Intrinsic::Exp: return "exp";
    case Intrinsic::Tanh: return "tanh";
    case Intrinsic::Pow: return "pow";
    case Intrinsic::Sqrt: return "sqrt";
  }
  return "?";
}

}  // namespace

ExprPtr int_imm(int64_t v) {
  Expr e = new_expr(ExprKind::IntImm, DType::I64);
  e.int_value = v;
  return make(std::move(e));
}

ExprPtr bool_imm(bool v) {
  Expr e = new_expr(ExprKind::IntImm, DType::Bool);
  e.int_value = v;
  return make(std::move(e));
}

ExprPtr float_imm(double v) {
  Expr e = new_expr(ExprKind::FloatImm, DType::F32);
  e.float_value = static_cast<float>(v);  // the value an f32 can actually hold
  return make(std::move(e));
}

ExprPtr var(VarId id) {
  Expr e = new_expr(ExprKind::Var, DType::I64);
  e.var = id;
  return make(std::move(e));
}

ExprPtr load(BufferId buffer, DType dtype, std::vector<ExprPtr> indices) {
  for (const ExprPtr& i : indices)
    if (i->type != DType::I64) throw LoopError("load: indices must be i64");
  Expr e = new_expr(ExprKind::Load, dtype);
  e.buffer = buffer;
  e.args = std::move(indices);
  return make(std::move(e));
}

ExprPtr binary(BinOp op, ExprPtr a, ExprPtr b) {
  if (a->type != b->type) throw LoopError(std::string("binary ") + binop_name(op) + ": operand types differ");
  DType in = a->type;
  DType out = in;
  switch (op) {
    case BinOp::Add:
    case BinOp::Sub:
    case BinOp::Mul:
      if (in == DType::Bool) throw LoopError("arithmetic on bool");
      break;
    case BinOp::Div:
    case BinOp::Max:
      if (in != DType::F32) throw LoopError(std::string(binop_name(op)) + " is defined for f32 only");
      break;
    case BinOp::Eq:
    case BinOp::Ne:
    case BinOp::Le:
      out = DType::Bool;
      break;
    case BinOp::And:
    case BinOp::Or:
      if (in != DType::Bool) throw LoopError("logical operator on a non-bool");
      break;
  }

  // Integer constant folding, enough to keep index expressions tidy.
  if (in == DType::I64) {
    if (op == BinOp::Add) {
      if (is_int_const(a) && is_int_const(b)) return int_imm(a->int_value + b->int_value);
      if (is_int_imm(a, 0)) return b;
      if (is_int_imm(b, 0)) return a;
    }
    if (op == BinOp::Mul) {
      if (is_int_const(a) && is_int_const(b)) return int_imm(a->int_value * b->int_value);
      if (is_int_imm(a, 0) || is_int_imm(b, 0)) return int_imm(0);
      if (is_int_imm(a, 1)) return b;
      if (is_int_imm(b, 1)) return a;
    }
  }

  Expr e = new_expr(ExprKind::Binary, out);
  e.binop = op;
  e.args = {std::move(a), std::move(b)};
  return make(std::move(e));
}

ExprPtr logical_not(ExprPtr a) {
  if (a->type != DType::Bool) throw LoopError("not: operand must be bool");
  Expr e = new_expr(ExprKind::Not, DType::Bool);
  e.args = {std::move(a)};
  return make(std::move(e));
}

ExprPtr call(Intrinsic fn, std::vector<ExprPtr> args) {
  size_t arity = fn == Intrinsic::Pow ? 2 : 1;
  if (args.size() != arity) throw LoopError(std::string(intrinsic_name(fn)) + ": wrong number of arguments");
  for (const ExprPtr& a : args)
    if (a->type != DType::F32) throw LoopError(std::string(intrinsic_name(fn)) + ": arguments must be f32");
  Expr e = new_expr(ExprKind::Call, DType::F32);
  e.intrinsic = fn;
  e.args = std::move(args);
  return make(std::move(e));
}

ExprPtr select(ExprPtr cond, ExprPtr if_true, ExprPtr if_false) {
  if (cond->type != DType::Bool) throw LoopError("select: condition must be bool");
  if (if_true->type != if_false->type) throw LoopError("select: branch types differ");
  Expr e = new_expr(ExprKind::Select, if_true->type);
  e.args = {std::move(cond), std::move(if_true), std::move(if_false)};
  return make(std::move(e));
}

ExprPtr cast(DType to, ExprPtr a) {
  if (a->type == to) return a;
  Expr e = new_expr(ExprKind::Cast, to);
  e.args = {std::move(a)};
  return make(std::move(e));
}

// ---- Statement constructors ---------------------------------------------------------

StmtPtr make_for(VarId v, int64_t extent, StmtPtr body) {
  Stmt s = new_stmt(StmtKind::For);
  s.var = v;
  s.extent = extent;
  s.body = std::move(body);
  return std::make_shared<const Stmt>(std::move(s));
}

StmtPtr make_store(BufferId buffer, std::vector<ExprPtr> indices, ExprPtr value) {
  Stmt s = new_stmt(StmtKind::Store);
  s.buffer = buffer;
  s.indices = std::move(indices);
  s.value = std::move(value);
  return std::make_shared<const Stmt>(std::move(s));
}

StmtPtr make_block(std::vector<StmtPtr> stmts) {
  Stmt s = new_stmt(StmtKind::Block);
  s.stmts = std::move(stmts);
  return std::make_shared<const Stmt>(std::move(s));
}

// Horner's rule: for shape [n0, n1, n2] and index [i0, i1, i2] the offset is
// (i0 * n1 + i1) * n2 + i2, one multiply and one add per axis.
ExprPtr flat_index(const Buffer& buffer, const std::vector<ExprPtr>& indices) {
  if (indices.size() != buffer.shape.size()) throw LoopError("access to " + buffer.name + " has the wrong number of indices");
  ExprPtr flat = int_imm(0);
  for (size_t d = 0; d < indices.size(); ++d) flat = add(mul(flat, int_imm(buffer.shape[d])), indices[d]);
  return flat;
}

// ---- Printer -----------------------------------------------------------------------

namespace {

std::string indices_string(const Program& p, const std::vector<ExprPtr>& indices) {
  std::string out = "[";
  for (size_t i = 0; i < indices.size(); ++i) out += (i ? ", " : "") + print_expr(p, indices[i]);
  return out + "]";
}

void print_stmt(const Program& p, const StmtPtr& s, int depth, std::string& out) {
  std::string pad(static_cast<size_t>(depth) * 2, ' ');
  switch (s->kind) {
    case StmtKind::For:
      out += pad + "for i" + std::to_string(s->var) + " in 0.." + std::to_string(s->extent) + ":\n";
      print_stmt(p, s->body, depth + 1, out);
      break;
    case StmtKind::Store:
      out += pad + p.buffers.at(s->buffer).name + indices_string(p, s->indices) + " = " + print_expr(p, s->value) + "\n";
      break;
    case StmtKind::Block:
      for (const StmtPtr& child : s->stmts) print_stmt(p, child, depth, out);
      break;
  }
}

}  // namespace

std::string print_expr(const Program& p, const ExprPtr& e) {
  switch (e->kind) {
    case ExprKind::IntImm:
      if (e->type == DType::Bool) return e->int_value ? "true" : "false";
      return std::to_string(e->int_value);
    case ExprKind::FloatImm: {
      char buf[32];
      std::snprintf(buf, sizeof buf, "%g", e->float_value);
      return buf;
    }
    case ExprKind::Var: return "i" + std::to_string(e->var);
    case ExprKind::Load: return p.buffers.at(e->buffer).name + indices_string(p, e->args);
    case ExprKind::Binary:
      if (e->binop == BinOp::Max) return "max(" + print_expr(p, e->args[0]) + ", " + print_expr(p, e->args[1]) + ")";
      return "(" + print_expr(p, e->args[0]) + " " + binop_name(e->binop) + " " + print_expr(p, e->args[1]) + ")";
    case ExprKind::Not: return "!" + print_expr(p, e->args[0]);
    case ExprKind::Call: {
      std::string out = std::string(intrinsic_name(e->intrinsic)) + "(";
      for (size_t i = 0; i < e->args.size(); ++i) out += (i ? ", " : "") + print_expr(p, e->args[i]);
      return out + ")";
    }
    case ExprKind::Select:
      return "(" + print_expr(p, e->args[0]) + " ? " + print_expr(p, e->args[1]) + " : " + print_expr(p, e->args[2]) + ")";
    case ExprKind::Cast: return std::string(dtype_name(e->type)) + "(" + print_expr(p, e->args[0]) + ")";
  }
  return "?";
}

std::string print_kernel(const Program& p, const Kernel& k) {
  std::string out = "kernel " + k.op + "  # " + k.label + "\n";
  print_stmt(p, k.body, 1, out);
  return out;
}

std::string print_program(const Program& p) {
  std::string out;
  for (const Buffer& b : p.buffers) {
    static const char* where[] = {"arena", "weights", "input", "output", "inline"};
    out += "buffer " + b.name + ": " + type_string({b.dtype, b.shape}) + " @ " + where[static_cast<int>(b.storage)];
    if (b.storage == Storage::Arena || b.storage == Storage::Weights) out += "+" + std::to_string(b.offset);
    else out += "[" + std::to_string(b.index) + "]";
    out += "\n";
  }
  out += "arena: " + std::to_string(p.arena_bytes) + " bytes\n\n";
  for (const Kernel& k : p.kernels) out += print_kernel(p, k) + "\n";
  return out;
}

// ---- Verifier -----------------------------------------------------------------------

namespace {

struct Verifier {
  const Program& p;
  std::vector<std::string> errors;
  std::string where;
  std::set<VarId> scope;

  void report(const std::string& what) { errors.push_back(where + ": " + what); }

  bool buffer_ok(BufferId id) {
    if (id >= 0 && static_cast<size_t>(id) < p.buffers.size()) return true;
    report("refers to a buffer that does not exist");
    return false;
  }

  void access(BufferId id, const std::vector<ExprPtr>& indices) {
    if (!buffer_ok(id)) return;
    const Buffer& b = p.buffers[id];
    if (indices.size() != b.shape.size())
      report(b.name + " has rank " + std::to_string(b.shape.size()) + " but is accessed with " + std::to_string(indices.size()) + " indices");
    for (const ExprPtr& i : indices) {
      if (i->type != DType::I64) report("index into " + b.name + " is not i64");
      expr(i);
    }
  }

  void expr(const ExprPtr& e) {
    switch (e->kind) {
      case ExprKind::IntImm:
      case ExprKind::FloatImm: return;
      case ExprKind::Var:
        if (!scope.count(e->var)) report("uses loop variable i" + std::to_string(e->var) + " outside its loop");
        return;
      case ExprKind::Load:
        access(e->buffer, e->args);
        if (buffer_ok(e->buffer) && p.buffers[e->buffer].dtype != e->type) report("load from " + p.buffers[e->buffer].name + " has the wrong type");
        return;
      default:
        for (const ExprPtr& a : e->args) expr(a);
    }
  }

  void stmt(const StmtPtr& s) {
    switch (s->kind) {
      case StmtKind::For:
        if (s->extent < 0) report("loop with a negative extent");
        if (!scope.insert(s->var).second) report("loop variable i" + std::to_string(s->var) + " is bound twice");
        stmt(s->body);
        scope.erase(s->var);
        return;
      case StmtKind::Store:
        access(s->buffer, s->indices);
        expr(s->value);
        if (buffer_ok(s->buffer)) {
          const Buffer& b = p.buffers[s->buffer];
          if (!b.writable()) report("stores to read-only buffer " + b.name);
          if (b.dtype != s->value->type) report("stores a " + std::string(dtype_name(s->value->type)) + " into " + b.name);
        }
        return;
      case StmtKind::Block:
        for (const StmtPtr& child : s->stmts) stmt(child);
        return;
    }
  }
};

}  // namespace

std::vector<std::string> verify_program(const Program& p) {
  Verifier v{p, {}, "", {}};
  for (const Buffer& b : p.buffers) {
    v.where = "buffer " + b.name;
    int64_t bytes = static_cast<int64_t>(Type{b.dtype, b.shape}.byte_size());
    switch (b.storage) {
      case Storage::Arena:
        if (b.offset < 0 || b.offset + bytes > p.arena_bytes) v.report("lies outside the arena");
        break;
      case Storage::Weights:
        if (b.offset < 0 || b.offset + bytes > p.weights_bytes) v.report("lies outside the weights file");
        break;
      case Storage::Input:
        if (b.index < 0 || static_cast<size_t>(b.index) >= p.inputs.size()) v.report("names an input that does not exist");
        break;
      case Storage::Output:
        if (b.index < 0 || static_cast<size_t>(b.index) >= p.outputs.size()) v.report("names an output that does not exist");
        break;
      case Storage::Inline:
        if (b.index < 0 || static_cast<size_t>(b.index) >= p.inline_data.size() ||
            static_cast<int64_t>(p.inline_data[b.index].size()) != bytes)
          v.report("inline data is missing or the wrong size");
        break;
    }
  }
  for (const Kernel& k : p.kernels) {
    v.where = "kernel " + k.label;
    v.scope.clear();
    v.stmt(k.body);
  }
  return v.errors;
}

void check_program(const Program& p) {
  std::vector<std::string> errors = verify_program(p);
  if (errors.empty()) return;
  std::string msg = "loop program verification failed:";
  for (const std::string& e : errors) msg += "\n  " + e;
  throw LoopError(msg);
}

}  // namespace mlc::loop
