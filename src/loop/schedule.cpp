#include "loop/schedule.h"

#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace mlc::loop {

namespace {

// ---- Small tree utilities ------------------------------------------------------------

bool contains_var(const ExprPtr& e, VarId v) {
  if (e->kind == ExprKind::Var) return e->var == v;
  for (const ExprPtr& a : e->args)
    if (contains_var(a, v)) return true;
  return false;
}

bool equal(const ExprPtr& a, const ExprPtr& b) {
  if (a == b) return true;
  if (a->kind != b->kind || a->type != b->type || a->args.size() != b->args.size()) return false;
  switch (a->kind) {
    case ExprKind::IntImm: if (a->int_value != b->int_value) return false; break;
    case ExprKind::FloatImm: if (a->float_value != b->float_value) return false; break;
    case ExprKind::Var: if (a->var != b->var) return false; break;
    case ExprKind::Load: if (a->buffer != b->buffer) return false; break;
    case ExprKind::Binary: if (a->binop != b->binop) return false; break;
    case ExprKind::Call: if (a->intrinsic != b->intrinsic) return false; break;
    default: break;
  }
  for (size_t i = 0; i < a->args.size(); ++i)
    if (!equal(a->args[i], b->args[i])) return false;
  return true;
}

bool equal(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); ++i)
    if (!equal(a[i], b[i])) return false;
  return true;
}

std::vector<ExprPtr> substitute(const std::vector<ExprPtr>& es, VarId v, const ExprPtr& r) {
  std::vector<ExprPtr> out;
  for (const ExprPtr& e : es) out.push_back(substitute(e, v, r));
  return out;
}

StmtPtr substitute(const StmtPtr& s, VarId v, const ExprPtr& r) {
  switch (s->kind) {
    case StmtKind::For: return make_for(s->var, substitute(s->extent, v, r), substitute(s->body, v, r), s->loop_kind, s->interleave);
    case StmtKind::Store: return make_store(s->buffer, substitute(s->indices, v, r), substitute(s->value, v, r));
    case StmtKind::Block: {
      std::vector<StmtPtr> out;
      for (const StmtPtr& c : s->stmts) out.push_back(substitute(c, v, r));
      return make_block(std::move(out));
    }
  }
  return s;
}

// A block with nested blocks spliced in, and no block at all around one statement.
StmtPtr block_of(std::vector<StmtPtr> stmts) {
  std::vector<StmtPtr> flat;
  for (StmtPtr& s : stmts) {
    if (s->kind == StmtKind::Block) flat.insert(flat.end(), s->stmts.begin(), s->stmts.end());
    else flat.push_back(std::move(s));
  }
  if (flat.size() == 1) return flat[0];
  return make_block(std::move(flat));
}

void max_var(const ExprPtr& e, VarId& m) {
  if (e->kind == ExprKind::Var) m = std::max(m, e->var);
  for (const ExprPtr& a : e->args) max_var(a, m);
}

void max_var(const StmtPtr& s, VarId& m) {
  switch (s->kind) {
    case StmtKind::For:
      m = std::max(m, s->var);
      max_var(s->extent, m);
      max_var(s->body, m);
      break;
    case StmtKind::Store:
      for (const ExprPtr& i : s->indices) max_var(i, m);
      max_var(s->value, m);
      break;
    case StmtKind::Block:
      for (const StmtPtr& c : s->stmts) max_var(c, m);
      break;
  }
}

VarId fresh_var(const StmtPtr& body) {
  VarId m = -1;
  max_var(body, m);
  return m + 1;
}

bool binds(const StmtPtr& s, VarId v) {
  if (s->kind == StmtKind::For) return s->var == v || binds(s->body, v);
  for (const StmtPtr& c : s->stmts)
    if (binds(c, v)) return true;
  return false;
}

// ---- Affine index expressions and value ranges ---------------------------------------
//
// Every index the lowering produces is a sum of loop variables times constants,
// plus a constant. Writing it in that form is what lets the dependence test
// reason about which elements an access can touch.

struct Affine {
  std::map<VarId, int64_t> coef;
  int64_t constant = 0;
};

std::optional<Affine> affine(const ExprPtr& e) {
  if (e->type != DType::I64) return std::nullopt;
  switch (e->kind) {
    case ExprKind::IntImm: return Affine{{}, e->int_value};
    case ExprKind::Var: return Affine{{{e->var, 1}}, 0};
    case ExprKind::Binary: {
      std::optional<Affine> a = affine(e->args[0]), b = affine(e->args[1]);
      if (!a || !b) return std::nullopt;
      if (e->binop == BinOp::Add || e->binop == BinOp::Sub) {
        const int64_t sign = e->binop == BinOp::Add ? 1 : -1;
        for (auto& [v, c] : b->coef) a->coef[v] += sign * c;
        a->constant += sign * b->constant;
        return a;
      }
      if (e->binop == BinOp::Mul) {
        if (!a->coef.empty() && !b->coef.empty()) return std::nullopt;  // variable times variable
        const Affine& vars = a->coef.empty() ? *b : *a;
        const int64_t k = a->coef.empty() ? a->constant : b->constant;
        Affine out;
        for (auto& [v, c] : vars.coef) out.coef[v] = c * k;
        out.constant = vars.constant * k;
        return out;
      }
      return std::nullopt;
    }
    default: return std::nullopt;
  }
}

// var -> the largest value it takes. Every loop starts at 0.
using Ranges = std::map<VarId, int64_t>;

int64_t bound(const Affine& a, const Ranges& ranges, bool upper) {
  int64_t out = a.constant;
  for (auto& [v, c] : a.coef) {
    auto it = ranges.find(v);
    if (it == ranges.end()) throw ScheduleError("index uses a variable with no known range");
    if ((c > 0) == upper) out += c * it->second;
  }
  return out;
}

// Adds the range of `loop`'s variable. A loop that never runs gets -1.
void add_range(const Stmt& loop, Ranges& ranges) {
  std::optional<Affine> extent = affine(loop.extent);
  if (!extent) throw ScheduleError("a loop extent is not an affine expression");
  ranges[loop.var] = bound(*extent, ranges, true) - 1;
}

void add_inner_ranges(const StmtPtr& s, Ranges& ranges, std::set<VarId>& inner) {
  if (s->kind == StmtKind::For) {
    add_range(*s, ranges);
    inner.insert(s->var);
    add_inner_ranges(s->body, ranges, inner);
  }
  for (const StmtPtr& c : s->stmts) add_inner_ranges(c, ranges, inner);
}

// ---- Accesses ----------------------------------------------------------------------------

struct Access {
  BufferId buffer;
  const std::vector<ExprPtr>* indices;
  bool store;
};

void collect(const ExprPtr& e, std::vector<Access>& out) {
  if (e->kind == ExprKind::Load) out.push_back({e->buffer, &e->args, false});
  for (const ExprPtr& a : e->args) collect(a, out);
}

void collect(const StmtPtr& s, std::vector<Access>& out) {
  switch (s->kind) {
    case StmtKind::For:
      collect(s->extent, out);
      collect(s->body, out);
      break;
    case StmtKind::Store:
      out.push_back({s->buffer, &s->indices, true});
      for (const ExprPtr& i : s->indices) collect(i, out);
      collect(s->value, out);
      break;
    case StmtKind::Block:
      for (const StmtPtr& c : s->stmts) collect(c, out);
      break;
  }
}

// The dependence test for one loop. `outer` holds the ranges of the loops around it.
bool independent(const Program& p, const Stmt& loop, Ranges ranges, std::string* why) {
  auto fail = [&](const std::string& what) {
    if (why) *why = what;
    return false;
  };
  const VarId v = loop.var;
  std::set<VarId> inner;
  try {
    add_range(loop, ranges);
    add_inner_ranges(loop.body, ranges, inner);
  } catch (const ScheduleError& e) {
    return fail(e.what());
  }

  std::vector<Access> accesses;
  collect(loop.body, accesses);
  std::set<BufferId> done;
  for (const Access& s : accesses) {
    if (!s.store || !done.insert(s.buffer).second) continue;
    const Buffer& target = p.buffers[s.buffer];

    // Everything in the body that can touch the bytes this store writes.
    std::vector<const Access*> group;
    for (const Access& x : accesses) {
      const Buffer& b = p.buffers[x.buffer];
      if (!overlaps(b, target)) continue;
      if (!same_view(b, target)) return fail(target.name + " and " + b.name + " share memory under different layouts");
      group.push_back(&x);
    }

    bool found = false;
    for (size_t d = 0; d < target.shape.size() && !found; ++d) {
      bool ok = true;
      int64_t stride = 0, lo = 0, hi = 0;
      std::map<VarId, int64_t> outer_part;
      for (size_t g = 0; g < group.size() && ok; ++g) {
        std::optional<Affine> a = affine((*group[g]->indices)[d]);
        if (!a || a->coef[v] < 1) {
          ok = false;
          break;
        }
        // Split the index into  stride * v  +  (inner variables and constant)  +  (outer variables).
        Affine rest;
        rest.constant = a->constant;
        std::map<VarId, int64_t> outer;
        for (auto& [var, c] : a->coef) {
          if (var == v || c == 0) continue;
          if (inner.count(var)) rest.coef[var] = c;
          else outer[var] = c;
        }
        const int64_t r_lo = bound(rest, ranges, false), r_hi = bound(rest, ranges, true);
        if (g == 0) {
          stride = a->coef[v];
          outer_part = outer;
          lo = r_lo;
          hi = r_hi;
        } else {
          ok = a->coef[v] == stride && outer == outer_part;
          lo = std::min(lo, r_lo);
          hi = std::max(hi, r_hi);
        }
      }
      // Two iterations v1 != v2 would collide only if their remainders
      // differed by a multiple of the stride, which a spread below it rules out.
      found = ok && hi - lo < stride;
    }
    if (!found) return fail("iterations of i" + std::to_string(v) + " do not write disjoint parts of " + target.name);
  }
  return true;
}

// True when two accesses to the same buffer are certain to name different
// elements on every iteration of the loop over `v`: along some axis that the
// loop does not move, their indices differ by a nonzero constant.
bool always_distinct(const std::vector<ExprPtr>& a, const std::vector<ExprPtr>& b, VarId v) {
  for (size_t d = 0; d < a.size() && d < b.size(); ++d) {
    if (contains_var(a[d], v) || contains_var(b[d], v)) continue;
    std::optional<Affine> x = affine(a[d]), y = affine(b[d]);
    if (!x || !y) continue;
    auto drop_zeros = [](Affine& f) { std::erase_if(f.coef, [](const auto& term) { return term.second == 0; }); };
    drop_zeros(*x);
    drop_zeros(*y);
    if (x->coef == y->coef && x->constant != y->constant) return true;
  }
  return false;
}

// Rebuilds `body`, replacing each loop that binds `v` by what `f` returns for
// it. `f` also gets the ranges of the loops around that loop.
StmtPtr rewrite_loops(const StmtPtr& s, VarId v, Ranges& ranges, const std::function<StmtPtr(const Stmt&, const Ranges&)>& f) {
  switch (s->kind) {
    case StmtKind::For: {
      if (s->var == v) return f(*s, ranges);
      if (!binds(s->body, v)) return s;
      Ranges inner = ranges;
      add_range(*s, inner);
      return make_for(s->var, s->extent, rewrite_loops(s->body, v, inner, f), s->loop_kind, s->interleave);
    }
    case StmtKind::Store: return s;
    case StmtKind::Block: {
      std::vector<StmtPtr> out;
      for (const StmtPtr& c : s->stmts) out.push_back(rewrite_loops(c, v, ranges, f));
      return make_block(std::move(out));
    }
  }
  return s;
}

StmtPtr rewrite_loops(const StmtPtr& body, VarId v, const std::function<StmtPtr(const Stmt&, const Ranges&)>& f) {
  if (!binds(body, v)) throw ScheduleError("no loop binds i" + std::to_string(v));
  Ranges ranges;
  return rewrite_loops(body, v, ranges, f);
}

int64_t require_const_extent(const Stmt& loop, const char* what) {
  int64_t n = const_extent(loop);
  if (n < 0) throw ScheduleError(std::string(what) + ": the loop over i" + std::to_string(loop.var) + " has no constant extent");
  return n;
}

void require_serial(const Stmt& loop, const char* what) {
  if (loop.loop_kind != LoopKind::Serial) throw ScheduleError(std::string(what) + ": the loop over i" + std::to_string(loop.var) + " is already scheduled");
}

void require_independent(const Program& p, const Stmt& loop, const Ranges& ranges, const char* what) {
  std::string why;
  if (!independent(p, loop, ranges, &why)) throw ScheduleError(std::string(what) + " is not legal: " + why);
}

bool extents_use(const StmtPtr& s, VarId v) {
  if (s->kind == StmtKind::For) return contains_var(s->extent, v) || extents_use(s->body, v);
  for (const StmtPtr& c : s->stmts)
    if (extents_use(c, v)) return true;
  return false;
}

std::vector<StmtPtr> stores_of(const StmtPtr& body) {
  if (body->kind == StmtKind::Store) return {body};
  if (body->kind == StmtKind::Block) {
    std::vector<StmtPtr> out;
    for (const StmtPtr& c : body->stmts) {
      if (c->kind != StmtKind::Store) return {};
      out.push_back(c);
    }
    return out;
  }
  return {};
}

}  // namespace

// ---- Analysis ----------------------------------------------------------------------------

bool overlaps(const Buffer& a, const Buffer& b) {
  if (a.storage != b.storage) return false;
  switch (a.storage) {
    case Storage::Input:
    case Storage::Output:
    case Storage::Inline: return a.index == b.index;  // one caller pointer or one table; distinct ones are distinct memory
    default: return a.offset < b.offset + b.bytes() && b.offset < a.offset + a.bytes();
  }
}

bool same_view(const Buffer& a, const Buffer& b) {
  return overlaps(a, b) && a.offset == b.offset && a.index == b.index && a.dtype == b.dtype && a.shape == b.shape;
}

bool is_independent(const Program& p, const StmtPtr& body, VarId v, std::string* why) {
  bool ok = true;
  rewrite_loops(body, v, [&](const Stmt& loop, const Ranges& ranges) {
    if (ok) ok = independent(p, loop, ranges, why);
    return make_for(loop.var, loop.extent, loop.body, loop.loop_kind, loop.interleave);
  });
  return ok;
}

ExprPtr substitute(const ExprPtr& e, VarId v, const ExprPtr& r) {
  if (e->kind == ExprKind::Var) return e->var == v ? r : e;
  if (e->args.empty() || !contains_var(e, v)) return e;
  std::vector<ExprPtr> args;
  for (const ExprPtr& a : e->args) args.push_back(substitute(a, v, r));
  switch (e->kind) {
    case ExprKind::Load: return load(e->buffer, e->type, std::move(args));
    case ExprKind::Binary: return binary(e->binop, args[0], args[1]);
    case ExprKind::Not: return logical_not(args[0]);
    case ExprKind::Call: return call(e->intrinsic, std::move(args));
    case ExprKind::Select: return select(args[0], args[1], args[2]);
    case ExprKind::Cast: return cast(e->type, args[0]);
    default: return e;
  }
}

std::optional<VectorPlan> plan_vector_loop(const Program& p, const Stmt& loop, bool vector_math, std::string* why) {
  auto fail = [&](const std::string& what) -> std::optional<VectorPlan> {
    if (why) *why = what;
    return std::nullopt;
  };
  const VarId v = loop.var;
  std::vector<StmtPtr> stores = stores_of(loop.body);
  if (stores.empty()) return fail("the loop body is not a list of stores");

  // An expression every lane can evaluate: f32 arithmetic on values that are
  // either the same for the whole loop or read from consecutive addresses.
  std::function<bool(const ExprPtr&, std::string&)> lane_ok = [&](const ExprPtr& e, std::string& what) {
    switch (e->kind) {
      case ExprKind::FloatImm: return true;
      case ExprKind::Load: {
        const Buffer& b = p.buffers[e->buffer];
        if (b.dtype != DType::F32) return what = "loads a non-float from " + b.name, false;
        for (size_t d = 0; d < e->args.size(); ++d) {
          if (!contains_var(e->args[d], v)) continue;
          std::optional<Affine> a = affine(e->args[d]);
          if (d + 1 != e->args.size() || !a || a->coef[v] != 1) return what = "reads " + b.name + " at addresses that are not consecutive", false;
        }
        return true;
      }
      case ExprKind::Binary:
        if (e->type != DType::F32) return what = "has a comparison", false;
        return lane_ok(e->args[0], what) && lane_ok(e->args[1], what);
      case ExprKind::Call:
        if (e->intrinsic == Intrinsic::Pow) return what = "calls pow", false;
        if (e->intrinsic != Intrinsic::Sqrt && !vector_math) return what = "calls exp or tanh and vector math is off", false;
        return lane_ok(e->args[0], what);
      default: return what = "has a select, cast, or integer value", false;
    }
  };

  std::vector<Access> accesses;
  collect(loop.body, accesses);

  VectorPlan plan;
  for (const StmtPtr& s : stores) {
    const Buffer& target = p.buffers[s->buffer];
    if (target.dtype != DType::F32) return fail("stores a non-float to " + target.name);
    bool moves = false;
    for (const ExprPtr& i : s->indices) moves = moves || contains_var(i, v);
    std::string what;

    if (moves) {
      // Elementwise. The target must advance by one element per iteration...
      for (size_t d = 0; d < s->indices.size(); ++d) {
        if (!contains_var(s->indices[d], v)) continue;
        std::optional<Affine> a = affine(s->indices[d]);
        if (d + 1 != s->indices.size() || !a || a->coef[v] != 1) return fail("writes " + target.name + " at addresses that are not consecutive");
      }
      // ...and anything else that touches the same memory must touch exactly
      // the element being written (or a row this store never reaches), so
      // that lanes cannot see each other's work.
      for (const Access& x : accesses) {
        const Buffer& b = p.buffers[x.buffer];
        if (!overlaps(b, target)) continue;
        if (!same_view(b, target) || !(equal(*x.indices, s->indices) || always_distinct(*x.indices, s->indices, v)))
          return fail("an iteration touches an element of " + target.name + " that another one writes");
      }
      if (!lane_ok(s->value, what)) return fail("the value stored to " + target.name + " " + what);
      plan.stores.push_back({false, BinOp::Add});
      continue;
    }

    // Reduction: t = t + e or t = max(t, e), where t is touched nowhere else.
    const ExprPtr& value = s->value;
    const bool shaped = value->kind == ExprKind::Binary && (value->binop == BinOp::Add || value->binop == BinOp::Max) &&
                        value->args[0]->kind == ExprKind::Load && value->args[0]->buffer == s->buffer && equal(value->args[0]->args, s->indices);
    if (!shaped) return fail("every iteration writes the same element of " + target.name + " and it is not a sum or a maximum");
    // Nothing else in the loop may read or write the accumulated element:
    // its value lives in a register until the loop ends. Other elements of
    // the same buffer are fine (a tile of accumulators is exactly that).
    for (const Access& x : accesses) {
      if (x.indices == &s->indices || x.indices == &value->args[0]->args) continue;  // this store and its own read
      const Buffer& b = p.buffers[x.buffer];
      if (!overlaps(b, target)) continue;
      if (!same_view(b, target) || !always_distinct(*x.indices, s->indices, v))
        return fail("the accumulator in " + target.name + " is also used elsewhere in the loop");
    }
    if (!lane_ok(value->args[1], what)) return fail("the value accumulated into " + target.name + " " + what);
    plan.stores.push_back({true, value->binop});
  }
  return plan;
}

// ---- Rewrites ----------------------------------------------------------------------------

StmtPtr split(const StmtPtr& body, VarId v, int64_t factor, SplitVars* vars) {
  if (factor < 1) throw ScheduleError("split: the factor must be positive");
  const VarId base = fresh_var(body);
  SplitVars out;
  StmtPtr result = rewrite_loops(body, v, [&](const Stmt& loop, const Ranges&) {
    const int64_t n = require_const_extent(loop, "split");
    require_serial(loop, "split");
    const int64_t chunks = n / factor, rest = n % factor;
    std::vector<StmtPtr> parts;
    out.inner = base + 1;
    if (chunks == 0) {  // shorter than one chunk: only rename, so the caller can address it uniformly
      parts.push_back(make_for(out.inner, n, substitute(loop.body, v, var(out.inner))));
    } else {
      StmtPtr inner = make_for(out.inner, factor, substitute(loop.body, v, add(mul(var(base), int_imm(factor)), var(out.inner))));
      if (chunks > 1) out.outer = base;
      // With one chunk the outer variable is 0 and its loop is not needed.
      parts.push_back(chunks > 1 ? make_for(base, chunks, inner) : substitute(inner, base, int_imm(0)));
      if (rest > 0) {
        out.tail = base + 2;
        parts.push_back(make_for(out.tail, rest, substitute(loop.body, v, add(int_imm(chunks * factor), var(out.tail)))));
      }
    }
    return block_of(std::move(parts));
  });
  if (vars) *vars = out;
  return result;
}

StmtPtr interchange(const Program& p, const StmtPtr& body, VarId v) {
  return rewrite_loops(body, v, [&](const Stmt& outer, const Ranges& ranges) {
    require_serial(outer, "interchange");
    const StmtPtr& inner = outer.body;
    if (inner->kind != StmtKind::For) throw ScheduleError("interchange: the loop over i" + std::to_string(v) + " does not contain exactly one loop");
    require_serial(*inner, "interchange");
    if (contains_var(inner->extent, v)) throw ScheduleError("interchange: the inner loop's extent depends on the outer variable");
    // Afterwards, an iteration (v, w) can run before (v', w') with v' < v. That
    // is harmless exactly when different values of v never depend on each other.
    require_independent(p, outer, ranges, "interchange");
    return make_for(inner->var, inner->extent, make_for(outer.var, outer.extent, inner->body));
  });
}

StmtPtr unroll_and_jam(const Program& p, const StmtPtr& body, VarId v, int64_t factor, SplitVars* vars) {
  if (factor < 1) throw ScheduleError("unroll_and_jam: the factor must be positive");
  const VarId base = fresh_var(body);
  SplitVars out;  // tail stays -1: the leftover iterations are unrolled completely

  // Replaces every store under `s` by `copies` copies, for iterations first, first+1, ...
  std::function<StmtPtr(const StmtPtr&, const ExprPtr&, int64_t)> jam = [&](const StmtPtr& s, const ExprPtr& first, int64_t copies) -> StmtPtr {
    switch (s->kind) {
      case StmtKind::For: return make_for(s->var, s->extent, jam(s->body, first, copies), s->loop_kind, s->interleave);
      case StmtKind::Block: {
        std::vector<StmtPtr> out_stmts;
        for (const StmtPtr& c : s->stmts) out_stmts.push_back(jam(c, first, copies));
        return block_of(std::move(out_stmts));
      }
      case StmtKind::Store: {
        std::vector<StmtPtr> out_stmts;
        for (int64_t r = 0; r < copies; ++r) out_stmts.push_back(substitute(s, v, add(first, int_imm(r))));
        return make_block(std::move(out_stmts));
      }
    }
    return s;
  };

  StmtPtr result = rewrite_loops(body, v, [&](const Stmt& loop, const Ranges& ranges) {
    const int64_t n = require_const_extent(loop, "unroll_and_jam");
    require_serial(loop, "unroll_and_jam");
    if (extents_use(loop.body, v)) throw ScheduleError("unroll_and_jam: an inner loop's extent depends on i" + std::to_string(v));
    // Jamming runs the copies' statements interleaved. Each copy still sees
    // its own statements in order, so the result is unchanged as long as the
    // copies do not depend on one another.
    if (factor > 1) require_independent(p, loop, ranges, "unroll_and_jam");
    const int64_t chunks = n / factor, rest = n % factor;
    std::vector<StmtPtr> parts;
    if (chunks > 1) {
      out.outer = base;
      parts.push_back(make_for(base, chunks, jam(loop.body, mul(var(base), int_imm(factor)), factor)));
    } else if (chunks == 1) {
      parts.push_back(jam(loop.body, int_imm(0), factor));
    }
    // The leftover iterations are jammed too, all of them at once. Left as a
    // loop they would run one at a time, and where the point of jamming is to
    // keep several accumulators going, one at a time is the slow case.
    if (rest > 0) parts.push_back(jam(loop.body, int_imm(chunks * factor), rest));
    return block_of(std::move(parts));
  });
  if (vars) *vars = out;
  return result;
}

StmtPtr vectorize(const Program& p, const StmtPtr& body, VarId v, int interleave, bool vector_math) {
  if (interleave < 1) throw ScheduleError("vectorize: interleave must be positive");
  return rewrite_loops(body, v, [&](const Stmt& loop, const Ranges&) {
    require_serial(loop, "vectorize");
    std::string why;
    if (!plan_vector_loop(p, loop, vector_math, &why)) throw ScheduleError("vectorize is not legal: " + why);
    return make_for(loop.var, loop.extent, loop.body, LoopKind::Vectorized, interleave);
  });
}

StmtPtr parallelize(const Program& p, const StmtPtr& body, VarId v) {
  return rewrite_loops(body, v, [&](const Stmt& loop, const Ranges& ranges) {
    require_serial(loop, "parallelize");
    require_independent(p, loop, ranges, "parallelize");
    return make_for(loop.var, loop.extent, loop.body, LoopKind::Parallel);
  });
}

// ---- The default schedule ---------------------------------------------------------------------
//
// Hand-picked choices that are right for this machine and this model. The
// autotuner (M5) replaces the constants with measured ones; the rewrites it
// applies are the same.

namespace {

bool has_loop(const StmtPtr& s) {
  if (s->kind == StmtKind::For) return true;
  for (const StmtPtr& c : s->stmts)
    if (has_loop(c)) return true;
  return false;
}

// Vectorizes every innermost loop that can be.
StmtPtr vectorize_innermost(const Program& p, const StmtPtr& s, const ScheduleOptions& o) {
  switch (s->kind) {
    case StmtKind::Store: return s;
    case StmtKind::Block: {
      std::vector<StmtPtr> out;
      for (const StmtPtr& c : s->stmts) out.push_back(vectorize_innermost(p, c, o));
      return make_block(std::move(out));
    }
    case StmtKind::For: {
      if (has_loop(s->body)) return make_for(s->var, s->extent, vectorize_innermost(p, s->body, o), s->loop_kind, s->interleave);
      const int64_t n = const_extent(*s);
      if (s->loop_kind != LoopKind::Serial || (n >= 0 && n < 4)) return s;
      std::optional<VectorPlan> plan = plan_vector_loop(p, *s, o.vmath);
      if (!plan) return s;
      // One accumulator is a chain: each addition waits for the one before it
      // (three cycles on this core, which can start four per cycle). A lone
      // sum therefore gets several independent accumulators. A loop that
      // already carries many, like a register-tiled matrix product, does not
      // need more.
      size_t reductions = 0;
      for (const VectorPlan::Item& item : plan->stores) reductions += item.reduction;
      const int interleave = reductions > 0 && reductions <= 2 ? 4 : 1;
      return make_for(s->var, s->extent, s->body, LoopKind::Vectorized, interleave);
    }
  }
  return s;
}

// A rough count of the work under `s`: stores executed, with a store that
// calls exp or tanh counted as twenty.
double work(const StmtPtr& s, Ranges ranges) {
  switch (s->kind) {
    case StmtKind::Store: {
      std::function<int(const ExprPtr&)> calls = [&](const ExprPtr& e) {
        int n = e->kind == ExprKind::Call;
        for (const ExprPtr& a : e->args) n += calls(a);
        return n;
      };
      return 1.0 + 20.0 * calls(s->value);
    }
    case StmtKind::Block: {
      double total = 0;
      for (const StmtPtr& c : s->stmts) total += work(c, ranges);
      return total;
    }
    case StmtKind::For:
      add_range(*s, ranges);
      return static_cast<double>(ranges[s->var] + 1) * work(s->body, ranges);
  }
  return 0;
}

// Runs each outermost loop of the kernel on threads, if it is big enough to
// repay waking them and its iterations are independent.
StmtPtr parallelize_outermost(const Program& p, const StmtPtr& s, const ScheduleOptions& o) {
  if (s->kind == StmtKind::Block) {
    std::vector<StmtPtr> out;
    for (const StmtPtr& c : s->stmts) out.push_back(parallelize_outermost(p, c, o));
    return make_block(std::move(out));
  }
  if (s->kind != StmtKind::For || s->loop_kind != LoopKind::Serial || const_extent(*s) < 2) return s;
  if (work(s, {}) < static_cast<double>(o.parallel_work) || !independent(p, *s, {}, nullptr)) return s;
  return make_for(s->var, s->extent, s->body, LoopKind::Parallel);
}

int64_t extent_of(const StmtPtr& s, VarId v) {
  if (s->kind == StmtKind::For) return s->var == v ? const_extent(*s) : extent_of(s->body, v);
  for (const StmtPtr& c : s->stmts) {
    int64_t n = extent_of(c, v);
    if (n >= 0) return n;
  }
  return -1;
}

// The matrix product  C[i,j] = init;  C[i,j] += A[i,p] * Bt[j,p]  (both
// operands are read along their rows). See docs/log.md, M4, for the reasoning
// behind each step.
StmtPtr apply_matmul(const Program& p, const Kernel& k, const MatmulParams& q) {
  StmtPtr body = k.body;

  // 1. Put one operand outside. Its rows are streamed through once, and the
  //    other operand is the one that is re-read.
  if (q.cols_outside) body = interchange(p, body, k.mm_i);
  const VarId big = q.cols_outside ? k.mm_j : k.mm_i, small = q.cols_outside ? k.mm_i : k.mm_j;
  const int64_t big_tile = q.cols_outside ? q.tile_cols : q.tile_rows, small_tile = q.cols_outside ? q.tile_rows : q.tile_cols;

  // 2. Cache tiling. Cut the outside operand into blocks of rows that fit in
  //    L1, and move the loop over the other operand's rows outside the loop
  //    over the rows of a block:  for block: for small: for row-in-block.
  //    Every row of the small operand then meets a block that is already in L1.
  std::vector<VarId> block_loops = {big};
  if (q.block_rows > 0) {
    SplitVars s;
    body = split(body, big, q.block_rows, &s);
    block_loops = {s.inner};
    if (s.tail >= 0) block_loops.push_back(s.tail);
    for (VarId v : block_loops) body = interchange(p, body, v);
  }

  // 3. Register tiling. Unroll both row loops into the dot-product loop, so
  //    one pass over p computes tile_rows * tile_cols elements of C at once,
  //    each in its own accumulator.
  body = unroll_and_jam(p, body, small, small_tile);
  for (VarId v : block_loops) body = unroll_and_jam(p, body, v, big_tile);
  return body;
}

}  // namespace

bool is_matmul(const Kernel& k) { return k.mm_i >= 0 && k.mm_j >= 0 && k.mm_p >= 0; }

MatmulShape matmul_shape(const Kernel& k) {
  if (!is_matmul(k)) throw ScheduleError("kernel " + k.label + " is not a matrix product in dot form");
  return {extent_of(k.body, k.mm_i), extent_of(k.body, k.mm_p), extent_of(k.body, k.mm_j)};
}

// The hand-picked schedule: what M4 arrived at by measurement on one machine.
MatmulParams default_matmul_params(const MatmulShape& shape, const ScheduleOptions& o) {
  const int64_t m = shape.rows, n = shape.cols, depth = shape.inner;
  MatmulParams q;

  // The register tile. Rows left over when the tile does not divide the
  // matrix get a smaller tile with fewer accumulators, which is slower, so
  // among the shapes on offer take the one that leaves the least over.
  q.tile_rows = o.tile_rows;
  q.tile_cols = o.tile_cols;
  if (o.choose_tile) {
    double best = 2;
    for (auto [r, c] : {std::pair{o.tile_rows, o.tile_cols}, {o.tile_cols, o.tile_rows}, {4, 4}}) {
      double leftover = static_cast<double>(m % r) / static_cast<double>(m) + static_cast<double>(n % c) / static_cast<double>(n);
      if (leftover < best - 0.005) {  // a clear improvement only: more accumulators is otherwise better
        best = leftover;
        q.tile_rows = r;
        q.tile_cols = c;
      }
    }
  }
  if (!o.regtile) q.tile_rows = q.tile_cols = 1;

  // The operand with more rows goes outside.
  q.cols_outside = n > m;
  const int64_t big_tile = q.cols_outside ? q.tile_cols : q.tile_rows, small_tile = q.cols_outside ? q.tile_rows : q.tile_cols;

  // The block. L1 also has to hold what the block is multiplied against: all
  // of the small operand if that is small enough to stay there, otherwise the
  // one tile of it in use.
  if (o.cachetile) {
    const int64_t row_bytes = 4 * depth, small_bytes = (q.cols_outside ? m : n) * row_bytes;
    const int64_t company = small_bytes <= o.block_bytes ? small_bytes : small_tile * row_bytes;
    const int64_t budget = std::max(o.block_bytes / 4, o.block_bytes + o.block_bytes / 8 - company);
    q.block_rows = std::max(std::max<int64_t>(1, budget / row_bytes) / big_tile * big_tile, big_tile);
  }
  return q;
}

void schedule_kernel(const Program& p, Kernel& k, const ScheduleOptions& o, const MatmulParams* params) {
  if (is_matmul(k) && (params || o.regtile || o.cachetile)) {
    MatmulParams q;
    auto tuned = o.tuned.find(k.signature);
    if (params) q = *params;
    else if (tuned != o.tuned.end()) q = tuned->second;
    else q = default_matmul_params(matmul_shape(k), o);
    if (q.tile_rows < 1 || q.tile_cols < 1 || q.block_rows < 0) throw ScheduleError("matrix product parameters out of range");
    k.body = apply_matmul(p, k, q);
  }
  if (o.vectorize) k.body = vectorize_innermost(p, k.body, o);
  if (o.parallel) k.body = parallelize_outermost(p, k.body, o);
}

Program schedule_program(Program p, const ScheduleOptions& o) {
  for (std::vector<Kernel>* list : {&p.init_kernels, &p.kernels}) {
    for (Kernel& k : *list) {
      try {
        schedule_kernel(p, k, o);
      } catch (const ScheduleError& e) {
        throw ScheduleError("scheduling kernel " + k.label + ": " + e.what());
      }
    }
  }
  check_program(p);
  return p;
}

}  // namespace mlc::loop
