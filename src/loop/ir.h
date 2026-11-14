// The loop IR: how the model is computed, as explicit loops over memory.
//
// The graph IR says "matmul(a, b)". The loop IR says "for i, for p, for j:
// C[i,j] = C[i,j] + A[i,p] * B[p,j]", and says where A, B, and C live. It is
// the level at which loop order, tiling, vectorization, and threading can be
// expressed, because all of those are statements about loops, and the graph
// has none.
//
// The whole language is three statements and a handful of expressions:
//
//   For    a counted loop, `for var in [0, extent)`, with a constant extent
//   Store  buffer[indices] = value
//   Block  statements in sequence
//
// There are no scalar variables other than loop counters. A reduction
// accumulates through memory (load, combine, store), and an op that needs a
// temporary gets a small buffer for it. That keeps the IR tiny and makes
// every dependence visible as a load or a store; turning those into registers
// is an optimization for later milestones.
#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "graph/ir.h"

namespace mlc::loop {

struct LoopError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- Buffers -----------------------------------------------------------------

using BufferId = int32_t;
using VarId = int32_t;

// Where a buffer's bytes live at run time.
enum class Storage : uint8_t {
  Arena,    // scratch memory owned by the runtime, at byte `offset`
  Weights,  // the memory-mapped weights file, at byte `offset`; read-only
  Input,    // caller's input pointer number `index`; read-only
  Output,   // caller's output pointer number `index`
  Inline,   // constant bytes carried in the program, `index` into inline_data; read-only
};

// A named, typed, shaped view of some storage. Indexing is multi-dimensional
// and row-major. Two buffers may view the same storage with different shapes;
// that is how a graph Reshape lowers to no code at all.
struct Buffer {
  std::string name;
  DType dtype = DType::F32;
  Shape shape;
  Storage storage = Storage::Arena;
  int64_t offset = 0;
  int32_t index = -1;
  bool writable() const { return storage == Storage::Arena || storage == Storage::Output; }
};

// ---- Expressions ---------------------------------------------------------------

enum class ExprKind : uint8_t { IntImm, FloatImm, Var, Load, Binary, Not, Call, Select, Cast };
enum class BinOp : uint8_t { Add, Sub, Mul, Div, Max, Eq, Ne, Le, And, Or };
enum class Intrinsic : uint8_t { Exp, Tanh, Pow, Sqrt };

struct Expr;
using ExprPtr = std::shared_ptr<const Expr>;  // immutable and shared: a tree may be reused freely

struct Expr {
  ExprKind kind = ExprKind::IntImm;
  DType type = DType::I64;                 // every expression has a scalar type: f32, i64, or bool
  int64_t int_value = 0;      // IntImm (also bool: 0 or 1)
  double float_value = 0;     // FloatImm (holds an f32 value)
  VarId var = -1;             // Var
  BufferId buffer = -1;       // Load
  BinOp binop = BinOp::Add;   // Binary
  Intrinsic intrinsic = Intrinsic::Exp;  // Call
  // Load: one index per buffer axis. Binary: a, b. Not, Cast: a. Call: the
  // arguments. Select: condition, value if true, value if false.
  std::vector<ExprPtr> args;
};

// Constructors. Each checks its operand types and throws LoopError on misuse.
// add and mul fold integer constants (x + 0, x * 1, 2 * 3), which keeps index
// arithmetic readable; nothing else is simplified.
ExprPtr int_imm(int64_t v);
ExprPtr bool_imm(bool v);
ExprPtr float_imm(double v);
ExprPtr var(VarId id);
ExprPtr load(BufferId buffer, DType dtype, std::vector<ExprPtr> indices);
ExprPtr binary(BinOp op, ExprPtr a, ExprPtr b);
ExprPtr logical_not(ExprPtr a);
ExprPtr call(Intrinsic fn, std::vector<ExprPtr> args);
ExprPtr select(ExprPtr cond, ExprPtr if_true, ExprPtr if_false);
ExprPtr cast(DType to, ExprPtr a);

inline ExprPtr add(ExprPtr a, ExprPtr b) { return binary(BinOp::Add, std::move(a), std::move(b)); }
inline ExprPtr sub(ExprPtr a, ExprPtr b) { return binary(BinOp::Sub, std::move(a), std::move(b)); }
inline ExprPtr mul(ExprPtr a, ExprPtr b) { return binary(BinOp::Mul, std::move(a), std::move(b)); }
inline ExprPtr div(ExprPtr a, ExprPtr b) { return binary(BinOp::Div, std::move(a), std::move(b)); }

// ---- Statements ----------------------------------------------------------------

enum class StmtKind : uint8_t { For, Store, Block };

// How a loop's iterations are to be executed. Everything is Serial until M4.
enum class LoopKind : uint8_t { Serial, Parallel, Vectorized };

struct Stmt;
using StmtPtr = std::shared_ptr<const Stmt>;

struct Stmt {
  StmtKind kind = StmtKind::Block;
  // For
  VarId var = -1;
  int64_t extent = 0;
  LoopKind loop_kind = LoopKind::Serial;
  StmtPtr body;
  // Store
  BufferId buffer = -1;
  std::vector<ExprPtr> indices;
  ExprPtr value;
  // Block
  std::vector<StmtPtr> stmts;
};

StmtPtr make_for(VarId var, int64_t extent, StmtPtr body);
StmtPtr make_store(BufferId buffer, std::vector<ExprPtr> indices, ExprPtr value);
StmtPtr make_block(std::vector<StmtPtr> stmts);

// ---- Program -------------------------------------------------------------------

// The loops for one graph node. Kernels run in order; each becomes one
// function in the generated code.
struct Kernel {
  std::string op;        // graph op name, e.g. "matmul"
  std::string label;     // e.g. "%75 bmm", for listings and profiles
  StmtPtr body;
};

struct Program {
  std::vector<Buffer> buffers;
  std::vector<std::vector<std::byte>> inline_data;
  std::vector<Kernel> kernels;
  int64_t arena_bytes = 0;            // size of the arena the program needs
  int64_t unplanned_arena_bytes = 0;  // what it would need if no buffer ever shared memory with another
  int64_t weights_bytes = 0;
  std::vector<Type> inputs;   // what the caller must pass, in order
  std::vector<Type> outputs;
};

// The single row-major offset for a multi-dimensional access:
// ((i0 * n1 + i1) * n2 + i2) ... Used by the interpreter and by codegen.
ExprPtr flat_index(const Buffer& buffer, const std::vector<ExprPtr>& indices);

// Pseudo-code listing of the whole program, or of one kernel.
std::string print_program(const Program& p);
std::string print_kernel(const Program& p, const Kernel& k);
std::string print_expr(const Program& p, const ExprPtr& e);

// Structural checks: every index count matches its buffer's rank, indices are
// i64, stored values have the buffer's dtype, stores go to writable storage,
// loop variables are in scope where used and not rebound, arena buffers fit
// in the arena. Returns one message per problem.
std::vector<std::string> verify_program(const Program& p);
void check_program(const Program& p);

// ---- Lowering and execution ---------------------------------------------------------

struct LowerOptions {
  // Memory planning: reuse arena space once a value is dead, and let an
  // elementwise op write over an operand it is the last reader of. When off,
  // every value gets arena space of its own, as in M2.
  bool plan_memory = true;
};

// Lowers every graph node to a naive loop nest, one kernel per node, and
// decides where every value lives.
Program lower_graph(const Graph& g, int64_t weights_bytes, const LowerOptions& options = {});

// Tree-walking interpreter. Slow, and checks every index against its
// buffer's bounds. `inputs` and `outputs` are raw pointers to caller memory
// laid out as Program::inputs / Program::outputs describe.
void run_program(const Program& p, std::span<const std::byte> weights, std::span<const void* const> inputs,
                 std::span<void* const> outputs);

}  // namespace mlc::loop
