// Loop transformations: the rewrites that turn a naive loop nest into a fast
// one, and the analysis that says when each rewrite is allowed.
//
// Two questions are kept apart throughout.
//
//   Legality:      does the rewritten nest compute the same values? Answered by
//                  dependence analysis on loads and stores, below. A rewrite
//                  that cannot prove it throws ScheduleError and changes nothing.
//   Profitability: is it faster? Never answered here. The caller (the default
//                  schedule at the bottom, later the autotuner) decides what to
//                  apply and with which parameters, and measures.
//
// A rewrite names the loop it acts on by its variable. Rewrites that copy a
// nest (a main part and a remainder) leave several loops binding the same
// variable in sibling positions; a variable then names all of them, and the
// next rewrite applies to each.
#pragma once

#include <map>
#include <optional>
#include <string>

#include "loop/ir.h"

namespace mlc::loop {

struct ScheduleError : LoopError {
  using LoopError::LoopError;
};

// ---- Analysis -----------------------------------------------------------------

// True when two buffers can name the same byte, and when they are the same
// bytes under the same layout (what an op running in place has: its output
// buffer is its input buffer).
bool overlaps(const Buffer& a, const Buffer& b);
bool same_view(const Buffer& a, const Buffer& b);

// The dependence test. A loop's iterations are independent when no iteration
// reads or writes an element that another iteration writes. Then they may run
// in any order, interleaved, or at the same time.
//
// The proof it looks for: for every buffer the loop body writes, there is one
// axis along which every access to that buffer (loads and stores alike) is at
// position  c * v + r, where v is the loop variable, c >= 1 is the same
// constant in all of them, and r varies over a range narrower than c. Each
// iteration then owns its own slice of the buffer along that axis. Anything it
// cannot prove counts as dependent.
//
// Checks every loop in `body` that binds `v`. On failure `why` says what was
// in the way.
bool is_independent(const Program& p, const StmtPtr& body, VarId v, std::string* why = nullptr);

// What a vectorized loop does with each store in its body, in order.
struct VectorPlan {
  struct Item {
    // Elementwise: iteration v writes element v of a contiguous run, so W
    // iterations become one vector store.
    // Reduction: every iteration combines into the same element,
    // t = t + e or t = max(t, e); the vector loop accumulates in a register.
    bool reduction = false;
    BinOp op = BinOp::Add;
  };
  std::vector<Item> stores;
};

// Decides whether `loop` can be run in vector lanes and how. `vector_math`
// says whether vector exp and tanh are available. Returns nothing, and a
// reason in `why`, if it cannot.
std::optional<VectorPlan> plan_vector_loop(const Program& p, const Stmt& loop, bool vector_math = true, std::string* why = nullptr);

// `e` with every use of variable `v` replaced by `replacement`.
ExprPtr substitute(const ExprPtr& e, VarId v, const ExprPtr& replacement);

// ---- Rewrites -----------------------------------------------------------------

struct SplitVars {
  VarId outer = -1;  // counts whole chunks; -1 if there is at most one
  VarId inner = -1;  // position inside a chunk
  VarId tail = -1;   // the leftover iterations after the last whole chunk; -1 if none
};

// for v in 0..N  ->  for outer in 0..N/f: for inner in 0..f: (v = outer*f + inner)
//                    for tail in 0..N%f:                      (v = (N/f)*f + tail)
// Iterations run in the same order as before, so this is always legal. On its
// own it changes nothing; it creates the loops that interchange moves.
StmtPtr split(const StmtPtr& body, VarId v, int64_t factor, SplitVars* vars = nullptr);

// for v: for w: S  ->  for w: for v: S.   Needs v's iterations independent.
StmtPtr interchange(const Program& p, const StmtPtr& body, VarId v);

// Unrolls `v` by `factor` and fuses the copies all the way down: every store
// under the loop is replaced by `factor` stores, for v, v+1, ..., and the
// loops in between are shared. Iterations left over when N is not a multiple
// of the factor are unrolled and fused the same way, all of them together.
// Needs v's iterations independent. vars->outer names the unrolled loop.
StmtPtr unroll_and_jam(const Program& p, const StmtPtr& body, VarId v, int64_t factor, SplitVars* vars = nullptr);

// Marks every loop binding `v` to run in vector lanes. See VectorPlan.
StmtPtr vectorize(const Program& p, const StmtPtr& body, VarId v, int interleave = 1, bool vector_math = true);

// Marks every loop binding `v` to run on several threads. Needs independence.
StmtPtr parallelize(const Program& p, const StmtPtr& body, VarId v);

// ---- The default schedule -------------------------------------------------------

// The schedule of one matrix product in dot form: everything the autotuner
// varies. A schedule is data; applying it is always the same rewrites.
struct MatmulParams {
  int tile_rows = 4, tile_cols = 5;  // the register tile; rows * cols accumulators. 1x1 means no register tiling
  bool cols_outside = false;         // which operand is blocked and streamed once: Bt's rows (true) or A's (false)
  int64_t block_rows = 0;            // rows of that operand per cache block; 0 means no cache tiling
  bool operator==(const MatmulParams&) const = default;
};

struct MatmulShape {
  int64_t rows, inner, cols;  // C[rows, cols] = sum over `inner`
};

struct ScheduleOptions {
  bool vectorize = true;   // run innermost loops in vector lanes where legal
  bool vmath = true;       // vector exp and tanh; without them loops that call those stay scalar
  bool regtile = true;     // matrix product: unroll-and-jam rows and columns into a tile of register accumulators
  bool cachetile = true;   // matrix product: block the larger operand so a block stays in the L1 cache
  bool parallel = true;    // run the outermost loop of a large kernel on several threads

  int tile_rows = 4, tile_cols = 5;    // the register tile: accumulators = rows * cols
  bool choose_tile = true;             // may swap the tile, or use 4x4, where that leaves fewer leftover rows
  int64_t block_bytes = 64 << 10;      // how much of L1 one cache block and its company may occupy
  int64_t parallel_work = 1 << 16;     // kernels estimated smaller than this stay on one thread

  // Measured schedules, by kernel signature. A matrix product found here uses
  // these parameters instead of the default ones.
  std::map<std::string, MatmulParams> tuned;
};

// The shape of a dot-form matrix product kernel, and the parameters the
// default schedule gives a product of that shape.
MatmulShape matmul_shape(const Kernel& k);
MatmulParams default_matmul_params(const MatmulShape& shape, const ScheduleOptions& options = {});

// Schedules one kernel in place. `params`, if given, overrides both the tuned
// table and the defaults for a matrix product.
void schedule_kernel(const Program& p, Kernel& k, const ScheduleOptions& options, const MatmulParams* params = nullptr);

// Applies the default schedule to every kernel and verifies the result.
Program schedule_program(Program p, const ScheduleOptions& options = {});

}  // namespace mlc::loop
