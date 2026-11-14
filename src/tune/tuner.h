// The autotuner: chooses schedule parameters by running candidates and timing
// them, instead of by reasoning about the machine.
//
// What is tuned is the schedule of each distinct matrix product (the register
// tile, which operand is blocked, the block size). Everything else in a
// schedule is either forced by legality or too cheap to matter.
//
// How:
//   1. Candidates are points in a small discrete space (Space, below).
//   2. A batch of candidates for one product is compiled into one shared
//      library: the same product several times over, each copy scheduled
//      differently. One compiler invocation, and all candidates are then timed
//      in the same process, in turn, pass after pass, so drift in the machine's
//      speed hits them equally.
//   3. Search is random sampling, then a few generations of mutating the best.
//   4. The best few and the hand-picked default are measured again, for
//      longer. A candidate replaces the default only if it wins that round by
//      more than a noise margin.
//   5. Results go into a JSON cache keyed by machine, thread count, and kernel
//      signature. Compiling with the cache reproduces the tuned build without
//      measuring anything.
#pragma once

#include <cstdint>
#include <functional>
#include <map>
#include <random>
#include <string>
#include <vector>

#include "graph/ir.h"
#include "loop/schedule.h"

namespace mlc::tune {

struct TuneError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- The machine ----------------------------------------------------------------

struct Machine {
  std::string cpu;            // e.g. "Apple M2"
  int64_t l1d_bytes = 32768;  // L1 data cache of a performance core
  int performance_cores = 1;
  std::string power;          // "AC", "battery", "AC, low power mode", ...; informational
};

// Asks the operating system (sysctl on macOS, sysconf elsewhere).
Machine detect_machine();

// ---- The cache -------------------------------------------------------------------

struct Entry {
  loop::MatmulParams params;
  double gflops = 0;          // measured rate of `params`
  double default_gflops = 0;  // measured rate of the default schedule, in the same round
};

struct Cache {
  Machine machine;
  // (threads, kernel signature) -> the schedule to use.
  std::map<std::pair<int, std::string>, Entry> products;
  // threads -> the size below which a kernel stays on one thread.
  std::map<int, int64_t> parallel_work;
};

// load_cache returns an empty cache for a file that does not exist.
Cache load_cache(const std::string& path);
void save_cache(const std::string& path, const Cache& cache);

// Copies the cache's choices for `threads` into `options`. Returns how many
// matrix-product schedules were installed. Installs nothing, and returns 0,
// if the cache was measured on a different CPU than `machine`.
size_t apply_cache(const Cache& cache, const Machine& machine, int threads, loop::ScheduleOptions& options);

// ---- The search space --------------------------------------------------------------

// A candidate, as coordinates. Neighbouring coordinates are similar schedules,
// which is what makes mutation a sensible move.
struct Point {
  int tile = 0;    // index into Space::tiles
  int block = 0;   // index into Space::block_fractions; -1 means no cache tiling
  bool cols_outside = false;
  bool operator==(const Point&) const = default;
  auto operator<=>(const Point&) const = default;
};

struct Space {
  loop::MatmulShape shape;
  int64_t l1d_bytes;
  // Register tiles that fit the 32 vector registers (rows * cols accumulators
  // plus rows + cols loaded operands) and carry at least 12 accumulators,
  // enough to keep the FMA units busy.
  std::vector<std::pair<int, int>> tiles;
  // Block sizes, as multiples of the L1 cache.
  std::vector<double> block_fractions = {1.0 / 16, 1.0 / 8, 3.0 / 16, 1.0 / 4, 3.0 / 8, 1.0 / 2, 3.0 / 4, 1.0, 2.0, 4.0};

  Space(const loop::MatmulShape& shape, int64_t l1d_bytes);

  loop::MatmulParams params(const Point& p) const;
  Point random(std::mt19937& rng) const;
  // A point that differs from `p` in one coordinate, by a small step.
  Point mutate(const Point& p, std::mt19937& rng) const;
  size_t size() const { return tiles.size() * (block_fractions.size() + 1) * 2; }
};

// ---- Measuring and searching ---------------------------------------------------------

// One matrix product of the model: the op, its attributes, and operand types.
struct Product {
  Op op = Op::MatMul;
  Attrs attrs;
  std::vector<Type> inputs;
  std::string signature;  // as the lowering names it
  double flops = 0;       // 2 * batch * rows * inner * cols
};

// The distinct dot-form matrix products of a graph, in order of first use.
std::vector<Product> products_of(const Graph& g);

struct TuneOptions {
  int threads = 1;
  int random = 16;        // random candidates in the first generation
  int generations = 2;    // rounds of mutation after it
  int children = 8;       // mutated candidates per round
  int finalists = 3;      // re-measured against the default at the end
  double seconds = 0.3;   // measuring time per candidate per round (at least three passes regardless)
  double margin = 0.02;   // a finalist must beat the default by this fraction (run-to-run noise is about 1%)
  uint32_t seed = 1;
  std::string work_dir;   // where candidate libraries are built
  std::function<void(const std::string&)> log;  // progress lines; may be empty
};

// Seconds per call for each candidate schedule of `product`: the minimum over
// several passes, all candidates timed in turn within each pass.
std::vector<double> measure(const Product& product, const std::vector<loop::MatmulParams>& candidates, const TuneOptions& options);

// Searches for the best schedule of one product.
Entry tune_product(const Product& product, const Machine& machine, const TuneOptions& options);

}  // namespace mlc::tune
