#include "tune/tuner.h"

#include <unistd.h>

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>

#include "codegen/codegen.h"
#include "support/json.h"

namespace mlc::tune {

using loop::MatmulParams;

// ---- The machine ------------------------------------------------------------------

namespace {

std::string command_output(const char* command) {
  std::string out;
  if (FILE* pipe = ::popen(command, "r")) {
    char buf[256];
    while (std::fgets(buf, sizeof buf, pipe)) out += buf;
    ::pclose(pipe);
  }
  return out;
}

}  // namespace

Machine detect_machine() {
  Machine m;
#ifdef __APPLE__
  char brand[128] = {};
  size_t size = sizeof brand;
  if (sysctlbyname("machdep.cpu.brand_string", brand, &size, nullptr, 0) == 0) m.cpu = brand;
  int64_t l1 = 0;
  size = sizeof l1;
  // perflevel0 is the performance cores; the efficiency cores have a smaller L1.
  if (sysctlbyname("hw.perflevel0.l1dcachesize", &l1, &size, nullptr, 0) == 0 && l1 > 0) m.l1d_bytes = l1;
  int cores = 0;
  size = sizeof cores;
  if (sysctlbyname("hw.perflevel0.physicalcpu", &cores, &size, nullptr, 0) == 0 && cores > 0) m.performance_cores = cores;
  std::string battery = command_output("pmset -g batt 2>/dev/null"), settings = command_output("pmset -g 2>/dev/null");
  m.power = battery.find("AC Power") != std::string::npos ? "AC" : battery.find("Battery Power") != std::string::npos ? "battery" : "unknown";
  if (settings.find("lowpowermode         1") != std::string::npos) m.power += ", low power mode";
#else
  m.cpu = "unknown";
  long l1 = ::sysconf(_SC_LEVEL1_DCACHE_SIZE);
  if (l1 > 0) m.l1d_bytes = l1;
  long cores = ::sysconf(_SC_NPROCESSORS_ONLN);
  if (cores > 0) m.performance_cores = static_cast<int>(cores);
  m.power = "unknown";
#endif
  if (m.cpu.empty()) m.cpu = "unknown";
  return m;
}

// ---- The cache ----------------------------------------------------------------------

namespace {

std::string quoted(const std::string& s) {
  std::string out = "\"";
  for (char c : s) {
    if (c == '"' || c == '\\') out += '\\';
    out += c;
  }
  return out + "\"";
}

}  // namespace

Cache load_cache(const std::string& path) {
  Cache cache;
  std::ifstream in(path);
  if (!in) return cache;
  std::ostringstream text;
  text << in.rdbuf();
  try {
    json::Value root = json::parse(text.str());
    cache.machine.cpu = root.at("cpu").as_string();
    cache.machine.l1d_bytes = root.at("l1d_bytes").as_int();
    cache.machine.performance_cores = static_cast<int>(root.at("performance_cores").as_int());
    cache.machine.power = root.at("power").as_string();
    for (const json::Member& m : root.at("parallel_work").as_object()) cache.parallel_work[std::stoi(m.key)] = m.value.as_int();
    for (const json::Value& p : root.at("products").as_array()) {
      Entry e;
      e.params.tile_rows = static_cast<int>(p.at("tile_rows").as_int());
      e.params.tile_cols = static_cast<int>(p.at("tile_cols").as_int());
      e.params.cols_outside = p.at("cols_outside").as_bool();
      e.params.block_rows = p.at("block_rows").as_int();
      e.gflops = p.at("gflops").as_double();
      e.default_gflops = p.at("default_gflops").as_double();
      cache.products[{static_cast<int>(p.at("threads").as_int()), p.at("signature").as_string()}] = e;
    }
  } catch (const std::exception& e) {
    throw TuneError(path + " is not a tuning cache: " + e.what());
  }
  return cache;
}

void save_cache(const std::string& path, const Cache& cache) {
  std::ofstream out(path);
  if (!out) throw TuneError("cannot write " + path);
  out << "{\n  \"cpu\": " << quoted(cache.machine.cpu) << ",\n  \"l1d_bytes\": " << cache.machine.l1d_bytes
      << ",\n  \"performance_cores\": " << cache.machine.performance_cores << ",\n  \"power\": " << quoted(cache.machine.power)
      << ",\n  \"parallel_work\": {";
  bool first = true;
  for (auto& [threads, work] : cache.parallel_work) {
    out << (first ? "" : ", ") << "\"" << threads << "\": " << work;
    first = false;
  }
  out << "},\n  \"products\": [\n";
  first = true;
  char rate[64];
  for (auto& [key, e] : cache.products) {
    std::snprintf(rate, sizeof rate, "\"gflops\": %.1f, \"default_gflops\": %.1f", e.gflops, e.default_gflops);
    out << (first ? "" : ",\n") << "    {\"threads\": " << key.first << ", \"signature\": " << quoted(key.second) << ", \"tile_rows\": "
        << e.params.tile_rows << ", \"tile_cols\": " << e.params.tile_cols << ", \"cols_outside\": " << (e.params.cols_outside ? "true" : "false")
        << ", \"block_rows\": " << e.params.block_rows << ", " << rate << "}";
    first = false;
  }
  out << "\n  ]\n}\n";
}

size_t apply_cache(const Cache& cache, const Machine& machine, int threads, loop::ScheduleOptions& options) {
  // A schedule is only as good as the measurement behind it, and the
  // measurement was of one CPU.
  if (cache.machine.cpu != machine.cpu) return 0;
  size_t installed = 0;
  for (auto& [key, e] : cache.products) {
    if (key.first != threads) continue;
    options.tuned[key.second] = e.params;
    ++installed;
  }
  auto work = cache.parallel_work.find(threads);
  if (work != cache.parallel_work.end()) options.parallel_work = work->second;
  return installed;
}

// ---- The search space ------------------------------------------------------------------

Space::Space(const loop::MatmulShape& s, int64_t l1) : shape(s), l1d_bytes(l1) {
  for (int r = 1; r <= 10; ++r)
    for (int c = 1; c <= 10; ++c)
      if (r * c >= 12 && r * c + r + c <= 32 && r <= shape.rows && c <= shape.cols) tiles.push_back({r, c});
  // A matrix too small for any of those still needs a tile.
  if (tiles.empty()) tiles.push_back({static_cast<int>(std::min<int64_t>(shape.rows, 4)), static_cast<int>(std::min<int64_t>(shape.cols, 4))});
}

MatmulParams Space::params(const Point& p) const {
  MatmulParams q;
  q.tile_rows = tiles.at(static_cast<size_t>(p.tile)).first;
  q.tile_cols = tiles.at(static_cast<size_t>(p.tile)).second;
  q.cols_outside = p.cols_outside;
  if (p.block >= 0) {
    // Whole tiles of the blocked operand, as many as fit in the fraction of L1.
    const int64_t tile = p.cols_outside ? q.tile_cols : q.tile_rows;
    const double bytes = block_fractions.at(static_cast<size_t>(p.block)) * static_cast<double>(l1d_bytes);
    const int64_t rows = static_cast<int64_t>(bytes / (4.0 * static_cast<double>(shape.inner)));
    q.block_rows = std::max(rows / tile * tile, tile);
  }
  return q;
}

Point Space::random(std::mt19937& rng) const {
  Point p;
  p.tile = static_cast<int>(rng() % tiles.size());
  p.block = static_cast<int>(rng() % (block_fractions.size() + 1)) - 1;
  p.cols_outside = rng() % 2;
  return p;
}

Point Space::mutate(const Point& p, std::mt19937& rng) const {
  Point q = p;
  switch (rng() % 3) {
    case 0: {
      // A tile within two steps: one more row and one fewer column, say.
      std::vector<int> near;
      for (size_t t = 0; t < tiles.size(); ++t) {
        int distance = std::abs(tiles[t].first - tiles[static_cast<size_t>(p.tile)].first) + std::abs(tiles[t].second - tiles[static_cast<size_t>(p.tile)].second);
        if (distance >= 1 && distance <= 2) near.push_back(static_cast<int>(t));
      }
      if (!near.empty()) q.tile = near[rng() % near.size()];
      break;
    }
    case 1: {
      // The next block size up or down. "No blocking" sits below the smallest.
      const int last = static_cast<int>(block_fractions.size()) - 1;
      q.block = std::clamp(p.block + (rng() % 2 ? 1 : -1), -1, last);
      break;
    }
    default: q.cols_outside = !p.cols_outside;
  }
  return q;
}

// ---- Measuring ----------------------------------------------------------------------------

std::vector<Product> products_of(const Graph& g) {
  std::vector<Product> out;
  for (const Node& n : g.nodes) {
    if ((n.op != Op::MatMul && n.op != Op::Gemm) || !n.attrs.trans_b) continue;
    Product p;
    p.op = n.op;
    p.attrs = n.attrs;
    for (ValueId in : n.inputs) p.inputs.push_back(g.type_of(in));
    if (std::any_of(out.begin(), out.end(), [&](const Product& q) { return q.op == p.op && q.inputs == p.inputs; })) continue;
    p.signature = std::string(op_name(n.op)) + " " + shape_string(p.inputs[0].shape) + " x " + shape_string(p.inputs[1].shape) + "^T";
    p.flops = 2.0 * static_cast<double>(num_elements(n.type.shape)) * static_cast<double>(p.inputs[0].shape.back());
    out.push_back(std::move(p));
  }
  return out;
}

namespace {

double seconds_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

// Builds one library holding `candidates.size()` copies of the product, each
// under its own schedule, and times them.
std::vector<double> measure_batch(const Product& product, const std::vector<MatmulParams>& candidates, const TuneOptions& o, int batch) {
  // The copies share their operands. Their results are never read, so the
  // memory planner gives every copy the same piece of the arena: the copies
  // differ in schedule and in nothing else. One small real output keeps the
  // program well-formed.
  Graph g;
  std::vector<ValueId> inputs;
  for (size_t i = 0; i < product.inputs.size(); ++i) inputs.push_back(g.add_input(product.inputs[i], "operand" + std::to_string(i)));
  for (size_t c = 0; c < candidates.size(); ++c) g.add(product.op, inputs, product.attrs);
  g.outputs = {g.add(Op::Tanh, {g.add_input({DType::F32, {1}}, "unused")})};

  loop::Program program = loop::lower_graph(g, 0);
  if (program.kernels.size() != candidates.size() + 1) throw TuneError("expected one kernel per candidate");
  loop::ScheduleOptions schedule;
  schedule.parallel = o.threads > 1;
  for (size_t c = 0; c < candidates.size(); ++c) loop::schedule_kernel(program, program.kernels[c], schedule, &candidates[c]);
  loop::check_program(program);

  // A fresh path for every library. The runtime has thread-local state, and
  // macOS never really unloads a library that does: loading a new build from
  // a path seen before would hand back the old code.
  std::string dir = o.work_dir + "/candidates_" + std::to_string(::getpid()) + "_" + std::to_string(batch);
  std::filesystem::create_directories(dir);
  std::ofstream(dir + "/candidates.cpp") << codegen::emit_cpp(program);
  codegen::build_shared_library(dir + "/candidates.cpp", dir + "/candidates.so");
  codegen::CompiledModel lib(dir + "/candidates.so", "");
  lib.set_threads(o.threads);

  std::vector<std::vector<float>> data;
  uint32_t state = 12345;
  for (const Type& t : program.inputs) {
    data.emplace_back(static_cast<size_t>(num_elements(t.shape)));
    for (float& v : data.back()) {
      state = state * 1664525u + 1013904223u;
      v = static_cast<float>(state >> 8) / 8388608.0f - 1.0f;
    }
  }
  std::vector<const void*> in_ptrs;
  for (auto& v : data) in_ptrs.push_back(v.data());
  float out_value = 0;
  void* out_ptrs[] = {&out_value};

  // Minimum over passes: noise on a quiet machine only ever adds time, so the
  // smallest sample is the best estimate of what the code can do.
  std::vector<double> best(candidates.size(), 1e30);
  auto t0 = std::chrono::steady_clock::now();
  lib.run(in_ptrs.data(), out_ptrs);  // a pass to fault the arena in, not timed
  for (int pass = 0; pass < 3 || seconds_since(t0) < o.seconds * static_cast<double>(candidates.size()); ++pass) {
    std::vector<codegen::CompiledModel::KernelTime> times = lib.run_profiled(in_ptrs.data(), out_ptrs);
    for (size_t c = 0; c < candidates.size(); ++c) best[c] = std::min(best[c], times[c].seconds);
  }
  std::filesystem::remove_all(dir);
  return best;
}

}  // namespace

std::vector<double> measure(const Product& product, const std::vector<MatmulParams>& candidates, const TuneOptions& o) {
  if (o.work_dir.empty()) throw TuneError("tuning needs a work directory");
  // Each copy is a few hundred lines of intrinsics; eight to a library keeps
  // the C++ compiler's time per candidate near its minimum.
  const size_t kBatch = 8;
  static int batch_number = 0;
  std::vector<double> out;
  for (size_t at = 0; at < candidates.size(); at += kBatch) {
    std::vector<MatmulParams> batch(candidates.begin() + static_cast<std::ptrdiff_t>(at),
                                    candidates.begin() + static_cast<std::ptrdiff_t>(std::min(candidates.size(), at + kBatch)));
    std::vector<double> times = measure_batch(product, batch, o, batch_number++);
    out.insert(out.end(), times.begin(), times.end());
  }
  return out;
}

// ---- Searching ----------------------------------------------------------------------------

Entry tune_product(const Product& product, const Machine& machine, const TuneOptions& o) {
  const Shape& a = product.inputs.at(0).shape;
  const Shape& bt = product.inputs.at(1).shape;
  loop::MatmulShape shape{a[a.size() - 2], a.back(), bt[bt.size() - 2]};
  Space space(shape, machine.l1d_bytes);
  std::mt19937 rng(o.seed);
  auto say = [&](const std::string& line) {
    if (o.log) o.log(line);
  };
  auto rate = [&](double seconds) { return product.flops / seconds / 1e9; };
  auto describe = [](const MatmulParams& q) {
    return std::to_string(q.tile_rows) + "x" + std::to_string(q.tile_cols) + (q.cols_outside ? " cols" : " rows") + " outside, " +
           (q.block_rows ? "blocks of " + std::to_string(q.block_rows) : std::string("no blocks"));
  };

  // Everything measured so far, by parameters. Two points can map to the same
  // parameters (block sizes round to whole tiles), and each is measured once.
  std::vector<std::pair<MatmulParams, double>> measured;
  std::vector<Point> points;  // the point each measured schedule came from, for mutation
  auto known = [&](const MatmulParams& q) {
    return std::any_of(measured.begin(), measured.end(), [&](const auto& m) { return m.first == q; });
  };
  auto run = [&](const std::vector<Point>& batch_points, const std::vector<MatmulParams>& batch) {
    if (batch.empty()) return;
    std::vector<double> times = measure(product, batch, o);
    for (size_t i = 0; i < batch.size(); ++i) {
      measured.push_back({batch[i], times[i]});
      points.push_back(batch_points[i]);
    }
  };
  auto ranking = [&] {
    std::vector<size_t> order(measured.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t x, size_t y) { return measured[x].second < measured[y].second; });
    return order;
  };

  // Generation 0: the default, plus random points.
  const MatmulParams fallback = loop::default_matmul_params(shape);
  std::vector<Point> gen_points = {Point{}};
  std::vector<MatmulParams> gen = {fallback};
  for (int tries = 0; static_cast<int>(gen.size()) < o.random + 1 && tries < o.random * 20; ++tries) {
    Point p = space.random(rng);
    MatmulParams q = space.params(p);
    if (std::find(gen.begin(), gen.end(), q) != gen.end()) continue;
    gen_points.push_back(p);
    gen.push_back(q);
  }
  run(gen_points, gen);
  say("  random: best " + describe(measured[ranking()[0]].first) + " at " + std::to_string(rate(measured[ranking()[0]].second)).substr(0, 6) + " GFLOP/s");

  // Later generations: small changes to the best few so far.
  for (int generation = 0; generation < o.generations; ++generation) {
    std::vector<size_t> order = ranking();
    gen_points.clear();
    gen.clear();
    for (int tries = 0; static_cast<int>(gen.size()) < o.children && tries < o.children * 20; ++tries) {
      size_t parent = order[static_cast<size_t>(tries) % std::min<size_t>(order.size(), 3)];
      // The default did not come from a point; its children are random points.
      Point p = measured[parent].first == fallback && parent == 0 ? space.random(rng) : space.mutate(points[parent], rng);
      MatmulParams q = space.params(p);
      if (known(q) || std::find(gen.begin(), gen.end(), q) != gen.end()) continue;
      gen_points.push_back(p);
      gen.push_back(q);
    }
    run(gen_points, gen);
    say("  generation " + std::to_string(generation + 1) + ": best " + describe(measured[ranking()[0]].first) + " at " +
        std::to_string(rate(measured[ranking()[0]].second)).substr(0, 6) + " GFLOP/s");
  }

  // The final: the leaders and the default, side by side, measured for longer.
  // Being the fastest of forty noisy samples is partly luck; this round is
  // what decides.
  std::vector<MatmulParams> finalists = {fallback};
  for (size_t i : ranking()) {
    if (static_cast<int>(finalists.size()) > o.finalists) break;
    if (!(measured[i].first == fallback)) finalists.push_back(measured[i].first);
  }
  TuneOptions longer = o;
  longer.seconds = o.seconds * 3;
  std::vector<double> times = measure(product, finalists, longer);
  size_t winner = 0;
  for (size_t i = 1; i < finalists.size(); ++i)
    if (times[i] < times[winner]) winner = i;
  if (times[winner] > times[0] * (1.0 - o.margin)) winner = 0;  // not clearly better: keep the default

  Entry e;
  e.params = finalists[winner];
  e.gflops = rate(times[winner]);
  e.default_gflops = rate(times[0]);
  say("  final: " + std::string(winner ? "tuned " : "default kept, ") + describe(e.params) + " at " + std::to_string(e.gflops).substr(0, 6) +
      " GFLOP/s (default " + describe(fallback) + " at " + std::to_string(e.default_gflops).substr(0, 6) + "); " +
      std::to_string(measured.size()) + " of " + std::to_string(space.size()) + " points tried");
  return e;
}

}  // namespace mlc::tune
