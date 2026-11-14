// The autotuner: its search space, its cache, and a real (very short) search.
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <set>

#include "check.h"
#include "graph/passes.h"
#include "helpers.h"
#include "loop/schedule.h"
#include "tune/tuner.h"

using namespace mlc;
using namespace mlc::test;

namespace {

std::string scratch(const char* name) {
  std::filesystem::path dir = std::filesystem::temp_directory_path() / ("mlc_tune_" + std::to_string(::getpid()) + "_" + name);
  std::filesystem::create_directories(dir);
  return dir.string();
}

Attrs transposed() {
  Attrs a;
  a.trans_b = true;
  return a;
}

}  // namespace

TEST(tune_space_holds_only_schedules_that_fit_the_registers) {
  tune::Space space({1024, 768, 3072}, 128 << 10);
  CHECK(space.tiles.size() > 10);
  for (auto [r, c] : space.tiles) {
    CHECK(r * c + r + c <= 32);  // accumulators plus loaded operands, in 32 vector registers
    CHECK(r * c >= 12);
  }
  std::mt19937 rng(7);
  for (int i = 0; i < 200; ++i) {
    tune::Point p = space.random(rng);
    loop::MatmulParams q = space.params(p);
    const int64_t tile = q.cols_outside ? q.tile_cols : q.tile_rows;
    CHECK(q.block_rows % tile == 0);              // a block is whole tiles
    CHECK((p.block < 0) == (q.block_rows == 0));  // -1 is "no cache tiling"
    tune::Point m = space.mutate(p, rng);
    int changed = (m.tile != p.tile) + (m.block != p.block) + (m.cols_outside != p.cols_outside);
    CHECK(changed <= 1);                          // a mutation is one small step (or none, at an edge)
    CHECK(m.tile >= 0 && m.tile < static_cast<int>(space.tiles.size()) && m.block >= -1 &&
          m.block < static_cast<int>(space.block_fractions.size()));
  }
  // A tile is never larger than the matrix.
  tune::Space small({3, 16, 5}, 128 << 10);
  for (auto [r, c] : small.tiles) CHECK(r <= 3 && c <= 5);
  tune::Space tiny({2, 16, 2}, 128 << 10);
  CHECK(tiny.tiles.size() == 1 && tiny.tiles[0] == std::pair(2, 2));
}

TEST(tune_every_point_of_the_space_is_a_correct_schedule) {
  // The tuner may pick any point, so every point has to compute the product.
  Graph g;
  g.outputs = {g.add(Op::MatMul, {random_f32(g, {9, 16}, 1), random_f32(g, {23, 16}, 2)}, transposed())};
  loop::Program naive = loop::lower_graph(g, 0, kAtRunTime);
  std::vector<std::vector<std::byte>> want = run_loops(naive);
  tune::Space space(loop::matmul_shape(naive.kernels[0]), 1024);  // a 1 KB "L1", so blocks are a few rows
  std::set<std::tuple<int, int, bool, int64_t>> seen;
  loop::ScheduleOptions serial;
  serial.parallel = false;
  for (int tile = 0; tile < static_cast<int>(space.tiles.size()); ++tile) {
    for (int block = -1; block < static_cast<int>(space.block_fractions.size()); ++block) {
      for (bool cols : {false, true}) {
        loop::MatmulParams q = space.params({tile, block, cols});
        if (!seen.insert({q.tile_rows, q.tile_cols, q.cols_outside, q.block_rows}).second) continue;
        loop::Program p = naive;
        loop::schedule_kernel(p, p.kernels[0], serial, &q);
        loop::check_program(p);
        CHECK(run_loops(p) == want);
      }
    }
  }
  CHECK(seen.size() > 100);
}

TEST(tune_cache_round_trips_and_is_keyed_by_machine_and_threads) {
  std::string dir = scratch("cache");
  std::string path = dir + "/cache.json";
  CHECK(tune::load_cache(path).products.empty());  // no file: an empty cache

  tune::Cache cache;
  cache.machine = {"Test CPU \"X\"", 65536, 6, "AC"};
  cache.products[{1, "gemm [16,768] x [2304,768]^T"}] = {{3, 6, true, 12}, 91.5, 88.0};
  cache.products[{4, "gemm [16,768] x [2304,768]^T"}] = {{4, 4, false, 0}, 300.0, 300.0};
  cache.parallel_work[4] = 1 << 14;
  tune::save_cache(path, cache);
  tune::Cache back = tune::load_cache(path);
  CHECK(back.machine.cpu == cache.machine.cpu && back.machine.l1d_bytes == 65536 && back.machine.performance_cores == 6);
  CHECK(back.products.size() == 2 && back.parallel_work == cache.parallel_work);
  const tune::Entry& e = back.products.at({1, "gemm [16,768] x [2304,768]^T"});
  CHECK((e.params == loop::MatmulParams{3, 6, true, 12}) && e.gflops == 91.5 && e.default_gflops == 88.0);

  loop::ScheduleOptions one, four, other;
  CHECK(tune::apply_cache(back, cache.machine, 1, one) == 1);
  CHECK(one.tuned.size() == 1 && one.tuned.begin()->second.tile_cols == 6 && one.parallel_work == loop::ScheduleOptions{}.parallel_work);
  CHECK(tune::apply_cache(back, cache.machine, 4, four) == 1 && four.parallel_work == (1 << 14));
  tune::Machine elsewhere = cache.machine;
  elsewhere.cpu = "Another CPU";
  CHECK(tune::apply_cache(back, elsewhere, 1, other) == 0 && other.tuned.empty());  // measurements do not travel

  std::ofstream(path) << "{\"cpu\": 3}";
  CHECK_THROWS(tune::load_cache(path), tune::TuneError);
  std::filesystem::remove_all(dir);
}

TEST(tune_products_are_named_as_the_lowering_names_them) {
  Graph g;
  ValueId x = g.add_input({DType::F32, {6, 8}}, "x");
  ValueId w = random_f32(g, {8, 8}, 1), bias = random_f32(g, {8}, 2);
  ValueId h = g.add(Op::Gemm, {x, w, bias});
  ValueId h2 = g.add(Op::Gemm, {h, w, bias});  // the same shapes again: one product, used twice
  g.outputs = {g.add(Op::MatMul, {h2, random_f32(g, {8, 5}, 3)})};
  Graph laid_out = eliminate_dead_code(transpose_matmul_operands(g));
  std::vector<tune::Product> products = tune::products_of(laid_out);
  CHECK(products.size() == 2);
  CHECK(products[0].signature == "gemm [6,8] x [8,8]^T" && products[0].flops == 2.0 * 6 * 8 * 8);
  CHECK(tune::products_of(g).empty());  // before the layout pass nothing is in dot form

  loop::Program p = loop::lower_graph(laid_out, 0);
  std::set<std::string> signatures;
  for (const loop::Kernel& k : p.kernels)
    if (!k.signature.empty()) signatures.insert(k.signature);
  CHECK(signatures == (std::set<std::string>{products[0].signature, products[1].signature}));

  // A tuned entry changes the loops of every kernel with that signature, and nothing else.
  loop::ScheduleOptions tuned;
  tuned.parallel = false;
  loop::ScheduleOptions plain = tuned;
  tuned.tuned[products[0].signature] = {2, 6, false, 2};
  loop::Program a = loop::schedule_program(p, plain), b = loop::schedule_program(p, tuned);
  int differing = 0;
  for (size_t k = 0; k < a.kernels.size(); ++k) differing += loop::print_kernel(a, a.kernels[k]) != loop::print_kernel(b, b.kernels[k]);
  CHECK(differing == 2);
  std::vector<float> data = noise(48, 4);
  const void* inputs[] = {data.data()};
  CHECK(run_loops(a, inputs) == run_loops(b, inputs));
}

TEST(tune_a_small_product_end_to_end) {
  // A real search, cut down to a handful of candidates: build, time, pick.
  tune::Product product;
  product.op = Op::MatMul;
  product.attrs = transposed();
  product.inputs = {{DType::F32, {24, 64}}, {DType::F32, {40, 64}}};
  product.signature = "matmul [24,64] x [40,64]^T";
  product.flops = 2.0 * 24 * 64 * 40;
  tune::TuneOptions o;
  o.random = 3;
  o.generations = 1;
  o.children = 2;
  o.finalists = 2;
  o.seconds = 0.01;
  o.work_dir = scratch("search");
  int lines = 0;
  o.log = [&](const std::string&) { ++lines; };
  tune::Machine machine = tune::detect_machine();
  CHECK(machine.l1d_bytes >= 16384 && machine.performance_cores >= 1 && !machine.cpu.empty());

  tune::Entry e = tune::tune_product(product, machine, o);
  CHECK(lines == 3);  // random, one generation, final
  CHECK(e.gflops > 0.5 && e.default_gflops > 0.5);
  CHECK(e.gflops >= e.default_gflops);  // the default is only ever replaced by something measured faster
  CHECK(e.params.tile_rows >= 1 && e.params.tile_rows * e.params.tile_cols + e.params.tile_rows + e.params.tile_cols <= 32);

  // Measuring the same schedule twice in one library gives two times of the same order.
  std::vector<double> twice = tune::measure(product, {e.params, e.params}, o);
  CHECK(twice.size() == 2 && twice[0] > 0 && twice[1] < twice[0] * 5 && twice[0] < twice[1] * 5);
  std::filesystem::remove_all(o.work_dir);
}
