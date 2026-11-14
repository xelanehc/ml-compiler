// The reference interpreter: executes a graph node by node with the plainest
// possible loops. It is the definition of what every op means, and the oracle
// that later stages (optimized graphs, loop IR, generated code) are checked
// against. It is written to be obviously right, not fast.
#pragma once

#include <array>
#include <cstddef>
#include <memory>
#include <span>
#include <vector>

#include "graph/ir.h"

namespace mlc {

// A tensor at run time: a type plus a pointer to row-major (C order) data.
// `owner` keeps the allocation alive. It is empty when the bytes belong to
// someone else: the weights file, an inline constant, or a caller's buffer.
// Several tensors may share one allocation (a reshape is the same bytes with
// a different shape), which is safe because nodes never write to their inputs.
struct Tensor {
  Type type;
  std::shared_ptr<std::byte[]> owner;
  const std::byte* ptr = nullptr;

  template <class T>
  const T* data() const { return reinterpret_cast<const T*>(ptr); }
  int64_t size() const { return num_elements(type.shape); }

  // A tensor that borrows `ptr`; the caller guarantees it outlives the tensor.
  static Tensor borrow(Type type, const std::byte* ptr) { return Tensor{std::move(type), nullptr, ptr}; }
};

struct RunStats {
  std::array<double, kNumOps> seconds{};  // wall time per op kind
  std::array<int, kNumOps> calls{};
  size_t allocated_bytes = 0;  // sum of every buffer the run allocated
  size_t peak_bytes = 0;       // most that was alive at one moment
};

// Runs the graph. `inputs` match Graph::inputs in order and type. `weights`
// is the mapped weights file that external constants point into.
// Returns one tensor per graph output. Throws GraphError on any mismatch.
std::vector<Tensor> run_graph(const Graph& g, std::span<const Tensor> inputs, std::span<const std::byte> weights,
                              RunStats* stats = nullptr);

}  // namespace mlc
