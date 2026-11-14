// The graph IR: what the model computes, with no decision yet about how.
//
// Design in one paragraph. The graph is a list of nodes in execution order.
// Every node produces exactly one tensor, so "node" and "value" are the same
// thing and a value is named by its node's index (a ValueId). That is static
// single assignment (SSA) form: each value is defined once, by one node, and
// never changes. Every value carries a static type, meaning a dtype and a
// fully known shape. Nodes are pure: the result depends only on the inputs and
// the node's attributes, nothing is modified in place. Purity is what later
// lets passes delete, merge, and reorder nodes without asking what else might
// be watching.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace mlc {

struct GraphError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// ---- Types -----------------------------------------------------------------

enum class DType : uint8_t { F32, I64, Bool };

size_t dtype_size(DType d);          // bytes per element: 4, 8, 1
const char* dtype_name(DType d);     // "f32", "i64", "bool"

using Shape = std::vector<int64_t>;  // a scalar has the empty shape, rank 0

int64_t num_elements(const Shape& s);
std::string shape_string(const Shape& s);  // "[1,16,768]"

struct Type {
  DType dtype = DType::F32;
  Shape shape;
  bool operator==(const Type&) const = default;
  size_t byte_size() const { return static_cast<size_t>(num_elements(shape)) * dtype_size(dtype); }
};

std::string type_string(const Type& t);  // "f32[1,16,768]"

// NumPy/PyTorch broadcasting of two shapes: align them at the right, and in
// each position the sizes must be equal or one of them must be 1.
// Throws GraphError when the shapes are incompatible.
Shape broadcast_shapes(const Shape& a, const Shape& b);

// ---- Operators -------------------------------------------------------------
//
// The set is the smallest one that covers what GPT-2 exports to. ATen has
// several spellings for most of these (mm/bmm, view/unsqueeze, mul.Tensor/
// mul.Scalar); the importer folds them onto one op each.

enum class Op : uint8_t {
  // Leaves
  Input,      // a graph input.                      attrs.index = position in Graph::inputs
  Const,      // a constant tensor.                  attrs.index = position in Graph::constants

  // Elementwise, with broadcasting
  Add, Sub, Mul,   // (a, b) same dtype, f32 or i64
  Pow,             // (a, b) f32
  Tanh,            // (a)    f32
  Eq, Ne, Le,      // (a, b) same dtype -> bool
  And,             // (a, b) bool
  Not,             // (a)    bool
  Where,           // (cond: bool, a, b) -> cond ? a : b
  Cast,            // (a) -> attrs.to

  // Layout: these move or repeat elements and compute nothing
  Reshape,    // (a) same elements, new shape.       attrs.dims = new shape
  Permute,    // (a) reorder axes.                   attrs.dims = permutation; output axis i is input axis dims[i]
  Broadcast,  // (a) repeat along size-1 axes.       attrs.dims = new shape
  Slice,      // (a) a[start:end] along one axis.    attrs.axis, attrs.start, attrs.end
  Concat,     // (a, b, ...) join along one axis.    attrs.axis
  Gather,     // (x, i0, ..., ik-1) -> x[i0, ..., ik-1]; the index tensors are i64 and broadcast together

  // Reductions and scans
  CumSum,     // (a) running sum along attrs.axis
  ReduceAny,  // (a: bool) logical OR along attrs.axis; attrs.keepdims

  // Linear algebra
  MatMul,     // (a, b) matrix product over the last two axes; leading axes are a batch
  Gemm,       // (a, b, bias) a @ b + bias, with a and b rank 2

  // Composites. Each could be spelled with the ops above; keeping them whole
  // preserves the structure a kernel generator wants to see.
  LayerNorm,  // (x, weight, bias) normalize over the last axis. attrs.eps
  Gelu,       // (x) 0.5 x (1 + tanh(sqrt(2/pi) (x + 0.044715 x^3))), the tanh approximation GPT-2 uses.
              //     Never imported; produced by fusion from its nine-node expansion.
  Softmax,    // (x) along attrs.axis. With attrs.causal, x is [..., T, T], the axis is the last one, and
              //     row i is normalized over columns 0..i only; columns after i are 0. Produced by fusion
              //     from "add a causal mask, then softmax".
};

inline constexpr int kNumOps = static_cast<int>(Op::Softmax) + 1;

const char* op_name(Op op);  // "matmul", "layer_norm", ...

// One struct holds every attribute any op uses; each op reads only its own
// fields and the rest stay at their defaults. A tagged union per op would be
// tighter, but this keeps equality (for CSE later) and printing trivial.
struct Attrs {
  int64_t axis = 0;           // Slice, Concat, CumSum, ReduceAny, Softmax
  int64_t start = 0;          // Slice
  int64_t end = 0;            // Slice
  std::vector<int64_t> dims;  // Reshape, Broadcast: target shape. Permute: the permutation.
  bool keepdims = false;      // ReduceAny
  double eps = 0.0;           // LayerNorm
  DType to = DType::F32;      // Cast
  int32_t index = -1;         // Input, Const
  bool causal = false;        // Softmax
  bool operator==(const Attrs&) const = default;
};

// Computes the result type of `op` applied to inputs of the given types, or
// throws GraphError explaining why the application is ill-formed. This one
// function is the definition of every op's typing rule: the builder uses it
// to type new nodes, and the verifier uses it to re-check existing ones.
// Input and Const have no inputs to infer from and are not handled here.
Type infer_type(Op op, std::span<const Type> inputs, const Attrs& attrs);

// ---- Graph -----------------------------------------------------------------

using ValueId = uint32_t;
inline constexpr ValueId kNoValue = ~ValueId{0};

// A constant's bytes live in one of two places: inline in the graph (scalars,
// small index tables), or in the weights file at a byte offset. Generated code
// never embeds weight data, so the big tensors stay external end to end.
struct Constant {
  Type type;
  std::vector<std::byte> inline_data;  // used when weights_offset < 0
  int64_t weights_offset = -1;
  bool is_external() const { return weights_offset >= 0; }
};

struct Node {
  Op op;
  Type type;                    // the type of the value this node produces
  std::vector<ValueId> inputs;  // each is the index of an earlier node
  Attrs attrs;
  std::string name;             // from the exporter; for humans only
};

struct Graph {
  std::vector<Node> nodes;          // in execution (topological) order
  std::vector<Constant> constants;
  std::vector<ValueId> inputs;      // Input nodes, in call-argument order
  std::vector<ValueId> outputs;

  const Type& type_of(ValueId v) const { return nodes.at(v).type; }

  // Appends a node, inferring its type from its inputs. Throws GraphError if
  // the op cannot be applied to them, so an ill-typed graph cannot be built
  // through this interface.
  ValueId add(Op op, std::vector<ValueId> inputs, Attrs attrs = {}, std::string name = {});

  ValueId add_input(Type type, std::string name);
  ValueId add_const(Constant constant, std::string name = {});
  // A rank-0 constant. `value` is converted to `dtype`.
  ValueId add_scalar(DType dtype, double value, std::string name = {});
};

// ---- Printer and verifier --------------------------------------------------

// The whole graph as text, one node per line:
//   %212: f32[12,16,64] = reshape(%211, dims=[12,16,64])   # view_6
std::string print_graph(const Graph& g);
std::string print_node(const Graph& g, ValueId v);

// A short report: node count and a histogram of ops.
std::string summarize_graph(const Graph& g);

// Checks every structural rule the rest of the compiler relies on: inputs are
// defined before use, each node's stored type is what infer_type derives,
// constants have the right amount of data, inputs and outputs are in range.
// Returns one message per problem; an empty result means the graph is valid.
std::vector<std::string> verify_graph(const Graph& g);

// Throws GraphError listing the problems, if there are any.
void check_graph(const Graph& g);

}  // namespace mlc
