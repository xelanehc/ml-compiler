#include "frontend/importer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>
#include <set>
#include <unordered_map>

namespace mlc {

namespace {

DType parse_dtype(const std::string& s) {
  if (s == "float32") return DType::F32;
  if (s == "int64") return DType::I64;
  if (s == "bool") return DType::Bool;
  throw ImportError("unsupported dtype '" + s + "'");
}

Type parse_type(const json::Value& v) {
  Type t;
  t.dtype = parse_dtype(v.at("dtype").as_string());
  for (const json::Value& d : v.at("shape").as_array()) t.shape.push_back(d.as_int());
  return t;
}

// Keyword arguments that describe where and how PyTorch stores a tensor. They
// carry no meaning for a CPU compiler with its own memory planning.
const std::set<std::string> kStorageKwargs = {"device", "layout", "pin_memory", "memory_format"};

// Lowers one graph.json document. The object lives for a single import.
//
// The central piece of state is `env_`, the symbol table: exported node name
// -> the IR value(s) that now stand for it. Most ATen nodes map to one value.
// A node returning a tuple maps to several, a node we fold away maps to the
// value it forwards, and an assertion maps to none.
class Importer {
 public:
  explicit Importer(const json::Value& doc) : doc_(doc) {}

  ImportedModel run() {
    if (doc_.at("format").as_int() != 1) throw ImportError("unsupported graph.json format version");

    ImportedModel model;
    model.weights_file = doc_.at("weights_file").as_string();
    model.weights_bytes = doc_.at("weights_bytes").as_int();

    for (const json::Value& in : doc_.at("inputs").as_array()) {
      const std::string& name = in.at("name").as_string();
      env_[name] = {g_.add_input(parse_type(in), name)};
    }

    // Parameters are recorded but not turned into nodes until something uses
    // them. GPT-2 exports the token embedding under two names (it is tied to
    // the output projection) and only one of them is ever referenced.
    for (const json::Value& p : doc_.at("params").as_array()) {
      Constant c;
      c.type = parse_type(p);
      c.weights_offset = p.at("offset").as_int();
      int64_t nbytes = p.at("nbytes").as_int();
      const std::string& name = p.at("name").as_string();
      if (nbytes != static_cast<int64_t>(c.type.byte_size())) throw ImportError("parameter " + name + ": nbytes does not match its shape");
      if (c.weights_offset < 0 || c.weights_offset + nbytes > model.weights_bytes)
        throw ImportError("parameter " + name + ": lies outside the weights file");
      params_.emplace(name, std::move(c));
    }

    for (const json::Value& node : doc_.at("nodes").as_array()) {
      lower_node(node);
      ++model.aten_nodes;
    }

    for (const json::Value& out : doc_.at("outputs").as_array()) g_.outputs.push_back(lookup(out.as_string()));

    check_graph(g_);
    model.graph = std::move(g_);
    return model;
  }

 private:
  const json::Value& doc_;
  Graph g_;
  std::unordered_map<std::string, std::vector<ValueId>> env_;
  std::unordered_map<std::string, Constant> params_;

  // The node being lowered, and which of its arguments the lowering looked at.
  const json::Value* node_ = nullptr;
  std::string name_;
  std::vector<bool> arg_read_;
  std::set<std::string> kwarg_read_;

  using Lowering = std::function<void(Importer&)>;
  static const std::unordered_map<std::string, Lowering>& lowerings();

  void lower_node(const json::Value& node) {
    node_ = &node;
    name_ = node.at("name").as_string();
    const std::string& op = node.at("op").as_string();
    arg_read_.assign(node.at("args").size(), false);
    kwarg_read_.clear();

    auto it = lowerings().find(op);
    if (it == lowerings().end()) throw ImportError("unsupported operator '" + op + "' (node " + name_ + ")");
    try {
      it->second(*this);

      // "No silent fallbacks" applies to arguments as much as to operators. If
      // the export passes something this lowering never looked at, we may be
      // computing a different function than PyTorch did, so refuse.
      for (size_t i = 0; i < arg_read_.size(); ++i)
        if (!arg_read_[i] && !node.at("args").at(i).is_null()) throw ImportError("argument " + std::to_string(i) + " is not understood");
      for (const json::Member& kw : node.at("kwargs").as_object())
        if (!kwarg_read_.count(kw.key) && !kStorageKwargs.count(kw.key) && !kw.value.is_null())
          throw ImportError("keyword argument '" + kw.key + "' is not understood");
    } catch (const std::exception& e) {
      throw ImportError("while lowering " + name_ + " = " + op + ": " + e.what());
    }
  }

  // ---- Reading arguments ----

  size_t nargs() const { return arg_read_.size(); }

  const json::Value& arg(size_t i) {
    if (i >= nargs()) throw ImportError("expected at least " + std::to_string(i + 1) + " arguments");
    arg_read_[i] = true;
    return node_->at("args").at(i);
  }

  bool has_arg(size_t i) const { return i < nargs() && !node_->at("args").at(i).is_null(); }

  const json::Value* kwarg(const std::string& key) {
    kwarg_read_.insert(key);
    const json::Value* v = node_->at("kwargs").find(key);
    return v && !v->is_null() ? v : nullptr;
  }

  static bool is_ref(const json::Value& a) { return a.is_object() && a.find("ref"); }

  ValueId lookup(const std::string& name) {
    auto it = env_.find(name);
    if (it == env_.end()) {
      auto p = params_.find(name);
      if (p == params_.end()) throw ImportError("reference to unknown value '" + name + "'");
      it = env_.emplace(name, std::vector<ValueId>{g_.add_const(p->second, name)}).first;
    }
    if (it->second.size() != 1) throw ImportError("'" + name + "' is not a single tensor");
    return it->second[0];
  }

  ValueId value(const json::Value& a) {
    if (!is_ref(a)) throw ImportError(std::string("expected a tensor argument, found ") + a.kind_name());
    return lookup(a.at("ref").as_string());
  }

  // A Python scalar: int, float, bool, or the {"float": "inf"} escape.
  static double number(const json::Value& a) {
    if (a.is_bool()) return a.as_bool() ? 1.0 : 0.0;
    if (a.is_number()) return a.as_double();
    if (a.is_object()) {
      if (const json::Value* f = a.find("float")) {
        const std::string& s = f->as_string();
        if (s == "inf") return std::numeric_limits<double>::infinity();
        if (s == "-inf") return -std::numeric_limits<double>::infinity();
        if (s == "nan") return std::numeric_limits<double>::quiet_NaN();
      }
    }
    throw ImportError(std::string("expected a number, found ") + a.kind_name());
  }

  static std::vector<int64_t> int_list(const json::Value& a) {
    std::vector<int64_t> out;
    for (const json::Value& v : a.as_array()) out.push_back(v.as_int());
    return out;
  }

  // Python allows negative axes counting from the end; the IR does not.
  static int64_t normalize_axis(int64_t axis, size_t rank) {
    int64_t r = static_cast<int64_t>(rank);
    if (axis < -r || axis >= r) throw ImportError("axis " + std::to_string(axis) + " out of range for rank " + std::to_string(rank));
    return axis < 0 ? axis + r : axis;
  }

  // The second operand of a binary op: a tensor, or a Python scalar that takes
  // on the tensor operand's dtype (PyTorch's rule for tensor-with-scalar
  // arithmetic). An int tensor with a fractional scalar would promote to
  // float in PyTorch; we do not need that and reject it.
  ValueId operand(const json::Value& a, DType like) {
    if (is_ref(a)) return value(a);
    if (like == DType::Bool) throw ImportError("scalar operand for a bool tensor");
    if (like == DType::I64 && !a.is_int()) throw ImportError("non-integer scalar operand for an int64 tensor");
    return g_.add_scalar(like, number(a));
  }

  // ---- Producing results ----

  Type exported_type(size_t i = 0) const {
    const json::Value& outs = node_->at("outputs");
    if (i >= outs.size()) throw ImportError("the export records no output " + std::to_string(i));
    return parse_type(outs.at(i));
  }

  ValueId emit(Op op, std::vector<ValueId> inputs, Attrs attrs = {}) { return g_.add(op, std::move(inputs), std::move(attrs), name_); }

  // Every result is compared with the type torch.export recorded. Our shape
  // inference and PyTorch's were written independently, so agreement on all
  // 852 nodes is real evidence that the typing rules in ir.cpp are right.
  void check_against_export(ValueId v, size_t i) const {
    Type want = exported_type(i);
    if (g_.type_of(v) != want)
      throw ImportError("we derive " + type_string(g_.type_of(v)) + " but the export says " + type_string(want));
  }

  void bind(ValueId v) {
    check_against_export(v, 0);
    env_[name_] = {v};
  }

  void bind_tuple(std::vector<ValueId> vs) {
    for (size_t i = 0; i < vs.size(); ++i)
      if (vs[i] != kNoValue) check_against_export(vs[i], i);
    env_[name_] = std::move(vs);
  }

  // A scalar constant repeated to `shape`. Emitted as scalar + Broadcast, not
  // as a filled buffer, so a [1,12,1024,1024] tensor of zeros costs 4 bytes
  // in the graph and constant folding can see that it is uniform.
  ValueId filled(DType dtype, double v, const Shape& shape) {
    ValueId s = g_.add_scalar(dtype, v, name_);
    if (shape.empty()) return s;
    Attrs a;
    a.dims = shape;
    return emit(Op::Broadcast, {s}, a);
  }

  // ---- Lowerings: elementwise ----

  void binary(Op op) {
    ValueId a = value(arg(0));
    ValueId b = operand(arg(1), g_.type_of(a).dtype);
    bind(emit(op, {a, b}));
  }

  void unary(Op op) { bind(emit(op, {value(arg(0))})); }

  void where() {
    ValueId cond = value(arg(0));
    bind(emit(Op::Where, {cond, value(arg(1)), value(arg(2))}));
  }

  // ---- Lowerings: layout ----

  void view() {
    ValueId x = value(arg(0));
    Attrs a;
    a.dims = int_list(arg(1));
    // One entry may be -1, meaning "whatever makes the element count match".
    int64_t known = 1, wildcard = -1;
    for (size_t i = 0; i < a.dims.size(); ++i) {
      if (a.dims[i] == -1) {
        if (wildcard >= 0) throw ImportError("more than one -1 in a view");
        wildcard = static_cast<int64_t>(i);
      } else {
        known *= a.dims[i];
      }
    }
    if (wildcard >= 0) {
      int64_t total = num_elements(g_.type_of(x).shape);
      if (known == 0 || total % known != 0) throw ImportError("cannot resolve -1 in a view");
      a.dims[wildcard] = total / known;
    }
    bind(emit(Op::Reshape, {x}, a));
  }

  void unsqueeze() {
    ValueId x = value(arg(0));
    Attrs a;
    a.dims = g_.type_of(x).shape;
    int64_t at = normalize_axis(arg(1).as_int(), a.dims.size() + 1);  // may insert after the last axis
    a.dims.insert(a.dims.begin() + at, 1);
    bind(emit(Op::Reshape, {x}, a));
  }

  void permute() {
    ValueId x = value(arg(0));
    Attrs a;
    for (int64_t d : int_list(arg(1))) a.dims.push_back(normalize_axis(d, g_.type_of(x).shape.size()));
    bind(emit(Op::Permute, {x}, a));
  }

  void expand() {
    ValueId x = value(arg(0));
    const Shape& in = g_.type_of(x).shape;
    Attrs a;
    a.dims = int_list(arg(1));
    if (a.dims.size() < in.size()) throw ImportError("expand to fewer axes than the input has");
    size_t shift = a.dims.size() - in.size();
    for (size_t i = 0; i < a.dims.size(); ++i) {
      if (a.dims[i] != -1) continue;  // -1 means "keep the input's size here"
      if (i < shift) throw ImportError("-1 in an expand for an axis the input does not have");
      a.dims[i] = in[i - shift];
    }
    bind(emit(Op::Broadcast, {x}, a));
  }

  // Python slicing: bounds may be negative (counted from the end), missing,
  // or past the end, and all of that is legal. Resolve it to plain bounds.
  void slice() {
    ValueId x = value(arg(0));
    const Shape& shape = g_.type_of(x).shape;
    Attrs a;
    a.axis = normalize_axis(has_arg(1) ? arg(1).as_int() : 0, shape.size());
    int64_t dim = shape[a.axis];
    auto clamp = [dim](int64_t i) { return std::max<int64_t>(0, std::min(dim, i < 0 ? i + dim : i)); };
    a.start = has_arg(2) ? clamp(arg(2).as_int()) : 0;
    a.end = has_arg(3) ? clamp(arg(3).as_int()) : dim;
    a.end = std::max(a.start, a.end);
    if (has_arg(4) && arg(4).as_int() != 1) throw ImportError("slice with a step other than 1");
    bind(emit(Op::Slice, {x}, a));
  }

  void split_with_sizes() {
    ValueId x = value(arg(0));
    std::vector<int64_t> sizes = int_list(arg(1));
    Attrs a;
    a.axis = normalize_axis(has_arg(2) ? arg(2).as_int() : 0, g_.type_of(x).shape.size());
    std::vector<ValueId> parts;
    for (int64_t size : sizes) {
      a.end = a.start + size;
      parts.push_back(emit(Op::Slice, {x}, a));
      a.start = a.end;
    }
    if (a.end != g_.type_of(x).shape[a.axis]) throw ImportError("split sizes do not add up to the axis length");
    bind_tuple(std::move(parts));
  }

  void cat() {
    std::vector<ValueId> parts;
    for (const json::Value& v : arg(0).as_array()) parts.push_back(value(v));
    if (parts.empty()) throw ImportError("cat of nothing");
    Attrs a;
    a.axis = normalize_axis(has_arg(1) ? arg(1).as_int() : 0, g_.type_of(parts[0]).shape.size());
    bind(emit(Op::Concat, std::move(parts), a));
  }

  void index() {
    std::vector<ValueId> ins = {value(arg(0))};
    for (const json::Value& v : arg(1).as_array()) {
      if (v.is_null()) throw ImportError("index with a skipped axis (None) is not supported");
      ins.push_back(value(v));
    }
    bind(emit(Op::Gather, std::move(ins)));
  }

  void embedding() {
    ValueId table = value(arg(0));
    bind(emit(Op::Gather, {table, value(arg(1))}));
  }

  // ---- Lowerings: constants ----

  DType dtype_kwarg(DType fallback) {
    const json::Value* v = kwarg("dtype");
    return v ? parse_dtype(v->at("dtype").as_string()) : fallback;
  }

  static DType dtype_of_scalar(const json::Value& a) { return a.is_bool() ? DType::Bool : a.is_int() ? DType::I64 : DType::F32; }

  void arange() {
    int64_t start = arg(0).as_int(), end = arg(1).as_int();
    int64_t step = has_arg(2) ? arg(2).as_int() : 1;
    if (step <= 0) throw ImportError("arange with a non-positive step");
    if (dtype_kwarg(DType::I64) != DType::I64) throw ImportError("arange of a non-int64 dtype");
    Constant c;
    for (int64_t v = start; v < end; v += step) {
      size_t at = c.inline_data.size();
      c.inline_data.resize(at + sizeof v);
      std::memcpy(c.inline_data.data() + at, &v, sizeof v);
    }
    c.type = {DType::I64, {static_cast<int64_t>(c.inline_data.size() / sizeof(int64_t))}};
    bind(g_.add_const(std::move(c), name_));
  }

  void full() {
    Shape shape = int_list(arg(0));
    const json::Value& v = arg(1);
    bind(filled(dtype_kwarg(dtype_of_scalar(v)), number(v), shape));
  }

  void scalar_tensor() {
    const json::Value& v = arg(0);
    bind(filled(dtype_kwarg(DType::F32), number(v), {}));
  }

  void full_like() {
    Type like = g_.type_of(value(arg(0)));
    bind(filled(dtype_kwarg(like.dtype), number(arg(1)), like.shape));
  }

  // ---- Lowerings: reductions ----

  void any_dim() {
    ValueId x = value(arg(0));
    Attrs a;
    a.axis = normalize_axis(arg(1).as_int(), g_.type_of(x).shape.size());
    a.keepdims = has_arg(2) && arg(2).as_bool();
    bind(emit(Op::ReduceAny, {x}, a));
  }

  void cumsum() {
    ValueId x = value(arg(0));
    // PyTorch's cumsum of a bool tensor counts: it returns int64. Our CumSum
    // keeps its operand's dtype, so the promotion becomes an explicit Cast.
    if (g_.type_of(x).dtype == DType::Bool) {
      Attrs c;
      c.to = DType::I64;
      x = emit(Op::Cast, {x}, c);
    }
    Attrs a;
    a.axis = normalize_axis(arg(1).as_int(), g_.type_of(x).shape.size());
    bind(emit(Op::CumSum, {x}, a));
  }

  // ---- Lowerings: linear algebra and composites ----

  void matmul() {
    ValueId a = value(arg(0));
    bind(emit(Op::MatMul, {a, value(arg(1))}));
  }

  // addmm(bias, a, b) = beta * bias + alpha * (a @ b). GPT-2's Conv1D layers
  // become this with beta = alpha = 1, which is exactly our Gemm.
  void addmm() {
    for (const char* key : {"beta", "alpha"})
      if (const json::Value* v = kwarg(key); v && number(*v) != 1.0) throw ImportError(std::string("addmm with ") + key + " != 1");
    ValueId bias = value(arg(0));
    ValueId a = value(arg(1));
    bind(emit(Op::Gemm, {a, value(arg(2)), bias}));
  }

  // native_layer_norm returns (output, mean, rstd). Only the output is ever
  // used at inference; the other two tuple slots are left empty, and a
  // getitem that asks for one is an error rather than a wrong answer.
  void layer_norm() {
    ValueId x = value(arg(0));
    std::vector<int64_t> normalized = int_list(arg(1));
    if (normalized.size() != 1) throw ImportError("layer norm over more than the last axis");
    if (!has_arg(2) || !has_arg(3)) throw ImportError("layer norm without weight and bias");
    ValueId weight = value(arg(2));
    ValueId bias = value(arg(3));
    Attrs a;
    a.eps = number(arg(4));
    bind_tuple({emit(Op::LayerNorm, {x, weight, bias}, a), kNoValue, kNoValue});
  }

  void softmax() {
    ValueId x = value(arg(0));
    Attrs a;
    a.axis = normalize_axis(arg(1).as_int(), g_.type_of(x).shape.size());
    if (has_arg(2) && arg(2).as_bool()) throw ImportError("softmax with half_to_float");
    bind(emit(Op::Softmax, {x}, a));
  }

  // ---- Lowerings that produce no node ----

  // Tuple indexing is resolved here, at import time, against the symbol table.
  void getitem() {
    const std::string& tuple = arg(0).at("ref").as_string();
    auto it = env_.find(tuple);
    if (it == env_.end()) throw ImportError("getitem on unknown value '" + tuple + "'");
    int64_t i = arg(1).as_int();
    if (i < 0 || static_cast<size_t>(i) >= it->second.size()) throw ImportError("tuple index out of range");
    ValueId v = it->second[i];
    if (v == kNoValue) throw ImportError("output " + std::to_string(i) + " of " + tuple + " is not computed by our lowering");
    bind(v);
  }

  // clone and alias exist because PyTorch tensors are mutable and can share
  // storage. Our values are immutable, so "a fresh copy of x" is just x.
  void identity() { bind(value(arg(0))); }

  // A check the exporter left in the graph. We can evaluate it right now,
  // because every dtype is static, and then it needs no node at all.
  void assert_metadata() {
    ValueId x = value(arg(0));
    for (size_t i = 1; i < nargs(); ++i) {
      const json::Value& a = arg(i);
      if (a.is_object() && a.find("dtype") && parse_dtype(a.at("dtype").as_string()) != g_.type_of(x).dtype)
        throw ImportError("dtype assertion fails");
    }
    if (kwarg("dtype") && dtype_kwarg(DType::F32) != g_.type_of(x).dtype) throw ImportError("dtype assertion fails");
    env_[name_] = {};
  }
};

// The whole ATen surface this compiler accepts. Anything not listed here stops
// the import with the operator's name.
const std::unordered_map<std::string, Importer::Lowering>& Importer::lowerings() {
  static const std::unordered_map<std::string, Lowering> table = {
      {"aten.add.Tensor", [](Importer& i) { i.binary(Op::Add); }},
      {"aten.sub.Tensor", [](Importer& i) { i.binary(Op::Sub); }},
      {"aten.mul.Tensor", [](Importer& i) { i.binary(Op::Mul); }},
      {"aten.mul.Scalar", [](Importer& i) { i.binary(Op::Mul); }},
      {"aten.pow.Tensor_Scalar", [](Importer& i) { i.binary(Op::Pow); }},
      {"aten.eq.Tensor", [](Importer& i) { i.binary(Op::Eq); }},
      {"aten.eq.Scalar", [](Importer& i) { i.binary(Op::Eq); }},
      {"aten.ne.Scalar", [](Importer& i) { i.binary(Op::Ne); }},
      {"aten.le.Tensor", [](Importer& i) { i.binary(Op::Le); }},
      {"aten.bitwise_and.Tensor", [](Importer& i) { i.binary(Op::And); }},
      {"aten.tanh.default", [](Importer& i) { i.unary(Op::Tanh); }},
      {"aten.logical_not.default", [](Importer& i) { i.unary(Op::Not); }},
      {"aten.where.self", [](Importer& i) { i.where(); }},

      {"aten.view.default", [](Importer& i) { i.view(); }},
      {"aten.unsqueeze.default", [](Importer& i) { i.unsqueeze(); }},
      {"aten.permute.default", [](Importer& i) { i.permute(); }},
      {"aten.expand.default", [](Importer& i) { i.expand(); }},
      {"aten.slice.Tensor", [](Importer& i) { i.slice(); }},
      {"aten.split_with_sizes.default", [](Importer& i) { i.split_with_sizes(); }},
      {"aten.cat.default", [](Importer& i) { i.cat(); }},
      {"aten.index.Tensor", [](Importer& i) { i.index(); }},
      {"aten.embedding.default", [](Importer& i) { i.embedding(); }},

      {"aten.arange.start_step", [](Importer& i) { i.arange(); }},
      {"aten.full.default", [](Importer& i) { i.full(); }},
      {"aten.scalar_tensor.default", [](Importer& i) { i.scalar_tensor(); }},
      {"aten.full_like.default", [](Importer& i) { i.full_like(); }},

      {"aten.any.dim", [](Importer& i) { i.any_dim(); }},
      {"aten.cumsum.default", [](Importer& i) { i.cumsum(); }},

      {"aten.mm.default", [](Importer& i) { i.matmul(); }},
      {"aten.bmm.default", [](Importer& i) { i.matmul(); }},
      {"aten.addmm.default", [](Importer& i) { i.addmm(); }},
      {"aten.native_layer_norm.default", [](Importer& i) { i.layer_norm(); }},
      {"aten._softmax.default", [](Importer& i) { i.softmax(); }},

      {"getitem", [](Importer& i) { i.getitem(); }},
      {"aten.clone.default", [](Importer& i) { i.identity(); }},
      {"aten.alias.default", [](Importer& i) { i.identity(); }},
      {"aten._assert_tensor_metadata.default", [](Importer& i) { i.assert_metadata(); }},
  };
  return table;
}

}  // namespace

ImportedModel import_graph(const json::Value& doc) {
  try {
    return Importer(doc).run();
  } catch (const json::Error& e) {
    throw ImportError(std::string("malformed graph.json: ") + e.what());
  } catch (const GraphError& e) {
    throw ImportError(e.what());
  }
}

ImportedModel import_graph_file(const std::string& path) {
  try {
    return import_graph(json::parse_file(path));
  } catch (const json::Error& e) {
    throw ImportError(path + ": " + e.what());
  }
}

}  // namespace mlc
