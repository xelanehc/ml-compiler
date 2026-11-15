# 4. The graph IR and the importer

The graph IR is the compiler's first internal form of the model. It answers one question: *what is computed?* Nothing in it says how, in what loop order, or where in memory.

## 4.1 The design in one paragraph

A graph is a list of **nodes** in execution order. Each node applies one operator to the results of earlier nodes and produces exactly one tensor. Because a node produces exactly one value, a value is simply named by its node's position in the list: `%29` is "the result of node 29". Every value has a fixed **type**: a dtype and a complete shape.

Three properties are worth names, because the rest of the compiler leans on them.

**Static single assignment (SSA).** Every value is defined exactly once and never changes. There is no "variable `x` that gets updated". The residual connection `x = x + attention(...)` in Python becomes a new value `%42 = add(%41, %6)`; `%6` still exists unchanged. With SSA, "where did this value come from?" always has one answer, which makes pattern matching (chapter 6) straightforward.

**Purity.** A node's result depends only on its inputs and its attributes. Nothing is modified in place and nothing has side effects. Consequences: a node nobody uses can be deleted; two nodes with the same operator and inputs compute the same thing and can be merged; a node whose inputs are constants can be computed at compile time. None of these would be safe if operations could modify shared state.

**Static types.** Every shape is known. The compiler can therefore check itself: after any transformation, the types must still be consistent.

## 4.2 The data structures (`src/graph/ir.h`)

```cpp
enum class DType { F32, I64, Bool };
using Shape = std::vector<int64_t>;
struct Type { DType dtype; Shape shape; };

struct Node {
  Op op;                        // which operator
  Type type;                    // type of the value this node produces
  std::vector<ValueId> inputs;  // indices of earlier nodes
  Attrs attrs;                  // operator parameters that are not tensors
  std::string name;             // PyTorch's name, for humans only
};

struct Graph {
  std::vector<Node> nodes;          // in execution order
  std::vector<Constant> constants;
  std::vector<ValueId> inputs;      // the Input nodes
  std::vector<ValueId> outputs;
};
```

**Attributes** are an operator's non-tensor parameters: which axis to slice, the new shape of a reshape, the permutation. All operators share one `Attrs` struct with a field for each kind (`axis`, `start`, `end`, `dims`, `keepdims`, `eps`, `to`, `index`, `causal`, `trans_b`); each operator reads only its own fields. A separate struct per operator would be tighter, but one struct makes "are these two nodes identical?" a single comparison.

**Constants** come in two kinds, in one struct:

```cpp
struct Constant {
  Type type;
  std::vector<std::byte> inline_data;  // small constants: the bytes themselves
  int64_t weights_offset = -1;         // weights: a position in the weights file
};
```

Scalars like `0.5` and small tables like the position indices `0..15` are **inline**. The 148 weight tensors are **external**: the graph holds only an offset. The compiler never loads weight data.

## 4.3 The 27 operators

| Group | Operators | Notes |
|---|---|---|
| Leaves | `Input`, `Const` | |
| Elementwise (with broadcasting) | `Add`, `Sub`, `Mul`, `Pow`, `Tanh`, `Eq`, `Ne`, `Le`, `And`, `Not`, `Where`, `Cast` | `Where(c, a, b)` is `c ? a : b` per element |
| Layout (move data, compute nothing) | `Reshape`, `Permute`, `Broadcast`, `Slice`, `Concat`, `Gather` | |
| Reductions and scans | `CumSum`, `ReduceAny` | running sum; logical OR along an axis |
| Linear algebra | `MatMul`, `Gemm` | `Gemm(a, b, bias) = a @ b + bias` |
| Composites | `LayerNorm`, `Gelu`, `Softmax` | |

Two design decisions are visible here.

**Many PyTorch spellings fold onto one operator.** ATen has `mm` (matrix times matrix) and `bmm` (batched); both become `MatMul`, which treats any leading axes as a batch. `view` and `unsqueeze` both become `Reshape`. `mul.Tensor` and `mul.Scalar` both become `Mul`.

**Composite operators are kept whole.** Layer norm could be written with subtract, multiply, mean, and square root nodes. It is kept as one `LayerNorm` node because the loop generator can produce a much better loop nest for "layer norm" than for a chain of generic pieces. `Gelu`, the causal form of `Softmax`, and the `trans_b` form of the matrix products never come from the importer; graph passes create them (chapter 6).

## 4.4 Shape inference: `infer_type`

One function defines, for every operator, what type the result has given the input types and attributes, or that the combination is illegal:

```cpp
Type infer_type(Op op, std::span<const Type> inputs, const Attrs& attrs);
```

Examples of its rules:

- `Add(a, b)`: dtypes must match and not be bool; result shape is the broadcast of the two shapes.
- `Eq(a, b)`: same, but the result dtype is `Bool`.
- `Reshape(a)` with `dims`: the product of `dims` must equal `a`'s element count.
- `Permute(a)` with `dims`: `dims` must be a permutation of `0..rank-1`; output axis `i` has the size of input axis `dims[i]`.
- `MatMul(a, b)`: both `f32`, same rank of at least 2, equal batch axes, and `a`'s last size equal to `b`'s second-to-last. Result: batch axes, `a`'s rows, `b`'s columns.
- `Gemm(a, b, bias)`: rank 2 each, inner sizes match, bias length equals the number of result columns.

It is used in two places. `Graph::add` calls it to compute the type of each new node, so an ill-typed graph cannot be constructed. The verifier calls it again on every node of a finished graph and checks the stored type matches. One definition, used both to build and to check.

## 4.5 The printer and the verifier

`print_graph` (`src/graph/printer.cpp`) produces the text form shown in section 2.5; `mlc dump` prints it and `mlc stats` prints a count of nodes per operator.

`verify_graph` (`src/graph/verifier.cpp`) checks the invariants everything else assumes:

- every input of a node is an earlier node (no use before definition, so no cycles);
- every node's stored type is what `infer_type` derives;
- every constant has exactly as many bytes as its type needs, or a valid offset;
- `inputs` and `outputs` refer to real nodes.

It returns a list of problems; `check_graph` throws if the list is not empty. It runs after import, before every interpreter run, and after every optimization pass. A pass that produces a malformed graph is caught at once, with the pass's name in the error.

## 4.6 The importer (`src/frontend/importer.cpp`)

The importer reads `graph.json` and builds the graph IR. It is a table: one entry per ATen operator, each a small function that reads the node's arguments and adds IR nodes.

### The symbol table

The central state is `env_`, a map from an exported node's *name* to the IR value or values that now stand for it. Most ATen nodes map to one IR value. A few are special:

- A node that returns a **tuple** maps to several values. `split_with_sizes` (splitting `qkv` in three) becomes three `Slice` nodes. `native_layer_norm` returns (output, mean, rstd); only the first is ever used, so the other two slots are left empty.
- **`getitem`** (pick element `i` of a tuple) creates no node. It looks the tuple up in `env_` and binds its own name to the `i`-th value. If something asks for a slot that was left empty, that is an error.
- **`clone` and `alias`** create no node. They exist in PyTorch because tensors are mutable; in an IR where values never change, "a copy of `x`" is just `x`.
- **`_assert_tensor_metadata`** is a check PyTorch left in the graph. All dtypes are static, so it is evaluated during import and vanishes.

### Parameters are created lazily

Parameters are recorded in a side table and only become `Const` nodes when something refers to them. The token embedding is exported under two names (tied weights), and only one is referenced.

### Python-isms resolved at import

PyTorch arguments carry conveniences the IR does not have:

| In PyTorch | In the IR |
|---|---|
| negative axis (`-1` means last) | always `0..rank-1` |
| `-1` in a `view` shape ("work it out") | the computed size |
| slice bounds that are negative, missing, or past the end | plain `start` and `end` |
| a Python scalar operand (`x * 0.5`) | a rank-0 `Const` node of `x`'s dtype |
| `full(shape, value)` and `full_like` | a scalar `Const` plus a `Broadcast` |
| `cumsum` of a bool tensor (returns int64) | an explicit `Cast`, then `CumSum` |
| `addmm(bias, a, b)` | `Gemm(a, b, bias)` |

The scalar-operand rule is why the IR has more nodes (1062) than the export (852): every Python number becomes a constant node.

The `full` rule deserves a comment. A `[1, 12, 1024, 1024]` tensor of zeros would be 48 MB as a constant. As "scalar zero, broadcast" it costs 4 bytes in the graph, and a later pass can see at a glance that the whole tensor is zero.

### No silent fallbacks

Two rules make the importer trustworthy:

1. **An operator not in the table is an error naming the operator.** There is no "skip what you do not understand".
2. **An argument the lowering did not read is also an error.** Each lowering function records which arguments it looked at. If the export passes, say, `alpha=2` to `addmm` and the lowering ignored it, the compiler would compute a different function than PyTorch. (Arguments that only describe storage, like `device`, are on an explicit ignore list.)

### Checking shape inference against PyTorch

Every node the importer creates is passed through `check_against_export`: the type our `infer_type` derived is compared with the type PyTorch recorded in the node's `outputs` field. The two were written independently, so agreement on every node of the real model, at four sequence lengths, is strong evidence the typing rules are right. A mismatch stops the import with both types in the message.

### The full table

| ATen operator | IR |
|---|---|
| `add.Tensor`, `sub.Tensor`, `mul.Tensor`, `mul.Scalar`, `pow.Tensor_Scalar` | `Add`, `Sub`, `Mul`, `Mul`, `Pow` |
| `eq.Tensor`, `eq.Scalar`, `ne.Scalar`, `le.Tensor`, `bitwise_and.Tensor` | `Eq`, `Eq`, `Ne`, `Le`, `And` |
| `tanh`, `logical_not`, `where.self` | `Tanh`, `Not`, `Where` |
| `view`, `unsqueeze` | `Reshape` |
| `permute`, `expand`, `slice.Tensor` | `Permute`, `Broadcast`, `Slice` |
| `split_with_sizes` | one `Slice` per piece |
| `cat`, `index.Tensor`, `embedding` | `Concat`, `Gather`, `Gather` |
| `arange.start_step` | an inline `Const` holding the integers |
| `full`, `full_like`, `scalar_tensor` | scalar `Const` (+ `Broadcast`) |
| `any.dim`, `cumsum` | `ReduceAny`, (`Cast` +) `CumSum` |
| `mm`, `bmm`, `addmm` | `MatMul`, `MatMul`, `Gemm` |
| `native_layer_norm`, `_softmax` | `LayerNorm`, `Softmax` |
| `getitem`, `clone`, `alias`, `_assert_tensor_metadata` | no node |

## 4.7 The JSON reader (`src/support/json.*`)

About 360 lines, written for this project instead of using a library. It is a recursive-descent parser (one function per kind of JSON value, each calling the others) producing a `Value` that is null, bool, integer, float, string, array, or object. It is strict (it rejects anything that is not valid JSON, including a bare `Infinity`) and it keeps integers and floats distinct, which the importer needs. Accessors like `as_int()` throw a `json::Error` if the value is the wrong kind, so a malformed graph file produces a clear message rather than a crash.

## 4.8 What the imported graph looks like

After import, at any `T`: 1062 nodes. 290 constants (148 weights, the rest scalars and index tables), 230 reshapes, 85 permutes, 61 broadcasts, 73 matrix products (48 `Gemm`, 25 `MatMul`), and the rest elementwise.

The first thirty nodes of `mlc dump artifacts/graphs/gpt2_T16.json` are a good example of what a faithful import of a traced program looks like:

```
%0: i64[1,16] = input(0)                                # input_ids
%1: i64[1,16] = reshape(%0, dims=[1,16])                # a view to the same shape: does nothing
%2: f32[50257,768] = const(weights+0)                   # the token embedding table
%3: f32[1,16,768] = gather(%2, %1)                      # embedding lookup
%4: i64[16] = const(inline)                             # 0, 1, ..., 15
%5: i64[] = const(0)
%6: i64[16] = add(%4, %5)                               # positions + 0: does nothing
...
%19: bool[1,16] = ne(%17, %18)
%20: i64[1,16] = cast(%19, to=i64)
%21: i64[1,16] = cumsum(%20, axis=1)                    # part of building the attention mask
```

A lot of it does nothing (`reshape` to the same shape, `+ 0`) or recomputes on every call something that never changes (the mask). That is normal for traced code and is exactly what chapter 6 cleans up.

## 4.9 Where in the code

| File | Contents |
|---|---|
| `src/graph/ir.h`, `ir.cpp` | types, operators, `Attrs`, `Constant`, `Node`, `Graph`, `infer_type`, `broadcast_shapes` |
| `src/graph/printer.cpp` | `print_graph`, `print_node`, `summarize_graph` |
| `src/graph/verifier.cpp` | `verify_graph`, `check_graph` |
| `src/frontend/importer.h`, `importer.cpp` | `import_graph_file`, the lowering table |
| `src/support/json.h`, `json.cpp` | the JSON reader |
| `tests/test_ir.cpp`, `test_importer.cpp`, `test_json.cpp` | tests for each |

## 4.10 Further reading

- Cytron et al., "Efficiently Computing Static Single Assignment Form and the Control Dependence Graph" (1991). The origin of SSA. This project's SSA is the trivial case, since there is no control flow, but the reason SSA is used everywhere is in the introduction.
- Cooper and Torczon, *Engineering a Compiler*, chapter 4 ("Intermediate Representations").
- Lattner et al., "MLIR: Scaling Compiler Infrastructure for Domain Specific Computation" (2021). How a production system organizes many IR levels; useful for seeing where this project's two levels sit.
- Crafting Interpreters (Nystrom), the chapter "Parsing Expressions", for recursive-descent parsing as used in the JSON reader.
