# 6. Graph optimizations

A **pass** is a function that takes a graph and returns a better graph that computes the same thing. This chapter covers the six passes in `src/graph/passes.cpp`, in the order they run, and the single mechanism they are all built on.

Run `build/release/mlc opt artifacts/graphs/gpt2_T16.json` to see them act on the real model:

```
pass        before   after  removed
dce           1062    1062        0
simplify      1062     951      111
fold           951     902       49
cse            902     780      122
gelu           780     691       89
attention      691     615       76
layout         615     686      -71
```

## 6.1 Why rewriting is safe here

Chapter 4 set this up. Values are defined once and never change (SSA), and nodes are pure. So the statement "node `v` computes the same tensor as value `w`" has an unambiguous meaning, and if it is true, every user of `v` can be pointed at `w` instead with no other consequences. Nothing else can observe the difference. All six passes are instances of that one move.

## 6.2 The rewriting engine

```cpp
using RewriteRule = std::function<ValueId(Graph& out, const Node& n, const Constant* constant)>;
Graph rewrite(const Graph& g, const RewriteRule& rule);
```

`rewrite` never edits a graph in place. It walks the old graph in order and builds a new one:

1. Take the next old node. Translate its inputs to the corresponding values of the new graph (a table `remap` records, for each old node, which new value stands for it).
2. Call the rule with the new graph and this translated node.
3. The rule either returns a value of the new graph that should stand for this node, or returns "no value", in which case the node is copied unchanged.
4. Record the answer in `remap`.

Two consequences of this design are worth understanding.

**A rule sees operands that have already been rewritten.** By the time the rule looks at a node, its inputs are values in the *new* graph, already simplified. So a chain collapses in one sweep: for `reshape(reshape(reshape(x)))`, when the outermost reshape is visited its operand is already a single reshape of `x`.

**A rule may add new nodes and may leave garbage behind.** If a rule replaces a node by something else, the nodes that only fed the old node are still copied into the new graph, now with no users. The engine does not clean them up. Dead code elimination does.

The engine also checks that the replacement has exactly the type of the node it replaces, and throws if not. A rule cannot accidentally change a shape.

## 6.3 Dead code elimination (DCE)

A node is **dead** if no output depends on it. `eliminate_dead_code` computes liveness backwards: mark the graph outputs live; walk the nodes from last to first; when a live node is reached, mark its inputs live. Because every user comes after the values it uses, one backward walk is enough. Then copy only the live nodes into a new graph.

Graph inputs are always kept, even if unused: they are part of the function's signature.

DCE runs after every other pass. That is why the table above charges each pass for everything it made removable: the pass rewires, DCE sweeps.

## 6.4 `simplify`: local cleanups (removes 111 nodes)

Each rule looks at one node and at most its immediate operands. No analysis is needed.

| Pattern | Replacement | In GPT-2 |
|---|---|---|
| `Broadcast` to the shape the operand already has | the operand | all 49 `expand` nodes |
| `Reshape` to the same shape | the operand | |
| `Reshape(Reshape(x))` | one `Reshape(x)`, or `x` if the final shape is `x`'s | view chains |
| `Permute` with the identity permutation | the operand | |
| `Permute(Permute(x))` | one composed `Permute(x)`, or `x` if they cancel | the head merge is three permutes in a row |
| `Slice` of the whole axis | the operand | |
| `Concat` of one input | the operand | |
| `x + 0`, `x * 1` (when the result has `x`'s type) | `x` | `positions + 0` |
| `Not(Not(x))` | `x` | |

**Composing permutations.** The outer permute says "my axis `i` is the middle value's axis `outer[i]`"; the inner says "my axis `k` is `x`'s axis `inner[k]`". So the result's axis `i` is `x`'s axis `inner[outer[i]]`. That one line of reasoning is the whole rule.

## 6.5 `fold`: constant folding (removes 49 nodes)

If all of a node's operands are known at compile time, compute the node now and replace it by a constant. Because the engine visits nodes in order and operands are already folded, a whole constant subgraph collapses from its leaves upward, one node at a time.

How a node is evaluated: build a tiny graph containing only that node and its constant operands, and run the reference interpreter on it. No second implementation of the operators is needed.

What it folds in GPT-2: the position indices, and the roughly forty nodes that build the causal attention mask from `arange`, `cumsum`, and comparisons.

Three limits, each with a reason:

- **External weights are never folded.** Folding `permute(weight)` would require reading 498 MB of weights into the compiler and writing the result into the generated code. The rule from chapter 3 stands: weight bytes stay in the weights file. (Chapter 7's *staging* handles these cases differently.)
- **`Broadcast` is never folded.** It would turn a 4-byte scalar into a full tensor, and destroy the information that the tensor is uniform.
- **Results over 1 MB are not folded.** At T=1024 the causal mask is `[1,1,1024,1024]` floats, 4 MB. Leaving it unfolded is fine: the `attention` pass below recognizes it anyway and then it becomes dead.

## 6.6 `cse`: common subexpression elimination (removes 122 nodes)

Two nodes with the same operator, the same operands, and the same attributes compute the same value. Keep the first, point users of the second at it.

The implementation is **value numbering**. For each node build a key: operator, result type, operand ids, attributes (for a constant: its bytes or its weights offset). Keep a hash map from key to the value already created for it. If the key has been seen, return that value; otherwise copy the node and record it.

The subtle part is that operand ids in the key are ids in the *new* graph, which have already been deduplicated. So two identical subtrees get identical keys from the bottom up, without ever comparing trees.

In GPT-2 almost everything it merges is duplicate scalar constants (every layer has its own `0.5`, its own `3.0`, its own copy of the attention scale) and, before `fold`, per-layer copies of the mask. It removes 122 nodes but only 11 kernels' worth of actual work, since constants never had code.

**A pitfall worth remembering.** Whenever a new attribute is added to `Attrs`, it must be added to the CSE key. Otherwise two nodes that differ only in that attribute would be merged. `causal` and `trans_b` were both added to the key when they were introduced.

## 6.7 `gelu`: recognizing GELU (removes 89 nodes)

The export spells GELU (section 2.3) as nine nodes per layer:

```
half  = mul(h, 0.5)
cube  = pow(h, 3.0)
inner = mul(add(h, mul(cube, 0.044715)), 0.7978845608)
out   = mul(half, add(tanh(inner), 1.0))
```

Each is a full pass over a `[1, T, 3072]` tensor with its own output buffer. The pass replaces the group by one `Gelu(h)` node, which the loop generator turns into a single pass.

**How pattern matching works.** The rule fires on the *last* node of the pattern, the final `mul`, and walks backwards through its operands checking each step:

1. Is one operand `mul(h, 0.5)`? Remember `h`.
2. Is the other `add(t, 1.0)`, with `t` a `tanh`?
3. Is the `tanh`'s operand `mul(s, 0.7978845608)`?
4. Is `s` an `add` with `h` as one operand? Call the other `c`.
5. Is `c` equal to `mul(cube, 0.044715)`, with `cube = pow(h, 3.0)`?
6. Is it the *same* `h` everywhere?

If every check passes, emit `Gelu(h)`. The eight nodes above the final `mul` now have no users and DCE removes them.

Details that make it robust:

- Multiplication and addition commute, so each step tries both operand orders.
- Constants are compared with a small tolerance, because `0.7978845608028654` arrives rounded to a 32-bit float.
- Lookalikes are rejected: a wrong constant, or a cube of a different tensor, fails a check and the nodes are left alone. There are tests for exactly these.

Matching is done on the compiler's own IR after `simplify` and `cse`, not on raw ATen. The earlier passes put the graph in a more regular form, which makes the pattern less fragile.

## 6.8 `attention`: the causal softmax (removes 76 nodes)

Each attention layer exports as:

```
s    = add(scores, mask)                    mask is 0 where j <= i, -inf where j > i
p    = softmax(s, last axis)
dead = not(reduce_any(not(eq(s, -inf))))    "is this whole row -inf?"
out  = where(dead, zeros, p)                such a row would be NaN; make it 0
```

Two observations make all of this redundant:

1. **The guard can never fire.** With a causal mask, row `i` always keeps column `i` (a position can attend to itself). So no row is entirely masked and `dead` is always false. (This assumes the scores themselves are finite.)
2. **Adding `-inf` before `exp` is a roundabout way of leaving columns out.** `exp(-inf) = 0`, so the masked columns contribute nothing to the sum and get probability 0.

So the whole group equals "softmax of row `i` over columns `0..i` only, with zeros after". The pass replaces it by one `Softmax` node with the attribute `causal=true`, applied directly to the raw `scores`. That does half the exponentials and needs no mask in memory at all.

**Recognizing the mask by value.** The pass does not try to recognize the forty-node computation that produces the mask. It asks the function `evaluate_constant` to compute the mask at compile time (by interpreting the part of the graph it depends on), and then checks the resulting numbers: is it `[1, ..., 1, T, T]` with 0 on and below the diagonal and `-inf` above? Whatever computes the mask, if it evaluates to that pattern, the rule fires. This is why it works at T=1024 even though `fold` declined to fold the 4 MB mask.

## 6.9 `layout`: transposed matrix products (adds 71 nodes)

This pass was added in M4 and only makes sense together with chapter 10, so here is just what it does.

Every matrix product `a @ b` is rewritten to take its second operand already transposed. The node gets the attribute `trans_b`, and its second input becomes `bt = transpose(b)`, with shape `[n, k]` instead of `[k, n]`. The value computed is the same.

The reason: element `(i, j)` of the product is then the dot product of row `i` of `a` and row `j` of `bt`. Both are contiguous runs of memory, which is the form the fast kernel wants.

Naively this inserts a transpose in front of every product. The pass builds each transposed operand with a helper that looks for cheaper ways:

- **The operand is already a transpose of something:** use that something. This cancels the transpose on the output projection (it was `transpose(embedding table)` to begin with) and on `q @ transpose(k)` in attention.
- **The operand is `x * scalar`:** then its transpose is `transpose(x) * scalar`. Moving the transpose inward lets it meet, and cancel, a transpose applied before the scaling. GPT-2 scales `k` after transposing it, so this rule is what makes the attention case cancel.
- **Otherwise:** insert a real `Permute`.

What is left: 48 transposes of dense-layer weights (which chapter 7's staging computes once at load time, not per call) and 12 transposes of the attention value matrix (small, per call). The node count goes *up* by 71, which is fine; the count was never the goal.

## 6.10 The pipeline and its switches

`optimize(g, options, stats)` runs DCE, then each enabled pass followed by DCE and the verifier, recording node counts before and after. If the verifier fails after a pass, the error names the pass.

Every pass can be turned off by name: `mlc compile ... --disable gelu,cse`. `--disable all` turns everything off. This is how the ablation tables (chapter 12) are produced: build the model with one thing off, measure, compare.

## 6.11 What the graph passes bought

Measured, not assumed (chapter 12 has the tables):

- **Graph size:** 1062 nodes to 615 (before `layout`). Kernels: 541 to 258. The generated C++ compiles in a third of the time.
- **Memory:** together with memory planning (chapter 7), the arena at T=1024 went from 6453 MB to 158 MB.
- **Speed: nothing measurable.** Matrix products are over 96% of the run time and these passes do not touch them. The passes removed about a quarter of the remaining work, and a quarter of 3% does not show. This was a surprise at the time and is one of the honest findings of the project: at this model's shape, graph-level optimization is about memory and cleanliness, not time.

## 6.12 Fusions that were planned and not built

STEERING.md originally listed more. Each was dropped for a recorded reason:

- **Matrix product + bias into `Gemm`:** not needed; ATen already exports `addmm`.
- **General fusion of elementwise chains:** after `gelu` and `attention`, the only elementwise operators left are 25 additions and 24 scalings. There is no chain left to fuse.
- **Residual add into the producing kernel:** the add already runs in place (chapter 7) and all 25 together are 0.1% of the run.

## 6.13 Where in the code

| File | Contents |
|---|---|
| `src/graph/passes.h` | declarations and the comment explaining each pass; `OptOptions` |
| `src/graph/passes.cpp` | `rewrite`, `copy_node`, `evaluate_constant`, the pattern helpers (`as`, `is_scalar`, `op_with_scalar`), every pass, `optimize` |
| `tests/test_passes.cpp` | each pass on small graphs, including the lookalike cases, and the whole pipeline on a toy transformer |

## 6.14 Further reading

- Cooper and Torczon, *Engineering a Compiler*, chapter 8 ("Introduction to Optimization"): local value numbering, dead code elimination, constant folding. The classical versions of sections 6.3 to 6.6.
- Muchnick, *Advanced Compiler Design and Implementation*, chapters 12 and 13, for the same topics in more depth.
- Click, "Global Code Motion / Global Value Numbering" (PLDI 1995). Value numbering on an SSA graph.
- Jia et al., "TASO: Optimizing Deep Learning Computation with Automatic Generation of Graph Substitutions" (SOSP 2019). What graph rewriting looks like when the rules are discovered automatically rather than written by hand.
