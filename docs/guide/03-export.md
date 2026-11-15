# 3. Getting the model out of PyTorch

The compiler is written in C++, and the model lives in Python as a `torch.nn.Module`. This chapter covers the bridge: three short Python files under `export/` that turn the model into two files the C++ side can read, and save the reference answers everything is later checked against.

## 3.1 Pinning down "the model" (`export/model.py`)

The Hugging Face `GPT2LMHeadModel` returns a rich object and can return cached state for fast generation. A compiler wants a pure function: one tensor in, one tensor out. So the model is wrapped:

```python
class GPT2Logits(torch.nn.Module):
    def forward(self, input_ids):
        return self.gpt2(input_ids=input_ids, use_cache=False).logits
```

`load_model()` returns this in `.eval()` mode (which turns off dropout, a training-only randomization).

The same file defines the sequence lengths `(16, 64, 256, 1024)` and `make_input_ids(seq_len)`, which produces fixed pseudo-random token ids from a seed. Everything in the project that needs the model or an input imports from this file: the exporter, the golden outputs, and every baseline in the benchmark. That is what guarantees all systems run the same network on the same input.

## 3.2 Tracing: `torch.export`

`torch.export.export(model, (ids,))` runs the model's Python code once with an example input and records every tensor operation that happens. The result is an `ExportedProgram`: a flat list of operations with no Python control flow left. This is **tracing**.

Two properties of the result matter:

- **Shapes are fixed** to those of the example input. That is exactly what this project wants (one compile per `T`).
- **Parameters are "lifted".** The weights become extra inputs to the traced graph, and a separate table (the *graph signature*) records which input is which parameter.

The operations recorded are **ATen** operators. ATen is PyTorch's C++ tensor library; every `torch.something` call ends up as one or more ATen operators such as `aten.addmm.default` or `aten.view.default`. The suffix (`default`, `Tensor`, `Scalar`) is the **overload**: `aten.mul.Tensor` multiplies two tensors, `aten.mul.Scalar` multiplies a tensor by a Python number.

## 3.3 Decomposition

As traced, the graph uses 31 distinct operators, some of them high-level (`layer_norm`, `dropout`, `transpose`, `masked_fill`). Calling

```python
core = exported.run_decompositions()
```

rewrites those into a smaller standard set called **core ATen**. After it the graph has 852 nodes using 37 distinct operators. `docs/export_ops.md` lists them with counts; the most common are

| Operator | Count | What it is |
|---|---:|---|
| `view` | 220 | same data, new shape |
| `permute` | 85 | reorder axes |
| `add.Tensor` | 64 | |
| `getitem` | 61 | pick one result out of a tuple |
| `expand` | 49 | broadcast to a larger shape |
| `addmm` | 48 | `bias + a @ b`, the dense layers |
| `mul.Tensor` | 48 | |
| `native_layer_norm` | 25 | |
| `bmm` | 24 | batched matrix product (attention) |
| `mm` | 1 | plain matrix product (the output projection) |

Why decompose? A smaller operator set means less for the importer to implement. The price is that some structure is lost: GELU arrives as nine primitive nodes, and the causal mask as about forty. Chapter 6 puts that structure back where it helps.

The exporter immediately re-runs the decomposed program and compares it with the original model, refusing to continue if any logit differs by more than 0.001. The observed difference is about 0.0001, which is ordinary floating-point reordering (section 9.7).

## 3.4 Golden outputs

For each `T`, before exporting, the script runs the original model and saves

```
artifacts/golden/T16_ids.npy       the input, int64 [1,16]
artifacts/golden/T16_logits.npy    PyTorch's answer, float32 [1,16,50257]
```

`.npy` is NumPy's simple array file format (a short text header, then raw bytes). These files are the **oracle**: the definition of the right answer. The C++ side has its own small `.npy` reader (`src/support/files.cpp`).

## 3.5 The serializer (`export/serialize.py`)

This file writes what `torch.export` produced to disk. Its governing rule: **it does not interpret anything.** It records operator names and arguments exactly as PyTorch gave them. Every decision about what an operator *means* is made in C++, where it can be tested. The Python is only a transcription.

### The weights file

`artifacts/graphs/gpt2_weights.bin`: every parameter tensor's raw bytes (little-endian 32-bit floats), one after another. 497 759 232 bytes.

- Each tensor starts on a **64-byte boundary** (padding bytes are inserted). 64 bytes is one cache line (chapter 9), so a tensor never shares a cache line with its neighbour.
- **Tied weights are stored once.** The token embedding and output projection are the same tensor under two parameter names. The serializer notices that two parameters share a data pointer and gives both the same offset.
- The file is written once and shared by all four sequence lengths. The script checks that every `T` produces the identical parameter table.

### The graph file

`artifacts/graphs/gpt2_T16.json` (one per `T`). Top-level fields:

```json
{
  "format": 1,
  "producer": "torch 2.15.0",
  "seq_len": 16,
  "weights_file": "gpt2_weights.bin",
  "weights_bytes": 497759232,
  "inputs":  [ {"name": "input_ids", "dtype": "int64", "shape": [1, 16]} ],
  "params":  [ {"name": "p_gpt2_transformer_wte_weight", "dtype": "float32",
                "shape": [50257, 768], "offset": 0, "nbytes": 154389504}, ... ],
  "nodes":   [ ... ],
  "outputs": [ "mm" ]
}
```

One entry of `nodes`, the first layer norm:

```json
{"name": "native_layer_norm",
 "op": "aten.native_layer_norm.default",
 "args": [ {"ref": "add_1"}, [768],
           {"ref": "p_gpt2_transformer_h_0_ln_1_weight"},
           {"ref": "p_gpt2_transformer_h_0_ln_1_bias"}, 1e-05 ],
 "kwargs": {},
 "outputs": [ {"dtype": "float32", "shape": [1, 16, 768]},
              {"dtype": "float32", "shape": [1, 16, 1]},
              {"dtype": "float32", "shape": [1, 16, 1]} ]}
```

- `args` are positional arguments, `kwargs` keyword arguments.
- `outputs` is the list of result types *PyTorch* computed. This node returns a tuple of three tensors. The importer will derive these types itself and compare (section 4.6).

Arguments are Python objects and JSON has few types, so a small encoding is used:

| Python value | JSON |
|---|---|
| another node's result | `{"ref": "add_1"}` |
| a `torch.dtype` | `{"dtype": "float32"}` |
| `inf`, `-inf`, `nan` | `{"float": "-inf"}` (JSON has no spelling for these) |
| device, memory layout | `{"torch": "cpu"}` (recorded, ignored) |
| `None`, bool, int, float, list | themselves |

A detail that matters later: JSON distinguishes `1` from `1.0` only by how it is written. The C++ JSON reader keeps integers and floats as different kinds, because the importer has to tell "add the integer 0" from "multiply by the float 0.5".

## 3.6 Running it

```sh
.venv/bin/python export/export_gpt2.py
```

For each `T` it: builds the input, runs PyTorch and saves the golden pair, exports and decomposes, checks the export against PyTorch, writes the graph JSON (and, the first time, the weights), and finally writes `docs/export_ops.md`. The outputs live under `artifacts/`, about 760 MB, which is not committed to git.

## 3.7 Design choices in this stage

| Choice | Why |
|---|---|
| Export with `torch.export`, not ONNX | ONNX would add a second operator vocabulary between PyTorch and the compiler. ATen is the source. |
| Decompose to core ATen | 37 operators to implement instead of several hundred possible ones. |
| Python does no interpretation | All semantics live in one language, with tests. |
| Weights outside the graph, addressed by offset | Generated code never contains 498 MB of literals. The file can be memory-mapped (section 8.5). |
| One export per `T` | Static shapes (section 1.5). |
| Pin `torch` and `transformers` versions (`requirements.txt`) | The exact operators exported, and therefore the patterns chapter 6 matches, depend on the PyTorch version. |

## 3.8 Where in the code

| File | Contents |
|---|---|
| `export/model.py` | `GPT2Logits`, `load_model`, `make_input_ids`, `SEQ_LENS`, golden paths |
| `export/export_gpt2.py` | the driver: golden outputs, export, decomposition, self-check, operator table |
| `export/serialize.py` | `encode_arg`, `encode_types`, `collect_params`, `layout_weights`, `write_graph` |
| `docs/export_ops.md` | generated: operators before and after decomposition |

## 3.9 Further reading

- PyTorch documentation, "torch.export" (`pytorch.org/docs/stable/export.html`). What tracing guarantees and what it cannot handle.
- PyTorch documentation, "IRs" / "Core ATen IR". The definition of the operator set this project consumes.
- NumPy documentation, "NPY format". The file format of the golden outputs.
