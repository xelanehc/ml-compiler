"""Write an ExportedProgram to disk in a form the C++ compiler can read.

Two files come out of this:

  gpt2_weights.bin   every parameter tensor, raw little-endian bytes, back to
                     back, each one starting on a 64-byte boundary
  gpt2_T<T>.json     the graph for one sequence length: inputs, parameters
                     (name, dtype, shape, offset into the weights file), the
                     nodes in execution order, and the outputs

This file deliberately does not understand the graph. It writes down what
torch.export produced, operator names and arguments as they are, and leaves
every decision about what an operator *means* to the C++ frontend. The only
translation done here is turning Python objects into JSON:

  another node's result   {"ref": "<node name>"}
  torch.dtype             {"dtype": "float32"}
  inf / -inf / nan        {"float": "-inf"}     (JSON has no spelling for these)
  device, layout, ...     {"torch": "<repr>"}   (kept only for the record)
  None, bool, int, float, list                  as themselves
"""

import json
import math
import sys

import numpy as np
import torch
from torch.export.graph_signature import InputKind

FORMAT_VERSION = 1
WEIGHT_ALIGN = 64  # bytes; one cache line, and enough for any SIMD load we will emit

DTYPE_NAMES = {torch.float32: "float32", torch.int64: "int64", torch.bool: "bool"}


def dtype_name(dtype):
    if dtype not in DTYPE_NAMES:
        raise SystemExit(f"exporter: dtype {dtype} is not supported by the compiler")
    return DTYPE_NAMES[dtype]


def encode_arg(arg):
    """Turn one fx argument into JSON-safe data. See the module docstring."""
    if isinstance(arg, torch.fx.Node):
        return {"ref": arg.name}
    if arg is None or isinstance(arg, (bool, int, str)):
        return arg
    if isinstance(arg, float):
        if math.isfinite(arg):
            return arg
        return {"float": "nan" if math.isnan(arg) else ("inf" if arg > 0 else "-inf")}
    if isinstance(arg, (list, tuple)):
        return [encode_arg(a) for a in arg]
    if isinstance(arg, torch.dtype):
        return {"dtype": dtype_name(arg)}
    if isinstance(arg, (torch.device, torch.layout, torch.memory_format)):
        return {"torch": str(arg)}
    raise SystemExit(f"exporter: cannot serialize argument {arg!r} of type {type(arg).__name__}")


def encode_types(val):
    """The static type(s) torch.export recorded for a node's result.

    Always a list: one entry for a tensor, several for a tuple, none for a
    node that returns nothing. The C++ importer re-derives every shape itself
    and compares against these, which checks our shape inference against
    PyTorch's on every node of the real model.
    """
    if val is None:
        return []
    if isinstance(val, torch.Tensor):
        return [{"dtype": dtype_name(val.dtype), "shape": list(val.shape)}]
    if isinstance(val, (list, tuple)):
        return [t for v in val for t in encode_types(v)]
    raise SystemExit(f"exporter: unexpected node value {type(val).__name__}")


def op_name(target):
    """'aten.view.default' for ATen operators, 'getitem' for tuple indexing."""
    if isinstance(target, torch._ops.OpOverload):
        return str(target)
    return getattr(target, "__name__", str(target))


def collect_params(exported):
    """Map each parameter placeholder to its tensor.

    torch.export "lifts" parameters: they become extra graph inputs, and the
    graph signature records which input is which parameter. GPT-2 ties the
    token embedding to the output projection, so two different placeholders
    can name the same storage; we notice that by data pointer and store it once.
    """
    params = []
    for spec in exported.graph_signature.input_specs:
        if spec.kind == InputKind.USER_INPUT:
            continue
        if spec.kind in (InputKind.PARAMETER, InputKind.BUFFER) and spec.target in exported.state_dict:
            tensor = exported.state_dict[spec.target]
        elif spec.target in exported.constants:
            tensor = exported.constants[spec.target]
        else:
            raise SystemExit(f"exporter: cannot find a tensor for input {spec.arg.name} ({spec.kind})")
        params.append((spec.arg.name, tensor.detach()))
    return params


def layout_weights(params):
    """Assign each distinct tensor an aligned offset. Returns (table, blobs, total_bytes)."""
    table, blobs, seen, offset = [], [], {}, 0
    for name, tensor in params:
        key = (tensor.data_ptr(), tuple(tensor.shape), tensor.dtype)
        if key not in seen:
            data = np.ascontiguousarray(tensor.numpy())
            offset = (offset + WEIGHT_ALIGN - 1) // WEIGHT_ALIGN * WEIGHT_ALIGN
            seen[key] = offset
            blobs.append((offset, data))
            offset += data.nbytes
        table.append(
            {
                "name": name,
                "dtype": dtype_name(tensor.dtype),
                "shape": list(tensor.shape),
                "offset": seen[key],
                "nbytes": tensor.numel() * tensor.element_size(),
            }
        )
    return table, blobs, offset


def write_weights(path, blobs, total_bytes):
    with open(path, "wb") as f:
        for offset, data in blobs:
            f.write(b"\0" * (offset - f.tell()))  # alignment padding
            f.write(data.tobytes())
        assert f.tell() == total_bytes


def write_graph(exported, seq_len, graph_path, weights_path, write_weights_file):
    """Serialize one exported program. Returns the parameter table."""
    if sys.byteorder != "little":
        raise SystemExit("exporter: the weights format is little-endian only")

    table, blobs, total_bytes = layout_weights(collect_params(exported))
    if write_weights_file:
        write_weights(weights_path, blobs, total_bytes)

    param_names = {p["name"] for p in table}
    inputs, nodes, outputs = [], [], []
    for node in exported.graph.nodes:
        if node.op == "placeholder":
            if node.name not in param_names:
                (ty,) = encode_types(node.meta["val"])
                inputs.append({"name": node.name, **ty})
        elif node.op == "call_function":
            nodes.append(
                {
                    "name": node.name,
                    "op": op_name(node.target),
                    "args": encode_arg(node.args),
                    "kwargs": {k: encode_arg(v) for k, v in node.kwargs.items()},
                    "outputs": encode_types(node.meta.get("val")),
                }
            )
        elif node.op == "output":
            outputs = [a.name for a in node.args[0]]
        else:
            raise SystemExit(f"exporter: unexpected fx node kind {node.op!r}")

    doc = {
        "format": FORMAT_VERSION,
        "producer": f"torch {torch.__version__}",
        "seq_len": seq_len,
        "weights_file": weights_path.name,
        "weights_bytes": total_bytes,
        "inputs": inputs,
        "params": table,
        "nodes": nodes,
        "outputs": outputs,
    }
    # allow_nan=False makes json.dumps raise instead of writing a bare
    # "Infinity", which is not JSON and which our parser would reject anyway.
    graph_path.write_text(json.dumps(doc, indent=1, allow_nan=False))
    return table
