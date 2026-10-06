"""Lossless TP slicing of E2M1 experts and compressed UE8M0 scales."""

from __future__ import annotations

import json
from contextlib import ExitStack
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from safetensors import safe_open

from b12x._lib.quant.x4t_scales import make_x4t_scale_batch
from b12x.moe.fused_moe.weights import X4TWeights

SCHEMA = "trellis-exact-mxfp4-checkpoint/1"
CODEC = "adjacent-pair-fixed-stream-u24-exceptions/1"
POSITION_MASK = (1 << 24) - 1


@lru_cache(maxsize=4)
def checkpoint_contract(root: str) -> dict:
    """Validate the container identity without reading its weight payloads."""
    directory = Path(root)
    manifest = json.loads((directory / "manifest.json").read_text())
    contract = json.loads((directory / "build-contract.json").read_text())
    for key, value in (
        ("schema", SCHEMA),
        ("codec", CODEC),
    ):
        if manifest.get(key) != value or contract.get(key) != value:
            raise ValueError(f"X4T requires {key}={value!r}")
    family = contract.get("family")
    if family not in ("deepseek_v41", "kimi_k3") or manifest.get("family") != family:
        raise ValueError("X4T requires a matching DS4.1 or Kimi-K3 family")
    names = contract["source_names"]
    files = {item["file"] for item in manifest["shards"]}
    for name in set(names.values()) | files:
        if Path(name).name != name or not name.endswith(".safetensors"):
            raise ValueError(
                "X4T shard names must be checkpoint-local safetensors files"
            )
    if set(names.values()) != files:
        raise ValueError("X4T manifest and source-name index disagree")
    return contract


def slice_scale_plane(fixed, exceptions, rows, columns, row_slice, column_slice):
    """Slice compressed bytes without floating-point reconstruction or fitting."""
    r0, r1 = row_slice
    c0, c1 = column_slice
    if not (0 <= r0 < r1 <= rows and r0 % 16 == r1 % 16 == 0):
        raise ValueError("X4T row slices must contain complete 16-row slabs")
    if not 0 <= c0 < c1 <= columns:
        raise ValueError("X4T column slice is outside the scale plane")
    if fixed.dtype != torch.uint8 or exceptions.dtype != torch.uint32:
        raise TypeError("X4T requires uint8 fixed bytes and uint32 exceptions")
    selectors = (columns + 7) // 8
    stream = fixed.numpy().reshape(rows // 16, 16 * (1 + selectors))
    bases = stream[:, :16]
    if (bases > 254).any():
        raise ValueError("X4T palette bases must be in 0..254")
    bits = np.unpackbits(
        stream[:, 16:].reshape(rows, selectors), axis=1, bitorder="little"
    )
    if bits[:, columns:].any():
        raise ValueError("X4T unused selector bits must be zero")
    selected = np.packbits(bits[r0:r1, c0:c1], axis=1, bitorder="little")
    result = np.concatenate(
        (bases[r0 // 16 : r1 // 16], selected.reshape((r1 - r0) // 16, -1)), 1
    )
    words = exceptions.numpy().reshape(-1)
    positions = words & POSITION_MASK
    if len(words) and (
        positions[-1] >= rows * columns or (positions[1:] <= positions[:-1]).any()
    ):
        raise ValueError("X4T exception positions must be unique, sorted and in range")
    rr, cc = positions // columns, positions % columns
    keep = (rr >= r0) & (rr < r1) & (cc >= c0) & (cc < c1)
    positions = (rr[keep] - r0) * (c1 - c0) + cc[keep] - c0
    words = (words[keep] & np.uint32(0xFF000000)) | positions
    return torch.from_numpy(result.copy()), torch.from_numpy(words.astype(np.uint32))


def read_exact_mxfp4_layer(
    root,
    layer_index,
    *,
    num_experts,
    hidden_size,
    intermediate_size,
    tp_rank,
    tp_size,
    device,
    w13_scale_scratch,
    w2_scale_scratch,
):
    """Load a gate/up-ordered TP extent; expanded scales remain caller-owned."""
    root = Path(root)
    contract = checkpoint_contract(str(root.resolve()))
    family = contract["family"]
    expected_geometry = {
        "deepseek_v41": (384, 5120, 2304),
        "kimi_k3": (896, 3584, 3072),
    }[family]
    supported_tp = (1, 2, 4, 8) if family == "deepseek_v41" else (1, 2, 4, 8, 12, 16)
    if (num_experts, hidden_size, intermediate_size) != expected_geometry:
        raise ValueError(f"X4T {family} requires expert geometry {expected_geometry}")
    if tp_size not in supported_tp or not 0 <= tp_rank < tp_size:
        raise ValueError(f"X4T {family} supports TP {supported_tp} with a valid rank")
    names = contract["source_names"]
    local = intermediate_size // tp_size
    first, last = tp_rank * local, (tp_rank + 1) * local
    w13 = torch.empty(
        (num_experts, 2 * local, hidden_size // 2), dtype=torch.uint8, device="cpu"
    )
    w2 = torch.empty(
        (num_experts, hidden_size, local // 2), dtype=torch.uint8, device="cpu"
    )
    fixed13, fixed2, exceptions13, exceptions2 = [], [], [], []
    with ExitStack() as stack:
        handles = {}

        def handle(name):
            filename = names[name]
            if filename not in handles:
                handles[filename] = stack.enter_context(
                    safe_open(root / "tensors" / filename, framework="pt", device="cpu")
                )
            return handles[filename]

        for expert in range(num_experts):
            f13, e13 = [], []
            for matrix, projection in enumerate(("w1", "w3", "w2")):
                if family == "kimi_k3":
                    stem = (
                        f"language_model.model.layers.{layer_index}."
                        f"block_sparse_moe.experts.{expert}.{projection}"
                    )
                    name, scale = stem + ".weight_packed", stem + ".weight_scale"
                else:
                    stem = f"layers.{layer_index}.ffn.experts.{expert}.{projection}"
                    name, scale = stem + ".weight", stem + ".scale"
                view = handle(name).get_slice(name)
                expected = (
                    [intermediate_size, hidden_size // 2]
                    if matrix < 2
                    else [hidden_size, intermediate_size // 2]
                )
                if view.get_shape() != expected or view.get_dtype() not in ("I8", "U8"):
                    raise ValueError(f"X4T nibble geometry/dtype mismatch: {name}")
                if matrix < 2:
                    w13[expert, matrix * local : (matrix + 1) * local].copy_(
                        view[first:last, :].view(torch.uint8)
                    )
                    rows, columns = intermediate_size, hidden_size // 32
                    row_slice, column_slice = (first, last), (0, columns)
                else:
                    w2[expert].copy_(view[:, first // 2 : last // 2].view(torch.uint8))
                    rows, columns = hidden_size, intermediate_size // 32
                    row_slice, column_slice = (0, rows), (first // 32, last // 32)
                reader = handle(scale)
                fixed, exceptions = slice_scale_plane(
                    reader.get_tensor(scale + ".exact_mxfp4_fixed"),
                    reader.get_tensor(scale + ".exact_mxfp4_exceptions"),
                    rows,
                    columns,
                    row_slice,
                    column_slice,
                )
                if matrix < 2:
                    f13.append(fixed)
                    if matrix:
                        words = exceptions.numpy().copy()
                        words += np.uint32(local * columns)
                        exceptions = torch.from_numpy(words)
                    e13.append(exceptions)
                else:
                    fixed2.append(fixed)
                    exceptions2.append(exceptions)
            fixed13.append(torch.cat(f13))
            exceptions13.append(torch.cat(e13))
    batch13 = make_x4t_scale_batch(
        fixed13,
        exceptions13,
        rows=2 * local,
        columns=hidden_size // 32,
        device=device,
        exception_task_rows=64,
    )
    batch2 = make_x4t_scale_batch(
        fixed2,
        exceptions2,
        rows=hidden_size,
        columns=local // 32,
        device=device,
        exception_task_rows=64,
    )
    return X4TWeights(
        w13=w13.to(device),
        w2=w2.to(device),
        w13_scales=batch13,
        w2_scales=batch2,
        w13_scale_scratch=w13_scale_scratch,
        w2_scale_scratch=w2_scale_scratch,
    )
