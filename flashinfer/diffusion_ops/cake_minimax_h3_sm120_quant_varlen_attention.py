"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import functools
import math
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_quant_varlen_attention import (
    gen_minimax_h3_sm120_quant_varlen_attention_module,
)
from ..utils import register_custom_op, register_fake_op

MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_NUM_HEADS = 56
# 128 query rows per unit, 128 keys per K/V tile (absolute-token aligned K blocks).
_BLOCK_M = 128
_BLOCK_N = 128
_MAX_HEADS = 1 << 15
_MAX_SEGMENT_TILES = 1 << 16
# Fixed per-unit overhead (Q TMA, pipeline fill, epilogue) in K/V-tile units for the LPT placement.
_UNIT_OVERHEAD_TILES = 2
_PARTIAL_FLOATS = 2 * MINIMAX_H3_HEAD_DIM


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_quant_varlen_attention_module().build_and_load()


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def normalize_cu_seqlens(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> Tuple[int, ...]:
    """Validate ``cu_seqlens`` and return its host copy as a tuple of ints.

    When ``cu_seqlens`` is a CUDA tensor and no ``cu_seqlens_host`` is given, the bounds are
    copied to the host once (a synchronizing ``tolist``); pass ``cu_seqlens_host`` to avoid it.
    """

    if isinstance(cu_seqlens, torch.Tensor):
        if cu_seqlens.ndim != 1 or cu_seqlens.dtype != torch.int32:
            raise ValueError("cu_seqlens must be a one-dimensional int32 tensor")
        if cu_seqlens_host is None:
            cu_seqlens_host = cu_seqlens.tolist()
        elif len(cu_seqlens_host) != int(cu_seqlens.shape[0]):
            raise ValueError("cu_seqlens_host must have the same length as cu_seqlens")
    elif cu_seqlens_host is None:
        cu_seqlens_host = cu_seqlens
    bounds = tuple(int(v) for v in cu_seqlens_host)
    if len(bounds) < 2 or bounds[0] != 0:
        raise ValueError("cu_seqlens must have at least two entries and start at zero")
    if any(b < a for a, b in zip(bounds, bounds[1:], strict=False)):
        raise ValueError("cu_seqlens must be non-decreasing")
    return bounds


def assign_unit_slots(unit_costs: List[int], num_ctas: int) -> List[int]:
    """Longest-processing-time-first placement of units into persistent-grid slots.

    Slot ``k * G + i`` is the ``k``-th unit of CTA ``i`` (``G = min(num_ctas, units)``).  The
    partial tail round receives the cheapest units; within every full round the CTAs that also
    own a tail unit receive that round's cheapest units.  Equal costs keep enumeration order so
    a segment's units stay head-major (the resident CTAs share one head's K/V through L2).
    """

    total = len(unit_costs)
    if total == 0:
        return []
    G = min(num_ctas, total)
    order = sorted(range(total), key=lambda u: -unit_costs[u])
    tail = total % G
    slots = [0] * total
    full_units = order[: total - tail]
    for k in range(len(full_units) // G):
        chunk = full_units[k * G : (k + 1) * G]
        for i in range(G):
            slots[k * G + i] = chunk[G - 1 - i]
    for i, unit in enumerate(order[total - tail :] if tail else []):
        slots[(total // G) * G + i] = unit
    return slots


class MiniMaxH3VarlenPlan:
    """Segment tables and persistent-grid unit table for one ``(cu_seqlens, heads, device)``."""

    def __init__(
        self,
        bounds: Tuple[int, ...],
        num_heads: int,
        device: torch.device,
        num_ctas: int,
    ):
        self.bounds = bounds
        self.num_heads = num_heads
        self.total_tokens = bounds[-1]
        self.num_segments = len(bounds) - 1
        begins = [bounds[s] for s in range(self.num_segments)]
        lens = [bounds[s + 1] - bounds[s] for s in range(self.num_segments)]
        tile_table: List[int] = []
        seg_tile_begin: List[int] = [0]
        units: List[Tuple[int, int, int]] = []
        costs: List[int] = []
        for seg, (begin, length) in enumerate(zip(begins, lens, strict=True)):
            q_tiles = _ceil_div(length, _BLOCK_M)
            if q_tiles >= _MAX_SEGMENT_TILES:
                raise ValueError(
                    f"a segment needs fewer than {_MAX_SEGMENT_TILES} query tiles"
                )
            for t in range(q_tiles):
                tile_table.extend((seg, t))
            seg_tile_begin.append(seg_tile_begin[-1] + q_tiles)
            if length > 0:
                cost = (
                    (begin + length - 1) // _BLOCK_N
                    - begin // _BLOCK_N
                    + 1
                    + _UNIT_OVERHEAD_TILES
                )
                for head in range(num_heads):
                    for t in range(q_tiles):
                        units.append((seg, head, t))
                        costs.append(cost)
        self.num_tiles = seg_tile_begin[-1]
        self.num_units = len(units)
        self.grid = max(1, min(int(num_ctas), max(self.num_units, 1)))
        unit_table: List[int] = []
        for unit in assign_unit_slots(costs, self.grid):
            seg, head, t = units[unit]
            unit_table.extend((seg, (head << 16) | t))
        self.num_kblocks = max(1, _ceil_div(self.total_tokens, _BLOCK_N))
        self.padded_tokens = self.num_kblocks * _BLOCK_N
        i32 = lambda values: torch.tensor(values, dtype=torch.int32, device=device)  # noqa: E731
        self.cu_seqlens = i32(list(bounds))
        self.seg_begin = i32(begins or [0])
        self.seg_len = i32(lens or [0])
        self.seg_tile_begin = i32(seg_tile_begin)
        self.tile_table = i32(tile_table or [0, 0])
        self.unit_table = i32(unit_table or [0, 0])


_PLANS: Dict[Tuple, MiniMaxH3VarlenPlan] = {}
_WORKSPACES: Dict[Tuple[int, str], torch.Tensor] = {}


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


def _plan(
    bounds: Tuple[int, ...], num_heads: int, device: torch.device
) -> MiniMaxH3VarlenPlan:
    index = _device_index(device)
    num_ctas = torch.cuda.get_device_properties(index).multi_processor_count
    key = (bounds, num_heads, index, num_ctas)
    plan = _PLANS.get(key)
    if plan is None:
        if len(_PLANS) >= 256:
            _PLANS.clear()
        plan = MiniMaxH3VarlenPlan(
            bounds, num_heads, torch.device("cuda", index), num_ctas
        )
        _PLANS[key] = plan
    return plan


def _workspace(index: int, name: str, numel: int, dtype: torch.dtype) -> torch.Tensor:
    """Grow-only per-device buffer (the quantized operands and scales are rewritten every call)."""

    key = (index, name)
    buffer = _WORKSPACES.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.empty(
            max(numel, 16), dtype=dtype, device=torch.device("cuda", index)
        )
        _WORKSPACES[key] = buffer
    return buffer


def workspace_bytes(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of the per-device workspace the operator needs for ``tokens`` packed rows."""

    padded = max(1, _ceil_div(tokens, _BLOCK_N)) * _BLOCK_N
    q8 = k8 = tokens * num_heads * MINIMAX_H3_HEAD_DIM
    vt8 = num_heads * MINIMAX_H3_HEAD_DIM * padded
    scales = (
        tokens * num_heads
        + num_heads * (padded // _BLOCK_N)
        + 2 * num_segments * num_heads * MINIMAX_H3_HEAD_DIM
    )
    partials = _ceil_div(tokens, _BLOCK_M) * num_heads * _PARTIAL_FLOATS
    return q8 + k8 + vt8 + 4 * (scales + partials)


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_fp8",
    mutates_args=(
        "out",
        "q8",
        "k8",
        "vt8",
        "q_scale",
        "k_scale",
        "v_scale",
        "mean_k",
        "partials",
    ),
)
def _minimax_h3_sm120_varlen_attention_fp8_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    seg_begin: torch.Tensor,
    seg_len: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q8: torch.Tensor,
    k8: torch.Tensor,
    vt8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    attention_grid: int,
    softmax_scale: float,
) -> None:
    _get_module().minimax_h3_sm120_varlen_attention_fp8(
        q,
        k,
        v,
        cu_seqlens,
        out,
        seg_begin,
        seg_len,
        seg_tile_begin,
        tile_table,
        unit_table,
        q8,
        k8,
        vt8,
        q_scale,
        k_scale,
        v_scale,
        mean_k,
        partials,
        num_segments,
        num_tiles,
        num_units,
        attention_grid,
        softmax_scale,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_fp8")
def _minimax_h3_sm120_varlen_attention_fp8_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: torch.Tensor,
    seg_begin: torch.Tensor,
    seg_len: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q8: torch.Tensor,
    k8: torch.Tensor,
    vt8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    attention_grid: int,
    softmax_scale: float,
) -> None:
    pass


def _check_thd(name: str, tensor: torch.Tensor, tokens: int, heads: int) -> None:
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {tensor.dtype}")
    if tensor.ndim != 3 or tuple(tensor.shape) != (tokens, heads, MINIMAX_H3_HEAD_DIM):
        raise ValueError(
            f"{name} must have shape [{tokens}, {heads}, {MINIMAX_H3_HEAD_DIM}], got {tuple(tensor.shape)}"
        )
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


@flashinfer_api
def minimax_h3_sm120_varlen_attention_fp8(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    r"""FP8 (E4M3) non-causal packed-varlen self-attention for MiniMax-H3 on SM120 (GB202).

    Computes, for every segment ``[a, b)`` of ``cu_seqlens`` and every head ``h``::

        out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]^T * softmax_scale) @ v[a:b, h]

    with FP8 E4M3 tensor-core operands and an FP32 softmax: Q is quantized per token, K per
    128-key block after subtracting its segment's per-channel mean (an exact softmax shift),
    the probabilities are stored as E4M3 with a 2^8 exponent bias and V per (segment, channel).
    Both QK^T and PV run ``mma.sync.m16n8k32 kind::f8f6f4``; the output is rounded to BF16
    once.  One runtime-variable kernel set serves any ``tokens`` and any segment lengths
    (including empty segments and lengths below one tile); the quantized operands live in a
    grow-only per-device workspace (``workspace_bytes``).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Contiguous ``bfloat16`` CUDA tensors of shape ``[tokens, heads, 128]`` (packed THD
        layout; MiniMax-H3 uses 56 heads).
    cu_seqlens : torch.Tensor
        ``int32`` tensor of shape ``[segments + 1]`` on the same device with ``cu_seqlens[0] = 0``,
        non-decreasing entries and ``cu_seqlens[-1] = tokens``.
    out : Optional[torch.Tensor]
        Optional pre-allocated output of the same shape/dtype as ``q``; allocated when omitted.
    cu_seqlens_host : Optional[Sequence[int]]
        Host copy of ``cu_seqlens`` (avoids a synchronizing device-to-host copy).  The
        segment plan is cached per ``(cu_seqlens, heads, device)``.
    softmax_scale : Optional[float]
        Softmax scale; defaults to ``1 / sqrt(128)``.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[tokens, heads, 128]`` attention output.
    """

    if q.ndim != 3:
        raise ValueError(f"q must be [tokens, heads, 128], got shape {tuple(q.shape)}")
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    if not 1 <= heads < _MAX_HEADS:
        raise ValueError(f"heads must lie in [1, {_MAX_HEADS}), got {heads}")
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_thd(name, tensor, tokens, heads)
    if not cu_seqlens.is_cuda or cu_seqlens.device != q.device:
        raise ValueError(
            "cu_seqlens must be an int32 CUDA tensor on the same device as q"
        )
    bounds = normalize_cu_seqlens(cu_seqlens, cu_seqlens_host)
    if bounds[-1] != tokens:
        raise ValueError(f"cu_seqlens[-1] = {bounds[-1]} must equal tokens = {tokens}")
    if out is None:
        out = torch.empty_like(q)
    else:
        _check_thd("out", out, tokens, heads)
        if out.device != q.device:
            raise ValueError("out must be on the same device as q")
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(MINIMAX_H3_HEAD_DIM)
    plan = _plan(bounds, heads, q.device)
    if plan.num_units == 0:
        return out  # every segment is empty: nothing to write
    index = _device_index(q.device)
    hd = MINIMAX_H3_HEAD_DIM
    q8 = _workspace(index, "q8", tokens * heads * hd, torch.uint8)
    k8 = _workspace(index, "k8", tokens * heads * hd, torch.uint8)
    vt8 = _workspace(index, "vt8", heads * hd * plan.padded_tokens, torch.uint8)
    q_scale = _workspace(index, "q_scale", tokens * heads, torch.float32)
    k_scale = _workspace(index, "k_scale", heads * plan.num_kblocks, torch.float32)
    v_scale = _workspace(
        index, "v_scale", plan.num_segments * heads * hd, torch.float32
    )
    mean_k = _workspace(index, "mean_k", plan.num_segments * heads * hd, torch.float32)
    partials = _workspace(
        index,
        "partials",
        max(1, plan.num_tiles * heads * _PARTIAL_FLOATS),
        torch.float32,
    )
    _minimax_h3_sm120_varlen_attention_fp8_impl(
        q,
        k,
        v,
        plan.cu_seqlens,
        out,
        plan.seg_begin,
        plan.seg_len,
        plan.seg_tile_begin,
        plan.tile_table,
        plan.unit_table,
        q8,
        k8,
        vt8,
        q_scale,
        k_scale,
        v_scale,
        mean_k,
        partials,
        plan.num_segments,
        plan.num_tiles,
        plan.num_units,
        plan.grid,
        float(softmax_scale),
    )
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_NUM_HEADS",
    "MiniMaxH3VarlenPlan",
    "assign_unit_slots",
    "minimax_h3_sm120_varlen_attention_fp8",
    "normalize_cu_seqlens",
    "workspace_bytes",
]
