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
from typing import Optional, Sequence

import torch

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    gen_minimax_h3_sm120_nvfp4_varlen_attention_module,
)
from ..utils import register_custom_op, register_fake_op
from .cake_minimax_h3_sm120_quant_varlen_attention import (
    _BLOCK_M,
    _BLOCK_N,
    _MAX_HEADS,
    _PARTIAL_FLOATS,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    _ceil_div,
    _check_thd,
    _device_index,
    _plan,
    _workspace,
    normalize_cu_seqlens,
)

# NVFP4 operand geometry: 16 channels per UE4M3 block scale, two E2M1 codes per byte.
_FP4_BLOCK = 16
_BLOCKS_PER_ROW = MINIMAX_H3_HEAD_DIM // _FP4_BLOCK
_ROW_BYTES4 = MINIMAX_H3_HEAD_DIM // 2
# One contiguous 1 KiB tile of K block scales per (head, 128-key block).
_KSF_TILE_BYTES = _BLOCK_N * _BLOCKS_PER_ROW


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_nvfp4_varlen_attention_module().build_and_load()


def workspace_bytes_nvfp4(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of the per-device workspace the NVFP4 operator needs for ``tokens`` packed rows."""

    num_kblocks = max(1, _ceil_div(tokens, _BLOCK_N))
    padded = num_kblocks * _BLOCK_N
    q4 = k4 = tokens * num_heads * _ROW_BYTES4
    q_sf = tokens * num_heads * _BLOCKS_PER_ROW
    k_sf = num_heads * num_kblocks * _KSF_TILE_BYTES
    vt8 = num_heads * MINIMAX_H3_HEAD_DIM * padded
    scales = (
        tokens * num_heads
        + max(1, num_segments) * num_heads
        + 2 * num_segments * num_heads * MINIMAX_H3_HEAD_DIM
    )
    num_tiles = _ceil_div(tokens, _BLOCK_M)
    partials = num_tiles * num_heads * _PARTIAL_FLOATS + num_tiles * num_heads
    counters = max(1, num_segments) * num_heads
    return q4 + k4 + q_sf + k_sf + vt8 + 4 * (scales + partials + counters)


_ZERO_WORKSPACES: dict[tuple[int, str], torch.Tensor] = {}


_TOK_SEG: dict = {}
_PLAN_ROWS: dict = {}
_TILE_ROW_INTS = 8
_UNIT_ROW_INTS = 4
_STATS_DIRECT_MAX_TOKENS = 4 * _BLOCK_M


def _plan_rows(plan, device: torch.device) -> tuple:
    """The NVFP4 kernels' fat plan tables, derived once per plan from the shared FP8 plan.

    ``tile_rows`` int32 ``[8 * num_stats_tiles]``: (segment, row_begin, row_end, first tile of the
    segment, end tile, segment length, 0, 0) per statistics tile, where a segment of up to
    ``_STATS_DIRECT_MAX_TOKENS`` tokens is one tile (direct finalize) and a longer one is split
    into ``_BLOCK_M``-token tiles; ``unit_rows`` int32 ``[4 * num_units]``: (segment,
    head << 16 | q_tile, segment begin, segment end) per persistent-grid slot.  One vector load
    per CTA / unit replaces the dependent table -> segment -> bounds chains.
    Returns ``(tile_rows, num_stats_tiles, unit_rows)``.
    """
    key = (id(plan), _device_index(device))
    cached = _PLAN_ROWS.get(key)
    if cached is not None and cached[3] is plan:
        return cached[0], cached[1], cached[2]
    bounds = [int(b) for b in plan.bounds]
    rows: list = []
    tiles = 0
    for seg, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=False)):
        length = b - a
        if length <= 0:
            continue
        step = length if length <= _STATS_DIRECT_MAX_TOKENS else _BLOCK_M
        count = -(-length // step)
        for t in range(count):
            begin = a + t * step
            rows.extend(
                (seg, begin, min(begin + step, b), tiles, tiles + count, length, 0, 0)
            )
        tiles += count
    tile_rows = (
        torch.tensor(rows, dtype=torch.int32)
        if rows
        else torch.zeros(_TILE_ROW_INTS, dtype=torch.int32)
    )
    cu = torch.tensor(bounds, dtype=torch.int32)
    ut = plan.unit_table.cpu().to(torch.int32).view(-1, 2)
    if plan.num_units == 0:
        unit_rows = torch.zeros(_UNIT_ROW_INTS, dtype=torch.int32)
    else:
        seg, packed = ut[:, 0], ut[:, 1]
        unit_rows = torch.stack([seg, packed, cu[seg], cu[seg + 1]], dim=1).reshape(-1)
    cached = (tile_rows.to(device), tiles, unit_rows.to(device), plan)
    _PLAN_ROWS[key] = cached
    return cached[0], cached[1], cached[2]


def _longest_segment(bounds: Sequence[int]) -> int:
    """Tokens of the longest segment (0 when all are empty); the TU picks the attention variant with it."""

    return max(
        (int(b) - int(a) for a, b in zip(bounds[:-1], bounds[1:], strict=False)),
        default=0,
    )


def _token_segment_ids(
    bounds: Sequence[int], padded_tokens: int, device: torch.device
) -> torch.Tensor:
    """int32 ``[padded_tokens]`` segment index of every token (padded tokens repeat the last one's).

    Read by the token-parallel quantizer instead of a per-key binary search over ``cu_seqlens``;
    cached per ``(bounds, padded_tokens, device)`` like the rest of the plan.
    """
    key = (tuple(int(b) for b in bounds), int(padded_tokens), _device_index(device))
    cached = _TOK_SEG.get(key)
    if cached is not None:
        return cached
    lens = [b - a for a, b in zip(bounds[:-1], bounds[1:], strict=False)]
    ids = torch.repeat_interleave(
        torch.arange(len(lens), dtype=torch.int32),
        torch.tensor(lens, dtype=torch.int64),
    )
    pad = int(padded_tokens) - int(ids.numel())
    if pad > 0:
        last = ids[-1] if ids.numel() > 0 else torch.zeros((), dtype=torch.int32)
        ids = torch.cat([ids, last.expand(pad)])
    if ids.numel() == 0:
        ids = torch.zeros(1, dtype=torch.int32)
    cached = ids.to(device)
    _TOK_SEG[key] = cached
    return cached


def _zero_workspace(
    index: int, name: str, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    """Grow-only per-device buffer that starts zeroed (the arrival counters of the fused
    statistics launch are reset by the kernel itself after every complete call)."""

    key = (index, name)
    buffer = _ZERO_WORKSPACES.get(key)
    if buffer is None or buffer.numel() < numel:
        buffer = torch.zeros(
            max(numel, 16), dtype=dtype, device=torch.device("cuda", index)
        )
        _ZERO_WORKSPACES[key] = buffer
    return buffer


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_nvfp4",
    mutates_args=(
        "out",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "vt8",
        "q_scale",
        "k_scale",
        "v_scale",
        "mean_k",
        "partials",
        "knorm_part",
        "counters",
    ),
)
def _minimax_h3_sm120_varlen_attention_nvfp4_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    vt8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    knorm_part: torch.Tensor,
    counters: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    attention_grid: int,
    longest_segment: int,
    softmax_scale: float,
) -> None:
    _get_module().minimax_h3_sm120_varlen_attention_nvfp4(
        q,
        k,
        v,
        tok_seg,
        out,
        tile_table,
        unit_table,
        q4,
        k4,
        q_sf,
        k_sf,
        vt8,
        q_scale,
        k_scale,
        v_scale,
        mean_k,
        partials,
        knorm_part,
        counters,
        num_segments,
        num_tiles,
        num_units,
        attention_grid,
        longest_segment,
        softmax_scale,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_nvfp4")
def _minimax_h3_sm120_varlen_attention_nvfp4_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    vt8: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    knorm_part: torch.Tensor,
    counters: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_units: int,
    attention_grid: int,
    longest_segment: int,
    softmax_scale: float,
) -> None:
    pass


@flashinfer_api
def minimax_h3_sm120_varlen_attention_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    r"""Experimental NVFP4-QK / FP8-PV non-causal packed-varlen self-attention for MiniMax-H3 on SM120 (GB202).

    Computes, for every segment ``[a, b)`` of ``cu_seqlens`` and every head ``h``::

        out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]^T * softmax_scale) @ v[a:b, h]

    with NVFP4 tensor-core operands for the scores and FP8 operands for the value product, and
    an FP32 softmax.  Q and K rows are centred by the segment's per-channel K mean (an exact
    softmax shift), rotated by a fixed orthonormal signed Hadamard transform (which spreads
    channel outliers over the row without changing any dot product), prescaled per token (Q) /
    per 128-key block (K) and stored as E2M1 codes with one UE4M3 scale per 16 channels; QK^T
    runs ``mma.sync.m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X``.  The probabilities are
    stored as E4M3 with a 2^8 exponent bias, V per (segment, channel) as E4M3, and PV runs
    ``mma.sync.m16n8k32 kind::f8f6f4``; the output is rounded to BF16 once.  One
    runtime-variable kernel set serves any ``tokens`` and any segment lengths (including empty
    segments and lengths below one tile); the quantized operands live in a grow-only per-device
    workspace (``workspace_bytes_nvfp4``) shared with :func:`minimax_h3_sm120_varlen_attention_fp8`.

    The E2M1 scores carry a larger quantization error than the FP8 operator (about 3x the
    relative L2 error on Gaussian inputs); this route is an experimental precision/latency
    trade-off and is validated against the FP32 oracle with the FP4 block-scaled tolerance
    ``atol = 1.0, rtol = 0.1`` (see the tests), not the FP8 operator's ``0.1``.

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
    tile_rows, num_stats_tiles, unit_rows = _plan_rows(plan, q.device)
    index = _device_index(q.device)
    hd = MINIMAX_H3_HEAD_DIM
    q4 = _workspace(index, "q4", tokens * heads * _ROW_BYTES4, torch.uint8)
    k4 = _workspace(index, "k4", tokens * heads * _ROW_BYTES4, torch.uint8)
    q_sf = _workspace(index, "q_sf", tokens * heads * _BLOCKS_PER_ROW, torch.uint8)
    k_sf = _workspace(
        index, "k_sf", heads * plan.num_kblocks * _KSF_TILE_BYTES, torch.uint8
    )
    vt8 = _workspace(index, "vt8", heads * hd * plan.padded_tokens, torch.uint8)
    q_scale = _workspace(index, "q_scale", tokens * heads, torch.float32)
    k_scale = _workspace(
        index, "k_scale", max(1, plan.num_segments) * heads, torch.float32
    )
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
    knorm_part = _workspace(
        index, "knorm_part", max(1, plan.num_tiles * heads), torch.float32
    )
    counters = _zero_workspace(
        index, "counters", max(1, plan.num_segments) * heads, torch.uint32
    )
    _minimax_h3_sm120_varlen_attention_nvfp4_impl(
        q,
        k,
        v,
        _token_segment_ids(plan.bounds, plan.padded_tokens, q.device),
        out,
        tile_rows,
        unit_rows,
        q4,
        k4,
        q_sf,
        k_sf,
        vt8,
        q_scale,
        k_scale,
        v_scale,
        mean_k,
        partials,
        knorm_part,
        counters,
        plan.num_segments,
        num_stats_tiles,
        plan.num_units,
        plan.grid,
        _longest_segment(plan.bounds),
        float(softmax_scale),
    )
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_NUM_HEADS",
    "minimax_h3_sm120_varlen_attention_nvfp4",
    "workspace_bytes_nvfp4",
]
