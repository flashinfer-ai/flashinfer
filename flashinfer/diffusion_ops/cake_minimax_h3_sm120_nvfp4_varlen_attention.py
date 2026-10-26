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
from collections import OrderedDict
from typing import Optional, Sequence

import torch
import tvm_ffi

from ..api_logging import flashinfer_api
from ..jit.cake_minimax_h3_sm120_nvfp4_fp8pv_varlen_attention import (
    ATTENTION_BASE,
    ATTENTION_SHORT,
    KV_STATS,
    QUANTIZE,
    QUANTIZE_MB3,
)
from ..jit.cake_minimax_h3_sm120_nvfp4_varlen_attention import (
    gen_minimax_h3_sm120_nvfp4_varlen_attention_module,
)
from ..jit.cake_program import CakeProgram
from ..utils import register_custom_op, register_fake_op
from .cake_minimax_h3_sm120_quant_varlen_attention import (
    _BLOCK_M,
    _BLOCK_N,
    _MAX_HEADS,
    _MAX_WORKSPACE_SETS,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    _ceil_div,
    _check_thd,
    _device_index,
    _plan,
    _sm_count,
    _workspace,
    normalize_cu_seqlens,
)

# NVFP4 operand geometry: 16 elements per UE4M3 block scale, two E2M1 codes per byte.
_FP4_BLOCK = 16
_BLOCKS_PER_ROW = MINIMAX_H3_HEAD_DIM // _FP4_BLOCK
_ROW_BYTES4 = MINIMAX_H3_HEAD_DIM // 2
# One contiguous 1 KiB tile of K (per key) or V (per channel) block scales per (head, 128-key block).
_SCALE_TILE_BYTES = _BLOCK_N * _BLOCKS_PER_ROW

# The opt-in NVFP4-QK / FP8-PV route (``pv_mode="fp8"``): three generated programs behind the same API.
# Its attention kernel exponentiates in the log2 domain, so the launch takes ``softmax_scale * log2(e)``.
_PV_MODES = ("fp4", "fp8")
_LOG2E = math.log2(math.e)
# The quantizer owns 32-token blocks; the K block scales are stored as 1 KiB tiles of 64 x 16 B rows.
_QUANT_TOKENS = 32
# The statistics launch keeps 256 floats per (tile, head): 128 K sums and 128 V channel maxima.
_PARTIAL_FLOATS_FP8PV = 2 * MINIMAX_H3_HEAD_DIM
# Grid rule of the route (``minimax_h3_sm120_nvfp4_fp8pv_launch_grids``): grid-stride statistics and quantizer
# launches at 4 and 8 CTAs per SM, one persistent attention CTA per SM.
_STATS_CTAS_PER_SM = 4
_QUANT_CTAS_PER_SM = 8
# Program selection (``minimax_h3_sm120_nvfp4_fp8pv_programs``): the ``min_blocks=3`` quantizer at up to 1024
# tokens, the ``short`` attention schedule when no segment is longer than 4096 tokens.
_QUANT_SMALL_MAX_TOKENS = 1024
_ATTN_SHORT_MAX_TOKENS = 4096
_FP8PV_PROGRAMS = (KV_STATS, QUANTIZE, QUANTIZE_MB3, ATTENTION_BASE, ATTENTION_SHORT)


@functools.cache
def _get_module():
    return gen_minimax_h3_sm120_nvfp4_varlen_attention_module().build_and_load()


def workspace_bytes_nvfp4(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of the per-device workspace the NVFP4 operator needs for ``tokens`` packed rows."""

    num_kblocks = max(1, _ceil_div(tokens, _BLOCK_N))
    padded = num_kblocks * _BLOCK_N
    q4 = k4 = tokens * num_heads * _ROW_BYTES4
    q_sf = tokens * num_heads * _BLOCKS_PER_ROW
    k_sf = v_sf = num_heads * num_kblocks * _SCALE_TILE_BYTES
    vt4 = num_heads * MINIMAX_H3_HEAD_DIM * (padded // 2)
    # One quantized block-mean row (+ scales) and one FP32 channel-mean row per (Q block, head): at most one
    # extra block per segment beyond tokens / 128.
    qtiles = _ceil_div(tokens, _BLOCK_M) + max(1, num_segments)
    qm = qtiles * num_heads * (_ROW_BYTES4 + _BLOCKS_PER_ROW)
    q_mean = qtiles * num_heads * MINIMAX_H3_HEAD_DIM
    mean_k = num_segments * num_heads * MINIMAX_H3_HEAD_DIM
    num_tiles = _ceil_div(tokens, _BLOCK_M) + max(1, num_segments)
    partials = num_tiles * num_heads * MINIMAX_H3_HEAD_DIM
    counters = max(1, num_segments) * num_heads
    barrier = 4  # grid-barrier word of the fused pre-processing launch
    return (
        q4
        + k4
        + q_sf
        + k_sf
        + v_sf
        + vt4
        + qm
        + 4 * (q_mean + mean_k + partials + counters + barrier)
    )


def workspace_bytes_nvfp4_fp8pv(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of the per-(device, stream) workspace ``pv_mode="fp8"`` needs for ``tokens`` packed rows.

    Packed E2M1 Q and K (64 B per row), their UE4M3 block scales (Q per row, K as 1 KiB tiles per
    (head, 128-key block)), the transposed E4M3 V over the padded keys, and the FP32 statistics:
    per-row Q scales, per-(segment, head) K prescales, K means and V channel scales, the statistics
    partials (an upper bound of the tile count: one direct tile per segment plus the 128-token tiles)
    and the arrival counters.
    """

    num_kblocks = max(1, _ceil_div(tokens, _BLOCK_N))
    padded = num_kblocks * _BLOCK_N
    segments = max(1, num_segments)
    tiles = _ceil_div(tokens, _BLOCK_M) + segments
    q4 = k4 = tokens * num_heads * _ROW_BYTES4
    q_sf = tokens * num_heads * _BLOCKS_PER_ROW
    k_sf = num_heads * num_kblocks * _SCALE_TILE_BYTES
    vt8 = num_heads * MINIMAX_H3_HEAD_DIM * padded
    floats = (
        tokens * num_heads  # q_scale
        + segments * num_heads  # k_scale
        + 2 * segments * num_heads * MINIMAX_H3_HEAD_DIM  # mean_k, v_scale
        + tiles * num_heads * _PARTIAL_FLOATS_FP8PV  # partials
        + tiles * num_heads  # knorm_part
        + segments * num_heads  # counters (uint32)
    )
    return q4 + k4 + q_sf + k_sf + vt8 + 4 * floats


_ZERO_WORKSPACES: dict[tuple[int, str], torch.Tensor] = {}
_ZERO_STREAM_WORKSPACES: "OrderedDict[tuple[int, int, str], torch.Tensor]" = (
    OrderedDict()
)


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


def _zero_stream_workspace(
    index: int, name: str, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    """Grow-only buffer ``name`` of the (device, current stream) set that starts zeroed.

    The ``pv_mode="fp8"`` statistics launch counts tile arrivals per (segment, head) and resets
    the counters itself after every complete call, so a buffer is shared between calls on the same
    stream; two streams get their own counters (interleaved arrivals would misidentify the last
    tile).  Bounded like the other workspace sets: the oldest (device, stream) entry is evicted.
    """

    key = (index, torch.cuda.current_stream(index).cuda_stream, name)
    buffer = _ZERO_STREAM_WORKSPACES.get(key)
    if buffer is None or buffer.numel() < numel:
        while len(_ZERO_STREAM_WORKSPACES) >= _MAX_WORKSPACE_SETS:
            _ZERO_STREAM_WORKSPACES.popitem(last=False)
        buffer = torch.zeros(
            max(numel, 16), dtype=dtype, device=torch.device("cuda", index)
        )
        _ZERO_STREAM_WORKSPACES[key] = buffer
    else:
        _ZERO_STREAM_WORKSPACES.move_to_end(key)
    return buffer


def _softmax_scale_log2(softmax_scale: float) -> float:
    """The log2-domain scale the ``pv_mode="fp8"`` attention launch takes: ``softmax_scale * log2(e)``."""
    return float(softmax_scale) * _LOG2E


@functools.cache
def _fp8pv_target(index: int) -> str:
    """The exact target (``"sm_120a"``) every ``pv_mode="fp8"`` program ships for device ``index``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    target = KV_STATS.target_for((capability[0], capability[1]))
    if target is None or any(
        target not in program.targets for program in _FP8PV_PROGRAMS
    ):
        raise RuntimeError(
            'pv_mode="fp8" requires exact compute capability 12.0 (SM120), got '
            f"{capability[0]}.{capability[1]}"
        )
    return target


def minimax_h3_sm120_nvfp4_fp8pv_programs(plan) -> tuple[CakeProgram, CakeProgram]:
    """The ``(quantizer, attention)`` programs ``pv_mode="fp8"`` launches for ``plan``: the
    ``min_blocks=3`` quantizer build (``QUANTIZE_MB3``) at up to 1024 tokens, else ``QUANTIZE``;
    the ``short`` attention schedule (``ATTENTION_SHORT``) when no segment is longer than 4096
    tokens, else ``ATTENTION_BASE``.  The statistics launch is always ``KV_STATS``."""
    bounds = [int(b) for b in plan.bounds]
    quantize = QUANTIZE_MB3 if bounds[-1] <= _QUANT_SMALL_MAX_TOKENS else QUANTIZE
    longest = max(
        (b - a for a, b in zip(bounds[:-1], bounds[1:], strict=False)), default=0
    )
    attention = ATTENTION_SHORT if longest <= _ATTN_SHORT_MAX_TOKENS else ATTENTION_BASE
    return quantize, attention


def minimax_h3_sm120_nvfp4_fp8pv_launch_grids(
    num_sms: int,
    *,
    num_heads: int,
    num_stats_tiles: Optional[int] = None,
    padded_tokens: Optional[int] = None,
    num_units: Optional[int] = None,
) -> tuple[Optional[int], Optional[int], Optional[int]]:
    """The ``(statistics, quantizer, attention)`` launch grids of ``pv_mode="fp8"`` on a device of
    ``num_sms`` SMs: ``max(1, min(num_stats_tiles * num_heads, 4 * num_sms))`` grid-stride CTAs over
    the (statistics tile, head) items, ``max(1, min((padded_tokens // 32) * num_heads, 8 * num_sms))``
    over the (32-token block, head) items, and one persistent attention CTA per SM up to the unit
    count, ``max(1, min(num_sms, max(num_units, 1)))`` (the plan's own grid).  A component whose
    fact is ``None`` is ``None`` (the kitchen's parity harness asks for one launch's grid at a time).
    """
    sms, heads = int(num_sms), int(num_heads)
    if sms <= 0 or heads <= 0:
        raise ValueError(
            f"num_sms and num_heads must be positive, got {num_sms}, {num_heads}"
        )
    stats = (
        None
        if num_stats_tiles is None
        else max(1, min(int(num_stats_tiles) * heads, _STATS_CTAS_PER_SM * sms))
    )
    quantize = (
        None
        if padded_tokens is None
        else max(
            1,
            min(
                (int(padded_tokens) // _QUANT_TOKENS) * heads, _QUANT_CTAS_PER_SM * sms
            ),
        )
    )
    attention = None if num_units is None else max(1, min(sms, max(int(num_units), 1)))
    return stats, quantize, attention


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_nvfp4_fp8pv",
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
def _minimax_h3_sm120_varlen_attention_nvfp4_fp8pv_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    out: torch.Tensor,
    tile_rows: torch.Tensor,
    unit_rows: torch.Tensor,
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
    num_stats_tiles: int,
    num_units: int,
    num_kblocks: int,
    padded_tokens: int,
    quantize_small: bool,
    attention_short: bool,
    stats_grid: int,
    quantize_grid: int,
    attention_grid: int,
    softmax_scale_log2: float,
    target: str,
) -> None:
    # Three PDL-chained launches on the current stream: statistics -> quantizer -> attention.  The
    # bindings take the host arguments in the kernels' signature order followed by the grid triple.
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    quantize = QUANTIZE_MB3 if quantize_small else QUANTIZE
    attention = ATTENTION_SHORT if attention_short else ATTENTION_BASE
    with tvm_ffi.use_torch_stream():
        KV_STATS.entry(target)(
            k,
            v,
            tile_rows,
            partials,
            knorm_part,
            counters,
            mean_k,
            v_scale,
            k_scale,
            num_stats_tiles,
            heads,
            stats_grid,
            1,
            1,
        )
        quantize.entry(target)(
            q,
            k,
            v,
            tok_seg,
            mean_k,
            v_scale,
            q4.view(torch.uint32),
            k4.view(torch.uint32),
            q_sf,
            k_sf,
            vt8.view(torch.uint32),
            q_scale,
            k_scale,
            tokens,
            padded_tokens,
            heads,
            num_kblocks,
            quantize_grid,
            1,
            1,
        )
        attention.entry(target)(
            q4.view(tokens, heads, _ROW_BYTES4),
            k4.view(tokens, heads, _ROW_BYTES4),
            k_sf.view(-1, 16),
            vt8.view(heads, MINIMAX_H3_HEAD_DIM, padded_tokens),
            out,
            q_scale,
            k_scale,
            v_scale,
            q_sf.view(torch.uint32),
            unit_rows,
            num_units,
            heads,
            num_kblocks,
            softmax_scale_log2,
            attention_grid,
            1,
            1,
        )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_nvfp4_fp8pv")
def _minimax_h3_sm120_varlen_attention_nvfp4_fp8pv_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    out: torch.Tensor,
    tile_rows: torch.Tensor,
    unit_rows: torch.Tensor,
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
    num_stats_tiles: int,
    num_units: int,
    num_kblocks: int,
    padded_tokens: int,
    quantize_small: bool,
    attention_short: bool,
    stats_grid: int,
    quantize_grid: int,
    attention_grid: int,
    softmax_scale_log2: float,
    target: str,
) -> None:
    pass


def _run_fp8pv(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    plan,
    softmax_scale: float,
    target: str,
) -> torch.Tensor:
    """The ``pv_mode="fp8"`` body: the workspace of the (device, stream), sliced to the plan's
    exact sizes (the bindings' checks read the tensors handed in), then the three launches."""
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    tile_rows, num_stats_tiles, unit_rows = _plan_rows(plan, q.device)
    index = _device_index(q.device)
    hd = MINIMAX_H3_HEAD_DIM
    segments = max(1, plan.num_segments)
    n_codes = tokens * heads * _ROW_BYTES4
    n_q_sf = tokens * heads * _BLOCKS_PER_ROW
    n_k_sf = heads * plan.num_kblocks * _SCALE_TILE_BYTES
    n_vt8 = heads * hd * plan.padded_tokens
    n_rows = tokens * heads
    n_seg = segments * heads
    n_stats = plan.num_segments * heads * hd
    n_partials = max(1, num_stats_tiles * heads * _PARTIAL_FLOATS_FP8PV)
    n_knorm = max(1, num_stats_tiles * heads)
    q4 = _workspace(index, "q4", n_codes, torch.uint8)[:n_codes]
    k4 = _workspace(index, "k4", n_codes, torch.uint8)[:n_codes]
    q_sf = _workspace(index, "q_sf", n_q_sf, torch.uint8)[:n_q_sf]
    k_sf = _workspace(index, "k_sf", n_k_sf, torch.uint8)[:n_k_sf]
    vt8 = _workspace(index, "vt8", n_vt8, torch.uint8)[:n_vt8]
    q_scale = _workspace(index, "q_scale", n_rows, torch.float32)[:n_rows]
    k_scale = _workspace(index, "k_scale", n_seg, torch.float32)[:n_seg]
    mean_k = _workspace(index, "mean_k", n_stats, torch.float32)[:n_stats]
    v_scale = _workspace(index, "v_scale", n_stats, torch.float32)[:n_stats]
    partials = _workspace(index, "partials", n_partials, torch.float32)[:n_partials]
    knorm_part = _workspace(index, "knorm_part", n_knorm, torch.float32)[:n_knorm]
    counters = _zero_stream_workspace(index, "counters_fp8pv", n_seg, torch.uint32)[
        :n_seg
    ]
    quantize, attention = minimax_h3_sm120_nvfp4_fp8pv_programs(plan)
    stats_grid, quantize_grid, attention_grid = (
        minimax_h3_sm120_nvfp4_fp8pv_launch_grids(
            _sm_count(index),
            num_heads=heads,
            num_stats_tiles=num_stats_tiles,
            padded_tokens=plan.padded_tokens,
            num_units=plan.num_units,
        )
    )
    _minimax_h3_sm120_varlen_attention_nvfp4_fp8pv_impl(
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
        num_stats_tiles,
        plan.num_units,
        plan.num_kblocks,
        plan.padded_tokens,
        quantize is QUANTIZE_MB3,
        attention is ATTENTION_SHORT,
        stats_grid,
        quantize_grid,
        attention_grid,
        _softmax_scale_log2(softmax_scale),
        target,
    )
    return out


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_nvfp4",
    mutates_args=(
        "out",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "qm4",
        "qm_sf",
        "vt4",
        "v_sf",
        "q_mean",
        "mean_k",
        "partials",
        "counters",
        "barrier",
    ),
)
def _minimax_h3_sm120_varlen_attention_nvfp4_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    qm4: torch.Tensor,
    qm_sf: torch.Tensor,
    vt4: torch.Tensor,
    v_sf: torch.Tensor,
    q_mean: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    barrier: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_qtiles: int,
    num_units: int,
    attention_grid: int,
    softmax_scale: float,
) -> None:
    _get_module().minimax_h3_sm120_varlen_attention_nvfp4(
        q,
        k,
        v,
        tok_seg,
        cu_seqlens,
        seg_tile_begin,
        out,
        tile_table,
        unit_table,
        q4,
        k4,
        q_sf,
        k_sf,
        qm4,
        qm_sf,
        vt4,
        v_sf,
        q_mean,
        mean_k,
        partials,
        counters,
        barrier,
        num_segments,
        num_tiles,
        num_qtiles,
        num_units,
        attention_grid,
        softmax_scale,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_nvfp4")
def _minimax_h3_sm120_varlen_attention_nvfp4_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    qm4: torch.Tensor,
    qm_sf: torch.Tensor,
    vt4: torch.Tensor,
    v_sf: torch.Tensor,
    q_mean: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    barrier: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_qtiles: int,
    num_units: int,
    attention_grid: int,
    softmax_scale: float,
) -> None:
    pass


@register_custom_op(
    "flashinfer::minimax_h3_sm120_varlen_attention_nvfp4_nodelta",
    mutates_args=(
        "out",
        "q4",
        "k4",
        "q_sf",
        "k_sf",
        "qm4",
        "qm_sf",
        "vt4",
        "v_sf",
        "q_mean",
        "mean_k",
        "partials",
        "counters",
        "barrier",
    ),
)
def _minimax_h3_sm120_varlen_attention_nvfp4_nodelta_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    qm4: torch.Tensor,
    qm_sf: torch.Tensor,
    vt4: torch.Tensor,
    v_sf: torch.Tensor,
    q_mean: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    barrier: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_qtiles: int,
    num_units: int,
    attention_grid: int,
    softmax_scale: float,
) -> None:
    _get_module().minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
        q,
        k,
        v,
        tok_seg,
        cu_seqlens,
        seg_tile_begin,
        out,
        tile_table,
        unit_table,
        q4,
        k4,
        q_sf,
        k_sf,
        qm4,
        qm_sf,
        vt4,
        v_sf,
        q_mean,
        mean_k,
        partials,
        counters,
        barrier,
        num_segments,
        num_tiles,
        num_qtiles,
        num_units,
        attention_grid,
        softmax_scale,
    )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_nvfp4_nodelta")
def _minimax_h3_sm120_varlen_attention_nvfp4_nodelta_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tok_seg: torch.Tensor,
    cu_seqlens: torch.Tensor,
    seg_tile_begin: torch.Tensor,
    out: torch.Tensor,
    tile_table: torch.Tensor,
    unit_table: torch.Tensor,
    q4: torch.Tensor,
    k4: torch.Tensor,
    q_sf: torch.Tensor,
    k_sf: torch.Tensor,
    qm4: torch.Tensor,
    qm_sf: torch.Tensor,
    vt4: torch.Tensor,
    v_sf: torch.Tensor,
    q_mean: torch.Tensor,
    mean_k: torch.Tensor,
    partials: torch.Tensor,
    counters: torch.Tensor,
    barrier: torch.Tensor,
    num_segments: int,
    num_tiles: int,
    num_qtiles: int,
    num_units: int,
    attention_grid: int,
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
    pv_mode: str = "fp4",
) -> torch.Tensor:
    r"""Experimental NVFP4 non-causal packed-varlen self-attention for MiniMax-H3 on SM120 (GB202),
    following the SageAttention3 FP4 recipe.

    Computes, for every segment ``[a, b)`` of ``cu_seqlens`` and every head ``h``::

        out[a:b, h] = softmax(q[a:b, h] @ k[a:b, h]^T * softmax_scale) @ v[a:b, h]

    with block-scaled NVFP4 tensor-core operands for both the scores and the value product and an
    FP32 softmax, as in SageAttention3 (``sageattention3_blackwell``): K rows are centred by the
    segment's per-channel mean and Q rows by their 128-row block's mean (the block-mean term
    ``qm K^T`` is added back inside the kernel, so the softmax is unchanged), Q / K / V are stored
    as E2M1 codes with one UE4M3 scale per 16 elements (block amax / 6; V transposed with per-16-key
    scales), QK^T and PV both run ``mma.sync.m16n8k64 kind::mxf4nvf4 block_scale scale_vec::4X``,
    and the probabilities use Sage3's two-level scheme (row maximum brought to 448 * 6, one UE4M3
    scale per 16 keys, E2M1 codes); the output is rounded to BF16 once.  One runtime-variable
    kernel set serves any ``tokens`` and any segment lengths (including empty segments and lengths
    below one tile); the quantized operands live in a grow-only per-device workspace
    (``workspace_bytes_nvfp4``) shared with :func:`minimax_h3_sm120_varlen_attention_fp8`.

    The E2M1 scores and probabilities carry a larger quantization error than the FP8 operator
    (about 4x the relative L2 error on Gaussian inputs, the same as SageAttention3's own); this
    route is an experimental precision/latency trade-off and is validated against the FP32 oracle
    with the FP4 block-scaled tolerance ``atol = 1.0, rtol = 0.1`` (see the tests), not the FP8
    operator's ``0.1``.

    Performance ceiling of the attention launch (measured on RTX 5090 and RTX PRO 6000 Blackwell,
    both GB202): the kernel needs 246 registers per thread, so one 8-warp CTA is resident per SM
    (two warps per SM sub-partition) and each warp alternates a ~1400-cycle ``mxf4nvf4`` MMA burst
    with a ~2700-cycle FP32 softmax / P-quantization phase that only the other warp's burst can
    overlap.  The tensor pipe is therefore active ~58 % of the time and the launch runs at ~60 %
    of the tensor-pipe floor on production plans (~55 % on 4096-token plans), against 84-88 % for
    the MMA-only skeleton; L2 and DRAM are idle (L2 hit > 99 %).  Ping-pong barrier placement,
    softmax emission order, FMA-pipe exp2, power-of-two P block scales, packed f16x2 P conversions,
    a precomputed ``qm K^T`` compensation table, K/V multicast / DSM sharing, a 64-key 3-CTA
    geometry and split-KV were each measured or bounded on both SKUs and none is faster within the
    FP4 error budget, so the attention kernel is unchanged; the softmax dependency chain under the
    two-warps-per-sub-partition register budget is the binding resource.

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
    pv_mode : str
        ``"fp4"`` (default): the SageAttention3-recipe operator above, unchanged.  ``"fp8"``: the
        opt-in NVFP4-QK / FP8-PV route -- the scores run on block-scaled NVFP4 operands (Q rotated by a
        signed Hadamard transform and K centred by the segment's per-channel mean, one UE4M3 scale per
        16 channels under a per-row / per-(segment, head) prescale) while the probabilities and V stay
        E4M3 as in :func:`minimax_h3_sm120_varlen_attention_fp8` (per-channel V scales, FA3 key
        permutation); three programmatic-dependent launches (fused statistics, quantizer, attention).
        It trades the FP4 probabilities for a smaller error (relative L2 about 0.14 on Gaussian inputs
        against about 0.19 for ``"fp4"`` and 0.05 for the FP8 operator) at a lower latency than the FP8
        operator; see the tests.  The same workspace names are reused; the arrival counters of the
        statistics launch are private to the (device, stream).  Any other value is a ``ValueError``.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[tokens, heads, 128]`` attention output.
    """

    if pv_mode not in _PV_MODES:
        raise ValueError(f"pv_mode must be one of {_PV_MODES}, got {pv_mode!r}")
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
    fp8pv_target = _fp8pv_target(_device_index(q.device)) if pv_mode == "fp8" else None
    plan = _plan(bounds, heads, q.device)
    if plan.num_units == 0:
        return out  # every segment is empty: nothing to write
    if fp8pv_target is not None:
        return _run_fp8pv(q, k, v, out, plan, float(softmax_scale), fp8pv_target)
    tile_rows, num_stats_tiles, unit_rows = _plan_rows(plan, q.device)
    index = _device_index(q.device)
    hd = MINIMAX_H3_HEAD_DIM
    qtiles = max(1, plan.num_tiles)
    q4 = _workspace(index, "q4", tokens * heads * _ROW_BYTES4, torch.uint8)
    k4 = _workspace(index, "k4", tokens * heads * _ROW_BYTES4, torch.uint8)
    q_sf = _workspace(index, "q_sf", tokens * heads * _BLOCKS_PER_ROW, torch.uint8)
    k_sf = _workspace(
        index, "k_sf", heads * plan.num_kblocks * _SCALE_TILE_BYTES, torch.uint8
    )
    qm4 = _workspace(index, "qm4", qtiles * heads * _ROW_BYTES4, torch.uint8)
    qm_sf = _workspace(index, "qm_sf", qtiles * heads * _BLOCKS_PER_ROW, torch.uint8)
    vt4 = _workspace(index, "vt4", heads * hd * (plan.padded_tokens // 2), torch.uint8)
    v_sf = _workspace(
        index, "v_sf", heads * plan.num_kblocks * _SCALE_TILE_BYTES, torch.uint8
    )
    q_mean = _workspace(index, "q_mean", qtiles * heads * hd, torch.float32)
    mean_k = _workspace(index, "mean_k", plan.num_segments * heads * hd, torch.float32)
    partials = _workspace(
        index, "partials", max(1, num_stats_tiles * heads * hd), torch.float32
    )
    counters = _zero_workspace(
        index, "counters", max(1, plan.num_segments) * heads, torch.uint32
    )
    barrier = _zero_workspace(index, "barrier", 4, torch.uint32)
    _minimax_h3_sm120_varlen_attention_nvfp4_impl(
        q,
        k,
        v,
        _token_segment_ids(plan.bounds, plan.padded_tokens, q.device),
        plan.cu_seqlens,
        plan.seg_tile_begin,
        out,
        tile_rows,
        unit_rows,
        q4,
        k4,
        q_sf,
        k_sf,
        qm4,
        qm_sf,
        vt4,
        v_sf,
        q_mean,
        mean_k,
        partials,
        counters,
        barrier,
        plan.num_segments,
        num_stats_tiles,
        plan.num_tiles,
        plan.num_units,
        plan.grid,
        float(softmax_scale),
    )
    return out


@flashinfer_api
def minimax_h3_sm120_varlen_attention_nvfp4_nodelta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    softmax_scale: Optional[float] = None,
) -> torch.Tensor:
    r"""Opt-in **uncompensated** variant of :func:`minimax_h3_sm120_varlen_attention_nvfp4` (same contract,
    same signature, same workspace and pre-processing launches; only the attention kernel differs).

    The default route quantizes each 128-row Q block after removing its block mean and adds the removed
    term ``qm K^T`` back to the scores inside the kernel, so its softmax equals the uncentred one up to
    FP4 rounding.  This route keeps the centring but **does not add the term back**: the scores of every
    Q block carry a per-key offset ``-(qm . k_j)``, which the softmax does not cancel (unlike the per-row
    constant of the K mean removal).  It is a deliberate precision / latency trade-off that is NOT part
    of the SageAttention3 recipe:

    * measured error vs the FP32 oracle on Gaussian inputs: relative L2 about 0.209 vs 0.191 for the
      default route (+9 %), max-abs about 0.064 vs 0.060; the FP4 block-scaled tolerance
      ``atol = 1.0, rtol = 0.1`` still holds on every element, and the tests bound the relative L2 at
      1.15x the default route's on the same inputs;
    * the deviation grows on short segments: the dropped term is the mean over a 128-row Q block, and a
      partial block of ``n`` rows has a block mean about ``sqrt(128 / n)`` larger.  Measured relative-L2
      ratio vs the default route for a single segment of N tokens (56 heads, Gaussian inputs): 1.10 at
      N >= 128 (full blocks), 1.13 at 100, 1.18 at 64, 1.23 at 51, 1.32 at 32, 1.64 at 16 (max-abs 0.79,
      still inside the FP4 tolerance).  In a packed stream the aggregate error is dominated by the short
      segments, whose outputs are larger per element.  **Segments of fewer than 8 tokens exceed the FP4
      tolerance on this route** (max-abs 1.2-2.7 at 2-6 tokens; a 1-token segment is exact by construction),
      so do not use it for streams that contain such segments -- the default route stays within tolerance
      there;
    * measured latency: the attention launch runs about 9-15 % faster on both GB202 SKUs (RTX 5090 and
      RTX PRO 6000 Blackwell; 220 registers instead of 246), the complete call proportionally less on
      short plans where pre-processing dominates.

    Use the default route unless the extra error has been validated for the model in question.  Inputs,
    output, ``cu_seqlens`` handling, workspace and caching are exactly those of the default route.

    Parameters
    ----------
    q, k, v : torch.Tensor
        Contiguous ``bfloat16`` CUDA tensors of shape ``[tokens, heads, 128]`` (packed THD
        layout; MiniMax-H3 uses 56 heads).
    cu_seqlens : torch.Tensor
        ``int32`` tensor of shape ``[segments + 1]`` on the same device with ``cu_seqlens[0] = 0``,
        non-decreasing entries and ``cu_seqlens[-1] = tokens``.  Segments shorter than 8 tokens are
        outside this route's FP4 tolerance (see above).
    out : Optional[torch.Tensor]
        Optional pre-allocated output of the same shape/dtype as ``q``; allocated when omitted.
    cu_seqlens_host : Optional[Sequence[int]]
        Host copy of ``cu_seqlens`` (avoids a synchronizing device-to-host copy).  The
        segment plan is cached per ``(cu_seqlens, heads, device)`` and shared with the default route.
    softmax_scale : Optional[float]
        Softmax scale; defaults to ``1 / sqrt(128)``.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[tokens, heads, 128]`` attention output of the uncompensated route.
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
    qtiles = max(1, plan.num_tiles)
    q4 = _workspace(index, "q4", tokens * heads * _ROW_BYTES4, torch.uint8)
    k4 = _workspace(index, "k4", tokens * heads * _ROW_BYTES4, torch.uint8)
    q_sf = _workspace(index, "q_sf", tokens * heads * _BLOCKS_PER_ROW, torch.uint8)
    k_sf = _workspace(
        index, "k_sf", heads * plan.num_kblocks * _SCALE_TILE_BYTES, torch.uint8
    )
    qm4 = _workspace(index, "qm4", qtiles * heads * _ROW_BYTES4, torch.uint8)
    qm_sf = _workspace(index, "qm_sf", qtiles * heads * _BLOCKS_PER_ROW, torch.uint8)
    vt4 = _workspace(index, "vt4", heads * hd * (plan.padded_tokens // 2), torch.uint8)
    v_sf = _workspace(
        index, "v_sf", heads * plan.num_kblocks * _SCALE_TILE_BYTES, torch.uint8
    )
    q_mean = _workspace(index, "q_mean", qtiles * heads * hd, torch.float32)
    mean_k = _workspace(index, "mean_k", plan.num_segments * heads * hd, torch.float32)
    partials = _workspace(
        index, "partials", max(1, num_stats_tiles * heads * hd), torch.float32
    )
    counters = _zero_workspace(
        index, "counters", max(1, plan.num_segments) * heads, torch.uint32
    )
    barrier = _zero_workspace(index, "barrier", 4, torch.uint32)
    _minimax_h3_sm120_varlen_attention_nvfp4_nodelta_impl(
        q,
        k,
        v,
        _token_segment_ids(plan.bounds, plan.padded_tokens, q.device),
        plan.cu_seqlens,
        plan.seg_tile_begin,
        out,
        tile_rows,
        unit_rows,
        q4,
        k4,
        q_sf,
        k_sf,
        qm4,
        qm_sf,
        vt4,
        v_sf,
        q_mean,
        mean_k,
        partials,
        counters,
        barrier,
        plan.num_segments,
        num_stats_tiles,
        plan.num_tiles,
        plan.num_units,
        plan.grid,
        float(softmax_scale),
    )
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_NUM_HEADS",
    "minimax_h3_sm120_nvfp4_fp8pv_launch_grids",
    "minimax_h3_sm120_nvfp4_fp8pv_programs",
    "minimax_h3_sm120_varlen_attention_nvfp4",
    "minimax_h3_sm120_varlen_attention_nvfp4_nodelta",
    "workspace_bytes_nvfp4",
    "workspace_bytes_nvfp4_fp8pv",
]
