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
from typing import Any, Dict, Optional, Sequence, Tuple

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
    ATTENTION,
    ATTENTION_LEAD,
    ATTENTION_NODELTA,
    ATTENTION_NODELTA_LEAD,
    ATTENTION_NODELTA_TAIL,
    ATTENTION_SFSEL_LEAD,
    ATTENTION_SFSEL_TAIL,
    ATTENTION_TAIL,
    ATTENTION_WS,
    ATTENTION_WS_NODELTA,
    ATTENTION_WS_NODELTA_LEAD,
    ATTENTION_WS_NODELTA_TAIL,
    PREPROCESS,
    PREPROCESS_HEAD_MAJOR,
    PREPROCESS_HEAD_MAJOR_KA,
    QUANTIZE_KV,
    QUANTIZE_KV_HBLK64,
    STATS,
)
from ..jit.cake_program import CakeProgram
from ..utils import register_custom_op, register_fake_op
from .cake_minimax_h3_sm120_quant_varlen_attention import (
    _BLOCK_M,
    _BLOCK_N,
    _MAX_HEADS,
    _MAX_PLANS,
    MINIMAX_H3_HEAD_DIM,
    MINIMAX_H3_NUM_HEADS,
    _capturing,
    _ceil_div,
    _check_thd,
    _current_stream,
    _device_index,
    _ffi_device,
    _owner,
    _owner_entry,
    _plan,
    _sm_count,
    _stream_lock,
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

# The NVFP4 route (``pv_mode="fp4"`` and the uncompensated entry): eight generated programs, two or three
# programmatic-dependent launches per call (``minimax_h3_sm120_nvfp4_programs``).  A plan of 2048 to 16384
# padded tokens runs the fused pre-processing (``PREPROCESS``: statistics, a software grid barrier, then the
# quantizer, in one cooperative launch); every other plan runs the statistics launch, which also quantizes Q
# (``STATS``), then the K / V quantizer (``QUANTIZE_KV``; ``QUANTIZE_KV_HBLK64``, the same items in 64-block
# head runs, above 16384 padded tokens).
_PREPROCESS_MIN_TOKENS = 2048
_PREPROCESS_MAX_TOKENS = 16384
# Every CTA of the fused pre-processing must be co-resident (its grid barrier waits for all of them): the grid
# stays within the three CTAs per SM the kernel's launch bound guarantees.
_PREPROCESS_CTAS_PER_SM = 3
# The attention launch takes the warp-specialised form (``ATTENTION_WS`` / ``ATTENTION_WS_NODELTA``) on parts of
# at least 188 SMs when the mean segment (empty segments included) is at least 1536 tokens and, on the
# compensated route, below 16384 tokens; the uncompensated route also takes it on parts of at least 170 SMs for a
# plan of the fused pre-processing (2048 to 16384 padded tokens) with that mean segment; else the base form
# (``ATTENTION`` / ``ATTENTION_NODELTA``).
_WS_MIN_SMS = 188
_WS_NODELTA_FUSED_MIN_SMS = 170
_WS_MIN_MEAN_SEGMENT = 1536
_WS_MAX_MEAN_SEGMENT = 16384
# The overlapped form of a plan of the fused pre-processing, on a part whose SM count has an overlap width P
# (``minimax_h3_sm120_nvfp4_overlap_grids``), in launch order: the head-major pre-processing producers (a plain
# launch of ``min(3 * P, sms - lead)`` CTAs, one per SM, each running three 256-thread workers), the lead
# attention launch over all but P SMs (a programmatic dependent of the producers) and the tail attention launch
# over P CTAs (a programmatic dependent of the lead), all three sharing the head-control words ``hctr`` (5 words
# per head, then 8): each head's operands are published by a ready flag the attention CTAs wait on before their
# loads, the tail ends after every lead CTA and the producers' last CTA have signalled their completion, and
# the words reset themselves in every complete call.  A CTA only waits on CTAs of an earlier launch of its own
# call, which have been resident (or have exited) before it runs, so a call completes whatever else runs on the
# device.
_OVERLAP_WIDTH = {170: 42, 188: 46}
# On the SM counts of ``_OVERLAP_SFSEL_SMS`` the compensated overlap of a plan above the fused pre-processing
# (more than 16384 padded tokens) launches ``ATTENTION_SFSEL_LEAD`` / ``ATTENTION_SFSEL_TAIL`` in place of
# ``ATTENTION_LEAD`` / ``ATTENTION_TAIL``: the same units, arguments, grids, head-control words and protocol, with
# the B scale records of four n8 tiles in one load per lane, each MMA selecting its tile's record (thread-id-b), so
# the outputs are bitwise equal.  That form is faster on those plans on 170 SMs, and slower on the plans of the
# fused pre-processing and on 188 SMs.
_OVERLAP_SFSEL_SMS = (170,)
# On the SM counts of ``_OVERLAP_WS_NODELTA_SMS`` the uncompensated overlap launches the warp-specialised lead and
# tail (``ATTENTION_WS_NODELTA_LEAD`` / ``ATTENTION_WS_NODELTA_TAIL``: the ``ATTENTION_WS_NODELTA`` unit -- 8 QK /
# softmax warps hand each tile's P record to 4 PV warps, 384 threads -- taking its units from the counter and
# waiting for each unit's head flag) in place of ``ATTENTION_NODELTA_LEAD`` / ``ATTENTION_NODELTA_TAIL``, with the
# same grids, head-control words and protocol, and the producers ``PREPROCESS_HEAD_MAJOR_KA`` in place of
# ``PREPROCESS_HEAD_MAJOR`` (the same launch and outputs; a K item waits for its head's K sums in the worker's second
# warp, beside the first warp's counts of the previous item and claim of the next one).
_OVERLAP_WS_NODELTA_SMS = (188,)
# A plan of more than 16384 padded tokens (the statistics / quantizer plans) takes the overlapped form too when its
# attention-weighted mean segment length -- the sum of the squared segment lengths over the token count, which the
# pre-processing share of the call falls with -- is at most ``_OVERLAP_LONG_MAX_SEGMENT[(SM count, delta)]`` tokens
# (no entry: never), with the overlap width ``_OVERLAP_WIDTH_LONG``.  On 170 SMs both routes and on 188 SMs the
# compensated route overlap every such plan: the hidden pre-processing pays for the overlap attention's per-unit Q
# load at any segment length.  On 188 SMs the uncompensated route overlaps such a plan up to an attention-weighted
# mean segment of 40960 tokens; above it the separate launches with the warp-specialised attention
# (``ATTENTION_WS_NODELTA``, three warps per SM sub-partition) finish first, unless the plan's mean segment keeps
# the separate launches on the base attention (``_warp_specialised``), which keeps the overlap.
_OVERLAP_WIDTH_LONG = {170: 64, 188: 46}
_OVERLAP_LONG_MAX_SEGMENT = {
    (170, True): 1 << 40,
    (170, False): 1 << 40,
    (188, True): 1 << 40,
    (188, False): 40960,
}
# Workers per producer CTA of the overlapped form (one 768-thread CTA runs three 256-thread workers).
_OVERLAP_PRODUCER_WORKERS = 3
_HCTR_WORDS_PER_HEAD = 5
_HCTR_TAIL_WORDS = 8
_NVFP4_PROGRAMS = (
    PREPROCESS,
    STATS,
    QUANTIZE_KV,
    QUANTIZE_KV_HBLK64,
    ATTENTION,
    ATTENTION_NODELTA,
    ATTENTION_WS,
    ATTENTION_WS_NODELTA,
    PREPROCESS_HEAD_MAJOR,
    ATTENTION_LEAD,
    ATTENTION_TAIL,
    ATTENTION_NODELTA_LEAD,
    ATTENTION_NODELTA_TAIL,
    ATTENTION_WS_NODELTA_LEAD,
    ATTENTION_WS_NODELTA_TAIL,
    PREPROCESS_HEAD_MAJOR_KA,
    ATTENTION_SFSEL_LEAD,
    ATTENTION_SFSEL_TAIL,
)
# The producer programs of the overlapped form (its first launch).
_NVFP4_PRODUCERS = (PREPROCESS_HEAD_MAJOR, PREPROCESS_HEAD_MAJOR_KA)


def workspace_bytes_nvfp4(tokens: int, num_heads: int, num_segments: int) -> int:
    """Bytes of one per-(device, stream) workspace set the NVFP4 operator needs for ``tokens`` packed rows."""

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


# The zero-start buffers of every workspace owner (``_owner``): one ``{name: buffer}`` set per owner, the least
# recently used set evicted whole.
_ZERO_STREAM_WORKSPACES: "OrderedDict[Tuple[int, ...], Dict[str, torch.Tensor]]" = (
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


_PLAN_ROWS_HM: dict = {}


def _plan_rows_head_major(plan, device: torch.device) -> torch.Tensor:
    """int32 ``[4 * num_units]`` unit rows of the overlapped form, head-major: every unit of head 0, then of head 1,
    ..., each head's units in segment and Q-tile order, with the rows of ``_plan_rows``.  The attention CTAs take
    the units in this order, so the heads become ready in the producers' order.  Cached per plan like
    ``_plan_rows``."""
    key = (id(plan), _device_index(device))
    cached = _PLAN_ROWS_HM.get(key)
    if cached is not None and cached[1] is plan:
        return cached[0]
    bounds = [int(b) for b in plan.bounds]
    per_head: list = []
    for seg, (a, b) in enumerate(zip(bounds[:-1], bounds[1:], strict=False)):
        for tile in range(_ceil_div(b - a, _BLOCK_M) if b > a else 0):
            per_head.append((seg, tile, a, b))
    rows = [
        (seg, (head << 16) | tile, a, b)
        for head in range(plan.num_heads)
        for seg, tile, a, b in per_head
    ]
    if len(rows) != plan.num_units:
        raise RuntimeError(
            f"head-major unit rows: {len(rows)} units, the plan has {plan.num_units}"
        )
    unit_rows = (
        torch.tensor(rows, dtype=torch.int32).reshape(-1)
        if rows
        else torch.zeros(_UNIT_ROW_INTS, dtype=torch.int32)
    )
    cached = (unit_rows.to(device), plan)
    _PLAN_ROWS_HM[key] = cached
    return cached[0]


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


def _zero_stream_workspace(
    owner: Tuple[int, ...], name: str, numel: int, dtype: torch.dtype
) -> torch.Tensor:
    """Grow-only buffer ``name`` of the zero-start set of ``owner`` (``_owner``; device ``owner[0]``).

    The statistics / pre-processing launches count tile arrivals per (segment, head), the fused
    pre-processing synchronises its grid on a barrier word and the overlapped form hands heads over
    on its head-control words; the kernels reset every such word after each complete call, so a
    buffer is shared between the calls of one owner, which are ordered, and every other owner (another
    stream, another graph capture) gets its own words (arrivals interleaved across calls would
    misidentify the last tile, release a barrier early or set a ready flag twice).  A buffer created
    during a graph capture is zeroed by a memset node of the graph, at every replay.  Bounded like the
    other workspace sets: the least recently used owner's set is evicted whole.
    """

    buffers = _owner_entry(_ZERO_STREAM_WORKSPACES, owner, dict)
    buffer = buffers.get(name)
    if buffer is None or buffer.numel() < numel:
        buffer = buffers[name] = torch.zeros(
            max(numel, 16), dtype=dtype, device=torch.device("cuda", owner[0])
        )
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
    """The ``pv_mode="fp8"`` body: under the (device, stream) lock (``_stream_lock``), the workspace
    set of the call's owner (``_owner``), sliced to the plan's exact sizes (the bindings' checks read
    the tensors handed in), then the three launches."""
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
    stream = _current_stream(index)
    with _stream_lock(index, stream):
        owner = _owner(index, stream)
        q4 = _workspace(owner, "q4", n_codes, torch.uint8)[:n_codes]
        k4 = _workspace(owner, "k4", n_codes, torch.uint8)[:n_codes]
        q_sf = _workspace(owner, "q_sf", n_q_sf, torch.uint8)[:n_q_sf]
        k_sf = _workspace(owner, "k_sf", n_k_sf, torch.uint8)[:n_k_sf]
        vt8 = _workspace(owner, "vt8", n_vt8, torch.uint8)[:n_vt8]
        q_scale = _workspace(owner, "q_scale", n_rows, torch.float32)[:n_rows]
        k_scale = _workspace(owner, "k_scale", n_seg, torch.float32)[:n_seg]
        mean_k = _workspace(owner, "mean_k", n_stats, torch.float32)[:n_stats]
        v_scale = _workspace(owner, "v_scale", n_stats, torch.float32)[:n_stats]
        partials = _workspace(owner, "partials", n_partials, torch.float32)[:n_partials]
        knorm_part = _workspace(owner, "knorm_part", n_knorm, torch.float32)[:n_knorm]
        counters = _zero_stream_workspace(owner, "counters_fp8pv", n_seg, torch.uint32)[
            :n_seg
        ]
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


# The K / V scale tiles as 16-byte rows: the attention binding's tensor-map view of them.
_SCALE_ROW_BYTES = 16


def _q_slots(bounds: Sequence[int]) -> int:
    """The most 128-row Q batches of one statistics tile of the plan (at least 1): a segment of up to
    ``_STATS_DIRECT_MAX_TOKENS`` tokens is one tile of its whole length, a longer one is split into
    ``_BLOCK_M``-token tiles (``_plan_rows``)."""
    slots = 1
    for a, b in zip(bounds[:-1], bounds[1:], strict=False):
        length = int(b) - int(a)
        if 0 < length <= _STATS_DIRECT_MAX_TOKENS:
            slots = max(slots, _ceil_div(length, _BLOCK_M))
    return slots


def minimax_h3_sm120_nvfp4_programs(
    plan, num_sms: int, *, delta: bool = True
) -> Tuple[CakeProgram, ...]:
    """The programs one call launches for ``plan`` on a device of ``num_sms`` SMs, in launch order.

    Pre-processing: ``(PREPROCESS,)`` (the fused statistics and quantizer, one cooperative launch) for a
    plan of 2048 to 16384 padded tokens, else ``(STATS, QUANTIZE_KV)`` -- the statistics launch, which
    also quantizes Q, then the K / V quantizer -- with ``QUANTIZE_KV_HBLK64`` in place of ``QUANTIZE_KV``
    above 16384 padded tokens.  Attention: the warp-specialised form (``ATTENTION_WS``, or
    ``ATTENTION_WS_NODELTA`` when ``delta`` is false; without ``delta`` also when ``num_sms >= 170``
    for a plan of the fused pre-processing) when ``num_sms >= 188`` and the mean segment length
    ``tokens / segments`` (empty segments included) is at least 1536 tokens and, with ``delta``, below
    16384 tokens; else ``ATTENTION`` / ``ATTENTION_NODELTA``.

    A plan of the fused pre-processing on a part whose SM count has an overlap width
    (``minimax_h3_sm120_nvfp4_overlap_grids``), and a longer plan whose attention-weighted mean segment is
    short enough or whose separate launches would not take the warp-specialised attention (``_overlap_long``),
    runs the overlapped form instead: ``(PREPROCESS_HEAD_MAJOR, ATTENTION_LEAD, ATTENTION_TAIL)``
    (``ATTENTION_SFSEL_LEAD`` / ``ATTENTION_SFSEL_TAIL`` for a longer plan on the SM counts of
    ``_OVERLAP_SFSEL_SMS``; ``ATTENTION_NODELTA_LEAD`` / ``ATTENTION_NODELTA_TAIL`` when ``delta`` is false, the warp-specialised
    ``ATTENTION_WS_NODELTA_LEAD`` / ``ATTENTION_WS_NODELTA_TAIL`` with the producers ``PREPROCESS_HEAD_MAJOR_KA``
    on the SM counts of ``_OVERLAP_WS_NODELTA_SMS``) -- the head-major
    pre-processing producers (a plain launch), the lead attention launch on the SMs the producers leave free
    (a programmatic dependent of the producers), then the tail attention launch (a programmatic dependent of
    the lead), which takes the remaining work on the producers' SMs as they drain."""
    padded = int(plan.padded_tokens)
    fused = _PREPROCESS_MIN_TOKENS <= padded <= _PREPROCESS_MAX_TOKENS
    if int(num_sms) in _OVERLAP_WIDTH and (
        fused or _overlap_long(plan, num_sms, delta)
    ):
        if delta:
            if int(num_sms) in _OVERLAP_SFSEL_SMS and not fused:
                return (
                    PREPROCESS_HEAD_MAJOR,
                    ATTENTION_SFSEL_LEAD,
                    ATTENTION_SFSEL_TAIL,
                )
            return (PREPROCESS_HEAD_MAJOR, ATTENTION_LEAD, ATTENTION_TAIL)
        if int(num_sms) in _OVERLAP_WS_NODELTA_SMS:
            return (
                PREPROCESS_HEAD_MAJOR_KA,
                ATTENTION_WS_NODELTA_LEAD,
                ATTENTION_WS_NODELTA_TAIL,
            )
        return (PREPROCESS_HEAD_MAJOR, ATTENTION_NODELTA_LEAD, ATTENTION_NODELTA_TAIL)
    if fused:
        pre: Tuple[CakeProgram, ...] = (PREPROCESS,)
    elif padded > _PREPROCESS_MAX_TOKENS:
        pre = (STATS, QUANTIZE_KV_HBLK64)
    else:
        pre = (STATS, QUANTIZE_KV)
    if _warp_specialised(plan, num_sms, delta, fused):
        attention = ATTENTION_WS if delta else ATTENTION_WS_NODELTA
    else:
        attention = ATTENTION if delta else ATTENTION_NODELTA
    return (*pre, attention)


def _warp_specialised(plan, num_sms: int, delta: bool, fused: bool) -> bool:
    """Whether the separate launches of ``plan`` take the warp-specialised attention: the mean segment
    ``tokens / segments`` (empty segments included) is at least 1536 tokens and either the part has at least 188
    SMs (with ``delta``, a mean below 16384 tokens) or, without ``delta``, the plan runs the fused pre-processing
    on a part of at least 170 SMs."""
    tokens, segments = int(plan.total_tokens), int(plan.num_segments)
    bound = _WS_MAX_MEAN_SEGMENT if delta else 0
    return tokens >= _WS_MIN_MEAN_SEGMENT * segments and (
        (int(num_sms) >= _WS_MIN_SMS and (bound <= 0 or tokens < bound * segments))
        or (not delta and fused and int(num_sms) >= _WS_NODELTA_FUSED_MIN_SMS)
    )


def _overlap_long(plan, num_sms: int, delta: bool) -> bool:
    """Whether a plan of more than 16384 padded tokens takes the overlapped form on a part of ``num_sms`` SMs: its
    attention-weighted mean segment length ``sum(len ** 2) / tokens`` is at most
    ``_OVERLAP_LONG_MAX_SEGMENT[(num_sms, delta)]``, or its separate launches would not take the warp-specialised
    attention (``_warp_specialised``)."""
    if int(plan.padded_tokens) <= _PREPROCESS_MAX_TOKENS:
        return False
    limit = _OVERLAP_LONG_MAX_SEGMENT.get((int(num_sms), bool(delta)))
    if limit is None:
        return False
    bounds = [int(b) for b in plan.bounds]
    squares = sum((b - a) ** 2 for a, b in zip(bounds[:-1], bounds[1:], strict=False))
    return squares <= limit * bounds[-1] or not _warp_specialised(
        plan, num_sms, delta, False
    )


def minimax_h3_sm120_nvfp4_launch_grids(
    num_sms: int,
    *,
    num_heads: int,
    num_stats_tiles: Optional[int] = None,
    q_slots: Optional[int] = None,
    padded_tokens: Optional[int] = None,
    num_units: Optional[int] = None,
) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
    """The ``(fused pre-processing, statistics, quantizer, attention)`` launch grids of the NVFP4 route
    on a device of ``num_sms`` SMs, all grid-stride or persistent:

    * fused pre-processing: ``max(1, min(max(num_stats_tiles, padded_tokens // 32) * num_heads,
      3 * num_sms))`` -- every CTA co-resident, within the kernel's launch bound of 3 CTAs per SM;
    * statistics: ``max(1, min((1 + q_slots) * num_stats_tiles * num_heads, 4 * num_sms))`` over the
      K item and the ``q_slots`` Q items of every (statistics tile, head);
    * quantizer: ``max(1, min((padded_tokens // 32) * num_heads, 8 * num_sms))`` over the
      (32-token block, head) items;
    * attention: one persistent CTA per SM up to the unit count, ``max(1, min(num_sms,
      max(num_units, 1)))`` (the plan's own grid).

    A component whose facts are ``None`` is ``None``."""
    sms, heads = int(num_sms), int(num_heads)
    if sms <= 0 or heads <= 0:
        raise ValueError(
            f"num_sms and num_heads must be positive, got {num_sms}, {num_heads}"
        )
    blocks = None if padded_tokens is None else int(padded_tokens) // _QUANT_TOKENS
    preprocess = (
        None
        if num_stats_tiles is None or blocks is None
        else max(
            1,
            min(
                max(int(num_stats_tiles), blocks) * heads,
                _PREPROCESS_CTAS_PER_SM * sms,
            ),
        )
    )
    stats = (
        None
        if num_stats_tiles is None or q_slots is None
        else max(
            1,
            min(
                (1 + int(q_slots)) * int(num_stats_tiles) * heads,
                _STATS_CTAS_PER_SM * sms,
            ),
        )
    )
    quantize = (
        None
        if blocks is None
        else max(1, min(blocks * heads, _QUANT_CTAS_PER_SM * sms))
    )
    attention = None if num_units is None else max(1, min(sms, max(int(num_units), 1)))
    return preprocess, stats, quantize, attention


def minimax_h3_sm120_nvfp4_overlap_grids(
    num_sms: int, *, num_units: int, padded_tokens: Optional[int] = None
) -> Optional[Tuple[int, int, int]]:
    """The ``(lead attention, producers, tail attention)`` launch grids of the overlapped form on a device of
    ``num_sms`` SMs, or ``None`` when that SM count has no overlap width ``P`` (170 SMs: 42, 188 SMs: 46;
    for a plan of more than 16384 padded tokens 64 and 46).  The tuple keeps this order; the launch order is
    producers, lead, tail (``minimax_h3_sm120_nvfp4_programs``).  The lead attention takes one persistent CTA
    per SM on all but ``P`` SMs up to the unit count, ``lead = max(1, min(num_sms - P, num_units))``; the
    producers ``min(3 * P, num_sms - lead)`` CTAs, one per SM, each running three workers (``P`` CTAs when
    the lead fills its SMs); the tail attention ``P`` CTAs.  Both attention launches take the units from one
    shared counter, so the tail CTAs run the units the lead CTAs have not taken."""
    sms = int(num_sms)
    if sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    width = _OVERLAP_WIDTH.get(sms)
    if width is None:
        return None
    if padded_tokens is not None and int(padded_tokens) > _PREPROCESS_MAX_TOKENS:
        width = _OVERLAP_WIDTH_LONG.get(sms, width)
    lead = max(1, min(sms - width, int(num_units)))
    return lead, min(_OVERLAP_PRODUCER_WORKERS * width, sms - lead), width


@functools.cache
def _nvfp4_target(index: int) -> str:
    """The exact target (``"sm_120a"``) every NVFP4 program ships for device ``index``."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    target = PREPROCESS.target_for((capability[0], capability[1]))
    if target is None or any(
        target not in program.targets for program in _NVFP4_PROGRAMS
    ):
        raise RuntimeError(
            "minimax_h3_sm120_varlen_attention_nvfp4 requires exact compute capability 12.0 "
            f"(SM120), got {capability[0]}.{capability[1]}"
        )
    return target


_NVFP4_ENTRIES: Dict[Tuple[str, int], Any] = {}


def _nvfp4_entry(target: str, slot: int) -> Any:
    """The ``run`` entry of ``_NVFP4_PROGRAMS[slot]`` on ``target`` (built on first use)."""
    entry = _NVFP4_ENTRIES.get((target, slot))
    if entry is None:
        entry = _NVFP4_ENTRIES[(target, slot)] = _NVFP4_PROGRAMS[slot].entry(target)
    return entry


def _nvfp4_buffers(
    plan, num_stats_tiles: int, owner: Tuple[int, ...]
) -> Tuple[torch.Tensor, ...]:
    """The grow-only workspace buffers a call of ``plan`` needs from the sets of its workspace owner
    ``owner`` (``_owner``): the quantized operands, scales and statistics (``_workspace``), then the
    arrival counters, the grid-barrier word and the overlapped form's head-control words
    (``_zero_stream_workspace``), which start zeroed and are reset by the kernels."""
    tokens, heads, hd = plan.total_tokens, plan.num_heads, MINIMAX_H3_HEAD_DIM
    qtiles = max(1, plan.num_tiles)
    n_rows4 = tokens * heads * _ROW_BYTES4
    n_sf = heads * plan.num_kblocks * _SCALE_TILE_BYTES
    return (
        _workspace(owner, "q4", n_rows4, torch.uint8),
        _workspace(owner, "k4", n_rows4, torch.uint8),
        _workspace(owner, "q_sf", tokens * heads * _BLOCKS_PER_ROW, torch.uint8),
        _workspace(owner, "k_sf", n_sf, torch.uint8),
        _workspace(owner, "qm4", qtiles * heads * _ROW_BYTES4, torch.uint8),
        _workspace(owner, "qm_sf", qtiles * heads * _BLOCKS_PER_ROW, torch.uint8),
        _workspace(owner, "vt4", heads * hd * (plan.padded_tokens // 2), torch.uint8),
        _workspace(owner, "v_sf", n_sf, torch.uint8),
        _workspace(owner, "q_mean", qtiles * heads * hd, torch.float32),
        _workspace(owner, "mean_k", plan.num_segments * heads * hd, torch.float32),
        _workspace(
            owner, "partials", max(1, num_stats_tiles * heads * hd), torch.float32
        ),
        _zero_stream_workspace(
            owner, "counters", max(1, plan.num_segments) * heads, torch.uint32
        ),
        _zero_stream_workspace(owner, "barrier", 4, torch.uint32),
        _zero_stream_workspace(
            owner,
            "hctr",
            _HCTR_WORDS_PER_HEAD * heads + _HCTR_TAIL_WORDS,
            torch.uint32,
        ),
    )


class _Nvfp4Launch:
    """One plan's launches of one route on one set of workspace buffers: ``args`` are the custom op's
    arguments from ``tok_seg`` through ``attention_grid``, every tensor among them sliced to the plan's
    exact size (the bindings encode their tensor maps from the shapes handed in) and already converted to
    a tvm-ffi tensor, so a call converts nothing but Q, K, V and the output."""

    __slots__ = ("args",)
    args: Tuple[Any, ...]

    def __init__(
        self,
        plan,
        delta: bool,
        buffers: Tuple[torch.Tensor, ...],
        index: int,
        target: str,
    ):
        tokens, heads, hd = plan.total_tokens, plan.num_heads, MINIMAX_H3_HEAD_DIM
        device = torch.device("cuda", index)
        tile_rows, num_stats_tiles, unit_rows = _plan_rows(plan, device)
        qtiles = max(1, plan.num_tiles)
        n_rows4 = tokens * heads * _ROW_BYTES4
        n_q_sf = tokens * heads * _BLOCKS_PER_ROW
        n_sf = heads * plan.num_kblocks * _SCALE_TILE_BYTES
        n_qm4 = qtiles * heads * _ROW_BYTES4
        n_qm_sf = qtiles * heads * _BLOCKS_PER_ROW
        n_vt4 = heads * hd * (plan.padded_tokens // 2)
        n_q_mean = qtiles * heads * hd
        n_mean_k = plan.num_segments * heads * hd
        n_partials = max(1, num_stats_tiles * heads * hd)
        n_counters = max(1, plan.num_segments) * heads
        n_hctr = _HCTR_WORDS_PER_HEAD * heads + _HCTR_TAIL_WORDS
        sizes = (
            n_rows4,
            n_rows4,
            n_q_sf,
            n_sf,
            n_qm4,
            n_qm_sf,
            n_vt4,
            n_sf,
            n_q_mean,
            n_mean_k,
            n_partials,
            n_counters,
            4,
            n_hctr,
        )
        (
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
            hctr,
        ) = (buffer[:size] for buffer, size in zip(buffers, sizes, strict=True))
        q_slots = _q_slots(plan.bounds)
        sms = _sm_count(index)
        programs = minimax_h3_sm120_nvfp4_programs(plan, sms, delta=delta)
        slots = tuple(_NVFP4_PROGRAMS.index(program) for program in programs)
        for slot in slots:
            _nvfp4_entry(target, slot)  # build / load before the first launch of a call
        preprocess_grid, stats_grid, quantize_grid, attention_grid = (
            minimax_h3_sm120_nvfp4_launch_grids(
                sms,
                num_heads=heads,
                num_stats_tiles=num_stats_tiles,
                q_slots=q_slots,
                padded_tokens=plan.padded_tokens,
                num_units=plan.num_units,
            )
        )
        fused = programs[0] is PREPROCESS
        overlap = any(programs[0] is producers for producers in _NVFP4_PRODUCERS)
        tail, tail_grid, units_per_head = -1, 1, 0
        if overlap:
            # (producers, lead attention, tail attention), the launch order: the producers take the
            # pre-processing slot, the lead the attention slot; the attention CTAs take the head-major units
            grids = minimax_h3_sm120_nvfp4_overlap_grids(
                sms, num_units=plan.num_units, padded_tokens=plan.padded_tokens
            )
            assert grids is not None
            attention_grid, preprocess_grid, tail_grid = grids
            tail = slots[2]
            slots = slots[:2]
            unit_rows = _plan_rows_head_major(plan, device)
            units_per_head = plan.num_units // heads
            fused = True
        u32 = torch.uint32
        tensors = (
            _token_segment_ids(plan.bounds, plan.padded_tokens, device),
            plan.cu_seqlens,
            plan.seg_tile_begin,
            tile_rows,
            unit_rows,
            q4.view(u32),
            k4.view(u32),
            q_sf,
            k_sf,
            qm4.view(u32),
            qm_sf,
            vt4.view(u32),
            v_sf,
            q_mean,
            mean_k,
            partials,
            counters,
            barrier,
            hctr,
            q4.view(tokens, heads, _ROW_BYTES4),
            k4.view(tokens, heads, _ROW_BYTES4),
            k_sf.view(-1, _SCALE_ROW_BYTES),
            vt4.view(heads, hd, plan.padded_tokens // 2),
            v_sf.view(-1, _SCALE_ROW_BYTES),
            q_sf.view(u32),
            qm_sf.view(u32),
        )
        self.args = (
            *(tvm_ffi.from_dlpack(t) for t in tensors),
            num_stats_tiles,
            q_slots,
            plan.padded_tokens,
            plan.num_kblocks,
            plan.num_units,
            slots[0],
            -1 if fused else slots[1],
            slots[-1],
            preprocess_grid if fused else stats_grid,
            quantize_grid,
            attention_grid,
            tail,
            tail_grid,
            units_per_head,
        )


# Per workspace owner (``_owner``): the workspace buffers the cached launches slice (the launches hold views of
# that owner's buffers) and the launches per (plan, route).  A grown buffer (of this route or of
# ``pv_mode="fp8"``, which shares the names) or a re-created set drops every launch of the owner; the most
# recent ``_MAX_WORKSPACE_SETS`` owners and ``_MAX_PLANS`` launches per owner are kept.
_NVFP4_LAUNCHES: "OrderedDict[Tuple[int, ...], Tuple[Tuple[torch.Tensor, ...], OrderedDict]]" = OrderedDict()


def _nvfp4_launch(
    plan, delta: bool, owner: Tuple[int, ...], target: str
) -> _Nvfp4Launch:
    index = owner[0]
    _, num_stats_tiles, _ = _plan_rows(plan, torch.device("cuda", index))
    buffers = _nvfp4_buffers(plan, num_stats_tiles, owner)
    cached = _owner_entry(_NVFP4_LAUNCHES, owner, lambda: (buffers, OrderedDict()))
    if any(a is not b for a, b in zip(cached[0], buffers, strict=True)):
        cached = _NVFP4_LAUNCHES[owner] = (buffers, OrderedDict())
    launches = cached[1]
    launch_key = (plan.key, delta)
    launch = launches.get(launch_key)
    if launch is None:
        launch = _Nvfp4Launch(plan, delta, buffers, index, target)
        while len(launches) >= _MAX_PLANS:
            launches.popitem(last=False)
        launches[launch_key] = launch
    return launch


# (device index, lead attention slot) of every overlapped form whose three programs have run once.
_NVFP4_OVERLAP_LOADED: set = set()


def _load_overlap(
    target: str,
    slots: Tuple[int, int, int],
    producer_args: Tuple[Any, ...],
    attention_args: Tuple[Any, ...],
    hctr: Any,
    tokens: int,
    num_kblocks: int,
    softmax_scale_log2: float,
) -> None:
    """Run the overlapped form's three programs once with no work, in the form's launch order, before its
    first overlapped call on a device.

    A function's first launch loads its module, which waits for the device; this keeps that stall out of
    the first overlapped call.  It only avoids the stall: correctness does not depend on it (every CTA of
    the form waits only on CTAs of an earlier launch of its own call, so a module load only delays a call).
    The instance has one head, one producer CTA, one lead and one tail attention CTA, no unit and no item,
    and runs on the call's own head-control words ``hctr``, which it leaves zero: the producers' last
    worker and the lead CTA each add one to the lead-done word, which the tail waits for and resets.  Only
    for an eager call, under the call's stream lock (nothing runs while a graph is captured)."""
    producers, lead, tail = slots
    # producers: no statistics tile and no token block, one head, one CTA
    _nvfp4_entry(target, producers)(
        *producer_args, hctr, 0, tokens, 0, 1, num_kblocks, 1, 1, 1
    )
    # lead, then tail attention: no unit, one head, one lead CTA and one tail CTA
    for program in (lead, tail):
        _nvfp4_entry(target, program)(
            *attention_args,
            0,
            1,
            num_kblocks,
            softmax_scale_log2,
            hctr,
            1,
            1,
            1,
            1,
            1,
            1,
        )


# One custom op serves both routes: the attention program slot it is handed selects the compensated or the
# uncompensated attention.
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
        "hctr",
    ),
)
def _minimax_h3_sm120_varlen_attention_nvfp4_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    tok_seg: Any,
    cu_seqlens: Any,
    seg_tile_begin: Any,
    tile_table: Any,
    unit_table: Any,
    q4: Any,
    k4: Any,
    q_sf: Any,
    k_sf: Any,
    qm4: Any,
    qm_sf: Any,
    vt4: Any,
    v_sf: Any,
    q_mean: Any,
    mean_k: Any,
    partials: Any,
    counters: Any,
    barrier: Any,
    hctr: Any,
    q4_rows: Any,
    k4_rows: Any,
    k_sf_rows: Any,
    vt4_rows: Any,
    v_sf_rows: Any,
    q_sf_words: Any,
    qm_sf_words: Any,
    num_stats_tiles: int,
    q_slots: int,
    padded_tokens: int,
    num_kblocks: int,
    num_units: int,
    first: int,
    second: int,
    attention: int,
    first_grid: int,
    second_grid: int,
    attention_grid: int,
    tail: int,
    tail_grid: int,
    units_per_head: int,
    softmax_scale_log2: float,
    stream: int,
    target: str,
) -> None:
    # Two or three programmatic-dependent launches on ``stream`` (the current torch stream): the fused
    # pre-processing (``second < 0``), or the statistics (which also quantize Q) and the K / V quantizer;
    # then the attention of either route.  ``first`` / ``second`` / ``attention`` index
    # ``_NVFP4_PROGRAMS``.  The bindings take the host arguments in the kernels' signature order followed
    # by the grid triple; the attention binding encodes its five tensor maps from the byte tensors handed
    # in its descriptor positions (``*_rows``, views of the quantized operands and scales).
    # The overlapped form (``tail >= 0``), in launch order: the head-major producers (``first``, a plain
    # launch over ``first_grid`` CTAs), the lead attention (``attention``, ``attention_grid`` CTAs, a
    # programmatic dependent of the producers) and the tail attention (``tail``, ``tail_grid`` CTAs, a
    # programmatic dependent of the lead), sharing the head-control words ``hctr``.
    tokens, heads = int(q.shape[0]), int(q.shape[1])
    with tvm_ffi.use_raw_stream(_ffi_device(q.device.index), stream):
        if tail >= 0:
            attention_args = (
                q4_rows,
                k4_rows,
                k_sf_rows,
                vt4_rows,
                v_sf_rows,
                out,
                q_sf_words,
                qm4,
                qm_sf_words,
                seg_tile_begin,
                unit_table,
            )
            producer_args = (
                q,
                k,
                v,
                tile_table,
                seg_tile_begin,
                tok_seg,
                cu_seqlens,
                partials,
                counters,
                mean_k,
                q_mean,
                q4,
                k4,
                q_sf,
                k_sf,
                qm4,
                qm_sf,
                vt4,
                v_sf,
            )
            key = (q.device.index, attention)
            if key not in _NVFP4_OVERLAP_LOADED and not _capturing(
                q.device.index, stream
            ):
                _load_overlap(
                    target,
                    (first, attention, tail),
                    producer_args,
                    attention_args,
                    hctr,
                    tokens,
                    num_kblocks,
                    softmax_scale_log2,
                )
                _NVFP4_OVERLAP_LOADED.add(key)
            _nvfp4_entry(target, first)(
                *producer_args,
                hctr,
                num_stats_tiles,
                tokens,
                padded_tokens,
                heads,
                num_kblocks,
                first_grid,
                1,
                1,
            )
            _nvfp4_entry(target, attention)(
                *attention_args,
                num_units,
                heads,
                num_kblocks,
                softmax_scale_log2,
                hctr,
                units_per_head,
                attention_grid,
                tail_grid,
                attention_grid,
                1,
                1,
            )
            _nvfp4_entry(target, tail)(
                *attention_args,
                num_units,
                heads,
                num_kblocks,
                softmax_scale_log2,
                hctr,
                units_per_head,
                attention_grid,
                tail_grid,
                tail_grid,
                1,
                1,
            )
            return
        if second < 0:
            _nvfp4_entry(target, first)(
                q,
                k,
                v,
                tile_table,
                seg_tile_begin,
                tok_seg,
                cu_seqlens,
                partials,
                counters,
                mean_k,
                q_mean,
                q4,
                k4,
                q_sf,
                k_sf,
                qm4,
                qm_sf,
                vt4,
                v_sf,
                barrier,
                num_stats_tiles,
                tokens,
                padded_tokens,
                heads,
                num_kblocks,
                first_grid,
                1,
                1,
            )
        else:
            _nvfp4_entry(target, first)(
                q,
                k,
                tile_table,
                seg_tile_begin,
                partials,
                counters,
                mean_k,
                q_mean,
                q4,
                q_sf,
                qm4,
                qm_sf,
                num_stats_tiles,
                heads,
                q_slots,
                first_grid,
                1,
                1,
            )
            _nvfp4_entry(target, second)(
                q,
                k,
                v,
                tok_seg,
                cu_seqlens,
                seg_tile_begin,
                mean_k,
                q_mean,
                q4,
                k4,
                q_sf,
                k_sf,
                qm4,
                qm_sf,
                vt4,
                v_sf,
                tokens,
                padded_tokens,
                heads,
                num_kblocks,
                second_grid,
                1,
                1,
            )
        _nvfp4_entry(target, attention)(
            q4_rows,
            k4_rows,
            k_sf_rows,
            vt4_rows,
            v_sf_rows,
            out,
            q_sf_words,
            qm4,
            qm_sf_words,
            seg_tile_begin,
            unit_table,
            num_units,
            heads,
            num_kblocks,
            softmax_scale_log2,
            attention_grid,
            1,
            1,
        )


@register_fake_op("flashinfer::minimax_h3_sm120_varlen_attention_nvfp4")
def _minimax_h3_sm120_varlen_attention_nvfp4_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    tok_seg: Any,
    cu_seqlens: Any,
    seg_tile_begin: Any,
    tile_table: Any,
    unit_table: Any,
    q4: Any,
    k4: Any,
    q_sf: Any,
    k_sf: Any,
    qm4: Any,
    qm_sf: Any,
    vt4: Any,
    v_sf: Any,
    q_mean: Any,
    mean_k: Any,
    partials: Any,
    counters: Any,
    barrier: Any,
    hctr: Any,
    q4_rows: Any,
    k4_rows: Any,
    k_sf_rows: Any,
    vt4_rows: Any,
    v_sf_rows: Any,
    q_sf_words: Any,
    qm_sf_words: Any,
    num_stats_tiles: int,
    q_slots: int,
    padded_tokens: int,
    num_kblocks: int,
    num_units: int,
    first: int,
    second: int,
    attention: int,
    first_grid: int,
    second_grid: int,
    attention_grid: int,
    tail: int,
    tail_grid: int,
    units_per_head: int,
    softmax_scale_log2: float,
    stream: int,
    target: str,
) -> None:
    pass


def _run_nvfp4(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    plan,
    delta: bool,
    softmax_scale: float,
) -> torch.Tensor:
    """The NVFP4 body of both routes: under the (device, stream) lock (``_stream_lock``), the plan's
    cached launch on the workspace set of the call's owner (``_owner``), then the two or three launches
    (the overlapped form's first eager call on a device also runs its no-work instance first)."""
    index = _device_index(q.device)
    target = _nvfp4_target(index)
    stream = _current_stream(index)
    with _stream_lock(index, stream):
        _minimax_h3_sm120_varlen_attention_nvfp4_impl(
            q,
            k,
            v,
            out,
            *_nvfp4_launch(plan, delta, _owner(index, stream), target).args,
            _softmax_scale_log2(softmax_scale),
            stream,
            target,
        )
    return out


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
    scale per 16 keys, E2M1 codes); the output is rounded to BF16 once.  Runtime-variable
    generated programs serve any ``tokens`` and any segment lengths (including empty segments and
    lengths below one tile), two or three programmatic-dependent launches per call on the current
    stream (``minimax_h3_sm120_nvfp4_programs``; grids ``minimax_h3_sm120_nvfp4_launch_grids``): a
    plan of 2048 to 16384 padded tokens runs the fused pre-processing (K sums and Q block means, a
    grid barrier, then the quantized operands, in one cooperative launch); any other plan runs the
    statistics launch, which also quantizes Q and the block-mean rows, then the K / V quantizer.  On a
    part of 170 or 188 SMs every plan of at least 2048 padded tokens runs the overlapped form instead
    (``minimax_h3_sm120_nvfp4_overlap_grids``): head-major pre-processing producers start first on a
    few SMs (a plain launch), the attention launch, a programmatic dependent of the producers, starts on
    the remaining SMs, each head's attention waiting for its operands, and a second attention launch, a
    programmatic dependent of the first, takes the remaining work on the producers' SMs as they drain;
    the outputs are bitwise the same.
    The attention launch takes the warp-specialised form on parts of at least 188 SMs when the mean
    segment (empty segments included) is at least 1536 and below 16384 tokens.  The quantized
    operands live in a grow-only workspace per (device, stream), and per capture for a call captured
    into a CUDA graph (``workspace_bytes_nvfp4``), whose buffer names are shared with
    ``pv_mode="fp8"``.  The programs ship for ``sm_120a`` alone: a call with a non-empty segment on a
    device of another compute capability than 12.0 raises ``RuntimeError``.

    Concurrency: safe on any number of streams, host threads, CUDA graphs and MPS clients. Exec graphs
    instantiated more than once from one captured graph (or cloned as child graphs) share this op's
    workspace and must not run concurrently.

    CUDA graphs: the first call for a given ``cu_seqlens_host`` builds the plan's device tables from host
    lists, a pageable host-to-device copy that a stream capture does not allow. Call the op once for that
    plan outside the capture before capturing it; later calls reuse the tables, and the capture then
    records kernel launches only. This holds with ``CUDA_MODULE_LOADING=LAZY`` as well: once the tables
    exist, the first launch of each program may happen inside the capture.

    The E2M1 scores and probabilities carry a larger quantization error than the FP8 operator
    (about 4x the relative L2 error on Gaussian inputs, the same as SageAttention3's own); this
    route is an experimental precision/latency trade-off and is validated against the FP32 oracle
    with the FP4 block-scaled tolerance ``atol = 1.0, rtol = 0.1`` (see the tests), not the FP8
    operator's ``0.1``.

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
        statistics launch are private to the (device, stream), and to the capture for a call captured
        into a CUDA graph.  Any other value is a ``ValueError``.

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
    return _run_nvfp4(q, k, v, out, plan, True, float(softmax_scale))


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
    * latency: the attention program differs (``ATTENTION_NODELTA`` / ``ATTENTION_WS_NODELTA``, which
      skip the compensation term); its warp-specialised form is taken from a mean segment of 1536 tokens
      with no upper bound, and on parts of at least 170 SMs also for a plan of 2048 to 16384 padded tokens;
      on parts of 188 SMs the overlapped form's two attention launches are warp-specialised too
      (``ATTENTION_WS_NODELTA_LEAD`` / ``ATTENTION_WS_NODELTA_TAIL``), behind the producers
      ``PREPROCESS_HEAD_MAJOR_KA`` (the outputs of ``PREPROCESS_HEAD_MAJOR``; a K item waits for its
      head's K sums in a second warp).

    Use the default route unless the extra error has been validated for the model in question.  Inputs,
    output, ``cu_seqlens`` handling, workspace and caching are exactly those of the default route.

    Concurrency: safe on any number of streams, host threads, CUDA graphs and MPS clients. Exec graphs
    instantiated more than once from one captured graph (or cloned as child graphs) share this op's
    workspace and must not run concurrently.

    CUDA graphs: the first call for a given ``cu_seqlens_host`` builds the plan's device tables from host
    lists, a pageable host-to-device copy that a stream capture does not allow. Call the op once for that
    plan outside the capture before capturing it; later calls reuse the tables, and the capture then
    records kernel launches only. This holds with ``CUDA_MODULE_LOADING=LAZY`` as well: once the tables
    exist, the first launch of each program may happen inside the capture.

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
    return _run_nvfp4(q, k, v, out, plan, False, float(softmax_scale))


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_NUM_HEADS",
    "minimax_h3_sm120_nvfp4_fp8pv_launch_grids",
    "minimax_h3_sm120_nvfp4_fp8pv_programs",
    "minimax_h3_sm120_nvfp4_launch_grids",
    "minimax_h3_sm120_nvfp4_overlap_grids",
    "minimax_h3_sm120_nvfp4_programs",
    "minimax_h3_sm120_varlen_attention_nvfp4",
    "minimax_h3_sm120_varlen_attention_nvfp4_nodelta",
    "workspace_bytes_nvfp4",
    "workspace_bytes_nvfp4_fp8pv",
]
