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

Cake backend: MiniMax-H3 packed-varlen noncausal attention (SM100/SM103).

Inputs are packed THD ``[T, H, 128]`` BF16 ``query``/``key``/``value`` with an
int32 ``cu_seqlens`` of length ``B + 1``.  For every segment ``[a, b)`` and head
``h``: ``O[a:b, h] = softmax(Q[a:b, h] @ K[a:b, h]^T * scale) @ V[a:b, h]`` with
FP32 softmax statistics, FP32 accumulation and one BF16 rounding.  Segments
never attend across boundaries, empty segments produce no rows and there is no
causal mask.

Two program families share this package:

* **BF16** (``variant="bf16"``): one persistent 2-CTA-cluster tcgen05 kernel
  (``kind::f16`` QK and PV).  The host derives a *segment plan* from
  ``cu_seqlens`` and ``num_heads``: ``seg_begin[s]``, ``seg_len[s]`` (empty
  segments dropped) and a *unit table* with two int32 per persistent-grid
  slot -- the segment index and ``head << 16 | cluster_in_segment`` -- where a
  unit is one head x one 512-row Q cluster (four 128-row Q tiles) of one
  segment.  Units are enumerated segment-major (heads slow, clusters fast) and
  placed into the statically strided slots (slot ``k * G + i`` is the ``k``-th
  unit of cluster ``i``, ``G = min(num_SMs / 2, total_tiles)``) by a
  longest-processing-time-first assignment over the per-unit cost
  ``ceil(seg_len / 128) + 2`` K/V blocks (``assign_unit_slots``).  When the
  unit count leaves a partial wave on the persistent grid, the planner splits
  the tail units into near-equal K/V block ranges (``choose_kv_splits``: the
  ``U mod G`` most expensive units ``k`` ways, chosen on the simulated LPT
  makespan plus combine cost); a split unit writes its rows normalized by its
  own softmax sum as FP16 plus FP32 ``(scaled log2 max, sum)`` into a partial
  slot of the plan's workspace, and the ``combine`` stage (one warp per output
  row) merges the slots with the exact FlashAttention formula into the BF16
  output.  The unit table therefore carries four int32 per slot (segment,
  ``head << 16 | cluster``, ``kv_block_begin << 16 | kv_blocks``, partial slot
  or -1) and the plan a ``combine_table`` (segment, ``head << 16 | cluster``,
  first partial slot, splits).
* **NVFP4** (``variant="nvfp4_fp4pv" | "nvfp4_fp8pv"``): one fused in-pipeline
  quantizer launch writes Q, K and V into a head-major, per-segment
  128-token-padded *packed layout* (``PB = sum(ceil(len / 128))`` packed
  blocks; grid ``heads * PB * 4`` CTAs, one per 32-token slice of a packed
  block), then one persistent 2-CTA-cluster kernel runs ``kind::mxf4nvf4`` QK
  (E2M1 x E2M1, UE4M3 block-16 scales) and either ``kind::mxf4nvf4`` PV (E2M1
  P anchored at the score-tile maximum, per-16-token UE4M3 V^T scales) or
  ``kind::f8f6f4`` PV (E4M3 P in TMEM, dense E4M3 V with one per-tensor scale
  ``448 / amax(V)``; the ``amax`` is one torch reduction before the quantizer
  launch).  The host builds the per-block tables consumed by the quantizer
  (``block_token``, ``block_valid``) and, per ``(cu_seqlens, heads)``, the
  five per-tile scheduler tables consumed by the attention kernel
  (``cl_head``, ``cl_seg_begin``, ``cl_seg_len``, ``cl_kv_base``,
  ``cl_q_block``; :func:`build_tile_tables`).  The attention launch is a
  programmatic dependent launch (PDL): the quantizer signals
  ``griddepcontrol.launch_dependents`` at its end and the attention prologue
  waits with ``griddepcontrol.wait`` after its barrier/TMEM setup.  The PDL
  launch attribute is baked into the generated attention host binding, so the
  runner only has to enqueue both launches on the same stream.  The same
  K/V-split planner applies: the per-tile tables gain ``cl_kv_begin``,
  ``cl_kv_blocks`` and ``cl_ws_slot``, split units park FP16 partials in the
  tile tables' workspace and the ``combine`` stage finishes them after the
  attention launch (an ordinary serial launch on the same stream).  Two
  attention programs are registered per architecture -- the dense
  ``attention`` (the first delivery's kernel: no K/V-split code, no split
  parameters) and ``attention_split`` (six more parameters: K/V range,
  partial slots, partial workspace, unit count) -- and the runner binds the
  one the plan needs with that program's parameter set
  (:func:`nvfp4_attention_stage`).

See ``README.md`` in this package and flashinfer-ai/flashinfer#4532.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import torch
import tvm_ffi

from .cake_jit import (
    MODULES,
    STAGES,
    load_cake_minimax_h3_varlen_attention_module,
    route_available,
    select_route,
)

HEAD_DIM = 128
BLOCK_M = 128  # Q rows per tile (both families)
CLUSTER_Q_BLOCKS = 4  # Q tiles per 2-CTA cluster tile (both families)
CLUSTER_Q_ROWS = CLUSTER_Q_BLOCKS * BLOCK_M
BLOCK_N = 128  # K/V rows per pipeline block (BF16 unit-cost unit)
# The BF16 unit table packs ``head << 16 | cluster_in_segment`` into one int32.
BF16_MAX_HEADS = 1 << 15
BF16_MAX_SEGMENT_CLUSTERS = 1 << 16
# Fixed per-unit overhead (Q staging, pipeline fill, epilogue) in K/V-block
# units for the longest-processing-time-first slot assignment.
BF16_UNIT_OVERHEAD_BLOCKS = 2
# K/V-split planner (mirrors the Cake production planner ``choose_kv_splits``):
# a unit is split into at most ``MAX_KV_SPLITS`` near-equal K/V block ranges;
# the cost model is in K/V-block units (combine launch + per-slot traffic) and
# a split is only taken below ``KV_SPLIT_MAX_WAVES`` waves when it beats the
# unsplit makespan by more than ``KV_SPLIT_MIN_GAIN``.
UNIT_WORDS = 4
COMBINE_WORDS = 4
PARTIAL_ROWS = CLUSTER_Q_ROWS  # FP16 rows per partial slot (one cluster's Q rows)
MAX_KV_SPLITS = 8
MAX_KV_BLOCKS = 1 << 16
COMBINE_FIXED_BLOCKS = 3.0
COMBINE_BLOCKS_PER_SLOT = 0.02
KV_SPLIT_MIN_GAIN = 0.03
KV_SPLIT_MAX_WAVES = 4
# Per-unit cost of the split-capable NVFP4 attention program relative to the
# dense program, per (pv_mode, arch), measured with both programs on the same
# plan: the sm_103a fp8pv split program runs 1.21x slower per unit on a
# 33k-token plan and an effective 1.30-1.42x on the 48-58 block units of the
# partial-wave rows (its block loop is rescheduled at the softmax register
# budget; the per-unit prologue/epilogue is a larger share of a short unit),
# so it carries 1.35: the 1.42-wave row stays dense, the 1.14-wave row still
# splits. The sm_100a fp4pv split program costs 1.03x, the other two
# combinations are at parity. A split
# plan runs the split program on every unit, so the planner scales the split
# candidates' makespan by this cost (mirrors the production planner's
# ``SPLIT_PROGRAM_COST``).
SPLIT_PROGRAM_COST: dict[tuple[str, str], float] = {
    ("fp8", "sm_103a"): 1.35,
    ("fp4", "sm_100a"): 1.03,
}
# Combine kernel: 128 threads = one warp per output row, four rows per CTA.
COMBINE_THREADS = 128
COMBINE_ROWS_PER_CTA = COMBINE_THREADS // 32
COMBINE_CTAS_PER_UNIT = PARTIAL_ROWS // COMBINE_ROWS_PER_CTA
SF_VEC = 16  # NVFP4 scale block along head_dim
# The fused NVFP4 quantizer runs one 256-thread CTA per 32-token slice of a
# packed 128-token block: grid = heads * PB * QUANTIZE_SUBS_PER_BLOCK.
QUANTIZE_SUB_TOKENS = 32
QUANTIZE_SUBS_PER_BLOCK = BLOCK_M // QUANTIZE_SUB_TOKENS
FP8_E4M3_MAX = 448.0
PV_MODES = ("fp4", "fp8")
NVFP4_VARIANT = {"fp4": "nvfp4_fp4pv", "fp8": "nvfp4_fp8pv"}
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

QK_MMA_DTYPE = {
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
    "nvfp4_fp8pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
}
PV_MMA_DTYPE = {
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales (tile-max P scale)",
    "nvfp4_fp8pv": "f8f6f4 e4m3 x e4m3, per-tensor v_scale",
}

# Exact keyword sets of the generated ``run`` entries (bound by the export's
# argument plans); ``grid`` is expanded to ``grid_x/y/z``.
BF16_ATTENTION_KWARGS = (
    "Q",
    "Q_raw",
    "K",
    "V",
    "O",
    "seg_begin",
    "seg_len",
    "unit_table",
    "partial_O",
    "partial_ML",
    "total_tiles",
    "num_heads",
    "softmax_scale_log2",
    "grid",
)
# K/V-split combine stage shared by all three program families.
COMBINE_KWARGS = (
    "partial_O",
    "partial_ML",
    "combine_table",
    "seg_begin",
    "seg_len",
    "O",
    "num_heads",
    "grid",
)
# Fused single-launch quantizers (``minimax_h3_varlen_nvfp4_quantize_qkv`` for
# fp4 PV, ``minimax_h3_varlen_nvfp4_quantize_qk_fp8v`` for fp8 PV).
QUANTIZE_COMMON_KWARGS = (
    "q",
    "k",
    "v",
    "q_fp4",
    "k_fp4",
    "q_scale",
    "k_scale",
)
QUANTIZE_TAIL_KWARGS = ("block_token", "block_valid", "heads", "PB", "grid")
QUANTIZE_QKV_KWARGS = (
    QUANTIZE_COMMON_KWARGS
    + ("v_fp4_t", "v_scale_lo", "v_scale_hi")
    + QUANTIZE_TAIL_KWARGS
)
QUANTIZE_QK_FP8V_KWARGS = (
    QUANTIZE_COMMON_KWARGS + ("v_fp8", "v_amax") + QUANTIZE_TAIL_KWARGS
)
# The dense ``attention`` program takes the five scheduler tables (the first
# delivery's kernel signature); the ``attention_split`` program adds the K/V
# range and partial-slot tables, the partial workspace and the unit count.
NVFP4_ATTENTION_COMMON_KWARGS = (
    "Q",
    "K",
    "SFQ",
    "SFK",
    "O",
    "cl_head",
    "cl_seg_begin",
    "cl_seg_len",
    "cl_kv_base",
    "cl_q_block",
    "total_clusters",
    "heads",
    "PB",
    "softmax_scale_log2",
    "grid",
)
NVFP4_ATTENTION_SPLIT_KWARGS = (
    "cl_kv_begin",
    "cl_kv_blocks",
    "cl_ws_slot",
    "partial_O",
    "partial_ML",
    "num_tiles",
)
NVFP4_ATTENTION_FP4PV_KWARGS = NVFP4_ATTENTION_COMMON_KWARGS + (
    "Vt",
    "SFVtLo",
    "SFVtHi",
)
NVFP4_ATTENTION_FP8PV_KWARGS = NVFP4_ATTENTION_COMMON_KWARGS + ("V", "v_amax")
NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS = (
    NVFP4_ATTENTION_COMMON_KWARGS
    + NVFP4_ATTENTION_SPLIT_KWARGS
    + ("Vt", "SFVtLo", "SFVtHi")
)
NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS = (
    NVFP4_ATTENTION_COMMON_KWARGS + NVFP4_ATTENTION_SPLIT_KWARGS + ("V", "v_amax")
)


# ---------------------------------------------------------------------------
# cu_seqlens handling and segment plans
# ---------------------------------------------------------------------------


def normalize_cu_seqlens(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    total_tokens: Optional[int] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> tuple[int, ...]:
    """Validate ``cu_seqlens`` and return its host copy as a tuple of ints.

    When ``cu_seqlens`` is a CUDA tensor and no ``cu_seqlens_host`` is given,
    the values are copied to the host (one synchronization).  Pass the host
    copy explicitly to keep preparation free of device synchronization.
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
    if total_tokens is not None and bounds[-1] != total_tokens:
        raise ValueError(
            f"cu_seqlens[-1]={bounds[-1]} must equal the THD token extent {total_tokens}"
        )
    return bounds


def _table(values: list[int], device: torch.device) -> torch.Tensor:
    # Empty tables keep one zero so the kernel always receives a valid pointer.
    return torch.tensor(values if values else [0], dtype=torch.int32, device=device)


@dataclass(frozen=True)
class BF16SegmentPlan:
    """Host segment plan of the BF16 kernel (non-empty segments only).

    A *unit* is one head x one cluster tile (512 consecutive Q rows of one
    segment); ``total_tiles = num_heads * total_clusters`` units run on a
    persistent grid of ``num_clusters`` 2-CTA clusters (``2 * num_clusters``
    CTAs).  ``unit_table`` holds ``UNIT_WORDS`` int32 per persistent-grid slot --
    the segment index, ``head << 16 | cluster_in_segment``,
    ``kv_block_begin << 16 | kv_blocks`` and the partial slot (-1 for an
    unsplit unit) -- in slot order: unit ``u`` of the table runs on cluster
    ``u % num_clusters`` as its ``u // num_clusters``-th unit (see
    :func:`assign_unit_slots`).  ``total_tiles`` counts scheduled units
    (split ranges included); ``combine_table`` holds ``COMBINE_WORDS`` int32
    per split (segment, head, cluster) unit and ``partial_O`` / ``partial_ML``
    are that plan's FP16 / FP32 partial workspace.
    """

    cu_seqlens: tuple[int, ...]
    num_heads: int
    num_segments: int
    total_clusters: int
    total_tiles: int
    num_clusters: int
    seg_begin: torch.Tensor
    seg_len: torch.Tensor
    unit_table: torch.Tensor
    combine_table: torch.Tensor
    partial_O: torch.Tensor
    partial_ML: torch.Tensor
    num_partial_slots: int
    num_combine_units: int
    max_kv_splits: int


def bf16_grid_clusters(device: torch.device) -> int:
    """Persistent-grid capacity of ``device`` in 2-CTA clusters (``num_SMs / 2``)."""
    return max(1, torch.cuda.get_device_properties(device).multi_processor_count // 2)


def assign_unit_slots(unit_costs: Sequence[int], num_clusters: int) -> list[int]:
    """Longest-processing-time-first placement of units into persistent-grid slots.

    Slot ``k * num_clusters + i`` is the ``k``-th unit of cluster ``i``.  The
    partial tail round receives the cheapest units, and within every full
    round the clusters that also own a tail unit receive that round's cheapest
    units.  Returns ``slot -> unit`` (a permutation).  Equal costs keep the
    enumeration order within a round (both sorts are stable), so a uniform
    plan reproduces the plain enumeration table exactly: consecutive slots
    stream one head's K/V.  Mirrors the production planner's
    ``assign_unit_slots``.
    """
    total = len(unit_costs)
    if total == 0:
        return []
    G = min(num_clusters, total)
    order = sorted(range(total), key=lambda u: -unit_costs[u])
    tail = total % G
    slots = [0] * total
    full_units = order[: total - tail]
    for k in range(len(full_units) // G):
        chunk = full_units[
            k * G : (k + 1) * G
        ]  # descending cost, ties in enumeration order
        for i, unit in enumerate(
            sorted(chunk, key=lambda u: unit_costs[u])
        ):  # ascending, stable
            slots[k * G + i] = unit
    for i, unit in enumerate(order[total - tail :] if tail else []):
        slots[(total // G) * G + i] = unit
    return slots


def split_chunks(blocks: int, splits: int) -> list[tuple[int, int]]:
    """Partition ``blocks`` K/V blocks into ``min(splits, blocks)`` near-equal ``(begin, count)`` ranges."""
    splits = max(1, min(int(splits), blocks))
    base, rem = divmod(blocks, splits)
    chunks: list[tuple[int, int]] = []
    begin = 0
    for j in range(splits):
        count = base + (1 if j < rem else 0)
        chunks.append((begin, count))
        begin += count
    return chunks


def lpt_makespan(costs: Sequence[float], num_clusters: int) -> float:
    """Makespan (max per-cluster load) of :func:`assign_unit_slots` for ``costs``."""
    if not costs:
        return 0.0
    G = min(num_clusters, len(costs))
    loads = [0.0] * G
    for slot, unit in enumerate(assign_unit_slots([int(c) for c in costs], G)):
        loads[slot % G] += costs[unit]
    return max(loads)


def choose_kv_splits(
    unit_blocks: Sequence[int],
    num_clusters: int,
    *,
    force: Optional[int] = None,
    program_cost: float = 1.0,
) -> list[int]:
    """Per-unit K/V split factors of one plan from the simulated makespan.

    ``force`` applies one factor to every unit.  Otherwise, for plans below
    ``KV_SPLIT_MAX_WAVES`` waves, the ``U mod G`` most expensive units (and,
    as the fallback, every unit) are split ``k`` ways for ``k`` in
    ``2..MAX_KV_SPLITS``; each candidate's longest-processing-time-first slot
    assignment is simulated, scaled by ``program_cost`` (the per-unit cost of
    the split-capable attention program relative to the dense one, see
    ``SPLIT_PROGRAM_COST``) and the combine cost added, and the best candidate
    is kept only when it beats the unsplit makespan by more than
    ``KV_SPLIT_MIN_GAIN``.  Mirrors the Cake production planner's
    ``choose_kv_splits``.
    """
    U = len(unit_blocks)
    if U == 0:
        return []
    if force is not None:
        return [max(1, min(int(force), int(b))) for b in unit_blocks]
    ones = [1] * U
    G = max(1, int(num_clusters))
    if U >= KV_SPLIT_MAX_WAVES * G:
        return ones
    order = sorted(range(U), key=lambda u: -unit_blocks[u])

    def total(split_of: list[int]) -> float:
        costs: list[float] = []
        slots = 0
        for u in range(U):
            chunks = split_chunks(int(unit_blocks[u]), split_of[u])
            if len(chunks) > 1:
                slots += len(chunks)
            costs.extend(count + BF16_UNIT_OVERHEAD_BLOCKS for _, count in chunks)
        combine = (
            COMBINE_FIXED_BLOCKS + COMBINE_BLOCKS_PER_SLOT * slots if slots else 0.0
        )
        span = lpt_makespan(costs, G)
        if slots:
            span *= program_cost
        return span + combine

    base = total(ones)
    tail = U % G
    candidates_n = {U}
    if tail:
        candidates_n.add(tail)
    best, best_split = base, ones
    for n in sorted(candidates_n):
        for k in range(2, MAX_KV_SPLITS + 1):
            split_of = ones[:]
            for u in order[:n]:
                split_of[u] = k
            t = total(split_of)
            if t < best:
                best, best_split = t, split_of
    if best > base * (1.0 - KV_SPLIT_MIN_GAIN):
        return ones
    return best_split


def _partial_workspace(
    partial_slots: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    slots = max(int(partial_slots), 1)
    return (
        torch.empty(
            slots * PARTIAL_ROWS * HEAD_DIM, dtype=torch.float16, device=device
        ),
        torch.empty(slots * PARTIAL_ROWS * 2, dtype=torch.float32, device=device),
    )


def build_bf16_segment_plan(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    device: torch.device,
    num_heads: int,
    *,
    num_clusters: Optional[int] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    kv_splits: Optional[int] = None,
) -> BF16SegmentPlan:
    """Build the BF16 segment plan (tables and partial workspace on ``device``).

    ``num_clusters`` is the persistent-grid capacity in 2-CTA clusters; it
    defaults to ``bf16_grid_clusters(device)`` (``device`` must then be a CUDA
    device).  ``kv_splits`` forces one K/V split factor on every unit
    (``None``: :func:`choose_kv_splits`).  The plan reproduces the Cake
    production planner's ``build_segment_plan`` table for table.
    """
    bounds = normalize_cu_seqlens(cu_seqlens, cu_seqlens_host=cu_seqlens_host)
    num_heads = int(num_heads)
    if not 0 < num_heads < BF16_MAX_HEADS:
        raise ValueError(f"num_heads must be in [1, {BF16_MAX_HEADS}), got {num_heads}")
    if kv_splits is not None and not 1 <= int(kv_splits) <= MAX_KV_SPLITS:
        raise ValueError(f"kv_splits must be in [1, {MAX_KV_SPLITS}], got {kv_splits}")
    if num_clusters is None:
        num_clusters = bf16_grid_clusters(device)
    segments = [(a, b - a) for a, b in zip(bounds, bounds[1:], strict=False) if b > a]
    begins = [a for a, _ in segments]
    lens = [length for _, length in segments]
    clusters = [(length + CLUSTER_Q_ROWS - 1) // CLUSTER_Q_ROWS for length in lens]
    if any(c >= BF16_MAX_SEGMENT_CLUSTERS for c in clusters):
        raise ValueError(
            f"a segment needs fewer than {BF16_MAX_SEGMENT_CLUSTERS} clusters"
        )
    blocks = [(length + BLOCK_N - 1) // BLOCK_N for length in lens]
    if any(b >= MAX_KV_BLOCKS for b in blocks):
        raise ValueError(f"a segment needs fewer than {MAX_KV_BLOCKS} K/V blocks")
    unit_blocks = [
        blocks[seg]
        for seg in range(len(lens))
        for _ in range(num_heads * clusters[seg])
    ]
    split_of = choose_kv_splits(unit_blocks, int(num_clusters), force=kv_splits)
    units: list[tuple[int, int, int, int, int, int]] = []
    costs: list[int] = []
    combine: list[int] = []
    partial_slots = 0
    unit_index = 0
    for seg, seg_clusters in enumerate(clusters):
        for head in range(num_heads):
            for c in range(seg_clusters):
                chunks = split_chunks(blocks[seg], split_of[unit_index])
                unit_index += 1
                if len(chunks) == 1:
                    units.append((seg, head, c, 0, blocks[seg], -1))
                    costs.append(blocks[seg] + BF16_UNIT_OVERHEAD_BLOCKS)
                    continue
                combine.extend((seg, (head << 16) | c, partial_slots, len(chunks)))
                for begin, count in chunks:
                    units.append((seg, head, c, begin, count, partial_slots))
                    costs.append(count + BF16_UNIT_OVERHEAD_BLOCKS)
                    partial_slots += 1
    total_tiles = len(units)
    num_clusters = min(int(num_clusters), max(total_tiles, 1))
    table: list[int] = []
    for unit in assign_unit_slots(costs, num_clusters):
        seg, head, c, begin, count, slot = units[unit]
        table.extend((seg, (head << 16) | c, (begin << 16) | count, slot))
    partial_O, partial_ML = _partial_workspace(partial_slots, device)
    return BF16SegmentPlan(
        cu_seqlens=bounds,
        num_heads=num_heads,
        num_segments=len(segments),
        total_clusters=sum(clusters),
        total_tiles=total_tiles,
        num_clusters=num_clusters,
        seg_begin=_table(begins, device),
        seg_len=_table(lens, device),
        unit_table=torch.tensor(
            table or [0] * UNIT_WORDS, dtype=torch.int32, device=device
        ),
        combine_table=torch.tensor(
            combine or [0] * COMBINE_WORDS, dtype=torch.int32, device=device
        ),
        partial_O=partial_O,
        partial_ML=partial_ML,
        num_partial_slots=partial_slots,
        num_combine_units=len(combine) // COMBINE_WORDS,
        max_kv_splits=max(split_of, default=1),
    )


def combine_kwargs(
    partial_O: torch.Tensor,
    partial_ML: torch.Tensor,
    combine_table: torch.Tensor,
    seg_begin: torch.Tensor,
    seg_len: torch.Tensor,
    out: torch.Tensor,
    num_heads: int,
    num_combine_units: int,
) -> dict[str, Any]:
    """Keyword bindings of the ``combine`` stage (``COMBINE_CTAS_PER_UNIT`` CTAs per split unit)."""
    kwargs = dict(
        partial_O=partial_O,
        partial_ML=partial_ML,
        combine_table=combine_table,
        seg_begin=seg_begin,
        seg_len=seg_len,
        O=out,
        num_heads=int(num_heads),
        grid=(max(int(num_combine_units), 1) * COMBINE_CTAS_PER_UNIT, 1, 1),
    )
    assert tuple(kwargs) == COMBINE_KWARGS
    return kwargs


@dataclass(frozen=True)
class PackedSegmentPlan:
    """Host plan of the NVFP4 packed layout (independent of heads and PV mode).

    Segment ``s`` (non-empty segments only) owns ``seg_blocks[s] =
    ceil(len / 128)`` packed 128-token blocks starting at packed block
    ``seg_tile_base[s]``; ``PB = sum(seg_blocks)``.  Token ``t`` of segment
    ``s`` lands at packed row ``seg_tile_base[s] * 128 + (t - seg_begin[s])``
    of its head; rows past the segment inside its padded region are
    zero-filled with the minimum scale.  ``block_token[p]`` / ``block_valid[p]``
    drive the quantizer (first THD token and valid row count of packed block
    ``p``); the four per-cluster ``cl_*`` tables (cluster tile -> segment
    start token, segment length, first packed block of the segment, first Q
    block of the cluster inside the segment) are the source of the per-tile
    scheduler tables built by :func:`build_tile_tables` for one head count.
    """

    cu_seqlens: tuple[int, ...]
    total_tokens: int
    seg_begin: tuple[int, ...]
    seg_len: tuple[int, ...]
    seg_blocks: tuple[int, ...]
    seg_tile_base: tuple[int, ...]
    cluster_off: tuple[int, ...]
    PB: int
    total_clusters: int
    block_token: torch.Tensor
    block_valid: torch.Tensor
    cl_seg_begin: torch.Tensor
    cl_seg_len: torch.Tensor
    cl_kv_base: torch.Tensor
    cl_q_block: torch.Tensor

    @property
    def num_segments(self) -> int:
        return len(self.seg_len)


def build_packed_segment_plan(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    device: torch.device,
    *,
    cu_seqlens_host: Optional[Sequence[int]] = None,
) -> PackedSegmentPlan:
    bounds = normalize_cu_seqlens(cu_seqlens, cu_seqlens_host=cu_seqlens_host)
    seg_begin: list[int] = []
    seg_len: list[int] = []
    seg_blocks: list[int] = []
    seg_tile_base: list[int] = []
    cluster_off: list[int] = []
    block_token: list[int] = []
    block_valid: list[int] = []
    cl_seg_begin: list[int] = []
    cl_seg_len: list[int] = []
    cl_kv_base: list[int] = []
    cl_q_block: list[int] = []
    PB = 0
    clusters = 0
    for a, b in zip(bounds, bounds[1:], strict=False):
        length = b - a
        if length <= 0:
            continue
        blocks = (length + BLOCK_M - 1) // BLOCK_M
        seg_begin.append(a)
        seg_len.append(length)
        seg_blocks.append(blocks)
        seg_tile_base.append(PB)
        cluster_off.append(clusters)
        for blk in range(blocks):
            block_token.append(a + blk * BLOCK_M)
            block_valid.append(min(BLOCK_M, length - blk * BLOCK_M))
        n_clusters = (length + CLUSTER_Q_ROWS - 1) // CLUSTER_Q_ROWS
        for c in range(n_clusters):
            cl_seg_begin.append(a)
            cl_seg_len.append(length)
            cl_kv_base.append(PB)
            cl_q_block.append(CLUSTER_Q_BLOCKS * c)
        PB += blocks
        clusters += n_clusters
    cluster_off.append(clusters)
    return PackedSegmentPlan(
        cu_seqlens=bounds,
        total_tokens=bounds[-1],
        seg_begin=tuple(seg_begin),
        seg_len=tuple(seg_len),
        seg_blocks=tuple(seg_blocks),
        seg_tile_base=tuple(seg_tile_base),
        cluster_off=tuple(cluster_off),
        PB=PB,
        total_clusters=clusters,
        block_token=_table(block_token, device),
        block_valid=_table(block_valid, device),
        cl_seg_begin=_table(cl_seg_begin, device),
        cl_seg_len=_table(cl_seg_len, device),
        cl_kv_base=_table(cl_kv_base, device),
        cl_q_block=_table(cl_q_block, device),
    )


@dataclass(frozen=True)
class TileTables:
    """Per-unit scheduler tables of the NVFP4 attention kernel for one ``(plan, heads)``.

    ``total_tiles`` scheduled units (K/V split ranges included) per table.
    Unit ``t`` runs head ``cl_head[t]`` on the cluster tile described by the
    next four tables (segment start token, segment length, first packed block
    of the segment, first Q block of the cluster tile inside the segment) over
    K/V blocks ``cl_kv_begin[t] .. cl_kv_begin[t] + cl_kv_blocks[t]`` of the
    segment, writing partial slot ``cl_ws_slot[t]`` (-1: the BF16 output
    directly).  The persistent grid is statically strided: cluster ``i`` runs
    units ``i, i + G, ...`` with ``G = grid_x / 2``.  ``combine_table``,
    ``seg_begin`` / ``seg_len`` and the partial workspace feed the ``combine``
    stage.
    """

    heads: int
    total_tiles: int
    cl_head: torch.Tensor
    cl_seg_begin: torch.Tensor
    cl_seg_len: torch.Tensor
    cl_kv_base: torch.Tensor
    cl_q_block: torch.Tensor
    cl_kv_begin: torch.Tensor
    cl_kv_blocks: torch.Tensor
    cl_ws_slot: torch.Tensor
    combine_table: torch.Tensor
    seg_begin: torch.Tensor
    seg_len: torch.Tensor
    partial_O: torch.Tensor
    partial_ML: torch.Tensor
    num_partial_slots: int
    num_combine_units: int
    max_kv_splits: int

    NAMES = (
        "cl_head",
        "cl_seg_begin",
        "cl_seg_len",
        "cl_kv_base",
        "cl_q_block",
        "cl_kv_begin",
        "cl_kv_blocks",
        "cl_ws_slot",
    )


def build_tile_tables(
    plan: PackedSegmentPlan,
    heads: int,
    device: torch.device,
    *,
    num_clusters: Optional[int] = None,
    kv_splits: Optional[int] = None,
    program_cost: float = 1.0,
) -> TileTables:
    """Per-unit tables: segment-major enumeration, K/V splits, then LPT slot placement.

    Units are enumerated by segment in descending length (ties keep the
    segment order), then head, then the segment's cluster tiles in order --
    consecutive units share one head's K/V, which keeps the concurrent working
    set L2-resident -- and each unit is cut into its planned number of
    near-equal K/V block ranges (:func:`choose_kv_splits`; ``kv_splits``
    forces one factor).  The units are then placed into the persistent grid's
    statically strided slots longest-processing-time first
    (:func:`assign_unit_slots`; equal costs keep the enumeration order).  The
    kernel's five per-tile tables are allocated first, in the production
    planner's order, so their placement relative to the workspace matches it.
    ``num_clusters`` is the grid capacity in 2-CTA clusters (default
    ``bf16_grid_clusters(device)``, a CUDA device).  Mirrors the Cake
    production planner's ``build_tile_tables``.
    """
    heads = int(heads)
    if heads <= 0:
        raise ValueError("heads must be positive")
    if kv_splits is not None and not 1 <= int(kv_splits) <= MAX_KV_SPLITS:
        raise ValueError(f"kv_splits must be in [1, {MAX_KV_SPLITS}], got {kv_splits}")
    if num_clusters is None:
        num_clusters = bf16_grid_clusters(device)
    segments = sorted(range(plan.num_segments), key=lambda s: -plan.seg_len[s])
    cl = {
        name: getattr(plan, name).tolist()
        for name in ("cl_seg_begin", "cl_seg_len", "cl_kv_base", "cl_q_block")
    }
    seg_blocks = [(length + BLOCK_N - 1) // BLOCK_N for length in plan.seg_len]
    unit_blocks = [
        seg_blocks[s]
        for s in segments
        for _ in range(heads * (plan.cluster_off[s + 1] - plan.cluster_off[s]))
    ]
    split_of = choose_kv_splits(
        unit_blocks, int(num_clusters), force=kv_splits, program_cost=program_cost
    )
    units: list[
        tuple[int, int, int, int, int]
    ] = []  # (cluster tile, head, kv_begin, kv_blocks, slot)
    costs: list[int] = []
    combine: list[int] = []
    partial_slots = 0
    unit_index = 0
    for s in segments:
        for head in range(heads):
            for c in range(plan.cluster_off[s], plan.cluster_off[s + 1]):
                chunks = split_chunks(seg_blocks[s], split_of[unit_index])
                unit_index += 1
                if len(chunks) == 1:
                    units.append((c, head, 0, seg_blocks[s], -1))
                    costs.append(seg_blocks[s] + BF16_UNIT_OVERHEAD_BLOCKS)
                    continue
                combine.extend(
                    (
                        s,
                        (head << 16) | (c - plan.cluster_off[s]),
                        partial_slots,
                        len(chunks),
                    )
                )
                for begin, count in chunks:
                    units.append((c, head, begin, count, partial_slots))
                    costs.append(count + BF16_UNIT_OVERHEAD_BLOCKS)
                    partial_slots += 1
    num_tiles = len(units)
    slots = [
        units[u]
        for u in assign_unit_slots(costs, min(int(num_clusters), max(num_tiles, 1)))
    ]
    cl_head = _table([head for _, head, _, _, _ in slots], device)
    cl_tables = {
        name: _table([values[c] for c, _, _, _, _ in slots], device)
        for name, values in cl.items()
    }
    partial_O, partial_ML = _partial_workspace(partial_slots, device)
    return TileTables(
        heads=heads,
        total_tiles=num_tiles,
        cl_head=cl_head,
        cl_kv_begin=_table([begin for _, _, begin, _, _ in slots], device),
        cl_kv_blocks=_table([count for _, _, _, count, _ in slots], device),
        cl_ws_slot=_table([slot for _, _, _, _, slot in slots], device),
        combine_table=torch.tensor(
            combine or [0] * COMBINE_WORDS, dtype=torch.int32, device=device
        ),
        seg_begin=_table(list(plan.seg_begin), device),
        seg_len=_table(list(plan.seg_len), device),
        partial_O=partial_O,
        partial_ML=partial_ML,
        num_partial_slots=partial_slots,
        num_combine_units=len(combine) // COMBINE_WORDS,
        max_kv_splits=max(split_of, default=1),
        **cl_tables,
    )


def nvfp4_quantize_grid(heads: int, PB: int) -> tuple[int, int, int]:
    """Launch grid of the fused quantizer: one CTA per 32-token slice of every packed block."""
    return (int(heads) * int(PB) * QUANTIZE_SUBS_PER_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Packed NVFP4 / FP8 operand workspace
# ---------------------------------------------------------------------------


def nvfp4_workspace_shapes(
    heads: int, PB: int, pv_mode: str
) -> dict[str, tuple[tuple[int, ...], torch.dtype]]:
    """Shapes and dtypes of the packed operand set for ``(heads, PB, pv_mode)``.

    ``PB`` is padded to at least one block so every buffer is non-empty.
    """
    if pv_mode not in PV_MODES:
        raise ValueError(f"pv_mode must be one of {PV_MODES}, got {pv_mode!r}")
    PB = max(int(PB), 1)
    rows = heads * PB * BLOCK_M
    shapes: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
        "q_fp4": ((rows, HEAD_DIM // 2), torch.uint8),
        "k_fp4": ((rows, HEAD_DIM // 2), torch.uint8),
        "q_scale": ((heads * PB * 32, 32), torch.uint8),
        "k_scale": ((heads * PB * 32, 32), torch.uint8),
    }
    if pv_mode == "fp4":
        shapes["v_fp4_t"] = ((heads * HEAD_DIM, PB * (BLOCK_M // 2)), torch.uint8)
        shapes["v_scale_lo"] = ((heads * PB * 16, 32), torch.uint8)
        shapes["v_scale_hi"] = ((heads * PB * 16, 32), torch.uint8)
    else:
        shapes["v_fp8"] = ((rows, HEAD_DIM), torch.uint8)
        # Two-stage max|V| reduction target: ``heads * HEAD_DIM`` partial maxima
        # (one per (head, dim) row of V viewed as [heads * HEAD_DIM, T]) then the
        # scalar. A single-output ATen reduction allocates its accumulation
        # buffer on every call; the two-stage form allocates nothing.
        shapes["v_amax_partial"] = ((heads * HEAD_DIM,), torch.float32)
        shapes["v_amax"] = ((1,), torch.float32)
    return shapes


def nvfp4_workspace_bytes(heads: int, PB: int, pv_mode: str) -> int:
    """Total bytes of the packed operand set (about 180 B per token-head)."""
    total = 0
    for shape, dtype in nvfp4_workspace_shapes(heads, PB, pv_mode).values():
        total += math.prod(shape) * torch.empty((), dtype=dtype).element_size()
    return total


def allocate_nvfp4_workspace(
    heads: int, PB: int, pv_mode: str, device: torch.device
) -> dict[str, torch.Tensor]:
    ws = {
        name: torch.empty(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in nvfp4_workspace_shapes(heads, PB, pv_mode).items()
    }
    for name in ("v_amax_partial", "v_amax"):
        if name in ws:
            ws[name].zero_()
    return ws


def reduce_v_amax(value: torch.Tensor, workspace: dict[str, torch.Tensor]) -> None:
    """``workspace["v_amax"][0] = max|value|`` with no device allocation.

    ``value`` is the contiguous BF16 ``[T, heads, HEAD_DIM]`` V operand; it is
    viewed as ``[heads * HEAD_DIM, T]`` (any row split of a contiguous tensor
    yields the same maximum), reduced per row into ``v_amax_partial`` and then
    to the scalar. The scalar never visits the host.
    """
    partial = workspace["v_amax_partial"]
    rows = int(partial.shape[0])
    flat = value.view(-1)
    if flat.numel() <= rows:
        # T == 1: a reduction over a size-1 dimension allocates in ATen; copy the
        # (at most ``rows``) values into the partial buffer elementwise instead.
        n = flat.numel()
        partial[:n].copy_(flat)
        partial[:n].abs_()
        partial[n:].zero_()
    else:
        torch.linalg.vector_norm(
            flat.view(rows, -1),
            ord=float("inf"),
            dim=1,
            dtype=torch.float32,
            out=partial,
        )
    torch.linalg.vector_norm(
        partial, ord=float("inf"), dim=0, keepdim=True, out=workspace["v_amax"]
    )


# ---------------------------------------------------------------------------
# Binding to the generated argument plans
# ---------------------------------------------------------------------------


def _arch_for(device: torch.device) -> str:
    capability = torch.cuda.get_device_capability(device)
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "MiniMax-H3 packed-varlen attention requires compute capability 10.0 or "
            f"10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(device: torch.device, variant: str = "bf16") -> bool:
    """True when this checkout registers ``variant`` for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(torch.cuda.get_device_capability(device))
    return arch is not None and route_available(variant, arch)


def _bind_stage(
    module_name: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name``."""
    record = MODULES[module_name]
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for kind, name in record["arg_plan"]:
        if kind == "grid":
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated module {module_name!r} expects argument {name!r} "
                f"({kind}); host binding provides {sorted(kwargs)}"
            )
    module = load_cake_minimax_h3_varlen_attention_module(module_name)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def _persistent_grid(
    device: torch.device, total_tiles: int, *, at_least_one: bool
) -> tuple[int, int, int]:
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    pairs = min(num_sms // 2, total_tiles)
    if at_least_one:
        pairs = max(1, pairs)
    return (2 * pairs, 1, 1)


def _check_thd(
    name: str,
    tensor: torch.Tensor,
    total_tokens: int,
    num_heads: int,
    device: torch.device,
) -> None:
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must live on {device}")
    if tensor.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be a bfloat16 tensor")
    if tuple(tensor.shape) != (total_tokens, num_heads, HEAD_DIM):
        raise ValueError(
            f"{name} must be [T, H, {HEAD_DIM}] = {(total_tokens, num_heads, HEAD_DIM)}, "
            f"got {tuple(tensor.shape)}"
        )
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def validate_minimax_h3_varlen_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: Optional[torch.Tensor],
) -> tuple[int, int, torch.device]:
    """Shape / dtype / device validation shared by both families.

    Returns ``(total_tokens, num_heads, device)``.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be THD [T, H, {HEAD_DIM}]")
    total_tokens, num_heads = int(query.shape[0]), int(query.shape[1])
    if num_heads <= 0:
        raise ValueError("query must have at least one head")
    device = query.device
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check_thd(name, tensor, total_tokens, num_heads, device)
    if out is not None:
        _check_thd("out", out, total_tokens, num_heads, device)
    return total_tokens, num_heads, device


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MiniMaxH3VarlenAttentionRunner:
    """Launch the prepared BF16 packed-varlen attention.

    Calling the runner or ``launch()`` writes the caller-owned ``out`` on the
    current stream with no CUDA allocation and no host synchronization and
    returns ``out``.  The segment plan is fixed at preparation; prepare a new
    runner when ``cu_seqlens``, shapes or tensor bindings change (values may
    change freely).  CUDA Graph capture belongs to the caller.
    """

    variant: str
    arch: str
    plan: BF16SegmentPlan
    main_kwargs: dict
    out: torch.Tensor
    _entry: Optional[Callable[..., Any]]
    _arguments: tuple
    _combine_entry: Optional[Callable[..., Any]] = None
    _combine_arguments: tuple = ()

    def launch(self) -> torch.Tensor:
        if self._entry is not None:  # zero tiles: nothing to launch
            # Tensor maps are encoded by the host binding and passed by value.
            with tvm_ffi.use_torch_stream():
                self._entry(*self._arguments)
                if self._combine_entry is not None:  # K/V-split units to merge
                    self._combine_entry(*self._combine_arguments)
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.main_kwargs["grid"][0])

    @property
    def route_metadata(self) -> dict[str, Any]:
        return dict(
            variant=self.variant,
            arch=self.arch,
            qk_mma_dtype=QK_MMA_DTYPE[self.variant],
            pv_mma_dtype=PV_MMA_DTYPE[self.variant],
            segment_count=self.plan.num_segments,
            cluster_tiles=self.plan.total_clusters,
            unit_count=self.plan.total_tiles,
            persistent_clusters=self.plan.num_clusters,
            kv_split_units=self.plan.num_combine_units,
            max_kv_splits=self.plan.max_kv_splits,
        )


@dataclass(frozen=True)
class MiniMaxH3VarlenNVFP4AttentionRunner:
    """Launch the prepared NVFP4 pipeline: quantize Q/K/V, then attend.

    ``launch()`` runs the complete pipeline (for ``pv_mode="fp8"`` one device
    ``amax(V)`` reduction into the workspace, then one fused quantizer launch,
    one attention launch and, for plans with K/V-split units, the ``combine``
    launch) into the caller-owned ``out`` with no CUDA allocation and no host
    synchronization.  ``quantize()`` and ``attention()`` run the stages
    separately so callers can time them on their own.  The attention launch
    uses one of two generated programs, chosen at preparation from the plan:
    the dense ``attention`` module for plans without split units and
    ``attention_split`` otherwise (``attention_stage``).  Packed operands live in ``workspace`` (allocated at
    preparation).  Prepare a new runner when ``cu_seqlens``, shapes or tensor
    bindings change; values may change freely since quantization is part of
    every launch.  CUDA Graph capture belongs to the caller.

    The attention binding is generated with the programmatic-dependent-launch
    attribute (``cudaLaunchAttributeProgrammaticStreamSerialization``): its
    prologue overlaps the quantizer tail and ``griddepcontrol.wait`` orders the
    packed-operand reads.  Both launches are enqueued on the current torch
    stream in order, which is all the handshake requires from the host.
    """

    variant: str
    pv_mode: str
    arch: str
    plan: PackedSegmentPlan
    tile_tables: TileTables
    workspace: dict[str, torch.Tensor]
    stage_kwargs: dict[str, dict]
    out: torch.Tensor
    _value: torch.Tensor
    _stages: tuple[tuple[str, Optional[Callable[..., Any]], tuple], ...]

    def _run(self, names: Sequence[str]) -> None:
        with tvm_ffi.use_torch_stream():
            for name, entry, arguments in self._stages:
                if name not in names or entry is None:
                    continue
                if name == "quantize" and self.pv_mode == "fp8":
                    reduce_v_amax(self._value, self.workspace)
                entry(*arguments)

    def quantize(self) -> None:
        self._run(("quantize",))

    def attention(self) -> torch.Tensor:
        # The bound attention program and, for K/V-split plans, the combine launch.
        self._run(tuple(s for s in STAGES[self.variant] if s != "quantize"))
        return self.out

    @property
    def attention_stage(self) -> str:
        """``"attention_split"`` for plans with K/V-split units, else ``"attention"``."""
        return nvfp4_attention_stage(self.tile_tables)

    def launch(self) -> torch.Tensor:
        self._run(STAGES[self.variant])
        return self.out

    __call__ = launch

    @property
    def num_ctas(self) -> int:
        return int(self.stage_kwargs["attention"]["grid"][0])

    @property
    def route_metadata(self) -> dict[str, Any]:
        return dict(
            variant=self.variant,
            pv_mode=self.pv_mode,
            arch=self.arch,
            qk_mma_dtype=QK_MMA_DTYPE[self.variant],
            pv_mma_dtype=PV_MMA_DTYPE[self.variant],
            segment_count=self.plan.num_segments,
            packed_blocks=self.plan.PB,
            cluster_tiles=self.plan.total_clusters,
            tile_count=self.tile_tables.total_tiles,
            kv_split_units=self.tile_tables.num_combine_units,
            max_kv_splits=self.tile_tables.max_kv_splits,
            attention_variant=self.attention_stage,
        )


def nvfp4_attention_stage(tiles: TileTables) -> str:
    """The attention program a plan binds: ``attention_split`` when the tile
    tables hold any K/V-split unit (partial slots), else the dense ``attention``.

    Both are the same generated program specialised at build time; the dense
    variant carries no K/V-range or partial-slot code, which keeps the softmax
    block loop of unsplit units at the schedule of the single-program kernel.
    """
    return "attention_split" if int(tiles.num_partial_slots) > 0 else "attention"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def prepare_minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> MiniMaxH3VarlenAttentionRunner:
    """Validate, plan and bind one BF16 packed-varlen attention problem.

    Every allocation happens here (only the optional output and the small
    int32 plan tables); the returned runner launches with none.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen attention supports backend='cake'")
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out
    )
    if isinstance(cu_seqlens, torch.Tensor) and (
        not cu_seqlens.is_cuda or cu_seqlens.device != device
    ):
        raise ValueError("cu_seqlens must be an int32 CUDA tensor on the query device")
    bounds = normalize_cu_seqlens(
        cu_seqlens, total_tokens=total_tokens, cu_seqlens_host=cu_seqlens_host
    )
    arch = _arch_for(device)
    route = select_route("bf16", arch)
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    if out is None:
        out = torch.empty(
            (total_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    plan = build_bf16_segment_plan(bounds, device, num_heads)
    total_tiles = int(plan.total_tiles)
    main_kwargs = dict(
        Q=query,
        Q_raw=query,
        K=key,
        V=value,
        O=out,
        seg_begin=plan.seg_begin,
        seg_len=plan.seg_len,
        unit_table=plan.unit_table,
        partial_O=plan.partial_O,
        partial_ML=plan.partial_ML,
        total_tiles=total_tiles,
        num_heads=num_heads,
        softmax_scale_log2=float(softmax_scale) / math.log(2.0),
        grid=(2 * int(plan.num_clusters), 1, 1),
    )
    assert tuple(main_kwargs) == BF16_ATTENTION_KWARGS
    entry: Optional[Callable[..., Any]] = None
    arguments: tuple = ()
    combine_entry: Optional[Callable[..., Any]] = None
    combine_arguments: tuple = ()
    if total_tiles > 0:
        entry, arguments = _bind_stage(route["modules"]["attention"], main_kwargs)
        if plan.num_combine_units > 0:
            combine_entry, combine_arguments = _bind_stage(
                route["modules"]["combine"],
                combine_kwargs(
                    plan.partial_O,
                    plan.partial_ML,
                    plan.combine_table,
                    plan.seg_begin,
                    plan.seg_len,
                    out,
                    num_heads,
                    plan.num_combine_units,
                ),
            )
    return MiniMaxH3VarlenAttentionRunner(
        "bf16",
        arch,
        plan,
        main_kwargs,
        out,
        entry,
        arguments,
        combine_entry,
        combine_arguments,
    )


def prepare_minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    workspace: Optional[dict[str, torch.Tensor]] = None,
    backend: str = "cake",
) -> MiniMaxH3VarlenNVFP4AttentionRunner:
    """Validate, plan and bind one NVFP4 packed-varlen attention pipeline.

    Every allocation happens here (the optional output, the int32 plan
    tables and, unless ``workspace`` is supplied, the packed operand set of
    :func:`nvfp4_workspace_shapes`); the returned runner launches with none.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen NVFP4 attention supports backend='cake'")
    if pv_mode not in PV_MODES:
        raise ValueError(f"pv_mode must be one of {PV_MODES}, got {pv_mode!r}")
    variant = NVFP4_VARIANT[pv_mode]
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out
    )
    if isinstance(cu_seqlens, torch.Tensor) and (
        not cu_seqlens.is_cuda or cu_seqlens.device != device
    ):
        raise ValueError("cu_seqlens must be an int32 CUDA tensor on the query device")
    bounds = normalize_cu_seqlens(
        cu_seqlens, total_tokens=total_tokens, cu_seqlens_host=cu_seqlens_host
    )
    arch = _arch_for(device)
    route = select_route(variant, arch)
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    if out is None:
        out = torch.empty(
            (total_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    plan = build_packed_segment_plan(bounds, device)
    if workspace is None:
        workspace = allocate_nvfp4_workspace(num_heads, plan.PB, pv_mode, device)
    else:
        expected = nvfp4_workspace_shapes(num_heads, plan.PB, pv_mode)
        for name, (shape, dtype) in expected.items():
            tensor = workspace.get(name)
            if (
                tensor is None
                or tuple(tensor.shape) != shape
                or tensor.dtype != dtype
                or tensor.device != device
                or not tensor.is_contiguous()
            ):
                raise ValueError(
                    f"workspace[{name!r}] must be a contiguous {dtype} tensor of shape {shape} on {device}"
                )
    PB = int(plan.PB)
    tiles = build_tile_tables(
        plan,
        num_heads,
        device,
        program_cost=SPLIT_PROGRAM_COST.get((pv_mode, arch), 1.0),
    )
    quantize_kwargs = dict(
        q=query,
        k=key,
        v=value,
        q_fp4=workspace["q_fp4"],
        k_fp4=workspace["k_fp4"],
        q_scale=workspace["q_scale"],
        k_scale=workspace["k_scale"],
    )
    if pv_mode == "fp4":
        quantize_kwargs.update(
            v_fp4_t=workspace["v_fp4_t"],
            v_scale_lo=workspace["v_scale_lo"],
            v_scale_hi=workspace["v_scale_hi"],
        )
    else:
        quantize_kwargs.update(v_fp8=workspace["v_fp8"], v_amax=workspace["v_amax"])
    quantize_kwargs.update(
        block_token=plan.block_token,
        block_valid=plan.block_valid,
        heads=num_heads,
        PB=PB,
        grid=nvfp4_quantize_grid(num_heads, PB),
    )
    assert tuple(quantize_kwargs) == (
        QUANTIZE_QKV_KWARGS if pv_mode == "fp4" else QUANTIZE_QK_FP8V_KWARGS
    )
    stage_kwargs: dict[str, dict] = {"quantize": quantize_kwargs}
    total_tiles = tiles.total_tiles
    attention_common = dict(
        Q=workspace["q_fp4"],
        K=workspace["k_fp4"],
        SFQ=workspace["q_scale"],
        SFK=workspace["k_scale"],
        O=out,
        cl_head=tiles.cl_head,
        cl_seg_begin=tiles.cl_seg_begin,
        cl_seg_len=tiles.cl_seg_len,
        cl_kv_base=tiles.cl_kv_base,
        cl_q_block=tiles.cl_q_block,
        total_clusters=int(plan.total_clusters),
        heads=num_heads,
        PB=PB,
        softmax_scale_log2=float(softmax_scale) / math.log(2.0),
        grid=_persistent_grid(device, total_tiles, at_least_one=True),
    )
    attention_split_extra = dict(
        cl_kv_begin=tiles.cl_kv_begin,
        cl_kv_blocks=tiles.cl_kv_blocks,
        cl_ws_slot=tiles.cl_ws_slot,
        partial_O=tiles.partial_O,
        partial_ML=tiles.partial_ML,
        num_tiles=total_tiles,
    )
    if pv_mode == "fp4":
        pv_operands = dict(
            Vt=workspace["v_fp4_t"],
            SFVtLo=workspace["v_scale_lo"],
            SFVtHi=workspace["v_scale_hi"],
        )
        dense_names, split_names = (
            NVFP4_ATTENTION_FP4PV_KWARGS,
            NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS,
        )
    else:
        pv_operands = dict(V=workspace["v_fp8"], v_amax=workspace["v_amax"])
        dense_names, split_names = (
            NVFP4_ATTENTION_FP8PV_KWARGS,
            NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS,
        )
    # The dense program takes the first delivery's parameter set, the split
    # program additionally the K/V range / partial-slot tables, the partial
    # workspace and the unit count; exactly one of them is bound.
    attention_kwargs = {**attention_common, **pv_operands}
    attention_split_kwargs = {
        **attention_common,
        **attention_split_extra,
        **pv_operands,
    }
    assert tuple(attention_kwargs) == dense_names
    assert tuple(attention_split_kwargs) == split_names
    stage_kwargs["attention"] = attention_kwargs
    stage_kwargs["attention_split"] = attention_split_kwargs
    stage_kwargs["combine"] = combine_kwargs(
        tiles.partial_O,
        tiles.partial_ML,
        tiles.combine_table,
        tiles.seg_begin,
        tiles.seg_len,
        out,
        num_heads,
        tiles.num_combine_units,
    )
    assert tuple(stage_kwargs) == STAGES[variant]
    attention_stage = nvfp4_attention_stage(tiles)
    stages: list[tuple[str, Optional[Callable[..., Any]], tuple]] = []
    for stage in STAGES[variant]:
        skipped = (
            total_tiles == 0
            or (stage == "combine" and tiles.num_combine_units == 0)
            or (stage in ("attention", "attention_split") and stage != attention_stage)
        )
        if skipped:
            stages.append((stage, None, ()))
            continue
        entry, arguments = _bind_stage(route["modules"][stage], stage_kwargs[stage])
        stages.append((stage, entry, arguments))
    return MiniMaxH3VarlenNVFP4AttentionRunner(
        variant,
        pv_mode,
        arch,
        plan,
        tiles,
        workspace,
        stage_kwargs,
        out,
        value,
        tuple(stages),
    )


# ---------------------------------------------------------------------------
# One-shot entry points (called by the thin core APIs)
# ---------------------------------------------------------------------------


def minimax_h3_varlen_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> torch.Tensor:
    runner = prepare_minimax_h3_varlen_attention(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend=backend,
    )
    return runner.launch()


def minimax_h3_varlen_nvfp4_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: torch.Tensor,
    *,
    pv_mode: str = "fp8",
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> torch.Tensor:
    runner = prepare_minimax_h3_varlen_nvfp4_attention(
        query,
        key,
        value,
        cu_seqlens,
        pv_mode=pv_mode,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend=backend,
    )
    return runner.launch()
