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
  ``448 / amax(V)``; the ``amax`` is one bandwidth-bound partial-max kernel
  (stage ``amax``, a fixed grid of ``AMAX_CTAS`` CTAs striding over V) whose
  per-CTA partials the quantizer folds after its own loads -- the quantizer
  is that kernel's programmatic dependent launch).  The host builds the
  per-block tables consumed by the quantizer
  (``block_token``, ``block_valid``; for ``pv_mode="fp4"`` on
  ``NVFP4_QUANTIZE_TP_ARCHES`` a plan of at most ``QUANTIZE_TP_MAX_PB``
  packed blocks binds the quantizer's ``quantize_tp`` build, which takes the
  two tables by value, :func:`nvfp4_quantize_stage`) and, per
  ``(cu_seqlens, heads)``, the five per-tile scheduler tables consumed by
  the attention kernel (``cl_head``, ``cl_seg_begin``, ``cl_seg_len``,
  ``cl_kv_base``, ``cl_q_block``; :func:`build_tile_tables`) and the unit
  count ``num_tiles``.  A unit with at most 256 valid Q rows from its first row
  runs one Q stage, and a two-stage unit of the partial tail round may be
  emitted as two 256-row single-stage halves (the BF16 tail-round Q-split);
  ``q_half`` in bits 16+ of ``cl_q_block`` carries the choice to every role of
  the kernel (0 two-stage, 1 / 2 single-stage on the tile's first / second 256
  rows); neither changes an output bit.  The dense program of
  ``NVFP4_TWO_STAGE_DENSE`` (sm_100a, fp8 PV) runs every unit on both Q
  stages and takes the cluster-tile count ``total_clusters`` instead of
  ``num_tiles``: its plans carry whole two-stage units only (``q_half`` 0, no
  halves).  On ``NVFP4_QS_DENSE`` (sm_100a, fp8 PV) a plan whose single-stage
  tables have tail-round Q-split units and no K/V-split units binds the
  single-stage dense program ``attention_qs`` with those tables instead
  (:func:`nvfp4_tile_tables`).  The attention launch is a
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
  ``attention`` (no K/V-split code, no split parameters) and
  ``attention_split`` (five more parameters: K/V range, partial slots,
  partial workspace) -- plus, on the fp8 route of ``NVFP4_QS_DENSE_ARCHES``,
  the single-stage dense ``attention_qs``, and the runner binds the
  one the plan needs with that program's parameter set
  (:func:`nvfp4_attention_stage`).

The BF16 route also carries several programs per architecture (round 13 of
the Cake kernel): three retraces of one kernel body with identical outputs --
``attention`` (the static persistent schedule), ``attention_dyn`` (the dynamic
whole-unit ticket scheduler: the first wave stays static, then the leader
CTA's load warp draws the cluster's next unit from a per-plan ticket counter
one unit ahead of its use; units are handed out in the LPT slot order, so
only a unit's cluster changes) and, on ``BF16_SREAD_ARCHES`` (sm_100a) only,
``attention_sread`` (the stage-0 cross-stage read wait issued before every P
store, no per-block guard on the publish tail).
:func:`build_bf16_segment_plan` decides from the plan numbers alone
(``plan.dyn_sched``: at least ``BF16_DYN_SCHED_MIN_WAVES`` waves and
``BF16_DYN_SCHED_MIN_CLUSTER_BLOCKS`` stage-weighted K/V blocks in every
cluster's static share, then at least ``BF16_DYN_SCHED_MIN_UNITS_PER_CLUSTER``
units per cluster or a static cluster imbalance above the cheapest unit;
``plan.sread_unified`` is its complement) and
:func:`prepare_minimax_h3_varlen_attention` binds that one program
(``runner.attention_stage``, ``route_metadata["attention_variant"]``).  Every
BF16 ``unit_table`` carries one more zero record after the ``num_clusters``
prefetch padding: word 0 is the ticket counter, which the kernel resets at
the end of every launch, so a cached plan needs no host memset.  The
``attention_dyn`` program also uses words 6 and 7 of padding record ``c``
(word 6 + ticket slot) as cluster ``c``'s two ticket mailboxes and leaves
its last tickets there, so the padding of a launched plan is launch state,
not zeros.  Every mailbox read follows the same launch's write, so a
relaunch needs no reset.  Launches of
one plan are stream-ordered (the plan cache keys the stream); never launch
one cached plan concurrently from two streams.

A plan whose every segment has at most ``BF16_TINY_SEGMENT_TOKENS`` tokens,
with at least one and at most ``BF16_TINY_MAX_SEGMENTS`` non-empty segments,
binds the fourth BF16 attention program instead, ``attention_tiny`` (on
``BF16_TINY_ARCHES``, sm_100a and sm_103a): one 4-warp CTA per (head, non-empty segment) on a
``(num_heads, segments, 1)`` grid, no cluster and no unit table; the segment
bounds go by value in the launch (``plan.tiny_cu_seqlens``, a CPU int32
tensor of ``BF16_TINY_MAX_SEGMENTS + 1`` entries that the binding copies into
the kernel parameters).  Its arithmetic is the other programs' one-K/V-block
unit, so its output is bitwise theirs.

See ``README.md`` in this package and flashinfer-ai/flashinfer#4532.
"""

from __future__ import annotations

import functools
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Union

import torch
import tvm_ffi

from .cake_jit import (
    BF16_SREAD_ARCHES,
    BF16_TINY_ARCHES,
    FP8_TINY_SEGMENT_TOKENS,
    MODULES,
    NVFP4_QS_DENSE_ARCHES,
    NVFP4_QUANTIZE_TP_ARCHES,
    STAGES,
    load_cake_minimax_h3_varlen_attention_module,
    route_available,
    route_program,
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
# Most recently used BF16 segment plans (device tables and the partial
# workspace), keyed by ``(cu_seqlens, device index, num_heads, kv_splits,
# q_split, stream)``.  The diffusion engine issues hundreds of calls per sample with
# one segment layout; a hit re-launches with the cached tables (no Python
# planning, no host-to-device copies, no allocation, no synchronization).
# Bounded: the oldest plan is evicted once the limit is reached, so varying
# segment layouts cannot grow device memory without bound.
BF16_PLAN_CACHE_CAPACITY = 256
# Cost of a single-stage unit (at most 256 valid Q rows: the kernel skips its
# second Q stage) relative to a two-stage unit, as a fraction NUM / DEN: half
# the MMA work, but the stage-0 softmax latency is no longer hidden behind
# the other stage's MMAs.
BF16_SINGLE_STAGE_COST_NUM = 3
BF16_SINGLE_STAGE_COST_DEN = 5
# Tail-round Q-split (mirrors the Cake production planner's ``Q_SPLIT_PLAN`` /
# ``choose_q_splits``): a two-stage unit of the partial tail round, where
# ``G - U mod G`` clusters would idle for a whole unit time, is emitted as two
# 256-row single-stage records (record word 5, ``q_half`` 1 / 2) when the
# simulated makespan improves by at least ``BF16_Q_SPLIT_MIN_GAIN``.  The bar
# is lower than the K/V split's because a half costs nothing beyond its own
# modelled single-stage inefficiency (no combine launch, no workspace).  The
# registered attention program decodes the word on both architectures (it is
# the Cake ``kernel_for_arch`` retrace traced with ``Q_SPLIT_DECODE``); plans
# built with ``q_split=False`` are the plain enumeration (word 5 is 0).
BF16_Q_SPLIT_PLAN = True
BF16_Q_SPLIT_MIN_GAIN = 0.005
# K/V-split planner (mirrors the Cake production planner ``choose_kv_splits``):
# a unit is split into at most ``MAX_KV_SPLITS`` near-equal K/V block ranges;
# the cost model is in K/V-block units (combine launch + per-slot traffic) and
# a split is only taken below ``KV_SPLIT_MAX_WAVES`` waves when it beats the
# unsplit makespan by more than ``KV_SPLIT_MIN_GAIN``.
# Eight int32 per unit record: segment token begin, segment length,
# ``head << 16 | cluster_in_segment``, ``kv_block_begin << 16 | kv_blocks``,
# partial slot, ``q_half`` and two reserved words; the table carries
# ``num_clusters`` zero records of padding (every kernel role prefetches the
# record of its next unit; words 6 and 7 of padding record ``c`` are the
# ``attention_dyn`` program's ticket mailboxes of cluster ``c``).
UNIT_WORDS = 8
COMBINE_WORDS = 4
# Per-plan BF16 program dispatch (mirrors the Cake production planner's
# ``dyn_sched_for_records`` / ``sread_unified_for_plan``, round 13): a plan
# binds the dynamic whole-unit ticket scheduler (``attention_dyn``) when it
# has at least ``BF16_DYN_SCHED_MIN_WAVES`` waves and every cluster's static
# share is at least ``BF16_DYN_SCHED_MIN_CLUSTER_BLOCKS`` stage-weighted K/V
# blocks (a two-stage unit counts its blocks twice) and then either the
# fullest cluster runs at least ``BF16_DYN_SCHED_MIN_UNITS_PER_CLUSTER`` units
# or the static LPT order is provably not makespan-optimal (its cluster
# imbalance ``W_max - W_min`` exceeds the cheapest unit's weight).  The ticket
# chain costs a fixed ~2.5 us at a cluster's first unit, so one-wave plans and
# plans of many small units keep the static schedule.  Every other plan
# records ``sread_unified`` and binds ``attention_sread`` on
# ``BF16_SREAD_ARCHES`` (``cake_jit``).
BF16_DYN_SCHED_MIN_WAVES = 2
BF16_DYN_SCHED_MIN_CLUSTER_BLOCKS = 8192
BF16_DYN_SCHED_MIN_UNITS_PER_CLUSTER = 64
# Tiny-plan BF16 program (``attention_tiny``): a plan whose every segment has
# at most ``BF16_TINY_SEGMENT_TOKENS`` tokens (the program's one 32-key block)
# and that has at most ``BF16_TINY_MAX_SEGMENTS`` non-empty segments (the
# length of its by-value bounds parameter, less one) binds it on
# ``BF16_TINY_ARCHES`` (``cake_jit``); any other plan, or any other
# architecture, binds the general programs.
BF16_TINY_SEGMENT_TOKENS = 32
BF16_TINY_MAX_SEGMENTS = 64
# Rows of one CTA's Q stage: a unit with at most this many valid rows from its
# first row runs a single stage (the kernel's record decode, ``two_stages``).
CTA_Q_ROWS = CLUSTER_Q_ROWS // 2
PARTIAL_ROWS = CLUSTER_Q_ROWS  # FP16 rows per partial slot (one cluster's Q rows)
MAX_KV_SPLITS = 8
MAX_KV_BLOCKS = 1 << 16
COMBINE_FIXED_BLOCKS = 3.0
COMBINE_BLOCKS_PER_SLOT = 0.02
KV_SPLIT_MIN_GAIN = 0.03
KV_SPLIT_MAX_WAVES = 4
# Per-unit cost of the split-capable NVFP4 attention program relative to the
# dense program, per (pv_mode, arch), measured with both programs on the same
# unsplit plans (same node). sm_103a fp8pv is at 1.00-1.01x since the fp8
# softmax body waits for the free P buffer after its exp2/pack block (before,
# ptxas shared that wait's scoreboard with an exp2 result in the split program
# and the first E4M3 pack waited for the previous PV drain: 1.21x per unit,
# carried as 1.35). The sm_100a fp4pv split program costs 1.03x, the other two
# combinations are at parity. A split plan runs the split program on every
# unit, so the planner scales the split candidates' makespan by this cost
# (mirrors the production planner's ``SPLIT_PROGRAM_COST``).
SPLIT_PROGRAM_COST: dict[tuple[str, str], float] = {
    ("fp8", "sm_103a"): 1.01,
    ("fp4", "sm_100a"): 1.03,
}
# NVFP4 single-stage units: a unit with at most ``CTA_Q_ROWS`` valid Q rows from
# its first row (``q_half`` 1 in bits 16+ of ``cl_q_block``), and either half of
# a tail-round Q-split unit (``q_half`` 1 / 2), runs its first softmax stage
# only (QK1, PV1 and the second stage's softmax and correction work are skipped
# by every role, which all branch on ``q_half == 0``).  Its LPT placement cost is ``ceil(blocks * NUM / DEN) + 2`` per
# ``(pv_mode, arch)`` (the single-stage K/V block period relative to the
# two-stage one; the BF16 value until measured).  The K/V-split decision does
# not see this discount, so it splits exactly the units it split before and
# every output row keeps its bits; only the placement and the Q halves move.
NVFP4_SINGLE_STAGE_COST: dict[tuple[str, str], tuple[int, int]] = {
    ("fp4", "sm_100a"): (BF16_SINGLE_STAGE_COST_NUM, BF16_SINGLE_STAGE_COST_DEN),
    ("fp8", "sm_100a"): (BF16_SINGLE_STAGE_COST_NUM, BF16_SINGLE_STAGE_COST_DEN),
    ("fp4", "sm_103a"): (BF16_SINGLE_STAGE_COST_NUM, BF16_SINGLE_STAGE_COST_DEN),
    ("fp8", "sm_103a"): (BF16_SINGLE_STAGE_COST_NUM, BF16_SINGLE_STAGE_COST_DEN),
}
# Tail-round Q-split of the NVFP4 plan (the BF16 rule, ``bf16_choose_q_splits``).
NVFP4_Q_SPLIT_PLAN = True
# ``cl_q_block`` word: bits 0-15 the cluster tile's first Q block inside its
# segment, bits 16+ ``q_half`` (0: two-stage whole tile; 1 / 2: single-stage on
# the tile's first / second 256 rows).
NVFP4_Q_HALF_SHIFT = 16
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
# fp8 PV: ``amax(V)`` is a fixed grid of AMAX_CTAS 256-thread CTAs, each
# striding over 32-byte (16 x BF16) vectors of the flat V tensor in chunks of
# AMAX_CHUNK vectors and writing one partial maximum; the quantizer folds the
# AMAX_CTAS partials (order independent, so the result is exact).
AMAX_CTAS = 512
AMAX_VECTOR = 16
AMAX_CHUNK = 256 * 4
PV_MODES = ("fp4", "fp8")
NVFP4_VARIANT = {"fp4": "nvfp4_fp4pv", "fp8": "nvfp4_fp8pv"}
# Stages of the quantization step (``runner.quantize()``); the rest is attention.
QUANTIZE_STAGES = ("amax", "quantize", "quantize_early_qk", "quantize_tp")
# The builds of the fused quantizer a route may carry; a plan binds one of them
# (:func:`nvfp4_quantize_stage`).
QUANTIZER_STAGES = ("quantize", "quantize_early_qk", "quantize_tp")
# ``quantize_early_qk`` is bound when the quantize grid fits in EARLY_QK_WAVES
# waves of EARLY_QK_CTAS_PER_SM resident CTAs per SM (its program: REG 32 x 256
# threads; a GPU test checks the built program against this constant).
EARLY_QK_WAVES = 2
EARLY_QK_CTAS_PER_SM = 8
# ``quantize_tp`` (fp4 PV, NVFP4_QUANTIZE_TP_ARCHES) takes ``block_token`` /
# ``block_valid`` by value, as int32 launch parameters of QUANTIZE_TP_MAX_PB
# entries: a plan of at most that many packed blocks binds it, with the tables
# zero-padded to that length (``PackedSegmentPlan.tp_block_token`` /
# ``tp_block_valid``, CPU tensors); a larger plan binds ``quantize``.
QUANTIZE_TP_MAX_PB = 128
QUANTIZE_TP_BY_VALUE = ("block_token", "block_valid")
SUPPORTED_COMPUTE_CAPABILITIES = {(10, 0): "sm_100a", (10, 3): "sm_103a"}

QK_MMA_DTYPE = {
    "fp8": "f8f6f4 e4m3 x e4m3, per-tensor q_descale * k_descale in the exponent",
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
    "nvfp4_fp8pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales",
}
PV_MMA_DTYPE = {
    "fp8": "f8f6f4 e4m3 P (unit scale) x e4m3, per-tensor v_descale in the epilogue",
    "bf16": "f16 bf16 x bf16",
    "nvfp4_fp4pv": "mxf4nvf4 e2m1 x e2m1, ue4m3 block-16 scales (tile-max P scale)",
    "nvfp4_fp8pv": "f8f6f4 e4m3 x e4m3, per-tensor v_scale",
}

# Exact keyword sets of the generated ``run`` entries (bound by the export's
# argument plans); ``grid`` is expanded to ``grid_x/y/z``.
BF16_ATTENTION_KWARGS = (
    "Q",
    "K",
    "V",
    "O",
    "O_raw",
    "unit_table",
    "partial_O",
    "partial_ML",
    "total_tiles",
    "num_heads",
    "softmax_scale_log2",
    "grid",
)
# The tiny-plan BF16 program: the operands, the contiguous output, the plan's
# by-value bounds (``BF16SegmentPlan.tiny_cu_seqlens``), the heads and the
# softmax scale; grid ``(num_heads, segments, 1)``.
BF16_TINY_ATTENTION_KWARGS = (
    "Q",
    "K",
    "V",
    "O",
    "cu_seqlens",
    "num_heads",
    "softmax_scale_log2",
    "grid",
)
# The all-FP8 attention program: the BF16 program's bindings plus the three
# per-tensor descales (one-element FP32 device tensors).
FP8_ATTENTION_KWARGS = (
    "Q",
    "K",
    "V",
    "O",
    "O_raw",
    "unit_table",
    "partial_O",
    "partial_ML",
    "q_descale",
    "k_descale",
    "v_descale",
    "total_tiles",
    "num_heads",
    "softmax_scale_log2",
    "grid",
)
# K/V-split combine stage shared by all program families.
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
    QUANTIZE_COMMON_KWARGS
    + ("v_fp8", "v_amax", "v_amax_partial")
    + QUANTIZE_TAIL_KWARGS
)
# fp8 PV only: the per-CTA ``max|V|`` partial kernel that precedes the quantizer.
AMAX_KWARGS = ("v", "v_amax_partial", "num_vectors", "num_chunks", "grid")
# The dense ``attention`` program takes the five scheduler tables and the unit
# count ``num_tiles``; the ``attention_split`` program adds the K/V range and
# partial-slot tables and the partial workspace.  The dense program of a
# ``NVFP4_TWO_STAGE_DENSE`` pair takes the plan's cluster-tile count
# ``total_clusters`` in place of ``num_tiles``, and the ``attention_qs`` program
# of a ``NVFP4_QS_DENSE`` pair the dense parameters (:func:`nvfp4_attention_kwargs`).
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
    "num_tiles",
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
# ``(pv_mode, arch)`` pairs whose dense ``attention`` program runs every unit on
# both Q stages: it takes ``total_clusters`` and runs ``heads * total_clusters``
# whole units, so a plan without K/V-split units binds plain tables there
# (``q_half`` 0 everywhere, whole-unit costs, no Q-split halves;
# :func:`build_tile_tables` ``two_stage_dense``).  Its split program, and every
# program of the other pairs, runs single-stage units and Q-split halves.  On
# sm_100a fp8 PV the single-stage form of the dense program costs more per
# unit than the placement it buys, except for the tail-round Q-split halves:
# a ``NVFP4_QS_DENSE`` pair binds the single-stage dense program
# ``attention_qs`` for a plan whose single-stage tables have Q-split units and
# no K/V-split units (:func:`nvfp4_tile_tables`), the two-stage ``attention``
# for every other plan without K/V-split units.
NVFP4_TWO_STAGE_DENSE: frozenset[tuple[str, str]] = frozenset({("fp8", "sm_100a")})
NVFP4_QS_DENSE: frozenset[tuple[str, str]] = frozenset(
    ("fp8", arch) for arch in NVFP4_QS_DENSE_ARCHES
)
# The attention stages of the NVFP4 routes; the runner binds one per plan.
NVFP4_ATTENTION_STAGES = ("attention", "attention_qs", "attention_split")
NVFP4_ATTENTION_TWO_STAGE_FP8PV_KWARGS = tuple(
    "total_clusters" if name == "num_tiles" else name
    for name in NVFP4_ATTENTION_FP8PV_KWARGS
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
    segment); the ``num_heads * total_clusters`` units run on a persistent
    grid of ``num_clusters`` 2-CTA clusters (``2 * num_clusters`` CTAs).
    ``unit_table`` holds ``UNIT_WORDS`` int32 per persistent-grid slot -- the
    segment's token begin and length, ``head << 16 | cluster_in_segment``,
    ``kv_block_begin << 16 | kv_blocks``, the partial slot (-1 for an unsplit
    unit), ``q_half`` (0 for a whole 512-row unit, 1 / 2 for the first / second
    256-row half of a tail-round Q-split unit, run as a single-stage unit) and
    two reserved words -- in slot order: unit ``u`` of the table runs on
    cluster ``u % num_clusters`` as its ``u // num_clusters``-th unit (see
    :func:`assign_unit_slots`), followed by ``num_clusters`` zero records (the
    kernel's one-unit-ahead record prefetch; the ``attention_dyn`` program uses
    words 6 and 7 of padding record ``c`` (word 6 + ticket slot) as cluster
    ``c``'s ticket mailboxes and leaves its last tickets there) and one more
    zero record whose word 0 is the
    dynamic scheduler's ticket counter (reset by the kernel at the end of
    every launch).  ``dyn_sched`` / ``sread_unified`` record which
    attention program the plan binds (``BF16_DYN_SCHED_MIN_WAVES`` and
    below; ``sread_unified`` applies on ``BF16_SREAD_ARCHES`` only, see
    :func:`bf16_attention_stage`).  ``total_tiles`` counts scheduled
    units (K/V ranges and Q-split halves included; ``num_q_split_units`` counts
    the halved units); ``combine_table`` holds ``COMBINE_WORDS`` int32
    per split (segment, head, cluster) unit and ``partial_O`` / ``partial_ML``
    are that plan's FP16 / FP32 partial workspace.  ``tiny_cu_seqlens`` is the
    ``attention_tiny`` program's by-value bounds (:func:`bf16_tiny_cu_seqlens`)
    when the plan qualifies for it, else ``None``; a plan that has them binds
    that program and leaves the tables above unused.
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
    num_q_split_units: int
    dyn_sched: bool
    sread_unified: bool
    tiny_cu_seqlens: Optional[torch.Tensor] = None


def bf16_tiny_cu_seqlens(bounds: Sequence[int]) -> Optional[torch.Tensor]:
    """The ``attention_tiny`` program's bounds for the validated host offsets
    ``bounds``: a CPU int32 tensor of ``BF16_TINY_MAX_SEGMENTS + 1`` entries
    holding the bounds of the non-empty segments (``[0, ..., T]``: an empty
    segment adds no token, so dropping it keeps every other bound), padded
    with ``T``.  ``None`` when a segment has more than
    ``BF16_TINY_SEGMENT_TOKENS`` tokens, or when no segment, or more than
    ``BF16_TINY_MAX_SEGMENTS`` segments, are non-empty."""
    lengths = [b - a for a, b in zip(bounds, bounds[1:], strict=False)]
    ends = [b for b, length in zip(bounds[1:], lengths, strict=True) if length > 0]
    if (
        not ends
        or len(ends) > BF16_TINY_MAX_SEGMENTS
        or max(lengths) > BF16_TINY_SEGMENT_TOKENS
    ):
        return None
    compact = [int(bounds[0]), *ends]
    padded = compact + [compact[-1]] * (BF16_TINY_MAX_SEGMENTS + 1 - len(compact))
    return torch.tensor(padded, dtype=torch.int32)


@functools.cache
def _device_facts(index: int) -> tuple[int, tuple[int, int]]:
    """``(multi_processor_count, compute capability)`` of CUDA device ``index``, read once."""
    props = torch.cuda.get_device_properties(index)
    return int(props.multi_processor_count), (int(props.major), int(props.minor))


def _device_index(device: torch.device) -> int:
    if device.type != "cuda":
        raise ValueError(f"expected a CUDA device, got {device}")
    return device.index if device.index is not None else torch.cuda.current_device()


def bf16_grid_clusters(device: torch.device) -> int:
    """Persistent-grid capacity of ``device`` in 2-CTA clusters (``num_SMs / 2``)."""
    return max(1, _device_facts(_device_index(device))[0] // 2)


def assign_unit_slots(unit_costs: Sequence[float], num_clusters: int) -> list[int]:
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


def bf16_unit_cost(blocks: int, single_stage: bool) -> int:
    """LPT cost of a unit's K/V range in block units (single-stage units are discounted)."""
    if single_stage:
        blocks = (
            blocks * BF16_SINGLE_STAGE_COST_NUM + BF16_SINGLE_STAGE_COST_DEN - 1
        ) // BF16_SINGLE_STAGE_COST_DEN
    return blocks + BF16_UNIT_OVERHEAD_BLOCKS


def bf16_choose_q_splits(
    costs: Sequence[int], half_costs: Sequence[Optional[int]], num_clusters: int
) -> list[bool]:
    """Which units of the partial tail round to run as two 256-row single-stage halves.

    ``costs[u]`` is the LPT cost of unit ``u`` and ``half_costs[u]`` the cost
    of one of its halves (``bf16_unit_cost(blocks, True)``), or ``None`` for a
    unit that must stay whole: a single-stage unit (nothing to halve) or a
    K/V-split range (the halves must stay clear of the combine path).  The
    statically strided grid runs ``U mod G`` units in its tail round while
    ``G - U mod G`` clusters idle for a whole unit time; each idle cluster can
    take one half.  The ``n`` most expensive halvable tail-round units are
    halved for ``n`` in ``1..idle`` (the makespan is a maximum, so the
    expensive units bound it; the halves are cheaper than every unit, so they
    and the unhalved tail units stay in the tail round) and the ``n`` with the
    smallest simulated makespan (:func:`lpt_makespan`, smallest ``n`` on ties)
    is taken when it beats the unhalved makespan by at least
    ``BF16_Q_SPLIT_MIN_GAIN``.  Returns the per-unit flag (all ``False``
    without a tail round, without a halvable candidate or without a paying
    ``n``).  At most ``G / 2`` makespan simulations.  Mirrors the Cake
    production planner's ``choose_q_splits``.
    """
    U = len(costs)
    flags = [False] * U
    if U == 0:
        return flags
    G = max(1, int(num_clusters))
    tail = U % G
    if tail == 0:
        return flags
    idle = G - tail
    if U < G:
        tail_units = list(range(U))
    else:
        tail_units = assign_unit_slots(costs, G)[(U // G) * G :]
    candidates = sorted(
        (u for u in tail_units if half_costs[u] is not None), key=lambda u: -costs[u]
    )
    if not candidates:
        return flags
    base = lpt_makespan(costs, G)
    best, best_n = base, 0
    for n in range(1, min(len(candidates), idle) + 1):
        halved = set(candidates[:n])
        sim: list[int] = []
        for u in range(U):
            half = half_costs[u]
            if half is not None and u in halved:
                sim.extend((half, half))
            else:
                sim.append(costs[u])
        span = lpt_makespan(sim, G)
        if span < best:
            best, best_n = span, n
    if best > base * (1.0 - BF16_Q_SPLIT_MIN_GAIN):
        return flags
    for u in candidates[:best_n]:
        flags[u] = True
    return flags


def bf16_unit_weights(table: Sequence[int]) -> list[int]:
    """Per-slot stage-weighted K/V blocks of a unit table (a two-stage unit counts its blocks twice).

    Mirrors the Cake planner's ``unit_weights``: ``two_stages`` is the kernel's record decode
    (``q_half == 0`` and more than ``CTA_Q_ROWS`` valid rows from the unit's first row).
    """
    weights: list[int] = []
    for slot in range(len(table) // UNIT_WORDS):
        words = table[slot * UNIT_WORDS : (slot + 1) * UNIT_WORDS]
        seg_len, c, kv_blocks, q_half = (
            words[1],
            words[2] & 0xFFFF,
            words[3] & 0xFFFF,
            words[5],
        )
        unit_row0 = c * CLUSTER_Q_ROWS + (q_half >> 1) * CTA_Q_ROWS
        two_stages = q_half == 0 and seg_len - unit_row0 > CTA_Q_ROWS
        weights.append(kv_blocks * (2 if two_stages else 1))
    return weights


def bf16_cluster_work(table: Sequence[int], num_clusters: int) -> list[int]:
    """Per-cluster work of the static slot order (slot ``k * G + i`` is unit ``k`` of cluster ``i``)."""
    G = max(1, int(num_clusters))
    work = [0] * G
    for slot, weight in enumerate(bf16_unit_weights(table)):
        work[slot % G] += weight
    return work


def bf16_dyn_sched(table: Sequence[int], total_tiles: int, num_clusters: int) -> bool:
    """Whether a plan binds the dynamic whole-unit ticket scheduler (mirrors ``dyn_sched_for_records``).

    ``table`` holds the ``total_tiles`` scheduled unit records (no padding records).
    """
    total_tiles, num_clusters = int(total_tiles), int(num_clusters)
    if num_clusters <= 0 or total_tiles < BF16_DYN_SCHED_MIN_WAVES * num_clusters:
        return False
    weights = bf16_unit_weights(table)
    work = [0] * num_clusters
    for slot, weight in enumerate(weights):
        work[slot % num_clusters] += weight
    if min(work) < BF16_DYN_SCHED_MIN_CLUSTER_BLOCKS:
        return False
    if -(-total_tiles // num_clusters) >= BF16_DYN_SCHED_MIN_UNITS_PER_CLUSTER:
        return True
    return max(work) - min(work) > min(weights)


def bf16_attention_stage(plan: "BF16SegmentPlan", arch: str) -> str:
    """The attention program ``plan`` binds on ``arch``: ``attention_tiny``, ``attention_dyn``, ``attention_sread``
    or ``attention``."""
    if plan.tiny_cu_seqlens is not None and arch in BF16_TINY_ARCHES:
        return "attention_tiny"
    if plan.dyn_sched:
        return "attention_dyn"
    if plan.sread_unified and arch in BF16_SREAD_ARCHES:
        return "attention_sread"
    return "attention"


def build_bf16_segment_plan(
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    device: torch.device,
    num_heads: int,
    *,
    num_clusters: Optional[int] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    kv_splits: Optional[int] = None,
    q_split: Optional[bool] = None,
) -> BF16SegmentPlan:
    """Build the BF16 segment plan (tables and partial workspace on ``device``).

    ``num_clusters`` is the persistent-grid capacity in 2-CTA clusters; it
    defaults to ``bf16_grid_clusters(device)`` (``device`` must then be a CUDA
    device).  ``kv_splits`` forces one K/V split factor on every unit
    (``None``: :func:`choose_kv_splits`).  ``q_split`` (``None``:
    ``BF16_Q_SPLIT_PLAN``) enables the tail-round Q-split: unsplit two-stage
    units of the partial tail round are emitted as two 256-row single-stage
    records (``q_half`` 1 / 2 in record word 5) when
    :func:`bf16_choose_q_splits` finds a paying makespan gain; the plan's
    ``num_q_split_units`` counts the halved units.  Off, the records are
    those of the plain plan (word 5 is 0).  The plan reproduces the Cake
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
    # (segment, head, cluster_in_segment, kv_block_begin, kv_blocks, partial_slot, q_half)
    units: list[tuple[int, int, int, int, int, int, int]] = []
    costs: list[int] = []
    # LPT cost of one 256-row half of the unit; None for units that stay whole.
    half_costs: list[Optional[int]] = []
    combine: list[int] = []
    partial_slots = 0
    unit_index = 0
    for seg, seg_clusters in enumerate(clusters):
        for head in range(num_heads):
            for c in range(seg_clusters):
                chunks = split_chunks(blocks[seg], split_of[unit_index])
                unit_index += 1
                # Units with at most half a cluster tile of valid rows run one Q stage.
                single_stage = lens[seg] - c * CLUSTER_Q_ROWS <= CLUSTER_Q_ROWS // 2
                if len(chunks) == 1:
                    units.append((seg, head, c, 0, blocks[seg], -1, 0))
                    costs.append(bf16_unit_cost(blocks[seg], single_stage))
                    # A two-stage unit may run as two 256-row single-stage halves.
                    half_costs.append(
                        None if single_stage else bf16_unit_cost(blocks[seg], True)
                    )
                    continue
                combine.extend((seg, (head << 16) | c, partial_slots, len(chunks)))
                for begin, count in chunks:
                    units.append((seg, head, c, begin, count, partial_slots, 0))
                    costs.append(bf16_unit_cost(count, single_stage))
                    half_costs.append(None)  # K/V ranges stay clear of the combine path
                    partial_slots += 1
    if q_split is None:
        q_split = bool(BF16_Q_SPLIT_PLAN)
    num_q_split_units = 0
    if q_split:
        halve = bf16_choose_q_splits(costs, half_costs, int(num_clusters))
        if any(halve):
            split_units: list[tuple[int, int, int, int, int, int, int]] = []
            split_costs: list[int] = []
            for u, (unit, cost) in enumerate(zip(units, costs, strict=True)):
                if not halve[u]:
                    split_units.append(unit)
                    split_costs.append(cost)
                    continue
                half_cost = half_costs[u]
                assert half_cost is not None
                # Consecutive in enumeration order: the stable LPT sort keeps
                # the two halves on neighbouring slots (one head's K/V in L2).
                for q_half in (1, 2):
                    split_units.append((*unit[:6], q_half))
                    split_costs.append(half_cost)
                num_q_split_units += 1
            units, costs = split_units, split_costs
    total_tiles = len(units)
    num_clusters = min(int(num_clusters), max(total_tiles, 1))
    table: list[int] = []
    for unit_index in assign_unit_slots(costs, num_clusters):
        seg, head, c, begin, count, slot, q_half = units[unit_index]
        table.extend(
            (
                begins[seg],
                lens[seg],
                (head << 16) | c,
                (begin << 16) | count,
                slot,
                q_half,
                0,
                0,
            )
        )
    partial_O, partial_ML = _partial_workspace(partial_slots, device)
    # Per-plan program dispatch from the scheduled records (see ``bf16_dyn_sched``).
    dyn_sched = bf16_dyn_sched(table, total_tiles, num_clusters)
    return BF16SegmentPlan(
        cu_seqlens=bounds,
        num_heads=num_heads,
        num_segments=len(segments),
        total_clusters=sum(clusters),
        total_tiles=total_tiles,
        num_clusters=num_clusters,
        seg_begin=_table(begins, device),
        seg_len=_table(lens, device),
        # ``num_clusters`` zero records of padding (every kernel role prefetches
        # the record of its next unit, ``tile_idx + num_clusters``; the
        # ``attention_dyn`` program keeps cluster ``c``'s two ticket mailboxes in
        # words 6 and 7 of padding record ``c``, word 6 + ticket slot) and one
        # more zero record whose word 0 is the
        # dynamic scheduler's ticket counter (reset by the kernel at the end of
        # every launch).
        unit_table=torch.tensor(
            (table or [0] * UNIT_WORDS) + [0] * (UNIT_WORDS * (num_clusters + 1)),
            dtype=torch.int32,
            device=device,
        ),
        combine_table=torch.tensor(
            combine or [0] * COMBINE_WORDS, dtype=torch.int32, device=device
        ),
        partial_O=partial_O,
        partial_ML=partial_ML,
        num_partial_slots=partial_slots,
        num_combine_units=len(combine) // COMBINE_WORDS,
        max_kv_splits=max(split_of, default=1),
        num_q_split_units=num_q_split_units,
        dyn_sched=dyn_sched,
        # Recorded arch-independently; the launcher applies ``BF16_SREAD_ARCHES``.
        sread_unified=not dyn_sched,
        tiny_cu_seqlens=bf16_tiny_cu_seqlens(bounds),
    )


_BF16_PLANS: "OrderedDict[tuple, BF16SegmentPlan]" = OrderedDict()


def cached_bf16_segment_plan(
    cu_seqlens: Sequence[int],
    device: torch.device,
    num_heads: int,
    *,
    kv_splits: Optional[int] = None,
    q_split: Optional[bool] = None,
) -> BF16SegmentPlan:
    """The BF16 segment plan of ``(cu_seqlens, device, num_heads, kv_splits,
    q_split)`` on the current stream, built once.

    ``cu_seqlens`` is the validated host tuple (:func:`normalize_cu_seqlens`);
    ``q_split`` (``None``: ``BF16_Q_SPLIT_PLAN``) is the tail-round Q-split
    option of :func:`build_bf16_segment_plan` and part of the key.
    Plans live in a most-recently-used cache of ``BF16_PLAN_CACHE_CAPACITY``
    entries; a hit returns the same tables and partial workspace, so repeated
    calls with one segment layout cost no planning, uploads or allocations.
    The key includes the current CUDA stream: launches of one plan are
    stream-ordered on their stream (a K/V-split plan's partial workspace is
    rewritten by every launch) and concurrent streams never share a plan.
    The dynamic scheduler's ticket counter (word 0 of the plan's last
    ``unit_table`` record) is rewritten by every launch as well and left at
    zero by the kernel, so a cached plan needs no host reset.
    """
    index = _device_index(device)
    q_split = bool(BF16_Q_SPLIT_PLAN) if q_split is None else bool(q_split)
    key = (
        tuple(int(v) for v in cu_seqlens),
        index,
        int(num_heads),
        None if kv_splits is None else int(kv_splits),
        q_split,
        torch.cuda.current_stream(index).cuda_stream,
    )
    plan = _BF16_PLANS.get(key)
    if plan is None:
        while len(_BF16_PLANS) >= BF16_PLAN_CACHE_CAPACITY:
            _BF16_PLANS.popitem(last=False)
        plan = build_bf16_segment_plan(
            key[0],
            torch.device("cuda", index),
            num_heads,
            kv_splits=kv_splits,
            q_split=q_split,
        )
        _BF16_PLANS[key] = plan
    else:
        _BF16_PLANS.move_to_end(key)
    return plan


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
    ``tp_block_token`` / ``tp_block_valid`` are the same two quantizer tables
    as the ``quantize_tp`` program's by-value parameters
    (:func:`quantize_tp_table`) when ``PB <= QUANTIZE_TP_MAX_PB``, else ``None``.
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
    tp_block_token: Optional[torch.Tensor] = None
    tp_block_valid: Optional[torch.Tensor] = None

    @property
    def num_segments(self) -> int:
        return len(self.seg_len)


def quantize_tp_table(values: Sequence[int]) -> Optional[torch.Tensor]:
    """A quantizer table as the ``quantize_tp`` program's by-value parameter:
    a CPU int32 tensor of exactly ``QUANTIZE_TP_MAX_PB`` entries holding
    ``values`` (one per packed block), zero-padded; ``None`` when ``values``
    has more than ``QUANTIZE_TP_MAX_PB`` entries.  The program reads entries
    ``0 .. PB - 1`` only."""
    if len(values) > QUANTIZE_TP_MAX_PB:
        return None
    padded = [int(v) for v in values] + [0] * (QUANTIZE_TP_MAX_PB - len(values))
    return torch.tensor(padded, dtype=torch.int32)


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
        tp_block_token=quantize_tp_table(block_token),
        tp_block_valid=quantize_tp_table(block_valid),
    )


@dataclass(frozen=True)
class TileTables:
    """Per-unit scheduler tables of the NVFP4 attention kernel for one ``(plan, heads)``.

    ``total_tiles`` scheduled units (K/V split ranges and Q halves included)
    per table.  Unit ``t`` runs head ``cl_head[t]`` on the cluster tile
    described by the next four tables (segment start token, segment length,
    first packed block of the segment, and ``cl_q_block[t]``: the first Q
    block of the cluster tile inside the segment in bits 0-15 and ``q_half``
    in bits 16+) over K/V blocks ``cl_kv_begin[t] .. cl_kv_begin[t] +
    cl_kv_blocks[t]`` of the segment, writing partial slot ``cl_ws_slot[t]``
    (-1: the BF16 output directly).  ``q_half`` 0 marks a whole tile with more
    than ``CTA_Q_ROWS`` valid rows from its first row, which runs both softmax
    stages; 1 a single-stage unit on the tile's first 256 rows (a tile with at
    most ``CTA_Q_ROWS`` valid rows, or the first half of a tile the planner
    split for its tail round); 2 the second half.  The persistent grid is
    statically strided: cluster ``i`` runs units ``i, i + G, ...`` with ``G =
    grid_x / 2``.  ``combine_table``, ``seg_begin`` / ``seg_len`` and the
    partial workspace feed the ``combine`` stage.
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
    num_q_split_units: int

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


def nvfp4_unit_cost(
    blocks: int, single_stage: bool, single_stage_cost: tuple[int, int]
) -> int:
    """LPT cost of an NVFP4 unit's K/V range in block units (single-stage units are discounted to NUM / DEN)."""
    if single_stage:
        num, den = single_stage_cost
        blocks = (blocks * num + den - 1) // den
    return blocks + BF16_UNIT_OVERHEAD_BLOCKS


def build_tile_tables(
    plan: PackedSegmentPlan,
    heads: int,
    device: torch.device,
    *,
    num_clusters: Optional[int] = None,
    kv_splits: Optional[int] = None,
    program_cost: float = 1.0,
    single_stage_cost: tuple[int, int] = (
        BF16_SINGLE_STAGE_COST_NUM,
        BF16_SINGLE_STAGE_COST_DEN,
    ),
    q_split: Optional[Union[bool, str]] = None,
    two_stage_dense: bool = False,
    q_split_dense: bool = False,
) -> TileTables:
    """Per-unit tables: segment-major enumeration, K/V splits, Q halves, then LPT slot placement.

    Units are enumerated by segment in descending length (ties keep the
    segment order), then head, then the segment's cluster tiles in order --
    consecutive units share one head's K/V, which keeps the concurrent working
    set L2-resident -- and each unit is cut into its planned number of
    near-equal K/V block ranges (:func:`choose_kv_splits`; ``kv_splits``
    forces one factor).  A unit's placement cost is its K/V blocks plus the
    unit overhead, the blocks discounted to ``single_stage_cost`` (NUM, DEN)
    for a single-stage unit (at most ``CTA_Q_ROWS`` valid rows from its first
    row, ``q_half`` 1: :func:`nvfp4_unit_cost`); the split decision itself does
    not see the discount.  ``q_split`` (``None``: ``NVFP4_Q_SPLIT_PLAN``) halves the
    tail-round two-stage unsplit units :func:`bf16_choose_q_splits` picks into
    two 256-row single-stage records (``q_half`` 1 / 2, consecutive in the
    enumeration); ``q_split="all"`` halves every such unit (tests).
    ``two_stage_dense`` (the dense program of a ``NVFP4_TWO_STAGE_DENSE`` pair)
    plans a plan without K/V-split units as whole two-stage units: ``q_half``
    0 everywhere, every cost undiscounted and no Q-split halves, so the units
    are the ``heads * total_clusters`` the program runs; a plan with split
    units binds the split program and keeps both.  ``q_split_dense`` (a
    ``NVFP4_QS_DENSE`` pair) decides the Q-splits on the single-stage costs
    first: a plan without K/V-split units whose tail round takes halves keeps
    them and its single-stage costs (it binds the single-stage dense program),
    so ``two_stage_dense`` applies only to a plan with neither.  The
    units are then placed into the persistent grid's statically strided slots
    longest-processing-time first (:func:`assign_unit_slots`; equal costs keep
    the enumeration order).  The
    kernel's five per-tile tables are allocated first, in the production
    planner's order, so their placement relative to the workspace matches it.
    ``num_clusters`` is the grid capacity in 2-CTA clusters (default
    ``bf16_grid_clusters(device)``, a CUDA device).  Mirrors the Cake
    production planner's ``build_tile_tables``.
    """
    pre = _tile_units(
        plan,
        heads,
        device,
        num_clusters=num_clusters,
        kv_splits=kv_splits,
        program_cost=program_cost,
        single_stage_cost=single_stage_cost,
    )
    units, costs, half_costs = pre.units, pre.costs, pre.half_costs
    if q_split_dense and two_stage_dense and not pre.combine:
        halve = _q_split_halves(costs, half_costs, pre.num_clusters, q_split)
        if any(halve):
            return _place_tile_units(plan, pre, units, costs, half_costs, halve, device)
    if two_stage_dense and not pre.combine:
        units, costs, half_costs = _two_stage_dense_units(plan, pre, single_stage_cost)
    halve = _q_split_halves(costs, half_costs, pre.num_clusters, q_split)
    return _place_tile_units(plan, pre, units, costs, half_costs, halve, device)


@dataclass(frozen=True)
class _TileUnits:
    """The host prefix of :func:`build_tile_tables` shared by both dense forms.

    The units in enumeration order with their LPT costs and Q-half costs
    (``None``: stays whole), the combine records and partial-slot count of
    the K/V-split units, ``split_of`` per enumerated unit, the cluster-tile
    columns ``cl`` and the resolved ``num_clusters``.  ``two_stage_dense`` and
    the Q-split decision act on it afterwards; nothing here touches a device.
    """

    heads: int
    num_clusters: int
    cl: dict[str, list[int]]
    split_of: list[int]
    units: list[tuple[int, int, int, int, int, int]]
    costs: list[int]
    half_costs: list[Optional[int]]
    combine: list[int]
    partial_slots: int


def _tile_units(
    plan: PackedSegmentPlan,
    heads: int,
    device: torch.device,
    *,
    num_clusters: Optional[int],
    kv_splits: Optional[int],
    program_cost: float,
    single_stage_cost: tuple[int, int],
) -> _TileUnits:
    """Enumeration, K/V splits (:func:`choose_kv_splits`) and costs of :func:`build_tile_tables`."""
    heads = int(heads)
    if heads <= 0:
        raise ValueError("heads must be positive")
    if kv_splits is not None and not 1 <= int(kv_splits) <= MAX_KV_SPLITS:
        raise ValueError(f"kv_splits must be in [1, {MAX_KV_SPLITS}], got {kv_splits}")
    if num_clusters is None:
        num_clusters = bf16_grid_clusters(device)
    segments = sorted(range(plan.num_segments), key=lambda s: -plan.seg_len[s])
    cl: dict[str, list[int]] = {
        "cl_seg_begin": [],
        "cl_seg_len": [],
        "cl_kv_base": [],
        "cl_q_block": [],
    }
    for s in range(plan.num_segments):
        for c in range(plan.cluster_off[s + 1] - plan.cluster_off[s]):
            cl["cl_seg_begin"].append(plan.seg_begin[s])
            cl["cl_seg_len"].append(plan.seg_len[s])
            cl["cl_kv_base"].append(plan.seg_tile_base[s])
            cl["cl_q_block"].append(CLUSTER_Q_BLOCKS * c)
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
        tuple[int, int, int, int, int, int]
    ] = []  # (cluster tile, head, kv_begin, kv_blocks, slot, q_half)
    costs: list[int] = []
    # LPT cost of one 256-row half of the unit; None for units that stay whole.
    half_costs: list[Optional[int]] = []
    combine: list[int] = []
    partial_slots = 0
    unit_index = 0
    for s in segments:
        for head in range(heads):
            for c in range(plan.cluster_off[s], plan.cluster_off[s + 1]):
                chunks = split_chunks(seg_blocks[s], split_of[unit_index])
                unit_index += 1
                # Units with at most half a cluster tile of valid rows run one Q stage.
                single_stage = (
                    plan.seg_len[s] - (c - plan.cluster_off[s]) * CLUSTER_Q_ROWS
                    <= CTA_Q_ROWS
                )
                if len(chunks) == 1:
                    units.append((c, head, 0, seg_blocks[s], -1, int(single_stage)))
                    costs.append(
                        nvfp4_unit_cost(seg_blocks[s], single_stage, single_stage_cost)
                    )
                    # A two-stage unit may run as two 256-row single-stage halves.
                    half_costs.append(
                        None
                        if single_stage
                        else nvfp4_unit_cost(seg_blocks[s], True, single_stage_cost)
                    )
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
                    units.append(
                        (c, head, begin, count, partial_slots, int(single_stage))
                    )
                    costs.append(
                        nvfp4_unit_cost(count, single_stage, single_stage_cost)
                    )
                    half_costs.append(None)  # K/V ranges stay clear of the combine path
                    partial_slots += 1
    return _TileUnits(
        heads=heads,
        num_clusters=num_clusters,
        cl=cl,
        split_of=split_of,
        units=units,
        costs=costs,
        half_costs=half_costs,
        combine=combine,
        partial_slots=partial_slots,
    )


def _two_stage_dense_units(
    plan: PackedSegmentPlan, pre: _TileUnits, single_stage_cost: tuple[int, int]
) -> tuple[list[tuple[int, int, int, int, int, int]], list[int], list[Optional[int]]]:
    """``two_stage_dense`` on a plan without K/V-split units: whole two-stage units, undiscounted, no halves."""
    assert not pre.combine
    # The dense program runs both Q stages of every unit: whole units, no halves.
    units = [(*unit[:5], 0) for unit in pre.units]
    costs = [nvfp4_unit_cost(unit[3], False, single_stage_cost) for unit in units]
    half_costs: list[Optional[int]] = [None for _ in units]
    assert len(units) == pre.heads * plan.total_clusters
    return units, costs, half_costs


def _q_split_halves(
    costs: Sequence[int],
    half_costs: Sequence[Optional[int]],
    num_clusters: int,
    q_split: Optional[Union[bool, str]],
) -> list[bool]:
    """The units :func:`build_tile_tables` halves (``q_split`` as there)."""
    if q_split is None:
        q_split = bool(NVFP4_Q_SPLIT_PLAN)
    if not q_split:
        return [False] * len(costs)
    if q_split == "all":
        return [half is not None for half in half_costs]
    return bf16_choose_q_splits(costs, half_costs, int(num_clusters))


def _place_tile_units(
    plan: PackedSegmentPlan,
    pre: _TileUnits,
    units: list[tuple[int, int, int, int, int, int]],
    costs: list[int],
    half_costs: Sequence[Optional[int]],
    halve: Sequence[bool],
    device: torch.device,
) -> TileTables:
    """The halves, the LPT slot placement and the device tables of :func:`build_tile_tables`."""
    heads, num_clusters, cl = pre.heads, pre.num_clusters, pre.cl
    split_of, combine, partial_slots = pre.split_of, pre.combine, pre.partial_slots
    num_q_split_units = 0
    if any(halve):
        split_units: list[tuple[int, int, int, int, int, int]] = []
        split_costs: list[int] = []
        for u, (unit, cost) in enumerate(zip(units, costs, strict=True)):
            if not halve[u]:
                split_units.append(unit)
                split_costs.append(cost)
                continue
            half_cost = half_costs[u]
            assert half_cost is not None
            # Consecutive in enumeration order: the stable LPT sort keeps
            # the two halves on neighbouring slots (one head's K/V in L2).
            for q_half in (1, 2):
                split_units.append((*unit[:5], q_half))
                split_costs.append(half_cost)
            num_q_split_units += 1
        units, costs = split_units, split_costs
    num_tiles = len(units)
    slots = [
        units[u]
        for u in assign_unit_slots(costs, min(int(num_clusters), max(num_tiles, 1)))
    ]
    cl_head = _table([unit[1] for unit in slots], device)
    cl_tables = {
        name: _table(
            [
                values[unit[0]] | (unit[5] << NVFP4_Q_HALF_SHIFT)
                if name == "cl_q_block"
                else values[unit[0]]
                for unit in slots
            ],
            device,
        )
        for name, values in cl.items()
    }
    partial_O, partial_ML = _partial_workspace(partial_slots, device)
    return TileTables(
        heads=heads,
        total_tiles=num_tiles,
        cl_head=cl_head,
        cl_kv_begin=_table([unit[2] for unit in slots], device),
        cl_kv_blocks=_table([unit[3] for unit in slots], device),
        cl_ws_slot=_table([unit[4] for unit in slots], device),
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
        num_q_split_units=num_q_split_units,
        **cl_tables,
    )


def nvfp4_quantize_grid(heads: int, PB: int) -> tuple[int, int, int]:
    """Launch grid of the fused quantizer: one CTA per 32-token slice of every packed block."""
    return (int(heads) * int(PB) * QUANTIZE_SUBS_PER_BLOCK, 1, 1)


# ---------------------------------------------------------------------------
# Packed NVFP4 / FP8 operand workspace
# ---------------------------------------------------------------------------

# The NVFP4 quantizers and the fp8 route's ``amax`` stage read ``query`` /
# ``key`` / ``value`` with 32-byte vector loads; the packed operands are
# written with stores of up to 16 bytes and read through TMA, which needs
# 16-byte-aligned bases.  Preparation refuses less-aligned tensors (a
# misaligned vector access is a sticky CUDA error).
NVFP4_INPUT_ALIGN = 32
NVFP4_WORKSPACE_ALIGN = 16


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
        # ``amax`` stage output (one partial ``max|V|`` per CTA) and the scalar
        # the quantizer publishes for the attention kernel.
        shapes["v_amax_partial"] = ((AMAX_CTAS,), torch.float32)
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


def amax_kwargs(
    value: torch.Tensor, workspace: dict[str, torch.Tensor]
) -> dict[str, Any]:
    """Keyword bindings of the ``amax`` stage for the contiguous BF16 V operand.

    The kernel views ``value`` as ``numel / AMAX_VECTOR`` 32-byte vectors
    (any split of a contiguous tensor yields the same maximum) and each of the
    ``AMAX_CTAS`` CTAs strides over the ``num_chunks`` chunks of ``AMAX_CHUNK``
    vectors; the scalar never visits the host.
    """
    numel = int(value.numel())
    if numel % AMAX_VECTOR != 0:
        raise ValueError(f"V.numel() must be a multiple of {AMAX_VECTOR}, got {numel}")
    if not value.is_contiguous():
        raise ValueError("V must be contiguous for the amax stage")
    num_vectors = numel // AMAX_VECTOR
    kwargs = dict(
        v=value,
        v_amax_partial=workspace["v_amax_partial"],
        num_vectors=num_vectors,
        num_chunks=(num_vectors + AMAX_CHUNK - 1) // AMAX_CHUNK,
        grid=(AMAX_CTAS, 1, 1),
    )
    assert tuple(kwargs) == AMAX_KWARGS
    return kwargs


# ---------------------------------------------------------------------------
# Binding to the generated argument plans
# ---------------------------------------------------------------------------


def _arch_for(device: torch.device) -> str:
    capability = _device_facts(_device_index(device))[1]
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(capability)
    if arch is None:
        raise ValueError(
            "MiniMax-H3 packed-varlen attention requires compute capability 10.0 or "
            f"10.3 (got {capability[0]}.{capability[1]})"
        )
    return arch


def generated_program_available(device: torch.device, variant: str = "bf16") -> bool:
    """True when this checkout registers ``variant`` for ``device``."""
    arch = SUPPORTED_COMPUTE_CAPABILITIES.get(_device_facts(_device_index(device))[1])
    return arch is not None and route_available(variant, arch)


def _bind_stage(
    module_name: str,
    arch: str,
    kwargs: dict[str, Any],
    *,
    exact: bool = False,
    by_value: Sequence[str] = (),
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated argument plan of ``module_name`` built for ``arch``.

    ``exact``: the argument plan names exactly the parameters in ``kwargs``
    (``grid`` for the grid dimensions) and ``module_name`` builds for
    ``arch``, so a program of another form is refused, not bound to tables
    planned for this one (the NVFP4 attention programs,
    :func:`nvfp4_attention_kwargs`).  ``by_value``: the parameters the host
    passes as by-value arrays (CPU tensors); the argument plan must take each
    of them as a ``param_array`` and ``module_name`` must build for ``arch``,
    so a program that takes them as device buffers is refused (the
    ``quantize_tp`` program, ``QUANTIZE_TP_BY_VALUE``).
    """
    record = MODULES[module_name]
    if by_value:
        kinds = {name: kind for kind, name in record["arg_plan"]}
        taken = {kinds.get(name) for name in by_value}
        if arch not in record["arches"] or taken != {"param_array"}:
            raise ValueError(
                f"generated module {module_name!r} (arches {record['arches']}) takes "
                f"{[(kinds.get(name), name) for name in by_value]}; the host binds "
                f"{list(by_value)} as param_array on {arch}"
            )
    if exact:
        planned = {
            "grid" if kind == "grid" else name for kind, name in record["arg_plan"]
        }
        if arch not in record["arches"] or planned != set(kwargs):
            raise ValueError(
                f"generated module {module_name!r} (arches {record['arches']}) takes "
                f"{sorted(planned)}; the host binds {sorted(kwargs)} on {arch}"
            )
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
    module = load_cake_minimax_h3_varlen_attention_module(module_name, arch)
    return getattr(module, record["ffi_entry"]), tuple(arguments)


def _persistent_grid(
    device: torch.device, total_tiles: int, *, at_least_one: bool
) -> tuple[int, int, int]:
    num_sms = _device_facts(_device_index(device))[0]
    pairs = min(num_sms // 2, total_tiles)
    if at_least_one:
        pairs = max(1, pairs)
    return (2 * pairs, 1, 1)


def minimax_h3_varlen_fp8_launch_grid(
    num_sms: int, *, total_tiles: int
) -> tuple[int, int, int]:
    """Launch grid of the ``fp8`` attention programs on a device of ``num_sms`` SMs
    for a plan of ``total_tiles`` units: ``2 * num_clusters`` CTAs (one 2-CTA
    cluster per persistent slot) with ``num_clusters = min(max(1, num_sms // 2),
    max(total_tiles, 1))`` -- the BF16 planner's cluster count
    (:func:`bf16_grid_clusters` capped by the unit count).  The export parity lane
    names this function as the route's grid rule."""
    pairs = min(max(1, int(num_sms) // 2), max(int(total_tiles), 1))
    return (2 * pairs, 1, 1)


# Element multiple that keeps a BF16 stride 16-byte aligned (TMA global
# strides and the BF16 kernel's 16-byte ragged-tail copies).
THD_STRIDE_ALIGN = 16 // 2


def _check_thd(
    name: str,
    tensor: torch.Tensor,
    total_tokens: int,
    num_heads: int,
    device: torch.device,
    *,
    contiguous: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """Validate one ``[T, H, 128]`` operand of ``dtype`` (BF16, or E4M3 for the ``fp8`` route).

    ``contiguous=False`` accepts any view with unit innermost stride, a head
    stride that is a 16-byte multiple of at least 128 elements, a token
    stride that is a 16-byte multiple of at least ``H * head_stride`` and a
    16-byte-aligned base: contiguous tensors, the column slices of a fused
    ``[T, 3 * H * 128]`` QKV projection (strides ``(3 * H * 128, 128, 1)``)
    and the slices of a ``[T, H, 3, 128]`` pack (strides ``(H * 384, 384,
    1)``).  ``contiguous=True`` additionally requires a contiguous tensor.  The
    16-byte stride rule is in bytes: 8 elements for BF16, 16 for E4M3.
    """
    if not tensor.is_cuda or tensor.device != device:
        raise ValueError(f"{name} must live on {device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must be a {str(dtype).removeprefix('torch.')} tensor")
    stride_align = 16 // tensor.element_size()
    if tuple(tensor.shape) != (total_tokens, num_heads, HEAD_DIM):
        raise ValueError(
            f"{name} must be [T, H, {HEAD_DIM}] = {(total_tokens, num_heads, HEAD_DIM)}, "
            f"got {tuple(tensor.shape)}"
        )
    row_stride, head_stride, elem_stride = (int(s) for s in tensor.stride())
    if elem_stride != 1:
        raise ValueError(
            f"{name} must have a unit innermost stride, got stride(2)={elem_stride}"
        )
    if head_stride < HEAD_DIM or head_stride % stride_align:
        raise ValueError(
            f"{name} head stride must be a 16-byte multiple of at least {HEAD_DIM} "
            f"elements, got {head_stride}"
        )
    if row_stride < num_heads * head_stride or row_stride % stride_align:
        raise ValueError(
            f"{name} token stride must be a 16-byte multiple of at least "
            f"H * head_stride = {num_heads * head_stride} elements, got {row_stride}"
        )
    if tensor.data_ptr() % 16:
        raise ValueError(f"{name} base address must be 16-byte aligned")
    if contiguous and not tensor.is_contiguous():
        raise ValueError(
            f"{name} must be contiguous, got strides {tuple(tensor.stride())}"
        )


def validate_minimax_h3_varlen_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    out: Optional[torch.Tensor],
    *,
    strided_qkv: bool,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[int, int, torch.device]:
    """Shape / dtype / device validation shared by the families (``dtype`` of ``query`` / ``key`` / ``value``).

    ``strided_qkv=True`` (the BF16 route) accepts strided ``[T, H, 128]``
    views for ``query`` / ``key`` / ``value`` (see ``_check_thd``; the three
    may differ in strides); ``False`` (the NVFP4 routes, whose quantizers
    read contiguous THD) requires contiguous tensors.  ``out`` is always
    contiguous.  Returns ``(total_tokens, num_heads, device)``.
    """
    if query.ndim != 3:
        raise ValueError(f"query must be THD [T, H, {HEAD_DIM}]")
    total_tokens, num_heads = int(query.shape[0]), int(query.shape[1])
    if num_heads <= 0:
        raise ValueError("query must have at least one head")
    device = query.device
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        _check_thd(
            name,
            tensor,
            total_tokens,
            num_heads,
            device,
            contiguous=not strided_qkv,
            dtype=dtype,
        )
    if out is not None:
        _check_thd("out", out, total_tokens, num_heads, device, contiguous=True)
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
    attention_stage: str
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
        return math.prod(int(d) for d in self.main_kwargs["grid"])

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
            q_split_units=self.plan.num_q_split_units,
            attention_variant=self.attention_stage,
            dyn_sched=bool(self.plan.dyn_sched),
            sread_unified=bool(self.plan.sread_unified)
            and self.arch in BF16_SREAD_ARCHES,
        )


@dataclass(frozen=True)
class MiniMaxH3VarlenFP8AttentionRunner:
    """Launch the prepared all-FP8 packed-varlen attention (``variant="fp8"``).

    Calling the runner or ``launch()`` writes the caller-owned ``out`` on the
    current stream with no CUDA allocation and no host synchronization and
    returns ``out``.  The segment plan and the descales are fixed at
    preparation (the descale tensors are read by the kernel at every launch,
    so their values may change freely); prepare a new runner when
    ``cu_seqlens``, shapes or tensor bindings change.  CUDA Graph capture
    belongs to the caller.
    """

    variant: str
    arch: str
    plan: BF16SegmentPlan
    main_kwargs: dict
    attention_stage: str
    out: torch.Tensor
    _entry: Optional[Callable[..., Any]]
    _arguments: tuple
    _combine_entry: Optional[Callable[..., Any]] = None
    _combine_arguments: tuple = ()

    def launch(self) -> torch.Tensor:
        if self._entry is not None:  # zero tiles: nothing to launch
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
            q_split_units=self.plan.num_q_split_units,
            attention_variant=self.attention_stage,
            dyn_sched=False,
            sread_unified=False,
        )


@dataclass(frozen=True)
class MiniMaxH3VarlenNVFP4AttentionRunner:
    """Launch the prepared NVFP4 pipeline: quantize Q/K/V, then attend.

    ``launch()`` runs the complete pipeline (for ``pv_mode="fp8"`` one device
    ``amax(V)`` reduction into the workspace, then one fused quantizer launch
    (one of two builds, see ``quantize_stage``),
    one attention launch and, for plans with K/V-split units, the ``combine``
    launch) into the caller-owned ``out`` with no CUDA allocation and no host
    synchronization.  ``quantize()`` and ``attention()`` run the stages
    separately so callers can time them on their own.  The attention launch
    uses one of the generated programs, chosen at preparation from the plan:
    the dense ``attention`` module for plans without split units (on a
    ``NVFP4_QS_DENSE`` pair ``attention_qs`` for the Q-split plans) and
    ``attention_split`` otherwise (``attention_stage``).  Packed operands live in ``workspace`` (allocated at
    preparation).  Prepare a new runner when ``cu_seqlens``, shapes or tensor
    bindings change; values may change freely since quantization is part of
    every launch.  CUDA Graph capture belongs to the caller.

    The attention binding is generated with the programmatic-dependent-launch
    attribute (``cudaLaunchAttributeProgrammaticStreamSerialization``): its
    prologue overlaps the quantizer tail and ``griddepcontrol.wait`` orders the
    packed-operand reads.  For ``pv_mode="fp8"`` the quantizer binding carries
    the same attribute towards the preceding ``amax`` stage (its Q/K/V loads
    overlap the partial-max kernel's tail; the fold of the partials waits),
    and ``quantize_early_qk`` also quantizes and stores Q and K before that
    wait.  This is safe because the first launch of a call (``amax`` for fp8,
    the quantizer, ``quantize`` or ``quantize_tp``, for fp4) carries no such
    attribute: it starts only after all earlier work on the stream, the
    previous call's attention and ``combine`` included, has finished.  All
    launches are enqueued on the current torch stream in order, which is all
    the handshakes require from the host.
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
                entry(*arguments)

    def quantize(self) -> None:
        # fp8: the ``amax`` partial-max kernel and the quantizer that folds it.
        self._run(tuple(s for s in STAGES[self.variant] if s in QUANTIZE_STAGES))

    def attention(self) -> torch.Tensor:
        # The bound attention program and, for K/V-split plans, the combine launch.
        self._run(tuple(s for s in STAGES[self.variant] if s not in QUANTIZE_STAGES))
        return self.out

    @property
    def attention_stage(self) -> str:
        """``"attention_split"`` for plans with K/V-split units, ``"attention_qs"``
        for the Q-split plans of a ``NVFP4_QS_DENSE`` pair, else ``"attention"``."""
        return nvfp4_attention_stage(self.tile_tables, self.pv_mode, self.arch)

    def launch(self) -> torch.Tensor:
        self._run(STAGES[self.variant])
        return self.out

    __call__ = launch

    @property
    def quantize_stage(self) -> str:
        """The quantizer the plan binds: ``"quantize_early_qk"`` (fp8),
        ``"quantize_tp"`` (fp4) or ``"quantize"`` (:func:`nvfp4_quantize_stage`)."""
        bound = [
            name
            for name, entry, _ in self._stages
            if entry is not None and name in QUANTIZER_STAGES
        ]
        return bound[0] if bound else "quantize"

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
            q_split_units=self.tile_tables.num_q_split_units,
            kv_split_units=self.tile_tables.num_combine_units,
            max_kv_splits=self.tile_tables.max_kv_splits,
            attention_variant=self.attention_stage,
            quantize_variant=self.quantize_stage,
        )


def nvfp4_attention_stage(
    tiles: TileTables, pv_mode: Optional[str] = None, arch: Optional[str] = None
) -> str:
    """The attention program a plan binds: ``attention_split`` when the tile
    tables hold any K/V-split unit (partial slots), ``attention_qs`` when the
    tables of a ``NVFP4_QS_DENSE`` ``(pv_mode, arch)`` pair hold Q-split units
    (:func:`nvfp4_tile_tables` plans them single-stage only then), else the
    dense ``attention``.

    They are the same generated program specialised at build time; the dense
    variant carries no K/V-range or partial-slot code, which keeps the softmax
    block loop of unsplit units at the schedule of the single-program kernel.
    """
    if int(tiles.num_partial_slots) > 0:
        return "attention_split"
    if (pv_mode, arch) in NVFP4_QS_DENSE and int(tiles.num_q_split_units) > 0:
        return "attention_qs"
    return "attention"


def nvfp4_attention_kwargs(pv_mode: str, arch: str, stage: str) -> tuple[str, ...]:
    """The parameters, in order, the host binds to the ``stage`` attention program
    (``"attention"`` / ``"attention_qs"`` / ``"attention_split"``) of
    ``(pv_mode, arch)``.

    The dense program of a ``NVFP4_TWO_STAGE_DENSE`` pair takes
    ``total_clusters`` in place of ``num_tiles``, and the ``attention_qs``
    program of a ``NVFP4_QS_DENSE`` pair the single-stage dense parameters
    (``num_tiles``); :func:`_bind_stage` refuses a generated program whose
    argument plan names a different set.
    """
    if stage == "attention_split":
        if pv_mode == "fp4":
            return NVFP4_ATTENTION_SPLIT_FP4PV_KWARGS
        return NVFP4_ATTENTION_SPLIT_FP8PV_KWARGS
    if stage == "attention_qs":
        if (pv_mode, arch) not in NVFP4_QS_DENSE:
            raise ValueError(f"no attention_qs program for {(pv_mode, arch)!r}")
        return NVFP4_ATTENTION_FP8PV_KWARGS
    if stage != "attention":
        raise ValueError(f"not an NVFP4 attention stage: {stage!r}")
    if (pv_mode, arch) in NVFP4_TWO_STAGE_DENSE:
        assert pv_mode == "fp8"
        return NVFP4_ATTENTION_TWO_STAGE_FP8PV_KWARGS
    if pv_mode == "fp4":
        return NVFP4_ATTENTION_FP4PV_KWARGS
    return NVFP4_ATTENTION_FP8PV_KWARGS


def nvfp4_tile_tables(
    plan: PackedSegmentPlan,
    heads: int,
    device: torch.device,
    pv_mode: str,
    arch: str,
    *,
    num_clusters: Optional[int] = None,
) -> TileTables:
    """The tile tables an NVFP4 plan of ``(pv_mode, arch)`` binds
    (:func:`build_tile_tables` with the pair's costs).

    One :func:`build_tile_tables` call with the pair's costs: a
    ``NVFP4_QS_DENSE`` pair passes ``q_split_dense``, so a plan with tail-round
    Q-split units and no K/V-split units keeps its single-stage tables (they
    bind ``attention_qs``, :func:`nvfp4_attention_stage`) and only a plan with
    neither takes the ``two_stage_dense`` form.  ``num_clusters``: as
    :func:`build_tile_tables`.
    """
    pair = (pv_mode, arch)
    return build_tile_tables(
        plan,
        heads,
        device,
        num_clusters=num_clusters,
        program_cost=SPLIT_PROGRAM_COST.get(pair, 1.0),
        single_stage_cost=NVFP4_SINGLE_STAGE_COST[pair],
        two_stage_dense=pair in NVFP4_TWO_STAGE_DENSE,
        q_split_dense=pair in NVFP4_QS_DENSE,
    )


def nvfp4_quantize_stage(
    pv_mode: str, heads: int, PB: int, num_sms: int, *, arch: Optional[str] = None
) -> str:
    """The quantizer a plan binds.  ``pv_mode="fp8"`` binds ``quantize_early_qk``
    when the quantize grid (``nvfp4_quantize_grid``) fits in ``EARLY_QK_WAVES``
    waves of ``EARLY_QK_CTAS_PER_SM`` CTAs on each of the device's ``num_sms``
    SMs; ``pv_mode="fp4"`` binds ``quantize_tp`` on an ``arch`` of
    ``NVFP4_QUANTIZE_TP_ARCHES`` when ``PB <= QUANTIZE_TP_MAX_PB``; every
    other plan binds ``quantize``.

    Each pair builds from one quantizer source (``pv_mode`` ``"fp8"`` /
    ``"fp8_early_qk"``, ``"fp4"`` / ``"fp4_tp"``), specialised at build time,
    and writes bitwise-identical outputs; ``quantize_early_qk`` quantizes and
    stores Q and K before its grid-dependency wait, so that work overlaps the
    ``amax`` stage, and ``quantize_tp`` reads ``block_token`` /
    ``block_valid`` from its launch parameters instead of global memory.
    """
    if pv_mode == "fp4":
        if arch in NVFP4_QUANTIZE_TP_ARCHES and int(PB) <= QUANTIZE_TP_MAX_PB:
            return "quantize_tp"
        return "quantize"
    if pv_mode != "fp8":
        return "quantize"
    ctas = nvfp4_quantize_grid(heads, PB)[0]
    if ctas <= EARLY_QK_WAVES * EARLY_QK_CTAS_PER_SM * int(num_sms):
        return "quantize_early_qk"
    return "quantize"


# ---------------------------------------------------------------------------
# Preparation
# ---------------------------------------------------------------------------


def _descale_tensor(
    name: str, value: Union[float, torch.Tensor], device: torch.device
) -> torch.Tensor:
    """One finite positive FP32 descale as a one-element CUDA tensor on ``device`` (a float is uploaded once, at
    preparation; a tensor is passed through and read by the kernel at every launch)."""
    if isinstance(value, torch.Tensor):
        if value.numel() != 1 or value.dtype != torch.float32:
            raise ValueError(
                f"{name} must be a one-element float32 tensor, got {tuple(value.shape)} {value.dtype}"
            )
        if not value.is_cuda or value.device != device:
            raise ValueError(f"{name} must live on {device}")
        return value.reshape(1)
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(f"{name} must be a finite positive descale, got {value!r}")
    return torch.tensor([scale], dtype=torch.float32, device=device)


def fp8_attention_stage(bounds: Sequence[int]) -> str:
    """The ``fp8`` route's attention program for a plan: ``attention_tiny`` (the ``tail_skip=32`` build) when every
    segment has at most ``FP8_TINY_SEGMENT_TOKENS`` tokens -- every K/V block is then a tail block and the output is
    bitwise the general program's -- else ``attention``."""
    longest = max((b - a for a, b in zip(bounds, bounds[1:], strict=False)), default=0)
    return "attention_tiny" if longest <= FP8_TINY_SEGMENT_TOKENS else "attention"


def _bind_in_tree(
    program: Any, target: str, kwargs: dict[str, Any]
) -> tuple[Callable[..., Any], tuple]:
    """Order ``kwargs`` by the generated program's argument order (``CakeProgram.args``: the binding's host
    parameters followed by the grid triple) and resolve its entry for ``target``."""
    grid = dict(zip(("grid_x", "grid_y", "grid_z"), kwargs["grid"], strict=True))
    arguments = []
    for name in program.args:
        if name in grid:
            arguments.append(grid[name])
        elif name in kwargs:
            arguments.append(kwargs[name])
        else:
            raise KeyError(
                f"generated program {program.name!r} expects argument {name!r}; "
                f"host binding provides {sorted(kwargs)}"
            )
    return program.entry(target), tuple(arguments)


def prepare_minimax_h3_varlen_fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    q_descale: Union[float, torch.Tensor],
    k_descale: Union[float, torch.Tensor],
    v_descale: Union[float, torch.Tensor],
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> MiniMaxH3VarlenFP8AttentionRunner:
    """Validate, plan and bind one all-FP8 packed-varlen attention problem.

    ``query`` / ``key`` / ``value`` are E4M3 (``torch.float8_e4m3fn``) ``[T, H,
    128]`` tensors or strided views of them (unit innermost stride, 16-byte
    aligned head / token strides and base, as the BF16 route; the three may
    differ in strides) with one per-tensor FP32 descale each: the kernel folds
    ``q_descale * k_descale`` into the softmax exponent and ``v_descale`` into
    the output scale, the probabilities are E4M3 at unit scale and ``out`` is
    BF16.  The segment plan is the BF16 planner's (:func:`cached_bf16_segment_plan`,
    shared cache); the attention program is the general one, or its tiny-plan
    build when every segment has at most ``FP8_TINY_SEGMENT_TOKENS`` tokens
    (:func:`fp8_attention_stage`), and K/V-split units are merged by the BF16
    route's combine program.  Preparation allocates the optional ``out`` and
    the descale tensors given as floats; the returned runner launches with no
    allocation.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen FP8 attention supports backend='cake'")
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out, strided_qkv=True, dtype=torch.float8_e4m3fn
    )
    if isinstance(cu_seqlens, torch.Tensor) and (
        not cu_seqlens.is_cuda or cu_seqlens.device != device
    ):
        raise ValueError("cu_seqlens must be an int32 CUDA tensor on the query device")
    bounds = normalize_cu_seqlens(
        cu_seqlens, total_tokens=total_tokens, cu_seqlens_host=cu_seqlens_host
    )
    arch = _arch_for(device)
    route = select_route("fp8", arch)
    if softmax_scale is None:
        softmax_scale = HEAD_DIM**-0.5
    if out is None:
        out = torch.empty(
            (total_tokens, num_heads, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
    descales = (
        _descale_tensor("q_descale", q_descale, device),
        _descale_tensor("k_descale", k_descale, device),
        _descale_tensor("v_descale", v_descale, device),
    )
    plan = cached_bf16_segment_plan(bounds, device, num_heads)
    total_tiles = int(plan.total_tiles)
    main_kwargs = dict(
        Q=query,
        K=key,
        V=value,
        O=out,
        O_raw=out,
        unit_table=plan.unit_table,
        partial_O=plan.partial_O,
        partial_ML=plan.partial_ML,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
        total_tiles=total_tiles,
        num_heads=num_heads,
        softmax_scale_log2=float(softmax_scale) / math.log(2.0),
        grid=minimax_h3_varlen_fp8_launch_grid(
            _device_facts(_device_index(device))[0], total_tiles=total_tiles
        ),
    )
    assert tuple(main_kwargs) == FP8_ATTENTION_KWARGS
    assert main_kwargs["grid"][0] == 2 * int(plan.num_clusters)
    attention_stage = fp8_attention_stage(bounds)
    entry: Optional[Callable[..., Any]] = None
    arguments: tuple = ()
    combine_entry: Optional[Callable[..., Any]] = None
    combine_arguments: tuple = ()
    if total_tiles > 0:
        program = route_program(route, attention_stage)
        entry, arguments = _bind_in_tree(program, arch, main_kwargs)
        if plan.num_combine_units > 0:
            combine_entry, combine_arguments = _bind_stage(
                route["modules"]["combine"],
                arch,
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
    return MiniMaxH3VarlenFP8AttentionRunner(
        "fp8",
        arch,
        plan,
        main_kwargs,
        attention_stage,
        out,
        entry,
        arguments,
        combine_entry,
        combine_arguments,
    )


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

    ``query`` / ``key`` / ``value`` may be strided ``[T, H, 128]`` views (unit
    innermost stride, 16-byte-aligned head and token strides and base; for
    example the column slices of a fused QKV projection or the slices of a
    ``[T, H, 3, 128]`` pack): the kernel reads them in place, no copies are
    made.  ``out`` is contiguous.  The segment plan comes from the
    most-recently-used plan cache (:func:`cached_bf16_segment_plan`): the
    first preparation of a ``(cu_seqlens, num_heads, device, stream)`` builds
    and uploads the tables, later ones reuse them, so the only allocation
    here is the optional output; the returned runner launches with none.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen attention supports backend='cake'")
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out, strided_qkv=True
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
    plan = cached_bf16_segment_plan(bounds, device, num_heads)
    total_tiles = int(plan.total_tiles)
    attention_stage = bf16_attention_stage(plan, arch)
    softmax_scale_log2 = float(softmax_scale) / math.log(2.0)
    if attention_stage == "attention_tiny":
        tiny = plan.tiny_cu_seqlens
        segments = sum(1 for a, b in zip(bounds, bounds[1:], strict=False) if b > a)
        main_kwargs = dict(
            Q=query,
            K=key,
            V=value,
            O=out,
            cu_seqlens=tiny,
            num_heads=num_heads,
            softmax_scale_log2=softmax_scale_log2,
            grid=(num_heads, segments, 1),
        )
        assert tuple(main_kwargs) == BF16_TINY_ATTENTION_KWARGS
    else:
        main_kwargs = dict(
            Q=query,
            K=key,
            V=value,
            O=out,
            O_raw=out,
            unit_table=plan.unit_table,
            partial_O=plan.partial_O,
            partial_ML=plan.partial_ML,
            total_tiles=total_tiles,
            num_heads=num_heads,
            softmax_scale_log2=softmax_scale_log2,
            grid=(2 * int(plan.num_clusters), 1, 1),
        )
        assert tuple(main_kwargs) == BF16_ATTENTION_KWARGS
    entry: Optional[Callable[..., Any]] = None
    arguments: tuple = ()
    combine_entry: Optional[Callable[..., Any]] = None
    combine_arguments: tuple = ()
    if attention_stage == "attention_tiny":
        # One CTA per (head, non-empty segment); the program's argument plan
        # must be exactly the tiny one (a route leaf naming another program
        # is refused, not bound to the wrong arguments).
        entry, arguments = _bind_stage(
            route["modules"][attention_stage], arch, main_kwargs, exact=True
        )
    elif total_tiles > 0:
        # Per-plan program dispatch: the plan's ``dyn_sched`` / ``sread_unified``
        # flags and the arch filter select one of the route's BF16 attention
        # programs (``bf16_attention_stage``).
        entry, arguments = _bind_stage(
            route["modules"][attention_stage], arch, main_kwargs
        )
        if plan.num_combine_units > 0:
            combine_entry, combine_arguments = _bind_stage(
                route["modules"]["combine"],
                arch,
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
        attention_stage,
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
    ``query`` / ``key`` / ``value`` need ``NVFP4_INPUT_ALIGN``-byte-aligned
    bases and a caller-owned ``workspace`` ``NVFP4_WORKSPACE_ALIGN``-byte-aligned
    tensors.
    """
    if backend != "cake":
        raise ValueError("MiniMax-H3 varlen NVFP4 attention supports backend='cake'")
    if pv_mode not in PV_MODES:
        raise ValueError(f"pv_mode must be one of {PV_MODES}, got {pv_mode!r}")
    variant = NVFP4_VARIANT[pv_mode]
    total_tokens, num_heads, device = validate_minimax_h3_varlen_inputs(
        query, key, value, out, strided_qkv=False
    )
    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if tensor.data_ptr() % NVFP4_INPUT_ALIGN:
            raise ValueError(
                f"{name} base address must be {NVFP4_INPUT_ALIGN}-byte aligned for "
                "the NVFP4 route (its quantizers read 32-byte vectors)"
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
            if tensor.data_ptr() % NVFP4_WORKSPACE_ALIGN:
                raise ValueError(
                    f"workspace[{name!r}] base address must be "
                    f"{NVFP4_WORKSPACE_ALIGN}-byte aligned"
                )
    PB = int(plan.PB)
    tiles = nvfp4_tile_tables(plan, num_heads, device, pv_mode, arch)
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
        quantize_kwargs.update(
            v_fp8=workspace["v_fp8"],
            v_amax=workspace["v_amax"],
            v_amax_partial=workspace["v_amax_partial"],
        )
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
    stage_kwargs: dict[str, dict] = {}
    if pv_mode == "fp8":
        stage_kwargs["amax"] = amax_kwargs(value, workspace)
    stage_kwargs["quantize"] = quantize_kwargs
    if pv_mode == "fp4":
        # The by-value-table quantizer takes the same parameters, the two
        # tables as QUANTIZE_TP_MAX_PB-entry CPU tensors (``None`` for a plan
        # that binds ``quantize``); one of the two is bound.
        stage_kwargs["quantize_tp"] = {
            **quantize_kwargs,
            "block_token": plan.tp_block_token,
            "block_valid": plan.tp_block_valid,
        }
    if pv_mode == "fp8":
        # The early-Q/K quantizer takes the same parameters; one of the two is bound.
        stage_kwargs["quantize_early_qk"] = quantize_kwargs
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
        num_tiles=total_tiles,
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
    )
    if pv_mode == "fp4":
        pv_operands = dict(
            Vt=workspace["v_fp4_t"],
            SFVtLo=workspace["v_scale_lo"],
            SFVtHi=workspace["v_scale_hi"],
        )
    else:
        pv_operands = dict(V=workspace["v_fp8"], v_amax=workspace["v_amax"])
    # Every program takes the unit count (the two-stage dense program the
    # plan's cluster-tile count instead); the split program additionally the
    # K/V range / partial-slot tables and the partial workspace; exactly one
    # of them is bound.
    dense_common = attention_common
    if (pv_mode, arch) in NVFP4_TWO_STAGE_DENSE:
        dense_common = {
            ("total_clusters" if name == "num_tiles" else name): (
                int(plan.total_clusters) if name == "num_tiles" else value
            )
            for name, value in attention_common.items()
        }
    attention_kwargs = {**dense_common, **pv_operands}
    attention_split_kwargs = {
        **attention_common,
        **attention_split_extra,
        **pv_operands,
    }
    assert tuple(attention_kwargs) == nvfp4_attention_kwargs(pv_mode, arch, "attention")
    assert tuple(attention_split_kwargs) == nvfp4_attention_kwargs(
        pv_mode, arch, "attention_split"
    )
    stage_kwargs["attention"] = attention_kwargs
    if pv_mode == "fp8":
        # The single-stage dense program of a NVFP4_QS_DENSE pair (bound for
        # its Q-split plans) takes the unit count; the route of any other
        # architecture names no such stage and never binds it.
        stage_kwargs["attention_qs"] = {**attention_common, **pv_operands}
        assert tuple(stage_kwargs["attention_qs"]) == NVFP4_ATTENTION_FP8PV_KWARGS
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
    attention_stage = nvfp4_attention_stage(tiles, pv_mode, arch)
    quantize_stage = nvfp4_quantize_stage(
        pv_mode, num_heads, PB, _device_facts(_device_index(device))[0], arch=arch
    )
    stages: list[tuple[str, Optional[Callable[..., Any]], tuple]] = []
    for stage in STAGES[variant]:
        skipped = (
            total_tiles == 0
            or (stage == "combine" and tiles.num_combine_units == 0)
            or (stage in NVFP4_ATTENTION_STAGES and stage != attention_stage)
            or (stage in QUANTIZER_STAGES and stage != quantize_stage)
        )
        if skipped:
            stages.append((stage, None, ()))
            continue
        # ``quantize_tp`` takes its tables by value: a route leaf naming a
        # program that takes them as device buffers is refused.
        entry, arguments = _bind_stage(
            route["modules"][stage],
            arch,
            stage_kwargs[stage],
            exact=stage in (*NVFP4_ATTENTION_STAGES, "quantize_tp"),
            by_value=QUANTIZE_TP_BY_VALUE if stage == "quantize_tp" else (),
        )
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
    """BF16 packed-varlen attention in one call: plan, bind and launch.

    The segment plan is resolved from ``cu_seqlens`` through the
    most-recently-used plan cache (:func:`cached_bf16_segment_plan`): the
    first call for a segment layout builds and uploads the tables, every
    later call with the same ``cu_seqlens`` / ``num_heads`` on the same
    device and stream re-launches with them (no planning, no host-to-device
    copies, no allocation when ``out`` is given).  Without ``cu_seqlens_host``
    the int32 CUDA tensor is read back to the host first (one stream
    synchronization); pass ``cu_seqlens_host`` (the same offsets as a
    Python sequence) to call without any device synchronization.  For CUDA
    Graph capture prepare once with
    :func:`prepare_minimax_h3_varlen_attention` (or warm this entry up with
    the exact ``cu_seqlens`` on the capture stream) and replay the runner.
    """
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


def minimax_h3_varlen_fp8_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    q_descale: Union[float, torch.Tensor],
    k_descale: Union[float, torch.Tensor],
    v_descale: Union[float, torch.Tensor],
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> torch.Tensor:
    """All-FP8 packed-varlen attention: :func:`prepare_minimax_h3_varlen_fp8_attention` then one launch."""
    return prepare_minimax_h3_varlen_fp8_attention(
        query,
        key,
        value,
        cu_seqlens,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend=backend,
    ).launch()


E4M3_MAX = 448.0
_QUANT_CHUNK_ELEMENTS = (
    1 << 28
)  # FP32 staging chunk of the convenience quantizer (1 GiB)


def quantize_minimax_h3_e4m3_per_tensor(
    x: torch.Tensor,
    *,
    out: Optional[torch.Tensor] = None,
    descale: Optional[torch.Tensor] = None,
    staging: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor E4M3 quantization of a BF16 ``[T, H, 128]`` tensor (or strided view) with torch ops.

    ``descale = amax(|x|) / 448`` (one-element FP32 CUDA tensor, the value the FP8 route's ``*_descale`` arguments
    take) and ``out = e4m3(clamp(x * 448 / amax))`` (contiguous ``torch.float8_e4m3fn``).  ``out`` / ``descale`` /
    ``staging`` may be caller-owned so repeated quantization of one problem allocates nothing: ``staging`` is an FP32
    buffer of at least ``min(numel, 2**28)`` elements.  No host synchronization: the amax stays on the device.
    """
    if x.dtype != torch.bfloat16 or x.dim() != 3 or x.shape[-1] != HEAD_DIM:
        raise ValueError(
            f"expected a BF16 [T, H, {HEAD_DIM}] tensor, got {tuple(x.shape)} {x.dtype}"
        )
    if out is None:
        out = torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=x.device)
    elif (
        out.shape != x.shape
        or out.dtype != torch.float8_e4m3fn
        or not out.is_contiguous()
    ):
        raise ValueError(
            "out must be a contiguous float8_e4m3fn tensor of the input's shape"
        )
    if descale is None:
        descale = torch.empty(1, dtype=torch.float32, device=x.device)
    amax = torch.linalg.vector_norm(x, ord=float("inf"), dtype=torch.float32)
    amax = torch.clamp(amax, min=torch.finfo(torch.float32).tiny)
    torch.div(amax, E4M3_MAX, out=descale.reshape(()))
    scale = torch.reciprocal(descale.reshape(()))
    rows = max(1, _QUANT_CHUNK_ELEMENTS // (x.shape[1] * HEAD_DIM))
    T = x.shape[0]
    n_stage = min(T, rows) * x.shape[1] * HEAD_DIM
    if staging is None:
        staging = torch.empty(n_stage, dtype=torch.float32, device=x.device)
    elif staging.numel() < n_stage or staging.dtype != torch.float32:
        raise ValueError(
            f"staging must be an FP32 buffer of at least {n_stage} elements"
        )
    for start in range(0, T, rows):
        stop = min(T, start + rows)
        chunk = staging[: (stop - start) * x.shape[1] * HEAD_DIM].view(
            stop - start, x.shape[1], HEAD_DIM
        )
        chunk.copy_(x[start:stop])
        chunk.mul_(scale).clamp_(-E4M3_MAX, E4M3_MAX)
        out[start:stop].copy_(chunk)
    return out, descale


class MiniMaxH3VarlenFP8AttentionBF16Runner:
    """The BF16 convenience form of the all-FP8 route: quantize, then attend.

    ``quantize()`` recomputes the three per-tensor descales and E4M3 operands from the bound BF16 tensors (torch
    ops into preallocated buffers, no allocation), ``attention()`` launches the prepared FP8 runner on them, and
    calling the runner does both.  The two halves are separable so quantization and attention can be timed apart.
    """

    def __init__(
        self,
        operands: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        e4m3: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        descales: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        staging: torch.Tensor,
        attention: MiniMaxH3VarlenFP8AttentionRunner,
    ) -> None:
        self.operands = operands
        self.e4m3 = e4m3
        self.descales = descales
        self._staging = staging
        self.attention_runner = attention
        self.out = attention.out

    def quantize(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        for x, x8, d in zip(self.operands, self.e4m3, self.descales, strict=True):
            quantize_minimax_h3_e4m3_per_tensor(
                x, out=x8, descale=d, staging=self._staging
            )
        return self.e4m3

    def attention(self) -> torch.Tensor:
        return self.attention_runner.launch()

    def launch(self) -> torch.Tensor:
        self.quantize()
        return self.attention_runner.launch()

    __call__ = launch


def prepare_minimax_h3_varlen_fp8_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> MiniMaxH3VarlenFP8AttentionBF16Runner:
    """Bind BF16 ``query`` / ``key`` / ``value`` (contiguous or the BF16 route's strided views) to the all-FP8 route.

    Allocates the three E4M3 operand buffers, the descale tensors and one FP32 staging chunk, quantizes once
    (per-tensor ``amax / 448`` descales, :func:`quantize_minimax_h3_e4m3_per_tensor`) and prepares
    :func:`prepare_minimax_h3_varlen_fp8_attention` on the E4M3 buffers with the descale tensors; the returned
    runner re-quantizes and launches with no further allocation.  The quantization is torch work on the host
    stream, not part of the kernel; time it apart from the attention launch.
    """
    validate_minimax_h3_varlen_inputs(
        query, key, value, out, strided_qkv=True, dtype=torch.bfloat16
    )
    device = query.device
    operands = (query, key, value)
    e4m3 = tuple(
        torch.empty(x.shape, dtype=torch.float8_e4m3fn, device=device) for x in operands
    )
    descales = tuple(
        torch.empty(1, dtype=torch.float32, device=device) for _ in operands
    )
    rows = max(1, _QUANT_CHUNK_ELEMENTS // (query.shape[1] * HEAD_DIM))
    staging = torch.empty(
        min(query.shape[0], rows) * query.shape[1] * HEAD_DIM,
        dtype=torch.float32,
        device=device,
    )
    for x, x8, d in zip(operands, e4m3, descales, strict=True):
        quantize_minimax_h3_e4m3_per_tensor(x, out=x8, descale=d, staging=staging)
    attention = prepare_minimax_h3_varlen_fp8_attention(
        e4m3[0],
        e4m3[1],
        e4m3[2],
        cu_seqlens,
        q_descale=descales[0],
        k_descale=descales[1],
        v_descale=descales[2],
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend=backend,
    )
    return MiniMaxH3VarlenFP8AttentionBF16Runner(
        operands, e4m3, descales, staging, attention
    )


def minimax_h3_varlen_fp8_attention_bf16(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cu_seqlens: Union[torch.Tensor, Sequence[int]],
    *,
    softmax_scale: Optional[float] = None,
    out: Optional[torch.Tensor] = None,
    cu_seqlens_host: Optional[Sequence[int]] = None,
    backend: str = "cake",
) -> torch.Tensor:
    """BF16 convenience entry: per-tensor E4M3 quantization of ``query`` / ``key`` / ``value`` (torch), then
    :func:`minimax_h3_varlen_fp8_attention` with the three ``amax / 448`` descales.  Equivalent to quantizing with
    :func:`quantize_minimax_h3_e4m3_per_tensor` and calling the FP8 entry on the results."""
    return prepare_minimax_h3_varlen_fp8_attention_bf16(
        query,
        key,
        value,
        cu_seqlens,
        softmax_scale=softmax_scale,
        out=out,
        cu_seqlens_host=cu_seqlens_host,
        backend=backend,
    ).attention()


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
    """NVFP4 packed-varlen attention in one call: plan, quantize, attend.

    Every call plans from ``cu_seqlens`` and allocates the packed operand
    workspace.  Without ``cu_seqlens_host`` the int32 CUDA tensor is read
    back to the host first (one stream synchronization); pass
    ``cu_seqlens_host`` to plan without any device synchronization.  For
    repeated launches of one problem or CUDA Graph capture prepare once
    with :func:`prepare_minimax_h3_varlen_nvfp4_attention` (optionally
    with a caller-owned ``workspace``) and call the runner.
    """
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
