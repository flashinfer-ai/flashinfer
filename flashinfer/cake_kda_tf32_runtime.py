"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations

from functools import partial

from flashinfer.jit import cake_kda_tf32 as _kda_registry
from flashinfer.jit.cake_kda_tf32 import _RENDERED_ROUTES, _factory
from flashinfer.jit.cake_kda_tf32 import device_arch as detect_gpu_arch
from flashinfer.kda_prefill import (
    _direct_m128_route,
    _make_lpt_task_bins,
    _pair_packed_beta_tma_source,
    _persistent_m128_roofline,
    _select_bf16_route,
    _should_use_bt16_dense_wavefront,
    _should_use_h12_direct_n32,
    _uses_measured_sm100_persistent_policy,
    _wave_quantized_bt16_prepare_ctas,
)

from enum import Enum


_COMPUTE_CAPABILITY = {"sm_100a": (10, 0), "sm_103a": (10, 3)}


# Canonical semantic compile axis shared by KDA schedules.
class KDAGateKind(str, Enum):
    """Gate algebra selected when a KDA schedule is traced."""

    LOWER_BOUND = "lower_bound"
    UNBOUNDED_SOFTPLUS = "unbounded_softplus"

    @property
    def trace_value(self) -> int:
        return {KDAGateKind.LOWER_BOUND: 1, KDAGateKind.UNBOUNDED_SOFTPLUS: 2}[self]


def gate_kind_from_lower_bound(lower_bound: float | None) -> KDAGateKind:
    """Resolve the public FlashKDA scalar contract to its compile-time algebra."""
    return (
        KDAGateKind.UNBOUNDED_SOFTPLUS
        if lower_bound is None
        else KDAGateKind.LOWER_BOUND
    )


def validate_kda_state_dtype(compute_dtype: str, *, state_dtype_is_fp32: bool) -> None:
    """Validate external initial/final state ABI; checkpoints remain BF16."""
    if compute_dtype not in ("bf16", "tf32"):
        raise ValueError("compute_dtype must be bf16 or tf32")
    if compute_dtype == "tf32" and (not state_dtype_is_fp32):
        raise ValueError(
            "TF32 compute requires FP32 external initial/final state; BF16 state is unsupported"
        )


def validate_kda_state_tensors(compute_dtype: str, initial_state, final_state) -> None:
    """Reject unsupported external state tensors before allocation or compilation."""
    import torch

    validate_kda_state_dtype(compute_dtype, state_dtype_is_fp32=True)
    for name, state in (("initial_state", initial_state), ("final_state", final_state)):
        if (
            compute_dtype == "tf32"
            and state is not None
            and (state.dtype != torch.float32)
        ):
            raise ValueError(
                f"TF32 compute requires FP32 external {name}; got {state.dtype}"
            )


def _build_kda_module(factory, *args, **kwargs):
    return factory(*args, **kwargs)


# FlashKDA-compatible runtime for Blackwell schedules.
#
# Minimum architecture: sm_100a.
#
# The public ``fwd`` boundary adds ``compute_dtype="bf16"`` to the FlashKDA
# arguments. BF16 compute retains BF16/FP32 external state support; TF32 compute
# requires FP32 initial/final state. Checkpoints remain BF16. Families without
# a validated TF32 implementation report an explicit unsupported request. BF16 H12 dispatches
# between the chunk-16 and chunk-32 direct M128 bodies at their measured sequence
# length crossover; sequences that fit in one 16-token tile and checkpoint
# intervals that require a physical 16-token boundary retain chunk-16.
# Active-FP32-beta H12 checkpoint64 requests on 152-SM GB300 use the M64 value
# split for 129-256-token residuals whose doubled task grid fits one SM wave.
# Other head counts dispatch among those direct bodies, the source static-binned
# persistent M128 body, its cross-CTA recurrence-piece specialization for
# quantization-bound uniform grids, and the M64 value-row split. FP32 state
# reuses those physical schedules as state-I/O specializations where validated.
# Under-parallelized ultra-long fixed layouts additionally use a split-sequence
# affine-prefix DAG derived from FlashInfer PR4779; its main, map, and correction
# windows reuse the same variable-shape direct-M128 identity. Remaining regions
# retain the source-derived compatibility fallback.
import heapq
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch
HEAD_DIM = 128
BF16_M128_CHUNK = 32
BF16_N16_M128_CHUNK = 16
# Most 32-token chunks of the longest sequence of a work-binned direct M128 N32
# serving launch that takes the loader's turnover build of its module (see
# _work_bin_turnover_route): with such short tasks a bin turns its state over
# often enough that loading the next task's initial state and storing the
# previous task's final state beside the current task's chunks pays.  A launch
# with a longer sequence keeps the routed work-binned program.  Measured per
# part: BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300 on a 152-SM sm_103a.
BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS = 8
BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300 = 16
BF16_BETA_TMA_MIN_HEADS = 8
BF16_DIRECT_ORDER_MAX_PASSES = 1
BF16_DIRECT_INSERTION_MAX_SEQUENCES = 16
BF16_DIRECT_FIXED_CHUNK_COST = 6
H12_DIRECT_N32_MAX_SEQ_LEN = 256
H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQ_LEN = 128
BF16_N32_REGISTER_INVERSE_MIN_SEQ_LEN = 256
INDEPENDENT_DVSPLIT_CTAS = 2
# Shortest longest-sequence of an M64 value-split launch that takes the
# loader's pair-share build of the module (see _m64_pair_share_factory), by
# the launch's long-task load, heads x tokens / longest sequence (its task
# count when the sequences are equal; a short sequence counts by its length):
# the first (most tasks, shortest longest-sequence) row that holds the load
# decides, and a launch beyond the last row keeps the routed program.
# Off-bench sweeps measured the pair-share build faster only on long
# sequences, and the more long tasks share the wave, the longer the sequence
# it needs: on a 148-SM B200 it is slower beyond 48 equal tasks at every
# length, while a 152-SM GB300 keeps it faster up to 72.
M64_PAIR_SHARE_MIN_SEQ_LEN = ((32, 1024), (48, 2048))
M64_PAIR_SHARE_MIN_SEQ_LEN_GB300 = ((32, 1024), (64, 2048), (72, 4096))
# The M64 value split's CTA: 1024 threads and the dynamic shared memory that
# keeps it at one CTA per SM (the cluster capacity probe launches this shape).
M64_CTA_THREADS = 1024
M64_CTA_SMEM_BYTES = 227328
# Architectures whose one-wave BF16 grid sweep selected the M64 value split over
# every direct M128 tile (see _should_use_bf16_one_wave_dvsplit).
BF16_ONE_WAVE_DVSPLIT_ARCHES = ("sm_100a", "sm_103a")
# The M64 value split's BF16-pool body carries the recurrent state between
# 32-token chunks in BF16 (the chunk state is re-derived as an MMA of the BF16
# projection copy with a BF16 decay diagonal).  With a bounded gate that
# rounding accumulates under trained Kimi-K3 deep-layer statistics (beta ~ 1,
# exp(A_log) ~ 1): worst-head final-state rrmse against FP32 Triton 0.011 at
# 384 tokens, 0.014 at 512, 0.051 at 2241, 0.19 at 8192 (CAKE-736 rounds 7-8,
# B200 and GB300, H12 and H16), while the direct M128 N32/N16 bodies carry FP32
# chunk state and stay below 0.01 at every length.  Only single-chunk
# residuals (no chunk-to-chunk carrier) keep the split's measured preference on
# a BF16 pool.  Since CAKE-736 round 9 the split's decay panels accumulate
# bf16(S) * (D - I) onto the FP32 TMEM state instead of re-deriving bf16(S) * D
# into it ("delta decay"): the state is never rounded between chunks and the
# rounding of the correction is scaled by |1 - d|, so the accumulated error is
# bounded by one BF16 ulp of |S| at every length.  The FP32-pool route is
# validated on that body (worst-head state rrmse <= 0.008 at 8192 tokens, B200
# and GB300) and keeps the split at every length.  The same body measures
# 0.0071-0.0086 on a BF16 pool (B200, T 512-8192), but no BF16-pool split module
# is exported, so lifting the guard below is an export-inventory change kept for
# a separate round; the guard stays for BF16 pools only.
BF16_DVSPLIT_STATE_CARRIER_MAX_SEQ_LEN = 64
BT16_CHUNK = 16
BT16_VALUE_SPLITS = 2
TF32_BT16_PREP_RESIDENT_CTAS = 6
TF32_BT16_CHECKPOINT_TASK_DIVISOR = 14
BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA = 2
BT16_GENERAL_HIGH_WORK_CHUNKS_PER_PREP_CTA = 8
BT16_GENERAL_HIGH_WORK_MIN_CHUNK_HEADS = 16384
BT16_H12_CHUNKS_PER_PREP_CTA = 4
BT16_H12_CPC1_MAX_TOTAL_CHUNKS = 128
BT16_DENSE_PREP_WAVES = 5
# Sequence x head tasks the affine composite accepts.  The window budget
# (``sm_count // num_heads`` windows) keeps the part grid to one wave; the cap
# only bounds host-side plan size (8 sequences x 16 heads).
AFFINE_SPLIT_MAX_TASKS = 128
LEGACY_AFFINE_SPLIT_MAX_TASKS = 32
AFFINE_SPLIT_MIN_CHUNKS = 256
AFFINE_SPLIT_MIN_CHUNKS_PER_PART = 32
AFFINE_SPLIT_MIN_PARTS = 8
AFFINE_SPLIT_LOW_PART_MIN_CHUNKS = 2048
BF16_AFFINE_SPLIT_MIN_CHUNKS = 128
# Measured cost model of the BF16 fused family, in microseconds.  The affine
# composite runs its main, correction and map passes on every window at once,
# so its time follows the longest window's chunk chain; the sequential fused
# body runs one chain per sequence x head task, so its time follows the longest
# sequence's chain (the number of tasks barely matters below the SM count).
# Both are affine in their chain length: (fixed, per 32-token chunk).  Fitted
# from paired forced-split / forced-sequential runs (fresh inputs, plan-cache
# hits, profiler GPU time) on 15 shapes per gate; H12 and H16 share the fit.
# sm_100a (B200, 148 SMs), 2026-09-25: composite 75 + 14.2 * chunks/window,
# sequential 20 + 4.0 * chunks (unbounded softplus, FP32 rows); composite
# 84 + 8.8 * chunks/window, sequential 16 + 1.75 * chunks (bounded gate).
# Windows never go below 8 chunks (256 tokens): the fused-body specialization
# flags flip at 128 / 256 / 512 tokens of launch max_seq_len (H12 early state
# pack, scalar beta / generic register inverse, sm_103a prediction-first), and
# the >= 256-token launch classes map onto exported programs (the 359-row
# export with 8-chunk windows added no program variant); the < 256-token
# class does not, so shorter windows would select unexported programs.
AFFINE_MIN_CHUNKS_PER_WINDOW = 8
# Resident-window budget in CTA waves (windows x heads per wave = SM count).
# One wave is the measured default; FLASHINFER_KDA_AFFINE_WINDOW_WAVES=2 is an
# A/B knob for shapes whose passes are chain-latency bound.
AFFINE_WINDOW_WAVES_ENV = "FLASHINFER_KDA_AFFINE_WINDOW_WAVES"
AFFINE_MAX_WINDOW_WAVES = 3
AFFINE_MULTI_WAVE_MIN_GAIN = 0.15


def _affine_window_waves() -> int | None:
    """Forced resident-window waves (``FLASHINFER_KDA_AFFINE_WINDOW_WAVES``), ``None`` = cost-model choice."""
    import os

    raw = os.environ.get(AFFINE_WINDOW_WAVES_ENV, "")
    if raw in ("", "auto"):
        return None
    try:
        waves = int(raw)
    except ValueError:
        return None
    return min(AFFINE_MAX_WINDOW_WAVES, max(1, waves))


AFFINE_BF16_COST_MODEL_US: dict[tuple[str, str], tuple[float, float, float, float]] = {
    # (gpu_arch, gate_kind): (composite_fixed, composite_per_window_chunk,
    #                          sequential_fixed, sequential_per_chunk)
    ("sm_100a", "unbounded_softplus"): (75.0, 14.2, 20.0, 4.0),
    ("sm_100a", "lower_bound"): (84.0, 8.8, 16.0, 1.75),
    # sm_103a (GB300, 152 SMs), 2026-09-25: composite 95 + 11.7 * chunks/window
    # (H12 87 + 12.05, H16 102 + 11.4), sequential 18 + 3.6 * chunks.
    ("sm_103a", "unbounded_softplus"): (95.0, 11.7, 18.0, 3.6),
    # bounded gate: composite 83 + 7.8 (H12 77 + 8.1, H16 89 + 7.5), sequential 14 + 1.66.
    ("sm_103a", "lower_bound"): (83.0, 7.8, 14.0, 1.66),
}
SMALL_BH_GROUP_SIZE = 8
SMALL_BH_MAX_TASKS = 8
SMALL_BH_MIN_SEQ_LEN = 2048
BF16_N32_PREDICTION_FIRST_MIN_HEADS = 24
BF16_N32_PREDICTION_FIRST_MIXED_MIN_SEQ_LEN = 512
BF16_ROUTE_DIRECT_M128 = "direct_m128"
BF16_ROUTE_DIRECT_M128_N16 = "direct_m128_n16"
BF16_ROUTE_HEAD_GROUPED_M128 = "head_grouped_persistent_m128"
BF16_ROUTE_LPT_M128 = "lpt_persistent_m128"
BF16_ROUTE_SCALAR_CHUNK_LPT_M128 = "scalar_chunk_lpt_m128"
BF16_ROUTE_SOURCE_VTILE_M128 = "vtile_m128"
BF16_ROUTE_PIECE_M128 = "piece_persistent_m128"
BF16_ROUTE_M64 = "independent_dvsplit_m64"
BF16_ROUTE_BT16_M64 = "bt16_prepare_chain_m64"
BF16_ROUTE_SMALL_BH_M128 = "small_bh_owner_helper_m128"
BF16_ROUTE_AFFINE_SPLIT_M128 = "affine_split_m128"


def _affine_split_policy() -> str:
    """``model`` (default) or ``legacy``.

    ``model`` decides the BF16-family split from the measured cost model
    (``AFFINE_BF16_COST_MODEL_US``): the composite is taken when its estimated
    time on the planned windows beats the sequential fused body's longest
    chain.  ``legacy`` keeps the pre-2026-09-23 gate (aggregate sequence x head
    tasks, 32-task cap, fixed chunk thresholds) for A/B measurement.
    ``packed`` is accepted as the old name of the default.
    """
    import os

    policy = os.environ.get("FLASHINFER_KDA_AFFINE_POLICY", "model")
    return "legacy" if policy == "legacy" else "model"


def _affine_window_counts(
    chunk_counts: list[int], targets: list[int], window_budget: int
) -> list[int]:
    """Share the resident-window budget across original sequences.

    Every sequence keeps at least one window; the remaining budget goes to the
    sequence whose windows are currently longest (stable sequence tie-break),
    never beyond its own target.
    """
    counts = targets.copy()
    if sum(targets) > window_budget:
        counts = [1] * len(chunk_counts)
        for _ in range(window_budget - len(chunk_counts)):
            eligible = [i for i in range(len(counts)) if counts[i] < targets[i]]
            if not eligible:
                break
            selected = max(
                eligible,
                key=lambda i: ((chunk_counts[i] + counts[i] - 1) // counts[i], -i),
            )
            counts[selected] += 1
    return counts


def _affine_window_chunks(chunks: int, parts: int, checkpoints: bool) -> int:
    """Chunks per window for one sequence; checkpointed windows start on 64-token boundaries."""
    per_part = (chunks + parts - 1) // parts
    if checkpoints:
        per_part += per_part % 2
    return per_part


def _affine_bf16_window_targets(
    chunk_counts: list[int], *, num_heads: int, sm_count: int, waves: int = 1
) -> list[int]:
    """Most windows the BF16 composite can give each sequence.

    ``waves`` waves of windows per head (``sm_count // num_heads`` each), at
    least ``AFFINE_MIN_CHUNKS_PER_WINDOW`` chunks per window (256-token
    launches): shorter windows only add per-window preparation while the
    main/correction passes stay chain-bound.
    """
    per_head = max(1, sm_count // num_heads) * waves
    return [
        min(per_head, max(1, chunks // AFFINE_MIN_CHUNKS_PER_WINDOW))
        for chunks in chunk_counts
    ]


def _affine_max_tasks() -> int:
    return (
        LEGACY_AFFINE_SPLIT_MAX_TASKS
        if _affine_split_policy() == "legacy"
        else AFFINE_SPLIT_MAX_TASKS
    )


def _affine_split_part_count(
    *,
    sm_count: int,
    tasks: int,
    chunks: int,
    fp32_indexed_state: bool = False,
    shared_tf32_factors: bool = False,
    unbounded_softplus: bool = False,
) -> int:
    """Resolve the runtime split scheduler without an exact-shape bucket."""
    min_chunks = AFFINE_SPLIT_MIN_CHUNKS
    if fp32_indexed_state:
        min_chunks = max(min_chunks, tasks * AFFINE_SPLIT_MIN_CHUNKS_PER_PART)
    if not shared_tf32_factors:
        min_chunks = max(BF16_AFFINE_SPLIT_MIN_CHUNKS, tasks * 8)
    if shared_tf32_factors and unbounded_softplus:
        min_chunks = max(64, tasks * 8)
    if (
        tasks <= 0
        or tasks > _affine_max_tasks()
        or 2 * tasks > sm_count
        or (chunks < min_chunks)
    ):
        return 1
    parts = min(
        sm_count,
        max(2, sm_count // tasks),
        max(2, chunks // AFFINE_SPLIT_MIN_CHUNKS_PER_PART),
    )
    if not shared_tf32_factors or unbounded_softplus:
        parts = min(sm_count, max(2, sm_count // tasks), max(2, chunks // 8))
    min_parts = AFFINE_SPLIT_MIN_PARTS if shared_tf32_factors else 4
    if parts < min_parts and chunks < AFFINE_SPLIT_LOW_PART_MIN_CHUNKS:
        return 1
    if shared_tf32_factors and (not unbounded_softplus):
        parts = min(sm_count, max(2, sm_count // tasks), max(2, chunks // 8))
    return parts


def _affine_window_budget(
    num_sequences: int,
    num_heads: int,
    sm_count: int,
    waves: int,
    *,
    unsplittable: int = 0,
) -> int:
    """Windows the composite may plan for one call.

    ``waves`` waves of ``sm_count // num_heads`` windows per head are shared by
    the sequences that can split; every sequence that cannot (``unsplittable``,
    target of one window) still gets its window on top of that budget.  Its
    CTAs run a chain of at most ``2 * AFFINE_MIN_CHUNKS_PER_WINDOW`` chunks and
    free their SMs long before a resident wave of long windows ends, so they
    never extend a wave; charging them against the budget only shortened the
    long sequences' window count (2x(8128+64) H16 on 148 SMs: 7 windows for
    two 254-chunk sequences, 86-chunk windows, instead of 9 and 64).
    """
    return max(num_sequences, sm_count // num_heads * waves + unsplittable)


def _affine_unsplittable(targets: list[int]) -> int:
    """Sequences whose window target is one: they cannot split and do not draw on the wave budget."""
    return sum(1 for target in targets if target <= 1)


def _affine_bf16_composite_estimate_us(
    chunk_counts: list[int],
    *,
    num_heads: int,
    sm_count: int,
    checkpoints: bool,
    model: tuple,
    waves: int,
):
    """Modelled composite microseconds on the windows ``waves`` waves would plan; ``None`` if nothing splits.

    Every pass is chain-bound on its longest window, and the resident waves of
    one pass run back to back, so the per-window-chunk slope scales with the
    wave count while the fixed part (launch chain, scan, epilogue) is paid
    once.  Checked on the B200 two-wave lanes (round v6): H16 2x8192 927 vs
    933 us measured, H12 2x(8128+64) 757 vs 777, H16 2x(8128+64) 984 vs 985,
    H16 4x4096 984 vs 914, H16 8192 501 vs 581.
    """
    targets = _affine_bf16_window_targets(
        chunk_counts, num_heads=num_heads, sm_count=sm_count, waves=waves
    )
    counts = _affine_window_counts(
        chunk_counts,
        targets,
        _affine_window_budget(
            len(chunk_counts),
            num_heads,
            sm_count,
            waves,
            unsplittable=_affine_unsplittable(targets),
        ),
    )
    if sum(counts) <= len(chunk_counts):
        return None
    window_chunks = max(
        _affine_window_chunks(chunks, parts, checkpoints)
        for chunks, parts in zip(chunk_counts, counts, strict=True)
    )
    composite_fixed, composite_chunk, _, _ = model
    return composite_fixed + waves * composite_chunk * window_chunks


def _affine_bf16_waves(
    chunk_counts: list[int],
    *,
    num_heads: int,
    sm_count: int,
    checkpoints: bool,
    gate_kind: str,
    gpu_arch: str | None,
) -> tuple[int, float | None]:
    """``(waves, composite_us)``: the forced wave count, else one wave unless more waves model clearly cheaper.

    A second or third wave is taken only when its estimate beats the one-wave
    composite by ``AFFINE_MULTI_WAVE_MIN_GAIN``: the model's error on the
    two-wave B200 lanes is up to 15 %, so smaller modelled gains are noise
    (H16 2x8192 927 vs 984 modelled, 933 vs 924 measured; H12 2x(8128+64) 757
    vs 813 modelled, 777 vs 772 measured).  With unsplittable sequences kept
    out of the wave budget (``_affine_window_budget``), 2x(8128+64) plans the
    same windows as 2x8192 (H16: 5+4 windows of 64 chunks, 984 modelled) and
    stays on one wave; the earlier 7-window plan (86-chunk windows, 1296) was
    the only case the two-wave rule accepted (measured 985 vs 1125 one wave).
    ``composite_us`` is ``None`` when the architecture/gate has no model or no
    wave count splits anything (the caller then keeps one wave).
    """
    forced = _affine_window_waves()
    model = AFFINE_BF16_COST_MODEL_US.get((gpu_arch, gate_kind))
    if model is None:
        return forced or 1, None
    best: tuple[int, float] | None = None
    for waves in (forced,) if forced else range(1, AFFINE_MAX_WINDOW_WAVES + 1):
        estimate = _affine_bf16_composite_estimate_us(
            chunk_counts,
            num_heads=num_heads,
            sm_count=sm_count,
            checkpoints=checkpoints,
            model=model,
            waves=waves,
        )
        if estimate is None:
            continue
        if best is None or estimate < best[1] * (1.0 - AFFINE_MULTI_WAVE_MIN_GAIN):
            best = (waves, estimate)
    return best if best is not None else (forced or 1, None)


def _affine_bf16_split_estimate_us(
    *,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    sm_count: int,
    checkpoints: bool,
    gate_kind: str,
    gpu_arch: str,
):
    """Estimated (composite, sequential) microseconds for a BF16-family call.

    The composite estimate is the cheapest wave count (``_affine_bf16_waves``).
    ``None`` when the architecture/gate has no measured model or the call is
    outside the composite's contract (task cap, window budget, no sequence
    that would actually split).
    """
    model = AFFINE_BF16_COST_MODEL_US.get((gpu_arch, gate_kind))
    tasks = len(sequence_lengths) * num_heads
    if (
        model is None
        or not sequence_lengths
        or min(sequence_lengths) <= 0
        or num_heads <= 0
        or tasks > _affine_max_tasks()
        or 2 * tasks > sm_count
    ):
        return None
    chunk_counts = [
        (length + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK for length in sequence_lengths
    ]
    _, composite_us = _affine_bf16_waves(
        chunk_counts,
        num_heads=num_heads,
        sm_count=sm_count,
        checkpoints=checkpoints,
        gate_kind=gate_kind,
        gpu_arch=gpu_arch,
    )
    if composite_us is None:
        return None
    _, _, sequential_fixed, sequential_chunk = model
    return composite_us, sequential_fixed + sequential_chunk * max(chunk_counts)


def _affine_split_windows(
    *,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    sm_count: int,
    fp32_indexed_state: bool,
    shared_tf32_factors: bool,
    checkpoints: bool,
    unbounded_softplus: bool = False,
    gpu_arch: str | None = None,
):
    """Partition original sequences into runtime windows without crossing boundaries."""
    tasks = len(sequence_lengths) * num_heads
    if (
        not sequence_lengths
        or min(sequence_lengths) <= 0
        or tasks > _affine_max_tasks()
    ):
        raise ValueError(
            "affine requires nonempty sequences and at most "
            f"{_affine_max_tasks()} sequence/head tasks"
        )
    chunk_counts = [
        (length + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK for length in sequence_lengths
    ]
    if not shared_tf32_factors and _affine_split_policy() == "model":
        # The cost model already decided the split and its wave count; give
        # every sequence the most windows that budget allows so the longest
        # window is shortest.
        if gpu_arch is None:
            # Planner-only callers (CPU tests) get the one-wave plan.
            try:
                gpu_arch = detect_gpu_arch()
            except Exception:
                gpu_arch = None
        waves, _ = _affine_bf16_waves(
            chunk_counts,
            num_heads=num_heads,
            sm_count=sm_count,
            checkpoints=checkpoints,
            gate_kind="unbounded_softplus" if unbounded_softplus else "lower_bound",
            gpu_arch=gpu_arch,
        )
        targets = _affine_bf16_window_targets(
            chunk_counts, num_heads=num_heads, sm_count=sm_count, waves=waves
        )
        window_budget = _affine_window_budget(
            len(sequence_lengths),
            num_heads,
            sm_count,
            waves,
            unsplittable=_affine_unsplittable(targets),
        )
    else:
        window_budget = _affine_window_budget(
            len(sequence_lengths), num_heads, sm_count, _affine_window_waves() or 1
        )
        targets = [
            _affine_split_part_count(
                sm_count=sm_count,
                tasks=num_heads,
                chunks=chunks,
                fp32_indexed_state=fp32_indexed_state,
                shared_tf32_factors=shared_tf32_factors,
                unbounded_softplus=unbounded_softplus,
            )
            for chunks in chunk_counts
        ]
        if len(sequence_lengths) > 1 and max(targets) > 1:
            chunks_per_window = 8
            targets = [
                max(target, min(window_budget, max(1, chunks // chunks_per_window)))
                for target, chunks in zip(targets, chunk_counts, strict=True)
            ]
    counts = _affine_window_counts(chunk_counts, targets, window_budget)
    token_offsets = [0]
    part_offsets = [0]
    for length, chunks, parts in zip(
        sequence_lengths, chunk_counts, counts, strict=True
    ):
        per_part = _affine_window_chunks(chunks, parts, checkpoints)
        start = token_offsets[-1]
        for chunk in range(per_part, chunks, per_part):
            token_offsets.append(start + chunk * BF16_M128_CHUNK)
        token_offsets.append(start + length)
        part_offsets.append(len(token_offsets) - 1)
    if part_offsets[-1] <= len(sequence_lengths):
        raise ValueError("affine split-sequence is below its runtime part crossover")
    return (tuple(token_offsets), tuple(part_offsets))


def _should_use_n32_register_inverse(
    *, direct_n16: bool, gate_kind: KDAGateKind, scalar_beta: bool, max_seq_len: int
) -> bool:
    """Select the live-accumulator inverse physical schedule.

    The register DAG removes two stage-local rendezvous, but its longer scalar
    transform is counterproductive on tiny generic tails.  Scalar-beta H12
    already uses that DAG across its measured N32 range.
    """
    return (
        not direct_n16
        and gate_kind == KDAGateKind.LOWER_BOUND
        and (scalar_beta or max_seq_len >= BF16_N32_REGISTER_INVERSE_MIN_SEQ_LEN)
    )


def _should_prioritize_n32_prediction(
    *,
    gpu_arch: str,
    route: str,
    uniform_sequences: bool,
    num_heads: int,
    max_seq_len: int,
    register_inverse: bool,
) -> bool:
    """Issue the recurrence-critical Kd prediction first on measured GB300 work.

    The Kd product releases residual recurrence immediately while independent
    Qd output projection finishes.  The ordering is stable for H24+ uniform
    work and for mixed grids whose longest sequence reaches 512 tokens.  Tiny
    mixed grids and H12 retain the joined Qd-first completion.
    """
    return (
        gpu_arch == "sm_103a"
        and route == BF16_ROUTE_DIRECT_M128
        and (num_heads >= BF16_N32_PREDICTION_FIRST_MIN_HEADS)
        and (max_seq_len >= BF16_N32_REGISTER_INVERSE_MIN_SEQ_LEN)
        and (
            uniform_sequences
            or max_seq_len >= BF16_N32_PREDICTION_FIRST_MIXED_MIN_SEQ_LEN
        )
        and register_inverse
    )


def _should_use_n32_tensor_state_decay(
    *,
    gpu_arch: str,
    route: str,
    uniform_sequences: bool,
    num_heads: int,
    total_tasks: int,
    max_seq_len: int,
    prediction_first: bool,
) -> bool:
    """Select the measured full-tile tensor-core state-decay schedule."""
    return (
        gpu_arch == "sm_103a"
        and route == BF16_ROUTE_DIRECT_M128
        and uniform_sequences
        and (num_heads >= 64)
        and (total_tasks >= 96)
        and (max_seq_len % BF16_M128_CHUNK == 0)
        and prediction_first
    )


def _pad_beta_tma_source(beta_flat, *, total_tokens: int, num_heads: int):
    """Allocate an encodable beta carrier for the fixed 8x32 TMA box.

    The descriptor row pitch must be a multiple of 16 bytes, so the BF16 head
    extent is rounded up to eight rather than merely clamped to eight.  The
    valid rectangle is refreshed by ``launch()``; keeping that copy in the
    launch stream makes in-place updates visible to CUDA Graph replay.
    """
    if tuple(beta_flat.shape) != (total_tokens, num_heads):
        raise ValueError("beta_flat must have shape [total_tokens, num_heads]")
    if beta_flat.stride(1) != 1:
        raise ValueError("beta must have unit stride in its head dimension")
    if (
        total_tokens >= BF16_M128_CHUNK
        and num_heads >= BF16_BETA_TMA_MIN_HEADS
        and (beta_flat.stride(0) * beta_flat.element_size() % 16 == 0)
    ):
        return beta_flat
    padded_tokens = max(total_tokens, BF16_M128_CHUNK)
    padded_heads = max(
        BF16_BETA_TMA_MIN_HEADS,
        (num_heads + BF16_BETA_TMA_MIN_HEADS - 1)
        // BF16_BETA_TMA_MIN_HEADS
        * BF16_BETA_TMA_MIN_HEADS,
    )
    return beta_flat.new_zeros((padded_tokens, padded_heads))


def _beta_tma_copy_required(
    offsets: tuple[int, ...] | list[int], *, chunk_tokens: int
) -> bool:
    """Return whether any sequence reaches a beta TMA load.

    Sequences shorter than one recurrence chunk use only scalar tail loads.
    Their tensor-map descriptor must still be encodable, but its padded data
    carrier is never read and therefore does not need a per-launch copy.
    """
    return any(
        (
            end - start >= chunk_tokens
            for start, end in zip(offsets, offsets[1:], strict=False)
        )
    )


def _flatten_token_storage(
    tensor, *, logical_tokens_per_batch: int, trailing_shape: tuple[int, ...], name: str
):
    """Expose logical token rows and an optional address-stable refresh copy.

    Serving allocators independently pad gate/beta storage.  Packed-varlen
    calls use B=1, so trimming the token extent is a zero-copy view and keeps
    the physical token pitch. A padded or independently batch-strided B>1
    carrier cannot be represented by the kernel's single flattened token
    coordinate, so preparation allocates one stable dense logical carrier and
    every launch refreshes it on the caller's stream before recurrence.
    """
    batch = int(tensor.shape[0])
    storage_tokens = int(tensor.shape[1])
    if tuple(tensor.shape[2:]) != trailing_shape:
        raise ValueError(
            f"{name} must have trailing shape {trailing_shape}, got {tuple(tensor.shape[2:])}"
        )
    if storage_tokens < logical_tokens_per_batch:
        raise ValueError(
            f"{name} token storage must cover at least {logical_tokens_per_batch} logical rows, got {storage_tokens}"
        )
    logical = tensor[:, :logical_tokens_per_batch]
    total_tokens = batch * logical_tokens_per_batch
    if batch == 1:
        return (logical[0], None)
    if storage_tokens == logical_tokens_per_batch and tensor.stride(
        0
    ) == storage_tokens * tensor.stride(1):
        return (
            logical.as_strided(
                (total_tokens, *trailing_shape),
                (logical.stride(1), *logical.stride()[2:]),
            ),
            None,
        )
    staging = tensor.new_empty((batch, logical_tokens_per_batch, *trailing_shape))
    staging.copy_(logical)
    return (staging.reshape(total_tokens, *trailing_shape), (staging, logical))


def _flatten_gate_rows(g, *, logical_tokens_per_batch: int, num_heads: int):
    """Collapse logical ``g`` rows while retaining their physical row pitch."""
    if g.stride(-1) != 1 or g.stride(-2) != HEAD_DIM:
        raise ValueError("g must be contiguous in its [H,128] row payload")
    return _flatten_token_storage(
        g,
        logical_tokens_per_batch=logical_tokens_per_batch,
        trailing_shape=(num_heads, HEAD_DIM),
        name="g",
    )


def _flatten_beta_rows(beta, *, logical_tokens_per_batch: int, num_heads: int):
    """Collapse logical beta rows, returning any required launch refresh copy."""
    if beta.stride(-1) != 1:
        raise ValueError("beta must have unit stride in its head dimension")
    beta_flat, refresh = _flatten_token_storage(
        beta,
        logical_tokens_per_batch=logical_tokens_per_batch,
        trailing_shape=(num_heads,),
        name="beta",
    )
    total_tokens = int(beta.shape[0]) * logical_tokens_per_batch
    if tuple(beta_flat.shape) != (total_tokens, num_heads):
        raise ValueError("beta must flatten to [total_tokens, num_heads]")
    return (beta_flat, refresh)


def _ffi_raw_pointer_carrier(tensor):
    """Return a contiguous zero-copy TensorView carrier for a raw pointer.

    Bindings that keep the contiguous-by-default host contract for a raw
    pointer argument (the TF32 ``raw_gate`` / ``beta_logits``, TMA-only q / k /
    v pointers, state pools) only need the tensor's dtype, device and data
    pointer.  A one-element view preserves the exact data pointer while
    satisfying that contract; the caller-owned strided tensor remains alive and
    is the device data source.  The BF16 fused bodies take ``g`` and ``beta``
    in place: their bindings accept any unit-innermost-stride view and the
    kernels address token rows through ``g_token_stride`` /
    ``beta_token_stride``.
    """
    if tensor.is_contiguous():
        return tensor
    carrier = tensor.as_strided((1,), (1,))
    if carrier.data_ptr() != tensor.data_ptr():
        raise RuntimeError("raw-pointer carrier must preserve the tensor data pointer")
    return carrier


def _ffi_arguments(arguments) -> tuple:
    """Positional FFI arguments with every torch tensor exported to tvm-ffi once.

    A tvm-ffi call converts each torch tensor argument through DLPack on every
    call and, on the first CUDA tensor, looks up torch's current stream for the
    call. A launch record holds the converted tensors instead: the same data
    pointer, shape, strides and dtype reach the binding, and the binding takes
    its stream from the enclosing :func:`_ffi_stream_context` (the caller's
    current torch stream, which is the capture stream under CUDA-graph capture).
    The tvm-ffi tensors keep their torch storage alive like the argument dicts.
    """
    import torch
    import tvm_ffi

    return tuple(
        tvm_ffi.from_dlpack(value) if isinstance(value, torch.Tensor) else value
        for value in arguments
    )


def _prepared_kernel_record(module, grid, bindings) -> tuple:
    """(native entry, positional arguments) of one ``PreparedKernel.launch``."""
    if not module._ready:
        raise RuntimeError("TMA descriptors must be initialized during preparation")
    return module._call, _ffi_arguments(module._arguments(grid, bindings))


def _run_launch_records(records) -> None:
    for call, arguments in records:
        call(*arguments)


def _settled_launch_records(launch) -> list:
    """Re-encode a part's moved descriptors on the current stream, then return its records."""
    if launch._descriptors_stale:
        launch._prepare_descriptors_in_stream()
    return launch._build_launch_records()


def _invalidate_launch_records(impl) -> None:
    """Drop the stored launch records of a launch and of its composite parts."""
    impl.__dict__.pop("_launch_records", None)
    for sub_name in AFFINE_SUB_LAUNCHES:
        sub = impl.__dict__.get(sub_name)
        if sub is not None:
            sub.__dict__.pop("_launch_records", None)


# Chain-route window-map bodies that may run ahead of the index gather and the
# main kernel.  Each body reads only its own tail inputs (q / k / g / beta
# tails, the zero value rows, the identity initial map, A_log, dt_bias, tail
# sequence offsets) and writes only its own map state and map output; its
# ``initial_state_f32`` argument (bound to the main kernel's final states) is
# never dereferenced, and every data access follows the body's grid-dependency
# wait.  The scan kernel waits on the main kernel at its first instruction and
# reads the map state that the index gather (a plain launch after the map)
# already ordered.  The generated programs the loader routes those two shipped
# modules to on sm_100a and sm_103a keep the same contract: their only work
# before the grid-dependency wait, which every thread runs, is barrier
# initialization and tensor-memory allocation.  The generated names come from
# the route table, so a new generated build of either module joins this set.
_MAP_FIRST_SHIPPED = (
    "cake_kda_bf16_f74a288bb51b335a1ccf",
    "cake_kda_bf16_5639519f5c942ed86e4f",
)
_MAP_FIRST_MODULES = frozenset(_MAP_FIRST_SHIPPED) | frozenset(
    routes[module]
    for routes in _RENDERED_ROUTES.values()
    for module in _MAP_FIRST_SHIPPED
    if module in routes
)


# Chain-route correction bodies that merge themselves into the composite.
# Each window reduce-adds its output into the caller's output tail,
# accumulates its checkpoint rows onto the rows the main windows wrote in
# place (row zero included), and a window that ends a sequence stores main
# final + correction final (one IEEE FP32 addition, as the epilogue did) into
# that sequence's pool slot, so no epilogue kernel follows the chain.  They
# are the loader's ``_RENDERED_MERGE_ROUTES`` builds of the correction
# modules (read at launch preparation); every other launch of those modules
# keeps the routed program.
def _correction_merge_routed(device) -> bool:
    """The loader names a merging correction build for ``device``."""
    import torch

    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(
        torch.cuda.get_device_capability(device)
    )
    return bool(_kda_registry._RENDERED_MERGE_ROUTES.get(arch))


class _CorrectionMergeUnavailable(Exception):
    """The chain's correction module has no merging build on this device."""


def _is_fused_m128_row(factory) -> bool:
    """``factory`` is the plain fused M128 factory row.

    That is ``partial(_factory, "compiled_bf16_fused_m128", ...)``, or the same
    row through the work-bin turnover factory (``_work_bin_turnover_factory``,
    which resolves the row like ``_factory``).  The fold and merge interceptors
    rewrite only this factory row, as they would ``_factory``'s; they raise
    their ``...Unavailable`` exception for every other one (fail closed).
    """
    return getattr(factory, "func", None) in (
        _factory,
        _work_bin_turnover_factory,
    ) and factory.args == ("compiled_bf16_fused_m128",)


def _built_from_table(launch, table: str) -> bool:
    """``launch``'s body is a program the loader table ``table`` names on this device.

    A fold or merge route drops another launch (the map pass, the pair-map
    producer, the epilogue) on the strength of its body; the composite checks
    the body it actually got before it does.
    """
    programs = getattr(_kda_registry, table, {}).get(detect_gpu_arch(), {})
    return getattr(getattr(launch, "module", None), "name", None) in set(
        programs.values()
    )


def _correction_merge_factory(
    name,
    *args,
    route_heads=None,
    route_work_bins=False,
    owner_helper=False,
    owner_helper_pool=False,
    **kwargs,
):
    """The merging build of the factory row's correction module.

    ``_factory`` resolves the row to its shipped module and that module
    through the routes; this takes the module's ``_RENDERED_MERGE_ROUTES``
    build instead (same argument plan, the merge on top).  The routing
    keywords are not part of the row's key, and a merging build has no
    work bins and no owner/helper build.
    """
    del route_heads, route_work_bins, owner_helper, owner_helper_pool
    arch = detect_gpu_arch()
    rows = _kda_registry.FACTORIES.get(arch, {}).get(name, {})
    module = rows.get(_kda_registry._key(args, kwargs))
    program = _kda_registry._RENDERED_MERGE_ROUTES.get(arch, {}).get(module)
    if program is None:
        raise _CorrectionMergeUnavailable
    return _kda_registry.PreparedKernel(program, arch)


# A minimal kernel the driver JIT-compiles for the cluster capacity probe: the
# capacity depends on the launch shape (block, shared memory, cluster), not on
# the body.
_PAIR_PROBE_PTX = b"""
.version 7.8
.target sm_90
.address_size 64
.visible .entry cake_kda_pair_probe()
{
    ret;
}
\x00"""


_M64_PAIR_CLUSTER_CAPACITY: dict = {}
_M64_PAIR_PROBE_WARNED = False


def _m64_pair_cluster_capacity(device: torch.device) -> int:
    """How many 2-CTA clusters of the M64 value split's CTA the device holds at once.

    ``cuOccupancyMaxActiveClusters`` for a probe kernel launched with the M64
    CTA's threads and dynamic shared memory (one CTA per SM) in clusters of
    two.  The driver's answer follows the part's floorsweeping, which can leave
    it below ``sm_count // 2``.  Only an answer is cached (per device): when
    the driver cannot answer, the call returns 0 (the runtime keeps the routed
    program), logs one warning per process and probes again on the next call.
    """
    capacity = _M64_PAIR_CLUSTER_CAPACITY.get(device)
    if capacity is None:
        capacity = _probe_m64_pair_cluster_capacity(device)
        if capacity is None:
            return 0
        _M64_PAIR_CLUSTER_CAPACITY[device] = capacity
    return capacity


def _probe_m64_pair_cluster_capacity(device: torch.device) -> Optional[int]:
    """The driver's answer for ``_m64_pair_cluster_capacity``; ``None`` on failure."""
    global _M64_PAIR_PROBE_WARNED
    import torch

    try:
        from cuda.bindings import driver

        def check(result):
            err, *values = result if isinstance(result, tuple) else (result,)
            if err != driver.CUresult.CUDA_SUCCESS:
                raise RuntimeError(f"CUDA driver error {err}")
            return values[0] if values else None

        with torch.cuda.device(device):
            torch.empty(0, device=device)  # the primary context is current
            module = check(driver.cuModuleLoadData(_PAIR_PROBE_PTX))
            try:
                function = check(
                    driver.cuModuleGetFunction(module, b"cake_kda_pair_probe")
                )
                check(
                    driver.cuFuncSetAttribute(
                        function,
                        driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                        M64_CTA_SMEM_BYTES,
                    )
                )
                config = driver.CUlaunchConfig()
                config.gridDimX, config.gridDimY, config.gridDimZ = 2, 1, 1
                config.blockDimX, config.blockDimY, config.blockDimZ = (
                    M64_CTA_THREADS,
                    1,
                    1,
                )
                config.sharedMemBytes = M64_CTA_SMEM_BYTES
                cluster = driver.CUlaunchAttribute()
                cluster.id = (
                    driver.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
                )
                cluster.value.clusterDim.x = 2
                cluster.value.clusterDim.y = 1
                cluster.value.clusterDim.z = 1
                config.attrs = [cluster]
                config.numAttrs = 1
                return int(check(driver.cuOccupancyMaxActiveClusters(function, config)))
            finally:
                driver.cuModuleUnload(module)
    except Exception as error:
        if not _M64_PAIR_PROBE_WARNED:
            _M64_PAIR_PROBE_WARNED = True
            from flashinfer.jit.core import logger

            logger.warning(
                "KDA prefill: the M64 pair-share cluster capacity probe failed "
                "(%s: %s); M64 launches keep the routed program until a probe "
                "succeeds",
                type(error).__name__,
                error,
            )
        return None


# Long M64 value-split launches whose clusters fit one wave take the loader's
# ``_RENDERED_PAIR_SHARE_ROUTES`` build of the module: the two value halves of
# a task run as a 2-CTA cluster and each prepares every second 32-token chunk
# for both (the same bits as the routed program).  Every other launch, and a
# module without a pair-share build, keeps the routed program.
def _m64_pair_share_factory(name, *args, **kwargs):
    """The pair-share build of the factory row's M64 module, else ``_factory``."""
    arch = detect_gpu_arch()
    rows = _kda_registry.FACTORIES.get(arch, {}).get(name, {})
    module = rows.get(_kda_registry._key(args, kwargs))
    program = _kda_registry._RENDERED_PAIR_SHARE_ROUTES.get(arch, {}).get(module)
    if program is None:
        return _factory(name, *args, **kwargs)
    return _kda_registry.PreparedKernel(program, arch)


def _work_bin_turnover_program(
    name,
    *args,
    route_heads=None,
    route_work_bins=False,
    owner_helper=False,
    owner_helper_pool=False,
    **kwargs,
):
    """The turnover build of the factory row's work-binned module, or ``None``.

    ``None`` for a call without work bins or with owner/helper CTAs (pooled or
    not), when the row has no module on this device, when the head-count routes
    take the module elsewhere at ``route_heads``, when the loader names no
    work-binned build of the module (``_factory`` would not take work bins), or
    when it names no turnover build of it (``_RENDERED_WORK_BIN_TURN_ROUTES``).
    The routing keywords are not part of the row's key (as in ``_factory``).
    """
    if not route_work_bins or owner_helper or owner_helper_pool:
        return None
    arch = detect_gpu_arch()
    rows = _kda_registry.FACTORIES.get(arch, {}).get(name, {})
    module = rows.get(_kda_registry._key(args, kwargs))
    if module is None:
        return None
    heads = getattr(_kda_registry, "_RENDERED_HEAD_ROUTES", {})
    if route_heads in heads.get(arch, {}).get(module, {}):
        return None
    if module not in _kda_registry._RENDERED_WORK_BIN_ROUTES.get(arch, {}):
        return None
    return _kda_registry._RENDERED_WORK_BIN_TURN_ROUTES.get(arch, {}).get(module)


def _work_bin_turnover_factory(name, *args, **kwargs):
    """The turnover build of the factory row's work-binned module.

    The same bins, bits and argument plan as the module's work-binned program,
    with each bin's state turned over in the background.  ``_build_body``
    takes this factory only where ``_work_bin_turnover_program`` names a
    build; any other call resolves through ``_factory``.
    """
    program = _work_bin_turnover_program(name, *args, **kwargs)
    if program is None:
        return _factory(name, *args, **kwargs)
    return _kda_registry.PreparedKernel(program, detect_gpu_arch())


def _work_bin_turnover_route(
    *,
    route_work_bins: bool,
    direct_n16: bool,
    n32_ft_slab: bool,
    max_seq_len: int,
    gpu_arch: str,
    sm_count: int,
) -> bool:
    """Whether a direct M128 launch may take its module's turnover build.

    Only a work-binned N32 serving launch whose longest sequence has at most
    the part's turnover chunk count.  Affine split parts (``_n32_ft_slab``)
    never do: their slab modules have no turnover build, and the composite's
    Fold / Merge classes intercept the factory for their own builds.
    """
    if not route_work_bins or direct_n16 or n32_ft_slab:
        return False
    limit = (
        BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS_GB300
        if gpu_arch == "sm_103a" and sm_count == 152
        else BF16_WORK_BIN_TURNOVER_MAX_TASK_CHUNKS
    )
    return max_seq_len <= BF16_M128_CHUNK * limit


# Bodies that reach an operand only through its TMA descriptor never
# dereference the raw pointer argument of the same name.  The generated FFI
# shim still applies the contiguous-by-default host contract to that pointer,
# so a strided packed-row q / k / v view travels through the one-element
# carrier above while the TMA view carries the token pitch.  Every other body
# reads the pitch from the TensorView itself (a hidden int64 argument expanded
# by the shim) and needs the strided view.  Mirrors the source runtime's table
# of the same name.
QKV_TMA_ONLY_RAW_POINTERS = {
    "compiled_tf32_bt16_prepare": ("q", "k"),
    "compiled_tf32_bt16_chain_m64_fp32_state": ("v",),
    "compiled_tf32_fused_n32": ("q", "k", "v"),
}


def _qkv_raw_pointer_names(factory):
    """Raw q / k / v arguments the body never dereferences (TMA-only reads)."""
    name = getattr(factory, "__name__", None)
    if name is None:
        # functools.partial(_factory, "<name>") wraps the generated module.
        name = getattr(factory, "args", ("",))[0]
    return QKV_TMA_ONLY_RAW_POINTERS.get(name, ())


def _qkv_raw_pointer_arg(tensor, name, carrier_names):
    """Bind a raw q / k / v pointer: the strided view, or its carrier."""
    if name in carrier_names:
        return _ffi_raw_pointer_carrier(tensor)
    return tensor


def _dvsplit_carrier_precision_ok(
    *,
    compute_dtype: str,
    bounded_gate: bool,
    max_seq_len: int,
    state_dtype_is_fp32: bool,
) -> bool:
    """Return whether the M64 value split may carry this sequence's state.

    The FP32-pool split carries FP32 chunk state (delta decay onto the FP32
    TMEM state; CAKE-736 round 9) and is admissible at every length.  The
    BF16-pool route keeps the round-8 guard: a bounded gate beyond
    ``BF16_DVSPLIT_STATE_CARRIER_MAX_SEQ_LEN`` tokens needs the FP32 chunk
    carrier of the direct M128 family instead (see the constant's note).
    Unbounded gates never reach the split, and TF32 keeps its own BT16 policy.
    """
    return state_dtype_is_fp32 or not (
        compute_dtype == "bf16"
        and bounded_gate
        and max_seq_len > BF16_DVSPLIT_STATE_CARRIER_MAX_SEQ_LEN
    )


def _should_use_bf16_one_wave_dvsplit(
    *,
    gpu_arch: str,
    sm_count: int,
    total_tasks: int,
    checkpoint_every_n_tokens: int,
    bounded_gate: bool,
    compute_dtype: str,
    force_direct_m128: bool = False,
    force_direct_m128_n32: bool = False,
    beta_tma_refresh: bool = False,
    num_heads: int = 0,
    max_seq_len: int = 0,
) -> bool:
    """Replace a one-wave BF16 direct M128 grid with the M64 value split.

    CUPTI grid sweeps on a 148-SM B200 and a 152-SM GB300 (H1/H6/H12/H24, 8
    through 4096 tokens, uniform and mixed packed sequences, CP64 and NoCP,
    BF16 logit and active FP32 beta) measured the two-CTA M64 value split
    faster than every direct M128 tile (N16, N32, page64 N32x2) in every case
    where both value CTAs of each task stay resident in one wave.  Beyond one
    wave the direct tiles keep their measured preference.  The route requires
    the bounded gate and a 32-token-aligned checkpoint interval; explicit
    direct requests keep their physical tile.
    """
    return (
        compute_dtype == "bf16"
        and gpu_arch in BF16_ONE_WAVE_DVSPLIT_ARCHES
        and bounded_gate
        and (checkpoint_every_n_tokens % BF16_M128_CHUNK == 0)
        and (0 < INDEPENDENT_DVSPLIT_CTAS * total_tasks <= sm_count)
        and (not force_direct_m128)
        and (not force_direct_m128_n32)
        # BF16 logit beta whose token pitch is not TMA-encodable is refreshed
        # into a padded carrier on every launch.  Only the short H12 direct
        # tiles avoid that carrier (scalar beta loads), so only those grids
        # keep their direct tile; every other direct tile pays the same copy.
        and not (
            beta_tma_refresh
            and num_heads == 12
            and max_seq_len <= H12_DIRECT_N32_MAX_SEQ_LEN
        )
    )


def _should_use_h12_active_beta_m64(
    *,
    gpu_arch: str,
    sm_count: int,
    num_heads: int,
    max_seq_len: int,
    total_tasks: int,
    active_beta_f32: bool,
    state_dtype_is_fp32: bool,
    indexed_state_pool: bool,
    in_place_state_pool: bool,
    checkpoint_every_n_tokens: int,
    gate_kind: KDAGateKind,
    force_direct_m128: bool = False,
    force_direct_m128_n32: bool = False,
) -> bool:
    """Select the measured one-wave GB300 active-beta H12 value split."""
    return (
        gpu_arch == "sm_103a"
        and sm_count == 152
        and (num_heads == 12)
        and (129 <= max_seq_len <= 256)
        and (INDEPENDENT_DVSPLIT_CTAS * total_tasks <= sm_count)
        and active_beta_f32
        and state_dtype_is_fp32
        and indexed_state_pool
        and in_place_state_pool
        and (checkpoint_every_n_tokens == 64)
        and (gate_kind == KDAGateKind.LOWER_BOUND)
        and (not force_direct_m128)
        and (not force_direct_m128_n32)
    )


def _bt16_chunks_per_prep_cta(
    *, num_heads: int, total_chunks: int, compute_dtype: str = "tf32"
) -> int:
    """Select the chunk-parallel prepare walk without changing kernel identity.

    H12 uses CPC4 once it has enough chunks to amortize each CTA's preamble;
    below that point a multi-chunk walk underfills the prepare grid, so CPC1
    remains faster through 128 total chunks.  The broader variable-head
    portfolio uses CPC2 below 16K chunk-heads to feed the four-resident-CTA
    prepare schedule without long partial waves. CPC8 above that point
    amortizes the preamble over a large grid.
    """
    if num_heads == 12:
        if total_chunks <= BT16_H12_CPC1_MAX_TOTAL_CHUNKS:
            return 1
        return BT16_H12_CHUNKS_PER_PREP_CTA
    if num_heads * total_chunks >= BT16_GENERAL_HIGH_WORK_MIN_CHUNK_HEADS:
        return BT16_GENERAL_HIGH_WORK_CHUNKS_PER_PREP_CTA
    return 6 if compute_dtype == "bf16" else BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA


# The owner/helper split of the unbounded direct N32 body: helper CTAs prepare a
# task's chunks (gate scan, normalization, pairwise products, block inverse) and
# hand each prepared stage to the task's owner CTA through a global ring, so the
# owner runs the recurrence alone. Every owner and helper CTA is resident at once
# (one CTA per SM, launched cooperatively): a task takes min(2, SMs / tasks - 1)
# helpers, and a sequence shorter than OWNER_HELPER_MIN_TOKENS keeps the one-CTA
# body. Each helper has OWNER_HELPER_SLOTS_PER_HELPER ring slots of 32 KiB per task.
OWNER_HELPER_MAX_HELPERS = 2
OWNER_HELPER_MIN_TOKENS = 1024
OWNER_HELPER_SLOTS_PER_HELPER = 5
OWNER_HELPER_SLOT_ELEMENTS = 16384


def _owner_helper_count(
    *, gpu_arch: str, sm_count: int, total_tasks: int, max_seq_len: int
) -> int:
    """Helper CTAs per task of the owner/helper split, or 0 for the one-CTA body."""
    if gpu_arch not in ("sm_100a", "sm_103a") or max_seq_len < OWNER_HELPER_MIN_TOKENS:
        return 0
    helpers = min(OWNER_HELPER_MAX_HELPERS, sm_count // max(total_tasks, 1) - 1)
    return helpers if helpers >= 1 else 0


# The pooled owner/helper split: the helper CTAs of a head form one pool of
# ``pool`` CTAs that prepares the head's chunks of every sequence in chunk-major
# order (item ``chunk * sequences + sequence`` goes to pool member ``item % pool``),
# so the SMs the per-task split leaves idle help as well. The grid is
# ``tasks + heads * pool``; the pool stays below ``(helpers + 1) * sequences`` so
# the program reads the same ``helpers`` (and ring depth) from its grid as the
# per-task split, and every wait of a member is on an item of a smaller index, so
# the resident grid cannot deadlock. The chunk-major order needs one chunk count
# for every sequence. The pool is taken where the per-task split has one helper
# per task and the pool gives each head at least 13/12 as many helpers: below
# that ratio, and wherever the split already has two helpers per task, the
# shared ring costs more than the extra helpers return.
OWNER_HELPER_POOL_MIN_RATIO = (13, 12)


def _owner_helper_pool_count(
    *, sm_count: int, num_seqs: int, num_heads: int, helpers: int, uniform_chunks: bool
) -> int:
    """Pooled helper CTAs per head, or 0 to keep ``tasks * (1 + helpers)`` CTAs.

    Taken where the per-task split has one helper per task, every sequence has
    the same chunk count and the pool gives a head at least
    ``OWNER_HELPER_POOL_MIN_RATIO`` times the split's helpers
    (``12 * pool >= 13 * sequences``).
    """
    if helpers != 1 or not uniform_chunks or num_seqs < 1 or num_heads < 1:
        return 0
    total_tasks = num_seqs * num_heads
    pool = min((sm_count - total_tasks) // num_heads, (helpers + 1) * num_seqs - 1)
    num, den = OWNER_HELPER_POOL_MIN_RATIO
    return pool if den * pool >= num * helpers * num_seqs else 0


def _should_use_small_bh_owner_helper(
    *,
    gpu_arch: str,
    sm_count: int,
    num_seqs: int,
    num_heads: int,
    max_seq_len: int,
    compute_dtype: str = "bf16",
    unbounded_softplus: bool = False,
) -> bool:
    """Select the small-BH SM100/SM103 region whose eight-CTA groups fully reside.

    The physical owner/helper schedule already resolves each task through
    ``seq_order`` and ``cu_seqlens``.  Fixed and packed layouts therefore share
    one variable-shape kernel identity; only the resolved task count, maximum
    sequence length, and residency bound select this schedule family.
    """
    total_tasks = num_seqs * num_heads
    bounded_min_chunks = 16 if 2 * SMALL_BH_GROUP_SIZE * total_tasks <= sm_count else 24
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and total_tasks > 0
        and (
            compute_dtype == "tf32"
            or (total_tasks <= SMALL_BH_MAX_TASKS and num_heads <= SMALL_BH_MAX_TASKS)
        )
        and (
            (max_seq_len + 31) // 32
            >= (8 if unbounded_softplus else bounded_min_chunks)
            if compute_dtype == "tf32"
            else max_seq_len >= SMALL_BH_MIN_SEQ_LEN
        )
        and (SMALL_BH_GROUP_SIZE * total_tasks <= sm_count)
    )


CHUNK = 32  # apply/pair-map kernel chunk (tokens)


def _affine_apply_runs(
    window_heads: int, sm_count: int, max_blocks: int, pairmap: bool
) -> int:
    """Block runs per window-head of the apply / pair-map grid.

    Both kernels take the whole shared-memory carve-out (one CTA per SM), so a
    grid of ``runs * window_heads`` items runs in ``ceil(items / sm_count)``
    waves of at most ``ceil(max_blocks / runs)`` blocks each.  The base split,
    ``ceil(sm_count / window_heads)`` runs, occupies every SM at least once.
    One alternative replaces it only when its waves x blocks cost is strictly
    lower: two runs when the window-heads exceed the SMs (the base single run
    leaves a sub-wave tail), and for the pair map a single run instead of the
    base two (one wave; an odd block count no longer rounds up per run).  The
    apply keeps its base split below one wave: its runs also stream the
    checkpoint rows, and fewer concurrent CTAs measured slower there.  Every
    block's arithmetic is independent of the split.
    """

    base = max(1, -(-sm_count // window_heads))
    if window_heads > sm_count:
        alternative = 2
    elif pairmap and base == 2:
        alternative = 1
    else:
        return base

    def cost(runs: int) -> int:
        waves = -(-(runs * window_heads) // sm_count)
        return waves * -(-max_blocks // min(runs, max_blocks))

    return alternative if cost(alternative) < cost(base) else base


def build_affine_apply_items(
    *,
    tail_lengths: list[int],
    cu_chunk_offsets: list[int],
    snap_starts: list[int],
    tail_offsets: list[int],
    num_heads: int,
    rows_enabled: bool,
    sm_count: int = 148,
    pairmap: bool = False,
    owns_final: bool = True,
) -> list[list[int]]:
    """Work items for the apply kernel, one per (window, head, block run).

    ``tail_lengths[w]`` is the token count of tail window ``w`` (composite part
    ``w + 1``), ``cu_chunk_offsets`` the main pass's per-part chunk prefix,
    ``snap_starts[w]`` the first snapshot row of the window,
    ``tail_offsets[w]`` its first tail token.  Each window-head is cut into
    ``_affine_apply_runs`` equal block runs (at least enough to occupy every
    SM once); the run owning the window's last block also evaluates
    ``S_w^in · M_final``.
    """

    items: list[list[int]] = []
    window_heads = max(1, len(tail_lengths) * num_heads)
    max_blocks = max(
        [((length + CHUNK - 1) // CHUNK + 1) // 2 for length in tail_lengths],
        default=1,
    )
    runs_per_window = _affine_apply_runs(
        window_heads, sm_count, max(1, max_blocks), pairmap
    )
    for w, length in enumerate(tail_lengths):
        chunks = (length + CHUNK - 1) // CHUNK
        blocks = (chunks + 1) // 2
        runs = min(runs_per_window, blocks)
        per_run = -(-blocks // runs)
        for head in range(num_heads):
            for p_begin in range(0, blocks, per_run):
                p_end = min(blocks, p_begin + per_run)
                owns_final_run = owns_final and p_end == blocks and not pairmap
                items.append(
                    [
                        w * num_heads + head,
                        snap_starts[w] * num_heads + head,
                        cu_chunk_offsets[w + 1] * num_heads + head,
                        p_begin,
                        p_end,
                        chunks,
                        tail_offsets[w],
                        w + 1,
                        (w * num_heads + head) if owns_final_run else -1,
                        head,
                        length,
                        1 if (rows_enabled and not pairmap) else 0,
                    ]
                )
    return items


def build_map_prefix_items(
    *, tail_lengths: list[int], snap_starts: list[int], num_heads: int
) -> list[list[int]]:
    """One item per (tail window, head): snapshot row of block 0, block count, window-map row."""

    items: list[list[int]] = []
    for w, length in enumerate(tail_lengths):
        blocks = (length + 63) // 64
        for head in range(num_heads):
            items.append(
                [snap_starts[w] * num_heads + head, blocks, w * num_heads + head, 0]
            )
    return items


@cache
def _device_sm_count(device: torch.device) -> int:
    """Resolve and cache the launch device's physical SM count."""
    import torch

    return int(torch.cuda.get_device_properties(device).multi_processor_count)


def _fused_affine_epilogue_enabled() -> bool:
    """Return whether the affine composite fuses its torch epilogue.

    ``FLASHINFER_KDA_AFFINE_FUSED_EPILOGUE=0`` restores the torch glue
    (checkpoint merge + index_copy_, tail add, final-state select/add and pool
    scatter); both paths are bitwise identical, the fused one saves the host
    time of ~10 torch launches per call.
    """
    import os

    return os.environ.get("FLASHINFER_KDA_AFFINE_FUSED_EPILOGUE", "1") != "0"


@dataclass(frozen=True)
class _FusedAffineEpilogue:
    """Static arguments of the fused affine epilogue launch."""

    run: Any
    index_prep: Any
    heads: int
    tail_elems: int
    pool_slot_stride: int
    first_rows: int
    num_rows: int
    # The index preparation of a composite whose correction windows merge themselves (no ``run`` launch).
    index_prep_merge: Any = None

    @staticmethod
    def build(impl) -> "_FusedAffineEpilogue | None":
        import torch

        from flashinfer.jit.cake_kda_affine_epilogue import load_for_device

        out_tail = impl._out_tail
        pool = impl._final_pool
        heads = int(impl._main_final.shape[1])
        row_elems = heads * HEAD_DIM * HEAD_DIM
        if not out_tail.is_contiguous() or out_tail.numel() % 8:
            return None
        if pool.ndim != 4 or pool.stride(0) % 8 or not pool[0].is_contiguous():
            return None
        if pool.dtype not in (torch.float32, torch.bfloat16):
            return None
        if pool.shape[1:] != (heads, HEAD_DIM, HEAD_DIM):
            return None
        first_rows = num_rows = 0
        if impl._checkpoint_output is not None and not impl._checkpoint_in_place:
            num_rows = int(impl._checkpoint_main.shape[0])
            first_rows = num_rows - int(impl._checkpoint_correction.shape[0])
            if impl._checkpoint_output.stride(0) != row_elems:
                return None
        try:
            module = load_for_device(impl._launch_device)
        except NotImplementedError:
            return None
        return _FusedAffineEpilogue(
            run=module.run,
            index_prep=module.index_prep,
            heads=heads,
            tail_elems=int(out_tail.numel()),
            pool_slot_stride=int(pool.stride(0)),
            first_rows=first_rows,
            num_rows=num_rows,
            index_prep_merge=getattr(module, "index_prep_merge", None),
        )


def _upload_int32_batch(device, host_lists: dict[str, list[int]]):
    """Upload several host int32 lists with one pinned, stream-ordered copy."""
    import torch

    return _upload_int_batch(device, host_lists, torch.int32)


def _upload_int_batch(device, host_lists: dict[str, list[int]], dtype):
    """Upload several host int32/int64 lists with one pinned, stream-ordered copy.

    Preparation used to issue one pageable ``torch.tensor(..., device=cuda)``
    per metadata list; each pageable copy stages through the driver and
    synchronizes the host.  Packing every list into one pinned buffer and
    issuing one non-blocking copy keeps preparation asynchronous and
    capturable.  The pinned source stays alive through PyTorch's caching host
    allocator until the copy completes.
    """
    from array import array

    import torch

    names = list(host_lists)
    lengths = [len(host_lists[name]) for name in names]
    total = sum(lengths)
    if total == 0:
        empty = torch.empty(0, dtype=dtype, device=device)
        return {name: empty for name in names}
    flat: list[int] = []
    for name in names:
        flat.extend(host_lists[name])
    typecode = {torch.int32: "i", torch.int64: "q"}[dtype]
    host = torch.frombuffer(array(typecode, flat), dtype=dtype).pin_memory()
    device_flat = host.to(device, non_blocking=True)
    # The pinned source is kept alive by the returned dict so that a copy
    # captured into a CUDA graph replays from stable host memory.
    uploads = {"_host_pinned": host}
    offset = 0
    for name, length in zip(names, lengths, strict=False):
        uploads[name] = device_flat[offset : offset + length]
        offset += length
    return uploads


def _should_use_uniform_piece_persistent(
    *,
    gpu_arch: str,
    sm_count: int,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
    max_seq_len: int,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> bool:
    """Select recurrence pieces when their occupancy-aware roofline wins.

    ``flashinfer.kda_prefill._should_use_uniform_piece_persistent`` adds the
    148 / 152-SM gate of the measured policy; the TF32 auto route keeps this
    SM-count-free rule (its persistent family is not exported).
    """
    if not uniform_sequences or max_seq_len <= 0:
        return False
    estimate = _persistent_m128_roofline(
        compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
        sm_count=sm_count,
        num_sequences=num_seqs,
        num_heads=num_heads,
        sequence_length=max_seq_len,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )
    return estimate is not None and estimate.piece_ns < estimate.direct_ns


def _direct_sequence_makespan(
    order: tuple[int, ...],
    task_costs: tuple[int, ...],
    *,
    num_heads: int,
    worker_count: int,
) -> int:
    """Model direct-CTA list scheduling for one sequence-group order."""
    if num_heads <= 0 or worker_count <= 0:
        raise ValueError(
            "direct sequence scheduling requires positive heads and workers"
        )
    if len(order) != len(task_costs) or set(order) != set(range(len(task_costs))):
        raise ValueError(
            "direct sequence order must be a permutation of every sequence"
        )
    if not task_costs or min(task_costs) <= 0:
        raise ValueError("direct sequence scheduling requires positive task costs")
    worker_load_counts = {0: worker_count}
    worker_loads = [0]

    def add_workers(load: int, count: int) -> None:
        if load in worker_load_counts:
            worker_load_counts[load] += count
        else:
            worker_load_counts[load] = count
            heapq.heappush(worker_loads, load)

    for seq_idx in order:
        task_cost = task_costs[seq_idx]
        remaining_tasks = num_heads
        while remaining_tasks:
            ready_at = heapq.heappop(worker_loads)
            ready_workers = worker_load_counts.pop(ready_at)
            scheduled_tasks = min(ready_workers, remaining_tasks)
            if ready_workers > scheduled_tasks:
                add_workers(ready_at, ready_workers - scheduled_tasks)
            add_workers(ready_at + task_cost, scheduled_tasks)
            remaining_tasks -= scheduled_tasks
    return max(worker_load_counts)


@cache
def _make_direct_sequence_order(
    seq_lens: tuple[int, ...], *, num_heads: int, worker_count: int, chunk_tokens: int
) -> tuple[int, ...]:
    """Order runtime sequence groups to reduce the direct grid's modeled tail.

    The kernel maps a contiguous ``num_heads``-CTA group to each sequence.
    Pure length order is LPT at the task level, but not necessarily after that
    grouping meets a finite SM count.  Each CTA's cost includes six chunk units
    for the fixed state prologue and final-state/output drain, calibrated from
    the contract's two-, four-, and 256-chunk H96 rows.  Starting from stable
    descending order, bounded best-improvement passes retain the same
    variable-shape kernel and ABI while accepting only a strict modeled
    makespan reduction.  Batches of at most 16 sequences test one pass of
    single-sequence insertion moves.  Larger batches retain LPT order: their
    many-wave tail has lower leverage, while a quadratic host search would be
    inappropriate on the launch-preparation path.
    """
    if num_heads <= 0 or worker_count <= 0 or chunk_tokens <= 0:
        raise ValueError("direct sequence scheduling inputs must be positive")
    if not seq_lens or min(seq_lens) <= 0:
        raise ValueError(
            "direct sequence scheduling requires non-empty positive lengths"
        )
    order = tuple(
        sorted(range(len(seq_lens)), key=lambda index: (-seq_lens[index], index))
    )
    if (
        len(order) < 2
        or len(order) > BF16_DIRECT_INSERTION_MAX_SEQUENCES
        or len(order) * num_heads <= worker_count
        or (len(set(seq_lens)) == 1)
    ):
        return order
    task_costs = tuple(
        (
            (length + chunk_tokens - 1) // chunk_tokens + BF16_DIRECT_FIXED_CHUNK_COST
            for length in seq_lens
        )
    )
    current_score = _direct_sequence_makespan(
        order, task_costs, num_heads=num_heads, worker_count=worker_count
    )
    for _ in range(BF16_DIRECT_ORDER_MAX_PASSES):
        best_order = order
        best_score = current_score
        moves = (
            (source, destination)
            for source in range(len(order))
            for destination in range(len(order))
            if source != destination
        )
        for source, destination in moves:
            candidate = list(order)
            moved_sequence = candidate.pop(source)
            candidate.insert(destination, moved_sequence)
            candidate_order = tuple(candidate)
            candidate_score = _direct_sequence_makespan(
                candidate_order,
                task_costs,
                num_heads=num_heads,
                worker_count=worker_count,
            )
            candidate_lengths = tuple((seq_lens[index] for index in candidate_order))
            best_lengths = tuple((seq_lens[index] for index in best_order))
            if candidate_score < best_score or (
                candidate_score == best_score
                and candidate_score < current_score
                and (candidate_lengths > best_lengths)
            ):
                best_order = candidate_order
                best_score = candidate_score
        if best_score >= current_score:
            break
        order = best_order
        current_score = best_score
    return order


@cache
def _make_direct_work_bins(
    seq_lens: tuple[int, ...], *, num_heads: int, worker_count: int, chunk_tokens: int
) -> tuple[tuple[int, ...], int]:
    """Bin table of a direct M128 body built with work bins.

    Each CTA of such a body runs one bin of (sequence, head) tasks back to
    back.  Tasks go in descending chunk count (sequences in descending length,
    stable in index; heads ascending) to the least-loaded bin, lowest bin
    first; the bin count is ``min(tasks, worker_count)``.  The table holds the
    bin offsets as absolute table indices, then the task ids
    ``sequence * num_heads + head``.  Returns the table and the bin count.
    """
    if num_heads <= 0 or worker_count <= 0 or chunk_tokens <= 0:
        raise ValueError("direct work-bin inputs must be positive")
    if not seq_lens or min(seq_lens) <= 0:
        raise ValueError("direct work bins require non-empty positive lengths")
    num_bins = min(len(seq_lens) * num_heads, worker_count)
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    loads = [(0, bin_idx) for bin_idx in range(num_bins)]
    order = sorted(range(len(seq_lens)), key=lambda index: (-seq_lens[index], index))
    for seq_idx in order:
        chunks = (seq_lens[seq_idx] + chunk_tokens - 1) // chunk_tokens
        for head_idx in range(num_heads):
            load, bin_idx = heapq.heappop(loads)
            bins[bin_idx].append(seq_idx * num_heads + head_idx)
            heapq.heappush(loads, (load + chunks, bin_idx))
    table = [num_bins + 1]
    for tasks in bins:
        table.append(table[-1] + len(tasks))
    for tasks in bins:
        table.extend(tasks)
    return tuple(table), num_bins


def _use_direct_work_bins(
    seq_lens: tuple[int, ...], *, num_heads: int, sm_count: int, chunk_tokens: int
) -> bool:
    """Whether a direct M128 launch asks for its program built with work bins.

    The one-task-per-CTA launch already starts the longest (sequence, head)
    tasks first, so bins pay only where an SM runs tasks back to back on the
    critical path: with more tasks than SMs and a bin assignment
    (``_make_direct_work_bins``) whose largest bin, in chunks, exceeds the
    longest task.  Where the longest task alone sets the makespan, with at
    most one task per SM, or with an empty sequence, the launch keeps the
    one-task-per-CTA program.
    """
    if not seq_lens or min(seq_lens) <= 0:
        return False
    if len(seq_lens) * num_heads <= sm_count:
        return False
    table, num_bins = _make_direct_work_bins(
        seq_lens, num_heads=num_heads, worker_count=sm_count, chunk_tokens=chunk_tokens
    )
    chunks = [(length + chunk_tokens - 1) // chunk_tokens for length in seq_lens]
    largest_bin = max(
        sum(
            chunks[task // num_heads] for task in table[table[index] : table[index + 1]]
        )
        for index in range(num_bins)
    )
    return largest_bin > max(chunks)


def _constrained_direct_m128_route(
    *,
    gpu_arch: str,
    num_heads: int,
    max_seq_len: int,
    force_direct_m128: bool,
    force_direct_m128_n32: bool,
    unbounded_softplus: bool,
    state_dtype_is_fp32: bool,
    indexed_state_pool: bool,
    checkpoint_every_n_tokens: int,
    preferred_route: str,
) -> str:
    """Resolve direct-only serving requests without bypassing H12 N32 policy.

    Indexed state, checkpoint output, and independently strided beta all force
    the direct kernel family, but they do not require its N16 physical tile.
    Retain N16 for the qualified 64-token row; longer short residuals and
    dense grids retain the unconstrained N32 preference. Checkpoint and beta
    storage constraints select a legal family; they must not reset its
    physical tile preference to N16.
    """
    if force_direct_m128_n32 or unbounded_softplus:
        return BF16_ROUTE_DIRECT_M128
    if (
        (not force_direct_m128)
        and state_dtype_is_fp32
        and indexed_state_pool
        and (checkpoint_every_n_tokens in (0, 64))
        and (max_seq_len > 64)
        and (checkpoint_every_n_tokens == 64 or max_seq_len >= 4 * BF16_M128_CHUNK)
        and (
            preferred_route == BF16_ROUTE_DIRECT_M128
            or _should_use_h12_direct_n32(
                compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
                num_heads=num_heads,
                max_sequence_length=max_seq_len,
            )
        )
    ):
        return BF16_ROUTE_DIRECT_M128
    return _direct_m128_route(num_heads=num_heads, max_sequence_length=max_seq_len)


def _require_tensor(
    tensor: Any, *, name: str, dtype: Any, ndim: int, contiguous: bool = True
) -> None:
    import torch

    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.ndim != ndim:
        raise ValueError(
            f"{name} must have rank {ndim}, got shape {tuple(tensor.shape)}"
        )
    if contiguous and (not tensor.is_contiguous()):
        raise ValueError(f"{name} must be contiguous")


def _validate_qkv_layout(q, k, v) -> bool:
    """Accept dense ``[B, T, H, 128]`` BF16 q / k / v or strided views of one packed row.

    Returns ``True`` for dense operands.  A strided operand must keep a dense
    ``[num_heads, 128]`` token payload, a token stride that is a multiple of 8
    elements and at least ``num_heads * 128``, and a plain batch stride
    (``shape[1] * token stride``) so ``[B, T]`` folds to ``[1, B*T]``.  Every
    prefill body reads each operand's token pitch from its TensorView (TMA
    descriptors and raw tail loads alike), so q, k and v may carry different
    pitches (e.g. a dense zero ``v``) and no body needs a dense copy.
    """
    import torch

    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _require_tensor(
            tensor, name=name, dtype=torch.bfloat16, ndim=4, contiguous=False
        )
    if q.is_contiguous() and k.is_contiguous() and v.is_contiguous():
        return True
    heads_x_dim = q.shape[2] * HEAD_DIM
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(3) != 1 or tensor.stride(2) != HEAD_DIM:
            raise ValueError(
                f"{name} must keep a dense [num_heads, {HEAD_DIM}] token payload"
            )
        if tensor.stride(1) < heads_x_dim or tensor.stride(1) % 8 != 0:
            raise ValueError(
                f"{name} token stride must be a multiple of 8 elements and at least "
                f"num_heads * {HEAD_DIM}; got {tensor.stride(1)}"
            )
        if tensor.shape[0] > 1 and tensor.stride(0) != tensor.shape[1] * tensor.stride(
            1
        ):
            raise ValueError(f"{name} batch stride must be shape[1] * token stride")
    return False


@cache
def _exported_fused_m128_bodies(gpu_arch: str) -> tuple[dict[str, Any], ...]:
    """Chunk and factory kwargs of every exported fused M128 body of ``gpu_arch``."""
    from flashinfer.jit.cake_kda_tf32 import FACTORIES

    return tuple(
        {"chunk": args[0], **dict(kwargs)}
        for args, kwargs in FACTORIES.get(gpu_arch, {}).get(
            "compiled_bf16_fused_m128", {}
        )
    )


def _fused_m128_body_exported(gpu_arch: str, chunk: int, **flags: Any) -> bool:
    """Whether an exported fused M128 body of ``chunk`` tokens carries ``flags``.

    A flag value of ``None`` requires the flag to be absent from the body's
    factory signature (the chunk-carrier flags exist only where they apply).
    The planner requests a measured specialization only where the registry
    exports it and otherwise keeps the exported sibling body.
    """
    return any(
        body["chunk"] == chunk
        and all(body.get(name) == value for name, value in flags.items())
        for body in _exported_fused_m128_bodies(gpu_arch)
    )


class FlashKDABlackwellBF16FusedLaunch:
    """Preallocated single-kernel launch for the production BF16 path."""

    schedule = "fused_pipelined_mmasync_tcgen05_swap_ab_n32"
    _force_direct_m128 = False
    _force_direct_m128_n32 = False
    _force_independent_dvsplit = False
    _force_small_bh_owner_helper = False
    _keepalive: tuple[Any, ...]
    _force_bt16_prepare_chain = False
    _state_dtype_is_fp32 = False
    _n32_ft_slab = False
    _affine_operator_export = False
    _pdl_wait_initial_state_f32 = False
    _pdl_publish_final_state = False
    _checkpoint_accumulate = False
    _affine_main_indexed_initial = False
    _active_beta_f32 = False

    def _build_body(self, factory, *args, work_bin_turnover=False, **kwargs):
        """Build one body and record which raw q / k / v pointers it never reads.

        With ``work_bin_turnover`` a ``_factory`` build takes the loader's
        turnover build of the resolved module (``_work_bin_turnover_factory``)
        where ``_work_bin_turnover_program`` names one; every other build keeps
        ``factory``.
        """
        if (
            work_bin_turnover
            and getattr(factory, "func", None) is _factory
            and _work_bin_turnover_program(
                *factory.args, *args, **{**factory.keywords, **kwargs}
            )
            is not None
        ):
            factory = partial(
                _work_bin_turnover_factory, *factory.args, **factory.keywords
            )
        module = self._build_kda_module(factory, *args, **kwargs)
        self._qkv_carrier_names[id(module)] = _qkv_raw_pointer_names(factory)
        return module

    @staticmethod
    def _build_kda_module(factory, *args, **kwargs):
        return _build_kda_module(factory, *args, **kwargs)

    def __init__(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale: float,
        out,
        A_log,
        dt_bias,
        lower_bound: float | None,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
        state_indices=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens: int = 0,
        backend: str = "cuda_cpp",
        compute_dtype: str = "bf16",
        sequence_lengths: tuple[int, ...] | None = None,
        _affine_factor_cache=None,
        _affine_factor_cache_mode: int = 0,
        _affine_map_only: bool = False,
        _affine_map_output: bool = False,
        _affine_cache_token_offset: int = 0,
        _affine_cache_part_offset: int = 0,
        _affine_active_beta_f32: bool = False,
        _affine_checkpoint_rows_resolved_on_device: bool = False,
    ) -> None:
        import torch

        # The affine composite resolves each window's destination row start on
        # the device at every launch (caller cu_starts gathered per window), so
        # the host cannot value-check the offsets; shape checks still apply.
        self._affine_checkpoint_rows_resolved_on_device = (
            _affine_checkpoint_rows_resolved_on_device
        )
        if _affine_active_beta_f32:
            if compute_dtype != "tf32":
                raise ValueError("active-beta affine windows require TF32 compute")
            self._active_beta_f32 = True
        validate_kda_state_tensors(compute_dtype, initial_state, final_state)
        validate_kda_state_dtype(
            compute_dtype, state_dtype_is_fp32=self._state_dtype_is_fp32
        )
        self.compute_dtype = compute_dtype
        self._qkv_carrier_names: dict[int, tuple[str, ...]] = {}
        _require_tensor(out, name="out", dtype=torch.bfloat16, ndim=4)
        # q / k / v may be strided views of a packed qkv row (token pitch >
        # num_heads * HEAD_DIM) as long as each token's [num_heads, HEAD_DIM]
        # payload is dense.  Every body reads that pitch from its TensorView,
        # so the views are launched in place: no dense copy.
        _validate_qkv_layout(q, k, v)
        _require_tensor(g, name="g", dtype=torch.bfloat16, ndim=4, contiguous=False)
        _require_tensor(
            beta,
            name="beta",
            dtype=torch.float32 if self._active_beta_f32 else torch.bfloat16,
            ndim=3,
            contiguous=False,
        )
        _require_tensor(A_log, name="A_log", dtype=torch.float32, ndim=1)
        _require_tensor(dt_bias, name="dt_bias", dtype=torch.float32, ndim=2)
        if q.shape != k.shape or q.shape != v.shape or q.shape != out.shape:
            raise ValueError("q, k, v, and out must have identical [B,T,H,128] shapes")
        batch, tokens_per_batch, num_heads, head_dim = q.shape
        if head_dim != HEAD_DIM:
            raise ValueError(f"FlashKDA requires K=V={HEAD_DIM}, got {head_dim}")
        if (
            g.shape[0] != batch
            or g.shape[1] < tokens_per_batch
            or g.shape[2:] != (num_heads, HEAD_DIM)
        ):
            raise ValueError(
                "g must have shape [B,T_storage,H,128] with T_storage >= q logical T"
            )
        if (
            beta.shape[0] != batch
            or beta.shape[1] < tokens_per_batch
            or beta.shape[2] != num_heads
        ):
            raise ValueError(
                "beta must have shape [B,T_storage,H] with T_storage >= q logical T"
            )
        if A_log.shape != (num_heads,):
            raise ValueError("A_log must have shape [H]")
        if dt_bias.shape != (num_heads, HEAD_DIM):
            raise ValueError("dt_bias must have shape [H,128]")
        gate_kind = gate_kind_from_lower_bound(lower_bound)
        unbounded_softplus = gate_kind == KDAGateKind.UNBOUNDED_SOFTPLUS
        if gate_kind == KDAGateKind.LOWER_BOUND and (
            not -5.0 <= float(lower_bound) <= 0.0
        ):
            raise ValueError("lower_bound must be None or in [-5.0, 0.0]")
        total_tokens = batch * tokens_per_batch
        fixed_layout = cu_seqlens is None
        if cu_seqlens is None:
            resolved_sequence_lengths = (tokens_per_batch,) * batch
            if (
                sequence_lengths is not None
                and tuple(sequence_lengths) != resolved_sequence_lengths
            ):
                raise ValueError(
                    "sequence_lengths must match the fixed [B,T] launch shape"
                )
            cu_seqlens = torch.arange(
                0,
                total_tokens + 1,
                tokens_per_batch,
                dtype=torch.int64,
                device=q.device,
            )
            num_seqs = batch
        else:
            _require_tensor(cu_seqlens, name="cu_seqlens", dtype=torch.int64, ndim=1)
            if batch != 1:
                raise ValueError("packed varlen mode requires B=1")
            num_seqs = cu_seqlens.numel() - 1
            if num_seqs <= 0:
                raise ValueError("cu_seqlens must contain at least two entries")
            if sequence_lengths is None:
                raise ValueError(
                    "Packed KDA preparation requires host sequence_lengths"
                )
            resolved_sequence_lengths = tuple(sequence_lengths)
            if (
                len(resolved_sequence_lengths) != num_seqs
                or any(
                    (
                        isinstance(length, bool)
                        or not isinstance(length, int)
                        or length <= 0
                        for length in resolved_sequence_lengths
                    )
                )
                or sum(resolved_sequence_lengths) != total_tokens
            ):
                raise ValueError(
                    "sequence_lengths must be positive host integers that exactly partition the packed token count"
                )
            offsets = [0]
            for length in resolved_sequence_lengths:
                offsets.append(offsets[-1] + length)
        if fixed_layout:
            offsets = [index * tokens_per_batch for index in range(batch + 1)]
        if offsets[0] != 0 or offsets[-1] != total_tokens:
            raise ValueError(
                "cu_seqlens must start at 0 and end at the packed token count"
            )
        if any(
            (end <= start for start, end in zip(offsets, offsets[1:], strict=False))
        ):
            raise ValueError(
                "cu_seqlens must describe non-empty, strictly increasing sequences"
            )
        ordered_sequences = sorted(
            range(num_seqs),
            key=lambda seq_idx: offsets[seq_idx + 1] - offsets[seq_idx],
            reverse=True,
        )
        gpu_arch = detect_gpu_arch()
        sm_count = _device_sm_count(q.device)
        total_tasks = num_seqs * num_heads
        n32_value_rows = 128
        n32_checkpoint_tma = False
        uniform_sequences = (
            len({offsets[index + 1] - offsets[index] for index in range(num_seqs)}) == 1
        )
        lpt_loads: tuple[int, ...] = ()
        if (
            _uses_measured_sm100_persistent_policy(
                compute_capability=_COMPUTE_CAPABILITY[gpu_arch], sm_count=sm_count
            )
            and (not uniform_sequences)
            and (total_tasks > sm_count)
        ):
            ordered_seq_lens = tuple(
                (offsets[index + 1] - offsets[index] for index in ordered_sequences)
            )
            _task_ids, _task_offsets, lpt_loads = _make_lpt_task_bins(
                ordered_seq_lens, num_heads=num_heads, sm_count=sm_count
            )
        # FP32 intermediate states are carried only by the fused direct M128
        # N32 body (the M64 value split, the N16 tile and the owner/helper
        # routes write BF16 rows), so an FP32 checkpoint request pins that
        # tile exactly like an explicit N32 request.
        fp32_checkpoint_request = bool(
            checkpoint_every_n_tokens
            and state_checkpoints is not None
            and state_checkpoints.dtype == torch.float32
        )
        # The FP32 chunk carrier also serves unbounded BF16-compute prefill on
        # the FP32 external state pool without a checkpoint request: its final
        # state resumes decode, and the BF16 carrier drifts to 0.02-0.05 rel L2
        # on real 8K activations (Triton 0.002).
        # BF16 checkpoint rows keep the legacy BF16 carrier: the body's
        # CHECKPOINT_DTYPE_IS_FP32 specialization also types the checkpoint
        # rows, so the FP32 carrier is only legal without rows or with FP32 rows.
        # Affine split parts (_n32_ft_slab) carry workspace slabs, not the
        # caller's state pool, and run outside the serving-native ABI: they
        # keep the BF16 carrier the composite was measured with.
        fp32_carrier_request = fp32_checkpoint_request or (
            unbounded_softplus
            and self._state_dtype_is_fp32
            and not self._n32_ft_slab
            and compute_dtype == "bf16"
            and state_checkpoints is None
        )
        force_direct_m128_n32 = self._force_direct_m128_n32 or fp32_carrier_request
        needs_direct_m128 = (
            unbounded_softplus
            or self._force_direct_m128
            or checkpoint_every_n_tokens != 0
            or (not beta.is_contiguous())
            or self._active_beta_f32
        )
        has_nondefault_state_slot_stride = any(
            (
                state is not None
                and state.ndim == 4
                and (state.stride(0) != num_heads * HEAD_DIM * HEAD_DIM)
                for state in (initial_state, final_state)
            )
        )
        max_seq_len = max(
            (end - start for start, end in zip(offsets, offsets[1:], strict=False))
        )
        use_tf32_direct_m128 = compute_dtype == "tf32" and self._force_direct_m128
        if self._force_independent_dvsplit:
            if gate_kind != KDAGateKind.LOWER_BOUND:
                raise ValueError("fused M64 requires bounded gates")
            route = BF16_ROUTE_M64
        elif use_tf32_direct_m128:
            route = (
                BF16_ROUTE_DIRECT_M128
                if self._force_direct_m128_n32 or unbounded_softplus
                else BF16_ROUTE_DIRECT_M128_N16
            )
        elif self._force_bt16_prepare_chain:
            if (
                unbounded_softplus
                or (state_indices is not None and (not self._state_dtype_is_fp32))
                or any(
                    (
                        state is not None
                        and state.stride(0) != num_heads * HEAD_DIM * HEAD_DIM
                        and (not self._state_dtype_is_fp32)
                        for state in (initial_state, final_state)
                    )
                )
            ):
                raise ValueError(
                    "forced BT16 prepare/chain requires bounded, contiguous tensors"
                )
            route = BF16_ROUTE_BT16_M64
        elif self._force_small_bh_owner_helper:
            if (
                unbounded_softplus
                and compute_dtype != "tf32"
                or (
                    compute_dtype != "tf32"
                    and (
                        total_tasks > SMALL_BH_MAX_TASKS
                        or num_heads > SMALL_BH_MAX_TASKS
                    )
                )
                or SMALL_BH_GROUP_SIZE * total_tasks > sm_count
                or (
                    compute_dtype == "bf16"
                    and (checkpoint_every_n_tokens != 0 or not beta.is_contiguous())
                )
                or (compute_dtype == "tf32" and checkpoint_every_n_tokens % 32 != 0)
            ):
                raise ValueError(
                    "forced small-BH owner/helper requires resident eight-CTA groups (BF16 allows at most eight tasks); BF16 requires bounded gates and contiguous beta without checkpoints, TF32 requires 32-token-aligned checkpoints"
                )
            route = BF16_ROUTE_SMALL_BH_M128
        elif _should_use_h12_active_beta_m64(
            gpu_arch=gpu_arch,
            sm_count=sm_count,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            total_tasks=total_tasks,
            active_beta_f32=self._active_beta_f32,
            state_dtype_is_fp32=self._state_dtype_is_fp32,
            indexed_state_pool=state_indices is not None,
            in_place_state_pool=initial_state is not None
            and initial_state is final_state,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            gate_kind=gate_kind,
            force_direct_m128=self._force_direct_m128,
            force_direct_m128_n32=force_direct_m128_n32,
        ):
            route = BF16_ROUTE_M64
        elif needs_direct_m128:
            route = _constrained_direct_m128_route(
                gpu_arch=gpu_arch,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
                force_direct_m128=self._force_direct_m128,
                force_direct_m128_n32=force_direct_m128_n32,
                unbounded_softplus=unbounded_softplus,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                indexed_state_pool=state_indices is not None,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                preferred_route=_select_bf16_route(
                    compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
                    sm_count=sm_count,
                    fixed_layout=fixed_layout,
                    num_sequences=num_seqs,
                    num_heads=num_heads,
                    uniform_sequences=uniform_sequences,
                    lpt_loads=lpt_loads,
                    max_sequence_length=max_seq_len,
                    use_initial_state=initial_state is not None,
                    store_final_state=final_state is not None,
                )
                if compute_dtype == "bf16"
                else BF16_ROUTE_DIRECT_M128_N16,
            )
        else:
            route = _select_bf16_route(
                compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_sequences=num_seqs,
                num_heads=num_heads,
                uniform_sequences=uniform_sequences,
                lpt_loads=lpt_loads,
                max_sequence_length=max_seq_len,
                use_initial_state=initial_state is not None,
                store_final_state=final_state is not None,
            )
            if (
                compute_dtype == "tf32"
                and route == BF16_ROUTE_DIRECT_M128
                and _direct_m128_route(
                    num_heads=num_heads, max_sequence_length=max_seq_len
                )
                == BF16_ROUTE_DIRECT_M128
                and _should_use_uniform_piece_persistent(
                    gpu_arch=gpu_arch,
                    sm_count=sm_count,
                    num_seqs=num_seqs,
                    num_heads=num_heads,
                    uniform_sequences=uniform_sequences,
                    max_seq_len=max_seq_len,
                    use_initial_state=initial_state is not None,
                    store_final_state=final_state is not None,
                )
            ):
                # The shared planner requests recurrence pieces on 148- and
                # 152-SM parts only; the TF32 launch keeps requesting them
                # wherever their roofline wins (the family is not exported).
                route = BF16_ROUTE_PIECE_M128
            if compute_dtype == "bf16" and route not in {
                BF16_ROUTE_DIRECT_M128,
                BF16_ROUTE_DIRECT_M128_N16,
                BF16_ROUTE_M64,
            }:
                # The BT16, small-BH, vtile, scalar-chunk and persistent
                # families have no exported BF16 body: their grids take the
                # direct family (the one-wave value split below still applies).
                route = _direct_m128_route(
                    num_heads=num_heads, max_sequence_length=max_seq_len
                )
        if (
            route == BF16_ROUTE_BT16_M64
            and (not self._state_dtype_is_fp32)
            and (state_indices is not None or has_nondefault_state_slot_stride)
        ):
            route = _direct_m128_route(
                num_heads=num_heads, max_sequence_length=max_seq_len
            )
        checkpoint_fits_n32 = (
            checkpoint_every_n_tokens == 0
            or checkpoint_every_n_tokens % BF16_M128_CHUNK == 0
        )
        if not checkpoint_fits_n32:
            if self._force_direct_m128_n32:
                raise ValueError(
                    "forced N32 requires checkpoint_every_n_tokens to be a multiple of 32"
                )
            route = BF16_ROUTE_DIRECT_M128_N16
        # The M64 module is threaded only through the cuda_cpp backend; the
        # direct PTX backend keeps its direct tile.
        if (
            route
            in {
                BF16_ROUTE_DIRECT_M128,
                BF16_ROUTE_DIRECT_M128_N16,
            }
            and backend == "cuda_cpp"
            and _should_use_bf16_one_wave_dvsplit(
                gpu_arch=gpu_arch,
                sm_count=sm_count,
                total_tasks=total_tasks,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                bounded_gate=gate_kind == KDAGateKind.LOWER_BOUND,
                compute_dtype=compute_dtype,
                force_direct_m128=self._force_direct_m128,
                force_direct_m128_n32=force_direct_m128_n32,
                # Logit beta whose token pitch the 8x32 TMA box cannot encode is
                # refreshed into a padded carrier each launch (see
                # _pad_beta_tma_source); active FP32 beta never uses that box.
                beta_tma_refresh=(
                    not self._active_beta_f32
                    and not (
                        total_tokens >= BF16_M128_CHUNK
                        and num_heads >= BF16_BETA_TMA_MIN_HEADS
                        and beta.stride(1) * beta.element_size() % 16 == 0
                    )
                ),
                num_heads=num_heads,
                max_seq_len=max_seq_len,
            )
        ):
            # The one-wave BF16 value split replaces every direct tile the
            # policy above resolved, including the checkpoint-constrained and
            # active-beta direct families; forced tiles were excluded above.
            route = BF16_ROUTE_M64
        if (
            route == BF16_ROUTE_M64
            and not self._force_independent_dvsplit
            # Active FP32 beta has no direct body beyond 256 tokens (the
            # direct active-beta family is the short H12 indexed schedule);
            # its value split keeps the BF16 carrier, documented in the
            # constant's note.
            and not self._active_beta_f32
            and not _dvsplit_carrier_precision_ok(
                compute_dtype=compute_dtype,
                bounded_gate=gate_kind == KDAGateKind.LOWER_BOUND,
                max_seq_len=max_seq_len,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
            )
        ):
            # Every automatic M64 selection above (one-wave split, H12
            # active-beta split, H64 fixed-layout split) on a BF16 pool shares
            # the BF16 chunk carrier; bounded sequences beyond one chunk take
            # the FP32 carrier of the direct family instead.  The N32 tile is
            # the measured preference wherever the checkpoint cadence allows it
            # (CAKE-736 round 8: N16 is 1.75x slower than the split at H12
            # 8192 tokens, N32 1.10x); explicit M64 requests keep their body.
            # FP32 pools keep the split: its body carries FP32 chunk state
            # (CAKE-736 round 9).
            route = (
                BF16_ROUTE_DIRECT_M128
                if checkpoint_fits_n32 and not self._force_direct_m128
                else BF16_ROUTE_DIRECT_M128_N16
            )
        if compute_dtype == "tf32":
            if route in {
                BF16_ROUTE_SMALL_BH_M128,
                BF16_ROUTE_SOURCE_VTILE_M128,
                BF16_ROUTE_SCALAR_CHUNK_LPT_M128,
            }:
                if not self._force_small_bh_owner_helper:
                    route = (
                        BF16_ROUTE_DIRECT_M128
                        if checkpoint_fits_n32
                        else BF16_ROUTE_DIRECT_M128_N16
                    )
            if route in {BF16_ROUTE_DIRECT_M128, BF16_ROUTE_DIRECT_M128_N16}:
                if unbounded_softplus and route != BF16_ROUTE_DIRECT_M128:
                    raise ValueError(
                        "TF32 unbounded gates require 32-token-aligned checkpoints"
                    )
                if (
                    not self._force_direct_m128
                    and checkpoint_fits_n32
                    and (16 < max_seq_len <= 512)
                ):
                    route = BF16_ROUTE_DIRECT_M128
                use_tf32_direct_m128 = True
        use_tf32_direct_n32 = use_tf32_direct_m128 and route == BF16_ROUTE_DIRECT_M128
        use_small_bh_owner_helper = route == BF16_ROUTE_SMALL_BH_M128
        use_tf32_owner_helper = compute_dtype == "tf32" and use_small_bh_owner_helper
        use_bt16_prepare_chain = route == BF16_ROUTE_BT16_M64
        use_bt16_s7_chain = (
            use_bt16_prepare_chain and BT16_VALUE_SPLITS * total_tasks > sm_count
        )
        use_independent_dvsplit = route == BF16_ROUTE_M64
        use_source_vtile_m128 = route == BF16_ROUTE_SOURCE_VTILE_M128
        use_bt16_beta_tma = (
            use_bt16_prepare_chain
            and (not self._active_beta_f32)
            and (num_heads % BF16_BETA_TMA_MIN_HEADS == 0)
            and _should_use_bt16_dense_wavefront(
                compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_sequences=num_seqs,
                num_heads=num_heads,
                max_sequence_length=max_seq_len,
            )
        )
        use_bt16_s9_chain = (
            use_bt16_prepare_chain and (not use_bt16_s7_chain) and total_tasks <= 8
        )
        use_direct_m128_n16 = route == BF16_ROUTE_DIRECT_M128_N16
        if (
            self._active_beta_f32
            and (
                not (
                    use_bt16_prepare_chain
                    or use_independent_dvsplit
                    or use_tf32_direct_m128
                    or use_tf32_owner_helper
                    or (
                        self._force_direct_m128
                        and self._state_dtype_is_fp32
                        and (gate_kind == KDAGateKind.LOWER_BOUND)
                    )
                )
            )
            and (
                gate_kind != KDAGateKind.LOWER_BOUND
                or num_heads != 12
                or (not self._state_dtype_is_fp32)
                or (state_indices is None)
                or (initial_state is None)
                or (initial_state is not final_state)
                or (checkpoint_every_n_tokens != 64)
                or (max_seq_len > 256)
                or (
                    route
                    not in {
                        BF16_ROUTE_DIRECT_M128,
                        BF16_ROUTE_DIRECT_M128_N16,
                        BF16_ROUTE_M64,
                    }
                )
            )
        ):
            raise ValueError(
                "active FP32 beta requires bounded H12 direct prefill with an indexed in-place FP32 state pool, 64-token checkpoints, and max_seq_len <= 256"
            )
        use_scalar_chunk_lpt_m128 = route == BF16_ROUTE_SCALAR_CHUNK_LPT_M128
        use_persistent_m128 = route in {
            BF16_ROUTE_HEAD_GROUPED_M128,
            BF16_ROUTE_LPT_M128,
            BF16_ROUTE_PIECE_M128,
        }
        if route in {BF16_ROUTE_DIRECT_M128, BF16_ROUTE_DIRECT_M128_N16}:
            direct_chunk_tokens = (
                BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK
            )
            ordered_sequences = list(
                _make_direct_sequence_order(
                    tuple(
                        (
                            offsets[index + 1] - offsets[index]
                            for index in range(num_seqs)
                        )
                    ),
                    num_heads=num_heads,
                    worker_count=sm_count,
                    chunk_tokens=direct_chunk_tokens,
                )
            )
        host_uploads: dict[str, list[int]] = {"seq_order": list(ordered_sequences)}
        if total_tasks <= 0:
            raise ValueError("num_seqs * num_heads must be positive")
        if state_indices is not None:
            _require_tensor(
                state_indices, name="state_indices", dtype=torch.int32, ndim=1
            )
            if state_indices.numel() != num_seqs:
                raise ValueError("state_indices must contain one slot per sequence")
        state_slot_stride = num_heads * HEAD_DIM * HEAD_DIM
        observed_state_stride = None
        expected_state_dtype = (
            torch.float32 if self._state_dtype_is_fp32 else torch.bfloat16
        )
        for name, state in (
            ("initial_state", initial_state),
            ("final_state", final_state),
        ):
            if state is None:
                continue
            _require_tensor(
                state, name=name, dtype=expected_state_dtype, ndim=4, contiguous=False
            )
            expected_slots = state.shape[0] if state_indices is not None else num_seqs
            if state.shape != (expected_slots, num_heads, HEAD_DIM, HEAD_DIM):
                raise ValueError(f"{name} must have shape [N_or_pool,H,128,128]")
            if state.stride()[1:] != (HEAD_DIM * HEAD_DIM, HEAD_DIM, 1):
                raise ValueError(
                    f"{name} must be contiguous inside each state-pool slot"
                )
            if (
                observed_state_stride is not None
                and state.stride(0) != observed_state_stride
            ):
                raise ValueError(
                    "initial_state and final_state must have the same slot stride"
                )
            observed_state_stride = state.stride(0)
            state_slot_stride = state.stride(0)
        if (
            checkpoint_every_n_tokens < 0
            or checkpoint_every_n_tokens % BF16_N16_M128_CHUNK != 0
        ):
            raise ValueError(
                f"checkpoint_every_n_tokens must be zero or a multiple of {BF16_N16_M128_CHUNK}"
            )
        if (
            use_independent_dvsplit
            and compute_dtype == "bf16"
            and checkpoint_every_n_tokens % 32
        ):
            raise ValueError(
                "BF16 fused M64 checkpoint interval must be a multiple of 32"
            )
        # BF16 checkpoints are the legacy observation contract.  FP32
        # checkpoints (the intermediate-state contract that resumes
        # recurrences from them) require FP32 external state I/O, and only
        # the fused M128 body carries the FP32 chunk carrier.
        fp32_checkpoints = bool(
            checkpoint_every_n_tokens
            and state_checkpoints is not None
            and state_checkpoints.dtype == torch.float32
        )
        if checkpoint_every_n_tokens:
            if state_checkpoints is None or checkpoint_cu_starts is None:
                raise ValueError(
                    "state_checkpoints and checkpoint_cu_starts are required when checkpointing"
                )
            if fp32_checkpoints and not self._state_dtype_is_fp32:
                raise ValueError(
                    "FP32 state_checkpoints require FP32 initial/final state"
                )
            _require_tensor(
                state_checkpoints,
                name="state_checkpoints",
                dtype=torch.float32 if fp32_checkpoints else torch.bfloat16,
                ndim=4,
            )
            _require_tensor(
                checkpoint_cu_starts,
                name="checkpoint_cu_starts",
                dtype=torch.int64,
                ndim=1,
            )
            if state_checkpoints.shape[1:] != (num_heads, HEAD_DIM, HEAD_DIM):
                raise ValueError("state_checkpoints must have shape [C,H,128,128]")
            if checkpoint_cu_starts.numel() != num_seqs + 1:
                raise ValueError("checkpoint_cu_starts must have shape [N+1]")
            expected_counts = [
                (end - start + checkpoint_every_n_tokens - 1)
                // checkpoint_every_n_tokens
                for start, end in zip(offsets, offsets[1:], strict=False)
            ]
            checkpoint_offsets = [0]
            for count in expected_counts:
                checkpoint_offsets.append(checkpoint_offsets[-1] + count)
            actual_counts = [
                end - start
                for start, end in zip(
                    checkpoint_offsets, checkpoint_offsets[1:], strict=False
                )
            ]
            if checkpoint_offsets[0] != 0 or actual_counts != expected_counts:
                raise ValueError(
                    "checkpoint_cu_starts must encode ceil(seq_len / checkpoint_every_n_tokens) rows"
                )
            if self._affine_checkpoint_rows_resolved_on_device:
                if state_checkpoints.shape[0] < checkpoint_offsets[-1]:
                    raise ValueError(
                        "state_checkpoints has fewer rows than the affine windows write"
                    )
            elif checkpoint_offsets[-1] != state_checkpoints.shape[0]:
                raise ValueError(
                    "state_checkpoints row count does not match checkpoint_cu_starts"
                )
        elif state_checkpoints is not None or checkpoint_cu_starts is not None:
            raise ValueError("checkpoint tensors require checkpoint_every_n_tokens > 0")
        q_flat = q.reshape(total_tokens, num_heads, HEAD_DIM)
        k_flat = k.reshape(total_tokens, num_heads, HEAD_DIM)
        v_flat = v.reshape(total_tokens, num_heads, HEAD_DIM)
        g_flat, g_refresh = _flatten_gate_rows(
            g, logical_tokens_per_batch=tokens_per_batch, num_heads=num_heads
        )
        # The BF16 bodies take ``g_flat`` / ``beta_abi_source`` in place: their
        # bindings accept any unit-innermost-stride view and the kernels address
        # token rows through ``g_token_stride`` / ``beta_token_stride``.  The
        # one-element carriers built here stay for the TF32 bodies (contiguous
        # raw-pointer contract of ``raw_gate`` / ``beta_logits``) and for the
        # keep-alive tuple.
        g_pointer = _ffi_raw_pointer_carrier(g_flat)
        out_flat = out.reshape(total_tokens, num_heads, HEAD_DIM)
        beta_flat, beta_refresh = _flatten_beta_rows(
            beta, logical_tokens_per_batch=tokens_per_batch, num_heads=num_heads
        )
        beta_abi_source = (
            q_flat.reshape(-1)[:8].reshape(1, 8) if self._active_beta_f32 else beta_flat
        )
        beta_pointer = _ffi_raw_pointer_carrier(beta_abi_source)
        use_scalar_beta = (
            gate_kind == KDAGateKind.LOWER_BOUND
            and route in {BF16_ROUTE_DIRECT_M128, BF16_ROUTE_DIRECT_M128_N16}
            and (num_heads == 12)
            or use_tf32_direct_m128
            or (self._active_beta_f32 and self._force_direct_m128)
            or (
                use_independent_dvsplit
                and (self._active_beta_f32 or compute_dtype == "tf32")
            )
        )
        use_n32_logical_page64 = (
            self._active_beta_f32
            and use_scalar_beta
            and (not use_direct_m128_n16)
            and (route == BF16_ROUTE_DIRECT_M128)
            and (num_heads == 12)
            and (state_indices is not None)
            and (max_seq_len <= H12_DIRECT_N32_MAX_SEQ_LEN)
        )
        # Work bins where a bin runs tasks back to back (the direct N32 and N16
        # bodies, in chunks of their own size).
        route_work_bins = (
            route in {BF16_ROUTE_DIRECT_M128, BF16_ROUTE_DIRECT_M128_N16}
            and (not use_tf32_direct_m128)
            and _use_direct_work_bins(
                tuple(offsets[index + 1] - offsets[index] for index in range(num_seqs)),
                num_heads=num_heads,
                sm_count=sm_count,
                chunk_tokens=(
                    BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK
                ),
            )
        )
        # A work-binned N32 serving launch whose longest sequence has at most
        # the part's turnover chunk count may take its module's turnover build.
        route_work_bin_turnover = _work_bin_turnover_route(
            route_work_bins=route_work_bins,
            direct_n16=use_direct_m128_n16,
            n32_ft_slab=self._n32_ft_slab,
            max_seq_len=max_seq_len,
            gpu_arch=gpu_arch,
            sm_count=sm_count,
        )
        # The bounded N32 body without the early state pack is faster on the
        # short H12 grids and bitwise identical to the early-pack body, so the
        # prepared path keeps it for every length; it also has the work-binned
        # builds the early-pack body lacks.
        use_early_n32_state_pack = False
        use_n32_register_inverse = _should_use_n32_register_inverse(
            direct_n16=use_direct_m128_n16,
            gate_kind=gate_kind,
            scalar_beta=use_scalar_beta,
            max_seq_len=max_seq_len,
        )
        use_n32_prediction_first = _should_prioritize_n32_prediction(
            gpu_arch=gpu_arch,
            route=route,
            uniform_sequences=uniform_sequences,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            register_inverse=use_n32_register_inverse,
        ) and _fused_m128_body_exported(
            gpu_arch,
            BF16_M128_CHUNK,
            n32_prediction_first=True,
            n32_ft_slab=self._n32_ft_slab,
        )
        use_n32_tensor_state_decay = (
            checkpoint_every_n_tokens == 0
            and _fused_m128_body_exported(
                gpu_arch, BF16_M128_CHUNK, tensor_state_decay=True
            )
            and _should_use_n32_tensor_state_decay(
                gpu_arch=gpu_arch,
                route=route,
                uniform_sequences=uniform_sequences,
                num_heads=num_heads,
                total_tasks=total_tasks,
                max_seq_len=max_seq_len,
                prediction_first=use_n32_prediction_first,
            )
        )
        # Standalone bodies take the FP32 chunk carrier for FP32 rows and for
        # unbounded prefill without rows (``fp32_carrier`` below); the registry
        # ships the pair-packed H12 beta layout only for some of those flavours.
        pair_packed_beta_exported = (
            compute_dtype != "bf16"
            or self._n32_ft_slab
            or _fused_m128_body_exported(
                gpu_arch,
                BF16_M128_CHUNK,
                gate_kind=gate_kind.value,
                n32_ft_slab=False,
                pair_packed_beta=True,
                checkpoint_dtype_is_fp32=True
                if fp32_checkpoints
                or (
                    unbounded_softplus
                    and self._state_dtype_is_fp32
                    and state_checkpoints is None
                )
                else None,
            )
        )
        pair_packed_beta_tma = (
            _pair_packed_beta_tma_source(beta_flat.unsqueeze(0))
            if not self._active_beta_f32
            and route == BF16_ROUTE_DIRECT_M128
            and (num_heads == 12)
            and (max_seq_len > H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQ_LEN)
            and pair_packed_beta_exported
            else None
        )
        use_pair_packed_beta = pair_packed_beta_tma is not None
        beta_tma = (
            q_flat.reshape(-1)[: BF16_M128_CHUNK * BF16_BETA_TMA_MIN_HEADS].reshape(
                BF16_M128_CHUNK, BF16_BETA_TMA_MIN_HEADS
            )
            if self._active_beta_f32
            and (route == BF16_ROUTE_M64 or use_tf32_direct_m128)
            else beta_abi_source
            if self._active_beta_f32
            else _pad_beta_tma_source(
                beta_flat, total_tokens=total_tokens, num_heads=num_heads
            )
            if use_bt16_prepare_chain
            else pair_packed_beta_tma
            if use_pair_packed_beta
            else _pad_beta_tma_source(
                beta_flat, total_tokens=total_tokens, num_heads=num_heads
            )
        )
        beta_tma_valid = None
        if (
            beta_tma is not beta_flat
            and (not self._active_beta_f32)
            and (not use_pair_packed_beta)
            and (not use_scalar_beta)
            and (not use_bt16_prepare_chain)
            and _beta_tma_copy_required(
                offsets,
                chunk_tokens=BT16_CHUNK
                if use_bt16_prepare_chain
                else BF16_N16_M128_CHUNK
                if use_direct_m128_n16
                else BF16_M128_CHUNK,
            )
        ):
            beta_tma_valid = beta_tma[:total_tokens, :num_heads]
        empty_state = torch.empty(1, dtype=torch.bfloat16, device=q.device)
        empty_checkpoint_tma = torch.empty(
            (1, 1, HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16, device=q.device
        )
        empty_i32 = torch.empty(1, dtype=torch.int32, device=q.device)
        empty_i64 = torch.empty(1, dtype=torch.int64, device=q.device)
        empty_chunk_offsets = torch.zeros(1, dtype=torch.int64, device=q.device)
        empty_f32 = torch.empty(1, dtype=torch.float32, device=q.device)
        empty_u32 = torch.empty(1, dtype=torch.uint32, device=q.device)
        initial_state_pointer = _ffi_raw_pointer_carrier(
            initial_state
            if initial_state is not None and (not self._state_dtype_is_fp32)
            else empty_state
        )
        final_state_pointer = _ffi_raw_pointer_carrier(
            final_state
            if final_state is not None and (not self._state_dtype_is_fp32)
            else empty_state
        )
        initial_state_f32_pointer = _ffi_raw_pointer_carrier(
            initial_state
            if initial_state is not None and self._state_dtype_is_fp32
            else empty_f32
        )
        final_state_f32_pointer = _ffi_raw_pointer_carrier(
            final_state
            if final_state is not None and self._state_dtype_is_fp32
            else empty_f32
        )
        mid_state = empty_state
        mid_state_ready = empty_u32
        bt16_qd = empty_state
        bt16_kd = empty_state
        bt16_w = empty_state
        bt16_qk = empty_state
        bt16_diag = torch.empty(1, dtype=torch.float32, device=q.device)
        bt16_total_chunks = 0
        bt16_chunks_per_prep_cta = BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA
        bt16_prepare_total_ctas = 0
        if use_bt16_prepare_chain:
            chunk_counts = [
                (end - start + BT16_CHUNK - 1) // BT16_CHUNK
                for start, end in zip(offsets, offsets[1:], strict=False)
            ]
            host_cu_chunks = [0]
            host_chunk_to_seq: list[int] = []
            for seq_idx, chunk_count in enumerate(chunk_counts):
                host_cu_chunks.append(host_cu_chunks[-1] + chunk_count)
                host_chunk_to_seq.extend([seq_idx] * chunk_count)
            bt16_total_chunks = host_cu_chunks[-1]
            bt16_chunks_per_prep_cta = _bt16_chunks_per_prep_cta(
                num_heads=num_heads,
                total_chunks=bt16_total_chunks,
                compute_dtype=compute_dtype,
            )
            bt16_prepare_total_ctas = (
                (bt16_total_chunks + bt16_chunks_per_prep_cta - 1)
                // bt16_chunks_per_prep_cta
                * num_heads
            )
            bt16_prepare_total_ctas = _wave_quantized_bt16_prepare_ctas(
                rectangular_ctas=bt16_prepare_total_ctas,
                num_heads=num_heads,
                sm_count=sm_count,
            )
            if compute_dtype == "tf32":
                bt16_prepare_total_ctas = max(
                    num_heads,
                    min(
                        bt16_prepare_total_ctas, TF32_BT16_PREP_RESIDENT_CTAS * sm_count
                    ),
                )
            if _should_use_bt16_dense_wavefront(
                compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_sequences=num_seqs,
                num_heads=num_heads,
                max_sequence_length=max_seq_len,
            ):
                bt16_prepare_total_ctas = min(
                    num_heads * bt16_total_chunks, BT16_DENSE_PREP_WAVES * sm_count
                )
            host_uploads["bt16_cu_chunks"] = list(host_cu_chunks)
            host_uploads["bt16_chunk_to_seq"] = list(host_chunk_to_seq)
            padded_tokens = bt16_total_chunks * BT16_CHUNK
            factor_shape = (1, num_heads, padded_tokens, HEAD_DIM)
            factor_dtype = torch.float32 if compute_dtype == "tf32" else torch.bfloat16
            bt16_qd = torch.empty(factor_shape, dtype=factor_dtype, device=q.device)
            bt16_kd = torch.empty_like(bt16_qd)
            bt16_w = (
                torch.empty(
                    (1, num_heads, bt16_total_chunks, HEAD_DIM, BT16_CHUNK),
                    dtype=factor_dtype,
                    device=q.device,
                )
                if compute_dtype == "tf32"
                else torch.empty_like(bt16_qd)
            )
            bt16_qk = torch.empty(
                (1, num_heads, bt16_total_chunks, BT16_CHUNK, BT16_CHUNK),
                dtype=factor_dtype,
                device=q.device,
            )
            bt16_diag = torch.empty(
                (1, num_heads, bt16_total_chunks, HEAD_DIM),
                dtype=torch.float32,
                device=q.device,
            )
        uploaded = _upload_int32_batch(q.device, host_uploads)
        seq_order = uploaded["seq_order"]
        task_ids = uploaded.get("task_ids", seq_order)
        task_offsets = uploaded.get("task_offsets", seq_order)
        task_token_starts = uploaded.get("task_token_starts", seq_order)
        task_token_counts = uploaded.get("task_token_counts", seq_order)
        task_state_sources = uploaded.get("task_state_sources", seq_order)
        task_state_destinations = uploaded.get("task_state_destinations", seq_order)
        bt16_cu_chunks = uploaded.get("bt16_cu_chunks", empty_i32)
        bt16_chunk_to_seq = uploaded.get("bt16_chunk_to_seq", empty_i32)
        self._metadata_host = uploaded.get("_host_pinned")
        serving_native_abi = (
            self._active_beta_f32
            or state_indices is not None
            or checkpoint_every_n_tokens != 0
            or (beta_flat.stride(0) != num_heads)
            or (state_slot_stride != num_heads * HEAD_DIM * HEAD_DIM)
        )
        if compute_dtype == "tf32" and (
            not (
                use_bt16_prepare_chain
                or use_independent_dvsplit
                or use_tf32_direct_m128
                or use_persistent_m128
                or use_tf32_owner_helper
            )
        ):
            raise NotImplementedError(
                "TF32 compute currently requires BT16, fused M64, direct M128, or grouped/LPT persistent M128"
            )
        self.prepare_module = None
        uses_default_fused_m128 = not (
            use_bt16_prepare_chain
            or use_small_bh_owner_helper
            or use_independent_dvsplit
            or use_tf32_direct_m128
            or use_source_vtile_m128
            or use_scalar_chunk_lpt_m128
            or use_persistent_m128
        )
        if (
            compute_dtype == "bf16"
            and uses_default_fused_m128
            and not self._n32_ft_slab
            and not _fused_m128_body_exported(
                gpu_arch,
                BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK,
                serving_native_abi=False,
                n32_ft_slab=False,
            )
        ):
            # Every exported standalone body takes the serving ABI; its runtime
            # parameters (use_state_indices, checkpoint cadence, beta and state
            # slot strides) express the plain call as well.
            serving_native_abi = True
        if fp32_checkpoints and not uses_default_fused_m128:
            raise NotImplementedError(
                "FP32 intermediate states are only carried by the fused direct M128 body; "
                f"route {route!r} writes BF16 checkpoints"
            )
        # FP32 chunk carrier: FP32 checkpoint rows, or unbounded BF16-compute
        # prefill on the FP32 external state pool (see fp32_carrier_request).
        fp32_carrier = fp32_checkpoints or (
            unbounded_softplus
            and self._state_dtype_is_fp32
            and not self._n32_ft_slab
            and compute_dtype == "bf16"
            and uses_default_fused_m128
            and state_checkpoints is None
        )
        # Affine split parts (_n32_ft_slab) run outside the serving-native ABI
        # and cannot take the checkpoint-typed carrier unless they carry FP32
        # rows.  Their map pass (BF16 state I/O, no rows) and row-less main /
        # correction passes otherwise keep the BF16 carrier, whose per-part
        # error compounds across the composed parts (0.002 -> 0.03 rel L2 over
        # twelve parts on real 8K unbounded activations).  The standalone FP32
        # state carrier keeps every part at sequential-body precision.
        fp32_state_carrier = (
            unbounded_softplus
            and self._n32_ft_slab
            and compute_dtype == "bf16"
            and uses_default_fused_m128
            and not fp32_carrier
        )
        if backend != "cuda_cpp" and (
            compute_dtype == "tf32"
            or not (uses_default_fused_m128 or use_persistent_m128)
        ):
            raise NotImplementedError(
                f"FlashKDA route {route!r} does not thread backend={backend!r} to its compiled module"
            )
        # The registry exports no kernel for these route families: the BF16
        # BT16 prepare/chain and the beta-TMA prepare, the BF16 small-B*H
        # owner/helper, the TF32 N16 direct tile, the V-tile, scalar-chunk LPT
        # and persistent bodies.  Their launch branches could only fail inside
        # the factory; reject them here by route name instead.
        unexported_route = (
            "bt16_prepare_beta_tma"
            if use_bt16_beta_tma
            else "bt16_prepare_chain"
            if use_bt16_prepare_chain and compute_dtype != "tf32"
            else "small_bh_owner_helper"
            if use_small_bh_owner_helper and not use_tf32_owner_helper
            else "tf32_n16"
            if compute_dtype == "tf32"
            and (use_independent_dvsplit or use_tf32_direct_m128)
            and not use_tf32_direct_n32
            else "source_vtile"
            if use_source_vtile_m128
            else "scalar_chunk_lpt"
            if use_scalar_chunk_lpt_m128
            else "persistent"
            if use_persistent_m128
            else None
        )
        if unexported_route is not None:
            raise NotImplementedError(
                f"FlashKDA route {route!r} with {compute_dtype} compute resolves to "
                f"the {unexported_route} kernel family, which is not exported"
            )
        owner_helpers = 0
        owner_helper_pool = 0
        if use_bt16_prepare_chain:
            self.prepare_module = self._build_body(
                partial(_factory, "compiled_tf32_bt16_prepare"),
                active_beta_f32=self._active_beta_f32,
            )
            self.module = self._build_body(
                partial(_factory, "compiled_tf32_bt16_chain_m64_fp32_state"),
                compact_output=use_bt16_s7_chain,
                split_prediction=use_bt16_s9_chain,
                serving_native_abi=serving_native_abi,
                write_checkpoints=bool(checkpoint_every_n_tokens),
            )
            self.schedule = (
                "bt16_tf32_m64_s6_fp32_compact_output"
                if use_bt16_s7_chain
                else "bt16_tf32_m64_s6_fp32_split_prediction"
                if use_bt16_s9_chain
                else "bt16_tf32_m64_s6_fp32"
            )
        # Every M64 route builds its own module below; resolving the fused M128
        # module here as well would compile an unused kernel and break the
        # exporter's physical-stage inventory.
        elif (
            not (use_tf32_direct_m128 or use_tf32_owner_helper)
            and not use_independent_dvsplit
        ):
            if fp32_carrier and use_direct_m128_n16:
                raise NotImplementedError(
                    "the FP32 chunk carrier requires the N32 direct M128 body; "
                    "the N16 checkpoint-TMA body only carries BF16 state"
                )
            if self._checkpoint_accumulate and not fp32_carrier:
                raise ValueError(
                    "checkpoint accumulation requires FP32 checkpoint rows"
                )
            # The owner/helper program is asked for on the unbounded N32 body of
            # an unforced plan; the factory keeps the one-CTA program for a
            # module without one.
            owner_helpers = (
                _owner_helper_count(
                    gpu_arch=gpu_arch,
                    sm_count=sm_count,
                    total_tasks=total_tasks,
                    max_seq_len=max_seq_len,
                )
                if unbounded_softplus
                and compute_dtype == "bf16"
                and backend == "cuda_cpp"
                and not use_direct_m128_n16
                and not (self._force_direct_m128 or self._force_direct_m128_n32)
                else 0
            )
            # The pooled program where its pool outnumbers the per-task
            # helpers of every head; the factory keeps the owner/helper program
            # for a module without one.
            owner_helper_pool = _owner_helper_pool_count(
                sm_count=sm_count,
                num_seqs=num_seqs,
                num_heads=num_heads,
                helpers=owner_helpers,
                uniform_chunks=len(
                    {
                        (end - start + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK
                        for start, end in zip(offsets, offsets[1:], strict=False)
                    }
                )
                == 1,
            )
            self.module = self._build_body(
                partial(
                    _factory,
                    "compiled_bf16_fused_m128",
                    route_work_bins=route_work_bins,
                    **({"owner_helper": True} if owner_helpers else {}),
                    **({"owner_helper_pool": True} if owner_helper_pool else {}),
                ),
                BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK,
                work_bin_turnover=route_work_bin_turnover,
                serving_native_abi=serving_native_abi,
                gate_kind=gate_kind,
                checkpoint_tma=bool(checkpoint_every_n_tokens and use_direct_m128_n16),
                **({"checkpoint_dtype_is_fp32": True} if fp32_carrier else {}),
                **({"fp32_state_carrier": True} if fp32_state_carrier else {}),
                **(
                    {"checkpoint_accumulate": True}
                    if self._checkpoint_accumulate
                    else {}
                ),
                pair_packed_beta=use_pair_packed_beta,
                scalar_beta=use_scalar_beta,
                active_beta_f32=self._active_beta_f32,
                logical_page64=use_n32_logical_page64,
                early_n32_state_pack=use_early_n32_state_pack,
                generic_register_inverse=use_n32_register_inverse,
                n32_prediction_first=use_n32_prediction_first,
                tensor_state_decay=use_n32_tensor_state_decay,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                n32_ft_slab=self._n32_ft_slab and (not use_direct_m128_n16),
                affine_operator_export=self._affine_operator_export,
                pdl_wait_initial_state_f32=self._pdl_wait_initial_state_f32,
                pdl_publish_final_state=self._pdl_publish_final_state,
                affine_main_indexed_initial=self._affine_main_indexed_initial,
                affine_main_indexed_initial_bf16=False,
                backend=backend,
                n16_short_four_stage=False,
            )
            self.schedule = (
                (
                    "fused_checkpoint_tma_direct_m128_n16"
                    if checkpoint_every_n_tokens
                    else "fused_h12_direct_m128_n16"
                    if num_heads == 12
                    else "fused_direct_m128_n16"
                )
                if use_direct_m128_n16
                else "fused_checkpoint_direct_m128_page64_n32x2"
                if use_n32_logical_page64 and checkpoint_every_n_tokens
                else "fused_h12_direct_m128_page64_n32x2"
                if use_n32_logical_page64
                else "fused_unbounded_softplus_direct_m128"
                if unbounded_softplus
                else "fused_checkpoint_direct_m128_n32"
                if checkpoint_every_n_tokens
                else "fused_prediction_first_direct_m128"
                if use_n32_prediction_first and (not use_n32_tensor_state_decay)
                else "fused_tensor_state_decay_direct_m128"
                if use_n32_tensor_state_decay
                else "fused_direct_m128"
            )
            if fp32_checkpoints:
                self.schedule += "_fp32_checkpoints"
            elif fp32_carrier or fp32_state_carrier:
                self.schedule += "_fp32_carrier"
        if use_tf32_owner_helper:
            n32_value_rows = (
                64 if 2 * SMALL_BH_GROUP_SIZE * total_tasks <= sm_count else 128
            )
            self.module = self._build_body(
                partial(_factory, "compiled_tf32_fused_n32"),
                owner_helpers=7,
                unbounded_softplus=unbounded_softplus,
                value_rows=n32_value_rows,
                prep_stages=3,
                active_beta_f32=self._active_beta_f32,
                state_dtype_is_fp32=True,
                round_tf32_operands=False,
                write_checkpoints=bool(checkpoint_every_n_tokens),
            )
            self.schedule = f"fused_tf32_small_bh_m{n32_value_rows}_owner7helper_ring21"
            if unbounded_softplus:
                self.schedule += "_unbounded_softplus"
        elif use_tf32_direct_n32:
            direct_operands = not (
                self._pdl_wait_initial_state_f32
                or self._pdl_publish_final_state
                or self._affine_main_indexed_initial
            )
            compact_state = (
                gpu_arch == "sm_103a"
                and direct_operands
                and (num_seqs * num_heads > sm_count)
                and (max_seq_len <= 256)
            )
            if (
                compact_state
                and (not checkpoint_every_n_tokens)
                and (128 < max_seq_len <= 256)
                and (
                    (min(resolved_sequence_lengths) + 31) // 32
                    == (max_seq_len + 31) // 32
                )
                and (5 * sm_count <= 2 * total_tasks)
                and (total_tasks <= 3 * sm_count)
            ):
                compact_state = False
            n32_prepare_stages = 1 if compact_state or max_seq_len <= 64 else 3
            if (
                gpu_arch == "sm_103a"
                and direct_operands
                and (2 * total_tasks <= sm_count)
                and (max_seq_len >= 32)
            ):
                n32_value_rows = 64
            n32_checkpoint_tma = bool(
                compact_state
                and checkpoint_every_n_tokens
                and (state_checkpoints is not None)
                and (state_checkpoints.data_ptr() % 16 == 0)
            )
            self.module = self._build_body(
                partial(_factory, "compiled_tf32_fused_n32"),
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                active_beta_f32=self._active_beta_f32,
                unbounded_softplus=unbounded_softplus,
                write_checkpoints=bool(checkpoint_every_n_tokens),
                prep_stages=n32_prepare_stages,
                round_tf32_operands=not direct_operands,
                compact_state=compact_state,
                value_rows=n32_value_rows,
                checkpoint_tma=n32_checkpoint_tma,
                pdl_wait_initial_state_f32=self._pdl_wait_initial_state_f32,
                pdl_publish_final_state=self._pdl_publish_final_state,
                affine_main_indexed_initial=self._affine_main_indexed_initial,
                affine_main_indexed_initial_bf16=False,
                affine_factor_cache=_affine_factor_cache_mode,
                affine_map_only=_affine_map_only,
                affine_map_output=_affine_map_output,
            )
            self.schedule = (
                f"fused_tf32_m{n32_value_rows}_local_factors_s{n32_prepare_stages}_n32"
            )
            if unbounded_softplus:
                self.schedule += "_unbounded_softplus"
            if compact_state:
                self.schedule += "_compact_state"
            if n32_checkpoint_tma:
                self.schedule += "_checkpoint_tma"
            if self._n32_ft_slab:
                self.schedule += "_slab"
            if self._pdl_wait_initial_state_f32 or self._pdl_publish_final_state:
                self.schedule += "_pdl"
        elif use_independent_dvsplit:
            pair_share_rows = (
                M64_PAIR_SHARE_MIN_SEQ_LEN_GB300
                if gpu_arch == "sm_103a" and sm_count == 152
                else M64_PAIR_SHARE_MIN_SEQ_LEN
            )
            pair_share_min_seq_len = next(
                (
                    min_seq_len
                    for most_tasks, min_seq_len in pair_share_rows
                    if num_heads * total_tokens <= most_tasks * max_seq_len
                ),
                None,
            )
            pair_share = (
                pair_share_min_seq_len is not None
                and max_seq_len >= pair_share_min_seq_len
                and total_tasks <= _m64_pair_cluster_capacity(q.device)
            )
            self.module = self._build_body(
                partial(
                    _m64_pair_share_factory if pair_share else _factory,
                    "compiled_bf16_fused_m64",
                ),
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                active_beta_f32=self._active_beta_f32,
            )
            self.schedule = (
                "fused_active_beta_checkpoint_dvsplit_m64"
                if self._active_beta_f32
                else "fused_m64_independent_dvsplit_fp32_state"
                if self._state_dtype_is_fp32
                else "fused_m64_independent_dvsplit"
            )
            if getattr(self.module, "name", None) in (
                _kda_registry._RENDERED_PAIR_SHARE_ROUTES.get(gpu_arch, {}).values()
            ):
                self.schedule += "_pair_share"
        self.route = route
        self.gate_kind = gate_kind.value
        self.gpu_arch = gpu_arch
        self.sm_count = sm_count
        self.beta_transport = (
            "tma"
            if use_independent_dvsplit
            and compute_dtype == "bf16"
            and (not self._active_beta_f32)
            else "scalar"
        )
        self.active_beta_f32 = self._active_beta_f32
        uses_logical_page64 = use_n32_logical_page64 or (
            use_independent_dvsplit and self._active_beta_f32
        )
        self.logical_page_tokens = 64 if uses_logical_page64 else None
        self.physical_subtile_tokens = 32 if uses_logical_page64 else None
        if use_independent_dvsplit or use_tf32_direct_m128 or use_tf32_owner_helper:
            self.logical_page_tokens = checkpoint_every_n_tokens or None
            self.physical_subtile_tokens = (
                16
                if compute_dtype == "tf32"
                and (not (use_tf32_direct_n32 or use_tf32_owner_helper))
                else 32
            )
        self.early_n32_state_pack = use_early_n32_state_pack
        self.ir = self.module.schedule
        num_work_bins = 0
        if getattr(self.module, "work_bins", False):
            # A body built with work bins reads its bin table through seq_order
            # and runs one CTA per bin.
            if route not in {BF16_ROUTE_DIRECT_M128, BF16_ROUTE_DIRECT_M128_N16}:
                raise NotImplementedError(
                    "work bins are built for the direct M128 N32 and N16 bodies only"
                )
            work_bin_table, num_work_bins = _make_direct_work_bins(
                tuple(offsets[index + 1] - offsets[index] for index in range(num_seqs)),
                num_heads=num_heads,
                worker_count=sm_count,
                chunk_tokens=(
                    BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK
                ),
            )
            work_bin_upload = _upload_int32_batch(
                q.device, {"seq_order": list(work_bin_table)}
            )
            self._work_bin_host = work_bin_upload["_host_pinned"]
            seq_order = work_bin_upload["seq_order"]
        if use_tf32_owner_helper:
            OWNER_RING_STAGES = 21
            STAGE_BYTES = 74752
            mid_state = torch.empty(
                total_tasks
                * (HEAD_DIM // n32_value_rows)
                * OWNER_RING_STAGES
                * (STAGE_BYTES // 4),
                dtype=torch.float32,
                device=q.device,
            )
            mid_state_ready = torch.zeros(
                total_tasks
                * (HEAD_DIM // n32_value_rows)
                * (2 * OWNER_RING_STAGES + 1),
                dtype=torch.uint32,
                device=q.device,
            )
        self._keepalive = (
            q,
            k,
            v,
            g,
            beta,
            out,
            A_log,
            dt_bias,
            initial_state,
            final_state,
            cu_seqlens,
            seq_order,
            task_ids,
            task_offsets,
            task_token_starts,
            task_token_counts,
            task_state_sources,
            task_state_destinations,
            mid_state,
            mid_state_ready,
            beta_tma,
            empty_state,
            empty_i32,
            empty_i64,
            empty_chunk_offsets,
            empty_f32,
            empty_u32,
            state_indices,
            state_checkpoints,
            checkpoint_cu_starts,
            beta_pointer,
            initial_state_pointer,
            final_state_pointer,
            bt16_cu_chunks,
            bt16_chunk_to_seq,
            bt16_qd,
            bt16_kd,
            bt16_w,
            bt16_qk,
            bt16_diag,
        )
        self.grid = (
            BT16_VALUE_SPLITS * total_tasks
            if use_bt16_prepare_chain
            else SMALL_BH_GROUP_SIZE * total_tasks
            if use_small_bh_owner_helper
            else INDEPENDENT_DVSPLIT_CTAS * total_tasks
            if use_independent_dvsplit
            else total_tasks,
            1,
            1,
        )
        if n32_value_rows == 64:
            self.grid = (2 * self.grid[0], 1, 1)
        if num_work_bins:
            if self.grid != (total_tasks, 1, 1):
                raise NotImplementedError(
                    "work bins replace the one-CTA-per-task direct grid only"
                )
            self.grid = (num_work_bins, 1, 1)
            if getattr(self.module, "name", None) in (
                _kda_registry._RENDERED_WORK_BIN_TURN_ROUTES.get(gpu_arch, {}).values()
            ):
                self.schedule += "_turnover"
        self.owner_helpers = (
            owner_helpers if getattr(self.module, "owner_helper", False) else 0
        )
        self.owner_helper_pool = (
            owner_helper_pool
            if self.owner_helpers and getattr(self.module, "owner_helper_pool", False)
            else 0
        )
        if self.owner_helper_pool:
            self.grid = (total_tasks + num_heads * self.owner_helper_pool, 1, 1)
            self.schedule += (
                f"_owner{self.owner_helpers}helper_pool{self.owner_helper_pool}"
            )
        elif self.owner_helpers:
            self.grid = (total_tasks * (1 + self.owner_helpers), 1, 1)
            self.schedule += f"_owner{self.owner_helpers}helper"
        self.prepare_grid = (bt16_prepare_total_ctas, 1, 1)
        qkv_carriers = self._qkv_carrier_names.get(id(self.module), ())
        self.args = {
            "q": _qkv_raw_pointer_arg(q_flat, "q", qkv_carriers),
            "q_tma": q_flat,
            "k": _qkv_raw_pointer_arg(k_flat, "k", qkv_carriers),
            "k_tma": k_flat,
            "v": _qkv_raw_pointer_arg(v_flat, "v", qkv_carriers),
            "v_tma": v_flat,
            "g": g_flat,
            "g_tma": g_flat,
            "beta": beta_abi_source,
            "beta_tma": beta_tma,
            "A_log": A_log,
            "dt_bias": dt_bias,
            "cu_seqlens": cu_seqlens,
            "initial_state": initial_state_pointer,
            "out": out_flat,
            "out_tma": out_flat,
            "final_state": final_state_pointer,
            "num_heads": num_heads,
            "use_initial_state": int(initial_state is not None),
            "store_final_state": int(final_state is not None),
            "scale": float(scale),
            "lower_bound": 0.0 if unbounded_softplus else float(lower_bound),
        }
        self.args["seq_order"] = seq_order
        if use_independent_dvsplit:
            self.args.update(
                state_indices_addr=state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                state_slot_stride=state_slot_stride,
                use_state_indices=int(state_indices is not None),
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
            self.args.update(
                state_checkpoints_addr=state_checkpoints.data_ptr()
                if state_checkpoints is not None
                else empty_state.data_ptr(),
                checkpoint_cu_starts_addr=checkpoint_cu_starts.data_ptr()
                if checkpoint_cu_starts is not None
                else empty_i64.data_ptr(),
                beta_active_out=_ffi_raw_pointer_carrier(beta_flat)
                if self._active_beta_f32
                else empty_f32,
                beta_token_stride=beta_flat.stride(0),
                g_token_stride=g_flat.stride(0),
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            )
        else:
            self.args.update(
                state_indices_addr=state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                state_checkpoints_addr=state_checkpoints.data_ptr()
                if state_checkpoints is not None
                else empty_state.data_ptr(),
                checkpoint_cu_starts_addr=checkpoint_cu_starts.data_ptr()
                if checkpoint_cu_starts is not None
                else empty_i64.data_ptr(),
                beta_token_stride=beta_flat.stride(0),
                state_slot_stride=state_slot_stride,
                use_state_indices=int(state_indices is not None),
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            )
        if uses_default_fused_m128:
            op_export = (
                _allocate_operator_export(offsets, num_heads, q.device)
                if self._affine_operator_export
                else None
            )
            self.operator_export = op_export
            self.args.update(
                g_token_stride=g_flat.stride(0),
                cu_chunk_offsets=(
                    op_export["cu_chunk_offsets"] if op_export else empty_chunk_offsets
                ),
                chunk_state=empty_state,
                state_checkpoint_needed=empty_u32,
                tape_qd=(op_export["qd_raw"] if op_export else empty_state),
                tape_kd=(op_export["kd_raw"] if op_export else empty_state),
                tape_kr=(op_export["ft_raw"] if op_export else empty_state),
                tape_j=(op_export["inv_raw"] if op_export else empty_state),
                tape_restore_factor=(
                    op_export["decay_beta"] if op_export else empty_f32
                ),
                tape_e=empty_state,
                tape_x=empty_state,
                tape_r=empty_state,
                norm_inv_out=empty_f32,
                decay_out=empty_state,
                beta_active_out=_ffi_raw_pointer_carrier(beta_flat)
                if self._active_beta_f32
                else empty_f32,
                initial_state_f32=empty_f32,
                final_state_f32=empty_f32,
                zero_workspace=empty_u32,
                zero_words=0,
                num_sequences=num_seqs,
                # The rank-three panel descriptor is typed like the rows the
                # module carries (FP32 carrier -> FP32 rows).
                state_checkpoints_tma=state_checkpoints
                if state_checkpoints is not None
                else torch.empty(
                    (1, 1, HEAD_DIM, HEAD_DIM),
                    dtype=torch.float32,
                    device=q.device,
                )
                if fp32_carrier
                else empty_checkpoint_tma,
            )
        if uses_default_fused_m128 and self._state_dtype_is_fp32:
            self.args.update(
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
        if self.owner_helpers:
            # The packet ring (chunk_state) and its two word arrays: the ready
            # words (zero_workspace) and the consumed words
            # (state_checkpoint_needed). The program leaves every word zero when
            # it exits, so one zeroed allocation serves every launch of the plan.
            # The pooled program keeps the per-task ring of the same helpers.
            slots = total_tasks * OWNER_HELPER_SLOTS_PER_HELPER * self.owner_helpers
            owner_helper_words = torch.zeros(
                2 * slots, dtype=torch.uint32, device=q.device
            )
            self.args.update(
                chunk_state=torch.empty(
                    slots * OWNER_HELPER_SLOT_ELEMENTS,
                    dtype=torch.bfloat16,
                    device=q.device,
                ),
                zero_workspace=owner_helper_words[:slots],
                state_checkpoint_needed=owner_helper_words[slots:],
            )
        self.prepare_args: dict[str, Any] = {}
        if use_bt16_prepare_chain:
            prepare_carriers = self._qkv_carrier_names.get(id(self.prepare_module), ())
            self.prepare_args = {
                "q": _qkv_raw_pointer_arg(q_flat, "q", prepare_carriers),
                "q_tma": q,
                "k": _qkv_raw_pointer_arg(k_flat, "k", prepare_carriers),
                "k_tma": k,
                "raw_gate": g_pointer,
                "raw_gate_tma": g_flat,
                "beta_logits": beta_pointer,
                "beta_active_f32": _ffi_raw_pointer_carrier(beta_flat)
                if self._active_beta_f32
                else empty_f32,
                "beta_logits_tma": beta_tma,
                "a_log": A_log,
                "dt_bias": dt_bias,
                "cu_seqlens": cu_seqlens,
                "cu_chunks": bt16_cu_chunks,
                "chunk_to_seq": bt16_chunk_to_seq,
                "ws_qd": bt16_qd,
                "ws_qd_tma": bt16_qd,
                "ws_kd": bt16_kd,
                "ws_kd_tma": bt16_kd,
                "ws_w": bt16_w,
                "ws_w_tma": bt16_w,
                "ws_qk_t": bt16_qk,
                "ws_diag": bt16_diag,
                "total_chunks": bt16_total_chunks,
                "num_heads": num_heads,
                "gate_lower_bound": float(lower_bound),
                "beta_token_stride": beta_flat.stride(0),
            }
            chain_carriers = self._qkv_carrier_names.get(id(self.module), ())
            self.args = {
                "ws_qd": bt16_qd,
                "ws_qd_tma": bt16_qd,
                "ws_kd": bt16_kd,
                "ws_kd_tma": bt16_kd,
                "ws_w": bt16_w,
                "ws_w_tma": bt16_w,
                "ws_qk": bt16_qk,
                "ws_qk_tma": bt16_qk,
                "ws_diag": bt16_diag,
                "ws_diag_tma": bt16_diag,
                "v": _qkv_raw_pointer_arg(v_flat, "v", chain_carriers),
                "v_tma": v,
                "cu_seqlens": cu_seqlens,
                "cu_chunks": bt16_cu_chunks,
                "seq_order": seq_order,
                "initial_state": initial_state_pointer,
                "out": out_flat,
                "out_tma": out,
                "final_state": final_state_pointer,
                "num_heads": num_heads,
                "use_initial_state": int(initial_state is not None),
                "store_final_state": int(final_state is not None),
                "scale": float(scale),
                "state_indices_addr": state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                "state_slot_stride": state_slot_stride,
                "use_state_indices": int(state_indices is not None),
                "initial_state_f32": initial_state_f32_pointer,
                "final_state_f32": final_state_f32_pointer,
                "state_checkpoints": state_checkpoints
                if state_checkpoints is not None
                else empty_state,
                "checkpoint_cu_starts": checkpoint_cu_starts
                if checkpoint_cu_starts is not None
                else empty_i64,
                "checkpoint_every_n_tokens": checkpoint_every_n_tokens,
            }
        if (
            use_independent_dvsplit or use_tf32_direct_m128 or use_tf32_owner_helper
        ) and compute_dtype == "tf32":
            qkv_carriers = self._qkv_carrier_names.get(id(self.module), ())
            self.args = {
                "q": _qkv_raw_pointer_arg(q_flat, "q", qkv_carriers),
                "q_tma": q,
                "k": _qkv_raw_pointer_arg(k_flat, "k", qkv_carriers),
                "k_tma": k,
                "raw_gate": g_pointer,
                "raw_gate_tma": g_flat,
                "beta_logits": beta_pointer,
                "beta_logits_tma": beta_tma,
                "beta_active_f32": _ffi_raw_pointer_carrier(beta_flat)
                if self._active_beta_f32
                else empty_f32,
                "a_log": A_log,
                "dt_bias": dt_bias,
                "cu_seqlens": cu_seqlens,
                "seq_order": seq_order,
                "v": _qkv_raw_pointer_arg(v_flat, "v", qkv_carriers),
                "out": out_flat,
                "initial_state": initial_state_pointer,
                "final_state": final_state_pointer,
                "initial_state_f32": initial_state_f32_pointer,
                "final_state_f32": final_state_f32_pointer,
                "state_indices_addr": state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                "state_slot_stride": state_slot_stride,
                "use_state_indices": int(state_indices is not None),
                "use_initial_state": int(initial_state is not None),
                "store_final_state": int(final_state is not None),
                "state_checkpoints": state_checkpoints
                if state_checkpoints is not None
                else empty_state,
                "checkpoint_cu_starts": checkpoint_cu_starts
                if checkpoint_cu_starts is not None
                else empty_i64,
                "checkpoint_every_n_tokens": checkpoint_every_n_tokens,
                "scale": float(scale),
                "num_heads": num_heads,
                "gate_lower_bound": 0.0 if unbounded_softplus else float(lower_bound),
                "beta_token_stride": beta_flat.stride(0),
            }
        if (use_tf32_direct_n32 or use_tf32_owner_helper) and compute_dtype == "tf32":
            self.args["state_checkpoints_tma"] = (
                state_checkpoints if n32_checkpoint_tma else empty_checkpoint_tma
            )
            self.args["v_tma"] = v
            self.args["affine_cache_token_offset"] = _affine_cache_token_offset
            self.args["affine_cache_part_offset"] = _affine_cache_part_offset
            self.args["owner_packet_tma"] = (
                _affine_factor_cache
                if _affine_factor_cache_mode
                else mid_state.view(-1, 128)
                if use_tf32_owner_helper
                else dt_bias
            )
            self.args["task_ids"] = task_ids
            self.args["task_offsets"] = task_offsets
            self.args.update(
                task_token_starts=task_token_starts,
                task_token_counts=task_token_counts,
                task_state_sources=task_state_sources,
                task_state_destinations=task_state_destinations,
                mid_state_f32=mid_state if use_tf32_owner_helper else empty_f32,
                mid_state_ready=mid_state_ready,
            )
            self.args["owner_packet_tail_tma"] = self.args["owner_packet_tma"]
            self.args["map_output_f32"] = empty_f32
        self._beta_tma_source = beta_flat
        self._beta_tma_valid = beta_tma_valid
        self._token_storage_refreshes = tuple(
            (refresh for refresh in (g_refresh, beta_refresh) if refresh is not None)
        )
        self._launch_device = q.device
        self._use_cuda_graph = use_bt16_prepare_chain or beta_tma_valid is not None
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._cuda_graph_warmed = False

    _descriptors_stale = False
    # Every operation of one launch() in order, as (callable, positional
    # arguments) with the arguments final; built once per binding and dropped
    # by a plan-cache rebind (_invalidate_launch_records).
    _launch_records: tuple | None = None

    def _prepare_descriptors_in_stream(self) -> None:
        """Encode and upload every TMA descriptor on the current stream."""
        if self.prepare_module is not None:
            self.prepare_module.prepare(grid=self.prepare_grid, **self.prepare_args)
        self.module.prepare(grid=self.grid, **self.args)
        self._descriptors_stale = False

    def _build_launch_records(self) -> list:
        """The staging copies and kernel launches of one launch(), in order."""
        records = [
            (destination.copy_, (source,))
            for destination, source in self._token_storage_refreshes
        ]
        if self._beta_tma_valid is not None:
            records.append((self._beta_tma_valid.copy_, (self._beta_tma_source,)))
        if self.prepare_module is not None:
            records.append(
                _prepared_kernel_record(
                    self.prepare_module, self.prepare_grid, self.prepare_args
                )
            )
        records.append(_prepared_kernel_record(self.module, self.grid, self.args))
        return records

    def _launch_in_stream(self) -> None:
        """Launch inside an already entered FFI/Torch stream context."""
        if self._descriptors_stale:
            # A plan-cache rebind moved a descriptor source; re-encode in
            # the same stream context as the launch it precedes.
            self._prepare_descriptors_in_stream()
        records = self._launch_records
        if records is None:
            records = self._launch_records = tuple(self._build_launch_records())
        _run_launch_records(records)

    def _launch_uncaptured(self) -> None:
        with _ffi_stream_context(self._launch_device):
            self._launch_in_stream()

    def launch(self) -> None:
        self._launch_uncaptured()

    def close(self) -> None:
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        return None


class FlashKDABlackwellBF16DirectM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """Prepare the compatible direct-M128 seed without auto-route substitution."""

    _force_direct_m128 = True


class FlashKDABlackwellFP32SmallBHLaunch(FlashKDABlackwellBF16FusedLaunch):
    """Reuse owner/helper compute with FP32 indexed state-pool I/O."""

    _force_small_bh_owner_helper = True
    _state_dtype_is_fp32 = True


class FlashKDABlackwellFP32BT16Launch(FlashKDABlackwellBF16FusedLaunch):
    """Reuse the BT16 factor/chain family with FP32 indexed state I/O."""

    _force_bt16_prepare_chain = True
    _state_dtype_is_fp32 = True


class FlashKDABlackwellFP32ActiveBetaBT16Launch(FlashKDABlackwellFP32BT16Launch):
    """Consume active FP32 beta directly in either BT16 compute family."""

    _active_beta_f32 = True


class FlashKDABlackwellFP32FusedM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """Reuse the direct-M128 compute schedule with FP32 indexed state I/O."""

    _state_dtype_is_fp32 = True


class FlashKDABlackwellFP32ActiveBetaFusedM128Launch(
    FlashKDABlackwellFP32FusedM128Launch
):
    """Consume active-FP32 beta through the qualified M128/M64 schedules."""

    _active_beta_f32 = True


class FlashKDABlackwellBF16M64Launch(FlashKDABlackwellBF16FusedLaunch):
    """Explicit fused M64 with BF16 state and either compute precision."""

    _force_independent_dvsplit = True


class FlashKDABlackwellFP32M64Launch(FlashKDABlackwellBF16M64Launch):
    """Explicit fused M64 with FP32 state and either compute precision."""

    _state_dtype_is_fp32 = True


class FlashKDABlackwellFP32ActiveBetaM64Launch(FlashKDABlackwellFP32M64Launch):
    """Fused M64 with active FP32 beta and FP32 external state."""

    _active_beta_f32 = True


class FlashKDABlackwellFP32DirectM128Launch(FlashKDABlackwellFP32FusedM128Launch):
    """Force the variable-shape direct M128 body with FP32 state I/O."""

    _force_direct_m128 = True


class FlashKDABlackwellFP32SlabM128Launch(FlashKDABlackwellFP32DirectM128Launch):
    """PR4779's split final-transform issue on the direct N32 schedule."""

    _force_direct_m128_n32 = True
    _n32_ft_slab = True


class FlashKDABlackwellFP32SlabM128PDLProducerLaunch(
    FlashKDABlackwellFP32SlabM128Launch
):
    """Publish the split-state workspace before the output epilogue drains."""

    _pdl_publish_final_state = True


class FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch(
    FlashKDABlackwellFP32SlabM128PDLProducerLaunch
):
    """Read part zero directly from the caller's indexed FP32 state pool."""

    _affine_main_indexed_initial = True


class FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportLaunch(
    FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch
):
    """Main windows that also export their chunk operators for the apply kernel (§4e)."""

    _affine_operator_export = True


class FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportFoldLaunch(
    FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportLaunch
):
    """Main windows that also write the block pair maps of the tail windows.

    The factory row is the operator-export main's; the body is that module's
    pair-map fold build (``_pair_map_fold_factory``), whose ``tape_e`` /
    ``norm_inv_out`` slots take the composite's pair / dpair buffers.  Raises
    ``_PairMapFoldUnavailable``, before any module is built, when the module
    has none.
    """

    @staticmethod
    def _build_kda_module(factory, *args, **kwargs):
        # Fail closed: any other factory (a wrapper included) has no fold
        # build here, and the composite then keeps the pair-map producer.
        if not _is_fused_m128_row(factory) or (
            _pair_map_fold_program(
                *factory.args, *args, **{**factory.keywords, **kwargs}
            )
            is None
        ):
            raise _PairMapFoldUnavailable
        factory = partial(_pair_map_fold_factory, *factory.args, **factory.keywords)
        return _build_kda_module(factory, *args, **kwargs)


class FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerMapFoldLaunch(
    FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch
):
    """Chain-route main windows that also write the window maps of the tail windows.

    The factory row is the indexed-initial main's; the body is that module's
    map-fold build (``_chain_map_fold_factory``), whose ``tape_e`` slot takes
    the composite's BF16 map states.  Raises ``_ChainMapFoldUnavailable``,
    before any module is built, when the module has none.
    """

    @staticmethod
    def _build_kda_module(factory, *args, **kwargs):
        # Fail closed: any other factory (a wrapper included) has no map-fold
        # build here, and the composite then keeps the map pass.
        if not _is_fused_m128_row(factory) or (
            _chain_map_fold_program(
                *factory.args, *args, **{**factory.keywords, **kwargs}
            )
            is None
        ):
            raise _ChainMapFoldUnavailable
        factory = partial(_chain_map_fold_factory, *factory.args, **factory.keywords)
        return _build_kda_module(factory, *args, **kwargs)


def _affine_lean_map_enabled() -> bool:
    """Kernel round 2: build the prefix maps with the pair-map producer + prefix chain instead of the map pass.

    Default on the apply route; ``CAKE_KDA_AFFINE_LEAN_MAP=0`` restores the map
    pass (the control for the paired comparison).  The map pass stays
    constructed for the plan-cache ABI and the reference comparison.
    """

    import os

    return os.environ.get("CAKE_KDA_AFFINE_LEAN_MAP", "1") != "0"


def _affine_pair_map_fold_enabled() -> bool:
    """Kernel round 7: the lean-map apply composite's main body writes the block pair maps itself.

    Default on where the loader names the main module's fold build
    (``_RENDERED_FOLD_ROUTES``); ``CAKE_KDA_AFFINE_PAIR_FOLD=0`` keeps the
    pair-map producer launch (the control for the paired comparison).
    """
    import os

    return os.environ.get("CAKE_KDA_AFFINE_PAIR_FOLD", "1") != "0"


def _pair_map_fold_routed(device) -> bool:
    """The loader names a pair-map fold build for ``device``."""
    import torch

    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(
        torch.cuda.get_device_capability(device)
    )
    return bool(_kda_registry._RENDERED_FOLD_ROUTES.get(arch))


class _PairMapFoldUnavailable(Exception):
    """The composite's main module has no pair-map fold build on this device."""


def _pair_map_fold_program(name, *args, **kwargs):
    """The pair-map fold build of the factory row's main module, or ``None``.

    ``None`` when the row has no module on this device, when the head-count
    routes take the module elsewhere at ``route_heads``, when the call asks
    for a work-binned or owner/helper build the loader names for the module
    (a fold build has neither), or when the loader names no fold build of the
    module (``_RENDERED_FOLD_ROUTES``).  The routing keywords are not part of
    the row's key (a pooled owner/helper build is an owner/helper build).
    """
    return _main_fold_program("_RENDERED_FOLD_ROUTES", name, *args, **kwargs)


def _main_fold_program(
    table,
    name,
    *args,
    route_heads=None,
    route_work_bins=False,
    owner_helper=False,
    owner_helper_pool=False,
    **kwargs,
):
    """The build the loader table ``table`` names for the factory row's module.

    ``None`` under the conditions of ``_pair_map_fold_program``, for the
    fold table ``table`` (``_RENDERED_FOLD_ROUTES`` or
    ``_RENDERED_MAP_FOLD_ROUTES``).
    """
    del owner_helper_pool
    arch = detect_gpu_arch()
    rows = _kda_registry.FACTORIES.get(arch, {}).get(name, {})
    module = rows.get(_kda_registry._key(args, kwargs))
    if module is None:
        return None
    heads = getattr(_kda_registry, "_RENDERED_HEAD_ROUTES", {})
    if route_heads in heads.get(arch, {}).get(module, {}):
        return None
    if route_work_bins and module in _kda_registry._RENDERED_WORK_BIN_ROUTES.get(
        arch, {}
    ):
        return None
    if owner_helper and module in _kda_registry._RENDERED_OWNER_HELPER_ROUTES.get(
        arch, {}
    ):
        return None
    return getattr(_kda_registry, table, {}).get(arch, {}).get(module)


def _pair_map_fold_factory(name, *args, **kwargs):
    """The pair-map fold build of the factory row's main module.

    ``_factory`` resolves the row to its shipped module and that module
    through the routes; this takes the module's ``_RENDERED_FOLD_ROUTES``
    build instead (same argument plan; the build also writes the block pair
    maps of the tail windows through ``tape_e`` / ``norm_inv_out``).
    """
    program = _pair_map_fold_program(name, *args, **kwargs)
    if program is None:
        raise _PairMapFoldUnavailable
    return _kda_registry.PreparedKernel(program, detect_gpu_arch())


def _chain_map_fold_enabled() -> bool:
    """Kernel round 8: the chain composite's main body writes the window maps itself.

    Default on where the loader names the main module's map-fold build
    (``_RENDERED_MAP_FOLD_ROUTES``); ``CAKE_KDA_CHAIN_MAP_FOLD=0`` keeps the
    map launch (the control for the paired comparison).
    """
    import os

    return os.environ.get("CAKE_KDA_CHAIN_MAP_FOLD", "1") != "0"


def _chain_map_fold_routed(device) -> bool:
    """The loader names a chain map-fold build for ``device``."""
    import torch

    arch = {(10, 0): "sm_100a", (10, 3): "sm_103a"}.get(
        torch.cuda.get_device_capability(device)
    )
    return bool(getattr(_kda_registry, "_RENDERED_MAP_FOLD_ROUTES", {}).get(arch))


class _ChainMapFoldUnavailable(Exception):
    """The chain composite's main module has no map-fold build on this device."""


def _chain_map_fold_program(name, *args, **kwargs):
    """The map-fold build of the factory row's main module, or ``None``.

    ``None`` under the conditions of ``_pair_map_fold_program``, for the
    loader's ``_RENDERED_MAP_FOLD_ROUTES``.
    """
    return _main_fold_program("_RENDERED_MAP_FOLD_ROUTES", name, *args, **kwargs)


def _chain_map_fold_factory(name, *args, **kwargs):
    """The map-fold build of the factory row's main module.

    Same argument plan as the routed program; the build also writes the BF16
    state of every tail window's map (identity initial state, zero values,
    the map pass's operation order) through ``tape_e``, window ``p`` at
    ``[p - 1, head]``.
    """
    program = _chain_map_fold_program(name, *args, **kwargs)
    if program is None:
        raise _ChainMapFoldUnavailable
    return _kda_registry.PreparedKernel(program, detect_gpu_arch())


def _affine_apply_route_enabled() -> bool:
    """Kernel round 2: evaluate the correction with the fused apply kernel instead of a chain.

    ``CAKE_KDA_AFFINE_APPLY=0`` keeps the correction chain (A/B and bitwise
    control).  The route needs BF16 compute and either no checkpoint rows or
    FP32 rows merged in place.
    """
    import os

    return os.environ.get("CAKE_KDA_AFFINE_APPLY", "1") != "0"


class FlashKDABlackwellBF16DirectM128PDLBridgeLaunch(
    FlashKDABlackwellBF16DirectM128Launch
):
    """Acquire the main workspace and publish the BF16 affine map."""

    _force_direct_m128_n32 = True
    _n32_ft_slab = True
    _pdl_wait_initial_state_f32 = True
    _pdl_publish_final_state = True


class FlashKDABlackwellFP32SlabM128PDLConsumerLaunch(
    FlashKDABlackwellFP32SlabM128Launch
):
    """Acquire the scanned FP32 carry before starting correction work."""

    _pdl_wait_initial_state_f32 = True


class FlashKDABlackwellFP32SlabM128PDLConsumerAccumulateLaunch(
    FlashKDABlackwellFP32SlabM128PDLConsumerLaunch
):
    """Correction windows that add their FP32 rows onto the main pass's rows in place."""

    _checkpoint_accumulate = True


class FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch(
    FlashKDABlackwellFP32SlabM128PDLConsumerLaunch
):
    """Correction windows that merge themselves into the affine composite.

    The factory row is the plain correction's; the body is that module's
    merging build (``_correction_merge_factory``), which accumulates its rows
    by build.  Raises ``_CorrectionMergeUnavailable`` when the module has none.
    """

    @staticmethod
    def _build_kda_module(factory, *args, **kwargs):
        # Fail closed: any other factory (a wrapper included) has no merging
        # build here, and the composite then keeps the epilogue.
        if not _is_fused_m128_row(factory):
            raise _CorrectionMergeUnavailable
        factory = partial(_correction_merge_factory, *factory.args, **factory.keywords)
        return _build_kda_module(factory, *args, **kwargs)


def _allocate_operator_export(offsets, num_heads: int, device):
    """Per-chunk-head operator slabs for the affine operator export (kernel round 2).

    Raw shared-memory images as the kernel copies them: ``qd_raw`` / ``kd_raw``
    are 128B-swizzled K-major (32 tokens x 128) BF16 tiles, ``ft_raw`` the
    128B-swizzled MN-major [Kr | MQK] slab (32 x 192), ``inv_raw`` the
    32B-swizzled (32 x 32) inverse, ``decay_beta`` the per-column state decay
    (128 f32) followed by the prep restore-factor block (129 restore factors,
    32 sigmoid beta at word offset 128 + 129, gate rate, pad).  Slab ``s`` belongs to
    chunk-head ``(cu_chunk_offsets[seq] + chunk) * num_heads + head``.
    """
    import torch

    OPERATOR_EXPORT_VEC_WORDS = (
        292  # HEAD_DIM + OPERATOR_EXPORT_RF_BLOCK_BYTES // 4 (source fused M128 kernel)
    )

    chunk = BF16_M128_CHUNK
    counts = [
        (end - start + chunk - 1) // chunk
        for start, end in zip(offsets, offsets[1:], strict=False)
    ]
    cu = [0]
    for count in counts:
        cu.append(cu[-1] + count)
    slabs = max(cu[-1] * num_heads, 1)
    bf16 = torch.bfloat16
    return {
        "chunk_counts": counts,
        "cu_chunk_offsets": torch.tensor(cu, dtype=torch.int64, device=device),
        "qd_raw": torch.empty(slabs, chunk, HEAD_DIM, dtype=bf16, device=device),
        "kd_raw": torch.empty(slabs, chunk, HEAD_DIM, dtype=bf16, device=device),
        "ft_raw": torch.empty(slabs, chunk, 192, dtype=bf16, device=device),
        "inv_raw": torch.empty(slabs, chunk, chunk, dtype=bf16, device=device),
        "decay_beta": torch.empty(
            slabs, OPERATOR_EXPORT_VEC_WORDS, dtype=torch.float32, device=device
        ),
    }


def _affine_rows_in_place_enabled() -> bool:
    """FP32 affine checkpoint rows: main writes the caller's rows, correction accumulates.

    ``CAKE_KDA_AFFINE_ROWS_IN_PLACE=0`` restores the private row windows plus
    the merging epilogue (same values; used for A/B and bitwise tests).
    """
    import os

    return os.environ.get("CAKE_KDA_AFFINE_ROWS_IN_PLACE", "1") != "0"


def _affine_window_row_starts(
    token_offsets, part_offsets, *, checkpoint_every_n_tokens=64
):
    """Per window: (original sequence index, checkpoint rows before the window in that sequence).

    Windows of a checkpointed sequence start on ``checkpoint_every_n_tokens``
    boundaries (``_affine_split_windows`` keeps an even chunk count per
    window), so each window's rows coincide with the sequence's own row grid
    and the window can address the caller's rows directly.
    """
    import bisect

    seq_ids, local_rows = [], []
    for part in range(len(token_offsets) - 1):
        seq = bisect.bisect_right(part_offsets, part) - 1
        local = token_offsets[part] - token_offsets[part_offsets[seq]]
        if local % checkpoint_every_n_tokens:
            raise ValueError("affine window starts must sit on checkpoint boundaries")
        seq_ids.append(seq)
        local_rows.append(local // checkpoint_every_n_tokens)
    return seq_ids, local_rows


class FlashKDABlackwellAffineSplitLaunch(FlashKDABlackwellBF16FusedLaunch):
    """PR #4779 split-sequence affine prefix over direct-M128 windows.

    Main, map, and correction launches reuse one variable-shape generated kernel
    identity. Runtime chunk windows expose independent CTAs; a segmented FP32
    prefix composes their affine transforms within each original sequence.  No exact sequence length is a
    compile-time specialization or dispatcher key.
    """

    route = BF16_ROUTE_AFFINE_SPLIT_M128
    schedule = "split_seq_affine_prefix_direct_m128_fp32_state"

    def __init__(self, *args, **kwargs) -> None:
        try:
            self._init(*args, **kwargs)
        except _CorrectionMergeUnavailable:
            # The correction module has no merging build, or the composite
            # cannot gather the merge's pool offsets: build it again with
            # private correction windows and the epilogue kernel.
            self.__dict__.clear()
            kwargs["_merge_correction"] = False
            self._init(*args, **kwargs)

    def _init(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale: float,
        out,
        A_log,
        dt_bias,
        lower_bound: float | None,
        initial_state=None,
        final_state=None,
        cu_seqlens=None,
        state_indices=None,
        state_checkpoints=None,
        checkpoint_cu_starts=None,
        checkpoint_every_n_tokens: int = 0,
        backend: str = "cuda_cpp",
        compute_dtype: str = "bf16",
        sequence_lengths: tuple[int, ...] | None = None,
        active_beta_f32: bool = False,
        _merge_correction: bool = True,
    ) -> None:
        import torch

        validate_kda_state_tensors(compute_dtype, initial_state, final_state)
        self.compute_dtype = compute_dtype
        self.active_beta_f32 = active_beta_f32
        if active_beta_f32 and (compute_dtype != "tf32" or lower_bound is None):
            raise ValueError("active-beta affine requires bounded TF32 compute")
        if q.ndim != 4 or not out.is_contiguous():
            raise ValueError("affine out requires a contiguous [B,T,H,128] tensor")
        # Strided q / k / v views reach every main-pass body in place; the
        # map / apply kernels read only the exported operators.
        _validate_qkv_layout(q, k, v)
        if any((tensor.shape != q.shape for tensor in (k, v, out))):
            raise ValueError("affine q/k/v/out shapes must match")
        if initial_state is None or final_state is None or state_indices is None:
            raise ValueError(
                "affine split-sequence requires indexed initial/final state"
            )
        batch, tokens_per_batch, heads, _ = q.shape
        tokens = batch * tokens_per_batch
        if cu_seqlens is None:
            resolved_lengths = (int(tokens_per_batch),) * batch
            if (
                sequence_lengths is not None
                and tuple(sequence_lengths) != resolved_lengths
            ):
                raise ValueError("sequence_lengths must match fixed [B,T]")
        else:
            _require_tensor(cu_seqlens, name="cu_seqlens", dtype=torch.int64, ndim=1)
            if batch != 1:
                raise ValueError("packed affine requires B=1")
            if sequence_lengths is None:
                raise ValueError(
                    "Packed KDA preparation requires host sequence_lengths"
                )
            resolved_lengths = tuple(sequence_lengths)
            if (
                len(resolved_lengths) + 1 != cu_seqlens.numel()
                or sum(resolved_lengths) != tokens
            ):
                raise ValueError("sequence_lengths must partition packed tokens")
        num_sequences = len(resolved_lengths)
        if not resolved_lengths or any(
            (
                not isinstance(n, int) or isinstance(n, bool) or n <= 0
                for n in resolved_lengths
            )
        ):
            raise ValueError("affine sequences must have positive integer lengths")
        _require_tensor(state_indices, name="state_indices", dtype=torch.int32, ndim=1)
        if state_indices.shape != (num_sequences,):
            raise ValueError(
                "affine state_indices must have one entry per original sequence"
            )
        g_flat, g_refresh = _flatten_gate_rows(
            g, logical_tokens_per_batch=tokens_per_batch, num_heads=heads
        )
        beta_flat, beta_refresh = _flatten_beta_rows(
            beta, logical_tokens_per_batch=tokens_per_batch, num_heads=heads
        )
        self._affine_input_refreshes = tuple(
            (x for x in (g_refresh, beta_refresh) if x is not None)
        )
        q, k, v, out = (x.view(1, tokens, heads, HEAD_DIM) for x in (q, k, v, out))
        g, beta = (g_flat.unsqueeze(0), beta_flat.unsqueeze(0))
        if initial_state.dtype != final_state.dtype:
            raise TypeError(
                "affine split-sequence initial/final state dtypes must match"
            )
        if initial_state.dtype not in (torch.bfloat16, torch.float32):
            raise TypeError("affine split-sequence state must be BF16 or FP32")
        self.gate_kind = gate_kind_from_lower_bound(lower_bound).value
        self._checkpoint_output = state_checkpoints
        self._use_output_projection = (
            compute_dtype == "tf32"
            and state_checkpoints is None
            and (lower_bound is not None)
        )
        if (
            state_checkpoints is not None
            or checkpoint_cu_starts is not None
            or checkpoint_every_n_tokens
        ):
            if (
                state_checkpoints is None
                or checkpoint_cu_starts is None
                or checkpoint_every_n_tokens != 64
            ):
                raise ValueError(
                    "affine checkpoints require output, offsets and interval64"
                )
            if not state_checkpoints.is_contiguous() or state_checkpoints.dtype not in (
                torch.bfloat16,
                torch.float32,
            ):
                raise TypeError(
                    "affine checkpoint output must be contiguous BF16 or FP32"
                )
            if (
                state_checkpoints.dtype == torch.float32
                and initial_state.dtype != torch.float32
            ):
                raise TypeError("FP32 affine checkpoints require an FP32 state pool")
            if state_checkpoints.ndim != 4 or state_checkpoints.shape[1:] != (
                q.shape[2],
                HEAD_DIM,
                HEAD_DIM,
            ):
                raise ValueError("affine checkpoint shape must be [C,H,128,128]")
            if (
                checkpoint_cu_starts.shape != (num_sequences + 1,)
                or checkpoint_cu_starts.dtype != torch.int64
            ):
                raise ValueError("affine checkpoint offsets must be int64[N+1]")
        token_offsets, part_offsets = _affine_split_windows(
            sequence_lengths=resolved_lengths,
            num_heads=heads,
            sm_count=_device_sm_count(q.device),
            fp32_indexed_state=initial_state.dtype == torch.float32,
            shared_tf32_factors=compute_dtype == "tf32",
            checkpoints=state_checkpoints is not None,
            unbounded_softplus=lower_bound is None,
        )
        num_parts = len(token_offsets) - 1
        self._num_sequences = num_sequences
        self.sequence_lengths = resolved_lengths
        # The bounded BF16 chain's correction windows merge themselves into
        # the output tail, the checkpoint rows (BF16 rows, then written in
        # place by the main windows) and the pool when the loader routes the
        # correction modules to merging programs.
        self._correction_merge = bool(
            _merge_correction
            and num_parts >= 2
            and compute_dtype == "bf16"
            and lower_bound is not None
            and (state_checkpoints is None or state_checkpoints.dtype == torch.bfloat16)
            and final_state.dtype == torch.float32
            and final_state.ndim == 4
            and final_state[0].is_contiguous()
            and _fused_affine_epilogue_enabled()
            and _correction_merge_routed(q.device)
        )
        # Every host-side metadata list of the composite (part windows, part
        # selectors, checkpoint-row plan) goes up in two pinned, stream-ordered
        # copies (int32 / int64) after the lists are complete; one pageable
        # ``torch.tensor(..., device=cuda)`` per list synchronized the host on
        # every plan-cache miss (Phase A contract for the fused body).
        first_part_tokens = token_offsets[1]
        tail_offsets = [offset - first_part_tokens for offset in token_offsets[1:]]
        host_i32 = {
            "part_cu_seqlens": list(part_offsets),
            "part_state_indices": [-1] * num_parts,
        }
        host_i64 = {
            "first_parts": list(part_offsets[:-1]),
            "last_parts": [end - 1 for end in part_offsets[1:]],
            "last_correction_parts": [max(0, end - 2) for end in part_offsets[1:]],
            "split_cu_seqlens": list(token_offsets),
            "tail_cu_seqlens": list(tail_offsets),
        }
        self._zero_first_correction = part_offsets[1] == 1
        self.num_parts = num_parts
        self.split_parts = num_parts
        state_shape = (num_sequences, heads, HEAD_DIM, HEAD_DIM)
        self._state_indices = state_indices
        self._state_indices_long = state_indices.to(dtype=torch.int64)
        self._initial_pool = initial_state
        self._initial_pool_pointer = _ffi_raw_pointer_carrier(initial_state)
        self._final_pool = final_state
        if initial_state.dtype != torch.float32:
            # Only the FP32-pool producer bodies are exported; the public entry
            # points reject other pools before this class is constructed.
            raise NotImplementedError(
                "affine split prefill requires an FP32 external state pool"
            )
        self.schedule = "split_seq_affine_prefix_direct_m128_fp32_state"
        if compute_dtype == "tf32":
            self.schedule += "_tf32_compute"
        # Torch-epilogue scratch (*_selected / checkpoint_merged) is allocated
        # only when the fused epilogue is unavailable (see below).
        self._main_initial = torch.zeros(
            (num_parts, heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=q.device
        )
        self._main_final = torch.empty_like(self._main_initial)
        map_dtype = torch.float32 if compute_dtype == "tf32" else torch.bfloat16
        self._map_initial = torch.zeros(
            (num_parts - 1, heads, HEAD_DIM, HEAD_DIM), dtype=map_dtype, device=q.device
        )
        identity = torch.eye(HEAD_DIM, dtype=map_dtype, device=q.device)
        self._map_initial.copy_(identity)
        self._map_state = torch.empty_like(self._map_initial)
        self._carry = torch.empty(
            (num_parts - 1, heads, HEAD_DIM, HEAD_DIM),
            dtype=torch.float32,
            device=q.device,
        )
        self._correction_final = torch.empty_like(self._carry)
        self._final_compact = torch.empty(
            state_shape, dtype=torch.float32, device=q.device
        )
        self._final_main_selected = None
        self._final_correction_selected = None
        self._zero_v = torch.zeros_like(v[:, first_part_tokens:])
        self._map_out = torch.empty_like(out[:, first_part_tokens:])
        # The correction output buffer is allocated below, once the route is
        # known: the apply route reduce-adds its correction into the output
        # tail and never touches it (kernel round 3, host lever).
        self._correction_out: torch.Tensor | None = None
        main_checkpoint_kwargs = {}
        correction_checkpoint_kwargs = {}
        self._checkpoint_in_place = False
        if state_checkpoints is not None:
            cp_counts = [(length + 63) // 64 for length in resolved_lengths]
            cp_count = sum(cp_counts)
            first_cp_count = (first_part_tokens + 63) // 64
            if state_checkpoints.shape[0] < cp_count:
                raise ValueError("affine checkpoint output is too small")
            main_cp_offsets = [0]
            for begin, end in zip(token_offsets, token_offsets[1:], strict=False):
                main_cp_offsets.append(main_cp_offsets[-1] + (end - begin + 63) // 64)
            correction_cp_offsets = [
                offset - first_cp_count for offset in main_cp_offsets[1:]
            ]
            self._checkpoint_start = checkpoint_cu_starts[:num_sequences]
            self._checkpoint_merged = None
            self._checkpoint_merged_first = None
            self._checkpoint_merged_tail = None
            self._first_cp_count = first_cp_count
            self._checkpoint_in_place = self._correction_merge or (
                state_checkpoints.dtype == torch.float32
                and _affine_rows_in_place_enabled()
            )
            if self._checkpoint_in_place:
                # FP32 rows (or the merging correction's BF16 rows): the main
                # windows write the caller's rows directly and the correction
                # windows add theirs in place, so no row is staged, re-read and
                # merged.  Each window's destination row
                # start (caller cu_starts of its sequence + rows before it) is
                # gathered on the device at every launch: no host readback.
                part_seq_ids, part_local_rows = _affine_window_row_starts(
                    token_offsets, part_offsets
                )
                host_i64["part_seq_ids"] = part_seq_ids
                host_i64["part_local_rows"] = part_local_rows
                self._part_row_starts = torch.zeros(
                    (num_parts + 1,), dtype=torch.int64, device=q.device
                )
                self._checkpoint_main = self._checkpoint_correction = None
                self.schedule += "_checkpoint64_rows_in_place"
            else:
                # Window checkpoints follow the output contract: BF16 rows keep
                # the main and correction windows exact before their merge.
                self._checkpoint_main = torch.empty(
                    (cp_count, heads, HEAD_DIM, HEAD_DIM),
                    dtype=state_checkpoints.dtype,
                    device=q.device,
                )
                self._checkpoint_correction = torch.empty(
                    (cp_count - first_cp_count, heads, HEAD_DIM, HEAD_DIM),
                    dtype=state_checkpoints.dtype,
                    device=q.device,
                )
                self._checkpoint_main_first = self._checkpoint_main[:first_cp_count]
                self._checkpoint_main_tail = self._checkpoint_main[first_cp_count:]
                host_i64["checkpoint_offsets"] = [
                    row for count in cp_counts for row in range(count)
                ]
                host_i64["checkpoint_sequence_ids"] = [
                    seq for seq, count in enumerate(cp_counts) for _ in range(count)
                ]
                host_i64["main_cp_offsets"] = list(main_cp_offsets)
                host_i64["correction_cp_offsets"] = list(correction_cp_offsets)
                self._checkpoint_indices = torch.empty(
                    (cp_count,), dtype=torch.int64, device=q.device
                )
                self.schedule += "_checkpoint64"
        if self._correction_merge:
            # Correction window (part - 1) that ends each sequence; a first
            # sequence of one window has no correction window and takes the
            # table's last entry.
            host_i64["merge_final_positions"] = [
                end - 2 if end >= 2 else num_parts - 1 for end in part_offsets[1:]
            ]
        uploaded_i32 = _upload_int_batch(q.device, host_i32, torch.int32)
        uploaded_i64 = _upload_int_batch(q.device, host_i64, torch.int64)
        if self._correction_merge:
            self._merge_final_positions = uploaded_i64["merge_final_positions"]
            # Pool element offset per correction window (-1: the window ends no
            # sequence), then the first sequence's; index prep fills the slots.
            self._merge_final_offsets = torch.full(
                (num_parts,), -1, dtype=torch.int64, device=q.device
            )
        self._metadata_host = (
            uploaded_i32.get("_host_pinned"),
            uploaded_i64.get("_host_pinned"),
        )
        self._part_cu_seqlens = uploaded_i32["part_cu_seqlens"]
        self._part_state_indices = uploaded_i32["part_state_indices"]
        self._first_parts = uploaded_i64["first_parts"]
        self._last_parts = uploaded_i64["last_parts"]
        self._last_correction_parts = uploaded_i64["last_correction_parts"]
        split_cu_seqlens = uploaded_i64["split_cu_seqlens"]
        tail_cu_seqlens = uploaded_i64["tail_cu_seqlens"]
        if state_checkpoints is not None and self._checkpoint_in_place:
            self._part_seq_ids = uploaded_i64["part_seq_ids"]
            self._part_local_rows = uploaded_i64["part_local_rows"]
            main_checkpoint_kwargs = dict(
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=self._part_row_starts,
                checkpoint_every_n_tokens=64,
                _affine_checkpoint_rows_resolved_on_device=True,
            )
            correction_checkpoint_kwargs = dict(
                state_checkpoints=state_checkpoints,
                checkpoint_cu_starts=self._part_row_starts[1:],
                checkpoint_every_n_tokens=64,
                _affine_checkpoint_rows_resolved_on_device=True,
            )
        elif state_checkpoints is not None:
            self._checkpoint_offsets = uploaded_i64["checkpoint_offsets"]
            self._checkpoint_sequence_ids = uploaded_i64["checkpoint_sequence_ids"]
            main_checkpoint_kwargs = dict(
                state_checkpoints=self._checkpoint_main,
                checkpoint_cu_starts=uploaded_i64["main_cp_offsets"],
                checkpoint_every_n_tokens=64,
            )
            correction_checkpoint_kwargs = dict(
                state_checkpoints=self._checkpoint_correction,
                checkpoint_cu_starts=uploaded_i64["correction_cp_offsets"],
                checkpoint_every_n_tokens=64,
            )
        main_factor_kwargs = {}
        tail_factor_kwargs = {}
        self._factor_cache = None
        if compute_dtype == "tf32":
            AFFINE_PACKET_BYTES = 58368
            packet_rows = AFFINE_PACKET_BYTES // 512
            packet_slots = (tokens + 31) // 32 + num_parts
            self._factor_cache = torch.empty(
                (packet_slots * heads * packet_rows, 128),
                device=q.device,
                dtype=torch.float32,
            )
            main_factor_kwargs = dict(
                _affine_factor_cache=self._factor_cache, _affine_factor_cache_mode=1
            )
            tail_factor_kwargs = dict(
                _affine_factor_cache=self._factor_cache,
                _affine_factor_cache_mode=2,
                _affine_cache_token_offset=first_part_tokens,
                _affine_cache_part_offset=1,
            )
            self.schedule += "_shared_factors"
        # The exported chunk operators (gate total, restore factors) are the
        # unbounded body's tile-anchored decay images; the bounded gate body
        # keeps the correction chain until its operators are exported too.
        self._apply_route = (
            _affine_apply_route_enabled()
            and compute_dtype == "bf16"
            and lower_bound is None
            and not self._use_output_projection
            and (state_checkpoints is None or self._checkpoint_in_place)
        )
        if not self._apply_route and not self._correction_merge:
            self._correction_out = torch.empty_like(out[:, first_part_tokens:])
        # Kernel round 7: on the lean-map apply route the main body also writes
        # the block pair maps of the tail windows (bitwise the pair-map
        # producer's) when the loader names the main module's fold build; the
        # composite then launches no pair-map producer.
        self._pair_map_fold = bool(
            self._apply_route
            and _affine_lean_map_enabled()
            and _affine_pair_map_fold_enabled()
            and _pair_map_fold_routed(q.device)
        )
        # Kernel round 8: on the chain route the main body also writes the
        # BF16 window maps of the tail windows (bitwise the map pass's) when
        # the loader names the main module's map-fold build; the composite
        # then launches no map pass.
        self._chain_map_fold = bool(
            not self._apply_route
            and not self._use_output_projection
            and compute_dtype == "bf16"
            and num_parts > 1
            and _chain_map_fold_enabled()
            and _chain_map_fold_routed(q.device)
        )
        main_launch_cls = (
            FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportFoldLaunch
            if self._pair_map_fold
            else FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportLaunch
            if self._apply_route
            else FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerMapFoldLaunch
            if self._chain_map_fold
            else FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch
        )
        main_args = (
            q,
            k,
            v,
            g,
            beta,
            scale,
            out,
            A_log,
            dt_bias,
            lower_bound,
            self._main_initial,
            self._main_final,
            split_cu_seqlens,
        )
        main_kwargs = dict(
            backend=backend,
            compute_dtype=compute_dtype,
            _affine_active_beta_f32=active_beta_f32,
            **main_checkpoint_kwargs,
            **main_factor_kwargs,
            sequence_lengths=tuple(
                (
                    b - a
                    for a, b in zip(token_offsets[0:], token_offsets[1:], strict=False)
                )
            ),
        )
        try:
            self._main = main_launch_cls(*main_args, **main_kwargs)
        except _ChainMapFoldUnavailable:
            # The main module has no map-fold build: the map pass writes the maps.
            self._chain_map_fold = False
            self._main = FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch(
                *main_args, **main_kwargs
            )
        except _PairMapFoldUnavailable:
            # The main module has no fold build (the body without checkpoint
            # rows, for one): the pair-map producer writes the maps.
            self._pair_map_fold = False
            self._main = (
                FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportLaunch(
                    *main_args, **main_kwargs
                )
            )
        # The map pass / pair-map producer is dropped only when the main body is
        # the fold build the loader names; any other body takes the fallback
        # above.
        if self._chain_map_fold and not _built_from_table(
            self._main, "_RENDERED_MAP_FOLD_ROUTES"
        ):
            self._chain_map_fold = False
            self._main = FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch(
                *main_args, **main_kwargs
            )
        if self._pair_map_fold and not _built_from_table(
            self._main, "_RENDERED_FOLD_ROUTES"
        ):
            self._pair_map_fold = False
            self._main = (
                FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerExportLaunch(
                    *main_args, **main_kwargs
                )
            )
        self._main.args.update(
            state_indices_addr=self._part_state_indices.data_ptr(),
            state_slot_stride=initial_state.stride(0),
            use_state_indices=1,
        )
        self._main.args["initial_state_f32"] = self._initial_pool_pointer
        if self._chain_map_fold:
            # The map-fold build writes every tail window's BF16 map state
            # through the main body's (otherwise unused) training-tape slot,
            # window p at [p - 1, head]: the scan reads it as the map pass's.
            self._main.args["tape_e"] = self._map_state.view(-1)
            self._main.__dict__.pop("_launch_records", None)
            self.schedule += "_mapfold"
        self._correction = None
        if not self._use_output_projection and not self._apply_route:
            correction_cls: type[FlashKDABlackwellFP32SlabM128PDLConsumerLaunch]
            if self._correction_merge:
                correction_cls = FlashKDABlackwellFP32SlabM128PDLConsumerMergeLaunch
            elif self._checkpoint_in_place:
                correction_cls = (
                    FlashKDABlackwellFP32SlabM128PDLConsumerAccumulateLaunch
                )
            else:
                correction_cls = FlashKDABlackwellFP32SlabM128PDLConsumerLaunch
            correction_out = (
                out[:, first_part_tokens:]
                if self._correction_merge
                else self._correction_out
            )
            assert correction_out is not None
            self._correction = correction_cls(
                q[:, first_part_tokens:],
                k[:, first_part_tokens:],
                self._zero_v,
                g[:, first_part_tokens:],
                beta[:, first_part_tokens:],
                scale,
                correction_out,
                A_log,
                dt_bias,
                lower_bound,
                self._carry,
                self._correction_final,
                tail_cu_seqlens,
                backend=backend,
                compute_dtype=compute_dtype,
                _affine_active_beta_f32=active_beta_f32,
                **correction_checkpoint_kwargs,
                **tail_factor_kwargs,
                sequence_lengths=tuple(
                    (
                        b - a
                        for a, b in zip(
                            token_offsets[1:], token_offsets[2:], strict=False
                        )
                    )
                ),
            )
            if self._correction_merge and not _built_from_table(
                self._correction, "_RENDERED_MERGE_ROUTES"
            ):
                # Not the merging build: rebuild with the epilogue (``__init__``).
                raise _CorrectionMergeUnavailable
            if self._correction_merge:
                # The merging body reads the per-window pool offsets through
                # ``state_indices_addr``, the main windows' final states through
                # ``norm_inv_out`` and stores into the pool (``final_state_f32``).
                self._correction.args.update(
                    state_indices_addr=self._merge_final_offsets.data_ptr(),
                    use_state_indices=0,
                    norm_inv_out=self._main_final,
                    final_state_f32=_ffi_raw_pointer_carrier(self._final_pool),
                )
                self.schedule += "_correction_merge"
        map_snapshot_kwargs = {}
        tail_lengths = [
            end - begin
            for begin, end in zip(tail_offsets, tail_offsets[1:], strict=False)
        ]
        if self._apply_route:
            # The map pass publishes its prefix maps (state entering every
            # 64-token block, row 0 = identity) for the apply kernel.
            snap_counts = [(length + 63) // 64 for length in tail_lengths]
            snap_starts = [0]
            for count in snap_counts:
                snap_starts.append(snap_starts[-1] + count)
            self._map_snapshots = torch.empty(
                (snap_starts[-1], heads, HEAD_DIM, HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
            self._map_snapshot_starts = snap_starts
            map_snapshot_kwargs = dict(
                state_checkpoints=self._map_snapshots,
                checkpoint_cu_starts=torch.tensor(
                    snap_starts, dtype=torch.int64, device=q.device
                ),
                checkpoint_every_n_tokens=64,
            )
        map_launch_cls = (
            FlashKDABlackwellFP32SlabM128PDLProducerLaunch
            if compute_dtype == "tf32"
            else FlashKDABlackwellBF16DirectM128PDLBridgeLaunch
        )
        self._map = map_launch_cls(
            q[:, first_part_tokens:],
            k[:, first_part_tokens:],
            self._zero_v,
            g[:, first_part_tokens:],
            beta[:, first_part_tokens:],
            scale,
            self._map_out,
            A_log,
            dt_bias,
            lower_bound,
            self._map_initial,
            self._map_state,
            tail_cu_seqlens,
            backend=backend,
            compute_dtype=compute_dtype,
            _affine_active_beta_f32=active_beta_f32,
            _affine_map_only=compute_dtype == "tf32",
            _affine_map_output=self._use_output_projection,
            **map_snapshot_kwargs,
            **tail_factor_kwargs,
            sequence_lengths=tuple(
                (
                    b - a
                    for a, b in zip(token_offsets[1:], token_offsets[2:], strict=False)
                )
            ),
        )
        self._apply_module = None
        if self._apply_route:
            self._apply_module = _build_kda_module(
                partial(_factory, "compiled_flashkda_affine_apply_m128")
            )
            export = self._main.operator_export
            cu_chunks = [0]
            for count in export["chunk_counts"]:
                cu_chunks.append(cu_chunks[-1] + count)
            items = build_affine_apply_items(
                tail_lengths=tail_lengths,
                cu_chunk_offsets=cu_chunks,
                snap_starts=self._map_snapshot_starts,
                tail_offsets=tail_offsets[:-1],
                num_heads=heads,
                rows_enabled=self._checkpoint_in_place,
                sm_count=_device_sm_count(q.device),
                # Kernel round 4 (lever 8b): the scan kernel stores every
                # sequence's final state and the row-0 carries itself; no
                # apply run evaluates S_w^in x M_final.
                owns_final=False,
            )
            self._apply_items = torch.tensor(
                items, dtype=torch.int32, device=q.device
            ).contiguous()
            self._apply_grid = (len(items), 1, 1)
            # Lean map chain: pair-map producer (this kernel, PAIRMAP mode) +
            # prefix-product chain replace the map pass.
            self._lean_map = _affine_lean_map_enabled()
            snap_total = self._map_snapshot_starts[-1]
            self._pair = torch.empty(
                (snap_total, heads, HEAD_DIM, HEAD_DIM),
                dtype=torch.bfloat16,
                device=q.device,
            )
            self._dpair = torch.empty(
                (snap_total, heads, HEAD_DIM), dtype=torch.float32, device=q.device
            )
            if self._pair_map_fold:
                # The fold build writes R_p and D_p of every 64-token block of
                # the tail windows through the main body's (otherwise unused)
                # training-tape slots, row (window snapshot start + block).
                self._main.args.update(
                    tape_e=self._pair.view(-1), norm_inv_out=self._dpair.view(-1)
                )
                self._main.__dict__.pop("_launch_records", None)
                # No pair-map producer is built, prepared or launched.
                self._pairmap_items = self._pairmap_grid = None
                self._pairmap_module = None
            else:
                pairmap_items = build_affine_apply_items(
                    tail_lengths=tail_lengths,
                    cu_chunk_offsets=cu_chunks,
                    snap_starts=self._map_snapshot_starts,
                    tail_offsets=tail_offsets[:-1],
                    num_heads=heads,
                    rows_enabled=False,
                    sm_count=_device_sm_count(q.device),
                    pairmap=True,
                )
                self._pairmap_items = torch.tensor(
                    pairmap_items, dtype=torch.int32, device=q.device
                ).contiguous()
                self._pairmap_grid = (len(pairmap_items), 1, 1)
                self._pairmap_module = _build_kda_module(
                    partial(_factory, "compiled_flashkda_affine_apply_m128"),
                    pairmap=True,
                )
            prefix_items = build_map_prefix_items(
                tail_lengths=tail_lengths,
                snap_starts=self._map_snapshot_starts,
                num_heads=heads,
            )
            self._prefix_items = torch.tensor(
                prefix_items, dtype=torch.int32, device=q.device
            ).contiguous()
            self._prefix_grid = (len(prefix_items), 1, 1)
            self._prefix_module = _build_kda_module(
                partial(_factory, "compiled_flashkda_map_prefix_m128")
            )
            # The scan kernel writes the BF16 hi/lo carry pair in place of a
            # host split pass (CARRY_HILO schedule).
            self._carry_hi = torch.empty_like(self._carry, dtype=torch.bfloat16)
            self._carry_lo = torch.empty_like(self._carry, dtype=torch.bfloat16)
            if not self._checkpoint_in_place:
                self._part_row_starts = torch.zeros(
                    (num_parts + 1,), dtype=torch.int64, device=q.device
                )
            self._apply_dummy_rows = torch.zeros(
                (1, heads, HEAD_DIM, HEAD_DIM), dtype=torch.float32, device=q.device
            )
            # Pair-map outputs of the shared kernel ABI (unused by the apply mode).
            self._apply_dummy_pair = torch.zeros(
                (1, heads, HEAD_DIM, HEAD_DIM), dtype=torch.bfloat16, device=q.device
            )
            self._apply_dummy_dpair = torch.zeros(
                (1, heads, HEAD_DIM), dtype=torch.float32, device=q.device
            )
            # The scan kernel scatters the final state into the caller's pool
            # slots and stores the FP32 carry into row 0 of every tail window,
            # so the composite has no torch epilogue on this route.
            if self._final_pool.ndim != 4 or not self._final_pool[0].is_contiguous():
                raise ValueError(
                    "affine apply route requires a [slots,H,128,128] state pool with contiguous slots"
                )
            if (
                self._checkpoint_in_place
                and not self._checkpoint_output[0].is_contiguous()
            ):
                raise ValueError(
                    "affine apply route requires contiguous FP32 checkpoint rows"
                )
            # The apply kernel reduce-adds the correction into the output tail
            # itself (no correction buffer, no host add).
            self._apply_out_fused = True
            self.schedule += "_apply"
            if self._lean_map:
                self.schedule += "_leanmap"
            if self._pair_map_fold:
                self.schedule += "_pairfold"
        if compute_dtype == "bf16":
            self._map.args["initial_state_f32"] = self._main_final
        # The hi/lo carry variant is keyed only on the apply route so the
        # published plain scan programs (bf16 and tf32) keep their build key.
        scan_kwargs = dict(carry_hilo=True) if self._apply_route else {}
        # A generated scan routed by head count (one body per band height)
        # covers ``ctas_per_head`` row bands of each sequence-head; the other
        # scan programs run 32 row-band CTAs per sequence-head.
        self._scan_module = _build_kda_module(
            partial(
                _factory, "compiled_flashkda_split_scan_bf16_m128", route_heads=heads
            ),
            use_pdl=True,
            compute_dtype=compute_dtype,
            backend=backend,
            **scan_kwargs,
        )
        self._scan_ctas_per_head = (
            getattr(self._scan_module, "ctas_per_head", None) or 32
        )
        if not self._apply_route:
            # Unused by the plain schedule; the kernel ABI still carries the pointers.
            self._carry_hi = self._carry.view(torch.bfloat16)
            self._carry_lo = self._carry_hi
        self._out_tail = out[:, first_part_tokens:]
        self._projection_module = None
        if self._use_output_projection:
            assert self._correction_out is not None
            self._projection_module = _build_kda_module(
                partial(_factory, "compiled_affine_output_projection")
            )
            self._map_coefficients = torch.empty_like(
                self._correction_out, dtype=torch.float32
            )
            self._map.args["map_output_f32"] = self._map_coefficients
            chunks = [
                (start, min(128, end - start), part)
                for part, (begin, end) in enumerate(
                    zip(tail_offsets, tail_offsets[1:], strict=False)
                )
                for start in range(begin, end, 128)
            ]
            self._projection_args = dict(
                carry=self._carry,
                carry_tma=self._carry,
                coefficients=self._map_coefficients,
                coefficients_tma=self._map_coefficients,
                out=self._correction_out,
                token_starts=torch.tensor(
                    [x[0] for x in chunks], device=q.device, dtype=torch.int32
                ),
                token_counts=torch.tensor(
                    [x[1] for x in chunks], device=q.device, dtype=torch.int32
                ),
                part_ids=torch.tensor(
                    [x[2] for x in chunks], device=q.device, dtype=torch.int32
                ),
                num_heads=heads,
            )
            self._projection_grid = (len(chunks) * heads, 1, 1)
            self.schedule += "_parallel_output_projection"
        self._keepalive = (
            q,
            k,
            v,
            g,
            beta,
            out,
            A_log,
            dt_bias,
            state_indices,
            split_cu_seqlens,
            tail_cu_seqlens,
        )
        self._launch_device = q.device
        self._fused_epilogue = None
        if not self._use_output_projection and _fused_affine_epilogue_enabled():
            self._fused_epilogue = _FusedAffineEpilogue.build(self)
            if self._fused_epilogue is not None and not self._correction_merge:
                self.schedule += "_fused_epilogue"
        if self._correction_merge and (
            self._fused_epilogue is None
            or self._fused_epilogue.index_prep_merge is None
        ):
            # The merge needs the per-launch pool offsets of the index prep.
            raise _CorrectionMergeUnavailable
        if self._fused_epilogue is None:
            # Torch epilogue scratch: a merged copy of the checkpoint rows plus
            # the selected final states (~4x the checkpoint rows of a 16K pack;
            # retained per plan-cache entry, so only when needed).
            self._final_main_selected = torch.empty_like(self._final_compact)
            self._final_correction_selected = torch.empty_like(self._final_compact)
            if self._checkpoint_output is not None and not self._checkpoint_in_place:
                self._checkpoint_merged = torch.empty_like(self._checkpoint_main)
                self._checkpoint_merged_first = self._checkpoint_merged[
                    : self._first_cp_count
                ]
                self._checkpoint_merged_tail = self._checkpoint_merged[
                    self._first_cp_count :
                ]
        self._use_cuda_graph = True
        self._cuda_graph_warmed = False
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        # The window-map kernel of the chain route runs ahead of the index
        # gather and the main kernel (see _MAP_FIRST_MODULES): it fills the
        # host gap before the main launch instead of waiting for the main
        # kernel it does not depend on.
        self._map_first = (
            not self._apply_route
            and not self._chain_map_fold
            and self._correction is not None
            and self._map.prepare_module is None
            and getattr(self._map.module, "name", None) in _MAP_FIRST_MODULES
        )

    def _launch_uncaptured(self) -> None:
        with _ffi_stream_context(self._launch_device):
            records = self._launch_records
            if records is None or self._descriptors_stale:
                self._launch_building_records()
            else:
                _run_launch_records(records)

    def _launch_building_records(self) -> None:
        """Launch once while (re)building the launch records.

        Runs on the first launch, after a plan-cache rebind (which drops the
        records) and on a hit under CUDA-graph capture (which marks the
        composite stale).  Every descriptor re-encode is enqueued right before
        the first kernel that reads it, and a deferred map/correction rebind is
        applied while the first chain kernel runs.  Later calls fire the
        stored records back to back with no host work between two launches.
        """
        if self._descriptors_stale:
            # A plan-cache hit under CUDA-graph capture marks the composite;
            # every part re-encodes its descriptors in its own stream context.
            for sub_name in AFFINE_SUB_LAUNCHES:
                sub = getattr(self, sub_name, None)
                if sub is not None:
                    sub._descriptors_stale = True
            # The apply-route kernels hold prepared TMA descriptors of caller
            # storage (output tail, checkpoint rows): re-prepare them too,
            # including the prefix chain whose descriptors are launch-owned.
            self._apply_descriptors_stale = bool(self._apply_route)
            self._apply_prefix_stale = bool(self._apply_route)
            self._descriptors_stale = False
        map_first = self._map_first
        if map_first:
            # The map is the first chain kernel: its rebind lands before it.
            flush_deferred_rebind(self)
        records: list = []

        def fire(part) -> None:
            _run_launch_records(part)
            records.extend(part)

        fire(
            [
                (destination.copy_, (source,))
                for destination, source in self._affine_input_refreshes
            ]
        )
        if map_first:
            fire(self._map_records())
        if self._correction_merge:
            # The index gather also places each sequence's pool offset at the
            # correction window that ends it.
            fire(
                [
                    (
                        self._fused_epilogue.index_prep_merge,
                        _ffi_arguments(
                            (
                                *self._index_prep_arguments(),
                                self._merge_final_positions,
                                self._merge_final_offsets,
                                self._fused_epilogue.pool_slot_stride,
                            )
                        ),
                    )
                ]
            )
        elif self._fused_epilogue is not None:
            # One kernel gathers every per-call index the parts consume
            # (window state indices, in-place row starts, the int64 state
            # indices of the epilogue).
            fire(
                [
                    (
                        self._fused_epilogue.index_prep,
                        _ffi_arguments(self._index_prep_arguments()),
                    )
                ]
            )
        else:
            fire([(self._gather_indices_with_torch, ())])
        fire(_settled_launch_records(self._main))
        # The map/correction rebinds of a plan-cache hit were deferred past
        # the first chain kernel; apply them while it runs.
        flush_deferred_rebind(self)
        if self._fused_epilogue is None and not self._apply_route:
            # The int64 index copy is only consumed by the final-state
            # scatter, so it follows the first chain kernel.
            fire([(self._state_indices_long.copy_, (self._state_indices,))])
        if not map_first:
            fire(self._map_records())
        scan_grid = (
            self._num_sequences
            * int(self._main_final.shape[1])
            * self._scan_ctas_per_head,
            1,
            1,
        )
        fire(
            [
                _prepared_kernel_record(
                    self._scan_module, scan_grid, self._scan_bindings()
                )
            ]
        )
        if self._use_output_projection:
            fire(
                [
                    _prepared_kernel_record(
                        self._projection_module,
                        self._projection_grid,
                        self._projection_args,
                    )
                ]
            )
        elif self._apply_route:
            fire(
                [
                    _prepared_kernel_record(
                        self._apply_module, self._apply_grid, self._apply_bindings()
                    )
                ]
            )
        else:
            fire(_settled_launch_records(self._correction))
        if not self._apply_route and not self._correction_merge:
            # The apply route's scan kernel wrote the final states into the
            # pool and the row-0 carries into the checkpoint rows.
            if self._fused_epilogue is not None:
                fire(
                    [
                        (
                            self._fused_epilogue.run,
                            _ffi_arguments(self._fused_epilogue_arguments()),
                        )
                    ]
                )
            else:
                fire([(self._torch_epilogue, ())])
        self._launch_records = tuple(records)

    def _map_records(self) -> list:
        """Window maps: the apply route's pair-map + prefix chain, else the map launch."""
        if self._apply_route and getattr(self, "_apply_descriptors_stale", False):
            if getattr(self, "_apply_prefix_stale", False):
                # Only the fused apply kernel stores through the caller's
                # out_tma / rows_tma (the pair-map producer runs the same ABI
                # in PAIRMAP mode and writes launch-owned maps only; the prefix
                # chain addresses launch-owned buffers), so a plan-cache rebind
                # re-uploads one descriptor set and the pair-map / prefix sets
                # re-prepare only on the capture / full-stale path.
                if self._pairmap_module is not None:
                    self._pairmap_module.prepare(
                        grid=self._pairmap_grid, **self._pairmap_bindings()
                    )
                self._prefix_module.prepare(
                    grid=self._prefix_grid, **self._prefix_bindings()
                )
                self._apply_prefix_stale = False
            elif getattr(self._pairmap_module, "encodes_on_host", False):
                # A program whose maps the loader encodes on the host keeps
                # every stored map current: the pair-map producer's maps
                # include the caller's out_tma / rows_tma sources this rebind
                # moved, and re-encoding them costs no device work.
                self._pairmap_module.prepare(
                    grid=self._pairmap_grid, **self._pairmap_bindings()
                )
            self._apply_module.prepare(grid=self._apply_grid, **self._apply_bindings())
            self._apply_descriptors_stale = False
        if self._apply_route and self._lean_map:
            prefix = _prepared_kernel_record(
                self._prefix_module, self._prefix_grid, self._prefix_bindings()
            )
            if self._pairmap_module is None:
                # The main body wrote the pair maps: the prefix chain alone.
                return [prefix]
            return [
                _prepared_kernel_record(
                    self._pairmap_module, self._pairmap_grid, self._pairmap_bindings()
                ),
                prefix,
            ]
        if getattr(self, "_chain_map_fold", False):
            # The main body wrote the window maps: no map launch.  The map
            # pass stays constructed for the plan-cache ABI.
            return []
        return _settled_launch_records(self._map)

    def _index_prep_arguments(self) -> tuple:
        rows = self._checkpoint_in_place
        return (
            self._state_indices,
            self._first_parts,
            self._part_state_indices,
            self._state_indices_long,
            self._checkpoint_start if rows else self._first_parts,
            self._part_seq_ids if rows else self._first_parts,
            self._part_local_rows if rows else self._first_parts,
            self._part_row_starts if rows else self._first_parts,
            self.num_parts if rows else 0,
        )

    def _scan_bindings(self) -> dict:
        heads = int(self._main_final.shape[1])
        return dict(
            split_state=self._main_final,
            map_state_bf16=self._map_state,
            carry=self._carry,
            carry_hi=self._carry_hi,
            carry_lo=self._carry_lo,
            num_heads=heads,
            part_cu_seqlens=self._part_cu_seqlens,
            **self._scan_epilogue_bindings(),
        )

    def _gather_indices_with_torch(self) -> None:
        import torch

        self._part_state_indices.index_copy_(0, self._first_parts, self._state_indices)
        if self._checkpoint_in_place:
            starts = self._part_row_starts[: self.num_parts]
            torch.index_select(
                self._checkpoint_start, 0, self._part_seq_ids, out=starts
            )
            starts.add_(self._part_local_rows)

    def _torch_epilogue(self) -> None:
        """Torch glue of the chain route without the fused epilogue kernel."""
        import torch

        if self._checkpoint_output is not None and not self._checkpoint_in_place:
            self._checkpoint_merged_first.copy_(self._checkpoint_main_first)
            torch.add(
                self._checkpoint_main_tail,
                self._checkpoint_correction,
                out=self._checkpoint_merged_tail,
            )
            if self._num_sequences == 1:
                torch.add(
                    self._checkpoint_offsets,
                    self._checkpoint_start,
                    out=self._checkpoint_indices,
                )
            else:
                torch.index_select(
                    self._checkpoint_start,
                    0,
                    self._checkpoint_sequence_ids,
                    out=self._checkpoint_indices,
                )
                self._checkpoint_indices.add_(self._checkpoint_offsets)
            self._checkpoint_output.index_copy_(
                0, self._checkpoint_indices, self._checkpoint_merged
            )
        assert self._correction_out is not None
        self._out_tail.add_(self._correction_out)
        if not self._use_output_projection:
            if self._num_sequences == 1:
                torch.add(
                    self._main_final[-1:],
                    self._correction_final[-1:],
                    out=self._final_compact,
                )
            else:
                torch.index_select(
                    self._main_final, 0, self._last_parts, out=self._final_main_selected
                )
                torch.index_select(
                    self._correction_final,
                    0,
                    self._last_correction_parts,
                    out=self._final_correction_selected,
                )
                if self._zero_first_correction:
                    self._final_correction_selected[:1].zero_()
                torch.add(
                    self._final_main_selected,
                    self._final_correction_selected,
                    out=self._final_compact,
                )
        self._final_pool.index_copy_(0, self._state_indices_long, self._final_compact)

    def _fused_epilogue_arguments(self) -> tuple:
        fused = self._fused_epilogue
        merge_rows = (
            self._checkpoint_output is not None and not self._checkpoint_in_place
        )
        if merge_rows:
            rows = (
                self._checkpoint_main,
                self._checkpoint_correction,
                self._checkpoint_output,
                self._checkpoint_offsets,
                self._checkpoint_sequence_ids,
                self._checkpoint_start,
            )
        else:
            # Unused by the kernel (has_rows=0); any tensors of the right kind.
            rows = (
                self._main_final,
                self._main_final,
                self._main_final,
                self._last_parts,
                self._last_parts,
                self._last_parts,
            )
        return (
            *rows,
            fused.first_rows,
            fused.num_rows,
            self._out_tail,
            # The apply route passes zero tail elements (the apply kernel
            # already reduce-added its correction), so any tensor of the
            # right kind stands in for the absent correction buffer.
            self._correction_out
            if self._correction_out is not None
            else self._out_tail,
            self._main_final,
            self._correction_final,
            self._last_parts,
            self._last_correction_parts,
            int(self._zero_first_correction),
            self._final_compact,
            self._final_pool,
            fused.pool_slot_stride,
            self._state_indices_long,
            self._num_sequences,
            fused.heads,
            # The apply kernel already reduce-added the correction into the
            # output tail; the epilogue then skips the tail add.
            0 if getattr(self, "_apply_out_fused", False) else fused.tail_elems,
            int(merge_rows),
        )

    def _pairmap_bindings(self, *, maps=None, map_final=None) -> dict:
        """Kernel arguments of the pair-map producer (apply kernel ABI, PAIRMAP mode)."""

        export = self._main.operator_export
        heads = int(self._carry.shape[1])
        maps = self._map_snapshots if maps is None else maps
        map_final = self._map_state if map_final is None else map_final
        return dict(
            items=self._pairmap_items,
            num_heads=heads,
            carry_hi=self._carry_hi,
            carry_hi_tma=self._carry_hi.view(1, -1, 1, HEAD_DIM),
            carry_lo=self._carry_lo,
            carry_lo_tma=self._carry_lo.view(1, -1, 1, HEAD_DIM),
            maps=maps,
            maps_tma=maps.view(1, -1, 1, HEAD_DIM),
            map_final=map_final,
            map_final_tma=map_final.view(1, -1, 1, HEAD_DIM),
            op_qd=export["qd_raw"],
            op_kd=export["kd_raw"],
            op_ft=export["ft_raw"],
            op_inv=export["inv_raw"],
            op_vec=export["decay_beta"],
            out=self._out_tail,
            out_tma=self._out_tail.reshape(-1, heads, HEAD_DIM),
            rows=self._apply_dummy_rows,
            rows_tma=self._apply_dummy_rows,
            row_starts=self._part_row_starts,
            final=self._correction_final,
            final_tma=self._correction_final,
            pair=self._pair,
            pair_tma=self._pair,
            dpair=self._dpair,
        )

    def _prefix_bindings(self, *, maps=None, map_final=None) -> dict:
        """Kernel arguments of the prefix-product chain."""

        maps = self._map_snapshots if maps is None else maps
        map_final = self._map_state if map_final is None else map_final
        return dict(
            items=self._prefix_items,
            num_heads=int(self._carry.shape[1]),
            pair=self._pair,
            pair_tma=self._pair.view(1, -1, 1, HEAD_DIM),
            dpair=self._dpair,
            maps=maps,
            maps_tma=maps,
            map_final=map_final,
            map_final_tma=map_final,
        )

    def _scan_epilogue_bindings(self) -> dict:
        """Scan-kernel arguments of the state epilogue (kernel round 4, lever 8b).

        On the apply route the scan scatters each sequence's final state into
        the caller's pool slot and stores the FP32 carry entering every tail
        window into row 0 of its checkpoint rows (the main window wrote that
        row as zeros).  Other routes keep their epilogue; the FP32 map
        schedule keeps its ``final_state[seq]`` output for the projection.
        The caller tensors are read at launch time, so a plan-cache rebind
        that moves the pool, the rows or the state indices needs no
        descriptor refresh here.
        """

        if not self._apply_route:
            return dict(
                final_state=self._final_compact,
                write_final_state=int(self._use_output_projection),
                final_indices=self._part_cu_seqlens,
                final_slot_stride=0,
                final_state_bf16=self._carry_hi,
                rows=self._main_final,
                row_starts=self._first_parts,
                write_rows=0,
            )
        # A pool sliced out of a larger buffer is not contiguous as a whole
        # (slot stride > slot size); the kernel indexes it by slot stride from
        # its data pointer, so hand the shim a pointer carrier (as the main
        # kernel's indexed initial-state pool already does).
        pool = _ffi_raw_pointer_carrier(self._final_pool)
        return dict(
            final_state=pool,
            write_final_state=1,
            final_indices=self._state_indices,
            final_slot_stride=int(self._final_pool.stride(0)),
            final_state_bf16=self._carry_hi,
            rows=self._checkpoint_output
            if self._checkpoint_in_place
            else self._apply_dummy_rows,
            row_starts=self._part_row_starts,
            write_rows=int(self._checkpoint_in_place),
        )

    def _apply_bindings(self) -> dict:
        """Kernel arguments of the fused apply kernel.

        hi + lo (written by the scan kernel) reproduces the FP32 carry to about
        16 mantissa bits.
        """

        export = self._main.operator_export
        heads = int(self._carry.shape[1])
        rows = (
            self._checkpoint_output
            if self._checkpoint_in_place
            else self._apply_dummy_rows
        )
        return dict(
            items=self._apply_items,
            num_heads=heads,
            carry_hi=self._carry_hi,
            carry_hi_tma=self._carry_hi.view(1, -1, 1, HEAD_DIM),
            carry_lo=self._carry_lo,
            carry_lo_tma=self._carry_lo.view(1, -1, 1, HEAD_DIM),
            maps=self._map_snapshots,
            maps_tma=self._map_snapshots.view(1, -1, 1, HEAD_DIM),
            map_final=self._map_state,
            map_final_tma=self._map_state.view(1, -1, 1, HEAD_DIM),
            op_qd=export["qd_raw"],
            op_kd=export["kd_raw"],
            op_ft=export["ft_raw"],
            op_inv=export["inv_raw"],
            op_vec=export["decay_beta"],
            out=self._out_tail,
            out_tma=self._out_tail.reshape(-1, heads, HEAD_DIM),
            rows=rows,
            rows_tma=rows,
            row_starts=self._part_row_starts,
            final=self._correction_final,
            final_tma=self._correction_final,
            pair=self._apply_dummy_pair,
            pair_tma=self._apply_dummy_pair,
            dpair=self._apply_dummy_dpair,
        )

    def close(self) -> None:
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._main.close()
        self._map.close()
        if self._correction is not None:
            self._correction.close()


def _launch_sequence_lengths(q, cu_seqlens, declared=None) -> tuple[int, ...]:
    """Resolve preparation metadata once; never inspect sequence values at launch."""
    if declared is not None:
        return tuple(declared)
    if cu_seqlens is None:
        return (int(q.shape[1]),) * int(q.shape[0])
    raise ValueError("Packed KDA preparation requires host sequence_lengths")


def _supports_fp32_small_bh_launch(args, kwargs) -> bool:
    """Resolve whether the FP32 call fits the shared owner/helper schedule."""

    def argument(position: int, name: str, default=None):
        return args[position] if len(args) > position else kwargs.get(name, default)

    q = argument(0, "q")
    beta = argument(4, "beta")
    lower_bound = argument(9, "lower_bound")
    cu_seqlens = argument(12, "cu_seqlens")
    state_checkpoints = argument(14, "state_checkpoints")
    checkpoint_cu_starts = argument(15, "checkpoint_cu_starts")
    checkpoint_every_n_tokens = int(argument(16, "checkpoint_every_n_tokens", 0))
    compute_dtype = argument(18, "compute_dtype", "bf16")
    if q is None or beta is None:
        return False
    if compute_dtype == "tf32":
        if checkpoint_every_n_tokens % 32 != 0:
            return False
    elif (
        lower_bound is None
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
        or (checkpoint_every_n_tokens != 0)
        or (not beta.is_contiguous())
    ):
        return False
    sequence_lengths = _launch_sequence_lengths(
        q, cu_seqlens, argument(19, "sequence_lengths")
    )
    return _should_use_small_bh_owner_helper(
        gpu_arch=detect_gpu_arch(),
        sm_count=_device_sm_count(q.device),
        num_seqs=len(sequence_lengths),
        num_heads=int(q.shape[2]),
        max_seq_len=max(sequence_lengths, default=0),
        compute_dtype=compute_dtype,
        unbounded_softplus=lower_bound is None,
    )


def _supports_fp32_bt16_launch(args, kwargs) -> bool:
    """Resolve whether the FP32 call fits the shared BT16 chain family."""

    def argument(position: int, name: str, default=None):
        return args[position] if len(args) > position else kwargs.get(name, default)

    q = argument(0, "q")
    beta = argument(4, "beta")
    lower_bound = argument(9, "lower_bound")
    initial_state = argument(10, "initial_state")
    final_state = argument(11, "final_state")
    cu_seqlens = argument(12, "cu_seqlens")
    state_checkpoints = argument(14, "state_checkpoints")
    checkpoint_cu_starts = argument(15, "checkpoint_cu_starts")
    checkpoint_every_n_tokens = int(argument(16, "checkpoint_every_n_tokens", 0))
    tf32_compute = argument(18, "compute_dtype", "bf16") == "tf32"
    checkpoint_request = (
        state_checkpoints is not None
        or checkpoint_cu_starts is not None
        or checkpoint_every_n_tokens != 0
    )
    if (
        q is None
        or beta is None
        or lower_bound is None
        or (not tf32_compute and (not beta.is_contiguous()))
    ):
        return False
    if checkpoint_request and (
        not tf32_compute
        or state_checkpoints is None
        or checkpoint_cu_starts is None
        or (checkpoint_every_n_tokens != 64)
    ):
        return False
    num_heads = int(q.shape[2])
    fixed_layout = cu_seqlens is None
    sequence_lengths = _launch_sequence_lengths(
        q, cu_seqlens, argument(19, "sequence_lengths")
    )
    if not sequence_lengths or min(sequence_lengths) <= 0:
        return False
    gpu_arch = detect_gpu_arch()
    sm_count = _device_sm_count(q.device)
    total_tasks = len(sequence_lengths) * int(num_heads)
    if tf32_compute and gpu_arch in ("sm_100a", "sm_103a"):
        if checkpoint_request:
            if max(sequence_lengths) >= 512:
                checkpoint_grid_fits = BT16_VALUE_SPLITS * total_tasks <= sm_count
            else:
                checkpoint_grid_fits = (
                    TF32_BT16_CHECKPOINT_TASK_DIVISOR * total_tasks <= 2 * sm_count
                )
            return (
                checkpoint_grid_fits
                and sm_count < 2 * SMALL_BH_GROUP_SIZE * total_tasks
                and (max(sequence_lengths) >= 384)
            )
        if BT16_VALUE_SPLITS * total_tasks <= sm_count and max(sequence_lengths) >= 384:
            return True
    if checkpoint_request:
        return False
    route = _select_bf16_route(
        compute_capability=_COMPUTE_CAPABILITY[gpu_arch],
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_sequences=len(sequence_lengths),
        num_heads=int(num_heads),
        uniform_sequences=len(set(sequence_lengths)) == 1,
        lpt_loads=(),
        max_sequence_length=max(sequence_lengths),
        use_initial_state=initial_state is not None,
        store_final_state=final_state is not None,
    )
    return route == BF16_ROUTE_BT16_M64


def _supports_affine_split_launch(args, kwargs) -> bool:
    """Gate FlashInfer PR #4779's variable-part affine-prefix schedule."""
    import torch

    def argument(position: int, name: str, default=None):
        return args[position] if len(args) > position else kwargs.get(name, default)

    q = argument(0, "q")
    beta = argument(4, "beta")
    initial_state = argument(10, "initial_state")
    final_state = argument(11, "final_state")
    cu_seqlens = argument(12, "cu_seqlens")
    state_indices = argument(13, "state_indices")
    state_checkpoints = argument(14, "state_checkpoints")
    checkpoint_cu_starts = argument(15, "checkpoint_cu_starts")
    checkpoint_every_n_tokens = int(argument(16, "checkpoint_every_n_tokens", 0))
    compute_dtype = argument(18, "compute_dtype", "bf16")
    if (
        q is None
        or beta is None
        or initial_state is None
        or (final_state is None)
        or (state_indices is None)
    ):
        return False
    k = argument(1, "k")
    v = argument(2, "v")
    out = argument(6, "out")
    if any(tensor is None for tensor in (k, v, out)) or not out.is_contiguous():
        return False
    # Round-5 lever 5b: serving hands q / k / v over as strided views of one
    # packed qkv row.  The split's main pass accepts every layout
    # ``_validate_qkv_layout`` accepts (dense, or in place / one dense copy),
    # so the gate must not send such calls to the sequential body.
    try:
        _validate_qkv_layout(q, k, v)
    except (TypeError, ValueError):
        return False
    checkpoint_request = (
        state_checkpoints is not None
        or checkpoint_cu_starts is not None
        or checkpoint_every_n_tokens != 0
    )
    if checkpoint_request and (
        state_checkpoints is None
        or checkpoint_cu_starts is None
        or checkpoint_every_n_tokens != 64
    ):
        return False
    # Unbounded (softplus) prefill on the FP32 external state pool is the
    # serving contract (final state resumes decode; checkpoint rows feed the
    # radix cache).  Its split parts carry the standalone FP32 state carrier,
    # which removed the composed per-part drift (real 8K activations: per-chunk
    # rel L2 flat at 0.002-0.0034 across nine parts, Triton 0.0026-0.0031),
    # and the composite is covered by the plan cache, so it takes the split at
    # the same part-count crossover as every other caller.
    lengths = _launch_sequence_lengths(q, cu_seqlens, argument(19, "sequence_lengths"))
    if not lengths or min(lengths) <= 0:
        return False
    gpu_arch = detect_gpu_arch()
    if gpu_arch not in ("sm_100a", "sm_103a"):
        return False
    heads = int(q.shape[2])
    if compute_dtype == "bf16" and argument(9, "lower_bound") is None:
        # The unbounded composite's window bodies are exported for a beta whose
        # token pitch differs from the head count (the parts then take the
        # serving ABI, as serving allocators hand beta over), for FP32
        # checkpoint rows (merged in place) with either pitch and, with the
        # plain pitch, for the pair-packed H12 beta of an even token count
        # without rows.  Other unbounded calls keep the sequential body, which
        # carries the same FP32 chunk state.
        if beta.ndim != 3:
            return False
        tokens_per_batch = int(q.shape[1])
        if int(beta.shape[0]) == 1 or (
            int(beta.shape[1]) == tokens_per_batch
            and beta.stride(0) == tokens_per_batch * beta.stride(1)
        ):
            plain_pitch = beta.stride(1) == heads
            paired_beta = plain_pitch and beta.data_ptr() % 16 == 0
        else:
            # Padded or batch-strided storage with B > 1: the parts read one
            # dense refresh copy (plain pitch, aligned base).
            plain_pitch = paired_beta = True
        if plain_pitch:
            if state_checkpoints is None:
                if heads != 12 or sum(lengths) % 2 or not paired_beta:
                    return False
            elif state_checkpoints.dtype != torch.float32:
                return False
    crossover = dict(
        sm_count=_device_sm_count(q.device),
        fp32_indexed_state=initial_state.dtype == torch.float32,
        shared_tf32_factors=compute_dtype == "tf32",
        unbounded_softplus=argument(9, "lower_bound") is None,
    )
    if not crossover["shared_tf32_factors"] and _affine_split_policy() == "model":
        # BF16 family: compare the measured cost model of the composite on the
        # windows it would actually get against the sequential body's longest
        # chain.  The fixed thresholds below were fitted on GB300 and put the
        # 8192-token break-even one chunk above an 8128-token member, which
        # left 8128+64 packs on a 1.0 ms chain where the composite takes 0.42
        # (B200, H12); the model also declines 2x8192 at H16 with the bounded
        # gate (0.62 vs 0.47 ms sequential), which the thresholds accepted.
        estimate = _affine_bf16_split_estimate_us(
            sequence_lengths=lengths,
            num_heads=heads,
            sm_count=crossover["sm_count"],
            checkpoints=checkpoint_request,
            gate_kind=(
                "unbounded_softplus"
                if crossover["unbounded_softplus"]
                else "lower_bound"
            ),
            gpu_arch=gpu_arch,
        )
        if estimate is not None:
            composite_us, sequential_us = estimate
            return composite_us < sequential_us
        if (gpu_arch, "lower_bound") in AFFINE_BF16_COST_MODEL_US:
            # Modelled architecture, but the call is outside the composite's
            # contract (task cap, half the SM count, nothing to split).
            return False
    chunk_counts = [
        (length + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK for length in lengths
    ]
    if len(lengths) == 1 or _affine_split_policy() == "legacy":
        return (
            _affine_split_part_count(
                tasks=len(lengths) * heads, chunks=max(chunk_counts), **crossover
            )
            >= 2
        )
    # Packed call: the sequential body's makespan is the longest sequence's
    # chunk chain (GB300 H16, rows every 64: ~3.7 us per chunk), while the
    # composite costs ~0.93 ms for a 16K pack regardless of how it is split.
    # Measured break-even is a 256-chunk (8192-token) longest member: packs
    # with a longer member gain up to 1.5x (1000+12000+3384: 1.39 -> 0.97 ms),
    # balanced packs below it lose (4x4096: 0.52 -> 0.96 ms).  Packs beyond
    # half the SM count in sequence x head tasks get too few windows for the
    # long member (6x500+13384: 1.64 -> 1.78 ms) and stay sequential.
    tasks = len(lengths) * heads
    if tasks > _affine_max_tasks() or 2 * tasks > crossover["sm_count"]:
        return False
    longest = max(chunk_counts)
    return (
        longest >= AFFINE_SPLIT_MIN_CHUNKS
        and _affine_split_part_count(tasks=heads, chunks=longest, **crossover) >= 2
    )


_TVM_DEVICES: dict[int, Any] = {}


def _ffi_stream_context(device):
    """FFI stream context on the current torch stream of ``device``.

    ``tvm_ffi.use_torch_stream()`` builds a ``torch.cuda.Stream`` and parses a
    device string on every entry; on the plan-cache hit path that sits before
    the first chain kernel.  The raw current-stream handle and a cached FFI
    device give the same context for a fraction of the cost (a CUDA-graph
    capture stream is the current stream as well).
    """
    import torch
    import tvm_ffi

    raw_stream_of = getattr(torch._C, "_cuda_getCurrentRawStream", None)
    if raw_stream_of is None:
        return tvm_ffi.use_torch_stream()
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    ffi_device = _TVM_DEVICES.get(index)
    if ffi_device is None:
        ffi_device = _TVM_DEVICES[index] = tvm_ffi.device(f"cuda:{index}")
    return tvm_ffi.use_raw_stream(ffi_device, raw_stream_of(index))


REBIND_INPUT_NAMES = (
    "q",
    "k",
    "v",
    "g",
    "beta",
    "out",
    "A_log",
    "dt_bias",
    "initial_state",
    "final_state",
    "cu_seqlens",
    "state_indices",
    "state_checkpoints",
    "checkpoint_cu_starts",
)


def _storage_base_address(tensor) -> int:
    """Byte address of the tensor's storage start without materializing a Storage."""
    return tensor.data_ptr() - tensor.storage_offset() * tensor.element_size()


def rebind_signature_and_facts(inputs: dict) -> tuple[tuple, tuple]:
    """One pass over the rebind inputs: (structural signature, address facts).

    The signature is the key two calls share when every caller tensor has the
    same shape, strides, dtype, device and 256-byte alignment, and the same
    tensors alias each other (``initial_state is final_state`` for an
    in-place pool, packed ``q``/``k``/``v`` slices of one projection buffer,
    ...).  The facts add the full address per tensor: equal facts mean a
    launch bound to the previous call already points at these tensors.

    This runs on every plan-cache hit before the first chain kernel, so it
    touches each tensor through the cheapest accessors only: ``torch.Size``
    is kept as the shape (it hashes and compares as a tuple), the element
    size comes from the dtype and the device index from ``get_device()``.
    """
    storage_groups: dict[int, int] = {}
    facts: list[Optional[tuple]] = []
    aliases: list[Optional[int]] = []
    addresses: list[Optional[tuple]] = []
    for name in REBIND_INPUT_NAMES:
        tensor = inputs.get(name)
        if tensor is None:
            facts.append(None)
            aliases.append(None)
            addresses.append(None)
            continue
        pointer = tensor.data_ptr()
        shape = tensor.shape
        stride = tensor.stride()
        dtype = tensor.dtype
        facts.append((shape, stride, dtype, pointer & 0xFF, tensor.get_device()))
        base = pointer - tensor.storage_offset() * dtype.itemsize
        aliases.append(storage_groups.setdefault(base, len(storage_groups)))
        addresses.append((pointer, shape, stride, dtype))
    return (tuple(facts), tuple(aliases)), tuple(addresses)


def rebind_signature(inputs: dict) -> tuple:
    """Structural key under which a prepared launch can be rebound to new inputs."""
    return rebind_signature_and_facts(inputs)[0]


@dataclass(frozen=True)
class _RebindView:
    container: str
    key: Any
    input_name: str
    byte_offset: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: Any


@dataclass(frozen=True)
class _RebindAddress:
    container: str
    key: Any
    input_name: str
    byte_offset: int


@dataclass
class RebindPlan:
    """Address-relative record of every prepared argument aliasing a caller tensor."""

    signature: tuple
    views: tuple[_RebindView, ...]
    addresses: tuple[_RebindAddress, ...]
    owned_keepalive: tuple[Any, ...]
    sub_owned_keepalive: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    # Specs grouped by caller input, so a partial rebind (``changed``) walks
    # only the moved inputs' specs instead of skipping through all of them.
    views_by_input: dict[str, tuple[_RebindView, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )
    addresses_by_input: dict[str, tuple[_RebindAddress, ...]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self):
        if not self.views_by_input:
            for spec in self.views:
                self.views_by_input.setdefault(spec.input_name, ())
                self.views_by_input[spec.input_name] += (spec,)
        if not self.addresses_by_input:
            for spec in self.addresses:
                self.addresses_by_input.setdefault(spec.input_name, ())
                self.addresses_by_input[spec.input_name] += (spec,)

    def specs_for(self, changed):
        """(views, addresses) to re-point: every spec, or only the moved inputs' specs."""
        if changed is None:
            return self.views, self.addresses
        views = []
        addresses = []
        for name in changed:
            views.extend(self.views_by_input.get(name, ()))
            addresses.extend(self.addresses_by_input.get(name, ()))
        return views, addresses


# Split-sequence affine composites hold three prepared part launches plus a
# few caller-aliasing attributes of their own; their containers are addressed
# as "<sub launch>.<container>" so one rebind plan covers the whole DAG.
AFFINE_SUB_LAUNCHES = ("_main", "_map", "_correction")
# Parts whose plan-cache rebinds may be applied after the first chain kernel.
AFFINE_DEFERRED_PARTS = ("_map", "_correction")
AFFINE_REBIND_ATTRIBUTES = (
    "_state_indices",
    "_initial_pool",
    "_initial_pool_pointer",
    "_final_pool",
    "_checkpoint_output",
    "_checkpoint_start",
    "_out_tail",
)
# Composite attributes that the apply-route kernels (pair-map producer and
# fused apply) address through prepared TMA descriptors of CALLER storage:
# the output tail (``out_tma``) and, for in-place checkpoint rows, the caller's
# checkpoint buffer (``rows_tma``).  When a plan-cache rebind moves one of
# them the composite must re-encode those descriptors before its next launch.
APPLY_ROUTE_DESCRIPTOR_ATTRIBUTES = ("_out_tail", "_checkpoint_output")


def _rebind_owner(impl, container_name: str):
    """Resolve (owner object, base container name) for a plan container."""
    if "." in container_name:
        sub_name, base = container_name.split(".", 1)
        return getattr(impl, sub_name), base
    return impl, container_name


def _rebind_containers(impl) -> dict[str, Any]:
    memo = impl.__dict__.get("_rebind_containers_memo")
    if memo is not None:
        return memo
    memo = _collect_rebind_containers(impl)
    impl._rebind_containers_memo = memo
    return memo


def _collect_rebind_containers(impl) -> dict[str, Any]:
    if hasattr(impl, "_main"):
        containers: dict[str, Any] = {}
        for sub_name in AFFINE_SUB_LAUNCHES:
            sub = getattr(impl, sub_name, None)
            if sub is None:
                continue
            for key, value in _rebind_containers(sub).items():
                containers[f"{sub_name}.{key}"] = value
        containers["attributes"] = {
            name: getattr(impl, name)
            for name in AFFINE_REBIND_ATTRIBUTES
            if getattr(impl, name, None) is not None
        }
        refreshes = getattr(impl, "_affine_input_refreshes", ())
        if refreshes:
            containers["affine_refresh_sources"] = {
                index: source for index, (_, source) in enumerate(refreshes)
            }
        return containers
    containers = {"args": impl.args}
    prepare_args = getattr(impl, "prepare_args", None)
    if prepare_args:
        containers["prepare_args"] = prepare_args
    refreshes = getattr(impl, "_token_storage_refreshes", ())
    if refreshes:
        # Sources are logical views of the caller's g/beta storage; the
        # destinations are launch-owned staging buffers.
        containers["refresh_sources"] = {
            index: source for index, (_, source) in enumerate(refreshes)
        }
    attributes = {}
    for name in ("_beta_tma_source", "_beta_tma_valid"):
        value = getattr(impl, name, None)
        if value is not None:
            attributes[name] = value
    if attributes:
        containers["attributes"] = attributes
    return containers


def capture_rebind_plan(impl, inputs: dict) -> RebindPlan:
    """Record how a freshly prepared launch aliases its caller tensors.

    Every tensor argument that shares storage with a caller tensor is stored
    as (input name, byte offset from that input's data pointer, view shape,
    view stride, dtype); every ``*_addr`` integer inside a caller storage is
    stored as (input name, byte offset).  Launch-owned buffers (dummies,
    staging copies, persistent-task metadata, TMA workspace) are not aliased
    and are left untouched by :func:`rebind_prepared_launch`.
    """
    import torch

    storages: dict[int, tuple[str, Any, int]] = {}
    for name in REBIND_INPUT_NAMES:
        tensor = inputs.get(name)
        if tensor is None:
            continue
        storage = tensor.untyped_storage()
        storages.setdefault(storage.data_ptr(), (name, tensor, storage.nbytes()))

    def classify(tensor):
        hit = storages.get(_storage_base_address(tensor))
        if hit is None:
            return None
        name, source, _ = hit
        return name, tensor.data_ptr() - source.data_ptr()

    views: list[_RebindView] = []
    addresses: list[_RebindAddress] = []
    for container_name, container in _rebind_containers(impl).items():
        for key, value in container.items():
            if isinstance(value, torch.Tensor):
                hit = classify(value)
                if hit is not None:
                    views.append(
                        _RebindView(
                            container_name,
                            key,
                            hit[0],
                            hit[1],
                            tuple(value.shape),
                            tuple(value.stride()),
                            value.dtype,
                        )
                    )
            elif (
                isinstance(value, int)
                and not isinstance(value, bool)
                and isinstance(key, str)
                and key.endswith("_addr")
            ):
                for base, (name, source, nbytes) in storages.items():
                    if base <= value < base + nbytes:
                        addresses.append(
                            _RebindAddress(
                                container_name, key, name, value - source.data_ptr()
                            )
                        )
                        break

    def owned_items(owner):
        return tuple(
            item
            for item in getattr(owner, "_keepalive", ())
            if not (isinstance(item, torch.Tensor) and classify(item) is not None)
        )

    sub_owned = {}
    if hasattr(impl, "_main"):
        for sub_name in AFFINE_SUB_LAUNCHES:
            sub = getattr(impl, sub_name, None)
            if sub is not None:
                sub_owned[sub_name] = owned_items(sub)
    return RebindPlan(
        signature=rebind_signature(inputs),
        views=tuple(views),
        addresses=tuple(addresses),
        owned_keepalive=owned_items(impl),
        sub_owned_keepalive=sub_owned,
    )


def rebind_prepared_launch(
    impl,
    plan: RebindPlan,
    inputs: dict,
    *,
    signature: tuple | None = None,
    changed: frozenset | None = None,
    addresses: tuple | None = None,
    defer_parts: tuple[str, ...] = (),
) -> bool:
    """Point a prepared launch at new caller tensors with the recorded layout.

    ``signature`` is the precomputed ``rebind_signature(inputs)``; when omitted
    it is recomputed and checked here.  Only host-side view construction
    happens in this call.  The return value reports whether any TMA-described
    argument (``*_tma`` keys) moved, in which case the caller must re-encode
    the descriptors (``prepare_descriptors``): one stream-ordered upload
    kernel, and therefore CUDA-graph capturable.  Unchanged descriptor
    sources may keep the workspace contents.  ``changed`` names the caller
    tensors whose address differs from the previous binding of this plan;
    when given, views and addresses of the other inputs are left in place
    (their tensors still hold the recorded addresses and layouts).
    ``addresses`` is the per-input ``(pointer, shape, stride, dtype)`` tuple
    from :func:`rebind_signature_and_facts` for these inputs; with it the
    rebind never re-reads tensor metadata.  ``defer_parts`` names composite
    sub launches (``"_map"``, ``"_correction"``) whose specs are recorded on
    ``impl`` and applied by the composite after its first chain kernel is
    in flight (:func:`flush_deferred_rebind`), keeping them off the host
    critical path; a pending deferred set is flushed before a new rebind.
    """
    if signature is None:
        signature = rebind_signature(inputs)
    if signature != plan.signature:
        raise ValueError("prepared launch signature does not match the new inputs")
    flush_deferred_rebind(impl)
    if addresses is None:
        addresses = rebind_signature_and_facts(inputs)[1]
    address_by_input = dict(zip(REBIND_INPUT_NAMES, addresses, strict=True))
    view_specs, address_specs = plan.specs_for(changed)
    if defer_parts:
        prefixes = tuple(f"{name}." for name in defer_parts)
        now_views: list[_RebindView] = []
        later_views: list[_RebindView] = []
        for spec in view_specs:
            (later_views if spec.container.startswith(prefixes) else now_views).append(
                spec
            )
        now_addresses: list[_RebindAddress] = []
        later_addresses: list[_RebindAddress] = []
        for spec in address_specs:
            (
                later_addresses
                if spec.container.startswith(prefixes)
                else now_addresses
            ).append(spec)
        view_specs, address_specs = now_views, now_addresses
    else:
        later_views = []
        later_addresses = []
    tma_moved = _apply_rebind_specs(
        impl, view_specs, address_specs, inputs, address_by_input, changed
    )
    new_inputs = tuple(
        inputs[name] for name in REBIND_INPUT_NAMES if inputs.get(name) is not None
    )
    impl._keepalive = plan.owned_keepalive + new_inputs
    deferred_owned = {}
    for sub_name, owned in plan.sub_owned_keepalive.items():
        if sub_name in defer_parts:
            deferred_owned[sub_name] = owned + new_inputs
            continue
        sub = getattr(impl, sub_name, None)
        if sub is not None:
            sub._keepalive = owned + new_inputs
    if later_views or later_addresses or deferred_owned:
        for spec in later_views:
            if (
                isinstance(spec.key, str)
                and spec.key.endswith("_tma")
                and (changed is None or spec.input_name in changed)
            ):
                tma_moved = True
        impl._deferred_rebind = (
            tuple(later_views),
            tuple(later_addresses),
            inputs,
            address_by_input,
            changed,
            deferred_owned,
        )
    return tma_moved


def flush_deferred_rebind(impl) -> None:
    """Apply the composite part rebinds recorded by ``defer_parts`` (no-op otherwise)."""
    pending = impl.__dict__.pop("_deferred_rebind", None)
    if pending is None:
        return
    views, addresses, inputs, address_by_input, changed, deferred_owned = pending
    _apply_rebind_specs(impl, views, addresses, inputs, address_by_input, changed)
    for sub_name, keepalive in deferred_owned.items():
        sub = getattr(impl, sub_name, None)
        if sub is not None:
            sub._keepalive = keepalive


def _apply_rebind_specs(
    impl, view_specs, address_specs, inputs: dict, address_by_input: dict, changed
) -> bool:
    """Re-point the given view/address specs; mark owners whose TMA sources moved."""
    import torch

    # The stored launch records hold the previous binding's arguments.
    _invalidate_launch_records(impl)
    containers = _rebind_containers(impl)
    tma_moved = False
    resolved: dict[tuple, Any] = {}

    def view_for(spec: _RebindView):
        cache_key = (
            spec.input_name,
            spec.byte_offset,
            spec.shape,
            spec.stride,
            spec.dtype,
        )
        tensor = resolved.get(cache_key)
        if tensor is not None:
            return tensor
        source = inputs[spec.input_name]
        _, shape, stride, dtype = address_by_input[spec.input_name]
        if (
            spec.byte_offset == 0
            and dtype == spec.dtype
            and shape == spec.shape
            and stride == spec.stride
        ):
            tensor = source
        elif dtype == spec.dtype and spec.byte_offset % dtype.itemsize == 0:
            tensor = source.as_strided(
                spec.shape,
                spec.stride,
                source.storage_offset() + spec.byte_offset // dtype.itemsize,
            )
        else:
            tensor = torch.empty(0, dtype=spec.dtype, device=source.device)
            base_bytes = source.data_ptr() - source.untyped_storage().data_ptr()
            if (base_bytes + spec.byte_offset) % tensor.element_size():
                raise ValueError("rebound view is not aligned to its element size")
            tensor.set_(
                source.untyped_storage(),
                (base_bytes + spec.byte_offset) // tensor.element_size(),
                spec.shape,
                spec.stride,
            )
        resolved[cache_key] = tensor
        return tensor

    stale_owners: list = []
    touched: set[str] = set()
    for spec in view_specs:
        touched.add(spec.container)
        container = containers[spec.container]
        replacement = view_for(spec)
        if isinstance(spec.key, str) and spec.key.endswith("_tma"):
            # With ``changed`` the moved inputs are known; otherwise compare
            # the recorded pointer with the previous binding.
            moved = (
                spec.input_name in changed
                if changed is not None
                else container[spec.key].data_ptr() != replacement.data_ptr()
            )
            if moved:
                tma_moved = True
                owner, _ = _rebind_owner(impl, spec.container)
                if owner not in stale_owners:
                    stale_owners.append(owner)
        elif (
            spec.container == "attributes"
            and spec.key in APPLY_ROUTE_DESCRIPTOR_ATTRIBUTES
        ):
            # The apply-route kernels hold prepared TMA descriptors of these
            # caller tensors (see _apply_bindings / _pairmap_bindings); a moved
            # source means the descriptors still address the previous call's
            # storage, so mark them for re-encoding in the launch stream.
            moved = (
                spec.input_name in changed
                if changed is not None
                else container[spec.key].data_ptr() != replacement.data_ptr()
            )
            if moved and getattr(impl, "_apply_route", False):
                if spec.key == "_out_tail" or impl._checkpoint_in_place:
                    tma_moved = True
                    impl._apply_descriptors_stale = True
        container[spec.key] = replacement
    for address in address_specs:
        touched.add(address.container)
        containers[address.container][address.key] = (
            address_by_input[address.input_name][0] + address.byte_offset
        )
    for container_name in touched:
        container = containers[container_name]
        owner, base = _rebind_owner(impl, container_name)
        if base == "refresh_sources":
            owner._token_storage_refreshes = tuple(
                (destination, container[index])
                for index, (destination, _) in enumerate(owner._token_storage_refreshes)
            )
        elif base == "affine_refresh_sources":
            owner._affine_input_refreshes = tuple(
                (destination, container[index])
                for index, (destination, _) in enumerate(owner._affine_input_refreshes)
            )
        elif base == "attributes":
            for name, value in container.items():
                setattr(owner, name, value)
    for owner in stale_owners:
        # Only the launches whose descriptor sources moved re-encode, inside
        # their launch stream context (see _launch_uncaptured); a composite
        # whose parts moved does not re-encode the untouched parts.
        owner._descriptors_stale = True
    return tma_moved


class FlashKDABlackwellLaunch:
    """Dispatch shared BF16/FP32 schedules and retain FP32 compatibility."""

    def __init__(self, *args, **kwargs) -> None:
        import torch

        def argument(position, name, default=None):
            return args[position] if len(args) > position else kwargs.get(name, default)

        initial_state = argument(10, "initial_state")
        final_state = argument(11, "final_state")
        compute_dtype = argument(18, "compute_dtype", "bf16")
        validate_kda_state_tensors(compute_dtype, initial_state, final_state)
        q = argument(0, "q")
        if q is not None and q.ndim == 4 and (len(args) <= 19):
            kwargs["sequence_lengths"] = _launch_sequence_lengths(
                q, argument(12, "cu_seqlens"), kwargs.get("sequence_lengths")
            )
        state_is_fp32 = compute_dtype == "tf32" or any(
            (
                state is not None and getattr(state, "dtype", None) == torch.float32
                for state in (initial_state, final_state)
            )
        )
        impl: type[FlashKDABlackwellBF16FusedLaunch]
        if _supports_affine_split_launch(args, kwargs):
            impl = FlashKDABlackwellAffineSplitLaunch
        elif state_is_fp32:
            # The BF16 BT16 and small-BH bodies are not exported; the fused
            # M128 launch routes those grids to its direct / M64 families.
            candidates: tuple[
                tuple[type[FlashKDABlackwellBF16FusedLaunch], Any], ...
            ] = ()
            if compute_dtype == "tf32":
                candidates = (
                    (FlashKDABlackwellFP32BT16Launch, _supports_fp32_bt16_launch),
                    (
                        FlashKDABlackwellFP32SmallBHLaunch,
                        _supports_fp32_small_bh_launch,
                    ),
                )
            impl = FlashKDABlackwellFP32FusedM128Launch
            for launch_cls, supports in candidates:
                if supports(args, kwargs):
                    impl = launch_cls
                    break
        else:
            impl = FlashKDABlackwellBF16FusedLaunch
        self._impl = impl(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._impl, name)

    def launch(self) -> None:
        self._impl.launch()

    def close(self) -> None:
        self._impl.close()


def _should_use_bf16_active_beta_m64(*, sm_count, num_seqs, num_heads, max_seq_len):
    """Split value ownership on sparse serving grids below affine crossover."""
    return (
        max_seq_len >= 512
        and (max_seq_len + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK
        < BF16_AFFINE_SPLIT_MIN_CHUNKS
        and (0 < 8 * num_seqs * num_heads <= sm_count)
    )


def prepare_active_beta_fwd(
    q,
    k,
    v,
    g,
    beta,
    scale,
    out,
    A_log,
    dt_bias,
    lower_bound,
    initial_state,
    final_state,
    cu_seqlens,
    state_indices,
    state_checkpoints,
    checkpoint_cu_starts,
    checkpoint_every_n_tokens: int = 64,
    use_qk_l2norm_in_kernel: bool = True,
    sequence_lengths: tuple[int, ...] | None = None,
    compute_dtype: str = "bf16",
) -> FlashKDABlackwellBF16FusedLaunch:
    """Prepare active FP32 beta using the legal runtime schedule preference."""
    if use_qk_l2norm_in_kernel is not True:
        raise ValueError("FlashKDA requires use_qk_l2norm_in_kernel=True")
    launch_cls: type[FlashKDABlackwellBF16FusedLaunch] = (
        FlashKDABlackwellFP32ActiveBetaBT16Launch
        if compute_dtype == "tf32"
        else FlashKDABlackwellFP32ActiveBetaFusedM128Launch
    )
    if compute_dtype == "bf16":
        lengths = _launch_sequence_lengths(q, cu_seqlens, sequence_lengths)
        if lengths and _should_use_bf16_active_beta_m64(
            sm_count=_device_sm_count(q.device),
            num_seqs=len(lengths),
            num_heads=int(q.shape[2]),
            max_seq_len=max(lengths),
        ):
            launch_cls = FlashKDABlackwellFP32ActiveBetaM64Launch
    args = (q, k, v, g, beta, scale, out, A_log, dt_bias, lower_bound)
    kwargs = dict(
        initial_state=initial_state,
        final_state=final_state,
        cu_seqlens=cu_seqlens,
        state_indices=state_indices,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        sequence_lengths=sequence_lengths,
        compute_dtype=compute_dtype,
    )
    if (
        compute_dtype == "tf32"
        and lower_bound is not None
        and _supports_affine_split_launch(args, kwargs)
    ):
        return FlashKDABlackwellAffineSplitLaunch(*args, active_beta_f32=True, **kwargs)
    return launch_cls(*args, **kwargs)


class PreparedBF16ActiveBetaKDA:
    """Prepared BF16 active-beta call for SM100a and SM103a.

    Routes without native FP32 active-beta input refresh stable BF16 logit
    storage on every launch, including launches captured in a CUDA graph.
    """

    compute_dtype = "bf16"

    def __init__(
        self,
        q,
        k,
        v,
        g,
        beta,
        scale,
        out,
        A_log,
        dt_bias,
        lower_bound,
        *,
        initial_state,
        final_state,
        cu_seqlens,
        sequence_lengths,
        state_indices,
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
        compute_dtype="bf16",
    ):
        import torch

        if compute_dtype != "bf16":
            raise ValueError("PreparedBF16ActiveBetaKDA requires BF16 compute")
        if not isinstance(q, torch.Tensor) or q.ndim != 4:
            raise TypeError("q must be a rank-4 torch.Tensor")
        if not isinstance(beta, torch.Tensor) or beta.ndim != 3:
            raise TypeError("beta must be a rank-3 torch.Tensor")
        if beta.dtype != torch.float32:
            raise TypeError("active beta must contain FP32 probabilities")
        batch, tokens, heads, _ = q.shape
        if beta.shape[0] != batch or beta.shape[1] < tokens or beta.shape[2] != heads:
            raise ValueError(
                "active beta must have shape [B,T_storage,H] with T_storage >= q T"
            )
        if beta.device != q.device:
            raise ValueError("active beta and q must be on the same device")
        lengths = _launch_sequence_lengths(q, cu_seqlens, sequence_lengths)
        max_seq_len = max(lengths, default=0)
        direct_schedule = bool(lengths) and (
            (heads == 12 and max_seq_len <= 256)
            or (
                max_seq_len >= 512
                and q.device.type == "cuda"
                and _should_use_bf16_active_beta_m64(
                    sm_count=_device_sm_count(q.device),
                    num_seqs=len(lengths),
                    num_heads=heads,
                    max_seq_len=max_seq_len,
                )
            )
            or (
                # The one-wave M64 value split consumes active FP32 beta
                # natively, so its grids skip the logit conversion adapter.
                # Long grids that the BF16 affine split would parallelize keep
                # the adapter path; the affine windows only accept logit beta.
                q.device.type == "cuda"
                and _should_use_bf16_one_wave_dvsplit(
                    gpu_arch=detect_gpu_arch(),
                    sm_count=_device_sm_count(q.device),
                    total_tasks=len(lengths) * heads,
                    checkpoint_every_n_tokens=int(checkpoint_every_n_tokens),
                    bounded_gate=lower_bound is not None,
                    compute_dtype="bf16",
                )
                and _affine_split_part_count(
                    sm_count=_device_sm_count(q.device),
                    tasks=len(lengths) * heads,
                    chunks=(max_seq_len + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK,
                    fp32_indexed_state=True,
                    shared_tf32_factors=False,
                    unbounded_softplus=lower_bound is None,
                )
                < 2
            )
        )
        self.direct_active_beta = bool(
            direct_schedule
            and state_indices is not None
            and isinstance(initial_state, torch.Tensor)
            and initial_state.dtype == torch.float32
            and initial_state is final_state
            and lower_bound is not None
            and checkpoint_every_n_tokens == 64
        )
        self.active_beta_logical = beta[:, :tokens, :]
        self.beta_logits_f32 = None
        self.beta_logits_bf16_logical = None
        self.beta_logits_bf16 = None
        kernel_beta = beta
        if not self.direct_active_beta:
            self.beta_logits_f32 = torch.empty(
                tuple(self.active_beta_logical.shape),
                dtype=torch.float32,
                device=beta.device,
            )
            self.beta_logits_bf16 = torch.empty_strided(
                tuple(beta.shape),
                tuple(beta.stride()),
                dtype=torch.bfloat16,
                device=beta.device,
            )
            # Only logical rows are refreshed. Padding remains non-semantic.
            self.beta_logits_bf16.fill_(float("nan"))
            self.beta_logits_bf16_logical = self.beta_logits_bf16[:, :tokens, :]
            kernel_beta = self.beta_logits_bf16
        prepare = (
            prepare_active_beta_fwd
            if self.direct_active_beta
            else FlashKDABlackwellLaunch
        )
        self.prepared = prepare(
            q,
            k,
            v,
            g,
            kernel_beta,
            scale,
            out,
            A_log,
            dt_bias,
            lower_bound,
            initial_state=initial_state,
            final_state=final_state,
            cu_seqlens=cu_seqlens,
            sequence_lengths=lengths,
            state_indices=state_indices,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            compute_dtype="bf16",
        )

    @property
    def schedule(self):
        return self.prepared.schedule

    @property
    def route(self):
        return self.prepared.route

    def launch(self):
        import torch

        if not self.direct_active_beta:
            torch.logit(self.active_beta_logical, eps=1.0e-6, out=self.beta_logits_f32)
            self.beta_logits_bf16_logical.copy_(self.beta_logits_f32)
        self.prepared.launch()

    def close(self):
        self.prepared.close()
