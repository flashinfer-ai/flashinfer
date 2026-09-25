"""
Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
Licensed under the Apache License, Version 2.0.
https://www.apache.org/licenses/LICENSE-2.0
"""

from __future__ import annotations
from functools import partial
from flashinfer.jit.cake_kda_tf32 import _factory, device_arch as detect_gpu_arch

"Canonical semantic and ABI compile axes shared by KDA schedules."
from enum import Enum


class KDAGateKind(str, Enum):
    """Gate algebra selected when a KDA schedule is traced."""

    LOWER_BOUND = "lower_bound"
    UNBOUNDED_SOFTPLUS = "unbounded_softplus"

    @property
    def trace_value(self) -> int:
        return {KDAGateKind.LOWER_BOUND: 1, KDAGateKind.UNBOUNDED_SOFTPLUS: 2}[self]


class KDAInputABI(str, Enum):
    """Input addressing contract selected independently of gate algebra."""

    GENERIC_STRIDED = "generic_strided"
    PACKED_PROJECTION = "packed_projection"

    @property
    def trace_value(self) -> int:
        return {KDAInputABI.GENERIC_STRIDED: 0, KDAInputABI.PACKED_PROJECTION: 1}[self]


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


def _build_persistent_scalar_schedule(
    sequence_lengths: list[int], num_heads: int, num_ctas: int
) -> tuple[list[int], list[int], int]:
    """LPT-bin-pack recurrent tasks and encode one word per 32-token chunk."""
    bins: list[tuple[int, int, list[tuple[int, int]]]] = [
        (0, cta, []) for cta in range(num_ctas)
    ]
    heapq.heapify(bins)
    ordered_sequences = sorted(
        range(len(sequence_lengths)),
        key=lambda sequence: (sequence_lengths[sequence] + 31) // 32,
        reverse=True,
    )
    for sequence in ordered_sequences:
        chunks = (sequence_lengths[sequence] + 31) // 32
        for head in range(num_heads):
            load, cta, tasks = heapq.heappop(bins)
            tasks.append((sequence * num_heads + head, chunks))
            heapq.heappush(bins, (load + chunks, cta, tasks))
    while True:
        light_index = min(
            range(num_ctas), key=lambda index: (bins[index][0], bins[index][1])
        )
        heavy_index = max(
            range(num_ctas), key=lambda index: (bins[index][0], -bins[index][1])
        )
        light_load, light_cta, light_tasks = bins[light_index]
        heavy_load, heavy_cta, heavy_tasks = bins[heavy_index]
        pair_tasks = light_tasks + heavy_tasks
        pair_load = light_load + heavy_load
        reachable = {0: 0}
        for task_index, (_task, chunks) in enumerate(pair_tasks):
            for load, mask in list(reachable.items()):
                reachable.setdefault(load + chunks, mask | 1 << task_index)
        split_load = min(
            reachable,
            key=lambda load: (max(load, pair_load - load), abs(pair_load - 2 * load)),
        )
        if max(split_load, pair_load - split_load) >= heavy_load:
            break
        split_mask = reachable[split_load]
        light_tasks = [
            task for index, task in enumerate(pair_tasks) if split_mask & 1 << index
        ]
        heavy_tasks = [
            task for index, task in enumerate(pair_tasks) if not split_mask & 1 << index
        ]
        bins[light_index] = (split_load, light_cta, light_tasks)
        bins[heavy_index] = (pair_load - split_load, heavy_cta, heavy_tasks)
    while num_ctas >= 3:
        ordered_bins = sorted(
            range(num_ctas), key=lambda index: (bins[index][0], bins[index][1])
        )
        light_index = ordered_bins[0]
        heavy_index = ordered_bins[-1]
        average_load = sum((load for load, _cta, _tasks in bins)) / num_ctas
        middle_index = min(
            ordered_bins[1:-1],
            key=lambda index: (
                abs(bins[index][0] - average_load),
                -max((chunks for _task, chunks in bins[index][2])),
                bins[index][1],
            ),
        )
        selected_indices = (light_index, middle_index, heavy_index)
        selected_tasks = [task for index in selected_indices for task in bins[index][2]]
        selected_load = sum((chunks for _task, chunks in selected_tasks))
        heavy_load = bins[heavy_index][0]
        if selected_load > 1024:
            break
        reachable_pairs = {(0, 0): 0}
        processed_load = 0
        for task_index, (_task, chunks) in enumerate(selected_tasks):
            next_pairs = dict(reachable_pairs)
            for (first_load, second_load), assignment in reachable_pairs.items():
                third_load = processed_load - first_load - second_load
                if first_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load + chunks, second_load),
                        assignment | 1 << 2 * task_index,
                    )
                if second_load + chunks < heavy_load:
                    next_pairs.setdefault(
                        (first_load, second_load + chunks),
                        assignment | 2 << 2 * task_index,
                    )
                if third_load + chunks >= heavy_load:
                    next_pairs.pop((first_load, second_load), None)
            reachable_pairs = next_pairs
            processed_load += chunks
        if not reachable_pairs:
            break
        (first_load, second_load), assignment = min(
            reachable_pairs.items(),
            key=lambda item: max(item[0][0], item[0][1], selected_load - sum(item[0])),
        )
        split_loads = (
            first_load,
            second_load,
            selected_load - first_load - second_load,
        )
        if max(split_loads) >= heavy_load:
            break
        split_tasks: list[list[tuple[int, int]]] = [[], [], []]
        for task_index, task in enumerate(selected_tasks):
            encoded_group = assignment >> 2 * task_index & 3
            group = 0 if encoded_group == 1 else 1 if encoded_group == 2 else 2
            split_tasks[group].append(task)
        for index, load, tasks in zip(
            selected_indices, split_loads, split_tasks, strict=True
        ):
            _old_load, cta, _old_tasks = bins[index]
            bins[index] = (load, cta, tasks)
    bins.sort(key=lambda item: item[1])
    counts = [load for load, _cta, _tasks in bins]
    stride = max(counts)
    schedule = [0] * (num_ctas * stride)
    for _load, cta, tasks in bins:
        slot = 0
        for task_id, chunks in tasks:
            for local_chunk in range(chunks):
                schedule[cta * stride + slot] = (
                    task_id | local_chunk << 10 | chunks << 18
                )
                slot += 1
    return (schedule, counts, stride)


def gpu_spec_by_sku(sku):
    if sku not in ("B200", "B300"):
        return None
    return {
        "compute": {"tensor_peak_tflops": {"bf16_dense": 2250}},
        "hbm": {"peak_bandwidth_gbps": 8000},
        "execution": {"max_threads_per_sm": 2048},
        "smem": {"kb_per_sm": 228},
        "tmem": {"cols_per_sm": 512},
    }


def _build_kda_module(factory, *args, **kwargs):
    return factory(*args, **kwargs)


'FlashKDA-compatible runtime for Blackwell schedules.\n\nMinimum architecture: sm_100a.\n\nThe public ``fwd`` boundary adds ``compute_dtype="bf16"`` to the FlashKDA\narguments. BF16 compute retains BF16/FP32 external state support; TF32 compute\nrequires FP32 initial/final state. Checkpoints remain BF16. Families without\na validated TF32 implementation report an explicit unsupported request. BF16 H12 dispatches\nbetween the chunk-16 and chunk-32 direct M128 bodies at their measured sequence\nlength crossover; sequences that fit in one 16-token tile and checkpoint\nintervals that require a physical 16-token boundary retain chunk-16.\nActive-FP32-beta H12 checkpoint64 requests on 152-SM GB300 use the M64 value\nsplit for 129-256-token residuals whose doubled task grid fits one SM wave.\nOther head counts dispatch among those direct bodies, the source static-binned\npersistent M128 body, its cross-CTA recurrence-piece specialization for\nquantization-bound uniform grids, and the M64 value-row split. FP32 state\nreuses those physical schedules as state-I/O specializations where validated.\nUnder-parallelized ultra-long fixed layouts additionally use a split-sequence\naffine-prefix DAG derived from FlashInfer PR4779; its main, map, and correction\nwindows reuse the same variable-shape direct-M128 identity. Remaining regions\nretain the source-derived compatibility fallback.\n'
import heapq
from dataclasses import dataclass, field
from functools import cache
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    import torch
HEAD_DIM = 128
BF16_M128_CHUNK = 32
BF16_N16_M128_CHUNK = 16
BF16_BETA_TMA_MIN_HEADS = 8
BF16_PERSISTENT_MIN_BALANCED_CTAS = 128
BF16_LPT_MAX_IMBALANCE_NUMERATOR = 21
BF16_LPT_MAX_IMBALANCE_DENOMINATOR = 20
BF16_GB200_LPT_MAX_IMBALANCE_NUMERATOR = 263
BF16_GB200_LPT_MAX_IMBALANCE_DENOMINATOR = 250
BF16_DIRECT_ORDER_MAX_PASSES = 1
BF16_DIRECT_INSERTION_MAX_SEQUENCES = 16
BF16_DIRECT_FIXED_CHUNK_COST = 6
H12_DIRECT_N32_MIN_SEQ_LEN = 64
H12_DIRECT_N32_MAX_SEQ_LEN = 256
H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQ_LEN = 128
BF16_N32_REGISTER_INVERSE_MIN_SEQ_LEN = 256
INDEPENDENT_DVSPLIT_CTAS = 2
INDEPENDENT_DVSPLIT_MIN_SEQ_LEN = 512
# Architectures whose one-wave BF16 grid sweep selected the M64 value split over
# every direct M128 tile (see _should_use_bf16_one_wave_dvsplit).
BF16_ONE_WAVE_DVSPLIT_ARCHES = ("sm_100a", "sm_103a")
BT16_CHUNK = 16
BT16_VALUE_SPLITS = 2
TF32_BT16_PREP_RESIDENT_CTAS = 6
TF32_BT16_CHECKPOINT_TASK_DIVISOR = 14
BT16_GENERAL_LOW_WORK_CHUNKS_PER_PREP_CTA = 2
BT16_GENERAL_HIGH_WORK_CHUNKS_PER_PREP_CTA = 8
BT16_GENERAL_HIGH_WORK_MIN_CHUNK_HEADS = 16384
BT16_H12_CHUNKS_PER_PREP_CTA = 4
BT16_H12_CPC1_MAX_TOTAL_CHUNKS = 128
BT16_PREP_WAVE_QUANT_MIN_WAVES = 8
BT16_PREP_WAVE_QUANT_MIN_RETAINED_PERCENT = 98
BT16_DENSE_PREP_WAVES = 5
BT16_DENSE_MIN_HEADS = 60
BT16_DENSE_MAX_HEADS = 64
BT16_DENSE_MIN_SEQ_LEN = 4096
BT16_N16_ONE_CHAIN_WAVE_MIN_SEQ_LEN = 512
BT16_N16_TWO_CHAIN_WAVE_MIN_SEQ_LEN = 3072
BT16_N16_MULTI_WAVE_MIN_SEQ_LEN = 512
BT16_N16_MAX_DIRECT_WAVES = 3
BT16_MID_MIN_SEQ_LEN = 4096
BT16_LONG_MIN_SEQ_LEN = 65536
BT16_MID_MAX_TASKS = 32
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
SMALL_BH_RING_STAGES = 35
SMALL_BH_PACKET_ELEMS = HEAD_DIM
SMALL_BH_PACKET_ROWS = 31488 // (SMALL_BH_PACKET_ELEMS * 2)
SMALL_BH_MAX_TASKS = 8
SMALL_BH_MIN_SEQ_LEN = 2048
BF16_N32_PREDICTION_FIRST_MIN_HEADS = 24
BF16_N32_PREDICTION_FIRST_MIXED_MIN_SEQ_LEN = 512
SOURCE_VTILE_PERSISTENT_WORKERS = 128
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


@dataclass(frozen=True)
class _PersistentM128Roofline:
    """Resolved occupancy and critical-path lower bounds in nanoseconds."""

    resident_ctas_per_sm: int
    worker_count: int
    handoff_count: int
    chunk_ns: float
    state_transfer_ns: float
    task_refill_ns: float
    direct_ns: float
    piece_ns: float


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


def _pair_packed_beta_tma_source(beta_flat, *, total_tokens: int, num_heads: int):
    """Return an aliasing H12 beta view with a TensorMap-legal row pitch.

    Pairing two consecutive token rows turns dense ``[T, 12]`` storage into
    ``[T/2, 24]`` without moving data.  Its 48-byte row pitch is TensorMap
    encodable, and the direct kernel rounds each sequence start down to the pair
    boundary before selecting the original token/head element in shared
    memory.  Unsupported layouts return ``None`` and retain the padded-copy
    fallback.
    """
    if tuple(beta_flat.shape) != (total_tokens, num_heads):
        raise ValueError("beta_flat must have shape [total_tokens, num_heads]")
    if (
        num_heads != 12
        or total_tokens % 2 != 0
        or (not beta_flat.is_contiguous())
        or (beta_flat.data_ptr() % 16 != 0)
    ):
        return None
    return beta_flat.view(total_tokens // 2, 24)


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

    The standalone serving kernel consumes the physical beta row pitch through
    its explicit ``beta_token_stride`` scalar.  The generic FFI shim only
    needs the tensor's dtype, device, and data pointer for that argument, but a
    raw pointer has a contiguous-by-default host contract.  A one-element view
    preserves the exact data pointer while satisfying that transport contract;
    the caller-owned strided tensor remains alive and is the device data source.
    """
    if tensor.is_contiguous():
        return tensor
    carrier = tensor.as_strided((1,), (1,))
    if carrier.data_ptr() != tensor.data_ptr():
        raise RuntimeError("raw-pointer carrier must preserve the tensor data pointer")
    return carrier


def _should_use_independent_dvsplit(
    *,
    gpu_arch: str,
    sm_count: int,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    max_seq_len: int,
) -> bool:
    """Select M64 when its doubled fixed-layout grid remains one resident wave."""
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and fixed_layout
        and (num_seqs == 1)
        and (max_seq_len >= INDEPENDENT_DVSPLIT_MIN_SEQ_LEN)
        and (INDEPENDENT_DVSPLIT_CTAS * num_heads <= sm_count)
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
    n16_short_four_stage: bool = False,
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
        and (not n16_short_four_stage)
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
    n16_short_four_stage: bool = False,
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
        and (not n16_short_four_stage)
    )


def _should_use_source_vtile_direct(
    *,
    gpu_arch: str,
    sm_count: int,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
    max_seq_len: int,
) -> bool:
    """Select the one-wave M128 schedule for long dense H96 work."""
    return (
        gpu_arch == "sm_103a"
        and fixed_layout
        and uniform_sequences
        and (num_heads == 96)
        and (num_seqs * num_heads <= sm_count)
        and (max_seq_len >= 4096)
    )


def _should_use_source_vtile_persistent(
    *,
    gpu_arch: str,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
    max_seq_len: int,
) -> bool:
    """Select the persistent M128 schedule by work-per-CTA bucket."""
    total_tasks = num_seqs * num_heads
    return (
        gpu_arch == "sm_103a"
        and (not fixed_layout)
        and uniform_sequences
        and (num_heads in (64, 96))
        and (total_tasks % SOURCE_VTILE_PERSISTENT_WORKERS == 0)
        and (total_tasks // SOURCE_VTILE_PERSISTENT_WORKERS in (4, 6))
        and (max_seq_len >= 512)
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


def _wave_quantized_bt16_prepare_ctas(
    *, rectangular_ctas: int, num_heads: int, sm_count: int
) -> int:
    """Trim a nearly complete final prepare wave without changing ownership.

    The flattened scheduler balances an arbitrary CTA count independently
    within every head, so a small reduction only gives a few CTAs one extra
    chunk.  When at least 98% of the rectangular grid remains, ending on a
    complete hardware wave is faster than launching the sparse residual wave.
    Short grids retain their original parallelism.
    """
    if rectangular_ctas < BT16_PREP_WAVE_QUANT_MIN_WAVES * sm_count:
        return rectangular_ctas
    full_wave_ctas = rectangular_ctas // sm_count * sm_count
    if (
        full_wave_ctas < num_heads
        or full_wave_ctas * 100
        < rectangular_ctas * BT16_PREP_WAVE_QUANT_MIN_RETAINED_PERCENT
    ):
        return rectangular_ctas
    return full_wave_ctas


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


def _should_use_bt16_prepare_chain(
    *,
    gpu_arch: str,
    sm_count: int,
    num_seqs: int,
    num_heads: int,
    max_seq_len: int,
    n16_alternative: bool = False,
) -> bool:
    """Select the decomposed BT16 path beyond measured route crossovers.

    Preparation is chunk parallel, while two independent M64 CTAs own each
    recurrent state.  Against the pair-packed H12 N16 direct kernel, BT16 wins
    from 512 tokens while its two value CTAs still fit in the same number of
    waves as direct M128.  Once the split chain adds a wave, uniform H12 work
    instead uses the variable-shape N32 direct kernel; nonuniform work retains
    the measured 3,072-token two-wave crossover.  Beyond two chain waves, N32
    also takes over at 512 tokens instead of paying the split quantization.
    Against the fixed/packed general families it wins from 4,096 tokens through
    32 one-wave tasks.  The existing owner/helper family remains faster below
    65,536 tokens for at most eight tasks. Partial chunks are admissible: the
    prepare kernel zero-extends their recurrence factors and the chain drops
    invalid output rows, so alignment is not a schedule-selection axis.
    """
    total_tasks = num_seqs * num_heads
    if n16_alternative:
        chain_waves = (BT16_VALUE_SPLITS * total_tasks + sm_count - 1) // sm_count
        if chain_waves <= 1:
            min_seq_len = BT16_N16_ONE_CHAIN_WAVE_MIN_SEQ_LEN
        elif chain_waves == 2:
            min_seq_len = BT16_N16_TWO_CHAIN_WAVE_MIN_SEQ_LEN
        else:
            min_seq_len = BT16_N16_MULTI_WAVE_MIN_SEQ_LEN
        max_tasks = BT16_N16_MAX_DIRECT_WAVES * sm_count
    elif total_tasks <= SMALL_BH_MAX_TASKS:
        min_seq_len = BT16_LONG_MIN_SEQ_LEN
        max_tasks = SMALL_BH_MAX_TASKS
    else:
        min_seq_len = BT16_MID_MIN_SEQ_LEN
        max_tasks = BT16_MID_MAX_TASKS
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and 0 < total_tasks <= max_tasks
        and (max_seq_len >= min_seq_len)
        and (n16_alternative or BT16_VALUE_SPLITS * total_tasks <= sm_count)
    )


def _should_use_bt16_dense_wavefront(
    *,
    gpu_arch: str,
    sm_count: int,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    max_seq_len: int,
) -> bool:
    """Select decomposed preparation when its dense chain stays one wave.

    The material schedule change replaces the fused five-stage M64 producer
    with a chunk-parallel factor kernel and the standalone two-way M64 chain.
    H60--H64 fills 120--128 of the measured 148/152 SMs without crossing a
    chain wave; five exact prepare waves then avoid the rectangular-grid
    quantization cliff at 12 CTAs per head.
    """
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and fixed_layout
        and (num_seqs == 1)
        and (BT16_DENSE_MIN_HEADS <= num_heads <= BT16_DENSE_MAX_HEADS)
        and (max_seq_len >= BT16_DENSE_MIN_SEQ_LEN)
        and (BT16_VALUE_SPLITS * num_heads <= sm_count)
    )


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
        )


def _uses_measured_sm100_persistent_policy(*, gpu_arch: str, sm_count: int) -> bool:
    """Return whether an exact measured SM100 persistent policy applies."""
    return gpu_arch == "sm_100a" and sm_count in (148, 152)


def _uniform_persistent_worker_count(total_tasks: int, *, worker_cap: int) -> int:
    """Choose a nearly full one-wave grid with equal grid-stride trip counts."""
    if total_tasks <= 0 or worker_cap <= 0:
        raise ValueError("total_tasks and worker_cap must be positive")
    if total_tasks <= worker_cap:
        return total_tasks
    trips = (total_tasks + worker_cap - 1) // worker_cap
    if total_tasks % trips == 0:
        balanced_workers = total_tasks // trips
        if balanced_workers >= BF16_PERSISTENT_MIN_BALANCED_CTAS:
            return balanced_workers
    return worker_cap


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
    import torch
    from array import array

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


def _make_lpt_task_bins(
    ordered_seq_lens: tuple[int, ...], *, num_heads: int, worker_count: int
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Greedily assign descending-length sequence/head tasks to CTA bins."""
    total_tasks = len(ordered_seq_lens) * num_heads
    if not ordered_seq_lens or num_heads <= 0 or (not 0 < worker_count <= total_tasks):
        raise ValueError("LPT bins require positive sequence/head/task counts")
    bins: list[list[int]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    # Least-loaded worker with the lowest index; a heap keyed on
    # (load, index) reproduces the linear-scan argmin in O(log W) per task.
    heap = [(0, index) for index in range(worker_count)]
    for ordered_seq_idx, seq_len in enumerate(ordered_seq_lens):
        chunk_count = (seq_len + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK
        for head_idx in range(num_heads):
            load, worker_idx = heapq.heappop(heap)
            bins[worker_idx].append(ordered_seq_idx * num_heads + head_idx)
            loads[worker_idx] = load + chunk_count
            heapq.heappush(heap, (load + chunk_count, worker_idx))
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_tasks in bins:
        task_ids.extend(worker_tasks)
        task_offsets.append(len(task_ids))
    return (tuple(task_ids), tuple(task_offsets), tuple(loads))


def _make_uniform_head_grouped_bins(
    *, num_seqs: int, num_heads: int, worker_count: int
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Partition head-major uniform tasks into contiguous balanced CTA bins."""
    total_tasks = num_seqs * num_heads
    if num_seqs <= 0 or num_heads <= 0 or (not 0 < worker_count <= total_tasks):
        raise ValueError("head-grouped bins require positive sequence/head/task counts")
    task_ids: list[int] = []
    task_offsets = [0]
    for worker_idx in range(worker_count):
        begin = worker_idx * total_tasks // worker_count
        end = (worker_idx + 1) * total_tasks // worker_count
        for head_major_idx in range(begin, end):
            head_idx, ordered_seq_idx = divmod(head_major_idx, num_seqs)
            task_ids.append(ordered_seq_idx * num_heads + head_idx)
        task_offsets.append(len(task_ids))
    return (tuple(task_ids), tuple(task_offsets))


def _make_uniform_piece_task_bins(
    *, num_seqs: int, num_heads: int, seq_len: int, worker_count: int
) -> tuple[
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    int,
    tuple[int, ...],
]:
    """Split quantization-bound uniform chains across persistent CTA bins.

    Whole-chain LPT leaves ``extra_tasks`` workers with one additional full
    recurrence chain.  Remove exactly those overflow chains, divide each into
    as many balanced chunk-aligned pieces as the worker/task geometry permits,
    and stagger successive pieces one whole chain later in distinct CTA bins.
    This preserves an acyclic recurrence DAG while reducing the integer
    makespan.  The same runtime scheduler covers both H64 and H96 uniform work;
    the resulting piece count is derived from the resolved grid rather than an
    exact shape guard.

    The four metadata arrays after ``task_offsets`` are per-dispatch-entry
    token starts/counts and optional source/destination handoff slots.  A
    negative handoff index denotes the original initial/final state boundary.
    """
    total_tasks = num_seqs * num_heads
    if (
        num_seqs <= 0
        or num_heads <= 0
        or seq_len <= 0
        or (worker_count <= 0)
        or (worker_count > total_tasks)
    ):
        raise ValueError("uniform piece bins require positive resolved work")
    chunk_count = (seq_len + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK
    bins: list[list[tuple[int, int, int, int, int]]] = [[] for _ in range(worker_count)]
    loads = [0] * worker_count
    # Uniform whole-chain LPT: task t lands on worker t % W (least loaded,
    # lowest index), which is exactly the linear-scan argmin result.
    for task_idx in range(total_tasks):
        worker_idx = task_idx % worker_count
        bins[worker_idx].append((task_idx, 0, seq_len, -1, -1))
        loads[worker_idx] += chunk_count
    base_tasks, extra_tasks = divmod(total_tasks, worker_count)
    piece_count = (
        min(base_tasks, worker_count // extra_tasks, chunk_count) if extra_tasks else 1
    )
    if piece_count >= 2:
        peak_load = (base_tasks + 1) * chunk_count
        peak_slots = [
            worker_idx for worker_idx, load in enumerate(loads) if load == peak_load
        ]
        if len(peak_slots) != extra_tasks:
            raise RuntimeError("uniform LPT peak count did not match task remainder")
        overflow_tasks = []
        for worker_idx in peak_slots:
            task = bins[worker_idx].pop()
            loads[worker_idx] -= chunk_count
            overflow_tasks.append(task[0])
        handoff_count = 0
        chunk_base = chunk_count // piece_count
        chunk_remainder = chunk_count % piece_count
        chunk_cuts = [0]
        for piece_idx in range(piece_count):
            piece_chunks = chunk_base + int(piece_idx >= piece_count - chunk_remainder)
            chunk_cuts.append(chunk_cuts[-1] + piece_chunks)
        for overflow_idx, task_idx in enumerate(overflow_tasks):
            handoffs = tuple(range(handoff_count, handoff_count + piece_count - 1))
            handoff_count += piece_count - 1
            for piece_idx in range(piece_count):
                chunk_start = chunk_cuts[piece_idx]
                chunk_end = chunk_cuts[piece_idx + 1]
                token_start = chunk_start * BF16_M128_CHUNK
                token_end = min(seq_len, chunk_end * BF16_M128_CHUNK)
                src = -1 if piece_idx == 0 else handoffs[piece_idx - 1]
                dst = -1 if piece_idx + 1 == piece_count else handoffs[piece_idx]
                worker_idx = piece_idx * extra_tasks + overflow_idx
                insert_at = min(1 + piece_idx, len(bins[worker_idx]))
                bins[worker_idx].insert(
                    insert_at,
                    (task_idx, token_start, token_end - token_start, src, dst),
                )
                loads[worker_idx] += chunk_end - chunk_start
    else:
        handoff_count = 0
    task_ids: list[int] = []
    task_token_starts: list[int] = []
    task_token_counts: list[int] = []
    task_state_sources: list[int] = []
    task_state_destinations: list[int] = []
    task_offsets = [0]
    for worker_tasks in bins:
        for task_idx, token_start, token_count, src, dst in worker_tasks:
            task_ids.append(task_idx)
            task_token_starts.append(token_start)
            task_token_counts.append(token_count)
            task_state_sources.append(src)
            task_state_destinations.append(dst)
        task_offsets.append(len(task_ids))
    return (
        tuple(task_ids),
        tuple(task_offsets),
        tuple(task_token_starts),
        tuple(task_token_counts),
        tuple(task_state_sources),
        tuple(task_state_destinations),
        handoff_count,
        tuple(loads),
    )


def _persistent_m128_roofline(
    *,
    gpu_arch: str,
    sm_count: int,
    num_seqs: int,
    num_heads: int,
    seq_len: int,
    use_initial_state: bool,
    store_final_state: bool,
) -> _PersistentM128Roofline | None:
    """Resolve occupancy and compare direct/piece roofline critical paths.

    Peak rates and per-SM capacities come from ``hardware.json``.  The
    physical schedule contract supplies its threads, SMEM, TMEM, tensor FLOPs,
    streaming bytes, and state footprint.  Equal uniform direct tasks execute
    in hardware waves; recurrence pieces execute on persistent resident CTAs,
    so their estimate is the longest path through both CTA-order and state
    handoff edges.  No input shape is used as a policy identity.
    """
    sku = {"sm_100a": "B200", "sm_103a": "B300"}.get(gpu_arch)
    if sku is None:
        return None
    if sm_count <= 0 or num_seqs <= 0 or num_heads <= 0 or (seq_len <= 0):
        raise ValueError("persistent-M128 roofline requires resolved positive extents")
    CHUNK_TOKENS = 32
    PERSISTENT_TASK_REFILL_CHUNKS = 2
    SMEM_BYTES_PER_CTA = 220672
    STATE_BYTES = 32768
    STREAM_BYTES_PER_CHUNK = 41024
    TENSOR_FLOPS_PER_CHUNK = 3407872
    THREADS_PER_CTA = 1024
    TMEM_COLS_PER_CTA = 256
    spec = gpu_spec_by_sku(sku)
    if spec is None:
        raise RuntimeError(f"missing hardware.json roofline entry for {sku}")
    try:
        peak_tflops = float(spec["compute"]["tensor_peak_tflops"]["bf16_dense"])
        peak_gbps = float(spec["hbm"]["peak_bandwidth_gbps"])
        max_threads_per_sm = int(spec["execution"]["max_threads_per_sm"])
        smem_per_sm = int(spec["smem"]["kb_per_sm"]) * 1024
        tmem_cols_per_sm = int(spec["tmem"]["cols_per_sm"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"incomplete persistent-M128 hardware model for {sku}"
        ) from exc
    if (
        min(peak_tflops, peak_gbps, max_threads_per_sm, smem_per_sm, tmem_cols_per_sm)
        <= 0
    ):
        raise RuntimeError(f"non-positive persistent-M128 hardware model for {sku}")
    resident_ctas_per_sm = min(
        max_threads_per_sm // THREADS_PER_CTA,
        smem_per_sm // SMEM_BYTES_PER_CTA,
        tmem_cols_per_sm // TMEM_COLS_PER_CTA,
    )
    if resident_ctas_per_sm <= 0:
        raise RuntimeError(
            f"persistent-M128 schedule is not resident on {sku}: threads={THREADS_PER_CTA}, smem={SMEM_BYTES_PER_CTA}, tmem_cols={TMEM_COLS_PER_CTA}"
        )
    worker_count = sm_count * resident_ctas_per_sm
    total_tasks = num_seqs * num_heads
    if total_tasks <= worker_count:
        return None
    (
        _task_ids,
        task_offsets,
        _token_starts,
        token_counts,
        state_sources,
        state_destinations,
        handoff_count,
        _loads,
    ) = _make_uniform_piece_task_bins(
        num_seqs=num_seqs,
        num_heads=num_heads,
        seq_len=seq_len,
        worker_count=worker_count,
    )
    if handoff_count == 0:
        return None
    worker_flops_per_ns = peak_tflops * 1000.0 / worker_count
    worker_bytes_per_ns = peak_gbps / worker_count
    chunk_ns = max(
        TENSOR_FLOPS_PER_CHUNK / worker_flops_per_ns,
        STREAM_BYTES_PER_CHUNK / worker_bytes_per_ns,
    )
    state_transfer_ns = STATE_BYTES / worker_bytes_per_ns
    task_refill_ns = PERSISTENT_TASK_REFILL_CHUNKS * chunk_ns
    chunks_per_task = (seq_len + CHUNK_TOKENS - 1) // CHUNK_TOKENS
    direct_task_ns = chunks_per_task * chunk_ns
    if use_initial_state:
        direct_task_ns += state_transfer_ns
    if store_final_state:
        direct_task_ns += state_transfer_ns
    direct_ns = (total_tasks + worker_count - 1) // worker_count * direct_task_ns
    entry_count = len(token_counts)
    edges: list[set[int]] = [set() for _ in range(entry_count)]
    indegree = [0] * entry_count

    def add_edge(source: int, destination: int) -> None:
        if destination not in edges[source]:
            edges[source].add(destination)
            indegree[destination] += 1

    for worker_idx in range(worker_count):
        begin = task_offsets[worker_idx]
        end = task_offsets[worker_idx + 1]
        for entry_idx in range(begin + 1, end):
            add_edge(entry_idx - 1, entry_idx)
    handoff_producers = {
        destination: entry_idx
        for entry_idx, destination in enumerate(state_destinations)
        if destination >= 0
    }
    if len(handoff_producers) != handoff_count:
        raise RuntimeError("piece roofline did not resolve every handoff producer")
    for entry_idx, source in enumerate(state_sources):
        if source >= 0:
            try:
                producer = handoff_producers[source]
            except KeyError as exc:
                raise RuntimeError(
                    f"piece roofline did not resolve handoff source {source}"
                ) from exc
            add_edge(producer, entry_idx)
    ready = [entry_idx for entry_idx, degree in enumerate(indegree) if degree == 0]
    heapq.heapify(ready)
    worker_first_entries = frozenset(task_offsets[:-1])
    earliest_start = [0.0] * entry_count
    finish = [0.0] * entry_count
    visited = 0
    while ready:
        entry_idx = heapq.heappop(ready)
        duration = (
            (token_counts[entry_idx] + CHUNK_TOKENS - 1) // CHUNK_TOKENS * chunk_ns
        )
        if entry_idx not in worker_first_entries:
            duration += task_refill_ns
        if state_sources[entry_idx] >= 0 or use_initial_state:
            duration += state_transfer_ns
        if state_destinations[entry_idx] >= 0 or store_final_state:
            duration += state_transfer_ns
        finish[entry_idx] = earliest_start[entry_idx] + duration
        visited += 1
        for successor in edges[entry_idx]:
            earliest_start[successor] = max(
                earliest_start[successor], finish[entry_idx]
            )
            indegree[successor] -= 1
            if indegree[successor] == 0:
                heapq.heappush(ready, successor)
    if visited != entry_count:
        raise RuntimeError("piece roofline dependency graph contains a cycle")
    return _PersistentM128Roofline(
        resident_ctas_per_sm=resident_ctas_per_sm,
        worker_count=worker_count,
        handoff_count=handoff_count,
        chunk_ns=chunk_ns,
        state_transfer_ns=state_transfer_ns,
        task_refill_ns=task_refill_ns,
        direct_ns=direct_ns,
        piece_ns=max(finish),
    )


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
    """Select recurrence pieces when their occupancy-aware roofline wins."""
    if not uniform_sequences or max_seq_len <= 0:
        return False
    estimate = _persistent_m128_roofline(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        num_seqs=num_seqs,
        num_heads=num_heads,
        seq_len=max_seq_len,
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


def _lpt_bins_are_balanced(loads: tuple[int, ...]) -> bool:
    """Return whether static bins are close enough to replace dynamic CTA scheduling."""
    return (
        bool(loads)
        and max(loads) * BF16_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
        <= sum(loads) * BF16_LPT_MAX_IMBALANCE_NUMERATOR
    )


def _should_use_lpt_persistent(
    *, gpu_arch: str, sm_count: int, num_heads: int, loads: tuple[int, ...]
) -> bool:
    """Select the measured H96 LPT route on exact SM100 device classes."""
    if (
        not _uses_measured_sm100_persistent_policy(gpu_arch=gpu_arch, sm_count=sm_count)
        or num_heads != 96
    ):
        return False
    if sm_count == 152:
        return (
            bool(loads)
            and max(loads) * BF16_GB200_LPT_MAX_IMBALANCE_DENOMINATOR * len(loads)
            <= sum(loads) * BF16_GB200_LPT_MAX_IMBALANCE_NUMERATOR
        )
    return _lpt_bins_are_balanced(loads)


def _should_use_scalar_chunk_lpt(
    *,
    gpu_arch: str,
    sm_count: int,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
    max_seq_len: int,
) -> bool:
    """Select the complete-chain LPT schedule on mixed dense work.

    This is one variable-shape physical schedule: the host tile scheduler
    assigns complete ``(sequence, head)`` recurrence chains to a one-wave CTA
    grid, while sequence lengths, head count, schedule stride, state slot, and
    state slot stride remain runtime values.  The range avoids both one-wave
    inputs, where direct CTAs are already balanced, and very long chains whose
    per-CTA serial work loses to the direct scheduler.
    """
    total_tasks = num_seqs * num_heads
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and (not uniform_sequences)
        and (num_heads in (64, 96))
        and (max_seq_len > 0)
        and (2 * sm_count <= total_tasks < 1024)
        and ((max_seq_len + BF16_M128_CHUNK - 1) // BF16_M128_CHUNK < 256)
    )


def _direct_m128_route(*, num_heads: int, max_seq_len: int = 0) -> str:
    """Resolve the direct tile from head and sequence schedule economics."""
    return (
        BF16_ROUTE_DIRECT_M128_N16
        if num_heads == 12 or 0 < max_seq_len <= BF16_N16_M128_CHUNK
        else BF16_ROUTE_DIRECT_M128
    )


def _validate_n16_short_four_stage_contract(
    *,
    enabled: bool,
    state_dtype_is_fp32: bool,
    num_heads: int,
    max_seq_len: int,
    checkpoint_every_n_tokens: int,
    bounded_gate: bool,
    indexed_state_pool: bool,
    in_place_state_pool: bool,
) -> None:
    """Reject any input outside the private short-serving S4 experiment."""
    if not enabled:
        return
    if not state_dtype_is_fp32:
        raise ValueError("short N16 S4 requires FP32 state I/O")
    if num_heads != 12:
        raise ValueError("short N16 S4 requires exactly 12 heads")
    if max_seq_len <= 0 or max_seq_len > 64:
        raise ValueError("short N16 S4 requires 1 <= max_seq_len <= 64")
    if checkpoint_every_n_tokens != 64:
        raise ValueError("short N16 S4 requires checkpoint_every_n_tokens=64")
    if not bounded_gate:
        raise ValueError("short N16 S4 requires the bounded gate contract")
    if not indexed_state_pool:
        raise ValueError("short N16 S4 requires indexed state-pool routing")
    if not in_place_state_pool:
        raise ValueError("short N16 S4 requires one in-place initial/final state pool")


def _should_use_h12_direct_n32(
    *, gpu_arch: str, num_heads: int, max_seq_len: int
) -> bool:
    """Select the measured H12 range where two N16 chunks lose to one N32."""
    return (
        gpu_arch in ("sm_100a", "sm_103a")
        and num_heads == 12
        and (H12_DIRECT_N32_MIN_SEQ_LEN <= max_seq_len <= H12_DIRECT_N32_MAX_SEQ_LEN)
    )


def _constrained_direct_m128_route(
    *,
    gpu_arch: str,
    num_heads: int,
    max_seq_len: int,
    force_direct_m128: bool,
    force_direct_m128_n32: bool,
    unbounded_softplus: bool,
    n16_short_four_stage: bool,
    state_dtype_is_fp32: bool,
    indexed_state_pool: bool,
    checkpoint_every_n_tokens: int,
    preferred_route: str,
) -> str:
    """Resolve direct-only serving requests without bypassing H12 N32 policy.

    Indexed state, checkpoint output, and independently strided beta all force
    the direct kernel family, but they do not require its N16 physical tile.
    Retain N16 for the qualified 64-token row and its private S4 experiment;
    longer short residuals and dense grids retain the unconstrained N32
    preference. Checkpoint and beta storage constraints select a legal family;
    they must not reset its physical tile preference to N16.
    """
    if force_direct_m128_n32 or unbounded_softplus:
        return BF16_ROUTE_DIRECT_M128
    if (
        not n16_short_four_stage
        and (not force_direct_m128)
        and state_dtype_is_fp32
        and indexed_state_pool
        and (checkpoint_every_n_tokens in (0, 64))
        and (max_seq_len > 64)
        and (checkpoint_every_n_tokens == 64 or max_seq_len >= 4 * BF16_M128_CHUNK)
        and (
            preferred_route == BF16_ROUTE_DIRECT_M128
            or _should_use_h12_direct_n32(
                gpu_arch=gpu_arch, num_heads=num_heads, max_seq_len=max_seq_len
            )
        )
    ):
        return BF16_ROUTE_DIRECT_M128
    return _direct_m128_route(num_heads=num_heads, max_seq_len=max_seq_len)


def _requires_exact_n16_recurrence(
    *,
    sm_count: int,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
) -> bool:
    """Select the measured N16 graph for the 148-SM H96/N128 holdout."""
    return (
        sm_count == 148
        and (not fixed_layout)
        and (num_seqs == 128)
        and (num_heads == 96)
        and uniform_sequences
    )


def _select_bf16_route(
    *,
    gpu_arch: str,
    sm_count: int,
    fixed_layout: bool,
    num_seqs: int,
    num_heads: int,
    uniform_sequences: bool,
    lpt_loads: tuple[int, ...],
    max_seq_len: int = 0,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> str:
    """Select one material BF16 schedule family from resolved host metadata."""
    direct_route = _direct_m128_route(num_heads=num_heads, max_seq_len=max_seq_len)
    if num_heads == 64 and _should_use_independent_dvsplit(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_M64
    if _should_use_source_vtile_direct(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_SOURCE_VTILE_M128
    if _should_use_source_vtile_persistent(
        gpu_arch=gpu_arch,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_SOURCE_VTILE_M128
    if _should_use_bt16_dense_wavefront(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_BT16_M64
    if direct_route == BF16_ROUTE_DIRECT_M128_N16:
        if _should_use_h12_direct_n32(
            gpu_arch=gpu_arch, num_heads=num_heads, max_seq_len=max_seq_len
        ):
            return BF16_ROUTE_DIRECT_M128
        total_tasks = num_seqs * num_heads
        direct_waves = (total_tasks + sm_count - 1) // sm_count
        chain_waves = (BT16_VALUE_SPLITS * total_tasks + sm_count - 1) // sm_count
        if (
            gpu_arch in ("sm_100a", "sm_103a")
            and uniform_sequences
            and (max_seq_len > H12_DIRECT_N32_MAX_SEQ_LEN)
            and (chain_waves > direct_waves)
        ):
            return BF16_ROUTE_DIRECT_M128
        if total_tasks > 2 * sm_count and max_seq_len >= 512:
            return BF16_ROUTE_DIRECT_M128
        if _should_use_bt16_prepare_chain(
            gpu_arch=gpu_arch,
            sm_count=sm_count,
            num_seqs=num_seqs,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            n16_alternative=True,
        ):
            return BF16_ROUTE_BT16_M64
        return direct_route
    if _should_use_bt16_prepare_chain(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        num_seqs=num_seqs,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_BT16_M64
    if _should_use_small_bh_owner_helper(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        num_seqs=num_seqs,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_SMALL_BH_M128
    if _requires_exact_n16_recurrence(
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
    ):
        return BF16_ROUTE_DIRECT_M128_N16
    if _should_use_independent_dvsplit(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_M64
    if _should_use_scalar_chunk_lpt(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_seq_len=max_seq_len,
    ):
        return BF16_ROUTE_SCALAR_CHUNK_LPT_M128
    total_tasks = num_seqs * num_heads
    if _should_use_uniform_piece_persistent(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        max_seq_len=max_seq_len,
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    ):
        return BF16_ROUTE_PIECE_M128
    if (
        _uses_measured_sm100_persistent_policy(gpu_arch=gpu_arch, sm_count=sm_count)
        and num_heads in (64, 96)
        and uniform_sequences
        and (total_tasks > sm_count)
    ):
        return BF16_ROUTE_HEAD_GROUPED_M128
    if (
        not uniform_sequences
        and total_tasks > sm_count
        and _should_use_lpt_persistent(
            gpu_arch=gpu_arch, sm_count=sm_count, num_heads=num_heads, loads=lpt_loads
        )
    ):
        return BF16_ROUTE_LPT_M128
    return direct_route


def select_bf16_schedule_route(
    *,
    gpu_arch: str,
    sm_count: int,
    fixed_layout: bool,
    sequence_lengths: tuple[int, ...],
    num_heads: int,
    use_initial_state: bool = True,
    store_final_state: bool = True,
) -> str:
    """Select one physical BF16 schedule from runtime-resolved shape metadata.

    This is the canonical metadata adapter for callers that need to share the
    production dispatch policy without reproducing its shape guards. Shape
    values select among materially different schedule families; they do not
    create per-shape kernel identities.
    """
    if sm_count <= 0 or num_heads <= 0:
        raise ValueError("sm_count and num_heads must be positive")
    if not sequence_lengths or any((length <= 0 for length in sequence_lengths)):
        raise ValueError("sequence_lengths must contain positive lengths")
    num_seqs = len(sequence_lengths)
    uniform_sequences = len(set(sequence_lengths)) == 1
    total_tasks = num_seqs * num_heads
    lpt_loads: tuple[int, ...] = ()
    if (
        _uses_measured_sm100_persistent_policy(gpu_arch=gpu_arch, sm_count=sm_count)
        and (not uniform_sequences)
        and (total_tasks > sm_count)
    ):
        ordered_seq_lens = tuple(sorted(sequence_lengths, reverse=True))
        _task_ids, _task_offsets, lpt_loads = _make_lpt_task_bins(
            ordered_seq_lens, num_heads=num_heads, worker_count=sm_count
        )
    return _select_bf16_route(
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=num_seqs,
        num_heads=num_heads,
        uniform_sequences=uniform_sequences,
        lpt_loads=lpt_loads,
        max_seq_len=max(sequence_lengths),
        use_initial_state=use_initial_state,
        store_final_state=store_final_state,
    )


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


class FlashKDABlackwellBF16FusedLaunch:
    """Preallocated single-kernel launch for the production BF16 path."""

    schedule = "fused_pipelined_mmasync_tcgen05_swap_ab_n32"
    _force_direct_m128 = False
    _force_direct_m128_n32 = False
    _force_independent_dvsplit = False
    _force_small_bh_owner_helper = False
    _persistent_task_schedule: str | None = None
    _keepalive: tuple[Any, ...]
    _force_bt16_prepare_chain = False
    _force_bt16_s7_chain = False
    _force_bt16_beta_tma = False
    _state_dtype_is_fp32 = False
    _n32_ft_slab = False
    _pdl_wait_initial_state_f32 = False
    _pdl_publish_final_state = False
    _checkpoint_accumulate = False
    _affine_main_indexed_initial = False
    _affine_main_indexed_initial_bf16 = False
    _n16_short_four_stage = False
    _active_beta_f32 = False

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
            compute_dtype,
            state_dtype_is_fp32=self._state_dtype_is_fp32
            and (not self._affine_main_indexed_initial_bf16),
        )
        self.compute_dtype = compute_dtype
        for name, tensor in (("q", q), ("k", k), ("v", v), ("out", out)):
            _require_tensor(tensor, name=name, dtype=torch.bfloat16, ndim=4)
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
        host_task_ids: tuple[int, ...] = ()
        host_task_offsets: tuple[int, ...] = ()
        host_task_token_starts: tuple[int, ...] = ()
        host_task_token_counts: tuple[int, ...] = ()
        host_task_state_sources: tuple[int, ...] = ()
        host_task_state_destinations: tuple[int, ...] = ()
        handoff_count = 0
        lpt_loads: tuple[int, ...] = ()
        if (
            _uses_measured_sm100_persistent_policy(gpu_arch=gpu_arch, sm_count=sm_count)
            and (not uniform_sequences)
            and (total_tasks > sm_count)
        ):
            ordered_seq_lens = tuple(
                (offsets[index + 1] - offsets[index] for index in ordered_sequences)
            )
            host_task_ids, host_task_offsets, lpt_loads = _make_lpt_task_bins(
                ordered_seq_lens, num_heads=num_heads, worker_count=sm_count
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
        _validate_n16_short_four_stage_contract(
            enabled=self._n16_short_four_stage,
            state_dtype_is_fp32=self._state_dtype_is_fp32,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            checkpoint_every_n_tokens=checkpoint_every_n_tokens,
            bounded_gate=not unbounded_softplus,
            indexed_state_pool=state_indices is not None,
            in_place_state_pool=initial_state is not None
            and initial_state is final_state,
        )
        full_n32_chunks = all(
            (
                (end - start) % BF16_M128_CHUNK == 0
                for start, end in zip(offsets, offsets[1:], strict=False)
            )
        )
        use_tf32_direct_m128 = compute_dtype == "tf32" and self._force_direct_m128
        if self._persistent_task_schedule is not None:
            if (
                gate_kind != KDAGateKind.LOWER_BOUND
                or checkpoint_every_n_tokens % BF16_M128_CHUNK
            ):
                raise ValueError(
                    "persistent N32 tasks require bounded gates and N32-aligned checkpoints"
                )
            if self._persistent_task_schedule == "grouped":
                route = BF16_ROUTE_HEAD_GROUPED_M128
            elif self._persistent_task_schedule == "lpt":
                route = BF16_ROUTE_LPT_M128
            elif self._persistent_task_schedule == "piece":
                if (
                    len(
                        set(
                            (
                                end - start
                                for start, end in zip(
                                    offsets, offsets[1:], strict=False
                                )
                            )
                        )
                    )
                    != 1
                ):
                    raise ValueError("piece task bins require uniform sequence lengths")
                route = BF16_ROUTE_PIECE_M128
            else:
                raise ValueError("unknown persistent task schedule")
        elif self._force_independent_dvsplit:
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
            n16_short_four_stage=self._n16_short_four_stage,
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
                n16_short_four_stage=self._n16_short_four_stage,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                indexed_state_pool=state_indices is not None,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                preferred_route=_select_bf16_route(
                    gpu_arch=gpu_arch,
                    sm_count=sm_count,
                    fixed_layout=fixed_layout,
                    num_seqs=num_seqs,
                    num_heads=num_heads,
                    uniform_sequences=uniform_sequences,
                    lpt_loads=lpt_loads,
                    max_seq_len=max_seq_len,
                    use_initial_state=initial_state is not None,
                    store_final_state=final_state is not None,
                )
                if compute_dtype == "bf16"
                else BF16_ROUTE_DIRECT_M128_N16,
            )
        else:
            route = _select_bf16_route(
                gpu_arch=gpu_arch,
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_seqs=num_seqs,
                num_heads=num_heads,
                uniform_sequences=uniform_sequences,
                lpt_loads=lpt_loads,
                max_seq_len=max_seq_len,
                use_initial_state=initial_state is not None,
                store_final_state=final_state is not None,
            )
        if (
            route == BF16_ROUTE_BT16_M64
            and (not self._state_dtype_is_fp32)
            and (state_indices is not None or has_nondefault_state_slot_stride)
        ):
            route = _direct_m128_route(num_heads=num_heads, max_seq_len=max_seq_len)
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
                n16_short_four_stage=self._n16_short_four_stage,
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
        use_bt16_s7_chain = self._force_bt16_s7_chain or (
            use_bt16_prepare_chain and BT16_VALUE_SPLITS * total_tasks > sm_count
        )
        use_independent_dvsplit = route == BF16_ROUTE_M64
        use_source_vtile_m128 = route == BF16_ROUTE_SOURCE_VTILE_M128
        source_vtile_worker_count = (
            total_tasks if fixed_layout else SOURCE_VTILE_PERSISTENT_WORKERS
        )
        source_vtile_persistent_tasks = (
            total_tasks // source_vtile_worker_count if use_source_vtile_m128 else 1
        )
        if self._force_bt16_beta_tma and (
            not use_bt16_prepare_chain
            or not fixed_layout
            or num_heads % BF16_BETA_TMA_MIN_HEADS != 0
        ):
            raise ValueError(
                "forced BT16 beta TMA requires a fixed, pitch-aligned BT16 route"
            )
        use_bt16_beta_tma = self._force_bt16_beta_tma or (
            use_bt16_prepare_chain
            and (not self._active_beta_f32)
            and (num_heads % BF16_BETA_TMA_MIN_HEADS == 0)
            and _should_use_bt16_dense_wavefront(
                gpu_arch=gpu_arch,
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_seqs=num_seqs,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
            )
        )
        use_bt16_s9_chain = (
            use_bt16_prepare_chain
            and (not use_bt16_s7_chain)
            and (
                total_tasks <= 8
                or (use_bt16_beta_tma and BT16_VALUE_SPLITS * total_tasks <= sm_count)
            )
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
        use_head_grouped_m128 = route == BF16_ROUTE_HEAD_GROUPED_M128
        use_lpt_persistent_m128 = route == BF16_ROUTE_LPT_M128
        use_scalar_chunk_lpt_m128 = route == BF16_ROUTE_SCALAR_CHUNK_LPT_M128
        use_piece_persistent_m128 = route == BF16_ROUTE_PIECE_M128
        use_persistent_m128 = (
            use_head_grouped_m128
            or use_lpt_persistent_m128
            or use_piece_persistent_m128
        )
        use_tf32_persistent_m128 = compute_dtype == "tf32" and use_persistent_m128
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
        persistent_worker_count = _uniform_persistent_worker_count(
            total_tasks, worker_cap=sm_count
        )
        if use_lpt_persistent_m128:
            persistent_worker_count = min(total_tasks, sm_count)
            host_task_ids, host_task_offsets, lpt_loads = _make_lpt_task_bins(
                tuple((offsets[i + 1] - offsets[i] for i in ordered_sequences)),
                num_heads=num_heads,
                worker_count=persistent_worker_count,
            )
        if use_head_grouped_m128:
            host_task_ids, host_task_offsets = _make_uniform_head_grouped_bins(
                num_seqs=num_seqs,
                num_heads=num_heads,
                worker_count=persistent_worker_count,
            )
        elif use_piece_persistent_m128:
            if self._persistent_task_schedule == "piece":
                persistent_worker_count = min(total_tasks, sm_count)
            else:
                piece_roofline = _persistent_m128_roofline(
                    gpu_arch=gpu_arch,
                    sm_count=sm_count,
                    num_seqs=num_seqs,
                    num_heads=num_heads,
                    seq_len=max_seq_len,
                    use_initial_state=initial_state is not None,
                    store_final_state=final_state is not None,
                )
                if piece_roofline is None:
                    raise RuntimeError(
                        "piece-persistent route selected without a resolved roofline advantage"
                    )
                persistent_worker_count = piece_roofline.worker_count
            (
                host_task_ids,
                host_task_offsets,
                host_task_token_starts,
                host_task_token_counts,
                host_task_state_sources,
                host_task_state_destinations,
                handoff_count,
                _piece_loads,
            ) = _make_uniform_piece_task_bins(
                num_seqs=num_seqs,
                num_heads=num_heads,
                seq_len=max_seq_len,
                worker_count=persistent_worker_count,
            )
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
        use_early_n32_state_pack = (
            use_scalar_beta
            and (not use_direct_m128_n16)
            and (not use_n32_logical_page64)
            and (max_seq_len <= H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQ_LEN)
        )
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
        )
        use_n32_tensor_state_decay = (
            checkpoint_every_n_tokens == 0
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
        pair_packed_beta_tma = (
            _pair_packed_beta_tma_source(
                beta_flat, total_tokens=total_tokens, num_heads=num_heads
            )
            if not self._active_beta_f32
            and route == BF16_ROUTE_DIRECT_M128
            and (num_heads == 12)
            and (max_seq_len > H12_DIRECT_N32_EARLY_STATE_PACK_MAX_SEQ_LEN)
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
            and (not use_bt16_prepare_chain or use_bt16_beta_tma)
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
        scalar_chunk_schedule_stride = 1
        if use_scalar_chunk_lpt_m128:
            (
                host_scalar_chunk_schedule,
                host_scalar_chunk_schedule_counts,
                scalar_chunk_schedule_stride,
            ) = _build_persistent_scalar_schedule(
                [end - start for start, end in zip(offsets, offsets[1:], strict=False)],
                num_heads,
                sm_count,
            )
            host_uploads["scalar_chunk_schedule"] = list(host_scalar_chunk_schedule)
            host_uploads["scalar_chunk_schedule_counts"] = list(
                host_scalar_chunk_schedule_counts
            )
        if use_persistent_m128:
            host_uploads["task_ids"] = list(host_task_ids)
            host_uploads["task_offsets"] = list(host_task_offsets)
        if use_piece_persistent_m128:
            host_uploads["task_token_starts"] = list(host_task_token_starts)
            host_uploads["task_token_counts"] = list(host_task_token_counts)
            host_uploads["task_state_sources"] = list(host_task_state_sources)
            host_uploads["task_state_destinations"] = list(host_task_state_destinations)
            mid_state = torch.empty(
                (handoff_count, HEAD_DIM, HEAD_DIM),
                dtype=torch.float32 if use_tf32_persistent_m128 else torch.bfloat16,
                device=q.device,
            )
            mid_state_ready = torch.zeros(
                handoff_count, dtype=torch.uint32, device=q.device
            )
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
                gpu_arch=gpu_arch,
                sm_count=sm_count,
                fixed_layout=fixed_layout,
                num_seqs=num_seqs,
                num_heads=num_heads,
                max_seq_len=max_seq_len,
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
        scalar_chunk_schedule = uploaded.get("scalar_chunk_schedule", empty_i32)
        scalar_chunk_schedule_counts = uploaded.get(
            "scalar_chunk_schedule_counts", empty_i32
        )
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
                or use_tf32_persistent_m128
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
        if use_bt16_prepare_chain and compute_dtype == "tf32":
            self.prepare_module = (
                _build_kda_module(
                    partial(_factory, "compiled_tf32_bt16_prepare_beta_tma")
                )
                if use_bt16_beta_tma
                else _build_kda_module(
                    partial(_factory, "compiled_tf32_bt16_prepare"),
                    active_beta_f32=self._active_beta_f32,
                )
            )
            self.module = _build_kda_module(
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
        elif use_bt16_prepare_chain:
            self.prepare_module = (
                _build_kda_module(
                    partial(_factory, "compiled_bf16_bt16_prepare_beta_tma")
                )
                if use_bt16_beta_tma
                else _build_kda_module(
                    partial(_factory, "compiled_bf16_bt16_prepare"),
                    active_beta_f32=self._active_beta_f32,
                )
            )
            self.module = (
                _build_kda_module(
                    partial(_factory, "compiled_fp32_bt16_chain_m64"),
                    stage_count=7
                    if use_bt16_s7_chain
                    else 9
                    if use_bt16_s9_chain
                    else 8,
                    serving_native_abi=serving_native_abi,
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                )
                if self._state_dtype_is_fp32
                else _build_kda_module(
                    partial(_factory, "compiled_bf16_bt16_chain_m64_s7"),
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                )
                if use_bt16_s7_chain
                else _build_kda_module(
                    partial(_factory, "compiled_bf16_bt16_chain_m64_s9"),
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                )
                if use_bt16_s9_chain
                else _build_kda_module(
                    partial(_factory, "compiled_bf16_bt16_chain_m64"),
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                )
            )
            self.schedule = (
                "decomposed_bt16_prepare_chain_m64_fp32_state_s7_two_resident"
                if self._state_dtype_is_fp32 and use_bt16_s7_chain
                else "decomposed_bt16_prepare_chain_m64_fp32_state_s9_underfilled"
                if self._state_dtype_is_fp32 and use_bt16_s9_chain
                else "decomposed_bt16_prepare_chain_m64_fp32_state"
                if self._state_dtype_is_fp32
                else "decomposed_bt16_prepare_chain_m64_wavefront_s7_two_resident"
                if use_bt16_s7_chain
                else "decomposed_bt16_prepare_chain_m64_wavefront_s9_underfilled"
                if use_bt16_s9_chain
                else "decomposed_bt16_prepare_chain_m64_wavefront"
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
            self.module = _build_kda_module(
                partial(_factory, "compiled_bf16_fused_m128"),
                BF16_N16_M128_CHUNK if use_direct_m128_n16 else BF16_M128_CHUNK,
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
                pdl_wait_initial_state_f32=self._pdl_wait_initial_state_f32,
                pdl_publish_final_state=self._pdl_publish_final_state,
                affine_main_indexed_initial=self._affine_main_indexed_initial,
                affine_main_indexed_initial_bf16=self._affine_main_indexed_initial_bf16,
                backend=backend,
                n16_short_four_stage=self._n16_short_four_stage,
            )
            self.schedule = (
                (
                    "fused_checkpoint_tma_direct_m128_n16_s4"
                    if self._n16_short_four_stage
                    else "fused_checkpoint_tma_direct_m128_n16"
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
            self.module = _build_kda_module(
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
        elif use_small_bh_owner_helper:
            self.module = _build_kda_module(
                partial(_factory, "compiled_small_bh_m128"),
                state_dtype_is_fp32=self._state_dtype_is_fp32,
                serving_native_abi=serving_native_abi,
            )
            self.schedule = (
                "fused_small_bh_m128_owner7helper_fp32_state_ring35"
                if self._state_dtype_is_fp32
                else "fused_small_bh_m128_owner7helper_compact_ring35"
            )
        elif (
            use_independent_dvsplit or use_tf32_direct_m128
        ) and compute_dtype == "tf32":
            if use_tf32_direct_n32:
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
                self.module = _build_kda_module(
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
                    affine_main_indexed_initial_bf16=self._affine_main_indexed_initial_bf16,
                    affine_factor_cache=_affine_factor_cache_mode,
                    affine_map_only=_affine_map_only,
                    affine_map_output=_affine_map_output,
                )
                self.schedule = f"fused_tf32_m{n32_value_rows}_local_factors_s{n32_prepare_stages}_n32"
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
            else:
                self.module = _build_kda_module(
                    partial(_factory, "compiled_tf32_fused"),
                    value_rows=128 if use_tf32_direct_m128 else 64,
                    state_dtype_is_fp32=self._state_dtype_is_fp32,
                    active_beta_f32=self._active_beta_f32,
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                    prep_stages=4 if self._n16_short_four_stage else 3,
                )
                self.schedule = (
                    "fused_tf32_m128_local_factors_s4_n16"
                    if self._n16_short_four_stage
                    else "fused_tf32_m128_local_factors_s3_n16"
                    if use_tf32_direct_m128
                    else "fused_tf32_m64_local_factors_s3"
                )
        elif use_independent_dvsplit:
            self.module = _build_kda_module(
                partial(_factory, "compiled_bf16_fused_m64"),
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
        elif use_source_vtile_m128:
            self.module = _build_kda_module(
                partial(_factory, "compiled_bf16_fused_m128_vtile"),
                full_chunks=full_n32_chunks,
                num_heads=num_heads,
                use_initial_state=initial_state is not None,
                store_final_state=final_state is not None,
                scale=float(scale),
                lower_bound=float(lower_bound),
                persistent_mode=source_vtile_persistent_tasks > 1,
                persistent_six_task_schedule=source_vtile_persistent_tasks == 6,
                persistent_stride_head_aligned=source_vtile_worker_count % num_heads
                == 0,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
            )
            self.schedule = (
                "fused_vtile_m128_persistent_fp32_state"
                if self._state_dtype_is_fp32 and source_vtile_persistent_tasks > 1
                else "fused_vtile_m128_persistent"
                if source_vtile_persistent_tasks > 1
                else "fused_vtile_m128_fp32_state"
                if self._state_dtype_is_fp32
                else "fused_vtile_m128"
            )
        elif use_scalar_chunk_lpt_m128:
            self.module = _build_kda_module(
                partial(_factory, "compiled_scalar_chunk_lpt_m128"),
                num_heads=num_heads,
                use_initial_state=initial_state is not None,
                store_final_state=final_state is not None,
                scale=float(scale),
                lower_bound=float(lower_bound),
                persistent_schedule=True,
                state_dtype_is_fp32=self._state_dtype_is_fp32,
            )
            self.schedule = (
                "fused_scalar_chunk_lpt_m128_fp32_state"
                if self._state_dtype_is_fp32
                else "fused_scalar_chunk_lpt_m128"
            )
        elif use_persistent_m128:
            if use_tf32_persistent_m128:
                self.module = _build_kda_module(
                    partial(_factory, "compiled_tf32_fused_n32"),
                    state_dtype_is_fp32=self._state_dtype_is_fp32,
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                    prep_stages=1 if max_seq_len <= 64 else 3,
                    persistent_tasks=True,
                    piece_tasks=use_piece_persistent_m128,
                )
            else:
                self.module = _build_kda_module(
                    partial(_factory, "compiled_bf16_persistent_m128"),
                    piece_tasks=use_piece_persistent_m128,
                    state_dtype_is_fp32=self._state_dtype_is_fp32,
                    write_checkpoints=bool(checkpoint_every_n_tokens),
                    backend=backend,
                )
            self.schedule = (
                "fused_persistent_m128_head_grouped_fp32_state"
                if self._state_dtype_is_fp32 and use_head_grouped_m128
                else "fused_persistent_m128_lpt_bins_fp32_state"
                if self._state_dtype_is_fp32 and use_lpt_persistent_m128
                else "fused_persistent_m128_recurrence_pieces_fp32_state"
                if self._state_dtype_is_fp32 and use_piece_persistent_m128
                else "fused_persistent_m128_head_grouped"
                if use_head_grouped_m128
                else "fused_persistent_m128_lpt_bins"
                if use_lpt_persistent_m128
                else "fused_persistent_m128_recurrence_pieces"
            )
        self.route = route
        self.gate_kind = gate_kind.value
        self.gpu_arch = gpu_arch
        self.sm_count = sm_count
        self.beta_transport = (
            "tma"
            if use_bt16_beta_tma
            or (
                use_independent_dvsplit
                and compute_dtype == "bf16"
                and (not self._active_beta_f32)
            )
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
        if use_tf32_persistent_m128:
            self.logical_page_tokens = 64
            self.physical_subtile_tokens = 32
            task_kind = (
                "head_grouped"
                if use_head_grouped_m128
                else "recurrence_pieces"
                if use_piece_persistent_m128
                else "lpt_bins"
            )
            self.schedule = f"fused_tf32_m128_persistent_{task_kind}_n32"
        self.ir = self.module.schedule
        packet_workspace = torch.empty(1, dtype=torch.bfloat16, device=q.device)
        packet_ready = torch.zeros(1, dtype=torch.uint32, device=q.device)
        packet_consumed = torch.zeros(1, dtype=torch.uint32, device=q.device)
        helper_done = torch.zeros(1, dtype=torch.uint32, device=q.device)
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
        elif use_small_bh_owner_helper:
            packet_workspace = torch.empty(
                (
                    total_tasks * SMALL_BH_RING_STAGES * SMALL_BH_PACKET_ROWS,
                    SMALL_BH_PACKET_ELEMS,
                ),
                dtype=torch.bfloat16,
                device=q.device,
            )
            packet_ready = torch.zeros(
                total_tasks * SMALL_BH_RING_STAGES, dtype=torch.uint32, device=q.device
            )
            packet_consumed = torch.zeros_like(packet_ready)
            helper_done = torch.zeros(total_tasks, dtype=torch.uint32, device=q.device)
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
            scalar_chunk_schedule,
            scalar_chunk_schedule_counts,
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
            packet_workspace,
            packet_ready,
            packet_consumed,
            helper_done,
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
            else sm_count
            if use_scalar_chunk_lpt_m128
            else source_vtile_worker_count
            if use_source_vtile_m128
            else persistent_worker_count
            if use_head_grouped_m128 or use_piece_persistent_m128
            else persistent_worker_count
            if use_lpt_persistent_m128
            else total_tasks,
            1,
            1,
        )
        if n32_value_rows == 64:
            self.grid = (2 * self.grid[0], 1, 1)
        self.prepare_grid = (bt16_prepare_total_ctas, 1, 1)
        self.args = {
            "q": q_flat,
            "q_tma": q_flat,
            "k": k_flat,
            "k_tma": k_flat,
            "v": v_flat,
            "v_tma": v_flat,
            "g": g_pointer,
            "g_tma": g_flat,
            "beta": beta_pointer,
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
        if use_independent_dvsplit or use_source_vtile_m128:
            self.args.update(
                state_indices_addr=state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                state_slot_stride=state_slot_stride,
                use_state_indices=int(state_indices is not None),
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
            if use_independent_dvsplit:
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
            if use_source_vtile_m128:
                self.args["uniform_seq_len"] = max_seq_len
                self.args["persistent_tasks"] = source_vtile_persistent_tasks
                self.args["persistent_stride"] = source_vtile_worker_count
        elif use_scalar_chunk_lpt_m128:
            self.args.update(
                tile_schedule=scalar_chunk_schedule,
                tile_schedule_counts=scalar_chunk_schedule_counts,
                schedule_stride=scalar_chunk_schedule_stride,
                state_indices_addr=state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                state_slot_stride=state_slot_stride,
                use_state_indices=int(state_indices is not None),
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
        elif use_persistent_m128:
            self.args.update(
                state_checkpoints=state_checkpoints
                if state_checkpoints is not None
                else empty_state,
                checkpoint_cu_starts=checkpoint_cu_starts
                if checkpoint_cu_starts is not None
                else empty_i64,
                checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                beta_token_stride=beta_flat.stride(0),
                g_token_stride=g_flat.stride(0),
            )
            self.args["task_ids"] = task_ids
            self.args["task_offsets"] = task_offsets
            self.args["task_token_starts"] = task_token_starts
            self.args["task_token_counts"] = task_token_counts
            self.args["task_state_sources"] = task_state_sources
            self.args["task_state_destinations"] = task_state_destinations
            self.args["mid_state"] = mid_state
            self.args["mid_state_ready"] = mid_state_ready
            self.args.update(
                state_indices_addr=state_indices.data_ptr()
                if state_indices is not None
                else empty_i32.data_ptr(),
                state_slot_stride=state_slot_stride,
                use_state_indices=int(state_indices is not None),
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
        elif not use_independent_dvsplit:
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
            self.args.update(
                g_token_stride=g_flat.stride(0),
                cu_chunk_offsets=empty_chunk_offsets,
                chunk_state=empty_state,
                state_checkpoint_needed=empty_u32,
                tape_qd=empty_state,
                tape_kd=empty_state,
                tape_kr=empty_state,
                tape_j=empty_state,
                tape_restore_factor=empty_f32,
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
                state_checkpoints_tma=state_checkpoints
                if state_checkpoints is not None and not fp32_checkpoints
                else empty_checkpoint_tma,
            )
        if use_small_bh_owner_helper:
            self.args.update(
                packet_workspace=packet_workspace,
                packet_ready=packet_ready,
                packet_consumed=packet_consumed,
                helper_done=helper_done,
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
        elif uses_default_fused_m128 and self._state_dtype_is_fp32:
            self.args.update(
                initial_state_f32=initial_state_f32_pointer,
                final_state_f32=final_state_f32_pointer,
            )
        self.prepare_args: dict[str, Any] = {}
        if use_bt16_prepare_chain:
            self.prepare_args = {
                "q": q_flat,
                "q_tma": q,
                "k": k_flat,
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
                "v": v_flat,
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
            use_independent_dvsplit
            or use_tf32_direct_m128
            or use_tf32_persistent_m128
            or use_tf32_owner_helper
        ) and compute_dtype == "tf32":
            self.args = {
                "q": q_flat,
                "q_tma": q,
                "k": k_flat,
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
                "v": v_flat,
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
        if (
            use_tf32_direct_n32 or use_tf32_persistent_m128 or use_tf32_owner_helper
        ) and compute_dtype == "tf32":
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
                mid_state_f32=mid_state
                if use_piece_persistent_m128 or use_tf32_owner_helper
                else empty_f32,
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

    def _prepare_descriptors_in_stream(self) -> None:
        """Encode and upload every TMA descriptor on the current stream."""
        if self.prepare_module is not None:
            self.prepare_module.prepare(grid=self.prepare_grid, **self.prepare_args)
        self.module.prepare(grid=self.grid, **self.args)
        self._descriptors_stale = False

    def _launch_in_stream(self) -> None:
        """Launch inside an already entered FFI/Torch stream context."""
        if self._descriptors_stale:
            # A plan-cache rebind moved a descriptor source; re-encode in
            # the same stream context as the launch it precedes.
            self._prepare_descriptors_in_stream()
        for destination, source in self._token_storage_refreshes:
            destination.copy_(source)
        if self._beta_tma_valid is not None:
            self._beta_tma_valid.copy_(self._beta_tma_source)
        if self.prepare_module is not None:
            self.prepare_module.launch(grid=self.prepare_grid, **self.prepare_args)
        self.module.launch(grid=self.grid, **self.args)

    def _launch_uncaptured(self) -> None:
        import tvm_ffi

        with tvm_ffi.use_torch_stream():
            self._launch_in_stream()

    def launch(self) -> None:
        self._launch_uncaptured()

    def close(self) -> None:
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        return None


class FlashKDABlackwellBF16GroupedM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """Grouped persistent tasks with independent BF16/TF32 compute choice."""

    _persistent_task_schedule = "grouped"


class FlashKDABlackwellFP32GroupedM128Launch(FlashKDABlackwellBF16GroupedM128Launch):
    """Grouped persistent tasks with FP32 external state."""

    _state_dtype_is_fp32 = True


class FlashKDABlackwellBF16LPTM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """LPT persistent tasks with independent BF16/TF32 compute choice."""

    _persistent_task_schedule = "lpt"


class FlashKDABlackwellFP32LPTM128Launch(FlashKDABlackwellBF16LPTM128Launch):
    """LPT persistent tasks with FP32 external state."""

    _state_dtype_is_fp32 = True


class FlashKDABlackwellBF16PieceM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """Uniform recurrence pieces with BF16/TF32 compute and device handoffs."""

    _persistent_task_schedule = "piece"


class FlashKDABlackwellFP32PieceM128Launch(FlashKDABlackwellBF16PieceM128Launch):
    """Uniform recurrence pieces with FP32 external state."""

    _state_dtype_is_fp32 = True


class FlashKDABlackwellBF16DirectM128Launch(FlashKDABlackwellBF16FusedLaunch):
    """Prepare the compatible direct-M128 seed without auto-route substitution."""

    _force_direct_m128 = True


class FlashKDABlackwellBF16DirectM128N32Launch(FlashKDABlackwellBF16DirectM128Launch):
    """Force N32 for explicit checkpoint A/B without changing production policy."""

    _force_direct_m128_n32 = True


class FlashKDABlackwellBF16SmallBHLaunch(FlashKDABlackwellBF16FusedLaunch):
    """Prepare the owner/helper candidate independently of auto dispatch."""

    _force_small_bh_owner_helper = True


class FlashKDABlackwellFP32SmallBHLaunch(FlashKDABlackwellBF16FusedLaunch):
    """Reuse owner/helper compute with FP32 indexed state-pool I/O."""

    _force_small_bh_owner_helper = True
    _state_dtype_is_fp32 = True


class FlashKDABlackwellBF16BT16Launch(FlashKDABlackwellBF16FusedLaunch):
    """Prepare the BT16 factor/chain family independently of auto dispatch."""

    _force_bt16_prepare_chain = True


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


class FlashKDABlackwellBF16ActiveBetaM64Launch(FlashKDABlackwellBF16M64Launch):
    """Fused M64 with active FP32 beta and BF16 external state."""

    _active_beta_f32 = True


class FlashKDABlackwellFP32ActiveBetaM64Launch(FlashKDABlackwellFP32M64Launch):
    """Fused M64 with active FP32 beta and FP32 external state."""

    _active_beta_f32 = True


class FlashKDABlackwellFP32DirectM128Launch(FlashKDABlackwellFP32FusedM128Launch):
    """Force the variable-shape direct M128 body with FP32 state I/O."""

    _force_direct_m128 = True


class FlashKDABlackwellFP32DirectM128N32Launch(FlashKDABlackwellFP32DirectM128Launch):
    """Force the FP32-state N32 body for explicit checkpoint A/B."""

    _force_direct_m128_n32 = True


class FlashKDABlackwellFP32ActiveBetaDirectM128Launch(
    FlashKDABlackwellFP32DirectM128Launch
):
    """Explicit direct M128 with active FP32 beta and either compute precision."""

    _active_beta_f32 = True


class FlashKDABlackwellFP32DirectM128N16S4Launch(FlashKDABlackwellFP32DirectM128Launch):
    """Private short-request N16 S4 candidate for explicit serving A/B."""

    _n16_short_four_stage = True


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


class FlashKDABlackwellFP32SlabM128PDLIndexedBF16InitialProducerLaunch(
    FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch
):
    """Read part zero directly from the caller's indexed BF16 state pool."""

    _affine_main_indexed_initial_bf16 = True


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
        active_beta_f32: bool = False,
    ) -> None:
        import torch

        validate_kda_state_tensors(compute_dtype, initial_state, final_state)
        self.compute_dtype = compute_dtype
        self.active_beta_f32 = active_beta_f32
        if active_beta_f32 and (compute_dtype != "tf32" or lower_bound is None):
            raise ValueError("active-beta affine requires bounded TF32 compute")
        if q.ndim != 4 or any(
            (not tensor.is_contiguous() for tensor in (q, k, v, out))
        ):
            raise ValueError("affine q/k/v/out require contiguous [B,T,H,128] tensors")
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
        self._external_state_is_fp32 = initial_state.dtype == torch.float32
        self.schedule = (
            "split_seq_affine_prefix_direct_m128_fp32_state"
            if self._external_state_is_fp32
            else "split_seq_affine_prefix_direct_m128_bf16_state"
        )
        if compute_dtype == "tf32":
            self.schedule += "_tf32_compute"
        # Torch-epilogue scratch (final_external / *_selected / checkpoint_merged)
        # is allocated only when the fused epilogue is unavailable (see below).
        self._final_external = None
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
        self._correction_out = torch.empty_like(out[:, first_part_tokens:])
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
            self._checkpoint_in_place = (
                state_checkpoints.dtype == torch.float32
                and _affine_rows_in_place_enabled()
            )
            if self._checkpoint_in_place:
                # FP32 rows: the main windows write the caller's rows directly
                # and the correction windows add theirs in place, so no row is
                # staged, re-read and merged.  Each window's destination row
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
        uploaded_i32 = _upload_int_batch(q.device, host_i32, torch.int32)
        uploaded_i64 = _upload_int_batch(q.device, host_i64, torch.int64)
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
        main_launch_cls = (
            FlashKDABlackwellFP32SlabM128PDLIndexedInitialProducerLaunch
            if self._external_state_is_fp32
            else FlashKDABlackwellFP32SlabM128PDLIndexedBF16InitialProducerLaunch
        )
        self._main = main_launch_cls(
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
        self._main.args.update(
            state_indices_addr=self._part_state_indices.data_ptr(),
            state_slot_stride=initial_state.stride(0),
            use_state_indices=1,
        )
        if self._external_state_is_fp32:
            self._main.args["initial_state_f32"] = self._initial_pool_pointer
        else:
            self._main.args["initial_state"] = self._initial_pool_pointer
        self._correction = None
        if not self._use_output_projection:
            correction_cls = (
                FlashKDABlackwellFP32SlabM128PDLConsumerAccumulateLaunch
                if self._checkpoint_in_place
                else FlashKDABlackwellFP32SlabM128PDLConsumerLaunch
            )
            self._correction = correction_cls(
                q[:, first_part_tokens:],
                k[:, first_part_tokens:],
                self._zero_v,
                g[:, first_part_tokens:],
                beta[:, first_part_tokens:],
                scale,
                self._correction_out,
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
            **tail_factor_kwargs,
            sequence_lengths=tuple(
                (
                    b - a
                    for a, b in zip(token_offsets[1:], token_offsets[2:], strict=False)
                )
            ),
        )
        if compute_dtype == "bf16":
            self._map.args["initial_state_f32"] = self._main_final
        self._scan_module = _build_kda_module(
            partial(_factory, "compiled_flashkda_split_scan_bf16_m128"),
            use_pdl=True,
            compute_dtype=compute_dtype,
            backend=backend,
        )
        self._out_tail = out[:, first_part_tokens:]
        self._projection_module = None
        if self._use_output_projection:
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
            if self._fused_epilogue is not None:
                self.schedule += "_fused_epilogue"
        if self._fused_epilogue is None:
            # Torch epilogue scratch: a merged copy of the checkpoint rows plus
            # the selected/converted final states (~4x the checkpoint rows of a
            # 16K pack; retained per plan-cache entry, so only when needed).
            self._final_external = torch.empty(
                state_shape, dtype=final_state.dtype, device=q.device
            )
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

    def _launch_uncaptured(self) -> None:
        import torch
        import tvm_ffi

        if self._descriptors_stale:
            # A plan-cache hit under CUDA-graph capture marks the composite;
            # every part re-encodes its descriptors in its own stream context.
            for sub_name in AFFINE_SUB_LAUNCHES:
                sub = getattr(self, sub_name, None)
                if sub is not None:
                    sub._descriptors_stale = True
            self._descriptors_stale = False
        with tvm_ffi.use_torch_stream():
            for destination, source in self._affine_input_refreshes:
                destination.copy_(source)
            if self._fused_epilogue is not None:
                # One kernel gathers every per-call index the parts consume
                # (window state indices, in-place row starts, the int64 state
                # indices of the epilogue); the torch sequence below costs
                # four launches on the host path before the first chain kernel.
                rows = self._checkpoint_in_place
                self._fused_epilogue.index_prep(
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
            else:
                self._part_state_indices.index_copy_(
                    0, self._first_parts, self._state_indices
                )
                if self._checkpoint_in_place:
                    starts = self._part_row_starts[: self.num_parts]
                    torch.index_select(
                        self._checkpoint_start, 0, self._part_seq_ids, out=starts
                    )
                    starts.add_(self._part_local_rows)
            # The parts share this stream context (no re-entry per part).
            self._main._launch_in_stream()
            if self._fused_epilogue is None:
                # The int64 index copy is only consumed by the final-state
                # scatter, so it follows the first chain kernel.
                self._state_indices_long.copy_(self._state_indices)
            self._map._launch_in_stream()
            self._scan_module.launch(
                grid=(self._num_sequences * int(self._main_final.shape[1]) * 32, 1, 1),
                split_state=self._main_final,
                map_state_bf16=self._map_state,
                carry=self._carry,
                num_heads=int(self._main_final.shape[1]),
                part_cu_seqlens=self._part_cu_seqlens,
                final_state=self._final_compact,
                write_final_state=int(self._use_output_projection),
            )
            if self._use_output_projection:
                self._projection_module.launch(
                    grid=self._projection_grid, **self._projection_args
                )
            else:
                self._correction._launch_in_stream()
            if self._fused_epilogue is not None:
                self._launch_fused_epilogue()
                return
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
                        self._main_final,
                        0,
                        self._last_parts,
                        out=self._final_main_selected,
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
            if self._external_state_is_fp32:
                self._final_pool.index_copy_(
                    0, self._state_indices_long, self._final_compact
                )
            else:
                self._final_external.copy_(self._final_compact)
                self._final_pool.index_copy_(
                    0, self._state_indices_long, self._final_external
                )

    def _launch_fused_epilogue(self) -> None:
        """One kernel: checkpoint-row merge/scatter, tail add, final state."""
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
        fused.run(
            *rows,
            fused.first_rows,
            fused.num_rows,
            self._out_tail,
            self._correction_out,
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
            fused.tail_elems,
            int(merge_rows),
        )

    def close(self) -> None:
        self._cuda_graph = None
        self._cuda_graph_capture_stream = None
        self._main.close()
        self._map.close()
        if self._correction is not None:
            self._correction.close()


class FlashKDABlackwellBF16BT16S7Launch(FlashKDABlackwellBF16BT16Launch):
    """Force the lower-SMEM BT16 chain for focused validation."""

    _force_bt16_s7_chain = True


class FlashKDABlackwellBF16BT16BetaTMALaunch(FlashKDABlackwellBF16BT16Launch):
    """Force the beta-TMA BT16 retrace for focused transport validation."""

    _force_bt16_beta_tma = True


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
        gpu_arch=gpu_arch,
        sm_count=sm_count,
        fixed_layout=fixed_layout,
        num_seqs=len(sequence_lengths),
        num_heads=int(num_heads),
        uniform_sequences=len(set(sequence_lengths)) == 1,
        lpt_loads=(),
        max_seq_len=max(sequence_lengths),
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
    if any(
        (
            tensor is None or not tensor.is_contiguous()
            for tensor in (q, argument(1, "k"), argument(2, "v"), argument(6, "out"))
        )
    ):
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


def _rebind_tensor_signature(tensor) -> tuple | None:
    """Layout facts a prepared launch depends on, excluding the base address.

    The low address bits are kept because preparation branches on 16-byte
    alignment (pair-packed beta TMA, checkpoint TMA) and TMA descriptors
    require 16-byte aligned bases.
    """
    if tensor is None:
        return None
    return (
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.dtype,
        tensor.data_ptr() & 0xFF,
        tensor.device.index,
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
        shape = tuple(tensor.shape)
        stride = tuple(tensor.stride())
        dtype = tensor.dtype
        facts.append((shape, stride, dtype, pointer & 0xFF, tensor.device.index))
        base = pointer - tensor.storage_offset() * tensor.element_size()
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


# Split-sequence affine composites hold three prepared part launches plus a
# few caller-aliasing attributes of their own; their containers are addressed
# as "<sub launch>.<container>" so one rebind plan covers the whole DAG.
AFFINE_SUB_LAUNCHES = ("_main", "_map", "_correction")
AFFINE_REBIND_ATTRIBUTES = (
    "_state_indices",
    "_initial_pool",
    "_initial_pool_pointer",
    "_final_pool",
    "_checkpoint_output",
    "_checkpoint_start",
    "_out_tail",
)


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
    """
    import torch

    if signature is None:
        signature = rebind_signature(inputs)
    if signature != plan.signature:
        raise ValueError("prepared launch signature does not match the new inputs")
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
        if (
            spec.byte_offset == 0
            and source.dtype == spec.dtype
            and tuple(source.shape) == spec.shape
            and tuple(source.stride()) == spec.stride
        ):
            tensor = source
        elif (
            source.dtype == spec.dtype and spec.byte_offset % source.element_size() == 0
        ):
            tensor = source.as_strided(
                spec.shape,
                spec.stride,
                source.storage_offset() + spec.byte_offset // source.element_size(),
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
    for spec in plan.views:
        if changed is not None and spec.input_name not in changed:
            continue
        touched.add(spec.container)
        container = containers[spec.container]
        replacement = view_for(spec)
        if (
            isinstance(spec.key, str)
            and spec.key.endswith("_tma")
            and container[spec.key].data_ptr() != replacement.data_ptr()
        ):
            tma_moved = True
            owner, _ = _rebind_owner(impl, spec.container)
            if owner not in stale_owners:
                stale_owners.append(owner)
        container[spec.key] = replacement
    for address in plan.addresses:
        if changed is not None and address.input_name not in changed:
            continue
        touched.add(address.container)
        containers[address.container][address.key] = (
            inputs[address.input_name].data_ptr() + address.byte_offset
        )
    new_inputs = tuple(
        inputs[name] for name in REBIND_INPUT_NAMES if inputs.get(name) is not None
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
    for sub_name, owned in plan.sub_owned_keepalive.items():
        sub = getattr(impl, sub_name, None)
        if sub is not None:
            sub._keepalive = owned + new_inputs
    for owner in stale_owners:
        # Only the launches whose descriptor sources moved re-encode, inside
        # their launch stream context (see _launch_uncaptured); a composite
        # whose parts moved does not re-encode the untouched parts.
        owner._descriptors_stale = True
    impl._keepalive = plan.owned_keepalive + new_inputs
    return tma_moved


class FlashKDABlackwellLaunch:
    """Dispatch shared BF16/FP32 schedules and retain FP32 compatibility."""

    def __init__(self, *args, **kwargs) -> None:
        import torch

        def argument(position, name, default=None):
            return args[position] if len(args) > position else kwargs.get(name, default)

        initial_state = argument(10, "initial_state")
        final_state = argument(11, "final_state")
        state_indices = argument(13, "state_indices")
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
        checkpoint_request = (
            argument(14, "state_checkpoints") is not None
            or argument(15, "checkpoint_cu_starts") is not None
            or argument(16, "checkpoint_every_n_tokens", 0) != 0
        )
        impl: type[FlashKDABlackwellBF16FusedLaunch]
        if _supports_affine_split_launch(args, kwargs):
            impl = FlashKDABlackwellAffineSplitLaunch
        elif state_is_fp32:
            candidates = (
                (FlashKDABlackwellFP32BT16Launch, _supports_fp32_bt16_launch),
                (FlashKDABlackwellFP32SmallBHLaunch, _supports_fp32_small_bh_launch),
            )
            if compute_dtype != "tf32":
                candidates = tuple(reversed(candidates))
            impl = (
                FlashKDABlackwellFP32FusedM128Launch
                if state_indices is not None
                or checkpoint_request
                or compute_dtype == "tf32"
                else FlashKDABlackwellFP32FusedM128Launch
            )
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


def prepare_fwd(
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
    initial_state=None,
    final_state=None,
    cu_seqlens=None,
    state_indices=None,
    state_checkpoints=None,
    checkpoint_cu_starts=None,
    checkpoint_every_n_tokens: int = 0,
    use_qk_l2norm_in_kernel: bool = True,
    compute_dtype: str = "bf16",
) -> FlashKDABlackwellLaunch:
    """Validate inputs and preallocate descriptors/workspace for repeated launch."""
    if use_qk_l2norm_in_kernel is not True:
        raise ValueError("FlashKDA requires use_qk_l2norm_in_kernel=True")
    return FlashKDABlackwellLaunch(
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
        checkpoint_every_n_tokens,
        compute_dtype=compute_dtype,
    )


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
