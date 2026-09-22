# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.
"""Host-only sparse MLA dispatch from grid size, KV work and resource limits."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import ceil


@dataclass(frozen=True)
class SparseMlaSelection:
    tuning: _SparseMlaTuning
    family: str
    tile: int
    splits: int
    head_dim_ctas: int
    reason: str


@dataclass(frozen=True)
class _Candidate:
    priority: int
    reason: str
    tuning: _SparseMlaTuning


@dataclass(frozen=True)
class _SparseMlaTuning:
    """Internal forced profiles for coverage and measured optimization."""

    family: str = "auto"
    tile_size_q: int = 0
    split_kv: int = 1
    scheduler: str = "nonpersistent"
    reduction: str = "gmem_separate"
    head_dim_ctas: int = 1
    gather_issue_warps: int = 1
    offset_cache: str = "strided"
    fuse_epilogue: bool = False
    direct_inputs: bool = False
    balanced_registers: bool = False
    reuse_kv: bool = False
    reuse_kv_stages: int = 0
    paired_correction: bool = False
    single_kv_stream: bool = False
    defer_max_update: bool = False
    uniform_offset_cache: bool = False
    kv_pipeline_stages: int = 0
    kv_tile_size: int = 128
    page_pipeline_stages: int = 0


def _candidates(
    *,
    rows,
    heads,
    query_length,
    capacity,
    dtype,
    sm_count,
):
    """Rank schedules by parallel work, gather traffic and available SMs."""
    max_rows, num_heads, max_seq_len_q, q_data_type = (rows, heads, query_length, dtype)
    if min(rows, heads, query_length, capacity, sm_count) < 1:
        raise ValueError("workload and device dimensions must be positive")
    if dtype not in ("bf16", "fp8"):
        raise ValueError("native BF16 or FP8 is required")
    throughput_h32 = num_heads == 32 and max_rows * 2 >= sm_count and (capacity >= 512)
    candidate = _SparseMlaTuning()
    options = []
    if q_data_type == "fp8":
        long_list = capacity >= 8192
        m16_work = max_rows * ((num_heads + 15) // 16)
        m16_one_step_work = m16_work * ((capacity + 255) // 256)
        use_keep = (
            num_heads >= 64
            and (
                capacity > 1024
                and m16_one_step_work > sm_count
                or m16_work >= sm_count // 2
            )
            or (num_heads == 16 and max_rows == 1 and (capacity <= 256))
        )
        use_keep = use_keep or (long_list and num_heads >= 16 and (max_rows >= 8))
        use_keep = use_keep or (capacity <= 256 and num_heads >= 16)
        use_keep = use_keep or throughput_h32
        candidate = _SparseMlaTuning(
            family="keep" if use_keep else "swap",
            tile_size_q=64 if use_keep else 16,
            split_kv=1 if throughput_h32 or (use_keep and capacity <= 1024) else 0,
            head_dim_ctas=1 if throughput_h32 else 0,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=not use_keep,
            reuse_kv=capacity >= 512 and use_keep,
            reuse_kv_stages=10 if capacity >= 512 and use_keep else 0,
        )
        base_clusters = max_rows * ((num_heads + 127) // 128)
        if (
            long_list
            and num_heads >= 128
            and (max_rows >= 8)
            and (base_clusters * 2 <= sm_count // 2)
        ):
            candidate = _SparseMlaTuning(
                family="2cta",
                tile_size_q=128,
                split_kv=0,
                fuse_epilogue=True,
                direct_inputs=True,
            )
        elif (
            num_heads >= 128
            and capacity >= 512
            and (base_clusters * 2 >= sm_count // 2)
        ):
            candidate = _SparseMlaTuning(
                family="2cta",
                tile_size_q=128,
                split_kv=1,
                fuse_epilogue=True,
                direct_inputs=True,
            )
        keep_work = max_rows * ((num_heads + 63) // 64)
        kv_tiles = (capacity + 127) // 128
        split_limit = min(kv_tiles, max(1, sm_count // 4 // keep_work))
        kv_steps = (kv_tiles + split_limit - 1) // split_limit
        keep_splits = (kv_tiles + kv_steps - 1) // kv_steps
        prefer_wide_q = (
            num_heads >= 64
            and capacity >= 512
            and (kv_steps <= 2)
            and (keep_work * keep_splits * 4 >= sm_count // 2)
        )
        cluster_splits = 1 << ((capacity + 255) // 256 - 1).bit_length()
        if prefer_wide_q:
            candidate = _SparseMlaTuning(
                family="keep",
                tile_size_q=64,
                split_kv=keep_splits,
                head_dim_ctas=4,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
            )
        elif (
            1 < cluster_splits <= 4
            and num_heads % 16 == 0
            and (m16_work * cluster_splits <= sm_count // 2)
        ):
            candidate = _SparseMlaTuning(
                family="swap",
                tile_size_q=16,
                split_kv=cluster_splits,
                reduction="cluster",
                head_dim_ctas=0,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
                balanced_registers=True,
            )
        m8_work = max_rows * ((num_heads + 7) // 8)
        if capacity <= 256 and num_heads >= 16 and (m8_work * 4 <= sm_count // 2):
            candidate = _SparseMlaTuning(
                family="swap",
                tile_size_q=8,
                split_kv=1,
                head_dim_ctas=4,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
                balanced_registers=True,
            )
        if num_heads <= 16 and max_rows * 4 >= sm_count and (capacity >= 512):
            candidate = _SparseMlaTuning(
                family="swap",
                tile_size_q=8 if num_heads <= 8 else 16,
                split_kv=2 if capacity > 1024 and max_rows * 2 <= sm_count else 1,
                head_dim_ctas=1,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
                balanced_registers=True,
                reuse_kv=True,
                reuse_kv_stages=12,
            )
        if num_heads == 32 and 32 <= max_rows <= sm_count // 2 and (capacity >= 512):
            candidate = _SparseMlaTuning(
                family="keep",
                tile_size_q=64,
                split_kv=1 if capacity <= 1024 else 2,
                head_dim_ctas=1,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
                reuse_kv=True,
                reuse_kv_stages=10,
            )
        one_tile_splits = (capacity + 127) // 128
        keep_work = max_rows * ((num_heads + 63) // 64)
        if (
            num_heads >= 64
            and 512 <= capacity <= 1024
            and (sm_count // 2 <= keep_work * one_tile_splits <= sm_count)
        ):
            candidate = _SparseMlaTuning(
                family="keep",
                tile_size_q=64,
                split_kv=one_tile_splits,
                head_dim_ctas=1,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=True,
                reuse_kv=True,
                reuse_kv_stages=10,
            )
        if (
            num_heads in (32, 64)
            and capacity >= 512
            and (max_rows >= (sm_count if num_heads == 32 else 2 * sm_count))
        ):
            candidate = _SparseMlaTuning(
                family="keep",
                tile_size_q=64,
                split_kv=1,
                head_dim_ctas=1,
                gather_issue_warps=4,
                offset_cache="coalesced",
                fuse_epilogue=True,
                direct_inputs=False,
                reuse_kv=True,
                reuse_kv_stages=10,
                uniform_offset_cache=True,
                scheduler="clc",
            )
    elif q_data_type == "bf16":
        throughput_bf16 = (
            num_heads >= 32
            and (max_rows >= sm_count or throughput_h32)
            and (capacity >= 512)
        )
        short_list = capacity <= 256 and num_heads <= 64
        tile = (
            (8 if num_heads <= 16 else 16)
            if short_list
            else 16
            if num_heads <= 32
            else 64
        )
        if throughput_bf16:
            tile = 64
        candidate = _SparseMlaTuning(
            family="swap" if tile < 64 else "keep",
            tile_size_q=tile,
            split_kv=1 if short_list or throughput_bf16 else 0,
            head_dim_ctas=1 if throughput_bf16 else 0,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=tile < 64,
            reuse_kv=throughput_bf16,
            reuse_kv_stages=4 if throughput_bf16 else 0,
            uniform_offset_cache=throughput_bf16
            and max_rows >= sm_count
            and (
                capacity >= 2048
                or (capacity >= 1024 and max_rows >= 2 * sm_count)
                or max_rows >= 8 * sm_count
            ),
        )
        cluster_splits = 1 << ((capacity + 255) // 256 - 1).bit_length()
        m16_work = max_rows * ((num_heads + 15) // 16)
        if (
            not short_list
            and tile == 16
            and (1 < cluster_splits <= 4)
            and (num_heads % 16 == 0)
            and (m16_work * cluster_splits <= sm_count // 2)
        ):
            candidate = replace(candidate, split_kv=cluster_splits, reduction="cluster")
    options.append(_Candidate(1, "base_parallelism_and_traffic", candidate))
    if (
        q_data_type == "bf16"
        and (capacity >= 512)
        and (max_rows * 4 >= sm_count)
        and (num_heads == 128 or (num_heads in (32, 64) and max_rows * 2 <= sm_count))
    ):
        candidate = _SparseMlaTuning(
            family="2cta",
            tile_size_q=128,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            kv_pipeline_stages=8,
            fuse_epilogue=True,
            uniform_offset_cache=num_heads == 128,
            scheduler="clc"
            if num_heads == 128 and max_rows >= sm_count
            else "nonpersistent",
        )
        options.append(_Candidate(2, "bf16_wide_cluster_grid", candidate))
    if (
        q_data_type == "bf16"
        and (num_heads == 128)
        and (2 <= sm_count // (2 * max_rows) <= 4)
        and (capacity >= 512)
    ):
        # Match the two head tiles to a grid needing two to four KV partitions
        # per resident wave. BK64 both retains two BF16 tiles and reduces the
        # padded work at split boundaries; large grids retain the M128 option.
        wave_splits = max(1, sm_count // 2 // max_rows)
        # Reduction accepts non-power-of-two partitions. Preserve the full
        # resident-wave budget, including three splits for intermediate grids.
        candidate = _SparseMlaTuning(
            family="keep",
            tile_size_q=64,
            split_kv=wave_splits,
            head_dim_ctas=1,
            gather_issue_warps=8,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            uniform_offset_cache=True,
            reuse_kv=True,
            reuse_kv_stages=10,
            kv_tile_size=64,
            page_pipeline_stages=2,
            defer_max_update=True,
        )
        options.append(_Candidate(3, "bf16_small_grid_reuse", candidate))
    if (
        q_data_type == "bf16"
        and (num_heads <= 16)
        and (max_rows * 2 >= sm_count)
        and (capacity >= 512)
    ):
        candidate = _SparseMlaTuning(
            family="swap",
            tile_size_q=8 if num_heads <= 8 else 16,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=6,
        )
        options.append(_Candidate(4, "bf16_small_head_reuse", candidate))
    if (
        q_data_type == "bf16"
        and (num_heads in (32, 64))
        and (max_rows * 2 >= sm_count)
        and (max_seq_len_q <= 8)
        and (capacity >= 512)
    ):
        candidate = _SparseMlaTuning(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            reuse_kv=True,
            reuse_kv_stages=5,
            uniform_offset_cache=capacity >= 2048
            and max_rows >= sm_count
            or (capacity >= 1024 and max_rows >= 4 * sm_count)
            or max_rows >= 8 * sm_count,
        )
        options.append(_Candidate(5, "bf16_wide_head_reuse", candidate))
    if (
        q_data_type == "fp8"
        and (num_heads <= 16)
        and (capacity >= 512)
        and (
            capacity <= 1024
            and max_rows >= (sm_count if num_heads > 8 else 2 * sm_count)
            or max_rows >= 8 * sm_count
        )
    ):
        candidate = _SparseMlaTuning(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=False,
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            scheduler="clc",
        )
        options.append(_Candidate(6, "fp8_small_head_throughput", candidate))
    if (
        q_data_type == "fp8"
        and (num_heads == 128)
        and (capacity >= 512)
        and (max_rows * 2 >= sm_count)
    ):
        candidate = _SparseMlaTuning(
            family="2cta",
            tile_size_q=128,
            split_kv=1,
            fuse_epilogue=True,
            direct_inputs=True,
            scheduler="static",
        )
        options.append(_Candidate(7, "fp8_cluster_throughput", candidate))
    if (
        q_data_type == "bf16"
        and (num_heads in (8, 16, 32, 64))
        and (capacity >= 512)
        and (max_seq_len_q >= 128)
        and (max_rows >= 8 * sm_count)
    ):
        keep = num_heads >= 32
        candidate = _SparseMlaTuning(
            family="keep" if keep else "swap",
            tile_size_q=64 if keep else 8 if num_heads <= 8 else 16,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=False,
            balanced_registers=not keep,
            reuse_kv=True,
            reuse_kv_stages=5 if keep else 6,
            uniform_offset_cache=keep,
            scheduler="static",
        )
        options.append(_Candidate(8, "bf16_prefill_reuse", candidate))
    if (
        q_data_type == "fp8"
        and (num_heads == 64)
        and (capacity >= 512)
        and (max_rows * 2 >= sm_count)
    ):
        # A full M64 head tile and at least half an SM wave also apply to
        # intermediate/long prefill. The physical attention work is identical
        # after query flattening; no decode-only gate is required here.
        candidate = _SparseMlaTuning(
            family="keep",
            tile_size_q=64,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=8,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            reuse_kv=True,
            reuse_kv_stages=10,
            uniform_offset_cache=True,
            scheduler="clc",
            paired_correction=True,
        )
        # Callers supply source-tagged storage rows prepared outside attention.
        candidate = replace(candidate, direct_inputs=False)
        options.append(_Candidate(9, "fp8_h64_prepared_routes", candidate))
    if (
        q_data_type == "fp8"
        and (num_heads == 128)
        and (max_seq_len_q <= 8)
        and (capacity >= 512)
        and (16 <= max_rows <= sm_count // 4)
    ):
        small_long_grid = capacity >= 2048 and max_rows <= sm_count // 8
        candidate = _SparseMlaTuning(
            family="2cta",
            tile_size_q=128,
            split_kv=4 if small_long_grid else 1,
            fuse_epilogue=True,
            direct_inputs=True,
        )
        options.append(_Candidate(10, "fp8_short_cluster_grid", candidate))
    if (
        q_data_type == "bf16"
        and (num_heads <= 16)
        and (capacity >= 512)
        and (max_rows * 2 >= sm_count)
        and (
            max_seq_len_q <= 8
            or (
                num_heads <= 8
                and capacity <= 1024
                and (max_seq_len_q >= 128)
                and (max_rows >= 8 * sm_count)
            )
        )
    ):
        candidate = _SparseMlaTuning(
            family="swap",
            tile_size_q=8 if num_heads <= 8 else 16,
            split_kv=1,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=6
            if num_heads > 8 or (max_seq_len_q <= 8 and max_rows >= 4 * sm_count)
            else 5,
            single_kv_stream=True,
            defer_max_update=True,
            scheduler="clc",
        )
        options.append(_Candidate(11, "bf16_single_stream", candidate))
    if (
        q_data_type == "fp8"
        and (max_seq_len_q <= 8)
        and (capacity >= 512)
        and (
            num_heads <= 16
            and 32 <= max_rows <= sm_count
            and (
                not (num_heads <= 8 and capacity >= 2048 and (max_rows * 2 >= sm_count))
            )
            or (
                num_heads == 32
                and 16 <= max_rows <= sm_count // 8
                and (capacity >= 2048)
            )
        )
    ):
        base_ctas = max_rows * ((num_heads + 15) // 16)
        wave_splits = max(1, sm_count // base_ctas)
        wave_splits = 1 << wave_splits.bit_length() - 1
        candidate = _SparseMlaTuning(
            family="swap",
            tile_size_q=16,
            split_kv=wave_splits,
            head_dim_ctas=1,
            gather_issue_warps=4,
            offset_cache="coalesced",
            fuse_epilogue=True,
            direct_inputs=True,
            balanced_registers=True,
            reuse_kv=True,
            reuse_kv_stages=10,
            single_kv_stream=True,
        )
        options.append(_Candidate(12, "fp8_resident_split_grid", candidate))
    return options


def _refine_resources(tuning, *, rows, heads, query_length, capacity, dtype, sm_count):
    max_rows, num_heads, max_seq_len_q, q_data_type = (
        rows,
        heads,
        query_length,
        dtype,
    )
    if (
        q_data_type == "bf16"
        and (tuning.family == "keep")
        and (tuning.reuse_kv_stages == 5)
        and (num_heads in (32, 64))
        and (max_rows * 2 >= sm_count)
        and (capacity >= 512)
    ):
        tuning = replace(tuning, gather_issue_warps=8, uniform_offset_cache=True)
    if (
        q_data_type == "fp8"
        and (num_heads == 128)
        and (capacity >= 2048)
        and (max_seq_len_q <= 8)
        and (max_rows >= 4 * sm_count)
    ):
        tuning = replace(tuning, balanced_registers=True)
    if (
        q_data_type == "bf16"
        and (tuning.family == "keep")
        and (tuning.reuse_kv_stages == 5)
        and (num_heads in (32, 64))
    ):
        tuning = replace(tuning, defer_max_update=True)
        if max_seq_len_q <= 8 and max_rows >= 2 * sm_count:
            tuning = replace(tuning, scheduler="clc", direct_inputs=True)
        if (
            num_heads == 32
            and max_seq_len_q <= 8
            and (max_rows >= 4 * sm_count)
            and (capacity >= 1024)
        ):
            tuning = replace(
                tuning, kv_tile_size=64, reuse_kv_stages=10, page_pipeline_stages=2
            )
    if (
        dtype == "bf16"
        and tuning.family == "keep"
        and heads == 64
        and rows >= sm_count
        and capacity >= 512
        and tuning.split_kv == 1
    ):
        # One fully utilized M64 head tile, with more than one resident wave.
        # Two BK64 KV tiles fit where two BF16 BK128 tiles cannot; the second
        # page buffer permits QK-next/PV-current overlap. Qualified across
        # decode, intermediate prefill, long prefill and HCA prefixes.
        tuning = replace(
            tuning,
            kv_tile_size=64,
            reuse_kv=True,
            reuse_kv_stages=10,
            page_pipeline_stages=2,
            gather_issue_warps=8,
            uniform_offset_cache=True,
            direct_inputs=True,
            scheduler="clc",
            defer_max_update=True,
        )
    return tuning


def _resolve(candidate, *, rows, heads, capacity, sm_count, forced=False):
    tuning = candidate.tuning
    family = tuning.family
    if family == "auto":
        family = "swap" if heads <= 32 else "keep" if heads <= 64 else "2cta"
    if family not in ("swap", "keep", "2cta"):
        raise ValueError("invalid internal sparse MLA family")
    tile = tuning.tile_size_q or {"swap": 16, "keep": 64, "2cta": 128}[family]
    work = rows * ceil(heads / tile)
    splits = tuning.split_kv
    if splits == 0:
        # Match the task graph's one- or two-tile steady-state K step.
        step = 128 if tile >= 64 else 256
        steps = max(1, ceil(capacity / step))
        target = sm_count // 2 if family == "2cta" else sm_count
        partitions = min(steps, max(1, target // work), 128)
        splits = ceil(steps / ceil(steps / partitions))
    head_dim_ctas = tuning.head_dim_ctas
    if head_dim_ctas == 0:
        remaining = work * splits
        head_dim_ctas = (
            1
            if family == "2cta" or remaining * 2 > sm_count
            else 4
            if remaining * 4 <= sm_count
            else 2
        )
    if not forced and tuning.reuse_kv and head_dim_ctas != 1:
        tuning = replace(tuning, reuse_kv=False, reuse_kv_stages=0)
    return SparseMlaSelection(
        tuning,
        family,
        tile,
        splits,
        head_dim_ctas,
        f"{candidate.reason}; {work} head tiles, {splits} KV splits, {sm_count} SMs",
    )


def validate_selection(selection, *, dtype, shared_memory_bytes=227 * 1024):
    """Structural guards, independent of performance ranking.

    Kernel configuration retains the exact shared-memory/TMEM/register checks.
    The lower-bound storage check here rejects impossible reuse candidates
    without importing the DSL or allocating any device tensors.
    """
    t = selection.tuning
    if dtype == "fp8" and t.defer_max_update:
        raise ValueError("native FP8 requires exact maxima")
    family, tile = selection.family, selection.tile
    if t.scheduler not in ("nonpersistent", "static", "clc"):
        raise ValueError("invalid internal sparse MLA scheduler")
    if selection.splits not in range(1, 129):
        raise ValueError("split_kv must be between 1 and 128 (or zero for auto)")
    if selection.head_dim_ctas not in (1, 2, 4):
        raise ValueError("head_dim_ctas must be one, two or four")
    if family == "keep" and tile != 64:
        raise ValueError("keep-AB supports M64")
    if family == "swap" and tile not in (8, 16, 32):
        raise ValueError("swap-AB supports M8, M16 and M32")
    if family == "2cta" and tile != 128:
        raise ValueError("two-CTA sparse attention supports M128")
    if family != "2cta" and selection.splits > 1 and t.scheduler != "nonpersistent":
        raise ValueError("1CTA persistent scheduling does not support split-KV")
    if t.reuse_kv and selection.head_dim_ctas != 1:
        raise ValueError("KV reuse requires one V partition")
    if family == "2cta":
        if t.scheduler == "clc" and dtype == "fp8":
            raise ValueError("unsupported 2CTA sparse profile")
        if t.reuse_kv or t.single_kv_stream:
            raise ValueError("KV reuse and single-stream profiles require 1CTA")
        if (
            t.fuse_epilogue
            and t.scheduler != "nonpersistent"
            and (selection.splits != 1 or (t.direct_inputs and dtype != "fp8"))
        ):
            raise ValueError(
                "2CTA persistent fusion requires one split; direct routes require FP8 static scheduling"
            )
        if (
            t.gather_issue_warps not in ((1, 4, 8) if dtype == "bf16" else (1,))
            or t.offset_cache != "strided"
        ):
            raise ValueError("unsupported two-CTA gather configuration")
        if selection.head_dim_ctas != 1 or (
            t.scheduler == "static" and dtype == "bf16"
        ):
            raise ValueError("unsupported two-CTA partition/scheduler configuration")
    if t.reuse_kv and t.reuse_kv_stages:
        element_bytes = 1 if dtype == "fp8" else 2
        minimum_smem = (
            t.reuse_kv_stages * t.kv_tile_size * 128 + tile * 512
        ) * element_bytes
        if minimum_smem > shared_memory_bytes:
            raise ValueError("KV stages and Q tile exceed shared-memory capacity")


def select_sparse_mla_profile(
    *,
    rows,
    heads,
    query_length,
    capacity,
    dtype,
    sm_count,
    forced=None,
    shared_memory_bytes=227 * 1024,
):
    """Choose one eligible ranked candidate using host-only workload facts."""
    if forced is not None:
        choice = _resolve(
            _Candidate(0, "explicit profile", forced),
            rows=rows,
            heads=heads,
            capacity=capacity,
            sm_count=sm_count,
            forced=True,
        )
        validate_selection(choice, dtype=dtype, shared_memory_bytes=shared_memory_bytes)
        return choice
    eligible = []
    for candidate in _candidates(
        rows=rows,
        heads=heads,
        query_length=query_length,
        capacity=capacity,
        dtype=dtype,
        sm_count=sm_count,
    ):
        refined = _refine_resources(
            candidate.tuning,
            rows=rows,
            heads=heads,
            query_length=query_length,
            capacity=capacity,
            dtype=dtype,
            sm_count=sm_count,
        )
        choice = _resolve(
            replace(candidate, tuning=refined),
            rows=rows,
            heads=heads,
            capacity=capacity,
            sm_count=sm_count,
        )
        try:
            validate_selection(
                choice, dtype=dtype, shared_memory_bytes=shared_memory_bytes
            )
        except ValueError:
            continue
        eligible.append((candidate.priority, choice))
    if not eligible:
        raise ValueError("no eligible sparse MLA profile for the planned workload")
    return max(eligible, key=lambda entry: entry[0])[1]
