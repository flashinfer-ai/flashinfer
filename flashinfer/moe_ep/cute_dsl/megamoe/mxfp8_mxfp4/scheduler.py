# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent FC1/FC2 scheduler contract for MXFP8 x MXFP4 MegaMoE.

This module is deliberately host executable.  It defines the record ABI and
the exact state-machine/claiming semantics that the CuTe-DSL scheduler warp
must implement, without importing the vendored W4A4 kernel.  Keeping the model
host-only makes scheduler changes cheap to validate and gives the integration
kernel a stable eight-int32 boundary.

The scheduler walks this hierarchy::

    group -> FC1/FC2 phase -> expert -> token block -> feature block

Static and atomic-counter modes order all FC1 claims before FC2 in each
group. ``AlternatingTileClaimer`` instead claims from two whole-batch task
spaces after an FC1 warm-up; ``record_for_alternating`` decodes its signed
phase-local indices. FC2 cannot be returned until its FC1 tasks are claimed.
Token blocks are the
slow axis in both phases.  ``tile_m_idx`` is the feature tile and
``tile_n_idx`` is the token tile, matching the swap-AB kernel orientation.

MXFP8 activations and MXFP4 weights both use E8M0 scales per K32.  K32 is a
format invariant; token-row padding remains independently configurable because
the data and scale workspaces can use different physical row padding.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
from typing import Iterable, Literal, Mapping, Sequence


SCALE_VECTOR_SIZE = 32
WORK_RECORD_WORDS = 8
PHASE_BITS = 16
PHASE_MASK = (1 << PHASE_BITS) - 1
PEEK_READY_BIT = 1 << PHASE_BITS
DONE_EXPERT = -1


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _round_up(value: int, alignment: int) -> int:
    return _ceil_div(value, alignment) * alignment


def longest_first_order(counts: Sequence[int]) -> tuple[int, ...]:
    """Visit largest experts first, breaking ties by their pool index."""
    return tuple(sorted(range(len(counts)), key=lambda e: (-counts[e], e)))


class Phase(IntEnum):
    """FC phase encoded in the low 16 bits of ``phase_and_peek``."""

    NONE = 0
    FC1 = 1
    FC2 = 2


@dataclass(frozen=True)
class WorkRecord:
    """Stable scheduler-to-mainloop/epilogue ABI.

    ``to_words`` is the canonical SMEM wire order.  Slot 3 intentionally
    occupies the base scheduler's historic ``k_tile_cnt`` position, but its
    MegaMoE meaning is ``cumulative_data_physical_row``.
    """

    expert_idx: int
    tile_m_idx: int
    tile_n_idx: int
    cumulative_data_physical_row: int
    cumulative_sf_physical_row: int
    cumulative_token_block_count: int
    valid_tokens_in_cta_tile: int
    phase_and_peek: int

    @property
    def is_valid(self) -> bool:
        return self.expert_idx >= 0

    @property
    def phase(self) -> Phase:
        return Phase(self.phase_and_peek & PHASE_MASK)

    @property
    def peek_ready(self) -> bool:
        return bool(self.phase_and_peek & PEEK_READY_BIT)

    @property
    def counter_slot(self) -> int:
        """Per-token-tile publication-counter slot used by both phases."""

        return self.cumulative_token_block_count + self.tile_n_idx

    def with_peek_ready(self, ready: bool) -> "WorkRecord":
        packed = self.phase_and_peek & ~PEEK_READY_BIT
        if ready:
            packed |= PEEK_READY_BIT
        return replace(self, phase_and_peek=packed)

    def to_words(self) -> tuple[int, ...]:
        return (
            self.expert_idx,
            self.tile_m_idx,
            self.tile_n_idx,
            self.cumulative_data_physical_row,
            self.cumulative_sf_physical_row,
            self.cumulative_token_block_count,
            self.valid_tokens_in_cta_tile,
            self.phase_and_peek,
        )

    @classmethod
    def from_words(cls, words: Sequence[int]) -> "WorkRecord":
        if len(words) != WORK_RECORD_WORDS:
            raise ValueError(
                f"work record requires {WORK_RECORD_WORDS} int32 words, "
                f"got {len(words)}"
            )
        return cls(*(int(word) for word in words))

    @classmethod
    def done(cls) -> "WorkRecord":
        return cls(DONE_EXPERT, 0, 0, 0, 0, 0, 0, int(Phase.NONE))


@dataclass(frozen=True)
class SchedulerConfig:
    """Compile/launch-time scheduler parameters.

    ``gate_up_size`` is the full interleaved FC1 output dimension (2 * I),
    not the post-SwiGLU intermediate dimension.  ``cluster_shape_mn`` is in
    scheduler-internal token/feature order.
    """

    expert_token_counts: tuple[int, ...]
    gate_up_size: int
    hidden_size: int
    cta_tile_m: int
    cta_tile_n: int
    cluster_shape_mn: tuple[int, int]
    group_hint: int
    token_padding_block: int
    sf_padding_block: int
    num_persistent_clusters: int
    fc2_ready_threshold: int
    expert_data_row_offsets: tuple[int, ...] | None = None
    expert_scale_row_offsets: tuple[int, ...] | None = None
    wait_for_dispatch: bool = True
    claim_order: Literal["expert", "longest_first"] = "expert"

    def __post_init__(self) -> None:
        if self.claim_order not in ("expert", "longest_first"):
            raise ValueError("claim_order must be expert or longest_first")
        positive = {
            "gate_up_size": self.gate_up_size,
            "hidden_size": self.hidden_size,
            "cta_tile_m": self.cta_tile_m,
            "cta_tile_n": self.cta_tile_n,
            "group_hint": self.group_hint,
            "token_padding_block": self.token_padding_block,
            "sf_padding_block": self.sf_padding_block,
            "num_persistent_clusters": self.num_persistent_clusters,
            "fc2_ready_threshold": self.fc2_ready_threshold,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if len(self.cluster_shape_mn) != 2 or any(
            extent <= 0 for extent in self.cluster_shape_mn
        ):
            raise ValueError("cluster_shape_mn must contain two positive extents")
        if not self.expert_token_counts:
            raise ValueError("expert_token_counts must not be empty")
        if any(count < 0 for count in self.expert_token_counts):
            raise ValueError("expert token counts must be non-negative")
        if self.gate_up_size % SCALE_VECTOR_SIZE:
            raise ValueError("gate_up_size must be divisible by MX block size 32")
        if self.hidden_size % SCALE_VECTOR_SIZE:
            raise ValueError("hidden_size must be divisible by MX block size 32")
        if (self.expert_data_row_offsets is None) != (
            self.expert_scale_row_offsets is None
        ):
            raise ValueError("data and scale row offsets must be provided together")
        for name, alignment in (
            ("expert_data_row_offsets", self.token_padding_block),
            ("expert_scale_row_offsets", self.sf_padding_block),
        ):
            offsets = getattr(self, name)
            if offsets is None:
                continue
            if len(offsets) != len(self.expert_token_counts) + 1 or offsets[0] != 0:
                raise ValueError(f"{name} must contain E+1 offsets starting at zero")
            if any(value < 0 or value % alignment for value in offsets):
                raise ValueError(f"{name} must contain aligned nonnegative offsets")
            for index, count in enumerate(self.expert_token_counts):
                if offsets[index + 1] - offsets[index] < count:
                    raise ValueError(f"{name} spans must contain each expert's rows")

    @property
    def cluster_tile_m(self) -> int:
        return self.cta_tile_m * self.cluster_shape_mn[0]

    @property
    def cluster_tile_n(self) -> int:
        return self.cta_tile_n * self.cluster_shape_mn[1]

    @property
    def fc1_feature_blocks(self) -> int:
        return _ceil_div(self.gate_up_size, self.cluster_tile_n)

    @property
    def fc2_feature_blocks(self) -> int:
        return _ceil_div(self.hidden_size, self.cluster_tile_n)


@dataclass(frozen=True)
class _ExpertLayout:
    token_count: int
    data_row: int
    sf_row: int
    token_block_row: int
    token_blocks: int


@dataclass(frozen=True)
class _ClusterWork:
    expert_idx: int
    token_block_idx: int
    feature_block_idx: int
    phase: Phase


class StaticTileClaimer:
    """Persistent-grid strided claiming: cluster + k * cluster_count."""

    def __init__(self, *, cluster_id: int, cluster_count: int) -> None:
        if cluster_count <= 0:
            raise ValueError("cluster_count must be positive")
        if cluster_id < 0 or cluster_id >= cluster_count:
            raise ValueError("cluster_id must be in [0, cluster_count)")
        self._next = cluster_id
        self._stride = cluster_count

    def claim(self) -> int:
        result = self._next
        self._next += self._stride
        return result


class AtomicTileClaimer:
    """Host mirror of leader-lane ``atomic_add(counter, 1)`` claiming.

    One instance models the global counter shared by all persistent clusters.
    The device scheduler broadcasts each returned value to the other CTAs in
    its cluster; consequently callers claim once per cluster, not once per CTA.
    """

    def __init__(self, initial: int = 0) -> None:
        if initial < 0:
            raise ValueError("initial counter must be non-negative")
        self._next = initial

    @property
    def value(self) -> int:
        return self._next

    def claim(self) -> int:
        result = self._next
        self._next += 1
        return result


def get_num_fc1_warmup_waves(total_blocks, clusters, fc1_n, fc2_n):
    """Warm-up bound for whole-batch FC1/FC2 alternation.

    Cover every token block touched by the first FC2 wave, and leave enough
    FC1 waves ahead that the narrower FC2 task space cannot catch its frontier.
    There is no pool-ring bound: MegaMoE retains the full FC1 intermediate.
    """
    if total_blocks == 0:
        return 0
    first_wave_blocks = min(total_blocks, _ceil_div(clusters, fc2_n))
    cover_first_wave = _ceil_div(first_wave_blocks * fc1_n, clusters)
    prevent_catchup = (
        _ceil_div(fc1_n + (total_blocks - 1) * max(fc1_n - fc2_n, 0), clusters) + 1
    )
    return min(
        _ceil_div(total_blocks * fc1_n, clusters),
        max(cover_first_wave, prevent_catchup),
    )


class AlternatingTileClaimer:
    """One cluster's state, with two counters shared by all clusters.

    ``claim`` returns None while a late FC2 claim waits for the FC1 frontier.
    The caller must let other clusters progress and retry. DONE is encoded as
    INT32_MAX; FC1 uses nonnegative indices and FC2 uses -(index + 1).
    """

    DONE = (1 << 31) - 1

    def __init__(self, config, fc1_counter, fc2_counter):
        blocks = sum(
            _ceil_div(n, config.cluster_tile_m) for n in config.expert_token_counts
        )
        self.fc1_n = config.fc1_feature_blocks
        self.fc2_n = config.fc2_feature_blocks
        self.total_fc1 = blocks * self.fc1_n
        self.total_fc2 = blocks * self.fc2_n
        self.pending = get_num_fc1_warmup_waves(
            blocks, config.num_persistent_clusters, self.fc1_n, self.fc2_n
        )
        self.fc1_counter = fc1_counter
        self.fc2_counter = fc2_counter
        self.fc1_exhausted = False
        self.fc2_exhausted = False
        self.waiting = None

    def claim(self):
        while self.waiting is None:
            if self.fc1_exhausted and self.fc2_exhausted:
                return self.DONE
            if self.pending or self.fc2_exhausted:
                index = self.fc1_counter.claim()
                if index < self.total_fc1:
                    self.pending = max(0, self.pending - 1)
                    return index
                self.fc1_exhausted = True
                self.pending = 0
            else:
                index = self.fc2_counter.claim()
                self.pending = int(not self.fc1_exhausted)
                if index < self.total_fc2:
                    self.waiting = index
                else:
                    self.fc2_exhausted = True
        index = self.waiting
        if self.fc1_counter.value < (index // self.fc2_n + 1) * self.fc1_n:
            return None
        self.waiting = None
        return -(index + 1)


class PersistentFc1Fc2Scheduler:
    """Host specification for the fused persistent scheduler state machine."""

    def __init__(self, config: SchedulerConfig) -> None:
        self.config = config
        self._experts = self._build_expert_layouts()
        self.order = (
            longest_first_order(config.expert_token_counts)
            if config.claim_order == "longest_first"
            else tuple(range(len(self._experts)))
        )
        self._cluster_work = tuple(self._build_cluster_work())

    def _build_expert_layouts(self) -> tuple[_ExpertLayout, ...]:
        data_row = 0
        sf_row = 0
        token_block_row = 0
        layouts: list[_ExpertLayout] = []
        for expert_idx, token_count in enumerate(self.config.expert_token_counts):
            if self.config.expert_data_row_offsets is not None:
                data_row = self.config.expert_data_row_offsets[expert_idx]
                sf_row = self.config.expert_scale_row_offsets[expert_idx]
            token_blocks = _ceil_div(token_count, self.config.cluster_tile_m)
            layouts.append(
                _ExpertLayout(
                    token_count=token_count,
                    data_row=data_row,
                    sf_row=sf_row,
                    token_block_row=token_block_row,
                    token_blocks=token_blocks,
                )
            )
            data_row += _round_up(token_count, self.config.token_padding_block)
            sf_row += _round_up(token_count, self.config.sf_padding_block)
            token_block_row += token_blocks
        return tuple(layouts)

    def _groups(self) -> Iterable[range]:
        """Greedy group boundaries in visitation order, not pool index order."""
        first = 0
        expert_count = len(self._experts)
        while first < expert_count:
            last = first
            fc1_tiles = 0
            while last < expert_count and fc1_tiles < self.config.group_hint:
                fc1_tiles += (
                    self._experts[self.order[last]].token_blocks
                    * self.config.fc1_feature_blocks
                )
                last += 1
            # A suffix of zero-token experts cannot reach group_hint.  The
            # loop still consumes the suffix, so progress is guaranteed.
            yield range(first, last)
            first = last

    def _build_cluster_work(self) -> Iterable[_ClusterWork]:
        for positions in self._groups():
            for phase, feature_blocks in (
                (Phase.FC1, self.config.fc1_feature_blocks),
                (Phase.FC2, self.config.fc2_feature_blocks),
            ):
                for pos in positions:
                    expert_idx = self.order[pos]
                    token_blocks = self._experts[expert_idx].token_blocks
                    for token_block_idx in range(token_blocks):
                        for feature_block_idx in range(feature_blocks):
                            yield _ClusterWork(
                                expert_idx,
                                token_block_idx,
                                feature_block_idx,
                                phase,
                            )

    @property
    def cluster_tile_count(self) -> int:
        return len(self._cluster_work)

    def record_for(
        self,
        cluster_linear_tile_idx: int,
        *,
        cta_coord_mn: tuple[int, int] = (0, 0),
    ) -> WorkRecord:
        """Decode a claimed cluster tile into one CTA's work record."""

        if cluster_linear_tile_idx < 0:
            raise ValueError("cluster_linear_tile_idx must be non-negative")
        if cluster_linear_tile_idx >= self.cluster_tile_count:
            return WorkRecord.done()

        cta_m, cta_n = cta_coord_mn
        cluster_m, cluster_n = self.config.cluster_shape_mn
        if cta_m < 0 or cta_m >= cluster_m or cta_n < 0 or cta_n >= cluster_n:
            raise ValueError("cta_coord_mn must lie inside cluster_shape_mn")

        work = self._cluster_work[cluster_linear_tile_idx]
        expert = self._experts[work.expert_idx]
        cta_token_block = work.token_block_idx * cluster_m + cta_m
        cta_feature_block = work.feature_block_idx * cluster_n + cta_n
        token_start = cta_token_block * self.config.cta_tile_m
        valid_tokens = min(
            max(expert.token_count - token_start, 0), self.config.cta_tile_m
        )
        return WorkRecord(
            expert_idx=work.expert_idx,
            tile_m_idx=cta_feature_block,
            tile_n_idx=cta_token_block,
            cumulative_data_physical_row=expert.data_row,
            cumulative_sf_physical_row=expert.sf_row,
            cumulative_token_block_count=expert.token_block_row,
            valid_tokens_in_cta_tile=valid_tokens,
            phase_and_peek=int(work.phase),
        )

    def record_for_alternating(self, claim, *, cta_coord_mn=(0, 0)):
        """Decode the signed phase-local claim, ignoring group_hint."""
        if claim == AlternatingTileClaimer.DONE:
            return WorkRecord.done()
        phase = Phase.FC1 if claim >= 0 else Phase.FC2
        index = claim if claim >= 0 else -claim - 1
        # Reuse the canonical record decoder, with the phase-local ordering.
        indices = [
            i for i, work in enumerate(self._cluster_work) if work.phase == phase
        ]
        if index >= len(indices):
            return WorkRecord.done()
        return self.record_for(indices[index], cta_coord_mn=cta_coord_mn)

    def enrich_readiness_peek(
        self,
        record: WorkRecord,
        *,
        fc1_ready_counters: Mapping[int, int] | Sequence[int] | None = None,
        fc1_done_counters: Mapping[int, int] | Sequence[int] | None = None,
    ) -> WorkRecord:
        """Pack a one-shot readiness-counter observation into bit 16.

        FC1 compares the dispatch counter against this CTA's valid token count.
        FC2 compares the FC1 publication counter against the configured fixed
        threshold.  This function never spins and never performs arithmetic on
        routing weights.
        """

        if not record.is_valid:
            return record.with_peek_ready(False)
        if record.phase is Phase.FC1 and not self.config.wait_for_dispatch:
            return record.with_peek_ready(True)
        counters = (
            fc1_ready_counters if record.phase is Phase.FC1 else fc1_done_counters
        )
        if counters is None:
            return record.with_peek_ready(False)
        observed = counters[record.counter_slot]
        threshold = (
            record.valid_tokens_in_cta_tile
            if record.phase is Phase.FC1
            else self.config.fc2_ready_threshold
        )
        return record.with_peek_ready(observed >= threshold)

    def records_for_claimer(
        self,
        claimer: StaticTileClaimer | AtomicTileClaimer,
        *,
        cta_coord_mn: tuple[int, int] = (0, 0),
    ) -> Iterable[WorkRecord]:
        """Yield valid records until a claim decodes to the DONE sentinel."""

        while True:
            record = self.record_for(claimer.claim(), cta_coord_mn=cta_coord_mn)
            if not record.is_valid:
                return
            yield record


__all__ = [
    "AlternatingTileClaimer",
    "get_num_fc1_warmup_waves",
    "longest_first_order",
    "AtomicTileClaimer",
    "DONE_EXPERT",
    "PEEK_READY_BIT",
    "PHASE_BITS",
    "PHASE_MASK",
    "PersistentFc1Fc2Scheduler",
    "Phase",
    "SCALE_VECTOR_SIZE",
    "SchedulerConfig",
    "StaticTileClaimer",
    "WORK_RECORD_WORDS",
    "WorkRecord",
]
