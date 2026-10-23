# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side workspace contract for MXFP8 x MXFP4 MegaMoE.

This module intentionally has no Torch, CUDA, CuTe, or NVSHMEM dependency.
The persistent kernel and its frontend use the same region table, while unit
tests can check every size and offset on a CPU-only host.

The dispatch payload is E4M3 (one byte per element).  Its E8M0 sideband has
one byte per consecutive K32 block and is moved by ``TokenInPullTokenBackPush``
in uint32 atoms, hence ``ceil(hidden / 128)`` uint32 values per token.  Source
rows must be zero-padded through the final partial uint32 atom.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Literal


MXFP8_SF_VEC_SIZE = 32
SF_BYTES_PER_UINT32 = 4
SF_ATOM_TOKEN_ROWS = 128
TOKEN_METADATA_BYTES = 8
GRID_SYNC_SLOT_COUNT = 2
NVLINK_SIGNAL_SLOT_COUNT = 2


def round_up(value: int, alignment: int) -> int:
    if value < 0:
        raise ValueError(f"value must be non-negative, got {value}")
    if alignment <= 0:
        raise ValueError(f"alignment must be positive, got {alignment}")
    return ((value + alignment - 1) // alignment) * alignment


def sf_uint32_per_token(hidden: int) -> int:
    """Number of uint32 transport atoms for one E8M0/K32 scale row."""

    if hidden <= 0 or hidden % MXFP8_SF_VEC_SIZE:
        raise ValueError("hidden must be a positive multiple of 32")
    sf_bytes = hidden // MXFP8_SF_VEC_SIZE
    return round_up(sf_bytes, SF_BYTES_PER_UINT32) // SF_BYTES_PER_UINT32


@dataclass(frozen=True)
class RegionSpec:
    """One typed region in an opaque byte workspace."""

    name: str
    dtype: str
    itemsize: int
    shape: tuple[int, ...]
    align: int

    def __post_init__(self) -> None:
        if self.itemsize <= 0:
            raise ValueError(f"{self.name}: itemsize must be positive")
        if self.align <= 0 or self.align & (self.align - 1):
            raise ValueError(f"{self.name}: align must be a positive power of two")
        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"{self.name}: shape dimensions must be non-negative")

    @property
    def numel(self) -> int:
        return prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.numel * self.itemsize

    @property
    def stride_row_major(self) -> tuple[int, ...]:
        if not self.shape:
            return ()
        strides = [1]
        for dim in reversed(self.shape[1:]):
            strides.append(strides[-1] * dim)
        return tuple(reversed(strides))


@dataclass(frozen=True)
class LaidOutRegion:
    spec: RegionSpec
    offset: int

    @property
    def end(self) -> int:
        return self.offset + self.spec.nbytes


@dataclass(frozen=True)
class RegionLayout:
    regions: tuple[LaidOutRegion, ...]
    total_bytes: int
    zero_prefix_bytes: int

    def __post_init__(self) -> None:
        names = [region.spec.name for region in self.regions]
        if len(names) != len(set(names)):
            raise ValueError("workspace region names must be unique")
        if self.zero_prefix_bytes % 4:
            raise ValueError("counter zero prefix must be int32 aligned")

    @property
    def zero_prefix_i32_count(self) -> int:
        return self.zero_prefix_bytes // 4

    def region(self, name: str) -> LaidOutRegion:
        for region in self.regions:
            if region.spec.name == name:
                return region
        raise KeyError(name)


def layout_regions(
    specs: tuple[RegionSpec, ...], *, first_data_region: str
) -> RegionLayout:
    cursor = 0
    laid_out: list[LaidOutRegion] = []
    first_data_offset: int | None = None
    for spec in specs:
        cursor = round_up(cursor, spec.align)
        if cursor % spec.itemsize:
            raise ValueError(f"{spec.name}: offset is not dtype aligned")
        if spec.name == first_data_region:
            first_data_offset = cursor
        laid_out.append(LaidOutRegion(spec, cursor))
        cursor += spec.nbytes
    if first_data_offset is None:
        raise ValueError(f"first data region {first_data_region!r} is absent")
    return RegionLayout(tuple(laid_out), round_up(cursor, 16), first_data_offset)


@dataclass(frozen=True)
class WorkspaceConfig:
    """Static inputs needed to size one rank's local and symmetric pools."""

    world_size: int
    num_topk: int
    num_experts_per_rank: int
    max_tokens_per_rank: int
    hidden: int
    intermediate: int
    token_padding_block: int
    sf_padding_block: int
    cluster_tile_tokens: int
    load_balance_mode: Literal["static", "atomic_counter", "alternating"] = "static"

    def __post_init__(self) -> None:
        positive = {
            "world_size": self.world_size,
            "num_topk": self.num_topk,
            "num_experts_per_rank": self.num_experts_per_rank,
            "max_tokens_per_rank": self.max_tokens_per_rank,
            "hidden": self.hidden,
            "intermediate": self.intermediate,
            "token_padding_block": self.token_padding_block,
            "sf_padding_block": self.sf_padding_block,
            "cluster_tile_tokens": self.cluster_tile_tokens,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.num_topk > 32:
            raise ValueError("num_topk must be at most 32")
        if self.hidden % MXFP8_SF_VEC_SIZE:
            raise ValueError("hidden must be a multiple of 32")
        if self.intermediate % MXFP8_SF_VEC_SIZE:
            raise ValueError("intermediate must be a multiple of 32")
        if self.sf_padding_block < self.token_padding_block:
            raise ValueError("sf_padding_block must cover token_padding_block")
        if self.sf_padding_block % SF_ATOM_TOKEN_ROWS:
            raise ValueError("sf_padding_block must be a multiple of 128")
        if self.cluster_tile_tokens % self.token_padding_block:
            raise ValueError(
                "cluster_tile_tokens must be a multiple of token_padding_block"
            )
        if self.load_balance_mode not in ("static", "atomic_counter", "alternating"):
            raise ValueError("unsupported load_balance_mode")

    @property
    def num_total_experts(self) -> int:
        return self.world_size * self.num_experts_per_rank

    @property
    def sf_uint32_per_token(self) -> int:
        return sf_uint32_per_token(self.hidden)

    @property
    def pool_token_capacity(self) -> int:
        max_recv = self.world_size * self.max_tokens_per_rank
        max_edges_per_token = min(self.num_topk, self.num_experts_per_rank)
        raw = max_recv * max_edges_per_token + self.num_experts_per_rank * (
            self.token_padding_block - 1
        )
        return round_up(raw, self.token_padding_block)

    @property
    def pool_sf_capacity(self) -> int:
        return (
            self.pool_token_capacity // self.token_padding_block
        ) * self.sf_padding_block

    @property
    def pool_task_tile_capacity(self) -> int:
        return (
            round_up(self.pool_token_capacity, self.cluster_tile_tokens)
            // self.cluster_tile_tokens
            + self.num_experts_per_rank
        )

    @property
    def fc1_done_slots(self) -> int:
        return (
            round_up(self.pool_token_capacity, self.cluster_tile_tokens)
            // self.cluster_tile_tokens
            + self.num_experts_per_rank
        )


def _region(
    name: str, dtype: str, itemsize: int, shape: tuple[int, ...], align: int = 16
) -> RegionSpec:
    return RegionSpec(name, dtype, itemsize, shape, align)


def local_region_specs(config: WorkspaceConfig) -> tuple[RegionSpec, ...]:
    """Return local regions in reset/data order.

    ``l1_topk_weights_buffer`` is compatibility baggage for the currently
    vendored communication helper, which unconditionally transports a score.
    The MXFP8 x MXFP4 compute path must not consume it: authoritative FP32
    routing scores remain on the home rank and are applied after FC2.
    """

    pool = config.pool_token_capacity
    specs: list[RegionSpec] = [
        _region("l1_arrival_count", "int32", 4, (config.pool_task_tile_capacity,)),
        _region("expert_send_count", "int64", 8, (config.num_total_experts,)),
        _region("fc1_done_counter", "int32", 4, (config.fc1_done_slots,)),
    ]
    if config.load_balance_mode != "static":
        # Alternating uses independent FC1/FC2 counters. Both live in the
        # zero-prefix region reset by the kernel tail on every replay.
        count = 2 if config.load_balance_mode == "alternating" else 1
        specs.append(_region("load_balance_counter", "int32", 4, (count,)))

    sf_cols = round_up(config.intermediate // MXFP8_SF_VEC_SIZE, 4)
    sf_rows_upper = pool + config.num_experts_per_rank * config.sf_padding_block
    specs.extend(
        [
            _region("l1_token_buffer", "uint8", 1, (pool, config.hidden), 128),
            # Both synchronization protocols carry their phase across launches.
            # Only first-use initialization may clear these regions.
            _region("grid_sync_counter", "int32", 4, (GRID_SYNC_SLOT_COUNT,)),
            _region("nvlink_barrier_counter", "int32", 4, (1,)),
            _region(
                "l1_sf_buffer",
                "int32",
                4,
                (config.pool_sf_capacity * config.sf_uint32_per_token,),
            ),
            _region("l1_topk_weights_buffer", "float32", 4, (pool,)),
            _region("token_src_metadata", "uint8", 1, (pool, TOKEN_METADATA_BYTES)),
            _region("fc1_output", "float8_e4m3", 1, (pool, config.intermediate), 128),
            _region(
                "fc1_output_sf",
                "float8_e8m0",
                1,
                (sf_rows_upper, sf_cols),
                128,
            ),
        ]
    )
    from .wait_stats_config import MEGA_STATS_WORDS, wait_stats_enabled

    if wait_stats_enabled():
        # Debug-only; after the data regions so the tail's zero-prefix reset
        # keeps it. Accumulates across launches; the reader zeroes it. The
        # per-CTA block is followed by the start-up ring.
        specs.append(_region("wait_stats", "int32", 4, (MEGA_STATS_WORDS,)))
    return tuple(specs)


def shared_region_specs(config: WorkspaceConfig) -> tuple[RegionSpec, ...]:
    max_slot = config.max_tokens_per_rank * config.num_topk
    return (
        _region(
            "expert_recv_count",
            "int64",
            8,
            (config.world_size, config.num_experts_per_rank),
        ),
        _region("expert_recv_count_sum", "int64", 8, (config.num_experts_per_rank,)),
        _region(
            "src_token_topk_idx",
            "int32",
            4,
            (config.num_experts_per_rank, config.world_size, max_slot),
        ),
        _region("nvlink_barrier_signal", "int32", 4, (NVLINK_SIGNAL_SLOT_COUNT,)),
    )


@dataclass(frozen=True)
class WorkspacePlan:
    config: WorkspaceConfig
    local: RegionLayout
    shared: RegionLayout

    @property
    def sizes(self) -> tuple[int, int]:
        return self.local.total_bytes, self.shared.total_bytes


def make_workspace_plan(config: WorkspaceConfig) -> WorkspacePlan:
    return WorkspacePlan(
        config=config,
        local=layout_regions(
            local_region_specs(config), first_data_region="l1_token_buffer"
        ),
        shared=layout_regions(
            shared_region_specs(config), first_data_region="src_token_topk_idx"
        ),
    )
