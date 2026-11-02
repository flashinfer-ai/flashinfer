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

# Host plan of the Cake W8A8 MXFP8 MegaMoE EP16 route: geometry constants,
# per-stage launch grids, static tile maps and buffer sizes.
#
# Everything here is a host-known scalar or a routing-independent tensor. The
# bindings receive grids and SM-derived scalars from this module and never
# query device properties themselves.

from __future__ import annotations

import functools
from dataclasses import dataclass

import torch

# Public identifier of the generated route (manifest ``cake.route_id``).
ROUTE_ID = "w8a8_megamoe_ep16_staged_v4"

# ---------------------------------------------------------------------------
# Route geometry (fixed by the device programs).
# ---------------------------------------------------------------------------
WORLD_SIZE = 16
LOCAL_EXPERTS = 32
NUM_EXPERTS = WORLD_SIZE * LOCAL_EXPERTS
TOP_K = 8
HIDDEN = 3072
INTERMEDIATE = 5120
FC1_ROWS = 2 * INTERMEDIATE  # gate || up rows of w13
MAX_TOKENS_PER_RANK = 64
MAX_RECV_TOKENS = WORLD_SIZE * MAX_TOKENS_PER_RANK
MAX_ROUTES = MAX_RECV_TOKENS * TOP_K
MAX_ROWS_PER_EXPERT = 128
TOTAL_EXPERT_ROWS = LOCAL_EXPERTS * MAX_ROWS_PER_EXPERT
MAX_ROUTES_PER_EXPERT = 32  # one N32 SwapAB row tile per expert per forward
SOURCE_SCALE_BLOCKS = HIDDEN // 32  # block-32 MXFP8 scales per published token
BLOCK_M = 128
BLOCK_K = 256
SCALE_TILE_BYTES = 1024  # one packed K-block scale tile
FC1_K128_TILES = HIDDEN // 128
FC2_K128_TILES = INTERMEDIATE // 128
FC1_N256_TILES = FC1_ROWS // 256  # 40 FC1 tiles per expert (SwapAB FC1)
FC2_SWAP_TILES = HIDDEN // BLOCK_M  # 24 FC2 hidden tiles per expert (SwapAB FC2)
FC1_SWAP_GRID_M = LOCAL_EXPERTS * FC1_N256_TILES  # 1280
FC2_SWAP_GRID_M = LOCAL_EXPERTS * FC2_SWAP_TILES  # 768
# Combine CTAs per token: one per 1024-column slice of the hidden axis.
COMBINE_CTAS_PER_TOKEN = 3
# local_pack packs and maps only routes whose per-expert row is below this
# bound (the SwapAB N32 route tile) and counts every further route in
# ``overflow`` instead of packing it; publish_push zeroes ``overflow`` every
# forward. The binding admits 1..128 (the per-expert row capacity); the route
# uses 32.
ACTIVE_ROWS_PER_EXPERT = MAX_ROUTES_PER_EXPERT  # 32
SCALE_TILE_K128 = 1  # local_pack writes K128 scale tiles for the SwapAB FC1
PG_FLAG_WORDS = 2 + 2 * 2 * WORLD_SIZE  # two double-buffered barrier phases
# Device-side peer pointer table slots. The host passes exactly WORLD_SIZE
# entries; the binding fills slots 0..15 and zeroes the rest.
PEER_TABLE_CAPACITY = 32
QUANTIZE_BLOCKS_PER_SM = 4

# The five forward programs in launch order (manifest ``forward_stages``).
# ``publish_push`` also zeroes this rank's combine read window of the term
# buffer and resets the routing state (expert counts, route map, overflow
# flag), so no reset program and no host memset sit between the launches;
# there is no input barrier (``local_pack`` waits on its local inbox flags)
# and no partial barrier (``fc2`` signals its completion into every rank's
# ``fc2_completion`` counter and ``combine`` waits on its local counter).
STAGES = (
    "publish_push",
    "local_pack",
    "fc1",
    "fc2",
    "combine",
)

# Push dispatch (manifest ``route_contract.push_dispatch``). ``publish_push``
# quantizes each staged token into one 3232-byte slot
# (row_fp8 [0, 3072) | scales [3072, 96) | ids_i32 [3168, 32) | weights_f32
# [3200, 32)) and bulk-copies it into ``push_inbox[destination][this_rank]
# [token]`` of every destination rank that owns one of the token's experts
# (owner = id // 32); a destination that owns none receives the 64-byte
# ids+weights tail only, so it can see that none of the routes is its own. It
# then stores ``push_generation[0] + 1`` into ``push_flags[this_rank][token]``
# of every destination with release semantics. ``local_pack`` CTA (source rank,
# token) waits until its LOCAL flag equals that value (an untimed system-scope
# acquire poll: a late peer stalls, never traps; manifest knob
# ``PUSH_WAIT_TIMEOUT_CYCLES = "0"``), reads the slot through coherent global
# loads, packs it per expert, and the last CTA of the grid advances
# ``push_generation[0]``. The flags and ``push_generation`` are zeroed once at
# allocation (every rank synchronizes its device and joins a process-group
# barrier before any rank's first forward) and never reset: a slot that is not
# pushed in a forward keeps an older value that never equals the expected
# generation, so every rank must execute the same number of forwards (one eager
# warmup on all ranks, then eager forwards or graph replays in lockstep). A
# rank overwrites a peer's slot in forward g + 1 only after its combine(g)
# passed, which waited until every rank's fc2(g) signalled completion, and a
# rank signals only after its local_pack(g) consumed the slot (manifest
# ``route_contract.push_dispatch.reuse_rule``). The peer tables are by-value
# 32-slot aggregates of which entries [0, 16) are used. ``local_pack`` binds
# ``tokens_per_rank`` = this forward's bound, ``ACTIVE_ROWS_PER_EXPERT`` = 32
# and ``SCALE_TILE_K128`` = 1, the constants the exported FC1 program expects.
PUSH_SLOT_BYTES = 3232
PUSH_SLOT_LAYOUT = {  # field -> (byte offset, byte count) inside one slot
    "row_fp8_e4m3": (0, HIDDEN),
    "scales_ue8m0": (HIDDEN, SOURCE_SCALE_BLOCKS),
    "ids_i32": (HIDDEN + SOURCE_SCALE_BLOCKS, 4 * TOP_K),
    "weights_f32": (HIDDEN + SOURCE_SCALE_BLOCKS + 4 * TOP_K, 4 * TOP_K),
}
# Symmetric inbox: uint8 [source rank, source token, slot bytes] per rank.
PUSH_INBOX_SHAPE = (WORLD_SIZE, MAX_TOKENS_PER_RANK, PUSH_SLOT_BYTES)
PUSH_INBOX_BYTES = WORLD_SIZE * MAX_TOKENS_PER_RANK * PUSH_SLOT_BYTES  # 3,309,568
# Symmetric flags: uint32 [source rank, source token] per rank, zeroed once.
PUSH_FLAGS_SHAPE = (WORLD_SIZE, MAX_TOKENS_PER_RANK)
PUSH_FLAGS_BYTES = WORLD_SIZE * MAX_TOKENS_PER_RANK * 4  # 4096
# Rank-local uint32 [2], zeroed once: [0] forwards completed, [1] CTA arrivals.
PUSH_GENERATION_WORDS = 2

# FC2 completion counters (manifest ``route_contract.fc2_completion``). Every
# rank owns ONE symmetric ``fc2_completion`` uint32 [2] (8 bytes), zeroed once
# at allocation together with the inbox flags and never written by the host
# afterwards. Word 0 (``FC2_DONE_WORD``) is the monotonic done counter: the
# last CTA of every rank's ``fc2`` adds +1 into word 0 of ALL 16 ranks (one
# warp-wide ``red.release.sys.global.add.u32`` whose ``.release`` is the
# signal's one system fence; no separate ``fence.release.sys`` precedes it),
# so it grows by exactly 16 per forward and is never reset. Word 1
# (``FC2_ARRIVALS_WORD``) is the rank-local CTA arrival counter of one ``fc2``
# grid (``atom.acq_rel.gpu``); the last arriver returns it to 0, so between
# forwards the counter reads ``[16 * forwards, 0]``. ``combine`` CTA thread 0
# reads ``generation = push_generation[0]`` (``g + 1`` during forward ``g``,
# the word ``local_pack(g)``'s last CTA stored) and polls its LOCAL word 0
# untimed with ``ld.acquire.sys.global.u32`` until ``observed - 16 *
# generation < 2^31`` unsigned (a wrap-safe ``>=``; a late peer stalls this
# rank, it never traps), then ``fence.acquire.sys`` and ``bar.sync``. No add of
# forward ``g + 1`` can land before ``combine(g)`` passed (a rank's
# ``fc2(g + 1)`` follows its ``local_pack(g + 1)``, which waits for every
# source's flags of ``g + 1``, released after that source's ``combine(g)``),
# so the target is reached exactly and the modular compare stays valid across
# the u32 wrap (every 2^28 forwards). The fc2 and combine bindings receive the
# local pointer and its 16-entry peer table (``CakePeerPointerTable<unsigned
# int, 32>``, entries [0, 16) used); slot ``rank`` of the table must be that
# local pointer (symmetric self entry, asserted once at allocation and
# re-checked by every launch).
FC2_COMPLETION_WORDS = 2
FC2_COMPLETION_BYTES = FC2_COMPLETION_WORDS * 4  # 8 bytes per rank
FC2_DONE_WORD = 0
FC2_ARRIVALS_WORD = 1

# Programmatic dependent launch (PDL) chain of the route. ``local_pack``
# executes ``griddepcontrol.launch_dependents`` at entry (before it waits on
# its inbox flags), ``fc1`` and ``fc2`` when a CTA begins its last tile (after
# their own wait); ``fc1`` and ``fc2`` prefill their first expert-weight tile
# through the static tile map and ``griddepcontrol.wait`` before they read the
# routed operands, and ``combine`` executes ``griddepcontrol.wait`` in every
# thread before its first global read and only then polls the completion
# counter. The bindings of exactly the dependents (``fc1``, ``fc2``,
# ``combine``; manifest ``launch.use_pdl``) launch with
# ``cudaLaunchAttributeProgrammaticStreamSerialization``, which makes these
# three edges programmatic in eager streams and in CUDA graphs. ``publish_push``
# and ``local_pack`` launch without attributes (``publish_push`` contains no
# ``griddepcontrol`` instruction). The forward is five launches.
PDL_EDGES = (("local_pack", "fc1"), ("fc1", "fc2"), ("fc2", "combine"))
PDL_STAGES = tuple(dependent for _primary, dependent in PDL_EDGES)
# Function attributes set once per process before a program's first launch
# (manifest ``launch.preferred_shared_memory_carveout``): ``local_pack``
# prefers the full shared-memory carveout; no other program sets one.
PREFERRED_SHARED_MEMORY_CARVEOUT = {"local_pack": 100}


def scale_bytes(rows: int, columns: int) -> int:
    """Byte size of the packed block-32 scale plane of ``[32, rows, columns]``.

    One ``SCALE_TILE_BYTES`` tile per ``BLOCK_M x BLOCK_K`` weight tile and
    expert; this equals one byte per 32-wide block (``rows * columns / 32``
    per expert), so the plane carries no padding.
    """

    if rows % BLOCK_M or columns % BLOCK_K:
        raise ValueError(
            f"scale plane needs rows % {BLOCK_M} == 0 and columns % {BLOCK_K} == 0, "
            f"got ({rows}, {columns})"
        )
    return LOCAL_EXPERTS * (rows // BLOCK_M) * (columns // BLOCK_K) * SCALE_TILE_BYTES


W13_SCALE_BYTES = scale_bytes(FC1_ROWS, HIDDEN)  # 31,457,280
W2_SCALE_BYTES = scale_bytes(HIDDEN, INTERMEDIATE)  # 15,728,640
FC1_INPUT_SCALE_BYTES = scale_bytes(MAX_ROWS_PER_EXPERT, HIDDEN)  # 393,216
FC2_INPUT_SCALE_BYTES = scale_bytes(MAX_ROWS_PER_EXPERT, INTERMEDIATE)  # 655,360


# ---------------------------------------------------------------------------
# Device facts (cached per device index; never queried per call).
# ---------------------------------------------------------------------------
@functools.cache
def sm_count(device_index: int) -> int:
    """Streaming multiprocessor count of ``device_index`` (cached)."""

    return int(torch.cuda.get_device_properties(device_index).multi_processor_count)


def persistent_grid_for_sm_count(count: int) -> int:
    """Grid of the persistent FC1/FC2 programs on a device with ``count`` SMs.

    The largest ``g`` with ``ceil(3 * count / 4) <= g <= count`` that divides
    both full-envelope tile counts (``FC1_SWAP_GRID_M`` = 1280 FC1 tiles and
    ``FC2_SWAP_GRID_M`` = 768 FC2 tiles), so every CTA runs exactly
    ``1280 / g`` FC1 tiles and ``768 / g`` FC2 tiles with no partial last
    wave; the even SM count when no such divisor exists. 148- and 152-SM
    Blackwell parts: 128 (the manifest's ``persistent_grid_by_sm_count``).
    """

    if count <= 0:
        raise RuntimeError(
            f"a device with {count} SMs cannot host the persistent FC1/FC2 grid"
        )
    lowest = (3 * count + 3) // 4  # ceil(3 * count / 4)
    for grid in range(count, lowest - 1, -1):
        if FC1_SWAP_GRID_M % grid == 0 and FC2_SWAP_GRID_M % grid == 0:
            return grid
    return count & ~1  # count >= 2 here: a 1-SM device takes the g = 1 divisor


@functools.cache
def persistent_grid(device_index: int) -> int:
    """Grid of the persistent FC1/FC2 programs of ``device_index`` (cached)."""

    return persistent_grid_for_sm_count(sm_count(device_index))


@functools.cache
def quantize_grid(device_index: int) -> int:
    """Grid of the weight-quantization setup program."""

    return max(1, sm_count(device_index) * QUANTIZE_BLOCKS_PER_SM)


# ---------------------------------------------------------------------------
# Per-forward grids.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StageGrids:
    """``grid.x`` of every forward stage for one rank-uniform token bound."""

    tokens_per_rank: int
    publish_push: int
    local_pack: int
    fc1: int
    fc2: int
    combine: int

    def as_dict(self) -> dict[str, int]:
        return {stage: getattr(self, stage) for stage in STAGES}


def stage_grids(tokens_per_rank: int, device_index: int) -> StageGrids:
    """Grids of the five forward stages for ``tokens_per_rank`` (1..64)."""

    if not isinstance(tokens_per_rank, int) or isinstance(tokens_per_rank, bool):
        raise TypeError("tokens_per_rank must be an int")
    if not 1 <= tokens_per_rank <= MAX_TOKENS_PER_RANK:
        raise ValueError(
            f"tokens_per_rank must be in [1, {MAX_TOKENS_PER_RANK}], got {tokens_per_rank}"
        )
    persistent = persistent_grid(device_index)
    return StageGrids(
        tokens_per_rank=tokens_per_rank,
        publish_push=tokens_per_rank,
        local_pack=WORLD_SIZE * tokens_per_rank,
        fc1=persistent,
        fc2=persistent,
        combine=COMBINE_CTAS_PER_TOKEN * tokens_per_rank,
    )


# ---------------------------------------------------------------------------
# Static (routing-independent) device tensors.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class StaticPlan:
    """Full-envelope tile maps and per-expert constants of one workspace.

    The SwapAB programs visit every local expert with its full tile set on
    every forward (32 experts x 40 FC1 tiles, 32 experts x 24 FC2 tiles);
    experts with zero routes in a forward are visited and produce no rows.
    """

    fc1_tile_expert: torch.Tensor  # int32 [1280]
    fc1_tile_m_local: torch.Tensor  # int32 [1280], multiples of 128 below 5120
    fc2_tile_expert: torch.Tensor  # int32 [768]
    fc2_tile_m_local: torch.Tensor  # int32 [768], multiples of 128 below 3072
    expert_row_offsets: torch.Tensor  # int32 [32] = arange(0, 4096, 128)
    ones: torch.Tensor  # float32 [32]
    zeros: torch.Tensor  # float32 [32]
    infinities: torch.Tensor  # float32 [32]

    @property
    def fc1_grid_m(self) -> int:
        return int(self.fc1_tile_expert.numel())

    @property
    def fc2_grid_m(self) -> int:
        return int(self.fc2_tile_expert.numel())


def static_plan(device: torch.device | str) -> StaticPlan:
    """Allocate the routing-independent tensors on ``device``."""

    experts = torch.arange(LOCAL_EXPERTS, dtype=torch.int32, device=device)
    return StaticPlan(
        fc1_tile_expert=experts.repeat_interleave(FC1_N256_TILES).contiguous(),
        fc1_tile_m_local=torch.arange(
            0, INTERMEDIATE, 128, dtype=torch.int32, device=device
        )
        .repeat(LOCAL_EXPERTS)
        .contiguous(),
        fc2_tile_expert=experts.repeat_interleave(FC2_SWAP_TILES).contiguous(),
        fc2_tile_m_local=torch.arange(
            0, HIDDEN, BLOCK_M, dtype=torch.int32, device=device
        )
        .repeat(LOCAL_EXPERTS)
        .contiguous(),
        expert_row_offsets=torch.arange(
            0, TOTAL_EXPERT_ROWS, MAX_ROWS_PER_EXPERT, dtype=torch.int32, device=device
        ),
        ones=torch.ones((LOCAL_EXPERTS,), dtype=torch.float32, device=device),
        zeros=torch.zeros((LOCAL_EXPERTS,), dtype=torch.float32, device=device),
        infinities=torch.full(
            (LOCAL_EXPERTS,), float("inf"), dtype=torch.float32, device=device
        ),
    )


__all__ = [
    "ACTIVE_ROWS_PER_EXPERT",
    "BLOCK_K",
    "BLOCK_M",
    "COMBINE_CTAS_PER_TOKEN",
    "FC1_INPUT_SCALE_BYTES",
    "FC1_K128_TILES",
    "FC1_N256_TILES",
    "FC1_ROWS",
    "FC1_SWAP_GRID_M",
    "FC2_ARRIVALS_WORD",
    "FC2_COMPLETION_BYTES",
    "FC2_COMPLETION_WORDS",
    "FC2_DONE_WORD",
    "FC2_INPUT_SCALE_BYTES",
    "FC2_K128_TILES",
    "FC2_SWAP_GRID_M",
    "FC2_SWAP_TILES",
    "HIDDEN",
    "INTERMEDIATE",
    "LOCAL_EXPERTS",
    "MAX_RECV_TOKENS",
    "MAX_ROUTES",
    "MAX_ROUTES_PER_EXPERT",
    "MAX_ROWS_PER_EXPERT",
    "MAX_TOKENS_PER_RANK",
    "NUM_EXPERTS",
    "PDL_EDGES",
    "PDL_STAGES",
    "PEER_TABLE_CAPACITY",
    "PG_FLAG_WORDS",
    "PREFERRED_SHARED_MEMORY_CARVEOUT",
    "PUSH_FLAGS_BYTES",
    "PUSH_FLAGS_SHAPE",
    "PUSH_GENERATION_WORDS",
    "PUSH_INBOX_BYTES",
    "PUSH_INBOX_SHAPE",
    "PUSH_SLOT_BYTES",
    "PUSH_SLOT_LAYOUT",
    "ROUTE_ID",
    "SCALE_TILE_K128",
    "SOURCE_SCALE_BLOCKS",
    "STAGES",
    "TOP_K",
    "TOTAL_EXPERT_ROWS",
    "W13_SCALE_BYTES",
    "W2_SCALE_BYTES",
    "WORLD_SIZE",
    "StageGrids",
    "StaticPlan",
    "persistent_grid",
    "persistent_grid_for_sm_count",
    "quantize_grid",
    "scale_bytes",
    "sm_count",
    "stage_grids",
    "static_plan",
]
