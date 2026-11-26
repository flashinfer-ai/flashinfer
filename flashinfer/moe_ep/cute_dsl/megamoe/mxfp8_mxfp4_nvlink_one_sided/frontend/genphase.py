# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Explicit Blackwell GenPhase frontend over its independently owned source fork.

The prototype selects native Blackwell MMA and a 512-column TMEM plan.
It does not import the Rubin frontend or source package.
"""

from __future__ import annotations

import dataclasses
import math
import os
from typing import Literal

import torch

from .block_scaled import (
    BlockScaledMoeConfig,
    BlockScaledSymmBuffer,
    _expected_weight_shapes,
    _validate_shared_weight_legs,
    _validate_weight_leg,
    block_scaled_mega_launch_thunk as sm100_genphase_mega_launch_thunk,
    block_scaled_mega_moe as sm100_genphase_mega_moe,
    chosen_token_tile_n,
    largest_expected_expert_load,
)

from .kernel_helpers import (
    concatenate_block_scaled_weights as concatenate_genphase_weights,
    preprocess_block_scaled_weights as preprocess_genphase_weights,
    preprocess_prequantized_block_scaled_weights as preprocess_genphase_prequantized_weights,
    quantize_mxfp8_block32 as quantize_genphase_mxfp8,
)


@dataclasses.dataclass(frozen=True)
class Sm100GenphaseMoeConfig(BlockScaledMoeConfig):
    """MXFP8 E4M3 or MXFP4-weight x MXFP8, BF16 combine, grouped GenPhase on SM100/103.

    Runs SwiGLU or SiTU (``activation``); the fused shared expert runs SwiGLU.

    Defaults to N128/CGA4; N64 and N128 at CGA2 and N64/CGA4 are explicit
    alternatives, and ``nvlink_one_sided_tiling`` picks among them by expected expert load.
    """

    kernel_variant: Literal["inference", "genphase"] = "genphase"
    cluster_shape_mn: tuple[int, int] = (4, 1)
    fc2_use_bulk: bool = True

    @property
    def instruction_k(self) -> int:
        return 32

    @property
    def resolved_mma_tiler_mnk(self) -> tuple[int, int, int]:
        return self.mma_tiler_mnk or (256, 128, 128)

    def __post_init__(self) -> None:
        super().__post_init__()
        geometry = (self.resolved_mma_tiler_mnk, self.cluster_shape_mn)
        if (
            self.kernel_variant != "genphase"
            or self.quant_kind not in ("mxfp8_e4m3", "mxfp4_mxfp8")
            or geometry
            not in (
                ((256, 128, 128), (4, 1)),
                ((256, 128, 128), (2, 1)),
                ((256, 64, 128), (4, 1)),
                ((256, 64, 128), (2, 1)),
                ((256, 192, 128), (2, 1)),
                ((256, 240, 128), (2, 1)),
            )
            or self.schedule_policy[0] != "grouped"
        ):
            raise ValueError(
                "Blackwell GenPhase requires MXFP8 E4M3 or MXFP4 x MXFP8, tile "
                "(256, 64 or 128, 128) at cluster (2, 1) or (4, 1) and grouped "
                "scheduling."
            )


SENTINEL_MAX_TOKENS = 128
SENTINEL_COMBINE_MAX_TOKENS = 256
SINGLE_GROUP_MAX_TOKENS = 256
PIPELINED_GROUPS_MIN_TOKENS = 1024
# FC1 tiles of a schedule group, and of a pipelined group under the pull dispatch
# (see ``nvlink_one_sided_tiling``).
SCHEDULE_GROUP_FC1_TILES = 256
PULL_PIPELINED_GROUP_FC1_TILES = 128
FENCE_JUST_IN_TIME_MIN_TOKENS = 512
FOUR_ROUNDS_MIN_TOKENS = 1024
EIGHT_ROUNDS_MIN_TOKENS = 2048
# Where the pull dispatch takes over, as (routes per reached target rank, least
# tokens per rank) pairs; see ``dispatch_protocol``.
PULL_THRESHOLDS = ((1.5, 512), (2.0, 2048))
# Where the Sentinel dispatch reaches past ``SENTINEL_MAX_TOKENS``, as (routes per
# reached target rank, most tokens per rank) pairs; see ``dispatch_protocol``.
SENTINEL_DISPATCH_THRESHOLDS = ((1.5, 256),)
# The selection model of a fused shared expert (``shared_head_fills_counter_window``),
# measured on B300 at EP8 (H=6144, top-8 of 256 experts, eight rounds): the rate at
# which the shared FC1 tiles run once the GPU holds its 1100 W power cap, in FLOP
# per us, and the counter dispatch's window before the first routed FC1 tile, a
# fixed part (sizes, plan, the first round's publication and refine) plus the
# first round's rows at the rate they land on a target rank, in bytes per us.
SUSTAINED_GEMM_FLOP_PER_US = 2.0e9
COUNTER_WINDOW_FIXED_US = 32.0
COUNTER_FIRST_ROUND_BYTES_PER_US = 3.24e5
# The window model of a fused shared expert's entry head (``shared_head_token_tiles``
# "window"), measured on B300 at EP8 (DeepSeek-V4.1-Flash, H=5120, one shared expert,
# before the power cap engages): the pull dispatch's window before its first routed
# FC1 tile, a fixed part plus the routes a rank receives at a rate in bytes per us;
# the Sentinel dispatch's window; when the head's first tiles start, from kernel entry
# or once the launch has chosen its token tile configuration; and the bytes per us
# one SM loads into its operand stages while every SM runs head tiles, which bound
# the head (the E4M3 weights and the tokens come from L2).
PULL_WINDOW_FIXED_US = 21.0
PULL_WINDOW_ROUTE_BYTES_PER_US = 2.5e7
SENTINEL_WINDOW_US = 10.0
SHARED_HEAD_ENTRY_START_US = 2.0
SHARED_HEAD_CHOICE_START_US = 7.0
SHARED_HEAD_SM_BYTES_PER_US = 9.0e4


def token_tile_n(expected_load: float) -> int:
    """Token tile N for an expected per-expert load (tokens).

    While few experts need a second N64 token block (an expected load up to 56)
    the N64 tile wastes the fewest MMA columns. Above that the tile is sized for
    the largest expert, estimated as ``load + 1.6 * sqrt(load)``: N128 while it
    holds that expert, as the tile with the deepest operand pipeline. Past it the
    expert takes the fewest tiles of at most 240 rows (two disjoint TMEM
    accumulator stages), since every tile streams the expert's weights again;
    N192 and N240 split an expert's rows evenly over its tiles, and the tile is
    the narrower of the two that holds that share. Each tile's MMAs run at its
    own instruction N, so a tile's width costs pipeline depth, not MMA columns.
    """
    if expected_load <= 56:
        return 64
    largest = largest_expected_expert_load(expected_load)
    if largest <= 140:
        return 128
    share = largest / math.ceil(largest / 240)
    return 192 if share <= 192 else 240


def counter_dispatch_rounds(
    max_tokens_per_rank: int, world_size: int, num_experts: int
) -> int:
    """Dispatch rounds of the counter protocols for a capacity.

    Rounds let each expert's FC1 wait for its own round of the payload instead of
    all of it. Publishing a round costs a system fence on the pushing SM and a
    counter update that queues behind bulk data on NVLink, so rounds only pay
    where the whole-payload wait is long: four rounds from
    ``FOUR_ROUNDS_MIN_TOKENS`` tokens per rank, eight from
    ``EIGHT_ROUNDS_MIN_TOKENS``. The kernel takes rounds for at most 32 local
    experts, splits them evenly over the rounds, and keeps one bit per (round,
    peer) in a 64-bit word.
    """
    experts_per_rank = num_experts // world_size
    if max_tokens_per_rank < FOUR_ROUNDS_MIN_TOKENS or experts_per_rank > 32:
        return 1
    rounds = 8 if max_tokens_per_rank >= EIGHT_ROUNDS_MIN_TOKENS else 4
    while rounds > 1 and (experts_per_rank % rounds or world_size * rounds > 64):
        rounds //= 2
    return rounds


def routes_per_target_token(world_size: int, num_experts: int, num_topk: int) -> float:
    """Expected routes of one token to one target rank it reaches, under uniform routing.

    ``num_topk`` distinct experts drawn uniformly reach a given rank unless all of
    them miss its ``num_experts / world_size`` experts; the ratio of the routes to
    the expected ranks reached is how often the pull dispatch, which fetches a
    token once per local route, moves a row that one push per target rank would
    move once.
    """
    experts_per_rank = num_experts // world_size
    miss = 1.0
    for drawn in range(num_topk):
        miss *= max(num_experts - experts_per_rank - drawn, 0) / (num_experts - drawn)
    return num_topk / (world_size * (1.0 - miss))


def counter_first_round_rows(
    max_tokens_per_rank: int, world_size: int, num_experts: int, num_topk: int
) -> float:
    """Expected token rows one target rank receives in the counter dispatch's first round.

    A token travels to a target rank once, in the round of its first expert
    there (``counter_dispatch_rounds``), so under uniform routing it is in the
    first round when any of its ``num_topk`` distinct experts is one of the
    target's first-round experts; every source rank, the target included, sends
    its share of the capacity.
    """
    rounds = counter_dispatch_rounds(max_tokens_per_rank, world_size, num_experts)
    first_round_experts = num_experts // world_size // rounds
    miss = 1.0
    for drawn in range(num_topk):
        miss *= max(num_experts - first_round_experts - drawn, 0) / (
            num_experts - drawn
        )
    return max_tokens_per_rank * world_size * (1.0 - miss)


def counter_dispatch_window_us(
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
) -> float:
    """Time from kernel entry to the counter dispatch's first routed FC1 tile, in us.

    The first tile in processing order belongs to a first-round expert and waits
    for that round from every source rank: ``COUNTER_WINDOW_FIXED_US`` plus the
    round's rows (``counter_first_round_rows``, E4M3 data and one UE8M0 scale per
    32 values each) at ``COUNTER_FIRST_ROUND_BYTES_PER_US``.
    """
    rows = counter_first_round_rows(
        max_tokens_per_rank, world_size, num_experts, num_topk
    )
    row_bytes = hidden + hidden // 32
    return COUNTER_WINDOW_FIXED_US + rows * row_bytes / COUNTER_FIRST_ROUND_BYTES_PER_US


def shared_fc1_head_us(
    max_tokens_per_rank: int, hidden: int, shared_intermediate: int
) -> float:
    """Time of a fused shared expert's FC1 tiles at ``SUSTAINED_GEMM_FLOP_PER_US``, in us.

    The shared expert computes every token of the capacity: ``2 * hidden * 2 *
    shared_intermediate`` FLOP per token for its gate and up projections.
    """
    flop = 2.0 * max_tokens_per_rank * hidden * 2 * shared_intermediate
    return flop / SUSTAINED_GEMM_FLOP_PER_US


def shared_head_fills_counter_window(
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
    shared_intermediate: int,
) -> bool:
    """Whether a fused shared expert's FC1 tiles take the counter dispatch's whole window.

    See ``dispatch_protocol``: the shared FC1 time against the window before the
    counter dispatch's first routed FC1 tile.
    """
    head = shared_fc1_head_us(max_tokens_per_rank, hidden, shared_intermediate)
    window = counter_dispatch_window_us(
        max_tokens_per_rank, world_size, num_experts, num_topk, hidden
    )
    return head >= window


def pull_dispatch_window_us(
    max_tokens_per_rank: int, num_topk: int, hidden: int
) -> float:
    """Time from kernel entry to the pull dispatch's first routed FC1 tile, in us.

    The plan and the walk's first fetches: ``PULL_WINDOW_FIXED_US`` plus the
    rank's routes (``num_topk`` per token of the capacity, E4M3 data and one
    UE8M0 scale per 32 values each) at ``PULL_WINDOW_ROUTE_BYTES_PER_US``.
    """
    row_bytes = hidden + hidden // 32
    return (
        PULL_WINDOW_FIXED_US
        + max_tokens_per_rank * num_topk * row_bytes / PULL_WINDOW_ROUTE_BYTES_PER_US
    )


def dispatch_window_us(
    dispatch: str,
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    hidden: int,
) -> float:
    """Time from kernel entry to the first routed FC1 tile under a dispatch protocol, in us.

    ``pull_dispatch_window_us``, ``counter_dispatch_window_us``, or
    ``SENTINEL_WINDOW_US`` for the Sentinel dispatch, whose experts become ready
    as their rows are verified.
    """
    if dispatch == "pull":
        return pull_dispatch_window_us(max_tokens_per_rank, num_topk, hidden)
    if dispatch == "counter":
        return counter_dispatch_window_us(
            max_tokens_per_rank, world_size, num_experts, num_topk, hidden
        )
    return SENTINEL_WINDOW_US


def shared_head_token_tiles(
    window_us: float,
    *,
    hidden: int,
    shared_intermediate: int,
    tile_rows: int,
    token_tiles: int,
    clusters: int,
    after_choice: bool,
) -> int:
    """Token tiles of a fused shared expert whose FC1 tiles the entry head finishes before ``window_us``.

    The head starts at ``SHARED_HEAD_ENTRY_START_US`` (``SHARED_HEAD_CHOICE_START_US``
    where it waits for the token tile choice) on the ``clusters`` the main
    kernel holds then. A head tile's CTA loads its 128 weight rows and half the
    tile's ``tile_rows`` tokens, ``hidden`` E4M3 values and one UE8M0 scale per
    32 each, at ``SHARED_HEAD_SM_BYTES_PER_US``; the head holds the tiles that
    take until the window, as whole token tiles of ``2 * shared_intermediate /
    256`` FC1 tiles, at least one.
    """
    start_us = (
        SHARED_HEAD_CHOICE_START_US if after_choice else SHARED_HEAD_ENTRY_START_US
    )
    cta_bytes = (128 + tile_rows / 2) * (hidden + hidden // 32)
    tile_us = cta_bytes / SHARED_HEAD_SM_BYTES_PER_US
    tiles = max(0.0, window_us - start_us) * clusters / tile_us
    blocks = 2 * shared_intermediate // 256
    return min(token_tiles, max(1, int(tiles) // blocks))


def dispatch_protocol(
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    *,
    hidden: int | None = None,
    shared_intermediate: int | None = None,
) -> str:
    """Dispatch protocol for a capacity: ``sentinel``, ``counter`` or ``pull``.

    The Sentinel dispatch up to ``SENTINEL_MAX_TOKENS`` tokens per rank (see
    ``nvlink_one_sided_tiling``), and up to the bounds of ``SENTINEL_DISPATCH_THRESHOLDS``
    where a reached rank takes few routes per token. There the pull dispatch
    takes over at 512 tokens per rank, and below that the counter dispatch would
    run in one round, holding every FC1 tile until the whole payload has landed.
    The Sentinel dispatch releases FC1 per expert instead; with the larger entry
    prefetch it then takes (``resolved_entry_prefetch_bytes``) the earlier start
    outweighs its per-byte verification and the reduce's landing-row reset:
    V4.1-Flash on eight GPUs (1.35 routes per reached rank) gains about 2.7 us
    at 256 tokens per rank. Above these bounds the pull dispatch where it
    repays its extra fetches, the counter dispatch elsewhere. The pull dispatch makes experts
    ready one by one in processing order from the end of the plan on, so the
    first FC1 starts long before the whole payload, or the counter dispatch's
    first round, has landed. It fetches a token once per local route, though,
    where one push per target rank moves it once, and its walk holds SMs the
    engine waits for. ``routes_per_target_token`` is the ratio of those fetches
    under uniform routing (top-6: 1.35 at EP8, 1.8 at EP4, 3.0 at EP2, 6 at
    EP1); the more routes a reached rank takes per token, the larger the
    capacity at which the earlier start outweighs the extra bytes.
    ``PULL_THRESHOLDS`` gives, for each bound on that ratio, the least tokens
    per rank that take the pull dispatch; above the last bound the counter
    dispatch stays.

    A fused shared expert of width ``shared_intermediate`` (with the layer's
    ``hidden``) runs its FC1 tiles from the main kernel's entry, before any
    routed tile is ready, so the window before the first routed FC1 tile is
    tensor time it can fill. The pull dispatch's window is short (the plan and
    the first fetches), and past it the grouped GEMMs run at their sustained
    rate, so the shared FC1 tiles that do not fit add their whole time. The
    counter dispatch's window lasts until the first round of the payload has
    landed. Where the pull dispatch would be chosen, the counter dispatch takes
    over once the shared FC1 tiles take that whole window
    (``shared_head_fills_counter_window``): then its later start leaves no tensor
    core idle, and it moves each token once per target rank, holds no walking
    CTAs and allocates no token pool. Where they do not, the counter dispatch
    would have to repay its idle tensor time with what it saves past its start,
    which the rule does not count on, so the pull dispatch stays.
    """
    if (hidden is None) != (shared_intermediate is None):
        raise ValueError("hidden and shared_intermediate go together")
    if max_tokens_per_rank <= SENTINEL_MAX_TOKENS:
        return "sentinel"
    routes = routes_per_target_token(world_size, num_experts, num_topk)
    for max_routes, max_tokens in SENTINEL_DISPATCH_THRESHOLDS:
        if routes <= max_routes and max_tokens_per_rank <= max_tokens:
            return "sentinel"
    for max_routes, min_tokens in PULL_THRESHOLDS:
        if routes <= max_routes:
            if max_tokens_per_rank < min_tokens:
                return "counter"
            if shared_intermediate is not None and shared_head_fills_counter_window(
                max_tokens_per_rank,
                world_size,
                num_experts,
                num_topk,
                hidden,
                shared_intermediate,
            ):
                return "counter"
            return "pull"
    return "counter"


def nvlink_one_sided_tiling(
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    *,
    hidden: int | None = None,
    shared_intermediate: int | None = None,
) -> dict:
    """Engine geometry and schedule for a capacity, as config keyword arguments.

    The expected load of one local expert is ``max_tokens_per_rank * world_size *
    num_topk / num_experts`` tokens and picks the token tile (``token_tile_n``);
    from N128 up one FC2 staging stage leaves SMEM for another AB stage. CGA2 uses
    every SM, and atomic work claims keep the CTAs evenly loaded.

    Up to ``SINGLE_GROUP_MAX_TOKENS`` tokens per rank one schedule group larger than any
    FC1 tile count runs all FC1 tiles before any FC2 tile, so FC2 never waits on
    FC1. Above it groups of about 256 FC1 tiles spread the FC2 tiles, and with them
    the combine traffic, over the kernel instead of bunching every push into its
    last quarter. From ``PIPELINED_GROUPS_MIN_TOKENS`` tokens per rank, where the
    combine needs most of the NVLink bandwidth, the groups are pipelined: a
    group's FC2 tiles are claimed among the next group's FC1 tiles, so the pushes
    leave at the rate of the whole engine rather than during a group's FC2 third,
    and an FC2 tile's FC1 tiles have had a whole group's time to finish. Below it
    the combine's bursts cost little, and the grouped order keeps each group's FC2
    tiles next to their FC1 tiles. Under the pull dispatch the pipelined groups hold
    about 128 FC1 tiles: a target rank's incoming link carries its fetches and its
    peers' pushes together, close to the engine's own time, and the pushes of the
    last group's FC2 tiles, which no FC1 tile is left to space out, leave faster
    than the link drains them. Halving the group halves that last burst and starts
    the pushes a group earlier; smaller groups make FC2 tiles wait for FC1 tiles
    of the group before. There the FC2 epilogue also pushes with warp-wide stores
    (``combine_warp_stores``): bulk copies into peers' slots drain at the link's
    pace through the TMA unit that also issues the mainloop's loads. From
    ``FENCE_JUST_IN_TIME_MIN_TOKENS`` the
    Fence dispatches (counter and pull) claim work just in time: a cluster takes
    its next tile only once the current one has passed its readiness gate, so FC2
    tiles are not claimed long before their FC1 tiles finish and the last tiles go
    to the clusters that free up first.

    The Sentinel protocols save the fence-and-flag round trips but cost work that
    grows with capacity, and the two directions pay it differently. The Sentinel
    dispatch verifies every landing byte while its CTAs hold SMs the engine waits
    for, and the reduce resets every landing row, so past ``SENTINEL_MAX_TOKENS``
    tokens per rank the counter dispatch's per-peer payload counters are cheaper,
    except where its whole-payload wait costs more (``dispatch_protocol``).
    The Sentinel combine only resets the combine slots the reduce consumed, while
    the counter combine holds every rank's main kernel in an all-rank tail barrier,
    so the Sentinel combine stays up to ``SENTINEL_COMBINE_MAX_TOKENS``. Large
    capacities take the pull dispatch where it repays its extra fetches
    (``dispatch_protocol``); it runs under the counter combine in one round.
    Elsewhere the counter dispatch travels in rounds from
    ``FOUR_ROUNDS_MIN_TOKENS`` (``counter_dispatch_rounds``), which need the
    counter combine as well. With a fused shared expert (``hidden`` and its
    ``shared_intermediate``) the counter dispatch also replaces the pull where
    the shared FC1 tiles fill its window (``dispatch_protocol``).
    """
    expected_load = max_tokens_per_rank * world_size * num_topk / num_experts
    tile_n = token_tile_n(expected_load)
    dispatch = dispatch_protocol(
        max_tokens_per_rank,
        world_size,
        num_experts,
        num_topk,
        hidden=hidden,
        shared_intermediate=shared_intermediate,
    )
    combine = (
        "sentinel" if max_tokens_per_rank <= SENTINEL_COMBINE_MAX_TOKENS else "counter"
    )
    single_group = max_tokens_per_rank <= SINGLE_GROUP_MAX_TOKENS
    pipelined = max_tokens_per_rank >= PIPELINED_GROUPS_MIN_TOKENS
    pull_pipelined = pipelined and dispatch == "pull"
    group_fc1_tiles = (
        PULL_PIPELINED_GROUP_FC1_TILES if pull_pipelined else SCHEDULE_GROUP_FC1_TILES
    )
    return dict(
        mma_tiler_mnk=(256, tile_n, 128),
        cluster_shape_mn=(2, 1),
        fc2_tma_stages=None if tile_n == 64 else 1,
        schedule_policy=("grouped", 4096 if single_group else group_fc1_tiles),
        pipelined_groups=pipelined,
        just_in_time_claims=(
            True
            if dispatch != "sentinel"
            and max_tokens_per_rank >= FENCE_JUST_IN_TIME_MIN_TOKENS
            else None
        ),
        work_id_mode="atomic_counter",
        epi_flag_batches=(1, 2),
        dispatch_protocol=dispatch,
        combine_protocol=combine,
        dispatch_rounds=(
            counter_dispatch_rounds(max_tokens_per_rank, world_size, num_experts)
            if dispatch == combine == "counter"
            else 1
        ),
    )


# Config fields that select engine geometry, scheduling and protocols without
# changing the problem, the I/O tensors or the weight layout: the keys of
# ``nvlink_one_sided_tiling`` and the remaining performance fields.
TUNING_FIELDS = frozenset(
    {
        "mma_tiler_mnk",
        "cluster_shape_mn",
        "fc2_tma_stages",
        "schedule_policy",
        "pipelined_groups",
        "work_id_mode",
        "epi_flag_batches",
        "dispatch_protocol",
        "combine_protocol",
        "weight_evict",
        "entry_prefetch_bytes",
        "token_ab_stages",
        "alternate_token_tile_n",
        "alternate_token_ab_stages",
        "alternate_weight_evict",
        "token_tile_choice",
        "comm_pusher_ctas",
        "dispatch_rounds",
        "comm_push_depth",
        "comm_pull_depth",
        "comm_pull_chunk_rows",
        "comm_pull_lead_rows",
        "comm_exchange_ctas",
        "pull_warps",
        "gather_b_warps",
        "pull_acquire_in_copy",
        "pull_walk_dedup",
        "pull_tile_ready",
        "fc1_token_pool",
        "fc1_token_pool_fallback",
        "refine_exchange_ctas",
        "round_publish_lag",
        "gather_padding_spread",
        "max_sm_count",
        "just_in_time_claims",
        "claim_lead_k_tiles",
        "discard_combine_slots",
        "combine_warp_stores",
        "shared_expert_entry",
        "shared_fc2_placement",
        "shared_alternate_token_tile",
        "shared_head_token_tiles",
    }
)


class Sm100GenphaseSymmBuffer(BlockScaledSymmBuffer):
    """Blackwell workspace; staging and launch follow the shared block-scaled API."""

    config: Sm100GenphaseMoeConfig
    shared_workspace: torch.Tensor | None
    local_workspace: torch.Tensor | None

    # SM100 and SM103 have the same block-scaled atom and TMEM capacity.
    # The compiler target below remains native to the actual device.
    _architecture = "sm_100"

    @staticmethod
    def _require_device() -> None:
        cc = torch.cuda.get_device_capability()
        if cc not in ((10, 0), (10, 3)):
            raise RuntimeError(f"Blackwell GenPhase requires SM100/103; got {cc}.")

    @staticmethod
    def _configure_dsl() -> None:
        cc = torch.cuda.get_device_capability()
        target = f"sm_{cc[0]}{cc[1]}a"
        configured = os.environ.get("CUTE_DSL_ARCH")
        if configured not in (None, target, target.removesuffix("a")):
            raise RuntimeError(
                f"Blackwell GenPhase requires CUTE_DSL_ARCH={target}; got {configured!r}."
            )
        os.environ["CUTE_DSL_ARCH"] = target
        from flashinfer.cute_dsl.availability import _dsl_captured_arch

        if _dsl_captured_arch() not in (target, target.removesuffix("a")):
            raise RuntimeError(
                f"Restart Python with CUTE_DSL_ARCH={target} before importing CuTe DSL."
            )

    def __init__(self, config: Sm100GenphaseMoeConfig, process_group=None) -> None:
        if not isinstance(config, Sm100GenphaseMoeConfig):
            raise TypeError("Blackwell GenPhase requires Sm100GenphaseMoeConfig.")
        super().__init__(config, process_group)

    def retune(self, knobs) -> None:
        """Collectively replace kernel storage before capture, preserving I/O views.

        Only ``TUNING_FIELDS`` may change, to values the config validates.
        Call it on an unpooled workspace, on every rank with the same knobs.
        """
        from . import comm

        comm.ensure_not_capturing("Blackwell GenPhase retune")
        if self._destroyed:
            raise RuntimeError("cannot retune a destroyed workspace")
        if not isinstance(knobs, dict) or not set(knobs) <= TUNING_FIELDS:
            raise ValueError("invalid Blackwell GenPhase tuning knobs")
        config = dataclasses.replace(self.config, **knobs)
        if config == self.config:
            return
        kernel = self._build_kernel(config)
        torch.cuda.synchronize()
        self._compiled = self._launch_key = self._launch_kwargs = None
        self._launch_weights = None
        self._pre_reduced_activation = self._pre_reduced_activation_sf = None
        comm.free_sym_tensor(self.shared_workspace)
        self.shared_workspace = None
        self.local_workspace = None
        self.config = config
        self.kernel = kernel
        self._allocate_kernel_workspace(kernel)


def validate_genphase_weights(transformed, config, device: torch.device) -> None:
    """Validate external GenPhase weights without allocating a workspace.

    Legacy SM100 transformed weights have a different physical layout and
    cannot be reused as a GenPhase payload. Under die-local weights a pair
    prepared without them is accepted in its own layout: the launch paths
    convert it once, on the first launch. With a shared expert the tuple
    carries its FC1 and FC2 legs after the routed ones.
    """
    legs = 4 if config.shared_expert else 2
    if not isinstance(transformed, tuple) or len(transformed) != legs:
        raise ValueError(
            "GenPhase transformed weights must be a (fc1, fc2) tuple"
            + (
                ", then the shared expert's (fc1, fc2)."
                if config.shared_expert
                else "."
            )
        )
    _validate_shared_weight_legs(transformed[2:] or None, config, device)
    transformed = transformed[:2]
    if config.resolved_die_local_weights and not any(
        isinstance(leg, tuple) and getattr(leg[0], "_mega_die_local", False)
        for leg in transformed
    ):
        config = dataclasses.replace(config, die_local_weights=False)
    fc1_shape, fc2_shape, fc1_sf, fc2_sf = _expected_weight_shapes(config)
    for name, leg, shape, sf in (
        ("fc1", transformed[0], fc1_shape, fc1_sf),
        ("fc2", transformed[1], fc2_shape, fc2_sf),
    ):
        _validate_weight_leg(name, leg, shape, sf, config, device)


__all__ = [
    "TUNING_FIELDS",
    "chosen_token_tile_n",
    "concatenate_genphase_weights",
    "Sm100GenphaseMoeConfig",
    "Sm100GenphaseSymmBuffer",
    "preprocess_genphase_weights",
    "preprocess_genphase_prequantized_weights",
    "validate_genphase_weights",
    "quantize_genphase_mxfp8",
    "sm100_genphase_mega_launch_thunk",
    "sm100_genphase_mega_moe",
]
