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


# The dispatch rule's bounds, in tokens per rank (``dispatch_protocol``): the
# Sentinel dispatch up to SENTINEL_MAX_TOKENS, and up to the bounds of
# SENTINEL_DISPATCH_THRESHOLDS, as (routes per reached target rank, most tokens per
# rank) pairs, where a reached rank takes few routes per token.
SENTINEL_MAX_TOKENS = 192
SENTINEL_DISPATCH_THRESHOLDS = ((1.5, 256),)
# The synchronization protocols a layer can run (``nvlink_one_sided_tiling``'s
# ``sync_protocol``) and the dispatch each one selects: "sentinel" the Sentinel
# dispatch with the Sentinel combine, "fence" the pull dispatch with the Fence
# (counter) combine; "auto" takes the dispatch rule.
SYNC_PROTOCOLS = ("auto", "fence", "sentinel")
SYNC_PROTOCOL_DISPATCH = {"sentinel": "sentinel", "fence": "pull"}
SINGLE_GROUP_MAX_TOKENS = 256
PIPELINED_GROUPS_MIN_TOKENS = 1024
# FC1 tiles of a schedule group, and of a pipelined group under the pull dispatch
# (see ``nvlink_one_sided_tiling``).
SCHEDULE_GROUP_FC1_TILES = 256
PULL_PIPELINED_GROUP_FC1_TILES = 128
PULL_LAZY_CLAIMS_MIN_TOKENS = 512
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

    ``pull_dispatch_window_us``, or ``SENTINEL_WINDOW_US`` for the Sentinel
    dispatch, whose experts become ready as their rows are verified.
    """
    if dispatch == "pull":
        return pull_dispatch_window_us(max_tokens_per_rank, num_topk, hidden)
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
    max_tokens_per_rank: int, world_size: int, num_experts: int, num_topk: int
) -> str:
    """Dispatch protocol the rule picks for a capacity: ``sentinel`` or ``pull``; the combine follows it.

    The rule of ``sync_protocol="auto"`` (``nvlink_one_sided_tiling``).

    The Sentinel dispatch up to ``SENTINEL_MAX_TOKENS`` tokens per rank, and up to
    the bounds of ``SENTINEL_DISPATCH_THRESHOLDS`` where a reached target rank
    takes few routes per token; the pull dispatch everywhere else. The Sentinel
    dispatch runs under the Sentinel combine and the pull dispatch under the
    counter (Fence) combine (``nvlink_one_sided_tiling``).

    The Sentinel dispatch pushes the payload at kernel entry and releases FC1 per
    expert as its rows verify, and its combine resets only the combine slots the
    reduce consumed. Its costs grow with the routes a rank receives: it verifies
    every landing byte while its CTAs hold SMs the engine waits for, and the
    reduce resets every landing row and combine slot. The pull dispatch fetches
    only once the plan exists, and fetches a token once per local route where a
    push moves it once per target rank (``routes_per_target_token``: top-6 of 384
    experts takes 1.35 routes per reached rank at EP8, 1.8 at EP4, 3.0 at EP2, 6
    at EP1); its combine's all-rank tail barrier costs 2 to 5 us up to 256 tokens
    per rank.

    Measured on B300 (the slowest rank's span), the Sentinel dispatch beats the
    pull at every point up to 192 tokens per rank: by 5 to 18 us on one to four
    GPUs, and by 2 to 5 us on Kimi-K3 (top-16 of 896 experts, 2.26 routes per
    reached rank) and HY4 (top-8 of 256, 1.51) on eight. Above 192 the pull is the
    faster at EP8 (Kimi-K3 by 11 us at 255; HY4 by up to 7 us at 256), except
    where a reached rank takes at most 1.5 routes per token: DeepSeek-V4.1-Flash
    on eight GPUs (1.35) keeps the Sentinel dispatch 5 to 7 us ahead at 256. Past
    256 the Sentinel's lead shrinks (3 to 6 us at 288 to 320, a tie near 384), and
    on one to four GPUs it stays 1 to 12 us ahead up to 320, both of which this
    rule leaves to the pull.
    """
    if max_tokens_per_rank <= SENTINEL_MAX_TOKENS:
        return "sentinel"
    routes = routes_per_target_token(world_size, num_experts, num_topk)
    for max_routes, max_tokens in SENTINEL_DISPATCH_THRESHOLDS:
        if routes <= max_routes and max_tokens_per_rank <= max_tokens:
            return "sentinel"
    return "pull"


def nvlink_one_sided_tiling(
    max_tokens_per_rank: int,
    world_size: int,
    num_experts: int,
    num_topk: int,
    sync_protocol: str = "auto",
) -> dict:
    """Engine geometry, schedule and protocols for a capacity, as config keyword arguments.

    ``sync_protocol`` picks the dispatch: ``"auto"`` by ``dispatch_protocol``'s
    rule, ``"sentinel"`` the Sentinel dispatch and ``"fence"`` the pull dispatch
    at any capacity (``SYNC_PROTOCOL_DISPATCH``). The combine and every schedule
    field below that depends on the dispatch follow the dispatch picked, so a
    protocol forced to the one the rule picks gives the same fields.

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
    ``PULL_LAZY_CLAIMS_MIN_TOKENS`` the pull dispatch claims work lazily: a
    cluster takes its next tile only once the current one has passed its readiness
    gate, so FC2 tiles are not claimed long before their FC1 tiles finish and the
    last tiles go to the clusters that free up first.

    The combine follows the dispatch. The Sentinel dispatch runs with the
    Sentinel combine, which only resets the combine slots the reduce consumed.
    The pull dispatch runs with the counter (Fence) combine, whose all-rank tail
    barrier is what orders every peer's fetches of a rank's staged activations
    before that rank stages again.
    """
    if sync_protocol not in SYNC_PROTOCOLS:
        raise ValueError(
            f"sync_protocol must be one of {SYNC_PROTOCOLS}, got {sync_protocol!r}"
        )
    expected_load = max_tokens_per_rank * world_size * num_topk / num_experts
    tile_n = token_tile_n(expected_load)
    dispatch = SYNC_PROTOCOL_DISPATCH.get(sync_protocol) or dispatch_protocol(
        max_tokens_per_rank, world_size, num_experts, num_topk
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
        lazy_claims=(
            True
            if dispatch == "pull" and max_tokens_per_rank >= PULL_LAZY_CLAIMS_MIN_TOKENS
            else None
        ),
        work_id_mode="atomic_counter",
        epi_flag_batches=(1, 2),
        dispatch_protocol=dispatch,
        combine_protocol="sentinel" if dispatch == "sentinel" else "counter",
    )


# Config fields that select engine geometry and scheduling without changing the
# problem, the I/O tensors, the weight layout or the protocols: the keys of
# ``nvlink_one_sided_tiling`` but the protocols, which its ``sync_protocol``
# picks, and the remaining performance fields.
TUNING_FIELDS = frozenset(
    {
        "mma_tiler_mnk",
        "cluster_shape_mn",
        "fc2_tma_stages",
        "schedule_policy",
        "pipelined_groups",
        "work_id_mode",
        "epi_flag_batches",
        "weight_evict",
        "entry_prefetch_bytes",
        "token_ab_stages",
        "alternate_token_tile_n",
        "alternate_token_ab_stages",
        "alternate_weight_evict",
        "token_tile_choice",
        "comm_pusher_ctas",
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
        "gather_padding_spread",
        "max_sm_count",
        "lazy_claims",
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
