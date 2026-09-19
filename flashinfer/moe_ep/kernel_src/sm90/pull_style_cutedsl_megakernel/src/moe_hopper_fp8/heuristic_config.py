# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Token-bucket launch heuristics for Hopper FP8 MegaMoE.

The table is derived from four-rank H200 DeepSeek-V4 P03 sweeps (2026-08-19
geometry, 2026-09-18 tail-split / group_hint / token-back retune).  Each entry
maximizes the slowest-rank effective TFLOPS over operand order, tile shape,
CGA shape, legacy/ping-pong scheduling, scheduler group size, tail-split pair
tasks and fc2 write-back placement.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

TOKEN_BUCKETS = tuple(1 << power for power in range(3, 16))
DEFAULT_MMA_TILER_MNK = (64, 128, 128)
DEFAULT_CLUSTER_SHAPE_MNK = (1, 1, 1)
DEFAULT_ACCUM_MODE = "1xacc"


DEFAULT_TOKEN_BACK_MODE = "epi_warps"
# group_hint that puts every expert into one scheduler group.
ALL_EXPERTS_GROUP = 1 << 20
# FP8_TAIL_SPLIT=1 turns tail-split pair tasks on for every selected config
# with a 2-CTA token cluster; other geometries keep the table value.
TAIL_SPLIT_ENV = "FP8_TAIL_SPLIT"


@dataclass(frozen=True)
class HopperFp8Config:
    swap_ab: bool
    pingpong: bool
    mma_tiler_mnk: Tuple[int, int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    accum_mode: str = DEFAULT_ACCUM_MODE
    # Cross-rank fc2 write-back placement; per-bucket winner of the four-rank
    # H200 epi-vs-reuse A/B (reuse_dispatch_warps only on blockwise 16384+).
    token_back_mode: str = DEFAULT_TOKEN_BACK_MODE
    # Scheduler group size in FC1 cluster tiles (None = one wave of clusters).
    # Several experts per group keep FC2 tiles from spinning on fc1_done right
    # behind their FC1 tiles; only matters for small experts.
    group_hint: Optional[int] = None
    # Tail-split pair tasks (see fc1_fc2_fuse_sched); legal only with a 2-CTA
    # token cluster: swap-AB cga (1,2,1) or non-swap cga (2,1,1).
    # FP8_TAIL_SPLIT=1 turns it on for every such bucket.
    tail_split_pairs: bool = False

    @property
    def token_cluster(self) -> int:
        """CTAs along the token axis (N after swapping A/B, M otherwise)."""
        return (
            self.cluster_shape_mnk[1]
            if self.swap_ab
            else self.cluster_shape_mnk[0]
        )

    @property
    def weight_cluster(self) -> int:
        """CTAs along the weight axis (M after swapping A/B, N otherwise)."""
        return (
            self.cluster_shape_mnk[0]
            if self.swap_ab
            else self.cluster_shape_mnk[1]
        )

    @property
    def tail_split_geometry(self) -> bool:
        """True when the kernel accepts ``tail_split_pairs``.

        The scheduler pairs the two CTAs of a token cluster on adjacent
        weight tiles, so exactly two CTAs along the tokens and one along the
        weights: swap-AB cga (1,2,1) or non-swap cga (2,1,1).
        """
        return self.token_cluster == 2 and self.weight_cluster == 1


@dataclass(frozen=True)
class HopperFp8ConfigSelection:
    config: HopperFp8Config
    source: str
    token_bucket: Optional[int]


def _config(
    *,
    swap_ab: bool,
    pingpong: bool,
    tile: Tuple[int, int, int],
    cga: Tuple[int, int, int],
    token_back: str = DEFAULT_TOKEN_BACK_MODE,
    group_hint: Optional[int] = None,
    tail_split_pairs: bool = False,
) -> HopperFp8Config:
    return HopperFp8Config(
        swap_ab=swap_ab,
        pingpong=pingpong,
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cga,
        token_back_mode=token_back,
        group_hint=group_hint,
        tail_split_pairs=tail_split_pairs,
    )


HEURISTIC_CONFIGS = {
    "per_tensor": {
        8: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1)),
        16: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(1, 2, 1)),
        32: _config(swap_ab=True, pingpong=False, tile=(256, 8, 128), cga=(2, 1, 1)),
        64: _config(swap_ab=True, pingpong=False, tile=(128, 8, 128), cga=(1, 2, 1)),
        128: _config(swap_ab=True, pingpong=True, tile=(128, 8, 128), cga=(1, 2, 1)),
        256: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1)),
        # 512 / 1024 (2026-09-18, 4x H200, two interleaved rounds): group_hint
        # 264 -3% / -10%; 1024 with tail-split pair tasks on top -15%.
        512: _config(swap_ab=True, pingpong=False, tile=(256, 64, 128), cga=(1, 1, 1),
                     group_hint=264),
        1024: _config(swap_ab=True, pingpong=True, tile=(128, 64, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True),
        # 2048-32768 (2026-09-18, 4x H200, two interleaved rounds): swap-AB
        # ping-pong M128N128 cga(1,2,1) with tail-split pair tasks beats the
        # previous entries by 25/25/20% (2048/4096/8192, with group_hint 264)
        # and 9/13% (16384/32768); epi_warps ties reuse_dispatch_warps there.
        2048: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True),
        4096: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True),
        8192: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True),
        16384: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                       tail_split_pairs=True),
        32768: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                       tail_split_pairs=True),
    },
    # per_tensor 8 -> cooperative swap M256N16 and 64 -> basic swap M128N64
    # (2026-09-02, fold layout): each +3..+4% over the ping-pong twin the
    # 2026-08-19 table selected, consistent across three interleaved runs on
    # two H200 nodes.
    # Swap-AB token tile N=8 (2026-09-10, wgmma m64n8k32, same-node
    # interleaved A/B x2 vs the N>=16 rows, 1830 MHz): per_tensor 64 basic
    # M128N8 +13.5% (16 tokens/expert fill two N=8 tiles instead of one N=64
    # tile that is 3/4 padding) and per_tensor 128 ping-pong M128N8 +4.6%.
    # per_tensor 32 moved from non-swap M64N256 to cooperative swap M256N8
    # CGA2x1 (+5.7% / +6.4% e2e over two rounds, compute +7%; the ping-pong
    # swap M128N8 twin only ties the non-swap tile).  Not switched: per_tensor
    # 16 (+1.8%) and blockwise 64 (+1.7..+2.9%) -- measurable but judged too
    # small to move the table; pt8 / bw8 / bw16 / bw32 (+0.2..+1.2%, within
    # the +-1% run noise); bw128 / bw256 / pt256 (-15..-42%: twice the tile
    # count outweighs the padding win once every N>=16 tile is full).
    # blockwise non-swap 512-32768: cooperative M64N256 (two epilogue WGs on
    # one tile), 2026-09-02 4x H200 under the fold_producer_warps layout:
    # +11..+30% over both the basic M64N128 tile and its ping-pong twin at the
    # same cluster shape / token-back (geomean +15.8%).  Under the old
    # producer-warpgroup layout the same cooperative tile ran 14% BEHIND
    # basic, which is why it was never selected before; the layout, not the
    # register refit, is what makes the 2-WG modes viable here.
    "blockwise": {
        8: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1)),
        16: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(1, 1, 1)),
        # 32: tail-split pair tasks -3% (2026-09-18, two rounds, all ranks).
        32: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(1, 2, 1),
                    tail_split_pairs=True),
        64: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1)),
        128: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1)),
        256: _config(swap_ab=True, pingpong=True, tile=(128, 32, 128), cga=(1, 2, 1)),
        # 512-8192 (2026-09-18, 4x H200, two interleaved rounds): group_hint
        # 264 -3% (512); epi_warps + group_hint 264 -21% / -16% (1024 / 2048);
        # epi_warps -3% (4096); tail-split + group_hint 264 + epi_warps -8%
        # (8192).  Swap-AB M128N128 is 5x slower than M64N256 under blockwise.
        512: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(1, 1, 1),
                     group_hint=264),
        1024: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 2, 1),
                      group_hint=264),
        2048: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 2, 1),
                      group_hint=264),
        4096: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(1, 1, 1)),
        8192: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1),
                      group_hint=264, tail_split_pairs=True),
        16384: _config(
            swap_ab=False,
            pingpong=False,
            tile=(64, 256, 128),
            cga=(1, 2, 1),
            token_back="reuse_dispatch_warps",
        ),
        32768: _config(
            swap_ab=False,
            pingpong=False,
            tile=(64, 256, 128),
            cga=(2, 1, 1),
            token_back="reuse_dispatch_warps",
        ),
    },
}


# generate_c (training forward) overrides.  The swap-AB M128N128 ping-pong
# kernel spills once the raw fc1_c store is added, so per_tensor 16384 keeps
# the previous entry there (2026-09-18: 13.4 ms vs 14.3 ms with the table
# entry; inference stays on the table entry, 12.0 ms vs 13.1 ms).
HEURISTIC_GENERATE_C_OVERRIDES = {
    "per_tensor": {
        16384: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1),
                       token_back="reuse_dispatch_warps"),
    },
    "blockwise": {},
}


def token_bucket(tokens_per_rank: int) -> int:
    """Map a positive token count to the next measured power-of-two bucket."""
    if tokens_per_rank <= 0:
        raise ValueError("tokens_per_rank must be positive")
    for bucket in TOKEN_BUCKETS:
        if tokens_per_rank <= bucket:
            return bucket
    return TOKEN_BUCKETS[-1]


def tail_split_env_enabled() -> bool:
    """Parse ``FP8_TAIL_SPLIT``; unset / ``0`` is off, ``1`` is on."""
    value = os.environ.get(TAIL_SPLIT_ENV, "0").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{TAIL_SPLIT_ENV} must be 0 or 1, got {value!r}.")


def _apply_tail_split_env_override(config: HopperFp8Config) -> HopperFp8Config:
    """Turn the split on when the env asks for it and the geometry allows it.

    Configs without exactly two CTAs along the tokens and one along the
    weights are returned unchanged (the kernel validator would reject the
    flag there), so a token sweep that crosses cga(1,1,1), cga(2,2,1) or swap
    cga(2,1,1) buckets keeps running.
    """
    if not tail_split_env_enabled() or config.tail_split_pairs:
        return config
    if not config.tail_split_geometry:
        return config
    return replace(config, tail_split_pairs=True)


def select_heuristic_config(
    scale_mode: str, tokens_per_rank: int, *, generate_c: bool = False
) -> HopperFp8ConfigSelection:
    normalized_scale_mode = scale_mode.replace("-", "_")
    try:
        configs = HEURISTIC_CONFIGS[normalized_scale_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported FP8 scale mode: {scale_mode!r}") from error
    bucket = token_bucket(tokens_per_rank)
    config = configs[bucket]
    if generate_c:
        config = HEURISTIC_GENERATE_C_OVERRIDES[normalized_scale_mode].get(bucket, config)
    return HopperFp8ConfigSelection(
        config=_apply_tail_split_env_override(config),
        source="heuristic",
        token_bucket=bucket,
    )


def resolve_hopper_fp8_config(
    scale_mode: str,
    tokens_per_rank: int,
    *,
    swap_ab: Optional[bool] = None,
    pingpong: Optional[bool] = None,
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mnk: Optional[Tuple[int, int, int]] = None,
    accum_mode: Optional[str] = None,
    generate_c: bool = False,
) -> HopperFp8ConfigSelection:
    """Select the heuristic unless geometry or scheduling was set manually."""
    manual = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    resolved_accum_mode = accum_mode or DEFAULT_ACCUM_MODE
    if not manual:
        selection = select_heuristic_config(
            scale_mode, tokens_per_rank, generate_c=generate_c
        )
        if resolved_accum_mode == selection.config.accum_mode:
            return selection
        return HopperFp8ConfigSelection(
            config=replace(selection.config, accum_mode=resolved_accum_mode),
            source=selection.source,
            token_bucket=selection.token_bucket,
        )

    resolved_swap_ab = bool(swap_ab)
    resolved_pingpong = bool(pingpong)
    resolved_tile = mma_tiler_mnk or DEFAULT_MMA_TILER_MNK
    if resolved_swap_ab and resolved_tile == DEFAULT_MMA_TILER_MNK:
        resolved_tile = (128, 32, 128) if resolved_pingpong else (256, 32, 128)
    return HopperFp8ConfigSelection(
        config=_apply_tail_split_env_override(
            HopperFp8Config(
                swap_ab=resolved_swap_ab,
                pingpong=resolved_pingpong,
                mma_tiler_mnk=resolved_tile,
                cluster_shape_mnk=(
                    cluster_shape_mnk or DEFAULT_CLUSTER_SHAPE_MNK
                ),
                accum_mode=resolved_accum_mode,
            )
        ),
        source="manual",
        token_bucket=None,
    )


__all__ = [
    "HEURISTIC_CONFIGS",
    "HEURISTIC_GENERATE_C_OVERRIDES",
    "HopperFp8Config",
    "HopperFp8ConfigSelection",
    "TAIL_SPLIT_ENV",
    "TOKEN_BUCKETS",
    "resolve_hopper_fp8_config",
    "select_heuristic_config",
    "tail_split_env_enabled",
    "token_bucket",
]
