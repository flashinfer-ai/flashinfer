# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Token-bucket launch heuristics for Hopper FP8 MegaMoE.

The table is derived from four-rank H200 DeepSeek-V4 P03 sweeps (geometry
first, then a tail-split / group_hint / token-back retune).  Each entry
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
        # 8-256 (4x H200, two interleaved rounds): group_hint 264
        # keeps FC2 tiles from spinning on fc1_done right behind their FC1
        # tiles (the one-wave default made that spin 25% of the TMA producer):
        # -2/-15/-3/-11/-12/-4%.
        8: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1),
                   group_hint=264),
        16: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(1, 2, 1),
                    group_hint=264),
        32: _config(swap_ab=True, pingpong=False, tile=(256, 8, 128), cga=(2, 1, 1),
                    group_hint=264),
        64: _config(swap_ab=True, pingpong=False, tile=(128, 8, 128), cga=(1, 2, 1),
                    group_hint=264),
        128: _config(swap_ab=True, pingpong=True, tile=(128, 8, 128), cga=(1, 2, 1),
                     group_hint=264),
        256: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1),
                     group_hint=264),
        # 512 / 1024 (4x H200, two interleaved rounds): group_hint
        # 264 -3% / -10%; 1024 with tail-split pair tasks on top -15%.
        512: _config(swap_ab=True, pingpong=False, tile=(256, 64, 128), cga=(1, 1, 1),
                     group_hint=264),
        1024: _config(swap_ab=True, pingpong=True, tile=(128, 64, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True),
        # 2048-32768 (4x H200, two interleaved rounds): swap-AB
        # ping-pong M128N128 cga(1,2,1) with tail-split pair tasks beats the
        # previous entries by 25/25/20% (2048/4096/8192, with group_hint 264)
        # and 9/13% (16384/32768).  Token-back re-measured 2026-09-22 (1830 MHz
        # locked node, two interleaved rounds, both rounds agree):
        # reuse_dispatch_warps beats epi_warps by +4.5/+4.1/+3.9/+5.0/+5.1% e2e
        # (+4.6/+7.1/+6.4/+4.4/+5.5% compute) on these five rows.
        2048: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True,
                      token_back="reuse_dispatch_warps"),
        4096: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True,
                      token_back="reuse_dispatch_warps"),
        8192: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True,
                      token_back="reuse_dispatch_warps"),
        16384: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                       tail_split_pairs=True, token_back="reuse_dispatch_warps"),
        32768: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                       tail_split_pairs=True, token_back="reuse_dispatch_warps"),
    },
    # per_tensor 8 -> cooperative swap M256N16 and 64 -> basic swap M128N64
    # (fold layout): each +3..+4% over the ping-pong twin the previous
    # table selected, consistent across three interleaved runs on
    # two H200 nodes.
    # Swap-AB token tile N=8 (wgmma m64n8k32, same-node
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
    # one tile), 4x H200 under the fold_producer_warps layout:
    # +11..+30% over both the basic M64N128 tile and its ping-pong twin at the
    # same cluster shape / token-back (geomean +15.8%).  Under the old
    # producer-warpgroup layout the same cooperative tile ran 14% BEHIND
    # basic, which is why it was never selected before; the layout, not the
    # register refit, is what makes the 2-WG modes viable here.
    "blockwise": {
        # 8-256 (4x H200, two interleaved rounds): group_hint 264
        # -2/-1/-17/-4/-3/-20% (see per_tensor); 32 also tail-split -3%.
        8: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1),
                   group_hint=264),
        16: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(1, 1, 1),
                    group_hint=264),
        # 32 / 256 (4x H200 1830 MHz, group_hint 264 everywhere):
        # the 1x128 h scale block needs 128 output channels per task tile, so
        # the swap-AB M128 ping-pong rows had to move to M256 cooperative
        # tiles.  32: M256N8 CGA1x1 1327 / 1329 us vs M256N16 CGA2x1 1334 /
        # 1341 and M256N8 CGA2x1 1333 / 1320 (old M128N16 pp 1311 / 1322,
        # ~1% lost).  256: M256N32 CGA2x1 1580 / 1583 us (old M128N32 pp
        # 1608 / 1609); without the hint M256N32 CGA1x1 was within 0.2%.
        32: _config(swap_ab=True, pingpong=False, tile=(256, 8, 128), cga=(1, 1, 1),
                    group_hint=264),
        64: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1),
                    group_hint=264),
        128: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1),
                     group_hint=264),
        256: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1),
                     group_hint=264),
        # 512-8192 (4x H200, two interleaved rounds): group_hint
        # 264 -3% (512); epi_warps + group_hint 264 -21% / -16% (1024 / 2048);
        # epi_warps -3% (4096); tail-split + group_hint 264 + epi_warps -8%
        # (8192).  Swap-AB M128N128 is 5x slower than M64N256 under blockwise.
        512: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(1, 1, 1),
                     group_hint=264),
        1024: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 2, 1),
                      group_hint=264),
        2048: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 2, 1),
                      group_hint=264),
        # 4096 (4x H200 1830 MHz, 1x128 h scales): M64N256 CGA2x1 + group_hint
        # 264 4090 us vs CGA1x1 5220-5314 (CGA2x2 4249, CGA1x2 4919); tail
        # split on top 3920 / 3958 us.
        4096: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1),
                      group_hint=264, tail_split_pairs=True),
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


# The table was tuned on four ranks with 384 experts and top-6.
_TUNED_WORLD_SIZE = 4
_TUNED_NUM_EXPERTS = 384
_TUNED_NUM_TOPK = 6

# The same model on eight ranks (two nodes x four H100, NCCL GIN; 2026-10-25,
# interleaved A/B rounds) halves the experts per rank and doubles their rows.
# Buckets whose measured best beats the four-rank row by >= 3%, reproduced in
# two rounds: pt 256 -26%, pt 1024 -11%, bw 1024 -4%, bw 4096 -13%.  Blockwise
# wants weight multicast (cga 2,1,1) with the one-wave group; per-tensor 1024
# takes the four-rank 2048 row.  Other eight-rank buckets were within noise.
EP8_OVERRIDES = {
    "per_tensor": {
        256: _config(swap_ab=True, pingpong=True, tile=(128, 32, 128), cga=(1, 2, 1),
                     group_hint=264),
        1024: _config(swap_ab=True, pingpong=True, tile=(128, 128, 128), cga=(1, 2, 1),
                      group_hint=264, tail_split_pairs=True,
                      token_back="reuse_dispatch_warps"),
    },
    "blockwise": {
        1024: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1),
                      tail_split_pairs=True),
        4096: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1),
                      tail_split_pairs=True),
    },
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
    scale_mode: str,
    tokens_per_rank: int,
    *,
    world_size: int = _TUNED_WORLD_SIZE,
    num_experts: int = _TUNED_NUM_EXPERTS,
    num_topk: int = _TUNED_NUM_TOPK,
) -> HopperFp8ConfigSelection:
    normalized_scale_mode = scale_mode.replace("-", "_")
    try:
        configs = HEURISTIC_CONFIGS[normalized_scale_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported FP8 scale mode: {scale_mode!r}") from error
    bucket = token_bucket(tokens_per_rank)
    config = configs[bucket]
    if (world_size, num_experts, num_topk) == (8, _TUNED_NUM_EXPERTS, _TUNED_NUM_TOPK):
        config = EP8_OVERRIDES[normalized_scale_mode].get(bucket, config)
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
    world_size: int = _TUNED_WORLD_SIZE,
    num_experts: int = _TUNED_NUM_EXPERTS,
    num_topk: int = _TUNED_NUM_TOPK,
) -> HopperFp8ConfigSelection:
    """Select the heuristic unless geometry or scheduling was set manually."""
    manual = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    resolved_accum_mode = accum_mode or DEFAULT_ACCUM_MODE
    if not manual:
        selection = select_heuristic_config(
            scale_mode,
            tokens_per_rank,
            world_size=world_size,
            num_experts=num_experts,
            num_topk=num_topk,
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
    if scale_mode == "blockwise" and resolved_pingpong:
        # The 1x128 h scale block needs 128 output channels per task tile;
        # ping-pong is an N128 / M128 schedule (64 channels).
        raise ValueError("blockwise FP8 has no ping-pong geometry (1x128 h scales need N256 / M256 tiles)")
    resolved_tile = mma_tiler_mnk or DEFAULT_MMA_TILER_MNK
    if resolved_swap_ab and resolved_tile == DEFAULT_MMA_TILER_MNK:
        resolved_tile = (128, 32, 128) if resolved_pingpong else (256, 32, 128)
    elif scale_mode == "blockwise" and resolved_tile == DEFAULT_MMA_TILER_MNK:
        resolved_tile = (64, 256, 128)
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
    "HopperFp8Config",
    "HopperFp8ConfigSelection",
    "TAIL_SPLIT_ENV",
    "EP8_OVERRIDES",
    "TOKEN_BUCKETS",
    "resolve_hopper_fp8_config",
    "select_heuristic_config",
    "tail_split_env_enabled",
    "token_bucket",
]
