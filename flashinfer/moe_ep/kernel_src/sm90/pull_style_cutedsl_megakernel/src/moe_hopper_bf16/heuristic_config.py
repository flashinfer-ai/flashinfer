# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Token-bucket launch heuristics for Hopper BF16 MegaMoE.

The table is derived from the 2026-08-19 four-rank H200 DeepSeek-V4 P03
sweep. Each entry maximizes the slowest-rank effective TFLOPS over operand
order, tile shape, CGA shape, and legacy/ping-pong scheduling.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from typing import Optional, Tuple

TOKEN_BUCKETS = tuple(1 << power for power in range(3, 16))
DEFAULT_MMA_TILER_MNK = (64, 128, 64)
DEFAULT_CLUSTER_SHAPE_MNK = (1, 1, 1)


DEFAULT_TOKEN_BACK_MODE = "epi_warps"
# group_hint that puts every expert into one scheduler group.
ALL_EXPERTS_GROUP = 1 << 20
# BF16_TAIL_SPLIT=1 turns tail-split pair tasks on for every selected config
# with a 2-CTA token cluster; other geometries keep the table value.
TAIL_SPLIT_ENV = "BF16_TAIL_SPLIT"


@dataclass(frozen=True)
class HopperBf16Config:
    swap_ab: bool
    pingpong: bool
    mma_tiler_mnk: Tuple[int, int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    # Cross-rank fc2 write-back placement; epi_warps won every bucket
    # (4x H200 SXM, 2026-09-17, 5-6% ahead of reuse_dispatch_warps at 16384+).
    token_back_mode: str = DEFAULT_TOKEN_BACK_MODE
    # Scheduler group size in FC1 cluster tiles (None = one wave of clusters).
    # Several experts per group keep FC2 tiles from spinning on fc1_done right
    # behind their FC1 tiles; only matters for small experts.
    group_hint: Optional[int] = None
    # Tail-split pair tasks (see fc1_fc2_fuse_sched); legal only with a 2-CTA
    # token cluster: swap-AB cga (1,2,1) or non-swap cga (2,1,1).
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
class HopperBf16ConfigSelection:
    config: HopperBf16Config
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
) -> HopperBf16Config:
    return HopperBf16Config(
        swap_ab=swap_ab,
        pingpong=pingpong,
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cga,
        token_back_mode=token_back,
        group_hint=group_hint,
        tail_split_pairs=tail_split_pairs,
    )


HEURISTIC_CONFIGS = {
    # BF16 schedule-twin study (a 4x H200 SXM node, 2026-09-09, 2 rounds each):
    # every bucket was measured against its basic / ping-pong / cooperative twin
    # (operand order, CGA and token-back preserved).  Cooperative (two epilogue
    # warpgroups: non-swap M64N256, swap M256xN) is either infeasible under
    # BF16 (1 AB stage) or 16-83% slower, and basic vs ping-pong is within ~1%
    # on most buckets.  Each entry below is the measured best twin; where basic
    # and ping-pong differ by <1% the faster median is taken as-is.
    # 8 / 16: basic beat ping-pong by 0.9% / 0.4% (1492 vs 1506, 2455 vs 2466 us).
    #
    # Tile K = 64 everywhere (two 4x H200 SXM nodes, 2026-09-11): with 2-byte
    # operands a K=128 stage holds two 128 B swizzle atoms and the AB ring gets
    # only half the FP8-era stage count (e.g. Mega M64N128: 3 vs 7).  K=64
    # halves the bytes per stage, doubles the stages (the per-stage WGMMA
    # count, bytes and MMA time then match FP8 K=128) and re-admits the N=256 /
    # swap M128N128 shapes that only got 2 stages.  Same geometry at K=64 vs
    # K=128: 16 -2.6% (2232 -> 2173 us), 512 -4.2%, 1024 -7.0%; the padding
    # bound buckets 8 / 32 / 64 / 128 / 256 are within +-1% on both nodes
    # (3-round A/B: -0.6 / +0.7 / +0.3 / +0.9 / -1.1%), so the table is kept
    # uniform at K=64 instead of mixing K.
    # 8-1024 (4x H200, 2026-09-19, two interleaved rounds): several experts per
    # scheduler group stop FC2 tiles from spinning on fc1_done right behind
    # their FC1 tiles (25% of the activation TMA producer on FP8 at 16 tokens);
    # group_hint 264 -5/-8/-5/-5/-6/-6/-8/-15% at 8..1024, one group for all
    # experts another 1.5-2% at 32 / 64 / 512.
    8: _config(swap_ab=True, pingpong=False, tile=(128, 16, 64), cga=(2, 1, 1),
               group_hint=264),
    16: _config(swap_ab=True, pingpong=False, tile=(128, 16, 64), cga=(1, 2, 1),
                group_hint=264),
    # Swap-AB token tile N=8 (a 4x H200 SXM node, 2026-09-10, 2 rounds each vs the
    # previous entry): with 32-128 tokens/rank an expert receives only a few
    # tokens, so the N>=16 token tiles are mostly padding and basic swap M128N8
    # wins: 32 non-swap pp M64N128 -> M128N8 CGA1x1 (3220 -> 2859 us, +12.6%);
    # 64 swap pp M128N64 -> M128N8 CGA1x1 (4610 -> 3300 us, +39.7%; CGA1x2 3400);
    # 128 swap basic M128N32 -> M128N8 CGA1x2 (3646 -> 3520 us, +3.6%).  Not
    # switched: 8 (+0.8%) and 16 (+1.4%) are within noise, 256 loses 47%+ at
    # N=8 (tiles already full), and cooperative M256N8 loses everywhere.
    32: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 1, 1),
                group_hint=ALL_EXPERTS_GROUP),
    64: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 1, 1),
                group_hint=ALL_EXPERTS_GROUP),
    # 128: basic M128N8 (see the N=8 note above); basic vs ping-pong within 0.4%.
    128: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 2, 1),
                 group_hint=264),
    # BF16 schedule-twin study (a 4x H200 SXM node, 2026-09-09): the cooperative
    # swap M256N32 entry lost to basic swap M128N32 by 27% (4322 vs 3408 us).
    256: _config(swap_ab=True, pingpong=False, tile=(128, 32, 64), cga=(2, 1, 1),
                 group_hint=264),
    # 512: the FP8-era swap-AB M256N64 tile leaves a single AB stage under 2-byte
    # operands (an 8x H200 NVL node, 2026-09-03); M128N64 CGA1x1 won that A/B and the
    # twin study puts basic marginally ahead of ping-pong (3937 vs 3940 us).
    512: _config(swap_ab=True, pingpong=False, tile=(128, 64, 64), cga=(1, 1, 1),
                 group_hint=ALL_EXPERTS_GROUP),
    # 1024: one 64-token CTA tile per expert, so the split keeps the second CTA
    # busy: 3895 -> 3800 us (-5%).  16 / 128 (N16 / N8) did not gain and stay off.
    1024: _config(swap_ab=True, pingpong=True, tile=(128, 64, 64), cga=(1, 2, 1),
                  group_hint=264, tail_split_pairs=True),
    # 2048-32768 (4x H200 SXM, 2026-09-17, block-balanced routing): swap ping-pong
    # M128N128 K64 CGA1x2 with tail-split pair tasks.  Ragged expert tails left the
    # second CTA of the 256-token cluster block on an all-padding tile (+43% /
    # +21% / +9% / +5% CTA tiles at 4096 / 8192 / 16384 / 32768, half of every task
    # at 2048); the split gives 2048 4964 -> 3983, 4096 7673 -> 6592, 8192 12747 ->
    # 11843, 16384 23378 -> 22843, 32768 46530 -> 45576 us.  The former non-swap
    # M64N256 CGA2x1 entries gain only 4% from it and lose (4357 / 7179 us).
    # group_hint 264 (~5 experts) removed 9-16% at 2048-8192 versus the one-wave
    # default (66); larger groups were 2-15% worse, and 16384+ are within 1%.
    2048: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
    4096: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
    8192: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
    # 16384 / 32768: epi_warps beat reuse_dispatch_warps by 5.2% / 5.7%; in-kernel
    # top-k reduce and standalone_warps lost.
    16384: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), tail_split_pairs=True),
    32768: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), tail_split_pairs=True),
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
    """Parse ``BF16_TAIL_SPLIT``; unset / ``0`` is off, ``1`` is on."""
    value = os.environ.get(TAIL_SPLIT_ENV, "0").strip()
    if value in ("", "0"):
        return False
    if value == "1":
        return True
    raise ValueError(f"{TAIL_SPLIT_ENV} must be 0 or 1, got {value!r}.")


def _apply_tail_split_env_override(config: HopperBf16Config) -> HopperBf16Config:
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


def select_heuristic_config(tokens_per_rank: int) -> HopperBf16ConfigSelection:
    bucket = token_bucket(tokens_per_rank)
    return HopperBf16ConfigSelection(
        config=_apply_tail_split_env_override(HEURISTIC_CONFIGS[bucket]),
        source="heuristic",
        token_bucket=bucket,
    )


def resolve_hopper_bf16_config(
    tokens_per_rank: int,
    *,
    swap_ab: Optional[bool] = None,
    pingpong: Optional[bool] = None,
    mma_tiler_mnk: Optional[Tuple[int, int, int]] = None,
    cluster_shape_mnk: Optional[Tuple[int, int, int]] = None,
) -> HopperBf16ConfigSelection:
    """Select the heuristic unless geometry or scheduling was set manually."""
    manual = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    if not manual:
        return select_heuristic_config(tokens_per_rank)

    resolved_swap_ab = bool(swap_ab)
    resolved_pingpong = bool(pingpong)
    resolved_tile = mma_tiler_mnk or DEFAULT_MMA_TILER_MNK
    if resolved_swap_ab and resolved_tile == DEFAULT_MMA_TILER_MNK:
        resolved_tile = (128, 32, 64) if resolved_pingpong else (256, 32, 64)
    return HopperBf16ConfigSelection(
        config=_apply_tail_split_env_override(
            HopperBf16Config(
                swap_ab=resolved_swap_ab,
                pingpong=resolved_pingpong,
                mma_tiler_mnk=resolved_tile,
                cluster_shape_mnk=(
                    cluster_shape_mnk or DEFAULT_CLUSTER_SHAPE_MNK
                ),
            )
        ),
        source="manual",
        token_bucket=None,
    )


__all__ = [
    "HEURISTIC_CONFIGS",
    "HopperBf16Config",
    "HopperBf16ConfigSelection",
    "TAIL_SPLIT_ENV",
    "TOKEN_BUCKETS",
    "resolve_hopper_bf16_config",
    "select_heuristic_config",
    "tail_split_env_enabled",
    "token_bucket",
]
