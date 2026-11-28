# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Token-bucket launch heuristics for Hopper BF16 MegaMoE.

The four-rank defaults select a tile shape and schedule for each token-count
bucket.
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
    # Cross-rank FC2 write-back placement.
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
    # A K=64 tile reduces shared-memory use per stage for two-byte
    # operands. Larger FC1 groups leave room for FC2 to start after
    # the tiles it depends on are ready.
    8: _config(swap_ab=True, pingpong=False, tile=(128, 16, 64), cga=(2, 1, 1),
               group_hint=264),
    16: _config(swap_ab=True, pingpong=False, tile=(128, 16, 64), cga=(1, 2, 1),
                group_hint=264),
    # Small N tiles reduce padding when experts receive few tokens.
    32: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 1, 1),
                group_hint=ALL_EXPERTS_GROUP),
    64: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 1, 1),
                group_hint=ALL_EXPERTS_GROUP),
    128: _config(swap_ab=True, pingpong=False, tile=(128, 8, 64), cga=(1, 2, 1),
                 group_hint=264),
    256: _config(swap_ab=True, pingpong=False, tile=(128, 32, 64), cga=(2, 1, 1),
                 group_hint=264),
    # The larger swap-AB tile leaves only one AB stage with BF16 operands.
    512: _config(swap_ab=True, pingpong=False, tile=(128, 64, 64), cga=(1, 1, 1),
                 group_hint=ALL_EXPERTS_GROUP),
    # Split short expert tails to keep both CTAs of the pair working.
    1024: _config(swap_ab=True, pingpong=True, tile=(128, 64, 64), cga=(1, 2, 1),
                  group_hint=264, tail_split_pairs=True),
    # Split ragged expert tails so the second CTA can work on another
    # tail instead of processing an all-padding tile.
    2048: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
    4096: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
    8192: _config(swap_ab=True, pingpong=True, tile=(128, 128, 64), cga=(1, 2, 1), group_hint=264, tail_split_pairs=True),
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
    value = os.environ.get("BF16_TAIL_SPLIT", "0").strip()
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
