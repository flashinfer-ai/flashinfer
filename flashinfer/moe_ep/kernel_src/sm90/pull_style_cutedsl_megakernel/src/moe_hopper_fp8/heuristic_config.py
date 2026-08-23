# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Token-bucket launch heuristics for Hopper FP8 MegaMoE.

The table is derived from the 2026-08-19 four-rank H200 DeepSeek-V4 P03
sweep. Each entry maximizes the slowest-rank effective TFLOPS over operand
order, tile shape, CGA shape, and legacy/ping-pong scheduling.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Optional, Tuple

TOKEN_BUCKETS = tuple(1 << power for power in range(3, 16))
DEFAULT_MMA_TILER_MNK = (64, 128, 128)
DEFAULT_CLUSTER_SHAPE_MNK = (1, 1, 1)
DEFAULT_ACCUM_MODE = "1xacc"


@dataclass(frozen=True)
class HopperFp8Config:
    swap_ab: bool
    pingpong: bool
    mma_tiler_mnk: Tuple[int, int, int]
    cluster_shape_mnk: Tuple[int, int, int]
    accum_mode: str = DEFAULT_ACCUM_MODE


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
) -> HopperFp8Config:
    return HopperFp8Config(
        swap_ab=swap_ab,
        pingpong=pingpong,
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cga,
    )


HEURISTIC_CONFIGS = {
    "per_tensor": {
        8: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(2, 1, 1)),
        16: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(1, 2, 1)),
        32: _config(swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(1, 1, 1)),
        64: _config(swap_ab=True, pingpong=True, tile=(128, 64, 128), cga=(1, 2, 1)),
        128: _config(swap_ab=True, pingpong=True, tile=(128, 32, 128), cga=(1, 2, 1)),
        256: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1)),
        512: _config(swap_ab=True, pingpong=False, tile=(256, 64, 128), cga=(1, 1, 1)),
        1024: _config(swap_ab=True, pingpong=True, tile=(128, 64, 128), cga=(1, 2, 1)),
        2048: _config(swap_ab=False, pingpong=True, tile=(64, 128, 128), cga=(2, 1, 1)),
        4096: _config(swap_ab=False, pingpong=True, tile=(64, 128, 128), cga=(2, 2, 1)),
        8192: _config(swap_ab=True, pingpong=True, tile=(128, 64, 128), cga=(1, 2, 1)),
        16384: _config(
            swap_ab=False, pingpong=False, tile=(64, 256, 128), cga=(2, 1, 1)
        ),
        32768: _config(
            swap_ab=False, pingpong=True, tile=(64, 128, 128), cga=(2, 2, 1)
        ),
    },
    "blockwise": {
        8: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1)),
        16: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(1, 1, 1)),
        32: _config(swap_ab=True, pingpong=True, tile=(128, 16, 128), cga=(1, 2, 1)),
        64: _config(swap_ab=True, pingpong=False, tile=(256, 32, 128), cga=(2, 1, 1)),
        128: _config(swap_ab=True, pingpong=False, tile=(256, 16, 128), cga=(2, 1, 1)),
        256: _config(swap_ab=True, pingpong=True, tile=(128, 32, 128), cga=(1, 2, 1)),
        512: _config(swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(1, 1, 1)),
        1024: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(2, 2, 1)
        ),
        2048: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(2, 2, 1)
        ),
        4096: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(1, 1, 1)
        ),
        8192: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(2, 1, 1)
        ),
        16384: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(1, 2, 1)
        ),
        32768: _config(
            swap_ab=False, pingpong=False, tile=(64, 128, 128), cga=(2, 1, 1)
        ),
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


def select_heuristic_config(
    scale_mode: str, tokens_per_rank: int
) -> HopperFp8ConfigSelection:
    normalized_scale_mode = scale_mode.replace("-", "_")
    try:
        configs = HEURISTIC_CONFIGS[normalized_scale_mode]
    except KeyError as error:
        raise ValueError(f"Unsupported FP8 scale mode: {scale_mode!r}") from error
    bucket = token_bucket(tokens_per_rank)
    return HopperFp8ConfigSelection(
        config=configs[bucket], source="heuristic", token_bucket=bucket
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
) -> HopperFp8ConfigSelection:
    """Select the heuristic unless geometry or scheduling was set manually."""
    manual = any(
        value is not None
        for value in (swap_ab, pingpong, mma_tiler_mnk, cluster_shape_mnk)
    )
    resolved_accum_mode = accum_mode or DEFAULT_ACCUM_MODE
    if not manual:
        selection = select_heuristic_config(scale_mode, tokens_per_rank)
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
        config=HopperFp8Config(
            swap_ab=resolved_swap_ab,
            pingpong=resolved_pingpong,
            mma_tiler_mnk=resolved_tile,
            cluster_shape_mnk=(cluster_shape_mnk or DEFAULT_CLUSTER_SHAPE_MNK),
            accum_mode=resolved_accum_mode,
        ),
        source="manual",
        token_bucket=None,
    )


__all__ = [
    "HEURISTIC_CONFIGS",
    "HopperFp8Config",
    "HopperFp8ConfigSelection",
    "TOKEN_BUCKETS",
    "resolve_hopper_fp8_config",
    "select_heuristic_config",
    "token_bucket",
]
