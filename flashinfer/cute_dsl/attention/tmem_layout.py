# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""TmemLayout — computed TMEM allocation plan.

Derives TMEM offsets from the AttentionConfig's tile shape instead of using
hardcoded magic numbers. The layout follows the pattern:

  S0 @ 0, S1 @ tile_m, O0 @ 2*tile_m, O1 @ 3*tile_m
  P0 aliased inside S region at tile_m//4, P1 at tile_m + tile_m//4
  Vec buffers (row_max, row_sum) at start of S0 and S1 regions
"""

from __future__ import annotations

from dataclasses import dataclass

from .config import AttentionConfig


@dataclass(frozen=True)
class TmemLayout:
    """TMEM offset map for attention kernel score/output/P buffers."""

    s0_offset: int
    s1_offset: int
    o0_offset: int
    o1_offset: int
    p0_offset: int
    p1_offset: int
    vec0_offset: int
    vec1_offset: int
    alloc_cols: int

    @staticmethod
    def from_config(config: AttentionConfig) -> TmemLayout:
        tile_m = config.mma_tiler[0]
        SM100_TMEM_CAPACITY_COLUMNS = 512
        return TmemLayout(
            s0_offset=0,
            s1_offset=tile_m,
            o0_offset=2 * tile_m,
            o1_offset=3 * tile_m,
            p0_offset=tile_m // 4,
            p1_offset=tile_m + tile_m // 4,
            vec0_offset=0,
            vec1_offset=tile_m,
            alloc_cols=SM100_TMEM_CAPACITY_COLUMNS,
        )
