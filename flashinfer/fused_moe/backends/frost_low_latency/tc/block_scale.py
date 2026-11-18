# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared FP32 promotion for native FP8 fragments with 128x128 weight scales."""

import cutlass
from cutlass import cute
from ..quantization import _mul


@cute.jit
def promote_block_scaled(
    partial,
    total,
    coordinates,
    weight_scales,
    input_scales,
    members,
    expert,
    output_width,
    tile_row,
    k_tile,
    rows_per_tile: cutlass.Constexpr,
    tile_k: cutlass.Constexpr,
    group_size: cutlass.Constexpr,
    split_gate: cutlass.Constexpr,
    input_route_divisor: cutlass.Constexpr,
):
    # Every K64 partial belongs to one K128 scale block. The tile itself is
    # K64 or K128, so it cannot cross that block within a partial promotion.
    scale_block = (k_tile * tile_k) // 128
    for element in cutlass.range_constexpr(cute.size(partial)):
        row, column = coordinates[element]
        weight_row = tile_row * rows_per_tile + row
        if cutlass.const_expr(split_gate):
            if row >= 32:
                weight_row = output_width + tile_row * 32 + row - 32
        # Compile-time member indices avoid dynamic indexing of register-held
        # route membership. Unused columns and invalid routes contribute zero.
        for member in cutlass.range_constexpr(group_size):
            if column == member and members[member] >= 0:
                scale = _mul(
                    weight_scales[expert, weight_row // 128, scale_block],
                    input_scales[scale_block, members[member] // input_route_divisor],
                )
                total[element] = total[element] + _mul(partial[element], scale)
