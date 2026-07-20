# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared masking predicates for PrimTS attention kernels."""


def kv_tile_needs_right_mask(tile_offset_k, tile_size_kv, visible_k_end):
    """Return whether a KV tile crosses a row-visible right bound."""
    return tile_offset_k + tile_size_kv > visible_k_end


def kv_tile_is_fully_visible(
    tile_offset_k,
    tile_size_kv,
    visible_k_begin,
    visible_k_end,
):
    """Return whether a KV tile is inside every row's visible interval.

    All intervals are half-open.  The arguments may be Python integers in
    host-side tests or CuTe DSL integer values while tracing a kernel.
    """
    return (tile_offset_k >= visible_k_begin) & (
        tile_offset_k + tile_size_kv <= visible_k_end
    )
