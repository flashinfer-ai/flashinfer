# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Geometry shared by the KPage kernels and their metadata builders.

from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Boolean, Float32, Int32, Uint32, const_expr

from flashinfer.experimental.vsa_kpage.csrc.utils.softmax import SoftmaxSm100

BLOCK = 128  # logical Q/KV block and MMA tile size
PAGE = 16  # K/V gather granularity in rows
FRAGS_PER_TILE = BLOCK // PAGE  # pages per 128-row K/V tile
HEAD_DIM = 128


@cute.jit
def apply_packed_mask_chunk(
    acc_s: cute.Tensor,
    chunk_idx: cutlass.Constexpr[int],
    mask: Int32,
) -> None:
    """Mask 32 score columns with one packed keep word (bit set = keep)."""
    col_base = chunk_idx * 32
    for bit_idx in cutlass.range_constexpr(32):
        keep = Boolean(Uint32(mask) & (Uint32(1) << bit_idx))
        col = col_base + bit_idx
        acc_s[col] = acc_s[col] if keep else -Float32.inf


@cute.jit
def update_row_max_precomputed(
    softmax: SoftmaxSm100, row_max_tile: Float32, is_first: bool
) -> Tuple[Float32, Float32]:
    """SoftmaxSm100.update_row_max for a tile maximum already reduced in hardware (tcgen05.ld red.max)."""
    if const_expr(is_first):
        row_max_new = row_max_tile
        row_max_safe = row_max_new if row_max_new != -Float32.inf else Float32(0.0)
        acc_scale = Float32(0.0)
    else:
        row_max_old = softmax.row_max[0]
        row_max_new = cute.arch.fmax(row_max_tile, row_max_old)
        row_max_safe = row_max_new if row_max_new != -Float32.inf else Float32(0.0)
        acc_scale_log2 = (row_max_old - row_max_safe) * softmax.scale_log2
        acc_scale = cute.math.exp2(acc_scale_log2, fastmath=True)
        if const_expr(softmax.rescale_threshold > 0.0):
            if acc_scale_log2 >= -softmax.rescale_threshold:
                row_max_new = row_max_old
                row_max_safe = row_max_old
                acc_scale = Float32(1.0)
    softmax.row_max[0] = row_max_new
    return row_max_safe, acc_scale
