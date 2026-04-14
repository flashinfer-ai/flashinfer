# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared softmax math primitives for attention kernels.

Used by FMHA (SoftmaxRole) to avoid
duplicating the core exp2-scale and packed row-sum reduction logic.
"""

import cutlass
import cutlass.cute as cute


@cute.jit
def exp2_scale(scores, scale_log2, row_max):
    """Apply e^((x - row_max) * scale_log2) to all paired elements in-place."""
    minus_max_scale = (0.0 - row_max) * scale_log2
    for i in cutlass.range_constexpr(0, cute.size(scores), 2):
        scores[i], scores[i + 1] = cute.arch.fma_packed_f32x2(
            (scores[i], scores[i + 1]),
            (scale_log2, scale_log2),
            (minus_max_scale, minus_max_scale),
        )
        scores[i] = cute.arch.exp2(scores[i])
        scores[i + 1] = cute.arch.exp2(scores[i + 1])


@cute.jit
def packed_row_sum(scores) -> tuple:
    """Reduce all elements of a 1D register fragment via packed f32x2 adds.

    Returns (sum_even_indices, sum_odd_indices) tuple; caller typically
    does ``vec[0] + vec[1]`` to get the scalar total.
    """
    row_sum_vec = (0.0, 0.0)
    for i in cutlass.range_constexpr(0, cute.size(scores), 2):
        row_sum_vec = cute.arch.add_packed_f32x2(
            row_sum_vec, (scores[i], scores[i + 1])
        )
    return row_sum_vec
