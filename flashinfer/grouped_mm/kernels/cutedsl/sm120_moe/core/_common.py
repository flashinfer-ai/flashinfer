# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Constants and integer helpers these SM120 MoE kernels share."""

TMA_ALIGN_BYTES = 16
UE8M0_PACK_NUM = 4
SF_ELEM_BYTES = 4
SF_M_ALIGN = TMA_ALIGN_BYTES // SF_ELEM_BYTES


def ceil_div(a, b):
    """Tiles/packs needed to cover ``a``; a floor here silently drops the partial tail."""
    return (a + b - 1) // b


def align(a, b):
    return ceil_div(a, b) * b


def compute_padded_offset(offset, expert_idx, alignment):
    """Where expert ``expert_idx``'s rows start in the per-expert-aligned scale-factor buffer."""
    return (offset + expert_idx * (alignment - 1)) // alignment * alignment
