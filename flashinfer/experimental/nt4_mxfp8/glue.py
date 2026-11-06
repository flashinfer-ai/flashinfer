# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Route localization and slot-ordered FP32 partial assembly."""

import triton
import triton.language as tl


@triton.jit
def _route_mask(
    Ids,
    Probs,
    LocalIds,
    LocalProbs,
    N: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(Ids + i, i < N, other=-1)
    owned = (expert >= RANK * 128) & (expert < (RANK + 1) * 128)
    p = tl.load(Probs + i, (i < N) & owned, other=0.0)
    tl.store(LocalIds + i, tl.where(owned, expert - RANK * 128, -1), i < N)
    tl.store(LocalProbs + i, p, i < N)


@triton.jit
def _weighted_partial(
    Terms, Ids, Probs, Output, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(K):
        valid = tl.load(Ids + row * K + slot) >= 0
        p = tl.load(Probs + row * K + slot, valid, other=0.0)
        value = tl.load(
            Terms + (row * K + slot) * H + h, (h < H) & valid, other=0.0
        ).to(tl.float32)
        acc = acc + value * p
    tl.store(Output + row * H + h, acc, h < H)


def route_mask(ids, probs, local_ids, local_probs, rank):
    _route_mask[(triton.cdiv(ids.numel(), 256),)](
        ids, probs, local_ids, local_probs, ids.numel(), rank, 256
    )


def weighted_partial(terms, ids, probs, output):
    # Contiguous [row, slot, channel] BF16 -> FP32, in slot order. Keep
    # multiply/add separate to make the numerical contract explicit.
    _weighted_partial[(ids.shape[0], triton.cdiv(output.shape[1], 256))](
        terms,
        ids,
        probs,
        output,
        output.shape[1],
        ids.shape[1],
        256,
        enable_fp_fusion=False,
    )
