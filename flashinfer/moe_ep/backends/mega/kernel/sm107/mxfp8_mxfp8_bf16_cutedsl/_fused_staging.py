# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused BF16-to-MXFP8 staging for the SM107 backend."""

import triton
import triton.language as tl


@triton.jit
def _stage_kernel(
    packet,
    output,
    scales,
    ids,
    weights,
    out_ids,
    out_weights,
    ROWS,  # runtime: token count varies per call; constexpr would JIT per value
    INPUT_STRIDE: tl.constexpr,
    TOPK: tl.constexpr,
    HIDDEN: tl.constexpr,
    BLOCKS: tl.constexpr,
    ROUTES: tl.constexpr,
    X_STRIDE: tl.constexpr,
    SF_STRIDE: tl.constexpr,
    SF_WIDTH: tl.constexpr,
    SF_BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    block = tl.arange(0, BLOCKS)
    col = tl.arange(0, 32)
    offset = block[:, None] * 32 + col[None, :]
    x = tl.load(
        packet + row * INPUT_STRIDE + offset,
        (row < ROWS) & (block[:, None] < HIDDEN // 32),
        0,
    ).to(tl.float32)
    maximum = tl.max(tl.abs(x), axis=1)
    # Match Torch amax's NaN propagation and the reference's nan_to_num.
    has_nan = tl.sum((x != x).to(tl.int32), axis=1) != 0
    scaled = maximum * (1.0 / 448.0)
    scaled = tl.where(has_nan | (scaled == float("inf")), 0.0, scaled)
    bits = tl.maximum(scaled, 1.1754943508222875e-38).to(tl.int32, bitcast=True)
    exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0).to(tl.int32)
    exponent = tl.minimum(tl.maximum(exponent, -126), 127)
    scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
    normalized = tl.div_rn(x, scale[:, None])
    q = normalized.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
    tl.store(
        output + row * X_STRIDE + offset,
        q,
        (row < ROWS) & (block[:, None] < HIDDEN // 32),
    )
    # Clear only padding, with disjoint stores for the actual scale bytes.
    padding = tl.arange(0, SF_BLOCK)
    tl.store(
        scales + row * SF_STRIDE + padding,
        0,
        (row < ROWS) & (padding >= HIDDEN // 32) & (padding < SF_WIDTH),
    )
    tl.store(
        scales + row * SF_STRIDE + block,
        exponent + 127,
        (row < ROWS) & (block < HIDDEN // 32),
    )

    route = tl.arange(0, ROUTES)
    index = tl.load(ids + row * TOPK + route, (row < ROWS) & (route < TOPK), -1)
    weight = tl.load(weights + row * TOPK + route, (row < ROWS) & (route < TOPK), 0)
    tl.store(out_ids + row * TOPK + route, index, route < TOPK)
    tl.store(out_weights + row * TOPK + route, weight, (row < ROWS) & (route < TOPK))


def fused_stage(
    hidden_states, topk_weights, topk_ids, x_fp8, x_sf, topk_idx_out, topk_weights_out
):
    import torch

    rows, hidden = hidden_states.shape
    topk = topk_ids.shape[1]
    _stage_kernel[(x_fp8.shape[0],)](
        hidden_states,
        x_fp8.view(torch.uint8),
        x_sf.view(torch.uint8),
        topk_ids,
        topk_weights,
        topk_idx_out,
        topk_weights_out,
        rows,
        hidden_states.stride(0),
        topk,
        hidden,
        triton.next_power_of_2(hidden // 32),
        triton.next_power_of_2(topk),
        x_fp8.stride(0),
        x_sf.stride(0),
        x_sf.shape[1],
        triton.next_power_of_2(x_sf.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
