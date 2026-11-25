# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Stage BF16 shared inputs with the standalone Torch MXFP8 numerical policy."""

import torch
import triton
import triton.language as tl


@triton.jit(do_not_specialize=["live_rows", "capacity"])
def _stage_shared_bf16_kernel(
    x,
    data,
    linear_sf,
    blocked_sf,
    live_rows: tl.int32,
    capacity: tl.int32,
    HIDDEN: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(1)
    cols = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    sf_cols = tl.program_id(0) * (BLOCK // 32) + tl.arange(0, BLOCK // 32)
    sf_width: tl.constexpr = HIDDEN // 32

    if row < live_rows:
        values = tl.load(x + row * HIDDEN + cols, cols < HIDDEN, 0).to(tl.float32)
        blocks = tl.reshape(values, (BLOCK // 32, 32))
        # Torch amax propagates NaN, whereas the routed stager uses fmax.
        absmax = tl.max(tl.abs(blocks), 1)
        has_nan = tl.sum((blocks != blocks).to(tl.int32), 1) != 0
        raw_scale = absmax * (1.0 / 448.0)
        raw_scale = tl.where(has_nan | (raw_scale == float("inf")), 0.0, raw_scale)
        bits = tl.maximum(raw_scale, 2.0**-126).to(tl.int32, bitcast=True)
        exponent = ((bits >> 23) & 255) - 127 + ((bits & 0x7FFFFF) != 0)
        exponent = tl.minimum(tl.maximum(exponent, -126), 127)
        scale = ((exponent + 127) << 23).to(tl.float32, bitcast=True)
        # Division preserves subnormal and overflow behavior at scale 2**-126;
        # multiplying by a capped approximate reciprocal would change bytes.
        scaled = tl.div_rn(blocks, scale[:, None])
        # Match the CUDA Torch cast: round to nearest, saturate finite overflow
        # and infinities, and canonicalize NaNs.
        encoded = scaled.to(tl.float8e4nv).to(tl.uint8, bitcast=True)
        encoded = tl.reshape(encoded, (BLOCK,))
        sf = (exponent + 127).to(tl.uint8)
    else:
        # Dense shared GEMMs consume capacity. Clear all padding on every
        # launch so graphs with different live extents can share the buffers.
        encoded = tl.full((BLOCK,), 0, tl.uint8)
        sf = tl.full((BLOCK // 32,), 0, tl.uint8)

    tl.store(data + row * HIDDEN + cols, encoded, (row < capacity) & (cols < HIDDEN))
    tl.store(
        linear_sf + row * sf_width + sf_cols,
        sf,
        (row < capacity) & (sf_cols < sf_width),
    )
    # The 32x4x4 scale atom, including row padding beyond capacity.
    offset = (
        ((row // 128) * (sf_width // 4) + sf_cols // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + sf_cols % 4
    )
    tl.store(blocked_sf + offset, sf, sf_cols < sf_width)


def stage_shared_bf16(x, workspace):
    """Write the existing data and scale buffers without temporary tensors."""
    capacity, hidden = workspace.shared_x.shape
    _stage_shared_bf16_kernel[
        (triton.cdiv(hidden, 1024), triton.cdiv(capacity, 128) * 128)
    ](
        x,
        workspace.shared_x.view(torch.uint8),
        workspace.shared_x_sf_linear.view(torch.uint8),
        workspace.shared_x_sf.view(torch.uint8),
        x.shape[0],
        capacity,
        hidden,
        1024,
        num_warps=4,
        enable_fp_fusion=False,
    )
