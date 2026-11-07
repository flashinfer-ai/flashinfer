# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
# Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad.
# Original source: csrc/resattn/output.py
"""Device helpers shared by all packed NVFP4 attention output paths."""

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32


@cute.jit
def quantize_nvfp4_block16(values: cute.TensorSSA, global_scale: Float32):
    """Quantize an FP32 block into NVFP4 values and a block scale."""
    assert cute.size(values.shape) == 16
    amax = Float32(0)
    for i in cutlass.range_constexpr(16):
        value = Float32(values[i])
        amax = cute.arch.fmax(amax, cute.arch.fmax(value, -value))
    scale = cutlass.Float8E4M3FN(amax * global_scale * Float32(1.0 / 6.0))
    rounded_scale = Float32(scale)
    multiplier = Float32(0)
    if rounded_scale != Float32(0):
        multiplier = global_scale * cute.arch.rcp_approx(rounded_scale)
    return (values * multiplier).to(cutlass.Float4E2M1FN), scale


@cute.jit
def output_scale_offset(row: Int32, block: Int32, hidden: int):
    """Byte offset of a block16 scale in CUTLASS's 128x4 swizzle."""
    k_tiles = cute.ceil_div(hidden, 64)
    return (
        (row // 128 * k_tiles + block // 4) * 512
        + row % 32 * 16
        + (row // 32) % 4 * 4
        + block % 4
    )
