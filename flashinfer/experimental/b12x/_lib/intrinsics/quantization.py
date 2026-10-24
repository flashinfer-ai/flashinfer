# Copyright (c) 2025 by the b12x authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Device block quantization helpers."""

from cutlass import Float32
from cutlass import Int32
from typing import Tuple
from cutlass import Uint32
from cutlass import Uint64
from cutlass import Uint8
import cutlass.cute as cute
import cutlass
from b12x._lib.intrinsics.constants import FLOAT4_E2M1_MAX, FLOAT8_E4M3_MAX
from b12x._lib.intrinsics.conversion import (
    cvt_e2m1x8_f32,
    cvt_f32_to_e4m3,
    cvt_f32x4_to_e4m3x4,
    fp8_e4m3_to_f32,
    fp8_e4m3_to_f32_and_rcp,
    pow2_ceil_ue8m0,
    ue8m0_to_output_scale,
)
from b12x._lib.intrinsics.math import (
    div_rn_f32,
    fabs_f32,
    fmax_f32,
    fmin_f32,
    rcp_approx_ftz,
)
from b12x._lib.intrinsics.memory import get_ptr_as_int64, ld_global_v4_u32
from b12x._lib.intrinsics.packed import (
    bfloat2_habs2,
    bfloat2_hmax2,
    bfloat2_mul,
    bfloat2_to_float2_scaled,
    habs2,
    half2_mul,
    half2_to_float2_scaled,
    hmax2,
)


# =============================================================================
# Helper Functions for SF Block Processing (block_size=16)
# =============================================================================


@cute.jit
def load_8_half2(
    mX: cute.Tensor, mW: cute.Tensor, row_offset: Int32, col_offset: Int32, H: int
):
    """Load 16 elements (8 half2 pairs) of X and W from global memory.

    Returns:
        x_h2: rmem_tensor of shape (8,) containing X as half2
        w_h2: rmem_tensor of shape (8,) containing W as half2
    """
    x_h2 = cute.make_rmem_tensor((8,), Uint32)
    w_h2 = cute.make_rmem_tensor((8,), Uint32)

    # Load X (2 x 128-bit loads = 16 elements)
    x_ptr0 = get_ptr_as_int64(mX, row_offset * H + col_offset)
    x_ptr1 = get_ptr_as_int64(mX, row_offset * H + col_offset + Int32(8))
    x_h2[0], x_h2[1], x_h2[2], x_h2[3] = ld_global_v4_u32(x_ptr0)
    x_h2[4], x_h2[5], x_h2[6], x_h2[7] = ld_global_v4_u32(x_ptr1)

    # Load W (2 x 128-bit loads = 16 elements)
    w_ptr0 = get_ptr_as_int64(mW, col_offset)
    w_ptr1 = get_ptr_as_int64(mW, col_offset + Int32(8))
    w_h2[0], w_h2[1], w_h2[2], w_h2[3] = ld_global_v4_u32(w_ptr0)
    w_h2[4], w_h2[5], w_h2[6], w_h2[7] = ld_global_v4_u32(w_ptr1)

    return x_h2, w_h2


@cute.jit
def half2_mul_8(x_h2: cute.Tensor, w_h2: cute.Tensor) -> cute.Tensor:
    """Multiply 8 half2 pairs element-wise."""
    xw_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        xw_h2[i] = half2_mul(x_h2[i], w_h2[i])
    return xw_h2


@cute.jit
def bfloat2_mul_8(x_h2: cute.Tensor, w_h2: cute.Tensor) -> cute.Tensor:
    """Multiply 8 bfloat2 pairs element-wise."""
    xw_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        xw_h2[i] = bfloat2_mul(x_h2[i], w_h2[i])
    return xw_h2


@cute.jit
def half2_max_abs_8(xw_h2: cute.Tensor) -> Uint32:
    """Compute max absolute value across 8 half2 values using tree reduction."""
    # Compute abs for all 8 values
    abs_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_h2[i] = habs2(xw_h2[i])

    # Tree reduction: 8 -> 4 -> 2 -> 1
    max_01 = hmax2(abs_h2[0], abs_h2[1])
    max_23 = hmax2(abs_h2[2], abs_h2[3])
    max_45 = hmax2(abs_h2[4], abs_h2[5])
    max_67 = hmax2(abs_h2[6], abs_h2[7])
    max_0123 = hmax2(max_01, max_23)
    max_4567 = hmax2(max_45, max_67)
    return hmax2(max_0123, max_4567)


@cute.jit
def bfloat2_max_abs_8(xw_h2: cute.Tensor) -> Uint32:
    """Compute max absolute value across 8 bfloat2 values using tree reduction."""
    # Compute abs for all 8 values
    abs_h2 = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        abs_h2[i] = bfloat2_habs2(xw_h2[i])

    # Tree reduction: 8 -> 4 -> 2 -> 1
    max_01 = bfloat2_hmax2(abs_h2[0], abs_h2[1])
    max_23 = bfloat2_hmax2(abs_h2[2], abs_h2[3])
    max_45 = bfloat2_hmax2(abs_h2[4], abs_h2[5])
    max_67 = bfloat2_hmax2(abs_h2[6], abs_h2[7])
    max_0123 = bfloat2_hmax2(max_01, max_23)
    max_4567 = bfloat2_hmax2(max_45, max_67)
    return bfloat2_hmax2(max_0123, max_4567)


@cute.jit
def half2_to_float16(xw_h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 half2 to 16 float32 with scaling."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y_f32[i * 2], y_f32[i * 2 + 1] = half2_to_float2_scaled(xw_h2[i], scale)
    return y_f32


@cute.jit
def bfloat2_to_float16(xw_h2: cute.Tensor, scale: Float32) -> cute.Tensor:
    """Convert 8 bfloat2 to 16 float32 with scaling."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(8):
        y_f32[i * 2], y_f32[i * 2 + 1] = bfloat2_to_float2_scaled(xw_h2[i], scale)
    return y_f32


@cute.jit
def quantize_and_pack_16(y_f32: cute.Tensor, value_scale: Float32) -> Uint64:
    """Quantize 16 float32 values to FP4 and pack into uint64."""
    t025 = value_scale * Float32(0.25)
    t075 = value_scale * Float32(0.75)
    t125 = value_scale * Float32(1.25)
    t175 = value_scale * Float32(1.75)
    t250 = value_scale * Float32(2.5)
    t350 = value_scale * Float32(3.5)
    t500 = value_scale * Float32(5.0)
    packed = Uint64(0)
    for i in cutlass.range_constexpr(16):
        q = y_f32[i]
        mag = fabs_f32(q)
        nibble = Uint8(0)
        if mag > t025 and mag < t075:
            nibble = Uint8(1)
        elif mag >= t075 and mag <= t125:
            nibble = Uint8(2)
        elif mag > t125 and mag < t175:
            nibble = Uint8(3)
        elif mag >= t175 and mag <= t250:
            nibble = Uint8(4)
        elif mag > t250 and mag < t350:
            nibble = Uint8(5)
        elif mag >= t350 and mag <= t500:
            nibble = Uint8(6)
        elif mag > t500:
            nibble = Uint8(7)
        if nibble != Uint8(0) and q < Float32(0.0):
            nibble = nibble | Uint8(0x8)
        packed = packed | (Uint64(nibble) << Uint64(i * 4))
    return packed


@cute.jit
def quantize_and_pack_16_fast(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Fast approximate FP4 quantize/pack for 16 float32 values."""
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = y_f32[i] * inv_scale

    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def quantize_and_pack_16_div_rn(y_f32: cute.Tensor, scale: Float32) -> Uint64:
    """FP4 quantize/pack of 16 float32 values by IEEE division, ``e2m1(x / scale)``.

    Bit-exact with DeepSeek-V4.1's reference ``fp4_quant_kernel``: each value is
    divided (``div.rn.f32``, not multiplied by a rounded reciprocal, which moves
    exact E2M1 midpoints off their ties) and converted with
    ``cvt.rn.satfinite.e2m1x2`` (round-to-nearest-even, saturating at +-6 like
    the reference's clamp).
    """
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = div_rn_f32(y_f32[i], scale)

    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


@cute.jit
def quantize_block_fp4(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Quantize 16 float32 values to packed FP4 + e4m3 scale byte.

    Given 16 values and their pre-computed max_abs, derives the NVFP4 block
    scale, quantizes to FP4, and packs into a uint64.  Returns
    (packed_fp4_u64, scale_byte).  The caller handles storage writes.
    """
    scale_float = max_abs * global_scale_val / Float32(FLOAT4_E2M1_MAX)
    scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
    scale_u32 = cvt_f32_to_e4m3(scale_float)
    scale_byte = cutlass.Uint8(scale_u32 & Uint32(0xFF))
    quantized_scale = fp8_e4m3_to_f32(scale_u32)
    packed64 = Uint64(0)
    if quantized_scale != Float32(0.0) and global_scale_val != Float32(0.0):
        packed64 = quantize_and_pack_16(values, quantized_scale / global_scale_val)
    return packed64, scale_byte


@cute.jit
def quantize_block_fp4_fast(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fast approximate FP4 block quantization using reciprocal/vector path."""
    scale_u32 = Uint32(0)
    scale_byte = cutlass.Uint8(0)
    packed64 = Uint64(0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))
        scale_float = global_scale_val * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = cutlass.Uint8(scale_u32 & Uint32(0xFF))
        inv_quantized_scale = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv_quantized_scale != Float32(0.0):
            packed64 = quantize_and_pack_16_fast(
                values, inv_quantized_scale * global_scale_val
            )
    return packed64, scale_byte


@cute.jit
def max_abs_16(values: cute.Tensor) -> Float32:
    """Compute the maximum absolute value of 16 float32 values."""
    result = fabs_f32(values[0])
    for i in cutlass.range_constexpr(1, 16):
        result = fmax_f32(result, fabs_f32(values[i]))
    return result


@cute.jit
def silu_mul_16(
    gate: cute.Tensor,
    up: cute.Tensor,
) -> cute.Tensor:
    """Fused SiLU(gate) * up for 16 float32 element pairs.

    Used in the CompactMoEKernel epilogue to fuse the activation function
    between GEMM1 and GEMM2, avoiding the gmem round-trip.
    """
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        g = gate[i]
        sigmoid_g = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-g, fastmath=False)
        )
        out[i] = g * sigmoid_g * up[i]
    return out


@cute.jit
def silu_mul_quantize_block_fp4(
    gate: cute.Tensor,
    up: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fused SiLU(gate)*up + FP4 quantize for 16 element pairs.

    Combines silu_mul_16, max_abs_16, and quantize_block_fp4 into a single
    helper for the CompactMoEKernel epilogue.
    """
    activated = silu_mul_16(gate, up)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


@cute.jit
def relu2_16(x: cute.Tensor) -> cute.Tensor:
    """Compute ReLU^2 for 16 float32 values."""
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        v = fmax_f32(x[i], Float32(0.0))
        out[i] = v * v
    return out


@cute.jit
def relu2_quantize_block_fp4(
    x: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, cutlass.Uint8]:
    """Fuse ReLU^2 and FP4 quantization for 16 float32 values."""
    activated = relu2_16(x)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


# =============================================================================
# MXFP8 (w4a8) Block Quantization: 32-wide UE8M0 + E4M3
# =============================================================================


@cute.jit
def max_abs_32(values: cute.Tensor) -> Float32:
    """Compute the maximum absolute value of 32 float32 values."""
    result = fabs_f32(values[0])
    for i in cutlass.range_constexpr(1, 32):
        result = fmax_f32(result, fabs_f32(values[i]))
    return result


@cute.jit
def quantize_block_fp8_mx(
    values: cute.Tensor,
    max_abs: Float32,
) -> Tuple[cute.Tensor, Uint32]:
    """Quantize 32 float32 values to E4M3 payload bytes + UE8M0 scale byte.

    The scale is ``pow2_ceil(max_abs / 448)`` so the scaled payload never
    exceeds the E4M3 range; the payload carries its own exponent, so the
    power-of-two block scale handles range only.  No global scale is
    involved (the MX scheme is self-ranging).  Returns
    ``(payload_u32x8, ue8m0_byte)`` with value ``4*i + j`` in byte ``j`` of
    ``payload[i]``.  An all-zero block yields scale byte 0 (decoded as a
    zero output scale) and a zero payload.
    """
    scale_f = max_abs * Float32(1.0 / FLOAT8_E4M3_MAX)
    _, scale_byte = pow2_ceil_ue8m0(scale_f)
    inv_scale = ue8m0_to_output_scale(scale_byte)
    payload = cute.make_rmem_tensor((8,), Uint32)
    for i in cutlass.range_constexpr(8):
        payload[i] = cvt_f32x4_to_e4m3x4(
            values[i * 4 + 0] * inv_scale,
            values[i * 4 + 1] * inv_scale,
            values[i * 4 + 2] * inv_scale,
            values[i * 4 + 3] * inv_scale,
        )
    return payload, scale_byte


@cute.jit
def silu_mul_32(
    gate: cute.Tensor,
    up: cute.Tensor,
) -> cute.Tensor:
    """Fused SiLU(gate) * up for 32 float32 element pairs."""
    out = cute.make_rmem_tensor((32,), Float32)
    for i in cutlass.range_constexpr(32):
        g = gate[i]
        sigmoid_g = cute.arch.rcp_approx(
            Float32(1.0) + cute.math.exp(-g, fastmath=False)
        )
        out[i] = g * sigmoid_g * up[i]
    return out


@cute.jit
def silu_mul_quantize_block_fp8_mx(
    gate: cute.Tensor,
    up: cute.Tensor,
) -> Tuple[cute.Tensor, Uint32]:
    """Fused SiLU(gate)*up + MXFP8 quantize for 32 element pairs."""
    activated = silu_mul_32(gate, up)
    block_max = max_abs_32(activated)
    return quantize_block_fp8_mx(activated, block_max)


@cute.jit
def relu2_32(x: cute.Tensor) -> cute.Tensor:
    """Compute ReLU^2 for 32 float32 values."""
    out = cute.make_rmem_tensor((32,), Float32)
    for i in cutlass.range_constexpr(32):
        v = fmax_f32(x[i], Float32(0.0))
        out[i] = v * v
    return out


@cute.jit
def relu2_quantize_block_fp8_mx(
    x: cute.Tensor,
) -> Tuple[cute.Tensor, Uint32]:
    """Fuse ReLU^2 and MXFP8 quantization for 32 float32 values."""
    activated = relu2_32(x)
    block_max = max_abs_32(activated)
    return quantize_block_fp8_mx(activated, block_max)


@cute.jit
def mx_scale_from_amax32(amax: Float32) -> Tuple[Float32, Float32]:
    """Per-32 MXFP8 block scale from a precomputed amax.

    Returns ``(scale, inv_scale)`` with ``scale = pow2_ceil(amax/448)`` —
    identical numerics to ``quantize_block_fp8_mx`` (zero amax yields
    (0, 0), silencing the block).
    """
    rounded, byte = pow2_ceil_ue8m0(amax * Float32(1.0 / FLOAT8_E4M3_MAX))
    inv_scale = ue8m0_to_output_scale(byte)
    return rounded, inv_scale


@cute.jit
def quant_dequant_e4m3_2(
    v0: Float32,
    v1: Float32,
    inv_scale: Float32,
    scale: Float32,
) -> Tuple[Float32, Float32]:
    """Quantize-dequantize a pair through E4M3 with a power-of-two block scale.

    Bit-matches the w4a8 prefill activation quantizer
    (``quantize_block_fp8_mx``): RN-saturating E4M3 of ``v*inv_scale``,
    decoded back and rescaled.  Used by the micro decode kernel's a8_mx mode
    so decode numerics track the prefill recipe.
    """
    q0 = fp8_e4m3_to_f32(cvt_f32_to_e4m3(v0 * inv_scale)) * scale
    q1 = fp8_e4m3_to_f32(cvt_f32_to_e4m3(v1 * inv_scale)) * scale
    return q0, q1


# =============================================================================
# Helper Functions for Float32 SF Block Processing
# =============================================================================


@cute.jit
def load_f32_16_from_smem(
    sH: cute.Tensor, row_idx: Int32, col_offset: Int32
) -> cute.Tensor:
    """Load 16 Float32 values from shared memory."""
    h_f32 = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        h_f32[i] = Float32(sH[row_idx, col_offset + i])
    return h_f32


@cute.jit
def compute_y_and_max_abs_f32(
    h_f32: cute.Tensor, w_f32: cute.Tensor, rstd: Float32
) -> Tuple[cute.Tensor, Float32]:
    """Compute y = h * rstd * w and max_abs for 16 Float32 values."""
    y_f32 = cute.make_rmem_tensor((16,), Float32)

    # Compute y and track max_abs
    y_f32[0] = h_f32[0] * rstd * w_f32[0]
    max_abs = fabs_f32(y_f32[0])

    for i in cutlass.range_constexpr(1, 16):
        y_f32[i] = h_f32[i] * rstd * w_f32[i]
        max_abs = fmax_f32(max_abs, fabs_f32(y_f32[i]))

    return y_f32, max_abs
