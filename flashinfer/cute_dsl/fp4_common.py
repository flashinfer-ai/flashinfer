"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Common utilities for FP4 quantization kernels using CuTe-DSL.

This module contains shared PTX intrinsics, helper functions, and reduction
utilities used by both rmsnorm_fp4quant.py and add_rmsnorm_fp4quant.py.
"""

import functools
import math
import operator
from typing import Callable, Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
SF_VEC_SIZE = 16  # Elements per scale factor block
COPY_BITS = 128  # 128-bit vectorized loads


# =============================================================================
# Architecture Detection
# =============================================================================


@functools.lru_cache(maxsize=16)
def get_sm_version(device: int | torch.device | str | None = None) -> int:
    """Get the SM version of a CUDA device.

    Args:
        device: CUDA device to query. Can be an int (device index), torch.device,
            device string (e.g., 'cuda:0'), or None to use current device.

    Returns:
        SM version as an integer (e.g., 100 for SM100).
    """
    if not torch.cuda.is_available():
        return 80
    if device is None:
        device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return props.major * 10 + props.minor


# =============================================================================
# PTX Intrinsics - Cluster Operations
# =============================================================================


@dsl_user_op
def set_block_rank(
    smem_ptr: cute.Pointer, peer_cta_rank_in_cluster: Int32, *, loc=None, ip=None
) -> Int32:
    """Map smem pointer to address at another CTA rank in the cluster."""
    smem_ptr_i32 = smem_ptr.toint(loc=loc, ip=ip).ir_value()
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [smem_ptr_i32, peer_cta_rank_in_cluster.ir_value()],
            "mapa.shared::cluster.u32 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def store_shared_remote(
    val: Float32,
    smem_ptr: cute.Pointer,
    mbar_ptr: cute.Pointer,
    peer_cta_rank_in_cluster: Int32,
    *,
    loc=None,
    ip=None,
) -> None:
    """Store Float32 value to shared memory on a remote CTA in the cluster."""
    remote_smem_ptr_i32 = set_block_rank(
        smem_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    remote_mbar_ptr_i32 = set_block_rank(
        mbar_ptr, peer_cta_rank_in_cluster, loc=loc, ip=ip
    ).ir_value()
    llvm.inline_asm(
        None,
        [remote_smem_ptr_i32, val.ir_value(loc=loc, ip=ip), remote_mbar_ptr_i32],
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.f32 [$0], $1, [$2];",
        "r,f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def elem_pointer(x: cute.Tensor, coord, *, loc=None, ip=None) -> cute.Pointer:
    """Get pointer to element at coordinate in tensor."""
    return x.iterator + cute.crd2idx(coord, x.layout, loc=loc, ip=ip)


# =============================================================================
# PTX Intrinsics - 128-bit Vectorized Global Loads/Stores
# =============================================================================


@dsl_user_op
def ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    v2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    v3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)

    return Uint32(v0), Uint32(v1), Uint32(v2), Uint32(v3)


@dsl_user_op
def st_global_u64(base_ptr: Int64, value: Uint64, *, loc=None, ip=None):
    """Store 64 bits to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint64(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u64 [$0], $1;",
        "l,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


# =============================================================================
# PTX Intrinsics - Math Operations
# =============================================================================


@dsl_user_op
def rcp_approx_ftz(a: Float32, *, loc=None, ip=None) -> Float32:
    """Fast reciprocal using PTX rcp.approx.ftz.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "rcp.approx.ftz.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values using PTX min.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "min.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values using PTX max.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "max.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fabs_f32(a: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of float32 using PTX abs.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Half2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def half2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two Half2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hadd2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two Half2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 absolute value - clears sign bits of both fp16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 max - element-wise max of 2 fp16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.f16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 fp16 values in half2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 h0, h1;
                .reg .f32 f0, f1;
                mov.b32 {h0, h1}, $1;
                cvt.f32.f16 f0, h0;
                cvt.f32.f16 f1, h1;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert half2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            .reg .f32 f0, f1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 f0, h0;
            cvt.f32.f16 f1, h1;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


# =============================================================================
# BFloat2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise: (a.x*b.x, a.y*b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "mul.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_add(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two BFloat2 values element-wise: (a.x+b.x, a.y+b.y)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "add.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 absolute value - clears sign bits of both bf16 values."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            "and.b32 $0, $1, 0x7FFF7FFF;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 max - element-wise max of 2 bf16 pairs."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(a).ir_value(loc=loc, ip=ip), Uint32(b).ir_value(loc=loc, ip=ip)],
            "max.bf16x2 $0, $1, $2;",
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_hmax_to_f32(x: Uint32, *, loc=None, ip=None) -> Float32:
    """Extract max of 2 bf16 values in bfloat2 as float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 lo, hi;
                .reg .f32 f0, f1;
                and.b32 lo, $1, 0xFFFF;
                shr.b32 hi, $1, 16;
                shl.b32 lo, lo, 16;
                shl.b32 hi, hi, 16;
                mov.b32 f0, lo;
                mov.b32 f1, hi;
                max.f32 $0, f0, f1;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 AND multiply by scale."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip), Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            .reg .f32 f0, f1;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 f0, lo;
            mov.b32 f1, hi;
            mul.f32 $0, f0, $3;
            mul.f32 $1, f1, $3;
        }
        """,
        "=f,=f,r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)

    return Float32(f0), Float32(f1)


# =============================================================================
# FP8 E4M3 Intrinsics
# =============================================================================


@dsl_user_op
def cvt_f32_to_e4m3(a: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to E4M3 using native cvt.rn.satfinite.e4m3x2.f32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .f32 zero;
                mov.f32 zero, 0f00000000;
                cvt.rn.satfinite.e4m3x2.f32 fp8_pair, zero, $1;
                cvt.u32.u16 $0, fp8_pair;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fp8_e4m3_to_f32_and_rcp(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32 AND compute reciprocal."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .u32 exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, result;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                rcp.approx.ftz.f32 result, fp8_float;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# UE8M0 Intrinsics (for MXFP4)
# =============================================================================


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """
    Convert float32 max value to UE8M0 scale factor.

    UE8M0 is unsigned 8-bit exponent-only format:
    - value = 2^(ue8m0 - 127)
    - ue8m0 = ceil(log2(max_val)) + 127

    Uses lg2.approx.f32 for fast log2 approximation.
    Uses cvt.rpi (round towards positive infinity, i.e., ceiling).
    Returns value clamped to [0, 255].
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;

                // Check for zero/negative
                setp.le.f32 p_zero, $1, 0f00000000;

                // Compute ceil(log2(max_val)) using cvt.rpi (round towards +inf)
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;

                // Add bias and clamp to [0, 255]
                add.s32 result, exp_int, 127;
                setp.lt.s32 p_neg, result, 0;
                setp.gt.s32 p_ovf, result, 255;
                selp.s32 result, 0, result, p_neg;
                selp.s32 result, 255, result, p_ovf;
                selp.s32 $0, 0, result, p_zero;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ue8m0_to_output_scale(ue8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """
    Convert UE8M0 to output_scale for MXFP4 quantization.

    UE8M0 value = 2^(ue8m0 - 127)
    Returns 1 / 2^(ue8m0 - 127) = 2^(127 - ue8m0)
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;

                // Check for zero
                setp.eq.u32 p_zero, $1, 0;

                // Compute 2^(127 - ue8m0) = 1 / 2^(ue8m0 - 127)
                sub.s32 neg_exp, 127, $1;
                cvt.rn.f32.s32 neg_exp_f, neg_exp;
                ex2.approx.f32 result, neg_exp_f;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# E2M1 Conversion
# =============================================================================


@dsl_user_op
def cvt_e2m1x8_f32(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    v4: Float32,
    v5: Float32,
    v6: Float32,
    v7: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert eight float32 values to eight E2M1 (4-bit) values packed into uint32."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
                Float32(v4).ir_value(loc=loc, ip=ip),
                Float32(v5).ir_value(loc=loc, ip=ip),
                Float32(v6).ir_value(loc=loc, ip=ip),
                Float32(v7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 byte0, byte1, byte2, byte3;
                cvt.rn.satfinite.e2m1x2.f32 byte0, $2, $1;
                cvt.rn.satfinite.e2m1x2.f32 byte1, $4, $3;
                cvt.rn.satfinite.e2m1x2.f32 byte2, $6, $5;
                cvt.rn.satfinite.e2m1x2.f32 byte3, $8, $7;
                mov.b32 $0, {byte0, byte1, byte2, byte3};
            }
            """,
            "=r,f,f,f,f,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


# =============================================================================
# Warp, Block, and Cluster Reduction Utilities
# =============================================================================


@cute.jit
def warp_reduce(val, op, width: cutlass.Constexpr[int] = 32):
    """Reduce across threads in a warp using butterfly shuffle."""
    if cutlass.const_expr(isinstance(val, cute.TensorSSA)):
        res = cute.make_rmem_tensor(val.shape, val.dtype)
        res.store(val)
        for i in cutlass.range_constexpr(cute.size(val.shape)):
            res[i] = warp_reduce(res[i], op, width)
        return res.load()
    else:
        for i in cutlass.range_constexpr(int(math.log2(width))):
            val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1 << i))
        return val


@cute.jit
def block_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    init_val: Float32,
) -> Float32:
    """Block reduction across multiple warps using shared memory."""
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()
    warps_per_row = cute.size(reduction_buffer.shape[1])
    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    if lane_idx == 0:
        reduction_buffer[row_idx, col_idx] = val
    cute.arch.barrier()

    block_reduce_val = init_val
    if lane_idx < warps_per_row:
        block_reduce_val = reduction_buffer[row_idx, lane_idx]
    return warp_reduce(block_reduce_val, op)


@cute.jit
def cluster_reduce(
    val: Float32,
    op: Callable,
    reduction_buffer: cute.Tensor,
    mbar_ptr: cute.Pointer,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
) -> Float32:
    """Cluster reduction across multiple CTAs using mbarrier."""
    cta_rank_in_cluster = cute.arch.block_idx_in_cluster()
    lane_idx = cute.arch.lane_idx()
    warp_idx = cute.arch.warp_idx()

    rows_per_block = reduction_buffer.shape[0]
    warps_per_row = reduction_buffer.shape[1][0]

    row_idx = warp_idx // warps_per_row
    col_idx = warp_idx % warps_per_row

    # Warp 0 sets up mbarrier transaction count
    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    # Each lane < cluster_n writes to a different CTA's shared memory
    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    # Wait for all cluster writes
    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

    # Reduce across all values
    num_total = warps_per_row * cluster_n
    num_iter = cute.ceil_div(num_total, 32)

    block_reduce_val = init_val
    for i in cutlass.range_constexpr(num_iter):
        idx = lane_idx + i * 32
        if idx < num_total:
            block_reduce_val = op(block_reduce_val, reduction_buffer[row_idx, idx])

    return warp_reduce(block_reduce_val, op)


@cute.jit
def row_reduce(
    x: cute.TensorSSA,
    op: cute.ReductionOp,
    threads_per_row: cutlass.Constexpr[int],
    reduction_buffer: cute.Tensor,
    mbar_ptr,
    cluster_n: cutlass.Constexpr[int],
    init_val: Float32,
):
    """Row reduction with optional cluster support."""
    local_val = x.reduce(op, init_val=init_val, reduction_profile=0)

    warp_op = {
        cute.ReductionOp.ADD: operator.add,
        cute.ReductionOp.MAX: cute.arch.fmax,
    }[op]
    warp_width = min(threads_per_row, 32)
    warp_val = warp_reduce(local_val, warp_op, width=warp_width)

    warps_per_row = max(threads_per_row // 32, 1)

    if cutlass.const_expr(warps_per_row > 1 or cluster_n > 1):
        if cutlass.const_expr(cluster_n == 1):
            return block_reduce(warp_val, warp_op, reduction_buffer, init_val)
        else:
            return cluster_reduce(
                warp_val, warp_op, reduction_buffer, mbar_ptr, cluster_n, init_val
            )
    else:
        return warp_val


# =============================================================================
# Predicate Utility
# =============================================================================


@cute.jit
def predicate_k(tXcX: cute.Tensor, limit: int) -> cute.Tensor:
    """Create predicate tensor for bounds checking."""
    tXpX = cute.make_rmem_tensor(
        cute.make_layout(
            (
                cute.size(tXcX, mode=[0, 1]),
                cute.size(tXcX, mode=[1]),
                cute.size(tXcX, mode=[2]),
            ),
            stride=(cute.size(tXcX, mode=[2]), 0, 1),
        ),
        cutlass.Boolean,
    )
    for rest_v in cutlass.range_constexpr(tXpX.shape[0]):
        for rest_k in cutlass.range_constexpr(tXpX.shape[2]):
            tXpX[rest_v, 0, rest_k] = cute.elem_less(
                tXcX[(0, rest_v), 0, rest_k][1], limit
            )
    return tXpX


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
def quantize_and_pack_16(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Quantize 16 float32 values to FP4 and pack into uint64."""
    # Scale values
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = y_f32[i] * inv_scale

    # Convert to E2M1 and pack
    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


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
