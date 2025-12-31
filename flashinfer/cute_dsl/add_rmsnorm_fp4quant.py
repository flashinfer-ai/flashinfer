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

Fused Add + RMSNorm + FP4 Quantization using CuTe-DSL
======================================================

High-performance fused kernel for element-wise addition followed by RMS normalization
and FP4 quantization. Supports both NVFP4 (block_size=16, E4M3 scales) and MXFP4
(block_size=32, UE8M0 scales) formats.

"""

import functools
import math
import operator
from typing import Callable, Tuple

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.torch as cutlass_torch
import torch
from cutlass import Float32, Int32, Int64, Uint32, Uint64, Uint8
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from .utils import make_ptr
from ..api_logging import flashinfer_api

# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0
FLOAT8_E4M3_MAX = 448.0
SF_VEC_SIZE = 16
COPY_BITS = 128


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
# PTX Intrinsics - Basic Operations
# =============================================================================


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
def fabs_f32(val: Float32, *, loc=None, ip=None) -> Float32:
    """Compute absolute value of float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
            "abs.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fmax_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max of two float32 values."""
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
def fmin_f32(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min of two float32 values (branchless clamping)."""
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


# =============================================================================
# FP8 E4M3 and UE8M0 Intrinsics
# =============================================================================


@dsl_user_op
def cvt_f32_to_e4m3(val: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 to FP8 E4M3."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(val).ir_value(loc=loc, ip=ip)],
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
    """Convert FP8 E4M3 to float32 and compute reciprocal."""
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


@dsl_user_op
def cvt_f32_to_ue8m0(max_val: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert float32 max value to UE8M0 scale factor."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(max_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg, p_ovf;
                .reg .f32 log2_val;
                .reg .s32 exp_int, result;
                setp.le.f32 p_zero, $1, 0f00000000;
                lg2.approx.f32 log2_val, $1;
                cvt.rpi.s32.f32 exp_int, log2_val;
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
    """Convert UE8M0 to output_scale (1 / 2^(ue8m0 - 127))."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(ue8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero;
                .reg .s32 neg_exp;
                .reg .f32 neg_exp_f, result;
                setp.eq.u32 p_zero, $1, 0;
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
# Half2 SIMD Intrinsics
# =============================================================================


@dsl_user_op
def half2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two Half2 values element-wise."""
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
def bfloat2_mul(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Multiply two BFloat2 values element-wise."""
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
def habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """Half2 absolute value."""
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
def bfloat2_habs2(x: Uint32, *, loc=None, ip=None) -> Uint32:
    """BFloat16x2 absolute value."""
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
    """BFloat16x2 max."""
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
    """Extract max of 2 bf16 values as float32."""
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
def half2_to_float2_scaled(
    h2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert half2 to float2 and multiply by scale."""
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


@dsl_user_op
def bfloat2_to_float2_scaled(
    bf2: Uint32, scale: Float32, *, loc=None, ip=None
) -> Tuple[Float32, Float32]:
    """Convert bfloat16x2 to float2 and multiply by scale."""
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


@dsl_user_op
def hadd2(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two Half2 values element-wise."""
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
def bfloat2_add(a: Uint32, b: Uint32, *, loc=None, ip=None) -> Uint32:
    """Add two BFloat2 values element-wise."""
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

    if warp_idx == 0:
        with cute.arch.elect_one():
            num_warps = rows_per_block * warps_per_row
            expected_bytes = num_warps * cluster_n * 4
            cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expected_bytes)

    if lane_idx < cluster_n:
        store_shared_remote(
            val,
            elem_pointer(reduction_buffer, (row_idx, (col_idx, cta_rank_in_cluster))),
            mbar_ptr,
            peer_cta_rank_in_cluster=lane_idx,
        )

    cute.arch.mbarrier_wait(mbar_ptr, phase=0)

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
# CuTe-DSL Kernel Class
# =============================================================================


class AddRMSNormFP4QuantKernel:
    """
    Fused Add + RMSNorm + FP4 Quantization Kernel.

    Computes: h = x + r, y = RMSNorm(h) * w, then quantizes y to FP4.
    Supports both NVFP4 (block_size=16) and MXFP4 (block_size=32) formats.
    """

    def __init__(
        self,
        dtype: cutlass.Numeric,
        H: int,
        block_size: int,
        output_swizzled: bool,
        is_fp16: bool,
        sm_version: int | None = None,
        scale_format: str | None = None,
    ):
        self.dtype = dtype
        self.H = H
        self.block_size = block_size
        self.output_swizzled = output_swizzled
        self.is_fp16 = is_fp16
        self.sm_version = sm_version if sm_version is not None else get_sm_version()

        if scale_format is None:
            self.scale_format = "ue8m0" if block_size == 32 else "e4m3"
        else:
            self.scale_format = scale_format

        assert block_size in (16, 32), f"block_size must be 16 or 32, got {block_size}"
        assert self.scale_format in ("e4m3", "ue8m0"), (
            "scale_format must be 'e4m3' or 'ue8m0'"
        )

        self.cluster_n = self._compute_cluster_n(H, dtype, self.sm_version)
        self.H_per_cta = H // self.cluster_n

        self.threads_per_row = self._compute_threads_per_row(self.H_per_cta)
        self.num_threads = self._compute_num_threads(self.H_per_cta)
        self.rows_per_block = self.num_threads // self.threads_per_row
        self.warps_per_row = max(self.threads_per_row // 32, 1)

        elem_bytes = dtype.width // 8
        self.vec_size = COPY_BITS // 8 // elem_bytes
        self.num_vec_blocks = max(
            1,
            (self.H_per_cta // self.vec_size + self.threads_per_row - 1)
            // self.threads_per_row,
        )
        self.cols_per_tile = self.vec_size * self.num_vec_blocks * self.threads_per_row

        self.num_sf_blocks_per_row = H // block_size

        if output_swizzled:
            num_col_vecs = H // block_size
            self.num_k_tiles = (num_col_vecs + 3) // 4
            self.k_tile_stride = 512

    @staticmethod
    def _compute_cluster_n(H: int, dtype: cutlass.Numeric, sm_version: int) -> int:
        """Compute optimal cluster size based on H and device shared memory.

        Dynamically determines the minimum cluster_n that fits within the
        device's shared memory limit, making it compatible with different
        GPU architectures (e.g., SM100 with 228KB vs SM120 with 128KB).
        """
        if sm_version < 90:
            return 1

        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        max_smem_bytes = props.shared_memory_per_block_optin
        elem_size = dtype.width // 8

        for cluster_n in [1, 2, 4, 8, 16]:
            if H % cluster_n != 0:
                continue
            smem_needed = AddRMSNormFP4QuantKernel._estimate_smem_bytes(
                H, cluster_n, elem_size
            )
            if smem_needed <= max_smem_bytes:
                return cluster_n

        return 16

    @staticmethod
    def _compute_threads_per_row(H_per_cta: int) -> int:
        """Compute optimal threads per row."""
        if H_per_cta <= 64:
            return 8
        elif H_per_cta <= 128:
            return 16
        elif H_per_cta <= 3072:
            return 32
        elif H_per_cta <= 6144:
            return 64
        elif H_per_cta <= 16384:
            return 128
        else:
            return 256

    @staticmethod
    def _compute_num_threads(H_per_cta: int) -> int:
        """Compute total threads per block."""
        return 128 if H_per_cta <= 16384 else 256

    @staticmethod
    def _estimate_smem_bytes(H: int, cluster_n: int, elem_size: int) -> int:
        """Estimate shared memory bytes needed for given configuration.

        This is used to dynamically determine cluster_n based on device
        shared memory limits.
        """
        H_per_cta = H // cluster_n
        threads_per_row = AddRMSNormFP4QuantKernel._compute_threads_per_row(H_per_cta)
        num_threads = AddRMSNormFP4QuantKernel._compute_num_threads(H_per_cta)
        rows_per_block = num_threads // threads_per_row
        warps_per_row = max(threads_per_row // 32, 1)

        vec_size = COPY_BITS // 8 // elem_size
        num_vec_blocks = max(
            1, (H_per_cta // vec_size + threads_per_row - 1) // threads_per_row
        )
        cols_per_tile = vec_size * num_vec_blocks * threads_per_row

        tile_bytes = rows_per_block * cols_per_tile * elem_size

        if cluster_n == 1:
            # 4 tiles: sX, sR, sW, sH + reduction buffer
            return 4 * tile_bytes + rows_per_block * warps_per_row * 4
        else:
            # 2 tiles: sX, sR + larger reduction buffer + mbarrier
            return 2 * tile_bytes + rows_per_block * warps_per_row * cluster_n * 4 + 8

    @staticmethod
    def _make_tv_layout(
        threads_per_row: int,
        rows_per_block: int,
        vec_size: int,
        num_vec_blocks: int,
    ) -> tuple:
        """Create Thread-Value layout for coalesced vectorized memory access."""
        shape = (
            (threads_per_row, rows_per_block),
            (vec_size, num_vec_blocks),
        )
        stride = (
            (vec_size * rows_per_block, 1),
            (rows_per_block, rows_per_block * vec_size * threads_per_row),
        )
        return shape, stride

    def _smem_size_in_bytes(self) -> int:
        """Calculate shared memory requirement."""
        elem_size = self.dtype.width // 8
        x_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
        r_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size

        if self.cluster_n == 1:
            w_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
            h_tile_bytes = self.rows_per_block * self.cols_per_tile * elem_size
            reduction_bytes = self.rows_per_block * self.warps_per_row * 4
        else:
            w_tile_bytes = 0
            h_tile_bytes = 0
            reduction_bytes = (
                self.rows_per_block * self.warps_per_row * self.cluster_n * 4
            )

        mbar_bytes = 8 if self.cluster_n > 1 else 0

        return (
            x_tile_bytes
            + r_tile_bytes
            + w_tile_bytes
            + h_tile_bytes
            + reduction_bytes
            + mbar_bytes
        )

    @cute.jit
    def __call__(
        self,
        x_ptr: cute.Pointer,
        r_ptr: cute.Pointer,
        w_ptr: cute.Pointer,
        y_ptr: cute.Pointer,
        s_ptr: cute.Pointer,
        M: Int32,
        eps: Float32,
        stream: cuda.CUstream,
    ):
        """Host function to launch the kernel."""
        H = self.H

        mX = cute.make_tensor(
            x_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mR = cute.make_tensor(
            r_ptr,
            layout=cute.make_ordered_layout((M, H), order=(1, 0)),
        )
        mW = cute.make_tensor(
            w_ptr,
            layout=cute.make_layout((H,)),
        )
        mY = cute.make_tensor(
            y_ptr,
            layout=cute.make_ordered_layout((M, H // 2), order=(1, 0)),
        )

        # Create mS tensor with appropriate layout based on swizzle mode
        if cutlass.const_expr(self.output_swizzled):
            # For swizzled output, use 1D layout
            # The swizzle writes use flat offsets: mS[swizzled_offset]
            # We compute the swizzled offset in the kernel, so just use a 1D layout
            # with stride 1 to treat the pointer as a flat array
            num_m_tiles = (M + Int32(127)) // Int32(128)
            swizzled_size = num_m_tiles * Int32(self.num_k_tiles * self.k_tile_stride)
            mS = cute.make_tensor(
                s_ptr,
                layout=cute.make_layout((swizzled_size,), stride=(Int32(1),)),
            )
        else:
            # For non-swizzled output, use 2D row-major layout
            mS = cute.make_tensor(
                s_ptr,
                layout=cute.make_ordered_layout(
                    (M, self.num_sf_blocks_per_row), order=(1, 0)
                ),
            )

        tv_shape, tv_stride = self._make_tv_layout(
            self.threads_per_row,
            self.rows_per_block,
            self.vec_size,
            self.num_vec_blocks,
        )
        tv_layout = cute.make_layout(tv_shape, stride=tv_stride)
        tiler_mn = (self.rows_per_block, self.cols_per_tile)

        self.kernel(mX, mR, mW, mY, mS, M, eps, tv_layout, tiler_mn).launch(
            grid=[cute.ceil_div(M, self.rows_per_block), self.cluster_n, 1],
            block=[self.num_threads, 1, 1],
            cluster=[1, self.cluster_n, 1]
            if cutlass.const_expr(self.cluster_n > 1)
            else None,
            smem=self._smem_size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mR: cute.Tensor,
        mW: cute.Tensor,
        mY: cute.Tensor,
        mS: cute.Tensor,
        M: Int32,
        eps: Float32,
        tv_layout: cute.Layout,
        tiler_mn: cute.Shape,
    ):
        """Device kernel with cluster sync and Half2 SIMD."""
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()

        H = self.H
        block_size = self.block_size
        num_sf_blocks_per_row = self.num_sf_blocks_per_row
        is_fp16 = self.is_fp16
        cluster_n = self.cluster_n

        if cutlass.const_expr(cluster_n > 1):
            cluster_y = cute.arch.block_idx()[1]
        else:
            cluster_y = cutlass.const_expr(0)

        threads_per_row = tv_layout.shape[0][0]
        warps_per_row = max(threads_per_row // 32, 1)
        rows_per_block = tiler_mn[0]

        lane_in_row = tidx % threads_per_row
        row_in_block = tidx // threads_per_row

        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))

        # Allocate shared memory
        smem = cutlass.utils.SmemAllocator()

        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        sR = smem.allocate_tensor(
            mR.element_type,
            cute.make_ordered_layout(tiler_mn, order=(1, 0)),
            byte_alignment=16,
        )

        if cutlass.const_expr(cluster_n == 1):
            sW = smem.allocate_tensor(
                mW.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )
            sH = smem.allocate_tensor(
                mX.element_type,
                cute.make_ordered_layout(tiler_mn, order=(1, 0)),
                byte_alignment=16,
            )

        if cutlass.const_expr(cluster_n == 1):
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, warps_per_row)),
                byte_alignment=4,
            )
            mbar_ptr = None
        else:
            reduction_buffer = smem.allocate_tensor(
                Float32,
                cute.make_layout((rows_per_block, (warps_per_row, cluster_n))),
                byte_alignment=4,
            )
            mbar_ptr = smem.allocate_array(Int64, num_elems=1)

        # Initialize cluster
        if cutlass.const_expr(cluster_n > 1):
            if tidx == 0:
                cute.arch.mbarrier_init(mbar_ptr, 1)
            cute.arch.mbarrier_init_fence()
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()

        # Create identity tensor
        idX = cute.make_identity_tensor(mX.shape)

        gX = cute.local_tile(mX, tiler_mn, (bidx, cluster_y))
        gR = cute.local_tile(mR, tiler_mn, (bidx, cluster_y))
        cX = cute.local_tile(idX, tiler_mn, (bidx, cluster_y))

        if cutlass.const_expr(cluster_n == 1):
            mW_expanded_layout = cute.prepend(
                mW.layout, cute.make_layout((tiler_mn[0],), stride=(0,))
            )
            mW_2d = cute.make_tensor(mW.iterator, mW_expanded_layout)
            gW = cute.local_tile(mW_2d, tiler_mn, (0, cluster_y))

        # TiledCopy
        copy_atom_load_async = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(),
            mX.element_type,
            num_bits_per_copy=COPY_BITS,
        )

        tiled_copy_load = cute.make_tiled_copy(
            copy_atom_load_async, tv_layout, tiler_mn
        )

        thr_copy_X = tiled_copy_load.get_slice(tidx)
        thr_copy_R = tiled_copy_load.get_slice(tidx)

        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tRgR = thr_copy_R.partition_S(gR)
        tRsR = thr_copy_R.partition_D(sR)
        tXcX = thr_copy_X.partition_S(cX)

        if cutlass.const_expr(cluster_n == 1):
            thr_copy_W = tiled_copy_load.get_slice(tidx)
            tWgW = thr_copy_W.partition_S(gW)
            tWsW = thr_copy_W.partition_D(sW)
            tHsH = thr_copy_X.partition_D(sH)

        tXrX = cute.make_fragment_like(tXgX)
        tRrR = cute.make_fragment_like(tRgR)

        # Bounds checking
        tXpX = predicate_k(tXcX, limit=H)
        row_coord = tXcX[(0, 0), 0, 0]
        row_in_bounds = row_coord[0] < M

        # Phase 1: Async copy
        if row_in_bounds:
            cute.copy(copy_atom_load_async, tXgX, tXsX, pred=tXpX)
            cute.copy(copy_atom_load_async, tRgR, tRsR, pred=tXpX)

        if cutlass.const_expr(cluster_n == 1):
            cute.copy(copy_atom_load_async, tWgW, tWsW, pred=tXpX)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)

        # Phase 2: h = x + r and sum of squares
        cute.autovec_copy(tXsX, tXrX)
        cute.autovec_copy(tRsR, tRrR)

        x_vals = tXrX.load().to(Float32)
        r_vals = tRrR.load().to(Float32)

        h_vals = x_vals + r_vals
        h_sq = h_vals * h_vals

        if cutlass.const_expr(cluster_n == 1):
            h_elem = h_vals.to(mX.element_type)
            tHsH.store(h_elem)

        sum_sq = row_reduce(
            h_sq,
            cute.ReductionOp.ADD,
            threads_per_row,
            reduction_buffer,
            mbar_ptr,
            cluster_n,
            Float32(0.0),
        )

        mean_sq = sum_sq / H
        rstd = cute.math.rsqrt(mean_sq + eps, fastmath=True)

        if cutlass.const_expr(cluster_n > 1):
            cute.arch.cluster_arrive_relaxed()
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        actual_row_idx = bidx * rows_per_block + row_in_block

        # Phase 3: RMSNorm + Quantize
        if actual_row_idx < M:
            num_sf_per_thread = (
                num_sf_blocks_per_row + threads_per_row - 1
            ) // threads_per_row

            for sf_iter in range(num_sf_per_thread):
                sf_idx = lane_in_row + sf_iter * threads_per_row

                if sf_idx < num_sf_blocks_per_row:
                    block_start = sf_idx * block_size

                    if cutlass.const_expr(block_size == 16):
                        if cutlass.const_expr(cluster_n == 1):
                            # Shared memory path
                            sh0 = Float32(sH[row_in_block, block_start + 0])
                            sh1 = Float32(sH[row_in_block, block_start + 1])
                            sh2 = Float32(sH[row_in_block, block_start + 2])
                            sh3 = Float32(sH[row_in_block, block_start + 3])
                            sh4 = Float32(sH[row_in_block, block_start + 4])
                            sh5 = Float32(sH[row_in_block, block_start + 5])
                            sh6 = Float32(sH[row_in_block, block_start + 6])
                            sh7 = Float32(sH[row_in_block, block_start + 7])
                            sh8 = Float32(sH[row_in_block, block_start + 8])
                            sh9 = Float32(sH[row_in_block, block_start + 9])
                            sh10 = Float32(sH[row_in_block, block_start + 10])
                            sh11 = Float32(sH[row_in_block, block_start + 11])
                            sh12 = Float32(sH[row_in_block, block_start + 12])
                            sh13 = Float32(sH[row_in_block, block_start + 13])
                            sh14 = Float32(sH[row_in_block, block_start + 14])
                            sh15 = Float32(sH[row_in_block, block_start + 15])

                            sw0 = Float32(sW[row_in_block, block_start + 0])
                            sw1 = Float32(sW[row_in_block, block_start + 1])
                            sw2 = Float32(sW[row_in_block, block_start + 2])
                            sw3 = Float32(sW[row_in_block, block_start + 3])
                            sw4 = Float32(sW[row_in_block, block_start + 4])
                            sw5 = Float32(sW[row_in_block, block_start + 5])
                            sw6 = Float32(sW[row_in_block, block_start + 6])
                            sw7 = Float32(sW[row_in_block, block_start + 7])
                            sw8 = Float32(sW[row_in_block, block_start + 8])
                            sw9 = Float32(sW[row_in_block, block_start + 9])
                            sw10 = Float32(sW[row_in_block, block_start + 10])
                            sw11 = Float32(sW[row_in_block, block_start + 11])
                            sw12 = Float32(sW[row_in_block, block_start + 12])
                            sw13 = Float32(sW[row_in_block, block_start + 13])
                            sw14 = Float32(sW[row_in_block, block_start + 14])
                            sw15 = Float32(sW[row_in_block, block_start + 15])

                            y0 = sh0 * rstd * sw0
                            y1 = sh1 * rstd * sw1
                            y2 = sh2 * rstd * sw2
                            y3 = sh3 * rstd * sw3
                            y4 = sh4 * rstd * sw4
                            y5 = sh5 * rstd * sw5
                            y6 = sh6 * rstd * sw6
                            y7 = sh7 * rstd * sw7
                            y8 = sh8 * rstd * sw8
                            y9 = sh9 * rstd * sw9
                            y10 = sh10 * rstd * sw10
                            y11 = sh11 * rstd * sw11
                            y12 = sh12 * rstd * sw12
                            y13 = sh13 * rstd * sw13
                            y14 = sh14 * rstd * sw14
                            y15 = sh15 * rstd * sw15

                            max_abs = fabs_f32(y0)
                            max_abs = fmax_f32(max_abs, fabs_f32(y1))
                            max_abs = fmax_f32(max_abs, fabs_f32(y2))
                            max_abs = fmax_f32(max_abs, fabs_f32(y3))
                            max_abs = fmax_f32(max_abs, fabs_f32(y4))
                            max_abs = fmax_f32(max_abs, fabs_f32(y5))
                            max_abs = fmax_f32(max_abs, fabs_f32(y6))
                            max_abs = fmax_f32(max_abs, fabs_f32(y7))
                            max_abs = fmax_f32(max_abs, fabs_f32(y8))
                            max_abs = fmax_f32(max_abs, fabs_f32(y9))
                            max_abs = fmax_f32(max_abs, fabs_f32(y10))
                            max_abs = fmax_f32(max_abs, fabs_f32(y11))
                            max_abs = fmax_f32(max_abs, fabs_f32(y12))
                            max_abs = fmax_f32(max_abs, fabs_f32(y13))
                            max_abs = fmax_f32(max_abs, fabs_f32(y14))
                            max_abs = fmax_f32(max_abs, fabs_f32(y15))

                            scale_float = max_abs * fp4_max_rcp
                            scale_float = fmin_f32(
                                scale_float, Float32(FLOAT8_E4M3_MAX)
                            )
                            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                            scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                            inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                            if cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_fp8

                            q0 = y0 * inv_scale
                            q1 = y1 * inv_scale
                            q2 = y2 * inv_scale
                            q3 = y3 * inv_scale
                            q4 = y4 * inv_scale
                            q5 = y5 * inv_scale
                            q6 = y6 * inv_scale
                            q7 = y7 * inv_scale
                            q8 = y8 * inv_scale
                            q9 = y9 * inv_scale
                            q10 = y10 * inv_scale
                            q11 = y11 * inv_scale
                            q12 = y12 * inv_scale
                            q13 = y13 * inv_scale
                            q14 = y14 * inv_scale
                            q15 = y15 * inv_scale

                            packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                            packed_hi = cvt_e2m1x8_f32(
                                q8, q9, q10, q11, q12, q13, q14, q15
                            )
                            packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(
                                packed_lo
                            )

                            out_offset = block_start // 2
                            out_ptr = get_ptr_as_int64(
                                mY, actual_row_idx * (H // 2) + out_offset
                            )
                            st_global_u64(out_ptr, packed64)

                        else:
                            # Global memory path (cluster mode)
                            h_ptr0 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start
                            )
                            h_ptr1 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start + Int32(8)
                            )
                            r_ptr0 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start
                            )
                            r_ptr1 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start + Int32(8)
                            )
                            w_ptr0 = get_ptr_as_int64(mW, block_start)
                            w_ptr1 = get_ptr_as_int64(mW, block_start + Int32(8))

                            x0, x1, x2, x3 = ld_global_v4_u32(h_ptr0)
                            x4, x5, x6, x7 = ld_global_v4_u32(h_ptr1)
                            r0, r1, r2, r3 = ld_global_v4_u32(r_ptr0)
                            r4, r5, r6, r7 = ld_global_v4_u32(r_ptr1)
                            w0, w1, w2, w3 = ld_global_v4_u32(w_ptr0)
                            w4, w5, w6, w7 = ld_global_v4_u32(w_ptr1)

                            if cutlass.const_expr(is_fp16):
                                h0 = hadd2(x0, r0)
                                h1 = hadd2(x1, r1)
                                h2 = hadd2(x2, r2)
                                h3 = hadd2(x3, r3)
                                h4 = hadd2(x4, r4)
                                h5 = hadd2(x5, r5)
                                h6 = hadd2(x6, r6)
                                h7 = hadd2(x7, r7)

                                hw0 = half2_mul(h0, w0)
                                hw1 = half2_mul(h1, w1)
                                hw2 = half2_mul(h2, w2)
                                hw3 = half2_mul(h3, w3)
                                hw4 = half2_mul(h4, w4)
                                hw5 = half2_mul(h5, w5)
                                hw6 = half2_mul(h6, w6)
                                hw7 = half2_mul(h7, w7)

                                abs0 = habs2(hw0)
                                abs1 = habs2(hw1)
                                abs2 = habs2(hw2)
                                abs3 = habs2(hw3)
                                abs4 = habs2(hw4)
                                abs5 = habs2(hw5)
                                abs6 = habs2(hw6)
                                abs7 = habs2(hw7)

                                max01 = hmax2(abs0, abs1)
                                max23 = hmax2(abs2, abs3)
                                max45 = hmax2(abs4, abs5)
                                max67 = hmax2(abs6, abs7)
                                max0123 = hmax2(max01, max23)
                                max4567 = hmax2(max45, max67)
                                max_hw = hmax2(max0123, max4567)

                                max_xw = hmax_to_f32(max_hw)
                                max_abs = max_xw * rstd

                                y0, y1 = half2_to_float2_scaled(hw0, rstd)
                                y2, y3 = half2_to_float2_scaled(hw1, rstd)
                                y4, y5 = half2_to_float2_scaled(hw2, rstd)
                                y6, y7 = half2_to_float2_scaled(hw3, rstd)
                                y8, y9 = half2_to_float2_scaled(hw4, rstd)
                                y10, y11 = half2_to_float2_scaled(hw5, rstd)
                                y12, y13 = half2_to_float2_scaled(hw6, rstd)
                                y14, y15 = half2_to_float2_scaled(hw7, rstd)
                            else:
                                h0 = bfloat2_add(x0, r0)
                                h1 = bfloat2_add(x1, r1)
                                h2 = bfloat2_add(x2, r2)
                                h3 = bfloat2_add(x3, r3)
                                h4 = bfloat2_add(x4, r4)
                                h5 = bfloat2_add(x5, r5)
                                h6 = bfloat2_add(x6, r6)
                                h7 = bfloat2_add(x7, r7)

                                hw0 = bfloat2_mul(h0, w0)
                                hw1 = bfloat2_mul(h1, w1)
                                hw2 = bfloat2_mul(h2, w2)
                                hw3 = bfloat2_mul(h3, w3)
                                hw4 = bfloat2_mul(h4, w4)
                                hw5 = bfloat2_mul(h5, w5)
                                hw6 = bfloat2_mul(h6, w6)
                                hw7 = bfloat2_mul(h7, w7)

                                abs0 = bfloat2_habs2(hw0)
                                abs1 = bfloat2_habs2(hw1)
                                abs2 = bfloat2_habs2(hw2)
                                abs3 = bfloat2_habs2(hw3)
                                abs4 = bfloat2_habs2(hw4)
                                abs5 = bfloat2_habs2(hw5)
                                abs6 = bfloat2_habs2(hw6)
                                abs7 = bfloat2_habs2(hw7)

                                max01 = bfloat2_hmax2(abs0, abs1)
                                max23 = bfloat2_hmax2(abs2, abs3)
                                max45 = bfloat2_hmax2(abs4, abs5)
                                max67 = bfloat2_hmax2(abs6, abs7)
                                max0123 = bfloat2_hmax2(max01, max23)
                                max4567 = bfloat2_hmax2(max45, max67)
                                max_hw = bfloat2_hmax2(max0123, max4567)

                                max_xw = bfloat2_hmax_to_f32(max_hw)
                                max_abs = max_xw * rstd

                                y0, y1 = bfloat2_to_float2_scaled(hw0, rstd)
                                y2, y3 = bfloat2_to_float2_scaled(hw1, rstd)
                                y4, y5 = bfloat2_to_float2_scaled(hw2, rstd)
                                y6, y7 = bfloat2_to_float2_scaled(hw3, rstd)
                                y8, y9 = bfloat2_to_float2_scaled(hw4, rstd)
                                y10, y11 = bfloat2_to_float2_scaled(hw5, rstd)
                                y12, y13 = bfloat2_to_float2_scaled(hw6, rstd)
                                y14, y15 = bfloat2_to_float2_scaled(hw7, rstd)

                            scale_float = max_abs * fp4_max_rcp
                            scale_float = fmin_f32(
                                scale_float, Float32(FLOAT8_E4M3_MAX)
                            )
                            scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                            scale_fp8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                            inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                            if cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_fp8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_fp8

                            q0 = y0 * inv_scale
                            q1 = y1 * inv_scale
                            q2 = y2 * inv_scale
                            q3 = y3 * inv_scale
                            q4 = y4 * inv_scale
                            q5 = y5 * inv_scale
                            q6 = y6 * inv_scale
                            q7 = y7 * inv_scale
                            q8 = y8 * inv_scale
                            q9 = y9 * inv_scale
                            q10 = y10 * inv_scale
                            q11 = y11 * inv_scale
                            q12 = y12 * inv_scale
                            q13 = y13 * inv_scale
                            q14 = y14 * inv_scale
                            q15 = y15 * inv_scale

                            packed_lo = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                            packed_hi = cvt_e2m1x8_f32(
                                q8, q9, q10, q11, q12, q13, q14, q15
                            )
                            packed64 = (Uint64(packed_hi) << Uint64(32)) | Uint64(
                                packed_lo
                            )

                            out_offset = block_start // 2
                            out_ptr = get_ptr_as_int64(
                                mY, actual_row_idx * (H // 2) + out_offset
                            )
                            st_global_u64(out_ptr, packed64)

                    else:
                        # block_size == 32 (MXFP4)
                        if cutlass.const_expr(cluster_n == 1):
                            # Shared memory path for MXFP4 - fully unrolled
                            sh0 = Float32(sH[row_in_block, block_start + Int32(0)])
                            sh1 = Float32(sH[row_in_block, block_start + Int32(1)])
                            sh2 = Float32(sH[row_in_block, block_start + Int32(2)])
                            sh3 = Float32(sH[row_in_block, block_start + Int32(3)])
                            sh4 = Float32(sH[row_in_block, block_start + Int32(4)])
                            sh5 = Float32(sH[row_in_block, block_start + Int32(5)])
                            sh6 = Float32(sH[row_in_block, block_start + Int32(6)])
                            sh7 = Float32(sH[row_in_block, block_start + Int32(7)])
                            sh8 = Float32(sH[row_in_block, block_start + Int32(8)])
                            sh9 = Float32(sH[row_in_block, block_start + Int32(9)])
                            sh10 = Float32(sH[row_in_block, block_start + Int32(10)])
                            sh11 = Float32(sH[row_in_block, block_start + Int32(11)])
                            sh12 = Float32(sH[row_in_block, block_start + Int32(12)])
                            sh13 = Float32(sH[row_in_block, block_start + Int32(13)])
                            sh14 = Float32(sH[row_in_block, block_start + Int32(14)])
                            sh15 = Float32(sH[row_in_block, block_start + Int32(15)])
                            sh16 = Float32(sH[row_in_block, block_start + Int32(16)])
                            sh17 = Float32(sH[row_in_block, block_start + Int32(17)])
                            sh18 = Float32(sH[row_in_block, block_start + Int32(18)])
                            sh19 = Float32(sH[row_in_block, block_start + Int32(19)])
                            sh20 = Float32(sH[row_in_block, block_start + Int32(20)])
                            sh21 = Float32(sH[row_in_block, block_start + Int32(21)])
                            sh22 = Float32(sH[row_in_block, block_start + Int32(22)])
                            sh23 = Float32(sH[row_in_block, block_start + Int32(23)])
                            sh24 = Float32(sH[row_in_block, block_start + Int32(24)])
                            sh25 = Float32(sH[row_in_block, block_start + Int32(25)])
                            sh26 = Float32(sH[row_in_block, block_start + Int32(26)])
                            sh27 = Float32(sH[row_in_block, block_start + Int32(27)])
                            sh28 = Float32(sH[row_in_block, block_start + Int32(28)])
                            sh29 = Float32(sH[row_in_block, block_start + Int32(29)])
                            sh30 = Float32(sH[row_in_block, block_start + Int32(30)])
                            sh31 = Float32(sH[row_in_block, block_start + Int32(31)])

                            sw0 = Float32(sW[row_in_block, block_start + Int32(0)])
                            sw1 = Float32(sW[row_in_block, block_start + Int32(1)])
                            sw2 = Float32(sW[row_in_block, block_start + Int32(2)])
                            sw3 = Float32(sW[row_in_block, block_start + Int32(3)])
                            sw4 = Float32(sW[row_in_block, block_start + Int32(4)])
                            sw5 = Float32(sW[row_in_block, block_start + Int32(5)])
                            sw6 = Float32(sW[row_in_block, block_start + Int32(6)])
                            sw7 = Float32(sW[row_in_block, block_start + Int32(7)])
                            sw8 = Float32(sW[row_in_block, block_start + Int32(8)])
                            sw9 = Float32(sW[row_in_block, block_start + Int32(9)])
                            sw10 = Float32(sW[row_in_block, block_start + Int32(10)])
                            sw11 = Float32(sW[row_in_block, block_start + Int32(11)])
                            sw12 = Float32(sW[row_in_block, block_start + Int32(12)])
                            sw13 = Float32(sW[row_in_block, block_start + Int32(13)])
                            sw14 = Float32(sW[row_in_block, block_start + Int32(14)])
                            sw15 = Float32(sW[row_in_block, block_start + Int32(15)])
                            sw16 = Float32(sW[row_in_block, block_start + Int32(16)])
                            sw17 = Float32(sW[row_in_block, block_start + Int32(17)])
                            sw18 = Float32(sW[row_in_block, block_start + Int32(18)])
                            sw19 = Float32(sW[row_in_block, block_start + Int32(19)])
                            sw20 = Float32(sW[row_in_block, block_start + Int32(20)])
                            sw21 = Float32(sW[row_in_block, block_start + Int32(21)])
                            sw22 = Float32(sW[row_in_block, block_start + Int32(22)])
                            sw23 = Float32(sW[row_in_block, block_start + Int32(23)])
                            sw24 = Float32(sW[row_in_block, block_start + Int32(24)])
                            sw25 = Float32(sW[row_in_block, block_start + Int32(25)])
                            sw26 = Float32(sW[row_in_block, block_start + Int32(26)])
                            sw27 = Float32(sW[row_in_block, block_start + Int32(27)])
                            sw28 = Float32(sW[row_in_block, block_start + Int32(28)])
                            sw29 = Float32(sW[row_in_block, block_start + Int32(29)])
                            sw30 = Float32(sW[row_in_block, block_start + Int32(30)])
                            sw31 = Float32(sW[row_in_block, block_start + Int32(31)])

                            # Compute normalized output: y = h * rstd * w
                            y0 = sh0 * rstd * sw0
                            y1 = sh1 * rstd * sw1
                            y2 = sh2 * rstd * sw2
                            y3 = sh3 * rstd * sw3
                            y4 = sh4 * rstd * sw4
                            y5 = sh5 * rstd * sw5
                            y6 = sh6 * rstd * sw6
                            y7 = sh7 * rstd * sw7
                            y8 = sh8 * rstd * sw8
                            y9 = sh9 * rstd * sw9
                            y10 = sh10 * rstd * sw10
                            y11 = sh11 * rstd * sw11
                            y12 = sh12 * rstd * sw12
                            y13 = sh13 * rstd * sw13
                            y14 = sh14 * rstd * sw14
                            y15 = sh15 * rstd * sw15
                            y16 = sh16 * rstd * sw16
                            y17 = sh17 * rstd * sw17
                            y18 = sh18 * rstd * sw18
                            y19 = sh19 * rstd * sw19
                            y20 = sh20 * rstd * sw20
                            y21 = sh21 * rstd * sw21
                            y22 = sh22 * rstd * sw22
                            y23 = sh23 * rstd * sw23
                            y24 = sh24 * rstd * sw24
                            y25 = sh25 * rstd * sw25
                            y26 = sh26 * rstd * sw26
                            y27 = sh27 * rstd * sw27
                            y28 = sh28 * rstd * sw28
                            y29 = sh29 * rstd * sw29
                            y30 = sh30 * rstd * sw30
                            y31 = sh31 * rstd * sw31

                            # Find max absolute value for scale computation
                            max_abs = fabs_f32(y0)
                            max_abs = fmax_f32(max_abs, fabs_f32(y1))
                            max_abs = fmax_f32(max_abs, fabs_f32(y2))
                            max_abs = fmax_f32(max_abs, fabs_f32(y3))
                            max_abs = fmax_f32(max_abs, fabs_f32(y4))
                            max_abs = fmax_f32(max_abs, fabs_f32(y5))
                            max_abs = fmax_f32(max_abs, fabs_f32(y6))
                            max_abs = fmax_f32(max_abs, fabs_f32(y7))
                            max_abs = fmax_f32(max_abs, fabs_f32(y8))
                            max_abs = fmax_f32(max_abs, fabs_f32(y9))
                            max_abs = fmax_f32(max_abs, fabs_f32(y10))
                            max_abs = fmax_f32(max_abs, fabs_f32(y11))
                            max_abs = fmax_f32(max_abs, fabs_f32(y12))
                            max_abs = fmax_f32(max_abs, fabs_f32(y13))
                            max_abs = fmax_f32(max_abs, fabs_f32(y14))
                            max_abs = fmax_f32(max_abs, fabs_f32(y15))
                            max_abs = fmax_f32(max_abs, fabs_f32(y16))
                            max_abs = fmax_f32(max_abs, fabs_f32(y17))
                            max_abs = fmax_f32(max_abs, fabs_f32(y18))
                            max_abs = fmax_f32(max_abs, fabs_f32(y19))
                            max_abs = fmax_f32(max_abs, fabs_f32(y20))
                            max_abs = fmax_f32(max_abs, fabs_f32(y21))
                            max_abs = fmax_f32(max_abs, fabs_f32(y22))
                            max_abs = fmax_f32(max_abs, fabs_f32(y23))
                            max_abs = fmax_f32(max_abs, fabs_f32(y24))
                            max_abs = fmax_f32(max_abs, fabs_f32(y25))
                            max_abs = fmax_f32(max_abs, fabs_f32(y26))
                            max_abs = fmax_f32(max_abs, fabs_f32(y27))
                            max_abs = fmax_f32(max_abs, fabs_f32(y28))
                            max_abs = fmax_f32(max_abs, fabs_f32(y29))
                            max_abs = fmax_f32(max_abs, fabs_f32(y30))
                            max_abs = fmax_f32(max_abs, fabs_f32(y31))

                            if cutlass.const_expr(self.scale_format == "ue8m0"):
                                scale_float = max_abs * fp4_max_rcp
                                scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                                scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                                inv_scale = ue8m0_to_output_scale(scale_ue8m0)
                            else:
                                scale_float = max_abs * fp4_max_rcp
                                scale_float = fmin_f32(
                                    scale_float, Float32(FLOAT8_E4M3_MAX)
                                )
                                scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                                scale_u8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                                inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                            if cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_u8

                            # Quantize to FP4
                            q0 = y0 * inv_scale
                            q1 = y1 * inv_scale
                            q2 = y2 * inv_scale
                            q3 = y3 * inv_scale
                            q4 = y4 * inv_scale
                            q5 = y5 * inv_scale
                            q6 = y6 * inv_scale
                            q7 = y7 * inv_scale
                            q8 = y8 * inv_scale
                            q9 = y9 * inv_scale
                            q10 = y10 * inv_scale
                            q11 = y11 * inv_scale
                            q12 = y12 * inv_scale
                            q13 = y13 * inv_scale
                            q14 = y14 * inv_scale
                            q15 = y15 * inv_scale
                            q16 = y16 * inv_scale
                            q17 = y17 * inv_scale
                            q18 = y18 * inv_scale
                            q19 = y19 * inv_scale
                            q20 = y20 * inv_scale
                            q21 = y21 * inv_scale
                            q22 = y22 * inv_scale
                            q23 = y23 * inv_scale
                            q24 = y24 * inv_scale
                            q25 = y25 * inv_scale
                            q26 = y26 * inv_scale
                            q27 = y27 * inv_scale
                            q28 = y28 * inv_scale
                            q29 = y29 * inv_scale
                            q30 = y30 * inv_scale
                            q31 = y31 * inv_scale

                            packed_lo_0 = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                            packed_hi_0 = cvt_e2m1x8_f32(
                                q8, q9, q10, q11, q12, q13, q14, q15
                            )
                            packed64_0 = (Uint64(packed_hi_0) << Uint64(32)) | Uint64(
                                packed_lo_0
                            )

                            packed_lo_1 = cvt_e2m1x8_f32(
                                q16, q17, q18, q19, q20, q21, q22, q23
                            )
                            packed_hi_1 = cvt_e2m1x8_f32(
                                q24, q25, q26, q27, q28, q29, q30, q31
                            )
                            packed64_1 = (Uint64(packed_hi_1) << Uint64(32)) | Uint64(
                                packed_lo_1
                            )

                            fp4_offset = actual_row_idx * (H // 2) + sf_idx * (
                                block_size // 2
                            )
                            fp4_ptr_0 = get_ptr_as_int64(mY, fp4_offset)
                            fp4_ptr_1 = get_ptr_as_int64(mY, fp4_offset + Int32(8))
                            st_global_u64(fp4_ptr_0, packed64_0)
                            st_global_u64(fp4_ptr_1, packed64_1)

                        else:
                            # Global memory path for MXFP4 (cluster mode)
                            # Load x, r, w as 4 x 128-bit loads each
                            x_ptr0 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start
                            )
                            x_ptr1 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start + Int32(8)
                            )
                            x_ptr2 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start + Int32(16)
                            )
                            x_ptr3 = get_ptr_as_int64(
                                mX, actual_row_idx * H + block_start + Int32(24)
                            )

                            r_ptr0 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start
                            )
                            r_ptr1 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start + Int32(8)
                            )
                            r_ptr2 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start + Int32(16)
                            )
                            r_ptr3 = get_ptr_as_int64(
                                mR, actual_row_idx * H + block_start + Int32(24)
                            )

                            w_ptr0 = get_ptr_as_int64(mW, block_start)
                            w_ptr1 = get_ptr_as_int64(mW, block_start + Int32(8))
                            w_ptr2 = get_ptr_as_int64(mW, block_start + Int32(16))
                            w_ptr3 = get_ptr_as_int64(mW, block_start + Int32(24))

                            x0, x1, x2, x3 = ld_global_v4_u32(x_ptr0)
                            x4, x5, x6, x7 = ld_global_v4_u32(x_ptr1)
                            x8, x9, x10, x11 = ld_global_v4_u32(x_ptr2)
                            x12, x13, x14, x15 = ld_global_v4_u32(x_ptr3)

                            r0, r1, r2, r3 = ld_global_v4_u32(r_ptr0)
                            r4, r5, r6, r7 = ld_global_v4_u32(r_ptr1)
                            r8, r9, r10, r11 = ld_global_v4_u32(r_ptr2)
                            r12, r13, r14, r15 = ld_global_v4_u32(r_ptr3)

                            w0, w1, w2, w3 = ld_global_v4_u32(w_ptr0)
                            w4, w5, w6, w7 = ld_global_v4_u32(w_ptr1)
                            w8, w9, w10, w11 = ld_global_v4_u32(w_ptr2)
                            w12, w13, w14, w15 = ld_global_v4_u32(w_ptr3)

                            if cutlass.const_expr(is_fp16):
                                h0 = hadd2(x0, r0)
                                h1 = hadd2(x1, r1)
                                h2 = hadd2(x2, r2)
                                h3 = hadd2(x3, r3)
                                h4 = hadd2(x4, r4)
                                h5 = hadd2(x5, r5)
                                h6 = hadd2(x6, r6)
                                h7 = hadd2(x7, r7)
                                h8 = hadd2(x8, r8)
                                h9 = hadd2(x9, r9)
                                h10 = hadd2(x10, r10)
                                h11 = hadd2(x11, r11)
                                h12 = hadd2(x12, r12)
                                h13 = hadd2(x13, r13)
                                h14 = hadd2(x14, r14)
                                h15 = hadd2(x15, r15)

                                hw0 = half2_mul(h0, w0)
                                hw1 = half2_mul(h1, w1)
                                hw2 = half2_mul(h2, w2)
                                hw3 = half2_mul(h3, w3)
                                hw4 = half2_mul(h4, w4)
                                hw5 = half2_mul(h5, w5)
                                hw6 = half2_mul(h6, w6)
                                hw7 = half2_mul(h7, w7)
                                hw8 = half2_mul(h8, w8)
                                hw9 = half2_mul(h9, w9)
                                hw10 = half2_mul(h10, w10)
                                hw11 = half2_mul(h11, w11)
                                hw12 = half2_mul(h12, w12)
                                hw13 = half2_mul(h13, w13)
                                hw14 = half2_mul(h14, w14)
                                hw15 = half2_mul(h15, w15)

                                abs0 = habs2(hw0)
                                abs1 = habs2(hw1)
                                abs2 = habs2(hw2)
                                abs3 = habs2(hw3)
                                abs4 = habs2(hw4)
                                abs5 = habs2(hw5)
                                abs6 = habs2(hw6)
                                abs7 = habs2(hw7)
                                abs8 = habs2(hw8)
                                abs9 = habs2(hw9)
                                abs10 = habs2(hw10)
                                abs11 = habs2(hw11)
                                abs12 = habs2(hw12)
                                abs13 = habs2(hw13)
                                abs14 = habs2(hw14)
                                abs15 = habs2(hw15)

                                max01 = hmax2(abs0, abs1)
                                max23 = hmax2(abs2, abs3)
                                max45 = hmax2(abs4, abs5)
                                max67 = hmax2(abs6, abs7)
                                max89 = hmax2(abs8, abs9)
                                maxab = hmax2(abs10, abs11)
                                maxcd = hmax2(abs12, abs13)
                                maxef = hmax2(abs14, abs15)
                                max0123 = hmax2(max01, max23)
                                max4567 = hmax2(max45, max67)
                                max89ab = hmax2(max89, maxab)
                                maxcdef = hmax2(maxcd, maxef)
                                max_lo = hmax2(max0123, max4567)
                                max_hi = hmax2(max89ab, maxcdef)
                                max_hw = hmax2(max_lo, max_hi)

                                max_xw = hmax_to_f32(max_hw)
                                max_abs = max_xw * rstd

                                y0, y1 = half2_to_float2_scaled(hw0, rstd)
                                y2, y3 = half2_to_float2_scaled(hw1, rstd)
                                y4, y5 = half2_to_float2_scaled(hw2, rstd)
                                y6, y7 = half2_to_float2_scaled(hw3, rstd)
                                y8, y9 = half2_to_float2_scaled(hw4, rstd)
                                y10, y11 = half2_to_float2_scaled(hw5, rstd)
                                y12, y13 = half2_to_float2_scaled(hw6, rstd)
                                y14, y15 = half2_to_float2_scaled(hw7, rstd)
                                y16, y17 = half2_to_float2_scaled(hw8, rstd)
                                y18, y19 = half2_to_float2_scaled(hw9, rstd)
                                y20, y21 = half2_to_float2_scaled(hw10, rstd)
                                y22, y23 = half2_to_float2_scaled(hw11, rstd)
                                y24, y25 = half2_to_float2_scaled(hw12, rstd)
                                y26, y27 = half2_to_float2_scaled(hw13, rstd)
                                y28, y29 = half2_to_float2_scaled(hw14, rstd)
                                y30, y31 = half2_to_float2_scaled(hw15, rstd)
                            else:
                                h0 = bfloat2_add(x0, r0)
                                h1 = bfloat2_add(x1, r1)
                                h2 = bfloat2_add(x2, r2)
                                h3 = bfloat2_add(x3, r3)
                                h4 = bfloat2_add(x4, r4)
                                h5 = bfloat2_add(x5, r5)
                                h6 = bfloat2_add(x6, r6)
                                h7 = bfloat2_add(x7, r7)
                                h8 = bfloat2_add(x8, r8)
                                h9 = bfloat2_add(x9, r9)
                                h10 = bfloat2_add(x10, r10)
                                h11 = bfloat2_add(x11, r11)
                                h12 = bfloat2_add(x12, r12)
                                h13 = bfloat2_add(x13, r13)
                                h14 = bfloat2_add(x14, r14)
                                h15 = bfloat2_add(x15, r15)

                                hw0 = bfloat2_mul(h0, w0)
                                hw1 = bfloat2_mul(h1, w1)
                                hw2 = bfloat2_mul(h2, w2)
                                hw3 = bfloat2_mul(h3, w3)
                                hw4 = bfloat2_mul(h4, w4)
                                hw5 = bfloat2_mul(h5, w5)
                                hw6 = bfloat2_mul(h6, w6)
                                hw7 = bfloat2_mul(h7, w7)
                                hw8 = bfloat2_mul(h8, w8)
                                hw9 = bfloat2_mul(h9, w9)
                                hw10 = bfloat2_mul(h10, w10)
                                hw11 = bfloat2_mul(h11, w11)
                                hw12 = bfloat2_mul(h12, w12)
                                hw13 = bfloat2_mul(h13, w13)
                                hw14 = bfloat2_mul(h14, w14)
                                hw15 = bfloat2_mul(h15, w15)

                                abs0 = bfloat2_habs2(hw0)
                                abs1 = bfloat2_habs2(hw1)
                                abs2 = bfloat2_habs2(hw2)
                                abs3 = bfloat2_habs2(hw3)
                                abs4 = bfloat2_habs2(hw4)
                                abs5 = bfloat2_habs2(hw5)
                                abs6 = bfloat2_habs2(hw6)
                                abs7 = bfloat2_habs2(hw7)
                                abs8 = bfloat2_habs2(hw8)
                                abs9 = bfloat2_habs2(hw9)
                                abs10 = bfloat2_habs2(hw10)
                                abs11 = bfloat2_habs2(hw11)
                                abs12 = bfloat2_habs2(hw12)
                                abs13 = bfloat2_habs2(hw13)
                                abs14 = bfloat2_habs2(hw14)
                                abs15 = bfloat2_habs2(hw15)

                                max01 = bfloat2_hmax2(abs0, abs1)
                                max23 = bfloat2_hmax2(abs2, abs3)
                                max45 = bfloat2_hmax2(abs4, abs5)
                                max67 = bfloat2_hmax2(abs6, abs7)
                                max89 = bfloat2_hmax2(abs8, abs9)
                                maxab = bfloat2_hmax2(abs10, abs11)
                                maxcd = bfloat2_hmax2(abs12, abs13)
                                maxef = bfloat2_hmax2(abs14, abs15)
                                max0123 = bfloat2_hmax2(max01, max23)
                                max4567 = bfloat2_hmax2(max45, max67)
                                max89ab = bfloat2_hmax2(max89, maxab)
                                maxcdef = bfloat2_hmax2(maxcd, maxef)
                                max_lo = bfloat2_hmax2(max0123, max4567)
                                max_hi = bfloat2_hmax2(max89ab, maxcdef)
                                max_hw = bfloat2_hmax2(max_lo, max_hi)

                                max_xw = bfloat2_hmax_to_f32(max_hw)
                                max_abs = max_xw * rstd

                                y0, y1 = bfloat2_to_float2_scaled(hw0, rstd)
                                y2, y3 = bfloat2_to_float2_scaled(hw1, rstd)
                                y4, y5 = bfloat2_to_float2_scaled(hw2, rstd)
                                y6, y7 = bfloat2_to_float2_scaled(hw3, rstd)
                                y8, y9 = bfloat2_to_float2_scaled(hw4, rstd)
                                y10, y11 = bfloat2_to_float2_scaled(hw5, rstd)
                                y12, y13 = bfloat2_to_float2_scaled(hw6, rstd)
                                y14, y15 = bfloat2_to_float2_scaled(hw7, rstd)
                                y16, y17 = bfloat2_to_float2_scaled(hw8, rstd)
                                y18, y19 = bfloat2_to_float2_scaled(hw9, rstd)
                                y20, y21 = bfloat2_to_float2_scaled(hw10, rstd)
                                y22, y23 = bfloat2_to_float2_scaled(hw11, rstd)
                                y24, y25 = bfloat2_to_float2_scaled(hw12, rstd)
                                y26, y27 = bfloat2_to_float2_scaled(hw13, rstd)
                                y28, y29 = bfloat2_to_float2_scaled(hw14, rstd)
                                y30, y31 = bfloat2_to_float2_scaled(hw15, rstd)

                            if cutlass.const_expr(self.scale_format == "ue8m0"):
                                scale_float = max_abs * fp4_max_rcp
                                scale_ue8m0 = cvt_f32_to_ue8m0(scale_float)
                                scale_u8 = Uint8(scale_ue8m0 & Uint32(0xFF))
                                inv_scale = ue8m0_to_output_scale(scale_ue8m0)
                            else:
                                scale_float = max_abs * fp4_max_rcp
                                scale_float = fmin_f32(
                                    scale_float, Float32(FLOAT8_E4M3_MAX)
                                )
                                scale_fp8_u32 = cvt_f32_to_e4m3(scale_float)
                                scale_u8 = Uint8(scale_fp8_u32 & Uint32(0xFF))
                                inv_scale = fp8_e4m3_to_f32_and_rcp(scale_fp8_u32)

                            if cutlass.const_expr(self.output_swizzled):
                                inner_k_idx = sf_idx % Int32(4)
                                inner_m_idx = (actual_row_idx % Int32(128)) // Int32(32)
                                outer_m_idx = actual_row_idx % Int32(32)
                                k_tile_idx = sf_idx // Int32(4)
                                m_tile_idx = actual_row_idx // Int32(128)
                                m_tile_stride = self.num_k_tiles * self.k_tile_stride
                                swizzled_offset = (
                                    m_tile_idx * m_tile_stride
                                    + k_tile_idx * self.k_tile_stride
                                    + outer_m_idx * Int32(16)
                                    + inner_m_idx * Int32(4)
                                    + inner_k_idx
                                )
                                mS[swizzled_offset] = scale_u8
                            else:
                                mS[actual_row_idx, sf_idx] = scale_u8

                            q0 = y0 * inv_scale
                            q1 = y1 * inv_scale
                            q2 = y2 * inv_scale
                            q3 = y3 * inv_scale
                            q4 = y4 * inv_scale
                            q5 = y5 * inv_scale
                            q6 = y6 * inv_scale
                            q7 = y7 * inv_scale
                            q8 = y8 * inv_scale
                            q9 = y9 * inv_scale
                            q10 = y10 * inv_scale
                            q11 = y11 * inv_scale
                            q12 = y12 * inv_scale
                            q13 = y13 * inv_scale
                            q14 = y14 * inv_scale
                            q15 = y15 * inv_scale
                            q16 = y16 * inv_scale
                            q17 = y17 * inv_scale
                            q18 = y18 * inv_scale
                            q19 = y19 * inv_scale
                            q20 = y20 * inv_scale
                            q21 = y21 * inv_scale
                            q22 = y22 * inv_scale
                            q23 = y23 * inv_scale
                            q24 = y24 * inv_scale
                            q25 = y25 * inv_scale
                            q26 = y26 * inv_scale
                            q27 = y27 * inv_scale
                            q28 = y28 * inv_scale
                            q29 = y29 * inv_scale
                            q30 = y30 * inv_scale
                            q31 = y31 * inv_scale

                            packed_lo_0 = cvt_e2m1x8_f32(q0, q1, q2, q3, q4, q5, q6, q7)
                            packed_hi_0 = cvt_e2m1x8_f32(
                                q8, q9, q10, q11, q12, q13, q14, q15
                            )
                            packed64_0 = (Uint64(packed_hi_0) << Uint64(32)) | Uint64(
                                packed_lo_0
                            )

                            packed_lo_1 = cvt_e2m1x8_f32(
                                q16, q17, q18, q19, q20, q21, q22, q23
                            )
                            packed_hi_1 = cvt_e2m1x8_f32(
                                q24, q25, q26, q27, q28, q29, q30, q31
                            )
                            packed64_1 = (Uint64(packed_hi_1) << Uint64(32)) | Uint64(
                                packed_lo_1
                            )

                            fp4_offset = actual_row_idx * (H // 2) + sf_idx * (
                                block_size // 2
                            )
                            fp4_ptr_0 = get_ptr_as_int64(mY, fp4_offset)
                            fp4_ptr_1 = get_ptr_as_int64(mY, fp4_offset + Int32(8))
                            st_global_u64(fp4_ptr_0, packed64_0)
                            st_global_u64(fp4_ptr_1, packed64_1)


# =============================================================================
# PyTorch API Functions
# =============================================================================


@functools.cache
def _get_compiled_kernel(
    hidden_size: int,
    block_size: int,
    is_fp16: bool,
    sm_version: int,
    scale_format: str,
    is_sf_swizzled_layout: bool,
) -> Callable:
    """Get a compiled kernel closure that takes torch.Tensor directly."""
    cutlass_dtype = cutlass.Float16 if is_fp16 else cutlass.BFloat16

    def get_cute_pointers(tensors):
        """Convert torch tensors to cute pointers."""
        if tensors is None:
            return [
                make_ptr(
                    cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # x
                make_ptr(
                    cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # r
                make_ptr(
                    cutlass_dtype, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # w
                make_ptr(
                    cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # y
                make_ptr(
                    cutlass.Uint8, 16, cute.AddressSpace.gmem, assumed_align=16
                ),  # s
            ]
        x, r, w, y, s = tensors
        return [
            make_ptr(
                cutlass_dtype, x.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass_dtype, r.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass_dtype, w.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint8, y.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
            make_ptr(
                cutlass.Uint8, s.data_ptr(), cute.AddressSpace.gmem, assumed_align=16
            ),
        ]

    kernel_obj = AddRMSNormFP4QuantKernel(
        dtype=cutlass_dtype,
        H=hidden_size,
        block_size=block_size,
        output_swizzled=is_sf_swizzled_layout,
        is_fp16=is_fp16,
        sm_version=sm_version,
        scale_format=scale_format,
    )

    compiled_kernel = cute.compile(
        kernel_obj,
        *get_cute_pointers(None),
        Int32(1),
        Float32(1e-6),
        cutlass_torch.current_stream(),
    )

    def tensor_api(
        x: torch.Tensor,
        r: torch.Tensor,
        w: torch.Tensor,
        y: torch.Tensor,
        s: torch.Tensor,
        M: int,
        eps: float,
    ) -> None:
        """Runtime API that converts tensors to pointers and calls the kernel."""
        nonlocal compiled_kernel
        compiled_kernel(
            *get_cute_pointers([x, r, w, y, s]),
            Int32(M),
            Float32(eps),
            cutlass_torch.current_stream(),
        )

    return tensor_api


@flashinfer_api
def add_rmsnorm_fp4quant(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    y_fp4: torch.Tensor,
    block_scale: torch.Tensor,
    eps: float = 1e-6,
    block_size: int = 16,
    scale_format: str | None = None,
    is_sf_swizzled_layout: bool = False,
) -> None:
    """
    Fused Add + RMS normalization + FP4 quantization using CuTe-DSL.

    Computes: ``h = input + residual``, then ``y = RMSNorm(h) * weight``,
    and finally quantizes ``y`` to FP4.

    Parameters
    ----------
    input : torch.Tensor
        Input tensor, shape ``(batch_size, hidden_size)`` or ``(batch_size, seq_len, hidden_size)``.
        Must be ``torch.float16`` or ``torch.bfloat16``.
    residual : torch.Tensor
        Residual tensor to add to input. Must have the same shape and dtype as ``input``.
    weight : torch.Tensor
        Weight tensor for RMSNorm, shape ``(hidden_size,)``.
        Must have the same dtype as input.
    y_fp4 : torch.Tensor
        Output tensor for quantized values in FP4_E2M1 format, packed as uint8.
        Two FP4 values are packed into each uint8 byte.
        Shape must be ``(batch_size, hidden_size // 2)`` or matching 3D input.
    block_scale : torch.Tensor
        Output tensor for per-block scale factors.

        - If ``is_sf_swizzled_layout=False`` (default): row-major layout with shape
          ``(batch_size, hidden_size // block_size)`` or matching 3D input.
        - If ``is_sf_swizzled_layout=True``: swizzled layout for efficient tensor core
          access, with shape ``(batch_size * hidden_size // block_size,)`` flattened.
          The swizzle pattern uses 128x4 tiles where scales are arranged as:
          ``[m_tile][k_tile][outer_m (32)][inner_m (4)][inner_k (4)]``.

        Dtype should be ``torch.float8_e4m3fn`` for E4M3 format or ``torch.uint8``
        for UE8M0 format.
    eps : float
        Epsilon for numerical stability in RMSNorm. Default is ``1e-6``.
    block_size : int
        Number of elements per quantization block. Default is ``16``.

        - ``16``: NVFP4 format with E4M3 scale factors
        - ``32``: MXFP4 format with UE8M0 scale factors
    scale_format : str, optional
        Scale factor format: ``"e4m3"`` or ``"ue8m0"``.
        If ``None``, auto-selects based on ``block_size``:
        ``"e4m3"`` for block_size=16, ``"ue8m0"`` for block_size=32.
    is_sf_swizzled_layout : bool
        If ``True``, output scale factors in swizzled layout optimized for
        tensor core GEMM operations. The swizzle uses 128x4 tiles with the pattern:
        ``[m_tile_idx * k_tiles * 512 + k_tile_idx * 512 + outer_m * 16 + inner_m * 4 + inner_k]``
        where ``outer_m = row % 32``, ``inner_m = (row % 128) // 32``, etc.
        Default is ``False`` (row-major layout).

    Notes
    -----
    - Requires SM100+ (Blackwell) for FP4 quantization PTX intrinsics.
    - For block_size=16 (NVFP4): uses E4M3 scale factors (max value 448.0).
    - For block_size=32 (MXFP4): uses UE8M0 scale factors (power-of-2 scales).
    - FP4 E2M1 format has a max representable value of 6.0.
    """
    is_3d = input.dim() == 3
    if is_3d:
        B, S, H = input.shape
        input = input.view(B * S, H).contiguous()
        residual = residual.view(B * S, H).contiguous()
        y_fp4_2d = y_fp4.view(B * S, -1)
        block_scale_2d = block_scale.view(B * S, -1)
    else:
        y_fp4_2d = y_fp4
        block_scale_2d = block_scale

    batch_size, hidden_size = input.shape
    dtype = input.dtype

    assert hidden_size % block_size == 0, "hidden_size must be divisible by block_size"
    assert hidden_size >= 64, "hidden_size must be >= 64"
    assert block_size in [16, 32], "block_size must be 16 or 32"

    is_fp16 = dtype == torch.float16
    actual_scale_format = (
        scale_format if scale_format else ("ue8m0" if block_size == 32 else "e4m3")
    )
    sm_version = get_sm_version(input.device)

    tensor_api = _get_compiled_kernel(
        hidden_size,
        block_size,
        is_fp16,
        sm_version,
        actual_scale_format,
        is_sf_swizzled_layout,
    )
    tensor_api(
        input.contiguous(),
        residual.contiguous(),
        weight.contiguous(),
        y_fp4_2d,
        block_scale_2d.view(torch.uint8),
        batch_size,
        eps,
    )


__all__ = [
    "AddRMSNormFP4QuantKernel",
    "add_rmsnorm_fp4quant",
    "get_sm_version",
]
