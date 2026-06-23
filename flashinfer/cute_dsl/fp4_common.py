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
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm


# =============================================================================
# Constants
# =============================================================================

FLOAT4_E2M1_MAX = 6.0  # Maximum value representable in FP4 E2M1
FLOAT8_E4M3_MAX = 448.0  # Maximum value representable in FP8 E4M3
SF_VEC_SIZE = 16  # Elements per scale factor block
MX_SF_VEC_SIZE = 32  # Elements per UE8M0 scale block (MXFP8 w4a8 activations)
_INV_FLOAT8_E4M3_MAX = 1.0 / FLOAT8_E4M3_MAX
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
def ld_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) using generic addressing (works for GMEM and SMEM)."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.v4.u32 {$0, $1, $2, $3}, [$4];",
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
def ld_global_nc_u32(base_ptr: Int64, *, loc=None, ip=None) -> Uint32:
    """Load 32 bits from global memory through the non-coherent cache."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.nc.u32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_nc_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits from global memory through the non-coherent cache."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.nc.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,l",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)),
    )


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
def st_global_u32(base_ptr: Int64, value: Uint32, *, loc=None, ip=None):
    """Store 32 bits to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.u32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64.

    WARNING: This uses ptrtoint which strips address space information.
    For SMEM tensors, the resulting Int64 is a raw SMEM offset that does NOT
    work with generic-addressing loads (ld.v4.u32). Use only with explicit
    address-space loads (ld.global.*) or for global memory tensors.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def get_smem_ptr_as_int32(
    tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None
) -> Int32:
    """Get the shared-memory byte address of tensor[offset] as Int32.

    Uses Pointer.toint() which preserves the SMEM address space (addrspace 3),
    returning a 32-bit SMEM address suitable for ld.shared.* instructions.
    """
    elem_ptr = tensor.iterator + Int32(offset)
    return elem_ptr.toint(loc=loc, ip=ip)


@dsl_user_op
def ld_shared_v2_u32(smem_addr: Int32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Load 64 bits (2 x uint32) from shared memory via ld.shared.v2.u32.

    Args:
        smem_addr: 32-bit shared memory address (from get_smem_ptr_as_int32).
                   Caller is responsible for ensuring 8-byte alignment.

    Returns:
        2 Uint32 values (8 bytes total).
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v2.u32 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    v0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    v1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(v0), Uint32(v1)


@dsl_user_op
def ld_shared_v4_u32(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Load 128 bits (4 x uint32) from shared memory via ld.shared.v4.u32.

    Args:
        smem_addr: 32-bit shared memory address (from get_smem_ptr_as_int32).

    Returns:
        4 Uint32 values (16 bytes total, e.g. 8 packed fp16 elements).
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.u32 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
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
def pack_16bit_to_u32(lo, hi, *, loc=None, ip=None) -> Uint32:
    """Pack two 16-bit scalar values (fp16 or bf16) into one Uint32 (half2/bfloat2).

    Uses PTX mov.b32 to bitwise-pack two 16-bit register values into a single
    32-bit register, suitable for half2/bfloat2 SIMD operations.
    """
    lo_ir = lo.ir_value(loc=loc, ip=ip)
    hi_ir = hi.ir_value(loc=loc, ip=ip)
    lo_i16 = llvm.bitcast(T.i16(), lo_ir, loc=loc, ip=ip)
    hi_i16 = llvm.bitcast(T.i16(), hi_ir, loc=loc, ip=ip)
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [lo_i16, hi_i16],
            "mov.b32 $0, {$1, $2};",
            "=r,h,h",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


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
def rcp_rn(a: Float32, *, loc=None, ip=None) -> Float32:
    """Round-to-nearest reciprocal using PTX div.rn.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, 0f3F800000, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def fadd_rn(a: Float32, b: Float32, loc=None, ip=None) -> Float32:
    """Round-to-nearest float32 addition."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "add.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fsub_rn(a: Float32, b: Float32, loc=None, ip=None) -> Float32:
    """Round-to-nearest float32 subtraction."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "sub.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fmul_rn(a: Float32, b: Float32, loc=None, ip=None) -> Float32:
    """Round-to-nearest float32 multiplication."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "mul.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fdiv_rn(a: Float32, b: Float32, loc=None, ip=None) -> Float32:
    """Round-to-nearest float32 division."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(a).ir_value(loc=loc, ip=ip), Float32(b).ir_value(loc=loc, ip=ip)],
            "div.rn.f32 $0, $1, $2;",
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
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
def cvt_e4m3x4_to_f32x4(
    packed: Uint32, *, loc=None, ip=None
) -> tuple[Float32, Float32, Float32, Float32]:
    """Convert 4 packed E4M3 bytes (in a uint32) to 4 float32 values.

    Uses e4m3x2 → f16x2 → f32 conversion path (SM89+/PTX ISA 7.8+).
    Input: uint32 containing bytes [b0, b1, b2, b3] (low to high).
    Output: (f0, f1, f2, f3) as Float32.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 pair_lo, pair_hi;
            .reg .b32 h2_lo, h2_hi;
            .reg .b16 h0, h1, h2, h3;
            mov.b32 {pair_lo, pair_hi}, $4;
            cvt.rn.f16x2.e4m3x2 h2_lo, pair_lo;
            cvt.rn.f16x2.e4m3x2 h2_hi, pair_hi;
            mov.b32 {h0, h1}, h2_lo;
            mov.b32 {h2, h3}, h2_hi;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
            cvt.f32.f16 $2, h2;
            cvt.f32.f16 $3, h3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    f0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    f1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    f2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    f3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)

    return Float32(f0), Float32(f1), Float32(f2), Float32(f3)


@dsl_user_op
def cvt_f32x2_to_half2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two float32 values into a half2 (uint32 containing two fp16 values).

    Uses cvt.rn.f16.f32 for each value, then packs into a single uint32.
    Matches __float22half2_rn() behavior in CUDA.
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 h0, h1;
                cvt.rn.f16.f32 h0, $1;
                cvt.rn.f16.f32 h1, $2;
                mov.b32 $0, {h0, h1};
            }
            """,
            "=r,f,f",
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


@dsl_user_op
def nvfp4_compute_output_scale(
    fp8_val: Uint32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute NVFP4 output_scale matching the CUDA kernel exactly.

    Converts E4M3 scale factor to float via hardware f16x2 path, then computes
    rcp(float_scale * rcp(global_scale)). Returns 0 when scale is zero.

    This matches quantization_utils.cuh:
        SFValue = static_cast<float>(tmp);
        outputScale = rcp_approx(SFValue * rcp_approx(SFScaleVal));
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32, rcp_gs, product, result;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;

                rcp.approx.ftz.f32 rcp_gs, $2;
                mul.f32 product, scale_f32, rcp_gs;
                rcp.approx.ftz.f32 result, product;

                setp.eq.f32 p_zero, scale_f32, 0f00000000;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def nvfp4_compute_output_scale_rn(
    fp8_val: Uint32, global_scale: Float32, block_amax: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute TE-exact NVFP4 output scale when FP4 quant fast math is disabled."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(fp8_val).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
                Float32(block_amax).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .b16 fp8_pair;
                .reg .b32 h2_32;
                .reg .b16 h_lo, h_hi;
                .reg .f32 scale_f32, rcp_gs, product, result, max_f32;

                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2_32, fp8_pair;
                mov.b32 {h_lo, h_hi}, h2_32;
                cvt.f32.f16 scale_f32, h_lo;

                div.rn.f32 rcp_gs, 0f3F800000, $2;
                mul.rn.f32 product, scale_f32, rcp_gs;
                div.rn.f32 result, 0f3F800000, product;
                mov.b32 max_f32, 0x7F7FFFFF;
                min.f32 result, result, max_f32;

                setp.eq.f32 p_zero, $3, 0f00000000;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
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


# =============================================================================
# PTX intrinsics used by SM120 MoE kernels
# =============================================================================


@dsl_user_op
def st_global_f32(base_ptr: Int64, value: Float32, *, loc=None, ip=None):
    """Store 32-bit float to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(value).ir_value(loc=loc, ip=ip),
        ],
        "st.global.f32 [$0], $1;",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits to global memory as four uint32 words."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.u32 [$0], {$1, $2, $3, $4};",
        "l,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# PTX Intrinsics — Shared Memory Operations
# =============================================================================


@dsl_user_op
def shared_ptr_to_u32(ptr: cute.Pointer, *, loc=None, ip=None) -> Int32:
    """Convert an address-space-3 shared-memory pointer to a u32 address."""
    return Int32(llvm.ptrtoint(T.i32(), ptr.llvm_ptr, loc=loc, ip=ip))


@dsl_user_op
def ld_shared_i32_relaxed(addr: Int32, *, loc=None, ip=None) -> Int32:
    """Load int32 from shared memory when CSE/reordering is acceptable."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.s32 $0, [$1];",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None):
    """Store int32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.s32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_f32(addr: Int32, *, loc=None, ip=None) -> Float32:
    """Load float32 from shared memory at a 32-bit byte address."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.f32 $0, [$1];",
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Store float32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.f32 [$0], $1;",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_u8(smem_addr: Int32, value: Uint8, *, loc=None, ip=None):
    """Store 8 bits to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint8(value).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u8 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


# =============================================================================
# PTX Intrinsics — Global Atomics
# =============================================================================


@dsl_user_op
def atomic_add_global_i32(addr: Int64, val: Int32, *, loc=None, ip=None) -> Int32:
    """Global memory int32 atomic add. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.add.s32 $0, [$1], $2;",
            "=r,l,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def atomic_cas_global_i32(
    addr: Int64, compare: Int32, value: Int32, *, loc=None, ip=None
) -> Int32:
    """Global memory int32 atomic compare-and-swap. Returns old value."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int64(addr).ir_value(loc=loc, ip=ip),
                Int32(compare).ir_value(loc=loc, ip=ip),
                Int32(value).ir_value(loc=loc, ip=ip),
            ],
            "atom.global.cas.b32 $0, [$1], $2, $3;",
            "=r,l,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_acquire_i32(addr: Int64, *, loc=None, ip=None) -> Int32:
    """Load int32 from global memory with acquire semantics."""
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [Int64(addr).ir_value(loc=loc, ip=ip)],
            "ld.global.acquire.gpu.s32 $0, [$1];",
            "=r,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """Store int32 to global memory with release semantics."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.global.release.gpu.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def spin_wait_global_eq_i32(addr: Int64, expected: Int32, *, loc=None, ip=None):
    """Spin while *addr == expected, then continue with acquire semantics."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(expected).ir_value(loc=loc, ip=ip),
        ],
        "{\n"
        ".reg .pred %p0;\n"
        ".reg .s32 %val;\n"
        "spin_loop:\n"
        "  ld.global.acquire.gpu.s32 %val, [$0];\n"
        "  setp.eq.s32 %p0, %val, $1;\n"
        "  @%p0 bra spin_loop;\n"
        "}",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def threadfence(*, loc=None, ip=None):
    """Emit a global-memory fence."""
    llvm.inline_asm(
        None,
        [],
        "membar.gl;",
        "",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# PTX Intrinsics — Scatter Atomics (BF16)
# =============================================================================


@dsl_user_op
def scatter_add_bf16x2(addr: Int64, val0_f32, val1_f32, *, loc=None, ip=None):
    """BF16x2 atomic reduction add to global memory.

    Packs two f32 values into bf16x2 via cvt.rn.satfinite, then does
    red.relaxed.gpu.global.add.noftz.bf16x2.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            val0_f32.ir_value(loc=loc, ip=ip),
            val1_f32.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 packed;"
        " cvt.rn.satfinite.bf16x2.f32 packed, $2, $1;"
        " red.relaxed.gpu.global.add.noftz.bf16x2 [$0], packed; }",
        "l,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def scatter_add_v4_bf16x2(
    addr: Int64, v0, v1, v2, v3, v4, v5, v6, v7, *, loc=None, ip=None
):
    """Vectorized BF16x2 atomic reduction of 8 f32 values."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            v0.ir_value(loc=loc, ip=ip),
            v1.ir_value(loc=loc, ip=ip),
            v2.ir_value(loc=loc, ip=ip),
            v3.ir_value(loc=loc, ip=ip),
            v4.ir_value(loc=loc, ip=ip),
            v5.ir_value(loc=loc, ip=ip),
            v6.ir_value(loc=loc, ip=ip),
            v7.ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b32 p0,p1,p2,p3;"
        " cvt.rn.satfinite.bf16x2.f32 p0, $2, $1;"
        " cvt.rn.satfinite.bf16x2.f32 p1, $4, $3;"
        " cvt.rn.satfinite.bf16x2.f32 p2, $6, $5;"
        " cvt.rn.satfinite.bf16x2.f32 p3, $8, $7;"
        " red.global.add.noftz.v4.bf16x2 [$0], {p0, p1, p2, p3}; }",
        "l,f,f,f,f,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


# =============================================================================
# FP8 Conversion — Non-reciprocal version
# =============================================================================


@dsl_user_op
def fp8_e4m3_to_f32(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert FP8 E4M3 to float32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p_zero, p_neg;
                .reg .u32 sign_u, exp_u, mant_u;
                .reg .s32 exp_s;
                .reg .f32 exp_f, mant_f, fp8_float, fp8_neg;

                setp.eq.u32 p_zero, $1, 0;
                and.b32 sign_u, $1, 0x80;
                and.b32 mant_u, $1, 7;
                shr.b32 exp_u, $1, 3;
                and.b32 exp_u, exp_u, 15;
                sub.s32 exp_s, exp_u, 7;
                cvt.rn.f32.s32 exp_f, exp_s;
                ex2.approx.f32 exp_f, exp_f;
                cvt.rn.f32.u32 mant_f, mant_u;
                fma.rn.f32 mant_f, mant_f, 0f3E000000, 0f3F800000;
                mul.f32 fp8_float, exp_f, mant_f;
                neg.f32 fp8_neg, fp8_float;
                setp.ne.u32 p_neg, sign_u, 0;
                selp.f32 fp8_float, fp8_neg, fp8_float, p_neg;
                selp.f32 $0, 0f00000000, fp8_float, p_zero;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def nvfp4_scale_from_amax(
    block_amax: Float32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute the NVFP4 block scale from an amax and global scale."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .f64 amax_d, gs_d, q_d, six_d;
                cvt.f64.f32 amax_d, $1;
                cvt.f64.f32 gs_d, $2;
                mov.f64 six_d, 0d4018000000000000;
                mul.rn.f64 q_d, amax_d, gs_d;
                div.rn.f64 q_d, q_d, six_d;
                cvt.rn.f32.f64 $0, q_d;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def nvfp4_scale_from_amax_rn(
    block_amax: Float32, global_scale: Float32, *, loc=None, ip=None
) -> Float32:
    """Compute NVFP4 block scale with round-to-nearest FP32 operations."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Float32(block_amax).ir_value(loc=loc, ip=ip),
                Float32(global_scale).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p_zero;
                .reg .f32 scale_mul, result;
                setp.eq.f32 p_zero, $1, 0f00000000;
                mul.rn.f32 scale_mul, $2, 0f3E2AAAAB;
                mul.rn.f32 result, $1, scale_mul;
                selp.f32 $0, 0f00000000, result, p_zero;
            }
            """,
            "=f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e4m3x2_to_f16x2_pair(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Decode 4 packed E4M3 bytes into two f16x2 registers."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 b16_01, b16_23;
            .reg .b32 hi32;
            cvt.u16.u32 b16_01, $2;
            shr.b32 hi32, $2, 16;
            cvt.u16.u32 b16_23, hi32;
            cvt.rn.f16x2.e4m3x2 $0, b16_01;
            cvt.rn.f16x2.e4m3x2 $1, b16_23;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def f16x2_to_f32x2(packed_h2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Unpack one f16x2 register into two f32 values."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32, f32)>"),
        [Uint32(packed_h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 lo, hi;
            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 $0, lo;
            cvt.f32.f16 $1, hi;
        }
        """,
        "=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), res, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), res, [1], loc=loc, ip=ip)),
    )


@dsl_user_op
def cvt_e4m3_to_f32_via_f16(fp8_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert one E4M3 byte to f32 through native E4M3-to-f16 conversion."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b16 fp8_pair;
                .reg .b32 h2;
                .reg .b16 lo, hi;
                cvt.u16.u32 fp8_pair, $1;
                cvt.rn.f16x2.e4m3x2 h2, fp8_pair;
                mov.b32 {lo, hi}, h2;
                cvt.f32.f16 $0, lo;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_s0e5m3_to_f16x2_broadcast(fp8_val: Uint32, *, loc=None, ip=None) -> Uint32:
    """Convert one S0E5M3 scale byte to an f16x2 with the scale in both lanes.

    S0E5M3 (sign-0, 5-exp, 3-mantissa) is a host-side reformat of the per-block
    E4M3 scale, rebiased to fp16's bias (exp 7->15, i.e. byte += 0x40) so the
    bits line up with fp16 directly: ``f16(byte) = byte << 7`` (exp -> bits 14-10,
    the 3 mantissa bits -> bits 9-7, low mantissa and sign = 0).  Broadcasting to
    both f16x2 lanes is then ``byte * 0x00800080`` = ``(byte<<7) | (byte<<23)`` --
    a single ``mul.lo.u32`` (the two shifted copies never overlap, so the mul is
    exactly the OR).  Replaces the 3-op E4M3 path (cvt.u16 + cvt.f16x2.e4m3x2 +
    prmt) with 1 op, and the per-block scale stays 1 byte (memory-neutral).
    Numerically exact for normal E4M3 scales (both formats carry 3 mantissa bits).
    """
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(fp8_val).ir_value(loc=loc, ip=ip)],
            "mul.lo.u32 $0, $1, 0x00800080;",
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp4_decode_4bytes(
    packed_u32: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Decode 4 packed FP4 bytes into four f16x2 registers."""
    res = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32, i32, i32, i32)>"),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b8 byte0, byte1, byte2, byte3;
            mov.b32 {byte0, byte1, byte2, byte3}, $4;
            cvt.rn.f16x2.e2m1x2 $0, byte0;
            cvt.rn.f16x2.e2m1x2 $1, byte1;
            cvt.rn.f16x2.e2m1x2 $2, byte2;
            cvt.rn.f16x2.e2m1x2 $3, byte3;
        }
        """,
        "=r,=r,=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Uint32(llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [2], loc=loc, ip=ip)),
        Uint32(llvm.extractvalue(T.i32(), res, [3], loc=loc, ip=ip)),
    )


@dsl_user_op
def fp4_decode_2(byte_val: Uint32, *, loc=None, ip=None) -> Uint32:
    """Decode one FP4 byte into one f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Uint32(byte_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b8 b0;
                cvt.u8.u32 b0, $1;
                cvt.rn.f16x2.e2m1x2 $0, b0;
            }
            """,
            "=r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_fp32x2_to_e2m1x2(v0: Float32, v1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert two f32 values into one packed E2M1 byte."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b;
                cvt.rn.satfinite.e2m1x2.f32 b, $2, $1;
                cvt.u32.u8 $0, b;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def quant_dequant_2(
    v0: Float32, v1: Float32, sf_f32: Float32, eff_scale: Float32
) -> Tuple[Float32, Float32]:
    """Quantize/dequantize two values through FP4."""
    inv_scale = Float32(1.0) / eff_scale
    fp4_byte = cvt_fp32x2_to_e2m1x2(v0 * inv_scale, v1 * inv_scale)
    h2 = fp4_decode_2(fp4_byte)
    f0, f1 = f16x2_to_f32x2(h2)
    return f0 * sf_f32, f1 * sf_f32


@dsl_user_op
def fp4_dot4_sum(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Decode 4 FP4 bytes and dot with four f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_packed).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {b0, b1, b2, b3}, $1;
                cvt.rn.f16x2.e2m1x2 h0, b0;
                cvt.rn.f16x2.e2m1x2 h1, b1;
                cvt.rn.f16x2.e2m1x2 h2, b2;
                cvt.rn.f16x2.e2m1x2 h3, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $2, acc;
                fma.rn.f16x2 acc, h1, $3, acc;
                fma.rn.f16x2 acc, h2, $4, acc;
                fma.rn.f16x2 acc, h3, $5, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def fp4_dot8_sum(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
    *,
    loc=None,
    ip=None,
) -> Float32:
    """Decode 8 FP4 bytes and dot with eight f16x2 inputs."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [
                Uint32(u_a).ir_value(loc=loc, ip=ip),
                Uint32(u_b).ir_value(loc=loc, ip=ip),
                Uint32(x0).ir_value(loc=loc, ip=ip),
                Uint32(x1).ir_value(loc=loc, ip=ip),
                Uint32(x2).ir_value(loc=loc, ip=ip),
                Uint32(x3).ir_value(loc=loc, ip=ip),
                Uint32(x4).ir_value(loc=loc, ip=ip),
                Uint32(x5).ir_value(loc=loc, ip=ip),
                Uint32(x6).ir_value(loc=loc, ip=ip),
                Uint32(x7).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b8 a0, a1, a2, a3;
                .reg .b8 b0, b1, b2, b3;
                .reg .b32 h0, h1, h2, h3, h4, h5, h6, h7;
                .reg .f16x2 acc;
                .reg .b16 lo, hi;
                .reg .f32 flo, fhi;
                mov.b32 {a0, a1, a2, a3}, $1;
                mov.b32 {b0, b1, b2, b3}, $2;
                cvt.rn.f16x2.e2m1x2 h0, a0;
                cvt.rn.f16x2.e2m1x2 h1, a1;
                cvt.rn.f16x2.e2m1x2 h2, a2;
                cvt.rn.f16x2.e2m1x2 h3, a3;
                cvt.rn.f16x2.e2m1x2 h4, b0;
                cvt.rn.f16x2.e2m1x2 h5, b1;
                cvt.rn.f16x2.e2m1x2 h6, b2;
                cvt.rn.f16x2.e2m1x2 h7, b3;
                mov.b32 acc, 0;
                fma.rn.f16x2 acc, h0, $3, acc;
                fma.rn.f16x2 acc, h1, $4, acc;
                fma.rn.f16x2 acc, h2, $5, acc;
                fma.rn.f16x2 acc, h3, $6, acc;
                fma.rn.f16x2 acc, h4, $7, acc;
                fma.rn.f16x2 acc, h5, $8, acc;
                fma.rn.f16x2 acc, h6, $9, acc;
                fma.rn.f16x2 acc, h7, $10, acc;
                mov.b32 {lo, hi}, acc;
                cvt.f32.f16 flo, lo;
                cvt.f32.f16 fhi, hi;
                add.f32 $0, flo, fhi;
            }
            """,
            "=f,r,r,r,r,r,r,r,r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@cute.jit
def _f16x2_dot_sum_f32acc(a_h2: Uint32, b_h2: Uint32) -> Float32:
    a0, a1 = f16x2_to_f32x2(a_h2)
    b0, b1 = f16x2_to_f32x2(b_h2)
    return a0 * b0 + a1 * b1


@cute.jit
def fp4_dot4_sum_f32acc(
    u_packed: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
) -> Float32:
    """Decode FP4 then accumulate a 4-byte dot product in f32."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_packed)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
    )


@cute.jit
def fp4_dot8_sum_f32acc(
    u_a: Uint32,
    u_b: Uint32,
    x0: Uint32,
    x1: Uint32,
    x2: Uint32,
    x3: Uint32,
    x4: Uint32,
    x5: Uint32,
    x6: Uint32,
    x7: Uint32,
) -> Float32:
    """Decode FP4 then accumulate an 8-byte dot product in f32."""
    h0, h1, h2, h3 = fp4_decode_4bytes(u_a)
    h4, h5, h6, h7 = fp4_decode_4bytes(u_b)
    return (
        _f16x2_dot_sum_f32acc(h0, x0)
        + _f16x2_dot_sum_f32acc(h1, x1)
        + _f16x2_dot_sum_f32acc(h2, x2)
        + _f16x2_dot_sum_f32acc(h3, x3)
        + _f16x2_dot_sum_f32acc(h4, x4)
        + _f16x2_dot_sum_f32acc(h5, x5)
        + _f16x2_dot_sum_f32acc(h6, x6)
        + _f16x2_dot_sum_f32acc(h7, x7)
    )


@dsl_user_op
def pack_f32x2_to_f16x2(a: Float32, b: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack two f32 values into one f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.f16.f32 lo, $1;
                cvt.rn.f16.f32 hi, $2;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


# =============================================================================
# FP4 Quantization — Fast approximate path
# =============================================================================


@cute.jit
def quantize_and_pack_16_fast(y_f32: cute.Tensor, inv_scale: Float32) -> Uint64:
    """Fast approximate FP4 quantize/pack for 16 float32 values."""
    q = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        q[i] = y_f32[i] * inv_scale

    packed_lo = cvt_e2m1x8_f32(q[0], q[1], q[2], q[3], q[4], q[5], q[6], q[7])
    packed_hi = cvt_e2m1x8_f32(q[8], q[9], q[10], q[11], q[12], q[13], q[14], q[15])
    return (Uint64(packed_hi) << Uint64(32)) | Uint64(packed_lo)


# =============================================================================
# FP4 Block Quantization Helpers (for MoE kernels)
# =============================================================================


@cute.jit
def quantize_block_fp4(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint8]:
    """Quantize 16 float32 values to packed FP4 + e4m3 scale byte.

    Given 16 values and their pre-computed max_abs, derives the NVFP4 block
    scale, quantizes to FP4, and packs into a uint64.  Returns
    (packed_fp4_u64, scale_byte).
    """
    scale_float = max_abs * global_scale_val / Float32(FLOAT4_E2M1_MAX)
    scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
    scale_u32 = cvt_f32_to_e4m3(scale_float)
    scale_byte = Uint8(scale_u32 & Uint32(0xFF))
    quantized_scale = fp8_e4m3_to_f32(scale_u32)
    packed64 = Uint64(0)
    if quantized_scale != Float32(0.0) and global_scale_val != Float32(0.0):
        # flashinfer's quantize_and_pack_16 multiplies values by an *inverse*
        # block scale (q = value * inv_scale). The dequantized block scale is
        # quantized_scale / global_scale_val, so the inverse passed here is its
        # reciprocal: global_scale_val / quantized_scale.
        packed64 = quantize_and_pack_16(values, global_scale_val / quantized_scale)
    return packed64, scale_byte


@cute.jit
def quantize_block_fp4_fast(
    values: cute.Tensor,
    max_abs: Float32,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint8]:
    """Fast approximate FP4 block quantization using reciprocal/vector path."""
    scale_u32 = Uint32(0)
    scale_byte = Uint8(0)
    packed64 = Uint64(0)
    if global_scale_val != Float32(0.0):
        fp4_max_rcp = rcp_approx_ftz(Float32(FLOAT4_E2M1_MAX))
        gs_recip = rcp_approx_ftz(global_scale_val)
        scale_float = gs_recip * (max_abs * fp4_max_rcp)
        scale_float = fmin_f32(scale_float, Float32(FLOAT8_E4M3_MAX))
        scale_u32 = cvt_f32_to_e4m3(scale_float)
        scale_byte = Uint8(scale_u32 & Uint32(0xFF))
        inv_quantized_scale = fp8_e4m3_to_f32_and_rcp(scale_u32)
        if inv_quantized_scale != Float32(0.0):
            packed64 = quantize_and_pack_16_fast(values, inv_quantized_scale * gs_recip)
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

    Used in MoE kernel epilogue to fuse the activation function
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
) -> Tuple[Uint64, Uint8]:
    """Fused SiLU(gate)*up + FP4 quantize for 16 element pairs."""
    activated = silu_mul_16(gate, up)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


# =============================================================================
# ReLU2 Activation — ReLU(x)² for non-gated MoE (Nemotron-Super)
# =============================================================================


@cute.jit
def relu2_16(x: cute.Tensor) -> cute.Tensor:
    """Compute ReLU²(x) = max(0, x)² for 16 float32 values."""
    out = cute.make_rmem_tensor((16,), Float32)
    for i in cutlass.range_constexpr(16):
        v = fmax_f32(x[i], Float32(0.0))
        out[i] = v * v
    return out


@cute.jit
def relu2_quantize_block_fp4(
    x: cute.Tensor,
    global_scale_val: Float32,
) -> Tuple[Uint64, Uint8]:
    """Fused ReLU² + FP4 quantize for 16 float32 values."""
    activated = relu2_16(x)
    block_max = max_abs_16(activated)
    return quantize_block_fp4(activated, block_max, global_scale_val)


# =============================================================================
# Additional primitives ported from b12x (b12x/cute/fp4.py) to support the
# rebased SM120/SM121 two-backend MoE (dynamic + micro) and the W4A8 tier.
# Ported verbatim from b12x HEAD 5af873a; see flashinfer/fused_moe/cute_dsl.
# =============================================================================


@dsl_user_op
def prefetch_global_l2(base_ptr: Int64, *, loc=None, ip=None) -> None:
    """Prefetch a global memory line into L2."""
    llvm.inline_asm(
        None,
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "prefetch.global.L2 [$0];",
        "l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_global_v4_f32(
    base_ptr: Int64,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from global memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def ldmatrix_m8n8x4_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x4.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32(), T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.i32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.i32(), result, [3], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1), Uint32(r2), Uint32(r3)


@dsl_user_op
def ldmatrix_m8n8x2_b16(
    smem_addr: Int32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Issue `ldmatrix.sync.aligned.m8n8.x2.shared.b16` from a shared-memory byte address."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Int32(smem_addr).ir_value(loc=loc, ip=ip)],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(r0), Uint32(r1)


@dsl_user_op
def cp_async_u32_shared_global(
    smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None
):
    """4-byte `cp.async.ca.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 4;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async_u64_shared_global(
    smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None
):
    """8-byte `cp.async.ca.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.ca.shared.global [$0], [$1], 8;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async4_shared_global(smem_addr: Int32, gmem_addr: Int64, *, loc=None, ip=None):
    """16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "cp.async.cg.shared.global [$0], [$1], 16;",
        "r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def cp_async4_shared_global_pred(
    smem_addr: Int32, gmem_addr: Int64, pred: Int32, *, loc=None, ip=None
):
    """Predicated 16-byte `cp.async.cg.shared.global` copy."""
    llvm.inline_asm(
        None,
        [
            Int32(pred).ir_value(loc=loc, ip=ip),
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Int64(gmem_addr).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .pred p; setp.ne.b32 p, $0, 0; @p cp.async.cg.shared.global [$1], [$2], 16; }",
        "r,r,l",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_global_release_i32(addr: Int64, val: Int32, *, loc=None, ip=None):
    """No-return global int32 add with a GPU-scope release fence."""
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Int32(val).ir_value(loc=loc, ip=ip),
        ],
        "fence.acq_rel.gpu;\nred.relaxed.gpu.global.add.s32 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_add_global_bf16x2(addr: Int64, packed: Uint32, *, loc=None, ip=None):
    """No-return global atomic add of a packed bf16x2 value (2 contiguous bf16).

    Used by the W4A16 TC-decode FC2 epilogue to fold the per-route partial
    outputs into the per-token output without a separate top-k-sum launch.
    The address must be 4-byte aligned and cover two consecutive bf16 lanes.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Uint32(packed).ir_value(loc=loc, ip=ip),
        ],
        "red.relaxed.gpu.global.add.noftz.bf16x2 [$0], $1;",
        "l,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def red_max_global_f32_nonnegative(addr: Int64, val: Float32, *, loc=None, ip=None):
    """No-return global max reduction for a non-negative fp32 value.

    Non-negative IEEE-754 float ordering matches unsigned integer bit ordering,
    so this emits a no-return u32 max reduction on the value's bit pattern.
    Callers must initialize the destination with a non-negative fp32 value.
    """
    llvm.inline_asm(
        None,
        [
            Int64(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .u32 vi; mov.b32 vi, $1; red.relaxed.gpu.global.max.u32 [$0], vi; }",
        "l,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def atomic_add_shared_i32(addr: Int32, val: Int32, *, loc=None, ip=None) -> Int32:
    """Shared-memory int32 atomic add (CTA-scope). Returns old value.

    Uses ``atom.shared.add.s32`` with a 32-bit shared-memory address
    (the native address width for smem on NVIDIA GPUs).
    """
    return Int32(
        llvm.inline_asm(
            T.i32(),
            [
                Int32(addr).ir_value(loc=loc, ip=ip),
                Int32(val).ir_value(loc=loc, ip=ip),
            ],
            "atom.shared.add.s32 $0, [$1], $2;",
            "=r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_shared_u32(addr: Int32, *, loc=None, ip=None) -> Uint32:
    """Load uint32 from shared memory at a 32-bit byte address."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Int32(addr).ir_value(loc=loc, ip=ip)],
            "ld.shared.u32 $0, [$1];",
            "=r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def st_shared_u32(addr: Int32, val: Uint32, *, loc=None, ip=None):
    """Store uint32 to shared memory at a 32-bit byte address."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Uint32(val).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.u32 [$0], $1;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def ld_shared_v4_f32(
    addr: Int32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Load 128 bits (4 x float32) from shared memory."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int32(addr).ir_value(loc=loc, ip=ip)],
        "ld.shared.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def broadcast_f32_to_half2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of an f16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.f16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def bfloat2_broadcast_lane(x: Uint32, lane: Int32, *, loc=None, ip=None) -> Uint32:
    """Duplicate one BF16 lane from a packed bf16x2 register into both lanes."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Uint32(x).ir_value(loc=loc, ip=ip),
                Int32(lane).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .pred p;
                .reg .b32 lo, hi, val, shifted;
                and.b32 lo, $1, 0x0000ffff;
                shr.u32 hi, $1, 16;
                setp.eq.s32 p, $2, 0;
                @p  mov.b32 val, lo;
                @!p mov.b32 val, hi;
                shl.b32 shifted, val, 16;
                or.b32 $0, val, shifted;
            }
            """,
            "=r,r,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def broadcast_f32_to_bfloat2(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack one float32 value into both lanes of a bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cvt.rn.satfinite.bf16x2.f32 $0, $1, $1;",
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def pack_f32x2_to_bfloat2(x0: Float32, x1: Float32, *, loc=None, ip=None) -> Uint32:
    """Pack 2 float32 values into one bf16x2 register."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(x0).ir_value(loc=loc, ip=ip),
                Float32(x1).ir_value(loc=loc, ip=ip),
            ],
            "cvt.rn.satfinite.bf16x2.f32 $0, $2, $1;",
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def packed_dequant_e2m1x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> BF16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 6;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e2m1x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE2M1 -> FP16 register dequant for one packed 4-value fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 out1, $2, 0x80008000;
            and.b32 tmp, $2, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 4;
            and.b32 out2, q, 0x80008000;
            and.b32 tmp, q, 0x70007000;
            shr.u32 tmp, tmp, 3;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value BF16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2, tmp;

            and.b32 tmp, $2, 0x80008000;
            shr.u32 out1, tmp, 1;
            and.b32 tmp, $2, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out1, out1, tmp;

            shl.b32 q, $2, 8;
            and.b32 tmp, q, 0x80008000;
            shr.u32 out2, tmp, 1;
            and.b32 tmp, q, 0x7F007F00;
            shr.u32 tmp, tmp, 4;
            or.b32 out2, out2, tmp;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e4m3x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """FE4M3 scale dequant for one packed 4-value FP16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 q, out1, out2;

            and.b32 out1, $2, 0xFF00FF00;
            shr.u32 out1, out1, 1;

            shl.b32 q, $2, 8;
            and.b32 out2, q, 0xFF00FF00;
            shr.u32 out2, out2, 1;

            mov.b32 $0, out2;
            mov.b32 $1, out1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e8m0x4_to_bfloat2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """E8M0 compute-scale dequant for one packed 4-value BF16 fragment.

    The W4A16 FP4 unpack path represents E2M1 values scaled by 2^-126.  Match
    the existing NVFP4 split by materializing E8M0 scales multiplied by 2^7 in
    the MMA input and applying the remaining compensation in the kernel
    epilogue.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .u32 b0, b1, b2, b3;
            .reg .u32 h0, h1, h2, h3;
            .reg .u32 t0, t1;

            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;

            add.u32 h0, b0, 7;
            add.u32 h1, b1, 7;
            add.u32 h2, b2, 7;
            add.u32 h3, b3, 7;
            shl.b32 h0, h0, 7;
            shl.b32 h1, h1, 7;
            shl.b32 h2, h2, 7;
            shl.b32 h3, h3, 7;

            shl.b32 t0, h2, 16;
            or.b32 $0, h0, t0;
            shl.b32 t1, h3, 16;
            or.b32 $1, h1, t1;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def packed_dequant_e8m0x4_to_half2x2(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """E8M0 compute-scale dequant for one packed 4-value FP16 fragment."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .u32 b0, b1, b2, b3;
            .reg .s32 e0, e1, e2, e3;
            .reg .f32 ef0, ef1, ef2, ef3;
            .reg .f32 f0, f1, f2, f3;
            .reg .b16 h0, h1, h2, h3;

            and.b32 b0, $2, 0x000000ff;
            shr.u32 b1, $2, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $2, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $2, 24;

            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;

            cvt.s32.u32 e0, b0;
            cvt.s32.u32 e1, b1;
            cvt.s32.u32 e2, b2;
            cvt.s32.u32 e3, b3;
            sub.s32 e0, e0, 120;
            sub.s32 e1, e1, 120;
            sub.s32 e2, e2, 120;
            sub.s32 e3, e3, 120;

            cvt.rn.f32.s32 ef0, e0;
            cvt.rn.f32.s32 ef1, e1;
            cvt.rn.f32.s32 ef2, e2;
            cvt.rn.f32.s32 ef3, e3;
            ex2.approx.f32 f0, ef0;
            ex2.approx.f32 f1, ef1;
            ex2.approx.f32 f2, ef2;
            ex2.approx.f32 f3, ef3;
            selp.f32 f0, 0f00000000, f0, p0;
            selp.f32 f1, 0f00000000, f1, p1;
            selp.f32 f2, 0f00000000, f2, p2;
            selp.f32 f3, 0f00000000, f3, p3;

            cvt.rn.f16.f32 h0, f0;
            cvt.rn.f16.f32 h1, f1;
            cvt.rn.f16.f32 h2, f2;
            cvt.rn.f16.f32 h3, f3;

            setp.eq.u32 p0, b0, 255;
            setp.eq.u32 p1, b1, 255;
            setp.eq.u32 p2, b2, 255;
            setp.eq.u32 p3, b3, 255;
            selp.b16 h0, 0x7e00, h0, p0;
            selp.b16 h1, 0x7e00, h1, p1;
            selp.b16 h2, 0x7e00, h2, p2;
            selp.b16 h3, 0x7e00, h3, p3;

            mov.b32 $0, {h0, h2};
            mov.b32 $1, {h1, h3};
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), result, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def bf16_mma_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def f32_to_tf32_bits(x: Float32, *, loc=None, ip=None) -> Uint32:
    """Round one float32 value to TF32 format and return its 32-bit operand bits."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .b32 tmp;
                cvt.rna.tf32.f32 tmp, $1;
                mov.b32 $0, tmp;
            }
            """,
            "=r,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def tf32_mma_m16n8k8_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def f16_mma_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def bf16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    b0_0: Uint32,
    b1_0: Uint32,
    b0_1: Uint32,
    b1_1: Uint32,
    a0: Uint32,
    a1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """BF16 MMA form used by the routed m-block-size-8 path.

    The dequantized RHS fragments feed the hardware A operand, while the
    routed activation fragment loaded with `ldmatrix.x2` feeds hardware B.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(b0_0).ir_value(loc=loc, ip=ip),
            Uint32(b1_0).ir_value(loc=loc, ip=ip),
            Uint32(b0_1).ir_value(loc=loc, ip=ip),
            Uint32(b1_1).ir_value(loc=loc, ip=ip),
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def f16_mma_rhs_fragments_as_mma_a_m16n8k16_f32(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    b0_0: Uint32,
    b1_0: Uint32,
    b0_1: Uint32,
    b1_1: Uint32,
    a0: Uint32,
    a1: Uint32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """FP16 MMA form used by the routed m-block-size-8 path."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(b0_0).ir_value(loc=loc, ip=ip),
            Uint32(b1_0).ir_value(loc=loc, ip=ip),
            Uint32(b0_1).ir_value(loc=loc, ip=ip),
            Uint32(b1_1).ir_value(loc=loc, ip=ip),
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$10, $11, $12, $13};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def mxfp8_mma_m16n8k32_f32_e4m3(
    d0: Float32,
    d1: Float32,
    d2: Float32,
    d3: Float32,
    a0: Uint32,
    a1: Uint32,
    a2: Uint32,
    a3: Uint32,
    b0: Uint32,
    b1: Uint32,
    sfa: Uint32,
    sfb: Uint32,
    bid_a: int = 0,
    tid_a: int = 0,
    bid_b: int = 0,
    tid_b: int = 0,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Warp MMA helper for SM120 MXFP8 block-scaled `m16n8k32` E4M3/E4M3."""
    i16_ty = cutlass._mlir.ir.IntegerType.get_signless(16)
    bid_a_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(bid_a))},
    ).result
    tid_a_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(tid_a))},
    ).result
    bid_b_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(bid_b))},
    ).result
    tid_b_i16 = cutlass._mlir.ir.Operation.create(
        "llvm.mlir.constant",
        results=[i16_ty],
        attributes={"value": cutlass._mlir.ir.IntegerAttr.get(i16_ty, int(tid_b))},
    ).result
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a0).ir_value(loc=loc, ip=ip),
            Uint32(a1).ir_value(loc=loc, ip=ip),
            Uint32(a2).ir_value(loc=loc, ip=ip),
            Uint32(a3).ir_value(loc=loc, ip=ip),
            Uint32(b0).ir_value(loc=loc, ip=ip),
            Uint32(b1).ir_value(loc=loc, ip=ip),
            Uint32(sfa).ir_value(loc=loc, ip=ip),
            bid_a_i16,
            tid_a_i16,
            Uint32(sfb).ir_value(loc=loc, ip=ip),
            bid_b_i16,
            tid_b_i16,
            Float32(d0).ir_value(loc=loc, ip=ip),
            Float32(d1).ir_value(loc=loc, ip=ip),
            Float32(d2).ir_value(loc=loc, ip=ip),
            Float32(d3).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0
        {$0, $1, $2, $3},
        {$4, $5, $6, $7},
        {$8, $9},
        {$0, $1, $2, $3},
        {$10},
        {$11, $12},
        {$13},
        {$14, $15};
        """,
        "=f,=f,=f,=f,r,r,r,r,r,r,r,h,h,r,h,h,0,1,2,3",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    r0 = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    r1 = llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)
    r2 = llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)
    r3 = llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)
    return Float32(r0), Float32(r1), Float32(r2), Float32(r3)


@dsl_user_op
def cvt_f32x4_to_e4m3x4(
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
) -> Uint32:
    """Convert 4 float32 values to 4 packed E4M3 bytes (v0 in the low byte)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(v0).ir_value(loc=loc, ip=ip),
                Float32(v1).ir_value(loc=loc, ip=ip),
                Float32(v2).ir_value(loc=loc, ip=ip),
                Float32(v3).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b16 lo, hi;
                cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1;
                cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3;
                mov.b32 $0, {lo, hi};
            }
            """,
            "=r,f,f,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e8m0_to_f32(e8m0_val: Uint32, *, loc=None, ip=None) -> Float32:
    """Convert a single E8M0 scale byte to its true f32 value 2**(byte-127).

    Byte 0 maps to 0.0. Unlike packed_dequant_e8m0x4_* (which fold a 2**7 bias
    into the MMA input), this returns the unbiased scale so callers can apply it
    directly in an f32 accumulator.
    """
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Uint32(e8m0_val).ir_value(loc=loc, ip=ip)],
            """
            {
                .reg .pred p0;
                .reg .u32 b0;
                .reg .s32 e0;
                .reg .f32 ef0;
                and.b32 b0, $1, 0x000000ff;
                setp.eq.u32 p0, b0, 0;
                cvt.s32.u32 e0, b0;
                sub.s32 e0, e0, 127;
                cvt.rn.f32.s32 ef0, e0;
                ex2.approx.f32 $0, ef0;
                selp.f32 $0, 0f00000000, $0, p0;
            }
            """,
            "=f,r",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def cvt_e8m0x4_to_f32x4(
    packed: Uint32, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """Decode 4 E8M0 scale bytes (packed u32) to 4 x true f32 (2**(byte-127))."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Uint32(packed).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred p0, p1, p2, p3;
            .reg .u32 b0, b1, b2, b3;
            .reg .s32 e0, e1, e2, e3;
            .reg .f32 ef0, ef1, ef2, ef3;
            and.b32 b0, $4, 0x000000ff;
            shr.u32 b1, $4, 8;
            and.b32 b1, b1, 0x000000ff;
            shr.u32 b2, $4, 16;
            and.b32 b2, b2, 0x000000ff;
            shr.u32 b3, $4, 24;
            setp.eq.u32 p0, b0, 0;
            setp.eq.u32 p1, b1, 0;
            setp.eq.u32 p2, b2, 0;
            setp.eq.u32 p3, b3, 0;
            cvt.s32.u32 e0, b0;
            cvt.s32.u32 e1, b1;
            cvt.s32.u32 e2, b2;
            cvt.s32.u32 e3, b3;
            sub.s32 e0, e0, 127;
            sub.s32 e1, e1, 127;
            sub.s32 e2, e2, 127;
            sub.s32 e3, e3, 127;
            cvt.rn.f32.s32 ef0, e0;
            cvt.rn.f32.s32 ef1, e1;
            cvt.rn.f32.s32 ef2, e2;
            cvt.rn.f32.s32 ef3, e3;
            ex2.approx.f32 $0, ef0;
            ex2.approx.f32 $1, ef1;
            ex2.approx.f32 $2, ef2;
            ex2.approx.f32 $3, ef3;
            selp.f32 $0, 0f00000000, $0, p0;
            selp.f32 $1, 0f00000000, $1, p1;
            selp.f32 $2, 0f00000000, $2, p2;
            selp.f32 $3, 0f00000000, $3, p3;
        }
        """,
        "=f,=f,=f,=f,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return (
        Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip)),
        Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip)),
    )


# =============================================================================
# FP4 (E2M1) Decode Intrinsics
# =============================================================================


@dsl_user_op
def e2m1x8_to_e4m3x8(packed_u32: Uint32, *, loc=None, ip=None) -> Tuple[Uint32, Uint32]:
    """Expand 8 packed E2M1 nibbles into 8 E4M3 bytes, losslessly.

    Every E2M1 value is exactly representable in E4M3, so this is a pure
    relabeling done with two prmt magnitude-LUT lookups plus a sign pass.
    Nibble i of the input becomes byte i of the (lo, hi) output pair.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [Uint32(packed_u32).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 clo, chi, cslo, cshi, mlo, mhi, slo, shi;
            .reg .b32 tbl_a, tbl_b, tbl_s, zero;
            // E4M3 encodings of {0, .5, 1, 1.5, 2, 3, 4, 6}.
            mov.b32 tbl_a, 0x3C383000;
            mov.b32 tbl_b, 0x4C484440;
            mov.b32 tbl_s, 0x00008000;
            mov.b32 zero, 0;
            // Magnitude codes (low 3 bits of each nibble) as prmt selectors.
            and.b32 clo, $2, 0x00007777;
            shr.u32 chi, $2, 16;
            and.b32 chi, chi, 0x00007777;
            prmt.b32 mlo, tbl_a, tbl_b, clo;
            prmt.b32 mhi, tbl_a, tbl_b, chi;
            // Sign bit (nibble bit 3) -> byte bit 7 via a 2-entry LUT.
            shr.u32 cslo, $2, 3;
            and.b32 cslo, cslo, 0x00001111;
            shr.u32 cshi, $2, 19;
            and.b32 cshi, cshi, 0x00001111;
            prmt.b32 slo, tbl_s, zero, cslo;
            prmt.b32 shi, tbl_s, zero, cshi;
            or.b32 $0, mlo, slo;
            or.b32 $1, mhi, shi;
        }
        """,
        "=r,=r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def e2m1x8_mul_residual_to_e4m3x8(
    packed_u32: Uint32, residual_h2: Uint32, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32]:
    """Expand 8 E2M1 nibbles to E4M3 bytes with a residual multiplier.

    Decodes to f16 pairs, multiplies by ``residual_h2`` (the same residual
    broadcast to both f16 lanes), and rounds to E4M3.  Used by the w4a8
    NVFP4 path where the per-K/16 e4m3 scale is decomposed into a shared
    UE8M0 hardware exponent and this in-register residual.
    """
    res = llvm.inline_asm(
        llvm.StructType.get_literal([T.i32(), T.i32()]),
        [
            Uint32(packed_u32).ir_value(loc=loc, ip=ip),
            Uint32(residual_h2).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 b0, b1, b2, b3;
            .reg .b32 h0, h1, h2, h3;
            .reg .b16 e0, e1, e2, e3;
            mov.b32 {b0, b1, b2, b3}, $2;
            cvt.rn.f16x2.e2m1x2 h0, b0;
            cvt.rn.f16x2.e2m1x2 h1, b1;
            cvt.rn.f16x2.e2m1x2 h2, b2;
            cvt.rn.f16x2.e2m1x2 h3, b3;
            mul.f16x2 h0, h0, $3;
            mul.f16x2 h1, h1, $3;
            mul.f16x2 h2, h2, $3;
            mul.f16x2 h3, h3, $3;
            cvt.rn.satfinite.e4m3x2.f16x2 e0, h0;
            cvt.rn.satfinite.e4m3x2.f16x2 e1, h1;
            cvt.rn.satfinite.e4m3x2.f16x2 e2, h2;
            cvt.rn.satfinite.e4m3x2.f16x2 e3, h3;
            mov.b32 $0, {e0, e1};
            mov.b32 $1, {e2, e3};
        }
        """,
        "=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    lo = llvm.extractvalue(T.i32(), res, [0], loc=loc, ip=ip)
    hi = llvm.extractvalue(T.i32(), res, [1], loc=loc, ip=ip)
    return Uint32(lo), Uint32(hi)


@dsl_user_op
def pow2_ceil_ue8m0(
    scale: Float32,
    *,
    loc=None,
    ip=None,
) -> Tuple[Float32, Uint32]:
    """Round a positive FP32 ``scale`` UP to a power of two, bit-exactly.

    Returns ``(rounded_fp32, ue8m0_byte)`` where:
      * ``rounded_fp32`` == ``__uint_as_float(power_of_2_round(__float_as_uint(scale)))``
      * ``ue8m0_byte``   == ``(__float_as_uint(rounded_fp32) >> 23) & 0xFF``

    Bit-exact replica of FlashInfer's fp8_quant.cuh rounding +
    scale_convert.cuh fp32_to_ue8m0. Integer ops only (no lg2.approx), so unlike
    ``cvt_f32_to_ue8m0`` it matches the FlashInfer reference bit-for-bit.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.i32()]),
        [Float32(scale).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .pred  p_mant;
            .reg .b32   bits, mant;

            // bits = __float_as_uint(scale)
            mov.b32 bits, $2;
            // mant = bits & 0x007FFFFF  (mantissa field)
            and.b32 mant, bits, 8388607;
            setp.ne.u32 p_mant, mant, 0;
            // if (mant) bits = (bits + 0x00800000) & 0x7F800000
            @p_mant add.u32 bits, bits, 8388608;
            @p_mant and.b32 bits, bits, 2139095040;
            // rounded fp32 scale = __uint_as_float(bits)
            mov.b32 $0, bits;
            // ue8m0 = (bits >> 23) & 0xFF
            bfe.u32 $1, bits, 23, 8;
        }
        """,
        "=f,=r,f",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    rounded = llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip)
    ue8m0 = llvm.extractvalue(T.i32(), result, [1], loc=loc, ip=ip)
    return Float32(rounded), Uint32(ue8m0)


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


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def as_grouped_scale_view(
    scale_storage: torch.Tensor, rows: int, cols: int
) -> torch.Tensor:
    batch = scale_storage.shape[0]
    rows_padded = align_up(rows, 128)
    cols_padded = align_up(cols // SF_VEC_SIZE, 4)
    sf = scale_storage.view(torch.float8_e4m3fn)
    sf = sf.view(batch, rows_padded // 128, cols_padded // 4, 32, 4, 4)
    return sf.permute(3, 4, 1, 5, 2, 0)


def _fp4_quantize_values(x: torch.Tensor) -> torch.Tensor:
    sign = torch.sign(x)
    x = torch.abs(x.clone())
    x[(x >= 0.0) & (x <= 0.25)] = 0.0
    x[(x > 0.25) & (x < 0.75)] = 0.5
    x[(x >= 0.75) & (x <= 1.25)] = 1.0
    x[(x > 1.25) & (x < 1.75)] = 1.5
    x[(x >= 1.75) & (x <= 2.5)] = 2.0
    x[(x > 2.5) & (x < 3.5)] = 3.0
    x[(x >= 3.5) & (x <= 5.0)] = 4.0
    x[x > 5.0] = 6.0
    return x * sign


def fp4_quantize_values_torch(x: torch.Tensor) -> torch.Tensor:
    """Pure-Torch FP4 E2M1 quantization with kernel-matching tie-breaking."""
    return _fp4_quantize_values(x)


def pow2_ceil_ue8m0_torch(scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact Torch replica of the ``pow2_ceil_ue8m0`` device intrinsic.

    Rounds positive fp32 scales UP to a power of two by bumping the exponent
    whenever the mantissa is nonzero.  Returns ``(rounded_fp32, ue8m0_u8)``;
    a zero scale maps to (0.0, byte 0).
    """
    bits = scale.to(torch.float32).contiguous().view(torch.int32)
    mant = bits & 0x007FFFFF
    bumped = torch.where(mant != 0, (bits + 0x00800000) & 0x7F800000, bits)
    rounded = bumped.view(torch.float32)
    byte = ((bumped >> 23) & 0xFF).to(torch.uint8)
    return rounded, byte


def _ue8m0_output_scale_torch(byte: torch.Tensor) -> torch.Tensor:
    """Torch replica of ``ue8m0_to_output_scale``: 2^(127-byte), 0 for byte 0."""
    inv_bits = (254 - byte.to(torch.int32)).clamp(min=0) << 23
    inv = inv_bits.view(torch.float32)
    return torch.where(byte == 0, torch.zeros_like(inv), inv)


def quant_dequant_mxfp8_torch(x: torch.Tensor) -> torch.Tensor:
    """Per-32-block MXFP8 quantize-dequantize roundtrip (oracle helper).

    Matches the in-kernel ``quantize_block_fp8_mx`` numerics bit-for-bit:
    UE8M0 ceil block scale, E4M3 RN-saturating payload.
    """
    orig_shape = x.shape
    cols = orig_shape[-1]
    if cols % MX_SF_VEC_SIZE != 0:
        raise ValueError(f"last dim must be divisible by {MX_SF_VEC_SIZE}, got {cols}")
    blocked = x.to(torch.float32).reshape(-1, cols // MX_SF_VEC_SIZE, MX_SF_VEC_SIZE)
    block_max = blocked.abs().amax(dim=-1, keepdim=True)
    rounded, byte = pow2_ceil_ue8m0_torch(block_max * _INV_FLOAT8_E4M3_MAX)
    inv = _ue8m0_output_scale_torch(byte)
    payload = (
        (blocked * inv)
        .clamp(-FLOAT8_E4M3_MAX, FLOAT8_E4M3_MAX)
        .to(torch.float8_e4m3fn)
        .to(torch.float32)
    )
    return (payload * rounded).reshape(orig_shape)


@dsl_user_op
def st_global_v4_f32(
    base_ptr: Int64,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to global memory."""
    llvm.inline_asm(
        None,
        [
            Int64(base_ptr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.global.v4.f32 [$0], {$1, $2, $3, $4};",
        "l,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_v4_u32(
    smem_addr: Int32,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x uint32) to shared memory. smem_addr is a u32 shared-memory address."""
    llvm.inline_asm(
        None,
        [
            Int32(smem_addr).ir_value(loc=loc, ip=ip),
            Uint32(v0).ir_value(loc=loc, ip=ip),
            Uint32(v1).ir_value(loc=loc, ip=ip),
            Uint32(v2).ir_value(loc=loc, ip=ip),
            Uint32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.u32 [$0], {$1, $2, $3, $4};",
        "r,r,r,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@dsl_user_op
def st_shared_v4_f32(
    addr: Int32,
    v0: Float32,
    v1: Float32,
    v2: Float32,
    v3: Float32,
    *,
    loc=None,
    ip=None,
):
    """Store 128 bits (4 x float32) to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(v0).ir_value(loc=loc, ip=ip),
            Float32(v1).ir_value(loc=loc, ip=ip),
            Float32(v2).ir_value(loc=loc, ip=ip),
            Float32(v3).ir_value(loc=loc, ip=ip),
        ],
        "st.shared.v4.f32 [$0], {$1, $2, $3, $4};",
        "r,f,f,f,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_bf16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to BF16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.bf16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def st_shared_f16_from_f32(addr: Int32, val: Float32, *, loc=None, ip=None):
    """Convert float32 to FP16 and store one 16-bit value to shared memory."""
    llvm.inline_asm(
        None,
        [
            Int32(addr).ir_value(loc=loc, ip=ip),
            Float32(val).ir_value(loc=loc, ip=ip),
        ],
        "{ .reg .b16 tmp; cvt.rn.f16.f32 tmp, $1; st.shared.b16 [$0], tmp; }",
        "r,f",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
