"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

PTX Intrinsics for CuTe-DSL RoPE Kernels
========================================

This module provides low-level PTX intrinsics used by the RoPE kernels:
- Math operations: sin_approx, cos_approx, ex2_approx, lg2_approx
- Memory operations: ld_global_v4_u32, st_global_v4_u32, ld_global_f32
- Type conversions: half2_to_float2, float2_to_half2, bfloat2_to_float2, float2_to_bfloat2
"""

from typing import Tuple

import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm


# =============================================================================
# Math PTX Intrinsics
# =============================================================================


@dsl_user_op
def sin_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute approximate sin using PTX sin.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "sin.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def cos_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute approximate cos using PTX cos.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cos.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def ex2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute 2^x using PTX ex2.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "ex2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


@dsl_user_op
def lg2_approx(x: Float32, *, loc=None, ip=None) -> Float32:
    """Compute log2(x) using PTX lg2.approx.f32."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "lg2.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


def powf_approx(base: Float32, exp: Float32) -> Float32:
    """Compute base^exp using 2^(exp * log2(base))."""
    return ex2_approx(exp * lg2_approx(base))


@dsl_user_op
def sincos_approx(x: Float32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """
    Compute approximate sin and cos simultaneously using PTX.

    Returns (sin_val, cos_val).
    """
    sin_val = Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "sin.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    cos_val = Float32(
        llvm.inline_asm(
            T.f32(),
            [Float32(x).ir_value(loc=loc, ip=ip)],
            "cos.approx.f32 $0, $1;",
            "=f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )
    return sin_val, cos_val


@dsl_user_op
def fmaxf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute max(a, b) for float32."""
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
def fminf(a: Float32, b: Float32, *, loc=None, ip=None) -> Float32:
    """Compute min(a, b) for float32."""
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


def clamp_f32(val: Float32, lo: Float32, hi: Float32) -> Float32:
    """Clamp val to [lo, hi]."""
    return fmaxf(lo, fminf(hi, val))


# =============================================================================
# Memory PTX Intrinsics
# =============================================================================


@dsl_user_op
def get_ptr_as_int64(tensor: cute.Tensor, offset: Int32, *, loc=None, ip=None) -> Int64:
    """Get the memory address of tensor[offset] as Int64 for PTX instructions."""
    elem_ptr = tensor.iterator + Int32(offset)
    ptr_int = llvm.ptrtoint(T.i64(), elem_ptr.llvm_ptr, loc=loc, ip=ip)
    return Int64(ptr_int)


@dsl_user_op
def ld_global_f32(base_ptr: Int64, *, loc=None, ip=None) -> Float32:
    """Load a single float32 from global memory."""
    return Float32(
        llvm.inline_asm(
            T.f32(),
            [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
            "ld.global.f32 $0, [$1];",
            "=f,l",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
            loc=loc,
            ip=ip,
        )
    )


@dsl_user_op
def ld_global_v4_f32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Float32, Float32, Float32, Float32]:
    """
    Load 128 bits (4 x float32) from global memory.

    This is the vectorized version of ld_global_f32, loading 4 consecutive
    float32 values in a single 128-bit memory transaction.

    Parameters
    ----------
    base_ptr : Int64
        Memory address (must be 16-byte aligned for optimal performance)

    Returns
    -------
    Tuple[Float32, Float32, Float32, Float32]
        Four consecutive float32 values from memory
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [Int64(base_ptr).ir_value(loc=loc, ip=ip)],
        "ld.global.v4.f32 {$0, $1, $2, $3}, [$4];",
        "=f,=f,=f,=f,l",
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
def ld_global_v4_u32(
    base_ptr: Int64, *, loc=None, ip=None
) -> Tuple[Uint32, Uint32, Uint32, Uint32]:
    """
    Load 128 bits (4 x uint32) from global memory.

    For fp16: loads 8 elements (4 half2 pairs)
    For bf16: loads 8 elements (4 bfloat2 pairs)
    """
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
def st_global_v4_u32(
    base_ptr: Int64,
    v0: Uint32,
    v1: Uint32,
    v2: Uint32,
    v3: Uint32,
    *,
    loc=None,
    ip=None,
) -> None:
    """
    Store 128 bits (4 x uint32) to global memory.

    For fp16: stores 8 elements (4 half2 pairs)
    For bf16: stores 8 elements (4 bfloat2 pairs)
    """
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
    )


# =============================================================================
# Type Conversion PTX Intrinsics
# =============================================================================


@dsl_user_op
def half2_to_float2(h2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert half2 (2 x fp16 packed in uint32) to 2 x float32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(h2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b16 h0, h1;
            mov.b32 {h0, h1}, $2;
            cvt.f32.f16 $0, h0;
            cvt.f32.f16 $1, h1;
        }
        """,
        "=f,=f,r",
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
def float2_to_half2(f0: Float32, f1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x float32 to half2 (2 x fp16 packed in uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
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
def bfloat2_to_float2(bf2: Uint32, *, loc=None, ip=None) -> Tuple[Float32, Float32]:
    """Convert bfloat2 (2 x bf16 packed in uint32) to 2 x float32."""
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32()]),
        [Uint32(bf2).ir_value(loc=loc, ip=ip)],
        """
        {
            .reg .b32 lo, hi;
            and.b32 lo, $2, 0xFFFF;
            shr.b32 hi, $2, 16;
            shl.b32 lo, lo, 16;
            shl.b32 hi, hi, 16;
            mov.b32 $0, lo;
            mov.b32 $1, hi;
        }
        """,
        "=f,=f,r",
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
def float2_to_bfloat2(f0: Float32, f1: Float32, *, loc=None, ip=None) -> Uint32:
    """Convert 2 x float32 to bfloat2 (2 x bf16 packed in uint32)."""
    return Uint32(
        llvm.inline_asm(
            T.i32(),
            [
                Float32(f0).ir_value(loc=loc, ip=ip),
                Float32(f1).ir_value(loc=loc, ip=ip),
            ],
            """
            {
                .reg .b32 lo, hi;
                shr.b32 lo, $1, 16;
                and.b32 hi, $2, 0xFFFF0000;
                or.b32 $0, lo, hi;
            }
            """,
            "=r,f,f",
            has_side_effects=False,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    )


__all__ = [
    # Math operations
    "sin_approx",
    "cos_approx",
    "ex2_approx",
    "lg2_approx",
    "powf_approx",
    "sincos_approx",
    "fmaxf",
    "fminf",
    "clamp_f32",
    # Memory operations
    "get_ptr_as_int64",
    "ld_global_f32",
    "ld_global_v4_f32",
    "ld_global_v4_u32",
    "st_global_v4_u32",
    # Type conversions
    "half2_to_float2",
    "float2_to_half2",
    "bfloat2_to_float2",
    "float2_to_bfloat2",
]
