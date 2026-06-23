"""b12x-MoE-specific FP4 CuTe-DSL primitives.

Moved out of flashinfer/cute_dsl/fp4_common.py (which stays focused on the
rmsnorm-shared helpers): these primitives and quantize_block_fp4 are used only
by the SM120/SM121 b12x MoE kernels under flashinfer/fused_moe/cute_dsl/.
Shared helpers are imported from fp4_common.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64, Uint8, Uint32, Uint64
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import llvm

from flashinfer.cute_dsl.fp4_common import (
    FLOAT4_E2M1_MAX,
    FLOAT8_E4M3_MAX,
    SF_VEC_SIZE,
    cvt_f32_to_e4m3,
    fmin_f32,
    fp8_e4m3_to_f32,
    max_abs_16,
    quantize_and_pack_16,
    relu2_16,
    silu_mul_16,
    ue8m0_to_output_scale,
)


MX_SF_VEC_SIZE = 32  # Elements per UE8M0 scale block (MXFP8 w4a8 activations)


_INV_FLOAT8_E4M3_MAX = 1.0 / FLOAT8_E4M3_MAX


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
