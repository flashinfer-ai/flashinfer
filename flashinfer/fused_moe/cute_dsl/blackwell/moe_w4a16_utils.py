# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Low-level conversion helpers for the SM100 NVFP4/BF16 MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import BFloat16, Uint32, dsl_user_op


_HAS_DIRECT_FP4_TO_BF16_CVT = cutlass.CUDA_VERSION.major > 13 or (
    cutlass.CUDA_VERSION.major == 13 and cutlass.CUDA_VERSION.minor >= 2
)


@dsl_user_op
def e2m1x8_e4m3_to_bf16x8(
    packed_e2m1: Uint32,
    e4m3_scale: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[BFloat16, ...]:
    """Decode one eight-value half of an NVFP4 block to BF16.

    CUDA 13.2 adds direct E2M1/E4M3 to BF16 conversion forms. Earlier toolkits
    decode and multiply in FP16, then widen the exact NVFP4 product to BF16.
    Both paths convert the shared scale once per block.
    """
    if _HAS_DIRECT_FP4_TO_BF16_CVT:
        asm = """
        {
            .reg .b8 q0, q1, q2, q3, scale;
            .reg .b16 scale_pair;
            .reg .b32 scale_bf16x2;

            mov.b32 {q0, q1, q2, q3}, $4;
            mov.b32 {scale, _, _, _}, $5;
            mov.b16 scale_pair, {scale, scale};
            cvt.rn.bf16x2.e4m3x2 scale_bf16x2, scale_pair;

            cvt.rn.bf16x2.e2m1x2 $0, q0;
            cvt.rn.bf16x2.e2m1x2 $1, q1;
            cvt.rn.bf16x2.e2m1x2 $2, q2;
            cvt.rn.bf16x2.e2m1x2 $3, q3;
            mul.rn.bf16x2 $0, $0, scale_bf16x2;
            mul.rn.bf16x2 $1, $1, scale_bf16x2;
            mul.rn.bf16x2 $2, $2, scale_bf16x2;
            mul.rn.bf16x2 $3, $3, scale_bf16x2;
        }
        """
    else:
        asm = """
        {
            .reg .b8 q0, q1, q2, q3, scale;
            .reg .b16 scale_pair, lo, hi;
            .reg .b32 scale_f16x2;
            .reg .f32 lo_f32, hi_f32;

            mov.b32 {q0, q1, q2, q3}, $4;
            mov.b32 {scale, _, _, _}, $5;
            mov.b16 scale_pair, {scale, scale};
            cvt.rn.f16x2.e4m3x2 scale_f16x2, scale_pair;

            cvt.rn.f16x2.e2m1x2 $0, q0;
            cvt.rn.f16x2.e2m1x2 $1, q1;
            cvt.rn.f16x2.e2m1x2 $2, q2;
            cvt.rn.f16x2.e2m1x2 $3, q3;
            mul.rn.f16x2 $0, $0, scale_f16x2;
            mul.rn.f16x2 $1, $1, scale_f16x2;
            mul.rn.f16x2 $2, $2, scale_f16x2;
            mul.rn.f16x2 $3, $3, scale_f16x2;

            mov.b32 {lo, hi}, $0;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $0, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $1;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $1, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $2;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $2, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $3;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $3, hi_f32, lo_f32;

        }
        """

    result = llvm.inline_asm(
        llvm.StructType.get_literal([Uint32.mlir_type] * 4),
        [
            Uint32(packed_e2m1).ir_value(loc=loc, ip=ip),
            Uint32(e4m3_scale).ir_value(loc=loc, ip=ip),
        ],
        asm,
        "=r,=r,=r,=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed_type = ir.VectorType.get([4], Uint32.mlir_type, loc=loc)
    packed_values = [
        llvm.extractvalue(Uint32.mlir_type, result, [idx], loc=loc, ip=ip)
        for idx in range(4)
    ]
    packed = vector.from_elements(packed_type, packed_values, loc=loc, ip=ip)
    result_type = ir.VectorType.get([8], BFloat16.mlir_type, loc=loc)
    result_vector = llvm.bitcast(result_type, packed, loc=loc, ip=ip)
    result_values = [
        vector.extract(result_vector, [], [idx], loc=loc, ip=ip) for idx in range(8)
    ]
    return tuple(BFloat16(value) for value in result_values)


@dsl_user_op
def decode_nvfp4_fragment_to_bf16(
    packed_fragment: cute.TensorSSA,
    scale_fragment: cute.TensorSSA,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Decode one transform thread's eight-value NVFP4 half-block to BF16."""
    packed_byte_count = cute.size(packed_fragment.shape)
    fragment_size = packed_byte_count * 2
    assert fragment_size == 8
    assert cute.size(scale_fragment.shape) == 1

    packed_word_type = ir.VectorType.get([1], Uint32.mlir_type, loc=loc)
    packed_word_vector = llvm.bitcast(
        packed_word_type, packed_fragment.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    packed_word = vector.extract(packed_word_vector, [], [0], loc=loc, ip=ip)
    scale_byte = vector.extract(
        scale_fragment.ir_value(loc=loc, ip=ip), [], [0], loc=loc, ip=ip
    )
    scale_bits = llvm.zext(Uint32.mlir_type, scale_byte, loc=loc, ip=ip)
    output_values = [
        value.ir_value(loc=loc, ip=ip)
        for value in e2m1x8_e4m3_to_bf16x8(
            Uint32(packed_word),
            Uint32(scale_bits),
            loc=loc,
            ip=ip,
        )
    ]

    output_type = ir.VectorType.get([fragment_size], BFloat16.mlir_type, loc=loc)
    output = vector.from_elements(output_type, output_values, loc=loc, ip=ip)
    return cute.TensorSSA(output, (fragment_size,), cutlass.BFloat16)


__all__ = ["decode_nvfp4_fragment_to_bf16", "e2m1x8_e4m3_to_bf16x8"]
