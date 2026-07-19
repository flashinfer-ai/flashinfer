# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Low-level conversion helpers for the SM100 NVFP4/BF16 MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import BFloat16, Uint8, Uint32, dsl_user_op


@dsl_user_op
def e2m1x16_e4m3_to_bf16x16(
    packed_e2m1_lo: Uint32,
    packed_e2m1_hi: Uint32,
    e4m3_scale: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[BFloat16, ...]:
    """Decode one 16-value NVFP4 block to BF16.

    CUDA 13.2 adds the direct E2M1/E4M3 to BF16 conversion forms. Keeping the
    shared scale and eight packed value pairs in one PTX region avoids redundant
    scale conversion before staging the BF16 ``tcgen05.mma`` operand.
    """
    result = llvm.inline_asm(
        llvm.StructType.get_literal([Uint32.mlir_type] * 8),
        [
            Uint32(packed_e2m1_lo).ir_value(loc=loc, ip=ip),
            Uint32(packed_e2m1_hi).ir_value(loc=loc, ip=ip),
            Uint32(e4m3_scale).ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 q0, q1, q2, q3, q4, q5, q6, q7, scale;
            .reg .b16 scale_pair;
            .reg .b32 scale_bf16x2;

            mov.b32 {q0, q1, q2, q3}, $8;
            mov.b32 {q4, q5, q6, q7}, $9;
            mov.b32 {scale, _, _, _}, $10;
            mov.b16 scale_pair, {scale, scale};
            cvt.rn.bf16x2.e4m3x2 scale_bf16x2, scale_pair;

            cvt.rn.bf16x2.e2m1x2 $0, q0;
            cvt.rn.bf16x2.e2m1x2 $1, q1;
            cvt.rn.bf16x2.e2m1x2 $2, q2;
            cvt.rn.bf16x2.e2m1x2 $3, q3;
            cvt.rn.bf16x2.e2m1x2 $4, q4;
            cvt.rn.bf16x2.e2m1x2 $5, q5;
            cvt.rn.bf16x2.e2m1x2 $6, q6;
            cvt.rn.bf16x2.e2m1x2 $7, q7;
            mul.rn.bf16x2 $0, $0, scale_bf16x2;
            mul.rn.bf16x2 $1, $1, scale_bf16x2;
            mul.rn.bf16x2 $2, $2, scale_bf16x2;
            mul.rn.bf16x2 $3, $3, scale_bf16x2;
            mul.rn.bf16x2 $4, $4, scale_bf16x2;
            mul.rn.bf16x2 $5, $5, scale_bf16x2;
            mul.rn.bf16x2 $6, $6, scale_bf16x2;
            mul.rn.bf16x2 $7, $7, scale_bf16x2;
        }
        """,
        "=r,=r,=r,=r,=r,=r,=r,=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    packed_type = ir.VectorType.get([8], Uint32.mlir_type, loc=loc)
    packed_values = [
        llvm.extractvalue(Uint32.mlir_type, result, [idx], loc=loc, ip=ip)
        for idx in range(8)
    ]
    packed = vector.from_elements(packed_type, packed_values, loc=loc, ip=ip)
    result_type = ir.VectorType.get([16], BFloat16.mlir_type, loc=loc)
    result_values = vector.to_elements(
        llvm.bitcast(result_type, packed, loc=loc, ip=ip), loc=loc, ip=ip
    )
    return tuple(BFloat16(value) for value in result_values)


@dsl_user_op
def decode_nvfp4_fragment_to_bf16(
    packed_fragment: cute.TensorSSA,
    scale_fragment: cute.TensorSSA,
    *,
    loc=None,
    ip=None,
) -> cute.TensorSSA:
    """Decode one 64-value NVFP4 transform fragment to BF16."""
    fragment_size = cute.size(packed_fragment.shape)
    scale_count = fragment_size // 16
    assert fragment_size == 64

    packed_type = ir.VectorType.get([fragment_size // 2], Uint8.mlir_type, loc=loc)
    packed = llvm.bitcast(
        packed_type, packed_fragment.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    packed_values = vector.to_elements(packed, loc=loc, ip=ip)
    scale_values = vector.to_elements(
        scale_fragment.ir_value(loc=loc, ip=ip), loc=loc, ip=ip
    )
    output_type = ir.VectorType.get([fragment_size], BFloat16.mlir_type, loc=loc)
    output_values: list[ir.Value] = []

    for scale_idx in range(scale_count):
        packed_offset = scale_idx * 8
        packed_block_type = ir.VectorType.get([8], Uint8.mlir_type, loc=loc)
        packed_block = vector.from_elements(
            packed_block_type,
            packed_values[packed_offset : packed_offset + 8],
            loc=loc,
            ip=ip,
        )
        packed_words_type = ir.VectorType.get([2], Uint32.mlir_type, loc=loc)
        packed_words = vector.to_elements(
            llvm.bitcast(packed_words_type, packed_block, loc=loc, ip=ip),
            loc=loc,
            ip=ip,
        )
        scale_e4m3 = scale_values[scale_idx]
        scale_bits = arith.bitcast(Uint8.mlir_type, scale_e4m3, loc=loc, ip=ip)
        scale_bits = Uint32(llvm.zext(Uint32.mlir_type, scale_bits, loc=loc, ip=ip))
        output_values.extend(
            value.ir_value(loc=loc, ip=ip)
            for value in e2m1x16_e4m3_to_bf16x16(
                Uint32(packed_words[0]),
                Uint32(packed_words[1]),
                scale_bits,
                loc=loc,
                ip=ip,
            )
        )

    output = vector.from_elements(output_type, output_values, loc=loc, ip=ip)
    return cute.TensorSSA(output, packed_fragment.shape, cutlass.BFloat16)


__all__ = ["decode_nvfp4_fragment_to_bf16", "e2m1x16_e4m3_to_bf16x16"]
