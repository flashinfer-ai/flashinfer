# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Low-level conversion helpers for SM100 NVFP4-weight MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cutlass_dsl import BFloat16, Float32, Uint8, Uint32, Uint64, dsl_user_op

from ....quantization.quantization_cute_dsl_utils import (
    INV_FLOAT8_E4M3_MAX,
    bfloat2x4_to_fp8x8_packed,
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)


_HAS_DIRECT_FP4_TO_BF16_CVT = cutlass.CUDA_VERSION.major > 13 or (
    cutlass.CUDA_VERSION.major == 13 and cutlass.CUDA_VERSION.minor >= 2
)


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

    CUDA 13.2 adds direct E2M1/E4M3 to BF16 conversion forms. Earlier toolkits
    decode and multiply in FP16, then widen the exact NVFP4 product to BF16.
    Both paths convert the shared scale once per block.
    """
    if _HAS_DIRECT_FP4_TO_BF16_CVT:
        asm = """
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
        """
    else:
        asm = """
        {
            .reg .b8 q0, q1, q2, q3, q4, q5, q6, q7, scale;
            .reg .b16 scale_pair, lo, hi;
            .reg .b32 scale_f16x2;
            .reg .f32 lo_f32, hi_f32;

            mov.b32 {q0, q1, q2, q3}, $8;
            mov.b32 {q4, q5, q6, q7}, $9;
            mov.b32 {scale, _, _, _}, $10;
            mov.b16 scale_pair, {scale, scale};
            cvt.rn.f16x2.e4m3x2 scale_f16x2, scale_pair;

            cvt.rn.f16x2.e2m1x2 $0, q0;
            cvt.rn.f16x2.e2m1x2 $1, q1;
            cvt.rn.f16x2.e2m1x2 $2, q2;
            cvt.rn.f16x2.e2m1x2 $3, q3;
            cvt.rn.f16x2.e2m1x2 $4, q4;
            cvt.rn.f16x2.e2m1x2 $5, q5;
            cvt.rn.f16x2.e2m1x2 $6, q6;
            cvt.rn.f16x2.e2m1x2 $7, q7;
            mul.rn.f16x2 $0, $0, scale_f16x2;
            mul.rn.f16x2 $1, $1, scale_f16x2;
            mul.rn.f16x2 $2, $2, scale_f16x2;
            mul.rn.f16x2 $3, $3, scale_f16x2;
            mul.rn.f16x2 $4, $4, scale_f16x2;
            mul.rn.f16x2 $5, $5, scale_f16x2;
            mul.rn.f16x2 $6, $6, scale_f16x2;
            mul.rn.f16x2 $7, $7, scale_f16x2;

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

            mov.b32 {lo, hi}, $4;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $4, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $5;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $5, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $6;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $6, hi_f32, lo_f32;

            mov.b32 {lo, hi}, $7;
            cvt.f32.f16 lo_f32, lo;
            cvt.f32.f16 hi_f32, hi;
            cvt.rn.bf16x2.f32 $7, hi_f32, lo_f32;
        }
        """

    result = llvm.inline_asm(
        llvm.StructType.get_literal([Uint32.mlir_type] * 8),
        [
            Uint32(packed_e2m1_lo).ir_value(loc=loc, ip=ip),
            Uint32(packed_e2m1_hi).ir_value(loc=loc, ip=ip),
            Uint32(e4m3_scale).ir_value(loc=loc, ip=ip),
        ],
        asm,
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
    result_vector = llvm.bitcast(result_type, packed, loc=loc, ip=ip)
    result_values = [
        vector.extract(result_vector, [], [idx], loc=loc, ip=ip) for idx in range(16)
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
    """Decode one NVFP4 transform fragment to BF16."""
    packed_byte_count = cute.size(packed_fragment.shape)
    fragment_size = packed_byte_count * 2
    scale_count = fragment_size // 16
    assert fragment_size in (64, 128)

    packed_values = [
        vector.extract(
            packed_fragment.ir_value(loc=loc, ip=ip), [], [idx], loc=loc, ip=ip
        )
        for idx in range(packed_byte_count)
    ]
    scale_values = [
        vector.extract(
            scale_fragment.ir_value(loc=loc, ip=ip), [], [idx], loc=loc, ip=ip
        )
        for idx in range(scale_count)
    ]
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
        packed_words_vector = llvm.bitcast(
            packed_words_type, packed_block, loc=loc, ip=ip
        )
        packed_words = [
            vector.extract(packed_words_vector, [], [idx], loc=loc, ip=ip)
            for idx in range(2)
        ]
        scale_e4m3 = scale_values[scale_idx]
        scale_bits = Uint32(llvm.zext(Uint32.mlir_type, scale_e4m3, loc=loc, ip=ip))
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
    return cute.TensorSSA(output, (fragment_size,), cutlass.BFloat16)


@dsl_user_op
def quantize_bf16_fragment_to_mxfp8(
    fragment: cute.TensorSSA,
    *,
    loc=None,
    ip=None,
) -> tuple[cute.TensorSSA, cute.TensorSSA]:
    """Quantize BF16 transform fragments to MXFP8 with 32-value UE8M0 blocks."""
    fragment_size = cute.size(fragment.shape)
    assert fragment_size in (64, 128)

    bf16_values = [
        vector.extract(fragment.ir_value(loc=loc, ip=ip), [], [idx], loc=loc, ip=ip)
        for idx in range(fragment_size)
    ]
    output_bytes: list[ir.Value] = []
    scale_bytes: list[ir.Value] = []

    for block_offset in range(0, fragment_size, 32):
        block_values = bf16_values[block_offset : block_offset + 32]
        absmax = Float32(0.0)
        for value in block_values:
            value_f32 = BFloat16(value).to(Float32)
            absmax = cute.arch.fmax(absmax, cute.arch.fmax(value_f32, -value_f32))

        scale_u32 = float_to_ue8m0_fast(
            absmax * Float32(INV_FLOAT8_E4M3_MAX), loc=loc, ip=ip
        )
        scale_bytes.append(scale_u32.to(Uint8).ir_value(loc=loc, ip=ip))
        inv_scale = ue8m0_to_inv_scale_fast(scale_u32, loc=loc, ip=ip)

        packed_bf16_pairs: list[Uint32] = []
        for pair_offset in range(0, 32, 2):
            pair_type = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
            pair = vector.from_elements(
                pair_type,
                block_values[pair_offset : pair_offset + 2],
                loc=loc,
                ip=ip,
            )
            packed_bf16_pairs.append(
                Uint32(llvm.bitcast(Uint32.mlir_type, pair, loc=loc, ip=ip))
            )

        packed_fp8_words: list[ir.Value] = []
        for pair_offset in range(0, 16, 4):
            packed_fp8_words.append(
                bfloat2x4_to_fp8x8_packed(
                    *packed_bf16_pairs[pair_offset : pair_offset + 4],
                    inv_scale,
                ).ir_value(loc=loc, ip=ip)
            )
        packed_fp8_type = ir.VectorType.get([4], Uint64.mlir_type, loc=loc)
        packed_fp8 = vector.from_elements(
            packed_fp8_type, packed_fp8_words, loc=loc, ip=ip
        )
        block_bytes_type = ir.VectorType.get([32], Uint8.mlir_type, loc=loc)
        block_bytes = llvm.bitcast(block_bytes_type, packed_fp8, loc=loc, ip=ip)
        output_bytes.extend(
            vector.extract(block_bytes, [], [idx], loc=loc, ip=ip) for idx in range(32)
        )

    output_bytes_type = ir.VectorType.get([fragment_size], Uint8.mlir_type, loc=loc)
    output_bytes_vector = vector.from_elements(
        output_bytes_type, output_bytes, loc=loc, ip=ip
    )
    output_type = ir.VectorType.get(
        [fragment_size], cutlass.Float8E4M3FN.mlir_type, loc=loc
    )
    output = llvm.bitcast(output_type, output_bytes_vector, loc=loc, ip=ip)

    scale_bytes_type = ir.VectorType.get(
        [fragment_size // 32], Uint8.mlir_type, loc=loc
    )
    scale_bytes_vector = vector.from_elements(
        scale_bytes_type, scale_bytes, loc=loc, ip=ip
    )
    scale_type = ir.VectorType.get(
        [fragment_size // 32], cutlass.Float8E8M0FNU.mlir_type, loc=loc
    )
    scales = llvm.bitcast(scale_type, scale_bytes_vector, loc=loc, ip=ip)

    return (
        cute.TensorSSA(output, (fragment_size,), cutlass.Float8E4M3FN),
        cute.TensorSSA(scales, (fragment_size // 32,), cutlass.Float8E8M0FNU),
    )


@dsl_user_op
def nvfp4_fragment_to_mxfp8(
    packed_fragment: cute.TensorSSA,
    scale_fragment: cute.TensorSSA,
    global_scale: Float32,
    *,
    loc=None,
    ip=None,
) -> tuple[cute.TensorSSA, cute.TensorSSA]:
    """Fully dequantize NVFP4 weights, then requantize them to MXFP8."""
    packed_byte_count = cute.size(packed_fragment.shape)
    fragment_size = packed_byte_count * 2
    scale_count = fragment_size // 16
    assert fragment_size in (64, 128)

    packed_values = [
        vector.extract(
            packed_fragment.ir_value(loc=loc, ip=ip), [], [idx], loc=loc, ip=ip
        )
        for idx in range(packed_byte_count)
    ]
    scale_values = [
        vector.extract(
            scale_fragment.ir_value(loc=loc, ip=ip), [], [idx], loc=loc, ip=ip
        )
        for idx in range(scale_count)
    ]
    scaled_type = ir.VectorType.get([fragment_size], BFloat16.mlir_type, loc=loc)
    scaled_values: list[ir.Value] = []
    for scale_idx in range(scale_count):
        packed_offset = scale_idx * 8
        packed_block = vector.from_elements(
            ir.VectorType.get([8], Uint8.mlir_type, loc=loc),
            packed_values[packed_offset : packed_offset + 8],
            loc=loc,
            ip=ip,
        )
        packed_words = llvm.bitcast(
            ir.VectorType.get([2], Uint32.mlir_type, loc=loc),
            packed_block,
            loc=loc,
            ip=ip,
        )
        scale_bits = Uint32(
            llvm.zext(
                Uint32.mlir_type,
                scale_values[scale_idx],
                loc=loc,
                ip=ip,
            )
        )
        decoded = e2m1x16_e4m3_to_bf16x16(
            Uint32(vector.extract(packed_words, [], [0], loc=loc, ip=ip)),
            Uint32(vector.extract(packed_words, [], [1], loc=loc, ip=ip)),
            scale_bits,
            loc=loc,
            ip=ip,
        )
        scaled_values.extend(
            (value.to(Float32) * global_scale).to(BFloat16).ir_value(loc=loc, ip=ip)
            for value in decoded
        )
    scaled = cute.TensorSSA(
        vector.from_elements(scaled_type, scaled_values, loc=loc, ip=ip),
        (fragment_size,),
        cutlass.BFloat16,
    )
    return quantize_bf16_fragment_to_mxfp8(scaled, loc=loc, ip=ip)


__all__ = [
    "decode_nvfp4_fragment_to_bf16",
    "e2m1x16_e4m3_to_bf16x16",
    "nvfp4_fragment_to_mxfp8",
    "quantize_bf16_fragment_to_mxfp8",
]
