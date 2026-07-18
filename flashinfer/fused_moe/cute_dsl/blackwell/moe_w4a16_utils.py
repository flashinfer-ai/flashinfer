# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: Apache-2.0

"""Low-level conversion helpers for the SM100 NVFP4/BF16 MoE kernels."""

import cutlass
import cutlass.cute as cute
from cutlass._mlir import ir
from cutlass._mlir.dialects import arith, llvm, vector
from cutlass.cutlass_dsl import BFloat16, Uint8, Uint16, Uint32, dsl_user_op


@dsl_user_op
def e2m1x2_e4m3_to_bf16x2(
    packed_e2m1: Uint32,
    e4m3_scale: Uint32,
    *,
    loc=None,
    ip=None,
) -> tuple[BFloat16, BFloat16]:
    """Decode two E2M1 values and multiply by one E4M3 scale in BF16.

    CUDA 13.2 adds the direct E2M1/E4M3 to BF16 conversion forms. Keeping the
    conversion and multiply packed avoids widening the weight operand before it
    is staged for the BF16 ``tcgen05.mma`` mainloop.
    """
    packed_e2m1 = Uint16(packed_e2m1)
    scale_pair = Uint16(Uint32(e4m3_scale) * Uint32(0x0101))
    result = llvm.inline_asm(
        Uint32.mlir_type,
        [
            packed_e2m1.ir_value(loc=loc, ip=ip),
            scale_pair.ir_value(loc=loc, ip=ip),
        ],
        """
        {
            .reg .b8 q;
            .reg .b16 scale;
            .reg .b32 q_bf16x2, scale_bf16x2;

            mov.b16 {q, _}, $1;
            cvt.rn.bf16x2.e2m1x2 q_bf16x2, q;
            mov.b16 scale, $2;
            cvt.rn.bf16x2.e4m3x2 scale_bf16x2, scale;
            mul.rn.bf16x2 $0, q_bf16x2, scale_bf16x2;
        }
        """,
        "=r,h,h",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    result_type = ir.VectorType.get([2], BFloat16.mlir_type, loc=loc)
    result_values = vector.to_elements(
        llvm.bitcast(result_type, result, loc=loc, ip=ip), loc=loc, ip=ip
    )
    return BFloat16(result_values[0]), BFloat16(result_values[1])


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
    output_values = []

    for packed_idx in range(fragment_size // 2):
        scale_idx = packed_idx // 8
        assert scale_idx < scale_count
        packed_e2m1 = packed_values[packed_idx]
        scale_e4m3 = scale_values[scale_idx]
        scale_bits = arith.bitcast(Uint8.mlir_type, scale_e4m3, loc=loc, ip=ip)
        packed_e2m1 = Uint32(llvm.zext(Uint32.mlir_type, packed_e2m1, loc=loc, ip=ip))
        scale_bits = Uint32(llvm.zext(Uint32.mlir_type, scale_bits, loc=loc, ip=ip))
        value0, value1 = e2m1x2_e4m3_to_bf16x2(packed_e2m1, scale_bits, loc=loc, ip=ip)
        output_values.extend(
            [value0.ir_value(loc=loc, ip=ip), value1.ir_value(loc=loc, ip=ip)]
        )

    output = vector.from_elements(output_type, output_values, loc=loc, ip=ip)
    return cute.TensorSSA(output, packed_fragment.shape, cutlass.BFloat16)


__all__ = ["decode_nvfp4_fragment_to_bf16", "e2m1x2_e4m3_to_bf16x2"]
