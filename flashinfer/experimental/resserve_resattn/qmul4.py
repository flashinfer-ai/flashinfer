# SPDX-FileCopyrightText: Copyright (c) 2026 Rong Shuo
# SPDX-License-Identifier: Apache-2.0
# Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad.
# Original source: csrc/resattn/qmul4.py
"""Packed QMUL4 conversion for the production LDSM-to-TMEM decode path."""

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.cutlass_dsl as cutlass_dsl
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm, vector
from cutlass.cute.typing import Int32

if not cutlass_dsl.target_version(min_version="13.4"):
    raise RuntimeError(
        "ResAttn QMUL4 requires the CUDA 13.4 PTX compiler; "
        "install nvidia-cutlass-dsl[cu13]==4.8.0."
    )


@dsl_user_op
def mul_nvfp4x4_ldsm_e4m3_scales(
    src_vec4: ir.Value,
    scales_vec4: ir.Value,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
    *,
    loc: Optional[ir.Location] = None,
    ip: Optional[ir.InsertionPoint] = None,
) -> ir.Value:
    """Dequantize four packed NVFP4 values to E4M3."""
    assert dst_type == cutlass.Float8E4M3FN, dst_type
    assert isinstance(src_vec4.type, ir.VectorType)
    assert isinstance(scales_vec4.type, ir.VectorType)
    assert src_vec4.type.shape == [4], src_vec4.type
    assert scales_vec4.type.shape == [4], scales_vec4.type

    src_i32 = llvm.bitcast(Int32.mlir_type, src_vec4, loc=loc, ip=ip)
    scales_i32 = llvm.bitcast(Int32.mlir_type, scales_vec4, loc=loc, ip=ip)

    result_i32 = llvm.inline_asm(
        T.i32(),
        [src_i32, scales_i32],
        "mul.rn.satfinite.e4m3x4.e2m1p4x4.e4m3x4 $0, $1, $2;",
        "=r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )

    result_i32_type = ir.VectorType.get([1], Int32.mlir_type, loc=loc)
    result_i32_vec = vector.from_elements(result_i32_type, [result_i32], loc=loc, ip=ip)
    result_type = ir.VectorType.get([4], dst_type.mlir_type, loc=loc)
    return vector.bitcast(result_type, result_i32_vec, loc=loc, ip=ip)


def dequantize_nvfp4x8_ldsm(
    src: cute.Tensor,
    scales: cute.Tensor,
    dst_type: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
) -> cute.TensorSSA:
    """Dequantize an eight-value NVFP4 fragment."""
    src_fragment = src.load()
    scale_fragment = scales.load()
    assert cute.size(src_fragment.shape) == 8, src_fragment.shape
    assert cute.size(scale_fragment.shape) == 8, scale_fragment.shape

    src_half_type = ir.VectorType.get([4], src.dtype.mlir_type)
    scale_half_type = ir.VectorType.get([4], scales.dtype.mlir_type)
    src_lo = vector.extract_strided_slice(src_half_type, src_fragment, [0], [4], [1])
    src_hi = vector.extract_strided_slice(src_half_type, src_fragment, [4], [4], [1])
    scale_lo = vector.extract_strided_slice(
        scale_half_type, scale_fragment, [0], [4], [1]
    )
    scale_hi = vector.extract_strided_slice(
        scale_half_type, scale_fragment, [4], [4], [1]
    )
    dst_lo = mul_nvfp4x4_ldsm_e4m3_scales(src_lo, scale_lo, dst_type)
    dst_hi = mul_nvfp4x4_ldsm_e4m3_scales(src_hi, scale_hi, dst_type)
    combined = vector.shuffle(dst_lo, dst_hi, list(range(8)))
    return cute.TensorSSA.from_vector(
        combined, shape=src_fragment.shape, dtype=dst_type
    )


__all__ = ["dequantize_nvfp4x8_ldsm"]
