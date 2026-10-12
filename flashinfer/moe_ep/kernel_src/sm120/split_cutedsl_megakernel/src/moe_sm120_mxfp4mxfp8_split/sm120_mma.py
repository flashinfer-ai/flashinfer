# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""SM120 block-scaled warp-MMA helpers for the swap-AB MegaMoE kernel.

This module owns the SM120 register-accumulator MXFP8 QMMA path. The target
instruction is PTX:

    mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X
      .m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0

The instruction issue path uses the low-level
``cute_nvgpu.arch.mma.SM120.block_scaled`` operation.
"""

from __future__ import annotations

from typing import Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.warp.mma as warp_mma
import cutlass.utils.blackwell_helpers as sm120_utils
from cutlass import Float32, Uint32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir import ir
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir, llvm
from cutlass.cute.nvgpu.warp.mma import _pack_shape


MMA_M = 16
MMA_N = 8
MMA_K = 32
MXFP8_BLOCK = 32
FP4_SHIFT_BITS = 2
SWAP_AB_INTERLEAVE = 8
CTA_TOKEN_TILE = 64
INTERMEDIATE_ALIGNMENT = 2 * MXFP8_BLOCK




MXFP8_MMA_PTX_E4M3 = (
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
    ".m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0"
)

MXFP4_MXFP8_MMA_PTX_E2M1_E4M3 = (
    "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X"
    ".m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0"
)




def make_swapab_m64n8k128_tiled_mma(
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
    sf_dtype: Type[cutlass.Numeric] = cutlass.Float8E8M0FNU,
) -> cute.TiledMma:
    """Create the warp-MMA tiler used by the SM120 Swap A/B mainloop.

    Four compute warps cover the four M16 atoms in one ``M64 x N8`` MMA
    slice.

    The CuTe MMA op is used here only to construct tiling/fragment layouts;
    the actual MMA issue is emitted by ``_arch_mma_m16n8k32_mxfp8`` below.
    """

    if a_dtype == b_dtype:
        op = warp_mma.MmaMXF8Op(a_dtype, acc_dtype, sf_dtype)
    else:
        op = warp_mma.MmaMXF8F6F4Op(
            a_dtype, b_dtype, acc_dtype, sf_dtype
        )
    permutation_mnk = sm120_utils.get_permutation_mnk(
        (64, 64, 128),
        MXFP8_BLOCK,
        True,
    )
    return cute.make_tiled_mma(
        op,
        cute.make_layout((4, 1, 1)),
        permutation_mnk=permutation_mnk,
    )


def make_sm120_ldmatrix_atom(
    operand_dtype: Type[cutlass.Numeric],
    *,
    transpose: bool,
    mixed_mode: bool = False,
) -> cute.CopyAtom:
    """Build the SM120 block-scaled SMEM->RMEM ldmatrix copy atom.

    Mixed FP4 operands use ``ldsm.b4x16_p64``.  That instruction does not
    support transpose, so the packed FP4 weight tensor must be K-major.
    """

    if mixed_mode and operand_dtype.width == 4:
        if transpose:
            raise ValueError("mixed FP4 ldmatrix requires K-major storage")
        return cute.make_copy_atom(
            cute.nvgpu.warp.LdMatrix8x16x8bOp(
                transpose=False,
                num_matrices=4,
                unpack_bits=4,
            ),
            cutlass.Int8,
        )

    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=4,
        ),
        operand_dtype,
    )


@cute.jit
def shift_fp4_fragment_for_mxf8f6f4(fragment: cute.Tensor) -> None:
    """Move unpacked E2M1 nibbles into the bit position consumed by QMMA."""

    fragment_i8 = cute.recast_tensor(fragment, cutlass.Int8)
    for i in cutlass.range_constexpr(cute.size(fragment_i8)):
        fragment_i8[i] = cutlass.Int8(fragment_i8[i] << FP4_SHIFT_BITS)


@dsl_user_op
def _arch_mma_m16n8k32_mxfp8(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_scalar,
    sfb_scalar,
    *,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    acc_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    byte_id_a: int = 0,
    byte_id_b: int = 0,
    thread_id_a: int = 0,
    thread_id_b: int = 0,
    loc=None,
    ip=None,
) -> None:
    """Issue one QMMA ``m16n8k32`` MXFP8 block-scaled instruction.

    Per lane:
      A = 4 x b32, B = 2 x b32, C/D = 4 x f32.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    sfa_i8 = sfa_scalar.bitcast(cutlass.Int8, loc=loc, ip=ip)
    sfb_i8 = sfb_scalar.bitcast(cutlass.Int8, loc=loc, ip=ip)

    shape_attr = _pack_shape((MMA_M, MMA_N, MMA_K), loc=loc, ip=ip).type.attribute
    res = _cute_nvgpu_ir.arch_mma_SM120_block_scaled(
        [acc_dtype.mlir_type] * 4,
        shape_attr,
        MXFP8_BLOCK,
        ir.TypeAttr.get(a_dtype.mlir_type),
        ir.TypeAttr.get(b_dtype.mlir_type),
        ir.TypeAttr.get(sf_dtype.mlir_type),
        [
            a_i32[0].ir_value(loc=loc, ip=ip),
            a_i32[1].ir_value(loc=loc, ip=ip),
            a_i32[2].ir_value(loc=loc, ip=ip),
            a_i32[3].ir_value(loc=loc, ip=ip),
        ],
        [
            b_i32[0].ir_value(loc=loc, ip=ip),
            b_i32[1].ir_value(loc=loc, ip=ip),
        ],
        [
            acc[0].ir_value(loc=loc, ip=ip),
            acc[1].ir_value(loc=loc, ip=ip),
            acc[2].ir_value(loc=loc, ip=ip),
            acc[3].ir_value(loc=loc, ip=ip),
        ],
        sfa_i8.ir_value(loc=loc, ip=ip),
        sfb_i8.ir_value(loc=loc, ip=ip),
        thread_id_a=thread_id_a,
        thread_id_b=thread_id_b,
        byte_id_a=cutlass.Int16(byte_id_a).ir_value(loc=loc, ip=ip),
        byte_id_b=cutlass.Int16(byte_id_b).ir_value(loc=loc, ip=ip),
        loc=loc,
        ip=ip,
    )

    acc[0] = acc_dtype(res[0])
    acc[1] = acc_dtype(res[1])
    acc[2] = acc_dtype(res[2])
    acc[3] = acc_dtype(res[3])


@dsl_user_op
def _arch_mma_m16n8k32_mxfp8_packed_sfb(
    acc: cute.Tensor,
    a_reg: cute.Tensor,
    b_reg: cute.Tensor,
    sfa_scalar,
    sfb_packed,
    *,
    byte_id_b: int,
    loc=None,
    ip=None,
) -> None:
    """Issue mixed W4A8 QMMA using one packed SFB register.

    PTX selects one byte from ``sfb_packed`` with ``byte_id_b``. This keeps
    the N16 scale word intact across the four K32 issues and avoids explicit
    SHF/LOP3 byte extraction in the persistent K2 loop.
    """

    a_i32 = a_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    b_i32 = b_reg.load(loc=loc, ip=ip).bitcast(cutlass.Int32, loc=loc, ip=ip)
    sfa_u8 = sfa_scalar.bitcast(cutlass.Uint8, loc=loc, ip=ip)
    bid_a = cutlass.Int16(0).ir_value(loc=loc, ip=ip)
    tid_a = cutlass.Int16(0).ir_value(loc=loc, ip=ip)
    bid_b = cutlass.Int16(byte_id_b).ir_value(loc=loc, ip=ip)
    tid_b = cutlass.Int16(0).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        llvm.StructType.get_literal([T.f32(), T.f32(), T.f32(), T.f32()]),
        [
            Uint32(a_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[2]).ir_value(loc=loc, ip=ip),
            Uint32(a_i32[3]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[0]).ir_value(loc=loc, ip=ip),
            Uint32(b_i32[1]).ir_value(loc=loc, ip=ip),
            Uint32(sfa_u8).ir_value(loc=loc, ip=ip),
            bid_a,
            tid_a,
            Uint32(sfb_packed).ir_value(loc=loc, ip=ip),
            bid_b,
            tid_b,
            Float32(acc[0]).ir_value(loc=loc, ip=ip),
            Float32(acc[1]).ir_value(loc=loc, ip=ip),
            Float32(acc[2]).ir_value(loc=loc, ip=ip),
            Float32(acc[3]).ir_value(loc=loc, ip=ip),
        ],
        """
        mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3.f32.ue8m0
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
    acc[0] = Float32(llvm.extractvalue(T.f32(), result, [0], loc=loc, ip=ip))
    acc[1] = Float32(llvm.extractvalue(T.f32(), result, [1], loc=loc, ip=ip))
    acc[2] = Float32(llvm.extractvalue(T.f32(), result, [2], loc=loc, ip=ip))
    acc[3] = Float32(llvm.extractvalue(T.f32(), result, [3], loc=loc, ip=ip))




@cute.jit
def issue_m64n8k32_mxfp8(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag,
    b_frag,
    sfa_frag,
    sfb_frag,
    *,
    n_group: int,
    active_n_groups: int,
    sfb_n_group: int = -1,
    sfa_m_group,
    k_inner: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one SM120 MXFP8 ``M64 x N8 x K32`` warp-MMA slice."""

    if cutlass.const_expr(n_group >= active_n_groups):
        return

    a_reg = a_frag[(None, 0, k_inner)]
    b_reg = b_frag[(None, n_group, k_inner)]

    sfa_scalar = sfa_frag[((0, 0), sfa_m_group, k_inner)]
    if cutlass.const_expr(sfb_n_group < 0):
        sfb_n_group = n_group
    sfb_scalar = sfb_frag[
        (
            (0, 0),
            (sfb_n_group % 2, (sfb_n_group // 2) % 2),
            k_inner,
            sfb_n_group // 4,
        )
    ]

    _arch_mma_m16n8k32_mxfp8(
        acc,
        a_reg,
        b_reg,
        sfa_scalar,
        sfb_scalar,
        a_dtype=a_dtype,
        b_dtype=b_dtype,
        acc_dtype=cutlass.Float32,
        sf_dtype=sf_dtype,
    )



@cute.jit
def issue_m64n8k32_mxfp8_packed_sfb(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    a_frag,
    b_frag,
    sfa_frag,
    sfb_packed,
    *,
    n_group: int,
    active_n_groups: int,
    sfa_m_group,
    k_inner: int,
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
) -> None:
    """Issue one QMMA using byte-id selection from packed SFB."""

    if cutlass.const_expr(n_group >= active_n_groups):
        return
    _arch_mma_m16n8k32_mxfp8_packed_sfb(
        acc,
        a_frag[(None, 0, k_inner)],
        b_frag[(None, n_group, k_inner)],
        sfa_frag[((0, 0), sfa_m_group, k_inner)],
        sfb_packed,
        byte_id_b=k_inner,
    )
