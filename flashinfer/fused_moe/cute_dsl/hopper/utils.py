# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVIDIA CUTLASS CuTe-DSL example
# examples/python/CuTeDSL/hopper/dense_gemm_persistent.py
# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES (BSD-3-Clause).

"""Shared utilities for the SM90 MoE GEMM kernels."""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import torch
from cutlass._mlir.dialects import llvm
from cutlass.cutlass_dsl import dsl_user_op


TORCH_TO_CUTLASS_DTYPE = {
    torch.bfloat16: cutlass.BFloat16,
    torch.float16: cutlass.Float16,
}


def make_smem_layouts(
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    a_dtype: type[cutlass.Numeric],
    a_layout: utils.LayoutEnum,
    b_dtype: type[cutlass.Numeric],
    b_layout: utils.LayoutEnum,
    ab_stage: int,
    c_dtype: type[cutlass.Numeric],
    c_layout: utils.LayoutEnum,
    epi_stage: int,
) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
    """Swizzled SMEM layouts for A/B mainloop stages and epilogue."""
    a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
    a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            a_layout,
            a_dtype,
            tile_shape_mnk[2 if a_is_k_major else 0],
        ),
        a_dtype,
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape, ab_stage),
        order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
    )

    b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
    b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
    b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            b_layout,
            b_dtype,
            tile_shape_mnk[2 if b_is_k_major else 1],
        ),
        b_dtype,
    )
    b_smem_layout_staged = cute.tile_to_shape(
        b_smem_layout_atom,
        cute.append(b_smem_shape, ab_stage),
        order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
    )

    c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            c_layout,
            c_dtype,
            epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0],
        ),
        c_dtype,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        cute.append(epi_tile, epi_stage),
        order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
    )

    return a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged


def make_tma_store_atoms_and_tensors(
    tensor_c: cute.Tensor,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: tuple[int, int],
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """TMA S2G atom for the epilogue store."""
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    tma_atom_c, tma_tensor_c = cute.nvgpu.cpasync.make_tiled_tma_atom(
        cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp(),
        tensor_c,
        epi_smem_layout,
        epi_tile,
    )

    return tma_atom_c, tma_tensor_c


def make_tma_atoms_and_tensors(
    tensor: cute.Tensor,
    smem_layout_staged: cute.ComposedLayout,
    smem_tile: tuple[int, int],
    mcast_dim: int,
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """TMA G2S atom; tensor modes beyond the tile become TMA coordinates
    (this is how B's expert mode is addressed without tensormap updates)."""
    op = (
        cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        if mcast_dim == 1
        else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
    )

    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    tma_atom, tma_tensor = cute.nvgpu.cpasync.make_tiled_tma_atom(
        op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=mcast_dim,
    )
    return tma_atom, tma_tensor


@dsl_user_op
def blk_copy(dst_gemm, src_smem, size, loc=None, ip=None):
    """Bulk-group copy from SMEM to GMEM (``size`` bytes)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            src_smem.iterator.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip),
            size.ir_value(loc=loc, ip=ip),
        ],
        "cp.async.bulk.global.shared::cta.bulk_group [$0], [$1], $2;",
        "l,r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_bf16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Bulk-group bf16 add-reduction from SMEM to GMEM (``size`` bytes)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.bf16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp16(dst_gemm, src_smem, size, loc=None, ip=None):
    """Bulk-group fp16 add-reduction from SMEM to GMEM (``size`` bytes)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.noftz.f16 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def blk_reduce_fp32(dst_gemm, src_smem, size, loc=None, ip=None):
    """Bulk-group f32 add-reduction from SMEM to GMEM (``size`` bytes)."""
    llvm.inline_asm(
        None,
        [
            dst_gemm.iterator.llvm_ptr,
            src_smem.iterator.llvm_ptr,
            size.ir_value(),
        ],
        "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.f32 [$0], [$1], $2;",
        "l,l,r",
        has_side_effects=True,
        loc=loc,
        ip=ip,
    )
