# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared SM120 FP4 block-scaled kernel construction helpers."""

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.utils.hopper_helpers as sm90_utils

from .utils import sm120_make_smem_layout_sfa, sm120_make_smem_layout_sfb


def make_sm120_fp4_mma_op(
    a_dtype,
    b_dtype,
    acc_dtype,
    sf_dtype,
    sf_vec_size: int,
):
    """Build the FP4 block-scaled warp MMA used by SM120 kernels."""
    if a_dtype != cutlass.Float4E2M1FN or b_dtype != cutlass.Float4E2M1FN:
        raise ValueError(
            "SM120 FP4 MMA requires Float4E2M1FN operands; "
            f"got a_dtype={a_dtype}, b_dtype={b_dtype}"
        )
    expected_sf_dtype = {
        16: cutlass.Float8E4M3FN,
        32: cutlass.Float8E8M0FNU,
    }.get(sf_vec_size)
    if expected_sf_dtype is None:
        raise ValueError(f"sf_vec_size must be 16 or 32; got {sf_vec_size}")
    if sf_dtype != expected_sf_dtype:
        raise ValueError(
            f"sf_vec_size={sf_vec_size} requires sf_dtype={expected_sf_dtype}; "
            f"got {sf_dtype}"
        )
    return cute.nvgpu.warp.MmaMXF4NVF4Op(a_dtype, acc_dtype, sf_dtype)


def make_sm120_fp4_ldmatrix_atom(
    operand_dtype,
    *,
    transpose: bool,
    num_matrices: int = 4,
):
    """Build the native FP4 SMEM-to-register ldmatrix copy atom."""
    if operand_dtype != cutlass.Float4E2M1FN:
        raise ValueError(
            "SM120 FP4 ldmatrix requires Float4E2M1FN; "
            f"got operand_dtype={operand_dtype}"
        )
    return cute.make_copy_atom(
        cute.nvgpu.warp.LdMatrix8x8x16bOp(
            transpose=transpose,
            num_matrices=num_matrices,
        ),
        operand_dtype,
    )


def compute_sm120_blockscaled_stages(
    tile_shape_mnk: tuple[int, int, int],
    a_dtype: Type[cutlass.Numeric],
    b_dtype: Type[cutlass.Numeric],
    sf_dtype: Type[cutlass.Numeric],
    sfa_smem_layout: cute.Layout,
    sfb_smem_layout: cute.Layout,
    epi_tile: tuple[int, int],
    c_dtype: Type[cutlass.Numeric],
    smem_capacity: int,
    occupancy: int,
) -> tuple[int, int]:
    """Compute raw mainloop and epilogue stage counts from SMEM capacity."""
    epi_stage_max = (tile_shape_mnk[1] // epi_tile[1]) * (
        tile_shape_mnk[0] // epi_tile[0]
    )
    epi_stage = min(epi_stage_max, 4)
    c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
    epi_bytes = c_bytes_per_stage * epi_stage

    a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
    b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
    ab_bytes_per_stage = (
        cute.size(a_shape) * a_dtype.width // 8
        + cute.size(b_shape) * b_dtype.width // 8
    )
    sf_bytes_per_stage = (
        cute.size(cute.filter_zeros(sfa_smem_layout).shape) * sf_dtype.width // 8
        + cute.size(cute.filter_zeros(sfb_smem_layout).shape) * sf_dtype.width // 8
    )
    mbar_helpers_bytes = 1024
    ab_stage = (
        (smem_capacity - occupancy * 1024) // occupancy - mbar_helpers_bytes - epi_bytes
    ) // (ab_bytes_per_stage + sf_bytes_per_stage)
    return ab_stage, epi_stage


def make_sm120_blockscaled_smem_layouts(
    tile_shape_mnk: tuple[int, int, int],
    epi_tile: tuple[int, int],
    a_dtype: Type[cutlass.Numeric],
    a_layout,
    b_dtype: Type[cutlass.Numeric],
    b_layout,
    ab_stage: int,
    c_dtype: Type[cutlass.Numeric],
    c_layout,
    epi_stage: int,
    sf_vec_size: int,
    tiled_mma: cute.TiledMma,
) -> tuple[
    cute.ComposedLayout,
    cute.ComposedLayout,
    cute.Layout,
    cute.Layout,
    cute.ComposedLayout,
]:
    """Create staged SMEM layouts for A, B, SFA, SFB, and C."""
    a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
    a_is_k_major = a_layout.is_k_major_a()
    b_is_k_major = b_layout.is_k_major_b()
    a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
    a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            a_layout,
            a_dtype,
            a_major_mode_size,
        ),
        a_dtype,
    )
    a_smem_layout_staged = cute.tile_to_shape(
        a_smem_layout_atom,
        cute.append(a_smem_shape, ab_stage),
        order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
    )

    b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
    b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
    b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            b_layout,
            b_dtype,
            b_major_mode_size,
        ),
        b_dtype,
    )
    b_smem_layout_staged = cute.tile_to_shape(
        b_smem_layout_atom,
        cute.append(b_smem_shape, ab_stage),
        order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
    )

    sfa_smem_layout_staged = sm120_make_smem_layout_sfa(
        tiled_mma,
        tile_shape_mnk,
        sf_vec_size,
        ab_stage,
    )
    sfb_smem_layout_staged = sm120_make_smem_layout_sfb(
        tiled_mma,
        tile_shape_mnk,
        sf_vec_size,
        ab_stage,
    )

    c_smem_shape = epi_tile
    c_major_mode_size = epi_tile[1] if c_layout.is_n_major_c() else epi_tile[0]
    c_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
        sm90_utils.get_smem_layout_atom(
            c_layout,
            c_dtype,
            c_major_mode_size,
        ),
        c_dtype,
    )
    epi_smem_layout_staged = cute.tile_to_shape(
        c_smem_layout_atom,
        cute.append(c_smem_shape, epi_stage),
        order=(1, 0, 2) if c_layout.is_m_major_c() else (0, 1, 2),
    )
    return (
        a_smem_layout_staged,
        b_smem_layout_staged,
        sfa_smem_layout_staged,
        sfb_smem_layout_staged,
        epi_smem_layout_staged,
    )


def make_sm120_tma_store_atom_and_tensor(
    tensor_c: cute.Tensor,
    epi_smem_layout_staged: cute.ComposedLayout,
    epi_tile: tuple[int, int],
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """Create the SMEM-to-GMEM TMA store atom and tensor for C."""
    epi_smem_layout = cute.slice_(epi_smem_layout_staged, (None, None, 0))
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileS2GOp(),
        tensor_c,
        epi_smem_layout,
        epi_tile,
    )


def make_sm120_tma_load_atom_and_tensor(
    tensor: cute.Tensor,
    smem_layout_staged: cute.ComposedLayout,
    smem_tile: tuple[int, int],
    mcast_dim: int = 1,
    internal_type: Optional[Type[cutlass.Numeric]] = None,
) -> tuple[cute.CopyAtom, cute.Tensor]:
    """Create a GMEM-to-SMEM TMA load atom and tensor."""
    op = (
        cpasync.CopyBulkTensorTileG2SOp()
        if mcast_dim == 1
        else cpasync.CopyBulkTensorTileG2SMulticastOp()
    )
    smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
    return cpasync.make_tiled_tma_atom(
        op,
        tensor,
        smem_layout,
        smem_tile,
        num_multicast=mcast_dim,
        internal_type=internal_type,
    )


__all__ = [
    "compute_sm120_blockscaled_stages",
    "make_sm120_blockscaled_smem_layouts",
    "make_sm120_fp4_ldmatrix_atom",
    "make_sm120_fp4_mma_op",
    "make_sm120_tma_load_atom_and_tensor",
    "make_sm120_tma_store_atom_and_tensor",
]
