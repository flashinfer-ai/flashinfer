# Copyright (c) 2025, Tri Dao.
# Copyright (c) 2026, Jerry Chen
# SPDX-License-Identifier: Apache-2.0 AND MIT
# Modifications Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Modifications are licensed under Apache-2.0. Pre-existing code retains
# its MIT terms; see LICENSING.md and THIRD_PARTY_LICENSES.txt.
"""SM90 WGMMA and shared-memory layout helpers."""

from typing import Type, Union, Optional
import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32, Boolean, const_expr
from cutlass.cute.nvgpu import warpgroup
from cutlass.cutlass_dsl import Numeric, dsl_user_op
import cutlass.utils.hopper_helpers as sm90_utils_og
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import LayoutEnum


@cute.jit
def gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: cutlass.Constexpr[bool] = False,
    wg_wait: cutlass.Constexpr[int] = 0,
    swap_AB: cutlass.Constexpr[bool] = False,
) -> None:
    if const_expr(swap_AB):
        gemm(tiled_mma, acc, tCrB, tCrA, zero_init=zero_init, wg_wait=wg_wait, swap_AB=False)
    else:
        warpgroup.fence()
        # We make a new mma_atom since we'll be modifying its attribute (accumulate).
        # Otherwise the compiler complains "operand #0 does not dominate this use"
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        mma_atom.set(warpgroup.Field.ACCUMULATE, not zero_init)
        K = cute.size(tCrA.shape[2])
        # for k in cutlass.range_constexpr(K):
        # # for k in cutlass.range(K, unroll=16):
        #     cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
        #     mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        # for k_outer in cutlass.range_constexpr(K // 16):
        # If we unroll full, the UR is not enough for large K.
        for k_outer in cutlass.range(K // 16, unroll=1):
            for k_inner in cutlass.range_constexpr(16):
                k = k_outer * 16 + k_inner
                cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
                mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        for k in cutlass.range_constexpr(K - (K // 16) * 16):
            k = (K // 16) * 16 + k
            cute.gemm(mma_atom, acc, tCrA[None, None, k], tCrB[None, None, k], acc)
            mma_atom.set(warpgroup.Field.ACCUMULATE, True)
        warpgroup.commit_group()
        if const_expr(wg_wait >= 0):
            warpgroup.wait_group(wg_wait)


def gemm_zero_init(
    tiled_mma: cute.TiledMma,
    shape: cute.Shape,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> cute.Tensor:
    if const_expr(swap_AB):
        return gemm_zero_init(tiled_mma, shape[::-1], tCrB, tCrA, B_idx, A_idx, wg_wait, swap_AB=False)
    else:
        acc = cute.make_rmem_tensor(tiled_mma.partition_shape_C(shape), Float32)
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=True, wg_wait=wg_wait)
        return acc


def gemm_w_idx(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    zero_init: Boolean,
    A_idx: Optional[Int32] = None,
    B_idx: Optional[Int32] = None,
    wg_wait: int = -1,
    swap_AB: bool = False,
) -> None:
    if const_expr(swap_AB):
        gemm_w_idx(tiled_mma, acc, tCrB, tCrA, zero_init, B_idx, A_idx, wg_wait, swap_AB=False)
    else:
        rA = tCrA if const_expr(A_idx is None) else tCrA[None, None, None, A_idx]
        rB = tCrB if const_expr(B_idx is None) else tCrB[None, None, None, B_idx]
        gemm(tiled_mma, acc, rA, rB, zero_init=zero_init, wg_wait=wg_wait)


def mma_partition_fragment_AB(
    thr_mma: cute.ThrMma,
    sA: Optional[cute.Tensor],
    sB: Optional[cute.Tensor],
    swap_AB: bool,
):
    if const_expr(not swap_AB):
        return (
            thr_mma.make_fragment_A(thr_mma.partition_A(sA)) if sA is not None else None,
            thr_mma.make_fragment_B(thr_mma.partition_B(sB)) if sB is not None else None,
        )
    else:
        return (
            thr_mma.make_fragment_B(thr_mma.partition_B(sA)) if sA is not None else None,
            thr_mma.make_fragment_A(thr_mma.partition_A(sB)) if sB is not None else None,
        )


@dsl_user_op
def make_smem_layout(
    dtype: Type[Numeric],
    layout: LayoutEnum,
    shape: cute.Shape,
    stage: Optional[int] = None,
    *,
    loc=None,
    ip=None,
) -> Union[cute.Layout, cute.ComposedLayout]:
    major_mode_size = shape[1] if layout.is_n_major_c() else shape[0]
    smem_layout_atom = warpgroup.make_smem_layout_atom(
        sm90_utils_og.get_smem_layout_atom(layout, dtype, major_mode_size),
        dtype,
    )
    order = (1, 0, 2) if const_expr(layout.is_m_major_c()) else (0, 1, 2)
    smem_layout_staged = cute.tile_to_shape(
        smem_layout_atom,
        cute.append(shape, stage) if const_expr(stage is not None) else shape,
        order=order if const_expr(stage is not None) else order[:2],
    )
    return smem_layout_staged
