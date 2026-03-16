# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Shared TMEM utilities for MLA compute roles.

Provides tmem_load_partition() — partitions TMEM output accumulator for
load/store by the rescale and epilogue roles.
"""

from types import SimpleNamespace

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05


@cute.jit
def tmem_load_partition(
    tmem_ptr: cutlass.Int32,
    tmem_o_offset: int,
    acc_dtype: cutlass.Constexpr,
    mma_pv_tiler: cutlass.Constexpr,
    cluster_shape_mnk: cutlass.Constexpr,
    warps_in_n: int,
    num_compute_warps: int,
    threads_per_warp: int,
    common_params: SimpleNamespace,
    tiled_mma_pv: cute.TiledMma,
    iter_n: int,
) -> tuple[
    cute.TiledMma,
    cute.TiledMma,
    cute.TiledMma,
    cute.TiledMma,
    cute.TiledMma,
    cute.TiledMma,
]:
    tOtO_shape = tiled_mma_pv.partition_shape_C(
        cute.select(mma_pv_tiler, mode=[0, 1])
    )
    tOtO = tiled_mma_pv.make_fragment_C(tOtO_shape)
    tOtO_layout = cute.append(
        tOtO.layout,
        cute.make_layout(
            common_params.L // mma_pv_tiler[1],
            stride=mma_pv_tiler[1] // warps_in_n,
        ),
    )
    tOtO = cute.make_tensor(tmem_ptr + tmem_o_offset, tOtO_layout)
    tOtO = tOtO[None, None, None, iter_n]

    tAcc = tOtO[(None, None), 0, 0]

    tmem_load_atom = cute.make_copy_atom(
        tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), acc_dtype
    )
    tmem_load_tiled_copy = tcgen05.make_tmem_copy(tmem_load_atom, tAcc)
    tmem_load_thr_copy = tmem_load_tiled_copy.get_slice(
        common_params.tidx % (num_compute_warps * threads_per_warp)
    )

    cta_pv_tiler = (
        mma_pv_tiler[0] // cluster_shape_mnk[0],
        mma_pv_tiler[1],
        mma_pv_tiler[2],
    )
    cta_pv_tiler_mn = cute.select(cta_pv_tiler, mode=[0, 1])

    gO = None
    if cutlass.const_expr(common_params.mAccO is not None):
        gO = cute.local_tile(
            common_params.mAccO[None, common_params.blk_coord[3], None, None],
            cta_pv_tiler_mn,
            (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
        )
        cO = cute.local_tile(
            cute.make_identity_tensor(
                common_params.mAccO[
                    None, common_params.blk_coord[3], None, None
                ].shape
            ),
            cta_pv_tiler_mn,
            (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
        )
    else:
        gO = cute.local_tile(
            common_params.mO,
            cta_pv_tiler_mn,
            (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
        )
        cO = cute.local_tile(
            cute.make_identity_tensor(common_params.mO.shape),
            cta_pv_tiler_mn,
            (common_params.blk_coord[0], iter_n, common_params.blk_coord[2]),
        )

    tTR_tAcc = tmem_load_thr_copy.partition_S(tAcc)
    tTR_gO = tmem_load_thr_copy.partition_D(gO)
    tTR_cO = tmem_load_thr_copy.partition_D(cO)
    tTR_rAcc = cute.make_fragment_like(tTR_gO, acc_dtype)
    return tmem_load_tiled_copy, tAcc, tTR_tAcc, tTR_gO, tTR_cO, tTR_rAcc
