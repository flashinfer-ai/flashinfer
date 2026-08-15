# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Epilogue store steps shared by the SM120 warp-specialized arms: convert, R2S, S2R, predicated R2G."""
import cutlass
import cutlass.cute as cute


def convert_acc(acc, out_dtype):
    """Accumulator -> output dtype, in registers."""
    tD = cute.make_fragment_like(acc, out_dtype)
    tD.store(acc.load().to(out_dtype))
    return tD


def acc_to_smem(tD, thr_mma, sC):
    """R2S: the converted accumulator into the epilogue buffer, along the MMA's own C partition."""
    cute.autovec_copy(tD, thr_mma.partition_C(sC))


def smem_to_reg(thr_st, sC):
    """S2R: the epilogue buffer back into registers, so sC is free before the gmem store drains."""
    tDsC = thr_st.partition_S(sC)
    tDrD = cute.make_fragment_like(tDsC)
    cute.autovec_copy(tDsC, tDrD)
    return tDrD


@cute.jit
def store_direct(thr_mma, tDrD, gD_tile, tile_mn, m_base, n_base, m_boundary, n_boundary):
    """R2G from the MMA's C partition without smem; scalar since cute.copy wants one predicate per access."""
    bm, bn = tile_mn
    cD = cute.make_identity_tensor((bm, bn))
    tDgD = thr_mma.partition_C(gD_tile)
    tDcD = thr_mma.partition_C(cD)
    for i in cutlass.range_constexpr(cute.size(tDrD)):
        mi, ni = tDcD[i]
        if m_base + mi < m_boundary and n_base + ni < n_boundary:
            tDgD[i] = tDrD[i]


@cute.jit
def store_predicated(atom, thr_st, tDrD, gD_tile, tile_mn, m_base, n_base, m_boundary, n_boundary):
    """R2G for one tile, dropping the rows and columns past the problem's edge."""
    bm, bn = tile_mn
    cD = cute.make_identity_tensor((bm, bn))
    tDgD = thr_st.partition_D(gD_tile)
    tDcD = thr_st.partition_S(cD)
    tDpD = cute.make_fragment_like(tDcD, cutlass.Boolean)
    for i in cutlass.range_constexpr(cute.size(tDpD)):
        mi, ni = tDcD[i]
        tDpD[i] = (m_base + mi) < m_boundary and (n_base + ni) < n_boundary
    cute.copy(atom, tDrD, tDgD, pred=tDpD)
