# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CUDA-core MXFP8 GEMV for very small M on SM12x.

out[M, N] = A[M, K] @ B[N, K]^T for M <= MB. Each CTA issues its first weight
loads, then dequantizes A into shared memory as BF16. That is exact: an E4M3
value times a power of two has at most four significant bits, so every
product is exact in FP32 and the result matches an FP32-accumulated MXFP8
GEMM up to summation order.

A group of LPR lanes owns one weight row; each lane loads 16 contiguous bytes
per step and every pair of lanes covers one 32-element scale block. Rows are
dealt to warps round-robin and the next row group is in flight while the
current one is consumed. With KSPL > 1, KSPL warps split one row group's K
and sum their partials in shared memory in a fixed order.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils

from . import ptx
from .common import sf_offset

I32 = cutlass.Int32
U32 = cutlass.Uint32
F32 = cutlass.Float32


@cute.jit
def _load_rows(
    rg,
    i,
    lane,
    ksl,
    mB,
    mSFB,
    wbuf,
    sbuf,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    LPR: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    KSPL: cutlass.Constexpr,
):
    KT = (K // 32 + 3) // 4
    KSL = K // KSPL
    n = rg * (32 // LPR) + lane // LPR
    if cutlass.const_expr(N % (32 // LPR) != 0):
        n = ptx.imin(n, I32(N - 1))
    lr = lane % LPR
    for s in cutlass.range_constexpr(KS):
        k = ksl * KSL + s * (LPR * 16) + lr * 16
        live = True
        if cutlass.const_expr(KSL % (LPR * 16) != 0 and s == KS - 1):
            live = s * (LPR * 16) + lr * 16 < KSL
        if live:
            v0, v1, v2, v3 = ptx.ldg_nc_v4_stream(mB.iterator + (n * K + k))
            wbuf[i, s, 0] = v0
            wbuf[i, s, 1] = v1
            wbuf[i, s, 2] = v2
            wbuf[i, s, 3] = v3
            sbuf[i, s] = ptx.ldg_nc_u8(mSFB.iterator + sf_offset(n, k // 32, KT))
        else:
            for v in cutlass.range_constexpr(4):
                wbuf[i, s, v] = U32(0)
            sbuf[i, s] = U32(0)


@cute.jit
def _consume_rows(
    rg,
    i,
    lane,
    warp,
    ksl,
    M,
    sXv,
    sR,
    wbuf,
    sbuf,
    acc,
    mC,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    MB: cutlass.Constexpr,
    LPR: cutlass.Constexpr,
    KS: cutlass.Constexpr,
    KSPL: cutlass.Constexpr,
    F16: cutlass.Constexpr,
    live_rg,
):
    KSL = K // KSPL
    lr = lane % LPR
    for m in cutlass.range_constexpr(MB):
        acc[m] = F32(0.0)
    wf = cute.make_rmem_tensor(cute.make_layout((16,)), F32)
    xw = cute.make_rmem_tensor(cute.make_layout((8,)), I32)
    for s in cutlass.range_constexpr(KS):
        k = ksl * KSL + s * (LPR * 16) + lr * 16
        live = True
        if cutlass.const_expr(KSL % (LPR * 16) != 0 and s == KS - 1):
            live = s * (LPR * 16) + lr * 16 < KSL
        if live:
            for v in cutlass.range_constexpr(4):
                f0, f1, f2, f3 = ptx.e4m3x4_to_f32(wbuf[i, s, v])
                wf[v * 4] = f0
                wf[v * 4 + 1] = f1
                wf[v * 4 + 2] = f2
                wf[v * 4 + 3] = f3
            wsc = ptx.pow2_e8m0(sbuf[i, s])
            for m in cutlass.range_constexpr(MB):
                cute.autovec_copy(sXv[(m * K + k) // 16, None], xw)
                p0 = F32(0.0)
                p1 = F32(0.0)
                for z in cutlass.range_constexpr(4):
                    a0 = ptx.bf16_lo(xw[z])
                    a1 = ptx.bf16_hi(xw[z])
                    b0 = ptx.bf16_lo(xw[4 + z])
                    b1 = ptx.bf16_hi(xw[4 + z])
                    p0 = p0 + wf[2 * z] * a0
                    p0 = p0 + wf[2 * z + 1] * a1
                    p1 = p1 + wf[8 + 2 * z] * b0
                    p1 = p1 + wf[8 + 2 * z + 1] * b1
                acc[m] = acc[m] + (p0 + p1) * wsc
    n = rg * (32 // LPR) + lane // LPR
    for m in cutlass.range_constexpr(MB):
        v = acc[m]
        for o in cutlass.range_constexpr(LPR.bit_length() - 1):
            v = v + cute.arch.shuffle_sync_bfly(v, 1 << o)
        acc[m] = v
    if cutlass.const_expr(KSPL == 1):
        if lr == 0 and n < N and live_rg:
            for m in cutlass.range_constexpr(MB):
                if m < M:
                    ptx.stg_half(mC.iterator + (m * N + n), acc[m], F16)
    else:
        RG = 32 // LPR
        if lr == 0:
            for m in cutlass.range_constexpr(MB):
                sR[(warp * RG + lane // LPR) * MB + m] = acc[m]
        cute.arch.barrier()
        if ksl == 0 and lr == 0 and n < N and live_rg:
            for m in cutlass.range_constexpr(MB):
                tot = F32(0.0)
                for q in cutlass.range_constexpr(KSPL):
                    tot = tot + sR[((warp + q) * RG + lane // LPR) * MB + m]
                if m < M:
                    ptx.stg_half(mC.iterator + (m * N + n), tot, F16)
        cute.arch.barrier()


@cute.jit
def _stage_a(
    q,
    tidx,
    M,
    mA,
    mSFA,
    sXv,
    d,
    pk,
    K: cutlass.Constexpr,
    MB: cutlass.Constexpr,
    W: cutlass.Constexpr,
):
    """Dequantize half a scale block (16 values) of A into shared memory."""
    KB = K // 32
    KT = (KB + 3) // 4
    nh = MB * KB * 2
    bi = q * (W * 32) + tidx
    blk = bi // 2
    half = bi % 2
    m = blk // KB
    kb = blk - m * KB
    if bi < nh:
        if m < M:
            sx = ptx.ldg_nc_u8(mSFA.iterator + sf_offset(m, kb, KT))
            u0, u1, u2, u3 = ptx.ldg_nc_v4(mA.iterator + (m * K + kb * 32 + half * 16))
            for e, u in enumerate((u0, u1, u2, u3)):
                f0, f1, f2, f3 = ptx.e4m3x4_to_f32(u)
                d[e * 4] = f0
                d[e * 4 + 1] = f1
                d[e * 4 + 2] = f2
                d[e * 4 + 3] = f3
            sc = ptx.pow2_e8m0(sx)
            for z2 in cutlass.range_constexpr(8):
                pk[z2] = ptx.pack_half2(d[2 * z2] * sc, d[2 * z2 + 1] * sc)
        else:
            for z2 in cutlass.range_constexpr(8):
                pk[z2] = I32(0)
        r0 = (m * K + kb * 32) // 16 + half
        for z in cutlass.range_constexpr(8):
            sXv[r0, z] = pk[z]


@cute.kernel
def _gemv_kernel(
    mA: cute.Tensor,
    mSFA: cute.Tensor,
    mB: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    MB: cutlass.Constexpr,
    LPR: cutlass.Constexpr,
    W: cutlass.Constexpr,
    G: cutlass.Constexpr,
    KSPL: cutlass.Constexpr,
    F16: cutlass.Constexpr,
):
    RG = 32 // LPR
    NRG = (N + RG - 1) // RG
    KS = (K // KSPL + LPR * 16 - 1) // (LPR * 16)
    KB = K // 32
    WG = W // KSPL
    TW = G * WG

    tidx, _, _ = cute.arch.thread_idx()
    bx, _, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    gw = bx * WG + warp // KSPL
    ksl = warp % KSPL

    smem = cutlass.utils.SmemAllocator()
    sXp = smem.allocate_array(I32, MB * K // 2, byte_alignment=16)
    sXv = cute.make_tensor(sXp, cute.make_layout((MB * K // 16, 8), stride=(8, 1)))
    sRp = smem.allocate_array(F32, W * RG * MB, byte_alignment=16)
    sR = cute.make_tensor(sRp, cute.make_layout((W * RG * MB,)))

    wbuf = cute.make_rmem_tensor(cute.make_layout((2, KS, 4)), U32)
    sbuf = cute.make_rmem_tensor(cute.make_layout((2, KS)), U32)
    acc = cute.make_rmem_tensor(cute.make_layout((MB,)), F32)

    if gw < NRG:
        _load_rows(gw, 0, lane, ksl, mB, mSFB, wbuf, sbuf, N, K, LPR, KS, KSPL)

    nh = MB * KB * 2
    d = cute.make_rmem_tensor(cute.make_layout((16,)), F32)
    pk = cute.make_rmem_tensor(cute.make_layout((8,)), I32)
    NQ = (nh + W * 32 - 1) // (W * 32)
    if cutlass.const_expr(NQ <= 2):
        for q in cutlass.range_constexpr(NQ):
            _stage_a(q, tidx, M, mA, mSFA, sXv, d, pk, K, MB, W)
    else:
        for q in cutlass.range(NQ, unroll=1):
            _stage_a(q, tidx, M, mA, mSFA, sXv, d, pk, K, MB, W)
    cute.arch.barrier()

    # With a K split the trip count must be uniform across the CTA (the
    # reduction has barriers); otherwise each warp walks only its own rows.
    if cutlass.const_expr(KSPL > 1):
        # Opaque runtime value: a constant trip count is fully unrolled and spills.
        nloc = ptx.imin(I32((NRG + TW - 1) // TW), I32((NRG + TW - 1) // TW))
    else:
        nloc = (NRG - gw + TW - 1) // TW
    for it2 in cutlass.range((nloc + 1) // 2, unroll=1):
        for hb in cutlass.range_constexpr(2):
            it = it2 * 2 + hb
            if it < nloc:
                rg = gw + it * TW
                if rg + TW < NRG:
                    _load_rows(
                        rg + TW,
                        1 - hb,
                        lane,
                        ksl,
                        mB,
                        mSFB,
                        wbuf,
                        sbuf,
                        N,
                        K,
                        LPR,
                        KS,
                        KSPL,
                    )
                _consume_rows(
                    rg,
                    hb,
                    lane,
                    warp,
                    ksl,
                    M,
                    sXv,
                    sR,
                    wbuf,
                    sbuf,
                    acc,
                    mC,
                    N,
                    K,
                    MB,
                    LPR,
                    KS,
                    KSPL,
                    F16,
                    rg < NRG,
                )


class Sm12xMxfp8Gemv:
    """Host wrapper; ``__call__(a, sfa, b, sfb, c)`` with runtime M <= ``mb``."""

    def __init__(self, n, k, mb, lpr, warps, grid, k_split, out_f16=False):
        self.n, self.k = n, k
        self.mb, self.lpr, self.warps, self.grid = mb, lpr, warps, grid
        self.k_split = k_split
        self.out_f16 = out_f16

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mSFA: cute.Tensor,
        mB: cute.Tensor,
        mSFB: cute.Tensor,
        mC: cute.Tensor,
        stream,
    ):
        m = cute.size(mA, mode=[0])
        _gemv_kernel(
            mA,
            mSFA,
            mB,
            mSFB,
            mC,
            m,
            self.n,
            self.k,
            self.mb,
            self.lpr,
            self.warps,
            self.grid,
            self.k_split,
            self.out_f16,
        ).launch(grid=[self.grid, 1, 1], block=[32 * self.warps, 1, 1], stream=stream)
