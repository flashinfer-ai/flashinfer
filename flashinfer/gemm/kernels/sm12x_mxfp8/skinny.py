# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Stream-K block-scaled MXFP8 GEMM for skinny M on SM12x.

out[M, N] = A[M, K] @ B[N, K]^T with UE8M0 scales in the F8_128x4 layout.

The weight is cut into units of (16 * RT rows) x (32 * KU K columns). Units are
ordered row-tile-major; the flat unit range is split evenly over the G CTAs
(stream-K) and each CTA's range evenly over its W warps, so every warp streams
the same number of weight bytes for any N and K. A warp that owns a whole row
tile writes the output directly. Row tiles shared by warps of a CTA are summed
in shared memory; row tiles shared by several CTAs are finished by the
last-arriving CTA from FP32 partials in global scratch. Every reduction runs
in a fixed order, so results do not depend on scheduling.

The MMA is m16n8k32 block-scaled with A and B swapped: weight rows are the
MMA M dimension and tokens the N dimension (TM = 8 * NT tokens per grid row).

Weight loads: lane t of a quad loads 8 * KU contiguous bytes of its row
(KU = 4 or 2 k32 blocks per unit). The K order inside one MMA is free as long
as A and B agree and the 32 values stay inside one scale block, so an in-quad
transpose hands every lane its piece of each block.

In the CTA-wide variant all warps work on the same K unit of a W * 16 * RT row
block and share one activation tile staged in shared memory one unit ahead.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils

from . import ptx
from .common import sf_offset

I32 = cutlass.Int32
U32 = cutlass.Uint32
F32 = cutlass.Float32


def _quad_transpose4(R, i, r, h, o, b1, b0):
    """In-quad transpose of 4 x 64-bit slots o..o+3: slot s of lane l <- slot l of lane s."""
    for k in range(2):
        for w in range(2):
            a = R[i, r, h, o + k, w]
            c = R[i, r, h, o + 2 + k, w]
            rc = ptx.shfl_bfly(ptx.sel(b1, a, c), 2)
            R[i, r, h, o + k, w] = ptx.sel(b1, rc, a)
            R[i, r, h, o + 2 + k, w] = ptx.sel(b1, c, rc)
    for k in range(2):
        for w in range(2):
            a = R[i, r, h, o + 2 * k, w]
            c = R[i, r, h, o + 2 * k + 1, w]
            rc = ptx.shfl_bfly(ptx.sel(b0, a, c), 1)
            R[i, r, h, o + 2 * k, w] = ptx.sel(b0, rc, a)
            R[i, r, h, o + 2 * k + 1, w] = ptx.sel(b0, c, rc)


def _quad_transpose2(R, i, r, h, lane, t):
    """KU = 2: lane t holds pieces 2t, 2t+1 (block t // 2); afterwards slot j
    holds piece t of block j (from lane 2j + t // 2 of the quad)."""
    qbase = lane - t
    par = U32(t & 1)
    new = [[None, None], [None, None]]
    for j in range(2):
        src = qbase + 2 * j + (t >> 1)
        for w in range(2):
            v0 = ptx.shfl_idx(R[i, r, h, 0, w], src)
            v1 = ptx.shfl_idx(R[i, r, h, 1, w], src)
            new[j][w] = ptx.sel(par, v1, v0)
    for j in range(2):
        for w in range(2):
            R[i, r, h, j, w] = new[j][w]


def _range(lo, n, parts, idx):
    return lo + (idx * n) // parts, lo + ((idx + 1) * n) // parts


@cute.jit
def _load_unit(
    u,
    i,
    g,
    t,
    m0,
    M,
    mB,
    mSFB,
    mA,
    mSFA,
    abuf,
    sabuf,
    xbuf,
    xsbuf,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    RT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    KU: cutlass.Constexpr,
    LX: cutlass.Constexpr,
    CLAMP: cutlass.Constexpr = None,
):
    """Load unit u of the weight (and, with LX, the matching activation slice)."""
    KB = K // 32
    KBU = (KB + KU - 1) // KU
    KT = (KB + 3) // 4
    TAIL = KB % KU != 0
    st = u // KBU
    kb0 = (u - st * KBU) * KU
    for r in cutlass.range_constexpr(RT):
        for h in cutlass.range_constexpr(2):
            n = st * (16 * RT) + r * 16 + g + 8 * h
            if cutlass.const_expr(N % (16 * RT) != 0 if CLAMP is None else CLAMP):
                n = ptx.imin(n, I32(N - 1))
            live = True
            if cutlass.const_expr(TAIL):
                live = kb0 + (t * KU) // 4 < KB
            for q in cutlass.range_constexpr(KU // 2):
                if live:
                    v0, v1, v2, v3 = ptx.ldg_nc_v4_stream(
                        mB.iterator + (n * K + kb0 * 32 + t * (8 * KU) + q * 16)
                    )
                    abuf[i, r, h, 2 * q, 0] = v0
                    abuf[i, r, h, 2 * q, 1] = v1
                    abuf[i, r, h, 2 * q + 1, 0] = v2
                    abuf[i, r, h, 2 * q + 1, 1] = v3
                else:
                    abuf[i, r, h, 2 * q, 0] = U32(0)
                    abuf[i, r, h, 2 * q, 1] = U32(0)
                    abuf[i, r, h, 2 * q + 1, 0] = U32(0)
                    abuf[i, r, h, 2 * q + 1, 1] = U32(0)
        ns = st * (16 * RT) + r * 16 + g + 8 * (t & 1)
        if cutlass.const_expr(CLAMP if CLAMP is not None else False):
            ns = ptx.imin(ns, I32(N - 1))
        if cutlass.const_expr(KU == 2):
            sabuf[i, r] = ptx.ldg_nc_u16(mSFB.iterator + sf_offset(ns, kb0, KT))
        else:
            sabuf[i, r] = ptx.ldg_nc_u32(mSFB.iterator + sf_offset(ns, kb0, KT))
    for j in cutlass.range_constexpr(NT if LX else 0):
        m = m0 + j * 8 + g
        if m < M:
            for b in cutlass.range_constexpr(KU):
                live = True
                if cutlass.const_expr(TAIL):
                    live = kb0 + b < KB
                if live:
                    q0, q1 = ptx.ldg_nc_v2(
                        mA.iterator + (m * K + (kb0 + b) * 32 + t * 8)
                    )
                    xbuf[i, j, b, 0] = q0
                    xbuf[i, j, b, 1] = q1
                else:
                    xbuf[i, j, b, 0] = U32(0)
                    xbuf[i, j, b, 1] = U32(0)
            if cutlass.const_expr(KU == 2):
                xsbuf[i, j] = ptx.ldg_nc_u16(mSFA.iterator + sf_offset(m, kb0, KT))
            else:
                xsbuf[i, j] = ptx.ldg_nc_u32(mSFA.iterator + sf_offset(m, kb0, KT))
        else:
            for b in cutlass.range_constexpr(KU):
                xbuf[i, j, b, 0] = U32(0)
                xbuf[i, j, b, 1] = U32(0)
            xsbuf[i, j] = U32(0x7F7F7F7F)


@cute.jit
def _write_out(
    vals,
    st,
    g,
    t,
    m0,
    M,
    mC,
    N: cutlass.Constexpr,
    RT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    F16: cutlass.Constexpr,
):
    for r in cutlass.range_constexpr(RT):
        for h in cutlass.range_constexpr(2):
            n = st * (16 * RT) + r * 16 + g + 8 * h
            if n < N:
                for j in cutlass.range_constexpr(NT):
                    m = m0 + j * 8 + 2 * t
                    if m < M:
                        ptx.stg_half(mC.iterator + (m * N + n), vals[r, j, 2 * h], F16)
                    if m + 1 < M:
                        ptx.stg_half(
                            mC.iterator + ((m + 1) * N + n), vals[r, j, 2 * h + 1], F16
                        )


@cute.jit
def _cta_reduce(
    st,
    warp,
    lane,
    g,
    t,
    c0,
    c1,
    by,
    bx,
    m0,
    M,
    red,
    sP,
    mC,
    mP,
    mCnt,
    N: cutlass.Constexpr,
    KBU: cutlass.Constexpr,
    RT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    U: cutlass.Constexpr,
    G: cutlass.Constexpr,
    W: cutlass.Constexpr,
    STN: cutlass.Constexpr,
    F16: cutlass.Constexpr,
):
    """Sum the CTA's warp pieces of tile st (ascending warp order), then either
    write the tile or hand the CTA partial to the cross-CTA fixup."""
    SLOT = RT * NT * 32 * 4
    tu0 = st * KBU
    tu1 = tu0 + KBU
    for r in cutlass.range_constexpr(RT):
        for j in cutlass.range_constexpr(NT):
            for v in cutlass.range_constexpr(4):
                red[r, j, v] = F32(0.0)
    for w2 in cutlass.range_constexpr(W):
        a2, b2 = _range(c0, c1 - c0, W, w2)
        if w2 >= warp and a2 < b2 and a2 < tu1 and b2 > tu0:
            pc = I32(1)
            if st == a2 // KBU:
                pc = I32(0)
            for r in cutlass.range_constexpr(RT):
                for j in cutlass.range_constexpr(NT):
                    off = ((w2 * 2 + pc) * RT * NT + r * NT + j) * 128 + lane * 4
                    for v in cutlass.range_constexpr(4):
                        red[r, j, v] = red[r, j, v] + sP[off + v]
    if tu0 >= c0 and tu1 <= c1:
        _write_out(red, st, g, t, m0, M, mC, N, RT, NT, F16)
    else:
        piece = I32(1)
        if st == c0 // KBU:
            piece = I32(0)
        base = ((by * G + bx) * 2 + piece) * SLOT
        for r in cutlass.range_constexpr(RT):
            for j in cutlass.range_constexpr(NT):
                off = base + ((r * NT + j) * 32 + lane) * 4
                ptx.stg_v4_f32(
                    mP.iterator + off,
                    red[r, j, 0],
                    red[r, j, 1],
                    red[r, j, 2],
                    red[r, j, 3],
                )
        cute.arch.sync_warp()
        cidx = by * STN + st
        old = I32(0)
        if lane == 0:
            old = ptx.atom_add_acq_rel_gpu(mCnt.iterator + cidx, I32(1))
        old = cute.arch.shuffle_sync(old, 0)
        cute.arch.sync_warp()
        cf = ((tu0 + 1) * G - 1) // U
        cl = (tu1 * G - 1) // U
        if old == cl - cf:
            for r in cutlass.range_constexpr(RT):
                for j in cutlass.range_constexpr(NT):
                    for v in cutlass.range_constexpr(4):
                        red[r, j, v] = F32(0.0)
            for c in cutlass.range(cf, cl + 1, unroll=4):
                pc = I32(1)
                if st == ((c * U) // G) // KBU:
                    pc = I32(0)
                cbase = ((by * G + c) * 2 + pc) * SLOT
                for r in cutlass.range_constexpr(RT):
                    for j in cutlass.range_constexpr(NT):
                        off = cbase + ((r * NT + j) * 32 + lane) * 4
                        p0, p1, p2, p3 = ptx.ldg_cg_v4_f32(mP.iterator + off)
                        red[r, j, 0] = red[r, j, 0] + p0
                        red[r, j, 1] = red[r, j, 1] + p1
                        red[r, j, 2] = red[r, j, 2] + p2
                        red[r, j, 3] = red[r, j, 3] + p3
            _write_out(red, st, g, t, m0, M, mC, N, RT, NT, F16)
            if lane == 0:
                ptx.st_relaxed_gpu_s32(mCnt.iterator + cidx, I32(0))


@cute.kernel
def _rowtile_kernel(
    mA: cute.Tensor,
    mSFA: cute.Tensor,
    mB: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    mP: cute.Tensor,
    mCnt: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    RT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    KU: cutlass.Constexpr,
    P: cutlass.Constexpr,
    W: cutlass.Constexpr,
    G: cutlass.Constexpr,
    F16: cutlass.Constexpr,
):
    KBU = (K // 32 + KU - 1) // KU
    KTAIL = (K // 32) % KU != 0
    STN = (N + 16 * RT - 1) // (16 * RT)
    U = STN * KBU
    TM = NT * 8
    SLOT = RT * NT * 32 * 4

    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    g = lane // 4
    t = lane % 4
    tb0 = U32(t & 1)
    tb1 = U32((t >> 1) & 1)
    m0 = by * TM
    c0, c1 = _range(0, U, G, bx)
    a, b = _range(c0, c1 - c0, W, warp)

    smem = cutlass.utils.SmemAllocator()
    sPp = smem.allocate_array(F32, W * 2 * SLOT, byte_alignment=16)
    sP = cute.make_tensor(sPp, cute.make_layout((W * 2 * SLOT,)))

    abuf = cute.make_rmem_tensor(cute.make_layout((P, RT, 2, KU, 2)), U32)
    sabuf = cute.make_rmem_tensor(cute.make_layout((P, RT)), U32)
    xbuf = cute.make_rmem_tensor(cute.make_layout((P, NT, KU, 2)), U32)
    xsbuf = cute.make_rmem_tensor(cute.make_layout((P, NT)), U32)
    acc = cute.make_rmem_tensor(cute.make_layout((RT, NT, 4)), F32)
    red = cute.make_rmem_tensor(cute.make_layout((RT, NT, 4)), F32)
    acc.fill(0.0)

    for i in cutlass.range_constexpr(P):
        if a + i < b:
            _load_unit(
                a + i,
                i,
                g,
                t,
                m0,
                M,
                mB,
                mSFB,
                mA,
                mSFA,
                abuf,
                sabuf,
                xbuf,
                xsbuf,
                N,
                K,
                RT,
                NT,
                KU,
                True,
            )

    nsteps = (b - a + P - 1) // P
    for s in cutlass.range(nsteps, unroll=1):
        for i in cutlass.range_constexpr(P):
            u = a + s * P + i
            if u < b:
                st = u // KBU
                kb0u = (u - st * KBU) * KU
                for r in cutlass.range_constexpr(RT):
                    for h in cutlass.range_constexpr(2):
                        if cutlass.const_expr(KU == 4):
                            _quad_transpose4(abuf, i, r, h, 0, tb1, tb0)
                        else:
                            _quad_transpose2(abuf, i, r, h, lane, t)
                for bb in cutlass.range_constexpr(KU):
                    blive = True
                    if cutlass.const_expr(KTAIL):
                        blive = kb0u + bb < K // 32
                    if blive:
                        for r in cutlass.range_constexpr(RT):
                            for j in cutlass.range_constexpr(NT):
                                d0, d1, d2, d3 = ptx.mma_mxf8(
                                    abuf[i, r, 0, bb, 0],
                                    abuf[i, r, 1, bb, 0],
                                    abuf[i, r, 0, bb, 1],
                                    abuf[i, r, 1, bb, 1],
                                    xbuf[i, j, bb, 0],
                                    xbuf[i, j, bb, 1],
                                    acc[r, j, 0],
                                    acc[r, j, 1],
                                    acc[r, j, 2],
                                    acc[r, j, 3],
                                    sabuf[i, r],
                                    xsbuf[i, j],
                                    bida=bb,
                                    bidb=bb,
                                )
                                acc[r, j, 0] = d0
                                acc[r, j, 1] = d1
                                acc[r, j, 2] = d2
                                acc[r, j, 3] = d3
                un = u + P
                if un < b:
                    _load_unit(
                        un,
                        i,
                        g,
                        t,
                        m0,
                        M,
                        mB,
                        mSFB,
                        mA,
                        mSFA,
                        abuf,
                        sabuf,
                        xbuf,
                        xsbuf,
                        N,
                        K,
                        RT,
                        NT,
                        KU,
                        True,
                    )
                if u - st * KBU == KBU - 1 or u == b - 1:
                    tu0 = st * KBU
                    if a <= tu0 and tu0 + KBU <= b:
                        _write_out(acc, st, g, t, m0, M, mC, N, RT, NT, F16)
                    else:
                        pc = I32(1)
                        if st == a // KBU:
                            pc = I32(0)
                        for r in cutlass.range_constexpr(RT):
                            for j in cutlass.range_constexpr(NT):
                                off = (
                                    (warp * 2 + pc) * RT * NT + r * NT + j
                                ) * 128 + lane * 4
                                for v in cutlass.range_constexpr(4):
                                    sP[off + v] = acc[r, j, v]
                    acc.fill(0.0)

    cute.arch.barrier()
    sta = a // KBU
    stb = (b - 1) // KBU
    head = (
        a < b
        and not (a <= sta * KBU and sta * KBU + KBU <= b)
        and (a == c0 or (a - 1) // KBU != sta)
    )
    tail = a < b and stb != sta and not (stb * KBU + KBU <= b)
    if head:
        _cta_reduce(
            sta,
            warp,
            lane,
            g,
            t,
            c0,
            c1,
            by,
            bx,
            m0,
            M,
            red,
            sP,
            mC,
            mP,
            mCnt,
            N,
            KBU,
            RT,
            NT,
            U,
            G,
            W,
            STN,
            F16,
        )
    if tail:
        _cta_reduce(
            stb,
            warp,
            lane,
            g,
            t,
            c0,
            c1,
            by,
            bx,
            m0,
            M,
            red,
            sP,
            mC,
            mP,
            mCnt,
            N,
            KBU,
            RT,
            NT,
            U,
            G,
            W,
            STN,
            F16,
        )


@cute.jit
def _cta_stage_load(
    ku,
    tidx,
    m0,
    M,
    mA,
    mSFA,
    xpre,
    K: cutlass.Constexpr,
    TM: cutlass.Constexpr,
    KU: cutlass.Constexpr,
):
    KB = K // 32
    KT = (KB + 3) // 4
    if tidx < TM * KU:
        tk = tidx // KU
        kb = ku * KU + tidx % KU
        m = m0 + tk
        xpre[16] = U32(0)
        if m < M and kb < KB:
            xpre[16] = U32(1)
            for h in cutlass.range_constexpr(2):
                u0, u1, u2, u3 = ptx.ldg_nc_v4(mA.iterator + (m * K + kb * 32 + h * 16))
                xpre[h * 4] = u0
                xpre[h * 4 + 1] = u1
                xpre[h * 4 + 2] = u2
                xpre[h * 4 + 3] = u3
            xpre[8] = ptx.ldg_nc_u8(mSFA.iterator + sf_offset(m, kb, KT))


@cute.jit
def _cta_stage_store(
    buf,
    tidx,
    xpre,
    sXq,
    sXs,
    TM: cutlass.Constexpr,
    KU: cutlass.Constexpr,
):
    XRW = (KU * 32 + 16) // 4
    if tidx < TM * KU:
        tk = tidx // KU
        b = tidx % KU
        base = (buf * TM + tk) * XRW + b * 8
        sbase = (buf * TM + tk) * KU + b
        if xpre[16] != U32(0):
            for z in cutlass.range_constexpr(8):
                sXq[base + z] = xpre[z]
            sXs[sbase] = xpre[8].to(cutlass.Uint8)
        else:
            for z in cutlass.range_constexpr(8):
                sXq[base + z] = U32(0)
            sXs[sbase] = cutlass.Uint8(127)


@cute.kernel
def _ctawide_kernel(
    mA: cute.Tensor,
    mSFA: cute.Tensor,
    mB: cute.Tensor,
    mSFB: cute.Tensor,
    mC: cute.Tensor,
    mP: cute.Tensor,
    mCnt: cute.Tensor,
    M: cutlass.Int32,
    N: cutlass.Constexpr,
    K: cutlass.Constexpr,
    RT: cutlass.Constexpr,
    NT: cutlass.Constexpr,
    KU: cutlass.Constexpr,
    P: cutlass.Constexpr,
    W: cutlass.Constexpr,
    G: cutlass.Constexpr,
    F16: cutlass.Constexpr,
):
    KB = K // 32
    KBU = (KB + KU - 1) // KU
    KTAIL = KB % KU != 0
    RB = W * RT * 16
    STNB = (N + RB - 1) // RB
    U = STNB * KBU
    TM = NT * 8
    SLOT = RT * NT * 32 * 4
    XRW = (KU * 32 + 16) // 4
    CLAMP = N % RB != 0

    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    warp = tidx // 32
    lane = tidx % 32
    g = lane // 4
    t = lane % 4
    tb0 = U32(t & 1)
    tb1 = U32((t >> 1) & 1)
    m0 = by * TM
    c0, c1 = _range(0, U, G, bx)

    smem = cutlass.utils.SmemAllocator()
    sXqp = smem.allocate_array(U32, 2 * TM * XRW, byte_alignment=16)
    sXq = cute.make_tensor(sXqp, cute.make_layout((2 * TM * XRW,)))
    sXsp = smem.allocate_array(cutlass.Uint8, 2 * TM * KU, byte_alignment=16)
    sXs = cute.make_tensor(sXsp, cute.make_layout((2 * TM * KU,)))
    sFp = smem.allocate_array(I32, 4, byte_alignment=16)
    sF = cute.make_tensor(sFp, cute.make_layout((4,)))

    abuf = cute.make_rmem_tensor(cute.make_layout((P, RT, 2, KU, 2)), U32)
    sabuf = cute.make_rmem_tensor(cute.make_layout((P, RT)), U32)
    xbuf = cute.make_rmem_tensor(cute.make_layout((1, NT, KU, 2)), U32)
    xsbuf = cute.make_rmem_tensor(cute.make_layout((1, NT)), U32)
    xpre = cute.make_rmem_tensor(cute.make_layout((17,)), U32)
    acc = cute.make_rmem_tensor(cute.make_layout((RT, NT, 4)), F32)
    red = cute.make_rmem_tensor(cute.make_layout((RT, NT, 4)), F32)
    bq = cute.make_rmem_tensor(cute.make_layout((NT, 2)), U32)
    sfb = cute.make_rmem_tensor(cute.make_layout((NT,)), U32)
    acc.fill(0.0)

    for i in cutlass.range_constexpr(P):
        u = c0 + i
        if u < c1:
            blk = u // KBU
            uw = (blk * W + warp) * KBU + (u - blk * KBU)
            _load_unit(
                uw,
                i,
                g,
                t,
                m0,
                M,
                mB,
                mSFB,
                mA,
                mSFA,
                abuf,
                sabuf,
                xbuf,
                xsbuf,
                N,
                K,
                RT,
                NT,
                KU,
                False,
                CLAMP,
            )
    if c0 < c1:
        _cta_stage_load(c0 - (c0 // KBU) * KBU, tidx, m0, M, mA, mSFA, xpre, K, TM, KU)
        _cta_stage_store(0, tidx, xpre, sXq, sXs, TM, KU)
    cute.arch.barrier()

    nsteps = (c1 - c0 + P - 1) // P
    for s in cutlass.range(nsteps, unroll=1):
        for i in cutlass.range_constexpr(P):
            u = c0 + s * P + i
            if u < c1:
                blk = u // KBU
                ku = u - blk * KBU
                cur = (u - c0) % 2
                if u + 1 < c1:
                    un = u + 1
                    _cta_stage_load(
                        un - (un // KBU) * KBU, tidx, m0, M, mA, mSFA, xpre, K, TM, KU
                    )
                for r in cutlass.range_constexpr(RT):
                    for h in cutlass.range_constexpr(2):
                        if cutlass.const_expr(KU == 4):
                            _quad_transpose4(abuf, i, r, h, 0, tb1, tb0)
                        else:
                            _quad_transpose2(abuf, i, r, h, lane, t)
                for bb in cutlass.range_constexpr(KU):
                    blive = True
                    if cutlass.const_expr(KTAIL):
                        blive = ku * KU + bb < KB
                    if blive:
                        for j in cutlass.range_constexpr(NT):
                            xo = (cur * TM + j * 8 + g) * XRW + bb * 8 + 2 * t
                            bq[j, 0] = sXq[xo]
                            bq[j, 1] = sXq[xo + 1]
                            sfb[j] = sXs[(cur * TM + j * 8 + g) * KU + bb].to(U32)
                        for r in cutlass.range_constexpr(RT):
                            for j in cutlass.range_constexpr(NT):
                                d0, d1, d2, d3 = ptx.mma_mxf8(
                                    abuf[i, r, 0, bb, 0],
                                    abuf[i, r, 1, bb, 0],
                                    abuf[i, r, 0, bb, 1],
                                    abuf[i, r, 1, bb, 1],
                                    bq[j, 0],
                                    bq[j, 1],
                                    acc[r, j, 0],
                                    acc[r, j, 1],
                                    acc[r, j, 2],
                                    acc[r, j, 3],
                                    sabuf[i, r],
                                    sfb[j],
                                    bida=bb,
                                    bidb=0,
                                )
                                acc[r, j, 0] = d0
                                acc[r, j, 1] = d1
                                acc[r, j, 2] = d2
                                acc[r, j, 3] = d3
                up = u + P
                if up < c1:
                    blp = up // KBU
                    uwp = (blp * W + warp) * KBU + (up - blp * KBU)
                    _load_unit(
                        uwp,
                        i,
                        g,
                        t,
                        m0,
                        M,
                        mB,
                        mSFB,
                        mA,
                        mSFA,
                        abuf,
                        sabuf,
                        xbuf,
                        xsbuf,
                        N,
                        K,
                        RT,
                        NT,
                        KU,
                        False,
                        CLAMP,
                    )
                if u + 1 < c1:
                    _cta_stage_store(1 - cur, tidx, xpre, sXq, sXs, TM, KU)
                if ku == KBU - 1 or u == c1 - 1:
                    tu0 = blk * KBU
                    st = blk * W + warp
                    if c0 <= tu0 and tu0 + KBU <= c1:
                        _write_out(acc, st, g, t, m0, M, mC, N, RT, NT, F16)
                    else:
                        pc = I32(1)
                        if blk == c0 // KBU:
                            pc = I32(0)
                        base = (((by * G + bx) * 2 + pc) * W + warp) * SLOT
                        for r in cutlass.range_constexpr(RT):
                            for j in cutlass.range_constexpr(NT):
                                off = base + ((r * NT + j) * 32 + lane) * 4
                                ptx.stg_v4_f32(
                                    mP.iterator + off,
                                    acc[r, j, 0],
                                    acc[r, j, 1],
                                    acc[r, j, 2],
                                    acc[r, j, 3],
                                )
                        cute.arch.barrier()
                        cidx = by * STNB + blk
                        if tidx == 0:
                            ptx.fence_acq_rel_gpu()
                            sF[0] = ptx.atom_add_acq_rel_gpu(
                                mCnt.iterator + cidx, I32(1)
                            )
                        cute.arch.barrier()
                        old = sF[0]
                        cf = ((tu0 + 1) * G - 1) // U
                        cl = ((tu0 + KBU) * G - 1) // U
                        if old == cl - cf:
                            ptx.fence_acq_rel_gpu()
                            for r in cutlass.range_constexpr(RT):
                                for j in cutlass.range_constexpr(NT):
                                    for v in cutlass.range_constexpr(4):
                                        red[r, j, v] = F32(0.0)
                            for c in cutlass.range(cf, cl + 1, unroll=4):
                                pcc = I32(1)
                                if blk == ((c * U) // G) // KBU:
                                    pcc = I32(0)
                                cb = (((by * G + c) * 2 + pcc) * W + warp) * SLOT
                                for r in cutlass.range_constexpr(RT):
                                    for j in cutlass.range_constexpr(NT):
                                        off = cb + ((r * NT + j) * 32 + lane) * 4
                                        p0, p1, p2, p3 = ptx.ldg_cg_v4_f32(
                                            mP.iterator + off
                                        )
                                        red[r, j, 0] = red[r, j, 0] + p0
                                        red[r, j, 1] = red[r, j, 1] + p1
                                        red[r, j, 2] = red[r, j, 2] + p2
                                        red[r, j, 3] = red[r, j, 3] + p3
                            _write_out(red, st, g, t, m0, M, mC, N, RT, NT, F16)
                            if tidx == 0:
                                ptx.st_relaxed_gpu_s32(mCnt.iterator + cidx, I32(0))
                    acc.fill(0.0)
                cute.arch.barrier()


class Sm12xMxfp8Skinny:
    """Host wrapper; ``__call__(a, sfa, b, sfb, c, partials, counters)``.

    ``nt`` token groups of 8 per grid row, ``rt`` 16-row MMA tiles per unit,
    ``ku`` k32 blocks per unit, ``p`` units prefetched per warp, ``warps`` per
    CTA, ``grid`` stream-K CTAs. ``partials`` (FP32) and
    ``counters`` (Int32, zero on entry and on exit) are the stream-K scratch;
    see :meth:`workspace_size`.
    """

    def __init__(self, n, k, nt, rt, ku, p, warps, grid, cta_wide, out_f16=False):
        self.n, self.k = n, k
        self.nt, self.rt, self.ku, self.p = nt, rt, ku, p
        self.warps, self.grid, self.cta_wide = warps, grid, cta_wide
        self.out_f16 = out_f16

    @staticmethod
    def workspace_size(m, n, nt, rt, warps, grid, cta_wide):
        """(partial floats, counters) needed for ``m`` rows."""
        rows = 16 * rt * (warps if cta_wide else 1)
        n_tiles = (n + rows - 1) // rows
        m_blocks = (m + 8 * nt - 1) // (8 * nt)
        slot = rt * nt * 32 * 4 * (warps if cta_wide else 1)
        return m_blocks * grid * 2 * slot, m_blocks * n_tiles

    @cute.jit
    def __call__(
        self,
        mA: cute.Tensor,
        mSFA: cute.Tensor,
        mB: cute.Tensor,
        mSFB: cute.Tensor,
        mC: cute.Tensor,
        mP: cute.Tensor,
        mCnt: cute.Tensor,
        stream,
    ):
        m = cute.size(mA, mode=[0])
        tm = 8 * self.nt
        kernel = _ctawide_kernel if self.cta_wide else _rowtile_kernel
        kernel(
            mA,
            mSFA,
            mB,
            mSFB,
            mC,
            mP,
            mCnt,
            m,
            self.n,
            self.k,
            self.rt,
            self.nt,
            self.ku,
            self.p,
            self.warps,
            self.grid,
            self.out_f16,
        ).launch(
            grid=[self.grid, (m + tm - 1) // tm, 1],
            block=[32 * self.warps, 1, 1],
            stream=stream,
        )
