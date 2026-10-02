# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Persistent warp-specialized MXFP8 GEMM for SM12x (small and mid M).

out[M, N] = A[M, K] @ B[N, K]^T, both K-major E4M3 with UE8M0 scales in the
F8_128x4 layout, read as given.

- One producer thread. A and B have separate shared-memory rings (SA and SB
  stages) with their own full/empty mbarriers, so the DRAM-streamed weights
  can run further ahead than the activations, which the producers first pull
  into L2 with bulk prefetches split over the grid. Launched with
  programmatic dependent launch: launch and barrier setup overlap the previous
  kernel (typically the activation quantization), and all global memory
  traffic follows ``griddepcontrol.wait``. Operands use TMA
  (128B or 64B swizzle); each stage's 512-byte scale chunks use a 1D bulk copy.
- WM x WN consumer warps: ldmatrix + ``mma.sync.m16n8k32`` block-scaled. A
  32-bit scale word holds the four k32 scales of one row of a 128-wide chunk.
  The next stage's scales and first fragments are loaded before the current
  stage's last MMAs.
- Persistent schedule: data-parallel tiles, then an optional stream-K region
  of ``sk_tiles`` tiles whose K iterations are split evenly over the grid.
  Every contributor stores its partial tile to its own FP32 slot; the CTA that
  completes the tile's K iterations sums the slots in CTA order (deterministic)
  and stores the result.

M, the grid size and ``sk_tiles`` are runtime values; N, K and the tile
configuration are compile-time. The stream-K decomposition follows the CUTLASS
stream-K scheduler (data-parallel waves plus a split remainder).
"""

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64
from cutlass.cute.nvgpu import cpasync
from cutlass.utils import SmemAllocator

from . import ptx
from .common import SF_CHUNK, ceil_div

SMEM_LIMIT = 101376


def scale_chunks(bn):
    """128-row weight scale chunks a BN-row tile can straddle."""
    if 128 % bn == 0 or bn % 128 == 0:
        return max(1, bn // 128)
    return (bn + 254) // 128


def stage_bytes(bm, bn, ks, kw):
    """(A stage, B stage) shared-memory bytes."""
    return ks * (bm * kw + SF_CHUNK), ks * (bn * kw + SF_CHUNK * scale_chunks(bn))


def max_b_stages(bm, bn, ks, kw, sa):
    a, b = stage_bytes(bm, bn, ks, kw)
    return (SMEM_LIMIT - 2400 - sa * a) // b


class Sm12xMxfp8Persistent:
    """Host wrapper; ``__call__(a, b, sfa, sfb, c, partials, counters, grid, sk_tiles)``.

    BM x BN CTA tile, WM x WN consumer warps, KW the K box width in bytes
    (64 or 128), KS boxes per stage (KW = 64 requires KS = 1), SB weight
    stages and SA activation stages.
    """

    def __init__(self, n, k, bm, bn, wm, wn, ks, sb, kw, sa, streamk, out_f16=False):
        self.N, self.K = n, k
        self.out_f16 = out_f16
        # Data-parallel kernels are compiled without the stream-K fixup.
        self.streamk = streamk
        self.BM, self.BN, self.WM, self.WN, self.KS = bm, bn, wm, wn, ks
        self.SB, self.SA, self.KW = sb, sa, kw
        assert kw in (64, 128) and (kw == 128 or ks == 1)
        assert 1 <= sa <= sb
        self.half = kw == 64
        self.CW = wm * wn
        self.CT = self.CW * 32
        self.TM = bm // wm
        self.TN = bn // wn
        assert self.TM % 16 == 0 and self.TN % 16 == 0
        assert bm in (32, 64, 128) and bn % 16 == 0 and 32 <= bn <= 256
        self.MI = self.TM // 16
        self.NI = self.TN // 8
        self.NK = kw // 32
        self.Kc = ceil_div(k, 128)
        self.Kb = ceil_div(k, kw)
        self.KI = ceil_div(self.Kb, ks)
        self.NBC = scale_chunks(bn)
        self.n_sf_rows_b = ceil_div(n, 128)
        self.tiles_n = ceil_div(n, bn)
        self.B_BYTES = bn * kw
        self.A_BYTES = bm * kw
        self.ACC = self.MI * self.NI * 4
        self.threads = (self.CW + 1) * 32
        sa_bytes = sa * ks * (self.A_BYTES + SF_CHUNK)
        sb_bytes = sb * ks * (self.B_BYTES + self.NBC * SF_CHUNK)
        self.smem_bytes = sa_bytes + sb_bytes + 4 * (sa + sb) * 8 + 16 + 2048
        assert self.smem_bytes <= SMEM_LIMIT

    def partial_floats(self, sk_tiles, grid):
        """FP32 scratch for ``sk_tiles`` stream-K tiles on ``grid`` CTAs."""
        if not sk_tiles:
            return 0
        si = sk_tiles * self.KI
        maxc = (self.KI * grid + si - 1) // si + 1
        return sk_tiles * maxc * self.BM * self.BN

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        cnt: cute.Tensor,
        grid: Int32,
        sk_tiles: Int32,
        stream,
    ):
        KW = self.KW
        m = cute.size(a, mode=[0])
        lb = cute.make_composed_layout(
            self.swizzle(), 0, cute.make_layout((self.BN, KW), stride=(KW, 1))
        )
        atom_b, tb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), b, lb, (self.BN, KW)
        )
        la = cute.make_composed_layout(
            self.swizzle(), 0, cute.make_layout((self.BM, KW), stride=(KW, 1))
        )
        atom_a, ta = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(), a, la, (self.BM, KW)
        )
        self.kernel(
            atom_a, ta, atom_b, tb, a, sfa, sfb, out, ws, cnt, m, sk_tiles
        ).launch(
            grid=(grid, 1, 1),
            block=(self.threads, 1, 1),
            smem=self.smem_bytes,
            stream=stream,
            use_pdl=True,
        )

    def swizzle(self):
        if self.KW == 128:
            return cute.make_swizzle(3, 4, 3)
        return cute.make_swizzle(2, 4, 3)

    def swz_row(self, row):
        """16-byte chunk XOR term of the 128B / 64B TMA swizzle for a K-major row."""
        if self.KW == 128:
            return row % 8
        return (row // 2) % 4

    # ---------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        atom_a: cute.CopyAtom,
        ta: cute.Tensor,
        atom_b: cute.CopyAtom,
        tb: cute.Tensor,
        a: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        ws: cute.Tensor,
        cnt: cute.Tensor,
        M: Int32,
        sk_tiles: Int32,
    ):
        BM, BN, KS, SA, SB, KW = self.BM, self.BN, self.KS, self.SA, self.SB, self.KW
        tx, _, _ = cute.arch.thread_idx()
        bid, _, _ = cute.arch.block_idx()
        G, _, _ = cute.arch.grid_dim()
        warp = tx // 32
        lane = tx % 32

        smem = SmemAllocator()
        sA = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((SA * KS * self.A_BYTES,)),
            byte_alignment=1024,
        )
        sB = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((SB * KS * self.B_BYTES,)),
            byte_alignment=1024,
        )
        sSA = smem.allocate_tensor(
            cutlass.Uint8, cute.make_layout((SA * KS * SF_CHUNK,)), byte_alignment=128
        )
        sSB = smem.allocate_tensor(
            cutlass.Uint8,
            cute.make_layout((SB * KS * self.NBC * SF_CHUNK,)),
            byte_alignment=128,
        )
        fullA = smem.allocate_array(cutlass.Int64, SA, byte_alignment=8)
        emptyA = smem.allocate_array(cutlass.Int64, SA, byte_alignment=8)
        fullB = smem.allocate_array(cutlass.Int64, SB, byte_alignment=8)
        emptyB = smem.allocate_array(cutlass.Int64, SB, byte_alignment=8)
        flag = smem.allocate_tensor(Int32, cute.make_layout((4,)), byte_alignment=16)

        swz = self.swizzle()
        b_view = cute.make_tensor(
            cute.recast_ptr(sB.iterator, swz, dtype=cutlass.Uint8),
            cute.make_layout((BN, KW, SB * KS), stride=(KW, 1, self.B_BYTES)),
        )
        gB = cute.local_tile(tb, (BN, KW), (None, None))
        tBs, tBg = cpasync.tma_partition(
            atom_b,
            0,
            cute.make_layout(1),
            cute.group_modes(b_view, 0, 2),
            cute.group_modes(gB, 0, 2),
        )
        a_view = cute.make_tensor(
            cute.recast_ptr(sA.iterator, swz, dtype=cutlass.Uint8),
            cute.make_layout((BM, KW, SA * KS), stride=(KW, 1, self.A_BYTES)),
        )
        gA = cute.local_tile(ta, (BM, KW), (None, None))
        tAs, tAg = cpasync.tma_partition(
            atom_a,
            0,
            cute.make_layout(1),
            cute.group_modes(a_view, 0, 2),
            cute.group_modes(gA, 0, 2),
        )

        if tx == 0:
            for s in cutlass.range_constexpr(SA):
                cute.arch.mbarrier_init(fullA + s, 1)
                cute.arch.mbarrier_init(emptyA + s, self.CW)
            for s in cutlass.range_constexpr(SB):
                cute.arch.mbarrier_init(fullB + s, 1)
                cute.arch.mbarrier_init(emptyB + s, self.CW)
            cute.arch.mbarrier_init_fence()
            cpasync.prefetch_descriptor(atom_a)
            cpasync.prefetch_descriptor(atom_b)
        cute.arch.sync_threads()

        tiles_m = (M + BM - 1) // BM
        T = tiles_m * self.tiles_n
        dp_tiles = T - sk_tiles
        SI = sk_tiles * self.KI
        sk_a = (bid * SI) // G
        sk_b = ((bid + 1) * SI) // G
        n_dp = Int32(0)
        if bid < dp_tiles:
            n_dp = (dp_tiles - bid + G - 1) // G
        total = n_dp * self.KI + (sk_b - sk_a)

        if warp == self.CW:
            if lane == 0:
                self.producer(
                    a,
                    M,
                    atom_a,
                    tAs,
                    tAg,
                    atom_b,
                    tBs,
                    tBg,
                    sfa,
                    sfb,
                    sSA,
                    sSB,
                    fullA,
                    emptyA,
                    fullB,
                    emptyB,
                    bid,
                    G,
                    tiles_m,
                    dp_tiles,
                    n_dp,
                    sk_a,
                    total,
                )
        else:
            cute.arch.griddepcontrol_wait()
            self.consumer(
                sA,
                sB,
                sSA,
                sSB,
                fullA,
                emptyA,
                fullB,
                emptyB,
                flag,
                out,
                ws,
                cnt,
                M,
                tx,
                warp,
                lane,
                bid,
                G,
                tiles_m,
                dp_tiles,
                sk_a,
                sk_b,
            )

    # ---------------------------------------------------------- schedule
    @cute.jit
    def locate(self, q, bid, G, dp_tiles, n_dp, sk_a):
        """Tile and K iteration of this CTA's q-th pipeline iteration."""
        tile = bid + (q // self.KI) * G
        kk = q % self.KI
        if q >= n_dp * self.KI:
            it = sk_a + (q - n_dp * self.KI)
            tile = dp_tiles + it // self.KI
            kk = it % self.KI
        return tile, kk

    @cute.jit
    def next_piece(self, t, it, G, dp_tiles, sk_b):
        """Advance the per-CTA piece iterator. Returns (tile, k0, k1, t, it)."""
        tile = t
        k0 = Int32(0)
        k1 = Int32(self.KI)
        if t < dp_tiles:
            t = t + G
        else:
            tl = it // self.KI
            k0 = it - tl * self.KI
            k1 = cutlass.min(Int32(self.KI), k0 + (sk_b - it))
            tile = dp_tiles + tl
            it = it + (k1 - k0)
        return tile, k0, k1, t, it

    @cute.jit
    def stage_k(self, kk):
        """(first box, valid boxes, first scale chunk, scale chunks) of iteration kk."""
        kc0 = kk * self.KS
        nkc = cutlass.min(Int32(self.KS), Int32(self.Kb) - kc0)
        sfc0 = kc0
        nsf = nkc
        if cutlass.const_expr(self.half):
            sfc0 = kk // 2
            nsf = Int32(1)
        return kc0, nkc, sfc0, nsf

    # ----------------------------------------------------------- producer
    @cute.jit
    def issue_b(
        self,
        atom_b,
        tBs,
        tBg,
        sfb,
        sSB,
        fullB,
        emptyB,
        q,
        bid,
        G,
        tiles_m,
        dp_tiles,
        n_dp,
        sk_a,
    ):
        KS, SB = self.KS, self.SB
        s = q % SB
        if q >= SB:
            cute.arch.mbarrier_wait(emptyB + s, (q // SB - 1) % 2)
        tile, kk = self.locate(q, bid, G, dp_tiles, n_dp, sk_a)
        nt = tile // tiles_m
        sfb_row = (nt * self.BN) // 128
        nbc = cutlass.min(Int32(self.NBC), Int32(self.n_sf_rows_b) - sfb_row)
        kc0, nkc, sfc0, nsf = self.stage_k(kk)
        cute.arch.mbarrier_arrive_and_expect_tx(
            fullB + s, nkc * self.B_BYTES + nsf * SF_CHUNK * nbc
        )
        for c in cutlass.range_constexpr(self.NBC):
            if c < nbc:
                ptx.bulk_g2s(
                    (sSB.iterator + (s * self.NBC + c) * KS * SF_CHUNK).toint(),
                    (
                        sfb.iterator + ((sfb_row + c) * self.Kc + sfc0) * SF_CHUNK
                    ).toint(),
                    nsf * SF_CHUNK,
                    (fullB + s).toint(),
                )
        for ks in cutlass.range_constexpr(KS):
            if ks < nkc:
                cute.copy(
                    atom_b,
                    tBg[None, nt, kc0 + ks],
                    tBs[None, s * KS + ks],
                    tma_bar_ptr=fullB + s,
                )

    @cute.jit
    def issue_a(
        self,
        atom_a,
        tAs,
        tAg,
        sfa,
        sSA,
        fullA,
        emptyA,
        q,
        bid,
        G,
        tiles_m,
        dp_tiles,
        n_dp,
        sk_a,
    ):
        KS, SA = self.KS, self.SA
        s = q % SA
        if q >= SA:
            cute.arch.mbarrier_wait(emptyA + s, (q // SA - 1) % 2)
        tile, kk = self.locate(q, bid, G, dp_tiles, n_dp, sk_a)
        mt = tile % tiles_m
        kc0, nkc, sfc0, nsf = self.stage_k(kk)
        sfa_row = (mt * self.BM) // 128
        cute.arch.mbarrier_arrive_and_expect_tx(
            fullA + s, nkc * self.A_BYTES + nsf * SF_CHUNK
        )
        ptx.bulk_g2s(
            (sSA.iterator + s * KS * SF_CHUNK).toint(),
            (sfa.iterator + (sfa_row * self.Kc + sfc0) * SF_CHUNK).toint(),
            nsf * SF_CHUNK,
            (fullA + s).toint(),
        )
        for ks in cutlass.range_constexpr(KS):
            if ks < nkc:
                cute.copy(
                    atom_a,
                    tAg[None, mt, kc0 + ks],
                    tAs[None, s * KS + ks],
                    tma_bar_ptr=fullA + s,
                )

    @cute.jit
    def prefetch_a(self, a, sfa, M, bid, G):
        """Pull the activations and their scales into L2, split over the grid.

        The activation ring is only SA stages deep, so with A still in DRAM
        every refill would expose a full DRAM round trip.
        """
        nbytes = M * self.K
        step = ((nbytes + G - 1) // G + 15) // 16 * 16
        lo = bid * step
        if lo < nbytes:
            ptx.bulk_prefetch_l2(
                (a.iterator + lo).toint(), cutlass.min(step, nbytes - lo)
            )
        sf_bytes = cutlass.min(
            ((M + 127) // 128) * self.Kc * SF_CHUNK, cute.size(sfa) // 16 * 16
        )
        if bid == G - 1 and sf_bytes > 0:
            ptx.bulk_prefetch_l2(sfa.iterator.toint(), sf_bytes)

    @cute.jit
    def producer(
        self,
        a,
        M,
        atom_a,
        tAs,
        tAg,
        atom_b,
        tBs,
        tBg,
        sfa,
        sfb,
        sSA,
        sSB,
        fullA,
        emptyA,
        fullB,
        emptyB,
        bid,
        G,
        tiles_m,
        dp_tiles,
        n_dp,
        sk_a,
        total,
    ):
        """Single thread. Weight stages run SB - SA iterations ahead of activations.

        Everything waits for the previous grid (``griddepcontrol.wait``), and
        the activation prefetch is issued before the first weight load:
        weight traffic queued ahead of it would delay the first MMA.
        """
        cute.arch.griddepcontrol_wait()
        self.prefetch_a(a, sfa, M, bid, G)
        L = self.SB - self.SA
        for qb in range(cutlass.min(Int32(L), total)):
            self.issue_b(
                atom_b,
                tBs,
                tBg,
                sfb,
                sSB,
                fullB,
                emptyB,
                qb,
                bid,
                G,
                tiles_m,
                dp_tiles,
                n_dp,
                sk_a,
            )
        for q in range(total):
            if q + L < total:
                self.issue_b(
                    atom_b,
                    tBs,
                    tBg,
                    sfb,
                    sSB,
                    fullB,
                    emptyB,
                    q + L,
                    bid,
                    G,
                    tiles_m,
                    dp_tiles,
                    n_dp,
                    sk_a,
                )
            self.issue_a(
                atom_a,
                tAs,
                tAg,
                sfa,
                sSA,
                fullA,
                emptyA,
                q,
                bid,
                G,
                tiles_m,
                dp_tiles,
                n_dp,
                sk_a,
            )
        cute.arch.griddepcontrol_launch_dependents()

    # ----------------------------------------------------------- consumer
    @cute.jit
    def wait_stage(self, fullA, fullB, q):
        cute.arch.mbarrier_wait(fullB + q % self.SB, (q // self.SB) % 2)
        cute.arch.mbarrier_wait(fullA + q % self.SA, (q // self.SA) % 2)

    @cute.jit
    def release_stage(self, emptyA, emptyB, q, lane):
        cute.arch.sync_warp()
        if lane == 0:
            cute.arch.mbarrier_arrive(emptyA + q % self.SA)
            cute.arch.mbarrier_arrive(emptyB + q % self.SB)

    @cute.jit
    def load_sf(self, sfa_w, sfb_w, sSA, sSB, q, ks, kk, moff, noff, wm, wn, g, lane):
        KS = self.KS
        MI, NI, TM, TN = self.MI, self.NI, self.TM, self.TN
        sa = q % self.SA
        sb = q % self.SB
        for mi in cutlass.range_constexpr(MI):
            r = moff + wm * TM + mi * 16 + g + (lane % 2) * 8
            sfa_w[mi] = ptx.lds_u32(
                sSA.iterator.toint()
                + (sa * KS + ks) * SF_CHUNK
                + ((r % 32) * 4 + r // 32) * 4
            )
        for ni in cutlass.range_constexpr(NI):
            r = noff + wn * TN + ni * 8 + g
            sfb_w[ni] = ptx.lds_u32(
                sSB.iterator.toint()
                + ((sb * self.NBC + r // 128) * KS + ks) * SF_CHUNK
                + (((r % 32) * 4 + (r % 128) // 32) * 4)
            )
        if cutlass.const_expr(self.half):
            # Odd 64-wide boxes use bytes 2 and 3 of the shared 128-wide scale word.
            sh = (kk % 2) * 16
            for mi in cutlass.range_constexpr(MI):
                sfa_w[mi] = sfa_w[mi] >> sh
            for ni in cutlass.range_constexpr(NI):
                sfb_w[ni] = sfb_w[ni] >> sh

    def load_frags(self, a_st, b_st, kb, a_row0, a_ch, b_row0, b_ch):
        KW = self.KW
        af = []
        for mi in range(self.MI):
            row = a_row0 + mi * 16
            ch = (kb * 2 + a_ch) ^ self.swz_row(row)
            af.append(ptx.ldsm_x4(a_st + row * KW + ch * 16))
        bf = []
        for nj in range(self.NI // 2):
            row = b_row0 + nj * 16
            ch = (kb * 2 + b_ch) ^ self.swz_row(row)
            bf.append(ptx.ldsm_x4(b_st + row * KW + ch * 16))
        return af, bf

    def mma_step(self, acc, af, bf, sfa_w, sfb_w, kb):
        MI, NI = self.MI, self.NI
        for mi in range(MI):
            for ni in range(NI):
                bb = bf[ni // 2]
                b0 = bb[0] if ni % 2 == 0 else bb[2]
                b1 = bb[1] if ni % 2 == 0 else bb[3]
                i = (mi * NI + ni) * 4
                d0, d1, d2, d3 = ptx.mma_mxf8(
                    af[mi][0],
                    af[mi][1],
                    af[mi][2],
                    af[mi][3],
                    b0,
                    b1,
                    acc[i],
                    acc[i + 1],
                    acc[i + 2],
                    acc[i + 3],
                    sfa_w[mi],
                    sfb_w[ni],
                    kb,
                    kb,
                )
                acc[i] = d0
                acc[i + 1] = d1
                acc[i + 2] = d2
                acc[i + 3] = d3

    def frags_to_reg(self, fr, af, bf):
        n = 0
        for t in af + bf:
            for v in t:
                fr[n] = v
                n += 1

    def frags_from_reg(self, fr):
        af = [tuple(fr[4 * mi + j] for j in range(4)) for mi in range(self.MI)]
        o = 4 * self.MI
        bf = [tuple(fr[o + 4 * nj + j] for j in range(4)) for nj in range(self.NI // 2)]
        return af, bf

    def unit_addr(self, sA, sB, q, ks):
        a_st = sA.iterator.toint() + ((q % self.SA) * self.KS + ks) * self.A_BYTES
        b_st = sB.iterator.toint() + ((q % self.SB) * self.KS + ks) * self.B_BYTES
        return a_st, b_st

    @cute.jit
    def mainloop(
        self,
        acc,
        sfa_w,
        sfb_w,
        sfa_n,
        sfb_n,
        fr,
        sA,
        sB,
        sSA,
        sSB,
        fullA,
        emptyA,
        fullB,
        emptyB,
        q,
        k0,
        k1,
        moff,
        noff,
        wm,
        wn,
        g,
        lane,
        a_row0,
        a_ch,
        b_row0,
        b_ch,
    ):
        """Iterations [k0, k1) of one tile."""
        KS, NK = self.KS, self.NK
        self.wait_stage(fullA, fullB, q)
        self.load_sf(sfa_w, sfb_w, sSA, sSB, q, 0, k0, moff, noff, wm, wn, g, lane)
        a0, b0 = self.unit_addr(sA, sB, q, 0)
        af0, bf0 = self.load_frags(a0, b0, 0, a_row0, a_ch, b_row0, b_ch)
        self.frags_to_reg(fr, af0, bf0)
        for kk in range(k0, k1):
            for ks in cutlass.range_constexpr(KS):
                a_st, b_st = self.unit_addr(sA, sB, q, ks)
                if cutlass.const_expr(ks > 0):
                    self.load_sf(
                        sfa_w, sfb_w, sSA, sSB, q, ks, kk, moff, noff, wm, wn, g, lane
                    )
                    af, bf = self.load_frags(a_st, b_st, 0, a_row0, a_ch, b_row0, b_ch)
                else:
                    af, bf = self.frags_from_reg(fr)
                for kb in cutlass.range_constexpr(NK):
                    last = (ks == KS - 1) and (kb == NK - 1)
                    if cutlass.const_expr(kb + 1 < NK):
                        naf, nbf = self.load_frags(
                            a_st, b_st, kb + 1, a_row0, a_ch, b_row0, b_ch
                        )
                    if cutlass.const_expr(last):
                        self.release_stage(emptyA, emptyB, q, lane)
                        if kk + 1 < k1:
                            self.wait_stage(fullA, fullB, q + 1)
                            self.load_sf(
                                sfa_n,
                                sfb_n,
                                sSA,
                                sSB,
                                q + 1,
                                0,
                                kk + 1,
                                moff,
                                noff,
                                wm,
                                wn,
                                g,
                                lane,
                            )
                            pa, pb = self.unit_addr(sA, sB, q + 1, 0)
                            paf, pbf = self.load_frags(
                                pa, pb, 0, a_row0, a_ch, b_row0, b_ch
                            )
                            self.frags_to_reg(fr, paf, pbf)
                    if cutlass.const_expr(ks == 0 or self.Kb % KS == 0):
                        self.mma_step(acc, af, bf, sfa_w, sfb_w, kb)
                    else:
                        if kk * KS + ks < self.Kb:
                            self.mma_step(acc, af, bf, sfa_w, sfb_w, kb)
                    if cutlass.const_expr(kb + 1 < NK):
                        af, bf = naf, nbf
            if kk + 1 < k1:
                for mi in cutlass.range_constexpr(self.MI):
                    sfa_w[mi] = sfa_n[mi]
                for ni in cutlass.range_constexpr(self.NI):
                    sfb_w[ni] = sfb_n[ni]
            q = q + 1
        return q

    @cute.jit
    def consumer(
        self,
        sA,
        sB,
        sSA,
        sSB,
        fullA,
        emptyA,
        fullB,
        emptyB,
        flag,
        out,
        ws,
        cnt,
        M,
        tx,
        warp,
        lane,
        bid,
        G,
        tiles_m,
        dp_tiles,
        sk_a,
        sk_b,
    ):
        BM, BN = self.BM, self.BN
        MI, NI, TM, TN = self.MI, self.NI, self.TM, self.TN
        wm = warp // self.WN
        wn = warp % self.WN
        g = lane // 4
        tq = lane % 4
        acc = cute.make_rmem_tensor((self.ACC,), Float32)
        sfa_w = cute.make_rmem_tensor((MI,), Int32)
        sfb_w = cute.make_rmem_tensor((NI,), Int32)
        sfa_n = cute.make_rmem_tensor((MI,), Int32)
        sfb_n = cute.make_rmem_tensor((NI,), Int32)
        fr = cute.make_rmem_tensor((4 * MI + 4 * (NI // 2),), Int32)
        # ldmatrix row and 16-byte chunk (before swizzling) of this lane
        a_row0 = wm * TM + lane % 16
        a_ch = lane // 16
        b_row0 = wn * TN + lane % 8 + (lane // 16) * 8
        b_ch = (lane // 8) % 2
        q = Int32(0)
        t = bid
        it = sk_a
        while (t < dp_tiles) | (it < sk_b):
            tile, k0, k1, t, it = self.next_piece(t, it, G, dp_tiles, sk_b)
            mt = tile % tiles_m
            nt = tile // tiles_m
            moff = (mt * BM) % 128
            noff = (nt * BN) % 128
            acc.fill(0.0)
            q = self.mainloop(
                acc,
                sfa_w,
                sfb_w,
                sfa_n,
                sfb_n,
                fr,
                sA,
                sB,
                sSA,
                sSB,
                fullA,
                emptyA,
                fullB,
                emptyB,
                q,
                k0,
                k1,
                moff,
                noff,
                wm,
                wn,
                g,
                lane,
                a_row0,
                a_ch,
                b_row0,
                b_ch,
            )
            m0 = mt * BM
            n0 = nt * BN
            if cutlass.const_expr(self.streamk):
                if k0 == 0 and k1 == self.KI:
                    self.store_tile(acc, out, M, m0, n0, wm, wn, g, tq)
                else:
                    # Each contributor stores its partial in its own slot; the CTA
                    # completing the tile's K iterations sums them in CTA order.
                    tl = tile - dp_tiles
                    SI = (tiles_m * self.tiles_n - dp_tiles) * self.KI
                    first_it = tl * self.KI
                    c_lo = ((first_it + 1) * G + SI - 1) // SI - 1
                    c_hi = ((first_it + self.KI) * G + SI - 1) // SI - 1
                    maxc = (self.KI * G + SI - 1) // SI + 1
                    self.write_slot(acc, ws, tl * maxc + (bid - c_lo), tx)
                    ptx.bar_sync(1, self.CT)
                    if tx == 0:
                        ptx.fence_acq_rel_gpu()
                        # The counter accumulates K iterations; the CTA completing KI stores.
                        old = ptx.atom_add_acq_rel_gpu(
                            (cnt.iterator + tl).toint(), k1 - k0
                        )
                        last = Int32(0)
                        if old + (k1 - k0) == self.KI:
                            last = Int32(1)
                            cnt[tl] = Int32(0)
                        flag[0] = last
                    ptx.bar_sync(1, self.CT)
                    if flag[0] == 1:
                        acc.fill(0.0)
                        for c in range(c_lo, c_hi + 1):
                            if (c * SI) // G < ((c + 1) * SI) // G:
                                self.add_slot(acc, ws, tl * maxc + (c - c_lo), tx)
                        self.store_tile(acc, out, M, m0, n0, wm, wn, g, tq)
                    ptx.bar_sync(1, self.CT)
            else:
                self.store_tile(acc, out, M, m0, n0, wm, wn, g, tq)

    # ----------------------------------------------------------- epilogue
    @cute.jit
    def write_slot(self, acc, ws, slot, tx):
        base = ws.iterator.toint() + Int64(slot) * Int64(self.BM * self.BN * 4)
        for j in cutlass.range_constexpr(self.ACC // 4):
            ptx.stg_v4_f32(
                base + ((j * self.CT + tx) * 16),
                acc[4 * j],
                acc[4 * j + 1],
                acc[4 * j + 2],
                acc[4 * j + 3],
            )

    @cute.jit
    def add_slot(self, acc, ws, slot, tx):
        base = ws.iterator.toint() + Int64(slot) * Int64(self.BM * self.BN * 4)
        for j in cutlass.range_constexpr(self.ACC // 4):
            v0, v1, v2, v3 = ptx.ldg_cg_v4_f32(base + ((j * self.CT + tx) * 16))
            acc[4 * j] = acc[4 * j] + v0
            acc[4 * j + 1] = acc[4 * j + 1] + v1
            acc[4 * j + 2] = acc[4 * j + 2] + v2
            acc[4 * j + 3] = acc[4 * j + 3] + v3

    @cute.jit
    def store_tile(self, acc, out, M, m0, n0, wm, wn, g, tq):
        if cutlass.const_expr(self.N % 8 == 0 and self.NI % 4 == 0):
            self.store_tile_v4(acc, out, M, m0, n0, wm, wn, g, tq)
        else:
            self.store_tile_b32(acc, out, M, m0, n0, wm, wn, g, tq)

    @cute.jit
    def store_tile_v4(self, acc, out, M, m0, n0, wm, wn, g, tq):
        """16-byte stores: a 4x4 word transpose inside each lane quad turns the
        MMA C fragments (2 columns x 4 n8 tiles per lane) into 8 consecutive
        columns per lane."""
        N = self.N
        obase = out.iterator.toint()
        lane_q0 = (cute.arch.lane_idx() // 4) * 4
        for mi in cutlass.range_constexpr(self.MI):
            for h in cutlass.range_constexpr(2):
                row = m0 + wm * self.TM + mi * 16 + g + h * 8
                for nb in cutlass.range_constexpr(self.NI // 4):
                    w = [
                        ptx.pack_half2(
                            acc[(mi * self.NI + nb * 4 + j) * 4 + 2 * h],
                            acc[(mi * self.NI + nb * 4 + j) * 4 + 2 * h + 1],
                            self.out_f16,
                        )
                        for j in range(4)
                    ]
                    # Round r: lane t sends its word for n8 tile (t + r) % 4 and
                    # receives the word lane (t - r) % 4 holds for n8 tile t.
                    rec = []
                    for r in cutlass.range_constexpr(4):
                        sel = (tq + r) % 4
                        send = cutlass.select_(
                            sel == 1,
                            w[1],
                            cutlass.select_(
                                sel == 2, w[2], cutlass.select_(sel == 3, w[3], w[0])
                            ),
                        )
                        if cutlass.const_expr(r == 0):
                            rec.append(send)
                        else:
                            rec.append(ptx.shfl_idx(send, lane_q0 + (tq + 4 - r) % 4))
                    v = []
                    for j in cutlass.range_constexpr(4):
                        r = (tq + 4 - j) % 4  # round in which lane j's word arrived
                        v.append(
                            cutlass.select_(
                                r == 1,
                                rec[1],
                                cutlass.select_(
                                    r == 2,
                                    rec[2],
                                    cutlass.select_(r == 3, rec[3], rec[0]),
                                ),
                            )
                        )
                    col = n0 + wn * self.TN + (nb * 4 + tq) * 8
                    if (row < M) & (col < N):
                        ptx.stg_v4_b32(
                            obase + (Int64(row) * N + Int64(col)) * 2,
                            v[0],
                            v[1],
                            v[2],
                            v[3],
                        )

    @cute.jit
    def store_tile_b32(self, acc, out, M, m0, n0, wm, wn, g, tq):
        N = self.N
        obase = out.iterator.toint()
        for mi in cutlass.range_constexpr(self.MI):
            for ni in cutlass.range_constexpr(self.NI):
                i = (mi * self.NI + ni) * 4
                col = n0 + wn * self.TN + ni * 8 + tq * 2
                for h in cutlass.range_constexpr(2):
                    row = m0 + wm * self.TM + mi * 16 + g + h * 8
                    if row < M:
                        addr = obase + (Int64(row) * N + Int64(col)) * 2
                        p = ptx.pack_half2(
                            acc[i + 2 * h], acc[i + 2 * h + 1], self.out_f16
                        )
                        if cutlass.const_expr(N % 2 == 0):
                            if col < N:
                                ptx.stg_b32(addr, p)
                        else:
                            if col < N:
                                ptx.stg_b16(addr, p & 0xFFFF)
                            if col + 1 < N:
                                ptx.stg_b16(addr + 2, (p >> 16) & 0xFFFF)
