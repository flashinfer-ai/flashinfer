# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SM120/SM121 routed-expert MoE with BF16 activations and NVFP4 expert weights.

Weights are read in place from the canonical W4A4 NVFP4 storage: packed E2M1
nibbles, 128x4-swizzled E4M3 block scales and a per-expert FP32 global scale.
The same buffers therefore serve both W4A16 (this module) and W4A4 kernels.

Two execution paths, chosen by the host from the number of routed slots:

* one routed slot per expert tile: a single persistent kernel runs both GEMMs,
  the activation and the router-weighted combine;
* otherwise: expert routing, GEMM1, activation, GEMM2 and a BF16 finalize as
  separate launches, with GEMM rows tiled per expert.
"""

import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.runtime import from_dlpack

RTPB = 256  # threads per routing / activation / finalize CTA
BYT = 16  # packed weight bytes per thread per step (one LDG.128 = 32 E2M1 codes)
KPT = 32  # K values per thread per step
REGCAP = 128  # per-thread register target of the persistent kernel
GMUL = 10  # persistent grid = SM count * GMUL, capped by the tile count


def _sc_views(gSC, S, BMAX):
    # [stok(S)][be(BMAX)][bm0(BMAX)][bc(BMAX)][nb(1)] inside one int32 allocation
    return (
        cute.make_tensor(gSC.iterator, cute.make_layout(S)),
        cute.make_tensor((gSC.iterator + S).align(4), cute.make_layout(BMAX)),
        cute.make_tensor((gSC.iterator + (S + BMAX)).align(4), cute.make_layout(BMAX)),
        cute.make_tensor(
            (gSC.iterator + (S + 2 * BMAX)).align(4), cute.make_layout(BMAX)
        ),
        cute.make_tensor((gSC.iterator + (S + 3 * BMAX)).align(4), cute.make_layout(1)),
    )


# ======================================================================================
# routing: histogram(topk_ids) -> offsets -> expert-sorted slot list + m-block descriptors.
# CTA 0 routes; the remaining CTAs zero the fp32 accumulators.
# ======================================================================================
@cute.kernel
def k_route(
    gIds: cute.Tensor,
    gWts: cute.Tensor,
    gSC: cute.Tensor,
    gFS: cute.Tensor,
    E: cutlass.Constexpr,
    EP: cutlass.Constexpr,
    S: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    BM: cutlass.Constexpr,
    NOF4: cutlass.Constexpr,
    NH4: cutlass.Constexpr,
    BMAX: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    # One int32 and one fp32 scratch allocation carry every routing/staging buffer.
    gStok, gBe, gBm0, gBc, gNb = _sc_views(gSC, S, BMAX)
    gSwt = cute.make_tensor(
        (gFS.iterator + (OOF + NOF4 * 4)).align(4), cute.make_layout(S)
    )
    gHf = gFS
    gOFf = cute.make_tensor((gFS.iterator + OOF).align(16), cute.make_layout(NOF4 * 4))

    if bid == 0:
        smem = utils.SmemAllocator()
        cnt = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(EP), byte_alignment=16
        )
        off = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(EP), byte_alignment=16
        )
        bof = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(EP), byte_alignment=16
        )
        sc = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(2 * RTPB), byte_alignment=16
        )
        sb = smem.allocate_tensor(
            cutlass.Int32, cute.make_layout(2 * RTPB), byte_alignment=16
        )

        CH = EP // RTPB
        for q in cutlass.range_constexpr(CH):
            cnt[tid * CH + q] = cutlass.Int32(0)
        cute.arch.barrier()

        NIT = (S + RTPB - 1) // RTPB
        for j in cutlass.range(NIT, unroll=1):
            s = j * RTPB + tid
            if s < S:
                cute.arch.atomic_add(cnt.iterator + gIds[s], cutlass.Int32(1))
        cute.arch.barrier()

        ls = cutlass.Int32(0)
        lb = cutlass.Int32(0)
        for q in cutlass.range_constexpr(CH):
            c = cnt[tid * CH + q]
            ls = ls + c
            lb = lb + (c + (BM - 1)) // BM
        sc[tid] = cutlass.Int32(0)
        sb[tid] = cutlass.Int32(0)
        sc[RTPB + tid] = ls
        sb[RTPB + tid] = lb
        cute.arch.barrier()
        d = 1
        while d < RTPB:
            vs = sc[RTPB + tid - d]
            vb = sb[RTPB + tid - d]
            cute.arch.barrier()
            sc[RTPB + tid] = sc[RTPB + tid] + vs
            sb[RTPB + tid] = sb[RTPB + tid] + vb
            cute.arch.barrier()
            d = d * 2
        a = sc[RTPB + tid] - ls
        b = sb[RTPB + tid] - lb
        for q in cutlass.range_constexpr(CH):
            i = tid * CH + q
            c = cnt[i]
            off[i] = a
            bof[i] = b
            a = a + c
            b = b + (c + (BM - 1)) // BM
        if tid == RTPB - 1:
            gNb[0] = sb[2 * RTPB - 1]
        cute.arch.barrier()

        for q in cutlass.range_constexpr(CH):
            i = tid * CH + q
            c = cnt[i]
            o = off[i]
            b0 = bof[i]
            for r in cutlass.range((c + (BM - 1)) // BM, unroll=1):
                gBe[b0 + r] = i
                gBm0[b0 + r] = o + r * BM
                gBc[b0 + r] = cutlass.min(c - r * BM, cutlass.Int32(BM))
        for q in cutlass.range_constexpr(CH):
            i = tid * CH + q
            cnt[i] = off[i]
        cute.arch.barrier()
        for j in cutlass.range(NIT, unroll=1):
            s = j * RTPB + tid
            if s < S:
                p = cute.arch.atomic_add(cnt.iterator + gIds[s], cutlass.Int32(1))
                gStok[p] = s // TOPK
                gSwt[p] = gWts[s]
    else:
        z = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        for q in cutlass.range_constexpr(4):
            z[q] = cutlass.Float32(0.0)
        i4 = (bid - 1) * RTPB + tid
        if i4 < NOF4:
            cute.zipped_divide(gOFf, (4,))[(None, i4)].store(z.load())
        if cutlass.const_expr(NH4 > 0):
            j4 = i4 - NOF4
            if j4 >= 0:
                if j4 < NH4:
                    cute.zipped_divide(gHf, (4,))[(None, j4)].store(z.load())


# ======================================================================================
# device helpers (plain Python, inlined at trace time)
# ======================================================================================
def _g2r_atom(dtype, bits, stream=True):
    # read-once weight/scale stream: `.cs` allocates the lines evict-first in L1 and L2 so the
    # single-use packed weights do not push the reused activation / block-scale lines out.
    kw = {}
    if stream:
        kw["load_cache_mode"] = cute.nvgpu.common.LoadCacheMode.STREAMING
    return cute.make_copy_atom(
        cute.nvgpu.common.CopyG2ROp(), dtype, num_bits_per_copy=bits, **kw
    )


def _algn(*byte_strides):
    # largest power-of-two <= 16 that every dynamic byte stride of the address is a
    # multiple of; `Pointer.align` rounds the address up, so it must never over-promise
    a = 16
    while a > 1 and any(s % a for s in byte_strides):
        a //= 2
    return a


def _sf_row_base(n, CP):
    # physical offset of the 128x4-swizzled E4M3 block scale for logical row n, k-group 0
    r = n % 128
    return (n // 128) * (CP * 128) + (r % 32) * 16 + (r // 32) * 4


def _dot16(wf, rx, form=0):
    # Packed-fp16 product of 16 MACs, three trace-time forms.  All skip the 16 f16->f32
    # converts a scalar fp32 chain would pay; they differ in instruction count vs the
    # depth of the dependency chain the scheduler has to hide:
    #   0  8 mul + shallow add tree   ~16 ops, depth ~4
    #   1  8 chained f16x2 FMAs       ~9 ops,  depth 8  (needs >=16 row accumulators)
    #   2  two 4-deep FMA chains      ~10 ops, depth 4  (chain-1 cost, tree-like depth)
    # 8 terms per accumulator lane either way, so fp16 rounding stays ~1e-4 relative.
    if form == 2:
        wv = cute.zipped_divide(wf, (2,))
        xv = cute.zipped_divide(rx, (2,))
        a0 = wv[(None, 0)].load() * xv[(None, 0)].load()
        a1 = wv[(None, 4)].load() * xv[(None, 4)].load()
        for q in range(1, 4):
            a0 = a0 + wv[(None, q)].load() * xv[(None, q)].load()
            a1 = a1 + wv[(None, q + 4)].load() * xv[(None, q + 4)].load()
        r = (a0 + a1).reduce(cute.ReductionOp.ADD, cutlass.Float16(0.0), 0)
    elif form == 1:
        wv = cute.zipped_divide(wf, (2,))
        xv = cute.zipped_divide(rx, (2,))
        a = wv[(None, 0)].load() * xv[(None, 0)].load()
        for q in range(1, 8):
            a = a + wv[(None, q)].load() * xv[(None, q)].load()
        r = a.reduce(cute.ReductionOp.ADD, cutlass.Float16(0.0), 0)
    else:
        r = (wf.load() * rx.load()).reduce(
            cute.ReductionOp.ADD, cutlass.Float16(0.0), 0
        )
    return cutlass.Float16(r).to(cutlass.Float32)


def _decode16(rw4v, j, wf):
    wfv = cute.zipped_divide(wf, (8,))
    for q in range(2):
        s = rw4v[(None, j * 2 + q)].load()
        wfv[(None, q)].store(
            cute.TensorSSA(cute.arch.cvt_f4e2m1x8_to_f16x8(s), s.shape, cutlass.Float16)
        )


def _sfsplit(v):
    a = (
        (v & cutlass.Uint16(0xFF))
        .to(cutlass.Uint8)
        .bitcast(cutlass.Float8E4M3FN)
        .to(cutlass.Float32)
    )
    b = (
        (v >> cutlass.Uint16(8))
        .to(cutlass.Uint8)
        .bitcast(cutlass.Float8E4M3FN)
        .to(cutlass.Float32)
    )
    return a, b


def _shifts(tpr):
    return [1 << i for i in range(tpr.bit_length() - 1)]


# ======================================================================================
# GEMM 1 : h[slot, 0:N13] = x @ W13[e].T in FP32.
# ======================================================================================
@cute.kernel
def k_gemm1(
    gX: cute.Tensor,
    gW: cute.Tensor,
    gSF: cute.Tensor,
    gGS: cute.Tensor,
    gSC: cute.Tensor,
    gFS: cute.Tensor,
    gA: cute.Tensor,
    gIds: cute.Tensor,
    BMAX_: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
    NOFL: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    K: cutlass.Constexpr,
    CP: cutlass.Constexpr,
    NSF: cutlass.Constexpr,
    BM: cutlass.Constexpr,
    KT: cutlass.Constexpr,
    KTP: cutlass.Constexpr,
    TPR: cutlass.Constexpr,
    SPLIT: cutlass.Constexpr,
    TNR: cutlass.Constexpr,
    NOROUTE: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    S: cutlass.Constexpr,
    GX: cutlass.Constexpr,
    GY: cutlass.Constexpr,
    ZN4: cutlass.Constexpr,
    DOTC: cutlass.Constexpr = 0,
):
    tid, _, _ = cute.arch.thread_idx()
    by, bx, bz = cute.arch.block_idx()
    WATOM = _g2r_atom(cutlass.Uint8, 128)
    gStok, gBe, gBm0, gBc, gNb = _sc_views(gSC, S, BMAX_)
    gH = gFS
    gOFz = cute.make_tensor((gFS.iterator + OOF).align(16), cute.make_layout(NOFL))

    # BM == 1 makes the expert sort a no-op: block b is simply routing slot b, so the two
    # GEMMs read topk_ids/topk_weights directly and the routing kernel is not launched at
    # all.  GEMM1 then also owns the zeroing of the fp32 combine buffer.
    nblocks = cutlass.Int32(S) if cutlass.const_expr(NOROUTE) else gNb[0]
    if by < nblocks:
        TPB = TNR * TPR
        STEP = TPR * KPT
        SPAD = STEP + STEP // 4
        KP = K // 2
        SFST = TPR * 128
        KSUB = K // SPLIT
        TQ = TNR // 4
        NSUB = 128 // TNR
        MG = max(1, BM // 8)  # row-group granularity of the partial-block guard

        if cutlass.const_expr(NOROUTE):
            e = gIds[by]
            m0 = by
            mc = cutlass.Int32(1)
        else:
            e = gBe[by]
            m0 = gBm0[by]
            mc = gBc[by]
        if cutlass.const_expr(ZN4 > 0):
            NCT = GX * GY * SPLIT
            ZP = (ZN4 + NCT * TPB - 1) // (NCT * TPB)
            cid = by + GX * (bx + GY * bz)
            zc = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
            for q in cutlass.range_constexpr(4):
                zc[q] = cutlass.Float32(0.0)
            for p in cutlass.range_constexpr(ZP):
                i4 = cid * TPB + tid + p * (NCT * TPB)
                if i4 < ZN4:
                    cute.zipped_divide(gOFz, (4,))[(None, i4)].store(zc.load())
        nr = tid // TPR
        kh = tid % TPR
        # rows of a CTA span all four 32-row sub-groups of one 128-row block-scale tile, so
        # every 32B block-scale sector the CTA touches is fully consumed; consecutive lanes
        # of a warp still land on consecutive weight rows.
        n = (bx // NSUB) * 128 + (nr // TQ) * 32 + (bx % NSUB) * TQ + (nr % TQ)
        nsafe = cutlass.min(n, cutlass.Int32(N13 - 1))

        smem = utils.SmemAllocator()
        sx = smem.allocate_tensor(
            cutlass.Float16, cute.make_layout(BM * KTP), byte_alignment=16
        )
        sx8 = cute.zipped_divide(sx, (8,))

        acc = cute.make_rmem_tensor(cute.make_layout(BM), cutlass.Float32)
        for i in cutlass.range_constexpr(BM):
            acc[i] = cutlass.Float32(0.0)

        # `iterator + dynamic offset` drops the pointer's alignment to 1 byte, which makes
        # autovec_copy fall back to 16 scalar LDG.E.U8 per 16B chunk.  Every expert/row base
        # here is a multiple of KP bytes, so re-assert the provable alignment and the copy
        # lowers to a single LDG.128.
        wbase = cutlass.Int64(e) * cutlass.Int64(N13 * KP) + cutlass.Int64(
            nsafe
        ) * cutlass.Int64(KP)
        gWg = cute.zipped_divide(
            cute.make_tensor(
                (gW.iterator + wbase).align(_algn(KP)), cute.make_layout(KP)
            ),
            (BYT,),
        )
        gSF16 = cute.recast_tensor(
            cute.make_tensor(
                (gSF.iterator + cutlass.Int64(e) * cutlass.Int64(NSF)).align(
                    _algn(NSF)
                ),
                cute.make_layout(NSF),
            ),
            cutlass.Uint16,
        )
        sfg = (_sf_row_base(nsafe, CP) + (kh // 2) * 512 + (kh % 2) * 2) // 2
        xoff = kh * 40

        NST = KT // STEP
        rws = [
            cute.make_rmem_tensor(cute.make_layout(BYT), cutlass.Uint8)
            for _ in range(NST)
        ]
        rw4s = [
            cute.zipped_divide(cute.recast_tensor(x, cutlass.Float4E2M1FN), (8,))
            for x in rws
        ]
        wf = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rx = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rxv = cute.zipped_divide(rx, (8,))
        xb = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)

        NKT = KSUB // KT
        NPASS = (BM * KT + TPB * 8 - 1) // (TPB * 8)
        for kti in cutlass.range(NKT, unroll=1):
            kt = bz * KSUB + kti * KT
            kb = kt // STEP
            # Issue every weight and scale fetch of this k-tile before the SMEM tile
            # barrier so they stay in flight while the activation tile is staged.
            for st in cutlass.range_constexpr(NST):
                cute.copy(WATOM, gWg[(None, (kb + st) * TPR + kh)], rws[st])
            sfv = [gSF16[sfg + (kb + st) * SFST] for st in range(NST)]
            cute.arch.barrier()
            for p in cutlass.range_constexpr(NPASS):
                f = p * (TPB * 8) + tid * 8
                if f < BM * KT:
                    m = f // KT
                    kk = f - m * KT
                    if cutlass.const_expr(NOROUTE):
                        row = by // TOPK
                    else:
                        row = gStok[m0 + cutlass.min(m, mc - 1)]
                    g8 = cute.zipped_divide(gX[(row, None)], (8,))
                    cute.autovec_copy(g8[(None, (kt + kk) // 8)], xb)
                    sidx = m * KTP + kk + (kk // 32) * 8
                    sx8[(None, sidx // 8)].store(xb.load().to(cutlass.Float16))
            cute.arch.barrier()

            for st in cutlass.range_constexpr(NST):
                sh = _sfsplit(sfv[st])
                for j in cutlass.range_constexpr(2):
                    _decode16(rw4s[st], j, wf)
                    # CTA-uniform guard: skip the padding rows of an expert's last m-block.
                    for g in cutlass.range_constexpr(BM // MG):
                        if cutlass.const_expr(BM > 1):
                            run = cutlass.Int32(g * MG) < mc
                        else:
                            run = True
                        if run:
                            for mm in cutlass.range_constexpr(MG):
                                m = g * MG + mm
                                sbase = m * KTP + st * SPAD + xoff + j * 16
                                for q in cutlass.range_constexpr(2):
                                    cute.autovec_copy(
                                        sx8[(None, (sbase + q * 8) // 8)],
                                        rxv[(None, q)],
                                    )
                                acc[m] = acc[m] + _dot16(wf, rx, DOTC) * sh[j]

        gs = gGS[e]
        for i in cutlass.range_constexpr(BM):
            for sft in _shifts(TPR):
                acc[i] = acc[i] + cute.arch.shuffle_sync_bfly(acc[i], sft)
        if kh == 0:
            if n < N13:
                for m in cutlass.range_constexpr(BM):
                    if cutlass.Int32(m) < mc:
                        o = cutlass.Int64(m0 + m) * cutlass.Int64(N13) + cutlass.Int64(
                            n
                        )
                        if cutlass.const_expr(SPLIT > 1):
                            cute.arch.atomic_add(gH.iterator + o, acc[m] * gs)
                        else:
                            cute.make_tensor(
                                (gH.iterator + o).align(4), cute.make_layout(1)
                            )[0] = acc[m] * gs


# ======================================================================================
# activation: a[slot, 0:I] = bf16(SwiGLU / ReLU^2 of h)
# ======================================================================================
@cute.kernel
def k_act(
    gH: cute.Tensor,
    gA: cute.Tensor,
    I: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    NEL4: cutlass.Constexpr,
    SWIGLU: cutlass.Constexpr,
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    i4 = bid * RTPB + tid
    if i4 < NEL4:
        base = i4 * 4
        s = base // I
        nn = base - s * I
        hb = cutlass.Int64(s) * cutlass.Int64(N13) + cutlass.Int64(nn)
        fg = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        cute.autovec_copy(
            cute.make_tensor(
                (gH.iterator + hb).align(_algn(N13 * 4)), cute.make_layout(4)
            ),
            fg,
        )
        ob = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
        if cutlass.const_expr(SWIGLU):
            fu = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
            cute.autovec_copy(
                cute.make_tensor(
                    (gH.iterator + (hb + cutlass.Int64(I))).align(
                        _algn(N13 * 4, I * 4)
                    ),
                    cute.make_layout(4),
                ),
                fu,
            )
            for q in cutlass.range_constexpr(4):
                g = fg[q]
                ob[q] = (g / (cutlass.Float32(1.0) + cute.math.exp(-g)) * fu[q]).to(
                    cutlass.BFloat16
                )
        else:
            for q in cutlass.range_constexpr(4):
                r = cutlass.max(fg[q], cutlass.Float32(0.0))
                ob[q] = (r * r).to(cutlass.BFloat16)
        cute.autovec_copy(ob, cute.zipped_divide(gA, (4,))[(None, i4)])


# ======================================================================================
# GEMM 2 : out[t] += router_weight * (a @ W2[e].T)   (fp32 atomics, split-K capable)
# ======================================================================================
@cute.kernel
def k_gemm2(
    gA: cute.Tensor,
    gW: cute.Tensor,
    gSF: cute.Tensor,
    gGS: cute.Tensor,
    gSC: cute.Tensor,
    gFS: cute.Tensor,
    gIds: cute.Tensor,
    gWts: cute.Tensor,
    gOb: cute.Tensor,
    BMAX_: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
    NOF: cutlass.Constexpr,
    H: cutlass.Constexpr,
    K: cutlass.Constexpr,
    CP: cutlass.Constexpr,
    NSF: cutlass.Constexpr,
    BM: cutlass.Constexpr,
    KT: cutlass.Constexpr,
    KTP: cutlass.Constexpr,
    TPR: cutlass.Constexpr,
    SPLIT: cutlass.Constexpr,
    TNR: cutlass.Constexpr,
    NOROUTE: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    S: cutlass.Constexpr,
    AFUSE: cutlass.Constexpr,
    SWIG2: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    DOTC: cutlass.Constexpr = 0,
):
    tid, _, _ = cute.arch.thread_idx()
    by, bx, bz = cute.arch.block_idx()
    WATOM = _g2r_atom(cutlass.Uint8, 128)
    gStok, gBe, gBm0, gBc, gNb = _sc_views(gSC, S, BMAX_)
    gSwt = cute.make_tensor((gFS.iterator + (OOF + NOF)).align(4), cute.make_layout(S))
    gOF = cute.make_tensor((gFS.iterator + OOF).align(16), cute.make_layout(NOF))
    gH2 = gFS

    nblocks = cutlass.Int32(S) if cutlass.const_expr(NOROUTE) else gNb[0]
    if by < nblocks:
        TPB = TNR * TPR
        STEP = TPR * KPT
        SPAD = STEP + STEP // 4
        KP = K // 2
        SFST = TPR * 128
        KSUB = K // SPLIT
        TQ = TNR // 4
        NSUB = 128 // TNR
        MG = max(1, BM // 8)

        if cutlass.const_expr(NOROUTE):
            e = gIds[by]
            m0 = by
            mc = cutlass.Int32(1)
        else:
            e = gBe[by]
            m0 = gBm0[by]
            mc = gBc[by]
        nr = tid // TPR
        kh = tid % TPR
        n = (bx // NSUB) * 128 + (nr // TQ) * 32 + (bx % NSUB) * TQ + (nr % TQ)
        nsafe = cutlass.min(n, cutlass.Int32(H - 1))

        smem = utils.SmemAllocator()
        sx = smem.allocate_tensor(
            cutlass.Float16, cute.make_layout(BM * KTP), byte_alignment=16
        )
        sx8 = cute.zipped_divide(sx, (8,))

        acc = cute.make_rmem_tensor(cute.make_layout(BM), cutlass.Float32)
        for m in cutlass.range_constexpr(BM):
            acc[m] = cutlass.Float32(0.0)

        gWg = cute.zipped_divide(
            cute.make_tensor(
                (
                    gW.iterator
                    + (
                        cutlass.Int64(e) * cutlass.Int64(H * KP)
                        + cutlass.Int64(nsafe) * cutlass.Int64(KP)
                    )
                ).align(_algn(KP)),
                cute.make_layout(KP),
            ),
            (BYT,),
        )
        gSF16 = cute.recast_tensor(
            cute.make_tensor(
                (gSF.iterator + cutlass.Int64(e) * cutlass.Int64(NSF)).align(
                    _algn(NSF)
                ),
                cute.make_layout(NSF),
            ),
            cutlass.Uint16,
        )
        sfg2 = (_sf_row_base(nsafe, CP) + (kh // 2) * 512 + (kh % 2) * 2) // 2
        xoff = kh * 40

        NST = KT // STEP
        rws = [
            cute.make_rmem_tensor(cute.make_layout(BYT), cutlass.Uint8)
            for _ in range(NST)
        ]
        rw4s = [
            cute.zipped_divide(cute.recast_tensor(r, cutlass.Float4E2M1FN), (8,))
            for r in rws
        ]
        wf = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rx = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rxv = cute.zipped_divide(rx, (8,))
        xb = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)

        NKT = KSUB // KT
        NPASS = (BM * KT + TPB * 8 - 1) // (TPB * 8)
        for kti in cutlass.range(NKT, unroll=1):
            kt = bz * KSUB + kti * KT
            kb = kt // STEP
            for st in cutlass.range_constexpr(NST):
                cute.copy(WATOM, gWg[(None, (kb + st) * TPR + kh)], rws[st])
            sfv = [gSF16[sfg2 + (kb + st) * SFST] for st in range(NST)]
            cute.arch.barrier()
            for p in cutlass.range_constexpr(NPASS):
                f = p * (TPB * 8) + tid * 8
                if f < BM * KT:
                    m = f // KT
                    kk = f - m * KT
                    row = m0 + cutlass.min(m, mc - 1)
                    sidx = m * KTP + kk + (kk // 32) * 8
                    if cutlass.const_expr(AFUSE):
                        # SwiGLU / ReLU^2 applied while staging h, so the standalone
                        # activation pass and the bf16 intermediate buffer both disappear
                        hb = cutlass.Int64(row) * cutlass.Int64(N13) + cutlass.Int64(
                            kt + kk
                        )
                        fg = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                        cute.autovec_copy(
                            cute.make_tensor(
                                (gH2.iterator + hb).align(_algn(N13 * 4)),
                                cute.make_layout(8),
                            ),
                            fg,
                        )
                        ab = cute.make_rmem_tensor(
                            cute.make_layout(8), cutlass.BFloat16
                        )
                        if cutlass.const_expr(SWIG2):
                            fu = cute.make_rmem_tensor(
                                cute.make_layout(8), cutlass.Float32
                            )
                            cute.autovec_copy(
                                cute.make_tensor(
                                    (gH2.iterator + (hb + cutlass.Int64(K))).align(
                                        _algn(N13 * 4, K * 4)
                                    ),
                                    cute.make_layout(8),
                                ),
                                fu,
                            )
                            for q in cutlass.range_constexpr(8):
                                g = fg[q]
                                ab[q] = (
                                    g
                                    / (cutlass.Float32(1.0) + cute.math.exp(-g))
                                    * fu[q]
                                ).to(cutlass.BFloat16)
                        else:
                            for q in cutlass.range_constexpr(8):
                                r = cutlass.max(fg[q], cutlass.Float32(0.0))
                                ab[q] = (r * r).to(cutlass.BFloat16)
                        sx8[(None, sidx // 8)].store(ab.load().to(cutlass.Float16))
                    else:
                        g8 = cute.zipped_divide(
                            cute.make_tensor(
                                (
                                    gA.iterator + cutlass.Int64(row) * cutlass.Int64(K)
                                ).align(_algn(K * 2)),
                                cute.make_layout(K),
                            ),
                            (8,),
                        )
                        cute.autovec_copy(g8[(None, (kt + kk) // 8)], xb)
                        sx8[(None, sidx // 8)].store(xb.load().to(cutlass.Float16))
            cute.arch.barrier()

            for st in cutlass.range_constexpr(NST):
                g0, g1 = _sfsplit(sfv[st])
                for j in cutlass.range_constexpr(2):
                    _decode16(rw4s[st], j, wf)
                    sh = g0 if j == 0 else g1
                    for g in cutlass.range_constexpr(BM // MG):
                        if cutlass.const_expr(BM > 1):
                            run = cutlass.Int32(g * MG) < mc
                        else:
                            run = True
                        if run:
                            for mm in cutlass.range_constexpr(MG):
                                m = g * MG + mm
                                sbase = m * KTP + st * SPAD + xoff + j * 16
                                for q in cutlass.range_constexpr(2):
                                    cute.autovec_copy(
                                        sx8[(None, (sbase + q * 8) // 8)],
                                        rxv[(None, q)],
                                    )
                                acc[m] = acc[m] + _dot16(wf, rx, DOTC) * sh

        gs = gGS[e]
        for m in cutlass.range_constexpr(BM):
            for sft in _shifts(TPR):
                acc[m] = acc[m] + cute.arch.shuffle_sync_bfly(acc[m], sft)
        if kh == 0:
            if n < H:
                for m in cutlass.range_constexpr(BM):
                    if cutlass.Int32(m) < mc:
                        if cutlass.const_expr(NOROUTE):
                            cute.arch.atomic_add(
                                gOF.iterator + ((by // TOPK) * H + n),
                                acc[m] * gs * gWts[by],
                            )
                        else:
                            cute.arch.atomic_add(
                                gOF.iterator + (gStok[m0 + m] * H + n),
                                acc[m] * gs * gSwt[m0 + m],
                            )


@cute.kernel
def k_final(
    gFS: cute.Tensor, gOf: cute.Tensor, NOF4: cutlass.Constexpr, OOF: cutlass.Constexpr
):
    tid, _, _ = cute.arch.thread_idx()
    bid, _, _ = cute.arch.block_idx()
    gOFf = cute.make_tensor((gFS.iterator + OOF).align(16), cute.make_layout(NOF4 * 4))
    i4 = bid * RTPB + tid
    if i4 < NOF4:
        f = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        cute.autovec_copy(cute.zipped_divide(gOFf, (4,))[(None, i4)], f)
        o = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
        o.store(f.load().to(cutlass.BFloat16))
        cute.autovec_copy(o, cute.zipped_divide(gOf, (4,))[(None, i4)])


# ======================================================================================
# Path A (BM == 1): ONE fused persistent kernel.  A dynamic work-stealing dispenser hands
# out GEMM1 tiles first and GEMM2 tiles only once the GEMM1 pool is empty, so every GEMM1
# tile is already owned by a RUNNING CTA when a consumer starts spinning -- the wait is
# deadlock-free at any grid size (no co-residency requirement, no cooperative launch).
# The SwiGLU/ReLU^2 epilogue is folded into GEMM2's SMEM staging and the bf16 finalize is
# done by the last CTA arriving on a per-token counter, so the whole MoE layer -- routing,
# both GEMMs, activation and the router-weighted combine -- is a single kernel launch.
# ======================================================================================
@cute.jit
def _grab(gptr: cute.Pointer, sptr: cute.Pointer, tid: cutlass.Int32):
    # CTA-uniform work ticket: lane 0 bumps the global dispenser, broadcasts through one
    # shared word.  Raw-pointer atomics only -- a tensor subscript store inside these
    # nested regions re-materialises the tensor and breaks IR dominance.
    cute.arch.barrier()
    if tid == 0:
        v = cute.arch.atomic_add(gptr, cutlass.Int32(1), scope="gpu")
        cute.arch.atomic_exch(sptr, v)
    cute.arch.barrier()


@cute.jit
def _spin_ge(ptr: cute.Pointer, target: cutlass.Int32):
    # acquiring atomic READ in the loop CONDITION (a rebind in the body is not carried).
    # Bounded so a protocol bug is a fast wrong answer instead of a session-long hang.
    v = cutlass.Int32(0)
    n = cutlass.Int32(0)
    while v < target and n < cutlass.Int32(4000000):
        v = cute.arch.atomic_add(ptr, cutlass.Int32(0), sem="acquire", scope="gpu")
        n = n + cutlass.Int32(1)


@cute.kernel
def k_fused(
    gX: cute.Tensor,
    gIds: cute.Tensor,
    gWts: cute.Tensor,
    gW13: cute.Tensor,
    gSF13: cute.Tensor,
    gG13: cute.Tensor,
    gW2: cute.Tensor,
    gSF2: cute.Tensor,
    gG2: cute.Tensor,
    gFS: cute.Tensor,
    gSC: cute.Tensor,
    gOb: cute.Tensor,
    S: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    H: cutlass.Constexpr,
    I: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    SWIGLU: cutlass.Constexpr,
    CP13: cutlass.Constexpr,
    NSF13: cutlass.Constexpr,
    CP2: cutlass.Constexpr,
    NSF2: cutlass.Constexpr,
    KT1: cutlass.Constexpr,
    KTP1: cutlass.Constexpr,
    TPR1: cutlass.Constexpr,
    TN1: cutlass.Constexpr,
    NT1: cutlass.Constexpr,
    SP1: cutlass.Constexpr,
    NST1: cutlass.Constexpr,
    KT2: cutlass.Constexpr,
    KTP2: cutlass.Constexpr,
    TPR2: cutlass.Constexpr,
    TN2: cutlass.Constexpr,
    NT2: cutlass.Constexpr,
    SP2: cutlass.Constexpr,
    NST2: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
    CTROFF: cutlass.Constexpr,
    SDOFF: cutlass.Constexpr,
    TCOFF: cutlass.Constexpr,
    G: cutlass.Constexpr,
    SMEL: cutlass.Constexpr,
    FALGN: cutlass.Constexpr,
    STREAM: cutlass.Constexpr = True,
):
    tid, _, _ = cute.arch.thread_idx()
    WATOM = _g2r_atom(cutlass.Uint8, 128, STREAM)
    SATOM = _g2r_atom(cutlass.Uint32, 32, STREAM)
    TPB = TN1 * TPR1
    NI1 = S * NT1 * SP1
    NI2 = S * NT2 * SP2
    TGT1 = NT1 * SP1
    TGTF = TOPK * NT2 * SP2

    # per-phase scale-staging geometry: one 128x4-swizzle column group of this CTA's
    # 128-row block is TN*4 contiguous bytes, so the whole k-tile's scales are
    # NST*(TPR/2) such chunks -- staged with 4 B per lane, padded by one u32 per chunk
    # so the u16 read-back is conflict-free
    CHS1 = TN1 + 1
    NCB1 = NST1 * (TPR1 // 2)
    NU1 = NCB1 * TN1
    NLS1 = (NU1 + TPB - 1) // TPB
    CHS2 = TN2 + 1
    NCB2 = NST2 * (TPR2 // 2)
    NU2 = NCB2 * TN2
    NLS2 = (NU2 + TPB - 1) // TPB

    smem = utils.SmemAllocator()
    sdisp = smem.allocate_tensor(cutlass.Int32, cute.make_layout(4), byte_alignment=16)
    ssf32 = smem.allocate_tensor(
        cutlass.Uint32,
        cute.make_layout(max(NCB1 * CHS1, NCB2 * CHS2)),
        byte_alignment=16,
    )
    ssf16 = cute.recast_tensor(ssf32, cutlass.Uint16)
    sx = smem.allocate_tensor(
        cutlass.Float16, cute.make_layout(SMEL), byte_alignment=16
    )
    sx8 = cute.zipped_divide(sx, (8,))

    c0p = gSC.iterator + CTROFF
    c1p = gSC.iterator + (CTROFF + 1)
    c3p = gSC.iterator + (CTROFF + 2)
    sdp = gSC.iterator + SDOFF
    tcp = gSC.iterator + TCOFF
    dp = sdisp.iterator
    gH = gFS
    gOF = cute.make_tensor((gFS.iterator + OOF).align(FALGN), cute.make_layout(1))

    # ---------------- phase 1 : h[slot, 0:N13] = x[token] @ W13[e].T ------------------
    STEP1 = TPR1 * KPT
    SPAD1 = STEP1 + STEP1 // 4
    KP1 = H // 2
    KSUB1 = H // SP1
    TQ1 = TN1 // 4
    NSUB1 = 128 // TN1
    NKT1 = KSUB1 // KT1
    NPASS1 = (KT1 + TPB * 8 - 1) // (TPB * 8)
    nr1 = tid // TPR1
    kh1 = tid % TPR1
    xoff1 = kh1 * 40
    sfs1 = (kh1 // 2) * (2 * CHS1) + (nr1 % TQ1) * 8 + (nr1 // TQ1) * 2 + (kh1 % 2)

    _grab(c0p, dp, tid)
    i = cute.arch.atomic_add(dp, cutlass.Int32(0))
    while i < cutlass.Int32(NI1):
        # every thread has now consumed the published ticket, so lane 0 may take the next
        # one; the barriers already inside this tile make it visible before the read below
        cute.arch.barrier()
        if tid == 0:
            cute.arch.atomic_exch(
                dp, cute.arch.atomic_add(c0p, cutlass.Int32(1), scope="gpu")
            )
        # slot varies FASTEST, as in the separate-launch grid: every routed slot is in
        # flight together so two slots on the same expert share its stream through L2
        by = i % cutlass.Int32(S)
        r = i // cutlass.Int32(S)
        bx = r % cutlass.Int32(NT1)
        bz = r // cutlass.Int32(NT1)
        row = by // cutlass.Int32(TOPK)
        e = gIds[(row, by - row * cutlass.Int32(TOPK))]
        n = (bx // NSUB1) * 128 + (nr1 // TQ1) * 32 + (bx % NSUB1) * TQ1 + (nr1 % TQ1)
        nsafe = cutlass.min(n, cutlass.Int32(N13 - 1))
        acc = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
        acc[0] = cutlass.Float32(0.0)
        wbase = cutlass.Int64(e) * cutlass.Int64(N13 * KP1) + cutlass.Int64(
            nsafe
        ) * cutlass.Int64(KP1)
        gWg = cute.zipped_divide(
            cute.make_tensor(
                (gW13.iterator + wbase).align(_algn(KP1)), cute.make_layout(KP1)
            ),
            (BYT,),
        )
        gSF32 = cute.recast_tensor(
            cute.make_tensor(
                (gSF13.iterator + cutlass.Int64(e) * cutlass.Int64(NSF13)).align(
                    _algn(NSF13)
                ),
                cute.make_layout(NSF13),
            ),
            cutlass.Uint32,
        )
        sfb32 = ((bx // NSUB1) * (CP13 * 128) + (bx % NSUB1) * (TQ1 * 16)) // 4
        rws = [
            cute.make_rmem_tensor(cute.make_layout(BYT), cutlass.Uint8)
            for _ in range(NST1)
        ]
        rw4s = [
            cute.zipped_divide(cute.recast_tensor(x, cutlass.Float4E2M1FN), (8,))
            for x in rws
        ]
        sfr = cute.make_rmem_tensor(cute.make_layout(NLS1), cutlass.Uint32)
        sfrv = cute.zipped_divide(sfr, (1,))
        wf = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rx = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rxv = cute.zipped_divide(rx, (8,))
        xbs = [
            cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
            for _ in range(NPASS1)
        ]
        for kti in cutlass.range(NKT1, unroll=1):
            kt = bz * KSUB1 + kti * KT1
            kb = kt // STEP1
            for st in cutlass.range_constexpr(NST1):
                cute.copy(WATOM, gWg[(None, (kb + st) * TPR1 + kh1)], rws[st])
            sfg32 = sfb32 + kb * ((TPR1 // 2) * 128)
            for it in cutlass.range_constexpr(NLS1):
                u = it * TPB + tid
                if u < NU1:
                    cute.copy(
                        SATOM,
                        cute.make_tensor(
                            (
                                gSF32.iterator + (sfg32 + (u // TN1) * 128 + (u % TN1))
                            ).align(4),
                            cute.make_layout(1),
                        ),
                        sfrv[(None, it)],
                    )
            # the activation fetch is issued BEFORE the tile barrier so its latency
            # overlaps the weight/scale fetches instead of stacking behind them
            for p in cutlass.range_constexpr(NPASS1):
                f = p * (TPB * 8) + tid * 8
                if f < KT1:
                    g8 = cute.zipped_divide(gX[(row, None)], (8,))
                    cute.autovec_copy(g8[(None, (kt + f) // 8)], xbs[p])
            cute.arch.barrier()
            for it in cutlass.range_constexpr(NLS1):
                u = it * TPB + tid
                if u < NU1:
                    ssf32[(u // TN1) * CHS1 + (u % TN1)] = sfr[it]
            for p in cutlass.range_constexpr(NPASS1):
                f = p * (TPB * 8) + tid * 8
                if f < KT1:
                    sidx = f + (f // 32) * 8
                    sx8[(None, sidx // 8)].store(xbs[p].load().to(cutlass.Float16))
            cute.arch.barrier()
            for st in cutlass.range_constexpr(NST1):
                g0, g1 = _sfsplit(ssf16[sfs1 + st * ((TPR1 // 2) * 2 * CHS1)])
                for jj in cutlass.range_constexpr(2):
                    _decode16(rw4s[st], jj, wf)
                    sh = g0 if jj == 0 else g1
                    sbase = st * SPAD1 + xoff1 + jj * 16
                    for q in cutlass.range_constexpr(2):
                        cute.autovec_copy(
                            sx8[(None, (sbase + q * 8) // 8)], rxv[(None, q)]
                        )
                    acc[0] = acc[0] + _dot16(wf, rx, 0) * sh
        gs1 = gG13[e]
        sfl1 = _shifts(TPR1)
        for si in cutlass.range_constexpr(len(sfl1)):
            acc[0] = acc[0] + cute.arch.shuffle_sync_bfly(acc[0], sfl1[si])
        if kh1 == 0:
            if n < N13:
                o = cutlass.Int64(by) * cutlass.Int64(N13) + cutlass.Int64(n)
                if cutlass.const_expr(SP1 > 1):
                    cute.arch.atomic_add(gH.iterator + o, acc[0] * gs1)
                else:
                    cute.make_tensor((gH.iterator + o).align(4), cute.make_layout(1))[
                        0
                    ] = acc[0] * gs1
        cute.arch.barrier()
        if tid == 0:
            cute.arch.atomic_add(sdp + by, cutlass.Int32(1), sem="release", scope="gpu")
        i = cute.arch.atomic_add(dp, cutlass.Int32(0))

    # ---------------- phase 2 : out[token] += r * (act(h) @ W2[e].T) ------------------
    STEP2 = TPR2 * KPT
    SPAD2 = STEP2 + STEP2 // 4
    KP2 = I // 2
    KSUB2 = I // SP2
    TQ2 = TN2 // 4
    NSUB2 = 128 // TN2
    NKT2 = KSUB2 // KT2
    NPASS2 = (KT2 + TPB * 8 - 1) // (TPB * 8)
    NFP = (H + TPB * 4 - 1) // (TPB * 4)
    nr2 = tid // TPR2
    kh2 = tid % TPR2
    xoff2 = kh2 * 40
    sfs2 = (kh2 // 2) * (2 * CHS2) + (nr2 % TQ2) * 8 + (nr2 // TQ2) * 2 + (kh2 % 2)

    _grab(c1p, dp, tid)
    i2 = cute.arch.atomic_add(dp, cutlass.Int32(0))
    while i2 < cutlass.Int32(NI2):
        cute.arch.barrier()
        if tid == 0:
            cute.arch.atomic_exch(
                dp, cute.arch.atomic_add(c1p, cutlass.Int32(1), scope="gpu")
            )
        by = i2 % cutlass.Int32(S)
        r = i2 // cutlass.Int32(S)
        bx = r % cutlass.Int32(NT2)
        bz = r // cutlass.Int32(NT2)
        tok = by // cutlass.Int32(TOPK)
        ksl = by - tok * cutlass.Int32(TOPK)
        e = gIds[(tok, ksl)]
        n = (bx // NSUB2) * 128 + (nr2 // TQ2) * 32 + (bx % NSUB2) * TQ2 + (nr2 % TQ2)
        nsafe = cutlass.min(n, cutlass.Int32(H - 1))
        acc2 = cute.make_rmem_tensor(cute.make_layout(1), cutlass.Float32)
        acc2[0] = cutlass.Float32(0.0)
        wbase2 = cutlass.Int64(e) * cutlass.Int64(H * KP2) + cutlass.Int64(
            nsafe
        ) * cutlass.Int64(KP2)
        gWg2 = cute.zipped_divide(
            cute.make_tensor(
                (gW2.iterator + wbase2).align(_algn(KP2)), cute.make_layout(KP2)
            ),
            (BYT,),
        )
        gSF62 = cute.recast_tensor(
            cute.make_tensor(
                (gSF2.iterator + cutlass.Int64(e) * cutlass.Int64(NSF2)).align(
                    _algn(NSF2)
                ),
                cute.make_layout(NSF2),
            ),
            cutlass.Uint32,
        )
        sfb2 = ((bx // NSUB2) * (CP2 * 128) + (bx % NSUB2) * (TQ2 * 16)) // 4
        rws2 = [
            cute.make_rmem_tensor(cute.make_layout(BYT), cutlass.Uint8)
            for _ in range(NST2)
        ]
        rw4s2 = [
            cute.zipped_divide(cute.recast_tensor(x, cutlass.Float4E2M1FN), (8,))
            for x in rws2
        ]
        sfr2 = cute.make_rmem_tensor(cute.make_layout(NLS2), cutlass.Uint32)
        sfrv2 = cute.zipped_divide(sfr2, (1,))
        wf2 = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rx2 = cute.make_rmem_tensor(cute.make_layout(16), cutlass.Float16)
        rxv2 = cute.zipped_divide(rx2, (8,))
        for kti in cutlass.range(NKT2, unroll=1):
            kt = bz * KSUB2 + kti * KT2
            kb = kt // STEP2
            for st in cutlass.range_constexpr(NST2):
                cute.copy(WATOM, gWg2[(None, (kb + st) * TPR2 + kh2)], rws2[st])
            sfg2 = sfb2 + kb * ((TPR2 // 2) * 128)
            for it in cutlass.range_constexpr(NLS2):
                u = it * TPB + tid
                if u < NU2:
                    cute.copy(
                        SATOM,
                        cute.make_tensor(
                            (
                                gSF62.iterator + (sfg2 + (u // TN2) * 128 + (u % TN2))
                            ).align(4),
                            cute.make_layout(1),
                        ),
                        sfrv2[(None, it)],
                    )
            # the producer wait only gates the h reads below, so it is issued AFTER this
            # tile's weight/scale fetches -- the latencies overlap and the barrier that
            # already separates the staging pass doubles as the CTA-wide acquire
            if kti == 0:
                if tid == 0:
                    _spin_ge(sdp + by, cutlass.Int32(TGT1))
            cute.arch.barrier()
            for it in cutlass.range_constexpr(NLS2):
                u = it * TPB + tid
                if u < NU2:
                    ssf32[(u // TN2) * CHS2 + (u % TN2)] = sfr2[it]
            for p in cutlass.range_constexpr(NPASS2):
                f = p * (TPB * 8) + tid * 8
                if f < KT2:
                    hb = cutlass.Int64(by) * cutlass.Int64(N13) + cutlass.Int64(kt + f)
                    fg = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                    cute.autovec_copy(
                        cute.make_tensor(
                            (gH.iterator + hb).align(_algn(N13 * 4)),
                            cute.make_layout(8),
                        ),
                        fg,
                    )
                    ab = cute.make_rmem_tensor(cute.make_layout(8), cutlass.BFloat16)
                    if cutlass.const_expr(SWIGLU):
                        fu = cute.make_rmem_tensor(cute.make_layout(8), cutlass.Float32)
                        cute.autovec_copy(
                            cute.make_tensor(
                                (gH.iterator + (hb + cutlass.Int64(I))).align(
                                    _algn(N13 * 4, I * 4)
                                ),
                                cute.make_layout(8),
                            ),
                            fu,
                        )
                        for q in cutlass.range_constexpr(8):
                            g = fg[q]
                            ab[q] = (
                                g / (cutlass.Float32(1.0) + cute.math.exp(-g)) * fu[q]
                            ).to(cutlass.BFloat16)
                    else:
                        for q in cutlass.range_constexpr(8):
                            rr = cutlass.max(fg[q], cutlass.Float32(0.0))
                            ab[q] = (rr * rr).to(cutlass.BFloat16)
                    sidx = f + (f // 32) * 8
                    sx8[(None, sidx // 8)].store(ab.load().to(cutlass.Float16))
            cute.arch.barrier()
            for st in cutlass.range_constexpr(NST2):
                g0, g1 = _sfsplit(ssf16[sfs2 + st * ((TPR2 // 2) * 2 * CHS2)])
                for jj in cutlass.range_constexpr(2):
                    _decode16(rw4s2[st], jj, wf2)
                    sh = g0 if jj == 0 else g1
                    sbase = st * SPAD2 + xoff2 + jj * 16
                    for q in cutlass.range_constexpr(2):
                        cute.autovec_copy(
                            sx8[(None, (sbase + q * 8) // 8)], rxv2[(None, q)]
                        )
                    acc2[0] = acc2[0] + _dot16(wf2, rx2, 0) * sh
        gs2 = gG2[e]
        sfl2 = _shifts(TPR2)
        for si in cutlass.range_constexpr(len(sfl2)):
            acc2[0] = acc2[0] + cute.arch.shuffle_sync_bfly(acc2[0], sfl2[si])
        if kh2 == 0:
            if n < H:
                cute.arch.atomic_add(
                    gOF.iterator + (tok * cutlass.Int32(H) + n),
                    acc2[0] * gs2 * gWts[(tok, ksl)],
                )
        cute.arch.barrier()
        if tid == 0:
            old = cute.arch.atomic_add(
                tcp + tok, cutlass.Int32(1), sem="acq_rel", scope="gpu"
            )
            cute.arch.atomic_exch(dp + 1, old)
        cute.arch.barrier()
        lastf = cute.arch.atomic_add(dp + 1, cutlass.Int32(0))
        # last contributor to this token converts fp32 -> bf16 and re-zeroes the fp32
        # staging slice, so the combine buffer needs no prologue clear on the next call
        if lastf == cutlass.Int32(TGTF - 1):
            fp = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
            ob = cute.make_rmem_tensor(cute.make_layout(4), cutlass.BFloat16)
            zf = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
            for q in cutlass.range_constexpr(4):
                zf[q] = cutlass.Float32(0.0)
            for p in cutlass.range_constexpr(NFP):
                a4 = p * (TPB * 4) + tid * 4
                if a4 < H:
                    fo = cutlass.Int64(tok) * cutlass.Int64(H) + cutlass.Int64(a4)
                    srct = cute.make_tensor(
                        (gOF.iterator + fo).align(FALGN), cute.make_layout(4)
                    )
                    cute.autovec_copy(srct, fp)
                    ob.store(fp.load().to(cutlass.BFloat16))
                    cute.autovec_copy(
                        ob,
                        cute.make_tensor(
                            (gOb.iterator + fo).align(FALGN // 2), cute.make_layout(4)
                        ),
                    )
                    cute.autovec_copy(zf, srct)
            if tid == 0:
                cute.arch.atomic_exch(tcp + tok, cutlass.Int32(0))
        i2 = cute.arch.atomic_add(dp, cutlass.Int32(0))

    # ------- self-clean: the last CTA out resets every dispenser / slot counter -------
    cute.arch.barrier()
    if tid == 0:
        d = cute.arch.atomic_add(c3p, cutlass.Int32(1), sem="acq_rel", scope="gpu")
        cute.arch.atomic_exch(dp + 2, d)
    cute.arch.barrier()
    dd = cute.arch.atomic_add(dp + 2, cutlass.Int32(0))
    if dd == cutlass.Int32(G - 1):
        for q in cutlass.range_constexpr((S + TPB - 1) // TPB):
            jz = q * TPB + tid
            if jz < S:
                cute.arch.atomic_exch(sdp + jz, cutlass.Int32(0))
        if tid == 0:
            cute.arch.atomic_exch(c0p, cutlass.Int32(0))
            cute.arch.atomic_exch(c1p, cutlass.Int32(0))
            cute.arch.atomic_exch(c3p, cutlass.Int32(0))


@cute.jit
def moe_jitA(
    gX,
    gIds,
    gWts,
    gW13,
    gSF13,
    gG13,
    gW2,
    gSF2,
    gG2,
    gFS,
    gSC,
    gO,
    stream: cuda.CUstream,
    S: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    H: cutlass.Constexpr,
    I: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    SWIGLU: cutlass.Constexpr,
    CP13: cutlass.Constexpr,
    NSF13: cutlass.Constexpr,
    CP2: cutlass.Constexpr,
    NSF2: cutlass.Constexpr,
    KT1: cutlass.Constexpr,
    KTP1: cutlass.Constexpr,
    TPR1: cutlass.Constexpr,
    TN1: cutlass.Constexpr,
    NT1: cutlass.Constexpr,
    SP1: cutlass.Constexpr,
    NST1: cutlass.Constexpr,
    KT2: cutlass.Constexpr,
    KTP2: cutlass.Constexpr,
    TPR2: cutlass.Constexpr,
    TN2: cutlass.Constexpr,
    NT2: cutlass.Constexpr,
    SP2: cutlass.Constexpr,
    NST2: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
    CTROFF: cutlass.Constexpr,
    SDOFF: cutlass.Constexpr,
    TCOFF: cutlass.Constexpr,
    G: cutlass.Constexpr,
    SMEL: cutlass.Constexpr,
    FALGN: cutlass.Constexpr,
    MBPM: cutlass.Constexpr,
    STREAM: cutlass.Constexpr,
):
    k_fused(
        gX,
        gIds,
        gWts,
        gW13,
        gSF13,
        gG13,
        gW2,
        gSF2,
        gG2,
        gFS,
        gSC,
        gO,
        S,
        TOPK,
        H,
        I,
        N13,
        SWIGLU,
        CP13,
        NSF13,
        CP2,
        NSF2,
        KT1,
        KTP1,
        TPR1,
        TN1,
        NT1,
        SP1,
        NST1,
        KT2,
        KTP2,
        TPR2,
        TN2,
        NT2,
        SP2,
        NST2,
        OOF,
        CTROFF,
        SDOFF,
        TCOFF,
        G,
        SMEL,
        FALGN,
        STREAM,
    ).launch(
        stream=stream,
        grid=(G, 1, 1),
        block=(TN1 * TPR1, 1, 1),
        smem=SMEL * 2
        + 16
        + max((TN1 + 1) * NST1 * (TPR1 // 2), (TN2 + 1) * NST2 * (TPR2 // 2)) * 4
        + 16,
        min_blocks_per_mp=MBPM,
    )


@cute.jit
def moe_jit(
    gX,
    gIds,
    gWts,
    gW13,
    gSF13,
    gG13,
    gW2,
    gSF2,
    gG2,
    gA,
    gFS,
    gO,
    gSC,
    stream: cuda.CUstream,
    H: cutlass.Constexpr,
    I: cutlass.Constexpr,
    E: cutlass.Constexpr,
    EP: cutlass.Constexpr,
    S: cutlass.Constexpr,
    TOPK: cutlass.Constexpr,
    BM: cutlass.Constexpr,
    SWIGLU: cutlass.Constexpr,
    N13: cutlass.Constexpr,
    CP13: cutlass.Constexpr,
    CP2: cutlass.Constexpr,
    NSF13: cutlass.Constexpr,
    NSF2: cutlass.Constexpr,
    KT1: cutlass.Constexpr,
    KTP1: cutlass.Constexpr,
    TPR1: cutlass.Constexpr,
    SP1: cutlass.Constexpr,
    KT2: cutlass.Constexpr,
    KTP2: cutlass.Constexpr,
    TPR2: cutlass.Constexpr,
    SP2: cutlass.Constexpr,
    BMAX: cutlass.Constexpr,
    NOF4: cutlass.Constexpr,
    NH4: cutlass.Constexpr,
    NEL4: cutlass.Constexpr,
    NT13: cutlass.Constexpr,
    NT2: cutlass.Constexpr,
    TN1: cutlass.Constexpr,
    TN2: cutlass.Constexpr,
    NOROUTE: cutlass.Constexpr,
    ZN4: cutlass.Constexpr,
    OOF: cutlass.Constexpr,
    AFUSE: cutlass.Constexpr,
    DOTC: cutlass.Constexpr = 0,
):
    nzb = (NOF4 + NH4 + RTPB - 1) // RTPB
    if cutlass.const_expr(not NOROUTE):
        k_route(gIds, gWts, gSC, gFS, E, EP, S, TOPK, BM, NOF4, NH4, BMAX, OOF).launch(
            stream=stream,
            grid=(1 + nzb, 1, 1),
            block=(RTPB, 1, 1),
            smem=(3 * EP + 4 * RTPB) * 4,
        )
    k_gemm1(
        gX,
        gW13,
        gSF13,
        gG13,
        gSC,
        gFS,
        gA,
        gIds,
        BMAX,
        OOF,
        NOF4 * 4,
        N13,
        H,
        CP13,
        NSF13,
        BM,
        KT1,
        KTP1,
        TPR1,
        SP1,
        TN1,
        NOROUTE,
        TOPK,
        S,
        BMAX,
        NT13,
        ZN4,
        DOTC,
    ).launch(
        stream=stream,
        grid=(BMAX, NT13, SP1),
        block=(TN1 * TPR1, 1, 1),
        smem=BM * KTP1 * 2,
    )
    if cutlass.const_expr(not AFUSE):
        k_act(gFS, gA, I, N13, NEL4, SWIGLU).launch(
            stream=stream, grid=((NEL4 + RTPB - 1) // RTPB, 1, 1), block=(RTPB, 1, 1)
        )
    k_gemm2(
        gA,
        gW2,
        gSF2,
        gG2,
        gSC,
        gFS,
        gIds,
        gWts,
        gO,
        BMAX,
        OOF,
        NOF4 * 4,
        H,
        I,
        CP2,
        NSF2,
        BM,
        KT2,
        KTP2,
        TPR2,
        SP2,
        TN2,
        NOROUTE,
        TOPK,
        S,
        AFUSE,
        SWIGLU,
        N13,
        DOTC,
    ).launch(
        stream=stream,
        grid=(BMAX, NT2, SP2),
        block=(TN2 * TPR2, 1, 1),
        smem=BM * KTP2 * 2,
    )
    k_final(gFS, gO, NOF4, OOF).launch(
        stream=stream, grid=((NOF4 + RTPB - 1) // RTPB, 1, 1), block=(RTPB, 1, 1)
    )


# ======================================================================================
# host glue
# ======================================================================================
def _pick_bm(S, E):
    # rows-per-CTA = next power of two >= the mean slots per expert.  Rounding UP costs no
    # extra padded rows versus rounding down (ceil(c/BM)*BM is the same for BM in [c, 2c))
    # but halves how often each expert's weights have to be streamed.
    r = S / max(1, min(E, S))
    bm = 1
    while bm < 32 and bm < r:
        bm *= 2
    return bm


def _divs(n):
    return [d for d in range(1, n + 1) if n % d == 0]


def _plan(
    K,
    N,
    nblk,
    BM,
    MAXNST=6,
    BUDGET=40960,
    GPU_THREADS=73728,
    max_split=None,
    tpb_req=None,
    ret_score=False,
    pernst_w=0.10,
):
    """Pick (threads-per-row, rows-per-CTA, split-K, loads-in-flight, n-tiles).

    Larger TPR gives each warp longer contiguous weight segments (TPR*16 bytes of one
    packed row per LDG.128); larger NST keeps more independent loads in flight; split-K
    only grows the grid until the launch covers the GPU; and a small N-tile count keeps
    the redundant activation re-staging (NT*BM*4/N of the weight bytes) in check.
    """
    nb128 = (N + 127) // 128
    best = None
    for tpr in (2, 4, 8):
        step = tpr * KPT
        if K % step:
            continue
        nstep = K // step
        # a small row-tile multiplies the number of N-tiles, and every N-tile re-stages the
        # same activation rows; that redundancy is only affordable while BM == 1.
        for TNR in (16, 32, 64, 128) if BM == 1 else (128,):
            tpb = TNR * tpr
            if tpb < 128 or tpb > 512:
                continue
            if tpb_req is not None and tpb != tpb_req:
                continue
            NT = nb128 * (128 // TNR)
            for split in _divs(nstep):
                if max_split is not None and split > max_split:
                    continue
                per = nstep // split
                if split > 1 and per < 2:
                    continue  # a split that leaves one k-step per CTA is all latency
                nst = 0
                for d in _divs(per):
                    if d <= MAXNST and BM * step * d * 5 // 2 <= BUDGET:
                        nst = max(nst, d)
                if nst == 0:
                    continue
                w = (nblk * NT * split * tpb) / float(GPU_THREADS)
                s = 0.45 * math.log(tpr) + 0.35 * math.log(min(nst, 5))
                s -= 0.25 * (NT * BM * 4.0 / N)
                s -= 0.03 * abs(math.log(tpb / 256.0))
                # k-tiles inside a CTA are serialised (the SMEM tile has to be re-staged
                # between them), so a config that fits the whole k-range in one tile keeps
                # every load of the CTA in flight at once.
                s -= pernst_w * math.log(per / nst)
                # with few CTAs per SM the static round-up of the last wave dominates
                s -= 0.6 / max(1.0, nblk * NT * split / 48.0)
                # resident-warp estimate: in-flight weight registers, decode buffers and
                # FP32 accumulators scale with nst and BM
                regs = 32 + 4 * nst + BM
                smem = max(1, BM * step * nst * 5 // 2)
                ctas = min(24, 65536 // (regs * tpb), 102400 // smem)
                s += 0.35 * math.log(max(1.0, min(48.0, ctas * tpb / 32.0)) / 12.0)
                if w < 1.0:
                    s -= 3.0 * (1.0 - w)
                elif w < 2.0:
                    s -= 0.20 * (2.0 - w)
                elif w > 400.0:
                    s -= 0.01 * (w - 400.0)
                if best is None or s > best[0]:
                    best = (s, tpr, TNR, split, nst, NT)
    if best is None:
        if ret_score:
            return None
        nb = (N + 127) // 128
        nst = 1
        for d in _divs(K // KPT):
            if d <= MAXNST and BM * KPT * d * 5 // 2 <= BUDGET:
                nst = max(nst, d)
        return 1, 128, 1, nst, nb
    if ret_score:
        return best
    return best[1], best[2], best[3], best[4], best[5]


def _planA(H, N13, I, S):
    # Both phases of the fused kernel share one CTA shape, so the two tilings are chosen
    # jointly: pick the threads-per-CTA whose GEMM1 + GEMM2 plans score best together.
    bestc = None
    for tpb in (128, 256, 512):
        p1 = _plan(H, N13, S, 1, max_split=1, tpb_req=tpb, ret_score=True)
        p2 = _plan(I, H, S, 1, tpb_req=tpb, ret_score=True)
        if p1 is None or p2 is None:
            continue
        sc = p1[0] + p2[0]
        if bestc is None or sc > bestc[0]:
            bestc = (sc, p1, p2)
    if bestc is None:
        return None
    return bestc[1], bestc[2]


_COMPILED: dict = {}
_SCRATCH: dict = {}


def _persistent_plan(T, H, I, N13, TOPK, S, dev, w13_sf, w2_sf):
    """Plan for the single persistent kernel (one routed slot per expert tile)."""
    tiles = _planA(H, N13, I, S)
    if tiles is None:
        return None
    (_, TPR1, TN1, SP1, NST1, NT1), (_, TPR2, TN2, SP2, NST2, NT2) = tiles
    KT1 = TPR1 * KPT * NST1
    KT2 = TPR2 * KPT * NST2
    KTP1 = KT1 + (KT1 // 32) * 8
    KTP2 = KT2 + (KT2 // 32) * 8
    num_items = max(S * NT1 * SP1, S * NT2 * SP2)
    grid_max = torch.cuda.get_device_properties(dev).multi_processor_count * GMUL
    per_cta = (num_items + grid_max - 1) // grid_max
    OOF = S * N13
    consts = dict(
        S=S,
        TOPK=TOPK,
        H=H,
        I=I,
        N13=N13,
        SWIGLU=N13 == 2 * I,
        CP13=w13_sf.shape[2],
        NSF13=w13_sf.shape[1] * w13_sf.shape[2],
        CP2=w2_sf.shape[2],
        NSF2=w2_sf.shape[1] * w2_sf.shape[2],
        KT1=KT1,
        KTP1=KTP1,
        TPR1=TPR1,
        TN1=TN1,
        NT1=NT1,
        SP1=SP1,
        NST1=NST1,
        KT2=KT2,
        KTP2=KTP2,
        TPR2=TPR2,
        TN2=TN2,
        NT2=NT2,
        SP2=SP2,
        NST2=NST2,
        OOF=OOF,
        CTROFF=0,
        SDOFF=4,
        TCOFF=4 + S,
        G=(num_items + per_cta - 1) // per_cta,
        SMEL=max(KTP1, KTP2),
        FALGN=_algn(OOF * 4, H * 4),
        MBPM=max(1, 65536 // (REGCAP * (TN1 * TPR1))),
        # Evict-first weight loads protect the reused lines, but forfeit the L2
        # reuse of slots that share an expert; keep them while sharing is rare.
        STREAM=bool(w13_sf.shape[0] >= S * 4),
    )
    return consts, OOF + T * H, 4 + S + T


def _tiled_plan(T, H, I, N13, E, TOPK, S, BM, w13_sf, w2_sf):
    """Plan for the separate routing, GEMM, activation and finalize launches."""
    nblk = min(min(E, S) + (S // BM if BM > 1 else 0), S)
    noroute = BM == 1
    TPR1, TN1, SP1, NST1, NT13 = _plan(H, N13, nblk, BM, max_split=1)
    TPR2, TN2, SP2, NST2, NT2 = _plan(I, H, nblk, BM)
    KT1 = TPR1 * KPT * NST1
    KT2 = TPR2 * KPT * NST2
    BMAX = min(S, (S + min(E, S) * (BM - 1)) // BM + 1)
    OOF = S * N13
    consts = dict(
        H=H,
        I=I,
        E=E,
        EP=((E + RTPB - 1) // RTPB) * RTPB,
        S=S,
        TOPK=TOPK,
        BM=BM,
        SWIGLU=N13 == 2 * I,
        N13=N13,
        CP13=w13_sf.shape[2],
        CP2=w2_sf.shape[2],
        NSF13=w13_sf.shape[1] * w13_sf.shape[2],
        NSF2=w2_sf.shape[1] * w2_sf.shape[2],
        KT1=KT1,
        KTP1=KT1 + (KT1 // 32) * 8,
        TPR1=TPR1,
        SP1=SP1,
        KT2=KT2,
        KTP2=KT2 + (KT2 // 32) * 8,
        TPR2=TPR2,
        SP2=SP2,
        BMAX=BMAX,
        NOF4=(T * H) // 4,
        NH4=(S * N13) // 4 if SP1 > 1 else 0,
        NEL4=(S * I) // 4,
        NT13=NT13,
        NT2=NT2,
        TN1=TN1,
        TN2=TN2,
        NOROUTE=noroute,
        ZN4=(T * H) // 4 if noroute else 0,
        OOF=OOF,
        AFUSE=S <= 512,
        # A single deep FMA chain pays off only with enough row accumulators.
        DOTC=1 if BM >= 16 else 2,
    )
    return consts, OOF + T * H + S, S + 3 * BMAX + 1


@torch.no_grad()
def run_moe_w4a16(
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w13: torch.Tensor,
    w13_sf: torch.Tensor,
    w13_gs: torch.Tensor,
    w2: torch.Tensor,
    w2_sf: torch.Tensor,
    w2_gs: torch.Tensor,
    out: torch.Tensor,
) -> torch.Tensor:
    """Routed-expert W4A16 MoE forward into ``out``.

    Args:
        hidden_states: BF16 ``[T, H]``.
        topk_ids: int32 ``[T, TOPK]`` expert ids.
        topk_weights: FP32 ``[T, TOPK]`` router weights.
        w13: uint8 ``[E, N13, H/2]``; ``N13 == 2*I`` is SwiGLU with gate rows
            first, ``N13 == I`` is ReLU2.
        w13_sf: E4M3 ``[E, pad128(N13), pad4(H/16)]`` 128x4-swizzled scales.
        w13_gs: FP32 ``[E]`` global scales.
        w2, w2_sf, w2_gs: the same for the ``[E, H, I/2]`` down projection.
        out: BF16 ``[T, H]`` output.

    Kernels are specialized and cached per token count and weight geometry;
    the first call for a new shape compiles and must not run under CUDA graph
    capture.
    """
    T, H = hidden_states.shape
    E, N13, _ = w13.shape
    I = w2.shape[2] * 2
    TOPK = topk_ids.shape[1]
    S = T * TOPK
    dev = hidden_states.device
    key = (dev, T, H, I, N13, E, TOPK, w13_sf.shape[1:], w2_sf.shape[1:])

    entry = _COMPILED.get(key)
    if entry is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM12x W4A16 MoE must be compiled for this shape before graph capture."
            )
        BM = _pick_bm(S, E)
        plan = (
            _persistent_plan(T, H, I, N13, TOPK, S, dev, w13_sf, w2_sf)
            if BM == 1
            else None
        )
        persistent = plan is not None
        if not persistent:
            plan = _tiled_plan(T, H, I, N13, E, TOPK, S, BM, w13_sf, w2_sf)
        consts, fp32_elems, int32_elems = plan
        # Zero-initialized: the persistent kernel restores counters and the FP32
        # combine slice to zero on exit, so only the first call needs a clear.
        scratch = [
            torch.zeros(fp32_elems, dtype=torch.float32, device=dev),
            torch.zeros(int32_elems, dtype=torch.int32, device=dev),
            torch.empty(max(1, S * I), dtype=torch.bfloat16, device=dev),
        ]
        entry = [persistent, consts, scratch, None]
        _COMPILED[key] = entry
    persistent, consts, scratch, compiled = entry

    w13_sf_u8 = w13_sf.view(torch.uint8)
    w2_sf_u8 = w2_sf.view(torch.uint8)
    args: tuple[torch.Tensor, ...]
    if persistent:
        args = (
            hidden_states,
            topk_ids,
            topk_weights,
            w13,
            w13_sf_u8,
            w13_gs,
            w2,
            w2_sf_u8,
            w2_gs,
            scratch[0],
            scratch[1],
            out,
        )
        launcher = moe_jitA
    else:
        args = (
            hidden_states,
            topk_ids.view(-1),
            topk_weights.view(-1),
            w13,
            w13_sf_u8,
            w13_gs,
            w2,
            w2_sf_u8,
            w2_gs,
            scratch[2],
            scratch[0],
            out.view(-1),
            scratch[1],
        )
        launcher = moe_jit
    if compiled is None:
        fake = [from_dlpack(t, assumed_align=16, enable_tvm_ffi=True) for t in args]
        stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        compiled = cute.compile(
            launcher, *fake, stream, *consts.values(), options="--enable-tvm-ffi"
        )
        entry[3] = compiled
    compiled(*args)
    return out
