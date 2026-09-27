# Copyright (c) 2026 KDA Team
# SPDX-License-Identifier: MIT
# Adapted from humanfia/kda-for-kda-release; see licenses/LICENSE.kda-for-kda.

# ruff: noqa: SIM117
# Keep the imported TIRx builder contexts and trace-time assignments intact.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright TIRx authors

"""KDA forward (B=1, K=V=128, bf16) -- split front-end family (two kernels); the front end uses no TMA stores.

Family: split-frontend.  The chunked recurrence (C=32 tokens, two 16-token sub-blocks) is cut into
its state-independent front end and its state-dependent chain, which run as two kernels:

  K1 kda_front  persistent over all SMs, one work item = (chunk c, head h), item id = c*H + h.
     Per item: gates (gd2, block-local products, chunk decay), operand tiles (X, QX, Kt, Kbar, Qt,
     A-operand rows), the two A chains on tcgen05 (Akk^T, Aqk^T), k-norms from the Akk diagonal,
     L = diag(b*kn) Akk diag(kn), the hierarchical 32x32 inverse (mma.sync), T1/T2, W1 = Kt^T T2^T
     (tcgen05), the Aqk tile and the q norms.  Results go to global scratch as bf16 tiles with the
     exact SMEM layouts the chain's MMAs consume (TMA stores): Kbar [j][k], Qt [i][k] (SW128B),
     T1 [i][j], AqkT [j][i], W1b [k][i] (SW64B), plus Dvec[128] and qn[32] (fp32).
  K2 kda_chain  one persistent CTA per head walks the chunks: TMA ring of the precomputed tiles
     (+ raw V), state ST[v][k] fp32 in TMEM, bf16 snapshot + decay split over two warpgroups,
     U'^T = V^T T1^T - STb W1b, ST += U'^T Kbar, O^T = STb Qt^T + U'^T AqkT, output epilogue.

Why: in the fused persistent-head kernel the chain's true per-chunk dependency waits on the front end
sharing its SM while other SMs idle.  Here the front end runs on every SM at full occupancy and the
chain kernel is a pure dependency loop with all operands staged ahead by TMA.

Math: identical to the fused kernel.

Timing note: the two kernels run concurrently, so the pair must be measured on a wall clock, not as a
sum of per-kernel durations.
"""

from tvm.backend.cuda.cpp.descriptors import encode_instr_descriptor_dense_uint32

import tirx_kernels.tirx_lite as txl

D = 128
C = 32
NB = 2
STAGES1 = 4
GSTAGES = 2
NSLOT = 8
TMEM1_COLS = 512
TMEM2_COLS = 512
LOG2E = 1.4426950408889634
SIG_C1 = -2.5 * LOG2E
VEC_F32 = 160
TILE_BYTES = C * D * 2
SMALL_BYTES = C * C * 2


TM_A = 0
TM_A_STRIDE = 128
TM_W1 = 256
NB_ROWS = 80
NPROD = 4
W1_LAG = 3

ROW_Y0 = 0
ROW_Z1 = 16
ROW_Z0 = 32
ROW_QZ0 = 48
ROW_QZ1 = 64

MMA = "tcgen05.mma.cta_group::1.kind::f16"
IDESC_N32 = encode_instr_descriptor_dense_uint32(
    128, 32, 16, "float32", "bfloat16", "bfloat16", False, False
)


IDESC_M64_N80 = encode_instr_descriptor_dense_uint32(
    64, 80, 16, "float32", "bfloat16", "bfloat16", False, False
)
IDESC_N128_TB = encode_instr_descriptor_dense_uint32(
    128, 128, 16, "float32", "bfloat16", "bfloat16", False, True
)
IDESC_N32_TA = encode_instr_descriptor_dense_uint32(
    128, 32, 16, "float32", "bfloat16", "bfloat16", True, False
)
IDESC_N32_TB_NEG = encode_instr_descriptor_dense_uint32(
    128, 32, 16, "float32", "bfloat16", "bfloat16", False, True, neg_b=True
)

TC_LD32 = "tcgen05.ld.sync.aligned.32x32b.x32.b32"
TC_ST32 = "tcgen05.st.sync.aligned.32x32b.x32.b32"
TC_ST16 = "tcgen05.st.sync.aligned.32x32b.x16.b32"
TC_LD16 = "tcgen05.ld.sync.aligned.32x32b.x16.b32"
TC_LD256_X4 = "tcgen05.ld.sync.aligned.16x256b.x4.b32"
STM_X4T = "stmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
WAIT_LD = "tcgen05.wait::ld.sync.aligned"
WAIT_ST = "tcgen05.wait::st.sync.aligned"
# tcgen05 accesses of different threads are ordered only through these fences: before_thread_sync after the last tcgen05.ld/st and before the arrive, after_thread_sync after the wait and before the next tcgen05 access
TC_FENCE_BEFORE = "tcgen05.fence::before_thread_sync"
TC_FENCE_AFTER = "tcgen05.fence::after_thread_sync"
FENCE_ASYNC = "fence.proxy.async.shared::cta"
TMA_G2S = "cp.async.bulk.tensor.3d.shared::cta.global.tile.mbarrier::complete_tx::bytes"
TMA_G2S_HINT = TMA_G2S + ".L2::cache_hint"
TMA_S2G = "cp.async.bulk.tensor.3d.global.shared::cta.tile.bulk_group"
BULK_G2S = "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes"
BULK_COMMIT = "cp.async.bulk.commit_group"
BULK_WAIT_READ = "cp.async.bulk.wait_group.read"
BULK_WAIT = "cp.async.bulk.wait_group"
MMA_K8 = "mma.sync.aligned.m16n8k8.row.col.f32.bf16.bf16.f32"
MMA_K16 = "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32"
LDM_X1 = "ldmatrix.sync.aligned.m8n8.x1.shared.b16"
LDM_X1T = "ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16"
LDM_X4 = "ldmatrix.sync.aligned.m8n8.x4.shared.b16"
LDM_X4T = "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16"
STM_X1 = "stmatrix.sync.aligned.m8n8.x1.shared.b16"
STM_X4 = "stmatrix.sync.aligned.m8n8.x4.shared.b16"


BAR_PREP = 1
BAR_PREP2 = 5
BAR_RDO = 3
BAR_TMEM = 4
BAR_TMEM_N1 = 192


def _f2(a, b):
    return txl.cuda.make_float2(a, b)


def _pack_bf16x2(dst, lo, hi):
    txl.ptx.cvt.rn.bf16x2.f32(dst, hi, lo)


def _bf16_lo(word):
    return txl.reinterpret("float32", word << txl.uint32(16))


def _bf16_hi(word):
    return txl.reinterpret("float32", word & txl.uint32(0xFFFF0000))


def _rng(name):
    token = txl.alloc_local([1], "uint32")
    txl.assign(token[0], txl.cuda.iket.range_start(name))
    return token


def _rng_end(token):
    if token is None:
        return
    txl.cuda.iket.range_end(token[0])


def _chain(d, a, b, idesc, accumulate, pred):
    txl.idioms.mma_chain(
        MMA, d, a=a, b=b, idesc=idesc, pred=pred, accumulate=accumulate, guard="pred"
    )


def _bid(bar_id):
    return txl.uint32(bar_id) if isinstance(bar_id, int) else bar_id


def _wg_arrive(bar, slot, bar_id, leader):
    """Whole-warpgroup handoff with one arrival: named barrier, then the leader thread arrives."""
    txl.ptx.bar.sync(_bid(bar_id), txl.uint32(128))
    with txl.If(leader), txl.Then():
        bar.arrive(slot)


def _taddr(tm, col, lane_bits):
    return txl.Cast("uint32", tm[0] + col + lane_bits)


SLOW_WAIT_NS = 200
# try_wait suspend hint (10 ms) for the front's off-critical-path roles: a waiter stays suspended until its barrier flips instead of re-polling
TRY_WAIT_HINT = 0x989680


def _slow_wait(bar, stage, parity, hint=TRY_WAIT_HINT):
    """mbarrier wait with a suspend hint, for waiters off the critical path (idle warps stop eating the busy warps' issue slots)."""
    ready = txl.local_scalar("uint32", init=txl.uint32(0))
    addr = bar.ptr_to([stage])
    with txl.While(ready == txl.uint32(0)):
        txl.ptx.mbarrier.try_wait.parity.acquire.cta.shared__cta.b64(
            ready, addr, txl.Cast("uint32", parity), txl.uint32(hint)
        )


PDONE_SLOTS = 16
SIG_BATCH = 4


def _signal_item(bar, li, leader):
    """A producer finished its global stores of item li: all of the warpgroup's / warp's stores precede the caller's
    bar.sync / warp_sync, the leader arrives on the item's completion barrier (the sig warp publishes it)."""
    with txl.If(leader), txl.Then():
        bar.arrive(li & (PDONE_SLOTS - 1))


def _tmem_preamble(s_tmem_addr, count):
    txl.ptx.bar.sync(txl.uint32(BAR_TMEM), txl.uint32(count))
    tm = txl.alloc_local([1], "int32")
    txl.ptx.ld.volatile.shared.s32(tm[0], txl.address_of(s_tmem_addr[0]))
    return tm


def make_front(H: int, arch: str):
    """Front-end kernel for a fixed head count: item it = c*H + h.  All per-item outputs are written
    straight to global memory with coalesced generic stores (row-major tiles, read back by K2 with TMA).

    Raw (un-normalised) key tiles; both norms come out of the A chain (diagonals of
    X Z^T and QX QZ^T); the A chain is M=64 x N=80 with A = [X ; QX] and B = [Y0 ; Z1 ; Z0 ; QZ0 ; QZ1]
    so that every L row / Aqk row lands in ONE TMEM lane (Layout F); the norms are folded into
    T1' = diag(kn) T diag(b) and T2' = T1' diag(kn); one warp per item inverts (four solvers)."""

    # H64 benefits from amortizing publication over eight completed items.
    # Keep four for H96, where more frequent publication measures faster.
    signal_batch = 8 if H == 64 else SIG_BATCH

    @txl.kernel(warps=20, arch=arch, min_blocks_per_sm=1, grid="num_ctas")
    def kda_front(
        q: txl.gptr[txl.bf16],
        k: txl.gptr[txl.bf16],
        g: txl.gptr[txl.bf16],
        beta: txl.gptr[txl.bf16],
        a_log: txl.gptr[txl.f32],
        dt_bias: txl.gptr[txl.f32],
        vec: txl.gptr[txl.f32],
        kbar_g: txl.gptr[txl.bf16],
        qt_g: txl.gptr[txl.bf16],
        t1_g: txl.gptr[txl.bf16],
        aqk_g: txl.gptr[txl.bf16],
        w1_g: txl.gptr[txl.bf16],
        flags: txl.gptr[txl.i32],
        q_map: txl.TensorMap,
        k_map: txl.TensorMap,
        g_map: txl.TensorMap,
        item_base: txl.i32,
        num_items: txl.i32,
        num_ctas: txl.i32,
        items_per_cta: txl.i32,
        do_signal: txl.i32,
    ):
        for buf in (q, k, g):
            txl.keep_alive(buf.ptr_to([0]))

        cta = txl.cta_id()
        lane = txl.lane_id()
        tid = txl.thread_id()
        last_it = cta + (items_per_cta - 1) * num_ctas
        n_cta = txl.Select(last_it < num_items, items_per_cta, items_per_cta - 1)

        sp = txl.specialize()

        prep = sp.role("prep", warps=list(range(8)), regs=128)
        rdo = sp.role("rdo", warps=[8, 9, 10, 11], regs=96)
        solve = sp.role("solve", warps=[12, 13, 14, 15], regs=88)
        mma2 = sp.role("mma2", warps=[16], regs=40)
        mma3 = sp.role("mma3", warps=[17], regs=40)
        tma = sp.role("tma", warps=[18], regs=40)
        aux = sp.role("aux", warps=[19], regs=40)

        smem = txl.smem_pool()
        s_tmem_addr = smem.alloc((1,), txl.i32, align=4)
        p_raw = txl.Pipeline(smem, STAGES1, full="tma", empty="mbar", init_empty=1)
        p_g = txl.Pipeline(smem, GSTAGES, full="tma", empty="mbar", init_empty=1)
        m_tiles = txl.MBarrier(smem, 2)
        m_akk = txl.TCGen05Bar(smem, 2)
        m_afree = txl.MBarrier(smem, 2)
        m_w1 = txl.TCGen05Bar(smem, 4)
        m_w1free = txl.MBarrier(smem, 4)
        m_beta = txl.MBarrier(smem, NSLOT)
        m_gates0 = txl.MBarrier(smem, 1)
        m_bdone = txl.MBarrier(smem, NSLOT)
        m_lready = txl.MBarrier(smem, 4)
        m_lfree = txl.MBarrier(smem, 4)
        m_tp = txl.MBarrier(smem, 4)
        m_pdone = txl.MBarrier(smem, PDONE_SLOTS)
        m_sfree = txl.MBarrier(smem, PDONE_SLOTS)
        for bar, cnt in (
            (m_tiles, 1),
            (m_akk, 1),
            (m_afree, 1),
            (m_w1, 1),
            (m_w1free, 1),
            (m_beta, 32),
            (m_gates0, 1),
            (m_bdone, 1),
            (m_lready, 1),
            (m_lfree, 1),
            (m_tp, 1),
            (m_pdone, NPROD),
            (m_sfree, 1),
        ):
            bar.init(cnt)

        s_q = smem.alloc((STAGES1, C, D), txl.bf16, swizzle=txl.SW128B)
        s_k = smem.alloc((STAGES1, C, D), txl.bf16, swizzle=txl.SW128B)
        s_g = smem.alloc((GSTAGES, C, D), txl.bf16, swizzle=txl.SW128B)
        s_a = smem.alloc((2, 2 * C, D), txl.bf16, swizzle=txl.SW128B)
        s_b = smem.alloc((2, NB_ROWS + 16, D), txl.bf16, swizzle=txl.SW128B)
        s_kt = smem.alloc((4, C, D), txl.bf16, swizzle=txl.SW128B)
        s_l = smem.alloc((4, C, C), txl.bf16, swizzle=txl.SW64B)
        s_t2 = smem.alloc((4, C, C), txl.bf16, swizzle=txl.SW64B)
        s_bt = smem.alloc((NSLOT, C), txl.f32, align=16)
        s_ha = smem.alloc((NSLOT, 4), txl.f32, align=16)
        s_hadt = smem.alloc((NSLOT, D), txl.f32, align=16)
        s_kn = smem.alloc((4, C), txl.f32, align=16)
        s_bblk = smem.alloc((4, NB, D), txl.f32, align=16)

        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elect():
            return txl.cuda.elect_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def elect_local():
            e = txl.local_scalar("uint32")
            txl.assign(e, txl.cuda.elect_sync())
            return e

        def item_of(li):
            return item_base + cta + li * num_ctas

        def it64(li):
            return txl.Cast("int64", item_of(li))

        def unpack(dst, i, word):
            txl.assign(dst[2 * i], _bf16_lo(word))
            txl.assign(dst[2 * i + 1], _bf16_hi(word))

        with prep:
            wid_all = txl.warp_id_in_role()
            parity = wid_all >> 2
            bar_id = txl.uint32(BAR_PREP) + txl.Cast("uint32", parity) * txl.uint32(
                BAR_PREP2 - BAR_PREP
            )
            wid = wid_all & 3
            blk = wid >> 1
            khalf = wid & 1
            grp = lane >> 4
            kq = lane & 15
            k0 = khalf * 64 + kq * 4
            k064 = txl.Cast("int64", k0)
            rowbase = blk * 16 + grp * 8
            tidp = txl.tid_in_role() & 127
            is_grp1 = grp == 1

            gd = txl.alloc_local([32], "float32")
            ff = txl.alloc_local([32], "float32")
            ew = txl.alloc_local([16], "uint32")
            fw = txl.alloc_local([16], "uint32")
            kws = txl.alloc_local([16], "uint32")
            qws = txl.alloc_local([16], "uint32")
            n_mine = (n_cta + 1 - parity) >> 1
            with txl.serial(n_mine) as ci:
                li = ci * 2 + parity
                it = item_of(li)
                itb = txl.Cast("int64", it)
                stage = txl.local_scalar("int32", init=li & (STAGES1 - 1))
                rphase = (li >> 2) & 1
                gstage = txl.local_scalar("int32", init=li & (GSTAGES - 1))
                slot8 = li & (NSLOT - 1)

                with txl.If((parity == 1) & (ci == 0)), txl.Then():
                    m_gates0.wait(0, 0)

                tk = _rng("prep-wait-consts")
                m_beta.wait(slot8, (li >> 3) & 1)
                _rng_end(tk)
                ha = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(ha, txl.address_of(s_ha[slot8, 0]))
                hadt = txl.alloc_local([4], "float32")
                txl.ptx["ld.shared.v4.f32"](
                    hadt[0],
                    hadt[1],
                    hadt[2],
                    hadt[3],
                    txl.address_of(s_hadt[slot8, k0]),
                )

                tk = _rng("prep-wait-raw")
                p_g.full.wait(gstage, (li >> 1) & 1)
                _rng_end(tk)
                gg = s_g[gstage]
                tk = _rng("prep-gates")
                gw = txl.alloc_local([2], "uint32")
                th = txl.alloc_local([4], "float32")
                xarg = txl.alloc_local([2], "uint64")
                gd2 = txl.alloc_local([2], "uint64")
                for t in range(8):
                    txl.ptx["ld.shared.v2.b32"](
                        gw[0], gw[1], gg.ptr_to(rowbase + t, k0)
                    )
                    for p in range(2):
                        txl.ptx.fma.rn.f32x2(
                            xarg[p],
                            _f2(_bf16_lo(gw[p]), _bf16_hi(gw[p])),
                            _f2(ha, ha),
                            _f2(hadt[2 * p], hadt[2 * p + 1]),
                        )
                        txl.ptx.tanh.approx.f32(th[2 * p], txl.cuda.float2_x(xarg[p]))
                        txl.ptx.tanh.approx.f32(
                            th[2 * p + 1], txl.cuda.float2_y(xarg[p])
                        )
                    for p in range(2):
                        txl.ptx.fma.rn.f32x2(
                            gd2[p],
                            _f2(th[2 * p], th[2 * p + 1]),
                            _f2(txl.float32(SIG_C1), txl.float32(SIG_C1)),
                            _f2(txl.float32(SIG_C1), txl.float32(SIG_C1)),
                        )
                        txl.ptx.ex2.approx.ftz.f32(
                            gd[4 * t + 2 * p], txl.cuda.float2_x(gd2[p])
                        )
                        txl.ptx.ex2.approx.ftz.f32(
                            gd[4 * t + 2 * p + 1], txl.cuda.float2_y(gd2[p])
                        )

                acc = txl.alloc_local([2], "uint64")
                for p in range(2):
                    txl.assign(acc[p], _f2(txl.float32(1.0), txl.float32(1.0)))
                for t in range(7, -1, -1):
                    for p in range(2):
                        txl.assign(ff[4 * t + 2 * p], txl.cuda.float2_x(acc[p]))
                        txl.assign(ff[4 * t + 2 * p + 1], txl.cuda.float2_y(acc[p]))
                        if t > 0:
                            txl.ptx.mul.rn.f32x2(
                                acc[p],
                                acc[p],
                                _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]),
                            )
                for t in range(1, 8):
                    for p in range(2):
                        txl.ptx.mul.rn.f32x2(
                            acc[p],
                            _f2(gd[4 * (t - 1) + 2 * p], gd[4 * (t - 1) + 2 * p + 1]),
                            _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]),
                        )
                        txl.assign(gd[4 * t + 2 * p], txl.cuda.float2_x(acc[p]))
                        txl.assign(gd[4 * t + 2 * p + 1], txl.cuda.float2_y(acc[p]))

                oth = txl.alloc_local([4], "float32")
                for j in range(4):
                    tot_w = txl.local_scalar("uint32")
                    txl.ptx.shfl_sync.bfly.b32(
                        tot_w,
                        txl.reinterpret("uint32", gd[28 + j]),
                        txl.uint32(16),
                        txl.uint32(31),
                        txl.uint32(0xFFFFFFFF),
                    )
                    txl.assign(oth[j], txl.reinterpret("float32", tot_w))
                efac = txl.alloc_local([4], "float32")
                ffac = txl.alloc_local([4], "float32")
                bb = txl.alloc_local([4], "float32")
                for j in range(4):
                    txl.assign(efac[j], txl.Select(is_grp1, oth[j], txl.float32(1.0)))
                    txl.assign(ffac[j], txl.Select(is_grp1, txl.float32(1.0), oth[j]))
                    txl.assign(bb[j], gd[28 + j] * oth[j])
                prod = txl.local_scalar("uint64")
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.f32x2(
                            prod,
                            _f2(gd[4 * t + 2 * p], gd[4 * t + 2 * p + 1]),
                            _f2(efac[2 * p], efac[2 * p + 1]),
                        )
                        _pack_bf16x2(
                            ew[2 * t + p],
                            txl.cuda.float2_x(prod),
                            txl.cuda.float2_y(prod),
                        )
                        txl.ptx.mul.rn.f32x2(
                            prod,
                            _f2(ff[4 * t + 2 * p], ff[4 * t + 2 * p + 1]),
                            _f2(ffac[2 * p], ffac[2 * p + 1]),
                        )
                        _pack_bf16x2(
                            fw[2 * t + p],
                            txl.cuda.float2_x(prod),
                            txl.cuda.float2_y(prod),
                        )
                bslot = parity * 2 + (ci & 1)
                with txl.If(grp == 0), txl.Then():
                    txl.ptx["st.shared.v4.f32"](
                        txl.address_of(s_bblk[bslot, blk, k0]),
                        bb[0],
                        bb[1],
                        bb[2],
                        bb[3],
                    )
                rbb = txl.alloc_local([4], "float32")
                for j in range(4):
                    txl.ptx.rcp.approx.ftz.f32(rbb[j], bb[j])
                txl.ptx.bar.sync(bar_id, txl.uint32(128))
                # release the g stage only after the group barrier, once every thread's g loads are consumed; an earlier arrive would let the TMA producer rewrite the stage under in-flight loads
                with txl.If(tidp == 0), txl.Then():
                    p_g.empty.arrive(gstage)
                with txl.If((parity == 0) & (ci == 0) & (tidp == 0)), txl.Then():
                    m_gates0.arrive(0)
                ob = txl.alloc_local([4], "float32")
                txl.ptx["ld.shared.v4.f32"](
                    ob[0],
                    ob[1],
                    ob[2],
                    ob[3],
                    txl.address_of(s_bblk[bslot, 1 - blk, k0]),
                )
                with txl.If((blk == 1) & (grp == 0)), txl.Then():
                    txl.ptx["st.global.v4.f32"](
                        vec.ptr_to([itb * txl.int64(VEC_F32) + k064]),
                        bb[0] * ob[0],
                        bb[1] * ob[1],
                        bb[2] * ob[2],
                        bb[3] * ob[3],
                    )
                _rng_end(tk)
                pf = txl.alloc_local([4], "float32")
                rf = txl.alloc_local([4], "float32")
                with txl.If(blk == 1):
                    with txl.Then():
                        for j in range(4):
                            txl.assign(pf[j], ob[j])
                            txl.assign(rf[j], txl.float32(1.0))
                    with txl.Else():
                        for j in range(4):
                            txl.assign(pf[j], txl.float32(1.0))
                            txl.assign(rf[j], ob[j])
                pfw = txl.alloc_local([2], "uint32")
                rfw = txl.alloc_local([2], "uint32")
                rbw = txl.alloc_local([2], "uint32")
                for p in range(2):
                    _pack_bf16x2(pfw[p], pf[2 * p], pf[2 * p + 1])
                    _pack_bf16x2(rfw[p], rf[2 * p], rf[2 * p + 1])
                    _pack_bf16x2(rbw[p], rbb[2 * p], rbb[2 * p + 1])

                tk = _rng("prep-wait-rawqk")
                p_raw.full.wait(stage, rphase)
                _rng_end(tk)
                tk = _rng("prep-wait-akk")
                with txl.If(li >= 2), txl.Then():
                    m_akk.wait(li & 1, ((li >> 1) - 1) & 1)
                _rng_end(tk)
                # Observe both MMA readers before the existing proxy fence:
                # it orders reuse of s_a/s_b and s_kt in the generic proxy.
                tk = _rng("prep-wait-w1")
                with txl.If(li >= 4), txl.Then():
                    m_w1.wait(li & 3, ((li - 4) >> 2) & 1)
                _rng_end(tk)
                tk = _rng("prep-tiles")
                txl.ptx[FENCE_ASYNC]()
                kq_t = s_k[stage]
                qq_t = s_q[stage]
                sa = s_a[li & 1]
                sb = s_b[li & 1]
                yrow0 = ROW_Y0 + NB_ROWS * blk + grp * 8
                zrow0 = ROW_Z0 - (ROW_Z0 - ROW_Z1) * blk + grp * 8
                qzrow0 = ROW_QZ0 + (ROW_QZ1 - ROW_QZ0) * blk + grp * 8
                gbase = (itb * txl.int64(C) + txl.Cast("int64", rowbase)) * txl.int64(
                    D
                ) + k064
                for t in range(8):
                    txl.ptx["ld.shared.v2.b32"](
                        kws[2 * t], kws[2 * t + 1], kq_t.ptr_to(rowbase + t, k0)
                    )
                    txl.ptx["ld.shared.v2.b32"](
                        qws[2 * t], qws[2 * t + 1], qq_t.ptr_to(rowbase + t, k0)
                    )
                sc = txl.alloc_local([2], "uint32")
                oa = txl.alloc_local([2], "uint32")
                ob_ = txl.alloc_local([2], "uint32")
                for t in range(8):
                    row = rowbase + t

                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], ew[2 * t + p])
                        txl.ptx.mul.rn.bf16x2(ob_[p], qws[2 * t + p], ew[2 * t + p])
                    txl.ptx["st.shared.v2.b32"](sa.ptr_to(row, k0), oa[0], oa[1])
                    txl.ptx["st.shared.v2.b32"](sa.ptr_to(C + row, k0), ob_[0], ob_[1])
                    # the Qt / Kbar global stores of this row are issued after the item's last fence.proxy.async (below), so no fence waits on in-flight global stores
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], fw[2 * t + p])
                    txl.ptx["st.shared.v2.b32"](sb.ptr_to(yrow0 + t, k0), oa[0], oa[1])

                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], fw[2 * t + p], rbw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                        txl.ptx.mul.rn.bf16x2(ob_[p], qws[2 * t + p], sc[p])
                    txl.ptx["st.shared.v2.b32"](sb.ptr_to(zrow0 + t, k0), oa[0], oa[1])
                    txl.ptx["st.shared.v2.b32"](
                        sb.ptr_to(qzrow0 + t, k0), ob_[0], ob_[1]
                    )
                _rng_end(tk)

                tk = _rng("prep-kt")
                ktq = s_kt[li & 3]
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], ew[2 * t + p], pfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                    txl.ptx["st.shared.v2.b32"](
                        ktq.ptr_to(rowbase + t, k0), oa[0], oa[1]
                    )
                txl.ptx[FENCE_ASYNC]()
                # Qt = q ew pf and Kbar = k fw rf rows -> global after the item's last async-proxy fence; the bar.sync below orders them before the leader's m_pdone arrive (published by the aux warp's gpu-scope fence with the item's flag)
                for t in range(8):
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], ew[2 * t + p], pfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], qws[2 * t + p], sc[p])
                    txl.ptx["st.global.v2.b32"](
                        qt_g.ptr_to([gbase + t * D]), oa[0], oa[1]
                    )
                    for p in range(2):
                        txl.ptx.mul.rn.bf16x2(sc[p], fw[2 * t + p], rfw[p])
                        txl.ptx.mul.rn.bf16x2(oa[p], kws[2 * t + p], sc[p])
                    txl.ptx["st.global.v2.b32"](
                        kbar_g.ptr_to([gbase + t * D]), oa[0], oa[1]
                    )
                txl.ptx.bar.sync(bar_id, txl.uint32(128))
                with txl.If(tidp == 0), txl.Then():
                    with txl.If(li >= PDONE_SLOTS - 2), txl.Then():
                        m_sfree.wait(
                            (li + 2) & (PDONE_SLOTS - 1),
                            ((li - (PDONE_SLOTS - 2)) >> 4) & 1,
                        )
                    m_tiles.arrive(li & 1)
                    p_raw.empty.arrive(stage)
                    m_pdone.arrive(li & (PDONE_SLOTS - 1))
                _rng_end(tk)

        with rdo:
            tid1 = txl.tid_in_role()
            lanebits = (tid1 << 16) & 0x600000
            wid1 = txl.warp_id_in_role()
            with txl.If(wid1 == 0), txl.Then():
                txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    txl.address_of(s_tmem_addr[0]), txl.uint32(TMEM1_COLS)
                )
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)

            blk1 = wid1 & 1
            is_x = wid1 < 2
            i_row = blk1 * 16 + lane
            active = lane < 16
            col_a = txl.int32(32) - blk1 * 32
            av = txl.alloc_local([32], "float32")
            zv = txl.alloc_local([16], "float32")
            knv = txl.alloc_local([32], "float32")
            pw = txl.alloc_local([16], "uint32")

            def select_tree(src, base, n):
                """src[base + lane] for lane < n (power-of-two balanced register gather)."""
                # Four predicate bits give a logarithmic-depth register gather.
                values = [src[base + c] for c in range(n)]
                bit = 1
                while len(values) > 1:
                    take_hi = (lane & bit) != 0
                    values = [
                        txl.local_scalar(
                            "float32",
                            init=txl.Select(take_hi, values[c + 1], values[c]),
                        )
                        for c in range(0, len(values), 2)
                    ]
                    bit *= 2
                return values[0]

            def apass(li):
                """Read the A chain of item li: L rows (+ k norms) -> s_l / s_kn slot li&3 ; Aqk rows (+ q norms) -> global."""
                aslot = li & 1
                lslot = li & 3
                tk = _rng("rdo-wait-akk")
                _slow_wait(m_akk, aslot, (li >> 1) & 1)
                with txl.If(li >= 4), txl.Then():
                    _slow_wait(m_lfree, lslot, ((li >> 2) - 1) & 1)
                _slow_wait(m_beta, li & (NSLOT - 1), (li >> 3) & 1)
                _rng_end(tk)
                tk = _rng("rdo-apass")
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](
                    *(av[i] for i in range(32)),
                    _taddr(tm, TM_A + TM_A_STRIDE * aslot + col_a, lanebits),
                )
                with txl.If(wid1 == 3), txl.Then():
                    txl.ptx[TC_LD16](
                        *(zv[i] for i in range(16)),
                        _taddr(tm, TM_A + TM_A_STRIDE * aslot + ROW_QZ1, lanebits),
                    )
                txl.ptx[WAIT_LD]()

                dsel = txl.local_scalar("float32")
                with txl.If(wid1 == 0), txl.Then():
                    txl.assign(dsel, select_tree(av, 0, 16))
                with txl.If((wid1 == 1) | (wid1 == 2)), txl.Then():
                    txl.assign(dsel, select_tree(av, 16, 16))
                with txl.If(wid1 == 3), txl.Then():
                    txl.assign(dsel, select_tree(zv, 0, 16))
                nrm = txl.local_scalar("float32")
                txl.ptx.rsqrt.approx.ftz.f32(nrm, dsel + txl.float32(1e-6))
                with txl.If(active), txl.Then():
                    with txl.If(is_x):
                        with txl.Then():
                            txl.ptx.st.shared.f32(
                                txl.address_of(s_kn[lslot, i_row]), nrm
                            )
                        with txl.Else():
                            txl.ptx.st.global_.f32(
                                vec.ptr_to(
                                    [
                                        it64(li) * txl.int64(VEC_F32)
                                        + txl.int64(D)
                                        + txl.Cast("int64", i_row)
                                    ]
                                ),
                                nrm,
                            )
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                rbase = (
                    it64(li) * txl.int64(C) + txl.Cast("int64", i_row)
                ) * txl.int64(C)
                with txl.If(active), txl.Then():
                    with txl.If(is_x):
                        with txl.Then():
                            bi = txl.local_scalar("float32")
                            txl.ptx.ld.shared.f32(
                                bi, txl.address_of(s_bt[li & (NSLOT - 1), i_row])
                            )
                            coef = bi * nrm
                            for p_ in range(8):
                                txl.ptx["ld.shared.v4.f32"](
                                    knv[4 * p_],
                                    knv[4 * p_ + 1],
                                    knv[4 * p_ + 2],
                                    knv[4 * p_ + 3],
                                    txl.address_of(s_kn[lslot, 4 * p_]),
                                )
                            prod = txl.local_scalar("uint64")
                            for p_ in range(16):
                                j0 = 2 * p_
                                txl.ptx.mul.rn.f32x2(
                                    prod, _f2(knv[j0], knv[j0 + 1]), _f2(coef, coef)
                                )
                                txl.ptx.mul.rn.f32x2(
                                    prod,
                                    _f2(
                                        txl.cuda.float2_x(prod), txl.cuda.float2_y(prod)
                                    ),
                                    _f2(av[j0], av[j0 + 1]),
                                )
                                v0 = txl.Select(
                                    txl.int32(j0) < i_row,
                                    txl.cuda.float2_x(prod),
                                    txl.float32(0.0),
                                )
                                v1 = txl.Select(
                                    txl.int32(j0 + 1) < i_row,
                                    txl.cuda.float2_y(prod),
                                    txl.float32(0.0),
                                )
                                _pack_bf16x2(pw[p_], v0, v1)
                            sl = s_l[lslot]
                            for q_ in range(4):
                                txl.ptx["st.shared.v4.b32"](
                                    sl.ptr_to(i_row, 8 * q_),
                                    pw[4 * q_],
                                    pw[4 * q_ + 1],
                                    pw[4 * q_ + 2],
                                    pw[4 * q_ + 3],
                                )
                        with txl.Else():
                            for p_ in range(16):
                                j0 = 2 * p_
                                v0 = txl.Select(
                                    txl.int32(j0) <= i_row, av[j0], txl.float32(0.0)
                                )
                                v1 = txl.Select(
                                    txl.int32(j0 + 1) <= i_row,
                                    av[j0 + 1],
                                    txl.float32(0.0),
                                )
                                _pack_bf16x2(pw[p_], v0, v1)
                            for q_ in range(2):
                                txl.ptx.st.global_.L2__evict_last.v8.b32(
                                    aqk_g.ptr_to([rbase + q_ * 16]),
                                    *(pw[8 * q_ + j] for j in range(8)),
                                )
                txl.ptx[TC_FENCE_BEFORE]()
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                with txl.If(tid1 == 0), txl.Then():
                    m_afree.arrive(aslot)
                    m_lready.arrive(lslot)
                _signal_item(m_pdone, li, tid1 == 0)
                _rng_end(tk)

            def w1_readout(lw):
                """W1' fp32 [k][i] of item lw (lane k) -> bf16 row k of the global tile w1_g[it][k][i]."""
                wslot = lw & 3
                tk = _rng("rdo-wait-w1")
                _slow_wait(m_w1, wslot, (lw >> 2) & 1)
                _rng_end(tk)
                tk = _rng("rdo-w1")
                txl.ptx[TC_FENCE_AFTER]()
                txl.ptx[TC_LD32](
                    *(av[i] for i in range(32)),
                    _taddr(tm, TM_W1 + 32 * wslot, lanebits),
                )
                txl.ptx[WAIT_LD]()
                txl.ptx[TC_FENCE_BEFORE]()
                for p_ in range(16):
                    _pack_bf16x2(pw[p_], av[2 * p_], av[2 * p_ + 1])
                wbase = (it64(lw) * txl.int64(D) + txl.Cast("int64", tid1)) * txl.int64(
                    C
                )
                for p_ in range(2):
                    txl.ptx.st.global_.L2__evict_last.v8.b32(
                        w1_g.ptr_to([wbase + p_ * 16]),
                        *(pw[8 * p_ + j] for j in range(8)),
                    )
                txl.ptx.bar.sync(txl.uint32(BAR_RDO), txl.uint32(128))
                with txl.If(tid1 == 0), txl.Then():
                    m_w1free.arrive(wslot)
                _signal_item(m_pdone, lw, tid1 == 0)
                _rng_end(tk)

            with txl.serial(n_cta) as li:
                with txl.If(li >= W1_LAG), txl.Then():
                    w1_readout(li - W1_LAG)
                apass(li)
            for kk in range(W1_LAG, 0, -1):
                with txl.If(n_cta >= kk), txl.Then():
                    w1_readout(n_cta - kk)
            txl.cuda.warpgroup_sync(BAR_RDO)
            with txl.If(wid1 == 0), txl.Then():
                txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    txl.Cast("uint32", tm[0]), txl.uint32(TMEM1_COLS)
                )

        with solve:
            sidx = txl.warp_id_in_role()

            def ldm_x4(insn, dst, av_, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                txl.ptx[insn](dst[0], dst[1], dst[2], dst[3], av_.ptr_to(row, col))

            def stm_x4(src, av_, base_row, base_col):
                lm = lane >> 3
                row = base_row + (lane & 7) + (lm & 1) * 8
                col = base_col + (lm >> 1) * 8
                txl.ptx[STM_X4](av_.ptr_to(row, col), src[0], src[1], src[2], src[3])

            def neg_pack(dst_word, a, b):
                neg = txl.local_scalar("uint64")
                txl.ptx.sub.rn.f32x2(
                    neg, _f2(txl.float32(0.0), txl.float32(0.0)), _f2(a, b)
                )
                _pack_bf16x2(dst_word, txl.cuda.float2_x(neg), txl.cuda.float2_y(neg))

            def mma_k8_zero(acc, a, b):
                txl.ptx[MMA_K8](
                    acc[0],
                    acc[1],
                    acc[2],
                    acc[3],
                    a[0],
                    a[1],
                    b[0],
                    txl.float32(0.0),
                    txl.float32(0.0),
                    txl.float32(0.0),
                    txl.float32(0.0),
                )

            def mma_k16(acc, a, b, acc_off, b_off, accumulate):
                cc = (
                    [acc[acc_off + i] for i in range(4)]
                    if accumulate
                    else [txl.float32(0.0)] * 4
                )
                txl.ptx[MMA_K16](
                    *(acc[acc_off + i] for i in range(4)),
                    a[0],
                    a[1],
                    a[2],
                    a[3],
                    b[b_off],
                    b[b_off + 1],
                    *cc,
                )

            def invert_diag_8x8(av_, block8):
                r = block8 + (lane & 7)
                words = txl.alloc_local([4], "uint32")
                txl.ptx["ld.shared.v4.b32"](
                    words[0], words[1], words[2], words[3], av_.ptr_to(r, block8)
                )
                row = [txl.local_scalar("float32") for _ in range(8)]
                for p in range(4):
                    unpack(row, p, words[p])
                for i in range(8):
                    with txl.If((lane & 7) == i), txl.Then():
                        txl.assign(row[i], txl.float32(1.0))
                rs = txl.local_scalar("float32")
                pv = txl.local_scalar("uint32")
                for src in range(7):
                    txl.ptx.neg.f32(rs, row[src])
                    for i in range(7):
                        if i < src:
                            txl.ptx.shfl_sync.idx.b32(
                                pv,
                                txl.reinterpret("uint32", row[i]),
                                txl.uint32(src),
                                txl.uint32(0x181F),
                                txl.uint32(0xFFFFFFFF),
                            )
                            with txl.If((lane & 7) > src), txl.Then():
                                txl.assign(
                                    row[i], row[i] + rs * txl.reinterpret("float32", pv)
                                )
                    with txl.If((lane & 7) > src), txl.Then():
                        txl.assign(row[src], rs)
                for p in range(4):
                    _pack_bf16x2(words[p], row[2 * p], row[2 * p + 1])
                txl.ptx["st.shared.v4.b32"](
                    av_.ptr_to(r, block8), words[0], words[1], words[2], words[3]
                )

            def inverse_8_to_16(av_, b16):
                a = txl.alloc_local([2], "uint32")
                b = txl.alloc_local([1], "uint32")
                acc = txl.alloc_local([4], "float32")
                dm = txl.local_scalar("uint32")
                cm = txl.local_scalar("uint32")
                txl.ptx[LDM_X1](dm, av_.ptr_to(b16 + 8 + (lane & 7), b16 + 8))
                txl.ptx[LDM_X1T](cm, av_.ptr_to(b16 + 8 + (lane & 7), b16))
                txl.assign(a[0], dm)
                txl.assign(a[1], dm)
                txl.assign(b[0], cm)
                mma_k8_zero(acc, a, b)
                neg_pack(a[0], acc[0], acc[1])
                neg_pack(a[1], acc[2], acc[3])
                txl.ptx[LDM_X1T](b[0], av_.ptr_to(b16 + (lane & 7), b16))
                mma_k8_zero(acc, a, b)
                _pack_bf16x2(dm, acc[0], acc[1])
                txl.ptx[STM_X1](av_.ptr_to(b16 + 8 + (lane & 7), b16), dm)

            def inverse_16_to_32(av_, b32):
                a = txl.alloc_local([4], "uint32")
                b = txl.alloc_local([4], "uint32")
                acc = txl.alloc_local([8], "float32")
                outw = txl.alloc_local([4], "uint32")
                ldm_x4(LDM_X4, a, av_, b32 + 16, b32 + 16)
                ldm_x4(LDM_X4T, b, av_, b32 + 16, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    neg_pack(a[p], acc[2 * p], acc[2 * p + 1])
                ldm_x4(LDM_X4T, b, av_, b32, b32)
                mma_k16(acc, a, b, 0, 0, False)
                mma_k16(acc, a, b, 4, 2, False)
                for p in range(4):
                    _pack_bf16x2(outw[p], acc[2 * p], acc[2 * p + 1])
                stm_x4(outw, av_, b32 + 16, b32)

            sl = s_l[sidx]
            st2 = s_t2[sidx]
            tw = txl.alloc_local([8], "uint32")
            tv = txl.alloc_local([16], "float32")
            bq = txl.alloc_local([16], "float32")
            n_mine = (n_cta - sidx + 3) >> 2
            with txl.serial(n_mine) as ci:
                li = ci * 4 + sidx
                slot8 = li & (NSLOT - 1)
                tk = _rng("solve-wait-l")
                _slow_wait(m_lready, sidx, ci & 1)
                _slow_wait(m_beta, slot8, (li >> 3) & 1)
                _rng_end(tk)
                tk = _rng("solve-inv")
                invert_diag_8x8(sl, (lane >> 3) * 8)
                txl.cuda.warp_sync()
                inverse_8_to_16(sl, 0)
                inverse_8_to_16(sl, 16)
                txl.cuda.warp_sync()
                inverse_16_to_32(sl, 0)
                txl.cuda.warp_sync()
                _rng_end(tk)
                tk = _rng("solve-t")

                kni = txl.local_scalar("float32")
                txl.ptx.ld.shared.f32(kni, txl.address_of(s_kn[sidx, lane]))
                tbase = (it64(li) * txl.int64(C) + txl.Cast("int64", lane)) * txl.int64(
                    C
                )
                with txl.If(li >= 4), txl.Then():
                    _slow_wait(m_w1, sidx, ((li - 4) >> 2) & 1)
                txl.ptx[FENCE_ASYNC]()
                for half in range(2):
                    c0 = 16 * half
                    txl.ptx["ld.shared.v4.b32"](
                        tw[0], tw[1], tw[2], tw[3], sl.ptr_to(lane, c0)
                    )
                    for p in range(4):
                        unpack(tv, p, tw[p])
                    txl.ptx["ld.shared.v4.b32"](
                        tw[0], tw[1], tw[2], tw[3], sl.ptr_to(lane, c0 + 8)
                    )
                    for p in range(4):
                        unpack(tv, 4 + p, tw[p])
                    for p in range(4):
                        txl.ptx["ld.shared.v4.f32"](
                            bq[4 * p],
                            bq[4 * p + 1],
                            bq[4 * p + 2],
                            bq[4 * p + 3],
                            txl.address_of(s_bt[slot8, c0 + 4 * p]),
                        )
                    prod = txl.local_scalar("uint64")
                    for p in range(8):
                        txl.ptx.mul.rn.f32x2(
                            prod,
                            _f2(tv[2 * p] * kni, tv[2 * p + 1] * kni),
                            _f2(bq[2 * p], bq[2 * p + 1]),
                        )
                        txl.assign(tv[2 * p], txl.cuda.float2_x(prod))
                        txl.assign(tv[2 * p + 1], txl.cuda.float2_y(prod))
                    for p in range(8):
                        _pack_bf16x2(tw[p], tv[2 * p], tv[2 * p + 1])
                    txl.ptx.st.global_.L2__evict_last.v8.b32(
                        t1_g.ptr_to([tbase + c0]), *(tw[p] for p in range(8))
                    )
                    for p in range(4):
                        txl.ptx["ld.shared.v4.f32"](
                            bq[4 * p],
                            bq[4 * p + 1],
                            bq[4 * p + 2],
                            bq[4 * p + 3],
                            txl.address_of(s_kn[sidx, c0 + 4 * p]),
                        )
                    for p in range(8):
                        txl.ptx.mul.rn.f32x2(
                            prod,
                            _f2(tv[2 * p], tv[2 * p + 1]),
                            _f2(bq[2 * p], bq[2 * p + 1]),
                        )
                        _pack_bf16x2(
                            tw[p % 4], txl.cuda.float2_x(prod), txl.cuda.float2_y(prod)
                        )
                        if p % 4 == 3:
                            txl.ptx["st.shared.v4.b32"](
                                st2.ptr_to(lane, c0 + 8 * (p // 4)),
                                tw[0],
                                tw[1],
                                tw[2],
                                tw[3],
                            )
                txl.ptx[FENCE_ASYNC]()
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    m_tp.arrive(sidx)
                    m_lfree.arrive(sidx)
                    m_bdone.arrive(slot8)
                    m_pdone.arrive(li & (PDONE_SLOTS - 1))
                _rng_end(tk)

        with mma2:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)
            e = elect_local()
            with txl.serial(n_cta) as li:
                aslot = li & 1
                tk = _rng("mma2-wait-tiles")
                _slow_wait(m_tiles, aslot, (li >> 1) & 1, 200)
                with txl.If(li >= 2), txl.Then():
                    _slow_wait(m_afree, aslot, ((li >> 1) - 1) & 1, 200)
                _rng_end(tk)
                tk = _rng("mma2-A")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(
                    tm[0] + TM_A + TM_A_STRIDE * aslot,
                    s_a[aslot],
                    s_b[aslot],
                    IDESC_M64_N80,
                    False,
                    e,
                )
                m_akk.arrive(aslot, pred=elect())
                _rng_end(tk)

        with mma3:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N1)
            e = elect_local()
            with txl.serial(n_cta) as lw:
                wslot = lw & 3
                tk = _rng("mma3-wait-tp")
                _slow_wait(m_tp, wslot, (lw >> 2) & 1, 200)
                with txl.If(lw >= 4), txl.Then():
                    _slow_wait(m_w1free, wslot, ((lw >> 2) - 1) & 1, 200)
                _rng_end(tk)
                tk = _rng("mma3-W1")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(
                    tm[0] + TM_W1 + 32 * wslot,
                    s_kt[wslot],
                    s_t2[wslot],
                    IDESC_N32_TA,
                    False,
                    e,
                )
                m_w1.arrive(wslot, pred=elect())
                _rng_end(tk)

        with tma:
            for m_ in (q_map, k_map, g_map):
                txl.ptx.prefetch.tensormap(txl.address_of(m_))
            st_rawp = txl.PipelineState(STAGES1, phase=1)
            st_gp = txl.PipelineState(GSTAGES, phase=1)

            with txl.serial(n_cta) as li:
                it = item_of(li)
                head = it % H
                tok0 = (it // H) * C
                tk = _rng("tma-wait-empty")
                _slow_wait(p_raw.empty, st_rawp.stage, st_rawp.phase)
                _rng_end(tk)
                with txl.If(elected()), txl.Then():
                    p_raw.full.arrive(st_rawp.stage, tx_count=2 * C * D * 2)
                    for d in (0, 64):
                        for m_, tile in ((q_map, s_q), (k_map, s_k)):
                            txl.ptx[TMA_G2S_HINT](
                                tile[st_rawp.stage].ptr_to(0, d),
                                txl.address_of(m_),
                                txl.int32(d),
                                tok0,
                                head,
                                p_raw.full.ptr_to([st_rawp.stage]),
                                txl.uint64(0x12F0000000000000),
                            )
                tk = _rng("tma-wait-gempty")
                _slow_wait(p_g.empty, st_gp.stage, st_gp.phase)
                _rng_end(tk)
                # Bridge the completed generic gate reads to the next TMA
                # overwrite, after acquiring the consumers' empty signal.
                txl.ptx[FENCE_ASYNC]()
                with txl.If(elected()), txl.Then():
                    p_g.full.arrive(st_gp.stage, tx_count=C * D * 2)
                    for d in (0, 64):
                        txl.ptx[TMA_G2S_HINT](
                            s_g[st_gp.stage].ptr_to(0, d),
                            txl.address_of(g_map),
                            txl.int32(d),
                            tok0,
                            head,
                            p_g.full.ptr_to([st_gp.stage]),
                            txl.uint64(0x12F0000000000000),
                        )
                st_gp.advance()
                st_rawp.advance()

        with aux:
            bw = txl.local_scalar("uint16")

            def load_beta(cc):
                """b = sigmoid(beta) of the item's 32 tokens, and the gate constants ha / hadt[k] of its head."""
                it_ = item_of(cc)
                head_ = it_ % H
                slot_ = cc & (NSLOT - 1)
                tok = txl.Cast("int64", (it_ // H) * C + lane)
                txl.ptx.ld.global_.u16(
                    bw, beta.ptr_to([tok * txl.int64(H) + txl.Cast("int64", head_)])
                )
                a_h = txl.local_scalar("float32")
                txl.ptx.ld.global_.f32(a_h, a_log.ptr_to([head_]))
                dt4 = txl.alloc_local([4], "float32")
                txl.ptx["ld.global.v4.f32"](
                    dt4[0],
                    dt4[1],
                    dt4[2],
                    dt4[3],
                    dt_bias.ptr_to([head_ * D + lane * 4]),
                )
                bf = txl.reinterpret(
                    "float32", txl.Cast("uint32", bw) << txl.uint32(16)
                )
                tb = txl.local_scalar("float32")
                txl.ptx.tanh.approx.f32(tb, bf * txl.float32(0.5))
                txl.ptx.st.shared.f32(
                    txl.address_of(s_bt[slot_, lane]),
                    tb * txl.float32(0.5) + txl.float32(0.5),
                )
                txl.ptx.ex2.approx.ftz.f32(a_h, a_h * txl.float32(LOG2E))
                ha = txl.local_scalar("float32", init=a_h * txl.float32(0.5))
                txl.ptx["st.shared.v4.f32"](
                    txl.address_of(s_hadt[slot_, lane * 4]),
                    ha * dt4[0],
                    ha * dt4[1],
                    ha * dt4[2],
                    ha * dt4[3],
                )
                with txl.If(lane == 0), txl.Then():
                    txl.ptx.st.shared.f32(txl.address_of(s_ha[slot_, 0]), ha)
                m_beta.arrive(slot_)

            for cc0 in range(NSLOT):
                with txl.If(n_cta > cc0), txl.Then():
                    load_beta(txl.int32(cc0))
            with txl.serial(n_cta) as li:
                with txl.If(li + NSLOT < n_cta), txl.Then():
                    tk = _rng("aux-wait-bdone")
                    _slow_wait(m_bdone, li & (NSLOT - 1), (li >> 3) & 1)
                    _rng_end(tk)
                    load_beta(li + NSLOT)

                with (
                    txl.If(
                        ((li & (signal_batch - 1)) == signal_batch - 1)
                        | (li == n_cta - 1)
                    ),
                    txl.Then(),
                ):
                    li0 = li & ~(signal_batch - 1)
                    tk = _rng("sig-wait")
                    for u in range(signal_batch):
                        with txl.If(li0 + u <= li), txl.Then():
                            _slow_wait(
                                m_pdone,
                                (li0 + u) & (PDONE_SLOTS - 1),
                                ((li0 + u) >> 4) & 1,
                            )
                    _rng_end(tk)
                    with txl.If((do_signal != 0) & (lane == 0)), txl.Then():
                        txl.ptx.fence.acq_rel.gpu()
                        for u in range(signal_batch):
                            with txl.If(li0 + u <= li), txl.Then():
                                txl.ptx.red.relaxed.gpu.global_.add.s32(
                                    flags.ptr_to([item_of(li0 + u)]), txl.int32(1)
                                )
                    txl.cuda.warp_sync()
                    with txl.If(lane == 0), txl.Then():
                        for u in range(signal_batch):
                            with txl.If(li0 + u <= li), txl.Then():
                                m_sfree.arrive((li0 + u) & (PDONE_SLOTS - 1))

    return kda_front


STAGES2 = 5
TM_S0 = 0
TM_SB0 = 256
TM_U0 = 384
TM_O0 = 448
BAR_ST = (3, 6, 9, 10)
BAR_EPI2 = 7
BAR_DONE2 = 8
BAR_TMEM_N2 = 704
TX2 = 4 * TILE_BYTES + 2 * SMALL_BYTES + VEC_F32 * 4


def make_chain(H: int, arch: str, hpc: int = 2):
    """Recurrence kernel: grid = H/hpc, one persistent CTA per group of hpc heads (hpc = 2: the two heads' chains
    are interleaved so that one chain's latency hides behind the other's work, freeing SMs for the concurrent
    front end; hpc = 1: one head per CTA, for head counts that leave the front end enough SMs anyway)."""
    assert H % hpc == 0 and hpc in (1, 2)

    @txl.kernel(warps=24, arch=arch, min_blocks_per_sm=1, grid=H // hpc)
    def kda_chain(
        v: txl.gptr[txl.bf16],
        state_in: txl.gptr[txl.f32],
        state_out: txl.gptr[txl.f32],
        out: txl.gptr[txl.bf16],
        vec: txl.gptr[txl.f32],
        kbar_g: txl.gptr[txl.bf16],
        qt_g: txl.gptr[txl.bf16],
        t1_g: txl.gptr[txl.bf16],
        aqk_g: txl.gptr[txl.bf16],
        w1_g: txl.gptr[txl.bf16],
        v_map: txl.TensorMap,
        kbar_map: txl.TensorMap,
        qt_map: txl.TensorMap,
        t1_map: txl.TensorMap,
        aqk_map: txl.TensorMap,
        w1_map: txl.TensorMap,
        o_map: txl.TensorMap,
        flags: txl.gptr[txl.i32],
        scale: txl.f32,
        num_chunks: txl.i32,
        flag_from: txl.i32,
        flag_target: txl.i32,
    ):
        for buf in (v, out, kbar_g, qt_g, t1_g, aqk_g, w1_g):
            txl.keep_alive(buf.ptr_to([0]))

        pair = txl.cta_id()
        head0 = pair * hpc
        tid = txl.thread_id()
        lane = txl.lane_id()

        sp = txl.specialize()

        st = sp.role("st", warps=list(range(16)), regs=88)
        epi = sp.role("epi", warps=[16, 17, 18, 19], regs=88)
        mma = sp.role("mma", warps=[20, 21], regs=40)
        tma = sp.role("tma", warps=[22], regs=40)
        idle = sp.role("idle", warps=[23], regs=40)

        smem = txl.smem_pool()
        s_tmem_addr = smem.alloc((1,), txl.i32, align=4)
        p_ring = txl.Pipeline(smem, STAGES2, full="tma", empty="tcgen05", init_empty=1)
        m_snap = txl.MBarrier(smem, 2)
        m_decay = txl.MBarrier(smem, 2)
        m_u = txl.TCGen05Bar(smem, 2)
        m_ub = txl.MBarrier(smem, 2)
        m_s = txl.TCGen05Bar(smem, 2)
        m_o = txl.TCGen05Bar(smem, 2)
        m_ofree = txl.MBarrier(smem, 2)
        m_dfree = txl.MBarrier(smem, STAGES2)
        for bar, cnt in (
            (m_snap, 2),
            (m_decay, 2),
            (m_u, 1),
            (m_ub, 1),
            (m_s, 1),
            (m_o, 1),
            (m_ofree, 1),
            (m_dfree, 1),
        ):
            bar.init(cnt)

        s_v = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_kbar = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_qt = smem.alloc((STAGES2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_w1 = smem.alloc((STAGES2, D, C), txl.bf16, swizzle=txl.SW64B)
        s_t1 = smem.alloc((STAGES2, C, C), txl.bf16, swizzle=txl.SW64B)
        s_aqkT = smem.alloc((STAGES2, C, C), txl.bf16, swizzle=txl.SW64B)
        s_dec = smem.alloc((STAGES2, D), txl.f32, align=16)
        s_qn = smem.alloc((STAGES2, C), txl.f32, align=16)
        s_o = smem.alloc((2, C, D), txl.bf16, swizzle=txl.SW128B)
        s_qne = smem.alloc((4, C), txl.f32, align=16)

        with txl.If(tid == 0), txl.Then():
            txl.ptx.fence.mbarrier_init.release.cluster()
        txl.cuda.cta_sync()

        def elect():
            return txl.cuda.elect_sync()

        def elected():
            return txl.cuda.elect_sync() != txl.uint32(0)

        def elect_local():
            e = txl.local_scalar("uint32")
            txl.assign(e, txl.cuda.elect_sync())
            return e

        def snapshot_quarter(
            tm, tm_s, tm_sb, c0, lanebits, stage, sv, sw, dvec, arrive_snap
        ):
            """Lanes = v.  Columns [c0, c0+32) of ST(h): bf16 copy -> TM_SB(h) columns [c0/2, c0/2+16), then ST *= D
            (in place).  arrive_snap(): called once the packed copy is stored (before the decay)."""
            txl.ptx[TC_LD32](
                *(sv[i] for i in range(32)), _taddr(tm, tm_s + c0, lanebits)
            )
            for vec_ in range(4):
                txl.ptx["ld.shared.v4.f32"](
                    dvec[vec_ * 4],
                    dvec[vec_ * 4 + 1],
                    dvec[vec_ * 4 + 2],
                    dvec[vec_ * 4 + 3],
                    txl.address_of(s_dec[stage, c0 + vec_ * 4]),
                )
            txl.ptx[WAIT_LD]()
            for p in range(16):
                _pack_bf16x2(sw[p], sv[2 * p], sv[2 * p + 1])
            txl.ptx[TC_ST16](
                _taddr(tm, tm_sb + c0 // 2, lanebits), *(sw[i] for i in range(16))
            )
            if arrive_snap is not None:
                txl.ptx[WAIT_ST]()
                txl.ptx[TC_FENCE_BEFORE]()
                arrive_snap()
            for p in range(8):
                prod = txl.local_scalar("uint64")
                txl.ptx.mul.rn.f32x2(
                    prod,
                    _f2(sv[2 * p], sv[2 * p + 1]),
                    _f2(dvec[2 * p], dvec[2 * p + 1]),
                )
                txl.assign(sv[2 * p], txl.cuda.float2_x(prod))
                txl.assign(sv[2 * p + 1], txl.cuda.float2_y(prod))
            for vec_ in range(4):
                txl.ptx["ld.shared.v4.f32"](
                    dvec[vec_ * 4],
                    dvec[vec_ * 4 + 1],
                    dvec[vec_ * 4 + 2],
                    dvec[vec_ * 4 + 3],
                    txl.address_of(s_dec[stage, c0 + 16 + vec_ * 4]),
                )
            for p in range(8):
                prod = txl.local_scalar("uint64")
                txl.ptx.mul.rn.f32x2(
                    prod,
                    _f2(sv[16 + 2 * p], sv[16 + 2 * p + 1]),
                    _f2(dvec[2 * p], dvec[2 * p + 1]),
                )
                txl.assign(sv[16 + 2 * p], txl.cuda.float2_x(prod))
                txl.assign(sv[16 + 2 * p + 1], txl.cuda.float2_y(prod))
            txl.ptx[TC_ST32](
                _taddr(tm, tm_s + c0, lanebits), *(sv[i] for i in range(32))
            )

        with st:
            wid_all = txl.warp_id_in_role()
            h = wid_all >> 3
            half = (wid_all >> 2) & 1
            bar_id = (
                txl.uint32(BAR_ST[0])
                + txl.Cast("uint32", h) * txl.uint32(BAR_ST[2] - BAR_ST[0])
                + txl.Cast("uint32", half) * txl.uint32(BAR_ST[1] - BAR_ST[0])
            )
            tid1 = txl.tid_in_role() & 127
            lanebits = (tid1 << 16) & 0x600000
            tm_s = TM_S0 + 128 * h
            tm_sb = TM_SB0 + 64 * h
            tm_u = TM_U0 + 32 * h
            tm_ub = tm_u
            cbase = half * 64
            with txl.If(wid_all == 0), txl.Then():
                txl.ptx["tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32"](
                    txl.address_of(s_tmem_addr[0]), txl.uint32(TMEM2_COLS)
                )
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            sv = txl.alloc_local([32], "float32")
            sw = txl.alloc_local([16], "uint32")
            dvec = txl.alloc_local([16], "float32")
            sbase = (
                txl.Cast("int64", head0 + h) * txl.int64(D) + txl.Cast("int64", tid1)
            ) * txl.int64(D) + txl.Cast("int64", cbase)
            if hpc == 1:
                # head-1 warps have no chain: skip the per-head work but still join the CTA-wide TMEM barriers
                st_if = txl.If(h < hpc)
                st_if.__enter__()
                st_then = txl.Then()
                st_then.__enter__()
            for s4 in range(2):
                for vec_ in range(4):
                    txl.ptx["ld.global.L1::no_allocate.L2::evict_first.v8.f32"](
                        *(sv[vec_ * 8 + j] for j in range(8)),
                        state_in.ptr_to([sbase + s4 * 32 + vec_ * 8]),
                    )
                txl.ptx[TC_ST32](
                    _taddr(tm, tm_s + cbase + s4 * 32, lanebits),
                    *(sv[i] for i in range(32)),
                )
            txl.ptx[WAIT_ST]()
            st_ring = txl.PipelineState(STAGES2, phase=0)
            if hpc == 2:
                with txl.If(h == 1), txl.Then():
                    st_ring.advance()
            with txl.serial(num_chunks) as c:
                stage = txl.local_scalar("int32", init=st_ring.stage)
                tk = _rng("st-wait-s")
                with txl.If(c > 0), txl.Then():
                    m_s.wait(h, (c - 1) & 1)
                p_ring.full.wait(stage, st_ring.phase)
                _rng_end(tk)
                txl.ptx[TC_FENCE_AFTER]()
                with txl.If((half == 0) & (tid1 < 32)), txl.Then():
                    qv = txl.local_scalar("float32")
                    txl.ptx.ld.shared.f32(qv, txl.address_of(s_qn[stage, tid1]))
                    txl.ptx.st.shared.f32(
                        txl.address_of(s_qne[(c & 1) * 2 + h, tid1]), qv * scale
                    )
                tk = _rng("st-snap")
                snapshot_quarter(
                    tm, tm_s, tm_sb, cbase, lanebits, stage, sv, sw, dvec, None
                )
                snapshot_quarter(
                    tm,
                    tm_s,
                    tm_sb,
                    cbase + 32,
                    lanebits,
                    stage,
                    sv,
                    sw,
                    dvec,
                    lambda: _wg_arrive(m_snap, h, bar_id, tid1 == 0),
                )
                txl.ptx[WAIT_ST]()
                txl.ptx[TC_FENCE_BEFORE]()
                txl.ptx[FENCE_ASYNC]()
                _wg_arrive(m_decay, h, bar_id, tid1 == 0)
                _rng_end(tk)
                with txl.If(half == 0), txl.Then():
                    tk = _rng("st-wait-u")
                    m_u.wait(h, c & 1)
                    _rng_end(tk)
                    tk = _rng("st-ub")
                    txl.ptx[TC_FENCE_AFTER]()
                    txl.ptx[TC_LD32](
                        *(sv[i] for i in range(32)), _taddr(tm, tm_u, lanebits)
                    )
                    txl.ptx[WAIT_LD]()
                    for p in range(16):
                        _pack_bf16x2(sw[p], sv[2 * p], sv[2 * p + 1])
                    txl.ptx[TC_ST16](
                        _taddr(tm, tm_ub, lanebits), *(sw[i] for i in range(16))
                    )
                    txl.ptx[WAIT_ST]()
                    txl.ptx[TC_FENCE_BEFORE]()
                    _wg_arrive(m_ub, h, bar_id, tid1 == 0)
                    _rng_end(tk)
                for _ in range(hpc):
                    st_ring.advance()
            m_s.wait(h, (num_chunks - 1) & 1)
            txl.ptx[TC_FENCE_AFTER]()
            for s4 in range(2):
                txl.ptx[TC_LD32](
                    *(sv[i] for i in range(32)),
                    _taddr(tm, tm_s + cbase + s4 * 32, lanebits),
                )
                txl.ptx[WAIT_LD]()
                for vec_ in range(4):
                    txl.ptx["st.global.L1::no_allocate.L2::evict_first.v8.f32"](
                        state_out.ptr_to([sbase + s4 * 32 + vec_ * 8]),
                        *(sv[vec_ * 8 + j] for j in range(8)),
                    )
            if hpc == 1:
                st_then.__exit__(None, None, None)
                st_if.__exit__(None, None, None)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))
            with txl.If(wid_all == 0), txl.Then():
                txl.ptx["tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned"]()
                txl.ptx["tcgen05.dealloc.cta_group::1.sync.aligned.b32"](
                    txl.Cast("uint32", tm[0]), txl.uint32(TMEM2_COLS)
                )

        with epi:
            tid3 = txl.tid_in_role()
            wq = tid3 >> 5
            lane3 = tid3 & 31
            lanebits3 = (tid3 << 16) & 0x600000
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            txl.ptx.prefetch.tensormap(txl.address_of(o_map))

            fr = txl.alloc_local([32], "float32")
            qs = txl.alloc_local([8], "float32")
            pk = txl.alloc_local([16], "uint32")

            i_l = ((lane3 >> 4) & 1) * 8 + (lane3 & 7)
            v_l = wq * 32 + ((lane3 >> 3) & 1) * 8
            ci = (lane3 & 3) * 2

            def epilogue(c, h):
                """o[i][v] = scale * qn_i * O^T[v][i] for chunk c of head h (TM_O(h), staging slot h)."""
                tk = _rng("epi-wait-o")
                m_o.wait(h, c & 1)
                _rng_end(tk)
                tk = _rng("epi")
                txl.ptx[TC_FENCE_AFTER]()
                for hh in range(2):
                    txl.ptx[TC_LD256_X4](
                        *(fr[16 * hh + i] for i in range(16)),
                        _taddr(tm, TM_O0 + 32 * h, lanebits3 + txl.int32(hh << 20)),
                    )
                for rep in range(4):
                    txl.ptx["ld.shared.v2.f32"](
                        qs[2 * rep],
                        qs[2 * rep + 1],
                        txl.address_of(s_qne[(c & 1) * 2 + h, 8 * rep + ci]),
                    )
                txl.ptx[WAIT_LD]()
                txl.ptx[TC_FENCE_BEFORE]()
                with txl.If(tid3 == 0), txl.Then():
                    txl.ptx[BULK_WAIT_READ](hpc - 1)
                txl.ptx.bar.sync(txl.uint32(BAR_EPI2), txl.uint32(128))
                with txl.If(tid3 == 0), txl.Then():
                    m_ofree.arrive(h)
                txl.ptx[FENCE_ASYNC]()
                for hh in range(2):
                    for rep in range(4):
                        for rb in range(2):
                            k = 16 * hh + 4 * rep + 2 * rb
                            _pack_bf16x2(
                                pk[8 * hh + 2 * rep + rb],
                                fr[k] * qs[2 * rep],
                                fr[k + 1] * qs[2 * rep + 1],
                            )
                ot = s_o[h]
                for hh in range(2):
                    for r0 in range(2):
                        txl.ptx[STM_X4T](
                            ot.ptr_to(16 * r0 + i_l, 16 * hh + v_l),
                            pk[8 * hh + 4 * r0],
                            pk[8 * hh + 4 * r0 + 1],
                            pk[8 * hh + 4 * r0 + 2],
                            pk[8 * hh + 4 * r0 + 3],
                        )
                txl.ptx[FENCE_ASYNC]()
                txl.ptx.bar.sync(txl.uint32(BAR_EPI2), txl.uint32(128))
                with txl.If(tid3 == 0), txl.Then():
                    for d in (0, 64):
                        txl.ptx[TMA_S2G](
                            txl.address_of(o_map),
                            txl.int32(d),
                            c * C,
                            head0 + h,
                            ot.ptr_to(0, d),
                        )
                    txl.ptx[BULK_COMMIT]()
                _rng_end(tk)

            with txl.serial(num_chunks) as c:
                for hh_ in range(hpc):
                    epilogue(c, txl.int32(hh_))
            with txl.If(tid3 == 0), txl.Then():
                txl.ptx[BULK_WAIT](0)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))

        with mma:
            tm = _tmem_preamble(s_tmem_addr, BAR_TMEM_N2)
            hm = txl.warp_id_in_role()
            tm_s = TM_S0 + 128 * hm
            tm_u = TM_U0 + 32 * hm
            tm_ub = tm_u
            tm_o = TM_O0 + 32 * hm
            tm_sb = TM_SB0 + 64 * hm
            st_r = txl.PipelineState(STAGES2, phase=0)
            if hpc == 2:
                with txl.If(hm == 1), txl.Then():
                    st_r.advance()
            e = elect_local()
            if hpc == 1:
                mma_if = txl.If(hm < hpc)
                mma_if.__enter__()
                mma_then = txl.Then()
                mma_then.__enter__()
            with txl.serial(num_chunks) as c:
                stage = txl.local_scalar("int32", init=st_r.stage)
                tk = _rng("mma-wait-ring")
                p_ring.full.wait(stage, st_r.phase)
                _rng_end(tk)
                tk = _rng("mma-U1")
                _chain(tm[0] + tm_u, s_v[stage], s_t1[stage], IDESC_N32_TA, False, e)
                _rng_end(tk)
                tk = _rng("mma-wait-snap")
                m_snap.wait(hm, c & 1)
                with txl.If(c >= 1), txl.Then():
                    m_ofree.wait(hm, (c - 1) & 1)
                _rng_end(tk)
                tk = _rng("mma-U2O1")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(
                    tm[0] + tm_u, tm[0] + tm_sb, s_w1[stage], IDESC_N32_TB_NEG, True, e
                )
                m_u.arrive(hm, pred=elect())
                _chain(tm[0] + tm_o, tm[0] + tm_sb, s_qt[stage], IDESC_N32, False, e)
                _rng_end(tk)
                tk = _rng("mma-wait-ub")
                m_ub.wait(hm, c & 1)
                m_decay.wait(hm, c & 1)
                _rng_end(tk)
                tk = _rng("mma-SO2")
                txl.ptx[TC_FENCE_AFTER]()
                _chain(
                    tm[0] + tm_s, tm[0] + tm_ub, s_kbar[stage], IDESC_N128_TB, True, e
                )
                m_s.arrive(hm, pred=elect())
                _chain(tm[0] + tm_o, tm[0] + tm_ub, s_aqkT[stage], IDESC_N32, True, e)
                m_o.arrive(hm, pred=elect())
                p_ring.empty.arrive(stage, pred=elect())
                _rng_end(tk)
                for _ in range(hpc):
                    st_r.advance()
            if hpc == 1:
                mma_then.__exit__(None, None, None)
                mma_if.__exit__(None, None, None)
            txl.ptx.bar.sync(txl.uint32(BAR_DONE2), txl.uint32(BAR_TMEM_N2))

        with tma:
            for m_ in (v_map, kbar_map, qt_map, t1_map, aqk_map, w1_map):
                txl.ptx.prefetch.tensormap(txl.address_of(m_))
            st_p = txl.PipelineState(STAGES2, phase=1)

            ready_upto = txl.local_scalar("int32", init=txl.int32(0))
            n_items = num_chunks * hpc
            with txl.serial(n_items) as j:
                c = j // hpc
                hh = j % hpc
                it = c * H + head0 + hh
                tk = _rng("tma-wait-flag")
                with txl.If((it >= flag_from) & (j >= ready_upto)), txl.Then():
                    jj = j + lane
                    itl = (jj // hpc) * H + head0 + (jj % hpc)
                    in_range = (jj < n_items) & (itl >= flag_from)
                    fl = txl.local_scalar("int32", init=flag_target)
                    nready = txl.local_scalar("int32", init=txl.int32(0))
                    with txl.While(nready == 0):
                        with txl.If(in_range), txl.Then():
                            txl.ptx.ld.acquire.gpu.global_.s32(fl, flags.ptr_to([itl]))
                        ballot = txl.local_scalar("uint32")
                        txl.ptx.vote_sync.ballot.b32(
                            ballot,
                            txl.ptx.pred(fl >= flag_target),
                            txl.uint32(0xFFFFFFFF),
                        )

                        trail = ballot & ~(ballot + txl.uint32(1))
                        txl.assign(nready, txl.Cast("int32", txl.popcount(trail)))
                        with txl.If(nready == 0), txl.Then():
                            txl.cuda.nano_sleep(txl.uint64(SLOW_WAIT_NS))
                    txl.ptx.fence.proxy.async_.global_()

                    with txl.If((lane < nready) & in_range), txl.Then():
                        txl.ptx.st.global_.s32(flags.ptr_to([itl]), txl.int32(0))
                    txl.assign(ready_upto, j + nready)
                _rng_end(tk)
                tk = _rng("tma-wait-empty")
                with txl.If(j >= STAGES2), txl.Then():
                    m_dfree.wait(st_p.stage, ((j - STAGES2) // STAGES2) & 1)
                p_ring.empty.wait(st_p.stage, st_p.phase)
                _rng_end(tk)
                stg = st_p.stage
                with txl.If(elected()), txl.Then():
                    p_ring.full.arrive(stg, tx_count=TX2)
                    for d in (0, 64):
                        txl.ptx[TMA_G2S_HINT](
                            s_v[stg].ptr_to(0, d),
                            txl.address_of(v_map),
                            txl.int32(d),
                            c * C,
                            head0 + hh,
                            p_ring.full.ptr_to([stg]),
                            txl.uint64(0x12F0000000000000),
                        )
                        txl.ptx[TMA_G2S](
                            s_kbar[stg].ptr_to(0, d),
                            txl.address_of(kbar_map),
                            txl.int32(d),
                            txl.int32(0),
                            it,
                            p_ring.full.ptr_to([stg]),
                        )
                        txl.ptx[TMA_G2S](
                            s_qt[stg].ptr_to(0, d),
                            txl.address_of(qt_map),
                            txl.int32(d),
                            txl.int32(0),
                            it,
                            p_ring.full.ptr_to([stg]),
                        )
                    txl.ptx[TMA_G2S](
                        s_t1[stg].ptr_to(0, 0),
                        txl.address_of(t1_map),
                        txl.int32(0),
                        txl.int32(0),
                        it,
                        p_ring.full.ptr_to([stg]),
                    )
                    txl.ptx[TMA_G2S](
                        s_aqkT[stg].ptr_to(0, 0),
                        txl.address_of(aqk_map),
                        txl.int32(0),
                        txl.int32(0),
                        it,
                        p_ring.full.ptr_to([stg]),
                    )
                    txl.ptx[TMA_G2S](
                        s_w1[stg].ptr_to(0, 0),
                        txl.address_of(w1_map),
                        txl.int32(0),
                        txl.int32(0),
                        it,
                        p_ring.full.ptr_to([stg]),
                    )
                    vbase = txl.Cast("int64", it) * txl.int64(VEC_F32)
                    txl.ptx[BULK_G2S](
                        txl.address_of(s_dec[stg, 0]),
                        vec.ptr_to([vbase]),
                        txl.uint32(D * 4),
                        p_ring.full.ptr_to([stg]),
                    )
                    txl.ptx[BULK_G2S](
                        txl.address_of(s_qn[stg, 0]),
                        vec.ptr_to([vbase + txl.int64(D)]),
                        txl.uint32(C * 4),
                        p_ring.full.ptr_to([stg]),
                    )

                st_p.advance()

        with idle:
            # scratch is dead once its TMA transfer completes: the idle warp, not the latency-sensitive TMA issuer, discards each completed item from L2
            st_d = txl.PipelineState(STAGES2, phase=0)
            with txl.serial(num_chunks * hpc) as j:
                p_ring.full.wait(st_d.stage, st_d.phase)
                itp = txl.Cast("int64", (j // hpc) * H + head0 + (j % hpc))
                l64 = txl.Cast("int64", lane)
                for tile_g, nlines in ((kbar_g, 64), (qt_g, 64), (w1_g, 64)):
                    for base_ in range(0, nlines, 32):
                        txl.ptx["discard.global.L2"](
                            tile_g.ptr_to(
                                [itp * txl.int64(C * D) + (l64 + base_) * txl.int64(64)]
                            )
                        )
                for tile_g in (t1_g, aqk_g):
                    with txl.If(lane < 16), txl.Then():
                        txl.ptx["discard.global.L2"](
                            tile_g.ptr_to(
                                [itp * txl.int64(C * C) + l64 * txl.int64(64)]
                            )
                        )
                with txl.If(lane < 5), txl.Then():
                    txl.ptx["discard.global.L2"](
                        vec.ptr_to([itp * txl.int64(VEC_F32) + l64 * txl.int64(32)])
                    )
                txl.cuda.warp_sync()
                with txl.If(lane == 0), txl.Then():
                    m_dfree.arrive(st_d.stage)
                st_d.advance()

    return kda_chain
