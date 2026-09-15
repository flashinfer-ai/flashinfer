"""MSA sparse-decode attention for SM90 (Hopper).

Ported from the Kernel-Factory winner (2.155x geomean over the shipped Triton
path on 60 held-out shapes, none slower). WGMMA N follows the GQA group so no N
lane idles at the common tp=8 geometry; partial results are combined in-kernel.

The compile cache keys on (topk, nsplit, stages, nc, D, page, wide_addr) --
schedule parameters only, never raw shapes, so one compile serves every shape.
"""

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.runtime import from_dlpack
from cutlass.cute.nvgpu import warpgroup
from cutlass.utils import LayoutEnum

_KT = 64        # keys per pipeline stage == WGMMA M of gemm-1 (hardware M)
_THREADS = 128
_MERGE_THREADS = 256
_LOG2E = 1.4426950408889634
_NEG = -3.0e38

# H100: 132 SMs, 228 KiB of shared memory per SM.
_NUM_SM = 132
_SMEM_PER_SM = 228 * 1024
_SMEM_MAX_CTA = 227 * 1024
_MAX_RESIDENT = 4
_PROLOGUE_TILES = 3
_COMBINE_TILES = 4
_MIN_SPLIT_TILES = 4
_SATURATED_CTA_PER_SM = 3
_MAX_STAGES = 6
_SPLITS = (1, 2, 4, 8, 16)


class MsaSparseDecode:
    def __init__(self, topk, nsplit, stages, nc, d, page, wide_addr):
        # Partials are read in batches whose fragment array stays inside 32
        # registers, so the fused combine never raises the mainloop's
        # allocation.  Eight fp16 per access is one 16-byte vector.
        self.mgs = max(1, min(nsplit, 8))
        self.topk = topk
        self.nsplit = nsplit
        self.stages = stages
        self.nc = nc
        self.d = d
        self.page = page
        self.wide_addr = wide_addr
        self.acc_dtype = cutlass.Float32

    # ------------------------------------------------------------------ host
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,     # (total_q*Hq*D,) bf16, flat
        mKV: cute.Tensor,    # (num_pages, Hkv, PAGE, 2D) fp8 packed cache
        mIdx: cute.Tensor,   # (Hkv, total_q, topk) i32
        mPT: cute.Tensor,    # (B, max_pages) i32
        mSeq: cute.Tensor,   # (B,) i32
        mO: cute.Tensor,     # (total_q*Hq*D,) bf16, flat
        wO: cute.Tensor,     # (base*nsplit*nc*D,) f16 normalized scratch, flat
        wL: cute.Tensor,     # (base*nsplit, nc) f32 scratch
        wM: cute.Tensor,     # (base*nsplit,) f32 scratch
        wC: cute.Tensor,     # (base,) i32 combine counters
        sq: cutlass.Int32,
        gsz: cutlass.Int32,
        ng: cutlass.Int32,
        hkv: cutlass.Int32,
        ntok: cutlass.Int32,
        stream: cuda.CUstream,
    ):
        st = self.stages
        nc = self.nc
        D = self.d
        PAGE = self.page
        fp8 = cutlass.Float8E4M3FN
        bf16 = cutlass.BFloat16
        # fp16 (not bf16) for the PV operands: SM90 converts fp8 -> f16x2 with a
        # single cvt, and fp16 also carries 3 more mantissa bits than bf16.
        pdt = cutlass.Float16

        # K and V are the two halves of each packed 2D-wide row.  Expose them as
        # 4-mode (token, d, kv_head, page) views of the one cache tensor so the
        # page and head indices stay separate TMA coordinates -- no host-side
        # slicing and no collapsed mode whose extent grows with the cache size.
        npage = cute.size(mKV, mode=[0])
        nhead = cute.size(mKV, mode=[1])
        kv_glayout = cute.make_layout(
            (PAGE, D, nhead, npage),
            stride=(2 * D, 1, PAGE * 2 * D, nhead * PAGE * 2 * D))
        mKg = cute.make_tensor(mKV.iterator, kv_glayout)
        mVg = cute.make_tensor(mKV.iterator + D, kv_glayout)

        kv_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.ROW_MAJOR, (_KT, nc, D), fp8, st)
        vb_layout_staged = sm90_utils.make_smem_layout_a(
            LayoutEnum.COL_MAJOR, (D, nc, _KT), pdt, 1)
        p_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (D, nc, _KT), pdt, 1)
        q_layout_staged = sm90_utils.make_smem_layout_b(
            LayoutEnum.ROW_MAJOR, (_KT, nc, D), fp8, 1)

        qk_mma = sm90_utils.make_trivial_tiled_mma(
            fp8, fp8,
            cute.nvgpu.OperandMajorMode.K, cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype, (1, 1, 1), tiler_mn=(_KT, nc))
        pv_mma = sm90_utils.make_trivial_tiled_mma(
            pdt, pdt,
            cute.nvgpu.OperandMajorMode.MN, cute.nvgpu.OperandMajorMode.K,
            self.acc_dtype, (1, 1, 1), tiler_mn=(64, nc))

        kv_layout_one = cute.slice_(kv_layout_staged, (None, None, 0))
        tma_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_atom_k, tma_tensor_k = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mKg, kv_layout_one, (_KT, D))
        tma_atom_v, tma_tensor_v = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_op, mVg, kv_layout_one, (_KT, D))
        tx_bytes = 2 * cute.size_in_bytes(fp8, kv_layout_one)

        tk = self.topk
        ntmax = tk * (PAGE // _KT)

        @cute.struct
        class SharedStorage:
            mbar: cute.struct.MemRange[cutlass.Int64, 2 * st]
            sPg: cute.struct.MemRange[cutlass.Int32, tk]
            sKst: cute.struct.MemRange[cutlass.Int32, tk]
            sNT: cute.struct.MemRange[cutlass.Int32, 1]
            sDone: cute.struct.MemRange[cutlass.Int32, 1]
            sRed: cute.struct.MemRange[cutlass.Float32, 8]
            sL: cute.struct.MemRange[cutlass.Float32, nc]
            sQ: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(q_layout_staged)], 1024]
            sP: cute.struct.Align[
                cute.struct.MemRange[pdt, cute.cosize(p_layout_staged)], 1024]
            sVb: cute.struct.Align[
                cute.struct.MemRange[pdt, cute.cosize(vb_layout_staged)], 1024]
            sK: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(kv_layout_staged)], 1024]
            sV: cute.struct.Align[
                cute.struct.MemRange[fp8, cute.cosize(kv_layout_staged)], 1024]

        self.shared_storage = SharedStorage

        base = ntok * hkv * ng

        self.kernel(
            tma_atom_k, tma_tensor_k, tma_atom_v, tma_tensor_v,
            mQ, mIdx, mPT, mSeq, mO, wO, wL, wM, wC, qk_mma, pv_mma,
            kv_layout_staged, vb_layout_staged, p_layout_staged, q_layout_staged,
            sq, gsz, ng, hkv, tx_bytes,
        ).launch(grid=(base * self.nsplit, 1, 1), block=[_THREADS, 1, 1],
                 stream=stream)

    # ---------------------------------------------------------------- device
    @cute.kernel
    def kernel(
        self,
        tma_atom_k: cute.CopyAtom,
        mKg: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mVg: cute.Tensor,
        mQ: cute.Tensor,
        mIdx: cute.Tensor,
        mPT: cute.Tensor,
        mSeq: cute.Tensor,
        mO: cute.Tensor,
        wO: cute.Tensor,
        wL: cute.Tensor,
        wM: cute.Tensor,
        wC: cute.Tensor,
        qk_mma: cute.TiledMma,
        pv_mma: cute.TiledMma,
        kv_layout_staged: cute.ComposedLayout,
        vb_layout_staged: cute.ComposedLayout,
        p_layout_staged: cute.ComposedLayout,
        q_layout_staged: cute.ComposedLayout,
        sq: cutlass.Int32,
        gsz: cutlass.Int32,
        ng: cutlass.Int32,
        hkv: cutlass.Int32,
        tx_bytes: cutlass.Int32,
    ):
        st = self.stages
        tk = self.topk
        ns = self.nsplit
        nc = self.nc
        D = self.d
        PAGE = self.page
        HALVES = PAGE // _KT
        ntmax = tk * HALVES
        # elements of the (nc, D) output tile owned by each of the 128 threads
        E = nc * D // _THREADS
        LANES = D // E              # threads cooperating on one output row
        fp8 = cutlass.Float8E4M3FN
        bf16 = cutlass.BFloat16
        pdt = cutlass.Float16

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % 32

        gidx = bidx // ns
        spl = bidx % ns
        gb = gidx % ng
        r1 = gidx // ng
        ii = r1 % sq
        r2 = r1 // sq
        h = r2 % hkv
        b = r2 // hkv
        t = b * sq + ii

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mbar = storage.mbar.data_ptr()
        sPg = storage.sPg.get_tensor(cute.make_layout(tk))
        sKst = storage.sKst.get_tensor(cute.make_layout(tk))
        sNT = storage.sNT.get_tensor(cute.make_layout(1))
        sDone = storage.sDone.get_tensor(cute.make_layout(1))
        sRed = storage.sRed.get_tensor(cute.make_layout(8))
        sL = storage.sL.get_tensor(cute.make_layout(nc))

        sK = storage.sK.get_tensor(kv_layout_staged.outer, swizzle=kv_layout_staged.inner)
        # Epilogue-only buffers alias the (by then dead) K/V staging ring.
        pf32 = cute.recast_ptr(storage.sK.data_ptr(), None, cutlass.Float32)
        sLK = cute.make_tensor(pf32, cute.make_layout((nc, _KT), stride=(_KT + 4, 1)))
        sOf = cute.make_tensor(pf32 + nc * (_KT + 4),
                               cute.make_layout((nc, D), stride=(D + 4, 1)))
        sV = storage.sV.get_tensor(kv_layout_staged.outer, swizzle=kv_layout_staged.inner)
        sVb = storage.sVb.get_tensor(vb_layout_staged.outer, swizzle=vb_layout_staged.inner)
        sP = storage.sP.get_tensor(p_layout_staged.outer, swizzle=p_layout_staged.inner)
        sQ = storage.sQ.get_tensor(q_layout_staged.outer, swizzle=q_layout_staged.inner)

        if warp_idx == 0:
            with cute.arch.elect_one():
                for s in cutlass.range_constexpr(st):
                    cute.arch.mbarrier_init(mbar + s, 1)
                    cute.arch.mbarrier_init(mbar + st + s, 1)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
        cute.arch.mbarrier_init_fence()

        sk = mSeq[b]
        qpos = sk - sq + ii

        # ---- index chain, issued first ---------------------------------
        # mIdx -> page_table is a dependent pair of global loads and it is what
        # gates the CTA's first TMA.  Issuing it here, with the Q load in
        # between the two halves, costs two global round trips before the first
        # gather instead of the three that Q-then-index would take.
        LDI = (ntmax > 32)
        ld_ok = (tidx < tk) if LDI else ((warp_idx == 0) & (lane < tk))
        li = tidx if LDI else lane
        sel = cutlass.Int32(-1)
        if ld_ok:
            sel = mIdx[h, t, li]

        # ---- Q: bf16 -> fp8 into sQ ------------------------------------
        # Global q/out/partial traffic is addressed in 16-byte chunks arranged
        # so consecutive lanes hold consecutive chunks of the same row.  The
        # (row, col-block) tiling the epilogue reduction uses gives each lane a
        # 32-byte stride instead, which splits every 32-byte sector across two
        # instructions and doubles the sectors the memory system has to move.
        CH = cutlass.const_expr(8)              # elements per 16-byte access
        CPR = cutlass.const_expr(D // CH)       # chunks per output row
        NCK = cutlass.const_expr(nc * D // (CH * _THREADS))
        RSTEP = cutlass.const_expr(_THREADS // CPR)
        crow = tidx // CPR
        ccol = tidx % CPR
        grow = tidx // LANES
        gcol = tidx % LANES
        qbase = t * (gsz * hkv) + h * gsz + gb * nc
        mQv = cute.tiled_divide(mQ, (CH,))
        qf = cute.make_rmem_tensor(cute.make_layout((CH, NCK)), bf16)
        for j in cutlass.range_constexpr(NCK):
            r = j * RSTEP + crow
            if (gb * nc + r) < gsz:
                if cutlass.const_expr(self.wide_addr):
                    off64 = ((cutlass.Int64(qbase) + cutlass.Int64(r))
                             * cutlass.Int64(D) + cutlass.Int64(ccol * CH))
                    cute.autovec_copy(
                        cute.make_tensor(mQ.iterator + off64,
                                         cute.make_layout(CH)), qf[(None, j)])
                else:
                    cute.autovec_copy(
                        mQv[(None,), (qbase + r) * CPR + ccol], qf[(None, j)])
            else:
                for e in cutlass.range_constexpr(CH):
                    qf[(e, j)] = bf16(0.0)

        # second half of the chain: in flight while Q is still on the wire
        good = ld_ok & (sel >= 0)
        selc = sel if good else cutlass.Int32(0)
        pg = cutlass.Int32(0)
        if ld_ok:
            pg = mPT[b, selc]

        # ---- tile count --------------------------------------------------
        # Selected block ids are ascending and the valid ones occupy a prefix,
        # so giving an invalid slot a key offset past every reachable position
        # makes the half-tile key offsets monotone over the whole list.  The
        # causally visible tiles are then a prefix of it and a population count
        # over the visibility predicate is the entire tile list -- no
        # compaction, no per-tile page/offset arrays.
        if ld_ok:
            sPg[li] = pg
            sKst[li] = (selc * PAGE) if good else cutlass.Int32(0x3FFFFFFF)
        if cutlass.const_expr(ntmax > 32):
            cute.arch.barrier()
            if tidx == 0:
                n = cutlass.Int32(0)
                for u in cutlass.range_constexpr(ntmax):
                    kbu = sKst[u // HALVES] + (u % HALVES) * _KT
                    n += cutlass.Int32(1) if kbu <= qpos else cutlass.Int32(0)
                sNT[0] = n
        # TMA partitions are pure address arithmetic and can be built before
        # either the tile count or the Q conversion finishes.
        gK = cute.flat_divide(mKg, (_KT, D))[None, None, None, 0, None, None]
        gV = cute.flat_divide(mVg, (_KT, D))[None, None, None, 0, None, None]
        tKsK, tKgK = cute.nvgpu.cpasync.tma_partition(
            tma_atom_k, 0, cute.make_layout(1),
            cute.group_modes(sK, 0, 2), cute.group_modes(gK, 0, 2))
        tVsV, tVgV = cute.nvgpu.cpasync.tma_partition(
            tma_atom_v, 0, cute.make_layout(1),
            cute.group_modes(sV, 0, 2), cute.group_modes(gV, 0, 2))

        # For the normal <=32-tile list, warp 0 alone produced the shared
        # page metadata.  A warp sync is therefore sufficient for it to count
        # visibility and open the first TMA ring before Q conversion/fencing.
        if cutlass.const_expr(ntmax <= 32):
            if warp_idx == 0:
                cute.arch.sync_warp()
                jb = (lane // HALVES) if lane < ntmax else cutlass.Int32(0)
                kb = sKst[jb] + (lane % HALVES) * _KT
                bal = cute.arch.vote_ballot_sync((lane < ntmax) & (kb <= qpos))
                ntw = cute.arch.popc(bal)
                if lane == 0:
                    sNT[0] = ntw
                if cutlass.const_expr(ns > 1):
                    a0 = (spl * ntw) // ns
                    a1 = ((spl + 1) * ntw) // ns
                    nl = a1 - a0
                else:
                    a0 = cutlass.Int32(0)
                    nl = ntw
                for s in cutlass.range_constexpr(st - 1):
                    if s < nl:
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                mbar + s, tx_bytes)
                        tl = a0 + s
                        hf = tl % HALVES
                        pj = sPg[tl // HALVES]
                        cute.copy(tma_atom_k, tKgK[(None, hf, h, pj)],
                                  tKsK[(None, s)], tma_bar_ptr=mbar + s)
                        cute.copy(tma_atom_v, tVgV[(None, hf, h, pj)],
                                  tVsV[(None, s)], tma_bar_ptr=mbar + s)

        sQ2 = cute.make_tensor(sQ.iterator, cute.select(sQ.layout, mode=[0, 1]))
        sQc = cute.tiled_divide(sQ2, (1, CH))
        qf8 = cute.make_rmem_tensor(cute.make_layout(CH), fp8)
        for j in cutlass.range_constexpr(NCK):
            qf8.store(qf[(None, j)].load().to(fp8))
            cute.autovec_copy(qf8, sQc[(0, None), j * RSTEP + crow, ccol])
        cute.arch.fence_proxy("async.shared", space="cta")

        if tidx < 8:
            sRed[tidx] = cutlass.Float32(_NEG)
        cute.arch.barrier()

        nt = sNT[0]
        if cutlass.const_expr(ns > 1):
            t0 = (spl * nt) // ns
            t1 = ((spl + 1) * nt) // ns
            nloc = t1 - t0
        else:
            t0 = cutlass.Int32(0)
            nloc = nt

        # ---- MMA partitions ----------------------------------------------
        qk_thr = qk_mma.get_slice(tidx)
        pv_thr = pv_mma.get_slice(tidx)

        tSrK = qk_mma.make_fragment_A(qk_thr.partition_A(sK))
        tSrQ = qk_mma.make_fragment_B(qk_thr.partition_B(sQ))[(None, None, None, 0)]
        tScS = qk_thr.partition_C(cute.make_identity_tensor((_KT, nc)))
        accS = cute.make_rmem_tensor(tScS.shape, self.acc_dtype)

        tOrV = pv_mma.make_fragment_A(pv_thr.partition_A(sVb))[(None, None, None, 0)]
        tOrP = pv_mma.make_fragment_B(pv_thr.partition_B(sP))[(None, None, None, 0)]
        tOcO = pv_thr.partition_C(cute.make_identity_tensor((D, nc)))
        accO = cute.make_rmem_tensor(tOcO.shape, self.acc_dtype)
        accO.fill(0.0)

        qk_mma.set(warpgroup.Field.ACCUMULATE, True)
        pv_mma.set(warpgroup.Field.ACCUMULATE, True)

        # ---- fp8 -> fp16 V conversion tiled copy -------------------------
        # Eight fp8 values expand to one 16-byte fp16 store.  Sixteen values
        # would expand to two stores whose lanes hit alternating shared slots,
        # doubling the shared-memory wavefronts for this staging copy.
        cvt_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), fp8,
                                       num_bits_per_copy=64)
        tpc = D // 8                        # threads spanning one fp8 row
        cvt_tiled = cute.make_tiled_copy_tv(
            cvt_atom,
            cute.make_ordered_layout((_THREADS // tpc, tpc), order=(1, 0)),
            cute.make_layout((1, 8)))
        cvt_thr = cvt_tiled.get_slice(tidx)

        sRed4 = cute.tiled_divide(sRed, (4,))
        rfrag = cute.make_rmem_tensor(cute.make_layout(4), cutlass.Float32)
        NKQ = cutlass.const_expr(cute.size(tSrK, mode=[2]))
        NS = cute.size(accS)
        NO = cute.size(accO)
        lp = cute.make_rmem_tensor(cute.make_layout(NS), self.acc_dtype)
        lp.fill(0.0)
        mrun = cutlass.Float32(0.0)
        scale2 = cutlass.Float32(_LOG2E / (D ** 0.5))
        lg = tidx // LANES

        sVb_one = sVb[None, None, 0]
        sVb_kd = cute.make_tensor(sVb_one.iterator,
                                  cute.select(sVb_one.layout, mode=[1, 0]))
        sP_one = sP[None, None, 0]
        sP_kg = cute.make_tensor(sP_one.iterator,
                                 cute.select(sP_one.layout, mode=[1, 0]))

        # ---- prologue loads (wide-topk only; the common path already opened
        # the ring before Q conversion) ------------------------------------
        if cutlass.const_expr(ntmax > 32) and warp_idx == 0:
            for s in cutlass.range_constexpr(st - 1):
                if s < nloc:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar + s, tx_bytes)
                    tl = t0 + s
                    hf = tl % HALVES
                    pj = sPg[tl // HALVES]
                    cute.copy(tma_atom_k, tKgK[(None, hf, h, pj)],
                              tKsK[(None, s)], tma_bar_ptr=mbar + s)
                    cute.copy(tma_atom_v, tVgV[(None, hf, h, pj)],
                              tVsV[(None, s)], tma_bar_ptr=mbar + s)

        # ---- mainloop -----------------------------------------------------
        for u in cutlass.range(nloc, unroll=1):
            ti = t0 + u
            stage = u % st
            kphase = (u // st) % 2

            if warp_idx == 0:
                lu = u + st - 1
                if lu < nloc:
                    ls = lu % st
                    cute.arch.mbarrier_wait(mbar + st + ls, ((lu // st) + 1) % 2)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(mbar + ls, tx_bytes)
                    tl = t0 + lu
                    hf = tl % HALVES
                    pj = sPg[tl // HALVES]
                    cute.copy(tma_atom_k, tKgK[(None, hf, h, pj)],
                              tKsK[(None, ls)], tma_bar_ptr=mbar + ls)
                    cute.copy(tma_atom_v, tVgV[(None, hf, h, pj)],
                              tVsV[(None, ls)], tma_bar_ptr=mbar + ls)

            cute.arch.mbarrier_wait(mbar + stage, kphase)

            # gemm 1: S^T = K . Q^T.  The first k step initializes accS in
            # WGMMA, removing the explicit register zero-fill dependency.
            warpgroup.fence()
            qk_mma.set(warpgroup.Field.ACCUMULATE, False)
            for kk in cutlass.range_constexpr(NKQ):
                cute.gemm(qk_mma, accS, tSrK[(None, None, kk, stage)],
                          tSrQ[(None, None, kk)], accS)
                qk_mma.set(warpgroup.Field.ACCUMULATE, True)
            warpgroup.commit_group()

            # Pull the whole V tile into registers while QK(u) and PV(u-1)
            # are still running.  Only conversion and the single-buffered
            # shared stores depend on the drain below.
            src = cvt_thr.partition_S(sV[None, None, stage])
            dst = cvt_thr.partition_D(sVb_kd)
            NCV = cutlass.const_expr(cute.size(src, mode=[1]))
            VW = cutlass.const_expr(cute.size(src, mode=[0]))
            vf8 = cute.make_rmem_tensor(cute.make_layout((VW, NCV)), fp8)
            for c in cutlass.range_constexpr(NCV):
                cute.autovec_copy(src[None, c, 0], vf8[(None, c)])

            # These values also depend only on shared state from the previous
            # iteration, so resolve them before waiting for the accumulator.
            cute.autovec_copy(sRed4[(None,), (u + 1) % 2], rfrag)
            tprev = cute.arch.fmax(cute.arch.fmax(rfrag[0], rfrag[1]),
                                   cute.arch.fmax(rfrag[2], rfrag[3]))
            mnew = cute.arch.fmax(mrun, tprev)
            kbase = sKst[ti // HALVES] + (ti % HALVES) * _KT

            warpgroup.wait_group(0)   # QK(u) and PV(u-1) retired

            vbf = cute.make_rmem_tensor(cute.make_layout(VW), pdt)
            for c in cutlass.range_constexpr(NCV):
                vbf.store(vf8[(None, c)].load().to(pdt))
                cute.autovec_copy(vbf, dst[None, c, 0])

            # Online softmax with a one-tile-lagged offset: the running max is
            # updated from the previous tile's warpgroup max, which is already
            # in SMEM, so no extra CTA barrier is needed here.  Any
            # non-decreasing offset sequence is exact; fmin caps the exponent
            # so a pathological logit can never produce inf/NaN.
            # The running max only moves when a tile brings a larger logit, and
            # after the first couple of tiles it usually does not.  Rescaling
            # under a predicate keeps the D*nc accumulator multiply out of the
            # steady-state loop entirely; the offset sequence stays
            # non-decreasing either way, which is all exactness needs.
            if mnew > mrun:
                corr = cute.arch.exp2(mrun - mnew)
                for e in cutlass.range_constexpr(NO):
                    accO[e] = accO[e] * corr
                for e in cutlass.range_constexpr(NS):
                    lp[e] = lp[e] * corr
            mrun = mnew

            tmax = cutlass.Float32(_NEG)
            if (kbase + _KT - 1) <= qpos:
                # every key in this tile is causally visible, so the per-element
                # predicate -- the bulk of the softmax instruction count -- goes
                # away for all but the one tile that straddles the query
                for e in cutlass.range_constexpr(NS):
                    sv = accS[e] * scale2
                    tmax = cute.arch.fmax(tmax, sv)
                    pv = cute.arch.exp2(cute.arch.fmin(sv - mnew, 120.0))
                    sP_kg[tScS[e]] = pv.to(pdt)
                    lp[e] = lp[e] + pv
            else:
                for e in cutlass.range_constexpr(NS):
                    sv = (accS[e] * scale2) if (kbase + tScS[e][0]) <= qpos \
                        else cutlass.Float32(_NEG)
                    tmax = cute.arch.fmax(tmax, sv)
                    pv = cute.arch.exp2(cute.arch.fmin(sv - mnew, 120.0))
                    sP_kg[tScS[e]] = pv.to(pdt)
                    lp[e] = lp[e] + pv
            # A single integer redux replaces five dependent shuffle/fmax
            # pairs for every split factor.  The signed-key transform is
            # monotone for every finite f32, so the maximum is bit-identical.
            tbits = tmax.bitcast(cutlass.Int32)
            tkey = tbits ^ ((tbits >> 31) & cutlass.Int32(0x7FFFFFFF))
            tkey = cute.arch.warp_redux_sync(tkey, "max")
            tmax = (tkey ^ ((tkey >> 31) & cutlass.Int32(0x7FFFFFFF))
                    ).bitcast(cutlass.Float32)
            if lane == 0:
                sRed[(u % 2) * 4 + warp_idx] = tmax
            # sVb / sP were written with generic stores; publish them to the
            # async proxy that WGMMA uses to read SMEM before the barrier.
            cute.arch.fence_proxy("async.shared", space="cta")
            cute.arch.barrier()
            if tidx == 0:
                cute.arch.mbarrier_arrive(mbar + st + stage)

            # gemm 2: O^T += V^T . P
            warpgroup.fence()
            cute.gemm(pv_mma, accO, tOrV, tOrP, accO)
            warpgroup.commit_group()

        warpgroup.wait_group(0)

        # ---- epilogue -----------------------------------------------------
        for e in cutlass.range_constexpr(NS):
            sLK[tScS[e][1], tScS[e][0]] = lp[e]
        cute.arch.barrier()
        LF = _KT // LANES
        lfrag = cute.make_rmem_tensor(cute.make_layout(LF), cutlass.Float32)
        sLKv = cute.tiled_divide(sLK, (1, LF))
        cute.autovec_copy(sLKv[(0, None), lg, gcol], lfrag)
        psum = cutlass.Float32(0.0)
        for e in cutlass.range_constexpr(LF):
            psum += lfrag[e]
        o = 1
        while o < LANES:
            psum += cute.arch.shuffle_sync_bfly(psum, o)
            o *= 2

        for e in cutlass.range_constexpr(NO):
            sOf[tOcO[e][1], tOcO[e][0]] = accO[e]
        if gcol == 0:
            sL[grow] = psum
        cute.arch.barrier()

        sOfv = cute.tiled_divide(sOf, (1, CH))
        ofrag = cute.make_rmem_tensor(cute.make_layout((CH, NCK)),
                                      cutlass.Float32)
        for j in cutlass.range_constexpr(NCK):
            cute.autovec_copy(sOfv[(0, None), j * RSTEP + crow, ccol],
                              ofrag[(None, j)])

        if cutlass.const_expr(ns > 1):
            # A split that got no tiles publishes the neutral element instead of
            # an 8 KiB block of zeros: -inf as its running max makes the merge
            # weight it exactly zero, so its output block never has to be
            # written or read.  Splits with nothing to do are the common case
            # whenever the selected list is shorter than the split factor.
            slot = gidx * ns + spl
            if nloc > 0:
                # Normalize before compression: the resulting convex
                # combination of V values fits fp16 without numerator overflow.
                # Flat CH-element tiling, same trick as q/out: with the partial
                # buffer's row strides dynamic the compiler cannot prove the
                # (slot, row, col) slice contiguous.  A flat tensor keeps the
                # unit stride static, so the fp16 store remains vectorized.
                wOv = cute.tiled_divide(wO, (CH,))
                ohf = cute.make_rmem_tensor(cute.make_layout(CH), pdt)
                for j in cutlass.range_constexpr(NCK):
                    r = j * RSTEP + crow
                    ll = sL[r]
                    rl = cute.arch.rcp_approx(ll) if ll > 0.0 \
                        else cutlass.Float32(0.0)
                    for e in cutlass.range_constexpr(CH):
                        ohf[e] = (ofrag[(e, j)] * rl).to(pdt)
                    cute.autovec_copy(
                        ohf, wOv[(None,), (slot * nc + r) * CPR + ccol])
            if tidx < nc:
                wL[slot, tidx] = sL[tidx]
            if tidx == 0:
                wM[slot] = mrun if nloc > 0 else cutlass.Float32(_NEG)

            # ---- fused combine ------------------------------------------
            # The split and the combine are one kernel.  Every split
            # publishes its partial, then the last one to arrive at this
            # output tile's counter reads the whole group back -- still hot
            # in L2 -- and writes the final bf16 row.  That removes the
            # second pass's HBM round trip of the partials entirely.
            cute.arch.barrier()
            if tidx == 0:
                # The CTA barrier sequences every thread's partial stores
                # before this lane-0 release/acquire arrival operation.
                sDone[0] = cute.arch.atomic_add(
                    wC.iterator + gidx, cutlass.Int32(1),
                    sem="acq_rel", scope="gpu")
            cute.arch.barrier()
            if sDone[0] == ns - 1:
                if tidx == 0:
                    # every split of this tile has already counted itself, so
                    # the counter can be returned to zero for the next launch
                    wC[gidx] = cutlass.Int32(0)
                # The last arrival's acquire half makes every peer's released
                # partial stores visible to all threads after the CTA barrier.
                RPT = cutlass.const_expr(nc * CPR)
                s0 = gidx * ns
                wm = cute.make_rmem_tensor(cute.make_layout(ns), cutlass.Float32)
                for s in cutlass.range_constexpr(ns):
                    wm[s] = wM[s0 + s]
                mx = cutlass.Float32(_NEG)
                for s in cutlass.range_constexpr(ns):
                    mx = cute.arch.fmax(mx, wm[s])
                wOc = cute.tiled_divide(wO, (CH,))
                mOc = cute.tiled_divide(mO, (CH,))
                GS = cutlass.const_expr(self.mgs)
                NB = cutlass.const_expr(-(-ns // GS))
                fr = cute.make_rmem_tensor(cute.make_layout((CH, GS)), pdt)
                ww = cute.make_rmem_tensor(cute.make_layout(ns), cutlass.Float32)
                # Prefetch every per-split row sum before the wide partial
                # loads.  Their addresses are independent, so this removes a
                # second exposed dependent global round trip from each row.
                wls = cute.make_rmem_tensor(cute.make_layout((ns, NCK)),
                                            cutlass.Float32)
                for j in cutlass.range_constexpr(NCK):
                    for s in cutlass.range_constexpr(ns):
                        wls[(s, j)] = wL[s0 + s, j * RSTEP + crow]
                for j in cutlass.range_constexpr(NCK):
                    r = j * RSTEP + crow
                    den = cutlass.Float32(0.0)
                    for s in cutlass.range_constexpr(ns):
                        ww[s] = cute.arch.exp2(wm[s] - mx) * wls[(s, j)]
                        den += ww[s]
                    rlm = cute.arch.rcp_approx(den) if den > 0.0 \
                        else cutlass.Float32(0.0)
                    am = cute.make_rmem_tensor(cute.make_layout(CH),
                                               cutlass.Float32)
                    for e in cutlass.range_constexpr(CH):
                        am[e] = cutlass.Float32(0.0)
                    for bb in cutlass.range_constexpr(NB):
                        # a batch of partials is read into one register array
                        # so the loads issue independently instead of chaining
                        # through a single reused fragment
                        for s in cutlass.range_constexpr(GS):
                            if cutlass.const_expr(bb * GS + s < ns):
                                cute.autovec_copy(
                                    wOc[(None,), (s0 + bb * GS + s) * RPT
                                        + r * CPR + ccol],
                                    fr[(None, s)])
                        for s in cutlass.range_constexpr(GS):
                            if cutlass.const_expr(bb * GS + s < ns):
                                for e in cutlass.range_constexpr(CH):
                                    am[e] = am[e] + ww[bb * GS + s] * \
                                        fr[(e, s)].to(cutlass.Float32)
                    ob = cute.make_rmem_tensor(cute.make_layout(CH),
                                               cutlass.BFloat16)
                    for e in cutlass.range_constexpr(CH):
                        ob[e] = (am[e] * rlm).to(cutlass.BFloat16)
                    if (gb * nc + r) < gsz:
                        if cutlass.const_expr(self.wide_addr):
                            ooff64 = ((cutlass.Int64(qbase) + cutlass.Int64(r))
                                      * cutlass.Int64(D)
                                      + cutlass.Int64(ccol * CH))
                            cute.autovec_copy(
                                ob, cute.make_tensor(mO.iterator + ooff64,
                                                     cute.make_layout(CH)))
                        else:
                            cute.autovec_copy(
                                ob, mOc[(None,), (qbase + r) * CPR + ccol])
        else:
            mOc = cute.tiled_divide(mO, (CH,))
            obf = cute.make_rmem_tensor(cute.make_layout(CH), cutlass.BFloat16)
            for j in cutlass.range_constexpr(NCK):
                r = j * RSTEP + crow
                ll = sL[r]
                rl = cute.arch.rcp_approx(ll) if ll > 0.0 \
                    else cutlass.Float32(0.0)
                for e in cutlass.range_constexpr(CH):
                    obf[e] = (ofrag[(e, j)] * rl).to(cutlass.BFloat16)
                if (gb * nc + r) < gsz:
                    if cutlass.const_expr(self.wide_addr):
                        ooff64 = ((cutlass.Int64(qbase) + cutlass.Int64(r))
                                  * cutlass.Int64(D)
                                  + cutlass.Int64(ccol * CH))
                        cute.autovec_copy(
                            obf, cute.make_tensor(mO.iterator + ooff64,
                                                  cute.make_layout(CH)))
                    else:
                        cute.autovec_copy(
                            obf, mOc[(None,), (qbase + r) * CPR + ccol])



_COMPILED = {}
_WS = {}          # scratch buffers (memory only, never carries results across calls)


def _scratch(n, nc, d, device):
    """Split-K partial buffers.  Pure scratch: written before it is read on
    every call, never carries a result from one call to the next."""
    key = (device, nc, d)
    buf = _WS.get(key)
    if buf is None or buf[0].shape[0] < n * nc * d:
        wo = torch.zeros((n * nc * d,), dtype=torch.float16, device=device)
        wl = torch.empty((n, nc), dtype=torch.float32, device=device)
        wm = torch.empty((n,), dtype=torch.float32, device=device)
        # combine counters.  Zeroed once here; the last split of every
        # output tile returns its counter to zero, so the buffer is
        # always back at its initial state when a launch ends.
        wc = torch.zeros((n,), dtype=torch.int32, device=device)
        buf = (
            wo, wl, wm,
            from_dlpack(wo, assumed_align=16, use_32bit_stride=True
                        ).mark_layout_dynamic(leading_dim=0),
            from_dlpack(wl, assumed_align=16, use_32bit_stride=True
                        ).mark_layout_dynamic(leading_dim=1),
            from_dlpack(wm, assumed_align=16, use_32bit_stride=True
                        ).mark_layout_dynamic(leading_dim=0),
            from_dlpack(wc, assumed_align=16, use_32bit_stride=True
                        ).mark_layout_dynamic(leading_dim=0),
        )
        _WS[key] = buf
    return buf[3], buf[4], buf[5], buf[6]


def _smem_bytes(nc, d, stages):
    """Shared-memory footprint of one CTA at this TMA ring depth."""
    return (1024 + nc * d + nc * _KT * 2 + d * _KT * 2
            + 2 * stages * _KT * d)


def _split(nc, d, base, tile_cap):
    """Choose split fan-in from H100 residency, waves, and per-CTA work.

    Splitting trades shorter tile loops for extra prologues and one fused
    combine.  The cost below scores those physical terms against the resident
    CTA slots admitted by the two-stage ring.  It therefore handles the whole
    continuous shape range without thresholds fitted to sampled shapes.
    """
    resident = _NUM_SM * min(_MAX_RESIDENT,
                             _SMEM_PER_SM // _smem_bytes(nc, d, 2))
    saturated = _NUM_SM * _SATURATED_CTA_PER_SM
    best_n, best_c = 1, None
    for n in _SPLITS:
        grid = base * n
        if n > 1 and grid > saturated and base < saturated:
            break
        tiles = -(-tile_cap // n)
        if n > 1 and tiles < _MIN_SPLIT_TILES and grid >= _NUM_SM:
            continue
        cost = (-(-grid // resident) * (_PROLOGUE_TILES + tiles)
                + (_COMBINE_TILES if n > 1 else 0))
        if best_c is None or cost <= best_c:
            best_c, best_n = cost, n
    return best_n


def _plan(nc, d, grid, tiles_per_cta):
    """Resident-CTA count and ring depth, from occupancy and smem capacity.

    The launch's footprint decides how many CTAs share an SM, and that is the
    dominant term: this kernel's inner loop is barrier-heavy, so an SM needs
    several CTAs in flight to have anything to issue while one of them sits on
    a gather.  `ceil(grid / SMs)` CTAs per SM already holds the whole grid in
    one scheduling wave; residency is sized one above that so an SM still has
    another CTA to issue from while one of them drains, and every byte of
    capacity past that point buys ring depth instead, which is the only thing
    that raises a single CTA's gather concurrency.  Both settings compute
    identical results.

    One stage costs 2*_KT*d bytes of fp8 K+V staging.  The fixed part is the
    fp16 V operand (d*_KT*2), the P operand (nc*_KT*2), Q (nc*d) and the small
    index/reduction arrays.  A ring deeper than the number of tiles a CTA can
    ever see is pure footprint, so the work bounds it too.
    """
    ctas = max(2, min(_MAX_RESIDENT, 1 + -(-grid // _NUM_SM)))
    fixed = d * _KT * 2 + nc * _KT * 2 + nc * d + 2048
    per_stage = 2 * _KT * d
    stages = (min(_SMEM_PER_SM // ctas, _SMEM_MAX_CTA) - fixed) // per_stage
    # Short sparse-attention mainloops do not amortize a capacity-maximal
    # TMA prologue.  Keep one slot of capacity in reserve while retaining the
    # mandatory two-stage producer/consumer overlap.
    return max(2, min(_MAX_STAGES, stages, max(2, tiles_per_cta)) - 1)


def run(q, kv, q2k_indices, page_table, seqused_k, out):
    total_q, Hq, D = q.shape
    num_pages, Hkv, page_size, two_d = kv.shape
    B = seqused_k.numel()
    sq = total_q // B
    topk = q2k_indices.shape[2]
    G = Hq // Hkv
    # WGMMA N follows the GQA group: no N lane is left idle for the common
    # tp=8 (G=8) configuration, and no KV byte is fetched twice for G>8.
    nc = 8 if G <= 8 else (16 if G <= 16 else 32)
    ng = (G + nc - 1) // nc

    base = total_q * Hkv * ng
    # The page-table width bounds every token's selected list.  Convert pages
    # to 64-key tiles, then let the occupancy/wave cost model choose the split.
    tile_cap = (page_size // _KT) * min(topk, page_table.shape[1])
    nsplit = _split(nc, D, base, tile_cap)
    stages = _plan(nc, D, base * nsplit, -(-tile_cap // nsplit))

    def _pack32(t, ld):
        return from_dlpack(
            t, assumed_align=16, use_32bit_stride=True
        ).mark_layout_dynamic(leading_dim=ld)

    safe32_q = q.numel() < (1 << 31)
    mQ = from_dlpack(q.view(-1), assumed_align=16,
                     use_32bit_stride=safe32_q
                     ).mark_layout_dynamic(leading_dim=0)
    mO = from_dlpack(out.view(-1), assumed_align=16,
                     use_32bit_stride=safe32_q
                     ).mark_layout_dynamic(leading_dim=0)
    # KV cache offsets span the full production range, so retain 64-bit
    # descriptor strides even when a sampled allocation happens to be small.
    mKV = from_dlpack(kv.view(torch.uint8), assumed_align=16
                      ).mark_layout_dynamic(leading_dim=3)
    mKV.element_type = cutlass.Float8E4M3FN
    mIdx = _pack32(q2k_indices, 2)
    mPT = _pack32(page_table, 1)
    mSeq = _pack32(seqused_k, 0)

    wO, wL, wM, wC = _scratch(base * nsplit if nsplit > 1 else 1,
                              nc, D, q.device)

    stream = cuda.CUstream(torch.cuda.current_stream().cuda_stream)
    args = (mQ, mKV, mIdx, mPT, mSeq, mO, wO, wL, wM, wC,
            cutlass.Int32(sq), cutlass.Int32(G), cutlass.Int32(ng),
            cutlass.Int32(Hkv), cutlass.Int32(total_q), stream)

    wide_addr = not safe32_q
    key = (topk, nsplit, stages, nc, D, page_size, wide_addr)
    fn = _COMPILED.get(key)
    if fn is None:
        fn = cute.compile(
            MsaSparseDecode(
                topk, nsplit, stages, nc, D, page_size, wide_addr), *args)
        _COMPILED[key] = fn
    fn(*args)
