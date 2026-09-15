"""MSA proxy-score fp8 decode schedule for SM90 (Hopper).

Ported from the Kernel-Factory winner (2.70x geomean over the shipped Triton
decode on held-out shapes). WGMMA with descriptor prefetch; index-K is consumed
as fp8 e4m3 natively.

The compile cache keys on (Hq, sq, batch_fast) -- head count, query tokens per
sequence (1 / 2 / 4 for no-MTP / MTP1 / MTP3) and one schedule flag. No raw
shapes, so one compile serves every batch and context length.
"""

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync, warpgroup
from cutlass.cute.runtime import from_dlpack

PAGE = 128
HALF = 64
DIM = 128
NTHREADS = 128
FP8 = cutlass.Float8E4M3FN


@cute.kernel
def _msa_kernel(
    tma_atom_k: cute.CopyAtom,
    mK: cute.Tensor,
    mQ: cute.Tensor,
    mPage: cute.Tensor,
    mSeq: cute.Tensor,
    mOut: cute.Tensor,
    sK_layout: cute.ComposedLayout,
    sQ_layout: cute.ComposedLayout,
    tiled_mma: cute.TiledMma,
    nbatch: cutlass.Int32,
    nq: cutlass.Constexpr,
    Hq: cutlass.Constexpr,
    sq: cutlass.Constexpr,
    log2_hq: cutlass.Constexpr,
    TN: cutlass.Constexpr,
    batch_fast: cutlass.Constexpr,
):
    NG = cutlass.const_expr(TN // 8)
    NJ = cutlass.const_expr(1 if nq == 1 else 2)
    NV = cutlass.const_expr(NG * NJ)
    ACTIVE_GROUPS = cutlass.const_expr(4 if TN == 16 else (nq + 1) // 2)
    QT = cutlass.const_expr(TN * 8)

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        cpasync.prefetch_descriptor(tma_atom_k)

    tidx, _, _ = cute.arch.thread_idx()
    bx, by, _ = cute.arch.block_idx()
    if cutlass.const_expr(batch_fast):
        b, t = bx, by
    else:
        t, b = bx, by

    lane = tidx % 32
    warp = tidx // 32
    qrow = tidx // 8
    neg = -cutlass.Float32.inf

    smem = utils.SmemAllocator()
    sK = smem.allocate_tensor(
        FP8, sK_layout.outer, byte_alignment=1024, swizzle=sK_layout.inner
    )
    sQ = smem.allocate_tensor(
        FP8, sQ_layout.outer, byte_alignment=1024, swizzle=sQ_layout.inner
    )
    sRed = smem.allocate_tensor(
        cutlass.Float32, cute.make_layout((4, TN), stride=(TN, 1)), byte_alignment=16
    )
    mbar = smem.allocate_array(element_type=cutlass.Int64, num_elems=2)

    page = mPage[b, t]
    sk = mSeq[b]
    nb = (sk + 127) // 128

    if t < nb:
        if warp == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(mbar, 1)
                cute.arch.mbarrier_init(mbar + 1, 1)
        cute.arch.mbarrier_init_fence()

        tKsK, tKgK = cpasync.tma_partition(
            tma_atom_k,
            0,
            cute.make_layout(1),
            cute.group_modes(sK, 0, 2),
            cute.group_modes(mK, 0, 2),
        )
        if warp == 0:
            for h in cutlass.range_constexpr(2):
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar + h, HALF * DIM)
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, 2 * page + h)],
                    tKsK[(None, h)],
                    tma_bar_ptr=mbar + h,
                )

        cp_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), FP8)
        tv_cp = cute.make_tiled_copy_tv(
            cp_atom,
            cute.make_layout((TN, 8), stride=(8, 1)),
            cute.make_layout((1, 16), stride=(0, 1)),
        )
        thr_cp = tv_cp.get_slice(tidx)
        gQall = cute.make_tensor(
            mQ.iterator,
            cute.make_layout((TN, DIM, nbatch), stride=(DIM, 1, nq * DIM)),
        )
        if tidx < QT and qrow < nq:
            tQs = thr_cp.partition_S(gQall[(None, None, b)])
            frg = cute.make_fragment_like(tQs)
            cute.copy(cp_atom, tQs, frg)
            cute.copy(cp_atom, frg, thr_cp.partition_D(sQ[(None, None, 0)]))

        cute.arch.fence_view_async_shared()
        cute.arch.sync_threads()

        thr_mma = tiled_mma.get_slice(0)
        tCrA = tiled_mma.make_fragment_A(thr_mma.partition_A(sK))
        tCrB = tiled_mma.make_fragment_B(thr_mma.partition_B(sQ))
        acc0 = thr_mma.make_fragment_C(thr_mma.partition_shape_C((HALF, TN)))
        acc1 = thr_mma.make_fragment_C(thr_mma.partition_shape_C((HALF, TN)))
        rB = tCrB[(None, None, None, 0)]

        row_base = warp * 16 + lane // 4
        col_base = 2 * (lane % 4)
        vals = [neg] * NV
        for h in cutlass.range_constexpr(2):
            cute.arch.mbarrier_wait(mbar + h, 0)
            rA = tCrA[(None, None, None, h)]
            acc = acc0 if cutlass.const_expr(h == 0) else acc1
            warpgroup.fence()
            for kb in cutlass.range_constexpr(cute.size(rA, mode=[2])):
                tiled_mma.set(warpgroup.Field.ACCUMULATE, kb != 0)
                cute.gemm(tiled_mma, acc, rA[None, None, kb], rB[None, None, kb], acc)
            warpgroup.commit_group()
        warpgroup.wait_group(1)

        for h in cutlass.range_constexpr(2):
            if cutlass.const_expr(h == 1):
                warpgroup.wait_group(0)
            acc = acc0 if cutlass.const_expr(h == 0) else acc1
            if t == nb - 1:
                lim = sk - sq - 128 * (nb - 1)
                for gg in cutlass.range_constexpr(NG):
                    for j in cutlass.range_constexpr(NJ):
                        climit = lim + ((col_base + 8 * gg + j) >> log2_hq)
                        v = vals[NJ * gg + j]
                        if lane % 4 < ACTIVE_GROUPS:
                            for i in cutlass.range_constexpr(2):
                                r = row_base + 64 * h + 8 * i
                                if cutlass.const_expr(NG == 1):
                                    a = acc[((j, i), 0, 0)]
                                else:
                                    a = acc[((j, i, gg), 0, 0)]
                                v = cutlass.max(v, a if r <= climit else neg)
                        vals[NJ * gg + j] = v
            else:
                for gg in cutlass.range_constexpr(NG):
                    for j in cutlass.range_constexpr(NJ):
                        v = vals[NJ * gg + j]
                        if lane % 4 < ACTIVE_GROUPS:
                            for i in cutlass.range_constexpr(2):
                                if cutlass.const_expr(NG == 1):
                                    a = acc[((j, i), 0, 0)]
                                else:
                                    a = acc[((j, i, gg), 0, 0)]
                                v = cutlass.max(v, a)
                        vals[NJ * gg + j] = v

        for off in cutlass.range_constexpr(3):
            sh = 4 << off
            for kk in cutlass.range_constexpr(NV):
                vals[kk] = cutlass.max(
                    vals[kk], cute.arch.shuffle_sync_bfly(vals[kk], sh)
                )

        if lane < ACTIVE_GROUPS:
            for gg in cutlass.range_constexpr(NG):
                for j in cutlass.range_constexpr(NJ):
                    sRed[warp, 8 * gg + 2 * lane + j] = vals[NJ * gg + j]
        cute.arch.sync_threads()

        if tidx < nq:
            o = cutlass.max(
                cutlass.max(sRed[0, tidx], sRed[1, tidx]),
                cutlass.max(sRed[2, tidx], sRed[3, tidx]),
            )
            mOut[tidx % Hq, t, b * sq + tidx // Hq] = o
    else:
        if tidx < nq:
            mOut[tidx % Hq, t, b * sq + tidx // Hq] = neg


@cute.jit
def _msa_host(
    stream,
    mK: cute.Tensor,
    mQ: cute.Tensor,
    mPage: cute.Tensor,
    mSeq: cute.Tensor,
    mOut: cute.Tensor,
    nq: cutlass.Constexpr,
    Hq: cutlass.Constexpr,
    sq: cutlass.Constexpr,
    log2_hq: cutlass.Constexpr,
    TN: cutlass.Constexpr,
    batch_fast: cutlass.Constexpr,
):
    tiled_mma = sm90_utils.make_trivial_tiled_mma(
        FP8,
        FP8,
        cute.nvgpu.OperandMajorMode.K,
        cute.nvgpu.OperandMajorMode.K,
        cutlass.Float32,
        (1, 1, 1),
        (64, TN),
    )
    tile_mnk = (HALF, TN, DIM)
    sK_layout = sm90_utils.make_smem_layout_a(
        utils.LayoutEnum.ROW_MAJOR, tile_mnk, FP8, 2
    )
    sQ_layout = sm90_utils.make_smem_layout_b(
        utils.LayoutEnum.ROW_MAJOR, tile_mnk, FP8, 1
    )
    one_stage_k = cute.slice_(sK_layout, (None, None, 0))

    npages = cute.size(mK, mode=[0])
    mK3 = cute.make_tensor(
        mK.iterator,
        cute.make_layout((HALF, DIM, 2 * npages), stride=(DIM, 1, HALF * DIM)),
    )
    tma_atom_k, mK_tma = cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(),
        mK3,
        one_stage_k,
        (HALF, DIM),
    )

    nbatch = cute.size(mPage, mode=[0])
    kernel = _msa_kernel(
        tma_atom_k,
        mK_tma,
        mQ,
        mPage,
        mSeq,
        mOut,
        sK_layout,
        sQ_layout,
        tiled_mma,
        nbatch,
        nq,
        Hq,
        sq,
        log2_hq,
        TN,
        batch_fast,
    )
    if cutlass.const_expr(batch_fast):
        kernel.launch(
            grid=[nbatch, cute.size(mPage, mode=[1]), 1],
            block=[NTHREADS, 1, 1],
            stream=stream,
        )
    else:
        kernel.launch(
            grid=[cute.size(mPage, mode=[1]), nbatch, 1],
            block=[NTHREADS, 1, 1],
            stream=stream,
        )



# --- CUDA-graph stream plumbing -------------------------------------------
# Without an explicit stream these launches go to the CuTe default stream, so
# torch.cuda.graph records NOTHING: the kernel runs eagerly during capture and
# replay is a no-op (verified: output written at capture, all-NaN after replay).
# vLLM captures decode graphs, so the launch must honour the current stream.
# Same pattern the sparse kernels already use.
import torch
import cuda.bindings.driver as _kf_cuda
_KF_STREAMS = {}


def _kf_stream():
    h = torch.cuda.current_stream().cuda_stream
    s = _KF_STREAMS.get(h)
    if s is None:
        s = _kf_cuda.CUstream(h)
        _KF_STREAMS[h] = s
    return s


_CACHE = {}


def run(q, k, cu_seqlens_q, page_table, seqused_k, max_score):
    total_q, Hq, D = q.shape
    num_pages = k.shape[0]
    B, max_k_tiles = page_table.shape
    sq = total_q // B
    nq = Hq * sq
    log2_hq = {1: 0, 2: 1, 4: 2}[Hq]
    TN = 8 if nq <= 8 else 16
    batch_fast = num_pages * 20 >= B * max_k_tiles * 17

    key = (Hq, sq, batch_fast)
    fn = _CACHE.get(key)
    if fn is None:
        mK = from_dlpack(k, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=3
        )
        mQ = from_dlpack(q, assumed_align=16, enable_tvm_ffi=True).mark_layout_dynamic(
            leading_dim=2
        )
        mPage = from_dlpack(
            page_table, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=1)
        mSeq = from_dlpack(
            seqused_k, assumed_align=4, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=0)
        mOut = from_dlpack(
            max_score, assumed_align=16, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=2)
        fn = cute.compile(
            _msa_host,
            _kf_stream(),
            mK,
            mQ,
            mPage,
            mSeq,
            mOut,
            nq,
            Hq,
            sq,
            log2_hq,
            TN,
            batch_fast,
            options="--enable-tvm-ffi",
        )
        _CACHE[key] = fn
    fn(_kf_stream(), k, q, page_table, seqused_k, max_score)
