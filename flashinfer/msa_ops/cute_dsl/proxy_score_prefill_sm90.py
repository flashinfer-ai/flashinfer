"""MSA proxy-score prefill schedule for SM90 (Hopper).

Two consumer warpgroups run m64n128k32 FP8 WGMMA into FP16 accumulators while a
producer warp streams paged K tiles through a TMA pipeline; the row-max epilogue
of sub-tile s overlaps sub-tile s+1 on the tensor core.

Shapes are dynamic: total_q, max_k_tiles, batch_size and the K-split are runtime
Int32 arguments, so one compile serves every shape. Only the two schedule
selections that change emitted code -- use_tma_q and fold -- are compile-time,
matching the cache-key granularity the SM12x backends use.
"""

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.cute.nvgpu.cpasync as cpasync
from cutlass.cute.nvgpu import warpgroup, OperandMajorMode
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# accumulator layout helpers (WGMMA C fragment -> (M, N) view)
# ---------------------------------------------------------------------------
def _layout_separate(thr, src, ref):
    lt = cute.make_layout(())
    ge = cute.make_layout(())
    for k, v in enumerate(ref):
        if cutlass.const_expr(v < thr):
            lt = cute.append(lt, src[k])
        else:
            ge = cute.append(ge, src[k])
    r = None
    if cutlass.const_expr(cute.rank(lt) == 1):
        r = cute.append(lt, ge)
    else:
        r = cute.append(cute.append(cute.make_layout(()), lt), ge)
    return r


def layout_acc_mn(tiled_mma, acc):
    separated = _layout_separate(
        tiled_mma.shape_mnk[0], acc[0], tiled_mma.tv_layout_C.stride[1]
    )
    V_M = separated[0]
    V_N = separated[1]
    if cutlass.const_expr(cute.rank(V_M) == 1):
        V_M1 = cute.append(V_M, acc[1])
    else:
        V_M1 = cute.append(cute.append(cute.make_layout(()), V_M), acc[1])
    if cutlass.const_expr(cute.rank(V_N) == 1):
        V_N1 = cute.append(V_N, acc[2])
    else:
        V_N1 = cute.append(cute.append(cute.make_layout(()), V_N), acc[2])
    if cutlass.const_expr(cute.rank(V_M1) == 1):
        r = cute.append(V_M1, V_N1)
    else:
        r = cute.append(cute.append(cute.make_layout(()), V_M1), V_N1)
    return r


def reduction_target_n(tiled_mma):
    separated = _layout_separate(
        tiled_mma.shape_mnk[0],
        cute.make_layout(tiled_mma.tv_layout_C.shape[0]),
        tiled_mma.tv_layout_C.stride[0],
    )
    return separated[1]


# ---------------------------------------------------------------------------
# mainloop helpers (module level: CuTe DSL rejects closures over traced values)
# ---------------------------------------------------------------------------
@cute.jit
def mma_issue(
    mma_op,
    acc,
    tCrQ,
    tCrK,
    sub: cutlass.Constexpr,
    kslice,
    nkb: cutlass.Constexpr,
):
    cute.nvgpu.warpgroup.fence()
    mma_atom = cute.make_mma_atom(mma_op)
    mma_atom.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
    for kb in cutlass.range_constexpr(nkb):
        cute.gemm(
            mma_atom,
            acc,
            tCrQ[(None, None, kb, sub)],
            tCrK[(None, None, kb, kslice)],
            acc,
        )
        mma_atom.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
    cute.nvgpu.warpgroup.commit_group()


@cute.jit
def mma_epilogue(
    acc,
    tCcS,
    acc_mn_layout: cutlass.Constexpr,
    n_rows: cutlass.Constexpr,
    red_target: cutlass.Constexpr,
    red_rank: cutlass.Constexpr,
    s_max,
    own,
    rows,
    mO,
    o_row_base,
    sub: cutlass.Constexpr,
    t,
    delta,
    kmax,
    page: cutlass.Constexpr,
    total_q,
    do_mask: cutlass.Constexpr = True,
):
    if cutlass.const_expr(do_mask):
        if cutlass.min(delta, kmax) < page - 1:
            for i in cutlass.range_constexpr(cute.size(acc)):
                cq, ck = tCcS[i]
                if ck > cutlass.min(cq + delta, kmax):
                    acc[i] = cutlass.Float16(float("-inf"))
    acc_mn = cute.make_tensor(acc.iterator, acc_mn_layout)
    for i in cutlass.range_constexpr(n_rows):
        s_max[i] = cutlass.Float32(
            acc_mn[i, None]
            .load()
            .reduce(cute.ReductionOp.MAX, cutlass.Float16(float("-inf")), 0)
        )
    for i in cutlass.range_constexpr(n_rows):
        for r in cutlass.range_constexpr(red_rank):
            s_max[i] = cute.arch.warp_reduction_max(
                s_max[i], threads_in_group=red_target.shape[r]
            )
    if own:
        gO = cute.make_tensor(
            mO.iterator + (o_row_base + 128 * sub + t * total_q),
            cute.make_layout(128),
        )
        for i in cutlass.range_constexpr(n_rows):
            gO[rows[i]] = s_max[i]


class MsaProxyScore:
    """Per-KV-block max of causally masked QK^T logits, chunked-prefill layout.

    A CTA owns ``128 * nsub`` query rows of one head and a contiguous run of
    KV pages.  Two consumer warpgroups run m64n128k32 FP8 WGMMA into FP16
    accumulators; one producer warp streams the paged K tiles through a TMA
    pipeline.  The ``nsub`` sub-tiles are committed as separate WGMMA groups so
    the row-max epilogue of sub-tile ``s`` runs while sub-tile ``s+1`` is still
    on the tensor core.
    """

    def __init__(
        self,
        nsub=2,
        nstage=4,
        page=128,
        head_dim=128,
        hq=1,
        use_tma_q=True,
        fold=0,
    ):
        self.hq = hq
        self.use_tma_q = use_tma_q
        self.nsub = nsub
        self.blk_m = 128 * nsub
        # Pair long and short causal query tiles on each SM only when their
        # spread is material relative to the KV range.  This is a compile-time
        # launch specialization, so the common long-K path keeps its natural
        # mapping with no device-side branch or extra arithmetic.
        self.fold = fold
        self.nstage = nstage
        self.page = page
        self.d = head_dim
        self.n_mma_wg = 2
        self.n_mma_threads = 128 * self.n_mma_wg
        self.n_threads = self.n_mma_threads + 32
        self.producer_warp = self.n_mma_threads // 32
        self.acc_dtype = cutlass.Float16
        self.ab_dtype = cutlass.Float8E4M3FN
        self.tile_mnk = (128, page, head_dim)

    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self, stream, mQ, mK4, mO, mCuQ, mPT, mSeqK, mPfx,
        total_q: cutlass.Int32, nkt: cutlass.Int32, batch_size: cutlass.Int32,
        nkchunk: cutlass.Int32, n_mtiles: cutlass.Int32,
    ):
        # rebuild K as (page, d, num_pages) with STATIC tile modes (TMA needs it)
        npages = mK4.shape[0]
        mKp = cute.make_tensor(
            mK4.iterator,
            cute.make_layout(
                (self.page, self.d, npages), stride=(self.d, 1, self.page * self.d)
            ),
        )
        qblocks = (total_q + 127) // 128
        mQp = cute.make_tensor(
            mQ.iterator,
            cute.make_layout(
                (128, self.d, qblocks, self.hq),
                stride=(self.hq * self.d, 1, 128 * self.hq * self.d, self.d),
            ),
        )
        tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.ab_dtype,
            self.ab_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            self.acc_dtype,
            (self.n_mma_wg, 1, 1),
            tiler_mn=(64, self.page),
        )

        # nsub independent 128-row A tiles live in the "stage" mode
        q_smem_layout = sm90_utils.make_smem_layout_a(
            utils.LayoutEnum.ROW_MAJOR, self.tile_mnk, self.ab_dtype, self.nsub
        )
        k_smem_layout_staged = sm90_utils.make_smem_layout_b(
            utils.LayoutEnum.ROW_MAJOR, self.tile_mnk, self.ab_dtype, self.nstage
        )
        q_smem_layout_one = cute.slice_(q_smem_layout, (None, None, 0))
        k_smem_layout_one = cute.slice_(k_smem_layout_staged, (None, None, 0))

        tma_atom_q, mQ_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mQp,
            q_smem_layout_one,
            (128, self.d),
        )
        tma_atom_k, mK_tma = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            mKp,
            k_smem_layout_one,
            (self.page, self.d),
        )

        hq = self.hq
        B = batch_size

        self.kernel(
            tma_atom_q,
            mQ_tma,
            tma_atom_k,
            mK_tma,
            mQ,
            mO,
            mCuQ,
            mPT,
            mSeqK,
            mPfx,
            total_q,
            nkt,
            nkchunk,
            n_mtiles,
            tiled_mma,
            q_smem_layout,
            k_smem_layout_staged,
        ).launch(
            grid=[hq * n_mtiles, nkchunk, B],
            block=[self.n_threads, 1, 1],
            min_blocks_per_mp=2,
            stream=stream,
        )

    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tma_atom_q: cute.CopyAtom,
        mQ_tma: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_tma: cute.Tensor,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mCuQ: cute.Tensor,
        mPT: cute.Tensor,
        mSeqK: cute.Tensor,
        mPfx: cute.Tensor,
        total_q: cutlass.Int32,
        nkt: cutlass.Int32,
        nkchunk: cutlass.Int32,
        n_mtiles: cutlass.Int32,
        tiled_mma: cute.TiledMma,
        q_smem_layout: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        bx, by, bz = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == self.producer_warp:
            if cutlass.const_expr(self.use_tma_q):
                cpasync.prefetch_descriptor(tma_atom_q)
            cpasync.prefetch_descriptor(tma_atom_k)

        # ---------------- work decomposition ----------------
        b = bz
        byy = by
        h = bx % self.hq
        mt = bx // self.hq

        # The H100 launch is sized for two resident CTAs per SM.  In blocks of
        # 132 linear CTA ids, reverse every other block and make the query tile
        # the slow axis.  Co-resident CTAs then receive complementary causal
        # tile counts instead of two similarly long (or short) traversals.
        if cutlass.const_expr(self.fold == 1):
            gx = self.hq * n_mtiles
            nlin = gx * nkchunk
            lid = bx + by * gx
            blk = lid // 132
            hi = (blk + 1) * 132
            if hi > nlin:
                hi = nlin
            kid = lid
            if blk % 2 == 1:
                kid = hi - 1 - (lid - blk * 132)
            byy = kid % nkchunk
            xid = kid // nkchunk
            h = xid % self.hq
            mt = xid // self.hq

        # Speculative first page id: with the round-robin KV split the CTA's
        # first tile index is byy, so this load is independent of every other
        # global read and collapses the start-up chain to one round trip.
        pg_first = mPT[b, byy]

        qlo = mCuQ[b]
        sqb = mCuQ[b + 1] - qlo
        sk = mSeqK[b]
        pfx = mPfx[b]

        m0 = mt * self.blk_m
        lim = sqb - self.blk_m
        if m0 > lim:
            m0 = lim
        if m0 < 0:
            m0 = cutlass.Int32(0)

        nb = (sk + (self.page - 1)) // self.page

        qmax = m0 + self.blk_m - 1
        if qmax > sqb - 1:
            qmax = sqb - 1
        qmax = qmax + pfx
        t_lim = qmax // self.page + 1
        if t_lim > nb:
            t_lim = nb

        # Live range [0, t_lim) handed out round-robin with stride nkchunk, so a
        # CTA's first KV tile index is byy: the page-id load no longer waits on
        # seqused_k / prefix_lens and one cold global round trip drops off the
        # CTA start-up critical path.
        n_comp = (t_lim - byy + nkchunk - 1) // nkchunk
        if n_comp < 0:
            n_comp = cutlass.Int32(0)

        # dead (-inf) range [t_lim, nkt), same stride
        n_dead = (nkt - t_lim - byy + nkchunk - 1) // nkchunk
        if n_dead < 0:
            n_dead = cutlass.Int32(0)

        # ---------------- shared memory ----------------
        @cute.struct
        class SharedStorage:
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.ab_dtype, cute.cosize(q_smem_layout)], 1024
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[
                    self.ab_dtype, cute.cosize(k_smem_layout_staged)
                ],
                1024,
            ]
            mbar: cute.struct.MemRange[cutlass.Int64, self.nstage * 2]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tx_bytes = cute.size_in_bytes(
            self.ab_dtype, cute.slice_(k_smem_layout_staged, (None, None, 0))
        )
        tx_bytes_q = cute.size_in_bytes(
            self.ab_dtype, cute.slice_(q_smem_layout, (None, None, 0))
        )
        kv_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar.data_ptr(),
            num_stages=self.nstage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Warp, self.n_mma_threads // 32
            ),
            tx_count=tx_bytes,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
            tidx=tidx,
            enable_multicast_signaling=True,
        )

        sQ = storage.sQ.get_tensor(q_smem_layout.outer, swizzle=q_smem_layout.inner)
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )

        # ---------------- producer warp ----------------
        if warp_idx == self.producer_warp:
            if cutlass.const_expr(not self.use_tma_q):
                cute.arch.barrier_arrive(
                    barrier_id=1, number_of_threads=self.n_threads, aligned=False
                )
            tQsQ, tQgQ = cpasync.tma_partition(
                tma_atom_q,
                0,
                cute.make_layout(1),
                cute.group_modes(sQ, 0, 2),
                cute.group_modes(mQ_tma, 0, 2),
            )
            tKsK, tKgK = cpasync.tma_partition(
                tma_atom_k,
                0,
                cute.make_layout(1),
                cute.group_modes(sK, 0, 2),
                cute.group_modes(mK_tma, 0, 2),
            )
            prod_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.nstage
            )
            pg_next = pg_first
            if n_comp > 0:
                page = cute.arch.make_warp_uniform(pg_first)
                if n_comp > 1:
                    pg_next = mPT[b, byy + nkchunk]
                kv_pipeline.producer_acquire(prod_state)
                if cutlass.const_expr(self.use_tma_q):
                    first_barrier = kv_pipeline.producer_get_barrier(prod_state)
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_expect_tx(first_barrier, tx_bytes_q)
                    qtile = (qlo + m0) // 128
                    cute.copy(
                        tma_atom_q,
                        tQgQ[(None, qtile, h)],
                        tQsQ[(None, 0)],
                        tma_bar_ptr=first_barrier,
                    )
                cute.copy(
                    tma_atom_k,
                    tKgK[(None, page)],
                    tKsK[(None, prod_state.index)],
                    tma_bar_ptr=kv_pipeline.producer_get_barrier(prod_state),
                )
                kv_pipeline.producer_commit(prod_state)
                prod_state.advance()
                for it in cutlass.range(1, n_comp, unroll=1):
                    page = cute.arch.make_warp_uniform(pg_next)
                    if it + 1 < n_comp:
                        pg_next = mPT[b, byy + (it + 1) * nkchunk]
                    kv_pipeline.producer_acquire(prod_state)
                    cute.copy(
                        tma_atom_k,
                        tKgK[(None, page)],
                        tKsK[(None, prod_state.index)],
                        tma_bar_ptr=kv_pipeline.producer_get_barrier(prod_state),
                    )
                    kv_pipeline.producer_commit(prod_state)
                    prod_state.advance()
        else:
            if cutlass.const_expr(not self.use_tma_q):
                if n_comp > 0:
                    copy_atom_q = cute.make_copy_atom(
                        cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                        self.ab_dtype,
                        num_bits_per_copy=128,
                    )
                    thr_layout = cute.make_ordered_layout(
                        (self.n_mma_threads // 8, 8), order=(1, 0)
                    )
                    val_layout = cute.make_ordered_layout(
                        (128 // (self.n_mma_threads // 8), 16), order=(1, 0)
                    )
                    tiled_copy_q = cute.make_tiled_copy_tv(
                        copy_atom_q, thr_layout, val_layout
                    )
                    thr_copy_q = tiled_copy_q.get_slice(tidx)
                    for s in cutlass.range_constexpr(self.nsub):
                        q_off = ((qlo + m0 + 128 * s) * (self.hq * self.d)
                                 + h * self.d)
                        gQ = cute.make_tensor(
                            (mQ.iterator + q_off).align(16),
                            cute.make_layout(
                                (128, self.d), stride=(self.hq * self.d, 1)
                            ),
                        )
                        tQgQ = thr_copy_q.partition_S(gQ)
                        tQsQ = thr_copy_q.partition_D(sQ[(None, None, s)])
                        cute.copy(thr_copy_q, tQgQ, tQsQ)
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                cute.arch.barrier(barrier_id=1, number_of_threads=self.n_threads)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsQ = thr_mma.partition_A(sQ)
            tCsK = thr_mma.partition_B(sK)
            tCrQ = tiled_mma.make_fragment_A(tCsQ)
            tCrK = tiled_mma.make_fragment_B(tCsK)

            acc_shape = thr_mma.partition_shape_C((128, self.page))
            accs = [thr_mma.make_fragment_C(acc_shape) for _ in range(self.nsub)]
            acc0 = accs[0]

            cS = cute.make_identity_tensor((128, self.page))
            tCcS = thr_mma.partition_C(cS)

            acc_mn_layout = layout_acc_mn(tiled_mma, acc0.layout)
            n_rows = cute.size(acc_mn_layout, mode=[0])
            cS_mn = cute.make_tensor(
                tCcS.iterator, layout_acc_mn(tiled_mma, tCcS.layout)
            )
            red_target = reduction_target_n(tiled_mma)
            red_rank = cute.rank(red_target)

            s_max = cute.make_rmem_tensor(cute.make_layout(n_rows), cutlass.Float32)

            own = cS_mn[0, 0][1] == 0
            rows = cute.make_rmem_tensor(cute.make_layout(n_rows), cutlass.Int32)
            for i in cutlass.range_constexpr(n_rows):
                rows[i] = cS_mn[i, 0][0]

            o_row_base = (h * nkt) * total_q + qlo + m0
            nkb = cute.size(tCrQ, mode=[2])

            read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.nstage
            )
            rel_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.nstage
            )

            mma_op = tiled_mma.op
            acc1 = thr_mma.make_fragment_C(acc_shape)
            if n_comp > 0:
                # Software-pipelined mainloop: the WGMMA group for tile i+1 is
                # already committed when tile i's row-max epilogue runs, so the
                # reduction/store work hides under the tensor core instead of
                # parking the whole warpgroup on wgmma.wait_group.  Two FP16
                # accumulators alternate, so the loop is unrolled by two to keep
                # the accumulator choice a compile-time constant.
                kv_pipeline.consumer_wait(read_state)
                mma_issue(mma_op, acc0, tCrQ, tCrK, 0, read_state.index, nkb)
                read_state.advance()

                # A tile needs causal masking only when
                #   t*page > m0 + pfx - (page-1)   or   t >= nb,
                # and the CTA walks t upward, so at most the final two tiles of
                # any CTA can be masked.  Everything before that runs a
                # branch-free epilogue: the per-element (cq, ck) coordinate
                # materialisation and the 64-way predicate loop disappear from
                # the hot path entirely (~4% of issued instructions).
                npair = (n_comp - 1) // 2
                npair_safe = (n_comp - 2) // 2
                if npair_safe < 0:
                    npair_safe = cutlass.Int32(0)
                if npair_safe > npair:
                    npair_safe = npair
                for j in cutlass.range(npair_safe, unroll=1):
                    t0 = byy + 2 * j * nkchunk
                    k0 = t0 * self.page
                    kv_pipeline.consumer_wait(read_state)
                    mma_issue(mma_op, acc1, tCrQ, tCrK, 0, read_state.index, nkb)
                    read_state.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    peek_next = kv_pipeline.consumer_try_wait(read_state)
                    mma_epilogue(
                        acc0, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, t0,
                        m0 + pfx - k0, sk - k0 - 1, self.page, total_q, False,
                    )
                    t1 = t0 + nkchunk
                    k1 = t1 * self.page
                    kv_pipeline.consumer_wait(read_state, peek_next)
                    mma_issue(mma_op, acc0, tCrQ, tCrK, 0, read_state.index, nkb)
                    read_state.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    mma_epilogue(
                        acc1, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, t1,
                        m0 + pfx - k1, sk - k1 - 1, self.page, total_q, False,
                    )

                for j in cutlass.range(npair_safe, npair, unroll=1):
                    t0 = byy + 2 * j * nkchunk
                    k0 = t0 * self.page
                    kv_pipeline.consumer_wait(read_state)
                    mma_issue(mma_op, acc1, tCrQ, tCrK, 0, read_state.index, nkb)
                    read_state.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    peek_next = kv_pipeline.consumer_try_wait(read_state)
                    mma_epilogue(
                        acc0, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, t0,
                        m0 + pfx - k0, sk - k0 - 1, self.page, total_q,
                    )
                    t1 = t0 + nkchunk
                    k1 = t1 * self.page
                    kv_pipeline.consumer_wait(read_state, peek_next)
                    mma_issue(mma_op, acc0, tCrQ, tCrK, 0, read_state.index, nkb)
                    read_state.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    mma_epilogue(
                        acc1, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, t1,
                        m0 + pfx - k1, sk - k1 - 1, self.page, total_q,
                    )

                tt = byy + 2 * npair * nkchunk
                kt = tt * self.page
                if n_comp - 2 * npair == 2:
                    kv_pipeline.consumer_wait(read_state)
                    mma_issue(mma_op, acc1, tCrQ, tCrK, 0, read_state.index, nkb)
                    read_state.advance()
                    cute.nvgpu.warpgroup.wait_group(1)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    mma_epilogue(
                        acc0, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, tt,
                        m0 + pfx - kt, sk - kt - 1, self.page, total_q,
                    )
                    tt2 = tt + nkchunk
                    kt2 = tt2 * self.page
                    cute.nvgpu.warpgroup.wait_group(0)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    mma_epilogue(
                        acc1, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, tt2,
                        m0 + pfx - kt2, sk - kt2 - 1, self.page, total_q,
                    )
                else:
                    cute.nvgpu.warpgroup.wait_group(0)
                    kv_pipeline.consumer_release(rel_state)
                    rel_state.advance()
                    mma_epilogue(
                        acc0, tCcS, acc_mn_layout, n_rows, red_target, red_rank,
                        s_max, own, rows, mO, o_row_base, 0, tt,
                        m0 + pfx - kt, sk - kt - 1, self.page, total_q,
                    )

            # ---- -inf tail (blocks past the sequence / fully masked) ----
            for it in cutlass.range(n_dead, unroll=1):
                t = t_lim + byy + it * nkchunk
                if tidx < self.blk_m:
                    gO = cute.make_tensor(
                        mO.iterator + (o_row_base + t * total_q),
                        cute.make_layout(self.blk_m),
                    )
                    gO[tidx] = -cutlass.Float32.inf



def select_split_k(total_q, hq, batch_size, nkt):
    n_mtiles = (total_q // batch_size + 127) // 128
    base_ctas = n_mtiles * hq * batch_size
    best_cost = 0x7FFFFFFF
    best_nk = 1
    for cand in [
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 20, 24, 32,
        40, 48, 64, 96, 128, -1, -2, -3, -4, -5, -6, -8, -10, -12,
        -16, -20, -24, -32, -40, -48,
    ]:
        nk = cand if cand > 0 else (-cand * 132) // base_ctas
        nk = max(1, min(nk, nkt))
        n_ctas = base_ctas * nk
        cost = ((n_ctas + 131) // 132) * ((nkt + nk - 1) // nk + 1)
        if n_ctas <= 132:
            cost *= 64
        if cost < best_cost:
            best_cost, best_nk = cost, nk
    return best_nk


