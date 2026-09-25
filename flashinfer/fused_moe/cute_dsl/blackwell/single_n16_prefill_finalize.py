"""Single-CTA feature128/token16 prefill G2. Minimum architecture SM100.

Private SM103/T512 selection preserves native quantized intermediates, routing,
caller workspace/stream and per-contribution BF16 weighted scatter. Two groups
reuse an eight-row shared bridge under the SM103 shared-memory limit.
"""

from typing import Optional

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from .utils import (
    UnalignedNamedBarrier,
    blk_reduce_bf16,
    blk_copy,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)


class SingleN16PrefillFinalizeKernel:
    def __init__(self, write_expanded_weighted=False):
        self.write_expanded_weighted = write_expanded_weighted
        self.enable_skinny_finalize = False
        self.enable_single_n16_prefill = True
        self.enable_contiguous_b_tma = True
        self.b_tma_bytes = 8192
        self.expected_shared_storage_bytes = 231424
        self.enable_narrow_a = False
        self.enable_sparse_narrow_a = False
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tile = (128, 16, 512)
        self.scale_tile = (128, 16, 128)
        self.num_ab_stage, self.num_acc_stage, self.num_c_stage = 3, 2, 1
        self.num_tmem_alloc_cols, self.threads_wo_sched = 128, 480
        self.threads_per_cta = 512
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=512)
        self.epilog_sync_barrier = UnalignedNamedBarrier(barrier_id=2, num_threads=128)
        self.tma_bytes = 32768
        self.num_tma_load_bytes = self.tma_bytes

    @cute.jit
    def scale_layout(self, rows, k, groups):
        # Exact selected finalize six-dimensional physical UE8M0 layout.
        return cute.make_ordered_layout(
            (32, 4, rows // 128, 4, k // 128, groups), order=(2, 1, 4, 0, 3, 5)
        )

    @cute.jit
    def epilogue_partition(self, acc_mn, tid):
        tc = tcgen05.make_tmem_copy(
            cute.make_copy_atom(
                tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32
            ),
            acc_mn,
        )
        thr = tc.get_slice(tid)
        coords = thr.partition_D(cute.make_identity_tensor((128, 16)))
        return tc, thr.partition_S(acc_mn), coords

    @cute.jit
    def output_shared_layout(self):
        return cute.make_layout((8, 128), stride=(128, 1))

    @cute.jit
    def scale_copy(self, shared, tensor):
        shared_compact = cute.filter_zeros(shared)
        tensor_compact = cute.filter_zeros(tensor)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        copy = tcgen05.make_s2t_copy(atom, tensor_compact)
        part = copy.get_slice(0)
        source = tcgen05.get_s2t_smem_desc_tensor(
            copy, part.partition_S(shared_compact)
        )
        return copy, source, part.partition_D(tensor_compact)

    @cute.jit
    def weight_scale_tma(self, pointer, features, k, experts):
        # Raw UE8M0 bytes, paired as Int16 without conversion. Four K128
        # slices form exactly the old contiguous 2048-byte warp12 transfer.
        source = cute.make_tensor(
            cute.recast_ptr(pointer, dtype=cutlass.Int16),
            cute.make_layout(
                (256, k // 128, features // 128, experts),
                stride=(1, 256, 256 * (k // 128), 256 * (k // 128) * (features // 128)),
            ),
        )
        destination = cute.make_layout((256, 4, 1, 1), stride=(1, 256, 1024, 1024))
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            source,
            destination,
            (256, 4, 1, 1),
        )

    @cute.jit
    def weight_scale_partition(self, atom, source, shared):
        destination = cute.make_tensor(
            cute.recast_ptr(shared.iterator, dtype=cutlass.Int16),
            cute.make_layout((256, 4, 1, 1, 3), stride=(1, 256, 1024, 1024, 1024)),
        )
        tiles = cute.local_tile(source, (256, 4, 1, 1), (None, None, None, None))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(destination, 0, 4),
            cute.group_modes(tiles, 0, 4),
        )

    @cute.jit
    def clamped_b_tma(self, pointer, k, swizzle):
        # Existing native intermediate bytes; 4D coordinates clamp only the
        # selected N16 sub-tile without reading another expert's padded rows.
        source = cute.make_tensor(
            pointer,
            cute.make_layout(
                (k, 16, 1 << 31, 1 << 31), stride=(1, k, (1 << 35) - k, k)
            ),
        )
        shared = cute.make_composed_layout(
            swizzle, 0, cute.make_layout((128, 16, 1, 1))
        )
        return cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            source,
            shared,
            (128, 16, 1, 1),
        )

    @cute.jit
    def clamped_b_partition(self, atom, mapped, tma_shared, row_base, limit):
        valid = cutlass.min(
            cutlass.Int32(16), cutlass.max(cutlass.Int32(0), limit - row_base)
        )
        pad = 16 - valid
        source = cute.domain_offset(
            (0, pad, 1 << 30, row_base - pad + (1 << 30)), mapped
        )
        tiles = cute.local_tile(source, (128, 16, 1, 1), (None, 0, 0, 0))
        return cpasync.tma_partition(
            atom,
            0,
            cute.make_layout(1),
            cute.group_modes(tma_shared, 0, 4),
            cute.group_modes(tiles, 0, 4),
        )

    @cute.jit
    def copy_clamped_b(self, atom, shared_copy, global_copy, stage, kt, barrier):
        for quarter in cutlass.range_constexpr(4):
            cute.copy(
                atom,
                global_copy[(None, kt * 4 + quarter)],
                shared_copy[(None, stage * 4 + quarter)],
                tma_bar_ptr=barrier,
            )

    @cute.jit
    def read_work(self, mp, state, records, work):
        mp.consumer_wait(state)
        for i in cutlass.range_constexpr(5):
            work[i] = records[(i, state.index)]
        cute.arch.fence_proxy("async.shared", space="cta")
        mp.consumer_release(state)

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer,
        b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        permuted_idx_to_expanded_idx_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        a_per_token_scale_ptr: Optional[cute.Pointer],
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,
        num_tokens: cutlass.Int64,
        top_k: cutlass.Int64,
        tile_size: cutlass.Constexpr,
        scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
    ):
        weights = cute.make_tensor(
            b_ptr, cute.make_layout((n, k, l), stride=(k, 1, n * k))
        )
        sfa_tma, sfa_ma = self.weight_scale_tma(b_sf_ptr, n, k, l)
        x = cute.make_tensor(a_ptr, cute.make_layout((m, k), stride=(k, 1)))
        xs = cute.make_tensor(
            cute.recast_ptr(a_sf_ptr, dtype=cutlass.Uint8), self.scale_layout(m, k, 1)
        )
        route_expert = cute.make_tensor(
            tile_idx_to_group_idx_ptr, cute.make_layout(m // 128)
        )
        route_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, cute.make_layout(m // 128)
        )
        routes = cute.make_tensor(permuted_idx_to_expanded_idx_ptr, cute.make_layout(m))
        active_tiles = cute.make_tensor(num_non_exiting_tiles_ptr, cute.make_layout(1))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(l))
        route_weights = cute.make_tensor(
            token_final_scales_ptr,
            cute.make_layout((num_tokens, top_k), stride=(top_k, 1)),
        )
        output_tokens = (
            num_tokens * top_k
            if cutlass.const_expr(self.write_expanded_weighted)
            else num_tokens
        )
        out = cute.make_tensor(
            c_ptr, cute.make_layout((output_tokens, n), stride=(n, 1))
        )
        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN,
            cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.sf_dtype,
            32,
            self.cta_group,
            self.tile[:2],
        )
        al = sm100_utils.make_smem_layout_a(mma, self.tile, cutlass.Int8, 3)
        bl = sm100_utils.make_smem_layout_b(mma, self.tile, cutlass.Float8E4M3FN, 3)
        b_tma, b_mapped = self.clamped_b_tma(a_ptr, k, bl.inner)
        sal = blockscaled_utils.make_smem_layout_sfa(mma, self.scale_tile, 32, 12)
        sbl = blockscaled_utils.make_smem_layout_sfb(mma, self.scale_tile, 32, 4)
        cluster = cute.tiled_divide(cute.make_layout((1, 1, 1, 1)), (mma.thr_id.shape,))
        tma, ma = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group),
            weights,
            cute.slice_(al, (None, None, None, 0)),
            self.tile,
            mma,
            cluster.shape,
            internal_type=cutlass.Int8,
        )

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[cutlass.Int64, 44]
            tmem_address: cutlass.Int32
            a: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int8, cute.cosize(al.outer)], 1024
            ]
            b: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(bl.outer)], 1024
            ]
            sa: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, 6144], 128]
            sb: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 768], 128]
            work: cute.struct.MemRange[cutlass.Int32, 10]
            output_rows: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, 1024], 128
            ]

        assert Storage.size_in_bytes() == self.expected_shared_storage_bytes  # type: ignore[attr-defined]
        self.shared_storage = Storage
        self.kernel(
            mma,
            tma,
            ma,
            sfa_tma,
            sfa_ma,
            b_tma,
            b_mapped,
            x,
            xs,
            routes,
            route_expert,
            route_limit,
            active_tiles,
            alpha,
            route_weights,
            out,
            al,
            bl,
            sal,
            sbl,
        ).launch(
            grid=(max_active_clusters, 1, 1),
            block=(512, 1, 1),
            smem=Storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mma,
        tma,
        ma,
        sfa_tma,
        sfa_ma,
        b_tma,
        b_mapped,
        x,
        xs,
        routes,
        route_expert,
        route_limit,
        active_tiles,
        alpha,
        route_weights,
        out,
        al,
        bl,
        sal,
        sbl,
    ):
        tid, _, _ = cute.arch.thread_idx()
        block_id, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        k = cute.size(x, mode=[1])
        features = cute.size(out, mode=[1])
        feature_tiles = features // 128
        token_subtiles = 1  # Split-only sparse filter admits native tile slot0 only.
        top_k = cute.size(route_weights, mode=[1])
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bars = storage.barriers.data_ptr()
        ap = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_bytes,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        bp = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars + 6,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.b_tma_bytes,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        sap = pipeline.PipelineTmaUmma.create(
            num_stages=3,
            barrier_storage=bars + 12,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=2048,
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        sbp = pipeline.PipelineAsync.create(
            num_stages=3,
            barrier_storage=bars + 18,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True,
        )
        tap = pipeline.PipelineAsyncUmma.create(
            num_stages=3,
            barrier_storage=bars + 24,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        tbp = pipeline.PipelineAsyncUmma.create(
            num_stages=3,
            barrier_storage=bars + 30,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        cp = pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            barrier_storage=bars + 36,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=None,
            defer_sync=True,
        )
        mp = pipeline.PipelineAsync.create(
            num_stages=2,
            barrier_storage=bars + 40,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 480),
            defer_sync=True,
        )
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sa = storage.a.get_tensor(al.outer, swizzle=al.inner)
        sb = storage.b.get_tensor(bl.outer, swizzle=bl.inner)
        b_tma_shared = storage.b.get_tensor(
            cute.make_layout((128, 16, 1, 1, 12)), swizzle=bl.inner
        )
        sfa = storage.sa.get_tensor(sal)
        sfa_copy, gfa_copy = self.weight_scale_partition(sfa_tma, sfa_ma, sfa)
        raw_sfb = storage.sb.get_tensor(cute.make_layout(768))
        compact_words = cute.make_tensor(
            cute.recast_ptr(raw_sfb.iterator, dtype=cutlass.Uint32),
            cute.make_layout(192),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_address.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=0,
            is_two_cta=False,
        )
        tmem.allocate(128)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        acc_layout = mma.make_fragment_C(mma.partition_shape_C(self.tile[:2])).layout
        sat = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.scale_tile, 32, cute.slice_(sal, (None, None, None, 0))
        )
        sbt = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.scale_tile, 32, cute.slice_(sbl, (None, None, None, 0))
        )
        ga = mma.get_slice(0).partition_A(
            cute.local_tile(ma, (128, 512), (None, None, None))
        )
        sa_copy, ga_copy = cpasync.tma_partition(
            tma,
            0,
            cute.make_layout(1),
            cute.group_modes(sa, 0, 3),
            cute.group_modes(ga, 0, 3),
        )
        fa, fb = mma.make_fragment_A(sa), mma.make_fragment_B(sb)
        records = storage.work.get_tensor(cute.make_layout((5, 2), stride=(1, 5)))
        work = cute.make_rmem_tensor((5,), cutlass.Int32)
        output_rows = storage.output_rows.get_tensor(self.output_shared_layout())

        # Warp15 schedules valid virtual N16 work from existing device routing.
        if warp == 15:
            ms = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            total = active_tiles[0] * feature_tiles * token_subtiles
            linear_work = cutlass.Int64(block_id)
            while linear_work < total:
                # Feature tiles vary fastest, so an even CTA stride does not
                # pin CTAs to empty virtual slots on sparse per-expert routing.
                feature = linear_work % feature_tiles
                slot = (linear_work // feature_tiles) % token_subtiles
                route_tile = linear_work // (token_subtiles * feature_tiles)
                limit = route_limit[route_tile]
                row_base = route_tile * 128 + slot * 16
                # Partition native128 routing tiles, not whole experts or N16 slots.
                split_live = cutlass.min(
                    cutlass.Int32(128),
                    cutlass.max(
                        cutlass.Int32(0), limit - cutlass.Int32(route_tile * 128)
                    ),
                )
                if (row_base < limit) & (split_live > 0) & (split_live <= 16):
                    mp.producer_acquire(ms)
                    expert = route_expert[route_tile]
                    with cute.arch.elect_one():
                        records[(0, ms.index)] = feature.to(cutlass.Int32)
                        records[(1, ms.index)] = route_tile.to(cutlass.Int32)
                        records[(2, ms.index)] = slot.to(cutlass.Int32)
                        records[(3, ms.index)] = expert
                        records[(4, ms.index)] = limit
                    cute.arch.fence_proxy("async.shared", space="cta")
                    mp.producer_commit(ms)
                    ms.advance()
                linear_work += grid_x
            mp.producer_acquire(ms)
            with cute.arch.elect_one():
                for i in cutlass.range_constexpr(5):
                    records[(i, ms.index)] = cutlass.Int32(-1)
            cute.arch.fence_proxy("async.shared", space="cta")
            mp.producer_commit(ms)
            ms.advance()
            mp.producer_tail(ms)

        # Warp 11: only native packed-weight TMA, one K512 transaction/stage.
        if warp == 11:
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_a = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_a, records, work)
            meta_a.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    ap.producer_acquire(a_state)
                    cute.copy(
                        tma,
                        ga_copy[(None, bm, kt, expert)],
                        sa_copy[(None, a_state.index)],
                        tma_bar_ptr=ap.producer_get_barrier(a_state),
                    )
                    a_state.advance()
                self.read_work(mp, meta_a, records, work)
                meta_a.advance()
            ap.producer_tail(a_state)

        # Warp8: four ordinary N16/K128 TMA copies into each native K512 slot.
        if warp == 8:
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_b, records, work)
            meta_b.advance()
            while work[0] >= 0:
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                b_shared_copy, b_global_copy = self.clamped_b_partition(
                    b_tma, b_mapped, b_tma_shared, row_base, limit
                )
                for kt in cutlass.range(k // 512, unroll=1):
                    bp.producer_acquire(b_state)
                    self.copy_clamped_b(
                        b_tma,
                        b_shared_copy,
                        b_global_copy,
                        b_state.index,
                        kt,
                        bp.producer_get_barrier(b_state),
                    )
                    b_state.advance()
                self.read_work(mp, meta_b, records, work)
                meta_b.advance()
            bp.producer_tail(b_state)

        # Warp9 keeps its 32 metadata consumers, including the sentinel.
        # It no longer touches the B data ring or contributes B arrivals.
        if warp == 9:
            meta_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_b, records, work)
            meta_b.advance()
            while work[0] >= 0:
                self.read_work(mp, meta_b, records, work)
                meta_b.advance()

        # Warp 12: one 2048-byte TMA into the unchanged expanded SFA ring.
        # Warp13 releases its shared lifetime only after CP reads complete.
        if warp == 12:
            sa_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_sa = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_sa, records, work)
            meta_sa.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sap.producer_acquire(sa_prod)
                    # kt is a K512 tile coordinate; local_tile maps it to
                    # the raw source K128 coordinate 4*kt.
                    cute.copy(
                        sfa_tma,
                        gfa_copy[(None, 0, kt, bm, expert)],
                        sfa_copy[(None, sa_prod.index)],
                        tma_bar_ptr=sap.producer_get_barrier(sa_prod),
                    )
                    sa_prod.advance()
                self.read_work(mp, meta_sa, records, work)
                meta_sa.advance()
            sap.producer_tail(sa_prod)

        # Warp 10: exactly 256 SFB bytes/K512, no padded 128-row expansion.
        if warp == 10:
            sb_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_sb = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_sb, records, work)
            meta_sb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sbp.producer_acquire(sb_prod)
                    g = lane % 4
                    for kc in cutlass.range_constexpr(4):
                        for row_half in cutlass.range_constexpr(2):
                            row = lane // 4 + row_half * 8
                            value = cutlass.Uint8(127)
                            if row_base + row < limit:
                                r = row_base + row
                                value = xs[
                                    (
                                        r % 32,
                                        (r % 128) // 32,
                                        r // 128,
                                        g,
                                        kt * 4 + kc,
                                        0,
                                    )
                                ].to(cutlass.Uint8)
                            raw_sfb[
                                sb_prod.index * 256 + kc * 64 + row_half * 32 + lane
                            ] = value
                    sbp.producer_commit(sb_prod)
                    sb_prod.advance()
                self.read_work(mp, meta_sb, records, work)
                meta_sb.advance()
            sbp.producer_tail(sb_prod)

        # Warp 13: four K128 CPs into this stage's independent 16-column
        # TMEM SFA slot. CP-to-MMA is the PTX pipelined cross-thread pattern.
        if warp == 13:
            sa_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 3
            )
            ta_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            meta_ta = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_ta, records, work)
            meta_ta.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):  # noqa: B007
                    sap.consumer_wait(sa_cons)
                    tap.producer_acquire(ta_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        ta_stage = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 32 + ta_prod.index * 16 + kc * 4,
                                dtype=self.sf_dtype,
                            ),
                            sat,
                        )
                        cpa, cpa_src, cpa_dst = self.scale_copy(sfa, ta_stage)
                        cute.copy(
                            cpa,
                            cpa_src[(None, None, None, None, sa_cons.index * 4 + kc)],
                            cpa_dst,
                        )
                    tcgen05_fence_before_thread_sync()
                    tap.producer_commit(ta_prod)
                    # This method emits an elected tcgen05.commit: shared SFA
                    # cannot be overwritten until the preceding CP reads finish.
                    sap.consumer_release(sa_cons)
                    sa_cons.advance()
                    ta_prod.advance()
                self.read_work(mp, meta_ta, records, work)
                meta_ta.advance()
            tap.producer_tail(ta_prod)

        # Warps 4..7: four-warp compact scale-factor publication.
        if warp >= 4 and warp < 8:
            sb_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 3
            )
            tb_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 3
            )
            copy_tid = (warp - 4) * 32 + lane
            meta_tb = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_tb, records, work)
            meta_tb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):  # noqa: B007
                    sbp.consumer_wait(sb_cons)
                    tbp.producer_acquire(tb_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        word_tensor = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 80 + tb_prod.index * 8 + kc * 2,
                                dtype=cutlass.Uint32,
                            ),
                            cute.make_layout((128, 1), stride=(65536, 1)),
                        )
                        st = tcgen05.make_tmem_copy(
                            cute.make_copy_atom(
                                tcgen05.St32x32bOp(tcgen05.Repetition.x1),
                                cutlass.Uint32,
                            ),
                            word_tensor,
                        )
                        thread_st = st.get_slice(copy_tid)
                        coords = thread_st.partition_S(
                            cute.make_identity_tensor((128, 1))
                        )
                        registers = cute.make_rmem_tensor(coords.shape, cutlass.Uint32)
                        assert cute.size(registers) == 1
                        word = cutlass.Uint32(0)
                        if lane < 16:
                            word = compact_words[
                                sb_cons.index * 64 + kc * 16 + lane
                            ].to(cutlass.Uint32)
                        registers.fill(word)
                        cute.copy(st, registers, thread_st.partition_D(word_tensor))
                    cute.arch.fence_view_async_tmem_store()
                    tcgen05_fence_before_thread_sync()
                    tbp.producer_commit(tb_prod)
                    # Every one of 128 readers has finished shared reads before
                    # the ordinary empty event; each also completed its own ST.
                    sbp.consumer_release(sb_cons)
                    sb_cons.advance()
                    tb_prod.advance()
                self.read_work(mp, meta_tb, records, work)
                meta_tb.advance()
            tbp.producer_tail(tb_prod)

        # Warp 14: obtains four independent ready tokens, issues precisely
        # sixteen ordered K32 MMAs, releases each lifetime by completion.
        if warp == 14:
            ac = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            bc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tac = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tbc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            meta_mma = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            acc_prod = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
            self.read_work(mp, meta_mma, records, work)
            meta_mma.advance()
            while work[0] >= 0:
                cp.producer_acquire(acc_prod)
                acc = cute.make_tensor(acc_ptr + acc_prod.index * 16, acc_layout)
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range(k // 512, unroll=1):  # noqa: B007
                    ap.consumer_wait(ac)
                    bp.consumer_wait(bc)
                    tap.consumer_wait(tac)
                    tbp.consumer_wait(tbc)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        ta_mma = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 32 + tac.index * 16 + kc * 4,
                                dtype=self.sf_dtype,
                            ),
                            sat,
                        )
                        tb_mma = cute.make_tensor(
                            cute.recast_ptr(
                                acc_ptr + 80 + tbc.index * 8 + kc * 2,
                                dtype=self.sf_dtype,
                            ),
                            sbt,
                        )
                        for sk in cutlass.range_constexpr(4):
                            mma.set(
                                tcgen05.Field.SFA, ta_mma[(None, None, sk)].iterator
                            )
                            mma.set(
                                tcgen05.Field.SFB, tb_mma[(None, None, sk)].iterator
                            )
                            kb = kc * 4 + sk
                            cute.gemm(
                                mma,
                                acc,
                                fa[(None, None, kb, ac.index)],
                                fb[(None, None, kb, bc.index)],
                                acc,
                            )
                            mma.set(tcgen05.Field.ACCUMULATE, True)
                    ap.consumer_release(ac)
                    bp.consumer_release(bc)
                    tap.consumer_release(tac)
                    tbp.consumer_release(tbc)
                    ac.advance()
                    bc.advance()
                    tac.advance()
                    tbc.advance()
                cp.producer_commit(acc_prod)
                acc_prod.advance()
                self.read_work(mp, meta_mma, records, work)
                meta_mma.advance()
            cp.producer_tail(acc_prod)

        # Four-warp TMEM load, token-major BF16 bridge, then existing bulk add.
        if warp < 4:
            meta_epi = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            acc_cons = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 2
            )
            self.read_work(mp, meta_epi, records, work)
            meta_epi.advance()
            while work[0] >= 0:
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                token = cutlass.Int32(0)
                output_row = cutlass.Int32(0)
                combined_scale = cutlass.Float32(0.0)
                # All warps initialize every lane before uniform shuffles.
                if lane < 16:
                    if row_base + lane < limit:
                        expanded = routes[row_base + lane]
                        token = (expanded // top_k).to(cutlass.Int32)
                        output_row = token
                        if cutlass.const_expr(self.write_expanded_weighted):
                            output_row = expanded.to(cutlass.Int32)
                        choice = expanded % top_k
                        combined_scale = cutlass.Float32(
                            alpha[work[3]] * route_weights[(token, choice)]
                        )
                cp.consumer_wait(acc_cons)
                tcgen05_fence_after_thread_sync()
                acc_out = cute.make_tensor(acc_ptr + acc_cons.index * 16, acc_layout)
                acc_mn = acc_out[((None, None), 0, 0)]
                tc, source, coords = self.epilogue_partition(acc_mn, tid)
                values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
                cute.copy(tc, source, values)
                cute.arch.fence_view_async_tmem_load()
                tcgen05_fence_before_thread_sync()
                # All N16 accumulator values are now in private registers.
                # Reuse the same eight-row bridge only after prior bulk reads finish.
                for row_group in cutlass.range_constexpr(2):
                    for vi in cutlass.range_constexpr(cute.size(values)):
                        channel, slot = coords[vi]
                        scale = cute.arch.shuffle_sync(
                            combined_scale, cutlass.Int32(slot)
                        )
                        if slot >= row_group * 8 and slot < (row_group + 1) * 8:
                            if row_base + slot < limit:
                                output_rows[(slot - row_group * 8, channel)] = (
                                    scale * values[vi]
                                ).to(cutlass.BFloat16)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()
                    if cutlass.const_expr(row_group == 0):
                        # Every epilogue thread completed the full N16 TMEM load.
                        # Group1 uses values, never the released accumulator slot.
                        cp.consumer_release(acc_cons)
                        acc_cons.advance()
                    group_token = cute.arch.shuffle_sync(
                        output_row, cutlass.Int32(lane + row_group * 8) % 32
                    )
                    if tid < 8:
                        if row_base + row_group * 8 + tid < limit:
                            destination = cute.domain_offset(
                                (group_token, work[0] * 128), out
                            )
                            if cutlass.const_expr(self.write_expanded_weighted):
                                blk_copy(
                                    destination,
                                    output_rows[tid, None],
                                    cutlass.Int32(256),
                                )
                            else:
                                blk_reduce_bf16(
                                    destination,
                                    output_rows[tid, None],
                                    cutlass.Int32(256),
                                )
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    self.epilog_sync_barrier.arrive_and_wait()
                self.read_work(mp, meta_epi, records, work)
                meta_epi.advance()

        # Every producer tail and final output completes before storage is freed.
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(acc_ptr)
