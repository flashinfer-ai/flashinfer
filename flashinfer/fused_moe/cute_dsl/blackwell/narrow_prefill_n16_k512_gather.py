"""Private native W4A8 token-N16/K512 sparse prefill gather. Minimum SM103; dispatch targets B300.

Retains caller routing tile128, packed prepared weights and token-major MXFP8
output. Six asynchronous operand/SF rings follow the qualified N8 primitive;
metadata and accumulator pipelines make the CTA persistent across output tiles.
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
from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    UnalignedNamedBarrier, native_situ_f32, native_tanh_f32,
    tcgen05_fence_after_thread_sync, tcgen05_fence_before_thread_sync,
)
from ....quantization.quantization_cute_dsl_utils import (
    float_to_ue8m0_fast, ue8m0_to_inv_scale_fast,
)


class NarrowPrefillN16K512GatherKernel:
    def __init__(
        self, *, topk=16, runtime_situ_linear_beta=False, enable_b_vector_load=False,
        enable_route_split_sparse=False
    ):
        self.top_k = topk
        self.runtime_situ_linear_beta = runtime_situ_linear_beta
        self.enable_skinny_decode = True
        self.enable_route_split_sparse = enable_route_split_sparse
        self.enable_b_vector_load = enable_b_vector_load
        self.enable_decode_compact_epilogue = False
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.cta_group = tcgen05.CtaGroup.ONE
        self.tile = (128, 16, 512)
        self.scale_tile = (128, 16, 128)
        self.num_ab_stage, self.num_acc_stage, self.num_c_stage = 3, 2, 0
        self.num_b_stage = 2
        self.num_tmem_alloc_cols, self.threads_wo_sched = 128, 480
        self.tmem_alloc_barrier = pipeline.NamedBarrier(barrier_id=1, num_threads=512)
        self.epilog_sync_barrier = UnalignedNamedBarrier(barrier_id=2, num_threads=128)
        self.tma_bytes = 32768

    @cute.jit
    def scale_copy(self, shared, tensor):
        shared_compact = cute.filter_zeros(shared)
        tensor_compact = cute.filter_zeros(tensor)
        atom = cute.make_copy_atom(tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype)
        copy = tcgen05.make_s2t_copy(atom, tensor_compact)
        part = copy.get_slice(0)
        source = tcgen05.get_s2t_smem_desc_tensor(copy, part.partition_S(shared_compact))
        return copy, source, part.partition_D(tensor_compact)

    @cute.jit
    def read_work(self, mp, state, records, work):
        mp.consumer_wait(state)
        for i in cutlass.range_constexpr(5):
            work[i] = records[(i, state.index)]
        cute.arch.fence_proxy("async.shared", space="cta")
        mp.consumer_release(state)

    @cute.jit
    def wrapper(
        self, a_ptr: cute.Pointer, b_ptr: cute.Pointer,
        a_sf_ptr: cute.Pointer, b_sf_ptr: cute.Pointer,
        c_ptr: cute.Pointer, c_sf_ptr: Optional[cute.Pointer], alpha_ptr: cute.Pointer,
        tile_idx_to_group_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer, token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        global_sf_ptr: Optional[cute.Pointer], a_per_token_scale_ptr: Optional[cute.Pointer],
        orig_m: cutlass.Int64, m: cutlass.Int64, n: cutlass.Int64,
        k: cutlass.Int64, l: cutlass.Int64,
        tile_size: cutlass.Constexpr, scaling_vector_size: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr, stream: cuda.CUstream,
        epilogue_op: cutlass.Constexpr = lambda x: x,
        situ_beta_ptr: Optional[cute.Pointer] = None, situ_linear_beta_ptr: Optional[cute.Pointer] = None,
        situ_beta_stride: cutlass.Int32 = 0, situ_linear_beta_stride: cutlass.Int32 = 0,
    ):
        weights = cute.make_tensor(b_ptr, cute.make_layout((n, k, l), stride=(k, 1, n*k)))
        weight_scales = cute.make_tensor(cute.recast_ptr(b_sf_ptr, dtype=cutlass.Uint8),
            cute.make_layout(l*n*k//32))
        x = cute.make_tensor(a_ptr, cute.make_layout((orig_m, k), stride=(k, 1)))
        xs = cute.make_tensor(cute.recast_ptr(a_sf_ptr, dtype=cutlass.Uint8),
            cute.make_layout((orig_m, k//32), stride=(k//32, 1)))
        route_expert = cute.make_tensor(tile_idx_to_group_idx_ptr, cute.make_layout(m//128))
        route_limit = cute.make_tensor(tile_idx_to_mn_limit_ptr, cute.make_layout(m//128))
        routes = cute.make_tensor(token_id_mapping_ptr, cute.make_layout(m))
        active_tiles = cute.make_tensor(num_non_exiting_tiles_ptr, cute.make_layout(1))
        alpha = cute.make_tensor(alpha_ptr, cute.make_layout(l))
        beta = cute.make_tensor(situ_beta_ptr, cute.make_layout(l, stride=situ_beta_stride))
        linear = None
        if cutlass.const_expr(self.runtime_situ_linear_beta):
            linear = cute.make_tensor(situ_linear_beta_ptr,
                cute.make_layout(l, stride=situ_linear_beta_stride))
        out = cute.make_tensor(c_ptr, cute.make_layout((m, n//2), stride=(n//2, 1)))
        out_sf = cute.make_tensor(cute.recast_ptr(c_sf_ptr, dtype=cutlass.Uint8),
            cute.make_layout(m*(n//2)//32))
        mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            cutlass.Float4E2M1FN, cutlass.Float8E4M3FN,
            tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K,
            self.sf_dtype, 32, self.cta_group, self.tile[:2])
        al = sm100_utils.make_smem_layout_a(mma, self.tile, cutlass.Int8, 3)
        bl = sm100_utils.make_smem_layout_b(mma, self.tile, cutlass.Float8E4M3FN, self.num_b_stage)
        sal = blockscaled_utils.make_smem_layout_sfa(mma, self.scale_tile, 32, 12)
        sbl = blockscaled_utils.make_smem_layout_sfb(mma, self.scale_tile, 32, 4)
        cluster = cute.tiled_divide(cute.make_layout((1, 1, 1, 1)), (mma.thr_id.shape,))
        tma, ma = cute.nvgpu.make_tiled_tma_atom_A(
            cpasync.CopyBulkTensorTileG2SOp(self.cta_group), weights,
            cute.slice_(al, (None, None, None, 0)), self.tile, mma,
            cluster.shape, internal_type=cutlass.Int8)

        @cute.struct
        class Storage:
            barriers: cute.struct.MemRange[cutlass.Int64, 44]
            tmem_address: cutlass.Int32
            a: cute.struct.Align[cute.struct.MemRange[cutlass.Int8, cute.cosize(al.outer)], 1024]
            b: cute.struct.Align[cute.struct.MemRange[cutlass.Float8E4M3FN, cute.cosize(bl.outer)], 1024]
            sa: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, 6144], 128]
            sb: cute.struct.Align[cute.struct.MemRange[cutlass.Uint8, 768], 128]
            work: cute.struct.MemRange[cutlass.Int32, 10]
            gates: cute.struct.Align[cute.struct.MemRange[cutlass.Float32, 1024], 128]
        self.shared_storage = Storage
        self.kernel(mma, tma, ma, weight_scales, x, xs, routes, route_expert,
            route_limit, active_tiles, alpha, beta, linear, out, out_sf,
            al, bl, sal, sbl, epilogue_op).launch(
                grid=(max_active_clusters, 1, 1), block=(512, 1, 1),
                smem=Storage.size_in_bytes(), stream=stream)

    @cute.kernel
    def kernel(self, mma, tma, ma, ws, x, xs, routes, route_expert,
               route_limit, active_tiles, alpha, beta, linear, out, out_sf,
               al, bl, sal, sbl, epilogue_op: cutlass.Constexpr):
        block_id, _, _ = cute.arch.block_idx()
        feature_tiles = cute.size(out, mode=[1]) // 64
        if cutlass.const_expr(self.enable_route_split_sparse):
            token_subtiles = cutlass.Int64(1)
        else:
            token_subtiles = cute.ceil_div(cute.size(x, mode=[0]), 16)
        if block_id < active_tiles[0] * feature_tiles * token_subtiles:
            self._kernel_body(
                mma, tma, ma, ws, x, xs, routes, route_expert,
                route_limit, active_tiles, alpha, beta, linear, out, out_sf,
                al, bl, sal, sbl, epilogue_op,
            )

    @cute.jit
    def _kernel_body(self, mma, tma, ma, ws, x, xs, routes, route_expert,
                     route_limit, active_tiles, alpha, beta, linear, out, out_sf,
                     al, bl, sal, sbl, epilogue_op: cutlass.Constexpr):
        tid, _, _ = cute.arch.thread_idx()
        block_id, _, _ = cute.arch.block_idx()
        grid_x, _, _ = cute.arch.grid_dim()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tid % 32
        k = cute.size(x, mode=[1])
        features = cute.size(out, mode=[1])
        feature_tiles = features // 64
        if cutlass.const_expr(self.enable_route_split_sparse):
            token_subtiles = cutlass.Int64(1)
        else:
            token_subtiles = cute.ceil_div(cute.size(x, mode=[0]), 16)
        storage = utils.SmemAllocator().allocate(self.shared_storage)
        bars = storage.barriers.data_ptr()
        ap = pipeline.PipelineTmaUmma.create(
            num_stages=3, barrier_storage=bars,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.tma_bytes, cta_layout_vmnk=None, defer_sync=True)
        if cutlass.const_expr(self.enable_b_vector_load):
            # Full-barrier arrivals track completion of each producer's cp.async.
            bp = PipelineCpAsyncUmma.create(
                num_stages=self.num_b_stage, barrier_storage=bars + 6,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 64),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=None, defer_sync=True)
        else:
            bp = pipeline.PipelineAsyncUmma.create(
                num_stages=self.num_b_stage, barrier_storage=bars + 6,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 64),
                consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                cta_layout_vmnk=None, defer_sync=True)
        sap = PipelineCpAsyncUmma.create(
            num_stages=3, barrier_storage=bars + 12,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None, defer_sync=True)
        sbp = pipeline.PipelineAsync.create(
            num_stages=3, barrier_storage=bars + 18,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            defer_sync=True)
        tap = pipeline.PipelineAsyncUmma.create(
            num_stages=3, barrier_storage=bars + 24,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None, defer_sync=True)
        tbp = pipeline.PipelineAsyncUmma.create(
            num_stages=3, barrier_storage=bars + 30,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            cta_layout_vmnk=None, defer_sync=True)
        cp = pipeline.PipelineUmmaAsync.create(
            num_stages=2, barrier_storage=bars+36,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 128),
            cta_layout_vmnk=None, defer_sync=True)
        mp = pipeline.PipelineAsync.create(
            num_stages=2, barrier_storage=bars+40,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 32),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 480),
            defer_sync=True)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        sa = storage.a.get_tensor(al.outer, swizzle=al.inner)
        sb = storage.b.get_tensor(bl.outer, swizzle=bl.inner)
        b_nks = storage.b.get_tensor(cute.composition(bl.outer,
            cute.make_layout((16, 512, self.num_b_stage), stride=(1, 16, 8192))), swizzle=bl.inner)
        sfa = storage.sa.get_tensor(sal)
        raw_sfa = cute.make_tensor(cute.recast_ptr(sfa.iterator, dtype=cutlass.Uint8),
                                   cute.make_layout(6144))
        raw_sfb = storage.sb.get_tensor(cute.make_layout(768))
        compact_words = cute.make_tensor(cute.recast_ptr(raw_sfb.iterator, dtype=cutlass.Uint32),
                                         cute.make_layout(192))
        tmem = utils.TmemAllocator(storage.tmem_address.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier, allocator_warp_id=0, is_two_cta=False)
        tmem.allocate(128)
        tmem.wait_for_alloc()
        acc_ptr = tmem.retrieve_ptr(cutlass.Float32)
        acc_layout = mma.make_fragment_C(mma.partition_shape_C(self.tile[:2])).layout
        sat = blockscaled_utils.make_tmem_layout_sfa(
            mma, self.scale_tile, 32, cute.slice_(sal, (None, None, None, 0)))
        sbt = blockscaled_utils.make_tmem_layout_sfb(
            mma, self.scale_tile, 32, cute.slice_(sbl, (None, None, None, 0)))
        ga = mma.get_slice(0).partition_A(
            cute.local_tile(ma, (128, 512), (None, None, None)))
        sa_copy, ga_copy = cpasync.tma_partition(tma, 0, cute.make_layout(1),
            cute.group_modes(sa, 0, 3), cute.group_modes(ga, 0, 3))
        fa, fb = mma.make_fragment_A(sa), mma.make_fragment_B(sb)
        records = storage.work.get_tensor(cute.make_layout((5, 2), stride=(1, 5)))
        work = cute.make_rmem_tensor((5,), cutlass.Int32)
        gates = storage.gates.get_tensor(cute.make_layout((16, 64), stride=(64, 1)))

        # Warp15 schedules valid virtual N16 work from existing device routing.
        if warp == 15:
            ms = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            total = active_tiles[0] * feature_tiles * token_subtiles
            linear_work = cutlass.Int64(block_id)
            while linear_work < total:
                # Feature tiles vary fastest, so an even CTA stride does not
                # pin half the CTAs to empty slot1 on sparse T9..16 routing.
                feature = linear_work % feature_tiles
                slot = (linear_work // feature_tiles) % token_subtiles
                route_tile = linear_work // (token_subtiles * feature_tiles)
                limit = route_limit[route_tile]
                row_base = route_tile * 128 + slot * 16
                admit_work = row_base < limit
                if cutlass.const_expr(self.enable_route_split_sparse):
                    live_rows = limit - route_tile * 128
                    admit_work = admit_work & (live_rows > 0) & (live_rows <= 16)
                if admit_work:
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
            a_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 3)
            meta_a = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_a, records, work)
            meta_a.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for pf in cutlass.range(0, cutlass.min(3, k // 512), unroll=1):
                    cute.prefetch(tma, ga_copy[(None, bm, cutlass.Int32(pf), expert)])
                for kt in cutlass.range(k // 512, unroll=1):
                    ap.producer_acquire(a_state)
                    cute.copy(tma, ga_copy[(None, bm, kt, expert)], sa_copy[(None, a_state.index)],
                              tma_bar_ptr=ap.producer_get_barrier(a_state))
                    if kt + 3 < k // 512:
                        cute.prefetch(tma, ga_copy[(None, bm, cutlass.Int32(kt + 3), expert)])
                    a_state.advance()
                self.read_work(mp, meta_a, records, work)
                meta_a.advance()
            ap.producer_tail(a_state)

        # Warps 8/9: independent B gather, 64 producers and full SW128 helper.
        if warp >= 8 and warp < 10:
            if cutlass.const_expr(self.enable_b_vector_load):
                b_atom = cute.make_copy_atom(
                    cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                    cutlass.Float8E4M3FN, num_bits_per_copy=128)
            b_state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, self.num_b_stage)
            local_tid = (warp - 8) * 32 + lane
            meta_b = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_b, records, work)
            meta_b.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                # Routing is invariant through this work item's K loop.
                # Refresh all eight constexpr-indexed registers for every item.
                routed_tokens = cute.make_rmem_tensor((8,), cutlass.Int32)
                routed_tokens.fill(cutlass.Int32(0))
                for vr in cutlass.range_constexpr(8):
                    row = (local_tid + vr * 64) // 32
                    if row_base + row < limit:
                        routed_tokens[vr] = routes[row_base + row] // self.top_k
                for kt in cutlass.range(k // 512, unroll=1):
                    bp.producer_acquire(b_state)
                    b_nk = b_nks[(None, None, b_state.index)]
                    for vr in cutlass.range_constexpr(8):
                        vi = local_tid + vr * 64
                        row, chunk = vi // 32, vi % 32
                        if cutlass.const_expr(self.enable_b_vector_load):
                            # 64x8 ownership covers the actual N16/K512 layout in aligned 16-byte slices.
                            # Invalid rows use safe token 0 and atom-level zero fill.
                            offset = cute.assume(
                                routed_tokens[vr] * k + kt * 512 + chunk * 16, divby=16)
                            source = cute.make_tensor(x.iterator + offset,
                                cute.make_layout((16,)))
                            destination = cute.make_tensor(
                                cute.local_tile(b_nk, (1, 16), (row, chunk)).iterator,
                                cute.make_layout((16,)))
                            pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
                            pred[0] = row_base + row < limit
                            cute.copy_atom_call(b_atom, source, destination, pred=pred)
                        else:
                            values = cute.make_rmem_tensor((1, 16), cutlass.Float8E4M3FN)
                            values.fill(cutlass.Float32(0).to(cutlass.Float8E4M3FN))
                            if row_base + row < limit:
                                token = routed_tokens[vr]
                                cute.autovec_copy(cute.local_tile(x, (1, 16),
                                    (token, kt * 32 + chunk)), values)
                            cute.autovec_copy(values, cute.local_tile(b_nk, (1, 16), (row, chunk)))
                    cute.arch.fence_proxy("async.shared", space="cta")
                    bp.producer_commit(b_state)
                    b_state.advance()
                self.read_work(mp, meta_b, records, work)
                meta_b.advance()
            bp.producer_tail(b_state)

        # Warp 12: expanded native SFA only. Its buffer is consumed by CP,
        # not released when CP is merely issued.
        if warp == 12:
            sa_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                cutlass.Uint8, num_bits_per_copy=128)
            sa_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 3)
            meta_sa = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_sa, records, work)
            meta_sa.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sap.producer_acquire(sa_prod)
                    for kc in cutlass.range_constexpr(4):
                        # The four quarter/group slices form one contiguous
                        # 16-byte lane slice. Both bases are already aligned.
                        global_offset = cute.assume(
                            (expert * feature_tiles + bm) * (k // 128) * 512
                            + (kt * 4 + kc) * 512 + lane * 16, divby=16)
                        shared_offset = cute.assume(
                            (sa_prod.index * 4 + kc) * 512 + lane * 16, divby=16)
                        source = cute.make_tensor(ws.iterator + global_offset,
                            cute.make_layout((16,)))
                        destination = cute.make_tensor(raw_sfa.iterator + shared_offset,
                            cute.make_layout((16,)))
                        cute.copy_atom_call(sa_atom, source, destination)
                    cute.arch.fence_proxy("async.shared", space="cta")
                    sap.producer_commit(sa_prod)
                    sa_prod.advance()
                self.read_work(mp, meta_sa, records, work)
                meta_sa.advance()
            sap.producer_tail(sa_prod)

        # Warp 10: exactly 256 SFB bytes/K512, no padded 128-row expansion.
        if warp == 10:
            sb_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 3)
            meta_sb = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_sb, records, work)
            meta_sb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sbp.producer_acquire(sb_prod)
                    for row_half in cutlass.range_constexpr(2):
                        row, g = lane // 4 + row_half * 8, lane % 4
                        for kc in cutlass.range_constexpr(4):
                            value = cutlass.Uint8(127)
                            if row_base + row < limit:
                                value = xs[routes[row_base + row] // self.top_k, kt * 16 + kc * 4 + g].to(cutlass.Uint8)
                            raw_sfb[sb_prod.index * 256 + kc * 64 + row_half * 32 + lane] = value
                    sbp.producer_commit(sb_prod)
                    sb_prod.advance()
                self.read_work(mp, meta_sb, records, work)
                meta_sb.advance()
            sbp.producer_tail(sb_prod)

        # Warp 13: four K128 CPs into this stage's independent 16-column
        # TMEM SFA slot. CP-to-MMA is the PTX pipelined cross-thread pattern.
        if warp == 13:
            sa_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            ta_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 3)
            meta_ta = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_ta, records, work)
            meta_ta.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sap.consumer_wait(sa_cons)
                    tap.producer_acquire(ta_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        ta_stage = cute.make_tensor(cute.recast_ptr(
                            acc_ptr + 32 + ta_prod.index * 16 + kc * 4, dtype=self.sf_dtype), sat)
                        cpa, cpa_src, cpa_dst = self.scale_copy(sfa, ta_stage)
                        cute.copy(cpa, cpa_src[(None, None, None, None, sa_cons.index * 4 + kc)], cpa_dst)
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
            sb_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tb_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 3)
            copy_tid = (warp - 4) * 32 + lane
            meta_tb = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_tb, records, work)
            meta_tb.advance()
            while work[0] >= 0:
                bm, expert = work[0], work[3]
                row_base, limit = work[1] * 128 + work[2] * 16, work[4]
                for kt in cutlass.range(k // 512, unroll=1):
                    sbp.consumer_wait(sb_cons)
                    tbp.producer_acquire(tb_prod)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        word_tensor = cute.make_tensor(cute.recast_ptr(
                            acc_ptr + 80 + tb_prod.index * 8 + kc * 2, dtype=cutlass.Uint32),
                            cute.make_layout((128, 1), stride=(65536, 1)))
                        st = tcgen05.make_tmem_copy(cute.make_copy_atom(
                            tcgen05.St32x32bOp(tcgen05.Repetition.x1), cutlass.Uint32), word_tensor)
                        thread_st = st.get_slice(copy_tid)
                        coords = thread_st.partition_S(cute.make_identity_tensor((128, 1)))
                        registers = cute.make_rmem_tensor(coords.shape, cutlass.Uint32)
                        assert cute.size(registers) == 1
                        word = cutlass.Uint32(0)
                        if lane < 16:
                            word = compact_words[sb_cons.index * 64 + kc * 16 + lane].to(cutlass.Uint32)
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
            bc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, self.num_b_stage)
            tac = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            tbc = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 3)
            meta_mma = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            acc_prod = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            self.read_work(mp, meta_mma, records, work)
            meta_mma.advance()
            while work[0] >= 0:
                cp.producer_acquire(acc_prod)
                acc = cute.make_tensor(acc_ptr + acc_prod.index * 16, acc_layout)
                mma.set(tcgen05.Field.ACCUMULATE, False)
                for kt in cutlass.range(k // 512, unroll=1):
                    # Probe all four current-stage full barriers before waiting.
                    a_ready = ap.consumer_try_wait(ac)
                    b_ready = bp.consumer_try_wait(bc)
                    ta_ready = tap.consumer_try_wait(tac)
                    tb_ready = tbp.consumer_try_wait(tbc)
                    ap.consumer_wait(ac, a_ready)
                    bp.consumer_wait(bc, b_ready)
                    tap.consumer_wait(tac, ta_ready)
                    tbp.consumer_wait(tbc, tb_ready)
                    tcgen05_fence_after_thread_sync()
                    for kc in cutlass.range_constexpr(4):
                        ta_mma = cute.make_tensor(cute.recast_ptr(
                            acc_ptr + 32 + tac.index * 16 + kc * 4, dtype=self.sf_dtype), sat)
                        tb_mma = cute.make_tensor(cute.recast_ptr(
                            acc_ptr + 80 + tbc.index * 8 + kc * 2, dtype=self.sf_dtype), sbt)
                        for sk in cutlass.range_constexpr(4):
                            mma.set(tcgen05.Field.SFA, ta_mma[(None, None, sk)].iterator)
                            mma.set(tcgen05.Field.SFB, tb_mma[(None, None, sk)].iterator)
                            kb = kc * 4 + sk
                            cute.gemm(mma, acc, fa[(None, None, kb, ac.index)],
                                      fb[(None, None, kb, bc.index)], acc)
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

        # The existing token-major bridge uses only the four epilogue warps.
        if warp < 4:
            meta_epi = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            acc_cons = pipeline.make_pipeline_state(pipeline.PipelineUserType.Consumer, 2)
            self.read_work(mp, meta_epi, records, work)
            meta_epi.advance()
            while work[0] >= 0:
                cp.consumer_wait(acc_cons)
                tcgen05_fence_after_thread_sync()
                acc_out = cute.make_tensor(acc_ptr + acc_cons.index * 16, acc_layout)
                acc_mn = acc_out[((None, None), 0, 0)]
                tc = tcgen05.make_tmem_copy(cute.make_copy_atom(
                    tcgen05.Ld32x32bOp(tcgen05.Repetition.x16), cutlass.Float32), acc_mn)
                thr = tc.get_slice(tid)
                coords = thr.partition_D(cute.make_identity_tensor((128, 16)))
                values = cute.make_rmem_tensor(coords.shape, cutlass.Float32)
                cute.copy(tc, thr.partition_S(acc_mn), values)
                cute.arch.fence_view_async_tmem_load()
                tcgen05_fence_before_thread_sync()
                if tid >= 64:
                    for vi in cutlass.range_constexpr(cute.size(values)):
                        channel, slot = coords[vi]
                        gates[(slot, channel - 64)] = values[vi]
                self.epilog_sync_barrier.arrive_and_wait()
                cp.consumer_release(acc_cons)
                acc_cons.advance()
                if tid < 64:
                    expert = work[3]
                    alpha_value, beta_value = alpha[expert], beta[expert]
                    linear_value, inverse_linear = cutlass.Float32(1.0), cutlass.Float32(1.0)
                    if cutlass.const_expr(self.runtime_situ_linear_beta):
                        linear_value = linear[expert]
                        inverse_linear = cutlass.Float32(1.0) / linear_value
                    activated = cute.make_rmem_tensor(values.shape, cutlass.Float32)
                    for vi in cutlass.range_constexpr(0, cute.size(values), 2):
                        channel0, slot0 = coords[vi]
                        channel1, slot1 = coords[vi + 1]
                        u0, u1 = cute.arch.mul_packed_f32x2(
                            (values[vi], values[vi+1]), (alpha_value, alpha_value))
                        g0, g1 = cute.arch.mul_packed_f32x2(
                            (gates[(slot0, channel0)], gates[(slot1, channel1)]),
                            (alpha_value, alpha_value))
                        v0 = native_situ_f32(g0, beta_value, fastmath=True)
                        v1 = native_situ_f32(g1, beta_value, fastmath=True)
                        if cutlass.const_expr(self.runtime_situ_linear_beta):
                            u0 = linear_value * native_tanh_f32(u0 * inverse_linear)
                            u1 = linear_value * native_tanh_f32(u1 * inverse_linear)
                        activated[vi], activated[vi+1] = cute.arch.mul_packed_f32x2((u0,u1),(v0,v1))
                    scale_values = epilogue_op(activated.load())
                    for vi in cutlass.range_constexpr(0, cute.size(values), 2):
                        channel0, slot0 = coords[vi]
                        channel1, slot1 = coords[vi+1]
                        max0 = cute.arch.warp_redux_sync(abs(scale_values[vi]),
                            kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=False)
                        max1 = cute.arch.warp_redux_sync(abs(scale_values[vi+1]),
                            kind="fmax", mask_and_clamp=0xFFFFFFFF, nan=False)
                        scale0, scale1 = cute.arch.mul_packed_f32x2((max0,max1),
                            (cutlass.Float32(1.0/448.0),cutlass.Float32(1.0/448.0)))
                        scale0, scale1 = cute.arch.mul_packed_f32x2((scale0,scale1),
                            (cutlass.Float32(1.0),cutlass.Float32(1.0)))
                        code0, code1 = float_to_ue8m0_fast(scale0), float_to_ue8m0_fast(scale1)
                        inv0 = ue8m0_to_inv_scale_fast(code0.to(cutlass.Uint32))
                        inv1 = ue8m0_to_inv_scale_fast(code1.to(cutlass.Uint32))
                        q0, q1 = cute.arch.mul_packed_f32x2(
                            (activated[vi],activated[vi+1]),(inv0,inv1))
                        row_base = work[1]*128+work[2]*16
                        row0, row1 = row_base+slot0, row_base+slot1
                        feature0, feature1 = work[0]*64+channel0, work[0]*64+channel1
                        if row0 < work[4]:
                            out[(row0,feature0)] = q0.to(cutlass.Float8E4M3FN)
                            if lane == 0:
                                sf0 = ((row0//128)*(features//128)*512+(feature0//128)*512
                                    +(row0%32)*16+((row0%128)//32)*4+(feature0%128)//32)
                                out_sf[sf0] = code0.to(cutlass.Uint8)
                        if row1 < work[4]:
                            out[(row1,feature1)] = q1.to(cutlass.Float8E4M3FN)
                            if lane == 0:
                                sf1 = ((row1//128)*(features//128)*512+(feature1//128)*512
                                    +(row1%32)*16+((row1%128)//32)*4+(feature1%128)//32)
                                out_sf[sf1] = code1.to(cutlass.Uint8)
                # Protect gate bridge reuse while the first two warps quantize.
                self.epilog_sync_barrier.arrive_and_wait()
                self.read_work(mp, meta_epi, records, work)
                meta_epi.advance()

        # Every producer tail and final output completes before storage is freed.
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(acc_ptr)
