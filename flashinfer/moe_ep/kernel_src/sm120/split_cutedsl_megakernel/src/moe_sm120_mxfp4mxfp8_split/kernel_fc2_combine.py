# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Physical FC2-only device body for SM120 Split-MegaMoE.

The host/workspace contract is shared with the split MegaMoE wrapper, but the
device kernel below contains only K2 scheduling, TMA, QMMA and token-back code.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- wheels without cute.iket
    try:
        from cutlass.cute.experimental import iket  # type: ignore
    except ImportError:
        from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils

from .custom_ext import Sm120SwapABMxfp8Fc12SchedExtension
from .fc1_fc2_fuse_sched import BlockPhase, MoEFusedFc12SchedulerParams
from .megamoe_kernel import Sm120MegaMoEMxfp8SwapABKernel
from .moe_persistent_scheduler import WorkTileState
from .moe_utils import spin_wait, spin_wait_i32_ge_inline
from .sm120_mma import (
    MMA_N,
    issue_m64n8k32_mxfp8,
    make_sm120_ldmatrix_atom,
    shift_fp4_fragment_for_mxf8f6f4,
)
from .split_timestamp import (
    FIELD_FIRST_TILE_ID,
    FIELD_FIRST_WORK,
    FIELD_KERNEL_ENTRY,
    FIELD_KERNEL_EXIT,
    FIELD_LAST_TILE_ID,
    FIELD_LAST_WORK,
    FIELD_MAINLOOP_NS,
    FIELD_READY_WAIT_CALLS,
    FIELD_READY_WAIT_NS,
    FIELD_STORE_NS,
    FIELD_TILE_COUNT,
    FIELD_TMA_A_TIMED_CALLS,
    FIELD_TMA_A_WAIT_CALLS,
    FIELD_TMA_A_WAIT_NS,
    FIELD_TMA_B_TIMED_CALLS,
    FIELD_TMA_B_WAIT_CALLS,
    FIELD_TMA_B_WAIT_NS,
    K2_TILE_FIELD_BF16_PACK_NS,
    K2_TILE_FIELD_DEQUEUE_BEGIN,
    K2_TILE_FIELD_DEQUEUE_END,
    K2_TILE_FIELD_LDSM_QMMA_NS,
    K2_TILE_FIELD_PEER_STORE_NS,
    K2_TILE_FIELD_PHASE_ADVANCE_NS,
    K2_TILE_FIELD_TILE_BEGIN,
    K2_TILE_FIELD_TILE_END,
    K2_TILE_FIELD_TILE_ID,
    K2_TILE_FIELD_TMA_A_WAIT_NS,
    K2_TILE_FIELD_TMA_B_WAIT_NS,
    K2_TILE_TRACE_TILES_PER_CTA,
    ROLE_K2,
    TRACE_TMA_WAIT_SAMPLE_STRIDE,
    k2_tile_trace_word,
    pack_tile_id,
    read_globaltimer,
    trace_word,
)
from src.token_comm import TokenSrcMetadata


UseScaleTma = True


class Sm120Fc2CombineKernel(Sm120MegaMoEMxfp8SwapABKernel):
    """SM120 K2 with a physically separate FC2-only device kernel body."""

    def __init__(
        self,
        *args,
        producer_sm_count: int,
        compact_k2: bool = True,
        **kwargs,
    ):
        super().__init__(
            *args,
            split_role="k2",
            producer_sm_count=producer_sm_count,
            compact_k2=compact_k2,
            **kwargs,
        )
        self.k2_tile_trace_enabled = self.jit_config.enable_k2_tile_trace

    @cute.kernel
    def fc2_combine_kernel_impl(self, tiled_mma: cute.TiledMma, tiled_mma_sfb: cute.TiledMma, tma_atom_fc1_weight: cute.CopyAtom, tma_tensor_fc1_weight: cute.Tensor, tma_atom_activation: cute.CopyAtom, tma_tensor_activation: cute.Tensor, tma_atom_fc1_weight_sf: cute.CopyAtom, tma_tensor_fc1_weight_sf: cute.Tensor, tma_atom_activation_sf: cute.CopyAtom, tma_tensor_activation_sf: cute.Tensor, tma_atom_fc2_weight: cute.CopyAtom, tma_tensor_fc2_weight: cute.Tensor, tma_atom_fc1_output_as_fc2_input: cute.CopyAtom, tma_tensor_fc1_output_as_fc2_input: cute.Tensor, tma_atom_fc2_weight_sf: cute.CopyAtom, tma_tensor_fc2_weight_sf: cute.Tensor, tma_atom_fc1_output_sf_as_fc2_input: cute.CopyAtom, tma_tensor_fc1_output_sf_as_fc2_input: cute.Tensor, fc1_weight_gemm: cute.Tensor, activation_gemm: cute.Tensor, fc1_output_gemm: cute.Tensor, fc1_weight_sf_gemm: cute.Tensor, activation_sf_gemm: cute.Tensor, fc1_output_sf_gemm: cute.Tensor, fc2_weight_gemm: cute.Tensor, fc2_output: cute.Tensor, fc2_weight_sf_gemm: cute.Tensor, fc1_output_sf_gemm_for_fc2_load: cute.Tensor, topk_scores: cute.Tensor, fc1_done_counter: cute.Tensor, combine_ready_flags: Optional[cute.Tensor], fc2_block_done_counter: Optional[cute.Tensor], fc1_alpha: Optional[cute.Tensor], fc2_alpha: Optional[cute.Tensor], fc1_norm_const: Optional[cute.Tensor], sched_params: MoEFusedFc12SchedulerParams, cluster_layout_vmnk: cute.Layout, cluster_layout_sfb_vmnk: cute.Layout, a_smem_layout_staged: cute.ComposedLayout, b_smem_layout_staged: cute.ComposedLayout, sfa_smem_layout_staged: cute.Layout, sfb_smem_layout_staged: cute.Layout, fc1_output_smem_layout_staged: cute.ComposedLayout, token_comm_args=None, green_trace: Optional[cute.Tensor]=None, k2_ready_queue_desc: Optional[cute.Tensor]=None, k2_ready_queue_ready: Optional[cute.Tensor]=None, k2_ready_queue_state: Optional[cute.Tensor]=None):
        """Device kernel for ready-aware FC2 + packed peer token-back."""
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        fc1_tiles_per_fc2_k_tile = self.mma_tiler[2] // (self.mma_tiler[0] // 2)
        ext_fc2_spin_threshold = (
            (fc2_weight_gemm.shape[1] + self.mma_tiler[2] - 1)
            // self.mma_tiler[2]
            * fc1_tiles_per_fc2_k_tile
            * len(self.compute_warp_id)
        )
        ext = Sm120SwapABMxfp8Fc12SchedExtension(sf_vec_size=self.sf_vec_size, fc1_done_counter_ptr=fc1_done_counter.iterator, fc2_spin_threshold=ext_fc2_spin_threshold, fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(token_comm_args))
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = False
        bidx, bidy, bidz = cute.arch.block_idx()
        gdimx, gdimy, gdimz = cute.arch.grid_dim()
        cta_linear_id = bidx + gdimx * (bidy + gdimy * bidz)
        trace_role = ROLE_K2
        if cutlass.const_expr(self.green_trace_role is not None):
            trace_role = self.green_trace_role
        if cutlass.const_expr(green_trace is not None):
            if warp_idx == cutlass.Int32(0):
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_KERNEL_ENTRY), read_globaltimer(), scope='gpu')
        mma_tile_coord_v = cutlass.Int32(0)
        is_leader_cta = True
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(cta_rank_in_cluster)
        tidx, _, _ = cute.arch.thread_idx()
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(sched_params, ext, num_drain_warps=0)

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 4]
            sched_storage: SchedStorage
            sA: cute.struct.Align[cute.struct.MemRange[self.smem_alloc_a_dtype, cute.cosize(a_smem_layout_staged)], 128]
            sB: cute.struct.Align[cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_layout_staged)], 128]
            sSFA: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_layout_staged)], 128]
            sSFB: cute.struct.Align[cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_layout_staged)], 128]
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        tma_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, len(self.compute_warp_id))
        a_producer, a_consumer = pipeline.PipelineTmaAsync.create(barrier_storage=storage.ab_full_mbar_ptr.data_ptr(), num_stages=self.num_ab_stage, producer_group=tma_pipeline_producer_group, consumer_group=ab_pipeline_consumer_group, tx_count=self.num_tma_a_load_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True).make_participants()
        b_producer, b_consumer = pipeline.PipelineTmaAsync.create(barrier_storage=storage.ab_full_mbar_ptr.data_ptr() + self.num_ab_stage * 2, num_stages=self.num_ab_stage, producer_group=tma_pipeline_producer_group, consumer_group=ab_pipeline_consumer_group, tx_count=self.num_tma_b_load_bytes, cta_layout_vmnk=cluster_layout_vmnk, defer_sync=True).make_participants()
        num_sched_consumer_threads = 32 * len((self.tma_a_warp_id, self.tma_b_warp_id, *self.compute_warp_id))
        scheduler = SchedCls.create(sched_params, cute.arch.block_idx(), cute.arch.grid_dim(), sched_storage=storage.sched_storage, num_consumer_threads=num_sched_consumer_threads, ext=ext)
        sched_consumer = scheduler.make_consumer()
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        sA = storage.sA.get_tensor(a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner)
        sB = storage.sB.get_tensor(b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner)
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        mma_tiler_k = self.mma_tiler[2]
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        fc2_ready_bundle_cnt = (k_tile_cnt_fc2 + self.fc2_ready_bundle_k_tiles - 1) // self.fc2_ready_bundle_k_tiles
        if warp_idx == self.sched_warp_id:
            if cutlass.const_expr(self.enable_token_comm and self.use_warpgroup_reg_realloc):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            queue_running = cutlass.Boolean(True)
            sched_tile_seq = cutlass.Int32(0)
            while queue_running:
                dequeue_begin = cutlass.Int64(0)
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if cute.arch.lane_idx() == cutlass.Int32(0):
                        dequeue_begin = read_globaltimer()
                queue_idx = cutlass.Int32(0)
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    queue_idx = cute.arch.atomic_add(
                        k2_ready_queue_state.iterator,
                        cutlass.Int32(1),
                        scope="gpu",
                    )
                queue_idx = cute.arch.shuffle_sync(
                    queue_idx, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    spin_wait(
                        k2_ready_queue_ready.iterator + queue_idx,
                        lambda ready: ready != cutlass.Int32(0),
                        fail_sleep_cycles=500,
                    )
                cute.arch.sync_warp()
                cute.arch.fence_acq_rel_gpu()
                dequeue_end = cutlass.Int64(0)
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if cute.arch.lane_idx() == cutlass.Int32(0):
                        dequeue_end = read_globaltimer()

                expert_idx = cutlass.Int32(0)
                tile_n_idx = cutlass.Int32(0)
                cumulative_data = cutlass.Int32(0)
                cumulative_sf = cutlass.Int32(0)
                cumulative_token_block = cutlass.Int32(0)
                valid_tokens = cutlass.Int32(0)
                hidden_bundle_begin = cutlass.Int32(0)
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    desc_base = queue_idx * cutlass.Int32(7)
                    expert_idx = k2_ready_queue_desc[desc_base + cutlass.Int32(0)]
                    tile_n_idx = k2_ready_queue_desc[desc_base + cutlass.Int32(1)]
                    cumulative_data = k2_ready_queue_desc[desc_base + cutlass.Int32(2)]
                    cumulative_sf = k2_ready_queue_desc[desc_base + cutlass.Int32(3)]
                    cumulative_token_block = k2_ready_queue_desc[
                        desc_base + cutlass.Int32(4)
                    ]
                    valid_tokens = k2_ready_queue_desc[desc_base + cutlass.Int32(5)]
                    hidden_bundle_begin = k2_ready_queue_desc[
                        desc_base + cutlass.Int32(6)
                    ]
                expert_idx = cute.arch.shuffle_sync(
                    expert_idx, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                tile_n_idx = cute.arch.shuffle_sync(
                    tile_n_idx, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                cumulative_data = cute.arch.shuffle_sync(
                    cumulative_data, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                cumulative_sf = cute.arch.shuffle_sync(
                    cumulative_sf, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                cumulative_token_block = cute.arch.shuffle_sync(
                    cumulative_token_block,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
                valid_tokens = cute.arch.shuffle_sync(
                    valid_tokens, offset=0, mask=0xFFFFFFFF, mask_and_clamp=31
                )
                hidden_bundle_begin = cute.arch.shuffle_sync(
                    hidden_bundle_begin,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )

                if expert_idx < cutlass.Int32(0):
                    queue_running = cutlass.Boolean(False)
                else:
                    ext.prefetch_for_expert(expert_idx)
                    hidden_tiles = (
                        self.hidden + self.mma_tiler[0] - 1
                    ) // self.mma_tiler[0]
                    for bundle_tile in cutlass.range_constexpr(
                        0, self.k2_ready_queue_bundle
                    ):
                        hidden_tile = (
                            hidden_bundle_begin + cutlass.Int32(bundle_tile)
                        )
                        if hidden_tile < cutlass.Int32(hidden_tiles):
                            if cutlass.const_expr(
                                self.k2_tile_trace_enabled
                                and green_trace is not None
                            ):
                                if cute.arch.lane_idx() == cutlass.Int32(0):
                                    if (
                                        sched_tile_seq
                                        < cutlass.Int32(
                                            K2_TILE_TRACE_TILES_PER_CTA
                                        )
                                    ):
                                        tile_dequeue_begin = dequeue_end
                                        if cutlass.const_expr(bundle_tile == 0):
                                            tile_dequeue_begin = dequeue_begin
                                        record_base = k2_tile_trace_word(
                                            cta_linear_id,
                                            sched_tile_seq,
                                            cutlass.Int32(0),
                                        )
                                        cute.arch.store(
                                            green_trace.iterator
                                            + record_base
                                            + K2_TILE_FIELD_TILE_ID,
                                            pack_tile_id(
                                                expert_idx,
                                                tile_n_idx,
                                                hidden_tile,
                                            ),
                                            scope="gpu",
                                        )
                                        cute.arch.store(
                                            green_trace.iterator
                                            + record_base
                                            + K2_TILE_FIELD_DEQUEUE_BEGIN,
                                            tile_dequeue_begin,
                                            scope="gpu",
                                        )
                                        cute.arch.store(
                                            green_trace.iterator
                                            + record_base
                                            + K2_TILE_FIELD_DEQUEUE_END,
                                            dequeue_end,
                                            scope="gpu",
                                        )
                            scheduler.current_work = ext.enrich_work_tile_info(
                                ext.WorkTileInfo(
                                    expert_idx=expert_idx,
                                    tile_m_idx=hidden_tile,
                                    tile_n_idx=tile_n_idx,
                                    cumulative_data_physical_row=cumulative_data,
                                    cumulative_sf_physical_row=cumulative_sf,
                                    cumulative_token_block_count=(
                                        cumulative_token_block
                                    ),
                                    valid_tokens_in_tile=valid_tokens,
                                    phase_and_peek=cutlass.Int32(
                                        BlockPhase.Linear2
                                    ),
                                    fc1_counter_index=tile_n_idx,
                                    valid_tokens_in_cluster=valid_tokens,
                                )
                            )
                            iket.range_push("schedule_fc2_queue_tile")
                            scheduler.publish_work()
                            iket.range_pop()
                            sched_tile_seq += cutlass.Int32(1)
            scheduler.current_work = ext.WorkTileInfo(
                expert_idx=cutlass.Int32(WorkTileState.DONE),
                tile_m_idx=cutlass.Int32(0),
                tile_n_idx=cutlass.Int32(0),
                cumulative_data_physical_row=cutlass.Int32(0),
                cumulative_sf_physical_row=cutlass.Int32(0),
                cumulative_token_block_count=cutlass.Int32(0),
                valid_tokens_in_tile=cutlass.Int32(0),
                phase_and_peek=cutlass.Int32(BlockPhase.None_),
                fc1_counter_index=cutlass.Int32(0),
                valid_tokens_in_cluster=cutlass.Int32(0),
            )
            scheduler.publish_work()
            scheduler.produce_tail()
        if cutlass.const_expr(self.enable_token_comm and self.use_warpgroup_reg_realloc):
            if warp_idx == self.sm120_aux_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
        if warp_idx == self.tma_a_warp_id:
            if cutlass.const_expr(self.enable_token_comm and self.use_warpgroup_reg_realloc):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2)
            a_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape)
            sfa_cta_layout = a_cta_layout
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                iket.range_push('tma_weight_fc2')
                k_tile_cnt = k_tile_cnt_fc2
                real_a, desc_ptr_a = ext.get_gmem_tensor('a', tma_tensor_fc2_weight, work_tile_info)
                if cutlass.const_expr(UseScaleTma):
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor('sfa', tma_tensor_fc2_weight_sf, work_tile_info)
                gA_mkl = cute.local_tile(real_a, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None))
                tCgA = thr_mma.partition_A(gA_mkl)
                if cutlass.const_expr(UseScaleTma):
                    gSFA_mkl = cute.local_tile(real_sfa, cute.slice_(self.mma_tiler_sfa, (None, 0, None)), (None, None, None))
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)
                tAsA, tAgA = cpasync.tma_partition(tma_atom_fc2_weight, block_in_cluster_coord_vmnk[2], a_cta_layout, cute.group_modes(sA, 0, 2), cute.group_modes(gA_mkl, 0, 2))
                if cutlass.const_expr(UseScaleTma):
                    tAsSFA, tAgSFA = cpasync.tma_partition(tma_atom_fc2_weight_sf, block_in_cluster_coord_vmnk[2], sfa_cta_layout, cute.group_modes(sSFA, 0, 2), cute.group_modes(gSFA_mkl, 0, 2))
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)
                mma_tile_m = work_tile_info.tile_m_idx
                tAgA_slice = tAgA[None, mma_tile_m, None, 0]
                if cutlass.const_expr(UseScaleTma):
                    sfa_tile_m = mma_tile_m
                    if cutlass.const_expr(self.mma_tiler_sfa[0] != self.mma_tiler[0]):
                        sfa_tile_m = mma_tile_m // cutlass.Int32(self.mma_tiler_sfa[0] // self.mma_tiler[0])
                    tAgSFA_slice = tAgSFA[None, sfa_tile_m, None, 0]
                a_producer.reset()
                peek_ab_empty_status = a_producer.try_acquire()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    iket.range_push('tma_weight_fc2_empty_wait')
                    handle = a_producer.acquire_and_advance(peek_ab_empty_status)
                    iket.range_pop()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = a_producer.try_acquire()
                    iket.range_push('tma_weight_fc2_issue')
                    cute.copy(tma_atom_fc2_weight, tAgA_slice[None, handle.count], tAsA[None, handle.index], tma_bar_ptr=handle.barrier, tma_desc_ptr=desc_ptr_a, mcast_mask=a_full_mcast_mask)
                    if cutlass.const_expr(UseScaleTma):
                        cute.copy(tma_atom_fc2_weight_sf, tAgSFA_slice[None, handle.count], tAsSFA[None, handle.index], tma_bar_ptr=handle.barrier, tma_desc_ptr=desc_ptr_sfa, mcast_mask=sfa_full_mcast_mask)
                    iket.range_pop()
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()
            a_producer.tail()
        if warp_idx == self.tma_b_warp_id:
            if cutlass.const_expr(self.enable_token_comm and self.use_warpgroup_reg_realloc):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)
            b_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1)
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(cluster_layout_sfb_vmnk, block_in_cluster_coord_sfb_vmnk, mcast_mode=1)
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            sfb_cta_layout = cute.make_layout(cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape)
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)
            fc2_spin_threshold = ext_fc2_spin_threshold
            work_tile_info = sched_consumer.consume_work()
            k2_ready_wait_calls = cutlass.Int64(0)
            k2_ready_wait_ns = cutlass.Int64(0)
            while work_tile_info.is_valid_tile:
                iket.range_push('tma_token_fc2')
                counter_slot = (
                    work_tile_info.cumulative_token_block_count
                    + work_tile_info.tile_n_idx
                    * cutlass.Int32(
                        self.mma_tiler[1]
                        // self.fc1_ready_tile_tokens
                    )
                )
                if cutlass.const_expr(token_comm_args is None):
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    iket.range_push('tma_token_fc2_wait')
                    spin_wait(counter_ptr, lambda v: v >= fc2_spin_threshold, fail_sleep_cycles=500)
                    iket.range_pop()
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.fence_proxy('async.global')
                k_tile_cnt = k_tile_cnt_fc2
                real_b, desc_ptr_b = ext.get_gmem_tensor('b', tma_tensor_fc1_output_as_fc2_input, work_tile_info)
                if cutlass.const_expr(UseScaleTma):
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor('sfb', tma_tensor_fc1_output_sf_as_fc2_input, work_tile_info)
                gB_nkl = cute.local_tile(real_b, cute.slice_(self.mma_tiler, (0, None, None)), (None, None, None))
                tCgB = thr_mma.partition_B(gB_nkl)
                if cutlass.const_expr(UseScaleTma):
                    gSFB_nkl = cute.local_tile(real_sfb, cute.slice_(self.mma_tiler_sfb, (0, None, None)), (None, None, None))
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)
                tBsB, tBgB = cpasync.tma_partition(tma_atom_fc1_output_as_fc2_input, block_in_cluster_coord_vmnk[1], b_cta_layout, cute.group_modes(sB, 0, 2), cute.group_modes(gB_nkl, 0, 2))
                if cutlass.const_expr(UseScaleTma):
                    tBsSFB, tBgSFB = cpasync.tma_partition(tma_atom_fc1_output_sf_as_fc2_input, block_in_cluster_coord_sfb_vmnk[1], sfb_cta_layout, cute.group_modes(sSFB, 0, 2), cute.group_modes(gSFB_nkl, 0, 2))
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)
                tBgB_slice = tBgB[None, work_tile_info.tile_n_idx, None, 0]
                if cutlass.const_expr(UseScaleTma):
                    sfb_tile_n_idx = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.mma_tiler[1] < 128):
                        sfb_tile_n_idx = work_tile_info.tile_n_idx // cutlass.Int32(128 // self.mma_tiler[1])
                    tBgSFB_slice = tBgSFB[None, sfb_tile_n_idx, None, 0]
                b_producer.reset()
                peek_ab_empty_status = b_producer.try_acquire()
                iket.range_push('tma_token_fc2_issue')
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    if cutlass.const_expr(token_comm_args is not None):
                        if k_tile % cutlass.Int32(self.fc2_ready_bundle_k_tiles) == cutlass.Int32(0):
                            ready_wait_start = cutlass.Int64(0)
                            if cutlass.const_expr(green_trace is not None):
                                if cute.arch.lane_idx() == cutlass.Int32(0):
                                    ready_wait_start = read_globaltimer()
                                    k2_ready_wait_calls += cutlass.Int64(1)
                            iket.range_push('tma_token_fc2_bundle_wait')
                            if cutlass.const_expr(self.fc2_ready_wait_enabled):
                                bundle_idx = k_tile // cutlass.Int32(self.fc2_ready_bundle_k_tiles)
                                bundle_begin = bundle_idx * cutlass.Int32(self.fc2_ready_bundle_k_tiles)
                                bundle_k_tiles = cutlass.min(cutlass.Int32(self.fc2_ready_bundle_k_tiles), k_tile_cnt_fc2 - bundle_begin)
                                producer_tiles_in_ready_block = (
                                    work_tile_info.valid_tokens_in_tile
                                    + cutlass.Int32(
                                        self.fc1_ready_tile_tokens - 1
                                    )
                                ) // cutlass.Int32(
                                    self.fc1_ready_tile_tokens
                                )
                                if cute.arch.lane_idx() == cutlass.Int32(0):
                                    spin_wait_i32_ge_inline(
                                        fc1_done_counter.iterator
                                        + counter_slot
                                        * fc2_ready_bundle_cnt
                                        + bundle_idx,
                                        bundle_k_tiles
                                        * cutlass.Int32(
                                            fc1_tiles_per_fc2_k_tile
                                        )
                                        * producer_tiles_in_ready_block,
                                        fail_sleep_cycles=500,
                                    )
                                cute.arch.sync_warp()
                            iket.range_pop()
                            if cutlass.const_expr(green_trace is not None):
                                if cute.arch.lane_idx() == cutlass.Int32(0):
                                    k2_ready_wait_ns += read_globaltimer() - ready_wait_start
                            cute.arch.fence_acq_rel_gpu()
                            cute.arch.fence_proxy('async.global')
                    handle = b_producer.acquire_and_advance(peek_ab_empty_status)
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if handle.count + 1 < k_tile_cnt:
                        peek_ab_empty_status = b_producer.try_acquire()
                    cute.copy(tma_atom_fc1_output_as_fc2_input, tBgB_slice[None, handle.count], tBsB[None, handle.index], tma_bar_ptr=handle.barrier, tma_desc_ptr=desc_ptr_b, mcast_mask=b_full_mcast_mask)
                    if cutlass.const_expr(UseScaleTma):
                        cute.copy(tma_atom_fc1_output_sf_as_fc2_input, tBgSFB_slice[None, handle.count], tBsSFB[None, handle.index], tma_bar_ptr=handle.barrier, tma_desc_ptr=desc_ptr_sfb, mcast_mask=sfb_full_mcast_mask)
                iket.range_pop()
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()
            if cutlass.const_expr(green_trace is not None):
                if cute.arch.lane_idx() == cutlass.Int32(0):
                    cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_READY_WAIT_CALLS), k2_ready_wait_calls, scope='gpu')
                    cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_READY_WAIT_NS), k2_ready_wait_ns, scope='gpu')
            b_producer.tail()
        if warp_idx < len(self.compute_warp_id):
            if cutlass.const_expr(self.enable_token_comm and self.use_warpgroup_reg_realloc):
                cute.arch.warpgroup_reg_alloc(self.epi_reg_cnt)
            compute_warp = warp_idx
            compute_m_warp = compute_warp
            lane_idx = cute.arch.lane_idx()
            lane_g = lane_idx >> cutlass.Int32(2)
            lane_t = lane_idx & cutlass.Int32(3)
            n_groups = self.mma_tiler[1] // MMA_N
            rFc2Store = cute.make_rmem_tensor((8,), self.fc2_output_dtype)
            rFc2StoreI32 = cute.recast_tensor(rFc2Store, cutlass.Int32)
            mma_tidx = tidx
            compute_tiled_mma = tiled_mma
            thr_mma = compute_tiled_mma.get_slice(mma_tidx)
            tCsA = thr_mma.partition_A(sA)
            sB_compute = sB
            sSFB_compute = sSFB
            tCsB = thr_mma.partition_B(sB_compute)
            tCrA = compute_tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = compute_tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrSFA = sm100_utils.partition_fragment_SFA(
                sSFA[None, None, 0], thr_mma, mma_tidx
            )
            tCrSFB = sm100_utils.partition_fragment_SFB(
                sSFB_compute[None, None, 0], thr_mma, mma_tidx
            )
            atom_copy_ldmatrix_A = make_sm120_ldmatrix_atom(
                self.a_dtype,
                transpose=self.a_layout.is_m_major_a(),
                mixed_mode=self.mixed_mode,
            )
            atom_copy_ldmatrix_B = make_sm120_ldmatrix_atom(self.b_dtype, transpose=self.b_layout.is_n_major_b())
            smem_tiled_copy_A = cute.make_tiled_copy_A(
                atom_copy_ldmatrix_A, compute_tiled_mma
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(
                atom_copy_ldmatrix_B, compute_tiled_mma
            )
            atom_copy_scale = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), self.sf_dtype)
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_scale,
                sm100_utils.get_layoutSFA_TV(compute_tiled_mma),
                (
                    cute.size(compute_tiled_mma.permutation_mnk[0]),
                    cute.size(compute_tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_scale,
                sm100_utils.get_layoutSFB_TV(compute_tiled_mma),
                (
                    cute.size(compute_tiled_mma.permutation_mnk[1]),
                    cute.size(compute_tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(mma_tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(mma_tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB_compute)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            thr_copy_SFA = smem_tiled_copy_SFA.get_slice(mma_tidx)
            thr_copy_SFB = smem_tiled_copy_SFB.get_slice(mma_tidx)
            tCsSFA_copy_view = thr_copy_SFA.partition_S(sSFA)
            tCsSFB_copy_view = thr_copy_SFB.partition_S(sSFB_compute)
            tCrSFA_copy_view = thr_copy_SFA.retile(tCrSFA)
            tCrSFB_copy_view = thr_copy_SFB.retile(tCrSFB)
            acc_shape = compute_tiled_mma.partition_shape_C(
                (self.mma_tiler[0], self.mma_tiler[1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            work_tile_info = sched_consumer.consume_work()
            fc2_detail_tiles_seen = cutlass.Int32(0)
            green_trace_seen_work = cutlass.Int32(0)
            k2_trace_tile_count = cutlass.Int64(0)
            k2_trace_first_tile_id = cutlass.Int64(0)
            k2_trace_last_tile_id = cutlass.Int64(0)
            k2_trace_tma_a_wait_calls = cutlass.Int64(0)
            k2_trace_tma_a_wait_ns = cutlass.Int64(0)
            k2_trace_tma_a_timed_calls = cutlass.Int64(0)
            k2_trace_tma_b_wait_calls = cutlass.Int64(0)
            k2_trace_tma_b_wait_ns = cutlass.Int64(0)
            k2_trace_tma_b_timed_calls = cutlass.Int64(0)
            k2_trace_mainloop_ns = cutlass.Int64(0)
            k2_trace_store_ns = cutlass.Int64(0)
            compute_tile_seq = cutlass.Int32(0)
            while work_tile_info.is_valid_tile:
                tile_begin = cutlass.Int64(0)
                tile_tma_a_wait_ns = cutlass.Int64(0)
                tile_tma_b_wait_ns = cutlass.Int64(0)
                tile_ldsm_qmma_ns = cutlass.Int64(0)
                tile_bf16_pack_ns = cutlass.Int64(0)
                tile_peer_store_ns = cutlass.Int64(0)
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            tile_begin = read_globaltimer()
                            if (
                                compute_tile_seq
                                < cutlass.Int32(K2_TILE_TRACE_TILES_PER_CTA)
                            ):
                                tile_record_base = k2_tile_trace_word(
                                    cta_linear_id,
                                    compute_tile_seq,
                                    cutlass.Int32(0),
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_TILE_ID,
                                    pack_tile_id(
                                        work_tile_info.expert_idx,
                                        work_tile_info.tile_n_idx,
                                        work_tile_info.tile_m_idx,
                                    ),
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_TILE_BEGIN,
                                    tile_begin,
                                    scope="gpu",
                                )
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            if green_trace_seen_work == cutlass.Int32(0):
                                cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_FIRST_WORK), read_globaltimer(), scope='gpu')
                                green_trace_seen_work = cutlass.Int32(1)
                k_tile_cnt = k_tile_cnt_fc2
                iket.range_push('sm120_fc2_tile')
                iket.range_push('sm120_fc2_mainloop')
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            packed_tile_id = pack_tile_id(work_tile_info.expert_idx, work_tile_info.tile_n_idx, work_tile_info.tile_m_idx)
                            if k2_trace_tile_count == cutlass.Int64(0):
                                k2_trace_first_tile_id = packed_tile_id
                            k2_trace_last_tile_id = packed_tile_id
                            k2_trace_tile_count += cutlass.Int64(1)
                trace_compute_detail = cutlass.Int32(0)
                trace_all_k_detail = cutlass.Int32(0)
                if bidx == cutlass.Int32(0):
                    if fc2_detail_tiles_seen == cutlass.Int32(0):
                        trace_compute_detail = cutlass.Int32(1)
                    if bidz == cutlass.Int32(0) and fc2_detail_tiles_seen < cutlass.Int32(4):
                        trace_compute_detail = cutlass.Int32(1)
                        trace_all_k_detail = cutlass.Int32(1)
                k2_trace_mainloop_start = cutlass.Int64(0)
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            k2_trace_mainloop_start = read_globaltimer()
                accumulators.fill(0.0)
                a_consumer.reset()
                b_consumer.reset()
                peek_a_full_status = cutlass.Boolean(1)
                peek_b_full_status = cutlass.Boolean(1)
                if k_tile_cnt > 0:
                    peek_a_full_status = a_consumer.try_wait()
                    peek_b_full_status = b_consumer.try_wait()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    trace_k_detail = cutlass.Int32(0)
                    if trace_compute_detail != cutlass.Int32(0):
                        if trace_all_k_detail != cutlass.Int32(0):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile < cutlass.Int32(2):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile == cutlass.Int32(11):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile == cutlass.Int32(12):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile == cutlass.Int32(23):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile == cutlass.Int32(24):
                            trace_k_detail = cutlass.Int32(1)
                        if k_tile + cutlass.Int32(2) >= k_tile_cnt:
                            trace_k_detail = cutlass.Int32(1)
                    if trace_k_detail != cutlass.Int32(0):
                        iket.range_push('sm120_fc2_wait_a')
                    k2_trace_a_wait_start = cutlass.Int64(0)
                    k2_trace_a_wait_active = cutlass.Boolean(0)
                    tile_a_wait_start = cutlass.Int64(0)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_a_wait_start = read_globaltimer()
                    if cutlass.const_expr(green_trace is not None):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                if peek_a_full_status == cutlass.Boolean(0):
                                    k2_trace_tma_a_wait_calls += cutlass.Int64(1)
                                    if (k_tile + fc2_detail_tiles_seen + cta_linear_id) % cutlass.Int32(TRACE_TMA_WAIT_SAMPLE_STRIDE) == cutlass.Int32(0):
                                        k2_trace_a_wait_start = read_globaltimer()
                                        k2_trace_a_wait_active = cutlass.Boolean(1)
                                        k2_trace_tma_a_timed_calls += cutlass.Int64(1)
                    handle_a = a_consumer.wait_and_advance(peek_a_full_status)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_tma_a_wait_ns += (
                                    read_globaltimer() - tile_a_wait_start
                                )
                    if cutlass.const_expr(green_trace is not None):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                if k2_trace_a_wait_active:
                                    k2_trace_tma_a_wait_ns += read_globaltimer() - k2_trace_a_wait_start
                    if trace_k_detail != cutlass.Int32(0):
                        iket.range_pop()
                        iket.range_push('sm120_fc2_wait_b')
                    k2_trace_b_wait_start = cutlass.Int64(0)
                    k2_trace_b_wait_active = cutlass.Boolean(0)
                    tile_b_wait_start = cutlass.Int64(0)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_b_wait_start = read_globaltimer()
                    if cutlass.const_expr(green_trace is not None):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                if peek_b_full_status == cutlass.Boolean(0):
                                    k2_trace_tma_b_wait_calls += cutlass.Int64(1)
                                    if (k_tile + fc2_detail_tiles_seen + cta_linear_id) % cutlass.Int32(TRACE_TMA_WAIT_SAMPLE_STRIDE) == cutlass.Int32(0):
                                        k2_trace_b_wait_start = read_globaltimer()
                                        k2_trace_b_wait_active = cutlass.Boolean(1)
                                        k2_trace_tma_b_timed_calls += cutlass.Int64(1)
                    handle_b = b_consumer.wait_and_advance(peek_b_full_status)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_tma_b_wait_ns += (
                                    read_globaltimer() - tile_b_wait_start
                                )
                    if cutlass.const_expr(green_trace is not None):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                if k2_trace_b_wait_active:
                                    k2_trace_tma_b_wait_ns += read_globaltimer() - k2_trace_b_wait_start
                    if trace_k_detail != cutlass.Int32(0):
                        iket.range_pop()
                    peek_a_full_status = cutlass.Boolean(1)
                    peek_b_full_status = cutlass.Boolean(1)
                    if handle_a.count + 1 < k_tile_cnt:
                        peek_a_full_status = a_consumer.try_wait()
                    if handle_b.count + 1 < k_tile_cnt:
                        peek_b_full_status = b_consumer.try_wait()
                    tCsA_p = tCsA_copy_view[None, None, None, handle_a.index]
                    tCsB_p = tCsB_copy_view[None, None, None, handle_b.index]
                    tCsSFA_p = tCsSFA_copy_view[None, None, None, handle_a.index]
                    tCsSFB_p = tCsSFB_copy_view[None, None, None, handle_b.index]
                    sfa_m_group = work_tile_info.tile_m_idx % cutlass.Int32(self.mma_tiler_sfa[0] // self.mma_tiler[0])
                    tCsSFA_selected = cute.make_tensor(tCsSFA_p.iterator + sfa_m_group * cutlass.Int32(8), tCsSFA_p.layout)
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_selected)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                    tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                    tile_compute_start = cutlass.Int64(0)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_compute_start = read_globaltimer()
                    cute.copy(smem_tiled_copy_A, tCsA_p[None, None, 0], tCrA_copy_view[None, None, 0])
                    if cutlass.const_expr(self.mixed_mode):
                        shift_fp4_fragment_for_mxf8f6f4(tCrA[None, None, 0])
                    cute.copy(smem_tiled_copy_B, tCsB_p[None, None, 0], tCrB_copy_view[None, None, 0])
                    cute.copy(smem_tiled_copy_SFA, tCsSFA_p_filtered[None, 0, 0], tCrSFA_copy_view_filtered[None, 0, 0])
                    cute.copy(smem_tiled_copy_SFB, tCsSFB_p_filtered[None, None, 0], tCrSFB_copy_view_filtered[None, 0, 0, None])
                    # SFB is staged as N128. Select the current token sub-tile
                    # with uniform branches so every register-fragment index
                    # remains compile-time constant, including the N16 path.
                    sfb_tiles_per_tma = 128 // self.mma_tiler[1]
                    sfb_tile_slot = (
                        work_tile_info.tile_n_idx
                        % cutlass.Int32(sfb_tiles_per_tma)
                    )
                    if trace_k_detail != cutlass.Int32(0):
                        iket.range_push('sm120_fc2_k128_compute')
                    for k_inner_mma in cutlass.range_constexpr(0, 4):
                        if cutlass.const_expr(k_inner_mma + 1 < 4):
                            k_inner_next = k_inner_mma + 1
                            cute.copy(smem_tiled_copy_A, tCsA_p[None, None, k_inner_next], tCrA_copy_view[None, None, k_inner_next])
                            if cutlass.const_expr(self.mixed_mode):
                                shift_fp4_fragment_for_mxf8f6f4(
                                    tCrA[None, None, k_inner_next]
                                )
                            cute.copy(smem_tiled_copy_B, tCsB_p[None, None, k_inner_next], tCrB_copy_view[None, None, k_inner_next])
                            cute.copy(smem_tiled_copy_SFA, tCsSFA_p_filtered[None, 0, k_inner_next], tCrSFA_copy_view_filtered[None, 0, k_inner_next])
                            cute.copy(smem_tiled_copy_SFB, tCsSFB_p_filtered[None, None, k_inner_next], tCrSFB_copy_view_filtered[None, 0, k_inner_next, None])
                        for sfb_slot in cutlass.range_constexpr(
                            0, sfb_tiles_per_tma
                        ):
                            if sfb_tile_slot == cutlass.Int32(sfb_slot):
                                for ng in cutlass.range_constexpr(0, n_groups):
                                    issue_m64n8k32_mxfp8(
                                        compute_tiled_mma,
                                        accumulators[None, None, ng],
                                        tCrA,
                                        tCrB,
                                        tCrSFA,
                                        tCrSFB,
                                        n_group=ng,
                                        active_n_groups=n_groups,
                                        sfb_n_group=sfb_slot * n_groups + ng,
                                        sfa_m_group=0,
                                        k_inner=k_inner_mma,
                                        a_dtype=self.a_dtype,
                                        b_dtype=self.b_dtype,
                                        sf_dtype=self.sf_dtype,
                                    )
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_ldsm_qmma_ns += (
                                    read_globaltimer() - tile_compute_start
                                )
                    if trace_k_detail != cutlass.Int32(0):
                        iket.range_pop()
                    handle_a.release()
                    handle_b.release()
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            k2_trace_mainloop_ns += read_globaltimer() - k2_trace_mainloop_start
                iket.range_pop()
                iket.range_push('sm120_fc2_store')
                k2_trace_store_start = cutlass.Int64(0)
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            k2_trace_store_start = read_globaltimer()
                hidden_base = work_tile_info.tile_m_idx * cutlass.Int32(self.mma_tiler[0])
                tile_token_base = work_tile_info.tile_n_idx * cutlass.Int32(self.mma_tiler[1])
                for ng in cutlass.range_constexpr(0, n_groups):
                    acc = accumulators[None, None, ng]
                    token0 = cutlass.Int32(ng * MMA_N) + lane_t * cutlass.Int32(2)
                    token1 = token0 + cutlass.Int32(1)
                    valid_token0 = token0
                    valid_token1 = token1
                    hidden0 = (
                        hidden_base
                        + compute_m_warp * cutlass.Int32(16)
                        + lane_g
                    )
                    hidden1 = hidden0 + cutlass.Int32(8)
                    pool_token0 = work_tile_info.cumulative_data_physical_row + tile_token_base + token0
                    pool_token1 = work_tile_info.cumulative_data_physical_row + tile_token_base + token1
                    tile_pack_start = cutlass.Int64(0)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_pack_start = read_globaltimer()
                    if cutlass.const_expr(self.fc2_packed_store):
                        partner_lane = lane_idx ^ cutlass.Int32(4)
                        rFc2Store[0] = acc[0].to(self.fc2_output_dtype)
                        rFc2Store[1] = cute.arch.shuffle_sync(acc[0], partner_lane).to(self.fc2_output_dtype)
                        rFc2Store[2] = acc[2].to(self.fc2_output_dtype)
                        rFc2Store[3] = cute.arch.shuffle_sync(acc[2], partner_lane).to(self.fc2_output_dtype)
                        rFc2Store[4] = acc[1].to(self.fc2_output_dtype)
                        rFc2Store[5] = cute.arch.shuffle_sync(acc[1], partner_lane).to(self.fc2_output_dtype)
                        rFc2Store[6] = acc[3].to(self.fc2_output_dtype)
                        rFc2Store[7] = cute.arch.shuffle_sync(acc[3], partner_lane).to(self.fc2_output_dtype)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_bf16_pack_ns += (
                                    read_globaltimer() - tile_pack_start
                                )
                    tile_peer_store_start = cutlass.Int64(0)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_peer_store_start = read_globaltimer()
                    if valid_token0 < work_tile_info.valid_tokens_in_tile:
                        fc2_store_token0 = pool_token0
                        if cutlass.const_expr(self.ibgda_k2_direct_staging):
                            fc2_store_token0 = cutlass.Int32(
                                token_comm_args.combine_sf[
                                    cutlass.Int32(
                                        self.ibgda_direct_stage_map_offset_i32
                                    )
                                    + pool_token0
                                ]
                            )
                        if cutlass.const_expr(self.fc2_packed_store):
                            if lane_g & cutlass.Int32(1) == cutlass.Int32(0):
                                if cutlass.const_expr(token_comm_args is not None and (not self.token_back_by_dispatch)):
                                    md0 = TokenSrcMetadata.load(token_comm_args.token_src_metadata.iterator.toint() + cutlass.Int64(pool_token0) * cutlass.Int64(TokenSrcMetadata.nbytes))
                                    local_row0 = cute.slice_(fc2_output, (md0.src_token, md0.src_topk, None))
                                    row0 = cute.make_tensor(token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(local_row0.iterator, md0.src_rank), local_row0.layout)
                                else:
                                    row0 = cute.slice_(fc2_output, (fc2_store_token0, 0, None))
                                row0_hidden0 = (row0.iterator + hidden0).align(4)
                                row0_hidden1 = (row0.iterator + hidden1).align(4)
                                row0_i32_0 = cute.make_tensor(cute.recast_ptr(row0_hidden0, dtype=cutlass.Int32), cute.make_layout(1))
                                row0_i32_1 = cute.make_tensor(cute.recast_ptr(row0_hidden1, dtype=cutlass.Int32), cute.make_layout(1))
                                row0_i32_0[0] = rFc2StoreI32[0]
                                row0_i32_1[0] = rFc2StoreI32[1]
                        elif cutlass.const_expr(token_comm_args is not None and (not self.token_back_by_dispatch)):
                            md0 = TokenSrcMetadata.load(token_comm_args.token_src_metadata.iterator.toint() + cutlass.Int64(pool_token0) * cutlass.Int64(TokenSrcMetadata.nbytes))
                            local_row0 = cute.slice_(fc2_output, (md0.src_token, md0.src_topk, None))
                            peer_row0 = cute.make_tensor(token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(local_row0.iterator, md0.src_rank), local_row0.layout)
                            peer_row0[hidden0] = acc[0].to(self.fc2_output_dtype)
                            peer_row0[hidden1] = acc[2].to(self.fc2_output_dtype)
                        else:
                            fc2_output[fc2_store_token0, 0, hidden0] = acc[0].to(self.fc2_output_dtype)
                            fc2_output[fc2_store_token0, 0, hidden1] = acc[2].to(self.fc2_output_dtype)
                    if valid_token1 < work_tile_info.valid_tokens_in_tile:
                        fc2_store_token1 = pool_token1
                        if cutlass.const_expr(self.ibgda_k2_direct_staging):
                            fc2_store_token1 = cutlass.Int32(
                                token_comm_args.combine_sf[
                                    cutlass.Int32(
                                        self.ibgda_direct_stage_map_offset_i32
                                    )
                                    + pool_token1
                                ]
                            )
                        if cutlass.const_expr(self.fc2_packed_store):
                            if lane_g & cutlass.Int32(1) == cutlass.Int32(0):
                                if cutlass.const_expr(token_comm_args is not None and (not self.token_back_by_dispatch)):
                                    md1 = TokenSrcMetadata.load(token_comm_args.token_src_metadata.iterator.toint() + cutlass.Int64(pool_token1) * cutlass.Int64(TokenSrcMetadata.nbytes))
                                    local_row1 = cute.slice_(fc2_output, (md1.src_token, md1.src_topk, None))
                                    row1 = cute.make_tensor(token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(local_row1.iterator, md1.src_rank), local_row1.layout)
                                else:
                                    row1 = cute.slice_(fc2_output, (fc2_store_token1, 0, None))
                                row1_hidden0 = (row1.iterator + hidden0).align(4)
                                row1_hidden1 = (row1.iterator + hidden1).align(4)
                                row1_i32_0 = cute.make_tensor(cute.recast_ptr(row1_hidden0, dtype=cutlass.Int32), cute.make_layout(1))
                                row1_i32_1 = cute.make_tensor(cute.recast_ptr(row1_hidden1, dtype=cutlass.Int32), cute.make_layout(1))
                                row1_i32_0[0] = rFc2StoreI32[2]
                                row1_i32_1[0] = rFc2StoreI32[3]
                        elif cutlass.const_expr(token_comm_args is not None and (not self.token_back_by_dispatch)):
                            md1 = TokenSrcMetadata.load(token_comm_args.token_src_metadata.iterator.toint() + cutlass.Int64(pool_token1) * cutlass.Int64(TokenSrcMetadata.nbytes))
                            local_row1 = cute.slice_(fc2_output, (md1.src_token, md1.src_topk, None))
                            peer_row1 = cute.make_tensor(token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(local_row1.iterator, md1.src_rank), local_row1.layout)
                            peer_row1[hidden0] = acc[1].to(self.fc2_output_dtype)
                            peer_row1[hidden1] = acc[3].to(self.fc2_output_dtype)
                        else:
                            fc2_output[fc2_store_token1, 0, hidden0] = acc[1].to(self.fc2_output_dtype)
                            fc2_output[fc2_store_token1, 0, hidden1] = acc[3].to(self.fc2_output_dtype)
                    if cutlass.const_expr(
                        self.k2_tile_trace_enabled and green_trace is not None
                    ):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                tile_peer_store_ns += (
                                    read_globaltimer() - tile_peer_store_start
                                )
                tile_peer_finalize_start = cutlass.Int64(0)
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            tile_peer_finalize_start = read_globaltimer()
                if cutlass.const_expr(combine_ready_flags is not None and fc2_block_done_counter is not None):
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=32 * len(self.compute_warp_id))
                    self.token_comm_hook_fc2_tile_complete(token_comm_args, combine_ready_flags, fc2_block_done_counter, work_tile_info, compute_warp=compute_warp, lane_idx=lane_idx)
                if cutlass.const_expr(self.token_back_by_dispatch):
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(barrier_id=self.epilog_sync_bar_id, number_of_threads=32 * len(self.compute_warp_id))
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            counter_slot = work_tile_info.expert_idx
                            if cutlass.const_expr(self.streaming_fc12):
                                counter_slot = work_tile_info.cumulative_data_physical_row // cutlass.Int32(self.mma_tiler[1]) + work_tile_info.tile_n_idx
                            cute.arch.atomic_add(token_comm_args.fc2_done_counter.iterator + counter_slot, cutlass.Int32(1), sem='release', scope='sys')
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            tile_peer_store_ns += (
                                read_globaltimer() - tile_peer_finalize_start
                            )
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            k2_trace_store_ns += read_globaltimer() - k2_trace_store_start
                fc2_detail_tiles_seen += cutlass.Int32(1)
                if cutlass.const_expr(green_trace is not None):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_LAST_WORK), read_globaltimer(), scope='gpu')
                iket.range_pop()
                iket.range_pop()
                tile_phase_start = cutlass.Int64(0)
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            tile_phase_start = read_globaltimer()
                next_work_tile_info = sched_consumer.consume_work()
                if cutlass.const_expr(
                    self.k2_tile_trace_enabled and green_trace is not None
                ):
                    if compute_warp == cutlass.Int32(0):
                        if lane_idx == cutlass.Int32(0):
                            tile_phase_end = read_globaltimer()
                            if (
                                compute_tile_seq
                                < cutlass.Int32(K2_TILE_TRACE_TILES_PER_CTA)
                            ):
                                tile_record_base = k2_tile_trace_word(
                                    cta_linear_id,
                                    compute_tile_seq,
                                    cutlass.Int32(0),
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_TMA_A_WAIT_NS,
                                    tile_tma_a_wait_ns,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_TMA_B_WAIT_NS,
                                    tile_tma_b_wait_ns,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_LDSM_QMMA_NS,
                                    tile_ldsm_qmma_ns,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_BF16_PACK_NS,
                                    tile_bf16_pack_ns,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_PEER_STORE_NS,
                                    tile_peer_store_ns,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_PHASE_ADVANCE_NS,
                                    tile_phase_end - tile_phase_start,
                                    scope="gpu",
                                )
                                cute.arch.store(
                                    green_trace.iterator
                                    + tile_record_base
                                    + K2_TILE_FIELD_TILE_END,
                                    tile_phase_end,
                                    scope="gpu",
                                )
                            compute_tile_seq += cutlass.Int32(1)
                work_tile_info = next_work_tile_info
            if cutlass.const_expr(green_trace is not None):
                if compute_warp == cutlass.Int32(0):
                    if lane_idx == cutlass.Int32(0):
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TILE_COUNT), k2_trace_tile_count, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_FIRST_TILE_ID), k2_trace_first_tile_id, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_LAST_TILE_ID), k2_trace_last_tile_id, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_A_WAIT_CALLS), k2_trace_tma_a_wait_calls, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_A_WAIT_NS), k2_trace_tma_a_wait_ns, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_A_TIMED_CALLS), k2_trace_tma_a_timed_calls, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_B_WAIT_CALLS), k2_trace_tma_b_wait_calls, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_B_WAIT_NS), k2_trace_tma_b_wait_ns, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_TMA_B_TIMED_CALLS), k2_trace_tma_b_timed_calls, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_MAINLOOP_NS), k2_trace_mainloop_ns, scope='gpu')
                        cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_STORE_NS), k2_trace_store_ns, scope='gpu')
        lane_idx = cute.arch.lane_idx()
        self.token_comm_hook_kernel_tail(token_comm_args, warp_idx=warp_idx, lane_idx=lane_idx, tidx=tidx)
        if cutlass.const_expr(green_trace is not None):
            if warp_idx == cutlass.Int32(0):
                if lane_idx == cutlass.Int32(0):
                    cute.arch.store(green_trace.iterator + trace_word(trace_role, cta_linear_id, FIELD_KERNEL_EXIT), read_globaltimer(), scope='gpu')

    # The shared host launcher calls this base-class hook.  K2 binds it to the
    # physically separate FC2-only device body above.
    fc1fc2_kernel_impl = fc2_combine_kernel_impl


__all__ = ["Sm120Fc2CombineKernel"]
