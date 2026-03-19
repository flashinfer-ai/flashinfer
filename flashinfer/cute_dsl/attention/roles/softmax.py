# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""SoftmaxRole — online softmax computation for attention kernels.

Handles:
- Row-max tracking and exp2 computation
- Row-sum accumulation
- KV-dimension masking (causal, sliding window, residual)
- Logits transform hooks via AttentionFusion
- Pipeline synchronization between MMA and correction stages

Extracted from BlackwellFusedMultiHeadAttentionForward.softmax / softmax_step.
"""

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.cute.typing import Int32, Float32

from cutlass.pipeline import PipelineProducer, PipelineConsumer

from .softmax_math import exp2_scale, packed_row_sum
from ..config import AttentionConfig, AttentionFusion
from ..tmem_layout import TmemLayout
from ..fusion.mask import (
    MaskType,
    apply_mask,
    get_unmasked_trip_count,
    get_masked_trip_count,
    get_kv_start_block_idx,
)
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class SoftmaxRole:
    """Online softmax warp group for attention kernels.

    Created from AttentionConfig + AttentionFusion + TmemLayout in the kernel's __init__.
    Tensor-type attributes (q_dtype, o_dtype) are set later via set_dtypes() in __call__.
    """

    def __init__(self, config: AttentionConfig, fusion: AttentionFusion, tmem: TmemLayout,
                 softmax0_warp_ids, softmax1_warp_ids, threads_per_warp):
        # From config
        self.qk_acc_dtype = config.qk_acc_dtype
        self.qk_mma_tiler = config.qk_mma_tiler
        self.cta_tiler = config.cta_tiler
        self.mask_type = config.mask_type
        self.window_left = config.window_left
        self.num_repeat_kv_heads = config.num_repeat_kv_heads

        # From TMEM layout
        self.tmem_vec0_offset = tmem.vec0_offset
        self.tmem_vec1_offset = tmem.vec1_offset
        self.tmem_p0_offset = tmem.p0_offset
        self.tmem_p1_offset = tmem.p1_offset

        # From fusion variant
        self.variant = fusion.variant
        self.has_score_mod = fusion.variant.has_score_mod
        self.has_logits_transform = fusion.variant.has_logits_transform
        self.has_statistics_update = fusion.variant.has_statistics_update
        self.has_params = fusion.has_params

        # Warp config
        self.softmax0_warp_ids = softmax0_warp_ids
        self.softmax1_warp_ids = softmax1_warp_ids
        self.threads_per_warp = threads_per_warp

        # Set later via set_dtypes()
        self.q_dtype = None
        self.o_dtype = None

    def set_dtypes(self, q_dtype, o_dtype):
        """Set tensor-type attributes known only at call time."""
        self.q_dtype = q_dtype
        self.o_dtype = o_dtype

    @cute.jit
    def step(
        self,
        stage: int,
        need_apply_mask: bool,
        iter_args: tuple,
        value_args: tuple,
        pipeline_args: tuple,
        atom_args: tuple,
        tensor_args: tuple,
        params: cute.Tensor | None,
    ) -> Tuple[
        Float32,
        Float32,
        PipelineProducer.ImmutableResourceHandle,
        PipelineConsumer,
        PipelineProducer,
        PipelineConsumer,
        PipelineProducer,
    ]:
        """Perform a single step of the softmax computation on a block of attention scores.

        This method processes one block of the attention matrix, computing numerically stable
        softmax by first finding the row maximum, subtracting it from all elements, applying
        exponential function, and then normalizing by the sum of exponentials. It also handles
        optional masking of attention scores.

        The method involves several key operations:
        1. Loading attention scores from tensor memory
        2. Applying optional masking based on position
        3. Computing row-wise maximum values for numerical stability
        4. Transforming scores using exp2(x*scale - max*scale)
        5. Computing row sums for normalization
        6. Coordinating pipeline synchronization between different processing stages
        """
        cS, row_max, row_sum, vec_i_handle, batch_coord, head_coord = iter_args
        qo_head_idx = head_coord
        kv_head_idx = qo_head_idx // self.num_repeat_kv_heads
        kv_tile_idx = cS[0][1] // self.qk_mma_tiler[1]

        seqlen_q, seqlen_k, scale_softmax_log2 = value_args
        (
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        ) = pipeline_args
        (
            qk_thr_mma,
            tiled_tmem_load,
            tiled_tmem_store,
            tiled_tmem_store_vec,
            thr_tmem_load,
            thr_tmem_store,
            thr_tmem_store_vec,
        ) = atom_args
        (
            tTMEM_LOADtS,
            tTMEM_STORE_VECtS,
            tTMEM_STOREtS_x4,
        ) = tensor_args

        if cutlass.const_expr(self.has_statistics_update):
            self.variant.params = params
            row_max, row_sum = self.variant.update_statistics(
                kv_tile_idx,
                qo_head_idx,
                row_max,
                row_sum,
                scale_softmax_log2,
            )

        tilePlikeFP32 = self.qk_mma_tiler[1] // Float32.width * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tScS_P_layout = cute.composition(
            tScS.layout, cute.make_layout((128, tilePlikeFP32))
        )
        tScS_P = cute.make_tensor(tScS.iterator, tScS_P_layout)
        tTMEM_LOADcS = thr_tmem_load.partition_D(tScS)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tTMEM_STOREcS = thr_tmem_store.partition_S(tScS_P)

        # Wait for Si
        si_handle = mma_si_consumer.wait_and_advance()
        tTMEM_LOADrS = cute.make_fragment(tTMEM_LOADcS.shape, self.qk_acc_dtype)
        cute.copy(tiled_tmem_load, tTMEM_LOADtS, tTMEM_LOADrS)
        if need_apply_mask:
            apply_mask(self.mask_type, self.window_left, tTMEM_LOADrS, tTMEM_LOADcS, seqlen_k, seqlen_k - seqlen_q)

        frg_cnt = 4
        frg_tile = cute.size(tTMEM_LOADrS) // frg_cnt
        tTMEM_LOADrS_frg = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile))
        tTMEM_LOADcS_frg = cute.logical_divide(tTMEM_LOADcS, cute.make_layout(frg_tile))

        if cutlass.const_expr(self.has_score_mod):
            if cutlass.const_expr(self.has_params):
                self.variant.params = params
            for j in range(frg_cnt):
                for k in range(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                    qo_idx, kv_idx = tTMEM_LOADcS_frg[k, j]
                    tTMEM_LOADrS_frg[k, j] = self.variant.score_mod(
                        tTMEM_LOADrS_frg[k, j],
                        batch_coord,
                        qo_idx,
                        kv_idx,
                        qo_head_idx,
                        kv_head_idx,
                    )

        old_row_max = row_max
        row_max = tTMEM_LOADrS.load().reduce(cute.ReductionOp.MAX, row_max, 0)
        row_max_safe = row_max

        if row_max == -cutlass.Float32.inf:
            row_max_safe = 0.0
        tTMEM_STORE_VECrS = cute.make_fragment(
            tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
        )

        tTMEM_STORE_VECrS[0] = old_row_max
        tTMEM_STORE_VECrS[1] = row_max_safe
        cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
        cute.arch.fence_view_async_tmem_store()
        # Notify correction wg that row_max is ready
        vec_i_handle.commit()

        tTMEM_STORErS_x4 = cute.make_fragment(tTMEM_STOREcS.shape, self.qk_acc_dtype)
        tTMEM_STORErS_x4_e = cute.make_tensor(
            cute.recast_ptr(tTMEM_STORErS_x4.iterator, dtype=self.q_dtype),
            tTMEM_LOADrS.layout,
        )

        scale = scale_softmax_log2

        # Sequence barrier wait
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle = s0_s1_sequence_producer.acquire_and_advance()
        else:
            sequence_consumer_handle = s0_s1_sequence_consumer.wait_and_advance()
        tTMEM_STORErS_x4_e_frg = cute.logical_divide(
            tTMEM_STORErS_x4_e, cute.make_layout(frg_tile)
        )
        ### the softmax computation part ### e^(xi*scale - mi*scale)
        if cutlass.const_expr(not self.has_logits_transform):
            for j in range(frg_cnt):
                exp2_scale(tTMEM_LOADrS_frg[None, j], scale, row_max_safe)
                s_vec = tTMEM_LOADrS_frg[None, j].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

        else:
            for j in range(frg_cnt):
                for k in range(cute.size(tTMEM_LOADrS_frg, mode=[0])):
                    qo_idx, kv_idx = tTMEM_LOADcS_frg[k, j]
                    tTMEM_LOADrS_frg[k, j] = self.variant.transform_logits(
                        tTMEM_LOADrS_frg[k, j],
                        batch_coord,
                        qo_idx,
                        kv_idx,
                        qo_head_idx,
                        kv_head_idx,
                    )
                s_vec = tTMEM_LOADrS_frg[None, j].load()
                tTMEM_STORErS_x4_e_frg[None, j].store(s_vec.to(self.q_dtype))

        # Sequence barrier arrive
        if cutlass.const_expr(stage == 0):
            sequence_producer_handle.commit()
        else:
            sequence_consumer_handle.release()
        cute.copy(tiled_tmem_store, tTMEM_STORErS_x4, tTMEM_STOREtS_x4)
        cute.arch.fence_view_async_tmem_store()
        # Notify tensor core warp that softmax(S->P) is ready
        si_handle.release()

        ### di = di-1 * (e^(mi-1 - mi) * scale) + sum e^(xi*scale - mi*scale)
        vec_i_handle = si_corr_producer.acquire_and_advance()
        acc_scale_ = scale * (old_row_max - row_max_safe)
        # * 0.5 compensates for initializing both packed elements with row_sum below
        acc_scale = cute.arch.exp2(acc_scale_) * 0.5
        row_sum *= acc_scale
        # 4-way unrolled reduction for ILP: 4 independent accumulator chains
        # run in parallel, then tree-reduce. local_row_sum_0 is seeded with
        # (row_sum, row_sum) so the old running sum folds into the reduction.
        local_row_sum_0 = (row_sum, row_sum)
        local_row_sum_1 = (0.0, 0.0)
        local_row_sum_2 = (0.0, 0.0)
        local_row_sum_3 = (0.0, 0.0)

        reduction_unroll = 4
        frg_tile_r = cute.size(tTMEM_LOADrS) // reduction_unroll
        tTMEM_LOADrS_frg_r = cute.logical_divide(tTMEM_LOADrS, cute.make_layout(frg_tile_r))

        for j in cutlass.range_constexpr(0, cute.size(tTMEM_LOADrS_frg_r, mode=[0]), 2):
            local_row_sum_0 = cute.arch.add_packed_f32x2(
                local_row_sum_0, (tTMEM_LOADrS_frg_r[j, 0], tTMEM_LOADrS_frg_r[j + 1, 0])
            )
            local_row_sum_1 = cute.arch.add_packed_f32x2(
                local_row_sum_1, (tTMEM_LOADrS_frg_r[j, 1], tTMEM_LOADrS_frg_r[j + 1, 1])
            )
            local_row_sum_2 = cute.arch.add_packed_f32x2(
                local_row_sum_2, (tTMEM_LOADrS_frg_r[j, 2], tTMEM_LOADrS_frg_r[j + 1, 2])
            )
            local_row_sum_3 = cute.arch.add_packed_f32x2(
                local_row_sum_3, (tTMEM_LOADrS_frg_r[j, 3], tTMEM_LOADrS_frg_r[j + 1, 3])
            )

        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_1)
        local_row_sum_2 = cute.arch.add_packed_f32x2(local_row_sum_2, local_row_sum_3)
        local_row_sum_0 = cute.arch.add_packed_f32x2(local_row_sum_0, local_row_sum_2)
        row_sum = local_row_sum_0[0] + local_row_sum_0[1]

        return (
            row_max,
            row_sum,
            vec_i_handle,
            mma_si_consumer,
            si_corr_producer,
            s0_s1_sequence_consumer,
            s0_s1_sequence_producer,
        )

    # For both softmax0 and softmax1 warp group
    @cute.jit
    def run(
        self,
        stage: int,
        seqlen_q: Int32,
        seqlen_k: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        qk_thr_mma: cute.core.ThrMma,
        tStS: cute.Tensor,
        tStSi: cute.Tensor,
        params: cute.Tensor | None,
        mma_si_consumer: PipelineConsumer,
        si_corr_producer: PipelineProducer,
        s0_s1_sequence_consumer: PipelineConsumer,
        s0_s1_sequence_producer: PipelineProducer,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        Handles softmax for either stage 0 or stage 1 (first or second half of
        the Q tile). Loops over KV tiles, calling step() for each, coordinating
        pipeline synchronization.
        """
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (
            self.threads_per_warp
            * (
                len(self.softmax0_warp_ids)
                if stage == 0
                else len(self.softmax1_warp_ids)
            )
        )

        cS_base = cute.make_identity_tensor(
            (self.qk_mma_tiler[0], self.qk_mma_tiler[1])
        )
        tilePlikeFP32 = self.qk_mma_tiler[1] // 32 * self.o_dtype.width
        tScS = qk_thr_mma.partition_C(cS_base)
        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))
        tmem_vec_offset = self.tmem_vec0_offset if stage == 0 else self.tmem_vec1_offset
        tStS_vec = cute.make_tensor(tStS.iterator + tmem_vec_offset, tStS_vec_layout)
        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)
        tStS_P_layout = cute.composition(
            tStS.layout, cute.make_layout((128, tilePlikeFP32))
        )
        tmem_p_offset = self.tmem_p0_offset if stage == 0 else self.tmem_p1_offset
        tStS_P = cute.make_tensor(tStS.iterator + tmem_p_offset, tStS_P_layout)
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStSi)
        thread_idx = tidx % (
            self.threads_per_warp
            * (
                len(self.softmax0_warp_ids)
                if stage == 0
                else len(self.softmax1_warp_ids)
            )
        )
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        tTMEM_LOADtS = thr_tmem_load.partition_S(tStSi)
        tmem_store_vec_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store_vec = tcgen05.make_tmem_copy(tmem_store_vec_atom, tStS_vec)
        thr_tmem_store_vec = tiled_tmem_store_vec.get_slice(thread_idx)
        tTMEM_STORE_VECtS = thr_tmem_store_vec.partition_D(tStS_vec)
        tTMEM_STORE_VECcS = thr_tmem_store_vec.partition_S(tScS_vec)
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStS_P)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)
        tTMEM_STOREtS_x4 = thr_tmem_store.partition_D(tStS_P)

        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            seqlen_q_ = seqlen_q
            seqlen_k_ = seqlen_k
            continue_cond = False

            if cutlass.const_expr(cum_seqlen_q is not None):
                cuseqlen_q = cum_seqlen_q[batch_coord]
                seqlen_q_ = cum_seqlen_q[batch_coord + 1] - cuseqlen_q
                continue_cond = (
                    not FmhaStaticTileScheduler.check_valid_work_for_seqlen_q(
                        self.cta_tiler[0],
                        curr_block_coord[0],
                        seqlen_q_,
                    )
                )

            if not continue_cond:
                if cutlass.const_expr(cum_seqlen_k is not None):
                    cuseqlen_k = cum_seqlen_k[batch_coord]
                    seqlen_k_ = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                row_max = -Float32.inf
                row_sum = 0.0
                value_args = (seqlen_q_, seqlen_k_, scale_softmax_log2)
                atom_args = (
                    qk_thr_mma,
                    tiled_tmem_load,
                    tiled_tmem_store,
                    tiled_tmem_store_vec,
                    thr_tmem_load,
                    thr_tmem_store,
                    thr_tmem_store_vec,
                )
                tensor_args = (
                    tTMEM_LOADtS,
                    tTMEM_STORE_VECtS,
                    tTMEM_STOREtS_x4,
                )

                kv_start_offset = get_kv_start_block_idx(
                    self.mask_type, self.window_left,
                    curr_block_coord, self.cta_tiler, seqlen_k_, seqlen_q_,
                ) * self.qk_mma_tiler[1]
                logical_offset = (
                    curr_block_coord[0] * self.cta_tiler[0]
                    + stage * self.qk_mma_tiler[0],
                    kv_start_offset,
                )
                cS = cute.domain_offset(logical_offset, cS_base)
                vec_i_handle = si_corr_producer.acquire_and_advance()
                unmask_count = get_unmasked_trip_count(
                    self.mask_type, self.window_left,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                    seqlen_q_,
                )
                batch_coord = curr_block_coord[2][1]
                head_coord = curr_block_coord[2][0]
                for i in cutlass.range(0, unmask_count, 1, unroll=1):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (
                        cS_iter,
                        row_max,
                        row_sum,
                        vec_i_handle,
                        batch_coord,
                        head_coord,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.step(
                        stage,
                        False,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        params,
                    )

                mask_count = get_masked_trip_count(
                    self.mask_type, self.window_left,
                    curr_block_coord,
                    self.cta_tiler,
                    seqlen_k_,
                    seqlen_q_,
                )

                for i in cutlass.range(
                    unmask_count, unmask_count + mask_count, 1, unroll=1
                ):
                    cS_iter = cute.domain_offset((0, i * self.qk_mma_tiler[1]), cS)
                    iter_args = (
                        cS_iter,
                        row_max,
                        row_sum,
                        vec_i_handle,
                        batch_coord,
                        head_coord,
                    )
                    pipeline_args = (
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    )
                    (
                        row_max,
                        row_sum,
                        vec_i_handle,
                        mma_si_consumer,
                        si_corr_producer,
                        s0_s1_sequence_consumer,
                        s0_s1_sequence_producer,
                    ) = self.step(
                        stage,
                        True,
                        iter_args,
                        value_args,
                        pipeline_args,
                        atom_args,
                        tensor_args,
                        params,
                    )
                si_handle = mma_si_consumer.wait_and_advance()
                tTMEM_STORE_VECrS = cute.make_fragment(
                    tTMEM_STORE_VECcS.shape, self.qk_acc_dtype
                )
                tTMEM_STORE_VECrS[0] = row_sum
                tTMEM_STORE_VECrS[1] = row_max
                cute.copy(tiled_tmem_store_vec, tTMEM_STORE_VECrS, tTMEM_STORE_VECtS)
                cute.arch.fence_view_async_tmem_store()
                vec_i_handle.commit()
                si_corr_producer.acquire()
                # Empty step to sync against pipe s
                si_handle.release()

            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop
