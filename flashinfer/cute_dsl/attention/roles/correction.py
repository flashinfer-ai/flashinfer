# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""CorrectionRole — output rescaling, epilogue, and orchestration for attention kernels.

Handles:
- Orchestration loop: pipeline sync with softmax/MMA, scale computation
- Rescaling partial output when row-max changes across KV tiles
- Final scaling, type conversion, optional output transform, SMEM write

Extracted from BlackwellFusedMultiHeadAttentionForward correction warp section.
"""

from typing import Optional, Type

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.typing import Int32, Float32

from cutlass.pipeline import PipelineProducer, PipelineConsumer

from ..config import AttentionConfig, AttentionFusion
from ..tmem_layout import TmemLayout
from ..fusion.mask import get_trip_count
from ..scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
)


class CorrectionRole:
    """Correction warp group for attention kernels.

    Created from AttentionConfig + AttentionFusion + TmemLayout in the kernel's __init__.
    Tensor-type attributes (o_dtype, o_layout, epi_tile) are set later via
    set_call_attrs() in __call__.
    """

    def __init__(
        self,
        config: AttentionConfig,
        fusion: AttentionFusion,
        tmem: TmemLayout,
        correction_warp_ids,
        threads_per_warp,
    ):
        # From config
        self.qk_acc_dtype = config.qk_acc_dtype
        self.qk_mma_tiler = config.qk_mma_tiler
        self.pv_mma_tiler = config.pv_mma_tiler
        self.pv_acc_dtype = config.pv_acc_dtype
        self.cta_tiler = config.cta_tiler
        self.mask_type = config.mask_type
        self.window_left = config.window_left

        # From TMEM layout
        self.tmem_vec0_offset = tmem.vec0_offset
        self.tmem_vec1_offset = tmem.vec1_offset

        # From fusion variant
        self.variant = fusion.variant
        self.has_logits_transform = fusion.variant.has_logits_transform
        self.has_output_transform = fusion.variant.has_output_transform

        # Warp config
        self.correction_warp_ids = correction_warp_ids
        self.threads_per_warp = threads_per_warp

        # Set later via set_call_attrs()
        self.o_dtype: Optional[Type[cutlass.Numeric]] = None
        self.o_layout = None
        self.epi_tile = None

    def set_call_attrs(self, o_dtype, o_layout, epi_tile):
        """Set tensor-type attributes known only at call time."""
        self.o_dtype = o_dtype
        self.o_layout = o_layout
        self.epi_tile = epi_tile

    @cute.jit
    def rescale(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.
        """
        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        tOcO = thr_mma.partition_C(cO)

        corr_tile_size = 16  # tuneable parameter
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        tOtO_i_layout = cute.composition(
            tOtO.layout, cute.make_layout((128, corr_tile_size))
        )
        tOcO_i_layout = cute.composition(
            tOcO.layout, cute.make_layout((128, corr_tile_size))
        )

        tOtO_i = cute.make_tensor(tOtO.iterator, tOtO_i_layout)
        tOcO_i = cute.make_tensor(tOcO.iterator, tOcO_i_layout)

        tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
        tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        thr_tmem_store = tiled_tmem_store.get_slice(thread_idx)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i)
        tTMEM_LOADcO = thr_tmem_load.partition_D(tOcO_i)

        tTMEM_STOREtO = thr_tmem_store.partition_D(tOtO_i)

        tTMrO = cute.make_fragment(
            (tTMEM_LOADcO.shape, 128 // corr_tile_size), self.pv_acc_dtype
        )
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMrO_i_ = tTMrO[None, i]
            tTMrO_i_layout = cute.composition(
                tTMrO_i_.layout, cute.make_layout(tTMrO.shape[0])
            )
            tTMrO_i = cute.make_tensor(tTMrO_i_.iterator, tTMrO_i_layout)
            tTMEM_LOADtO_i = cute.make_tensor(
                tTMEM_LOADtO.iterator + i * corr_tile_size, tTMEM_LOADtO.layout
            )
            tTMEM_STOREtO_i = cute.make_tensor(
                tTMEM_STOREtO.iterator + i * corr_tile_size, tTMEM_STOREtO.layout
            )

            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO_i)
            for j in range(0, cute.size(tTMrO_i), 2):
                tTMrO_i[j], tTMrO_i[j + 1] = cute.arch.mul_packed_f32x2(
                    (tTMrO_i[j], tTMrO_i[j + 1]),
                    (scale, scale),
                )
            cute.copy(tiled_tmem_store, tTMrO_i, tTMEM_STOREtO_i)

    @cute.jit
    def epilog(
        self,
        thr_mma: cute.core.ThrMma,
        tOtO: cute.Tensor,
        scale: Float32,
        m: Float32,
        d: Float32,
        sO: cute.Tensor,
        batch_coord: Int32,
        head_coord: Int32,
        qo_idx_offset: Int32,
    ):
        """Apply final scaling and transformation to attention output before writing to global memory.

        Performs:
        1. Loading of accumulated attention results from tensor memory
        2. Application of the final output scaling factor
        3. Type conversion (typically from higher precision accumulator to output precision)
        4. Reorganization of data for optimal memory access patterns
        5. Preparation for efficient TMA store operations
        """
        assert self.o_dtype is not None
        assert self.epi_tile is not None

        pv_tiled_mma_shape = (
            self.pv_mma_tiler[0],
            self.pv_mma_tiler[1],
        )
        cO = cute.make_identity_tensor(pv_tiled_mma_shape)
        cO_custom = cute.make_identity_tensor(pv_tiled_mma_shape)

        corr_tile_size = 32 * 8 // self.o_dtype.width
        tOsO = thr_mma.partition_C(sO)
        tOcO = thr_mma.partition_C(cO)
        tOcO_custom = thr_mma.partition_C(cO_custom)

        tOtO_i = cute.logical_divide(tOtO, cute.make_layout((128, corr_tile_size)))
        tOcO_i = cute.logical_divide(tOcO, cute.make_layout((128, corr_tile_size)))
        tOsO_i = cute.logical_divide(tOsO, cute.make_layout((128, corr_tile_size)))
        tOcO_custom_i = cute.logical_divide(
            tOcO_custom, cute.make_layout((128, corr_tile_size))
        )
        tidx, _, _ = cute.arch.thread_idx()
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils.get_tmem_load_op(
            self.pv_mma_tiler,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            use_2cta_instrs=False,
        )

        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO_i[(None, None), 0]
        )

        thr_tmem_load = tiled_tmem_load.get_slice(thread_idx)
        smem_copy_atom = sm100_utils.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tTMEM_LOADtO = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tTMEM_LOADsO = thr_tmem_load.partition_D(tOsO_i[(None, None), None])
        tTMEM_LOADoO = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        tTMEM_LOADcO_custom = thr_tmem_load.partition_D(
            tOcO_custom_i[(None, None), None]
        )

        scale_rcp_d = scale / d if not self.has_logits_transform else scale
        rcp_d = 1 / d if m != -Float32.inf else 0.0
        for i in range(self.cta_tiler[2] // corr_tile_size):
            tTMEM_LOADtO_i = tTMEM_LOADtO[None, 0, 0, i]
            tTMEM_LOADsO_i = tTMEM_LOADsO[None, 0, 0, i]
            tTMrO = cute.make_fragment(
                tTMEM_LOADoO[None, 0, 0, i].shape, self.pv_acc_dtype
            )
            cute.copy(tiled_tmem_load, tTMEM_LOADtO_i, tTMrO)
            if cutlass.const_expr(not self.has_output_transform):
                for j in range(0, cute.size(tTMrO), 2):
                    tTMrO[j], tTMrO[j + 1] = cute.arch.mul_packed_f32x2(
                        (tTMrO[j], tTMrO[j + 1]),
                        (scale_rcp_d, scale_rcp_d),
                    )
            else:
                tTMcO_custom = tTMEM_LOADcO_custom[None, 0, 0, i]
                for j in range(0, cute.size(tTMrO)):
                    qo_idx = qo_idx_offset + tTMcO_custom[j][0]
                    tTMrO[j] = self.variant.transform_output(
                        tTMrO[j],
                        batch_coord,
                        qo_idx,
                        head_coord,
                        m,
                        rcp_d,
                        scale,
                    )
            tSMrO = cute.make_fragment(tTMrO.shape, self.o_dtype)
            o_vec = tTMrO.load()
            tSMrO.store(o_vec.to(self.o_dtype))
            cute.copy(tiled_smem_store, tSMrO, tTMEM_LOADsO_i)

        # fence view async shared
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared,
            space=cute.arch.SharedSpace.shared_cta,
        )

    @cute.jit
    def run(
        self,
        qk_thr_mma: cute.core.ThrMma,
        pv_thr_mma: cute.core.ThrMma,
        tStS: cute.Tensor,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        sO: cute.Tensor,
        seqlen_q_global: Int32,
        seqlen_k_global: Int32,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        s0_corr_consumer: PipelineConsumer,
        s1_corr_consumer: PipelineConsumer,
        mma_corr_consumer: PipelineConsumer,
        corr_epi_producer: PipelineProducer,
        tile_sched_params: FmhaStaticTileSchedulerParams,
        tmem_dealloc_mbar_ptr: Int32,
    ):
        """Correction warp orchestration loop.

        For each work tile, synchronizes with softmax (vec buffers) and MMA
        (output partials) pipelines, computes rescaling factors from row-max
        changes, delegates to rescale() and epilog(), and signals the epilogue
        warp when output is ready.
        """
        tidx, _, _ = cute.arch.thread_idx()

        cS = cute.make_identity_tensor((self.qk_mma_tiler[0], self.qk_mma_tiler[1]))
        tScS = qk_thr_mma.partition_C(cS)

        tStS_vec_layout = cute.composition(tStS.layout, cute.make_layout((128, 2)))

        tStS_vec0 = cute.make_tensor(
            tStS.iterator + self.tmem_vec0_offset, tStS_vec_layout
        )
        tStS_vec1 = cute.make_tensor(
            tStS.iterator + self.tmem_vec1_offset, tStS_vec_layout
        )

        tScS_vec_layout = cute.composition(tScS.layout, cute.make_layout((128, 2)))
        tScS_vec = cute.make_tensor(tScS.iterator, tScS_vec_layout)

        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(2)),
            self.qk_acc_dtype,
        )

        tiled_tmem_load_vec = tcgen05.make_tmem_copy(tmem_load_v_atom, tStS_vec0)
        thread_idx = tidx % (self.threads_per_warp * len(self.correction_warp_ids))
        thr_tmem_load_vec = tiled_tmem_load_vec.get_slice(thread_idx)

        tTMEM_LOAD_VECtS0 = thr_tmem_load_vec.partition_S(tStS_vec0)
        tTMEM_LOAD_VECtS1 = thr_tmem_load_vec.partition_S(tStS_vec1)
        tTMEM_LOAD_VECcS = thr_tmem_load_vec.partition_D(tScS_vec)

        tile_sched = create_fmha_static_tile_scheduler(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()

        while work_tile.is_valid_tile:
            curr_block_coord = work_tile.tile_idx
            batch_coord = curr_block_coord[2][1]
            head_coord = curr_block_coord[2][0]
            qo_idx_offset = curr_block_coord[0] * self.cta_tiler[0]

            seqlen_q_ = seqlen_q_global
            seqlen_k = seqlen_k_global
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
                    seqlen_k = cum_seqlen_k[batch_coord + 1] - cuseqlen_k
                # Ignore first signal from softmax as no correction is required
                vec0_handle = s0_corr_consumer.wait_and_advance()
                vec0_handle.release()
                vec1_handle = s1_corr_consumer.wait_and_advance()
                seqlen_kv_loop_steps = (
                    get_trip_count(
                        self.mask_type,
                        self.window_left,
                        curr_block_coord,
                        self.cta_tiler,
                        seqlen_k,
                        seqlen_q_,
                    )
                    - 1
                )
                for _i in cutlass.range(0, seqlen_kv_loop_steps, 1, unroll=1):
                    # wait for vec0 (row_wise current max & previous max)
                    vec0_handle = s0_corr_consumer.wait_and_advance()
                    tTMEM_LOAD_VECrS = cute.make_fragment(
                        tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                    )
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                    scale_ = scale_softmax_log2 * (
                        tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                    )
                    scale = cute.arch.exp2(scale_)

                    # wait for o0
                    o0_handle_consumer = mma_corr_consumer.wait_and_advance()
                    if cutlass.const_expr(not self.has_logits_transform):
                        self.rescale(pv_thr_mma, tOtO0, scale)
                    # release vec1 & o0
                    vec1_handle.release()
                    cute.arch.fence_view_async_tmem_store()
                    o0_handle_consumer.release()

                    # wait for vec1 (row_wise current max & previous max)
                    vec1_handle = s1_corr_consumer.wait_and_advance()
                    cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                    scale_ = scale_softmax_log2 * (
                        tTMEM_LOAD_VECrS[0] - tTMEM_LOAD_VECrS[1]
                    )
                    scale = cute.arch.exp2(scale_)

                    o1_handle_consumer = mma_corr_consumer.wait_and_advance()
                    if cutlass.const_expr(not self.has_logits_transform):
                        self.rescale(pv_thr_mma, tOtO1, scale)
                    vec0_handle.release()
                    cute.arch.fence_view_async_tmem_store()
                    o1_handle_consumer.release()
                # End of seqlen_corr_loop_steps
                vec1_handle.release()

                # wait for vec0 (row_wise global sum)
                vec0_handle = s0_corr_consumer.wait_and_advance()
                tTMEM_LOAD_VECrS = cute.make_fragment(
                    tTMEM_LOAD_VECcS.shape, self.qk_acc_dtype
                )
                cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS0, tTMEM_LOAD_VECrS)
                cute.arch.fence_view_async_tmem_load()
                vec0_handle.release()
                # wait for o0
                o0_handle_consumer = mma_corr_consumer.wait_and_advance()
                o0_final_handle = corr_epi_producer.acquire_and_advance()

                epilogue_scale = scale_output
                d = tTMEM_LOAD_VECrS[0]  # row sum
                m = tTMEM_LOAD_VECrS[1]  # row max
                self.epilog(
                    pv_thr_mma,
                    tOtO0,
                    epilogue_scale,
                    m,
                    d,
                    sO[None, None, 0],
                    batch_coord,
                    head_coord,
                    qo_idx_offset,
                )
                o0_handle_consumer.release()
                o0_final_handle.commit()

                # wait for vec1 (row_wise global sum)
                vec1_handle = s1_corr_consumer.wait_and_advance()
                cute.copy(tiled_tmem_load_vec, tTMEM_LOAD_VECtS1, tTMEM_LOAD_VECrS)
                cute.arch.fence_view_async_tmem_load()
                vec1_handle.release()
                # wait for o1
                o1_handle_consumer = mma_corr_consumer.wait_and_advance()
                o1_final_handle = corr_epi_producer.acquire_and_advance()

                epilogue_scale = scale_output
                d = tTMEM_LOAD_VECrS[0]  # row sum
                m = tTMEM_LOAD_VECrS[1]  # row max
                self.epilog(
                    pv_thr_mma,
                    tOtO1,
                    epilogue_scale,
                    m,
                    d,
                    sO[None, None, 1],
                    batch_coord,
                    head_coord,
                    qo_idx_offset + self.qk_mma_tiler[0],
                )
                o1_handle_consumer.release()
                o1_final_handle.commit()
            # Advance to next tile
            tile_sched.advance_to_next_work()
            work_tile = tile_sched.get_current_work()
        # End of persistent scheduler loop
        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
