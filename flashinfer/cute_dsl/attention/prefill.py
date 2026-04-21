# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
from cutlass.cute.typing import Int32, Float32

from .config import AttentionConfig, AttentionFusion
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE, PREFILL_TRANSFORM_SCHEDULE
from .mainloop_spec import make_prefill_mainloop_spec
from .collective_builder import build_fmha_launch_params
from .scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler_params,
)
from .roles.softmax import SoftmaxRole
from .roles.correction import CorrectionRole
from .roles.epilogue import EpilogueRole
from .roles.loader_tma import LoaderRole
from .roles.mma import MmaRole

import warnings

warnings.filterwarnings(
    "ignore",
    message="This loop is no longer unrolled and may cause performance regression",
)

"""Blackwell SM100 fused multi-head attention (FMHA) kernel using CuTe DSL.

Warp-specialized persistent kernel with TMA loads/stores, pipelined QK and PV
MMA stages, online softmax with correction, and optional causal/sliding-window
masking.  Supports fp16, bf16, and fp8 input types.
"""


class BlackwellFusedMultiHeadAttentionForward:
    def __init__(
        self,
        config: AttentionConfig,
        fusion: AttentionFusion | None = None,
        warp_schedule: WarpSchedule | None = None,
    ):
        """Initializes a Blackwell Fused Multi-Head Attention (FMHA) kernel.

        :param config: Core attention configuration (dtypes, tile shapes, mode).
        :param fusion: Optional customization callbacks (logits/output transforms, sinks).
        :param warp_schedule: Warp role assignment and register budgets. Defaults to PREFILL_SCHEDULE.
        """

        self.config = config
        self.fusion = fusion if fusion is not None else AttentionFusion()
        self.has_logits_transform = self.fusion.variant.has_logits_transform

        if warp_schedule is not None:
            self.schedule = warp_schedule
        elif self.has_logits_transform:
            self.schedule = PREFILL_TRANSFORM_SCHEDULE
        else:
            self.schedule = PREFILL_SCHEDULE

        self.mainloop = make_prefill_mainloop_spec(
            config,
            self.schedule,
            self.has_logits_transform,
        )
        self.tmem = self.mainloop.tmem_layout

    @cute.jit
    def __call__(
        self,
        q_in: cute.Tensor,
        k_in: cute.Tensor,
        v_in: cute.Tensor,
        o_in: cute.Tensor,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        cum_seqlen_q: cute.Tensor | None,
        s_q_all: Int32,
        cum_seqlen_k: cute.Tensor | None,
        s_k_all: Int32,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        params_in: cute.Tensor | None,
        stream,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        :param q_in: The query tensor (NHD layout)
        :param k_in: The key tensor (NHD layout)
        :param v_in: The value tensor (NHD layout)
        :param o_in: The output tensor (NHD layout, with padding before data pointer)
        :param problem_size: ``(b, s_q, s_k, h_q, h_k, d)``
        :param cum_seqlen_q: Cumulative query sequence lengths, or None
        :param cum_seqlen_k: Cumulative KV sequence lengths, or None
        :param scale_softmax_log2: ``log2(e) * sm_scale``
        :param scale_output: Output scaling factor
        :param params_in: Variant runtime data tensor, or None
        :param stream: CUDA stream
        """
        b, s_q, s_k, h_q, h_k, d = problem_size
        h_r = h_q // h_k

        o_offset = -s_q * d * h_r * h_k
        b_q = 1
        b_kv = 1
        b_o = s_q * (1 + b)
        stride_b_q = 0
        stride_b_kv = 0
        stride_b_o = d * h_r * h_k

        # (s, d, ((h_r, h_k), b))
        q_layout = cute.make_layout(
            (s_q_all, d, ((h_r, h_k), b_q)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_q)),
        )
        q = cute.make_tensor(q_in.iterator, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k_all, d, ((h_r, h_k), b_kv)),
            stride=(d * h_k, 1, ((0, d), stride_b_kv)),
        )
        k = cute.make_tensor(k_in.iterator, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, s_k_all, ((h_r, h_k), b_kv)),
            stride=(1, d * h_k, ((0, d), stride_b_kv)),
        )
        v = cute.make_tensor(v_in.iterator, v_layout)
        # (s, d, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_o)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_o)),
        )
        o = cute.make_tensor(o_in.iterator + o_offset, o_layout)

        params = (
            cute.make_tensor(
                params_in.iterator,
                cute.make_layout(
                    self.fusion.params_shape,
                    stride=self.fusion.params_strides,
                ),
            )
            if self.fusion.has_params
            else None
        )

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q.element_type
        self.k_dtype = k.element_type
        self.v_dtype = v.element_type
        self.o_dtype = o.element_type

        self.tile_sched_params, grid = self._compute_grid(
            cute.shape((s_q, d, ((h_r, h_k), b))),
            self.config.cta_tiler,
            self.config.is_persistent,
        )

        self.q_major_mode = utils.LayoutEnum.from_tensor(q).mma_major_mode()
        self.k_major_mode = utils.LayoutEnum.from_tensor(k).mma_major_mode()
        self.v_major_mode = utils.LayoutEnum.from_tensor(v).mma_major_mode()
        self.o_layout = utils.LayoutEnum.from_tensor(o)

        if cutlass.const_expr(self.q_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of q is not supported")
        if cutlass.const_expr(self.k_major_mode != tcgen05.OperandMajorMode.K):
            raise RuntimeError("The layout of k is not supported")
        if cutlass.const_expr(self.v_major_mode != tcgen05.OperandMajorMode.MN):
            raise RuntimeError("The layout of v is not supported")

        # check type consistency
        if cutlass.const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if cutlass.const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        self.mainloop = self.mainloop.resolve(self.q_dtype.width)

        self.softmax_role = SoftmaxRole(
            self.config,
            self.fusion,
            self.tmem,
            softmax0_warp_ids=self.schedule.softmax0_warp_ids,
            softmax1_warp_ids=self.schedule.softmax1_warp_ids,
            threads_per_warp=self.schedule.threads_per_warp,
        )
        if cutlass.const_expr(not self.has_logits_transform):
            self.correction_role = CorrectionRole(
                self.config,
                self.fusion,
                self.tmem,
                correction_warp_ids=self.schedule.correction_warp_ids,
                threads_per_warp=self.schedule.threads_per_warp,
            )
        self.epilogue_role = EpilogueRole(self.config)
        self.loader_role = LoaderRole(self.config)
        self.mma_role = MmaRole(
            self.config,
            tmem_alloc_cols=self.tmem.alloc_cols,
            tmem_alloc_sync_bar_id=self.schedule.tmem_alloc_sync_bar_id,
            threads_per_warp=self.schedule.threads_per_warp,
            has_logits_transform=self.has_logits_transform,
        )
        self.softmax_role.set_dtypes(self.q_dtype, self.o_dtype)

        lp = build_fmha_launch_params(
            self.mainloop,
            q,
            k,
            v,
            o,
            self.q_dtype,
            self.k_dtype,
            self.v_dtype,
            self.o_dtype,
            self.q_major_mode,
            self.k_major_mode,
            self.v_major_mode,
            self.o_layout,
        )
        self.shared_storage = lp.SharedStorage

        smem_bytes = lp.SharedStorage.size_in_bytes()
        smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")
        if cutlass.const_expr(smem_bytes > smem_capacity):
            head_dim = self.config.mma_tiler[2]
            raise ValueError(
                f"SharedStorage requires {smem_bytes} bytes but SM100 provides "
                f"{smem_capacity} bytes. Reduce head_dim (currently {head_dim}) "
                f"or tile size."
            )

        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kv_bytes = lp.tma_copy_kv_bytes

        if cutlass.const_expr(not self.has_logits_transform):
            self.correction_role.set_call_attrs(self.o_dtype, lp.o_layout, lp.epi_tile)
        else:
            self.softmax_role.set_call_attrs(lp.o_layout, lp.epi_tile)

        self.kernel(
            lp.qk_tiled_mma,
            lp.pv_tiled_mma,
            lp.tma_atom_q,
            lp.tma_tensor_q,
            lp.tma_atom_k,
            lp.tma_tensor_k,
            lp.tma_atom_v,
            lp.tma_tensor_v,
            lp.tma_atom_o,
            lp.tma_tensor_o,
            cum_seqlen_q,
            cum_seqlen_k,
            scale_softmax_log2,
            scale_output,
            params,
            lp.q_smem_layout_staged,
            lp.k_smem_layout_staged,
            lp.p_tmem_layout_staged,
            lp.v_smem_layout_staged,
            lp.o_smem_layout_staged,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.schedule.threads_per_cta, 1, 1],
            cluster=lp.cluster_shape_mnk,
            smem=lp.SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _create_pipelines(self, storage):
        """Create all inter-warp pipelines from the topology and storage barriers."""
        barrier_ptrs = {
            edge.name: getattr(storage, edge.barrier_field_name).data_ptr()
            for edge in self.mainloop.pipeline_topology.edges
        }
        tx_counts = {"q": self.tma_copy_q_bytes, "kv": self.tma_copy_kv_bytes}
        return self.mainloop.pipeline_topology.create_pipelines(
            barrier_ptrs,
            tx_counts,
            self.schedule.threads_per_warp,
        )

    @cute.jit
    def _create_mma_fragments(
        self,
        qk_tiled_mma,
        pv_tiled_mma,
        sQ,
        sK,
        sV,
        p_tmem_layout_staged,
    ):
        """Partition MMA operands and create TMEM offset tensors for double-buffered accumulators."""
        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)

        tStS = qk_thr_mma.make_fragment_C(
            qk_thr_mma.partition_shape_C(
                (self.config.qk_mma_tiler[0], self.config.qk_mma_tiler[1])
            )
        )
        tOtO = pv_thr_mma.make_fragment_C(
            pv_thr_mma.partition_shape_C(
                (self.config.pv_mma_tiler[0], self.config.pv_mma_tiler[1])
            )
        )

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem.s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem.s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem.o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem.o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        p_scale = self.config.qk_acc_dtype.width // self.q_dtype.width
        tOrP0 = cute.make_tensor(
            tOrP.iterator + p_scale * self.tmem.p0_offset, tOrP.layout
        )
        tOrP1 = cute.make_tensor(
            tOrP.iterator + p_scale * self.tmem.p1_offset, tOrP.layout
        )

        return (
            qk_thr_mma,
            pv_thr_mma,
            tSrQ,
            tSrK,
            tOrV,
            tStS,
            tStS0,
            tStS1,
            tOtO0,
            tOtO1,
            tOrP0,
            tOrP1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        qk_tiled_mma: cute.TiledMma,
        pv_tiled_mma: cute.TiledMma,
        tma_atom_q: cute.CopyAtom,
        mQ_qdl: cute.Tensor,
        tma_atom_k: cute.CopyAtom,
        mK_kdl: cute.Tensor,
        tma_atom_v: cute.CopyAtom,
        mV_dkl: cute.Tensor,
        tma_atom_o: cute.CopyAtom,
        mO_qdl: cute.Tensor,
        cum_seqlen_q: cute.Tensor | None,
        cum_seqlen_k: cute.Tensor | None,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        params: cute.Tensor | None,
        q_smem_layout_staged: cute.ComposedLayout,
        k_smem_layout_staged: cute.ComposedLayout,
        p_tmem_layout_staged: cute.ComposedLayout,
        v_smem_layout_staged: cute.ComposedLayout,
        o_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: FmhaStaticTileSchedulerParams,
    ):
        """FMHA device kernel: warp-specialized attention with pipelined TMA loads."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()

        if warp_idx == self.schedule.load_warp_id:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        pipes = self._create_pipelines(storage)
        load_q_producer, load_q_consumer = pipes["load_q"]
        load_kv_producer, load_kv_consumer = pipes["load_kv"]
        mma_s0_producer, mma_s0_consumer = pipes["mma_s0"]
        mma_s1_producer, mma_s1_consumer = pipes["mma_s1"]
        s0_s1_sequence_producer, s0_s1_sequence_consumer = pipes["s0_s1_sequence"]
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        # Standard path pipelines (correction warp)
        s0_corr_producer = s0_corr_consumer = None
        s1_corr_producer = s1_corr_consumer = None
        corr_epi_producer = corr_epi_consumer = None
        mma_corr_producer = mma_corr_consumer = None
        if cutlass.const_expr(not self.has_logits_transform):
            s0_corr_producer, s0_corr_consumer = pipes["s0_corr"]
            s1_corr_producer, s1_corr_consumer = pipes["s1_corr"]
            corr_epi_producer, corr_epi_consumer = pipes["corr_epi"]
            mma_corr_producer, mma_corr_consumer = pipes["mma_corr"]

        # Transform path pipelines (softmax -> epilogue)
        s0_epi_producer = s0_epi_consumer = None
        s1_epi_producer = s1_epi_consumer = None
        if cutlass.const_expr(self.has_logits_transform):
            s0_epi_producer, s0_epi_consumer = pipes["s0_epi"]
            s1_epi_producer, s1_epi_consumer = pipes["s1_epi"]

        if warp_idx == self.schedule.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.schedule.tmem_dealloc_arrive_count,
            )
        cute.arch.mbarrier_init_fence()

        sQ = storage.sQ.get_tensor(
            q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner
        )
        sK = storage.sK.get_tensor(
            k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner
        )
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner),
            v_smem_layout_staged.outer,
        )
        sO = storage.sO.get_tensor(
            o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner
        )

        (
            qk_thr_mma,
            pv_thr_mma,
            tSrQ,
            tSrK,
            tOrV,
            tStS,
            tStS0,
            tStS1,
            tOtO0,
            tOtO1,
            tOrP0,
            tOrP1,
        ) = self._create_mma_fragments(
            qk_tiled_mma,
            pv_tiled_mma,
            sQ,
            sK,
            sV,
            p_tmem_layout_staged,
        )

        cute.arch.barrier(
            barrier_id=self.schedule.cta_sync_bar_id,
            number_of_threads=self.schedule.threads_per_cta,
        )
        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_empty)

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_other)
            self.loader_role.run(
                qk_thr_mma,
                pv_thr_mma,
                tma_atom_q,
                tma_atom_k,
                tma_atom_v,
                mQ_qdl,
                mK_kdl,
                mV_dkl,
                sQ,
                sK,
                sV,
                cum_seqlen_q,
                cum_seqlen_k,
                load_q_producer,
                load_kv_producer,
                tile_sched_params,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_other)
            self.mma_role.run(
                qk_tiled_mma,
                pv_tiled_mma,
                tStS0,
                tStS1,
                tOtO0,
                tOtO1,
                tSrQ,
                tSrK,
                tOrP0,
                tOrP1,
                tOrV,
                mQ_qdl.shape[0],
                mK_kdl.shape[0],
                cum_seqlen_q,
                cum_seqlen_k,
                load_q_consumer,
                load_kv_consumer,
                mma_s0_producer,
                mma_s1_producer,
                mma_corr_producer,  # None for transform path
                tile_sched_params,
                storage,
                tmem_dealloc_mbar_ptr,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_other)
            self.epilogue_role.run(
                tma_atom_o,
                mO_qdl,
                sO,
                cum_seqlen_q,
                corr_epi_consumer,  # None for transform path
                s0_epi_consumer,  # None for standard path
                s1_epi_consumer,  # None for standard path
                tile_sched_params,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax0
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx < self.schedule.softmax1_warp_ids[0]:
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.schedule.num_regs_softmax)

            self.softmax_role.run(
                stage=0,
                seqlen_q=mQ_qdl.shape[0],
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                scale_output=scale_output,
                qk_thr_mma=qk_thr_mma,
                pv_thr_mma=pv_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                tOtO=tOtO0,
                sO=sO[None, None, 0] if self.has_logits_transform else None,
                params=params,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                si_epi_producer=s0_epi_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.softmax1_warp_ids[0]
            and warp_idx < self.schedule.softmax1_upper_warp_id
        ):
            # increase register after decreasing
            cute.arch.warpgroup_reg_alloc(self.schedule.num_regs_softmax)

            self.softmax_role.run(
                stage=1,
                seqlen_q=mQ_qdl.shape[0],
                seqlen_k=mK_kdl.shape[0],
                cum_seqlen_q=cum_seqlen_q,
                cum_seqlen_k=cum_seqlen_k,
                scale_softmax_log2=scale_softmax_log2,
                scale_output=scale_output,
                qk_thr_mma=qk_thr_mma,
                pv_thr_mma=pv_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                tOtO=tOtO1,
                sO=sO[None, None, 1] if self.has_logits_transform else None,
                params=params,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                si_epi_producer=s1_epi_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(not self.has_logits_transform):
            if (
                warp_idx >= self.schedule.softmax1_upper_warp_id
                and warp_idx < self.schedule.mma_warp_id
            ):
                cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_correction)
                self.correction_role.run(
                    qk_thr_mma,
                    pv_thr_mma,
                    tStS,
                    tOtO0,
                    tOtO1,
                    sO,
                    mQ_qdl.shape[0],
                    mK_kdl.shape[0],
                    cum_seqlen_q,
                    cum_seqlen_k,
                    scale_softmax_log2,
                    scale_output,
                    s0_corr_consumer,
                    s1_corr_consumer,
                    mma_corr_consumer,
                    corr_epi_producer,
                    tile_sched_params,
                    tmem_dealloc_mbar_ptr,
                )
        return

    @staticmethod
    def _compute_grid(
        o_shape: cute.Shape,
        cta_tiler: Tuple[int, int, int],
        is_persistent: bool,
    ) -> Tuple[FmhaStaticTileSchedulerParams, Tuple[int, int, int]]:
        tile_sched_params = create_fmha_static_tile_scheduler_params(
            is_persistent,
            (
                cute.ceil_div(cute.size(o_shape[0]), cta_tiler[0]),
                cute.size(o_shape[2][0]),
                cute.size(o_shape[2][1]),
            ),
        )
        grid = FmhaStaticTileScheduler.get_grid_shape(tile_sched_params)
        return tile_sched_params, grid
