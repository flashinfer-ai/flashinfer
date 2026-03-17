# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.

# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.

# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import math
from typing import Type, Tuple, Optional

import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.utils as utils
from cutlass.cute.typing import Int32, Float32, Boolean

from .config import AttentionConfig, AttentionFusion
from .tmem_layout import TmemLayout
from .warp_schedule import WarpSchedule, PREFILL_SCHEDULE
from .pipeline_topology import PipelineTopology, make_prefill_topology
from .mainloop_spec import MainloopSpec, make_prefill_mainloop_spec
from .collective_builder import build_fmha_launch_params
from .fusion.mask import (
    MaskType,
    apply_mask,
    get_trip_count,
    get_masked_trip_count,
    get_unmasked_trip_count,
    get_kv_start_block_idx,
)
from .scheduler.persistent import (
    FmhaStaticTileScheduler,
    FmhaStaticTileSchedulerParams,
    create_fmha_static_tile_scheduler,
    create_fmha_static_tile_scheduler_params,
)
from .roles.softmax import SoftmaxRole
from .roles.correction import CorrectionRole
from .roles.epilogue import EpilogueRole
from .roles.loader_tma import LoaderRole
from .roles.mma import MmaRole

import warnings

# Ignore this specific warning
warnings.filterwarnings(
    "ignore",
    message="This loop is no longer unrolled and may cause performance regression",
)

# Or ignore all UserWarnings (more broad)
warnings.filterwarnings("ignore", category=UserWarning)

"""
A fused multi-head attention (FMHA) example for the NVIDIA Blackwell SM100 architecture using CUTE DSL

This example demonstrates an implementation of fused multi-head attention using a TMA + Blackwell SM100
TensorCore warp-specialized persistent kernel. The implementation integrates the Q*K^T matrix multiplication,
softmax normalization, and softmax(Q*K^T)*V into a single kernel, avoiding intermediate data movement between
global memory and shared memory, thus improving computational efficiency.

The kernel implements key optimizations including:
- Warp specialization for different computation phases (load, MMA, softmax, correction, epilogue)
- Pipeline stages between different warps for overlapping computation and memory access
- Support for different precision data types
- Optional causal masking for autoregressive models

To run this example:

.. code-block:: bash

    python examples/blackwell/fmha.py                                     \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent

The above example runs FMHA with batch size 4, sequence length 1024, 8 attention heads, and head
dimension 64. The Blackwell tcgen05 MMA tile shape is (128, 128), and the kernel uses fp16 for input/output
with fp32 for accumulation.

To collect performance with NCU profiler:

.. code-block:: bash

    ncu python examples/blackwell/fmha.py                                 \
      --qk_acc_dtype Float32 --pv_acc_dtype Float32                       \
      --mma_tiler_mn 128,128                                              \
      --q_shape 4,1024,8,64 --k_shape 4,1024,8,64                         \
      --is_persistent --warmup_iterations 10                              \
      --iterations 10 --skip_ref_check

Constraints for this example:
* Supported head dimensions: 32, 64, and 128
* Number of heads in Q must be divisible by number of heads in K
* mma_tiler_mn must be 128,128
* Batch size must be the same for Q, K, and V tensors
* For causal masking, use --is_causal (note: specify without =True/False)
* For persistent scheduling, use --is_persistent (note: specify without =True/False)
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
        self.schedule = warp_schedule if warp_schedule is not None else PREFILL_SCHEDULE
        self.mainloop = make_prefill_mainloop_spec(config, self.schedule)
        self.tmem = self.mainloop.tmem_layout

    @cute.jit
    def __call__(
        self,
        q_iter: cute.Pointer,
        k_iter: cute.Pointer,
        v_iter: cute.Pointer,
        o_iter: cute.Pointer,
        problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32],
        cum_seqlen_q: cute.Tensor | None,
        s_q_all: Int32,
        cum_seqlen_k: cute.Tensor | None,
        s_k_all: Int32,
        scale_softmax_log2: Float32,
        scale_output: Float32,
        sink_iter: cute.Pointer | None,
        stream: cuda.CUstream,
    ):
        """Execute the Fused Multi-Head Attention operation on the provided tensors.

        This method prepares the input tensors for processing, validates their shapes and types,
        configures the computation parameters, and launches the CUDA kernel.

        The method handles:
        1. Tensor layout transformations for specific memory access patterns
        2. Validation of tensor shapes and data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch with appropriate parameters

        :param q_iter: The query tensor pointer
        :type q_iter: cute.Pointer
        :param k_iter: The key tensor pointer
        :type k_iter: cute.Pointer
        :param v_iter: The value tensor pointer
        :type v_iter: cute.Pointer
        :param o_iter: The output tensor pointer
        :type o_iter: cute.Pointer
        :param problem_size: The problem size with shape [b, s_q, s_k, h_q, h_k, d]. If cum_seqlen_q or cum_seqlen_k is not None, s_q and s_k are the max of the cumulative sequence length respectively.
        :type problem_size: Tuple[Int32, Int32, Int32, Int32, Int32, Int32]
        :param cum_seqlen_q: The cumulative sequence length tensor for query
        :type cum_seqlen_q: cute.Tensor | None
        :param cum_seqlen_k: The cumulative sequence length tensor for key
        :type cum_seqlen_k: cute.Tensor | None
        :param scale_softmax_log2: The log2 scale factor for softmax
        :type scale_softmax_log2: Float32
        :param scale_output: The scale factor for the output
        :type scale_output: Float32
        :param sink_iter: The sink tensor pointer
        :type sink_iter: cute.Pointer | None
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream
        :raises TypeError: If tensor data types don't match or aren't supported
        :raises RuntimeError: If tensor layouts aren't in supported formats
        """
        b, s_q, s_k, h_q, h_k, d = problem_size
        h_r = h_q // h_k

        q_offset = 0
        kv_offset = 0
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
        q = cute.make_tensor(q_iter + q_offset, q_layout)
        # (s, d, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        k_layout = cute.make_layout(
            (s_k_all, d, ((h_r, h_k), b_kv)),
            stride=(d * h_k, 1, ((0, d), stride_b_kv)),
        )
        k = cute.make_tensor(k_iter + kv_offset, k_layout)
        # (d, s, ((h_r, h_k), b)), 0-stride for h_r to broadcast
        v_layout = cute.make_layout(
            (d, s_k_all, ((h_r, h_k), b_kv)),
            stride=(1, d * h_k, ((0, d), stride_b_kv)),
        )
        v = cute.make_tensor(v_iter + kv_offset, v_layout)
        # (s, d, ((h_r, h_k), b))
        o_layout = cute.make_layout(
            (s_q, d, ((h_r, h_k), b_o)),
            stride=(d * h_r * h_k, 1, ((d, d * h_r), stride_b_o)),
        )
        o = cute.make_tensor(o_iter + o_offset, o_layout)

        sink = (
            cute.make_tensor(sink_iter, cute.make_layout((h_q,)))
            if self.fusion.use_attention_sink
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
        self.mainloop.resolve(self.q_dtype.width)

        self.softmax_role = SoftmaxRole(
            self.config, self.fusion, self.tmem,
            softmax0_warp_ids=self.schedule.softmax0_warp_ids,
            softmax1_warp_ids=self.schedule.softmax1_warp_ids,
            threads_per_warp=self.schedule.threads_per_warp,
        )
        self.correction_role = CorrectionRole(
            self.config, self.fusion, self.tmem,
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
        )
        self.softmax_role.set_dtypes(self.q_dtype, self.o_dtype)

        lp = build_fmha_launch_params(
            self.mainloop, q, k, v, o,
            self.q_dtype, self.k_dtype, self.v_dtype, self.o_dtype,
            self.q_major_mode, self.k_major_mode, self.v_major_mode, self.o_layout,
        )
        self.shared_storage = lp.SharedStorage
        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kv_bytes = lp.tma_copy_kv_bytes

        self.correction_role.set_call_attrs(self.o_dtype, lp.o_layout, lp.epi_tile)

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
            sink,
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
            barrier_ptrs, tx_counts, self.schedule.threads_per_warp,
        )

    @cute.jit
    def _create_mma_fragments(
        self, qk_tiled_mma, pv_tiled_mma, sQ, sK, sV, p_tmem_layout_staged,
    ):
        """Partition MMA operands and create TMEM offset tensors for double-buffered accumulators."""
        qk_thr_mma = qk_tiled_mma.get_slice(0)
        pv_thr_mma = pv_tiled_mma.get_slice(0)
        tSrQ = qk_thr_mma.make_fragment_A(sQ)
        tSrK = qk_thr_mma.make_fragment_B(sK)
        tOrV = pv_thr_mma.make_fragment_B(sV)

        tStS = qk_thr_mma.make_fragment_C(qk_thr_mma.partition_shape_C(
            (self.config.qk_mma_tiler[0], self.config.qk_mma_tiler[1])
        ))
        tOtO = pv_thr_mma.make_fragment_C(pv_thr_mma.partition_shape_C(
            (self.config.pv_mma_tiler[0], self.config.pv_mma_tiler[1])
        ))

        tStS0 = cute.make_tensor(tStS.iterator + self.tmem.s0_offset, tStS.layout)
        tStS1 = cute.make_tensor(tStS.iterator + self.tmem.s1_offset, tStS.layout)
        tOtO0 = cute.make_tensor(tOtO.iterator + self.tmem.o0_offset, tOtO.layout)
        tOtO1 = cute.make_tensor(tOtO.iterator + self.tmem.o1_offset, tOtO.layout)

        tP = cute.make_tensor(tStS.iterator, p_tmem_layout_staged.outer)
        tOrP = pv_thr_mma.make_fragment_A(tP)[None, None, None, 0]
        p_scale = self.config.qk_acc_dtype.width // self.q_dtype.width
        tOrP0 = cute.make_tensor(tOrP.iterator + p_scale * self.tmem.p0_offset, tOrP.layout)
        tOrP1 = cute.make_tensor(tOrP.iterator + p_scale * self.tmem.p1_offset, tOrP.layout)

        return (
            qk_thr_mma, pv_thr_mma,
            tSrQ, tSrK, tOrV, tStS,
            tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1,
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
        sink: cute.Tensor | None,
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
        s0_corr_producer, s0_corr_consumer = pipes["s0_corr"]
        s1_corr_producer, s1_corr_consumer = pipes["s1_corr"]
        corr_epi_producer, corr_epi_consumer = pipes["corr_epi"]
        mma_corr_producer, mma_corr_consumer = pipes["mma_corr"]
        s0_s1_sequence_producer, s0_s1_sequence_consumer = pipes["s0_s1_sequence"]
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()

        if warp_idx == self.schedule.empty_warp_id:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr,
                self.schedule.tmem_dealloc_arrive_count,
            )
        cute.arch.mbarrier_init_fence()

        sQ = storage.sQ.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sK = storage.sK.get_tensor(k_smem_layout_staged.outer, swizzle=k_smem_layout_staged.inner)
        sV = cute.make_tensor(cute.recast_ptr(sK.iterator, v_smem_layout_staged.inner), v_smem_layout_staged.outer)
        sO = storage.sO.get_tensor(o_smem_layout_staged.outer, swizzle=o_smem_layout_staged.inner)

        (
            qk_thr_mma, pv_thr_mma,
            tSrQ, tSrK, tOrV, tStS,
            tStS0, tStS1, tOtO0, tOtO1, tOrP0, tOrP1,
        ) = self._create_mma_fragments(
            qk_tiled_mma, pv_tiled_mma, sQ, sK, sV, p_tmem_layout_staged,
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
                qk_thr_mma, pv_thr_mma,
                tma_atom_q, tma_atom_k, tma_atom_v,
                mQ_qdl, mK_kdl, mV_dkl,
                sQ, sK, sV,
                cum_seqlen_q, cum_seqlen_k,
                load_q_producer, load_kv_producer,
                tile_sched_params,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_other)
            self.mma_role.run(
                qk_tiled_mma, pv_tiled_mma,
                tStS0, tStS1, tOtO0, tOtO1,
                tSrQ, tSrK, tOrP0, tOrP1, tOrV,
                mQ_qdl.shape[0], mK_kdl.shape[0],
                cum_seqlen_q, cum_seqlen_k,
                load_q_consumer, load_kv_consumer,
                mma_s0_producer, mma_s1_producer,
                mma_corr_producer,
                tile_sched_params, storage, tmem_dealloc_mbar_ptr,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.epilogue_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_other)
            self.epilogue_role.run(
                tma_atom_o, mO_qdl, sO,
                cum_seqlen_q,
                corr_epi_consumer,
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
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS0,
                sink=sink,
                mma_si_consumer=mma_s0_consumer,
                si_corr_producer=s0_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax1
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx < self.schedule.correction_warp_ids[0]
            and warp_idx >= self.schedule.softmax1_warp_ids[0]
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
                qk_thr_mma=qk_thr_mma,
                tStS=tStS,
                tStSi=tStS1,
                sink=sink,
                mma_si_consumer=mma_s1_consumer,
                si_corr_producer=s1_corr_producer,
                s0_s1_sequence_consumer=s0_s1_sequence_consumer,
                s0_s1_sequence_producer=s0_s1_sequence_producer,
                tile_sched_params=tile_sched_params,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.schedule.correction_warp_ids[0] and warp_idx < self.schedule.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.schedule.num_regs_correction)
            self.correction_role.run(
                qk_thr_mma, pv_thr_mma,
                tStS, tOtO0, tOtO1, sO,
                mQ_qdl.shape[0], mK_kdl.shape[0],
                cum_seqlen_q, cum_seqlen_k,
                scale_softmax_log2, scale_output,
                s0_corr_consumer, s1_corr_consumer,
                mma_corr_consumer, corr_epi_producer,
                tile_sched_params, tmem_dealloc_mbar_ptr,
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
