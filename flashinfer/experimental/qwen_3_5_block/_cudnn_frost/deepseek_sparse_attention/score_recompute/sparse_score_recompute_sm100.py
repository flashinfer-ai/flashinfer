# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Sparse Backward Score Kernel — SM100 Cute-DSL Implementation.

Dual-mode kernel controlled by score_type ("indexer" or "attention"):

  Indexer mode (score_type="indexer"):
    S[b,q,i] = sum_h [ReLU(Q_h · K_{topk[b,q,i]}^T) · W_{b,q,h}]
    predict[b,q,:] = softmax(S[b,q,:])  (over topk dim)

  Attention mode (score_type="attention"):
    P[b,q,h,i] = exp(Q_h · K_{topk[b,q,i]}^T · scale - LSE[b,q,h])
    S[b,q,i] = sum_h P[b,q,h,i]
    target[b,q,:] = S[b,q,:] / sum(S[b,q,:])  (L1-norm over topk dim)

K is sparse-loaded via cp.async based on topk_indices (not TMA).

Design: SwapAB (K as A, Q_packed as B), PackGQA,
        1 epilogue warpgroup.
        WG2 (128 threads) loads K in SM90-style gather (16 groups x 8 threads).
        CLC persistent scheduling.
        Optional k_block_size splits head_dim into chunks for reduced sK SMEM.
"""

import math
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineTmaUmma,
    PipelineAsyncUmma,
    PipelineUserType,
    make_pipeline_state,
    pipeline_init_arrive,
    pipeline_init_wait,
    PipelineClcFetchAsync,
)
import cutlass.utils as utils
from cutlass.utils.blackwell_helpers import (
    make_trivial_tiled_mma as _make_trivial_tiled_mma,
    make_smem_layout_a as _make_smem_layout_a,
    make_smem_layout_b as _make_smem_layout_b,
)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import LayoutEnum, SmemAllocator

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm100.gemm import gemm_ptx_partial as _gemm_ptx_partial
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils import copy as copy_utils

mul_packed_f32x2 = partial(cute.arch.mul_packed_f32x2, rnd="rn")
add_packed_f32x2 = partial(cute.arch.add_packed_f32x2, rnd="rn")
fma_packed_f32x2 = partial(cute.arch.fma_packed_f32x2, rnd="rn")


class SparseScoreRecomputeSm100:
    """
    SM100 Cute-DSL kernel for sparse backward score recomputation.

    Dual-mode via score_type:
      - "indexer": ReLU(QK) * W -> head reduce -> softmax
      - "attention": exp(QK*scale - LSE) -> head reduce -> L1-norm

    SwapAB design:
      - A = K (n_tile x head_dim), sparse-loaded via cp.async
      - B = Q_packed (m_tile x head_dim), loaded via TMA
      - C = S^T in TMEM: (n_tile, m_tile)

    Warp layout (12 warps total):
      - Warp 0:     Load (TMA Q, per-head data, topk indices)
      - Warp 1:     MMA  (QK GEMM via TCGen05, swapAB)
      - Warp 2:     CLC scheduler (producer)
      - Warp 3:     Idle
      - Warps 4-7:  Epilogue warpgroup (score reduction + normalization)
      - Warps 8-11: K loading warpgroup (sparse cp.async gather)
    """

    arch = 100
    WARP_SIZE = 32
    WARPGROUP_SIZE = 128

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 32,
        m_block_size: int = 32,
        n_block_size: int = 128,
        topk: int = 512,
        kv_stage: int = 4,
        score_type: str = "indexer",
        have_topk_length: bool = False,
        topk_in_smem: bool = True,
        k_block_size: int | None = None,
        topk_indices_global: bool = True,
    ):
        assert score_type in ("indexer", "attention")
        self.score_type = score_type
        self.head_dim = head_dim
        self.qhead_per_kvhead = qhead_per_kvhead
        self.m_block_size = m_block_size
        self.n_block_size = n_block_size
        self.topk = topk
        self.kv_stage = kv_stage
        self.topk_indices_global = topk_indices_global
        self.sched_warp_id = 2
        self.num_clc_stage = 1
        self.num_clc_response_bytes = 16
        assert topk % n_block_size == 0, f"topk ({topk}) must be a multiple of n_block_size ({n_block_size})"
        self.num_n_blocks = topk // n_block_size
        self.have_topk_length = have_topk_length
        self.topk_in_smem = topk_in_smem

        # Pad head_dim to multiple of 16
        hdim_multiple_of = 16
        self.head_dim_padded = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        assert self.head_dim_padded % 64 == 0, (
            f"head_dim_padded ({self.head_dim_padded}) must be a multiple of 64 " f"(K loading uses 8 threads × 8 elements = 64 per iteration)"
        )

        # K-dimension splitting: process head_dim in num_k_chunks chunks of
        # k_block_size each, reducing sK SMEM per stage and enabling more
        # kv_stages for better K load / MMA overlap.
        self.k_block_size = k_block_size if k_block_size is not None else self.head_dim_padded
        assert self.head_dim_padded % self.k_block_size == 0, (
            f"head_dim_padded ({self.head_dim_padded}) must be a multiple of " f"k_block_size ({self.k_block_size})"
        )
        assert self.k_block_size % 64 == 0, (
            f"k_block_size ({self.k_block_size}) must be a multiple of 64 " f"(K loading uses 8 threads × 8 elements = 64 per iteration)"
        )
        self.num_k_chunks = self.head_dim_padded // self.k_block_size

        self.q_tokens_per_tile = m_block_size // qhead_per_kvhead
        assert self.q_tokens_per_tile == 1, "Only 1 q_token per tile is supported"

        # swapAB: K is A (M-dim=n_block), Q_packed is B (N-dim=m_block)
        # K dimension of the MMA tiler is k_block_size (not full head_dim).
        self.mma_tiler_qk = (n_block_size, m_block_size, self.k_block_size)
        self.qk_acc_dtype = Float32

        # Warp config: 12 warps
        self.load_warp_id = 0
        self.mma_warp_id = 1
        self.epilogue_wg_warp_ids = (4, 5, 6, 7)
        self.k_load_wg_warp_ids = (8, 9, 10, 11)
        self.num_warps = 12
        self.threads_per_cta = self.WARP_SIZE * self.num_warps  # 384
        # Register allocation
        # NOTE: num_regs_epilogue=256 accommodates pre-loading all per-head
        # data (W/LSE) into registers before the n_block loop, which requires
        # m_block_size BF16 (indexer) or m_block_size F32 (attention) extra regs.
        # With m_block_size=qhead_per_kvhead=64, this needs 64 BF16=32 F32 regs
        # (indexer) or 64 F32 regs (attention), fitting within 256. For
        # qhead_per_kvhead=128, this would need 128 BF16=64 or 128 F32 regs,
        # likely causing spills. In that case, fall back to the per-n_block
        # autovec_copy pattern or use a chunked pre-load strategy.
        self.num_regs_load = 48
        self.num_regs_mma = 48
        self.num_regs_epilogue = 256

        # TMEM: accumulator slots reused across n_blocks (multi-round when
        # num_n_blocks > num_tmem_slots). S barriers are sized to match slots,
        # with phase cycling handling cross-round synchronization.
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_s_stride = self.m_block_size
        self.num_tmem_slots = min(
            self.num_n_blocks,
            SM100_TMEM_CAPACITY_COLUMNS // self.m_block_size,
        )
        self.tmem_total = self.tmem_s_stride * self.num_tmem_slots
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # Pipeline barrier counts (Q uses 1 stage with combined tx_count)
        self.Q_mbar_size = 2
        self.K_mbar_size = 2 * self.kv_stage
        # S barriers: paired (s_full, s_empty) per TMEM slot (not per n_block).
        # Layout: [s_full[0], s_empty[0], s_full[1], s_empty[1], ...]
        # When num_n_blocks > num_tmem_slots, slots and barriers are reused
        # across rounds with phase cycling.
        self.S_mbar_size = 2 * self.num_tmem_slots
        self.s_empty_arrive_count = self.WARPGROUP_SIZE

        # Cross-warp reduce sync barrier: warpgroup (128 threads) sync during score reduction
        self.reduce_sync_mbar_size = 2
        self.reduce_sync_arrive_count = self.WARPGROUP_SIZE

        # sScoreAll: cross-warp scratch for reduce (one entry per warp in epilogue warpgroup)
        # Indexer epilogue uses 2 reduce passes (max + sum) that reuse the same buffer.
        # Allocate 2x to give each pass its own region, avoiding potential reuse races.
        self.num_warps_in_epi_wg = len(self.epilogue_wg_warp_ids)
        self.sScoreAll_size = self.num_warps_in_epi_wg * 2

        # SMEM alignment
        self.buffer_align_bytes = 1024
        # Cluster shape (no multi-CTA clustering for simplicity)
        self.cluster_shape_mn = (1, 1)

    # =========================================================================
    # Host-side entry: layout transpose, PackGQA, TMA creation, launch
    # =========================================================================
    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (bs, seqlen_q, n_heads_q, head_dim) BF16
        mK: cute.Tensor,  # (bs, seqlen_k, head_dim) BF16
        mPerHead: cute.Tensor,  # (bs, seqlen_q, n_heads_q) — W (BF16) or LSE (FP32)
        mTopkIdx: cute.Tensor,  # (bs, seqlen_q, topk) INT32
        mOut: cute.Tensor,  # (bs, seqlen_q, topk) FP32
        mTopkLength: cute.Tensor | None,  # (bs, seqlen_q) INT32; None (compiled out) unless have_topk_length
        softmax_scale: Float32 | float,
        stream: cuda.CUstream,
    ):
        """Host-side: layout transpose, PackGQA, TMA creation, kernel launch."""

        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.per_head_dtype = mPerHead.element_type

        self.tma_copy_bytes = {
            "Q": self.m_block_size * self.head_dim_padded * (self.q_dtype.width // 8),
        }

        # --- Layout transpose: (bs, seqlen, heads, hdim) -> (seqlen, hdim, heads, bs) ---
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=[1, 3, 2, 0]))

        # --- PackGQA: reshape Q to pack qhpkv heads into seqlen_q dim ---
        shape_Q_packed = (
            (self.qhead_per_kvhead, mQ.shape[0]),  # packed M dim
            mQ.shape[1],  # head_dim
            1,  # n_heads_kv
            *mQ.shape[3:],  # bs (and beyond)
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),  # (stride_head, stride_seqlen)
            mQ.stride[1],  # stride_hdim
            mQ.stride[2] * self.qhead_per_kvhead,  # stride across kv_head groups
            *mQ.stride[3:],  # stride_bs
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        # --- K layout: (bs, seqlen_k, head_dim) -> (seqlen_k, head_dim, bs) ---
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=[1, 2, 0]))

        cta_group = tcgen05.CtaGroup.ONE
        self.q_major_mode = LayoutEnum.from_tensor(mQ).mma_major_mode()

        tiled_mma_qk = _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            cute.nvgpu.OperandMajorMode.K,
            self.q_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # --- SMEM layouts --- (K is A operand in swapAB; loaded via cp.async, not TMA)
        sK_layout = _make_smem_layout_a(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.k_dtype,
            self.kv_stage,
        )
        sQ_layout = _make_smem_layout_b(
            tiled_mma_qk,
            self.mma_tiler_qk,
            self.q_dtype,
            self.num_k_chunks,
        )

        # TMA atom for Q only (K is sparse-loaded, no TMA)
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        self.sQ_layout = sQ_layout
        self.sK_layout = sK_layout

        # --- PerHead layout: (bs, seqlen_q, n_heads_q) -> (seqlen_q, n_heads_q, bs) ---
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=[1, 2, 0]))

        # --- TopkIdx layout: (bs, seqlen_q, topk) -> (seqlen_q, topk, bs) ---
        mTopkIdx = cute.make_tensor(mTopkIdx.iterator, cute.select(mTopkIdx.layout, mode=[1, 2, 0]))

        # --- Output layout: (bs, seqlen_q, topk) -> (seqlen_q, topk, bs) ---
        mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=[1, 2, 0]))

        # --- TopkLength layout: (bs, seqlen_q) -> (seqlen_q, bs); the slot is
        # compiled out (None at compile and launch) when have_topk_length is False ---
        if const_expr(mTopkLength is not None):
            mTopkLength = cute.make_tensor(mTopkLength.iterator, cute.select(mTopkLength.layout, mode=[1, 0]))

        # --- Grid and kernel dispatch (CLC persistent scheduling) ---
        seqlen_q_packed = cute.size(mQ.shape[0])
        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.m_block_size)
        batch_size = cute.size(mQ.shape[3]) if cute.rank(mQ.shape) > 3 else 1
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        self.kernel(
            mQ,
            mK,
            mPerHead,
            mTopkIdx,
            mOut,
            mTopkLength,
            softmax_scale,
            tma_atom_Q,
            tiled_mma_qk,
            sQ_layout,
            sK_layout,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    # =========================================================================
    # Device-side kernel
    # =========================================================================
    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mPerHead,
        mTopkIdx,
        mOut,
        mTopkLength,
        softmax_scale: Float32 | float,
        tma_atom_Q,
        tiled_mma_qk,
        sQ_layout,
        sK_layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
    ):
        """Device-side kernel entry with CLC persistent scheduling."""
        seqlen_q_packed = cute.size(mQ.shape[0])
        seqlen_k = cute.size(mK.shape[0])
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        neg_log2_e = Float32(-math.log2(math.e))

        # --- TMA descriptor prefetch ---
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)

        # =====================================================================
        # Shared memory allocation
        # =====================================================================
        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sPerHead_size = self.m_block_size * 2  # double-buffer for LOAD/EPI overlap
        sTopkIdx_size = self.topk * 2 if self.topk_in_smem else 1

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.S_mbar_size]
            reduce_sync_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.reduce_sync_mbar_size]
            metadata_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 4]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sPerHead: cute.struct.Align[
                cute.struct.MemRange[self.per_head_dtype, sPerHead_size],
                128,
            ]
            sTopkIdx: cute.struct.Align[
                cute.struct.MemRange[Int32, sTopkIdx_size],
                128,
            ]
            sScoreAll: cute.struct.Align[
                cute.struct.MemRange[Float32, self.sScoreAll_size],
                self.buffer_align_bytes,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size],
                self.buffer_align_bytes,
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sK_size],
                self.buffer_align_bytes,
            ]

        smem = SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # Extract all pointers/tensors from storage so no 'if' body references storage
        # (avoids DSLRuntimeError: SharedStorage as user-defined Python object in if)
        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        K_mbar_ptr = storage.K_mbar_ptr.data_ptr()
        S_mbar_ptr = storage.S_mbar_ptr.data_ptr()
        reduce_sync_mbar_ptr = storage.reduce_sync_mbar_ptr.data_ptr()
        metadata_mbar_ptr = storage.metadata_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sPerHead = storage.sPerHead.get_tensor(cute.make_layout((self.m_block_size,), stride=(1,)))
        sTopkIdx = storage.sTopkIdx.get_tensor(cute.make_layout((self.topk if self.topk_in_smem else 1,), stride=(1,)))
        sScoreAll = storage.sScoreAll.get_tensor(cute.make_layout((self.sScoreAll_size,), stride=(1,)))

        # =====================================================================
        # Pipeline setup
        # =====================================================================
        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        # Q pipeline (1 stage: all k_chunks share one barrier with combined tx_count)
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # K pipeline (cp.async, WG2 = 128 threads)
        pipeline_K = PipelineAsyncUmma.create(
            barrier_storage=K_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=CooperativeGroup(Agent.Thread, self.WARPGROUP_SIZE),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            defer_sync=True,
        )

        # S barriers: paired (s_full, s_empty) per TMEM slot
        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, 1)
        if warp_idx == 0:
            for _si in cutlass.range_constexpr(self.num_tmem_slots):
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si, 1)
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si + 1, self.s_empty_arrive_count)
            cute.arch.mbarrier_init(reduce_sync_mbar_ptr, self.reduce_sync_arrive_count)
            for _mi in cutlass.range_constexpr(2):
                cute.arch.mbarrier_init(metadata_mbar_ptr + 2 * _mi, self.WARP_SIZE)
                cute.arch.mbarrier_init(metadata_mbar_ptr + 2 * _mi + 1, self.WARPGROUP_SIZE)

        # CLC persistent scheduling pipeline
        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = self.WARP_SIZE * (1 + cluster_size * (1 + 1 + 8))  # sched(1) + load(1) + mma(1) + epilogue+Kload(8) warps
        clc_pipeline = PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.num_clc_stage,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, num_clc_consumer_threads),
            tx_count=self.num_clc_response_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=cluster_layout_vmnk, is_relaxed=True)
        cute.arch.sync_threads()
        pipeline_init_wait(cluster_shape_mn=cluster_layout_vmnk)

        clc_consumer_state = make_pipeline_state(PipelineUserType.Consumer, self.num_clc_stage)
        tile_sched = utils.ClcDynamicPersistentTileScheduler.create(
            tile_sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            clc_response_ptr,
        )
        work_tile = tile_sched.initial_work_tile_info()

        Q_producer, Q_consumer = pipeline_Q.make_participants()

        # TMEM accumulator layout reference (per-n_block tensors constructed in epilogue)
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS_ref = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        warp_group_idx = tidx // self.WARPGROUP_SIZE

        # =====================================================================
        # Warpgroup 0 (warps 0-3): Load + MMA + Scheduler
        # =====================================================================
        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            # -----------------------------------------------------------------
            # Load warp (warp 0): PerHead data + TopkIdx + TMA Q
            # -----------------------------------------------------------------
            if warp_idx == self.load_warp_id:
                seqlen_q_load = seqlen_q_packed // self.qhead_per_kvhead
                rows_per_thread = cute.ceil_div(self.m_block_size, self.WARP_SIZE)
                topk_rows_per_thread = cute.ceil_div(self.topk, self.WARP_SIZE)
                lane_id = tidx % self.WARP_SIZE
                tile_count = Int32(0)

                while work_tile.is_valid_tile:
                    m_block = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]

                    # Metadata has its own producer/consumer handoff. Q/K MMA
                    # completion does not order every metadata writer, and a
                    # later Q tile can otherwise overwrite a live metadata slot.
                    metadata_slot = tile_count % 2
                    metadata_phase = (tile_count // 2) & 1
                    cute.arch.mbarrier_wait(metadata_mbar_ptr + 2 * metadata_slot + 1, metadata_phase ^ 1)

                    # PerHead data load (double-buffered)
                    per_head_buf_off = (tile_count % 2) * self.m_block_size
                    for ri in cutlass.range_constexpr(rows_per_thread):
                        row = ri * self.WARP_SIZE + lane_id
                        if row < self.m_block_size:
                            m_packed_idx = m_block * self.m_block_size + row
                            m_idx = m_packed_idx // self.qhead_per_kvhead
                            h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
                            if m_idx < seqlen_q_load:
                                per_head_value = mPerHead[m_idx, h_idx, batch_idx]
                                if cutlass.const_expr(self.score_type == "attention"):
                                    sPerHead[per_head_buf_off + row] = Float32(per_head_value) * neg_log2_e
                                else:
                                    sPerHead[per_head_buf_off + row] = per_head_value
                            else:
                                sPerHead[per_head_buf_off + row] = self.per_head_dtype(0)

                    if const_expr(self.topk_in_smem):
                        # TopkIdx load (double-buffered)
                        topk_idx_buf_off = (tile_count % 2) * self.topk
                        for ri in cutlass.range(topk_rows_per_thread, unroll_full=True):
                            topk_pos = ri * self.WARP_SIZE + lane_id
                            if topk_pos < self.topk:
                                if m_block < seqlen_q_load:
                                    sTopkIdx[topk_idx_buf_off + topk_pos] = mTopkIdx[m_block, topk_pos, batch_idx]
                                else:
                                    sTopkIdx[topk_idx_buf_off + topk_pos] = Int32(-1)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(metadata_mbar_ptr + 2 * metadata_slot)

                    # TMA Q load (1 barrier, num_k_chunks TMA copies)
                    Q_producer.reset()
                    mQ_cur = mQ[None, None, 0, batch_idx]
                    handle_Q = Q_producer.acquire_and_advance()
                    for _kc in cutlass.range_constexpr(self.num_k_chunks):
                        gQ = cute.local_tile(
                            mQ_cur,
                            (self.m_block_size, self.k_block_size),
                            (m_block, _kc),
                        )
                        sQ_cur = sQ[None, None, None, _kc]
                        load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                            tma_atom_Q,
                            0,
                            cute.make_layout(1),
                            gQ,
                            sQ_cur,
                            single_stage=True,
                        )
                        load_Q_fn(tma_bar_ptr=handle_Q.barrier)

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    tile_count = tile_count + 1
                    clc_consumer_state.advance()
                Q_producer.tail()

            # -----------------------------------------------------------------
            # MMA warp (warp 1): QK GEMM via TCGen05
            # -----------------------------------------------------------------
            if warp_idx == self.mma_warp_id:
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
                cute.arch.sync_warp()

                s_empty_phase_bits = Int32((1 << self.num_tmem_slots) - 1)
                K_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.kv_stage)
                while work_tile.is_valid_tile:
                    m_block = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]

                    # Wait for all Q k_chunks (single barrier covers all TMA copies)
                    Q_consumer.reset()
                    handle_Q = Q_consumer.wait_and_advance()

                    tSrK = tiled_mma_qk.make_fragment_A(sK)
                    tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                    qk_mma_op = tiled_mma_qk.op

                    q_token_idx = m_block
                    n_block = Int32(self.num_n_blocks - 1)

                    # Attention (hd=512): skip invalid blocks to save expensive MMA.
                    # Indexer (hd=128): process all blocks — static bounds enable better codegen.
                    if const_expr(self.have_topk_length and self.score_type == "attention"):
                        topK_mma = mTopkLength[q_token_idx, batch_idx]
                        n_block_max = (topK_mma + self.n_block_size - 1) // self.n_block_size
                        while n_block >= n_block_max:
                            slot = n_block % Int32(self.num_tmem_slots)
                            s_empty_phase = self._phase_for_slot(s_empty_phase_bits, slot)
                            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot + 1, s_empty_phase)
                            with cute.arch.elect_one():
                                cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot)
                            s_empty_phase_bits = self._toggle_phase_for_slot(
                                s_empty_phase_bits,
                                slot,
                            )
                            n_block = n_block - 1

                    while n_block >= Int32(0):
                        slot = n_block % Int32(self.num_tmem_slots)
                        s_empty_phase = self._phase_for_slot(s_empty_phase_bits, slot)
                        cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot + 1, s_empty_phase)

                        for _kc in cutlass.range_constexpr(self.num_k_chunks):
                            pipeline_K.consumer_wait(K_mma_state)

                            tSrKi = tSrK[None, None, None, K_mma_state.index]
                            sK_cur = sK[None, None, None, K_mma_state.index]
                            tSrQ_kc = tSrQ[None, None, None, _kc]
                            sQ_kc = sQ[None, None, None, _kc]
                            _gemm_ptx_partial(
                                qk_mma_op,
                                slot * self.tmem_s_stride,
                                tSrKi,
                                tSrQ_kc,
                                sA=sK_cur,
                                sB=sQ_kc,
                                zero_init=const_expr(_kc == 0),
                            )

                            pipeline_K.consumer_release(K_mma_state)
                            K_mma_state.advance()

                        with cute.arch.elect_one():
                            tcgen05.commit(S_mbar_ptr + 2 * slot)

                        s_empty_phase_bits = self._toggle_phase_for_slot(
                            s_empty_phase_bits,
                            slot,
                        )
                        n_block = n_block - 1

                    handle_Q.release()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

                # TMEM dealloc
                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                tmem_ptr_dealloc = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                cute.arch.dealloc_tmem(tmem_ptr_dealloc, Int32(self.tmem_alloc_cols))

            # -----------------------------------------------------------------
            # Scheduler warp (warp 2): CLC producer loop
            # -----------------------------------------------------------------
            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                clc_producer_state = make_pipeline_state(PipelineUserType.ProducerConsumer, self.num_clc_stage)
                while work_tile.is_valid_tile:
                    clc_pipeline.producer_acquire(clc_producer_state)
                    mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                    tile_sched.advance_to_next_work(mbarrier_addr)
                    clc_producer_state.advance()
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                clc_pipeline.producer_tail(clc_producer_state)

        # =====================================================================
        # Warpgroup 1 (warps 4-7): Epilogue — score reduction + normalization
        # =====================================================================
        if warp_group_idx == 1:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            s_full_phase_bits = Int32(0)
            reduce_phase = Int32(0)
            tile_count = Int32(0)
            while work_tile.is_valid_tile:
                m_block = work_tile.tile_idx[0]
                batch_idx = work_tile.tile_idx[2]
                metadata_slot = tile_count % 2
                metadata_phase = (tile_count // 2) & 1
                cute.arch.mbarrier_wait(metadata_mbar_ptr + 2 * metadata_slot, metadata_phase)
                per_head_offset = (tile_count % 2) * self.m_block_size
                topk_idx_offset = (tile_count % 2) * self.topk
                if cutlass.const_expr(self.score_type == "attention"):
                    if cutlass.const_expr(self.n_block_size >= 128):
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_n128(
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr,
                            sTopkIdx,
                            mTopkIdx,
                            mOut,
                            mTopkLength,
                            m_block,
                            batch_idx,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            topk_idx_offset=topk_idx_offset,
                        )
                    else:
                        s_full_phase_bits, reduce_phase = self._epilogue_attention(
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr,
                            sTopkIdx,
                            mTopkIdx,
                            mOut,
                            mTopkLength,
                            m_block,
                            batch_idx,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            topk_idx_offset=topk_idx_offset,
                        )
                else:
                    s_full_phase_bits, reduce_phase = self._epilogue_indexer(
                        tiled_mma_qk,
                        tStS_ref,
                        mPerHead,
                        sPerHead,
                        sScoreAll,
                        S_mbar_ptr,
                        reduce_sync_mbar_ptr,
                        sTopkIdx,
                        mTopkIdx,
                        mOut,
                        mTopkLength,
                        m_block,
                        batch_idx,
                        tidx,
                        s_full_phase_bits,
                        reduce_phase,
                        softmax_scale,
                        per_head_offset=per_head_offset,
                        topk_idx_offset=topk_idx_offset,
                    )
                cute.arch.mbarrier_arrive(metadata_mbar_ptr + 2 * metadata_slot + 1)
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                tile_count = tile_count + 1
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # =====================================================================
        # Warpgroup 2 (warps 8-11): Sparse K loading via cp.async
        # =====================================================================
        if warp_group_idx == 2:
            cute.arch.setmaxregister_decrease(self.num_regs_load)
            wg_tidx = tidx % self.WARPGROUP_SIZE

            async_copy_atom = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.k_dtype,
                num_bits_per_copy=128,
            )
            async_thr_copy = cute.make_tiled_copy_tv(
                async_copy_atom,
                cute.make_layout((1,)),
                cute.make_layout((8,)),
            ).get_slice(0)

            # SM90-style gather: 16 groups x 8 threads, each group loads ROWS_PER_GROUP rows
            GROUP_SIZE = const_expr(8)
            NUM_GROUPS = const_expr(self.WARPGROUP_SIZE // 8)
            ROWS_PER_GROUP = const_expr(self.n_block_size // NUM_GROUPS)
            idx_in_group = wg_tidx % GROUP_SIZE
            group_idx_local = wg_tidx // GROUP_SIZE

            if const_expr(self.score_type == "indexer"):
                # Indexer is K-load-bound (MMA 16 cy vs K gather ~2000 cy/stage).
                # 2-deep cp.async pipeline: issue stage s+1 before waiting for
                # stage s, keeping 2 groups in flight to improve L2 utilization.
                # (Full-depth tested: worse — delaying MMA start hurts more
                # than extra L2 parallelism helps.)
                K_issue_state = make_pipeline_state(PipelineUserType.Producer, self.kv_stage)
                K_commit_state = make_pipeline_state(PipelineUserType.Producer, self.kv_stage)

                while work_tile.is_valid_tile:
                    m_block = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    q_token_idx = m_block
                    topK_cur = Int32(self.topk)
                    n_block_max = (topK_cur + self.n_block_size - 1) // self.n_block_size

                    # -- Prologue: issue first n_block, no wait --
                    n_block = n_block_max - 1
                    for _kc in cutlass.range_constexpr(self.num_k_chunks):
                        pipeline_K.producer_acquire(K_issue_state)
                        sK_stage = sK[None, None, None, K_issue_state.index]
                        sK_slice = cute.composition(
                            sK_stage,
                            cute.make_layout((self.n_block_size, self.k_block_size)),
                        )
                        k_offset = const_expr(_kc * self.k_block_size)
                        self._load_k_rows(
                            mK,
                            sK_slice,
                            mTopkIdx,
                            mTopkLength,
                            async_copy_atom,
                            async_thr_copy,
                            n_block,
                            q_token_idx,
                            batch_idx,
                            seqlen_k,
                            topK_cur,
                            idx_in_group,
                            group_idx_local,
                            k_offset=k_offset,
                        )
                        cute.arch.cp_async_commit_group()
                        K_issue_state.advance()
                    n_block = n_block - 1

                    # -- Main: issue next, wait/commit previous --
                    while n_block >= Int32(0):
                        for _kc in cutlass.range_constexpr(self.num_k_chunks):
                            pipeline_K.producer_acquire(K_issue_state)
                            sK_stage = sK[None, None, None, K_issue_state.index]
                            sK_slice = cute.composition(
                                sK_stage,
                                cute.make_layout((self.n_block_size, self.k_block_size)),
                            )
                            k_offset = const_expr(_kc * self.k_block_size)
                            self._load_k_rows(
                                mK,
                                sK_slice,
                                mTopkIdx,
                                mTopkLength,
                                async_copy_atom,
                                async_thr_copy,
                                n_block,
                                q_token_idx,
                                batch_idx,
                                seqlen_k,
                                topK_cur,
                                idx_in_group,
                                group_idx_local,
                                k_offset=k_offset,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(1)
                            cute.arch.fence_view_async_shared()
                            pipeline_K.producer_commit(K_commit_state)
                            K_commit_state.advance()
                            K_issue_state.advance()
                        n_block = n_block - 1

                    # -- Drain: commit last issued stage --
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    pipeline_K.producer_commit(K_commit_state)
                    K_commit_state.advance()

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

            else:
                # Attention: original K load with wait_group(0).
                # 2-deep tested but regressed -14% (kv_stage=2 too tight,
                # code bloat from prologue/drain outweighs L2 overlap gain).
                K_load_state = make_pipeline_state(PipelineUserType.Producer, self.kv_stage)

                while work_tile.is_valid_tile:
                    m_block = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    q_token_idx = m_block

                    if const_expr(self.have_topk_length):
                        topK_cur = mTopkLength[q_token_idx, batch_idx]
                    else:
                        topK_cur = Int32(self.topk)
                    n_block_max = (topK_cur + self.n_block_size - 1) // self.n_block_size

                    n_block = n_block_max - 1
                    while n_block >= Int32(0):
                        for _kc in cutlass.range_constexpr(self.num_k_chunks):
                            pipeline_K.producer_acquire(K_load_state)
                            sK_stage = sK[None, None, None, K_load_state.index]
                            sK_slice = cute.composition(
                                sK_stage,
                                cute.make_layout((self.n_block_size, self.k_block_size)),
                            )
                            k_offset = const_expr(_kc * self.k_block_size)
                            self._load_k_rows(
                                mK,
                                sK_slice,
                                mTopkIdx,
                                mTopkLength,
                                async_copy_atom,
                                async_thr_copy,
                                n_block,
                                q_token_idx,
                                batch_idx,
                                seqlen_k,
                                topK_cur,
                                idx_in_group,
                                group_idx_local,
                                k_offset=k_offset,
                            )
                            cute.arch.cp_async_commit_group()
                            cute.arch.cp_async_wait_group(0)
                            cute.arch.fence_view_async_shared()
                            pipeline_K.producer_commit(K_load_state)
                            K_load_state.advance()
                        n_block = n_block - 1

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()

    # =========================================================================
    # Sparse K loading helpers (16 groups x 8 threads per row)
    # =========================================================================
    @cute.jit
    def _load_k_rows(
        self,
        mK: cute.Tensor,
        sK_slice: cute.Tensor,
        mTopkIdx: cute.Tensor,
        mTopkLength: cute.Tensor | None,
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        n_block: Int32,
        q_token_idx: Int32,
        batch_idx: Int32,
        seqlen_k: Int32,
        topK_cur: Int32,
        idx_in_group: Int32,
        group_idx_local: Int32,
        k_offset: int = 0,
    ):
        """Load all K rows for one pipeline stage (one n_block × one k_chunk).

        When ``self.topk_indices_global`` (default), ``mTopkIdx`` carries the
        public global KV ids (``b * seqlen_k + local``); we decode back to
        local-per-batch before indexing into mK ``(seqlen_k, head_dim, bs)``.
        With the flag False, ids are already local (legacy callers — e.g.
        ``indexer_bwd``'s grad path that still expects local).
        """
        NUM_GROUPS = const_expr(self.WARPGROUP_SIZE // 8)
        ROWS_PER_GROUP = const_expr(self.n_block_size // NUM_GROUPS)
        batch_offset = batch_idx * seqlen_k if const_expr(self.topk_indices_global) else Int32(0)
        for r in cutlass.range_constexpr(ROWS_PER_GROUP):
            row = r * NUM_GROUPS + group_idx_local
            topk_pos = n_block * self.n_block_size + row
            if const_expr(self.have_topk_length and self.score_type == "attention"):
                if topk_pos < topK_cur:
                    topk_raw = Int32(mTopkIdx[q_token_idx, topk_pos, batch_idx])
                    topk_local = topk_raw - batch_offset
                    self._copy_row(
                        mK,
                        sK_slice,
                        row,
                        idx_in_group,
                        copy_atom,
                        thr_copy,
                        topk_local,
                        batch_idx,
                        k_offset=k_offset,
                    )
            else:
                topk_local = Int32(-1)
                if topk_pos < topK_cur:
                    topk_raw = Int32(mTopkIdx[q_token_idx, topk_pos, batch_idx])
                    # Invalid (-1) stays negative after the offset subtraction
                    # and is rejected by the bounds check below.
                    topk_local = topk_raw - batch_offset
                if topk_local >= 0 and topk_local < seqlen_k:
                    self._copy_row(
                        mK,
                        sK_slice,
                        row,
                        idx_in_group,
                        copy_atom,
                        thr_copy,
                        topk_local,
                        batch_idx,
                        k_offset=k_offset,
                    )

    @cute.jit
    def _copy_row(
        self,
        mK: cute.Tensor,
        sK_slice: cute.Tensor,
        row: Int32,
        idx_in_group: Int32,
        copy_atom: cute.CopyAtom,
        thr_copy: cute.TiledCopy,
        topk_idx: Int32,
        batch_idx: Int32,
        k_offset: int = 0,
    ):
        """Copy one (partial) K row from gmem to smem via cp.async (8 threads per row).

        When k_block_size < head_dim_padded, k_offset selects which k_block_size
        slice of the row to copy. sK_slice has shape (n_block_size, k_block_size).
        """
        gK_row_raw = mK[topk_idx, None, batch_idx]
        gK_row_offset = gK_row_raw.iterator + k_offset
        gK_row = cute.make_tensor(
            cute.make_ptr(
                self.k_dtype,
                gK_row_offset.llvm_ptr,
                cute.AddressSpace.gmem,
                assumed_align=16,
            ),
            cute.make_layout((self.k_block_size,)),
        )
        gK_chunks = cute.flat_divide(gK_row, (8,))
        sK_row = sK_slice[row, None]
        sK_chunks = cute.flat_divide(sK_row, (8,))
        for tile in cutlass.range_constexpr(self.k_block_size // 64):
            chunk_idx = tile * 8 + idx_in_group
            g_chunk = gK_chunks[None, chunk_idx]
            s_chunk = sK_chunks[None, chunk_idx]
            tSg = thr_copy.partition_S(g_chunk)
            tSs = thr_copy.partition_D(s_chunk)
            cute.copy(copy_atom, tSg, tSs)

    @cute.jit
    def _zero_row(self, sK_slice: cute.Tensor, row: Int32, idx_in_group: Int32):
        """Zero-fill one K row in smem, cooperative across 8 threads in a group."""
        sK_row = sK_slice[row, None]
        sK_chunks = cute.flat_divide(sK_row, (8,))
        for tile in cutlass.range_constexpr(self.k_block_size // 64):
            chunk_idx = tile * 8 + idx_in_group
            sK_chunks[None, chunk_idx].fill(0)

    @cute.jit
    def _phase_for_slot(self, phase_bits, slot):
        return (phase_bits >> slot) & Int32(1)

    @cute.jit
    def _toggle_phase_for_slot(self, phase_bits, slot):
        return phase_bits ^ (Int32(1) << slot)

    # =========================================================================
    # Cross-warp reduce helpers (epilogue warpgroup)
    # =========================================================================
    @cute.jit
    def _intra_inter_warp_reduce_max(
        self,
        sScoreAll,
        reduce_sync_mbar_ptr,
        reduce_sync_phase,
        warp_id_in_wg,
        local_value,
    ):
        """Reduce local max across 4 warps via smem scratch + mbarrier sync."""
        warp_max = cute.arch.warp_redux_sync(local_value, "fmax")
        with cute.arch.elect_one():
            sScoreAll[warp_id_in_wg] = warp_max
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        global_max = sScoreAll[0]
        for wi in cutlass.range_constexpr(self.num_warps_in_epi_wg - 1):
            v = sScoreAll[wi + 1]
            global_max = v if v > global_max else global_max

        # The next reduction reuses these four words. Join readers before
        # any warp can overwrite its slot for the next max/sum or query tile.
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        return global_max, reduce_sync_phase

    @cute.jit
    def _intra_inter_warp_reduce_sum(
        self,
        sScoreAll,
        reduce_sync_mbar_ptr,
        reduce_sync_phase,
        warp_id_in_wg,
        local_value,
    ):
        """Reduce local sum across 4 warps via smem scratch + mbarrier sync."""
        warp_sum = cute.arch.warp_reduction_sum(local_value)
        with cute.arch.elect_one():
            sScoreAll[warp_id_in_wg] = warp_sum
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        global_sum = sScoreAll[0]
        for wi in cutlass.range_constexpr(self.num_warps_in_epi_wg - 1):
            global_sum = global_sum + sScoreAll[wi + 1]

        # The next reduction reuses these four words. Join readers before
        # any warp can overwrite its slot for the next max/sum or query tile.
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        return global_sum, reduce_sync_phase

    @cute.jit
    def _inter_warp_sync_sum(
        self,
        sScoreAll,
        reduce_sync_mbar_ptr,
        reduce_sync_phase,
        warp_id_in_wg,
        warp_sum,
    ):
        """Cross-warp sum sync with pre-computed warp-level sum (skip warp_reduction_sum)."""
        with cute.arch.elect_one():
            sScoreAll[warp_id_in_wg] = warp_sum
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        global_sum = sScoreAll[0]
        for wi in cutlass.range_constexpr(self.num_warps_in_epi_wg - 1):
            global_sum = global_sum + sScoreAll[wi + 1]

        # The next reduction reuses these four words. Join readers before
        # any warp can overwrite its slot for the next max/sum or query tile.
        cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
        cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_sync_phase)
        reduce_sync_phase = reduce_sync_phase ^ 1

        return global_sum, reduce_sync_phase

    # =========================================================================
    # Epilogue: indexer mode — ReLU(QK) * W, head reduce, softmax
    # =========================================================================
    @cute.jit
    def _epilogue_indexer(
        self,
        tiled_mma_qk,
        tStS_ref,
        mPerHead,
        sW,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        sTopkIdx,
        mTopkIdx,
        mOut,
        mTopkLength,
        m_block,
        batch_idx,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        topk_idx_offset=None,
    ):
        """Epilogue: TMEM->RF, ReLU, *W, head reduce -> scores -> softmax -> predict.

        1. n_block loop: ReLU(QK)*W head reduce, accumulate scores in rmem
        2. Post-loop: mask invalid positions, compute softmax, write predict to gmem

        Non-compact: W loaded directly from GMEM into registers (bypass SMEM
        to avoid LDS bandwidth contention with K scatter-gather loading).
        Compact: W loaded from SMEM (original path, preserves compact perf).
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)

        # Coordinate tensor: MMA partition -> remap to TMEM copy destination layout
        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

        tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        sW_off = Int32(0) if per_head_offset is None else per_head_offset
        sTopkIdx_off = Int32(0) if topk_idx_offset is None else topk_idx_offset
        qhpkv = self.qhead_per_kvhead

        W_ILP = 4
        rW_all = cute.make_rmem_tensor((self.m_block_size,), self.per_head_dtype)
        rW_all_f32 = cute.recast_tensor(rW_all, Float32)

        # Load W from GMEM directly (bypass SMEM, reduce LDS contention
        # with K scatter-gather cp.async writes to SMEM).
        q_token_idx_w = m_block
        gW_raw = mPerHead[q_token_idx_w, None, batch_idx]
        gW_f32_ptr = cute.make_ptr(
            Float32,
            gW_raw.iterator.llvm_ptr,
            cute.AddressSpace.gmem,
            assumed_align=16,
        )
        W_src_f32 = cute.make_tensor(
            gW_f32_ptr,
            cute.make_layout((self.m_block_size // 2,)),
        )

        kv_offset = tScS[0][0]
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        # ---- Phase 1: n_block loop — head reduce, hold scores in rmem ----
        # Process all n_blocks unconditionally (no block-level skip).
        # MMA produces all blocks with static bounds; invalid scores masked in Phase 2.
        rScores = [Float32(0.0) for _ in range(self.num_n_blocks)]

        q_token_idx = m_block

        if const_expr(self.have_topk_length):
            topK_epi = mTopkLength[q_token_idx, batch_idx]

        for _ri in cutlass.range_constexpr(self.num_n_blocks):
            n_blk = self.num_n_blocks - 1 - _ri
            _slot = const_expr(n_blk % self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, _slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * _slot, s_full_phase)
            if const_expr(_ri == 0):
                cute.autovec_copy(W_src_f32, rW_all_f32)
            tmem_ptr_cur = cute.make_ptr(
                Float32,
                _slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * _slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                _slot,
            )
            local_sum = (Float32(0.0), Float32(0.0))
            for ho in cutlass.range_constexpr(qhpkv // 2 // W_ILP):
                for ci in cutlass.range_constexpr(W_ILP):
                    idx0 = (ho * W_ILP + ci) * 2
                    idx1 = idx0 + 1
                    w_pair = (Float32(rW_all[idx0]), Float32(rW_all[idx1]))

                    val0 = tSrS[idx0]
                    val0 = val0 if val0 > Float32(0.0) else Float32(0.0)
                    val1 = tSrS[idx1]
                    val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                    prod = mul_packed_f32x2((val0, val1), w_pair)
                    local_sum = add_packed_f32x2(local_sum, prod)

            rScores[n_blk] = (local_sum[0] + local_sum[1]) * Float32(softmax_scale)

        # ---- Phase 2: softmax (scores in rmem, only 4 FP32 smem for cross-warp) ----
        # Mask invalid positions + local max
        local_max = -Float32.inf
        for ei in cutlass.range_constexpr(self.num_n_blocks):
            pos = kv_offset + ei * self.n_block_size
            if const_expr(self.have_topk_length):
                if pos >= topK_epi:
                    rScores[ei] = -Float32.inf
            else:
                if const_expr(self.topk_in_smem):
                    topk_idx = Int32(sTopkIdx[sTopkIdx_off + pos])
                else:
                    topk_idx = Int32(mTopkIdx[q_token_idx, pos, batch_idx])
                if topk_idx < 0:
                    rScores[ei] = -Float32.inf
            if rScores[ei] > local_max:
                local_max = rScores[ei]

        global_max, reduce_phase = self._intra_inter_warp_reduce_max(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            local_max,
        )

        if global_max > Float32(-1e30):
            # Compute exp(score - max) and local sum
            log2_e = Float32(math.log2(math.e))
            local_exp_sum = Float32(0.0)
            for ei in cutlass.range_constexpr(self.num_n_blocks):
                exp_val = cute.math.exp2((rScores[ei] - global_max) * log2_e)
                rScores[ei] = exp_val
                local_exp_sum = local_exp_sum + exp_val

            sScoreAll_sum = cute.make_tensor(
                sScoreAll.iterator + self.num_warps_in_epi_wg,
                cute.make_layout((self.num_warps_in_epi_wg,), stride=(1,)),
            )
            global_sum, reduce_phase = self._intra_inter_warp_reduce_sum(
                sScoreAll_sum,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                local_exp_sum,
            )

            inv_sum = Float32(1.0) / global_sum
            for ei in cutlass.range_constexpr(self.num_n_blocks):
                pos = kv_offset + ei * self.n_block_size
                mOut[q_token_idx, pos, batch_idx] = rScores[ei] * inv_sum
        else:
            for ei in cutlass.range_constexpr(self.num_n_blocks):
                pos = kv_offset + ei * self.n_block_size
                mOut[q_token_idx, pos, batch_idx] = Float32(0.0)
            # 1 dummy sync to match the if-branch's sum reduce sync
            cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
            cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
            reduce_phase = reduce_phase ^ 1

        return s_full_phase_bits, reduce_phase

    # =========================================================================
    # Epilogue: attention mode, n_block_size>=128 — Ld32x32bOp path
    # =========================================================================
    @cute.jit
    def _epilogue_attention_n128(
        self,
        tiled_mma_qk,
        tStS_ref,
        sLSE,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        sTopkIdx,
        mTopkIdx,
        mOut,
        mTopkLength,
        m_block,
        batch_idx,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        topk_idx_offset=None,
    ):
        """Epilogue (attention, n_block_size>=128): TMEM->RF, exp2(S*scale - LSE), head reduce -> L1-norm.

        With n_block_size>=128 (MMA M>=128), TMEM output occupies contiguous
        lanes, so we use Ld32x32bOp. Each thread owns all m_block_size heads;
        no shfl_xor or is_active gating needed.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(8)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)

        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

        tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        sLSE_off = Int32(0) if per_head_offset is None else per_head_offset
        sTopkIdx_off = Int32(0) if topk_idx_offset is None else topk_idx_offset
        qhpkv = self.qhead_per_kvhead

        LSE_ILP = 4
        sLSE_f32_ptr = cute.make_ptr(
            Float32,
            (sLSE.iterator + sLSE_off).llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        sLSE_1d = cute.make_tensor(
            sLSE_f32_ptr,
            cute.make_layout((self.m_block_size,)),
        )
        rLSE_all = cute.make_rmem_tensor((self.m_block_size,), Float32)

        kv_offset = tScS[0][0]
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_idx = m_block

        # ---- Phase 1: n_block loop — softmax recovery + head reduce ----
        rScores = [Float32(0.0) for _ in range(self.num_n_blocks)]
        warp_sum_acc = Float32(0.0)

        if const_expr(self.have_topk_length):
            topK_epi = mTopkLength[q_token_idx, batch_idx]
            n_block_max_epi = (topK_epi + self.n_block_size - 1) // self.n_block_size

        for _ri in cutlass.range_constexpr(self.num_n_blocks):
            n_blk = self.num_n_blocks - 1 - _ri
            _slot = const_expr(n_blk % self.num_tmem_slots)

            s_full_phase = self._phase_for_slot(s_full_phase_bits, _slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * _slot, s_full_phase)
            if const_expr(_ri == 0):
                cute.autovec_copy(sLSE_1d, rLSE_all)
            # Keep the TMEM load unconditional: putting tcgen05 TMEM load ops
            # under the top-k-length runtime guard trips CUTLASS DSL compile.
            # Invalid blocks are ignored by should_accumulate_score below.
            tmem_ptr_cur = cute.make_ptr(
                Float32,
                _slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * _slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                _slot,
            )
            should_accumulate_score = const_expr(True)
            if const_expr(self.have_topk_length):
                should_accumulate_score = n_blk < n_block_max_epi
            if should_accumulate_score:
                local_sum = (Float32(0.0), Float32(0.0))
                for ho in cutlass.range_constexpr(qhpkv // 2 // LSE_ILP):
                    for ci in cutlass.range_constexpr(LSE_ILP):
                        idx0 = (ho * LSE_ILP + ci) * 2
                        idx1 = idx0 + 1
                        lse_pair = (rLSE_all[idx0], rLSE_all[idx1])

                        val0 = tSrS[idx0]
                        val1 = tSrS[idx1]

                        val0, val1 = fma_packed_f32x2(
                            (val0, val1),
                            (scale_log2_e, scale_log2_e),
                            lse_pair,
                        )
                        val0 = cute.math.exp2(val0)
                        val1 = cute.math.exp2(val1)

                        local_sum = add_packed_f32x2(local_sum, (val0, val1))

                rScores[n_blk] = local_sum[0] + local_sum[1]

                pos = kv_offset + n_blk * self.n_block_size
                if const_expr(self.have_topk_length):
                    if pos >= topK_epi:
                        rScores[n_blk] = Float32(0.0)
                else:
                    if const_expr(self.topk_in_smem):
                        topk_idx = Int32(sTopkIdx[sTopkIdx_off + pos])
                    else:
                        topk_idx = Int32(mTopkIdx[q_token_idx, pos, batch_idx])
                    if topk_idx < 0:
                        rScores[n_blk] = Float32(0.0)
                warp_sum_acc = warp_sum_acc + cute.arch.warp_reduction_sum(rScores[n_blk])

        # ---- Phase 2: L1-norm (scores already non-negative from exp) ----
        global_sum, reduce_phase = self._inter_warp_sync_sum(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            warp_sum_acc,
        )

        if global_sum > Float32(1e-10):
            inv_sum = Float32(1.0) / global_sum
            for ei in cutlass.range_constexpr(self.num_n_blocks):
                pos = kv_offset + ei * self.n_block_size
                mOut[q_token_idx, pos, batch_idx] = rScores[ei] * inv_sum
        else:
            for ei in cutlass.range_constexpr(self.num_n_blocks):
                pos = kv_offset + ei * self.n_block_size
                mOut[q_token_idx, pos, batch_idx] = Float32(0.0)

        return s_full_phase_bits, reduce_phase

    # =========================================================================
    # Epilogue: attention mode, n_block_size<128 — Ld16x64bOp path
    # =========================================================================
    @cute.jit
    def _epilogue_attention(
        self,
        tiled_mma_qk,
        tStS_ref,
        sLSE,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        sTopkIdx,
        mTopkIdx,
        mOut,
        mTopkLength,
        m_block,
        batch_idx,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        topk_idx_offset=None,
    ):
        """Epilogue (attention mode): TMEM->RF, exp2(S*scale - LSE), head reduce -> L1-norm.

        With n_block_size=64 (MMA M=64), TMEM output occupies non-contiguous
        lanes (0-15, 32-47, 64-79, 96-111), so we use Ld16x64bOp instead of
        Ld32x32bOp. Per PTX Figure 184, each lane is shared by Thread i and
        Thread i^2 (e.g. T0+T2 → lane 0, T1+T3 → lane 8). Each thread holds
        m_block_size/2 heads; a single shfl_xor(2) combines the partial sums.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld16x64bOp(tcgen05.copy.Repetition(self.m_block_size // 2)),
            Float32,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)

        thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

        tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
        tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

        sLSE_off = Int32(0) if per_head_offset is None else per_head_offset
        sTopkIdx_off = Int32(0) if topk_idx_offset is None else topk_idx_offset

        # Number of heads each thread processes (half of m_block_size)
        num_heads_per_thr = self.m_block_size // 2
        LSE_ILP = 4

        sLSE_f32_ptr = cute.make_ptr(
            Float32,
            (sLSE.iterator + sLSE_off).llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )

        # Broadcast SMEM LSE view: shape (n_block_size, m_block_size) with
        # stride=(0,1) so every M-row sees the same per-head LSE values.
        # Partitioning with the same MMA+TMEM_copy chain as tSrS ensures
        # tSrLSE[i] pairs with tSrS[i] regardless of fragment layout.
        sLSE_2d = cute.make_tensor(
            sLSE_f32_ptr,
            cute.make_layout((self.n_block_size, self.m_block_size), stride=(0, 1)),
        )
        tSsLSE = thr_tmem_load.partition_D(thr_mma.partition_C(sLSE_2d))
        tSrLSE = cute.make_rmem_tensor(tSsLSE.shape, Float32)

        kv_offset = tScS[0][0]
        warp_id_in_wg = tidx_wg // self.WARP_SIZE
        # Thread i and Thread i^2 share a lane; only one should contribute
        # to warp-level sums / write output.  (tidx_wg % 4) < 2 selects
        # exactly one thread per lane (T0,T1,T4,T5,...  not T2,T3,T6,T7,...).
        is_active = (tidx_wg % 4) < 2

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_idx = m_block

        # ---- Phase 1: n_block loop — softmax recovery + head reduce, hold scores in rmem ----
        rScores = [Float32(0.0) for _ in range(self.num_n_blocks)]
        warp_sum_acc = Float32(0.0)

        if const_expr(self.have_topk_length):
            topK_epi = mTopkLength[q_token_idx, batch_idx]
            n_block_max_epi = (topK_epi + self.n_block_size - 1) // self.n_block_size

        for _ri in cutlass.range_constexpr(self.num_n_blocks):
            n_blk = self.num_n_blocks - 1 - _ri
            _slot = const_expr(n_blk % self.num_tmem_slots)

            s_full_phase = self._phase_for_slot(s_full_phase_bits, _slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * _slot, s_full_phase)
            if const_expr(_ri == 0):
                cute.autovec_copy(tSsLSE, tSrLSE)
            # Keep the TMEM load unconditional: putting tcgen05 TMEM load ops
            # under the top-k-length runtime guard trips CUTLASS DSL compile.
            # Invalid blocks are ignored by should_accumulate_score below.
            tmem_ptr_cur = cute.make_ptr(
                Float32,
                _slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()
            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * _slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                _slot,
            )
            should_accumulate_score = const_expr(True)
            if const_expr(self.have_topk_length):
                should_accumulate_score = n_blk < n_block_max_epi
            if should_accumulate_score:
                local_sum = (Float32(0.0), Float32(0.0))
                for ho in cutlass.range_constexpr(num_heads_per_thr // 2 // LSE_ILP):
                    for ci in cutlass.range_constexpr(LSE_ILP):
                        idx0 = (ho * LSE_ILP + ci) * 2
                        idx1 = idx0 + 1
                        lse_pair = (tSrLSE[idx0], tSrLSE[idx1])

                        val0 = tSrS[idx0]
                        val1 = tSrS[idx1]

                        val0, val1 = fma_packed_f32x2(
                            (val0, val1),
                            (scale_log2_e, scale_log2_e),
                            lse_pair,
                        )
                        val0 = cute.math.exp2(val0)
                        val1 = cute.math.exp2(val1)

                        local_sum = add_packed_f32x2(local_sum, (val0, val1))

                partial_score = local_sum[0] + local_sum[1]

                # Mask invalid KV positions at partial level
                pos = kv_offset + n_blk * self.n_block_size
                if const_expr(self.have_topk_length):
                    if pos >= topK_epi:
                        partial_score = Float32(0.0)
                else:
                    if const_expr(self.topk_in_smem):
                        topk_idx = Int32(sTopkIdx[sTopkIdx_off + pos])
                    else:
                        topk_idx = Int32(mTopkIdx[q_token_idx, pos, batch_idx])
                    if topk_idx < 0:
                        partial_score = Float32(0.0)

                warp_sum_acc = warp_sum_acc + cute.arch.warp_reduction_sum(partial_score)

                partner_partial = cute.arch.shuffle_sync_bfly(
                    partial_score,
                    offset=2,
                    mask=-1,
                    mask_and_clamp=31,
                )
                rScores[n_blk] = partial_score + partner_partial

        # ---- Phase 2: L1-norm (scores already non-negative from exp) ----
        global_sum, reduce_phase = self._inter_warp_sync_sum(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            warp_sum_acc,
        )

        if is_active:
            if global_sum > Float32(1e-10):
                inv_sum = Float32(1.0) / global_sum
                for ei in cutlass.range_constexpr(self.num_n_blocks):
                    pos = kv_offset + ei * self.n_block_size
                    mOut[q_token_idx, pos, batch_idx] = rScores[ei] * inv_sum
            else:
                for ei in cutlass.range_constexpr(self.num_n_blocks):
                    pos = kv_offset + ei * self.n_block_size
                    mOut[q_token_idx, pos, batch_idx] = Float32(0.0)

        return s_full_phase_bits, reduce_phase
