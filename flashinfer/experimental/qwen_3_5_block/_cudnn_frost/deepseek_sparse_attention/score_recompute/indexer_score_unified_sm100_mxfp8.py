# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL unified MXFP8 dense indexer-score kernel.

This module owns the MXFP8 blockscaled QK producer and unified indexer epilogue
used by both forward score and backward dense indexer score.

High-level flow:
  1. reinterpret FP8 Q/K and packed E8M0 scales into CuTe/TMA layouts,
  2. load Q/K and SFB/SFA scales into SMEM,
  3. copy scales into TMEM SFB/SFA fields and issue blockscaled QK UMMA,
  4. reduce heads in FP32 to write scores and optional LSE.

Head groups: the tile is always 128 packed (token, head) rows -- the blockscaled
scale atom is 128 rows, so the Q scale packing (``packed_m = token * group + head``)
and the TMA / TMEM scale plumbing do not depend on the group size.  64 and 32 heads
keep their tuned epilogues (one, resp. two tokens per epilogue warpgroup); the small
groups 4 / 8 / 16 pack ``128 // group`` tokens per tile and the two epilogue
warpgroups split the tile's 128 columns in halves (``_epilogue_indexer_dense_group``).
A plain-e4m3 scorer (no block scales) is this kernel with all-ones E8M0 blobs.
"""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass.cute.nvgpu import cpasync
import cutlass.cute.nvgpu.tcgen05 as tcgen05
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    PipelineTmaUmma,
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
    make_blockscaled_trivial_tiled_mma as _make_blockscaled_trivial_tiled_mma,
    cluster_shape_to_tma_atom_A as _cluster_shape_to_tma_atom_A,
    cluster_shape_to_tma_atom_SFB as _cluster_shape_to_tma_atom_SFB,
)
from cutlass.utils import blockscaled_layout as _blockscaled_layout
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._cutlass_compat import LayoutEnum, SmemAllocator

import cuda.bindings.driver as cuda

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils import copy as copy_utils
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.sm100.gemm import (
    gemm_ptx_partial_mxfp8 as _gemm_ptx_partial_mxfp8,
)
from .indexer_score_unified_sm100 import IndexerScoreUnifiedSm100
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from .dense_score_recompute_sm100 import (
    add_packed_f32x2,
    fma_packed_f32x2,
)


class IndexerScoreUnifiedSm100Mxfp8(IndexerScoreUnifiedSm100):
    """Shared MXFP8 dense score kernel with unified indexer output semantics."""

    arch = 100
    # The tile stays 128 packed rows wide for every group (the scale atom), so the
    # smallest group (4 heads) packs 32 tokens; 64 / 32 heads pack 2 / 4 as before.
    max_q_tokens_per_tile = 32
    # Block-scaled format of Q/K, named in errors.  E4M3 Q/K run tcgen05
    # kind::mxf8f6f4, whose MMA reads K=32 values and one E8M0 scale per row.
    format_name = "MXFP8"
    mma_inst_k = 32

    def __init__(
        self,
        head_dim: int,
        qhead_per_kvhead: int = 64,
        m_block_size: int = 128,
        n_block_size: int = 128,
        kv_stage: int = 4,
        k_block_size: int | None = None,
        ratio: int = 1,
        is_varlen: bool = False,
        sf_vec_size: int = 32,
        compute_lse: bool = False,
        is_compressed_logits: bool = False,
    ):
        # This specialization is deliberately narrow.  The current production
        # MXFP8 indexer path is D128/sf_vec_size=32 with 128x128 tiles; keeping
        # the compile keys constrained avoids dragging attention/D512 branches
        # into the generated code.
        if head_dim != 128:
            raise ValueError(f"SM100 unified indexer {self.format_name} currently requires head_dim=128")
        if sf_vec_size != 32:
            raise ValueError(f"SM100 dense score {self.format_name} currently requires sf_vec_size=32")
        if m_block_size != 128 or n_block_size != 128:
            raise ValueError(f"SM100 dense score {self.format_name} currently requires m_block_size=n_block_size=128")
        if qhead_per_kvhead not in (4, 8, 16, 32, 64):
            raise ValueError(f"SM100 unified indexer {self.format_name} requires qhead_per_kvhead in (4, 8, 16, 32, 64), got {qhead_per_kvhead}")
        k_block_size = 64 if k_block_size is None else k_block_size
        # k_block_size is an internal K-split knob (num_k_chunks = head_dim_padded //
        # k_block_size; the per-chunk scale phase + base-class SMEM derive from it), so
        # any multiple of sf_vec_size and of the MMA K that divides head_dim_padded is
        # valid (default 64).
        _hd_padded = ((head_dim + 15) // 16) * 16
        if k_block_size <= 0 or k_block_size % sf_vec_size != 0 or _hd_padded % k_block_size != 0:
            raise ValueError(
                f"SM100 unified indexer {self.format_name} requires k_block_size to be a "
                f"positive multiple of sf_vec_size ({sf_vec_size}) that divides "
                f"head_dim_padded ({_hd_padded}); got {k_block_size}"
            )
        if k_block_size % self.mma_inst_k != 0:
            raise ValueError(
                f"SM100 unified indexer {self.format_name} requires k_block_size to be a multiple of the MMA K ({self.mma_inst_k}); got {k_block_size}"
            )

        # Reduce prefetch depth only for the dense H64 ratio-four LSE schedule.
        # Only override the interface's 24-stage configuration; compressed-logit
        # paths and other stage counts retain their existing configuration.
        interface_kv_stages = 24
        dense_lse_kv_stages = 8
        if kv_stage == interface_kv_stages and qhead_per_kvhead == 64 and ratio == 4 and compute_lse and not is_compressed_logits:
            kv_stage = dense_lse_kv_stages

        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=kv_stage,
            k_block_size=k_block_size,
            ratio=ratio,
            is_varlen=is_varlen,
            compute_lse=compute_lse,
            is_compressed_logits=is_compressed_logits,
        )
        # Base class initializes the BF16 dense-indexer tiler and the unified
        # epilogue state.  The fields below override the parts that differ for
        # blockscaled MXFP8 QK: scale dtype/layout, TMEM allocation, register
        # split, and the k64 scale pipeline.
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU
        # Load warp keeps fewer registers so the epilogue warpgroups have enough
        # register budget for head reduction and optional online LSE.
        self.num_regs_load = 64
        self.num_regs_epilogue = 64
        # k_block_size=64 splits D128 into two data chunks.  K data and K scale
        # use separate TMA barriers so the load warp can prefetch the next scale
        # tile while the MMA warp consumes K data stages.
        # MXFP8 unified currently targets D128 only, so one full-head scale tile
        # covers all four E8M0 scale groups.  Q/SFB occupies a single TMEM slot.
        self.scale_tile_k = self.head_dim_padded
        self.num_q_scale_stages = self.num_k_chunks
        self.k_scale_stage = self.kv_stage
        ratio_value = int(ratio)
        # Ratio>1 v4 rows use a hand-written partial MXFP8 GEMM wrapper.  It
        # preserves the same blockscaled operands but reduces overhead in the
        # short-K compressed path.
        self.use_direct_mxfp8_gemm = ratio_value > 1
        self.num_qsfb_tmem_slots = 1
        self.sScoreAll_single_size = self.num_warps_in_epi_wg * 2
        # Warpgroups:
        #   WG0: load/MMA/scheduler warps
        #   WG1: QH64 stage 0 / QH32 stages 0-1 epilogue
        #   WG2: QH64 stage 1 / QH32 stages 2-3 epilogue
        self.epilogue_wg0_warp_ids = (4, 5, 6, 7)
        self.epilogue_wg1_warp_ids = (8, 9, 10, 11)
        self.num_warps = 12
        self.threads_per_cta = self.WARP_SIZE * self.num_warps
        self.s_empty_arrive_count = 2 * self.WARPGROUP_SIZE
        self.num_epilogue_warps_for_clc = 8
        self.tmem_dealloc_arrive_count = 2
        self.reduce_sync_mbar_size = 4
        self.sScoreAll_size = self.sScoreAll_single_size * 2

        # Three slots let ratio-one H64 forward overlap MMA with head reduction.
        # Other ratio-one paths retain two; ratio>1 keeps three for direct GEMM.
        self.num_tmem_slots = 2 if ratio == 1 and (qhead_per_kvhead == 32 or compute_lse) else 3
        self.tmem_s_stride = self.m_block_size
        self.tmem_sfa_cols = (self.n_block_size // 32) * 4
        self.tmem_sfb_cols = (self.m_block_size // 32) * 4
        self.tmem_k_sfa_base_offset = 384
        self.tmem_q_sfb_offset = 432
        self.tmem_total = self.tmem_q_sfb_offset + self.num_qsfb_tmem_slots * self.tmem_sfb_cols
        self.tmem_alloc_cols = 512
        self.KScale_mbar_size = 2 * self.k_scale_stage
        self.S_mbar_size = 2 * self.num_tmem_slots

    @staticmethod
    def _qk_tensor(t: cute.Tensor) -> cute.Tensor:
        """Q or K as this kernel reads it, from the tensor it is launched with (the same tensor here)."""
        return t

    def _make_data_tiled_mma(self, blockscaled_tiled_mma_qk, cta_group):
        """Tiled MMA that lays out and partitions the Q/K data tiles: the unscaled E4M3 MMA of the QK tile."""
        return _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerHead: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mOut: cute.Tensor,
        mDenom: cute.Tensor | None,
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
        mCandBatchOffsets: cute.Tensor | None = None,
    ):
        # mDenom is None when compute_lse is off: the slot is compiled out (R3).
        # Runtime tensor metadata is needed to build CuTe layouts and compile
        # the matching kernel.  The element dtypes become compile-time constants
        # inside cute.compile.
        mQ = self._qk_tensor(mQ)
        mK = self._qk_tensor(mK)
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.sf_dtype = mQScale.element_type
        is_varlen = mCuSeqlensQ is not None

        # The API accepts either BSHD or packed THD.  The compile key records
        # is_varlen, and these asserts make accidental mismatches fail during
        # compilation instead of silently interpreting strides incorrectly.
        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQ is not None and mCuSeqlensK is not None
            assert mCuSeqlensQScalePadded is not None
            assert mCuSeqlensKScalePadded is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQ is None and mCuSeqlensK is None
            assert mCuSeqlensQScalePadded is None
            assert mCuSeqlensKScalePadded is None

        # Normalize BSHD and THD tensors into the K-major/Q-major shapes expected
        # by the SM100 TMA helpers.  THD keeps batch in cu_seqlens; BSHD exposes
        # the batch as the final tensor mode.
        Q_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        K_layout_transpose = [0, 2, 1] if const_expr(is_varlen) else [1, 3, 2, 0]
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose))
        mK = cute.make_tensor(mK.iterator, cute.select(mK.layout, mode=K_layout_transpose))

        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0])
        seqlen_q_packed = seqlen_q_static * self.qhead_per_kvhead

        # The packed scale tensors expose their caller-defined, atom-aligned MN
        # address extent directly in the second mode.
        scale_l_size = cute.size(mQScale.shape[0])
        q_scale_layout = _blockscaled_layout.tile_atom_to_shape_SF(
            (cute.size(mQScale.shape[1]), self.head_dim_padded, scale_l_size),
            self.sf_vec_size,
        )
        k_scale_layout = _blockscaled_layout.tile_atom_to_shape_SF(
            (cute.size(mKScale.shape[1]), self.head_dim_padded, scale_l_size),
            self.sf_vec_size,
        )
        mQScale = cute.make_tensor(mQScale.iterator, q_scale_layout)
        mKScale = cute.make_tensor(mKScale.iterator, k_scale_layout)

        # Fold qhead_per_kvhead into the M dimension so one m tile covers two
        # QH64 tokens or four QH32 tokens. The epilogue reverses this packed
        # index when writing outputs.
        shape_Q_packed = (
            (self.qhead_per_kvhead, mQ.shape[0]),
            mQ.shape[1],
            1,
            *mQ.shape[3:],
        )
        stride_Q_packed = (
            (mQ.stride[2], mQ.stride[0]),
            mQ.stride[1],
            mQ.stride[2] * self.qhead_per_kvhead,
            *mQ.stride[3:],
        )
        mQ = cute.make_tensor(mQ.iterator, cute.make_layout(shape_Q_packed, stride=stride_Q_packed))

        cta_group = tcgen05.CtaGroup.ONE
        self.q_major_mode = LayoutEnum.from_tensor(mQ).mma_major_mode()
        self.k_major_mode = LayoutEnum.from_tensor(mK).mma_major_mode()

        # blockscaled_tiled_mma_qk is the UMMA op with SFA/SFB fields.  tiled_mma_qk
        # (``_make_data_tiled_mma``) lays out and partitions the data tensors; it may
        # be the block-scaled op itself.
        blockscaled_tiled_mma_qk = _make_blockscaled_trivial_tiled_mma(
            self.k_dtype,
            self.k_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            cta_group,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_qk = self._make_data_tiled_mma(blockscaled_tiled_mma_qk, cta_group)

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

        # Data SMEM follows the data tiled MMA's layouts; scale SMEM follows CUTLASS
        # blockscaled SFA/SFB layouts so tcgen05 can consume E8M0 factors from
        # TMEM fields rather than doing explicit dequant math in registers.
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
        scale_tile_k = self.scale_tile_k
        scale_mma_tiler_qk = (self.n_block_size, self.m_block_size, scale_tile_k)
        sKScale_layout = _blockscaled_layout.make_smem_layout_sfa(
            blockscaled_tiled_mma_qk,
            scale_mma_tiler_qk,
            self.sf_vec_size,
            self.k_scale_stage,
        )
        sQScale_layout = _blockscaled_layout.make_smem_layout_sfb(
            blockscaled_tiled_mma_qk,
            scale_mma_tiler_qk,
            self.sf_vec_size,
            self.num_q_scale_stages,
        )
        self.sQ_layout = sQ_layout
        self.sK_layout = sK_layout
        self.sQScale_layout = sQScale_layout
        self.sKScale_layout = sKScale_layout

        # Data TMA atoms load the Q/K tiles into SMEM.  Scale TMA atoms load
        # packed E8M0 factors as Int16 internally because CUTLASS blockscaled
        # layouts store two E8M0 bytes per logical compact element.
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_K, mK = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
        )

        k_scale_tma_op = _cluster_shape_to_tma_atom_A(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id)
        q_scale_tma_op = _cluster_shape_to_tma_atom_SFB(self.cluster_shape_mn, blockscaled_tiled_mma_qk.thr_id)
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        tma_atom_KScale, mKScale = cute.nvgpu.make_tiled_tma_atom_A(
            k_scale_tma_op,
            mKScale,
            k_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        tma_atom_QScale, mQScale = cute.nvgpu.make_tiled_tma_atom_B(
            q_scale_tma_op,
            mQScale,
            q_scale_smem_layout,
            scale_mma_tiler_qk,
            blockscaled_tiled_mma_qk,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )

        q_data_tma_bytes = cute.size_in_bytes(self.q_dtype, cute.select(sQ_layout, mode=[0, 1, 2]))
        k_data_tma_bytes = cute.size_in_bytes(self.k_dtype, cute.select(sK_layout, mode=[0, 1, 2]))
        q_scale_tma_bytes = cute.size_in_bytes(self.sf_dtype, q_scale_smem_layout)
        k_scale_tma_bytes = cute.size_in_bytes(self.sf_dtype, k_scale_smem_layout)
        # KScale has its own pipeline and barrier accounting.  One full-D128
        # scale tile is reused by the two k64 data chunks for the same n block.
        self.tma_copy_bytes = {
            "Q": self.num_k_chunks * q_data_tma_bytes + q_scale_tma_bytes,
            "K": k_data_tma_bytes,
            "KScale": k_scale_tma_bytes,
        }

        PerHead_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=PerHead_transpose))

        if const_expr(not self.is_compressed_logits):
            Out_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
            mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_transpose))

        if const_expr(mDenom is not None):
            Denom_transpose = [0] if const_expr(is_varlen) else [1, 0]
            mDenom = cute.make_tensor(mDenom.iterator, cute.select(mDenom.layout, mode=Denom_transpose))
        else:
            assert not self.compute_lse, "mDenom is None but compute_lse=True writes the denominator"

        # Persistent scheduler grid: one logical work tile per m block and
        # batch.  The kernel itself walks all dense K blocks for that m tile.
        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
        # Launch signature keeps Q/K scales explicit so the same unified object
        # can serve forward score-only and backward score+LSE dispatches.
        self.kernel(
            mQ,
            mK,
            mPerHead,
            mQScale,
            mKScale,
            mOut,
            mDenom,
            softmax_scale,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_QScale,
            tma_atom_KScale,
            tiled_mma_qk,
            blockscaled_tiled_mma_qk,
            sQ_layout,
            sK_layout,
            sQScale_layout,
            sKScale_layout,
            tile_sched_params,
            max_seqlen_q,
            max_seqlen_k,
            mCuSeqlensQ,
            mCuSeqlensK,
            mCuSeqlensQScalePadded,
            mCuSeqlensKScalePadded,
            mQCausalOffsets,
            mCandBatchOffsets,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk,
            stream=stream,
        )

    @cute.jit
    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        # Scale factors are loaded by TMA into SMEM and then copied into TMEM
        # SFA/SFB fields with tcgen05 S2T copies.  filter_zeros keeps the copy
        # compact for the blockscaled layout's sparse logical strides.
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_compact_s2t_)
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    @cute.jit
    def _gemm_blockscaled_qk(
        self,
        blockscaled_tiled_mma_qk,
        tStS_stage,
        tSrK_stage,
        tSrQ_stage,
        sK_stage,
        sQ_stage,
        tCtKSFA,
        tCtQSFB,
        accumulate_first: cutlass.Constexpr[bool] = False,
        kphase_offset: cutlass.Constexpr[int] = 0,
    ):
        if const_expr(self.use_direct_mxfp8_gemm):
            # Ratio>1 score rows are short and mask-heavy.  The direct wrapper
            # emits the same blockscaled UMMA operation with less CuTe loop
            # overhead, while still wiring SFA/SFB TMEM descriptors.
            _gemm_ptx_partial_mxfp8(
                blockscaled_tiled_mma_qk.op,
                tStS_stage.iterator.toint(),
                tSrK_stage,
                tSrQ_stage,
                sK_stage,
                sQ_stage,
                tCtKSFA,
                tCtQSFB,
                zero_init=const_expr(not accumulate_first),
                kphase_offset=kphase_offset,
            )
            return

        # Generic blockscaled CuTe path.  kphase_offset, the index of this chunk's
        # first MMA along K, selects its E8M0 scale group.
        num_kphases = cute.size(tSrK_stage, mode=[2])
        for kphase_idx in cutlass.range(num_kphases, unroll_full=True):
            kphase_coord = (None, None, kphase_idx)
            sf_kphase_coord = (None, None, kphase_idx + kphase_offset)
            blockscaled_tiled_mma_qk.set(
                tcgen05.Field.ACCUMULATE,
                accumulate_first or (kphase_idx != 0),
            )
            blockscaled_tiled_mma_qk.set(tcgen05.Field.SFA, tCtKSFA[sf_kphase_coord].iterator)
            blockscaled_tiled_mma_qk.set(tcgen05.Field.SFB, tCtQSFB[sf_kphase_coord].iterator)
            cute.gemm(
                blockscaled_tiled_mma_qk,
                tStS_stage,
                tSrK_stage[kphase_coord],
                tSrQ_stage[kphase_coord],
                tStS_stage,
            )

    @cute.jit
    def _epilogue_indexer_dense_single_q(
        self,
        q_token_stage: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sW,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        mOut,
        mDenom: cute.Tensor | None,
        num_n_blocks_compute,
        seqlen_k,
        seqlen_q,
        max_seqlen_k,
        q_causal_offset,
        m_block,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        batch_idx=None,
        cand_batch_offsets=None,
    ):
        """Epilogue fast path for one packed q token.

        Two epilogue warpgroups call this helper once each for the two q tokens
        in a D128/H64 m tile. Splitting q rows avoids serializing their head
        reductions in one warpgroup.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sW_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

        # Ratio-1 rows have dense valid columns and benefit from shorter ILP.
        # Ratio>1 rows are sparse in K and use deeper ILP to keep the reduction
        # pipe fed while masking removes many columns.
        W_ILP = 4 if cutlass.const_expr(self.ratio == 1) else 8
        sW_1d = cute.make_tensor(
            sW.iterator + sW_off,
            cute.make_layout((self.m_block_size,)),
        )
        rW_all = cute.make_rmem_tensor((self.m_block_size,), Float32)

        warp_id_in_wg = tidx_wg // self.WARP_SIZE
        log2_e = Float32(math.log2(math.e))

        q_token_idx = m_block * q_tokens_per_tile + Int32(q_token_stage)
        col_limit = (q_causal_offset + q_token_idx + Int32(1)) // ratio
        if cutlass.const_expr(self.is_compressed_logits):
            q_global_start = Int64(q_causal_offset)
            if cutlass.const_expr(cand_batch_offsets is not None):
                cand_batch_base = Int64(cand_batch_offsets[batch_idx])
            else:
                cand_batch_base = Int64(batch_idx) * self._row_offset_i64(
                    Int64(seqlen_q),
                    q_global_start,
                )
            cand_row_offset = self._row_offset_i64(Int64(q_token_idx), q_global_start)
        local_max = -Float32.inf
        local_sum_exp = Float32(0.0)
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - 1
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)
            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))
            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sW_1d, rW_all)
                first_block = Int32(0)

            tmem_ptr_cur = cute.make_ptr(
                Float32,
                slot * self.tmem_s_stride,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                slot,
            )

            kv_offset = tScS[0][0]
            pos = kv_offset + n_blk * self.n_block_size

            if q_token_idx < seqlen_q and pos < col_limit and pos < seqlen_k:
                acc_score_0 = (Float32(0.0), Float32(0.0))
                acc_score_1 = (Float32(0.0), Float32(0.0))

                # Two independent accumulators reduce dependency depth in the
                # hot ReLU(QK)*W loop.
                for ho in cutlass.range_constexpr(qhpkv // 2 // W_ILP):
                    for ci in cutlass.range_constexpr(W_ILP):
                        idx0 = q_token_stage * qhpkv + (ho * W_ILP + ci) * 2
                        idx1 = idx0 + 1
                        w_pair = (rW_all[idx0], rW_all[idx1])

                        val0 = tSrS[idx0]
                        val0 = val0 if val0 > Float32(0.0) else Float32(0.0)
                        val1 = tSrS[idx1]
                        val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                        if cutlass.const_expr(ci % 2 == 0):
                            acc_score_0 = fma_packed_f32x2(
                                (val0, val1),
                                w_pair,
                                acc_score_0,
                            )
                        else:
                            acc_score_1 = fma_packed_f32x2(
                                (val0, val1),
                                w_pair,
                                acc_score_1,
                            )

                acc_score = add_packed_f32x2(acc_score_0, acc_score_1)
                score = (acc_score[0] + acc_score[1]) * Float32(softmax_scale)
                if cutlass.const_expr(self.is_compressed_logits):
                    mOut[cand_batch_base + cand_row_offset + Int64(pos)] = score
                else:
                    mOut[q_token_idx, pos] = score

                if cutlass.const_expr(self.compute_lse):
                    # Online LSE is numerically stable and avoids rereading the
                    # output row.  It is compiled out for forward score-only.
                    new_max = score if score > local_max else local_max
                    local_rescale = cute.math.exp2(
                        (local_max - new_max) * log2_e,
                        fastmath=True,
                    )
                    local_sum_exp = local_sum_exp * local_rescale + cute.math.exp2((score - new_max) * log2_e, fastmath=True)
                    local_max = new_max
            n_blk = n_blk - 1

        if cutlass.const_expr(not self.compute_lse):
            return s_full_phase_bits, reduce_phase

        # Reduce the online LSE state across the epilogue warpgroup.  The else
        # branch still toggles the barrier phase so both q-token epilogues keep
        # the same synchronization cadence when a row has no valid columns.
        sScoreAll_sum = cute.make_tensor(
            sScoreAll.iterator + self.num_warps_in_epi_wg,
            cute.make_layout((self.num_warps_in_epi_wg,), stride=(1,)),
        )
        inv_log2_e = Float32(1.0 / math.log2(math.e))

        global_max, reduce_phase = self._intra_inter_warp_reduce_max(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            local_max,
        )

        lse_val = -Float32.inf
        if global_max > Float32(-1e30):
            global_rescale = cute.math.exp2(
                (local_max - global_max) * log2_e,
                fastmath=True,
            )
            adjusted_sum = local_sum_exp * global_rescale

            global_sum_exp, reduce_phase = self._intra_inter_warp_reduce_sum(
                sScoreAll_sum,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                adjusted_sum,
            )
            lse_val = global_max + cute.math.log2(global_sum_exp) * inv_log2_e
        else:
            cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
            cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
            reduce_phase = reduce_phase ^ 1

        if q_token_idx < seqlen_q:
            with cute.arch.elect_one():
                mDenom[q_token_idx] = lse_val

        return s_full_phase_bits, reduce_phase

    @cute.jit
    def _epilogue_indexer_dense_qh32_pair(
        self,
        q_token_stage_base: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sW,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        mOut,
        mDenom: cute.Tensor | None,
        num_n_blocks_compute,
        seqlen_k,
        seqlen_q,
        max_seqlen_k,
        q_causal_offset,
        m_block,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        batch_idx=None,
        cand_batch_offsets=None,
    ):
        """QH32 epilogue for two packed q tokens in one warpgroup.

        The accumulator remains a 128-row tile. Each epilogue warpgroup consumes
        one 64-row half as two 32-head query tokens while retaining the existing
        single TMEM wait/load/release per n block.
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sW_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

        W_ILP = 4 if cutlass.const_expr(self.ratio == 1) else 8
        sW_1d = cute.make_tensor(
            sW.iterator + sW_off,
            cute.make_layout((self.m_block_size,)),
        )
        rW_all = cute.make_rmem_tensor((self.m_block_size,), Float32)

        warp_id_in_wg = tidx_wg // self.WARP_SIZE
        log2_e = Float32(math.log2(math.e))

        q_token_base = m_block * q_tokens_per_tile + Int32(q_token_stage_base)
        q_token_idxs = [q_token_base + Int32(qi) for qi in range(2)]
        col_limits = [(q_causal_offset + q_token_idxs[qi] + Int32(1)) // ratio for qi in range(2)]
        if cutlass.const_expr(self.is_compressed_logits):
            q_global_start = Int64(q_causal_offset)
            if cutlass.const_expr(cand_batch_offsets is not None):
                cand_batch_base = Int64(cand_batch_offsets[batch_idx])
            else:
                cand_batch_base = Int64(batch_idx) * self._row_offset_i64(
                    Int64(seqlen_q),
                    q_global_start,
                )
            cand_row_offsets = [self._row_offset_i64(Int64(q_token_idxs[qi]), q_global_start) for qi in range(2)]

        local_max = [-Float32.inf for _ in range(2)]
        local_sum_exp = [Float32(0.0) for _ in range(2)]
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - Int32(1)
        use_half_tmem = self.compute_lse and self.ratio == 4
        use_two_sums = not (self.compute_lse and self.ratio == 1)
        tmem_head_offset = q_token_stage_base * qhpkv if cutlass.const_expr(use_half_tmem) else 0
        half_tmem_rows = 128
        half_tmem_cols = 64
        half_tmem_repetition = 16
        if cutlass.const_expr(use_half_tmem):
            # Each epilogue warpgroup consumes two of the four H32 query rows:
            # 128 KV rows x 64 head columns, using half the full-tile load width.
            tStS_ref = cute.make_tensor(tStS_ref.iterator, cute.make_layout(((half_tmem_rows, half_tmem_cols), 1, 1), stride=tStS_ref.layout.stride))
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(half_tmem_repetition if use_half_tmem else self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_ref).get_slice(tidx_wg)
            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))
            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sW_1d, rW_all)
                first_block = Int32(0)

            tmem_ptr_cur = cute.make_ptr(
                Float32,
                slot * self.tmem_s_stride + tmem_head_offset,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                slot,
            )

            kv_offset = tScS[0][0]
            pos = kv_offset + n_blk * self.n_block_size

            for qi in cutlass.range_constexpr(2):
                q_token_idx = q_token_idxs[qi]
                col_limit = col_limits[qi]
                if q_token_idx < seqlen_q and pos < col_limit and pos < seqlen_k:
                    acc_score_0 = (Float32(0.0), Float32(0.0))
                    acc_score_1 = (Float32(0.0), Float32(0.0))
                    acc_score_2 = (Float32(0.0), Float32(0.0))
                    acc_score_3 = (Float32(0.0), Float32(0.0))

                    for ho in cutlass.range_constexpr(qhpkv // 2 // W_ILP):
                        for ci in cutlass.range_constexpr(W_ILP):
                            idx0 = (q_token_stage_base + qi) * qhpkv + (ho * W_ILP + ci) * 2
                            idx1 = idx0 + 1
                            w_pair = (rW_all[idx0], rW_all[idx1])

                            val0 = tSrS[idx0 - tmem_head_offset]
                            val0 = val0 if val0 > Float32(0.0) else Float32(0.0)
                            val1 = tSrS[idx1 - tmem_head_offset]
                            val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                            if cutlass.const_expr((ci % 2 == 0) if use_two_sums else (ci < W_ILP // 4)):
                                acc_score_0 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_0,
                                )
                            elif cutlass.const_expr(use_two_sums or ci < W_ILP // 2):
                                acc_score_1 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_1,
                                )
                            elif cutlass.const_expr(ci < (W_ILP * 3) // 4):
                                acc_score_2 = fma_packed_f32x2((val0, val1), w_pair, acc_score_2)
                            else:
                                acc_score_3 = fma_packed_f32x2((val0, val1), w_pair, acc_score_3)

                    acc_score = add_packed_f32x2(acc_score_0, acc_score_1)
                    if cutlass.const_expr(not use_two_sums):
                        acc_score = add_packed_f32x2(acc_score, add_packed_f32x2(acc_score_2, acc_score_3))
                    score = (acc_score[0] + acc_score[1]) * Float32(softmax_scale)
                    if cutlass.const_expr(self.is_compressed_logits):
                        mOut[cand_batch_base + cand_row_offsets[qi] + Int64(pos)] = score
                    else:
                        mOut[q_token_idx, pos] = score

                    if cutlass.const_expr(self.compute_lse):
                        new_max = score if score > local_max[qi] else local_max[qi]
                        local_rescale = cute.math.exp2(
                            (local_max[qi] - new_max) * log2_e,
                            fastmath=True,
                        )
                        local_sum_exp[qi] = local_sum_exp[qi] * local_rescale + cute.math.exp2(
                            (score - new_max) * log2_e,
                            fastmath=True,
                        )
                        local_max[qi] = new_max
            n_blk = n_blk - 1

        if cutlass.const_expr(not self.compute_lse):
            return s_full_phase_bits, reduce_phase

        sScoreAll_sum = cute.make_tensor(
            sScoreAll.iterator + self.num_warps_in_epi_wg,
            cute.make_layout((self.num_warps_in_epi_wg,), stride=(1,)),
        )
        inv_log2_e = Float32(1.0 / math.log2(math.e))

        for qi in cutlass.range_constexpr(2):
            global_max, reduce_phase = self._intra_inter_warp_reduce_max(
                sScoreAll,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                local_max[qi],
            )

            lse_val = -Float32.inf
            if global_max > Float32(-1e30):
                global_rescale = cute.math.exp2(
                    (local_max[qi] - global_max) * log2_e,
                    fastmath=True,
                )
                adjusted_sum = local_sum_exp[qi] * global_rescale

                global_sum_exp, reduce_phase = self._intra_inter_warp_reduce_sum(
                    sScoreAll_sum,
                    reduce_sync_mbar_ptr,
                    reduce_phase,
                    warp_id_in_wg,
                    adjusted_sum,
                )
                lse_val = global_max + cute.math.log2(global_sum_exp) * inv_log2_e
            else:
                cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
                cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
                reduce_phase = reduce_phase ^ 1

            q_token_idx = q_token_idxs[qi]
            if q_token_idx < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_idx] = lse_val

        return s_full_phase_bits, reduce_phase

    @cute.jit
    def _epilogue_indexer_dense_group(
        self,
        wg_index: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sW,
        sScoreAll,
        S_mbar_ptr,
        reduce_sync_mbar_ptr,
        mOut,
        mDenom,
        num_n_blocks_compute,
        seqlen_k,
        seqlen_q,
        max_seqlen_k,
        q_causal_offset,
        m_block,
        tidx,
        s_full_phase_bits,
        reduce_phase,
        softmax_scale,
        per_head_offset=None,
        batch_idx=None,
        cand_batch_offsets=None,
    ):
        """Epilogue of the SMALL head groups (4 / 8 / 16 heads per KV head).

        The MXFP8 tile stays 128 packed rows wide (its scale atom), so a group of ``qhpkv``
        heads packs ``q_tokens_per_tile = 128 // qhpkv`` query tokens per tile (32 / 16 / 8).
        Each of the two epilogue warpgroups owns one 64-column half of the accumulator --
        ``tokens_per_wg = q_tokens_per_tile // 2`` tokens -- and loads ONLY its half (the
        half-tile TMEM load of the QH32 LSE schedule), so a thread carries 64 scores and 64
        weights instead of the whole 128-column row.  Per n block and token: ReLU(QK) * W over
        the group's columns (``W_ILP`` pairs per unrolled step, the whole even group for 4 / 8
        heads), the ratio-causal column limit, one compact or dense store, the optional online
        LSE.  The TMEM wait / load / release cadence per n block is the one every epilogue of
        this kernel keeps (one arrive per thread on the slot's empty barrier).
        """
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sW_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        tokens_per_wg = self.q_tokens_per_tile // 2
        half_cols = tokens_per_wg * qhpkv
        ratio = Int32(self.ratio)

        # The head reduce walks the group's columns in pairs, W_ILP pairs per unrolled step: the
        # whole (even) group for 4 / 8 heads so no column is left outside the loop.
        W_ILP = min(8, qhpkv // 2)
        q_token_stage_base = wg_index * tokens_per_wg
        tmem_head_offset = q_token_stage_base * qhpkv
        sW_half = cute.make_tensor(
            sW.iterator + sW_off + tmem_head_offset,
            cute.make_layout((half_cols,), stride=(1,)),
        )
        rW_half = cute.make_rmem_tensor((half_cols,), Float32)

        warp_id_in_wg = tidx_wg // self.WARP_SIZE
        log2_e = Float32(math.log2(math.e))

        q_token_base = m_block * self.q_tokens_per_tile + Int32(q_token_stage_base)
        q_token_idxs = [q_token_base + Int32(qi) for qi in range(tokens_per_wg)]
        col_limits = [(q_causal_offset + q_token_idxs[qi] + Int32(1)) // ratio for qi in range(tokens_per_wg)]
        if cutlass.const_expr(self.is_compressed_logits):
            q_global_start = Int64(q_causal_offset)
            if cutlass.const_expr(cand_batch_offsets is not None):
                cand_batch_base = Int64(cand_batch_offsets[batch_idx])
            else:
                cand_batch_base = Int64(batch_idx) * self._row_offset_i64(
                    Int64(seqlen_q),
                    q_global_start,
                )
            cand_row_offsets = [self._row_offset_i64(Int64(q_token_idxs[qi]), q_global_start) for qi in range(tokens_per_wg)]

        local_max = [-Float32.inf for _ in range(tokens_per_wg)]
        local_sum_exp = [Float32(0.0) for _ in range(tokens_per_wg)]
        first_block = Int32(1)

        # This warpgroup's half of the accumulator tile: 128 key rows x half_cols fp32 columns at
        # column offset tmem_head_offset of every slot.  The Ld32x32b repetition covers the half in
        # four loads, as the full tile's m_block_size // 4 does for 128 columns.
        half_repetition = half_cols // 4
        tStS_half = cute.make_tensor(
            tStS_ref.iterator,
            cute.make_layout(((self.n_block_size, half_cols), 1, 1), stride=tStS_ref.layout.stride),
        )

        n_blk = num_n_blocks_compute - Int32(1)
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(half_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tStS_half).get_slice(tidx_wg)
            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))
            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_half.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sW_half, rW_half)
                first_block = Int32(0)

            tmem_ptr_cur = cute.make_ptr(
                Float32,
                slot * self.tmem_s_stride + tmem_head_offset,
                mem_space=cute.AddressSpace.tmem,
                assumed_align=16,
            )
            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_half.layout)
            tStS_t2r_cur = thr_tmem_load.partition_S(tStS_cur)

            cute.copy(thr_tmem_load, tStS_t2r_cur, tSrS)
            cute.arch.fence_view_async_tmem_load()

            cute.arch.mbarrier_arrive(S_mbar_ptr + 2 * slot + 1)
            s_full_phase_bits = self._toggle_phase_for_slot(
                s_full_phase_bits,
                slot,
            )

            kv_offset = tScS[0][0]
            pos = kv_offset + n_blk * self.n_block_size

            for qi in cutlass.range_constexpr(tokens_per_wg):
                q_token_idx = q_token_idxs[qi]
                col_limit = col_limits[qi]
                if q_token_idx < seqlen_q and pos < col_limit and pos < seqlen_k:
                    acc_score_0 = (Float32(0.0), Float32(0.0))
                    acc_score_1 = (Float32(0.0), Float32(0.0))

                    for ho in cutlass.range_constexpr(qhpkv // 2 // W_ILP):
                        for ci in cutlass.range_constexpr(W_ILP):
                            # column within this warpgroup's half: token qi's heads are contiguous
                            idx0 = qi * qhpkv + (ho * W_ILP + ci) * 2
                            idx1 = idx0 + 1
                            w_pair = (rW_half[idx0], rW_half[idx1])

                            val0 = tSrS[idx0]
                            val0 = val0 if val0 > Float32(0.0) else Float32(0.0)
                            val1 = tSrS[idx1]
                            val1 = val1 if val1 > Float32(0.0) else Float32(0.0)

                            if cutlass.const_expr(ci % 2 == 0):
                                acc_score_0 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_0,
                                )
                            else:
                                acc_score_1 = fma_packed_f32x2(
                                    (val0, val1),
                                    w_pair,
                                    acc_score_1,
                                )

                    acc_score = add_packed_f32x2(acc_score_0, acc_score_1)
                    score = (acc_score[0] + acc_score[1]) * Float32(softmax_scale)
                    if cutlass.const_expr(self.is_compressed_logits):
                        mOut[cand_batch_base + cand_row_offsets[qi] + Int64(pos)] = score
                    else:
                        mOut[q_token_idx, pos] = score

                    if cutlass.const_expr(self.compute_lse):
                        new_max = score if score > local_max[qi] else local_max[qi]
                        local_rescale = cute.math.exp2(
                            (local_max[qi] - new_max) * log2_e,
                            fastmath=True,
                        )
                        local_sum_exp[qi] = local_sum_exp[qi] * local_rescale + cute.math.exp2(
                            (score - new_max) * log2_e,
                            fastmath=True,
                        )
                        local_max[qi] = new_max
            n_blk = n_blk - 1

        if cutlass.const_expr(not self.compute_lse):
            return s_full_phase_bits, reduce_phase

        sScoreAll_sum = cute.make_tensor(
            sScoreAll.iterator + self.num_warps_in_epi_wg,
            cute.make_layout((self.num_warps_in_epi_wg,), stride=(1,)),
        )
        inv_log2_e = Float32(1.0 / math.log2(math.e))

        for qi in cutlass.range_constexpr(tokens_per_wg):
            global_max, reduce_phase = self._intra_inter_warp_reduce_max(
                sScoreAll,
                reduce_sync_mbar_ptr,
                reduce_phase,
                warp_id_in_wg,
                local_max[qi],
            )

            lse_val = -Float32.inf
            if global_max > Float32(-1e30):
                global_rescale = cute.math.exp2(
                    (local_max[qi] - global_max) * log2_e,
                    fastmath=True,
                )
                adjusted_sum = local_sum_exp[qi] * global_rescale

                global_sum_exp, reduce_phase = self._intra_inter_warp_reduce_sum(
                    sScoreAll_sum,
                    reduce_sync_mbar_ptr,
                    reduce_phase,
                    warp_id_in_wg,
                    adjusted_sum,
                )
                lse_val = global_max + cute.math.log2(global_sum_exp) * inv_log2_e
            else:
                cute.arch.mbarrier_arrive(reduce_sync_mbar_ptr)
                cute.arch.mbarrier_wait(reduce_sync_mbar_ptr, reduce_phase)
                reduce_phase = reduce_phase ^ 1

            q_token_idx = q_token_idxs[qi]
            if q_token_idx < seqlen_q:
                with cute.arch.elect_one():
                    mDenom[q_token_idx] = lse_val

        return s_full_phase_bits, reduce_phase

    @cute.kernel
    def kernel(
        self,
        mQ,
        mK,
        mPerHead,
        mQScale,
        mKScale,
        mOut,
        mDenom: cute.Tensor | None,
        softmax_scale: Float32 | float,
        tma_atom_Q,
        tma_atom_K,
        tma_atom_QScale,
        tma_atom_KScale,
        tiled_mma_qk,
        blockscaled_tiled_mma_qk,
        sQ_layout,
        sK_layout,
        sQScale_layout,
        sKScale_layout,
        tile_sched_params: utils.ClcDynamicPersistentTileSchedulerParams,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        mCandBatchOffsets: cute.Tensor | None = None,
    ):
        is_varlen = mCuSeqlensQ is not None
        if const_expr(is_varlen):
            assert self.is_varlen
            assert mCuSeqlensQScalePadded is not None
            assert mCuSeqlensKScalePadded is not None
        else:
            assert not self.is_varlen
            assert mCuSeqlensQScalePadded is None
            assert mCuSeqlensKScalePadded is None

        seqlen_q_static = max_seqlen_q if const_expr(is_varlen) else cute.size(mQ.shape[0]) // self.qhead_per_kvhead
        seqlen_k_static = max_seqlen_k if const_expr(is_varlen) else cute.size(mK.shape[0])
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=seqlen_q_static,
            seqlen_k_static=seqlen_k_static,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            tile_m=self.m_block_size,
            tile_n=self.n_block_size,
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx = cute.arch.thread_idx()[0]
        neg_log2_e = Float32(-math.log2(math.e))

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_QScale)
            cpasync.prefetch_descriptor(tma_atom_KScale)

        sQ_size = cute.cosize(sQ_layout)
        sK_size = cute.cosize(sK_layout)
        sQScale_size = cute.cosize(sQScale_layout)
        sKScale_size = cute.cosize(sKScale_layout)
        sPerHead_size = self.m_block_size * 2

        # Shared storage is split by lifetime:
        #   - TMA barriers for Q/K and standalone KScale prefetch,
        #   - TMEM full/empty barriers between MMA and epilogue,
        #   - sPerHead double-buffered by work tile,
        #   - sScoreAll scratch for warpgroup reductions.
        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
            KScale_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.KScale_mbar_size]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.S_mbar_size]
            reduce_sync_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.reduce_sync_mbar_size]
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            tmem_holding_buf: Int32
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            clc_response: cute.struct.MemRange[cutlass.Int32, 4]
            sPerHead: cute.struct.Align[
                cute.struct.MemRange[Float32, sPerHead_size],
                128,
            ]
            sScoreAll: cute.struct.Align[
                cute.struct.MemRange[Float32, self.sScoreAll_size],
                self.buffer_align_bytes,
            ]
            sQScale: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sQScale_size],
                self.buffer_align_bytes,
            ]
            sKScale: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, sKScale_size],
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

        Q_mbar_ptr = storage.Q_mbar_ptr.data_ptr()
        K_mbar_ptr = storage.K_mbar_ptr.data_ptr()
        KScale_mbar_ptr = storage.KScale_mbar_ptr.data_ptr()
        S_mbar_ptr = storage.S_mbar_ptr.data_ptr()
        reduce_sync_mbar_ptr = storage.reduce_sync_mbar_ptr.data_ptr()
        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        tmem_holding_buf = storage.tmem_holding_buf.ptr
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sQScale = storage.sQScale.get_tensor(sQScale_layout)
        sKScale = storage.sKScale.get_tensor(sKScale_layout)
        sPerHead = storage.sPerHead.get_tensor(cute.make_layout((self.m_block_size,), stride=(1,)))
        sScoreAll = storage.sScoreAll.get_tensor(cute.make_layout((self.sScoreAll_size,), stride=(1,)))

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        is_first_cta_in_cluster = cta_rank_in_cluster == 0

        # Q is loaded once per work tile.  K and KScale are staged per n block so
        # the MMA warp can consume K blocks from right to left.
        pipeline_Q = PipelineTmaUmma.create(
            barrier_storage=Q_mbar_ptr,
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_K = PipelineTmaUmma.create(
            barrier_storage=K_mbar_ptr,
            num_stages=self.kv_stage,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["K"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        pipeline_KScale = PipelineTmaUmma.create(
            barrier_storage=KScale_mbar_ptr,
            num_stages=self.k_scale_stage,
            producer_group=CooperativeGroup(Agent.Thread, 1),
            consumer_group=CooperativeGroup(Agent.Thread, 1),
            tx_count=self.tma_copy_bytes["KScale"],
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        if warp_idx == 1:
            cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, self.tmem_dealloc_arrive_count)
        if warp_idx == 0:
            for _si in cutlass.range_constexpr(self.num_tmem_slots):
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si, 1)
                cute.arch.mbarrier_init(S_mbar_ptr + 2 * _si + 1, self.s_empty_arrive_count)
            for _ri in cutlass.range_constexpr(self.reduce_sync_mbar_size // 2):
                cute.arch.mbarrier_init(
                    reduce_sync_mbar_ptr + 2 * _ri,
                    self.reduce_sync_arrive_count,
                )

        cluster_size = cute.size(self.cluster_shape_mn)
        num_clc_consumer_threads = self.WARP_SIZE * (1 + cluster_size * (1 + 1 + self.num_epilogue_warps_for_clc))
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
        K_producer, K_consumer = pipeline_K.make_participants()
        KScale_producer, KScale_consumer = pipeline_KScale.make_participants()

        # TMEM is partitioned into accumulator slots followed by SFA/SFB columns.
        # Accumulator slots rotate by n_block modulo num_tmem_slots; each slot
        # has its own full/empty phase bit pair.
        thr_mma_qk = tiled_mma_qk.get_slice(0)
        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        tStS_fake = thr_mma_qk.make_fragment_C(qk_acc_shape)
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        tStS_ref = cute.make_tensor(tmem_ptr, tStS_fake.layout)

        sf_tmem_ptr = cute.recast_ptr(tStS_ref.iterator, dtype=self.sf_dtype)
        tmem_sf_offset_scale = self.qk_acc_dtype.width // self.sf_dtype.width
        k_scale_smem_layout = cute.slice_(sKScale_layout, (None, None, None, 0))
        q_scale_smem_layout = cute.slice_(sQScale_layout, (None, None, None, 0))
        tCtKSFA_layout = _blockscaled_layout.make_tmem_layout_sfa(
            blockscaled_tiled_mma_qk,
            (self.n_block_size, self.m_block_size, self.head_dim_padded),
            self.sf_vec_size,
            k_scale_smem_layout,
        )
        tCtQSFB_layout = _blockscaled_layout.make_tmem_layout_sfb(
            blockscaled_tiled_mma_qk,
            (self.n_block_size, self.m_block_size, self.head_dim_padded),
            self.sf_vec_size,
            q_scale_smem_layout,
        )
        tCtQSFB = cute.make_tensor(
            sf_tmem_ptr + tmem_sf_offset_scale * self.tmem_q_sfb_offset,
            tCtQSFB_layout,
        )
        (
            tiled_copy_s2t_qsfb,
            tCsQSFB_compact_s2t,
            tCtQSFB_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB)

        warp_group_idx = tidx // self.WARPGROUP_SIZE

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            if warp_idx == self.load_warp_id:
                # Load warp: stage W, Q/QScale, then walk K blocks from high to
                # low columns.  The traversal order matches the bottom-right
                # causal mask and lets the epilogue consume completed slots in
                # the same order.
                rows_per_thread = cute.ceil_div(self.m_block_size, self.WARP_SIZE)
                lane_id = tidx % self.WARP_SIZE
                tile_count = Int32(0)
                scale_batch_idx = Int32(-1)
                q_scale_m_block = Int32(0)
                k_scale_n_block = Int32(0)

                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    if const_expr(is_varlen):
                        scale_l_idx = 0
                        if batch_idx != scale_batch_idx:
                            q_scale_m_block = mCuSeqlensQScalePadded[batch_idx] * self.qhead_per_kvhead // self.m_block_size
                            k_scale_n_block = mCuSeqlensKScalePadded[batch_idx] // self.n_block_size
                            scale_batch_idx = batch_idx
                    else:
                        scale_l_idx = batch_idx
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.m_block_size,
                    )
                    is_valid_m_block = m_block_sched < num_m_blocks_cur

                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks_compute = self._dense_compute_n_blocks(
                            m_block,
                            seqlen.seqlen_q,
                            seqlen.seqlen_k,
                            q_causal_offset,
                        )
                        per_head_buf_off = (tile_count % 2) * self.m_block_size
                        mPerHead_cur = seqlen.offset_batch_Q(mPerHead, batch_idx, dim=2)
                        for ri in cutlass.range_constexpr(rows_per_thread):
                            row = ri * self.WARP_SIZE + lane_id
                            if row < self.m_block_size:
                                m_packed_idx = m_block * self.m_block_size + row
                                m_idx = m_packed_idx // self.qhead_per_kvhead
                                h_idx = m_packed_idx - m_idx * self.qhead_per_kvhead
                                if m_idx < seqlen.seqlen_q:
                                    per_head_value = mPerHead_cur[m_idx, h_idx]
                                    sPerHead[per_head_buf_off + row] = Float32(per_head_value)
                                else:
                                    sPerHead[per_head_buf_off + row] = Float32(0.0)

                        cute.arch.fence_view_async_shared()

                        Q_producer.reset()
                        mQ_cur = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)[None, None, 0]
                        mQScale_cur = mQScale[None, None, scale_l_idx]
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

                            if const_expr(_kc == 0):
                                gQScale = cute.local_tile(
                                    mQScale_cur,
                                    (self.m_block_size, self.scale_tile_k),
                                    (q_scale_m_block + m_block, 0),
                                )
                                sQScale_cur = sQScale[
                                    None,
                                    None,
                                    None,
                                    0,
                                ]
                                load_QScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_QScale,
                                    0,
                                    cute.make_layout(1),
                                    gQScale,
                                    sQScale_cur,
                                    filter_zeros=True,
                                    single_stage=True,
                                )
                                load_QScale_fn(tma_bar_ptr=handle_Q.barrier)

                        K_producer.reset()
                        KScale_producer.reset()
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        mKScale_cur = mKScale[None, None, scale_l_idx]
                        n_block_k = num_n_blocks_compute - 1
                        while n_block_k >= Int32(0):
                            # KScale is a full-D128 scale tile and can be
                            # prefetched independently from each K data chunk.
                            handle_KScale = KScale_producer.acquire_and_advance()
                            gKScale = cute.local_tile(
                                mKScale_cur,
                                (self.n_block_size, self.head_dim_padded),
                                (k_scale_n_block + n_block_k, 0),
                            )
                            sKScale_stage = sKScale[None, None, None, handle_KScale.index]
                            load_KScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                tma_atom_KScale,
                                0,
                                cute.make_layout(1),
                                gKScale,
                                sKScale_stage,
                                filter_zeros=True,
                                single_stage=True,
                            )
                            load_KScale_fn(tma_bar_ptr=handle_KScale.barrier)

                            for _kc in cutlass.range_constexpr(self.num_k_chunks):
                                handle_K = K_producer.acquire_and_advance()
                                gK = cute.local_tile(
                                    mK_cur,
                                    (self.n_block_size, self.k_block_size),
                                    (n_block_k, _kc),
                                )
                                sK_stage = sK[None, None, None, handle_K.index]
                                load_K_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_K,
                                    0,
                                    cute.make_layout(1),
                                    gK,
                                    sK_stage,
                                    single_stage=True,
                                )
                                load_K_fn(tma_bar_ptr=handle_K.barrier)
                            n_block_k = n_block_k - 1
                        tile_count = tile_count + 1

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                Q_producer.tail()
                K_producer.tail()
                KScale_producer.tail()

            if warp_idx == self.mma_warp_id:
                # MMA warp: wait for Q once, then for each K block copy SFA/SFB
                # from SMEM to TMEM and issue blockscaled QK GEMM into a rotating
                # accumulator slot.
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
                cute.arch.sync_warp()

                s_empty_phase_bits = Int32((1 << self.num_tmem_slots) - 1)
                K_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.kv_stage)
                KScale_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.k_scale_stage)
                while work_tile.is_valid_tile:
                    m_block_sched = work_tile.tile_idx[0]
                    batch_idx = work_tile.tile_idx[2]
                    seqlen = SeqlenInfoCls(batch_idx)
                    q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                    num_m_blocks_cur = cute.ceil_div(
                        seqlen.seqlen_q * self.qhead_per_kvhead,
                        self.m_block_size,
                    )
                    is_valid_m_block = m_block_sched < num_m_blocks_cur

                    if is_valid_m_block:
                        m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                        num_n_blocks_compute = self._dense_compute_n_blocks(
                            m_block,
                            seqlen.seqlen_q,
                            seqlen.seqlen_k,
                            q_causal_offset,
                        )
                        Q_consumer.reset()
                        handle_Q = Q_consumer.wait_and_advance()

                        tSrK = tiled_mma_qk.make_fragment_A(sK)
                        tSrQ = tiled_mma_qk.make_fragment_B(sQ)
                        q_stage_coord = (None, None, None, None, 0)
                        cute.copy(
                            tiled_copy_s2t_qsfb,
                            tCsQSFB_compact_s2t[q_stage_coord],
                            tCtQSFB_compact_s2t,
                        )

                        n_block = num_n_blocks_compute - 1

                        while n_block >= Int32(0):
                            slot = n_block % Int32(self.num_tmem_slots)
                            s_empty_phase = self._phase_for_slot(s_empty_phase_bits, slot)
                            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot + 1, s_empty_phase)

                            tmem_ptr_cur = cute.make_ptr(
                                Float32,
                                slot * self.tmem_s_stride,
                                mem_space=cute.AddressSpace.tmem,
                                assumed_align=16,
                            )
                            tStS_cur = cute.make_tensor(tmem_ptr_cur, tStS_ref.layout)

                            pipeline_KScale.consumer_wait(KScale_mma_state)
                            k_scale_stage_coord = (None, None, None, None, KScale_mma_state.index)
                            # Each n block has one full-D128 SFA tile shared by
                            # both k64 data chunks. Copy it once into the selected
                            # slot's SFA columns before accumulating the chunks.
                            ksfa_dyn_col = Int32(self.tmem_k_sfa_base_offset) + slot * Int32(self.tmem_sfa_cols)
                            tCtKSFA_dyn = cute.make_tensor(
                                sf_tmem_ptr + Int32(tmem_sf_offset_scale) * ksfa_dyn_col,
                                tCtKSFA_layout,
                            )
                            (
                                tiled_copy_s2t_ksfa_dyn,
                                tCsKSFA_compact_s2t_dyn,
                                tCtKSFA_compact_s2t_dyn,
                            ) = self.mainloop_s2t_copy_and_partition(sKScale, tCtKSFA_dyn)
                            cute.copy(
                                tiled_copy_s2t_ksfa_dyn,
                                tCsKSFA_compact_s2t_dyn[k_scale_stage_coord],
                                tCtKSFA_compact_s2t_dyn,
                            )
                            for _kc in cutlass.range_constexpr(self.num_k_chunks):
                                pipeline_K.consumer_wait(K_mma_state)
                                tSrKi = tSrK[None, None, None, K_mma_state.index]
                                tSrQ_kc = tSrQ[None, None, None, _kc]
                                sK_stage = sK[None, None, None, K_mma_state.index]
                                sQ_stage = sQ[None, None, None, _kc]
                                kphase_offset = const_expr(_kc * (self.k_block_size // self.mma_inst_k))
                                self._gemm_blockscaled_qk(
                                    blockscaled_tiled_mma_qk,
                                    tStS_cur,
                                    tSrKi,
                                    tSrQ_kc,
                                    sK_stage,
                                    sQ_stage,
                                    tCtKSFA_dyn,
                                    tCtQSFB,
                                    accumulate_first=const_expr(_kc != 0),
                                    kphase_offset=kphase_offset,
                                )
                                pipeline_K.consumer_release(K_mma_state)
                                K_mma_state.advance()

                            pipeline_KScale.consumer_release(KScale_mma_state)
                            KScale_mma_state.advance()

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

                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                tmem_ptr_dealloc = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                cute.arch.dealloc_tmem(tmem_ptr_dealloc, Int32(self.tmem_alloc_cols))

            if warp_idx == self.sched_warp_id and is_first_cta_in_cluster:
                # Scheduler warp advances persistent work tiles after all local
                # consumers have released the current tile through CLC.
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

        if warp_group_idx == 1:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            s_full_phase_bits = Int32(0)
            reduce_phase = Int32(0)
            tile_count = Int32(0)
            sScoreAll_epi0 = cute.make_tensor(
                sScoreAll.iterator,
                cute.make_layout((self.sScoreAll_single_size,), stride=(1,)),
            )
            while work_tile.is_valid_tile:
                m_block_sched = work_tile.tile_idx[0]
                batch_idx = work_tile.tile_idx[2]
                seqlen = SeqlenInfoCls(batch_idx)
                q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                num_m_blocks_cur = cute.ceil_div(
                    seqlen.seqlen_q * self.qhead_per_kvhead,
                    self.m_block_size,
                )
                is_valid_m_block = m_block_sched < num_m_blocks_cur

                if is_valid_m_block:
                    m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                    num_n_blocks_compute = self._dense_compute_n_blocks(
                        m_block,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        q_causal_offset,
                    )
                    per_head_offset = (tile_count % 2) * self.m_block_size
                    if cutlass.const_expr(self.is_compressed_logits):
                        mOut_cur = mOut
                    else:
                        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                    mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1) if const_expr(mDenom is not None) else None
                    if cutlass.const_expr(self.qhead_per_kvhead == 64):
                        # Preserve the tuned QH64 single-token specialization.
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_single_q(
                            0,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi0,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    elif cutlass.const_expr(self.qhead_per_kvhead == 32):
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_qh32_pair(
                            0,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi0,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    else:
                        # 4 / 8 / 16 heads: this warpgroup owns the first half of the tile's tokens.
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_group(
                            0,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi0,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    tile_count = tile_count + 1
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg0_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        if warp_group_idx == 2:
            cute.arch.setmaxregister_increase(self.num_regs_epilogue)
            s_full_phase_bits = Int32(0)
            reduce_phase = Int32(0)
            tile_count = Int32(0)
            sScoreAll_epi1 = cute.make_tensor(
                sScoreAll.iterator + self.sScoreAll_single_size,
                cute.make_layout((self.sScoreAll_single_size,), stride=(1,)),
            )
            reduce_sync_mbar_ptr_epi1 = reduce_sync_mbar_ptr + 2
            while work_tile.is_valid_tile:
                m_block_sched = work_tile.tile_idx[0]
                batch_idx = work_tile.tile_idx[2]
                seqlen = SeqlenInfoCls(batch_idx)
                q_causal_offset = Int32(0) if const_expr(mQCausalOffsets is None) else mQCausalOffsets[batch_idx]
                num_m_blocks_cur = cute.ceil_div(
                    seqlen.seqlen_q * self.qhead_per_kvhead,
                    self.m_block_size,
                )
                is_valid_m_block = m_block_sched < num_m_blocks_cur

                if is_valid_m_block:
                    m_block = num_m_blocks_cur - Int32(1) - m_block_sched
                    num_n_blocks_compute = self._dense_compute_n_blocks(
                        m_block,
                        seqlen.seqlen_q,
                        seqlen.seqlen_k,
                        q_causal_offset,
                    )
                    per_head_offset = (tile_count % 2) * self.m_block_size
                    if cutlass.const_expr(self.is_compressed_logits):
                        mOut_cur = mOut
                    else:
                        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                    mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1) if const_expr(mDenom is not None) else None
                    if cutlass.const_expr(self.qhead_per_kvhead == 64):
                        # Preserve the tuned QH64 single-token specialization.
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_single_q(
                            1,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi1,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr_epi1,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    elif cutlass.const_expr(self.qhead_per_kvhead == 32):
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_qh32_pair(
                            2,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi1,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr_epi1,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    else:
                        # 4 / 8 / 16 heads: this warpgroup owns the second half of the tile's tokens.
                        s_full_phase_bits, reduce_phase = self._epilogue_indexer_dense_group(
                            1,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi1,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr_epi1,
                            mOut_cur,
                            mDenom_cur,
                            num_n_blocks_compute,
                            seqlen.seqlen_k,
                            seqlen.seqlen_q,
                            max_seqlen_k,
                            q_causal_offset,
                            m_block,
                            tidx,
                            s_full_phase_bits,
                            reduce_phase,
                            softmax_scale,
                            per_head_offset=per_head_offset,
                            batch_idx=batch_idx,
                            cand_batch_offsets=mCandBatchOffsets,
                        )
                    tile_count = tile_count + 1
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if warp_idx == self.epilogue_wg1_warp_ids[-1]:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
