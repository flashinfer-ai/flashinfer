# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SM100 CuTeDSL dense backward attention score MXFP8 kernel."""

from __future__ import annotations

import math
from functools import partial
from typing import Tuple

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cutlass_dsl import dsl_user_op
from cutlass import Float32, Int32, const_expr
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

from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils import copy as copy_utils
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.seqlen import SeqlenInfoQK
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.score_recompute.dense_score_recompute_sm100 import (
    DenseScoreRecomputeSm100,
    add_packed_f32x2,
    fma_packed_f32x2,
)


@dsl_user_op
def _make_smem_layout_sfa_dynamic_inst_k(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    sfa_tile_shape = (
        mma_tiler_mnk[0] // cute.size(tiled_mma.thr_id.shape),
        mma_tiler_mnk[2],
    )
    smem_layout = cute.tile_to_shape(
        _blockscaled_layout.BlockScaledBasicChunk(sf_vec_size).layout,
        sfa_tile_shape,
        (2, 1),
    )
    mma_tile_inst_m = mma_tiler_mnk[0] // cute.size(tiled_mma.shape_mnk, mode=[0])
    mma_tile_inst_k = mma_tiler_mnk[2] // cute.size(tiled_mma.shape_mnk, mode=[2])
    sfa_tile_shape = cute.shape_div(sfa_tile_shape, (mma_tile_inst_m, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfa_tile_shape)
    smem_layout = cute.logical_divide(smem_layout, ((128, sf_vec_size),))
    return cute.append(
        smem_layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))),
    )


@dsl_user_op
def _make_smem_layout_sfb_dynamic_inst_k(
    tiled_mma: cute.TiledMma,
    mma_tiler_mnk: cute.Tile,
    sf_vec_size: int,
    num_stages: int,
    *,
    loc=None,
    ip=None,
) -> cute.Layout:
    sfb_tile_shape = (cute.round_up(mma_tiler_mnk[1], 128), mma_tiler_mnk[2])
    smem_layout = cute.tile_to_shape(
        _blockscaled_layout.BlockScaledBasicChunk(sf_vec_size).layout,
        sfb_tile_shape,
        (2, 1),
    )
    mma_tile_inst_n = mma_tiler_mnk[1] // cute.size(tiled_mma.shape_mnk, mode=[1])
    mma_tile_inst_k = mma_tiler_mnk[2] // cute.size(tiled_mma.shape_mnk, mode=[2])
    sfb_tile_shape = cute.shape_div(sfb_tile_shape, (mma_tile_inst_n, mma_tile_inst_k))
    smem_layout = cute.tiled_divide(smem_layout, sfb_tile_shape)
    smem_layout = cute.logical_divide(smem_layout, ((128, sf_vec_size),))
    return cute.append(
        smem_layout,
        cute.make_layout(num_stages, stride=cute.cosize(cute.filter_zeros(smem_layout))),
    )


class BwdDenseAttnScoreSm100Mxfp8(DenseScoreRecomputeSm100):
    """SM100 dense backward attention score kernel for MXFP8 Q/K inputs."""

    arch = 100

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
    ):
        if head_dim != 128 and not (head_dim == 512 and qhead_per_kvhead in (64, 128)):
            raise ValueError("SM100 dense attention score MXFP8 currently requires head_dim=128, " "or attention head_dim=512 with qhead_per_kvhead=64/128")
        if sf_vec_size != 32:
            raise ValueError("SM100 dense attention score MXFP8 currently requires sf_vec_size=32")
        if m_block_size != 128 or n_block_size != 128:
            raise ValueError("SM100 dense attention score MXFP8 currently requires " "m_block_size=n_block_size=128")
        k_block_size = head_dim if k_block_size is None else k_block_size
        valid_k_block_sizes = (64, head_dim)
        if head_dim == 512:
            valid_k_block_sizes = (64, 128, 256)
        if k_block_size not in valid_k_block_sizes:
            raise ValueError("SM100 dense attention score MXFP8 currently supports " "k_block_size=head_dim or 64 (and 128 for head_dim=512)")

        super().__init__(
            head_dim=head_dim,
            qhead_per_kvhead=qhead_per_kvhead,
            m_block_size=m_block_size,
            n_block_size=n_block_size,
            kv_stage=kv_stage,
            score_type="attention",
            k_block_size=k_block_size,
            ratio=ratio,
            is_varlen=is_varlen,
        )
        self.sf_vec_size = sf_vec_size
        self.sf_dtype = cutlass.Float8E8M0FNU
        self.sf_groups = head_dim // sf_vec_size
        self.num_regs_load = 64
        self.num_regs_epilogue = 112 if head_dim == 512 and qhead_per_kvhead == 64 else 96
        self.use_dual_attention_epilogue = qhead_per_kvhead == 64 and m_block_size == 128
        self.use_h128_attention_head_split = head_dim == 512 and qhead_per_kvhead == 128 and m_block_size == 128
        self.use_d128_attention_head_split = head_dim == 128 and qhead_per_kvhead == 64 and m_block_size == 128
        self.use_dual_epilogue = self.use_dual_attention_epilogue or self.use_h128_attention_head_split
        self.use_quad_epilogue = self.use_d128_attention_head_split
        self.use_multi_epilogue = self.use_dual_epilogue or self.use_quad_epilogue
        self.chunked_scale_pipeline = head_dim > 128
        self.use_dynamic_scale_layout = head_dim == 512 and k_block_size == 256
        self.scale_tile_k = 256 if self.use_dynamic_scale_layout else 128 if self.chunked_scale_pipeline else self.head_dim_padded
        self.scale_chunks_per_tile = self.scale_tile_k // self.k_block_size
        self.num_q_scale_stages = self.head_dim_padded // self.scale_tile_k if self.chunked_scale_pipeline else self.num_k_chunks
        self.use_multi_qsfb_tmem = head_dim == 512
        self.num_qsfb_tmem_slots = self.num_q_scale_stages if self.use_multi_qsfb_tmem else 1
        self.sScoreAll_single_size = self.num_warps_in_epi_wg * 2
        self.num_epilogue_warps_for_clc = 4
        self.tmem_dealloc_arrive_count = 1
        if self.use_quad_epilogue:
            self.epilogue_wg0_warp_ids = (4, 5, 6, 7)
            self.epilogue_wg1_warp_ids = (8, 9, 10, 11)
            self.epilogue_wg2_warp_ids = (12, 13, 14, 15)
            self.epilogue_wg3_warp_ids = (16, 17, 18, 19)
            self.num_warps = 20
            self.threads_per_cta = self.WARP_SIZE * self.num_warps
            self.s_empty_arrive_count = 4 * self.WARPGROUP_SIZE
            self.num_epilogue_warps_for_clc = 16
            self.tmem_dealloc_arrive_count = 4
            self.reduce_sync_mbar_size = 8
            self.sScoreAll_size = self.sScoreAll_single_size * 4
        elif self.use_dual_epilogue:
            self.epilogue_wg0_warp_ids = (4, 5, 6, 7)
            self.epilogue_wg1_warp_ids = (8, 9, 10, 11)
            self.num_warps = 12
            self.threads_per_cta = self.WARP_SIZE * self.num_warps
            self.s_empty_arrive_count = 2 * self.WARPGROUP_SIZE
            self.num_epilogue_warps_for_clc = 8
            self.tmem_dealloc_arrive_count = 2
            self.reduce_sync_mbar_size = 4
            self.sScoreAll_size = self.sScoreAll_single_size * 2

        # m=n=128 MXFP8 attention TMEM plan:
        # D512/k128 and D128/H64 use 3 accumulator slots:
        # 3 accum + 3 SFA + 1 SFB = 448 TMEM columns. Other attention
        # shapes stay on the BF16-like 2-slot policy.
        self.num_tmem_slots = 3 if ((head_dim == 512 and self.k_block_size == 128) or (head_dim == 128 and qhead_per_kvhead == 64)) else 2
        self.tmem_s_stride = self.m_block_size
        self.tmem_acc_offsets = (0, 128, 256)
        self.tmem_sfa_cols = (self.n_block_size // 32) * 4
        self.tmem_sfb_cols = (self.m_block_size // 32) * 4
        self.tmem_k_sfa_base_offset = 384
        self.tmem_q_sfb_offsets = (432, 448, 464, 480)
        self.tmem_q_sfb_offset = self.tmem_q_sfb_offsets[0]
        self.tmem_total = self.tmem_q_sfb_offset + self.num_qsfb_tmem_slots * self.tmem_sfb_cols
        self.tmem_alloc_cols = 512
        self.S_mbar_size = 2 * self.num_tmem_slots

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mPerHead: cute.Tensor,
        mQScale: cute.Tensor,
        mKScale: cute.Tensor,
        mOut: cute.Tensor,
        mDenom: cute.Tensor,
        softmax_scale: Float32 | float,
        max_seqlen_q: Int32,
        max_seqlen_k: Int32,
        mCuSeqlensQ: cute.Tensor | None,
        mCuSeqlensK: cute.Tensor | None,
        mCuSeqlensQScalePadded: cute.Tensor | None,
        mCuSeqlensKScalePadded: cute.Tensor | None,
        mQCausalOffsets: cute.Tensor | None,
        stream: cuda.CUstream,
    ):
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.sf_dtype = mQScale.element_type
        is_varlen = mCuSeqlensQ is not None

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

        tiled_mma_qk = _make_trivial_tiled_mma(
            self.q_dtype,
            self.q_dtype,
            self.k_major_mode,
            self.q_major_mode,
            self.qk_acc_dtype,
            cta_group,
            self.mma_tiler_qk[:2],
        )
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

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_qk.thr_id.shape,),
        )

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
        if const_expr(self.use_dynamic_scale_layout):
            sKScale_layout = _make_smem_layout_sfa_dynamic_inst_k(
                blockscaled_tiled_mma_qk,
                scale_mma_tiler_qk,
                self.sf_vec_size,
                self.kv_stage,
            )
            sQScale_layout = _make_smem_layout_sfb_dynamic_inst_k(
                blockscaled_tiled_mma_qk,
                scale_mma_tiler_qk,
                self.sf_vec_size,
                self.num_q_scale_stages,
            )
        else:
            sKScale_layout = _blockscaled_layout.make_smem_layout_sfa(
                blockscaled_tiled_mma_qk,
                scale_mma_tiler_qk,
                self.sf_vec_size,
                self.kv_stage,
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
        self.tma_copy_bytes = {
            "Q": (
                self.num_k_chunks * q_data_tma_bytes + self.num_q_scale_stages * q_scale_tma_bytes
                if const_expr(self.chunked_scale_pipeline)
                else self.num_k_chunks * q_data_tma_bytes + q_scale_tma_bytes
            ),
            "K": k_data_tma_bytes + k_scale_tma_bytes,
        }

        PerHead_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mPerHead = cute.make_tensor(mPerHead.iterator, cute.select(mPerHead.layout, mode=PerHead_transpose))

        Out_transpose = [0, 1] if const_expr(is_varlen) else [1, 2, 0]
        mOut = cute.make_tensor(mOut.iterator, cute.select(mOut.layout, mode=Out_transpose))

        Denom_transpose = [0] if const_expr(is_varlen) else [1, 0]
        mDenom = cute.make_tensor(mDenom.iterator, cute.select(mDenom.layout, mode=Denom_transpose))

        num_m_blocks = cute.ceil_div(seqlen_q_packed, self.m_block_size)
        batch_size = cute.size(mCuSeqlensQ.shape[0]) - 1 if const_expr(is_varlen) else cute.size(mQ.shape[3])
        tile_sched_params = utils.ClcDynamicPersistentTileSchedulerParams((num_m_blocks, 1, batch_size), (*self.cluster_shape_mn, 1))
        grid_dim = utils.ClcDynamicPersistentTileScheduler.get_grid_shape(tile_sched_params)
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
    def _epilogue_attention_dense_single_q(
        self,
        q_token_stage: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sLSE,
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
    ):
        """Attention epilogue for one packed q token, used by the 2-WG path."""
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sLSE_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

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

        coord_tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
            Float32,
        )
        coord_thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        coord_cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        kv_offset = (
            tcgen05.make_tmem_copy(
                coord_tmem_load_atom,
                tStS_ref,
            )
            .get_slice(tidx_wg)
            .partition_D(coord_thr_mma.partition_C(coord_cS))[0][0]
        )
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_idx = m_block * q_tokens_per_tile + Int32(q_token_stage)
        local_sum_acc = Float32(0.0)
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - 1
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_ref,
            ).get_slice(tidx_wg)

            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sLSE_1d, rLSE_all)
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

            local_sum = (Float32(0.0), Float32(0.0))
            for ho in cutlass.range_constexpr(qhpkv // 2 // LSE_ILP):
                for ci in cutlass.range_constexpr(LSE_ILP):
                    idx0 = q_token_stage * qhpkv + (ho * LSE_ILP + ci) * 2
                    idx1 = idx0 + 1
                    lse_pair = (rLSE_all[idx0], rLSE_all[idx1])

                    val0 = tSrS[idx0]
                    val1 = tSrS[idx1]
                    val0, val1 = fma_packed_f32x2(
                        (val0, val1),
                        (scale_log2_e, scale_log2_e),
                        lse_pair,
                    )
                    val0 = cute.math.exp2(val0, fastmath=True)
                    val1 = cute.math.exp2(val1, fastmath=True)
                    local_sum = add_packed_f32x2(local_sum, (val0, val1))

            score = local_sum[0] + local_sum[1]
            col_limit = (q_causal_offset + q_token_idx + Int32(1)) // ratio
            score_out = score
            if pos >= col_limit or pos >= seqlen_k or q_token_idx >= seqlen_q:
                score = Float32(0.0)
                score_out = -Float32.inf

            if q_token_idx < seqlen_q and pos < max_seqlen_k:
                mOut[q_token_idx, pos] = score_out

            local_sum_acc = local_sum_acc + score
            n_blk = n_blk - 1

        n_blk_zero = cute.ceil_div(max_seqlen_k, self.n_block_size) - Int32(1)
        while n_blk_zero >= num_n_blocks_compute:
            pos = kv_offset + n_blk_zero * self.n_block_size
            if q_token_idx < seqlen_q and pos < max_seqlen_k:
                mOut[q_token_idx, pos] = -Float32.inf
            n_blk_zero = n_blk_zero - 1

        warp_sum = cute.arch.warp_reduction_sum(local_sum_acc)
        global_sum, reduce_phase = self._inter_warp_sync_sum(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            warp_sum,
        )

        if q_token_idx < seqlen_q:
            with cute.arch.elect_one():
                mDenom[q_token_idx] = global_sum

        return s_full_phase_bits, reduce_phase

    @cute.jit
    def _epilogue_attention_dense_q_head_half(
        self,
        q_token_stage: cutlass.Constexpr[int],
        head_half: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sLSE,
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
    ):
        """D128 attention epilogue split by q token and 32-head half."""
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sLSE_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        q_tokens_per_tile = self.q_tokens_per_tile
        ratio = Int32(self.ratio)

        heads_per_half = qhpkv // 2
        head_base = q_token_stage * qhpkv + head_half * heads_per_half
        LSE_ILP = 4
        sLSE_1d = cute.make_tensor(
            sLSE.iterator + sLSE_off + head_base,
            cute.make_layout((heads_per_half,)),
        )
        rLSE_half = cute.make_rmem_tensor((heads_per_half,), Float32)

        coord_tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
            Float32,
        )
        coord_thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        coord_cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        kv_offset = (
            tcgen05.make_tmem_copy(
                coord_tmem_load_atom,
                tStS_ref,
            )
            .get_slice(tidx_wg)
            .partition_D(coord_thr_mma.partition_C(coord_cS))[0][0]
        )
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_idx = m_block * q_tokens_per_tile + Int32(q_token_stage)
        local_sum_acc = Float32(0.0)
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - 1
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_ref,
            ).get_slice(tidx_wg)

            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sLSE_1d, rLSE_half)
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

            local_sum = (Float32(0.0), Float32(0.0))
            for ho in cutlass.range_constexpr(heads_per_half // 2 // LSE_ILP):
                for ci in cutlass.range_constexpr(LSE_ILP):
                    local_idx0 = (ho * LSE_ILP + ci) * 2
                    local_idx1 = local_idx0 + 1
                    idx0 = head_base + local_idx0
                    idx1 = idx0 + 1
                    lse_pair = (rLSE_half[local_idx0], rLSE_half[local_idx1])

                    val0 = tSrS[idx0]
                    val1 = tSrS[idx1]
                    val0, val1 = fma_packed_f32x2(
                        (val0, val1),
                        (scale_log2_e, scale_log2_e),
                        lse_pair,
                    )
                    val0 = cute.math.exp2(val0, fastmath=True)
                    val1 = cute.math.exp2(val1, fastmath=True)
                    local_sum = add_packed_f32x2(local_sum, (val0, val1))

            score = local_sum[0] + local_sum[1]
            col_limit = (q_causal_offset + q_token_idx + Int32(1)) // ratio
            if pos >= col_limit or pos >= seqlen_k or q_token_idx >= seqlen_q:
                score = Float32(0.0)

            if q_token_idx < seqlen_q and pos < max_seqlen_k:
                if pos < col_limit and pos < seqlen_k:
                    out_row = mOut[q_token_idx, None]
                    cute.arch.atomic_add(
                        (out_row.iterator + pos).llvm_ptr,
                        score,
                    )
                elif const_expr(head_half == 0):
                    mOut[q_token_idx, pos] = -Float32.inf

            local_sum_acc = local_sum_acc + score
            n_blk = n_blk - 1

        if const_expr(head_half == 0):
            n_blk_zero = cute.ceil_div(max_seqlen_k, self.n_block_size) - Int32(1)
            while n_blk_zero >= num_n_blocks_compute:
                pos = kv_offset + n_blk_zero * self.n_block_size
                if q_token_idx < seqlen_q and pos < max_seqlen_k:
                    mOut[q_token_idx, pos] = -Float32.inf
                n_blk_zero = n_blk_zero - 1

        warp_sum = cute.arch.warp_reduction_sum(local_sum_acc)
        global_sum, reduce_phase = self._inter_warp_sync_sum(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            warp_sum,
        )

        if q_token_idx < seqlen_q and warp_id_in_wg == Int32(0):
            with cute.arch.elect_one():
                cute.arch.atomic_add(
                    (mDenom.iterator + q_token_idx).llvm_ptr,
                    global_sum,
                )

        return s_full_phase_bits, reduce_phase

    @cute.jit
    def _epilogue_attention_dense_h128_head_half(
        self,
        head_half: cutlass.Constexpr[int],
        tiled_mma_qk,
        tStS_ref,
        sLSE,
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
    ):
        """H128 attention epilogue split across two 64-head warpgroups."""
        tidx_wg = tidx % self.WARPGROUP_SIZE

        sLSE_off = Int32(0) if per_head_offset is None else per_head_offset
        qhpkv = self.qhead_per_kvhead
        ratio = Int32(self.ratio)

        heads_per_half = qhpkv // 2
        head_base = head_half * heads_per_half
        LSE_ILP = 4
        sLSE_f32_ptr = cute.make_ptr(
            Float32,
            (sLSE.iterator + sLSE_off + head_base).llvm_ptr,
            cute.AddressSpace.smem,
            assumed_align=16,
        )
        sLSE_1d = cute.make_tensor(
            sLSE_f32_ptr,
            cute.make_layout((heads_per_half,)),
        )
        rLSE_half = cute.make_rmem_tensor((heads_per_half,), Float32)

        coord_tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
            Float32,
        )
        coord_thr_mma = tiled_mma_qk.get_slice(tidx_wg)
        coord_cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
        kv_offset = (
            tcgen05.make_tmem_copy(
                coord_tmem_load_atom,
                tStS_ref,
            )
            .get_slice(tidx_wg)
            .partition_D(coord_thr_mma.partition_C(coord_cS))[0][0]
        )
        warp_id_in_wg = tidx_wg // self.WARP_SIZE

        log2_e = Float32(math.log2(math.e))
        scale_log2_e = Float32(softmax_scale) * log2_e

        q_token_idx = m_block
        local_sum_acc = Float32(0.0)
        first_block = Int32(1)

        n_blk = num_n_blocks_compute - 1
        while n_blk >= Int32(0):
            tmem_load_atom = cute.make_copy_atom(
                tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.tmem_repetition)),
                Float32,
            )
            thr_tmem_load = tcgen05.make_tmem_copy(
                tmem_load_atom,
                tStS_ref,
            ).get_slice(tidx_wg)

            thr_mma = tiled_mma_qk.get_slice(tidx_wg)
            cS = cute.make_identity_tensor(self.mma_tiler_qk[:2])
            tScS = thr_tmem_load.partition_D(thr_mma.partition_C(cS))

            tSrS_shape = thr_tmem_load.partition_D(cute.make_identity_tensor(tStS_ref.shape)).shape
            tSrS = cute.make_rmem_tensor(tSrS_shape, Float32)

            slot = n_blk % Int32(self.num_tmem_slots)
            s_full_phase = self._phase_for_slot(s_full_phase_bits, slot)
            cute.arch.mbarrier_wait(S_mbar_ptr + 2 * slot, s_full_phase)

            if first_block == Int32(1):
                cute.autovec_copy(sLSE_1d, rLSE_half)
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

            local_sum = (Float32(0.0), Float32(0.0))
            for ho in cutlass.range_constexpr(heads_per_half // 2 // LSE_ILP):
                for ci in cutlass.range_constexpr(LSE_ILP):
                    local_idx0 = (ho * LSE_ILP + ci) * 2
                    local_idx1 = local_idx0 + 1
                    idx0 = head_base + local_idx0
                    idx1 = idx0 + 1
                    lse_pair = (rLSE_half[local_idx0], rLSE_half[local_idx1])

                    val0 = tSrS[idx0]
                    val1 = tSrS[idx1]
                    val0, val1 = fma_packed_f32x2(
                        (val0, val1),
                        (scale_log2_e, scale_log2_e),
                        lse_pair,
                    )
                    val0 = cute.math.exp2(val0, fastmath=True)
                    val1 = cute.math.exp2(val1, fastmath=True)
                    local_sum = add_packed_f32x2(local_sum, (val0, val1))

            score = local_sum[0] + local_sum[1]
            col_limit = (q_causal_offset + q_token_idx + Int32(1)) // ratio
            if pos >= col_limit or pos >= seqlen_k or q_token_idx >= seqlen_q:
                score = Float32(0.0)

            if q_token_idx < seqlen_q and pos < max_seqlen_k:
                if pos < col_limit and pos < seqlen_k:
                    out_row = mOut[q_token_idx, None]
                    cute.arch.atomic_add(
                        (out_row.iterator + pos).llvm_ptr,
                        score,
                    )
                elif const_expr(head_half == 0):
                    mOut[q_token_idx, pos] = -Float32.inf

            local_sum_acc = local_sum_acc + score
            n_blk = n_blk - 1

        if const_expr(head_half == 0):
            n_blk_zero = cute.ceil_div(max_seqlen_k, self.n_block_size) - Int32(1)
            while n_blk_zero >= num_n_blocks_compute:
                pos = kv_offset + n_blk_zero * self.n_block_size
                if q_token_idx < seqlen_q and pos < max_seqlen_k:
                    mOut[q_token_idx, pos] = -Float32.inf
                n_blk_zero = n_blk_zero - 1

        warp_sum = cute.arch.warp_reduction_sum(local_sum_acc)
        global_sum, reduce_phase = self._inter_warp_sync_sum(
            sScoreAll,
            reduce_sync_mbar_ptr,
            reduce_phase,
            warp_id_in_wg,
            warp_sum,
        )

        if q_token_idx < seqlen_q and warp_id_in_wg == Int32(0):
            with cute.arch.elect_one():
                cute.arch.atomic_add(
                    (mDenom.iterator + q_token_idx).llvm_ptr,
                    global_sum,
                )

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
        mDenom,
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

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.Q_mbar_size]
            K_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.K_mbar_size]
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
        tCtQSFB1 = cute.make_tensor(
            sf_tmem_ptr + tmem_sf_offset_scale * self.tmem_q_sfb_offsets[1],
            tCtQSFB_layout,
        )
        tCtQSFB2 = cute.make_tensor(
            sf_tmem_ptr + tmem_sf_offset_scale * self.tmem_q_sfb_offsets[2],
            tCtQSFB_layout,
        )
        tCtQSFB3 = cute.make_tensor(
            sf_tmem_ptr + tmem_sf_offset_scale * self.tmem_q_sfb_offsets[3],
            tCtQSFB_layout,
        )
        (
            tiled_copy_s2t_qsfb,
            tCsQSFB_compact_s2t,
            tCtQSFB_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB)
        (
            tiled_copy_s2t_qsfb1,
            tCsQSFB_compact_s2t1,
            tCtQSFB1_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB1)
        (
            tiled_copy_s2t_qsfb2,
            tCsQSFB_compact_s2t2,
            tCtQSFB2_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB2)
        (
            tiled_copy_s2t_qsfb3,
            tCsQSFB_compact_s2t3,
            tCtQSFB3_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sQScale, tCtQSFB3)

        warp_group_idx = tidx // self.WARPGROUP_SIZE

        if warp_group_idx == 0:
            cute.arch.setmaxregister_decrease(self.num_regs_load)

            if warp_idx == self.load_warp_id:
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
                                    sPerHead[per_head_buf_off + row] = Float32(per_head_value) * neg_log2_e
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

                            if const_expr((self.chunked_scale_pipeline and _kc % self.scale_chunks_per_tile == 0) or _kc == 0):
                                scale_chunk = const_expr(_kc // self.scale_chunks_per_tile if self.chunked_scale_pipeline else 0)
                                gQScale = cute.local_tile(
                                    mQScale_cur,
                                    (self.m_block_size, self.scale_tile_k),
                                    (q_scale_m_block + m_block, scale_chunk),
                                )
                                sQScale_cur = sQScale[
                                    None,
                                    None,
                                    None,
                                    scale_chunk,
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
                        mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[None, None, 0]
                        mKScale_cur = mKScale[None, None, scale_l_idx]
                        n_block_k = num_n_blocks_compute - 1
                        while n_block_k >= Int32(0):
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
                                gKScale = cute.local_tile(
                                    mKScale_cur,
                                    (self.n_block_size, self.scale_tile_k),
                                    (
                                        k_scale_n_block + n_block_k,
                                        const_expr(_kc // self.scale_chunks_per_tile if self.chunked_scale_pipeline else 0),
                                    ),
                                )
                                sKScale_stage = sKScale[None, None, None, handle_K.index]
                                load_KScale_fn, _, _ = copy_utils.tma_get_copy_fn(
                                    tma_atom_KScale,
                                    0,
                                    cute.make_layout(1),
                                    gKScale,
                                    sKScale_stage,
                                    filter_zeros=True,
                                    single_stage=True,
                                )
                                load_KScale_fn(tma_bar_ptr=handle_K.barrier)
                            n_block_k = n_block_k - 1
                        tile_count = tile_count + 1

                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                Q_producer.tail()
                K_producer.tail()

            if warp_idx == self.mma_warp_id:
                tmem_alloc_cols = Int32(self.tmem_alloc_cols)
                cute.arch.alloc_tmem(tmem_alloc_cols, tmem_holding_buf)
                cute.arch.sync_warp()

                s_empty_phase_bits = Int32((1 << self.num_tmem_slots) - 1)
                K_mma_state = make_pipeline_state(PipelineUserType.Consumer, self.kv_stage)
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
                        if const_expr(not self.chunked_scale_pipeline):
                            q_stage_coord = (None, None, None, None, 0)
                            cute.copy(
                                tiled_copy_s2t_qsfb,
                                tCsQSFB_compact_s2t[q_stage_coord],
                                tCtQSFB_compact_s2t,
                            )
                        elif const_expr(self.use_multi_qsfb_tmem):
                            q_stage_coord0 = (None, None, None, None, 0)
                            q_stage_coord1 = (None, None, None, None, 1)
                            cute.copy(
                                tiled_copy_s2t_qsfb,
                                tCsQSFB_compact_s2t[q_stage_coord0],
                                tCtQSFB_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_qsfb1,
                                tCsQSFB_compact_s2t1[q_stage_coord1],
                                tCtQSFB1_compact_s2t,
                            )
                            if const_expr(self.num_q_scale_stages >= 4):
                                q_stage_coord2 = (None, None, None, None, 2)
                                q_stage_coord3 = (None, None, None, None, 3)
                                cute.copy(
                                    tiled_copy_s2t_qsfb2,
                                    tCsQSFB_compact_s2t2[q_stage_coord2],
                                    tCtQSFB2_compact_s2t,
                                )
                                cute.copy(
                                    tiled_copy_s2t_qsfb3,
                                    tCsQSFB_compact_s2t3[q_stage_coord3],
                                    tCtQSFB3_compact_s2t,
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

                            for _kc in cutlass.range_constexpr(self.num_k_chunks):
                                pipeline_K.consumer_wait(K_mma_state)
                                tSrKi = tSrK[None, None, None, K_mma_state.index]
                                tSrQ_kc = tSrQ[None, None, None, _kc]
                                sK_stage = sK[None, None, None, K_mma_state.index]
                                sQ_stage = sQ[None, None, None, _kc]
                                k_stage_coord = (None, None, None, None, K_mma_state.index)
                                kphase_offset = const_expr(
                                    ((_kc % self.scale_chunks_per_tile) * (self.k_block_size // self.sf_vec_size))
                                    if self.chunked_scale_pipeline
                                    else _kc * (self.k_block_size // self.sf_vec_size)
                                )
                                if const_expr(self.use_multi_qsfb_tmem and (_kc // self.scale_chunks_per_tile) == 0):
                                    tCtQSFB_cur = tCtQSFB
                                elif const_expr(self.use_multi_qsfb_tmem and (_kc // self.scale_chunks_per_tile) == 1):
                                    tCtQSFB_cur = tCtQSFB1
                                elif const_expr(self.use_multi_qsfb_tmem and (_kc // self.scale_chunks_per_tile) == 2):
                                    tCtQSFB_cur = tCtQSFB2
                                elif const_expr(self.use_multi_qsfb_tmem and (_kc // self.scale_chunks_per_tile) == 3):
                                    tCtQSFB_cur = tCtQSFB3
                                else:
                                    tCtQSFB_cur = tCtQSFB
                                if const_expr(self.chunked_scale_pipeline and not self.use_multi_qsfb_tmem and _kc % self.scale_chunks_per_tile == 0):
                                    cute.copy(
                                        tiled_copy_s2t_qsfb,
                                        tCsQSFB_compact_s2t[
                                            (
                                                None,
                                                None,
                                                None,
                                                None,
                                                _kc // self.scale_chunks_per_tile,
                                            )
                                        ],
                                        tCtQSFB_compact_s2t,
                                    )

                                if const_expr((self.chunked_scale_pipeline and _kc % self.scale_chunks_per_tile == 0) or _kc == 0):
                                    cute.copy(
                                        tiled_copy_s2t_ksfa_dyn,
                                        tCsKSFA_compact_s2t_dyn[k_stage_coord],
                                        tCtKSFA_compact_s2t_dyn,
                                    )
                                self._gemm_blockscaled_qk(
                                    blockscaled_tiled_mma_qk,
                                    tStS_cur,
                                    tSrKi,
                                    tSrQ_kc,
                                    sK_stage,
                                    sQ_stage,
                                    tCtKSFA_dyn,
                                    tCtQSFB_cur,
                                    accumulate_first=const_expr(_kc != 0),
                                    kphase_offset=kphase_offset,
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

                cute.arch.relinquish_tmem_alloc_permit()
                cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
                tmem_ptr_dealloc = cute.arch.retrieve_tmem_ptr(
                    Float32,
                    alignment=16,
                    ptr_to_buffer_holding_addr=tmem_holding_buf,
                )
                cute.arch.dealloc_tmem(tmem_ptr_dealloc, Int32(self.tmem_alloc_cols))

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
                    mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                    mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                    if cutlass.const_expr(self.use_d128_attention_head_split):
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_q_head_half(
                            0,
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
                        )
                    elif cutlass.const_expr(self.use_h128_attention_head_split):
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_h128_head_half(
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
                        )
                    elif cutlass.const_expr(self.use_dual_attention_epilogue):
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_single_q(
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
                        )
                    else:
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense(
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll,
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
                        )
                    tile_count = tile_count + 1
                clc_pipeline.consumer_wait(clc_consumer_state)
                work_tile = tile_sched.get_current_work()
                clc_pipeline.consumer_release(clc_consumer_state)
                clc_consumer_state.advance()
            if cutlass.const_expr(self.use_multi_epilogue):
                should_dealloc_arrive = warp_idx == self.epilogue_wg0_warp_ids[-1]
            else:
                should_dealloc_arrive = warp_idx == self.epilogue_wg_warp_ids[-1]
            if should_dealloc_arrive:
                with cute.arch.elect_one():
                    cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        if cutlass.const_expr(self.use_multi_epilogue):
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
                        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                        mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                        if cutlass.const_expr(self.use_d128_attention_head_split):
                            s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_q_head_half(
                                0,
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
                            )
                        elif cutlass.const_expr(self.use_h128_attention_head_split):
                            s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_h128_head_half(
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
                            )
                        elif cutlass.const_expr(self.use_dual_attention_epilogue):
                            s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_single_q(
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
                            )
                        tile_count = tile_count + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if warp_idx == self.epilogue_wg1_warp_ids[-1]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
        if cutlass.const_expr(self.use_d128_attention_head_split):
            if warp_group_idx == 3:
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)
                s_full_phase_bits = Int32(0)
                reduce_phase = Int32(0)
                tile_count = Int32(0)
                sScoreAll_epi2 = cute.make_tensor(
                    sScoreAll.iterator + self.sScoreAll_single_size * 2,
                    cute.make_layout((self.sScoreAll_single_size,), stride=(1,)),
                )
                reduce_sync_mbar_ptr_epi2 = reduce_sync_mbar_ptr + 4
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
                        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                        mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_q_head_half(
                            1,
                            0,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi2,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr_epi2,
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
                        )
                        tile_count = tile_count + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if warp_idx == self.epilogue_wg2_warp_ids[-1]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

            if warp_group_idx == 4:
                cute.arch.setmaxregister_increase(self.num_regs_epilogue)
                s_full_phase_bits = Int32(0)
                reduce_phase = Int32(0)
                tile_count = Int32(0)
                sScoreAll_epi3 = cute.make_tensor(
                    sScoreAll.iterator + self.sScoreAll_single_size * 3,
                    cute.make_layout((self.sScoreAll_single_size,), stride=(1,)),
                )
                reduce_sync_mbar_ptr_epi3 = reduce_sync_mbar_ptr + 6
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
                        mOut_cur = seqlen.offset_batch_Q(mOut, batch_idx, dim=2)
                        mDenom_cur = seqlen.offset_batch_Q(mDenom, batch_idx, dim=1)
                        s_full_phase_bits, reduce_phase = self._epilogue_attention_dense_q_head_half(
                            1,
                            1,
                            tiled_mma_qk,
                            tStS_ref,
                            sPerHead,
                            sScoreAll_epi3,
                            S_mbar_ptr,
                            reduce_sync_mbar_ptr_epi3,
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
                        )
                        tile_count = tile_count + 1
                    clc_pipeline.consumer_wait(clc_consumer_state)
                    work_tile = tile_sched.get_current_work()
                    clc_pipeline.consumer_release(clc_consumer_state)
                    clc_consumer_state.advance()
                if warp_idx == self.epilogue_wg3_warp_ids[-1]:
                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)
