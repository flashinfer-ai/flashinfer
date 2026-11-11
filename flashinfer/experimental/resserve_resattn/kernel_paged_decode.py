# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2025 Rong Shuo (modifications)
# Ported from ResInfer 51b906836a69fa43915040314df57bc9f23030ad.
# Original source: csrc/resattn/kernel_paged_decode.py
# SPDX-License-Identifier: BSD-3-Clause
"""Paged NVFP4 decode adapted from FlashInfer cute_dsl/attention/gqa_decode.py.
NVIDIA BSD-3-Clause; local extensions add paging, residue heads and fused reducers.
FP4 QK with Q residue and QMUL4 V conversion yields BF16 or packed NVFP4 output."""

import math
from typing import Literal, Type, Tuple
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05, OperandMajorMode
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import (
    Agent,
    CooperativeGroup,
    NamedBarrier as nbar,
)

import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.cute.typing import (
    Float32,
    BFloat16,
    Int32,
    Int64,
    Optional,
)

from .output import quantize_nvfp4_block16, output_scale_offset
from .qmul4 import (
    dequantize_nvfp4x8_ldsm,
)

mma_modes = (0, 1, 2)


def _sf_s2t_partition(sf_dtype, sSF, tSF):
    """Partition SMEM-to-TMEM scale copies, following CUTLASS mainloop_s2t_copy_and_partition."""
    tCsSF_compact = cute.filter_zeros(sSF)
    tCtSF_compact = cute.filter_zeros(tSF)
    copy_atom_s2t = cute.make_copy_atom(
        tcgen05.Cp4x32x128bOp(tcgen05.CtaGroup.ONE), sf_dtype
    )
    tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
    thr_copy_s2t = tiled_copy_s2t.get_slice(0)
    tCsSF_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
    tCsSF_s2t = tcgen05.get_s2t_smem_desc_tensor(tiled_copy_s2t, tCsSF_s2t_)
    tCtSF_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
    return tiled_copy_s2t, tCsSF_s2t, tCtSF_s2t


mma_dice = (None, None, None)
cpy_dice = (None,) + mma_dice
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4
max_reduction_splits = 1 << max_reduction_iters

log2_e = math.log2(math.e)
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind="fmax", nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem="relaxed", scope="cta")


class GroupedQueryAttentionDecodePagedFp4:
    """Paged single-token attention with NVFP4 KV and Q residue."""

    def __init__(
        self,
        headdim,
        grouped_head_tile,
        prediction_tile=1,
        sequence_tile=256,
        reduction_mode: Literal[
            "kernel", "gmem_fused", "cga_sharded", "none"
        ] = "kernel",
        *,
        page_size=128,
        logical_grouped_heads: int | None = None,
        single_softmax_wg: bool = False,
        smem_correction_broadcast: bool = False,
    ):
        """Configure attention tiles and split-K reduction."""
        if (
            grouped_head_tile not in (8, 16)
            or prediction_tile != 1
            or headdim not in (128, 256)
        ):
            raise NotImplementedError(
                "production ResAttn requires residue Q8/Q16, D128/D256, decode"
            )
        if reduction_mode not in ("none", "kernel", "gmem_fused", "cga_sharded"):
            raise ValueError(f"unsupported production reducer: {reduction_mode}")
        if headdim == 256 and reduction_mode not in ("none", "kernel"):
            raise NotImplementedError("D256 supports BF16 none/kernel reduction only")
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == "kernel"

        self.do_gmem_fused_red = reduction_mode == "gmem_fused"
        self.do_partial_red = self.do_kernel_red or self.do_gmem_fused_red
        self.do_cga_red = reduction_mode == "cga_sharded"
        self.do_none_red = reduction_mode == "none"

        self.logical_grouped_heads = logical_grouped_heads

        self.single_softmax_wg = single_softmax_wg

        self.smem_correction_broadcast = smem_correction_broadcast

        self.is_q8 = grouped_head_tile == 8

        self.threads_per_cta = (4 if single_softmax_wg else 5) * warpgroup_threads
        self.page_size = page_size

        assert page_size == 128, (
            "page_size must be 128 to align scale-factor atoms; page_size=64 is unsupported"
        )
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        if logical_grouped_heads is not None:
            assert 0 < logical_grouped_heads <= grouped_head_tile
        if smem_correction_broadcast:
            assert single_softmax_wg

    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        d, h_q, s_q, b = tuple(reversed(t_bshd.shape))[:4]
        stride_d, stride_h, stride_s, stride_b = tuple(reversed(t_bshd.stride))[:4]

        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    @staticmethod
    def _sf_layout_hb(m, k, l_shape, sf_vec_size):
        base = blockscaled_utils.tile_atom_to_shape_SF((m, k, 1), sf_vec_size)
        slice_sz = cute.cosize(base)
        h_ext = l_shape[0]
        return cute.make_layout(
            (base.shape[0], base.shape[1], l_shape),
            stride=(
                base.stride[0],
                base.stride[1],
                (slice_sz, h_ext * slice_sz),
            ),
        )

    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)
        dshb = (3, 1, 2, 0)
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        h_q, s_q, b = tuple(reversed(t_bsh.shape))[:3]
        stride_h, stride_s, stride_b = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    @cute.jit
    def __call__(
        self,
        kv_splits: Int32,
        q_bshd: cute.Tensor,
        k_bshd: cute.Tensor,
        v_bshd: cute.Tensor,
        q_sf: cute.Tensor,
        k_sf: cute.Tensor,
        v_sf: cute.Tensor,
        o_bshd: cute.Tensor,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        o_partial_bshd: Optional[cute.Tensor],
        l_partial_bsh: Optional[cute.Tensor],
        m_partial_bsh: Optional[cute.Tensor],
        completion_counter: Optional[cute.Tensor],
        scale_s: Float32,
        scale_o: Float32,
        seq_lens: cute.Tensor,
        block_tables: cute.Tensor,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        """Launch paged NVFP4 decode attention."""

        assert q_bshd.dtype == cutlass.Float4E2M1FN
        assert q_bshd.shape[1] == 1, "production decode requires one Q token"
        assert q_sf is not None and k_sf is not None
        quantize_o = cutlass.const_expr(o_bshd.dtype == cutlass.Float4E2M1FN)
        if cutlass.const_expr(quantize_o):
            assert self.headdim == 128, "D256 NVFP4 output is not supported"
            assert self.do_none_red or self.do_kernel_red or self.do_gmem_fused_red, (
                "fused NVFP4 output requires no reduction, the deterministic "
                "kernel reducer, or the fused last-CTA reducer"
            )
            assert o_block_scale is not None and o_global_scale is not None
            assert o_block_scale.dtype == cutlass.Float8E4M3FN
            assert o_global_scale.dtype == Float32
            assert self.prediction_tile == 1
            if cutlass.const_expr(self.do_gmem_fused_red):
                assert completion_counter is not None
                assert completion_counter.dtype == Int32
        else:
            assert o_block_scale is None and o_global_scale is None
            assert not self.do_gmem_fused_red, (
                "gmem_fused V0 is restricted to direct NVFP4 output"
            )
        kq_dtype = q_bshd.dtype
        v_cache_dtype = v_bshd.dtype

        mma_dtype = cutlass.Float8E4M3FN
        sf_dtype = cutlass.Float8E4M3FN
        sf_vec_size = 16
        acc_dtype = Float32
        assert k_bshd.dtype == kq_dtype

        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)

        mma_tile_m = 128

        mma_tile_k = min(128 * 8 // mma_dtype.width, self.headdim)

        min_mma_tile_n = 8 if self.is_q8 or mma_dtype.width < 8 else 16
        blk_tile_n = blk_tile_h * blk_tile_p
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)

        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0

        tiled_mma_kq = sm100_utils.make_blockscaled_trivial_tiled_mma(
            kq_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            sf_dtype,
            sf_vec_size,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
        )

        mma_tile_n_sfb = max(128, mma_tile_n)
        tiled_mma_kq_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            kq_dtype,
            OperandMajorMode.K,
            OperandMajorMode.K,
            sf_dtype,
            sf_vec_size,
            tcgen05.CtaGroup.ONE,
            (mma_tile_mnk[0], mma_tile_n_sfb),
        )

        assert v_cache_dtype == cutlass.Float4E2M1FN
        assert v_sf is not None
        vp_dtype = cutlass.Float8E4M3FN

        mma_tile_k_vp = mma_tile_k
        mma_tile_m_vp = mma_tile_m
        mma_tile_mnk_vp = (mma_tile_m_vp, mma_tile_n, mma_tile_k_vp)
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m_vp)
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(
            mma_dtype,
            mma_dtype,
            OperandMajorMode.K,
            (OperandMajorMode.K if self.is_q8 else OperandMajorMode.MN),
            acc_dtype,
            tcgen05.CtaGroup.ONE,
            mma_tile_mnk[:2],
            (tcgen05.OperandSource.TMEM),
        )

        self.p_stages = p_stages = 4
        self.o_stages = o_stages = 2

        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages

        sf_cols = (mma_tile_m // 32) * 4
        sf_cols += (mma_tile_n_sfb // 32) * 4

        cvt_stage_cols = mma_tile_k_vp * vp_dtype.width // 32
        cvt_budget = tmem_capacity_cols - tmem_alloc_cols - sf_cols
        cvt_stages = min(8, cvt_budget // cvt_stage_cols)
        assert cvt_stages >= 2
        self.cvt_stages = cvt_stages
        tmem_alloc_cols += cvt_stages * cvt_stage_cols

        _budget = tmem_capacity_cols - tmem_alloc_cols - sf_cols
        max_s_stages = max(1, _budget // tmem_s_stage_cols)
        self.s_stages = s_stages = min(max_s_stages, p_stages)

        tmem_alloc_cols += tmem_s_stage_cols * s_stages + sf_cols
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols

        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2

        mk_stage_bits = mma_tile_m_vp * mma_tile_k_vp * vp_dtype.width
        k_stage_bits = mma_tile_m * mma_tile_k * kq_dtype.width
        nk_stage_bits = mma_tile_n * mma_tile_k * kq_dtype.width
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        sfk_stage_bits = mma_tile_m * (mma_tile_k // sf_vec_size) * sf_dtype.width

        sfk_stage_bits += mma_tile_m * (self.headdim // sf_vec_size) * sf_dtype.width
        sfq_total_bits = (
            mma_tile_n_sfb * (mma_tile_k // sf_vec_size) * sf_dtype.width * tiles_dk
        )

        smem_alloc_bits += Int32.width
        if cutlass.const_expr(self.do_gmem_fused_red):
            smem_alloc_bits += Int32.width

        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_cga_red):
            smem_alloc_bits += blk_tile_n * blk_tile_d * acc_dtype.width
            smem_alloc_bits += max_reduction_splits * warp_threads * 2 * acc_dtype.width
            smem_alloc_bits += mbarrier_bits

        sfv_stage_bits = (mma_tile_m_vp // sf_vec_size) * mma_tile_k_vp * sf_dtype.width

        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits
        smem_alloc_bits += sfq_total_bits
        smem_alloc_bits += s_stages * pipe_stage_bits
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits
        smem_alloc_bits += cvt_stages * pipe_stage_bits
        alignment_bits = 1024 - (smem_alloc_bits % 1024)

        smem_capacity_bits = utils.get_smem_capacity_in_bytes("sm_100") // 1 * 8

        smem_allocator_guard_bits = 512 * 8
        remaining_bits = (
            smem_capacity_bits
            - smem_alloc_bits
            - alignment_bits
            - smem_allocator_guard_bits
        )
        kv_pipe_count = 2
        kv_pair_bits = (
            k_stage_bits
            + sfk_stage_bits
            + mk_stage_bits
            + sfv_stage_bits
            + kv_pipe_count * pipe_stage_bits
        )
        kv_stages = remaining_bits // kv_pair_bits
        if cutlass.const_expr(self.is_q8):
            kv_stages = min(kv_stages, 7)

        h_k = k_bshd.shape[2]
        o_bshd_ = o_partial_bshd if self.do_partial_red else o_bshd

        mQ_nkl = self.gemm_view(self.gqa_pack(q_bshd, h_k), True)
        mK_mkl = self.gemm_view(k_bshd, True)
        mV_mkl = self.gemm_view(v_bshd, False)

        mO_final_mnl = self.gemm_view(self.gqa_pack(o_bshd, h_k), False)

        mO_mnl = self.gemm_view(self.gqa_pack(o_bshd_, h_k), False)

        smem_layout_q = sm100_utils.make_smem_layout_b(
            tiled_mma_kq, mma_tile_mnk, kq_dtype, tiles_dk
        )
        smem_layout_k = sm100_utils.make_smem_layout_a(
            tiled_mma_kq, mma_tile_mnk, kq_dtype, kv_stages
        )

        smem_layout_v = sm100_utils.make_smem_layout_a(
            tiled_mma_vp,
            mma_tile_mnk_vp,
            cutlass.Int8,
            kv_stages,
            is_k_major=False,
        )
        _sf_blocks = mma_tile_m_vp // 16
        _tile_sf = _sf_blocks * mma_tile_k_vp
        smem_layout_v_sf_d = cute.make_layout(
            (4 * _sf_blocks, mma_tile_k_vp // 4, kv_stages),
            stride=(1, 4 * _sf_blocks, _tile_sf),
        )

        smem_layout_sfk = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma_kq, mma_tile_mnk, sf_vec_size, kv_stages
        )
        _ksf_stage_bytes = mma_tile_m * (self.headdim // sf_vec_size)
        smem_layout_sfk_raw = cute.make_layout(
            (16, _ksf_stage_bytes // 16, kv_stages),
            stride=(1, 16, _ksf_stage_bytes),
        )
        smem_layout_sfq = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_kq_sfb,
            (mma_tile_mnk[0], mma_tile_n_sfb, mma_tile_mnk[2]),
            sf_vec_size,
            tiles_dk,
        )

        quantize_o_main = cutlass.const_expr(quantize_o and self.do_none_red)
        o_smem_dtype = Float32 if quantize_o_main else mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(
            tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype
        )
        smem_layout_o = cute.tile_to_shape(
            smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0)
        )
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m_vp, mma_tile_n))

        smem_layout_o = cute.select(smem_layout_o, (0, 1, 2))

        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()

        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        tma_atom_q, tma_tensor_q = cute.nvgpu.make_tiled_tma_atom_B(
            tma_load_op,
            mQ_nkl,
            cute.select(smem_layout_q, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_atom_k, tma_tensor_k = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK_mkl,
            cute.select(smem_layout_k, mma_modes),
            tma_tile_mnk,
            tiled_mma_kq,
        )
        tma_tile_mnk_vp = (mma_tile_m_vp, tma_tile_n, mma_tile_k_vp)
        tma_atom_v, tma_tensor_v = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV_mkl,
            cute.select(smem_layout_v, mma_modes),
            tma_tile_mnk_vp,
            tiled_mma_vp,
            internal_type=cutlass.Int8,
        )

        _hkv = v_sf.shape[1]
        mVSF_d = cute.make_tensor(
            v_sf.iterator,
            cute.make_layout(
                (4 * (self.headdim // 16), self.page_size // 4, (_hkv, v_sf.shape[0])),
                stride=(
                    1,
                    4 * (self.headdim // 16),
                    (v_sf.stride[1], v_sf.stride[0]),
                ),
            ),
        )
        tma_atom_v_sf_d, tma_tensor_v_sf_d = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mVSF_d,
            cute.slice_(smem_layout_v_sf_d, (None, None, 0)),
            (4 * (mma_tile_m_vp // 16), mma_tile_k_vp // 4),
        )

        _ksf_bytes = self.page_size * (self.headdim // sf_vec_size)
        mSFK = cute.make_tensor(
            k_sf.iterator,
            cute.make_layout(
                (16, _ksf_bytes // 16, (k_sf.shape[1], k_sf.shape[0])),
                stride=(
                    1,
                    16,
                    (k_sf.stride[1], k_sf.stride[0]),
                ),
            ),
        )
        tma_atom_sfk, tma_tensor_sfk = cute.nvgpu.cpasync.make_tiled_tma_atom(
            tma_load_op,
            mSFK,
            cute.slice_(smem_layout_sfk_raw, (None, None, 0)),
            (16, _ksf_bytes // 16),
        )
        sfq_layout = self._sf_layout_hb(
            mma_tile_n_sfb, self.headdim, mQ_nkl.shape[2], sf_vec_size
        )
        mSFQ = cute.make_tensor(q_sf.iterator, sfq_layout)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            (1, 1), tiled_mma_kq_sfb.thr_id
        )
        tma_atom_sfq, tma_tensor_sfq = cute.nvgpu.make_tiled_tma_atom_B(
            sfb_op,
            mSFQ,
            cute.select(smem_layout_sfq, mma_modes),
            (mma_tile_mnk[0], mma_tile_n_sfb, mma_tile_mnk[2]),
            tiled_mma_kq_sfb,
            internal_type=cutlass.Int16,
        )

        tma_atom_o = tma_tensor_o = None
        if cutlass.const_expr(not quantize_o_main and not self.do_cga_red):
            tma_atom_o, tma_tensor_o = cute.nvgpu.cpasync.make_tiled_tma_atom(
                tma_store_op,
                mO_mnl,
                cute.select(smem_layout_o, mode=[0, 1]),
                tma_tile_mnk[:2],
            )

        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_partial_red):
            assert m_partial_bsh.dtype == l_partial_bsh.dtype == acc_dtype
            assert o_partial_bshd.dtype in (acc_dtype, BFloat16)

            mM_partial_nl = self.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = self.gemm_view_bsh(l_partial_bsh, h_k)

        scale_s_log2_e = scale_s * log2_e

        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(mQ_nkl.shape[2])
        grid_x = 1 if self.do_none_red else kv_splits
        grid = (grid_x, grid_y, grid_z)
        cluster_x = kv_splits if (self.do_cga_red) else 1

        self.decode(
            blk_tile_shpd,
            mma_tile_mnk,
            mma_tile_mnk_vp,
            tiled_mma_kq,
            tiled_mma_vp,
            mma_dtype,
            kq_dtype,
            vp_dtype,
            o_smem_dtype,
            smem_layout_q,
            tma_atom_q,
            tma_tensor_q,
            smem_layout_k,
            tma_atom_k,
            tma_tensor_k,
            tiled_mma_kq_sfb,
            smem_layout_sfk,
            smem_layout_sfk_raw,
            tma_atom_sfk,
            tma_tensor_sfk,
            smem_layout_sfq,
            tma_atom_sfq,
            tma_tensor_sfq,
            smem_layout_v,
            smem_layout_v_sf_d,
            tma_atom_v,
            tma_tensor_v,
            tma_atom_v_sf_d,
            tma_tensor_v_sf_d,
            smem_layout_o,
            tma_atom_o,
            mO_mnl if quantize_o_main or self.do_cga_red else tma_tensor_o,
            mO_mnl if self.do_gmem_fused_red else None,
            (o_block_scale if quantize_o_main or self.do_gmem_fused_red else None),
            (o_global_scale if quantize_o_main or self.do_gmem_fused_red else None),
            (mO_final_mnl if self.do_cga_red or self.do_gmem_fused_red else None),
            mL_partial_nl,
            mM_partial_nl,
            completion_counter,
            scale_s_log2_e,
            scale_o,
            seq_lens,
            block_tables,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=[cluster_x, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

        if cutlass.const_expr(self.do_kernel_red):
            self.launch_reduction(
                self.headdim,
                o_bshd,
                o_partial_bshd,
                l_partial_bsh,
                m_partial_bsh,
                scale_o,
                o_block_scale,
                o_global_scale,
                self.grouped_head_tile,
                (
                    self.logical_grouped_heads
                    if self.logical_grouped_heads is not None
                    else self.grouped_head_tile
                ),
                stream,
                enable_pdl,
            )

    @cute.jit
    def mma_vp_qmul4_tmem_step(
        self,
        initialize_o: cutlass.Constexpr,
        tiled_mma_vp: cute.TiledMma,
        tAtV_cvt: cute.Tensor,
        tBsP_desc: cute.Tensor,
        tCtO: cute.Tensor,
        p_consumer,
        o_producer,
        v_ready_consumer,
        tiles_sk: cutlass.Constexpr,
        tiles_dm: cutlass.Constexpr,
    ):
        """Compute one VP tile."""
        p_token = p_consumer.try_wait()
        o_token = o_producer.try_acquire()
        v_ready_token = v_ready_consumer.try_wait()

        p_handle = p_consumer.wait_and_advance(p_token)
        o_handle = o_producer.acquire_and_advance(o_token)
        for sk in cutlass.range_constexpr(tiles_sk):
            for dm in cutlass.range_constexpr(tiles_dm):
                v_mma_handle = v_ready_consumer.wait_and_advance(v_ready_token)
                is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                if not is_last_iter:
                    v_ready_token = v_ready_consumer.try_wait()

                mmas_k = cute.size(tAtV_cvt.shape[2])
                for mma_k in cutlass.range_constexpr(mmas_k):
                    tiled_mma_vp.set(
                        tcgen05.Field.ACCUMULATE,
                        not (initialize_o and sk == 0 and dm == 0 and mma_k == 0),
                    )
                    cute.gemm(
                        tiled_mma_vp,
                        tCtO[mma_dice + (dm, o_handle.index)],
                        tAtV_cvt[
                            None,
                            None,
                            mma_k,
                            v_mma_handle.index,
                        ],
                        tBsP_desc[
                            None,
                            None,
                            mma_k,
                            sk,
                            p_handle.index,
                        ],
                        tCtO[mma_dice + (dm, o_handle.index)],
                    )

                v_mma_handle.release()

        p_handle.release()
        o_handle.commit()
        return tiled_mma_vp, p_consumer, o_producer, v_ready_consumer

    @cute.kernel
    def decode(
        self,
        blk_tile_shpd: cute.Tile,
        mma_tile_mnk: cute.Tile,
        mma_tile_mnk_vp: cute.Tile,
        tiled_mma_kq: cute.TiledMma,
        tiled_mma_vp: cute.TiledMma,
        mma_dtype: Type[cutlass.Numeric],
        kq_dtype: Type[cutlass.Numeric],
        vp_dtype: Type[cutlass.Numeric],
        out_dtype: Type[cutlass.Numeric],
        smem_layout_q: cute.ComposedLayout,
        tma_atom_q: cute.CopyAtom,
        mQ: cute.Tensor,
        smem_layout_k: cute.ComposedLayout,
        tma_atom_k: cute.CopyAtom,
        mK: cute.Tensor,
        tiled_mma_kq_sfb: cute.TiledMma,
        smem_layout_sfk: cute.Layout,
        smem_layout_sfk_raw: Optional[cute.Layout],
        tma_atom_sfk: cute.CopyAtom,
        mSFK: cute.Tensor,
        smem_layout_sfq: cute.Layout,
        tma_atom_sfq: cute.CopyAtom,
        mSFQ: cute.Tensor,
        smem_layout_v: cute.ComposedLayout,
        smem_layout_v_sf_d: cute.Layout,
        tma_atom_v: cute.CopyAtom,
        mV: cute.Tensor,
        tma_atom_v_sf_d: cute.CopyAtom,
        mVSF_d: cute.Tensor,
        smem_layout_o: cute.ComposedLayout,
        tma_atom_o: Optional[cute.CopyAtom],
        mO: cute.Tensor,
        mOPartialGlobal: Optional[cute.Tensor],
        mOSF: Optional[cute.Tensor],
        mOGlobalScale: Optional[cute.Tensor],
        mOFinal: Optional[cute.Tensor],
        mL_partial: Optional[cute.Tensor],
        mM_partial: Optional[cute.Tensor],
        mCompletion: Optional[cute.Tensor],
        scale_s_log2_e: Float32,
        scale_o: Float32,
        mSeqLens: cute.Tensor,
        mBlockTables: cute.Tensor,
    ):
        svector_align = 16

        stensor_align = 1024
        smem = utils.SmemAllocator()

        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))

        q_dtype = k_dtype = kq_dtype

        sf_dtype = cutlass.Float8E4M3FN
        o_dtype = out_dtype
        acc_dtype = Float32
        quantize_o = cutlass.const_expr(mO.dtype == cutlass.Float4E2M1FN)
        if cutlass.const_expr(quantize_o):
            assert mOSF is not None and mOGlobalScale is not None
            assert tma_atom_o is None

        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)
        blk_tile_n = blk_tile_h * blk_tile_p
        mma_tile_m, mma_tile_n, mma_tile_k = mma_tile_mnk
        mma_tile_k_vp = mma_tile_mnk_vp[2]
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k_vp
        mma_tile_m_vp = mma_tile_mnk_vp[0]
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m_vp)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)

        do_gmem_fused_red = self.do_gmem_fused_red
        do_partial_red = self.do_partial_red
        do_cga_red = self.do_cga_red
        do_none_red = self.do_none_red

        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_kv_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_qo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = tma_kv_warp_id
        warpgroup_id += 1

        conversion_warpgroup_id = warpgroup_id
        warpgroup_id += 1

        softmax_warpgroups = 1 if self.single_softmax_wg else 2
        softmax_warpgroup_ids = tuple(
            range(warpgroup_id, warpgroup_id + softmax_warpgroups)
        )
        warpgroup_id += softmax_warpgroups

        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads

        max_sw_regs_per_wg_thread = 256
        mma_tma_regs = 64
        conversion_regs = 64

        softmax_regs = 120 if softmax_warpgroups == 1 else 96

        role_reg_budget = (softmax_warpgroups + 3) * (
            64 * 1024 // self.threads_per_cta // 8 * 8
        ) - 8
        correction_regs = min(
            max_sw_regs_per_wg_thread,
            role_reg_budget
            - mma_tma_regs
            - conversion_regs
            - softmax_regs * softmax_warpgroups,
        )
        assert (
            mma_tma_regs
            + conversion_regs
            + softmax_regs * softmax_warpgroups
            + correction_regs
        ) <= role_reg_budget

        kv_splits, grid_tiles_hp, tiles_hb = cute.arch.grid_dim()
        kv_split_idx, coord_hp, coord_hb = cute.arch.block_idx()
        coord_hp_linear = coord_hp
        coord_hb_linear = coord_hb
        if cutlass.const_expr(do_none_red):
            kv_splits, kv_split_idx = (1, 0)
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1

        grouped_heads, prediction = mQ.shape[0]
        heads_k, batches = tiles_hb = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        coord_hb = cute.idx2crd(coord_hb, tiles_hb)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        coord_hg, coord_p = coord_hp
        coord_hk, coord_b = coord_hb

        seqlen = mSeqLens[coord_b]
        page_size = self.page_size
        pages_s = blk_tile_s // page_size
        page_count = cute.ceil_div(seqlen, page_size)

        tiles_s = cute.ceil_div(seqlen, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        prefetch_iters = min(2, self.s_stages - 1)
        exit_early = kv_split_idx >= tiles_s
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        if warp_idx == init_warp and not exit_early:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v_sf_d)
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1

        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)

        if warp_idx == init_warp and not exit_early:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1

        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        tma_threads = warp_threads
        sM_producer_nbar = nbar(3, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(4, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(5, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(7, softmax_threads + correction_threads)
        sM_final_nbar = nbar(9, correction_threads + reduction_threads)
        sL_final_nbar = nbar(10, correction_threads + reduction_threads)
        sO_final_nbar = nbar(11, correction_threads + tma_threads)
        sM_mutex_nbar = nbar(12, softmax_threads * softmax_warpgroups)

        o_init_nbar = nbar(14, correction_threads + warp_threads)

        cga_stats_ready_nbar = nbar(15, correction_threads + reduction_threads)

        gmem_completion_nbar = nbar(15, warpgroup_threads + correction_threads)

        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)

        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative

        cga_partial_mbar = None
        if cutlass.const_expr(do_cga_red):
            cga_partial_mbar = smem.allocate_array(Int64, 1)
            if warp_idx == init_warp:
                with cute.arch.elect_one():
                    partial_o_bytes = blk_tile_n * blk_tile_d * acc_dtype.width // 8
                    partial_stats_bytes = (
                        kv_splits * warp_threads * 2 * acc_dtype.width // 8
                    )
                    cute.arch.mbarrier_init(cga_partial_mbar, 1)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(
                        cga_partial_mbar,
                        Int32(partial_o_bytes + partial_stats_bytes),
                    )
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()

        q_load_mbar = smem.allocate_array(Int64, 1)
        sfq_mbar = smem.allocate_array(Int64, 1)
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                cute.arch.mbarrier_init(sfq_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
                sfq_expect = cutlass.Int32(
                    cute.size_in_bytes(sf_dtype, smem_layout_sfq)
                )
                cute.arch.mbarrier_arrive_and_expect_tx(sfq_mbar, sfq_expect)
        init_warp += 1

        kv_stages = smem_layout_k.shape[-1]
        k_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        k_stage_bytes += cute.size_in_bytes(
            sf_dtype, cute.slice_(smem_layout_sfk_raw, (None, None, 0))
        )
        k_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        k_producer, k_consumer = pipeline.PipelineTmaUmma.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=mma_group,
            tx_count=k_stage_bytes,
            barrier_storage=k_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()
        v_raw_stage_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN,
            cute.select(smem_layout_v, mma_modes),
        )
        v_raw_stage_bytes += cute.size_in_bytes(
            sf_dtype, cute.slice_(smem_layout_v_sf_d, (None, None, 0))
        )
        v_raw_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        v_raw_producer, v_raw_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=kv_stages,
            producer_group=tma_group,
            consumer_group=warpgroup_cooperative,
            tx_count=v_raw_stage_bytes,
            barrier_storage=v_raw_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            enable_multicast_signaling=True,
            defer_sync=True,
        ).make_participants()
        v_ready_stages = self.cvt_stages
        v_ready_pipeline_ptr = smem.allocate_array(Int64, v_ready_stages * 2)
        v_ready_producer, v_ready_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=v_ready_stages,
            producer_group=warpgroup_cooperative,
            consumer_group=mma_group,
            barrier_storage=v_ready_pipeline_ptr,
            cta_layout_vmnk=mcast_layout,
            defer_sync=True,
        ).make_participants()

        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        s_producer, s_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=s_stages,
            producer_group=mma_group,
            consumer_group=softmax_group,
            barrier_storage=s_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        p_producer, p_consumer = pipeline.PipelineAsyncUmma.create(
            num_stages=p_stages,
            producer_group=softmax_group,
            consumer_group=mma_group,
            barrier_storage=p_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        o_producer, o_consumer = pipeline.PipelineUmmaAsync.create(
            num_stages=o_stages,
            producer_group=mma_group,
            consumer_group=correction_group,
            barrier_storage=o_pipeline_ptr,
            defer_sync=True,
        ).make_participants()

        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)

        tAsK = smem.allocate_tensor(
            k_dtype, smem_layout_k.outer, stensor_align, smem_layout_k.inner
        )

        v_smem_dtype = cutlass.Int8
        tAsV = smem.allocate_tensor(
            v_smem_dtype,
            smem_layout_v.outer,
            stensor_align,
            smem_layout_v.inner,
        )
        tAsVRaw = tAsV
        tAsVSFD = smem.allocate_tensor(sf_dtype, smem_layout_v_sf_d, stensor_align)

        _sfb = mma_tile_m_vp // 16
        _tsf_s = _sfb * mma_tile_k_vp
        tAsVSFD = cute.make_tensor(
            tAsVSFD.iterator,
            cute.make_layout(
                (_sfb, (4, mma_tile_k_vp // 4), kv_stages),
                stride=(4, (1, 4 * _sfb), _tsf_s),
            ),
        )
        tBsQ = smem.allocate_tensor(
            q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner
        )
        tAsSFK = smem.allocate_tensor(sf_dtype, smem_layout_sfk, stensor_align)
        tAsSFKRaw = smem.allocate_tensor(sf_dtype, smem_layout_sfk_raw, stensor_align)
        tBsSFQ = smem.allocate_tensor(sf_dtype, smem_layout_sfq, stensor_align)
        _ksf_stage_bytes = mma_tile_m * (mma_tile_k // 16)
        _tAsSFKAtomBytes = cute.recast_tensor(tAsSFK, cutlass.Uint8)
        tAsSFKAtomU32 = cute.make_tensor(
            cute.recast_ptr(_tAsSFKAtomBytes.iterator, dtype=cutlass.Uint32),
            cute.make_layout(
                (_ksf_stage_bytes // 4, kv_stages),
                stride=(1, _ksf_stage_bytes // 4),
            ),
        )
        _tAsSFKRawBytes = cute.recast_tensor(tAsSFKRaw, cutlass.Uint8)
        tAsSFKRawU64 = cute.make_tensor(
            cute.recast_ptr(_tAsSFKRawBytes.iterator, dtype=cutlass.Uint64),
            cute.make_layout(
                (mma_tile_m * tiles_dk, kv_stages),
                stride=(1, mma_tile_m * tiles_dk),
            ),
        )

        tCtS_shape = tiled_mma_kq.partition_shape_C(
            (mma_tile_m, mma_tile_n, tiles_sm, s_stages)
        )
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)

        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(
            tiled_mma_vp, blk_tile_nm, vp_dtype, p_stages
        )

        tBsP_nm = smem.allocate_tensor(
            vp_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner
        )

        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))

        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)

        sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)

        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)

        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_stages))

        tAtV_cvt_shape = tiled_mma_vp.partition_shape_A(
            (mma_tile_m_vp, mma_tile_k_vp, self.cvt_stages)
        )
        tAtV_cvt = thrblk_mma_vp.make_fragment_A(tAtV_cvt_shape)

        sM_layout = cute.make_layout(blk_tile_n)
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx] = -Float32.inf
        init_warp += 1

        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1

        sGmemLast = None
        if cutlass.const_expr(do_gmem_fused_red):
            sGmemLast = smem.allocate_tensor(Int32, cute.make_layout(1), svector_align)

        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)

        sCgaO = None
        sCgaStats = None
        if cutlass.const_expr(do_cga_red):
            sCgaO = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout((blk_tile_d * blk_tile_n,)),
                svector_align,
            )
            sCgaStats = smem.allocate_tensor(
                acc_dtype,
                cute.make_layout(
                    (max_reduction_splits, warp_threads, 2),
                    stride=(warp_threads * 2, 2, 1),
                ),
                svector_align,
            )

        if cutlass.const_expr(do_cga_red):
            cute.arch.cluster_wait()

        cute.arch.sync_threads()
        assert init_warp < (self.threads_per_cta // warp_threads), (
            f"used {init_warp + 1} init warps, {self.threads_per_cta // warp_threads} warps available"
        )

        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0

        tAtV_cvt = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=vp_dtype),
            tAtV_cvt.layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tAtV_cvt)

        tCtS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)

        tCtL = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)

        tCtO = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)

        tCtSFK_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma_kq,
            (mma_tile_m, mma_tile_n, mma_tile_k),
            16,
            cute.slice_(smem_layout_sfk, (None, None, None, 0)),
        )
        tCtSFK = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
            tCtSFK_layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFK)
        tCtSFQ_layout = blockscaled_utils.make_tmem_layout_sfb(
            tiled_mma_kq_sfb,
            (mma_tile_m, max(128, mma_tile_n), mma_tile_k),
            16,
            cute.slice_(smem_layout_sfq, (None, None, None, 0)),
        )
        tCtSFQ = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + tmem_offset, dtype=sf_dtype),
            tCtSFQ_layout,
        )
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtSFQ)

        assert tmem_offset <= tmem_alloc_cols, (
            f"\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated"
        )

        partial_o = None
        if cutlass.const_expr(do_cga_red):
            partial_o = cute.make_rmem_tensor((blk_tile_n,), acc_dtype)
            partial_o.fill(0)

        if exit_early:
            if warpgroup_idx == correction_warpgroup_id:
                sM_final_nbar.arrive()
                sL_final_nbar.arrive()
            if cutlass.const_expr(quantize_o):
                if warp_idx == tma_kv_warp_id:
                    mO_bytes = cute.recast_tensor(mO, cutlass.Uint8)
                    logical_heads = self.logical_grouped_heads
                    if cutlass.const_expr(logical_heads is None):
                        logical_heads = grouped_heads
                    d_blocks = blk_tile_d // 16
                    hidden = heads_k * logical_heads * blk_tile_d
                    for block in cutlass.range(
                        lane_idx, blk_tile_h * d_blocks, warp_threads
                    ):
                        group_head = coord_hg * blk_tile_h + block // d_blocks
                        d_block = block % d_blocks
                        if group_head < logical_heads:
                            for byte in cutlass.range_constexpr(8):
                                mO_bytes[
                                    d_block * 8 + byte,
                                    (group_head, 0),
                                    (coord_hk, coord_b),
                                ] = cutlass.Uint8(0)
                            k_idx = (
                                coord_hk * logical_heads + group_head
                            ) * d_blocks + d_block
                            sf_offset = output_scale_offset(coord_b, k_idx, hidden)
                            mOSF[sf_offset] = sf_dtype(0)

        elif warp_idx == tma_kv_warp_id:
            gK = cute.local_tile(
                mK, tiler=(mma_tile_m, mma_tile_k), coord=(0, None, (coord_hk, None))
            )

            gV = cute.local_tile(
                mV,
                tiler=(mma_tile_m_vp, mma_tile_k_vp),
                coord=(None, None, (coord_hk, None)),
            )

            tAgK = thrblk_mma_kq.partition_A(gK)
            tAgV = thrblk_mma_vp.partition_A(gV)

            tGSsK, tGSgK = cute.nvgpu.cpasync.tma_partition(
                tma_atom_k,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsK, 0, 3),
                gmem_tensor=cute.group_modes(tAgK, 0, 3),
            )

            tGSsV, tGSgV = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsV, 0, 3),
                gmem_tensor=cute.group_modes(tAgV, 0, 3),
            )
            gVSFD = cute.local_tile(
                mVSF_d,
                tiler=(4 * (mma_tile_m_vp // 16), mma_tile_k_vp // 4),
                coord=(None, None, (coord_hk, None)),
            )
            tGSsVSFD, tGSgVSFD = cute.nvgpu.cpasync.tma_partition(
                tma_atom_v_sf_d,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsVSFD, 0, 2),
                gmem_tensor=cute.group_modes(gVSFD, 0, 2),
            )
            gSFK = cute.local_tile(
                mSFK,
                tiler=(16, self.page_size * (self.headdim // 16) // 16),
                coord=(None, None, (coord_hk, None)),
            )
            tGSsSFK, tGSgSFK = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfk,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tAsSFKRaw, 0, 2),
                gmem_tensor=cute.group_modes(gSFK, 0, 2),
            )

            cute.arch.griddepcontrol_wait()

            v_prefetch_iters = 0 if self.is_q8 else prefetch_iters
            prefetch_tiles = v_prefetch_iters * kv_splits
            k_token_tma = True
            v_raw_token_tma = True
            for s in cutlass.range(kv_split_idx, prefetch_tiles + tiles_s, kv_splits):
                if s < tiles_s:
                    for sm in cutlass.range_constexpr(tiles_sm):
                        logical_k = s * pages_s + sm
                        phys_k = Int32(-1)
                        if logical_k < page_count:
                            phys_k = mBlockTables[coord_b, logical_k]
                        for dk in cutlass.range_constexpr(tiles_dk):
                            k_handle = k_producer.acquire_and_advance(k_token_tma)
                            k_token_tma = k_producer.try_acquire()
                            cute.copy(
                                tma_atom_k,
                                tGSgK[None, dk, phys_k],
                                tGSsK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                            cute.copy(
                                tma_atom_sfk,
                                tGSgSFK[None, 0, 0, phys_k],
                                tGSsSFK[None, k_handle.index],
                                tma_bar_ptr=k_handle.barrier,
                            )
                if s >= prefetch_tiles:
                    s_v = s - prefetch_tiles
                    for sk in cutlass.range_constexpr(tiles_sk):
                        logical_v = s_v * pages_s + sk
                        phys_v = Int32(-1)
                        if logical_v < page_count:
                            phys_v = mBlockTables[coord_b, logical_v]
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = v_raw_producer.acquire_and_advance(
                                v_raw_token_tma
                            )
                            v_raw_token_tma = v_raw_producer.try_acquire()
                            cute.copy(
                                tma_atom_v,
                                tGSgV[None, dm, 0, phys_v],
                                tGSsV[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )
                            cute.copy(
                                tma_atom_v_sf_d,
                                tGSgVSFD[None, dm, 0, phys_v],
                                tGSsVSFD[None, v_handle.index],
                                tma_bar_ptr=v_handle.barrier,
                            )

        elif warp_idx == tma_qo_warp_id:
            gQ = cute.local_tile(
                mQ,
                tiler=(blk_tile_hp, blk_tile_d),
                coord=(coord_hp, 0, coord_hb),
            )

            gQ_mma = cute.flat_divide(gQ, (mma_tile_n, mma_tile_k))
            gQ_mma = gQ_mma[None, None, 0, None]

            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)

            tBsQ_tma, tBgQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_q,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgQ, 0, 3),
            )

            cute.arch.griddepcontrol_wait()

            cute.copy(
                tma_atom_q,
                tBgQ_tma,
                tBsQ_tma,
                tma_bar_ptr=q_load_mbar,
            )

            gSFQ = cute.local_tile(
                mSFQ,
                tiler=(max(128, mma_tile_n), blk_tile_d),
                coord=(0, 0, coord_hb),
            )
            gSFQ_mma = cute.flat_divide(gSFQ, (max(128, mma_tile_n), mma_tile_k))[
                None, None, 0, None
            ]
            thrblk_sfb = tiled_mma_kq_sfb.get_slice(0)
            tBgSFQ = thrblk_sfb.partition_B(gSFQ_mma)
            tBsSFQ_tma, tBgSFQ_tma = cute.nvgpu.cpasync.tma_partition(
                tma_atom_sfq,
                mcast_coord,
                mcast_layout,
                smem_tensor=cute.group_modes(tBsSFQ, 0, 3),
                gmem_tensor=cute.group_modes(tBgSFQ, 0, 3),
            )
            tBsSFQ_tma = cute.filter_zeros(tBsSFQ_tma)
            tBgSFQ_tma = cute.filter_zeros(tBgSFQ_tma)
            cute.copy(
                tma_atom_sfq,
                tBgSFQ_tma,
                tBsSFQ_tma,
                tma_bar_ptr=sfq_mbar,
            )

            coord_b_partial = (
                kv_split_idx * batches + coord_b if do_partial_red else coord_b
            )
            gO = None
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                gO = cute.local_tile(
                    mO,
                    tiler=(blk_tile_d, blk_tile_hp),
                    coord=(0, coord_hp, (coord_hk, coord_b_partial)),
                )
            sO_tma = gO_tma = None
            if cutlass.const_expr(not quantize_o and not do_cga_red):
                gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
                gO_mma = gO_mma[None, None, None, 0]

                sO_tma, gO_tma = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_o,
                    mcast_coord,
                    mcast_layout,
                    smem_tensor=cute.group_modes(sO_mma, 0, 2),
                    gmem_tensor=cute.group_modes(gO_mma, 0, 2),
                )

            if cutlass.const_expr(do_cga_red):
                pass
            elif cutlass.const_expr(quantize_o):
                sO_final_nbar.arrive_and_wait()

                assert blk_tile_p == 1 and blk_tile_d % 16 == 0
                d_scale_blocks = blk_tile_d // 16
                scale_blocks = blk_tile_h * d_scale_blocks
                mO_bytes = cute.recast_tensor(mO, cutlass.Uint8)
                global_scale = Float32(mOGlobalScale[0])
                logical_grouped_heads = self.logical_grouped_heads
                if cutlass.const_expr(logical_grouped_heads is None):
                    logical_grouped_heads = grouped_heads
                hidden = heads_k * logical_grouped_heads * blk_tile_d
                row_idx = coord_b

                for block_idx in cutlass.range(lane_idx, scale_blocks, warp_threads):
                    local_head = block_idx // d_scale_blocks
                    d_block = block_idx % d_scale_blocks
                    group_head = coord_hg * blk_tile_h + local_head
                    if group_head < logical_grouped_heads:
                        values = cute.make_rmem_tensor((16,), Float32)
                        for value_idx in cutlass.range_constexpr(16):
                            value = Float32(
                                sO_mma[
                                    d_block * 16 + value_idx,
                                    local_head,
                                    0,
                                ]
                            )
                            values[value_idx] = value
                        packed_values, sf_value = quantize_nvfp4_block16(
                            values.load(), global_scale
                        )

                        packed_fp4 = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
                        packed_fp4.store(packed_values)
                        packed_bytes = cute.recast_tensor(packed_fp4, cutlass.Uint8)
                        for byte_idx in cutlass.range_constexpr(8):
                            mO_bytes[
                                d_block * 8 + byte_idx,
                                (group_head, 0),
                                (coord_hk, coord_b),
                            ] = packed_bytes[byte_idx]

                        global_head = coord_hk * logical_grouped_heads + group_head
                        k_idx = global_head * d_scale_blocks + d_block
                        sf_offset = output_scale_offset(row_idx, k_idx, hidden)
                        mOSF[sf_offset] = sf_value
            else:
                sO_final_nbar.arrive_and_wait()
                cute.copy(tma_atom_o, sO_tma, gO_tma)
                if cutlass.const_expr(do_gmem_fused_red):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=False)

        elif warpgroup_idx == conversion_warpgroup_id:
            mma_k_bits = mma_tile_k_vp * vp_dtype.width

            tmem_store_atom_v = cute.make_copy_atom(
                tcgen05.St16x128bOp(tcgen05.Repetition(mma_k_bits // 128)),
                vp_dtype,
            )
            smem_load_atom_v = cute.make_copy_atom(
                cute.nvgpu.warp.LdMatrix16x16x8bOp(
                    transpose=True,
                    num_matrices=2,
                    unpack_bits=4,
                ),
                cutlass.Int8,
            )
            tmem_store_v = tcgen05.make_tmem_copy(
                tmem_store_atom_v, tAtV_cvt[mma_dice + (0,)]
            )
            thr_store_v = tmem_store_v.get_slice(warpgroup_tidx)
            tVrV_cvt_shape = thr_store_v.partition_S(tAtV_cvt).shape[:-1]
            tVtV_cvt = thr_store_v.partition_D(tAtV_cvt)

            smem_load_v = cute.make_tiled_copy_S(smem_load_atom_v, tmem_store_v)
            thr_load_v = smem_load_v.get_slice(warpgroup_tidx)
            tVsV = thr_load_v.partition_S(tAsVRaw)

            tVsV = cute.make_tensor(tVsV.iterator.align(16), tVsV.layout)
            tVrV_shape = thr_load_v.partition_D(tAsVRaw).shape[:-1]

            _sfb = mma_tile_m_vp // 16
            sVS = cute.make_tensor(
                tAsVSFD.iterator,
                cute.make_layout(
                    ((16, _sfb), (4, mma_tile_k_vp // 4), kv_stages),
                    stride=((0, 4), (1, 4 * _sfb), _sfb * mma_tile_k_vp),
                ),
            )
            sVS_mma = cute.group_modes(
                cute.flat_divide(sVS, (mma_tile_m_vp, mma_tile_k_vp)),
                2,
                4,
            )
            tAsVS = thrblk_mma_vp.partition_A(sVS_mma)
            tVsVS = thr_load_v.partition_D(tAsVS)

            v_token = v_raw_consumer.try_wait()
            cvt_token = v_ready_producer.try_acquire()
            for _s in cutlass.range(iters_s):
                for _sk in cutlass.range_constexpr(tiles_sk):
                    for _dm in cutlass.range_constexpr(tiles_dm):
                        v_handle = v_raw_consumer.wait_and_advance(v_token)
                        cvt_handle = v_ready_producer.acquire_and_advance(cvt_token)

                        tVrVS = cute.make_rmem_tensor_like(tVsVS[cpy_dice + (None, 0)])
                        cute.autovec_copy(
                            tVsVS[cpy_dice + (None, v_handle.index)],
                            tVrVS,
                        )
                        tVrV = cute.make_rmem_tensor(tVrV_shape, cutlass.Int8)
                        tVrV_cvt = cute.make_rmem_tensor(tVrV_cvt_shape, vp_dtype)

                        tVrVS_expanded = cute.make_rmem_tensor(tVrV_shape, sf_dtype)
                        cute.autovec_copy(tVrVS, tVrVS_expanded)
                        cute.copy(
                            thr_load_v,
                            tVsV[cpy_dice + (v_handle.index,)],
                            tVrV,
                        )
                        cute.arch.fence_view_async_shared()
                        v_handle.release()

                        _frag_values = cute.size(tVrV_shape)
                        assert _frag_values % 8 == 0
                        _src_flat = cute.make_tensor(
                            tVrV.iterator,
                            cute.make_layout(_frag_values),
                        )
                        _sf_flat = cute.make_tensor(
                            tVrVS_expanded.iterator,
                            cute.make_layout(_frag_values),
                        )
                        _dst_flat = cute.make_tensor(
                            tVrV_cvt.iterator,
                            cute.make_layout(_frag_values),
                        )
                        for _block in cutlass.range_constexpr(_frag_values // 8):
                            _src8 = cute.local_tile(_src_flat, (8,), (_block,))
                            _sf8 = cute.local_tile(_sf_flat, (8,), (_block,))
                            _dst8 = cute.local_tile(_dst_flat, (8,), (_block,))
                            _dst8.store(dequantize_nvfp4x8_ldsm(_src8, _sf8))
                        cute.copy(
                            thr_store_v,
                            tVrV_cvt,
                            tVtV_cvt[cpy_dice + (cvt_handle.index,)],
                        )
                        cute.arch.fence_view_async_tmem_store()
                        cvt_handle.commit()
                        v_token = v_raw_consumer.try_wait()
                        cvt_token = v_ready_producer.try_acquire()

        elif warp_idx == mma_kq_warp_id:
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)

            copy_sfk, tCsSFK_s2t, tCtSFK_s2t = _sf_s2t_partition(
                sf_dtype, tAsSFK, tCtSFK
            )
            copy_sfq, tCsSFQ_s2t, tCtSFQ_s2t = _sf_s2t_partition(
                sf_dtype, tBsSFQ, tCtSFQ
            )

            cute.arch.mbarrier_wait(q_load_mbar, phase=0)

            cute.arch.mbarrier_wait(sfq_mbar, phase=0)
            if cutlass.const_expr(tiles_dk == 1):
                cute.copy(
                    copy_sfq,
                    tCsSFQ_s2t[None, None, None, None, 0],
                    tCtSFQ_s2t,
                )

            for _s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()

                k_token = k_consumer.try_wait()

                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = k_consumer.wait_and_advance(k_token)
                        if cutlass.const_expr(tiles_dk > 1):
                            cute.copy(
                                copy_sfq,
                                tCsSFQ_s2t[None, None, None, None, dk],
                                tCtSFQ_s2t,
                            )
                        sf_stage = k_handle.index
                        for _i in cutlass.range_constexpr(mma_tile_m // warp_threads):
                            _row = lane_idx + _i * warp_threads
                            _value = tAsSFKRawU64[_row * tiles_dk + dk, sf_stage]
                            _atom_lo = (_row % 32) * 16 + (_row // 32) * 4
                            tAsSFKAtomU32[_atom_lo // 4, sf_stage] = cutlass.Uint32(
                                _value
                            )
                            tAsSFKAtomU32[(_atom_lo + 512) // 4, sf_stage] = (
                                cutlass.Uint32(_value >> 32)
                            )
                        cute.arch.fence_view_async_shared()
                        cute.copy(
                            copy_sfk,
                            tCsSFK_s2t[
                                None,
                                None,
                                None,
                                None,
                                sf_stage,
                            ],
                            tCtSFK_s2t,
                        )
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if not is_last_iter:
                            k_token = k_consumer.try_wait()

                        mmas_k = cute.size(tAsK.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            tiled_mma_kq.set(
                                tcgen05.Field.SFA,
                                tCtSFK[None, None, mma_k].iterator,
                            )
                            tiled_mma_kq.set(
                                tcgen05.Field.SFB,
                                tCtSFQ[None, None, mma_k].iterator,
                            )
                            cute.gemm(
                                tiled_mma_kq,
                                tCtS[mma_dice + (sm, s_handle.index)],
                                tAsK_desc[None, None, mma_k, k_handle.index],
                                tBsQ_desc[None, None, mma_k, dk],
                                tCtS[mma_dice + (sm, s_handle.index)],
                            )
                            if dk == 0 and mma_k == 0:
                                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                s_handle.commit()

        elif warp_idx == mma_vp_warp_id:
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            tiled_mma_vp.set(
                tcgen05.Field.ACCUMULATE,
                True,
            )

            o_init_nbar.arrive_and_wait()

            (
                tiled_mma_vp,
                p_consumer,
                o_producer,
                v_ready_consumer,
            ) = self.mma_vp_qmul4_tmem_step(
                True,
                tiled_mma_vp,
                tAtV_cvt,
                tBsP_desc,
                tCtO,
                p_consumer,
                o_producer,
                v_ready_consumer,
                tiles_sk,
                tiles_dm,
            )
            if iters_s > 1:
                (
                    tiled_mma_vp,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                ) = self.mma_vp_qmul4_tmem_step(
                    True,
                    tiled_mma_vp,
                    tAtV_cvt,
                    tBsP_desc,
                    tCtO,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                    tiles_sk,
                    tiles_dm,
                )
            for _s in cutlass.range(2, iters_s):
                (
                    tiled_mma_vp,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                ) = self.mma_vp_qmul4_tmem_step(
                    False,
                    tiled_mma_vp,
                    tAtV_cvt,
                    tBsP_desc,
                    tCtO,
                    p_consumer,
                    o_producer,
                    v_ready_consumer,
                    tiles_sk,
                    tiles_dm,
                )

            if iters_s == 1:
                o_producer.commit()
                o_producer.advance()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)

        elif warpgroup_idx in softmax_warpgroup_ids:
            softmax_phase = (
                warpgroup_idx - softmax_warpgroup_ids[0]
            ) % softmax_warpgroups
            softmax_tL_producer_nbar = with_phase(tL_producer_nbar, softmax_phase)
            softmax_tL_consumer_nbar = with_phase(tL_consumer_nbar, softmax_phase)
            sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
            sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
            if softmax_phase == 1:
                s_consumer.advance()
                p_producer.advance()
                sM_release_nbar.arrive()
                if iters_s == 1:
                    softmax_tL_producer_nbar.arrive()
            elif cutlass.const_expr(softmax_warpgroups == 1):
                if iters_s == 1:
                    with_phase(tL_producer_nbar, 1).arrive()

            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            tmem_load_op_s = tcgen05.Ld32x32bOp(tcgen05.Repetition(tmem_repeat_op_s))
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)

            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)

            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)

            tStS = tStS[None, 0, 0, 0, None, None]
            tSsP = tSsP[None, 0, 0, 0, None, None]

            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)

            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)

            tStL = thr_load_l.partition_S(
                tCtL if cutlass.const_expr(softmax_warpgroups == 1) else tCtL_phase
            )
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape

            tStL = (
                tStL[None, 0, 0, 0, None]
                if cutlass.const_expr(softmax_warpgroups == 1)
                else tStL[None, 0, 0, 0]
            )
            tSrL_shape = tSrL_shape[:1]

            masked_iters_s = masked_start_s = masked_coord_s = 0
            is_last_split = kv_split_idx == (tiles_s - 1) % kv_splits
            is_last_phase = softmax_phase == (iters_s - 1) % softmax_warpgroups
            if seqlen % blk_tile_s != 0:
                if is_last_split and is_last_phase:
                    masked_iters_s = 1
                    masked_coord_s = tiles_s - 1

            num_loops = 2

            previous_max_lane = -Float32.inf
            for loop in cutlass.range_constexpr(num_loops):
                is_masked_loop = loop == 1
                for s in cutlass.range(
                    masked_start_s if is_masked_loop else softmax_phase,
                    masked_iters_s if is_masked_loop else (iters_s - masked_iters_s),
                    softmax_warpgroups,
                ):
                    logical_iter_s = s
                    if cutlass.const_expr(is_masked_loop and softmax_warpgroups == 1):
                        logical_iter_s = iters_s - masked_iters_s + s
                    s_token = s_consumer.try_wait()
                    p_token = p_producer.try_acquire()

                    s_handle = s_consumer.wait_and_advance(s_token)
                    tStS_s = tStS[None, None, s_handle.index]
                    tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                    cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                    cute.arch.fence_view_async_tmem_load()
                    s_handle.release()

                    if cutlass.const_expr(is_masked_loop):
                        masked = cute.make_rmem_tensor(
                            (blk_tile_h, blk_tile_p, tiles_sm), acc_dtype
                        )
                        masked.store(tSrS_s.load().reshape(masked.shape))
                        offset_s = (masked_coord_s + s) * blk_tile_s + warpgroup_tidx
                        offset_p = seqlen - prediction + coord_p * blk_tile_p
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for p in cutlass.range_constexpr(blk_tile_p):
                                masked_p = masked[None, p, sm]
                                is_oob = offset_s + sm * mma_tile_m > offset_p + p
                                mask = -Float32.inf if is_oob else Float32(0)
                                masked_p.store(masked_p.load() + mask)
                        scores = masked.load().reshape((blk_tile_n, tiles_sm))
                    else:
                        scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))

                    _nf = blk_tile_n // 2
                    _s3 = scores.reshape((_nf, 2, tiles_sm))
                    _fold = _s3[None, 0, None] + _s3[None, 1, None]
                    _dup = cute.make_rmem_tensor((_nf, 2, tiles_sm), acc_dtype)
                    _dup_b = _dup[None, 0, None]
                    _dup_b.store(_fold)
                    _dup_r = _dup[None, 1, None]
                    _dup_r.store(_fold)
                    scores = _dup.load().reshape((blk_tile_n, tiles_sm))

                    rM = cute.make_rmem_tensor_like(sM)
                    rM.store(
                        scores.reduce(
                            cute.ReductionOp.MAX,
                            init_val=-Float32.inf,
                            reduction_profile=(None, 0),
                        )
                    )

                    rM_lane = Float32(0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rM[n] = warp_fmax(rM[n])

                        if n == lane_idx:
                            rM_lane = rM[n]
                    rM_lane *= scale_s_log2_e

                    if cutlass.const_expr(softmax_warpgroups > 1):
                        sM_acquire_nbar.arrive_and_wait()
                    sM_consumer_nbar.arrive_and_wait()
                    if lane_store_max:
                        if cutlass.const_expr(softmax_warpgroups == 1):
                            if rM_lane - previous_max_lane > Float32(8):
                                smem_fmax(
                                    sM.iterator + sM.layout(lane_idx),
                                    rM_lane,
                                )
                        else:
                            smem_fmax(sM.iterator + sM.layout(lane_idx), rM_lane)

                    sM_producer_nbar.arrive_and_wait()
                    colmax = sM.load()
                    if cutlass.const_expr(softmax_warpgroups == 1):
                        if lane_store_max:
                            previous_max_lane = sM[lane_idx]
                    if cutlass.const_expr(softmax_warpgroups > 1):
                        sM_release_nbar.arrive()

                    p_handle = p_producer.acquire_and_advance(p_token)
                    tSsP_s = tSsP[None, None, p_handle.index]

                    probs = exp2(scale_s_log2_e * scores - colmax)

                    _p_quant = probs.to(mma_dtype).reshape(tSsP_s.shape)
                    tSsP_s.store(_p_quant)
                    cute.arch.fence_view_async_shared()
                    p_handle.commit()

                    tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                    colsum = probs[None, 0]
                    for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                        colsum += probs[None, sm]
                    tSrL.store(colsum.reshape(tSrL.shape))

                    if cutlass.const_expr(softmax_warpgroups == 1):
                        l_phase = logical_iter_s % o_stages
                        iter_tL_consumer_nbar = with_phase(tL_consumer_nbar, l_phase)
                        iter_tL_producer_nbar = with_phase(tL_producer_nbar, l_phase)
                        tStL_iter = tStL[None, l_phase]
                    else:
                        iter_tL_consumer_nbar = softmax_tL_consumer_nbar
                        iter_tL_producer_nbar = softmax_tL_producer_nbar
                        tStL_iter = tStL
                    iter_tL_consumer_nbar.arrive_and_wait()
                    cute.copy(tmem_store_atom_l, tSrL, tStL_iter)
                    cute.arch.fence_view_async_tmem_store()
                    iter_tL_producer_nbar.arrive()

                    if cutlass.const_expr(softmax_warpgroups > 1):
                        s_consumer.advance()
                        p_producer.advance()

        elif warpgroup_idx == correction_warpgroup_id:
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)

            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)

            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)

            tOtL = thr_load_o.partition_S(tCtL)

            tOtO = tOtO[None, 0, 0, 0, None, None]
            tOsO = tOsO[None, 0, 0, 0, None]
            tOtL = tOtL[None, 0, 0, 0, None]

            correction_broadcast_atom = None
            if cutlass.const_expr(self.smem_correction_broadcast):
                correction_broadcast_atom = cute.make_copy_atom(
                    cute.nvgpu.CopyUniversalOp(),
                    acc_dtype,
                    num_bits_per_copy=128,
                )

            def materialize_correction(
                correction_lane,
                should_rescale,
                sL=sL,
                warpgroup_widx=warpgroup_widx,
                lane_idx=lane_idx,
                lane_store_max=lane_store_max,
                correction_broadcast_atom=correction_broadcast_atom,
                blk_tile_n=blk_tile_n,
                smem_correction_broadcast=self.smem_correction_broadcast,
            ):
                correction = cute.make_rmem_tensor((blk_tile_n,), Float32)
                correction.fill(Float32(1))
                if should_rescale:
                    if cutlass.const_expr(smem_correction_broadcast):
                        if lane_store_max:
                            sL[lane_idx, warpgroup_widx] = correction_lane
                        cute.arch.sync_warp()
                        cute.copy(
                            correction_broadcast_atom,
                            sL[None, warpgroup_widx],
                            correction,
                        )
                    else:
                        for n in cutlass.range_constexpr(blk_tile_n):
                            correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                return correction.load()

            def colsum_load(
                phase,
                blk_tile_n=blk_tile_n,
                tOtL=tOtL,
                tOrO_shape=tOsO.shape[:1],
                tmem_load_atom_o=tmem_load_atom_o,
                tL_producer_nbar=tL_producer_nbar,
                tL_consumer_nbar=tL_consumer_nbar,
            ):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)

            tOrO = cute.make_rmem_tensor(tOsO.shape, acc_dtype)
            tOrO.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
            cute.copy(tmem_store_atom_o, tOrO[None, 0], tOtL[None, 1])
            cute.arch.fence_view_async_tmem_store()
            o_init_nbar.arrive_and_wait()

            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()

            colsum_p = cute.make_rmem_tensor((blk_tile_n, o_stages), Float32)
            colsum_0, colsum_1 = colsum_p[None, 0], colsum_p[None, 1]
            colsum_p.fill(Float32(0))

            sM_lane_prev_prev = sM_lane_prev = -Float32.inf
            for s in cutlass.range_constexpr(o_stages):
                sM_lane_prev_prev = sM_lane_prev
                if not (s == 1 and iters_s == 1):
                    sM_producer_nbar.arrive_and_wait()
                    if lane_store_max:
                        sM_lane_prev = sM[lane_idx]
                    sM_consumer_nbar.arrive()

            phase = 0
            for s in cutlass.range(iters_s - o_stages, unroll=o_stages):
                colsum_s = colsum_load(phase)

                sM_producer_nbar.arrive_and_wait()
                if s == iters_s - o_stages - 1:
                    sM_final_nbar.arrive()

                sM_lane = sM_lane_prev_prev
                if lane_store_max:
                    sM_lane = sM[lane_idx]
                sM_consumer_nbar.arrive()

                o_token = o_consumer.try_wait()
                o_handle = o_consumer.wait_and_advance(o_token)

                should_rescale = (
                    cute.arch.vote_ballot_sync(sM_lane_prev_prev != sM_lane) != 0
                )
                correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                correction = materialize_correction(correction_lane, should_rescale)

                correction_o = correction.reshape(tOsO.shape[:1])
                if should_rescale:
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOtO_dm = tOtO[None, dm, phase]
                        tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                        cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                        tOrO_dm.store(correction_o * tOrO_dm.load())
                        cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                    cute.arch.fence_view_async_tmem_store()
                o_handle.release()

                if should_rescale:
                    colsum_s *= correction
                    if phase == 0:
                        colsum_0.store(correction * colsum_0.load() + colsum_s)
                    elif phase == 1:
                        colsum_1.store(correction * colsum_1.load() + colsum_s)
                else:
                    if phase == 0:
                        colsum_0.store(colsum_0.load() + colsum_s)
                    elif phase == 1:
                        colsum_1.store(colsum_1.load() + colsum_s)

                sM_lane_prev_prev, sM_lane_prev = sM_lane_prev, sM_lane
                phase ^= 1

            if iters_s <= o_stages:
                sM_final_nbar.arrive()

            should_rescale_tail = (
                cute.arch.vote_ballot_sync(sM_lane_prev_prev != sM_lane_prev) != 0
            )
            correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
            correction = materialize_correction(correction_lane, should_rescale_tail)

            tail_phase = iters_s % o_stages
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p[None, phase].load()
                    colsum_final += colsum_p[None, phase ^ 1].load()
                    colsum_final += correction * colsum_prev

                    rL_lane = Float32(0.0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                        if n == lane_idx:
                            rL_lane = rL_n

                    if lane_store_max:
                        sL[lane_idx, warpgroup_widx] = rL_lane

            sL_final_nbar.arrive_and_wait()

            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                phase_s = tail_phase ^ s
                tOtO_phase = tOtO[None, None, phase_s]
                tOrO_tail_s = tOrO_tail[None, None, s]
                o_handle = o_consumer.wait_and_advance()
                cute.copy(tmem_load_atom_o, tOtO_phase, tOrO_tail_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()

            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()

            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction.reshape((blk_tile_n, 1)) * output_prev

            if cutlass.const_expr(do_cga_red):
                partial_o.store(output_final.reshape((blk_tile_n,)))
            else:
                if cutlass.const_expr(do_none_red):
                    sM_final_nbar.arrive_and_wait()
                    normalization = sM.load()
                    output_final *= normalization.reshape((blk_tile_n, 1))

                tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
                cute.arch.fence_view_async_shared()
                sO_final_nbar.arrive()

        if cutlass.const_expr(do_cga_red):
            if warpgroup_idx == correction_warpgroup_id:
                cga_stats_ready_nbar.arrive_and_wait()

                dsmem_o_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyDsmemStoreOp(),
                    acc_dtype,
                    num_bits_per_copy=128,
                )
                dsmem_stats_atom = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyDsmemStoreOp(),
                    acc_dtype,
                )

                owner_cta = warpgroup_tidx % kv_splits
                owner_thread = warpgroup_tidx // kv_splits
                owner_threads = warpgroup_threads // kv_splits

                partial_o_offset = (
                    kv_split_idx * owner_threads + owner_thread
                ) * blk_tile_n
                remote_partial_mbar = cute.arch.map_dsmem_ptr(
                    cga_partial_mbar, owner_cta
                )
                remote_o = cute.make_tensor(
                    cute.arch.map_dsmem_ptr(
                        sCgaO.iterator + partial_o_offset, owner_cta
                    ).align(16),
                    cute.make_layout((blk_tile_n,)),
                )
                cute.copy(
                    dsmem_o_atom,
                    partial_o,
                    remote_o,
                    mbar_ptr=remote_partial_mbar,
                )

                if warpgroup_widx == 0:
                    local_stat = cute.make_rmem_tensor(cute.make_layout(1), acc_dtype)
                    for target in cutlass.range(kv_splits):
                        stats_offset = (kv_split_idx * warp_threads + lane_idx) * 2
                        remote_stats_mbar = cute.arch.map_dsmem_ptr(
                            cga_partial_mbar, target
                        )

                        for stat in cutlass.range_constexpr(2):
                            if lane_idx < blk_tile_n:
                                if stat == 0:
                                    local_stat[0] = sM[lane_idx]
                                else:
                                    local_stat[0] = sL[lane_idx, 0]
                            else:
                                local_stat[0] = (
                                    -Float32.inf if stat == 0 else Float32(0)
                                )
                            remote_stat_ptr = cute.arch.map_dsmem_ptr(
                                sCgaStats.iterator + stats_offset + stat,
                                target,
                            )
                            cute.copy(
                                dsmem_stats_atom,
                                local_stat,
                                cute.make_tensor(remote_stat_ptr, cute.make_layout(1)),
                                mbar_ptr=remote_stats_mbar,
                            )

                cute.arch.fence_acq_rel_cta()
                cute.arch.mbarrier_wait(cga_partial_mbar, phase=0)

                if owner_cta == kv_split_idx:
                    final_o = cute.make_rmem_tensor((blk_tile_n,), o_dtype)
                    for head in cutlass.range_constexpr(blk_tile_n):
                        reduced_max = sCgaStats[0, head, 0]
                        reduced_sum = sCgaStats[0, head, 1]
                        reduced_o = Float32(sCgaO[owner_thread * blk_tile_n + head])

                        for source in cutlass.range(1, kv_splits):
                            local_max = sCgaStats[source, head, 0]
                            next_max = cute.arch.fmax(reduced_max, local_max)
                            corr_old = Float32(0)
                            corr_new = Float32(0)
                            if reduced_sum > 0:
                                corr_old = exp2(reduced_max - next_max)
                            if sCgaStats[source, head, 1] > 0:
                                corr_new = exp2(local_max - next_max)
                            local_o = Float32(
                                sCgaO[
                                    (source * owner_threads + owner_thread) * blk_tile_n
                                    + head
                                ]
                            )
                            reduced_sum = (
                                reduced_sum * corr_old
                                + sCgaStats[source, head, 1] * corr_new
                            )
                            reduced_o = reduced_o * corr_old + local_o * corr_new
                            reduced_max = next_max

                        value = Float32(0)
                        if reduced_sum > 0:
                            value = (
                                reduced_o * cute.arch.rcp_approx(reduced_sum) * scale_o
                            )
                        final_o[head] = o_dtype(value)

                    for head in cutlass.range_constexpr(blk_tile_n):
                        group_head = coord_hg * blk_tile_h + head
                        if group_head < mOFinal.shape[1][0]:
                            mOFinal[
                                warpgroup_tidx,
                                (group_head, 0),
                                (coord_hk, coord_b),
                            ] = final_o[head]

        if warp_idx == reduction_warp_id:
            if cutlass.const_expr(do_partial_red):
                self.reduction_epilogue(
                    blk_tile_hp,
                    coord_hp,
                    coord_hb,
                    kv_split_idx,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    mM_partial,
                    mL_partial,
                )
            elif cutlass.const_expr(do_cga_red):
                self.reduction_cga_local_stats(
                    blk_tile_n,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    cga_stats_ready_nbar,
                    sL,
                )
            elif cutlass.const_expr(do_none_red):
                self.reduction_none(
                    lane_store_max,
                    lane_idx,
                    sM_final_nbar,
                    sL_final_nbar,
                    sM,
                    sL,
                    scale_o,
                )
            cute.arch.griddepcontrol_launch_dependents()

        if cutlass.const_expr(do_gmem_fused_red):
            if warpgroup_idx == 0 or warpgroup_idx == correction_warpgroup_id:
                gmem_completion_nbar.arrive_and_wait()

                completion_idx = coord_hb_linear * grid_tiles_hp + coord_hp_linear
                last_reducer_ctas = cutlass.const_expr(
                    1 if self.grouped_head_tile == 8 else 4
                )
                if tidx == 0:
                    cute.arch.fence_acq_rel_gpu()
                    old_count = cute.arch.atomic_add(
                        mCompletion.iterator + completion_idx,
                        Int32(1),
                        sem="acq_rel",
                        scope="gpu",
                    )
                    sGmemLast[0] = Int32(0)
                    if old_count >= kv_splits - Int32(last_reducer_ctas):
                        sGmemLast[0] = kv_splits - old_count

                    if old_count == kv_splits - Int32(1):
                        cute.arch.atomic_exch(
                            mCompletion.iterator + completion_idx,
                            Int32(0),
                            sem="release",
                            scope="gpu",
                        )
                    elif old_count >= kv_splits - Int32(last_reducer_ctas):
                        counter_value = Int32(-1)
                        while counter_value != Int32(0):
                            counter_value = cute.arch.atomic_add(
                                mCompletion.iterator + completion_idx,
                                Int32(0),
                                sem="acquire",
                                scope="gpu",
                            )

                gmem_completion_nbar.arrive_and_wait()

                if warpgroup_idx == correction_warpgroup_id and sGmemLast[0] != Int32(
                    0
                ):
                    self.reduction_gmem_fused_nvfp4(
                        blk_tile_shpd,
                        coord_hg,
                        coord_p,
                        coord_hk,
                        coord_b,
                        heads_k,
                        batches,
                        kv_splits,
                        warpgroup_tidx,
                        sGmemLast[0] - Int32(1),
                        mOPartialGlobal,
                        mOFinal,
                        mOSF,
                        mOGlobalScale,
                        mL_partial,
                        mM_partial,
                        scale_o,
                    )

        return

    @staticmethod
    @cute.jit
    def reduction_gmem_fused_nvfp4(
        blk_tile_shpd: cute.Tile,
        coord_hg: Int32,
        coord_p: Int32,
        coord_hk: Int32,
        coord_b: Int32,
        heads_k: int,
        batches: int,
        kv_splits: Int32,
        warpgroup_tidx: Int32,
        reducer_rank: Int32,
        mO_partial: cute.Tensor,
        mO_final: cute.Tensor,
        mOSF: cute.Tensor,
        mOGlobalScale: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        scale_o: Float32,
    ):
        """Merge split partials and quantize the attention output."""
        blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d = blk_tile_shpd
        assert blk_tile_p == 1 and blk_tile_d % 16 == 0
        assert mO_final.dtype == cutlass.Float4E2M1FN
        assert mOSF is not None and mOGlobalScale is not None

        d_scale_blocks = blk_tile_d // 16
        local_head = warpgroup_tidx // d_scale_blocks
        d_block = warpgroup_tidx % d_scale_blocks
        group_head = coord_hg * blk_tile_h + local_head
        logical_grouped_heads, prediction = mO_final.shape[1]

        partial_vector_elems = 128 // mO_partial.dtype.width
        assert 16 % partial_vector_elems == 0
        partial_o_vector_atom = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            mO_partial.dtype,
            num_bits_per_copy=128,
        )

        reducer_warp = warpgroup_tidx // 32
        if reducer_warp == reducer_rank and group_head < logical_grouped_heads:
            max_final = -Float32.inf
            for source in cutlass.range(kv_splits, unroll=4):
                max_final = cute.arch.fmax(
                    max_final,
                    mM_partial[
                        (group_head, coord_p),
                        (coord_hk, coord_b),
                        source,
                    ],
                )

            final_values = cute.make_rmem_tensor((16,), Float32)
            final_values.fill(Float32(0))
            sum_final = Float32(0)
            if max_final > -Float32.inf:
                for source in cutlass.range(kv_splits, unroll=4):
                    max_partial = mM_partial[
                        (group_head, coord_p),
                        (coord_hk, coord_b),
                        source,
                    ]
                    if max_partial > -Float32.inf:
                        correction = exp2(max_partial - max_final)
                        sum_final += (
                            correction
                            * mL_partial[
                                (group_head, coord_p),
                                (coord_hk, coord_b),
                                source,
                            ]
                        )
                        coord_b_partial = source * batches + coord_b
                        partial_o_head = mO_partial[
                            None,
                            (group_head, coord_p),
                            (coord_hk, coord_b_partial),
                        ]
                        for vector_idx in cutlass.range_constexpr(
                            16 // partial_vector_elems
                        ):
                            partial_o_vector = cute.local_tile(
                                partial_o_head,
                                (partial_vector_elems,),
                                (d_block * (16 // partial_vector_elems) + vector_idx,),
                            )
                            partial_values = cute.make_rmem_tensor(
                                (partial_vector_elems,), mO_partial.dtype
                            )
                            cute.copy(
                                partial_o_vector_atom,
                                partial_o_vector,
                                partial_values,
                            )
                            partial_values_f32 = partial_values.load().to(Float32)
                            for value_in_vector in cutlass.range_constexpr(
                                partial_vector_elems
                            ):
                                value_idx = (
                                    vector_idx * partial_vector_elems + value_in_vector
                                )
                                final_values[value_idx] += (
                                    correction * partial_values_f32[value_in_vector]
                                )

            if sum_final != Float32(0):
                final_values.store(
                    final_values.load() * (cute.arch.rcp_approx(sum_final) * scale_o)
                )

            packed_values, sf_value = quantize_nvfp4_block16(
                final_values.load(), Float32(mOGlobalScale[0])
            )

            packed_fp4 = cute.make_rmem_tensor((16,), cutlass.Float4E2M1FN)
            packed_fp4.store(packed_values)
            packed_bytes = cute.recast_tensor(packed_fp4, cutlass.Uint8)
            mO_bytes = cute.recast_tensor(mO_final, cutlass.Uint8)
            for byte_idx in cutlass.range_constexpr(8):
                mO_bytes[
                    d_block * 8 + byte_idx,
                    (group_head, coord_p),
                    (coord_hk, coord_b),
                ] = packed_bytes[byte_idx]

            global_head = coord_hk * logical_grouped_heads + group_head
            k_idx = global_head * d_scale_blocks + d_block
            hidden = heads_k * logical_grouped_heads * blk_tile_d
            row_idx = coord_b * prediction + coord_p
            sf_offset = output_scale_offset(row_idx, k_idx, hidden)
            mOSF[sf_offset] = sf_value

    @staticmethod
    @cute.jit
    def reduction_cga_local_stats(
        blk_tile_n: int,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        cga_stats_ready_nbar: nbar,
        sL: cute.Tensor,
    ):
        """Collapse one CTA's four softmax-warp sums for CGA exchange."""
        sM_final_nbar.arrive_and_wait()
        sL_final_nbar.arrive_and_wait()
        if lane_idx < blk_tile_n:
            local_sum = sL[lane_idx, 0]
            for warp in cutlass.range_constexpr(1, warpgroup_warps):
                local_sum += sL[lane_idx, warp]
            sL[lane_idx, 0] = local_sum
        cute.arch.fence_view_async_shared()
        cga_stats_ready_nbar.arrive()

    @staticmethod
    @cute.jit
    def reduction_none(
        lane_store_max: bool,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        scale_o: Float32,
    ):
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()

    @staticmethod
    @cute.jit
    def reduction_epilogue(
        blk_tile_hp: Tuple[int, int],
        coord_hp: Tuple[Int32, Int32],
        coord_hb: Tuple[Int32, Int32],
        kv_split_idx: Int32,
        lane_idx: Int32,
        sM_final_nbar: nbar,
        sL_final_nbar: nbar,
        sM: cute.Tensor,
        sL: cute.Tensor,
        mM_partial: cute.Tensor,
        mL_partial: cute.Tensor,
    ):
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)

        blk_tile_h, blk_tile_p = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n

        grouped_heads, prediction = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        idx_hg, idx_p = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction

        cute.arch.fence_acq_rel_cta()
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            sM_lane = sM[lane_idx]
            gM_partial[lane_idx] = sM_lane

        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            sL_lane = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]
            gL_partial[lane_idx] = sL_lane

    @staticmethod
    @cute.jit
    def launch_reduction(
        d_per_blk: int,
        o_bshd: cute.Tensor,
        o_partial_bshd: cute.Tensor,
        l_partial_bsh: cute.Tensor,
        m_partial_bsh: cute.Tensor,
        scale_o: Float32,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        padded_grouped_heads: int,
        logical_grouped_heads: int,
        stream: cuda.CUstream,
        enable_pdl: bool = True,
    ):
        splits, b, s_q, h_q_partial, d = o_partial_bshd.shape
        h_q = o_bshd.shape[2]
        quantize_o = cutlass.const_expr(o_bshd.dtype == cutlass.Float4E2M1FN)
        if cutlass.const_expr(quantize_o):
            assert o_block_scale is not None and o_global_scale is not None
            assert d_per_blk % 16 == 0
            assert h_q_partial * logical_grouped_heads == (h_q * padded_grouped_heads)
        else:
            assert o_block_scale is None and o_global_scale is None

        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)

        o_dhsb = reverse(o_bshd)
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)

        d_per_thr = 16 if quantize_o else 32 // o_bshd.dtype.width
        thr_per_blk = d_per_blk // d_per_thr
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = splits * 2 * Float32.width // 8

        GroupedQueryAttentionDecodePagedFp4.reduction_kernel(
            (thr_per_blk, d_per_thr, d_per_blk),
            o_dhsb,
            o_partial_dhsb,
            l_partial_hsb,
            m_partial_hsb,
            scale_o,
            o_block_scale,
            o_global_scale,
            padded_grouped_heads,
            logical_grouped_heads,
        ).launch(
            grid=[d_blks, h_q * s_q, b],
            block=[thr_per_blk, 1, 1],
            cluster=[1, 1, 1],
            stream=stream,
            smem=smem_bytes,
            min_blocks_per_mp=1,
            use_pdl=enable_pdl,
        )

    @staticmethod
    @cute.kernel
    def reduction_kernel(
        tile_d: cute.Tile,
        o_dhsb: cute.Tensor,
        o_partial_dhsb: cute.Tensor,
        l_partial_hsb: cute.Tensor,
        m_partial_hsb: cute.Tensor,
        scale_o: Float32,
        o_block_scale: Optional[cute.Tensor],
        o_global_scale: Optional[cute.Tensor],
        padded_grouped_heads: int,
        logical_grouped_heads: int,
    ):
        thr_per_blk, d_per_thr, d_per_blk = tile_d
        d, h_q_partial, s_q, b, splits = o_partial_dhsb.shape
        _, h_q, _, _ = o_dhsb.shape
        d_blk_idx, coord_hs, coord_b = cute.arch.block_idx()
        coord_h, coord_s = cute.idx2crd(coord_hs, (h_q, s_q))
        tidx, _, _ = cute.arch.thread_idx()

        coord_hk = coord_h // logical_grouped_heads
        coord_hg = coord_h % logical_grouped_heads
        coord_h_partial = coord_hk * padded_grouped_heads + coord_hg

        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + tidx * d_per_thr < d

        coord_o = (d_blk_idx, coord_h, coord_s, coord_b)
        coord_partial = (
            d_blk_idx,
            coord_h_partial,
            coord_s,
            coord_b,
            None,
        )
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o)
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_partial)

        stats_coord = (coord_h_partial, coord_s, coord_b, None)
        gM_partial = cute.local_tile(m_partial_hsb, (1,), stats_coord)
        gL_partial = cute.local_tile(l_partial_hsb, (1,), stats_coord)
        gM_partial_0 = gM_partial[None, tidx]
        gL_partial_0 = gL_partial[None, tidx]

        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + splits, partial_layout)
        sL_partial_0 = sL_partial[None, tidx]
        sM_partial_0 = sM_partial[None, tidx]

        cpasync_atom = cute.make_copy_atom(
            cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32
        )

        partial_o_dtype = o_partial_dhsb.dtype
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), partial_o_dtype)
        tv_layout = cute.make_ordered_layout((thr_per_blk, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(tidx)

        tCgO_partial = thr_copy.partition_S(gO_partial)
        tCgO_partial = tCgO_partial[None, 0, None]
        tCgO = thr_copy.partition_D(gO)
        tCgO = tCgO[None, 0]
        tCrO_final = cute.zeros_like(tCgO, Float32)

        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()

        if tidx < splits:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)

        for split_idx in cutlass.range(thr_per_blk + tidx, splits, thr_per_blk):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)

            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)

        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

        max_final = -Float32.inf
        for split_idx in cutlass.range(splits, unroll=8):
            max_final = cute.arch.fmax(max_final, sM_partial[0, split_idx])
        sum_final = Float32(0)
        if max_final > -Float32.inf and not_oob_d:
            for split_idx in cutlass.range(splits, unroll=8):
                max_partial = sM_partial[0, split_idx]
                if max_partial > -Float32.inf:
                    correction = exp2(max_partial - max_final)
                    sum_final += correction * sL_partial[0, split_idx]
                    partial_o = tCgO_partial[None, split_idx].load()
                    tCrO_final += correction * partial_o.to(Float32)
            tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o

        cute.arch.griddepcontrol_launch_dependents()

        quantize_o = cutlass.const_expr(o_dhsb.dtype == cutlass.Float4E2M1FN)
        if not_oob_d:
            if cutlass.const_expr(quantize_o):
                assert o_block_scale is not None and o_global_scale is not None
                packed_values, sf_value = quantize_nvfp4_block16(
                    tCrO_final, Float32(o_global_scale[0])
                )
                tCgO.store(packed_values)

                d_block = (d_blk_idx * d_per_blk + tidx * d_per_thr) // 16
                d_scale_blocks = d // 16
                k_idx = coord_h * d_scale_blocks + d_block
                hidden = h_q * d
                row_idx = coord_b * s_q + coord_s
                sf_offset = output_scale_offset(row_idx, k_idx, hidden)
                o_block_scale[sf_offset] = sf_value
            else:
                tCgO.store(tCrO_final.to(o_dhsb.dtype))

        return
