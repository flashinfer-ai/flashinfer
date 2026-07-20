# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/contiguous/forward.py @ 6627d342 (2026-07-19) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# SM120 (Blackwell GeForce / DGX Spark) forward pass.
#
# This is a real SM120 forward kernel skeleton:
# - SM90-style outer structure (tile scheduler + TMA producer/consumer split)
# - SM80-style warp MMA math core
# - 160-thread CTA: 4 compute warps + 1 TMA producer warp
#
# The initial slice is intentionally narrow:
# - fixed-length and packed varlen Q/K/V
# - no block sparsity

import math
from functools import partial
from typing import Callable, Optional, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.cute.nvgpu import cpasync, warp, warpgroup
from cutlass.utils import LayoutEnum
import cutlass.utils.hopper_helpers as sm90_utils_basic
import cutlass.utils as utils_basic

from flashinfer.experimental.sm12x.attention._shared.cute import copy as cute_copy
from flashinfer.experimental.sm12x.attention._shared.cute import ops as cute_ops
from flashinfer.experimental.sm12x.attention._shared.cute import (
    pipeline as cute_pipeline,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous import layout_utils
from flashinfer.experimental.sm12x.attention._shared.contiguous.cute_dsl_utils import (
    assume_tensor_aligned,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.mask import (
    AttentionMask,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.softmax import Softmax
from flashinfer.experimental.sm12x.attention._shared.contiguous.seqlen_info import (
    SeqlenInfoQK,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.block_info import (
    BlockInfo,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.pack_gqa import (
    PackGQA,
    pack_gqa_layout,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.named_barrier import (
    NamedBarrierFwd,
)
from flashinfer.experimental.sm12x.attention._shared.contiguous.tile_scheduler import (
    SingleTileScheduler,
    SingleTileVarlenScheduler,
    TileSchedulerArguments,
)


def _make_contiguous_shared_storage_cls(
    dtype,
    *,
    q_numel: int,
    k_numel: int,
    v_numel: int,
    num_stages: int,
    buffer_align_bytes: int = 1024,
):
    """Build the allocation type used for both fit policy and the kernel."""
    sQ_struct, sK_struct, sV_struct = [
        cute.struct.Align[
            cute.struct.MemRange[dtype, int(numel)],
            buffer_align_bytes,
        ]
        for numel in (q_numel, k_numel, v_numel)
    ]
    mbar_ptr_Q_struct = cute.struct.MemRange[cutlass.Int64, 1]
    mbar_ptr_K_struct = cute.struct.MemRange[cutlass.Int64, num_stages * 2]
    mbar_ptr_V_struct = cute.struct.MemRange[cutlass.Int64, num_stages * 2]

    @cute.struct
    class SharedStorageQKV:
        mbar_ptr: mbar_ptr_Q_struct
        mbar_ptr_K: mbar_ptr_K_struct
        mbar_ptr_V: mbar_ptr_V_struct
        sV: sV_struct
        sQ: sQ_struct
        sK: sK_struct

    return SharedStorageQKV


@cute.jit
def warp_mma_gemm(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsA: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_A: cute.TiledCopy,
    smem_thr_copy_B: cute.TiledCopy,
    A_in_regs: cutlass.Constexpr = False,
    B_in_regs: cutlass.Constexpr = False,
):
    tCrA_copy_view = smem_thr_copy_A.retile(tCrA)
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    if const_expr(not A_in_regs):
        cute.copy(smem_thr_copy_A, tCsA[None, None, 0], tCrA_copy_view[None, None, 0])
    if const_expr(not B_in_regs):
        cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCsA.shape[2])):
        if k < cute.size(tCsA.shape[2]) - 1:
            if const_expr(not A_in_regs):
                cute.copy(
                    smem_thr_copy_A,
                    tCsA[None, None, k + 1],
                    tCrA_copy_view[None, None, k + 1],
                )
            if const_expr(not B_in_regs):
                cute.copy(
                    smem_thr_copy_B,
                    tCsB[None, None, k + 1],
                    tCrB_copy_view[None, None, k + 1],
                )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


@cute.jit
def warp_mma_gemm_rs(
    tiled_mma: cute.TiledMma,
    acc: cute.Tensor,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    tCsB: cute.Tensor,
    smem_thr_copy_B: cute.TiledCopy,
):
    tCrB_copy_view = smem_thr_copy_B.retile(tCrB)
    cute.copy(smem_thr_copy_B, tCsB[None, None, 0], tCrB_copy_view[None, None, 0])
    for k in cutlass.range_constexpr(cute.size(tCrA.shape[2])):
        if const_expr(k < cute.size(tCrA.shape[2]) - 1):
            cute.copy(
                smem_thr_copy_B,
                tCsB[None, None, k + 1],
                tCrB_copy_view[None, None, k + 1],
            )
        cute.gemm(tiled_mma, acc, tCrA[None, None, k], tCrB[None, None, k], acc)


class ContiguousAttentionForwardKernel:
    arch = 120

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 160,
        num_compute_warps: int = 4,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        mma_pv_is_rs: bool = True,
    ):
        self.dtype = dtype
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim if head_dim_v is None else head_dim_v
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        assert self.score_mod is None, (
            "score_mod is not part of the initial sm12x transplant"
        )
        assert self.mask_mod is None, (
            "mask_mod is not part of the initial sm12x transplant"
        )
        self.mma_pv_is_rs = mma_pv_is_rs
        assert self.mma_pv_is_rs, (
            "SM120 rewrite currently only supports register-sourced PV"
        )
        self.buffer_align_bytes = 1024
        self.num_compute_warps = num_compute_warps
        assert self.num_compute_warps >= 1
        self.num_threads_per_warp = 32
        self.producer_warp_idx = self.num_compute_warps
        self.use_tma_KV = True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        learnable_sink_type: Type[cutlass.Numeric] | None,
    ):
        if const_expr(not (mQ_type == mO_type)):
            raise TypeError("Q and O tensors must have the same data type")
        if const_expr(not (mK_type == mV_type)):
            raise TypeError("K and V tensors must have the same data type")
        if const_expr(mQ_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("Q/O tensors must be Float16 or BFloat16")
        if const_expr(mK_type not in [cutlass.Float16, cutlass.BFloat16]):
            raise TypeError("K/V tensors must be Float16 or BFloat16")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(learnable_sink_type not in [None, Float32]):
            raise TypeError("learnable sink tensor must be Float32")
        assert mQ_type == self.dtype
        assert mK_type == self.dtype

    def _setup_attributes(self):
        (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        ) = self._get_smem_layout_atom()
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom, (self.tile_m, self.tile_hdim), (0, 1)
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.tile_n, self.tile_hdim, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_n, self.tile_hdimv, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom, (self.tile_m, self.tile_hdimv), (0, 1)
        )
        self.sP_layout = (
            cute.tile_to_shape(sP_layout_atom, (self.tile_m, self.tile_n), (0, 1))
            if const_expr(sP_layout_atom is not None)
            else None
        )

        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0
        assert self.num_producer_threads % tQK_shape_dim_1 == 0
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        assert self.tile_m % tO_layout.shape[0] == 0
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = vQKV_layout
        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(
            atom_async_copy, tQ_layout, vQKV_layout
        )
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(
            atom_async_copy, tK_layout, vQKV_layout
        )
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(
            atom_async_copy, tV_layout, vQKV_layout
        )
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(
            atom_universal_copy, tO_layout, vO_layout
        )

    @staticmethod
    def shared_storage_bytes(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
    ) -> int:
        tile_hdim = int(math.ceil(head_dim / 16) * 16)
        tile_hdimv = int(math.ceil(head_dim_v / 16) * 16)
        SharedStorage = _make_contiguous_shared_storage_cls(
            dtype,
            q_numel=tile_m * tile_hdim,
            k_numel=tile_n * tile_hdim * num_stages,
            v_numel=tile_n * tile_hdimv * num_stages,
            num_stages=num_stages,
        )
        return int(SharedStorage.size_in_bytes())

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        num_compute_warps=4,
    ) -> bool:
        del is_causal
        if dtype not in [cutlass.Float16, cutlass.BFloat16]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if num_compute_warps < 1:
            return False
        if tile_m % (num_compute_warps * 16) != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        if num_threads != (num_compute_warps + 1) * 32:
            return False
        smem_usage = ContiguousAttentionForwardKernel.shared_storage_bytes(
            dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            num_stages,
        )
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        return True

    def _get_smem_layout_atom(self):
        sQ_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdim
            ),
            self.dtype,
        )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = warpgroup.make_smem_layout_atom(
            sm90_utils_basic.get_smem_layout_atom(
                LayoutEnum.ROW_MAJOR, self.dtype, self.tile_hdimv
            ),
            self.dtype,
        )
        sO_layout_atom = sV_layout_atom
        sP_layout_atom = None
        return (
            sQ_layout_atom,
            sK_layout_atom,
            sV_layout_atom,
            sO_layout_atom,
            sP_layout_atom,
        )

    def _get_tiled_mma(self):
        tiled_mma_qk = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            warp.MmaF16BF16Op(self.dtype, Float32, (16, 8, 16)),
            (self.num_compute_warps, 1, 1),
            permutation_mnk=(self.num_compute_warps * 16, 16, 16),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        return _make_contiguous_shared_storage_cls(
            self.dtype,
            q_numel=cute.cosize(self.sQ_layout),
            k_numel=cute.cosize(self.sK_layout),
            v_numel=cute.cosize(self.sV_layout),
            num_stages=self.num_stages,
            buffer_align_bytes=self.buffer_align_bytes,
        )

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        del tma_atom_O
        rO = cute.make_fragment_like(acc_O, self.dtype)
        rO.store(acc_O.load().to(self.dtype))
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        smem_copy_atom_O = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dtype,
            num_bits_per_copy=2 * self.dtype.width,
        )
        smem_thr_copy_O = cute.make_tiled_copy_C(smem_copy_atom_O, tiled_mma).get_slice(
            tidx
        )
        taccOrO = smem_thr_copy_O.retile(rO)
        taccOsO = smem_thr_copy_O.partition_D(sO)
        cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )
        if const_expr(mLSE is not None):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
            else:
                offset = (
                    seqlen.offset_q
                    if const_expr(not self.pack_gqa)
                    else (0, seqlen.offset_q)
                )
                mLSE_cur = cute.domain_offset((offset,), mLSE[None, head_idx])
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(
                    thr_mma.partition_C(gLSE_expanded)
                )
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(
                    thr_mma.get_slice(0).partition_C(cO)
                )
                if taccOcO[0][1] == 0:
                    for m in cutlass.range_constexpr(cute.size(taccOgLSE.shape[1])):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(
                    mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q
                )

        if const_expr(not seqlen.has_cu_seqlens_q):
            mO_cur = mO[None, None, head_idx, batch_idx]
        else:
            offset = (
                seqlen.offset_q
                if const_expr(not self.pack_gqa)
                else (0, seqlen.offset_q)
            )
            mO_cur = cute.domain_offset((offset, 0), mO[None, None, head_idx])
        cute.arch.barrier(
            barrier_id=int(NamedBarrierFwd.Epilogue),
            number_of_threads=self.num_epilogue_threads,
        )
        gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
        tOsO = gmem_thr_copy_O.partition_S(sO)
        tOrO = cute.make_fragment_like(tOsO, self.dtype)
        cute.autovec_copy(tOsO, tOrO)
        if const_expr(not self.pack_gqa):
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            tOgO = gmem_thr_copy_O.partition_D(gO)
            tOcO = gmem_thr_copy_O.partition_S(cO)
            t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
            tOpO = cute_ops.predicate_k(tOcO, limit=mO.shape[1])
            for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                if (
                    t0OcO[0, rest_m, 0][0]
                    < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                ):
                    cute.copy(
                        gmem_tiled_copy_O,
                        tOrO[None, rest_m, None],
                        tOgO[None, rest_m, None],
                        pred=tOpO[None, rest_m, None]
                        if const_expr(self.check_hdim_v_oob)
                        else None,
                    )
        else:
            pack_gqa.store_O(
                mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q
            )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        has_attention_sink_bias: cutlass.Constexpr = False,
        blocksparse_tensors=None,
        aux_tensors=None,
        logical_num_batch_static: cutlass.Constexpr = 1,
        logical_seqlen_q_static: cutlass.Constexpr = 0,
        logical_seqlen_k_static: cutlass.Constexpr = 0,
        stream: cuda.CUstream = None,
    ):
        assert blocksparse_tensors is None
        self._check_type(
            *(
                t.element_type if t is not None else None
                for t in (
                    mQ,
                    mK,
                    mV,
                    mO,
                    mLSE,
                    mCuSeqlensQ,
                    mCuSeqlensK,
                    learnable_sink,
                )
            )
        )

        self.num_threads = (self.num_compute_warps + 1) * self.num_threads_per_warp
        self.num_mma_threads = self.num_compute_warps * self.num_threads_per_warp
        self.num_producer_threads = self.num_threads_per_warp
        self.num_Q_load_threads = self.num_mma_threads
        self.num_epilogue_threads = self.num_mma_threads
        self.num_mma_regs = 248
        self.num_producer_regs = 80
        self.use_tma_Q = True
        self.use_tma_KV = True
        self.use_tma_O = False

        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = (
            [1, 3, 2, 0] if const_expr(cute.rank(mQ) == 4) else [0, 2, 1]
        )
        O_layout_transpose = (
            [1, 3, 2, 0] if const_expr(cute.rank(mO) == 4) else [0, 2, 1]
        )
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        KV_layout_transpose = (
            [1, 3, 2, 0] if const_expr(cute.rank(mK) == 4) else [0, 2, 1]
        )
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        if const_expr(mLSE is not None):
            LSE_layout_transpose = (
                [2, 1, 0] if const_expr(cute.rank(mLSE) == 3) else [1, 0]
            )
            mLSE = cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )

        q_heads_unpacked = mQ.shape[2]
        kv_heads = mK.shape[2]
        logical_num_head = kv_heads if const_expr(self.pack_gqa) else q_heads_unpacked
        logical_q_rows_static = logical_seqlen_q_static * (
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1
        )
        logical_num_block = cute.ceil_div(logical_q_rows_static, self.tile_m)
        logical_total_q = (
            mQ.shape[0] * (self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1)
            if const_expr(mCuSeqlensQ is not None)
            else logical_q_rows_static * logical_num_batch_static
        )

        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        self.launch_smem_bytes = int(SharedStorage.size_in_bytes())
        expected_launch_smem_bytes = self.shared_storage_bytes(
            self.dtype,
            self.tile_hdim,
            self.tile_hdimv,
            self.tile_m,
            self.tile_n,
            self.num_stages,
        )
        if const_expr(self.launch_smem_bytes != expected_launch_smem_bytes):
            raise AssertionError(
                "contiguous typed shared-memory policy/allocation mismatch: "
                f"policy={expected_launch_smem_bytes}, allocation={self.launch_smem_bytes}"
            )
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if const_expr(self.launch_smem_bytes > smem_capacity):
            raise ValueError(
                f"contiguous typed shared storage requires {self.launch_smem_bytes} B, "
                f"but SM120 permits {smem_capacity} B per CTA"
            )

        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(
                    mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1
                )

        gmem_tiled_copy_Q = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_KV = cpasync.CopyBulkTensorTileG2SOp()
        gmem_tiled_copy_O = cpasync.CopyBulkTensorTileS2GOp()
        sK_tma_layout = cute.select(self.sK_layout, mode=[0, 1])
        sV_tma_layout = cute.select(self.sV_layout, mode=[0, 1])
        self.tma_copy_bytes = {
            "Q": cute.size_in_bytes(mQ.element_type, self.sQ_layout),
            "K": cute.size_in_bytes(mK.element_type, sK_tma_layout),
            "V": cute.size_in_bytes(mV.element_type, sV_tma_layout),
        }

        tma_atom_Q, tma_tensor_Q = cpasync.make_tiled_tma_atom(
            gmem_tiled_copy_Q,
            mQ,
            self.sQ_layout,
            (self.tile_m, self.tile_hdim),
        )
        TileScheduler = (
            SingleTileVarlenScheduler
            if const_expr(mCuSeqlensQ is not None)
            else SingleTileScheduler
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=logical_num_block,
            num_head=logical_num_head,
            num_batch=(
                logical_num_batch_static
                if const_expr(mCuSeqlensQ is None)
                else mCuSeqlensQ.shape[0] - 1
            ),
            seqlen_k=logical_seqlen_k_static,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=logical_total_q,
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            element_size=self.dtype.width // 8,
            lpt=self.is_causal or self.is_local,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        softmax_scale_log2, softmax_scale = cute_ops.compute_softmax_scale_log2(
            softmax_scale
        )
        tma_atom_O, tma_tensor_O = None, None
        if const_expr(self.use_tma_O):
            tma_atom_O, tma_tensor_O = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_O,
                mO,
                self.sO_layout,
                (self.tile_m, self.tile_hdimv),
            )
        tma_atom_K, tma_tensor_K = (None, None)
        tma_atom_V, tma_tensor_V = (None, None)
        if const_expr(self.use_tma_KV):
            tma_atom_K, tma_tensor_K = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mK,
                sK_tma_layout,
                (self.tile_n, self.tile_hdim),
                1,
            )
            tma_atom_V, tma_tensor_V = cpasync.make_tiled_tma_atom(
                gmem_tiled_copy_KV,
                mV,
                sV_tma_layout,
                (self.tile_n, self.tile_hdimv),
                1,
            )
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K if const_expr(self.use_tma_KV) else mK,
            tma_tensor_V if const_expr(self.use_tma_KV) else mV,
            tma_tensor_O if const_expr(self.use_tma_O) else mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            learnable_sink,
            has_attention_sink_bias,
            tma_atom_Q,
            tma_atom_K if const_expr(self.use_tma_KV) else None,
            tma_atom_V if const_expr(self.use_tma_KV) else None,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            TileScheduler,
            SharedStorage,
            logical_seqlen_q_static,
            logical_seqlen_k_static,
            aux_tensors,
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mAttentionSinkBias: cute.Tensor,
        has_attention_sink_bias: cutlass.Constexpr,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: Optional[cute.CopyAtom],
        tma_atom_V: Optional[cute.CopyAtom],
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr,
        SharedStorage: cutlass.Constexpr,
        logical_seqlen_q_static: cutlass.Constexpr,
        logical_seqlen_k_static: cutlass.Constexpr,
        aux_tensors=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_Q)
            cpasync.prefetch_descriptor(tma_atom_K)
            cpasync.prefetch_descriptor(tma_atom_V)
            if const_expr(tma_atom_O is not None):
                cpasync.prefetch_descriptor(tma_atom_O)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        mbar_ptr_Q = storage.mbar_ptr.data_ptr()
        if warp_idx == 0:
            cute.arch.mbarrier_init(mbar_ptr_Q, 1)
        cute.arch.sync_threads()

        pipeline_kv_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, self.num_compute_warps
        )
        pipeline_kv_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread
        )
        pipeline_k = cute_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_K.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["K"],
            defer_sync=True,
        )
        pipeline_v = cute_pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mbar_ptr_V.data_ptr(),
            num_stages=self.num_stages,
            producer_group=pipeline_kv_producer_group,
            consumer_group=pipeline_kv_consumer_group,
            tx_count=self.tma_copy_bytes["V"],
            defer_sync=False,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sVt = layout_utils.transpose_view(sV)
        sO = storage.sQ.get_tensor(
            sO_layout.outer, swizzle=sO_layout.inner, dtype=self.dtype
        )

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead
            if const_expr(self.pack_gqa)
            else 1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=logical_seqlen_q_static,
            seqlen_k_static=logical_seqlen_k_static,
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
        )
        TileSchedulerCls = partial(TileScheduler.create, tile_sched_params)

        if warp_idx == self.producer_warp_idx:
            cute.arch.setmaxregister_decrease(self.num_producer_regs)
            self.load(
                mQ,
                mK,
                mV,
                sQ,
                sK,
                sV,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
            )
        elif warp_idx < self.num_compute_warps:
            cute.arch.setmaxregister_increase(self.num_mma_regs)
            tidx = cute.arch.thread_idx()[0]
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                mQ,
                mO,
                mLSE,
                sQ,
                sK,
                sV,
                sVt,
                sO,
                pipeline_k,
                pipeline_v,
                mbar_ptr_Q,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_O,
                tma_atom_O,
                tidx,
                softmax_scale_log2,
                softmax_scale,
                mAttentionSinkBias,
                has_attention_sink_bias,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                aux_tensors,
            )

    @cute.jit
    def load(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_k: cute_pipeline.PipelineTmaAsync,
        pipeline_v: cute_pipeline.PipelineTmaAsync,
        mbar_ptr_Q: cutlass.Pointer,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
    ):
        kv_producer_state = cute_pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.num_stages
        )
        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            del split_idx
            seqlen = SeqlenInfoCls(batch_idx)
            if const_expr(cute.rank(mQ) == 4):
                mQ_batch = seqlen.offset_batch_Q(mQ, batch_idx, dim=3)
            elif const_expr(seqlen.has_cu_seqlens_q):
                mQ_batch = seqlen.offset_batch_Q(mQ, batch_idx, dim=2)
            else:
                mQ_batch = mQ
            mQ_cur = mQ_batch[None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, (self.tile_m, self.tile_hdim), (m_block, 0))
            load_Q, _, _ = cute_copy.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), gQ, sQ, single_stage=True
            )
            head_idx_kv = (
                head_idx
                if const_expr(self.pack_gqa)
                else head_idx // self.qhead_per_kvhead
            )
            if const_expr(cute.rank(mK) == 4):
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=3)[
                    None, None, head_idx_kv
                ]
            elif const_expr(seqlen.has_cu_seqlens_k):
                mK_cur = seqlen.offset_batch_K(mK, batch_idx, dim=2)[
                    None, None, head_idx_kv
                ]
                mV_cur = seqlen.offset_batch_K(mV, batch_idx, dim=2)[
                    None, None, head_idx_kv
                ]
            else:
                mK_cur = mK[None, None, head_idx_kv]
                mV_cur = mV[None, None, head_idx_kv]
            gK = cute.local_tile(mK_cur, (self.tile_n, self.tile_hdim), (None, 0))
            gV = cute.local_tile(mV_cur, (self.tile_n, self.tile_hdimv), (None, 0))
            load_K, _, _ = cute_copy.tma_get_copy_fn(
                tma_atom_K,
                0,
                cute.make_layout(1),
                gK,
                sK,
            )
            load_K = cute_copy.tma_producer_copy_fn(load_K, pipeline_k)
            load_V, _, _ = cute_copy.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                gV,
                sV,
            )
            load_V = cute_copy.tma_producer_copy_fn(load_V, pipeline_v)

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            with cute.arch.elect_one():
                cute.arch.mbarrier_arrive_and_expect_tx(
                    mbar_ptr_Q, self.tma_copy_bytes["Q"]
                )
            load_Q(tma_bar_ptr=mbar_ptr_Q)
            for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                n_block = n_block_max - 1 - n_tile
                pipeline_k.producer_acquire(kv_producer_state)
                load_K(src_idx=n_block, producer_state=kv_producer_state)
                pipeline_v.producer_acquire(kv_producer_state)
                load_V(src_idx=n_block, producer_state=kv_producer_state)
                kv_producer_state.advance()

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

        pipeline_k.producer_tail(kv_producer_state.clone())
        pipeline_v.producer_tail(kv_producer_state.clone())

    @cute.jit
    def mma_one_n_block(
        self,
        n_block: Int32,
        kv_consumer_state,
        thr_mma_qk: cute.TiledMma,
        thr_mma_pv: cute.TiledMma,
        tSrQ: cute.Tensor,
        tSrK: cute.Tensor,
        tOrVt: cute.Tensor,
        acc_O: cute.Tensor,
        smem_thr_copy_Q: cute.TiledCopy,
        smem_thr_copy_K: cute.TiledCopy,
        smem_thr_copy_V: cute.TiledCopy,
        tSsQ: cute.Tensor,
        tSsK: cute.Tensor,
        tOsVt: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        softmax: Softmax,
        seqlen: SeqlenInfoQK,
        batch_idx: Int32,
        head_idx: Int32,
        m_block: Int32,
        mask_fn: Callable,
        aux_tensors=None,
        fastdiv_mods=None,
        is_first_n_block: cutlass.Constexpr = False,
    ):
        pipeline_k.consumer_wait(
            kv_consumer_state, pipeline_k.consumer_try_wait(kv_consumer_state)
        )
        acc_shape_S = thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        warp_mma_gemm(
            thr_mma_qk,
            acc_S,
            tSrQ,
            tSrK,
            tSsQ,
            tSsK[
                None,
                None,
                None,
                kv_consumer_state.index if const_expr(self.num_stages > 1) else 0,
            ],
            smem_thr_copy_Q,
            smem_thr_copy_K,
        )
        pipeline_k.consumer_release(kv_consumer_state)

        mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(
            acc_S, is_first=is_first_n_block, check_inf=True
        )
        softmax.rescale_O(acc_O, row_scale)

        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        tOrP = layout_utils.reshape_acc_to_frgA(rP)

        pipeline_v.consumer_wait(
            kv_consumer_state, pipeline_v.consumer_try_wait(kv_consumer_state)
        )
        warp_mma_gemm_rs(
            thr_mma_pv,
            acc_O,
            tOrP,
            tOrVt,
            tOsVt[
                None,
                None,
                None,
                kv_consumer_state.index if const_expr(self.num_stages > 1) else 0,
            ],
            smem_thr_copy_V,
        )
        pipeline_v.consumer_release(kv_consumer_state)
        kv_consumer_state.advance()
        return kv_consumer_state

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        mQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sVt: cute.Tensor,
        sO: cute.Tensor,
        pipeline_k: cutlass.pipeline.PipelineAsync,
        pipeline_v: cutlass.pipeline.PipelineAsync,
        mbar_ptr_Q: cutlass.Pointer,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: cute.CopyAtom,
        tidx: Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        mAttentionSinkBias: cute.Tensor,
        has_attention_sink_bias: cutlass.Constexpr,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        aux_tensors=None,
    ):
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
        tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
        tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        taccOcO = thr_mma_pv.partition_C(cO)
        taccOcO_mn = layout_utils.reshape_acc_to_mn(taccOcO)

        smem_copy_atom_QK = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
            self.dtype,
        )
        smem_copy_atom_V = cute.make_copy_atom(
            warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
            self.dtype,
        )
        smem_thr_copy_Q = cute_ops.make_tiled_copy_A(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_K = cute_ops.make_tiled_copy_B(
            smem_copy_atom_QK, tiled_mma_qk
        ).get_slice(tidx)
        smem_thr_copy_V = cute_ops.make_tiled_copy_B(
            smem_copy_atom_V, tiled_mma_pv
        ).get_slice(tidx)
        tSsQ = smem_thr_copy_Q.partition_S(sQ)
        tSsK = smem_thr_copy_K.partition_S(sK)
        tOsVt = smem_thr_copy_V.partition_S(sVt)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        softmax_num_rows = acc_O.shape[0][0] * acc_O.shape[1]
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=softmax_num_rows,
            softmax_scale=softmax_scale,
        )
        q_consumer_phase = Int32(0)
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            del split_idx
            seqlen = SeqlenInfoCls(batch_idx)
            cute.arch.mbarrier_wait(mbar_ptr_Q, phase=q_consumer_phase)
            q_consumer_phase ^= 1

            softmax.reset()
            acc_O.fill(0.0)
            kv_consumer_state = cute_pipeline.make_pipeline_state(
                cutlass.pipeline.PipelineUserType.Consumer, self.num_stages
            )

            mask = AttentionMask(
                self.tile_m,
                self.tile_n,
                seqlen,
                block_info.window_size_left,
                block_info.window_size_right,
                self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            )
            mask_fn = partial(
                mask.apply_mask,
                batch_idx=batch_idx,
                head_idx=head_idx,
                m_block=m_block,
                thr_mma=thr_mma_qk,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                aux_tensors=aux_tensors,
                fastdiv_mods=None,
                mask_mod=self.mask_mod,
            )

            n_block_min, n_block_max = block_info.get_n_block_min_max(seqlen, m_block)
            if n_block_max > n_block_min:
                kv_consumer_state = self.mma_one_n_block(
                    n_block_max - 1,
                    kv_consumer_state,
                    thr_mma_qk,
                    thr_mma_pv,
                    tSrQ,
                    tSrK,
                    tOrVt,
                    acc_O,
                    smem_thr_copy_Q,
                    smem_thr_copy_K,
                    smem_thr_copy_V,
                    tSsQ,
                    tSsK,
                    tOsVt,
                    pipeline_k,
                    pipeline_v,
                    softmax,
                    seqlen,
                    batch_idx,
                    head_idx,
                    m_block,
                    partial(mask_fn, mask_seqlen=True),
                    aux_tensors=aux_tensors,
                    is_first_n_block=True,
                )
                n_block_max -= 1

                if const_expr(self.is_causal or self.is_local):
                    n_block_min_causal_local_mask = (
                        block_info.get_n_block_min_causal_local_mask(
                            seqlen, m_block, n_block_min
                        )
                    )
                    for n_tile in cutlass.range(
                        n_block_max - n_block_min_causal_local_mask, unroll=1
                    ):
                        kv_consumer_state = self.mma_one_n_block(
                            n_block_max - 1 - n_tile,
                            kv_consumer_state,
                            thr_mma_qk,
                            thr_mma_pv,
                            tSrQ,
                            tSrK,
                            tOrVt,
                            acc_O,
                            smem_thr_copy_Q,
                            smem_thr_copy_K,
                            smem_thr_copy_V,
                            tSsQ,
                            tSsK,
                            tOsVt,
                            pipeline_k,
                            pipeline_v,
                            softmax,
                            seqlen,
                            batch_idx,
                            head_idx,
                            m_block,
                            partial(mask_fn, mask_seqlen=False),
                            aux_tensors=aux_tensors,
                        )
                    n_block_max = cutlass.min(
                        n_block_max, n_block_min_causal_local_mask
                    )

                n_block_min_before_local_mask = (
                    block_info.get_n_block_min_before_local_mask(
                        seqlen, m_block, n_block_min
                    )
                )
                n_block_min_before_local_mask = cutlass.min(
                    n_block_min_before_local_mask, n_block_max
                )
                for n_tile in cutlass.range(
                    n_block_max - n_block_min_before_local_mask, unroll=1
                ):
                    kv_consumer_state = self.mma_one_n_block(
                        n_block_max - 1 - n_tile,
                        kv_consumer_state,
                        thr_mma_qk,
                        thr_mma_pv,
                        tSrQ,
                        tSrK,
                        tOrVt,
                        acc_O,
                        smem_thr_copy_Q,
                        smem_thr_copy_K,
                        smem_thr_copy_V,
                        tSsQ,
                        tSsK,
                        tOsVt,
                        pipeline_k,
                        pipeline_v,
                        softmax,
                        seqlen,
                        batch_idx,
                        head_idx,
                        m_block,
                        partial(mask_fn, mask_seqlen=False),
                        aux_tensors=aux_tensors,
                    )
                n_block_max = n_block_min_before_local_mask

                if const_expr(
                    self.is_local and block_info.window_size_left is not None
                ):
                    for n_tile in cutlass.range(n_block_max - n_block_min, unroll=1):
                        kv_consumer_state = self.mma_one_n_block(
                            n_block_max - 1 - n_tile,
                            kv_consumer_state,
                            thr_mma_qk,
                            thr_mma_pv,
                            tSrQ,
                            tSrK,
                            tOrVt,
                            acc_O,
                            smem_thr_copy_Q,
                            smem_thr_copy_K,
                            smem_thr_copy_V,
                            tSsQ,
                            tSsK,
                            tOsVt,
                            pipeline_k,
                            pipeline_v,
                            softmax,
                            seqlen,
                            batch_idx,
                            head_idx,
                            m_block,
                            partial(mask_fn, mask_seqlen=False),
                            aux_tensors=aux_tensors,
                        )
            sink_val = None
            if const_expr(has_attention_sink_bias):
                sink_val = cute.make_rmem_tensor(softmax.num_rows, Float32)
                for r in cutlass.range_constexpr(softmax.num_rows):
                    if const_expr(self.pack_gqa):
                        row_idx = m_block * self.tile_m + taccOcO_mn[r, 0][0]
                        q_token_idx = row_idx // self.qhead_per_kvhead
                        q_head_in_group = row_idx - q_token_idx * self.qhead_per_kvhead
                        q_head_idx = head_idx * self.qhead_per_kvhead + q_head_in_group
                    else:
                        q_head_idx = head_idx
                    sink_val[r] = mAttentionSinkBias[q_head_idx]
            row_scale = softmax.finalize(sink_val=sink_val)
            softmax.rescale_O(acc_O, row_scale)
            self.epilogue(
                acc_O,
                softmax.row_sum,
                mO,
                mLSE,
                sO,
                seqlen,
                gmem_tiled_copy_O,
                tma_atom_O,
                tiled_mma_pv,
                tidx,
                m_block,
                head_idx,
                batch_idx,
            )

            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.PFull),
                number_of_threads=self.num_threads,
            )
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()
