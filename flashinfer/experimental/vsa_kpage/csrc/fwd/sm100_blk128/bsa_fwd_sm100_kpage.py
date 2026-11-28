# ruff: noqa: F841, B007
# Preserve upstream CuTe DSL constructs during the KPage extraction.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# KPage variant of the SM100 blk128 block-sparse forward (bsa_fwd_sm100.py).
#
# The K/V producer is replaced by a page gather: each 128-row smem stage is
# assembled from eight 16-row pages taken from anywhere in K/V, so pages of
# different sparse blocks share a tile and a block with vc valid rows costs
# 16 * ceil(vc / 16) rows instead of 128.  Page ids at or beyond the tensor
# are TMA out-of-bounds coordinates and load as zeros (dummy pages).  Fully
# valid pages form an unmasked prefix of each walk; the partial suffix carries
# four packed keep words per tile that the softmax warps apply before the row
# max.  The metadata builder makes every walk an even number of tiles, so the
# odd-tile handling of the base kernel is gone.
#
# BF16, non-causal, MHA, head_dim 128.  Q blocks with <= 64 valid rows are
# handled by bsa_fwd_sm100_kpage_q64.py.
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
import types
from dataclasses import dataclass
from functools import partial
from typing import Callable, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as cutlass_pipeline
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cutlass_dsl import BaseDSL
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cuda.bindings.driver as cuda

import cutlass.utils.blackwell_helpers as sm100_utils_basic
from flashinfer.experimental.vsa_kpage._cutlass_compat import (
    Arch,
    LayoutEnum,
    SmemAllocator,
    TmemAllocator,
)

from flashinfer.experimental.vsa_kpage.csrc.utils import (
    copy_utils,
    kernel_utils as utils,
    layout_utils,
    mma_sm100_desc as sm100_desc,
    pipeline as pipeline_custom,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.tcgen05_mma_helpers import (
    i64_to_i32x2,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.block_sparse_tile_scheduler import (
    TileSchedulerProtocol,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_tile_scheduler import (
    KPagePersistentTileScheduler,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.cute_dsl_utils import (
    ParamsBase,
    assume_tensor_aligned,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.subset_barrier import (
    _SubsetNamedBarrier,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.named_barrier import (
    NamedBarrierFwdSm100,
)
from flashinfer.experimental.vsa_kpage.csrc.utils.softmax import SoftmaxSm100
from flashinfer.experimental.vsa_kpage.csrc.fwd.sm100_blk64 import bsa_fwd_helpers
from flashinfer.experimental.vsa_kpage.csrc.utils import kpage_utils
from flashinfer.experimental.vsa_kpage.csrc.utils.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
)

from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import (
    FRAGS_PER_TILE,
    PAGE,
)


# =============================================================================
# Public Kernel Class
# =============================================================================
class BlockSparseAttnForwardSm100KPage:
    tile_m = 128
    tile_n = 128

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tail_page: cutlass.Constexpr[bool] = False,
        nonpositive_scale: cutlass.Constexpr[bool] = False,
    ):
        """``tail_page``: seqlen_k is not a multiple of 16, so the K/V page descriptors
        are built over the row extent (a page past seqlen_k loads as zeros) instead
        of the 16-row page view, whose whole-page TMA boxes are a few percent faster.
        ``nonpositive_scale``: softmax_scale <= 0.  max(scale * S) is not scale * max(S) then,
        and the -inf of masked keys times the scale is NaN / +inf, so the scores are scaled
        before the masks and the softmax statistics run with a unit scale (as in the base
        blk64 forward)."""
        self.qhead_per_kvhead = qhead_per_kvhead
        self.tail_page = tail_page
        self.nonpositive_scale = nonpositive_scale
        assert head_dim == 128 and (head_dim_v is None or head_dim_v == 128), (
            "page smem slices assume head_dim 128 (BF16, SW128)"
        )
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        self.check_hdim_v_oob = head_dim_v != self.head_dim_v_padded
        self.m_block_size = self.tile_m
        self.n_block_size = self.tile_n
        self.q_stage = 1
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        self.split_P_arrive = self.n_block_size // 4 * 3
        self.split_P_arrive = int(self.split_P_arrive / 32) * 32  # multiple of 32
        assert self.split_P_arrive % 32 == 0
        assert self.split_P_arrive < self.n_block_size
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch >= Arch.sm_100 and self.arch <= Arch.sm_110f, (
            "Only SM 10.x and 11.x are supported"
        )

        self.cta_tiler = (self.m_block_size, self.n_block_size, self.head_dim_padded)
        self.mma_tiler_qk = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_pv = (
            self.m_block_size,
            self.head_dim_v_padded,
            self.n_block_size,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.CLC
        is_sm103 = self.arch >= Arch.sm_103 and self.arch <= Arch.sm_103f
        assert isinstance(is_sm103, bool)
        # The hardware row max is of the unscaled scores, which is not the max of the scaled ones for a nonpositive scale.
        self.use_ldred_rowmax = is_sm103 and not nonpositive_scale
        self.enable_ex2_emu = self.head_dim_padded <= 128 and not is_sm103

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.mma_warp_id = 12
        self.epilogue_warp_ids = (13,)
        self.load_warp_ids = (14, 15)
        self.tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")

        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.softmax0_warp_ids,
                *self.softmax1_warp_ids,
                *self.correction_warp_ids,
                self.mma_warp_id,
                *self.load_warp_ids,
                *self.epilogue_warp_ids,
            )
        )

        self.s_stage = 2  # Always 2: for q_stage=1 it's n-direction
        self.tmem_s_offset = [0, self.n_block_size]  # e.g., 0, 128
        self.tmem_o_offset = [
            self.tmem_s_offset[-1] + self.n_block_size + i * self.head_dim_v_padded
            for i in range(self.s_stage)
        ]  # e.g., 256, 384
        self.tmem_total = self.tmem_o_offset[-1] + self.head_dim_v_padded
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_s_to_p_offset = self.n_block_size // 2
        self.tmem_p_offset = [
            self.tmem_s_offset[i] + self.tmem_s_to_p_offset for i in range(2)
        ]  # 0, 128

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self.num_regs_softmax = 184
        self.num_regs_correction = 96
        self.num_regs_other = 48
        assert (
            len(self.softmax0_warp_ids + self.softmax1_warp_ids) * self.num_regs_softmax
            + len(self.correction_warp_ids) * self.num_regs_correction
            + (
                self.threads_per_cta // cute.arch.WARP_SIZE
                - len(self.softmax0_warp_ids + self.softmax1_warp_ids)
                - len(self.correction_warp_ids)
            )
            * self.num_regs_other
            == 2048
        ), "register budgets must sum to 64K"

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.kv_stage = 4
        self.kv_elems_per_stage = self.n_block_size * self.head_dim_padded

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, s_q, h, d)
        mK: cute.Tensor,  # (b_k, s_k, h_k, d), partial Page16 suffixes zero
        mV: cute.Tensor,  # (b_k, s_k, h_k, dv)
        mO: cute.Tensor,  # (b, s_q, h, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mPageTable: cute.Tensor,  # (total_tiles, 8) int32 page ids, 32B-aligned rows
        mMaskTable: cute.Tensor,  # (total_tiles, 4) int32 packed keep masks
        mTileOffsets: cute.Tensor,  # (batch, heads, num_q_blocks), int32 TILE offset of each walk
        mFullTileCounts: cute.Tensor,  # unmasked N128 prefix length
        mBlockNums: cute.Tensor,  # (batch, heads, num_q_blocks), int32 EVEN tile counts
        mWorkIndices: Optional[cute.Tensor],  # compact blk128 work queue
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
        """
        # setup static attributes before smem/grid/tma computation
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.o_dtype = mO.element_type
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # Kernel boundary contract is BHSD. Convert (B, H, S, D) to
        # the mainloop/TMA-friendly (S, D, H, B) view internally.
        Q_layout_transpose = [2, 3, 1, 0]
        mQ = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [2, 3, 1, 0]
        mK, mV = [
            cute.make_tensor(
                t.iterator, cute.select(t.layout, mode=KV_layout_transpose)
            )
            for t in (mK, mV)
        ]
        O_layout_transpose = [2, 3, 1, 0]
        LSE_layout_transpose = [2, 1, 0]
        num_splits = Int32(1)
        mO = cute.make_tensor(
            mO.iterator, cute.select(mO.layout, mode=O_layout_transpose)
        )
        mLSE = (
            cute.make_tensor(
                mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose)
            )
            if const_expr(mLSE is not None)
            else None
        )
        # (s, d, h, b) -> (d, s, h, b)
        V_layout_transpose = [1, 0, 2, 3]
        mV = cute.make_tensor(
            mV.iterator, cute.select(mV.layout, mode=V_layout_transpose)
        )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(self.q_dtype != cutlass.BFloat16):
            raise TypeError("blk128 forward requires BF16 Q/K/V")
        if const_expr(self.o_dtype != cutlass.BFloat16):
            raise TypeError("blk128 forward requires BF16 output")
        self._setup_attributes()
        # Tuned defaults for the polynomial ex2 emulation path.
        self.ex2_emu_freq = 0
        self.ex2_emu_start_frg = 0
        if const_expr(self.enable_ex2_emu):
            self.ex2_emu_freq = 10

        q_major_mode = cute.nvgpu.OperandMajorMode.K
        k_major_mode = cute.nvgpu.OperandMajorMode.K
        v_major_mode = cute.nvgpu.OperandMajorMode.MN
        self.o_layout = LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.k_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            self.v_dtype,
            p_major_mode,
            v_major_mode,
            self.pv_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_pv[:2],
            p_source,
        )

        self.epi_tile = (self.m_block_size, self.head_dim_v_padded)

        sQ_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_qk, self.mma_tiler_qk, self.q_dtype, self.q_stage
        )
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        sV_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_pv, self.mma_tiler_pv, self.v_dtype, self.kv_stage
        )
        sO_layout = sm100_utils_basic.make_smem_layout_epi(
            self.o_dtype, self.o_layout, self.epi_tile, self.s_stage
        )
        self.tma_copy_bytes = {
            name: cute.size_in_bytes(
                mX.element_type, cute.select(layout, mode=[0, 1, 2])
            )
            for name, mX, layout in [
                ("Q", mQ, sQ_layout),
                ("K", mK, sK_layout),
                ("V", mV, sV_layout),
            ]
        }
        # TMA load for Q
        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
        tma_store_op = cpasync.CopyBulkTensorTileS2GOp()

        tma_atom_Q, mQ = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mQ,
            cute.select(sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_qk,
            tiled_mma_qk,
        )

        # K/V TMA atoms move one 16-row page.  The page index is a tensor mode,
        # so ids >= npages are out of bounds and the descriptor zero-fills them.
        # sK stage ((128,16),1,(4,2),4):((64,1),0,(16,8192),16384); one page is
        # the slice (16,(64,2)):(64,(1,8192)), slot stride 1024, stage 16384.
        assert sK_layout.outer.stride[0][0] == 64 and sK_layout.outer.stride[0][1] == 1
        assert sK_layout.outer.shape[0][1] == 16
        assert sK_layout.outer.shape[2][0] == 4 and sK_layout.outer.shape[2][1] == 2
        assert (
            sK_layout.outer.stride[2][0] == 16 and sK_layout.outer.stride[2][1] == 8192
        )
        assert sK_layout.outer.stride[3] == 16384
        # sV stage (((64,2),16),1,8,4):(((1,8192),64),0,1024,16384); one page is
        # the slice ((64,2),16):((1,8192),64), slot stride 1024, stage 16384.
        assert (
            sV_layout.outer.stride[0][0][0] == 1
            and sV_layout.outer.stride[0][0][1] == 8192
        )
        assert (
            sV_layout.outer.shape[0][0][0] == 64 and sV_layout.outer.shape[0][0][1] == 2
        )
        assert sV_layout.outer.shape[0][1] == 16 and sV_layout.outer.stride[0][1] == 64
        assert sV_layout.outer.stride[2] == 1024
        assert sV_layout.outer.stride[3] == 16384

        sK_page_layout = cute.make_composed_layout(
            sK_layout.inner,
            0,
            cute.make_layout((PAGE, (64, 2)), stride=(64, (1, 8192))),
        )
        npages = cute.size(mK.shape[0]) // PAGE
        mK_page = mK
        if const_expr(not self.tail_page):
            mK_page = cute.make_tensor(
                mK.iterator,
                cute.make_layout(
                    (PAGE, self.head_dim_padded, npages, mK.shape[2], mK.shape[3]),
                    stride=(
                        mK.stride[0],
                        mK.stride[1],
                        PAGE * mK.stride[0],
                        mK.stride[2],
                        mK.stride[3],
                    ),
                ),
            )
        tma_atom_K, mK = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mK_page,
            sK_page_layout,
            (PAGE, self.head_dim_padded),
        )
        sV_page_layout = cute.make_composed_layout(
            sV_layout.inner,
            0,
            cute.make_layout(((64, 2), PAGE), stride=((1, 8192), 64)),
        )
        mV_page = mV
        if const_expr(not self.tail_page):
            mV_page = cute.make_tensor(
                mV.iterator,
                cute.make_layout(
                    (self.head_dim_v_padded, PAGE, npages, mV.shape[2], mV.shape[3]),
                    stride=(
                        mV.stride[0],
                        mV.stride[1],
                        PAGE * mV.stride[1],
                        mV.stride[2],
                        mV.stride[3],
                    ),
                ),
            )
        tma_atom_V, mV = cpasync.make_tiled_tma_atom(
            tma_load_op,
            mV_page,
            sV_page_layout,
            (self.head_dim_v_padded, PAGE),
        )

        tma_atom_O, mO = cpasync.make_tiled_tma_atom(
            tma_store_op,
            mO,
            cute.select(sO_layout, mode=[0, 1]),
            self.epi_tile,
        )

        TileScheduler = KPagePersistentTileScheduler
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mQ.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            mV.shape[
                0
            ],  # Note that this is different from Sm90 since we transpose mV in Sm100
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=True,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=self.scheduling_mode,
            num_work=(
                cute.size(mWorkIndices)
                if const_expr(mWorkIndices is not None)
                else None
            ),
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)

        clc_response_size = self.sched_stages * 4
        clc_mbar_size = self.sched_stages * 2

        @cute.struct
        class SharedStorage:
            # m_barriers for pipelines
            mbar_load_Q: cute.struct.MemRange[Int64, self.q_stage * 2]
            mbar_load_KV: cute.struct.MemRange[Int64, self.kv_stage * 2]
            mbar_S_full_P_full_O_rescaled: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_P_full_lastsplit: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_full: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_softmax_stats: cute.struct.MemRange[Int64, self.s_stage * 2]
            mbar_O_epi: cute.struct.MemRange[Int64, self.s_stage * 2]
            # Tmem holding buffer
            tmem_holding_buf: Int32
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.m_block_size * 2]
            # CLC buffers (mbarriers + response storage)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size], 16
            ]
            sO: cute.struct.Align[
                cute.struct.MemRange[self.o_dtype, sO_size], self.buffer_align_bytes
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(sK_layout)],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(
            softmax_scale
        )
        fastdiv_mods = None

        # Launch the kernel synchronously
        self.kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_O,
            softmax_scale_log2,
            softmax_scale,
            sQ_layout,
            sK_layout,
            tP_layout,
            sV_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            mPageTable,
            mMaskTable,
            mTileOffsets,
            mFullTileCounts,
            mBlockNums,
            mWorkIndices,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            stream=stream,
            min_blocks_per_mp=1,
        )

    #  GPU device kernel
    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,  # (s_q, d, h, b)
        mK: cute.Tensor,  # (s_k, d, h_k, b_k)
        mV: cute.Tensor,  # (d, s_k, h_k, b_k)
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: Optional[cute.CopyAtom],
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mFullTileCounts: cute.Tensor,
        mBlockNums: cute.Tensor,
        mWorkIndices: Optional[cute.Tensor],
    ):
        """The device kernel implementation of the Fused Multi-Head Attention.

        This kernel coordinates multiple specialized warps to perform different phases of the FMHA computation:
        1. Load warp: Loads Q, K, V data from global memory to shared memory using TMA
        2. MMA warp: Performs matrix multiplications (Q*K^T and P*V)
        3. Softmax warps: Compute softmax normalization on attention scores
        4. Correction warps: Apply adjustments to intermediate results
        5. Epilogue warp: Handles final output transformation and storage

        The kernel implements a complex pipeline with overlapping computation and memory operations,
        using tensor memory access (TMA) for efficient data loading, warp specialization for different
        computation phases, and optional attention masking.
        """

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == 0:
            for tma_atom in (tma_atom_Q, tma_atom_K, tma_atom_V, tma_atom_O):
                if const_expr(tma_atom is not None):
                    cpasync.prefetch_descriptor(tma_atom)

        # Alloc
        smem = SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_alloc_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.TmemPtr),
            num_threads=cute.arch.WARP_SIZE
            * len(
                (
                    self.mma_warp_id,
                    *self.softmax0_warp_ids,
                    *self.softmax1_warp_ids,
                    *self.correction_warp_ids,
                )
            ),
        )
        # Tensor memory dealloc barrier init
        tmem = TmemAllocator(
            storage.tmem_holding_buf,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.mma_warp_id,
        )

        ThreadCooperativeGroup = partial(
            pipeline.CooperativeGroup, pipeline.Agent.Thread
        )
        mma_warp = ThreadCooperativeGroup(len([self.mma_warp_id]))
        q_tma_warp = ThreadCooperativeGroup(1)
        kv_tma_warps = ThreadCooperativeGroup(len(self.load_warp_ids))
        softmax_warps = ThreadCooperativeGroup(len(self.softmax0_warp_ids))
        softmax_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids)
        )
        correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        softmax_correction_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.softmax0_warp_ids + self.correction_warp_ids)
        )
        epilogue_threads = ThreadCooperativeGroup(
            cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
        )
        pipeline_q = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_Q.data_ptr(),
            num_stages=self.q_stage,
            producer_group=q_tma_warp,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["Q"],
            defer_sync=True,
        )
        pipeline_kv = pipeline_custom.PipelineTmaUmma.create(
            barrier_storage=storage.mbar_load_KV.data_ptr(),
            num_stages=self.kv_stage,
            producer_group=kv_tma_warps,
            consumer_group=mma_warp,
            tx_count=self.tma_copy_bytes["K"] // len(self.load_warp_ids),
            defer_sync=True,
        )
        # This pipeline is not the typical producer-consumer pipeline. The "producer" mma warp
        # uses it to signal that S is ready, and the softmax threads wait for S to be ready.
        # When softmax threads write P to tmem and the correction threads have rescaled O, they
        # signal as "consumer". The mma warp then waits for that signal to do the P @ V gemm.
        pipeline_s_p_o = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_S_full_P_full_O_rescaled.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=softmax_correction_threads,
            defer_sync=True,
        )
        pipeline_p_lastsplit = pipeline_custom.PipelineAsyncUmma.create(
            barrier_storage=storage.mbar_P_full_lastsplit.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_warps,
            consumer_group=mma_warp,
            defer_sync=True,
        )
        # MMA warp uses this to signal to the correction warps that O is ready.
        pipeline_o_acc = pipeline_custom.PipelineUmmaAsync.create(
            barrier_storage=storage.mbar_O_full.data_ptr(),
            num_stages=self.s_stage,
            producer_group=mma_warp,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        pipeline_sm_stats = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_softmax_stats.data_ptr(),
            num_stages=self.s_stage,
            producer_group=softmax_threads,
            consumer_group=correction_threads,
            defer_sync=True,
        )
        # Should put the NamedBarrier inside the pipeline class so we'll just have pipeline_sm_stats
        sm_stats_barrier = _SubsetNamedBarrier(
            barrier_id=int(NamedBarrierFwdSm100.SoftmaxStatsW0),
            num_threads=cute.arch.WARP_SIZE * 2,
        )
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.s_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # Raw K/V smem base plus the per-page smem layouts (16-row slices of
        # the swizzled stage layouts).
        sKV_ptr = storage.sK.data_ptr()
        sK_page_layout = cute.make_composed_layout(
            sK_layout.inner,
            0,
            cute.make_layout((PAGE, (64, 2)), stride=(64, (1, 8192))),
        )
        sV_page_layout = cute.make_composed_layout(
            sV_layout.inner,
            0,
            cute.make_layout(((64, 2), PAGE), stride=((1, 8192), 64)),
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sK.iterator, sV_layout.inner), sV_layout.outer
        )
        sO = storage.sO.get_tensor(sO_layout.outer, swizzle=sO_layout.inner)

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.s_stage * self.m_block_size * 2)
        )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        qk_acc_shape = thr_mma_qk.partition_shape_C(self.mma_tiler_qk[:2])
        # This is a fake tensor, by right we need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tStS = thr_mma_qk.make_fragment_C(cute.append(qk_acc_shape, self.s_stage))
        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO = thr_mma_pv.make_fragment_C(cute.append(pv_acc_shape, self.s_stage))
        tOtO = cute.make_tensor(tOtO.iterator + self.tmem_o_offset[0], tOtO.layout)
        tP = cute.make_tensor(tStS.iterator, tP_layout.outer)
        tOrP = thr_mma_pv.make_fragment_A(tP)[None, None, None, 0]
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP.layout,
                cute.make_layout((self.s_stage,), stride=(tP_stage_stride,)),
            ),
        )

        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
        )
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_mbar_ptr = storage.clc_mbar_ptr.data_ptr()
        clc_pipeline_producer_group = cutlass_pipeline.CooperativeGroup(
            cutlass_pipeline.Agent.Thread
        )
        clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
            cutlass_pipeline.Agent.Thread,
            self.threads_per_cta,
        )
        clc_pipeline = cutlass_pipeline.PipelineClcFetchAsync.create(
            barrier_storage=clc_mbar_ptr,
            num_stages=self.sched_stages,
            producer_group=clc_pipeline_producer_group,
            consumer_group=clc_pipeline_consumer_group,
            tx_count=16,
        )
        tile_scheduler = self.tile_scheduler_cls.create(
            tile_sched_params,
            clc_response_ptr=clc_response_ptr,
            work_indices=mWorkIndices,
        )
        clc_consumer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Consumer,
            self.sched_stages,
        )
        tile_scheduler.set_clc_pipeline(clc_pipeline, clc_consumer_state)

        pipeline_init_wait()

        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                mQ,
                mK,
                mV,
                sQ,
                sKV_ptr,
                sK_page_layout,
                sV_page_layout,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                pipeline_q,
                pipeline_kv,
                clc_pipeline,
                tile_scheduler,
                mPageTable,
                mTileOffsets,
                mBlockNums,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            # Alloc tensor memory buffer
            tmem.allocate(cute.arch.get_max_tmem_alloc_cols("sm_100"))
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.mma(
                tiled_mma_qk,
                tiled_mma_pv,
                sQ,
                sK,
                sV,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
                SeqlenInfoCls,
                tile_scheduler,
                mBlockNums,
            )
            # Dealloc the tensor memory buffer
            tmem.relinquish_alloc_permit()
            tmem_alloc_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        # ///////////////////////////////////////////////////////////////////////////////
        #  Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.epilogue_warp_ids[0]
            and warp_idx <= self.epilogue_warp_ids[-1]
        ):
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.epilogue_s2g(
                mO,
                sO,
                tma_atom_O,
                pipeline_o_epi,
                SeqlenInfoCls,
                tile_scheduler,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Softmax
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx <= self.softmax1_warp_ids[-1]:
            # increase register after decreasing
            cute.arch.setmaxregister_increase(self.num_regs_softmax)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            softmax_loop = partial(
                self.softmax_loop,
                softmax_scale_log2=(
                    Float32(1.0)
                    if const_expr(self.nonpositive_scale)
                    else softmax_scale_log2
                ),
                softmax_scale=softmax_scale,
                score_scale_log2=softmax_scale_log2,
                thr_mma_qk=thr_mma_qk,
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                tile_scheduler=tile_scheduler,
                mMaskTable=mMaskTable,
                mTileOffsets=mTileOffsets,
                mFullTileCounts=mFullTileCounts,
                mBlockNums=mBlockNums,
            )

            stage = Int32(0 if warp_idx < self.softmax1_warp_ids[0] else 1)
            softmax_loop(stage=stage, tStS=tStS)

            tmem_alloc_barrier.arrive()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Correction
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.correction_warp_ids[0] and warp_idx < self.mma_warp_id:
            cute.arch.setmaxregister_decrease(self.num_regs_correction)
            # sync with mma warp before retrieving tmem ptr
            tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.qk_acc_dtype)
            self.correction_loop(
                thr_mma_qk,
                thr_mma_pv,
                tStS,
                tOtO,
                sScale,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                pipeline_o_epi,
                (
                    Float32(1.0)
                    if const_expr(self.nonpositive_scale)
                    else softmax_scale_log2
                ),
                SeqlenInfoCls,
                tile_scheduler,
                mBlockNums,
            )
            tmem_alloc_barrier.arrive()

        return

    @cute.jit
    def load(
        self,
        thr_mma_qk: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sK_page_layout: cute.ComposedLayout,
        sV_page_layout: cute.ComposedLayout,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        clc_pipeline: Optional[cutlass_pipeline.PipelineClcFetchAsync],
        tile_scheduler: "TileSchedulerProtocol",
        mPageTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mBlockNums: cute.Tensor,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        load_warp_rank = warp_idx - Int32(self.load_warp_ids[0])

        q_producer_phase = Int32(1)
        kv_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.kv_stage
        )
        clc_producer_state = cutlass_pipeline.make_pipeline_state(
            cutlass_pipeline.PipelineUserType.Producer,
            self.sched_stages,
        )
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
            tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))  # (128, 128)
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )  # (128, 128, 1)

            head_idx_kv = head_idx // self.qhead_per_kvhead
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            # Slice one head and batch before TMA partitioning; page ids stay
            # descriptor coordinates, including the out-of-bounds dummy id.
            if const_expr(self.tail_page):
                gK_tma = cute.group_modes(
                    cute.local_tile(
                        mK[None, None, head_idx_kv, batch_idx],
                        (PAGE, self.head_dim_padded),
                        (None, 0),
                    ),
                    0,
                    2,
                )
                gV_tma = cute.group_modes(
                    cute.local_tile(
                        mV[None, None, head_idx_kv, batch_idx],
                        (self.head_dim_v_padded, PAGE),
                        (0, None),
                    ),
                    0,
                    2,
                )
            else:
                gK_tma = cute.group_modes(
                    mK[None, None, None, head_idx_kv, batch_idx],
                    0,
                    2,
                )
                gV_tma = cute.group_modes(
                    mV[None, None, None, head_idx_kv, batch_idx],
                    0,
                    2,
                )
            load_K = partial(
                self.load_KV_paged,
                tma_atom_K,
                gK_tma,
                sKV_ptr,
                sK_page_layout,
                mPageTable,
                load_warp_rank,
                pipeline_kv=pipeline_kv,
                K_or_V="K",
            )
            load_V = partial(
                self.load_KV_paged,
                tma_atom_V,
                gV_tma,
                sKV_ptr,
                sV_page_layout,
                mPageTable,
                load_warp_rank,
                pipeline_kv=pipeline_kv,
                K_or_V="V",
            )

            # Tile t of this work item uses page-table row tile_base + t.
            raw_block_count = mBlockNums[
                batch_idx, head_idx, m_block
            ]  # EVEN by builder
            process_tile = raw_block_count > Int32(0)
            block_iter_count = raw_block_count
            tile_base = mTileOffsets[batch_idx, head_idx, m_block]

            if process_tile:
                load_K(
                    row=tile_base + block_iter_count - 1,
                    producer_state=kv_producer_state,
                )  # K0
                if warp_idx == self.load_warp_ids[0]:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                kv_producer_state.advance()

                # q_stage=1 intra-warp overlap
                # Load order: K[N-1], Q, K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2
                q_producer_phase ^= 1

                # Prologue: K[N-2]
                load_K(
                    row=tile_base + block_iter_count - 2,
                    producer_state=kv_producer_state,
                )
                kv_producer_state.advance()

                # Flat main loop: N-2 iterations, each loads V then K
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: V for the tile whose S was computed earlier
                    load_V(
                        row=tile_base + block_iter_count - 1 - i,
                        producer_state=kv_producer_state,
                    )
                    kv_producer_state.advance()
                    # K[N-3-i]: K for the next QK GEMM
                    load_K(
                        row=tile_base + block_iter_count - 3 - i,
                        producer_state=kv_producer_state,
                    )
                    kv_producer_state.advance()

                # Epilogue: last 2 V loads
                load_V(row=tile_base + 1, producer_state=kv_producer_state)
                kv_producer_state.advance()
                load_V(row=tile_base, producer_state=kv_producer_state)
                kv_producer_state.advance()

            tile_scheduler.prefetch_next_work()
            if load_warp_rank == Int32(0):
                clc_pipeline.producer_acquire(clc_producer_state)
                mbarrier_addr = clc_pipeline.producer_get_barrier(clc_producer_state)
                tile_scheduler.advance_to_next_work(mbarrier_addr=mbarrier_addr)
                clc_producer_state.advance()
            work_tile = tile_scheduler.consumer_advance()
            # End of persistent scheduler loop

        pipeline_kv.producer_tail(kv_producer_state)
        # This is equivalent to pipeline_q.producer_tail
        if warp_idx == self.load_warp_ids[0]:
            pipeline_q.producer_acquire_w_index_phase(
                self.q_stage - 1, q_producer_phase
            )
        if load_warp_rank == Int32(0):
            clc_pipeline.producer_tail(clc_producer_state)

    @cute.jit
    def mma(
        self,
        tiled_mma_qk: cute.ThrMma,
        tiled_mma_pv: cute.ThrMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        tile_scheduler: "TileSchedulerProtocol",
        mBlockNums: cute.Tensor,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        qk_mma_op = tiled_mma_qk.op
        pv_mma_op = tiled_mma_pv.op
        q_smem_base = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, sQ.element_type.width, sQ.layout[0]),
            sQ.iterator.type.swizzle_type,
            sm100_desc.Major.K,
        )
        k_smem_base = sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, sK.element_type.width, sK.layout[0]),
            sK.iterator.type.swizzle_type,
            sm100_desc.Major.K,
        )
        q_smem_start = sm100_desc.make_smem_desc_start_addr(
            sQ[None, None, None, self.q_stage - 1].iterator
        )
        sm100_utils.declare_ptx_smem_desc(
            q_smem_start,
            q_smem_base,
            tSrQ[None, None, None, 0].layout,
            var_name_prefix="bsa_fwd_q_smem_desc",
        )
        sm100_utils.declare_ptx_idesc(qk_mma_op, var_name="bsa_fwd_qk_mma_idesc")
        gemm_Si = [
            partial(
                sm100_utils.gemm_ptx_precomputed_varname,
                self.tmem_s_offset[stage],
                smem_desc_base_b=k_smem_base,
                tCrB_layout=tSrK[None, None, None, 0].layout,
                smem_var_name_prefix="bsa_fwd_q_smem_desc",
                idesc_var_name="bsa_fwd_qk_mma_idesc",
                smem_offset=0,
                zero_init=True,
            )
            for stage in range(self.s_stage)
        ]
        gemm_Pi = [
            partial(
                sm100_utils.gemm_ptx_partial,
                pv_mma_op,
                self.tmem_o_offset[stage],
                tOrP[None, None, None, stage],
                sA=None,
                split_arrive=(self.split_P_arrive if self.split_P_arrive > 0 else None),
            )
            for stage in range(self.s_stage)
        ]
        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline_custom.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            raw_block_count = mBlockNums[
                batch_idx, head_idx, m_block
            ]  # EVEN by builder
            process_tile = raw_block_count > Int32(0)
            block_iter_count = raw_block_count

            if process_tile:
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # GEMM order: S0=Q@K0, S1=Q@K1, {O0+=P0@V0, S0=Q@K2}, {O1+=P1@V1, S1=Q@K3}, ...
                # ================================================================

                # Prologue: wait Q0
                pipeline_q.consumer_wait_w_index_phase(0, mma_q_consumer_phase)
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index, Ki_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                sK_cur = sK[None, None, None, Ki_index]
                gemm_Si[0](
                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                pipeline_kv.consumer_wait(mma_kv_consumer_state)
                Ki_index, Ki_phase = (
                    mma_kv_consumer_state.index,
                    mma_kv_consumer_state.phase,
                )
                sK_cur = sK[None, None, None, Ki_index]
                gemm_Si[1](
                    smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                        sK_cur.iterator
                    )
                )
                pipeline_s_p_o.producer_commit_w_index(1)  # signal S1 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # Flat loop: N-2 iterations, alternating stage 0/1
                block_loop_count = block_iter_count - 2
                O_acc_s0 = False
                O_acc_s1 = False
                # Pre-declare loop variables for DSL type tracking
                Vi_index = mma_kv_consumer_state.index
                Vi_phase = mma_kv_consumer_state.phase
                Ki_index = mma_kv_consumer_state.index
                Ki_phase = mma_kv_consumer_state.phase
                mma_kv_release_state = mma_kv_consumer_state.clone()
                tOrVi = tOrV[None, None, None, Vi_index]
                sV_cur = sV[None, None, None, Vi_index]
                sK_cur = sK[None, None, None, Ki_index]

                pair_count = (
                    block_loop_count // 2
                )  # N even => block_loop_count = N-2 is even
                for i in cutlass.range(pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        if const_expr(stage == 0):
                            phase_cur, O_acc_cur = phase_s0, O_acc_s0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                        pipeline_s_p_o.producer_acquire_w_index_phase(
                            stage,
                            phase_cur,
                        )
                        # Wait V
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index, Vi_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        mma_kv_consumer_state.advance()
                        gemm_Pi[stage](
                            tCrB=tOrVi,
                            sB=sV_cur,
                            zero_init=not O_acc_cur,
                            mbar_ptr=(
                                pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
                                if self.split_P_arrive > 0
                                else None
                            ),
                            mbar_phase=phase_cur,
                        )
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        # Overlap the independent K wait with the issued PV MMA.
                        pipeline_kv.consumer_wait(mma_kv_consumer_state)
                        Ki_index, Ki_phase = (
                            mma_kv_consumer_state.index,
                            mma_kv_consumer_state.phase,
                        )
                        sK_cur = sK[None, None, None, Ki_index]
                        gemm_Si[stage](
                            smem_desc_start_b=sm100_desc.make_smem_desc_start_addr(
                                sK_cur.iterator
                            )
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        # Release V and K
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                    pipeline_kv.consumer_wait(mma_kv_consumer_state)
                    Vi_index, Vi_phase = (
                        mma_kv_consumer_state.index,
                        mma_kv_consumer_state.phase,
                    )
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    pipeline_s_p_o.producer_acquire_w_index_phase(epi_s, epi_p)
                    gemm_Pi[epi_stage_constexpr](
                        tCrB=tOrVi,
                        sB=sV_cur,
                        zero_init=epi_zi,
                        mbar_ptr=(
                            pipeline_p_lastsplit.sync_object_full.get_barrier(epi_s)
                            if self.split_P_arrive > 0
                            else None
                        ),
                        mbar_phase=epi_p,
                    )
                    pipeline_o_acc.producer_commit_w_index(epi_s)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                phase_s0 ^= 1
                phase_s1 ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        score_scale_log2: Float32,
        thr_mma_qk: cute.ThrMma,
        tStS: cute.Tensor,  # ((TILE_M, TILE_N), 1, 1, q_stage)
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        tile_scheduler: "TileSchedulerProtocol",
        mMaskTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mFullTileCounts: cute.Tensor,
        mBlockNums: cute.Tensor,
    ):
        """Compute softmax on attention scores from QK matrix multiplication.

        This method handles the softmax computation for either the first or second half of the
        attention matrix, depending on the 'stage' parameter. It calculates row-wise maximum
        and sum values needed for stable softmax computation, applies optional masking, and
        transforms raw attention scores into probability distributions.

        The implementation uses specialized memory access patterns and efficient math operations
        for computing exp(x) using exp2 functions. It also coordinates pipeline
        synchronization between MMA, correction, and sequence processing stages.
        """
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * (len(self.softmax0_warp_ids))
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tSAcc = tStS[(None, None), 0, 0, stage]  # (128, 128)
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))

        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tStP_layout = cute.composition(
            tSAcc.layout, cute.make_layout((self.m_block_size, tilePlikeFP32))
        )
        tStP = cute.make_tensor(tSAcc.iterator + self.tmem_s_to_p_offset, tStP_layout)

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)),
            self.qk_acc_dtype,
        )
        thr_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tSAcc).get_slice(tidx)
        tStS_t2r = thr_tmem_load.partition_S(tSAcc)  # (((32,32),1),1,4)

        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )
        thr_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tStP).get_slice(tidx)
        tStP_r2t = thr_tmem_store.partition_D(tStP)  # (((16,32),1),1,4)

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        warp_idx_in_wg = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            raw_block_count = mBlockNums[
                batch_idx, head_idx, m_block
            ]  # EVEN by builder
            has_work = raw_block_count > Int32(0)

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0 if const_expr(self.q_dtype.width == 16) else 0.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                thr_mma_qk=thr_mma_qk,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                thr_tmem_load=thr_tmem_load,
                thr_tmem_store=thr_tmem_store,
                tStS_t2r=tStS_t2r,
                tStP_r2t=tStP_r2t,
                sScale=sScale,
                stage=stage,
                score_scale_log2=score_scale_log2,
            )

            # Always acquire pipeline_sm_stats to stay in sync with correction
            pipeline_sm_stats.producer_acquire_w_index_phase(
                stage, sm_stats_producer_phase
            )
            sm_stats_producer_phase ^= 1

            if has_work:
                # block_iter_count is even: each WG processes exactly half the tiles
                # WG0 (stage=0): logical indices N-1, N-3, ... (stride 2)
                # WG1 (stage=1): logical indices N-2, N-4, ... (stride 2)
                block_iter_count = raw_block_count
                wg_count = block_iter_count // 2
                full_tile_count = mFullTileCounts[batch_idx, head_idx, m_block]
                partial_tile_count = block_iter_count - full_tile_count
                masked_wg_count = (partial_tile_count + 1 - stage) // 2
                full_wg_count = wg_count - masked_wg_count
                mask_row = (
                    mTileOffsets[batch_idx, head_idx, m_block]
                    + block_iter_count
                    - 1
                    - stage
                )

                if masked_wg_count > 0:
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        mMaskTable=mMaskTable,
                        mask_row=mask_row,
                        masked=True,
                        is_first=True,
                    )
                    mask_row -= 2
                    for n_tile in cutlass.range(masked_wg_count - 1, unroll=1):
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            mMaskTable=mMaskTable,
                            mask_row=mask_row,
                            masked=True,
                        )
                        mask_row -= 2
                    if full_wg_count > 0:
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            mMaskTable=mMaskTable,
                            mask_row=mask_row,
                        )
                        for n_tile in cutlass.range(full_wg_count - 1, unroll=1):
                            mma_si_consumer_phase, sm_stats_producer_phase = (
                                softmax_step(
                                    mma_si_consumer_phase,
                                    sm_stats_producer_phase,
                                    mMaskTable=mMaskTable,
                                    mask_row=mask_row,
                                )
                            )
                else:
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        mMaskTable=mMaskTable,
                        mask_row=mask_row,
                        is_first=True,
                    )
                    for n_tile in cutlass.range(wg_count - 1, unroll=1):
                        mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                            mma_si_consumer_phase,
                            sm_stats_producer_phase,
                            mMaskTable=mMaskTable,
                            mask_row=mask_row,
                        )

                sScale[tidx + stage * self.m_block_size] = softmax.row_sum[0]
                if const_expr(mLSE is not None or self.q_stage == 1):
                    sScale[
                        tidx
                        + stage * self.m_block_size
                        + self.s_stage * self.m_block_size
                    ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)
            else:
                # Empty tile: arrive barrier once (synthetic "no work" signal for correction)
                sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_sm_stats.producer_tail
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)

    @cute.jit
    def softmax_step(
        self,
        mma_si_consumer_phase: Int32,
        sm_stats_producer_phase: Int32,
        softmax: "SoftmaxSm100",
        thr_mma_qk: cute.ThrMma,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        thr_tmem_load: cute.CopyAtom,
        thr_tmem_store: cute.CopyAtom,
        tStS_t2r: cute.Tensor,
        tStP_r2t: cute.Tensor,
        sScale: cute.Tensor,
        stage: int | Int32,
        score_scale_log2: Float32,
        mMaskTable: cute.Tensor,
        mask_row: Int32,
        masked: bool = False,
        is_first: bool = False,
    ) -> Tuple[cute.Int32, cute.Int32]:
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
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        tilePlikeFP32 = self.mma_tiler_qk[1] // Float32.width * self.v_dtype.width
        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tScS = tScS[(None, None), 0, 0]  # (128, 128)
        cta_qk_tiler = (
            self.mma_tiler_qk[0] // thr_mma_qk.thr_id.shape,
            self.mma_tiler_qk[1],
        )
        tScS_shape = cta_qk_tiler  # (128, 128)
        tScP_shape = (tScS_shape[0], tilePlikeFP32)  # (128, 64)

        # Prefetch packed mask metadata before waiting for scores.
        mask_words = cute.make_rmem_tensor((4,), Int32)
        if const_expr(masked):
            for chunk in cutlass.range_constexpr(4):
                mask_words[chunk] = mMaskTable[mask_row, chunk]

        # Wait for Si
        pipeline_s_p_o.consumer_wait_w_index_phase(stage, mma_si_consumer_phase)
        tSrS_t2r = cute.make_rmem_tensor(
            thr_tmem_load.partition_D(tScS).shape, self.qk_acc_dtype
        )
        if const_expr(self.use_ldred_rowmax and not masked):
            tSrS_flat = cute.make_tensor(tSrS_t2r.iterator, cute.make_layout((128,)))
            hw_row_max = -Float32.inf
            tmem_s_addr = Int32(stage) * Int32(
                self.tmem_s_offset[1] - self.tmem_s_offset[0]
            )
            for chunk in cutlass.range_constexpr(4):
                values, chunk_max = bsa_fwd_helpers.tmem_load_red_max_32dp32b32x(
                    tmem_s_addr + chunk * 32
                )
                hw_row_max = cute.arch.fmax(hw_row_max, chunk_max)
                for value_idx in cutlass.range_constexpr(32):
                    tSrS_flat[chunk * 32 + value_idx] = values[value_idx]
        else:
            cute.copy(thr_tmem_load, tStS_t2r, tSrS_t2r)
            hw_row_max = Float32(-Float32.inf)

        if const_expr(self.nonpositive_scale):
            # Scale the finite scores before the -inf mask sentinels; the softmax then runs with a unit scale.
            tSrS_scaled = cute.make_tensor(tSrS_t2r.iterator, cute.make_layout((128,)))
            for i in cutlass.range_constexpr(0, 128, 2):
                tSrS_scaled[i], tSrS_scaled[i + 1] = cute.arch.mul_packed_f32x2(
                    (tSrS_scaled[i], tSrS_scaled[i + 1]),
                    (score_scale_log2, score_scale_log2),
                )

        if const_expr(masked):
            tSrS_flat = cute.make_tensor(tSrS_t2r.iterator, cute.make_layout((128,)))
            for chunk in cutlass.range_constexpr(4):
                kpage_utils.apply_packed_mask_chunk(
                    tSrS_flat,
                    chunk,
                    mask_words[chunk],
                )

        if const_expr(self.use_ldred_rowmax and not masked):
            row_max, acc_scale = kpage_utils.update_row_max_precomputed(
                softmax, hw_row_max, is_first
            )
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            thread_idx = thr_tmem_load.thr_idx
            sScale[thread_idx + stage * self.m_block_size] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=stage * 4 + warp_idx)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrP_r2t_f32 = cute.make_rmem_tensor(
            thr_tmem_store.partition_S(cute.make_identity_tensor(tScP_shape)).shape,
            Float32,
        )
        tSrP_r2t = cute.make_tensor(
            cute.recast_ptr(tSrP_r2t_f32.iterator, dtype=self.q_dtype), tSrS_t2r.layout
        )
        softmax.apply_exp2_convert(
            tSrS_t2r,
            tSrP_r2t,
            ex2_emu_freq=self.ex2_emu_freq,
            ex2_emu_start_frg=self.ex2_emu_start_frg,
        )
        for i in cutlass.range_constexpr(cute.size(tStP_r2t.shape[2])):
            cute.copy(
                thr_tmem_store, tSrP_r2t_f32[None, None, i], tStP_r2t[None, None, i]
            )
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = (
                    cute.size(tStP_r2t.shape[2])
                    * self.split_P_arrive
                    // self.n_block_size
                )
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        softmax.update_row_sum(tSrS_t2r.load(), acc_scale, is_first)
        pipeline_sm_stats.producer_acquire_w_index_phase(stage, sm_stats_producer_phase)
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_qk: cute.ThrMma,
        thr_mma_pv: cute.ThrMma,
        tStS: cute.Tensor,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        pipeline_o_epi: pipeline.PipelineAsync,
        softmax_scale_log2: Float32,
        SeqlenInfoCls: Callable,
        tile_scheduler: "TileSchedulerProtocol",
        mBlockNums: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4

        tScS = thr_mma_qk.partition_C(cute.make_identity_tensor(self.mma_tiler_qk[:2]))
        tStScale_layout = cute.composition(
            tStS.layout, cute.make_layout((self.m_block_size, 1))
        )
        tStScales = tuple(
            cute.make_tensor(
                tStS.iterator + self.tmem_vec_offset[stage], tStScale_layout
            )
            for stage in range(self.s_stage)
        )
        tScScale = cute.composition(tScS, cute.make_layout((self.m_block_size, 1)))
        tmem_load_v_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(1)), self.qk_acc_dtype
        )
        thr_tmem_load_vec = tcgen05.make_tmem_copy(
            tmem_load_v_atom, tStScales[0]
        ).get_slice(tidx)

        tStScales_t2r = [
            thr_tmem_load_vec.partition_S(tStScales[stage])
            for stage in range(self.s_stage)
        ]
        tSrScale_t2r_shape = thr_tmem_load_vec.partition_D(tScScale).shape

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            # For q_stage=1, always need row_max for combine; use -inf as default
            stats = [
                (
                    Float32(0.0),
                    (
                        -Float32.inf
                        if const_expr(mLSE is not None or self.q_stage == 1)
                        else None
                    ),
                    True,
                )
            ] * self.s_stage

            has_work = mBlockNums[batch_idx, head_idx, m_block] > Int32(0)
            lse_max_safe = Float32(0.0)
            lse_sum_combined = Float32(0.0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # Wait for both softmax warp groups.
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                tSrScale_t2r = cute.make_rmem_tensor(tSrScale_t2r_shape, Float32)
                # q_stage=1 correction loop
                block_iter_count = mBlockNums[
                    batch_idx, head_idx, m_block
                ]  # EVEN by builder
                corr_pair_count = (block_iter_count - 2) // 2
                # Rescale the two alternating stages as a pair.
                for i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        scale = sScale[tidx + stage * self.m_block_size]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if should_rescale:
                            self.correction_rescale(
                                thr_mma_pv,
                                tOtO[None, None, None, stage],
                                tidx,
                                scale,
                            )
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.s_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                # N even: no remainder. Release final sm_stats stage 1.
                pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Both sO stages are safe after the corresponding MMA work.

                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.m_block_size]
                    if const_expr(mLSE is not None or self.q_stage == 1):
                        row_max = sScale[
                            tidx
                            + stage * self.m_block_size
                            + self.s_stage * self.m_block_size
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    acc_O_mn_row_is_zero_or_nan = row_sum == 0.0 or row_sum != row_sum
                    stats[stage] = (row_sum, row_max, acc_O_mn_row_is_zero_or_nan)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, zero_or_nan0 = stats[0]
                row_sum1, row_max1, zero_or_nan1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # row_max is in original S space (unscaled). To compute rescale
                # factors we need exp2((row_max - max_combined) * scale_log2).
                # For empty/padding stages (zero_or_nan=True), row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if not zero_or_nan0 else -Float32.inf
                rm1 = row_max1 if not zero_or_nan1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = (
                    max_combined if max_combined != -Float32.inf else Float32(0.0)
                )
                scale0 = (
                    cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan0
                    else Float32(0.0)
                )
                scale1 = (
                    cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True)
                    if not zero_or_nan1
                    else Float32(0.0)
                )
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                combined_zero_or_nan = (
                    sum_combined <= 0.0 or sum_combined != sum_combined
                )
                if const_expr(mLSE is not None):
                    lse_max_safe = max_safe
                    lse_sum_combined = sum_combined
                inv_sum = cute.arch.rcp_approx(
                    sum_combined if not combined_zero_or_nan else 1.0
                )
                final_scale0 = scale0 * inv_sum
                final_scale1 = scale1 * inv_sum

                # Wait for both O accumulators from MMA warp
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_o_acc.consumer_wait_w_index_phase(
                        stage, o_corr_consumer_phase
                    )
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    final_scale0,
                    final_scale1,
                    sO[None, None, 0],
                )
                # Release both O buffers in tmem
                for stage in cutlass.range_constexpr(self.s_stage):
                    pipeline_s_p_o.consumer_release_w_index(stage)
                pipeline_o_epi.producer_commit_w_index(0)

                o_corr_consumer_phase ^= 1
                sm_stats_consumer_phase ^= 1
                corr_epi_producer_phase ^= 1
            else:
                # Empty tile (block_count == 0): sync pipelines and write O=0.
                # Match softmax's 1 barrier arrive per stage.
                for stage_idx in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(
                        index=stage_idx * 4 + warp_idx
                    )
                    pipeline_sm_stats.consumer_release_w_index(stage_idx)
                sm_stats_consumer_phase ^= 1
                # Write O=0 via correction_epilogue_combine with scale=0.
                # Note: reads tmem which may have values from a previous tile;
                # 0.0 * finite = 0.0. For the very first tile, tmem is hardware-zero-initialized.
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                self.correction_epilogue_combine(
                    thr_mma_pv,
                    tOtO[None, None, None, 0],
                    tOtO[None, None, None, 1],
                    tidx,
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                pipeline_o_epi.producer_commit_w_index(0)
                # o_corr_consumer_phase NOT toggled (pipeline_o_acc not touched)
                corr_epi_producer_phase ^= 1

            if const_expr(mLSE is not None):
                mLSE_cur = mLSE[None, head_idx, batch_idx]
                m_tile_idx = m_block
                gLSE = cute.local_tile(mLSE_cur, (self.m_block_size,), (m_tile_idx,))
                lse_invalid = (
                    lse_sum_combined <= 0.0 or lse_sum_combined != lse_sum_combined
                )
                lse_value = (
                    (
                        lse_max_safe * softmax_scale_log2
                        + cute.math.log2(lse_sum_combined, fastmath=True)
                    )
                    * math.log(2.0)
                    if not lse_invalid
                    else -Float32.inf
                )
                seqlen_q = seqlen.seqlen_q
                if tidx < seqlen_q - m_tile_idx * self.m_block_size:
                    gLSE[tidx] = lse_value

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # This is equivalent to pipeline_o_epi.consumer_tail() for the correction warps
        pipeline_o_epi.producer_acquire_w_index_phase(
            self.q_stage - 1, corr_epi_producer_phase
        )

    @cute.jit
    def correction_rescale(
        self,
        thr_mma: cute.ThrMma,
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale an O accumulator through CuTe TMEM copy atoms."""
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))
        corr_tile_size = 32
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tile_layout = cute.make_layout((self.m_block_size, corr_tile_size))
        tOtO_i = cute.logical_divide(tOtO, tile_layout)
        tOcO_i = cute.logical_divide(tOcO, tile_layout)
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_load_atom, tOtO_i[(None, None), 0]
        )
        tiled_tmem_store = tcgen05.make_tmem_copy(
            tmem_store_atom, tOtO_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        thr_tmem_store = tiled_tmem_store.get_slice(tidx)
        tOtO_t2r = thr_tmem_load.partition_S(tOtO_i[(None, None), None])
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        tOtO_r2t = thr_tmem_store.partition_D(tOtO_i[(None, None), None])

        num_fragments = self.head_dim_v_padded // corr_tile_size
        for i in cutlass.range_constexpr(num_fragments):
            tOrO_frg = cute.make_rmem_tensor(
                tOcO_t2r[None, 0, 0, i].shape,
                self.pv_acc_dtype,
            )
            cute.copy(
                tiled_tmem_load,
                tOtO_t2r[None, 0, 0, i],
                tOrO_frg,
            )
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            cute.copy(
                tiled_tmem_store,
                tOrO_frg,
                tOtO_r2t[None, 0, 0, i],
            )
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_combine(
        self,
        thr_mma: cute.ThrMma,
        tOtO0: cute.Tensor,
        tOtO1: cute.Tensor,
        tidx: Int32,
        scale0: Float32,
        scale1: Float32,
        sO: cute.Tensor,
    ):
        """Combine two partial O accumulators (O0, O1) from intra-warp overlap and write to smem/gmem.

        For q_stage=1 intra-warp overlap, O0 and O1 accumulate results from even and odd
        n_blocks respectively. This method reads both from TMEM, applies their respective
        scales (derived from combined softmax stats), adds them, and writes the result.
        """
        corr_tile_size = 8 * 32 // self.o_dtype.width
        tOsO = thr_mma.get_slice(0).partition_C(sO)
        tOcO = thr_mma.partition_C(cute.make_identity_tensor(self.mma_tiler_pv[:2]))

        tOtO0_i = cute.logical_divide(
            tOtO0, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOtO1_i = cute.logical_divide(
            tOtO1, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOcO_i = cute.logical_divide(
            tOcO, cute.make_layout((self.m_block_size, corr_tile_size))
        )
        tOsO_i = cute.logical_divide(
            tOsO, cute.make_layout((self.m_block_size, corr_tile_size))
        )

        epi_subtile = (self.epi_tile[0], corr_tile_size)
        tmem_copy_atom = sm100_utils_basic.get_tmem_load_op(
            self.mma_tiler_pv,
            self.o_layout,
            self.o_dtype,
            self.pv_acc_dtype,
            epi_subtile,
            False,
        )
        tiled_tmem_load = tcgen05.make_tmem_copy(
            tmem_copy_atom, tOtO0_i[(None, None), 0]
        )
        thr_tmem_load = tiled_tmem_load.get_slice(tidx)
        smem_copy_atom = sm100_utils_basic.get_smem_store_op(
            self.o_layout, self.o_dtype, self.pv_acc_dtype, tiled_tmem_load
        )
        tiled_smem_store = cute.make_tiled_copy_D(smem_copy_atom, tiled_tmem_load)

        tOtO0_t2r = thr_tmem_load.partition_S(tOtO0_i[(None, None), None])
        tOtO1_t2r = thr_tmem_load.partition_S(tOtO1_i[(None, None), None])
        tOsO_s2r = copy_utils.partition_D_position_independent(
            thr_tmem_load, tOsO_i[(None, None), None]
        )
        tOcO_t2r = thr_tmem_load.partition_D(tOcO_i[(None, None), None])
        for i in cutlass.range(
            self.head_dim_v_padded // corr_tile_size, unroll_full=True
        ):
            tOtO0_t2r_i = tOtO0_t2r[None, 0, 0, i]
            tOtO1_t2r_i = tOtO1_t2r[None, 0, 0, i]
            tOsO_r2s_i = tOsO_s2r[None, 0, 0, i]
            frg_shape = tOcO_t2r[None, 0, 0, i].shape
            tOrO0_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            tOrO1_frg = cute.make_rmem_tensor(frg_shape, self.pv_acc_dtype)
            # When both scales are 0 (empty tile), skip tmem reads to avoid 0*NaN=NaN.
            is_zero_output = scale0 == Float32(0.0) and scale1 == Float32(0.0)
            if not is_zero_output:
                cute.copy(tiled_tmem_load, tOtO0_t2r_i, tOrO0_frg)
                cute.copy(tiled_tmem_load, tOtO1_t2r_i, tOrO1_frg)
                # Combined: O = O0 * scale0 + O1 * scale1
                for j in cutlass.range(0, cute.size(tOrO0_frg), 2, unroll_full=True):
                    o0_a, o0_b = cute.arch.mul_packed_f32x2(
                        (tOrO0_frg[j], tOrO0_frg[j + 1]), (scale0, scale0)
                    )
                    o1_a, o1_b = cute.arch.mul_packed_f32x2(
                        (tOrO1_frg[j], tOrO1_frg[j + 1]), (scale1, scale1)
                    )
                    tOrO0_frg[j], tOrO0_frg[j + 1] = cute.arch.add_packed_f32x2(
                        (o0_a, o0_b), (o1_a, o1_b)
                    )
            else:
                tOrO0_frg.fill(Float32(0.0))
            copy_utils.cvt_copy(tiled_smem_store, tOrO0_frg, tOsO_r2s_i)
        cute.arch.fence_view_async_shared()

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        tile_scheduler: "TileSchedulerProtocol",
    ):
        epi_consumer_phase = Int32(0)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls()

            mO_cur = mO[None, None, None, batch_idx][None, None, head_idx]
            tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))  # (128, 128)
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )  # (128, 128, 1)
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O,
                0,
                cute.make_layout(1),
                sO,
                gO,
            )
            for stage in cutlass.range(self.q_stage, unroll_full=True):
                pipeline_o_epi.consumer_wait_w_index_phase(
                    stage,
                    epi_consumer_phase,
                )
                store_O(src_idx=stage, dst_idx=stage)
                cute.arch.cp_async_bulk_commit_group()
            for stage in cutlass.range_constexpr(self.q_stage):
                cute.arch.cp_async_bulk_wait_group(
                    self.q_stage - 1 - stage,
                    read=True,
                )
                pipeline_o_epi.consumer_release_w_index(stage)

            epi_consumer_phase ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()

    @cute.jit
    def load_KV_paged(
        self,
        tma_atom_X: cute.CopyAtom,
        gX: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sX_page_layout: cute.ComposedLayout,
        mPageTable: cute.Tensor,
        load_warp_rank: Int32,
        row: Int32,
        pipeline_kv: pipeline.PipelineAsync,
        producer_state: pipeline.PipelineState,
        K_or_V: Literal["K", "V"],
    ):
        """Fill one N128 K or V stage with two Page16 loader warps."""
        assert K_or_V in ("K", "V")
        pages_per_loader = FRAGS_PER_TILE // len(self.load_warp_ids)
        pg_frag = cute.make_rmem_tensor((pages_per_loader,), Int32)
        pg_row = cute.make_tensor(
            mPageTable.iterator
            + row * FRAGS_PER_TILE
            + load_warp_rank * pages_per_loader,
            cute.make_layout(pages_per_loader),
        )
        cute.autovec_copy(pg_row, pg_frag)

        pipeline_kv.producer_acquire(producer_state)
        tma_bar_ptr = pipeline_kv.producer_get_barrier(producer_state)
        stage_base = producer_state.index * self.kv_elems_per_stage
        for slot in cutlass.range_constexpr(pages_per_loader):
            sub = load_warp_rank * pages_per_loader + slot
            sX_sub = cute.make_tensor(
                cute.recast_ptr(
                    sKV_ptr + stage_base + sub * (PAGE * 64),
                    sX_page_layout.inner,
                ),
                sX_page_layout.outer,
            )
            tXsX, tXgX = cpasync.tma_partition(
                tma_atom_X,
                0,
                cute.make_layout(1),
                cute.group_modes(sX_sub, 0, 2),
                gX,
            )
            cute.copy(
                tma_atom_X,
                tXgX[None, pg_frag[slot]],
                tXsX[None],
                tma_bar_ptr=tma_bar_ptr,
            )


# =============================================================================
# Local Helper Namespace
# =============================================================================
def _make_local_namespace(*names: str):
    return types.SimpleNamespace(
        **{name: globals()[name] for name in names if name in globals()}
    )


# =============================================================================
# Sequence Metadata
# =============================================================================
@dataclass(frozen=True)
class SeqlenInfoQK:
    seqlen_q: Int32
    seqlen_k: Int32

    @staticmethod
    def create(
        seqlen_q_static: Int32,
        seqlen_k_static: Int32,
    ):
        return SeqlenInfoQK(
            seqlen_q_static,
            seqlen_k_static,
        )


@cute.jit
def gemm_ptx_partial(
    op: cute.nvgpu.tcgen05.mma.MmaOp,
    acc_tmem_addr: Int32,
    tCrA: cute.Tensor,
    tCrB: cute.Tensor,
    sA: Optional[cute.Tensor],
    sB: cute.Tensor,
    mbar_ptr: Optional[cutlass.Pointer] = None,
    mbar_phase: Optional[Int32] = None,
    split_arrive: Optional[int] = None,
    zero_init: bool | Boolean = False,
    tA_addr: Optional[Int32] = None,
) -> None:
    is_ts = op.a_src == cute.nvgpu.tcgen05.OperandSource.TMEM
    if const_expr(not is_ts):
        assert sA is not None, "sA must be provided when a_src is not TMEM"
    sA_layout = sA.layout if sA is not None else tCrA.layout
    sB_layout = sB.layout
    idesc: int = const_expr(sm100_desc.mma_op_to_idesc(op))
    if const_expr(not is_ts):
        sA_swizzle = sA.iterator.type.swizzle_type
        smem_desc_base_a: int = const_expr(
            sm100_desc.make_smem_desc_base(
                cute.recast_layout(128, op.a_dtype.width, sA_layout[0]),
                sA_swizzle,
                (
                    sm100_desc.Major.K
                    if const_expr(op.a_major_mode == cute.nvgpu.OperandMajorMode.K)
                    else sm100_desc.Major.MN
                ),
            )
        )
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
        smem_desc_base_a_lo = const_expr(smem_desc_base_a_lo)
        smem_desc_a_hi = const_expr(smem_desc_a_hi)
    else:
        smem_desc_base_a = None
        smem_desc_base_a_lo, smem_desc_a_hi = None, None
    sB_swizzle = sB.iterator.type.swizzle_type
    smem_desc_base_b: int = const_expr(
        sm100_desc.make_smem_desc_base(
            cute.recast_layout(128, op.b_dtype.width, sB_layout[0]),
            sB_swizzle,
            (
                sm100_desc.Major.K
                if const_expr(op.b_major_mode == cute.nvgpu.OperandMajorMode.K)
                else sm100_desc.Major.MN
            ),
        )
    )
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    smem_desc_base_b_lo = const_expr(smem_desc_base_b_lo)
    smem_desc_b_hi = const_expr(smem_desc_b_hi)

    tCrA_layout = (
        tCrA.layout
        if const_expr(not is_ts)
        else cute.recast_layout(32, tCrA.element_type.width, tCrA.layout)
    )
    offset_a = [
        cute.crd2idx((0, 0, k), tCrA_layout) for k in range(cute.size(tCrA.shape[2]))
    ]
    offset_b = [
        cute.crd2idx((0, 0, k), tCrB.layout) for k in range(cute.size(tCrB.shape[2]))
    ]
    offset_b_diff = [
        offset_b[k] - offset_b[k - 1] for k in range(1, cute.size(tCrB.shape[2]))
    ]

    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(
            smem_desc_base_a_lo
            | sm100_desc.make_smem_desc_start_addr(sA[None, None, 0].iterator)
        )
    else:
        smem_desc_start_a_lo = None
    smem_desc_start_b_lo = Int32(
        smem_desc_base_b_lo
        | sm100_desc.make_smem_desc_start_addr(sB[None, None, 0].iterator)
    )
    zero_init_is_dynamic = isinstance(zero_init, Boolean)
    pred_str = "p" if zero_init_is_dynamic else "0" if zero_init else "1"
    pred_input = zero_init if zero_init_is_dynamic else not zero_init
    pred_setp = "setp.eq.b32" if zero_init_is_dynamic else "setp.ne.b32"
    if const_expr(not is_ts):
        assert mbar_ptr is None, "mbar_ptr must be None when a_src is not TMEM"
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value(),
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(pred_input).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_a_lo_start, smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_a, smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            "mov.b32 smem_desc_a_lo_start, $0;\n\t"
            "mov.b32 smem_desc_b_lo_start, $1;\n\t"
            f"mov.b32 smem_desc_a_hi, {hex(smem_desc_a_hi)};\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_a, {{smem_desc_a_lo_start, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_a_lo, smem_desc_a_lo_start, {hex(offset_a[k])};\n\t"
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_a, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    "@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], smem_desc_a, smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(1, cute.size(tCrA.shape[2]))
            )
            + "}\n",
            "r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )
    else:
        # For TS gemm, somehow tCrA.iterator.toint() returns 0 no matter what, so we need to
        # explicitly pass in the tA_addr for correctness.
        tA_addr = tCrA[None, None, 0].iterator.toint() if tA_addr is None else tA_addr
        input_args = [
            Int32(cute.arch.make_warp_uniform(tA_addr)).ir_value(),
            Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
            Int32(pred_input).ir_value(),
            Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
        ]
        if const_expr(mbar_ptr is not None):
            assert mbar_phase is not None, (
                "mbar_phase must be provided when mbar_ptr is not None"
            )
            assert split_arrive is not None, (
                "split_arrive must be provided when mbar_ptr is not None"
            )
            assert split_arrive % op.shape_mnk[2] == 0, (
                "split_arrive must be a multiple of the MMA K extent"
            )
            split_arrive_idx = split_arrive // op.shape_mnk[2]
            assert 1 <= split_arrive_idx <= cute.size(tCrA.shape[2]), (
                "split_arrive must map to a K-tile index within [1, num_k_tiles]"
            )
            input_args.append(mbar_ptr.toint().ir_value())
            input_args.append(Int32(mbar_phase).ir_value())
            mbar_wait_str = (
                ".reg .pred P1; \n\t"
                "LAB_WAIT: \n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 P1, [$4], $5, 10000000; \n\t"
                "@P1 bra DONE; \n\t"
                "bra     LAB_WAIT; \n\t"
                "DONE: \n\t"
            )
        else:
            mbar_wait_str = ""
        llvm.inline_asm(
            None,
            input_args,
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 idesc;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 tmem_a;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_b_hi;\n\t"
            ".reg .b64 smem_desc_b;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 idesc, {hex(idesc)};\n\t"
            f"mov.b32 tmem_acc, $3;\n\t"
            f"mov.b32 tmem_a, $0;\n\t"
            f"mov.b32 smem_desc_b_lo_start, $1;\n\t"
            # The post-wait loop below updates smem_desc_b_lo incrementally, and the
            # pre-wait loop that would otherwise seed it is empty when
            # split_arrive_idx == 1, so initialize it from the base descriptor.
            + (
                "mov.b32 smem_desc_b_lo, smem_desc_b_lo_start;\n\t"
                if mbar_ptr is not None
                else ""
            )
            + f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 smem_desc_b, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            f"{pred_setp} p, $2, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a], smem_desc_b, idesc, {pred_str};\n\t"
            + "".join(
                (
                    f"add.u32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                )
                for k in range(
                    1,
                    (
                        cute.size(tCrA.shape[2])
                        if const_expr(mbar_ptr is None)
                        else split_arrive_idx
                    ),
                )
            )
            + mbar_wait_str
            + (
                "".join(
                    (
                        f"add.u32 smem_desc_b_lo, smem_desc_b_lo, {hex(offset_b_diff[k - 1])};\n\t"
                        f"mov.b64 smem_desc_b, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                        f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], [tmem_a + {hex(offset_a[k])}], smem_desc_b, idesc, 1;\n\t"
                    )
                    for k in range(split_arrive_idx, cute.size(tCrA.shape[2]))
                )
                if const_expr(mbar_ptr is not None)
                else ""
            )
            + "}\n",
            "r,r,r,r" if const_expr(mbar_ptr is None) else "r,r,r,r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_smem_desc(
    smem_desc_start_a: Int32,  # If TS, then this is the tmem start address for A
    smem_desc_base_a: Optional[int],
    tCrA_layout: cute.Layout,
    var_name_prefix: str = "smem_desc",
) -> None:
    is_ts = const_expr(smem_desc_base_a is None)
    num_k_tile = cute.size(tCrA_layout.shape[2])
    smem_desc_base_a_lo, smem_desc_a_hi = None, None
    if const_expr(not is_ts):
        smem_desc_base_a_lo, smem_desc_a_hi = i64_to_i32x2(smem_desc_base_a)
    tCrA_layout = (
        tCrA_layout
        if const_expr(not is_ts)
        # currently hard-coding the width to 16
        else cute.recast_layout(32, 16, tCrA_layout)
    )
    offset_a = [cute.crd2idx((0, 0, k), tCrA_layout) for k in range(num_k_tile)]
    smem_desc_start_a_lo = None
    if const_expr(not is_ts):
        smem_desc_start_a_lo = Int32(smem_desc_base_a_lo | smem_desc_start_a)
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [Int32(cute.arch.make_warp_uniform(smem_desc_start_a_lo)).ir_value()],
            f".reg .b32 {var_name_prefix}_lo;\n\t"
            f".reg .b64 {var_name_prefix}_<{num_k_tile}>;\n\t"
            f"mov.b64 {var_name_prefix}_0, {{$0, {hex(smem_desc_a_hi)}}};\n\t"
            + "".join(
                (
                    f"add.s32 {var_name_prefix}_lo, $0, {hex(offset_a[k])};\n\t"
                    f"mov.b64 {var_name_prefix}_{k}, {{{var_name_prefix}_lo, {hex(smem_desc_a_hi)}}};\n\t"
                )
                for k in range(1, num_k_tile)
            ),
            "r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


@cute.jit
def declare_ptx_idesc(
    op: cute.nvgpu.tcgen05.mma.MmaOp, var_name: str = "idesc"
) -> None:
    idesc = const_expr(sm100_desc.mma_op_to_idesc(op))
    llvm.inline_asm(
        None,
        [],
        f".reg .b32 {var_name};\n\t"  # noqa
        f"mov.b32 {var_name}, {hex(idesc)};\n\t",
        constraints="",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
    )


@cute.jit
def gemm_ptx_precomputed_varname(
    acc_tmem_addr: Int32,
    smem_desc_start_b: Int32,
    smem_desc_base_b: int,
    tCrB_layout: cute.Layout,
    smem_var_name_prefix: str,
    idesc_var_name: str,
    smem_offset: int,
    zero_init: bool | Boolean = False,
) -> None:
    is_ts = False
    num_k_tile = cute.size(tCrB_layout.shape[2])
    smem_desc_base_b_lo, smem_desc_b_hi = i64_to_i32x2(smem_desc_base_b)
    offset_b = [cute.crd2idx((0, 0, k), tCrB_layout) for k in range(num_k_tile)]

    smem_desc_start_b_lo = Int32(smem_desc_base_b_lo | smem_desc_start_b)
    zero_init_is_dynamic = isinstance(zero_init, Boolean)
    pred_str = "p" if zero_init_is_dynamic else "0" if zero_init else "1"
    pred_input = zero_init if zero_init_is_dynamic else not zero_init
    pred_setp = "setp.eq.b32" if zero_init_is_dynamic else "setp.ne.b32"
    if const_expr(not is_ts):
        llvm.inline_asm(
            None,
            [
                Int32(cute.arch.make_warp_uniform(smem_desc_start_b_lo)).ir_value(),
                Int32(pred_input).ir_value(),
                Int32(cute.arch.make_warp_uniform(acc_tmem_addr)).ir_value(),
            ],
            "{\n\t"
            ".reg .pred leader_thread;\n\t"
            ".reg .pred p;\n\t"
            ".reg .b32 tmem_acc;\n\t"
            ".reg .b32 smem_desc_b_lo_start;\n\t"
            ".reg .b32 smem_desc_a_lo, smem_desc_b_lo;\n\t"
            ".reg .b32 smem_desc_a_hi, smem_desc_b_hi;\n\t"
            f".reg .b64 smem_desc_b_<{num_k_tile}>;\n\t"
            "elect.sync _|leader_thread, -1;\n\t"
            f"mov.b32 tmem_acc, $2;\n\t"
            "mov.b32 smem_desc_b_lo_start, $0;\n\t"
            f"mov.b32 smem_desc_b_hi, {hex(smem_desc_b_hi)};\n\t"
            f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_0;\n\t"
            f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
            f"mov.b64 {smem_var_name_prefix}_0, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
            f"mov.b64 smem_desc_b_0, {{smem_desc_b_lo_start, smem_desc_b_hi}};\n\t"
            + "".join(
                (
                    f"mov.b64 {{smem_desc_a_lo, smem_desc_a_hi}}, {smem_var_name_prefix}_{k};\n\t"
                    f"add.s32 smem_desc_a_lo, smem_desc_a_lo, {smem_offset};\n\t"
                    f"add.s32 smem_desc_b_lo, smem_desc_b_lo_start, {hex(offset_b[k])};\n\t"
                    f"mov.b64 {smem_var_name_prefix}_{k}, {{smem_desc_a_lo, smem_desc_a_hi}};\n\t"
                    f"mov.b64 smem_desc_b_{k}, {{smem_desc_b_lo, smem_desc_b_hi}};\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + f"{pred_setp} p, $1, 0;\n\t"
            f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], {smem_var_name_prefix}_0, smem_desc_b_0, {idesc_var_name}, {pred_str};\n\t"
            + "".join(
                (
                    f"@leader_thread tcgen05.mma.cta_group::1.kind::f16 [tmem_acc], {smem_var_name_prefix}_{k}, smem_desc_b_{k}, {idesc_var_name}, 1;\n\t"
                )
                for k in range(1, num_k_tile)
            )
            + "}\n",
            "r,r,r",
            has_side_effects=True,
            is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT,
        )


sm100_utils = _make_local_namespace(
    "declare_ptx_smem_desc",
    "declare_ptx_idesc",
    "gemm_ptx_partial",
    "gemm_ptx_precomputed_varname",
)
