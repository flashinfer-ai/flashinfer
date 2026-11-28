# ruff: noqa: F841, B007
# Preserve upstream CuTe DSL constructs during the KPage extraction.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# KPage M64 x N256 forward for blk128 sparse metadata, derived from the SM100
# blk64 forward (sm100_blk64/bsa_fwd_sm100.py).  Runs the Q blocks that have
# at most 64 valid rows: a 64-row Q tile against 256-row K/V stages gathered
# from sixteen 16-row pages, same page tables and packed keep masks as
# bsa_fwd_sm100_kpage.py.  BF16, non-causal, MHA, head_dim 128.
# Based on the cutlass example and cute-dsl example:
# https://github.com/NVIDIA/cutlass/tree/main/examples/77_blackwell_fmha
# https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/blackwell/fmha.py

import math
from functools import partial
from typing import Callable, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.cute.nvgpu.tcgen05 as tcgen05
import cutlass.pipeline as cutlass_pipeline
from cutlass import Boolean, Float32, Int32, Int64, const_expr, pipeline
from cutlass.cute.nvgpu import cpasync
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
from flashinfer.experimental.vsa_kpage.csrc.utils import copy_utils, layout_utils
from flashinfer.experimental.vsa_kpage.csrc.utils import kernel_utils as utils
from flashinfer.experimental.vsa_kpage.csrc.utils import pipeline as pipeline_custom
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
from flashinfer.experimental.vsa_kpage.csrc.utils.seqlen_info import SeqlenInfoQK
from flashinfer.experimental.vsa_kpage.csrc.utils.softmax import SoftmaxSm100
from flashinfer.experimental.vsa_kpage.csrc.utils.tile_scheduler import (
    SchedulingMode,
    TileSchedulerArguments,
)

from flashinfer.experimental.vsa_kpage.csrc.fwd.sm100_blk64 import bsa_fwd_helpers
from flashinfer.experimental.vsa_kpage.csrc.utils import kpage_utils
from flashinfer.experimental.vsa_kpage.csrc.utils.kpage_utils import PAGE

FRAGS_PER_HALF = 8


class BlockSparseAttnForwardSm100KPageQ64:
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
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.head_dim_padded = int(
            math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of
        )
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.head_dim_v_padded = int(
            math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of
        )
        assert head_dim == 128 and head_dim_v == 128, (
            "page smem slices assume head_dim 128 (BF16, SW128)"
        )
        self.m_block_size = 64
        self.sparse_block_size = 64
        self.sparse_blocks_per_kv = 4
        self.n_block_size = 256
        self.kv_mma_k = 128
        self.kv_elems_per_stage = self.n_block_size * self.kv_mma_k
        self.q_stage = 1
        # If split_P_arrive, the softmax warps write some columns of P first, signal to the MMA warp
        # to being the P @ V MMA, then write the rest of P and signal again. This allows some overlap
        # between compute the last couple columns of P and the P @ V MMA.
        # Match C++ blk64 kSplitNumer/kSplitDenom = 1/4:
        # softmax releases the first 32 P columns early, then MMA waits on
        # p_lastsplit before consuming the remaining P fragments.
        self.split_P_arrive = 32
        self.arch = BaseDSL._get_dsl().get_arch_enum()
        assert self.arch.is_family_of(Arch.sm_100f) or self.arch.is_family_of(
            Arch.sm_110f
        ), "Only SM 10.x and 11.x are supported"

        self.cta_tiler = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        self.mma_tiler_qk = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_padded,
        )
        # WS TS PV follows the C++ blk64 trait: M=64, N=256, K=128.
        # The final 128 output columns are recovered by the correction epilogue's
        # WS TMEM load/store mapping.
        self.mma_tiler_pv = (
            self.m_block_size,
            self.n_block_size,
            self.head_dim_v_padded,
        )
        self.qk_acc_dtype = Float32
        self.pv_acc_dtype = Float32
        self.sched_stages = 1
        self.scheduling_mode = SchedulingMode.CLC
        self.kv_stage = 3

        self.softmax0_warp_ids = (0, 1, 2, 3)
        self.softmax1_warp_ids = (4, 5, 6, 7)
        self.correction_warp_ids = (8, 9, 10, 11)
        self.stats_stride = cute.arch.WARP_SIZE * len(self.correction_warp_ids)
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
        # Match C++ blk64: two 128-column S/P stages and two 128-column O stages.
        self.tmem_s_offset = [0, 128]
        self.tmem_o_offset = [256, 384]
        self.tmem_total = 512
        assert self.tmem_total <= self.tmem_alloc_cols
        self.tmem_p_offset = self.tmem_s_offset
        self.tmem_s_to_p_offset = 0
        self.epi_exchange_f32_count = 4 * 4 * 32 * 32
        # Filled in once the input dtype is known.  This is an offset in
        # elements of the shared K/V allocation, not an offset in bytes.
        self.epi_sO_input_offset = self.epi_exchange_f32_count * 2

        # vec buffer for row_max & row_sum
        self.tmem_vec_offset = self.tmem_s_offset

        self.num_regs_softmax = 176
        self.num_regs_correction = 88
        self.num_regs_other = 56
        assert all(
            value % 8 == 0
            for value in (
                self.num_regs_softmax,
                self.num_regs_correction,
                self.num_regs_other,
            )
        )

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        """Set up configurations and parameters for the FMHA kernel operation.

        This method initializes and configures various attributes required for the
        execution of the fused multi-head attention kernel, mainly about the pipeline stages:

        - Sets up staging parameters for Q, K, V inputs and accumulator data
        - Configures pipeline stages for softmax, correction, and epilogue operations
        """

        assert self.s_stage >= self.q_stage

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,  # (b, h, s_q, d)
        mK: cute.Tensor,  # (b_k, h_k, s_k, d), partial Page16 suffixes zero
        mV: cute.Tensor,  # (b_k, h_k, s_k, dv), partial Page16 suffixes zero
        mO: cute.Tensor,  # (b, h, s_q, dv)
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mPageTable: cute.Tensor,  # (total_half_tiles, 8), int32 page ids
        mMaskTable: cute.Tensor,  # (total_half_tiles, 4), packed keep masks
        mTileOffsets: cute.Tensor,  # (batch, heads, 2 * num_q_blocks), half-row units
        mFullIterCounts: cute.Tensor,  # unmasked N256 prefix length
        mBlockNums: cute.Tensor,  # (batch, heads, 2 * num_q_blocks), even N256 iterations
        mWorkIndices: Optional[cute.Tensor],  # static persistent work order
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
        self.epi_sO_input_offset = self.epi_exchange_f32_count * (
            Float32.width // self.k_dtype.width
        )
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        Q_layout_transpose = [2, 3, 1, 0]
        mQ_seq = cute.make_tensor(
            mQ.iterator, cute.select(mQ.layout, mode=Q_layout_transpose)
        )
        mQ = mQ_seq
        # (s_k, d, h_k, b_k)
        KV_layout_transpose = [2, 3, 1, 0]
        mK_seq, mV_seq = [
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
        k_dim_half = self.head_dim_padded // 2
        v_dim_part = self.head_dim_v_padded // 2
        mK = mK_seq
        mV = cute.make_tensor(
            mV_seq.iterator, cute.select(mV_seq.layout, mode=[1, 0, 2, 3])
        )

        # check type consistency
        if const_expr(self.q_dtype != self.k_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.k_dtype}")
        if const_expr(self.q_dtype != self.v_dtype):
            raise TypeError(f"Type mismatch: {self.q_dtype} != {self.v_dtype}")
        if const_expr(self.q_dtype != cutlass.BFloat16):
            raise TypeError("blk64 forward requires BF16 Q/K/V")
        if const_expr(self.o_dtype != cutlass.BFloat16):
            raise TypeError("blk64 forward requires BF16 output")
        self._setup_attributes()
        q_major_mode = cute.nvgpu.OperandMajorMode.K
        k_major_mode = cute.nvgpu.OperandMajorMode.K
        v_major_mode = cute.nvgpu.OperandMajorMode.MN
        self.o_layout = LayoutEnum.from_tensor(mO)
        # the intermediate tensor p is from tmem & mK-major
        p_source = tcgen05.OperandSource.TMEM
        p_major_mode = cute.nvgpu.OperandMajorMode.K
        # The public tcgen05 tiled-MMA builder does not expose mma.ws. Use this
        # object for CuTe fragment
        # partitioning and issue the WS instruction in ws_qk_gemm below.
        tiled_mma_qk = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            q_major_mode,
            k_major_mode,
            self.qk_acc_dtype,
            tcgen05.CtaGroup.ONE,
            self.mma_tiler_qk[:2],
        )
        tiled_mma_pv = sm100_utils_basic.make_trivial_tiled_mma(
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
        sQ_tma_layout = None
        sK_layout = sm100_utils_basic.make_smem_layout_b(
            tiled_mma_qk, self.mma_tiler_qk, self.k_dtype, self.kv_stage
        )
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            tiled_mma_pv, self.mma_tiler_pv, self.q_dtype, self.s_stage
        )
        # C++ SmemLayoutVDual in the MMA-fragment shape expected by CuTeDSL.
        # The K-groups must stay nested as (4,2): the second dim half jumps by
        # 16384 elements, while groups inside one half advance by 1024.
        sV_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (
                    ((v_dim_part, self.sparse_blocks_per_kv), 16),
                    1,
                    (4, 2),
                    self.kv_stage,
                ),
                stride=(
                    ((1, v_dim_part * self.sparse_block_size), v_dim_part),
                    0,
                    (
                        16 * v_dim_part,
                        v_dim_part * self.sparse_block_size * self.sparse_blocks_per_kv,
                    ),
                    self.kv_elems_per_stage,
                ),
            ),
        )
        # Exact C++ SmemLayoutKWide:
        #   Sw<3,4,3> o _0 o (_64,_64,_2):(_64,_1,_16384)
        # A sparse sub-block advances by 4096 elements, while the second
        # dim half jumps by the full 256-token half stride.
        sK_tma_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (self.sparse_block_size, k_dim_half, 2),
                stride=(k_dim_half, 1, self.n_block_size * k_dim_half),
            ),
        )
        # Exact C++ SmemLayoutVWide:
        #   Sw<3,4,3> o _0 o (_64,_64,_2):(_1,_64,_4096)
        # It writes both 64-dim halves for one sparse 64-token block while
        # preserving the offsets consumed by the full VDual PV layout.
        sV_tma_layout = cute.make_composed_layout(
            cute.make_swizzle(3, 4, 3),
            0,
            cute.make_layout(
                (v_dim_part, self.sparse_block_size, 2),
                stride=(
                    1,
                    v_dim_part,
                    v_dim_part * self.sparse_block_size,
                ),
            ),
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

        # K/V TMA atoms move one 16-row page; an N256 stage is sixteen pages.
        # The stage barrier still expects the full 64 KiB transfer.
        npages = cute.size(mK.shape[0]) // PAGE
        VK = mK
        if const_expr(not self.tail_page):
            VK = cute.make_tensor(
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
        sK_page_layout = cute.make_composed_layout(
            sK_tma_layout.inner,
            0,
            cute.make_layout(
                (PAGE, (k_dim_half, 2)),
                stride=(k_dim_half, (1, self.n_block_size * k_dim_half)),
            ),
        )
        tma_atom_K, mK = cpasync.make_tiled_tma_atom(
            tma_load_op,
            VK,
            sK_page_layout,
            (PAGE, self.head_dim_padded),
        )

        VV = mV
        if const_expr(not self.tail_page):
            VV = cute.make_tensor(
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
        sV_page_layout = cute.make_composed_layout(
            sV_tma_layout.inner,
            0,
            cute.make_layout(
                ((v_dim_part, 2), PAGE),
                stride=((1, v_dim_part * self.sparse_block_size), v_dim_part),
            ),
        )
        tma_atom_V, mV = cpasync.make_tiled_tma_atom(
            tma_load_op,
            VV,
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
            cute.size(
                mBlockNums.shape[2]
            ),  # 2 * num_q_blocks: the stride of the work IDs and walk tables, not ceil(seqlen_q / 64)
            cute.size(mQ.shape[2]),
            cute.size(mQ.shape[3]),
            num_splits,
            cute.size(mK.shape[0]),
            mQ.shape[1],
            self.head_dim_v_padded,
            total_q=cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=self.cta_tiler[:2],
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=True,
            is_split_kv=False,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(
            tile_sched_args,
            scheduling_mode=self.scheduling_mode,
            num_work=cute.size(mWorkIndices),
        )
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)

        sO_size = cute.cosize(sO_layout)
        sQ_size = cute.cosize(sQ_layout)
        # K/V storage is reused by the correction exchange and O epilogue.
        sO_size_in_k_elements = (
            sO_size * self.o_dtype.width + self.k_dtype.width - 1
        ) // self.k_dtype.width
        sKV_size = max(
            cute.cosize(sK_layout),
            self.epi_sO_input_offset + sO_size_in_k_elements,
        )

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
            # Per warp-pair reduction barriers for correction exchange.
            reduce_mbar: cute.struct.MemRange[Int64, 2]
            # Smem tensors
            # store row max and row sum
            sScale: cute.struct.MemRange[Float32, self.s_stage * self.stats_stride * 2]
            oStats: cute.struct.MemRange[Float32, 4 * 32 * 2]
            # CLC buffers (mbarriers + response storage)
            clc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, clc_mbar_size]
            clc_response: cute.struct.Align[
                cute.struct.MemRange[Int32, clc_response_size],
                16,
            ]
            sQ: cute.struct.Align[
                cute.struct.MemRange[self.q_dtype, sQ_size], self.buffer_align_bytes
            ]
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, sKV_size],
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
            sQ_tma_layout,
            sK_layout,
            sK_tma_layout,
            tP_layout,
            sV_layout,
            sV_tma_layout,
            sO_layout,
            tiled_mma_qk,
            tiled_mma_pv,
            tile_sched_params,
            mPageTable,
            mMaskTable,
            mTileOffsets,
            mFullIterCounts,
            mWorkIndices,
            mBlockNums,
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
        mK: cute.Tensor,  # Rank-5 Int64 or rank-6 sparse-block view
        mV: cute.Tensor,  # Rank-5 Int64 or rank-6 sparse-block view
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_O: cute.CopyAtom,
        softmax_scale_log2: Float32,
        softmax_scale: Float32 | None,
        sQ_layout: cute.ComposedLayout,
        sQ_tma_layout: Optional[cute.ComposedLayout],
        sK_layout: cute.ComposedLayout,
        sK_tma_layout: cute.ComposedLayout,
        tP_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sV_tma_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tile_sched_params: ParamsBase,
        mPageTable: cute.Tensor,
        mMaskTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mFullIterCounts: cute.Tensor,
        mWorkIndices: Optional[cute.Tensor],
        mBlockNums: cute.Tensor,
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
        # Tensor memory allocator
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
        reduce_mbar_ptr = storage.reduce_mbar.data_ptr()
        pipeline_o_epi = pipeline_custom.PipelineAsync.create(
            barrier_storage=storage.mbar_O_epi.data_ptr(),
            num_stages=self.s_stage,
            producer_group=correction_threads,
            consumer_group=epilogue_threads,
            defer_sync=True,
        )

        if warp_idx == 0:
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(reduce_mbar_ptr + 0, cute.arch.WARP_SIZE * 2)
                cute.arch.mbarrier_init(reduce_mbar_ptr + 1, cute.arch.WARP_SIZE * 2)

        # Fence pipeline barrier initialization before any warp uses it.
        pipeline_init_arrive(is_relaxed=True)

        #  Generate smem tensor Q/K/V/O
        # (MMA, MMA_Q, MMA_D, PIPE)
        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner)
        sKV_ptr = storage.sK.data_ptr()
        k_dim_half = self.head_dim_padded // 2
        v_dim_part = self.head_dim_v_padded // 2
        sK_page_layout = cute.make_composed_layout(
            sK_tma_layout.inner,
            0,
            cute.make_layout(
                (PAGE, (k_dim_half, 2)),
                stride=(k_dim_half, (1, self.n_block_size * k_dim_half)),
            ),
        )
        sV_page_layout = cute.make_composed_layout(
            sV_tma_layout.inner,
            0,
            cute.make_layout(
                ((v_dim_part, 2), PAGE),
                stride=((1, v_dim_part * self.sparse_block_size), v_dim_part),
            ),
        )
        # (MMA, MMA_K, MMA_D, PIPE)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        # (MMA, MMA_K, MMA_D, PIPE)
        # Strip swizzle info to reuse smem
        sV = cute.make_tensor(
            cute.recast_ptr(sKV_ptr, sV_layout.inner), sV_layout.outer
        )
        # Match the C++ blk64 SharedStorage union: epilogue exchange/sO reuses
        # the large KV buffer after the mainloop has finished consuming it.
        sO = cute.make_tensor(
            cute.recast_ptr(
                sKV_ptr + self.epi_sO_input_offset, sO_layout.inner, self.o_dtype
            ),
            sO_layout.outer,
        )

        sScale = storage.sScale.get_tensor(
            cute.make_layout(self.s_stage * self.stats_stride * 2)
        )
        oStats = storage.oStats.get_tensor(cute.make_layout(4 * 32 * 2))
        oExchange = cute.make_tensor(
            cute.recast_ptr(sKV_ptr, dtype=Float32),
            cute.make_layout(self.epi_exchange_f32_count),
        )

        thr_mma_qk = tiled_mma_qk.get_slice(0)
        thr_mma_pv = tiled_mma_pv.get_slice(0)

        pv_acc_shape = thr_mma_pv.partition_shape_C(self.mma_tiler_pv[:2])
        tOtO_base = thr_mma_pv.make_fragment_C(pv_acc_shape)
        tOtO = cute.make_tensor(
            tOtO_base.iterator + self.tmem_o_offset[0],
            cute.append(
                tOtO_base.layout,
                cute.make_layout(
                    (self.s_stage,),
                    stride=(self.tmem_o_offset[1] - self.tmem_o_offset[0],),
                ),
            ),
        )
        p_frag_shape = thr_mma_pv.partition_shape_A((self.m_block_size, self.kv_mma_k))
        tOrP_base = thr_mma_pv.make_fragment_A(p_frag_shape)
        # Need to multiply by width ratio bc tP is in v_dtype but tmem offsets are in FP32
        tP_width_ratio = Float32.width // self.v_dtype.width
        # Need to adjust the stage stride manually since the two stages aren't contiguous in tmem
        tP_stage_stride = (
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        ) * tP_width_ratio
        tOrP = cute.make_tensor(
            tOrP_base.iterator + self.tmem_p_offset[0] * tP_width_ratio,
            cute.append(
                tOrP_base.layout,
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
        num_clc_consumer_warps = self.threads_per_cta // cute.arch.WARP_SIZE
        clc_pipeline_consumer_group = cutlass_pipeline.CooperativeGroup(
            cutlass_pipeline.Agent.Thread,
            cute.arch.WARP_SIZE * num_clc_consumer_warps,
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

        # Synchronize the CTA before tensor memory allocation.
        pipeline_init_wait()

        # ///////////////////////////////////////////////////////////////////////////////
        #  EMPTY / CLC SCHEDULER WARP
        # ///////////////////////////////////////////////////////////////////////////////
        # ///////////////////////////////////////////////////////////////////////////////
        #  LOAD
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx >= self.load_warp_ids[0] and warp_idx <= self.load_warp_ids[-1]:
            cute.arch.setmaxregister_decrease(self.num_regs_other)
            self.load(
                thr_mma_qk,
                thr_mma_pv,
                mQ,
                mK,
                mV,
                sQ,
                sKV_ptr,
                sQ_tma_layout,
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
                tOtO,
                tOrP,
                pipeline_q,
                pipeline_kv,
                pipeline_s_p_o,
                pipeline_p_lastsplit,
                pipeline_o_acc,
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
                sScale=sScale,
                mLSE=mLSE,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
                tile_scheduler=tile_scheduler,
                mMaskTable=mMaskTable,
                mTileOffsets=mTileOffsets,
                mFullIterCounts=mFullIterCounts,
                mBlockNums=mBlockNums,
            )

            # Keep stage constexpr, matching the C++ template<int Stage> path.
            if warp_idx < self.softmax1_warp_ids[0]:
                softmax_loop(stage=0)
            if (
                warp_idx < self.correction_warp_ids[0]
                and warp_idx >= self.softmax1_warp_ids[0]
            ):
                softmax_loop(stage=1)

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
                thr_mma_pv,
                tOtO,
                sScale,
                mLSE,
                sO,
                pipeline_s_p_o,
                pipeline_o_acc,
                pipeline_sm_stats,
                sm_stats_barrier,
                reduce_mbar_ptr,
                pipeline_o_epi,
                (
                    Float32(1.0)
                    if const_expr(self.nonpositive_scale)
                    else softmax_scale_log2
                ),
                oStats,
                oExchange,
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
        thr_mma_pv: cute.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        sQ: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sQ_tma_layout: Optional[cute.ComposedLayout],
        sK_page_layout: cute.ComposedLayout,
        sV_page_layout: cute.ComposedLayout,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        clc_pipeline: Optional[cutlass_pipeline.PipelineClcFetchAsync],
        tile_scheduler: TileSchedulerProtocol,
        mPageTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mBlockNums: cute.Tensor,
    ):
        num_load_threads = len(self.load_warp_ids) * cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
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
        tiler_gQ = ((self.mma_tiler_qk[0] * self.q_stage), self.head_dim_padded)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            mQ_cur = mQ[None, None, None, batch_idx][None, None, head_idx]
            gQ = cute.local_tile(mQ_cur, tiler_gQ, (m_block, 0))
            gQ = layout_utils.select(
                cute.flat_divide(gQ, (self.mma_tiler_qk[0],)), mode=[0, 2, 1]
            )
            tSgQ = thr_mma_qk.partition_A(gQ)
            load_Q_fn, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q, 0, cute.make_layout(1), tSgQ, sQ
            )

            head_idx_kv = head_idx // self.qhead_per_kvhead
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
                    mK[None, None, None, head_idx_kv, batch_idx], 0, 2
                )
                gV_tma = cute.group_modes(
                    mV[None, None, None, head_idx_kv, batch_idx], 0, 2
                )
            block_iter_count = mBlockNums[batch_idx, head_idx, m_block]
            process_tile = block_iter_count > Int32(0)
            tile_base = mTileOffsets[batch_idx, head_idx, m_block]

            if process_tile:
                # Match C++ blk64: issue the one-shot Q TMA first, then K[N-1]/K[N-2].
                if warp_idx == self.load_warp_ids[0]:
                    pipeline_q.producer_acquire_w_index_phase(0, q_producer_phase)
                    tma_bar_ptr = pipeline_q.sync_object_full.get_barrier(0)
                    load_Q_fn(src_idx=0, dst_idx=0, tma_bar_ptr=tma_bar_ptr)
                q_producer_phase ^= 1

                kv_producer_state = self.load_KV_pages(
                    tma_atom_K,
                    gK_tma,
                    sKV_ptr,
                    sK_page_layout,
                    mPageTable,
                    tile_base + (block_iter_count - 1) * 2,
                    kv_producer_state,
                    pipeline_kv,
                    load_warp_rank=load_warp_rank,
                    K_or_V="K",
                )
                # q_stage=1 intra-warp overlap
                # Load order: Q, K[N-1], K[N-2], {V[N-1-i], K[N-3-i]}x(N-2), V[1], V[0]
                block_loop_count = block_iter_count - 2

                # Prologue: K[N-2]
                kv_producer_state = self.load_KV_pages(
                    tma_atom_K,
                    gK_tma,
                    sKV_ptr,
                    sK_page_layout,
                    mPageTable,
                    tile_base + (block_iter_count - 2) * 2,
                    kv_producer_state,
                    pipeline_kv,
                    load_warp_rank=load_warp_rank,
                    K_or_V="K",
                )

                # Flat main loop: N-2 iterations, each loads V then K
                for i in cutlass.range(block_loop_count, unroll=1):
                    # V[N-1-i]: V for the block whose S was computed earlier
                    kv_producer_state = self.load_KV_pages(
                        tma_atom_V,
                        gV_tma,
                        sKV_ptr,
                        sV_page_layout,
                        mPageTable,
                        tile_base + (block_iter_count - 1 - i) * 2,
                        kv_producer_state,
                        pipeline_kv,
                        load_warp_rank=load_warp_rank,
                        K_or_V="V",
                    )
                    # K[N-3-i]: K for the next QK GEMM
                    kv_producer_state = self.load_KV_pages(
                        tma_atom_K,
                        gK_tma,
                        sKV_ptr,
                        sK_page_layout,
                        mPageTable,
                        tile_base + (block_iter_count - 3 - i) * 2,
                        kv_producer_state,
                        pipeline_kv,
                        load_warp_rank=load_warp_rank,
                        K_or_V="K",
                    )

                # Epilogue: last 2 V loads
                kv_producer_state = self.load_KV_pages(
                    tma_atom_V,
                    gV_tma,
                    sKV_ptr,
                    sV_page_layout,
                    mPageTable,
                    tile_base + 2,
                    kv_producer_state,
                    pipeline_kv,
                    K_or_V="V",
                    load_warp_rank=load_warp_rank,
                )
                kv_producer_state = self.load_KV_pages(
                    tma_atom_V,
                    gV_tma,
                    sKV_ptr,
                    sV_page_layout,
                    mPageTable,
                    tile_base,
                    kv_producer_state,
                    pipeline_kv,
                    K_or_V="V",
                    load_warp_rank=load_warp_rank,
                )

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
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        tOtO: cute.Tensor,
        tOrP: cute.Tensor,
        pipeline_q: pipeline.PipelineAsync,
        pipeline_kv: pipeline.PipelineAsync,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        tile_scheduler: TileSchedulerProtocol,
        mBlockNums: cute.Tensor,
    ):
        tSrQ = tiled_mma_qk.make_fragment_A(sQ)
        tSrK = tiled_mma_qk.make_fragment_B(sK)
        tOrV = tiled_mma_pv.make_fragment_B(sV)
        # q_stage=1: both stages use the same Q (intra-warp overlap across n_blocks)
        tSrQ0 = tSrQ[None, None, None, 0]
        sQ0 = sQ[None, None, None, 0]
        tOrP0 = tOrP[None, None, None, 0]
        tOrP1 = tOrP[None, None, None, 1]

        qk_mma_op = tiled_mma_qk.op
        pv_mma_op = tiled_mma_pv.op

        mma_q_consumer_phase = Int32(0)
        mma_kv_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.kv_stage
        )
        q_full_mbar = Int32(pipeline_q.sync_object_full.get_barrier(0).toint())
        spo_empty_mbar0 = Int32(pipeline_s_p_o.sync_object_empty.get_barrier(0).toint())
        spo_empty_mbar1 = Int32(pipeline_s_p_o.sync_object_empty.get_barrier(1).toint())
        # Pipeline s_p_o phases for stage 0 and stage 1.
        # These phases must persist across tiles
        # so that mbarrier phase stays in sync when block_iter_count varies.
        phase_s0 = Int32(0)
        phase_s1 = Int32(0)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx

            block_iter_count = mBlockNums[batch_idx, head_idx, m_block]
            process_tile = block_iter_count > Int32(0)

            if process_tile:
                # ================================================================
                # q_stage=1: intra-warp overlap across n_block direction
                # Pipeline KV order: K0, K1, V0, K2, V1, K3, V2, ...
                # GEMM order: S0=Q@K0, S1=Q@K1, {O0+=P0@V0, S0=Q@K2}, {O1+=P1@V1, S1=Q@K3}, ...
                # ================================================================

                # Prologue: wait Q0
                bsa_fwd_helpers.mbar_wait(q_full_mbar, mma_q_consumer_phase)
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                mma_q_consumer_phase ^= 1

                # S0 = Q @ K0
                bsa_fwd_helpers.mbar_wait(
                    Int32(
                        pipeline_kv.sync_object_full.get_barrier(
                            mma_kv_consumer_state.index
                        ).toint()
                    ),
                    mma_kv_consumer_state.phase,
                )
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                self.ws_qk_gemm(
                    qk_mma_op, 0, tSrQ0, tSrK[None, None, None, Ki_index], sQ0, sK_cur
                )
                pipeline_s_p_o.producer_commit_w_index(0)  # signal S0 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # S1 = Q @ K1
                bsa_fwd_helpers.mbar_wait(
                    Int32(
                        pipeline_kv.sync_object_full.get_barrier(
                            mma_kv_consumer_state.index
                        ).toint()
                    ),
                    mma_kv_consumer_state.phase,
                )
                bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                Ki_index = mma_kv_consumer_state.index
                sK_cur = sK[None, None, None, Ki_index]
                self.ws_qk_gemm(
                    qk_mma_op, 1, tSrQ0, tSrK[None, None, None, Ki_index], sQ0, sK_cur
                )
                pipeline_s_p_o.producer_commit_w_index(1)  # signal S1 ready
                pipeline_kv.consumer_release(mma_kv_consumer_state)
                mma_kv_consumer_state.advance()

                # Flat loop: N-2 iterations, alternating stage 0/1
                block_loop_count = block_iter_count - 2
                O_acc_s0 = False  # per-stage accumulate flags
                O_acc_s1 = False
                # Pre-declare loop variables for DSL type tracking
                Vi_index = mma_kv_consumer_state.index
                Ki_index = mma_kv_consumer_state.index
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
                            spo_empty_mbar = spo_empty_mbar0
                            tOrP_stage = tOrP0
                        else:
                            phase_cur, O_acc_cur = phase_s1, O_acc_s1
                            spo_empty_mbar = spo_empty_mbar1
                            tOrP_stage = tOrP1
                        bsa_fwd_helpers.mbar_wait(spo_empty_mbar, phase_cur)
                        # Wait V
                        bsa_fwd_helpers.mbar_wait(
                            Int32(
                                pipeline_kv.sync_object_full.get_barrier(
                                    mma_kv_consumer_state.index
                                ).toint()
                            ),
                            mma_kv_consumer_state.phase,
                        )
                        bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                        mma_kv_release_state = mma_kv_consumer_state.clone()
                        Vi_index = mma_kv_consumer_state.index
                        tOrVi = tOrV[None, None, None, Vi_index]
                        sV_cur = sV[None, None, None, Vi_index]
                        mma_kv_consumer_state.advance()
                        self.ws_pv_gemm(
                            pv_mma_op,
                            stage,
                            tOrP_stage,
                            tOrVi,
                            sV_cur,
                            not O_acc_cur,
                            phase_cur,
                            pipeline_p_lastsplit,
                        )
                        pipeline_kv.consumer_release(mma_kv_release_state)
                        # Overlap the independent K wait with the preceding PV issue.
                        bsa_fwd_helpers.mbar_wait(
                            Int32(
                                pipeline_kv.sync_object_full.get_barrier(
                                    mma_kv_consumer_state.index
                                ).toint()
                            ),
                            mma_kv_consumer_state.phase,
                        )
                        bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                        Ki_index = mma_kv_consumer_state.index
                        sK_cur = sK[None, None, None, Ki_index]
                        self.ws_qk_gemm(
                            qk_mma_op,
                            stage,
                            tSrQ0,
                            tSrK[None, None, None, Ki_index],
                            sQ0,
                            sK_cur,
                        )
                        pipeline_s_p_o.producer_commit_w_index(stage)
                        if const_expr(stage == 0):
                            phase_s0 ^= 1
                            O_acc_s0 = True
                        else:
                            phase_s1 ^= 1
                            O_acc_s1 = True
                        pipeline_kv.consumer_release(mma_kv_consumer_state)
                        mma_kv_consumer_state.advance()

                # release Q0
                pipeline_q.consumer_release_w_index(0)

                # Epilogue: 2 PV GEMMs (N even: stage 0 first, stage 1 second)
                for epi_stage_constexpr in cutlass.range_constexpr(2):
                    if const_expr(epi_stage_constexpr == 0):
                        epi_s, epi_p, epi_zi = 0, phase_s0, not O_acc_s0
                        spo_empty_mbar = spo_empty_mbar0
                        tOrP_epi = tOrP0
                    else:
                        epi_s, epi_p, epi_zi = 1, phase_s1, not O_acc_s1
                        spo_empty_mbar = spo_empty_mbar1
                        tOrP_epi = tOrP1
                    bsa_fwd_helpers.mbar_wait(spo_empty_mbar, epi_p)
                    bsa_fwd_helpers.mbar_wait(
                        Int32(
                            pipeline_kv.sync_object_full.get_barrier(
                                mma_kv_consumer_state.index
                            ).toint()
                        ),
                        mma_kv_consumer_state.phase,
                    )
                    bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                    Vi_index = mma_kv_consumer_state.index
                    tOrVi = tOrV[None, None, None, Vi_index]
                    sV_cur = sV[None, None, None, Vi_index]
                    self.ws_pv_gemm(
                        pv_mma_op,
                        epi_stage_constexpr,
                        tOrP_epi,
                        tOrVi,
                        sV_cur,
                        epi_zi,
                        epi_p,
                        pipeline_p_lastsplit,
                    )
                    pipeline_o_acc.producer_commit_w_index(epi_s)
                    pipeline_kv.consumer_release(mma_kv_consumer_state)
                    mma_kv_consumer_state.advance()
                # Epilogue did one acquire per stage; advance phases for next tile
                phase_s0 ^= 1
                phase_s1 ^= 1

            # Advance to next tile
            work_tile = tile_scheduler.consumer_advance()
        # End of persistent scheduler loop

        # We don't need pipeline_s_p_o.producer_tail() since there's no dangling mbarrier at the end
        # We don't need pipeline_o_acc.producer_tail() since we don't call
        # pipeline_o_acc.producer_acquire() inside the loop.

    @cute.jit
    def ws_pv_gemm(
        self,
        pv_mma_op: cute.nvgpu.tcgen05.mma.MmaOp,
        stage: int,
        tCrP: cute.Tensor,
        tCrV: cute.Tensor,
        sV: cute.Tensor,
        zero_init: bool | Boolean,
        phase: Int32,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
    ) -> None:
        bsa_fwd_helpers.gemm_ptx_partial(
            pv_mma_op,
            self.tmem_o_offset[stage],
            tCrP,
            tCrV,
            sA=None,
            sB=sV,
            mbar_ptr=(
                pipeline_p_lastsplit.sync_object_full.get_barrier(stage)
                if const_expr(self.split_P_arrive > 0)
                else None
            ),
            mbar_phase=phase,
            split_arrive=(
                self.split_P_arrive if const_expr(self.split_P_arrive > 0) else None
            ),
            zero_init=zero_init,
            tA_addr=Int32(self.tmem_p_offset[stage]),
        )

    @cute.jit
    def ws_qk_gemm(
        self,
        qk_mma_op: cute.nvgpu.tcgen05.mma.MmaOp,
        stage: int,
        tCrQ: cute.Tensor,
        tCrK: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
    ) -> None:
        bsa_fwd_helpers.gemm_ptx_partial(
            qk_mma_op,
            self.tmem_s_offset[stage],
            tCrQ,
            tCrK,
            sA=sQ,
            sB=sK,
            zero_init=True,
        )

    # for both softmax0 and softmax1 warp group
    @cute.jit
    def softmax_loop(
        self,
        stage: int | Int32,
        softmax_scale_log2: Float32,
        softmax_scale: Float32,
        score_scale_log2: Float32,
        sScale: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        tile_scheduler: TileSchedulerProtocol,
        mMaskTable: cute.Tensor,
        mTileOffsets: cute.Tensor,
        mFullIterCounts: cute.Tensor,
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

        mma_si_consumer_phase = Int32(0)
        sm_stats_producer_phase = Int32(1)

        tmem_s_addr = Int32(stage) * Int32(
            self.tmem_s_offset[1] - self.tmem_s_offset[0]
        )
        tmem_p_addr = Int32(stage) * Int32(
            self.tmem_p_offset[1] - self.tmem_p_offset[0]
        )
        sm_stats_bar_index = stage * 4 + warp_idx

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            block_iter_count = mBlockNums[batch_idx, head_idx, m_block]
            has_work = block_iter_count > Int32(0)

            softmax = SoftmaxSm100.create(
                softmax_scale_log2,
                rescale_threshold=8.0,
                softmax_scale=softmax_scale,
            )
            softmax.reset()

            softmax_step = partial(
                self.softmax_step,
                softmax=softmax,
                pipeline_s_p_o=pipeline_s_p_o,
                pipeline_p_lastsplit=pipeline_p_lastsplit,
                pipeline_sm_stats=pipeline_sm_stats,
                sm_stats_barrier=sm_stats_barrier,
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
                # block_iter_count is the padded number of 4-sparse-block KV groups.
                # WG0 (stage=0): kv groups N-1, N-3, ...
                # WG1 (stage=1): kv groups N-2, N-4, ...
                wg_count = block_iter_count // 2
                full_iter_count = mFullIterCounts[batch_idx, head_idx, m_block]
                partial_iter_count = block_iter_count - full_iter_count
                masked_wg_count = (partial_iter_count + 1 - stage) // 2
                full_wg_count = wg_count - masked_wg_count
                mask_row = (
                    mTileOffsets[batch_idx, head_idx, m_block]
                    + (block_iter_count - 1 - stage) * 2
                )
                mask_word_base = (warp_idx // 2) * 2

                # The first step is the generic step started from the reset state (row_max=-inf, row_sum=0 gives
                # acc_scale == 0, i.e. the is_first result), so every step shares one code copy per mask variant: a
                # separate once-per-tile copy of the large softmax step thrashes the instruction cache.
                for _ in cutlass.range(masked_wg_count, unroll=1):
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        tidx=tidx,
                        tmem_s_addr=tmem_s_addr,
                        tmem_p_addr=tmem_p_addr,
                        sm_stats_bar_index=sm_stats_bar_index,
                        mMaskTable=mMaskTable,
                        mask_row=mask_row,
                        mask_word_base=mask_word_base,
                        masked=True,
                    )
                    mask_row -= 4
                for _ in cutlass.range(full_wg_count, unroll=1):
                    mma_si_consumer_phase, sm_stats_producer_phase = softmax_step(
                        mma_si_consumer_phase,
                        sm_stats_producer_phase,
                        tidx=tidx,
                        tmem_s_addr=tmem_s_addr,
                        tmem_p_addr=tmem_p_addr,
                        sm_stats_bar_index=sm_stats_bar_index,
                        mMaskTable=mMaskTable,
                        mask_row=mask_row,
                        mask_word_base=mask_word_base,
                    )

                sScale[tidx + stage * self.stats_stride] = softmax.row_sum[0]
                if const_expr(mLSE is not None or self.q_stage == 1):
                    sScale[
                        tidx
                        + stage * self.stats_stride
                        + self.s_stage * self.stats_stride
                    ] = softmax.row_max[0]
                sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)
            else:
                # Empty tile: arrive barrier once (synthetic "no work" signal for correction)
                sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)

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
        softmax: SoftmaxSm100,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_p_lastsplit: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        sScale: cute.Tensor,
        stage: int | Int32,
        score_scale_log2: Float32,
        tidx: Int32,
        tmem_s_addr: Int32,
        tmem_p_addr: Int32,
        sm_stats_bar_index: Int32,
        mMaskTable: cute.Tensor,
        mask_row: Int32,
        mask_word_base: Int32,
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
        # Prefetch packed mask metadata before waiting for scores.
        mask_words = cute.make_rmem_tensor((4,), Int32)
        if const_expr(masked):
            mask_words[0] = mMaskTable[mask_row, mask_word_base]
            mask_words[1] = mMaskTable[mask_row, mask_word_base + 1]
            mask_words[2] = mMaskTable[mask_row + 1, mask_word_base]
            mask_words[3] = mMaskTable[mask_row + 1, mask_word_base + 1]

        # Wait for Si
        bsa_fwd_helpers.mbar_wait(
            Int32(pipeline_s_p_o.sync_object_full.get_barrier(stage).toint()),
            mma_si_consumer_phase,
        )

        # The hardware row max is of the unscaled scores, which is not the max of the scaled ones for a nonpositive scale.
        use_ldred_rowmax = (
            self.arch >= Arch.sm_103
            and self.arch <= Arch.sm_103f
            and not self.nonpositive_scale
        )
        tSrS_t2r = cute.make_rmem_tensor((128,), Float32)
        row_max_tile = -Float32.inf
        for c in cutlass.range_constexpr(4):
            if const_expr(use_ldred_rowmax and not masked):
                vals, row_max_chunk = bsa_fwd_helpers.tmem_load_red_max_32dp32b32x(
                    tmem_s_addr + c * 32
                )
                row_max_tile = utils.fmax(row_max_tile, row_max_chunk)
            else:
                vals = bsa_fwd_helpers.tmem_load_32dp32b32x(tmem_s_addr + c * 32)
            for j in cutlass.range_constexpr(32):
                tSrS_t2r[c * 32 + j] = vals[j]
            if const_expr(self.nonpositive_scale):
                # Scale the finite scores before the -inf mask sentinels; the softmax then runs with a unit scale.
                for j in cutlass.range_constexpr(0, 32, 2):
                    tSrS_t2r[c * 32 + j], tSrS_t2r[c * 32 + j + 1] = (
                        cute.arch.mul_packed_f32x2(
                            (tSrS_t2r[c * 32 + j], tSrS_t2r[c * 32 + j + 1]),
                            (score_scale_log2, score_scale_log2),
                        )
                    )
            if const_expr(masked):
                kpage_utils.apply_packed_mask_chunk(tSrS_t2r, c, mask_words[c])

        if const_expr(use_ldred_rowmax and not masked):
            row_max, acc_scale = kpage_utils.update_row_max_precomputed(
                softmax,
                row_max_tile,
                is_first,
            )
        else:
            row_max, acc_scale = softmax.update_row_max(tSrS_t2r.load(), is_first)

        if const_expr(not is_first):
            sScale[tidx + stage * self.stats_stride] = acc_scale
        # Notify correction wg that row_max is ready
        sm_stats_barrier.arrive_w_index(index=sm_stats_bar_index)

        softmax.scale_subtract_rowmax(tSrS_t2r, row_max)
        tSrS_frg = cute.logical_divide(tSrS_t2r, cute.make_layout(32))
        if const_expr(not is_first):
            sum0 = (softmax.row_sum[0] * acc_scale, 0.0)
        else:
            sum0 = (0.0, 0.0)
        sum1 = (0.0, 0.0)
        sum2 = (0.0, 0.0)
        sum3 = (0.0, 0.0)
        for i in cutlass.range_constexpr(4):
            p_frag = cute.make_rmem_tensor((16,), Int32)
            for j in cutlass.range_constexpr(0, 32, 8):
                e0 = cute.math.exp2(tSrS_frg[j + 0, i], fastmath=True)
                e1 = cute.math.exp2(tSrS_frg[j + 1, i], fastmath=True)
                tSrS_frg[j + 0, i] = e0
                tSrS_frg[j + 1, i] = e1

                e2 = cute.math.exp2(tSrS_frg[j + 2, i], fastmath=True)
                e3 = cute.math.exp2(tSrS_frg[j + 3, i], fastmath=True)
                tSrS_frg[j + 2, i] = e2
                tSrS_frg[j + 3, i] = e3

                e4 = cute.math.exp2(tSrS_frg[j + 4, i], fastmath=True)
                e5 = cute.math.exp2(tSrS_frg[j + 5, i], fastmath=True)
                tSrS_frg[j + 4, i] = e4
                tSrS_frg[j + 5, i] = e5

                e6 = cute.math.exp2(tSrS_frg[j + 6, i], fastmath=True)
                e7 = cute.math.exp2(tSrS_frg[j + 7, i], fastmath=True)
                tSrS_frg[j + 6, i] = e6
                tSrS_frg[j + 7, i] = e7
                p_frag[j // 2 + 0] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e0, e1)
                p_frag[j // 2 + 1] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e2, e3)
                p_frag[j // 2 + 2] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e4, e5)
                p_frag[j // 2 + 3] = bsa_fwd_helpers.cvt_f32x2_to_bf16x2(e6, e7)
            bsa_fwd_helpers.tmem_store_bf16x16(tmem_p_addr + i * 16, p_frag)
            if const_expr(self.split_P_arrive > 0):
                split_P_arrive_idx = self.split_P_arrive // 32
                if const_expr(i + 1 == split_P_arrive_idx):
                    # Notify mma warp that the 1st half of P is ready
                    cute.arch.fence_view_async_tmem_store()
                    pipeline_s_p_o.consumer_release_w_index(stage)
            for j in cutlass.range_constexpr(0, 32, 8):
                sum0 = cute.arch.add_packed_f32x2(
                    sum0, (tSrS_frg[j + 0, i], tSrS_frg[j + 1, i])
                )
                sum1 = cute.arch.add_packed_f32x2(
                    sum1, (tSrS_frg[j + 2, i], tSrS_frg[j + 3, i])
                )
                sum2 = cute.arch.add_packed_f32x2(
                    sum2, (tSrS_frg[j + 4, i], tSrS_frg[j + 5, i])
                )
                sum3 = cute.arch.add_packed_f32x2(
                    sum3, (tSrS_frg[j + 6, i], tSrS_frg[j + 7, i])
                )
        # Notify mma warp that the 2nd half of P is ready
        cute.arch.fence_view_async_tmem_store()
        if const_expr(self.split_P_arrive > 0):
            cute.arch.sync_warp()
            with cute.arch.elect_one():
                pipeline_p_lastsplit.producer_commit_w_index(stage)
        else:
            pipeline_s_p_o.consumer_release_w_index(stage)
        sum0 = cute.arch.add_packed_f32x2(sum0, sum1)
        sum2 = cute.arch.add_packed_f32x2(sum2, sum3)
        sum0 = cute.arch.add_packed_f32x2(sum0, sum2)
        softmax.row_sum[0] = sum0[0] + sum0[1]
        bsa_fwd_helpers.mbar_wait(
            Int32(pipeline_sm_stats.sync_object_empty.get_barrier(stage).toint()),
            sm_stats_producer_phase,
        )
        return mma_si_consumer_phase ^ 1, sm_stats_producer_phase ^ 1

    @cute.jit
    def correction_loop(
        self,
        thr_mma_pv: cute.ThrMma,
        tOtO: cute.Tensor,
        sScale: cute.Tensor,
        mLSE: cute.Tensor,
        sO: cute.Tensor,
        pipeline_s_p_o: pipeline.PipelineAsync,
        pipeline_o_acc: pipeline.PipelineAsync,
        pipeline_sm_stats: pipeline.PipelineAsync,
        sm_stats_barrier: pipeline.NamedBarrier,
        reduce_mbar_ptr: cute.Pointer,
        pipeline_o_epi: pipeline.PipelineAsync,
        softmax_scale_log2: Float32,
        oStats: cute.Tensor,
        oExchange: cute.Tensor,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
        mBlockNums: cute.Tensor,
    ):
        tidx = cute.arch.thread_idx()[0] % (
            cute.arch.WARP_SIZE * len(self.correction_warp_ids)
        )
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4
        reduce_mbar_addr = Int32((reduce_mbar_ptr + (warp_idx & 1)).toint())

        # First iter: no correction is required
        # Notify mma warp that O has been rescaled
        for stage in cutlass.range_constexpr(self.s_stage):
            pipeline_s_p_o.consumer_release_w_index(stage)

        sm_stats_consumer_phase = Int32(0)
        o_corr_consumer_phase = Int32(0)
        corr_epi_producer_phase = Int32(1)

        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

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

            block_iter_count = mBlockNums[batch_idx, head_idx, m_block]
            has_work = block_iter_count > Int32(0)

            if has_work:
                # Ignore first signal from softmax as no correction is required
                sm_stats_barrier.arrive_and_wait_w_index(index=0 * 4 + warp_idx)
                pipeline_sm_stats.consumer_release_w_index(0)
                # Wait for both softmax warp groups.
                sm_stats_barrier.arrive_and_wait_w_index(index=1 * 4 + warp_idx)
                sm_stats_consumer_phase ^= 1

                # q_stage=1 correction loop
                corr_pair_count = (block_iter_count - 2) // 2
                # Rescale the two alternating stages as a pair.
                for i in cutlass.range(corr_pair_count, unroll=1):
                    for stage in cutlass.range_constexpr(self.s_stage):
                        sm_stats_barrier.arrive_and_wait_w_index(
                            index=stage * 4 + warp_idx
                        )
                        scale = sScale[tidx + stage * self.stats_stride]
                        should_rescale = cute.arch.vote_ballot_sync(scale < 1.0) != 0
                        if should_rescale:
                            self.correction_rescale(
                                tOtO[None, None, None, stage], tidx, scale
                            )
                        pipeline_s_p_o.consumer_release_w_index(stage)
                        pipeline_sm_stats.consumer_release_w_index(
                            self.s_stage - 1 - stage
                        )
                    sm_stats_consumer_phase ^= 1
                # N even: no remainder. Release final sm_stats stage 1.
                pipeline_sm_stats.consumer_release_w_index(1)
                # End of seqlen_corr_loop_steps

                # Read final softmax stats for both stages
                for stage in cutlass.range_constexpr(self.s_stage):
                    sm_stats_barrier.arrive_and_wait_w_index(index=stage * 4 + warp_idx)
                    row_sum = sScale[tidx + stage * self.stats_stride]
                    if const_expr(mLSE is not None or self.q_stage == 1):
                        row_max = sScale[
                            tidx
                            + stage * self.stats_stride
                            + self.s_stage * self.stats_stride
                        ]
                    else:
                        row_max = None
                    pipeline_sm_stats.consumer_release_w_index(stage)
                    row_is_valid = row_sum > Float32(0.0)
                    stats[stage] = (row_sum, row_max, row_is_valid)

                # q_stage=1: combine O0 and O1, then write single output
                row_sum0, row_max0, valid0 = stats[0]
                row_sum1, row_max1, valid1 = stats[1]

                # Compute combined scales for the two partial O accumulators
                # row_max is in original S space (unscaled). To compute rescale
                # factors we need exp2((row_max - max_combined) * scale_log2).
                # For empty/padding stages, row_max may be 0.0 instead of -inf
                # due to softmax's safe_max clamping. Use -inf for those to avoid polluting max_combined.
                rm0 = row_max0 if valid0 else -Float32.inf
                rm1 = row_max1 if valid1 else -Float32.inf
                max_combined = cutlass.max(rm0, rm1)
                max_safe = max_combined if max_combined > -Float32.inf else Float32(0.0)
                scale0 = (
                    cute.math.exp2((rm0 - max_safe) * softmax_scale_log2, fastmath=True)
                    if valid0
                    else Float32(0.0)
                )
                scale1 = (
                    cute.math.exp2((rm1 - max_safe) * softmax_scale_log2, fastmath=True)
                    if valid1
                    else Float32(0.0)
                )
                sum_combined = row_sum0 * scale0 + row_sum1 * scale1
                my_sum = sum_combined
                my_max = max_combined

                # Wait for both O accumulators from MMA warp
                for stage in cutlass.range_constexpr(self.s_stage):
                    bsa_fwd_helpers.mbar_wait(
                        Int32(
                            pipeline_o_acc.sync_object_full.get_barrier(stage).toint()
                        ),
                        o_corr_consumer_phase,
                    )
                    bsa_fwd_helpers.tcgen05_fence_after_thread_sync()
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                mLSE_cur = (
                    mLSE[None, head_idx, batch_idx]
                    if const_expr(mLSE is not None)
                    else None
                )
                self.correction_epilogue_combine_ws_raw(
                    tOtO[None, None, None, 0].iterator.toint(),
                    tOtO[None, None, None, 1].iterator.toint(),
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    softmax_scale_log2,
                    scale0,
                    scale1,
                    my_sum,
                    my_max,
                    sO[None, None, 0],
                    oStats,
                    oExchange,
                    reduce_mbar_addr,
                    mLSE_cur,
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
                # Write O=0 through the raw WS exchange path without reading stale TMEM.
                pipeline_o_epi.producer_acquire_w_index_phase(
                    0, corr_epi_producer_phase
                )
                mLSE_cur = (
                    mLSE[None, head_idx, batch_idx]
                    if const_expr(mLSE is not None)
                    else None
                )
                self.correction_epilogue_combine_ws_raw(
                    tOtO[None, None, None, 0].iterator.toint(),
                    tOtO[None, None, None, 1].iterator.toint(),
                    tidx,
                    m_block,
                    seqlen.seqlen_q,
                    softmax_scale_log2,
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    Float32(0.0),
                    sO[None, None, 0],
                    oStats,
                    oExchange,
                    reduce_mbar_addr,
                    mLSE_cur,
                )
                # Do NOT release pipeline_s_p_o (MMA didn't commit)
                pipeline_o_epi.producer_commit_w_index(0)
                # o_corr_consumer_phase NOT toggled (pipeline_o_acc not touched)
                corr_epi_producer_phase ^= 1

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
        tOtO: cute.Tensor,
        tidx: Int32,
        scale: Float32,
    ):
        """Rescale intermediate attention results based on softmax normalization factor.

        This method performs a crucial correction step in the attention computation pipeline.
        When processing attention in blocks, the softmax normalization factors may change
        as new blocks are processed. This method rescales previously computed partial
        output values to account for updated normalization factors.

        The implementation uses efficient tensor memory operations to:
        1. Load existing partial attention output from tensor memory
        2. Apply the scaling factor to all elements
        3. Store the rescaled results back to tensor memory
        """
        # TMEM addresses encode the datapath in the upper 16 bits and the
        # column in the lower 16 bits.  Build the copy view explicitly so the
        # CuTe atom follows the same physical 32-datapath tiles as the WS MMA.
        corr_tile_size = 32
        tmem_datapath_stride = 1 << 16
        tmem_tile_layout = cute.make_layout(
            (cute.arch.WARP_SIZE, corr_tile_size),
            stride=(tmem_datapath_stride, 1),
        )
        coord_tile = cute.make_identity_tensor((cute.arch.WARP_SIZE, corr_tile_size))
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(corr_tile_size)),
            self.pv_acc_dtype,
        )

        num_fragments = self.head_dim_v_padded // corr_tile_size
        for i in cutlass.range_constexpr(num_fragments):
            tOtO_i = cute.make_tensor(
                tOtO.iterator + i * corr_tile_size,
                tmem_tile_layout,
            )
            tiled_tmem_load = tcgen05.make_tmem_copy(tmem_load_atom, tOtO_i)
            tiled_tmem_store = tcgen05.make_tmem_copy(tmem_store_atom, tOtO_i)
            copy_thread_idx = tidx % cute.arch.WARP_SIZE
            thr_tmem_load = tiled_tmem_load.get_slice(copy_thread_idx)
            thr_tmem_store = tiled_tmem_store.get_slice(copy_thread_idx)
            tOtO_t2r = thr_tmem_load.partition_S(tOtO_i)
            tOtO_r2t = thr_tmem_store.partition_D(tOtO_i)
            tOrO_frg = cute.make_rmem_tensor(
                thr_tmem_load.partition_D(coord_tile).shape,
                self.pv_acc_dtype,
            )
            cute.copy(tiled_tmem_load, tOtO_t2r, tOrO_frg)
            for j in cutlass.range(0, cute.size(tOrO_frg), 2, unroll_full=True):
                tOrO_frg[j], tOrO_frg[j + 1] = cute.arch.mul_packed_f32x2(
                    (tOrO_frg[j], tOrO_frg[j + 1]),
                    (scale, scale),
                )
            cute.copy(tiled_tmem_store, tOrO_frg, tOtO_r2t)
        cute.arch.fence_view_async_tmem_store()

    @cute.jit
    def correction_epilogue_combine_ws_raw(
        self,
        tmem_o0_addr: Int32,
        tmem_o1_addr: Int32,
        tidx: Int32,
        m_block: Int32,
        seqlen_q: Int32,
        softmax_scale_log2: Float32,
        scale0: Float32,
        scale1: Float32,
        my_sum: Float32,
        my_max: Float32,
        sO: cute.Tensor,
        oStats: cute.Tensor,
        oExchange: cute.Tensor,
        reduce_mbar_addr: Int32,
        mLSE_cur: Optional[cute.Tensor] = None,
    ):
        """C++ blk64-style WS epilogue combine using raw TMEM/SMEM addressing."""
        corr_warp = tidx // cute.arch.WARP_SIZE
        lane_idx = tidx % cute.arch.WARP_SIZE
        partner_warp = corr_warp ^ 2

        # Exchange the two warp-pair row stats: (0,2) and (1,3).
        oStats[(partner_warp * 64) + lane_idx * 2 + 0] = my_sum
        oStats[(partner_warp * 64) + lane_idx * 2 + 1] = my_max
        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(0))

        partner_sum = oStats[(corr_warp * 64) + lane_idx * 2 + 0]
        partner_max = oStats[(corr_warp * 64) + lane_idx * 2 + 1]
        max_total = cutlass.max(my_max, partner_max)
        max_total_safe = max_total if max_total > -Float32.inf else Float32(0.0)
        my_rescale = (
            cute.math.exp2(
                (my_max - max_total_safe) * softmax_scale_log2, fastmath=True
            )
            if my_sum > Float32(0.0)
            else Float32(0.0)
        )
        partner_rescale = (
            cute.math.exp2(
                (partner_max - max_total_safe) * softmax_scale_log2, fastmath=True
            )
            if partner_sum > Float32(0.0)
            else Float32(0.0)
        )
        sum_total = my_sum * my_rescale + partner_sum * partner_rescale
        total_is_valid = sum_total > Float32(0.0)
        inv_sum_total = (
            cute.arch.rcp_approx(sum_total) if total_is_valid else Float32(0.0)
        )
        my_weight = my_rescale * inv_sum_total
        my_scale0 = scale0 * my_weight
        my_scale1 = scale1 * my_weight

        exchange_warp_base = corr_warp * 4 * 32 * 32
        exchange_addr = Int32(
            (oExchange.iterator + exchange_warp_base + lane_idx * 4).toint()
        )
        # The TMEM load is warp-collective. Keep all lanes converged through
        # the exchange even when an empty attention row has zero scales.
        # An empty tile never wrote the O accumulators, so 0 * (stale TMEM) could be NaN:
        # when every lane of the warp has zero scales, skip the load and store zeros.
        is_zero_output = my_scale0 == Float32(0.0) and my_scale1 == Float32(0.0)
        warp_needs_tmem = cute.arch.vote_ballot_sync(not is_zero_output) != 0
        if warp_needs_tmem:
            bsa_fwd_helpers.tmem_combine_store_exchange_4x32dp32b32x(
                Int32(tmem_o0_addr),
                Int32(tmem_o1_addr),
                exchange_addr,
                my_scale0,
                my_scale1,
            )
        else:
            bsa_fwd_helpers.smem_zero_store_exchange_4x32dp32b32x(exchange_addr)

        bsa_fwd_helpers.mbar_arrive_and_wait(reduce_mbar_addr, Int32(1))

        out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
        if corr_warp < 2 and out_row < self.m_block_size:
            own_warp_base = corr_warp * 4 * 32 * 32
            partner_warp_base = partner_warp * 4 * 32 * 32
            lane_col_swizzle = lane_idx & 7
            for c in cutlass.range_constexpr(4):
                off = c * 32 * 32 + lane_idx * 4
                col0 = (((c * 4) + 0) ^ lane_col_swizzle) * 8
                col1 = (((c * 4) + 1) ^ lane_col_swizzle) * 8
                col2 = (((c * 4) + 2) ^ lane_col_swizzle) * 8
                col3 = (((c * 4) + 3) ^ lane_col_swizzle) * 8
                bsa_fwd_helpers.smem_exchange_reduce_store_bf16x32(
                    Int32((oExchange.iterator + own_warp_base + off).toint()),
                    Int32((oExchange.iterator + partner_warp_base + off).toint()),
                    Int32((sO.iterator + sO.layout((out_row, col0))).toint()),
                    Int32((sO.iterator + sO.layout((out_row, col1))).toint()),
                    Int32((sO.iterator + sO.layout((out_row, col2))).toint()),
                    Int32((sO.iterator + sO.layout((out_row, col3))).toint()),
                )

        cute.arch.fence_view_async_shared()

        # Keep every lane converged through the two warp-pair exchange barriers
        # above. On a partial final Q tile, predicating this store before the
        # raw inline-assembly mbarrier collectives introduces tail-dependent
        # lane divergence and can deadlock the CTA on SM100.
        if const_expr(mLSE_cur is not None):
            out_row = (corr_warp & 1) * cute.arch.WARP_SIZE + lane_idx
            valid_rows = seqlen_q - m_block * self.m_block_size
            if corr_warp < 2 and out_row < valid_rows:
                LN2 = math.log(2.0)
                lse_log2 = max_total_safe * softmax_scale_log2 + cute.math.log2(
                    sum_total, fastmath=True
                )
                lse = lse_log2 * LN2 if total_is_valid else -Float32.inf
                mLSE_cur[m_block * self.m_block_size + out_row] = lse

    @cute.jit
    def epilogue_s2g(
        self,
        mO: cute.Tensor,
        sO: cute.Tensor,
        tma_atom_O: cute.CopyAtom,
        pipeline_o_epi: pipeline.PipelineAsync,
        SeqlenInfoCls: Callable,
        tile_scheduler: TileSchedulerProtocol,
    ):
        epi_consumer_phase = Int32(0)
        tiler_gO = ((self.mma_tiler_pv[0] * self.q_stage), self.head_dim_v_padded)
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            m_block, head_idx, batch_idx, split_idx = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)

            tidx = cute.arch.thread_idx()[0] % (
                cute.arch.WARP_SIZE * len(self.epilogue_warp_ids)
            )
            mO_cur = mO[None, None, None, batch_idx][None, None, head_idx]
            gO = cute.local_tile(mO_cur, tiler_gO, (m_block, 0))
            gO = layout_utils.select(
                cute.flat_divide(gO, (self.mma_tiler_pv[0],)), mode=[0, 2, 1]
            )
            gO = cute.flat_divide(gO, (self.mma_tiler_pv[0],))[None, 0, None, None]

            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O,
                0,
                cute.make_layout(1),
                sO,
                gO,
            )
            for stage in cutlass.range_constexpr(self.q_stage):
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
    def load_KV_pages(
        self,
        tma_atom_X: cute.CopyAtom,
        gX: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sX_page_layout: cute.ComposedLayout,
        mPageTable: cute.Tensor,
        row: Int32,
        producer_state: pipeline.PipelineState,
        pipeline_kv: pipeline.PipelineAsync,
        load_warp_rank: Int32,
        K_or_V: cutlass.Constexpr[str],
    ):
        """Fill one N256 K or V stage with two Page16 loader warps."""
        assert K_or_V in ("K", "V")
        pages_per_loader = 2 * FRAGS_PER_HALF // len(self.load_warp_ids)
        row_partition = cute.make_tensor(
            mPageTable.iterator
            + row * FRAGS_PER_HALF
            + load_warp_rank * pages_per_loader,
            cute.make_layout(pages_per_loader),
        )
        lane_idx = cute.arch.lane_idx()
        page_id = Int32(0)
        if lane_idx < pages_per_loader:
            page_id = row_partition[lane_idx]

        pipeline_kv.producer_acquire(producer_state)
        tma_bar_ptr = pipeline_kv.producer_get_barrier(producer_state)
        stage_base = producer_state.index * self.kv_elems_per_stage
        self.issue_page_partition(
            tma_atom_X,
            gX,
            sKV_ptr,
            sX_page_layout,
            page_id,
            stage_base,
            tma_bar_ptr,
            K_or_V,
            pages_per_partition=pages_per_loader,
            owner=load_warp_rank,
        )
        producer_state.advance()
        return producer_state

    @cute.jit
    def issue_page_partition(
        self,
        tma_atom_X: cute.CopyAtom,
        gX: cute.Tensor,
        sKV_ptr: cute.Pointer,
        sX_page_layout: cute.ComposedLayout,
        lane_page_id: Int32,
        stage_base: Int32,
        tma_bar_ptr: cute.Pointer,
        K_or_V: cutlass.Constexpr[str],
        pages_per_partition: cutlass.Constexpr[int],
        owner: Int32,
    ) -> None:
        """Issue one loader warp's contiguous partition of Page16 TMAs."""
        for slot in cutlass.range_constexpr(pages_per_partition):
            page_id = cute.arch.shuffle_sync(lane_page_id, Int32(slot))
            sub = owner * pages_per_partition + slot
            logical_block = sub // 4
            page_in_block = sub & 3
            if const_expr(K_or_V == "K"):
                physical_block = (logical_block & 1) * 2 + logical_block // 2
                smem_offset = (
                    stage_base
                    + physical_block
                    * self.sparse_block_size
                    * (self.head_dim_padded // 2)
                    + page_in_block * PAGE * (self.head_dim_padded // 2)
                )
            else:
                smem_offset = (
                    stage_base
                    + logical_block * self.sparse_block_size * self.head_dim_v_padded
                    + page_in_block * PAGE * (self.head_dim_v_padded // 2)
                )
            sX_sub = cute.make_tensor(
                cute.recast_ptr(sKV_ptr + smem_offset, sX_page_layout.inner),
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
                tXgX[None, page_id],
                tXsX[None],
                tma_bar_ptr=tma_bar_ptr,
            )
