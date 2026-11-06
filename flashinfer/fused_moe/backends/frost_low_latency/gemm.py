# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Routed eight-warp specialization of FlashInfer's low-latency GEMM.

The upstream TMA producers and epilogue are inherited without modification.
This specialization supplies independent runtime expert/token indices, a
caller-owned shared-memory arena for the fused activation, and a bounded TMEM
allocation. It uses one CTA per output tile and no persistent scheduler.
"""

from typing import Optional, Tuple

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass.pipeline import (
    PipelineAsync,
    PipelineUmmaAsync,
    PipelineAsyncUmma,
    CooperativeGroup,
    Agent,
    pipeline_init_arrive,
    pipeline_init_wait,
    make_pipeline_state,
    PipelineUserType,
)
from flashinfer.gemm.kernels.cute_dsl.low_latency_blockscaled_gemm import (
    LowLatencyBlockscaledGemmKernel as _Base,
    WorkTileInfo,
)


class RoutedBlockscaledGemm(_Base):
    """One expert/token GEMM with direct SFB copies and eight warp roles."""

    def __init__(
        self,
        *,
        tile_k=128,
        stages=4,
        allocation_columns=64,
        input_route_divisor=1,
        use_pdl=True,
    ):
        if allocation_columns not in (32, 64, 128, 256):
            raise ValueError("TMEM allocation must be 32, 64, 128, or 256 columns")
        if input_route_divisor < 1:
            raise ValueError("input_route_divisor must be positive")
        super().__init__(
            mma_tiler_mnk=(128, 8, tile_k),
            num_ab_stage=stages,
            num_sfb_tmem_stage=1,
            sf_vec_size=32,
            split_k=1,
            sfb_tmem_store=False,
            use_pdl=use_pdl,
        )
        self.allocation_columns = allocation_columns
        self.threads_per_cta = 256
        self.global_k_parts = 1
        self.input_route_divisor = input_route_divisor

    def _setup_attributes(self):
        super()._setup_attributes()
        required = (
            self.cta_n
            + self._num_sfa_tmem_cols
            + self._sfb_tmem_cols_per_stage * self.num_sfb_tmem_stage
        )
        if required > self.allocation_columns:
            raise ValueError(
                f"TMEM requires {required} columns, got {self.allocation_columns}"
            )
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage
        MailboxElems = self._mailbox_total_elems

        @cute.struct
        class SharedStorage:
            # AB pipeline barriers: full[0..DMA_Stage-1] + empty[DMA_Stage..2*DMA_Stage-1]
            ab_pipeline_bars: cute.struct.MemRange[cutlass.Int64, DMA_Stage * 2]
            # SFB pipeline barriers (unused on the direct-copy path)
            sfb_pipeline_bars: cute.struct.MemRange[cutlass.Int64, SFB_Stage * 2]
            # MMA→epilog pipeline: full[0] + empty[1]
            mma_epilog_bars: cute.struct.MemRange[cutlass.Int64, 2]
            # Standalone barriers coordinate tensor memory and split-K reduction
            tmem_allocation_result_barrier: cutlass.Int64
            tma_epilog_full_barrier: cutlass.Int64
            tmem_base_ptr: cutlass.Int32
            dsmem_mailbox_barrier: cutlass.Int64
            dsmem_mailbox: cute.struct.MemRange[cutlass.Float32, MailboxElems]

            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                128,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                128,
            ]
            sSFB_flat: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self._sfb_smem_layout_flat)
                ],
                128,
            ]
            sSFB_padded: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self._sfb_smem_layout_padded)
                ],
                128,
            ]

        self.shared_storage = SharedStorage

    @cute.jit
    def prepare(
        self,
        a_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        scale_ptr: Optional[cute.Pointer],
        bias_ptr: Optional[cute.Pointer],
        problem_mnkl: Tuple,
        route_ids: cute.Tensor,
        stream: cuda.CUstream,
    ):
        """Execute the GEMM kernel warp specialized for DMA_A, DMA_B, MMA, SFB, and EPILOG.
        - DMA_A: loads A tiles via TMA
        - DMA_B: loads B tiles via TMA
        - MMA: performs the block-scaled matrix multiply
        - SFB: stages SFB into TMEM through registers (sfb_tmem_store only)
        - EPILOG: TMEM->RMEM->GMEM store

        :param a_ptr: Pointer to the A operand.
        :type a_ptr: cute.Pointer
        :param sfa_ptr: Pointer to the SFA operand.
        :type sfa_ptr: cute.Pointer
        :param b_ptr: Pointer to the B operand.
        :type b_ptr: cute.Pointer
        :param sfb_ptr: Pointer to the SFB operand.
        :type sfb_ptr: cute.Pointer
        :param c_ptr: Pointer to the C operand.
        :type c_ptr: cute.Pointer
        :param scale_ptr: Pointer to the optional FP32 output scale.
        :type scale_ptr: Optional[cute.Pointer]
        :param bias_ptr: Pointer to the optional M-element bias, or None.
        :type bias_ptr: Optional[cute.Pointer]
        :param problem_mnkl: Problem shape (M, N, K, L).
        :type problem_mnkl: Tuple
        :param stream: CUDA stream for asynchronous execution.
        :type stream: cuda.CUstream
        """
        m, n, k, expert_l, input_l, route_l = problem_mnkl
        assert not self.use_bias and not self.use_scale
        a = cute.make_tensor(
            a_ptr, cute.make_ordered_layout((m, k, expert_l), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, cute.make_ordered_layout((n, k, input_l), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, cute.make_ordered_layout((m, n, route_l), order=(0, 1, 2))
        )
        self.a_dtype = a_ptr.value_type
        self.b_dtype = b_ptr.value_type
        self.c_dtype = c_ptr.value_type
        if cutlass.const_expr(self.use_scale):
            assert scale_ptr is not None
            scale = cute.make_tensor(scale_ptr, cute.make_layout((1,)))
        else:
            assert scale_ptr is None
            scale = None
        if cutlass.const_expr(self.use_bias):
            assert bias_ptr is not None
            bias = cute.make_tensor(
                bias_ptr,
                cute.make_layout((m, n, route_l), stride=(1, 0, 0)),
            )
            self.bias_dtype = bias_ptr.value_type
        else:
            assert bias_ptr is None
            bias = None
            self.bias_dtype = self.c_dtype
        self.sf_dtype = sfa_ptr.value_type
        self.mxf8f6f4 = self.needs_unpack_tma(self.a_dtype, self.b_dtype)
        self.smem_alloc_a_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.a_dtype.width < 8 else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Int8 if self.mxf8f6f4 and self.b_dtype.width < 8 else self.b_dtype
        )
        self.a_major_mode = cutlass.utils.LayoutEnum.from_tensor(a).mma_major_mode()
        self.b_major_mode = cutlass.utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.c_layout = cutlass.utils.LayoutEnum.from_tensor(c)
        sfa = cute.make_tensor(
            sfa_ptr, blockscaled_utils.tile_atom_to_shape_SF(a.shape, self.sf_vec_size)
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb = cute.make_tensor(
                sfb_ptr,
                cute.make_ordered_layout(
                    (n, cute.assume(k // self.sf_vec_size, 16), input_l),
                    order=(1, 0, 2),
                ),
            )
        else:
            # Give each CTA N tile a separate 128-row SFB atom
            n_padded_sfb = cute.ceil_div(n, self.cta_n) * cute.round_up(self.cta_n, 128)
            b_shape_for_sfb = cute.make_ordered_layout(
                (n_padded_sfb, k, input_l), order=(1, 0, 2)
            ).shape
            sfb = cute.make_tensor(
                sfb_ptr,
                blockscaled_utils.tile_atom_to_shape_SF(
                    b_shape_for_sfb, self.sf_vec_size
                ),
            )
        self._setup_attributes()
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.cta_m, self.cta_n),
        )
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_a_dtype
            if self.mxf8f6f4 and self.a_dtype.width < 8
            else None,
        )
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            b,
            b_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=self.smem_alloc_b_dtype
            if self.mxf8f6f4 and self.b_dtype.width < 8
            else None,
        )
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_sfa, tma_tensor_sfa = cute.nvgpu.make_tiled_tma_atom_A(
            sfa_op,
            sfa,
            sfa_smem_layout,
            (self.cta_m, self.cta_n, self.cta_k),
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_smem_layout = cute.slice_(self._sfb_smem_layout_flat, (None, None, 0))
            sfb_k = self.cta_k // self.sf_vec_size
            sfb_tma_op = cpasync.CopyBulkTensorTileG2SOp(tcgen05.CtaGroup.ONE)
            tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
                sfb_tma_op,
                sfb,
                sfb_smem_layout,
                (self.cta_n, sfb_k),
            )
        else:
            sfb_op = sm100_utils.cluster_shape_to_tma_atom_B(
                self.cluster_shape_mn, self._tiled_mma_sfb.thr_id
            )
            sfb_smem_layout = cute.slice_(
                self._sfb_smem_layout_padded, (None, None, None, 0)
            )
            tma_atom_sfb, tma_tensor_sfb = cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                sfb,
                sfb_smem_layout,
                self._mma_tiler_sfb,
                self._tiled_mma_sfb,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Int16,
            )
        # Count input bytes before unpacking mixed-width operands
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.tma_bytes_a = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.tma_bytes_b = (b_copy_size + sfb_copy_size) * atom_thr_size
        grid = (
            cute.ceil_div(c.layout.shape[0], self.cta_m),
            cute.ceil_div(c.layout.shape[1], self.cta_n),
            c.layout.shape[2] * self.global_k_parts,
        )
        assert self.split_k == 1
        return (
            tiled_mma,
            self._tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_sfa,
            tma_tensor_sfa,
            tma_atom_sfb,
            tma_tensor_sfb,
            c,
            scale,
            bias,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self._sfb_smem_layout_flat,
            self._sfb_smem_layout_padded,
            route_ids,
        ), grid

    @cute.jit
    def device_body(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,  # (Gemm_M, Gemm_K, Gemm_L) — TMA coordinate tensor for A
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,  # (Gemm_N, Gemm_K, Gemm_L) — TMA coordinate tensor for B
        tma_atom_sfa: cute.CopyAtom,
        sfa: cute.Tensor,  # SFA TMA coordinate tensor
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,  # SFB TMA coordinate tensor
        mC_mnl: cute.Tensor,  # (Gemm_M, Gemm_N, Gemm_L) — output tensor in GMEM
        mScale: Optional[cute.Tensor],  # single output scale
        mBias_mnl: Optional[cute.Tensor],  # (Gemm_M, Gemm_N, Gemm_L):(1,0,0)
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        b_smem_layout_staged: cute.ComposedLayout,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_flat_staged: cute.Layout,  # compact 3D SFB layout
        sfb_smem_layout_padded_staged: cute.Layout,  # padded 4D SFB layout
        route_ids: cute.Tensor,
        tile_coords: Tuple,
        shared_base: cute.Pointer,
    ):
        """Eight warps: A/SFA load, B/SFB load, MMA, idle, four epilogue warps.

        Scale factors use the direct SMEM-to-TMEM path, so the original four
        register-mediated scale producer warps are not needed.
        """
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        tidx, _, _ = cute.arch.thread_idx()
        bidx, bidy, bidz = tile_coords

        # Prefetch TMA descriptors (warp 0 only)
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_sfb)

        # Derive the split rank and batch index from the cluster topology
        # Independent CTAs produce FP32 partials; cluster topology stays1.
        # Runtime expert/input/output batches are independent; each split has a disjoint C view.
        split_rank = bidz % self.global_k_parts
        route_idx = bidz // self.global_k_parts
        expert_idx = route_ids[route_idx]
        input_idx = route_idx // self.input_route_divisor
        mC_mnl = cute.make_tensor(
            mC_mnl.iterator + split_rank * cute.size(mC_mnl), mC_mnl.layout
        )

        # Assign this CTA its global split-K range
        total_k_tiles = cute.ceil_div(cute.size(mA_mkl, mode=[1]), self.cta_k)
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        k_tiles_per_split = cute.ceil_div(total_k_tiles, self.global_k_parts)
        k_start = split_rank * k_tiles_per_split
        if k_start > total_k_tiles:
            k_start = total_k_tiles
        k_end = k_start + k_tiles_per_split
        if k_end > total_k_tiles:
            k_end = total_k_tiles
        work_tile_info = WorkTileInfo(
            M_idx=bidx // cute.size(tiled_mma.thr_id.shape),
            N_idx=bidy,
            L_idx=route_idx,
            K_idx_start=k_start,
            K_idx_end=k_end,
        )
        work_tile_a = WorkTileInfo(
            work_tile_info.M_idx, bidy, expert_idx, k_start, k_end
        )
        work_tile_b = WorkTileInfo(
            work_tile_info.M_idx, bidy, input_idx, k_start, k_end
        )
        k_tile_count = work_tile_info.K_idx_end - work_tile_info.K_idx_start

        cta_rank_in_cluster = cutlass.Int32(0)
        # tma_partition uses the M/N coordinates to build multicast masks
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        # SMEM allocation (SharedStorage struct for barriers + TMEM ptr)
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # AB full barriers track both arrivals and TMA transaction bytes. The
        # remaining barriers use plain arrive/wait synchronization

        storage = self.shared_storage(shared_base)

        # Defer synchronization until every pipeline and standalone barrier is ready
        # Two TMA producer warps feed one MMA consumer warp
        ab_pipeline = PipelineAsync.create(
            barrier_storage=storage.ab_pipeline_bars.data_ptr(),
            num_stages=DMA_Stage,
            producer_group=CooperativeGroup(Agent.Thread, 2),
            consumer_group=CooperativeGroup(Agent.Thread),
            defer_sync=True,
        )
        ab_full_bar = ab_pipeline.sync_object_full.barrier_storage
        ab_empty_bar = ab_pipeline.sync_object_empty.barrier_storage

        # One-stage MMA-to-epilogue completion pipeline
        mma_epilog_pipeline = PipelineUmmaAsync.create(
            barrier_storage=storage.mma_epilog_bars.data_ptr(),
            num_stages=1,
            producer_group=CooperativeGroup(Agent.Thread),
            consumer_group=CooperativeGroup(Agent.Thread, 128),
            defer_sync=True,
        )
        mma_epilog_full_bar = mma_epilog_pipeline.sync_object_full.barrier_storage

        # Raw barrier pointers
        tmem_alloc_result_bar = storage.tmem_allocation_result_barrier.ptr
        tma_epilog_full_bar = storage.tma_epilog_full_barrier.ptr
        tmem_base_smem_ptr = storage.tmem_base_ptr.ptr
        mailbox_mbar = storage.dsmem_mailbox_barrier.ptr
        dsmem_mailbox_ptr = storage.dsmem_mailbox.data_ptr()
        # The register-mediated path has a separate SFB pipeline
        if cutlass.const_expr(self.sfb_tmem_store):
            sfb_pipeline = PipelineAsyncUmma.create(
                barrier_storage=storage.sfb_pipeline_bars.data_ptr(),
                num_stages=SFB_Stage,
                producer_group=CooperativeGroup(Agent.Thread, 4),
                consumer_group=CooperativeGroup(Agent.Thread),
                defer_sync=True,
            )
            sfb_full_bar = sfb_pipeline.sync_object_full.barrier_storage
            sfb_empty_bar = sfb_pipeline.sync_object_empty.barrier_storage
        else:
            sfb_full_bar = storage.sfb_pipeline_bars.data_ptr()
            sfb_empty_bar = sfb_full_bar + SFB_Stage

        # Initialize standalone TMEM and split-K barriers
        if warp_idx == 0:
            with cute.arch.elect_one():
                # The TMEM barrier covers allocation and release across MMA,
                # epilogue, and the optional SFB warps
                if cutlass.const_expr(self.sfb_tmem_store):
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128 + 128)
                else:
                    cute.arch.mbarrier_init(tmem_alloc_result_bar, 32 + 128)
                if cutlass.const_expr(self.use_bias):
                    # All 32 B-load threads arrive after issuing activation loads
                    cute.arch.mbarrier_init(tma_epilog_full_bar, 32)
                if self.split_k > 1:
                    cute.arch.mbarrier_init(mailbox_mbar, 1)

        # Publish barrier initialization to every warp
        cluster_layout = (
            cute.make_layout((*self.cluster_shape_mn, self.split_k))
            if self.split_k > 1
            else None
        )
        pipeline_init_arrive(cluster_shape_mn=cluster_layout, is_relaxed=True)
        pipeline_init_wait(cluster_shape_mn=cluster_layout)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB_flat = storage.sSFB_flat.get_tensor(sfb_smem_layout_flat_staged)
        sSFB_padded = storage.sSFB_padded.get_tensor(sfb_smem_layout_padded_staged)

        # Warps 0/1 load, warp 2 computes, warp 3 is idle, warps 4-7 store.
        if warp_idx == 0:
            self.dma_a_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_atom_a,
                mA_mkl,
                sA,
                tma_atom_sfa,
                sfa,
                sSFA,
                tiled_mma,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_a,
                k_tile_count,
            )
        elif warp_idx == 1:
            sSFB = sSFB_flat if cutlass.const_expr(self.sfb_tmem_store) else sSFB_padded
            self.dma_b_warp(
                ab_full_bar,
                ab_empty_bar,
                tma_epilog_full_bar,
                tma_atom_b,
                mB_nkl,
                sB,
                tma_atom_sfb,
                mSFB_nkl,
                sSFB,
                tiled_mma,
                tiled_mma_sfb,
                cluster_layout_vmnk,
                block_in_cluster_coord_vmnk,
                mma_tile_coord_v,
                work_tile_b,
                k_tile_count,
            )
        elif warp_idx == 2:
            self.mma_warp(
                ab_full_bar,
                ab_empty_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                sfb_full_bar,
                sfb_empty_bar,
                tiled_mma,
                sA,
                sB,
                sSFA,
                sSFB_padded,
                sfa_smem_layout_staged,
                sfb_smem_layout_padded_staged,
                tmem_base_smem_ptr,
                k_tile_count,
            )
        elif warp_idx >= 4:
            epi_tid = tidx - 128
            self.epilog_warp(
                tma_epilog_full_bar,
                mma_epilog_full_bar,
                tmem_alloc_result_bar,
                tiled_mma,
                mC_mnl,
                mScale,
                mBias_mnl,
                tmem_base_smem_ptr,
                mailbox_mbar,
                dsmem_mailbox_ptr,
                epi_tid,
                mma_tile_coord_v,
                work_tile_info,
                split_rank,
                k_tile_count,
            )

        # Register-mediated path only: warps 3-6 stage SFB into TMEM (idle on direct copy)
        if cutlass.const_expr(self.sfb_tmem_store):
            if warp_idx >= 3 and warp_idx <= 6:
                self.sfb_warp(
                    ab_full_bar,
                    sfb_full_bar,
                    sfb_empty_bar,
                    tmem_alloc_result_bar,
                    tmem_base_smem_ptr,
                    sSFB_flat,
                    k_tile_count,
                )

        # Final sync
        cute.arch.barrier()
        return

    @cute.jit
    def mma_warp(
        self,
        ab_full_bar,
        ab_empty_bar,
        mma_epilog_full_bar,
        tmem_alloc_result_bar,
        sfb_full_bar,
        sfb_empty_bar,
        tiled_mma: cute.TiledMma,
        sA: cute.Tensor,  # ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage)
        sB: cute.Tensor,  # ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage)
        sSFA: cute.Tensor,
        sSFB: cute.Tensor,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_padded_staged: cute.Layout,
        tmem_base_smem_ptr,
        k_tile_count: cutlass.Int32,
    ):
        """Perform block-scaled MMA with the staged scale factors."""
        DMA_Stage = self.num_ab_stage
        SFB_Stage = self.num_sfb_tmem_stage

        # A and B fragments are SMEM descriptors consumed by cute.gemm
        # tCrA: ((Mma_M, Mma_K), NumMma_M, NumMma_K, DMA_Stage) — SMEM descriptors for A
        tCrA = tiled_mma.make_fragment_A(sA)
        # tCrB: ((Mma_N, Mma_K), NumMma_N, NumMma_K, DMA_Stage) — SMEM descriptors for B
        tCrB = tiled_mma.make_fragment_B(sB)

        # Build the accumulator layout before attaching its TMEM base pointer
        # acc_shape / tCtAcc: ((Mma_M, Mma_N), NumMma_M, NumMma_N)
        acc_shape = tiled_mma.partition_shape_C((self.cta_m, self.cta_n))
        tCtAcc_fake = tiled_mma.make_fragment_C(acc_shape)

        # Bound TMEM to the live accumulator and scale tiles, allowing more CTAs.
        num_tmem_cols = (
            self.allocation_columns
        )  # same bound checked against all live tensors
        cute.arch.alloc_tmem(num_tmem_cols, tmem_base_smem_ptr)

        # All 32 MMA threads publish the allocation without waiting
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)

        # Relinquish TMEM allocation lock early so that the next prefetch CTA can be launched
        cute.arch.relinquish_tmem_alloc_permit()

        # Retrieve TMEM ptr from SMEM and create the accumulator tensor view
        tmem_ptr = cute.arch.retrieve_tmem_ptr(
            self.acc_dtype,
            16,
            tmem_base_smem_ptr,
        )
        tCtAcc = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)

        # SFA/SFB TMEM tensors and S2T copy partition
        sfa_tmem_ptr = cute.recast_ptr(tmem_ptr + self.cta_n, dtype=self.sf_dtype)
        sfa_mem_single = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))
        tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
            tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfa_mem_single
        )
        tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

        sfb_tmem_ptr = cute.recast_ptr(
            tmem_ptr + self._num_sfa_tmem_cols + self.cta_n, dtype=self.sf_dtype
        )

        # SFA: SMEM-to-TMEM copy
        (
            tiled_copy_s2t_sfa,
            tCsSFA_compact_s2t,
            tCtSFA_compact_s2t,
        ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)

        # MMA mainloop
        if cutlass.const_expr(self.sfb_tmem_store):
            # Register-mediated staging uses two N atoms per SFB subpartition
            sfb_kblock_stride = self._num_n_atoms * self._sf_per_mma_k
            sfb_stage_stride = self.mma_inst_tile_k * sfb_kblock_stride
            tCtSFB_layout_staged = cute.make_layout(
                (
                    (
                        (
                            (self._sf_atom_mn, self._num_n_atoms),
                            128 // self._sf_atom_mn,
                        ),
                        (self.sf_vec_size, self._sf_per_mma_k),
                    ),
                    1,
                    self.mma_inst_tile_k,
                    SFB_Stage,
                ),
                stride=(
                    (
                        (
                            (self._sfb_lane_stride, self._sf_per_mma_k),
                            self._sf_atom_mn * self._sfb_lane_stride,
                        ),
                        (0, 1),
                    ),
                    0,
                    sfb_kblock_stride,
                    sfb_stage_stride,
                ),
            )
            tCtSFB_staged = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout_staged)

            # Register-mediated SFB mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            sfb_state = make_pipeline_state(PipelineUserType.Consumer, SFB_Stage)
            smem_wait_done = cutlass.Boolean(False)
            sfb_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                old_sfb_index = sfb_state.index
                old_sfb_phase = sfb_state.phase
                sfb_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Wait for the SFB warps to finish this TMEM stage
                if ~sfb_wait_done:
                    cute.arch.mbarrier_wait(sfb_full_bar + old_sfb_index, old_sfb_phase)

                # Select the staged SFB tensor used by this GEMM
                tCtSFB_mma = tCtSFB_staged[(None, None, None, old_sfb_index)]

                # A*SFA * B*SFB -> Acc
                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB_mma],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    # Release the SFB and operand pipeline stages
                    tcgen05.commit(sfb_empty_bar + old_sfb_index, None, self.cta_group)
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

                sfb_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    sfb_wait_done = cute.arch.mbarrier_try_wait(
                        sfb_full_bar + sfb_state.index, sfb_state.phase
                    )
        else:
            # Direct staging reuses one standard SFB tensor memory stage
            sfb_smem_single = cute.slice_(
                sfb_smem_layout_padded_staged, (None, None, None, 0)
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma, tiled_mma.shape_mnk, self.sf_vec_size, sfb_smem_single
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            # Configure the direct SFB copy from shared to tensor memory
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            # Direct SFB staging mainloop
            ab_state = make_pipeline_state(PipelineUserType.Consumer, DMA_Stage)
            smem_wait_done = cutlass.Boolean(False)

            for k_tile in cutlass.range(k_tile_count):
                if ~smem_wait_done:
                    cute.arch.mbarrier_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                old_ab_index = ab_state.index
                ab_state.advance()

                smem_wait_done = cutlass.Boolean(False)
                if k_tile < (k_tile_count - 1):
                    smem_wait_done = cute.arch.mbarrier_try_wait(
                        ab_full_bar + ab_state.index, ab_state.phase
                    )

                # Copy SFA from shared memory to tensor memory
                s2t_stage_coord = (None, None, None, None, old_ab_index)
                cute.copy(
                    tiled_copy_s2t_sfa,
                    tCsSFA_compact_s2t[s2t_stage_coord],
                    tCtSFA_compact_s2t,
                )

                # Copy SFB directly from shared memory to tensor memory
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[s2t_stage_coord],
                    tCtSFB_compact_s2t,
                )

                tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                tile_crd = (None, None, None, old_ab_index)
                cute.gemm(
                    tiled_mma,
                    tCtAcc,
                    [tCrA[tile_crd], tCtSFA],
                    [tCrB[tile_crd], tCtSFB],
                    tCtAcc,
                )

                with cute.arch.elect_one():
                    tcgen05.commit(ab_empty_bar + old_ab_index, None, self.cta_group)

        # Signal that the accumulator is ready for the epilogue
        with cute.arch.elect_one():
            tcgen05.commit(mma_epilog_full_bar, None, self.cta_group)

        # Release the tensor memory allocation
        cute.arch.mbarrier_arrive(tmem_alloc_result_bar)
        cute.arch.mbarrier_wait(tmem_alloc_result_bar, 1)
        cute.arch.dealloc_tmem(tmem_ptr, num_tmem_cols)
