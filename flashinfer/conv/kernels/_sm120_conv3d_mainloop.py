# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# mypy: disable-error-code="attr-defined, call-overload"
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""SM120 NVFP4 Conv3d mainloop with direct A and compact SFA staging.

This module intentionally keeps the validated CUTLASS ping-pong consumer,
scheduler, and epilogue structure. The DMA warp directly gathers packed FP4 A
from physical-halo NDHWC and compact NDHWC16 scales into their standard SMEM
layouts with cp.async. A/SFA share the official ``PipelineCpAsync`` protocol,
while B/SFB retain the standard TMA pipeline; the two pipeline states advance
in lockstep for each K tile.
"""

from __future__ import annotations

# Preserve the validated CUTLASS DSL loop and register-allocation structure.
# ruff: noqa: B007, F841

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm120_utils

from ._sm120_blockscaled_dispatch import make_ldmatrix_atom


class CompactSfaPingpongMainloop:
    """Device kernel for physical-halo activation and compact C16 scales."""

    @cute.jit
    def p3_epilog_sync(self, warp_group_idx):
        if cutlass.const_expr(self.p3_parallel_epilogue):
            if warp_group_idx == 0:
                self.p3_epilog_sync_barrier_0.arrive_and_wait()
            else:
                self.p3_epilog_sync_barrier_1.arrive_and_wait()
        else:
            self.epilog_sync_barrier.arrive_and_wait()

    @cute.kernel
    def conv_kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        mA_ndhwc: cute.Tensor,
        mA_index_fm: cute.Tensor,
        mA_zero: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        tma_atom_sfb: cute.CopyAtom,
        mSFB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        mC_raw: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
        conv_T: cutlass.Constexpr,
        conv_R: cutlass.Constexpr,
        conv_S: cutlass.Constexpr,
        stride_d: cutlass.Constexpr,
        stride_h: cutlass.Constexpr,
        stride_w: cutlass.Constexpr,
        dil_d: cutlass.Constexpr,
        dil_h: cutlass.Constexpr,
        dil_w: cutlass.Constexpr,
        pad_d: cutlass.Constexpr,
        pad_h: cutlass.Constexpr,
        pad_w: cutlass.Constexpr,
        input_D: cutlass.Constexpr,
        input_H: cutlass.Constexpr,
        input_W: cutlass.Constexpr,
        output_Z: cutlass.Constexpr,
        output_P: cutlass.Constexpr,
        output_Q: cutlass.Constexpr,
        input_N: cutlass.Constexpr,
        input_C: cutlass.Constexpr,
        K_gemm_tile: cutlass.Constexpr,
    ):
        # The producer contract is deliberately narrow. The physical halo turns
        # Conv padding into zero and C being tile-K aligned means one K tile never
        # crosses a filter-position boundary.
        assert pad_d == 0 and pad_h == 0 and pad_w == 0
        assert K_gemm_tile == self.tile_shape_mnk[2]
        assert input_C % K_gemm_tile == 0
        assert K_gemm_tile % (self.sf_vec_size * 4) == 0
        n_tiles_per_work = 2 if self.p3_n_pair else 1

        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = (
            cute.size_in_bytes(self.b_dtype, b_smem_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        ) * n_tiles_per_work

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        mainloop_pipeline_array_ptr = storage.mainloop_pipeline_array_ptr.data_ptr()
        sfa_pipeline_array_ptr = storage.sfa_pipeline_array_ptr.data_ptr()
        math_wg_order_barrier_array_ptr = (
            storage.math_wg_order_barrier_array_ptr.data_ptr()
        )

        mainloop_consumer_warps = self.num_mma_warps // 2
        if cutlass.const_expr(self.p3_n_pair):
            mainloop_consumer_warps = self.num_mma_warps
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, mainloop_consumer_warps
            ),
            tx_count=tma_copy_bytes,
            barrier_storage=mainloop_pipeline_array_ptr,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
        )
        direct_a_producer_threads = self.num_threads_per_warp
        if cutlass.const_expr(self.p3_a_copy_layout == "coalesced"):
            direct_a_producer_threads *= self.p3_a_producer_warps
        sfa_consumer_threads = (self.num_mma_warps // 2) * self.num_threads_per_warp
        if cutlass.const_expr(self.p3_n_pair):
            sfa_consumer_threads = self.num_mma_warps * self.num_threads_per_warp
        sfa_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=sfa_pipeline_array_ptr,
            num_stages=self.ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, direct_a_producer_threads
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                sfa_consumer_threads,
            ),
            defer_sync=True,
            name="flashinfer_conv3d_nvfp4_direct_a_compact_sfa",
        )
        warp_group_idx = cute.arch.make_warp_uniform(tidx // 128)
        math_wg_order_barrier = self.make_and_init_order_barrier(
            math_wg_order_barrier_array_ptr, warp_group_idx
        )
        cute.arch.mbarrier_init_fence()
        math_wg_order_state = math_wg_order_barrier.state

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)

        gA_mkl = cute.local_tile(
            mA_mkl,
            cute.slice_(self.tile_shape_mnk, (None, 0, None)),
            (None, None, None),
        )
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gSFB_nkl = cute.local_tile(
            mSFB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        gC_mnl = cute.local_tile(
            mC_mnl,
            cute.slice_(self.tile_shape_mnk, (None, None, 0)),
            (None, None, None),
        )

        thr_mma = tiled_mma.get_slice(tidx % 128)
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        b_cta_crd = cluster_coord_mnk[0]
        tBsB, tBgB = cpasync.tma_partition(
            tma_atom_b,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )
        tBsSFB, tBgSFB = cpasync.tma_partition(
            tma_atom_sfb,
            b_cta_crd,
            b_cta_layout,
            cute.group_modes(sSFB, 0, 2),
            cute.group_modes(gSFB_nkl, 0, 2),
        )
        tBsSFB = cute.filter_zeros(tBsSFB)
        tBgSFB = cute.filter_zeros(tBgSFB)

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm120_utils.partition_fragment_SFA(sSFA[None, None, 0], thr_mma, tidx)
        tCrSFB = sm120_utils.partition_fragment_SFB(sSFB[None, None, 0], thr_mma, tidx)
        tCgC = thr_mma.partition_C(gC_mnl)
        tile_c_coord = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1])
        )
        tCcC = thr_mma.partition_C(tile_c_coord)
        accumulators = cute.make_rmem_tensor(tCgC.shape[:3], self.acc_dtype)

        cute.arch.sync_threads()

        k_tile_cnt = cute.size(gA_mkl, mode=[3])
        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        mainloop_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        mainloop_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        sfa_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.ab_stage
        )
        sfa_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.ab_stage
        )
        if warp_idx >= self.tma_load_warp_id:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            direct_a_producer_warps = 1
            if cutlass.const_expr(self.p3_a_copy_layout == "coalesced"):
                direct_a_producer_warps = self.p3_a_producer_warps
            if warp_idx < self.tma_load_warp_id + direct_a_producer_warps:
                lane = tidx % self.num_threads_per_warp
                producer_warp = warp_idx - self.tma_load_warp_id
                a_copy_bits = self.p3_a_copy_bits
                a_copy_elems = a_copy_bits // self.a_dtype.width
                a_atom_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS
                    ),
                    self.a_dtype,
                    num_bits_per_copy=a_copy_bits,
                )
                sfa_atom_copy = cute.make_copy_atom(
                    cute.nvgpu.cpasync.CopyG2SOp(
                        cache_mode=cute.nvgpu.LoadCacheMode.ALWAYS
                    ),
                    mSFA_mkl.element_type,
                    num_bits_per_copy=32,
                )
                sfa_pred = cute.make_rmem_tensor((1,), cutlass.Boolean)
                a_vec_layout = cute.make_layout((a_copy_elems,), stride=(1,))
                sf_vec4_layout = cute.make_layout((4,), stride=(1,))
                output_pq = output_P * output_Q
                output_zpq = output_Z * output_pq
                input_hw = input_H * input_W
                input_dhw = input_D * input_hw
                output_m = input_N * output_zpq
                storage_sf_groups = cute.size(mSFA_mkl, mode=[1])
                sfa_chunks_per_tile = K_gemm_tile // (self.sf_vec_size * 4)
                k_tiles_per_filter_position = input_C // K_gemm_tile
                sfa_stage_stride = cute.cosize(sfa_smem_layout)
                if cutlass.const_expr(self.p3_a_copy_layout == "coalesced"):
                    a_vectors_per_row = K_gemm_tile // a_copy_elems
                    a_rows_per_instruction = (
                        self.num_threads_per_warp // a_vectors_per_row
                    )
                    a_rows_per_warp = self.tile_shape_mnk[0] // self.p3_a_producer_warps
                    a_row_iterations = a_rows_per_warp // a_rows_per_instruction
                    a_lane_k = lane % a_vectors_per_row
                    a_lane_row = lane // a_vectors_per_row
                    voxel_bases = cute.make_rmem_tensor(
                        (a_row_iterations,), cutlass.Int32
                    )
                else:
                    voxel_bases = cute.make_rmem_tensor((4,), cutlass.Int32)
                    row_valid = cute.make_rmem_tensor((4,), cutlass.Boolean)

                while work_tile.is_valid_tile:
                    tile_coord_mnl = work_tile.tile_idx
                    cta_m_offset = tile_coord_mnl[0] * self.tile_shape_mnk[0]

                    # Physical halo makes every filter position an affine
                    # offset from a row's source voxel. Cache that base once
                    # per output tile instead of repeating flat-M divisions
                    # in every K stage.
                    if cutlass.const_expr(self.p3_a_copy_layout == "coalesced"):
                        for row_iter in cutlass.range_constexpr(a_row_iterations):
                            pre_coal_local_m = (
                                producer_warp * a_rows_per_warp
                                + a_lane_row
                                + a_rows_per_instruction * row_iter
                            )
                            pre_coal_m_global = cta_m_offset + pre_coal_local_m
                            pre_coal_m_valid = pre_coal_m_global < output_m
                            pre_coal_m_safe = (
                                pre_coal_m_global if pre_coal_m_valid else 0
                            )
                            pre_coal_n = pre_coal_m_safe // output_zpq
                            pre_coal_zpq = pre_coal_m_safe - pre_coal_n * output_zpq
                            pre_coal_z = pre_coal_zpq // output_pq
                            pre_coal_pq = pre_coal_zpq - pre_coal_z * output_pq
                            pre_coal_p = pre_coal_pq // output_Q
                            pre_coal_q = pre_coal_pq - pre_coal_p * output_Q
                            voxel_bases[row_iter] = (
                                pre_coal_n * input_dhw
                                + pre_coal_z * stride_d * input_hw
                                + pre_coal_p * stride_h * input_W
                                + pre_coal_q * stride_w
                            )
                    else:
                        for row_quad in cutlass.range_constexpr(4):
                            m_global = cta_m_offset + lane + 32 * row_quad
                            m_is_valid = m_global < output_m
                            m_safe = m_global if m_is_valid else 0
                            n_idx = m_safe // output_zpq
                            zpq_rem = m_safe - n_idx * output_zpq
                            z_idx = zpq_rem // output_pq
                            pq_rem = zpq_rem - z_idx * output_pq
                            p_idx = pq_rem // output_Q
                            q_idx = pq_rem - p_idx * output_Q
                            voxel_bases[row_quad] = (
                                n_idx * input_dhw
                                + z_idx * stride_d * input_hw
                                + p_idx * stride_h * input_W
                                + q_idx * stride_w
                            )
                            row_valid[row_quad] = m_is_valid

                    sfa_producer_state.reset_count()
                    for k_tile in range(0, k_tile_cnt, 1, unroll=1):
                        if warp_idx == self.tma_load_warp_id:
                            mainloop_pipeline.producer_acquire(sfa_producer_state)
                        sfa_pipeline.producer_acquire(sfa_producer_state)

                        fpos_idx = (
                            sfa_producer_state.count // k_tiles_per_filter_position
                        )
                        channel_tile_idx = (
                            sfa_producer_state.count
                            - fpos_idx * k_tiles_per_filter_position
                        )
                        channel_base = channel_tile_idx * K_gemm_tile
                        channel_group_base = channel_base // self.sf_vec_size
                        filter_t = fpos_idx // (conv_R * conv_S)
                        filter_rem = fpos_idx - filter_t * (conv_R * conv_S)
                        filter_r = filter_rem // conv_S
                        filter_s = filter_rem - filter_r * conv_S
                        filter_voxel_offset = (
                            filter_t * dil_d * input_hw
                            + filter_r * dil_h * input_W
                            + filter_s * dil_w
                        )
                        stage_base = sfa_producer_state.index * sfa_stage_stride
                        sA_stage = sA[None, None, sfa_producer_state.index]

                        if cutlass.const_expr(self.p3_a_copy_layout == "coalesced"):
                            for row_iter in cutlass.range_constexpr(a_row_iterations):
                                coal_local_m = (
                                    producer_warp * a_rows_per_warp
                                    + a_lane_row
                                    + a_rows_per_instruction * row_iter
                                )
                                coal_m_global = cta_m_offset + coal_local_m
                                coal_voxel = voxel_bases[row_iter] + filter_voxel_offset
                                coal_local_k = a_lane_k * a_copy_elems
                                coal_src_elem = cute.assume(
                                    coal_voxel * input_C + channel_base + coal_local_k,
                                    divby=a_copy_elems,
                                )
                                coal_g_src = cute.make_tensor(
                                    mA_ndhwc.iterator + coal_src_elem,
                                    a_vec_layout,
                                )
                                coal_smem_a_elem = cute.assume(
                                    sA_stage.layout((coal_local_m, coal_local_k)),
                                    divby=a_copy_elems,
                                )
                                coal_s_dst = cute.make_tensor(
                                    sA_stage.iterator + coal_smem_a_elem,
                                    a_vec_layout,
                                )
                                cute.copy_atom_call(
                                    a_atom_copy,
                                    coal_g_src,
                                    coal_s_dst,
                                )

                                if a_lane_k < sfa_chunks_per_tile:
                                    coal_smem_row_base = (coal_local_m % 32) * 16 + (
                                        coal_local_m // 32
                                    ) * 4
                                    coal_sfa_src_elem = cute.assume(
                                        coal_voxel * storage_sf_groups
                                        + channel_group_base
                                        + a_lane_k * 4,
                                        divby=4,
                                    )
                                    coal_smem_elem = cute.assume(
                                        stage_base
                                        + coal_smem_row_base
                                        + 512 * a_lane_k,
                                        divby=4,
                                    )
                                    coal_g_sfa_src = cute.make_tensor(
                                        mSFA_mkl.iterator + coal_sfa_src_elem,
                                        sf_vec4_layout,
                                    )
                                    coal_s_sfa_dst = cute.make_tensor(
                                        sSFA.iterator + coal_smem_elem,
                                        sf_vec4_layout,
                                    )
                                    sfa_pred[0] = coal_m_global < output_m
                                    cute.copy_atom_call(
                                        sfa_atom_copy,
                                        coal_g_sfa_src,
                                        coal_s_sfa_dst,
                                        pred=sfa_pred,
                                    )
                        else:
                            for row_quad in cutlass.range_constexpr(4):
                                voxel = voxel_bases[row_quad] + filter_voxel_offset
                                row_local_m = lane + 32 * row_quad
                                smem_row_base = lane * 16 + row_quad * 4
                                sfa_pred[0] = row_valid[row_quad]

                                for a_chunk in cutlass.range_constexpr(
                                    K_gemm_tile // a_copy_elems
                                ):
                                    local_k = a_chunk * a_copy_elems
                                    src_elem = cute.assume(
                                        voxel * input_C + channel_base + local_k,
                                        divby=a_copy_elems,
                                    )
                                    g_src = cute.make_tensor(
                                        mA_ndhwc.iterator + src_elem,
                                        a_vec_layout,
                                    )
                                    s_dst = cute.make_tensor(
                                        sA_stage.iterator
                                        + sA_stage.layout((row_local_m, local_k)),
                                        a_vec_layout,
                                    )
                                    cute.copy_atom_call(
                                        a_atom_copy,
                                        g_src,
                                        s_dst,
                                    )

                                for chunk in cutlass.range_constexpr(
                                    sfa_chunks_per_tile
                                ):
                                    src_elem = (
                                        voxel * storage_sf_groups
                                        + channel_group_base
                                        + chunk * 4
                                    )
                                    smem_elem = stage_base + smem_row_base + 512 * chunk
                                    g_src = cute.make_tensor(
                                        mSFA_mkl.iterator + src_elem,
                                        sf_vec4_layout,
                                    )
                                    s_dst = cute.make_tensor(
                                        sSFA.iterator + smem_elem,
                                        sf_vec4_layout,
                                    )
                                    cute.copy_atom_call(
                                        sfa_atom_copy,
                                        g_src,
                                        s_dst,
                                        pred=sfa_pred,
                                    )

                        sfa_pipeline.producer_commit(sfa_producer_state)

                        if warp_idx == self.tma_load_warp_id:
                            mainloop_barrier = mainloop_pipeline.producer_get_barrier(
                                sfa_producer_state
                            )
                            for n_in_work in cutlass.range_constexpr(n_tiles_per_work):
                                n_tile = (
                                    tile_coord_mnl[1] * n_tiles_per_work + n_in_work
                                )
                                b_stage = (
                                    sfa_producer_state.index + n_in_work * self.ab_stage
                                )
                                cute.copy(
                                    tma_atom_b,
                                    tBgB[
                                        (
                                            None,
                                            n_tile,
                                            None,
                                            tile_coord_mnl[2],
                                        )
                                    ][(None, sfa_producer_state.count)],
                                    tBsB[(None, b_stage)],
                                    tma_bar_ptr=mainloop_barrier,
                                )
                                cute.copy(
                                    tma_atom_sfb,
                                    tBgSFB[
                                        (
                                            None,
                                            n_tile,
                                            None,
                                            tile_coord_mnl[2],
                                        )
                                    ][(None, sfa_producer_state.count)],
                                    tBsSFB[(None, b_stage)],
                                    tma_bar_ptr=mainloop_barrier,
                                )
                            mainloop_pipeline.producer_commit(sfa_producer_state)
                        sfa_producer_state.advance()

                    tile_sched.advance_to_next_work()
                    work_tile = tile_sched.get_current_work()

                if warp_idx == self.tma_load_warp_id:
                    mainloop_tail_state = sfa_producer_state.clone()
                    mainloop_pipeline.producer_tail(mainloop_tail_state)
                sfa_pipeline.producer_tail(sfa_producer_state)

        elif warp_idx < self.tma_load_warp_id:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            num_k_blocks = cute.size(tCrA, mode=[2])

            atom_copy_ldmatrix_a = make_ldmatrix_atom(
                self.a_dtype,
                transpose=self.a_layout.is_m_major_a(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            atom_copy_ldmatrix_b = make_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b(),
                num_matrices=4,
                mixed_mode=self.mixed_mode,
            )
            smem_tiled_copy_a = cute.make_tiled_copy_A(atom_copy_ldmatrix_a, tiled_mma)
            smem_tiled_copy_b = cute.make_tiled_copy_B(atom_copy_ldmatrix_b, tiled_mma)
            atom_copy_ldmatrix_sf = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.sf_dtype
            )
            smem_tiled_copy_sfa = cute.make_tiled_copy(
                atom_copy_ldmatrix_sf,
                sm120_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_sfb = cute.make_tiled_copy(
                atom_copy_ldmatrix_sf,
                sm120_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )

            thr_copy_ldmatrix_a = smem_tiled_copy_a.get_slice(tidx % 128)
            thr_copy_ldmatrix_b = smem_tiled_copy_b.get_slice(tidx % 128)
            tCsA_copy_view = thr_copy_ldmatrix_a.partition_S(sA)
            tCrA_copy_view = thr_copy_ldmatrix_a.retile(tCrA)
            tCsB_copy_view = thr_copy_ldmatrix_b.partition_S(sB)
            tCrB_copy_view = thr_copy_ldmatrix_b.retile(tCrB)
            thr_copy_ldmatrix_sfa = smem_tiled_copy_sfa.get_slice(tidx % 128)
            thr_copy_ldmatrix_sfb = smem_tiled_copy_sfb.get_slice(tidx % 128)
            tCsSFA_copy_view = thr_copy_ldmatrix_sfa.partition_S(sSFA)
            tCrSFA_copy_view = thr_copy_ldmatrix_sfa.retile(tCrSFA)
            tCsSFB_copy_view = thr_copy_ldmatrix_sfb.partition_S(sSFB)
            tCrSFB_copy_view = thr_copy_ldmatrix_sfb.retile(tCrSFB)

            if cutlass.const_expr(not self.p3_n_pair):
                if warp_group_idx == 1:
                    tile_sched.advance_to_next_work()
                    mainloop_consumer_state = self.advance(
                        mainloop_consumer_state, k_tile_cnt
                    )
                    sfa_consumer_state = self.advance(sfa_consumer_state, k_tile_cnt)
                    work_tile = tile_sched.get_current_work()

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                consumer_tile_coord_mnl = tile_coord_mnl
                if cutlass.const_expr(self.p3_n_pair):
                    consumer_tile_coord_mnl = (
                        tile_coord_mnl[0],
                        tile_coord_mnl[1] * n_tiles_per_work + warp_group_idx,
                        tile_coord_mnl[2],
                    )
                gC_mnl_slice = gC_mnl[(None, None, *consumer_tile_coord_mnl)]
                accumulators.fill(0.0)
                mainloop_consumer_state.reset_count()
                sfa_consumer_state.reset_count()
                if cutlass.const_expr(not self.p3_n_pair):
                    math_wg_order_barrier.wait(math_wg_order_state)

                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_state
                    )
                peek_sfa_full_status = cutlass.Boolean(1)
                if sfa_consumer_state.count < k_tile_cnt:
                    peek_sfa_full_status = sfa_pipeline.consumer_try_wait(
                        sfa_consumer_state
                    )
                mainloop_pipeline.consumer_wait(
                    mainloop_consumer_state, peek_ab_full_status
                )
                sfa_pipeline.consumer_wait(sfa_consumer_state, peek_sfa_full_status)

                tCsA_p = tCsA_copy_view[None, None, None, mainloop_consumer_state.index]
                b_consumer_stage = mainloop_consumer_state.index
                if cutlass.const_expr(self.p3_n_pair):
                    b_consumer_stage += warp_group_idx * self.ab_stage
                tCsB_p = tCsB_copy_view[None, None, None, b_consumer_stage]
                tCsSFA_p = tCsSFA_copy_view[
                    None, None, None, mainloop_consumer_state.index
                ]
                tCsSFB_p = tCsSFB_copy_view[None, None, None, b_consumer_stage]
                cute.copy(
                    smem_tiled_copy_a,
                    tCsA_p[None, None, 0],
                    tCrA_copy_view[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_b,
                    tCsB_p[None, None, 0],
                    tCrB_copy_view[None, None, 0],
                )
                tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                cute.copy(
                    smem_tiled_copy_sfa,
                    tCsSFA_p_filtered[None, None, 0],
                    tCrSFA_copy_view_filtered[None, None, 0],
                )
                cute.copy(
                    smem_tiled_copy_sfb,
                    tCsSFB_p_filtered[None, None, 0],
                    tCrSFB_copy_view_filtered[None, None, 0],
                )

                for _k_tile in range(0, k_tile_cnt - 1, 1, unroll=1):
                    for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                        k_block_next = (
                            0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                        )
                        if k_block_idx == num_k_blocks - 1:
                            mainloop_pipeline.consumer_release(mainloop_consumer_state)
                            mainloop_consumer_state.advance()
                            sfa_pipeline.consumer_release(sfa_consumer_state)
                            sfa_consumer_state.advance()

                            peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                                mainloop_consumer_state
                            )
                            peek_sfa_full_status = sfa_pipeline.consumer_try_wait(
                                sfa_consumer_state
                            )
                            tCsA_p = tCsA_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            b_consumer_stage = mainloop_consumer_state.index
                            if cutlass.const_expr(self.p3_n_pair):
                                b_consumer_stage += warp_group_idx * self.ab_stage
                            tCsB_p = tCsB_copy_view[
                                None,
                                None,
                                None,
                                b_consumer_stage,
                            ]
                            tCsSFA_p = tCsSFA_copy_view[
                                None,
                                None,
                                None,
                                mainloop_consumer_state.index,
                            ]
                            tCsSFB_p = tCsSFB_copy_view[
                                None,
                                None,
                                None,
                                b_consumer_stage,
                            ]
                            mainloop_pipeline.consumer_wait(
                                mainloop_consumer_state, peek_ab_full_status
                            )
                            sfa_pipeline.consumer_wait(
                                sfa_consumer_state, peek_sfa_full_status
                            )

                        cute.copy(
                            smem_tiled_copy_a,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_b,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                        cute.copy(
                            smem_tiled_copy_sfa,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_sfb,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            [
                                tCrA[None, None, k_block_idx],
                                tCrSFA[None, None, k_block_idx],
                            ],
                            [
                                tCrB[None, None, k_block_idx],
                                tCrSFB[None, None, k_block_idx],
                            ],
                            accumulators,
                        )

                for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                    k_block_next = (
                        0 if k_block_idx + 1 == num_k_blocks else k_block_idx + 1
                    )
                    if k_block_idx == num_k_blocks - 1:
                        mainloop_pipeline.consumer_release(mainloop_consumer_state)
                        mainloop_consumer_state.advance()
                        sfa_pipeline.consumer_release(sfa_consumer_state)
                        sfa_consumer_state.advance()

                    if k_block_next > 0:
                        cute.copy(
                            smem_tiled_copy_a,
                            tCsA_p[None, None, k_block_next],
                            tCrA_copy_view[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_b,
                            tCsB_p[None, None, k_block_next],
                            tCrB_copy_view[None, None, k_block_next],
                        )
                        tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_p)
                        tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                        tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                        tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                        cute.copy(
                            smem_tiled_copy_sfa,
                            tCsSFA_p_filtered[None, None, k_block_next],
                            tCrSFA_copy_view_filtered[None, None, k_block_next],
                        )
                        cute.copy(
                            smem_tiled_copy_sfb,
                            tCsSFB_p_filtered[None, None, k_block_next],
                            tCrSFB_copy_view_filtered[None, None, k_block_next],
                        )
                    cute.gemm(
                        tiled_mma,
                        accumulators,
                        [
                            tCrA[None, None, k_block_idx],
                            tCrSFA[None, None, k_block_idx],
                        ],
                        [
                            tCrB[None, None, k_block_idx],
                            tCrSFB[None, None, k_block_idx],
                        ],
                        accumulators,
                    )

                if cutlass.const_expr(self.p3_n_pair):
                    if cutlass.const_expr(not self.p3_parallel_epilogue):
                        math_wg_order_barrier.wait(math_wg_order_state)
                else:
                    math_wg_order_state = math_wg_order_barrier.arrive(
                        math_wg_order_state
                    )
                    math_wg_order_barrier.wait(math_wg_order_state)
                    mainloop_consumer_state = self.advance(
                        mainloop_consumer_state, k_tile_cnt
                    )
                    sfa_consumer_state = self.advance(sfa_consumer_state, k_tile_cnt)

                copy_atom_r2s = sm120_utils.sm120_get_smem_store_op(
                    self.c_layout,
                    elem_ty_d=self.c_dtype,
                    elem_ty_acc=self.acc_dtype,
                )
                copy_atom_c = cute.make_copy_atom(
                    cute.nvgpu.warp.StMatrix8x8x16bOp(self.c_layout.is_m_major_c(), 2),
                    self.c_dtype,
                )
                tiled_copy_c_atom = cute.make_tiled_copy_C_atom(copy_atom_c, tiled_mma)
                tiled_copy_r2s = cute.make_tiled_copy_S(
                    copy_atom_r2s, tiled_copy_c_atom
                )
                thr_copy_r2s = tiled_copy_r2s.get_slice(tidx % 128)
                tRS_sD = thr_copy_r2s.partition_D(sC)
                tRS_rAcc = tiled_copy_r2s.retile(accumulators)
                tRS_cC = tiled_copy_r2s.retile(tCcC)
                rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
                tRS_rD_layout = cute.make_layout(rD_shape[:3])
                tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
                sepi_for_tma_partition = cute.group_modes(sC, 0, 2)
                tcgc_for_tma_partition = cute.zipped_divide(gC_mnl_slice, self.epi_tile)
                bSG_sD, bSG_gD = cpasync.tma_partition(
                    tma_atom_c,
                    0,
                    cute.make_layout(1),
                    sepi_for_tma_partition,
                    tcgc_for_tma_partition,
                )
                tma_store_pipeline = pipeline.PipelineTmaStore.create(
                    num_stages=self.epi_stage,
                    producer_group=pipeline.CooperativeGroup(
                        pipeline.Agent.Thread,
                        self.num_mma_warps * self.num_threads_per_warp,
                    ),
                )
                epi_rest_m = bSG_gD.shape[1][0]
                epi_rest_n = bSG_gD.shape[1][1]
                epi_tile_m = self.epi_tile[0]
                epi_tile_n = self.epi_tile[1]
                mma_tile_m = self.tile_shape_mnk[0] // cute.size(tRS_rAcc, mode=[1])
                mma_tile_n = self.tile_shape_mnk[1] // cute.size(tRS_rAcc, mode=[2])

                for epi_m in cutlass.range_constexpr(epi_rest_m):
                    for epi_n in cutlass.range_constexpr(epi_rest_n):
                        tRS_rD_out = cute.make_rmem_tensor(
                            tRS_rD_layout.shape, self.c_dtype
                        )
                        mma_m_per_epi_m = epi_tile_m // mma_tile_m
                        mma_n_per_epi_n = epi_tile_n // mma_tile_n
                        for mma_n_in_epi in cutlass.range_constexpr(mma_n_per_epi_n):
                            for mma_m_in_epi in cutlass.range_constexpr(
                                mma_m_per_epi_m
                            ):
                                mma_n = epi_n * mma_n_per_epi_n + mma_n_in_epi
                                mma_m = epi_m * mma_m_per_epi_m + mma_m_in_epi
                                tRS_rD_slice = tRS_rD[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                tRS_rAcc_slice = tRS_rAcc[(None, mma_m, mma_n)]
                                tRS_cC_slice = tRS_cC[(None, mma_m, mma_n)]
                                tRS_rD_out_slice = tRS_rD_out[
                                    (None, mma_m_in_epi, mma_n_in_epi)
                                ]
                                for elem_idx in cutlass.range_constexpr(
                                    cute.size(tRS_rD_slice)
                                ):
                                    if cutlass.const_expr(
                                        not self.p3_fuse_bias
                                        and not self.p3_fuse_residual
                                    ):
                                        tRS_rD_slice[elem_idx] = tRS_rAcc_slice[
                                            elem_idx
                                        ]
                                    else:
                                        coord = tRS_cC_slice[elem_idx]
                                        global_m = (
                                            consumer_tile_coord_mnl[0]
                                            * self.tile_shape_mnk[0]
                                            + coord[0]
                                        )
                                        global_n = (
                                            consumer_tile_coord_mnl[1]
                                            * self.tile_shape_mnk[1]
                                            + coord[1]
                                        )
                                        value = cutlass.Float32(
                                            tRS_rAcc_slice[elem_idx]
                                        ) * cutlass.Float32(mA_zero[0])
                                        if cutlass.const_expr(
                                            self.p3_epilogue_mode == "strict"
                                        ):
                                            value = cutlass.Float32(
                                                value.to(self.c_dtype)
                                            )
                                        if cutlass.const_expr(self.p3_fuse_bias):
                                            value = value + cutlass.Float32(
                                                mA_zero[2 + global_n]
                                            )
                                            if cutlass.const_expr(
                                                self.p3_epilogue_mode == "strict"
                                            ):
                                                value = cutlass.Float32(
                                                    value.to(self.c_dtype)
                                                )
                                        if cutlass.const_expr(self.p3_fuse_residual):
                                            output_m = cute.size(mC_mnl, mode=[0])
                                            if (
                                                mA_zero[1] != 0.0
                                                and global_m < output_m
                                            ):
                                                output_n = cute.size(mC_mnl, mode=[1])
                                                residual_offset = (
                                                    global_m * output_n + global_n
                                                )
                                                residual = cute.make_tensor(
                                                    mA_index_fm.iterator
                                                    + residual_offset,
                                                    cute.make_layout(1),
                                                )
                                                value = value + cutlass.Float32(
                                                    residual[0]
                                                )
                                            if cutlass.const_expr(
                                                self.p3_epilogue_mode == "strict"
                                            ):
                                                value = cutlass.Float32(
                                                    value.to(self.c_dtype)
                                                )
                                        tRS_rD_out_slice[elem_idx] = value.to(
                                            self.c_dtype
                                        )

                        if cutlass.const_expr(
                            not self.p3_fuse_bias and not self.p3_fuse_residual
                        ):
                            epilogue_values = tRS_rD.load()
                            if cutlass.const_expr(self.p3_fuse_alpha):
                                epilogue_values = epilogue_values * cutlass.Float32(
                                    mA_zero[0]
                                )
                            tRS_rD_out.store(epilogue_values.to(self.c_dtype))
                        epi_buffer = (epi_m * epi_rest_n + epi_n) % self.epi_stage
                        if cutlass.const_expr(self.p3_parallel_epilogue):
                            epi_buffer += warp_group_idx * self.epi_stage
                        self.p3_epilog_sync(warp_group_idx)
                        cute.copy(
                            tiled_copy_r2s,
                            tRS_rD_out,
                            tRS_sD[(None, None, None, epi_buffer)],
                        )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.p3_epilog_sync(warp_group_idx)
                        if warp_idx % 4 == 0:
                            cute.copy(
                                tma_atom_c,
                                bSG_sD[(None, epi_buffer)],
                                bSG_gD[(None, (epi_m, epi_n))],
                            )
                            tma_store_pipeline.producer_commit()
                            tma_store_pipeline.producer_acquire()

                tile_sched.advance_to_next_work()
                if cutlass.const_expr(not self.p3_n_pair):
                    tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()
                tma_store_pipeline.producer_tail()
                if cutlass.const_expr(
                    not self.p3_n_pair or not self.p3_parallel_epilogue
                ):
                    math_wg_order_state = math_wg_order_barrier.arrive(
                        math_wg_order_state
                    )

        return
