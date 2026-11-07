# Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Grouped BF16/FP16 Hopper projection, adapted from CUTLASS dense_gemm.py.

One CTA discovers up to eight live routes to one expert. Weight-only TMA
feeds WGMMA while four warps gather disjoint activation rows. BF16/FP16
issue weight TMA before gathering; FP8 retains gather-before-TMA scheduling.
FC1 reads the original up/gate layout and applies the shared activation;
FC2 writes typed expert partials for fixed-order combine.
"""

import cuda.bindings.driver as cuda
import cutlass
from cutlass import cute
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from .route_members import route_members
from .block_scale import promote_block_scaled
from ..fma.activations import apply_activation
from ..quantization import _mul
from .....cute_dsl.fp4_common import cvt_f32_to_e4m3


class HopperGroupedProjection:
    def __init__(
        self,
        fc1,
        group_size,
        stages,
        tile_k,
        top_k,
        activation="swiglu",
        gated=True,
        parameters=None,
        block_scaled=False,
    ):
        if (
            group_size not in (2, 4, 8)
            or stages not in (2, 4, 8)
            or tile_k not in (64, 128)
        ):
            raise ValueError("Unsupported grouped Hopper projection tactic")
        self.fc1 = fc1
        self.split_gate = fc1 and gated
        self.group_size = group_size
        self.stages = stages
        self.tile_k = tile_k
        self.input_route_divisor = top_k if fc1 else 1
        self.activation = activation
        self.parameters = parameters
        self.block_scaled = block_scaled
        self.rows_per_tile = 32 if self.split_gate else 64
        self.acc_dtype = cutlass.Float32
        self.cluster_shape_mn = (1, 1)
        self.tile_shape_mnk = (64, 8, tile_k)
        self.atom_layout_mnk = (1, 1, 1)
        self.mma_warp_groups = 1
        self.num_threads_per_warp_group = 128
        self.threads_per_cta = 128
        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] not in [8, 16, 32, 64, 128, 256]:
            raise ValueError("CTA tile shape N must be 64/128/256")
        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            self.tile_k,
        )
        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.epi_tile = (64, 8)
        (self.ab_stage, self.epi_stage) = (self.stages, 2)
        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = self._make_smem_layouts(
            self.tile_shape_mnk,
            self.epi_tile,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
            self.c_dtype,
            self.c_layout,
            self.epi_stage,
        )

    @cute.jit
    def __call__(
        self,
        weights: cute.Tensor,
        inputs: cute.Tensor,
        ids: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
        dequant=None,
        mid_quant=None,
    ):
        a = cute.make_tensor(
            weights.iterator,
            cute.make_layout(
                (
                    cute.size(weights, mode=[1]),
                    cute.size(weights, mode=[2]),
                    cute.size(weights, mode=[0]),
                ),
                stride=(weights.stride[1], weights.stride[2], weights.stride[0]),
            ),
        )
        b = cute.make_tensor(
            inputs.iterator,
            cute.make_layout(
                (1, cute.size(inputs, mode=[1]), cute.size(inputs, mode=[0])),
                stride=(
                    cute.size(inputs, mode=[1]),
                    inputs.stride[1],
                    inputs.stride[0],
                ),
            ),
        )
        route_ids = cute.make_tensor(ids.iterator, cute.make_layout(cute.size(ids)))
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = cutlass.Float32
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.COL_MAJOR
        if cutlass.const_expr(
            self.a_dtype.width == 16 and self.a_dtype != self.b_dtype
        ):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(self.a_dtype.width != self.b_dtype.width):
            raise TypeError(
                f"Type width mismatch: {self.a_dtype.width} != {self.b_dtype.width}"
            )
        if cutlass.const_expr(
            self.a_dtype
            not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float8E4M3FN)
        ):
            raise TypeError("Grouped Hopper projection requires BF16, FP16 or E4M3")
        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
            if cutlass.const_expr(dequant is None or mid_quant is None):
                raise TypeError(
                    "Grouped E4M3 projection requires live calibration scales"
                )
        self._setup_attributes()
        if cutlass.const_expr(self.split_gate):
            weight_layout = cute.composition(
                self.a_smem_layout_staged,
                cute.make_layout(
                    (32, self.tile_k, self.ab_stage), stride=(1, 64, 64 * self.tile_k)
                ),
            )
        else:
            weight_layout = self.a_smem_layout_staged
        tma_atom_a, tma_tensor_a = self._make_tma_atoms_and_tensors(
            a, weight_layout, (self.rows_per_tile, self.tile_k), 1
        )
        grid = (cute.size(output, mode=[1]) // self.rows_per_tile, 1, cute.size(ids))

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(self.a_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(self.b_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage
        self.kernel(
            b,
            output,
            route_ids,
            dequant,
            mid_quant,
            tma_atom_a,
            tma_tensor_a,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )
        return

    @cute.kernel
    def kernel(
        self,
        raw_b: cute.Tensor,
        output: cute.Tensor,
        route_ids: cute.Tensor,
        dequant,
        mid_quant,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
    ):
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
        (bidx, bidy, bidz) = cute.arch.block_idx()
        (tidx, _, _) = cute.arch.thread_idx()
        (pid_m, pid_n) = (bidx, bidy)
        expert = route_ids[bidz].to(cutlass.Int64)
        if expert < 0 or expert >= cute.size(mA_mkl, mode=[2]):
            if tidx < self.rows_per_tile:
                output[bidz, pid_m * self.rows_per_tile + tidx] = output.element_type(0)
        else:
            members = route_members(route_ids, bidz, self.group_size)
            if members[0] >= 0:
                cta_rank_in_cluster = cute.arch.make_warp_uniform(
                    cute.arch.block_idx_in_cluster()
                )
                cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
                a_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=1
                )
                b_mcast_mask = cute.make_layout_image_mask(
                    cta_layout_mnk, cluster_coord_mnk, mode=0
                )
                a_mcast_mask = a_mcast_mask if self.is_a_mcast else 0
                b_mcast_mask = b_mcast_mask if self.is_b_mcast else 0
                a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
                tma_copy_bytes = cute.size_in_bytes(self.a_dtype, a_smem_layout)
                smem = cutlass.utils.SmemAllocator()
                storage = smem.allocate(self.shared_storage)
                mainloop_pipeline_array_ptr = (
                    storage.mainloop_pipeline_array_ptr.data_ptr()
                )
                mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread
                )
                mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
                num_warps = self.threads_per_cta // 32
                consumer_arrive_cnt = mcast_size * num_warps
                mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, consumer_arrive_cnt
                )
                cta_layout_vmnk = cute.make_layout((1, *cta_layout_mnk.shape))
                mainloop_pipeline = pipeline.PipelineTmaAsync.create(
                    barrier_storage=mainloop_pipeline_array_ptr,
                    num_stages=self.ab_stage,
                    producer_group=mainloop_pipeline_producer_group,
                    consumer_group=mainloop_pipeline_consumer_group,
                    tx_count=tma_copy_bytes,
                    cta_layout_vmnk=cta_layout_vmnk,
                    defer_sync=True,
                )
                pipeline_init_arrive(
                    cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
                )
                sA = storage.sA.get_tensor(
                    a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
                )
                sB = storage.sB.get_tensor(
                    b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
                )
                sC_ptr = cute.recast_ptr(
                    sA.iterator, epi_smem_layout_staged.inner, dtype=self.c_dtype
                )
                sC = cute.make_tensor(sC_ptr, epi_smem_layout_staged.outer)
                gA_mkl = cute.local_tile(
                    mA_mkl,
                    (self.rows_per_tile, 8, self.tile_k),
                    (pid_m, pid_n, None, route_ids[bidz]),
                    proj=(1, None, 1),
                )
                if cutlass.const_expr(self.split_gate):
                    gGate_mkl = cute.local_tile(
                        mA_mkl,
                        (32, 8, self.tile_k),
                        (
                            pid_m + cute.size(mA_mkl, mode=[0]) // 64,
                            pid_n,
                            None,
                            route_ids[bidz],
                        ),
                        proj=(1, None, 1),
                    )
                gC_mnl = cute.make_identity_tensor((64, 8))
                warp_group_idx = cute.arch.make_warp_uniform(
                    tidx // self.num_threads_per_warp_group
                )
                warp_group_thread_layout = cute.make_layout(
                    self.mma_warp_groups, stride=self.num_threads_per_warp_group
                )
                thr_mma = tiled_mma.get_slice(warp_group_thread_layout(warp_group_idx))
                tCgC = thr_mma.partition_C(gC_mnl)
                a_cta_layout = cute.make_layout(
                    cute.slice_(cta_layout_mnk, (0, None, 0)).shape
                )
                a_cta_crd = cluster_coord_mnk[1]
                if cutlass.const_expr(self.split_gate):
                    half_outer = cute.composition(
                        sA.layout,
                        cute.make_layout(
                            (32, self.tile_k, self.ab_stage),
                            stride=(1, 64, 64 * self.tile_k),
                        ),
                    )
                    sUp = cute.make_tensor(sA.iterator, half_outer)
                    # Slice the swizzled layout; row*K is not the gate offset.
                    sGate = cute.make_tensor(sA[(32, None, None)].iterator, half_outer)
                    sA_for_tma_partition = cute.group_modes(sUp, 0, 2)
                else:
                    sA_for_tma_partition = cute.group_modes(sA, 0, 2)
                gA_for_tma_partition = cute.group_modes(gA_mkl, 0, 2)
                (tAsA, tAgA_mkl) = cute.nvgpu.cpasync.tma_partition(
                    tma_atom_a,
                    a_cta_crd,
                    a_cta_layout,
                    sA_for_tma_partition,
                    gA_for_tma_partition,
                )
                if cutlass.const_expr(self.split_gate):
                    (tGsA, tGgA_mkl) = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_a,
                        a_cta_crd,
                        a_cta_layout,
                        cute.group_modes(sGate, 0, 2),
                        cute.group_modes(gGate_mkl, 0, 2),
                    )
                tCsA = thr_mma.partition_A(sA)
                tCsB = thr_mma.partition_B(sB)
                tCrA = tiled_mma.make_fragment_A(tCsA)
                tCrB = tiled_mma.make_fragment_B(tCsB)
                acc_shape = tCgC.shape
                coordinates = tiled_mma.get_slice(tidx).partition_C(
                    cute.make_identity_tensor((64, 8))
                )
                accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                    total = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
                    total.fill(0)
                pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
                k_tile_cnt = cute.size(gA_mkl, mode=[2])
                prefetch_k_tile_cnt = cutlass.max(
                    cutlass.min(self.ab_stage, k_tile_cnt), 0
                )
                mainloop_producer_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer, self.ab_stage
                )
                for _prefetch_idx in cutlass.range(prefetch_k_tile_cnt, unroll=1):
                    tAgA_k = tAgA_mkl[None, mainloop_producer_state.count]
                    tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                    if warp_idx == 0:
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                            )
                            if cutlass.const_expr(self.split_gate):
                                cute.copy(
                                    tma_atom_a,
                                    tGgA_mkl[None, mainloop_producer_state.count],
                                    tGsA[None, mainloop_producer_state.index],
                                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                    mcast_mask=a_mcast_mask,
                                )
                            mainloop_pipeline.producer_commit(mainloop_producer_state)
                    cute.arch.sync_threads()
                    for bn in cutlass.range_constexpr(8):
                        if warp_idx == bn % 4:
                            for segment in cutlass.range_constexpr(
                                self.tile_shape_mnk[2] // 32
                            ):
                                bk = tidx % 32 + segment * 32
                                bv = self.b_dtype(0)
                                if cutlass.const_expr(bn < self.group_size):
                                    if (
                                        members[bn] >= 0
                                        and mainloop_producer_state.count
                                        * self.tile_shape_mnk[2]
                                        + bk
                                        < cute.size(raw_b, mode=[1])
                                    ):
                                        bv = raw_b[
                                            0,
                                            mainloop_producer_state.count
                                            * self.tile_shape_mnk[2]
                                            + bk,
                                            members[bn] // self.input_route_divisor,
                                        ]
                                sB[bn, bk, mainloop_producer_state.index] = bv
                    cute.arch.fence_proxy("async.shared", space="cta")
                    cute.arch.sync_threads()
                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                        if warp_idx == 0:
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                    mainloop_producer_state
                                ),
                                mcast_mask=a_mcast_mask,
                            )
                            if cutlass.const_expr(self.split_gate):
                                cute.copy(
                                    tma_atom_a,
                                    tGgA_mkl[None, mainloop_producer_state.count],
                                    tGsA[None, mainloop_producer_state.index],
                                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                    mcast_mask=a_mcast_mask,
                                )
                            mainloop_pipeline.producer_commit(mainloop_producer_state)
                    mainloop_producer_state.advance()
                k_pipe_mmas = 1
                mainloop_consumer_read_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                mainloop_consumer_release_state = pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Consumer, self.ab_stage
                )
                peek_ab_full_status = cutlass.Boolean(1)
                if mainloop_consumer_read_state.count < k_tile_cnt:
                    peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                        mainloop_consumer_read_state
                    )
                tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                num_k_blocks = cute.size(tCrA, mode=[2])
                for _k_tile in cutlass.range_constexpr(k_pipe_mmas):
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        tCrA_1phase = tCrA[k_block_coord]
                        tCrB_1phase = tCrB[k_block_coord]
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA_1phase,
                            tCrB_1phase,
                            accumulators,
                        )
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                            # Native FP8 WGMMA K=32. Promote at most two instructions
                            # at a time independently of the TMA staging tile width.
                            if (
                                k_block_idx + 1
                            ) % 2 == 0 or k_block_idx + 1 == num_k_blocks:
                                cute.nvgpu.warpgroup.commit_group()
                                cute.nvgpu.warpgroup.wait_group(0)
                                if cutlass.const_expr(self.block_scaled):
                                    promote_block_scaled(
                                        accumulators,
                                        total,
                                        coordinates,
                                        dequant,
                                        mid_quant,
                                        members,
                                        expert,
                                        cute.size(output, mode=[1]),
                                        pid_m,
                                        _k_tile,
                                        self.rows_per_tile,
                                        self.tile_k,
                                        self.group_size,
                                        self.split_gate,
                                        self.input_route_divisor,
                                    )
                                else:
                                    for element in cutlass.range_constexpr(
                                        cute.size(accumulators)
                                    ):
                                        total[element] = (
                                            total[element] + accumulators[element]
                                        )
                                tiled_mma.set(
                                    cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                                )
                                cute.nvgpu.warpgroup.fence()
                    if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                        cute.nvgpu.warpgroup.commit_group()
                    mainloop_consumer_read_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if mainloop_consumer_read_state.count < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )
                for _k_tile in cutlass.range(k_pipe_mmas, k_tile_cnt, 1, unroll=1):
                    mainloop_pipeline.consumer_wait(
                        mainloop_consumer_read_state, peek_ab_full_status
                    )
                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                        tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, False)
                    cute.nvgpu.warpgroup.fence()
                    for k_block_idx in cutlass.range(num_k_blocks, unroll_full=True):
                        k_block_coord = (
                            None,
                            None,
                            k_block_idx,
                            mainloop_consumer_read_state.index,
                        )
                        tCrA_1phase = tCrA[k_block_coord]
                        tCrB_1phase = tCrB[k_block_coord]
                        cute.gemm(
                            tiled_mma,
                            accumulators,
                            tCrA_1phase,
                            tCrB_1phase,
                            accumulators,
                        )
                        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                            tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                            # Native FP8 WGMMA K=32. Promote at most two instructions
                            # at a time independently of the TMA staging tile width.
                            if (
                                k_block_idx + 1
                            ) % 2 == 0 or k_block_idx + 1 == num_k_blocks:
                                cute.nvgpu.warpgroup.commit_group()
                                cute.nvgpu.warpgroup.wait_group(0)
                                if cutlass.const_expr(self.block_scaled):
                                    promote_block_scaled(
                                        accumulators,
                                        total,
                                        coordinates,
                                        dequant,
                                        mid_quant,
                                        members,
                                        expert,
                                        cute.size(output, mode=[1]),
                                        pid_m,
                                        _k_tile,
                                        self.rows_per_tile,
                                        self.tile_k,
                                        self.group_size,
                                        self.split_gate,
                                        self.input_route_divisor,
                                    )
                                else:
                                    for element in cutlass.range_constexpr(
                                        cute.size(accumulators)
                                    ):
                                        total[element] = (
                                            total[element] + accumulators[element]
                                        )
                                tiled_mma.set(
                                    cute.nvgpu.warpgroup.Field.ACCUMULATE, False
                                )
                                cute.nvgpu.warpgroup.fence()
                    if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                        cute.nvgpu.warpgroup.commit_group()
                    cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                    # WGMMA completion must precede producer reuse of this stage.
                    mainloop_pipeline.consumer_release(mainloop_consumer_release_state)
                    mainloop_consumer_read_state.advance()
                    mainloop_consumer_release_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if mainloop_consumer_read_state.count < k_tile_cnt:
                        peek_ab_full_status = mainloop_pipeline.consumer_try_wait(
                            mainloop_consumer_read_state
                        )
                    if mainloop_producer_state.count < k_tile_cnt:
                        tAgA_k = tAgA_mkl[None, mainloop_producer_state.count]
                        tAsA_pipe = tAsA[None, mainloop_producer_state.index]
                        if warp_idx == 0:
                            mainloop_pipeline.producer_acquire(mainloop_producer_state)
                            if cutlass.const_expr(self.a_dtype != cutlass.Float8E4M3FN):
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_k,
                                    tAsA_pipe,
                                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                    mcast_mask=a_mcast_mask,
                                )
                                if cutlass.const_expr(self.split_gate):
                                    cute.copy(
                                        tma_atom_a,
                                        tGgA_mkl[None, mainloop_producer_state.count],
                                        tGsA[None, mainloop_producer_state.index],
                                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                            mainloop_producer_state
                                        ),
                                        mcast_mask=a_mcast_mask,
                                    )
                                mainloop_pipeline.producer_commit(
                                    mainloop_producer_state
                                )
                        cute.arch.sync_threads()
                        for bn in cutlass.range_constexpr(self.group_size):
                            if warp_idx == bn % 4:
                                for segment in cutlass.range_constexpr(
                                    self.tile_shape_mnk[2] // 32
                                ):
                                    bk = tidx % 32 + segment * 32
                                    bv = self.b_dtype(0)
                                    if cutlass.const_expr(bn < self.group_size):
                                        if (
                                            members[bn] >= 0
                                            and mainloop_producer_state.count
                                            * self.tile_shape_mnk[2]
                                            + bk
                                            < cute.size(raw_b, mode=[1])
                                        ):
                                            bv = raw_b[
                                                0,
                                                mainloop_producer_state.count
                                                * self.tile_shape_mnk[2]
                                                + bk,
                                                members[bn] // self.input_route_divisor,
                                            ]
                                    sB[bn, bk, mainloop_producer_state.index] = bv
                        cute.arch.fence_proxy("async.shared", space="cta")
                        cute.arch.sync_threads()
                        if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                            if warp_idx == 0:
                                cute.copy(
                                    tma_atom_a,
                                    tAgA_k,
                                    tAsA_pipe,
                                    tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                        mainloop_producer_state
                                    ),
                                    mcast_mask=a_mcast_mask,
                                )
                                if cutlass.const_expr(self.split_gate):
                                    cute.copy(
                                        tma_atom_a,
                                        tGgA_mkl[None, mainloop_producer_state.count],
                                        tGsA[None, mainloop_producer_state.index],
                                        tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                            mainloop_producer_state
                                        ),
                                        mcast_mask=a_mcast_mask,
                                    )
                                mainloop_pipeline.producer_commit(
                                    mainloop_producer_state
                                )
                        mainloop_producer_state.advance()
                cute.nvgpu.warpgroup.wait_group(0)
                if cute.size(self.cluster_shape_mn) > 1:
                    cute.arch.cluster_arrive()
                    cute.arch.cluster_wait()
                else:
                    cute.arch.sync_threads()
                # Reuse weight shared storage only after all WGMMA consumers finish.
                thread_c = tiled_mma.get_slice(tidx).partition_C(sC[None, None, 0])
                for element in cutlass.range_constexpr(cute.size(accumulators)):
                    if cutlass.const_expr(self.a_dtype == cutlass.Float8E4M3FN):
                        thread_c[element] = total[element]
                    else:
                        thread_c[element] = accumulators[element]
                cute.arch.sync_threads()
                lane = cute.arch.lane_idx()
                for member in cutlass.range_constexpr(self.group_size):
                    if cutlass.const_expr(self.split_gate):
                        if warp_idx == member % 4 and members[member] >= 0:
                            value = self.epilogue(
                                sC[lane, member, 0],
                                sC[lane + 32, member, 0],
                                dequant,
                                mid_quant,
                                expert,
                            )
                            output[members[member], pid_m * 32 + lane] = value.to(
                                output.element_type
                            )
                    else:
                        if warp_idx // 2 == member % 2 and members[member] >= 0:
                            row = (warp_idx % 2) * 32 + lane
                            value = self.epilogue(
                                sC[row, member, 0],
                                cutlass.Float32(0),
                                dequant,
                                mid_quant,
                                expert,
                            )
                            output[members[member], pid_m * 64 + row] = value.to(
                                output.element_type
                            )
        return

    @cute.jit
    def epilogue(self, value, gate, dequant, mid_quant, expert):
        if cutlass.const_expr(
            self.a_dtype == cutlass.Float8E4M3FN and not self.block_scaled
        ):
            value = _mul(value, dequant[expert])
            if cutlass.const_expr(self.fc1):
                gate = _mul(gate, dequant[expert])
                value = value.to(cutlass.BFloat16).to(cutlass.Float32)
                gate = gate.to(cutlass.BFloat16).to(cutlass.Float32)
        if cutlass.const_expr(self.fc1):
            value = apply_activation(
                value, gate, self.activation, False, self.parameters
            )
            if cutlass.const_expr(
                self.a_dtype == cutlass.Float8E4M3FN and not self.block_scaled
            ):
                value = (
                    cvt_f32_to_e4m3(_mul(value, mid_quant[0]))
                    .to(cutlass.Uint8)
                    .bitcast(cutlass.Float8E4M3FN)
                )
        return value

    @staticmethod
    def _make_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        epi_tile: tuple[int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
        c_dtype: type[cutlass.Numeric],
        c_layout: utils.LayoutEnum,
        epi_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout, cute.ComposedLayout]:
        a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            a_layout, tile_shape_mnk, a_dtype, ab_stage
        )
        b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            b_layout, tile_shape_mnk, b_dtype, ab_stage
        )
        epi_smem_layout_staged = sm90_utils.make_smem_layout_epi(
            c_dtype, c_layout, epi_tile, epi_stage
        )
        return (a_smem_layout_staged, b_smem_layout_staged, epi_smem_layout_staged)

    @staticmethod
    def _make_tma_atoms_and_tensors(
        tensor: cute.Tensor,
        smem_layout_staged: cute.ComposedLayout,
        smem_tile: tuple[int, int],
        mcast_dim: int,
    ) -> tuple[cute.CopyAtom, cute.Tensor]:
        op = (
            cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
            if mcast_dim == 1
            else cute.nvgpu.cpasync.CopyBulkTensorTileG2SMulticastOp()
        )
        smem_layout = cute.slice_(smem_layout_staged, (None, None, 0))
        (tma_atom, tma_tensor) = cute.nvgpu.cpasync.make_tiled_tma_atom(
            op, tensor, smem_layout, smem_tile, num_multicast=mcast_dim
        )
        return (tma_atom, tma_tensor)
