# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import math
from typing import Tuple, Optional
import torch
import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute

import cutlass.cute.nvgpu.cpasync as cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from .mla_config import MLAConfig
from .mla_warp_schedule import MLAWarpSchedule, MLA_DECODE_SCHEDULE
from .pipeline_topology import PipelineTopology, make_mla_topology
from .mainloop_spec import MLAMainloopSpec, make_mla_mainloop_spec
from .collective_builder import build_mla_launch_params
from .roles.mla_loader import MLALoaderRole
from .roles.mla_mma import MLAMmaRole
from .roles.mla_compute import MLAComputeRole
from .scheduler.mla_persistent import (
    MLAStaticTileScheduler,
    MLAStaticTileSchedulerParams,
    create_mla_static_tile_scheduler_params,
)

import warnings

warnings.filterwarnings("ignore", category=UserWarning)


LOG2_E = 1.4426950408889634074


class BlackwellMultiLatentAttentionForward:
    def __init__(
        self,
        config: MLAConfig,
        warp_schedule: Optional[MLAWarpSchedule] = None,
    ):
        """Initializes a Blackwell Multi-Latent Attention (MLA) decode kernel.

        :param config: MLA configuration (dims, dtypes, tile shapes, mode).
        :param warp_schedule: Warp role assignment and register budgets. Defaults to MLA_DECODE_SCHEDULE.
        """

        self.config = config
        self.schedule = warp_schedule if warp_schedule is not None else MLA_DECODE_SCHEDULE
        self.mainloop = make_mla_mainloop_spec(config, self.schedule)

    @cute.jit
    def __call__(
        self,
        q_latent: cute.Tensor,
        q_rope: cute.Tensor,
        c_latent: cute.Tensor,
        c_rope: cute.Tensor,
        page_table: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor,
        workspace: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: Optional[cute.Tensor],
        block_split_kvs: Optional[cute.Tensor],
        softmax_scale: cutlass.Float32,
        output_scale: cutlass.Float32,
        stream: cuda.CUstream,
    ):
        """Execute the Multi-Head Latent Attention operation on the provided tensors.

        The method handles:
        1. Initialization of workspace for temporary split KV buffers
        2. Validation of tensor data types
        3. Initialization of hardware-specific parameters and memory layouts
        4. Configuration of TMA (Tensor Memory Access) operations
        5. Grid and work scheduling computation
        6. Kernel launch(split KV kernel and reduction kernel) with appropriate parameters

        :param q_latent: The query tensor with shape [num_head, latent_dim, batch_size]
        :type q_latent: cute.Tensor
        :param q_rope: The query RoPE tensor with shape [num_head, rope_dim, batch_size]
        :type q_rope: cute.Tensor
        :param c_latent: The key tensor with shape [seq_len, latent_dim, batch_size]
        :type c_latent: cute.Tensor
        :param c_rope: The key RoPE tensor with shape [seq_len, rope_dim, batch_size]
        :type c_rope: cute.Tensor
        :param page_table: The page table tensor with shape [page_count, batch_size]
        :type page_table: cute.Tensor
        :param o: The output tensor with shape [num_head, latent_dim, batch_size]
        :type o: cute.Tensor
        :param lse: The LSE tensor with shape [num_head, batch_size]
        :type lse: cute.Tensor
        :param workspace: The workspace tensor with 1-d shape prepared for acc_o and acc_lse
        :type workspace: cute.Tensor
        :param split_kv: The scalar factor for split KV
        :type split_kv: cutlass.Int32
        :param cache_seqs: The cache sequences tensor with shape [batch_size]
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: The block split KV tensor with shape [batch_size]
        :type block_split_kvs: cute.Tensor
        :param softmax_scale: The scale factor for softmax
        :type softmax_scale: cutlass.Float32
        :param output_scale: The scale factor for the output
        :type output_scale: cutlass.Float32
        :param stream: The CUDA stream to execute the kernel on
        :type stream: cuda.CUstream

        :raises TypeError: If tensor data types don't match or aren't supported
        """

        # setup static attributes before smem/grid/tma computation
        self.q_dtype = q_latent.element_type
        self.k_dtype = c_latent.element_type
        self.v_dtype = c_latent.element_type
        self.o_dtype = o.element_type

        # check type consistency
        if cutlass.const_expr(
            self.q_dtype != self.k_dtype or self.q_dtype != self.v_dtype
        ):
            raise TypeError(
                f"Type mismatch: {self.q_dtype} != {self.k_dtype} or {self.q_dtype} != {self.v_dtype}"
            )
        # check leading dimensions of input/output
        if cutlass.const_expr(q_latent.stride[1] != 1 or q_rope.stride[1] != 1):
            raise ValueError("q_latent and q_rope must have leading dimension 1")
        if cutlass.const_expr(c_latent.stride[1] != 1 or c_rope.stride[1] != 1):
            raise ValueError("c_latent and c_rope must have leading dimension 1")
        if cutlass.const_expr(o.stride[1] != 1):
            raise ValueError("o must have leading dimension 1")
        if cutlass.const_expr(lse.stride[0] != 1):
            raise ValueError("lse must have leading dimension 0")

        acc_o, acc_lse = None, None
        if cutlass.const_expr(workspace is not None):
            H, D, B = q_latent.shape[0], q_latent.shape[1], q_latent.shape[2]
            align = 128 // self.q_dtype.width
            acc_o_layout = cute.make_layout(
                (H, split_kv, D, B),
                stride=(
                    cute.assume(split_kv * D, align),
                    cute.assume(D, align),
                    1,
                    cute.assume(H * split_kv * D, align),
                ),
            )
            acc_o_iter = cute.recast_ptr(workspace.iterator, dtype=self.config.acc_dtype)
            acc_o = cute.make_tensor(acc_o_iter, acc_o_layout)
            acc_lse_layout = cute.make_layout(
                (H, split_kv, B), stride=(split_kv, 1, H * split_kv)
            )
            acc_lse_iter = cute.recast_ptr(
                workspace.iterator + cute.cosize(acc_o_layout) * self.config.acc_dtype.width // 8,
                dtype=self.config.acc_dtype,
            )
            acc_lse = cute.make_tensor(acc_lse_iter, acc_lse_layout)

        c_latent_tranpose_layout = cute.select(c_latent.layout, mode=[1, 0, 2])
        c_latent_transpose = cute.make_tensor(
            c_latent.iterator, c_latent_tranpose_layout
        )

        self.mainloop.resolve(self.k_dtype.width)

        self.loader_role = MLALoaderRole(self.config)
        self.mma_role = MLAMmaRole(self.config, self.mainloop)
        self.exchange_sync_bar = pipeline.NamedBarrier(
            barrier_id=self.schedule.exchange_sync_bar_id,
            num_threads=self.schedule.exchange_sync_num_threads,
        )
        self.compute_role = MLAComputeRole(
            self.config, self.mainloop, self.schedule, self.exchange_sync_bar
        )
        self.compute_role.q_dtype = self.q_dtype
        self.compute_role.o_dtype = self.o_dtype
        self.tmem_ptr_sync_bar = pipeline.NamedBarrier(
            barrier_id=self.schedule.tmem_ptr_sync_bar_id,
            num_threads=self.schedule.tmem_ptr_sync_num_threads,
        )

        lp = build_mla_launch_params(
            self.mainloop, self.schedule,
            q_latent, q_rope, c_latent, c_rope, c_latent_transpose,
            self.q_dtype, self.k_dtype, self.v_dtype,
        )
        self.tma_copy_q_bytes = lp.tma_copy_q_bytes
        self.tma_copy_kc_bytes = lp.tma_copy_kc_bytes

        tile_sched_params, grid = self._compute_grid(
            o, split_kv,
            self.config.cluster_shape_mnk,
            self.config.max_active_clusters,
            self.config.is_persistent,
        )

        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            lp.qk_tiled_mma,
            lp.pv_tiled_mma,
            lp.tma_atom_q_latent,
            lp.tma_tensor_q_latent,
            lp.tma_atom_q_rope,
            lp.tma_tensor_q_rope,
            lp.tma_atom_c_latent,
            lp.tma_tensor_c_latent,
            lp.tma_atom_c_rope,
            lp.tma_tensor_c_rope,
            lp.tma_atom_c_latent_transpose,
            lp.tma_tensor_c_latent_transpose,
            page_table,
            o,
            lse,
            acc_o,
            acc_lse,
            split_kv,
            cache_seqs,
            block_split_kvs,
            softmax_scale_log2,
            output_scale,
            lp.q_smem_layout_staged,
            lp.kc_smem_layout_staged,
            lp.p_smem_layout_staged,
            lp.vc_smem_layout_staged,
            lp.cta_layout_vmnk,
            tile_sched_params,
            lp.SharedStorage,
        ).launch(
            grid=grid,
            block=[self.schedule.threads_per_cta(self.config.is_cpasync), 1, 1],
            cluster=self.config.cluster_shape_mnk,
            smem=lp.SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )
        if cutlass.const_expr(acc_o is not None):
            self.reduction_kernel(
                o,
                lse,
                acc_o,
                acc_lse,
                split_kv,
                cache_seqs,
                block_split_kvs,
            ).launch(
                grid=(q_latent.shape[0], 1, q_latent.shape[2]),
                block=[self.schedule.threads_per_warp * self.schedule.num_compute_warps, 1, 1],
                smem=split_kv * self.config.acc_dtype.width // 8,
                stream=stream,
                min_blocks_per_mp=1,
            )

    @cute.jit
    def _create_pipelines(self, storage, cta_layout_vmnk):
        """Create all inter-warp pipelines from the topology and storage barriers."""
        barrier_ptrs = {
            "load_q": storage.load_q_mbar_ptr.data_ptr(),
            "load_kv": storage.load_kv_mbar_ptr.data_ptr(),
            "mma_s": storage.mma_s_mbar_ptr.data_ptr(),
            "p_mma": storage.p_mma_mbar_ptr.data_ptr(),
            "mma_o": storage.mma_o_mbar_ptr.data_ptr(),
        }
        tx_counts = {"q": self.tma_copy_q_bytes, "kv": self.tma_copy_kc_bytes}
        return self.mainloop.pipeline_topology.create_pipelines_native(
            barrier_ptrs, tx_counts, self.schedule.threads_per_warp, cta_layout_vmnk,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        tma_atom_q_latent: cute.CopyAtom,
        mQL: cute.Tensor,
        tma_atom_q_rope: cute.CopyAtom,
        mQR: cute.Tensor,
        tma_atom_c_latent: cute.CopyAtom,
        mCL: cute.Tensor,
        tma_atom_c_rope: cute.CopyAtom,
        mKR: cute.Tensor,
        tma_atom_c_latent_transpose: cute.CopyAtom,
        mCLT: cute.Tensor,
        mPT: cute.Tensor,
        mO: Optional[cute.Tensor],
        mLSE: Optional[cute.Tensor],
        mAccO: Optional[cute.Tensor],
        mAccLSE: Optional[cute.Tensor],
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        q_smem_layout_staged: cute.ComposedLayout,
        kc_smem_layout_staged: cute.ComposedLayout,
        p_smem_layout_staged: cute.ComposedLayout,
        vc_smem_layout_staged: cute.ComposedLayout,
        cta_layout_vmnk: cute.Layout,
        tile_sched_params: MLAStaticTileSchedulerParams,
        SharedStorage: cutlass.Constexpr,
    ):
        """MLA split-KV device kernel: warp-specialized decode with pipelined TMA loads."""

        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma_qk.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0

        # Coords inside cluster
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )

        # Prefetch tma descriptor
        if warp_idx == 0:
            cpasync.prefetch_descriptor(tma_atom_q_latent)
            cpasync.prefetch_descriptor(tma_atom_q_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent)
            cpasync.prefetch_descriptor(tma_atom_c_rope)
            cpasync.prefetch_descriptor(tma_atom_c_latent_transpose)

        # Alloc
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr
        tmem_holding_buf = storage.tmem_holding_buf

        # Tensor memory dealloc barrier init
        if warp_idx == self.schedule.load_tma_warp_id:
            num_tmem_dealloc_threads = self.schedule.threads_per_warp * self.schedule.num_compute_warps
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(tmem_dealloc_mbar_ptr, num_tmem_dealloc_threads)
        cute.arch.mbarrier_init_fence()

        pipelines = self._create_pipelines(storage, cta_layout_vmnk)
        load_q_pipeline = pipelines["load_q"]
        load_kv_pipeline = pipelines["load_kv"]
        mma_s_pipeline = pipelines["mma_s"]
        p_mma_pipeline = pipelines["p_mma"]
        mma_o_pipeline = pipelines["mma_o"]

        # Cluster arrive after barrier init
        if cutlass.const_expr(cute.size(self.config.cluster_shape_mnk) > 1):
            cute.arch.cluster_arrive_relaxed()

        sQ = storage.smem_q.get_tensor(q_smem_layout_staged.outer, swizzle=q_smem_layout_staged.inner)
        sKC = storage.smem_kc.get_tensor(kc_smem_layout_staged.outer, swizzle=kc_smem_layout_staged.inner)
        sVC = cute.make_tensor(cute.recast_ptr(sKC.iterator, vc_smem_layout_staged.inner), vc_smem_layout_staged.outer)
        sP = storage.smem_p.get_tensor(p_smem_layout_staged.outer, swizzle=p_smem_layout_staged.inner)
        smem_exchange = storage.smem_exchange.get_tensor(
            cute.make_layout(self.schedule.num_compute_warps * self.schedule.threads_per_warp)
        )

        #
        # Cluster wait before tensor memory alloc
        #
        if cutlass.const_expr(cute.size(self.config.cluster_shape_mnk) > 1):
            cute.arch.cluster_wait()
        else:
            cute.arch.barrier()

        # ///////////////////////////////////////////////////////////////////////////////
        #  Load warps
        # ///////////////////////////////////////////////////////////////////////////////
        if cutlass.const_expr(self.config.is_cpasync):
            # TODO: add cp async load variant.
            pass
        else:
            if warp_idx == self.schedule.load_tma_warp_id:
                self.loader_role.run(
                    tiled_mma_qk, tiled_mma_pv,
                    tma_atom_q_latent, mQL,
                    tma_atom_q_rope, mQR,
                    tma_atom_c_latent, mCL,
                    tma_atom_c_rope, mKR,
                    tma_atom_c_latent_transpose, mCLT,
                    sQ, sKC, sVC,
                    load_q_pipeline, load_kv_pipeline,
                    self.mainloop.load_q_stages, self.mainloop.load_kv_stages,
                    mPT, tile_sched_params,
                    split_kv, cache_seqs, block_split_kvs,
                )

        # ///////////////////////////////////////////////////////////////////////////////
        #  MMA warp
        # ///////////////////////////////////////////////////////////////////////////////
        if warp_idx == self.schedule.mma_warp_id:
            self.mma_role.run(
                tiled_mma_qk, tiled_mma_pv,
                sQ, sKC, sP, sVC,
                load_q_pipeline, load_kv_pipeline,
                mma_s_pipeline, p_mma_pipeline, mma_o_pipeline,
                tmem_holding_buf, tmem_dealloc_mbar_ptr,
                self.tmem_ptr_sync_bar,
                mCL.shape[1], is_leader_cta,
                tile_sched_params,
                split_kv, cache_seqs, block_split_kvs,
            )

        # ///////////////////////////////////////////////////////////////////////////////
        #  Compute warp
        # ///////////////////////////////////////////////////////////////////////////////
        if (
            warp_idx >= self.schedule.compute_warp_ids[0]
            and warp_idx <= self.schedule.compute_warp_ids[-1]
        ):
            self.compute_role.run(
                tiled_mma_qk, tiled_mma_pv,
                sP, smem_exchange,
                mma_s_pipeline, p_mma_pipeline, mma_o_pipeline,
                tmem_holding_buf, tmem_dealloc_mbar_ptr,
                self.tmem_ptr_sync_bar,
                mO, mLSE, mAccO, mAccLSE,
                mCL.shape[1], cache_seqs,
                split_kv, block_split_kvs,
                softmax_scale_log2, output_scale,
                tidx, cta_rank_in_cluster,
                tile_sched_params,
            )

        return

    @cute.kernel
    def reduction_kernel(
        self,
        mO: cute.Tensor,
        mLSE: cute.Tensor,
        mAccO: cute.Tensor,
        mAccLSE: cute.Tensor,
        split_kv: cutlass.Int32,
        cache_seqs: cute.Tensor,
        block_split_kvs: cute.Tensor,
    ):
        """The reduction kernel for Multi-Head Latent Attention (MLA) that combines intermediate results
        from multiple split_kv blocks into final outputs.

        :param mO: Output tensor for storing final results
        :type mO: cute.Tensor
        :param mLSE: Log-sum-exp tensor for storing final LSE values
        :type mLSE: cute.Tensor
        :param mAccO: Accumulated output tensor from split_kv blocks
        :type mAccO: cute.Tensor
        :param mAccLSE: Accumulated LSE tensor from split_kv blocks
        :type mAccLSE: cute.Tensor
        :param split_kv: Number of split_kv blocks
        :type split_kv: cutlass.Int32
        :param cache_seqs: Cache sequence lengths tensor
        :type cache_seqs: cute.Tensor
        :param block_split_kvs: Per-block split_kv values tensor (for variable split_kv)
        :type block_split_kvs: cute.Tensor
        """
        # avoid register indexing on array.
        MAX_SPLITS = 256
        bidx, _, bidz = cute.arch.block_idx()
        tidx, _, _ = cute.arch.thread_idx()
        blk_coord = (bidx, 0, bidz)
        local_split_kv = (
            block_split_kvs[blk_coord[2]] if self.config.is_var_split_kv else split_kv
        )
        k_tile_total = cute.ceil_div(cache_seqs[blk_coord[2]], self.config.mma_qk_tiler[1])
        k_tile_per_cta = cute.ceil_div(k_tile_total, local_split_kv)
        local_split_kv = cute.ceil_div(k_tile_total, k_tile_per_cta)

        # Alloc shared memory
        smem = utils.SmemAllocator()
        storage = smem.allocate(MAX_SPLITS * self.config.acc_dtype.width // 8, 16)
        lse_scale_ptr = cute.recast_ptr(storage, dtype=self.config.acc_dtype)
        smem_lse_scale = cute.make_tensor(lse_scale_ptr, cute.make_layout(MAX_SPLITS))

        gLSE = mAccLSE[blk_coord[0], None, blk_coord[2]]
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp_idx == 0:
            # calculate the global lse and exp ^ (local_lse - global_lse)
            lse_per_thread = cute.ceil_div(MAX_SPLITS, self.schedule.threads_per_warp)

            local_lse = cute.make_fragment(
                cute.make_layout(lse_per_thread), self.config.lse_dtype
            )
            lse_max = -self.config.lse_dtype.inf
            # find the max lse
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                local_lse[i] = (
                    gLSE[split_kv_idx]
                    if cute.elem_less(split_kv_idx, local_split_kv)
                    else -self.config.lse_dtype.inf
                )
                # reduce the local lse
                lse_max = cute.arch.fmax(lse_max, local_lse[i])
            lse_max = cute.arch.warp_reduction_max(lse_max)
            lse_max = lse_max if lse_max != -self.config.lse_dtype.inf else 0.0
            # calculate sum_lse
            sum_lse = 0.0
            for i in range(lse_per_thread):
                sum_lse += cute.arch.exp2(local_lse[i] - lse_max)
            sum_lse = cute.arch.warp_reduction_sum(sum_lse)
            # calculate the global_lse
            global_lse = (
                lse_max + cute.math.log2(sum_lse)
                if sum_lse != self.config.lse_dtype(0.0) or sum_lse != sum_lse
                else self.config.lse_dtype.inf
            )
            if tidx == 0:
                mLSE[blk_coord[0], blk_coord[2]] = global_lse
            # store the scale to shared memory
            for i in range(lse_per_thread):
                split_kv_idx = tidx + i * self.schedule.threads_per_warp
                if cute.elem_less(split_kv_idx, local_split_kv):
                    smem_lse_scale[split_kv_idx] = cute.arch.exp2(
                        local_lse[i] - global_lse
                    )

        cute.arch.barrier()

        elements_per_thread = cute.ceil_div(
            self.config.latent_dim, self.schedule.threads_per_warp * self.schedule.num_compute_warps
        )
        gAccO = mAccO[blk_coord[0], None, None, blk_coord[2]]
        rAccO = cute.make_fragment(
            cute.make_layout(elements_per_thread), self.config.acc_dtype
        )
        rAccO.fill(0.0)
        for i in range(local_split_kv):
            for j in range(elements_per_thread):
                element_idx = tidx + j * self.schedule.threads_per_warp * self.schedule.num_compute_warps
                rAccO[j] += gAccO[i, element_idx] * smem_lse_scale[i]
        for j in range(elements_per_thread):
            element_idx = tidx + j * self.schedule.threads_per_warp * self.schedule.num_compute_warps
            mO[blk_coord[0], element_idx, blk_coord[2]] = rAccO[j].to(self.o_dtype)
        return

    @staticmethod
    def _compute_grid(
        o: cute.Tensor,
        split_kv: cutlass.Int32,
        cluster_shape_mnk: Tuple[int, int, int],
        max_active_clusters: int,
        is_persistent: bool,
    ) -> Tuple[MLAStaticTileSchedulerParams, Tuple[int, int, int]]:
        """Compute grid shape for the output tensor C.

        :param c: The output tensor C
        :type c: cute.Tensor
        :param cta_tile_shape_mnk: The shape (M, N, K) of the CTA tile.
        :type cta_tile_shape_mnk: tuple[int, int, int]
        :param cluster_shape_mn: Shape of each cluster in M, N dimensions.
        :type cluster_shape_mn: tuple[int, int]

        :return: Tile scheduler parameters and grid shape.
        :rtype: tuple[MLAStaticTileSchedulerParams, tuple[int, int, int]]
        """
        o_shape = o.shape
        tile_sched_params = create_mla_static_tile_scheduler_params(
            is_persistent,
            cute.size(o_shape[2]),
            cluster_shape_mnk,
            split_kv,
        )
        grid = MLAStaticTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )

        return tile_sched_params, grid

