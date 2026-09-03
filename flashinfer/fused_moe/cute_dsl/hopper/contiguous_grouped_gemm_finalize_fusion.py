# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel structure adapted from NVIDIA CUTLASS CuTe-DSL example
# examples/python/CuTeDSL/hopper/dense_gemm_persistent.py (BSD-3-Clause).

"""SM90 contiguous grouped GEMM with fused MoE finalize (MoE GEMM2).

Computes, for each valid permuted row ``r`` of expert ``e``:

    out[token(r), :] += router_scale(r) * (A[r] @ B[e].T)     (fused mode)
    out[expanded(r), :] = A[r] @ B[e].T                       (deterministic)

- A is the (contiguous, permuted) GEMM1 output — plain TMA loads on a single
  TMA pipeline for A+B, with the expert index as B's TMA L-coordinate (no
  tensormap updates).
- A **meta warp** (producer warpgroup, warp 1) prefetches per-row
  ``(output_row, scale)`` into SMEM one tile ahead:
  ``scale = token_final_scales[token, k_slot]`` (fused) or 1.0
  (deterministic); rows use ``permuted_idx_to_expanded_idx`` whose padding
  entries are garbage and handled branchlessly.
- Epilogue: per-element ``acc * sMetaScale[row]`` (row derived from an
  identity-tensor partition), converted to the output dtype, staged in a
  row-padded linear SMEM tile, then scattered **one row per thread** with
  ``cp.reduce.async.bulk...add`` (fused) or ``cp.async.bulk`` (deterministic).
  In fused mode the destination must be zero-initialized by the host.

The epilogue stages the whole (tile_m, tile_n) tile in one ``sC`` buffer
(no TMA store, so no subtile ring is needed), filling it per 32-column
subtile — one whole-tile copy is not order-safe because retile and
partition_D flatten differently; ``sC`` rows are padded to 16 B alignment
as `cp.{reduce.,}async.bulk` requires.
"""

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import utils as hopper_utils
from .utils import blk_copy, blk_reduce_bf16, blk_reduce_fp16, blk_reduce_fp32


class Sm90ContiguousGroupedGemmFinalizeFusionKernel:
    """Persistent warp-specialized MoE GEMM2 with fused finalize.

    :param acc_dtype: Accumulator dtype (Float32 for bf16 inputs).
    :param tile_shape_mn: CTA tile (M, N). M in {64, 128}; N % 8 == 0,
        N <= 256. tile M must equal ``moe_sort``'s ``tile_tokens_dim``.
    :param topk: MoE top-k.
    :param use_fused_finalize: True -> atomic scatter-reduce into the
        zero-initialized ``[num_tokens, N_total]`` output; False -> plain
        scatter into ``[num_tokens * topk, N_total]`` (deterministic order,
        host applies scales via ``moe_unpermute``).
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        topk: int,
        use_fused_finalize: bool = True,
        tile_k: int = 64,
        cluster_shape_mn: tuple[int, int] = (1, 1),
        swizzle_size: int = 1,
        raster_along_m: bool = False,
        enable_pdl: bool = True,
    ):
        """``tile_k``: CTA K-tile in elements — 64 (default) or 32. A 32-wide
        tile supports reduction dimensions not divisible by 64 and halves the
        SMEM atom to SW64, increasing the stage count allowed by capacity.

        ``cluster_shape_mn``: (1, 1) or (1, 2). (1, 2) pairs two N-tile CTAs
        of the same M-tile and TMA-multicasts A between them — halves the
        intermediate re-read traffic (A is re-read once per N-tile otherwise;
        the ncu roofline showed both GEMMs L2/DRAM-bound). The two CTAs read
        different B N-slices, so B is never multicast; the meta warp runs in
        both CTAs on the same rows (duplicated but harmless)."""
        if tile_k not in (32, 64):
            raise ValueError("tile_k must be 32 or 64 for 16-bit inputs")
        if cluster_shape_mn not in ((1, 1), (1, 2)):
            raise ValueError("cluster_shape_mn must be (1, 1) or (1, 2)")
        self.tile_k = tile_k
        self.acc_dtype = acc_dtype
        self.topk = topk
        self.use_fused_finalize = use_fused_finalize
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.enable_pdl = enable_pdl
        self.tile_shape_mnk = (*tile_shape_mn, 1)
        # 2-warpgroup configs use the 40/232 register budgets
        self.atom_layout_mnk = (
            (2, 1, 1)
            if self.tile_shape_mnk[0] > 64 and self.tile_shape_mnk[1] > 128
            else (1, 1, 1)
        )
        self.tiled_mma = None

        self.occupancy = 1
        self.num_dma_warp_groups = 1
        self.num_mma_warp_groups = math.prod(self.atom_layout_mnk)
        self.num_warps_per_warp_group = 4
        self.num_threads_per_warp_group = self.num_warps_per_warp_group * 32
        self.threads_per_cta = (
            self.num_dma_warp_groups + self.num_mma_warp_groups
        ) * self.num_threads_per_warp_group
        self.load_warp_id = 0
        self.meta_warp_id = 1
        self.load_register_requirement = 40
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.meta_stage = 2

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )
        # 16 B row alignment for cp.{reduce.,}async.bulk.
        self.c_row_pad = None  # set in _setup_attributes (dtype dependent)

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] % 8 != 0 or not 8 <= self.tile_shape_mnk[1] <= 256:
            raise ValueError("CTA tile shape N must be a multiple of 8, <= 256")

        self.tiled_mma = sm90_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_layout.sm90_mma_major_mode(),
            self.b_layout.sm90_mma_major_mode(),
            self.acc_dtype,
            self.atom_layout_mnk,
            tiler_mn=(64, self.tile_shape_mnk[1]),
        )
        mma_inst_shape_k = cute.size(self.tiled_mma.shape_mnk, mode=[2])
        if self.tile_k % mma_inst_shape_k != 0:
            raise ValueError(
                f"tile_k={self.tile_k} not a multiple of the MMA instruction "
                f"K ({mma_inst_shape_k})"
            )
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            self.tile_k,
        )

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        # Always pad: keeps every row start 16 B aligned regardless of tile_n.
        self.c_row_pad = 16 // (self.c_dtype.width // 8)

        self.ab_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.c_dtype,
            self.c_row_pad,
            self.meta_stage,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
        ) = self._make_ab_smem_layouts(
            self.tile_shape_mnk,
            self.a_dtype,
            self.a_layout,
            self.b_dtype,
            self.b_layout,
            self.ab_stage,
        )

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        out: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        token_final_scales: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Compile-time entry.

        :param a: Permuted GEMM1 output (permuted_m, K, 1), K-major.
        :param b: Expert weights (N, K, E), K-major.
        :param out: (num_tokens, N, 1) zero-initialized (fused) or
            (num_tokens * topk, N, 1) (deterministic), N-major.
        :param token_final_scales: (num_tokens, topk) Float32 router scales.
        """
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = out.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)

        if cutlass.const_expr(self.a_dtype.width != 16):
            raise TypeError("this kernel supports 16-bit A/B (bf16/fp16) only")
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")
        if cutlass.const_expr(
            self.c_dtype not in (cutlass.BFloat16, cutlass.Float16, cutlass.Float32)
        ):
            raise TypeError("finalize scatter supports bf16/fp16/f32 outputs")

        self._setup_attributes()

        # A is multicast across the cluster's N dimension (same M-tile, two
        # N-tiles share the same A rows); B differs per CTA, never multicast.
        tma_atom_a, tma_tensor_a = hopper_utils.make_tma_atoms_and_tensors(
            a,
            self.a_smem_layout_staged,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[1],
        )
        tma_atom_b, tma_tensor_b = hopper_utils.make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            1,
        )

        tile_sched_params, grid = self._compute_grid(
            a,
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1]),
            b,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            max_active_clusters,
        )

        c_row_stride = self.tile_shape_mnk[1] + self.c_row_pad

        @cute.struct
        class SharedStorage:
            mainloop_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.ab_stage * 2
            ]
            meta_pipeline_array_ptr: cute.struct.MemRange[
                cutlass.Int64, self.meta_stage * 2
            ]
            sMetaTokenIdx: cute.struct.MemRange[
                cutlass.Int32, self.tile_shape_mnk[0] * self.meta_stage
            ]
            sMetaScale: cute.struct.MemRange[
                cutlass.Float32, self.tile_shape_mnk[0] * self.meta_stage
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
            sC: cute.struct.Align[
                cute.struct.MemRange[
                    self.c_dtype, self.tile_shape_mnk[0] * c_row_stride
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            tma_atom_a,
            tma_tensor_a,
            tma_atom_b,
            tma_tensor_b,
            out,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            token_final_scales,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=self.enable_pdl,
        )
        return

    @cute.kernel
    def kernel(
        self,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        mOut: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        token_final_scales: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_a)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        a_mcast_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=1
        )
        a_mcast_mask = a_mcast_mask if self.cluster_shape_mn[1] > 1 else 0

        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_copy_bytes = cute.size_in_bytes(
            self.a_dtype, a_smem_layout
        ) + cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        mainloop_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        # One arrive per consumer warp, scaled by the multicast size (A mcast
        # across cluster N + non-mcast B).
        mcast_size = self.cluster_shape_mn[1] + 1 - 1
        consumer_arrive_cnt = (
            mcast_size * self.num_mma_warp_groups * self.num_warps_per_warp_group
        )
        mainloop_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, consumer_arrive_cnt
        )
        mainloop_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.mainloop_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=mainloop_pipeline_producer_group,
            consumer_group=mainloop_pipeline_consumer_group,
            tx_count=tma_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        # Meta pipeline: warp 1 (32 threads) produces per-row (out_idx, scale);
        # every MMA thread consumes.
        meta_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        meta_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads
        )
        meta_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.meta_pipeline_array_ptr.data_ptr(),
            num_stages=self.meta_stage,
            producer_group=meta_producer_group,
            consumer_group=meta_consumer_group,
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        # Row-padded linear sC (no swizzle): rows start 16 B aligned for
        # cp.{reduce.,}async.bulk.
        c_row_stride = self.tile_shape_mnk[1] + self.c_row_pad
        c_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1], 1),
            stride=(c_row_stride, 1, self.tile_shape_mnk[0] * c_row_stride),
        )
        meta_smem_layout = cute.make_layout(
            (self.tile_shape_mnk[0], self.meta_stage),
            stride=(1, self.tile_shape_mnk[0]),
        )
        sC = storage.sC.get_tensor(c_smem_layout)
        sMetaTokenIdx = storage.sMetaTokenIdx.get_tensor(meta_smem_layout)
        sMetaScale = storage.sMetaScale.get_tensor(meta_smem_layout)

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
        gToken_ml = cute.local_tile(
            token_id_mapping,
            (self.tile_shape_mnk[0],),
            (None,),
        )

        a_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (0, None, 0)).shape)
        a_cta_crd = cluster_coord_mnk[1]
        tAsA, tAgA = cute.nvgpu.cpasync.tma_partition(
            tma_atom_a,
            a_cta_crd,
            a_cta_layout,
            cute.group_modes(sA, 0, 2),
            cute.group_modes(gA_mkl, 0, 2),
        )
        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            0,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )

        warp_group_idx = cute.arch.make_warp_uniform(
            tidx // self.num_threads_per_warp_group
        )
        mma_warp_group_thread_layout = cute.make_layout(
            self.num_mma_warp_groups, stride=self.num_threads_per_warp_group
        )
        thr_mma = tiled_mma.get_slice(
            mma_warp_group_thread_layout(warp_group_idx - self.num_dma_warp_groups)
        )

        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)

        acc_tile_coords = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1], 1)
        )
        tCgAcc = thr_mma.partition_C(acc_tile_coords)
        accumulators = cute.make_rmem_tensor(tCgAcc.shape[:3], self.acc_dtype)

        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # PDL: only GEMM1's output (A, the intermediate) needs the grid
        # dependency wait. Everything older is transitively visible before
        # this grid can even start executing — it launches on GEMM1's
        # trigger, which fired after GEMM1's own wait cleared on moe_sort —
        # so the moe_sort index maps, router scales, and static weights are
        # safe to read before it. The wait therefore lives in the load warp, right
        # before the A TMA loop; the meta warp and the consumer warpgroups
        # touch GEMM1 data solely through the mainloop pipeline the load
        # warp gates.
        num_valid_tiles = num_non_exiting_tiles[0]

        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

        # ------------------------------------------------------------------
        # TMA warp (producer warpgroup, warp 0): A + B loads.
        # ------------------------------------------------------------------
        if warp_idx == self.load_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            mainloop_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            # A is GEMM1's output — the one read that must wait (see above).
            cute.arch.griddepcontrol_wait()

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m_tile_idx = tile_coord_mnl[0]
                if m_tile_idx < num_valid_tiles:
                    expert_idx = tile_idx_to_expert_idx[m_tile_idx]
                    tAgA_mkl = tAgA[(None, m_tile_idx, None, 0)]
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, expert_idx)]

                    mainloop_producer_state.reset_count()
                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        mainloop_pipeline.producer_acquire(mainloop_producer_state)
                        tAgA_k = tAgA_mkl[(None, mainloop_producer_state.count)]
                        tAsA_pipe = tAsA[(None, mainloop_producer_state.index)]
                        tBgB_k = tBgB_nkl[(None, mainloop_producer_state.count)]
                        tBsB_pipe = tBsB[(None, mainloop_producer_state.index)]
                        cute.copy(
                            tma_atom_a,
                            tAgA_k,
                            tAsA_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                            mcast_mask=a_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_b,
                            tBgB_k,
                            tBsB_pipe,
                            tma_bar_ptr=mainloop_pipeline.producer_get_barrier(
                                mainloop_producer_state
                            ),
                        )
                        mainloop_pipeline.producer_commit(mainloop_producer_state)
                        mainloop_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            mainloop_pipeline.producer_tail(mainloop_producer_state)

        # ------------------------------------------------------------------
        # Meta warp (producer warpgroup, warp 1): per-row (out_idx, scale).
        # ------------------------------------------------------------------
        if warp_idx == self.meta_warp_id:
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()
            meta_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.meta_stage
            )
            lane = tidx % 32
            rows_per_pass = self.tile_shape_mnk[0] // 32

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m_tile_idx = tile_coord_mnl[0]
                if m_tile_idx < num_valid_tiles:
                    mn_limit = tile_idx_to_mn_limit[m_tile_idx]
                    gToken_tile = gToken_ml[(None, m_tile_idx)]

                    meta_pipeline.producer_acquire(meta_producer_state)
                    stage = meta_producer_state.index
                    for j in cutlass.range_constexpr(rows_per_pass):
                        r = lane + j * 32
                        row_global = m_tile_idx * self.tile_shape_mnk[0] + r
                        expanded_idx = gToken_tile[r]
                        # Padding rows hold garbage (not -1) — clamp
                        # branchlessly, then zero out via validity.
                        safe_idx = cutlass.max(expanded_idx, 0)
                        token_idx = safe_idx // self.topk
                        k_slot = safe_idx % self.topk
                        is_valid_row = cutlass.Int32(row_global < mn_limit)
                        gather_tok = token_idx * is_valid_row
                        gather_k = k_slot * is_valid_row
                        if cutlass.const_expr(self.use_fused_finalize):
                            scale = token_final_scales[(gather_tok, gather_k)]
                            out_idx = token_idx
                        else:
                            scale = cutlass.Float32(1.0)
                            out_idx = safe_idx
                        sMetaTokenIdx[(r, stage)] = out_idx
                        sMetaScale[(r, stage)] = scale
                    cute.arch.fence_proxy("async.shared", space="cta")
                    meta_pipeline.producer_commit(meta_producer_state)
                    meta_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            meta_pipeline.producer_tail(meta_producer_state)

        # ------------------------------------------------------------------
        # MMA warpgroups: WGMMA mainloop + finalize epilogue.
        # ------------------------------------------------------------------
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            mainloop_consumer_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            mainloop_consumer_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            meta_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.meta_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            # Per-subtile epilogue over 32-column windows — the same
            # structure as the gather kernel's epilogue: the retiled
            # accumulator is consumed in flat n-fast subtile chunks of
            # size_tRS_rD elements; pairing the whole tile in one shot is
            # NOT order-safe (retile and partition_D flatten differently).
            copy_atom_r2s = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), self.c_dtype
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(False, 4),
                self.c_dtype,
            )
            tiled_copy_C_Atom = cute.make_tiled_copy_C_atom(copy_atom_C, tiled_mma)
            tiled_copy_r2s = cute.make_tiled_copy_S(
                copy_atom_r2s,
                tiled_copy_C_Atom,
            )
            thr_copy_r2s = tiled_copy_r2s.get_slice(
                tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group
            )
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            is_cooperative = self.atom_layout_mnk == (2, 1, 1)
            epi_m = (
                min(128, self.tile_shape_mnk[0])
                if is_cooperative
                else min(64, self.tile_shape_mnk[0])
            )
            epi_tile = (epi_m, 32)
            m_subs = self.tile_shape_mnk[0] // epi_tile[0]
            n_subs = self.tile_shape_mnk[1] // epi_tile[1]
            sC_2d = sC[(None, None, 0)]
            # Per-element (row, col) coordinates within one subtile — same
            # partition order as the data stores below.
            epi_coords = cute.make_identity_tensor(epi_tile)
            tRS_cD = thr_copy_r2s.partition_D(epi_coords)

            rD_shape = cute.shape(
                thr_copy_r2s.partition_D(cute.local_tile(sC_2d, epi_tile, (0, 0)))
            )
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD_out)

            k_pipe_mmas = 1
            prologue_mma_cnt = cutlass.min(k_pipe_mmas, k_tile_cnt)

            copy_bytes = cutlass.Int32(self.tile_shape_mnk[1] * self.c_dtype.width // 8)
            epi_tidx = tidx - self.num_dma_warp_groups * self.num_threads_per_warp_group

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m_tile_idx = tile_coord_mnl[0]
                if m_tile_idx < num_valid_tiles:
                    mn_limit = tile_idx_to_mn_limit[m_tile_idx]

                    # MAINLOOP (standard WGMMA prologue + steady state)
                    mainloop_consumer_read_state.reset_count()
                    mainloop_consumer_release_state.reset_count()
                    accumulators.fill(0.0)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.fence()

                    for _k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                        mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                mainloop_consumer_read_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[k_block_coord],
                                tCrB[k_block_coord],
                                accumulators,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        mainloop_consumer_read_state.advance()

                    for _k_tile in cutlass.range(
                        prologue_mma_cnt, k_tile_cnt, 1, unroll=1
                    ):
                        mainloop_pipeline.consumer_wait(mainloop_consumer_read_state)
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                mainloop_consumer_read_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[k_block_coord],
                                tCrB[k_block_coord],
                                accumulators,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        cute.nvgpu.warpgroup.wait_group(k_pipe_mmas)
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_release_state
                        )
                        mainloop_consumer_release_state.advance()
                        mainloop_consumer_read_state.advance()

                    cute.nvgpu.warpgroup.wait_group(0)
                    for _k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                        mainloop_pipeline.consumer_release(
                            mainloop_consumer_release_state
                        )
                        mainloop_consumer_release_state.advance()

                    # EPILOGUE: scale rows, stage in sC, scatter-reduce.
                    meta_pipeline.consumer_wait(meta_consumer_state)
                    meta_stage_idx = meta_consumer_state.index

                    # Drain the PREVIOUS tile's scatter before overwriting
                    # sC (deferred from issue time so the bulk-reduce drains
                    # under this tile's whole mainloop instead of stalling
                    # its own tile's tail; adds commute, so overlapping
                    # scatters from adjacent tiles are order-safe). Each
                    # issuing thread waits its own groups; the barrier keeps
                    # any thread from overwriting sC bytes another thread's
                    # in-flight scatter still reads.
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    self.epilog_sync_barrier.arrive_and_wait()

                    for m_sub in cutlass.range_constexpr(m_subs):
                        for n_sub in cutlass.range_constexpr(n_subs):
                            chunk_base = (m_sub * n_subs + n_sub) * size_tRS_rD
                            for v in cutlass.range_constexpr(size_tRS_rD):
                                row = epi_m * m_sub + tRS_cD[v][0]
                                scale = sMetaScale[(row, meta_stage_idx)]
                                tRS_rD_out[v] = self.c_dtype(
                                    tRS_rAcc[chunk_base + v] * scale
                                )
                            sC_sub = cute.local_tile(sC_2d, epi_tile, (m_sub, n_sub))
                            tRS_sD_sub = thr_copy_r2s.partition_D(sC_sub)
                            cute.copy(tiled_copy_r2s, tRS_rD_out, tRS_sD_sub)

                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                    # One row per thread; rows beyond mn_limit are skipped.
                    if epi_tidx < self.tile_shape_mnk[0]:
                        row_global = m_tile_idx * self.tile_shape_mnk[0] + epi_tidx
                        if row_global < mn_limit:
                            out_row = sMetaTokenIdx[(epi_tidx, meta_stage_idx)]
                            coord_n = tile_coord_mnl[1] * self.tile_shape_mnk[1]
                            dst = cute.domain_offset((out_row, coord_n, 0), mOut)
                            src_row = sC[(epi_tidx, None, 0)]
                            if cutlass.const_expr(self.use_fused_finalize):
                                if cutlass.const_expr(self.c_dtype is cutlass.BFloat16):
                                    blk_reduce_bf16(dst, src_row, copy_bytes)
                                elif cutlass.const_expr(
                                    self.c_dtype is cutlass.Float16
                                ):
                                    blk_reduce_fp16(dst, src_row, copy_bytes)
                                else:
                                    blk_reduce_fp32(dst, src_row, copy_bytes)
                            else:
                                blk_copy(dst, src_row, copy_bytes)
                    cute.arch.cp_async_bulk_commit_group()

                    meta_pipeline.consumer_release(meta_consumer_state)
                    meta_consumer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            # Drain this thread's last outstanding scatter before exit (the
            # per-tile drain is deferred to sC reuse, so the final tile's
            # groups are still in flight when the scheduler runs dry).
            cute.arch.cp_async_bulk_wait_group(0, read=True)

        # PDL: every issuing thread has waited its own scatter groups above,
        # so all scatter-reduce writes are drained before this trigger; the
        # dependent grid's wait then observes them.
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        token_final_scales_ptr: cute.Pointer,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        num_tokens: cutlass.Int64,
        out_rows: cutlass.Int64,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Pointer-based entry (dims dynamic; dtype/tactic specialize)."""
        num_tiles = m // self.tile_shape_mnk[0]
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        out = cute.make_tensor(
            out_ptr,
            layout=cute.make_ordered_layout((out_rows, n, 1), order=(1, 0, 2)),
        )
        tile_idx_to_expert_idx = cute.make_tensor(
            tile_idx_to_expert_idx_ptr, layout=cute.make_layout((num_tiles,))
        )
        tile_idx_to_mn_limit = cute.make_tensor(
            tile_idx_to_mn_limit_ptr, layout=cute.make_layout((num_tiles,))
        )
        token_id_mapping = cute.make_tensor(
            token_id_mapping_ptr, layout=cute.make_layout((m,))
        )
        num_non_exiting_tiles = cute.make_tensor(
            num_non_exiting_tiles_ptr, layout=cute.make_layout((1,))
        )
        token_final_scales = cute.make_tensor(
            token_final_scales_ptr,
            layout=cute.make_ordered_layout((num_tokens, self.topk), order=(1, 0)),
        )
        return self(
            a,
            b,
            out,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            token_final_scales,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        c_dtype: type[cutlass.Numeric],
        c_row_pad: int,
        meta_stage: int,
        smem_capacity: int,
        occupancy: int,
    ) -> int:
        """A/B stage count after reserving the sC tile + meta buffers."""
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes = (
            tile_shape_mnk[0] * (tile_shape_mnk[1] + c_row_pad) * c_dtype.width // 8
        )
        meta_bytes = tile_shape_mnk[0] * meta_stage * 8  # i32 + f32
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy
            - (mbar_helpers_bytes + c_bytes + meta_bytes + 2048)
        ) // ab_bytes_per_stage
        return ab_stage

    @staticmethod
    def _make_ab_smem_layouts(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        a_layout: utils.LayoutEnum,
        b_dtype: type[cutlass.Numeric],
        b_layout: utils.LayoutEnum,
        ab_stage: int,
    ) -> tuple[cute.ComposedLayout, cute.ComposedLayout]:
        a_smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        a_is_k_major = a_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        a_major_mode_size = tile_shape_mnk[2 if a_is_k_major else 0]
        a_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(a_layout, a_dtype, a_major_mode_size),
            a_dtype,
        )
        a_smem_layout_staged = cute.tile_to_shape(
            a_smem_layout_atom,
            cute.append(a_smem_shape, ab_stage),
            order=(0, 1, 2) if a_is_k_major else (1, 0, 2),
        )

        b_smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        b_is_k_major = b_layout.sm90_mma_major_mode() == cute.nvgpu.OperandMajorMode.K
        b_major_mode_size = tile_shape_mnk[2 if b_is_k_major else 1]
        b_smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(b_layout, b_dtype, b_major_mode_size),
            b_dtype,
        )
        b_smem_layout_staged = cute.tile_to_shape(
            b_smem_layout_atom,
            cute.append(b_smem_shape, ab_stage),
            order=(0, 1, 2) if b_is_k_major else (1, 0, 2),
        )
        return a_smem_layout_staged, b_smem_layout_staged

    @staticmethod
    def _compute_grid(
        a: cute.Tensor,
        cta_tile_shape_mn: tuple[int, int],
        b: cute.Tensor,
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ):
        """Grid over (padded M tiles) x (N tiles); N from B's row count."""
        num_m_tiles = cute.size(a, mode=[0]) // cta_tile_shape_mn[0]
        num_n_tiles = cute.ceil_div(cute.size(b, mode=[0]), cta_tile_shape_mn[1])
        num_ctas_mnl = (num_m_tiles, num_n_tiles, 1)
        cluster_shape_mnl = (*cluster_shape_mn, 1)

        tile_sched_params = utils.PersistentTileSchedulerParams(
            num_ctas_mnl,
            cluster_shape_mnl,
            swizzle_size,
            raster_along_m,
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid
