# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
#
# Kernel structure adapted from NVIDIA CUTLASS CuTe-DSL example
# examples/python/CuTeDSL/hopper/dense_gemm_persistent.py (BSD-3-Clause).

"""SM90 gather grouped GEMM with fused gated activation (MoE GEMM1).

Computes, for each valid permuted row ``r`` belonging to expert ``e``:

    C[r, j] = act(gate_j) * up_j    where
    [up_j, gate_j] = A[token(r)] @ B[e].T   (columns interleaved, see below)

- **Gather fusion**: A rows are fetched directly from the *unpermuted* token
  activations using ``permuted_idx_to_expanded_idx`` (``token_id_mapping``) —
  the MoE permute is never materialized. The gather is done with cp.async
  (LDGSTS) by the producer warpgroup; rows beyond an expert's real count
  (``tile_idx_to_mn_limit``) are predicated off.
- **Activation fusion**: the epilogue applies SiLU-gating in f32 registers and
  stores ``N/2`` output columns.
- **Weight interleave**: B's N dimension holds up/gate interleaved at
  **32-column granularity**: ``[up 0:32 | gate 0:32 | up 32:64 | ...]``.
  This matches the epilogue subtile width, so an even accumulator subtile is
  always "up" and the following odd subtile is its "gate". Consequently
  ``tile_n % 64 == 0`` is required for gated activations.

Pipeline structure:

- A and B ride **separate pipelines**: A on ``PipelineCpAsync`` (128 gather
  threads produce with ``cp.async.mbarrier.arrive``), B on ``PipelineTmaAsync``
  (warp 0). The consumer warpgroups wait on both per K-tile.
- Gather SMEM destinations come from a **tiled-copy ``partition_D`` over the
  staged (swizzled) A layout** — the same machinery the epilogue uses — so
  the cp.async writes and the WGMMA descriptor agree.
- Cluster: A is never multicast (cp.async cannot). Cluster (2, 1) multicasts
  B across same-expert M-tile pairs; the MoE pipeline keeps GEMM1 fixed at
  (1, 1) — L2 already dedupes concurrent same-expert B reads, so the
  multicast saves no bytes while unshared pairs pay dual half-tile copies
  (design doc §4.5). (2, 1) remains reachable via the host wrapper's
  ``cluster_shape_mn``.

Gather geometry (bf16, CTA K-tile 64 = 128 B per row):
  128 producer threads; thread ``t`` covers row-group ``t // 8`` and 16-byte
  chunk ``t % 8``; ``tile_m / 16`` row-iterations of one cp.async.128 each.
"""

import math

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.hopper_helpers as sm90_utils
from cutlass.cute.nvgpu import cpasync
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait

from . import utils as hopper_utils
from ..common.kernel_utils import silu_f32


class Sm90ContiguousGatherGroupedGemmActFusionKernel:
    """Persistent warp-specialized MoE GEMM1: gather + grouped GEMM + SwiGLU.

    :param acc_dtype: Accumulator dtype (Float32 for bf16 inputs).
    :param tile_shape_mn: CTA tile (M, N) over the **accumulator** (N counts
        interleaved up+gate columns; the C output has N/2 columns).
        M in {64, 128}; N % 64 == 0, N <= 256.
    :param topk: MoE top-k (token id = ``token_id_mapping[row] // topk``).
    :param raster_along_m: Persistent walk order.

    Output C is ``[permuted_m, N_total/2]`` where ``N_total`` is B's row count
    per expert. Rows in padding tiles or beyond ``mn_limit`` hold garbage and
    must be masked downstream.
    """

    def __init__(
        self,
        acc_dtype: type[cutlass.Numeric],
        tile_shape_mn: tuple[int, int],
        topk: int,
        cluster_shape_mn: tuple[int, int] = (1, 1),
        swizzle_size: int = 1,
        raster_along_m: bool = False,
        enable_pdl: bool = True,
    ):
        """``cluster_shape_mn``: (1, 1) or (2, 1). (2, 1) pairs two adjacent
        M-tile CTAs and TMA-multicasts B when both tiles belong to the SAME
        expert (contiguous-grouped layout makes same-expert tiles adjacent) —
        a roofline upper bound of ~-25% GEMM1 DRAM traffic at ~1.5
        m-tiles/expert, not realized in practice: L2 dedup already covers the
        shared reads, so the MoE pipeline keeps (1, 1). When the pair spans an
        expert boundary each CTA loads its own B (both halves, self mask).
        Validity is CLUSTER-granular: if only the pair's first tile is real,
        the second CTA still runs the full pipeline with zero valid rows
        (gather fully predicated off, expert borrowed from the peer, garbage
        written to its own padded C tile) — required because the cluster
        empty-barrier arrivals are static. A is per-CTA cp.async (never
        multicast)."""
        if cluster_shape_mn not in ((1, 1), (2, 1)):
            raise ValueError("cluster_shape_mn must be (1, 1) or (2, 1)")
        self.acc_dtype = acc_dtype
        self.topk = topk
        self.cluster_shape_mn = cluster_shape_mn
        self.swizzle_size = swizzle_size
        self.raster_along_m = raster_along_m
        self.enable_pdl = enable_pdl
        self.tile_shape_mnk = (*tile_shape_mn, 1)
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
        self.load_warp_id = 0  # TMA-B warp within the producer warpgroup
        self.epi_store_warp_id = (
            self.num_dma_warp_groups * self.num_warps_per_warp_group
        )
        # Register budgets must fit the 64K-register file:
        # threads_producer * load_req + threads_consumer * mma_req <= 65536.
        # With 2 consumer warpgroups (384 threads) only 40/232 fits
        # (128*40 + 256*232 = 64512); with 1 consumer warpgroup there is
        # headroom for a larger gather producer.
        self.load_register_requirement = 40 if self.num_mma_warp_groups == 2 else 56
        self.mma_register_requirement = 232
        self.smem_capacity = utils.get_smem_capacity_in_bytes("sm_90")

        self.ab_stage = None
        self.epi_stage = None

        self.a_smem_layout_staged = None
        self.b_smem_layout_staged = None
        self.epi_smem_layout_staged = None

        self.buffer_align_bytes = 1024

        self.num_mma_threads = (
            self.num_mma_warp_groups * self.num_threads_per_warp_group
        )
        self.num_gather_threads = 128  # whole producer warpgroup gathers A
        self.epilog_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.num_mma_threads
        )

    def _setup_attributes(self):
        if self.tile_shape_mnk[0] not in [64, 128]:
            raise ValueError("CTA tile shape M must be 64/128")
        if self.tile_shape_mnk[1] % 64 != 0 or not 64 <= self.tile_shape_mnk[1] <= 256:
            raise ValueError(
                "CTA tile shape N must be a multiple of 64 (up/gate pairs of "
                "32-col epilogue subtiles), <= 256"
            )

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
        mma_inst_tile_k = 4
        self.tile_shape_mnk = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1],
            mma_inst_shape_k * mma_inst_tile_k,
        )
        if self.tile_shape_mnk[2] != 64:
            raise ValueError("the gather loader assumes a 16-bit K-tile of 64")

        self.cta_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))

        # C tile: half the accumulator columns (gated activation).
        self.cta_tile_shape_c = (
            self.tile_shape_mnk[0],
            self.tile_shape_mnk[1] // 2,
        )

        is_cooperative = self.atom_layout_mnk == (2, 1, 1)
        # Epilogue subtile: n=32 == the up/gate interleave granularity.
        self.epi_tile = (
            (min(128, self.tile_shape_mnk[0]), 32)
            if is_cooperative
            else (min(64, self.tile_shape_mnk[0]), 32)
        )

        self.ab_stage, self.epi_stage = self._compute_stages(
            self.tile_shape_mnk,
            self.a_dtype,
            self.b_dtype,
            self.epi_tile,
            self.c_dtype,
            self.smem_capacity,
            self.occupancy,
        )

        (
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
        ) = hopper_utils.make_smem_layouts(
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
        a: cute.Tensor,
        b: cute.Tensor,
        c: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Compile-time entry.

        :param a: Unpermuted token activations (orig_m, K, 1), K-major.
        :param b: Expert weights (N, K, E), K-major, up/gate interleaved at
            32 columns along N.
        :param c: Output (permuted_m, N/2, 1), N-major.
        :param token_id_mapping: (permuted_m,) Int32
            ``permuted_idx_to_expanded_idx`` (garbage on padding rows).
        """
        self.a_dtype = a.element_type
        self.b_dtype = b.element_type
        self.c_dtype = c.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(a)
        self.b_layout = utils.LayoutEnum.from_tensor(b)
        self.c_layout = utils.LayoutEnum.from_tensor(c)

        if cutlass.const_expr(self.a_dtype.width != 16):
            raise TypeError("this kernel supports 16-bit A/B (bf16/fp16) only")
        if cutlass.const_expr(self.a_dtype != self.b_dtype):
            raise TypeError(f"Type mismatch: {self.a_dtype} != {self.b_dtype}")

        self._setup_attributes()

        # A is gathered with cp.async — no TMA atom for A.
        tma_atom_b, tma_tensor_b = hopper_utils.make_tma_atoms_and_tensors(
            b,
            self.b_smem_layout_staged,
            (self.tile_shape_mnk[1], self.tile_shape_mnk[2]),
            self.cluster_shape_mn[0],
        )

        tma_atom_c, tma_tensor_c = hopper_utils.make_tma_store_atoms_and_tensors(
            c,
            self.epi_smem_layout_staged,
            self.epi_tile,
        )

        tile_sched_params, grid = self._compute_grid(
            c,
            self.cta_tile_shape_c,
            self.cluster_shape_mn,
            self.swizzle_size,
            self.raster_along_m,
            max_active_clusters,
        )

        @cute.struct
        class SharedStorage:
            a_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
            b_pipeline_array_ptr: cute.struct.MemRange[cutlass.Int64, self.ab_stage * 2]
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
                    self.c_dtype,
                    cute.cosize(self.epi_smem_layout_staged),
                ],
                self.buffer_align_bytes,
            ]

        self.shared_storage = SharedStorage

        self.kernel(
            a,
            tma_atom_b,
            tma_tensor_b,
            tma_atom_c,
            tma_tensor_c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            self.tiled_mma,
            self.cta_layout_mnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.epi_smem_layout_staged,
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
        mA_mkl: cute.Tensor,
        tma_atom_b: cute.CopyAtom,
        mB_nkl: cute.Tensor,
        tma_atom_c: cute.CopyAtom,
        mC_mnl: cute.Tensor,
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        token_id_mapping: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tiled_mma: cute.TiledMma,
        cta_layout_mnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        epi_smem_layout_staged: cute.ComposedLayout,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        if warp_idx == 0:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_b)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_c)

        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        cluster_coord_mnk = cta_layout_mnk.get_flat_coord(cta_rank_in_cluster)
        # B multicast masks (cluster (2,1) only): full = both M-pair CTAs,
        # self = local-only through the multicast atom.
        b_full_mask = cute.make_layout_image_mask(
            cta_layout_mnk, cluster_coord_mnk, mode=0
        )
        b_self_mask = cutlass.Int16(1) << cta_rank_in_cluster

        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        b_copy_bytes = cute.size_in_bytes(self.b_dtype, b_smem_layout)

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        # A pipeline: cp.async gather producers (128 threads), WGMMA consumers.
        a_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_gather_threads
        )
        a_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mma_threads
        )
        a_pipeline = pipeline.PipelineCpAsync.create(
            barrier_storage=storage.a_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=a_producer_group,
            consumer_group=a_consumer_group,
            defer_sync=True,
        )

        # B pipeline: TMA producer (1 thread), per-warp consumer arrives
        # (scaled by the B multicast size across the cluster M pair).
        b_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        b_consumer_arrive_cnt = (
            self.cluster_shape_mn[0]
            * self.num_mma_warp_groups
            * self.num_warps_per_warp_group
        )
        b_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, b_consumer_arrive_cnt
        )
        b_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.b_pipeline_array_ptr.data_ptr(),
            num_stages=self.ab_stage,
            producer_group=b_producer_group,
            consumer_group=b_consumer_group,
            tx_count=b_copy_bytes,
            cta_layout_vmnk=cute.make_layout((1, *cta_layout_mnk.shape)),
            defer_sync=True,
        )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sC = storage.sC.get_tensor(
            epi_smem_layout_staged.outer, swizzle=epi_smem_layout_staged.inner
        )

        # (bN, bK, RestN, RestK, RestL=E)
        gB_nkl = cute.local_tile(
            mB_nkl,
            cute.slice_(self.tile_shape_mnk, (0, None, None)),
            (None, None, None),
        )
        # C tiles over the halved-N output: rank-2 tiler over the rank-3
        # tensor -> (bM, bN, RestM, RestN, RestL).
        gC_mnl = cute.local_tile(
            mC_mnl,
            self.cta_tile_shape_c,
            (None, None, None),
        )
        # Token-id map tiled per M-tile.
        gToken_ml = cute.local_tile(
            token_id_mapping,
            (self.tile_shape_mnk[0],),
            (None,),
        )

        b_cta_layout = cute.make_layout(cute.slice_(cta_layout_mnk, (None, 0, 0)).shape)
        rank_m = cluster_coord_mnk[0]
        tBsB, tBgB = cute.nvgpu.cpasync.tma_partition(
            tma_atom_b,
            rank_m,
            b_cta_layout,
            cute.group_modes(sB, 0, 2),
            cute.group_modes(gB_nkl, 0, 2),
        )
        # Peer-share partition: in the unshared (different experts) case each
        # CTA also issues the peer's share of the split tile locally.
        if cutlass.const_expr(self.cluster_shape_mn[0] == 2):
            tBsB_peer, tBgB_peer = cute.nvgpu.cpasync.tma_partition(
                tma_atom_b,
                1 - rank_m,
                b_cta_layout,
                cute.group_modes(sB, 0, 2),
                cute.group_modes(gB_nkl, 0, 2),
            )
        else:
            tBsB_peer, tBgB_peer = tBsB, tBgB

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

        # Accumulator over the full (tile_m, tile_n) acc tile. Partitioning C
        # geometry cannot be used here (C has N/2 cols); shape the accumulator
        # from an acc-tile-sized coordinate tensor instead.
        acc_tile_coords = cute.make_identity_tensor(
            (self.tile_shape_mnk[0], self.tile_shape_mnk[1], 1)
        )
        tCgAcc = thr_mma.partition_C(acc_tile_coords)
        accumulators = cute.make_rmem_tensor(tCgAcc.shape[:3], self.acc_dtype)

        k_tile_cnt = cutlass.Int32(cute.size(gB_nkl, mode=[3]))

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        # PDL: everything above is descriptor prefetch, SMEM/pipeline setup,
        # and address arithmetic; the moe_sort outputs (num_non_exiting_tiles,
        # tile maps, token_id_mapping) must not be read before this wait.
        cute.arch.griddepcontrol_wait()

        num_valid_tiles = num_non_exiting_tiles[0]

        # Gather geometry: thread t covers row-group t//8 and 16 B chunk t%8;
        # tile_m/16 row-iterations per K-tile.
        tidx_in_warpgroup = tidx % 128
        gather_row = tidx_in_warpgroup // 8
        gather_chunk = tidx_in_warpgroup % 8
        m_iters = self.tile_shape_mnk[0] // 16
        elems_per_chunk = 8  # 16 B of 16-bit elements

        a_atom_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cute.nvgpu.LoadCacheMode.GLOBAL),
            mA_mkl.element_type,
            num_bits_per_copy=128,
        )
        a_thr_layout = cute.make_layout((16, 8), stride=(8, 1))
        a_val_layout = cute.make_layout((1, 8), stride=(8, 1))
        a_tiled_copy = cute.make_tiled_copy_tv(a_atom_copy, a_thr_layout, a_val_layout)
        a_thr_copy = a_tiled_copy.get_slice(tidx % 128)
        # (V, RestM = tile_m/16, RestK = 1, STAGE)
        tAsA_gather = a_thr_copy.partition_D(sA)

        # ------------------------------------------------------------------
        # Load warps (producer warpgroup): all four warps cp.async A
        # through the permute map; warp 0's elected lane TMA-loads B.
        # ------------------------------------------------------------------
        is_dma_warp_group = warp_group_idx < self.num_dma_warp_groups
        if is_dma_warp_group:
            cute.arch.setmaxregister_decrease(self.load_register_requirement)

            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            a_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.ab_stage
            )

            token_offset = cute.make_rmem_tensor(
                cute.make_layout((m_iters,)), cutlass.Int32
            )
            row_pred = cute.make_rmem_tensor(
                cute.make_layout((m_iters,)), cutlass.Boolean
            )

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m_tile_idx = tile_coord_mnl[0]
                # CLUSTER-granular validity: with cluster (2,1) the whole
                # pair processes if its FIRST tile is real (the cluster
                # empty-barrier arrivals are static, so both members must
                # pump every stage). An invalid second member runs with zero
                # valid rows and the peer's expert.
                base_tile_idx = m_tile_idx - rank_m
                if base_tile_idx < num_valid_tiles:
                    self_valid = m_tile_idx < num_valid_tiles
                    safe_tile_idx = m_tile_idx if self_valid else base_tile_idx
                    expert_idx = tile_idx_to_expert_idx[safe_tile_idx]
                    mn_limit = (
                        tile_idx_to_mn_limit[m_tile_idx]
                        if self_valid
                        else m_tile_idx * self.tile_shape_mnk[0]
                    )
                    if cutlass.const_expr(self.cluster_shape_mn[0] == 2):
                        # base+1 is always < max_num_tiles (grid extent), so
                        # the read is in-bounds even when logically invalid.
                        peer_tile_idx = base_tile_idx + 1
                        peer_valid = peer_tile_idx < num_valid_tiles
                        same_expert = peer_valid & (
                            tile_idx_to_expert_idx[base_tile_idx]
                            == tile_idx_to_expert_idx[peer_tile_idx]
                        )
                    else:
                        same_expert = cutlass.Boolean(False)
                    tBgB_nkl = tBgB[(None, tile_coord_mnl[1], None, expert_idx)]
                    tBgB_peer_nkl = tBgB_peer[
                        (None, tile_coord_mnl[1], None, expert_idx)
                    ]

                    # Per-tile gather rows: token id and validity per row-iter
                    # (K-invariant, computed once per tile).
                    gToken_tile = gToken_ml[(None, m_tile_idx)]
                    for i in cutlass.range_constexpr(m_iters):
                        row_in_tile = gather_row + i * 16
                        row_global = m_tile_idx * self.tile_shape_mnk[0] + row_in_tile
                        is_valid_row = row_global < mn_limit
                        row_pred[i] = is_valid_row
                        token_offset[i] = (
                            gToken_tile[row_in_tile] // self.topk if is_valid_row else 0
                        )

                    a_producer_state.reset_count()
                    b_producer_state.reset_count()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        # B: TMA by warp 0. Shared expert -> multicast the own
                        # share to both CTAs; otherwise load both shares locally
                        # through the multicast atom (self mask).
                        if warp_idx == self.load_warp_id:
                            b_pipeline.producer_acquire(b_producer_state)
                            tBgB_k = tBgB_nkl[(None, b_producer_state.count)]
                            tBsB_pipe = tBsB[(None, b_producer_state.index)]
                            if cutlass.const_expr(self.cluster_shape_mn[0] == 2):
                                if same_expert:
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB_k,
                                        tBsB_pipe,
                                        tma_bar_ptr=b_pipeline.producer_get_barrier(
                                            b_producer_state
                                        ),
                                        mcast_mask=b_full_mask,
                                    )
                                else:
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB_k,
                                        tBsB_pipe,
                                        tma_bar_ptr=b_pipeline.producer_get_barrier(
                                            b_producer_state
                                        ),
                                        mcast_mask=b_self_mask,
                                    )
                                    tBgB_peer_k = tBgB_peer_nkl[
                                        (None, b_producer_state.count)
                                    ]
                                    tBsB_peer_pipe = tBsB_peer[
                                        (None, b_producer_state.index)
                                    ]
                                    cute.copy(
                                        tma_atom_b,
                                        tBgB_peer_k,
                                        tBsB_peer_pipe,
                                        tma_bar_ptr=b_pipeline.producer_get_barrier(
                                            b_producer_state
                                        ),
                                        mcast_mask=b_self_mask,
                                    )
                            else:
                                cute.copy(
                                    tma_atom_b,
                                    tBgB_k,
                                    tBsB_pipe,
                                    tma_bar_ptr=b_pipeline.producer_get_barrier(
                                        b_producer_state
                                    ),
                                )
                            b_pipeline.producer_commit(b_producer_state)
                        b_producer_state.advance()

                        # A: cp.async gather by all 128 producer threads.
                        a_pipeline.producer_acquire(a_producer_state)
                        k_base = a_producer_state.count * self.tile_shape_mnk[2]
                        chunk_col = gather_chunk * elems_per_chunk
                        for i in cutlass.range_constexpr(m_iters):
                            # Source: token row in the unpermuted activations.
                            src_off = cute.assume(
                                token_offset[i] * mA_mkl.layout[0].stride,
                                divby=8,
                            ) + cute.assume(k_base + chunk_col, divby=8)
                            tAgA_slice = cute.make_tensor(
                                mA_mkl.iterator + src_off,
                                layout=cute.make_layout((elems_per_chunk,)),
                            )
                            # Destination: partitioned slice of the swizzled
                            # sA (addressing handled by the copy lowering).
                            tAsA_slice = tAsA_gather[
                                (None, i, 0, a_producer_state.index)
                            ]
                            pred = cute.make_rmem_tensor(
                                cute.make_layout((1,)), cutlass.Boolean
                            )
                            pred[0] = row_pred[i]
                            cute.copy_atom_call(
                                a_atom_copy, tAgA_slice, tAsA_slice, pred=pred
                            )

                        a_pipeline.producer_commit(a_producer_state)
                        a_producer_state.advance()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            a_pipeline.producer_tail(a_producer_state)
            if warp_idx == self.load_warp_id:
                b_pipeline.producer_tail(b_producer_state)

        # ------------------------------------------------------------------
        # MMA warpgroups: WGMMA mainloop + gated SwiGLU epilogue.
        # ------------------------------------------------------------------
        if not is_dma_warp_group:
            cute.arch.setmaxregister_increase(self.mma_register_requirement)
            tile_sched = utils.StaticPersistentTileScheduler.create(
                tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
            )
            work_tile = tile_sched.initial_work_tile_info()

            a_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            a_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            b_read_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )
            b_release_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.ab_stage
            )

            num_k_blocks = cute.size(tCrA, mode=[2])

            copy_atom_r2s = sm90_utils.sm90_get_smem_store_op(
                self.c_layout,
                elem_ty_d=self.c_dtype,
                elem_ty_acc=self.acc_dtype,
            )
            copy_atom_C = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix8x8x16bOp(
                    self.c_layout.is_m_major_c(),
                    4,
                ),
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
            tRS_sD = thr_copy_r2s.partition_D(sC)
            # (R2S, R2S_M, R2S_N): per-thread fragment x epi-subtile grid over
            # the full (tile_m, tile_n) accumulator.
            tRS_rAcc = tiled_copy_r2s.retile(accumulators)

            rD_shape = cute.shape(thr_copy_r2s.partition_S(sC))
            tRS_rD_layout = cute.make_layout(rD_shape[:3])
            tRS_rD = cute.make_rmem_tensor(tRS_rD_layout.shape, self.acc_dtype)
            tRS_rD_out = cute.make_rmem_tensor(tRS_rD_layout.shape, self.c_dtype)
            size_tRS_rD = cute.size(tRS_rD)

            k_pipe_mmas = 1
            prologue_mma_cnt = cutlass.min(k_pipe_mmas, k_tile_cnt)

            tma_store_producer_group = pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mma_threads,
            )
            tma_store_pipeline = pipeline.PipelineTmaStore.create(
                num_stages=self.epi_stage,
                producer_group=tma_store_producer_group,
            )

            # Epi-subtile grid over the accumulator: (m_subs, n_subs) with
            # n_subs = tile_n/32; C subtiles pair (2*nc, 2*nc+1) -> nc.
            m_subs = self.tile_shape_mnk[0] // self.epi_tile[0]
            n_subs = self.tile_shape_mnk[1] // self.epi_tile[1]
            n_subs_c = n_subs // 2

            while work_tile.is_valid_tile:
                tile_coord_mnl = work_tile.tile_idx
                m_tile_idx = tile_coord_mnl[0]
                if (m_tile_idx - rank_m) < num_valid_tiles:
                    gC_mnl_slice = gC_mnl[
                        (None, None, m_tile_idx, tile_coord_mnl[1], 0)
                    ]

                    # MAINLOOP
                    a_read_state.reset_count()
                    a_release_state.reset_count()
                    b_read_state.reset_count()
                    b_release_state.reset_count()
                    accumulators.fill(0.0)
                    tiled_mma.set(cute.nvgpu.warpgroup.Field.ACCUMULATE, True)
                    cute.nvgpu.warpgroup.fence()

                    for _k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                        a_pipeline.consumer_wait(a_read_state)
                        b_pipeline.consumer_wait(b_read_state)
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                a_read_state.index,
                            )
                            cute.gemm(
                                tiled_mma,
                                accumulators,
                                tCrA[k_block_coord],
                                tCrB[k_block_coord],
                                accumulators,
                            )
                        cute.nvgpu.warpgroup.commit_group()
                        a_read_state.advance()
                        b_read_state.advance()

                    for _k_tile in cutlass.range(
                        prologue_mma_cnt, k_tile_cnt, 1, unroll=1
                    ):
                        a_pipeline.consumer_wait(a_read_state)
                        b_pipeline.consumer_wait(b_read_state)
                        for k_block_idx in cutlass.range_constexpr(num_k_blocks):
                            k_block_coord = (
                                None,
                                None,
                                k_block_idx,
                                a_read_state.index,
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

                        a_pipeline.consumer_release(a_release_state)
                        b_pipeline.consumer_release(b_release_state)
                        a_release_state.advance()
                        b_release_state.advance()
                        a_read_state.advance()
                        b_read_state.advance()

                    cute.nvgpu.warpgroup.wait_group(0)
                    for _k_tile in cutlass.range(0, prologue_mma_cnt, 1, unroll=1):
                        a_pipeline.consumer_release(a_release_state)
                        b_pipeline.consumer_release(b_release_state)
                        a_release_state.advance()
                        b_release_state.advance()

                    # EPILOGUE: silu(gate) * up over 32-col subtile pairs.
                    tCgC_for_tma_partition = cute.zipped_divide(
                        gC_mnl_slice, self.epi_tile
                    )
                    bSG_sD, bSG_gD = cute.nvgpu.cpasync.tma_partition(
                        tma_atom_c,
                        0,
                        cute.make_layout(1),
                        cute.group_modes(sC, 0, 2),
                        tCgC_for_tma_partition,
                    )

                    num_prev_epi_tiles = tile_sched.num_tiles_executed * (
                        m_subs * n_subs_c
                    )
                    for m_sub in cutlass.range_constexpr(m_subs):
                        for nc in cutlass.range_constexpr(n_subs_c):
                            # Flat n-fast subtile chunking of the retiled
                            # accumulator: subtile s occupies elements
                            # [s*size_tRS_rD, (s+1)*size_tRS_rD) with
                            # s = m_sub * n_subs + n_sub. Even n_sub = up,
                            # odd n_sub = gate (32-col interleave).
                            up_base = (m_sub * n_subs + 2 * nc) * size_tRS_rD
                            gate_base = up_base + size_tRS_rD
                            for v in cutlass.range_constexpr(size_tRS_rD):
                                g = tRS_rAcc[gate_base + v]
                                u = tRS_rAcc[up_base + v]
                                tRS_rD[v] = silu_f32(g, fastmath=True) * u

                            acc_vec = tRS_rD.load()
                            tRS_rD_out.store(acc_vec.to(self.c_dtype))

                            epi_idx = m_sub * n_subs_c + nc
                            epi_buffer = (num_prev_epi_tiles + epi_idx) % cute.size(
                                tRS_sD, mode=[3]
                            )
                            cute.copy(
                                tiled_copy_r2s,
                                tRS_rD_out,
                                tRS_sD[(None, None, None, epi_buffer)],
                            )

                            cute.arch.fence_proxy(
                                "async.shared",
                                space="cta",
                            )
                            self.epilog_sync_barrier.arrive_and_wait()

                            if warp_idx == self.epi_store_warp_id:
                                cute.copy(
                                    tma_atom_c,
                                    bSG_sD[(None, epi_buffer)],
                                    bSG_gD[(None, (m_sub, nc))],
                                )
                                tma_store_pipeline.producer_commit()
                                tma_store_pipeline.producer_acquire()

                            self.epilog_sync_barrier.arrive_and_wait()

                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tma_store_pipeline.producer_tail()

        # PDL: all C tiles are TMA-store drained (producer_tail above), so the
        # dependent grid's griddepcontrol.wait observes completed writes.
        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        c_ptr: cute.Pointer,
        tile_idx_to_expert_idx_ptr: cute.Pointer,
        tile_idx_to_mn_limit_ptr: cute.Pointer,
        token_id_mapping_ptr: cute.Pointer,
        num_non_exiting_tiles_ptr: cute.Pointer,
        orig_m: cutlass.Int64,
        m: cutlass.Int64,
        n: cutlass.Int64,
        k: cutlass.Int64,
        l: cutlass.Int64,  # noqa: E741
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Pointer-based entry (dims dynamic; dtype/tactic specialize).

        :param orig_m: Unpermuted token count (rows of A).
        :param m: Padded permuted row count (``max_num_tiles * tile_m``).
        :param n: Interleaved up+gate weight rows per expert (2I per rank);
            C gets ``n // 2`` columns.
        """
        num_tiles = m // self.tile_shape_mnk[0]
        a = cute.make_tensor(
            a_ptr, layout=cute.make_ordered_layout((orig_m, k, 1), order=(1, 0, 2))
        )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((n, k, l), order=(1, 0, 2))
        )
        c = cute.make_tensor(
            c_ptr, layout=cute.make_ordered_layout((m, n // 2, 1), order=(1, 0, 2))
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
        return self(
            a,
            b,
            c,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            token_id_mapping,
            num_non_exiting_tiles,
            max_active_clusters=max_active_clusters,
            stream=stream,
        )

    @staticmethod
    def _compute_stages(
        tile_shape_mnk: tuple[int, int, int],
        a_dtype: type[cutlass.Numeric],
        b_dtype: type[cutlass.Numeric],
        epi_tile: tuple[int, int],
        c_dtype: type[cutlass.Numeric],
        smem_capacity: int,
        occupancy: int,
    ) -> tuple[int, int]:
        """A/B stage count from leftover SMEM after a fixed 4-stage epilogue."""
        a_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
        b_shape = cute.slice_(tile_shape_mnk, (0, None, None))
        ab_bytes_per_stage = (
            cute.size(a_shape) * a_dtype.width // 8
            + cute.size(b_shape) * b_dtype.width // 8
        )
        c_bytes_per_stage = cute.size(epi_tile) * c_dtype.width // 8
        epi_stage = 4
        epi_bytes = c_bytes_per_stage * epi_stage

        # Two pipelines' mbarrier arrays + slack.
        mbar_helpers_bytes = 1024

        ab_stage = (
            smem_capacity // occupancy - (mbar_helpers_bytes + epi_bytes)
        ) // ab_bytes_per_stage
        return ab_stage, epi_stage

    @staticmethod
    def _compute_grid(
        c: cute.Tensor,
        cta_tile_shape_c: tuple[int, int],
        cluster_shape_mn: tuple[int, int],
        swizzle_size: int,
        raster_along_m: bool,
        max_active_clusters: cutlass.Constexpr,
    ):
        """Static persistent grid over C's padded (max) tile extent."""
        gc = cute.zipped_divide(c, tiler=cta_tile_shape_c)
        num_ctas_mnl = gc[(0, (None, None, None))].shape
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
