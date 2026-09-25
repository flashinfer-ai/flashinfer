# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Swap-AB block-scaled grouped GEMM for few-tokens-per-expert MoE (SM100/SM103).

The existing MoE GEMM kernels tile the *token* dimension as MMA-M = 128. When an
expert receives only a handful of routed rows (decode, small prefill, MoE-TP
shards) every 128-row token tile spends full-size ``tcgen05.mma`` work on
padding while streaming the expert weights as the N operand. ``kind::mxf8f6f4``
fixes MMA-M at 128 for one CTA, so the only way to shrink the token dimension is
to swap the operands:

    acc[weight_rows(128) x tokens(N_TILE)] = W_e[rows x K] (A, TMA)
                                            x X[rows(N_TILE) x K]^T (B, cp.async gather)

with ``N_TILE`` in {8, 16, 32, 64, 128}. Weights therefore stream through TMA
exactly once per row group while the MMA work per weight byte drops by
128/N_TILE. For the narrow tiles (8-32) the row operand is gathered straight
from the unpermuted activations by a dedicated ``cp.async`` warp (GEMM1: rows
indexed through ``permuted_idx_to_expanded_idx``; GEMM2: contiguous permuted
rows), so no separate permute kernel is needed. The wide tiles (64, 128; or
``row_tma=True``) load the row operand with the TMA unit instead: GEMM1 uses
``cp.async.bulk.tensor ... tile::gather4`` driven by a per-permuted-row token
index (``permuted_idx_to_token_idx``; padding rows carry an out-of-range index
and are zero-filled by TMA), GEMM2 a plain tile load of the contiguous
permuted rows. The gather warp then only places the UE8M0 scale bytes. Its UE8M0 scale factors are plain
``(rows, K/32)`` bytes; the gather warp drops each row's 4 bytes per 128-wide
K tile into the scale-factor smem atom the ``tcgen05.cp`` S2T copy expects.
Every output column is one routed row of one expert; TMEM lane == weight row.

Two epilogues share the mainloop:

* ``"situ_mxfp8"`` (GEMM1): rows of the 128-row weight tile are the 64-row
  up/gate interleave produced by ``prepare_cute_dsl_mxfp4_weights``. Gate lanes
  (warps 2-3) compute ``beta*tanh(g/beta)*sigmoid(g)``, up lanes (warps 0-1)
  compute ``linear_beta*tanh(u/linear_beta)`` (or ``u``), the product is formed
  after an smem exchange, and the MXFP8 requantization group of 32 along the
  intermediate dimension is exactly one warp: a 32-lane max reduction per token
  column yields the UE8M0 scale. Output: ``act[rows, I]`` E4M3 + plain
  ``(rows, I/32)`` scale-factor bytes (consumed by the GEMM2 gather warp).
  Optionally zero-fills the finalize output buffer so GEMM2 can reduce into it.
* ``"finalize"`` (GEMM2): scale each token column by ``alpha * route_weight``,
  transpose the 128 x N_TILE tile through smem and ``cp.reduce.async.bulk`` add
  it into ``out[token, h0:h0+128]`` (BF16), the same reduction the existing
  finalize kernel uses.
* ``"partial"`` (GEMM2, pre-finalize): like ``"finalize"`` but writes
  ``alpha * acc`` to ``out[expanded_idx, h0:h0+128]`` with a plain bulk copy, so
  the caller owns the top-k combine.

Routing rows are grouped in tiles of ``N_TILE`` permuted rows per expert
(``moe_sort(tile_tokens_dim=N_TILE)``); ``tile_idx_to_expert_idx`` /
``tile_idx_to_mn_limit`` are indexed by row group. Requires SM100-class
tcgen05/TMEM (sm_100a, sm_103a).
"""

from typing import Optional, Tuple, Type

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
from cutlass._mlir.dialects import cute_nvgpu as _cute_nvgpu_ir
from cutlass.cute.nvgpu import cpasync, tcgen05

from flashinfer.quantization.quantization_cute_dsl_utils import (
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    UnalignedNamedBarrier,
    red_add_bf16x2_pair_pred,
    st_bf16_pred,
    st_bf16_pred_rowaddr,
    mapa_shared_cluster_u32,
    st_async_f32_cluster,
    mbarrier_arrive_cluster,
    griddepcontrol_launch_dependents,
    griddepcontrol_wait,
    native_situ_f32,
    native_tanh_f32,
    tcgen05_fence_after_thread_sync,
    tcgen05_fence_before_thread_sync,
)

EPILOGUE_KINDS = ("situ_mxfp8", "finalize", "partial")


class Sm100BlockScaledSwapAbGroupedGemmKernel:
    """Persistent warp-specialized swap-AB grouped GEMM (see module docstring).

    Warp roles: epilogue (0-3), MMA (4), TMA (5), scheduler (6), gather (7..).
    """

    def __init__(
        self,
        sf_vec_size: int,
        n_tile: int,
        k_blocks_per_stage: int,
        epilogue_kind: str,
        enable_pdl: bool = False,
        use_linear_beta: bool = True,
        max_ab_stages: int = 12,
        num_acc_stages: int = 2,
        num_tile_stages: int = 8,
        meta_in_sched: bool = True,
        perf_probe: int = 0,
        weight_l2_hint: Optional[int] = None,
        row_tma: Optional[bool] = None,
        gather_warps: Optional[int] = None,
        m_group: int = 1,
        group_rows: Optional[int] = None,
        sf_blocked: bool = False,
        wide_out: bool = False,
        pdl_trigger_early: bool = False,
        late_dep_wait: bool = False,
        pdl_trigger_after_wait: bool = False,
        split_k: int = 1,
        split_max_items: int = 0,
        cluster_split: bool = False,
    ):
        if epilogue_kind not in EPILOGUE_KINDS:
            raise ValueError(f"unknown epilogue_kind {epilogue_kind!r}")
        if n_tile not in (8, 16, 32, 64, 128):
            raise ValueError("n_tile must be 8, 16, 32, 64 or 128")
        if k_blocks_per_stage not in (4, 8, 12):
            # The gather warp addresses whole 128-element K atoms (one 128-byte
            # swizzle atom per FP8 row, one 512-byte SF atom per 128 rows).
            # 16 K-blocks per stage failed validation and was slower; not offered.
            raise ValueError("k_blocks_per_stage must be 4, 8 or 12")
        self.sf_vec_size = sf_vec_size
        self.n_tile = n_tile
        self.k_blocks_per_stage = k_blocks_per_stage
        self.epilogue_kind = epilogue_kind
        self.is_situ = epilogue_kind == "situ_mxfp8"
        self.is_finalize = epilogue_kind == "finalize"
        self.is_partial = epilogue_kind == "partial"
        # ``partial`` outputs whose ``rows * cols`` passes 2^31 elements form the
        # store offset in 64 bits (one wide IMAD per store); below that the
        # 32-bit tensor layout math is kept unchanged.
        self.wide_out = wide_out
        self.enable_pdl = enable_pdl
        # Trigger the programmatic dependents at kernel entry instead of at
        # the end of the epilogue: for launches that usually find few or no
        # work items (the wide launches of the split form) the dependent
        # grid becomes resident during this grid's prologue and wait.
        self.pdl_trigger_early = bool(pdl_trigger_early)
        # Dependent-side prefetch (plain routing -> GEMM1 -> GEMM2 chain):
        # ``late_dep_wait`` lets only the warps loading the row operand (the
        # predecessor's output) wait on the dependency, so the scheduler and
        # TMA warps stream the routing tables and the weight stages while the
        # predecessor still runs; every other input of this launch was written
        # by grids that completed before the predecessor started. Requires the
        # predecessor to trigger no earlier than after its own dependency wait
        # (``pdl_trigger_after_wait`` on it).
        self.late_dep_wait = bool(late_dep_wait)
        # Device-side adaptive split-K (finalize epilogue only: its
        # ``red.global.add`` output makes K partials additive). Work items are
        # (m_chunk * split_k + split, row_group); the scheduler warp publishes
        # each item's K range and splits only while the valid items
        # (num_valid_groups * m_chunks) fit ``split_max_items`` CTAs, so a row
        # with more tiles than SMs keeps whole-K items (split > 0 skipped).
        self.split_k = int(split_k)
        self.split_max_items = int(split_max_items)
        if self.split_k not in (1, 2, 3, 4):
            raise ValueError("split_k must be 1..4")
        # Cluster split-K (SiTU epilogue): the two CTAs of a (1, 1, 2) cluster
        # take the two K halves of one item; the peer (rank 1) ships its
        # alpha-scaled 128 x n_tile FP32 accumulator into the leader's shared
        # memory (``st.async`` + mbarrier complete_tx) and the leader adds it
        # before the activation epilogue. Unsplit launches remap the pair
        # onto two independent items (the original raster), so the cluster
        # only groups CTAs.
        self.cluster_split = bool(cluster_split)
        if self.cluster_split:
            if epilogue_kind != "situ_mxfp8" or self.split_k != 2:
                raise ValueError("cluster_split needs the situ_mxfp8 epilogue and split_k=2")
            if m_group != 1 or n_tile > 32:
                raise ValueError("cluster_split needs m_group=1 and n_tile <= 32")
        if self.split_k > 1 and epilogue_kind != "finalize" and not self.cluster_split:
            raise ValueError("split_k > 1 requires the finalize epilogue")
        self.pdl_trigger_after_wait = bool(pdl_trigger_after_wait)
        if self.pdl_trigger_early and self.pdl_trigger_after_wait:
            raise ValueError("pdl_trigger_early and pdl_trigger_after_wait are exclusive")
        self.use_linear_beta = use_linear_beta
        # GEMM1 gathers activation rows through the permuted->expanded map;
        # GEMM2 reads the already-permuted GEMM1 output rows contiguously.
        self.gather_rows = self.is_situ
        # Row operand through the TMA unit instead of the cp.async gather
        # warps: default for the contiguous rows of GEMM2 at wide tiles (tile
        # load); GEMM1's gathered rows default to LDGSTS because ``gather4``
        # moves 128-byte rows at only ~6-11 ns per row per SM (B300).
        if row_tma is None:
            row_tma = n_tile >= 64 and not self.gather_rows
        self.row_tma = bool(row_tma)
        # cp.async gather warps (row passes and SF atoms dealt round-robin).
        # Two warps already saturate the cp.async issue at n64; more warps
        # lower the per-thread register cap (no setmaxnreg here) and slow the
        # epilogue. With the rows on TMA the gather warps only move SF bytes.
        if gather_warps is None:
            gather_warps = 2 if (n_tile >= 64 and not self.row_tma) else 1
        if gather_warps not in (1, 2, 4, 8):
            raise ValueError("gather_warps must be 1, 2, 4 or 8")
        if (n_tile // 4) % gather_warps != 0:
            raise ValueError("gather_warps must divide the n_tile // 4 row passes")
        self.num_gather_warps = int(gather_warps)
        # Weight M-tiles per work item. Every mainloop stage carries ``m_group``
        # 128-row weight tiles that all multiply the same (gathered) token stage,
        # accumulating into ``m_group`` TMEM accumulators: the token operand is
        # fetched once per group instead of once per weight tile, and the
        # weight stream owns a larger share of the smem stages in flight.
        if m_group not in (1, 2, 3, 4):
            raise ValueError("m_group must be 1..4")
        self.m_group = int(m_group)
        # Rows per ``moe_sort`` group. With a per-work-item row-group list the
        # work items address ``n_tile``-row sub-tiles of ``group_rows``-row sort
        # groups; expert and row bound are looked up per sort group.
        if group_rows is None:
            group_rows = n_tile
        if group_rows % n_tile != 0:
            raise ValueError("group_rows must be a multiple of n_tile")
        self.group_rows = int(group_rows)
        self.row_group_ratio = self.group_rows // n_tile
        # SiTU output scales in the tcgen05 block-scaled SFA atom layout
        # (``(32, 4, rows/128, 4, K/128)`` order (2,1,4,0,3)) so a dense
        # contiguous grouped GEMM2 can consume the rows directly, instead of
        # the plain ``(rows, K/32)`` bytes read by the swap GEMM2. For GEMM2
        # the flag means the *input* row scales are in that layout (written
        # by mixed dense / sf_blocked swap GEMM1 tiles).
        self.sf_blocked = bool(sf_blocked)
        self.sf_blocked_read = self.sf_blocked and not self.gather_rows
        # Wide tiles keep the per-column finalize metadata in the scheduler's
        # smem ring (slot held until the tile's stores are done) instead of
        # n_tile register pairs per epilogue thread.
        self.hold_meta = n_tile > 32
        if n_tile > 32 and not meta_in_sched:
            raise ValueError("n_tile > 32 requires meta_in_sched=True")
        if max_ab_stages < 2 or num_acc_stages < 1:
            raise ValueError("need >= 2 mainloop stages and >= 1 accumulator stage")
        self.max_ab_stages = max_ab_stages
        self.num_acc_stages = num_acc_stages
        if num_tile_stages < 2:
            raise ValueError("need >= 2 tile-info stages")
        self.num_tile_stages = num_tile_stages
        # Where the finalize/partial epilogue metadata (expert alpha and, per
        # routed column, the output row and route weight) is resolved: by the
        # scheduler warp into a per-tile smem ring (True) or by the epilogue
        # warps themselves one tile ahead in registers (False).
        self.meta_in_sched = bool(meta_in_sched)
        # Timing attribution only (output invalid): 1 skips the epilogue
        # store/reduce, 2 also skips the TMEM accumulator load, 3 skips the
        # gather warp's row-operand cp.async copies (barriers still cycle).
        self.perf_probe = perf_probe
        # Optional L2 cache policy (``createpolicy`` encoding) for the weight
        # and weight-scale TMA loads; weights stream once per row group.
        self.weight_l2_hint = weight_l2_hint
        self.acc_dtype = cutlass.Float32
        self.cta_group = tcgen05.CtaGroup.ONE
        self.cluster_shape_mn = (1, 1)
        # K is deferred to _setup_attributes
        self.mma_tiler = (128, n_tile, 1)
        self.occupancy = 1

        self.epilog_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_warp_id = 5
        self.sched_warp_id = 6
        self.gather_warp_id = 7
        self.threads_per_warp = 32
        self.threads_per_cta = self.threads_per_warp * (7 + self.num_gather_warps)
        self.threads_wo_sched = self.threads_per_warp * (6 + self.num_gather_warps)
        self.num_epilog_threads = self.threads_per_warp * len(self.epilog_warp_id)

        self.cta_sync_barrier = pipeline.NamedBarrier(
            barrier_id=1, num_threads=self.threads_per_cta
        )
        self.epilog_sync_barrier = UnalignedNamedBarrier(
            barrier_id=2, num_threads=self.num_epilog_threads
        )
        self.tmem_alloc_barrier = UnalignedNamedBarrier(
            barrier_id=3, num_threads=32 * (1 + len(self.epilog_warp_id))
        )
        self.sched_sync_barrier = UnalignedNamedBarrier(
            barrier_id=4, num_threads=self.threads_per_warp
        )
        self.num_smem_capacity = utils.get_smem_capacity_in_bytes("sm_100")

    # ------------------------------------------------------------------
    # Static configuration
    # ------------------------------------------------------------------
    def _setup_attributes(self):
        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        # SF atoms cover 32 MN rows x (4 K blocks); the row-operand SF smem tile
        # keeps the 128-row atom layout the S2T copy expects, the gather warp
        # fills rows [0, n_tile) of atom 0 and the MMA reads at the TMEM base.
        self.mma_inst_shape_mn_sfb = (self.mma_inst_shape_mn[0], 128)

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )
        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        self.mma_inst_tile_k = self.k_blocks_per_stage
        self.mma_tiler = (
            self.mma_tiler[0],
            self.mma_tiler[1],
            mma_inst_shape_k * self.mma_inst_tile_k,
        )
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cta_tile_shape_mnk_sfb = self.mma_tiler_sfb
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)), (tiled_mma.thr_id.shape,)
        )
        # One TMEM load of up to 32 columns per subtile, like the reference kernels.
        self.epi_tile = (self.mma_tiler[0], min(32, self.n_tile))

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_tile_stage,
        ) = self._compute_stages(
            tiled_mma,
            tiled_mma_sfb,
            self.mma_tiler,
            self.smem_alloc_a_dtype,
            self.smem_alloc_b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            self.num_smem_capacity,
            self.epilogue_smem_bytes(),
            self.max_ab_stages,
            self.num_acc_stages,
            self._ring_smem_bytes(),
            self.m_group,
        )
        # Tile-info depth bounds how many tiles the producer side may run
        # ahead of the epilogue; short-K tiles need more than 2.
        self.num_tile_stage = self.num_tile_stages
        # Metadata ring depth: one slot per tile-info stage when the scheduler
        # warp fills it, a single unused slot otherwise.
        self.num_meta_stage = self.num_tile_stage if self.meta_in_sched else 1

        # A / SFA stages hold ``m_group`` weight tiles each (slot = stage * m_group + j).
        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.smem_alloc_a_dtype,
            self.num_ab_stage * self.m_group,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.smem_alloc_b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage * self.m_group,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, self.num_ab_stage
        )

        sf_atom_mn = 32
        self.num_sfa_tmem_cols_per_tile = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * self.mma_inst_tile_k
        self.num_sfa_tmem_cols = self.num_sfa_tmem_cols_per_tile * self.m_group
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * self.mma_inst_tile_k
        # Two SF buffers so the S2T copy for stage k+1 can be issued before the
        # A tile of stage k+1 lands (it only needs SFA/SFB, which arrive early).
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_acc_tmem_cols_per_tile = self.cta_tile_shape_mnk[1]
        self.num_accumulator_tmem_cols = (
            self.num_acc_tmem_cols_per_tile * self.num_acc_stage * self.m_group
        )
        total = self.num_accumulator_tmem_cols + self.num_sf_tmem_cols
        alloc = 32
        while alloc < total:
            alloc *= 2
        if alloc > 512:
            raise ValueError(
                f"TMEM budget exceeded: {total} columns (acc {self.num_accumulator_tmem_cols}"
                f" + SF {self.num_sf_tmem_cols}); reduce k_blocks_per_stage or m_group"
            )
        self.num_tmem_alloc_cols = alloc

    def _ring_smem_bytes(self) -> int:
        """Tile-info ring plus the scheduler-filled metadata ring (sTok/sScale)."""
        meta_stages = self.num_tile_stages if self.meta_in_sched else 1
        return 7 * 4 * self.num_tile_stages + (2 * self.n_tile + 1) * 4 * meta_stages

    def epilogue_smem_bytes(self) -> int:
        # SharedStorage carries the SiTU gate exchange (64 x n F32) for every
        # epilogue kind; finalize/partial write straight from registers.
        n = min(self.n_tile, 32)
        red = 128 * self.n_tile * 4 if self.cluster_split else 16
        return 64 * (n + 1) * 4 + red

    @staticmethod
    def _compute_stages(
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        num_smem_capacity: int,
        epilogue_bytes: int,
        max_ab_stages: int,
        num_acc_stage: int,
        ring_bytes: int = 0,
        m_group: int = 1,
    ) -> Tuple[int, int, int]:
        num_tile_stage = 2
        a_one = sm100_utils.make_smem_layout_a(tiled_mma, mma_tiler_mnk, a_dtype, 1)
        b_one = sm100_utils.make_smem_layout_b(tiled_mma, mma_tiler_mnk, b_dtype, 1)
        sfa_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, mma_tiler_mnk, sf_vec_size, 1
        )
        sfb_tiler = (mma_tiler_mnk[0], 128, mma_tiler_mnk[2])
        sfb_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, sfb_tiler, sf_vec_size, 1
        )
        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_one) * m_group
            + cute.size_in_bytes(b_dtype, b_one)
            + cute.size_in_bytes(sf_dtype, sfa_one) * m_group
            + cute.size_in_bytes(sf_dtype, sfb_one)
        )
        # Barriers (2 KB), the tile-info / metadata rings, 1 KB alignment
        # padding for each of the six aligned buffers and 2 KB slack: the
        # launch fails if the struct exceeds the SM capacity, so this reserve
        # is deliberately generous.
        reserved_bytes = 2048 + ring_bytes + epilogue_bytes + 6 * 1024 + 2048
        num_ab_stage = (num_smem_capacity - reserved_bytes) // ab_bytes_per_stage
        num_ab_stage = min(int(num_ab_stage), max_ab_stages)
        if num_ab_stage < 2:
            raise ValueError("not enough shared memory for two mainloop stages")
        return num_acc_stage, num_ab_stage, num_tile_stage

    @staticmethod
    def _compute_grid(
        num_m_tiles: int,
        num_row_groups: int,
        max_active_clusters: cutlass.Constexpr,
    ):
        tile_sched_params = utils.PersistentTileSchedulerParams(
            (num_m_tiles, num_row_groups, 1), (1, 1, 1), raster_along_m=True
        )
        grid = utils.StaticPersistentTileScheduler.get_grid_shape(
            tile_sched_params, max_active_clusters
        )
        return tile_sched_params, grid

    # ------------------------------------------------------------------
    # Host-side launch
    # ------------------------------------------------------------------
    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b: cute.Tensor,
        sfa: cute.Tensor,
        sfb: cute.Tensor,
        out: cute.Tensor,
        out_sf: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_row_group: Optional[cute.Tensor],
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        situ_beta: Optional[cute.Tensor],
        situ_linear_beta: Optional[cute.Tensor],
        zero_buf: Optional[cute.Tensor],
        permuted_idx_to_token_idx: Optional[cute.Tensor],
        top_k: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        """Launch the kernel.

        :param a: weights ``(rows_w, K, L)`` K-major (E2M1); ``L`` = local experts
        :param b: row operand ``(rows_b, K, 1)`` K-major (E4M3): the unpermuted
            activations ``(T, K)`` for ``situ_mxfp8`` (rows gathered through
            ``permuted_idx_to_expanded_idx``), the permuted GEMM1 output
            ``(R, K)`` otherwise
        :param sfa: A scale factors in the MMA atom layout of ``a``'s shape
        :param sfb: B scale factors as plain ``(rows_b, K/32)`` UE8M0 bytes
        :param out: ``situ_mxfp8``: ``(R, rows_w/2, 1)`` E4M3; ``finalize``:
            ``(T, rows_w, 1)`` BF16 (zero-initialised); ``partial``:
            ``(R, rows_w, 1)`` BF16 rows in permuted order (deferred
            finalize: ``alpha * acc`` without the route weight)
        :param out_sf: ``situ_mxfp8`` only: plain ``(R, rows_w/64)`` UInt8 scales
        :param tile_idx_to_expert_idx: expert per row group ``(G,)``
        :param tile_idx_to_mn_limit: exclusive valid permuted-row bound per group
        :param num_non_exiting_tiles: ``(1,)`` number of valid row groups
        :param tile_idx_to_row_group: optional ``(G,)`` int32 row group per work
            item (dual-width dispatch: a work list over a subset of the sorted
            row groups); ``None`` means work item i is row group i
        :param alpha: per-expert scale ``(L,)`` f32
        :param permuted_idx_to_expanded_idx: ``(R,)`` int32 (all kinds)
        :param token_final_scales: ``(T, topK)`` route weights (finalize)
        :param situ_beta / situ_linear_beta: ``(1,)`` or ``(L,)`` f32 runtime
            SiTU parameters (situ_mxfp8); a size-1 tensor is broadcast
        :param zero_buf: optional flat Int64 buffer zero-filled by the
            scheduler warps (the finalize output for the following GEMM2)
        :param permuted_idx_to_token_idx: ``(R,)`` int32 token row per permuted
            row, ``rows_b`` (out of range -> TMA zero-fill) for padding rows;
            required by the TMA row operand of ``situ_mxfp8``
        :param top_k: routed experts per token (expanded idx = token*top_k+k)
        """
        self.a_dtype: Type[cutlass.Numeric] = a.element_type
        self.b_dtype: Type[cutlass.Numeric] = b.element_type
        self.sf_dtype: Type[cutlass.Numeric] = sfa.element_type
        self.out_dtype: Type[cutlass.Numeric] = out.element_type
        # ``a`` may carry a tile-major hierarchical layout (see ``wrapper``);
        # both forms are K-major for the MMA.
        self.a_major_mode = utils.LayoutEnum.ROW_MAJOR.mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(b).mma_major_mode()
        self.needs_unpack = self.a_dtype.width < 8 or self.b_dtype.width < 8
        self.smem_alloc_a_dtype = (
            cutlass.Int8 if self.a_dtype.width < 8 else self.a_dtype
        )
        self.smem_alloc_b_dtype = (
            cutlass.Int8 if self.b_dtype.width < 8 else self.b_dtype
        )
        self.top_k = top_k
        self.beta_broadcast = True
        self.linear_beta_broadcast = True
        if cutlass.const_expr(situ_beta is not None):
            self.beta_broadcast = situ_beta.shape[0] == 1
        if cutlass.const_expr(situ_linear_beta is not None):
            self.linear_beta_broadcast = situ_linear_beta.shape[0] == 1

        self._setup_attributes()

        a_flat_shape = (
            cute.size(a.shape[0]),
            cute.size(a.shape[1]),
            cute.size(a.shape[2]),
        )
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(
            a_flat_shape, self.sf_vec_size
        )
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)
        # The grid covers every row-group slot; slots >= num_non_exiting_tiles
        # exit through the scheduler warp.
        num_row_groups = tile_idx_to_expert_idx.shape[0]
        if cutlass.const_expr(tile_idx_to_row_group is not None):
            num_row_groups = tile_idx_to_row_group.shape[0]

        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            self.mma_inst_shape_mn_sfb,
        )

        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_a, tma_tensor_a = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=(self.smem_alloc_a_dtype if self.a_dtype.width < 8 else None),
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
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        self.num_tma_load_bytes = (a_copy_size + sfa_copy_size) * self.m_group

        # Row operand through TMA: gather4 over the unpermuted activations
        # (GEMM1) or a plain tile load of the permuted rows (GEMM2). Both share
        # the A/SFA stage barrier (transaction bytes added below).
        tma_atom_b = None
        tma_tensor_b = None
        coord_b = None
        if cutlass.const_expr(self.row_tma):
            b_smem_layout = cute.slice_(
                self.b_smem_layout_staged, (None, None, None, 0)
            )
            if cutlass.const_expr(self.gather_rows):
                if cutlass.const_expr(permuted_idx_to_token_idx is None):
                    raise ValueError(
                        "situ_mxfp8 with the TMA row operand needs permuted_idx_to_token_idx"
                    )
                tma_atom_b, tma_tensor_b, coord_b = self._make_gather4_tma_atom_b(
                    b, b_smem_layout, tiled_mma, permuted_idx_to_token_idx
                )
            else:
                b_op = sm100_utils.cluster_shape_to_tma_atom_B(
                    self.cluster_shape_mn, tiled_mma.thr_id
                )
                tma_atom_b, tma_tensor_b = cute.nvgpu.make_tiled_tma_atom_B(
                    b_op,
                    b,
                    b_smem_layout,
                    self.mma_tiler,
                    tiled_mma,
                    self.cluster_layout_vmnk.shape,
                )
            self.num_tma_load_bytes += cute.size_in_bytes(self.b_dtype, b_smem_layout)

        num_m_tiles = cute.ceil_div(a.shape[0], self.cta_tile_shape_mnk[0])
        # Work items are (weight M-tile chunk, row group).
        num_m_chunks = cute.ceil_div(num_m_tiles, self.m_group)
        self.tile_sched_params, grid = self._compute_grid(
            num_m_chunks * self.split_k, num_row_groups, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        n_tile = self.n_tile
        # Transposed staging for the GEMM2 epilogues: (n_tile, 128 + pad) BF16
        # Gate exchange for the SiTU epilogue: (64, n_tile) F32
        epi_n = min(32, n_tile)
        # Row stride padded to epi_n + 1 words: thread t owns row t, so an
        # unpadded stride of 32 words put every lane of a warp in one bank
        # (ncu: 18-30-way conflicts on the gate exchange).
        self.exch_smem_layout = cute.make_layout((64, epi_n), stride=(epi_n + 1, 1))

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 7 * self.num_tile_stage], 1
            ]
            # Per-tile epilogue metadata filled by the scheduler warp: output
            # row per column, route weight per column and the expert alpha
            # (slot n_tile of sScale).
            sTok: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, self.n_tile * self.num_meta_stage],
                16,
            ]
            sScale: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, (self.n_tile + 1) * self.num_meta_stage
                ],
                16,
            ]
            ab_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            b_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_acc_stage * 2]
            tile_info_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_tile_stage * 2
            ]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_a_dtype,
                    cute.cosize(self.a_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.smem_alloc_b_dtype,
                    cute.cosize(self.b_smem_layout_staged.outer),
                ],
                self.buffer_align_bytes,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfa_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(self.sfb_smem_layout_staged)
                ],
                self.buffer_align_bytes,
            ]
            sExch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 64 * (min(n_tile, 32) + 1)], 16
            ]
            # Cluster split-K: the peer's partial accumulator (column-major,
            # thread t owns row t) and its full / empty mbarriers.
            sRed: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32, 128 * n_tile if self.cluster_split else 4
                ],
                16,
            ]
            red_full_mbar: cutlass.Int64
            red_empty_mbar: cutlass.Int64

        self.shared_storage = SharedStorage

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            tma_atom_a,
            tma_tensor_a,
            tma_atom_sfa,
            tma_tensor_sfa,
            b,
            sfb,
            tma_atom_b,
            tma_tensor_b,
            coord_b,
            out,
            out_sf,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            tile_idx_to_row_group,
            alpha,
            permuted_idx_to_expanded_idx,
            token_final_scales,
            situ_beta,
            situ_linear_beta,
            zero_buf,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.exch_smem_layout,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 2 if self.cluster_split else 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _make_gather4_tma_atom_b(
        self,
        b: cute.Tensor,
        b_smem_layout: cute.ComposedLayout,
        tiled_mma: cute.TiledMma,
        row_index: cute.Tensor,
    ):
        """``tile::gather4`` TMA atom for the row operand of GEMM1.

        The data tensor is the unpermuted ``(rows_b, K)`` activation matrix;
        the per-permuted-row token index drives the four gather coordinates of
        every instruction, so the returned TMA tensor is indexed by permuted
        row (``(R, K)``) like the index tensor. Mirrors the B projection of
        ``make_tiled_tma_atom_B`` (the smem layout is the MMA operand layout).
        """
        b_2d = cute.make_tensor(
            b.iterator,
            cute.make_ordered_layout((b.shape[0], b.shape[1]), order=(1, 0)),
        )
        coord = cute.make_tensor(
            row_index.iterator,
            cute.make_layout((row_index.shape[0], b.shape[1]), stride=(1, 0)),
        )
        op = cpasync.CopyBulkTensor2DGather4G2SOp(cta_group=self.cta_group)
        ident = cute.make_identity_layout(b_2d.shape)
        g_tile = cute.composition(ident, (self.mma_tiler[1], self.mma_tiler[2]))
        cta_v_map = tiled_mma._thrfrg_B(g_tile)
        cta_v_map = cute.get(cta_v_map, mode=[1])
        cta_v_map = cute.dice(cta_v_map, (1, (1,) * cute.rank(g_tile)))
        smem_ir = (
            b_smem_layout.value if hasattr(b_smem_layout, "value") else b_smem_layout
        )
        res = _cute_nvgpu_ir.atom_make_non_exec_2d_gather4_tma_load(
            b_2d.value,
            coord.layout,
            smem_ir,
            cta_v_map,
            op._to_ir(),
            num_multicast=1,
        )
        atom = cute.CopyAtom(op, cpasync.CopyBulkTensorTileG2SNonExecTrait(res[0]))
        return atom, res[1], coord

    def mainloop_s2t_copy_and_partition(
        self, sSF: cute.Tensor, tSF: cute.Tensor
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)
        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group), self.sf_dtype
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)
        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)
        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    def epilog_tmem_copy_and_partition(
        self,
        tidx: cutlass.Int32,
        tAcc: cute.Tensor,
        gC: cute.Tensor,
        epi_tile: cute.Tile,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """TMEM -> RF: one thread owns one accumulator row, 32 columns per load.

        ``gC`` is any (CTA_M, CTA_N) row-major tensor; only its layout is used to
        derive the register fragment shape (as in the reference kernels).
        """
        copy_atom_t2r = sm100_utils.get_tmem_load_op(
            self.cta_tile_shape_mnk,
            utils.LayoutEnum.ROW_MAJOR,
            cutlass.Float32,
            self.acc_dtype,
            epi_tile,
            False,
        )
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N, STAGE)
        tAcc_epi = cute.flat_divide(tAcc[((None, None), 0, 0, None)], epi_tile)
        tiled_copy_t2r = tcgen05.make_tmem_copy(
            copy_atom_t2r, tAcc_epi[(None, None, 0, 0, 0)]
        )
        thr_copy_t2r = tiled_copy_t2r.get_slice(tidx)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N, STAGE)
        tTR_tAcc = thr_copy_t2r.partition_S(tAcc_epi)
        # (EPI_TILE_M, EPI_TILE_N, EPI_M, EPI_N)
        gC_epi = cute.flat_divide(gC, epi_tile)
        # (T2R, T2R_M, T2R_N, EPI_M, EPI_N)
        tTR_gC = thr_copy_t2r.partition_D(gC_epi)
        tTR_rAcc = cute.make_rmem_tensor(
            tTR_gC[(None, None, None, 0, 0)].shape, self.acc_dtype
        )
        return tiled_copy_t2r, tTR_tAcc, tTR_rAcc

    # ------------------------------------------------------------------
    # Device kernel
    # ------------------------------------------------------------------
    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        tma_atom_a: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        mB: cute.Tensor,
        mSFB: cute.Tensor,
        tma_atom_b: Optional[cute.CopyAtom],
        mB_tma: Optional[cute.Tensor],
        mIdx: Optional[cute.Tensor],
        out: cute.Tensor,
        out_sf: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
        tile_idx_to_row_group: Optional[cute.Tensor],
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        situ_beta: Optional[cute.Tensor],
        situ_linear_beta: Optional[cute.Tensor],
        zero_buf: Optional[cute.Tensor],
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        exch_smem_layout: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if cutlass.const_expr(self.pdl_trigger_early):
            griddepcontrol_launch_dependents()
        n_tile = self.n_tile
        hold_meta = self.hold_meta

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            if cutlass.const_expr(self.row_tma):
                cpasync.prefetch_descriptor(tma_atom_b)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        # Cluster split-K groups CTAs only for the partial exchange: every
        # TMA / pipeline coordinate keeps the single-CTA (1, 1) layout.
        red_rank = cutlass.Int32(0)
        if cutlass.const_expr(self.cluster_split):
            red_rank = cta_rank_in_cluster
            cta_rank_in_cluster = cutlass.Int32(0)
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        ab_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
            tx_count=self.num_tma_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        # Row operand + its scale factors: cp.async lanes of the gather warps -> MMA.
        b_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp * self.num_gather_warps
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_epilog_threads
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
        )
        tile_info_pipeline = pipeline.PipelineAsync.create(
            barrier_storage=storage.tile_info_mbar_ptr.data_ptr(),
            num_stages=self.num_tile_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_wo_sched
            ),
        )

        if cutlass.const_expr(self.cluster_split):
            # Leader: ``red_full`` (one arrive.expect_tx per split item, the
            # peer's bytes complete it); peer: ``red_empty`` (one remote
            # arrive per consumed item). Both are visible cluster-wide before
            # any remote access.
            if tidx == 0:
                cute.arch.mbarrier_init(storage.red_full_mbar.ptr, 1)
                cute.arch.mbarrier_init(storage.red_empty_mbar.ptr, 1)
            cute.arch.mbarrier_init_fence()

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_alloc_barrier,
            allocator_warp_id=self.epilog_warp_id[0],
            is_two_cta=False,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
        )
        tmem.allocate(self.num_tmem_alloc_cols)

        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer, swizzle=a_smem_layout_staged.inner
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer, swizzle=b_smem_layout_staged.inner
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sExch = storage.sExch.get_tensor(exch_smem_layout)
        sRed = storage.sRed.get_tensor(
            cute.make_layout(
                (128, n_tile if self.cluster_split else 1),
                stride=(1, 128),
            )
        )
        info_layout = cute.make_layout((7, self.num_tile_stage), stride=(1, 7))
        sInfo = storage.sInfo.get_tensor(info_layout)
        sTok = storage.sTok.get_tensor(
            cute.make_layout((n_tile, self.num_meta_stage), stride=(1, n_tile))
        )
        sScale = storage.sScale.get_tensor(
            cute.make_layout((n_tile + 1, self.num_meta_stage), stride=(1, n_tile + 1))
        )

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))
        num_m_tiles = cutlass.Int32(cute.size(gA_mkl, mode=[2]))
        m_group = self.m_group

        thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
        tCgA = thr_mma.partition_A(gA_mkl)
        tCgSFA = thr_mma.partition_A(gSFA_mkl)

        a_cta_layout = cute.make_layout(
            cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
        )
        tAsA, tAgA = cpasync.tma_partition(
            tma_atom_a,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sA, 0, 3),
            cute.group_modes(tCgA, 0, 3),
        )
        tAsSFA, tAgSFA = cpasync.tma_partition(
            tma_atom_sfa,
            block_in_cluster_coord_vmnk[2],
            a_cta_layout,
            cute.group_modes(sSFA, 0, 3),
            cute.group_modes(tCgSFA, 0, 3),
        )
        tAsSFA = cute.filter_zeros(tAsSFA)
        tAgSFA = cute.filter_zeros(tAgSFA)

        # Row operand through TMA (see ``row_tma``): gather4 tiles are indexed
        # by (row group, K tile) of the permuted-row index tensor; the plain
        # tile load follows the A partitioning with the N-mode projection.
        tBsB = None
        tBgB = None
        tBgI = None
        if cutlass.const_expr(self.row_tma):
            if cutlass.const_expr(self.gather_rows):
                b_tiler = (self.mma_tiler[1], self.mma_tiler[2])
                gB_tma = cute.zipped_divide(mB_tma, b_tiler)
                gI_tma = cute.zipped_divide(mIdx, b_tiler)
                tBsB, tBgB, tBgI = cpasync.tma_partition(
                    tma_atom_b,
                    0,
                    cute.make_layout(1),
                    cute.group_modes(sB, 0, 3),
                    [gB_tma, gI_tma],
                )
            else:
                gB_nkl = cute.local_tile(
                    mB_tma,
                    cute.slice_(self.mma_tiler, (0, None, None)),
                    (None, None, None),
                )
                tCgB = thr_mma.partition_B(gB_nkl)
                b_cta_layout = cute.make_layout(
                    cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
                )
                tBsB, tBgB = cpasync.tma_partition(
                    tma_atom_b,
                    block_in_cluster_coord_vmnk[1],
                    b_cta_layout,
                    cute.group_modes(sB, 0, 3),
                    cute.group_modes(tCgB, 0, 3),
                )

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage * self.m_group)
        )

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        tile_info_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_tile_stage
        )
        # Programmatic dependent launch: the prologue above (allocation, barrier
        # init, descriptor prefetch) does not depend on the predecessor grid;
        # wait here, before the first read of a routing output.
        if cutlass.const_expr(self.late_dep_wait):
            # Only the row operand is the predecessor's output: its loaders
            # wait, the scheduler / TMA / MMA / epilogue warps start at once.
            reads_rows = (warp_idx >= self.gather_warp_id) & (
                warp_idx < self.gather_warp_id + self.num_gather_warps
            )
            if cutlass.const_expr(self.row_tma and not self.gather_rows):
                reads_rows = reads_rows | (warp_idx == self.tma_warp_id)
            if reads_rows:
                griddepcontrol_wait()
        else:
            griddepcontrol_wait()
            if cutlass.const_expr(self.pdl_trigger_after_wait):
                griddepcontrol_launch_dependents()
        num_valid_groups = num_non_exiting_tiles[0]
        # Split-K decision (grid-uniform): valid work items vs the CTA budget.
        sk_cnt_split = k_tile_cnt // self.split_k
        sk_do_split = cutlass.Boolean(0)
        sk_m_chunks = (num_m_tiles + m_group - 1) // m_group
        if cutlass.const_expr(self.split_k > 1):
            sk_do_split = (num_valid_groups * sk_m_chunks) <= cutlass.Int32(
                self.split_max_items
            )
        if cutlass.const_expr(self.cluster_split):
            # Grid-uniform: an unsplit launch never touches its peer CTA, so
            # only split launches pay the cluster barrier (the mbarrier init
            # above is then cluster-visible before any remote access).
            if sk_do_split:
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

        # First tile before the CTA-wide sync so consumers can start immediately.
        if warp_idx == self.sched_warp_id:
            if work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                if cutlass.const_expr(self.split_k > 1):
                    # Linear item index over the (m_chunks * split_k, groups) raster.
                    # Unsplit: items 0..m_chunks*valid_groups-1 keep the original
                    # (chunk, group) raster (no idle CTAs); split: split is fastest.
                    sk_lin = cur[1] * (sk_m_chunks * self.split_k) + cur[0]
                    sk_group = sk_lin // sk_m_chunks
                    sk_chunk = sk_lin - sk_group * sk_m_chunks
                    sk_cnt = k_tile_cnt
                    sk_begin = cutlass.Int32(0)
                    if sk_do_split:
                        sk_group = cur[1]
                        sk_chunk = cur[0] // self.split_k
                        sk_cnt = sk_cnt_split
                        sk_begin = (cur[0] - sk_chunk * self.split_k) * sk_cnt_split
                else:
                    sk_group = cur[1]
                    sk_chunk = cur[0]
                    sk_cnt = k_tile_cnt
                    sk_begin = cutlass.Int32(0)
                if sk_group < num_valid_groups:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    sched_row_group = sk_group
                    sched_lookup = sk_group
                    if cutlass.const_expr(tile_idx_to_row_group is not None):
                        sched_row_group = tile_idx_to_row_group[sk_group]
                        sched_lookup = sched_row_group // self.row_group_ratio
                    expert_idx = tile_idx_to_expert_idx[sched_lookup]
                    mn_limit = tile_idx_to_mn_limit[sched_lookup]
                    if cutlass.const_expr(self.meta_in_sched):
                        # Epilogue metadata for this tile: the scheduler warp
                        # runs tiles ahead, so the dependent route lookups
                        # never sit on the epilogue's per-tile critical path.
                        meta_stage = tile_info_producer_state.index
                        sched_lane = tidx % self.threads_per_warp
                        if sched_lane == 0:
                            sScale[(n_tile, meta_stage)] = cutlass.Float32(
                                alpha[expert_idx]
                            )
                        if cutlass.const_expr(not self.is_situ):
                            for mq in cutlass.range_constexpr((n_tile + 31) // 32):
                                meta_col = mq * 32 + sched_lane
                                if meta_col < n_tile:
                                    meta_prow = sched_row_group * n_tile + meta_col
                                    if cutlass.const_expr(self.is_finalize):
                                        meta_valid = meta_prow < mn_limit
                                        meta_expanded = permuted_idx_to_expanded_idx[
                                            meta_prow
                                        ]
                                        meta_safe = cutlass.max(
                                            meta_expanded, cutlass.Int32(0)
                                        )
                                        meta_token = meta_safe // self.top_k
                                        meta_topk = meta_safe % self.top_k
                                        meta_gather = meta_token * cutlass.Int32(
                                            meta_valid
                                        )
                                        sScale[(meta_col, meta_stage)] = (
                                            cutlass.Float32(
                                                token_final_scales[
                                                    (meta_gather, meta_topk)
                                                ]
                                            )
                                        )
                                        sTok[(meta_col, meta_stage)] = meta_token
                                    else:
                                        # Deferred output: the permuted row itself.
                                        sScale[(meta_col, meta_stage)] = (
                                            cutlass.Float32(1.0)
                                        )
                                        sTok[(meta_col, meta_stage)] = meta_prow
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = sk_chunk
                        sInfo[(1, tile_info_producer_state.index)] = sched_row_group
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = sk_begin
                        sInfo[(6, tile_info_producer_state.index)] = sk_cnt
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        self.cta_sync_barrier.arrive_and_wait()

        # Zero-fill the finalize output for the following GEMM2 (grid-strided,
        # scheduler warp of every CTA) so no separate fill kernel is needed.
        if cutlass.const_expr(zero_buf is not None):
            if warp_idx == self.sched_warp_id:
                grid_x, grid_y, grid_z = cute.arch.grid_dim()
                # Linear CTA id / count: the persistent grid is (1, 1, #CTAs).
                cta_linear = bidx + grid_x * (bidy + grid_y * bidz)
                num_ctas = grid_x * grid_y * grid_z
                zero_lane = tidx % self.threads_per_warp
                zero_words = cute.size(zero_buf)
                for zi in cutlass.range(
                    cta_linear * self.threads_per_warp + zero_lane,
                    zero_words,
                    num_ctas * self.threads_per_warp,
                ):
                    zero_buf[zi] = cutlass.Int64(0)

        #
        # Scheduler warp: row groups >= num_valid_groups carry no work. Because
        # the raster order is m-fastest, once an invalid group is reached every
        # later work item is invalid as well.
        #
        if warp_idx == self.sched_warp_id:
            is_continue = cutlass.Boolean(1)
            while work_tile.is_valid_tile and is_continue:
                cur = work_tile.tile_idx
                if cutlass.const_expr(self.split_k > 1):
                    # Linear item index over the (m_chunks * split_k, groups) raster.
                    # Unsplit: items 0..m_chunks*valid_groups-1 keep the original
                    # (chunk, group) raster (no idle CTAs); split: split is fastest.
                    sk_lin = cur[1] * (sk_m_chunks * self.split_k) + cur[0]
                    sk_group = sk_lin // sk_m_chunks
                    sk_chunk = sk_lin - sk_group * sk_m_chunks
                    sk_cnt = k_tile_cnt
                    sk_begin = cutlass.Int32(0)
                    if sk_do_split:
                        sk_group = cur[1]
                        sk_chunk = cur[0] // self.split_k
                        sk_cnt = sk_cnt_split
                        sk_begin = (cur[0] - sk_chunk * self.split_k) * sk_cnt_split
                else:
                    sk_group = cur[1]
                    sk_chunk = cur[0]
                    sk_cnt = k_tile_cnt
                    sk_begin = cutlass.Int32(0)
                if sk_group < num_valid_groups:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    sched_row_group = sk_group
                    sched_lookup = sk_group
                    if cutlass.const_expr(tile_idx_to_row_group is not None):
                        sched_row_group = tile_idx_to_row_group[sk_group]
                        sched_lookup = sched_row_group // self.row_group_ratio
                    expert_idx = tile_idx_to_expert_idx[sched_lookup]
                    mn_limit = tile_idx_to_mn_limit[sched_lookup]
                    if cutlass.const_expr(self.meta_in_sched):
                        # Epilogue metadata for this tile: the scheduler warp
                        # runs tiles ahead, so the dependent route lookups
                        # never sit on the epilogue's per-tile critical path.
                        meta_stage = tile_info_producer_state.index
                        sched_lane = tidx % self.threads_per_warp
                        if sched_lane == 0:
                            sScale[(n_tile, meta_stage)] = cutlass.Float32(
                                alpha[expert_idx]
                            )
                        if cutlass.const_expr(not self.is_situ):
                            for mq in cutlass.range_constexpr((n_tile + 31) // 32):
                                meta_col = mq * 32 + sched_lane
                                if meta_col < n_tile:
                                    meta_prow = sched_row_group * n_tile + meta_col
                                    if cutlass.const_expr(self.is_finalize):
                                        meta_valid = meta_prow < mn_limit
                                        meta_expanded = permuted_idx_to_expanded_idx[
                                            meta_prow
                                        ]
                                        meta_safe = cutlass.max(
                                            meta_expanded, cutlass.Int32(0)
                                        )
                                        meta_token = meta_safe // self.top_k
                                        meta_topk = meta_safe % self.top_k
                                        meta_gather = meta_token * cutlass.Int32(
                                            meta_valid
                                        )
                                        sScale[(meta_col, meta_stage)] = (
                                            cutlass.Float32(
                                                token_final_scales[
                                                    (meta_gather, meta_topk)
                                                ]
                                            )
                                        )
                                        sTok[(meta_col, meta_stage)] = meta_token
                                    else:
                                        # Deferred output: the permuted row itself.
                                        sScale[(meta_col, meta_stage)] = (
                                            cutlass.Float32(1.0)
                                        )
                                        sTok[(meta_col, meta_stage)] = meta_prow
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = sk_chunk
                        sInfo[(1, tile_info_producer_state.index)] = sched_row_group
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                        sInfo[(5, tile_info_producer_state.index)] = sk_begin
                        sInfo[(6, tile_info_producer_state.index)] = sk_cnt
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                else:
                    is_continue = cutlass.Boolean(0)
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

            tile_info_pipeline.producer_acquire(tile_info_producer_state)
            with cute.arch.elect_one():
                sInfo[(0, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(1, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(2, tile_info_producer_state.index)] = cutlass.Int32(-1)
                sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(4, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(5, tile_info_producer_state.index)] = cutlass.Int32(0)
                sInfo[(6, tile_info_producer_state.index)] = cutlass.Int32(0)
            cute.arch.fence_proxy("async.shared", space="cta")
            self.sched_sync_barrier.arrive_and_wait()
            tile_info_pipeline.producer_commit(tile_info_producer_state)
            tile_info_producer_state.advance()
            tile_info_pipeline.producer_tail(tile_info_producer_state)

        #
        # TMA producer warp
        #
        if warp_idx == self.tma_warp_id:
            ab_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((7,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(7):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                m_chunk = tile_info[0]
                row_group = tile_info[1]
                expert_idx = tile_info[2]
                # ((atom_v, rest_v), loopK) per weight tile of the chunk; tiles
                # past the end are clamped to the last valid one (their
                # accumulators are never stored).
                m_tile_0 = cutlass.min(m_chunk * m_group, num_m_tiles - 1)
                tAgA_s0 = tAgA[(None, m_tile_0, None, expert_idx)]
                tAgSFA_s0 = tAgSFA[(None, m_tile_0, None, expert_idx)]
                tAgA_s1 = tAgA_s0
                tAgSFA_s1 = tAgSFA_s0
                tAgA_s2 = tAgA_s0
                tAgSFA_s2 = tAgSFA_s0
                tAgA_s3 = tAgA_s0
                tAgSFA_s3 = tAgSFA_s0
                if cutlass.const_expr(m_group > 1):
                    m_tile_1 = cutlass.min(m_chunk * m_group + 1, num_m_tiles - 1)
                    tAgA_s1 = tAgA[(None, m_tile_1, None, expert_idx)]
                    tAgSFA_s1 = tAgSFA[(None, m_tile_1, None, expert_idx)]
                if cutlass.const_expr(m_group > 2):
                    m_tile_2 = cutlass.min(m_chunk * m_group + 2, num_m_tiles - 1)
                    tAgA_s2 = tAgA[(None, m_tile_2, None, expert_idx)]
                    tAgSFA_s2 = tAgSFA[(None, m_tile_2, None, expert_idx)]
                if cutlass.const_expr(m_group > 3):
                    m_tile_3 = cutlass.min(m_chunk * m_group + 3, num_m_tiles - 1)
                    tAgA_s3 = tAgA[(None, m_tile_3, None, expert_idx)]
                    tAgSFA_s3 = tAgSFA[(None, m_tile_3, None, expert_idx)]
                tBgB_slice = None
                if cutlass.const_expr(self.row_tma and not self.gather_rows):
                    tBgB_slice = tBgB[(None, row_group, None, 0)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < tile_info[6]:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                for k_tile in cutlass.range(0, tile_info[6], 1, unroll=1):  # noqa: B007
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    for jt in cutlass.range_constexpr(m_group):
                        slot_t = ab_producer_state.index * m_group + jt
                        if cutlass.const_expr(jt == 0):
                            tAgA_sj = tAgA_s0
                            tAgSFA_sj = tAgSFA_s0
                        elif cutlass.const_expr(jt == 1):
                            tAgA_sj = tAgA_s1
                            tAgSFA_sj = tAgSFA_s1
                        elif cutlass.const_expr(jt == 2):
                            tAgA_sj = tAgA_s2
                            tAgSFA_sj = tAgSFA_s2
                        else:
                            tAgA_sj = tAgA_s3
                            tAgSFA_sj = tAgSFA_s3
                        tAgA_k = tAgA_sj[(None, tile_info[5] + ab_producer_state.count)]
                        tAgSFA_k = tAgSFA_sj[(None, tile_info[5] + ab_producer_state.count)]
                        tAsA_pipe = tAsA[(None, slot_t)]
                        tAsSFA_pipe = tAsSFA[(None, slot_t)]
                        if cutlass.const_expr(self.weight_l2_hint is not None):
                            w_policy = cutlass.Int64(self.weight_l2_hint)
                            cute.copy(
                                tma_atom_a,
                                tAgA_k,
                                tAsA_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=w_policy,
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_k,
                                tAsSFA_pipe,
                                tma_bar_ptr=tma_bar,
                                cache_policy=w_policy,
                            )
                        else:
                            cute.copy(
                                tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar
                            )
                            cute.copy(
                                tma_atom_sfa,
                                tAgSFA_k,
                                tAsSFA_pipe,
                                tma_bar_ptr=tma_bar,
                            )
                    if cutlass.const_expr(self.row_tma):
                        tBsB_pipe = tBsB[(None, ab_producer_state.index)]
                        if cutlass.const_expr(self.gather_rows):
                            b_crd = (None, (row_group, tile_info[5] + ab_producer_state.count))
                            cute.copy(
                                tma_atom_b,
                                [tBgB[b_crd], tBgI[b_crd]],
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                            )
                        else:
                            cute.copy(
                                tma_atom_b,
                                tBgB_slice[(None, tile_info[5] + ab_producer_state.count)],
                                tBsB_pipe,
                                tma_bar_ptr=tma_bar,
                            )
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < tile_info[6]:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(7):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Gather warps: row operand + scale factors via cp.async
        #
        num_gather = self.num_gather_warps
        is_gather_warp = (warp_idx >= self.gather_warp_id) & (
            warp_idx < self.gather_warp_id + num_gather
        )
        if is_gather_warp:
            gather_sub = warp_idx - self.gather_warp_id
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((7,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(7):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            lane_g = tidx % self.threads_per_warp
            # Row tile in smem: SW128 K-major; per 128-element K atom the
            # logical byte(row, k) = kt*n_tile*128 + row*128 + k (the swizzled
            # smem iterator applies the 128B XOR pattern). SF smem atom (128
            # rows x 4 K blocks = 512 bytes per 128-element K atom): byte
            # (row, kblock) = kt*512 + (row % 32)*16 + (row // 32)*4 + kblock.
            chunk = lane_g % 8
            row_in_pass = lane_g // 8
            n_pass = n_tile // 4  # 4 rows x 8 16-byte chunks per warp pass
            n_sf = max(1, n_tile // 32)  # 4-byte SF copies per lane per k atom
            # Passes and SF atoms are dealt round-robin over the gather warps.
            n_pass_w = (n_pass + num_gather - 1) // num_gather
            n_sf_w = (n_sf + num_gather - 1) // num_gather
            n_kt = self.k_blocks_per_stage // 4  # 128-element K atoms per stage
            k_stage = self.mma_tiler[2]
            b_bytes_per_stage = n_tile * k_stage
            sf_bytes_per_stage = 512 * n_kt
            num_rows_b = mB.shape[0]
            k_cols = mB.shape[1]
            sf_cols = mSFB.shape[1]
            b_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
                self.b_dtype,
                num_bits_per_copy=128,
            )
            sf_atom_copy = cute.make_copy_atom(
                cpasync.CopyG2SOp(), self.sf_dtype, num_bits_per_copy=32
            )
            row_src = cute.make_rmem_tensor((n_pass_w,), cutlass.Int32)
            row_ok = cute.make_rmem_tensor((n_pass_w,), cutlass.Boolean)
            sf_src = cute.make_rmem_tensor((n_sf_w,), cutlass.Int32)
            sf_ok = cute.make_rmem_tensor((n_sf_w,), cutlass.Boolean)
            sf_dst = cute.make_rmem_tensor((n_sf_w,), cutlass.Int32)
            # Predicate-off cp.async copies still zero-fill their destination,
            # so every copy keeps its own slot: padding rows are zeroed in place
            # (the MMA reads them) and SF atoms past n_tile land in the unused
            # tail of the 128-row SF block. Passes are dealt evenly (asserted
            # in the constructor), so no row folds onto another.
            pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

            while is_valid_tile:
                row_group = tile_info[1]
                mn_limit = tile_info[4]
                row_base = row_group * n_tile
                # Source row per (pass, lane): the token row for GEMM1, the
                # permuted row itself for GEMM2. Rows beyond mn_limit or with a
                # garbage expanded index are skipped (their columns are dropped).
                if cutlass.const_expr(not self.row_tma):
                    for i in cutlass.range_constexpr(n_pass_w):
                        p = gather_sub + num_gather * i
                        row = p * 4 + row_in_pass
                        prow = row_base + row
                        ok = prow < mn_limit
                        src_row = prow
                        if cutlass.const_expr(self.gather_rows):
                            safe_row = cutlass.min(prow, row_base + n_tile - 1)
                            expanded = permuted_idx_to_expanded_idx[safe_row]
                            tok = expanded // self.top_k
                            ok = ok & (expanded >= 0) & (tok < num_rows_b)
                            src_row = tok
                        row_src[i] = src_row * cutlass.Int32(ok)
                        row_ok[i] = ok
                for i in cutlass.range_constexpr(n_sf_w):
                    q = gather_sub + num_gather * i
                    srow = q * 32 + lane_g
                    prow = row_base + srow
                    ok = (prow < mn_limit) & (srow < n_tile)
                    src_row = prow
                    if cutlass.const_expr(self.gather_rows):
                        safe_row = cutlass.min(prow, row_base + n_tile - 1)
                        expanded = permuted_idx_to_expanded_idx[safe_row]
                        tok = expanded // self.top_k
                        ok = ok & (expanded >= 0) & (tok < num_rows_b)
                        src_row = tok
                    if cutlass.const_expr(self.sf_blocked_read):
                        # Byte offset of the row inside its 128-row SF block
                        # column ((32, 4, R/128, 4, K/128) order (2,1,4,0,3)):
                        # (row % 32) * 16 + ((row // 32) % 4) * 4, plus
                        # (row // 128) * 512 * (K/128) block rows.
                        src_row = (
                            (src_row % 32) * 16
                            + ((src_row // 32) % 4) * 4
                            + (src_row // 128) * (sf_cols * 128)
                        )
                    sf_src[i] = src_row * cutlass.Int32(ok)
                    sf_ok[i] = ok
                    sf_dst[i] = lane_g * 16 + q * 4

                b_producer_state.reset_count()
                for k_tile in cutlass.range(0, tile_info[6], 1, unroll=1):  # noqa: B007
                    b_pipeline.producer_acquire(b_producer_state)
                    stage = b_producer_state.index
                    k0 = (tile_info[5] + b_producer_state.count) * k_stage
                    sB_stage = sB.iterator + stage * b_bytes_per_stage
                    sSFB_stage = sSFB.iterator + stage * sf_bytes_per_stage
                    if cutlass.const_expr(self.perf_probe != 3):
                        for kt in cutlass.range_constexpr(n_kt):
                            for i in cutlass.range_constexpr(
                                0 if self.row_tma else n_pass_w
                            ):
                                row = (gather_sub + num_gather * i) * 4 + row_in_pass
                                # sB.iterator carries the SW128 swizzle: address the
                                # logical (k atom, row, 16-byte chunk), it applies the XOR.
                                dst_off = kt * n_tile * 128 + row * 128 + chunk * 16
                                src_off = cute.assume(
                                    row_src[i] * k_cols + k0 + kt * 128 + chunk * 16,
                                    divby=16,
                                )
                                g_b = cute.make_tensor(
                                    mB.iterator + src_off,
                                    layout=cute.make_layout((16,)),
                                )
                                s_b = cute.make_tensor(
                                    sB_stage + dst_off, layout=cute.make_layout((16,))
                                )
                                pred1[0] = row_ok[i]
                                cute.copy_atom_call(b_atom_copy, g_b, s_b, pred=pred1)
                            # 4 UE8M0 bytes per row per 128-wide K atom -> SF atom row.
                            for i in cutlass.range_constexpr(n_sf_w):
                                if cutlass.const_expr(self.sf_blocked_read):
                                    # 512-byte SF atom per 128-wide K atom.
                                    sf_src_off = cute.assume(
                                        sf_src[i]
                                        + ((tile_info[5] + b_producer_state.count) * n_kt + kt) * 512,
                                        divby=4,
                                    )
                                else:
                                    sf_src_off = cute.assume(
                                        sf_src[i] * sf_cols
                                        + (tile_info[5] + b_producer_state.count)
                                        * self.k_blocks_per_stage
                                        + kt * 4,
                                        divby=4,
                                    )
                                sf_g = cute.make_tensor(
                                    mSFB.iterator + sf_src_off,
                                    layout=cute.make_layout((4,)),
                                )
                                sf_s = cute.make_tensor(
                                    sSFB_stage + kt * 512 + sf_dst[i],
                                    layout=cute.make_layout((4,)),
                                )
                                pred1[0] = sf_ok[i]
                                cute.copy_atom_call(
                                    sf_atom_copy, sf_g, sf_s, pred=pred1
                                )
                    b_pipeline.producer_commit(b_producer_state)
                    b_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(7):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            b_pipeline.producer_tail(b_producer_state)

        #
        # MMA warp
        #
        if warp_idx == self.mma_warp_id:
            self.tmem_alloc_barrier.arrive_and_wait()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(acc_tmem_ptr, tCtAcc_fake.layout)
            sf_tmem_base = acc_tmem_ptr + self.num_accumulator_tmem_cols
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            # TMEM layout the MMA reads (n_tile rows); the S2T copy fills the
            # full 128-row SF tile and the MMA reads at a per-group offset.
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB_full_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma_sfb,
                self.mma_tiler_sfb,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            # One SFA TMEM region per weight tile of the chunk.
            sfb_tmem_ptr = cute.recast_ptr(
                sf_tmem_base + self.num_sfa_tmem_cols, dtype=self.sf_dtype
            )
            sfa_cols = self.num_sfa_tmem_cols_per_tile
            tCtSFA_0 = cute.make_tensor(
                cute.recast_ptr(sf_tmem_base, dtype=self.sf_dtype), tCtSFA_layout
            )
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_s2t_0,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_0)
            # The S2T tiled copy is built per destination TMEM tensor (the
            # reference kernels never share one across tensors), so keep one
            # tiled copy per weight tile of the chunk.
            tiled_copy_s2t_sfa_0 = tiled_copy_s2t_sfa
            tiled_copy_s2t_sfa_1 = tiled_copy_s2t_sfa
            tiled_copy_s2t_sfa_2 = tiled_copy_s2t_sfa
            tiled_copy_s2t_sfa_3 = tiled_copy_s2t_sfa
            tCtSFA_1 = tCtSFA_0
            tCtSFA_s2t_1 = tCtSFA_s2t_0
            tCtSFA_2 = tCtSFA_0
            tCtSFA_s2t_2 = tCtSFA_s2t_0
            tCtSFA_3 = tCtSFA_0
            tCtSFA_s2t_3 = tCtSFA_s2t_0
            if cutlass.const_expr(m_group > 1):
                tCtSFA_1 = cute.make_tensor(
                    cute.recast_ptr(sf_tmem_base + sfa_cols, dtype=self.sf_dtype),
                    tCtSFA_layout,
                )
                tiled_copy_s2t_sfa_1, _, tCtSFA_s2t_1 = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_1)
                )
            if cutlass.const_expr(m_group > 2):
                tCtSFA_2 = cute.make_tensor(
                    cute.recast_ptr(sf_tmem_base + 2 * sfa_cols, dtype=self.sf_dtype),
                    tCtSFA_layout,
                )
                tiled_copy_s2t_sfa_2, _, tCtSFA_s2t_2 = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_2)
                )
            if cutlass.const_expr(m_group > 3):
                tCtSFA_3 = cute.make_tensor(
                    cute.recast_ptr(sf_tmem_base + 3 * sfa_cols, dtype=self.sf_dtype),
                    tCtSFA_layout,
                )
                tiled_copy_s2t_sfa_3, _, tCtSFA_s2t_3 = (
                    self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA_3)
                )
            tCtSFB_mma = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_full = cute.make_tensor(sfb_tmem_ptr, tCtSFB_full_layout)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB_full)

            ab_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_ab_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((7,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(7):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < tile_info[6]:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                acc_stage_index = acc_producer_state.index

                acc_pipeline.producer_acquire(acc_producer_state)
                tcgen05_fence_after_thread_sync()

                # One mainloop stage: SF S2T copies, then every weight tile of
                # the chunk multiplies the shared token stage into its own
                # accumulator. The first stage is peeled so the ACCUMULATE
                # flag resets each accumulator statically.
                ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                b_pipeline.consumer_wait(b_consumer_state)
                # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads
                cute.arch.fence_proxy("async.shared", space="cta")
                sfb_stage_coord = (None, None, None, None, ab_consumer_state.index)
                cute.copy(
                    tiled_copy_s2t_sfb,
                    tCsSFB_compact_s2t[sfb_stage_coord],
                    tCtSFB_compact_s2t,
                )
                num_kblocks = cute.size(tCrA, mode=[2])
                for jm0_i in cutlass.range_constexpr(m_group):
                    jm0 = jm0_i
                    slot_jm0 = ab_consumer_state.index * m_group + jm0
                    if cutlass.const_expr(jm0 == 0):
                        tCtSFA_sj = tCtSFA_0
                    elif cutlass.const_expr(jm0 == 1):
                        tCtSFA_sj = tCtSFA_1
                    elif cutlass.const_expr(jm0 == 2):
                        tCtSFA_sj = tCtSFA_2
                    else:
                        tCtSFA_sj = tCtSFA_3
                    if cutlass.const_expr(jm0 == 0):
                        cute.copy(
                            tiled_copy_s2t_sfa_0,
                            tCsSFA_compact_s2t[(None, None, None, None, slot_jm0)],
                            tCtSFA_s2t_0,
                        )
                    elif cutlass.const_expr(jm0 == 1):
                        cute.copy(
                            tiled_copy_s2t_sfa_1,
                            tCsSFA_compact_s2t[(None, None, None, None, slot_jm0)],
                            tCtSFA_s2t_1,
                        )
                    elif cutlass.const_expr(jm0 == 2):
                        cute.copy(
                            tiled_copy_s2t_sfa_2,
                            tCsSFA_compact_s2t[(None, None, None, None, slot_jm0)],
                            tCtSFA_s2t_2,
                        )
                    else:
                        cute.copy(
                            tiled_copy_s2t_sfa_3,
                            tCsSFA_compact_s2t[(None, None, None, None, slot_jm0)],
                            tCtSFA_s2t_3,
                        )
                    tCtAcc_j = tCtAcc_base[
                        (None, None, None, acc_stage_index * m_group + jm0)
                    ]
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for kblock_idx in cutlass.range_constexpr(num_kblocks):
                        a_coord = (None, None, kblock_idx, slot_jm0)
                        b_coord = (None, None, kblock_idx, ab_consumer_state.index)
                        sf_kblock_coord = (None, None, kblock_idx)
                        tiled_mma.set(
                            tcgen05.Field.SFA, tCtSFA_sj[sf_kblock_coord].iterator
                        )
                        tiled_mma.set(
                            tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                        )
                        cute.gemm(
                            tiled_mma, tCtAcc_j, tCrA[a_coord], tCrB[b_coord], tCtAcc_j
                        )
                        tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                ab_pipeline.consumer_release(ab_consumer_state)
                ab_consumer_state.advance()
                b_pipeline.consumer_release(b_consumer_state)
                b_consumer_state.advance()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < tile_info[6]:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )
                for k_tile in cutlass.range(1, tile_info[6], 1):  # noqa: B007
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    b_pipeline.consumer_wait(b_consumer_state)
                    # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads
                    cute.arch.fence_proxy("async.shared", space="cta")
                    sfb_stage_coord = (None, None, None, None, ab_consumer_state.index)
                    cute.copy(
                        tiled_copy_s2t_sfb,
                        tCsSFB_compact_s2t[sfb_stage_coord],
                        tCtSFB_compact_s2t,
                    )
                    num_kblocks = cute.size(tCrA, mode=[2])
                    for jm_i in cutlass.range_constexpr(m_group):
                        jm = jm_i
                        slot_jm = ab_consumer_state.index * m_group + jm
                        if cutlass.const_expr(jm == 0):
                            tCtSFA_sj = tCtSFA_0
                        elif cutlass.const_expr(jm == 1):
                            tCtSFA_sj = tCtSFA_1
                        elif cutlass.const_expr(jm == 2):
                            tCtSFA_sj = tCtSFA_2
                        else:
                            tCtSFA_sj = tCtSFA_3
                        if cutlass.const_expr(jm == 0):
                            cute.copy(
                                tiled_copy_s2t_sfa_0,
                                tCsSFA_compact_s2t[(None, None, None, None, slot_jm)],
                                tCtSFA_s2t_0,
                            )
                        elif cutlass.const_expr(jm == 1):
                            cute.copy(
                                tiled_copy_s2t_sfa_1,
                                tCsSFA_compact_s2t[(None, None, None, None, slot_jm)],
                                tCtSFA_s2t_1,
                            )
                        elif cutlass.const_expr(jm == 2):
                            cute.copy(
                                tiled_copy_s2t_sfa_2,
                                tCsSFA_compact_s2t[(None, None, None, None, slot_jm)],
                                tCtSFA_s2t_2,
                            )
                        else:
                            cute.copy(
                                tiled_copy_s2t_sfa_3,
                                tCsSFA_compact_s2t[(None, None, None, None, slot_jm)],
                                tCtSFA_s2t_3,
                            )
                        tCtAcc_j = tCtAcc_base[
                            (None, None, None, acc_stage_index * m_group + jm)
                        ]
                        for kblock_idx in cutlass.range_constexpr(num_kblocks):
                            a_coord = (None, None, kblock_idx, slot_jm)
                            b_coord = (None, None, kblock_idx, ab_consumer_state.index)
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA_sj[sf_kblock_coord].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc_j,
                                tCrA[a_coord],
                                tCrB[b_coord],
                                tCtAcc_j,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    b_pipeline.consumer_release(b_consumer_state)
                    b_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < tile_info[6]:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(7):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            acc_pipeline.producer_tail(acc_producer_state)

        #
        # Epilogue warps
        #
        if warp_idx <= self.epilog_warp_id[-1]:
            self.tmem_alloc_barrier.arrive_and_wait()
            tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            tCtAcc_base = cute.make_tensor(tmem_ptr, tCtAcc_fake.layout)
            epi_tidx = tidx % self.num_epilog_threads
            lane = epi_tidx % self.threads_per_warp
            (
                tiled_copy_t2r,
                tTR_tAcc_base,
                tTR_rAcc,
            ) = self.epilog_tmem_copy_and_partition(
                epi_tidx,
                tCtAcc_base,
                cute.make_tensor(
                    out.iterator,
                    cute.make_layout(
                        (self.cta_tile_shape_mnk[0], self.cta_tile_shape_mnk[1]),
                        stride=(self.cta_tile_shape_mnk[1], 1),
                    ),
                ),
                epi_tile,
            )

            acc_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_acc_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((7,), cutlass.Int32)
            cur_tok = cute.make_rmem_tensor((n_tile,), cutlass.Int32)
            cur_scale = cute.make_rmem_tensor((n_tile,), cutlass.Float32)
            meta_alpha = cute.make_rmem_tensor((1,), cutlass.Float32)
            # Epilogue-side metadata (meta_in_sched=False): the expert alpha
            # and per-column output row / route weight of the tile held in
            # ``tile_info`` are loaded one tile ahead into these registers.
            pf_alpha = cute.make_rmem_tensor((1,), cutlass.Float32)
            pf_tok = cute.make_rmem_tensor((n_tile,), cutlass.Int32)
            pf_scale = cute.make_rmem_tensor((n_tile,), cutlass.Float32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(7):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            if cutlass.const_expr(self.meta_in_sched):
                meta_alpha[0] = sScale[(n_tile, tile_info_consumer_state.index)]
                if cutlass.const_expr(not self.is_situ and not hold_meta):
                    for c in cutlass.range_constexpr(n_tile):
                        cur_tok[c] = sTok[(c, tile_info_consumer_state.index)]
                        cur_scale[c] = sScale[(c, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            if cutlass.const_expr(not hold_meta):
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            if cutlass.const_expr(not self.meta_in_sched):
                pf_alpha[0] = cutlass.Float32(
                    alpha[cutlass.max(tile_info[2], cutlass.Int32(0))]
                )
                if cutlass.const_expr(not self.is_situ):
                    for c in cutlass.range_constexpr(n_tile):
                        pf_prow = tile_info[1] * n_tile + c
                        if cutlass.const_expr(self.is_finalize):
                            pf_valid = pf_prow < tile_info[4]
                            pf_expanded = permuted_idx_to_expanded_idx[pf_prow]
                            pf_safe_idx = cutlass.max(pf_expanded, cutlass.Int32(0))
                            pf_token_idx = pf_safe_idx // self.top_k
                            pf_topk_idx = pf_safe_idx % self.top_k
                            pf_gather_tok = pf_token_idx * cutlass.Int32(pf_valid)
                            pf_scale[c] = cutlass.Float32(
                                token_final_scales[(pf_gather_tok, pf_topk_idx)]
                            )
                            pf_tok[c] = pf_token_idx
                        else:
                            # Deferred output: the permuted row itself.
                            pf_scale[c] = cutlass.Float32(1.0)
                            pf_tok[c] = pf_prow

            inv_fp8_max = cutlass.Float32(1.0 / 448.0)
            # Cluster split-K exchange state. The peer's first ``red_empty``
            # wait passes (parity 1 of a fresh barrier), the leader's first
            # ``red_full`` wait needs the first partial (parity 0).
            cs_full_phase = cutlass.Int32(0)
            cs_empty_phase = cutlass.Int32(1)
            cs_remote_red = cutlass.Int32(0)
            cs_remote_full = cutlass.Int32(0)
            cs_remote_empty = cutlass.Int32(0)
            if cutlass.const_expr(self.cluster_split):
                cs_remote_red = mapa_shared_cluster_u32(
                    storage.sRed.data_ptr(), cutlass.Int32(0)
                ) + cutlass.Int32(4) * epi_tidx
                cs_remote_full = mapa_shared_cluster_u32(
                    storage.red_full_mbar.ptr, cutlass.Int32(0)
                )
                cs_remote_empty = mapa_shared_cluster_u32(
                    storage.red_empty_mbar.ptr, cutlass.Int32(1)
                )

            while is_valid_tile:
                m_chunk = tile_info[0]
                row_group = tile_info[1]
                expert_idx = tile_info[2]
                mn_limit = tile_info[4]
                row_base = row_group * n_tile
                meta_stage = tile_info_consumer_state.index
                if cutlass.const_expr(self.meta_in_sched):
                    alpha_val = meta_alpha[0]
                else:
                    alpha_val = pf_alpha[0]
                    # Keep this tile's prefetched metadata before peeking ahead.
                    if cutlass.const_expr(not self.is_situ):
                        for c in cutlass.range_constexpr(n_tile):
                            cur_tok[c] = pf_tok[c]
                            cur_scale[c] = pf_scale[c]
                    # Peek the next tile and start its metadata loads now so
                    # they overlap this tile's accumulator wait and stores.
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for i in cutlass.range_constexpr(7):
                        tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                    cute.arch.fence_proxy("async.shared", space="cta")
                    tile_info_pipeline.consumer_release(tile_info_consumer_state)
                    tile_info_consumer_state.advance()
                    pf_alpha[0] = cutlass.Float32(
                        alpha[cutlass.max(tile_info[2], cutlass.Int32(0))]
                    )
                    if cutlass.const_expr(not self.is_situ):
                        for c in cutlass.range_constexpr(n_tile):
                            pf_prow = tile_info[1] * n_tile + c
                            if cutlass.const_expr(self.is_finalize):
                                pf_valid = pf_prow < tile_info[4]
                                pf_expanded = permuted_idx_to_expanded_idx[pf_prow]
                                pf_safe_idx = cutlass.max(pf_expanded, cutlass.Int32(0))
                                pf_token_idx = pf_safe_idx // self.top_k
                                pf_topk_idx = pf_safe_idx % self.top_k
                                pf_gather_tok = pf_token_idx * cutlass.Int32(pf_valid)
                                pf_scale[c] = cutlass.Float32(
                                    token_final_scales[(pf_gather_tok, pf_topk_idx)]
                                )
                                pf_tok[c] = pf_token_idx
                            else:
                                # Deferred output: the permuted row itself.
                                pf_scale[c] = cutlass.Float32(1.0)
                                pf_tok[c] = pf_prow

                acc_stage_index = acc_consumer_state.index
                acc_pipeline.consumer_wait(acc_consumer_state)
                tcgen05_fence_after_thread_sync()
                # Every weight tile of the chunk drains its own accumulator; tiles
                # past num_m_tiles (partial last chunk) are skipped uniformly.
                for je in cutlass.range_constexpr(m_group):
                    m_tile = m_chunk * m_group + je
                    acc_slot_e = acc_stage_index * m_group + je
                    if m_tile < num_m_tiles:
                        vals = cute.make_rmem_tensor((n_tile,), cutlass.Float32)
                        epi_n = epi_tile[1]
                        tTR_tAcc = tTR_tAcc_base[
                            (
                                None,
                                None,
                                None,
                                None,
                                None,
                                acc_slot_e,
                            )
                        ]
                        tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                        if cutlass.const_expr(self.perf_probe < 2):
                            for sub in cutlass.range_constexpr(n_tile // epi_n):
                                cute.copy(
                                    tiled_copy_t2r,
                                    tTR_tAcc[(None, None, None, sub)],
                                    tTR_rAcc,
                                )
                                acc_vec = tTR_rAcc.load()
                                for c in cutlass.range_constexpr(epi_n):
                                    vals[sub * epi_n + c] = acc_vec[c] * alpha_val
                        cute.arch.fence_view_async_tmem_load()
                        tcgen05_fence_before_thread_sync()

                        # Cluster split-K: exchange the K halves of a split item.
                        cs_do_store = cutlass.Boolean(1)
                        if cutlass.const_expr(self.cluster_split):
                            cs_split = tile_info[6] < k_tile_cnt
                            if cs_split:
                                if red_rank == 1:
                                    # Peer: wait for the leader to have read the
                                    # previous partial, then ship this one.
                                    cute.arch.mbarrier_wait(
                                        storage.red_empty_mbar.ptr, cs_empty_phase
                                    )
                                    cs_empty_phase = cutlass.Int32(1) - cs_empty_phase
                                    for c in cutlass.range_constexpr(n_tile):
                                        st_async_f32_cluster(
                                            cs_remote_red + cutlass.Int32(4 * 128 * c),
                                            vals[c],
                                            cs_remote_full,
                                        )
                                    cs_do_store = cutlass.Boolean(0)
                                else:
                                    if epi_tidx == 0:
                                        cute.arch.mbarrier_arrive_and_expect_tx(
                                            storage.red_full_mbar.ptr, 128 * n_tile * 4
                                        )
                                    cute.arch.mbarrier_wait(
                                        storage.red_full_mbar.ptr, cs_full_phase
                                    )
                                    cs_full_phase = cutlass.Int32(1) - cs_full_phase
                                    for c in cutlass.range_constexpr(n_tile):
                                        vals[c] = vals[c] + sRed[(epi_tidx, c)]
                                    # Every leader thread has read the buffer.
                                    self.epilog_sync_barrier.arrive_and_wait()
                                    if epi_tidx == 0:
                                        mbarrier_arrive_cluster(cs_remote_empty)
                        if cutlass.const_expr(self.perf_probe == 0):
                            if cutlass.const_expr(self.is_situ):
                                if cs_do_store:
                                    # ---- SiTU + MXFP8 requantization (transposed) ----
                                    beta_idx = cutlass.Int32(0)
                                    if cutlass.const_expr(not self.beta_broadcast):
                                        beta_idx = expert_idx
                                    beta = cutlass.Float32(situ_beta[beta_idx])
                                    is_gate_lane = epi_tidx >= 64
                                    linear_beta = cutlass.Float32(1.0)
                                    inv_linear_beta = cutlass.Float32(1.0)
                                    if cutlass.const_expr(self.use_linear_beta):
                                        lb_idx = cutlass.Int32(0)
                                        if cutlass.const_expr(
                                            not self.linear_beta_broadcast
                                        ):
                                            lb_idx = expert_idx
                                        linear_beta = cutlass.Float32(
                                            situ_linear_beta[lb_idx]
                                        )
                                        inv_linear_beta = cutlass.Float32(1.0) / linear_beta
                                    # act = up_out * gate_out for intermediate j = m_tile*64 + epi_tidx
                                    j = m_tile * 64 + epi_tidx
                                    num_sub = n_tile // epi_n
                                    amax = cute.make_rmem_tensor((epi_n,), cutlass.Float32)
                                    for sub in cutlass.range_constexpr(num_sub):
                                        # One 32-column subtile per gate exchange so the
                                        # exchange buffer is independent of n_tile.
                                        if is_gate_lane:
                                            for c in cutlass.range_constexpr(epi_n):
                                                g = native_situ_f32(
                                                    vals[sub * epi_n + c],
                                                    beta,
                                                    fastmath=True,
                                                )
                                                sExch[(epi_tidx - 64, c)] = g
                                        else:
                                            if cutlass.const_expr(self.use_linear_beta):
                                                for c in cutlass.range_constexpr(epi_n):
                                                    vals[sub * epi_n + c] = (
                                                        linear_beta
                                                        * native_tanh_f32(
                                                            vals[sub * epi_n + c]
                                                            * inv_linear_beta
                                                        )
                                                    )
                                        cute.arch.fence_proxy("async.shared", space="cta")
                                        self.epilog_sync_barrier.arrive_and_wait()
                                        if not is_gate_lane:
                                            for c in cutlass.range_constexpr(epi_n):
                                                v = (
                                                    vals[sub * epi_n + c]
                                                    * sExch[(epi_tidx, c)]
                                                )
                                                vals[sub * epi_n + c] = v
                                                amax[c] = cute.arch.fmax(v, -v)
                                            # One warp == one 32-wide requant group along j.
                                            for c in cutlass.range_constexpr(epi_n):
                                                v = amax[c]
                                                for sh in cutlass.range_constexpr(5):
                                                    v = cute.arch.fmax(
                                                        v,
                                                        cute.arch.shuffle_sync_bfly(
                                                            v, 1 << sh
                                                        ),
                                                    )
                                                amax[c] = v
                                            for c in cutlass.range_constexpr(epi_n):
                                                prow = row_base + sub * epi_n + c
                                                if prow < mn_limit:
                                                    scale_code = float_to_ue8m0_fast(
                                                        amax[c] * inv_fp8_max
                                                    )
                                                    inv_scale = ue8m0_to_inv_scale_fast(
                                                        scale_code
                                                    )
                                                    q = vals[sub * epi_n + c] * inv_scale
                                                    out[(prow, j, 0)] = q.to(self.out_dtype)
                                                    if lane == 0:
                                                        sf_kb = m_tile * 2 + epi_tidx // 32
                                                        if cutlass.const_expr(
                                                            self.sf_blocked
                                                        ):
                                                            out_sf[
                                                                (
                                                                    prow % 32,
                                                                    (prow // 32) % 4,
                                                                    prow // 128,
                                                                    sf_kb % 4,
                                                                    sf_kb // 4,
                                                                    0,
                                                                )
                                                            ] = scale_code.to(cutlass.Uint8)
                                                        else:
                                                            # plain (rows, I/32) scale bytes; j // 32
                                                            out_sf[(prow, sf_kb)] = (
                                                                scale_code.to(cutlass.Uint8)
                                                            )
                                        # The exchange buffer is reused by the next subtile.
                                        self.epilog_sync_barrier.arrive_and_wait()
                            else:
                                # ---- finalize / partial: per-element reduce / store ----
                                # Thread ``epi_tidx`` owns hidden index h for every routed
                                # column. Adjacent lanes pair their BF16 values so one
                                # ``red.global.add.bf16x2`` per lane pair covers a coalesced
                                # 256-byte row segment: no smem staging, no barrier, and
                                # nothing on the TMA unit. Column validity is a predicate,
                                # not a branch, so the unrolled shuffles pipeline.
                                # ``vals`` already carries the expert alpha.
                                h = m_tile * 128 + epi_tidx
                                is_even_lane = (lane % 2) == 0
                                for c in cutlass.range_constexpr(n_tile):
                                    if cutlass.const_expr(hold_meta):
                                        v = vals[c] * sScale[(c, meta_stage)]
                                        tok = sTok[(c, meta_stage)]
                                    else:
                                        v = vals[c] * cur_scale[c]
                                        tok = cur_tok[c]
                                    col_ok = (row_base + c) < mn_limit
                                    if cutlass.const_expr(self.is_finalize):
                                        dst = cute.domain_offset((tok, h, 0), out)
                                        v_hi = cute.arch.shuffle_sync_bfly(v, 1)
                                        red_add_bf16x2_pair_pred(
                                            dst,
                                            v,
                                            v_hi,
                                            cutlass.Int32(col_ok & is_even_lane),
                                        )
                                    elif cutlass.const_expr(self.wide_out):
                                        # Deferred rows past 2^31 elements (MoE-TP
                                        # shard, T >= 16384): 64-bit byte offset
                                        # ``tok * (2 * cols) + 2 * h`` per store.
                                        st_bf16_pred_rowaddr(
                                            out,
                                            tok,
                                            out.shape[1] * 2,
                                            h * 2,
                                            v,
                                            cutlass.Int32(col_ok),
                                        )
                                    else:
                                        dst = cute.domain_offset((tok, h, 0), out)
                                        st_bf16_pred(dst, v, cutlass.Int32(col_ok))

                tcgen05_fence_before_thread_sync()
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                if cutlass.const_expr(self.meta_in_sched):
                    if cutlass.const_expr(hold_meta):
                        # This tile's metadata slot is released only now.
                        cute.arch.fence_proxy("async.shared", space="cta")
                        tile_info_pipeline.consumer_release(tile_info_consumer_state)
                        tile_info_consumer_state.advance()
                    tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                    for i in cutlass.range_constexpr(7):
                        tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                    meta_alpha[0] = sScale[(n_tile, tile_info_consumer_state.index)]
                    if cutlass.const_expr(not self.is_situ and not hold_meta):
                        for c in cutlass.range_constexpr(n_tile):
                            cur_tok[c] = sTok[(c, tile_info_consumer_state.index)]
                            cur_scale[c] = sScale[(c, tile_info_consumer_state.index)]
                    if cutlass.const_expr(not hold_meta):
                        cute.arch.fence_proxy("async.shared", space="cta")
                        tile_info_pipeline.consumer_release(tile_info_consumer_state)
                        tile_info_consumer_state.advance()
                is_valid_tile = tile_info[3] == 1
            if cutlass.const_expr(hold_meta):
                # Release the terminal (invalid) tile-info slot as well.
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        if cutlass.const_expr(
            not (self.pdl_trigger_early or self.pdl_trigger_after_wait)
        ):
            griddepcontrol_launch_dependents()
        if cutlass.const_expr(self.cluster_split):
            if sk_do_split:
                cute.arch.cluster_arrive_relaxed()
                cute.arch.cluster_wait()

    # ------------------------------------------------------------------
    # Raw-pointer wrapper (compiled once per tactic, shapes are runtime)
    # ------------------------------------------------------------------
    @cute.jit
    def wrapper(
        self,
        a_ptr: cute.Pointer,
        b_ptr: cute.Pointer,
        sfa_ptr: cute.Pointer,
        sfb_ptr: cute.Pointer,
        out_ptr: cute.Pointer,
        out_sf_ptr: Optional[cute.Pointer],
        tile_expert_ptr: cute.Pointer,
        tile_limit_ptr: cute.Pointer,
        num_tiles_ptr: cute.Pointer,
        alpha_ptr: cute.Pointer,
        permuted_idx_ptr: Optional[cute.Pointer],
        token_scales_ptr: Optional[cute.Pointer],
        beta_ptr: Optional[cute.Pointer],
        linear_beta_ptr: Optional[cute.Pointer],
        zero_ptr: Optional[cute.Pointer],
        row_index_ptr: Optional[cute.Pointer],
        row_group_ptr: Optional[cute.Pointer],
        rows_w: cutlass.Int32,
        k: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        rows_b: cutlass.Int32,
        rows_perm: cutlass.Int32,
        # 32-bit extents: the deferred (``partial``) output holds every
        # permuted row times the hidden size and passes 2^31 elements for the
        # largest prefill shapes; that variant (``wide_out``) forms a 64-bit
        # byte offset per store (``st_bf16_pred_rowaddr``) and the layout math
        # stays 32-bit.
        out_rows: cutlass.Int32,
        out_cols: cutlass.Int32,
        num_tokens: cutlass.Int32,
        group_capacity: cutlass.Int32,
        list_capacity: cutlass.Int32,
        zero_words: cutlass.Int32,
        top_k: cutlass.Constexpr,
        beta_count: cutlass.Constexpr,
        linear_beta_count: cutlass.Constexpr,
        max_active_clusters: cutlass.Constexpr,
        tiled_a: cutlass.Constexpr,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(tiled_a):
            # Tile-major weights: each (128 rows x 128 K) tile is one contiguous
            # 8 KB block, stored (L, rows_w/128, k/128, 128, 128); the TMA box
            # then streams whole DRAM pages instead of 128 x 64 B row pieces.
            m_tiles = rows_w // 128
            k_tiles = k // 128
            a = cute.make_tensor(
                a_ptr,
                layout=cute.make_layout(
                    ((128, m_tiles), (128, k_tiles), num_local_experts),
                    stride=(
                        (128, k_tiles * 16384),
                        (1, 16384),
                        m_tiles * k_tiles * 16384,
                    ),
                ),
            )
        else:
            a = cute.make_tensor(
                a_ptr,
                layout=cute.make_ordered_layout(
                    (rows_w, k, num_local_experts), order=(1, 0, 2)
                ),
            )
        b = cute.make_tensor(
            b_ptr, layout=cute.make_ordered_layout((rows_b, k, 1), order=(1, 0, 2))
        )
        sfa = cute.make_tensor(sfa_ptr, layout=cute.make_layout((1,)))
        sfb = cute.make_tensor(
            sfb_ptr,
            layout=cute.make_ordered_layout((rows_b, k // 32), order=(1, 0)),
        )
        out = cute.make_tensor(
            out_ptr,
            layout=cute.make_ordered_layout((out_rows, out_cols, 1), order=(1, 0, 2)),
        )
        out_sf = None
        if cutlass.const_expr(out_sf_ptr is not None):
            if cutlass.const_expr(self.sf_blocked):
                out_sf = cute.make_tensor(
                    out_sf_ptr,
                    layout=cute.make_ordered_layout(
                        (32, 4, out_rows // 128, 4, out_cols // 128, 1),
                        order=(2, 1, 4, 0, 3, 5),
                    ),
                )
            else:
                out_sf = cute.make_tensor(
                    out_sf_ptr,
                    layout=cute.make_ordered_layout(
                        (out_rows, out_cols // 32), order=(1, 0)
                    ),
                )
        tile_expert = cute.make_tensor(
            tile_expert_ptr, layout=cute.make_layout((group_capacity,))
        )
        tile_limit = cute.make_tensor(
            tile_limit_ptr, layout=cute.make_layout((group_capacity,))
        )
        num_tiles = cute.make_tensor(num_tiles_ptr, layout=cute.make_layout((1,)))
        tile_row_group = None
        if cutlass.const_expr(row_group_ptr is not None):
            tile_row_group = cute.make_tensor(
                row_group_ptr, layout=cute.make_layout((list_capacity,))
            )
        alpha = cute.make_tensor(
            alpha_ptr, layout=cute.make_layout((num_local_experts,))
        )
        permuted_idx = None
        if cutlass.const_expr(permuted_idx_ptr is not None):
            permuted_idx = cute.make_tensor(
                permuted_idx_ptr, layout=cute.make_layout((rows_perm,))
            )
        zero_buf = None
        if cutlass.const_expr(zero_ptr is not None):
            zero_buf = cute.make_tensor(
                zero_ptr, layout=cute.make_layout((zero_words,))
            )
        row_index = None
        if cutlass.const_expr(row_index_ptr is not None):
            row_index = cute.make_tensor(
                row_index_ptr, layout=cute.make_layout((rows_perm,))
            )
        token_scales = None
        if cutlass.const_expr(token_scales_ptr is not None):
            token_scales = cute.make_tensor(
                token_scales_ptr,
                layout=cute.make_ordered_layout((num_tokens, top_k), order=(1, 0)),
            )
        beta = None
        if cutlass.const_expr(beta_ptr is not None):
            beta = cute.make_tensor(beta_ptr, layout=cute.make_layout((beta_count,)))
        linear_beta = None
        if cutlass.const_expr(linear_beta_ptr is not None):
            linear_beta = cute.make_tensor(
                linear_beta_ptr, layout=cute.make_layout((linear_beta_count,))
            )
        self(
            a,
            b,
            sfa,
            sfb,
            out,
            out_sf,
            tile_expert,
            tile_limit,
            num_tiles,
            tile_row_group,
            alpha,
            permuted_idx,
            token_scales,
            beta,
            linear_beta,
            zero_buf,
            row_index,
            top_k,
            max_active_clusters,
            stream,
        )
