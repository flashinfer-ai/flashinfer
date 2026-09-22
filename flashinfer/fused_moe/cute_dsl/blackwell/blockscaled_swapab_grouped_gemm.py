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
128/N_TILE. The row operand is gathered straight from the unpermuted
activations by a dedicated ``cp.async`` warp (GEMM1: rows indexed through
``permuted_idx_to_expanded_idx``; GEMM2: contiguous permuted rows), so no
separate permute kernel is needed. Its UE8M0 scale factors are plain
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
from cutlass.cute.nvgpu import cpasync, tcgen05

from flashinfer.quantization.quantization_cute_dsl_utils import (
    float_to_ue8m0_fast,
    ue8m0_to_inv_scale_fast,
)

from .custom_pipeline import PipelineCpAsyncUmma
from .utils import (
    UnalignedNamedBarrier,
    blk_copy,
    blk_reduce_bf16,
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

    Warp roles: epilogue (0-3), MMA (4), TMA (5), scheduler (6), gather (7).
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
    ):
        if epilogue_kind not in EPILOGUE_KINDS:
            raise ValueError(f"unknown epilogue_kind {epilogue_kind!r}")
        if n_tile not in (8, 16, 32, 64, 128):
            raise ValueError("n_tile must be 8, 16, 32, 64 or 128")
        if k_blocks_per_stage not in (4, 8):
            # The gather warp addresses whole 128-element K atoms (one 128-byte
            # swizzle atom per FP8 row, one 512-byte SF atom per 128 rows).
            # 16 K-blocks per stage failed validation and was slower; not offered.
            raise ValueError("k_blocks_per_stage must be 4 or 8")
        self.sf_vec_size = sf_vec_size
        self.n_tile = n_tile
        self.k_blocks_per_stage = k_blocks_per_stage
        self.epilogue_kind = epilogue_kind
        self.is_situ = epilogue_kind == "situ_mxfp8"
        self.is_finalize = epilogue_kind == "finalize"
        self.is_partial = epilogue_kind == "partial"
        self.enable_pdl = enable_pdl
        self.use_linear_beta = use_linear_beta
        # GEMM1 gathers activation rows through the permuted->expanded map;
        # GEMM2 reads the already-permuted GEMM1 output rows contiguously.
        self.gather_rows = self.is_situ
        if max_ab_stages < 2 or num_acc_stages < 1:
            raise ValueError("need >= 2 mainloop stages and >= 1 accumulator stage")
        self.max_ab_stages = max_ab_stages
        self.num_acc_stages = num_acc_stages
        # Timing attribution only (output invalid): 1 skips the epilogue math,
        # 2 the S2T/MMA issue, 3 the row-operand cp.async copies.
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
        self.threads_per_cta = self.threads_per_warp * 8
        self.threads_wo_sched = self.threads_per_warp * 7
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
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma, self.mma_tiler, self.smem_alloc_a_dtype, self.num_ab_stage
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma, self.mma_tiler, self.smem_alloc_b_dtype, self.num_ab_stage
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma, self.mma_tiler, self.sf_vec_size, self.num_ab_stage
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma_sfb, self.mma_tiler_sfb, self.sf_vec_size, self.num_ab_stage
        )

        sf_atom_mn = 32
        self.num_sfa_tmem_cols = (
            self.cta_tile_shape_mnk[0] // sf_atom_mn
        ) * self.mma_inst_tile_k
        self.num_sfb_tmem_cols = (
            self.cta_tile_shape_mnk_sfb[1] // sf_atom_mn
        ) * self.mma_inst_tile_k
        # Two SF buffers so the S2T copy for stage k+1 can be issued before the
        # A tile of stage k+1 lands (it only needs SFA/SFB, which arrive early).
        self.num_sf_tmem_cols = self.num_sfa_tmem_cols + self.num_sfb_tmem_cols
        self.num_accumulator_tmem_cols = self.cta_tile_shape_mnk[1] * self.num_acc_stage
        total = self.num_accumulator_tmem_cols + self.num_sf_tmem_cols
        alloc = 32
        while alloc < total:
            alloc *= 2
        if alloc > 512:
            raise ValueError(
                f"TMEM budget exceeded: {total} columns (acc {self.num_accumulator_tmem_cols}"
                f" + SF {self.num_sf_tmem_cols}); reduce k_blocks_per_stage"
            )
        self.num_tmem_alloc_cols = alloc

    def epilogue_smem_bytes(self) -> int:
        # SharedStorage carries every epilogue buffer regardless of kind:
        # transpose staging n x (128 + 8) BF16, gate exchange 64 x n F32 and
        # the per-column scale / token index arrays.
        n = self.n_tile
        return n * (128 + 8) * 2 + 64 * n * 4 + n * 8

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
            cute.size_in_bytes(a_dtype, a_one)
            + cute.size_in_bytes(b_dtype, b_one)
            + cute.size_in_bytes(sf_dtype, sfa_one)
            + cute.size_in_bytes(sf_dtype, sfb_one)
        )
        # Barriers/tile info (2 KB) plus 1 KB alignment padding for each of the
        # six aligned buffers and 2 KB slack: the launch fails if the struct
        # exceeds the SM capacity, so this reserve is deliberately generous.
        reserved_bytes = 2048 + epilogue_bytes + 6 * 1024 + 2048
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
        alpha: cute.Tensor,
        permuted_idx_to_expanded_idx: Optional[cute.Tensor],
        token_final_scales: Optional[cute.Tensor],
        situ_beta: Optional[cute.Tensor],
        situ_linear_beta: Optional[cute.Tensor],
        zero_buf: Optional[cute.Tensor],
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
            ``(T*topK, rows_w, 1)`` BF16
        :param out_sf: ``situ_mxfp8`` only: plain ``(R, rows_w/64)`` UInt8 scales
        :param tile_idx_to_expert_idx: expert per row group ``(G,)``
        :param tile_idx_to_mn_limit: exclusive valid permuted-row bound per group
        :param num_non_exiting_tiles: ``(1,)`` number of valid row groups
        :param alpha: per-expert scale ``(L,)`` f32
        :param permuted_idx_to_expanded_idx: ``(R,)`` int32 (all kinds)
        :param token_final_scales: ``(T, topK)`` route weights (finalize)
        :param situ_beta / situ_linear_beta: ``(1,)`` or ``(L,)`` f32 runtime
            SiTU parameters (situ_mxfp8); a size-1 tensor is broadcast
        :param zero_buf: optional flat Int64 buffer zero-filled by the
            scheduler warps (the finalize output for the following GEMM2)
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
        sfa_layout = blockscaled_utils.tile_atom_to_shape_SF(a_flat_shape, self.sf_vec_size)
        sfa = cute.make_tensor(sfa.iterator, sfa_layout)
        # The grid covers every row-group slot; slots >= num_non_exiting_tiles
        # exit through the scheduler warp.
        num_row_groups = tile_idx_to_expert_idx.shape[0]

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
        self.num_tma_load_bytes = a_copy_size + sfa_copy_size

        num_m_tiles = cute.ceil_div(a.shape[0], self.cta_tile_shape_mnk[0])
        self.tile_sched_params, grid = self._compute_grid(
            num_m_tiles, num_row_groups, max_active_clusters
        )

        self.buffer_align_bytes = 1024
        n_tile = self.n_tile
        # Transposed staging for the GEMM2 epilogues: (n_tile, 128 + pad) BF16
        self.c_smem_layout = cute.make_layout((n_tile, 128), stride=(128 + 8, 1))
        # Gate exchange for the SiTU epilogue: (64, n_tile) F32
        self.exch_smem_layout = cute.make_layout((64, n_tile), stride=(n_tile, 1))

        @cute.struct
        class SharedStorage:
            sInfo: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 5 * self.num_tile_stage], 1
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
            sC: cute.struct.Align[
                cute.struct.MemRange[cutlass.BFloat16, n_tile * (128 + 8)],
                128,
            ]
            sExch: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, 64 * n_tile], 16
            ]
            sColScale: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, n_tile], 16
            ]
            sColToken: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, n_tile], 16
            ]

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
            out,
            out_sf,
            tile_idx_to_expert_idx,
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
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
            self.c_smem_layout,
            self.exch_smem_layout,
            self.epi_tile,
            self.tile_sched_params,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(1, 1, 1),
            smem=self.shared_storage.size_in_bytes(),  # type: ignore[attr-defined]
            stream=stream,
            min_blocks_per_mp=1,
            use_pdl=self.enable_pdl,
        )
        return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
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
        out: cute.Tensor,
        out_sf: Optional[cute.Tensor],
        tile_idx_to_expert_idx: cute.Tensor,
        tile_idx_to_mn_limit: cute.Tensor,
        num_non_exiting_tiles: cute.Tensor,
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
        c_smem_layout: cute.Layout,
        exch_smem_layout: cute.Layout,
        epi_tile: cute.Tile,
        tile_sched_params: utils.PersistentTileSchedulerParams,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        n_tile = self.n_tile

        if warp_idx == self.tma_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)

        bidx, bidy, bidz = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
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
        # Row operand + its scale factors: 32 cp.async lanes -> MMA.
        b_pipeline = PipelineCpAsyncUmma.create(
            barrier_storage=storage.b_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.threads_per_warp
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
        sC = storage.sC.get_tensor(c_smem_layout)
        sExch = storage.sExch.get_tensor(exch_smem_layout)
        sColScale = storage.sColScale.get_tensor(cute.make_layout((n_tile,)))
        sColToken = storage.sColToken.get_tensor(cute.make_layout((n_tile,)))
        info_layout = cute.make_layout((5, self.num_tile_stage), stride=(1, 5))
        sInfo = storage.sInfo.get_tensor(info_layout)

        # (bM, bK, loopM, loopK, loopL)
        gA_mkl = cute.local_tile(
            mA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        gSFA_mkl = cute.local_tile(
            mSFA_mkl, cute.slice_(self.mma_tiler, (None, 0, None)), (None, None, None)
        )
        k_tile_cnt = cutlass.Int32(cute.size(gA_mkl, mode=[3]))

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

        tCrA = tiled_mma.make_fragment_A(sA)
        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        tCtAcc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )

        tile_sched = utils.StaticPersistentTileScheduler.create(
            tile_sched_params, cute.arch.block_idx(), cute.arch.grid_dim()
        )
        work_tile = tile_sched.initial_work_tile_info()
        tile_info_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.num_tile_stage
        )
        num_valid_groups = num_non_exiting_tiles[0]

        # First tile before the CTA-wide sync so consumers can start immediately.
        if warp_idx == self.sched_warp_id:
            if work_tile.is_valid_tile:
                cur = work_tile.tile_idx
                if cur[1] < num_valid_groups:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    expert_idx = tile_idx_to_expert_idx[cur[1]]
                    mn_limit = tile_idx_to_mn_limit[cur[1]]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.sched_sync_barrier.arrive_and_wait()
                    tile_info_pipeline.producer_commit(tile_info_producer_state)
                    tile_info_producer_state.advance()
                tile_sched.advance_to_next_work()
                work_tile = tile_sched.get_current_work()

        self.cta_sync_barrier.arrive_and_wait()
        griddepcontrol_wait()

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
                if cur[1] < num_valid_groups:
                    tile_info_pipeline.producer_acquire(tile_info_producer_state)
                    expert_idx = tile_idx_to_expert_idx[cur[1]]
                    mn_limit = tile_idx_to_mn_limit[cur[1]]
                    with cute.arch.elect_one():
                        sInfo[(0, tile_info_producer_state.index)] = cur[0]
                        sInfo[(1, tile_info_producer_state.index)] = cur[1]
                        sInfo[(2, tile_info_producer_state.index)] = expert_idx
                        sInfo[(3, tile_info_producer_state.index)] = cutlass.Int32(1)
                        sInfo[(4, tile_info_producer_state.index)] = mn_limit
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
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                m_tile = tile_info[0]
                row_group = tile_info[1]
                expert_idx = tile_info[2]
                # ((atom_v, rest_v), loopK)
                tAgA_slice = tAgA[(None, m_tile, None, expert_idx)]
                tAgSFA_slice = tAgSFA[(None, m_tile, None, expert_idx)]

                ab_producer_state.reset_count()
                peek_ab_empty_status = cutlass.Boolean(1)
                if ab_producer_state.count < k_tile_cnt:
                    peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                        ab_producer_state
                    )
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    tAgA_k = tAgA_slice[(None, ab_producer_state.count)]
                    tAgSFA_k = tAgSFA_slice[(None, ab_producer_state.count)]
                    tAsA_pipe = tAsA[(None, ab_producer_state.index)]
                    tAsSFA_pipe = tAsSFA[(None, ab_producer_state.index)]
                    tma_bar = ab_pipeline.producer_get_barrier(ab_producer_state)
                    ab_pipeline.producer_acquire(
                        ab_producer_state, peek_ab_empty_status
                    )
                    cute.copy(tma_atom_a, tAgA_k, tAsA_pipe, tma_bar_ptr=tma_bar)
                    cute.copy(tma_atom_sfa, tAgSFA_k, tAsSFA_pipe, tma_bar_ptr=tma_bar)
                    ab_producer_state.advance()
                    peek_ab_empty_status = cutlass.Boolean(1)
                    if ab_producer_state.count < k_tile_cnt:
                        peek_ab_empty_status = ab_pipeline.producer_try_acquire(
                            ab_producer_state
                        )

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()
            ab_pipeline.producer_tail(ab_producer_state)

        #
        # Gather warp: row operand + scale factors via cp.async
        #
        if warp_idx == self.gather_warp_id:
            b_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_ab_stage
            )
            tile_info_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_tile_stage
            )
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            lane_g = tidx % self.threads_per_warp
            # Row tile in smem: SW128 K-major; per 128-element K atom the
            # logical byte(row, k) = kt*n_tile*128 + row*128 + k (the swizzled
            # smem iterator applies the 128B XOR pattern). SF smem atom: byte
            # (row, kblock) = kt*512 + row*16 + kblock (atom 0 of 128 rows).
            chunk = lane_g % 8
            row_in_pass = lane_g // 8
            n_pass = n_tile // 4  # 4 rows x 8 16-byte chunks per warp pass
            n_sf = max(1, n_tile // 32)  # 4-byte SF copies per lane per k atom
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
            row_src = cute.make_rmem_tensor((n_pass,), cutlass.Int32)
            row_ok = cute.make_rmem_tensor((n_pass,), cutlass.Boolean)
            sf_src = cute.make_rmem_tensor((n_sf,), cutlass.Int32)
            sf_ok = cute.make_rmem_tensor((n_sf,), cutlass.Boolean)
            pred1 = cute.make_rmem_tensor(cute.make_layout((1,)), cutlass.Boolean)

            while is_valid_tile:
                row_group = tile_info[1]
                mn_limit = tile_info[4]
                row_base = row_group * n_tile
                # Source row per (pass, lane): the token row for GEMM1, the
                # permuted row itself for GEMM2. Rows beyond mn_limit or with a
                # garbage expanded index are skipped (their columns are dropped).
                for p in cutlass.range_constexpr(n_pass):
                    prow = row_base + p * 4 + row_in_pass
                    ok = prow < mn_limit
                    src_row = prow
                    if cutlass.const_expr(self.gather_rows):
                        expanded = permuted_idx_to_expanded_idx[prow]
                        tok = expanded // self.top_k
                        ok = ok & (expanded >= 0) & (tok < num_rows_b)
                        src_row = tok
                    row_src[p] = src_row * cutlass.Int32(ok)
                    row_ok[p] = ok
                for q in cutlass.range_constexpr(n_sf):
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
                    sf_src[q] = src_row * cutlass.Int32(ok)
                    sf_ok[q] = ok

                b_producer_state.reset_count()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                    b_pipeline.producer_acquire(b_producer_state)
                    stage = b_producer_state.index
                    k0 = b_producer_state.count * k_stage
                    sB_stage = sB.iterator + stage * b_bytes_per_stage
                    sSFB_stage = sSFB.iterator + stage * sf_bytes_per_stage
                    if cutlass.const_expr(True):
                        for kt in cutlass.range_constexpr(n_kt):
                            for p in cutlass.range_constexpr(n_pass):
                                row = p * 4 + row_in_pass
                                # sB.iterator carries the SW128 swizzle: address the
                                # logical (k atom, row, 16-byte chunk), it applies the XOR.
                                dst_off = kt * n_tile * 128 + row * 128 + chunk * 16
                                src_off = cute.assume(
                                    row_src[p] * k_cols + k0 + kt * 128 + chunk * 16,
                                    divby=16,
                                )
                                g_b = cute.make_tensor(
                                    mB.iterator + src_off,
                                    layout=cute.make_layout((16,)),
                                )
                                s_b = cute.make_tensor(
                                    sB_stage + dst_off, layout=cute.make_layout((16,))
                                )
                                pred1[0] = row_ok[p]
                                cute.copy_atom_call(b_atom_copy, g_b, s_b, pred=pred1)
                            # 4 UE8M0 bytes per row per 128-wide K atom -> SF atom row.
                            for q in cutlass.range_constexpr(n_sf):
                                srow = q * 32 + lane_g
                                sf_src_off = cute.assume(
                                    sf_src[q] * sf_cols
                                    + b_producer_state.count * self.k_blocks_per_stage
                                    + kt * 4,
                                    divby=4,
                                )
                                sf_g = cute.make_tensor(
                                    mSFB.iterator + sf_src_off,
                                    layout=cute.make_layout((4,)),
                                )
                                sf_s = cute.make_tensor(
                                    sSFB_stage + kt * 512 + srow * 16,
                                    layout=cute.make_layout((4,)),
                                )
                                pred1[0] = sf_ok[q]
                                cute.copy_atom_call(
                                    sf_atom_copy, sf_g, sf_s, pred=pred1
                                )
                    b_pipeline.producer_commit(b_producer_state)
                    b_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
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
            sfa_tmem_ptr = cute.recast_ptr(sf_tmem_base, dtype=self.sf_dtype)
            sfb_tmem_ptr = cute.recast_ptr(
                sf_tmem_base + self.num_sfa_tmem_cols, dtype=self.sf_dtype
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)
            tCtSFB_mma = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)
            tCtSFB_full = cute.make_tensor(sfb_tmem_ptr, tCtSFB_full_layout)
            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
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
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            while is_valid_tile:
                ab_consumer_state.reset_count()
                peek_ab_full_status = cutlass.Boolean(1)
                if ab_consumer_state.count < k_tile_cnt:
                    peek_ab_full_status = ab_pipeline.consumer_try_wait(
                        ab_consumer_state
                    )

                acc_stage_index = acc_producer_state.index
                tCtAcc = tCtAcc_base[(None, None, None, acc_stage_index)]

                acc_pipeline.producer_acquire(acc_producer_state)
                tcgen05_fence_after_thread_sync()
                tiled_mma.set(tcgen05.Field.ACCUMULATE, False)

                for k_tile in cutlass.range(k_tile_cnt):  # noqa: B007
                    ab_pipeline.consumer_wait(ab_consumer_state, peek_ab_full_status)
                    b_pipeline.consumer_wait(b_consumer_state)
                    # cp.async (generic proxy) writes -> tcgen05 (async proxy) reads
                    cute.arch.fence_proxy("async.shared", space="cta")
                    if cutlass.const_expr(True):
                        s2t_stage_coord = (
                            None,
                            None,
                            None,
                            None,
                            ab_consumer_state.index,
                        )
                        if cutlass.const_expr(True):
                            cute.copy(
                                tiled_copy_s2t_sfa,
                                tCsSFA_compact_s2t[s2t_stage_coord],
                                tCtSFA_compact_s2t,
                            )
                            cute.copy(
                                tiled_copy_s2t_sfb,
                                tCsSFB_compact_s2t[s2t_stage_coord],
                                tCtSFB_compact_s2t,
                            )
                    if cutlass.const_expr(True):
                        num_kblocks = cute.size(tCrA, mode=[2])
                        for kblock_idx in cutlass.range_constexpr(num_kblocks):
                            kblock_coord = (
                                None,
                                None,
                                kblock_idx,
                                ab_consumer_state.index,
                            )
                            sf_kblock_coord = (None, None, kblock_idx)
                            tiled_mma.set(
                                tcgen05.Field.SFA, tCtSFA[sf_kblock_coord].iterator
                            )
                            tiled_mma.set(
                                tcgen05.Field.SFB, tCtSFB_mma[sf_kblock_coord].iterator
                            )
                            cute.gemm(
                                tiled_mma,
                                tCtAcc,
                                tCrA[kblock_coord],
                                tCrB[kblock_coord],
                                tCtAcc,
                            )
                            tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                    ab_pipeline.consumer_release(ab_consumer_state)
                    ab_consumer_state.advance()
                    b_pipeline.consumer_release(b_consumer_state)
                    b_consumer_state.advance()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if ab_consumer_state.count < k_tile_cnt:
                        peek_ab_full_status = ab_pipeline.consumer_try_wait(
                            ab_consumer_state
                        )

                acc_pipeline.producer_commit(acc_producer_state)
                acc_producer_state.advance()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
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
            tile_info = cute.make_rmem_tensor((5,), cutlass.Int32)
            tile_info_pipeline.consumer_wait(tile_info_consumer_state)
            for i in cutlass.range_constexpr(5):
                tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
            is_valid_tile = tile_info[3] == 1
            cute.arch.fence_proxy("async.shared", space="cta")
            tile_info_pipeline.consumer_release(tile_info_consumer_state)
            tile_info_consumer_state.advance()

            inv_fp8_max = cutlass.Float32(1.0 / 448.0)

            while is_valid_tile:
                m_tile = tile_info[0]
                row_group = tile_info[1]
                expert_idx = tile_info[2]
                mn_limit = tile_info[4]
                row_base = row_group * n_tile
                alpha_val = cutlass.Float32(alpha[expert_idx])

                # Column metadata (one thread per token column).
                if cutlass.const_expr(self.is_situ):
                    pass
                else:
                    if epi_tidx < n_tile:
                        prow = row_base + epi_tidx
                        is_valid_col = prow < mn_limit
                        expanded = permuted_idx_to_expanded_idx[prow]
                        safe_idx = cutlass.max(expanded, cutlass.Int32(0))
                        token_idx = safe_idx // self.top_k
                        topk_idx = safe_idx % self.top_k
                        col_scale = alpha_val
                        out_row = safe_idx
                        if cutlass.const_expr(self.is_finalize):
                            gather_tok = token_idx * cutlass.Int32(is_valid_col)
                            route = cutlass.Float32(
                                token_final_scales[(gather_tok, topk_idx)]
                            )
                            col_scale = alpha_val * route
                            out_row = token_idx
                        sColScale[epi_tidx] = col_scale
                        sColToken[epi_tidx] = out_row
                    cute.arch.fence_proxy("async.shared", space="cta")
                    self.epilog_sync_barrier.arrive_and_wait()

                acc_stage_index = acc_consumer_state.index
                acc_pipeline.consumer_wait(acc_consumer_state)
                tcgen05_fence_after_thread_sync()
                vals = cute.make_rmem_tensor((n_tile,), cutlass.Float32)
                epi_n = epi_tile[1]
                tTR_tAcc = tTR_tAcc_base[
                    (None, None, None, None, None, acc_stage_index)
                ]
                tTR_tAcc = cute.group_modes(tTR_tAcc, 3, cute.rank(tTR_tAcc))
                for sub in cutlass.range_constexpr(n_tile // epi_n):
                    cute.copy(
                        tiled_copy_t2r, tTR_tAcc[(None, None, None, sub)], tTR_rAcc
                    )
                    acc_vec = tTR_rAcc.load()
                    for c in cutlass.range_constexpr(epi_n):
                        vals[sub * epi_n + c] = acc_vec[c] * alpha_val
                cute.arch.fence_view_async_tmem_load()
                tcgen05_fence_before_thread_sync()
                acc_pipeline.consumer_release(acc_consumer_state)
                acc_consumer_state.advance()

                if cutlass.const_expr(True):
                    if cutlass.const_expr(self.is_situ):
                        # ---- SiTU + MXFP8 requantization (transposed) ----
                        beta_idx = cutlass.Int32(0)
                        if cutlass.const_expr(not self.beta_broadcast):
                            beta_idx = expert_idx
                        beta = cutlass.Float32(situ_beta[beta_idx])
                        is_gate_lane = epi_tidx >= 64
                        if is_gate_lane:
                            for c in cutlass.range_constexpr(n_tile):
                                g = native_situ_f32(vals[c], beta, fastmath=True)
                                sExch[(epi_tidx - 64, c)] = g
                        else:
                            if cutlass.const_expr(self.use_linear_beta):
                                lb_idx = cutlass.Int32(0)
                                if cutlass.const_expr(not self.linear_beta_broadcast):
                                    lb_idx = expert_idx
                                linear_beta = cutlass.Float32(situ_linear_beta[lb_idx])
                                inv_linear_beta = cutlass.Float32(1.0) / linear_beta
                                for c in cutlass.range_constexpr(n_tile):
                                    vals[c] = linear_beta * native_tanh_f32(
                                        vals[c] * inv_linear_beta
                                    )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if not is_gate_lane:
                            # act = up_out * gate_out for intermediate j = m_tile*64 + epi_tidx
                            amax = cute.make_rmem_tensor((n_tile,), cutlass.Float32)
                            for c in cutlass.range_constexpr(n_tile):
                                vals[c] = vals[c] * sExch[(epi_tidx, c)]
                                amax[c] = cute.arch.fmax(vals[c], -vals[c])
                            # One warp == one 32-wide requant group along j.
                            for c in cutlass.range_constexpr(n_tile):
                                v = amax[c]
                                for sh in cutlass.range_constexpr(5):
                                    v = cute.arch.fmax(
                                        v, cute.arch.shuffle_sync_bfly(v, 1 << sh)
                                    )
                                amax[c] = v
                            j = m_tile * 64 + epi_tidx
                            for c in cutlass.range_constexpr(n_tile):
                                prow = row_base + c
                                if prow < mn_limit:
                                    scale_code = float_to_ue8m0_fast(
                                        amax[c] * inv_fp8_max
                                    )
                                    inv_scale = ue8m0_to_inv_scale_fast(scale_code)
                                    q = vals[c] * inv_scale
                                    out[(prow, j, 0)] = q.to(self.out_dtype)
                                    if lane == 0:
                                        # plain (rows, I/32) scale bytes; j // 32
                                        out_sf[(prow, m_tile * 2 + epi_tidx // 32)] = (
                                            scale_code.to(cutlass.Uint8)
                                        )
                        self.epilog_sync_barrier.arrive_and_wait()
                    else:
                        # ---- finalize / partial: scale, transpose, bulk reduce ----
                        for c in cutlass.range_constexpr(n_tile):
                            sC[(c, epi_tidx)] = (vals[c] * sColScale[c]).to(
                                cutlass.BFloat16
                            )
                        cute.arch.fence_proxy("async.shared", space="cta")
                        self.epilog_sync_barrier.arrive_and_wait()
                        if epi_tidx < n_tile:
                            prow = row_base + epi_tidx
                            if prow < mn_limit:
                                h0 = m_tile * 128
                                dst = cute.domain_offset(
                                    (sColToken[epi_tidx], h0, 0), out
                                )
                                copy_bytes = cutlass.Int32(128 * 2)
                                if cutlass.const_expr(self.is_finalize):
                                    blk_reduce_bf16(
                                        dst, sC[(epi_tidx, None)], copy_bytes
                                    )
                                else:
                                    blk_copy(dst, sC[(epi_tidx, None)], copy_bytes)
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                        self.epilog_sync_barrier.arrive_and_wait()

                tile_info_pipeline.consumer_wait(tile_info_consumer_state)
                for i in cutlass.range_constexpr(5):
                    tile_info[i] = sInfo[(i, tile_info_consumer_state.index)]
                is_valid_tile = tile_info[3] == 1
                cute.arch.fence_proxy("async.shared", space="cta")
                tile_info_pipeline.consumer_release(tile_info_consumer_state)
                tile_info_consumer_state.advance()

            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)

        griddepcontrol_launch_dependents()

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
        rows_w: cutlass.Int32,
        k: cutlass.Int32,
        num_local_experts: cutlass.Int32,
        rows_b: cutlass.Int32,
        rows_perm: cutlass.Int32,
        out_rows: cutlass.Int32,
        out_cols: cutlass.Int32,
        num_tokens: cutlass.Int32,
        group_capacity: cutlass.Int32,
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
            alpha,
            permuted_idx,
            token_scales,
            beta,
            linear_beta,
            zero_buf,
            top_k,
            max_active_clusters,
            stream,
        )
