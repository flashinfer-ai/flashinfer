# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owned Nemotron adaptation of the matching kernel-team drop file.
# Upstream snapshot: 50d583aa0. See README.md for provenance.
# Extracted from research commit 52e7da3f4; GPU parity must be requalified.
"""
Fused FC1/clamped squared ReLU/FC2 MXFP8 kernel for Nemotron on SM103.
"""

from typing import Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

from moe_mxfp8_glu.kernel_mxfp8_glu_fc12 import (
    Sm100SwigluMxfp8Fc12Kernel as _BaseSm100SwigluMxfp8Fc12Kernel,
)

import cutlass
import cutlass.cute as cute

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from .epilogue_mxfp8 import GluMxfp8Epilogue
from .scheduler_group_lookup import GroupedLookupScheduler
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from moe_nvfp4_swapab.custom_ext import GluMxFp8Fc12SchedExtension
from common.host_utils import get_cutedsl_target_arch
from moe_nvfp4_swapab.moe_utils import spin_wait
from src.token_comm import TokenInPullTokenBackPush

# =============================================================================
# Sm100ClampedRelu2Mxfp8Fc12Kernel
# =============================================================================


class Sm100ClampedRelu2Mxfp8Fc12Kernel(_BaseSm100SwigluMxfp8Fc12Kernel):
    token_comm: TokenInPullTokenBackPush
    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # `_compute_stages`.  Bump if `SharedStorage` over-allocates SMEM.
    _SmemMiscBudget = 1024

    # Supported (ab_dtype, sf_vec_size) pairings.
    # MXFP8 → Float8E4M3FN / Float8E5M2 + sf_vec_size=32  (FP8-E8M0 scales, MmaMXF8Op)
    VALID_AB_DTYPE_SF_SIZE: dict = {
        32: (
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ),
    }

    GateUpInterleave: int = 32

    def __init__(self, group_hint: int) -> None:
        # Fixed Nemotron geometry; the public plan validates the device and tensors.
        self.ab_dtype = cutlass.Float8E4M3FN
        self.c_dtype = cutlass.BFloat16
        self.acc_dtype = cutlass.Float32
        self.mma_tiler_mnk = self.mma_tiler = (128, 256, 128)
        self.cluster_shape_mn = (1, 1)
        self.use_2cta_instrs = False
        self.cta_group = tcgen05.CtaGroup.ONE
        self.static_expert_shape = (128, 4096, 768)
        self.force_static_sched = True
        self.clc_bundle_size = None
        self.num_sched_stages: Optional[int] = None
        self.group_hint = group_hint
        self.token_padding_block = 64
        self.sf_padding_block = 128
        self.sf_vec_size = 32
        self.load_balance_mode = "static"
        self.scenario = "2Dx3D"
        self.scheduler_lookup = True
        self.scheduler_single_group = False
        self.fc2_tile_n = 256
        self.fc2_mma_tiler = self.mma_tiler
        self.mixed_fc2_geometry = False
        self.fc2_disjoint_acc = False
        self.fc2_late_release = False
        self.fc2_in_kernel_topk_reduce = False
        self.token_back_by_dispatch = False
        self.epi_flag_batch = (1, 1)
        self.apply_topk_in_fc1 = False
        self.generate_c = False
        self.use_stg_fc1 = False
        self.gate_up_clamp = 16.0
        self.arch = get_cutedsl_target_arch()

        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id, self.tma_a_warp_id, self.tma_b_warp_id = 4, 5, 6
        self.sched_warp_id = 7
        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        self.token_back_warp_id = None
        self.token_back_standalone = False
        self.threads_per_cta = 384
        # IDs 8 and 9 remain reserved for MXFP8 warp-pair absmax exchange.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)
        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _create_fc2_mma(self):
        return sm100_utils.make_blockscaled_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
            self.cta_group,
            (self.mma_tiler[0], self.fc2_tile_n),
        )

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares `mma_tiler_mnk` and SMEM layouts across phases.
        """
        self.dispatch_warp_id = (8, 9, 10, 11)
        num_token_back_warps = 0
        self.threads_per_cta = 32 * (
            len(self.epilogue_warp_id)
            + 1  # mma
            + 1  # tma_a
            + 1  # tma_b
            + 1  # sched
            + len(self.dispatch_warp_id)
            + num_token_back_warps
        )

        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0] // (1),
            cute.round_up(self.mma_inst_shape_mn[1], 128),
        )

        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        # SFB-specific tiler: rounded-up MN; same K as main tiler.
        self.mma_tiler_sfb = (
            self.mma_inst_shape_mn_sfb[0],
            self.mma_inst_shape_mn_sfb[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.cluster_layout_sfb_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma_sfb.thr_id.shape,),
        )

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        # Epilogue is autonomous: it owns all epi-side decisions (overlap,
        # acc stages, subtile dispatch, TMA commit/drain, piggyback red.add).
        # We pass kernel-level params + `allow_overlap_acc` hint and read
        # the decisions back via @property below.
        _epi_common = dict(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
        )
        self.epilogue = GluMxfp8Epilogue(**_epi_common)
        self.epilogue._fc2_tile_n = self.fc2_tile_n
        self.epilogue._hardware_tanh = getattr(self, "hardware_tanh", False)

        self.num_sched_stages = 2

        # sD stages locked to subtile_cnt: one full CTA output tile lives in
        # sD at a time.  Per-subtile producer_acquire/commit go
        # away — only one batched commit + drain happens per task tile (in
        # the epilogue's `run` loop body).
        self.num_d_stage = 4  # One complete 32-token output tile per epilogue warp.
        # fc1 output (fp8 quantised) SMEM — always present.
        d_bytes_total = self.epilogue.bytes_per_stage * self.num_d_stage
        c_bytes_total = 0
        self.num_c_raw_stage = 0

        (
            self.num_acc_stage,
            self.num_a_stage,
            self.num_b_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            c_bytes_total + d_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_a_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_b_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_a_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_b_stage,
        )
        self.fc2_b_smem_layout_staged = self.b_smem_layout_staged
        self.fc2_sfb_smem_layout_staged = self.sfb_smem_layout_staged
        self.d_smem_layout_staged = self.epilogue.staged_smem_layout(
            self.num_d_stage,
        )
        self.c_smem_layout_staged = None

        # Read epilogue's autonomous decisions.
        self.overlapping_accum = self.epilogue.overlapping_accum
        self.num_acc_pipeline_stages = self.epilogue.num_acc_pipeline_stages
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_sfa_tmem_cols = self.epilogue.num_sfa_tmem_cols
        self.num_sfb_tmem_cols = self.epilogue.num_sfb_tmem_cols
        self.num_sf_tmem_cols = self.epilogue.num_sf_tmem_cols
        self.num_accumulator_tmem_cols = self.epilogue.num_accumulator_tmem_cols
        self.iter_acc_early_release_in_epilogue = self.epilogue.iter_acc_early_release

        # TMA load bytes per stage (A + B + SFA + SFB).
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.atom_thr_size = (
            atom_thr_size  # store as Python int for use in @cute.kernel
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        self.num_tma_load_a_bytes = (a_copy_size + sfa_copy_size) * atom_thr_size
        self.num_tma_load_b_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size
        self.fc2_tma_load_b_bytes = (
            cute.size_in_bytes(
                self.b_dtype,
                cute.slice_(self.fc2_b_smem_layout_staged, (None, None, None, 0)),
            )
            + cute.size_in_bytes(
                self.sf_dtype,
                cute.slice_(self.fc2_sfb_smem_layout_staged, (None, None, None, 0)),
            )
        ) * atom_thr_size

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
    ) -> Tuple[int, int, int, int]:
        """Compute stage counts for ACC, A, B, and scheduler."""
        num_acc_stage = 2

        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
            tiled_mma,
            mma_tiler_mnk,
            a_dtype,
            1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma,
            mma_tiler_mnk,
            b_dtype,
            1,
        )
        sfa_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_mnk,
            sf_vec_size,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_staged_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )
        b_bytes_per_stage = cute.size_in_bytes(
            b_dtype, b_smem_layout_staged_one
        ) + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)

        fixed_overhead = self._smem_misc_budget_bytes() + c_bytes_total

        num_ab_stage = (
            smem_capacity // occupancy - fixed_overhead
        ) // ab_bytes_per_stage
        num_a_stage = num_ab_stage
        num_b_stage = num_ab_stage

        smem_per_cta = smem_capacity // occupancy
        unused_smem = smem_per_cta - fixed_overhead - num_ab_stage * ab_bytes_per_stage
        if unused_smem > b_bytes_per_stage:
            num_b_stage = num_b_stage + 1
            unused_smem = unused_smem - b_bytes_per_stage

        return num_acc_stage, num_a_stage, num_b_stage, num_sched_stages

    # ── MegaMoE communication hook stubs ─────────────────────────────────────
    #
    # No-op base implementations for the lean fc1+fc2 path.  Mirroring the
    # identically-named stubs in the NVFP4 base (`kernel_fc12.py`).
    # `Sm100NemotronMegaMoEMxfp8Kernel` (megamoe_kernel_mxfp8.py) overrides all of
    # them to delegate to `src/token_comm.py`.

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self,
        token_comm_args,
        work_tile_info,
    ):
        self.token_comm.fc1_tma_b_predispatch_spin(
            token_comm_args,
            work_tile_info,
        )

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        self.token_comm.dispatch_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        # All role-specific async drains above must complete before exit.
        # Publish each thread's writes before the CTA rendezvous. The next
        # same-stream kernel replaces the grid-wide completion rendezvous.
        cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()

    @cute.jit
    def __call__(
        self,
        # ── fc1 (Linear1) problem tensors ────────────────────────────────
        activation: cute.Tensor,  # (token_sum_padded, hidden)
        fc1_weight: cute.Tensor,  # (experts, hidden, intermediate_gateup)
        activation_sf: cute.Tensor,  # (token_sum_padded_sf, hidden / sf_vec_size)
        fc1_weight_sf: cute.Tensor,  # (experts, intermediate_gateup_padded * hidden / sf_vec_size)
        # ── fc1 workspace consumed as fc2 GEMM-B ─────────────────────────
        fc1_output: cute.Tensor,  # (token_sum_padded, intermediate_downproj)
        fc1_output_sf: cute.Tensor,  # (token_sum_padded_sf, intermediate_downproj / sf_vec_size)
        # ── fc2 (Linear2) problem tensors ────────────────────────────────
        fc2_weight: cute.Tensor,  # (experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,  # (experts, hidden_padded * intermediate_downproj / sf_vec_size)
        fc2_output: cute.Tensor,  # (token_sum_padded, hidden) BFloat16, hidden stride-1
        # ── topk weights (Path A) ────────────────────────────────────────
        topk_scores: cute.Tensor,  # (token_sum_padded,) Float32
        # ── Cross-phase workspace ────────────────────────────────────────
        fc1_done_counter: cute.Tensor,  # (max_token_block_per_rank,) Int32
        # ── Sched / runtime ──────────────────────────────────────────────
        offs: Optional[cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        # ── Optional epi-side scaling ────────────────────────────────────
        norm_const_tensor: Optional[cute.Tensor] = None,
        global_activation_sf: Optional[cute.Tensor] = None,
        global_fc1_weight_sf: Optional[cute.Tensor] = None,
        # ── Optional dynamic load-balance counter ────────────────────────
        load_balance_counter: Optional[cute.Tensor] = None,
        # ── Sizes-mode per-expert token count (MegaMoE path) ─────────────
        # Exactly one of `offs` / `expert_token_sizes` must be non-None.
        # MegaMoE subclass passes a zero-copy `i32 stride=(2,)` view onto
        # `expert_recv_count_sum` filled by the dispatch warps.
        expert_token_sizes: Optional[cute.Tensor] = None,
        token_comm_args=None,
        fc1_c: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch the fused Nemotron MXFP8 expert kernel."""

        # Bind data-tensor shapes to codegen-time expert dims when requested.
        # Strides, token rows, and SF tensors stay runtime-dynamic because they
        # encode host padding/swizzle choices.
        (
            experts_static,
            intermediate_gateup_static,
            hidden_static,
        ) = self.static_expert_shape
        intermediate_downproj_static = intermediate_gateup_static

        fc1_weight = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (experts_static, hidden_static, intermediate_gateup_static),
                stride=fc1_weight.stride,
            ),
        )
        fc2_weight = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (experts_static, intermediate_downproj_static, hidden_static),
                stride=fc2_weight.stride,
            ),
        )
        activation = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (activation.shape[0], hidden_static),
                stride=activation.stride,
            ),
        )
        fc1_output = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (fc1_output.shape[0], intermediate_downproj_static),
                stride=fc1_output.stride,
            ),
        )
        if cutlass.const_expr(len(fc2_output.shape) == 3):
            fc2_output = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                    stride=fc2_output.stride,
                ),
            )
        else:
            fc2_output = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], hidden_static),
                    stride=fc2_output.stride,
                ),
            )

        # ── GEMM-domain transform for fc1 phase (non-swap-AB) ──
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # A_gemm (fc1 activations): (tokens_sum, hidden) -> (M=tokens, K=hidden, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )

        # B_gemm (fc1 weights): (experts, hidden, intermediate_gateup) with hidden stride-1 (K-major)
        # -> (N=intermediate_gateup, K=hidden, L=experts).
        experts, hidden_b, intermediate_gateup = fc1_weight.shape
        fc1_weight_gemm = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (intermediate_gateup, hidden_b, experts),
                stride=(
                    fc1_weight.stride[2],
                    fc1_weight.stride[1],
                    fc1_weight.stride[0],
                ),
            ),
        )

        # D_gemm is a user-view output tensor; epilogue owns its store path.
        intermediate_downproj = fc1_output.shape[1]
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )

        # SFA / SFB scale tensors (atom-tiled) — fc1 phase.
        #   SFA (mma M-side) = activation_sf (activation scales, A-side)
        #   SFB (mma N-side) = fc1_weight_sf (weight scales, B-side)
        tokens_sum_padded = activation_sf.shape[0]
        hidden_padded = activation_sf.shape[1] * self.sf_vec_size
        activation_sf_gemm = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, hidden_padded, 1), self.sf_vec_size
            ),
        )
        intermediate_gateup_padded_mul_hidden_padded = fc1_weight_sf.shape[1]
        intermediate_gateup_padded = (
            intermediate_gateup_padded_mul_hidden_padded * self.sf_vec_size
        ) // hidden_padded
        fc1_weight_sf_gemm = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (intermediate_gateup_padded, hidden_padded, experts),
                self.sf_vec_size,
            ),
        )

        # ── GEMM-domain transform for fc2 phase ──
        #
        # fc2 roles: M=hidden, N=tokens_sum, K=intermediate_downproj.

        # A_gemm (fc2 weights): (experts, intermediate_downproj, hidden)
        # -> (M=hidden, K=intermediate_downproj, L=experts).
        experts2, intermediate_downproj_b2, hidden_b2 = fc2_weight.shape
        fc2_weight_gemm = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (hidden_b2, intermediate_downproj_b2, experts2),
                stride=(
                    fc2_weight.stride[2],
                    fc2_weight.stride[1],
                    fc2_weight.stride[0],
                ),
            ),
        )

        # fc2 phase B operand = fc1 output reused (no new view needed:
        # `fc1_output_gemm` was built from `fc1_output.iterator` with the same
        # (tokens_sum, intermediate_downproj, fake-L=1) layout that fc2's
        # GEMM-B view wants; reuse it directly when wiring fc2 TMA-B atom).

        # C_gemm (fc2 output, BFloat16; STG.256-driven by the epilogue's
        # `Fc2UnpackPermuteStg` — no TMA store path).
        # We still build a GEMM-domain view so `ext.get_gmem_tensor("d", ...)`
        # can apply the per-expert offset.  TMA atom is NOT built for fc2_output.
        # MegaMoE passes a 3D (max_tokens, topk, hidden) combine output; the lean
        # path passes 2D (tokens, hidden).  In both cases build a 3D GEMM-domain
        # view (rows, hidden, fake-L=1) folding the topk axis into the row
        # stride.  In MegaMoE mode the epilogue bypasses this view for STG (it
        # uses Fc2OutputDest from token_comm_args), but get_gmem_tensor("d", ...)
        # is still called, so the layout must be valid.
        if cutlass.const_expr(len(fc2_output.shape) == 3):
            fc2_hidden_out = fc2_output.shape[2]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[2], c0),
                ),
            )
        else:
            fc2_hidden_out = fc2_output.shape[1]
            fc2_output_gemm = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (tokens_sum, fc2_hidden_out, c1),
                    stride=(fc2_output.stride[0], fc2_output.stride[1], c0),
                ),
            )

        # SFA / SFB for fc2:
        #   SFA (mma M-side) = fc2_weight_sf (fc2 weight scales, sf_vec_size)
        #   SFB (mma N-side) = fc1_output_sf (fc1 epilogue SFs, uses sf_vec_size)
        # Both paths produce SFs with self.sf_vec_size:
        fc1_out_sf_vec_size = self.sf_vec_size
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * fc1_out_sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                fc1_out_sf_vec_size,
            ),
        )

        hidden_padded_fc2_mul_intermediate_downproj_padded = fc2_weight_sf.shape[1]
        hidden_padded_fc2 = (
            hidden_padded_fc2_mul_intermediate_downproj_padded * self.sf_vec_size
        ) // intermediate_downproj_padded
        fc2_weight_sf_gemm = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (hidden_padded_fc2, intermediate_downproj_padded, experts2),
                self.sf_vec_size,
            ),
        )

        expert_cnt = experts
        # `intermediate_gateup` (= fc1_weight.shape[2]) is what we pass to the
        # scheduler via `expert_shape`; see `MoESchedulerParamsBase`
        # docstring for the precise contract.
        hidden_dim = hidden

        # ── Infer dtypes and major modes ──
        # Phases share dtypes by construction. For MXFP8: A/B/fc1_output
        # are Float8E4M3FN/Float8E5M2 and SFs are Float8E8M0FNU.
        # `self.fc1_output_dtype` drives the sD SMEM element type and flows
        # into the epilogue ctor as `fc1_output_dtype`.
        self.a_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = activation_sf_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(
            activation_gemm
        ).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(
            fc1_weight_gemm
        ).mma_major_mode()
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)

        self._setup_attributes()
        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        # ── fc1 TMA atoms ──

        # TMA load A1 (= fc1 activations, non-swap-AB: A=activations)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = (
            cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                activation_gemm,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )

        # TMA load B1 (= fc1 weights, non-swap-AB: B=weights)
        b_op = sm100_utils.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc1_weight_gemm,
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )

        # TMA load SFA1 (= activation_sf, non-swap-AB: SFA=activation SFs, A-side)
        sfa_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfa_smem_layout = cute.slice_(
            self.sfa_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc1_activation_sf, tma_tensor_fc1_activation_sf = (
            cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                activation_sf_gemm,
                sfa_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )

        # TMA load SFB1 (= fc1_weight_sf, non-swap-AB: SFB=weight SFs, B-side)
        sfb_op = sm100_utils.cluster_shape_to_tma_atom_SFB(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        sfb_smem_layout = cute.slice_(
            self.sfb_smem_layout_staged, (None, None, None, 0)
        )
        tma_atom_fc1_weight_sf, tma_tensor_fc1_weight_sf = (
            cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                fc1_weight_sf_gemm,
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )

        # TMA store for fc1 MXFP8 output.
        # Per-subtile issue lives in
        # `self.epilogue.tma_store_fc1_output`; commit / drain lives
        # inside the epilogue's `run` loop body.
        fc1_output_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            fc1_output_tma_op,
            fc1_output_gemm,
            self.epilogue.smem_layout_one_stage,
            self.epilogue.epi_tile,
        )

        tma_atom_c = tma_atom_fc1_output
        tma_tensor_c = tma_tensor_fc1_output

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
        # per-thread STG.
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, 1),
                self.sf_vec_size,
            ),
        )

        fc2_mma = self._create_fc2_mma()
        fc2_b_layout = cute.slice_(self.fc2_b_smem_layout_staged, (None, None, None, 0))
        fc2_sfb_layout = cute.slice_(
            self.fc2_sfb_smem_layout_staged, (None, None, None, 0)
        )
        # ── fc2 TMA atoms: fc1_output → A-side (M=tokens), fc2_weight → B-side (N=hidden) ──
        # Non-swap-AB fc2: activation (fc1_output) is GEMM-A (M=tokens),
        # weight (fc2_weight) is GEMM-B (N=hidden). Same SMEM layouts as fc1.

        tma_atom_fc2_activation, tma_tensor_fc2_activation = (
            cute.nvgpu.make_tiled_tma_atom_A(
                a_op,
                fc1_output_gemm,
                a_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc2_weight, tma_tensor_fc2_weight = cute.nvgpu.make_tiled_tma_atom_B(
            b_op,
            fc2_weight_gemm,
            fc2_b_layout,
            self.fc2_mma_tiler,
            fc2_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_fc2_activation_sf, tma_tensor_fc2_activation_sf = (
            cute.nvgpu.make_tiled_tma_atom_A(
                sfa_op,
                fc1_output_sf_gemm_for_fc2_load,
                sfa_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )
        tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = (
            cute.nvgpu.make_tiled_tma_atom_B(
                sfb_op,
                fc2_weight_sf_gemm,
                fc2_sfb_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )

        # ── Scheduler params + grid + launch ──
        #
        # `expert_cnt` / `intermediate_gateup` / `hidden_dim` are
        # extracted from the (possibly rewritten) tensor shapes above:
        #   - static path (`static_expert_shape` bound): they are
        #     codegen-time Python int constants; the new base
        #     `MoESchedulerParamsBase.__init__` preserves the Python
        #     int type and `__extract_mlir_values__` skips them, so
        #     they remain inlined literals across the scheduler's scf
        #     region boundaries (no demotion to iter_arg / kernel-arg).
        #   - dynamic path: they are runtime Int32 from tensor metadata.
        #
        # `expert_shape[1]` carries `intermediate_gateup` semantics
        # (= fc1_weight.shape[2]) per the `MoESchedulerParamsBase.__init__`
        # contract.  The fused fc12 scheduler reads it as fc1 GEMM-M
        # (under swap-AB) and derives `num_fc1_intermediate_blocks`
        # from it.
        # atomic_counter mode requires a host-allocated GMEM Int32 scalar
        # whose pointer lives in scheduler params; static mode passes
        # None (params validate this).  Caller's contract from __call__:
        # `load_balance_counter` is required iff `load_balance_mode ==
        # 'atomic_counter'`; otherwise may be None.
        load_balance_counter_ptr = None

        if cutlass.const_expr((offs is None) == (expert_token_sizes is None)):
            raise ValueError(
                "Exactly one of `offs` / `expert_token_sizes` must be "
                "provided; got "
                f"offs={'set' if offs is not None else 'None'}, "
                f"expert_token_sizes="
                f"{'set' if expert_token_sizes is not None else 'None'}."
            )

        sched_params = MoEFusedFc12SchedulerParams(
            scenario=self.scenario,
            expert_shape=(expert_cnt, intermediate_gateup, hidden_dim),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            load_balance_mode=(self.load_balance_mode),
            load_balance_counter_ptr=load_balance_counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=False,
            expert_token_prefix_sum=offs,
            expert_token_sizes=expert_token_sizes,
        )
        grid = sched_params.get_grid_shape(max_active_clusters)

        self.kernel(
            tiled_mma,
            tiled_mma_sfb,
            # fc1 TMA atoms / tensors (non-swap-AB: A=activations, B=weights)
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_fc1_activation_sf,
            tma_tensor_fc1_activation_sf,
            tma_atom_fc1_weight_sf,
            tma_tensor_fc1_weight_sf,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc2_activation_sf,
            tma_tensor_fc2_activation_sf,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            # GEMM-domain tensors (fc1)
            activation_gemm,
            fc1_weight_gemm,
            fc1_output_gemm,
            activation_sf_gemm,
            fc1_weight_sf_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2)
            fc2_weight_gemm,
            fc2_output_gemm,
            fc2_weight_sf_gemm,
            fc1_output_sf_gemm_for_fc2_load,
            # topk + cross-phase sync workspace
            topk_scores,
            fc1_done_counter,
            # Scheduling
            offs,
            sched_params,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            self.d_smem_layout_staged,
            self.fc2_b_smem_layout_staged,
            self.fc2_sfb_smem_layout_staged,
            self.c_smem_layout_staged,
            tma_atom_c,
            tma_tensor_c,
            token_comm_args,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_activation_1: cute.CopyAtom,
        tma_tensor_fc1_activation_1: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        tma_tensor_weight: cute.Tensor,
        tma_atom_fc1_activation_1_sf: cute.CopyAtom,
        tma_tensor_fc1_activation_1_sf: cute.Tensor,
        tma_atom_fc1_weight_sf: cute.CopyAtom,
        tma_tensor_fc1_weight_sf: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc2_activation_sf: cute.CopyAtom,
        tma_tensor_fc2_activation_sf: cute.Tensor,
        tma_atom_fc2_weight_sf: cute.CopyAtom,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_weight_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_weight_gemm: cute.Tensor,
        fc2_output_gemm: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm_for_fc2_load: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        offs: Optional[cute.Tensor],
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        d_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        fc2_b_smem_layout_staged: cute.ComposedLayout,
        fc2_sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Optional[Union[cute.Layout, cute.ComposedLayout]] = None,
        tma_atom_c: Optional[cute.CopyAtom] = None,
        tma_tensor_c: Optional[cute.Tensor] = None,
        token_comm_args=None,
    ):
        """Persistent Nemotron compute with four epilogue and four dispatch warps.

        The epilogue owns accumulator consumption, TMA drains and FC1 readiness
        publication; the detached tail resets dispatch state after all CTAs exit.
        """

        # fc2 waits for all fc1 intermediate N-tiles in the same token block.
        # Each N-tile is processed by atom_thr_size CTAs (both CTA0 and CTA1 increment
        # the counter), so the threshold must account for both CTAs' contributions.
        ext_fc2_spin_threshold = (
            (fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
            * self.epilogue._atom_thr_size
        )

        ext = GluMxFp8Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
            cluster_m=self.epilogue._atom_thr_size,
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2

        bidx, _, _ = cute.arch.block_idx()
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        block_in_cluster_coord_sfb_vmnk = cluster_layout_sfb_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )
        tidx, _, _ = cute.arch.thread_idx()

        # SharedStorage.
        SchedCls = sched_params.get_scheduler_type()
        SchedCls = GroupedLookupScheduler
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages * 2
            ]
            sched_storage: SchedStorage  # type: ignore[valid-type]  # CuTe-generated struct.
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        # ── Pipelines: separate producer/consumer groups for A and B. ──

        a_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        a_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mcast_ctas_a
        )
        a_producer, a_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.a_full_mbar_ptr.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=a_pipeline_producer_group,
            consumer_group=a_pipeline_consumer_group,
            tx_count=self.num_tma_load_a_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        b_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 1)
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mcast_ctas_b
        )
        b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.fc2_tma_load_b_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        acc_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        num_acc_consumer_threads = (
            len(self.epilogue_warp_id) * 32 * (2 if use_2cta_instrs else 1)
        )
        acc_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_acc_consumer_threads
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full_mbar_ptr.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=acc_pipeline_producer_group,
            consumer_group=acc_pipeline_consumer_group,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        # TMEM allocator
        tmem_alloc_barrier = pipeline.NamedBarrier(
            barrier_id=self.tmem_alloc_sync_bar_id,
            num_threads=32 * len((self.mma_warp_id, *self.epilogue_warp_id)),
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=tmem_alloc_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar_ptr.ptr,
            arch=self.arch,
        )

        # Sched
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.mma_warp_id,
                *self.epilogue_warp_id,
            )
        )
        scheduler = SchedCls.create(
            sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            sched_storage=storage.sched_storage,
            num_consumer_threads=num_sched_consumer_threads,
            ext=ext,
            single_group=self.scheduler_single_group,
        )
        sched_consumer = scheduler.make_consumer()

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offsets latency overlaps with pipeline setup.
        # Under MegaMoE + static load-balance, `internal_init` walks
        # per-expert sizes from `expert_recv_count_sum` -- those are not
        # valid until the dispatch barrier completes, so we defer init to the
        # sched warp (after `token_comm_hook_sched_warp_pre_init_wait`).
        early_internal_init = (self.load_balance_mode == "atomic_counter") or (
            not self.enable_token_comm
        )

        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ── SMEM tensors A / B / SFA / SFB (shared by fc1 / fc2) ──
        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=self.b_dtype,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfa_smem_layout_staged,
            byte_alignment=128,
        )
        sSFB = smem.allocate_tensor(
            element_type=self.sf_dtype,
            layout=sfb_smem_layout_staged,
            byte_alignment=128,
        )

        fc2_mma = self._create_fc2_mma()
        sB2 = cute.make_tensor(sB.iterator, fc2_b_smem_layout_staged.outer)
        sSFB2 = cute.make_tensor(sSFB.iterator, fc2_sfb_smem_layout_staged)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE).  Under
        # `overlapping_accum` the two acc buffers share TMEM with SF
        # columns; we shrink the stage stride so the second buffer
        # starts at `(tile_n - num_sf_tmem_cols)` cols instead of tile_n.
        acc_fake = tiled_mma.make_fragment_C(cute.append(acc_shape, self.num_acc_stage))
        if cutlass.const_expr(self.overlapping_accum):
            acc_fake = cute.make_tensor(
                acc_fake.iterator,
                cute.make_layout(
                    acc_fake.shape,
                    stride=(
                        acc_fake.stride[0],
                        acc_fake.stride[1],
                        acc_fake.stride[2],
                        (self.mma_tiler[1] - self.num_sf_tmem_cols)
                        * acc_fake.stride[0][1],
                    ),
                ),
            )

        # Cluster wait before TMEM alloc.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        mma_tiler_k = self.mma_tiler[2]
        # `fc1_weight_gemm.shape[1]` / `fc2_weight_gemm.shape[1]`
        # both resolve to `hidden` / `intermediate_downproj`.  Under
        # `static_expert_shape` they are codegen-time Python ints
        # (rewritten on `fc1_weight` / `fc2_weight` at `__call__`
        # entry); otherwise they are runtime Int32 from tensor metadata.
        # The arithmetic below folds to an immediate in the static path.
        k_tile_cnt_fc1 = (fc1_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        # fc2 spin threshold: each fc1 N-tile (intermediate direction) is incremented
        # by atom_thr_size CTAs.  In non-swap-AB, fc1_weight_gemm.shape[0] is the N
        # dimension (intermediate_gateup), so divide by cta_tile_shape_mnk[1] (N tile),
        # not cta_tile_shape_mnk[0] (M/token tile).  Matches ext_fc2_spin_threshold.
        fc2_spin_threshold = (
            (fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1)
            // self.cta_tile_shape_mnk[1]
        ) * self.epilogue._atom_thr_size

        if warp_idx == self.sched_warp_id:
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                # Diagnostic stage filtering preserves the original work enumeration.
                # Skipped stages execute no MMA/epilogue; enumeration overhead remains.
                ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                scheduler.publish_work()
                scheduler.gen_next_work()
            # Sentinel publish (current_work is already invalid here).
            scheduler.publish_work()
            scheduler.produce_tail()

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (warps 5 / 6)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads activations/SFA into the A pipeline.
        # TMA-B loads weights/SFB into the B pipeline and waits for
        # fc1 workspace readiness in the fc2 phase.

        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            _iket_active = tidx == cutlass.Int32(160)
            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )

            # non-swap-AB FC1: activation (A) is partitioned per-CTA (like original B).
            # b_cta_layout=(2,) and mcast_mode=1 → each CTA loads its own token range.
            b_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )

            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            sfa_cta_layout = a_cta_layout

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                if is_phase_linear1:
                    # ── fc1 phase A-side (non-swap-AB: A=activations, per-CTA partitioned) ──
                    # Activations are split per CTA (tokens 0-127 for CTA 0, 128-255 for CTA 1).
                    # Use b_cta_layout + m-coord so each CTA loads its own token range.
                    # mcast_mode=1 → same M-coord = only self → no actual multicast. ✓
                    if _iket_active:
                        iket.range_push("tma_weight_fc1")
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args,
                        work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc1_activation",
                        tma_tensor_fc1_activation_1,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc1_activation_sf",
                        tma_tensor_fc1_activation_1_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = a_producer.acquire_and_advance(peek_a_empty_status)
                        peek_a_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_a_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_activation_1,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc1_activation_1_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )
                else:
                    # ── fc2 phase A-side: load fc1_output (M=tokens) + wait for fc1 done ──
                    #
                    # Non-swap-AB fc2: A=fc1_output (M=tokens). tile_m_idx is the
                    # CTA-level token block. Counter wait moved here from TMA-B.
                    if _iket_active:
                        iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count
                        + work_tile_info.tile_m_idx
                        // cutlass.Int32(self.epilogue._atom_thr_size)
                    )
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    # Always spin (no peek shortcut) to guarantee counter=4 in this warp,
                    # then use acquire semantics + cross-proxy fence to ensure fc1_output
                    # writes (from generic proxy) are visible to the TMA async proxy load.
                    if _iket_active:
                        iket.range_push("tma_token_fc2_a_wait")
                    spin_wait(
                        counter_ptr,
                        lambda v: v >= fc2_spin_threshold,
                        fail_sleep_cycles=20,
                    )
                    if _iket_active:
                        iket.range_pop()
                    cute.arch.load(
                        counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu"
                    )
                    cute.arch.fence_proxy("async")
                    cute.arch.fence_proxy("async.global")

                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc2_activation",
                        tma_tensor_fc2_activation,
                        work_tile_info,
                    )
                    real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                        "fc2_activation_sf",
                        tma_tensor_fc2_activation_sf,
                        work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_activation,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc2_activation_sf,
                        block_in_cluster_coord_vmnk[2],
                        sfa_cta_layout,
                        cute.group_modes(sSFA, 0, 3),
                        cute.group_modes(tCgSFA, 0, 3),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)

                    # fc2 A-side = fc1_output (M=tokens). tAgA is cluster-level
                    # indexed, so divide tile_m_idx by cluster_m to get the
                    # cluster block index — same formula as fc1 A-side.
                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    tAgSFA_slice = tAgSFA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = a_producer.acquire_and_advance(peek_a_empty_status)
                        peek_a_empty_status = cutlass.Boolean(1)
                        # Candidate issues the current transfer before probing the
                        # next slot. Keep all acquire/release operations intact.
                        if handle.count + 1 < k_tile_cnt:
                            peek_a_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_activation,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_activation_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfa,
                            mcast_mask=sfa_full_mcast_mask,
                        )

                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            a_producer.tail()

        # ── TMA-B warp (warp 6) ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            _iket_active = tidx == cutlass.Int32(192)
            b_full_mcast_mask = None
            sfb_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
                )
                sfb_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_sfb_vmnk,
                    block_in_cluster_coord_sfb_vmnk,
                    mcast_mode=1,
                )

            # non-swap-AB FC1: weight (B) is multicast (like original A).
            # a_full_mcast_mask with mcast_mode=2 → covers all CTAs with same N-coord = all CTAs.
            a_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )

            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )

                if is_phase_linear1:
                    # ── fc1 phase B-side (fc1_weight, N-side GEMM-B) ──
                    # Weight is expert-indexed (key "a") and N-side: use tile_n_idx
                    # (not tile_m_idx) to select the N-tile. tile_m_idx counts token
                    # blocks and grows with more tokens, causing OOB on the weight's
                    # N-dimension when tile_m_idx >= 2.
                    if _iket_active:
                        iket.range_push("tma_token_fc1")

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc1_weight",
                        tma_tensor_weight,
                        work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc1_weight_sf",
                        tma_tensor_fc1_weight_sf,
                        work_tile_info,
                    )

                    # N-K tiling for N-side weight (N=intermediate, K=hidden).
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    # No coord flip for MXFP8 (non-swap-AB); partition_B for N-side.
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc1_weight_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    # Use tile_n_idx for N-side weight: invariant across token blocks.
                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    tBgSFB_slice = tBgSFB[(None, work_tile_info.tile_n_idx, None, 0)]

                    b_producer.reset()
                    peek_b_empty_status = b_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = b_producer.acquire_and_advance(peek_b_empty_status)
                        peek_b_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_b_empty_status = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_weight,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=a_full_mcast_mask,  # same as A-loading for weights
                        )
                        cute.copy(
                            tma_atom_fc1_weight_sf,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                else:
                    # ── fc2 phase B-side: load fc2_weight (N=hidden), no counter wait ──
                    # fc2_weight is independent of fc1; counter wait is in TMA-A.
                    if _iket_active:
                        iket.range_push("tma_weight_fc2")
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc2_weight",
                        tma_tensor_fc2_weight,
                        work_tile_info,
                    )
                    real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                        "fc2_weight_sf",
                        tma_tensor_fc2_weight_sf,
                        work_tile_info,
                    )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.fc2_mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(
                            (self.mma_tiler_sfb),
                            (0, None, None),
                        ),
                        (None, None, None),
                    )
                    tCgB = fc2_mma.get_slice(mma_tile_coord_v).partition_B(gB_nkl)
                    tCgSFB = (thr_mma_sfb).partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB2, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc2_weight_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB2, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    # fc2 B-side = fc2_weight (N=hidden). tile_n_idx is the
                    # CTA-level hidden block index; invariant across token blocks.
                    fc2_b_hidden_tile = work_tile_info.tile_n_idx
                    tBgB_slice = tBgB[(None, fc2_b_hidden_tile, None, 0)]
                    tBgSFB_slice = tBgSFB[(None, fc2_b_hidden_tile, None, 0)]

                    b_producer.reset()
                    peek_b_empty_status = b_producer.try_acquire()

                    for _k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = b_producer.acquire_and_advance(peek_b_empty_status)
                        peek_b_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_b_empty_status = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_weight,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        cute.copy(
                            tma_atom_fc2_weight_sf,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_sfb,
                            mcast_mask=sfb_full_mcast_mask,
                        )
                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            b_producer.tail()

        # ════════════════════════════════════════════════════════════════════
        # MMA warp (warp 4)
        # ════════════════════════════════════════════════════════════════════
        #
        # Both phases share tiled_mma and TMEM; only K-tile count differs.
        if warp_idx == self.mma_warp_id:
            _iket_active = tidx == cutlass.Int32(128)

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)
            _fc2_acc_fake = fc2_mma.make_fragment_C(
                fc2_mma.partition_shape_C(self.fc2_mma_tiler[:2])
            )  # noqa: F841
            tCrB2 = fc2_mma.make_fragment_B(sB2)  # noqa: F841

            # SFA TMEM tensor (placed after the acc cols).
            sfa_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFA_layout = blockscaled_utils.make_tmem_layout_sfa(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfa_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFA = cute.make_tensor(sfa_tmem_ptr, tCtSFA_layout)

            # SFB TMEM tensor (after acc + SFA cols).
            sfb_tmem_ptr = cute.recast_ptr(
                acc_tmem_ptr + self.num_accumulator_tmem_cols + self.num_sfa_tmem_cols,
                dtype=self.sf_dtype,
            )
            tCtSFB_layout = blockscaled_utils.make_tmem_layout_sfb(
                tiled_mma,
                self.mma_tiler,
                self.sf_vec_size,
                cute.slice_(sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB = cute.make_tensor(sfb_tmem_ptr, tCtSFB_layout)

            (
                tiled_copy_s2t_sfa,
                tCsSFA_compact_s2t,
                tCtSFA_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFA, tCtSFA)
            (
                tiled_copy_s2t_sfb,
                tCsSFB_compact_s2t,
                tCtSFB_compact_s2t,
            ) = self.mainloop_s2t_copy_and_partition(sSFB, tCtSFB)

            fc2_sfb_tmem_layout = blockscaled_utils.make_tmem_layout_sfb(
                fc2_mma,
                self.fc2_mma_tiler,
                self.sf_vec_size,
                cute.slice_(fc2_sfb_smem_layout_staged, (None, None, None, 0)),
            )
            tCtSFB2 = cute.make_tensor(sfb_tmem_ptr, fc2_sfb_tmem_layout)
            fc2_copy_sfb, fc2_src_sfb, fc2_dst_sfb = (
                self.mainloop_s2t_copy_and_partition(sSFB2, tCtSFB2)
            )

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages
            )

            # K-tile counts `k_tile_cnt_fc1` / `k_tile_cnt_fc2` come
            # from the enclosing scope (computed once before the TMA warps).

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = work_tile_info.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                # Prebind k_tile_cnt due to DSL AST.
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    if _iket_active:
                        iket.range_push("mma_fc1")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    if _iket_active:
                        iket.range_push("mma_fc2")

                if cutlass.const_expr(self.overlapping_accum):
                    acc_stage_index = acc_producer_state.phase ^ 1
                else:
                    acc_stage_index = acc_producer_state.index

                if is_leader_cta:
                    tCtAcc = acc_base[(None, None, None, acc_stage_index)]

                    if _iket_active:
                        iket.range_push("mma_ab_wait")
                    a_consumer.reset()
                    b_consumer.reset()
                    peek_a_full_status = cutlass.Boolean(1)
                    peek_b_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_a_full_status = a_consumer.try_wait()
                        peek_b_full_status = b_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)
                    if _iket_active:
                        iket.range_pop()

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle_a = a_consumer.wait_and_advance(peek_a_full_status)
                        handle_b = b_consumer.wait_and_advance(peek_b_full_status)
                        peek_a_full_status = cutlass.Boolean(1)
                        peek_b_full_status = cutlass.Boolean(1)
                        # Experiment: avoid probing the next stage before issuing
                        # the ready tile. The default preserves the original schedule.
                        if handle_a.count + 1 < k_tile_cnt:
                            peek_a_full_status = a_consumer.try_wait()
                            peek_b_full_status = b_consumer.try_wait()

                        cute.copy(
                            tiled_copy_s2t_sfa,
                            tCsSFA_compact_s2t[
                                (None, None, None, None, handle_a.index)
                            ],
                            tCtSFA_compact_s2t,
                        )
                        cute.copy(
                            tiled_copy_s2t_sfb,
                            tCsSFB_compact_s2t[
                                (None, None, None, None, handle_b.index)
                            ],
                            tCtSFB_compact_s2t,
                        )

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[(None, None, None, handle_a.index)], tCtSFA],
                            [tCrB[(None, None, None, handle_b.index)], tCtSFB],
                            tCtAcc,
                        )
                        handle_a.release()
                        handle_b.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                if _iket_active:
                    iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ── sD SMEM (fc1 output staging; fc2 doesn't use it) ──
        # Always allocated for correct typing in const_expr-gated TMA dead branches.
        sD = smem.allocate_tensor(
            element_type=self.fc1_output_dtype,
            layout=d_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=d_smem_layout_staged.inner,
        )

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to `self.epilogue.run(...)` -- the epilogue owns
        # the entire 2-phase task-tile loop.
        if warp_idx < self.mma_warp_id:
            epi_warp_idx = warp_idx

            tmem.allocate(self.num_tmem_alloc_cols)
            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_tensor = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            # acc_tensor = cute.make_tensor(
            #    acc_tmem_ptr,
            #    cute.make_layout(
            #        (((128, self.cta_tile_shape_mnk[1]), 1),),
            #        stride=(((1 << 16, 1), 0),),
            #    ),
            # )

            # Build common kwargs shared by both epilogue flavours.
            _smem_c_raw_arg = None
            _tma_atom_c_arg = None
            _gmem_c_arg = None
            _gmem_fc1_output_arg = tma_tensor_fc1_output
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sD,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=_gmem_fc1_output_arg,
                gmem_fc1_output_sf=fc1_output_sf_gemm,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                smem_c_buffer=_smem_c_raw_arg,
                tma_atom_c=_tma_atom_c_arg,
                gmem_c=_gmem_c_arg,
                warp_idx=epi_warp_idx,
                tidx=tidx,
                alpha=cutlass.Float32(1.0),
                norm_const=cutlass.Float32(1.0),
            )

            self.epilogue.run(**_run_kwargs, token_comm_args=token_comm_args)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            cute.arch.fence_acq_rel_sys()

        if warp_idx >= self.dispatch_warp_id[0]:
            lane_idx_for_dispatch = cute.arch.lane_idx()
            self.token_comm_hook_dispatch_warp_body(
                token_comm_args,
                token_comm_storage,
                warp_idx=warp_idx,
                lane_idx=lane_idx_for_dispatch,
                tidx=tidx,
            )

        lane_idx = cute.arch.lane_idx()
        self.token_comm_hook_kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )
