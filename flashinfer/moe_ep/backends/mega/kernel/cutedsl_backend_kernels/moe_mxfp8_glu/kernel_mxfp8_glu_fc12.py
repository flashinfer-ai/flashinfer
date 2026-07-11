# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Fused fc1+fc2 GLU MXFP8 MegaMoE kernel for SM100.
"""

from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

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

from moe_nvfp4_swapab.epilogue import SwapABSwigluFp4Epilogue, _TmemTranspose16x32Core  # noqa: F401
from moe_mxfp8_glu.epilogue_mxfp8 import GluMxfp8Epilogue
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from moe_nvfp4_swapab.custom_ext import GluMxFp8Fc12SchedExtension
from common.megamoe_constants import (
    Mxfp8BlockSize,
    Nvfp4BlockSize,  # noqa: F401
    SupportedMmaTileM,
    SupportedMmaTileN,
)
from moe_nvfp4_swapab.moe_utils import spin_wait


# =============================================================================
# Sm100SwigluMxfp8Fc12Kernel
# =============================================================================


class Sm100SwigluMxfp8Fc12Kernel:
    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    # Supported (ab_dtype, sf_vec_size) pairings.
    # MXFP8 → Float8E4M3FN / Float8E5M2 + sf_vec_size=32  (FP8-E8M0 scales, MmaMXF8Op)
    VALID_AB_DTYPE_SF_SIZE: dict = {
        32: (
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ),
    }

    # Interleave granularity for gate and up in SwiGLU / GeGlu
    GateUpInterleave: int = 32

    def __init__(
        self,
        # Geometry (no defaults; perf-sensitive + coupled by the validator:
        # mma_tiler_m / cluster_m == 128;  use_2cta_instrs ⇔ mma_tiler_m == 256)
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        # Fused fc1+fc2 sched-side knobs
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        # Optional sched knobs (None = sane internal default).
        # ``static_expert_shape`` binds (experts, intermediate_gateup, hidden)
        # at codegen time; None keeps those dims runtime-dynamic.
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
        scenario: Literal["2Dx3D"] = "2Dx3D",
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        gate_up_clamp: Optional[float] = None,
    ) -> None:
        # v1 only the lean static-sched path; dyn_2d3d 12-warp path
        # (empty + drain_aux warps) is future work.
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (lean 7-warp). "
                "Dynamic CLC (force_static_sched=False) is future work."
            )

        # Validate (ab_dtype, sf_vec_size) pairing.
        if sf_vec_size in self.VALID_AB_DTYPE_SF_SIZE:
            valid_ab = self.VALID_AB_DTYPE_SF_SIZE[sf_vec_size]
            if ab_dtype not in valid_ab:
                raise ValueError(
                    f"ab_dtype={ab_dtype.__name__} is not valid for "
                    f"sf_vec_size={sf_vec_size}. "
                    f"Expected one of: {[t.__name__ for t in valid_ab_tuple]}."  # type: ignore[name-defined]  # noqa: F821 - upstream kernel-team code; valid_ab_tuple undefined, should likely be valid_ab
                )
        else:
            raise NotImplementedError(f"sf_vec_size must be {Mxfp8BlockSize} (MXFP8)")

        if scenario != "2Dx3D":
            raise NotImplementedError(
                f"v1 fused fc12 only supports scenario='2Dx3D' (forward); "
                f"got {scenario!r}."
            )
        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be 'static' or 'atomic_counter'; "
                f"got {load_balance_mode!r}."
            )

        # Only (M=256, N=256) with 2-CTA instructions is validated now.
        m, n, _k = mma_tiler_mnk
        if (m, n) != (256, 256) or not use_2cta_instrs:
            raise ValueError(
                "Sm100SwigluMxfp8Fc12Kernel only supports mma_tiler (M, N) = "
                "(256, 256) with use_2cta_instrs=True; "
                f"got mma_tiler_mnk={mma_tiler_mnk}, use_2cta_instrs={use_2cta_instrs}."
            )

        # Store ab_dtype so workspace-size helpers can use it without tensors.
        self.ab_dtype = ab_dtype

        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.epi_flag_batch = epi_flag_batch
        self.gate_up_clamp = abs(gate_up_clamp) if gate_up_clamp is not None else None

        self.acc_dtype = acc_dtype
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = use_2cta_instrs
        self.force_static_sched = force_static_sched
        # static_expert_shape / clc_bundle_size / num_sched_stages: accepted
        # for API parity with the runner; scheduler reads expert_cnt from
        # ``offs.shape[0]`` at runtime when ``static_expert_shape`` is None.
        self.static_expert_shape = static_expert_shape
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages

        # Fused fc12 sched-side knobs
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode

        self.sf_vec_size = sf_vec_size
        self.scenario = scenario
        self.arch = "sm_100"

        self._validate_mma_tiler_and_cluster_shape()
        self.mma_tiler = mma_tiler_mnk

        self.cta_group = (
            tcgen05.CtaGroup.TWO if use_2cta_instrs else tcgen05.CtaGroup.ONE
        )

        # Warp specialization (lean 8-warp / 256 thread)
        self.occupancy = 1
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_a_warp_id = 5
        self.tma_b_warp_id = 6
        self.sched_warp_id = 7
        self.threads_per_cta = 32 * len(
            (
                self.mma_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                *self.epilogue_warp_id,
            )
        )

        # NamedBarrier IDs.
        #
        # Per-subtile rotated-leader scheme lives inside the epilogue; this
        # kernel only owns the four reserved IDs and forwards
        # them via ``self.epilog_sync_bar_id`` to the epilogue ctor.
        # IDs 8 and 9 are reserved for MXFP8 warp-pair absmax exchange.
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)

        # MegaMoE toggle.  False for the lean base; subclasses (e.g.
        # ``Sm100MegaMoEMxfp8Kernel``) set this to True in their own
        # ``__init__`` after ``super().__init__`` returns.
        # ``_setup_attributes()`` reads it inside ``__call__`` to expand the
        # warp topology to the 12-warp MegaMoE layout.
        self.enable_token_comm: bool = False
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints.

        ``mma_tiler_n`` is restricted to {128, 256}.  Short-N is handled by
        the swap-AB scheduler via subtile-level early-exit.
        """
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if m not in SupportedMmaTileM:
            raise ValueError(f"mma_tiler M ({m}) must be one of {SupportedMmaTileM}")

        per_cta_m = m // (2 if self.use_2cta_instrs else 1)
        if per_cta_m != 128:
            raise ValueError(
                f"per-CTA mma_tiler M must be 128, got {per_cta_m} "
                f"(mma_tiler_m={m}, use_2cta_instrs={self.use_2cta_instrs})"
            )

        if n not in SupportedMmaTileN:
            raise ValueError(
                f"mma_tiler N ({n}) must be one of {SupportedMmaTileN} in fused fc12 "
                f"(N=64 SFB hack is dropped; swap-AB sched handles short-N "
                f"via subtile early-exit)."
            )

        sf_k_granularity = self.sf_vec_size * 4
        if k % sf_k_granularity != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a multiple of "
                f"sf_vec_size * 4 = {sf_k_granularity}"
            )

        if cm % (2 if self.use_2cta_instrs else 1) != 0:
            raise ValueError(
                f"cluster_shape M ({cm}) must be even when use_2cta_instrs=True"
            )

        is_pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if cm * cn > 16 or not is_pow2(cm) or not is_pow2(cn) or cm > 4 or cn > 4:
            raise ValueError(
                f"Invalid cluster_shape ({cm}, {cn}): each dim must be "
                f"a power of 2 and <= 4, product must be <= 16"
            )

        # v1 swap-AB requires cluster_n == 1.
        if cn != 1:
            raise NotImplementedError(
                f"v1 fused fc12 requires cluster_n == 1 (got {cn}).  "
                f"cluster_n > 1 needs sentinel-style acc/ab pipeline release."
            )

    def _create_tiled_mmas(self) -> Tuple[cute.TiledMma, cute.TiledMma]:
        """Return ``(tiled_mma, tiled_mma_sfb)``.

        Both phases share the same MMA configuration because ``mma_tiler_mnk``
        is shared.  Phase selection is
        purely a matter of which TMA load fills SMEM / which acc TMEM stage
        the MMA writes -- the tiled MMA atoms themselves are phase-invariant.

        SFB always uses ``CtaGroup.ONE``: SFB is not multicast across the
        2-CTA pair under ``use_2cta_instrs``.
        """
        common = (
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.sf_dtype,
            self.sf_vec_size,
        )
        tiled_mma = sm100_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            self.cta_group,
            self.mma_inst_shape_mn,
        )
        tiled_mma_sfb = sm100_utils.make_blockscaled_trivial_tiled_mma(
            *common,
            tcgen05.CtaGroup.ONE,
            self.mma_inst_shape_mn_sfb,
        )
        return tiled_mma, tiled_mma_sfb

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        """
        if self.enable_token_comm:
            self.dispatch_warp_id = (8, 9, 10, 11)
            num_token_back_warps = (
                len(self.token_back_warp_id) if self.token_back_standalone else 0
            )
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
            self.mma_inst_shape_mn[0] // (2 if self.use_2cta_instrs else 1),
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
        # We pass kernel-level params + ``allow_overlap_acc`` hint and read
        # the decisions back via @property below.
        # fc2 output dtype is hard-coded BFloat16 in both epilogues.
        _epi_common = dict(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            allow_overlap_acc=True,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
        )
        self.epilogue = GluMxfp8Epilogue(**_epi_common)

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # sC stages locked to subtile_cnt: one full CTA output tile lives in
        # sC at a time.  Per-subtile producer_acquire/commit go
        # away — only one batched commit + drain happens per task tile (in
        # the epilogue's ``run`` loop body).
        self.num_c_stage = self.epilogue.subtile_cnt
        c_bytes_total = self.epilogue.bytes_per_stage * self.num_c_stage

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            c_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.make_smem_layout_sfb(
            tiled_mma,
            self.mma_tiler,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.c_smem_layout_staged = self.epilogue.staged_smem_layout(
            self.num_c_stage,
        )

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
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

    def _smem_misc_budget_bytes(self) -> int:
        """Per-CTA SMEM reserved outside the AB-stage pipeline.

        Lean fc12 path: only the fixed ``_SmemMiscBudget``.  Subclasses (the
        MegaMoE wrapper) extend this to also reserve SMEM for the dispatch
        warps' pull_buffer / pull_mbar / smem_expert_count regions;
        ``_compute_stages`` reads this hook so the AB-stage count comes out
        small enough to leave room for those extra regions.
        """
        return self._SmemMiscBudget

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
    ) -> Tuple[int, int, int]:
        """Compute stage counts for ACC, AB+SF, and scheduler."""
        num_acc_stage = 2

        a_smem_layout_stage_one = sm100_utils.make_smem_layout_a(
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
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )

        fixed_overhead = self._smem_misc_budget_bytes() + c_bytes_total

        num_ab_stage = (
            smem_capacity // occupancy - fixed_overhead
        ) // ab_bytes_per_stage
        return num_acc_stage, num_ab_stage, num_sched_stages

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size for one fused fc1+fc2 launch."""
        sf_padding_block = self.sf_padding_block
        sf_vec_size = self.sf_vec_size  # noqa: F841

        mma_tiler_n = self.mma_tiler_mnk[1]

        data_total_rows, _hidden = fc1_activation_tensor.shape
        experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        intermediate_downproj = intermediate_gateup // 2

        # Conservative upper bound for sf_total_rows.
        sf_total_rows_upper = data_total_rows + experts * sf_padding_block

        # fc1_output byte size depends on the ab_dtype element width:
        #   MXFP8 (Float8E4M3FN/Float8E5M2, 8-bit): 1 element per byte → inter bytes/row
        fc1_output_bytes = (
            data_total_rows * intermediate_downproj * self.ab_dtype.width // 8
        )

        # fc1_output_sf sf_vec_size matches the kernel's sf_vec_size.
        fc1_out_sf_vec_size = self.sf_vec_size
        sf_block_cols = ((intermediate_downproj // fc1_out_sf_vec_size) + 3) // 4 * 4
        fc1_output_sf_bytes = sf_total_rows_upper * sf_block_cols

        # fc1_done_counter: one Int32 per global token block, plus expert slack.
        counter_slots_upper = (
            data_total_rows + mma_tiler_n - 1
        ) // mma_tiler_n + experts
        fc1_done_counter_bytes = counter_slots_upper * 4

        # load_balance_counter: Int32 scalar.
        if self.load_balance_mode == "atomic_counter":
            load_balance_counter_bytes = 4
        else:
            load_balance_counter_bytes = 0

        total = (
            fc1_output_bytes
            + fc1_output_sf_bytes
            + fc1_done_counter_bytes
            + load_balance_counter_bytes
        )

        # 128B align (TMA tensor base address alignment requirement).
        alignment = 128
        total = ((total + alignment - 1) // alignment) * alignment
        return total

    def mainloop_s2t_copy_and_partition(
        self,
        sSF: cute.Tensor,
        tSF: cute.Tensor,
    ) -> Tuple[cute.TiledCopy, cute.Tensor, cute.Tensor]:
        """SMEM → TMEM tiled copy + partition for SFA / SFB."""
        tCsSF_compact = cute.filter_zeros(sSF)
        tCtSF_compact = cute.filter_zeros(tSF)

        copy_atom_s2t = cute.make_copy_atom(
            tcgen05.Cp4x32x128bOp(self.cta_group),
            self.sf_dtype,
        )
        tiled_copy_s2t = tcgen05.make_s2t_copy(copy_atom_s2t, tCtSF_compact)
        thr_copy_s2t = tiled_copy_s2t.get_slice(0)

        tCsSF_compact_s2t_ = thr_copy_s2t.partition_S(tCsSF_compact)
        tCsSF_compact_s2t = tcgen05.get_s2t_smem_desc_tensor(
            tiled_copy_s2t, tCsSF_compact_s2t_
        )
        tCtSF_compact_s2t = thr_copy_s2t.partition_D(tCtSF_compact)

        return tiled_copy_s2t, tCsSF_compact_s2t, tCtSF_compact_s2t

    # ── MegaMoE communication hook stubs ─────────────────────────────────────
    #
    # No-op base implementations for the lean fc1+fc2 path.  Mirroring the
    # identically-named stubs in the NVFP4 base (``kernel_fc12.py``).
    # ``Sm100MegaMoEMxfp8Kernel`` (megamoe_kernel_mxfp8.py) overrides all of
    # them to delegate to ``src/token_comm.py``.

    def token_comm_extra_smem_storage_class(self) -> type:
        """Return a ``@cute.struct`` for dispatch-warp SMEM, or None."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return dispatch->fc1 release counter pointer, or None (lean: disabled)."""
        return None

    def sched_ext_fc1_peek_threshold(self) -> int:
        """Return the fc1 ready-counter peek threshold for GluMxFp8Fc12SchedExtension.

        Must match the spin threshold in ``token_comm_hook_fc1_tma_b_predispatch_spin``
        so that an early peek hit does not skip the spin and expose stale pool rows.
        Default 0 → use ``valid_tokens_in_tile`` (no cluster, base class behaviour).
        MegaMoE overrides to return ``cluster_tile_tokens`` to match the cluster spin.
        """
        return 0

    def sched_ext_fc1_counter_cumul_scale(self) -> int:
        """Return the scale factor for the fc1 ready-counter slot formula.

        Slot = scale * (cumul + fc1_counter_index) + tile_m_idx % scale.
        Default 1 = cluster-level granularity (slot = cumul + cluster_token_block_idx).
        A MegaMoE subclass can override to cluster_m for per-CTA granularity.
        """
        return 1

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Sched warp: wait for dispatch barrier before reading sizes.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info
    ):
        """TMA warp: spin until dispatch-pulled tokens are resident.  No-op base."""
        pass

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
        """Body for dispatch warps 8-11 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Body for standalone token-back warps 12-15 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_kernel_tail(self, token_comm_args, *, warp_idx, lane_idx, tidx):
        """All-warp kernel tail (NVLink release, etc.).  No-op base."""
        pass

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
        # Exactly one of ``offs`` / ``expert_token_sizes`` must be non-None.
        # MegaMoE subclass passes a zero-copy ``i32 stride=(2,)`` view onto
        # ``expert_recv_count_sum`` filled by the dispatch warps.
        expert_token_sizes: Optional[cute.Tensor] = None,
        # ── MegaMoE token-comm bundle (None on the lean path) ────────────
        # All MegaMoE-specific device branches are gated by
        # ``cutlass.const_expr(token_comm_args is not None)`` so they vanish
        # at codegen time on the lean path.
        token_comm_args=None,
    ) -> None:
        """Launch the fused fc1+fc2 GLU MXFP8 kernel."""

        # Bind data-tensor shapes to codegen-time expert dims when requested.
        # Strides, token rows, and SF tensors stay runtime-dynamic because they
        # encode host padding/swizzle choices.
        if cutlass.const_expr(self.static_expert_shape is not None):
            (
                experts_static,
                intermediate_gateup_static,
                hidden_static,
            ) = self.static_expert_shape
            intermediate_downproj_static = intermediate_gateup_static // 2

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
            # fc2_output is 2D (tokens, hidden) on the lean path and 3D
            # (max_tokens, topk, hidden) on the MegaMoE path.  Bind only the
            # hidden dim to a codegen-time const; keep the other dims dynamic.
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

        # C_gemm is a user-view output tensor; epilogue owns its store path.
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
        # ``fc1_output_gemm`` was built from ``fc1_output.iterator`` with the same
        # (tokens_sum, intermediate_downproj, fake-L=1) layout that fc2's
        # GEMM-B view wants; reuse it directly when wiring fc2 TMA-B atom).

        # C_gemm (fc2 output, BFloat16; STG.256-driven by the epilogue's
        # ``Fc2UnpackPermuteStg`` — no TMA store path).
        # We still build a GEMM-domain view so ``ext.get_gmem_tensor("d", ...)``
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
        # ``intermediate_gateup`` (= fc1_weight.shape[2]) is what we pass to the
        # scheduler via ``expert_shape``; see ``MoESchedulerParamsBase``
        # docstring for the precise contract.
        hidden_dim = hidden

        # ── Infer dtypes and major modes ──
        # Phases share dtypes by construction. For MXFP8: A/B/fc1_output
        # are Float8E4M3FN/Float8E5M2 and SFs are Float8E8M0FNU.
        # ``self.fc1_output_dtype`` drives the sC SMEM element type and flows
        # into the epilogue ctor as ``fc1_output_dtype``.
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
        # ``self.epilogue.tma_store_fc1_output``; commit / drain lives
        # inside the epilogue's ``run`` loop body.
        fc1_output_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            fc1_output_tma_op,
            fc1_output_gemm,
            self.epilogue.smem_layout_one_stage,
            self.epilogue.epi_tile,
        )

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
        # per-thread STG.
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, 1),
                self.sf_vec_size,
            ),
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
            b_smem_layout,
            self.mma_tiler,
            tiled_mma,
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
                sfb_smem_layout,
                self.mma_tiler_sfb,
                tiled_mma_sfb,
                self.cluster_layout_sfb_vmnk.shape,
                internal_type=cutlass.Uint64,
            )
        )

        # ── Scheduler params + grid + launch ──
        #
        # ``expert_cnt`` / ``intermediate_gateup`` / ``hidden_dim`` are
        # extracted from the (possibly rewritten) tensor shapes above:
        #   - static path (``static_expert_shape`` bound): they are
        #     codegen-time Python int constants; the new base
        #     ``MoESchedulerParamsBase.__init__`` preserves the Python
        #     int type and ``__extract_mlir_values__`` skips them, so
        #     they remain inlined literals across the scheduler's scf
        #     region boundaries (no demotion to iter_arg / kernel-arg).
        #   - dynamic path: they are runtime Int32 from tensor metadata.
        #
        # ``expert_shape[1]`` carries ``intermediate_gateup`` semantics
        # (= fc1_weight.shape[2]) per the ``MoESchedulerParamsBase.__init__``
        # contract.  The fused fc12 scheduler reads it as fc1 GEMM-M
        # (under swap-AB) and derives ``num_fc1_intermediate_blocks``
        # from it.
        # atomic_counter mode requires a host-allocated GMEM Int32 scalar
        # whose pointer lives in scheduler params; static mode passes
        # None (params validate this).  Caller's contract from __call__:
        # ``load_balance_counter`` is required iff ``load_balance_mode ==
        # 'atomic_counter'``; otherwise may be None.
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            if cutlass.const_expr(load_balance_counter is None):
                raise ValueError(
                    "load_balance_counter must be provided when "
                    "load_balance_mode == 'atomic_counter'"
                )
            load_balance_counter_ptr = load_balance_counter.iterator
        else:
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
            load_balance_mode=self.load_balance_mode,
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
            self.c_smem_layout_staged,
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
        # debug: (total_tokens, intermediate_half*3) fp32 for swiglu comparison
        # Scheduling
        offs: Optional[cute.Tensor],
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        token_comm_args=None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB GLU MXFP8 grouped GEMM.

        Lean (``force_static_sched=True``) path: 7-warp specialization with
        no empty / drain_aux warps and no expert-wise TMA desc rewriting
        (every desc is tile-invariant under swap-AB).

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))  # noqa: F841
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))  # noqa: F841
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, None, 0))  # noqa: F841
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, None, 0))  # noqa: F841

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
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages * 2
            ]
            sched_storage: SchedStorage  # type: ignore[valid-type]
            tmem_dealloc_mbar_ptr: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # MegaMoE-only dispatch-warp SMEM (pull_buffer, mbarriers, etc.).
        # Kept out of ``SharedStorage`` so the lean path never allocates it.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None

        # ── Pipelines: two TMA producer warps share the AB pipeline. ──

        ab_pipeline_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 2)
        num_tma_producer = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_tma_producer
        )
        ab_producer, ab_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes // 2,
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
        )
        sched_consumer = scheduler.make_consumer()

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offsets latency overlaps with pipeline setup.
        # Under MegaMoE + static load-balance, ``internal_init`` walks
        # per-expert sizes from ``expert_recv_count_sum`` -- those are not
        # valid until the dispatch barrier completes, so we defer init to the
        # sched warp (after ``token_comm_hook_sched_warp_pre_init_wait``).
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

        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE).  Under
        # ``overlapping_accum`` the two acc buffers share TMEM with SF
        # columns; we shrink the stage stride so the second buffer
        # starts at ``(256 - num_sf_tmem_cols)`` cols instead of 256.
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
                        (256 - self.num_sf_tmem_cols) * acc_fake.stride[0][1],
                    ),
                ),
            )

        # Cluster wait before TMEM alloc.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        mma_tiler_k = self.mma_tiler[2]
        # ``fc1_weight_gemm.shape[1]`` / ``fc2_weight_gemm.shape[1]``
        # both resolve to ``hidden`` / ``intermediate_downproj``.  Under
        # ``static_expert_shape`` they are codegen-time Python ints
        # (rewritten on ``fc1_weight`` / ``fc2_weight`` at ``__call__``
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

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (warp 7) — lean path
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
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
        # TMA-A loads weights/SFA; TMA-B loads activations/SFB and waits for
        # fc1 workspace readiness in fc2 phase.  Both feed the same AB pipeline.

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
                    # MegaMoE: spin until the dispatch warps have pulled this
                    # task tile's token activations into the L1 token buffer.
                    # No-op on the lean path (activations resident at launch).
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

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
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

                    # DEBUG: print counter slot/value after spin exits for first hidden N-tile only.
                    # Using tidx == tma_a_warp_id*32 since tidx==0 never fires in warp 5.
                    if tidx == cutlass.Int32(
                        32 * self.tma_a_warp_id
                    ) and work_tile_info.tile_n_idx == cutlass.Int32(0):
                        counter_val_post = cute.arch.load(  # noqa: F841
                            counter_ptr, counter_ptr.dtype, cop="cg"
                        )
                        # Also load first Int32 from fc1_output for this tile to
                        # check if TMA S2G stores are visible (non-zero = stored).
                        fc1_byte_offset = (
                            work_tile_info.cumulative_data_physical_row
                            + work_tile_info.tile_m_idx
                            // cutlass.Int32(self.epilogue._atom_thr_size)
                            * cutlass.Int32(self.epilogue._cta_tile_m)
                        ) * fc1_output_gemm.stride[0]
                        fc1_probe_ptr = cute.make_ptr(
                            cutlass.Int32,
                            fc1_output_gemm.iterator.toint() + fc1_byte_offset,
                            cute.AddressSpace.gmem,
                        )
                        fc1_first_i32 = cute.arch.load(  # noqa: F841
                            fc1_probe_ptr, cutlass.Int32, cop="cg"
                        )
                        # cute.printf(
                        #     "[fc2_spin_exit] slot=%d counter=%d thresh=%d "
                        #     "tile_m=%d cumul=%d fc1_first_i32=0x%08x",
                        #     counter_slot,
                        #     counter_val_post,
                        #     fc2_spin_threshold,
                        #     work_tile_info.tile_m_idx,
                        #     work_tile_info.cumulative_token_block_count,
                        #     fc1_first_i32,
                        # )

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

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
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

            ab_producer.tail()

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

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
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
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc2_weight_sf,
                        block_in_cluster_coord_sfb_vmnk[1],
                        sfb_cta_layout,
                        cute.group_modes(sSFB, 0, 3),
                        cute.group_modes(tCgSFB, 0, 3),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)

                    # fc2 B-side = fc2_weight (N=hidden). tile_n_idx is the
                    # CTA-level hidden block index; invariant across token blocks.
                    fc2_b_hidden_tile = work_tile_info.tile_n_idx
                    tBgB_slice = tBgB[(None, fc2_b_hidden_tile, None, 0)]
                    tBgSFB_slice = tBgSFB[(None, fc2_b_hidden_tile, None, 0)]

                    ab_producer.reset()
                    peek_ab_empty_status = ab_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):  # noqa: B007
                        handle = ab_producer.acquire_and_advance(peek_ab_empty_status)
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = ab_producer.try_acquire()
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

            ab_producer.tail()

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

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_pipeline_stages
            )

            # K-tile counts ``k_tile_cnt_fc1`` / ``k_tile_cnt_fc2`` come
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
                    ab_consumer.reset()
                    peek_ab_full_status = cutlass.Boolean(1)
                    if k_tile_cnt > 0:
                        peek_ab_full_status = ab_consumer.try_wait()
                        acc_pipeline.producer_acquire(acc_producer_state)
                    if _iket_active:
                        iket.range_pop()

                    tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = ab_consumer.wait_and_advance(peek_ab_full_status)
                        peek_ab_full_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_full_status = ab_consumer.try_wait()

                        s2t_stage_coord = (None, None, None, None, handle.index)
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

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        tile_crd = (None, None, None, handle.index)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            [tCrA[tile_crd], tCtSFA],
                            [tCrB[tile_crd], tCtSFB],
                            tCtAcc,
                        )
                        handle.release()

                    if k_tile_cnt > 0:
                        acc_pipeline.producer_commit(acc_producer_state)
                if k_tile_cnt > 0:
                    acc_producer_state.advance()

                if _iket_active:
                    iket.range_pop()

                work_tile_info = sched_consumer.consume_work()

            acc_pipeline.producer_tail(acc_producer_state)

        # ── sC SMEM (fc1 output staging; fc2 doesn't use it) ──
        sC = smem.allocate_tensor(
            element_type=self.fc1_output_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        #
        # Fully delegated to ``self.epilogue.run(...)`` -- the epilogue owns
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
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sC,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=tma_tensor_fc1_output,
                gmem_fc1_output_sf=fc1_output_sf_gemm,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                warp_idx=epi_warp_idx,
                tidx=tidx,
                alpha=cutlass.Float32(1.0),
                norm_const=cutlass.Float32(1.0),
            )

            # MegaMoE: pass token_comm_args only when it is a real bundle (not
            # None).  Passing Python None explicitly to @cute.jit methods
            # triggers a CuteDSL codegen issue; const_expr dispatch avoids any
            # None-as-JIT-argument path.
            if cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(**_run_kwargs, token_comm_args=token_comm_args)
            else:
                self.epilogue.run(**_run_kwargs)

            tmem.relinquish_alloc_permit()
            tmem.free(acc_tmem_ptr)
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (warps 8-11; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` → warps 8-11 don't exist (threads_per_cta
        # = 256), so the guard is const_expr-eliminated in the lean path.
        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx >= self.dispatch_warp_id[0]:
                lane_idx_for_dispatch = cute.arch.lane_idx()
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        self.token_comm_hook_dispatch_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                    else:
                        self.token_comm_hook_token_back_warp_body(
                            token_comm_args,
                            token_comm_storage,
                            warp_idx=warp_idx,
                            lane_idx=lane_idx_for_dispatch,
                            tidx=tidx,
                        )
                else:
                    self.token_comm_hook_dispatch_warp_body(
                        token_comm_args,
                        token_comm_storage,
                        warp_idx=warp_idx,
                        lane_idx=lane_idx_for_dispatch,
                        tidx=tidx,
                    )

            # ════════════════════════════════════════════════════════════════════
            # Kernel tail hook (MegaMoE-only; lean base = no-op)
            # ════════════════════════════════════════════════════════════════════
            lane_idx = cute.arch.lane_idx()
            self.token_comm_hook_kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
