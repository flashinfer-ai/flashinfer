# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""
Fused fc1+fc2 GLU BF16 MegaMoE kernel for SM100.
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

from moe_bf16_glu.epilogue_bf16 import GluBf16Epilogue
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from moe_bf16_glu.custom_ext_bf16 import GluBf16Fc12SchedExtension
from common.megamoe_constants import (
    SupportedMmaTileM,
    SupportedMmaTileN,
)
from common.host_utils import get_cutedsl_target_arch
from moe_nvfp4_swapab.moe_utils import spin_wait


# =============================================================================
# Sm100SwigluBf16Fc12Kernel
# =============================================================================

class Sm100SwigluBf16Fc12Kernel:

    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    # Supported A/B element types (BF16 in / FP32 accumulate, MmaF16BF16Op).
    # Float16 is a plausible future addition but is untested.
    VALID_AB_DTYPES: tuple = (cutlass.BFloat16,)

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
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        # Optional sched knobs (None = sane internal default).
        # ``static_expert_shape`` binds (experts, intermediate_gateup, hidden)
        # at codegen time; None keeps those dims runtime-dynamic.
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        scenario: Literal["2Dx3D"] = "2Dx3D",
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Tuple[int, int] = (1, 1),
        gate_up_clamp: Optional[float] = None,
        apply_topk_in_fc1: bool = False,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
    ) -> None:
        # v1 only the lean static-sched path; dyn_2d3d 12-warp path
        # (empty + drain_aux warps) is future work.
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (lean 7-warp). "
                "Dynamic CLC (force_static_sched=False) is future work."
            )

        if ab_dtype not in self.VALID_AB_DTYPES:
            raise ValueError(
                f"ab_dtype={ab_dtype.__name__} is not supported; expected "
                f"one of: {[t.__name__ for t in self.VALID_AB_DTYPES]}."
            )

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
                "Sm100SwigluBf16Fc12Kernel only supports mma_tiler (M, N) = "
                "(256, 256) with use_2cta_instrs=True; "
                f"got mma_tiler_mnk={mma_tiler_mnk}, use_2cta_instrs={use_2cta_instrs}."
            )

        # Problem-shape store granularity: the fc1 STG path (use_stg_fc1)
        # stores full 256-wide gate+up tiles without N predication, so it
        # needs intermediate % 256.  The default R2S+TMA store path clamps at
        # the tensor extent and the fc1/fc2 tile loops are ceil-counted with
        # TMA OOB zero-fill, so its only hard requirement is the gate/up
        # interleave pair: gate and up alternate in 32-column groups, making
        # one indivisible pair unit 64 gate+up columns (intermediate % 64).
        # fc2 stores 32-wide hidden subtiles (predicated at 32-column
        # granularity only).
        if static_expert_shape is not None:
            _experts_s, intermediate_gateup_s, hidden_s = static_expert_shape
            gateup_granularity = 256 if use_stg_fc1 else 64
            if intermediate_gateup_s % gateup_granularity != 0:
                raise ValueError(
                    f"intermediate (gate+up width) must be a multiple of "
                    f"{gateup_granularity} "
                    f"(use_stg_fc1={use_stg_fc1}); got {intermediate_gateup_s}."
                )
            if hidden_s % 32 != 0:
                raise ValueError(
                    f"hidden must be a multiple of 32; got {hidden_s}."
                )

        # Store ab_dtype so workspace-size helpers can use it without tensors.
        self.ab_dtype = ab_dtype
        self.c_dtype = cutlass.BFloat16

        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.epi_flag_batch = epi_flag_batch
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.generate_c = generate_c
        self.use_stg_fc1 = use_stg_fc1
        self.gate_up_clamp = (
            abs(gate_up_clamp) if gate_up_clamp is not None else None
        )

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
        self.load_balance_mode = load_balance_mode

        self.scenario = scenario
        self.arch = get_cutedsl_target_arch()

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
        self.epilog_sync_bar_id = 1
        self.tmem_alloc_sync_bar_id = 2
        self.tmem_dealloc_sync_bar_id = 3
        self.epi_subtile_bar_ids = (4, 5, 6, 7)

        # MegaMoE toggle.  False for the lean base; subclasses (e.g.
        # ``Sm100MegaMoEBf16Kernel``) set this to True in their own
        # ``__init__`` after ``super().__init__`` returns.
        # ``_setup_attributes()`` reads it inside ``__call__`` to expand the
        # warp topology to the 12-warp MegaMoE layout.
        self.enable_token_comm: bool = False
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False

        self.smem_capacity = utils.get_smem_capacity_in_bytes()
        self.num_tmem_alloc_cols = cute.arch.get_max_tmem_alloc_cols(self.arch)

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints.

        ``mma_tiler_n`` is restricted to {128, 256}.  Short-N is handled by
        the fused-fc12 scheduler via subtile-level early-exit.
        """
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        # Only cluster (2, 1) is validated: the epilogue / counter / dispatch
        # math is derived for one 2-CTA instruction pair per cluster.
        if (cm, cn) != (2, 1):
            raise ValueError(
                f"Sm100SwigluBf16Fc12Kernel only supports cluster_shape "
                f"(M, N) = (2, 1); got ({cm}, {cn})."
            )

        if m not in SupportedMmaTileM:
            raise ValueError(
                f"mma_tiler M ({m}) must be one of {SupportedMmaTileM}"
            )

        per_cta_m = m // (2 if self.use_2cta_instrs else 1)
        if per_cta_m != 128:
            raise ValueError(
                f"per-CTA mma_tiler M must be 128, got {per_cta_m} "
                f"(mma_tiler_m={m}, use_2cta_instrs={self.use_2cta_instrs})"
            )

        if n not in SupportedMmaTileN:
            raise ValueError(
                f"mma_tiler N ({n}) must be one of {SupportedMmaTileN} in fused fc12 "
                f"(short-N is handled via subtile-level early-exit)."
            )

        # K is validated against the MMA instruction K in _setup_attributes.

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

        # v1 fused fc12 requires cluster_n == 1.
        if cn != 1:
            raise NotImplementedError(
                f"v1 fused fc12 requires cluster_n == 1 (got {cn}).  "
                f"cluster_n > 1 needs sentinel-style acc/ab pipeline release."
            )

    def _create_tiled_mma(self) -> cute.TiledMma:
        """Return the dense BF16 tiled MMA.

        Both phases share the same MMA configuration because ``mma_tiler_mnk``
        is shared.  Phase selection is
        purely a matter of which TMA load fills SMEM / which acc TMEM stage
        the MMA writes -- the tiled MMA atom itself is phase-invariant.
        """
        return sm100_utils.make_trivial_tiled_mma(
            self.a_dtype,
            self.b_dtype,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_inst_shape_mn,
        )

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

        tiled_mma = self._create_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )

        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue is autonomous: it owns all epi-side decisions (acc stages,
        # subtile dispatch, TMA commit/drain, piggyback red.add).  We pass
        # kernel-level params and read the decisions back via @property
        # below.  fc2 output dtype is hard-coded BFloat16.
        _epi_common = dict(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            generate_c=self.generate_c,
            use_stg_fc1=self.use_stg_fc1,
        )
        self.epilogue = GluBf16Epilogue(**_epi_common)

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # The TMA path stages one full CTA output tile in sD.  The direct-STG
        # path writes each BF16 vector from registers to GMEM and has no sD
        # consumer, so do not reserve that SMEM; the freed bytes can become an
        # additional A+B mainloop stage.
        self.num_d_stage = (
            0 if self.use_stg_fc1 else self.epilogue.subtile_cnt
        )
        d_bytes_total = self.epilogue.bytes_per_stage * self.num_d_stage
        # Raw gate+up SMEM (ping-pong) — only when generate_c=True.
        c_bytes_total = 0
        if self.generate_c:
            from moe_bf16_glu.epilogue_bf16 import Fc1CTMAStages
            self.num_c_raw_stage = Fc1CTMAStages
            c_bytes_total += self.epilogue.c_bytes_per_stage * self.num_c_raw_stage
        else:
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
        if self.use_stg_fc1:
            self.d_smem_layout_staged = None
        else:
            self.d_smem_layout_staged = self.epilogue.staged_smem_layout(
                self.num_d_stage,
            )
        # Raw gate+up SMEM layout (only meaningful when generate_c=True; pass as
        # dummy to kernel when False).
        if self.generate_c:
            self.c_smem_layout_staged = self.epilogue.staged_c_smem_layout(
                self.num_c_raw_stage
            )
        else:
            self.c_smem_layout_staged = None

        # Read epilogue's autonomous decisions.
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_accumulator_tmem_cols = self.epilogue.num_accumulator_tmem_cols

        # TMA load bytes per stage (A + B).
        atom_thr_size = cute.size(tiled_mma.thr_id.shape)
        self.atom_thr_size = atom_thr_size  # store as Python int for use in @cute.kernel
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_a_bytes = a_copy_size * atom_thr_size
        self.num_tma_load_b_bytes = b_copy_size * atom_thr_size

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
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
    ) -> Tuple[int, int, int]:
        """Compute stage counts for ACC, AB, and scheduler.
        """
        num_acc_stage = 2

        a_smem_layout_staged_one = sm100_utils.make_smem_layout_a(
            tiled_mma, mma_tiler_mnk, a_dtype, 1,
        )
        b_smem_layout_staged_one = sm100_utils.make_smem_layout_b(
            tiled_mma, mma_tiler_mnk, b_dtype, 1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_staged_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
        )
        b_bytes_per_stage = cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)

        fixed_overhead = (
            self._smem_misc_budget_bytes() + c_bytes_total
        )

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
        print(
            f"[fc12 stages] num_ab_stage={num_a_stage, num_b_stage} "
            f"num_acc_stage={num_acc_stage} "
            f"misc_budget={self._smem_misc_budget_bytes()} "
            f"c_bytes_total={c_bytes_total} "
            f"smem_cap={smem_capacity} "
            f"unused_smem={unused_smem}"
        )

        return num_acc_stage, num_a_stage, num_b_stage, num_sched_stages

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size for one fused fc1+fc2 launch."""
        data_total_rows, _hidden = fc1_activation_tensor.shape
        experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        intermediate_downproj = intermediate_gateup // 2

        # fc1_output byte size follows the ab_dtype element width (BF16: 2 B).
        fc1_output_bytes = (
            data_total_rows * intermediate_downproj * self.ab_dtype.width // 8
        )

        # fc1_done_counter: one Int32 per cluster token block (the token axis
        # is M; one block covers per-CTA tile M x cluster_m tokens, matching
        # the scheduler's ``cumulative_token_block_count`` units), plus
        # expert slack for per-expert ceil rounding.
        per_cta_tile_m = self.mma_tiler_mnk[0] // (
            2 if self.use_2cta_instrs else 1
        )
        cluster_tile_tokens = per_cta_tile_m * self.cluster_shape_mn[0]
        counter_slots_upper = (
            (data_total_rows + cluster_tile_tokens - 1) // cluster_tile_tokens
            + experts
        )
        fc1_done_counter_bytes = counter_slots_upper * 4

        # load_balance_counter: Int32 scalar.
        if self.load_balance_mode == "atomic_counter":
            load_balance_counter_bytes = 4
        else:
            load_balance_counter_bytes = 0

        total = (
            fc1_output_bytes
            + fc1_done_counter_bytes
            + load_balance_counter_bytes
        )

        # 128B align (TMA tensor base address alignment requirement).
        alignment = 128
        total = ((total + alignment - 1) // alignment) * alignment
        return total

    # ── MegaMoE communication hook stubs ─────────────────────────────────────
    #
    # No-op base implementations for the lean fc1+fc2 path.
    # ``Sm100MegaMoEBf16Kernel`` (megamoe_kernel_bf16.py) overrides all of
    # them to delegate to ``src/token_comm.py``.

    def token_comm_extra_smem_storage_class(self) -> type:
        """Return a ``@cute.struct`` for dispatch-warp SMEM, or None."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return dispatch->fc1 release counter pointer, or None (lean: disabled)."""
        return None

    def sched_ext_fc1_peek_threshold(self) -> int:
        """Return the fc1 ready-counter peek threshold for GluBf16Fc12SchedExtension.

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
    def token_comm_hook_fc1_tma_b_predispatch_spin(self, token_comm_args, work_tile_info):
        """TMA warp: spin until dispatch-pulled tokens are resident.  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
    ):
        """Body for dispatch warps 8-11 (MegaMoE-only).  No-op base."""
        pass

    @cute.jit
    def token_comm_hook_token_back_warp_body(
        self, token_comm_args, token_comm_storage, *, warp_idx, lane_idx, tidx,
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
        activation: cute.Tensor,           # (token_sum_padded, hidden)
        fc1_weight: cute.Tensor,           # (experts, hidden, intermediate_gateup)
        # ── fc1 workspace consumed as fc2 GEMM-A ─────────────────────────
        fc1_output: cute.Tensor,         # (token_sum_padded, intermediate_downproj)
        # ── fc2 (Linear2) problem tensors ────────────────────────────────
        fc2_weight: cute.Tensor,          # (experts, intermediate_downproj, hidden)
        fc2_output: cute.Tensor,         # (token_sum_padded, hidden) BFloat16, hidden stride-1
        # ── topk weights (Path A) ────────────────────────────────────────
        topk_scores: cute.Tensor,     # (token_sum_padded,) Float32
        # ── Cross-phase workspace ────────────────────────────────────────
        fc1_done_counter: cute.Tensor,  # (max_token_block_per_rank,) Int32
        # ── Sched / runtime ──────────────────────────────────────────────
        offs: Optional[cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
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
        # ── Raw fc1 accumulator output (generate_c path) ─────────────────
        # Shape: (token_sum_padded, intermediate_gateup) Float32.
        fc1_c: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch the fused fc1+fc2 GLU BF16 kernel."""

        # Bind data-tensor shapes to codegen-time expert dims when requested.
        # Strides and token rows stay runtime-dynamic because they encode
        # host padding choices.
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
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
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
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
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

        expert_cnt = experts
        # ``intermediate_gateup`` (= fc1_weight.shape[2]) is what we pass to the
        # scheduler via ``expert_shape``; see ``MoESchedulerParamsBase``
        # docstring for the precise contract.
        hidden_dim = hidden

        # ── Infer dtypes and major modes ──
        # Phases share dtypes by construction: A/B/fc1_output are BFloat16.
        # ``self.fc1_output_dtype`` drives the sD SMEM element type and flows
        # into the epilogue ctor as ``fc1_output_dtype``.
        self.a_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(activation_gemm).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(fc1_weight_gemm).mma_major_mode()
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        # ── fc1 TMA atoms ──

        # TMA load A1 (= fc1 activations, non-swap-AB: A=activations)
        a_op = sm100_utils.cluster_shape_to_tma_atom_A(
            self.cluster_shape_mn, tiled_mma.thr_id
        )
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            activation_gemm,
            a_smem_layout,
            self.mma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
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

        # TMA store for fc1 BF16 output.
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

        # TMA store for the raw fc1 accumulator (gate+up) — generate_c path.
        # GMEM shape: (tokens_sum, intermediate_gateup, 1).
        # TMA tile: self.epilogue.epi_tile_c = (128 tokens, 64).
        if cutlass.const_expr(self.generate_c):
            c_gemm = cute.make_tensor(
                fc1_c.iterator,
                cute.make_layout(
                    (tokens_sum, intermediate_gateup, 1),
                    stride=(fc1_c.stride[0], fc1_c.stride[1], 0),
                ),
            )
            tma_atom_c, tma_tensor_c = cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileS2GOp(),
                c_gemm,
                self.epilogue.c_smem_layout_one_stage,
                self.epilogue.epi_tile_c,
            )
        else:
            # Dummy — never used by the kernel (const_expr gated).
            tma_atom_c = tma_atom_fc1_output
            tma_tensor_c = tma_tensor_fc1_output

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
        tma_atom_fc2_weight, tma_tensor_fc2_weight = (
            cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                fc2_weight_gemm,
                b_smem_layout,
                self.mma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
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
        # contract; the fused fc12 scheduler derives
        # ``num_fc1_intermediate_blocks`` from it.
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
            # Required (positive) by the shared scheduler params; the
            # resulting row bookkeeping has no consumer in this kernel, so
            # pass the round_up-neutral 1.
            sf_padding_block=1,
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
            # fc1 TMA atoms / tensors (A=activations, B=weights)
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            # GEMM-domain tensors (fc1)
            activation_gemm,
            fc1_weight_gemm,
            fc1_output_gemm,
            # GEMM-domain tensors (fc2)
            fc2_weight_gemm,
            fc2_output_gemm,
            # topk + cross-phase sync workspace
            topk_scores,
            fc1_done_counter,
            # Scheduling
            offs,
            sched_params,
            self.cluster_layout_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.d_smem_layout_staged,
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
        # fc1 TMA atoms / tensors
        tma_atom_fc1_activation_1: cute.CopyAtom,
        tma_tensor_fc1_activation_1: cute.Tensor,
        tma_atom_weight: cute.CopyAtom,
        tma_tensor_weight: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors (fc1_output→A, fc2_weight→B)
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_weight_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_weight_gemm: cute.Tensor,
        fc2_output_gemm: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        # Scheduling
        offs: Optional[cute.Tensor],
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        d_smem_layout_staged: Optional[
            Union[cute.Layout, cute.ComposedLayout]
        ],
        c_smem_layout_staged:  Optional[Union[cute.Layout, cute.ComposedLayout]] = None,
        tma_atom_c: Optional[cute.CopyAtom] = None,
        tma_tensor_c: Optional[cute.Tensor] = None,
        token_comm_args=None,
    ):
        """Device kernel for the fused fc1+fc2 GLU BF16 grouped GEMM.

        Lean (``force_static_sched=True``) path: 7-warp specialization with
        no empty / drain_aux warps and no expert-wise TMA desc rewriting.

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, None, 0))

        # fc2 waits for all fc1 intermediate N-tiles in the same token block.
        # Each N-tile is processed by atom_thr_size CTAs (both CTA0 and CTA1 increment
        # the counter), so the threshold must account for both CTAs' contributions.
        ext_fc2_spin_threshold = (
            fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[1] - 1
        ) // self.cta_tile_shape_mnk[1] * self.epilogue._atom_thr_size

        ext = GluBf16Fc12SchedExtension(
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
        tidx, _, _ = cute.arch.thread_idx()

        # SharedStorage.
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            a_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_a_stage * 2]
            b_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_b_stage * 2]
            acc_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_stage * 2
            ]
            sched_storage: SchedStorage
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

        # ── Pipelines: separate producer/consumer groups for A and B. ──

        a_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1
        )
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
        b_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 1
        )
        b_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, self.num_mcast_ctas_b
        )
        b_producer, b_consumer = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_full_mbar_ptr.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=b_pipeline_producer_group,
            consumer_group=b_pipeline_consumer_group,
            tx_count=self.num_tma_load_b_bytes,
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
            num_stages=self.num_acc_stage,
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
        )
        sched_consumer = scheduler.make_consumer()

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offsets latency overlaps with pipeline setup.
        # Under MegaMoE + static load-balance, ``internal_init`` walks
        # per-expert sizes from ``expert_recv_count_sum`` -- those are not
        # valid until the dispatch barrier completes, so we defer init to the
        # sched warp (after ``token_comm_hook_sched_warp_pre_init_wait``).
        early_internal_init = (
            (self.load_balance_mode == "atomic_counter")
            or (not self.enable_token_comm)
        )

        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ── SMEM tensors A / B (shared by fc1 / fc2) ──
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
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])

        # acc_fake layout: (MMA, MMA_M, MMA_N, STAGE).  The two acc stages
        # tile TMEM back-to-back (2 x 256 cols = the full 512-col budget).
        acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
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
        # TMA-A loads activations into the A pipeline (and waits for fc1
        # workspace readiness in the fc2 phase).
        # TMA-B loads weights into the B pipeline.

        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            _iket_active = (tidx == cutlass.Int32(160))
            a_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
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

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    # ── fc1 phase A-side (non-swap-AB: A=activations, per-CTA partitioned) ──
                    # Activations are split per CTA (tokens 0-127 for CTA 0, 128-255 for CTA 1).
                    # Use b_cta_layout + m-coord so each CTA loads its own token range.
                    # mcast_mode=1 → same M-coord = only self → no actual multicast. ✓
                    if _iket_active:
                        iket.range_push("tma_token_fc1")
                    # MegaMoE: spin until the dispatch warps have pulled this
                    # task tile's token activations into the L1 token buffer.
                    # No-op on the lean path (activations resident at launch).
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args, work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc1_activation", tma_tensor_fc1_activation_1, work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_activation_1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )

                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(
                            peek_a_empty_status
                        )
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
                else:
                    # ── fc2 phase A-side: load fc1_output (M=tokens) + wait for fc1 done ──
                    #
                    # Non-swap-AB fc2: A=fc1_output (M=tokens). tile_m_idx is the
                    # CTA-level token block. Counter wait moved here from TMA-B.
                    if _iket_active:
                        iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count
                        + work_tile_info.tile_m_idx // cutlass.Int32(self.epilogue._atom_thr_size)
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
                    cute.arch.load(counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu")
                    cute.arch.fence_proxy("async")
                    cute.arch.fence_proxy("async.global")

                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "fc2_activation", tma_tensor_fc2_activation, work_tile_info,
                    )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_activation,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )

                    # fc2 A-side = fc1_output (M=tokens). tAgA is cluster-level
                    # indexed, so divide tile_m_idx by cluster_m to get the
                    # cluster block index — same formula as fc1 A-side.
                    mma_tile_m = work_tile_info.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]

                    a_producer.reset()
                    peek_a_empty_status = a_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(
                            peek_a_empty_status
                        )
                        peek_a_empty_status = cutlass.Boolean(1)
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

                if _iket_active:
                    iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            a_producer.tail()

        # ── TMA-B warp (warp 6) ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            _iket_active = (tidx == cutlass.Int32(192))
            b_full_mcast_mask = None
            if cutlass.const_expr(self.is_b_mcast or use_2cta_instrs):
                b_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
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

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )

                if is_phase_linear1:
                    # ── fc1 phase B-side (fc1_weight, N-side GEMM-B) ──
                    # Weight is expert-indexed (key "a") and N-side: use tile_n_idx
                    # (not tile_m_idx) to select the N-tile. tile_m_idx counts token
                    # blocks and grows with more tokens, causing OOB on the weight's
                    # N-dimension when tile_m_idx >= 2.
                    if _iket_active:
                        iket.range_push("tma_weight_fc1")

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc1_weight", tma_tensor_weight, work_tile_info,
                    )

                    # N-K tiling for N-side weight (N=intermediate, K=hidden).
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )

                    # Use tile_n_idx for N-side weight: invariant across token blocks.
                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]

                    b_producer.reset()
                    peek_b_empty_status = b_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = b_producer.acquire_and_advance(
                            peek_b_empty_status
                        )
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
                else:
                    # ── fc2 phase B-side: load fc2_weight (N=hidden), no counter wait ──
                    # fc2_weight is independent of fc1; counter wait is in TMA-A.
                    if _iket_active:
                        iket.range_push("tma_weight_fc2")
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "fc2_weight", tma_tensor_fc2_weight, work_tile_info,
                    )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )

                    # fc2 B-side = fc2_weight (N=hidden). tile_n_idx is the
                    # CTA-level hidden block index; invariant across token blocks.
                    fc2_b_hidden_tile = work_tile_info.tile_n_idx
                    tBgB_slice = tBgB[(None, fc2_b_hidden_tile, None, 0)]

                    b_producer.reset()
                    peek_b_empty_status = b_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = b_producer.acquire_and_advance(
                            peek_b_empty_status
                        )
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
            _iket_active = (tidx == cutlass.Int32(128))

            tCrA = tiled_mma.make_fragment_A(sA)
            tCrB = tiled_mma.make_fragment_B(sB)

            tmem.wait_for_alloc()
            acc_tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
            acc_base = cute.make_tensor(acc_tmem_ptr, acc_fake.layout)

            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_acc_stage
            )

            # K-tile counts ``k_tile_cnt_fc1`` / ``k_tile_cnt_fc2`` come
            # from the enclosing scope (computed once before the TMA warps).

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
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
                        if handle_a.count + 1 < k_tile_cnt:
                            peek_a_full_status = a_consumer.try_wait()
                            peek_b_full_status = b_consumer.try_wait()

                        tiled_mma.set(tcgen05.Field.ACCUMULATE, k_tile != 0)
                        cute.gemm(
                            tiled_mma,
                            tCtAcc,
                            tCrA[(None, None, None, handle_a.index)],
                            tCrB[(None, None, None, handle_b.index)],
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

        # ── sD SMEM (fc1 TMA-output staging; fc2 doesn't use it) ──
        # Direct STG consumes the BF16 registers directly, so keeping this
        # allocation would waste exactly one A+B stage for the baseline shape.
        if cutlass.const_expr(self.use_stg_fc1):
            sD = None
        else:
            sD = smem.allocate_tensor(
                element_type=self.fc1_output_dtype,
                layout=d_smem_layout_staged.outer,
                byte_alignment=128,
                swizzle=d_smem_layout_staged.inner,
            )

        # ── sC SMEM (raw gate+up Float32, ping-pong; only when generate_c=True) ──
        if cutlass.const_expr(self.generate_c):
            sC = smem.allocate_tensor(
                element_type=self.epilogue._c_dtype,
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

            # Build common kwargs for the epilogue.
            if cutlass.const_expr(self.generate_c):
                _smem_c_raw_arg = sC
                _tma_atom_c_arg = tma_atom_c
                _gmem_c_arg = tma_tensor_c
            else:
                _smem_c_raw_arg = None
                _tma_atom_c_arg = None
                _gmem_c_arg = None
            if cutlass.const_expr(self.use_stg_fc1):
                _gmem_fc1_output_arg = fc1_output_gemm
            else:
                _gmem_fc1_output_arg = tma_tensor_fc1_output
            _run_kwargs = dict(
                tmem_acc_tensor=acc_tensor,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sD,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=_gmem_fc1_output_arg,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                smem_c_buffer=_smem_c_raw_arg,
                tma_atom_c=_tma_atom_c_arg,
                gmem_c=_gmem_c_arg,
                warp_idx=epi_warp_idx,
                tidx=tidx,
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
