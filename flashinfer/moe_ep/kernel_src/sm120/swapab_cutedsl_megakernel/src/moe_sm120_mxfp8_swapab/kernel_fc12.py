# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Fused fc1+fc2 swap-AB SwiGLU MXFP8 kernel for SM120."""

import os
from typing import Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    try:
        from cutlass.cute.experimental import iket  # type: ignore
    except ImportError:
        from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from .fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from .custom_ext import Sm120SwapABMxfp8Fc12SchedExtension
from common.megamoe_constants import (
    Mxfp8BlockSize,
)
from .moe_utils import spin_wait, spin_wait_i32_ge_inline
from .sm120_mma import (
    MMA_N,
    SWAP_AB_INTERLEAVE,
    issue_m64n8k32_mxfp8,
    issue_m64n8k128_mxfp8,
    make_sm120_ldmatrix_atom,
    make_swapab_m64n8k128_tiled_mma,
)
from common.moe_utils import (
    cvt_f32_to_f8_to_f32,
    cvt_f32x4_to_f8x4_pack_i32,
)
from common.megamoe_constants import (
    Fp8E4M3RcpLimit,
    Fp8E5M2RcpLimit,
    Fp32Max,
    Log2E,
)
from src.token_comm import TokenSrcMetadata


UseScaleTma = True


# token_comm_args is an opaque subclass-owned bundle.  The base only forwards it
# to hook methods; ``None`` keeps the lean fc1+fc2 path free of token-comm IR.


# =============================================================================
# Sm120SwapABSwigluMxfp8Fc12Kernel
# =============================================================================


class Sm120SwapABSwigluMxfp8Fc12Kernel:
    """Fused fc1+fc2 swap-AB SwiGLU MXFP8 grouped GEMM for MoE on SM120.

    This class owns the local fc1/fc2 GEMM pipeline and exposes token-comm
    hooks for the MegaMoE subclass.
    """

    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, TMEM allocator state).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    def __init__(
        self,
        # Geometry.
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        # Fused fc1+fc2 scheduler knobs.
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        # Optional scheduler/codegen knobs.
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 32,
        scenario: Literal["2Dx3D"] = "2Dx3D",
        *,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        fc2_output_dtype: Type[cutlass.Numeric],
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        apply_topk_in_fc1: bool = False,
        gate_up_clamp: Optional[float] = None,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
    ) -> None:
        if not force_static_sched:
            raise NotImplementedError(
                "v1 only implements force_static_sched=True (lean 7-warp). "
                "Dynamic CLC (force_static_sched=False) is not wired here."
            )
        if sf_vec_size != Mxfp8BlockSize:
            raise NotImplementedError(
                f"SM120 MXFP8 requires sf_vec_size={Mxfp8BlockSize}; "
                f"got {sf_vec_size}."
            )
        if ab_dtype not in (cutlass.Float8E4M3FN, cutlass.Float8E5M2):
            raise NotImplementedError(
                "SM120 MXFP8 only supports Float8E4M3FN/Float8E5M2 operands; "
                f"got {ab_dtype}."
            )
        if use_2cta_instrs:
            raise NotImplementedError(
                "SM120 warp-level MMA path does not support the legacy 2-CTA mode."
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

        self.acc_dtype = acc_dtype
        self.mma_tiler_mnk = mma_tiler_mnk
        self.cluster_shape_mnk = cluster_shape_mnk
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = use_2cta_instrs
        self.force_static_sched = force_static_sched
        self.static_expert_shape = static_expert_shape
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages

        # Fused fc12 sched-side knobs
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.sf_padding_block = sf_padding_block
        self.load_balance_mode = load_balance_mode
        # Compile-time experiment: 0 keeps the original grouped order; C>0
        # emits C token tiles of FC1 followed by the same C tiles of FC2.
        stream_env = os.environ.get("MEGA_FC12_STREAMING", "0").strip().lower()
        if stream_env not in ("0", "1", "false", "true", "off", "on"):
            raise ValueError(
                "MEGA_FC12_STREAMING must be a boolean value, got "
                f"{stream_env!r}."
            )
        self.streaming_fc12 = stream_env in ("1", "true", "on")
        stream_tiles_env = os.environ.get("MEGA_FC12_STREAM_TILES")
        if stream_tiles_env is None:
            self.streaming_fc12_tiles = 1 if self.streaming_fc12 else 0
        else:
            self.streaming_fc12_tiles = int(stream_tiles_env)
            if self.streaming_fc12_tiles < 0:
                raise ValueError(
                    "MEGA_FC12_STREAM_TILES must be non-negative, got "
                    f"{self.streaming_fc12_tiles}."
                )
            self.streaming_fc12 = self.streaming_fc12_tiles > 0
        self.fc2_ready_bundle_k_tiles = int(
            os.environ.get("MEGA_FC2_READY_BUNDLE_K_TILES", "12")
        )
        if self.fc2_ready_bundle_k_tiles <= 0:
            raise ValueError(
                "MEGA_FC2_READY_BUNDLE_K_TILES must be positive, got "
                f"{self.fc2_ready_bundle_k_tiles}."
            )
        self.sf_vec_size = sf_vec_size
        self.scenario = scenario
        self.arch = "sm_120"

        self.ab_dtype = ab_dtype
        self.fc2_output_dtype = fc2_output_dtype
        packed_store_env = os.environ.get("MEGA_FC2_PACKED_STORE")
        if packed_store_env is None:
            self.fc2_packed_store = fc2_output_dtype is cutlass.BFloat16
        else:
            packed_store_env = packed_store_env.strip().lower()
            if packed_store_env not in (
                "0", "1", "false", "true", "off", "on"
            ):
                raise ValueError(
                    "MEGA_FC2_PACKED_STORE must be a boolean value, got "
                    f"{packed_store_env!r}."
                )
            self.fc2_packed_store = packed_store_env in ("1", "true", "on")
        if self.fc2_packed_store and fc2_output_dtype is not cutlass.BFloat16:
            raise NotImplementedError(
                "MEGA_FC2_PACKED_STORE currently requires BF16 output."
            )
        self.non_ubulk_fc2_store = non_ubulk_fc2_store
        self.in_kernel_fc2_reduce = in_kernel_fc2_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.gate_up_clamp = gate_up_clamp
        self.epi_flag_batch = epi_flag_batch

        self._validate_mma_tiler_and_cluster_shape()
        self.mma_tiler = mma_tiler_mnk

        # Subclasses set this before __call__ reaches _setup_attributes.
        self.enable_token_comm: bool = False
        # Keep the base fc12 entry on a compact SM120-native body while the
        # copied persistent body is replaced incrementally.
        self.use_inline_sm120_body: bool = True

        # SM120 design topology.  The copied SM100 body below still names
        # "epilogue_warp_id" and "mma_warp_id" separately; the real SM120
        # path uses compute_warp_id for both MMA and epilogue work.
        self.compute_warp_id = (0, 1, 2, 3)
        self.sm120_tma_a_warp_id = 4
        self.sm120_tma_b_warp_id = 5
        self.sm120_sched_warp_id = 6
        self.sm120_aux_warp_id = 7
        self.sm120_dispatch_warp_id = (8, 9, 10, 11)

        self.occupancy = 1
        self.epilogue_warp_id = self.compute_warp_id
        self.mma_warp_id = self.compute_warp_id[-1] + 1
        self.tma_a_warp_id = self.sm120_tma_a_warp_id
        self.tma_b_warp_id = self.sm120_tma_b_warp_id
        self.sched_warp_id = self.sm120_sched_warp_id
        # Installed by token-comm subclasses.
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False
        self.threads_per_cta = 32 * len(
            (
                *self.compute_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
            )
        )

        # Barrier 1 is reused by ordered FC1 epilogue rendezvous.
        self.epilog_sync_bar_id = 1
        # Aux warp publishes bundled FC1 readiness to the TMA-B warp.
        self.fc2_ready_bar_id = 2
        self.fc2_ready_work_bar_id = 3
        self.fc2_ready_bar_alt_id = 4
        self.fc2_ready_work_bar_alt_id = 5

        # MegaMoE-only register policy.  Lean/base fc12 keeps its original
        # register allocation because setmaxnreg emission is gated by
        # ``self.enable_token_comm`` inside the device kernel.
        self.epi_reg_cnt = 256
        self.task_reg_cnt = 72

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)
        self.num_tmem_alloc_cols = 0

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry for the SM120 MXFP8 swap-AB path."""
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if m != 64:
            raise ValueError(f"SM120 MXFP8 swap-AB starts with mma_tiler M=64; got {m}.")

        if n not in (32, 64, 128):
            raise ValueError(f"SM120 MXFP8 swap-AB supports N in (32,64,128); got {n}.")

        if k % 32 != 0:
            raise ValueError(
                f"SM120 MXFP8 K ({k}) must be a multiple of the m16n8k32 K atom."
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
        if self.a_dtype != self.b_dtype:
            raise NotImplementedError(
                "SM120 MXFP8 v1 expects A/B to have the same FP8 dtype; "
                f"got A={self.a_dtype}, B={self.b_dtype}."
            )
        tiled_mma = make_swapab_m64n8k128_tiled_mma(
            ab_dtype=self.a_dtype,
            acc_dtype=self.acc_dtype,
            sf_dtype=self.sf_dtype,
        )
        # The same warp-MMA atom partitions SFB.  Its N dimension is walked as
        # eight N8 slices in the compute loop, so no SM100-style rounded-N SFB
        # tiled MMA is needed.
        tiled_mma_sfb = make_swapab_m64n8k128_tiled_mma(
            ab_dtype=self.a_dtype,
            acc_dtype=self.acc_dtype,
            sf_dtype=self.sf_dtype,
        )
        return tiled_mma, tiled_mma_sfb

    @staticmethod
    def _round_up_to(value: int, align: int) -> int:
        return ((value + align - 1) // align) * align

    @staticmethod
    def _make_sm120_operand_smem_layout(
        tiled_mma: cute.TiledMma,
        tile_shape_mnk: Tuple[int, int, int],
        num_stages: int,
        *,
        is_a: bool,
    ) -> cute.Layout:
        """Plain staged SMEM layout for SM120 warp-level MMA operands.

        Generic Blackwell helpers derive swizzled layouts from instruction
        fields such as ``a_major_mode``. SM120 MXFP8 warp MMA does not expose
        those fields, so this path uses a compact MNK-stage layout and lets
        ``partition_A/B`` create the per-warp fragments.
        """
        if is_a:
            base_shape = tiled_mma.partition_shape_A(
                cute.dice(tile_shape_mnk, (1, None, 1))
            )
        else:
            base_shape = tiled_mma.partition_shape_B(
                cute.dice(tile_shape_mnk, (None, 1, 1))
            )
        atom_layout = cute.make_layout(base_shape[0])
        atom_elems = cute.size(base_shape[0])
        rest_mn_elems = cute.size(base_shape[1])
        base = cute.make_layout(
            base_shape,
            stride=(
                atom_layout.stride,
                atom_elems,
                atom_elems * rest_mn_elems,
            ),
        )
        return cute.append(
            base,
            cute.make_layout(
                num_stages, stride=cute.cosize(cute.filter_zeros(base))
            ),
        )

    @staticmethod
    def _make_sm120_data_smem_layout(
        tile_shape_mnk: Tuple[int, int, int],
        num_stages: int,
        *,
        operand_layout,
        dtype: Type[cutlass.Numeric],
        is_a: bool,
    ) -> cute.ComposedLayout:
        """SM120 TMA/ldmatrix-compatible data SMEM layout.

        This mirrors the Blackwell GeForce dense block-scaled GEMM examples:
        the top-level SMEM layout is the CTA operand tile ``(M,K)`` or
        ``(N,K)`` plus stage, and the inner swizzle comes from the standard
        SM90/SM120 shared-memory atom helper.
        """
        if is_a:
            smem_shape = cute.slice_(tile_shape_mnk, (None, 0, None))
            is_k_major = operand_layout.is_k_major_a()
            major_mode_size = tile_shape_mnk[2 if is_k_major else 0]
        else:
            smem_shape = cute.slice_(tile_shape_mnk, (0, None, None))
            is_k_major = operand_layout.is_k_major_b()
            major_mode_size = tile_shape_mnk[2 if is_k_major else 1]

        smem_layout_atom = cute.nvgpu.warpgroup.make_smem_layout_atom(
            sm90_utils.get_smem_layout_atom(
                operand_layout,
                dtype,
                major_mode_size,
            ),
            dtype,
        )
        return cute.tile_to_shape(
            smem_layout_atom,
            cute.append(smem_shape, num_stages),
            order=(0, 1, 2) if is_k_major else (1, 0, 2),
        )

    @staticmethod
    def _make_sm120_scale_rank4_smem_layout(scale_layout: cute.Layout) -> cute.Layout:
        """Wrap SM120 scale layout as (atom/rest_mn, dummy, rest_k, stage).

        ``blockscaled_layout.sm120_make_smem_layout_sf{a,b}`` returns a
        two-dimensional scale layout plus stage.  TMA helpers expect staged
        A/B layouts to expose three operand modes plus stage, so insert a
        zero-stride dummy rest-M/N mode.
        """
        return cute.make_layout(
            (
                scale_layout.shape[0],
                1,
                scale_layout.shape[1],
                scale_layout.shape[2],
            ),
            stride=(
                scale_layout.stride[0],
                0,
                scale_layout.stride[1],
                scale_layout.stride[2],
            ),
        )

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        Warp topology / ``threads_per_cta`` are fixed in ``__init__`` (the
        lean default here, the 12-warp MegaMoE layout in the token-comm
        subclass), so this method does not touch them.
        """
        self.mma_inst_shape_mn = (64, MMA_N)
        self.mma_inst_shape_mn_sfb = (
            self.mma_inst_shape_mn[0],
            self.mma_inst_shape_mn[1],
        )

        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        self.mma_tiler_sfa = (
            self._round_up_to(self.mma_tiler[0], 128),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.mma_tiler_sfb = (
            self.mma_tiler[0],
            self._round_up_to(self.mma_tiler[1], 128),
            self.mma_tiler[2],
        )
        self.cta_tile_shape_mnk = self.mma_tiler
        self.cta_tile_shape_mnk_sfb = (
            self.mma_tiler_sfb[0],
            self.mma_tiler_sfb[1],
            self.mma_tiler_sfb[2],
        )

        # SM120 warp-level MMA has no CTA-group/V split. Keep
        # the copied scheduler/TMA code's VMNK convention, but make V a single
        # logical CTA slot instead of deriving it from the warp MMA lane layout.
        self.cluster_layout_vmnk = cute.make_layout((1, *self.cluster_shape_mn, 1))
        self.cluster_layout_sfb_vmnk = cute.make_layout((1, *self.cluster_shape_mn, 1))

        # Multicast CTA counts
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.num_mcast_ctas_sfb = cute.size(self.cluster_layout_sfb_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1
        self.is_sfb_mcast = self.num_mcast_ctas_sfb > 1

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # FC1 epilogue SMEM: a reusable N16 x downproj32 FP32 reduction tile
        # plus the complete N64 x downproj32 FP8 STSM destination tile.  For the
        # default M64 x N64 tile this is 2KB + 2KB, half the old 8KB scratch.
        downproj_per_cta = self.mma_tiler[0] // 2
        c_bytes_total = (
            16 * downproj_per_cta * cutlass.Float32.width // 8
            + self.mma_tiler[1]
            * downproj_per_cta
            * self.fc1_output_dtype.width
            // 8
        )

        (
            self.num_acc_stage,
            max_num_ab_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            tiled_mma_sfb,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            self.sf_dtype,
            self.sf_vec_size,
            c_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
            self._smem_misc_budget_bytes(),
        )
        default_num_ab_stage = min(2, max_num_ab_stage)
        if self.enable_token_comm and getattr(self, "world_size", 1) == 1:
            # With no peer traffic, use the deepest feasible TMA/MMA pipeline.
            # Multi-rank caps the default at two stages to leave resources for
            # co-resident dispatch and token-back.
            default_num_ab_stage = max_num_ab_stage
        requested_num_ab_stage = int(
            os.environ.get("MEGA_NUM_AB_STAGES", str(default_num_ab_stage))
        )
        if requested_num_ab_stage < 1 or requested_num_ab_stage > max_num_ab_stage:
            raise ValueError(
                "MEGA_NUM_AB_STAGES must be in the SMEM-feasible range "
                f"[1, {max_num_ab_stage}], got {requested_num_ab_stage}."
            )
        self.num_ab_stage = requested_num_ab_stage
        print(
            f"[fc12 stages] num_ab_stage={self.num_ab_stage} "
            f"max_num_ab_stage={max_num_ab_stage} "
            f"num_acc_stage={self.num_acc_stage} "
            f"misc_budget={self._smem_misc_budget_bytes()} "
            f"c_bytes_total={c_bytes_total} smem_cap={self.smem_capacity} "
            f"token_back_standalone={self.token_back_standalone} "
            f"streaming_fc12={self.streaming_fc12} "
            f"stream_tiles={self.streaming_fc12_tiles}"
        )

        self.a_smem_layout_staged = self._make_sm120_data_smem_layout(
            self.mma_tiler,
            self.num_ab_stage,
            operand_layout=self.a_layout,
            dtype=self.a_dtype,
            is_a=True,
        )
        self.b_smem_layout_staged = self._make_sm120_data_smem_layout(
            self.mma_tiler,
            self.num_ab_stage,
            operand_layout=self.b_layout,
            dtype=self.b_dtype,
            is_a=False,
        )
        self.sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma,
            self.mma_tiler_sfa,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma_sfb,
            self.mma_tiler_sfb,
            self.sf_vec_size,
            self.num_ab_stage,
        )
        self.overlapping_accum = False
        self.num_acc_pipeline_stages = 0
        self.num_sfa_tmem_cols = 0
        self.num_sf_tmem_cols = 0
        self.num_accumulator_tmem_cols = 0

        # TMA load bytes per stage.  SM120 uses one independent TMA pipeline
        # for weight/SFA and one for activation/SFB so the two producer warps
        # never share a producer state machine.
        atom_thr_size = 1
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        sfa_copy_size = cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        sfb_copy_size = cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        if cutlass.const_expr(UseScaleTma):
            self.num_tma_a_load_bytes = (a_copy_size + sfa_copy_size) * atom_thr_size
            self.num_tma_b_load_bytes = (b_copy_size + sfb_copy_size) * atom_thr_size
        else:
            self.num_tma_a_load_bytes = a_copy_size * atom_thr_size
            self.num_tma_b_load_bytes = b_copy_size * atom_thr_size
        self.num_tma_load_bytes = (
            a_copy_size + b_copy_size + sfa_copy_size + sfb_copy_size
        ) * atom_thr_size

    def _smem_misc_budget_bytes(self) -> int:
        """SMEM bytes reserved for everything outside the AB / SF stage
        buffers and the ``sC`` epilogue staging.

        Hook for subclasses that need additional SMEM regions outside
        the base's main ``SharedStorage`` (e.g. MegaMoE dispatch warps
        allocate their own pull_buffer / pull_mbar / smem_expert_count
        via ``token_comm_extra_smem_storage_class``).  Subclass
        overrides add their region size to the returned value so
        ``_compute_stages`` properly subtracts it from the AB-stage
        SMEM budget.  Base default returns the 1024-byte
        miscellaneous reservation (mbarriers, sched work-tile buffer,
        TMEM allocator state).
        """
        return self._SmemMiscBudget

    def _compute_stages(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        mma_tiler_mnk: Tuple[int, int, int],
        a_dtype: Type[cutlass.Numeric],
        b_dtype: Type[cutlass.Numeric],
        sf_dtype: Type[cutlass.Numeric],
        sf_vec_size: int,
        c_bytes_total: int,
        smem_capacity: int,
        occupancy: int,
        num_sched_stages: int,
        misc_budget: int,
    ) -> Tuple[int, int, int]:
        """Compute stage counts for ACC, AB+SF, and scheduler.

        ``misc_budget`` is the byte count consumed by everything
        outside ``ab_bytes_per_stage * num_ab_stage + c_bytes_total``
        (mbarriers / sched work-tile buffer / TMEM allocator state in
        the lean path; plus the dispatch warps' pull_buffer / mbar /
        per-CTA expert histogram under MegaMoE).  Provided by the
        ``_smem_misc_budget_bytes`` hook so subclasses can extend the
        reservation without touching this helper.
        """
        num_acc_stage = 2

        # Use the exact layouts allocated by ``_setup_attributes`` below.
        # The logical MMA-fragment layouts are smaller than the swizzled
        # TMA/ldmatrix data layouts and also miss MXFP8 scale padding; using
        # them here can over-select the pipeline depth and exceed SM120's
        # 99-KiB dynamic-SMEM limit once MegaMoE dispatch scratch is added.
        a_smem_layout_stage_one = self._make_sm120_data_smem_layout(
            mma_tiler_mnk,
            1,
            operand_layout=self.a_layout,
            dtype=a_dtype,
            is_a=True,
        )
        b_smem_layout_staged_one = self._make_sm120_data_smem_layout(
            mma_tiler_mnk,
            1,
            operand_layout=self.b_layout,
            dtype=b_dtype,
            is_a=False,
        )
        mma_tiler_sfa = (
            Sm120SwapABSwigluMxfp8Fc12Kernel._round_up_to(mma_tiler_mnk[0], 128),
            mma_tiler_mnk[1],
            mma_tiler_mnk[2],
        )
        mma_tiler_sfb = (
            mma_tiler_mnk[0],
            Sm120SwapABSwigluMxfp8Fc12Kernel._round_up_to(mma_tiler_mnk[1], 128),
            mma_tiler_mnk[2],
        )
        sfa_smem_layout_staged_one = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_sfa,
            sf_vec_size,
            1,
        )
        sfb_smem_layout_staged_one = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma_sfb,
            mma_tiler_sfb,
            sf_vec_size,
            1,
        )

        ab_bytes_per_stage = (
            cute.size_in_bytes(a_dtype, a_smem_layout_stage_one)
            + cute.size_in_bytes(b_dtype, b_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfa_smem_layout_staged_one)
            + cute.size_in_bytes(sf_dtype, sfb_smem_layout_staged_one)
        )

        fixed_overhead = misc_budget + c_bytes_total

        num_ab_stage = (
            smem_capacity // occupancy - fixed_overhead
        ) // ab_bytes_per_stage
        num_ab_stage = min(num_ab_stage, 5)
        return num_acc_stage, num_ab_stage, num_sched_stages

    def get_workspace_size_in_bytes(
        self,
        fc1_activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Compute opaque workspace size for one fused fc1+fc2 launch."""
        sf_padding_block = self.sf_padding_block
        sf_vec_size = self.sf_vec_size

        mma_tiler_n = self.mma_tiler_mnk[1]

        data_total_rows, _hidden = fc1_activation_tensor.shape
        experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        intermediate_downproj = intermediate_gateup // 2

        # Conservative upper bound for sf_total_rows.
        sf_total_rows_upper = data_total_rows + experts * sf_padding_block

        # fc1_output: MXFP8 stores one element per byte.
        fc1_output_bytes = data_total_rows * intermediate_downproj

        # fc1_output_sf: SF atom layout rounds inner SF-block axis to 4.
        sf_block_cols = (
            (intermediate_downproj // sf_vec_size) + 3
        ) // 4 * 4
        fc1_output_sf_bytes = sf_total_rows_upper * sf_block_cols

        # fc1_done_counter: one Int32 per global token block, plus expert slack.
        counter_slots_upper = (
            (data_total_rows + mma_tiler_n - 1) // mma_tiler_n
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
            + fc1_output_sf_bytes
            + fc1_done_counter_bytes
            + load_balance_counter_bytes
        )

        # 128B align (TMA tensor base address alignment requirement).
        alignment = 128
        total = ((total + alignment - 1) // alignment) * alignment
        return total

    # =============================================================================
    # MegaMoE hooks (overridden by subclasses)
    # =============================================================================
    #
    # The base class never emits any MegaMoE-specific PTX -- all hooks below are
    # plain ``pass`` defaults, plus ``token_comm_extra_smem_storage_class`` which
    # returns ``None``.  Subclasses that fuse dispatch / combine override these
    # methods to (a) declare their extra SMEM struct, (b) acquire/peek the
    # dispatch->fc1 release counter, (c) emit the dispatch warps' work body,
    # (d) wire the kernel-tail rendezvous + cross-rank NVLink barrier.  No
    # MegaMoE workspace name (l1_*, %smid, NVLink slot id, ...) ever leaks
    # into the base; every such decision is the subclass's to make.
    #
    # Hooks are called from ``fc1fc2_kernel_impl`` and run inside ``@cute.kernel``
    # tracing, so they may issue PTX / TMA / NamedBarrier / spin_wait freely.
    # ``token_comm_args`` is forwarded as-is (the base never reads its fields).

    def token_comm_extra_smem_storage_class(self) -> Optional[type]:
        """Return an ``@cute.struct`` class for the subclass's extra SMEM
        region (= ``token_comm_storage``), or ``None`` if no extra SMEM is
        needed.  The base inner kernel allocates the returned struct
        adjacent to the main ``SharedStorage`` and forwards the resulting
        handle to ``token_comm_hook_dispatch_warp_body`` (the only hook
        that consumes it in the current design)."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return the pointer the sched-warp peek (inside
        ``Sm120SwapABMxfp8Fc12SchedExtension``) should watch as the
        dispatch->fc1 release counter, or ``None`` to disable the fc1
        phase peek entirely.  Called once at ext construction time."""
        return None

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Emitted on the sched warp BEFORE the late ``internal_init`` call.
        Default: no-op (lean path: there is nothing to wait for).
        MegaMoE: arrive_and_wait on the dispatch->sched NamedBarrier so the
        sched warp does not read ``expert_recv_count_sum`` (= sizes view)
        until this CTA's dispatch warps have walked through the cross-rank
        NVLink slot=0 acquire fence inside ``_dispatch_barrier``."""
        pass

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info,
    ):
        """Emitted on the TMA-B warp at the head of each fc1-phase task tile,
        before its K-loop.  Default: no-op.  MegaMoE: blocking spin on the
        dispatch->fc1 release counter at ``cumulative_token_block_count +
        tile_n_idx`` until it reaches ``work_tile_info.valid_tokens_in_tile``,
        unless ``work_tile_info.peek_ready`` already saturated it.  Skipping
        this in the lean path is correct because in the lean path the
        per-tile input is already resident in GMEM at launch time."""
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
        """Subclass dispatch warp body; no-op in the lean kernel."""
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
        """Subclass standalone token-back warp body; no-op in the lean kernel."""
        pass

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        """Subclass kernel-tail hook; no-op in the lean kernel."""
        pass

    @cute.jit
    def _launch_sm120_inline_fc12(
        self,
        activation: cute.Tensor,
        fc1_weight: cute.Tensor,
        activation_sf: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_output: cute.Tensor,
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        offs: Optional[cute.Tensor],
        max_active_clusters: cutlass.Constexpr,
        stream: cuda.CUstream,
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        load_balance_counter: Optional[cute.Tensor],
        expert_token_sizes: Optional[cute.Tensor],
    ) -> None:
        """Launch the compact SM120 swap-AB fc1 body owned by this class.

        This is the compile-first SM120 FC1+FC2 body:
        scheduler warp publishes one FC1 tile, warp 4 loads weight/SFA, warp 5
        loads activation/SFB, and compute warps 0-3 execute ldmatrix + MXFP8
        QMMA + FC1 SwiGLU/amax/quant.
        """
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
            fc2_output = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                    stride=fc2_output.stride,
                ),
            )

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
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )
        tokens_sum_padded = activation_sf.shape[0]
        hidden_padded = activation_sf.shape[1] * self.sf_vec_size
        activation_sf_gemm = cute.make_tensor(
            activation_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, hidden_padded, 1),
                self.sf_vec_size,
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
        intermediate_downproj = fc1_output.shape[1]
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens_sum, intermediate_downproj, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )
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
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * self.sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                self.sf_vec_size,
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

        self.a_layout = utils.LayoutEnum.from_tensor(fc1_weight_gemm)
        self.b_layout = utils.LayoutEnum.from_tensor(activation_gemm)
        self.a_dtype = fc1_weight_gemm.element_type
        self.b_dtype = activation_gemm.element_type
        self.sf_dtype = fc1_weight_sf_gemm.element_type
        self.fc1_output_dtype = fc1_output.element_type

        a_smem_layout_staged = self._make_sm120_data_smem_layout(
            self.mma_tiler_mnk,
            num_stages=1,
            operand_layout=self.a_layout,
            dtype=self.a_dtype,
            is_a=True,
        )
        b_smem_layout_staged = self._make_sm120_data_smem_layout(
            self.mma_tiler_mnk,
            num_stages=1,
            operand_layout=self.b_layout,
            dtype=self.b_dtype,
            is_a=False,
        )
        tiled_mma = make_swapab_m64n8k128_tiled_mma(
            ab_dtype=self.a_dtype,
            acc_dtype=cutlass.Float32,
            sf_dtype=cutlass.Float8E8M0FNU,
        )
        mma_tiler_sfa = (
            self._round_up_to(self.mma_tiler_mnk[0], 128),
            self.mma_tiler_mnk[1],
            self.mma_tiler_mnk[2],
        )
        mma_tiler_sfb = (
            self.mma_tiler_mnk[0],
            self._round_up_to(self.mma_tiler_mnk[1], 128),
            self.mma_tiler_mnk[2],
        )
        sfa_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfa(
            tiled_mma,
            mma_tiler_sfa,
            self.sf_vec_size,
            1,
        )
        sfb_smem_layout_staged = blockscaled_utils.sm120_make_smem_layout_sfb(
            tiled_mma,
            mma_tiler_sfb,
            self.sf_vec_size,
            1,
        )
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))

        tma_atom_a, tma_tensor_a = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc1_weight_gemm,
            cute.slice_(a_smem_layout_staged, (None, None, 0)),
            cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
            num_multicast=1,
        )
        tma_atom_b, tma_tensor_b = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            activation_gemm,
            cute.slice_(b_smem_layout_staged, (None, None, 0)),
            cute.slice_(self.mma_tiler_mnk, (0, None, None)),
            num_multicast=1,
        )
        tma_atom_sfa, tma_tensor_sfa = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc1_weight_sf_gemm,
            sfa_smem_layout,
            cute.slice_(mma_tiler_sfa, (None, 0, None)),
            num_multicast=1,
            internal_type=cutlass.Int16,
        )
        tma_atom_sfb, tma_tensor_sfb = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            activation_sf_gemm,
            sfb_smem_layout,
            cute.slice_(mma_tiler_sfb, (0, None, None)),
            num_multicast=1,
            internal_type=cutlass.Int16,
        )
        fc1_output_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.mma_tiler_mnk[1], self.mma_tiler_mnk[0] // 2),
            1,
        )
        tma_atom_fc2_weight, tma_tensor_fc2_weight = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc2_weight_gemm,
            cute.slice_(a_smem_layout_staged, (None, None, 0)),
            cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
            num_multicast=1,
        )
        tma_atom_fc1_output_as_fc2_input, tma_tensor_fc1_output_as_fc2_input = (
            cpasync.make_tiled_tma_atom(
                cpasync.CopyBulkTensorTileG2SOp(),
                fc1_output_gemm,
                cute.slice_(b_smem_layout_staged, (None, None, 0)),
                cute.slice_(self.mma_tiler_mnk, (0, None, None)),
                num_multicast=1,
            )
        )
        tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc2_weight_sf_gemm,
            sfa_smem_layout,
            cute.slice_(mma_tiler_sfa, (None, 0, None)),
            num_multicast=1,
            internal_type=cutlass.Int16,
        )
        (
            tma_atom_fc1_output_sf_as_fc2_input,
            tma_tensor_fc1_output_sf_as_fc2_input,
        ) = cpasync.make_tiled_tma_atom(
            cpasync.CopyBulkTensorTileG2SOp(),
            fc1_output_sf_gemm_for_fc2_load,
            sfb_smem_layout,
            cute.slice_(mma_tiler_sfb, (0, None, None)),
            num_multicast=1,
            internal_type=cutlass.Int16,
        )

        a_stage_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_stage_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        tma_a_copy_bytes = (
            cute.size_in_bytes(self.a_dtype, a_stage_layout)
            + cute.size_in_bytes(self.sf_dtype, sfa_smem_layout)
        )
        tma_b_copy_bytes = (
            cute.size_in_bytes(self.b_dtype, b_stage_layout)
            + cute.size_in_bytes(self.sf_dtype, sfb_smem_layout)
        )
        fc1_swiglu_layout = cute.make_layout(
            (16, self.mma_tiler_mnk[0] // 2),
            stride=(self.mma_tiler_mnk[0] // 2, 1),
        )

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
                "Exactly one of `offs` / `expert_token_sizes` must be provided."
            )
        sched_params = MoEFusedFc12SchedulerParams(
            scenario=self.scenario,
            expert_shape=(experts, intermediate_gateup, hidden),
            cta_tile_shape_mnk=self.mma_tiler_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=load_balance_counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=True,
            expert_token_prefix_sum=offs,
            expert_token_sizes=expert_token_sizes,
            streaming_fc12=self.streaming_fc12,
            streaming_fc12_tiles=self.streaming_fc12_tiles,
        )
        grid = sched_params.get_grid_shape(max_active_clusters)
        # FC1 epilogue publishes one arrival per compute warp for each
        # intermediate tile in the same token block.  FC2 may read the
        # workspace only after all compute-warps have written their slices.
        ext_fc2_spin_threshold = (
            (
                fc1_weight_gemm.shape[0] + self.mma_tiler_mnk[0] - 1
            )
            // self.mma_tiler_mnk[0]
        ) * len(self.compute_warp_id)
        sched_ext = Sm120SwapABMxfp8Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=None,
        )
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(
            sched_params,
            sched_ext,
            num_drain_warps=0,
        )

        @cute.struct
        class SharedStorage:
            pipeline_mbar: cute.struct.MemRange[cutlass.Int64, 4]
            sched_storage: SchedStorage
            sA: cute.struct.Align[
                cute.struct.MemRange[self.a_dtype, cute.cosize(a_smem_layout_staged)],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[self.b_dtype, cute.cosize(b_smem_layout_staged)],
                128,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfa_smem_layout_staged)],
                128,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[self.sf_dtype, cute.cosize(sfb_smem_layout_staged)],
                128,
            ]
            sFc1Swiglu: cute.struct.Align[
                cute.struct.MemRange[cutlass.Float32, cute.cosize(fc1_swiglu_layout)],
                128,
            ]
            sFc1Output: cute.struct.Align[
                cute.struct.MemRange[
                    self.fc1_output_dtype,
                    cute.cosize(fc1_output_smem_layout_staged),
                ],
                128,
            ]

        self.shared_storage = SharedStorage
        self.sm120_inline_fc12_kernel(
            fc2_output,
            fc1_done_counter,
            fc1_output,
            fc1_output_sf,
            tma_atom_a,
            tma_atom_b,
            tma_tensor_a,
            tma_tensor_b,
            tma_atom_sfa,
            tma_atom_sfb,
            tma_tensor_sfa,
            tma_tensor_sfb,
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc1_output_as_fc2_input,
            tma_tensor_fc1_output_as_fc2_input,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            tma_atom_fc1_output_sf_as_fc2_input,
            tma_tensor_fc1_output_sf_as_fc2_input,
            topk_scores,
            sched_params,
            sched_ext,
            tiled_mma,
            mma_tiler_sfa,
            mma_tiler_sfb,
            a_smem_layout_staged,
            b_smem_layout_staged,
            sfa_smem_layout_staged,
            sfb_smem_layout_staged,
            fc1_output_smem_layout_staged,
            fc1_swiglu_layout,
            tma_a_copy_bytes,
            tma_b_copy_bytes,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.kernel
    def sm120_inline_fc12_kernel(
        self,
        fc2_output: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        fc1_output: cute.Tensor,
        fc1_output_sf: cute.Tensor,
        tma_atom_a: cute.CopyAtom,
        tma_atom_b: cute.CopyAtom,
        mA_mkl: cute.Tensor,
        mB_nkl: cute.Tensor,
        tma_atom_sfa: cute.CopyAtom,
        tma_atom_sfb: cute.CopyAtom,
        mSFA_mkl: cute.Tensor,
        mSFB_nkl: cute.Tensor,
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc1_output_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_as_fc2_input: cute.Tensor,
        tma_atom_fc2_weight_sf: cute.CopyAtom,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        tma_atom_fc1_output_sf_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_sf_as_fc2_input: cute.Tensor,
        topk_scores: cute.Tensor,
        sched_params: MoEFusedFc12SchedulerParams,
        sched_ext: Sm120SwapABMxfp8Fc12SchedExtension,
        tiled_mma: cute.TiledMma,
        mma_tiler_sfa: cutlass.Constexpr,
        mma_tiler_sfb: cutlass.Constexpr,
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        fc1_output_smem_layout_staged: cute.ComposedLayout,
        fc1_swiglu_layout: cute.Layout,
        tma_a_copy_bytes: cutlass.Constexpr,
        tma_b_copy_bytes: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sFc1Swiglu = storage.sFc1Swiglu.get_tensor(fc1_swiglu_layout)
        sFc1OutputStaged = storage.sFc1Output.get_tensor(
            fc1_output_smem_layout_staged.outer,
            swizzle=fc1_output_smem_layout_staged.inner,
        )
        sFc1Output = cute.slice_(sFc1OutputStaged, (None, None, 0))

        producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread)
        consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.compute_warp_id)
        )
        a_producer, a_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_a_copy_bytes,
            barrier_storage=storage.pipeline_mbar.data_ptr(),
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        ).make_participants()
        b_producer, b_consumer = pipeline.PipelineTmaAsync.create(
            num_stages=1,
            producer_group=producer_group,
            consumer_group=consumer_group,
            tx_count=tma_b_copy_bytes,
            barrier_storage=storage.pipeline_mbar.data_ptr() + 2,
            cta_layout_vmnk=cute.make_layout((1, 1, 1, 1)),
        ).make_participants()

        SchedCls = sched_params.get_scheduler_type()
        scheduler = SchedCls.create(
            sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            sched_storage=storage.sched_storage,
            num_consumer_threads=32
            * (
                len(self.compute_warp_id)
                + 2
            ),
            ext=sched_ext,
        )
        sched_consumer = scheduler.make_consumer()

        cta_layout_mnk = cute.make_layout((1, 1, 1))
        a_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (0, None, 0)).shape
        )
        b_cta_layout = cute.make_layout(
            cute.slice_(cta_layout_mnk, (None, 0, 0)).shape
        )

        mma_tidx = tidx % cutlass.Int32(128)
        thr_mma = tiled_mma.get_slice(mma_tidx)
        tCsA = thr_mma.partition_A(sA)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
        tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
        tCrSFA = sm100_utils.partition_fragment_SFA(
            sSFA[None, None, 0],
            thr_mma,
            mma_tidx,
        )
        tCrSFB = sm100_utils.partition_fragment_SFB(
            sSFB[None, None, 0],
            thr_mma,
            mma_tidx,
        )
        atom_copy_ldmatrix_A = make_sm120_ldmatrix_atom(
            self.a_dtype,
            transpose=self.a_layout.is_m_major_a(),
        )
        atom_copy_ldmatrix_B = make_sm120_ldmatrix_atom(
            self.b_dtype,
            transpose=self.b_layout.is_n_major_b(),
        )
        smem_tiled_copy_A = cute.make_tiled_copy_A(atom_copy_ldmatrix_A, tiled_mma)
        smem_tiled_copy_B = cute.make_tiled_copy_B(atom_copy_ldmatrix_B, tiled_mma)
        thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(mma_tidx)
        thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(mma_tidx)
        tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
        tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
        tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
        tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)

        atom_copy_scale = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.sf_dtype,
        )
        smem_tiled_copy_SFA = cute.make_tiled_copy(
            atom_copy_scale,
            sm100_utils.get_layoutSFA_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[0]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        smem_tiled_copy_SFB = cute.make_tiled_copy(
            atom_copy_scale,
            sm100_utils.get_layoutSFB_TV(tiled_mma),
            (
                cute.size(tiled_mma.permutation_mnk[1]),
                cute.size(tiled_mma.permutation_mnk[2]),
            ),
        )
        thr_copy_SFA = smem_tiled_copy_SFA.get_slice(mma_tidx)
        thr_copy_SFB = smem_tiled_copy_SFB.get_slice(mma_tidx)
        tCsSFA_copy_view = thr_copy_SFA.partition_S(sSFA)
        tCsSFB_copy_view = thr_copy_SFB.partition_S(sSFB)
        tCrSFA_copy_view = thr_copy_SFA.retile(tCrSFA)
        tCrSFB_copy_view = thr_copy_SFB.retile(tCrSFB)

        acc_shape_mn8 = tiled_mma.partition_shape_C((self.mma_tiler_mnk[0], MMA_N))
        acc0 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc1 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc2 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc3 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc4 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc5 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc6 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc7 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc8 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc9 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc10 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc11 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc12 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc13 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc14 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        acc15 = cute.make_rmem_tensor(acc_shape_mn8, cutlass.Float32)
        accs = (
            acc0,
            acc1,
            acc2,
            acc3,
            acc4,
            acc5,
            acc6,
            acc7,
            acc8,
            acc9,
            acc10,
            acc11,
            acc12,
            acc13,
            acc14,
            acc15,
        )
        downproj_per_cta: cutlass.Constexpr = self.mma_tiler_mnk[0] // 2
        n_groups: cutlass.Constexpr = self.mma_tiler_mnk[1] // MMA_N

        mma_tiler_k = self.mma_tiler_mnk[2]
        k_tile_cnt_fc1 = (mA_mkl.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (
            tma_tensor_fc2_weight.shape[1] + mma_tiler_k - 1
        ) // mma_tiler_k

        if warp_idx == self.sched_warp_id:
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                sched_ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                scheduler.publish_work()
                scheduler.gen_next_work()
            scheduler.publish_work()
            scheduler.produce_tail()

        # Complete the 4-warp setmaxnreg group containing TMA-A, TMA-B, and
        # scheduler warps.  This reserved warp has no work-scheduler role.
        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx == self.sm120_aux_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

        if warp_idx == self.tma_a_warp_id:
            cpasync.prefetch_descriptor(tma_atom_a)
            cpasync.prefetch_descriptor(tma_atom_sfa)
            cpasync.prefetch_descriptor(tma_atom_fc2_weight)
            cpasync.prefetch_descriptor(tma_atom_fc2_weight_sf)
            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    real_a, _ = sched_ext.get_gmem_tensor(
                        "a", mA_mkl, work_tile_info,
                    )
                    real_sfa, _ = sched_ext.get_gmem_tensor(
                        "sfa", mSFA_mkl, work_tile_info,
                    )
                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(mma_tiler_sfa, (None, 0, None)),
                        (None, None, None),
                    )
                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_a,
                        cutlass.Int32(0),
                        a_cta_layout,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA_mkl, 0, 2),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_sfa,
                        cutlass.Int32(0),
                        a_cta_layout,
                        cute.group_modes(sSFA, 0, 2),
                        cute.group_modes(gSFA_mkl, 0, 2),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)
                    tAgA_slice = tAgA[(None, work_tile_info.tile_m_idx, None, 0)]
                    sfa_tile_m = work_tile_info.tile_m_idx
                    if cutlass.const_expr(mma_tiler_sfa[0] != self.mma_tiler_mnk[0]):
                        sfa_tile_m = work_tile_info.tile_m_idx // cutlass.Int32(
                            mma_tiler_sfa[0] // self.mma_tiler_mnk[0]
                        )
                    tAgSFA_slice = tAgSFA[(None, sfa_tile_m, None, 0)]
                    a_producer.reset()
                    peek_empty = a_producer.try_acquire()
                    for k_tile in cutlass.range(0, k_tile_cnt_fc1, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(peek_empty)
                        peek_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt_fc1:
                            peek_empty = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_a,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        cute.copy(
                            tma_atom_sfa,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                else:
                    real_a, _ = sched_ext.get_gmem_tensor(
                        "a", tma_tensor_fc2_weight, work_tile_info,
                    )
                    real_sfa, _ = sched_ext.get_gmem_tensor(
                        "sfa", tma_tensor_fc2_weight_sf, work_tile_info,
                    )
                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler_mnk, (None, 0, None)),
                        (None, None, None),
                    )
                    gSFA_mkl = cute.local_tile(
                        real_sfa,
                        cute.slice_(mma_tiler_sfa, (None, 0, None)),
                        (None, None, None),
                    )
                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        cutlass.Int32(0),
                        a_cta_layout,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA_mkl, 0, 2),
                    )
                    tAsSFA, tAgSFA = cpasync.tma_partition(
                        tma_atom_fc2_weight_sf,
                        cutlass.Int32(0),
                        a_cta_layout,
                        cute.group_modes(sSFA, 0, 2),
                        cute.group_modes(gSFA_mkl, 0, 2),
                    )
                    tAsSFA = cute.filter_zeros(tAsSFA)
                    tAgSFA = cute.filter_zeros(tAgSFA)
                    tAgA_slice = tAgA[(None, work_tile_info.tile_m_idx, None, 0)]
                    sfa_tile_m = work_tile_info.tile_m_idx
                    if cutlass.const_expr(mma_tiler_sfa[0] != self.mma_tiler_mnk[0]):
                        sfa_tile_m = work_tile_info.tile_m_idx // cutlass.Int32(
                            mma_tiler_sfa[0] // self.mma_tiler_mnk[0]
                        )
                    tAgSFA_slice = tAgSFA[(None, sfa_tile_m, None, 0)]
                    a_producer.reset()
                    peek_empty = a_producer.try_acquire()
                    for k_tile in cutlass.range(0, k_tile_cnt_fc2, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(peek_empty)
                        peek_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt_fc2:
                            peek_empty = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_weight,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        cute.copy(
                            tma_atom_fc2_weight_sf,
                            tAgSFA_slice[(None, handle.count)],
                            tAsSFA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                work_tile_info = sched_consumer.consume_work()
            a_producer.tail()

        if warp_idx == self.tma_b_warp_id:
            cpasync.prefetch_descriptor(tma_atom_b)
            cpasync.prefetch_descriptor(tma_atom_sfb)
            cpasync.prefetch_descriptor(tma_atom_fc1_output_as_fc2_input)
            cpasync.prefetch_descriptor(tma_atom_fc1_output_sf_as_fc2_input)
            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    real_b, _ = sched_ext.get_gmem_tensor(
                        "b", mB_nkl, work_tile_info,
                    )
                    real_sfb, _ = sched_ext.get_gmem_tensor(
                        "sfb", mSFB_nkl, work_tile_info,
                    )
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler_mnk, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_b,
                        cutlass.Int32(0),
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nkl, 0, 2),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_sfb,
                        cutlass.Int32(0),
                        b_cta_layout,
                        cute.group_modes(sSFB, 0, 2),
                        cute.group_modes(gSFB_nkl, 0, 2),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)
                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    sfb_tile_n = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.mma_tiler_mnk[1] < 128):
                        sfb_tile_n = work_tile_info.tile_n_idx // cutlass.Int32(
                            128 // self.mma_tiler_mnk[1]
                        )
                    tBgSFB_slice = tBgSFB[(None, sfb_tile_n, None, 0)]
                    b_producer.reset()
                    peek_empty = b_producer.try_acquire()
                    for k_tile in cutlass.range(0, k_tile_cnt_fc1, 1, unroll=1):
                        handle = b_producer.acquire_and_advance(peek_empty)
                        peek_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt_fc1:
                            peek_empty = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_b,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        cute.copy(
                            tma_atom_sfb,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                else:
                    counter_slot = (
                        work_tile_info.cumulative_data_physical_row
                        // cutlass.Int32(self.mma_tiler_mnk[1])
                        + work_tile_info.tile_n_idx
                    )
                    spin_wait(
                        fc1_done_counter.iterator + counter_slot,
                        lambda v: v >= sched_ext.fc2_spin_threshold,
                    )
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.fence_proxy("async.global")
                    real_b, _ = sched_ext.get_gmem_tensor(
                        "b", tma_tensor_fc1_output_as_fc2_input, work_tile_info,
                    )
                    real_sfb, _ = sched_ext.get_gmem_tensor(
                        "sfb",
                        tma_tensor_fc1_output_sf_as_fc2_input,
                        work_tile_info,
                    )
                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler_mnk, (0, None, None)),
                        (None, None, None),
                    )
                    gSFB_nkl = cute.local_tile(
                        real_sfb,
                        cute.slice_(mma_tiler_sfb, (0, None, None)),
                        (None, None, None),
                    )
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc1_output_as_fc2_input,
                        cutlass.Int32(0),
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nkl, 0, 2),
                    )
                    tBsSFB, tBgSFB = cpasync.tma_partition(
                        tma_atom_fc1_output_sf_as_fc2_input,
                        cutlass.Int32(0),
                        b_cta_layout,
                        cute.group_modes(sSFB, 0, 2),
                        cute.group_modes(gSFB_nkl, 0, 2),
                    )
                    tBsSFB = cute.filter_zeros(tBsSFB)
                    tBgSFB = cute.filter_zeros(tBgSFB)
                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    sfb_tile_n = work_tile_info.tile_n_idx
                    if cutlass.const_expr(self.mma_tiler_mnk[1] < 128):
                        sfb_tile_n = work_tile_info.tile_n_idx // cutlass.Int32(
                            128 // self.mma_tiler_mnk[1]
                        )
                    tBgSFB_slice = tBgSFB[(None, sfb_tile_n, None, 0)]
                    b_producer.reset()
                    peek_empty = b_producer.try_acquire()
                    for k_tile in cutlass.range(0, k_tile_cnt_fc2, 1, unroll=1):
                        handle = b_producer.acquire_and_advance(peek_empty)
                        peek_empty = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt_fc2:
                            peek_empty = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_output_as_fc2_input,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                        cute.copy(
                            tma_atom_fc1_output_sf_as_fc2_input,
                            tBgSFB_slice[(None, handle.count)],
                            tBsSFB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                        )
                work_tile_info = sched_consumer.consume_work()
            b_producer.tail()

        if warp_idx < cutlass.Int32(4):
            lane_idx = cute.arch.lane_idx()
            lane_g = lane_idx >> cutlass.Int32(2)
            lane_t = lane_idx & cutlass.Int32(3)
            compute_warp = warp_idx
            warp_dp_base = compute_warp * cutlass.Int32(SWAP_AB_INTERLEAVE)
            dp_in_cta = warp_dp_base + lane_g
            stsm_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix16x8x8bOp(
                    transpose=True,
                    num_matrices=1,
                ),
                self.fc1_output_dtype,
            )
            stsm_tiled_copy = cute.make_tiled_copy_C_atom(stsm_atom, tiled_mma)
            stsm_thr_copy = stsm_tiled_copy.get_slice(mma_tidx)
            stsm_remap_layout = cute.make_layout(
                ((16, 4), MMA_N),
                stride=(
                    (1, 16 * SWAP_AB_INTERLEAVE),
                    16,
                ),
            )
            sFc1OutputAtom = cute.composition(
                cute.local_tile(sFc1Output, (16, downproj_per_cta), (0, 0)),
                stsm_remap_layout,
            )
            rFc1Output = cute.make_rmem_tensor(
                cute.shape(stsm_thr_copy.partition_S(sFc1OutputAtom)),
                self.fc1_output_dtype,
            )
            rFc1OutputPacked = cute.recast_tensor(rFc1Output, cutlass.Int32)
            rFc1OutputLocal = cute.make_rmem_tensor((4,), cutlass.Float32)
            rFc1Store = cute.make_rmem_tensor((4,), self.fc1_output_dtype)
            rFc1StoreI32 = cute.recast_tensor(rFc1Store, cutlass.Int32)
            work_tile_info = sched_consumer.consume_work()
            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                k_tile_cnt = k_tile_cnt_fc2
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                for ng in cutlass.range_constexpr(0, n_groups):
                    accs[ng].fill(0.0)
                a_consumer.reset()
                b_consumer.reset()
                peek_a = a_consumer.try_wait()
                peek_b = b_consumer.try_wait()
                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle_a = a_consumer.wait_and_advance(peek_a)
                    handle_b = b_consumer.wait_and_advance(peek_b)
                    peek_a = cutlass.Boolean(1)
                    peek_b = cutlass.Boolean(1)
                    if handle_a.count + 1 < k_tile_cnt:
                        peek_a = a_consumer.try_wait()
                    if handle_b.count + 1 < k_tile_cnt:
                        peek_b = b_consumer.try_wait()
                    tCsA_p = tCsA_copy_view[None, None, None, handle_a.index]
                    tCsB_p = tCsB_copy_view[None, None, None, handle_b.index]
                    tCsSFA_p = tCsSFA_copy_view[None, None, None, handle_a.index]
                    tCsSFB_p = tCsSFB_copy_view[None, None, None, handle_b.index]
                    sfa_m_group = work_tile_info.tile_m_idx % cutlass.Int32(
                        mma_tiler_sfa[0] // self.mma_tiler_mnk[0]
                    )
                    # The staged SFA tile covers two M64 weight-scale groups.
                    # Move the runtime group selection into the SMEM address;
                    # the compact RMEM fragment can then use constant group 0.
                    tCsSFA_selected = cute.make_tensor(
                        tCsSFA_p.iterator + sfa_m_group * cutlass.Int32(8),
                        tCsSFA_p.layout,
                    )
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_selected)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                    tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered[None, 0, 0],
                        tCrSFA_copy_view_filtered[None, 0, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered[None, None, 0],
                        tCrSFB_copy_view_filtered[None, 0, 0, None],
                    )

                    tCrSFB_mma = tCrSFB
                    if cutlass.const_expr(self.mma_tiler_mnk[1] < 128):
                        # SFB is staged as N128; select this CTA tile's
                        # register subfragment before issuing its N groups.
                        sfb_tiles_per_tma = 128 // self.mma_tiler_mnk[1]
                        sfb_fragment_shift = (
                            work_tile_info.tile_n_idx
                            % cutlass.Int32(sfb_tiles_per_tma)
                        ) * cutlass.Int32(n_groups // 4)
                        tCrSFB_mma = cute.make_tensor(
                            tCrSFB.iterator + sfb_fragment_shift,
                            tCrSFB.layout,
                        )
                    for k_inner_mma in cutlass.range_constexpr(0, 4):
                        if cutlass.const_expr(k_inner_mma + 1 < 4):
                            k_inner_next = k_inner_mma + 1
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_inner_next],
                                tCrA_copy_view[None, None, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_inner_next],
                                tCrB_copy_view[None, None, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, 0, k_inner_next],
                                tCrSFA_copy_view_filtered[None, 0, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, k_inner_next],
                                tCrSFB_copy_view_filtered[
                                    None, 0, k_inner_next, None
                                ],
                            )
                        for ng in cutlass.range_constexpr(0, n_groups):
                            issue_m64n8k32_mxfp8(
                                tiled_mma,
                                accs[ng],
                                tCrA,
                                tCrB,
                                tCrSFA,
                                tCrSFB_mma,
                                n_group=ng,
                                active_n_groups=n_groups,
                                sfa_m_group=0,
                                k_inner=k_inner_mma,
                                ab_dtype=self.ab_dtype,
                                sf_dtype=self.sf_dtype,
                            )
                    handle_a.release()
                    handle_b.release()

                if is_phase_linear1:
                    fc1_col_base = (
                        work_tile_info.tile_m_idx * cutlass.Int32(downproj_per_cta)
                    )
                    tile_token_base = work_tile_info.tile_n_idx * cutlass.Int32(
                        self.mma_tiler_mnk[1]
                    )
                    # This lean path computes the physical atom offset by
                    # hand below, so it uses the compact scale-block index.
                    sf_col = fc1_col_base // cutlass.Int32(self.sf_vec_size)
                    rcp_limit = Fp8E4M3RcpLimit
                    if cutlass.const_expr(self.fc1_output_dtype == cutlass.Float8E5M2):
                        rcp_limit = Fp8E5M2RcpLimit
                    q_limit = cutlass.Float32(1.0 / rcp_limit)
                    sf_cols = cutlass.Int32(fc1_output_sf.shape[1])
                    sf_k_atoms = (sf_cols + cutlass.Int32(3)) // cutlass.Int32(4)
                    rAmax = cute.make_rmem_tensor((4,), cutlass.Float32)

                    for ng_pair in cutlass.range_constexpr(0, n_groups // 2):
                        acc0 = accs[ng_pair * 2]
                        acc1 = accs[ng_pair * 2 + 1]
                        token0 = lane_t * cutlass.Int32(2)
                        token1 = token0 + cutlass.Int32(1)
                        token2 = token0 + cutlass.Int32(MMA_N)
                        token3 = token1 + cutlass.Int32(MMA_N)
                        pair_base = cutlass.Int32(ng_pair * 2 * MMA_N)

                        gate0 = acc0[0]
                        gate1 = acc0[1]
                        gate2 = acc1[0]
                        gate3 = acc1[1]
                        val0 = acc0[2] * gate0 * cute.arch.rcp_approx(
                            cute.math.exp2(gate0 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val1 = acc0[3] * gate1 * cute.arch.rcp_approx(
                            cute.math.exp2(gate1 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val2 = acc1[2] * gate2 * cute.arch.rcp_approx(
                            cute.math.exp2(gate2 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val3 = acc1[3] * gate3 * cute.arch.rcp_approx(
                            cute.math.exp2(gate3 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        if pair_base + token0 >= work_tile_info.valid_tokens_in_tile:
                            val0 = cutlass.Float32(0.0)
                        if pair_base + token1 >= work_tile_info.valid_tokens_in_tile:
                            val1 = cutlass.Float32(0.0)
                        if pair_base + token2 >= work_tile_info.valid_tokens_in_tile:
                            val2 = cutlass.Float32(0.0)
                        if pair_base + token3 >= work_tile_info.valid_tokens_in_tile:
                            val3 = cutlass.Float32(0.0)
                        amax0 = cute.math.absf(val0)
                        amax1 = cute.math.absf(val1)
                        amax2 = cute.math.absf(val2)
                        amax3 = cute.math.absf(val3)
                        for xor_mask in (4, 8, 16):
                            peer_lane = lane_idx ^ cutlass.Int32(xor_mask)
                            amax0 = cute.arch.fmax(
                                amax0, cute.arch.shuffle_sync(amax0, peer_lane)
                            )
                            amax1 = cute.arch.fmax(
                                amax1, cute.arch.shuffle_sync(amax1, peer_lane)
                            )
                            amax2 = cute.arch.fmax(
                                amax2, cute.arch.shuffle_sync(amax2, peer_lane)
                            )
                            amax3 = cute.arch.fmax(
                                amax3, cute.arch.shuffle_sync(amax3, peer_lane)
                            )
                        if lane_g == cutlass.Int32(0):
                            sFc1Swiglu[token0, compute_warp] = amax0
                            sFc1Swiglu[token1, compute_warp] = amax1
                            sFc1Swiglu[token2, compute_warp] = amax2
                            sFc1Swiglu[token3, compute_warp] = amax3

                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=32 * len(self.compute_warp_id),
                        )
                        for i in cutlass.range_constexpr(0, 4):
                            local_token = (
                                lane_t * cutlass.Int32(2)
                                + cutlass.Int32(i & 1)
                                + cutlass.Int32((i >> 1) * MMA_N)
                            )
                            amax = cutlass.Float32(0.0)
                            if lane_g == cutlass.Int32(0):
                                for warp in cutlass.range_constexpr(
                                    0, len(self.compute_warp_id)
                                ):
                                    amax = cute.arch.fmax(
                                        amax,
                                        sFc1Swiglu[local_token, warp],
                                    )
                            rAmax[i] = cute.arch.shuffle_sync(amax, lane_t)

                        for i in cutlass.range_constexpr(0, 4):
                            local_token = (
                                lane_t * cutlass.Int32(2)
                                + cutlass.Int32(i & 1)
                                + cutlass.Int32((i >> 1) * MMA_N)
                            )
                            tile_token = pair_base + local_token
                            raw_token = tile_token_base + tile_token
                            amax = rAmax[i]
                            scale = cvt_f32_to_f8_to_f32(
                                amax * cutlass.Float32(rcp_limit),
                                self.sf_dtype,
                            )
                            inv = cute.arch.fmin(
                                cute.arch.rcp_approx(scale),
                                cutlass.Float32(Fp32Max),
                            )
                            if lane_g == cutlass.Int32(0):
                                if tile_token < work_tile_info.valid_tokens_in_tile:
                                    sf_flat = (
                                        (raw_token // cutlass.Int32(128))
                                        * cutlass.Int32(512)
                                        * sf_k_atoms
                                        + (raw_token % cutlass.Int32(32))
                                        * cutlass.Int32(16)
                                        + (
                                            (raw_token // cutlass.Int32(32))
                                            % cutlass.Int32(4)
                                        )
                                        * cutlass.Int32(4)
                                        + (sf_col % cutlass.Int32(4))
                                        + (sf_col // cutlass.Int32(4))
                                        * cutlass.Int32(512)
                                    )
                                    sf_row = (
                                        work_tile_info.cumulative_sf_physical_row
                                        + sf_flat // sf_cols
                                    )
                                    sf_col_physical = (
                                        sf_flat - (sf_flat // sf_cols) * sf_cols
                                    )
                                    fc1_output_sf[sf_row, sf_col_physical] = (
                                        amax * cutlass.Float32(rcp_limit)
                                    ).to(self.sf_dtype)

                            value = val0
                            if cutlass.const_expr(i == 1):
                                value = val1
                            if cutlass.const_expr(i == 2):
                                value = val2
                            if cutlass.const_expr(i == 3):
                                value = val3
                            q = cutlass.Float32(0.0)
                            if tile_token < work_tile_info.valid_tokens_in_tile:
                                q = cute.arch.fmin(
                                    q_limit,
                                    cute.arch.fmax(
                                        -q_limit,
                                        value * inv,
                                    ),
                                )
                            rFc1OutputLocal[i] = q

                        transpose_src_lane = (
                            (lane_g >> cutlass.Int32(1))
                            + lane_t * cutlass.Int32(8)
                        )
                        transpose_src_lane_1 = (
                            transpose_src_lane + cutlass.Int32(4)
                        )
                        packed_local = cutlass.Int32(
                            cvt_f32x4_to_f8x4_pack_i32(
                                rFc1OutputLocal,
                                self.fc1_output_dtype,
                            )
                        )
                        packed0 = cute.arch.shuffle_sync(
                            packed_local, transpose_src_lane
                        )
                        packed1 = cute.arch.shuffle_sync(
                            packed_local, transpose_src_lane_1
                        )
                        packed_output = (
                            (packed0 & cutlass.Int32(0x000000FF))
                            | (
                                (packed1 & cutlass.Int32(0x000000FF))
                                << cutlass.Int32(8)
                            )
                            | (packed0 & cutlass.Int32(0x00FF0000))
                            | (
                                (packed1 & cutlass.Int32(0x00FF0000))
                                << cutlass.Int32(8)
                            )
                        )
                        if (lane_g & cutlass.Int32(1)) != cutlass.Int32(0):
                            packed_output = (
                                ((packed0 >> cutlass.Int32(8)) & cutlass.Int32(0xFF))
                                | (packed1 & cutlass.Int32(0x0000FF00))
                                | (
                                    ((packed0 >> cutlass.Int32(24)) & cutlass.Int32(0xFF))
                                    << cutlass.Int32(16)
                                )
                                | (packed1 & cutlass.Int32(-0x01000000))
                            )
                        rFc1OutputPacked[0] = packed_output

                        sFc1OutputPair = cute.composition(
                            cute.local_tile(
                                sFc1Output,
                                (16, downproj_per_cta),
                                (ng_pair, 0),
                            ),
                            stsm_remap_layout,
                        )
                        cute.copy(
                            stsm_tiled_copy,
                            rFc1Output,
                            stsm_thr_copy.partition_D(sFc1OutputPair),
                        )
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=32 * len(self.compute_warp_id),
                        )

                    chunks_per_row = downproj_per_cta // 4
                    store_iters = (
                        self.mma_tiler_mnk[1] * chunks_per_row
                    ) // (32 * len(self.compute_warp_id))
                    for store_iter in cutlass.range_constexpr(0, store_iters):
                        linear_chunk = (
                            mma_tidx
                            + cutlass.Int32(
                                store_iter * 32 * len(self.compute_warp_id)
                            )
                        )
                        store_token = linear_chunk // cutlass.Int32(chunks_per_row)
                        store_dp = (
                            linear_chunk % cutlass.Int32(chunks_per_row)
                        ) * cutlass.Int32(4)
                        for j in cutlass.range_constexpr(0, 4):
                            rFc1Store[j] = sFc1Output[
                                store_token,
                                store_dp + cutlass.Int32(j),
                            ]
                        if store_token < work_tile_info.valid_tokens_in_tile:
                            output_row = (
                                work_tile_info.cumulative_data_physical_row
                                + tile_token_base
                                + store_token
                            )
                            output_col = fc1_col_base + store_dp
                            output_ptr = (
                                fc1_output.iterator
                                + output_row * fc1_output.stride[0]
                                + output_col * fc1_output.stride[1]
                            ).align(4)
                            output_i32 = cute.make_tensor(
                                cute.recast_ptr(output_ptr, dtype=cutlass.Int32),
                                cute.make_layout(1),
                            )
                            output_i32[0] = rFc1StoreI32[0]
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=32 * len(self.compute_warp_id),
                    )
                    if lane_idx == cutlass.Int32(0):
                        counter_slot = (
                            work_tile_info.cumulative_data_physical_row
                            // cutlass.Int32(self.mma_tiler_mnk[1])
                            + work_tile_info.tile_n_idx
                        )
                        cute.arch.atomic_add(
                            fc1_done_counter.iterator + counter_slot,
                            cutlass.Int32(1),
                            sem="release",
                            scope="sys",
                        )
                else:
                    hidden_base = (
                        work_tile_info.tile_m_idx * cutlass.Int32(self.mma_tiler_mnk[0])
                    )
                    tile_token_base = work_tile_info.tile_n_idx * cutlass.Int32(
                        self.mma_tiler_mnk[1]
                    )
                    for ng in cutlass.range_constexpr(0, n_groups):
                        acc = accs[ng]
                        token0 = cutlass.Int32(ng * MMA_N) + lane_t * cutlass.Int32(2)
                        token1 = token0 + cutlass.Int32(1)
                        hidden0 = hidden_base + compute_warp * cutlass.Int32(16) + lane_g
                        hidden1 = hidden0 + cutlass.Int32(8)
                        row0 = (
                            work_tile_info.cumulative_data_physical_row
                            + tile_token_base
                            + token0
                        )
                        row1 = (
                            work_tile_info.cumulative_data_physical_row
                            + tile_token_base
                            + token1
                        )
                        if token0 < work_tile_info.valid_tokens_in_tile:
                            fc2_output[row0, 0, hidden0] = acc[0].to(
                                self.fc2_output_dtype
                            )
                            fc2_output[row0, 0, hidden1] = acc[2].to(
                                self.fc2_output_dtype
                            )
                        if token1 < work_tile_info.valid_tokens_in_tile:
                            fc2_output[row1, 0, hidden0] = acc[1].to(
                                self.fc2_output_dtype
                            )
                            fc2_output[row1, 0, hidden1] = acc[3].to(
                                self.fc2_output_dtype
                            )
                work_tile_info = sched_consumer.consume_work()

    @cute.jit
    def __call__(
        self,
        # ── fc1 (Linear1) problem tensors ────────────────────────────────
        activation: cute.Tensor,           # (token_sum_padded, hidden) MXFP8
        fc1_weight: cute.Tensor,           # (experts, hidden, intermediate_gateup) MXFP8
        activation_sf: cute.Tensor,         # (token_sum_padded_sf, hidden / sf_vec_size) FP8
        fc1_weight_sf: cute.Tensor,         # (experts, intermediate_gateup_padded * hidden / sf_vec_size) FP8
        # ── fc1 workspace consumed as fc2 GEMM-B ─────────────────────────
        fc1_output: cute.Tensor,         # (token_sum_padded, intermediate_downproj) MXFP8
        fc1_output_sf: cute.Tensor,      # (token_sum_padded_sf, intermediate_downproj / sf_vec_size) FP8
        # ── fc2 (Linear2) problem tensors ────────────────────────────────
        fc2_weight: cute.Tensor,          # (experts, intermediate_downproj, hidden) MXFP8
        fc2_weight_sf: cute.Tensor,        # (experts, hidden_padded * intermediate_downproj / sf_vec_size) FP8
        # MoE-domain ``(token_max, topk, hidden)`` output.
        fc2_output: cute.Tensor,
        # ── topk weights (Path A) ────────────────────────────────────────
        topk_scores: cute.Tensor,     # (token_sum_padded,) Float32
        # ── Cross-phase workspace ────────────────────────────────────────
        fc1_done_counter: cute.Tensor,  # (max_token_block_per_rank,) Int32
        # ── Sched / runtime ──────────────────────────────────────────────
        # Exactly one of ``offs`` or ``expert_token_sizes`` must be provided.
        offs: Optional[cute.Tensor] = None,  # (experts,) Int32 cumulative end offsets
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        # ── Optional epi-side scaling ────────────────────────────────────
        fc1_alpha: Optional[cute.Tensor] = None,
        fc2_alpha: Optional[cute.Tensor] = None,
        fc1_norm_const: Optional[cute.Tensor] = None,
        # ── Optional dynamic load-balance counter ────────────────────────
        load_balance_counter: Optional[cute.Tensor] = None,
        # ── Sizes-mode per-expert token count (MegaMoE path) ─────────────
        # (experts,) Int32 raw token counts (NOT cumulative).
        expert_token_sizes: Optional[cute.Tensor] = None,
        # ── MegaMoE bundle (Optional) ────────────────────────────────────
        # Opaque subclass bundle; None for the lean path.
        token_comm_args=None,
    ) -> None:
        """Launch the fused fc1+fc2 swap-AB SwiGLU MXFP8 kernel."""

        if cutlass.const_expr(self.use_inline_sm120_body and not self.enable_token_comm):
            self._launch_sm120_inline_fc12(
                activation,
                fc1_weight,
                activation_sf,
                fc1_weight_sf,
                fc1_output,
                fc1_output_sf,
                fc2_weight,
                fc2_weight_sf,
                fc2_output,
                topk_scores,
                fc1_done_counter,
                offs,
                max_active_clusters,
                stream,
                fc1_alpha,
                fc2_alpha,
                fc1_norm_const,
                load_balance_counter,
                expert_token_sizes,
            )
            return

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
            # fc2_output is MoE-domain ``(token_max, topk, hidden)``; bind
            # the hidden dim to its codegen-time const but keep ``topk``
            # caller-supplied (lean = 1 const, MegaMoE = num_topk const,
            # both already folded by the caller) and ``token_max`` runtime.
            fc2_output = cute.make_tensor(
                fc2_output.iterator,
                cute.make_layout(
                    (fc2_output.shape[0], fc2_output.shape[1], hidden_static),
                    stride=fc2_output.stride,
                ),
            )

        # ── GEMM-domain fake-MNKL transform (swap-AB) for fc1 phase ──
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # A_gemm (fc1 weights): (experts, hidden, intermediate_gateup)
        # -> (M=intermediate_gateup, K=hidden, L=experts).
        experts, hidden_b, intermediate_gateup = fc1_weight.shape
        fc1_weight_gemm = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (intermediate_gateup, hidden_b, experts),
                stride=(fc1_weight.stride[2], fc1_weight.stride[1], fc1_weight.stride[0]),
            ),
        )

        # B_gemm (fc1 activations): (tokens_sum, hidden) -> (N, K, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
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
        #   SFA (mma M-side) = fc1_weight_sf (weight scales)
        #   SFB (mma N-side) = activation_sf (activation scales)
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
                stride=(fc2_weight.stride[2], fc2_weight.stride[1], fc2_weight.stride[0]),
            ),
        )

        # fc2 phase B operand = fc1 output reused (no new view needed:
        # ``fc1_output_gemm`` was built from ``fc1_output.iterator`` with the same
        # (tokens_sum, intermediate_downproj, fake-L=1) layout that fc2's
        # GEMM-B view wants; reuse it directly when wiring fc2 TMA-B atom).

        # fc2_output is MoE-domain ``(token_max, topk, hidden)`` already;
        # we do NOT build a GEMM-domain wrapper for it.  The epilogue builds
        # a full CTA-token-tile return view from ``token_comm_args`` and
        # resolves per-token destinations inside the fc2 store path.  No
        # sched ext ``"c"`` path in this kernel anymore.

        # SFA / SFB for fc2:
        #   SFA (mma M-side) = fc2_weight_sf (fc2 weight scales)
        #   SFB (mma N-side) = fc1_output_sf (post-SwiGLU MXFP8 SFs from fc1)
        # fc2 output has no SF; no SFC built.
        tokens_sum_padded_sf = fc1_output_sf.shape[0]
        intermediate_downproj_padded = fc1_output_sf.shape[1] * self.sf_vec_size
        fc1_output_sf_gemm_for_fc2_load = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded_sf, intermediate_downproj_padded, 1),
                self.sf_vec_size,
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
        # Phases share dtypes by construction (fc1_weight and fc2_weight are
        # both MXFP8; activation and fc1_output are both MXFP8; scales are
        # all FP8).  ``self.fc1_output_dtype`` selects the fc1 MXFP8 output
        # that lives in sC; passed to the epilogue ctor as ``fc1_output_dtype``.
        self.a_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = fc1_weight_sf_gemm.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(fc1_weight_gemm)
        self.b_layout = utils.LayoutEnum.from_tensor(activation_gemm)
        self.a_major_mode = self.a_layout.mma_major_mode()
        self.b_major_mode = self.b_layout.mma_major_mode()

        self._setup_attributes()
        tiled_mma, tiled_mma_sfb = self._create_tiled_mmas()

        # ── fc1 TMA atoms ──

        # TMA load A1 (= fc1 weights)
        a_op = cpasync.CopyBulkTensorTileG2SOp()
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = cpasync.make_tiled_tma_atom(
            a_op,
            fc1_weight_gemm,
            a_smem_layout,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            num_multicast=self.num_mcast_ctas_a,
        )

        # TMA load B1 (= fc1 activations)
        b_op = cpasync.CopyBulkTensorTileG2SOp()
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_activation, tma_tensor_activation = cpasync.make_tiled_tma_atom(
            b_op,
            activation_gemm,
            b_smem_layout,
            cute.slice_(self.mma_tiler, (0, None, None)),
            num_multicast=self.num_mcast_ctas_b,
        )

        if cutlass.const_expr(UseScaleTma):
            # TMA load SFA1 (= fc1_weight_sf, fc1 weight SFs)
            sfa_op = cpasync.CopyBulkTensorTileG2SOp()
            sfa_smem_layout = cute.slice_(self.sfa_smem_layout_staged, (None, None, 0))
            tma_atom_fc1_weight_sf, tma_tensor_fc1_weight_sf = cpasync.make_tiled_tma_atom(
                sfa_op,
                fc1_weight_sf_gemm,
                sfa_smem_layout,
                cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
                num_multicast=self.num_mcast_ctas_a,
                internal_type=cutlass.Int16,
            )

            # TMA load SFB1 (= activation_sf, fc1 activation SFs)
            sfb_op = cpasync.CopyBulkTensorTileG2SOp()
            sfb_smem_layout = cute.slice_(self.sfb_smem_layout_staged, (None, None, 0))
            tma_atom_activation_sf, tma_tensor_activation_sf = cpasync.make_tiled_tma_atom(
                sfb_op,
                activation_sf_gemm,
                sfb_smem_layout,
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                num_multicast=self.num_mcast_ctas_sfb,
                internal_type=cutlass.Int16,
            )
        else:
            tma_atom_fc1_weight_sf = tma_atom_fc1_weight
            tma_tensor_fc1_weight_sf = tma_tensor_fc1_weight
            tma_atom_activation_sf = tma_atom_activation
            tma_tensor_activation_sf = tma_tensor_activation

        # FC1 output is assembled by four m16n8 STSM stores per compute warp,
        # then cooperatively written to the workspace with 32-bit global stores.
        fc1_output_smem_layout_staged = sm100_utils.make_smem_layout_epi(
            self.fc1_output_dtype,
            utils.LayoutEnum.COL_MAJOR,
            (self.mma_tiler[1], self.mma_tiler[0] // 2),
            1,
        )

        # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
        # per-thread STG.
        fc1_output_sf_gemm = cute.make_tensor(
            fc1_output_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                (tokens_sum_padded, intermediate_downproj, c1),
                self.sf_vec_size,
            ),
        )

        # ── fc2 TMA atoms: same SMEM layouts, phase-specific descriptors. ──

        tma_atom_fc2_weight, tma_tensor_fc2_weight = cpasync.make_tiled_tma_atom(
            a_op,
            fc2_weight_gemm,
            a_smem_layout,
            cute.slice_(self.mma_tiler, (None, 0, None)),
            num_multicast=self.num_mcast_ctas_a,
        )
        tma_atom_fc1_output_as_fc2_input, tma_tensor_fc1_output_as_fc2_input = cpasync.make_tiled_tma_atom(
            b_op,
            fc1_output_gemm,
            b_smem_layout,
            cute.slice_(self.mma_tiler, (0, None, None)),
            num_multicast=self.num_mcast_ctas_b,
        )
        if cutlass.const_expr(UseScaleTma):
            tma_atom_fc2_weight_sf, tma_tensor_fc2_weight_sf = cpasync.make_tiled_tma_atom(
                sfa_op,
                fc2_weight_sf_gemm,
                sfa_smem_layout,
                cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
                num_multicast=self.num_mcast_ctas_a,
                internal_type=cutlass.Int16,
            )
            tma_atom_fc1_output_sf_as_fc2_input, tma_tensor_fc1_output_sf_as_fc2_input = cpasync.make_tiled_tma_atom(
                sfb_op,
                fc1_output_sf_gemm_for_fc2_load,
                sfb_smem_layout,
                cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                num_multicast=self.num_mcast_ctas_sfb,
                internal_type=cutlass.Int16,
            )
        else:
            tma_atom_fc2_weight_sf = tma_atom_fc2_weight
            tma_tensor_fc2_weight_sf = tma_tensor_fc2_weight
            tma_atom_fc1_output_sf_as_fc2_input = tma_atom_fc1_output_as_fc2_input
            tma_tensor_fc1_output_sf_as_fc2_input = tma_tensor_fc1_output_as_fc2_input

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

        # Pick the scheduler data source.  Exactly one of ``offs`` /
        # ``expert_token_sizes`` is non-None (caller's contract; also
        # re-checked by ``MoEFusedFc12SchedulerParams`` below).  The
        # lean fc1+fc2 path goes through ``offs`` (cumulative-end, host
        # precomputed); the MegaMoE subclass goes through
        # ``expert_token_sizes`` (zero-copy ``i32 stride=(2,)`` view onto
        # ``expert_recv_count_sum`` so the sched warp can walk per-expert
        # token counts produced earlier in the same launch by the
        # dispatch warps).  Routing happens at codegen time via the
        # const-expr discrimination inside the scheduler.
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
            is_swap_ab=True,
            expert_token_prefix_sum=offs,
            expert_token_sizes=expert_token_sizes,
            streaming_fc12=self.streaming_fc12,
            streaming_fc12_tiles=self.streaming_fc12_tiles,
        )
        grid = sched_params.get_grid_shape(max_active_clusters)

        # ``token_comm_args`` is the MegaMoE-only bundle (Optional, accepted
        # via the public ``__call__`` kwarg above).  When None (lean base
        # usage), every MegaMoE-specific code branch inside the device
        # kernel is gated by ``cutlass.const_expr(token_comm_args is not
        # None)`` and vanishes at codegen time.

        self.fc1fc2_kernel_impl(
            tiled_mma,
            tiled_mma_sfb,
            # fc1 TMA atoms / tensors
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_activation,
            tma_tensor_activation,
            tma_atom_fc1_weight_sf,
            tma_tensor_fc1_weight_sf,
            tma_atom_activation_sf,
            tma_tensor_activation_sf,
            # fc2 TMA atoms / tensors
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc1_output_as_fc2_input,
            tma_tensor_fc1_output_as_fc2_input,
            tma_atom_fc2_weight_sf,
            tma_tensor_fc2_weight_sf,
            tma_atom_fc1_output_sf_as_fc2_input,
            tma_tensor_fc1_output_sf_as_fc2_input,
            # GEMM-domain tensors (fc1)
            fc1_weight_gemm,
            activation_gemm,
            fc1_output_gemm,
            fc1_weight_sf_gemm,
            activation_sf_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2; fc2's GEMM-B view = fc1_output_gemm
            # reused, so it is NOT re-passed here).  ``fc2_output`` stays
            # in MoE-domain ``(token_max, topk, hidden)`` -- the inner
            # kernel forwards it directly to the epilogue return tile.
            fc2_weight_gemm,
            fc2_output,
            fc2_weight_sf_gemm,
            fc1_output_sf_gemm_for_fc2_load,
            # topk + cross-phase sync workspace
            topk_scores,
            fc1_done_counter,
            # Optional epilogue runtime args
            fc1_alpha,
            fc2_alpha,
            fc1_norm_const,
            # Scheduling (``offs`` now lives inside ``sched_params`` as
            # ``expert_token_prefix_sum``; the inner kernel reads it via
            # ``self.params`` and no longer needs a separate copy).
            sched_params,
            self.cluster_layout_vmnk,
            self.cluster_layout_sfb_vmnk,
            # SMEM layouts
            self.a_smem_layout_staged,
            self.b_smem_layout_staged,
            self.sfa_smem_layout_staged,
            self.sfb_smem_layout_staged,
            fc1_output_smem_layout_staged,
            # MegaMoE bundle (None under the lean path).
            token_comm_args,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.kernel
    def fc1fc2_kernel_impl(
        self,
        tiled_mma: cute.TiledMma,
        tiled_mma_sfb: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_weight: cute.CopyAtom,
        tma_tensor_fc1_weight: cute.Tensor,
        tma_atom_activation: cute.CopyAtom,
        tma_tensor_activation: cute.Tensor,
        tma_atom_fc1_weight_sf: cute.CopyAtom,
        tma_tensor_fc1_weight_sf: cute.Tensor,
        tma_atom_activation_sf: cute.CopyAtom,
        tma_tensor_activation_sf: cute.Tensor,
        # fc2 TMA atoms / tensors
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc1_output_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_as_fc2_input: cute.Tensor,
        tma_atom_fc2_weight_sf: cute.CopyAtom,
        tma_tensor_fc2_weight_sf: cute.Tensor,
        tma_atom_fc1_output_sf_as_fc2_input: cute.CopyAtom,
        tma_tensor_fc1_output_sf_as_fc2_input: cute.Tensor,
        # GEMM-domain tensors (fc1)
        fc1_weight_gemm: cute.Tensor,
        activation_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2; fc2's GEMM-B view = ``fc1_output_gemm``
        # reused, so it is NOT in this list -- see the caller).
        # ``fc2_output`` is MoE-domain ``(token_max, topk, hidden)`` --
        # no GEMM-domain wrapper is built; the epilogue return tile consumes
        # the MoE-domain shape directly.
        fc2_weight_gemm: cute.Tensor,
        fc2_output: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc1_output_sf_gemm_for_fc2_load: cute.Tensor,
        # topk + cross-phase sync workspace
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        # Optional epilogue runtime args
        fc1_alpha: Optional[cute.Tensor],
        fc2_alpha: Optional[cute.Tensor],
        fc1_norm_const: Optional[cute.Tensor],
        # Scheduling (the per-expert token range tensor is carried inside
        # ``sched_params`` as ``expert_token_prefix_sum`` or
        # ``expert_token_sizes`` -- never passed separately).
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        cluster_layout_sfb_vmnk: cute.Layout,
        # SMEM layouts
        a_smem_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        sfa_smem_layout_staged: cute.Layout,
        sfb_smem_layout_staged: cute.Layout,
        fc1_output_smem_layout_staged: cute.ComposedLayout,
        # MegaMoE-only bundle (None for the lean fc1+fc2 path).  All
        # MegaMoE-specific code (dispatch warps emit, fc1 spin on
        # ``l1_arrival_count``, combine STG redirect, kernel-tail NVLink
        # barrier) is gated by ``cutlass.const_expr(token_comm_args is not
        # None)`` so when None those branches vanish at codegen time.
        token_comm_args=None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB SwiGLU MXFP8 grouped GEMM.

        Lean (``force_static_sched=True``) path: 7-warp specialization with
        no empty / drain_aux warps and no expert-wise TMA desc rewriting
        (every desc is tile-invariant under swap-AB).

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))
        sfa_smem_layout = cute.slice_(sfa_smem_layout_staged, (None, None, 0))
        sfb_smem_layout = cute.slice_(sfb_smem_layout_staged, (None, None, 0))

        # FC1 epilogue publishes one arrival per compute warp for each
        # intermediate tile in the same token block.  FC2 may read the
        # workspace only after all compute-warps have written their slices.
        ext_fc2_spin_threshold = (
            (
                fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[0] - 1
            )
            // self.cta_tile_shape_mnk[0]
        ) * len(self.compute_warp_id)

        # The ``token_comm_hook_fc1_ready_counter_ptr`` hook lets a MegaMoE
        # subclass plug in the dispatch->fc1 release counter pointer so the
        # ext's sched-warp peek can cover the fc1 phase as well.  Base
        # returns None, leaving the lean fc1+fc2 path with only the
        # fc1->fc2 peek active.
        ext = Sm120SwapABMxfp8Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)
        use_2cta_instrs = False

        bidx, _, bidz = cute.arch.block_idx()
        mma_tile_coord_v = cutlass.Int32(0)
        is_leader_cta = True
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
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 4]
            sched_storage: SchedStorage
            sFc2ReadySlot: cute.struct.MemRange[cutlass.Int32, 2]
            sA: cute.struct.Align[
                cute.struct.MemRange[
                    self.a_dtype, cute.cosize(a_smem_layout_staged)
                ],
                128,
            ]
            sB: cute.struct.Align[
                cute.struct.MemRange[
                    self.b_dtype, cute.cosize(b_smem_layout_staged)
                ],
                128,
            ]
            sSFA: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfa_smem_layout_staged)
                ],
                128,
            ]
            sSFB: cute.struct.Align[
                cute.struct.MemRange[
                    self.sf_dtype, cute.cosize(sfb_smem_layout_staged)
                ],
                128,
            ]
            sFc1Swiglu: cute.struct.Align[
                cute.struct.MemRange[
                    cutlass.Float32,
                    16 * (self.mma_tiler[0] // 2),
                ],
                128,
            ]
            sFc1Output: cute.struct.Align[
                cute.struct.MemRange[
                    self.fc1_output_dtype,
                    cute.cosize(fc1_output_smem_layout_staged),
                ],
                128,
            ]

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)

        # MegaMoE-only ``token_comm_storage``: standalone SMEM region whose
        # struct shape is owned by the subclass (e.g. dispatch pull_buffer
        # / per-warp mbarriers / per-expert SMEM histogram).  Kept disjoint
        # from the base ``SharedStorage`` so the lean path neither allocates
        # nor names it.  None when the subclass returns None (base default);
        # any subclass that needs SMEM returns its own ``@cute.struct``
        # class from ``token_comm_extra_smem_storage_class`` and consumes
        # the handle inside ``token_comm_hook_dispatch_warp_body``.
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls, byte_alignment=128)
        else:
            token_comm_storage = None

        # ── Pipelines: independent A/SFA and B/SFB TMA producer warp streams. ──

        tma_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread
        )
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, len(self.compute_warp_id)
        )
        a_producer, a_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=tma_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_a_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()
        b_producer, b_consumer = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr() + self.num_ab_stage * 2,
            num_stages=self.num_ab_stage,
            producer_group=tma_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_b_load_bytes,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        ).make_participants()

        # Sched
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                *self.compute_warp_id,
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

        # Early-init iff ``internal_init`` does NOT depend on sizes.  Sizes
        # under MegaMoE come from ``expert_recv_count_sum`` filled by the
        # dispatch warps; if static load-balance mode + token_comm are
        # both active, ``internal_init`` walks the per-expert sizes during
        # the first-tile decode and MUST run AFTER the dispatch_barrier
        # completes (i.e. after the sched warp drains NamedBarrier 9 in
        # the per-warp split below).  The other three combos can keep the
        # existing "atomic overlaps cluster barrier" timing.
        early_internal_init = (
            (self.load_balance_mode == "atomic_counter")
            or (not self.enable_token_comm)
        )

        # Issue the first scheduler claim before cluster init wait so the
        # atomic/offets latency overlaps with pipeline setup.
        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx,
                sched_warp_id=self.sched_warp_id,
            )

        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)

        # ── SMEM tensors A / B / SFA / SFB (shared by fc1 / fc2) ──
        sA = storage.sA.get_tensor(
            a_smem_layout_staged.outer,
            swizzle=a_smem_layout_staged.inner,
        )
        sB = storage.sB.get_tensor(
            b_smem_layout_staged.outer,
            swizzle=b_smem_layout_staged.inner,
        )
        sSFA = storage.sSFA.get_tensor(sfa_smem_layout_staged)
        sSFB = storage.sSFB.get_tensor(sfb_smem_layout_staged)
        sFc2ReadySlot = storage.sFc2ReadySlot.get_tensor(cute.make_layout(2))
        # Two N8 accumulator groups share one N16 x downproj32 FP32 reduction
        # tile.  Quantized output accumulates in an STSM-compatible swizzled
        # N64 x downproj32 tile before the compute threads publish it to GMEM.
        sFc1Swiglu = storage.sFc1Swiglu.get_tensor(
            cute.make_layout(
                (16, self.mma_tiler[0] // 2),
                stride=(self.mma_tiler[0] // 2, 1),
            )
        )
        sFc1OutputStaged = storage.sFc1Output.get_tensor(
            fc1_output_smem_layout_staged.outer,
            swizzle=fc1_output_smem_layout_staged.inner,
        )
        sFc1Output = cute.slice_(sFc1OutputStaged, (None, None, 0))

        # Cluster wait before TMA / compute work.
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
        fc1_tiles_per_fc2_k_tile = self.mma_tiler[2] // (
            self.mma_tiler[0] // 2
        )
        fc2_ready_bundle_cnt = (
            k_tile_cnt_fc2 + self.fc2_ready_bundle_k_tiles - 1
        ) // self.fc2_ready_bundle_k_tiles

        # TMA-B hands FC2 counter slots to the otherwise-idle warp 7 through
        # one SMEM word.  The aux warp polls one counter per K bundle and
        # releases TMA-B with a CTA-local named barrier.
        if warp_idx == self.sm120_aux_warp_id:
            aux_lane_idx = cute.arch.lane_idx()
            if cutlass.const_expr(token_comm_args is not None):
                aux_running = cutlass.Boolean(True)
                aux_ready_slot = cutlass.Int32(0)
                while aux_running:
                    if aux_ready_slot == cutlass.Int32(0):
                        cute.arch.barrier(
                            barrier_id=self.fc2_ready_work_bar_id,
                            number_of_threads=64,
                        )
                    else:
                        cute.arch.barrier(
                            barrier_id=self.fc2_ready_work_bar_alt_id,
                            number_of_threads=64,
                        )
                    aux_counter_slot = sFc2ReadySlot[aux_ready_slot]
                    if aux_counter_slot < cutlass.Int32(0):
                        aux_running = cutlass.Boolean(False)
                    else:
                        for aux_bundle_idx in cutlass.range(
                            0, fc2_ready_bundle_cnt, 1, unroll=1
                        ):
                            aux_bundle_begin = (
                                aux_bundle_idx
                                * cutlass.Int32(self.fc2_ready_bundle_k_tiles)
                            )
                            aux_bundle_k_tiles = cutlass.min(
                                cutlass.Int32(self.fc2_ready_bundle_k_tiles),
                                k_tile_cnt_fc2 - aux_bundle_begin,
                            )
                            if aux_lane_idx == cutlass.Int32(0):
                                spin_wait_i32_ge_inline(
                                    fc1_done_counter.iterator
                                    + aux_counter_slot * fc2_ready_bundle_cnt
                                    + aux_bundle_idx,
                                    aux_bundle_k_tiles
                                    * cutlass.Int32(fc1_tiles_per_fc2_k_tile),
                                    fail_sleep_cycles=500,
                                )
                            if aux_ready_slot == cutlass.Int32(0):
                                cute.arch.barrier(
                                    barrier_id=self.fc2_ready_bar_id,
                                    number_of_threads=64,
                                )
                            else:
                                cute.arch.barrier(
                                    barrier_id=self.fc2_ready_bar_alt_id,
                                    number_of_threads=64,
                                )
                        aux_ready_slot = cutlass.Int32(1) - aux_ready_slot

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (warp 6)
        # ════════════════════════════════════════════════════════════════════
        if warp_idx == self.sched_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            # MegaMoE subclass uses this hook to wait for this CTA's
            # dispatch warps to finish ``_dispatch_barrier`` -- only then
            # is ``expert_recv_count_sum`` (and therefore the sizes view
            # the scheduler reads in static mode, plus everything
            # dispatch_pull writes per token) visible.  Base no-op:
            # nothing to wait for in the lean path.
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            # Late init (only token_comm + static lands here -- the other
            # three combos finished ``internal_init`` before
            # pipeline_init_arrive above and ``early_internal_init`` is
            # True for them).
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                # Keep one static event name with a phase payload so IKET
                # versions with payload support can reconstruct the publish
                # stream (0 = FC1, 1 = FC2).
                iket.mark(
                    "schedule_tile_phase",
                    scheduler.current_work.phase,
                )
                if (
                    scheduler.current_work.phase
                    == cutlass.Int32(BlockPhase.Linear1)
                ):
                    iket.range_push("schedule_fc1_tile")
                else:
                    iket.range_push("schedule_fc2_tile")
                scheduler.publish_work()
                iket.range_pop()
                scheduler.gen_next_work()
            # Sentinel publish (current_work is already invalid here).
            scheduler.publish_work()
            scheduler.produce_tail()

        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx == self.sm120_aux_warp_id:
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

        # ════════════════════════════════════════════════════════════════════
        # TMA load warps (warps 5 / 6)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A loads weights/SFA; TMA-B loads activations/SFB and waits for
        # fc1 workspace readiness in fc2 phase.  The two streams have separate
        # producer states and the compute warps wait on both per K tile.


        # ── TMA-A warp (warp 5) ─────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

            a_full_mcast_mask = None
            sfa_full_mcast_mask = None
            if cutlass.const_expr(self.is_a_mcast or use_2cta_instrs):
                a_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )
                sfa_full_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=2
                )

            a_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, 0, None, 0)).shape
            )
            sfa_cta_layout = a_cta_layout

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )

                if is_phase_linear1:
                    # ── fc1 phase A-side ─────────────────────────────────
                    iket.range_push("tma_weight_fc1")
                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a", tma_tensor_fc1_weight, work_tile_info,
                    )
                    if cutlass.const_expr(UseScaleTma):
                        real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                            "sfa", tma_tensor_fc1_weight_sf, work_tile_info,
                        )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    if cutlass.const_expr(UseScaleTma):
                        gSFA_mkl = cute.local_tile(
                            real_sfa,
                            cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
                            (None, None, None),
                        )
                        tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc1_weight,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA_mkl, 0, 2),
                    )
                    if cutlass.const_expr(UseScaleTma):
                        tAsSFA, tAgSFA = cpasync.tma_partition(
                            tma_atom_fc1_weight_sf,
                            block_in_cluster_coord_vmnk[2],
                            sfa_cta_layout,
                            cute.group_modes(sSFA, 0, 2),
                            cute.group_modes(gSFA_mkl, 0, 2),
                        )
                        tAsSFA = cute.filter_zeros(tAsSFA)
                        tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    if cutlass.const_expr(UseScaleTma):
                        sfa_tile_m = mma_tile_m
                        if cutlass.const_expr(self.mma_tiler_sfa[0] != self.mma_tiler[0]):
                            sfa_tile_m = mma_tile_m // cutlass.Int32(
                                self.mma_tiler_sfa[0] // self.mma_tiler[0]
                            )
                        tAgSFA_slice = tAgSFA[(None, sfa_tile_m, None, 0)]

                    a_producer.reset()
                    peek_ab_empty_status = a_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_weight,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        if cutlass.const_expr(UseScaleTma):
                            cute.copy(
                                tma_atom_fc1_weight_sf,
                                tAgSFA_slice[(None, handle.count)],
                                tAsSFA[(None, handle.index)],
                                tma_bar_ptr=handle.barrier,
                                tma_desc_ptr=desc_ptr_sfa,
                                mcast_mask=sfa_full_mcast_mask,
                            )
                else:
                    # ── fc2 phase A-side (no readiness gate) ─────────────
                    iket.range_push("tma_weight_fc2")
                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a", tma_tensor_fc2_weight, work_tile_info,
                    )
                    if cutlass.const_expr(UseScaleTma):
                        real_sfa, desc_ptr_sfa = ext.get_gmem_tensor(
                            "sfa", tma_tensor_fc2_weight_sf, work_tile_info,
                        )

                    gA_mkl = cute.local_tile(
                        real_a,
                        cute.slice_(self.mma_tiler, (None, 0, None)),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA_mkl)
                    if cutlass.const_expr(UseScaleTma):
                        gSFA_mkl = cute.local_tile(
                            real_sfa,
                            cute.slice_(self.mma_tiler_sfa, (None, 0, None)),
                            (None, None, None),
                        )
                        tCgSFA = thr_mma.partition_A(gSFA_mkl)

                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_fc2_weight,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 2),
                        cute.group_modes(gA_mkl, 0, 2),
                    )
                    if cutlass.const_expr(UseScaleTma):
                        tAsSFA, tAgSFA = cpasync.tma_partition(
                            tma_atom_fc2_weight_sf,
                            block_in_cluster_coord_vmnk[2],
                            sfa_cta_layout,
                            cute.group_modes(sSFA, 0, 2),
                            cute.group_modes(gSFA_mkl, 0, 2),
                        )
                        tAsSFA = cute.filter_zeros(tAsSFA)
                        tAgSFA = cute.filter_zeros(tAgSFA)

                    mma_tile_m = work_tile_info.tile_m_idx
                    tAgA_slice = tAgA[(None, mma_tile_m, None, 0)]
                    if cutlass.const_expr(UseScaleTma):
                        sfa_tile_m = mma_tile_m
                        if cutlass.const_expr(self.mma_tiler_sfa[0] != self.mma_tiler[0]):
                            sfa_tile_m = mma_tile_m // cutlass.Int32(
                                self.mma_tiler_sfa[0] // self.mma_tiler[0]
                            )
                        tAgSFA_slice = tAgSFA[(None, sfa_tile_m, None, 0)]

                    a_producer.reset()
                    peek_ab_empty_status = a_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = a_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = a_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc2_weight,
                            tAgA_slice[(None, handle.count)],
                            tAsA[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_a,
                            mcast_mask=a_full_mcast_mask,
                        )
                        if cutlass.const_expr(UseScaleTma):
                            cute.copy(
                                tma_atom_fc2_weight_sf,
                                tAgSFA_slice[(None, handle.count)],
                                tAsSFA[(None, handle.index)],
                                tma_bar_ptr=handle.barrier,
                                tma_desc_ptr=desc_ptr_sfa,
                                mcast_mask=sfa_full_mcast_mask,
                            )

                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            a_producer.tail()

        # ── TMA-B warp (warp 6) ─────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

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

            b_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape
            )
            sfb_cta_layout = cute.make_layout(
                cute.slice_(cluster_layout_sfb_vmnk, (0, None, 0, 0)).shape
            )

            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_sfb = tiled_mma_sfb.get_slice(mma_tile_coord_v)

            # fc2-spin saturation threshold (work-tile-invariant -- the
            # per-(expert, token_block) per-CTA-event count along the
            # ``intermediate_gateup`` axis is a global constant under v1
            # mma_tiler, depending only on the geometry).
            #
            # ``fc1_weight_gemm.shape[0]`` resolves to ``intermediate_gateup``,
            # which is a codegen-time Python int under ``static_expert_shape``
            # (rewritten on ``fc1_weight`` at ``__call__`` entry) or a
            # runtime Int32 from tensor metadata otherwise.  ``//
            # cta_tile_shape_mnk[0]`` then folds to an immediate in the
            # static path (divisor is always a Python int constant); in
            # the dynamic path it's still loop-invariant and hoisted here
            # so the work-tile loop body just reads a register.
            #
            # FC1 epilogue publishes one arrival per compute warp for each
            # intermediate tile in the same token block.  FC2 may read the
            # workspace only after all compute-warps have written their slices.
            fc2_spin_threshold = (
                (
                    fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[0] - 1
                )
                // self.cta_tile_shape_mnk[0]
            ) * len(self.compute_warp_id)

            work_tile_info = sched_consumer.consume_work()
            b_ready_slot = cutlass.Int32(0)

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )

                if is_phase_linear1:
                    # ── fc1 phase B-side (activation + activation_sf) ────
                    iket.range_push("tma_token_fc1")

                    # MegaMoE subclass uses this hook to spin on the
                    # dispatch->fc1 release counter for this task tile
                    # before issuing the TMA loads.  Base no-op: in the
                    # lean path the activation tensor is fully resident
                    # in GMEM by launch time, no per-tile wait required.
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args, work_tile_info,
                    )

                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b", tma_tensor_activation, work_tile_info,
                    )
                    if cutlass.const_expr(UseScaleTma):
                        real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                            "sfb", tma_tensor_activation_sf, work_tile_info,
                        )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    if cutlass.const_expr(UseScaleTma):
                        gSFB_nkl = cute.local_tile(
                            real_sfb,
                            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                            (None, None, None),
                        )
                        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_activation,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nkl, 0, 2),
                    )
                    if cutlass.const_expr(UseScaleTma):
                        tBsSFB, tBgSFB = cpasync.tma_partition(
                            tma_atom_activation_sf,
                            block_in_cluster_coord_sfb_vmnk[1],
                            sfb_cta_layout,
                            cute.group_modes(sSFB, 0, 2),
                            cute.group_modes(gSFB_nkl, 0, 2),
                        )
                        tBsSFB = cute.filter_zeros(tBsSFB)
                        tBgSFB = cute.filter_zeros(tBgSFB)

                    tBgB_slice = tBgB[(None, work_tile_info.tile_n_idx, None, 0)]
                    if cutlass.const_expr(UseScaleTma):
                        # Map the CTA N tile into the rounded N128 SFB tile.
                        sfb_tile_n_idx = work_tile_info.tile_n_idx
                        if cutlass.const_expr(self.mma_tiler[1] < 128):
                            sfb_tile_n_idx = (
                                work_tile_info.tile_n_idx
                                // cutlass.Int32(128 // self.mma_tiler[1])
                            )
                        tBgSFB_slice = tBgSFB[
                            (None, sfb_tile_n_idx, None, 0)
                        ]

                    b_producer.reset()
                    peek_ab_empty_status = b_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        handle = b_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_activation,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        if cutlass.const_expr(UseScaleTma):
                            cute.copy(
                                tma_atom_activation_sf,
                                tBgSFB_slice[(None, handle.count)],
                                tBsSFB[(None, handle.index)],
                                tma_bar_ptr=handle.barrier,
                                tma_desc_ptr=desc_ptr_sfb,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                else:
                    # ── fc2 phase B-side ─────────────────────────────────
                    # Lean FC12 waits once for the complete token block.
                    # MegaMoE instead hands this counter slot to the aux warp,
                    # which releases TMA-B one K bundle at a time.
                    iket.range_push("tma_token_fc2")
                    counter_slot = (
                        work_tile_info.cumulative_data_physical_row
                        // cutlass.Int32(self.mma_tiler[1])
                        + work_tile_info.tile_n_idx
                    )
                    if cutlass.const_expr(token_comm_args is None):
                        counter_ptr = fc1_done_counter.iterator + counter_slot
                        iket.range_push("tma_token_fc2_wait")
                        spin_wait(
                            counter_ptr,
                            lambda v: v >= fc2_spin_threshold,
                            fail_sleep_cycles=500,
                        )
                        iket.range_pop()
                        cute.arch.fence_acq_rel_sys()
                        cute.arch.fence_proxy("async.global")
                    else:
                        b_lane_idx = cute.arch.lane_idx()
                        if b_lane_idx == cutlass.Int32(0):
                            sFc2ReadySlot[b_ready_slot] = counter_slot
                        if b_ready_slot == cutlass.Int32(0):
                            cute.arch.barrier(
                                barrier_id=self.fc2_ready_work_bar_id,
                                number_of_threads=64,
                            )
                        else:
                            cute.arch.barrier(
                                barrier_id=self.fc2_ready_work_bar_alt_id,
                                number_of_threads=64,
                            )

                    # fc1 workspace is fc2 GEMM-B/SFB for this token block.
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b",
                        tma_tensor_fc1_output_as_fc2_input,
                        work_tile_info,
                    )
                    if cutlass.const_expr(UseScaleTma):
                        real_sfb, desc_ptr_sfb = ext.get_gmem_tensor(
                            "sfb",
                            tma_tensor_fc1_output_sf_as_fc2_input,
                            work_tile_info,
                        )

                    gB_nkl = cute.local_tile(
                        real_b,
                        cute.slice_(self.mma_tiler, (0, None, None)),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB_nkl)
                    if cutlass.const_expr(UseScaleTma):
                        gSFB_nkl = cute.local_tile(
                            real_sfb,
                            cute.slice_(self.mma_tiler_sfb, (0, None, None)),
                            (None, None, None),
                        )
                        tCgSFB = thr_mma_sfb.partition_B(gSFB_nkl)

                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc1_output_as_fc2_input,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 2),
                        cute.group_modes(gB_nkl, 0, 2),
                    )
                    if cutlass.const_expr(UseScaleTma):
                        tBsSFB, tBgSFB = cpasync.tma_partition(
                            tma_atom_fc1_output_sf_as_fc2_input,
                            block_in_cluster_coord_sfb_vmnk[1],
                            sfb_cta_layout,
                            cute.group_modes(sSFB, 0, 2),
                            cute.group_modes(gSFB_nkl, 0, 2),
                        )
                        tBsSFB = cute.filter_zeros(tBsSFB)
                        tBgSFB = cute.filter_zeros(tBgSFB)

                    tBgB_slice = tBgB[
                        (None, work_tile_info.tile_n_idx, None, 0)
                    ]
                    if cutlass.const_expr(UseScaleTma):
                        # Map the CTA N tile into the rounded N128 SFB tile.
                        sfb_tile_n_idx = work_tile_info.tile_n_idx
                        if cutlass.const_expr(self.mma_tiler[1] < 128):
                            sfb_tile_n_idx = (
                                work_tile_info.tile_n_idx
                                // cutlass.Int32(128 // self.mma_tiler[1])
                            )
                        tBgSFB_slice = tBgSFB[
                            (None, sfb_tile_n_idx, None, 0)
                        ]

                    # Step 3: K-loop with 2x cute.copy per tile (B +
                    # SFB).  Same cadence as the fc1 phase above; we
                    # uses the B/SFB pipeline producer owned by this warp.
                    b_producer.reset()
                    peek_ab_empty_status = b_producer.try_acquire()

                    for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                        if cutlass.const_expr(token_comm_args is not None):
                            if (
                                k_tile
                                % cutlass.Int32(self.fc2_ready_bundle_k_tiles)
                                == cutlass.Int32(0)
                            ):
                                if b_ready_slot == cutlass.Int32(0):
                                    cute.arch.barrier(
                                        barrier_id=self.fc2_ready_bar_id,
                                        number_of_threads=64,
                                    )
                                else:
                                    cute.arch.barrier(
                                        barrier_id=self.fc2_ready_bar_alt_id,
                                        number_of_threads=64,
                                    )
                                cute.arch.fence_proxy("async.global")
                        handle = b_producer.acquire_and_advance(
                            peek_ab_empty_status
                        )
                        peek_ab_empty_status = cutlass.Boolean(1)
                        if handle.count + 1 < k_tile_cnt:
                            peek_ab_empty_status = b_producer.try_acquire()
                        cute.copy(
                            tma_atom_fc1_output_as_fc2_input,
                            tBgB_slice[(None, handle.count)],
                            tBsB[(None, handle.index)],
                            tma_bar_ptr=handle.barrier,
                            tma_desc_ptr=desc_ptr_b,
                            mcast_mask=b_full_mcast_mask,
                        )
                        if cutlass.const_expr(UseScaleTma):
                            cute.copy(
                                tma_atom_fc1_output_sf_as_fc2_input,
                                tBgSFB_slice[(None, handle.count)],
                                tBsSFB[(None, handle.index)],
                                tma_bar_ptr=handle.barrier,
                                tma_desc_ptr=desc_ptr_sfb,
                                mcast_mask=sfb_full_mcast_mask,
                            )
                    if cutlass.const_expr(token_comm_args is not None):
                        b_ready_slot = cutlass.Int32(1) - b_ready_slot
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            if cutlass.const_expr(token_comm_args is not None):
                b_lane_idx = cute.arch.lane_idx()
                if b_lane_idx == cutlass.Int32(0):
                    sFc2ReadySlot[b_ready_slot] = cutlass.Int32(-1)
                if b_ready_slot == cutlass.Int32(0):
                    cute.arch.barrier(
                        barrier_id=self.fc2_ready_work_bar_id,
                        number_of_threads=64,
                    )
                else:
                    cute.arch.barrier(
                        barrier_id=self.fc2_ready_work_bar_alt_id,
                        number_of_threads=64,
                    )
            b_producer.tail()

        # ════════════════════════════════════════════════════════════════════
        # SM120 compute warps (warps 0-3)
        # ════════════════════════════════════════════════════════════════════
        if warp_idx < len(self.compute_warp_id):
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.warpgroup_reg_alloc(self.epi_reg_cnt)

            compute_warp = warp_idx
            lane_idx = cute.arch.lane_idx()
            lane_g = lane_idx >> cutlass.Int32(2)
            lane_t = lane_idx & cutlass.Int32(3)
            warp_dp_base = compute_warp * SWAP_AB_INTERLEAVE
            dp_in_cta = warp_dp_base + lane_g
            downproj_per_cta = self.mma_tiler[0] // 2
            n_groups = self.mma_tiler[1] // MMA_N

            stsm_atom = cute.make_copy_atom(
                cute.nvgpu.warp.StMatrix16x8x8bOp(
                    transpose=True,
                    num_matrices=1,
                ),
                self.fc1_output_dtype,
            )
            stsm_tiled_copy = cute.make_tiled_copy_C_atom(stsm_atom, tiled_mma)
            stsm_thr_copy = stsm_tiled_copy.get_slice(tidx)
            stsm_remap_layout = cute.make_layout(
                ((16, 4), MMA_N),
                stride=(
                    (1, 16 * SWAP_AB_INTERLEAVE),
                    16,
                ),
            )
            sFc1OutputAtom = cute.composition(
                cute.local_tile(sFc1Output, (16, downproj_per_cta), (0, 0)),
                stsm_remap_layout,
            )
            stsm_src_shape = cute.shape(
                stsm_thr_copy.partition_S(sFc1OutputAtom)
            )
            rFc1Output = cute.make_rmem_tensor(
                stsm_src_shape,
                self.fc1_output_dtype,
            )
            rFc1OutputPacked = cute.recast_tensor(rFc1Output, cutlass.Int32)
            rFc1OutputLocal = cute.make_rmem_tensor((4,), cutlass.Float32)
            rFc1Store = cute.make_rmem_tensor((4,), self.fc1_output_dtype)
            rFc1StoreI32 = cute.recast_tensor(rFc1Store, cutlass.Int32)
            rFc2Store = cute.make_rmem_tensor((8,), self.fc2_output_dtype)
            rFc2StoreI32 = cute.recast_tensor(rFc2Store, cutlass.Int32)

            thr_mma = tiled_mma.get_slice(tidx)
            tCsA = thr_mma.partition_A(sA)
            tCsB = thr_mma.partition_B(sB)
            tCrA = tiled_mma.make_fragment_A(tCsA[None, None, None, 0])
            tCrB = tiled_mma.make_fragment_B(tCsB[None, None, None, 0])
            tCrSFA = sm100_utils.partition_fragment_SFA(
                sSFA[None, None, 0], thr_mma, tidx
            )
            tCrSFB = sm100_utils.partition_fragment_SFB(
                sSFB[None, None, 0], thr_mma, tidx
            )

            atom_copy_ldmatrix_A = make_sm120_ldmatrix_atom(
                self.a_dtype,
                transpose=self.a_layout.is_m_major_a(),
            )
            atom_copy_ldmatrix_B = make_sm120_ldmatrix_atom(
                self.b_dtype,
                transpose=self.b_layout.is_n_major_b(),
            )
            smem_tiled_copy_A = cute.make_tiled_copy_A(
                atom_copy_ldmatrix_A, tiled_mma
            )
            smem_tiled_copy_B = cute.make_tiled_copy_B(
                atom_copy_ldmatrix_B, tiled_mma
            )
            atom_copy_scale = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.sf_dtype,
            )
            smem_tiled_copy_SFA = cute.make_tiled_copy(
                atom_copy_scale,
                sm100_utils.get_layoutSFA_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[0]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            smem_tiled_copy_SFB = cute.make_tiled_copy(
                atom_copy_scale,
                sm100_utils.get_layoutSFB_TV(tiled_mma),
                (
                    cute.size(tiled_mma.permutation_mnk[1]),
                    cute.size(tiled_mma.permutation_mnk[2]),
                ),
            )
            thr_copy_ldmatrix_A = smem_tiled_copy_A.get_slice(tidx)
            thr_copy_ldmatrix_B = smem_tiled_copy_B.get_slice(tidx)
            tCsA_copy_view = thr_copy_ldmatrix_A.partition_S(sA)
            tCsB_copy_view = thr_copy_ldmatrix_B.partition_S(sB)
            tCrA_copy_view = thr_copy_ldmatrix_A.retile(tCrA)
            tCrB_copy_view = thr_copy_ldmatrix_B.retile(tCrB)
            thr_copy_SFA = smem_tiled_copy_SFA.get_slice(tidx)
            thr_copy_SFB = smem_tiled_copy_SFB.get_slice(tidx)
            tCsSFA_copy_view = thr_copy_SFA.partition_S(sSFA)
            tCsSFB_copy_view = thr_copy_SFB.partition_S(sSFB)
            tCrSFA_copy_view = thr_copy_SFA.retile(tCrSFA)
            tCrSFB_copy_view = thr_copy_SFB.retile(tCrSFB)

            acc_shape = tiled_mma.partition_shape_C(
                (self.mma_tiler[0], self.mma_tiler[1])
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)

            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                k_tile_cnt = cutlass.Int32(0)
                if is_phase_linear1:
                    k_tile_cnt = k_tile_cnt_fc1
                    iket.range_push("sm120_fc1_tile")
                    iket.range_push("sm120_fc1_mainloop")
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    iket.range_push("sm120_fc2_tile")
                    iket.range_push("sm120_fc2_mainloop")

                accumulators.fill(0.0)

                a_consumer.reset()
                b_consumer.reset()
                peek_a_full_status = cutlass.Boolean(1)
                peek_b_full_status = cutlass.Boolean(1)
                if k_tile_cnt > 0:
                    peek_a_full_status = a_consumer.try_wait()
                    peek_b_full_status = b_consumer.try_wait()

                for k_tile in cutlass.range(0, k_tile_cnt, 1, unroll=1):
                    handle_a = a_consumer.wait_and_advance(peek_a_full_status)
                    handle_b = b_consumer.wait_and_advance(peek_b_full_status)
                    peek_a_full_status = cutlass.Boolean(1)
                    peek_b_full_status = cutlass.Boolean(1)
                    if handle_a.count + 1 < k_tile_cnt:
                        peek_a_full_status = a_consumer.try_wait()
                    if handle_b.count + 1 < k_tile_cnt:
                        peek_b_full_status = b_consumer.try_wait()

                    tCsA_p = tCsA_copy_view[None, None, None, handle_a.index]
                    tCsB_p = tCsB_copy_view[None, None, None, handle_b.index]
                    tCsSFA_p = tCsSFA_copy_view[None, None, None, handle_a.index]
                    tCsSFB_p = tCsSFB_copy_view[None, None, None, handle_b.index]
                    sfa_m_group = work_tile_info.tile_m_idx % cutlass.Int32(
                        self.mma_tiler_sfa[0] // self.mma_tiler[0]
                    )
                    # partition_fragment_SFA packs the two M64 groups of the
                    # staged M128 scale tile eight bytes apart. Select the
                    # group in SMEM so the RMEM fragment is statically indexed.
                    tCsSFA_selected = cute.make_tensor(
                        tCsSFA_p.iterator + sfa_m_group * cutlass.Int32(8),
                        tCsSFA_p.layout,
                    )
                    tCsSFA_p_filtered = cute.filter_zeros(tCsSFA_selected)
                    tCsSFB_p_filtered = cute.filter_zeros(tCsSFB_p)
                    tCrSFA_copy_view_filtered = cute.filter_zeros(tCrSFA_copy_view)
                    tCrSFB_copy_view_filtered = cute.filter_zeros(tCrSFB_copy_view)
                    cute.copy(
                        smem_tiled_copy_A,
                        tCsA_p[None, None, 0],
                        tCrA_copy_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_B,
                        tCsB_p[None, None, 0],
                        tCrB_copy_view[None, None, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_SFA,
                        tCsSFA_p_filtered[None, 0, 0],
                        tCrSFA_copy_view_filtered[None, 0, 0],
                    )
                    cute.copy(
                        smem_tiled_copy_SFB,
                        tCsSFB_p_filtered[None, None, 0],
                        tCrSFB_copy_view_filtered[None, 0, 0, None],
                    )

                    tCrSFB_mma = tCrSFB
                    if cutlass.const_expr(self.mma_tiler[1] < 128):
                        # SFB is staged as N128; select this CTA tile's
                        # register subfragment before issuing its N groups.
                        sfb_tiles_per_tma = 128 // self.mma_tiler[1]
                        sfb_fragment_shift = (
                            work_tile_info.tile_n_idx
                            % cutlass.Int32(sfb_tiles_per_tma)
                        ) * cutlass.Int32(n_groups // 4)
                        tCrSFB_mma = cute.make_tensor(
                            tCrSFB.iterator + sfb_fragment_shift,
                            tCrSFB.layout,
                        )

                    for k_inner_mma in cutlass.range_constexpr(0, 4):
                        if cutlass.const_expr(k_inner_mma + 1 < 4):
                            k_inner_next = k_inner_mma + 1
                            cute.copy(
                                smem_tiled_copy_A,
                                tCsA_p[None, None, k_inner_next],
                                tCrA_copy_view[None, None, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_B,
                                tCsB_p[None, None, k_inner_next],
                                tCrB_copy_view[None, None, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFA,
                                tCsSFA_p_filtered[None, 0, k_inner_next],
                                tCrSFA_copy_view_filtered[None, 0, k_inner_next],
                            )
                            cute.copy(
                                smem_tiled_copy_SFB,
                                tCsSFB_p_filtered[None, None, k_inner_next],
                                tCrSFB_copy_view_filtered[
                                    None, 0, k_inner_next, None
                                ],
                            )
                        for ng in cutlass.range_constexpr(0, n_groups):
                            issue_m64n8k32_mxfp8(
                                tiled_mma,
                                accumulators[None, None, ng],
                                tCrA,
                                tCrB,
                                tCrSFA,
                                tCrSFB_mma,
                                n_group=ng,
                                active_n_groups=n_groups,
                                sfa_m_group=0,
                                k_inner=k_inner_mma,
                                ab_dtype=self.ab_dtype,
                                sf_dtype=self.sf_dtype,
                            )
                    handle_a.release()
                    handle_b.release()

                iket.range_pop()
                if is_phase_linear1:
                    iket.range_push("sm120_fc1_epilogue")
                    real_fc1_output, _ = ext.get_gmem_tensor(
                        "c", fc1_output_gemm, work_tile_info,
                    )
                    real_fc1_output_sf, _ = ext.get_gmem_tensor(
                        "sfc", fc1_output_sf_gemm, work_tile_info,
                    )
                    if cutlass.const_expr(token_comm_args is not None):
                        real_topk_scores, _ = ext.get_gmem_tensor(
                            "topk", topk_scores, work_tile_info,
                        )
                    fc1_col_base = (
                        work_tile_info.tile_m_idx * cutlass.Int32(downproj_per_cta)
                    )
                    tile_token_base = work_tile_info.tile_n_idx * cutlass.Int32(
                        self.mma_tiler[1]
                    )
                    # SF tensor K coordinates are element coordinates.  The
                    # within-block vec mode is stride-zero, so block starts
                    # are 0, sf_vec_size, 2*sf_vec_size, ... .
                    sf_col = fc1_col_base
                    rcp_limit = Fp8E4M3RcpLimit
                    if cutlass.const_expr(self.fc1_output_dtype == cutlass.Float8E5M2):
                        rcp_limit = Fp8E5M2RcpLimit
                    q_limit = cutlass.Float32(1.0 / rcp_limit)

                    for ng_pair in cutlass.range_constexpr(0, n_groups // 2):
                        acc0 = accumulators[None, None, ng_pair * 2]
                        acc1 = accumulators[None, None, ng_pair * 2 + 1]

                        token0 = lane_t * cutlass.Int32(2)
                        token1 = token0 + cutlass.Int32(1)
                        token2 = token0 + cutlass.Int32(MMA_N)
                        token3 = token1 + cutlass.Int32(MMA_N)
                        token_pair_base = cutlass.Int32(ng_pair * 2 * MMA_N)
                        tile_token0 = token_pair_base + token0
                        tile_token1 = token_pair_base + token1
                        tile_token2 = token_pair_base + token2
                        tile_token3 = token_pair_base + token3

                        gate0 = acc0[0]
                        gate1 = acc0[1]
                        gate2 = acc1[0]
                        gate3 = acc1[1]
                        up0 = acc0[2]
                        up1 = acc0[3]
                        up2 = acc1[2]
                        up3 = acc1[3]
                        val0 = up0 * gate0 * cute.arch.rcp_approx(
                            cute.math.exp2(gate0 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val1 = up1 * gate1 * cute.arch.rcp_approx(
                            cute.math.exp2(gate1 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val2 = up2 * gate2 * cute.arch.rcp_approx(
                            cute.math.exp2(gate2 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )
                        val3 = up3 * gate3 * cute.arch.rcp_approx(
                            cute.math.exp2(gate3 * (-Log2E))
                            + cutlass.Float32(1.0)
                        )

                        raw_token0 = tile_token_base + tile_token0
                        raw_token1 = tile_token_base + tile_token1
                        raw_token2 = tile_token_base + tile_token2
                        raw_token3 = tile_token_base + tile_token3
                        if cutlass.const_expr(token_comm_args is not None):
                            if tile_token0 < work_tile_info.valid_tokens_in_tile:
                                val0 = val0 * cutlass.Float32(
                                    real_topk_scores[raw_token0]
                                )
                            if tile_token1 < work_tile_info.valid_tokens_in_tile:
                                val1 = val1 * cutlass.Float32(
                                    real_topk_scores[raw_token1]
                                )
                            if tile_token2 < work_tile_info.valid_tokens_in_tile:
                                val2 = val2 * cutlass.Float32(
                                    real_topk_scores[raw_token2]
                                )
                            if tile_token3 < work_tile_info.valid_tokens_in_tile:
                                val3 = val3 * cutlass.Float32(
                                    real_topk_scores[raw_token3]
                                )

                        if tile_token0 >= work_tile_info.valid_tokens_in_tile:
                            val0 = cutlass.Float32(0.0)
                        if tile_token1 >= work_tile_info.valid_tokens_in_tile:
                            val1 = cutlass.Float32(0.0)
                        if tile_token2 >= work_tile_info.valid_tokens_in_tile:
                            val2 = cutlass.Float32(0.0)
                        if tile_token3 >= work_tile_info.valid_tokens_in_tile:
                            val3 = cutlass.Float32(0.0)
                        amax0 = cute.math.absf(val0)
                        amax1 = cute.math.absf(val1)
                        amax2 = cute.math.absf(val2)
                        amax3 = cute.math.absf(val3)
                        for xor_mask in (4, 8, 16):
                            peer_lane = lane_idx ^ cutlass.Int32(xor_mask)
                            amax0 = cute.arch.fmax(
                                amax0, cute.arch.shuffle_sync(amax0, peer_lane)
                            )
                            amax1 = cute.arch.fmax(
                                amax1, cute.arch.shuffle_sync(amax1, peer_lane)
                            )
                            amax2 = cute.arch.fmax(
                                amax2, cute.arch.shuffle_sync(amax2, peer_lane)
                            )
                            amax3 = cute.arch.fmax(
                                amax3, cute.arch.shuffle_sync(amax3, peer_lane)
                            )
                        if lane_g == cutlass.Int32(0):
                            sFc1Swiglu[token0, compute_warp] = amax0
                            sFc1Swiglu[token1, compute_warp] = amax1
                            sFc1Swiglu[token2, compute_warp] = amax2
                            sFc1Swiglu[token3, compute_warp] = amax3

                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=32 * len(self.compute_warp_id),
                        )

                        amax0 = cutlass.Float32(0.0)
                        amax1 = cutlass.Float32(0.0)
                        amax2 = cutlass.Float32(0.0)
                        amax3 = cutlass.Float32(0.0)
                        if lane_g == cutlass.Int32(0):
                            for warp in cutlass.range_constexpr(
                                0, len(self.compute_warp_id)
                            ):
                                amax0 = cute.arch.fmax(
                                    amax0, sFc1Swiglu[token0, warp]
                                )
                                amax1 = cute.arch.fmax(
                                    amax1, sFc1Swiglu[token1, warp]
                                )
                                amax2 = cute.arch.fmax(
                                    amax2, sFc1Swiglu[token2, warp]
                                )
                                amax3 = cute.arch.fmax(
                                    amax3, sFc1Swiglu[token3, warp]
                                )
                        amax0 = cute.arch.shuffle_sync(amax0, lane_t)
                        amax1 = cute.arch.shuffle_sync(amax1, lane_t)
                        amax2 = cute.arch.shuffle_sync(amax2, lane_t)
                        amax3 = cute.arch.shuffle_sync(amax3, lane_t)

                        scale0 = cvt_f32_to_f8_to_f32(
                            amax0 * cutlass.Float32(rcp_limit), self.sf_dtype,
                        )
                        scale1 = cvt_f32_to_f8_to_f32(
                            amax1 * cutlass.Float32(rcp_limit), self.sf_dtype,
                        )
                        scale2 = cvt_f32_to_f8_to_f32(
                            amax2 * cutlass.Float32(rcp_limit), self.sf_dtype,
                        )
                        scale3 = cvt_f32_to_f8_to_f32(
                            amax3 * cutlass.Float32(rcp_limit), self.sf_dtype,
                        )
                        inv0 = cute.arch.fmin(
                            cute.arch.rcp_approx(scale0), cutlass.Float32(Fp32Max)
                        )
                        inv1 = cute.arch.fmin(
                            cute.arch.rcp_approx(scale1), cutlass.Float32(Fp32Max)
                        )
                        inv2 = cute.arch.fmin(
                            cute.arch.rcp_approx(scale2), cutlass.Float32(Fp32Max)
                        )
                        inv3 = cute.arch.fmin(
                            cute.arch.rcp_approx(scale3), cutlass.Float32(Fp32Max)
                        )

                        if lane_g == cutlass.Int32(0):
                            if tile_token0 < work_tile_info.valid_tokens_in_tile:
                                real_fc1_output_sf[raw_token0, sf_col, 0] = (
                                    amax0 * cutlass.Float32(rcp_limit)
                                ).to(self.sf_dtype)
                            if tile_token1 < work_tile_info.valid_tokens_in_tile:
                                real_fc1_output_sf[raw_token1, sf_col, 0] = (
                                    amax1 * cutlass.Float32(rcp_limit)
                                ).to(self.sf_dtype)
                            if tile_token2 < work_tile_info.valid_tokens_in_tile:
                                real_fc1_output_sf[raw_token2, sf_col, 0] = (
                                    amax2 * cutlass.Float32(rcp_limit)
                                ).to(self.sf_dtype)
                            if tile_token3 < work_tile_info.valid_tokens_in_tile:
                                real_fc1_output_sf[raw_token3, sf_col, 0] = (
                                    amax3 * cutlass.Float32(rcp_limit)
                                ).to(self.sf_dtype)

                        q0 = cutlass.Float32(0.0)
                        q1 = cutlass.Float32(0.0)
                        q2 = cutlass.Float32(0.0)
                        q3 = cutlass.Float32(0.0)
                        if tile_token0 < work_tile_info.valid_tokens_in_tile:
                            q0 = cute.arch.fmin(
                                q_limit,
                                cute.arch.fmax(
                                    -q_limit,
                                    val0 * inv0,
                                ),
                            )
                        if tile_token1 < work_tile_info.valid_tokens_in_tile:
                            q1 = cute.arch.fmin(
                                q_limit,
                                cute.arch.fmax(
                                    -q_limit,
                                    val1 * inv1,
                                ),
                            )
                        if tile_token2 < work_tile_info.valid_tokens_in_tile:
                            q2 = cute.arch.fmin(
                                q_limit,
                                cute.arch.fmax(
                                    -q_limit,
                                    val2 * inv2,
                                ),
                            )
                        if tile_token3 < work_tile_info.valid_tokens_in_tile:
                            q3 = cute.arch.fmin(
                                q_limit,
                                cute.arch.fmax(
                                    -q_limit,
                                    val3 * inv3,
                                ),
                            )

                        rFc1OutputLocal[0] = q0
                        rFc1OutputLocal[1] = q1
                        rFc1OutputLocal[2] = q2
                        rFc1OutputLocal[3] = q3
                        transpose_src_lane = (
                            (lane_g >> cutlass.Int32(1))
                            + lane_t * cutlass.Int32(8)
                        )
                        transpose_src_lane_1 = (
                            transpose_src_lane + cutlass.Int32(4)
                        )
                        packed_local = cutlass.Int32(
                            cvt_f32x4_to_f8x4_pack_i32(
                                rFc1OutputLocal,
                                self.fc1_output_dtype,
                            )
                        )
                        packed0 = cute.arch.shuffle_sync(
                            packed_local, transpose_src_lane
                        )
                        packed1 = cute.arch.shuffle_sync(
                            packed_local, transpose_src_lane_1
                        )
                        packed_output = (
                            (packed0 & cutlass.Int32(0x000000FF))
                            | (
                                (packed1 & cutlass.Int32(0x000000FF))
                                << cutlass.Int32(8)
                            )
                            | (packed0 & cutlass.Int32(0x00FF0000))
                            | (
                                (packed1 & cutlass.Int32(0x00FF0000))
                                << cutlass.Int32(8)
                            )
                        )
                        if (lane_g & cutlass.Int32(1)) != cutlass.Int32(0):
                            packed_output = (
                                ((packed0 >> cutlass.Int32(8)) & cutlass.Int32(0xFF))
                                | (packed1 & cutlass.Int32(0x0000FF00))
                                | (
                                    ((packed0 >> cutlass.Int32(24)) & cutlass.Int32(0xFF))
                                    << cutlass.Int32(16)
                                )
                                | (packed1 & cutlass.Int32(-0x01000000))
                            )
                        rFc1OutputPacked[0] = packed_output

                        sFc1OutputPair = cute.composition(
                            cute.local_tile(
                                sFc1Output,
                                (16, downproj_per_cta),
                                (ng_pair, 0),
                            ),
                            stsm_remap_layout,
                        )
                        stsm_dst = stsm_thr_copy.partition_D(sFc1OutputPair)
                        cute.copy(stsm_tiled_copy, rFc1Output, stsm_dst)

                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=32 * len(self.compute_warp_id),
                        )

                    chunks_per_row = downproj_per_cta // 4
                    store_iters = (
                        self.mma_tiler[1] * chunks_per_row
                    ) // (32 * len(self.compute_warp_id))
                    for store_iter in cutlass.range_constexpr(0, store_iters):
                        linear_chunk = (
                            tidx
                            + cutlass.Int32(
                                store_iter * 32 * len(self.compute_warp_id)
                            )
                        )
                        store_token = linear_chunk // cutlass.Int32(chunks_per_row)
                        store_dp = (
                            linear_chunk % cutlass.Int32(chunks_per_row)
                        ) * cutlass.Int32(4)
                        for j in cutlass.range_constexpr(0, 4):
                            rFc1Store[j] = sFc1Output[
                                store_token,
                                store_dp + cutlass.Int32(j),
                            ]
                        if store_token < work_tile_info.valid_tokens_in_tile:
                            output_row = tile_token_base + store_token
                            output_col = fc1_col_base + store_dp
                            output_ptr = (
                                real_fc1_output.iterator
                                + output_row * real_fc1_output.stride[0]
                                + output_col * real_fc1_output.stride[1]
                            ).align(4)
                            output_i32 = cute.make_tensor(
                                cute.recast_ptr(output_ptr, dtype=cutlass.Int32),
                                cute.make_layout(1),
                            )
                            output_i32[0] = rFc1StoreI32[0]
                    cute.arch.fence_acq_rel_sys()
                    cute.arch.barrier(
                        barrier_id=self.epilog_sync_bar_id,
                        number_of_threads=32 * len(self.compute_warp_id),
                    )

                    counter_slot = (
                        work_tile_info.cumulative_data_physical_row
                        // cutlass.Int32(self.mma_tiler[1])
                        + work_tile_info.tile_n_idx
                    )
                    if cutlass.const_expr(token_comm_args is not None):
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                ready_bundle_idx = (
                                    work_tile_info.tile_m_idx
                                    // cutlass.Int32(
                                        fc1_tiles_per_fc2_k_tile
                                        * self.fc2_ready_bundle_k_tiles
                                    )
                                )
                                cute.arch.atomic_add(
                                    fc1_done_counter.iterator
                                    + counter_slot * fc2_ready_bundle_cnt
                                    + ready_bundle_idx,
                                    cutlass.Int32(1),
                                    sem="release",
                                    scope="gpu",
                                )
                    else:
                        if lane_idx == cutlass.Int32(0):
                            cute.arch.atomic_add(
                                fc1_done_counter.iterator + counter_slot,
                                cutlass.Int32(1),
                                sem="release",
                                scope="sys",
                            )
                else:
                    iket.range_push("sm120_fc2_store")
                    hidden_base = (
                        work_tile_info.tile_m_idx * cutlass.Int32(self.mma_tiler[0])
                    )
                    tile_token_base = work_tile_info.tile_n_idx * cutlass.Int32(
                        self.mma_tiler[1]
                    )
                    for ng in cutlass.range_constexpr(0, n_groups):
                        acc = accumulators[None, None, ng]


                        token0 = cutlass.Int32(ng * MMA_N) + lane_t * cutlass.Int32(2)
                        token1 = token0 + cutlass.Int32(1)
                        hidden0 = hidden_base + compute_warp * cutlass.Int32(16) + lane_g
                        hidden1 = hidden0 + cutlass.Int32(8)
                        pool_token0 = (
                            work_tile_info.cumulative_data_physical_row
                            + tile_token_base
                            + token0
                        )
                        pool_token1 = (
                            work_tile_info.cumulative_data_physical_row
                            + tile_token_base
                            + token1
                        )
                        if cutlass.const_expr(self.fc2_packed_store):
                            partner_lane = lane_idx ^ cutlass.Int32(4)
                            rFc2Store[0] = acc[0].to(self.fc2_output_dtype)
                            rFc2Store[1] = cute.arch.shuffle_sync(
                                acc[0], partner_lane
                            ).to(self.fc2_output_dtype)
                            rFc2Store[2] = acc[2].to(self.fc2_output_dtype)
                            rFc2Store[3] = cute.arch.shuffle_sync(
                                acc[2], partner_lane
                            ).to(self.fc2_output_dtype)
                            rFc2Store[4] = acc[1].to(self.fc2_output_dtype)
                            rFc2Store[5] = cute.arch.shuffle_sync(
                                acc[1], partner_lane
                            ).to(self.fc2_output_dtype)
                            rFc2Store[6] = acc[3].to(self.fc2_output_dtype)
                            rFc2Store[7] = cute.arch.shuffle_sync(
                                acc[3], partner_lane
                            ).to(self.fc2_output_dtype)
                        if token0 < work_tile_info.valid_tokens_in_tile:
                            if cutlass.const_expr(self.fc2_packed_store):
                                if (lane_g & cutlass.Int32(1)) == cutlass.Int32(0):
                                    if cutlass.const_expr(
                                        token_comm_args is not None
                                        and not self.token_back_by_dispatch
                                    ):
                                        md0 = TokenSrcMetadata.load(
                                            token_comm_args.token_src_metadata.iterator.toint()
                                            + cutlass.Int64(pool_token0)
                                            * cutlass.Int64(TokenSrcMetadata.nbytes)
                                        )
                                        local_row0 = cute.slice_(
                                            fc2_output,
                                            (md0.src_token, md0.src_topk, None),
                                        )
                                        row0 = cute.make_tensor(
                                            token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(
                                                local_row0.iterator, md0.src_rank
                                            ),
                                            local_row0.layout,
                                        )
                                    else:
                                        row0 = cute.slice_(
                                            fc2_output, (pool_token0, 0, None)
                                        )
                                    row0_hidden0 = (
                                        row0.iterator + hidden0
                                    ).align(4)
                                    row0_hidden1 = (
                                        row0.iterator + hidden1
                                    ).align(4)
                                    row0_i32_0 = cute.make_tensor(
                                        cute.recast_ptr(
                                            row0_hidden0, dtype=cutlass.Int32
                                        ),
                                        cute.make_layout(1),
                                    )
                                    row0_i32_1 = cute.make_tensor(
                                        cute.recast_ptr(
                                            row0_hidden1, dtype=cutlass.Int32
                                        ),
                                        cute.make_layout(1),
                                    )
                                    row0_i32_0[0] = rFc2StoreI32[0]
                                    row0_i32_1[0] = rFc2StoreI32[1]
                            else:
                                if cutlass.const_expr(
                                    token_comm_args is not None
                                    and not self.token_back_by_dispatch
                                ):
                                    md0 = TokenSrcMetadata.load(
                                        token_comm_args.token_src_metadata.iterator.toint()
                                        + cutlass.Int64(pool_token0)
                                        * cutlass.Int64(TokenSrcMetadata.nbytes)
                                    )
                                    local_row0 = cute.slice_(
                                        fc2_output,
                                        (md0.src_token, md0.src_topk, None),
                                    )
                                    peer_row0 = cute.make_tensor(
                                        token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(
                                            local_row0.iterator, md0.src_rank
                                        ),
                                        local_row0.layout,
                                    )
                                    peer_row0[hidden0] = acc[0].to(self.fc2_output_dtype)
                                    peer_row0[hidden1] = acc[2].to(self.fc2_output_dtype)
                                else:
                                    fc2_output[pool_token0, 0, hidden0] = acc[0].to(
                                        self.fc2_output_dtype
                                    )
                                    fc2_output[pool_token0, 0, hidden1] = acc[2].to(
                                        self.fc2_output_dtype
                                    )
                        if token1 < work_tile_info.valid_tokens_in_tile:
                            if cutlass.const_expr(self.fc2_packed_store):
                                if (lane_g & cutlass.Int32(1)) == cutlass.Int32(0):
                                    if cutlass.const_expr(
                                        token_comm_args is not None
                                        and not self.token_back_by_dispatch
                                    ):
                                        md1 = TokenSrcMetadata.load(
                                            token_comm_args.token_src_metadata.iterator.toint()
                                            + cutlass.Int64(pool_token1)
                                            * cutlass.Int64(TokenSrcMetadata.nbytes)
                                        )
                                        local_row1 = cute.slice_(
                                            fc2_output,
                                            (md1.src_token, md1.src_topk, None),
                                        )
                                        row1 = cute.make_tensor(
                                            token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(
                                                local_row1.iterator, md1.src_rank
                                            ),
                                            local_row1.layout,
                                        )
                                    else:
                                        row1 = cute.slice_(
                                            fc2_output, (pool_token1, 0, None)
                                        )
                                    row1_hidden0 = (
                                        row1.iterator + hidden0
                                    ).align(4)
                                    row1_hidden1 = (
                                        row1.iterator + hidden1
                                    ).align(4)
                                    row1_i32_0 = cute.make_tensor(
                                        cute.recast_ptr(
                                            row1_hidden0, dtype=cutlass.Int32
                                        ),
                                        cute.make_layout(1),
                                    )
                                    row1_i32_1 = cute.make_tensor(
                                        cute.recast_ptr(
                                            row1_hidden1, dtype=cutlass.Int32
                                        ),
                                        cute.make_layout(1),
                                    )
                                    row1_i32_0[0] = rFc2StoreI32[2]
                                    row1_i32_1[0] = rFc2StoreI32[3]
                            else:
                                if cutlass.const_expr(
                                    token_comm_args is not None
                                    and not self.token_back_by_dispatch
                                ):
                                    md1 = TokenSrcMetadata.load(
                                        token_comm_args.token_src_metadata.iterator.toint()
                                        + cutlass.Int64(pool_token1)
                                        * cutlass.Int64(TokenSrcMetadata.nbytes)
                                    )
                                    local_row1 = cute.slice_(
                                        fc2_output,
                                        (md1.src_token, md1.src_topk, None),
                                    )
                                    peer_row1 = cute.make_tensor(
                                        token_comm_args.peer_rank_ptr_mapper.ptr_map_to_rank(
                                            local_row1.iterator, md1.src_rank
                                        ),
                                        local_row1.layout,
                                    )
                                    peer_row1[hidden0] = acc[1].to(self.fc2_output_dtype)
                                    peer_row1[hidden1] = acc[3].to(self.fc2_output_dtype)
                                else:
                                    fc2_output[pool_token1, 0, hidden0] = acc[1].to(
                                        self.fc2_output_dtype
                                    )
                                    fc2_output[pool_token1, 0, hidden1] = acc[3].to(
                                        self.fc2_output_dtype
                                    )

                    if cutlass.const_expr(self.token_back_by_dispatch):
                        # token-back consumes this tile from the local FC2
                        # workspace only after all four compute warps have
                        # made their disjoint hidden slices system-visible.
                        # Publish exactly once per scheduler FC2 work tile.
                        # Grouped token-back aggregates per expert; streaming
                        # token-back uses one counter slot per token tile.
                        cute.arch.fence_acq_rel_sys()
                        cute.arch.barrier(
                            barrier_id=self.epilog_sync_bar_id,
                            number_of_threads=32 * len(self.compute_warp_id),
                        )
                        if compute_warp == cutlass.Int32(0):
                            if lane_idx == cutlass.Int32(0):
                                counter_slot = work_tile_info.expert_idx
                                if cutlass.const_expr(self.streaming_fc12):
                                    counter_slot = (
                                        work_tile_info.cumulative_data_physical_row
                                        // cutlass.Int32(self.mma_tiler[1])
                                        + work_tile_info.tile_n_idx
                                    )
                                cute.arch.atomic_add(
                                    token_comm_args.fc2_done_counter.iterator
                                    + counter_slot,
                                    cutlass.Int32(1),
                                    sem="release",
                                    scope="sys",
                                )

                iket.range_pop()
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()


        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (warp 8-11; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` means warps 8-11 don't exist at all
        # (threads_per_cta = 256 in lean mode), so the hook call is
        # entirely const_expr-eliminated.  When ``enable_token_comm=True``
        # the subclass implements the full dispatch chain inside this
        # hook (prep -> cross-rank barrier -> per-token pull -> release
        # to fc1 -> arrive on dispatch-to-sched NamedBarrier).
        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx >= self.dispatch_warp_id[0]:
                cute.arch.warpgroup_reg_dealloc(self.task_reg_cnt)

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
        # Kernel tail hook (MegaMoE-only path; lean base = no-op)
        # ════════════════════════════════════════════════════════════════════
        #
        # All 12 warps fall through to this point in MegaMoE mode (warp
        # 8-11 already exited the dispatch warp body hook above; warps
        # 0-7 just finished GEMM / epi work).  The subclass hook owns
        # the kernel-tail rendezvous (12-warp NamedBarrier) and the
        # cross-rank NVLink release.  Base no-op: lean path has no peer
        # ranks and no kernel-tail concept.
        lane_idx = cute.arch.lane_idx()
        self.token_comm_hook_kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )
