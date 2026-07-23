# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Swap-AB fused fc1+fc2 GLU FP8 kernel for SM90."""

from typing import Literal, Optional, Tuple, Type, Union

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from src.iket_compat import iket
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.typing import Float32
import cutlass.utils as utils
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils.hopper_helpers as sm90_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils

from moe_hopper_fp8.epilogue_fp8_swapab import (
    SwapABFp8GluEpilogue,
    SwapABTileMChoices,
    SwapABTokenTileNChoices,
)
from moe_hopper_fp8.kernel_fp8_glu_fc12 import (
    Sm90SwigluFp8Fc12Kernel as _Sm90Fp8Fc12KernelBase,
)
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from moe_nvfp4_swapab.custom_ext import SwapABSwigluFp4Fc12SchedExtension
from common.megamoe_constants import (
    Fp8BlockScaleK,
    Fp8DispatchScaleAtomK,
    Fp8E4M3RcpLimit,
    Fp8E5M2RcpLimit,
    Fp8E8M0SfVecSize,
    Fp8Fc2ActivationScaleK,
    Fp8GateUpInterleave,
    Fp8WeightScaleBlockK,
    Fp8WeightScaleBlockN,
)
from moe_nvfp4_swapab.moe_utils import spin_wait


# =============================================================================
# Sm90SwapABSwigluFp8Fc12Kernel
# =============================================================================

class Sm90SwapABSwigluFp8Fc12Kernel(_Sm90Fp8Fc12KernelBase):

    _setmaxnreg_min = 24
    _setmaxnreg_max = 256
    _setmaxnreg_granularity = 8
    _sm90_cta_register_budget = 65536

    # SMEM budget for all "non-problem-tensor" buffers (mbarriers, sched
    # work-tile buffer, and dispatch scratch).  Reserved at host side in
    # ``_compute_stages``.  Bump if ``SharedStorage`` over-allocates SMEM.
    _SmemMiscBudget = 1024

    # Supported (ab_dtype, sf_vec_size) pairings.  The sf_vec_size parameter is
    # the legacy per-tensor E8M0 SF byte granularity: one E8M0 byte covers 32
    # FP8 elements, and four such bytes are carried as one 128-element dispatch
    # scale atom. Gate/up interleave is tracked separately by
    # Fp8GateUpInterleave.
    VALID_AB_DTYPE_SF_SIZE: dict = {
        Fp8E8M0SfVecSize: (cutlass.Float8E4M3FN, cutlass.Float8E5M2,),
    }

    # Interleave granularity for gate and up in SwiGLU / GeGlu
    GateUpInterleave: int = Fp8GateUpInterleave

    def __init__(
        self,
        # Geometry (no defaults; perf-sensitive + coupled by the validator):
        # Swap-AB uses M=weight channels and N=tokens.
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
        sf_vec_size: int = Fp8E8M0SfVecSize,
        ab_dtype: Type[cutlass.Numeric] = cutlass.Float4E2M1FN,
        fp8_scale_mode: Literal["per_tensor", "blockwise"] = "per_tensor",
        fp8_accum_mode: Literal["1xacc", "2xacc"] = "1xacc",
        scenario: Literal["2Dx3D"] = "2Dx3D",
        fc2_in_kernel_topk_reduce: bool = False,
        apply_topk_in_fc1: bool = False,
        token_back_by_dispatch: bool = False,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        gate_up_clamp: Optional[float] = None,
    ) -> None:
        # v1 only implements the static scheduler. Dynamic CLC with drain
        # auxiliary warps remains future work.
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
                    f"Expected one of: {[t.__name__ for t in valid_ab]}."
                )
        else:
            raise NotImplementedError(
                f"sf_vec_size must be {Fp8E8M0SfVecSize} "
                "(FP8 legacy E8M0 SF ABI)"
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

        # 1CTA-only Hopper FP8 path.
        m, n, _k = mma_tiler_mnk
        if use_2cta_instrs:
            raise ValueError(
                "Sm90SwapABSwigluFp8Fc12Kernel in moe_hopper_fp8 is "
                "1CTA-only; "
                "use_2cta_instrs must be False."
            )
        if m not in SwapABTileMChoices or n not in SwapABTokenTileNChoices:
            raise ValueError(
                "Sm90SwapABSwigluFp8Fc12Kernel only supports "
                f"mma_tiler M in {SwapABTileMChoices} and N in "
                f"{SwapABTokenTileNChoices} with "
                "use_2cta_instrs=False; "
                f"got mma_tiler_mnk={mma_tiler_mnk}, use_2cta_instrs={use_2cta_instrs}."
            )

        # Store ab_dtype so workspace-size helpers can use it without tensors.
        self.ab_dtype = ab_dtype
        if fp8_scale_mode not in ("per_tensor", "blockwise"):
            raise ValueError(
                f"fp8_scale_mode must be 'per_tensor' or 'blockwise', "
                f"got {fp8_scale_mode!r}."
            )
        self.fp8_scale_mode = fp8_scale_mode
        if fp8_accum_mode not in ("1xacc", "2xacc"):
            raise ValueError(
                "fp8_accum_mode must be '1xacc' or '2xacc', "
                f"got {fp8_accum_mode!r}."
            )
        self.fp8_accum_mode = fp8_accum_mode
        if ab_dtype == cutlass.Float8E4M3FN:
            self.fp8_output_rcp_limit = Fp8E4M3RcpLimit
        elif ab_dtype == cutlass.Float8E5M2:
            self.fp8_output_rcp_limit = Fp8E5M2RcpLimit
        else:
            raise ValueError(
                f"Unsupported Hopper FP8 ab_dtype for output quant: {ab_dtype}."
            )

        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.token_back_by_dispatch = token_back_by_dispatch
        self.epi_flag_batch = epi_flag_batch
        self.gate_up_clamp = (
            abs(gate_up_clamp) if gate_up_clamp is not None else None
        )

        self.acc_dtype = acc_dtype
        if cluster_shape_mnk != (1, 1, 1):
            raise ValueError(
                "Sm90SwapABSwigluFp8Fc12Kernel in moe_hopper_fp8 is "
                "1CTA-only; "
                f"cluster_shape_mnk must be (1, 1, 1), got {cluster_shape_mnk}."
            )

        self.mma_tiler_mnk = mma_tiler_mnk
        self.is_swap_ab = True
        self._set_iket_range_names()
        self.cluster_shape_mn = (cluster_shape_mnk[0], cluster_shape_mnk[1])
        self.use_2cta_instrs = False
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
        self.arch = "sm_90"

        self._validate_mma_tiler_and_cluster_shape()
        # Each physical warpgroup owns a contiguous M=128 half. Its tiled MMA
        # uses one m64nN atom and exposes two independent M64 rest tiles.
        self.wgmma_tile_m = 128
        self.wgmma_tile_n = n
        self.mma_tiler = self.mma_tiler_mnk
        self.token_tile_size = n
        self.wgmma_m_splits = self.mma_tiler[0] // self.wgmma_tile_m
        self.wgmma_tiler = (
            self.wgmma_tile_m,
            self.wgmma_tile_n,
            self.mma_tiler[2],
        )

        self.atom_layout_mnk = (1, 1, 1)

        # Warp specialization. Each physical SM90 WGMMA+epilogue warpgroup
        # owns one contiguous M=128 tile. M=128 uses one group; M=256 uses two.
        # Keep one legacy MMA-only empty warp after the scheduler.
        self.occupancy = 1
        self.epilogue_warps_per_warpgroup = 4
        self.epilogue_warp_id = tuple(
            range(
                self.epilogue_warps_per_warpgroup
                * self.wgmma_m_splits
            )
        )
        self.wgmma_warpgroup_count = self.wgmma_m_splits
        self.tma_a_warp_id = len(self.epilogue_warp_id)
        self.tma_b_warp_id = self.tma_a_warp_id + 1
        self.sched_warp_id = self.tma_b_warp_id + 1
        self.empty_warp_id = self.sched_warp_id + 1
        self.threads_per_cta = 32 * len(
            (
                *self.epilogue_warp_id,
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.sched_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier IDs. FC1 store uses two WG-local barriers at ids 2/3;
        # this kernel forwards the task-boundary and FC1-store base IDs to the
        # epilogue ctor. IDs 8/9/10 are reserved by Mega token-comm hooks.
        self.epilog_sync_bar_id = 1
        self.fc1_store_sync_bar_id = 2

        # MegaMoE toggle.  False for the lean base; subclasses (e.g.
        # ``Sm90MegaMoEFp8Kernel``) set this to True in their own
        # ``__init__`` after ``super().__init__`` returns.
        # ``_setup_attributes()`` reads it inside ``__call__`` to expand the
        # warp topology to the MegaMoE layout.
        self.enable_token_comm: bool = False
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.token_back_standalone: bool = False

        # MegaMoE-only register policy.  These defaults are safe for the
        # standalone-token-back CTA; Mega subclasses can raise epi_reg_cnt when
        # token-back reuses dispatch warps and the CTA has fewer resident warps.
        self.epi_reg_cnt = 200
        self.tma_a_reg_cnt = 32
        self.tma_b_reg_cnt = 32
        self.sched_reg_cnt = 40
        self.empty_reg_cnt = 24
        self.dispatch_reg_cnt = 48
        self.token_back_reg_cnt = 32
        self.task_reg_cnt = 32

        self.smem_capacity = utils.get_smem_capacity_in_bytes(self.arch)

    def _validate_mma_tiler_and_cluster_shape(self) -> None:
        """Validate user-provided geometry against v1 fused-fc12 constraints.

        ``mma_tiler_n`` is a codegen-time token tile selected from the Hopper
        WGMMA-supported specializations used by this kernel.
        """
        m, n, k = self.mma_tiler_mnk
        cm, cn = self.cluster_shape_mn

        if m not in SwapABTileMChoices or n not in SwapABTokenTileNChoices:
            raise ValueError(
                "Swap-AB Hopper FP8 requires M in "
                f"{SwapABTileMChoices} and N in {SwapABTokenTileNChoices}, "
                f"got ({m}, {n})."
            )

        dispatch_scale_atom_k = Fp8DispatchScaleAtomK
        if k % dispatch_scale_atom_k != 0:
            raise ValueError(
                f"mma_tiler K ({k}) must be a multiple of "
                f"FP8 dispatch scale atom K = {dispatch_scale_atom_k}"
            )

        if (cm, cn) != (1, 1):
            raise ValueError(
                f"1CTA Hopper FP8 path requires cluster_shape_mn=(1, 1), got {(cm, cn)}."
            )

        is_pow2 = lambda x: x > 0 and (x & (x - 1)) == 0
        if cm * cn > 16 or not is_pow2(cm) or not is_pow2(cn) or cm > 4 or cn > 4:
            raise ValueError(
                f"Invalid cluster_shape ({cm}, {cn}): each dim must be "
                f"a power of 2 and <= 4, product must be <= 16"
            )

        # No cluster multicast is wired in this 1CTA-only fork.

    def _setup_attributes(self) -> None:
        """Set up MMA / cluster / tile shapes, SMEM layouts, stage counts.

        The fc12 path shares ``mma_tiler_mnk`` and SMEM layouts across phases.
        """
        if self.enable_token_comm:
            dispatch_warp_start = self.empty_warp_id + 1
            self.dispatch_warp_id = tuple(
                range(dispatch_warp_start, dispatch_warp_start + 4)
            )
            if self.token_back_standalone:
                token_back_warp_start = dispatch_warp_start + 4
                self.token_back_warp_id = tuple(
                    range(token_back_warp_start, token_back_warp_start + 4)
                )
            token_back_warp_ids = (
                self.token_back_warp_id if self.token_back_standalone else ()
            )
            self.threads_per_cta = 32 * len(
                (
                    *self.epilogue_warp_id,
                    self.tma_a_warp_id,
                    self.tma_b_warp_id,
                    self.sched_warp_id,
                    self.empty_warp_id,
                    *self.dispatch_warp_id,
                    *token_back_warp_ids,
                )
            )

        self.mma_inst_shape_mn = (self.mma_tiler[0], self.mma_tiler[1])

        tiled_mma = self._create_tiled_mma()

        mma_inst_shape_k = cute.size(tiled_mma.shape_mnk, mode=[2])
        assert self.mma_tiler[2] % mma_inst_shape_k == 0, (
            f"mma_tiler K ({self.mma_tiler[2]}) must be a multiple of "
            f"MMA instruction K ({mma_inst_shape_k})"
        )

        # Hopper WGMMA's ``thr_id`` describes the 128-thread warpgroup, not a
        # CTA-group split along M.  The MoE scheduler must see the full CTA
        # tile shape.
        self.cta_tile_shape_mnk = self.mma_tiler_mnk

        self.cluster_layout_mnk = cute.make_layout((*self.cluster_shape_mn, 1))
        self.cluster_layout_vmnk = cute.make_layout((1, *self.cluster_layout_mnk.shape))

        # Multicast CTA counts.  This fork currently validates cluster=(1,1,1),
        # so both resolve to one; keep the fields for API parity with helpers.
        self.num_mcast_ctas_a = self.cluster_shape_mn[1]
        self.num_mcast_ctas_b = self.cluster_shape_mn[0]
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        # Epilogue is autonomous: it owns acc stage, subtile dispatch, TMA
        # commit/drain, and piggyback red.add decisions.
        # The MMA->epilogue acc pipeline is a strict single-stage handoff.
        # fc2 output dtype is hard-coded BFloat16 in both epilogues.
        _epi_common = dict(
            mma_tiler_mnk=self.mma_tiler_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            sf_vec_size=self.sf_vec_size,
            fc1_output_dtype=self.fc1_output_dtype,
            fc1_output_layout=self.fc1_output_layout,
            acc_dtype=self.acc_dtype,
            epilog_sync_bar_id=self.epilog_sync_bar_id,
            fc1_store_sync_bar_id=self.fc1_store_sync_bar_id,
            epilogue_warp_ids=self.epilogue_warp_id,
            static_expert_shape=self.static_expert_shape,
            fc2_in_kernel_topk_reduce=self.fc2_in_kernel_topk_reduce,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            glu_clamp=self.gate_up_clamp,
            fp8_scale_mode=self.fp8_scale_mode,
            fp8_output_rcp_limit=self.fp8_output_rcp_limit,
        )
        self.epilogue = SwapABFp8GluEpilogue(**_epi_common)

        if self.num_sched_stages is None:
            self.num_sched_stages = 2

        # sC stages are WG-private FC1 store buffers: each WG folds M=128 to
        # one token-N x output-channel64 stage. M=128/M=256 therefore use
        # one/two stages respectively.
        self.num_c_stage = self.epilogue.subtile_cnt
        c_bytes_total = (
            self.epilogue.bytes_per_stage * self.num_c_stage
            + self.epilogue.fc1_amax_smem_bytes
        )

        (
            self.num_acc_stage,
            self.num_ab_stage,
            self.num_sched_stages,
        ) = self._compute_stages(
            tiled_mma,
            self.mma_tiler,
            self.a_dtype,
            self.b_dtype,
            c_bytes_total,
            self.smem_capacity,
            self.occupancy,
            self.num_sched_stages,
        )

        self.a_smem_layout_staged = sm90_utils.make_smem_layout_a(
            self.a_layout, self.mma_tiler, self.a_dtype, self.num_ab_stage,
        )
        self.b_smem_layout_staged = sm90_utils.make_smem_layout_b(
            self.b_layout,
            self.mma_tiler,
            self.b_dtype,
            self.num_ab_stage,
        )
        self.activation_sf_smem_layout_staged = cute.make_layout(
            (self.token_tile_size, 4, self.num_ab_stage),
            stride=(4, 1, self.token_tile_size * 4),
        )
        self.c_smem_layout_staged = self.epilogue.staged_smem_layout(
            self.num_c_stage,
        )

        # Read epilogue's autonomous decisions.
        self.num_acc_stage = 1

        # Blockwise activation scales share the AB stage/full barrier. Four
        # contiguous FP32 planes keep each TMA row transaction 16-byte aligned.
        atom_thr_size = 1
        self.atom_thr_size = atom_thr_size  # store as Python int for use in @cute.kernel
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        a_copy_size = cute.size_in_bytes(self.a_dtype, a_smem_layout)
        b_copy_size = cute.size_in_bytes(self.b_dtype, b_smem_layout)
        self.num_tma_load_bytes = (
            a_copy_size
            + b_copy_size
            + self._activation_sf_bytes_per_stage()
        ) * atom_thr_size

    def wgmma_warpgroup_init(
        self,
        tiled_mma,
        sA: cute.Tensor,
        sB: cute.Tensor,
        wg_idx,
    ):
        """Initialize WGMMA operand fragments and AB consumer state."""
        warpgroup_thread_layout = cute.make_layout(
            self.wgmma_m_splits,
            stride=32 * self.epilogue_warps_per_warpgroup,
        )
        thr_mma = tiled_mma.get_slice(warpgroup_thread_layout(wg_idx))
        sA_wg = cute.local_tile(
            sA,
            cute.slice_(self.wgmma_tiler, (None, 0, None)),
            (wg_idx, 0, None),
        )
        tCsA = thr_mma.partition_A(sA_wg)
        tCsB = thr_mma.partition_B(sB)
        tCrA = tiled_mma.make_fragment_A(tCsA)
        tCrB = tiled_mma.make_fragment_B(tCsB)
        cC = cute.make_identity_tensor((self.wgmma_tiler[0], self.wgmma_tiler[1]))
        tCgC = thr_mma.partition_C(cC)

        ab_consumer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Consumer, self.num_ab_stage
        )

        return (
            tCrA,
            tCrB,
            tCgC.shape[:3],
            ab_consumer_state,
        )

    def _activation_scale_rmem_layout(self) -> cute.Layout:
        return cute.make_layout(2 * (self.wgmma_tile_n // 8))

    @cute.jit
    def _load_activation_scales_blockwise_fragment(
        self,
        smem_activation_sf: cute.Tensor,
        activation_scales: cute.Tensor,
        stage_idx,
        scale_plane,
        local_warp_idx: int,
        tidx,
    ) -> None:
        lane_mod = (tidx % 32) % 4
        token_group_count = self.wgmma_tile_n // 8
        for token_group in cutlass.range_constexpr(token_group_count):
            token0 = cutlass.Int32(token_group * 8) + lane_mod * cutlass.Int32(2)
            token1 = token0 + cutlass.Int32(1)
            activation_scales[token_group * 2] = Float32(
                smem_activation_sf[token0, scale_plane, stage_idx]
            )
            activation_scales[token_group * 2 + 1] = Float32(
                smem_activation_sf[token1, scale_plane, stage_idx]
            )

    @cute.jit
    def _promote_accum_temp_blockwise_token_pairs(
        self,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        activation_scales: cute.Tensor,
        weight_scale: Float32,
    ) -> None:
        accum_regs_per_m64 = self.wgmma_tile_n // 2  # n // 8 * 4 = n // 2
        token_group_count = self.wgmma_tile_n // 8
        for token_group in cutlass.range_constexpr(token_group_count):
            token0_scale = activation_scales[token_group * 2] * weight_scale
            token1_scale = activation_scales[token_group * 2 + 1] * weight_scale
            for m_sub in cutlass.range_constexpr(2):
                m_sub_base = m_sub * accum_regs_per_m64
                base = m_sub_base + token_group * 4
                accumulators[base + 0] = (
                    accumulators[base + 0] + accum_temp[base + 0] * token0_scale
                )
                accumulators[base + 1] = (
                    accumulators[base + 1] + accum_temp[base + 1] * token1_scale
                )
                accumulators[base + 2] = (
                    accumulators[base + 2] + accum_temp[base + 2] * token0_scale
                )
                accumulators[base + 3] = (
                    accumulators[base + 3] + accum_temp[base + 3] * token1_scale
                )

    @cute.jit
    def _promote_accum_temp_blockwise_fc1(
        self,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        activation_scales: cute.Tensor,
        weight_scale: Float32,
    ) -> None:
        self._promote_accum_temp_blockwise_token_pairs(
            accumulators=accumulators,
            accum_temp=accum_temp,
            activation_scales=activation_scales,
            weight_scale=weight_scale,
        )

    @cute.jit
    def _promote_accum_temp_blockwise_fc2(
        self,
        accumulators: cute.Tensor,
        accum_temp: cute.Tensor,
        activation_scales: cute.Tensor,
        weight_scale: Float32,
    ) -> None:
        self._promote_accum_temp_blockwise_token_pairs(
            accumulators=accumulators,
            accum_temp=accum_temp,
            activation_scales=activation_scales,
            weight_scale=weight_scale,
        )

    @cute.jit
    def __call__(
        self,
        # ── fc1 (Linear1) problem tensors ────────────────────────────────
        activation: cute.Tensor,           # (token_sum_padded, hidden)
        fc1_weight: cute.Tensor,           # (experts, hidden, intermediate_gateup)
        activation_sf: cute.Tensor,         # per-tensor legacy SF or blockwise fp32 activation scale
        fc1_weight_sf: cute.Tensor,         # per-tensor legacy SF or blockwise fp32 fc1 weight scale
        fc1_activation_dequant_scale: cute.Tensor,  # (1,) Float32
        fc1_weight_dequant_scale: cute.Tensor,      # (experts,) Float32
        # ── fc1 workspace consumed as fc2 GEMM-B ─────────────────────────
        fc1_output: cute.Tensor,         # (token_sum_padded, intermediate_downproj)
        fc1_output_sf: cute.Tensor,      # per-tensor legacy SF or blockwise fp32 fc2 activation scale
        # ── fc2 (Linear2) problem tensors ────────────────────────────────
        fc2_weight: cute.Tensor,          # (experts, intermediate_downproj, hidden)
        fc2_weight_sf: cute.Tensor,        # per-tensor legacy SF or blockwise fp32 fc2 weight scale
        fc2_activation_dequant_scale: cute.Tensor,  # (1,) Float32
        fc2_weight_dequant_scale: cute.Tensor,      # (experts,) Float32
        fc2_output: cute.Tensor,         # (token_sum_padded, hidden) BFloat16, hidden stride-1
        # ── topk weights (Path A) ────────────────────────────────────────
        topk_scores: cute.Tensor,     # (token_sum_padded,) Float32
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
        """Launch the fused fc1+fc2 GLU FP8 kernel."""

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

        # ── GEMM-domain transform for fc1 phase (swap-AB) ──
        c1 = cutlass.Int32(1)
        c0 = cutlass.Int32(0)

        # B_gemm (fc1 activations): (tokens_sum, hidden) -> (N=tokens, K=hidden, L=1).
        tokens_sum, hidden = activation.shape
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens_sum, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )

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

        # C_gemm is a user-view output tensor; epilogue owns its store path.
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
        # Phases share dtypes by construction. Scale tensors stay outside the
        # raw WGMMA and are interpreted by the selected scale-mode path.
        # ``self.fc1_output_dtype`` drives the sC SMEM element type and flows
        # into the epilogue ctor as ``fc1_output_dtype``.
        self.a_dtype: Type[cutlass.Numeric] = fc1_weight_gemm.element_type
        self.b_dtype: Type[cutlass.Numeric] = activation_gemm.element_type
        self.fc1_output_dtype: Type[cutlass.Numeric] = fc1_output_gemm.element_type
        self.sf_dtype: Type[cutlass.Numeric] = activation_sf.element_type
        self.a_layout = utils.LayoutEnum.from_tensor(fc1_weight_gemm)
        self.b_layout = utils.LayoutEnum.from_tensor(activation_gemm)
        self.a_major_mode = self.a_layout.sm90_mma_major_mode()
        self.b_major_mode = self.b_layout.sm90_mma_major_mode()
        self.fc1_output_layout = utils.LayoutEnum.from_tensor(fc1_output_gemm)

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        # ── fc1 TMA atoms ──

        # TMA load A1 (= fc1 weights).
        # 1CTA SM90 path uses plain tensor-tile TMA; no cluster multicast.
        a_op = cpasync.CopyBulkTensorTileG2SOp()
        a_smem_layout = cute.slice_(self.a_smem_layout_staged, (None, None, 0))
        tma_atom_fc1_weight, tma_tensor_fc1_weight = cpasync.make_tiled_tma_atom(
            a_op,
            fc1_weight_gemm,
            a_smem_layout,
            cute.slice_(self.mma_tiler, (None, 0, None)),
        )

        # TMA load B1 (= fc1 activations).
        b_op = cpasync.CopyBulkTensorTileG2SOp()
        b_smem_layout = cute.slice_(self.b_smem_layout_staged, (None, None, 0))
        tma_atom_fc1_activation, tma_tensor_fc1_activation = cpasync.make_tiled_tma_atom(
            b_op,
            activation_gemm,
            b_smem_layout,
            cute.slice_(self.mma_tiler, (0, None, None)),
        )

        # TMA store for the FC1 FP8 output.
        # The shared epilogue helper issues each per-WG 64x64 store; commit /
        # drain remains in the epilogue's ``run`` loop body.
        fc1_output_tma_op = cpasync.CopyBulkTensorTileS2GOp()
        tma_atom_fc1_output, tma_tensor_fc1_output = cpasync.make_tiled_tma_atom(
            fc1_output_tma_op,
            fc1_output_gemm,
            self.epilogue.smem_layout_one_stage,
            self.epilogue.epi_tile,
        )

        if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
            # DeepGEMM-style FP8 blockwise scale views.  The host passes these
            # tensors as dense FP32 arrays and the kernel interprets them by
            # phase:
            #   activation_sf     : (tokens, hidden // 128)
            #   fc1_weight_sf     : (experts, intermediate_gateup // 128, hidden // 128)
            #   fc1_output_sf     : (tokens, intermediate_downproj // 64)
            #   fc2_weight_sf     : (experts, hidden // 128, intermediate_downproj // 128)
            activation_sf_gemm = cute.make_tensor(
                activation_sf.iterator,
                cute.make_layout(
                    (activation_sf.shape[0], hidden // Fp8BlockScaleK, 1),
                    stride=(activation_sf.stride[0], activation_sf.stride[1], 0),
                ),
            )
            fc1_weight_sf_gemm = cute.make_tensor(
                fc1_weight_sf.iterator,
                cute.make_layout(
                    (
                        intermediate_gateup // Fp8WeightScaleBlockN,
                        hidden_b // Fp8WeightScaleBlockK,
                        experts,
                    ),
                    stride=(
                        fc1_weight_sf.stride[1],
                        fc1_weight_sf.stride[2],
                        fc1_weight_sf.stride[0],
                    ),
                ),
            )
            fc1_output_sf_gemm = cute.make_tensor(
                fc1_output_sf.iterator,
                cute.make_layout(
                    (
                        fc1_output_sf.shape[0],
                        intermediate_downproj // Fp8Fc2ActivationScaleK,
                        1,
                    ),
                    stride=(fc1_output_sf.stride[0], fc1_output_sf.stride[1], 0),
                ),
            )
            fc2_weight_sf_gemm = cute.make_tensor(
                fc2_weight_sf.iterator,
                cute.make_layout(
                    (
                        hidden_b2 // Fp8WeightScaleBlockN,
                        intermediate_downproj_b2 // Fp8WeightScaleBlockK,
                        experts2,
                    ),
                    stride=(
                        fc2_weight_sf.stride[1],
                        fc2_weight_sf.stride[2],
                        fc2_weight_sf.stride[0],
                    ),
                ),
            )
        else:
            # fc1 SFC GMEM tensor (= fc1_output_sf user view).  No TMA atom; it is
            # per-thread STG.
            tokens_sum_padded = fc1_output_sf.shape[0]
            fc1_output_sf_gemm = cute.make_tensor(
                fc1_output_sf.iterator,
                blockscaled_utils.tile_atom_to_shape_SF(
                    (tokens_sum_padded, intermediate_downproj, 1),
                    self.sf_vec_size,
                ),
            )
            activation_sf_gemm = activation_sf
            fc1_weight_sf_gemm = fc1_weight_sf
            fc2_weight_sf_gemm = fc2_weight_sf

        if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
            activation_sf_smem_layout = cute.slice_(
                self.activation_sf_smem_layout_staged,
                (None, None, 0),
            )
            activation_sf_tiler = (self.token_tile_size, 4)
            activation_sf_op = cpasync.CopyBulkTensorTileG2SOp()
            (
                tma_atom_fc1_activation_sf,
                tma_tensor_fc1_activation_sf,
            ) = cpasync.make_tiled_tma_atom(
                activation_sf_op,
                activation_sf_gemm,
                activation_sf_smem_layout,
                activation_sf_tiler,
            )
            (
                tma_atom_fc2_activation_sf,
                tma_tensor_fc2_activation_sf,
            ) = cpasync.make_tiled_tma_atom(
                activation_sf_op,
                fc1_output_sf_gemm,
                activation_sf_smem_layout,
                activation_sf_tiler,
            )
        else:
            # Compile-time placeholders; the per-tensor kernel removes all SF
            # TMA and SMEM consumers.
            tma_atom_fc1_activation_sf = tma_atom_fc1_activation
            tma_tensor_fc1_activation_sf = tma_tensor_fc1_activation
            tma_atom_fc2_activation_sf = tma_atom_fc1_activation
            tma_tensor_fc2_activation_sf = tma_tensor_fc1_activation

        # ── fc2 TMA atoms: fc2_weight → A-side, fc1_output → B-side ──

        tma_atom_fc2_weight, tma_tensor_fc2_weight = (
            cpasync.make_tiled_tma_atom(
                a_op,
                fc2_weight_gemm,
                a_smem_layout,
                cute.slice_(self.mma_tiler, (None, 0, None)),
            )
        )
        tma_atom_fc2_activation, tma_tensor_fc2_activation = (
            cpasync.make_tiled_tma_atom(
                b_op,
                fc1_output_gemm,
                b_smem_layout,
                cute.slice_(self.mma_tiler, (0, None, None)),
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
            is_swap_ab=True,
            expert_token_prefix_sum=offs,
            expert_token_sizes=expert_token_sizes,
        )
        grid = sched_params.get_grid_shape(max_active_clusters)

        self.kernel(
            tiled_mma,
            # fc1 TMA atoms / tensors (swap-AB: A=weights, B=activations)
            tma_atom_fc1_weight,
            tma_tensor_fc1_weight,
            tma_atom_fc1_activation,
            tma_tensor_fc1_activation,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            # fc2 TMA atoms / tensors (fc2_weight→A, fc1_output→B)
            tma_atom_fc2_weight,
            tma_tensor_fc2_weight,
            tma_atom_fc2_activation,
            tma_tensor_fc2_activation,
            tma_atom_fc1_activation_sf,
            tma_tensor_fc1_activation_sf,
            tma_atom_fc2_activation_sf,
            tma_tensor_fc2_activation_sf,
            # GEMM-domain tensors (fc1)
            activation_gemm,
            fc1_weight_gemm,
            activation_sf_gemm,
            fc1_weight_sf_gemm,
            fc1_activation_dequant_scale,
            fc1_weight_dequant_scale,
            fc1_output_gemm,
            fc1_output_sf_gemm,
            # GEMM-domain tensors (fc2)
            fc2_weight_gemm,
            fc2_weight_sf_gemm,
            fc2_activation_dequant_scale,
            fc2_weight_dequant_scale,
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
            self.activation_sf_smem_layout_staged,
            self.c_smem_layout_staged,
            token_comm_args,
        ).launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
        )


    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        # fc1 TMA atoms / tensors
        tma_atom_fc1_weight: cute.CopyAtom,
        tma_tensor_fc1_weight: cute.Tensor,
        tma_atom_fc1_activation: cute.CopyAtom,
        tma_tensor_fc1_activation: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # fc2 TMA atoms / tensors (fc2_weight→A, fc1_output→B)
        tma_atom_fc2_weight: cute.CopyAtom,
        tma_tensor_fc2_weight: cute.Tensor,
        tma_atom_fc2_activation: cute.CopyAtom,
        tma_tensor_fc2_activation: cute.Tensor,
        tma_atom_fc1_activation_sf: cute.CopyAtom,
        tma_tensor_fc1_activation_sf: cute.Tensor,
        tma_atom_fc2_activation_sf: cute.CopyAtom,
        tma_tensor_fc2_activation_sf: cute.Tensor,
        # GEMM-domain tensors (fc1)
        activation_gemm: cute.Tensor,
        fc1_weight_gemm: cute.Tensor,
        activation_sf_gemm: cute.Tensor,
        fc1_weight_sf_gemm: cute.Tensor,
        fc1_activation_dequant_scale: cute.Tensor,
        fc1_weight_dequant_scale: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        fc1_output_sf_gemm: cute.Tensor,
        # GEMM-domain tensors (fc2)
        fc2_weight_gemm: cute.Tensor,
        fc2_weight_sf_gemm: cute.Tensor,
        fc2_activation_dequant_scale: cute.Tensor,
        fc2_weight_dequant_scale: cute.Tensor,
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
        activation_sf_smem_layout_staged: cute.Layout,
        c_smem_layout_staged: Union[cute.Layout, cute.ComposedLayout],
        token_comm_args=None,
    ):
        """Device kernel for fused fc1+fc2 swap-AB FP8 grouped GEMM.

        The lean path uses one/two WGMMA+epilogue warpgroups for M=128/256,
        followed by TMA-A, TMA-B, scheduler, and one empty legacy-MMA warp.
        It has no drain-aux warps or expert-wise TMA descriptor rewriting
        because every descriptor is tile-invariant under swap-AB.

        Epilogue is fully owned by ``self.epilogue.run(...)`` -- the four epi
        warps make a single call that drives the entire 2-phase task-tile
        loop (acc consumer state, subtile dispatch, TMA commit/drain, and
        the piggyback ``red.release.gpu.add.s32`` to ``fc1_done_counter``).
        """
        a_smem_layout = cute.slice_(a_smem_layout_staged, (None, None, 0))
        b_smem_layout = cute.slice_(b_smem_layout_staged, (None, None, 0))

        # fc2 waits for all fc1 intermediate N-tiles in the same token block.
        # 1CTA path: each N-tile contributes exactly one fc1-done increment.
        ext_fc2_spin_threshold = (
            fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[0] - 1
        ) // self.cta_tile_shape_mnk[0]

        ext = SwapABSwigluFp4Fc12SchedExtension(
            sf_vec_size=self.sf_vec_size,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
        )

        warp_idx = cute.arch.warp_idx()
        warp_idx = cute.arch.make_warp_uniform(warp_idx)

        tidx, _, _ = cute.arch.thread_idx()
        epilogue_group_idx = warp_idx // cutlass.Int32(
            self.epilogue_warps_per_warpgroup
        )
        local_warp_idx = warp_idx - epilogue_group_idx * cutlass.Int32(
            self.epilogue_warps_per_warpgroup
        )

        # SharedStorage.
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            ab_full_mbar_ptr: cute.struct.MemRange[cutlass.Int64, self.num_ab_stage * 2]
            weight_sf_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.num_ab_stage * 2
            ]
            sched_storage: SchedStorage

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

        ab_pipeline_producer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, 2
        )
        # PipelineTmaAsync empty barriers are released by one signalling lane
        # per WGMMA consumer warp (per multicast target), matching the Hopper
        # dense GEMM examples.  Do not use the full 128-thread warpgroup here.
        mcast_size = self.num_mcast_ctas_a + self.num_mcast_ctas_b - 1
        num_wgmma_consumer_threads = mcast_size * len(self.epilogue_warp_id)
        ab_pipeline_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_wgmma_consumer_threads
        )
        ab_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.ab_full_mbar_ptr.data_ptr(),
            num_stages=self.num_ab_stage,
            producer_group=ab_pipeline_producer_group,
            consumer_group=ab_pipeline_consumer_group,
            tx_count=self.num_tma_load_bytes // 2,
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        ab_producer = ab_pipeline.make_producer()
        if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
            weight_sf_pipeline = pipeline.PipelineCpAsync.create(
                barrier_storage=storage.weight_sf_mbar_ptr.data_ptr(),
                num_stages=self.num_ab_stage,
                producer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32
                ),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * len(self.epilogue_warp_id)
                ),
                defer_sync=True,
            )
            weight_sf_producer = weight_sf_pipeline.make_producer()
        else:
            weight_sf_pipeline = ab_pipeline
            weight_sf_producer = ab_producer

        # Sched
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_a_warp_id,
                self.tma_b_warp_id,
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
        if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
            weight_sf_smem_layout_staged = cute.make_layout(
                (self.wgmma_warpgroup_count, self.num_ab_stage),
                stride=(1, self.wgmma_warpgroup_count),
            )
            sActivationSf = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=activation_sf_smem_layout_staged,
                byte_alignment=128,
            )
            sWeightSf = smem.allocate_tensor(
                element_type=cutlass.Float32,
                layout=weight_sf_smem_layout_staged,
                byte_alignment=16,
            )
        else:
            sActivationSf = sB
            sWeightSf = sB

        # Cluster wait after pipeline init and before producer/consumer use.
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)

        if cutlass.const_expr(self.enable_token_comm):
            if warp_idx < cutlass.Int32(len(self.epilogue_warp_id)):
                cute.arch.setmaxregister_increase(self.epi_reg_cnt)
            elif warp_idx == self.tma_a_warp_id:
                cute.arch.setmaxregister_decrease(self.tma_a_reg_cnt)
            elif warp_idx == self.tma_b_warp_id:
                cute.arch.setmaxregister_decrease(self.tma_b_reg_cnt)
            elif warp_idx == self.sched_warp_id:
                cute.arch.setmaxregister_decrease(self.sched_reg_cnt)
            elif warp_idx == self.empty_warp_id:
                cute.arch.setmaxregister_decrease(self.empty_reg_cnt)
            else:
                if cutlass.const_expr(self.token_back_standalone):
                    if warp_idx < self.token_back_warp_id[0]:
                        cute.arch.setmaxregister_decrease(self.dispatch_reg_cnt)
                    else:
                        cute.arch.setmaxregister_decrease(self.token_back_reg_cnt)
                else:
                    cute.arch.setmaxregister_decrease(self.dispatch_reg_cnt)

        mma_tiler_k = self.mma_tiler[2]
        # ``fc1_weight_gemm.shape[1]`` / ``fc2_weight_gemm.shape[1]``
        # both resolve to ``hidden`` / ``intermediate_downproj``.  Under
        # ``static_expert_shape`` they are codegen-time Python ints
        # (rewritten on ``fc1_weight`` / ``fc2_weight`` at ``__call__``
        # entry); otherwise they are runtime Int32 from tensor metadata.
        # The arithmetic below folds to an immediate in the static path.
        k_tile_cnt_fc1 = (fc1_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        k_tile_cnt_fc2 = (fc2_weight_gemm.shape[1] + mma_tiler_k - 1) // mma_tiler_k
        # Each raw gate/up CTA-M tile publishes one completion for the same
        # compile-time token-N block.
        fc2_spin_threshold = (
            (fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[0] - 1)
            // self.cta_tile_shape_mnk[0]
        )

        # ════════════════════════════════════════════════════════════════════
        # Scheduler warp (after the two TMA warps)
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
        # TMA load warps (immediately after the epilogue groups)
        # ════════════════════════════════════════════════════════════════════
        #
        # TMA-A and TMA-B feed the shared operand pipeline. In blockwise mode,
        # weight-side TMA-A stages one K128 weight scale per WGMMA group through
        # cp.async while TMA-B also loads the activation scales.

        # ── TMA-A warp ──────────────────────────────────────────────────────
        if warp_idx == self.tma_a_warp_id:
            # ``warp_idx`` is warp-uniform, so all 32 lanes execute every IKET
            # push/pop in this producer branch.
            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )
                if is_phase_linear1:
                    # Covers descriptor lookup and weight TMA issue. Blockwise
                    # mode additionally stages weight scales with cp.async.
                    iket.range_push(self._iket_fc1_weight_load_range)
                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a", tma_tensor_fc1_weight, work_tile_info,
                    )
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        output_scale_block_base = (
                            work_tile_info.tile_m_idx
                            * cutlass.Int32(self.wgmma_warpgroup_count)
                        )
                        ab_producer, weight_sf_producer = (
                            self._tma_load_a_with_weight_sf_task_tile(
                                tma_atom=tma_atom_fc1_weight,
                                real_a=real_a,
                                desc_ptr_a=desc_ptr_a,
                                sA=sA,
                                weight_sf_gemm=fc1_weight_sf_gemm,
                                smem_weight_sf=sWeightSf,
                                ab_producer=ab_producer,
                                weight_sf_producer=weight_sf_producer,
                                work_tile_info=work_tile_info,
                                tile_m_idx=work_tile_info.tile_m_idx,
                                output_scale_block_base=output_scale_block_base,
                                k_tile_cnt=k_tile_cnt_fc1,
                                tidx=tidx,
                            )
                        )
                    else:
                        ab_producer = self._tma_load_a_task_tile(
                            tma_atom_fc1_weight,
                            real_a,
                            desc_ptr_a,
                            sA,
                            ab_producer,
                            work_tile_info.tile_m_idx,
                            k_tile_cnt_fc1,
                        )
                else:
                    # Covers descriptor lookup and weight TMA issue. Blockwise
                    # mode additionally stages weight scales with cp.async.
                    iket.range_push(self._iket_fc2_weight_load_range)
                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_ptr_a = ext.get_gmem_tensor(
                        "a", tma_tensor_fc2_weight, work_tile_info,
                    )
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        output_scale_block_base = (
                            work_tile_info.tile_m_idx
                            * cutlass.Int32(self.wgmma_warpgroup_count)
                        )
                        ab_producer, weight_sf_producer = (
                            self._tma_load_a_with_weight_sf_task_tile(
                                tma_atom=tma_atom_fc2_weight,
                                real_a=real_a,
                                desc_ptr_a=desc_ptr_a,
                                sA=sA,
                                weight_sf_gemm=fc2_weight_sf_gemm,
                                smem_weight_sf=sWeightSf,
                                ab_producer=ab_producer,
                                weight_sf_producer=weight_sf_producer,
                                work_tile_info=work_tile_info,
                                tile_m_idx=work_tile_info.tile_m_idx,
                                output_scale_block_base=output_scale_block_base,
                                k_tile_cnt=k_tile_cnt_fc2,
                                tidx=tidx,
                            )
                        )
                    else:
                        ab_producer = self._tma_load_a_task_tile(
                            tma_atom_fc2_weight,
                            real_a,
                            desc_ptr_a,
                            sA,
                            ab_producer,
                            work_tile_info.tile_m_idx,
                            k_tile_cnt_fc2,
                        )

                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                weight_sf_producer.tail()
            ab_producer.tail()

        # ── TMA-B warp ──────────────────────────────────────────────────────
        if warp_idx == self.tma_b_warp_id:
            # ``warp_idx`` is warp-uniform, so all 32 lanes execute every IKET
            # push/pop in this producer branch.
            work_tile_info = sched_consumer.consume_work()

            while work_tile_info.is_valid_tile:
                is_phase_linear1 = (
                    work_tile_info.phase == cutlass.Int32(BlockPhase.Linear1)
                )

                if is_phase_linear1:
                    # Covers optional Mega dispatch readiness, descriptor
                    # lookup, and activation TMA issue. Blockwise mode also
                    # issues activation-scale TMA loads.
                    iket.range_push(self._iket_fc1_activation_load_range)
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args, work_tile_info,
                    )
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b", tma_tensor_fc1_activation, work_tile_info,
                    )
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        if cutlass.const_expr(self.enable_token_comm):
                            sf_token_base = (
                                work_tile_info.cumulative_sf_physical_row
                                + work_tile_info.tile_n_idx
                                * cutlass.Int32(self.token_tile_size)
                            )
                        else:
                            sf_token_base = (
                                work_tile_info.cumulative_data_physical_row
                                + work_tile_info.tile_n_idx
                                * cutlass.Int32(self.token_tile_size)
                            )
                        ab_producer = (
                            self._tma_load_b_with_activation_sf_task_tile(
                                tma_atom_fc1_activation,
                                real_b,
                                desc_ptr_b,
                                sB,
                                tma_atom_fc1_activation_sf,
                                tma_tensor_fc1_activation_sf,
                                sActivationSf,
                                ab_producer,
                                work_tile_info.tile_n_idx,
                                sf_token_base
                                // cutlass.Int32(self.token_tile_size),
                                k_tile_cnt_fc1,
                                k_tiles_per_scale_group=4,
                            )
                        )
                    else:
                        ab_producer = self._tma_load_b_task_tile(
                            tma_atom_fc1_activation,
                            real_b,
                            desc_ptr_b,
                            sB,
                            ab_producer,
                            work_tile_info.tile_n_idx,
                            k_tile_cnt_fc1,
                        )
                else:
                    # Covers the FC1-done wait/fences, descriptor lookup, and
                    # FC2-activation TMA issue. Blockwise mode also issues
                    # activation-scale TMA loads.
                    iket.range_push(self._iket_fc2_activation_load_range)
                    counter_slot = (
                        work_tile_info.cumulative_token_block_count
                        + work_tile_info.tile_n_idx
                    )
                    counter_ptr = fc1_done_counter.iterator + counter_slot
                    # Nested inside the swap-AB FC2 activation-load range:
                    # only the FC1 completion-counter spin. The enclosing
                    # range continues through acquire/fence and TMA issue.
                    iket.range_push("swapab_fc2_fc1_done_spin")
                    spin_wait(
                        counter_ptr,
                        lambda v: v >= fc2_spin_threshold,
                        fail_sleep_cycles=20,
                    )
                    iket.range_pop()
                    cute.arch.load(
                        counter_ptr, counter_ptr.dtype, sem="acquire", scope="gpu"
                    )
                    cute.arch.fence_proxy("async")
                    cute.arch.fence_proxy("async.global")
                    real_b, desc_ptr_b = ext.get_gmem_tensor(
                        "b", tma_tensor_fc2_activation, work_tile_info,
                    )
                    if cutlass.const_expr(self.fp8_scale_mode == "blockwise"):
                        sf_token_base = (
                            work_tile_info.cumulative_data_physical_row
                            + work_tile_info.tile_n_idx
                            * cutlass.Int32(self.token_tile_size)
                        )
                        ab_producer = (
                            self._tma_load_b_with_activation_sf_task_tile(
                                tma_atom_fc2_activation,
                                real_b,
                                desc_ptr_b,
                                sB,
                                tma_atom_fc2_activation_sf,
                                tma_tensor_fc2_activation_sf,
                                sActivationSf,
                                ab_producer,
                                work_tile_info.tile_n_idx,
                                sf_token_base
                                // cutlass.Int32(self.token_tile_size),
                                k_tile_cnt_fc2,
                                k_tiles_per_scale_group=2,
                            )
                        )
                    else:
                        ab_producer = self._tma_load_b_task_tile(
                            tma_atom_fc2_activation,
                            real_b,
                            desc_ptr_b,
                            sB,
                            ab_producer,
                            work_tile_info.tile_n_idx,
                            k_tile_cnt_fc2,
                        )
                iket.range_pop()
                work_tile_info = sched_consumer.consume_work()

            ab_producer.tail()

        # The old independent MMA-only role marker remains as one empty warp
        # after the scheduler. M=256 has a second M-split WGMMA+epilogue WG;
        # M=128 does not launch it.

        # ── sC SMEM (fc1 output staging; fc2 doesn't use it) ──
        sC = smem.allocate_tensor(
            element_type=self.fc1_output_dtype,
            layout=c_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=c_smem_layout_staged.inner,
        )
        sFc1Amax = smem.allocate_tensor(
            element_type=cutlass.Float32,
            layout=self.epilogue.fc1_amax_smem_layout,
            byte_alignment=16,
        )

        # ════════════════════════════════════════════════════════════════════
        # Epilogue warps (one or two four-warp groups)
        # ════════════════════════════════════════════════════════════════════
        #
        # Epilogue owns the full task-tile loop. Every WG handles one M=128
        # slice and all active groups consume the same AB pipeline stage.
        if warp_idx < cutlass.Int32(len(self.epilogue_warp_id)):
            if cutlass.const_expr(self.wgmma_m_splits == 1):
                n_half = 0
            else:
                n_half = epilogue_group_idx

            (
                tCrA,
                tCrB,
                acc_shape,
                ab_consumer_state,
            ) = self.wgmma_warpgroup_init(
                tiled_mma=tiled_mma,
                sA=sA,
                sB=sB,
                wg_idx=n_half,
            )
            accumulators = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            if cutlass.const_expr(
                self.fp8_scale_mode == "blockwise"
                or self.fp8_accum_mode == "2xacc"
            ):
                accum_temp = cute.make_rmem_tensor(acc_shape, self.acc_dtype)
            else:
                accum_temp = accumulators
            # Emit one representative, warp-uniform WGMMA task range per
            # physical warpgroup. Every lane in local warp 0 participates.
            _iket_active = local_warp_idx == cutlass.Int32(0)

            # Build common kwargs shared by both epilogue flavours.
            _run_kwargs = dict(
                sched_consumer=sched_consumer,
                sched_ext=ext,
                smem_fc1_output_buffer=sC,
                smem_fc1_amax=sFc1Amax,
                tma_atom_fc1_output=tma_atom_fc1_output,
                gmem_fc1_output=tma_tensor_fc1_output,
                gmem_fc1_output_sf=fc1_output_sf_gemm,
                smem_activation_sf=sActivationSf,
                smem_weight_sf=sWeightSf,
                gmem_topk_scores=topk_scores,
                gmem_fc2_output=fc2_output_gemm,
                gmem_fc1_done_counter=fc1_done_counter,
                local_warp_idx=local_warp_idx,
                tidx=tidx,
                fc1_activation_dequant_scale=fc1_activation_dequant_scale,
                fc1_weight_dequant_scale=fc1_weight_dequant_scale,
                fc2_activation_dequant_scale=fc2_activation_dequant_scale,
                fc2_weight_dequant_scale=fc2_weight_dequant_scale,
                norm_const=cutlass.Float32(1.0),
                tiled_mma=tiled_mma,
                tCrA=tCrA,
                tCrB=tCrB,
                accumulators=accumulators,
                accum_temp=accum_temp,
                run_wgmma_task_tile=self.run_wgmma_task_tile,
                ab_pipeline=ab_pipeline,
                weight_sf_pipeline=weight_sf_pipeline,
                ab_consumer_state=ab_consumer_state,
                k_tile_cnt_fc1=k_tile_cnt_fc1,
                k_tile_cnt_fc2=k_tile_cnt_fc2,
                _iket_active=_iket_active,
                n_half=n_half,
            )

            # MegaMoE: pass token_comm_args only when it is a real bundle (not
            # None).  Passing Python None explicitly to @cute.jit methods
            # triggers a CuteDSL codegen issue; const_expr dispatch avoids any
            # None-as-JIT-argument path.
            if cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(
                    **_run_kwargs, token_comm_args=token_comm_args
                )
            else:
                self.epilogue.run(**_run_kwargs)

            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # ════════════════════════════════════════════════════════════════════
        # Dispatch warps hook (four warps after the empty warp; MegaMoE-only)
        # ════════════════════════════════════════════════════════════════════
        #
        # ``enable_token_comm=False`` removes this role at compile time.
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
