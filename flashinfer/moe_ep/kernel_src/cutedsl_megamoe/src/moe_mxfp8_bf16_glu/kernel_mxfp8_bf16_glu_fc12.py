# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Lean fused FC1+FC2 kernel for MXFP8-weight/BF16-activation MoE.

This module is deliberately separate from the native block-scaled MXFP8 and
NVFP4 kernels.  Both FC1 and FC2 use the swap-AB geometry (weight is GEMM A,
tokens are GEMM B), but the tensor core instruction is dense BF16:

    FP8 weight + E8M0/K32 --CUDA-core transform--> BF16 TMEM/SMEM A
    BF16 activation/handoff ----------------------> BF16 SMEM B
    dense tcgen05 BF16 x BF16 --------------------> FP32 accumulator

The first implementation target is the lean, communication-free shape
``(256, 128|256, 128)``, two-CTA MMA, cluster ``(2, 1)``.  Four transform warps
occupy warp ids 8-11.  The scheduler and phase-coordinate conventions follow
``moe_nvfp4_swapab.kernel_fc12``.

The accumulator consumer is the swap-AB BF16 epilogue: FC1 performs SwiGLU and
a direct BF16 hand-off store; FC2 restores token-major order and stores BF16.
"""

from __future__ import annotations

from typing import Literal, Optional, Tuple, Type

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover - older/newer CuTeDSL API location
    from src.iket_compat import iket
from cutlass.cute.typing import AddressSpace
import cutlass.pipeline as pipeline
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
import cutlass.utils.blockscaled_layout as blockscaled_utils
import cutlass.utils.mixed_input_helpers as mixed_input_utils
from cutlass.utils.mixed_input_helpers import TransformMode
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass._mlir.dialects import vector as mlir_vector

from common.host_utils import get_cutedsl_target_arch
from moe_mxfp8_bf16_glu.epilogue_mxfp8_bf16 import (
    Fc1GateUpInterleave,
    MixedGluBf16Epilogue,
)
from moe_nvfp4_swapab.custom_ext import (
    SwapABSwigluFp4Fc12SchedExtension,
    SwapABSwigluFp4Fc12WorkTileInfo,
)
from moe_nvfp4_swapab.epilogue_refactor import NvFp4OptinalEpiArgs
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import (
    BlockPhase,
    MoEFusedFc12SchedulerParams,
)
from moe_nvfp4_swapab.moe_utils import spin_wait


def _scale_partition_mxfp8(
    src_copy_a: cute.TiledCopy,
    tCsS: cute.Tensor,
    transform_local_tidx: cutlass.Int32,
) -> tuple[cute.TiledCopy, cute.Tensor, cute.Tensor, cute.Tensor]:
    """Partition E8M0 scale fragments while preserving their source dtype.

    The generic mixed-input helper assumes the scale already has the MMA
    dtype. MXFP8 stores E8M0 bytes, so transform warps load E8M0 into registers
    and explicitly convert it before multiplication.
    """

    smem_thr_copy_s = src_copy_a.get_slice(transform_local_tidx)
    tSsS_trans = smem_thr_copy_s.partition_S(tCsS)
    per_stage_layout = tSsS_trans[(None, None, None, None, 0)].layout
    tSrS_copy = cute.make_rmem_tensor(
        cute.filter_zeros(per_stage_layout).shape,
        tCsS.element_type,
    )
    tSrS = cute.make_tensor(
        tSrS_copy.iterator,
        cute.make_layout(
            per_stage_layout.shape,
            stride=tSrS_copy.layout.stride,
        ),
    )
    return smem_thr_copy_s, tSsS_trans, tSrS_copy, tSrS


class _MixedBf16NoPeekSchedExtension(
    SwapABSwigluFp4Fc12SchedExtension
):
    """Correctness-first scheduler extension without the optional FC2 peek.

    The inherited non-blocking counter peek currently trips a native CuTeDSL
    trace bug when combined with the mixed transform pipelines.  Rebuilding
    the work tile with the original ``phase_and_peek`` leaves ``peek_ready``
    clear while avoiding an object alias across the scheduler's JIT boundary.
    The FC2 TMA-B warp therefore always executes the existing blocking
    ``spin_wait`` before loading the BF16 hand-off.  This changes only latency
    hiding, not the release/acquire protocol or numerical behavior.
    """

    @cute.jit
    def enrich_work_tile_info(
        self,
        base_work: SwapABSwigluFp4Fc12WorkTileInfo,
    ) -> SwapABSwigluFp4Fc12WorkTileInfo:
        return SwapABSwigluFp4Fc12WorkTileInfo(
            expert_idx=base_work.expert_idx,
            tile_m_idx=base_work.tile_m_idx,
            tile_n_idx=base_work.tile_n_idx,
            cumulative_data_physical_row=(
                base_work.cumulative_data_physical_row
            ),
            cumulative_sf_physical_row=(
                base_work.cumulative_sf_physical_row
            ),
            cumulative_token_block_count=(
                base_work.cumulative_token_block_count
            ),
            valid_tokens_in_cta_tile=base_work.valid_tokens_in_cta_tile,
            phase_and_peek=base_work.phase_and_peek,
        )


class Sm100SwapABMxfp8Bf16Fc12Kernel:
    """Lean swap-AB MXFP8-weight/BF16 FC12 kernel for SM100.

    N128 uses double-buffered accumulators and TMEM transformed-A. N256 either
    stores transformed-A in SMEM or uses the K128 accumulator-overlap TMEM
    implementation. The latter splits each K128 work tile into two internal
    K64 transform/MMA phases. All offsets remain compile-time constants and
    unsupported geometries fail before tracing.
    """

    ScaleGranularityK = 32
    GateUpInterleave = Fc1GateUpInterleave
    _SmemMiscBudget = 2048
    _TmemColsTotal = 512

    _SupportedImplementationConfigs = {
        ((256, 128, 128), "tmem", False, 128),
        ((256, 256, 128), "smem", False, 128),
        ((256, 256, 128), "tmem", True, 64),
    }

    @classmethod
    def _derive_pipeline_stages(
        cls,
        mma_tiler_mnk: Tuple[int, int, int],
        transform_buffer: str,
        accumulator_overlap: bool,
        transform_k_tile: int,
    ) -> dict[str, int]:
        """Validate an implementation tuple and derive all pipeline depths."""

        config = (
            mma_tiler_mnk,
            transform_buffer,
            accumulator_overlap,
            transform_k_tile,
        )
        if config not in cls._SupportedImplementationConfigs:
            supported = sorted(
                cls._SupportedImplementationConfigs,
                key=repr,
            )
            raise ValueError(
                "unsupported mixed implementation config "
                f"{config!r}; supported configs are {supported!r}."
            )

        n = mma_tiler_mnk[1]
        overlapped_cols = 64 if accumulator_overlap else 0
        acc_cols = 2 * n - overlapped_cols
        if transform_buffer == "tmem":
            transform_cols_per_stage = (
                (transform_k_tile // 2 + 3) // 4
            ) * 4
            common_depth = (
                cls._TmemColsTotal - acc_cols
            ) // transform_cols_per_stage
        else:
            # The retained SMEM path is double buffered. More stages exceed
            # its shared-memory budget for the supported N256 geometry.
            common_depth = 2
        if common_depth < 2:
            raise ValueError(
                "mixed implementation leaves room for fewer than two "
                f"pipeline stages: config={config!r}."
            )
        return {
            "a": common_depth,
            "scale": common_depth,
            "transformed_a": common_depth,
            "b": 2,
            "acc": 2,
        }

    @classmethod
    def get_tmem_col_budget(
        cls,
        mma_tiler_mnk: Tuple[int, int, int],
        transform_buffer: str,
        accumulator_overlap: bool,
        transform_k_tile: int,
    ) -> dict[str, int]:
        """Return the compile-time TMEM column plan for supported geometries.

        Dense FP32 C consumes one column per MMA-N value.  Transformed BF16 A
        packs two K values into each 32-bit TMEM column.  This pure helper is
        intentionally usable by CPU-only contract tests and host diagnostics.
        The traced layout calculation below checks the same values against
        CuTeDSL's actual fragment layouts.
        """
        stages = cls._derive_pipeline_stages(
            mma_tiler_mnk,
            transform_buffer,
            accumulator_overlap,
            transform_k_tile,
        )
        _m, n, _public_k = mma_tiler_mnk
        acc_stages = 2
        acc_cols_per_stage = n
        overlapped_acc_cols = 64 if accumulator_overlap else 0
        transformed_a_cols_per_stage = (
            (transform_k_tile // 2 + 3) // 4
        ) * 4
        acc_stage_stride = acc_cols_per_stage - overlapped_acc_cols
        acc_cols = acc_cols_per_stage + (acc_stages - 1) * acc_stage_stride
        transformed_a_cols = (
            stages["transformed_a"] * transformed_a_cols_per_stage
            if transform_buffer == "tmem"
            else 0
        )
        return {
            "acc_stages": acc_stages,
            "acc_cols_per_stage": acc_cols_per_stage,
            "acc_stage_stride": acc_stage_stride,
            "overlapped_acc_cols": overlapped_acc_cols,
            "acc_cols": acc_cols,
            "transformed_a_cols_per_stage": transformed_a_cols_per_stage,
            "transformed_a_cols": transformed_a_cols,
            "required_cols": acc_cols + transformed_a_cols,
        }

    def __init__(
        self,
        mma_tiler_mnk: Tuple[int, int, int] = (256, 128, 128),
        cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1),
        use_2cta_instrs: bool = True,
        group_hint: int = 1,
        token_padding_block: int = 64,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: int = 2,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        gate_up_clamp: Optional[float] = None,
        apply_topk_in_fc1: bool = False,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        transform_buffer: Literal["smem", "tmem"] = "tmem",
        accumulator_overlap: bool = False,
        transform_k_tile: Literal[64, 128] = 128,
    ) -> None:
        supported_mma_tilers = (
            (256, 128, 128),
            (256, 256, 128),
        )
        if mma_tiler_mnk not in supported_mma_tilers:
            raise ValueError(
                "mixed FC12 requires mma_tiler_mnk in "
                f"{supported_mma_tilers}; "
                f"got {mma_tiler_mnk}."
            )
        pipeline_stages = self._derive_pipeline_stages(
            mma_tiler_mnk,
            transform_buffer,
            accumulator_overlap,
            transform_k_tile,
        )
        if cluster_shape_mnk != (2, 1, 1):
            raise ValueError(
                "mixed FC12 requires cluster_shape_mnk=(2, 1, 1); "
                f"got {cluster_shape_mnk}."
            )
        if not use_2cta_instrs:
            raise ValueError("mixed FC12 requires two-CTA tcgen05 MMA.")
        if not force_static_sched:
            raise NotImplementedError(
                "mixed FC12 only supports the lean force_static_sched path."
            )
        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                "load_balance_mode must be 'static' or 'atomic_counter'."
            )
        if token_padding_block <= 0:
            raise ValueError("token_padding_block must be positive.")
        if clc_bundle_size is not None:
            raise ValueError(
                "clc_bundle_size is only meaningful for the dynamic CLC "
                "scheduler; mixed FC12 requires force_static_sched=True."
            )
        self.mma_tiler_mnk = mma_tiler_mnk
        self.transform_buffer = transform_buffer
        self.overlapping_accum = accumulator_overlap
        self.transform_k_tile = transform_k_tile
        # mma_tiler_mnk describes the public/raw tile. transform_k_tile
        # explicitly describes the internal transform/MMA subdivision.
        self.mma_tiler = (
            mma_tiler_mnk[0],
            mma_tiler_mnk[1],
            transform_k_tile,
        )
        self.tma_k_tile = mma_tiler_mnk[2]
        self.tma_k_reuse = self.tma_k_tile // self.mma_tiler[2]
        self.tma_tiler = (
            mma_tiler_mnk[0],
            mma_tiler_mnk[1],
            self.tma_k_tile,
        )
        self.scale_tma_tile_k = mma_tiler_mnk[2]
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self.use_2cta_instrs = use_2cta_instrs
        self.cta_group = tcgen05.CtaGroup.TWO
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.load_balance_mode = load_balance_mode
        self.static_expert_shape = static_expert_shape
        self.force_static_sched = force_static_sched
        self.clc_bundle_size = clc_bundle_size
        self.num_sched_stages = num_sched_stages
        self.acc_dtype = acc_dtype
        self.epi_flag_batch = epi_flag_batch
        self.gate_up_clamp = gate_up_clamp
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.transformed_a_storage = transform_buffer
        self.transform_a_source = (
            tcgen05.OperandSource.SMEM
            if transform_buffer == "smem"
            else tcgen05.OperandSource.TMEM
        )
        self.arch = get_cutedsl_target_arch()
        self.occupancy = 1

        # Lean topology.  A MegaMoE subclass sets ``enable_token_comm=True``
        # after ``super().__init__``; ``_setup_warp_topology`` then inserts
        # dispatch at 8-11 and moves transform to 12-15.
        self.enable_token_comm: bool = False
        self.epilogue_warp_id = (0, 1, 2, 3)
        self.mma_warp_id = 4
        self.tma_a_warp_id = 5
        self.tma_b_warp_id = 6
        self.sched_warp_id = 7
        self.dispatch_warp_id: Optional[Tuple[int, int, int, int]] = None
        self.transform_warp_id = (8, 9, 10, 11)
        self.threads_per_cta = 12 * 32

        self.epilog_sync_barrier = pipeline.NamedBarrier(
            1, 32 * len(self.epilogue_warp_id)
        )
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(
            2, self.threads_per_cta
        )
        self.transform_sync_barrier = pipeline.NamedBarrier(
            3, 32 * len(self.transform_warp_id)
        )

        # Derive concrete pipeline depths from the validated implementation
        # tuple and its SMEM/TMEM capacity instead of exposing stage-count
        # knobs to the runner.
        self.num_a_stage = pipeline_stages["a"]
        self.num_scale_stage = pipeline_stages["scale"]
        self.num_b_stage = pipeline_stages["b"]
        self.num_transformed_a_stage = pipeline_stages["transformed_a"]
        self.num_acc_stage = pipeline_stages["acc"]
        self.tmem_col_budget = self.get_tmem_col_budget(
            mma_tiler_mnk,
            transform_buffer,
            accumulator_overlap,
            transform_k_tile,
        )
        if self.tmem_col_budget["required_cols"] > 512:
            raise ValueError(
                "mixed FC12 TMEM budget exceeds 512 columns: "
                f"{self.tmem_col_budget}."
            )
        self.smem_buffer_align_bytes = 1024

    def _setup_warp_topology(self) -> None:
        """Select the lean or token-communication warp layout.

        Lean remains the original 12-warp layout.  MegaMoE adds four dispatch
        warps without making them scheduler consumers; the four mixed-input
        transform warps move to the final warpgroup.
        """
        if self.enable_token_comm:
            expected_dispatch = (8, 9, 10, 11)
            if (
                self.dispatch_warp_id is not None
                and self.dispatch_warp_id != expected_dispatch
            ):
                raise ValueError(
                    "mixed FC12 token communication requires dispatch warps "
                    "8-11."
                )
            self.dispatch_warp_id = expected_dispatch
            self.transform_warp_id = (12, 13, 14, 15)
            self.threads_per_cta = 16 * 32
        else:
            self.dispatch_warp_id = None
            self.transform_warp_id = (8, 9, 10, 11)
            self.threads_per_cta = 12 * 32

        # The TMEM allocation rendezvous covers every launched thread.  The
        # transform rendezvous always covers exactly the four transform warps.
        self.tmem_ptr_sync_barrier = pipeline.NamedBarrier(
            2, self.threads_per_cta
        )
        self.transform_sync_barrier = pipeline.NamedBarrier(
            3, 32 * len(self.transform_warp_id)
        )

    def name(self) -> str:
        weight = getattr(self, "a_dtype", "fp8")
        overlap = "_overlap2" if self.overlapping_accum else ""
        return (
            "sm100_swapab_mxfp8_bf16_glu_fc12_"
            f"{weight}_m{self.mma_tiler[0]}n{self.mma_tiler[1]}"
            f"k{self.tma_k_tile}{overlap}_2cta_"
            f"a{self.transformed_a_storage}"
        )

    def get_workspace_size_in_bytes(
        self,
        activation_tensor,
        fc1_weight_tensor,
    ) -> int:
        """Return workspace bytes for BF16 hand-off and phase counter.

        The public interface accepts the two views separately, but this helper
        mirrors the existing FC12 runners so a later runner can partition one
        opaque allocation without changing the kernel ABI.
        """
        token_rows, _hidden = activation_tensor.shape
        experts, _hidden_w, intermediate_gateup = fc1_weight_tensor.shape
        intermediate = intermediate_gateup // 2
        fc1_bytes = token_rows * intermediate * cutlass.BFloat16.width // 8
        cluster_tile_tokens = self.mma_tiler_mnk[1]
        counter_slots = (
            (token_rows + cluster_tile_tokens - 1) // cluster_tile_tokens
            + experts
        )
        counter_bytes = counter_slots * 4
        load_balance_bytes = 4 if self.load_balance_mode == "atomic_counter" else 0
        total = fc1_bytes + counter_bytes + load_balance_bytes
        return ((total + 127) // 128) * 128

    def _create_tiled_mma(self) -> cute.TiledMma:
        return sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            cutlass.BFloat16,
            self.a_major_mode,
            self.b_major_mode,
            self.acc_dtype,
            self.cta_group,
            self.mma_tiler[:2],
            self.transform_a_source,
        )

    def _setup_attributes(self) -> None:
        self._setup_warp_topology()
        tiled_mma = self._create_tiled_mma()
        self.cta_tile_shape_mnk = (
            self.mma_tiler[0] // cute.size(tiled_mma.thr_id.shape),
            self.mma_tiler[1],
            self.mma_tiler[2],
        )
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout((*self.cluster_shape_mn, 1)),
            (tiled_mma.thr_id.shape,),
        )
        self.num_mcast_ctas_a = cute.size(self.cluster_layout_vmnk.shape[2])
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_a_mcast = self.num_mcast_ctas_a > 1
        self.is_b_mcast = self.num_mcast_ctas_b > 1

        self.epilogue = MixedGluBf16Epilogue(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            fc1_output_dtype=cutlass.BFloat16,
            acc_dtype=self.acc_dtype,
            static_expert_shape=self.static_expert_shape,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            in_kernel_fc2_reduce=self.fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=self.token_back_by_dispatch,
            gate_up_clamp=self.gate_up_clamp,
            epi_flag_batch=self.epi_flag_batch,
            accumulator_overlap=self.overlapping_accum,
        )
        self.num_acc_stage = self.epilogue.num_acc_stage
        self.num_acc_pipeline_stages = (
            self.epilogue.num_acc_pipeline_stages
        )

        self.a_smem_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.tma_tiler,
            self.a_dtype,
            self.num_a_stage,
        )
        self.transformed_a_layout_staged = sm100_utils.make_smem_layout_a(
            tiled_mma,
            self.mma_tiler,
            cutlass.BFloat16,
            self.num_transformed_a_stage,
        )
        self.b_smem_layout_staged = sm100_utils.make_smem_layout_b(
            tiled_mma,
            self.tma_tiler,
            cutlass.BFloat16,
            self.num_b_stage,
        )

        self.scale_tile_shape = (
            self.cta_tile_shape_mnk[0],
            self.scale_tma_tile_k,
        )
        scale_layout_staged = blockscaled_utils.make_smem_layout_sf(
            self.scale_tile_shape,
            self.ScaleGranularityK,
            self.num_scale_stage,
        )
        trivial_swizzle = cute.make_swizzle(0, 4, 3)
        self.scale_smem_layout_staged = cute.make_composed_layout(
            trivial_swizzle, 0, scale_layout_staged
        )
        self.scale_smem_layout_per_stage = cute.slice_(
            self.scale_smem_layout_staged, (None, None, 0)
        )

        # TMEM = two FP32 accumulator stages plus optional BF16
        # transformed-A stages.  The N256/K128 overlap implementation uses
        # two internal K64 phases and overlaps accumulator stages by 64
        # columns; the N256/K128 SMEM implementation stores transformed-A in
        # SMEM instead.
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_one = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        acc_cols_per_stage = utils.get_num_tmem_alloc_cols(acc_one, True)
        if acc_cols_per_stage != self.tmem_col_budget["acc_cols_per_stage"]:
            raise ValueError(
                "actual accumulator TMEM layout disagrees with the host "
                f"budget: actual={acc_cols_per_stage}, "
                f"planned={self.tmem_col_budget}."
            )
        self.num_acc_tmem_cols = self.tmem_col_budget["acc_cols"]
        elements_per_tmem_col = 32 // cutlass.BFloat16.width
        a_cols_per_stage = cute.round_up(
            self.cta_tile_shape_mnk[2] // elements_per_tmem_col, 4
        )
        if (
            a_cols_per_stage
            != self.tmem_col_budget["transformed_a_cols_per_stage"]
        ):
            raise ValueError(
                "actual transformed-A TMEM layout disagrees with the host "
                f"budget: actual={a_cols_per_stage}, "
                f"planned={self.tmem_col_budget}."
            )
        self.num_transformed_a_tmem_cols = (
            a_cols_per_stage * self.num_transformed_a_stage
            if self.transformed_a_storage == "tmem"
            else 0
        )
        required_cols = (
            self.num_acc_tmem_cols + self.num_transformed_a_tmem_cols
        )
        self.num_tmem_required_cols = required_cols
        if required_cols > 512:
            raise ValueError(
                f"mixed TMEM requirement {required_cols} exceeds 512 columns."
            )
        self.num_tmem_alloc_cols = 1 << max(
            5, (required_cols - 1).bit_length()
        )

        self.num_transformed_a_smem_bytes = (
            cute.size_in_bytes(
                cutlass.BFloat16, self.transformed_a_layout_staged.outer
            )
            if self.transformed_a_storage == "smem"
            else 0
        )

        a_stage = cute.slice_(
            self.a_smem_layout_staged, (None, None, None, 0)
        )
        b_stage = cute.slice_(
            self.b_smem_layout_staged, (None, None, None, 0)
        )
        self.num_tma_load_bytes_a = cute.size_in_bytes(self.a_dtype, a_stage)
        self.num_tma_load_bytes_b = (
            cute.size_in_bytes(cutlass.BFloat16, b_stage)
            * cute.size(tiled_mma.thr_id.shape)
        )
        self.num_tma_load_bytes_scale = cute.size_in_bytes(
            self.scale_dtype, self.scale_smem_layout_per_stage
        )

    def _validate_inputs(
        self,
        activation: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_output: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_output: cute.Tensor,
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        offs: Optional[cute.Tensor],
        load_balance_counter: Optional[cute.Tensor],
        token_comm_args,
    ) -> None:
        for name, tensor, expected_rank in (
            ("activation", activation, 2),
            ("fc1_weight", fc1_weight, 3),
            ("fc1_weight_sf", fc1_weight_sf, 2),
            ("fc1_output", fc1_output, 2),
            ("fc2_weight", fc2_weight, 3),
            ("fc2_weight_sf", fc2_weight_sf, 2),
            ("fc2_output", fc2_output, 3),
            ("topk_scores", topk_scores, 1),
            ("fc1_done_counter", fc1_done_counter, 1),
        ):
            if cutlass.const_expr(cute.rank(tensor) != expected_rank):
                raise ValueError(
                    f"{name} must be rank {expected_rank}, got "
                    f"{cute.rank(tensor)}."
                )

        fp8_types = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        if cutlass.const_expr(fc1_weight.element_type not in fp8_types):
            raise TypeError("fc1_weight must be E4M3FN or E5M2.")
        if cutlass.const_expr(fc2_weight.element_type is not fc1_weight.element_type):
            raise TypeError("FC1 and FC2 weights must use the same FP8 dtype.")
        if cutlass.const_expr(
            fc1_weight_sf.element_type is not cutlass.Float8E8M0FNU
            or fc2_weight_sf.element_type is not cutlass.Float8E8M0FNU
        ):
            raise TypeError("both weight scale tensors must be E8M0FNU.")
        for name, tensor in (
            ("activation", activation),
            ("fc1_output", fc1_output),
            ("fc2_output", fc2_output),
        ):
            if cutlass.const_expr(tensor.element_type is not cutlass.BFloat16):
                raise TypeError(f"{name} must be BFloat16.")
        if cutlass.const_expr(topk_scores.element_type is not cutlass.Float32):
            raise TypeError("topk_scores must be Float32.")
        if cutlass.const_expr(
            fc1_done_counter.element_type is not cutlass.Int32
        ):
            raise TypeError("fc1_done_counter must be Int32.")

        if self.enable_token_comm:
            if token_comm_args is None:
                raise ValueError(
                    "token_comm_args is required when enable_token_comm=True."
                )
            if offs is not None:
                raise ValueError(
                    "offs must be None in token-communication sizes mode."
                )
        else:
            if offs is None:
                raise ValueError(
                    "offs is required for the lean prefix-sum scheduler."
                )
            if cutlass.const_expr(cute.rank(offs) != 1):
                raise ValueError("offs must be rank 1.")
            if cutlass.const_expr(offs.element_type is not cutlass.Int32):
                raise TypeError("offs must be Int32.")

        if cutlass.const_expr(load_balance_counter is not None):
            if cutlass.const_expr(cute.rank(load_balance_counter) != 1):
                raise ValueError("load_balance_counter must be rank 1.")
            if cutlass.const_expr(
                load_balance_counter.element_type is not cutlass.Int32
            ):
                raise TypeError("load_balance_counter must be Int32.")

    # -------------------------------------------------------------------------
    # MegaMoE token-communication hooks
    # -------------------------------------------------------------------------
    #
    # The base implementations are deliberately empty.  They keep the lean
    # kernel free of token-communication IR while giving a MegaMoE subclass the
    # same device-kernel integration points as the NVFP4 swap-AB FC12 base.
    # ``token_comm_args`` is an opaque subclass-owned bundle.

    def token_comm_extra_smem_storage_class(self) -> Optional[type]:
        """Return the subclass-owned dispatch SMEM struct, if one is needed."""
        return None

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        """Return a dispatch-to-FC1 readiness counter pointer, if available.

        The correctness-first mixed scheduler extension currently disables its
        optional nonblocking peek.  This pointer remains part of the hook
        surface so a future extension can enable that optimization without
        changing the kernel ABI; FC1 correctness comes from the blocking TMA-B
        hook below.
        """
        return None

    def token_comm_scheduler_expert_token_sizes(
        self,
        token_comm_args,
        expert_count,
    ) -> cute.Tensor:
        """Build the scheduler's raw per-expert token-count view.

        Standard ``TokenCommArgs`` carries ``expert_recv_count_sum`` as an
        ``Int64[E]`` tensor whose low 32 bits hold the count.  Reinterpreting
        its base as ``Int32`` with stride two is the same zero-copy sizes view
        used by the BF16 and NVFP4 MegaMoE kernels.  A custom bundle may expose
        an already-formed ``expert_token_sizes`` tensor instead.
        """
        if hasattr(token_comm_args, "expert_token_sizes"):
            expert_token_sizes = token_comm_args.expert_token_sizes
            if expert_token_sizes is not None:
                if cutlass.const_expr(
                    cute.rank(expert_token_sizes) != 1
                    or expert_token_sizes.element_type is not cutlass.Int32
                ):
                    raise TypeError(
                        "token_comm_args.expert_token_sizes must be a rank-1 "
                        "Int32 tensor."
                    )
                if cutlass.const_expr(
                    expert_token_sizes.shape[0] != expert_count
                ):
                    raise ValueError(
                        "token_comm_args.expert_token_sizes must contain one "
                        "entry per local expert."
                    )
                return expert_token_sizes

        if not hasattr(token_comm_args, "expert_recv_count_sum"):
            raise ValueError(
                "token_comm_args must provide expert_recv_count_sum or "
                "expert_token_sizes for scheduler sizes mode."
            )
        expert_recv_count_sum = token_comm_args.expert_recv_count_sum
        if cutlass.const_expr(
            cute.rank(expert_recv_count_sum) != 1
            or expert_recv_count_sum.element_type is not cutlass.Int64
        ):
            raise TypeError(
                "token_comm_args.expert_recv_count_sum must be a rank-1 "
                "Int64 tensor whose low 32 bits contain each expert count."
            )
        if cutlass.const_expr(
            expert_recv_count_sum.shape[0] != expert_count
        ):
            raise ValueError(
                "token_comm_args.expert_recv_count_sum must contain one "
                "entry per local expert."
            )
        sizes_iterator = cute.make_ptr(
            cutlass.Int32,
            expert_recv_count_sum.iterator.toint(),
            AddressSpace.gmem,
            assumed_align=8,
        )
        return cute.make_tensor(
            sizes_iterator,
            cute.make_layout((expert_count,), stride=(2,)),
        )

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        """Wait for dispatch-produced scheduler metadata (lean: no-op)."""
        pass

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self,
        token_comm_args,
        work_tile_info,
    ):
        """Wait until a dispatched FC1 token tile is resident (lean: no-op)."""
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
        """Run one dispatch warp's communication body (lean: no-op)."""
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
        """Run the all-warp communication tail/rendezvous (lean: no-op)."""
        pass

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_output: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_output: cute.Tensor,
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        offs: Optional[cute.Tensor] = None,
        max_active_clusters: cutlass.Constexpr = None,
        stream: cuda.CUstream = None,
        load_balance_counter: Optional[cute.Tensor] = None,
        token_comm_args=None,
    ) -> None:
        """Trace and launch the lean two-phase mixed mainloop.

        Tensor contracts:

        * ``activation``: ``[physical_tokens, hidden]`` BF16, hidden-major.
        * ``fc1_weight``: ``[experts, hidden, 2I]`` FP8, hidden stride one.
        * ``fc1_weight_sf``: existing atom-swizzled E8M0/K32 storage.
        * ``fc1_output``: ``[physical_tokens, I]`` BF16 workspace.
        * ``fc2_weight``: ``[experts, I, hidden]`` FP8, I stride one.
        * ``fc2_weight_sf``: existing atom-swizzled E8M0/K32 storage.
        * ``fc2_output``: ``[physical_tokens, 1, hidden]`` BF16 destination.
        * Lean scheduler: pass cumulative expert ends in ``offs`` and leave
          ``token_comm_args`` unset.
        * Mega scheduler: omit ``offs``; ``token_comm_args`` must expose
          ``expert_recv_count_sum`` or a prebuilt ``expert_token_sizes`` view.
        """
        self._validate_inputs(
            activation,
            fc1_weight,
            fc1_weight_sf,
            fc1_output,
            fc2_weight,
            fc2_weight_sf,
            fc2_output,
            topk_scores,
            fc1_done_counter,
            offs,
            load_balance_counter,
            token_comm_args,
        )

        # Bind expert/feature dimensions to codegen-time constants when the
        # runner supplies a static expert shape.  Token rows and all strides
        # remain runtime-dynamic.
        if cutlass.const_expr(self.static_expert_shape is not None):
            (
                experts_static,
                intermediate_gateup_static,
                hidden_static,
            ) = self.static_expert_shape
            intermediate_static = intermediate_gateup_static // 2
            fc1_weight = cute.make_tensor(
                fc1_weight.iterator,
                cute.make_layout(
                    (
                        experts_static,
                        hidden_static,
                        intermediate_gateup_static,
                    ),
                    stride=fc1_weight.stride,
                ),
            )
            fc2_weight = cute.make_tensor(
                fc2_weight.iterator,
                cute.make_layout(
                    (experts_static, intermediate_static, hidden_static),
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
                    (fc1_output.shape[0], intermediate_static),
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

        tokens, hidden = activation.shape
        experts, hidden_w1, intermediate_gateup = fc1_weight.shape
        experts_w2, intermediate, hidden_w2 = fc2_weight.shape

        # Swap-AB views.  Weight is A (M,K,L=expert); token data is B
        # (N,K,L=1).  No physical transpose is performed.
        fc1_weight_gemm = cute.make_tensor(
            fc1_weight.iterator,
            cute.make_layout(
                (intermediate_gateup, hidden_w1, experts),
                stride=(
                    fc1_weight.stride[2],
                    fc1_weight.stride[1],
                    fc1_weight.stride[0],
                ),
            ),
        )
        activation_gemm = cute.make_tensor(
            activation.iterator,
            cute.make_layout(
                (tokens, hidden, 1),
                stride=(activation.stride[0], activation.stride[1], 0),
            ),
        )
        fc1_output_gemm = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout(
                (tokens, intermediate, 1),
                stride=(fc1_output.stride[0], fc1_output.stride[1], 0),
            ),
        )
        fc2_weight_gemm = cute.make_tensor(
            fc2_weight.iterator,
            cute.make_layout(
                (hidden_w2, intermediate, experts_w2),
                stride=(
                    fc2_weight.stride[2],
                    fc2_weight.stride[1],
                    fc2_weight.stride[0],
                ),
            ),
        )

        self.a_dtype = fc1_weight.element_type
        self.scale_dtype = fc1_weight_sf.element_type
        self.a_major_mode = utils.LayoutEnum.from_tensor(
            fc1_weight_gemm
        ).mma_major_mode()
        self.b_major_mode = utils.LayoutEnum.from_tensor(
            activation_gemm
        ).mma_major_mode()
        if cutlass.const_expr(
            self.a_major_mode != cute.nvgpu.OperandMajorMode.K
        ):
            raise ValueError("MXFP8 weight must be K-major.")
        if cutlass.const_expr(
            self.b_major_mode != cute.nvgpu.OperandMajorMode.K
        ):
            raise ValueError("BF16 activation/handoff must be K-major.")

        self._setup_attributes()
        tiled_mma = self._create_tiled_mma()

        a_op = mixed_input_utils.get_tma_atom_kind(
            self.is_a_mcast, self.use_2cta_instrs, False
        )
        b_op = mixed_input_utils.get_tma_atom_kind(
            self.is_b_mcast, self.use_2cta_instrs, True
        )
        a_smem_stage = cute.slice_(
            self.a_smem_layout_staged, (None, None, None, 0)
        )
        b_smem_stage = cute.slice_(
            self.b_smem_layout_staged, (None, None, None, 0)
        )

        tma_atom_w1, tma_tensor_w1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc1_weight_gemm,
            a_smem_stage,
            self.tma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_w2, tma_tensor_w2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            fc2_weight_gemm,
            a_smem_stage,
            self.tma_tiler,
            tiled_mma,
            self.cluster_layout_vmnk.shape,
        )
        tma_atom_activation, tma_tensor_activation = (
            cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                activation_gemm,
                b_smem_stage,
                self.tma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )
        tma_atom_fc1_output, tma_tensor_fc1_output = (
            cute.nvgpu.make_tiled_tma_atom_B(
                b_op,
                fc1_output_gemm,
                b_smem_stage,
                self.tma_tiler,
                tiled_mma,
                self.cluster_layout_vmnk.shape,
            )
        )

        # Keep the public 32x4x4 atom-swizzled ABI.  These are logical expanded
        # MKL views backed by the caller's flattened/swizzled scale planes.
        w1_scale_gemm = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                fc1_weight_gemm.shape, self.ScaleGranularityK
            ),
        )
        w2_scale_gemm = cute.make_tensor(
            fc2_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(
                fc2_weight_gemm.shape, self.ScaleGranularityK
            ),
        )
        tma_atom_w1_scale, tma_tensor_w1_scale = cpasync.make_tiled_tma_atom(
            a_op,
            w1_scale_gemm,
            self.scale_smem_layout_staged,
            self.scale_tile_shape,
            self.num_mcast_ctas_a,
            internal_type=cutlass.Uint8,
        )
        tma_atom_w2_scale, tma_tensor_w2_scale = cpasync.make_tiled_tma_atom(
            a_op,
            w2_scale_gemm,
            self.scale_smem_layout_staged,
            self.scale_tile_shape,
            self.num_mcast_ctas_a,
            internal_type=cutlass.Uint8,
        )

        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            if cutlass.const_expr(load_balance_counter is None):
                raise ValueError(
                    "load_balance_counter is required in atomic_counter mode."
                )
            load_balance_counter_ptr = load_balance_counter.iterator
        else:
            load_balance_counter_ptr = None

        sched_kwargs = dict(
            scenario="2Dx3D",
            expert_shape=(experts, intermediate_gateup, hidden),
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            # There is no token-indexed SF plane.  The fused scheduler still
            # carries its legacy SF cumulative field; a neutral unit padding
            # keeps that unused field well-defined.
            sf_padding_block=1,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=load_balance_counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=True,
        )
        # Select one scheduler range source at trace time and omit the absent
        # optional argument entirely.  This preserves the lean prefix-sum ABI
        # and avoids forwarding Python ``None`` through CuTeDSL call layers.
        if cutlass.const_expr(self.enable_token_comm):
            expert_token_sizes = (
                self.token_comm_scheduler_expert_token_sizes(
                    token_comm_args,
                    experts,
                )
            )
            sched_params = MoEFusedFc12SchedulerParams(
                **sched_kwargs,
                expert_token_sizes=expert_token_sizes,
            )
        else:
            sched_params = MoEFusedFc12SchedulerParams(
                **sched_kwargs,
                expert_token_prefix_sum=offs,
            )
        grid = sched_params.get_grid_shape(max_active_clusters)

        kernel_args = (
            tiled_mma,
            tma_atom_w1,
            tma_tensor_w1,
            tma_atom_w1_scale,
            tma_tensor_w1_scale,
            tma_atom_activation,
            tma_tensor_activation,
            tma_atom_w2,
            tma_tensor_w2,
            tma_atom_w2_scale,
            tma_tensor_w2_scale,
            tma_atom_fc1_output,
            tma_tensor_fc1_output,
            fc1_weight_gemm,
            w1_scale_gemm,
            activation_gemm,
            fc2_weight_gemm,
            w2_scale_gemm,
            fc1_output_gemm,
            fc2_output,
            topk_scores,
            fc1_done_counter,
            sched_params,
            self.cluster_layout_vmnk,
            self.a_smem_layout_staged,
            self.scale_smem_layout_staged,
            self.transformed_a_layout_staged,
            self.b_smem_layout_staged,
        )
        # Do not pass Python ``None`` explicitly through a CuTeDSL call.  The
        # lean specialization uses the device kernel's default argument; a
        # MegaMoE subclass supplies its concrete opaque bundle.
        if cutlass.const_expr(token_comm_args is not None):
            compiled_kernel = self.kernel(*kernel_args, token_comm_args)
        else:
            compiled_kernel = self.kernel(*kernel_args)
        compiled_kernel.launch(
            grid=grid,
            block=[self.threads_per_cta, 1, 1],
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=self.occupancy,
        )

    @cute.jit
    def _transform_k_tiles(
        self,
        k_tile_cnt,
        is_fc1,
        iket_active,
        a_load_pipeline,
        scale_load_pipeline,
        transformed_a_pipeline,
        a_consumer_state,
        scale_consumer_state,
        transformed_producer_state,
        tAsA_input,
        transform_tiler,
        tSsScale_transform,
        tSrScale_copy,
        tArA_load,
        tSrScale_load,
        tArA_transform,
        tArA_transform_store,
        tAsA_transform,
        dst_copy_a,
    ):
        """Transform one work tile using the caller-specialized K trip count."""

        a_consumer_state.reset_count()
        scale_consumer_state.reset_count()
        transformed_producer_state.reset_count()
        peek_a = a_load_pipeline.consumer_try_wait(a_consumer_state)
        peek_scale = scale_load_pipeline.consumer_try_wait(
            scale_consumer_state
        )
        peek_transformed = transformed_a_pipeline.producer_try_acquire(
            transformed_producer_state
        )
        phase_payload = cutlass.Int32(is_fc1)
        coverage_stage_payload = (
            cutlass.Int32(302) + phase_payload * cutlass.Int32(10)
        )
        if iket_active:
            # Closes the setup segment opened by the caller before branch
            # cloning and helper dispatch.
            iket.range_pop()

        scale_reuse = (
            2 if cutlass.const_expr(self.mma_tiler[2] == 64) else 1
        )
        scale_tile_cnt = k_tile_cnt // scale_reuse
        for _scale_k in cutlass.range(0, scale_tile_cnt, 1, unroll=1):
            if iket_active:
                iket.range_push(
                    "mixed_transform_scale_wait", phase_payload
                )
            scale_load_pipeline.consumer_wait(
                scale_consumer_state, peek_scale
            )
            if iket_active:
                iket.range_pop()

            scale_slice = tSsScale_transform[
                (
                    None,
                    None,
                    None,
                    None,
                    scale_consumer_state.index,
                )
            ]
            scale_slice = cute.make_tensor(
                scale_slice.iterator,
                cute.filter_zeros(scale_slice.layout),
            )
            cute.autovec_copy(scale_slice, tSrScale_copy)
            current_scale_state = scale_consumer_state.clone()
            if cutlass.const_expr(self.tma_k_reuse == 2):
                a_load_pipeline.consumer_wait(a_consumer_state, peek_a)
                current_macro_a_state = a_consumer_state.clone()
                macro_input_slice = tAsA_input[
                    (
                        None,
                        None,
                        None,
                        None,
                        a_consumer_state.index,
                    )
                ]
                macro_input_slice = cute.flat_divide(
                    macro_input_slice, transform_tiler
                )
                macro_input_slice = cute.group_modes(
                    macro_input_slice, 1, cute.rank(macro_input_slice)
                )
            for scale_half_idx in cutlass.range_constexpr(scale_reuse):
                if iket_active:
                    iket.range_push(
                        "mixed_role_coverage", coverage_stage_payload
                    )
                    iket.range_push(
                        "mixed_transform_consumer_wait", phase_payload
                    )
                if cutlass.const_expr(self.tma_k_reuse == 1):
                    a_load_pipeline.consumer_wait(a_consumer_state, peek_a)
                if iket_active:
                    iket.range_pop()
                    iket.range_push(
                        "mixed_transform_producer_acquire", phase_payload
                    )
                transformed_a_pipeline.producer_acquire(
                    transformed_producer_state,
                    peek_transformed,
                )
                if iket_active:
                    iket.range_pop()
                    iket.range_push(
                        "mixed_transform_stage_work", phase_payload
                    )

                if cutlass.const_expr(self.tma_k_reuse == 1):
                    input_slice = tAsA_input[
                        (
                            None,
                            None,
                            None,
                            None,
                            a_consumer_state.index,
                        )
                    ]
                    input_slice = cute.flat_divide(
                        input_slice, transform_tiler
                    )
                    input_slice = cute.group_modes(
                        input_slice, 1, cute.rank(input_slice)
                    )
                    current_a_state = a_consumer_state.clone()
                else:
                    input_slice = macro_input_slice
                fragment_begin = (
                    scale_half_idx
                    if cutlass.const_expr(self.tma_k_reuse == 2)
                    else 0
                )
                fragment_end = (
                    fragment_begin + 1
                    if cutlass.const_expr(self.tma_k_reuse == 2)
                    else cute.size(tArA_load, mode=[1])
                )
                for fragment_idx in cutlass.range_constexpr(
                    fragment_begin, fragment_end, 1
                ):
                    cute.autovec_copy(
                        input_slice[(None, fragment_idx)],
                        tArA_load[(None, fragment_idx)],
                    )
                    if cutlass.const_expr(
                        self.tma_k_reuse == 1
                        and
                        fragment_idx
                        == cute.size(tArA_load, mode=[1]) - 1
                    ):
                        a_consumer_state.advance()

                    transformed = mixed_input_utils.cvt_tensor_a(
                        tArA_load[(None, fragment_idx)],
                        cutlass.BFloat16,
                        False,
                    )
                    value_count = cute.size(transformed.shape)
                    if cutlass.const_expr(
                        self.transform_a_source
                        == tcgen05.OperandSource.SMEM
                    ):
                        scale_fragment = tSrScale_load[
                            (None, fragment_idx)
                        ].load().to(cutlass.BFloat16)
                        assert value_count == cute.size(scale_fragment.shape)
                        scale = cute.TensorSSA(
                            scale_fragment,
                            transformed.shape,
                            cutlass.BFloat16,
                        )
                    else:
                        scale_fragment_idx = (
                            scale_half_idx
                            if cutlass.const_expr(self.mma_tiler[2] == 64)
                            else fragment_idx
                        )
                        scale_lo = tSrScale_load[
                            (None, (0, scale_fragment_idx))
                        ].load().to(cutlass.BFloat16)
                        scale_hi = tSrScale_load[
                            (None, (1, scale_fragment_idx))
                        ].load().to(cutlass.BFloat16)
                        scale_count = cute.size(scale_lo.shape)
                        assert value_count == 2 * scale_count
                        scale_expanded = mlir_vector.shuffle(
                            scale_lo,
                            scale_hi,
                            tuple(range(value_count)),
                        )
                        scale = cute.TensorSSA(
                            scale_expanded,
                            transformed.shape,
                            cutlass.BFloat16,
                        )
                    transformed = transformed * scale
                    transform_store_idx = (
                        0
                        if cutlass.const_expr(self.tma_k_reuse == 2)
                        else fragment_idx
                    )
                    tArA_transform_store[
                        (None, transform_store_idx)
                    ].store(transformed)

                transformed_store_destination = tAsA_transform[
                    (
                        None,
                        None,
                        None,
                        None,
                        transformed_producer_state.index,
                    )
                ]
                mixed_input_utils.store_transformed_a(
                    tArA_transform,
                    transformed_store_destination,
                    dst_copy_a,
                )
                self.transform_sync_barrier.arrive_and_wait()
                if cutlass.const_expr(
                    self.transform_a_source == tcgen05.OperandSource.TMEM
                ):
                    cute.arch.fence_view_async_tmem_store()
                else:
                    cute.arch.fence_proxy("async.shared", space="cta")

                if cutlass.const_expr(self.tma_k_reuse == 1):
                    a_load_pipeline.consumer_release(current_a_state)
                transformed_a_pipeline.producer_commit(
                    transformed_producer_state
                )
                transformed_producer_state.advance()

                peek_a = cutlass.Boolean(1)
                peek_transformed = cutlass.Boolean(1)
                if transformed_producer_state.count < k_tile_cnt:
                    if cutlass.const_expr(self.tma_k_reuse == 1):
                        peek_a = a_load_pipeline.consumer_try_wait(
                            a_consumer_state
                        )
                    peek_transformed = (
                        transformed_a_pipeline.producer_try_acquire(
                            transformed_producer_state
                        )
                    )
                if iket_active:
                    iket.range_pop()
                    iket.range_pop()

            if cutlass.const_expr(self.tma_k_reuse == 2):
                a_load_pipeline.consumer_release(current_macro_a_state)
                a_consumer_state.advance()
                peek_a = cutlass.Boolean(1)
                if a_consumer_state.count < scale_tile_cnt:
                    peek_a = a_load_pipeline.consumer_try_wait(
                        a_consumer_state
                    )
            scale_load_pipeline.consumer_release(current_scale_state)
            scale_consumer_state.advance()
            peek_scale = cutlass.Boolean(1)
            if scale_consumer_state.count < scale_tile_cnt:
                peek_scale = scale_load_pipeline.consumer_try_wait(
                    scale_consumer_state
                )

        return (
            a_consumer_state,
            scale_consumer_state,
            transformed_producer_state,
        )

    @cute.jit
    def _mma_k_tiles(
        self,
        k_tile_cnt,
        is_fc1,
        iket_active,
        tiled_mma,
        tCtAcc,
        tCrA,
        tCrB,
        transformed_a_pipeline,
        b_load_pipeline,
        acc_pipeline,
        transformed_consumer_state,
        b_consumer_state,
        acc_producer_state,
    ):
        """Consume one transformed-A/B work tile with a specialized K bound."""

        transformed_consumer_state.reset_count()
        b_consumer_state.reset_count()
        peek_transformed = transformed_a_pipeline.consumer_try_wait(
            transformed_consumer_state
        )
        # Keep these names phase-neutral to stay within IKET's per-kernel
        # event-name budget. NativeDump preserves ``is_fc1`` as the payload:
        # 1 means FC1 and 0 means FC2.
        phase_payload = cutlass.Int32(is_fc1)
        if iket_active:
            iket.range_push("mixed_mma_producer_acquire", phase_payload)
        acc_pipeline.producer_acquire(acc_producer_state)
        if iket_active:
            iket.range_pop()
        tiled_mma.set(tcgen05.Field.ACCUMULATE, False)
        if iket_active:
            # Closes the setup segment opened by the caller.
            iket.range_pop()

        coverage_stage_payload = (
            cutlass.Int32(402) + phase_payload * cutlass.Int32(10)
        )

        b_tile_cnt = k_tile_cnt // self.tma_k_reuse
        for _b_tile in cutlass.range(0, b_tile_cnt, 1, unroll=1):
            b_load_pipeline.consumer_wait(b_consumer_state)
            for b_half_idx in cutlass.range_constexpr(self.tma_k_reuse):
                if iket_active:
                    iket.range_push(
                        "mixed_role_coverage", coverage_stage_payload
                    )
                    iket.range_push(
                        "mixed_mma_consumer_wait", phase_payload
                    )
                transformed_a_pipeline.consumer_wait(
                    transformed_consumer_state,
                    peek_transformed,
                )
                if iket_active:
                    iket.range_pop()
                    iket.range_push("mixed_mma_stage_work", phase_payload)
                num_kblocks = cute.size(tCrA, mode=[2])
                for kblock in cutlass.range(
                    num_kblocks, unroll_full=True
                ):
                    b_k_coord = (
                        (kblock, b_half_idx)
                        if cutlass.const_expr(self.tma_k_reuse == 2)
                        else kblock
                    )
                    cute.gemm(
                        tiled_mma,
                        tCtAcc,
                        tCrA[
                            (
                                None,
                                None,
                                kblock,
                                transformed_consumer_state.index,
                            )
                        ],
                        tCrB[
                            (
                                None,
                                None,
                                b_k_coord,
                                b_consumer_state.index,
                            )
                        ],
                        tCtAcc,
                    )
                    tiled_mma.set(tcgen05.Field.ACCUMULATE, True)
                transformed_a_pipeline.consumer_release(
                    transformed_consumer_state
                )
                transformed_consumer_state.advance()
                peek_transformed = cutlass.Boolean(1)
                if transformed_consumer_state.count < k_tile_cnt:
                    peek_transformed = (
                        transformed_a_pipeline.consumer_try_wait(
                            transformed_consumer_state
                        )
                    )
                if iket_active:
                    iket.range_pop()
                    iket.range_pop()
            b_load_pipeline.consumer_release(b_consumer_state)
            b_consumer_state.advance()
        acc_pipeline.producer_commit(acc_producer_state)
        return tiled_mma, transformed_consumer_state, b_consumer_state

    @cute.kernel
    def kernel(
        self,
        tiled_mma: cute.TiledMma,
        # FC1 weight-A, scale-A, token-B
        tma_atom_w1: cute.CopyAtom,
        tma_tensor_w1: cute.Tensor,
        tma_atom_w1_scale: cute.CopyAtom,
        tma_tensor_w1_scale: cute.Tensor,
        tma_atom_activation: cute.CopyAtom,
        tma_tensor_activation: cute.Tensor,
        # FC2 weight-A, scale-A, BF16 handoff-B
        tma_atom_w2: cute.CopyAtom,
        tma_tensor_w2: cute.Tensor,
        tma_atom_w2_scale: cute.CopyAtom,
        tma_tensor_w2_scale: cute.Tensor,
        tma_atom_fc1_output: cute.CopyAtom,
        tma_tensor_fc1_output: cute.Tensor,
        # Logical tensors used for dynamic expert/task slicing
        fc1_weight_gemm: cute.Tensor,
        w1_scale_gemm: cute.Tensor,
        activation_gemm: cute.Tensor,
        fc2_weight_gemm: cute.Tensor,
        w2_scale_gemm: cute.Tensor,
        fc1_output_gemm: cute.Tensor,
        fc2_output: cute.Tensor,
        topk_scores: cute.Tensor,
        fc1_done_counter: cute.Tensor,
        sched_params: MoEFusedFc12SchedulerParams,
        cluster_layout_vmnk: cute.Layout,
        a_smem_layout_staged: cute.ComposedLayout,
        scale_smem_layout_staged: cute.ComposedLayout,
        transformed_a_layout_staged: cute.ComposedLayout,
        b_smem_layout_staged: cute.ComposedLayout,
        token_comm_args=None,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        tidx, _, _ = cute.arch.thread_idx()
        bidx, _, _ = cute.arch.block_idx()
        use_2cta_instrs = cute.size(tiled_mma.thr_id.shape) == 2
        mma_tile_coord_v = bidx % cute.size(tiled_mma.thr_id.shape)
        is_leader_cta = mma_tile_coord_v == 0
        cta_rank_in_cluster = cute.arch.make_warp_uniform(
            cute.arch.block_idx_in_cluster()
        )
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(
            cta_rank_in_cluster
        )

        a_smem_stage = cute.slice_(
            a_smem_layout_staged, (None, None, None, 0)
        )
        scale_smem_stage = cute.slice_(
            scale_smem_layout_staged, (None, None, 0)
        )
        b_smem_stage = cute.slice_(
            b_smem_layout_staged, (None, None, None, 0)
        )

        # The swap scheduler's SF cumulative field is unused, but its
        # expert-indexed "sfa" view is exactly the weight-scale mapping needed
        # by both phases.
        ext_fc2_spin_threshold = (
            fc1_weight_gemm.shape[0] + self.cta_tile_shape_mnk[0] - 1
        ) // self.cta_tile_shape_mnk[0]
        ext = _MixedBf16NoPeekSchedExtension(
            sf_vec_size=self.ScaleGranularityK,
            fc1_done_counter_ptr=fc1_done_counter.iterator,
            fc2_spin_threshold=ext_fc2_spin_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
        )
        SchedCls = sched_params.get_scheduler_type()
        SchedStorage = SchedCls.make_storage_struct(
            sched_params, ext, num_drain_warps=0
        )

        @cute.struct
        class SharedStorage:
            a_load_full: cute.struct.MemRange[
                cutlass.Int64, self.num_a_stage
            ]
            a_load_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_a_stage
            ]
            scale_load_full: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_stage
            ]
            scale_load_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_scale_stage
            ]
            transformed_a_full: cute.struct.MemRange[
                cutlass.Int64, self.num_transformed_a_stage
            ]
            transformed_a_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_transformed_a_stage
            ]
            b_load_full: cute.struct.MemRange[
                cutlass.Int64, self.num_b_stage
            ]
            b_load_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_b_stage
            ]
            acc_full: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages
            ]
            acc_empty: cute.struct.MemRange[
                cutlass.Int64, self.num_acc_pipeline_stages
            ]
            sched_storage: SchedStorage
            tmem_dealloc_mbar: cutlass.Int64
            tmem_holding_buf: cutlass.Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        TokenCommStorageCls = self.token_comm_extra_smem_storage_class()
        if cutlass.const_expr(TokenCommStorageCls is not None):
            token_comm_storage = smem.allocate(TokenCommStorageCls)
        else:
            token_comm_storage = None
        epi_smem_storage = smem.allocate(
            self.epilogue.get_epi_storage_type()
        )

        transform_tidx = (
            tidx - 32 * self.transform_warp_id[0]
            if tidx >= 32 * self.transform_warp_id[0]
            else tidx
        )
        a_load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.a_load_full.data_ptr(),
            num_stages=self.num_a_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * len(self.transform_warp_id),
            ),
            tx_count=self.num_tma_load_bytes_a,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_tidx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        scale_load_pipeline = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.scale_load_full.data_ptr(),
            num_stages=self.num_scale_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                self.num_mcast_ctas_a * len(self.transform_warp_id),
            ),
            tx_count=self.num_tma_load_bytes_scale,
            cta_layout_vmnk=cluster_layout_vmnk,
            tidx=transform_tidx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        transformed_a_pipeline = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.transformed_a_full.data_ptr(),
            num_stages=self.num_transformed_a_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                32
                * len(self.transform_warp_id)
                * cute.size(cluster_layout_vmnk, mode=[0]),
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )
        b_load_pipeline = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.b_load_full.data_ptr(),
            num_stages=self.num_b_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, self.num_mcast_ctas_b
            ),
            tx_count=self.num_tma_load_bytes_b,
            cta_layout_vmnk=cluster_layout_vmnk,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )
        acc_pipeline = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_full.data_ptr(),
            num_stages=self.num_acc_pipeline_stages,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread
            ),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread,
                cute.size(cluster_layout_vmnk, mode=[0])
                * 32
                * len(self.epilogue_warp_id),
            ),
            cta_layout_vmnk=cluster_layout_vmnk,
            defer_sync=True,
        )

        tmem = utils.TmemAllocator(
            storage.tmem_holding_buf.ptr,
            barrier_for_retrieve=self.tmem_ptr_sync_barrier,
            allocator_warp_id=self.epilogue_warp_id[0],
            is_two_cta=use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc_mbar.ptr,
            arch=self.arch,
        )

        # Dispatch warps do not consume scheduler work.  The count is identical
        # in lean and MegaMoE layouts: A-TMA, B-TMA, MMA, four transform warps,
        # and four epilogue warps.
        num_sched_consumer_threads = 32 * len(
            (
                self.tma_a_warp_id,
                self.tma_b_warp_id,
                self.mma_warp_id,
                *self.transform_warp_id,
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
        early_internal_init = (
            self.load_balance_mode == "atomic_counter"
            or not self.enable_token_comm
        )
        if cutlass.const_expr(early_internal_init):
            scheduler.internal_init(
                warp_idx=warp_idx, sched_warp_id=self.sched_warp_id
            )

        pipeline_init_arrive(
            cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True
        )

        sA = smem.allocate_tensor(
            element_type=self.a_dtype,
            layout=a_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=a_smem_layout_staged.inner,
        )
        sScale = smem.allocate_tensor(
            element_type=self.scale_dtype,
            layout=scale_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=scale_smem_layout_staged.inner,
        )
        sB = smem.allocate_tensor(
            element_type=cutlass.BFloat16,
            layout=b_smem_layout_staged.outer,
            byte_alignment=128,
            swizzle=b_smem_layout_staged.inner,
        )

        sATransformed = None
        if cutlass.const_expr(
            self.transform_a_source == tcgen05.OperandSource.SMEM
        ):
            sATransformed = smem.allocate_tensor(
                element_type=cutlass.BFloat16,
                layout=transformed_a_layout_staged.outer,
                byte_alignment=128,
                swizzle=transformed_a_layout_staged.inner,
            )

        tCrB = tiled_mma.make_fragment_B(sB)
        acc_shape = tiled_mma.partition_shape_C(self.mma_tiler[:2])
        acc_fake = tiled_mma.make_fragment_C(
            cute.append(acc_shape, self.num_acc_stage)
        )
        if cutlass.const_expr(self.overlapping_accum):
            acc_fake = cute.make_tensor(
                acc_fake.iterator,
                cute.make_layout(
                    acc_fake.shape,
                    stride=(
                        acc_fake.stride[0],
                        acc_fake.stride[1],
                        acc_fake.stride[2],
                        self.tmem_col_budget["acc_stage_stride"]
                        * acc_fake.stride[0][1],
                    ),
                ),
            )

        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(self.num_tmem_alloc_cols)
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc_dtype)
        accumulators = cute.make_tensor(tmem_ptr, acc_fake.layout)
        if cutlass.const_expr(
            self.transform_a_source == tcgen05.OperandSource.TMEM
        ):
            transformed_a_ptr = cute.recast_ptr(
                tmem_ptr + self.num_acc_tmem_cols,
                dtype=cutlass.BFloat16,
            )
            tCrA = cute.make_tensor(
                transformed_a_ptr,
                tiled_mma.make_fragment_A(
                    transformed_a_layout_staged.outer
                ).layout,
            )
        else:
            tCrA = tiled_mma.make_fragment_A(sATransformed)

        k_tile_cnt_fc1 = (
            fc1_weight_gemm.shape[1] + self.mma_tiler[2] - 1
        ) // self.mma_tiler[2]
        k_tile_cnt_fc2 = (
            fc2_weight_gemm.shape[1] + self.mma_tiler[2] - 1
        ) // self.mma_tiler[2]

        if warp_idx == self.sched_warp_id:
            if cutlass.const_expr(self.enable_token_comm):
                self.token_comm_hook_sched_warp_pre_init_wait(
                    token_comm_args
                )
            if cutlass.const_expr(not early_internal_init):
                scheduler.internal_init(
                    warp_idx=warp_idx,
                    sched_warp_id=self.sched_warp_id,
                )
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                ext.prefetch_for_expert(
                    scheduler.current_work.expert_idx
                )
                scheduler.publish_work()
                scheduler.gen_next_work()
            scheduler.publish_work()
            scheduler.produce_tail()

        # Weight/data and its E8M0 plane are issued by one warp.  With the
        # fixed K128 tile there is one staged 32x4x4 SF atom per K tile, so the
        # two producer states advance in lockstep without the K64 shared-atom
        # lifetime special case.
        if warp_idx == self.tma_a_warp_id:
            tma_a_iket_active = (
                tidx == cutlass.Int32(32 * self.tma_a_warp_id)
            )
            a_mcast_mask = None
            if cutlass.const_expr(
                self.is_a_mcast or self.use_2cta_instrs
            ):
                a_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=2,
                )
            a_cta_layout = cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk, (0, 0, None, 0)
                ).shape
            )
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            thr_mma_leader = tiled_mma.get_slice(0)

            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_a_stage
            )
            scale_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_scale_stage,
            )
            work = sched_consumer.consume_work()
            while work.is_valid_tile:
                is_fc1 = work.phase == cutlass.Int32(BlockPhase.Linear1)
                phase_payload = cutlass.Int32(is_fc1)
                if tma_a_iket_active:
                    iket.range_push(
                        "mixed_role_tile_lifetime",
                        cutlass.Int32(10) + phase_payload,
                    )
                    iket.range_push(
                        "mixed_role_coverage",
                        cutlass.Int32(101)
                        + phase_payload * cutlass.Int32(10),
                    )
                if is_fc1:
                    k_tile_cnt = k_tile_cnt_fc1
                    real_a, desc_a = ext.get_gmem_tensor(
                        "a", tma_tensor_w1, work
                    )
                    real_scale, desc_scale = ext.get_gmem_tensor(
                        "sfa", tma_tensor_w1_scale, work
                    )
                    gA = cute.local_tile(
                        real_a,
                        cute.slice_(
                            self.tma_tiler, (None, 0, None)
                        ),
                        (None, None, None),
                    )
                    gScale = cute.local_tile(
                        real_scale,
                        (
                            self.mma_tiler[0],
                            self.scale_tile_shape[1],
                        ),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA)
                    tCgScale = thr_mma.partition_A(gScale)
                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_w1,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tCsScale = thr_mma_leader.partition_A(sScale)
                    tSsScale, tSgScale = (
                        mixed_input_utils.scale_tma_partition(
                            tCsScale,
                            tCgScale,
                            tma_atom_w1_scale,
                            block_in_cluster_coord_vmnk,
                            a_cta_layout,
                        )
                    )
                    mma_tile_m = work.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[
                        (None, mma_tile_m, None, 0)
                    ]
                    tSgScale_slice = tSgScale[
                        (None, mma_tile_m, None, 0)
                    ]
                    rest = cute.filter_zeros(
                        tSgScale_slice[(0, None)].layout
                    )
                    tSgScale_filtered = cute.make_tensor(
                        tSgScale_slice.iterator,
                        cute.make_layout(
                            (
                                tSgScale_slice.layout[0].shape,
                                rest.shape,
                            ),
                            stride=(
                                tSgScale_slice.layout[0].stride,
                                rest.stride,
                            ),
                        ),
                    )

                    a_state.reset_count()
                    scale_state.reset_count()
                    peek_a = a_load_pipeline.producer_try_acquire(
                        a_state
                    )
                    peek_scale = (
                        scale_load_pipeline.producer_try_acquire(
                            scale_state
                        )
                    )
                    if tma_a_iket_active:
                        iket.range_pop()
                    scale_reuse = (
                        2
                        if cutlass.const_expr(self.mma_tiler[2] == 64)
                        else 1
                    )
                    scale_tile_cnt = k_tile_cnt // scale_reuse
                    for _scale_k in cutlass.range(
                        0, scale_tile_cnt, 1, unroll=1
                    ):
                        scale_load_pipeline.producer_acquire(
                            scale_state, peek_scale
                        )
                        cute.copy(
                            tma_atom_w1_scale,
                            tSgScale_filtered[(None, scale_state.count)],
                            tSsScale[(None, scale_state.index)],
                            tma_bar_ptr=(
                                scale_load_pipeline.producer_get_barrier(
                                    scale_state
                                )
                            ),
                            tma_desc_ptr=desc_scale,
                            mcast_mask=a_mcast_mask,
                        )
                        scale_state.advance()
                        peek_scale = cutlass.Boolean(1)
                        if scale_state.count < scale_tile_cnt:
                            peek_scale = (
                                scale_load_pipeline.producer_try_acquire(
                                    scale_state
                                )
                            )
                        if cutlass.const_expr(self.tma_k_reuse == 2):
                            a_load_pipeline.producer_acquire(a_state, peek_a)
                            cute.copy(
                                tma_atom_w1,
                                tAgA_slice[(None, a_state.count)],
                                tAsA[(None, a_state.index)],
                                tma_bar_ptr=(
                                    a_load_pipeline.producer_get_barrier(
                                        a_state
                                    )
                                ),
                                tma_desc_ptr=desc_a,
                                mcast_mask=a_mcast_mask,
                            )
                            a_load_pipeline.producer_commit(a_state)
                            a_state.advance()
                            peek_a = cutlass.Boolean(1)
                            if a_state.count < scale_tile_cnt:
                                peek_a = (
                                    a_load_pipeline.producer_try_acquire(
                                        a_state
                                    )
                                )
                        for _scale_half in cutlass.range_constexpr(
                            scale_reuse
                        ):
                            if tma_a_iket_active:
                                iket.range_push(
                                    "mixed_role_coverage",
                                    cutlass.Int32(102)
                                    + phase_payload * cutlass.Int32(10),
                                )
                                iket.range_push(
                                    "mixed_tma_a_producer_acquire",
                                    phase_payload,
                                )
                            if cutlass.const_expr(self.tma_k_reuse == 1):
                                a_load_pipeline.producer_acquire(
                                    a_state, peek_a
                                )
                            if tma_a_iket_active:
                                iket.range_pop()
                                iket.range_push(
                                    "mixed_tma_a_stage_work", phase_payload
                                )
                            if cutlass.const_expr(self.tma_k_reuse == 1):
                                cute.copy(
                                    tma_atom_w1,
                                    tAgA_slice[(None, a_state.count)],
                                    tAsA[(None, a_state.index)],
                                    tma_bar_ptr=(
                                        a_load_pipeline.producer_get_barrier(
                                            a_state
                                        )
                                    ),
                                    tma_desc_ptr=desc_a,
                                    mcast_mask=a_mcast_mask,
                                )
                                a_load_pipeline.producer_commit(a_state)
                                a_state.advance()
                                peek_a = cutlass.Boolean(1)
                                if a_state.count < k_tile_cnt:
                                    peek_a = (
                                        a_load_pipeline.producer_try_acquire(
                                            a_state
                                        )
                                    )
                            if tma_a_iket_active:
                                iket.range_pop()
                                iket.range_pop()
                else:
                    k_tile_cnt = k_tile_cnt_fc2
                    real_a, desc_a = ext.get_gmem_tensor(
                        "a", tma_tensor_w2, work
                    )
                    real_scale, desc_scale = ext.get_gmem_tensor(
                        "sfa", tma_tensor_w2_scale, work
                    )
                    gA = cute.local_tile(
                        real_a,
                        cute.slice_(
                            self.tma_tiler, (None, 0, None)
                        ),
                        (None, None, None),
                    )
                    gScale = cute.local_tile(
                        real_scale,
                        (
                            self.mma_tiler[0],
                            self.scale_tile_shape[1],
                        ),
                        (None, None, None),
                    )
                    tCgA = thr_mma.partition_A(gA)
                    tCgScale = thr_mma.partition_A(gScale)
                    tAsA, tAgA = cpasync.tma_partition(
                        tma_atom_w2,
                        block_in_cluster_coord_vmnk[2],
                        a_cta_layout,
                        cute.group_modes(sA, 0, 3),
                        cute.group_modes(tCgA, 0, 3),
                    )
                    tCsScale = thr_mma_leader.partition_A(sScale)
                    tSsScale, tSgScale = (
                        mixed_input_utils.scale_tma_partition(
                            tCsScale,
                            tCgScale,
                            tma_atom_w2_scale,
                            block_in_cluster_coord_vmnk,
                            a_cta_layout,
                        )
                    )
                    mma_tile_m = work.tile_m_idx // cute.size(
                        tiled_mma.thr_id.shape
                    )
                    tAgA_slice = tAgA[
                        (None, mma_tile_m, None, 0)
                    ]
                    tSgScale_slice = tSgScale[
                        (None, mma_tile_m, None, 0)
                    ]
                    rest = cute.filter_zeros(
                        tSgScale_slice[(0, None)].layout
                    )
                    tSgScale_filtered = cute.make_tensor(
                        tSgScale_slice.iterator,
                        cute.make_layout(
                            (
                                tSgScale_slice.layout[0].shape,
                                rest.shape,
                            ),
                            stride=(
                                tSgScale_slice.layout[0].stride,
                                rest.stride,
                            ),
                        ),
                    )

                    a_state.reset_count()
                    scale_state.reset_count()
                    peek_a = a_load_pipeline.producer_try_acquire(
                        a_state
                    )
                    peek_scale = (
                        scale_load_pipeline.producer_try_acquire(
                            scale_state
                        )
                    )
                    if tma_a_iket_active:
                        iket.range_pop()
                    scale_reuse = (
                        2
                        if cutlass.const_expr(self.mma_tiler[2] == 64)
                        else 1
                    )
                    scale_tile_cnt = k_tile_cnt // scale_reuse
                    for _scale_k in cutlass.range(
                        0, scale_tile_cnt, 1, unroll=1
                    ):
                        scale_load_pipeline.producer_acquire(
                            scale_state, peek_scale
                        )
                        cute.copy(
                            tma_atom_w2_scale,
                            tSgScale_filtered[(None, scale_state.count)],
                            tSsScale[(None, scale_state.index)],
                            tma_bar_ptr=(
                                scale_load_pipeline.producer_get_barrier(
                                    scale_state
                                )
                            ),
                            tma_desc_ptr=desc_scale,
                            mcast_mask=a_mcast_mask,
                        )
                        scale_state.advance()
                        peek_scale = cutlass.Boolean(1)
                        if scale_state.count < scale_tile_cnt:
                            peek_scale = (
                                scale_load_pipeline.producer_try_acquire(
                                    scale_state
                                )
                            )
                        if cutlass.const_expr(self.tma_k_reuse == 2):
                            a_load_pipeline.producer_acquire(a_state, peek_a)
                            cute.copy(
                                tma_atom_w2,
                                tAgA_slice[(None, a_state.count)],
                                tAsA[(None, a_state.index)],
                                tma_bar_ptr=(
                                    a_load_pipeline.producer_get_barrier(
                                        a_state
                                    )
                                ),
                                tma_desc_ptr=desc_a,
                                mcast_mask=a_mcast_mask,
                            )
                            a_load_pipeline.producer_commit(a_state)
                            a_state.advance()
                            peek_a = cutlass.Boolean(1)
                            if a_state.count < scale_tile_cnt:
                                peek_a = (
                                    a_load_pipeline.producer_try_acquire(
                                        a_state
                                    )
                                )
                        for _scale_half in cutlass.range_constexpr(
                            scale_reuse
                        ):
                            if tma_a_iket_active:
                                iket.range_push(
                                    "mixed_role_coverage",
                                    cutlass.Int32(102)
                                    + phase_payload * cutlass.Int32(10),
                                )
                                iket.range_push(
                                    "mixed_tma_a_producer_acquire",
                                    phase_payload,
                                )
                            if cutlass.const_expr(self.tma_k_reuse == 1):
                                a_load_pipeline.producer_acquire(
                                    a_state, peek_a
                                )
                            if tma_a_iket_active:
                                iket.range_pop()
                                iket.range_push(
                                    "mixed_tma_a_stage_work", phase_payload
                                )
                            if cutlass.const_expr(self.tma_k_reuse == 1):
                                cute.copy(
                                    tma_atom_w2,
                                    tAgA_slice[(None, a_state.count)],
                                    tAsA[(None, a_state.index)],
                                    tma_bar_ptr=(
                                        a_load_pipeline.producer_get_barrier(
                                            a_state
                                        )
                                    ),
                                    tma_desc_ptr=desc_a,
                                    mcast_mask=a_mcast_mask,
                                )
                                a_load_pipeline.producer_commit(a_state)
                                a_state.advance()
                                peek_a = cutlass.Boolean(1)
                                if a_state.count < k_tile_cnt:
                                    peek_a = (
                                        a_load_pipeline.producer_try_acquire(
                                            a_state
                                        )
                                    )
                            if tma_a_iket_active:
                                iket.range_pop()
                                iket.range_pop()
                if tma_a_iket_active:
                    iket.range_pop()
                work = sched_consumer.consume_work()
            a_load_pipeline.producer_tail(a_state)
            scale_load_pipeline.producer_tail(scale_state)

        if warp_idx == self.tma_b_warp_id:
            tma_b_iket_active = (
                tidx == cutlass.Int32(32 * self.tma_b_warp_id)
            )
            b_mcast_mask = None
            if cutlass.const_expr(
                self.is_b_mcast or self.use_2cta_instrs
            ):
                b_mcast_mask = cpasync.create_tma_multicast_mask(
                    cluster_layout_vmnk,
                    block_in_cluster_coord_vmnk,
                    mcast_mode=1,
                )
            b_cta_layout = cute.make_layout(
                cute.slice_(
                    cluster_layout_vmnk, (0, None, 0, 0)
                ).shape
            )
            thr_mma = tiled_mma.get_slice(mma_tile_coord_v)
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_b_stage
            )
            work = sched_consumer.consume_work()
            while work.is_valid_tile:
                is_fc1 = work.phase == cutlass.Int32(BlockPhase.Linear1)
                phase_payload = cutlass.Int32(is_fc1)
                if tma_b_iket_active:
                    iket.range_push(
                        "mixed_role_tile_lifetime",
                        cutlass.Int32(20) + phase_payload,
                    )
                    iket.range_push(
                        "mixed_role_coverage",
                        cutlass.Int32(201)
                        + phase_payload * cutlass.Int32(10),
                    )
                if is_fc1:
                    if cutlass.const_expr(self.enable_token_comm):
                        self.token_comm_hook_fc1_tma_b_predispatch_spin(
                            token_comm_args,
                            work,
                        )
                    k_tile_cnt = k_tile_cnt_fc1
                    real_b, desc_b = ext.get_gmem_tensor(
                        "b", tma_tensor_activation, work
                    )
                    gB = cute.local_tile(
                        real_b,
                        cute.slice_(
                            self.tma_tiler, (0, None, None)
                        ),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB)
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_activation,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBgB_slice = tBgB[
                        (None, work.tile_n_idx, None, 0)
                    ]
                    b_state.reset_count()
                    peek_b = b_load_pipeline.producer_try_acquire(
                        b_state
                    )
                    if tma_b_iket_active:
                        iket.range_pop()
                    b_tile_cnt = k_tile_cnt // self.tma_k_reuse
                    for _k in cutlass.range(
                        0, b_tile_cnt, 1, unroll=1
                    ):
                        if tma_b_iket_active:
                            iket.range_push(
                                "mixed_role_coverage",
                                cutlass.Int32(202)
                                + phase_payload * cutlass.Int32(10),
                            )
                        if tma_b_iket_active:
                            iket.range_push(
                                "mixed_tma_b_producer_acquire",
                                phase_payload,
                            )
                        b_load_pipeline.producer_acquire(
                            b_state, peek_b
                        )
                        if tma_b_iket_active:
                            iket.range_pop()
                            iket.range_push(
                                "mixed_tma_b_stage_work", phase_payload
                            )
                        cute.copy(
                            tma_atom_activation,
                            tBgB_slice[(None, b_state.count)],
                            tBsB[(None, b_state.index)],
                            tma_bar_ptr=(
                                b_load_pipeline.producer_get_barrier(
                                    b_state
                                )
                            ),
                            tma_desc_ptr=desc_b,
                            mcast_mask=b_mcast_mask,
                        )
                        b_load_pipeline.producer_commit(b_state)
                        b_state.advance()
                        peek_b = cutlass.Boolean(1)
                        if b_state.count < b_tile_cnt:
                            peek_b = (
                                b_load_pipeline.producer_try_acquire(
                                    b_state
                                )
                            )
                        if tma_b_iket_active:
                            iket.range_pop()
                            iket.range_pop()
                else:
                    counter_slot = (
                        work.cumulative_token_block_count
                        + work.tile_n_idx
                    )
                    counter_ptr = (
                        fc1_done_counter.iterator + counter_slot
                    )
                    if not work.peek_ready:
                        if tma_b_iket_active:
                            iket.range_push(
                                "mixed_fc2_tma_b_consumer_wait"
                            )
                        spin_wait(
                            counter_ptr,
                            lambda value: (
                                value >= ext_fc2_spin_threshold
                            ),
                            fail_sleep_cycles=500,
                        )
                        if tma_b_iket_active:
                            iket.range_pop()
                    k_tile_cnt = k_tile_cnt_fc2
                    real_b, desc_b = ext.get_gmem_tensor(
                        "b", tma_tensor_fc1_output, work
                    )
                    gB = cute.local_tile(
                        real_b,
                        cute.slice_(
                            self.tma_tiler, (0, None, None)
                        ),
                        (None, None, None),
                    )
                    tCgB = thr_mma.partition_B(gB)
                    tBsB, tBgB = cpasync.tma_partition(
                        tma_atom_fc1_output,
                        block_in_cluster_coord_vmnk[1],
                        b_cta_layout,
                        cute.group_modes(sB, 0, 3),
                        cute.group_modes(tCgB, 0, 3),
                    )
                    tBgB_slice = tBgB[
                        (None, work.tile_n_idx, None, 0)
                    ]
                    b_state.reset_count()
                    peek_b = b_load_pipeline.producer_try_acquire(
                        b_state
                    )
                    if tma_b_iket_active:
                        iket.range_pop()
                    b_tile_cnt = k_tile_cnt // self.tma_k_reuse
                    for _k in cutlass.range(
                        0, b_tile_cnt, 1, unroll=1
                    ):
                        if tma_b_iket_active:
                            iket.range_push(
                                "mixed_role_coverage",
                                cutlass.Int32(202)
                                + phase_payload * cutlass.Int32(10),
                            )
                        if tma_b_iket_active:
                            iket.range_push(
                                "mixed_tma_b_producer_acquire",
                                phase_payload,
                            )
                        b_load_pipeline.producer_acquire(
                            b_state, peek_b
                        )
                        if tma_b_iket_active:
                            iket.range_pop()
                            iket.range_push(
                                "mixed_tma_b_stage_work", phase_payload
                            )
                        cute.copy(
                            tma_atom_fc1_output,
                            tBgB_slice[(None, b_state.count)],
                            tBsB[(None, b_state.index)],
                            tma_bar_ptr=(
                                b_load_pipeline.producer_get_barrier(
                                    b_state
                                )
                            ),
                            tma_desc_ptr=desc_b,
                            mcast_mask=b_mcast_mask,
                        )
                        b_load_pipeline.producer_commit(b_state)
                        b_state.advance()
                        peek_b = cutlass.Boolean(1)
                        if b_state.count < b_tile_cnt:
                            peek_b = (
                                b_load_pipeline.producer_try_acquire(
                                    b_state
                                )
                            )
                        if tma_b_iket_active:
                            iket.range_pop()
                            iket.range_pop()
                if tma_b_iket_active:
                    iket.range_pop()
                work = sched_consumer.consume_work()
            b_load_pipeline.producer_tail(b_state)

        # Use an explicit closed-open range: in MegaMoE mode dispatch occupies
        # warps 8-11 and only transform warps 12-15 may enter this body.
        if (warp_idx >= self.transform_warp_id[0]) and (
            warp_idx
            < self.transform_warp_id[0] + len(self.transform_warp_id)
        ):
            transform_local_tidx = (
                tidx - 32 * self.transform_warp_id[0]
            )
            transform_iket_active = (
                transform_local_tidx == cutlass.Int32(0)
            )
            copy_atom_a_input = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.a_dtype,
                num_bits_per_copy=32,
            )
            a_smem_shape = tiled_mma.partition_shape_A(
                cute.dice(self.mma_tiler, (1, None, 1))
            )
            copy_atom_a_transform = (
                mixed_input_utils.get_copy_atom_a_transform(
                    cutlass.BFloat16,
                    self.use_2cta_instrs,
                    self.transform_a_source,
                    a_smem_shape,
                    self.a_dtype,
                )
            )
            (
                src_copy_a,
                dst_copy_a,
                tAsA_input,
                tAsA_transform,
            ) = mixed_input_utils.transform_partition(
                self.transform_a_source,
                TransformMode.ConvertScale,
                copy_atom_a_input,
                copy_atom_a_transform,
                sA,
                (
                    tCrA
                    if self.transform_a_source == tcgen05.OperandSource.TMEM
                    else sATransformed
                ),
                transform_local_tidx,
            )
            tArA = cute.make_rmem_tensor(
                tAsA_input[(None, None, None, None, 0)].shape,
                self.a_dtype,
            )
            transform_rmem_shape = tAsA_input[
                (None, None, None, None, 0)
            ].shape
            if cutlass.const_expr(self.tma_k_reuse == 2):
                transform_rmem_shape = (
                    transform_rmem_shape[0],
                    transform_rmem_shape[1],
                    transform_rmem_shape[2],
                    transform_rmem_shape[3] // 2,
                )
            tArA_transform = cute.make_rmem_tensor(
                transform_rmem_shape, cutlass.BFloat16
            )

            thr_mma_leader = tiled_mma.get_slice(0)
            tCsScale = thr_mma_leader.partition_A(sScale)
            (
                _scale_copy,
                tSsScale_transform,
                tSrScale_copy,
                tSrScale,
            ) = _scale_partition_mxfp8(
                src_copy_a, tCsScale, transform_local_tidx
            )
            assert cute.size(tSrScale, mode=[0]) == cute.size(
                tArA, mode=[0]
            )
            assert cute.size(tSrScale) == cute.size(tArA)

            transform_tiler_size = min(
                cute.size(
                    cute.coalesce(tAsA_input.layout), mode=[0]
                ),
                64,
            )
            transform_tiler = cute.make_layout(
                transform_tiler_size
            )
            tArA_load = cute.flat_divide(tArA, transform_tiler)
            tArA_load = cute.group_modes(
                tArA_load, 1, cute.rank(tArA_load)
            )
            tSrScale_load = cute.flat_divide(
                tSrScale, transform_tiler
            )
            tSrScale_load = cute.group_modes(
                tSrScale_load, 1, cute.rank(tSrScale_load)
            )
            tArA_transform_store = cute.flat_divide(
                tArA_transform, transform_tiler
            )
            tArA_transform_store = cute.group_modes(
                tArA_transform_store,
                1,
                cute.rank(tArA_transform_store),
            )

            a_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_a_stage
            )
            scale_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_scale_stage,
            )
            transformed_producer_state = (
                pipeline.make_pipeline_state(
                    pipeline.PipelineUserType.Producer,
                    self.num_transformed_a_stage,
                )
            )
            work = sched_consumer.consume_work()
            while work.is_valid_tile:
                is_fc1 = work.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                phase_payload = cutlass.Int32(is_fc1)
                if transform_iket_active:
                    iket.range_push(
                        "mixed_role_tile_lifetime",
                        cutlass.Int32(30) + phase_payload,
                    )
                    iket.range_push(
                        "mixed_role_coverage",
                        cutlass.Int32(301)
                        + phase_payload * cutlass.Int32(10),
                    )
                if cutlass.const_expr(
                    self.static_expert_shape is not None
                ):
                    # The K bounds remain phase-specific Python ints. Clone
                    # every mutable PipelineState inside its runtime branch so
                    # tracing one @cute.jit specialization cannot leak
                    # branch-local SSA into the sibling specialization.
                    if is_fc1:
                        fc1_a_consumer_state = a_consumer_state.clone()
                        fc1_scale_consumer_state = (
                            scale_consumer_state.clone()
                        )
                        fc1_transformed_producer_state = (
                            transformed_producer_state.clone()
                        )
                        (
                            a_consumer_state,
                            scale_consumer_state,
                            transformed_producer_state,
                        ) = self._transform_k_tiles(
                            k_tile_cnt_fc1,
                            is_fc1,
                            transform_iket_active,
                            a_load_pipeline,
                            scale_load_pipeline,
                            transformed_a_pipeline,
                            fc1_a_consumer_state,
                            fc1_scale_consumer_state,
                            fc1_transformed_producer_state,
                            tAsA_input,
                            transform_tiler,
                            tSsScale_transform,
                            tSrScale_copy,
                            tArA_load,
                            tSrScale_load,
                            tArA_transform,
                            tArA_transform_store,
                            tAsA_transform,
                            dst_copy_a,
                        )
                    else:
                        fc2_a_consumer_state = a_consumer_state.clone()
                        fc2_scale_consumer_state = (
                            scale_consumer_state.clone()
                        )
                        fc2_transformed_producer_state = (
                            transformed_producer_state.clone()
                        )
                        (
                            a_consumer_state,
                            scale_consumer_state,
                            transformed_producer_state,
                        ) = self._transform_k_tiles(
                            k_tile_cnt_fc2,
                            is_fc1,
                            transform_iket_active,
                            a_load_pipeline,
                            scale_load_pipeline,
                            transformed_a_pipeline,
                            fc2_a_consumer_state,
                            fc2_scale_consumer_state,
                            fc2_transformed_producer_state,
                            tAsA_input,
                            transform_tiler,
                            tSsScale_transform,
                            tSrScale_copy,
                            tArA_load,
                            tSrScale_load,
                            tArA_transform,
                            tArA_transform_store,
                            tAsA_transform,
                            dst_copy_a,
                        )
                else:
                    k_tile_cnt = cutlass.Int32(0)
                    if is_fc1:
                        k_tile_cnt = k_tile_cnt_fc1
                    else:
                        k_tile_cnt = k_tile_cnt_fc2
                    (
                        a_consumer_state,
                        scale_consumer_state,
                        transformed_producer_state,
                    ) = self._transform_k_tiles(
                        k_tile_cnt,
                        is_fc1,
                        transform_iket_active,
                        a_load_pipeline,
                        scale_load_pipeline,
                        transformed_a_pipeline,
                        a_consumer_state,
                        scale_consumer_state,
                        transformed_producer_state,
                        tAsA_input,
                        transform_tiler,
                        tSsScale_transform,
                        tSrScale_copy,
                        tArA_load,
                        tSrScale_load,
                        tArA_transform,
                        tArA_transform_store,
                        tAsA_transform,
                        dst_copy_a,
                    )
                if transform_iket_active:
                    iket.range_pop()
                work = sched_consumer.consume_work()
            transformed_a_pipeline.producer_tail(
                transformed_producer_state
            )

        if warp_idx == self.mma_warp_id:
            mma_iket_active = (
                tidx
                == cutlass.Int32(32 * self.mma_warp_id)
            )
            transformed_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer,
                self.num_transformed_a_stage,
            )
            b_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_b_stage
            )
            acc_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer,
                self.num_acc_pipeline_stages,
            )
            work = sched_consumer.consume_work()
            while work.is_valid_tile:
                is_fc1 = work.phase == cutlass.Int32(
                    BlockPhase.Linear1
                )
                phase_payload = cutlass.Int32(is_fc1)
                if is_leader_cta:
                    if mma_iket_active:
                        iket.range_push(
                            "mixed_role_tile_lifetime",
                            cutlass.Int32(40) + phase_payload,
                        )
                        iket.range_push(
                            "mixed_role_coverage",
                            cutlass.Int32(401)
                            + phase_payload * cutlass.Int32(10),
                        )
                if is_leader_cta:
                    if cutlass.const_expr(self.overlapping_accum):
                        acc_stage_idx = acc_producer_state.phase ^ 1
                    else:
                        acc_stage_idx = acc_producer_state.index
                    tCtAcc = accumulators[
                        (
                            None,
                            None,
                            None,
                            acc_stage_idx,
                        )
                    ]
                    if cutlass.const_expr(
                        self.static_expert_shape is not None
                    ):
                        # TiledMma.set mutates the wrapper's trait value. Give
                        # each runtime branch a fresh wrapper and fresh
                        # consumer states, then explicitly rebind the values
                        # returned by that helper specialization.
                        if is_fc1:
                            fc1_tiled_mma = tiled_mma.with_()
                            fc1_transformed_consumer_state = (
                                transformed_consumer_state.clone()
                            )
                            fc1_b_consumer_state = (
                                b_consumer_state.clone()
                            )
                            (
                                tiled_mma,
                                transformed_consumer_state,
                                b_consumer_state,
                            ) = self._mma_k_tiles(
                                k_tile_cnt_fc1,
                                is_fc1,
                                mma_iket_active,
                                fc1_tiled_mma,
                                tCtAcc,
                                tCrA,
                                tCrB,
                                transformed_a_pipeline,
                                b_load_pipeline,
                                acc_pipeline,
                                fc1_transformed_consumer_state,
                                fc1_b_consumer_state,
                                acc_producer_state,
                            )
                        else:
                            fc2_tiled_mma = tiled_mma.with_()
                            fc2_transformed_consumer_state = (
                                transformed_consumer_state.clone()
                            )
                            fc2_b_consumer_state = (
                                b_consumer_state.clone()
                            )
                            (
                                tiled_mma,
                                transformed_consumer_state,
                                b_consumer_state,
                            ) = self._mma_k_tiles(
                                k_tile_cnt_fc2,
                                is_fc1,
                                mma_iket_active,
                                fc2_tiled_mma,
                                tCtAcc,
                                tCrA,
                                tCrB,
                                transformed_a_pipeline,
                                b_load_pipeline,
                                acc_pipeline,
                                fc2_transformed_consumer_state,
                                fc2_b_consumer_state,
                                acc_producer_state,
                            )
                    else:
                        k_tile_cnt = cutlass.Int32(0)
                        if is_fc1:
                            k_tile_cnt = k_tile_cnt_fc1
                        else:
                            k_tile_cnt = k_tile_cnt_fc2
                        (
                            tiled_mma,
                            transformed_consumer_state,
                            b_consumer_state,
                        ) = self._mma_k_tiles(
                            k_tile_cnt,
                            is_fc1,
                            mma_iket_active,
                            tiled_mma,
                            tCtAcc,
                            tCrA,
                            tCrB,
                            transformed_a_pipeline,
                            b_load_pipeline,
                            acc_pipeline,
                            transformed_consumer_state,
                            b_consumer_state,
                            acc_producer_state,
                        )
                if is_leader_cta:
                    if mma_iket_active:
                        iket.range_push(
                            "mixed_role_coverage",
                            cutlass.Int32(403)
                            + phase_payload * cutlass.Int32(10),
                        )
                acc_producer_state.advance()
                if is_leader_cta:
                    if mma_iket_active:
                        iket.range_pop()
                        iket.range_pop()
                work = sched_consumer.consume_work()
            acc_pipeline.producer_tail(acc_producer_state)

        if warp_idx < self.mma_warp_id:
            optional_epi_args = NvFp4OptinalEpiArgs(
                fc1_alpha=None,
                fc2_alpha=None,
                fc1_norm_const=None,
                topk_scores=(
                    topk_scores
                    if cutlass.const_expr(self.apply_topk_in_fc1)
                    else None
                ),
            )
            run_kwargs = dict(
                epi_smem_storage=epi_smem_storage,
                tmem_ptr=tmem_ptr,
                acc_pipeline=acc_pipeline,
                sched_consumer=sched_consumer,
                sched_ext=ext,
                fc1_output=fc1_output_gemm,
                fc2_output=fc2_output,
                fc1_done_counter=fc1_done_counter,
                tidx=tidx,
                optional_epi_args=optional_epi_args,
            )
            # Passing Python ``None`` explicitly into a ``@cute.jit`` method
            # can crash native tracing.  Omit the optional argument for lean;
            # only the concrete MegaMoE bundle crosses this boundary.
            if cutlass.const_expr(token_comm_args is not None):
                self.epilogue.run(
                    **run_kwargs,
                    token_comm_args=token_comm_args,
                )
            else:
                self.epilogue.run(**run_kwargs)
            tmem.relinquish_alloc_permit()
            self.epilog_sync_barrier.arrive_and_wait()
            tmem.free(tmem_ptr)
            if cutlass.const_expr(self.enable_token_comm):
                cute.arch.fence_acq_rel_sys()

        # Dispatch warps execute concurrently with the mixed mainloop.  The
        # explicit range prevents the relocated transform warpgroup (12-15)
        # from entering the dispatch body.
        if cutlass.const_expr(self.enable_token_comm):
            if (warp_idx >= self.dispatch_warp_id[0]) and (
                warp_idx
                < self.dispatch_warp_id[0] + len(self.dispatch_warp_id)
            ):
                self.token_comm_hook_dispatch_warp_body(
                    token_comm_args,
                    token_comm_storage,
                    warp_idx=warp_idx,
                    lane_idx=cute.arch.lane_idx(),
                    tidx=tidx,
                )

            # All 16 warps reach the subclass-owned final rendezvous/release.
            self.token_comm_hook_kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=cute.arch.lane_idx(),
                tidx=tidx,
            )


__all__ = ["Sm100SwapABMxfp8Bf16Fc12Kernel"]
