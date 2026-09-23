# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Packed NVFP4 weights decoded into operand-A TMEM inside fused MegaMoE.

The local W4A16 helpers load packed weights and block scales, then decode
BF16 tiles directly into the TMEM operand pipeline. Both GEMMs use
dynamic routed-token widths with M128/M256, N64/N128 and K256 allocation.
Static and atomic schedulers fuse BF16 dispatch, both GEMMs, and combine.
"""

import os
from typing import Any, List, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.utils import mixed_input_helpers as mixed_input_utils
from cutlass.utils import blockscaled_layout as blockscaled_utils

from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import (
    Sm100W4A16GroupedGemmKernel,
)
from .workspace import _RegionSpec, _layout_regions, _round_up
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import get_cutedsl_target_arch
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
    CombineFormat,
    TokenSrcMetadata,
)
from .custom_ext import W4A16Fc12SchedExtension
from .fc1_fc2_fuse_sched import BlockPhase, MoEFusedFc12SchedulerParams
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import spin_wait
from .token_comm import W4A16TokenComm
from flashinfer.moe_ep.kernel_src.sm100.cutedsl_megamoe import (
    TokenCommArgs as ExtractedTokenCommArgs,
)

from .epilogue import W4A16Epilogue, W4A16EpiArgs
from .topk_reduce import Bf16TopkReduce
from . import dynamic_mainloop

# Communication ABI: one packed i64 provenance record and two signal slots.
_TokenMetadataBytes = TokenSrcMetadata.nbytes
_GridSyncSlotCount = 2
_NvlinkSlotCount = 2


class _MegaMixedInput(Sm100W4A16GroupedGemmKernel):
    """Reuse local W4A16 layouts with Mega-fitted raw and decoded-TMEM stages."""

    def _compute_stages_and_tmem_cols(
        self,
        tiled_mma,
        mma_tiler_mnk,
        cta_tile_shape_mnk,
        epi_tile,
        a_dtype,
        b_dtype,
        c_dtype,
        c_layout,
        transform_a_source,
        smem_buffer_align_bytes,
        use_fused_finalize,
        use_clc_scheduler,
    ):
        assert mma_tiler_mnk in (
            (128, 64, 256),
            (128, 128, 256),
            (256, 64, 256),
            (256, 128, 256),
        )
        assert transform_a_source == tcgen05.OperandSource.TMEM
        acc_shape = tiled_mma.partition_shape_C(mma_tiler_mnk[:2])
        acc_one = tiled_mma.make_fragment_C(cute.append(acc_shape, 1))
        cols_per_acc = utils.get_num_tmem_alloc_cols(acc_one, True)
        cols_per_a = cute.round_up(cta_tile_shape_mnk[2] // 2, 4)
        assert cols_per_acc == mma_tiler_mnk[1] and cols_per_a == 128
        # Reuse the local W4A16 TMEM-capacity rule: after two accumulator
        # stages, N64 fits three decoded K256 tiles; N128 still fits two.
        max_tmem_cols = cute.arch.get_max_tmem_alloc_cols("sm_100")
        transform_stages = (max_tmem_cols - 2 * cols_per_acc) // cols_per_a
        assert transform_stages in (2, 3)
        # load, transform, acc, unused-C, unused-tile-info, ACC-cols, A-cols.
        return (
            self._raw_stage_count,
            transform_stages,
            2,
            1,
            1,
            2 * cols_per_acc,
            transform_stages * cols_per_a,
        )

    @cute.jit
    def _finish_transform_stage(
        self,
        a_load2trans_pipeline: pipeline.PipelineTmaAsync,
        trans2mma_pipeline: pipeline.PipelineAsyncUmma,
        cur_a_load2trans_consumer_state: pipeline.PipelineState,
        trans2mma_producer_state: pipeline.PipelineState,
    ) -> None:
        # All raw/SF reads precede the final warp-wide TMEM store issue.
        # Raw SMEM can return before those independent TMEM stores complete.
        a_load2trans_pipeline.consumer_release(cur_a_load2trans_consumer_state)
        cute.arch.fence_view_async_tmem_store()
        trans2mma_pipeline.producer_commit(trans2mma_producer_state)


class Sm100W4A16MegaMoEKernel:
    """BF16 dispatch and compute with online-decoded NVFP4 expert weights."""

    def __init__(
        self,
        *,
        mma_tiler_mnk,
        cluster_shape_mnk,
        use_2cta_instrs,
        group_hint,
        token_padding_block,
        static_expert_shape,
        world_size,
        num_topk,
        max_tokens_per_rank,
        hidden,
        load_balance_mode="static",
        force_static_sched=True,
        num_sched_stages=None,
        ab_dtype=cutlass.BFloat16,
        acc_dtype=cutlass.Float32,
        in_kernel_fc2_reduce=False,
        token_back_mode="epi_warps",
        apply_topk_in_fc1=False,
        gate_up_clamp=None,
        epi_flag_batch=(1, 1),
        flag_batch=1,
        combine_format=None,
        non_ubulk_fc2_store=True,
        scenario="2Dx3D",
        swiglu_alpha=None,
        swiglu_beta=None,
        situ_beta=None,
        situ_linear_beta=None,
    ):
        if not force_static_sched or scenario != "2Dx3D":
            raise ValueError("W4A16 requires forward 2Dx3D scheduler records.")
        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError("Unsupported W4A16 load_balance_mode.")
        self.apply_topk_in_fc1 = apply_topk_in_fc1
        if token_back_mode not in ("epi_warps", "reuse_dispatch_warps"):
            raise ValueError(
                "W4A16 supports epi_warps or reuse_dispatch_warps token return."
            )
        by_dispatch = token_back_mode == "reuse_dispatch_warps"
        self.in_kernel_fc2_reduce = in_kernel_fc2_reduce
        if self.in_kernel_fc2_reduce and not by_dispatch:
            raise ValueError("W4A16 in-kernel FC2 reduction requires dispatch return.")
        if mma_tiler_mnk not in (
            (128, 64, 256),
            (128, 128, 256),
            (256, 64, 256),
            (256, 128, 256),
        ):
            raise ValueError(
                "W4A16 MegaMoE requires mma_tiler_mnk=M128/M256, N64/N128, K256."
            )
        if cluster_shape_mnk != (2, 1, 1) and not (
            cluster_shape_mnk == (1, 1, 1) and mma_tiler_mnk == (128, 64, 256)
        ):
            raise ValueError(
                "W4A16 requires cluster (2,1,1), or (1,1,1) for M128/N64/K256."
            )
        if use_2cta_instrs != (mma_tiler_mnk[0] == 256):
            raise ValueError("W4A16 MMA M128/M256 requires one/two-CTA instructions.")
        if static_expert_shape is None or hidden != static_expert_shape[2]:
            raise ValueError("W4A16 requires static_expert_shape matching hidden.")
        _, gateup, _ = static_expert_shape
        if hidden % 32 or gateup % 128:
            raise ValueError("W4A16 requires H%32=0 and I%64=0.")
        if token_padding_block <= 0 or mma_tiler_mnk[1] % token_padding_block:
            raise ValueError("Token padding must divide the routed-token tile.")
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        if (
            ab_dtype is not cutlass.BFloat16
            or acc_dtype is not cutlass.Float32
            or combine_format.act_dtype is not cutlass.BFloat16
            or combine_format.is_quantized
            or not non_ubulk_fc2_store
        ):
            raise ValueError(
                "W4A16 requires BF16 activation/return and FP32 accumulation."
            )

        self.mma_tiler = mma_tiler_mnk
        self.cluster_shape_mn = cluster_shape_mnk[:2]
        self.use_2cta_instrs = use_2cta_instrs
        self.static_expert_shape = static_expert_shape
        self.num_sched_stages = num_sched_stages or 3
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.load_balance_mode = load_balance_mode
        self.scenario = scenario
        self.arch = get_cutedsl_target_arch()
        self.ab_dtype = cutlass.BFloat16
        self.gate_up_clamp = gate_up_clamp
        self.swiglu_alpha = swiglu_alpha
        self.swiglu_beta = swiglu_beta
        self.situ_beta = situ_beta
        self.situ_linear_beta = situ_linear_beta
        self.epi_flag_batch = epi_flag_batch
        self.flag_batch = flag_batch

        self.world_size = world_size
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.topk_reduce = Bf16TopkReduce(hidden, num_topk)
        self.num_experts_per_rank = static_expert_shape[0]
        self.num_total_experts = world_size * self.num_experts_per_rank
        self.intermediate_gateup = gateup
        self.intermediate_downproj = gateup // 2
        self.hidden_bytes = 2 * hidden
        self.cluster_tile_tokens = mma_tiler_mnk[1] * cluster_shape_mnk[1]
        self._fc1_k_tiles = (hidden + 255) // 256
        self._fc2_k_tiles = (gateup // 2 + 255) // 256

        # Five independent warpgroup roles, including exactly two decoders.
        self.num_transform_warpgroups = 2
        self.num_transform_warps = 8
        self.transform_warp_id = tuple(range(12, 20))
        self.threads_per_cta = 640
        self.tmem_alloc_sync_bar_id = 2
        self.token_back_mode = token_back_mode
        self.token_back_by_dispatch = by_dispatch
        self.token_back_schedule_mode = (
            self.load_balance_mode if by_dispatch else "static"
        )

        # Workspace and transport start in BF16 form. No FP4 activation
        # constructor or intermediate quantization state is instantiated.
        (
            self.pool_token_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()
        cluster_fc2_tile_hidden = (
            mma_tiler_mnk[0] * cluster_shape_mnk[0] // (2 if use_2cta_instrs else 1)
        )
        fc2_publishes_per_token_cluster_tile = (
            (hidden + cluster_fc2_tile_hidden - 1) // cluster_fc2_tile_hidden
        ) * cluster_shape_mnk[0]
        self.token_comm = W4A16TokenComm(
            world_size=world_size,
            num_topk=num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=hidden,
            fc1_token_dtype=cutlass.BFloat16,
            combine_format=combine_format,
            token_back_by_dispatch=by_dispatch,
            fc2_publishes_per_token_cluster_tile=fc2_publishes_per_token_cluster_tile,
            token_back_reduce_topk=self.in_kernel_fc2_reduce,
            token_back_standalone=False,
            sf_uint32_per_token=0,
            token_padding_block=token_padding_block,
            sf_padding_block=1,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=8,
            num_other_warps=16,
            flag_batch=flag_batch,
            is_swap_ab=True,
            token_back_schedule_mode=self.token_back_schedule_mode,
        )
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        self._local_region_by_name = {r.name: r for r in self._local_region_specs}
        self._shared_region_by_name = {r.name: r for r in self._shared_region_specs}
        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["expert_recv_count_bank1"]
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    def name(self):
        m, n, k = self.mma_tiler
        cm, cn = self.cluster_shape_mn
        return (
            f"megamoe_w4a16_{m}x{n}x{k}_cluster{cm}x{cn}_{self.load_balance_mode}"
            f"_expert{self.static_expert_shape}_group{self.group_hint}"
            f"_stages{self.num_sched_stages}"
            f"_return{self.token_back_mode}_epiflag{self.epi_flag_batch}"
            f"_clamp{self.gate_up_clamp}_ep{self.world_size}_topk{self.num_topk}"
            f"_tokens{self.max_tokens_per_rank}_flag{self.flag_batch}"
            f"_swiglua{self.swiglu_alpha}_swiglub{self.swiglu_beta}"
            f"_situb{self.situ_beta}_situlinb{self.situ_linear_beta}"
            + ("_ikr" if self.in_kernel_fc2_reduce else "")
            + ("_topk_fc1" if self.apply_topk_in_fc1 else "")
        )

    def _make_mixed(self, fragment_size, output_tensor, raw_stages, activation_stages):
        mixed = _MegaMixedInput(
            acc_dtype=cutlass.Float32,
            use_2cta_instrs=self.use_2cta_instrs,
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            group_count=self.num_experts_per_rank,
            activation_type=None,
            swiglu_alpha=1.0,
            swiglu_beta=0.0,
            swiglu_limit=float("inf"),
            situ_beta=None,
            situ_linear_beta=None,
            use_fused_finalize=False,
            enable_pdl=False,
            use_clc_scheduler=False,
            raster_along_m=True,
            transform_fragment_size=fragment_size,
            m_cluster_aligned=False,
        )
        mixed.a_dtype = cutlass.Float4E2M1FN
        mixed.a_scale_dtype = cutlass.Float8E4M3FN
        mixed.b_dtype = mixed.c_dtype = mixed.mma_dtype = cutlass.BFloat16
        mixed.a_major_mode = mixed.b_major_mode = tcgen05.OperandMajorMode.K
        mixed.c_layout = utils.LayoutEnum.from_tensor(output_tensor)
        mixed.num_transform_warpgroups = self.num_transform_warpgroups
        mixed.num_transform_warps = self.num_transform_warps
        mixed.transform_warp_id = self.transform_warp_id
        mixed._raw_stage_count = raw_stages
        mixed._setup_attributes()
        # The inherited layout helper shares a load count for A and B.
        # Rebuild B with its selected depth, independent of the raw and TMEM rings.
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            mixed.mma_dtype,
            mixed.a_major_mode,
            mixed.b_major_mode,
            mixed.acc_dtype,
            mixed.cta_group,
            mixed.mma_tiler[:2],
            mixed.transform_a_source,
        )
        _, _, mixed.smem_layout_b = mixed_input_utils.compute_smem_layout(
            tiled_mma,
            mixed.mma_tiler,
            mixed.a_dtype,
            mixed.b_dtype,
            activation_stages,
            mixed.num_trans2mma_stage,
        )
        return mixed

    @staticmethod
    def _make_shared_storage(
        sched_storage_cls, raw_stages, transform_stages, activation_stages
    ):
        @cute.struct
        class SharedStorage:
            raw_barriers: cute.struct.MemRange[cutlass.Int64, 2 * raw_stages]
            transform_barriers: cute.struct.MemRange[
                cutlass.Int64, 2 * transform_stages
            ]
            activation_barriers: cute.struct.MemRange[
                cutlass.Int64, 2 * activation_stages
            ]
            acc_barriers: cute.struct.MemRange[cutlass.Int64, 4]
            sched_storage: sched_storage_cls  # type: ignore[valid-type]
            tmem_dealloc: cutlass.Int64
            tmem_holding: cutlass.Int32

        return SharedStorage

    @staticmethod
    def _smem_size(shared_storage_cls, comm_storage_cls, mixed):
        # Keep this order and alignment identical to kernel()'s SmemAllocator.
        allocations = (
            (shared_storage_cls.__sizeof__(), shared_storage_cls.__alignof__()),
            (comm_storage_cls.__sizeof__(), comm_storage_cls.__alignof__()),
            (cute.size_in_bytes(cutlass.Float4E2M1FN, mixed.smem_layout_a), 128),
            (cute.size_in_bytes(cutlass.Float8E4M3FN, mixed.smem_layout_scale), 128),
            (cute.size_in_bytes(cutlass.BFloat16, mixed.smem_layout_b), 128),
        )
        size = 0
        for nbytes, alignment in allocations:
            size = (size + alignment - 1) // alignment * alignment + nbytes
        return size

    def _fit_raw_stages(self, output_tensor, sched_params):
        sched_storage_cls = sched_params.get_scheduler_type().make_storage_struct(
            sched_params, W4A16Fc12SchedExtension, num_drain_warps=0
        )
        comm_storage_cls = self.token_comm.extra_smem_storage_class()
        # CuTe's 227KiB per-block capacity already excludes CUDA's 1KiB
        # reservation from the SM's 228KiB. All user storage is counted below.
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        # Prefer three activation stages; preserve the two-stage fallback for
        # geometries where three cannot coexist with the minimum raw ring.
        for activation_stages in (3, 2):
            for raw_stages in range(5, 1, -1):
                mixed = self._make_mixed(
                    128, output_tensor, raw_stages, activation_stages
                )
                storage_cls = self._make_shared_storage(
                    sched_storage_cls,
                    raw_stages,
                    mixed.num_trans2mma_stage,
                    activation_stages,
                )
                if self._smem_size(storage_cls, comm_storage_cls, mixed) <= capacity:
                    return mixed, storage_cls, activation_stages
        raise ValueError("W4A16 MegaMoE cannot fit two raw stages in shared memory.")

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,  # (T, hidden) BF16
        topk_idx: cute.Tensor,  # (T, num_topk) Int64
        topk_weights: cute.Tensor,  # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc2_alpha: cute.Tensor,
        # Combine destination (peer write target via the epilogue Fc2OutputDest).
        combine_output: cute.Tensor,  # (T, 1 if in-kernel reduce else num_topk, H) BF16
        reduced_output: Optional[cute.Tensor],  # (active local T, H) BF16
        staging_inputs,  # optional source bundle; hidden strides=None means contiguous
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
        fc1_norm_const: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch the BF16 MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``topk_weights`` / ``combine_output`` MUST point
            into memory reachable via
            ``peer_rank_ptr_mapper.ptr_map_to_rank(...)`` (NVSHMEM symmetric
            heap).  Single-rank degenerate runs are allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc2_weight`` are local-only.

        ``combine_output`` holds BF16 ``(max_tokens_per_rank, num_topk, hidden)``
        partials for external reduction, or ``(max_tokens_per_rank, 1, hidden)``
        outputs for dispatch-return in-kernel reduction. Token communication
        maps each pool row back to its source rank and token.
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        hidden = self.hidden

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + BF16 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_bf16 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        l1_topk_weights_buffer = self._view_local(
            local_workspace,
            "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(local_workspace, "l1_arrival_count")
        # token_src_metadata storage = (pool_token_capacity, TokenSrcMetadata.nbytes) Uint8;
        # dispatch_pull writes one packed Int64 per pool token row (see TokenSrcMetadata).
        token_src_metadata = self._view_local(
            local_workspace,
            "token_src_metadata",
        )
        expert_send_count = self._view_local(local_workspace, "expert_send_count")
        grid_sync_counter = self._view_local(local_workspace, "grid_sync_counter")
        nvlink_barrier_counter = self._view_local(
            local_workspace,
            "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_done_counter = self._view_local(local_workspace, "fc1_done_counter")

        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace,
                "load_balance_counter",
            )

        token_back_schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace,
                "token_back_schedule_counter",
            ).iterator

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace,
                "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * hidden * 2,),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(local_workspace, "fc2_done_counter")
            combine_output_u8 = cute.recast_tensor(combine_output, cutlass.Uint8)
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_u8 = combine_output
            fc2_output_target = combine_output

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace,
            "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(shared_workspace, "expert_recv_count")
        expert_recv_count_sum = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace,
            "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum`` buffer --
        # low32 bits hold per-expert total token count after _dispatch_barrier;
        # zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._make_typed_view(
            shared_workspace,
            self._shared_offsets["expert_recv_count_sum"],
            cutlass.Int32,
            (self.num_experts_per_rank,),
            (2,),
            self._shared_region_by_name["expert_recv_count_sum"].align,
        )
        local_zero_prefix = self._make_typed_view(
            local_workspace,
            0,
            cutlass.Int32,
            (self.local_zero_i32_count,),
            (1,),
            16,
        )
        shared_zero_prefix = self._make_typed_view(
            shared_workspace,
            0,
            cutlass.Int32,
            (self.shared_zero_i32_count,),
            (1,),
            16,
        )

        token_comm_args = ExtractedTokenCommArgs(
            input_token_buffer=activation,
            # BF16 tokens carry no scale-factor sideband: the SF slots ride the
            # TokenCommArgs None-skipping serialization, and dispatch's SF
            # loops are compiled out by sf_uint32_per_token=0.
            input_sf_buffer=None,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=None,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=local_zero_prefix,
            shared_zero_prefix=shared_zero_prefix,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=peer_rank_ptr_mapper_host.rank_idx,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            sm_count=sm_count,
        )

        if cutlass.const_expr(staging_inputs is not None):
            live_rows, hidden_source, ids_source, scores_source = staging_inputs
            staging_inputs = (
                cute.make_tensor(
                    hidden_source[0],
                    cute.make_layout(
                        (live_rows, self.hidden),
                        stride=(self.hidden, 1)
                        if cutlass.const_expr(hidden_source[1] is None)
                        else hidden_source[1],
                    ),
                ),
                cute.make_tensor(
                    ids_source[0],
                    cute.make_layout((live_rows, self.num_topk), stride=ids_source[1]),
                ),
                cute.make_tensor(
                    scores_source[0],
                    cute.make_layout(
                        (live_rows, self.num_topk), stride=scores_source[1]
                    ),
                ),
            )

        self._launch_fc12(
            activation=l1_token_buffer_bf16,
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc1_alpha=fc1_alpha,
            fc2_alpha=fc2_alpha,
            fc1_norm_const=fc1_norm_const,
            fc1_output=fc1_output,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            reduced_output=reduced_output,
            staging_inputs=staging_inputs,
            recv_counter_bank=self._view_local(local_workspace, "recv_counter_bank"),
            fc1_done_counter=fc1_done_counter,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            expert_token_sizes=expert_token_sizes,
            token_comm_args=token_comm_args,
        )

    @cute.jit
    def _launch_fc12(
        self,
        activation,
        fc1_weight,
        fc1_weight_sf,
        fc1_alpha,
        fc2_alpha,
        fc1_norm_const: Optional[cute.Tensor],
        fc1_output,
        fc2_weight,
        fc2_weight_sf,
        fc2_output,
        reduced_output: Optional[cute.Tensor],
        staging_inputs: Optional[Tuple[cute.Tensor, cute.Tensor, cute.Tensor]],
        recv_counter_bank: cute.Tensor,
        fc1_done_counter,
        max_active_clusters,
        stream,
        load_balance_counter,
        expert_token_sizes,
        token_comm_args,
    ):
        e, gateup, h = self.static_expert_shape
        i = gateup // 2
        # The shared NVFP4 [E, K/2, N] views retain K-major backing. Reinterpret
        # packed bytes as logical FP4 A; FC1 already has gate16/up16 row order.
        a1 = cute.make_tensor(
            cute.recast_ptr(fc1_weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((gateup, h, e), stride=(h, 1, gateup * h)),
        )
        a2 = cute.make_tensor(
            cute.recast_ptr(fc2_weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((h, i, e), stride=(i, 1, h * i)),
        )
        # Native SF planes are [E, padded_size], also shared with W4A4. Accept
        # either the E4M3 view or its uint8 alias without changing the bytes.
        s1 = cute.make_tensor(
            cute.recast_ptr(fc1_weight_sf.iterator, dtype=cutlass.Float8E4M3FN),
            blockscaled_utils.tile_atom_to_shape_SF(a1.shape, 16),
        )
        s2 = cute.make_tensor(
            cute.recast_ptr(fc2_weight_sf.iterator, dtype=cutlass.Float8E4M3FN),
            blockscaled_utils.tile_atom_to_shape_SF(a2.shape, 16),
        )
        b1 = cute.make_tensor(
            activation.iterator,
            cute.make_layout((activation.shape[0], h, 1), stride=(h, 1, 0)),
        )
        b2 = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout((fc1_output.shape[0], i, 1), stride=(i, 1, 0)),
        )
        c_layout_view = cute.make_tensor(
            fc1_output.iterator,
            cute.make_layout((i, fc1_output.shape[0], 1), stride=(1, i, 0)),
        )
        # The minimum-depth layout establishes the scheduler's CTA geometry.
        mix = self._make_mixed(128, c_layout_view, 2, 2)
        self.cta_tile_shape_mnk = mix.cta_tile_shape_mnk
        counter_ptr = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            counter_ptr = load_balance_counter.iterator
        sched = MoEFusedFc12SchedulerParams(
            scenario=self.scenario,
            expert_shape=self.static_expert_shape,
            cta_tile_shape_mnk=self.cta_tile_shape_mnk,
            cluster_shape_mn=self.cluster_shape_mn,
            group_hint=self.group_hint,
            token_padding_block=self.token_padding_block,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=counter_ptr,
            override_num_stages=self.num_sched_stages,
            expert_token_sizes=expert_token_sizes,
        )
        (
            self.mixed_fc1,
            self.shared_storage_cls,
            self.num_activation_stages,
        ) = self._fit_raw_stages(c_layout_view, sched)
        mix = self.mixed_fc1
        self.mixed_fc2 = self._make_mixed(
            32, c_layout_view, mix.num_load2trans_stage, self.num_activation_stages
        )
        self.num_acc_stage = 2
        self.epilogue = W4A16Epilogue(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            token_back_by_dispatch=self.token_back_by_dispatch,
            in_kernel_fc2_reduce=self.in_kernel_fc2_reduce,
            apply_topk_in_fc1=self.apply_topk_in_fc1,
            epi_flag_batch=self.epi_flag_batch,
            static_expert_shape=self.static_expert_shape,
            gate_up_clamp=self.gate_up_clamp,
            swiglu_alpha=self.swiglu_alpha,
            swiglu_beta=self.swiglu_beta,
            situ_beta=self.situ_beta,
            situ_linear_beta=self.situ_linear_beta,
        )
        assert self.epilogue.acc_tmem_cols * self.num_acc_stage == mix.num_acc_tmem_cols
        tiled_mma = sm100_utils.make_trivial_tiled_mma(
            cutlass.BFloat16,
            mix.a_major_mode,
            mix.b_major_mode,
            cutlass.Float32,
            mix.cta_group,
            self.mma_tiler[:2],
            mix.transform_a_source,
        )
        a_op = mixed_input_utils.get_tma_atom_kind(
            mix.is_a_mcast, self.use_2cta_instrs, False
        )
        b_op = mixed_input_utils.get_tma_atom_kind(
            mix.is_b_mcast, self.use_2cta_instrs, True
        )
        raw_stage = cute.slice_(mix.smem_layout_a, (None, None, None, 0))
        sf_stage = cute.slice_(mix.smem_layout_scale_tma, (None, None, None, 0))
        b_stage = cute.slice_(mix.smem_layout_b, (None, None, None, 0))
        wa1, wt1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a1,
            raw_stage,
            self.mma_tiler,
            tiled_mma,
            mix.cluster_layout_vmnk.shape,
        )
        wa2, wt2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            a2,
            raw_stage,
            self.mma_tiler,
            tiled_mma,
            mix.cluster_layout_vmnk.shape,
        )
        sa1, st1 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            s1,
            sf_stage,
            self.mma_tiler,
            tiled_mma,
            mix.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        sa2, st2 = cute.nvgpu.make_tiled_tma_atom_A(
            a_op,
            s2,
            sf_stage,
            self.mma_tiler,
            tiled_mma,
            mix.cluster_layout_vmnk.shape,
            internal_type=cutlass.Int16,
        )
        ba1, bt1 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b1, b_stage, self.mma_tiler, tiled_mma, mix.cluster_layout_vmnk.shape
        )
        ba2, bt2 = cute.nvgpu.make_tiled_tma_atom_B(
            b_op, b2, b_stage, self.mma_tiler, tiled_mma, mix.cluster_layout_vmnk.shape
        )
        self.a_tx_bytes = cute.size_in_bytes(
            cutlass.Float4E2M1FN, raw_stage
        ) + cute.size_in_bytes(cutlass.Float8E4M3FN, mix.smem_layout_scale_per_stage)
        self.b_tx_bytes = cute.size_in_bytes(cutlass.BFloat16, b_stage) * cute.size(
            tiled_mma.thr_id.shape
        )
        grid = sched.get_grid_shape(max_active_clusters)
        self.kernel(
            tiled_mma,
            wa1,
            wt1,
            sa1,
            st1,
            ba1,
            bt1,
            wa2,
            wt2,
            sa2,
            st2,
            ba2,
            bt2,
            b2,
            fc2_output,
            fc1_alpha,
            fc1_done_counter,
            sched,
            mix.cluster_layout_vmnk,
            mix.smem_layout_a,
            mix.smem_layout_scale,
            mix.smem_layout_scale_tma,
            mix.smem_layout_b,
            mix.smem_layout_a_transform,
            token_comm_args,
            fc2_alpha,
            reduced_output,
            staging_inputs,
            recv_counter_bank,
            fc1_norm_const,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            cluster=(*self.cluster_shape_mn, 1),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.jit
    def _weight_task(
        self,
        mma,
        ext,
        work,
        atom,
        tensor,
        sf_atom,
        sf_tensor,
        s_raw,
        s_sf_tma,
        cluster_layout,
        cluster_coord,
        pipe,
        state,
        k_count,
    ):
        """Packed A+SF TMA path from local W4A16 and swapped Mega slicing."""
        real_a, _ = ext.get_gmem_tensor("a", tensor, work)
        real_s, _ = ext.get_gmem_tensor("sfa", sf_tensor, work)
        thr = mma.get_slice(cute.arch.block_idx()[0] % cute.size(mma.thr_id.shape))
        ga = cute.local_tile(
            real_a, (self.mma_tiler[0], self.mma_tiler[2]), (None, None, None)
        )
        gs = cute.local_tile(
            real_s, (self.mma_tiler[0], self.mma_tiler[2]), (None, None, None)
        )
        cta_layout = cute.make_layout(
            cute.slice_(cluster_layout, (0, 0, None, 0)).shape
        )
        dst_a, src_a = cpasync.tma_partition(
            atom,
            cluster_coord[2],
            cta_layout,
            cute.group_modes(s_raw, 0, 3),
            cute.group_modes(thr.partition_A(ga), 0, 3),
        )
        dst_s, src_s = cpasync.tma_partition(
            sf_atom,
            cluster_coord[2],
            cta_layout,
            cute.group_modes(s_sf_tma, 0, 3),
            cute.group_modes(thr.partition_A(gs), 0, 3),
        )
        dst_s, src_s = cute.filter_zeros(dst_s), cute.filter_zeros(src_s)
        mma_tile_m = work.tile_m_idx // cute.size(mma.thr_id.shape)
        src_a = src_a[(None, mma_tile_m, None, 0)]
        src_s = src_s[(None, mma_tile_m, None, 0)]
        mask = cpasync.create_tma_multicast_mask(
            cluster_layout, cluster_coord, mcast_mode=2
        )
        state.reset_count()
        for _ in cutlass.range(k_count, unroll=1):
            pipe.producer_acquire(state)
            cute.copy(
                atom,
                src_a[(None, state.count)],
                dst_a[(None, state.index)],
                tma_bar_ptr=pipe.producer_get_barrier(state),
                mcast_mask=mask,
                cache_policy=cutlass.Int64(0x12F0000000000000),
            )
            cute.copy(
                sf_atom,
                src_s[(None, state.count)],
                dst_s[(None, state.index)],
                tma_bar_ptr=pipe.producer_get_barrier(state),
                mcast_mask=mask,
            )
            pipe.producer_commit(state)
            state.advance()
        return state

    @cute.jit
    def _activation_task(
        self,
        mma,
        ext,
        work,
        atom,
        tensor,
        s_b,
        cluster_layout,
        cluster_coord,
        pipe,
        state,
        k_count,
    ):
        real_b, _ = ext.get_gmem_tensor("b", tensor, work)
        # Match the existing swapped Mega dynamic-N split in both FC phases.
        # Only two-CTA MMA splits B at align16(valid)/2. One-CTA
        # MMA consumes the full routed-N tile, multicast across the cluster.
        if cutlass.const_expr(self.use_2cta_instrs):
            if cute.arch.block_idx()[0] % 2 != 0:
                shift = dynamic_mainloop.compute_non_leader_cta_load_shift(
                    valid_tokens_in_tile=work.valid_tokens_in_cta_tile,
                    mma_tiler_n=self.mma_tiler[1],
                )
                real_b = cute.domain_offset((shift, 0, 0), real_b)
        thr = mma.get_slice(cute.arch.block_idx()[0] % cute.size(mma.thr_id.shape))
        gb = cute.local_tile(
            real_b, (self.mma_tiler[1], self.mma_tiler[2]), (None, None, None)
        )
        cta_layout = cute.make_layout(
            cute.slice_(cluster_layout, (0, None, 0, 0)).shape
        )
        dst, src = cpasync.tma_partition(
            atom,
            cluster_coord[1],
            cta_layout,
            cute.group_modes(s_b, 0, 3),
            cute.group_modes(thr.partition_B(gb), 0, 3),
        )
        src = src[(None, work.tile_n_idx, None, 0)]
        mask = cpasync.create_tma_multicast_mask(
            cluster_layout, cluster_coord, mcast_mode=1
        )
        state.reset_count()
        for _ in cutlass.range(k_count, unroll=1):
            pipe.producer_acquire(state)
            cute.copy(
                atom,
                src[(None, state.count)],
                dst[(None, state.index)],
                tma_bar_ptr=pipe.producer_get_barrier(state),
                mcast_mask=mask,
                cache_policy=cutlass.Int64(0x14F0000000000000),
            )
            pipe.producer_commit(state)
            state.advance()
        return state

    @cute.jit
    def _prepare_reduce_worker(
        self, token_comm_args, reduced_output, score_reg, reduce_thread
    ):
        workers_per_cta = self.threads_per_cta - self.token_comm.num_dispatch_threads
        # Decode groups own the first strips; each WG spans the grid.
        worker_idx = (
            (reduce_thread // 128) * token_comm_args.sm_count
            + self.token_comm._cta_linear_id()
        ) * 128 + reduce_thread % 128
        worker_stride = token_comm_args.sm_count * workers_per_cta
        num_workers = reduced_output.shape[0] * self.topk_reduce.hidden_tiles
        token_idx = cutlass.Int32(0)
        hidden_tile_idx = cutlass.Int32(0)
        # Dispatch publishes local scores before the scheduler releases work.
        # Preparation never reads still-pending remote combine writes.
        if worker_idx < num_workers:
            token_idx, hidden_tile_idx = self.topk_reduce._prepare_bf16_worker(
                None
                if cutlass.const_expr(self.apply_topk_in_fc1)
                else token_comm_args.input_topk_weights_buffer,
                score_reg,
                worker_idx,
            )
        return worker_idx, worker_stride, num_workers, token_idx, hidden_tile_idx

    @cute.kernel
    def kernel(
        self,
        mma,
        wa1,
        wt1,
        sa1,
        st1,
        ba1,
        bt1,
        wa2,
        wt2,
        sa2,
        st2,
        ba2,
        bt2,
        fc1_output,
        fc2_output,
        fc1_alpha,
        fc1_done,
        sched_params,
        cluster_layout,
        raw_layout,
        scale_layout,
        scale_tma_layout,
        activation_layout,
        transform_layout,
        token_comm_args,
        fc2_alpha,
        reduced_output: Optional[cute.Tensor],
        staging_inputs: Optional[Tuple[cute.Tensor, cute.Tensor, cute.Tensor]],
        recv_counter_bank: cute.Tensor,
        fc1_norm_const: Optional[cute.Tensor],
    ):
        mix = self.mixed_fc1
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        # Snapshot before any dispatch/scheduler use. The bank word changes
        # only after all producers retire; these views retain this generation.
        bank = recv_counter_bank[0]
        count_offset = bank * (self.shared_zero_i32_count // 2)
        token_comm_args.expert_recv_count = cute.make_tensor(
            token_comm_args.expert_recv_count.iterator + count_offset,
            token_comm_args.expert_recv_count.layout,
        )
        token_comm_args.expert_recv_count_sum = cute.make_tensor(
            token_comm_args.expert_recv_count_sum.iterator + count_offset,
            token_comm_args.expert_recv_count_sum.layout,
        )
        clear_offset = 2 * count_offset
        if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") != "1"):
            # Reclaim the previous bank after input publication. Kernel replay
            # instead retains the active-bank tail reset for peer counters.
            clear_offset = self.shared_zero_i32_count - clear_offset
        token_comm_args.shared_zero_prefix = cute.make_tensor(
            token_comm_args.shared_zero_prefix.iterator + clear_offset,
            token_comm_args.shared_zero_prefix.layout,
        )
        sched_params.expert_token_sizes = cute.make_tensor(
            sched_params.expert_token_sizes.iterator + 2 * count_offset,
            sched_params.expert_token_sizes.layout,
        )
        cta_v_size = cute.size(mma.thr_id.shape)
        cta_v = cute.arch.block_idx()[0] % cta_v_size
        leader = cta_v == 0
        cluster_coord = cluster_layout.get_flat_coord(
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        )
        fc2_threshold = cute.ceil_div(
            self.intermediate_gateup, self.cta_tile_shape_mnk[0]
        )
        ext = W4A16Fc12SchedExtension(
            sf_vec_size=16,
            fc1_done_counter_ptr=fc1_done.iterator,
            fc2_spin_threshold=fc2_threshold,
            fc1_ready_counter_ptr=self.token_comm.fc1_ready_counter_ptr(
                token_comm_args
            ),
        )
        sched_cls = sched_params.get_scheduler_type()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)
        comm_storage = smem.allocate(self.token_comm.extra_smem_storage_class())
        raw_pipe = pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.raw_barriers.data_ptr(),
            num_stages=mix.num_load2trans_stage,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, mix.num_mcast_ctas_a * self.num_transform_warps
            ),
            tx_count=self.a_tx_bytes,
            cta_layout_vmnk=cluster_layout,
            tidx=tidx - 384 if tidx >= 384 else tidx,
            mcast_mode_mn=(1, 0),
            defer_sync=True,
        )
        transform_pipe = pipeline.PipelineAsyncUmma.create(
            barrier_storage=storage.transform_barriers.data_ptr(),
            num_stages=mix.num_trans2mma_stage,
            producer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 32 * self.num_transform_warps * cta_v_size
            ),
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
        )
        activation_pipe = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.activation_barriers.data_ptr(),
            num_stages=self.num_activation_stages,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, mix.num_mcast_ctas_b
            ),
            tx_count=self.b_tx_bytes,
            cta_layout_vmnk=cluster_layout,
            mcast_mode_mn=(0, 1),
            defer_sync=True,
        )
        acc_pipe = pipeline.PipelineUmmaAsync.create(
            barrier_storage=storage.acc_barriers.data_ptr(),
            num_stages=2,
            producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread),
            consumer_group=pipeline.CooperativeGroup(
                pipeline.Agent.Thread, 128 * cta_v_size
            ),
            cta_layout_vmnk=cluster_layout,
            defer_sync=True,
        )
        tmem = utils.TmemAllocator(
            storage.tmem_holding.ptr,
            barrier_for_retrieve=pipeline.NamedBarrier(
                barrier_id=self.tmem_alloc_sync_bar_id,
                num_threads=32 * (5 + self.num_transform_warps),
            ),
            allocator_warp_id=0,
            is_two_cta=self.use_2cta_instrs,
            two_cta_tmem_dealloc_mbar_ptr=storage.tmem_dealloc.ptr,
            arch=self.arch,
        )
        scheduler = sched_cls.create(
            sched_params,
            cute.arch.block_idx(),
            cute.arch.grid_dim(),
            sched_storage=storage.sched_storage,
            num_consumer_threads=32 * (7 + self.num_transform_warps),
            ext=ext,
        )
        early_init = self.load_balance_mode == "atomic_counter"
        if cutlass.const_expr(early_init):
            scheduler.internal_init(warp_idx=warp, sched_warp_id=7)
        pipeline_init_arrive(cluster_shape_mn=self.cluster_shape_mn, is_relaxed=True)
        s_raw = smem.allocate_tensor(
            cutlass.Float4E2M1FN,
            raw_layout.outer,
            byte_alignment=128,
            swizzle=raw_layout.inner,
        )
        s_sf = smem.allocate_tensor(
            cutlass.Float8E4M3FN,
            scale_layout.outer,
            byte_alignment=128,
            swizzle=scale_layout.inner,
        )
        s_sf_tma = cute.make_tensor(s_sf.iterator, scale_tma_layout)
        s_b = smem.allocate_tensor(
            cutlass.BFloat16,
            activation_layout.outer,
            byte_alignment=128,
            swizzle=activation_layout.inner,
        )
        acc_fake = mma.make_fragment_C(
            cute.append(mma.partition_shape_C(self.mma_tiler[:2]), 2)
        )
        pipeline_init_wait(cluster_shape_mn=self.cluster_shape_mn)
        tmem.allocate(512)
        if warp < 5 or warp >= 12:
            tmem.wait_for_alloc()

        if warp >= 4 and warp < 8:
            # A warpgroup must execute the same register-redistribution instruction.
            cute.arch.setmaxregister_decrease(80)

        if warp == 7:
            self.token_comm.sched_warp_pre_init_wait(token_comm_args)
            if cutlass.const_expr(not early_init):
                scheduler.internal_init(warp_idx=warp, sched_warp_id=7)
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                scheduler.publish_work()
                scheduler.gen_next_work()
            scheduler.publish_work()
            scheduler.produce_tail()

        # Each role owns its scheduler state, including mutations inside jit calls.
        if warp == 5:
            consumer = scheduler.make_consumer()
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, mix.num_load2trans_stage
            )
            work = consumer.consume_work()
            while work.is_valid_tile:
                if work.phase == cutlass.Int32(BlockPhase.Linear1):
                    state = self._weight_task(
                        mma,
                        ext,
                        work,
                        wa1,
                        wt1,
                        sa1,
                        st1,
                        s_raw,
                        s_sf_tma,
                        cluster_layout,
                        cluster_coord,
                        raw_pipe,
                        state,
                        self._fc1_k_tiles,
                    )
                else:
                    state = self._weight_task(
                        mma,
                        ext,
                        work,
                        wa2,
                        wt2,
                        sa2,
                        st2,
                        s_raw,
                        s_sf_tma,
                        cluster_layout,
                        cluster_coord,
                        raw_pipe,
                        state,
                        self._fc2_k_tiles,
                    )
                work = consumer.consume_work()
            raw_pipe.producer_tail(state)

        if warp == 6:
            consumer = scheduler.make_consumer()
            state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, self.num_activation_stages
            )
            work = consumer.consume_work()
            while work.is_valid_tile:
                if work.phase == cutlass.Int32(BlockPhase.Linear1):
                    self.token_comm.fc1_tma_b_predispatch_spin(token_comm_args, work)
                    state = self._activation_task(
                        mma,
                        ext,
                        work,
                        ba1,
                        bt1,
                        s_b,
                        cluster_layout,
                        cluster_coord,
                        activation_pipe,
                        state,
                        self._fc1_k_tiles,
                    )
                else:
                    if not work.peek_ready:
                        spin_wait(
                            fc1_done.iterator
                            + work.cumulative_token_block_count
                            + work.tile_n_idx,
                            lambda v: v >= fc2_threshold,
                            fail_sleep_cycles=500,
                        )
                    state = self._activation_task(
                        mma,
                        ext,
                        work,
                        ba2,
                        bt2,
                        s_b,
                        cluster_layout,
                        cluster_coord,
                        activation_pipe,
                        state,
                        self._fc2_k_tiles,
                    )
                work = consumer.consume_work()
            activation_pipe.producer_tail(state)

        if warp == 4:
            consumer = scheduler.make_consumer()
            acc = cute.make_tensor(tmem.retrieve_ptr(cutlass.Float32), acc_fake.layout)
            # Match the inherited transform destination after both acc stages.
            a_ptr = cute.recast_ptr(
                acc.iterator + mix.num_acc_tmem_cols, dtype=cutlass.BFloat16
            )
            a_frag = cute.make_tensor(
                a_ptr, mma.make_fragment_A(transform_layout.outer).layout
            )
            b_frag = mma.make_fragment_B(s_b)
            a_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, mix.num_trans2mma_stage
            )
            b_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, self.num_activation_stages
            )
            acc_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 2
            )
            work = consumer.consume_work()
            while work.is_valid_tile:
                k_count = cutlass.Int32(self._fc2_k_tiles)
                if work.phase == cutlass.Int32(BlockPhase.Linear1):
                    k_count = cutlass.Int32(self._fc1_k_tiles)
                a_state.reset_count()
                b_state.reset_count()
                if leader:
                    acc_pipe.producer_acquire(acc_state)
                    tile_acc = acc[(None, None, None, acc_state.index)]
                    for k_tile in cutlass.range(k_count, unroll=1):
                        transform_pipe.consumer_wait(a_state)
                        activation_pipe.consumer_wait(b_state)
                        dynamic_mainloop.issue_dynamic_bf16_mma_tile(
                            acc_tensor=tile_acc,
                            a_frag_tile=a_frag[(None, None, None, a_state.index)],
                            b_frag_tile=b_frag[(None, None, None, b_state.index)],
                            k_tile_idx=k_tile,
                            valid_tokens_in_tile=work.valid_tokens_in_cta_tile,
                            mma_tiler_mnk=self.mma_tiler,
                        )
                        transform_pipe.consumer_release(a_state)
                        activation_pipe.consumer_release(b_state)
                        a_state.advance()
                        b_state.advance()
                    acc_pipe.producer_commit(acc_state)
                acc_state.advance()
                work = consumer.consume_work()
            acc_pipe.producer_tail(acc_state)

        if warp < 4:
            consumer = scheduler.make_consumer()
            cute.arch.setmaxregister_increase(144)
            self.epilogue.run(
                tmem_ptr=tmem.retrieve_ptr(cutlass.Float32),
                acc_pipeline=acc_pipe,
                sched_consumer=consumer,
                sched_ext=ext,
                fc1_output=fc1_output,
                fc2_output=fc2_output,
                fc1_done_counter=fc1_done,
                tidx=tidx,
                optional_epi_args=W4A16EpiArgs(
                    fc1_alpha=fc1_alpha,
                    fc2_alpha=fc2_alpha,
                    fc1_norm_const=fc1_norm_const,
                ),
                token_comm_args=token_comm_args,
            )
            cute.arch.fence_acq_rel_sys()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem.retrieve_ptr(cutlass.Float32), 512)

        if warp >= 8 and warp < 12:
            cute.arch.setmaxregister_decrease(64)
            if cutlass.const_expr(staging_inputs is not None):
                self.token_comm.stage_inputs(
                    staging_inputs[0],
                    staging_inputs[1],
                    staging_inputs[2],
                    token_comm_args,
                    warp_idx=warp,
                    lane_idx=cute.arch.lane_idx(),
                )
            self.token_comm.dispatch_warp_body(
                token_comm_args,
                comm_storage,
                warp_idx=warp,
                lane_idx=cute.arch.lane_idx(),
                tidx=tidx,
            )
            if cutlass.const_expr(os.environ.get("MEGA_USE_NCU", "0") != "1"):
                # Input publication retires previous-bank readers on every
                # rank. The existing drain publishes these clears before the
                # next invocation can write this bank, overlapping live GEMMs.
                self.token_comm.tail_reset_counters(
                    token_comm_args,
                    token_comm_args.shared_zero_prefix,
                    cta_linear_id=self.token_comm._cta_linear_id(),
                    local_warp_idx=warp - self.token_comm.dispatch_warp_start,
                    lane_idx=cute.arch.lane_idx(),
                )
        # Keep combine state out of the other compute roles' live ranges.
        if cutlass.const_expr(
            not self.in_kernel_fc2_reduce and reduced_output is not None
        ):
            score_reg = None
            if cutlass.const_expr(not self.apply_topk_in_fc1):
                score_reg = cute.make_rmem_tensor(
                    (self.num_topk,), token_comm_args.input_topk_weights_buffer.dtype
                )
            worker_idx = cutlass.Int32(0)
            worker_stride = cutlass.Int32(0)
            num_workers = cutlass.Int32(0)
            token_idx = cutlass.Int32(0)
            hidden_tile_idx = cutlass.Int32(0)
        if warp >= 12:
            consumer = scheduler.make_consumer()
            # Both decode groups retain their initial 96-register allocation.
            # Control80 and dispatch64 donate exactly what epilogue144 needs:
            # 128*(144+80+64+96+96) = 640*96 = 61440 registers.
            # Each group still decodes a disjoint half of the same K tile.
            transform_group_idx = (warp - self.transform_warp_id[0]) // 4
            transform_local_tidx = tidx - 32 * (
                self.transform_warp_id[0] + transform_group_idx * 4
            )
            # Shared NVFP4 gate16/up16 rows already match each MMA/epilogue
            # warp's TMEM band. Decode the packed weights and SF directly;
            # expert global alpha remains separate from BF16 weight decoding.
            accumulators = cute.make_tensor(
                tmem.retrieve_ptr(cutlass.Float32), acc_fake.layout
            )
            copy_in = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(), cutlass.Float4E2M1FN, num_bits_per_copy=32
            )
            copy_out = cute.make_copy_atom(
                tcgen05.St32x32bOp(tcgen05.Repetition(8), tcgen05.Unpack.NONE),
                cutlass.BFloat16,
            )
            sf_for_transform = mma.get_slice(0).partition_A(s_sf)
            parts_fc1 = self.mixed_fc1._setup_transform_partitions(
                mma,
                copy_in,
                copy_out,
                s_raw,
                transform_layout,
                None,
                sf_for_transform,
                accumulators,
                transform_local_tidx,
                transform_group_idx,
            )
            parts_fc2 = self.mixed_fc2._setup_transform_partitions(
                mma,
                copy_in,
                copy_out,
                s_raw,
                transform_layout,
                None,
                sf_for_transform,
                accumulators,
                transform_local_tidx,
                transform_group_idx,
            )
            raw_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, mix.num_load2trans_stage
            )
            transform_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, mix.num_trans2mma_stage
            )
            work = consumer.consume_work()
            while work.is_valid_tile:
                if work.phase == cutlass.Int32(BlockPhase.Linear1):
                    raw_state, transform_state = self.mixed_fc1._transform_tile(
                        raw_pipe,
                        transform_pipe,
                        raw_state,
                        transform_state,
                        *parts_fc1,
                        cutlass.Int32(self._fc1_k_tiles),
                    )
                else:
                    raw_state, transform_state = self.mixed_fc2._transform_tile(
                        raw_pipe,
                        transform_pipe,
                        raw_state,
                        transform_state,
                        *parts_fc2,
                        cutlass.Int32(self._fc2_k_tiles),
                    )
                work = consumer.consume_work()
            if cutlass.const_expr(
                not self.in_kernel_fc2_reduce and reduced_output is not None
            ):
                # Prepare while the final MMA may still be consuming weights.
                (
                    worker_idx,
                    worker_stride,
                    num_workers,
                    token_idx,
                    hidden_tile_idx,
                ) = self._prepare_reduce_worker(
                    token_comm_args,
                    reduced_output,
                    score_reg,
                    tidx - 32 * self.transform_warp_id[0],
                )
            transform_pipe.producer_tail(transform_state)

        tail_barrier = pipeline.NamedBarrier(
            barrier_id=self.token_comm.kernel_tail_named_barrier_id,
            num_threads=self.token_comm.kernel_tail_threads,
        )
        tail_barrier.arrive_and_wait()
        self.token_comm.kernel_tail_drain(
            token_comm_args,
            warp_idx=warp,
            lane_idx=cute.arch.lane_idx(),
        )
        if cutlass.const_expr(
            not self.in_kernel_fc2_reduce and reduced_output is not None
        ):
            if warp < 8:
                # Other retired roles prepare while dispatch completes the drain.
                (
                    worker_idx,
                    worker_stride,
                    num_workers,
                    token_idx,
                    hidden_tile_idx,
                ) = self._prepare_reduce_worker(
                    token_comm_args,
                    reduced_output,
                    score_reg,
                    tidx + 32 * self.num_transform_warps,
                )
            # bar.sync is aligned: every role must execute one common site.
            tail_barrier.arrive_and_wait()
            if warp < 8 or warp >= 12:
                combine = cute.recast_tensor(
                    token_comm_args.combine_output, cutlass.BFloat16
                )
                while worker_idx < num_workers:
                    self.topk_reduce._reduce_bf16_worker(
                        combine,
                        None
                        if cutlass.const_expr(self.apply_topk_in_fc1)
                        else token_comm_args.input_topk_weights_buffer,
                        reduced_output,
                        token_idx,
                        hidden_tile_idx,
                        score_reg,
                    )
                    worker_idx += worker_stride
                    if worker_idx < num_workers:
                        token_idx, hidden_tile_idx = (
                            self.topk_reduce._prepare_bf16_worker(
                                None
                                if cutlass.const_expr(self.apply_topk_in_fc1)
                                else token_comm_args.input_topk_weights_buffer,
                                score_reg,
                                worker_idx,
                            )
                        )
            else:
                self.token_comm.kernel_tail_cleanup(
                    token_comm_args,
                    warp_idx=warp,
                    lane_idx=cute.arch.lane_idx(),
                )
        else:
            self.token_comm.kernel_tail_cleanup(
                token_comm_args, warp_idx=warp, lane_idx=cute.arch.lane_idx()
            )
        if (
            self.token_comm._cta_linear_id() == 0
            and warp == self.token_comm.dispatch_warp_start
            and cute.arch.lane_idx() == 0
        ):
            recv_counter_bank[0] = bank ^ cutlass.Int32(1)

    def _pool_shapes(self) -> Tuple[int, int]:
        # Every source token may select every local expert up to top-k. Each
        # expert adds tail padding, then one partial scheduler tile of slack.
        raw = self.world_size * self.max_tokens_per_rank * min(
            self.num_topk, self.num_experts_per_rank
        ) + self.num_experts_per_rank * (self.token_padding_block - 1)
        pool_token_capacity = _round_up(raw, self.token_padding_block)
        pool_task_tile_capacity = (
            pool_token_capacity + self.cluster_tile_tokens - 1
        ) // self.cluster_tile_tokens + self.num_experts_per_rank
        return pool_token_capacity, pool_task_tile_capacity

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        pool_token_capacity = self.pool_token_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        intermediate_downproj = self.intermediate_downproj
        cluster_tile_tokens = self.cluster_tile_tokens

        # fc1_done_counter slot granularity is the cluster tile on the token
        # (M) axis, plus one boundary slot per expert.
        fc1_done_slots = (
            pool_token_capacity + cluster_tile_tokens - 1
        ) // cluster_tile_tokens + num_experts_per_rank

        # Accumulating-counter prefix: tail_reset_counters bulk-zeros all bytes
        # before l1_token_buffer so back-to-back launches do not inherit counters.
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity,),
                16,
            ),
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (num_total_experts,),
                16,
            ),
            _RegionSpec(
                "grid_sync_counter",
                cutlass.Int32,
                (_GridSyncSlotCount,),
                16,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots,),
                16,
            ),
        ]
        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_done_counter",
                    cutlass.Int32,
                    (num_experts_per_rank,),
                    16,
                )
            )
            if self.token_back_schedule_mode == "atomic_counter":
                specs.append(
                    _RegionSpec(
                        "token_back_schedule_counter",
                        cutlass.Int32,
                        (1,),
                        16,
                    )
                )
        if self.load_balance_mode == "atomic_counter":
            specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )

        # Data buffers. l1_token_buffer must be the first data region because
        # __init__ derives local_zero_i32_count from its offset.
        specs += [
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored as
            # Uint8 bytes; the BF16 view at the same offset is built in __call__.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Persisted across launches; the sense-reversing nvlink barrier rides
            # this phase counter across non-ncu relaunches.
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
            _RegionSpec("recv_counter_bank", cutlass.Int32, (1,), 16),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (pool_token_capacity,),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            # fc1 -> fc2 hand-off buffer in the activation dtype (BFloat16).
            _RegionSpec(
                "fc1_output",
                self.ab_dtype,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
        ]

        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    cutlass.BFloat16,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                )
            )

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        max_slot = max_tokens_per_rank * num_topk

        # Two equally aligned counter banks. Normal launches reclaim the
        # previous bank while computing; the next launch publishes to it.
        return [
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (num_experts_per_rank,),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_bank1",
                cutlass.Int64,
                (world_size, num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum_bank1",
                cutlass.Int64,
                (num_experts_per_rank,),
                16,
            ),
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
            _RegionSpec(
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount,),
                16,
            ),
        ]

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)``."""
        return self._local_total, self._shared_total

    @staticmethod
    def _make_typed_view(
        byte_workspace: cute.Tensor,
        byte_offset: int,
        cute_dtype: Any,
        shape: Tuple[int, ...],
        stride: Optional[Tuple[int, ...]],
        assumed_align: int,
    ) -> cute.Tensor:
        """Build a typed cute view at ``byte_offset`` of the opaque workspace."""
        byte_ptr = byte_workspace.iterator + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(typed_iter, cute.make_layout(shape, stride=stride))

    def _view_local(self, workspace: cute.Tensor, name: str) -> cute.Tensor:
        return self._partition_region(
            workspace, self._local_offsets[name], self._local_region_by_name[name]
        )

    def _view_shared(self, workspace: cute.Tensor, name: str) -> cute.Tensor:
        return self._partition_region(
            workspace, self._shared_offsets[name], self._shared_region_by_name[name]
        )

    def _partition_region(
        self, workspace: cute.Tensor, offset: int, spec: _RegionSpec
    ) -> cute.Tensor:
        return self._make_typed_view(
            workspace,
            offset,
            spec.cute_dtype,
            spec.shape,
            spec.stride_row_major,
            spec.align,
        )
