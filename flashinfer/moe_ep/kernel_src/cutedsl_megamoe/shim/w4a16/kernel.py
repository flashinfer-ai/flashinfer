# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (c) 2026 FlashInfer contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Packed NVFP4 weights decoded into operand-A TMEM inside fused MegaMoE.

The local W4A16 helpers load packed weights and block scales, then decode
BF16 tiles directly into the TMEM operand pipeline. Both GEMMs use
dynamic routed-token widths with M128/M256, N64/N128 and K256 allocation.
The swapped Mega scheduler and BF16 dispatch/combine remain one kernel.
"""

from typing import Optional

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.pipeline import pipeline_init_arrive, pipeline_init_wait
from cutlass.cutlass_dsl import extract_mlir_values, new_from_mlir_values
from cutlass.base_dsl.dsl import extract_mlir_attributes
from cutlass.utils import mixed_input_helpers as mixed_input_utils
from cutlass.utils import blockscaled_layout as blockscaled_utils

from flashinfer.fused_moe.cute_dsl.blackwell.moe_w4a16_kernel import (
    Sm100W4A16GroupedGemmKernel,
)
from moe_bf16_glu.megamoe_kernel_bf16 import (
    Sm100MegaMoEBf16Kernel,
    _layout_regions,
)
from moe_nvfp4_swapab.megamoe_kernel import Sm100MegaMoEKernel
from moe_nvfp4_swapab.custom_ext import SwapABSwigluFp4Fc12SchedExtension
from moe_nvfp4_swapab.fc1_fc2_fuse_sched import BlockPhase, MoEFusedFc12SchedulerParams
from moe_nvfp4_swapab.moe_utils import spin_wait
from src.token_comm import TokenInPullTokenBackPush
from src.token_comm import TokenCommArgs as ExtractedTokenCommArgs

from moe_nvfp4_swapab.epilogue_refactor import NvFp4OptinalEpiArgs
from .epilogue import W4A16Epilogue
from . import dynamic_mainloop


class _ScaledTokenCommArgs:
    """Keep local FC2 weight scaling separate from the communication ABI."""

    def __init__(self, comm, fc2_alpha):
        self.comm = comm
        self.fc2_alpha = fc2_alpha

    def __getattr__(self, name):
        return getattr(self.comm, name)

    def __extract_mlir_values__(self):
        return extract_mlir_values(self.comm) + extract_mlir_values(self.fc2_alpha)

    def __extract_mlir_attributes__(self):
        return extract_mlir_attributes(self.comm) + extract_mlir_attributes(
            self.fc2_alpha
        )

    def __new_from_mlir_values__(self, values):
        n = len(extract_mlir_values(self.comm))
        return _ScaledTokenCommArgs(
            new_from_mlir_values(self.comm, values[:n]),
            new_from_mlir_values(self.fc2_alpha, values[n:]),
        )


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


class Sm100W4A16MegaMoEKernel(Sm100MegaMoEKernel):
    """BF16 dispatch and compute with online-decoded NVFP4 expert weights."""

    _make_typed_view = staticmethod(Sm100MegaMoEBf16Kernel._make_typed_view)

    def __init__(self, *, local_rank, **kwargs):
        kwargs = dict(kwargs)
        kwargs.pop("ab_dtype", None)
        kwargs.pop("generate_c", None)
        kwargs.pop("use_stg_fc1", None)
        ikr = kwargs.pop("fc2_in_kernel_topk_reduce", False)
        token_back_mode = kwargs.get("token_back_mode", "epi_warps")
        by_dispatch = kwargs.pop(
            "token_back_by_dispatch", token_back_mode == "reuse_dispatch_warps"
        )
        if (
            ikr
            or kwargs.get("in_kernel_fc2_reduce", False)
            or kwargs.get("apply_topk_in_fc1", False)
        ):
            raise ValueError("W4A16 MegaMoE uses external post-FC2 routing.")
        if token_back_mode not in ("epi_warps", "reuse_dispatch_warps"):
            raise ValueError(
                "W4A16 supports epi_warps or reuse_dispatch_warps token return."
            )
        if by_dispatch != (token_back_mode == "reuse_dispatch_warps"):
            raise ValueError("token_back_by_dispatch must match token_back_mode.")
        if kwargs["mma_tiler_mnk"] not in (
            (128, 64, 256),
            (128, 128, 256),
            (256, 64, 256),
            (256, 128, 256),
        ):
            raise ValueError(
                "W4A16 MegaMoE requires mma_tiler_mnk=M128/M256, N64/N128, K256."
            )
        if kwargs["cluster_shape_mnk"] != (2, 1, 1) and not (
            kwargs["cluster_shape_mnk"] == (1, 1, 1)
            and kwargs["mma_tiler_mnk"] == (128, 64, 256)
        ):
            raise ValueError(
                "W4A16 requires cluster (2,1,1), or (1,1,1) for M128/N64/K256."
            )
        if kwargs["use_2cta_instrs"] != (kwargs["mma_tiler_mnk"][0] == 256):
            raise ValueError("W4A16 MMA M128/M256 requires one/two-CTA instructions.")
        _, gateup, hidden = kwargs["static_expert_shape"]
        if hidden % 32 or gateup % 128:
            raise ValueError("W4A16 requires H%32=0 and I%64=0.")
        # Match local W4A16's ceil-div local_tile K extent. A/B descriptors
        # retain logical K; TMA zero-fills the last packed/BF16 input tile.
        self._fc1_k_tiles = (hidden + 255) // 256
        self._fc2_k_tiles = (gateup // 2 + 255) // 256
        if kwargs["token_padding_block"] > kwargs["mma_tiler_mnk"][1]:
            raise ValueError("Token padding must not exceed the routed-token tile.")
        kwargs["sf_padding_block"] = 1
        super().__init__(fc2_output_dtype=cutlass.BFloat16, **kwargs)
        self.num_sched_stages = self.num_sched_stages or 3
        self.local_rank = local_rank
        self.ab_dtype = cutlass.BFloat16
        self.fc2_in_kernel_topk_reduce = False
        self.generate_c = False
        self.use_stg_fc1 = False
        self.hidden_bytes = 2 * self.hidden
        self.sf_uint32_per_token = 0
        self.sf_padding_block = 1
        # Reuse the local W4A16 same-K-tile split across two transform groups.
        self.num_transform_warpgroups = 2
        self.num_transform_warps = 4 * self.num_transform_warpgroups
        self.transform_warp_id = tuple(range(12, 12 + self.num_transform_warps))
        self.threads_per_cta = 32 * (12 + self.num_transform_warps)
        # Mirror swapped Mega's per-expert completion threshold. Every FC2
        # feature CTA publishes, including the padded CTA of a partial cluster.
        cluster_fc2_tile_hidden = (
            self.mma_tiler[0]
            * self.cluster_shape_mn[0]
            // (2 if self.use_2cta_instrs else 1)
        )
        fc2_publishes_per_token_cluster_tile = (
            (self.hidden + cluster_fc2_tile_hidden - 1) // cluster_fc2_tile_hidden
        ) * self.cluster_shape_mn[0]
        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.BFloat16,
            combine_format=self.combine_format,
            token_back_by_dispatch=self.token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=fc2_publishes_per_token_cluster_tile,
            token_back_reduce_topk=False,
            token_back_standalone=False,
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=8,
            num_other_warps=8 + self.num_transform_warps,
            flag_batch=self.flag_batch,
            is_swap_ab=True,
            token_back_schedule_mode=self.token_back_schedule_mode,
        )
        # Rebuild the initially constructed NVFP4 region metadata for BF16
        # tokens/intermediates. No GPU storage has been allocated at this point.
        self._local_region_specs = Sm100MegaMoEBf16Kernel._build_local_region_specs(
            self
        )
        self._shared_region_specs = Sm100MegaMoEBf16Kernel._build_shared_region_specs(
            self
        )
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        self._local_region_by_name = {r.name: r for r in self._local_region_specs}
        self._shared_region_by_name = {r.name: r for r in self._shared_region_specs}
        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["src_token_topk_idx"]
        self.require_zero_workspace_leading_bytes = local_leading, shared_leading
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    def name(self):
        return "w4a16_" + super().name()

    def _make_mixed(self, fragment_size, output_tensor, raw_stages):
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
        # Only packed A+SF use the fitted depth. The inherited layout helper
        # shares a load count for A and B; recover its original two-stage B
        # layout explicitly, leaving activation and both TMEM rings unchanged.
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
            2,
            mixed.num_trans2mma_stage,
        )
        return mixed

    @staticmethod
    def _make_shared_storage(sched_storage_cls, raw_stages, transform_stages):
        @cute.struct
        class SharedStorage:
            raw_barriers: cute.struct.MemRange[cutlass.Int64, 2 * raw_stages]
            transform_barriers: cute.struct.MemRange[
                cutlass.Int64, 2 * transform_stages
            ]
            activation_barriers: cute.struct.MemRange[cutlass.Int64, 4]
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
            sched_params, SwapABSwigluFp4Fc12SchedExtension, num_drain_warps=0
        )
        comm_storage_cls = self.token_comm_extra_smem_storage_class()
        # CuTe's 227KiB per-block capacity already excludes CUDA's 1KiB
        # reservation from the SM's 228KiB. All user storage is counted below.
        capacity = utils.get_smem_capacity_in_bytes("sm_100")
        for raw_stages in range(5, 1, -1):
            mixed = self._make_mixed(128, output_tensor, raw_stages)
            storage_cls = self._make_shared_storage(
                sched_storage_cls, raw_stages, mixed.num_trans2mma_stage
            )
            if self._smem_size(storage_cls, comm_storage_cls, mixed) <= capacity:
                return mixed, storage_cls
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
        fc1_c: Optional[cute.Tensor],  # fc1 c output
        # Combine destination (peer write target via the epilogue Fc2OutputDest).
        combine_output: cute.Tensor,  # (T, num_topk, hidden) BF16
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch the BF16 MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``topk_weights`` / ``combine_output`` MUST point
            into memory reachable via
            ``peer_rank_ptr_mapper.ptr_map_to_rank(...)`` (NVSHMEM symmetric
            heap).  Single-rank degenerate runs are allowed.
          * ``topk_idx`` is read on the local rank only.
          * ``fc1_weight`` / ``fc2_weight`` are local-only.

        ``combine_output`` is the MoE-domain ``(max_tokens_per_rank, num_topk,
        hidden)`` BF16 storage; the epilogue maps each pool row back to the
        source rank's ``[src_token, src_topk, :]`` slot via ``token_comm_args``
        (form A; host reduces the topk axis).
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
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
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

        token_comm_args = _ScaledTokenCommArgs(token_comm_args, fc2_alpha)
        self._launch_fc12(
            activation=l1_token_buffer_bf16,
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc1_alpha=fc1_alpha,
            fc1_output=fc1_output,
            fc1_c=fc1_c,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            offs=None,
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
        fc1_output,
        fc2_weight,
        fc2_weight_sf,
        fc2_output,
        topk_scores,
        fc1_done_counter,
        offs,
        max_active_clusters,
        stream,
        load_balance_counter,
        expert_token_sizes,
        token_comm_args,
        fc1_c=None,
    ):
        e, gateup, h = self.static_expert_shape
        i = gateup // 2
        # Packed uint8 ABI becomes logical FP4 A without copying. Both stages
        # expose prepared feature order; the transform permutes gate/up below.
        a1 = cute.make_tensor(
            cute.recast_ptr(fc1_weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((gateup, h, e), stride=(h, 1, gateup * h)),
        )
        a2 = cute.make_tensor(
            cute.recast_ptr(fc2_weight.iterator, dtype=cutlass.Float4E2M1FN),
            cute.make_layout((h, i, e), stride=(i, 1, h * i)),
        )
        # Weight preparation with block_scale_interleave() changes only internal
        # scale layout. These arguments point to those flat E4M3 buffers.
        s1 = cute.make_tensor(
            fc1_weight_sf.iterator,
            blockscaled_utils.tile_atom_to_shape_SF(a1.shape, 16),
        )
        s2 = cute.make_tensor(
            fc2_weight_sf.iterator,
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
        mix = self._make_mixed(128, c_layout_view, 2)
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
            sf_padding_block=1,
            load_balance_mode=self.load_balance_mode,
            load_balance_counter_ptr=counter_ptr,
            override_num_stages=self.num_sched_stages,
            is_swap_ab=True,
            expert_token_prefix_sum=None,
            expert_token_sizes=expert_token_sizes,
        )
        self.mixed_fc1, self.shared_storage_cls = self._fit_raw_stages(
            c_layout_view, sched
        )
        mix = self.mixed_fc1
        self.mixed_fc2 = self._make_mixed(32, c_layout_view, mix.num_load2trans_stage)
        self.cluster_layout_vmnk = mix.cluster_layout_vmnk
        self.num_acc_stage = self.num_acc_pipeline_stages = 2
        self.num_tmem_alloc_cols = 512
        self.epilogue = W4A16Epilogue(
            mma_tiler_mnk=self.mma_tiler,
            cluster_shape_mn=self.cluster_shape_mn,
            use_2cta_instrs=self.use_2cta_instrs,
            fc1_output_dtype=cutlass.BFloat16,
            combine_format=self.combine_format,
            non_ubulk_fc2_store=True,
            in_kernel_fc2_reduce=False,
            token_back_by_dispatch=self.token_back_by_dispatch,
            epi_flag_batch=self.epi_flag_batch,
            acc_dtype=cutlass.Float32,
            allow_overlap_acc=False,
            static_expert_shape=self.static_expert_shape,
            gate_up_clamp=self.gate_up_clamp,
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
        ).launch(
            grid=sched.get_grid_shape(max_active_clusters),
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
    ):
        mix = self.mixed_fc1
        tidx = cute.arch.thread_idx()[0]
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        cta_v_size = cute.size(mma.thr_id.shape)
        cta_v = cute.arch.block_idx()[0] % cta_v_size
        leader = cta_v == 0
        cluster_coord = cluster_layout.get_flat_coord(
            cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        )
        fc2_threshold = cute.ceil_div(
            self.intermediate_gateup, self.cta_tile_shape_mnk[0]
        )
        ext = SwapABSwigluFp4Fc12SchedExtension(
            sf_vec_size=16,
            fc1_done_counter_ptr=fc1_done.iterator,
            fc2_spin_threshold=fc2_threshold,
            fc1_ready_counter_ptr=self.token_comm_hook_fc1_ready_counter_ptr(
                token_comm_args
            ),
        )
        sched_cls = sched_params.get_scheduler_type()
        smem = utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage_cls)
        comm_storage = smem.allocate(self.token_comm_extra_smem_storage_class())
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
            num_stages=2,
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
        consumer = scheduler.make_consumer()
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
            self.token_comm_hook_sched_warp_pre_init_wait(token_comm_args)
            if cutlass.const_expr(not early_init):
                scheduler.internal_init(warp_idx=warp, sched_warp_id=7)
            scheduler.gen_next_work()
            while scheduler.current_work.is_valid_tile:
                ext.prefetch_for_expert(scheduler.current_work.expert_idx)
                scheduler.publish_work()
                scheduler.gen_next_work()
            scheduler.publish_work()
            scheduler.produce_tail()

        if warp == 5:
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
            state = pipeline.make_pipeline_state(pipeline.PipelineUserType.Producer, 2)
            work = consumer.consume_work()
            while work.is_valid_tile:
                if work.phase == cutlass.Int32(BlockPhase.Linear1):
                    self.token_comm_hook_fc1_tma_b_predispatch_spin(
                        token_comm_args, work
                    )
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

        if warp >= 12:
            # Redistribute only the CTA's initial 640*96 register allocation.
            # 128*(176+80+64+80+80) = 61440 registers after redistribution.
            cute.arch.setmaxregister_decrease(80)
            # Both groups decode disjoint halves of the same K tile. This
            # 80-register budget keeps both groups within the CTA register pool.
            transform_group_idx = (warp - self.transform_warp_id[0]) // 4
            transform_local_tidx = tidx - 32 * (
                self.transform_warp_id[0] + transform_group_idx * 4
            )
            # Mirror local W4A16's gated transform view: the public prepared
            # weights remain gate32/up32, but each MMA/epilogue warp owns a
            # gate16/up16 TMEM band. Permute packed A and SF identically,
            # before the existing BF16 transform; never fold global alpha.
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
            gated_rows = cute.make_layout(((16, 2), (2, 2)), stride=((1, 32), (16, 64)))
            fc1_raw = cute.composition(s_raw, ((gated_rows, None), None, None, None))
            fc1_sf = cute.composition(s_sf, ((gated_rows, None), None, None))
            sf_fc1_transform = mma.get_slice(0).partition_A(fc1_sf)
            sf_for_transform = mma.get_slice(0).partition_A(s_sf)
            parts_fc1 = self.mixed_fc1._setup_transform_partitions(
                mma,
                copy_in,
                copy_out,
                fc1_raw,
                transform_layout,
                None,
                sf_fc1_transform,
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
            transform_pipe.producer_tail(transform_state)

        if warp == 4:
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
                pipeline.PipelineUserType.Consumer, 2
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
                            a_from_tmem=True,
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
            cute.arch.setmaxregister_increase(176)
            self.epilogue.run(
                epi_smem_storage=None,
                tmem_ptr=tmem.retrieve_ptr(cutlass.Float32),
                acc_pipeline=acc_pipe,
                sched_consumer=consumer,
                sched_ext=ext,
                tma_atom_fc1_output=None,
                fc1_output=fc1_output,
                fc1_output_sf=None,
                fc2_output=fc2_output,
                fc1_done_counter=fc1_done,
                tidx=tidx,
                optional_epi_args=NvFp4OptinalEpiArgs(
                    fc1_alpha=fc1_alpha,
                    fc2_alpha=token_comm_args.fc2_alpha,
                    fc1_norm_const=None,
                    topk_scores=None,
                ),
                token_comm_args=token_comm_args,
            )
            cute.arch.fence_acq_rel_sys()
            tmem.relinquish_alloc_permit()
            tmem.free(tmem.retrieve_ptr(cutlass.Float32), 512)

        if warp >= 8 and warp < 12:
            cute.arch.setmaxregister_decrease(64)
            self.token_comm_hook_dispatch_warp_body(
                token_comm_args,
                comm_storage,
                warp_idx=warp,
                lane_idx=cute.arch.lane_idx(),
                tidx=tidx,
            )
        self.token_comm_hook_kernel_tail(
            token_comm_args, warp_idx=warp, lane_idx=cute.arch.lane_idx(), tidx=tidx
        )
