# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE kernel for MXFP8 weights and BF16 token activations.

This module adds token dispatch/combine plumbing to
``Sm100SwapABMxfp8Bf16Fc12Kernel``.  The local GEMMs remain the mixed
MXFP8-weight/BF16-activation swap-AB pipeline:

* weights are K-major E4M3FN or E5M2 with atom-swizzled E8M0/K32 scales;
* dispatched input tokens and the FC1->FC2 hand-off are BF16;
* the token axis is MMA N, so one dispatch task tile contains
  ``mma_tiler_n * cluster_n`` token rows.

Phase-3 exposes two BF16 epilogue-warp combine forms:

* static expert shape;
* DeepGEMM routing semantics (top-k score is absorbed by FC1);
* Form A writes one ``(token, topk, hidden)`` result per route;
* Form B atomically reduces routes into ``(token, 1, hidden)``;
* epilogue-warps direct peer STG or reuse of the dispatch warps for token-back;
* no activation scale-factor sideband.

The world-size-one path uses the same peer-mapping and communication ABI as a
future multi-rank launch.  It is therefore a degenerate communication run, not
a separate local-only implementation.
"""

# Do not enable ``from __future__ import annotations`` here.  CuTeDSL struct
# introspection requires live annotation objects in class bodies.

from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from moe_mxfp8_bf16_glu.kernel_mxfp8_bf16_glu_fc12 import (
    Sm100SwapABMxfp8Bf16Fc12Kernel,
)
from moe_nvfp4_swapab.megamoe_kernel import (
    _GridSyncSlotCount,
    _NvlinkSlotCount,
    _RegionSpec,
    _TokenMetadataBytes,
    _layout_regions,
    _round_up,
)
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
)


class Sm100MegaMoEMxfp8Bf16Kernel(Sm100SwapABMxfp8Bf16Fc12Kernel):
    """Fused dispatch + mixed FC1/FC2 + BF16 Form-A/Form-B combine."""

    def __init__(
        self,
        # Mixed FC12 base configuration.
        mma_tiler_mnk: Tuple[int, int, int] = (256, 128, 128),
        cluster_shape_mnk: Tuple[int, int, int] = (2, 1, 1),
        use_2cta_instrs: bool = True,
        group_hint: int = 1,
        token_padding_block: int = 64,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = 2,
        transform_buffer: Literal["smem", "tmem"] = "tmem",
        accumulator_overlap: bool = False,
        transform_k_tile: Literal[64, 128] = 128,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        ab_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        # MegaMoE constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        fc2_in_kernel_topk_reduce: bool = False,
        token_back_by_dispatch: bool = False,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        flag_batch: int = 1,
        gate_up_clamp: Optional[float] = None,
        apply_topk_in_fc1: bool = True,
        generate_c: bool = False,
        use_stg_fc1: bool = False,
        combine_format: Optional[CombineFormat] = None,
    ) -> None:
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm100MegaMoEMxfp8Bf16Kernel requires "
                "static_expert_shape != None."
            )
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )
        if world_size <= 0:
            raise ValueError(f"world_size must be positive, got {world_size}.")
        if not (0 <= local_rank < world_size):
            raise ValueError(
                f"local_rank must be in [0, {world_size}), got {local_rank}."
            )
        if not (1 <= num_topk <= 32):
            raise ValueError(f"num_topk must be in [1, 32], got {num_topk}.")
        if max_tokens_per_rank <= 0:
            raise ValueError(
                "max_tokens_per_rank must be positive, got "
                f"{max_tokens_per_rank}."
            )
        if ab_dtype is not cutlass.BFloat16:
            raise NotImplementedError(
                "Phase-3 mixed MegaMoE requires BF16 token activations."
            )
        if acc_dtype is not cutlass.Float32:
            raise NotImplementedError(
                "Phase-3 mixed MegaMoE requires FP32 accumulation."
            )
        if token_back_mode == "standalone_warps":
            raise NotImplementedError(
                "standalone token-back warps conflict with the mixed "
                "MXFP8-to-BF16 transform warps at warp ids 12-15; use "
                "'epi_warps' or 'reuse_dispatch_warps'."
            )
        if token_back_mode not in ("epi_warps", "reuse_dispatch_warps"):
            raise ValueError(
                "token_back_mode must be 'epi_warps' or "
                f"'reuse_dispatch_warps'; got {token_back_mode!r}."
            )
        expected_dispatch_token_back = token_back_mode != "epi_warps"
        if token_back_by_dispatch != expected_dispatch_token_back:
            raise ValueError(
                "token_back_by_dispatch must match token_back_mode; it is "
                "true exactly for token_back_mode='reuse_dispatch_warps'."
            )
        if not apply_topk_in_fc1:
            raise NotImplementedError(
                "Phase-3 MVP requires DeepGEMM routing semantics "
                "(apply_topk_in_fc1=True)."
            )
        if generate_c:
            raise NotImplementedError(
                "Phase-3 MVP does not expose the raw FC1 gate/up output."
            )
        if use_stg_fc1:
            raise NotImplementedError(
                "The mixed FC12 path already uses its fixed BF16 FC1 STG "
                "handoff; the BF16 Mega use_stg_fc1 knob is not applicable."
            )
        if combine_format is None:
            combine_format = CombineFormat.parse("bf16")
        if (
            combine_format.is_quantized
            or combine_format.act_dtype is not cutlass.BFloat16
        ):
            raise NotImplementedError(
                "Phase-3 supports BF16 Form-A/Form-B combine only."
            )

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=(
                2 if num_sched_stages is None else num_sched_stages
            ),
            transform_buffer=transform_buffer,
            accumulator_overlap=accumulator_overlap,
            transform_k_tile=transform_k_tile,
            acc_dtype=acc_dtype,
            epi_flag_batch=epi_flag_batch,
            gate_up_clamp=gate_up_clamp,
            apply_topk_in_fc1=apply_topk_in_fc1,
            fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
        )

        # The mixed base calls _setup_warp_topology while tracing.  Enabling
        # communication relocates transform warps to 12-15 and reserves 8-11
        # for dispatch, yielding the fixed 16-warp CTA.
        self.enable_token_comm = True
        self.dispatch_warp_id = (8, 9, 10, 11)
        self.transform_warp_id = (12, 13, 14, 15)
        self.threads_per_cta = 16 * 32

        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden
        self.combine_format = combine_format
        self.fc2_in_kernel_topk_reduce = fc2_in_kernel_topk_reduce
        self.token_back_by_dispatch = token_back_by_dispatch
        self.token_back_mode = token_back_mode
        self.token_back_standalone = False
        self.token_back_schedule_mode = (
            self.load_balance_mode
            if self.token_back_by_dispatch
            else "static"
        )

        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2
        self.num_total_experts = world_size * self.num_experts_per_rank

        # BF16 dispatch moves raw token bytes and no activation SF sideband.
        self.hidden_bytes = hidden * (cutlass.BFloat16.width // 8)
        self.sf_uint32_per_token = 0

        # Swap-AB token axis is MMA N.  Cluster N is one for the supported
        # mixed geometry, but retain the general formula for later layouts.
        self.cluster_tile_tokens = (
            mma_tiler_mnk[1] * cluster_shape_mnk[1]
        )
        if self.cluster_tile_tokens % self.token_padding_block != 0:
            raise ValueError(
                "cluster_tile_tokens must be divisible by "
                f"token_padding_block; got {self.cluster_tile_tokens} and "
                f"{self.token_padding_block}."
            )

        (
            self.pool_token_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Four epilogue + MMA + two TMA + scheduler + four transform warps
        # cohabit the CTA with the four dispatch warps.
        num_other_warps = (
            len(self.epilogue_warp_id)
            + 1
            + 1
            + 1
            + 1
            + len(self.transform_warp_id)
        )
        # Each FC2 CTA publishes once after storing its hidden tile.  In
        # swap-AB the hidden axis is MMA M; account for the per-CTA width of a
        # two-CTA instruction and for every CTA in cluster M.  A too-small
        # threshold lets token-back read incomplete staging rows, while a
        # too-large threshold deadlocks the dispatch warps.
        cluster_fc2_tile_hidden = (
            self.mma_tiler[0]
            * self.cluster_shape_mn[0]
            // (2 if self.use_2cta_instrs else 1)
        )
        fc2_publishes_per_token_cluster_tile = (
            (
                self.hidden
                + cluster_fc2_tile_hidden
                - 1
            )
            // cluster_fc2_tile_hidden
        ) * self.cluster_shape_mn[0]
        if not self.token_back_by_dispatch:
            fc2_publishes_per_token_cluster_tile = 0

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.BFloat16,
            combine_format=self.combine_format,
            token_back_by_dispatch=self.token_back_by_dispatch,
            fc2_publishes_per_token_cluster_tile=(
                fc2_publishes_per_token_cluster_tile
            ),
            token_back_reduce_topk=(
                self.token_back_by_dispatch
                and self.fc2_in_kernel_topk_reduce
            ),
            token_back_standalone=False,
            sf_uint32_per_token=0,
            token_padding_block=self.token_padding_block,
            sf_padding_block=1,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            is_swap_ab=True,
            flag_batch=flag_batch,
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
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            region.name: region for region in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            region.name: region for region in self._shared_region_specs
        }

        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["src_token_topk_idx"]
        self.require_zero_workspace_leading_bytes = (
            local_leading,
            shared_leading,
        )
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    # ------------------------------------------------------------------
    # Static workspace planning
    # ------------------------------------------------------------------

    def _pool_shapes(self) -> Tuple[int, int]:
        max_recv = self.world_size * self.max_tokens_per_rank
        max_local_replicas = min(
            self.num_topk, self.num_experts_per_rank
        )
        raw_token_capacity = (
            max_recv * max_local_replicas
            + self.num_experts_per_rank
            * (self.token_padding_block - 1)
        )
        pool_token_capacity = _round_up(
            raw_token_capacity, self.token_padding_block
        )
        pool_task_tile_capacity = (
            (
                pool_token_capacity
                + self.cluster_tile_tokens
                - 1
            )
            // self.cluster_tile_tokens
            + self.num_experts_per_rank
        )
        return pool_token_capacity, pool_task_tile_capacity

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        fc1_done_slots = (
            (
                self.pool_token_capacity
                + self.cluster_tile_tokens
                - 1
            )
            // self.cluster_tile_tokens
            + self.num_experts_per_rank
        )
        specs: List[_RegionSpec] = [
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (self.pool_task_tile_capacity,),
                16,
            ),
            _RegionSpec(
                "expert_send_count",
                cutlass.Int64,
                (self.num_total_experts,),
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
                    (self.num_experts_per_rank,),
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

        # The counter reset prefix ends at l1_token_buffer.  The persistent
        # NVLink phase counter is intentionally placed after this boundary.
        specs += [
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (self.pool_token_capacity, self.hidden_bytes),
                128,
            ),
            _RegionSpec(
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
            _RegionSpec(
                "l1_topk_weights_buffer",
                cutlass.Float32,
                (self.pool_token_capacity,),
                16,
            ),
            _RegionSpec(
                "token_src_metadata",
                cutlass.Uint8,
                (self.pool_token_capacity, _TokenMetadataBytes),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                cutlass.BFloat16,
                (
                    self.pool_token_capacity,
                    self.intermediate_downproj,
                ),
                128,
            ),
        ]
        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    cutlass.BFloat16,
                    (
                        self.pool_token_capacity,
                        1,
                        self.hidden,
                    ),
                    128,
                )
            )
        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        max_slot = self.max_tokens_per_rank * self.num_topk
        return [
            _RegionSpec(
                "expert_recv_count",
                cutlass.Int64,
                (self.world_size, self.num_experts_per_rank),
                16,
            ),
            _RegionSpec(
                "expert_recv_count_sum",
                cutlass.Int64,
                (self.num_experts_per_rank,),
                16,
            ),
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (
                    self.num_experts_per_rank,
                    self.world_size,
                    max_slot,
                ),
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
        """Return ``(local_workspace_bytes, shared_workspace_bytes)``."""
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
        byte_ptr = byte_workspace.iterator + Int64(byte_offset)
        typed_iter = cute.make_ptr(
            cute_dtype,
            byte_ptr.toint(),
            AddressSpace.gmem,
            assumed_align=assumed_align,
        )
        return cute.make_tensor(
            typed_iter, cute.make_layout(shape, stride=stride)
        )

    def _partition_region(
        self,
        byte_workspace: cute.Tensor,
        offsets: Dict[str, int],
        spec: _RegionSpec,
        *,
        cute_dtype: Optional[Any],
        shape: Optional[Tuple[int, ...]],
        stride: Optional[Tuple[int, ...]],
    ) -> cute.Tensor:
        dtype = spec.cute_dtype if cute_dtype is None else cute_dtype
        region_shape = spec.shape if shape is None else shape
        region_stride = stride
        if region_stride is None:
            if cute_dtype is None and shape is None:
                region_stride = spec.stride_row_major
            else:
                row_major: List[int] = [1]
                for dim in reversed(list(region_shape)[1:]):
                    row_major.append(row_major[-1] * dim)
                row_major.reverse()
                region_stride = tuple(row_major)
        return self._make_typed_view(
            byte_workspace,
            offsets[spec.name],
            dtype,
            region_shape,
            region_stride,
            spec.align,
        )

    def _view_local(
        self,
        local_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            local_workspace,
            self._local_offsets,
            self._local_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    def _view_shared(
        self,
        shared_workspace: cute.Tensor,
        name: str,
        *,
        cute_dtype: Optional[Any] = None,
        shape: Optional[Tuple[int, ...]] = None,
        stride: Optional[Tuple[int, ...]] = None,
    ) -> cute.Tensor:
        return self._partition_region(
            shared_workspace,
            self._shared_offsets,
            self._shared_region_by_name[name],
            cute_dtype=cute_dtype,
            shape=shape,
            stride=stride,
        )

    @staticmethod
    def _sf_flat_elements(
        output_features: int, reduction_features: int
    ) -> int:
        """Elements in one 128-row/4-column-padded swizzled SF plane."""
        padded_rows = _round_up(output_features, 128)
        scale_columns = reduction_features // 32
        padded_columns = _round_up(scale_columns, 4)
        return padded_rows * padded_columns

    def _validate_public_inputs(
        self,
        activation: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        combine_output: cute.Tensor,
        local_workspace: cute.Tensor,
        shared_workspace: cute.Tensor,
    ) -> None:
        """Validate the public Phase-3 tensor ABI before workspace carving."""
        for name, tensor, expected_rank in (
            ("activation", activation, 2),
            ("topk_idx", topk_idx, 2),
            ("topk_weights", topk_weights, 2),
            ("fc1_weight", fc1_weight, 3),
            ("fc1_weight_sf", fc1_weight_sf, 2),
            ("fc2_weight", fc2_weight, 3),
            ("fc2_weight_sf", fc2_weight_sf, 2),
            ("combine_output", combine_output, 3),
            ("local_workspace", local_workspace, 1),
            ("shared_workspace", shared_workspace, 1),
        ):
            if cutlass.const_expr(cute.rank(tensor) != expected_rank):
                raise ValueError(
                    f"{name} must be rank {expected_rank}, got "
                    f"{cute.rank(tensor)}."
                )

        fp8_types = (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        if cutlass.const_expr(fc1_weight.element_type not in fp8_types):
            raise TypeError("fc1_weight must be E4M3FN or E5M2.")
        if cutlass.const_expr(
            fc2_weight.element_type is not fc1_weight.element_type
        ):
            raise TypeError(
                "FC1 and FC2 weights must use the same MXFP8 dtype."
            )
        if cutlass.const_expr(
            fc1_weight_sf.element_type is not cutlass.Float8E8M0FNU
            or fc2_weight_sf.element_type is not cutlass.Float8E8M0FNU
        ):
            raise TypeError("both weight scale tensors must be E8M0FNU.")
        if cutlass.const_expr(
            activation.element_type is not cutlass.BFloat16
            or combine_output.element_type is not cutlass.BFloat16
        ):
            raise TypeError(
                "activation and combine_output must be BFloat16."
            )
        if cutlass.const_expr(
            topk_idx.element_type is not cutlass.Int64
        ):
            raise TypeError("topk_idx must be Int64.")
        if cutlass.const_expr(
            topk_weights.element_type is not cutlass.Float32
        ):
            raise TypeError("topk_weights must be Float32.")
        if cutlass.const_expr(
            local_workspace.element_type is not cutlass.Uint8
            or shared_workspace.element_type is not cutlass.Uint8
        ):
            raise TypeError("both opaque workspaces must be Uint8.")

        experts = self.num_experts_per_rank
        hidden = self.hidden
        gateup = self.intermediate_gateup
        down = self.intermediate_downproj
        expected_static_dims = (
            ("activation.hidden", activation.shape[1], hidden),
            ("topk_idx.topk", topk_idx.shape[1], self.num_topk),
            (
                "topk_weights.topk",
                topk_weights.shape[1],
                self.num_topk,
            ),
            ("fc1_weight.experts", fc1_weight.shape[0], experts),
            ("fc1_weight.K", fc1_weight.shape[1], hidden),
            ("fc1_weight.M", fc1_weight.shape[2], gateup),
            ("fc2_weight.experts", fc2_weight.shape[0], experts),
            ("fc2_weight.K", fc2_weight.shape[1], down),
            ("fc2_weight.M", fc2_weight.shape[2], hidden),
            ("combine_output.tokens", combine_output.shape[0],
             self.max_tokens_per_rank),
            (
                "combine_output.topk",
                combine_output.shape[1],
                1 if self.fc2_in_kernel_topk_reduce else self.num_topk,
            ),
            ("combine_output.hidden", combine_output.shape[2], hidden),
            ("fc1_weight_sf.experts", fc1_weight_sf.shape[0], experts),
            (
                "fc1_weight_sf.storage",
                fc1_weight_sf.shape[1],
                self._sf_flat_elements(gateup, hidden),
            ),
            ("fc2_weight_sf.experts", fc2_weight_sf.shape[0], experts),
            (
                "fc2_weight_sf.storage",
                fc2_weight_sf.shape[1],
                self._sf_flat_elements(hidden, down),
            ),
        )
        for name, actual, expected in expected_static_dims:
            # mark_layout_dynamic may stage even logically fixed dimensions.
            # The host runner checks every shape unconditionally; retain the
            # same check here whenever the dimension survives as a Python int.
            if cutlass.const_expr(isinstance(actual, int)):
                if cutlass.const_expr(actual != expected):
                    raise ValueError(
                        f"{name} must be {expected}, got {actual}."
                    )

        token_dims = (
            activation.shape[0],
            topk_idx.shape[0],
            topk_weights.shape[0],
        )
        if cutlass.const_expr(
            all(isinstance(dim, int) for dim in token_dims)
        ):
            if cutlass.const_expr(
                token_dims[0] != token_dims[1]
                or token_dims[0] != token_dims[2]
            ):
                raise ValueError(
                    "activation, topk_idx, and topk_weights must have the "
                    "same token dimension."
                )
            if cutlass.const_expr(
                token_dims[0] > self.max_tokens_per_rank
            ):
                raise ValueError(
                    "activation token count exceeds max_tokens_per_rank."
                )

        workspace_dims = (
            local_workspace.shape[0],
            shared_workspace.shape[0],
        )
        if cutlass.const_expr(
            all(isinstance(dim, int) for dim in workspace_dims)
        ):
            if cutlass.const_expr(
                workspace_dims[0] < self._local_total
                or workspace_dims[1] < self._shared_total
            ):
                raise ValueError(
                    "opaque workspace is smaller than "
                    "get_workspace_sizes()."
                )

        for name, stride in (
            ("activation hidden/K", activation.stride[1]),
            ("topk_idx topk", topk_idx.stride[1]),
            ("topk_weights topk", topk_weights.stride[1]),
            ("fc1_weight hidden/K", fc1_weight.stride[1]),
            ("fc1_weight_sf storage", fc1_weight_sf.stride[1]),
            ("fc2_weight down/K", fc2_weight.stride[1]),
            ("fc2_weight_sf storage", fc2_weight_sf.stride[1]),
            ("combine_output hidden", combine_output.stride[2]),
            ("local_workspace", local_workspace.stride[0]),
            ("shared_workspace", shared_workspace.stride[0]),
        ):
            if cutlass.const_expr(isinstance(stride, int)):
                if cutlass.const_expr(stride != 1):
                    raise ValueError(
                        f"{name} dimension must have stride 1."
                    )

    # ------------------------------------------------------------------
    # Public launch ABI
    # ------------------------------------------------------------------

    @cute.jit
    def __call__(
        self,
        activation: cute.Tensor,
        topk_idx: cute.Tensor,
        topk_weights: cute.Tensor,
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        combine_output: cute.Tensor,
        local_workspace: cute.Tensor,
        shared_workspace: cute.Tensor,
        peer_rank_ptr_mapper_host,
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Launch dispatch, mixed FC12, and direct BF16 combine."""
        self._validate_public_inputs(
            activation,
            topk_idx,
            topk_weights,
            fc1_weight,
            fc1_weight_sf,
            fc2_weight,
            fc2_weight_sf,
            combine_output,
            local_workspace,
            shared_workspace,
        )

        cluster_size = (
            self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        )
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = (
            peer_rank_ptr_mapper_host.make_device_obj()
        )

        l1_token_buffer_u8 = self._view_local(
            local_workspace, "l1_token_buffer"
        )
        l1_token_buffer_bf16 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            cutlass.BFloat16,
            (self.pool_token_capacity, self.hidden),
            (self.hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )
        l1_topk_weights_buffer = self._view_local(
            local_workspace, "l1_topk_weights_buffer"
        )
        l1_arrival_count = self._view_local(
            local_workspace, "l1_arrival_count"
        )
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata"
        )
        expert_send_count = self._view_local(
            local_workspace, "expert_send_count"
        )
        grid_sync_counter = self._view_local(
            local_workspace, "grid_sync_counter"
        )
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter"
        )
        fc1_output = self._view_local(
            local_workspace, "fc1_output"
        )
        fc1_done_counter = self._view_local(
            local_workspace, "fc1_done_counter"
        )

        load_balance_counter = None
        if cutlass.const_expr(
            self.load_balance_mode == "atomic_counter"
        ):
            load_balance_counter = self._view_local(
                local_workspace, "load_balance_counter"
            )

        token_back_schedule_counter = None
        if cutlass.const_expr(
            self.token_back_schedule_mode == "atomic_counter"
        ):
            token_back_schedule_counter = self._view_local(
                local_workspace, "token_back_schedule_counter"
            ).iterator

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace, "fc2_output_workspace"
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (self.pool_token_capacity * self.hidden * 2,),
                (1,),
                self._local_region_by_name[
                    "fc2_output_workspace"
                ].align,
            )
            fc2_done_counter = self._view_local(
                local_workspace, "fc2_done_counter"
            )
            combine_output_for_comm = cute.recast_tensor(
                combine_output, cutlass.Uint8
            )
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_for_comm = combine_output
            fc2_output_target = combine_output

        expert_recv_count = self._view_shared(
            shared_workspace, "expert_recv_count"
        )
        expert_recv_count_sum = self._view_shared(
            shared_workspace, "expert_recv_count_sum"
        )
        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx"
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal"
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
            combine_output=combine_output_for_comm,
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
            # The communication ABI models the owning rank as a runtime i32;
            # keep this aligned with the other MegaMoE kernels.
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

        # In token-communication mode the mixed base derives sizes-mode
        # scheduling from token_comm_args.expert_recv_count_sum.  Deliberately
        # omit ``offs`` rather than explicitly passing Python ``None`` through
        # CuTeDSL tracing; lean launches continue to provide prefix sums.
        super().__call__(
            activation=l1_token_buffer_bf16,
            fc1_weight=fc1_weight,
            fc1_weight_sf=fc1_weight_sf,
            fc1_output=fc1_output,
            fc2_weight=fc2_weight,
            fc2_weight_sf=fc2_weight_sf,
            fc2_output=fc2_output_target,
            topk_scores=l1_topk_weights_buffer,
            fc1_done_counter=fc1_done_counter,
            max_active_clusters=max_active_clusters,
            stream=stream,
            load_balance_counter=load_balance_counter,
            token_comm_args=token_comm_args,
        )

    # ------------------------------------------------------------------
    # Token communication hooks consumed by the mixed FC12 base
    # ------------------------------------------------------------------

    def token_comm_extra_smem_storage_class(self) -> type:
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(
        self, token_comm_args
    ):
        self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info
    ):
        self.token_comm.fc1_tma_b_predispatch_spin(
            token_comm_args, work_tile_info
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
        self.token_comm.kernel_tail(
            token_comm_args,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )


__all__ = ["Sm100MegaMoEMxfp8Bf16Kernel"]
