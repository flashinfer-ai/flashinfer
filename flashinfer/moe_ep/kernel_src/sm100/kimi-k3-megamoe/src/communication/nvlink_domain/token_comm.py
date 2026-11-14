# Exported for FlashInfer; local changes are recorded in VENDOR.md.
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Metadata-push routing and fused non-deterministic token communication."""
from ...helpers import codegen_options as _kimi_options

import dataclasses
import os
from typing import Callable, ClassVar, Literal, Optional, Tuple, Union

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Boolean, Int32, Int64
from cutlass.utils.blockscaled_layout import tile_atom_to_shape_SF

from ...api import ImplDesc, KernelComponent, OptionalRequirement, ProblemDesc
from ...helpers.device_workspace import DeviceWorkspace
from ...helpers.dsl_helpers import mark_alignment, smem_exclusive_prefix
from ...helpers.flag_batch import make_flag_batch_tracker
from ...helpers.iket_compat import iket
from ...helpers import epoch as _epoch  # v33-epoch
from ...helpers import prod_config as _prod_config  # prod1
from ...helpers import b10 as _b10  # build 10
from ...helpers.ptx_helpers import (
    cp_async_bulk_s2g,
    cp_reduce_async_bulk_add_bf16_s2g,
    cp_reduce_async_bulk_add_u32_s2g,
    ldg_nl1_b32,
    ldg_nl1_b32_pred,
    ldg_nl1_b64,
    ldg_nl1_b64_pred,
    ld_acquire_sys_b64,
    nanosleep,
    read_clock64,
    red_add_relaxed_sys_s32,
    red_or_relaxed_sys_b32,
    stg_b32_pred,
    stg_b64,
    stg_f32,
    st_relaxed_sys_b64,
    tma_load_1d,
)
from ...helpers.smem_workspace import SmemWorkspace
from ...helpers.software_sync import NvlinkBarrier
from ...helpers.utils import SF_TILE_ROWS_N224, ceil_div, round_up, sf_row_in_expert
from ...quant_def import CombineFormat, QuantKind
from ..token_protocol import TokenSrcMetadata
from .symmetric_buffer import SymmetricBufferDevice

TokenBackMode = Literal["epi_warps", "standalone_warps", "reuse_dispatch_warps"]
TokenBackScheduleMode = Literal["static", "atomic_counter"]


@dataclasses.dataclass(frozen=True)
class TokenCommArgs:
    """Device views materialized inside the fused kernel region.

    ``pre_reduced_activation`` is the per-topk combine staging plane: TokenComm's own symmetric region, except
    under in-kernel top-k reduction where it degenerates to a view of the caller's 2D output. That REDG
    accumulates, so in that mode the incoming content is the caller's accumulation base -- zero, or a
    shared-expert result.
    """

    activation: cute.Tensor
    activation_sf: cute.Tensor
    pre_reduced_activation: cute.Tensor
    pre_reduced_activation_sf: Optional[cute.Tensor]
    peer_rank_ptr_mapper: SymmetricBufferDevice

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "TokenCommArgs":
        if values:
            raise ValueError(f"TokenCommArgs expected no MLIR values, got {len(values)}.")
        return self


@dataclasses.dataclass(frozen=True)
class _SortedElement:
    flat_topk_index: Int32
    topk_score: Optional[cutlass.Float32]

    def pack(self) -> Union[Int64, Int32]:
        if cutlass.const_expr(self.topk_score is None):
            return self.flat_topk_index
        scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
        scratch[0] = self.flat_topk_index
        cute.recast_tensor(scratch, cutlass.Float32)[1] = self.topk_score
        return cute.recast_tensor(scratch, cutlass.Int64)[0]

    @classmethod
    def from_packed(cls, packed: Union[Int64, Int32]) -> "_SortedElement":
        if cutlass.const_expr(type(packed).width == 32):
            return cls(flat_topk_index=Int32(packed), topk_score=None)
        scratch = cute.make_rmem_tensor((2,), cutlass.Int32)
        cute.recast_tensor(scratch, cutlass.Int64)[0] = packed
        return cls(
            flat_topk_index=scratch[0], topk_score=cute.recast_tensor(scratch, cutlass.Float32)[1]
        )


@cute.jit
def _copy_atom(dtype, num_bits_per_copy: int):
    return cute.make_copy_atom(
        cute.nvgpu.CopyUniversalOp(), dtype, num_bits_per_copy=num_bits_per_copy
    )


class _MetadataPushRouter(KernelComponent):
    """Sort and push routing metadata into each destination rank's final pool."""

    router_smem_limit_bytes: ClassVar[int] = 227 * 1024
    router_warps_per_cta: ClassVar[int] = 16

    sizes_by_rank_region = "nvlink.token_comm.sizes_by_rank"
    sizes_region = "nvlink.token_comm.sizes"
    sizes_ready_region = "nvlink.token_comm.sizes_ready"
    metadata_ready_region = "nvlink.token_comm.metadata_ready"
    sorted_metadata_region = "nvlink.token_comm.sorted_metadata"
    sorted_scores_region = "nvlink.token_comm.sorted_scores"
    pool_expert_base_region = "nvlink.token_comm.pool_expert_base"
    token_src_metadata_region = "nvlink.token_comm.token_src_metadata"  # nosec B105
    fc1_topk_scores_region = "nvlink.token_comm.fc1_topk_scores"
    source_expert_base_region = "nvlink.token_comm.source_expert_base"
    push_destination_base_region = "nvlink.token_comm.push_destination_base"
    sorted_metadata_ready_region = "nvlink.token_comm.sorted_metadata_ready"
    push_table_ready_region = "nvlink.token_comm.push_table_ready"
    router_size_counter_region = "nvlink.token_comm.router_size_counter"
    router_histogram_done_region = "nvlink.token_comm.router_histogram_done"
    source_base_ready_region = "nvlink.token_comm.source_base_ready"
    # [b10 G] pull-gather router: per source rank and local expert, the source's 8-byte message (valid bit | source
    # sorted base << 24 | route count).
    pull_message_region = "nvlink.token_comm.pull_message"

    router_data_histogram_region = "nvlink.token_comm.router_smem.data_histogram"
    router_data_prefix_region = "nvlink.token_comm.router_smem.data_prefix"
    router_data_warp_totals_region = "nvlink.token_comm.router_smem.data_warp_totals"
    router_data_sorted_region = "nvlink.token_comm.router_smem.data_sorted"
    router_data_base_region = "nvlink.token_comm.router_smem.data_base"
    router_helper_size_matrix_region = "nvlink.token_comm.router_smem.helper_size_matrix"
    router_helper_totals_region = "nvlink.token_comm.router_smem.helper_totals"
    router_helper_prefix_region = "nvlink.token_comm.router_smem.helper_prefix"
    router_helper_warp_totals_region = "nvlink.token_comm.router_smem.helper_warp_totals"
    router_helper_load_mbarrier_region = "nvlink.token_comm.router_smem.helper_load_mbarrier"
    router_gather_table_region = "nvlink.token_comm.router_smem.gather_table"  # [b10 G]
    # s25-dedup (exact dispatch dedup): per routed element, the source-side link to the token's primary slot on the same
    # destination rank (local), and per destination pool row the final link pushed with the metadata (shared).
    sorted_link_region = "nvlink.token_comm.sorted_link"
    token_src_link_region = "nvlink.token_comm.token_src_link"  # nosec B105
    router_data_reserve_region = "nvlink.token_comm.router_smem.data_reserve"
    router_data_primary_region = "nvlink.token_comm.router_smem.data_primary"
    # s25-dedup v5 runtime gate: this source's duplicate (token, rank) count (DATA CTAs), its on/off decision (helper),
    # and per destination the mask of sources that wrote dedup links this launch (set remotely before sizes_ready)
    dedup_dup_count_region = "nvlink.token_comm.dedup_dup_count"
    dedup_enabled_region = "nvlink.token_comm.dedup_enabled"
    dedup_source_mask_region = "nvlink.token_comm.dedup_source_mask"

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "promised_launchable_sm_count": int,
            "router_smem_limit_bytes": OptionalRequirement(int),
        }

    def __init__(
        self, problem_desc: ProblemDesc, impl_desc: ImplDesc, dedup_tokens_per_slot: int = 0
    ) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]
        # s25-dedup: > 0 = push a dispatch-dedup link for every routed element (TokenComm.dispatch_dedup); the value is
        # the destination's FC1-ready slot size (tokens_per_fc1_ready_slot), needed to locate the primary's slot.
        self.dedup_tokens_per_slot = int(dedup_tokens_per_slot)
        self._dedup_partial = False  # trace-time: the DATA CTAs computed links (multi-CTA path, topk-aligned lanes)
        # v5 runtime gate: a source turns dedup on for this launch only if >= DEDUP_MIN_X1000 / 1000 of its routed slots
        # land on a (token, rank) pair it already sends (real top-16 / 16-rank routing: ~350; bench_moe's materialised
        # routing: 0-10, Zipf 78, receiver hotspot 234). Below it the launch runs prod2's dispatch path (no link work).
        self.dedup_min_x1000 = int(_kimi_options.environ.get("MEGAMOE_DEDUP_MIN_X1000", "125") or "125")

        self.token_padding_block = impl_desc["token_padding_block"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)

        self._validate_router_configuration()
        self.expert_count_padded = round_up(self.expert_count, 4)
        self.expert_count_with_trash = self.expert_count_padded + 1
        self.router_elements_per_lane, self.router_data_cta_count = (
            self._router_launch_configuration()
        )
        self.router_tokens_per_cta = self.router_elements_per_lane * self.router_warps_per_cta * 32
        self.router_push_cta_count = ceil_div(self.expert_count, self.router_warps_per_cta)
        # [b10 G] pull-gather router: multi-DATA-CTA geometry, no dedup links, no FC1 top-k scores, and the gather
        # table (_build_router_smem_workspace) within the helper's size matrix (world size >= 4; not at 1 or 2 ranks).
        gather_table_words = 3 * self.world_size * self.experts_per_rank + 2 * self.experts_per_rank + 1
        self.pull_gather = bool(
            _b10.on("G", self.max_tokens_per_rank)
            and self.router_data_cta_count > 1
            and not self.dedup_tokens_per_slot
            and not self.apply_topk_at_fc1
            and round_up(gather_table_words, 4) <= self.world_size * round_up(self.expert_count, 4)
        )
        # [b10 Q] router launched as a PDL dependent of the activation quantize (which triggers right after its own
        # griddepcontrol.wait): the router's local work overlaps the quantize on SMs the quantize grid leaves free;
        # the helper waits for the quantize grid before its message lets any peer pull this rank's rows.
        self.quant_overlap = bool(self.pull_gather and _b10.on("Q", self.max_tokens_per_rank))
        # [b10 G] gather grid: at least 8 CTAs, one per 2048 routed slots above that, capped at the push grid. Router
        # CTAs hold SMs the PDL-launched MegaMoE CTAs need; 4 / 8 / 16 / 56 CTAs measured within 0.5% of each other at
        # 512 / 768 tokens per rank (job b10-i0), 8 the best overall.
        self.router_gather_cta_count = min(
            self.router_push_cta_count, max(8, ceil_div(self.max_tokens_per_rank * self.topk, 2048))
        )
        gather_ctas_env = _kimi_options.environ.get("MEGAMOE_B10_GATHER_CTAS", "").strip()  # diag / tuning override
        if gather_ctas_env:
            self.router_gather_cta_count = max(1, min(self.router_push_cta_count, int(gather_ctas_env)))
        self.router_grid_cta_count = max(
            self.router_data_cta_count + 1,
            self.router_gather_cta_count if self.pull_gather else self.router_push_cta_count,
        )
        if self.router_grid_cta_count > self.promised_launchable_sm_count:
            raise ValueError(
                "Router grid exceeds promised_launchable_sm_count; "
                "all metadata-push CTAs must be concurrently resident."
            )
        self.worst_case_token_count = self.worst_case_padded_tokens(self.token_padding_block)
        self._router_smem_workspace = self._build_router_smem_workspace()

        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None

    def _validate_router_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "token_padding_block",
            "promised_launchable_sm_count",
            "router_smem_limit_bytes",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("TokenComm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    def worst_case_padded_tokens(self, block: int) -> int:
        source_token_capacity = self.world_size * self.max_tokens_per_rank
        routes_per_source_token = min(self.topk, self.experts_per_rank)
        route_capacity = source_token_capacity * routes_per_source_token
        active_expert_capacity = min(self.experts_per_rank, route_capacity)
        route_budget_blocks = (
            active_expert_capacity + (route_capacity - active_expert_capacity) // block
        )
        expert_bound_blocks = active_expert_capacity * int(ceil_div(source_token_capacity, block))
        return min(route_budget_blocks, expert_bound_blocks) * block

    def _router_launch_configuration(self) -> Tuple[int, int]:
        def next_power_of_two(value: int) -> int:
            return 1 << (max(value, 1) - 1).bit_length()

        routed_token_capacity = self.max_tokens_per_rank * self.topk
        minimum_cta_capacity = 2048
        maximum_cta_capacity = 16384
        maximum_data_cta_count = 128
        maximum_supported_tokens = maximum_cta_capacity * maximum_data_cta_count
        if routed_token_capacity > maximum_supported_tokens:
            raise NotImplementedError(
                f"The router supports at most {maximum_supported_tokens} routed tokens per rank."
            )
        cta_capacity = min(
            maximum_cta_capacity,
            next_power_of_two(max(routed_token_capacity, minimum_cta_capacity)),
        )
        # [prod1, from s2-rcap] Router DATA-CTA capacity cap: fewer routed (token, k) elements per CTA shorten its
        # latency-bound phases (input load, SMEM histogram atomics, counting sort, write-out) at the cost of more DATA
        # CTAs (bounded by 128 and the launchable SM count). Exact: DATA CTAs already reserve per-expert pool ranges
        # with relaxed atomics in arbitrary order, so within-expert row order is already free and every row's FC1 /
        # FC2 / quantize / token-back / reduce is position-independent (DUMP bit-identical u2k/u4k/u8k/h8k/Fz-u8k).
        # Default 2048 for max_tokens_per_rank <= 4096 (u2k -1.1..1.2%, u4k -0.3..1.0%); off above (u8k -0.5%,
        # h8k +0.4%, u16k / Fz-u8k neutral). MEGAMOE_ROUTER_CAP=<pow2 in [2048, 16384]> overrides, 0 = off.
        router_cap = _prod_config.router_cap(self.max_tokens_per_rank)
        if router_cap:
            if router_cap & (router_cap - 1) or not minimum_cta_capacity <= router_cap <= maximum_cta_capacity:
                raise ValueError(f"MEGAMOE_ROUTER_CAP must be a power of two in [2048, 16384], got {router_cap}")
            floor_capacity = next_power_of_two(
                max(minimum_cta_capacity, ceil_div(routed_token_capacity, maximum_data_cta_count))
            )
            while (
                ceil_div(routed_token_capacity, max(router_cap, floor_capacity)) + 1
                > self.promised_launchable_sm_count
            ):
                floor_capacity *= 2
            cta_capacity = min(cta_capacity, max(router_cap, floor_capacity))
        elements_per_lane = cta_capacity // (self.router_warps_per_cta * 32)
        data_cta_count = ceil_div(routed_token_capacity, cta_capacity)
        return elements_per_lane, data_cta_count

    def _build_router_smem_workspace(self) -> SmemWorkspace:
        workspace = SmemWorkspace()
        workspace.register_mbarrier(self.router_helper_load_mbarrier_region, 1)
        overlay = workspace.create_overlay("nvlink.token_comm.router_smem.role")
        data_lifetime = overlay.add_lifetime("data_cta")
        data_lifetime.register_tensor(
            self.router_data_histogram_region, cutlass.Int32, (self.expert_count_with_trash,)
        )
        data_lifetime.register_tensor(
            self.router_data_prefix_region,
            cutlass.Int32,
            (self.expert_count_with_trash,),
            byte_alignment=16,
        )
        data_lifetime.register_tensor(
            self.router_data_warp_totals_region, cutlass.Int32, (self.router_warps_per_cta,)
        )
        data_lifetime.register_tensor(
            self.router_data_sorted_region,
            (cutlass.Int64 if self.apply_topk_at_fc1 else cutlass.Int32),
            (self.router_tokens_per_cta,),
            byte_alignment=16,
        )
        if self.router_data_cta_count > 1:
            data_lifetime.register_tensor(
                self.router_data_base_region,
                cutlass.Int32,
                (self.expert_count_padded,),
                byte_alignment=16,
            )
            if self.dedup_tokens_per_slot:
                data_lifetime.register_tensor(
                    self.router_data_reserve_region,
                    cutlass.Int32,
                    (self.expert_count_padded,),
                    byte_alignment=16,
                )
                # per (CTA-local token, destination rank): min over the token's elements on that rank of
                # (local expert << 22 | run position) -> the primary (smallest expert) and its run position
                data_lifetime.register_tensor(
                    self.router_data_primary_region,
                    cutlass.Int32,
                    (max(1, self.router_tokens_per_cta // self.topk) * self.world_size,),
                    byte_alignment=16,
                )

        if self.pull_gather:
            # [b10 G] counts [W][epr] | source sorted bases [W][epr] | pool-base prefix [epr] | dense prefix [epr] | total
            # | ring offset per task [epr][W]. The helper CTA also runs the gather role right after its messages with no
            # barrier in between: safe because this table overlays only the helper lifetime's size matrix (registered
            # first, never used by the pull-gather helper), not the totals / prefix its message loop reads
            # (pull_gather requires that fit).
            gather_lifetime = overlay.add_lifetime("gather")
            gather_lifetime.register_tensor(
                self.router_gather_table_region,
                cutlass.Int32,
                (round_up(3 * self.world_size * self.experts_per_rank + 2 * self.experts_per_rank + 1, 4),),
                byte_alignment=16,
            )
        helper_lifetime = overlay.add_lifetime("helper_cta")
        helper_lifetime.register_tensor(
            self.router_helper_size_matrix_region,
            cutlass.Int32,
            (self.world_size, self.expert_count_padded),
            stride=(self.expert_count_padded, 1),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_totals_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_prefix_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            byte_alignment=16,
        )
        helper_lifetime.register_tensor(
            self.router_helper_warp_totals_region,
            cutlass.Int32,
            (self.router_warps_per_cta,),
            byte_alignment=16,
        )
        workspace.finalize(max_bytes=self.router_smem_limit_bytes)
        return workspace

    @property
    def router_smem_workspace(self) -> SmemWorkspace:
        return self._router_smem_workspace

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register router-private state and Router-to-Main outputs."""
        self._register_router_workspace(workspace)

    def _register_router_workspace(self, workspace: DeviceWorkspace) -> None:
        maximum_routed_tokens = self.max_tokens_per_rank * self.topk
        workspace.register(
            self.sizes_by_rank_region,
            cutlass.Int32,
            (self.world_size, self.expert_count_padded),
            buffer_space="shared",
            stride=(self.expert_count_padded, 1),
        )
        workspace.register(
            self.sizes_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="shared",
            reset="tail_reset",
        )
        workspace.register(
            self.sizes_ready_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
        )
        workspace.register(
            self.metadata_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="shared",
            reset="tail_reset",
        )
        workspace.register(
            self.sorted_metadata_region,
            cutlass.Int64,
            (maximum_routed_tokens,),
            buffer_space=("shared" if self.pull_gather else "local"),  # [b10 G] destinations read it
        )
        if self.pull_gather:
            workspace.register(
                self.pull_message_region,
                cutlass.Int64,
                (self.world_size * self.experts_per_rank,),
                buffer_space="shared",
                reset="tail_reset",
            )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.sorted_scores_region,
                cutlass.Float32,
                (maximum_routed_tokens,),
                buffer_space="local",
            )
        workspace.register(
            self.token_src_metadata_region,
            cutlass.Int64,
            (self.worst_case_token_count,),
            buffer_space="shared",
            byte_alignment=16,
        )
        if self.apply_topk_at_fc1:
            workspace.register(
                self.fc1_topk_scores_region,
                cutlass.Float32,
                (self.worst_case_token_count,),
                buffer_space="shared",
            )
        if self.dedup_tokens_per_slot:
            workspace.register(
                self.dedup_dup_count_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
            workspace.register(
                self.dedup_enabled_region, cutlass.Int32, (1,), buffer_space="local", reset="tail_reset"
            )
            workspace.register(
                self.dedup_source_mask_region, cutlass.Int32, (1,), buffer_space="shared", reset="tail_reset"
            )
            # s25-dedup. A source with dedup on rewrites the link of every pool row it pushes, each launch; rows from
            # sources whose mask bit is clear are never read as links (no reset needed).
            workspace.register(
                self.sorted_link_region,
                cutlass.Int32,
                (maximum_routed_tokens,),
                buffer_space="local",
            )

            workspace.register(
                self.token_src_link_region,
                cutlass.Int64,
                (self.worst_case_token_count,),
                buffer_space="shared",
                byte_alignment=16,
            )
        workspace.register(
            self.pool_expert_base_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            buffer_space="local",
        )
        workspace.register(
            self.source_expert_base_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.push_destination_base_region,
            cutlass.Int32,
            (self.expert_count_padded,),
            buffer_space="local",
        )
        workspace.register(
            self.sorted_metadata_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )
        workspace.register(
            self.push_table_ready_region,
            cutlass.Int32,
            (1,),
            buffer_space="local",
            reset="tail_reset",
        )
        if self.router_data_cta_count > 1:
            workspace.register(
                self.router_size_counter_region,
                cutlass.Int32,
                (self.expert_count_with_trash,),
                buffer_space="local",
                reset="tail_reset",
            )
            workspace.register(
                self.router_histogram_done_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )
            workspace.register(
                self.source_base_ready_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "_MetadataPushRouter":
        if values:
            raise ValueError("_MetadataPushRouter carries no MLIR values.")
        return self

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        """Launch counting-sort DATA, size-exchange HELPER, and metadata PUSH roles."""
        if cutlass.const_expr(self.apply_topk_at_fc1 and topk_scores is None):
            raise ValueError("apply_topk_at_fc1 requires router topk_scores.")
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_object()
        self._router_kernel(
            topk_indices,
            topk_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper,
            device_workspace,
        ).launch(
            grid=[self.router_grid_cta_count, 1, 1],
            block=[self.router_warps_per_cta * 32, 1, 1],
            min_blocks_per_mp=1,
            stream=stream,
            use_pdl=bool(self.quant_overlap),
        )

    @cute.kernel
    def _router_kernel(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper: SymmetricBufferDevice,
        device_workspace: DeviceWorkspace,
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        linear_cta_idx, _, _ = cute.arch.block_idx()
        cute.arch.griddepcontrol_launch_dependents()
        block_thread_count = self.router_warps_per_cta * 32
        grid_thread_idx = thread_idx + linear_cta_idx * block_thread_count
        warp_idx = cute.arch.make_warp_uniform(thread_idx // Int32(32))
        lane_idx = thread_idx % Int32(32)

        storage_type = self._router_smem_workspace.storage_class()
        smem_allocator = cutlass.utils.SmemAllocator()
        storage = smem_allocator.allocate(storage_type)
        smem_base = storage.buffer.data_ptr()

        device_workspace.assign_device_members(local_workspace, shared_workspace)
        if cutlass.const_expr(_epoch.enabled()):
            # v33-epoch: router e starts after MegaMoE e - 1 completed (plain launch), so it reads the same epoch as
            # MegaMoE e (which advances it only after every rank's router e contributions were consumed).
            device_workspace.load_epoch()
        self._device_workspace = device_workspace
        self._peer_rank_ptr_mapper = peer_rank_ptr_mapper
        self._router_local_rank = local_rank
        self._router_thread_idx = thread_idx
        self._router_linear_cta_idx = linear_cta_idx
        self._router_grid_thread_idx = grid_thread_idx
        self._router_warp_idx = warp_idx
        self._router_lane_idx = lane_idx

        # s25-dedup: links need whole tokens per DATA-CTA register round and the multi-CTA write-out; otherwise every
        # pushed link is "none".
        _epv = 128 // topk_indices.dtype.width
        self._dedup_partial = bool(
            self.dedup_tokens_per_slot
            and self.router_data_cta_count > 1
            and (self.router_warps_per_cta * 32 * _epv) % self.topk == 0
            and self.expert_count_padded < 2048
            and self.topk <= 32
            and self.world_size <= 32
            and 32 % (self.topk // _epv if self.topk % _epv == 0 else 64) == 0
        )
        if cutlass.const_expr(self.router_data_cta_count == 1):
            self._router_single_cta(topk_indices, topk_scores, smem_base)
        else:
            self._router_multiple_ctas(topk_indices, topk_scores, smem_base)
        if cutlass.const_expr(self.pull_gather):
            if linear_cta_idx < Int32(self.router_gather_cta_count):
                self._router_gather_metadata(smem_base)
        elif linear_cta_idx < Int32(self.router_push_cta_count):
            self._router_push_metadata()

        device_workspace.remove_device_members()
        self._device_workspace = None
        self._peer_rank_ptr_mapper = None
        self._router_local_rank = None
        self._router_thread_idx = None
        self._router_linear_cta_idx = None
        self._router_grid_thread_idx = None
        self._router_warp_idx = None
        self._router_lane_idx = None

    @cute.jit
    def _router_single_cta(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            trash_bucket = self.expert_count_padded
            histogram = self._router_smem_workspace.tensor(
                self.router_data_histogram_region, smem_base
            )
            prefix = self._router_smem_workspace.tensor(self.router_data_prefix_region, smem_base)
            warp_totals = self._router_smem_workspace.tensor(
                self.router_data_warp_totals_region, smem_base
            )
            sorted_elements = self._router_smem_workspace.tensor(
                self.router_data_sorted_region, smem_base
            )

            zero_round_count = ceil_div(self.expert_count_with_trash, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                expert = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_with_trash):
                    histogram[expert] = Int32(0)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            within_expert_indices = self._build_histogram(expert_registers, histogram)
            iket.range_pop()

            iket.range_push("router.prefix_and_publish")
            publish_sizes = self._broadcast_sizes_to_peers(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,)))
            )
            total_valid_routes = smem_exclusive_prefix(
                cute.make_tensor(histogram.iterator, cute.make_layout((self.expert_count_padded,))),
                cute.make_tensor(prefix.iterator, cute.make_layout((self.expert_count_padded,))),
                warp_totals,
                block_thread_count,
                self._router_thread_idx,
                self._router_lane_idx,
                self._router_warp_idx,
            )
            if self._router_thread_idx == Int32(0):
                prefix[trash_bucket] = total_valid_routes
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    source_expert_base[expert] = prefix[expert]
            cute.arch.sync_threads()
            publish_sizes()
            iket.range_pop()

            iket.range_push("router.sort")
            self._sort_router_elements(
                expert_registers,
                within_expert_indices,
                score_registers,
                sorted_elements,
                prefix,
                topk_indices.dtype,
            )
            cute.arch.sync_threads()
            iket.range_pop()
            iket.range_push("router.write_out")
            self._dump_contiguous_router_output(sorted_elements, total_valid_routes)
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )
            iket.range_pop()
        elif self._router_linear_cta_idx == Int32(self.router_data_cta_count):
            self._router_helper_single_cta(smem_base)

    @cute.jit
    def _router_multiple_ctas(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor], smem_base: cute.Pointer
    ) -> None:
        if self._router_linear_cta_idx < Int32(self.router_data_cta_count):
            block_thread_count = self.router_warps_per_cta * 32
            expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)
            histogram = self._router_smem_workspace.tensor(
                self.router_data_histogram_region, smem_base
            )
            dump_base = self._router_smem_workspace.tensor(self.router_data_base_region, smem_base)

            zero_round_count = ceil_div(self.expert_count_with_trash, block_thread_count)
            for zero_round in cutlass.range_constexpr(zero_round_count):
                expert = Int32(zero_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_with_trash):
                    histogram[expert] = Int32(0)
            if cutlass.const_expr(self._dedup_partial):
                dedup_primary = self._router_smem_workspace.tensor(self.router_data_primary_region, smem_base)
                primary_cells = cute.size(dedup_primary)
                for zero_round in cutlass.range_constexpr(ceil_div(primary_cells, block_thread_count)):
                    cell = Int32(zero_round * block_thread_count) + self._router_thread_idx
                    if cell < Int32(primary_cells):
                        dedup_primary[cell] = Int32(0x7FFFFFFF)

            iket.range_push("router.histogram")
            expert_registers, score_registers = self._load_router_inputs(topk_indices, topk_scores)
            cute.arch.sync_threads()
            within_expert_indices = self._build_histogram(expert_registers, histogram)
            if cutlass.const_expr(self._dedup_partial):
                self._dedup_count_duplicates(expert_registers, topk_indices.dtype)  # before histogram_done's release
            if cutlass.const_expr(self._dedup_partial):
                # s25-dedup: pack (expert, within-expert index) into one register per element for the write-out, which
                # carries two more stores per element (32 elements / lane at 16384 elements / CTA would spill)
                for register_idx in cutlass.range_constexpr(cute.size(expert_registers)):
                    within_expert_indices[register_idx] = expert_registers[register_idx] | (
                        within_expert_indices[register_idx] << Int32(11)
                    )
                expert_registers = None
            iket.range_pop()

            iket.range_push("router.reserve")
            size_counter = self._device_workspace.ptr(self.router_size_counter_region)
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    dump_base[expert] = Int32(
                        cute.arch.atomic_add(
                            size_counter + expert, histogram[expert], sem="relaxed", scope="gpu"
                        )
                    )
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.router_histogram_done_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )
            iket.range_pop()

            iket.range_push("router.wait_source_base")
            source_base_ready = self._device_workspace.ptr(self.source_base_ready_region)
            if self._router_thread_idx == Int32(0):
                while cute.arch.load(source_base_ready, Int32, sem="acquire", scope="gpu") != Int32(
                    1
                ):
                    nanosleep(150)
            cute.arch.sync_threads()
            iket.range_pop()

            iket.range_push("router.write_out")
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            dedup_reserve = None
            if cutlass.const_expr(self._dedup_partial):
                dedup_reserve = (
                    self._router_smem_workspace.tensor(self.router_data_reserve_region, smem_base),
                    self._router_smem_workspace.tensor(self.router_data_primary_region, smem_base),
                )
            for expert_round in cutlass.range_constexpr(expert_round_count):
                expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
                if expert < Int32(self.expert_count_padded):
                    if cutlass.const_expr(self._dedup_partial):
                        dedup_reserve[0][expert] = dump_base[expert]  # this CTA's offset inside the source's expert run
                    dump_base[expert] = dump_base[expert] + source_expert_base[expert]
            cute.arch.sync_threads()
            self._scatter_router_output(
                expert_registers, within_expert_indices, score_registers, dump_base, topk_indices.dtype,
                dedup_reserve,
            )
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sorted_metadata_ready_region),
                    Int32(1),
                    sem="release",
                    scope="gpu",
                )
            iket.range_pop()
        elif self._router_linear_cta_idx == Int32(self.router_data_cta_count):
            if cutlass.const_expr(self.pull_gather):
                self._router_helper_pull(smem_base)
            else:
                self._router_helper_multiple_ctas(smem_base)

    @cute.jit
    def _router_helper_single_cta(self, smem_base: cute.Pointer) -> None:
        iket.range_push("router.compute_push_tables")
        self._compute_push_tables(smem_base)
        iket.range_pop()

    @cute.jit
    def _router_helper_multiple_ctas(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        totals = self._router_smem_workspace.tensor(self.router_helper_totals_region, smem_base)
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(
            self.router_helper_warp_totals_region, smem_base
        )
        size_counter = self._device_workspace.tensor(self.router_size_counter_region)
        source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
        expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)

        histogram_done = self._device_workspace.ptr(self.router_histogram_done_region)
        iket.range_push("router.wait_histogram")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(histogram_done, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(150)
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.broadcast_sizes")
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                totals[expert] = size_counter[expert]
        cute.arch.sync_threads()

        publish_sizes = self._broadcast_sizes_to_peers(totals)
        total_routes = smem_exclusive_prefix(
            totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        dedup_on = Int32(0)
        if cutlass.const_expr(self._dedup_partial):
            # v5 runtime gate (every helper thread computes it; the DATA CTAs' counts precede histogram_done)
            duplicates = cute.arch.load(self._device_workspace.ptr(self.dedup_dup_count_region), Int32)
            if Int64(duplicates) * Int64(1000) >= Int64(total_routes) * Int64(self.dedup_min_x1000):
                if total_routes > Int32(0):
                    dedup_on = Int32(1)
            if self._router_thread_idx == Int32(0):
                cute.arch.store(self._device_workspace.ptr(self.dedup_enabled_region), dedup_on)
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                source_expert_base[expert] = prefix[expert]
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.source_base_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )
        publish_sizes(dedup_on)
        iket.range_pop()

        iket.range_push("router.compute_push_tables")
        self._compute_push_tables(smem_base)
        iket.range_pop()

    @cute.jit
    def _router_helper_pull(self, smem_base: cute.Pointer) -> None:
        """[b10 G] Helper of the pull-gather router (multi-DATA-CTA geometry).

        Same local prefix as the push helper (source_expert_base, source_base_ready). Then, once the DATA CTAs'
        sorted metadata is written, one system fence and one 8-byte message per (destination, destination-local
        expert): valid bit | this source's sorted base for that expert << 24 | its route count. The message replaces
        the size-row broadcast, the push tables and the metadata push: every destination reads the sizes it needs
        from the 16 messages it receives and gathers the metadata itself (_router_gather_metadata). Pool layout,
        row order within each (expert, source) run and the metadata bytes are the push design's, so the pools and
        every output are bit-identical."""
        block_thread_count = self.router_warps_per_cta * 32
        epr = self.experts_per_rank
        world = self.world_size
        me = Int32(self._router_local_rank)
        totals = self._router_smem_workspace.tensor(self.router_helper_totals_region, smem_base)
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(self.router_helper_warp_totals_region, smem_base)
        table = self._router_smem_workspace.tensor(self.router_helper_size_matrix_region, smem_base)
        size_counter = self._device_workspace.tensor(self.router_size_counter_region)
        source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
        expert_round_count = ceil_div(self.expert_count_padded, block_thread_count)

        histogram_done = self._device_workspace.ptr(self.router_histogram_done_region)
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(histogram_done, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(32)
        cute.arch.sync_threads()
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                totals[expert] = size_counter[expert]
        cute.arch.sync_threads()
        smem_exclusive_prefix(
            totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        for expert_round in cutlass.range_constexpr(expert_round_count):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                source_expert_base[expert] = prefix[expert]
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.source_base_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )

        # The DATA CTAs' scatter (sorted metadata) must be visible to every peer before it sees our message.
        sorted_metadata_ready = self._device_workspace.ptr(self.sorted_metadata_ready_region)
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sorted_metadata_ready, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(32)
            if cutlass.const_expr(self.quant_overlap):
                cute.arch.griddepcontrol_wait()  # [b10 Q] the quantized rows peers will pull are complete
            cute.arch.fence_acq_rel_sys()
        cute.arch.sync_threads()

        message_address = self._device_workspace.ptr(self.pull_message_region).toint()
        pair_count = world * epr
        pair_round_count = ceil_div(pair_count, block_thread_count)
        for pair_round in cutlass.range_constexpr(pair_round_count):
            pair = Int32(pair_round * block_thread_count) + self._router_thread_idx
            if pair < Int32(pair_count):
                destination_rank = pair // Int32(epr)
                local_expert = pair - destination_rank * Int32(epr)
                expert = pair  # destination-major global expert id
                word = (
                    Int64(1 << 62)
                    | (Int64(prefix[expert]) << Int64(24))
                    | Int64(totals[expert])
                )
                st_relaxed_sys_b64(
                    self._peer_rank_ptr_mapper.map(
                        Int64(message_address)
                        + Int64(me * Int32(epr) + local_expert) * Int64(8),
                        destination_rank,
                        Int64(0),
                    ),
                    word,
                )


    @cute.jit
    def _router_gather_metadata(self, smem_base: cute.Pointer) -> None:
        """[b10 G] Gather role (router CTAs < router_gather_cta_count, after their DATA / helper work).

        Each gather CTA reads the 16 sources' messages for this rank's local experts into SMEM and derives the pool
        layout itself (expert totals, padded pool-base prefix: the push tables' formula; ring order within an expert
        as the push). CTA 0 also publishes sizes, pool bases and the adaptive-dispatch size columns, then sizes_ready.
        Then one warp per (local expert, ring position) task copies that source's sorted run for the expert
        (contiguous on the source) into this rank's pool metadata (contiguous here); a warp owns every (gather
        warps)-th task and loads the first 32 routes of all its tasks before any store (one NVLink round trip at small
        T); longer runs continue four 32-route chunks at a time. Each CTA then signals this rank's metadata_ready.
        Same slots and bytes as the push."""
        block_thread_count = self.router_warps_per_cta * 32
        world = self.world_size
        epr = self.experts_per_rank
        pair_count = world * epr
        pair_round_count = ceil_div(pair_count, block_thread_count)
        warp_count = self.router_gather_cta_count * self.router_warps_per_cta
        tasks_per_warp = ceil_div(pair_count, warp_count)
        me = Int32(self._router_local_rank)
        lane = self._router_lane_idx
        owner_expert_begin = me * Int32(epr)
        is_publisher = self._router_linear_cta_idx == Int32(0)
        table = self._router_smem_workspace.tensor(self.router_gather_table_region, smem_base)
        count_base = 0
        source_base = pair_count
        pool_prefix_base = 2 * pair_count
        dense_prefix_base = 2 * pair_count + epr
        total_slot = 2 * pair_count + 2 * epr
        ring_offset_base = 2 * pair_count + 2 * epr + 1

        # 1. the 16 sources' messages for our local experts
        message_address = self._device_workspace.ptr(self.pull_message_region).toint()
        sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
        for pair_round in cutlass.range_constexpr(pair_round_count):
            pair = Int32(pair_round * block_thread_count) + self._router_thread_idx
            if pair < Int32(pair_count):
                word_address = Int64(message_address) + Int64(pair) * Int64(8)
                word = ld_acquire_sys_b64(word_address)
                while word == Int64(0):
                    nanosleep(32)
                    word = ld_acquire_sys_b64(word_address)
                count = Int32(word & Int64(0xFFFFFF))
                table[Int32(count_base) + pair] = count
                table[Int32(source_base) + pair] = Int32((word >> Int64(24)) & Int64(0x3FFFFFFF))
                if is_publisher:
                    source_rank = pair // Int32(epr)
                    sizes_by_rank[source_rank, owner_expert_begin + (pair - source_rank * Int32(epr))] = count
        cute.arch.sync_threads()

        # 2a. ring offset of every task (task = local expert * world + ring position): a segmented scan of the counts in
        # ring order over each expert's 16 lanes (a warp covers two experts per round)
        if cutlass.const_expr(world == 16):
            for pair_round in cutlass.range_constexpr(pair_round_count):
                task = Int32(pair_round * block_thread_count) + self._router_thread_idx
                local_expert = task // Int32(world)
                ring_position = task - local_expert * Int32(world)
                count = Int32(0)
                if task < Int32(pair_count):
                    count = table[Int32(count_base) + ((me + ring_position) % Int32(world)) * Int32(epr) + local_expert]
                inclusive = count
                for step in cutlass.range_constexpr(4):
                    up = Int32(cute.arch.shuffle_sync_up(inclusive, 1 << step))
                    if (lane % Int32(16)) >= Int32(1 << step):
                        inclusive = inclusive + up
                if task < Int32(pair_count):
                    table[Int32(ring_offset_base) + task] = inclusive - count
        else:
            for pair_round in cutlass.range_constexpr(pair_round_count):
                task = Int32(pair_round * block_thread_count) + self._router_thread_idx
                local_expert = task // Int32(world)
                if task < Int32(pair_count):
                    ring_position = task - local_expert * Int32(world)
                    ring_offset = Int32(0)
                    for earlier in cutlass.range_constexpr(world - 1):
                        if Int32(earlier) < ring_position:
                            ring_offset = ring_offset + table[
                                Int32(count_base) + ((me + Int32(earlier)) % Int32(world)) * Int32(epr) + local_expert
                            ]
                    table[Int32(ring_offset_base) + task] = ring_offset

        # 2b. expert totals and the two exclusive scans (warp 0, lanes own experts l, l + 32, ...)
        sizes = self._device_workspace.tensor(self.sizes_region)
        pool_expert_base = self._device_workspace.tensor(self.pool_expert_base_region)
        if self._router_warp_idx == Int32(0):
            padded_carry = Int32(0)
            dense_carry = Int32(0)
            for chunk in cutlass.range_constexpr(ceil_div(epr, 32)):
                local_expert = Int32(chunk * 32) + lane
                expert_total = Int32(0)
                if local_expert < Int32(epr):
                    for source_rank in cutlass.range_constexpr(world):
                        expert_total = expert_total + table[Int32(count_base + source_rank * epr) + local_expert]
                padded = (
                    (expert_total + Int32(self.token_padding_block - 1)) // Int32(self.token_padding_block)
                ) * Int32(self.token_padding_block)
                padded_scan = padded
                dense_scan = expert_total
                for step in cutlass.range_constexpr(5):
                    padded_up = Int32(cute.arch.shuffle_sync_up(padded_scan, 1 << step))
                    dense_up = Int32(cute.arch.shuffle_sync_up(dense_scan, 1 << step))
                    if lane >= Int32(1 << step):
                        padded_scan = padded_scan + padded_up
                        dense_scan = dense_scan + dense_up
                if local_expert < Int32(epr):
                    pool_base = padded_carry + padded_scan - padded
                    table[Int32(pool_prefix_base) + local_expert] = pool_base
                    table[Int32(dense_prefix_base) + local_expert] = dense_carry + dense_scan - expert_total
                    if is_publisher:
                        sizes[owner_expert_begin + local_expert] = expert_total
                        pool_expert_base[local_expert] = pool_base
                padded_carry = padded_carry + Int32(cute.arch.shuffle_sync(padded_scan, 31))
                dense_carry = dense_carry + Int32(cute.arch.shuffle_sync(dense_scan, 31))
            if lane == Int32(0):
                table[Int32(total_slot)] = dense_carry
        cute.arch.sync_threads()
        if is_publisher:
            if self._router_thread_idx == Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.sizes_ready_region),
                    Int32(world),
                    sem="release",
                    scope="gpu",
                )

        # 3. this warp's tasks (task = local expert * world + ring position; source = (me + ring position) % world)
        source_metadata = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        local_metadata = self._device_workspace.ptr(self.token_src_metadata_region).toint()
        warp_global = self._router_linear_cta_idx * Int32(self.router_warps_per_cta) + self._router_warp_idx
        counts = []
        sources = []
        destinations = []
        for k in cutlass.range_constexpr(tasks_per_warp):
            task = cutlass.min(warp_global + Int32(k * warp_count), Int32(pair_count - 1))
            live = Int32(warp_global + Int32(k * warp_count) < Int32(pair_count))
            local_expert = task // Int32(world)
            ring_position = task - local_expert * Int32(world)
            source_rank = (me + ring_position) % Int32(world)
            ring_offset = table[Int32(ring_offset_base) + task]
            pair = source_rank * Int32(epr) + local_expert
            counts.append(table[Int32(count_base) + pair] * live)
            sources.append(
                self._peer_rank_ptr_mapper.map(Int64(source_metadata), source_rank, Int64(0))
                + Int64(table[Int32(source_base) + pair]) * Int64(TokenSrcMetadata.nbytes)
            )
            destinations.append(
                Int64(local_metadata)
                + Int64(table[Int32(pool_prefix_base) + local_expert] + ring_offset) * Int64(TokenSrcMetadata.nbytes)
            )
        values = []
        for k in cutlass.range_constexpr(tasks_per_warp):
            values.append(
                ldg_nl1_b64_pred(
                    sources[k] + Int64(lane) * Int64(TokenSrcMetadata.nbytes), Int32(lane < counts[k])
                )
            )
        for k in cutlass.range_constexpr(tasks_per_warp):
            stg_b64(
                destinations[k] + Int64(lane) * Int64(TokenSrcMetadata.nbytes), values[k], Int32(lane < counts[k])
            )
        for k in cutlass.range_constexpr(tasks_per_warp):
            # runs longer than 32 routes (larger buckets): four 32-route chunks in flight per step
            step_count = (cutlass.max(counts[k] - Int32(32), Int32(0)) + Int32(127)) // Int32(128)
            for step in cutlass.range(step_count, unroll=1):
                chunk_values = []
                for sub in cutlass.range_constexpr(4):
                    route = Int32(32) + step * Int32(128) + Int32(sub * 32) + lane
                    chunk_values.append(
                        ldg_nl1_b64_pred(
                            sources[k] + Int64(route) * Int64(TokenSrcMetadata.nbytes), Int32(route < counts[k])
                        )
                    )
                for sub in cutlass.range_constexpr(4):
                    route = Int32(32) + step * Int32(128) + Int32(sub * 32) + lane
                    stg_b64(
                        destinations[k] + Int64(route) * Int64(TokenSrcMetadata.nbytes),
                        chunk_values[sub],
                        Int32(route < counts[k]),
                    )
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.metadata_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )

    @cute.jit
    def _router_push_metadata(self) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        sorted_metadata_ready = self._device_workspace.ptr(self.sorted_metadata_ready_region)
        push_table_ready = self._device_workspace.ptr(self.push_table_ready_region)
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sorted_metadata_ready, Int32, sem="acquire", scope="gpu") != Int32(
                self.router_data_cta_count
            ):
                nanosleep(150)
            while cute.arch.load(push_table_ready, Int32, sem="acquire", scope="gpu") != Int32(1):
                nanosleep(150)
        cute.arch.sync_threads()

        global_expert = (
            self._router_linear_cta_idx * Int32(self.router_warps_per_cta) + self._router_warp_idx
        )
        if global_expert < Int32(self.expert_count):
            sizes_by_rank = self._device_workspace.tensor(self.sizes_by_rank_region)
            source_expert_base = self._device_workspace.tensor(self.source_expert_base_region)
            push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
            route_count = sizes_by_rank[self._router_local_rank, global_expert]
            source_begin = source_expert_base[global_expert]
            destination_begin = push_destination_base[global_expert]
            destination_rank = global_expert // Int32(self.experts_per_rank)
            peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
            source_metadata = self._device_workspace.ptr(self.sorted_metadata_region)
            destination_metadata_address = (
                self._device_workspace.ptr(self.token_src_metadata_region).toint() + peer_offset
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                source_scores = self._device_workspace.ptr(self.sorted_scores_region)
                destination_scores_address = (
                    self._device_workspace.ptr(self.fc1_topk_scores_region).toint() + peer_offset
                )
            push_links = Int32(0)
            if cutlass.const_expr(self._dedup_partial):
                # v5: links only when this source turned dedup on (helper, before push_table_ready which we acquired)
                push_links = cute.arch.load(self._device_workspace.ptr(self.dedup_enabled_region), Int32)
            if cutlass.const_expr(self.dedup_tokens_per_slot):
                destination_link_address = (
                    self._device_workspace.ptr(self.token_src_link_region).toint() + peer_offset
                )
                if cutlass.const_expr(self._dedup_partial):
                    dedup_tables = self._dedup_destination_tables(destination_rank, push_links)
                    source_links = self._device_workspace.ptr(self.sorted_link_region)
            route_round_count = (route_count + Int32(31)) // Int32(32)
            for route_round in cutlass.range(route_round_count, unroll=1):
                route = Int32(route_round) * Int32(32) + self._router_lane_idx
                if cutlass.const_expr(self._dedup_partial):
                    # every lane runs the table shuffles (lanes past route_count carry "no link")
                    partial_link = Int32(-1)
                    if route < route_count:
                        if push_links != Int32(0):
                            partial_link = cute.arch.load(source_links + (source_begin + route), Int32)
                    final_link = self._dedup_final_link(partial_link, dedup_tables)
                if route < route_count:
                    source_position = source_begin + route
                    destination_position = destination_begin + route
                    metadata = cute.arch.load(source_metadata + source_position, cutlass.Int64)
                    stg_b64(
                        destination_metadata_address
                        + Int64(destination_position) * Int64(TokenSrcMetadata.nbytes),
                        metadata,
                    )
                    if cutlass.const_expr(self._dedup_partial):
                        if push_links != Int32(0):
                            stg_b64(destination_link_address + Int64(destination_position) * Int64(8), final_link)
                    if cutlass.const_expr(self.apply_topk_at_fc1):
                        score = cute.arch.load(source_scores + source_position, cutlass.Float32)
                        stg_f32(
                            destination_scores_address + Int64(destination_position) * Int64(4),
                            score,
                        )

        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.fence_acq_rel_sys()
        # Keep notifier threads behind the leader's system fence.
        cute.arch.sync_threads()
        metadata_ready_address = self._device_workspace.ptr(self.metadata_ready_region).toint()
        rank_round_count = ceil_div(self.world_size, block_thread_count)
        for rank_round in cutlass.range_constexpr(rank_round_count):
            destination_rank = Int32(rank_round * block_thread_count) + self._router_thread_idx
            if destination_rank < Int32(self.world_size):
                red_add_relaxed_sys_s32(
                    self._peer_rank_ptr_mapper.map(
                        metadata_ready_address, destination_rank, Int64(0)
                    ),
                    Int32(1),
                )

    @cute.jit
    def _dedup_destination_tables(self, destination_rank: Int32, push_links: Int32) -> list:
        """s25-dedup push: per-lane exclusive prefixes over the destination's local experts (lane l holds experts l,
        l + 32, ...): pool base (token_padding_block-padded totals = the destination's pool_expert_base), FC1-ready slot
        base (ceil(total / tokens_per_slot), = token_in's cursor_slot_begin), the total itself and this source's
        push_destination_base (so the route loop needs no dependent global load). Totals are complete (the helper
        waited sizes_ready before releasing push_table_ready). Warp-collective."""
        epr = self.experts_per_rank
        chunks = ceil_div(epr, 32)
        lane = self._router_lane_idx
        sizes = self._device_workspace.tensor(self.sizes_region)
        tps = Int32(self.dedup_tokens_per_slot)
        block = Int32(self.token_padding_block)
        push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
        tables = []
        carry_pool = Int32(0)
        carry_slot = Int32(0)
        for chunk in cutlass.range_constexpr(chunks):
            local_expert = Int32(chunk * 32) + lane
            total = Int32(0)
            push_base = Int32(0)
            if local_expert < Int32(epr):
                if push_links != Int32(0):
                    total = sizes[destination_rank * Int32(epr) + local_expert]
                    push_base = push_destination_base[destination_rank * Int32(epr) + local_expert]
            padded = (total + block - Int32(1)) // block * block
            slots = (total + tps - Int32(1)) // tps
            inclusive_pool = padded
            inclusive_slot = slots
            for step in cutlass.range_constexpr(5):
                offset = 1 << step
                up_pool = Int32(cute.arch.shuffle_sync(inclusive_pool, cutlass.max(lane - Int32(offset), Int32(0))))
                up_slot = Int32(cute.arch.shuffle_sync(inclusive_slot, cutlass.max(lane - Int32(offset), Int32(0))))
                has_up = Int32(lane >= Int32(offset))
                inclusive_pool = inclusive_pool + up_pool * has_up
                inclusive_slot = inclusive_slot + up_slot * has_up
            tables.append(
                (carry_pool + inclusive_pool - padded, carry_slot + inclusive_slot - slots, total, push_base)
            )
            carry_pool = carry_pool + Int32(cute.arch.shuffle_sync(inclusive_pool, Int32(31)))
            carry_slot = carry_slot + Int32(cute.arch.shuffle_sync(inclusive_slot, Int32(31)))
        return tables

    @cute.jit
    def _dedup_final_link(self, partial_link: Int32, tables: list) -> Int64:
        """s25-dedup push: source link (primary local expert, run position) -> destination link, an int64 in the
        destination's own index spaces: pool row of the primary (bits 0-25), its FC1-ready slot (26-45) and that slot's
        row count = the ready target (46-55); -1 = this row is not a secondary. Warp-collective (shuffles)."""
        epr = self.experts_per_rank
        tps = Int32(self.dedup_tokens_per_slot)
        secondary = Int32(partial_link >= Int32(0))
        primary_local = ((partial_link >> Int32(22)) & Int32(0xFF)) * secondary
        run_position = partial_link & Int32(0x3FFFFF)
        source_lane = primary_local & Int32(31)
        pool_base = Int32(0)
        slot_base = Int32(0)
        total = Int32(0)
        push_base = Int32(0)
        for chunk in cutlass.range_constexpr(len(tables)):
            chunk_pool, chunk_slot, chunk_total, chunk_push = tables[chunk]
            in_chunk = Int32((primary_local >> Int32(5)) == Int32(chunk))
            pool_base = pool_base + Int32(cute.arch.shuffle_sync(chunk_pool, source_lane)) * in_chunk
            slot_base = slot_base + Int32(cute.arch.shuffle_sync(chunk_slot, source_lane)) * in_chunk
            total = total + Int32(cute.arch.shuffle_sync(chunk_total, source_lane)) * in_chunk
            push_base = push_base + Int32(cute.arch.shuffle_sync(chunk_push, source_lane)) * in_chunk
        pool_row = push_base + run_position
        in_expert = pool_row - pool_base
        slot_in_expert = in_expert // tps
        target = cutlass.min(tps, total - slot_in_expert * tps)
        link = (
            Int64(pool_row) | (Int64(slot_base + slot_in_expert) << Int64(26)) | (Int64(target) << Int64(46))
        )
        return link * Int64(secondary) - Int64(1 - secondary)

    @cute.jit
    def _broadcast_sizes_to_peers(self, smem_expert_counts: cute.Tensor) -> Callable[[], None]:
        block_thread_count = self.router_warps_per_cta * 32
        row_bytes = Int32(self.expert_count_padded * 4)
        matrix_address = self._device_workspace.ptr(self.sizes_by_rank_region).toint()
        total_address = self._device_workspace.ptr(self.sizes_region).toint()
        rank_round_count = ceil_div(self.world_size, self.router_warps_per_cta)
        for rank_round in cutlass.range_constexpr(rank_round_count):
            destination_rank = self._router_warp_idx + Int32(rank_round * self.router_warps_per_cta)
            if destination_rank < Int32(self.world_size):
                peer_offset = self._peer_rank_ptr_mapper.map(Int64(0), destination_rank, Int64(0))
                destination_row_address = (
                    matrix_address
                    + peer_offset
                    + Int64(Int32(self._router_local_rank) * Int32(self.expert_count_padded))
                    * Int64(4)
                )
                destination_row = cute.make_ptr(
                    cutlass.Int32, destination_row_address, AddressSpace.gmem, assumed_align=16
                )
                destination_total = cute.make_ptr(
                    cutlass.Int32, total_address + peer_offset, AddressSpace.gmem, assumed_align=16
                )
                with cute.arch.elect_one():
                    cp_async_bulk_s2g(destination_row, smem_expert_counts.iterator, row_bytes)
                    cp_reduce_async_bulk_add_u32_s2g(
                        destination_total, smem_expert_counts.iterator, row_bytes
                    )
                    cute.arch.cp_async_bulk_commit_group()

        def finalize(dedup_on=None) -> None:
            cute.arch.cp_async_bulk_wait_group(0)
            if cutlass.const_expr(dedup_on is not None and self._dedup_partial):
                # s25-dedup v5: tell every destination that this source's pool rows carry links this launch (ordered
                # before sizes_ready by the system fence below)
                if dedup_on != Int32(0):
                    mask_address = self._device_workspace.ptr(self.dedup_source_mask_region).toint()
                    for mask_round in cutlass.range_constexpr(ceil_div(self.world_size, self.router_warps_per_cta * 32)):
                        mask_rank = Int32(mask_round * self.router_warps_per_cta * 32) + self._router_thread_idx
                        if mask_rank < Int32(self.world_size):
                            red_or_relaxed_sys_b32(
                                self._peer_rank_ptr_mapper.map(mask_address, mask_rank, Int64(0)),
                                Int32(1) << Int32(self._router_local_rank),
                            )
            cute.arch.sync_threads()
            if self._router_thread_idx == Int32(0):
                cute.arch.fence_acq_rel_sys()
            cute.arch.sync_threads()
            ready_address = self._device_workspace.ptr(self.sizes_ready_region).toint()
            ready_round_count = ceil_div(self.world_size, block_thread_count)
            for ready_round in cutlass.range_constexpr(ready_round_count):
                destination_rank = Int32(ready_round * block_thread_count) + self._router_thread_idx
                if destination_rank < Int32(self.world_size):
                    red_add_relaxed_sys_s32(
                        self._peer_rank_ptr_mapper.map(ready_address, destination_rank, Int64(0)),
                        Int32(1),
                    )

        return finalize

    @cute.jit
    def _compute_push_tables(self, smem_base: cute.Pointer) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        owner_expert_begin = Int32(self._router_local_rank) * Int32(self.experts_per_rank)
        matrix_bytes = self.world_size * self.expert_count_padded * 4

        size_matrix = self._router_smem_workspace.tensor(
            self.router_helper_size_matrix_region, smem_base
        )
        padded_totals = self._router_smem_workspace.tensor(
            self.router_helper_totals_region, smem_base
        )
        prefix = self._router_smem_workspace.tensor(self.router_helper_prefix_region, smem_base)
        warp_totals = self._router_smem_workspace.tensor(
            self.router_helper_warp_totals_region, smem_base
        )
        load_mbarrier = self._router_smem_workspace.ptr(
            self.router_helper_load_mbarrier_region, smem_base
        )
        sizes = self._device_workspace.tensor(self.sizes_region)

        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_init(load_mbarrier, 1)

        sizes_ready = self._device_workspace.ptr(self.sizes_ready_region)
        iket.range_push("router.wait_sizes_ready")
        if self._router_thread_idx == Int32(0):
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.world_size
            ):
                nanosleep(150)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        iket.range_pop()

        iket.range_push("router.load_sizes_and_prefix")
        if self._router_thread_idx == Int32(0):
            cute.arch.mbarrier_arrive_and_expect_tx(load_mbarrier, Int32(matrix_bytes))
            tma_load_1d(
                size_matrix.iterator,
                self._device_workspace.ptr(self.sizes_by_rank_region),
                load_mbarrier,
                Int32(matrix_bytes),
            )

        padded_expert_rounds = ceil_div(self.expert_count_padded, block_thread_count)
        for expert_round in cutlass.range_constexpr(padded_expert_rounds):
            expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if expert < Int32(self.expert_count_padded):
                expert_size = sizes[expert]
                padded_totals[expert] = (
                    (expert_size + Int32(self.token_padding_block - 1))
                    // Int32(self.token_padding_block)
                ) * Int32(self.token_padding_block)
        cute.arch.sync_threads()
        smem_exclusive_prefix(
            padded_totals,
            prefix,
            warp_totals,
            block_thread_count,
            self._router_thread_idx,
            self._router_lane_idx,
            self._router_warp_idx,
        )
        pool_expert_base = self._device_workspace.tensor(self.pool_expert_base_region)
        local_expert_rounds = ceil_div(self.experts_per_rank, block_thread_count)
        for expert_round in cutlass.range_constexpr(local_expert_rounds):
            local_expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if local_expert < Int32(self.experts_per_rank):
                pool_expert_base[local_expert] = (
                    prefix[owner_expert_begin + local_expert] - prefix[owner_expert_begin]
                )

        cute.arch.mbarrier_wait(load_mbarrier, 0)
        iket.range_pop()

        iket.range_push("router.build_push_destinations")
        push_destination_base = self._device_workspace.tensor(self.push_destination_base_region)
        padded_expert_rounds = ceil_div(self.expert_count_padded, block_thread_count)
        for expert_round in cutlass.range_constexpr(padded_expert_rounds):
            global_expert = Int32(expert_round * block_thread_count) + self._router_thread_idx
            if global_expert < Int32(self.expert_count):
                destination_rank = global_expert // Int32(self.experts_per_rank)
                destination_expert_begin = destination_rank * Int32(self.experts_per_rank)
                destination_pool_base = prefix[global_expert] - prefix[destination_expert_begin]
                local_ring_position = (
                    Int32(self._router_local_rank) - destination_rank + Int32(self.world_size)
                ) % Int32(self.world_size)
                source_ring_offset = Int32(0)
                for ring_position in cutlass.range_constexpr(self.world_size):
                    source_rank = (destination_rank + Int32(ring_position)) % Int32(self.world_size)
                    if Int32(ring_position) < local_ring_position:
                        source_ring_offset = (
                            source_ring_offset + size_matrix[source_rank, global_expert]
                        )
                push_destination_base[global_expert] = destination_pool_base + source_ring_offset
        cute.arch.sync_threads()
        if self._router_thread_idx == Int32(0):
            cute.arch.atomic_add(
                self._device_workspace.ptr(self.push_table_ready_region),
                Int32(1),
                sem="release",
                scope="gpu",
            )
        iket.range_pop()

    @cute.jit
    def _load_router_inputs(
        self, topk_indices: cute.Tensor, topk_scores: Optional[cute.Tensor]
    ) -> Tuple[cute.Tensor, Optional[cute.Tensor]]:
        elements_per_vector = 128 // topk_indices.dtype.width
        grid_thread_count = self.router_data_cta_count * self.router_warps_per_cta * 32
        tile_span = elements_per_vector * grid_thread_count
        maximum_elements = self.max_tokens_per_rank * self.topk
        actual_token_count = Int32(self.max_tokens_per_rank)
        actual_elements = Int32(maximum_elements)
        load_round_count = ceil_div(maximum_elements, tile_span)
        elements_per_thread = load_round_count * elements_per_vector

        topk_flat = cute.make_tensor(topk_indices.iterator, cute.make_layout((maximum_elements,)))
        topk_vectors = cute.logical_divide(
            cute.zipped_divide(topk_flat, (tile_span,)), (elements_per_vector, None)
        )
        load_atom = _copy_atom(topk_indices.dtype, 128)
        expert_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Int32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices = cute.make_rmem_tensor((elements_per_thread,), topk_indices.dtype)
            raw_vectors = cute.zipped_divide(raw_indices, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        mark_alignment(
                            topk_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        raw_vectors[None, load_round],
                    )
        else:
            expert_vectors = cute.zipped_divide(expert_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        load_atom,
                        mark_alignment(
                            topk_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        expert_vectors[None, load_round],
                    )

        score_registers = None
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_registers = cute.make_rmem_tensor((elements_per_thread,), cutlass.Float32)
            scores_flat = cute.make_tensor(
                topk_scores.iterator, cute.make_layout((maximum_elements,))
            )
            score_vectors = cute.logical_divide(
                cute.zipped_divide(scores_flat, (tile_span,)), (elements_per_vector, None)
            )
            score_atom = _copy_atom(cutlass.Float32, elements_per_vector * 32)
            score_register_vectors = cute.zipped_divide(score_registers, (elements_per_vector,))
            for load_round in cutlass.range_constexpr(load_round_count):
                tile_begin = Int32(load_round * tile_span) + self._router_grid_thread_idx * Int32(
                    elements_per_vector
                )
                if tile_begin < actual_elements:
                    cute.copy(
                        score_atom,
                        mark_alignment(
                            score_vectors[(None, self._router_grid_thread_idx), load_round], 16
                        ),
                        score_register_vectors[None, load_round],
                    )

        expert_registers_u32 = cute.recast_tensor(expert_registers, cutlass.Uint32)
        if cutlass.const_expr(topk_indices.dtype.width == 64):
            raw_indices_i32 = cute.recast_tensor(raw_indices, cutlass.Int32)
        for register_idx in cutlass.range_constexpr(elements_per_thread):
            if cutlass.const_expr(topk_indices.dtype.width == 64):
                expert_registers[register_idx] = raw_indices_i32[2 * register_idx]
            token_idx, _ = self._router_value_coordinate(register_idx, topk_indices.dtype)
            is_invalid = (
                expert_registers_u32[register_idx] >= cutlass.Uint32(self.expert_count)
            ) | (token_idx >= actual_token_count)
            if is_invalid:
                expert_registers[register_idx] = Int32(self.expert_count_padded)
        return expert_registers, score_registers

    @cute.jit
    def _build_histogram(
        self, expert_registers: cute.Tensor, histogram: cute.Tensor
    ) -> cute.Tensor:
        register_count = cute.size(expert_registers)
        within_expert_indices = cute.make_rmem_tensor((register_count,), cutlass.Int32)
        for register_idx in cutlass.range_constexpr(register_count):
            within_expert_indices[register_idx] = Int32(
                cute.arch.atomic_add(
                    histogram.iterator + expert_registers[register_idx],
                    Int32(1),
                    sem="relaxed",
                    scope="cta",
                )
            )
        cute.arch.sync_threads()
        return within_expert_indices

    @cute.jit
    def _sort_router_elements(
        self,
        expert_registers: cute.Tensor,
        within_expert_indices: cute.Tensor,
        score_registers: Optional[cute.Tensor],
        sorted_elements: cute.Tensor,
        expert_run_starts: cute.Tensor,
        topk_index_type: type,
    ) -> None:
        register_count = cute.size(expert_registers)
        for register_idx in cutlass.range_constexpr(register_count):
            token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_index_type)
            flat_topk_index = token_idx * Int32(self.topk) + topk_slot
            destination = (
                expert_run_starts[expert_registers[register_idx]]
                + within_expert_indices[register_idx]
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                sorted_elements[destination] = _SortedElement(
                    flat_topk_index, score_registers[register_idx]
                ).pack()
            else:
                sorted_elements[destination] = _SortedElement(flat_topk_index, None).pack()

    @cute.jit
    def _router_value_coordinate(
        self, register_idx: int, topk_index_type: type
    ) -> Tuple[Int32, Int32]:
        elements_per_vector = 128 // topk_index_type.width
        tile_span = (
            elements_per_vector * self.router_data_cta_count * self.router_warps_per_cta * 32
        )
        flat_index = Int32(
            register_idx // elements_per_vector * tile_span + register_idx % elements_per_vector
        ) + self._router_grid_thread_idx * Int32(elements_per_vector)
        return (flat_index // Int32(self.topk), flat_index % Int32(self.topk))

    @cute.jit
    def _dump_contiguous_router_output(
        self, sorted_elements: cute.Tensor, total_valid_routes: Int32
    ) -> None:
        block_thread_count = self.router_warps_per_cta * 32
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        dump_round_count = (total_valid_routes + Int32(block_thread_count - 1)) // Int32(
            block_thread_count
        )
        for dump_round in cutlass.range(dump_round_count, unroll=4):
            position = Int32(dump_round * block_thread_count) + self._router_thread_idx
            predicate = Int32(position < total_valid_routes)
            element = _SortedElement.from_packed(sorted_elements[position])
            metadata = TokenSrcMetadata(
                src_rank=Int32(self._router_local_rank),
                src_token=(element.flat_topk_index // Int32(self.topk)),
                src_topk=(element.flat_topk_index % Int32(self.topk)),
            )
            stg_b64(
                metadata_address + Int64(position) * Int64(TokenSrcMetadata.nbytes),
                metadata.pack(),
                predicate,
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                stg_f32(score_address + Int64(position) * Int64(4), element.topk_score, predicate)

    @cute.jit
    def _scatter_router_output(
        self,
        expert_registers: cute.Tensor,
        within_expert_indices: cute.Tensor,
        score_registers: Optional[cute.Tensor],
        expert_dump_bases: cute.Tensor,
        topk_index_type: type,
        dedup_reserve: Optional[cute.Tensor] = None,
    ) -> None:
        """Multi-CTA router write-out, element-parallel: every thread stores its own routed elements to
        ``dump_base[e] + within_expert_index`` (dump_base = this rank's source base for expert ``e`` + this CTA's
        reservation inside it). The within-expert order is the histogram atomics' order, the same slot an
        expert-sorted SMEM staging copy would give the element, so no SMEM sort and no per-expert loop (896 experts /
        16 warps = 56 serial iterations per warp at ~2 routes each under the 2048-element router cap) are needed."""
        metadata_address = self._device_workspace.ptr(self.sorted_metadata_region).toint()
        if cutlass.const_expr(self.apply_topk_at_fc1):
            score_address = self._device_workspace.ptr(self.sorted_scores_region).toint()
        register_count = cute.size(within_expert_indices)
        if cutlass.const_expr(self._dedup_partial):
            link_address = self._device_workspace.ptr(self.sorted_link_region).toint()
            # helper's decision for this launch (stored before its source_base_ready release, which we acquired)
            dedup_on = cute.arch.load(self._device_workspace.ptr(self.dedup_enabled_region), Int32)
            if dedup_on != Int32(0):
                self._dedup_primary_min(within_expert_indices, dedup_reserve, topk_index_type)  # ends with a barrier
        for register_idx in cutlass.range_constexpr(register_count):
            if cutlass.const_expr(self._dedup_partial):
                # packed (expert | within << 11), see _router_multiple_ctas
                expert = within_expert_indices[register_idx] & Int32(0x7FF)
                within = within_expert_indices[register_idx] >> Int32(11)
            else:
                expert = expert_registers[register_idx]
                within = within_expert_indices[register_idx]
            predicate = Int32(expert < Int32(self.expert_count_padded))  # the trash bucket is not written out
            safe_expert = cutlass.min(expert, Int32(self.expert_count_padded - 1))
            token_idx, topk_slot = self._router_value_coordinate(register_idx, topk_index_type)
            output_position = expert_dump_bases[safe_expert] + within
            metadata = TokenSrcMetadata(
                src_rank=Int32(self._router_local_rank),
                src_token=token_idx,
                src_topk=topk_slot,
            )
            stg_b64(
                metadata_address + Int64(output_position) * Int64(TokenSrcMetadata.nbytes),
                metadata.pack(),
                predicate,
            )
            if cutlass.const_expr(self.apply_topk_at_fc1):
                stg_f32(
                    score_address + Int64(output_position) * Int64(4),
                    score_registers[register_idx],
                    predicate,
                )
            if cutlass.const_expr(self._dedup_partial):
                # s25-dedup: this element's link, by sorted position (next to its metadata)
                if dedup_on != Int32(0):
                    stg_b32_pred(
                        link_address + Int64(output_position) * Int64(4),
                        self._dedup_source_link(register_idx, expert, dedup_reserve[1], topk_index_type),
                        predicate,
                    )

    @cute.jit
    def _dedup_cell(self, register_idx: int, expert: Int32, topk_index_type: type, thread_idx: Int32) -> Int32:
        """SMEM cell (CTA-local token, destination rank) of this thread's element register_idx. Within a register round a
        DATA CTA holds block_threads x epv consecutive routed elements (_load_router_inputs), i.e. whole tokens."""
        elements_per_vector = 128 // topk_index_type.width
        block_thread_count = self.router_warps_per_cta * 32
        local_element = (
            Int32((register_idx // elements_per_vector) * block_thread_count * elements_per_vector)
            + thread_idx * Int32(elements_per_vector)
            + Int32(register_idx % elements_per_vector)
        )
        rank = cutlass.min(expert, Int32(self.expert_count - 1)) // Int32(self.experts_per_rank)
        return (local_element // Int32(self.topk)) * Int32(self.world_size) + rank

    @cute.jit
    def _dedup_count_duplicates(self, expert_registers: cute.Tensor, topk_index_type: type) -> None:
        """s25-dedup v5, DATA CTAs (histogram phase): count this CTA's routed slots whose (token, destination rank) pair
        an earlier slot of the token already covers = valid slots - distinct ranks per token. A token's slots of one
        register round sit in `group` aligned lanes: OR the rank bits / sum the valid counts over the group with xor
        shuffles, one lane per group adds. One relaxed atomic per warp."""
        elements_per_vector = 128 // topk_index_type.width
        group = self.topk // elements_per_vector
        epr = self.experts_per_rank
        lane = self._router_lane_idx
        duplicates = Int32(0)
        for round_idx in cutlass.range_constexpr(cute.size(expert_registers) // elements_per_vector):
            rank_bits = Int32(0)
            valid_count = Int32(0)
            for n in cutlass.range_constexpr(elements_per_vector):
                expert = expert_registers[round_idx * elements_per_vector + n]
                valid = Int32(expert < Int32(self.expert_count))
                rank_bits = rank_bits | ((Int32(1) << (cutlass.min(expert, Int32(self.expert_count - 1)) // Int32(epr))) * valid)
                valid_count = valid_count + valid
            step = 1
            while step < group:
                rank_bits = rank_bits | Int32(cute.arch.shuffle_sync(rank_bits, lane ^ Int32(step)))
                valid_count = valid_count + Int32(cute.arch.shuffle_sync(valid_count, lane ^ Int32(step)))
                step *= 2
            if lane % Int32(group) == Int32(0):
                duplicates = duplicates + valid_count - Int32(cute.arch.popc(rank_bits))
        step = 16
        while step >= 1:
            duplicates = duplicates + Int32(cute.arch.shuffle_sync(duplicates, lane ^ Int32(step)))
            step //= 2
        if lane == Int32(0):
            if duplicates > Int32(0):
                cute.arch.atomic_add(
                    self._device_workspace.ptr(self.dedup_dup_count_region), duplicates, sem="relaxed", scope="gpu"
                )

    @cute.jit
    def _dedup_primary_min(self, packed: cute.Tensor, dedup_tables: tuple, topk_index_type: type) -> None:
        """s25-dedup, source side, pass 1 (write-out, after the reservation): for every (token, destination rank) the
        PRIMARY is the token's smallest expert on that rank, i.e. the first of its slots in the destination's
        expert-major dispatch order and the only one pulled over NVLink. One SMEM atomic min per routed element on cell
        (CTA-local token, rank) of (local expert << 22 | run position), run position = this CTA's reservation in the
        source's run of the expert + within-expert index (< 2^22: <= max_T rows per source and expert). Elements are
        packed (expert | within << 11) by _router_multiple_ctas. Ends with a CTA barrier."""
        reserve, primary = dedup_tables
        epr = self.experts_per_rank
        # A token's elements of one register index sit in `group` aligned lanes; two of them on the same rank would hit
        # the same cell in the same instruction (serialized replays: +11 us router at 35% duplicates, 8192 bucket). An
        # xor-shuffle pre-check within the group lets only the smallest such expert issue its atomic.
        group = self.topk // (128 // topk_index_type.width)
        lane = self._router_lane_idx
        for register_idx in cutlass.range_constexpr(cute.size(packed)):
            expert = packed[register_idx] & Int32(0x7FF)
            valid = Int32(expert < Int32(self.expert_count))
            rank_or_none = (expert // Int32(epr)) * valid + Int32(1 << 20) * (Int32(1) - valid)
            beaten = Int32(0)
            for x in cutlass.range_constexpr(1, group):
                other = Int32(cute.arch.shuffle_sync(expert, lane ^ Int32(x)))
                other_valid = Int32(other < Int32(self.expert_count))
                beaten = beaten | (
                    other_valid * Int32(other // Int32(epr) == rank_or_none) * Int32(other < expert)
                )
            if valid * (Int32(1) - beaten) != Int32(0):
                local_expert = expert - (expert // Int32(epr)) * Int32(epr)
                key = (local_expert << Int32(22)) | (reserve[expert] + (packed[register_idx] >> Int32(11)))
                cute.arch.atomic_min(
                    primary.iterator + self._dedup_cell(register_idx, expert, topk_index_type, self._router_thread_idx),
                    key,
                    sem="relaxed",
                    scope="cta",
                )
        cute.arch.sync_threads()

    @cute.jit
    def _dedup_source_link(
        self, register_idx: int, expert: Int32, primary: cute.Tensor, topk_index_type: type
    ) -> Int32:
        """s25-dedup, source side, pass 2 (write-out): -1 for primaries / invalid elements, else
        (1 << 30) | (primary local expert << 22) | primary run position; the push turns it into destination
        coordinates."""
        epr = self.experts_per_rank
        local_expert = expert - (expert // Int32(epr)) * Int32(epr)
        key = primary[self._dedup_cell(register_idx, expert, topk_index_type, self._router_thread_idx)]
        secondary = Int32(expert < Int32(self.expert_count)) * Int32((key >> Int32(22)) != local_expert)
        return ((Int32(1 << 30) | key) * secondary) - (Int32(1) - secondary)

    @cute.jit
    def local_expert_sizes(
        self, device_workspace: DeviceWorkspace, local_rank: Int32
    ) -> cute.Tensor:
        """Return this rank's contiguous expert-size view."""
        sizes = device_workspace.tensor(self.sizes_region)
        expert_begin = local_rank * Int32(self.experts_per_rank)
        return cute.make_tensor(
            sizes.iterator + expert_begin, cute.make_layout((self.experts_per_rank,))
        )

    @property
    def metadata_ready_target(self) -> int:
        if self.pull_gather:
            return self.router_gather_cta_count  # [b10 G] this rank's own gather CTAs
        return self.router_push_cta_count * self.world_size

    @cute.jit
    def sizes_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.sizes_region)

    @cute.jit
    def pool_expert_base_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.pool_expert_base_region)

    @cute.jit
    def token_src_metadata_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.token_src_metadata_region)

    @cute.jit
    def wait_for_sizes_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        if lane_idx == Int32(0):
            sizes_ready = device_workspace.ptr(self.sizes_ready_region)
            while cute.arch.load(sizes_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.world_size
            ):
                nanosleep(sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def wait_for_metadata_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        thread_idx, _, _ = cute.arch.thread_idx()
        lane_idx = thread_idx % Int32(32)
        if lane_idx == Int32(0):
            metadata_ready = device_workspace.ptr(self.metadata_ready_region)
            while cute.arch.load(metadata_ready, Int32, sem="acquire", scope="sys") != Int32(
                self.metadata_ready_target
            ):
                # [b10 G] a local flag on the start-up critical path: poll finely
                nanosleep(64 if self.pull_gather else sleep_cycles)
        cute.arch.sync_warp()

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return device_workspace.tensor(self.token_src_metadata_region)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.apply_topk_at_fc1):
            return None
        return device_workspace.tensor(self.fc1_topk_scores_region)


class TokenCommNonDeterministic(KernelComponent):
    """Public fused-kernel token communication component."""

    transfer_warp_count: ClassVar[int] = 4
    transfer_thread_count: ClassVar[int] = transfer_warp_count * 32
    standalone_chunk_bytes: ClassVar[int] = 2048
    minimum_pacing_window_cycles: ClassVar[int] = 512
    standalone_max_backoff_cycles: ClassVar[int] = 500
    adaptive_minimum_sleep_cycles: ClassVar[int] = 50
    transfer_lifetime_barrier_id: ClassVar[int] = 9
    grid_sync_barrier_id: ClassVar[int] = 10
    standalone_size_barrier_id: ClassVar[int] = 11
    token_in_size_barrier_id: ClassVar[int] = 12

    fc1_ready_region = "nvlink.token_comm.fc1_ready"
    fc1_activation_region = "nvlink.token_comm.fc1_activation"
    fc1_activation_sf_region = "nvlink.token_comm.fc1_activation_sf"
    fc2_done_region = "nvlink.token_comm.fc2_done"
    fc2_activation_region = "nvlink.token_comm.fc2_activation"
    fc2_activation_sf_region = "nvlink.token_comm.fc2_activation_sf"
    pre_reduced_activation_region = "nvlink.token_comm.pre_reduced_activation"
    pre_reduced_activation_sf_region = "nvlink.token_comm.pre_reduced_activation_sf"
    token_back_schedule_region = "nvlink.token_comm.token_back_schedule"  # nosec B105

    token_in_mbarrier_region = "nvlink.token_comm.main_smem.token_in_mbarriers"  # nosec B105
    token_back_mbarrier_region = "nvlink.token_comm.main_smem.token_back_mbarriers"  # nosec B105
    expert_sizes_smem_region = "nvlink.token_comm.main_smem.expert_sizes"
    token_in_activation_smem_region = "nvlink.token_comm.main_smem.token_in_activation"  # nosec B105
    token_in_sf_smem_region = "nvlink.token_comm.main_smem.token_in_sf"  # nosec B105
    token_back_activation_smem_region = "nvlink.token_comm.main_smem.token_back_activation"  # nosec B105
    token_back_sf_smem_region = "nvlink.token_comm.main_smem.token_back_sf"  # nosec B105

    @classmethod
    def problem_desc_require(cls) -> dict[str, type]:
        return {
            "world_size": int,
            "expert_count": int,
            "topk": int,
            "max_tokens_per_rank": int,
            "hidden_size": int,
            "quant_kind": str,
            "combine_format": CombineFormat,
            "apply_topk_at_fc1": bool,
        }

    @classmethod
    def impl_desc_require(cls) -> dict[str, type]:
        return {
            "token_padding_block": int,
            "sf_padding_block": int,
            "tokens_per_fc1_ready_slot": int,
            "fc2_done_signals_per_token_tile": int,
            "promised_launchable_sm_count": int,
            "token_in_flag_batch": int,
            "token_back_mode": str,
            "token_back_schedule_mode": str,
            "reduce_topk_in_kernel": bool,
            "router_smem_limit_bytes": OptionalRequirement(int),
        }

    def __init__(self, problem_desc: ProblemDesc, impl_desc: ImplDesc) -> None:
        self._validate_desc_inputs(problem_desc, impl_desc)

        self.world_size = problem_desc["world_size"]
        self.expert_count = problem_desc["expert_count"]
        self.topk = problem_desc["topk"]
        self.max_tokens_per_rank = problem_desc["max_tokens_per_rank"]
        self.hidden_size = problem_desc["hidden_size"]
        self.quant_kind = QuantKind(problem_desc["quant_kind"])
        self.combine_format = problem_desc["combine_format"]
        self.apply_topk_at_fc1 = problem_desc["apply_topk_at_fc1"]

        self.token_padding_block = impl_desc["token_padding_block"]
        self.sf_padding_block = impl_desc["sf_padding_block"]
        self.tokens_per_fc1_ready_slot = impl_desc["tokens_per_fc1_ready_slot"]
        # mma tile N = 224: activation SF rows are laid out 256 per 224-token tile (helpers.utils).
        self.sf_tile_tokens = impl_desc.get("sf_tile_tokens")
        self.fc2_done_signals_per_token_tile = impl_desc["fc2_done_signals_per_token_tile"]
        self.promised_launchable_sm_count = impl_desc["promised_launchable_sm_count"]
        self.token_in_flag_batch = impl_desc["token_in_flag_batch"]
        self.token_back_mode: TokenBackMode = impl_desc["token_back_mode"]
        self.token_back_schedule_mode: TokenBackScheduleMode = impl_desc["token_back_schedule_mode"]
        self.reduce_topk_in_kernel = impl_desc["reduce_topk_in_kernel"]
        self.router_smem_limit_bytes = impl_desc.get("router_smem_limit_bytes", 227 * 1024)
        # v12-dispatch: MEGAMOE_DISPATCH_REG_ROWS=R (0 = off = the original token_in loop) moves 1 + R rows per
        # transfer-warp round (1 TMA row + R register rows). Needs whole 8-byte words per row.
        # default 1 (adopted, every config; helpers/prod_config.py); MEGAMOE_DISPATCH_REG_ROWS=R overrides.
        self.dispatch_reg_rows = int(
            _prod_config.flag("MEGAMOE_DISPATCH_REG_ROWS", self.quant_kind, self.combine_format)
        )
        if not 0 <= self.dispatch_reg_rows <= 3:
            raise ValueError("MEGAMOE_DISPATCH_REG_ROWS must be in [0, 3].")
        # MEGAMOE_DISPATCH_LSU_ONLY=1 (with REG_ROWS=R >= 1): drop the TMA row, R register rows per round. Takes the
        # peer loads off the SM's TMA unit (diagnostic for TMA contention with the epilogue's bulk stores).
        self.dispatch_lsu_only = (
            _kimi_options.environ.get("MEGAMOE_DISPATCH_LSU_ONLY", "0") == "1" and self.dispatch_reg_rows > 0
        )
        if self.hidden_size * int(QuantKind(problem_desc["quant_kind"]).activation_dtype.width) % 64 != 0:
            self.dispatch_reg_rows = 0
            self.dispatch_lsu_only = False
        # MEGAMOE_DISPATCH_ADAPT_X100=T (default 200; 0 = off = prod1 PTX) picks the dispatch depth per rank at runtime.
        # After sizes_ready each transfer warp sums the rows this rank pulls from every peer (its local experts) and
        # compares the largest source with the mean over peers. A source above T/100 x the mean means a few source
        # ranks carry most of the dispatch (variable tokens per rank): their NVLink egress saturates, and register-row
        # peer loads (REG_ROWS >= 1) pile up in the SM's memory pipeline behind the congested links, stretching every
        # epilogue release fence on the SM 2-3x during the dispatch window. Such ranks run the same pull loop with
        # one row per round (the TMA row): the register rows' peer loads are predicated off (no transaction), their
        # stores / ready bumps are off and their index stays on row 0, so pool bytes and fc1_ready counts are
        # identical (bit-exact either way). One loop, no second code path (v1 s4-adisp compiled both loops: spills).
        self.dispatch_adapt_x100 = 0
        if self.dispatch_reg_rows > 0 and not self.dispatch_lsu_only:
            # Default 200 (2x): TMA-only wins only for extreme source imbalance (Fge-vN4k, metric 2.14-2.46: -4.6%); at
            # metric 1.33-1.75 (vU, vU2, vN2k, Fg/Fge-vU) it LOSES 1-9% (half the rows in flight per round), uniform /
            # Zipf / Gaussian routing with equal tokens sit at <= 1.06.
            self.dispatch_adapt_x100 = int(_kimi_options.environ.get("MEGAMOE_DISPATCH_ADAPT_X100", "200") or "0")
        # [b10 F] first pull round TMA-row-only (s16-edge lever R, K = 1): needs the adaptive loop above.
        self.r0tma = int(self.dispatch_adapt_x100 > 0 and _b10.on("F", self.max_tokens_per_rank))
        # [b10 T] dense-row total of the pull loop by a warp reduction
        self.dense_total_warp = _b10.on("T", self.max_tokens_per_rank)
        # v16-fc2blk: MEGAMOE_FC2_BLOCK_DONE=1 counts FC2 completion per 256-token block (the FC1-ready slot
        # index) instead of per local expert, so token-back starts a block's rows as soon as that block's FC2
        # tiles are done rather than after the whole expert. Exact: same rows, bytes and release/acquire pairs.
        self.fc2_block_done = (
            _kimi_options.environ.get("MEGAMOE_FC2_BLOCK_DONE", "0") == "1"
            and self.token_back_mode != "standalone_warps"  # nosec B105
        )
        # v26-tbtail: MEGAMOE_FC2_TAIL_BLOCKS=K -- per-256-token-block FC2 readiness ONLY for the last K local
        # experts (the experts whose FC2 tiles finish last); every other expert keeps the per-expert counter. The
        # last experts' rows then stream out as their token blocks complete instead of all waiting for the final
        # FC2 tile, which shortens the token-back tail without changing token-back's overlap with compute elsewhere
        # (the whole-kernel version, MEGAMOE_FC2_BLOCK_DONE, lost ~60 us). Block counters live after the per-expert
        # ones in the same fc2_done region. Exact: same rows / bytes / destinations, same release/acquire pairs.
        # default 3 (helpers/prod_config.py); only acts with token-back on the transfer warps
        # (reuse_dispatch_warps): epi_warps token-back never waits on fc2_done. MEGAMOE_FC2_TAIL_BLOCKS=0 disables.
        self.fc2_tail_blocks = int(
            _prod_config.fc2_tail_blocks(self.quant_kind, self.combine_format, self.max_tokens_per_rank)
        )
        if (
            self.fc2_block_done
            or self.token_back_mode == "standalone_warps"  # nosec B105
            or not 0 <= self.fc2_tail_blocks <= self.experts_per_rank
        ):
            self.fc2_tail_blocks = 0
        # MEGAMOE_S12_TBX=1: token-back rows ride along the dispatch rounds instead of waiting for the
        # whole pull loop to finish. In reuse_dispatch_warps mode the same 4 transfer warps per CTA pull dispatch
        # (~40% of the NVLink read direction at 4k) and only then push token-back (~37-40% of the write direction),
        # so the two phases never overlap and the transfer chain ends after compute (gpum: u4k window end = token-back
        # end). With TBX each dispatch round also moves one token-back chunk of a row whose expert's FC2 is complete:
        # lane 1 issues the local TMA load into a SEPARATE per-warp token-back stage and the bulk store to the source
        # rank (bulk async-groups are per thread, so the dispatch lane never waits for the remote write drain).
        # Exact: same rows, bytes, destinations and release/acquire edges as token_back(); only the timing moves.
        # The kernel drops TBX (no tag) when the extra stage would cost an AB stage (N = 256: 2.5 KB slack).
        # MEGAMOE_S12_TBX: auto (default: on for NVFP4 at mma tile N = 240, the measured win; the kernel narrows it),
        # 1 = on wherever the extra stage fits (late issue), 2 = on with the top-of-round issue order, 0 = off.
        tbx_env = _kimi_options.environ.get("MEGAMOE_S12_TBX", "auto") or "auto"
        self.tbx_auto = tbx_env == "auto"
        self.tbx_late = tbx_env != "2"  # late issue (after the dispatch round's peer loads) unless 2
        # MEGAMOE_S12_TBDB (default 1, with TBX): after dispatch, token-back ping-pongs its chunks between the
        # TBX stage and the warp's now-idle token-in stage (lane 1 waits with wait_group(1)), so one chunk's remote drain
        # overlaps the next chunk's local load: 2 chunks in flight per warp instead of 1. Even chunk counts only (NVFP4
        # FP8: 2 x 1792 B; MX FP8 rows are one chunk). u4k -2.0%, Fg-u8k -1.0%, Fge-vN4k -0.7% on top of TBX; u8k / u16k
        # neutral; 0 = single-stage token_back().
        tbx_output_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
        tbx_stage_bytes = self.bytes_per_token + (
            self.activation_sf_hidden_padded * int(self.activation_sf_dtype.width) // 8
        )
        tbx_chunk_bytes = self.bytes_per_token if tbx_stage_bytes < tbx_output_bytes else tbx_output_bytes
        self.tbx_db = (
            _kimi_options.environ.get("MEGAMOE_S12_TBDB", "1") != "0" and ceil_div(tbx_output_bytes, tbx_chunk_bytes) % 2 == 0
        )
        self.tbx = (
            tbx_env in ("auto", "1", "2")
            and self.token_back_mode == "reuse_dispatch_warps"  # nosec B105
            and self.token_back_schedule_mode == "atomic_counter"  # nosec B105
            and self.token_back_push_data
            and self.dispatch_reg_rows > 0
            and not self.dispatch_lsu_only
            and not self.fc2_block_done
            and not self.reduce_topk_in_kernel
        )

        # s25-dedup: MEGAMOE_DEDUP (unset = per-config default below; 1 = on; 0 = off). Exact dispatch dedup: a token routed to several experts on
        # one destination rank crosses NVLink once. The source router links every secondary slot of (token, rank) to
        # the primary (the token's smallest expert there); a secondary row whose primary's FC1-ready slot is already
        # complete copies the payload from the primary's local pool row instead of the peer (scales still come from
        # the peer: 4 B loads, cheap; they would need the swizzled pool SF layout otherwise). Not-yet-ready primaries
        # fall back to the peer pull, so nothing ever waits. Same bytes land in every pool row: bit-exact by
        # construction. Real routers put ~35% of (token, rank) slots on an already-visited rank at top-16 / 16 ranks
        # (bench_moe's column-major materialisation puts each token on all 16 ranks: no duplicates). Register-row
        # dispatch path only (prod1 default).
        # prod2: default OFF (unset = 0 = cand2 / prod1 PTX). Ungated v4 adds ~94 us inside MegaMoE on no-duplicate
        # variable-token steps (Fge-vU +4.3%, u8k +1.1%) while saving 1.7-3.8% on realistic routing; it ships as an
        # opt-in (MEGAMOE_DEDUP=1) until the no-duplicate overhead is removed (docs/optimizations.md "cand3 vs cand2 perf").
        # s25-dedup v5: default ON (Blackwell, buckets below) with a per-launch runtime gate: a source does
        # the dedup router work and pushes links only when >= 1/8 of its slots duplicate a (token, rank) pair
        # (MEGAMOE_DEDUP_MIN_X1000), and token_in loads links only from sources whose bit is set in this launch's mask,
        # so duplicate-free launches run prod2's dispatch path.
        # Default (measured, b6-a5..a8, gate engaged at max_T 15674): BF16 combine from the 16384 bucket (variable-token
        # steps, link-bound: NVFP4 Sge-vU -4.3 / -4.5% cold / sustained, duplicate-free Fge-vU +0.1-0.2%). FP8 combine
        # stays opt-in (MEGAMOE_DEDUP=1): at 16384 it saves 1.3-3.9% on realistic routing but the gate pass costs
        # +0.7-1.7% on duplicate-free variable-token steps (Fge-vU). Below 16384 NVFP4 / MX FP8 are compute-bound and
        # dedup's router work costs (Srnd-u8k +1-2%); BF16 combine at 8192 gains 1.5-2.9% on realistic routing but costs
        # up to +0.9% on duplicate-free steps. prod2 build 6.
        dedup_min_bucket = 16384 if int(self.combine_format.act_dtype.width) == 16 else None
        dedup_env = (_kimi_options.environ.get("MEGAMOE_DEDUP") or "").strip()
        self.dispatch_dedup = (
            (
                dedup_env == "1"
                or (
                    dedup_env == ""
                    and _prod_config.defaults_active()
                    # bucket semantics: the 16384 bucket is 8192 < max_T <= 16384 (PROD1_TOKEN_BUCKETS); the kernel's
                    # max_T can be any value inside it (e.g. a 15674-token variable-token step), so compare to the
                    # previous bucket, not to the bucket's upper bound.
                    and dedup_min_bucket is not None
                    and self.max_tokens_per_rank > dedup_min_bucket // 2
                )
            )
            and self.dispatch_reg_rows > 0
            and not self.dispatch_lsu_only
            and self.expert_count // self.world_size <= 256
            and self.tokens_per_fc1_ready_slot < 1024
        )
        # MEGAMOE_DEDUP_STATS=1 (diagnostic, own kernel tag): global transfer warp 0 of every rank prints its rows /
        # linked (secondary) rows / rows served from the local primary once per launch.
        self.dedup_stats = self.dispatch_dedup and _kimi_options.environ.get("MEGAMOE_DEDUP_STATS", "0") == "1"
        self._validate_configuration()
        self._router = _MetadataPushRouter(
            problem_desc, impl_desc, self.tokens_per_fc1_ready_slot if self.dispatch_dedup else 0
        )
        if self.dispatch_dedup and (
            self._router.worst_case_token_count >= (1 << 26) or self.max_fc1_ready_slot_count >= (1 << 20)
        ):
            raise ValueError("MEGAMOE_DEDUP: pool rows / FC1-ready slots exceed the link encoding; set MEGAMOE_DEDUP=0.")
        self._nvlink_barrier = NvlinkBarrier(
            world_size=self.world_size, barrier_id=self.grid_sync_barrier_id
        )
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def _validate_configuration(self) -> None:
        positive_fields = (
            "world_size",
            "expert_count",
            "topk",
            "max_tokens_per_rank",
            "hidden_size",
            "token_padding_block",
            "sf_padding_block",
            "tokens_per_fc1_ready_slot",
            "promised_launchable_sm_count",
            "router_smem_limit_bytes",
        )
        for field_name in positive_fields:
            value = getattr(self, field_name)
            if value <= 0:
                raise ValueError(f"{field_name} must be positive, got {value}.")
        if self.expert_count % self.world_size != 0:
            raise ValueError(
                f"expert_count must be divisible by world_size, got {self.expert_count} and {self.world_size}."
            )
        if self.expert_count > 16384:
            raise NotImplementedError("TokenComm supports at most 16384 global experts.")
        if self.topk > self.expert_count:
            raise ValueError(
                f"topk must not exceed expert_count, got {self.topk} and {self.expert_count}."
            )
        if self.token_back_mode not in ("epi_warps", "standalone_warps", "reuse_dispatch_warps"):
            raise ValueError(f"Unsupported token_back_mode {self.token_back_mode!r}.")
        if self.token_back_schedule_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"token_back_schedule_mode must be static or atomic_counter, got {self.token_back_schedule_mode!r}."
            )
        if not 1 <= self.token_in_flag_batch <= 32:
            raise ValueError(
                f"token_in_flag_batch must be in [1, 32], got {self.token_in_flag_batch}."
            )
        if self.tokens_per_fc1_ready_slot % self.token_padding_block != 0:
            raise ValueError("tokens_per_fc1_ready_slot must be divisible by token_padding_block.")
        if self.token_back_enabled and self.fc2_done_signals_per_token_tile <= 0:
            raise ValueError(
                "fc2_done_signals_per_token_tile must be positive when token-back is enabled."
            )
        if self.reduce_topk_in_kernel and self.combine_format.act_dtype is not cutlass.BFloat16:
            raise ValueError("In-kernel top-k reduction requires BF16 combine data.")
        element_block = self.activation_sf_vector_size * 4
        if self.hidden_size % element_block != 0:
            raise ValueError(
                f"{self.quant_kind} requires hidden_size divisible by {element_block}."
            )
        if self.sf_padding_block % 128 != 0:
            raise ValueError("sf_padding_block must be a multiple of 128.")

    @property
    def experts_per_rank(self) -> int:
        return self.expert_count // self.world_size

    @property
    def activation_dtype(self) -> type:
        return self.quant_kind.activation_dtype

    @property
    def activation_sf_dtype(self) -> type:
        return self.quant_kind.sf_dtype

    @property
    def activation_sf_vector_size(self) -> int:
        return self.quant_kind.sf_vec_size

    @property
    def bytes_per_token(self) -> int:
        return self.hidden_size * int(self.activation_dtype.width) // 8

    @property
    def activation_sf_hidden_padded(self) -> int:
        valid_hidden = self.hidden_size // self.activation_sf_vector_size
        elements_per_16_bytes = 128 // int(self.activation_sf_dtype.width)
        return int(round_up(valid_hidden, elements_per_16_bytes))

    @property
    def combine_sf_hidden_padded(self) -> int:
        if not self.combine_format.is_quantized:
            return 0
        valid_hidden = self.hidden_size // self.combine_format.scale_block
        elements_per_16_bytes = 128 // int(self.combine_format.scale_dtype.width)
        return int(round_up(valid_hidden, elements_per_16_bytes))

    @property
    def token_back_push_data(self) -> bool:
        return self.token_back_mode != "epi_warps"  # nosec B105

    @property
    def token_back_push_sf(self) -> bool:
        return self.combine_format.is_quantized

    @property
    def token_back_enabled(self) -> bool:
        return self.token_back_push_data or self.token_back_push_sf

    @property
    def worst_case_token_count(self) -> int:
        return self._router.worst_case_token_count

    @property
    def worst_case_sf_token_count(self) -> int:
        if self.sf_tile_tokens:
            return (
                self._router.worst_case_padded_tokens(self.sf_tile_tokens)
                // self.sf_tile_tokens
                * SF_TILE_ROWS_N224
            )
        return self._router.worst_case_padded_tokens(self.sf_padding_block)

    @property
    def max_fc1_ready_slot_count(self) -> int:
        return (
            self._router.worst_case_padded_tokens(self.tokens_per_fc1_ready_slot)
            // self.tokens_per_fc1_ready_slot
        )

    @property
    def router_smem_workspace(self) -> SmemWorkspace:
        return self._router.router_smem_workspace

    @property
    def expert_count_padded(self) -> int:
        return self._router.expert_count_padded

    @property
    def expert_count_with_trash(self) -> int:
        return self._router.expert_count_with_trash

    @property
    def router_elements_per_lane(self) -> int:
        return self._router.router_elements_per_lane

    @property
    def router_warps_per_cta(self) -> int:
        return self._router.router_warps_per_cta

    @property
    def router_data_cta_count(self) -> int:
        return self._router.router_data_cta_count

    @property
    def router_tokens_per_cta(self) -> int:
        return self._router.router_tokens_per_cta

    @property
    def router_push_cta_count(self) -> int:
        return self._router.router_push_cta_count

    @property
    def router_grid_cta_count(self) -> int:
        return self._router.router_grid_cta_count

    @property
    def metadata_ready_target(self) -> int:
        return self._router.metadata_ready_target

    def register_device_workspace(self, workspace: DeviceWorkspace) -> None:
        self._router.register_device_workspace(workspace)
        self._register_main_workspace(workspace)
        self._nvlink_barrier.register_device_workspace(workspace)

    @cute.jit
    def launch_router(
        self,
        topk_indices: cute.Tensor,
        topk_scores: Optional[cute.Tensor],
        local_rank: Int32,
        local_workspace: cute.Pointer,
        shared_workspace: cute.Pointer,
        peer_rank_ptr_mapper_host,
        device_workspace: DeviceWorkspace,
        stream: cuda.CUstream,
    ) -> None:
        self._router.launch_router(
            topk_indices,
            topk_scores,
            local_rank,
            local_workspace,
            shared_workspace,
            peer_rank_ptr_mapper_host,
            device_workspace,
            stream,
        )

    @cute.jit
    def local_expert_sizes(
        self, device_workspace: DeviceWorkspace, local_rank: Int32
    ) -> cute.Tensor:
        return self._router.local_expert_sizes(device_workspace, local_rank)

    @cute.jit
    def wait_for_sizes_ready(
        self, device_workspace: DeviceWorkspace, sleep_cycles: int = 1000
    ) -> None:
        self._router.wait_for_sizes_ready(device_workspace, sleep_cycles)

    @cute.jit
    def token_src_metadata_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return self._router.token_src_metadata_tensor(device_workspace)

    @cute.jit
    def fc1_topk_scores_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        return self._router.fc1_topk_scores_tensor(device_workspace)

    @cute.jit
    def assign_device_members(
        self,
        *,
        device_workspace: DeviceWorkspace,
        token_comm_args: TokenCommArgs,
        local_rank: Int32,
        linear_cta_idx: Int32,
    ) -> None:
        self._device_workspace = device_workspace
        self._token_comm_args = token_comm_args
        self._local_rank = local_rank
        self._linear_cta_idx = linear_cta_idx
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        self._transfer_warp_idx = cute.arch.make_warp_uniform(transfer_thread_idx // Int32(32))
        self._lane_idx = transfer_thread_idx % Int32(32)
        self._nvlink_barrier.assign_device_members(
            device_workspace, token_comm_args.peer_rank_ptr_mapper
        )

    def remove_device_members(self) -> None:
        self._nvlink_barrier.remove_device_members()
        self._device_workspace = None
        self._token_comm_args = None
        self._local_rank = None
        self._linear_cta_idx = None
        self._transfer_warp_idx = None
        self._lane_idx = None

    def __extract_mlir_values__(self) -> list:
        return []

    def __new_from_mlir_values__(self, values: list) -> "TokenCommNonDeterministic":
        if values:
            raise ValueError("TokenCommNonDeterministic carries no MLIR values.")
        return self

    def _register_main_workspace(self, workspace: DeviceWorkspace) -> None:
        """Register the fused kernel's GMEM regions. Three groups, each stating its own existence condition.

        FC1 pools, always present: the dispatched payload ``token_in`` pulls in, addressed in POOL index space
        (per-expert padded runs of tokens routed into this rank's experts, from every rank).

        Token-back machinery, iff ``token_back_enabled``: the transfer warps' counters plus the rank-local FC2
        staging they read, also in pool index space. ``epi_warps`` needs none of it, because there the FC2
        epilogue reaches the peers itself. The scale plane is registered separately from the data plane and is
        staged locally in EVERY mode: pushing scales per token would scatter one warp's 32 lanes across up to 32
        ranks and explode the NVLink request count.

        Combine plane, iff ``not reduce_topk_in_kernel``: the symmetric per-topk landing zone peers deliver into,
        addressed in (source token, source topk) space. It is the DESTINATION of the round trip whose source is
        the staging above, so its condition is deliberately independent of who performs the transfer --
        ``epi_warps`` with a separate reduce registers this plane while registering no token-back machinery at
        all. Peers reach it through the symmetric heap, so it cannot be a caller tensor: only this component
        knows the wire dtype and the padded scale stride. Its ``data`` reset keeps it out of both the host zero
        prefix and the per-launch tail reset, since every cell it exposes is rewritten each launch and the plane
        is far too large to be worth clearing.
        """
        workspace.register(
            self.fc1_ready_region,
            cutlass.Int32,
            (self.max_fc1_ready_slot_count,),
            buffer_space="local",
            reset="tail_reset",
        )
        activation_element_count = self.worst_case_token_count * self.hidden_size
        workspace.register(
            self.fc1_activation_region,
            self.activation_dtype,
            (activation_element_count,),
            buffer_space="local",
            byte_alignment=128,
        )
        activation_sf_element_count = (
            self.worst_case_sf_token_count * self.activation_sf_hidden_padded
        )
        workspace.register(
            self.fc1_activation_sf_region,
            self.activation_sf_dtype,
            (activation_sf_element_count,),
            buffer_space="local",
            byte_alignment=128,
        )

        if self.token_back_enabled:
            workspace.register(
                self.fc2_done_region,
                cutlass.Int32,
                (
                    self.max_fc1_ready_slot_count
                    if self.fc2_block_done
                    else self.experts_per_rank
                    + (self.max_fc1_ready_slot_count if self.fc2_tail_blocks else 0),
                ),
                buffer_space="local",
                reset="tail_reset",
            )
        if self.token_back_push_data:
            fc2_element_count = self.worst_case_token_count * self.hidden_size
            workspace.register(
                self.fc2_activation_region,
                self.combine_format.act_dtype,
                (fc2_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.token_back_push_sf:
            fc2_sf_element_count = self.worst_case_token_count * self.combine_sf_hidden_padded
            workspace.register(
                self.fc2_activation_sf_region,
                self.combine_format.scale_dtype,
                (fc2_sf_element_count,),
                buffer_space="local",
                byte_alignment=128,
            )
        if self.token_back_enabled and self.token_back_schedule_mode == "atomic_counter":  # nosec B105
            workspace.register(
                self.token_back_schedule_region,
                cutlass.Int32,
                (1,),
                buffer_space="local",
                reset="tail_reset",
            )

        if not self.reduce_topk_in_kernel:
            workspace.register(
                self.pre_reduced_activation_region,
                self.combine_format.act_dtype,
                (self.max_tokens_per_rank, self.topk, self.hidden_size),
                buffer_space="shared",
                mem_order=(2, 1, 0),
                byte_alignment=128,
            )
            if self.combine_format.is_quantized:
                workspace.register(
                    self.pre_reduced_activation_sf_region,
                    self.combine_format.scale_dtype,
                    (self.max_tokens_per_rank, self.topk, self.combine_sf_hidden_padded),
                    buffer_space="shared",
                    mem_order=(2, 1, 0),
                    byte_alignment=128,
                )

    def register_smem_regions(self, workspace: SmemWorkspace) -> None:
        workspace.register_mbarrier(self.token_in_mbarrier_region, self.transfer_warp_count)
        if self.token_back_enabled:
            workspace.register_mbarrier(self.token_back_mbarrier_region, self.transfer_warp_count)
        workspace.register_tensor(
            self.expert_sizes_smem_region,
            cutlass.Int32,
            (self.experts_per_rank,),
            byte_alignment=16,
        )
        transfer_overlay = workspace.create_overlay("nvlink.token_comm.main_smem.transfer")
        token_in_lifetime = transfer_overlay.add_lifetime("token_in")
        token_in_lifetime.register_tensor(
            self.token_in_activation_smem_region,
            self.activation_dtype,
            (self.transfer_warp_count, self.hidden_size),
            byte_alignment=16,
        )
        token_in_lifetime.register_tensor(
            self.token_in_sf_smem_region,
            self.activation_sf_dtype,
            (
                self.transfer_warp_count,
                (self.activation_sf_vector_size, self.activation_sf_hidden_padded),
            ),
            stride=(self.activation_sf_hidden_padded, (0, 1)),
            byte_alignment=16,
        )
        if not self.token_back_enabled:
            return

        if self.token_back_mode == "standalone_warps":  # nosec B105
            token_back_lifetime = workspace.create_overlay(
                "nvlink.token_comm.main_smem.standalone_token_back"
            ).add_lifetime("token_back")
        elif self.tbx:
            # TBX: token-back's stage lives next to (not on top of) the token-in stage. Same chunking as
            # the overlaid stage below (available_bytes_per_warp is still the token-in stage size).
            token_back_lifetime = workspace.create_overlay(
                "nvlink.token_comm.main_smem.tbx_token_back"
            ).add_lifetime("token_back")
        else:
            token_back_lifetime = transfer_overlay.add_lifetime("token_back")

        if self.token_back_mode == "standalone_warps":  # nosec B105
            available_bytes_per_warp = self.standalone_chunk_bytes
        else:
            activation_bytes = self.bytes_per_token
            sf_bytes = self.activation_sf_hidden_padded * int(self.activation_sf_dtype.width) // 8
            available_bytes_per_warp = activation_bytes + sf_bytes

        if self.token_back_push_data:
            bytes_per_output_token = (
                self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            )
            if self.token_back_mode == "standalone_warps":  # nosec B105
                chunk_bytes = self.standalone_chunk_bytes
            elif available_bytes_per_warp < bytes_per_output_token:
                chunk_bytes = self.bytes_per_token
            else:
                chunk_bytes = bytes_per_output_token
            if (
                self.token_back_mode != "standalone_warps"  # nosec B105
                and bytes_per_output_token % chunk_bytes != 0
            ):
                raise ValueError("Token-back data chunk bytes must divide one row.")
            chunk_elements = chunk_bytes * 8 // int(self.combine_format.act_dtype.width)
            token_back_lifetime.register_tensor(
                self.token_back_activation_smem_region,
                self.combine_format.act_dtype,
                (self.transfer_warp_count, chunk_elements),
                byte_alignment=16,
            )
        if self.token_back_push_sf:
            sf_row_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            if sf_row_bytes > available_bytes_per_warp:
                raise ValueError("Token-back scale row exceeds its per-warp stage.")
            token_back_lifetime.register_tensor(
                self.token_back_sf_smem_region,
                self.combine_format.scale_dtype,
                (
                    self.transfer_warp_count,
                    (self.combine_format.scale_block, self.combine_sf_hidden_padded),
                ),
                stride=(self.combine_sf_hidden_padded, (0, 1)),
                byte_alignment=16,
            )

    @cute.jit
    def fc1_ready_counter_pointer(self, device_workspace: DeviceWorkspace) -> cute.Pointer:
        return device_workspace.ptr(self.fc1_ready_region)

    @cute.jit
    def fc1_activation_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_region),
            cute.make_layout(
                (self.worst_case_token_count, self.hidden_size), stride=(self.hidden_size, 1)
            ),
        )

    @cute.jit
    def fc1_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> cute.Tensor:
        layout = tile_atom_to_shape_SF(
            (self.worst_case_sf_token_count, self.hidden_size, 1), self.activation_sf_vector_size
        )
        return cute.make_tensor(
            device_workspace.ptr(self.fc1_activation_sf_region), cute.select(layout, mode=[0, 1])
        )

    @cute.jit
    def fc2_done_counter_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_enabled):
            return None
        return device_workspace.tensor(self.fc2_done_region)

    @cute.jit
    def fc2_activation_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_push_data):
            return None
        return cute.make_tensor(
            device_workspace.ptr(self.fc2_activation_region),
            cute.make_layout(
                (self.worst_case_token_count, 1, self.hidden_size),
                stride=(self.hidden_size, self.hidden_size, 1),
            ),
        )

    @cute.jit
    def fc2_activation_sf_tensor(self, device_workspace: DeviceWorkspace) -> Optional[cute.Tensor]:
        if cutlass.const_expr(not self.token_back_push_sf):
            return None
        return cute.make_tensor(
            device_workspace.ptr(self.fc2_activation_sf_region),
            cute.make_layout(
                (
                    self.worst_case_token_count,
                    1,
                    (
                        self.combine_format.scale_block,
                        self.hidden_size // self.combine_format.scale_block,
                    ),
                ),
                stride=(self.combine_sf_hidden_padded, self.combine_sf_hidden_padded, (0, 1)),
            ),
        )

    @cute.jit
    def pre_reduced_activation_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        """The (tokens, topk, hidden) combine staging plane, or None under in-kernel top-k reduction."""
        if cutlass.const_expr(self.reduce_topk_in_kernel):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_region)

    @cute.jit
    def pre_reduced_activation_sf_tensor(
        self, device_workspace: DeviceWorkspace
    ) -> Optional[cute.Tensor]:
        """The scale plane parallel to ``pre_reduced_activation_tensor``; only quantized wire formats carry one."""
        if cutlass.const_expr(self.reduce_topk_in_kernel or not self.combine_format.is_quantized):
            return None
        return device_workspace.tensor(self.pre_reduced_activation_sf_region)

    @cute.jit
    def token_in(self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer) -> None:
        """Wait for pushed metadata, then pull activation payloads into local pools."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        global_warp_idx = self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        global_warp_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)

        sizes = self._router.sizes_tensor(self._device_workspace)
        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)

        iket.range_push("token_in.wait_sizes_ready")
        self._router.wait_for_sizes_ready(self._device_workspace)
        iket.range_pop()
        iket.range_push("token_in.stage_sizes")
        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        owner_expert_begin = self._local_rank * Int32(self.experts_per_rank)
        source_sizes = cute.make_tensor(
            sizes.iterator + owner_expert_begin, cute.make_layout((self.experts_per_rank,))
        )
        copy_elements = 4 if self.experts_per_rank % 4 == 0 else 1
        source_size_vectors = cute.zipped_divide(
            (
                mark_alignment(source_sizes, 16)
                if cutlass.const_expr(copy_elements == 4)
                else source_sizes
            ),
            (copy_elements,),
        )
        destination_size_vectors = cute.zipped_divide(owned_sizes, (copy_elements,))
        size_vector_count = cute.size(destination_size_vectors, mode=[1])
        size_copy_atom = _copy_atom(cutlass.Int32, copy_elements * 32)
        size_copy_rounds = ceil_div(size_vector_count, self.transfer_thread_count)
        size_copy_registers = cute.make_rmem_tensor(
            (copy_elements, size_copy_rounds), cutlass.Int32
        )
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
            vector_idx = Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
            if vector_idx < Int32(size_vector_count):
                cute.copy(
                    size_copy_atom,
                    source_size_vectors[None, vector_idx],
                    size_copy_registers[None, size_copy_round],
                )
        iket.range_pop()

        # rows per round: 1 + R (register rows on) or 1 (TMA row only); warp-uniform.
        use_reg_rows = None
        if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
            # Warp-uniform (uniform register): keeps the kernel's register allocation, and so its 12 / 20 B of spills,
            # identical to prod1 (a plain register adds 24 / 132 B of spills that land in the FC1 epilogue).
            use_reg_rows = cute.arch.make_warp_uniform(self._dispatch_use_reg_rows(lane_idx))
        iket.range_push("token_in.wait_metadata_ready")
        self._router.wait_for_metadata_ready(self._device_workspace)
        iket.range_pop()

        for size_copy_round in cutlass.range_constexpr(size_copy_rounds):
            vector_idx = Int32(size_copy_round * self.transfer_thread_count) + transfer_thread_idx
            if vector_idx < Int32(size_vector_count):
                cute.copy(
                    size_copy_atom,
                    size_copy_registers[None, size_copy_round],
                    destination_size_vectors[None, vector_idx],
                )
        iket.range_push("token_in.size_barrier")
        token_in_size_barrier = pipeline.NamedBarrier(
            barrier_id=self.token_in_size_barrier_id, num_threads=self.transfer_thread_count
        )
        token_in_size_barrier.arrive_and_wait()
        iket.range_pop()
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):  # nosec B105
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id,
                num_threads=2 * self.transfer_thread_count,
            )
            sizes_ready_barrier.arrive()

        iket.range_push("token_in.pull_payload")
        token_in_mbarriers = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base)
        token_in_activation = smem_workspace.tensor(self.token_in_activation_smem_region, smem_base)
        token_in_sf = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)
        warp_mbarrier = token_in_mbarriers + transfer_warp_idx
        warp_activation_stage = token_in_activation[transfer_warp_idx, None]
        warp_sf_stage = token_in_sf[transfer_warp_idx, (None, None)]
        if lane_idx == Int32(0):
            cute.arch.mbarrier_init(warp_mbarrier, 1)
        cute.arch.sync_warp()

        fc1_activation_pointer = self._device_workspace.ptr(self.fc1_activation_region)
        fc1_activation_sf = self.fc1_activation_sf_tensor(self._device_workspace)
        fc1_ready_counter = self._device_workspace.ptr(self.fc1_ready_region)
        activation_bytes = (
            cute.cosize(warp_activation_stage) * int(self.activation_dtype.width) // 8
        )
        activation_sf_bytes = cute.cosize(warp_sf_stage) * int(self.activation_sf_dtype.width) // 8
        sf_copy_elements = 4
        source_sf_values = cute.slice_(warp_sf_stage, (0, None))
        source_sf_vectors = cute.zipped_divide(source_sf_values, (sf_copy_elements,))
        sf_copy_atom = _copy_atom(
            self.activation_sf_dtype, sf_copy_elements * int(self.activation_sf_dtype.width)
        )

        if cutlass.const_expr(self.dispatch_reg_rows > 0):
            # v12-dispatch (MEGAMOE_DISPATCH_REG_ROWS=R): each warp moves 1 + R of its rows per round instead of 1.
            # Row 0 of a round takes the existing TMA path (peer -> SMEM stage -> bulk store), rows 1..R go
            # peer -> registers -> local pool with 8-byte loads / stores, so 1 + R rows (and their metadata loads)
            # are in flight per warp with no extra SMEM (the kernel has 1.6 KB of SMEM slack; one more TMA stage
            # per warp would cost an AB stage). Same rows, same destination bytes, same per-row fc1_ready counter
            # increments as the default path: the pool contents and final counters are identical.
            # MEGAMOE_DISPATCH_LSU_ONLY=1: no TMA row; all R rows of a round take the register path.
            reg_row0 = 0 if self.dispatch_lsu_only else 1
            batch_rows = reg_row0 + self.dispatch_reg_rows
            row_bytes = self.bytes_per_token
            data_word_count = row_bytes // 8
            data_rounds = ceil_div(data_word_count, 32)
            sf_vector_bytes = sf_copy_elements * int(self.activation_sf_dtype.width) // 8
            # Same vector index space as the SMEM path's destination_sf_vectors (valid scales, 4 per vector).
            source_sf_vector_count = self.hidden_size // self.activation_sf_vector_size // sf_copy_elements
            sf_rounds = ceil_div(source_sf_vector_count, 32)
            activation_base = self._token_comm_args.activation.iterator.toint()
            if cutlass.const_expr(self.dispatch_dedup):
                dedup_link_pointer = self._device_workspace.ptr(self._router.token_src_link_region)
                # v5: sources that wrote links for this launch (bits set before their sizes_ready, which we waited for);
                # 0 = no link loads at all this launch (prod2's dispatch path). Warp-uniform.
                dedup_mask = cute.arch.make_warp_uniform(
                    cute.arch.load(self._device_workspace.ptr(self._router.dedup_source_mask_region), Int32)
                )

            total_dense = Int32(0)
            if cutlass.const_expr(self.dense_total_warp):
                # [b10 T] lanes sum a strided share of the staged sizes, then a butterfly: one smem load per lane
                # instead of a 56-step serial chain on the path to the first pull.
                for chunk in cutlass.range_constexpr(ceil_div(self.experts_per_rank, 32)):
                    expert_idx = Int32(chunk * 32) + lane_idx
                    if expert_idx < Int32(self.experts_per_rank):
                        total_dense = total_dense + owned_sizes[expert_idx]
                for step in cutlass.range_constexpr(5):
                    total_dense = total_dense + Int32(cute.arch.shuffle_sync(total_dense, lane_idx ^ Int32(16 >> step)))
            else:
                for expert_idx in cutlass.range(self.experts_per_rank, unroll=1):
                    total_dense = total_dense + owned_sizes[expert_idx]
            my_rows = Int32(0)
            if global_warp_idx < total_dense:
                my_rows = (
                    total_dense - global_warp_idx + global_warp_count - Int32(1)
                ) // global_warp_count
            last_dense = global_warp_idx + (my_rows - Int32(1)) * global_warp_count
            if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
                # 1 + R rows per round with register rows, 1 (the TMA row) without.
                rows_per_round = Int32(reg_row0) + Int32(self.dispatch_reg_rows) * use_reg_rows
                if cutlass.const_expr(self.r0tma > 0):
                    # [b10 F] the first round is TMA-only (one row), then rows_per_round rows per round.
                    r0_rest = cutlass.max(my_rows - Int32(self.r0tma), Int32(0))
                    batch_count = my_rows - r0_rest + (r0_rest + rows_per_round - Int32(1)) // rows_per_round
                else:
                    batch_count = (my_rows + rows_per_round - Int32(1)) // rows_per_round
            else:
                batch_count = (my_rows + Int32(batch_rows - 1)) // Int32(batch_rows)

            pull_phase = Int32(0)
            flag_tracker = make_flag_batch_tracker(
                use_async=self.token_in_flag_batch == 1,
                flag_address=Int64(0),
                accumulated_flags=Int32(0),
                phase=Int32(0),
                thread_idx=lane_idx,
            )
            if cutlass.const_expr(self.dedup_stats):
                stat_rows = Int32(0)
                stat_linked = Int32(0)
                stat_local = Int32(0)
            cursor_expert = Int32(0)
            cursor_begin = Int32(0)
            cursor_end = owned_sizes[0]
            cursor_sf_begin = Int32(0)
            cursor_slot_begin = Int32(0)
            if cutlass.const_expr(self.tbx):
                # TBX state: one claimed token-back row (atomic_counter claims, monotonic per warp), the
                # chunk step inside it, the token-back mbarrier parity, and a forward-only expert cursor over it.
                tbx_mbarrier = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base) + transfer_warp_idx
                tbx_act_stage = smem_workspace.tensor(self.token_back_activation_smem_region, smem_base)[
                    transfer_warp_idx, None
                ]
                tbx_sf_stage = None
                if cutlass.const_expr(self.token_back_push_sf):
                    tbx_sf_stage = smem_workspace.tensor(self.token_back_sf_smem_region, smem_base)[
                        transfer_warp_idx, (None, None)
                    ]
                tbx_chunks = self._tbx_chunk_count(tbx_act_stage, tbx_sf_stage)
                self._tbx_chunk_bytes = cute.cosize(tbx_act_stage) * int(self.combine_format.act_dtype.width) // 8
                if lane_idx == Int32(0):
                    cute.arch.mbarrier_init(tbx_mbarrier, 1)
                cute.arch.sync_warp()
                tbx_fc2_done = self._device_workspace.ptr(self.fc2_done_region)
                tbx_row = self.next_token(Int32(0))
                tbx_step = Int32(0)
                tbx_phase = Int32(0)
                tbx_expert = Int32(0)
                tbx_begin = Int32(0)
                tbx_end = owned_sizes[0]
                tbx_ready = Int32(-1)
                tbx_pool = Int32(0)
                tbx_src_rank = Int32(0)
                tbx_src_token = Int32(0)
                tbx_src_topk = Int32(0)
            for batch_idx in cutlass.range(batch_count, unroll=1):
                if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
                    if cutlass.const_expr(self.r0tma > 0):
                        # [b10 F] register rows off in the first round (warp-uniform, like use_reg_rows)
                        round_reg = Int32(0)
                        first_row = batch_idx
                        if batch_idx >= Int32(self.r0tma):
                            round_reg = use_reg_rows
                            first_row = Int32(self.r0tma) + (batch_idx - Int32(self.r0tma)) * rows_per_round
                        round_reg = cute.arch.make_warp_uniform(round_reg)
                    else:
                        round_reg = use_reg_rows
                        first_row = batch_idx * rows_per_round
                else:
                    first_row = batch_idx * Int32(batch_rows)
                if cutlass.const_expr(self.tbx and not self.tbx_late):
                    # TBX issue: the previous chunk's remote store drained during the last round (lane 1
                    # owns every TBX bulk group); then, if the claimed row's expert has finished FC2, load its next
                    # chunk into the token-back stage. Non-blocking: a not-yet-ready row just waits for a later round.
                    if lane_idx == Int32(1):
                        cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.sync_warp()
                    tbx_issued = Int32(0)
                    if tbx_row < total_dense:
                        while tbx_row >= tbx_end:
                            tbx_begin = tbx_end
                            tbx_expert = tbx_expert + Int32(1)
                            tbx_end = tbx_begin + owned_sizes[tbx_expert]
                        if tbx_expert < Int32(self.experts_per_rank - self.fc2_tail_blocks):
                            if tbx_ready != tbx_expert:
                                tbx_target = (
                                    (tbx_end - tbx_begin + Int32(self.tokens_per_fc1_ready_slot - 1))
                                    // Int32(self.tokens_per_fc1_ready_slot)
                                ) * Int32(self.fc2_done_signals_per_token_tile)
                                tbx_seen = Int32(0)
                                if lane_idx == Int32(0):
                                    tbx_seen = cute.arch.load(
                                        tbx_fc2_done + tbx_expert, Int32, sem="relaxed", scope="gpu"
                                    )
                                tbx_seen = cute.arch.shuffle_sync(tbx_seen, Int32(0))
                                if tbx_seen >= tbx_target:
                                    # Acquire edge for the FC2 output rows (the counter only grows in a launch).
                                    if (
                                        cute.arch.load(
                                            tbx_fc2_done + tbx_expert, Int32, sem="acquire", scope="gpu"
                                        )
                                        >= tbx_target
                                    ):
                                        tbx_ready = tbx_expert
                            if tbx_ready == tbx_expert:
                                if tbx_step == Int32(0):
                                    tbx_pool = pool_expert_bases[tbx_expert] + (tbx_row - tbx_begin)
                                    tbx_metadata = TokenSrcMetadata.load(
                                        token_metadata_pointer.toint()
                                        + Int64(tbx_pool) * Int64(TokenSrcMetadata.nbytes)
                                    )
                                    tbx_src_rank = tbx_metadata.src_rank
                                    tbx_src_token = tbx_metadata.src_token
                                    tbx_src_topk = tbx_metadata.src_topk
                                self._tbx_load(tbx_step, tbx_pool, tbx_mbarrier, tbx_act_stage, tbx_sf_stage)
                                tbx_issued = Int32(1)
                row_ok = []
                row0_dense = None
                pool_rows = []
                sf_rows = []
                slot_rows = []
                for row in cutlass.range_constexpr(batch_rows):
                    dense_token_idx = global_warp_idx + (first_row + Int32(row)) * global_warp_count
                    valid = Int32(1)
                    if cutlass.const_expr(row > 0):
                        valid = Int32(0)
                        if first_row + Int32(row) < my_rows:
                            valid = Int32(1)
                    # Rows past this warp's last row resolve to its last row: loads stay in bounds, stores and
                    # the ready-counter bump are predicated off.
                    dense_token_idx = cutlass.min(dense_token_idx, last_dense)
                    if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
                        # TMA-row-only rounds: rows > 0 are off and alias row 0 (no cursor advance).
                        if cutlass.const_expr(row == 0):
                            row0_dense = dense_token_idx
                        else:
                            valid = valid * round_reg
                            dense_token_idx = row0_dense + (dense_token_idx - row0_dense) * round_reg
                    while dense_token_idx >= cursor_end:
                        cursor_size = cursor_end - cursor_begin
                        if cutlass.const_expr(self.sf_tile_tokens):
                            cursor_sf_begin = cursor_sf_begin + (
                                (cursor_size + Int32(self.sf_tile_tokens - 1))
                                // Int32(self.sf_tile_tokens)
                            ) * Int32(SF_TILE_ROWS_N224)
                        else:
                            cursor_sf_begin = cursor_sf_begin + (
                                (cursor_size + Int32(self.sf_padding_block - 1))
                                // Int32(self.sf_padding_block)
                            ) * Int32(self.sf_padding_block)
                        cursor_slot_begin = cursor_slot_begin + (
                            (cursor_size + Int32(self.tokens_per_fc1_ready_slot - 1))
                            // Int32(self.tokens_per_fc1_ready_slot)
                        )
                        cursor_begin = cursor_end
                        cursor_expert = cursor_expert + Int32(1)
                        cursor_end = cursor_begin + owned_sizes[cursor_expert]
                    token_in_expert = dense_token_idx - cursor_begin
                    row_ok.append(valid)
                    pool_rows.append(pool_expert_bases[cursor_expert] + token_in_expert)
                    sf_rows.append(cursor_sf_begin + sf_row_in_expert(token_in_expert, self.sf_tile_tokens))
                    slot_rows.append(
                        cursor_slot_begin + token_in_expert // Int32(self.tokens_per_fc1_ready_slot)
                    )

                metadata_rows = []
                for row in cutlass.range_constexpr(batch_rows):
                    metadata_rows.append(
                        TokenSrcMetadata.load(
                            token_metadata_pointer.toint()
                            + Int64(pool_rows[row]) * Int64(TokenSrcMetadata.nbytes)
                        )
                    )
                dedup_local = []
                if cutlass.const_expr(self.dispatch_dedup):
                    # s25-dedup: a secondary row whose primary's FC1-ready slot is complete reads the payload from the
                    # primary's pool row (local) instead of the peer. Same acquire edge FC1 uses before its TMA reads.
                    dedup_links = []
                    for row in cutlass.range_constexpr(batch_rows):
                        link = Int64(-1)
                        if dedup_mask != Int32(0):
                            link = cute.arch.load(dedup_link_pointer + pool_rows[row], Int64)
                        dedup_links.append(link)
                    for row in cutlass.range_constexpr(batch_rows):
                        link = dedup_links[row]
                        local_row = Int64(-1)
                        # a link counts only if its source wrote links this launch (else it is a stale entry); the
                        # whole check sits under the warp-uniform mask test so a duplicate-free launch skips it
                        if dedup_mask != Int32(0):
                          if ((dedup_mask >> metadata_rows[row].src_rank) & Int32(1)) * row_ok[row] != Int32(0):
                            if link >= Int64(0):
                                link_slot = Int32((link >> Int64(26)) & Int64(0xFFFFF))
                                link_target = Int32((link >> Int64(46)) & Int64(0x3FF))
                                if (
                                    cute.arch.load(
                                        fc1_ready_counter + link_slot, Int32, sem="acquire", scope="gpu"
                                    )
                                    >= link_target
                                ):
                                    local_row = link & Int64(0x3FFFFFF)
                        dedup_local.append(local_row)
                        if cutlass.const_expr(self.dedup_stats):
                            stat_rows = stat_rows + row_ok[row]
                            stat_linked = stat_linked + row_ok[row] * Int32(link >= Int64(0))
                            stat_local = stat_local + Int32(local_row >= Int64(0))

                data_regs = []
                sf_regs = []
                for row in cutlass.range_constexpr(batch_rows):
                    metadata = metadata_rows[row]
                    peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                        Int64(0), metadata.src_rank, Int64(0)
                    )
                    remote_activation_address = (
                        activation_base
                        + peer_offset
                        + Int64(metadata.src_token) * Int64(row_bytes)
                    )
                    if cutlass.const_expr(self.dispatch_dedup):
                        # payload source: the primary's local pool row when ready, else the peer (scales: always peer)
                        if dedup_local[row] >= Int64(0):
                            remote_activation_address = fc1_activation_pointer.toint() + dedup_local[row] * Int64(
                                row_bytes
                            )
                    remote_sf = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.activation_sf.dtype,
                            self._token_comm_args.activation_sf.iterator.toint() + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(
                                self._token_comm_args.activation_sf.iterator.max_alignment
                            ),
                        ),
                        self._token_comm_args.activation_sf.layout,
                    )
                    remote_sf_row = remote_sf[Int64(metadata.src_token), None]
                    if cutlass.const_expr(row < reg_row0):
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                warp_mbarrier, Int32(activation_bytes + activation_sf_bytes)
                            )
                            tma_load_1d(
                                warp_activation_stage.iterator,
                                Int64(remote_activation_address),
                                warp_mbarrier,
                                Int32(activation_bytes),
                            )
                            tma_load_1d(
                                warp_sf_stage.iterator,
                                remote_sf_row.iterator,
                                warp_mbarrier,
                                Int32(activation_sf_bytes),
                            )
                        cute.arch.sync_warp()
                    else:
                        row_words = []
                        for word_round in cutlass.range_constexpr(data_rounds):
                            word_idx = lane_idx + Int32(word_round * 32)
                            if cutlass.const_expr((word_round + 1) * 32 > data_word_count):
                                word_idx = cutlass.min(word_idx, Int32(data_word_count - 1))
                            if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
                                # no peer transaction for rows that are off (TMA-only rounds, padding).
                                row_words.append(
                                    ldg_nl1_b64_pred(
                                        Int64(remote_activation_address) + Int64(word_idx) * Int64(8),
                                        row_ok[row],
                                    )
                                )
                            else:
                                row_words.append(
                                    ldg_nl1_b64(Int64(remote_activation_address) + Int64(word_idx) * Int64(8))
                                )
                        data_regs.append(row_words)
                        remote_sf_address = remote_sf_row.iterator.toint()
                        row_sf = []
                        for sf_round in cutlass.range_constexpr(sf_rounds):
                            sf_vector_idx = lane_idx + Int32(sf_round * 32)
                            if cutlass.const_expr((sf_round + 1) * 32 > source_sf_vector_count):
                                sf_vector_idx = cutlass.min(
                                    sf_vector_idx, Int32(source_sf_vector_count - 1)
                                )
                            if cutlass.const_expr(self.dispatch_adapt_x100 > 0):
                                row_sf.append(
                                    ldg_nl1_b32_pred(
                                        Int64(remote_sf_address)
                                        + Int64(sf_vector_idx) * Int64(sf_vector_bytes),
                                        row_ok[row],
                                    )
                                )
                            else:
                                row_sf.append(
                                    ldg_nl1_b32(
                                        Int64(remote_sf_address)
                                        + Int64(sf_vector_idx) * Int64(sf_vector_bytes)
                                    )
                                )
                        sf_regs.append(row_sf)

                if cutlass.const_expr(self.tbx and self.tbx_late):
                    # TBX issue (late, default): placed after the dispatch round's peer loads are in
                    # flight, so lane 1's wait for its previous remote store overlaps the dispatch read latency instead
                    # of serializing in front of it (the top-of-round order, MEGAMOE_S12_TBX=2, slowed dispatch ~20%).
                    # TBX issue: the previous chunk's remote store drained during the last round (lane 1
                    # owns every TBX bulk group); then, if the claimed row's expert has finished FC2, load its next
                    # chunk into the token-back stage. Non-blocking: a not-yet-ready row just waits for a later round.
                    if lane_idx == Int32(1):
                        cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.sync_warp()
                    tbx_issued = Int32(0)
                    if tbx_row < total_dense:
                        while tbx_row >= tbx_end:
                            tbx_begin = tbx_end
                            tbx_expert = tbx_expert + Int32(1)
                            tbx_end = tbx_begin + owned_sizes[tbx_expert]
                        if tbx_expert < Int32(self.experts_per_rank - self.fc2_tail_blocks):
                            if tbx_ready != tbx_expert:
                                tbx_target = (
                                    (tbx_end - tbx_begin + Int32(self.tokens_per_fc1_ready_slot - 1))
                                    // Int32(self.tokens_per_fc1_ready_slot)
                                ) * Int32(self.fc2_done_signals_per_token_tile)
                                tbx_seen = Int32(0)
                                if lane_idx == Int32(0):
                                    tbx_seen = cute.arch.load(
                                        tbx_fc2_done + tbx_expert, Int32, sem="relaxed", scope="gpu"
                                    )
                                tbx_seen = cute.arch.shuffle_sync(tbx_seen, Int32(0))
                                if tbx_seen >= tbx_target:
                                    # Acquire edge for the FC2 output rows (the counter only grows in a launch).
                                    if (
                                        cute.arch.load(
                                            tbx_fc2_done + tbx_expert, Int32, sem="acquire", scope="gpu"
                                        )
                                        >= tbx_target
                                    ):
                                        tbx_ready = tbx_expert
                            if tbx_ready == tbx_expert:
                                if tbx_step == Int32(0):
                                    tbx_pool = pool_expert_bases[tbx_expert] + (tbx_row - tbx_begin)
                                    tbx_metadata = TokenSrcMetadata.load(
                                        token_metadata_pointer.toint()
                                        + Int64(tbx_pool) * Int64(TokenSrcMetadata.nbytes)
                                    )
                                    tbx_src_rank = tbx_metadata.src_rank
                                    tbx_src_token = tbx_metadata.src_token
                                    tbx_src_topk = tbx_metadata.src_topk
                                self._tbx_load(tbx_step, tbx_pool, tbx_mbarrier, tbx_act_stage, tbx_sf_stage)
                                tbx_issued = Int32(1)
                if cutlass.const_expr(reg_row0 == 1):
                    # Row 0: unchanged SMEM path.
                    cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)
                    destination_activation = cute.make_ptr(
                        self.activation_dtype,
                        fc1_activation_pointer.toint() + Int64(pool_rows[0]) * Int64(row_bytes),
                        AddressSpace.gmem,
                        assumed_align=16,
                    )
                    if cutlass.const_expr(self.tbx):
                        # TBX: lane 0 owns every dispatch bulk group, lane 1 every token-back group.
                        if lane_idx == Int32(0):
                            cp_async_bulk_s2g(
                                destination_activation,
                                warp_activation_stage.iterator,
                                Int32(activation_bytes),
                            )
                    else:
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(
                                destination_activation,
                                warp_activation_stage.iterator,
                                Int32(activation_bytes),
                            )
                    cute.arch.sync_warp()
                    cute.arch.cp_async_bulk_commit_group()

                    destination_sf_row = fc1_activation_sf[Int64(sf_rows[0]), ((None, None), None)]
                    destination_sf_values = cute.slice_(destination_sf_row, (0, None, None))
                    destination_sf_values = cute.group_modes(destination_sf_values, 0, 2)
                    destination_sf_vectors = cute.zipped_divide(
                        destination_sf_values, (sf_copy_elements,)
                    )
                    sf_vector_count = cute.size(destination_sf_vectors, mode=[1])
                    for sf_round in cutlass.range_constexpr(ceil_div(sf_vector_count, 32)):
                        sf_vector_idx = Int32(sf_round * 32) + lane_idx
                        if sf_vector_idx < Int32(sf_vector_count):
                            cute.copy(
                                sf_copy_atom,
                                source_sf_vectors[None, sf_vector_idx],
                                destination_sf_vectors[None, sf_vector_idx],
                            )

                # Register rows: registers -> local pool (predicated off past this warp's last row).
                for row in cutlass.range_constexpr(reg_row0, batch_rows):
                    store_ok = row_ok[row]
                    destination_row_address = fc1_activation_pointer.toint() + Int64(
                        pool_rows[row]
                    ) * Int64(row_bytes)
                    for word_round in cutlass.range_constexpr(data_rounds):
                        word_idx = lane_idx + Int32(word_round * 32)
                        word_ok = store_ok
                        if cutlass.const_expr((word_round + 1) * 32 > data_word_count):
                            if word_idx >= Int32(data_word_count):
                                word_ok = Int32(0)
                            word_idx = cutlass.min(word_idx, Int32(data_word_count - 1))
                        stg_b64(
                            destination_row_address + Int64(word_idx) * Int64(8),
                            data_regs[row - reg_row0][word_round],
                            word_ok,
                        )
                    row_destination_sf = fc1_activation_sf[Int64(sf_rows[row]), ((None, None), None)]
                    row_destination_sf = cute.slice_(row_destination_sf, (0, None, None))
                    row_destination_sf = cute.group_modes(row_destination_sf, 0, 2)
                    row_destination_sf_vectors = cute.zipped_divide(
                        row_destination_sf, (sf_copy_elements,)
                    )
                    for sf_round in cutlass.range_constexpr(sf_rounds):
                        sf_vector_idx = lane_idx + Int32(sf_round * 32)
                        sf_ok = store_ok
                        if cutlass.const_expr((sf_round + 1) * 32 > source_sf_vector_count):
                            if sf_vector_idx >= Int32(source_sf_vector_count):
                                sf_ok = Int32(0)
                            sf_vector_idx = cutlass.min(
                                sf_vector_idx, Int32(source_sf_vector_count - 1)
                            )
                        stg_b32_pred(
                            row_destination_sf_vectors[None, sf_vector_idx].iterator.toint(),
                            sf_regs[row - reg_row0][sf_round],
                            sf_ok,
                        )

                if cutlass.const_expr(self.tbx):
                    if tbx_issued == Int32(1):
                        # TBX store: the chunk's local load had the whole dispatch round to land.
                        cute.arch.mbarrier_wait(tbx_mbarrier, tbx_phase)
                        tbx_phase = tbx_phase ^ Int32(1)
                        self._tbx_store(
                            tbx_step, tbx_src_rank, tbx_src_token, tbx_src_topk, tbx_act_stage, tbx_sf_stage
                        )
                        tbx_step = tbx_step + Int32(1)
                        if tbx_step == Int32(tbx_chunks):
                            tbx_step = Int32(0)
                            tbx_row = self.next_token(tbx_row)
                    # Only the elected lane (lane 0) issued dispatch bulk groups; lane 1's TBX group must not be
                    # waited for here (bulk async-groups are per thread).
                    if lane_idx == Int32(0):
                        cute.arch.cp_async_bulk_wait_group(0, read=True)
                elif cutlass.const_expr(reg_row0 == 1):
                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                cute.arch.sync_warp()
                for row in cutlass.range_constexpr(batch_rows):
                    ready_address = (fc1_ready_counter + slot_rows[row]).toint()
                    if cutlass.const_expr(row > 0):
                        # A zero address is skipped by the tracker's fire(): no bump for a padding row.
                        ready_address = ready_address * Int64(row_ok[row])
                    flag_tracker = flag_tracker.accumulate(
                        Int32(0), self.token_in_flag_batch, ready_address
                    )
                cute.arch.sync_warp()
                pull_phase = pull_phase ^ Int32(1)
            flag_tracker.fire()
            if cutlass.const_expr(self.dedup_stats):
                if global_warp_idx == Int32(0) and lane_idx == Int32(0):
                    cute.printf(
                        "[dedup] rank %d warp0 rows %d linked %d local %d\n",
                        self._local_rank,
                        stat_rows,
                        stat_linked,
                        stat_local,
                    )
            if cutlass.const_expr(self.tbx):
                # TBX hand-off: drain lane 1's last store, finish a half-pushed row (its expert is ready),
                # then hand the claimed-but-unstarted row and the mbarrier parity to token_back().
                if lane_idx == Int32(1):
                    cute.arch.cp_async_bulk_wait_group(0)
                cute.arch.sync_warp()
                while tbx_step > Int32(0):
                    self._tbx_load(tbx_step, tbx_pool, tbx_mbarrier, tbx_act_stage, tbx_sf_stage)
                    cute.arch.mbarrier_wait(tbx_mbarrier, tbx_phase)
                    tbx_phase = tbx_phase ^ Int32(1)
                    self._tbx_store(
                        tbx_step, tbx_src_rank, tbx_src_token, tbx_src_topk, tbx_act_stage, tbx_sf_stage
                    )
                    if lane_idx == Int32(1):
                        cute.arch.cp_async_bulk_wait_group(0)
                    cute.arch.sync_warp()
                    tbx_step = tbx_step + Int32(1)
                    if tbx_step == Int32(tbx_chunks):
                        tbx_step = Int32(0)
                        tbx_row = self.next_token(tbx_row)
                tbx_row_out = tbx_row
                tbx_phase_out = tbx_phase
                tbx_in_phase_out = pull_phase
        else:
            next_dense_token = global_warp_idx
            expert_valid_begin = Int32(0)
            expert_sf_begin = Int32(0)
            expert_ready_slot_begin = Int32(0)
            pull_phase = Int32(0)
            flag_tracker = make_flag_batch_tracker(
                use_async=self.token_in_flag_batch == 1,
                flag_address=Int64(0),
                accumulated_flags=Int32(0),
                phase=Int32(0),
                thread_idx=lane_idx,
            )

            local_expert = Int32(0)
            while local_expert < Int32(self.experts_per_rank):
                expert_token_count = owned_sizes[local_expert]
                expert_valid_end = expert_valid_begin + expert_token_count
                pull_count = Int32(0)
                if next_dense_token < expert_valid_end:
                    pull_count = (
                        expert_valid_end - next_dense_token + global_warp_count - Int32(1)
                    ) // global_warp_count

                for pull_round in cutlass.range(pull_count, unroll=1):
                    dense_token_idx = next_dense_token + Int32(pull_round) * global_warp_count
                    token_in_expert = dense_token_idx - expert_valid_begin
                    pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                    sf_token_idx = expert_sf_begin + sf_row_in_expert(token_in_expert, self.sf_tile_tokens)
                    ready_slot_idx = expert_ready_slot_begin + token_in_expert // Int32(
                        self.tokens_per_fc1_ready_slot
                    )

                    metadata = TokenSrcMetadata.load(
                        token_metadata_pointer.toint()
                        + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                    )
                    peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                        Int64(0), metadata.src_rank, Int64(0)
                    )
                    remote_activation_address = (
                        self._token_comm_args.activation.iterator.toint()
                        + peer_offset
                        + Int64(metadata.src_token) * Int64(self.bytes_per_token)
                    )
                    remote_sf = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.activation_sf.dtype,
                            self._token_comm_args.activation_sf.iterator.toint() + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(self._token_comm_args.activation_sf.iterator.max_alignment),
                        ),
                        self._token_comm_args.activation_sf.layout,
                    )
                    remote_sf_row = remote_sf[Int64(metadata.src_token), None]

                    with cute.arch.elect_one():
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            warp_mbarrier, Int32(activation_bytes + activation_sf_bytes)
                        )
                        tma_load_1d(
                            warp_activation_stage.iterator,
                            Int64(remote_activation_address),
                            warp_mbarrier,
                            Int32(activation_bytes),
                        )
                        tma_load_1d(
                            warp_sf_stage.iterator,
                            remote_sf_row.iterator,
                            warp_mbarrier,
                            Int32(activation_sf_bytes),
                        )
                    cute.arch.sync_warp()
                    cute.arch.mbarrier_wait(warp_mbarrier, pull_phase)

                    destination_activation = cute.make_ptr(
                        self.activation_dtype,
                        fc1_activation_pointer.toint()
                        + Int64(pool_token_idx) * Int64(self.bytes_per_token),
                        AddressSpace.gmem,
                        assumed_align=16,
                    )
                    with cute.arch.elect_one():
                        cp_async_bulk_s2g(
                            destination_activation,
                            warp_activation_stage.iterator,
                            Int32(activation_bytes),
                        )
                    cute.arch.sync_warp()
                    cute.arch.cp_async_bulk_commit_group()

                    destination_sf_row = fc1_activation_sf[Int64(sf_token_idx), ((None, None), None)]
                    destination_sf_values = cute.slice_(destination_sf_row, (0, None, None))
                    destination_sf_values = cute.group_modes(destination_sf_values, 0, 2)
                    destination_sf_vectors = cute.zipped_divide(
                        destination_sf_values, (sf_copy_elements,)
                    )
                    sf_vector_count = cute.size(destination_sf_vectors, mode=[1])
                    for sf_round in cutlass.range_constexpr(ceil_div(sf_vector_count, 32)):
                        sf_vector_idx = Int32(sf_round * 32) + lane_idx
                        if sf_vector_idx < Int32(sf_vector_count):
                            cute.copy(
                                sf_copy_atom,
                                source_sf_vectors[None, sf_vector_idx],
                                destination_sf_vectors[None, sf_vector_idx],
                            )

                    cute.arch.cp_async_bulk_wait_group(0, read=True)
                    cute.arch.sync_warp()
                    ready_address = (fc1_ready_counter + ready_slot_idx).toint()
                    flag_tracker = flag_tracker.accumulate(
                        Int32(0), self.token_in_flag_batch, ready_address
                    )
                    cute.arch.sync_warp()
                    pull_phase = pull_phase ^ Int32(1)

                next_dense_token = next_dense_token + pull_count * global_warp_count
                expert_valid_begin = expert_valid_end
                if cutlass.const_expr(self.sf_tile_tokens):
                    expert_sf_begin = expert_sf_begin + (
                        (expert_token_count + Int32(self.sf_tile_tokens - 1))
                        // Int32(self.sf_tile_tokens)
                    ) * Int32(SF_TILE_ROWS_N224)
                else:
                    expert_sf_begin = expert_sf_begin + (
                        (expert_token_count + Int32(self.sf_padding_block - 1))
                        // Int32(self.sf_padding_block)
                    ) * Int32(self.sf_padding_block)
                expert_ready_slot_begin = expert_ready_slot_begin + (
                    (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                    // Int32(self.tokens_per_fc1_ready_slot)
                )
                local_expert = local_expert + Int32(1)

            flag_tracker.fire()
        cute.arch.sync_warp()
        iket.range_pop()
        if cutlass.const_expr(
            self.token_back_enabled
            and self.token_back_mode != "standalone_warps"  # nosec B105
            and not self.tbx  # TBX: token-back's stage is not overlaid on token-in's
        ):
            iket.range_push("token_in.transfer_barrier")
            transfer_lifetime_barrier = pipeline.NamedBarrier(
                barrier_id=self.transfer_lifetime_barrier_id, num_threads=self.transfer_thread_count
            )
            transfer_lifetime_barrier.arrive_and_wait()
            iket.range_pop()
        if cutlass.const_expr(self.tbx and self.tbx_db):
            return tbx_row_out, tbx_phase_out, tbx_in_phase_out
        if cutlass.const_expr(self.tbx):
            return tbx_row_out, tbx_phase_out

    def _tbx_chunk_count(self, act_stage, sf_stage) -> int:
        """Data chunks per token-back row through the TBX stage (the scale row rides with the last)."""
        output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
        chunk_bytes = cute.cosize(act_stage) * int(self.combine_format.act_dtype.width) // 8
        if self.token_back_push_sf:
            output_sf_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            if cute.cosize(sf_stage) * int(self.combine_format.scale_dtype.width) // 8 < output_sf_bytes:
                raise ValueError("MegaMoE token-back (TBX) needs the whole scale row in one token-back SF stage.")
        return ceil_div(output_token_bytes, chunk_bytes)

    @cute.jit
    def _tbx_load(
        self, step: Int32, pool_token_idx: Int32, mbarrier: cute.Pointer, act_stage, sf_stage
    ) -> None:
        """Lane 1: TMA-load chunk `step` of FC2 output row pool_token_idx (+ its scale row on the last
        chunk) into the warp's TBX stage; completion lands on `mbarrier`."""
        output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
        chunk_bytes = self._tbx_chunk_bytes
        chunk_count = ceil_div(output_token_bytes, chunk_bytes)
        fc2_activation_address = self._device_workspace.ptr(self.fc2_activation_region).toint() + Int64(
            pool_token_idx
        ) * Int64(output_token_bytes)
        if cutlass.const_expr(self.token_back_push_sf):
            output_sf_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            fc2_sf_address = self._device_workspace.ptr(self.fc2_activation_sf_region).toint() + Int64(
                pool_token_idx
            ) * Int64(output_sf_bytes)
        if self._lane_idx == Int32(1):
            for chunk_idx in cutlass.range_constexpr(chunk_count):
                if step == Int32(chunk_idx):
                    chunk_bytes_this_round = min(chunk_bytes, output_token_bytes - chunk_idx * chunk_bytes)
                    if cutlass.const_expr(chunk_idx == chunk_count - 1 and self.token_back_push_sf):
                        cute.arch.mbarrier_arrive_and_expect_tx(
                            mbarrier, Int32(chunk_bytes_this_round + output_sf_bytes)
                        )
                        tma_load_1d(
                            act_stage.iterator,
                            fc2_activation_address + Int64(chunk_idx * chunk_bytes),
                            mbarrier,
                            Int32(chunk_bytes_this_round),
                        )
                        tma_load_1d(sf_stage.iterator, fc2_sf_address, mbarrier, Int32(output_sf_bytes))
                    else:
                        cute.arch.mbarrier_arrive_and_expect_tx(mbarrier, Int32(chunk_bytes_this_round))
                        tma_load_1d(
                            act_stage.iterator,
                            fc2_activation_address + Int64(chunk_idx * chunk_bytes),
                            mbarrier,
                            Int32(chunk_bytes_this_round),
                        )
        cute.arch.sync_warp()

    @cute.jit
    def _tbx_store(
        self, step: Int32, src_rank: Int32, src_token: Int32, src_topk: Int32, act_stage, sf_stage
    ) -> None:
        """Lane 1: bulk-store the staged chunk (+ scale row on the last chunk) to the source rank's combine
        row [src_token, src_topk] -- the same destination bytes token_back() writes -- and commit one group."""
        output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
        chunk_bytes = self._tbx_chunk_bytes
        chunk_count = ceil_div(output_token_bytes, chunk_bytes)
        peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(Int64(0), src_rank, Int64(0))
        remote_activation = cute.make_tensor(
            cute.make_ptr(
                self._token_comm_args.pre_reduced_activation.dtype,
                self._token_comm_args.pre_reduced_activation.iterator.toint() + peer_offset,
                AddressSpace.gmem,
                assumed_align=self._token_comm_args.pre_reduced_activation.iterator.max_alignment,
            ),
            self._token_comm_args.pre_reduced_activation.layout,
        )
        destination_row = remote_activation[Int64(src_token), src_topk, None]
        if cutlass.const_expr(self.token_back_push_sf):
            output_sf_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            remote_sf = cute.make_tensor(
                cute.make_ptr(
                    self._token_comm_args.pre_reduced_activation_sf.dtype,
                    self._token_comm_args.pre_reduced_activation_sf.iterator.toint() + peer_offset,
                    AddressSpace.gmem,
                    assumed_align=self._token_comm_args.pre_reduced_activation_sf.iterator.max_alignment,
                ),
                self._token_comm_args.pre_reduced_activation_sf.layout,
            )
            destination_sf_row = remote_sf[Int64(src_token), src_topk, None]
        if self._lane_idx == Int32(1):
            for chunk_idx in cutlass.range_constexpr(chunk_count):
                if step == Int32(chunk_idx):
                    chunk_bytes_this_round = min(chunk_bytes, output_token_bytes - chunk_idx * chunk_bytes)
                    cp_async_bulk_s2g(
                        cute.make_ptr(
                            cutlass.Uint8,
                            destination_row.iterator.toint() + Int64(chunk_idx * chunk_bytes),
                            AddressSpace.gmem,
                            assumed_align=16,
                        ),
                        act_stage.iterator,
                        Int32(chunk_bytes_this_round),
                    )
                    if cutlass.const_expr(chunk_idx == chunk_count - 1 and self.token_back_push_sf):
                        cp_async_bulk_s2g(
                            cute.make_ptr(
                                cutlass.Uint8,
                                destination_sf_row.iterator.toint(),
                                AddressSpace.gmem,
                                assumed_align=16,
                            ),
                            sf_stage.iterator,
                            Int32(output_sf_bytes),
                        )
            cute.arch.cp_async_bulk_commit_group()
        cute.arch.sync_warp()

    @cute.jit
    def _dispatch_use_reg_rows(self, lane_idx: Int32) -> Int32:
        """1 unless one source rank carries > DISPATCH_ADAPT_X100 / 100 x the mean per-peer row count this
        rank pulls (sizes_by_rank rows of the peers, this rank's local experts). Warp-uniform; needs sizes_ready (the
        matrix rows are published before the sizes_ready increments)."""
        sizes_by_rank = self._device_workspace.tensor(self._router.sizes_by_rank_region)
        owner_expert_begin = self._local_rank * Int32(self.experts_per_rank)
        rows_from_lane = Int32(0)  # <= max_tokens_per_rank * topk per source
        if lane_idx < Int32(self.world_size):
            if lane_idx != self._local_rank:
                for expert_idx in cutlass.range(self.experts_per_rank, unroll=8):
                    rows_from_lane = rows_from_lane + sizes_by_rank[lane_idx, owner_expert_begin + expert_idx]
        rows_max = rows_from_lane
        rows_sum = rows_from_lane
        for step in cutlass.range_constexpr(5):
            source_lane = lane_idx ^ Int32(16 >> step)
            rows_max = cutlass.max(rows_max, Int32(cute.arch.shuffle_sync(rows_max, source_lane)))
            rows_sum = rows_sum + Int32(cute.arch.shuffle_sync(rows_sum, source_lane))
        use_reg_rows = Int32(0)
        if Int64(rows_max) * Int64((self.world_size - 1) * 100) <= Int64(rows_sum) * Int64(self.dispatch_adapt_x100):
            use_reg_rows = Int32(1)
        return use_reg_rows

    @cute.jit
    def token_back_db(
        self, smem_workspace: SmemWorkspace, smem_base: cute.Pointer, tbx_row: Int32, tbx_phase: Int32, in_phase: Int32
    ) -> None:
        """TBDB: token_back() with two stages per warp after dispatch. Chunk c of a row goes through stage A
        (the TBX token-back stage, mbarrier parity tbx_phase) when c is even and stage B (the warp's idle token-in stage,
        parity in_phase) when c is odd; lane 1 owns every bulk group and waits with wait_group(1) before reusing a
        stage, so the previous chunk's remote drain overlaps this chunk's local load. Same rows, bytes, destinations,
        readiness edges (per expert, tail blocks) and row claims as token_back(); no adaptive pacing."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
        chunk_count = ceil_div(output_token_bytes, self._tbx_chunk_bytes)
        if cutlass.const_expr(chunk_count % 2 != 0):
            raise ValueError("MegaMoE token-back double-buffering (TBDB) needs an even token-back chunk count.")
        mbarrier_a = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base) + transfer_warp_idx
        mbarrier_b = smem_workspace.ptr(self.token_in_mbarrier_region, smem_base) + transfer_warp_idx
        act_a = smem_workspace.tensor(self.token_back_activation_smem_region, smem_base)[transfer_warp_idx, None]
        act_b = smem_workspace.tensor(self.token_in_activation_smem_region, smem_base)[transfer_warp_idx, None]
        if cutlass.const_expr(cute.cosize(act_b) * int(self.activation_dtype.width) // 8 < self._tbx_chunk_bytes):
            raise ValueError("MegaMoE token-back double-buffering (TBDB): the token-in stage is smaller than a token-back chunk.")
        sf_a = None
        sf_b = None
        if cutlass.const_expr(self.token_back_push_sf):
            sf_a = smem_workspace.tensor(self.token_back_sf_smem_region, smem_base)[transfer_warp_idx, (None, None)]
            sf_b = smem_workspace.tensor(self.token_in_sf_smem_region, smem_base)[transfer_warp_idx, (None, None)]
        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        fc2_done = self._device_workspace.ptr(self.fc2_done_region)

        next_dense_token = tbx_row
        phase_a = tbx_phase
        phase_b = in_phase
        expert_valid_begin = Int32(0)
        local_expert = Int32(0)
        expert_block_begin = Int32(0)
        waited_block = Int32(-1)
        iket.range_push("token_back.db")
        while local_expert < Int32(self.experts_per_rank):
            expert_token_count = owned_sizes[local_expert]
            expert_valid_end = expert_valid_begin + expert_token_count
            tail_expert = Boolean(False)
            if cutlass.const_expr(self.fc2_tail_blocks > 0):
                tail_expert = local_expert >= Int32(self.experts_per_rank - self.fc2_tail_blocks)
            if next_dense_token < expert_valid_end and not tail_expert:
                token_tile_count = (
                    expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1)
                ) // Int32(self.tokens_per_fc1_ready_slot)
                completion_target = token_tile_count * Int32(self.fc2_done_signals_per_token_tile)
                while (
                    cute.arch.load(fc2_done + local_expert, Int32, sem="acquire", scope="gpu")
                    < completion_target
                ):
                    nanosleep(500)
            while next_dense_token < expert_valid_end:
                token_in_expert = next_dense_token - expert_valid_begin
                if cutlass.const_expr(self.fc2_tail_blocks > 0):
                    if tail_expert:
                        row_block = Int32(self.experts_per_rank) + expert_block_begin + token_in_expert // Int32(
                            self.tokens_per_fc1_ready_slot
                        )
                        if row_block != waited_block:
                            while cute.arch.load(
                                fc2_done + row_block, Int32, sem="acquire", scope="gpu"
                            ) < Int32(self.fc2_done_signals_per_token_tile):
                                nanosleep(500)
                            waited_block = row_block
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint() + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                for chunk_idx in cutlass.range_constexpr(chunk_count):
                    if lane_idx == Int32(1):
                        cute.arch.cp_async_bulk_wait_group(1)
                    cute.arch.sync_warp()
                    if cutlass.const_expr(chunk_idx % 2 == 0):
                        self._tbx_load(Int32(chunk_idx), pool_token_idx, mbarrier_a, act_a, sf_a)
                        cute.arch.mbarrier_wait(mbarrier_a, phase_a)
                        phase_a = phase_a ^ Int32(1)
                        self._tbx_store(
                            Int32(chunk_idx), metadata.src_rank, metadata.src_token, metadata.src_topk, act_a, sf_a
                        )
                    else:
                        self._tbx_load(Int32(chunk_idx), pool_token_idx, mbarrier_b, act_b, sf_b)
                        cute.arch.mbarrier_wait(mbarrier_b, phase_b)
                        phase_b = phase_b ^ Int32(1)
                        self._tbx_store(
                            Int32(chunk_idx), metadata.src_rank, metadata.src_token, metadata.src_topk, act_b, sf_b
                        )
                next_dense_token = self.next_token(next_dense_token)
            if cutlass.const_expr(self.fc2_tail_blocks > 0):
                expert_block_begin = expert_block_begin + (
                    (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                    // Int32(self.tokens_per_fc1_ready_slot)
                )
            expert_valid_begin = expert_valid_end
            local_expert = local_expert + Int32(1)
        if lane_idx == Int32(1):
            cute.arch.cp_async_bulk_wait_group(0)
        cute.arch.sync_warp()
        iket.range_pop()

    @cute.jit
    def _stateless_pace(self, reference_window: Int32, current_window: Int32) -> None:
        sleep_cycles = Int32(0)
        if current_window < reference_window:
            sleep_cycles = reference_window - current_window
        elif current_window > reference_window:
            sleep_cycles = cutlass.min(
                current_window - reference_window, Int32(self.standalone_max_backoff_cycles)
            )
        if sleep_cycles > Int32(0):
            nanosleep(sleep_cycles)

    @cute.jit
    def _adaptive_pace(
        self, average_window: Int32, current_window: Int32, low_window: int, high_window: int
    ) -> Int32:
        sleep_cycles = Int32(0)
        if current_window > average_window:
            sleep_cycles = current_window - average_window
            average_window = average_window + (
                (current_window - average_window + Int32(3)) // Int32(4)
            )
            if sleep_cycles > Int32(high_window):
                sleep_cycles = Int32(high_window)
        else:
            sleep_cycles = average_window - current_window
            average_window = average_window - (
                (average_window - current_window + Int32(3)) // Int32(4)
            )
        # with cute.arch.elect_one():
        # cute.printf("avg_window: {}, current_window: {}, sleep_cycles: {}", average_window, current_window,
        # sleep_cycles)
        if sleep_cycles > Int32(self.adaptive_minimum_sleep_cycles):
            nanosleep(sleep_cycles)
        if average_window > Int32(high_window):
            average_window = Int32(high_window)
        if average_window < Int32(low_window):
            average_window = Int32(low_window)
        return average_window

    @cute.jit
    def token_back(
        self,
        smem_workspace: SmemWorkspace,
        smem_base: cute.Pointer,
        tbx_row: Optional[Int32] = None,
        tbx_phase: Optional[Int32] = None,
    ) -> None:
        """Push completed FC2 data and scale rows to source ranks.

        TBX: token_in already initialized this warp's token-back mbarrier, pushed whole rows during the
        dispatch rounds and claimed `tbx_row` (not started); resume from that row with mbarrier parity `tbx_phase`."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        if cutlass.const_expr(not self.token_back_enabled):
            return
        if cutlass.const_expr(
            self.combine_format.is_quantized
            and self._token_comm_args.pre_reduced_activation_sf is None
        ):
            raise ValueError("Quantized token-back requires a scale destination.")

        global_worker_idx = (
            self._linear_cta_idx * Int32(self.transfer_warp_count) + transfer_warp_idx
        )
        global_worker_count = Int32(self.promised_launchable_sm_count * self.transfer_warp_count)
        token_back_mbarriers = smem_workspace.ptr(self.token_back_mbarrier_region, smem_base)
        worker_mbarrier = token_back_mbarriers + transfer_warp_idx
        if cutlass.const_expr(self.token_back_push_data):
            token_back_activation = smem_workspace.tensor(
                self.token_back_activation_smem_region, smem_base
            )
            worker_activation_stage = token_back_activation[transfer_warp_idx, None]
            activation_chunk_bytes = (
                cute.cosize(worker_activation_stage) * int(self.combine_format.act_dtype.width) // 8
            )
        if cutlass.const_expr(self.token_back_push_sf):
            token_back_sf = smem_workspace.tensor(self.token_back_sf_smem_region, smem_base)
            worker_sf_stage = token_back_sf[transfer_warp_idx, (None, None)]
            sf_chunk_bytes = (
                cute.cosize(worker_sf_stage) * int(self.combine_format.scale_dtype.width) // 8
            )
        if cutlass.const_expr(tbx_row is None):
            if lane_idx == Int32(0):
                cute.arch.mbarrier_init(worker_mbarrier, 1)
        cute.arch.sync_warp()

        owned_sizes = smem_workspace.tensor(self.expert_sizes_smem_region, smem_base)
        if cutlass.const_expr(self.token_back_mode == "standalone_warps"):  # nosec B105
            sizes_ready_barrier = pipeline.NamedBarrier(
                barrier_id=self.standalone_size_barrier_id,
                num_threads=2 * self.transfer_thread_count,
            )
            sizes_ready_barrier.arrive_and_wait()

        pool_expert_bases = self._router.pool_expert_base_tensor(self._device_workspace)
        token_metadata_pointer = self._router.token_src_metadata_pointer(self._device_workspace)
        fc2_done = self._device_workspace.ptr(self.fc2_done_region)
        if cutlass.const_expr(self.token_back_push_data):
            fc2_activation_pointer = self._device_workspace.ptr(self.fc2_activation_region)
            output_token_bytes = self.hidden_size * int(self.combine_format.act_dtype.width) // 8
            activation_chunk_count = ceil_div(output_token_bytes, activation_chunk_bytes)
            data_window_unit = activation_chunk_bytes * 2
            reuse_data_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps"  # nosec B105
                and data_window_unit > self.minimum_pacing_window_cycles
            )
            # Preserve the empirical low:initial:high ratio of 1:2.5:5.
            data_average_window = Int32(data_window_unit)
            data_low_window = data_window_unit * 2 // 5
            data_high_window = data_window_unit * 2
        if cutlass.const_expr(self.token_back_push_sf):
            fc2_sf_pointer = self._device_workspace.ptr(self.fc2_activation_sf_region)
            output_sf_bytes = (
                self.combine_sf_hidden_padded * int(self.combine_format.scale_dtype.width) // 8
            )
            sf_chunk_count = ceil_div(output_sf_bytes, sf_chunk_bytes)
            sf_window_unit = ceil_div(sf_chunk_bytes * 2, 3)
            reuse_sf_pacing_enabled = (
                self.token_back_mode == "reuse_dispatch_warps"  # nosec B105
                and sf_window_unit > self.minimum_pacing_window_cycles
            )
            sf_average_window = Int32(sf_window_unit)
            sf_low_window = sf_window_unit * 2 // 5
            sf_high_window = sf_window_unit * 2

        next_dense_token = global_worker_idx - global_worker_count
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            next_dense_token = Int32(0)
        if cutlass.const_expr(tbx_row is not None):
            next_dense_token = tbx_row
        else:
            next_dense_token = self.next_token(next_dense_token)
        expert_valid_begin = Int32(0)
        transfer_phase = Int32(0)
        if cutlass.const_expr(tbx_phase is not None):
            transfer_phase = tbx_phase

        iket.range_push("token_back.work")
        local_expert = Int32(0)
        # v16-fc2blk: first FC2-done slot of the current expert / last slot this warp has waited for.
        expert_block_begin = Int32(0)
        waited_block = Int32(-1)
        while local_expert < Int32(self.experts_per_rank):
            expert_token_count = owned_sizes[local_expert]
            expert_valid_end = expert_valid_begin + expert_token_count
            tail_expert = Boolean(False)
            if cutlass.const_expr(self.fc2_tail_blocks > 0):
                tail_expert = local_expert >= Int32(self.experts_per_rank - self.fc2_tail_blocks)
            if cutlass.const_expr(self.fc2_block_done):
                pass
            elif next_dense_token < expert_valid_end and not tail_expert:
                token_tile_count = (
                    expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1)
                ) // Int32(self.tokens_per_fc1_ready_slot)
                completion_target = token_tile_count * Int32(self.fc2_done_signals_per_token_tile)
                iket.range_push("token_back.wait_fc2")
                while (
                    cute.arch.load(fc2_done + local_expert, Int32, sem="acquire", scope="gpu")
                    < completion_target
                ):
                    nanosleep(500)
                iket.range_pop()

            while next_dense_token < expert_valid_end:
                token_in_expert = next_dense_token - expert_valid_begin
                if cutlass.const_expr(self.fc2_block_done):
                    row_block = expert_block_begin + token_in_expert // Int32(
                        self.tokens_per_fc1_ready_slot
                    )
                    if row_block != waited_block:
                        iket.range_push("token_back.wait_fc2")
                        while cute.arch.load(
                            fc2_done + row_block, Int32, sem="acquire", scope="gpu"
                        ) < Int32(self.fc2_done_signals_per_token_tile):
                            nanosleep(500)
                        iket.range_pop()
                        waited_block = row_block
                elif cutlass.const_expr(self.fc2_tail_blocks > 0):
                    if tail_expert:
                        row_block = Int32(self.experts_per_rank) + expert_block_begin + token_in_expert // Int32(
                            self.tokens_per_fc1_ready_slot
                        )
                        if row_block != waited_block:
                            iket.range_push("token_back.wait_fc2")
                            while cute.arch.load(
                                fc2_done + row_block, Int32, sem="acquire", scope="gpu"
                            ) < Int32(self.fc2_done_signals_per_token_tile):
                                nanosleep(500)
                            iket.range_pop()
                            waited_block = row_block
                pool_token_idx = pool_expert_bases[local_expert] + token_in_expert
                metadata = TokenSrcMetadata.load(
                    token_metadata_pointer.toint()
                    + Int64(pool_token_idx) * Int64(TokenSrcMetadata.nbytes)
                )
                destination_topk = metadata.src_topk
                if cutlass.const_expr(self.reduce_topk_in_kernel):
                    destination_topk = Int32(0)
                peer_offset = self._token_comm_args.peer_rank_ptr_mapper.map(
                    Int64(0), metadata.src_rank, Int64(0)
                )
                is_remote_token = metadata.src_rank != self._local_rank

                if cutlass.const_expr(self.token_back_push_data):
                    iket.range_push("token_back.push_data")
                    local_activation_address = fc2_activation_pointer.toint() + Int64(
                        pool_token_idx
                    ) * Int64(output_token_bytes)
                    remote_activation = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation.dtype,
                            self._token_comm_args.pre_reduced_activation.iterator.toint()
                            + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(
                                self._token_comm_args.pre_reduced_activation.iterator.max_alignment
                            ),
                        ),
                        self._token_comm_args.pre_reduced_activation.layout,
                    )
                    destination_row = remote_activation[
                        Int64(metadata.src_token), destination_topk, None
                    ]
                    for chunk_idx in cutlass.range_constexpr(activation_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * activation_chunk_bytes)
                        chunk_bytes_this_round = min(
                            activation_chunk_bytes,
                            output_token_bytes - chunk_idx * activation_chunk_bytes,
                        )
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_data_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"  # nosec B105
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(
                            reuse_data_pacing_enabled or stateless_data_pacing_enabled
                        ):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                worker_mbarrier, current_chunk_bytes
                            )
                            tma_load_1d(
                                worker_activation_stage.iterator,
                                local_activation_address + chunk_byte_offset,
                                worker_mbarrier,
                                current_chunk_bytes,
                            )
                        cute.arch.mbarrier_wait(worker_mbarrier, transfer_phase)
                        destination_chunk = cute.make_ptr(
                            cutlass.Uint8,
                            destination_row.iterator.toint() + chunk_byte_offset,
                            AddressSpace.gmem,
                            assumed_align=16,
                        )
                        with cute.arch.elect_one():
                            if cutlass.const_expr(self.reduce_topk_in_kernel):
                                cp_reduce_async_bulk_add_bf16_s2g(
                                    destination_chunk,
                                    worker_activation_stage.iterator,
                                    current_chunk_bytes,
                                )
                            else:
                                cp_async_bulk_s2g(
                                    destination_chunk,
                                    worker_activation_stage.iterator,
                                    current_chunk_bytes,
                                )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        transfer_phase = transfer_phase ^ Int32(1)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(reuse_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                data_average_window = self._adaptive_pace(
                                    data_average_window,
                                    current_window,
                                    data_low_window,
                                    data_high_window,
                                )
                        elif cutlass.const_expr(stateless_data_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                self._stateless_pace(Int32(current_window_unit), current_window)
                    iket.range_pop()

                if cutlass.const_expr(self.token_back_push_sf):
                    iket.range_push("token_back.push_sf")
                    local_sf_address = fc2_sf_pointer.toint() + Int64(pool_token_idx) * Int64(
                        output_sf_bytes
                    )
                    remote_sf = cute.make_tensor(
                        cute.make_ptr(
                            self._token_comm_args.pre_reduced_activation_sf.dtype,
                            self._token_comm_args.pre_reduced_activation_sf.iterator.toint()
                            + peer_offset,
                            AddressSpace.gmem,
                            assumed_align=(
                                self._token_comm_args.pre_reduced_activation_sf.iterator.max_alignment
                            ),
                        ),
                        self._token_comm_args.pre_reduced_activation_sf.layout,
                    )
                    destination_sf_row = remote_sf[
                        Int64(metadata.src_token), destination_topk, None
                    ]
                    for chunk_idx in cutlass.range_constexpr(sf_chunk_count):
                        chunk_byte_offset = Int64(chunk_idx * sf_chunk_bytes)
                        chunk_bytes_this_round = min(
                            sf_chunk_bytes, output_sf_bytes - chunk_idx * sf_chunk_bytes
                        )
                        current_chunk_bytes = Int32(chunk_bytes_this_round)
                        current_window_unit = ceil_div(chunk_bytes_this_round * 2, 3)
                        stateless_sf_pacing_enabled = (
                            self.token_back_mode != "reuse_dispatch_warps"  # nosec B105
                            and current_window_unit > self.minimum_pacing_window_cycles
                        )
                        round_start_clock = Int64(0)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(
                            reuse_sf_pacing_enabled or stateless_sf_pacing_enabled
                        ):
                            if is_remote_token:
                                round_start_clock = read_clock64()
                            else:
                                round_start_clock = round_start_clock
                        with cute.arch.elect_one():
                            cute.arch.mbarrier_arrive_and_expect_tx(
                                worker_mbarrier, current_chunk_bytes
                            )
                            tma_load_1d(
                                worker_sf_stage.iterator,
                                local_sf_address + chunk_byte_offset,
                                worker_mbarrier,
                                current_chunk_bytes,
                            )
                        cute.arch.mbarrier_wait(worker_mbarrier, transfer_phase)
                        destination_chunk = cute.make_ptr(
                            cutlass.Uint8,
                            destination_sf_row.iterator.toint() + chunk_byte_offset,
                            AddressSpace.gmem,
                            assumed_align=16,
                        )
                        with cute.arch.elect_one():
                            cp_async_bulk_s2g(
                                destination_chunk, worker_sf_stage.iterator, current_chunk_bytes
                            )
                        cute.arch.cp_async_bulk_commit_group()
                        cute.arch.cp_async_bulk_wait_group(0)
                        transfer_phase = transfer_phase ^ Int32(1)
                        cute.arch.sync_warp()
                        if cutlass.const_expr(reuse_sf_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                sf_average_window = self._adaptive_pace(
                                    sf_average_window, current_window, sf_low_window, sf_high_window
                                )
                        elif cutlass.const_expr(stateless_sf_pacing_enabled):
                            if is_remote_token:
                                current_window = Int32(read_clock64() - round_start_clock)
                                self._stateless_pace(Int32(current_window_unit), current_window)
                    iket.range_pop()

                cute.arch.sync_warp()
                next_dense_token = self.next_token(next_dense_token)

            if cutlass.const_expr(self.fc2_block_done or self.fc2_tail_blocks > 0):
                expert_block_begin = expert_block_begin + (
                    (expert_token_count + Int32(self.tokens_per_fc1_ready_slot - 1))
                    // Int32(self.tokens_per_fc1_ready_slot)
                )
            expert_valid_begin = expert_valid_end
            local_expert = local_expert + Int32(1)
        iket.range_pop()
        # with cute.arch.elect_one():
        #     cute.printf(" final data_average_window: {} ", data_average_window)

    @cute.jit
    def next_token(self, current_token: Int32) -> Int32:
        global_worker_count = self.promised_launchable_sm_count * self.transfer_warp_count
        schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            schedule_counter = self._device_workspace.ptr(self.token_back_schedule_region)
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):  # nosec B105
            claimed_token = Int32(0)
            if self._lane_idx == Int32(0):
                claimed_token = cute.arch.atomic_add(
                    schedule_counter, Int32(1), sem="relaxed", scope="gpu"
                )
            return Int32(cute.arch.shuffle_sync(claimed_token, Int32(0)))
        return current_token + global_worker_count

    @cute.jit
    def reset_next_epoch(self) -> None:
        """v33-epoch: zero the other parity copy of every tail_reset region (launch e - 1's state, reused by e + 1).

        Called by the token-in transfer warps at kernel start, before they wait for the router. Safe: launch e - 1's
        kernel and router completed before this kernel started (stream order), every peer write into our shared copy
        of launch e - 1 (router size / ready / metadata counters) was awaited by our launch e - 1, and peers write that
        copy again only in their router e + 1, which starts after their MegaMoE e passed the tail barrier, i.e. after
        we arrived there (after these stores; grid-sync release + sys fence + NVLink signal chain).
        """
        # Thread / CTA indices re-read here (not the members bound at kernel entry): keeps their live ranges out of
        # the epilogue code, which ptxas otherwise spills into (STL per FC1 subtile).
        thread_idx, _, _ = cute.arch.thread_idx()
        transfer_thread_idx = thread_idx % Int32(self.transfer_thread_count)
        linear_cta_idx = self._linear_cta_idx
        total_reset_threads = self.promised_launchable_sm_count * self.transfer_thread_count
        global_reset_thread = linear_cta_idx * Int32(self.transfer_thread_count) + transfer_thread_idx
        self._device_workspace.reset_next_epoch_space("shared", global_reset_thread, total_reset_threads)
        self._device_workspace.reset_next_epoch_space("local", global_reset_thread, total_reset_threads)

    @cute.jit
    def reset_tail(self) -> None:
        """Reset communication state with the four token-in transfer warps."""
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        transfer_thread_idx = transfer_warp_idx * Int32(32) + lane_idx
        if cutlass.const_expr(_epoch.enabled()):
            self._reset_tail_epoch(transfer_thread_idx)
        else:
            self._reset_tail_stock(transfer_thread_idx)

    @cute.jit
    def _reset_tail_epoch(self, transfer_thread_idx: Int32) -> None:
        """v33-epoch tail: one grid sync + one cross-rank barrier, no zero pass, then advance the epoch.

        The barrier is still required: (1) every peer's token-back pushes into our combine plane must be complete
        before our TopkReduce reads it (the kernel exits only after CTA 0 passes it); (2) peers' router e + 1 must not
        overwrite our token_src_metadata / fc1_topk_scores / sizes_by_rank before our launch e stopped reading them,
        nor touch our next tail copy before we zeroed it (both happen before our arrival).
        """
        iket.range_push("tail.nvlink_drain")
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=True,
            epilogue_grid_sync=False,
        )
        iket.range_pop()
        # Every CTA of this rank passed the prologue grid sync, i.e. every thread performed its epoch load (kernel-entry
        # bar.sync) and every tail access of this launch; peers' router e + 1 (the next reader) starts after this
        # kernel completes.
        if self._linear_cta_idx == Int32(0) and transfer_thread_idx == Int32(0):
            self._device_workspace.bump_epoch()
        if cutlass.const_expr(_kimi_options.environ.get("MEGA_USE_NCU", "0") == "1"):
            self._nvlink_barrier.finalize(
                1,
                self.transfer_thread_count,
                Int32(self.promised_launchable_sm_count),
                self._linear_cta_idx,
                transfer_thread_idx,
            )

    @cute.jit
    def _reset_tail_stock(self, transfer_thread_idx: Int32) -> None:
        transfer_warp_idx = self._transfer_warp_idx
        lane_idx = self._lane_idx
        iket.range_push("tail.nvlink_drain")
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=True,
            epilogue_grid_sync=False,
        )
        iket.range_pop()
        total_reset_threads = self.promised_launchable_sm_count * self.transfer_thread_count
        global_reset_thread = (
            self._linear_cta_idx * Int32(self.transfer_thread_count) + transfer_thread_idx
        )
        iket.range_push("tail.reset_workspace")
        self._device_workspace.reset_tail_space("shared", global_reset_thread, total_reset_threads)
        self._device_workspace.reset_tail_space("local", global_reset_thread, total_reset_threads)
        iket.range_pop()
        iket.range_push("tail.nvlink_publish")
        self._nvlink_barrier.arrive_and_wait(
            self.transfer_thread_count,
            Int32(self.promised_launchable_sm_count),
            self._linear_cta_idx,
            transfer_thread_idx,
            prologue_grid_sync=True,
            epilogue_grid_sync=False,
        )
        iket.range_pop()
        if cutlass.const_expr(_kimi_options.environ.get("MEGA_USE_NCU", "0") == "1"):
            iket.range_push("tail.ncu_finalize")
            self._nvlink_barrier.finalize(
                2,
                self.transfer_thread_count,
                Int32(self.promised_launchable_sm_count),
                self._linear_cta_idx,
                transfer_thread_idx,
            )
            iket.range_pop()


__all__ = ["TokenBackMode", "TokenBackScheduleMode", "TokenCommArgs", "TokenCommNonDeterministic"]
