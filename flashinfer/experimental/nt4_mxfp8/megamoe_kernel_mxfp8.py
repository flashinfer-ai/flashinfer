# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Owned NT4 adaptation of the matching kernel-team drop file.
# Upstream snapshot: 50d583aa0. See README.md for provenance.
# Extracted from research commit 52e7da3f4; GPU parity must be requalified.
"""Local MXFP8 dispatch, fused FC1/activation/FC2, and detached cleanup.

Reuses the vendor token-pull machinery with world_size=1. The caller owns NVLS;
epilogue stores preserve separate BF16 contributions for FP32 weighted assembly.
Workspace layout and dispatch readiness follow the existing MegaMoE protocol.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here
# (PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``
# element-type introspection).  See moe_nvfp4_swapab/megamoe_kernel.py.

from typing import Any, Dict, List, Optional, Tuple

import triton
import triton.language as tl

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64, Int32


from .kernel_mxfp8_glu_fc12 import Sm100ClampedRelu2Mxfp8Fc12Kernel
from src.token_comm import (
    CombineFormat,
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenInPullTokenBackPush,
)

# Reuse the region-layout helpers + module constants from the NVFP4 mega kernel
# so the two paths stay byte-for-byte consistent in their workspace plumbing.
from moe_nvfp4_swapab.megamoe_kernel import (
    _RegionSpec,
    _round_up,
    _layout_regions,
    _DispatchWarpCount,
    _TokenMetadataBytes,
    _GridSyncSlotCount,
    _NvlinkSlotCount,
)

# =============================================================================
# Sm100Nt4MegaMoEMxfp8Kernel
# =============================================================================


class Sm100Nt4MegaMoEMxfp8Kernel(Sm100ClampedRelu2Mxfp8Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel (MXFP8)."""

    def __init__(
        self, group_hint: int, *, max_tokens_per_rank: int, hardware_tanh: bool
    ) -> None:
        super().__init__(group_hint)
        self.hardware_tanh = hardware_tanh
        self.detached_tail = True
        self.combine_format = CombineFormat.parse("bf16")
        self.token_back_mode = "epi_warps"
        self.sf_block_pad = None
        self.token_back_enabled = False
        self.token_back_schedule_mode = "static"
        self.world_size = 1
        self.local_rank = 0
        self.num_topk = 10
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = self.hidden_bytes = 768
        self.num_experts_per_rank = self.num_total_experts = 128
        self.intermediate_gateup = self.intermediate_downproj = 4096
        self.sf_uint32_per_token = 6
        # Non-swap-AB readiness counters count M-axis token tiles.
        self.cluster_tile_tokens = 128

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()

        # Cohabiting warps outside the dispatch group: epilogue + mma + tma_a
        # + tma_b + sched.
        num_other_warps = len(self.epilogue_warp_id) + 1 + 1 + 1 + 1

        self.token_comm = TokenInPullTokenBackPush(
            world_size=self.world_size,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=self.ab_dtype,
            combine_format=self.combine_format,
            token_back_by_dispatch=False,
            fc2_publishes_per_token_cluster_tile=0,
            token_back_standalone=self.token_back_standalone,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=self.dispatch_warp_id[0],
            num_other_warps=num_other_warps,
            is_swap_ab=False,
            flag_batch=1,
            token_back_schedule_mode=self.token_back_schedule_mode,
        )

        # Region layout (same call drives both get_workspace_sizes() and the
        # __call__ partition).
        self._local_region_specs = self._build_local_region_specs()
        self._shared_region_specs = self._build_shared_region_specs()
        self._local_offsets, self._local_total = _layout_regions(
            self._local_region_specs
        )
        self._shared_offsets, self._shared_total = _layout_regions(
            self._shared_region_specs
        )
        self._local_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._local_region_specs
        }
        self._shared_region_by_name: Dict[str, _RegionSpec] = {
            r.name: r for r in self._shared_region_specs
        }
        local_leading = self._local_offsets["l1_token_buffer"]
        shared_leading = self._shared_offsets["src_token_topk_idx"]
        self.local_zero_i32_count = local_leading // 4
        self.shared_zero_i32_count = shared_leading // 4

    def sched_ext_fc1_peek_threshold(self) -> int:
        # Peek threshold must match the spin threshold (cluster_tile_tokens = 128)
        # so an early peek hit does not skip the spin and expose stale pool rows.
        return self.cluster_tile_tokens

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM for dispatch pull mbarriers, expert scratch, and token buffer.

        Must match ``TokenInPullTokenBackPush.extra_smem_storage_class``:
        ``pull_mbar[Int64, 4] + smem_expert_count[Int32, num_total_experts]
        + pull_buffer[Uint8, 4 * hidden_bytes]``.
        Standalone token-back adds ``tb_pull_mbar[Int64, 4]`` and
        ``tb_pull_buffer[Uint8, 4 * tb_chunk_bytes]``.
        """
        pull_mbar_bytes = _DispatchWarpCount * 8
        expert_count_bytes = self.num_total_experts * 4
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        total = (
            _round_up(pull_mbar_bytes, 16)
            + _round_up(expert_count_bytes, 16)
            + _round_up(pull_buffer_bytes, 128)
        )
        return total

    def _smem_misc_budget_bytes(self) -> int:
        """Base misc reservation plus dispatch-warp SMEM."""
        return super()._smem_misc_budget_bytes() + self._dispatch_smem_bytes()

    # =========================================================================
    # Pool sizing (first-principles; identical to the NVFP4 path)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = max_recv * max_per_token + num_experts_per_rank * (
            token_padding_block - 1
        )
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = (
            pool_token_capacity // token_padding_block
        ) * sf_padding_block
        cluster_m = self.cluster_shape_mn[0]  # noqa: F841
        pool_task_tile_capacity = (
            pool_token_capacity + cluster_tile_tokens - 1
        ) // cluster_tile_tokens + num_experts_per_rank
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        mma_tiler_m = self.mma_tiler_mnk[0]
        sf_vec_size = self.sf_vec_size
        sf_padding_block = self.sf_padding_block

        sf_total_rows_upper = (
            pool_token_capacity + num_experts_per_rank * sf_padding_block
        )
        sf_block_cols = (((intermediate_downproj // sf_vec_size) + 3) // 4) * 4
        fc1_done_slots = (
            pool_token_capacity + mma_tiler_m - 1
        ) // mma_tiler_m + num_experts_per_rank

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

        # Data buffers. l1_token_buffer must be the first data region because
        # __init__ derives local_zero_i32_count from its offset.
        specs += [
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored as
            # Uint8 bytes; the fp8 view at the same offset is built in __call__.
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
            # 1D Int32 atom-flat SF buffer (dispatch_pull's 32b read/write); the
            # E8M0 view is built at the same offset in __call__.
            _RegionSpec(
                "l1_sf_buffer",
                cutlass.Int32,
                (pool_sf_capacity * sf_uint32_per_token,),
                16,
            ),
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
            # MXFP8: fc1_output uses ab_dtype (Float8E4M3FN or Float8E5M2).
            _RegionSpec(
                "fc1_output",
                self.ab_dtype,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            # MXFP8: fc1_output_sf uses Float8E8M0FNU (E8M0 block scales).
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float8E8M0FNU,
                (sf_total_rows_upper, sf_block_cols),
                128,
            ),
        ]

        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        max_slot = max_tokens_per_rank * num_topk

        # Shared counter prefix is bulk-zeroed between tail barriers; keep
        # src_token_topk_idx as the first data region used to derive the prefix.
        specs = [
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
        specs.append(
            _RegionSpec(
                "combine_quant",
                self.combine_format.act_dtype,
                (max_tokens_per_rank, num_topk, self.hidden),
                128,
            )
        )
        return specs

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)``."""
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers (mirror the NVFP4 mega kernel)
    # =========================================================================

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
        dt = cute_dtype if cute_dtype is not None else spec.cute_dtype
        sh = shape if shape is not None else spec.shape
        st = stride
        if st is None:
            if cute_dtype is None and shape is None:
                st = spec.stride_row_major
            else:
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace,
            offsets[spec.name],
            dt,
            sh,
            st,
            spec.align,
        )

    # =========================================================================
    # __call__
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,  # (T, hidden) fp8
        activation_sf: cute.Tensor,  # (T, round_up(hidden, sf_atom_block_k)) E8M0
        topk_idx: cute.Tensor,  # (T, num_topk) Int64
        topk_weights: cute.Tensor,  # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_c: Optional[cute.Tensor],  # fc1 c output
        # Final combined output the caller consumes: 2D (T, hidden) BF16.
        # Opaque workspaces.
        local_workspace: cute.Tensor,  # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,  # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
    ) -> None:
        """Dispatch local tokens and write separate BF16 routing contributions.

        The world-size-one pointer mapper preserves the vendor dispatch layout.
        Detached cleanup completes before the caller's FP32 weighted assembly.
        """
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        sf_per_token_fp8 = self.sf_uint32_per_token * 4  # 4 E8M0 SFs per Int32

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + fp8 view
        # (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(local_workspace, "l1_token_buffer")
        l1_token_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            self.ab_dtype,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        # L1 SF buffer: Int32 view (dispatch_pull's [j, t] 2D indexing) + E8M0
        # view (base.activation_sf re-views via tile_atom_to_shape_SF off the
        # iterator, so the stride here is informational only).
        l1_sf_buffer_i32 = self._view_local(local_workspace, "l1_sf_buffer")
        l1_sf_buffer_e8m0 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_sf_buffer"],
            cutlass.Float8E8M0FNU,
            (pool_sf_capacity, sf_per_token_fp8),
            (sf_per_token_fp8, 1),
            self._local_region_by_name["l1_sf_buffer"].align,
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
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(local_workspace, "fc1_done_counter")

        load_balance_counter: Optional[cute.Tensor] = None

        token_back_schedule_counter = None
        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace,
                "token_back_schedule_counter",
            ).iterator

        combine_target = self._view_shared(shared_workspace, "combine_quant")

        fc2_output_workspace_u8 = None
        combine_output_u8 = cute.recast_tensor(combine_target, cutlass.Uint8)
        fc2_output_target = combine_target

        fc2_done_counter = None

        fc2_output_sf = None
        combine_sf = None

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
            input_sf_buffer=activation_sf,
            topk_idx=topk_idx,
            input_topk_weights_buffer=topk_weights,
            expert_send_count=expert_send_count,
            expert_recv_count=expert_recv_count,
            expert_recv_count_sum=expert_recv_count_sum,
            src_token_topk_idx=src_token_topk_idx,
            fc1_input_token_buffer=l1_token_buffer_u8,
            fc1_input_sf_buffer=l1_sf_buffer_i32,
            fc1_input_topk_weights_buffer=l1_topk_weights_buffer,
            fc1_ready_counter=l1_arrival_count,
            token_src_metadata=token_src_metadata,
            combine_output=combine_output_u8,
            combine_sf=combine_sf,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_output_sf=fc2_output_sf,
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
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        super().__call__(
            activation=l1_token_buffer_fp8,
            fc1_weight=fc1_weight,
            activation_sf=l1_sf_buffer_e8m0,
            fc1_weight_sf=fc1_weight_sf,
            fc1_output=fc1_output,
            fc1_output_sf=fc1_output_sf,
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

        # Stream ordering drains every compute CTA before the lightweight
        # finalizer starts. No compute SMEM/TMEM allocation survives this boundary.
        self.detached_tail_kernel(token_comm_args).launch(
            grid=(1, 1, 1),
            block=(128, 1, 1),
            stream=stream,
        )

    @cute.jit
    def detached_peer_barrier(self, args, warp_idx, lane_idx):
        # Single CTA: no software grid barrier and no dependence on the original
        # compute grid size. Preserve the existing cross-PE signal phase counter.
        cute.arch.sync_threads()
        self.token_comm.nvlink_barrier(
            args.nvlink_barrier_signal,
            args.nvlink_barrier_counter,
            args.grid_sync_counter,
            args.peer_rank_ptr_mapper,
            Int32(0),
            warp_idx,
            lane_idx,
            num_sms=1,
            prologue_grid_sync=False,
            epilogue_grid_sync=False,
        )
        cute.arch.sync_threads()

    @cute.kernel
    def detached_tail_kernel(self, args):
        tidx = cute.arch.thread_idx()[0]
        warp_idx = cute.arch.warp_idx()
        lane_idx = cute.arch.lane_idx()
        self.detached_peer_barrier(args, warp_idx, lane_idx)
        i = Int32(tidx)
        while i < Int32(cute.size(args.shared_zero_prefix)):
            args.shared_zero_prefix[i] = Int32(0)
            i += Int32(128)
        cute.arch.fence_acq_rel_sys()
        self.detached_peer_barrier(args, warp_idx, lane_idx)
        # Reset local state only after the last barrier has stopped using it.
        i = Int32(tidx)
        while i < Int32(cute.size(args.local_zero_prefix)):
            args.local_zero_prefix[i] = Int32(0)
            i += Int32(128)


@triton.jit
def _route_mask(
    Ids,
    Probs,
    LocalIds,
    LocalProbs,
    N: tl.constexpr,
    RANK: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    expert = tl.load(Ids + i, i < N, other=-1)
    owned = (expert >= RANK * 128) & (expert < (RANK + 1) * 128)
    p = tl.load(Probs + i, (i < N) & owned, other=0.0)
    tl.store(LocalIds + i, tl.where(owned, expert - RANK * 128, -1), i < N)
    tl.store(LocalProbs + i, p, i < N)


@triton.jit
def _weighted_partial(
    Terms, Ids, Probs, Output, H: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    h = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for slot in tl.static_range(K):
        valid = tl.load(Ids + row * K + slot) >= 0
        p = tl.load(Probs + row * K + slot, valid, other=0.0)
        value = tl.load(
            Terms + (row * K + slot) * H + h, (h < H) & valid, other=0.0
        ).to(tl.float32)
        acc = acc + value * p
    tl.store(Output + row * H + h, acc, h < H)


def route_mask(ids, probs, local_ids, local_probs, rank):
    _route_mask[(triton.cdiv(ids.numel(), 256),)](
        ids, probs, local_ids, local_probs, ids.numel(), rank, 256
    )


def weighted_partial(terms, ids, probs, output):
    # Contiguous [row, slot, channel] BF16 -> FP32, in slot order. Keep
    # multiply/add separate to make the numerical contract explicit.
    _weighted_partial[(ids.shape[0], triton.cdiv(output.shape[1], 256))](
        terms,
        ids,
        probs,
        output,
        output.shape[1],
        ids.shape[1],
        256,
        enable_fp_fusion=False,
    )
