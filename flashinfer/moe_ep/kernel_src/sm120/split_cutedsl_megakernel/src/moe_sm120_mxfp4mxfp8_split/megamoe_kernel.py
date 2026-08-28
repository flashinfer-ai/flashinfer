# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""MegaMoE fused dispatch + fc1 + fc2 + combine kernel.

The base class owns the local fc1/fc2 GEMM pipeline.  This subclass owns the
token-communication hooks, workspace partitioning, and the MegaMoE argument
bundle.  ``static_expert_shape`` is required because dispatch storage and pool
sizes are codegen-time quantities.

Shared / local workspace split:

  SHARED  : src_token_topk_idx, expert_recv_count, expert_recv_count_sum,
            nvlink_barrier_signal
  LOCAL   : expert_send_count, grid_sync_counter, l1_token_buffer,
            l1_sf_buffer, l1_topk_weights_buffer, l1_arrival_count,
            token_src_metadata, fc1_output, fc1_output_sf,
            fc1_done_counter, (optionally) load_balance_counter

User tensors are not in the opaque workspaces. ``activation``,
``activation_sf``, ``topk_weights``, and ``combine_output`` must be reachable
through the symmetric-heap peer mapper; ``topk_idx`` and weights are local.

Dispatch/pool alignment constraints are unified at construction time:
``token_padding_block`` (base) and ``block_m`` (dispatch) become the
same constant, similarly for ``sf_padding_block`` / ``sf_block_m``;
C3 reduces to a divisibility check that ``cluster_tile_tokens`` is a
multiple of ``token_padding_block``.
"""

# NOTE: ``from __future__ import annotations`` is intentionally NOT used here.
# PEP 563 string-ifies class-body annotations, which breaks ``@cute.struct``'s
# element-type introspection (it reads ``__annotations__`` and demands the
# values be live ``cute.struct.MemRange[...] / struct / array / base_dsl
# scalar`` objects, not their string forms).  The lean fc1+fc2 base
# (``kernel_fc12.py``) and the dispatch standalone (``src/dispatch_kernel.py``)
# both already follow this convention.  Self-references (the single
# ``"TokenCommArgs"`` forward ref on ``__new_from_mlir_values__``) stay
# quoted explicitly.

import dataclasses
from typing import Any, Dict, List, Literal, Optional, Tuple, Type

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cute.typing import AddressSpace
from cutlass.cutlass_dsl import Int64

from .kernel_fc12 import Sm120SwapABSwigluMxfp8Fc12Kernel
from .jit_config import Sm120JitConfig
from .moe_utils import spin_wait
from .token_comm import Sm120SysmemTokenInPullTokenBackPush
from src.token_comm import (
    TokenCommArgs as ExtractedTokenCommArgs,
    TokenSrcMetadata,
)


# =============================================================================
# Module-level constants.
# =============================================================================

# NamedBarrier IDs.  Base reserves 1-7; this subclass uses 8 and 9.
_KernelTailNamedBarrierId = 8        # 12-warp rendezvous (384 threads)
_DispatchToSchedNamedBarrierId = 9   # 4 dispatch + 1 sched (160 threads)

# Dispatch warp count.
_DispatchWarpCount = 4

# Per-pool-slot provenance record consumed by combine STG redirect (S3) and
# token-back; one i64 = {src_rank, src_token, src_topk} (see TokenSrcMetadata).
_TokenMetadataBytes = TokenSrcMetadata.nbytes

# NVLink signal slots used by the DeepGEMM-style phase/sign barrier.
# A separate local counter selects phase/sign; the signal slots are not reset
# by tail cleanup.
_NvlinkSlotCount = 2

# Grid-sync counter slots. ``software_grid_sync`` phase-flips bit 31 within
# each slot; split K1 and K2 use separate slots so concurrent grids cannot
# advance one another's phase.
_GridSyncSlotCount = 2


# =============================================================================
# Region spec + layout helpers
# =============================================================================


@dataclasses.dataclass(frozen=True)
class _RegionSpec:
    """One region in either the local or shared workspace.

    Byte size = ``ceil(numel * cute_dtype.width / 8)``.  ``align`` is
    the region's start-byte alignment (TMA store / load destinations
    want 128 B; counters / metadata want 16 B).
    """

    name: str
    cute_dtype: Any
    shape: Tuple[int, ...]
    align: int

    @property
    def numel(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n

    @property
    def stride_row_major(self) -> Tuple[int, ...]:
        """Row-major stride matching ``shape`` (rightmost dim contiguous)."""
        if len(self.shape) == 0:
            return ()
        out: List[int] = [1]
        for d in reversed(self.shape[1:]):
            out.append(out[-1] * d)
        out.reverse()
        return tuple(out)

    @property
    def nbytes(self) -> int:
        bits = self.numel * int(self.cute_dtype.width)
        return (bits + 7) // 8


def _round_up(x: int, m: int) -> int:
    return ((x + m - 1) // m) * m


def _layout_regions(
    regions: List[_RegionSpec],
) -> Tuple[Dict[str, int], int]:
    """Place ``regions`` sequentially honouring each region's ``align``.
    Returns ``(name -> byte_offset)`` and the total byte count (rounded
    up to 16 B for downstream safety).

    Drives both ``get_workspace_sizes()`` (total only) and the
    ``__call__`` partition (offsets) -- keeping the host allocation
    and the device view construction in sync without any explicit
    handshake.
    """
    offsets: Dict[str, int] = {}
    cursor = 0
    for r in regions:
        cursor = _round_up(cursor, r.align)
        offsets[r.name] = cursor
        cursor += r.nbytes
    total = _round_up(cursor, 16)
    return offsets, total


# =============================================================================
# Sm120MegaMoEMxfp8SwapABKernel
# =============================================================================


class Sm120MegaMoEMxfp8SwapABKernel(Sm120SwapABSwigluMxfp8Fc12Kernel):
    """MegaMoE-complete fused dispatch + fc1 + fc2 + combine kernel."""

    def __init__(
        self,
        # Base-class kwargs (forwarded 1:1 to ``super().__init__``).
        mma_tiler_mnk: Tuple[int, int, int],
        cluster_shape_mnk: Tuple[int, int, int],
        use_2cta_instrs: bool,
        group_hint: int,
        token_padding_block: int,
        sf_padding_block: int,
        load_balance_mode: str = "static",
        static_expert_shape: Optional[Tuple[int, int, int]] = None,
        force_static_sched: bool = True,
        clc_bundle_size: Optional[int] = None,
        num_sched_stages: Optional[int] = None,
        num_ab_stages_override: Optional[int] = None,
        acc_dtype: Type[cutlass.Numeric] = cutlass.Float32,
        sf_vec_size: int = 32,
        scenario: str = "2Dx3D",
        # MegaMoE-specific independent constants.
        *,
        world_size: int,
        local_rank: int,
        num_topk: int,
        max_tokens_per_rank: int,
        hidden: int,
        comm_backend: Literal["p2p_direct", "nvshmem_ibgda"],
        ibgda_dispatch_chunk_tokens: int,
        fc2_output_dtype: Type[cutlass.Numeric],
        non_ubulk_fc2_store: bool = True,
        in_kernel_fc2_reduce: bool = False,
        token_back_mode: Literal[
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ] = "epi_warps",
        apply_topk_in_fc1: bool = True,
        gate_up_clamp: Optional[float] = None,
        epi_flag_batch: Optional[Tuple[int, int]] = (1, 1),
        flag_batch: int = 1,
        dispatch_pull_mode: Literal[
            "auto", "token_strided", "tile_cooperative"
        ] = "auto",
        dispatch_warps_per_tile: int = 8,
        dispatch_compute_overlap: Optional[bool] = None,
        fc1_ready_tile_tokens: Optional[int] = None,
        fc1_producer_tile_tokens: Optional[int] = None,
        k2_token_tile_tokens: Optional[int] = None,
        split_role: Literal["fused", "k1", "k2"] = "fused",
        producer_sm_count: Optional[int] = None,
        compact_k2: bool = True,
        k2_tail_reclaim: bool = False,
        skip_global_tail: bool = False,
        k1_ready_queue_workspace: bool = True,
        k2_ready_queue: bool = True,
        k2_ready_queue_bundle: int = 16,
        k2_natural_regs: bool = False,
        k2_min_blocks_per_sm: int = 1,
        green_trace_role: Optional[int] = None,
        k1_ready_queue_m_rotation: int = 0,
        jit_config: Optional[Sm120JitConfig] = None,
    ) -> None:
        if comm_backend not in ("p2p_direct", "nvshmem_ibgda"):
            raise ValueError(
                "comm_backend must be p2p_direct or nvshmem_ibgda, got "
                f"{comm_backend!r}."
            )
        if static_expert_shape is None:
            raise NotImplementedError(
                "Sm120MegaMoEMxfp8SwapABKernel currently requires "
                "static_expert_shape != None (dynamic-shape MegaMoE is "
                "not wired)."
            )
        # Keep the explicit ``hidden`` kwarg in lockstep with static shape;
        # dispatch SMEM sizing reads it before tensor layouts are rewritten.
        if hidden != static_expert_shape[2]:
            raise ValueError(
                f"hidden ({hidden}) must equal "
                f"static_expert_shape[2] ({static_expert_shape[2]})."
            )
        if split_role not in ("fused", "k1", "k2"):
            raise ValueError(
                "split_role must be one of 'fused' / 'k1' / 'k2', got "
                f"{split_role!r}."
            )
        if split_role == "k2" and producer_sm_count is None:
            raise ValueError(
                "producer_sm_count is required for split_role='k2'."
            )
        if (
            split_role != "fused"
            and token_back_mode != "epi_warps"
            and not (
                comm_backend == "nvshmem_ibgda"
                and token_back_mode == "reuse_dispatch_warps"
            )
        ):
            raise ValueError(
                "Split K1/K2 require token_back_mode='epi_warps', except "
                "the nvshmem_ibgda backend which requires "
                "'reuse_dispatch_warps'."
            )
        if comm_backend == "nvshmem_ibgda" and token_back_mode != "reuse_dispatch_warps":
            raise ValueError(
                "nvshmem_ibgda requires token_back_mode='reuse_dispatch_warps'."
            )
        if comm_backend == "nvshmem_ibgda" and in_kernel_fc2_reduce:
            raise ValueError(
                "nvshmem_ibgda requires the independent K3 top-k reduce path."
            )
        if (
            split_role != "fused"
            and load_balance_mode != "static"
            and not (
                split_role == "k2"
                and k2_tail_reclaim
                and load_balance_mode == "atomic_counter"
            )
        ):
            raise NotImplementedError(
                "Split K1 requires the static scheduler. K2 may use the "
                "atomic scheduler only for shared-queue tail reclaim."
            )

        # token_back_mode selects where the cross-rank fc2 push-back runs:
        #   epi_warps            -> epilogue warps STG directly to the peer
        #   standalone_warps     -> dedicated warp group 12-15, concurrent
        #                           with dispatch_pull
        #   reuse_dispatch_warps -> dispatch warps 8-11 push after dispatch_pull
        # The two non-epi modes both stage fc2 to a local workspace first, i.e.
        # token_back_by_dispatch=True; epi_warps keeps the epilogue STG redirect.
        if token_back_mode not in (
            "epi_warps", "standalone_warps", "reuse_dispatch_warps"
        ):
            raise ValueError(
                f"token_back_mode must be 'epi_warps', 'standalone_warps', "
                f"or 'reuse_dispatch_warps'; got {token_back_mode!r}."
            )
        token_back_by_dispatch = token_back_mode != "epi_warps"

        super().__init__(
            mma_tiler_mnk=mma_tiler_mnk,
            cluster_shape_mnk=cluster_shape_mnk,
            use_2cta_instrs=use_2cta_instrs,
            group_hint=group_hint,
            token_padding_block=token_padding_block,
            sf_padding_block=sf_padding_block,
            load_balance_mode=load_balance_mode,
            static_expert_shape=static_expert_shape,
            force_static_sched=force_static_sched,
            clc_bundle_size=clc_bundle_size,
            num_sched_stages=num_sched_stages,
            num_ab_stages_override=num_ab_stages_override,
            acc_dtype=acc_dtype,
            sf_vec_size=sf_vec_size,
            scenario=scenario,
            weight_dtype=cutlass.Float4E2M1FN,
            activation_dtype=cutlass.Float8E4M3FN,
            fc2_output_dtype=fc2_output_dtype,
            non_ubulk_fc2_store=non_ubulk_fc2_store,
            in_kernel_fc2_reduce=in_kernel_fc2_reduce,
            token_back_by_dispatch=token_back_by_dispatch,
            apply_topk_in_fc1=apply_topk_in_fc1,
            gate_up_clamp=gate_up_clamp,
            epi_flag_batch=epi_flag_batch,
            green_trace_role=green_trace_role,
            jit_config=jit_config,
        )

        self.split_role = split_role
        self.comm_backend = comm_backend
        self.producer_sm_count = producer_sm_count
        self.k2_tail_reclaim = k2_tail_reclaim
        self.skip_global_tail = skip_global_tail
        self.k2_ready_queue = (
            split_role in ("k1", "k2")
            and k2_ready_queue
        )
        self.k1_ready_queue_workspace = (
            split_role in ("k1", "k2")
            and k1_ready_queue_workspace
        )
        self.k1_ready_queue = (
            split_role == "k1" and self.k1_ready_queue_workspace
        )
        self.k2_ready_queue_bundle = k2_ready_queue_bundle
        if self.k2_ready_queue_bundle <= 0:
            raise ValueError("k2_ready_queue_bundle must be positive.")
        self.fc1_ready_tile_tokens = (
            mma_tiler_mnk[1]
            if fc1_ready_tile_tokens is None
            else fc1_ready_tile_tokens
        )
        self.fc1_producer_tile_tokens = (
            mma_tiler_mnk[1]
            if fc1_producer_tile_tokens is None
            else fc1_producer_tile_tokens
        )
        self.k2_token_tile_tokens = (
            mma_tiler_mnk[1]
            if k2_token_tile_tokens is None
            else k2_token_tile_tokens
        )
        self.k2_ready_tile_tokens = max(
            self.fc1_ready_tile_tokens,
            self.k2_token_tile_tokens,
        )
        if self.fc1_ready_tile_tokens % token_padding_block != 0:
            raise ValueError(
                "fc1_ready_tile_tokens must be divisible by "
                f"token_padding_block, got {self.fc1_ready_tile_tokens} "
                f"and {token_padding_block}."
            )
        if (
            self.fc1_producer_tile_tokens <= 0
            or self.k2_ready_tile_tokens
            % self.fc1_producer_tile_tokens
            != 0
        ):
            raise ValueError(
                "fc1_producer_tile_tokens must be a positive divisor of "
                "k2_ready_tile_tokens, got "
                f"{self.fc1_producer_tile_tokens} and "
                f"{self.k2_ready_tile_tokens}."
            )
        if (
            self.k2_token_tile_tokens <= 0
            or self.k2_ready_tile_tokens % self.k2_token_tile_tokens != 0
        ):
            raise ValueError(
                "k2_token_tile_tokens must be a positive divisor of "
                "k2_ready_tile_tokens, got "
                f"{self.k2_token_tile_tokens} and "
                f"{self.k2_ready_tile_tokens}."
            )
        self.fc1_producer_tiles_per_ready_block = (
            self.k2_ready_tile_tokens
            // self.fc1_producer_tile_tokens
        )
        self.compact_k2 = compact_k2
        default_natural_regs = (
            split_role == "k2"
            and compact_k2
            and mma_tiler_mnk[1] == 32
        )
        use_natural_regs = (
            k2_natural_regs if split_role == "k2" else default_natural_regs
        )
        self.use_warpgroup_reg_realloc = not (
            split_role == "k2" and compact_k2 and use_natural_regs
        )
        if split_role == "k2" and compact_k2:
            if k2_min_blocks_per_sm <= 0:
                raise ValueError("k2_min_blocks_per_sm must be positive")
            self.occupancy = k2_min_blocks_per_sm
        self.scheduler_phase_mode = (
            "fc1" if split_role == "k1"
            else "fc2" if split_role == "k2"
            else "fused"
        )
        self.scheduler_generation_mode = self.scheduler_phase_mode
        self.fc2_ready_wait_enabled = not (
            split_role == "k2" and producer_sm_count == 0
        )
        # K2 consumes expert extents published by K1, so its static scheduler
        # must still defer the first decode even though K2 has no dispatch
        # warps of its own.
        self.scheduler_needs_token_layout = True
        # Keep the shared FC12 body and its original register allocation in
        # both phases, while removing the four dispatch warps from compact K2.
        self.enable_token_comm = True
        self.has_dispatch_warps = split_role != "k2" or not compact_k2
        self.dispatch_warp_id = (
            self.sm120_dispatch_warp_id if self.has_dispatch_warps else None
        )
        # Standalone token-back: a dedicated 4-warp group (12-15) doing
        # token_back_by_push concurrently with dispatch_pull, selected by the
        # user-facing token_back_mode knob ("standalone_warps").
        self.token_back_mode = token_back_mode
        self.token_back_standalone = (
            split_role == "fused" and token_back_mode == "standalone_warps"
        )
        self.token_back_warp_id = (12, 13, 14, 15) if self.token_back_standalone else None
        num_token_back_warps = (
            len(self.token_back_warp_id) if self.token_back_standalone else 0
        )
        base_warps = (
            len(self.compute_warp_id)
            + 1  # tma_a
            + 1  # tma_b
            + 1  # scheduler
            + 1  # aux / reserved warp
        )
        self.threads_per_cta = 32 * (
            base_warps
            + (len(self.dispatch_warp_id) if self.dispatch_warp_id else 0)
            + num_token_back_warps
        )

        # Independent MegaMoE-specific constants.
        self.world_size = world_size
        self.local_rank = local_rank
        self.num_topk = num_topk
        self.max_tokens_per_rank = max_tokens_per_rank
        self.hidden = hidden

        # static_expert_shape = (num_experts_per_rank, intermediate_gateup, hidden).
        self.num_experts_per_rank = static_expert_shape[0]
        self.intermediate_gateup = static_expert_shape[1]
        self.intermediate_downproj = self.intermediate_gateup // 2

        # MXFP8 E4M3/E5M2: 1 byte/element.
        self.hidden_bytes = self.hidden
        # Dispatch pulls SF in uint32 units; host activation_sf rows must pad
        # to this ceiling with zero-filled bytes.
        sf_atom_k_elements = 4 * self.sf_vec_size
        self.sf_uint32_per_token = (
            (self.hidden + sf_atom_k_elements - 1) // sf_atom_k_elements
        )
        # Cross-rank totals: per-rank count * world_size.
        self.num_total_experts = world_size * self.num_experts_per_rank

        # K2 may compute N64 while K1 publishes N128 ready blocks.
        self.cluster_tile_tokens = self.fc1_ready_tile_tokens


        # One dispatch task tile must map to contiguous pool blocks.
        if self.cluster_tile_tokens % self.token_padding_block != 0:
            raise ValueError(
                f"C3 violated: cluster_tile_tokens "
                f"({self.cluster_tile_tokens}) must be a multiple of "
                f"token_padding_block ({self.token_padding_block}); "
                f"otherwise pool row offsets and release counter slots "
                f"will not align."
            )

        # Cache region sizing inputs used by workspace layout and __call__.
        (
            self.pool_token_capacity,
            self.pool_sf_capacity,
            self.pool_task_tile_capacity,
        ) = self._pool_shapes()
        self.ibgda_k2_direct_staging = (
            self.comm_backend == "nvshmem_ibgda"
        )
        fc1_done_slots = (
            (
                self.pool_token_capacity
                + self.fc1_ready_tile_tokens
                - 1
            )
            // self.fc1_ready_tile_tokens
            + self.num_experts_per_rank
        )
        fc2_k_tiles = (
            self.intermediate_downproj + self.mma_tiler_mnk[2] - 1
        ) // self.mma_tiler_mnk[2]
        fc2_ready_bundles = (
            fc2_k_tiles + self.fc2_ready_bundle_k_tiles - 1
        ) // self.fc2_ready_bundle_k_tiles
        fc2_hidden_tiles = (
            self.hidden + self.mma_tiler_mnk[0] - 1
        ) // self.mma_tiler_mnk[0]
        queue_bundles_per_token_block = (
            fc2_hidden_tiles + self.k2_ready_queue_bundle - 1
        ) // self.k2_ready_queue_bundle
        k2_token_tiles_per_ready_block = (
            self.k2_ready_tile_tokens // self.k2_token_tile_tokens
        )
        self.k2_ready_queue_capacity = (
            fc1_done_slots
            * queue_bundles_per_token_block
            * k2_token_tiles_per_ready_block
        ) + 256
        self.k1_ready_queue_m_tiles = (
            self.intermediate_gateup + self.mma_tiler_mnk[0] - 1
        ) // self.mma_tiler_mnk[0]
        self.k1_ready_queue_capacity = (
            (
                self.pool_task_tile_capacity
                * self.k1_ready_queue_m_tiles
            )
            + 256
            if self.k1_ready_queue_workspace
            else 0
        )
        self.k1_ready_queue_base = self.k2_ready_queue_capacity
        # Cohabit warps in this CTA outside the dispatch group:
        # 4 compute/MMA warps + TMA A + TMA B + scheduler + reserved warp 7.
        if split_role == "k2" and compact_k2:
            # K2 has no dispatch group. Reuse compute warps 0-3 solely for
            # the kernel-tail rank release/reset after their FC2 work; the
            # remaining producer/scheduler/aux warps are the four cohabitants.
            token_comm_dispatch_warp_start = 0
            num_other_warps = (
                len(self.compute_warp_id) - 4 + 4
            )
        else:
            token_comm_dispatch_warp_start = self.dispatch_warp_id[0]
            num_other_warps = 4 + 1 + 1 + 1 + 1
        # fc2 epi publishes once per CTA per work tile; edge hidden tiles
        # still publish (no in-bound gating), so ceil_div on the hidden axis.
        cluster_fc2_tile_hidden = (
            self.mma_tiler[0] * self.cluster_shape_mn[0]
            // (2 if self.use_2cta_instrs else 1)
        )
        fc2_publishes_per_token_cluster_tile = (
            (self.hidden + cluster_fc2_tile_hidden - 1)
            // cluster_fc2_tile_hidden
        ) * self.cluster_shape_mn[0]

        # Homomorphic to the fc1+fc2 scheduler: atomic_counter token-back only
        # nets a win with enough tokens, the same condition that selects the
        # atomic_counter fc1+fc2 scheduler.  Static when token-back is off.
        self.token_back_schedule_mode = (
            self.load_balance_mode if self.token_back_by_dispatch else "static"
        )

        if dispatch_pull_mode == "auto":
            # Per-tile ready counters already let compute overlap dispatch.
            # Token-strided pull avoids the extra warp rendezvous and release
            # reductions of the cooperative publication experiment.
            dispatch_pull_mode = "token_strided"
        if dispatch_compute_overlap is None:
            raise ValueError(
                "dispatch_compute_overlap must be resolved by heuristic.py"
            )
        assert dispatch_compute_overlap is not None

        self.token_comm = Sm120SysmemTokenInPullTokenBackPush(
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_topk=self.num_topk,
            num_experts_per_rank=self.num_experts_per_rank,
            num_total_experts=self.num_total_experts,
            hidden=self.hidden,
            fc1_token_dtype=cutlass.Float8E4M3FN,
            fc2_output_dtype=(
                self.fc2_output_dtype if self.token_back_by_dispatch else None
            ),
            fc2_publishes_per_token_cluster_tile=fc2_publishes_per_token_cluster_tile,
            token_back_reduce_topk=(
                self.token_back_by_dispatch and self.in_kernel_fc2_reduce
            ),
            token_back_standalone=self.token_back_standalone,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            cluster_tile_tokens=self.cluster_tile_tokens,
            cluster_shape_mn=self.cluster_shape_mn,
            dispatch_warp_start=token_comm_dispatch_warp_start,
            num_other_warps=num_other_warps,
            flag_batch=flag_batch,
            is_swap_ab=True,
            token_back_schedule_mode=self.token_back_schedule_mode,
            dispatch_pull_mode=dispatch_pull_mode,
            dispatch_warps_per_tile=dispatch_warps_per_tile,
            dispatch_compute_overlap=dispatch_compute_overlap,
            streaming_fc12=self.streaming_fc12,
            k1_ready_queue=self.k1_ready_queue,
            k1_ready_queue_m_tiles=self.k1_ready_queue_m_tiles,
            k1_ready_queue_base=self.k1_ready_queue_base,
            k1_ready_queue_m_rotation=k1_ready_queue_m_rotation,
            max_tokens_per_rank=self.max_tokens_per_rank,
            ibgda_dispatch_chunk_tokens=ibgda_dispatch_chunk_tokens,
            comm_backend=self.comm_backend,
        )

        # Reuse the existing IBGDA local control/SF region for the compact
        # pool-row -> per-peer TX-staging-row map.  K2 reads this map in its
        # epilogue and stores directly in combine send order, so the sender
        # no longer gathers full hidden rows into a cyclic TX slot.
        self.ibgda_direct_stage_map_offset_i32 = (
            self.token_comm.ibgda_dispatch_ready_words
            + self.token_comm.ibgda_owner_plan_ready_words
            + self.pool_token_capacity * self.sf_uint32_per_token
        )
        self.token_comm.ibgda_direct_stage_map_offset_i32 = (
            self.ibgda_direct_stage_map_offset_i32
        )

        # Region layout (same call drives both get_workspace_sizes() and
        # the __call__ partition).
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

    # =========================================================================
    # SMEM budget hook (base override)
    # =========================================================================

    def _dispatch_smem_bytes(self) -> int:
        """SMEM bytes for dispatch pull mbarriers and aliased count/token scratch."""
        pull_mbar_bytes = _DispatchWarpCount * 8
        pull_buffer_bytes = _DispatchWarpCount * self.hidden_bytes
        total = (
            _round_up(pull_mbar_bytes, 16)
            + _round_up(pull_buffer_bytes, 128)
            + 128
        )
        if self.token_back_standalone:
            total += (
                _round_up(_DispatchWarpCount * 8, 16)
                + _round_up(
                    _DispatchWarpCount * self.token_comm.tb_chunk_bytes, 128
                )
            )
        return total

    def _smem_misc_budget_bytes(self) -> int:
        """Base misc reservation plus dispatch-warp SMEM."""
        dispatch_bytes = 0
        if self.has_dispatch_warps:
            dispatch_bytes = self._dispatch_smem_bytes()
        return super()._smem_misc_budget_bytes() + dispatch_bytes

    # =========================================================================
    # Pool sizing (first-principles)
    # =========================================================================

    def _pool_shapes(self) -> Tuple[int, int, int]:
        """Worst-case pool sizes.

        ``pool_token_capacity``: every received token from any peer can
        replicate to ``min(num_topk, num_experts_per_rank)`` local
        experts; worst case is ``world_size * max_tokens_per_rank``
        tokens received, each replicated up to that bound.  Each of
        the ``num_experts_per_rank`` experts wastes up to
        ``token_padding_block - 1`` rows at its tail; round the whole
        sum up to the pool-layout granularity ``token_padding_block``.

        ``pool_sf_capacity``: same number of expert blocks as the data
        pool, each padded to ``sf_padding_block`` rows (UTCCP 4x32
        swizzle that the SF TMA load expects).

        ``pool_task_tile_capacity``: ``ceil(pool_token_capacity,
        cluster_tile_tokens)``.  C3 makes ``cluster_tile_tokens`` a
        multiple of ``token_padding_block`` so this stays exact.
        """
        world_size = self.world_size
        max_tokens_per_rank = self.max_tokens_per_rank
        num_topk = self.num_topk
        num_experts_per_rank = self.num_experts_per_rank
        token_padding_block = self.token_padding_block
        sf_padding_block = self.sf_padding_block
        cluster_tile_tokens = self.cluster_tile_tokens

        max_recv = world_size * max_tokens_per_rank
        max_per_token = min(num_topk, num_experts_per_rank)
        raw = (
            max_recv * max_per_token
            + num_experts_per_rank * (token_padding_block - 1)
        )
        pool_token_capacity = _round_up(raw, token_padding_block)
        pool_sf_capacity = (
            (pool_token_capacity // token_padding_block) * sf_padding_block
        )
        # Upper bound for sum_e ceil(valid_e, cluster_tile_tokens).  The
        # per-expert slack covers each expert's final partial task tile.
        pool_task_tile_capacity = (
            (pool_token_capacity + cluster_tile_tokens - 1) // cluster_tile_tokens
            + num_experts_per_rank
        )
        return (
            pool_token_capacity,
            pool_sf_capacity,
            pool_task_tile_capacity,
        )

    # =========================================================================
    # Region tables
    # =========================================================================

    def _build_local_region_specs(self) -> List[_RegionSpec]:
        """Local-only regions (no peer access via ``peer_rank_ptr_mapper.map`` in
        ``src/dispatch_kernel.py``).
        """
        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        pool_task_tile_capacity = self.pool_task_tile_capacity
        num_experts_per_rank = self.num_experts_per_rank
        num_total_experts = self.num_total_experts
        hidden_bytes = self.hidden_bytes
        sf_uint32_per_token = self.sf_uint32_per_token
        intermediate_downproj = self.intermediate_downproj
        ready_block_n = self.fc1_ready_tile_tokens
        sf_vec_size = self.sf_vec_size
        sf_padding_block = self.sf_padding_block

        # fc1_output_sf / fc1_done_counter sizing mirrors base
        # ``get_workspace_size_in_bytes`` (kernel_fc12.py ~lines 525-543).
        sf_total_rows_upper = (
            pool_token_capacity + num_experts_per_rank * sf_padding_block
        )
        sf_block_cols = (
            (((intermediate_downproj // sf_vec_size) + 3) // 4) * 4
        )
        fc1_done_slots = (
            (pool_token_capacity + ready_block_n - 1) // ready_block_n
            + num_experts_per_rank
        )
        fc2_k_tiles = (
            intermediate_downproj + self.mma_tiler_mnk[2] - 1
        ) // self.mma_tiler_mnk[2]
        fc2_ready_bundles = (
            fc2_k_tiles + self.fc2_ready_bundle_k_tiles - 1
        ) // self.fc2_ready_bundle_k_tiles
        fc2_hidden_tiles = (
            self.hidden + self.mma_tiler_mnk[0] - 1
        ) // self.mma_tiler_mnk[0]
        queue_bundles_per_token_block = (
            fc2_hidden_tiles + self.k2_ready_queue_bundle - 1
        ) // self.k2_ready_queue_bundle
        k2_token_tiles_per_ready_block = (
            self.fc1_ready_tile_tokens // self.k2_token_tile_tokens
        )
        ready_queue_capacity = self.k2_ready_queue_capacity
        combined_ready_queue_capacity = (
            ready_queue_capacity
            + self.k1_ready_queue_capacity
        )

        specs: List[_RegionSpec] = [
            # Keep the dispatch release atomics below the 4-GiB workspace
            # boundary.  Large DSV4 pools place this counter after >4 GiB of
            # token/SF storage if it follows the data buffers; on SM120 that
            # address is corrupted by the 32-bit addressing path used by the
            # fine-grained arrival atomics, so a later replay never publishes
            # all K1 work.  Region lookup is name-based, so moving this small
            # control allocation does not change the kernel interface.
            _RegionSpec(
                "l1_arrival_count",
                cutlass.Int32,
                (pool_task_tile_capacity,),
                16,
            ),
            # L1 input pool (dispatch_pull writes -> fc1 reads).  Stored
            # as Uint8 bytes; the MXFP8 view at the same offset is
            # built inside ``__call__``.
            _RegionSpec(
                "l1_token_buffer",
                cutlass.Uint8,
                (pool_token_capacity, hidden_bytes),
                128,
            ),
            # Stored as Int32 (dispatch_pull's 32 b read/write); the FP8
            # view for activation_sf is built at the same offset.
            # 1D Int32 atom-flat buffer.  Total Int32 count = pool_sf_capacity
            # (M-axis token positions) * sf_uint32_per_token (K-atom count),
            # laid out atom-by-atom per cute SFA layout.  dispatch writes
            # individual Int32 slots via the linear offset returned by
            # ``src/sf_swizzle.py:sf_atom_int32_offset``; the mma side
            # re-views this same byte buffer through ``tile_atom_to_shape_SF``
            # which reads back the atom-swizzled bytes.
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
                "nvlink_barrier_counter",
                cutlass.Int32,
                (1,),
                16,
            ),
            _RegionSpec(
                "fc1_output",
                cutlass.Float8E4M3FN,
                (pool_token_capacity, intermediate_downproj),
                128,
            ),
            _RegionSpec(
                "fc1_output_sf",
                cutlass.Float8E8M0FNU,
                (sf_total_rows_upper, sf_block_cols),
                128,
            ),
            _RegionSpec(
                "fc1_done_counter",
                cutlass.Int32,
                (fc1_done_slots * fc2_ready_bundles,),
                16,
            ),
        ]
        if self.comm_backend == "nvshmem_ibgda":
            # NVSHMEM get_warp writes a contiguous local destination.  The
            # FC1 scale-factor pool is atom-swizzled, so use a compact plain
            # staging row and let dispatch lanes scatter it into the existing
            # swizzled layout after each blocking get.
            specs.append(
                _RegionSpec(
                    "ibgda_sf_staging",
                    cutlass.Int32,
                    (
                        self.token_comm.ibgda_dispatch_ready_words
                        + self.token_comm.ibgda_owner_plan_ready_words
                        + pool_token_capacity * sf_uint32_per_token
                        + pool_token_capacity,
                    ),
                    16,
                )
            )
        if self.token_back_by_dispatch:
            specs.append(
                _RegionSpec(
                    "fc2_output_workspace",
                    self.fc2_output_dtype,
                    (pool_token_capacity, 1, self.hidden),
                    128,
                )
            )
            specs.append(
                _RegionSpec(
                    "fc2_done_counter",
                    cutlass.Int32,
                    (
                        fc1_done_slots
                        if self.streaming_fc12
                        else num_experts_per_rank,
                    ),
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

        if self.load_balance_mode == "atomic_counter" or self.k2_tail_reclaim:
            specs.append(
                _RegionSpec(
                    "load_balance_counter",
                    cutlass.Int32,
                    (1,),
                    16,
                )
            )
        if self.k2_ready_queue:
            specs.extend(
                [
                    _RegionSpec(
                        "k2_ready_queue_desc",
                        cutlass.Int32,
                        (combined_ready_queue_capacity * 7,),
                        16,
                    ),
                    _RegionSpec(
                        "k2_ready_queue_ready",
                        cutlass.Int32,
                        (combined_ready_queue_capacity,),
                        16,
                    ),
                    _RegionSpec(
                        "k2_ready_queue_state",
                        cutlass.Int32,
                        (7 if self.k1_ready_queue_workspace else 4,),
                        16,
                    ),
                ]
            )
        return specs

    def _build_shared_region_specs(self) -> List[_RegionSpec]:
        """Shared (peer-mapped) regions -- every entry is reached from
        some ``peer_rank_ptr_mapper.map(local_ptr, peer_rank, byte_off)``
        call site inside ``src/dispatch_kernel.py``:

          * ``src_token_topk_idx`` -- ``_dispatch_prep`` round 3
          * ``expert_recv_count`` / ``expert_recv_count_sum``
            -- ``_dispatch_barrier`` step 2 (b64 store + sys-atomic-add)
          * ``nvlink_barrier_signal``
            -- ``_nvlink_barrier_3stage`` stage B (two reusable phase slots)
        """
        world_size = self.world_size
        num_topk = self.num_topk
        max_tokens_per_rank = self.max_tokens_per_rank
        num_experts_per_rank = self.num_experts_per_rank

        # ``MAX_SLOT`` in ``_dispatch_prep`` round 3: every (token, topk)
        # edge any peer might publish for this rank's local experts.
        max_slot = max_tokens_per_rank * num_topk

        return [
            _RegionSpec(
                "src_token_topk_idx",
                cutlass.Int32,
                (num_experts_per_rank, world_size, max_slot),
                16,
            ),
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
                "nvlink_barrier_signal",
                cutlass.Int32,
                (_NvlinkSlotCount * world_size,),
                16,
            ),
        ]

    # =========================================================================
    # Public: workspace size query
    # =========================================================================

    def get_workspace_sizes(self) -> Tuple[int, int]:
        """Return ``(local_ws_bytes, shared_ws_bytes)`` -- the byte
        budgets for the two opaque workspaces the host must allocate.
        Both totals are invariant across launches; per-launch ``T``
        may be <= ``max_tokens_per_rank``.
        """
        return self._local_total, self._shared_total

    # =========================================================================
    # Workspace partition helpers
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
        # Large MegaMoE problems can place later workspace regions above the
        # 2 GiB / 4 GiB boundary.  Keep the base adjustment in 64-bit pointer
        # arithmetic so region starts such as fc1_output_sf / counters do not
        # wrap before the typed view is built.
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
        """Partition a region of the local workspace.  With no overrides,
        uses the region's declared dtype + shape + row-major stride;
        overrides let dual-view callers build alternate-dtype views at
        the same byte offset.
        """
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
                # Derive row-major from the (possibly overridden) shape.
                out: List[int] = [1]
                for d in reversed(list(sh)[1:]):
                    out.append(out[-1] * d)
                out.reverse()
                st = tuple(out)
        return self._make_typed_view(
            byte_workspace, offsets[spec.name], dt, sh, st, spec.align,
        )

    # =========================================================================
    # __call__
    # =========================================================================

    @cute.jit
    def __call__(
        self,
        # User-domain inputs (peer-mapped on the symmetric heap).
        activation: cute.Tensor,           # (T, hidden) MXFP8 E4M3
        activation_sf: cute.Tensor,        # (T, round_up(hidden, sf_atom_block_k)) FP8
        topk_idx: cute.Tensor,             # (T, num_topk) Int64
        topk_weights: cute.Tensor,         # (T, num_topk) Float32
        # Per-rank model weights (local-only; not in workspace).
        fc1_weight: cute.Tensor,
        fc1_weight_sf: cute.Tensor,
        fc2_weight: cute.Tensor,
        fc2_weight_sf: cute.Tensor,
        fc1_alpha: cute.Tensor,
        fc2_alpha: cute.Tensor,
        fc1_norm_const: cute.Tensor,
        # Combine destination (peer write target under S3; local fc2
        # output region under S2 -- same memory, same caller).
        combine_output: cute.Tensor,       # (T, num_topk, hidden) BF16
        combine_ready_flags: Optional[cute.Tensor],
        fc2_block_done_counter: Optional[cute.Tensor],
        # Opaque workspaces.
        local_workspace: cute.Tensor,      # (local_ws_bytes,) Uint8
        shared_workspace: cute.Tensor,     # (shared_ws_bytes,) Uint8
        # Runtime host payload; packed into ``SymBuffer{world_size}``
        # before entering the device kernel.
        peer_rank_ptr_mapper_host,
        # Codegen / runtime.
        max_active_clusters: cutlass.Constexpr,
        stream,
        green_trace: Optional[cute.Tensor] = None,
    ) -> None:
        """Launch the MegaMoE-complete fused kernel.

        Pointer-mapping contract:
          * ``activation`` / ``activation_sf`` / ``topk_weights`` MUST
            point into memory reachable via ``peer_rank_ptr_mapper.map(...)``
            (typically NVSHMEM symmetric heap).  Single-rank degenerate
            runs (``peer_rank_ptr_mapper.offsets[local_rank] == 0`` by NVSHMEM
            convention) are allowed.
          * ``topk_idx`` is read on the local rank only; placement is
            unconstrained (cuda local or sym heap).
          * ``fc1_weight`` / ``fc1_weight_sf`` / ``fc2_weight`` /
            ``fc2_weight_sf`` are local-only.
          * ``combine_output`` is the per-rank S3 combine STG target;
            under S2 it acts as the rank's local BF16 fc2 output.
            Placement: sym heap (peer write target) or local in the
            single-rank degenerate case.

        Workspace zero-init contract: caller is currently expected to
        zero ``shared_workspace`` before launch (the dispatch
        primitives' counters / signals rely on a clean state).  This
        contract may be tightened later to have the kernel take
        ownership of the reset.
        """
        # ``max_active_clusters`` and ``cluster_size`` are both Python ints
        # at trace time, so the product folds to a Python int that flows
        # cleanly to every dispatch primitive's ``num_sms: Constexpr[int]``
        # slot.
        cluster_size = self.cluster_shape_mn[0] * self.cluster_shape_mn[1]
        sm_count = max_active_clusters * cluster_size
        peer_rank_ptr_mapper = peer_rank_ptr_mapper_host.make_device_obj()

        pool_token_capacity = self.pool_token_capacity
        pool_sf_capacity = self.pool_sf_capacity
        hidden = self.hidden
        sf_per_token_fp8 = self.sf_uint32_per_token * 4  # 4 FP8 SFs per Int32

        # L1 token buffer: Uint8 view (dispatch_pull byte arith) + MXFP8
        # view (fc1 GEMM mainloop).  Same byte offset.
        l1_token_buffer_u8 = self._view_local(
            local_workspace, "l1_token_buffer",
        )
        l1_token_buffer_mxfp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_token_buffer"],
            cutlass.Float8E4M3FN,
            (pool_token_capacity, hidden),
            (hidden, 1),
            self._local_region_by_name["l1_token_buffer"].align,
        )

        # L1 SF buffer: Int32 view (dispatch_pull's [j, t] 2D indexing) +
        # FP8 view (base.activation_sf re-views via tile_atom_to_shape_SF
        # off the iterator, so the stride here is informational only).
        l1_sf_buffer_i32 = self._view_local(
            local_workspace, "l1_sf_buffer",
        )
        l1_sf_buffer_fp8 = self._make_typed_view(
            local_workspace,
            self._local_offsets["l1_sf_buffer"],
            cutlass.Float8E8M0FNU,
            (pool_sf_capacity, sf_per_token_fp8),
            (sf_per_token_fp8, 1),
            self._local_region_by_name["l1_sf_buffer"].align,
        )

        l1_topk_weights_buffer = self._view_local(
            local_workspace, "l1_topk_weights_buffer",
        )
        l1_arrival_count = self._view_local(
            local_workspace, "l1_arrival_count",
        )
        # token_src_metadata is one packed Uint64 per pool token:
        # low32=src_token, high32=(src_rank<<16)|src_topk.
        token_src_metadata = self._view_local(
            local_workspace, "token_src_metadata",
        )
        expert_send_count = self._view_local(
            local_workspace, "expert_send_count",
        )
        grid_sync_counter_all = self._view_local(
            local_workspace, "grid_sync_counter",
        )
        # K1 and K2 are independent persistent grids in split mode.  Sharing
        # one sense-reversing counter lets launch skew or an interrupted run
        # advance the other grid's phase and can strand every polling CTA.
        grid_sync_slot = 1 if self.split_role == "k2" else 0
        grid_sync_counter = self._make_typed_view(
            local_workspace,
            self._local_offsets["grid_sync_counter"] + grid_sync_slot * 4,
            cutlass.Int32,
            (1,),
            (1,),
            4,
        )
        nvlink_barrier_counter = self._view_local(
            local_workspace, "nvlink_barrier_counter",
        )
        fc1_output = self._view_local(local_workspace, "fc1_output")
        fc1_output_sf = self._view_local(local_workspace, "fc1_output_sf")
        fc1_done_counter = self._view_local(
            local_workspace, "fc1_done_counter",
        )
        if cutlass.const_expr(self.k2_ready_queue):
            k2_ready_queue_desc = self._view_local(
                local_workspace, "k2_ready_queue_desc",
            )
            k2_ready_queue_ready = self._view_local(
                local_workspace, "k2_ready_queue_ready",
            )
            k2_ready_queue_state = self._view_local(
                local_workspace, "k2_ready_queue_state",
            )
        else:
            k2_ready_queue_desc = None
            k2_ready_queue_ready = None
            k2_ready_queue_state = None
        load_balance_counter: Optional[cute.Tensor] = None
        if cutlass.const_expr(self.load_balance_mode == "atomic_counter"):
            load_balance_counter = self._view_local(
                local_workspace, "load_balance_counter",
            )

        # Shared regions.
        src_token_topk_idx = self._view_shared(
            shared_workspace, "src_token_topk_idx",
        )
        expert_recv_count = self._view_shared(
            shared_workspace, "expert_recv_count",
        )
        expert_recv_count_sum = self._view_shared(
            shared_workspace, "expert_recv_count_sum",
        )
        nvlink_barrier_signal = self._view_shared(
            shared_workspace, "nvlink_barrier_signal",
        )

        # i32 stride=(2,) view onto the i64 ``expert_recv_count_sum``
        # buffer -- low32 bits hold per-expert total token count after
        # _dispatch_barrier; zero-copy alias for sizes-mode scheduling.
        expert_token_sizes = self._view_shared(
            shared_workspace,
            "expert_recv_count_sum",
            cute_dtype=cutlass.Int32,
            shape=(self.num_experts_per_rank,),
            stride=(2,),
        )

        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_workspace_native = self._view_local(
                local_workspace, "fc2_output_workspace",
            )
            fc2_output_workspace_u8 = self._make_typed_view(
                local_workspace,
                self._local_offsets["fc2_output_workspace"],
                cutlass.Uint8,
                (pool_token_capacity * self.hidden * (
                    int(self.fc2_output_dtype.width) // 8
                ),),
                None,
                self._local_region_by_name["fc2_output_workspace"].align,
            )
            fc2_done_counter = self._view_local(
                local_workspace, "fc2_done_counter",
            )
            combine_output_u8 = cute.recast_tensor(
                combine_output, cutlass.Uint8,
            )
        else:
            fc2_output_workspace_native = None
            fc2_output_workspace_u8 = None
            fc2_done_counter = None
            combine_output_u8 = combine_output

        if cutlass.const_expr(self.comm_backend == "nvshmem_ibgda"):
            ibgda_sf_staging = self._view_local(
                local_workspace, "ibgda_sf_staging",
            )
        else:
            ibgda_sf_staging = None

        if cutlass.const_expr(self.token_back_schedule_mode == "atomic_counter"):
            token_back_schedule_counter = self._view_local(
                local_workspace, "token_back_schedule_counter",
            ).iterator
        else:
            token_back_schedule_counter = None

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
            combine_sf=ibgda_sf_staging,
            fc2_output_workspace=fc2_output_workspace_u8,
            fc2_done_counter=fc2_done_counter,
            token_back_schedule_counter=token_back_schedule_counter,
            nvlink_barrier_signal=nvlink_barrier_signal,
            nvlink_barrier_counter=nvlink_barrier_counter,
            grid_sync_counter=grid_sync_counter,
            local_zero_prefix=grid_sync_counter_all,
            shared_zero_prefix=nvlink_barrier_signal,
            peer_rank_ptr_mapper=peer_rank_ptr_mapper,
            world_size=self.world_size,
            local_rank=self.local_rank,
            num_total_experts=self.num_total_experts,
            num_experts_per_rank=self.num_experts_per_rank,
            num_topk=self.num_topk,
            hidden_bytes=self.hidden_bytes,
            sf_uint32_per_token=self.sf_uint32_per_token,
            token_padding_block=self.token_padding_block,
            sf_padding_block=self.sf_padding_block,
            sm_count=sm_count,
        )

        # C1 / C2 are tautological (token_padding_block == "block_m";
        # sf_padding_block == "sf_block_m") so the pool layout and the
        # sched cumulative-row offsets align by construction.
        #
        # ``combine_output`` is MoE-domain storage. Non-reduce modes use
        # ``(max_tokens_per_rank, num_topk, hidden)`` and host-reduce topk;
        # REDG modes use ``(max_tokens_per_rank, 1, hidden)`` and reduce in
        # kernel.  The epilogue return tile maps local pool rows back to the
        # source rank's token row through ``token_comm_args``.
        if cutlass.const_expr(self.token_back_by_dispatch):
            fc2_output_target = fc2_output_workspace_native
        else:
            fc2_output_target = combine_output

        Sm120SwapABSwigluMxfp8Fc12Kernel.__call__(self,
                activation=l1_token_buffer_mxfp8,
                fc1_weight=fc1_weight,
                activation_sf=l1_sf_buffer_fp8,
                fc1_weight_sf=fc1_weight_sf,
                fc1_output=fc1_output,
                fc1_output_sf=fc1_output_sf,
                fc2_weight=fc2_weight,
                fc2_weight_sf=fc2_weight_sf,
                fc2_output=fc2_output_target,
                topk_scores=l1_topk_weights_buffer,
                fc1_done_counter=fc1_done_counter,
                combine_ready_flags=combine_ready_flags,
                fc2_block_done_counter=fc2_block_done_counter,
                fc1_alpha=fc1_alpha,
                fc2_alpha=fc2_alpha,
                fc1_norm_const=fc1_norm_const,
                offs=None,
                max_active_clusters=max_active_clusters,
                stream=stream,
                load_balance_counter=load_balance_counter,
                expert_token_sizes=expert_token_sizes,
                token_comm_args=token_comm_args,
                green_trace=green_trace,
                k2_ready_queue_desc=k2_ready_queue_desc,
                k2_ready_queue_ready=k2_ready_queue_ready,
                k2_ready_queue_state=k2_ready_queue_state,
            )

    # =========================================================================
    # TokenComm delegation surface consumed by the fc1/fc2 base kernel
    # =========================================================================

    def token_comm_extra_smem_storage_class(self) -> Optional[type]:
        if not self.has_dispatch_warps:
            return None
        return self.token_comm.extra_smem_storage_class()

    def token_comm_hook_fc1_ready_counter_ptr(self, token_comm_args):
        if self.split_role == "k2":
            return None
        return self.token_comm.fc1_ready_counter_ptr(token_comm_args)

    @cute.jit
    def token_comm_hook_sched_warp_pre_init_wait(self, token_comm_args):
        if cutlass.const_expr(self.split_role == "k2"):
            # Each lane covers a disjoint subset of local experts. K1 writes
            # the high 32 bits only after the cross-rank count exchange, so
            # observing every marker makes all low-32 token extents safe for
            # K2's first scheduler decode.
            lane_idx = cute.arch.lane_idx()
            expected_publishers = cutlass.Int64(
                self.world_size * self.producer_sm_count
            )
            expert_idx = lane_idx
            while expert_idx < cutlass.Int32(self.num_experts_per_rank):
                spin_wait(
                    token_comm_args.expert_recv_count_sum.iterator + expert_idx,
                    lambda packed: (packed >> cutlass.Int64(32))
                    >= expected_publishers,
                    fail_sleep_cycles=500,
                )
                expert_idx = expert_idx + cutlass.Int32(32)
            cute.arch.sync_warp()
            cute.arch.fence_acq_rel_gpu()
        else:
            self.token_comm.sched_warp_pre_init_wait(token_comm_args)

    @cute.jit
    def token_comm_hook_fc1_tma_b_predispatch_spin(
        self, token_comm_args, work_tile_info,
    ):
        if cutlass.const_expr(
            self.split_role != "k2" and not self.k1_ready_queue
        ):
            self.token_comm.fc1_tma_b_predispatch_spin(
                token_comm_args, work_tile_info,
            )

    @cute.jit
    def token_comm_hook_dispatch_warp_body(
        self,
        token_comm_args,
        token_comm_storage,
        k1_ready_queue_desc,
        k1_ready_queue_ready,
        k1_ready_queue_state,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(self.split_role != "k2"):
            self.token_comm.dispatch_warp_body(
                token_comm_args,
                token_comm_storage,
                k1_ready_queue_desc,
                k1_ready_queue_ready,
                k1_ready_queue_state,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )

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
        self.token_comm.token_back_warp_body(
            token_comm_args,
            token_comm_storage,
            warp_idx=warp_idx,
            lane_idx=lane_idx,
            tidx=tidx,
        )

    @cute.jit
    def token_comm_hook_tail_reset_shared_counters(
        self,
        token_comm_args,
        *,
        cta_linear_id,
        local_warp_idx,
        lane_idx,
    ):
        self.token_comm.tail_reset_shared_counters(
            token_comm_args,
            cta_linear_id=cta_linear_id,
            local_warp_idx=local_warp_idx,
            lane_idx=lane_idx,
        )

    @cute.jit
    def token_comm_hook_fc1_cta_complete(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(self.split_role == "k1"):
            pipeline.NamedBarrier(
                barrier_id=_KernelTailNamedBarrierId,
                num_threads=self.threads_per_cta,
            ).arrive_and_wait()

    @cute.jit
    def token_comm_hook_kernel_tail(
        self,
        token_comm_args,
        *,
        warp_idx,
        lane_idx,
        tidx,
    ):
        if cutlass.const_expr(
            self.comm_backend == "nvshmem_ibgda"
            and self.split_role == "k1"
        ):
            self.token_comm.kernel_tail_ibgda(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
        elif cutlass.const_expr(
            self.split_role == "k1"
            or (self.split_role == "k2" and self.skip_global_tail)
            or (
                self.comm_backend == "nvshmem_ibgda"
                and self.split_role == "k2"
            )
        ):
            # K1 only publishes intermediate readiness. K2 owns the global
            # completion barrier because its peer stores are the final
            # cross-rank writes consumed by K3. Tail-reclaim worker kernels
            # defer that barrier to the one-CTA finalizer graph node.
            pass
        else:
            self.token_comm.kernel_tail(
                token_comm_args,
                warp_idx=warp_idx,
                lane_idx=lane_idx,
                tidx=tidx,
            )
