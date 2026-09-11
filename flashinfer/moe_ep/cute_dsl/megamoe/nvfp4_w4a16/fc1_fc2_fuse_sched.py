# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""W4A16-owned fused FC1 + FC2 MegaMoE scheduler."""

from copy import copy
from dataclasses import dataclass
from enum import IntEnum
from typing import Any, List, Literal, Optional, Tuple

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass.cutlass_dsl import (
    Boolean,
    Int32,
    Integer,
    extract_mlir_values,
    new_from_mlir_values,
    const_expr,
    dsl_user_op,
)
from cutlass._mlir import ir

try:
    from cutlass.cute import iket  # type: ignore
except ImportError:  # pragma: no cover -- fallback for wheels without cute.iket
    from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import iket  # type: ignore[no-redef]

from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    MoEWorkTileInfo,
    MoESchedulerParamsBase,
    MoESchedulerBase,
    WorkTileState,
    _DEFAULT_SCHED_EXT,
)
from flashinfer.moe_ep.kernel_src.cutedsl_megamoe import (
    compute_expert_token_count_from_sizes,
    mbarrier_arrive_expect_tx_on_peer,
    store_i32_to_peer_cluster_smem_async,
)


# =============================================================================
# Block phase
# =============================================================================


class BlockPhase(IntEnum):
    """Fused fc1+fc2 work-tile phase. ``None_`` reserved as sentinel for sched
    invalid tiles (alongside ``WorkTileState.DONE``)."""

    None_ = 0
    Linear1 = 1
    Linear2 = 2


# =============================================================================
# Persistent state objects
# =============================================================================


@dataclass(eq=False)
class _FusedFc12SchedState:
    """Ordered register state; cumulatives denote the current expert's start."""

    current_group_first_expert: Int32
    current_group_last_expert_exclusive: Int32
    current_phase: Int32
    current_expert_idx: Int32
    current_expert_tile_start: Int32
    current_expert_tile_end: Int32
    current_group_fc1_subphase_end: Int32
    current_group_end: Int32
    cumulative_fc1_tiles_at_group_end: Int32
    cumulative_fc2_tiles_at_group_end: Int32
    current_data_cumul: Int32
    current_token_block_cumul: Int32
    group_start_data_cumul: Int32
    group_start_token_block_cumul: Int32
    current_token_block_count: Int32
    current_this_expert_token_cnt: Int32
    current_work_linear_tile_idx: Int32

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return extract_mlir_values(tuple(vars(self).values()))

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        return type(self)(*new_from_mlir_values(tuple(vars(self).values()), values))


@dataclass(eq=False)
class _DynamicLoadBalanceState:
    """Atomic broadcast state, including the first claim cached before init."""

    counter_ptr: Any
    broadcast_ptr: Any
    is_leader_cta: Boolean
    producer_state: Any
    consumer_state: Any
    atomic_res: Int32

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return extract_mlir_values(tuple(vars(self).values()))

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        # Nested pipeline states rebuild from their own extracted value counts.
        fields = []
        idx = 0
        for field in vars(self).values():
            count = len(extract_mlir_values(field))
            fields.append(new_from_mlir_values(field, values[idx : idx + count]))
            idx += count
        return type(self)(*fields)


# =============================================================================
# Scheduler Parameters
# =============================================================================


class MoEFusedFc12SchedulerParams(MoESchedulerParamsBase):
    """Codegen-time + runtime parameters for the fused fc1+fc2 mega scheduler.

    Inherits ``expert_shape``, ``cta_tile_shape_mnk``, ``cluster_shape_mn``,
    ``scenario``, ``is_swap_ab``, ``num_sched_stages`` handling from
    ``MoESchedulerParamsBase``.  ``cta_tile_shape_mnk`` is shared by fc1 / fc2
    (v1 simplification).

    This params type currently backs the inference fc12 path.  In that path
    ``expert_shape[1]`` is ``intermediate_gateup`` (gate + up concatenated).
    Future training/non-swap MegaMoE variants should add their own params
    contract instead of overloading this one.
    """

    def __init__(
        self,
        scenario: Literal["2Dx3D"],
        expert_shape: Tuple[int | Int32, int | Int32, int | Int32],
        cta_tile_shape_mnk: Tuple[int, int, int],
        cluster_shape_mn: Tuple[int, int],
        group_hint: int,
        token_padding_block: int,
        expert_token_sizes: cute.Tensor,
        load_balance_mode: Literal["static", "atomic_counter"] = "static",
        load_balance_counter_ptr=None,
        override_num_stages: Optional[int] = None,
    ):
        """Create fused fc12 scheduler params."""
        if scenario != "2Dx3D":
            raise ValueError(f"fused fc1+fc2 only supports 2Dx3D, got {scenario!r}")
        if load_balance_mode not in ("static", "atomic_counter"):
            raise ValueError(
                f"load_balance_mode must be one of 'static' / 'atomic_counter', "
                f"got {load_balance_mode!r}"
            )
        if load_balance_mode == "atomic_counter" and load_balance_counter_ptr is None:
            raise ValueError(
                "load_balance_counter_ptr must be provided when load_balance_mode == "
                "'atomic_counter' (GMEM int32 ptr, host-allocated and zero-init per launch)"
            )
        if group_hint <= 0:
            raise ValueError(f"group_hint must be positive, got {group_hint}")
        if token_padding_block <= 0:
            raise ValueError(
                f"token_padding_block must be positive, got {token_padding_block}"
            )
        super().__init__(
            scenario=scenario,
            expert_shape=expert_shape,
            cta_tile_shape_mnk=cta_tile_shape_mnk,
            cluster_shape_mn=cluster_shape_mn,
            override_num_stages=override_num_stages,
            is_swap_ab=True,
        )
        self.group_hint = group_hint
        self.token_padding_block = token_padding_block
        self.load_balance_mode = load_balance_mode
        self.load_balance_counter_ptr = load_balance_counter_ptr
        self.expert_token_sizes = expert_token_sizes

    def get_scheduler_type(self) -> type:
        return MoEFusedFc12PersistentTileScheduler

    def get_grid_shape(self, max_active_clusters: int) -> Tuple[int, int, int]:
        return (
            self.cluster_shape_mn[1],
            self.cluster_shape_mn[0],
            max_active_clusters,
        )

    def _mlir_fields(self):
        fields = [
            name
            for name in ("expert_cnt", "intermediate", "hidden")
            if isinstance(getattr(self, name), Int32)
        ]
        if self.load_balance_mode == "atomic_counter":
            fields.append("load_balance_counter_ptr")
        fields.append("expert_token_sizes")
        return fields

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return extract_mlir_values(
            tuple(getattr(self, name) for name in self._mlir_fields())
        )

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        # Copy post-swap constants without re-entering the swapping constructor.
        result = copy(self)
        idx = 0
        for name in self._mlir_fields():
            field = getattr(self, name)
            count = len(extract_mlir_values(field))
            setattr(
                result, name, new_from_mlir_values(field, values[idx : idx + count])
            )
            idx += count
        if self.load_balance_mode == "static":
            result.load_balance_counter_ptr = None
        assert idx == len(values), "Fused fc12 scheduler parameter value count mismatch"
        return result


# =============================================================================
# Scheduler — Fused fc1 + fc2 Persistent (Device-side)
# =============================================================================


class MoEFusedFc12PersistentTileScheduler(MoESchedulerBase):
    """Mega scheduler for fused fc1+fc2 grouped GEMM under swap-AB.

    Tile space: ``(group, phase, expert, token_block, intermediate_or_hidden_block)``.
    Within a group: full fc1 sub-segment (all experts in expert order, each expert
    expanded as ``token_block`` slow / ``intermediate_block`` fast) → full fc2
    sub-segment (same expert order, each expert expanded short-side-first).
    """

    def __init__(
        self,
        params: MoEFusedFc12SchedulerParams,
        num_persistent_clusters: Int32,
        cta_id_in_cluster: cute.Coord,
        current_work: MoEWorkTileInfo,
        fused_state: _FusedFc12SchedState,
        dynamic_state: Optional[_DynamicLoadBalanceState],
        # Cached scheduler-wide derived constants (computed once in create()
        # from params.intermediate / params.hidden / params.cluster_tile_n;
        # avoid recomputing in the hot path of advance / decode).
        num_fc1_intermediate_blocks: Int32,
        num_fc2_hidden_blocks: Int32,
        ext,
        sched_pipeline,
        smem_buf_tensor,
        num_sched_stages: int,
        cluster_pipeline,
        producer_state,
    ):
        self.params = params
        self.num_persistent_clusters = num_persistent_clusters
        self.cta_id_in_cluster = cta_id_in_cluster
        self.current_work = current_work
        self._fused_state = fused_state
        self._dynamic_state = dynamic_state
        self._num_fc1_intermediate_blocks = num_fc1_intermediate_blocks
        self._num_fc2_hidden_blocks = num_fc2_hidden_blocks
        self._ext = ext
        self._pipeline = sched_pipeline
        self._smem_buf_tensor = smem_buf_tensor
        self._num_sched_stages = num_sched_stages
        self._cluster_pipeline = cluster_pipeline
        self._producer_state = producer_state
        # Codegen-time Python attribute (NOT MLIR-serialized).  Set to True
        # by ``internal_init`` to mark "scheduler state has been greedily
        # advanced one step (atomic_add cached for atomic_counter mode /
        # current_work decoded for static mode)".  ``gen_next_work``'s
        # ``cutlass.const_expr(self._first_advance_pending)`` branch reads
        # this at trace time to elide the corresponding work for the first
        # call site, then sets it back to False so the second trace site
        # (while-body) compiles the vanilla path.
        self._first_advance_pending: bool = False

    @staticmethod
    def make_storage_struct(
        params: MoEFusedFc12SchedulerParams,
        ext=_DEFAULT_SCHED_EXT,
        **kwargs,
    ) -> type:
        num_tile_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        @cute.struct
        class StaticSchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, fields_per_stage * num_tile_stages],
                16,
            ]

        @cute.struct
        class AtomicCounterSchedulerStorage:
            sched_mbar: cute.struct.MemRange[cutlass.Int64, num_tile_stages * 2]
            sched_buf: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, fields_per_stage * num_tile_stages],
                16,
            ]
            cluster_pipeline_mbar: cute.struct.MemRange[cutlass.Int64, 2]
            cluster_broadcast_slot: cute.struct.Align[
                cute.struct.MemRange[cutlass.Int32, 1],
                16,
            ]

        if params.load_balance_mode == "atomic_counter":
            return AtomicCounterSchedulerStorage
        return StaticSchedulerStorage

    @staticmethod
    @dsl_user_op
    def create(
        params: MoEFusedFc12SchedulerParams,
        block_idx: Tuple[Integer, Integer, Integer],
        grid_dim: Tuple[Integer, Integer, Integer],
        sched_storage,
        num_consumer_threads: int,
        ext=_DEFAULT_SCHED_EXT,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> "MoEFusedFc12PersistentTileScheduler":
        if num_consumer_threads <= 0:
            raise ValueError(
                f"num_consumer_threads must be positive, got {num_consumer_threads}"
            )
        num_stages = params.num_sched_stages
        fields_per_stage = ext.WorkTileInfo.TotalFields

        num_persistent_clusters = cute.size(grid_dim, loc=loc, ip=ip) // cute.size(
            params.cluster_shape_mn, loc=loc, ip=ip
        )

        bidx, bidy, bidz = block_idx

        # ``params.cluster_shape_mn`` is scheduler-internal.  Under swap-AB,
        # launch axes map to the opposite internal M/N slots.
        cta_id_in_cluster = (
            Int32(bidy % params.cluster_shape_mn[0]),
            Int32(bidx % params.cluster_shape_mn[1]),
            Int32(0),
        )

        # State machine sentinel init.  The 0 values for current_group_end /
        # current_expert_tile_end force gen_next_work's first call to enter
        # advance_group() and advance_expert_within_phase(), which then fill
        # the rest of the state from counts. current_expert_idx starts at -1
        # so the first advance increments to expert 0.
        #
        # All cumul fields (current_*_cumul / group_start_*_cumul) start at 0;
        # current_this_expert_token_cnt and current_token_block_count also start
        # at 0 so that the first advance_expert call inside the first
        # advance_group call pushes a no-op (round_up(0, ...) = 0) into the
        # cumul state before reading expert 0's valid count.
        fused_state = _FusedFc12SchedState(
            current_group_first_expert=Int32(0),
            current_group_last_expert_exclusive=Int32(0),
            current_phase=Int32(BlockPhase.Linear1),
            current_expert_idx=Int32(-1),
            current_expert_tile_start=Int32(0),
            current_expert_tile_end=Int32(0),
            current_group_fc1_subphase_end=Int32(0),
            current_group_end=Int32(0),
            cumulative_fc1_tiles_at_group_end=Int32(0),
            cumulative_fc2_tiles_at_group_end=Int32(0),
            current_data_cumul=Int32(0),
            current_token_block_cumul=Int32(0),
            group_start_data_cumul=Int32(0),
            group_start_token_block_cumul=Int32(0),
            current_token_block_count=Int32(0),
            current_this_expert_token_cnt=Int32(0),
            current_work_linear_tile_idx=Int32(bidz),
        )

        # Cached derived constants (scheduler-wide, computed once).
        #
        # ``params.intermediate`` carries ``intermediate_gateup`` semantics
        # (= ``mat_b.shape[2]``, the full gate+up-concat fc1 weight dim;
        # see ``MoESchedulerParamsBase.__init__`` docstring).  fc1 GEMM-M
        # under swap-AB IS that gateup axis, so the number of cluster work
        # tiles along intermediate per ``(expert, token_block)`` is just
        # ``ceil_div(intermediate_gateup, cluster_tile_n_post_swap)`` --
        # the same formula ``MoEStaticPersistentTileScheduler._get_cluster_
        # tile_counts`` uses for its swap-AB 2Dx3D N-axis count.  A prior
        # ``2 *`` multiplier here was a bug that doubled the per-tile-block
        # work tile count (assumed ``params.intermediate`` was the half-dim
        # ``intermediate_downproj``); removed to match the base scheduler.
        #
        # ``params.hidden`` is single-semantic (fc2 GEMM-M / fc2 output cols).
        intermediate_gateup = params.intermediate
        hidden = params.hidden
        num_fc1_intermediate_blocks = (
            intermediate_gateup + params.cluster_tile_n - 1
        ) // params.cluster_tile_n
        num_fc2_hidden_blocks = (
            hidden + params.cluster_tile_n - 1
        ) // params.cluster_tile_n

        # current_work init must use ext.WorkTileInfo to match the shape that
        # gen_next_work writes; otherwise MLIR serialization slots would differ.
        current_work = ext.WorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_tile=Int32(0),
            phase_and_peek=Int32(BlockPhase.None_),
        )

        sched_producer_group = pipeline.CooperativeGroup(pipeline.Agent.Thread, 32)
        sched_consumer_group = pipeline.CooperativeGroup(
            pipeline.Agent.Thread, num_consumer_threads
        )
        sched_pipeline = pipeline.PipelineAsync.create(
            num_stages=num_stages,
            producer_group=sched_producer_group,
            consumer_group=sched_consumer_group,
            barrier_storage=sched_storage.sched_mbar.data_ptr(),
            defer_sync=True,
        )
        smem_buf_tensor = cute.make_tensor(
            sched_storage.sched_buf.data_ptr(),
            cute.make_layout(
                (fields_per_stage, num_stages),
                stride=(1, fields_per_stage),
            ),
        )
        producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, num_stages
        )

        # Build the broadcast pipeline with the other scheduler pipelines.
        # internal_init still claims the first atomic tile before common init.
        cluster_pipeline = None
        dynamic_state: Optional[_DynamicLoadBalanceState] = None
        if const_expr(params.load_balance_mode == "atomic_counter"):
            cluster_size = params.cluster_shape_mn[0] * params.cluster_shape_mn[1]
            cluster_pipeline = pipeline.PipelineAsync.create(
                num_stages=1,
                producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 1),
                consumer_group=pipeline.CooperativeGroup(
                    pipeline.Agent.Thread, 32 * cluster_size
                ),
                barrier_storage=sched_storage.cluster_pipeline_mbar.data_ptr(),
                defer_sync=True,
            )
            is_leader_cta = (
                cta_id_in_cluster[0] + cta_id_in_cluster[1] + cta_id_in_cluster[2]
            ) == Int32(0)
            cluster_producer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Producer, 1
            )
            cluster_consumer_state = pipeline.make_pipeline_state(
                pipeline.PipelineUserType.Consumer, 1
            )
            dynamic_state = _DynamicLoadBalanceState(
                counter_ptr=params.load_balance_counter_ptr,
                broadcast_ptr=sched_storage.cluster_broadcast_slot.data_ptr(),
                is_leader_cta=is_leader_cta,
                producer_state=cluster_producer_state,
                consumer_state=cluster_consumer_state,
                atomic_res=Int32(0),
            )

        return MoEFusedFc12PersistentTileScheduler(
            params=params,
            num_persistent_clusters=num_persistent_clusters,
            cta_id_in_cluster=cta_id_in_cluster,
            current_work=current_work,
            fused_state=fused_state,
            dynamic_state=dynamic_state,
            num_fc1_intermediate_blocks=num_fc1_intermediate_blocks,
            num_fc2_hidden_blocks=num_fc2_hidden_blocks,
            ext=ext,
            sched_pipeline=sched_pipeline,
            smem_buf_tensor=smem_buf_tensor,
            num_sched_stages=num_stages,
            cluster_pipeline=cluster_pipeline,
            producer_state=producer_state,
        )

    # -------------------------------------------------------------------------
    # internal_init: first-tile pre-init before pipeline_init_arrive
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def internal_init(
        self,
        warp_idx,
        sched_warp_id: int,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Claim/decode the first work tile during kernel prologue."""
        if const_expr(self.params.load_balance_mode == "atomic_counter"):
            # Pre-claim and broadcast first dynamic tile id.
            if warp_idx == sched_warp_id:
                tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
                atomic_res = Int32(0)
                if self._dynamic_state.is_leader_cta and tidx % 32 == Int32(0):
                    atomic_res = cute.arch.atomic_add(
                        self._dynamic_state.counter_ptr,
                        Int32(1),
                        loc=loc,
                        ip=ip,
                    )
                atomic_res = cute.arch.shuffle_sync(
                    atomic_res,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )
                self._dynamic_state.atomic_res = atomic_res
                self._dynamic_state = self._dynamic_state  # DSL carry
            else:
                self._dynamic_state = self._dynamic_state  # balance scf.if yield
        else:
            # Static mode eagerly decodes the first tile.
            if warp_idx == sched_warp_id:
                cluster_linear_tile_idx = self._advance_work_linear_tile_idx_static(
                    loc=loc, ip=ip
                )
                self._gen_work_from_cluster_idx(cluster_linear_tile_idx, loc=loc, ip=ip)
                self._fused_state = self._fused_state  # DSL carry
                self.current_work = self.current_work
            else:
                self._fused_state = self._fused_state  # balance scf.if yield
                self.current_work = self.current_work

        # Codegen-time signal: gen_next_work's first trace site sees
        # this True and emits the first-tile-finalize path; second trace
        # site (while-body) sees False and emits the vanilla path.
        self._first_advance_pending = True

    # -------------------------------------------------------------------------
    # State-machine advance helpers (group → phase → expert)
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def _advance_work_linear_tile_idx_static(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Static stride mode: read current_work_linear_tile_idx, advance by
        num_persistent_clusters for next iteration, return the read value."""
        state = self._fused_state
        cluster_linear_tile_idx = state.current_work_linear_tile_idx
        state.current_work_linear_tile_idx = (
            cluster_linear_tile_idx + self.num_persistent_clusters
        )
        return cluster_linear_tile_idx

    @dsl_user_op
    @cute.jit
    def _advance_work_linear_tile_idx_dynamic(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> Int32:
        """Atomic-counter mode: returns the cluster-linear tile idx broadcast
        to all CTAs in the cluster.

        Internally const_expr-forks on ``self._first_advance_pending``
        (codegen-time Python bool):

          - True (first trace site, post-``internal_init``): use the cached
            ``self._dynamic_state.atomic_res`` ``atom.add`` result; skip
            issuing a fresh atomic.  The cached value was computed and
            shuffled across the sched warp by ``internal_init`` BEFORE
            ``pipeline_init_arrive`` so its memory round trip overlaps with
            cluster init wait.
          - False (vanilla, second-and-onward trace sites): leader CTA
            lane 0 issues a fresh ``atom.global.add.s32`` and shuffles the
            result across the sched warp.

        Cluster-internal protocol (DSMEM broadcast + cluster_pipeline mbar
        wait) is identical between the two paths.  Mirrors
        ``cute_dsl_kernel_library/dsl_kernels/moe/moe_persistent_scheduler.py``
        ``_fetch_next_cluster_idx`` (lines 838-894).
        """
        ds = self._dynamic_state
        cluster_pipeline = self._cluster_pipeline
        broadcast_tensor = cute.make_tensor(ds.broadcast_ptr, cute.make_layout((1,)))
        cluster_size = self.params.cluster_shape_mn[0] * self.params.cluster_shape_mn[1]

        # --- Producer side (leader CTA only) ---
        if ds.is_leader_cta:
            cluster_pipeline.producer_acquire(ds.producer_state)
            full_barrier_ptr = cluster_pipeline.sync_object_full.get_barrier(
                ds.producer_state.index, loc=loc, ip=ip
            )
            tidx, _, _ = cute.arch.thread_idx(loc=loc, ip=ip)
            lane_idx = tidx % Int32(32)

            if cutlass.const_expr(self._first_advance_pending):
                # First-tile path: consume the cached atomic_res that
                # internal_init shuffled across the sched warp.
                atomic_idx = ds.atomic_res
            else:
                # Vanilla path: lane 0 atom.add, shuffle to all lanes.
                atomic_idx = Int32(0)
                if lane_idx == Int32(0):
                    atomic_idx = cute.arch.atomic_add(
                        ds.counter_ptr,
                        Int32(1),
                        loc=loc,
                        ip=ip,
                    )
                atomic_idx = cute.arch.shuffle_sync(
                    atomic_idx,
                    offset=0,
                    mask=0xFFFFFFFF,
                    mask_and_clamp=31,
                )

            # DSMEM fan-out: lanes [0, cluster_size) each write to one peer
            # CTA.  Each lane targets a distinct peer (lane_idx == peer rank).
            if lane_idx < Int32(cluster_size):
                store_i32_to_peer_cluster_smem_async(
                    ds.broadcast_ptr,
                    atomic_idx,
                    full_barrier_ptr,
                    lane_idx,
                    loc=loc,
                    ip=ip,
                )
                # Set expect_tx on the peer mbarrier to match the 4-byte
                # store above; pairs with the consumer_wait below.
                mbarrier_arrive_expect_tx_on_peer(
                    full_barrier_ptr,
                    Int32(4),
                    lane_idx,
                    loc=loc,
                    ip=ip,
                )
        ds.producer_state.advance()

        # --- Consumer side (all CTAs sched warp threads) ---
        cluster_pipeline.consumer_wait(ds.consumer_state)
        cluster_idx = broadcast_tensor[0]
        cute.arch.fence_acq_rel_cta()
        cluster_pipeline.sync_object_empty.arrive(ds.consumer_state.index, Int32(0))
        ds.consumer_state.advance()

        return cluster_idx

    @dsl_user_op
    @cute.jit
    def _advance_expert_within_phase(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Advance to the next expert in the current phase."""
        state = self._fused_state
        params = self.params
        cluster_tile_m = params.cluster_tile_m

        # Push previous expert into cumul state before bumping expert_idx.
        token_padding = params.token_padding_block
        prev_valid = state.current_this_expert_token_cnt
        state.current_data_cumul = state.current_data_cumul + (
            (prev_valid + Int32(token_padding - 1)) // Int32(token_padding)
        ) * Int32(token_padding)
        state.current_token_block_cumul = (
            state.current_token_block_cumul + state.current_token_block_count
        )

        state.current_expert_idx = state.current_expert_idx + Int32(1)
        this_expert_token_cnt = compute_expert_token_count_from_sizes(
            self.params.expert_token_sizes,
            state.current_expert_idx,
            loc=loc,
            ip=ip,
        )
        state.current_this_expert_token_cnt = this_expert_token_cnt
        state.current_token_block_count = (
            this_expert_token_cnt + Int32(cluster_tile_m) - 1
        ) // Int32(cluster_tile_m)

        # --- Step 3: slide expert_tile_start / expert_tile_end.
        state.current_expert_tile_start = state.current_expert_tile_end
        # Prebind due to DSL AST.
        tiles_in_expert = Int32(0)
        if state.current_phase == Int32(BlockPhase.Linear1):
            tiles_in_expert = (
                state.current_token_block_count * self._num_fc1_intermediate_blocks
            )
        else:
            tiles_in_expert = (
                state.current_token_block_count * self._num_fc2_hidden_blocks
            )
        state.current_expert_tile_end = (
            state.current_expert_tile_start + tiles_in_expert
        )

    @dsl_user_op
    @cute.jit
    def _switch_to_fc2(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Switch current group from Linear1 to Linear2."""
        state = self._fused_state
        state.current_phase = Int32(BlockPhase.Linear2)
        state.current_expert_idx = state.current_group_first_expert - Int32(1)
        state.current_expert_tile_end = state.current_group_fc1_subphase_end
        # Zero previous-expert cache before rewinding to group start.
        state.current_this_expert_token_cnt = Int32(0)
        state.current_token_block_count = Int32(0)
        state.current_data_cumul = state.group_start_data_cumul
        state.current_token_block_cumul = state.group_start_token_block_cumul
        self._advance_expert_within_phase(loc=loc, ip=ip)

    @dsl_user_op
    @cute.jit
    def _advance_group(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Open the next group and prime its first Linear1 expert."""
        state = self._fused_state
        params = self.params
        cluster_tile_m = params.cluster_tile_m

        # Push residual experts from the just-finished group into cumul state.
        residual_expert_idx = state.current_expert_idx
        residual_group_last_expert_exclusive = state.current_group_last_expert_exclusive
        while residual_expert_idx + Int32(1) < residual_group_last_expert_exclusive:
            self._advance_expert_within_phase(loc=loc, ip=ip)
            self._fused_state = self._fused_state
            state = self._fused_state
            residual_expert_idx = state.current_expert_idx
            residual_group_last_expert_exclusive = (
                state.current_group_last_expert_exclusive
            )
        state = self._fused_state

        # Final push; cumul now reflects the next group's first expert start.
        token_padding = params.token_padding_block
        prev_valid = state.current_this_expert_token_cnt
        state.current_data_cumul = state.current_data_cumul + (
            (prev_valid + Int32(token_padding - 1)) // Int32(token_padding)
        ) * Int32(token_padding)
        state.current_token_block_cumul = (
            state.current_token_block_cumul + state.current_token_block_count
        )

        # --- Step 3: snapshot new group_start cumul checkpoint.
        state.group_start_data_cumul = state.current_data_cumul
        state.group_start_token_block_cumul = state.current_token_block_cumul

        # --- Step 4: roll group state forward.
        base_fc1 = state.cumulative_fc1_tiles_at_group_end
        base_fc2 = state.cumulative_fc2_tiles_at_group_end

        state.current_group_first_expert = state.current_group_last_expert_exclusive

        # Greedy walk: accumulate per-expert fc1+fc2 tile counts until fc1
        # cumulative crosses (base + group_hint), or experts exhausted.
        threshold = base_fc1 + Int32(params.group_hint)
        cumulative_fc1 = base_fc1
        cumulative_fc2 = base_fc2
        expert_cursor = state.current_group_first_expert

        while expert_cursor < self.expert_cnt and cumulative_fc1 < threshold:
            token_count_e = compute_expert_token_count_from_sizes(
                self.params.expert_token_sizes,
                expert_cursor,
                loc=loc,
                ip=ip,
            )
            token_block_count_e = (token_count_e + Int32(cluster_tile_m) - 1) // Int32(
                cluster_tile_m
            )
            cumulative_fc1 = (
                cumulative_fc1 + token_block_count_e * self._num_fc1_intermediate_blocks
            )
            cumulative_fc2 = (
                cumulative_fc2 + token_block_count_e * self._num_fc2_hidden_blocks
            )
            expert_cursor = expert_cursor + Int32(1)

        state.current_group_last_expert_exclusive = expert_cursor
        state.cumulative_fc1_tiles_at_group_end = cumulative_fc1
        state.cumulative_fc2_tiles_at_group_end = cumulative_fc2

        group_total_fc1_tiles = cumulative_fc1 - base_fc1
        group_total_fc2_tiles = cumulative_fc2 - base_fc2

        # Previous group's end = this group's start in tile space.
        group_start_tile = state.current_group_end
        state.current_group_fc1_subphase_end = group_start_tile + group_total_fc1_tiles
        state.current_group_end = (
            state.current_group_fc1_subphase_end + group_total_fc2_tiles
        )

        # No-op push barrier before priming the group's first expert.
        state.current_phase = Int32(BlockPhase.Linear1)
        state.current_expert_idx = state.current_group_first_expert - Int32(1)
        state.current_expert_tile_end = group_start_tile
        state.current_this_expert_token_cnt = Int32(0)
        state.current_token_block_count = Int32(0)
        self._advance_expert_within_phase(loc=loc, ip=ip)

    # -------------------------------------------------------------------------
    # Fast-path decode
    # -------------------------------------------------------------------------

    @dsl_user_op
    @cute.jit
    def _decode_inside_expert(
        self,
        cluster_linear_tile_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> MoEWorkTileInfo:
        """Decode one cluster-linear tile using the current scheduler state."""
        state = self._fused_state
        params = self.params
        cta_tile_m = params.cta_tile_shape_mnk[0]

        local_id = cluster_linear_tile_idx - state.current_expert_tile_start

        # Prebind due to DSL AST.
        cluster_token_block_idx = Int32(0)
        cluster_intermediate_or_hidden_block_idx = Int32(0)

        is_fc1 = state.current_phase == Int32(BlockPhase.Linear1)
        if is_fc1:
            num_fc1_intermediate_blocks = self._num_fc1_intermediate_blocks
            cluster_token_block_idx = local_id // num_fc1_intermediate_blocks
            cluster_intermediate_or_hidden_block_idx = (
                local_id - cluster_token_block_idx * num_fc1_intermediate_blocks
            )
        else:
            num_fc2_hidden_blocks = self._num_fc2_hidden_blocks
            # Keep token-block as the slow axis in both phases.
            cluster_token_block_idx = local_id // num_fc2_hidden_blocks
            cluster_intermediate_or_hidden_block_idx = (
                local_id - cluster_token_block_idx * num_fc2_hidden_blocks
            )

        # Cluster → CTA granularity (mirrors MoESchedulerBase._get_work_tile_for_linear_idx)
        cta_token_block_idx = (
            cluster_token_block_idx * params.cluster_shape_mn[0]
            + self.cta_id_in_cluster[0]
        )
        cta_intermediate_or_hidden_block_idx = (
            cluster_intermediate_or_hidden_block_idx * params.cluster_shape_mn[1]
            + self.cta_id_in_cluster[1]
        )

        # valid_tokens_in_cta_tile: clip cta_tile_m tokens at the current expert
        # right boundary.
        token_idx_start_in_expert = cta_token_block_idx * Int32(cta_tile_m)
        remaining_in_expert = (
            state.current_this_expert_token_cnt - token_idx_start_in_expert
        )
        remaining_in_expert = cutlass.max(remaining_in_expert, Int32(0))
        valid_tokens_in_cta_tile = cutlass.min(remaining_in_expert, Int32(cta_tile_m))

        # Swap scheduler-internal M/N back to GEMM-domain M/N on output.
        tile_m_idx = cta_intermediate_or_hidden_block_idx
        tile_n_idx = cta_token_block_idx

        # ext.enrich_work_tile_info may OR the peek bit into phase_and_peek.
        return self._ext.WorkTileInfo(
            expert_idx=state.current_expert_idx,
            tile_m_idx=tile_m_idx,
            tile_n_idx=tile_n_idx,
            cumulative_data_physical_row=state.current_data_cumul,
            cumulative_token_block_count=state.current_token_block_cumul,
            valid_tokens_in_cta_tile=valid_tokens_in_cta_tile,
            phase_and_peek=state.current_phase,
        )

    @dsl_user_op
    @cute.jit
    def _gen_work_from_cluster_idx(
        self,
        cluster_linear_tile_idx: Int32,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Decode and publish current work from one cluster-linear tile id."""
        state = self._fused_state

        # Sentinel-by-default work tile; conditionally overwritten by decode.
        base_work = self._ext.WorkTileInfo(
            expert_idx=Int32(WorkTileState.DONE),
            tile_m_idx=Int32(0),
            tile_n_idx=Int32(0),
            cumulative_data_physical_row=Int32(0),
            cumulative_token_block_count=Int32(0),
            valid_tokens_in_cta_tile=Int32(0),
            phase_and_peek=Int32(BlockPhase.None_),
        )

        # DSL carry for mutated self and while-condition fields.
        outer_group_end = state.current_group_end
        outer_group_last_expert_exclusive = state.current_group_last_expert_exclusive
        while (
            cluster_linear_tile_idx >= outer_group_end
            and outer_group_last_expert_exclusive < self.expert_cnt
        ):
            self._advance_group(loc=loc, ip=ip)
            self._fused_state = self._fused_state  # DSL carry
            state = (
                self._fused_state
            )  # re-bind alias inside body so refresh below sees new SSA
            outer_group_end = state.current_group_end
            outer_group_last_expert_exclusive = (
                state.current_group_last_expert_exclusive
            )
        state = self._fused_state  # re-bind alias to the post-while yielded SSA

        is_valid = cluster_linear_tile_idx < state.current_group_end
        if is_valid:
            # fc1 → fc2 phase transition inside current group, if crossed.
            if (
                state.current_phase == Int32(BlockPhase.Linear1)
                and cluster_linear_tile_idx >= state.current_group_fc1_subphase_end
            ):
                self._switch_to_fc2(loc=loc, ip=ip)
                self._fused_state = self._fused_state  # DSL carry
            else:
                self._fused_state = self._fused_state  # balanced else-side rebind
            state = self._fused_state  # re-bind alias

            # non-PyIR: carry loop-condition fields as locals.
            inner_expert_tile_end = state.current_expert_tile_end
            while cluster_linear_tile_idx >= inner_expert_tile_end:
                self._advance_expert_within_phase(loc=loc, ip=ip)
                self._fused_state = self._fused_state  # DSL carry
                state = self._fused_state  # re-bind alias inside body
                inner_expert_tile_end = state.current_expert_tile_end
            state = self._fused_state  # re-bind alias

            base_work = self._decode_inside_expert(
                cluster_linear_tile_idx, loc=loc, ip=ip
            )
        else:
            # Balance scf.if yield for self.
            self._fused_state = self._fused_state

        self.current_work = self._ext.enrich_work_tile_info(base_work)

    @dsl_user_op
    @cute.jit
    def gen_next_work(
        self,
        *,
        loc: Optional[ir.Location] = None,
        ip: Optional[ir.InsertionPoint] = None,
    ) -> None:
        """Produce the next work tile for this cluster's persistent loop.

        Codegen-time fork on ``self._first_advance_pending`` (Python bool,
        set True by ``internal_init``):

          - First trace site (kernel main loop's first
            ``scheduler.gen_next_work()``): static mode is a noop
            (``self.current_work`` was decoded by ``internal_init``);
            atomic mode runs the full ``_advance_work_linear_tile_idx_dynamic
            + _gen_work_from_cluster_idx`` pipeline, with
            ``_advance_work_linear_tile_idx_dynamic`` itself reading
            ``_first_advance_pending`` to consume the cached
            ``ds.atomic_res`` (set by ``internal_init`` BEFORE
            ``pipeline_init_arrive``) instead of issuing a fresh
            ``atom.add``.  At the tail, ``_first_advance_pending`` flips
            to False so subsequent trace sites pick the vanilla path.

          - Second trace site (inside-while-body call): vanilla
            ``_advance_work_linear_tile_idx_*`` + ``_gen_work_from_cluster_idx``
            for both modes.

        First trace site consumes pre-init work; later trace sites advance normally.
        """
        iket.range_push("produce_tile_id")
        # static mode first call short-circuits: internal_init already wrote
        # the first work tile to self.current_work, so just leave it alone.
        # All other (mode, call-site) combinations run the full advance +
        # decode pipeline.  ``_advance_work_linear_tile_idx_dynamic`` itself
        # const_expr-forks on _first_advance_pending to consume the cached
        # atomic_res on its own first trace site.
        if cutlass.const_expr(
            self._first_advance_pending and self.params.load_balance_mode == "static"
        ):
            pass
        else:
            if const_expr(self.params.load_balance_mode == "atomic_counter"):
                cluster_linear_tile_idx = self._advance_work_linear_tile_idx_dynamic(
                    loc=loc, ip=ip
                )
            else:
                cluster_linear_tile_idx = self._advance_work_linear_tile_idx_static(
                    loc=loc, ip=ip
                )
            self._gen_work_from_cluster_idx(cluster_linear_tile_idx, loc=loc, ip=ip)

        # Codegen-time flip after the first trace site so subsequent traces
        # (the while-body call) pick the vanilla path.  This Python
        # attribute write is observed at trace time by the next jit
        # invocation of gen_next_work / _advance_work_linear_tile_idx_dynamic.
        if cutlass.const_expr(self._first_advance_pending):
            self._first_advance_pending = False
        iket.range_pop()

    def _mlir_fields(self):
        fields = (
            "params",
            "num_persistent_clusters",
            "cta_id_in_cluster",
            "current_work",
            "_fused_state",
            "_num_fc1_intermediate_blocks",
            "_num_fc2_hidden_blocks",
        )
        if self.params.load_balance_mode == "atomic_counter":
            fields += ("_dynamic_state",)
        return fields + ("_producer_state",)

    def __extract_mlir_values__(self) -> List[ir.Value]:
        return extract_mlir_values(
            tuple(getattr(self, name) for name in self._mlir_fields())
        )

    def __new_from_mlir_values__(self, values: List[ir.Value]):
        # Pipeline/storage references and the first-trace flag remain Python state.
        result = copy(self)
        idx = 0
        for name in self._mlir_fields():
            field = getattr(self, name)
            count = len(extract_mlir_values(field))
            setattr(
                result, name, new_from_mlir_values(field, values[idx : idx + count])
            )
            idx += count
        if self.params.load_balance_mode == "static":
            result._dynamic_state = None
        return result
