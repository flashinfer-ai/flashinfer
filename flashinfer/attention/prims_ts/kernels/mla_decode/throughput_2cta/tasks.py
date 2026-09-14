# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task definitions for the throughput 2CTA MLA decode TS kernel.

The selected graph depends on dtype and scheduler policy. BF16 uses a 12-warp
combined TMA/MMA graph, with an additional scheduler task under CLC. FP8 uses a
16-warp split K/V and QK/PV graph with two softmax groups.

Domain semantics: domain = k_tile_count (total k-tiles to process).
Tasks with domain_start=1 handle the first k-tile in HEAD, then LOOP
covers k-tiles 1..N-1, and TAIL handles cleanup.

Stagger pattern (K loads 1 ahead of V, QK MMA 1 ahead of PV MMA):
  k-tile 0: load K[0], QK MMA -> S[0]
  k-tile 1: load K[1]+V[0], PV MMA(P[0],V[0])->O[0], QK MMA->S[1]
  ...
  k-tile N-1: load K[N-1]+V[N-2], PV MMA(P[N-2],V[N-2])->O[N-2], QK MMA->S[N-1]
  tail: load V[N-1], PV MMA(P[N-1],V[N-1])->O[N-1]

The BF16 register roles are:
  LoadTmaTask     (warp 9,    1 warp,   96 regs; owns page-ID window)
  MmaTask         (warp 8,    1 warp,   96 regs)
  SoftmaxTask     (warps 0-3, 4 warps, 192 regs)
  CorrectionTask  (warps 4-7, 4 warps, 208 regs)
  PaddingTask     (warps 10-11,          96 regs; non-CLC alignment)
  SchedulerTask   (warp 11,              96 regs; CLC only)
"""

from collections.abc import Callable
from dataclasses import dataclass

import cutlass
import cutlass.cute as cute

from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.enums import ScheduleStage
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.resources import StageInfo
from cutlass.experimental.task_scheduling.task import Task

from .resources import (
    MlaTsWorkTileInfo,
    MlaWorkQueue,
    WorkThrottleBarrierResource,
    Dsv4PageOffsetRingResource,
    PageOffsetWindowResource,
    SmemQResource,
    SmemKResource,
    SmemKVResource,
    SmemVResource,
    SmemPResource,
    TmemSResource,
    TmemCorrResource,
    TmemOResource,
    GmemOResource,
)
from ..helpers.schedule import (
    captured_loop_bounds,
    staged_kv_tma_load,
    staged_pv_mma,
    staged_pv_mma_v_tile,
    staged_pv_mma_v_tile_per_n,
    staged_pv_mma_v_tile_source_direct_p,
    staged_pv_mma_v_tile_per_n_source_direct_p,
    staged_qk_mma,
    staged_qk_mma_k_tile,
    staged_qk_mma_k_tile_from_acquired_s,
    work_queue_tail,
)


def _fixed_lane_work_tile_bounds(fixed_lane, cumulative_k_parity, actual_domain):
    """Model fixed-lane work-tile bounds for CPU contract tests.

    Generated code keeps this arithmetic inline because returning staged DSL
    values through a plain-Python tuple breaks staged-frontend state threading.
    """

    mapped_start = (fixed_lane + cumulative_k_parity) % 2
    lane_iterations = (actual_domain - mapped_start + 1) // 2
    fixed_loop_end = fixed_lane + lane_iterations * 2
    return mapped_start, fixed_loop_end


@dataclass(frozen=True)
class Dsv4PairRingEvent:
    """One observable ownership event in DSV4's W9 selector-ring contract.

    This is deliberately a host-only model.  It is not part of generated
    device code and does not claim to encode the source's exact instruction
    interleaving.  It fixes the non-negotiable state-machine properties before
    the TaskManager implementation changes: two producer stages per K-tile
    pair, plus one held consumer stage per operand and pair.
    """

    kind: str
    operand: str
    tile: int
    pair: int
    half: int
    stage: int


def dsv4_pair_ring_event_plan(k_tiles: int) -> tuple[Dsv4PairRingEvent, ...]:
    """Return the source W9 pair-ring ownership plan for ``k_tiles`` K128 tiles.

    ``SmemPageOffsetsKv`` has six physical stages.  For pair ``p``, W9 commits
    two identical ``int32[256]`` payloads: the K stage ``2*p`` and V stage
    ``2*p+1`` (modulo six).  Each payload contains tile ``2*p`` in half zero
    and tile ``2*p+1`` in half one.  W12--W15 wait once per operand/pair,
    retain that state for both Gather4 tiles, then release it after the second
    tile -- or after the odd tail's sole first half.

    Producer events are listed first because W9 is a distinct warp role.  The
    consumer portion retains the source's load order: HEAD consumes K0; LOOP
    iteration ``i`` consumes K(i+1) before V(i); TAIL releases K, then consumes
    V(last), then releases V.  This remains a resource-ownership oracle, not
    a substitute for a SASS or timing comparison.
    """

    if k_tiles < 0:
        raise ValueError(f"k_tiles must be non-negative, got {k_tiles}")

    events: list[Dsv4PairRingEvent] = []
    pair_count = (k_tiles + 1) // 2
    for pair in range(pair_count):
        for operand, stage_offset in (("K", 0), ("V", 1)):
            events.append(
                Dsv4PairRingEvent(
                    "produce", operand, pair * 2, pair, 0, (pair * 2 + stage_offset) % 6
                )
            )

    if not k_tiles:
        return tuple(events)

    def append_consumer(kind: str, operand: str, tile: int) -> None:
        pair, half = divmod(tile, 2)
        stage_offset = 0 if operand == "K" else 1
        events.append(
            Dsv4PairRingEvent(
                kind, operand, tile, pair, half, (2 * pair + stage_offset) % 6
            )
        )

    # HEAD: wait/read the K pair then issue K[0].
    append_consumer("wait", "K", 0)
    append_consumer("gather", "K", 0)

    # LOOP: K is deliberately one tile ahead of V.  The K and V selector
    # stages are independently held for the two halves of their pair.
    for loop_offset in range(k_tiles - 1):
        k_tile = loop_offset + 1
        if k_tile % 2 == 0:
            append_consumer("wait", "K", k_tile)
        append_consumer("gather", "K", k_tile)
        if k_tile % 2 == 1 and k_tile != k_tiles - 1:
            append_consumer("release", "K", k_tile)

        if loop_offset % 2 == 0:
            append_consumer("wait", "V", loop_offset)
        append_consumer("gather", "V", loop_offset)
        if loop_offset % 2 == 1:
            append_consumer("release", "V", loop_offset)

    # TAIL: source releases the final K stage before issuing V[last].
    append_consumer("release", "K", k_tiles - 1)
    last_tile = k_tiles - 1
    if last_tile % 2 == 0:
        append_consumer("wait", "V", last_tile)
    append_consumer("gather", "V", last_tile)
    append_consumer("release", "V", last_tile)
    return tuple(events)


def _capture_clc_work_tile_body(
    work_queue,
    body: Callable[..., None],
    non_skippable_prelude: Callable[[], object] | None = None,
    non_skippable_control: Callable[[], None] | None = None,
    *,
    use_clc_dynamic: bool = False,
) -> None:
    """Capture one MLA tile with data work skippable and WQ progress mandatory.

    Pure register-state initializers may run in ``non_skippable_prelude`` so
    values they create dominate the separately guarded HEAD, LOOP, and TAIL
    regions emitted by the stock skipped-tile executor.  The prelude must not
    issue memory operations or advance pipeline state.  A source protocol may
    instead use ``non_skippable_control`` for pipeline events that advance once
    per WorkId even when that tile has no active data work.
    """

    def run_body():
        prelude_state = (
            non_skippable_prelude() if non_skippable_prelude is not None else None
        )
        if non_skippable_control is not None:
            non_skippable_control()
        if non_skippable_prelude is None:
            body()
        else:
            body(prelude_state)

    if work_queue is not None and use_clc_dynamic:
        with work_tile_loop(
            work_queue,
            skip_if=MlaWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            prelude_state = (
                non_skippable_prelude() if non_skippable_prelude is not None else None
            )
            if non_skippable_control is not None:
                non_skippable_control()
            with work_tiles.skippable():
                if non_skippable_prelude is None:
                    body()
                else:
                    body(prelude_state)
            work_queue_tail(work_queue, advance_label="advance_tile")
        return

    run_body()
    work_queue_tail(work_queue, advance_label="advance_tile")


class MlaClcTask(Task):
    """Stock Task persistent loop with an MLA-specific dynamic K domain."""

    def __init__(
        self, *args, skip_terminal_producer_tails: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._skip_terminal_producer_tails = skip_terminal_producer_tails

    @cute.jit
    def _work_tile_scalars(self, work_tile):
        """Flatten a runtime MLA work tile into scalar loop-carried values.

        CuTe DSL containers cannot be rebound inside ``scf.while`` once any
        member becomes a runtime SSA value.  Mutating the members in place is
        accepted by the frontend, but does not make those mutations explicit
        loop yields.  Keep the CLC cursor as eight scalars instead and rebuild
        the value object only for one task-body invocation.
        """

        cluster_idx, seq_q_idx, batch_idx, split_kv_idx = work_tile.tile_idx
        return (
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            work_tile.is_valid_tile,
            work_tile.k_len,
            work_tile.k_tile_count,
            work_tile.k_index_base,
        )

    @cute.jit
    def _work_tile_from_scalars(
        self,
        cluster_idx,
        seq_q_idx,
        batch_idx,
        split_kv_idx,
        is_valid,
        k_len,
        k_tile_count,
        k_index_base,
    ):
        """Materialize a short-lived work-tile value from scalar state."""

        return MlaTsWorkTileInfo(
            (cluster_idx, seq_q_idx, batch_idx, split_kv_idx),
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        )

    @cute.jit
    def _run_clc_work_tile_from_scalars(
        self,
        cluster_idx,
        seq_q_idx,
        batch_idx,
        split_kv_idx,
        is_valid,
        k_len,
        k_tile_count,
        k_index_base,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one CLC tile without carrying a container across the loop."""

        work_tile = self._work_tile_from_scalars(
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        )
        if cutlass.const_expr(self._has_skip_if):
            skip_work_tile = self._should_skip_work_tile(work_tile)
            self._run_task_body_impl(work_tile, skip_work_tile, context)
        else:
            self._run_task_body_impl(work_tile, context=context)

    @cute.jit
    def get_domain(self, tile_coord):
        """Recompute the loop bound required by the stock Task public API."""

        assert isinstance(self.work_queue, MlaWorkQueue)
        return self.work_queue.k_tile_count_for_tile(tile_coord)

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Run CLC with an explicitly scalarized MLA work-tile cursor.

        Once B/S scheduler extents become runtime values, ``MlaTsWorkTileInfo``
        carries staged fields. Rebinding it after WorkQueue publishes the next
        response triggers ``CONTAINER_OBJECT_REPLACED``; mutating it in place
        fails to yield every member through the staged while loop. Scalarize
        all eight fields so the loop-carried dependency is explicit without
        changing the acquire/work/tail protocol.
        """

        assert self.work_queue is not None
        work_tile = self.work_queue.initial_work_tile_info()
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, context)
        (
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        ) = self._work_tile_scalars(
            self.work_queue._get_consumer_var_from_ts("work_tile")
        )
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource.pipeline_config.advance_on_acquire
                and not self._is_fork_secondary(resource)
            ):
                self._thread_advance_on_acquire_state(resource)
        while is_valid:
            self._run_clc_work_tile_from_scalars(
                cluster_idx,
                seq_q_idx,
                batch_idx,
                split_kv_idx,
                is_valid,
                k_len,
                k_tile_count,
                k_index_base,
                context,
            )
            (
                cluster_idx,
                seq_q_idx,
                batch_idx,
                split_kv_idx,
                is_valid,
                k_len,
                k_tile_count,
                k_index_base,
            ) = self._work_tile_scalars(
                self.work_queue._get_consumer_var_from_ts("work_tile")
            )

        terminal_work_tile = self._work_tile_from_scalars(
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        )
        self._run_post_work_loop_entries(terminal_work_tile, context)
        if cutlass.const_expr(not self._skip_terminal_producer_tails):
            for resource in self.dst_resources:
                if cutlass.const_expr(
                    resource.pipeline_config is not None
                    and resource is not self.work_queue
                    and not self._is_fork_secondary(resource)
                ):
                    pipeline_config = resource.pipeline_config
                    assert pipeline_config is not None
                    if cutlass.const_expr(pipeline_config.advance_on_acquire):
                        self._thread_advance_on_acquire_state(resource)
                    self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()


class MlaDsv4WholeTileGuardClcTask(MlaClcTask):
    """Run DSV4 data under one guard between common head and tail entries.

    Task's generic skipped-tile executor supports arbitrary interleaving of
    skippable and non-skippable HEAD/LOOP/TAIL entries, so it emits separate
    dynamic guards for those regions. DSV4's fused MMA and Q/K/V load roles
    have a stricter source shape: pure cursor initialization and the W12--W15
    throttle form a common head, all data events form one guarded region, and
    WorkQueue progress forms a common tail. Keep those common paths outside
    the branch instead of cloning them into active and skipped CFG diamonds.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # This executor deliberately reorders nothing. Its single-envelope
        # lowering is legal only for the captured source shape: common HEAD
        # entries precede data HEAD entries, and common TAIL entries follow
        # data TAIL entries. Fail construction if a future schedule interleaves
        # either class instead of silently changing its protocol.
        seen_data_head = False
        for is_skippable, _ in self._head_exec_groups:
            if is_skippable:
                seen_data_head = True
            elif seen_data_head:
                raise ValueError(
                    "DSV4 whole-tile guard requires a common-head/data-head envelope"
                )
        seen_common_tail = False
        for is_skippable, _ in self._tail_exec_groups:
            if not is_skippable:
                seen_common_tail = True
            elif seen_common_tail:
                raise ValueError(
                    "DSV4 whole-tile guard requires a data-tail/common-tail envelope"
                )

    @cute.jit
    def _run_task_body_impl_schedule(
        self,
        work_tile,
        skip_work_tile=None,
        context: ResourceContext | None = None,
    ) -> None:
        """Lower common HEAD -> one guarded data body -> common TAIL."""

        if cutlass.const_expr(skip_work_tile is None):
            Task._run_task_body_impl_schedule(self, work_tile, skip_work_tile, context)
            return

        # Common entries do not consume the K loop, but Task's entry helpers
        # require its domain in StageInfo. Reuse the already-decoded value so
        # a padded WorkId does not perform a second metadata lookup.
        common_domain = work_tile.k_tile_count

        for is_skippable, head_entries in self._head_exec_groups:
            if cutlass.const_expr(not is_skippable):
                self._run_head_entry_group(
                    head_entries, work_tile, common_domain, context
                )

        if skip_work_tile == cutlass.Boolean(False):
            # Preserve the current active-data domain path in this CFG-only
            # experiment. Replacing it with the cached value is a separate
            # optimization axis.
            data_domain = self.get_domain(work_tile.tile_idx)
            for is_skippable, head_entries in self._head_exec_groups:
                if cutlass.const_expr(is_skippable):
                    self._run_head_entry_group(
                        head_entries, work_tile, data_domain, context
                    )
            if cutlass.const_expr(self._has_last_iter):
                self._run_loop_peeled(data_domain, work_tile, context)
            else:
                self._run_loop_simple(data_domain, work_tile, context)
            for is_skippable, tail_entries in self._tail_exec_groups:
                if cutlass.const_expr(is_skippable):
                    self._run_tail_entry_group(
                        tail_entries, work_tile, data_domain, context
                    )

        for is_skippable, tail_entries in self._tail_exec_groups:
            if cutlass.const_expr(not is_skippable):
                self._run_tail_entry_group(
                    tail_entries, work_tile, common_domain, context
                )

    @cute.jit
    def _run_clc_work_tile_from_scalars(
        self,
        cluster_idx,
        seq_q_idx,
        batch_idx,
        split_kv_idx,
        is_valid,
        k_len,
        k_tile_count,
        k_index_base,
        context: ResourceContext | None = None,
    ) -> None:
        work_tile = self._work_tile_from_scalars(
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        )
        if cutlass.const_expr(self._has_skip_if):
            skip_work_tile = self._should_skip_work_tile(work_tile)
            self._run_task_body_impl(work_tile, skip_work_tile, context)
        else:
            self._run_task_body_impl(work_tile, context=context)


class MlaTask(Task):
    """Task subclass that recomputes MLA k-domain per persistent work tile."""

    def __init__(
        self, *args, skip_terminal_producer_tails: bool = False, **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self._skip_terminal_producer_tails = skip_terminal_producer_tails

    @cute.jit
    def _run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run a persistent work tile using the cached split-KV domain."""

        # WorkQueue decomposes the MLA persistent tile and caches the K-domain.
        # Keep task bodies on that cached value so page-offset/TMA/MMA/softmax
        # paths do not each rebuild the same split-KV arithmetic.
        self.domain = work_tile.k_tile_count
        self._run_task_body_impl(work_tile, context=context)

    @cute.jit
    def _drain_mla_work_tile_tails(self) -> None:
        """Drain producer tails after a persistent work-tile body completes."""

        if cutlass.const_expr(not self._skip_terminal_producer_tails):
            for resource in self.dst_resources:
                if cutlass.const_expr(
                    resource.pipeline_config is not None
                    and resource is not self.work_queue
                    and not self._is_fork_secondary(resource)
                ):
                    if cutlass.const_expr(
                        resource.pipeline_config.producer_acquire_interleave_stride > 1
                        or resource.pipeline_config.producer_commit_interleave_stride
                        > 1
                    ):
                        # Interleaved producers own lane-specific pipeline states.
                        # The generic producer tail still drains at resource
                        # granularity, so calling it here can wait on a peer lane's
                        # physical stages.  Consumer wait/release drains the live
                        # lane for these score/P resources.
                        pass
                    else:
                        self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue is not None
            and self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Schedule one or more persistent work tiles for this task instance."""

        params = self.work_queue.tile_sched_params

        if cutlass.const_expr(not params.is_persistent):
            work_tile = self.work_queue._work_tile_from_block_idx(cute.arch.block_idx())
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
            self._run_pre_work_loop_entries(work_tile, context)
            # Runtime K/Q metadata can make a statically launched split empty.
            # Keep the CTA on the ordinary initialized-pipeline path, but skip
            # its captured HEAD/LOOP/TAIL data work when the domain is zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self._run_one_mla_work_tile(work_tile, context)
            self._run_post_work_loop_entries(work_tile, context)
            self._drain_mla_work_tile_tails()
            return

        current_work_linear_idx = cute.arch.block_idx()[0]
        num_blocks = (
            params.cluster_shape_mnk[0]
            * params.problem_shape_s
            * params.problem_shape_b
            * params.split_kv
        )
        work_tile = self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, context)
        while current_work_linear_idx < num_blocks:
            work_tile.update_from(
                self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
            )
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

            # Variable K and causal Q visibility can leave individual logical
            # splits empty. Skip them and continue grid-striding rather than
            # running a captured HEAD/TAIL sequence with domain zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self._run_one_mla_work_tile(work_tile, context)

            # Each warp branch advances from the same scalar tile id, keeping
            # the persistent loop state compact across task bodies.
            current_work_linear_idx += cute.size(cute.arch.grid_dim())
            # self.dummy keeps the captured persistent loop body live even when
            # a specialized task instance has no visible local result.
            self.dummy = True
        work_tile.update_from(
            self.work_queue._work_tile_from_linear_idx(current_work_linear_idx)
        )
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
        self._run_post_work_loop_entries(work_tile, context)
        self._drain_mla_work_tile_tails()


class MlaDsv4SourceSHandoffTask(MlaTask):
    """Persistent W8 executor with TRTLLM's cross-work-tile S state machine.

    The base task's HEAD/LOOP/TAIL are per work tile.  Generated DSV4 instead
    opens two S stages once before the persistent loop and balances two commits
    once at exit, so these operations cannot be regular captured HEAD/TAIL
    entries.
    """

    def __init__(self, *args, source_tmem_lifecycle: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dsv4_source_tmem_lifecycle = source_tmem_lifecycle
        source_resources = [
            resource
            for resource in self.dst_resources
            if isinstance(resource, TmemSResource)
        ]
        if len(source_resources) != 1:
            raise ValueError(
                "MlaDsv4SourceSHandoffTask requires exactly one TmemSResource destination"
            )
        self._dsv4_source_s = source_resources[0]

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        self._dsv4_source_s.source_preacquire_pair()
        super()._run_task_body_persistent(context)

    @cute.jit
    def _drain_mla_work_tile_tails(self) -> None:
        # Generated DSV4 balances the two speculative S pre-acquires with
        # exactly two terminal commits.  Calling MlaTask's generic
        # ``producer_tail`` for S afterwards waits/drains a third protocol
        # that does not exist and deadlocks the real specialization.  Drain
        # every other destination exactly as the base implementation does,
        # but make the source S balance the sole terminal operation for S.
        self._dsv4_source_s.source_balance_after_persistent_loop()
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource is not self._dsv4_source_s
                and not (
                    self._dsv4_source_tmem_lifecycle
                    and isinstance(resource, TmemOResource)
                )
                and resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                if cutlass.const_expr(
                    resource.pipeline_config.producer_acquire_interleave_stride > 1
                    or resource.pipeline_config.producer_commit_interleave_stride > 1
                ):
                    pass
                else:
                    self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue is not None
            and self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()


class MlaDsv4SourceSHandoffClcTask(MlaDsv4WholeTileGuardClcTask):
    """CLC W8 executor with the source cross-work-tile S cursor protocol.

    ``Task`` owns the CLC WorkId loop, while ``MlaDsv4SourceSHandoffTask``
    above owns the faster static grid-stride loop.  Source S lookahead is
    outside either loop: acquire S0/S1 once before the first WorkId and emit
    exactly two balancing commits after the terminal WorkId.  The stock CLC
    tail cannot express that exception because it advances and drains every
    ``advance_on_acquire`` destination uniformly, so keep the source S tail
    explicit while preserving the stock loop and all other resource tails.
    """

    def __init__(self, *args, source_tmem_lifecycle: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dsv4_source_tmem_lifecycle = source_tmem_lifecycle
        source_resources = [
            resource
            for resource in self.dst_resources
            if isinstance(resource, TmemSResource)
        ]
        if len(source_resources) != 1:
            raise ValueError(
                "MlaDsv4SourceSHandoffClcTask requires exactly one "
                "TmemSResource destination"
            )
        self._dsv4_source_s = source_resources[0]

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Run the WorkId loop between source's one-time S head and tail."""

        assert self.work_queue is not None
        self._dsv4_source_s.source_preacquire_pair()

        work_tile = self.work_queue.initial_work_tile_info()
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
        self._run_pre_work_loop_entries(work_tile, context)
        (
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        ) = self._work_tile_scalars(
            self.work_queue._get_consumer_var_from_ts("work_tile")
        )

        while is_valid:
            self._run_clc_work_tile_from_scalars(
                cluster_idx,
                seq_q_idx,
                batch_idx,
                split_kv_idx,
                is_valid,
                k_len,
                k_tile_count,
                k_index_base,
                context,
            )
            (
                cluster_idx,
                seq_q_idx,
                batch_idx,
                split_kv_idx,
                is_valid,
                k_len,
                k_tile_count,
                k_index_base,
            ) = self._work_tile_scalars(
                self.work_queue._get_consumer_var_from_ts("work_tile")
            )

        terminal_work_tile = self._work_tile_from_scalars(
            cluster_idx,
            seq_q_idx,
            batch_idx,
            split_kv_idx,
            is_valid,
            k_len,
            k_tile_count,
            k_index_base,
        )
        self._run_post_work_loop_entries(terminal_work_tile, context)
        self._dsv4_source_s.source_balance_after_persistent_loop()

        # Match Task's CLC tail for every destination except S.  S has just
        # consumed its only legal source tail above; a generic producer_tail
        # would wait on a third protocol and deadlock.
        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource is not self._dsv4_source_s
                and not (
                    self._dsv4_source_tmem_lifecycle
                    and isinstance(resource, TmemOResource)
                )
                and resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                pipeline_config = resource.pipeline_config
                assert pipeline_config is not None
                if cutlass.const_expr(pipeline_config.advance_on_acquire):
                    self._thread_advance_on_acquire_state(resource)
                self._producer_tail(resource)
        if cutlass.const_expr(
            self.work_queue in self.dst_resources
            and self.work_queue.pipeline_config is not None
        ):
            self.work_queue.producer_tail()


class MlaDsv4PairProducerTask(MlaTask):
    """W9 task whose domain is sparse K-tile pairs, not individual K tiles.

    The rest of the kernel retains the full K-tile domain.  Only W9 produces
    two selector stages per adjacent pair, so mapping its local pair index back
    to ``2 * pair`` avoids a dynamic branch around ``cp.async``/commit.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._dsv4_full_k_domain = cutlass.Int32(0)

    @cute.jit
    def _run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        self._dsv4_full_k_domain = work_tile.k_tile_count
        self.domain = (self._dsv4_full_k_domain + cutlass.Int32(1)) // cutlass.Int32(2)
        self._run_task_body_impl(work_tile, context=context)

    @cute.jit
    def _create_stage_info(
        self,
        resource,
        idx,
        work_tile=None,
        is_producer=None,
        resolved_domain=None,
        label=None,
        schedule_stage=None,
        routing_slot=None,
        context: ResourceContext | None = None,
    ) -> StageInfo:
        """Expose pair base ``2*p`` to W9's source page-index calculation."""

        base_info = Task._create_stage_info(
            self,
            resource,
            idx,
            work_tile,
            is_producer,
            resolved_domain,
            label,
            schedule_stage,
            routing_slot,
            context=context,
        )
        return StageInfo(
            loop_offset=cutlass.Int32(base_info.loop_offset) * cutlass.Int32(2),
            loop_start=cutlass.Int32(base_info.loop_start) * cutlass.Int32(2),
            loop_end=self._dsv4_full_k_domain,
            loop_step=cutlass.Int32(base_info.loop_step) * cutlass.Int32(2),
            stage_idx=base_info.stage_idx,
            label=base_info.label,
            barrier=base_info.barrier,
            work_tile=base_info.work_tile,
            num_active_stages=base_info.num_active_stages,
            context=base_info.context,
            task_cache=base_info.task_cache,
        )


class MlaDsv4PairLoadTask(MlaTask):
    """W12--W15 source-pair consumer with K-ahead-of-V scheduling.

    The captured schedule has one K and one V producer segment.  This task
    deliberately replays those segments in generated-source order instead of
    the generic same-iteration K/V order.  The W9 ring's wait/release state is
    advanced only at a pair boundary, while K/V TMA stage indices keep the full
    logical K-tile index.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._init_dsv4_pair_loop_groups()

    def _init_dsv4_pair_loop_groups(self) -> None:
        """Decode the source-pair schedule by semantic labels, not offsets.

        ``create_load_dsv4_task`` deliberately has two page-ring segments:
        ``wait/read_k -> K acquire/work/commit -> release`` followed by the
        corresponding V segment.  The old fixed ``[0:2]/[2:5]/...`` slices
        silently became wrong if a TaskManager version inserted (for example)
        a ``ConsumerTryWait``.  Use the routing labels as the ABI boundary and
        reject a changed data-flow shape explicitly.
        """

        loop_entries = [
            (resource, stage, call_id, label)
            for resource, stage, call_id, _guard, label in self.loop_schedule_list
        ]

        def parse_operand_segment(label: str, start: int):
            read_positions = [
                idx
                for idx in range(start, len(loop_entries))
                if loop_entries[idx][2] is not None and loop_entries[idx][3] == label
            ]
            if len(read_positions) != 1:
                raise ValueError(
                    f"DSV4 pair-ring expected exactly one {label!r} loop entry, "
                    f"found {len(read_positions)}"
                )
            read_idx = read_positions[0]
            page_ring = loop_entries[read_idx][0]
            wait_entries = loop_entries[start : read_idx + 1]
            if not wait_entries or any(
                entry[0] is not page_ring for entry in wait_entries
            ):
                raise ValueError(
                    f"DSV4 pair-ring {label!r} wait group must use one page-offset ring"
                )
            if wait_entries[-1][1] != ScheduleStage.ConsumerWork:
                raise ValueError(
                    f"DSV4 pair-ring {label!r} must end its wait group in ConsumerWork"
                )
            if any(
                entry[1]
                not in {
                    ScheduleStage.ConsumerTryWait,
                    ScheduleStage.ConsumerWait,
                    ScheduleStage.ConsumerWork,
                }
                for entry in wait_entries
            ):
                raise ValueError(
                    f"DSV4 pair-ring {label!r} has an unexpected page-ring wait stage"
                )

            release_positions = [
                idx
                for idx in range(read_idx + 1, len(loop_entries))
                if loop_entries[idx][0] is page_ring
                and loop_entries[idx][1] == ScheduleStage.ConsumerRelease
            ]
            if not release_positions:
                raise ValueError(
                    f"DSV4 pair-ring {label!r} is missing page-ring ConsumerRelease"
                )
            release_idx = release_positions[0]
            work_entries = loop_entries[read_idx + 1 : release_idx]
            expected_work_stages = (
                ScheduleStage.ProducerAcquire,
                ScheduleStage.ProducerWork,
                ScheduleStage.ProducerCommit,
            )
            if (
                len(work_entries) != len(expected_work_stages)
                or tuple(entry[1] for entry in work_entries) != expected_work_stages
                or len({id(entry[0]) for entry in work_entries}) != 1
                or work_entries[0][0] is page_ring
            ):
                raise ValueError(
                    f"DSV4 pair-ring {label!r} requires one K/V acquire-work-commit group"
                )
            return (
                wait_entries,
                work_entries,
                [loop_entries[release_idx]],
                release_idx + 1,
                work_entries[0][0],
            )

        (
            self._dsv4_k_wait_entries,
            self._dsv4_k_work_entries,
            self._dsv4_k_release_entries,
            next_start,
            k_resource,
        ) = parse_operand_segment("read_k_stage", 0)
        (
            self._dsv4_v_wait_entries,
            self._dsv4_v_work_entries,
            self._dsv4_v_release_entries,
            next_start,
            v_resource,
        ) = parse_operand_segment("read_v_stage", next_start)
        if k_resource is v_resource or next_start != len(loop_entries):
            raise ValueError(
                "DSV4 pair-ring requires adjacent, distinct K and V loop segments"
            )

    @cute.jit
    def _run_loop_simple(
        self,
        resolved_domain,
        work_tile,
        context: ResourceContext | None,
    ) -> None:
        """Replay generated W12--W15 HEAD/LOOP/TAIL for one K-tile domain."""

        # CLC rewrites a padded/skipped tile's dynamic domain to zero while
        # retaining the outer WorkQueue advance.  Unlike Task's generic loop,
        # this source-shaped body also carries its own HEAD and TAIL, so all
        # K/V/page-ring state must stay untouched for that tile.  The source
        # throttle advances independently once per WorkId and is captured as
        # a non-skippable head entry outside this data-path body.
        if resolved_domain > cutlass.Int32(0):
            # HEAD: Q was issued by the captured head schedule.  Consume
            # selector K-pair 0 and issue K0 before any V work.
            self._run_group_entries(
                self._dsv4_k_wait_entries,
                cutlass.Int32(0),
                work_tile,
                resolved_domain,
                context,
            )
            self._run_group_entries(
                self._dsv4_k_work_entries,
                cutlass.Int32(0),
                work_tile,
                resolved_domain,
                context,
            )

            # LOOP i: K[i+1] is one tile ahead of V[i].  Every operand keeps
            # its own W9 selector stage for the pair's two halves.
            for loop_offset in cutlass.range(
                cutlass.Int32(0),
                resolved_domain - cutlass.Int32(1),
                cutlass.Int32(1),
            ):
                next_k_tile = loop_offset + cutlass.Int32(1)
                if (next_k_tile & cutlass.Int32(1)) == cutlass.Int32(0):
                    self._run_group_entries(
                        self._dsv4_k_wait_entries,
                        next_k_tile,
                        work_tile,
                        resolved_domain,
                        context,
                    )
                self._run_group_entries(
                    self._dsv4_k_work_entries,
                    next_k_tile,
                    work_tile,
                    resolved_domain,
                    context,
                )
                if (next_k_tile & cutlass.Int32(1)) == cutlass.Int32(
                    1
                ) and next_k_tile != resolved_domain - cutlass.Int32(1):
                    self._run_group_entries(
                        self._dsv4_k_release_entries,
                        next_k_tile,
                        work_tile,
                        resolved_domain,
                        context,
                    )

                if (loop_offset & cutlass.Int32(1)) == cutlass.Int32(0):
                    self._run_group_entries(
                        self._dsv4_v_wait_entries,
                        loop_offset,
                        work_tile,
                        resolved_domain,
                        context,
                    )
                self._run_group_entries(
                    self._dsv4_v_work_entries,
                    loop_offset,
                    work_tile,
                    resolved_domain,
                    context,
                )
                if (loop_offset & cutlass.Int32(1)) == cutlass.Int32(1):
                    self._run_group_entries(
                        self._dsv4_v_release_entries,
                        loop_offset,
                        work_tile,
                        resolved_domain,
                        context,
                    )

            # TAIL: release the final K selector before issuing V[last].
            last_tile = resolved_domain - cutlass.Int32(1)
            self._run_group_entries(
                self._dsv4_k_release_entries,
                last_tile,
                work_tile,
                resolved_domain,
                context,
            )
            if (last_tile & cutlass.Int32(1)) == cutlass.Int32(0):
                self._run_group_entries(
                    self._dsv4_v_wait_entries,
                    last_tile,
                    work_tile,
                    resolved_domain,
                    context,
                )
            self._run_group_entries(
                self._dsv4_v_work_entries,
                last_tile,
                work_tile,
                resolved_domain,
                context,
            )
            self._run_group_entries(
                self._dsv4_v_release_entries,
                last_tile,
                work_tile,
                resolved_domain,
                context,
            )


class MlaDsv4PairClcProducerTask(MlaClcTask):
    """CLC form of W9's pair-index producer.

    The generic CLC task owns the WorkId persistent loop.  Only W9's local
    domain is changed from K128 tiles to adjacent K-tile pairs; its stage
    metadata still exposes the pair base in source token-index coordinates.
    """

    @cute.jit
    def get_domain(self, tile_coord):
        assert isinstance(self.work_queue, MlaWorkQueue)
        full_k_domain = self.work_queue.k_tile_count_for_tile(tile_coord)
        return (full_k_domain + cutlass.Int32(1)) // cutlass.Int32(2)

    @cute.jit
    def _create_stage_info(
        self,
        resource,
        idx,
        work_tile=None,
        is_producer=None,
        resolved_domain=None,
        label=None,
        schedule_stage=None,
        routing_slot=None,
        context: ResourceContext | None = None,
    ) -> StageInfo:
        """Map CLC pair-loop offsets to the full K128 selector row."""

        base_info = Task._create_stage_info(
            self,
            resource,
            idx,
            work_tile,
            is_producer,
            resolved_domain,
            label,
            schedule_stage,
            routing_slot,
            context=context,
        )
        full_k_domain = work_tile.k_tile_count
        return StageInfo(
            loop_offset=cutlass.Int32(base_info.loop_offset) * cutlass.Int32(2),
            loop_start=cutlass.Int32(base_info.loop_start) * cutlass.Int32(2),
            loop_end=full_k_domain,
            loop_step=cutlass.Int32(base_info.loop_step) * cutlass.Int32(2),
            stage_idx=base_info.stage_idx,
            label=base_info.label,
            barrier=base_info.barrier,
            work_tile=base_info.work_tile,
            num_active_stages=base_info.num_active_stages,
            context=base_info.context,
            task_cache=base_info.task_cache,
        )


class MlaDsv4PairClcLoadTask(MlaDsv4WholeTileGuardClcTask):
    """CLC form of the source-order W12--W15 pair-ring consumer."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        MlaDsv4PairLoadTask._init_dsv4_pair_loop_groups(self)

    # The pair timeline depends only on Task's captured schedule entries and
    # the dynamic full-K domain. Reuse the already-qualified source-order
    # body; unlike its __init__, it has no MlaTask-specific super() state.
    _run_loop_simple = MlaDsv4PairLoadTask._run_loop_simple


class MlaInterleavedTask(MlaTask):
    """Persistent task that carries its interleave lane across work tiles."""

    def __init__(self, *args, **kwargs) -> None:
        """Create staged parity and index-remapping state for this task."""

        super().__init__(*args, **kwargs)
        self._cumulative_k_parity = cutlass.Int32(0)
        self._mapped_domain_start = cutlass.Int32(self.domain_start)
        self._actual_domain = cutlass.Int32(0)
        self._fixed_loop_end = cutlass.Int32(self.domain_start)

    @cute.jit
    def _run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one tile by mapping its local offsets onto this fixed lane."""

        fixed_lane = cutlass.Int32(self.domain_start)
        self._actual_domain = work_tile.k_tile_count
        self._mapped_domain_start = (
            fixed_lane + self._cumulative_k_parity
        ) % cutlass.Int32(2)
        lane_iterations = (
            self._actual_domain - self._mapped_domain_start + cutlass.Int32(1)
        ) // cutlass.Int32(2)
        self._fixed_loop_end = fixed_lane + lane_iterations * cutlass.Int32(2)
        self.domain = self._fixed_loop_end
        self._run_task_body_impl(work_tile, context=context)
        self._cumulative_k_parity = (
            self._cumulative_k_parity + self._actual_domain
        ) % cutlass.Int32(2)

    @cute.jit
    def _create_stage_info(
        self,
        resource,
        idx,
        work_tile=None,
        is_producer=None,
        resolved_domain=None,
        label=None,
        schedule_stage=None,
        routing_slot=None,
        context: ResourceContext | None = None,
    ) -> StageInfo:
        """Return base pipeline state with the work tile's actual K offset."""

        base_info = Task._create_stage_info(
            self,
            resource,
            idx,
            work_tile,
            is_producer,
            resolved_domain,
            label,
            schedule_stage,
            routing_slot,
            context=context,
        )
        actual_loop_offset = self._mapped_domain_start + (
            cutlass.Int32(base_info.loop_offset) - cutlass.Int32(self.domain_start)
        )
        return StageInfo(
            loop_offset=actual_loop_offset,
            loop_start=self._mapped_domain_start,
            loop_end=self._actual_domain,
            loop_step=base_info.loop_step,
            stage_idx=base_info.stage_idx,
            label=base_info.label,
            barrier=base_info.barrier,
            work_tile=base_info.work_tile,
            num_active_stages=base_info.num_active_stages,
            context=base_info.context,
            task_cache=base_info.task_cache,
        )


def create_load_tma_task(
    page_offset_window: PageOffsetWindowResource,
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    iterations_qk: int = 9,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the TMA load task (warp 9, 1 warp, 96 regs).

    domain_start=1: HEAD handles k-tile[0], LOOP handles k-tiles 1..N-1.

    HEAD: consume page offsets[0], load Q, load K[0].
    LOOP: consume page offsets[n], load K[n], load V[n-1].
    TAIL: load V[last].
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def load_tma_prelude(page_offset_window, smem_q, smem_kv):
        """Create register and descriptor state before any dynamic skip guard."""
        cached_page_state = page_offset_window.init_read_state()
        smem_q.init_load_state()
        smem_kv.init_load_state()
        return cached_page_state

    def load_tma_body(page_offset_window, smem_q, smem_kv, cached_page_state):
        """Load Q once and load K/V tiles with the K-before-V cadence."""
        cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
            cached_page_state
        )

        # HEAD: Q is independent of page IDs.  Enqueue it before the TMA warp
        # refreshes its first coalesced 32-entry page-table window.
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
            page_offset_window.read_page_offset_window(
                cached_k_pages=cached_k_pages,
                cached_v_pages=cached_v_pages,
                cached_next_v_pages=cached_next_v_pages,
                cached_window_page=cached_window_page,
                init_v_cache=True,
            )
        )
        staged_kv_tma_load(
            smem_kv,
            iterations_qk,
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            is_v=False,
        )

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: cache K[n]/V[n] offsets, load K[n], then deferred V[n-1].
            cached_k_pages, cached_v_pages, cached_next_v_pages, cached_window_page = (
                page_offset_window.read_page_offset_window(
                    cached_k_pages=cached_k_pages,
                    cached_v_pages=cached_v_pages,
                    cached_next_v_pages=cached_next_v_pages,
                    cached_window_page=cached_window_page,
                )
            )
            # K sub-tiles then deferred V sub-tiles, each with its own local
            # sub-tile index; ``is_v`` tells the loader which path to take.
            staged_kv_tma_load(
                smem_kv,
                iterations_qk,
                cached_k_pages,
                cached_v_pages,
                cached_next_v_pages,
                is_v=False,
            )
            staged_kv_tma_load(
                smem_kv,
                iterations_pv,
                cached_k_pages,
                cached_v_pages,
                cached_next_v_pages,
                is_v=True,
            )

        # TAIL: V[last] uses the next-V offsets cached by the final wait.
        cached_k_pages, cached_v_pages, cached_next_v_pages = (
            page_offset_window.forward_page_ids(
                cached_k_pages=cached_k_pages,
                cached_v_pages=cached_v_pages,
                cached_next_v_pages=cached_next_v_pages,
            )
        )
        staged_kv_tma_load(
            smem_kv,
            iterations_pv,
            cached_k_pages,
            cached_v_pages,
            cached_next_v_pages,
            is_v=True,
            use_next_v_pages=True,
        )

    @schedule
    def load_tma_schedule(page_offset_window, smem_q, smem_kv, work_queue=None):
        """Capture one active TMA tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda cached_page_state: load_tma_body(
                page_offset_window,
                smem_q,
                smem_kv,
                cached_page_state,
            ),
            lambda: load_tma_prelude(page_offset_window, smem_q, smem_kv),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        captured_schedule = load_tma_schedule(page_offset_window, smem_q, smem_kv)
    else:
        captured_schedule = load_tma_schedule(
            page_offset_window,
            smem_q,
            smem_kv,
            work_queue,
        )

    src = [page_offset_window]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_kv],
        warp_idx=9,
        num_warps=1,
        schedule=captured_schedule,
        name="LoadTmaTask",
        **task_kwargs,
    )


def create_load_k_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 Q/K TMA task (warp 9).

    Q is loaded once before the loop. K uses one whole-tile pipeline stage per
    logical K tile and reads page offsets directly from GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def load_k_schedule(smem_q, smem_k, work_queue=None):
        """Load Q once and then publish one K stage per loop tile."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()

        with domain_loop(loop_start, loop_end, loop_step):
            smem_k.acquire()
            smem_k.tma_load_direct()
            smem_k.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        load_k_schedule(smem_q, smem_k)
        if work_queue is None
        else load_k_schedule(smem_q, smem_k, work_queue)
    )

    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_k],
        warp_idx=9,
        num_warps=1,
        schedule=schedule_result,
        name="LoadKTask",
        **task_kwargs,
    )


def create_load_v_task(
    smem_v: SmemVResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 10,
    num_warps: int = 1,
    **task_kwargs,
) -> Task:
    """Create the FP8 V TMA task.

    Dense FP8 uses W10. DSV4 sparse attention distributes Gather4 issues over
    W12-W15 while TaskWarpLeader aggregates their transaction-barrier arm.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def load_v_schedule(smem_v, work_queue=None):
        """Publish one V stage per logical K tile."""
        smem_v.init_load_state()
        with domain_loop(loop_start, loop_end, loop_step):
            smem_v.acquire()
            smem_v.tma_load_direct()
            smem_v.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        load_v_schedule(smem_v)
        if work_queue is None
        else load_v_schedule(smem_v, work_queue)
    )

    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[smem_v],
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=schedule_result,
        name="LoadVTask",
        **task_kwargs,
    )


def create_load_q_task(
    smem_q: SmemQResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 9,
    **task_kwargs,
) -> Task:
    """Create the single-warp Q TMA producer used by DSV4 compatibility mode."""
    # Q itself has no per-K-tile work, but the empty loop below carries the
    # dynamic-domain metadata required by the persistent Task runtime.
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def load_q_schedule(smem_q, work_queue=None):
        smem_q.init_load_state()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        with domain_loop(loop_start, loop_end, loop_step):
            pass
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        load_q_schedule(smem_q)
        if work_queue is None
        else load_q_schedule(smem_q, work_queue)
    )
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[smem_q],
        warp_idx=warp_idx,
        num_warps=1,
        schedule=schedule_result,
        name="LoadQTask",
        **task_kwargs,
    )


def create_load_page_offsets_dsv4_task(
    page_offset_ring: Dsv4PageOffsetRingResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 9,
    **task_kwargs,
) -> Task:
    """Create the W9 page-index producer, with the source pair contract explicit.

    TRTLLM-gen commits two identical ``int32[256]`` stages *per adjacent
    K128-tile pair*: one belongs to K and one to V.  Each stage contains both
    tiles in its two 128-entry halves.  This captured schedule deliberately
    exposes the two producer and K/V consumer segments by semantic label.
    ``MlaDsv4PairProducerTask`` and ``MlaDsv4PairLoadTask`` replay those
    segments with the source ``loopOffset += 2`` HEAD/LOOP/TAIL state machine;
    the legacy task class alone retains the per-K-tile diagnostic schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def page_offsets_schedule(page_offset_ring, work_queue=None):
        page_offset_ring.init_load_state()
        with domain_loop(loop_start, loop_end, loop_step):
            page_offset_ring.acquire()
            page_offset_ring.prefetch_pair()
            page_offset_ring.commit()
            page_offset_ring.acquire()
            page_offset_ring.prefetch_pair()
            page_offset_ring.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        page_offsets_schedule(page_offset_ring)
        if work_queue is None
        else page_offsets_schedule(page_offset_ring, work_queue)
    )
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[page_offset_ring],
        warp_idx=warp_idx,
        num_warps=1,
        schedule=schedule_result,
        name="LoadPageOffsetsTask",
        **task_kwargs,
    )


def create_load_dsv4_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    smem_v: SmemVResource,
    page_offset_ring: Dsv4PageOffsetRingResource,
    work_queue: MlaWorkQueue = None,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    warp_idx: int = 12,
    num_warps: int = 4,
    use_raw_k_gather: bool = True,
    use_raw_v_gather: bool = False,
    use_source_pair_ring: bool = False,
    **task_kwargs,
) -> Task:
    """Create DSV4's four-warp K/V Gather4 producer task.

    The generated DSV4 CSA kernel assigns W12--W15 one combined Q/K/V load
    task.  Only W12 issues the one-time Q TMA transaction, while all four
    warps issue K/V Gather4 transactions after W9 has published a page-index
    ring stage.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def load_dsv4_prelude(
        smem_q,
        smem_k,
        smem_v,
        page_offset_ring,
    ):
        """Initialize per-task register cursors outside the CLC skip guard."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_v.init_load_state()
        page_offset_ring.init_read_state()

    def load_dsv4_control(work_throttle):
        """Advance source's W12--W15 -> W10 edge once per WorkId."""

        work_throttle.try_acquire()
        work_throttle.acquire()
        work_throttle.commit()

    def load_dsv4_body(
        smem_q,
        smem_k,
        smem_v,
        page_offset_ring,
    ):
        """Publish Q once and K/V stages through the four-warp Gather4 group."""

        # Every W12--W15 lane participates in the source Q pipeline's
        # acquire/commit.  ``SmemQResource.tma_load`` itself gates the TMA
        # issue to W12, which preserves the source's one-warp issue pattern
        # without changing the pipeline's collective arrival protocol.
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()

        # The default schedule consumes one K and one V selector stage per
        # K128 tile.  The optional source-pair task replays these same entry
        # groups with source HEAD/LOOP/TAIL timing and holds each stage over
        # both pair halves; selecting the pair Gather4 method here ensures
        # its page-index half follows the runtime K-tile parity.
        with domain_loop(loop_start, loop_end, loop_step):
            page_offset_ring.wait()
            k_page_offset_stage = page_offset_ring.read_k_stage()
            smem_k.acquire()
            if cutlass.const_expr(use_raw_k_gather):
                if cutlass.const_expr(use_source_pair_ring):
                    smem_k.tma_load_from_page_ring_pair(
                        page_offset_stage=k_page_offset_stage
                    )
                else:
                    smem_k.tma_load_from_page_ring(
                        k_tile_delta=0, page_offset_stage=k_page_offset_stage
                    )
            else:
                # Keep the W9 ring wait/release above so this tests only the
                # Gather4 issuer/descriptor path, not a different DAG.
                smem_k.tma_load_sparse(k_tile_delta=0)
            smem_k.commit()
            page_offset_ring.release()

            page_offset_ring.wait()
            v_page_offset_stage = page_offset_ring.read_v_stage()
            smem_v.acquire()
            # ``use_raw_v_gather`` is a Python/CUTE compile-time constant
            # selected by the task builder from the BMM2 K dimension.  A
            # resource here is a schedule proxy, so its configuration is not
            # introspectable from this closure.
            if cutlass.const_expr(use_raw_v_gather):
                if cutlass.const_expr(use_source_pair_ring):
                    smem_v.tma_load_from_page_ring_pair(
                        page_offset_stage=v_page_offset_stage
                    )
                else:
                    smem_v.tma_load_from_page_ring(
                        k_tile_delta=0, page_offset_stage=v_page_offset_stage
                    )
            else:
                # This path is semantically correct with the current split
                # PV K64 schedule.  Raw source V writes one K128 operand and
                # is held behind the fused-K128 PV implementation.
                smem_v.tma_load_sparse(k_tile_delta=0)
            smem_v.commit()
            page_offset_ring.release()

    @schedule
    def load_dsv4_schedule(
        smem_q,
        smem_k,
        smem_v,
        page_offset_ring,
        work_queue=None,
        work_throttle=None,
    ):
        """Capture active Q/K/V work and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda _load_state: load_dsv4_body(
                smem_q,
                smem_k,
                smem_v,
                page_offset_ring,
            ),
            lambda: load_dsv4_prelude(
                smem_q,
                smem_k,
                smem_v,
                page_offset_ring,
            ),
            (
                None
                if work_throttle is None
                else lambda: load_dsv4_control(work_throttle)
            ),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        schedule_result = load_dsv4_schedule(smem_q, smem_k, smem_v, page_offset_ring)
    elif work_throttle is None:
        schedule_result = load_dsv4_schedule(
            smem_q, smem_k, smem_v, page_offset_ring, work_queue
        )
    else:
        schedule_result = load_dsv4_schedule(
            smem_q, smem_k, smem_v, page_offset_ring, work_queue, work_throttle
        )
    src = [page_offset_ring]
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_k, smem_v]
    if work_throttle is not None:
        dst.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=schedule_result,
        name="LoadDsv4Task",
        **task_kwargs,
    )


def create_mma_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_s: TmemSResource,
    tmem_o: TmemOResource,
    iterations_qk: int = 9,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the MMA task (warp 8, 1 warp, 96 regs).

    domain_start=1: HEAD handles first QK MMA, LOOP handles PV+QK pairs.
    run_only_on_cta_id=0 keeps the schedule on the CTA-pair leader, matching
    the 2CTA UMMA contract.

    HEAD: consume Q, QK MMA for k-tile[0] -> S[0].
    LOOP: PV MMA for k-tile[n-1] -> O[n-1], then QK MMA for k-tile[n] -> S[n].
    TAIL: PV MMA for last k-tile -> O[last], release Q.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def mma_body(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_throttle=None,
    ):
        """Run QK one tile ahead of PV and keep UMMA on the leader CTA."""
        # HEAD: wait Q once and compute S[0] from K[0].
        smem_q.wait()
        if work_throttle is not None:
            # The leader MMA reaching Q proves that this cluster has started
            # the current tile.  Permit the scheduler to prepare one more work
            # ID without relying on CTA-private resource-ownership fields.
            work_throttle.try_acquire()
            work_throttle.acquire()
            work_throttle.commit()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: compute S[n] before PV[n-1]. This gives softmax the QK
            # instruction window to produce P[n-1] before PV consumes it.
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)
            staged_pv_mma(
                smem_kv,
                smem_p,
                tmem_o,
                iterations_pv,
            )

        # TAIL: finish PV[last], then release the Q descriptor stage.
        staged_pv_mma(
            smem_kv,
            smem_p,
            tmem_o,
            iterations_pv,
            is_tail=True,
        )
        smem_q.release()

    @schedule
    def mma_schedule(
        smem_q,
        smem_kv,
        smem_p,
        tmem_s,
        tmem_o,
        work_queue=None,
        work_throttle=None,
    ):
        """Capture one active MMA tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda: mma_body(
                smem_q,
                smem_kv,
                smem_p,
                tmem_s,
                tmem_o,
                work_throttle,
            ),
            use_clc_dynamic=use_clc_dynamic,
        )

    if work_queue is None:
        captured_schedule = mma_schedule(smem_q, smem_kv, smem_p, tmem_s, tmem_o)
    elif work_throttle is None:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
        )
    else:
        captured_schedule = mma_schedule(
            smem_q,
            smem_kv,
            smem_p,
            tmem_s,
            tmem_o,
            work_queue,
            work_throttle,
        )

    src = [smem_q, smem_kv, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    dst = [tmem_s, tmem_o]
    if work_throttle is not None:
        dst.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=8,
        num_warps=1,
        schedule=captured_schedule,
        name="MmaTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_qk_task(
    smem_q: SmemQResource,
    smem_kv: SmemKVResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 9,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK-only MMA task (warp 8).

    QK stays one k-tile ahead of PV. Keeping QK and PV on separate warps avoids
    serializing the two UMMA issue streams while preserving the existing K-before-V
    TMA cadence and one-softmax schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue=None):
        """Consume Q/K stages and publish S for every k-tile."""
        smem_q.wait()
        smem_q.q_desc()
        staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma(smem_kv, tmem_s, iterations_qk)

        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_schedule(smem_q, smem_kv, tmem_s)
        if work_queue is None
        else mma_qk_schedule(smem_q, smem_kv, tmem_s, work_queue)
    )

    src = [smem_q, smem_kv]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaQkTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_pv_task(
    smem_kv: SmemKVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 8,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV-only MMA task (warp 11).

    The PV task consumes P[n-1]/V[n-1] while QK produces S[n], then handles the
    final P/V tile in TAIL. This matches the existing correction schedule.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)

    @schedule
    def mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue=None):
        """Consume delayed V and P stages and accumulate O."""
        with domain_loop(loop_start, loop_end, loop_step):
            staged_pv_mma(smem_kv, smem_p, tmem_o, iterations_pv)

        staged_pv_mma(
            smem_kv,
            smem_p,
            tmem_o,
            iterations_pv,
            is_tail=True,
        )
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_schedule(smem_kv, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_schedule(smem_kv, smem_p, tmem_o, work_queue)
    )

    src = [smem_kv, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_o],
        warp_idx=11,
        num_warps=1,
        schedule=schedule_result,
        name="MmaPvTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_qk_direct_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    tmem_s: TmemSResource,
    iterations_qk: int = 5,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 QK task with one K stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue=None):
        """Consume Q once and publish one S stage per K tile."""
        smem_q.wait()
        smem_q.q_desc()
        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk_mma_k_tile(smem_k, tmem_s, iterations_qk)
        smem_q.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_qk_direct_schedule(smem_q, smem_k, tmem_s)
        if work_queue is None
        else mma_qk_direct_schedule(smem_q, smem_k, tmem_s, work_queue)
    )

    src = [smem_q, smem_k]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_s],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaQkTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_pv_direct_task(
    smem_v: SmemVResource,
    smem_p: SmemPResource,
    tmem_o: TmemOResource,
    iterations_pv: int = 4,
    iterations_pv_k: int = 2,
    iterations_pv_n: int = 2,
    per_n_o_pipeline: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 PV task with one V stage per domain-loop iteration."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue=None):
        """Consume one P/V pair and publish one O stage per K tile."""
        with domain_loop(loop_start, loop_end, loop_step):
            if per_n_o_pipeline:
                staged_pv_mma_v_tile_per_n(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations_pv_k=iterations_pv_k,
                    iterations_pv_n=iterations_pv_n,
                )
            else:
                staged_pv_mma_v_tile(smem_v, smem_p, tmem_o, iterations_pv)
        work_queue_tail(work_queue, advance_label="advance_tile")

    schedule_result = (
        mma_pv_direct_schedule(smem_v, smem_p, tmem_o)
        if work_queue is None
        else mma_pv_direct_schedule(smem_v, smem_p, tmem_o, work_queue)
    )

    src = [smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_o],
        warp_idx=11,
        num_warps=1,
        schedule=schedule_result,
        name="MmaPvTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )


def create_mma_dsv4_fused_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    smem_v: SmemVResource,
    smem_p: SmemPResource,
    tmem_s: TmemSResource,
    tmem_o: TmemOResource,
    iterations_qk: int = 4,
    iterations_pv_k: int = 1,
    iterations_pv_n: int = 2,
    per_n_o_pipeline: bool = True,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    source_s_handoff: bool = False,
    source_direct_p: bool = False,
    source_tmem_lifecycle: bool = False,
    **task_kwargs,
) -> Task:
    """Create source-style DSV4 W8: QK is exactly one KV tile ahead of PV.

    HEAD publishes S[0].  LOOP starts at one, so it publishes S[n] into the
    second S-pipeline state before consuming P[n-1]/V[n-1].  Keeping
    ``domain_start=1`` is essential: it gives HEAD and the first LOOP body
    distinct producer pipeline states, matching the generated kernel's two
    explicit S pre-acquires.  TAIL consumes the final P/V pair.  This frees
    W11 for the source padding role.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)
    if source_direct_p and not source_s_handoff:
        raise ValueError("source direct-P requires the source S handoff")
    if source_tmem_lifecycle and not source_s_handoff:
        raise ValueError("source TMEM lifecycle requires the source S handoff")

    def mma_dsv4_fused_body(smem_q, smem_k, smem_v, smem_p, tmem_s, tmem_o):
        """Run the source HEAD -> (QK[n], PV[n-1]) -> TAIL FSM on W8."""
        smem_q.wait()
        smem_q.q_desc()
        staged_qk = (
            staged_qk_mma_k_tile_from_acquired_s
            if source_s_handoff
            else staged_qk_mma_k_tile
        )
        staged_qk(smem_k, tmem_s, iterations_qk)

        with domain_loop(loop_start, loop_end, loop_step):
            staged_qk(smem_k, tmem_s, iterations_qk)
            if source_s_handoff:
                # Source QK[n] commits first, then this acquire observes the
                # prior softmax P store through S[n-1]'s consumer release.
                tmem_s.acquire()
            if source_direct_p and per_n_o_pipeline:
                staged_pv_mma_v_tile_per_n_source_direct_p(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations_pv_k=iterations_pv_k,
                    iterations_pv_n=iterations_pv_n,
                )
            elif source_direct_p:
                staged_pv_mma_v_tile_source_direct_p(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations=iterations_pv_k * iterations_pv_n,
                )
            elif per_n_o_pipeline:
                staged_pv_mma_v_tile_per_n(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations_pv_k=iterations_pv_k,
                    iterations_pv_n=iterations_pv_n,
                )
            else:
                staged_pv_mma_v_tile(
                    smem_v,
                    smem_p,
                    tmem_o,
                    iterations=iterations_pv_k * iterations_pv_n,
                )

        if source_s_handoff:
            # With the source split-cursor pipeline, acquire already runs two
            # stages ahead of QK commit.  Waiting on its current state is
            # therefore equivalent to generated
            # makePipelineState(tmemS0ProdState, 1), and TaskManager carries
            # the advanced acquire cursor through the captured TAIL region.
            tmem_s.acquire()
        if source_direct_p and per_n_o_pipeline:
            staged_pv_mma_v_tile_per_n_source_direct_p(
                smem_v,
                smem_p,
                tmem_o,
                iterations_pv_k=iterations_pv_k,
                iterations_pv_n=iterations_pv_n,
                is_tail=True,
            )
        elif source_direct_p:
            staged_pv_mma_v_tile_source_direct_p(
                smem_v,
                smem_p,
                tmem_o,
                iterations=iterations_pv_k * iterations_pv_n,
                is_tail=True,
            )
        elif per_n_o_pipeline:
            staged_pv_mma_v_tile_per_n(
                smem_v,
                smem_p,
                tmem_o,
                iterations_pv_k=iterations_pv_k,
                iterations_pv_n=iterations_pv_n,
                is_tail=True,
            )
        else:
            staged_pv_mma_v_tile(
                smem_v,
                smem_p,
                tmem_o,
                iterations=iterations_pv_k * iterations_pv_n,
                is_tail=True,
            )
        smem_q.release()

    @schedule
    def mma_dsv4_fused_schedule(
        smem_q, smem_k, smem_v, smem_p, tmem_s, tmem_o, work_queue=None
    ):
        """Capture one active fused-MMA tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda: mma_dsv4_fused_body(
                smem_q,
                smem_k,
                smem_v,
                smem_p,
                tmem_s,
                tmem_o,
            ),
            use_clc_dynamic=use_clc_dynamic,
        )

    schedule_result = (
        mma_dsv4_fused_schedule(smem_q, smem_k, smem_v, smem_p, tmem_s, tmem_o)
        if work_queue is None
        else mma_dsv4_fused_schedule(
            smem_q, smem_k, smem_v, smem_p, tmem_s, tmem_o, work_queue
        )
    )
    src = [smem_q, smem_k, smem_v, smem_p]
    if work_queue is not None:
        src.append(work_queue)
    selected_task_class = task_class
    if source_s_handoff:
        selected_task_class = (
            MlaDsv4SourceSHandoffClcTask
            if issubclass(task_class, MlaClcTask)
            else MlaDsv4SourceSHandoffTask
        )
    source_task_kwargs = (
        {"source_tmem_lifecycle": source_tmem_lifecycle} if source_s_handoff else {}
    )
    return selected_task_class(
        src_resources=src,
        dst_resources=[tmem_s, tmem_o],
        warp_idx=8,
        num_warps=1,
        schedule=schedule_result,
        name="MmaDsv4FusedTask",
        run_only_on_cta_id=0,
        **source_task_kwargs,
        **task_kwargs,
    )


def create_softmax_task(
    tmem_s: TmemSResource,
    tmem_corr: TmemCorrResource,
    smem_p: SmemPResource,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 0,
    name: str = "SoftmaxTask",
    softmax_group_id: int = 0,
    release_s_after_p: bool = False,
    source_direct_p: bool = False,
    source_early_corr: bool = False,
    source_softmax_pipeline: bool = False,
    **task_kwargs,
) -> Task:
    """Create the Softmax task (warps 0-3, 4 warps, 192 regs).

    domain_start=0: processes all k_tile_count S tiles.

    LOOP: consume S, compute softmax, produce correction factors + P.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)
    if source_early_corr and softmax_group_id != 0:
        raise ValueError("source early correction supports the DSV4 softmax group only")

    def softmax_prelude(tmem_s):
        """Create softmax register arrays before the dynamic skip guard."""
        init_softmax_state = (
            tmem_s.init_softmax_state_odd
            if softmax_group_id == 1
            else tmem_s.init_softmax_state
        )
        return init_softmax_state()

    def softmax_body(tmem_s, tmem_corr, smem_p, softmax_state):
        """Consume S tiles, materialize P, and publish correction factors."""
        load_s = tmem_s.load_s_odd if softmax_group_id == 1 else tmem_s.load_s
        finish_softmax = (
            tmem_s.finish_softmax_odd
            if softmax_group_id == 1
            else tmem_s.finish_softmax
        )
        finish_row_sum = (
            tmem_s.finish_row_sum_odd
            if softmax_group_id == 1
            else tmem_s.finish_row_sum
        )
        store_corr = (
            tmem_corr.store_corr_odd if softmax_group_id == 1 else tmem_corr.store_corr
        )
        store_p = smem_p.store_p_odd if softmax_group_id == 1 else smem_p.store_p
        (
            qk_acc_regs,
            row_max,
            row_sum,
            row_sum_out,
            row_max_new,
            correction_factor_out,
            no_correction_out,
        ) = softmax_state

        with domain_loop(loop_start, loop_end, loop_step):
            tmem_s.wait()
            if softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = load_s(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = load_s(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            # The generated DSV4 CSA schedule holds S until the softmax warp
            # has stored and fenced P. Its next S producer acquire is then the
            # P-ready dependency. Keep the legacy release position for all
            # other schedules while the direct-P port remains an isolator.
            if not release_s_after_p:
                tmem_s.release()
            if source_early_corr:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = tmem_s.finish_softmax_max(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
                # Generated DSV4 publishes old/new max before P exp/store so
                # Correction[n] can overlap Softmax's P[n] critical path.
                tmem_corr.acquire()
                tmem_corr.store_source_early(
                    row_max_old=row_max,
                    row_max_new=row_max_new,
                )
                tmem_corr.commit()
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = tmem_s.materialize_softmax_p(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            elif softmax_group_id == 1:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = finish_softmax(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_max_odd=row_max,
                    row_sum_odd=row_sum,
                    row_sum_out_odd=row_sum_out,
                    row_max_new_odd=row_max_new,
                    correction_factor_out_odd=correction_factor_out,
                    no_correction_out_odd=no_correction_out,
                )
            else:
                (
                    qk_acc_regs,
                    row_max,
                    row_sum,
                    row_sum_out,
                    row_max_new,
                    correction_factor_out,
                    no_correction_out,
                ) = finish_softmax(
                    qk_acc_regs=qk_acc_regs,
                    row_max=row_max,
                    row_sum=row_sum,
                    row_sum_out=row_sum_out,
                    row_max_new=row_max_new,
                    correction_factor_out=correction_factor_out,
                    no_correction_out=no_correction_out,
                )
            if source_softmax_pipeline:
                # The source software pipeline delays FP8 conversion by 16
                # MUFU elements, so the packed registers must stay local to
                # TmemS.materialize_softmax_p until its direct SMEM store.
                # That method also emits the first async-shared view fence.
                assert source_direct_p
                smem_p.mark_p_source_pipelined_store()
            elif source_direct_p:
                # Source uses S release as the P-ready signal. Its P buffer
                # is selected by logical K-tile parity, with no P mbarrier.
                smem_p.store_p_source_direct(qk_acc_regs=qk_acc_regs)
            else:
                smem_p.acquire()
                if softmax_group_id == 1:
                    store_p(qk_acc_regs_odd=qk_acc_regs)
                else:
                    store_p(qk_acc_regs=qk_acc_regs)
                smem_p.commit()
            if release_s_after_p:
                tmem_s.release()
            if softmax_group_id == 1:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs_odd=qk_acc_regs,
                    row_sum_odd=row_sum,
                    correction_factor_out_odd=correction_factor_out,
                )
            else:
                row_sum, row_sum_out = finish_row_sum(
                    qk_acc_regs=qk_acc_regs,
                    row_sum=row_sum,
                    correction_factor_out=correction_factor_out,
                )
            if not source_early_corr:
                tmem_corr.acquire()
                if softmax_group_id == 1:
                    store_corr(
                        row_sum_out_odd=row_sum_out,
                        row_max_new_odd=row_max_new,
                        correction_factor_out_odd=correction_factor_out,
                        no_correction_out_odd=no_correction_out,
                    )
                else:
                    store_corr(
                        row_sum_out=row_sum_out,
                        row_max_new=row_max_new,
                        correction_factor_out=correction_factor_out,
                        no_correction_out=no_correction_out,
                    )
                tmem_corr.commit()

        if source_early_corr:
            # One extra token terminates the K early-message stream and makes
            # final row stats available only after the online sum is complete.
            tmem_corr.acquire()
            tmem_corr.store_source_final(row_sum=row_sum, row_max=row_max)
            tmem_corr.commit()

    @schedule
    def softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue=None):
        """Capture one active softmax tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda softmax_state: softmax_body(
                tmem_s,
                tmem_corr,
                smem_p,
                softmax_state,
            ),
            lambda: softmax_prelude(tmem_s),
            use_clc_dynamic=use_clc_dynamic,
        )

    schedule_result = (
        softmax_schedule(tmem_s, tmem_corr, smem_p)
        if work_queue is None
        else softmax_schedule(tmem_s, tmem_corr, smem_p, work_queue)
    )

    src = [tmem_s]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[tmem_corr, smem_p],
        warp_idx=warp_idx,
        num_warps=4,
        schedule=schedule_result,
        name=name,
        **task_kwargs,
    )


def create_correction_task(
    tmem_corr: TmemCorrResource,
    tmem_o: TmemOResource,
    gmem_o: GmemOResource,
    iterations_pv_n: int = 1,
    per_n_o_pipeline: bool = False,
    source_early_corr: bool = False,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the Correction task (warps 4-7, 4 warps, 208 regs).

    domain_start=1: HEAD handles initial correction (no O yet), LOOP
    handles correction+O pairs, TAIL handles final O + epilogue.

    TMEM visibility is ensured by kernel-level named barrier sync before
    task_manager.run(), so o_init pipeline is no longer needed here.

    HEAD: consume Corr[0] (initial max/sum, no O rescaling).
    LOOP: consume Corr[n] + O[n-1], rescale accumulated O.
    TAIL: consume O[last], final epilogue store O + LSE to GMEM.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 1)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    def correction_prelude(tmem_corr):
        """Create correction and epilogue register state before skip guards."""
        return tmem_corr.init_load_state()

    def correction_body(tmem_corr, tmem_o, gmem_o, correction_state):
        """Apply online-softmax correction and store the final O/LSE result."""
        row_sum, row_max, correction_factor, no_correction = correction_state

        # HEAD: consume the first correction factors. There is no O tile yet.
        tmem_corr.wait()
        if source_early_corr:
            tmem_corr.consume_source_head()
        else:
            row_sum, row_max, correction_factor, no_correction = tmem_corr.load_corr()
        tmem_corr.release()

        with domain_loop(loop_start, loop_end, loop_step):
            # LOOP: consume Corr[n] and rescale O[n-1].
            tmem_corr.wait()
            if source_early_corr:
                row_sum, row_max, correction_factor, no_correction = (
                    tmem_corr.load_source_early()
                )
            else:
                row_sum, row_max, correction_factor, no_correction = (
                    tmem_corr.load_corr()
                )
            tmem_corr.release()
            if per_n_o_pipeline:
                for iter_n in range(iterations_pv_n):
                    tmem_o.wait()
                    tmem_o.rescale_o_slice(
                        correction_factor=correction_factor,
                        no_correction=no_correction,
                        iter_n=iter_n,
                    )
                    tmem_o.release()
            else:
                tmem_o.wait()
                tmem_o.rescale_o(
                    correction_factor=correction_factor,
                    no_correction=no_correction,
                )
                tmem_o.release()

        if source_early_corr:
            # Source sends a distinct terminal sum/max token after all K
            # early-max tokens; it is the epilogue's normalization state.
            tmem_corr.wait()
            row_sum, row_max, correction_factor, no_correction = (
                tmem_corr.load_source_final()
            )
            tmem_corr.release()

        # TAIL: consume final O. Do not call rescale_o here; the correction was
        # already applied in LOOP and the last loop correction value would be
        # stale for a second application. epilogue_store also writes LSE.
        if per_n_o_pipeline:
            epilogue_row_sum, epilogue_row_max = (
                tmem_corr.prepare_epilogue_slice_store()
            )
            for iter_n in range(iterations_pv_n):
                tmem_o.wait()
                gmem_o.epilogue_store_slice(
                    row_sum=epilogue_row_sum,
                    row_max=epilogue_row_max,
                    iter_n=iter_n,
                )
                tmem_o.release()
        else:
            tmem_o.wait()
            gmem_o.epilogue_store()
            tmem_o.release()

    @schedule
    def correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue=None):
        """Capture one active correction tile and unconditional queue progress."""

        _capture_clc_work_tile_body(
            work_queue,
            lambda correction_state: correction_body(
                tmem_corr,
                tmem_o,
                gmem_o,
                correction_state,
            ),
            lambda: correction_prelude(tmem_corr),
            use_clc_dynamic=use_clc_dynamic,
        )

    captured_schedule = (
        correction_schedule(tmem_corr, tmem_o, gmem_o)
        if work_queue is None
        else correction_schedule(tmem_corr, tmem_o, gmem_o, work_queue)
    )

    src = [tmem_corr, tmem_o]
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[gmem_o],
        warp_idx=4,
        num_warps=4,
        schedule=captured_schedule,
        name="CorrectionTask",
        **task_kwargs,
    )


def create_padding_task(
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    warp_idx: int = 11,
    num_warps: int = 1,
    **task_kwargs,
) -> Task:
    """Create a padding task for unused warps in producer warpgroup 2.

    Empty task for warp-group alignment.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)
    use_clc_dynamic = bool(work_queue is not None and work_queue.use_clc_dynamic)

    @schedule
    def padding_schedule(work_queue=None):
        """Reserve unused producer warps without issuing kernel work."""

        def padding_body():
            with domain_loop(loop_start, loop_end, loop_step):
                pass

        _capture_clc_work_tile_body(
            work_queue,
            padding_body,
            use_clc_dynamic=use_clc_dynamic,
        )

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[],
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        name="PaddingTask",
        **task_kwargs,
    )


def create_scheduler_task(
    work_queue: MlaWorkQueue,
    work_throttle: WorkThrottleBarrierResource = None,
    task_class: type = MlaTask,
    warp_idx: int = 11,
    throttle_per_workid: bool = False,
    **task_kwargs,
) -> Task:
    """Create the cluster-wide CLC scheduler task."""

    @schedule
    def scheduler_schedule(work_queue, work_throttle=None):
        """Fetch and distribute the next logical MLA cluster tile."""

        with work_tile_loop(
            work_queue,
            skip_if=MlaWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            if cutlass.const_expr(work_throttle is not None and throttle_per_workid):
                work_throttle.wait()
                work_throttle.release()
            with work_tiles.skippable():
                with domain_loop(0, 0, 1):
                    pass
                if cutlass.const_expr(
                    work_throttle is not None and not throttle_per_workid
                ):
                    work_throttle.wait()
                    work_throttle.release()
            work_queue.acquire()
            work_queue.fetch_work_tile()
            work_queue.commit()
            work_queue_tail(work_queue, advance_label="advance_tile")

    captured_schedule = (
        scheduler_schedule(work_queue)
        if work_throttle is None
        else scheduler_schedule(work_queue, work_throttle)
    )
    src = [work_queue]
    if work_throttle is not None:
        src.append(work_throttle)
    return task_class(
        src_resources=src,
        dst_resources=[work_queue],
        warp_idx=warp_idx,
        num_warps=1,
        schedule=captured_schedule,
        name="SchedulerTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )
