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

"""Task base classes, work queues and schedule helpers shared by the throughput 2CTA MLA decode kernel tasks."""

import cutlass.cute as cute
import cutlass
from collections.abc import Callable
from cutlass.experimental.task_scheduling.memory import ResourceContext
from cutlass.experimental.task_scheduling.resources import StageInfo
from cutlass.experimental.task_scheduling.schedule_builder import work_tile_loop
from cutlass.experimental.task_scheduling.task import Task
from ......device.task import task_context_kwargs
from ...device.schedule import work_queue_tail
from ..work_queue import MlaWorkQueue


class MlaClcTask(Task):
    """Stock Task persistent loop with an MLA-specific dynamic K domain."""

    @cute.jit
    def get_domain(self, tile_coord):
        """Recompute the loop bound required by the stock Task public API."""

        assert isinstance(self.work_queue, MlaWorkQueue)
        return self.work_queue.k_tile_count_for_tile(tile_coord)


class MlaTask(Task):
    """Task subclass that recomputes MLA k-domain per persistent work tile."""

    @cute.jit
    def _run_task_body_persistent(
        self,
        context: ResourceContext | None = None,
    ) -> None:
        """Schedule one or more persistent work tiles for this task instance."""

        params = self.work_queue.tile_sched_params

        if cutlass.const_expr(not params.is_persistent):
            work_tile = self.work_queue.work_tile_from_block_idx(cute.arch.block_idx())
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
            self._run_pre_work_loop_entries(work_tile, **task_context_kwargs(context))
            # Runtime K/Q metadata can make a statically launched split empty.
            # Keep the CTA on the ordinary initialized-pipeline path, but skip
            # its captured HEAD/LOOP/TAIL data work when the domain is zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self.run_one_mla_work_tile(work_tile, context)
            self._run_post_work_loop_entries(work_tile, **task_context_kwargs(context))
            self.drain_mla_work_tile_tails()
            return

        current_work_linear_idx = cute.arch.block_idx()[0]
        num_blocks = (
            params.cluster_shape_mnk[0]
            * params.problem_shape_s
            * params.problem_shape_b
            * params.split_kv
        )
        work_tile = self.work_queue.work_tile_from_linear_idx(current_work_linear_idx)
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

        self._run_pre_work_loop_entries(work_tile, **task_context_kwargs(context))
        while current_work_linear_idx < num_blocks:
            work_tile.update_from(
                self.work_queue.work_tile_from_linear_idx(current_work_linear_idx)
            )
            self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)

            # Variable K and causal Q visibility can leave individual logical
            # splits empty. Skip them and continue grid-striding rather than
            # running a captured HEAD/TAIL sequence with domain zero.
            if work_tile.k_tile_count > cutlass.Int32(0):
                self.run_one_mla_work_tile(work_tile, context)

            # Each warp branch advances from the same scalar tile id, keeping
            # the persistent loop state compact across task bodies.
            current_work_linear_idx += cute.size(cute.arch.grid_dim())
            # self.dummy keeps the captured persistent loop body live even when
            # a specialized task instance has no visible local result.
            self.dummy = True
        work_tile.update_from(
            self.work_queue.work_tile_from_linear_idx(current_work_linear_idx)
        )
        self.work_queue._set_consumer_var_from_ts("work_tile", work_tile)
        self._run_post_work_loop_entries(work_tile, **task_context_kwargs(context))
        self.drain_mla_work_tile_tails()

    @cute.jit
    def drain_mla_work_tile_tails(self) -> None:
        """Drain producer tails after a persistent work-tile body completes."""

        for resource in self.dst_resources:
            if cutlass.const_expr(
                resource.pipeline_config is not None
                and resource is not self.work_queue
                and not self._is_fork_secondary(resource)
            ):
                if cutlass.const_expr(
                    resource.pipeline_config.producer_acquire_interleave_stride > 1
                    or resource.pipeline_config.producer_commit_interleave_stride > 1
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
    def run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run a persistent work tile using the cached split-KV domain."""

        # WorkQueue decomposes the MLA persistent tile and caches the K-domain.
        # Keep task bodies on that cached value so page-offset/TMA/MMA/softmax
        # paths do not each rebuild the same split-KV arithmetic.
        self.domain = work_tile.k_tile_count
        self._run_task_body_impl(work_tile, **task_context_kwargs(context))


class MlaInterleavedTask(MlaTask):
    """Persistent task that carries its interleave lane across work tiles."""

    def __init__(self, *args, **kwargs) -> None:
        """Create staged parity and index-remapping state for this task."""

        super().__init__(*args, **kwargs)
        self.cumulative_k_parity = cutlass.Int32(0)
        self.mapped_domain_start = cutlass.Int32(self.domain_start)
        self.actual_domain = cutlass.Int32(0)
        self.fixed_loop_end = cutlass.Int32(self.domain_start)

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
            **task_context_kwargs(context),
        )
        actual_loop_offset = self.mapped_domain_start + (
            cutlass.Int32(base_info.loop_offset) - cutlass.Int32(self.domain_start)
        )
        return StageInfo(
            loop_offset=actual_loop_offset,
            loop_start=self.mapped_domain_start,
            loop_end=self.actual_domain,
            loop_step=base_info.loop_step,
            stage_idx=base_info.stage_idx,
            label=base_info.label,
            barrier=base_info.barrier,
            work_tile=base_info.work_tile,
            num_active_stages=base_info.num_active_stages,
            context=base_info.context,
            task_cache=base_info.task_cache,
        )

    @cute.jit
    def run_one_mla_work_tile(
        self,
        work_tile,
        context: ResourceContext | None = None,
    ) -> None:
        """Run one tile by mapping its local offsets onto this fixed lane."""

        fixed_lane = cutlass.Int32(self.domain_start)
        self.actual_domain = work_tile.k_tile_count
        self.mapped_domain_start = (
            fixed_lane + self.cumulative_k_parity
        ) % cutlass.Int32(2)
        lane_iterations = (
            self.actual_domain - self.mapped_domain_start + cutlass.Int32(1)
        ) // cutlass.Int32(2)
        self.fixed_loop_end = fixed_lane + lane_iterations * cutlass.Int32(2)
        self.domain = self.fixed_loop_end
        self._run_task_body_impl(work_tile, **task_context_kwargs(context))
        self.cumulative_k_parity = (
            self.cumulative_k_parity + self.actual_domain
        ) % cutlass.Int32(2)


def capture_clc_work_tile_body(
    work_queue,
    body: Callable[..., None],
    non_skippable_prelude: Callable[[], object] | None = None,
    *,
    use_clc_dynamic: bool = False,
) -> None:
    """Capture one MLA tile with data work skippable and WQ progress mandatory.

    Pure register-state initializers may run in ``non_skippable_prelude`` so
    values they create dominate the separately guarded HEAD, LOOP, and TAIL
    regions emitted by the stock skipped-tile executor.  The prelude must not
    issue memory operations or advance pipeline state.
    """

    def run_body():
        prelude_state = (
            non_skippable_prelude() if non_skippable_prelude is not None else None
        )
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
            with work_tiles.skippable():
                if non_skippable_prelude is None:
                    body()
                else:
                    body(prelude_state)
            work_queue_tail(work_queue, advance_label="advance_tile")
        return

    run_body()
    work_queue_tail(work_queue, advance_label="advance_tile")


def fixed_lane_work_tile_bounds(fixed_lane, cumulative_k_parity, actual_domain):
    """Model fixed-lane work-tile bounds for CPU contract tests.

    Generated code keeps this arithmetic inline because returning staged DSL
    values through a plain-Python tuple breaks staged-frontend state threading.
    """

    mapped_start = (fixed_lane + cumulative_k_parity) % 2
    lane_iterations = (actual_domain - mapped_start + 1) // 2
    fixed_loop_end = fixed_lane + lane_iterations * 2
    return mapped_start, fixed_loop_end
