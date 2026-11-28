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

"""Scheduler and padding tasks of the throughput 2CTA MLA decode kernel."""

from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import captured_loop_bounds, work_queue_tail
from ..resources import WorkThrottleBarrierResource
from .common import MlaTask, capture_clc_work_tile_body
from ..work_queue import MlaWorkQueue


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

        capture_clc_work_tile_body(
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
    **task_kwargs,
) -> Task:
    """Create the BF16 cluster-wide CLC scheduler on warp 11."""

    @schedule
    def scheduler_schedule(work_queue, work_throttle=None):
        """Fetch and distribute the next logical MLA cluster tile."""

        with work_tile_loop(
            work_queue,
            skip_if=MlaWorkQueue.skip_work_tile_if,
        ) as work_tiles:
            with work_tiles.skippable():
                with domain_loop(0, 0, 1):
                    pass
                if work_throttle is not None:
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
        warp_idx=11,
        num_warps=1,
        schedule=captured_schedule,
        name="SchedulerTask",
        run_only_on_cta_id=0,
        **task_kwargs,
    )
