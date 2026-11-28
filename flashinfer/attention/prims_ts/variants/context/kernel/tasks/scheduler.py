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

"""Scheduler and padding tasks of the context kernel."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_padding_task(
    work_queue: WorkQueue | None,
    warp_idx: int = 15,
    num_warps: int = 1,
    num_registers: int = 32,
    name: str = "PaddingTask",
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp padding task (warp 15 in the D128 schedule).

    Required in ALL modes (persistent and non-persistent) because
    ``setmaxnreg.sync`` requires every warp in the warp group to
    participate. In D128, warps 12-15 form warp group 3; without the padding
    task its final warp never calls ``setmaxregister``, deadlocking the group.

    In persistent mode the task also consumes work_queue tiles so that
    the auxiliary warp participates in the persistent outer loop.

    In CLC dynamic mode, the padding task is replaced by a scheduler task
    (see ``create_scheduler_task``).
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)
    src = src_resources(work_queue=work_queue)

    @schedule
    def padding_schedule(wq: WorkQueue | None = None) -> None:
        """Captured schedule for warp-group register participation."""
        with (
            work_tile_schedule_loop(wq, skip_if=skip_work_tile_if),
            domain_loop(loop_start, loop_end, loop_step),
        ):
            pass

    captured_schedule = schedule_with_work_queue(
        padding_schedule, work_queue=work_queue
    )
    return task_class(
        src_resources=src,
        dst_resources=[],
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        num_registers=num_registers,
        name=name,
        **task_kwargs,
    )


def create_scheduler_task(
    work_queue: WorkQueue,
    warp_idx: int = 15,
    num_registers: int = 32,
    task_class: type[Task] = Task,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp CLC scheduler task (warp 15 in D128).

    Replaces the padding task in CLC dynamic persistent mode.
    Issues CLC tile-fetch queries (producer side) and participates in
    the persistent outer loop.  Still satisfies the ``setmaxnreg.sync``
    requirement for the final warp group.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)

    @schedule
    def scheduler_schedule(wq: WorkQueue) -> None:
        """Captured schedule for CLC work-tile fetches."""
        with work_tile_schedule_loop(wq):
            with domain_loop(loop_start, loop_end, loop_step):
                pass
            # Producer side: issue CLC tile-fetch query.
            wq.acquire()
            wq.fetch_work_tile()
            wq.commit()

    captured_schedule = scheduler_schedule(work_queue)
    return task_class(
        src_resources=[work_queue],
        dst_resources=[work_queue],
        warp_idx=warp_idx,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="SchedulerTask",
        **task_kwargs,
    )
