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

"""Scheduler and padding tasks of the latency 1CTA MLA decode kernel."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from ......device.schedule import schedule_token_throttle_tail
from ...device.schedule import runtime_work_tile_skip_if, work_tile_schedule_loop
from .common import MlaDecodeTask


def create_padding_task(
    cfg,
    work_queue: WorkQueue | None = None,
    *,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the padding task used to reserve otherwise idle warps."""

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def padding_schedule(work_queue=None):
        """Captured no-op schedule with optional persistent work-queue tail."""
        # Keep the task-scheduling and domain-loop scopes distinct for CuTe DSL.
        with work_tile_schedule_loop(  # noqa: SIM117
            work_queue, skip_if=work_tile_skip_if
        ):
            with domain_loop(0, 1, 1):
                pass

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
    src = [work_queue] if work_queue is not None else []
    return task_class(
        src_resources=src,
        dst_resources=[],
        cfg=cfg,
        warp_idx=cfg.padding_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.padding_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="PaddingTask",
        **kw,
    )


def create_throughput_latency_scheduler_task(
    work_queue: WorkQueue,
    schedule_token_throttle,
    cfg,
    *,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the CLC dynamic persistent scheduler task."""

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def scheduler_schedule(work_queue, schedule_token_throttle=None):
        """Fetch and publish the next dynamic persistent work tile."""

        with work_tile_loop(work_queue, skip_if=work_tile_skip_if) as work_tiles:
            if work_tile_skip_if is None:
                with domain_loop(0, 0, 1):
                    pass
                schedule_token_throttle_tail(schedule_token_throttle)
            else:
                # Runtime-empty packed-Q tiles do not publish a load token, but
                # the scheduler must still fetch and advance past them.
                with work_tiles.skippable():
                    with domain_loop(0, 0, 1):
                        pass
                    schedule_token_throttle_tail(schedule_token_throttle)
            work_queue.acquire()
            work_queue.fetch_work_tile()
            work_queue.commit()
            work_queue.wait()
            work_queue.get_and_advance_work_tile()
            work_queue.release()

    captured_schedule = (
        scheduler_schedule(work_queue)
        if schedule_token_throttle is None
        else scheduler_schedule(work_queue, schedule_token_throttle)
    )
    src = [work_queue]
    if schedule_token_throttle is not None:
        src.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=[work_queue],
        cfg=cfg,
        warp_idx=cfg.scheduler_warp_idx,
        num_warps=cfg.scheduler_num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_regs,
        name="SchedulerTask",
        **kw,
    )
