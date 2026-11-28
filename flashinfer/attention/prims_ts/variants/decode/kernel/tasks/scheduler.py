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

"""Scheduler and padding tasks."""

from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import (
    domain_loop,
    schedule,
    work_tile_loop,
)
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from .common import DecodeGenTask, TaskKwarg, decode_work_tile_schedule, work_queue_tail
from .....device.schedule import schedule_token_throttle_tail


# ======================================================================
# PaddingTask — fills the unused tail warps of one warp group
# ======================================================================
def create_padding_task(
    cfg: FmhaDecodeConfig,
    work_queue: WorkQueue | None = None,
    *,
    warp_idx: int,
    num_warps: int,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the padding task used to balance persistent warp groups."""

    @schedule
    def padding_schedule(work_queue: WorkQueue | None = None) -> None:
        """Run the padding schedule and advance the work queue if present."""

        def padding_body() -> None:
            # Empty loop body: this warp participates in task/register
            # scheduling so the persistent warpgroup layout remains balanced.
            with domain_loop(0, 1, 1, unroll=1):
                pass

        decode_work_tile_schedule(cfg, work_queue, padding_body)

    captured_schedule = (
        padding_schedule() if work_queue is None else padding_schedule(work_queue)
    )
    src = []
    if work_queue is not None:
        src = [work_queue]
    return task_class(
        src_resources=src,
        dst_resources=[],
        cfg=cfg,
        warp_idx=warp_idx,
        num_warps=num_warps,
        schedule=captured_schedule,
        num_registers=cfg.padding_task_num_registers(warp_idx, num_warps),
        name="PaddingTask",
        **kw,
    )


def create_scheduler_task(
    work_queue: WorkQueue,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the CLC scheduler task.

    Two-inst profiles place scheduler/load/padding roles in WG3. One-inst Keeps
    profiles place scheduler at warp 9 beside MMA, page-or-padding, and load in
    WG2; a separate padding task keeps WG3 and the 512-thread CLC contract full.
    """

    @schedule
    def scheduler_schedule(
        work_queue: WorkQueue,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Fetch and publish the next persistent-scheduler work tile."""
        with work_tile_loop(work_queue):
            # The scheduler owns work-tile discovery for persistent kernels.
            # Empty domain keeps the generated schedule shape consistent with
            # other TS tasks while all real work happens through WorkQueue.
            with domain_loop(0, 0, 1, unroll=1):
                pass
            schedule_token_throttle_tail(schedule_token_throttle)
            # ProdAcquire/ProdWork/ProdCommit: fetch and publish the next work
            # tile to all persistent tasks.
            work_queue.acquire()
            work_queue.fetch_work_tile()
            work_queue.commit()
            work_queue_tail(work_queue)

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
        num_registers=cfg.mma_load_task_num_registers,
        name="SchedulerTask",
        **kw,
    )
