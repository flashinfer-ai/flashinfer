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

"""Epilogue tasks of the throughput 2CTA MLA decode kernel: output stores."""

import cutlass.cute as cute
from cutlass.experimental import primitives as prims
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from ...device.schedule import work_queue_tail
from .common import MlaTask


class MlaOutputTmaTask(MlaTask):
    """Drain global TMA writes once after the persistent work loop."""

    @cute.jit
    def drain_mla_work_tile_tails(self):
        MlaTask.drain_mla_work_tile_tails(self)
        prims.cp_async_bulk_wait_group(0)


class MlaStagedOutputTask(MlaTask):
    """Correction task that drains its per-warp output TMA stores at exit."""

    @cute.jit
    def drain_mla_work_tile_tails(self):
        MlaTask.drain_mla_work_tile_tails(self)
        prims.cp_async_bulk_wait_group(0)


def create_output_tma_task(output, work_queue=None, **task_kwargs):
    """Consume two quantized N slices on the otherwise idle sparse FP8 warp."""

    @schedule
    def output_schedule(output, work_queue=None):
        # This task only has epilogue work; declare its empty K loop.
        with domain_loop(0, 0, 1):
            pass
        for iter_n in range(2):
            output.wait()
            output.tma_store_output(iter_n=iter_n)
            output.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    captured = (
        output_schedule(output)
        if work_queue is None
        else output_schedule(output, work_queue)
    )
    return MlaOutputTmaTask(
        src_resources=[output] + ([work_queue] if work_queue is not None else []),
        dst_resources=[],
        warp_idx=10,
        num_warps=1,
        schedule=captured,
        name="OutputTmaTask",
        **task_kwargs,
    )
