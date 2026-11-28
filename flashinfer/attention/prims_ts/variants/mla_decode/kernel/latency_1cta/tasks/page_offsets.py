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

"""Page-offset tasks of the latency 1CTA MLA decode kernel: paged-KV page-ID staging."""

from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ......device.schedule import page_offsets_produce
from ......device.stage import WorkStage
from ...device.schedule import runtime_work_tile_skip_if, work_tile_schedule_loop
from .common import MlaDecodeTask


def create_load_page_offsets_task(
    smem_page_offsets,
    work_queue: WorkQueue | None,
    cfg,
    *,
    domain,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create the page-offset staging task for the selected 1CTA cadence.

    Swaps-MMA-AB stages only K0/K1 offsets because each delayed V tile shares
    its K tile's page map and reuses the register-cached IDs. Keeps-MMA-AB
    retains its independent K/V offset cadence. The produced offset stages are
    consumed by the generic load task.
    """

    work_tile_skip_if = runtime_work_tile_skip_if(work_queue)

    @schedule
    def page_offsets_schedule(smem_page_offsets, work_queue=None):
        """Stage page offsets in the selected K-before-V cadence."""
        smem_page_offsets.init_load_state()

        with work_tile_schedule_loop(work_queue, skip_if=work_tile_skip_if):
            smem_page_offsets.bind_sparse_request()
            if cfg.single_kv_pipe:
                page_offsets_produce(
                    smem_page_offsets, "load_k0", section=WorkStage.Head
                )
                with domain_loop(0, domain, 1):
                    page_offsets_produce(
                        smem_page_offsets, "load_k0", section=WorkStage.Loop
                    )
                    if not cfg.sparse_reuse_kv:
                        page_offsets_produce(
                            smem_page_offsets, "load_v0", section=WorkStage.Loop
                        )
                if not cfg.sparse_reuse_kv:
                    page_offsets_produce(
                        smem_page_offsets, "load_v0", section=WorkStage.Tail
                    )
            elif cfg.serial_swap_reuse:
                page_offsets_produce(
                    smem_page_offsets, "load_k0", section=WorkStage.Head
                )
                with domain_loop(0, domain, 1):
                    page_offsets_produce(
                        smem_page_offsets, "load_k1", section=WorkStage.Loop
                    )
                    page_offsets_produce(
                        smem_page_offsets, "load_k0", section=WorkStage.Loop
                    )
                page_offsets_produce(
                    smem_page_offsets, "load_k1", section=WorkStage.Tail
                )
            else:
                page_offsets_produce(
                    smem_page_offsets, "load_k0", section=WorkStage.Head
                )
                page_offsets_produce(
                    smem_page_offsets, "load_k1", section=WorkStage.Head
                )

                with domain_loop(0, domain, 1):
                    page_offsets_produce(
                        smem_page_offsets, "load_k0", section=WorkStage.Loop
                    )
                    page_offsets_produce(
                        smem_page_offsets, "load_k1", section=WorkStage.Loop
                    )

    schedule_result = (
        page_offsets_schedule(smem_page_offsets)
        if work_queue is None
        else page_offsets_schedule(smem_page_offsets, work_queue)
    )
    return make_page_offsets_task(
        smem_page_offsets,
        work_queue,
        cfg,
        schedule_result=schedule_result,
        warp_idx=warp_idx,
        num_warps=num_warps,
        task_class=task_class,
        **kw,
    )


def make_page_offsets_task(
    smem_page_offsets,
    work_queue: WorkQueue | None,
    cfg,
    *,
    schedule_result,
    warp_idx=None,
    num_warps=None,
    task_class=MlaDecodeTask,
    **kw,
) -> Task:
    """Create a page-offset staging task from a captured schedule result."""
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets],
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_regs,
        name="LoadPageTableTask",
        **kw,
    )
