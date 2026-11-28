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

"""Page-offset tasks of the context kernel: paged-KV page-ID staging."""

from cutlass import Int32
from cutlass.experimental.task_scheduling.resources import WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..resources.gmem_qkv import GmemQKVResource
from ..resources.smem_kv import SmemPageOffsetsKvResource
from .common import (
    captured_loop_bounds,
    packed_context_skip_predicate,
    schedule_with_work_queue,
    src_resources,
    work_tile_schedule_loop,
)
from typing import Any


def create_page_offsets_task(
    gmem_qkv: GmemQKVResource,
    smem_page_offsets_kv: SmemPageOffsetsKvResource,
    work_queue: WorkQueue | None,
    task_class: type[Task] = Task,
    num_registers: int = 32,
    smem_page_offsets_v: SmemPageOffsetsKvResource | None = None,
    **task_kwargs: Any,
) -> Task:
    """Create the one-warp paged-KV page-offsets prefetch task.

    Replaces ``create_padding_task`` when ``cfg.use_paged_kv`` is True. The
    configuration's empty/scheduler warp prefetches page-table entries into
    SMEM so the load warp can read cached page IDs when issuing paged TMA
    copies. This also preserves that warp's ``setmaxnreg.sync`` participation.

    Paired D128 CLC does not instantiate this task: its load warp reads page
    IDs directly, leaving warp 15 exclusively responsible for CLC. Staged
    D256 can use this task with CLC because page offsets run on the empty warp
    while the freed epilogue warp owns scheduling. No task therefore combines
    dynamic work-queue and page-offset production through the public DSL API.
    """
    loop_start, loop_end, loop_step = captured_loop_bounds(task_class, task_kwargs)
    skip_work_tile_if = packed_context_skip_predicate(work_queue)
    src = src_resources(gmem_qkv, work_queue=work_queue)
    dst = [smem_page_offsets_kv]
    if smem_page_offsets_v is not None:
        dst.append(smem_page_offsets_v)
    staged_single_instance = (
        smem_page_offsets_kv.cfg.single_qkv_instance
        and smem_page_offsets_kv.cfg.has_tmem_p_pipeline
    )
    page_window_period = smem_page_offsets_kv.cfg.page_table_window_entries // (
        smem_page_offsets_kv.cfg.kv_tile_n
        // smem_page_offsets_kv.cfg.num_tokens_per_page
    )

    def page_offsets_schedule_body(
        gqkv: GmemQKVResource,
        spo: SmemPageOffsetsKvResource,
        spov: SmemPageOffsetsKvResource | None,
        wq: WorkQueue | None = None,
    ) -> None:
        """Captured schedule for K/V page-table prefetch."""
        spo.init_load_state()
        if spov is not None:
            spov.init_load_state()
        with work_tile_schedule_loop(wq, skip_if=skip_work_tile_if):
            (
                kv_tile_start,
                kv_request_begin,
                kv_page_idx_ub,
            ) = gqkv.compute_page_coords()
            if spov is None:
                prefetch_page_offsets_for_work_tile(
                    spo,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_step=loop_step,
                    staged_single_instance=staged_single_instance,
                )
            else:
                prefetch_reused_page_windows_for_work_tile(
                    spo,
                    spov,
                    kv_tile_start=kv_tile_start,
                    kv_request_begin=kv_request_begin,
                    kv_page_idx_ub=kv_page_idx_ub,
                    loop_start=loop_start,
                    loop_end=loop_end,
                    loop_step=loop_step,
                    page_window_period=page_window_period,
                )

    @schedule
    def page_offsets_schedule(
        gqkv: GmemQKVResource,
        spo: SmemPageOffsetsKvResource,
        wq: WorkQueue | None = None,
    ) -> None:
        page_offsets_schedule_body(gqkv, spo, None, wq)

    @schedule
    def reused_page_windows_schedule(
        gqkv: GmemQKVResource,
        spok: SmemPageOffsetsKvResource,
        spov: SmemPageOffsetsKvResource,
        wq: WorkQueue | None = None,
    ) -> None:
        page_offsets_schedule_body(gqkv, spok, spov, wq)

    if smem_page_offsets_v is None:
        captured_schedule = schedule_with_work_queue(
            page_offsets_schedule,
            gmem_qkv,
            smem_page_offsets_kv,
            work_queue=work_queue,
        )
    else:
        captured_schedule = schedule_with_work_queue(
            reused_page_windows_schedule,
            gmem_qkv,
            smem_page_offsets_kv,
            smem_page_offsets_v,
            work_queue=work_queue,
        )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        warp_idx=smem_page_offsets_kv.cfg.empty_warp_id,
        num_warps=1,
        schedule=captured_schedule,
        num_registers=num_registers,
        name="PageTableTask",
        **task_kwargs,
    )


def prefetch_page_offsets_for_work_tile(
    spo: SmemPageOffsetsKvResource,
    *,
    kv_tile_start: Int32,
    kv_request_begin: Int32,
    kv_page_idx_ub: Int32,
    loop_start: int,
    loop_end: int,
    loop_step: int,
    staged_single_instance: bool,
) -> None:
    """Produce page-ID stages in the same logical order as the load task."""
    if staged_single_instance:
        # D>128 overlaps QK(i) with PV(i-1): K0, then K_i/V_{i-1}, then V_last.
        # One page-ID stage is shared by all head-dimension slices of a logical
        # K or V tile, so this producer fires once per tile rather than per slice.
        spo.acquire()
        spo.load_k(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        with domain_loop(loop_start + 1, loop_end, loop_step):
            spo.acquire()
            spo.load_k(
                kv_tile_start=kv_tile_start,
                kv_request_begin=kv_request_begin,
                kv_page_idx_ub=kv_page_idx_ub,
            )
            spo.commit()
            spo.acquire()
            spo.load_v(
                previous=True,
                kv_tile_start=kv_tile_start,
                kv_request_begin=kv_request_begin,
                kv_page_idx_ub=kv_page_idx_ub,
            )
            spo.commit()
        spo.acquire()
        spo.load_v(
            previous=False,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        return

    with domain_loop(loop_start, loop_end, loop_step):
        spo.acquire()
        spo.load_k(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()
        spo.acquire()
        spo.load_v(
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spo.commit()


def prefetch_reused_page_windows_for_work_tile(
    spok: SmemPageOffsetsKvResource,
    spov: SmemPageOffsetsKvResource,
    *,
    kv_tile_start: Int32,
    kv_request_begin: Int32,
    kv_page_idx_ub: Int32,
    loop_start: object,
    loop_end: object,
    loop_step: object,
    page_window_period: int,
) -> None:
    """Publish independent K/V page windows at their structural cadence."""
    if (
        not isinstance(loop_start, int)
        or not isinstance(loop_end, int)
        or not isinstance(loop_step, int)
        or loop_start != 0
        or loop_step != 1
        or loop_end < page_window_period
        or loop_end % page_window_period != 0
    ):
        raise ValueError(
            "reused page windows require a compile-time K/V domain "
            "divisible by the topology-derived page-window period"
        )

    spok.acquire()
    spok.load_k(
        tile_offset=0,
        kv_tile_start=kv_tile_start,
        kv_request_begin=kv_request_begin,
        kv_page_idx_ub=kv_page_idx_ub,
    )
    spok.commit()
    spov.acquire()
    spov.load_v(
        tile_offset=0,
        kv_tile_start=kv_tile_start,
        kv_request_begin=kv_request_begin,
        kv_page_idx_ub=kv_page_idx_ub,
    )
    spov.commit()

    with domain_loop(page_window_period, loop_end, page_window_period):
        spok.acquire()
        spok.load_k(
            tile_offset=0,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spok.commit()
        spov.acquire()
        spov.load_v(
            tile_offset=0,
            kv_tile_start=kv_tile_start,
            kv_request_begin=kv_request_begin,
            kv_page_idx_ub=kv_page_idx_ub,
        )
        spov.commit()
