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

"""Load tasks of the throughput 2CTA MLA decode kernel: Q/K/V TMA loads."""

from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ...device.schedule import captured_loop_bounds, staged_kv_tma_load, work_queue_tail
from ..resources import (
    PageOffsetWindowResource,
    SmemKResource,
    SmemKVResource,
    SmemQResource,
    SmemVResource,
)
from .common import MlaTask, capture_clc_work_tile_body
from ..work_queue import MlaWorkQueue


def create_load_combined_task(
    smem_q,
    smem_k,
    smem_v,
    page_offsets,
    work_queue=None,
    task_class=MlaTask,
    **task_kwargs,
):
    """Acquire K/V together and reuse each CTA's local K values for V."""
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def combined_schedule(smem_q, smem_k, smem_v, page_offsets, work_queue=None):
        """Acquire paired K/V stages, convert once, and publish V after peer DSMEM completion."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_v.init_load_state()
        smem_k.bind_sparse_request()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        with domain_loop(loop_start, loop_end, loop_step):
            page_offsets.wait()
            page_stage = page_offsets.sparse_stage()
            smem_k.acquire()
            smem_v.acquire()
            smem_v.combined_stage()
            smem_k.tma_load_shared(page_offset_stage=page_stage)
            smem_k.commit()
            smem_v.tma_load_shared(page_offset_stage=page_stage)
            smem_v.commit()
            page_offsets.release()
        work_queue_tail(work_queue, advance_label="advance_tile")

    captured = combined_schedule(smem_q, smem_k, smem_v, page_offsets, work_queue)
    return task_class(
        src_resources=[page_offsets] + ([work_queue] if work_queue is not None else []),
        dst_resources=[smem_q, smem_k, smem_v],
        warp_idx=smem_k.cfg.load_k_warp_id,
        num_warps=8,
        schedule=captured,
        name="LoadCombinedTask",
        **task_kwargs,
    )


def create_load_k_task(
    smem_q: SmemQResource,
    smem_k: SmemKResource,
    page_offsets=None,
    page_window=None,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 Q/K TMA task on its configured loader warps.

    Q is loaded once before the loop. K uses one whole-tile pipeline stage per
    logical K tile. Sparse indices are prefetched through the shared offset
    ring (``page_offsets``); ordinary paged inputs take page IDs from the
    prefetched register window of ``page_window``. Pass exactly one of them.
    """
    if (page_offsets is None) == (page_window is None):
        raise ValueError("pass exactly one of page_offsets or page_window")
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    def load_k_body(
        smem_q, smem_k, page_offsets=None, page_window=None, work_queue=None
    ):
        """Load Q once and then publish one K stage per loop tile."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        if page_offsets is None:
            k_pages, window_page, next_window_page = page_window.init_dense_k_window()
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()

        with domain_loop(loop_start, loop_end, loop_step):
            if page_offsets is not None:
                page_offsets.wait()
                page_stage = page_offsets.sparse_stage()
                smem_k.acquire()
                smem_k.tma_load_shared(page_offset_stage=page_stage)
                smem_k.commit()
                # The last quad has been copied into TMA operands. Only now
                # may the metadata producer overwrite this shared ring slot.
                page_offsets.release()
            else:
                # Read the page IDs before waiting for the free slot so the
                # window refresh overlaps the wait instead of following it.
                k_pages, window_page, next_window_page = (
                    page_window.read_dense_k_window(
                        cached_k_pages=k_pages,
                        cached_window_page=window_page,
                        cached_next_window_page=next_window_page,
                    )
                )
                smem_k.acquire()
                smem_k.tma_load_paged(cached_page_ids=k_pages)
                smem_k.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    @schedule
    def load_k_schedule(smem_q, smem_k, page_offsets, work_queue=None):
        load_k_body(smem_q, smem_k, page_offsets, None, work_queue)

    @schedule
    def load_k_dense_schedule(smem_q, smem_k, page_window, work_queue=None):
        load_k_body(smem_q, smem_k, None, page_window, work_queue)

    if page_offsets is None:
        schedule_result = (
            load_k_dense_schedule(smem_q, smem_k, page_window)
            if work_queue is None
            else load_k_dense_schedule(smem_q, smem_k, page_window, work_queue)
        )
    else:
        schedule_result = (
            load_k_schedule(smem_q, smem_k, page_offsets)
            if work_queue is None
            else load_k_schedule(smem_q, smem_k, page_offsets, work_queue)
        )

    src = [page_offsets if page_offsets is not None else page_window] + (
        [work_queue] if work_queue is not None else []
    )
    return task_class(
        src_resources=src,
        dst_resources=[smem_q, smem_k],
        warp_idx=smem_k.cfg.load_k_warp_id,
        num_warps=smem_k.cfg.load_num_warps,
        schedule=schedule_result,
        name="LoadKTask",
        **task_kwargs,
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

        capture_clc_work_tile_body(
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
        warp_idx=smem_kv.cfg.load_tma_warp_id,
        num_warps=smem_kv.cfg.load_num_warps,
        schedule=captured_schedule,
        name="LoadTmaTask",
        **task_kwargs,
    )


def create_load_v_task(
    smem_v: SmemVResource,
    page_offsets=None,
    page_window=None,
    work_queue: MlaWorkQueue = None,
    task_class: type = MlaTask,
    **task_kwargs,
) -> Task:
    """Create the FP8 V TMA task on its configured loader warps.

    Sparse indices come from the shared offset ring (``page_offsets``); ordinary
    paged inputs take page IDs from ``page_window``. Pass exactly one of them.
    """
    if (page_offsets is None) == (page_window is None):
        raise ValueError("pass exactly one of page_offsets or page_window")
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    def load_v_body(smem_v, page_offsets=None, page_window=None, work_queue=None):
        """Publish one V stage per logical K tile."""
        smem_v.init_load_state()
        if page_offsets is None:
            v_pages, window_page, next_window_page = page_window.init_dense_v_window()
        with domain_loop(loop_start, loop_end, loop_step):
            if page_offsets is not None:
                page_offsets.wait()
                page_stage = page_offsets.sparse_stage()
                smem_v.acquire()
                smem_v.tma_load_shared(page_offset_stage=page_stage)
                smem_v.commit()
                # Keep the ring slot protected through every shared read.
                page_offsets.release()
            else:
                v_pages, window_page, next_window_page = (
                    page_window.read_dense_v_window(
                        cached_v_pages=v_pages,
                        cached_window_page=window_page,
                        cached_next_window_page=next_window_page,
                    )
                )
                smem_v.acquire()
                smem_v.tma_load_paged(cached_page_ids=v_pages)
                smem_v.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    @schedule
    def load_v_schedule(smem_v, page_offsets, work_queue=None):
        load_v_body(smem_v, page_offsets, None, work_queue)

    @schedule
    def load_v_dense_schedule(smem_v, page_window, work_queue=None):
        load_v_body(smem_v, None, page_window, work_queue)

    if page_offsets is None:
        schedule_result = (
            load_v_dense_schedule(smem_v, page_window)
            if work_queue is None
            else load_v_dense_schedule(smem_v, page_window, work_queue)
        )
    else:
        schedule_result = (
            load_v_schedule(smem_v, page_offsets)
            if work_queue is None
            else load_v_schedule(smem_v, page_offsets, work_queue)
        )

    src = [page_offsets if page_offsets is not None else page_window] + (
        [work_queue] if work_queue is not None else []
    )
    return task_class(
        src_resources=src,
        dst_resources=[smem_v],
        warp_idx=smem_v.cfg.load_v_warp_id,
        num_warps=smem_v.cfg.load_num_warps,
        schedule=schedule_result,
        name="LoadVTask",
        **task_kwargs,
    )


def create_sparse_offsets_task(
    page_offsets, work_queue=None, task_class=MlaTask, **task_kwargs
):
    loop_start, loop_end, loop_step = captured_loop_bounds(task_kwargs, 0)

    @schedule
    def offsets_schedule(page_offsets, work_queue=None):
        page_offsets.init_sparse_state()
        page_offsets.bind_sparse_request()
        with domain_loop(loop_start, loop_end, loop_step):
            page_offsets.acquire()
            page_offsets.load_sparse_offsets()
            page_offsets.commit()
        work_queue_tail(work_queue, advance_label="advance_tile")

    result = (
        offsets_schedule(page_offsets)
        if work_queue is None
        else offsets_schedule(page_offsets, work_queue)
    )
    return task_class(
        src_resources=[work_queue] if work_queue is not None else [],
        dst_resources=[page_offsets],
        warp_idx=9,
        num_warps=1,
        schedule=result,
        name="SparseOffsetsTask",
        **task_kwargs,
    )
