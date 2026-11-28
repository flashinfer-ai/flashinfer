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

"""Page-offset task: stages paged-KV page offsets for the load warps."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from .common import (
    DecodeGenTask,
    TaskKwarg,
    can_hold_native_page_window,
    decode_work_tile_schedule,
    schedule_with_optional_resources,
)
from .....device.stage import WorkStage
from .....device.schedule import (
    page_offsets_produce,
)


def create_page_offsets_task(
    smem_page_offsets: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    membership_lifetime: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch page-table entries for the shared-KV load task.

    A held native window is published once per CTA work item and retained
    through the V tail. Other profiles mirror LoadTask's K/V cadence.
    """
    hold_page_window = can_hold_native_page_window(cfg, smem_page_offsets)

    head_labels = ("load_k0",) if cfg.num_insts_kv == 1 else ("load_k0", "load_k1")
    tail_labels = ("load_v0",) if cfg.num_insts_kv == 1 else ("load_v0", "load_v1")

    def page_offsets_schedule_body(
        smem_page_offsets: MemoryResource,
        membership_lifetime: MemoryResource | None,
    ) -> None:
        """Schedule page-offset prefetches for the shared-KV load cadence."""
        if membership_lifetime is not None:
            membership_lifetime.acquire()
        smem_page_offsets.init_load_state()
        if hold_page_window:
            # K0's held window covers every tile assigned to this CTA work
            # item, and native paged KV uses those same locators for V.
            page_offsets_produce(smem_page_offsets, "load_k0", WorkStage.Head)
            if membership_lifetime is not None:
                membership_lifetime.commit()
            # Preserve the runtime domain contract even though this fast path
            # needs no per-iteration page-window work.
            with domain_loop(0, domain, 1, unroll=1):
                pass
            return
        # Page offsets are staged by a separate producer so the load warp can
        # issue K/V TMA copies without reading page tables itself.
        # HEAD: produce page IDs for the two prefetched K tiles.
        for label in head_labels:
            produce_staged_page_offsets(smem_page_offsets, label, WorkStage.Head, cfg)
            if label == "load_k0" and membership_lifetime is not None:
                membership_lifetime.commit()

        # LOOP: mirror LoadTask's K/V production cadence exactly.
        if cfg.num_insts_kv == 1:
            loop_labels = ("load_k0", "load_v0")
        elif cfg.uses_two_inst_tmem_p:
            loop_labels = ("load_v0", "load_k0", "load_v1", "load_k1")
        else:
            loop_labels = ("load_k0", "load_v0", "load_k1", "load_v1")
        with domain_loop(0, domain, 1, unroll=1):
            for label in loop_labels:
                produce_staged_page_offsets(
                    smem_page_offsets, label, WorkStage.Loop, cfg
                )

        # TAIL: produce page IDs for the final two V loads.
        for label in tail_labels:
            produce_staged_page_offsets(smem_page_offsets, label, WorkStage.Tail, cfg)

    @schedule_with_optional_resources
    def page_offsets_schedule(
        smem_page_offsets: MemoryResource,
        membership_lifetime: MemoryResource | None,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap page-offset data work in packed persistent skip handling."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(smem_page_offsets, membership_lifetime),
        )

    captured_schedule = page_offsets_schedule(
        smem_page_offsets, membership_lifetime, work_queue
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets]
        + ([membership_lifetime] if membership_lifetime is not None else []),
        q_bound_resources=((smem_page_offsets, False),),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def create_page_offsets_task_one_inst_qkv(
    smem_page_offsets: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    membership_lifetime: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch offsets for one-instance Keeps, using the configured producer warps."""
    hold_page_window = can_hold_native_page_window(cfg, smem_page_offsets)

    def page_offsets_schedule_body(
        smem_page_offsets: MemoryResource,
        membership_lifetime: MemoryResource | None,
    ) -> None:
        """Schedule page-offset prefetches for the one-inst QKV load cadence."""
        if membership_lifetime is not None:
            membership_lifetime.acquire()
        smem_page_offsets.init_load_state()

        if hold_page_window:
            # Publish the complete CTA-local physical-KV128 locator span once.
            # The load task retains this stage from its K head through V tail.
            page_offsets_produce(smem_page_offsets, "load_k0", WorkStage.Head)
            if membership_lifetime is not None:
                membership_lifetime.commit()
            # Keep the task domain identical to the ordinary cadence even
            # though the held stage needs no per-iteration producer work.
            with domain_loop(0, domain, 1, unroll=1):
                pass
            return

        page_offsets_produce(smem_page_offsets, "load_k0", WorkStage.Head)
        if membership_lifetime is not None:
            membership_lifetime.commit()
        with domain_loop(0, domain, 1, unroll=1):
            for label in ("load_k0", "load_v0"):
                page_offsets_produce(smem_page_offsets, label, WorkStage.Loop)
        page_offsets_produce(smem_page_offsets, "load_v0", WorkStage.Tail)

    @schedule_with_optional_resources
    def page_offsets_schedule(
        smem_page_offsets: MemoryResource,
        membership_lifetime: MemoryResource | None,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap one-inst page-offset work in packed persistent skip handling."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(smem_page_offsets, membership_lifetime),
        )

    schedule_result = page_offsets_schedule(
        smem_page_offsets, membership_lifetime, work_queue
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets]
        + ([membership_lifetime] if membership_lifetime is not None else []),
        q_bound_resources=((smem_page_offsets, False),),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def create_page_offsets_task_split_kv(
    smem_page_offsets_k: MemoryResource,
    smem_page_offsets_v: MemoryResource,
    work_queue: WorkQueue | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    membership_lifetime: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Prefetch split-KV K/V page windows with independent consumer state."""

    def page_offsets_schedule_body(
        smem_page_offsets_k: MemoryResource,
        smem_page_offsets_v: MemoryResource,
        membership_lifetime: MemoryResource | None,
    ) -> None:
        """Publish one page-offset stage for every paired K/V tile."""
        if membership_lifetime is not None:
            membership_lifetime.acquire()
        smem_page_offsets_k.init_load_state()
        smem_page_offsets_v.init_load_state()

        # HEAD: publish the initial K0/K1 pair as two independent stages.
        page_offsets_produce(smem_page_offsets_k, "load_k0", WorkStage.Head)
        if membership_lifetime is not None:
            membership_lifetime.commit()
        if cfg.num_insts_kv != 1:
            page_offsets_produce(smem_page_offsets_k, "load_k1", WorkStage.Head)

        # LOOP: mirror LoadTask's cross-resource consumption order exactly.
        with domain_loop(0, domain, 1, unroll=1):
            if not cfg.uses_dual_kv_transform:
                page_offsets_produce(smem_page_offsets_v, "load_v0", WorkStage.Loop)
            page_offsets_produce(smem_page_offsets_k, "load_k0", WorkStage.Loop)
            if cfg.uses_dual_kv_transform:
                page_offsets_produce(smem_page_offsets_v, "load_v0", WorkStage.Loop)
            if cfg.num_insts_kv != 1:
                page_offsets_produce(smem_page_offsets_v, "load_v1", WorkStage.Loop)
                page_offsets_produce(smem_page_offsets_k, "load_k1", WorkStage.Loop)

        # TAIL: drain V0/V1 through distinct page-offset stages.
        page_offsets_produce(smem_page_offsets_v, "load_v0", WorkStage.Tail)
        if cfg.num_insts_kv != 1:
            page_offsets_produce(smem_page_offsets_v, "load_v1", WorkStage.Tail)

    @schedule_with_optional_resources
    def page_offsets_schedule(
        smem_page_offsets_k: MemoryResource,
        smem_page_offsets_v: MemoryResource,
        membership_lifetime: MemoryResource | None,
        work_queue: WorkQueue | None = None,
    ) -> None:
        """Wrap split page-offset work in packed persistent skip handling."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: page_offsets_schedule_body(
                smem_page_offsets_k,
                smem_page_offsets_v,
                membership_lifetime,
            ),
        )

    schedule_result = page_offsets_schedule(
        smem_page_offsets_k, smem_page_offsets_v, membership_lifetime, work_queue
    )
    src = []
    if work_queue is not None:
        src.append(work_queue)
    return task_class(
        src_resources=src,
        dst_resources=[smem_page_offsets_k, smem_page_offsets_v]
        + ([membership_lifetime] if membership_lifetime is not None else []),
        q_bound_resources=(
            (smem_page_offsets_k, False),
            (smem_page_offsets_v, False),
        ),
        cfg=cfg,
        warp_idx=cfg.page_offsets_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.page_offsets_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="PageTableTask",
        **kw,
    )


def page_offsets_consume(
    smem_page_offsets: MemoryResource | None, label: str = "read_offsets"
) -> None:
    """Consume the paged-KV offsets that match the next K/V TMA load."""
    if smem_page_offsets is None:
        return
    # ConsWait: wait until the page-offset producer staged the page IDs.
    smem_page_offsets.wait()
    # ConsWork: expose the page IDs to the K/V load resource.
    getattr(smem_page_offsets, label)()


def produce_staged_page_offsets(
    smem_page_offsets: MemoryResource,
    label: str,
    section: WorkStage,
    cfg: FmhaDecodeConfig,
) -> None:
    """Produce the page-offset stage shared by one logical K/V load."""
    _ = cfg
    page_offsets_produce(smem_page_offsets, label, section)
