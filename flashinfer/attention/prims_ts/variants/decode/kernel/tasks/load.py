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

"""Load tasks: Q, K/V (dense, paged and block-sparse) and their scales."""

import cutlass
from cutlass.experimental.task_scheduling.resources import MemoryResource, WorkQueue
from cutlass.experimental.task_scheduling.schedule_builder import domain_loop, schedule
from cutlass.experimental.task_scheduling.task import Task
from ..config import FmhaDecodeConfig
from ..device.constants import KV_INST0, KV_INST1, KV_KIND_K, KV_KIND_V
from .common import (
    DecodeGenTask,
    TaskKwarg,
    can_hold_native_page_window,
    decode_work_tile_schedule,
    schedule_with_optional_resources,
)
from .....device.schedule import schedule_token_throttle_head
from .page_offsets import page_offsets_consume
from .....device.schedule import page_offsets_release
from .....device.stage import WorkStage
from typing import Any


def create_block_sparse_load_tasks_per_inst(
    smem_q: MemoryResource,
    smem_k0: MemoryResource,
    smem_k1: MemoryResource,
    smem_v0: MemoryResource,
    smem_v1: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    sparse_kv_metadata0: MemoryResource,
    sparse_kv_metadata1: MemoryResource,
    sparse_softmax_metadata0: MemoryResource,
    sparse_softmax_metadata1: MemoryResource,
    warp_indices: tuple[int, int],
    sage_k_scales0: MemoryResource | None = None,
    sage_k_scales1: MemoryResource | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> tuple[Task, Task]:
    """Assign each sparse K/V instruction stream to an independent load warp.

    Load0 alone owns Q and the persistent schedule-token throttle. Both tasks
    consume the same logical work tile, while their K/V, sparse-metadata and
    Sage K scale pipelines remain disjoint.
    """

    if not cfg.use_block_sparse:
        raise ValueError("per-instance load tasks require block-sparse metadata")

    load0 = create_load_task_split_kv(
        smem_q,
        smem_k0,
        None,
        smem_v0,
        None,
        work_queue,
        schedule_token_throttle,
        cfg,
        domain=domain,
        sparse_kv_metadata0=sparse_kv_metadata0,
        sparse_softmax_metadata0=sparse_softmax_metadata0,
        sage_k_scales0=sage_k_scales0,
        warp_idx=warp_indices[0],
        task_name="LoadTask0",
        task_class=task_class,
        **kw,
    )
    load1 = create_load_task_split_kv(
        None,
        None,
        smem_k1,
        None,
        smem_v1,
        work_queue,
        None,
        cfg,
        domain=domain,
        sparse_kv_metadata1=sparse_kv_metadata1,
        sparse_softmax_metadata1=sparse_softmax_metadata1,
        sage_k_scales1=sage_k_scales1,
        warp_idx=warp_indices[1],
        task_name="LoadTask1",
        task_class=task_class,
        **kw,
    )
    return load0, load1


def create_load_task(
    smem_q: MemoryResource,
    smem_kv: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    sparse_kv_metadata0: MemoryResource | None = None,
    sparse_kv_metadata1: MemoryResource | None = None,
    sparse_softmax_metadata0: MemoryResource | None = None,
    sparse_softmax_metadata1: MemoryResource | None = None,
    sage_k_scales0: MemoryResource | None = None,
    sage_k_scales1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create the shared-KV load task and optional page-offset dependency.

    ``sage_k_scales0/1`` are a Sage plan's ``sfK`` rings, which the task
    fills after each K tile.
    """
    hold_page_window = can_hold_native_page_window(cfg, smem_page_offsets)
    head_labels = ("load_k0",) if cfg.num_insts_kv == 1 else ("load_k0", "load_k1")
    tail_labels = ("load_v0",) if cfg.num_insts_kv == 1 else ("load_v0", "load_v1")

    def load_schedule_body(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        schedule_token_throttle: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None = None,
        sparse_kv_metadata1: MemoryResource | None = None,
        sparse_softmax_metadata0: MemoryResource | None = None,
        sparse_softmax_metadata1: MemoryResource | None = None,
        sage_k_scales0: MemoryResource | None = None,
        sage_k_scales1: MemoryResource | None = None,
    ) -> None:
        """Build the shared-KV load cadence for HEAD, LOOP, and TAIL."""
        smem_q.init_load_state()
        smem_kv.init_load_state()
        for resource in (
            sparse_kv_metadata0,
            sparse_kv_metadata1,
            sparse_softmax_metadata0,
            sparse_softmax_metadata1,
        ):
            if resource is not None:
                resource.init_load_state()
        cached_page_ids = None
        if smem_page_offsets is not None:
            if cfg.num_head_dim_stages_kv > 1 and not cfg.uses_scattered_page_route:
                cached_page_ids = smem_page_offsets.init_cached_read_state()
            else:
                smem_page_offsets.init_read_state()

        def kv_load(label: str, section: WorkStage) -> None:
            """Run one staged shared-KV load with optional page-offset handoff."""
            nonlocal cached_page_ids
            cached_page_ids = staged_kv_load(
                smem_kv,
                label,
                section,
                cfg,
                smem_page_offsets=smem_page_offsets,
                page_offsets_label=label.replace("load_", "read_offsets_"),
                manage_page_offsets=not hold_page_window,
                cached_page_ids=cached_page_ids,
            )

        # The resources that follow each instance's K tile: its route metadata
        # and Softmax route stage of a block-sparse plan, and its ``sfK`` ring
        # of a Sage plan.
        k_resources_by_label = {
            "load_k0": (sparse_kv_metadata0, sparse_softmax_metadata0, sage_k_scales0),
            "load_k1": (sparse_kv_metadata1, sparse_softmax_metadata1, sage_k_scales1),
        }

        # HEAD: issue every instance's route-record load before Q, so their
        # global-memory latency overlaps the wait for the previous work tile
        # to release Q and the Q load itself. One cadence serves dense and
        # block-sparse loads: the route helpers skip a dense profile's
        # ``None`` metadata resources, and the K-scale publish fills a Sage
        # plan's ring from the dense tile or from the route.
        prefetch_by_label = {
            label: prefetch_sparse_route(k_resources_by_label[label][0], "head")
            for label in head_labels
        }
        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        if hold_page_window:
            # The native K/V caches share one page table. Keep its single
            # consumer stage live across every K/V and head-dimension load
            # owned by this CTA work tile. Sparse routes size it for the full local
            # span; other native split routes retain the aligned 32-ID window.
            if cfg.num_head_dim_stages_kv > 1 and not cfg.uses_scattered_page_route:
                smem_page_offsets.wait()
            else:
                page_offsets_consume(smem_page_offsets)
        head_routes = []
        for label in head_labels:
            kv_metadata, softmax_metadata, sage_k_scales = k_resources_by_label[label]
            route = resolve_and_store_sparse_route(
                kv_metadata, WorkStage.Head, prefetch_by_label[label]
            )
            # Each resolution issues its successor's record load ahead of its
            # K copies, which gives the load the longest lead time.
            prefetch_by_label[label] = prefetch_sparse_route(kv_metadata, "first_loop")
            kv_load(label, WorkStage.Head)
            head_routes.append((softmax_metadata, route, sage_k_scales))
        # Issue all K tiles before a metadata FIFO can backpressure the load warp.
        for softmax_metadata, route, sage_k_scales in head_routes:
            publish_sparse_softmax_route(softmax_metadata, route)
            publish_k_scales(sage_k_scales, route, WorkStage.Head)

        # LOOP: each iter prefetches the full ``num_insts_kv`` K/V pair set.
        # When P aliases the consumed S columns, MMA must consume each V/P pair
        # before the following same-instance QK overwrites S. Keep the
        # shared-ring producer order identical to that consumer order.
        if cfg.num_insts_kv == 1:
            loop_labels = ("load_k0", "load_v0")
        elif cfg.uses_two_inst_tmem_p:
            loop_labels = ("load_v0", "load_k0", "load_v1", "load_k1")
        else:
            loop_labels = ("load_k0", "load_v0", "load_k1", "load_v1")
        with domain_loop(0, domain, 1, unroll=1):
            # Generic V-first profiles consume their retained route before the
            # matching K label replaces it. Each route goes to its Softmax
            # consumer right after its own K issue instead of waiting for the
            # other instance's loads.
            for label in loop_labels:
                kv_metadata, softmax_metadata, sage_k_scales = k_resources_by_label.get(
                    label, (None, None, None)
                )
                route = resolve_and_store_sparse_route(
                    kv_metadata, WorkStage.Loop, prefetch_by_label.get(label)
                )
                prefetch_by_label[label] = prefetch_sparse_route(
                    kv_metadata, "next_loop"
                )
                kv_load(label, WorkStage.Loop)
                publish_sparse_softmax_route(softmax_metadata, route)
                publish_k_scales(sage_k_scales, route, WorkStage.Loop)

        # TAIL: load the final V tile for each active instance's last BMM2.
        for label in tail_labels:
            kv_load(label, WorkStage.Tail)
        if hold_page_window:
            page_offsets_release(smem_page_offsets)

    @schedule_with_optional_resources
    def load_schedule(
        smem_q: MemoryResource,
        smem_kv: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        sage_k_scales0: MemoryResource | None,
        sage_k_scales1: MemoryResource | None,
        work_queue: WorkQueue | None,
        schedule_token_throttle: MemoryResource | None,
    ) -> None:
        """Schedule shared-KV loads with only the resources in this profile."""

        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_kv,
                smem_page_offsets,
                schedule_token_throttle,
                sparse_kv_metadata0,
                sparse_kv_metadata1,
                sparse_softmax_metadata0,
                sparse_softmax_metadata1,
                sage_k_scales0,
                sage_k_scales1,
            ),
            lambda: schedule_token_throttle_head(schedule_token_throttle),
        )

    sparse_resources = (
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
    )
    sparse_resources_present = tuple(
        resource is not None for resource in sparse_resources
    )
    has_sparse_metadata = any(sparse_resources_present)
    if has_sparse_metadata and not all(sparse_resources_present):
        raise ValueError("shared sparse K/V requires both route/Softmax pairs")
    if has_sparse_metadata and smem_page_offsets is not None:
        raise ValueError("block-sparse and paged-KV cannot share a load task")

    captured_schedule = load_schedule(
        smem_q,
        smem_kv,
        smem_page_offsets,
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
        sage_k_scales0,
        sage_k_scales1,
        work_queue,
        schedule_token_throttle,
    )
    src = []
    for sparse_kv_metadata in (sparse_kv_metadata0, sparse_kv_metadata1):
        if sparse_kv_metadata is not None:
            src.append(sparse_kv_metadata)
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_kv]
    for resource in (*sparse_resources, sage_k_scales0, sage_k_scales1):
        if resource is not None and resource not in dst:
            dst.append(resource)
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((smem_q, True), (smem_kv, False)),
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=captured_schedule,
        num_registers=cfg.mma_load_task_num_registers,
        name="LoadTask",
        **kw,
    )


def create_load_task_one_inst_qkv(
    smem_q: MemoryResource,
    smem_k: MemoryResource,
    smem_v: MemoryResource,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Load schedule for the one-inst keepsMmaAb QKV path."""
    hold_page_window = can_hold_native_page_window(cfg, smem_page_offsets)

    def load_schedule_body(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_page_offsets: MemoryResource | None,
        schedule_token_throttle: MemoryResource | None,
    ) -> None:
        """Build the one-inst Q/K/V load cadence."""
        smem_q.init_load_state()
        smem_k.init_load_state()
        smem_v.init_load_state()
        if smem_page_offsets is not None:
            smem_page_offsets.init_read_state()

        def load_tile(resource: MemoryResource, label: str, section: WorkStage) -> None:
            """Load every head-dim stage from the active page window."""
            if not hold_page_window:
                page_offsets_consume(
                    smem_page_offsets, label.replace("load", "read_offsets")
                )
            for head_dim_stage_idx in range(cfg.num_raw_kv_head_dim_stages):
                resource.acquire()
                getattr(resource, label)(
                    section=section, head_dim_stage_idx=head_dim_stage_idx
                )
                resource.commit()
            if not hold_page_window:
                page_offsets_release(smem_page_offsets)

        smem_q.acquire()
        smem_q.tma_load()
        smem_q.commit()
        if hold_page_window:
            page_offsets_consume(smem_page_offsets)
        load_tile(smem_k, "load_k0", WorkStage.Head)

        with domain_loop(0, domain, 1, unroll=1):
            load_tile(smem_k, "load_k0", WorkStage.Loop)
            load_tile(smem_v, "load_v0", WorkStage.Loop)

        load_tile(smem_v, "load_v0", WorkStage.Tail)
        if hold_page_window:
            page_offsets_release(smem_page_offsets)

    @schedule
    def load_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        work_queue: WorkQueue | None = None,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Schedule one-inst Q/K/V loads without page-offset resources."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q, smem_k, smem_v, None, schedule_token_throttle
            ),
            lambda: schedule_token_throttle_head(schedule_token_throttle),
        )

    @schedule
    def load_page_offsets_schedule(
        smem_q: MemoryResource,
        smem_k: MemoryResource,
        smem_v: MemoryResource,
        smem_page_offsets: MemoryResource,
        work_queue: WorkQueue | None = None,
        schedule_token_throttle: MemoryResource | None = None,
    ) -> None:
        """Schedule one-inst Q/K/V loads with paged-KV offsets."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                schedule_token_throttle,
            ),
            lambda: schedule_token_throttle_head(schedule_token_throttle),
        )

    if smem_page_offsets is None:
        if work_queue is None:
            schedule_result = load_schedule(smem_q, smem_k, smem_v)
        elif schedule_token_throttle is None:
            schedule_result = load_schedule(smem_q, smem_k, smem_v, work_queue)
        else:
            schedule_result = load_schedule(
                smem_q, smem_k, smem_v, work_queue, schedule_token_throttle
            )
    else:
        if work_queue is None:
            schedule_result = load_page_offsets_schedule(
                smem_q, smem_k, smem_v, smem_page_offsets
            )
        elif schedule_token_throttle is None:
            schedule_result = load_page_offsets_schedule(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                work_queue,
            )
        else:
            schedule_result = load_page_offsets_schedule(
                smem_q,
                smem_k,
                smem_v,
                smem_page_offsets,
                work_queue,
                schedule_token_throttle,
            )
    src = []
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if work_queue is not None:
        src.append(work_queue)
    dst = [smem_q, smem_k, smem_v]
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=((smem_q, True), (smem_k, False), (smem_v, False)),
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name="LoadTask",
        **kw,
    )


def create_load_task_split_kv(
    smem_q: MemoryResource | None,
    smem_k0: MemoryResource | None,
    smem_k1: MemoryResource | None,
    smem_v0: MemoryResource | None,
    smem_v1: MemoryResource | None,
    work_queue: WorkQueue | None,
    schedule_token_throttle: MemoryResource | None,
    cfg: FmhaDecodeConfig,
    *,
    domain: int | cutlass.Int32,
    smem_page_offsets: MemoryResource | None = None,
    smem_page_offsets_v: MemoryResource | None = None,
    sparse_kv_metadata0: MemoryResource | None = None,
    sparse_kv_metadata1: MemoryResource | None = None,
    sparse_softmax_metadata0: MemoryResource | None = None,
    sparse_softmax_metadata1: MemoryResource | None = None,
    sage_k_scales0: MemoryResource | None = None,
    sage_k_scales1: MemoryResource | None = None,
    warp_idx: int | None = None,
    num_warps: int | None = None,
    task_name: str = "LoadTask",
    task_class: type[DecodeGenTask] = DecodeGenTask,
    **kw: TaskKwarg,
) -> Task:
    """Create loads for independent K0/K1/V0/V1 resources.

    Here ``split`` describes those resources, not split-KV reduction.
    ``sage_k_scales0/1`` are a Sage plan's ``sfK`` rings, which the task
    fills after each K tile.
    """
    if schedule_token_throttle is not None and work_queue is None:
        raise ValueError("schedule-token throttle requires a work queue")
    if smem_page_offsets_v is not None and smem_page_offsets is None:
        raise ValueError("V page offsets require K page offsets")
    if (smem_k0 is None) != (smem_v0 is None) or (smem_k1 is None) != (smem_v1 is None):
        raise ValueError("each split K resource requires its matching V resource")
    if smem_k0 is None and smem_k1 is None:
        raise ValueError("at least one split K/V instance is required")
    hold_page_window = smem_page_offsets_v is None and can_hold_native_page_window(
        cfg, smem_page_offsets
    )

    def load_schedule_body(
        smem_q: MemoryResource | None,
        smem_k0: MemoryResource | None,
        smem_k1: MemoryResource | None,
        smem_v0: MemoryResource | None,
        smem_v1: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        smem_page_offsets: MemoryResource | None,
        smem_page_offsets_v: MemoryResource | None = None,
        schedule_token_throttle: MemoryResource | None = None,
        sage_k_scales0: MemoryResource | None = None,
        sage_k_scales1: MemoryResource | None = None,
    ) -> None:
        """Build the split-resource K/V load cadence for all schedule phases."""
        if smem_q is not None:
            smem_q.init_load_state()
        sage_k_scales_by_label = {"load_k0": sage_k_scales0, "load_k1": sage_k_scales1}
        active_instances = (
            (
                smem_k0,
                smem_v0,
                sparse_kv_metadata0,
                sparse_softmax_metadata0,
                "load_k0",
                "load_v0",
            ),
            (
                smem_k0 if smem_k1 is None and not cfg.use_block_sparse else smem_k1,
                smem_v0 if smem_v1 is None and not cfg.use_block_sparse else smem_v1,
                sparse_kv_metadata1,
                sparse_softmax_metadata1,
                "load_k1",
                "load_v1",
            ),
        )
        active_instances = active_instances[: cfg.num_insts_kv]
        # Preserve the original full-resource lowering order while allowing a
        # per-instance task to omit the other stream's resources.
        for resource in (smem_k0, smem_k1, smem_v0, smem_v1):
            if resource is not None:
                resource.init_load_state()
        for _, _, sparse_kv_metadata, _, _, _ in active_instances:
            if sparse_kv_metadata is not None:
                sparse_kv_metadata.init_load_state()
        for _, _, _, sparse_softmax_metadata, _, _ in active_instances:
            if sparse_softmax_metadata is not None:
                sparse_softmax_metadata.init_load_state()
        if smem_page_offsets is not None:
            smem_page_offsets.init_read_state()
        if smem_page_offsets_v is not None:
            smem_page_offsets_v.init_read_state()

        smem_page_offsets_k = smem_page_offsets
        smem_page_offsets_v_local = (
            smem_page_offsets_v
            if smem_page_offsets_v is not None
            else smem_page_offsets
        )

        def load_tile(
            resource: MemoryResource,
            label: str,
            offsets: MemoryResource | None,
            section: WorkStage,
        ) -> None:
            """Acquire, load all head-dim stages, and release page offsets."""
            if not hold_page_window:
                page_offsets_consume(offsets, label.replace("load", "read_offsets"))
            for head_dim_stage_idx in range(cfg.num_raw_kv_head_dim_stages):
                resource.acquire()
                getattr(resource, label)(
                    section=section, head_dim_stage_idx=head_dim_stage_idx
                )
                resource.commit()
            if not hold_page_window:
                page_offsets_release(offsets)

        # Issue every instance's HEAD route-record load before Q, so their
        # global-memory latency overlaps the wait for the previous work tile
        # to release Q and the Q load itself.
        prefetch_by_label = {
            load_k: prefetch_sparse_route(sparse_kv_metadata, "head")
            for smem_k, _, sparse_kv_metadata, _, load_k, _ in active_instances
            if smem_k is not None
        }
        if smem_q is not None:
            smem_q.acquire()
            smem_q.tma_load()
            smem_q.commit()

        if hold_page_window:
            # Both instruction streams use the same immutable locator span.
            # Keep its stage live across all K/V tiles, not one stage per load.
            page_offsets_consume(smem_page_offsets)

        head_routes = []
        for (
            smem_k,
            _,
            sparse_kv_metadata,
            sparse_softmax_metadata,
            load_k,
            _,
        ) in active_instances:
            if smem_k is None:
                continue
            route = resolve_and_store_sparse_route(
                sparse_kv_metadata, WorkStage.Head, prefetch_by_label[load_k]
            )
            load_tile(smem_k, load_k, smem_page_offsets_k, WorkStage.Head)
            head_routes.append(
                (sparse_softmax_metadata, route, sage_k_scales_by_label[load_k])
            )
        # In the combined task, preserve both K issues ahead of Softmax
        # backpressure. A per-instance task naturally stages its sole route.
        for sparse_softmax_metadata, route, sage_k_scales in head_routes:
            publish_sparse_softmax_route(sparse_softmax_metadata, route)
            publish_k_scales(sage_k_scales, route, WorkStage.Head)
        # Successor record loads go out after the K copies and the route
        # publish, here and in the loop. Queued ahead of the copies, a record
        # load delays the shared-memory reads of the route origins that feed
        # their coordinates, and with them the work tile's first K tiles and
        # Softmax routes.
        prefetch_by_label = {
            load_k: prefetch_sparse_route(sparse_kv_metadata, "first_loop")
            for smem_k, _, sparse_kv_metadata, _, load_k, _ in active_instances
            if smem_k is not None
        }

        with domain_loop(0, domain, 1, unroll=1):
            # V consumes the retained route before the next K overwrites it.
            # Each route goes to its Softmax consumer right after its own K
            # issue instead of waiting for the other instance's loads.
            for (
                smem_k,
                smem_v,
                sparse_kv_metadata,
                sparse_softmax_metadata,
                load_k,
                load_v,
            ) in active_instances:
                if smem_k is None:
                    continue
                assert smem_v is not None
                if not cfg.uses_dual_kv_transform:
                    load_tile(smem_v, load_v, smem_page_offsets_v_local, WorkStage.Loop)
                route = resolve_and_store_sparse_route(
                    sparse_kv_metadata, WorkStage.Loop, prefetch_by_label[load_k]
                )
                load_tile(smem_k, load_k, smem_page_offsets_k, WorkStage.Loop)
                publish_sparse_softmax_route(sparse_softmax_metadata, route)
                publish_k_scales(sage_k_scales_by_label[load_k], route, WorkStage.Loop)
                if cfg.uses_dual_kv_transform:
                    load_tile(smem_v, load_v, smem_page_offsets_v_local, WorkStage.Loop)
                prefetch_by_label[load_k] = prefetch_sparse_route(
                    sparse_kv_metadata, "next_loop"
                )

        for _, smem_v, _, _, _, load_v in active_instances:
            if smem_v is not None:
                load_tile(
                    smem_v,
                    load_v,
                    smem_page_offsets_v_local,
                    WorkStage.Tail,
                )
        if hold_page_window:
            page_offsets_release(smem_page_offsets)

    @schedule_with_optional_resources
    def load_schedule(
        smem_q: MemoryResource | None,
        smem_k0: MemoryResource | None,
        smem_k1: MemoryResource | None,
        smem_v0: MemoryResource | None,
        smem_v1: MemoryResource | None,
        sparse_kv_metadata0: MemoryResource | None,
        sparse_kv_metadata1: MemoryResource | None,
        sparse_softmax_metadata0: MemoryResource | None,
        sparse_softmax_metadata1: MemoryResource | None,
        smem_page_offsets: MemoryResource | None,
        smem_page_offsets_v: MemoryResource | None,
        sage_k_scales0: MemoryResource | None,
        sage_k_scales1: MemoryResource | None,
        work_queue: WorkQueue | None,
        schedule_token_throttle: MemoryResource | None,
    ) -> None:
        """Schedule split K/V loads with only the supplied optional resources."""
        decode_work_tile_schedule(
            cfg,
            work_queue,
            lambda: load_schedule_body(
                smem_q,
                smem_k0,
                smem_k1,
                smem_v0,
                smem_v1,
                sparse_kv_metadata0,
                sparse_kv_metadata1,
                sparse_softmax_metadata0,
                sparse_softmax_metadata1,
                smem_page_offsets,
                smem_page_offsets_v,
                schedule_token_throttle,
                sage_k_scales0,
                sage_k_scales1,
            ),
            lambda: schedule_token_throttle_head(schedule_token_throttle),
        )

    for smem_k, sparse_kv_metadata, sparse_softmax_metadata, sage_k_scales in (
        (smem_k0, sparse_kv_metadata0, sparse_softmax_metadata0, sage_k_scales0),
        (smem_k1, sparse_kv_metadata1, sparse_softmax_metadata1, sage_k_scales1),
    ):
        if smem_k is None and (
            sparse_kv_metadata is not None or sparse_softmax_metadata is not None
        ):
            raise ValueError("inactive K/V instances cannot own sparse metadata")
        if sparse_softmax_metadata is not None and sparse_kv_metadata is None:
            raise ValueError("Softmax sparse metadata requires retained KV metadata")
        if sage_k_scales is not None and smem_k is None:
            raise ValueError("a Sage K scale ring follows its instance's K tiles")
    if sparse_kv_metadata0 is not None or sparse_kv_metadata1 is not None:
        if smem_page_offsets is not None or smem_page_offsets_v is not None:
            raise ValueError(
                "block-sparse and paged-KV load resources cannot be combined"
            )

    schedule_result = load_schedule(
        smem_q,
        smem_k0,
        smem_k1,
        smem_v0,
        smem_v1,
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
        smem_page_offsets,
        smem_page_offsets_v,
        sage_k_scales0,
        sage_k_scales1,
        work_queue,
        schedule_token_throttle,
    )
    src = []
    # Route resolution is ConsumerWork and K/V retention is ProducerWork on
    # the same pipeline-free resource, so register both sides of that route.
    for sparse_kv_metadata in (sparse_kv_metadata0, sparse_kv_metadata1):
        if sparse_kv_metadata is not None:
            src.append(sparse_kv_metadata)
    if smem_page_offsets is not None:
        src.append(smem_page_offsets)
    if smem_page_offsets_v is not None:
        src.append(smem_page_offsets_v)
    if work_queue is not None:
        src.append(work_queue)
    dst = [
        resource
        for resource in (smem_q, smem_k0, smem_k1, smem_v0, smem_v1)
        if resource is not None
    ]
    for resource in (
        sparse_kv_metadata0,
        sparse_kv_metadata1,
        sparse_softmax_metadata0,
        sparse_softmax_metadata1,
        sage_k_scales0,
        sage_k_scales1,
    ):
        if resource is not None:
            dst.append(resource)
    if schedule_token_throttle is not None:
        dst.append(schedule_token_throttle)
    q_bound_resources = tuple(
        (resource, resource is smem_q)
        for resource in (smem_q, smem_k0, smem_k1, smem_v0, smem_v1)
        if resource is not None
    )
    return task_class(
        src_resources=src,
        dst_resources=dst,
        q_bound_resources=q_bound_resources,
        cfg=cfg,
        warp_idx=cfg.load_warp_idx if warp_idx is None else warp_idx,
        num_warps=cfg.load_num_warps if num_warps is None else num_warps,
        schedule=schedule_result,
        num_registers=cfg.mma_load_task_num_registers,
        name=task_name,
        **kw,
    )


def prefetch_sparse_route(
    sparse_kv_metadata: MemoryResource | None,
    target: str,
) -> tuple[Any, Any] | None:
    """Issue one prepared route-record load; dense profiles get ``None``.

    Route records are loaded one resolution ahead so their global-memory
    latency overlaps other load-warp work instead of stalling it. Callers
    pass the result to the instance's next ``resolve_and_store_sparse_route``,
    the way ``staged_kv_load`` threads its cached page IDs.
    """

    if sparse_kv_metadata is None:
        return None
    return sparse_kv_metadata.prefetch_route(target=target)


def publish_k_scales(
    sage_k_scales: MemoryResource | None,
    route: tuple[Any, Any, Any, Any] | None,
    section: WorkStage,
) -> None:
    """Fill one ``sfK`` ring stage for the K tile just loaded.

    The words are the tile's atoms of the prepared scales, a block-sparse
    route's or a dense tile's; the ring's ``AsyncLoad`` commit completes the
    stage once the copies land.
    """

    if sage_k_scales is None:
        return
    sage_k_scales.acquire()
    if route is None:
        sage_k_scales.copy_tile(section=section)
    else:
        resolved_record_word, resolved_origin1, _, _ = route
        sage_k_scales.copy_route(
            resolved_record_word=resolved_record_word,
            resolved_origin1=resolved_origin1,
        )
    sage_k_scales.commit()


def publish_sparse_softmax_route(
    sparse_softmax_metadata: MemoryResource | None,
    route: tuple[Any, Any, Any, Any] | None,
) -> None:
    """Stage one resolved route for its paired Softmax consumer."""

    if sparse_softmax_metadata is None:
        return
    assert route is not None
    (
        resolved_record_word,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    ) = route
    sparse_softmax_metadata.acquire()
    sparse_softmax_metadata.store_route(
        resolved_record_word=resolved_record_word,
        resolved_origin1=resolved_origin1,
        resolved_atom_validity=resolved_atom_validity,
        route_record_word_offset=route_record_word_offset,
    )
    sparse_softmax_metadata.commit()


def resolve_and_store_sparse_route(
    sparse_kv_metadata: MemoryResource | None,
    section: WorkStage,
    prefetch: tuple[Any, Any] | None,
) -> tuple[Any, Any, Any, Any] | None:
    """Resolve one prepared route and retain it for the matching K/V pair.

    ``prefetch`` is the record load issued earlier by
    ``prefetch_sparse_route``. Dense profiles pass ``None`` and get ``None``.
    """

    if sparse_kv_metadata is None:
        return None
    assert prefetch is not None
    prefetched_record_word, prefetched_record_offset = prefetch
    (
        resolved_record_word,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    ) = sparse_kv_metadata.resolve_route(
        section=section,
        prefetched_record_word_slot=prefetched_record_word,
        prefetched_record_offset_slot=prefetched_record_offset,
    )
    sparse_kv_metadata.store_route(
        resolved_record_word=resolved_record_word,
        resolved_origin1=resolved_origin1,
        resolved_atom_validity=resolved_atom_validity,
        route_record_word_offset=route_record_word_offset,
    )
    return (
        resolved_record_word,
        resolved_origin1,
        resolved_atom_validity,
        route_record_word_offset,
    )


def staged_kv_load(
    resource: MemoryResource,
    label: str,
    section: WorkStage,
    cfg: FmhaDecodeConfig,
    *,
    smem_page_offsets: MemoryResource | None = None,
    page_offsets_label: str = "read_offsets",
    manage_page_offsets: bool = True,
    cached_page_ids: Any = None,
) -> Any:
    """Issue all head-dim stages for one logical K/V load.

    H256 SwapsMmaAb uses two 128-wide K/V stages, but both slices address the
    same token tile and therefore the same page IDs. Keep one page-offset
    consumer stage live across the complete logical K/V load.
    """
    reuse_page_ids = (
        smem_page_offsets is not None
        and cfg.num_head_dim_stages_kv > 1
        and not cfg.uses_scattered_page_route
    )
    # Optional ConsWait/ConsWork: fetch page IDs for this logical K/V tile.
    # The cached D256 path performs its ConsumerWork below while materializing
    # the register array; single-stage kernels keep the original no-cache path.
    if manage_page_offsets:
        if reuse_page_ids:
            smem_page_offsets.wait()
        else:
            page_offsets_consume(smem_page_offsets, page_offsets_label)
    load_label = label
    if reuse_page_ids:
        inst_id = KV_INST1 if label.endswith("1") else KV_INST0
        kv_kind = KV_KIND_V if "_v" in label else KV_KIND_K
        cached_page_ids = smem_page_offsets.cache_page_ids(
            cached_page_ids=cached_page_ids,
            inst_id=inst_id,
            kv_kind=kv_kind,
            section=section,
        )
        load_label = f"{label}_cached"
    for head_dim_stage_idx in range(cfg.num_raw_kv_head_dim_stages):
        # ProdAcquire/ProdWork/ProdCommit: issue one K or V slice into the
        # matching SMEM stage.
        resource.acquire()
        if cached_page_ids is None:
            getattr(resource, load_label)(
                section=section, head_dim_stage_idx=head_dim_stage_idx
            )
        else:
            getattr(resource, load_label)(
                cached_page_ids=cached_page_ids,
                section=section,
                head_dim_stage_idx=head_dim_stage_idx,
            )
        resource.commit()
    # ConsRelease: every head-dimension slice consumed the staged page IDs.
    if manage_page_offsets:
        page_offsets_release(smem_page_offsets)
    return cached_page_ids
