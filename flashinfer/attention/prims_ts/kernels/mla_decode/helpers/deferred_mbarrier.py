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

"""Deferred mbarrier initialization for TaskManager-scheduled MLA kernels.

A kernel may initialize every active mbarrier from one staged region after
TaskManager assigns the unified-SMEM pointers, instead of letting each
CUTLASS pipeline constructor emit its own ``mbarrier.init`` ops.  Opt in per
kernel via ``cfg.uses_deferred_mbarrier_init`` and route resource pipeline
construction through :class:`DeferredPipelineMixin`.

Public surface used by the kernels:

- :class:`DeferredPipelineMixin`
- :func:`prepare_deferred_mbarriers`
- :func:`initialize_deferred_mbarriers`
"""

from dataclasses import replace

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
from cutlass import Int32, Int64
from cutlass.experimental.task_scheduling.enums import PipelineType
from cutlass.pipeline.helpers import _get_thread_arrive_count

# Deferred mbarrier init: a kernel may initialize every active mbarrier from
# one staged region after TaskManager assigns the unified-SMEM pointers.  This
# relies on private CUTLASS pipeline internals, so pin that surface the first
# time a deferred pipeline is built and let a DSL upgrade fail loudly there
# instead of producing a wrong arrival count and a silent hang.  The check is
# deliberately not run at import time so that a mismatch (or unavailable
# source text) only affects kernels that opt in via
# ``cfg.uses_deferred_mbarrier_init``.
_DEFERRED_MBARRIER_FIELDS = frozenset(
    {
        "barrier_storage",
        "op_type",
        "cg",
        "num_stages",
        "tx_count",
        "arrive_count",
        "mbarrier_base",
        "mbarrier_layout",
        "name",
    }
)

_PRIVATE_PIPELINE_SURFACE_CHECKED = False


def _check_private_pipeline_surface() -> None:
    import inspect
    import re

    try:
        recast_source = inspect.getsource(pipeline.MbarrierArray.recast_to_new_op_type)
    except OSError as exc:
        raise RuntimeError(
            "deferred mbarrier init cannot verify the private nvidia-cutlass-dsl "
            "pipeline surface because the source of "
            "MbarrierArray.recast_to_new_op_type is unavailable (pyc-only or "
            "zipped install); install nvidia-cutlass-dsl with source files or "
            "disable cfg.uses_deferred_mbarrier_init"
        ) from exc
    recast_fields = frozenset(
        re.findall(r"new_mbarrier_array\.(\w+)\s*=", recast_source)
    )
    missing_helpers = [
        f"{owner.__name__}.{attr}"
        for owner, attr in (
            (pipeline.PipelineTmaUmma, "_compute_mcast_arrival_mask"),
            (pipeline.PipelineTmaUmma, "_compute_is_leader_cta"),
            (pipeline.PipelineUmmaAsync, "_compute_tmem_sync_mask"),
            (pipeline.PipelineUmmaAsync, "_compute_peer_cta_rank"),
            (pipeline.PipelineClcFetchAsync, "_init_full_barrier_arrive_signal"),
        )
        if not hasattr(owner, attr)
    ]
    if recast_fields != _DEFERRED_MBARRIER_FIELDS or missing_helpers:
        raise RuntimeError(
            "nvidia-cutlass-dsl changed the private pipeline surface mirrored by "
            "the deferred mbarrier init: MbarrierArray fields "
            f"{sorted(recast_fields ^ _DEFERRED_MBARRIER_FIELDS)} differ and/or "
            f"helpers {missing_helpers} are missing; update "
            "_make_deferred_mbarrier_array / _create_deferred_pipeline"
        )


def _ensure_private_pipeline_surface() -> None:
    """Run the private-surface check once, on first deferred pipeline build."""

    global _PRIVATE_PIPELINE_SURFACE_CHECKED
    if _PRIVATE_PIPELINE_SURFACE_CHECKED:
        return
    _check_private_pipeline_surface()
    _PRIVATE_PIPELINE_SURFACE_CHECKED = True


def _make_deferred_mbarrier_array(
    barrier_storage,
    num_stages: int,
    agent,
    tx_count: int = 0,
    name: str = "",
):
    """Build ``MbarrierArray`` state without emitting constructor init ops.

    The field set written here is checked on the first deferred pipeline build
    against ``MbarrierArray.recast_to_new_op_type``.  initializes all active
    arrays together once TaskManager has assigned their unified-SMEM pointers.
    """

    sync = object.__new__(pipeline.MbarrierArray)
    sync.barrier_storage = barrier_storage
    sync.tx_count = tx_count
    sync.num_stages = num_stages
    sync.op_type, sync.cg = agent
    sync.arrive_count = _get_thread_arrive_count(sync.cg)
    sync.mbarrier_layout = pipeline.MbarrierLayout.V0
    sync.name = name
    sync.mbarrier_base = barrier_storage
    return sync


def _pipeline_cta_layout(pipeline_config):
    layout = pipeline_config.cta_layout_vmnk
    if layout is None:
        return cute.make_layout((1, 1, 1, 1))
    if not isinstance(layout, cute.Layout):
        return cute.make_layout(layout)
    return layout


def _create_deferred_pipeline(resource, pipeline_config):
    """Build the active CUTLASS pipeline objects with all barrier init deferred."""

    _ensure_private_pipeline_surface()

    barrier_ptr = pipeline_config.barrier_ptr
    if barrier_ptr is None:
        raise ValueError("deferred mbarrier init requires unified barrier storage")
    stages = pipeline_config.num_stages
    full_ptr = barrier_ptr.align(min_align=8)
    empty_ptr = full_ptr + stages
    pipeline_type = pipeline_config.pipeline_type

    if pipeline_type == PipelineType.AsyncAsync:
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline_config.async_producer_op, pipeline_config.producer_group),
            name=f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        return pipeline.PipelineAsync(full, empty, stages, None, None)

    if pipeline_type == PipelineType.TmaUmma:
        cta_layout = _pipeline_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.TmaLoad, pipeline_config.producer_group),
            pipeline_config.num_bytes,
            f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.TCGen05Mma, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        if cute.size(cta_layout) == 1:
            producer_mask = None
            is_leader_cta = True
        else:
            producer_mask = pipeline.PipelineTmaUmma._compute_mcast_arrival_mask(
                cta_layout, pipeline_config.mcast_mode_mn
            )
            is_leader_cta = pipeline.PipelineTmaUmma._compute_is_leader_cta(cta_layout)
        consumer_mask = producer_mask
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cute.size(cta_layout, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )
        result = pipeline.PipelineTmaUmma(
            full,
            empty,
            stages,
            producer_mask,
            consumer_mask,
            is_leader_cta,
            cta_group,
        )
        return resource._apply_task_warp_leader_to_tma_umma(result)

    if pipeline_type == PipelineType.UmmaAsync:
        cta_layout = _pipeline_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.TCGen05Mma, pipeline_config.producer_group),
            name=f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        producer_mask = (
            None
            if cute.size(cta_layout) == 1
            else pipeline.PipelineUmmaAsync._compute_tmem_sync_mask(cta_layout)
        )
        consumer_mask = (
            None
            if cute.size(cta_layout, mode=[0]) == 1
            else pipeline.PipelineUmmaAsync._compute_peer_cta_rank()
        )
        cta_group = (
            cute.nvgpu.tcgen05.CtaGroup.ONE
            if cute.size(cta_layout, mode=[0]) == 1
            else cute.nvgpu.tcgen05.CtaGroup.TWO
        )
        return pipeline.PipelineUmmaAsync(
            full,
            empty,
            stages,
            producer_mask,
            consumer_mask,
            cta_group,
        )

    if pipeline_type == PipelineType.ClcFetchAsync:
        cta_layout = _pipeline_cta_layout(pipeline_config)
        full = _make_deferred_mbarrier_array(
            full_ptr,
            stages,
            (pipeline.PipelineOp.ClcLoad, pipeline_config.producer_group),
            pipeline_config.num_bytes,
            f"{resource.name}.full",
        )
        empty = _make_deferred_mbarrier_array(
            empty_ptr,
            stages,
            (pipeline.PipelineOp.AsyncThread, pipeline_config.consumer_group),
            name=f"{resource.name}.empty",
        )
        tidx, _, _ = cute.arch.thread_idx()
        producer_mask, is_signaling_thread = (
            pipeline.PipelineClcFetchAsync._init_full_barrier_arrive_signal(
                cta_layout, tidx
            )
        )
        return pipeline.PipelineClcFetchAsync(
            full,
            empty,
            stages,
            producer_mask,
            0,
            is_signaling_thread,
        )

    raise ValueError(
        f"deferred mbarrier init does not support pipeline type {pipeline_type}"
    )


def _arrival_count_runs(arrival_counts: tuple[int, ...]):
    """Compress a static per-slot count table for compact scalar selects."""

    runs = []
    start = 0
    current = arrival_counts[0]
    for index, count in enumerate(arrival_counts[1:], 1):
        if count != current:
            runs.append((start, index, current))
            start = index
            current = count
    runs.append((start, len(arrival_counts), current))
    return tuple(runs)


def _select_arrival_count(slot, arrival_counts: tuple[int, ...]):
    result = Int32(arrival_counts[0])
    for start, end, count in _arrival_count_runs(arrival_counts)[1:]:
        in_run = slot >= Int32(start)
        if end != len(arrival_counts):
            in_run = in_run & (slot < Int32(end))
        result = Int32(cutlass.select_(in_run, Int32(count), result))
    return result


def prepare_deferred_mbarriers(task_manager):
    """Allocate and bind one contiguous barrier block before resource create.

    This kernel supplies its payload arrays from the kernel prologue, so
    TaskManager has no unified ``SmemAllocator``.  Binding the barrier
    storage explicitly makes contiguity a contract rather than an
    accident of static-allocation placement.
    """

    barrier_resources = tuple(
        resource
        for resource in task_manager.resources
        if resource.pipeline_config is not None and resource.pipeline_group is None
    )
    total_slots = sum(
        2 * resource.pipeline_config.num_stages for resource in barrier_resources
    )
    if total_slots <= 0 or total_slots > 64:
        raise ValueError(
            f"deferred mbarrier init requires 1..64 active slots, got {total_slots}"
        )
    barrier_storage = cutlass.Array(
        Int64,
        total_slots,
        space=cutlass.AddressSpace.smem,
        alignment=8,
    )
    base_ptr = cute.make_ptr(
        Int64,
        barrier_storage.data_ptr(),
        mem_space=cutlass.AddressSpace.smem,
    )
    offset = 0
    for resource in barrier_resources:
        pipeline_config = resource.pipeline_config
        resource.pipeline_config = replace(
            pipeline_config,
            barrier_ptr=base_ptr + offset,
        )
        offset += 2 * pipeline_config.num_stages
    task_manager._deferred_barrier_resources = barrier_resources
    task_manager._deferred_barrier_storage = barrier_storage
    return barrier_storage


@cute.jit
def _initialize_deferred_mbarrier_block(
    base_ptr,
    arrival_counts: cutlass.Constexpr,
) -> None:
    """Emit the single staged W0 initialization region."""

    warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
    if warp_idx == 0:
        tidx, _, _ = cute.arch.thread_idx()
        lane = tidx & Int32(31)
        first_slots = min(len(arrival_counts), 32)
        if first_slots == 32:
            cute.arch.mbarrier_init(
                base_ptr + lane,
                _select_arrival_count(lane, arrival_counts),
            )
        else:
            if lane < Int32(first_slots):
                cute.arch.mbarrier_init(
                    base_ptr + lane,
                    _select_arrival_count(lane, arrival_counts),
                )
        if len(arrival_counts) > 32:
            second_slot = lane + Int32(32)
            if lane < Int32(len(arrival_counts) - 32):
                cute.arch.mbarrier_init(
                    base_ptr + second_slot,
                    _select_arrival_count(second_slot, arrival_counts),
                )


def initialize_deferred_mbarriers(task_manager) -> None:
    """Initialize the explicitly contiguous barrier block from W0 lanes."""

    barrier_resources = task_manager._deferred_barrier_resources
    if not barrier_resources:
        return

    arrival_counts = []
    for resource in barrier_resources:
        pipeline_object = resource.pipeline
        stages = resource.pipeline_config.num_stages
        full_count = pipeline_object.sync_object_full.arrive_count
        empty_count = pipeline_object.sync_object_empty.arrive_count
        if not isinstance(full_count, int) or not isinstance(empty_count, int):
            raise ValueError("deferred mbarrier init requires static arrival counts")
        arrival_counts.extend([full_count] * stages)
        arrival_counts.extend([empty_count] * stages)
    arrival_counts = tuple(arrival_counts)
    if len(arrival_counts) > 64:
        raise ValueError("deferred mbarrier init supports at most 64 active slots")

    base_ptr = barrier_resources[0].pipeline_config.barrier_ptr.align(min_align=8)
    _initialize_deferred_mbarrier_block(base_ptr, arrival_counts)


class DeferredPipelineMixin:
    """Route pipeline construction through the deferred-init path.

    Every barrier-owning MLA resource inherits this once instead of carrying
    its own copy of the ``is_dynamic_token_sparse`` dispatch.
    """

    def create_pipeline(self, pipeline_config):
        cfg = getattr(self, "cfg", None)
        if cfg is not None and getattr(cfg, "uses_deferred_mbarrier_init", False):
            return _create_deferred_pipeline(self, pipeline_config)
        return super().create_pipeline(pipeline_config)
