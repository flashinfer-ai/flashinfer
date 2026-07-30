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

"""Pure-Python schedule construction tests for BatchedGemm TS tasks."""

import inspect
from dataclasses import dataclass
from types import SimpleNamespace

import cutlass
import cutlass.pipeline as pipeline
import pytest

from cutlass.experimental.task_scheduling.enums import (
    PipelineType,
    ScheduleStage,
    SignalingThreads,
    TileSchedulerType,
    WorkAttr,
)
from cutlass.experimental.task_scheduling.resources import (
    MemoryResource,
    PdlLaunchBarrier,
    PdlWaitBarrier,
    PipelineConfig,
    TileSchedulerConfig,
    WorkQueue,
)

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    BatchMode,
    BiasType,
    DType,
    RouteImpl,
    SfLayout,
    SfSmemToTmemCopy,
    TileScheduler,
    _ldgsts_sfb_producer_commit_prefetch_depth,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _make_pipeline_configs,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
    _runtime_config,
)
from flashinfer.prims_ts.batched_gemm.smem_misc_resources import (
    BatchedGemmWorkQueue,
    ProxyClusterBarrierResource,
)
from flashinfer.prims_ts.batched_gemm.smem_sf_resources import (
    SmemSfLdgstsBResource,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_tasks import (
    create_cast_a_task,
    create_copy_sfa_task,
    create_copy_sfab_task,
    create_copy_sfb_task,
    create_epilogue_task,
    create_gather_task,
    create_load_a_task,
    create_load_b_task,
    create_load_sfa_task,
    create_load_sfb_task,
    create_mma_task,
    create_padding_task,
    create_sync_task,
    create_workid_task,
)


from cutlass.experimental.task_scheduling import pipeline as ts_pipeline
from cutlass.experimental.task_scheduling.resources import (
    TaskLocalVariable,
    consumer_work as consumer_work_decorator,
    producer_work,
)


@dataclass(kw_only=True)
class _DummyResource(MemoryResource):
    """Stub resource for schedule-shape tests.

    Provides all named work methods that the schedule bodies reference so
    that the schedule builder can resolve labels.
    """

    coord_a_k: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_a_mn: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_a_l: TaskLocalVariable = TaskLocalVariable.uninitialized()
    expert_idx: TaskLocalVariable = TaskLocalVariable.uninitialized()
    mn_limit: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_b_k: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_b_mn: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_b_l: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_sfa_k: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_sfa_mn: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_sfb_k: TaskLocalVariable = TaskLocalVariable.uninitialized()
    coord_sfb_mn: TaskLocalVariable = TaskLocalVariable.uninitialized()
    desc_a_mma_base: TaskLocalVariable = TaskLocalVariable.uninitialized()
    smem_a_stage_ptr: TaskLocalVariable = TaskLocalVariable.uninitialized()
    desc_b_mma_base: TaskLocalVariable = TaskLocalVariable.uninitialized()
    smem_b_stage_ptr: TaskLocalVariable = TaskLocalVariable.uninitialized()
    desc_a_s2t_base: TaskLocalVariable = TaskLocalVariable.uninitialized()
    smem_sfa_stage_ptr: TaskLocalVariable = TaskLocalVariable.uninitialized()
    desc_b_s2t_base: TaskLocalVariable = TaskLocalVariable.uninitialized()
    sfa_stage_col_offset: TaskLocalVariable = TaskLocalVariable.uninitialized()
    sfb_stage_col_offset: TaskLocalVariable = TaskLocalVariable.uninitialized()
    tmem_cast_a_addr: TaskLocalVariable = TaskLocalVariable.uninitialized()
    t2r_rmem: TaskLocalVariable = TaskLocalVariable.uninitialized()
    t2r_rmem_1: TaskLocalVariable = TaskLocalVariable.uninitialized()
    t2r_output_call_idx: TaskLocalVariable = TaskLocalVariable.uninitialized()
    producer_commit_prefetch_depth: int = 0

    def __post_init__(self) -> None:
        int32 = cutlass.Int32(0)
        int64 = cutlass.Int64(0)
        f32 = cutlass.Float32(0.0)
        for name in (
            "coord_a_k",
            "coord_a_mn",
            "coord_a_l",
            "expert_idx",
            "mn_limit",
            "coord_b_k",
            "coord_b_mn",
            "coord_b_l",
            "coord_sfa_k",
            "coord_sfa_mn",
            "coord_sfb_k",
            "coord_sfb_mn",
            "sfa_stage_col_offset",
            "sfb_stage_col_offset",
            "tmem_cast_a_addr",
            "t2r_output_call_idx",
        ):
            setattr(
                self,
                name,
                TaskLocalVariable(dtype=cutlass.Int32, default=int32),
            )
        for name in (
            "desc_a_mma_base",
            "smem_a_stage_ptr",
            "desc_b_mma_base",
            "smem_b_stage_ptr",
            "desc_a_s2t_base",
            "smem_sfa_stage_ptr",
            "desc_b_s2t_base",
        ):
            setattr(
                self,
                name,
                TaskLocalVariable(dtype=cutlass.Int64, default=int64),
            )
        self.t2r_rmem = TaskLocalVariable(dtype=cutlass.Float32, default=f32)
        self.t2r_rmem_1 = TaskLocalVariable(dtype=cutlass.Float32, default=f32)

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    def init_coords_state(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def init_load_state(self, stage_info):
        pass

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    def init_mma_state(self, stage_info):
        pass

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    def init_s2t_state(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def init_copy_state(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def init_accumulator_state(self, stage_info):
        pass

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    def init_epilogue_state(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def init_store_state(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def prepare_gather_tile(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def prepare_sfa_tile(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def prepare_sfb_tile(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def drain_loop(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def drain_tail(self, stage_info, *, prefetch_idx: cutlass.Constexpr[int]):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def sync_compact_sfb_copy(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def sync_sttm_copy(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def sync_cast_a_warps(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def advance_mma_overlap_window(self, stage_info):
        pass

    @producer_work(work_attrs=WorkAttr.AUXILIARY)
    def init_epilogue_tile_state(self, stage_info):
        pass

    # Gmem-style consumer work
    @consumer_work_decorator(
        returns=("coord_a_k", "coord_a_mn", "coord_a_l", "expert_idx", "mn_limit")
    )
    def compute_a_coords_head(self, stage_info):
        pass

    @consumer_work_decorator(
        returns=("coord_a_k", "coord_a_mn", "coord_a_l", "expert_idx", "mn_limit")
    )
    def compute_a_coords_loop(self, stage_info):
        pass

    @consumer_work_decorator(
        returns=("coord_b_k", "coord_b_mn", "coord_b_l", "mn_limit")
    )
    def compute_b_coords_head(self, stage_info):
        pass

    @consumer_work_decorator(
        returns=("coord_b_k", "coord_b_mn", "coord_b_l", "mn_limit")
    )
    def compute_b_coords_loop(self, stage_info):
        pass

    @consumer_work_decorator(returns=("coord_sfa_k", "coord_sfa_mn"))
    def compute_sfa_coords_head(
        self, stage_info, *, prefetch_idx: cutlass.Constexpr[int]
    ):
        pass

    @consumer_work_decorator(returns=("coord_sfa_k", "coord_sfa_mn"))
    def compute_sfa_coords_loop(self, stage_info):
        pass

    @consumer_work_decorator(returns=("coord_sfb_k", "coord_sfb_mn"))
    def compute_sfb_coords_head(
        self, stage_info, *, prefetch_idx: cutlass.Constexpr[int]
    ):
        pass

    @consumer_work_decorator(work_attrs=WorkAttr.AUXILIARY)
    def init_tile_state(self, stage_info):
        pass

    @consumer_work_decorator(returns=("coord_sfb_k", "coord_sfb_mn"))
    def compute_sfb_coords_loop(self, stage_info):
        pass

    # Smem-style producer work
    @producer_work()
    def load_a_tile(
        self,
        stage_info,
        *,
        coord_a_k,
        coord_a_mn,
        coord_a_l,
        expert_idx,
        mn_limit,
    ):
        pass

    @producer_work()
    def load_b_tile(self, stage_info, *, coord_b_k, coord_b_mn, coord_b_l, mn_limit):
        pass

    @producer_work()
    def load_sfa_tile(self, stage_info, *, coord_sfa_k, coord_sfa_mn):
        pass

    @producer_work()
    def load_sfb_tile(self, stage_info, *, coord_sfb_k, coord_sfb_mn):
        pass

    # SmemSf consumer
    @consumer_work_decorator(returns=("desc_a_s2t_base", "smem_sfa_stage_ptr"))
    def build_sfa_s2t_desc(self, stage_info):
        pass

    @consumer_work_decorator(returns=("desc_b_s2t_base",))
    def build_sfb_s2t_desc(self, stage_info):
        pass

    # SmemAB consumer
    @consumer_work_decorator(returns=("desc_a_mma_base", "smem_a_stage_ptr"))
    def build_mma_desc_a(self, stage_info):
        pass

    @consumer_work_decorator(returns=("desc_a_mma_base", "smem_a_stage_ptr"))
    def build_mma_desc_a_at_stage(self, stage_info, *, pipeline_stage_idx):
        pass

    @consumer_work_decorator(returns=("desc_b_mma_base", "smem_b_stage_ptr"))
    def build_mma_desc_b(self, stage_info):
        pass

    @consumer_work_decorator(returns=("desc_b_mma_base", "smem_b_stage_ptr"))
    def build_mma_desc_b_at_stage(self, stage_info, *, pipeline_stage_idx):
        pass

    # TmemSf producer/consumer
    @producer_work()
    def copy_sfa(self, stage_info, *, desc_a_s2t_base, smem_sfa_stage_ptr):
        pass

    @producer_work()
    def copy_sfb(self, stage_info, *, desc_b_s2t_base):
        pass

    @producer_work()
    def copy_sfab(
        self, stage_info, *, desc_a_s2t_base, smem_sfa_stage_ptr, desc_b_s2t_base
    ):
        pass

    @producer_work()
    def cast_a(self, stage_info, *, smem_a_stage_ptr, smem_sfa_stage_ptr):
        pass

    @consumer_work_decorator(returns=("sfa_stage_col_offset",))
    def publish_sfa_offset(self, stage_info):
        pass

    @consumer_work_decorator(returns=("sfb_stage_col_offset",))
    def publish_sfb_offset(self, stage_info):
        pass

    @consumer_work_decorator(returns=("sfa_stage_col_offset", "sfb_stage_col_offset"))
    def publish_sfab_offset(self, stage_info):
        pass

    @consumer_work_decorator(returns=("tmem_cast_a_addr",))
    def publish_cast_a_addr(self, stage_info):
        pass

    # TmemC consumer
    @consumer_work_decorator(returns=("t2r_rmem", "t2r_rmem_1", "t2r_output_call_idx"))
    def consumer_work(self, stage_info, *, subtile_idx: cutlass.Constexpr[int] = 0):
        # ``subtile_idx`` is optional so this stub stands in for both the named
        # TmemC.consumer_work (which passes subtile_idx) and the standard
        # ``proxy.consumer_work()`` pipeline op (which passes nothing).
        pass

    @consumer_work_decorator(
        returns=("t2r_rmem", "t2r_rmem_1", "t2r_output_call_idx"),
        work_attrs=WorkAttr.AUXILIARY,
    )
    def load_overlap_subtile(self, stage_info, *, subtile_idx: cutlass.Constexpr[int]):
        pass

    # TmemC
    @producer_work()
    def mma(
        self,
        stage_info,
        *,
        desc_a_mma_base,
        smem_a_stage_ptr,
        desc_b_mma_base,
        smem_b_stage_ptr,
    ):
        pass

    @producer_work()
    def mma_fused_sf(
        self,
        stage_info,
        *,
        desc_a_mma_base,
        smem_a_stage_ptr,
        desc_b_mma_base,
        smem_b_stage_ptr,
        desc_a_s2t_base,
        smem_sfa_stage_ptr,
        desc_b_s2t_base,
    ):
        pass

    @producer_work()
    def mma_separate_sf(
        self,
        stage_info,
        *,
        desc_a_mma_base,
        smem_a_stage_ptr,
        desc_b_mma_base,
        smem_b_stage_ptr,
        sfa_stage_col_offset,
        sfb_stage_col_offset,
    ):
        pass

    @producer_work()
    def mma_cast_a(
        self, stage_info, *, desc_b_mma_base, smem_b_stage_ptr, tmem_cast_a_addr
    ):
        pass

    # GmemC
    @producer_work()
    def store_epilogue(
        self,
        stage_info,
        *,
        t2r_rmem,
        t2r_rmem_1,
        t2r_output_call_idx,
        subtile_idx: cutlass.Constexpr[int],
    ):
        pass

@dataclass(kw_only=True)
class _DummyProxyResource(MemoryResource):
    """Proxy stub whose consumer publishes one pipeline-stage token."""

    consumer_stage_idx: TaskLocalVariable = TaskLocalVariable.uninitialized()

    def __post_init__(self) -> None:
        self.consumer_stage_idx = TaskLocalVariable(
            dtype=cutlass.Int32,
            default=cutlass.Int32(0),
        )

    @consumer_work_decorator(returns=consumer_stage_idx)
    def consumer_work(self, stage_info):
        pass

def _res(name: str, **kwargs) -> MemoryResource:
    if name == "Proxy":
        return _DummyProxyResource(name=name, **kwargs)
    return _DummyResource(name=name, **kwargs)

def _work_queue() -> WorkQueue:
    scheduler_config = TileSchedulerConfig(
        tile_scheduler_type=TileSchedulerType.ClcDynamicPersistent,
        tile_scheduler_params=None,
    )
    return BatchedGemmWorkQueue(
        tile_scheduler_config=scheduler_config,
        cfg=SimpleNamespace(use_early_exit=False, is_swap_ab=False),
        name="WorkQueue",
    )


def test_fast_drain_work_queue_reserves_response_and_barrier_smem():
    scheduler_config = TileSchedulerConfig(
        tile_scheduler_type=TileSchedulerType.ClcDynamicPersistent,
        tile_scheduler_params=None,
    )
    work_queue = BatchedGemmWorkQueue(
        tile_scheduler_config=scheduler_config,
        cfg=SimpleNamespace(
            is_persistent=True,
            use_early_exit=True,
            use_clc_fast_drain=True,
            is_swap_ab=False,
        ),
        name="WorkQueue",
    )

    allocations = work_queue.get_smem_requirements()

    assert [
        (alloc.name, alloc.size_bytes, alloc.alignment) for alloc in allocations
    ] == [
        ("WorkQueue_fast_drain_response", 64, 16),
        ("WorkQueue_fast_drain_mbar", 8, 8),
    ]


@pytest.mark.parametrize(
    ("is_persistent", "use_early_exit", "use_clc_fast_drain"),
    (
        (False, True, True),
        (True, False, True),
        (True, True, False),
    ),
)
def test_fast_drain_work_queue_omits_smem_when_disabled(
    is_persistent, use_early_exit, use_clc_fast_drain
):
    scheduler_config = TileSchedulerConfig(
        tile_scheduler_type=TileSchedulerType.ClcDynamicPersistent,
        tile_scheduler_params=None,
    )
    work_queue = BatchedGemmWorkQueue(
        tile_scheduler_config=scheduler_config,
        cfg=SimpleNamespace(
            is_persistent=is_persistent,
            use_early_exit=use_early_exit,
            use_clc_fast_drain=use_clc_fast_drain,
            is_swap_ab=False,
        ),
        name="WorkQueue",
    )

    assert work_queue.get_smem_requirements() == []


def _producer_aux_labels(schedule_list) -> list[str | None]:
    return [
        label
        for _resource, stage, _call_idx, *rest in schedule_list
        for label in [rest[-1]]
        if stage == ScheduleStage.ProducerAuxWork
    ]


def _all_producer_aux_labels(task) -> list[str | None]:
    return (
        _producer_aux_labels(task.head_schedule_list)
        + _producer_aux_labels(task.loop_schedule_list)
        + _producer_aux_labels(task.tail_schedule_list)
    )


def _cfg(**overrides) -> SimpleNamespace:
    values = {
        "tile_scheduler": int(TileScheduler.PERSISTENT),
        "is_persistent": True,
        "load_a_warp_idx": 0,
        "num_load_a_warps": 1,
        "load_a_task_regs": 24,
        "load_b_warp_idx": 1,
        "num_load_b_warps": 1,
        "load_b_task_regs": 24,
        "gather_warp_idx": 2,
        "num_gather_warps": 1,
        "gather_regs": 24,
        "sync_regs": 24,
        "has_routed_sfs": True,
        "uses_ldgsts_routed_sfs": True,
        "uses_tma_routed_sfs": False,
        "has_deepseek_fp8": False,
        "has_cluster": True,
        "sfb_smem_to_tmem_copy": int(SfSmemToTmemCopy.UTCCP),
        "smem_sfb_layout": int(SfLayout.R128c4),
        "load_sfa_warp_idx": 3,
        "num_load_sfa_warps": 1,
        "load_sfa_task_regs": 24,
        "load_sfb_warp_idx": 4,
        "num_load_sfb_warps": 1,
        "load_sfb_task_regs": 24,
        "load_sfab_warp_idx": 12,
        "num_load_sfab_warps": 0,
        "load_sfab_regs": 48,
        "copy_sfa_warp_idx": 5,
        "num_copy_sfa_warps": 1,
        "copy_sfa_task_regs": 24,
        "copy_sfb_warp_idx": 6,
        "num_copy_sfb_warps": 1,
        "copy_sfb_task_regs": 24,
        "cast_a_warp_idx_first": 7,
        "num_cast_a_warps": 1,
        "cast_a_regs": 24,
        "mma_warp_idx": 8,
        "num_mma_warps": 1,
        "mma_regs": 24,
        "use_tile256_tmem_overlap": True,
        "num_epilogue_warps": 4,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "is_swap_ab": False,
        "epi_tile_n": 32,
        "tile_n": 64,
        "epilogue_warp_idx": 9,
        "epilogue_regs": 24,
        "workid_warp_idx": 10,
        "num_workid_warps": 1,
        "workid_regs": 24,
        "padding_warp_idx": 11,
        "num_padding_warps": 1,
        "padding_regs": 24,
        "use_early_exit": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_persistent_task_factories_build_captured_schedule_lists():
    cfg = _cfg(use_early_exit=True)
    work_queue = _work_queue()
    work_throttle = _res("WorkThrottle")
    pdl_wait = PdlWaitBarrier(name="PdlWait")
    pdl_launch = PdlLaunchBarrier(name="PdlLaunch")
    num_k_tiles = 5

    load_a = create_load_a_task(
        cfg,
        _res("GmemA"),
        _res("SmemA"),
        work_queue,
        num_k_tiles,
        work_throttle=work_throttle,
        pdl_wait_resource=pdl_wait,
        pdl_launch_resource=pdl_launch,
    )
    load_sfb = create_load_sfb_task(
        cfg,
        _res("GmemSfB"),
        _res("SmemSfB"),
        work_queue,
        num_k_tiles,
        pdl_wait_resource=pdl_wait,
        pdl_launch_resource=pdl_launch,
    )
    workid = create_workid_task(
        cfg,
        work_queue,
        num_k_tiles,
        work_throttle=work_throttle,
    )

    tasks = [
        load_a,
        create_load_b_task(
            cfg,
            _res("GmemB"),
            _res("SmemB"),
            work_queue,
            num_k_tiles,
            work_throttle=work_throttle,
            pdl_wait_resource=pdl_wait,
            pdl_launch_resource=pdl_launch,
        ),
        create_gather_task(
            cfg,
            _res("GmemAct"),
            _res("SmemGather"),
            work_queue,
            num_k_tiles,
            pdl_wait_resource=pdl_wait,
            pdl_launch_resource=pdl_launch,
        ),
        create_sync_task(
            cfg,
            _res("Proxy"),
            _res("GatherSmem"),
            _res("TmaSmem"),
            work_queue,
            num_k_tiles,
            sync_warp_idx=12,
        ),
        create_load_sfa_task(
            cfg,
            _res("GmemSfA"),
            _res("SmemSfA"),
            work_queue,
            num_k_tiles,
            pdl_wait_resource=pdl_wait,
            pdl_launch_resource=pdl_launch,
        ),
        load_sfb,
        create_copy_sfa_task(
            cfg, _res("SmemSfA"), _res("TmemSfA"), work_queue, num_k_tiles
        ),
        create_copy_sfb_task(
            cfg, _res("SmemSfB"), _res("TmemSfB"), work_queue, num_k_tiles
        ),
        create_copy_sfab_task(
            cfg,
            _res("SmemSfA"),
            _res("SmemSfB"),
            _res("TmemSfAb"),
            work_queue,
            num_k_tiles,
        ),
        create_cast_a_task(
            cfg,
            _res("SmemA"),
            _res("SmemSfA"),
            _res("TmemCastA"),
            work_queue,
            num_k_tiles,
        ),
        create_mma_task(
            cfg,
            _res("SmemA"),
            _res("SmemB"),
            _res("SmemSfA"),
            _res("SmemSfB"),
            _res("TmemC"),
            work_queue,
            num_k_tiles,
            proxy_cluster=_res("Proxy"),
        ),
        create_mma_task(
            cfg,
            _res("SmemA"),
            _res("SmemB"),
            None,
            None,
            _res("TmemC"),
            work_queue,
            num_k_tiles,
            tmem_sfa=_res("TmemSfA"),
            tmem_sfb=_res("TmemSfB"),
        ),
        create_mma_task(
            cfg,
            _res("SmemA"),
            _res("SmemB"),
            None,
            None,
            _res("TmemC"),
            work_queue,
            num_k_tiles,
            tmem_sfab=_res("TmemSfAb"),
        ),
        create_mma_task(
            cfg,
            _res("SmemA"),
            _res("SmemB"),
            None,
            None,
            _res("TmemC"),
            work_queue,
            num_k_tiles,
            tmem_cast_a=_res("TmemCastA"),
        ),
        create_epilogue_task(
            cfg, _res("TmemC"), _res("GmemC"), work_queue, num_k_tiles
        ),
        workid,
        create_padding_task(cfg, work_queue, num_k_tiles),
    ]

    assert len(tasks) == 17
    assert all(
        task.head_schedule_list or task.loop_schedule_list or task.tail_schedule_list
        for task in tasks
    )
    assert load_sfb.domain_start == 0
    assert (id(pdl_wait), ScheduleStage.ConsumerWork, 0) in (
        load_a.pre_work_loop_head_slots
    )
    assert (id(pdl_launch), ScheduleStage.ProducerWork, 0) in (
        load_a.post_work_loop_tail_slots
    )
    assert all(
        task.skip_if is BatchedGemmWorkQueue.should_skip_work_tile for task in tasks
    )
    assert load_a.skippable_head_slots
    assert any(slot[0] == id(work_throttle) for slot in workid.skippable_tail_slots)
    assert (id(work_queue), ScheduleStage.ConsumerWork, 0) not in (
        load_a.skippable_tail_slots
    )


def test_persistent_task_factories_omit_skip_when_early_exit_disabled():
    cfg = _cfg(use_early_exit=False)
    work_queue = _work_queue()
    load_a = create_load_a_task(
        cfg,
        _res("GmemA"),
        _res("SmemA"),
        work_queue,
        num_k_tiles=5,
    )

    assert load_a.skip_if is None
    assert load_a.skippable_head_slots == frozenset()
    assert load_a.skippable_tail_slots == frozenset()


def test_mma_task_rejects_partial_sf_pairs():
    with pytest.raises(ValueError, match="smem_sfa and smem_sfb together"):
        create_mma_task(
            _cfg(),
            _res("SmemA"),
            _res("SmemB"),
            _res("SmemSfA"),
            None,
            _res("TmemC"),
            _work_queue(),
            num_k_tiles=5,
        )

    with pytest.raises(ValueError, match="tmem_sfa and tmem_sfb together"):
        create_mma_task(
            _cfg(),
            _res("SmemA"),
            _res("SmemB"),
            None,
            None,
            _res("TmemC"),
            _work_queue(),
            num_k_tiles=5,
            tmem_sfa=_res("TmemSfA"),
        )


def test_mma_task_rejects_conflicting_sf_modes_and_cast_a():
    with pytest.raises(ValueError, match="at most one SF mode"):
        create_mma_task(
            _cfg(),
            _res("SmemA"),
            _res("SmemB"),
            _res("SmemSfA"),
            _res("SmemSfB"),
            _res("TmemC"),
            _work_queue(),
            num_k_tiles=5,
            tmem_sfa=_res("TmemSfA"),
            tmem_sfb=_res("TmemSfB"),
        )

    with pytest.raises(ValueError, match="forbids tmem_cast_a with SF modes"):
        create_mma_task(
            _cfg(),
            _res("SmemA"),
            _res("SmemB"),
            None,
            None,
            _res("TmemC"),
            _work_queue(),
            num_k_tiles=5,
            tmem_sfab=_res("TmemSfAb"),
            tmem_cast_a=_res("TmemCastA"),
        )


def test_load_a_try_acquire_is_per_k_tile():
    cfg = _cfg(use_early_exit=False)
    work_queue = _work_queue()
    smem_a = _res("SmemA")
    load_a = create_load_a_task(
        cfg,
        _res("GmemA"),
        smem_a,
        work_queue,
        num_k_tiles=5,
    )

    assert all(
        stage != ScheduleStage.ProducerTryAcquire
        for _, stage, *_ in load_a.head_schedule_list
    )
    assert any(
        resource is smem_a and stage == ScheduleStage.ProducerTryAcquire
        for resource, stage, *_ in load_a.loop_schedule_list
    )


def test_non_cluster_routed_sf_ldgsts_schedules_pre_commit_hook():
    cfg = _cfg(use_early_exit=False, has_cluster=False)
    work_queue = _work_queue()
    smem_sfa = _res("SmemSfA")
    load_sfa = create_load_sfa_task(
        cfg,
        _res("GmemSfA"),
        smem_sfa,
        work_queue,
        num_k_tiles=5,
    )

    assert any(
        resource is smem_sfa and stage == ScheduleStage.ProducerAuxWork
        for resource, stage, *_ in load_sfa.loop_schedule_list
    )
    assert any(
        resource is smem_sfa
        and stage == ScheduleStage.ProducerAuxWork
        and entry[-1] == "drain_loop"
        for entry in load_sfa.loop_schedule_list
        for resource, stage, *_ in [entry]
    )


def test_ldgsts_sf_pipeline_configs_use_async_load_producers():
    cfg = make_config(
        cluster_m=2,
        route_sfs_act=int(RouteImpl.LDGSTS),
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        tile_n=128,
    )
    smem_sfa_cfg = _make_pipeline_configs(cfg)["smem_sfa"]
    assert smem_sfa_cfg.pipeline_type == PipelineType.AsyncUmma
    assert smem_sfa_cfg.umma_consumer_producer_op == pipeline.PipelineOp.AsyncLoad
    assert smem_sfa_cfg.producer_group.size == cfg.num_load_sfa_warps * 32

    r128c4_sfb_cfg = make_config(
        cluster_m=2,
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=128,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    smem_sfa_cfg = _make_pipeline_configs(r128c4_sfb_cfg)["smem_sfa"]
    smem_sfb_cfg = _make_pipeline_configs(r128c4_sfb_cfg)["smem_sfb"]
    assert r128c4_sfb_cfg.use_combined_sfab_copy
    assert smem_sfa_cfg.advance_on_wait
    assert smem_sfb_cfg.pipeline_type == PipelineType.AsyncUmma
    assert smem_sfb_cfg.umma_consumer_producer_op == pipeline.PipelineOp.AsyncThread
    assert (
        smem_sfb_cfg.producer_group.size
        == r128c4_sfb_cfg.num_load_sfb_warps * 32 * r128c4_sfb_cfg.cluster_m
    )
    assert smem_sfb_cfg.advance_on_acquire
    assert smem_sfb_cfg.advance_on_wait
    assert _ldgsts_sfb_producer_commit_prefetch_depth(r128c4_sfb_cfg) == 3

    compact_cfg = make_config(
        cluster_m=2,
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_n=64,
    )
    smem_sfb_cfg = _make_pipeline_configs(compact_cfg)["smem_sfb"]
    assert compact_cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
    assert compact_cfg.smem_sfb_layout == int(SfLayout.R8c4)
    assert smem_sfb_cfg.pipeline_type == PipelineType.AsyncAsync
    assert smem_sfb_cfg.async_producer_op == pipeline.PipelineOp.AsyncLoad
    assert smem_sfb_cfg.producer_group.size == compact_cfg.num_load_sfb_warps * 32
    assert not smem_sfb_cfg.advance_on_acquire


def test_ts_async_umma_commit_uses_2sm_cluster_arrive():
    source = inspect.getsource(ts_pipeline.TSPipelineAsyncUmma.producer_commit)

    assert "0xFEFFFFFF" in source
    assert source.count("mbarrier.arrive.shared::cluster.b64") == 1


def test_ldgsts_sf_wait_group_depth_matches_generated_throttle():
    cfg = make_config(
        cluster_m=2,
        num_stages_smem_sfb=6,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=128,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    res = SmemSfLdgstsBResource(
        cfg=cfg,
        pipeline_config=_make_pipeline_configs(cfg)["smem_sfb"],
        producer_commit_prefetch_depth=_ldgsts_sfb_producer_commit_prefetch_depth(cfg),
        name="SmemSfB",
    )
    assert res.cp_async_wait_group_depth == 2

    two_stage_cfg = make_config(
        cluster_m=2,
        num_stages_smem_sfb=2,
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=128,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    two_stage_res = SmemSfLdgstsBResource(
        cfg=two_stage_cfg,
        pipeline_config=_make_pipeline_configs(two_stage_cfg)["smem_sfb"],
        producer_commit_prefetch_depth=_ldgsts_sfb_producer_commit_prefetch_depth(
            two_stage_cfg
        ),
        name="SmemSfB",
    )
    assert two_stage_res.cp_async_wait_group_depth == 0


def test_ldgsts_sfb_advance_on_acquire_prefetches_head_and_drains_tail():
    cfg = make_config(
        cluster_m=2,
        num_stages_smem_sfb=6,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=128,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    smem_sfb = SmemSfLdgstsBResource(
        name="SmemSfB",
        cfg=cfg,
        pipeline_config=_make_pipeline_configs(cfg)["smem_sfb"],
        producer_commit_prefetch_depth=_ldgsts_sfb_producer_commit_prefetch_depth(cfg),
    )
    load_sfb = create_load_sfb_task(
        cfg,
        _res("GmemSfB"),
        smem_sfb,
        _work_queue(),
        num_k_tiles=8,
    )

    head_stages = [
        stage
        for resource, stage, *_ in load_sfb.head_schedule_list
        if resource is smem_sfb
    ]
    loop_stages = [
        stage
        for resource, stage, *_ in load_sfb.loop_schedule_list
        if resource is smem_sfb
    ]
    tail_stages = [
        stage
        for resource, stage, *_ in load_sfb.tail_schedule_list
        if resource is smem_sfb
    ]

    assert head_stages.count(ScheduleStage.ProducerWork) == 3
    assert ScheduleStage.ProducerCommit not in head_stages
    loop_entries = [
        entry for entry in load_sfb.loop_schedule_list if entry[0] is smem_sfb
    ]
    tail_entries = [
        entry for entry in load_sfb.tail_schedule_list if entry[0] is smem_sfb
    ]

    assert loop_stages[:2] == [
        ScheduleStage.ProducerWork,
        ScheduleStage.ProducerCommit,
    ]
    assert loop_entries[0][-1] == "drain_loop"
    assert tail_stages == [
        ScheduleStage.ProducerWork,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ProducerWork,
        ScheduleStage.ProducerCommit,
        ScheduleStage.ProducerWork,
        ScheduleStage.ProducerCommit,
    ]
    assert [entry[-1] for entry in tail_entries[::2]] == ["drain_tail"] * 3
    assert [entry[2] for entry in tail_entries[::2]] == [0, 1, 2]


def test_ldgsts_sfa_prefetch_drains_tail_with_prefetch_ordinals():
    smem_sfa = _res(
        "SmemSfA",
        pipeline_config=SimpleNamespace(advance_on_acquire=True),
        producer_commit_prefetch_depth=3,
    )
    load_sfa = create_load_sfa_task(
        _cfg(),
        _res("GmemSfA"),
        smem_sfa,
        _work_queue(),
        num_k_tiles=8,
    )

    tail_entries = [
        entry for entry in load_sfa.tail_schedule_list if entry[0] is smem_sfa
    ]

    assert [entry[-1] for entry in tail_entries[::2]] == ["drain_tail"] * 3
    assert [entry[2] for entry in tail_entries[::2]] == [0, 1, 2]


def test_ldgsts_sfb_prefetch_depth_is_derived_from_host_k_tiles():
    cfg = make_config(
        cluster_m=2,
        num_stages_smem_sfb=6,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_k=256,
        tile_n=128,
        mma_n=16,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )

    assert _ldgsts_sfb_producer_commit_prefetch_depth(cfg) == 3

    validate_config(cfg, problem_mnk=(128, 128, 256))

    assert _ldgsts_sfb_producer_commit_prefetch_depth(cfg) == 1


def test_combined_tmem_sfab_uses_umma_producer_commit_pipeline():
    cfg = make_config(
        cluster_m=2,
        route_sfs_act=int(RouteImpl.LDGSTS),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        tile_n=128,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    assert cfg.use_combined_sfab_copy

    tmem_sfab_cfg = _make_pipeline_configs(cfg)["tmem_sfab"]
    assert tmem_sfab_cfg.pipeline_type == PipelineType.UmmaUmma
    assert tmem_sfab_cfg.producer_group.size == 1
    assert tmem_sfab_cfg.consumer_group.size == 1


def test_combined_tmem_sfab_delays_commit_and_smem_release():
    cfg = _cfg()
    smem_sfa = _res("SmemSfA")
    smem_sfb = _res("SmemSfB")
    tmem_sfab = _res("TmemSfAb")

    copy_sfab = create_copy_sfab_task(
        cfg,
        smem_sfa,
        smem_sfb,
        tmem_sfab,
        _work_queue(),
        num_k_tiles=4,
    )

    head_entries = [
        entry
        for entry in copy_sfab.head_schedule_list
        if entry[0] in {smem_sfa, smem_sfb, tmem_sfab}
        and entry[1] != ScheduleStage.ConsumerAuxWork
        and entry[1] != ScheduleStage.ProducerAuxWork
    ]
    loop_entries = [
        entry
        for entry in copy_sfab.loop_schedule_list
        if entry[0] in {smem_sfa, smem_sfb, tmem_sfab}
        and entry[1] != ScheduleStage.ConsumerAuxWork
        and entry[1] != ScheduleStage.ProducerAuxWork
    ]
    tail_entries = [
        entry
        for entry in copy_sfab.tail_schedule_list
        if entry[0] in {smem_sfa, smem_sfb, tmem_sfab}
        and entry[1] != ScheduleStage.ConsumerAuxWork
        and entry[1] != ScheduleStage.ProducerAuxWork
    ]

    assert (tmem_sfab, ScheduleStage.ProducerCommit) not in [
        (entry[0], entry[1]) for entry in head_entries
    ]
    assert [entry[1] for entry in head_entries[:4]] == [
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerWait,
    ]
    assert [entry[1] for entry in head_entries[-2:]] == [
        ScheduleStage.ProducerAcquire,
        ScheduleStage.ProducerWork,
    ]
    assert [entry[1] for entry in loop_entries[:3]] == [
        ScheduleStage.ProducerCommit,
        ScheduleStage.ConsumerRelease,
        ScheduleStage.ConsumerRelease,
    ]
    assert [entry[1] for entry in loop_entries[3:7]] == [
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerTryWait,
        ScheduleStage.ConsumerWait,
        ScheduleStage.ConsumerWait,
    ]
    assert [entry[1] for entry in tail_entries] == [
        ScheduleStage.ProducerCommit,
        ScheduleStage.ConsumerRelease,
        ScheduleStage.ConsumerRelease,
    ]


def test_runtime_config_preserves_plain_bf16_stages():
    cfg = make_config(
        bias_type=int(BiasType.M),
        cluster_m=2,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        epi_tile_n=64,
        mma_k=16,
        mma_m=256,
        mma_n=64,
        num_stages_a=5,
        num_stages_b=5,
        tile_k=128,
        tile_n=64,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_tma_store=1,
    )

    normalized = _runtime_config(cfg, in_hidden=3072)

    assert normalized.num_stages_a == 5
    assert normalized.num_stages_b == 5


def test_runtime_config_preserves_unsupported_mma_unroll_for_validation():
    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_unroll_loop_2x_for_mma=1,
    )

    normalized = _runtime_config(cfg, in_hidden=64)

    assert normalized.use_unroll_loop_2x_for_mma == 1
    with pytest.raises(ValueError, match="use_unroll_loop_2x_for_mma"):
        validate_config(normalized)


def test_runtime_config_preserves_scaled_fp4_stages():
    cfg = make_config(
        bias_type=int(BiasType.M),
        cluster_m=2,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        epi_tile_n=128,
        mma_k=64,
        mma_m=256,
        mma_n=128,
        num_stages_a=6,
        num_stages_b=6,
        num_stages_smem_sfa=6,
        num_stages_smem_sfb=6,
        num_stages_tmem_sfa=6,
        num_stages_tmem_sfb=6,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.NONE),
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        tile_k=256,
        tile_n=128,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_tma_store=1,
    )

    normalized = _runtime_config(cfg, in_hidden=3072)

    assert normalized.num_stages_a == 6
    assert normalized.num_stages_b == 6
    assert normalized.num_stages_smem_sfa == 6
    assert normalized.num_stages_smem_sfb == 6


def test_tmem_sf_utccp_pipeline_configs_use_umma_producer_umma_consumer():
    cfg = make_config(
        cluster_m=2,
        tile_n=128,
        mma_n=128,
        epi_tile_n=64,
        tile_k=256,
        sf_layout_b=int(SfLayout.R128c4),
    )
    assert cfg.sfa_smem_to_tmem_copy == int(SfSmemToTmemCopy.UTCCP)
    assert cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.UTCCP)

    pcfgs = _make_pipeline_configs(cfg)
    assert pcfgs["tmem_sfa"].pipeline_type == PipelineType.UmmaUmma
    assert pcfgs["tmem_sfb"].pipeline_type == PipelineType.UmmaUmma
    assert pcfgs["tmem_sfb"].producer_group.size == 1
    assert pcfgs["tmem_sfb"].consumer_group.size == 1

    compact_cfg = make_config(
        cluster_m=2,
        tile_n=64,
    )
    assert compact_cfg.sfb_smem_to_tmem_copy == int(SfSmemToTmemCopy.LDS_STTM)
    assert compact_cfg.smem_sfb_layout == int(SfLayout.R8c4)

    compact_pcfgs = _make_pipeline_configs(compact_cfg)
    assert compact_pcfgs["tmem_sfa"].pipeline_type == PipelineType.UmmaUmma
    assert compact_pcfgs["tmem_sfb"].pipeline_type == PipelineType.AsyncUmma


def test_proxy_cluster_barrier_uses_ts_async_umma_pipeline():
    proxy_cfg = PipelineConfig.create_async_umma_pipeline_cfg(
        num_stages=2,
        producer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=64),
        consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, size=1),
        cta_layout_vmnk=(2, 1, 1, 1),
        producer_signaling_threads=SignalingThreads.All,
        consumer_signaling_threads=SignalingThreads.CtaLeader,
    )

    assert proxy_cfg.pipeline_type == PipelineType.AsyncUmma
    assert "create_pipeline" not in ProxyClusterBarrierResource.__dict__
