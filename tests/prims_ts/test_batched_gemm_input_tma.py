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

"""Input-TMA task, physical-ring and WorkId ownership contracts."""

from collections import Counter
from dataclasses import fields

import pytest

from tests.prims_ts.python_body import python_body
from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _build_schedule_validate,
)
from tests.prims_ts.test_batched_gemm_routed_geometry import options


def _independent_pair(cga_shape=(4, 2), *, partitioned=False, scheduler="clc"):
    opts = options(512, cga_shape, scheduler)
    opts.update(dense_split_tma_pipeline=0, dense_partition_tma_loads=int(partitioned))
    if scheduler == "clc":
        opts["num_stages_workid"] = 1
    return make_config(**opts)


def _host_masks(monkeypatch, shape, rank):
    from flashinfer.prims_ts.batched_gemm import mixed_cga_ts

    x, y = rank % shape[0], rank // shape[0]
    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(cga, "Boolean", bool)
    monkeypatch.setattr(cga.cutlass, "range_constexpr", range)
    monkeypatch.setattr(cga, "active_cga_coords", lambda cfg: (*shape, 1, rank, x, y))
    monkeypatch.setattr(cga.cute.arch, "block_idx_in_cluster", lambda: rank)
    monkeypatch.setattr(cga.cute.arch, "make_warp_uniform", lambda value: value)
    for name in (
        "_rank_bit",
        "_atom_mask",
        "_column_mask",
        "_row_mask",
        "_x_parity_mask",
        "_project_payload_to_barrier_owners",
        "cga_policy_mask",
        "input_tma_release",
        "input_tma_payload_mask",
    ):
        monkeypatch.setattr(cga, name, python_body(getattr(cga, name)))
    monkeypatch.setattr(mixed_cga_ts, "input_tma_release", cga.input_tma_release)


@pytest.mark.parametrize(
    "shape,cluster_dim_x",
    [
        ((2, 1), 2),
        ((2, 4), 2),
        ((4, 2), 4),
        ((4, 4), 4),
        pytest.param((2, 1), 0, id="auto-2x1"),
        pytest.param((2, 4), 0, id="auto-2x4"),
    ],
)
@pytest.mark.parametrize("fallback", [False, True])
def test_omitted_gather_bytes_reach_all_full_barrier_owners(
    monkeypatch, shape, cluster_dim_x, fallback
):
    from types import SimpleNamespace
    from tests.prims_ts.routing_matrix_utils import matrix_options

    opts = matrix_options(512, shape, routed=True)
    opts["cluster_dim_x"] = cluster_dim_x
    cfg = make_config(**opts)
    active_shape = (2, 1) if fallback else shape
    complete = python_body(cga.complete_input_tma_bytes)
    masks = python_body(cga.cga_tma_masks)
    monkeypatch.setattr(cga, "cga_tma_masks", masks)
    for rank in range(active_shape[0] * active_shape[1]):
        _host_masks(monkeypatch, active_shape, rank)
        credits = []
        monkeypatch.setattr(
            cga.prims,
            "mapa",
            lambda barrier, owner: SimpleNamespace(data_ptr=lambda: owner),
        )
        monkeypatch.setattr(
            cga.prims,
            "cvta_to",
            lambda *args: pytest.fail(
                "owner credits must retain cluster address space"
            ),
        )

        def record(instruction, *, read_only_args):
            assert instruction.startswith(
                "mbarrier.complete_tx.relaxed.cluster.shared::cluster."
            )
            credits.append(tuple(read_only_args))

        monkeypatch.setattr(cga.prims, "inline_ptx_hl", record)
        complete(cfg, "a", object(), 1536)
        # A multicasts along N; adjacent X payloads complete at even X.
        x = rank % active_shape[0]
        owners = [(x & ~1) + y * active_shape[0] for y in range(active_shape[1])]
        assert credits == [(owner, 1536) for owner in owners]


@pytest.mark.parametrize("shape", [(2, 1), (2, 4), (4, 1), (4, 2), (4, 4)])
@pytest.mark.parametrize("partitioned", [False, True])
@pytest.mark.parametrize("scheduler", ["clc", "static"])
def test_independent_pair_actual_tasks_and_release_counts(
    monkeypatch, shape, partitioned, scheduler
):
    cfg = _independent_pair(shape, partitioned=partitioned, scheduler=scheduler)
    validate_config(cfg)
    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    task = next(t for t in tasks if t.name == "PairedBSfBLoad")
    b, sb = task.dst_resources[:2]
    assert b.pipeline_group is sb.pipeline_group is None
    # Real byte credits are per receiving UMMA atom, never divided by the
    # number of elected issuers or combined tasks.
    assert (
        b.pipeline_config.num_bytes
        == cfg.tile_n * cfg.tile_k * cfg.dtype_b_smem_bits // 8
    )
    assert sb.pipeline_config.num_bytes == cfg.num_bytes_sfb_per_stage * 2
    topo = cga.HostCgaTopology(*shape)
    expected_b = {
        x + shape[0] * y for y in range(shape[1]) for x in range(min(2, shape[0]))
    }
    extended_sfb = shape != (2, 1) or partitioned
    expected_sb = (
        expected_b
        if partitioned or not extended_sfb
        else {shape[0] * y for y in range(shape[1])}
    )
    actual_task = {
        rank
        for rank in range(shape[0] * shape[1])
        if topo.selected(
            getattr(task, "cga_policy", 0), rank % shape[0], rank // shape[0]
        )
    }
    assert actual_task == expected_b | expected_sb
    for operand, expected in (("b", expected_b), ("sfb", expected_sb)):
        control = cga.input_tma_control(cfg, operand)
        actual = {
            rank
            for rank in range(shape[0] * shape[1])
            if topo.selected(control.participants, rank % shape[0], rank // shape[0])
        }
        assert actual == expected
        arrivals = Counter()
        for leader in range(0, shape[0] * shape[1], 2):
            _host_masks(monkeypatch, shape, leader)
            _, owners, release, count = python_body(cga.cga_tma_masks)(cfg, operand)
            row = leader // shape[0]
            expected_release = {rank for rank in expected if rank // shape[0] == row}
            assert release == sum(1 << rank for rank in expected_release)
            assert count == shape[0] // 2
            arrivals.update(expected_release)
            if extended_sfb or operand == "b":
                assert owners & (1 << leader)
        assert dict(arrivals) == {rank: shape[0] // 2 for rank in expected}
    for rank in range(shape[0] * shape[1]):
        actual_threads = 32 * sum(
            t.num_warps
            for t in tasks
            if t.name != "WorkScheduleTask"
            and topo.selected(
                getattr(t, "cga_policy", 0), rank % shape[0], rank // shape[0]
            )
            and t.run_only_on_cta_id in (None, rank)
        )
        assert cga.static_queue_consumer_threads(cfg, rank) == actual_threads
    actual_threads = 32 * sum(
        t.num_warps
        for t in tasks
        for rank in range(shape[0] * shape[1])
        if topo.selected(getattr(t, "cga_policy", 0), rank % shape[0], rank // shape[0])
        and t.run_only_on_cta_id in (None, rank)
    )
    if scheduler == "clc":
        assert cga.host_clc_consumer_threads(cfg, *shape) == actual_threads


def test_independent_member_try_acquire_work_and_tail_are_gated(monkeypatch):
    from cutlass.experimental.task_scheduling.task import Task
    from flashinfer.prims_ts.batched_gemm import mixed_cga_ts as ts

    cfg = _independent_pair()
    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    task = next(t for t in tasks if t.name == "PairedBSfBLoad")
    assert isinstance(task, ts.InputTmaTask)
    b, sb = task.dst_resources[:2]
    calls = []
    for hook in (
        "_producer_try_acquire",
        "_producer_acquire",
        "_producer_work",
        "_producer_tail",
    ):
        monkeypatch.setattr(
            Task,
            hook,
            lambda self, resource, *args, hook=hook: calls.append(
                (hook, resource.name)
            ),
        )
    monkeypatch.setattr(
        Task,
        "_producer_try_acquire_if",
        lambda self, resource, selected: calls.append(
            ("_producer_try_acquire", resource.name)
        )
        if selected
        else None,
    )
    for rank in (0, 1, 4, 5):
        calls.clear()
        monkeypatch.setattr(
            ts,
            "cga_policy_selected",
            lambda cfg, policy: cga.HostCgaTopology(4, 2).selected(
                policy, rank % 4, rank // 4
            ),
        )
        for resource in (b, sb):
            python_body(ts.InputTmaTask._producer_try_acquire)(task, resource, None)
            python_body(ts.InputTmaTask._producer_acquire)(task, resource, None)
            python_body(ts.InputTmaTask._producer_work)(task, resource, None, None)
            python_body(ts.InputTmaTask._producer_tail)(task, resource)
        assert Counter(calls) == Counter(
            (hook, resource.name)
            for resource in ((b, sb) if rank % 4 == 0 else (b,))
            for hook in (
                "_producer_try_acquire",
                "_producer_acquire",
                "_producer_work",
                "_producer_tail",
            )
        )


@pytest.mark.parametrize("kind", ["shared_separate_tasks", "all_fused", "split_pairs"])
def test_shared_physical_barrier_union_and_complete_bytes(monkeypatch, kind):
    from tests.prims_ts.test_batched_gemm_dense import (
        _dense_config,
        _preloaded_epilogue_config,
    )

    if kind == "shared_separate_tasks":
        cfg = _dense_config(cluster_dim_y=4, fallback_cluster_dim_x=0)
    else:
        cfg = _preloaded_epilogue_config(
            cluster_dim_x=4,
            cluster_dim_y=2,
            dense_split_tma_pipeline=int(kind == "split_pairs"),
        )
    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    groups = {
        id(r.pipeline_group): r.pipeline_group
        for t in tasks
        for r in t.dst_resources
        if getattr(r, "mixed_cga_kind", None) and r.pipeline_group is not None
    }
    assert len(groups) == (2 if kind == "split_pairs" else 1)
    for group in groups.values():
        assert group.pipeline_config.num_bytes == sum(
            r.pipeline_config.num_bytes for r in group.members
        )
        producer_tasks = [
            t for t in tasks if any(r in group.members for r in t.dst_resources)
        ]
        if kind == "shared_separate_tasks":
            assert len(producer_tasks) == 4
        else:
            assert len(producer_tasks) == 1
        shape = cfg.preferred_cluster_shape[:2]
        topo = cga.HostCgaTopology(*shape)
        waiters = {
            rank
            for rank in range(shape[0] * shape[1])
            if any(
                topo.selected(
                    getattr(t, "cga_policy", 0), rank % shape[0], rank // shape[0]
                )
                for t in producer_tasks
            )
        }
        arrivals = Counter()
        for rank in range(0, shape[0] * shape[1], 2):
            _host_masks(monkeypatch, shape, rank)
            release, count = group._release_mask_and_count()
            x, y = rank % shape[0], rank // shape[0]
            expected = {
                r
                for r in waiters
                if (
                    r // shape[0] == y
                    if group.split_operand == "b"
                    else (r % shape[0]) // 2 == x // 2
                    if group.split_operand == "a"
                    else r // shape[0] == y or (r % shape[0]) // 2 == x // 2
                )
            }
            assert release == sum(1 << r for r in expected)
            arrivals.update(expected)
        assert dict(arrivals) == {rank: count for rank in waiters}


@pytest.mark.parametrize("kind", ["cast_a", "compact_sfb", "routed_sfa"])
def test_local_async_keeps_thread_arrivals_and_transaction_bytes(kind):
    from cutlass.experimental.task_scheduling.enums import PipelineType
    from tests.prims_ts.test_batched_gemm_mixed_cga import _fp4_fc2_mixed_config
    from tests.prims_ts.test_batched_gemm_mx import _mxfp4_bf16_base, _finalize_tmem
    from tests.prims_ts.test_batched_gemm_rubin import RUBIN_ROUTE_ACT_FP4
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        RouteImpl,
    )

    if kind == "cast_a":
        cfg = make_config(
            **_finalize_tmem(_mxfp4_bf16_base(has_activation_epilogue=False))
        )
        expected = {
            "a": (cfg.num_bytes_a_tma_per_stage, 128),
            "sfa": (cfg.num_bytes_sfa_per_stage, 128),
        }
    elif kind == "compact_sfb":
        cfg = _fp4_fc2_mixed_config()
        expected = {"sfb": (cfg.num_bytes_sfb_tma_per_stage, 128)}
    else:
        cfg = make_config(
            **RUBIN_ROUTE_ACT_FP4,
            tile_k=512,
            batch_mode=int(BatchMode.BATCH_M),
            transpose_mma_output=0,
            route_act=int(RouteImpl.TMA),
            route_sfs_act=int(RouteImpl.TMA),
            cluster_dim_x=4,
            cluster_dim_y=2,
        )
        expected = {"sfa": (cfg.num_bytes_sfa_per_stage, 128)}
    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    resources = {
        r.mixed_cga_kind: r
        for t in tasks
        for r in t.dst_resources
        if getattr(r, "mixed_cga_kind", None)
    }
    for operand, (num_bytes, consumers) in expected.items():
        resource = resources[operand]
        if kind == "routed_sfa":
            from flashinfer.prims_ts.batched_gemm.smem_sf_resources import (
                SmemSfGatherAResource,
            )

            # Device assembly uses the gather resource with this pipeline.
            resource = SmemSfGatherAResource(
                cfg=cfg, pipeline_config=resource.pipeline_config, name="SmemSfA"
            )
            assert resource.mixed_cga_kind == "sfa"
        control = cga.input_tma_control(cfg, operand)
        assert control.full_arming == "local_async"
        assert control.issuers == control.participants == cga.CgaPolicy.ALL
        pcfg = resource.pipeline_config
        assert pcfg.pipeline_type == PipelineType.TmaAsync
        assert pcfg.num_bytes == num_bytes
        assert pcfg.consumer_group.size == consumers


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "shape,partitioned,scheduler",
    [
        ((4, 2), False, "clc"),
        ((4, 4), True, "clc"),
        ((4, 2), False, "static"),
        ((2, 1), False, "clc"),
    ],
)
def test_independent_pair_ring_reuse_gpu(monkeypatch, shape, partitioned, scheduler):
    import torch
    from flashinfer.utils import get_compute_capability
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    cfg = _independent_pair(shape, partitioned=partitioned, scheduler=scheduler)
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    from cuda.bindings import driver as cuda_drv

    torch_stream = torch.cuda.Stream()
    monkeypatch.setattr(
        runner,
        "get_default_stream",
        lambda: cuda_drv.CUstream(torch_stream.cuda_stream),
    )
    original_compile = runner._compile_for_launch
    graphs = []

    def compile_with_graph(io, stream):
        compiled = original_compile(io, stream)
        assert not io["cfg"].is_swap_ab and not io["c_row_stride"]
        output_bytes = io["shape"][0] * io["shape"][1] * io["cfg"].dtype_c_bits // 8
        warmed = False

        def launch(*args):
            nonlocal warmed
            if not warmed:
                compiled(*args)
                warmed = True
                return
            graph = torch.cuda.CUDAGraph(keep_graph=True)
            with torch.cuda.graph(graph, stream=torch_stream):
                compiled(*args)
            status, nodes, count = cuda_drv.cuGraphGetNodes(
                cuda_drv.CUgraph(graph.raw_cuda_graph()), 16
            )
            assert status == cuda_drv.CUresult.CUDA_SUCCESS and count > 0
            assert any(
                cuda_drv.cuGraphNodeGetType(node)
                == (
                    cuda_drv.CUresult.CUDA_SUCCESS,
                    cuda_drv.CUgraphNodeType.CU_GRAPH_NODE_TYPE_KERNEL,
                )
                for node in nodes[:count]
            )
            for _ in range(32):
                graph.replay()
            # Poison only C's logical storage, retaining the allocation guards.
            # The final checked replay must overwrite it on the same stream.
            (status,) = cuda_drv.cuMemsetD8Async(
                io["c0_dp"]._pointer, 0xFF, output_bytes, stream
            )
            assert status == cuda_drv.CUresult.CUDA_SUCCESS
            graph.replay()
            graphs.append(graph)

        return launch

    monkeypatch.setattr(runner, "_compile_for_launch", compile_with_graph)
    with torch.cuda.stream(torch_stream):
        result = runner.reference_check(
            num_experts=3,
            num_tokens=1031,
            top_k=2,
            problem_n=1280,
            problem_k=2304,
            routing="uniform",
            repeat_launches=5,
            output_guard_elements=256,
            **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
        )
    assert len(graphs) == 4
    assert result and result.max_abs == 0


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "loader_mode,cluster_y", [("separate", 1), ("separate", 4), ("fused", 2)]
)
def test_shared_barrier_ring_reuse_gpu(monkeypatch, loader_mode, cluster_y):
    import torch
    from flashinfer.utils import get_compute_capability
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from tests.prims_ts.test_batched_gemm_dense import (
        _dense_config,
        _preloaded_epilogue_config,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    cfg = (
        _preloaded_epilogue_config(
            cluster_dim_x=4, cluster_dim_y=cluster_y, dense_single_expert=0
        )
        if loader_mode == "fused"
        else _dense_config(cluster_dim_y=cluster_y)
    )
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    result = runner.reference_check(
        num_experts=3,
        num_tokens=1031,
        top_k=2,
        problem_n=1280,
        problem_k=2304,
        routing="uniform",
        repeat_launches=5,
        output_guard_elements=256,
        **{f.name: getattr(cfg, f.name) for f in fields(cfg) if f.init},
    )
    assert result and result.max_abs == 0


@pytest.mark.parametrize("mismatch", ["task", "count"])
def test_clc_verifier_rejects_participation_mismatch(mismatch):
    from dataclasses import replace
    from cutlass import pipeline
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _BatchedGemmTaskManager,
    )
    from flashinfer.prims_ts.batched_gemm.smem_misc_resources import (
        BatchedGemmWorkQueue,
    )
    from tests.prims_ts.test_batched_gemm_dense import _dense_config

    cfg = _dense_config(cluster_dim_y=1)
    tasks, resources, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    manager = object.__new__(_BatchedGemmTaskManager)
    manager.tasks = tasks
    manager.resources = resources
    manager._verify_pipeline_thread_counts()
    queue = next(r for r in resources if isinstance(r, BatchedGemmWorkQueue))
    assert queue.pipeline_config.consumer_group.size == 672
    if mismatch == "task":
        next(t for t in tasks if t.name == "LoadSfBTask").cga_policy = cga.CgaPolicy.ALL
    else:
        queue.pipeline_config = replace(
            queue.pipeline_config,
            consumer_group=pipeline.CooperativeGroup(pipeline.Agent.Thread, 704),
        )
    with pytest.raises(AssertionError):
        manager._verify_pipeline_thread_counts()


@pytest.mark.parametrize("ctas", [1, 2])
@pytest.mark.parametrize("operand", ["a", "b", "sfa", "sfb", "gather_a", "gather_b"])
def test_legacy_pipeline_masks_come_from_control(monkeypatch, ctas, operand):
    """Poison stock ownership to detect a legacy resource bypassing control."""
    from cutlass.experimental.task_scheduling.resources import MemoryResource
    from cutlass.pipeline.sm100 import PipelineTmaUmma
    from flashinfer.prims_ts.batched_gemm import mixed_cga_ts as ts
    from flashinfer.prims_ts.batched_gemm.smem_ab_resources import (
        SmemAResource,
        SmemBResource,
        SmemTmaGatherResource,
    )
    from flashinfer.prims_ts.batched_gemm.smem_sf_resources import (
        SmemSfAResource,
        SmemSfBResource,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _make_pipeline_configs,
    )
    from tests.prims_ts.test_batched_gemm_mixed_cga import _fp4_fc2_mixed_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout

    cfg = _fp4_fc2_mixed_config(
        umma_cta_count=ctas,
        mma_m=128 * ctas,
        cluster_dim_x=ctas,
        cluster_dim_y=1,
        fallback_cluster_dim_x=0,
        fallback_cluster_dim_y=1,
        tile_n=128,
        mma_n=128,
        sf_layout_b=int(SfLayout.R128c4),
        num_stages_tmem_sfa=1,
        num_stages_tmem_sfb=1,
        num_stages_tmem_acc=1,
    )
    kind = operand.removeprefix("gather_")
    cls = (
        SmemTmaGatherResource
        if operand.startswith("gather_")
        else {
            "a": SmemAResource,
            "b": SmemBResource,
            "sfa": SmemSfAResource,
            "sfb": SmemSfBResource,
        }[kind]
    )
    resource = object.__new__(cls)
    for key, value in dict(
        cfg=cfg,
        mixed_cga_kind=kind,
        pipeline_config=_make_pipeline_configs(cfg)["smem_" + kind],
    ).items():
        object.__setattr__(resource, key, value)
    full, empty = object(), object()
    base = PipelineTmaUmma(full, empty, 3, 0xDEAD, 0xDEAD, False, cfg.cta_group)
    initialized = []
    monkeypatch.setattr(
        MemoryResource,
        "create_pipeline",
        lambda self, pcfg: (initialized.append(pcfg), base)[1],
    )
    monkeypatch.setattr(ts, "cga_tma_masks", python_body(cga.cga_tma_masks))
    monkeypatch.setattr(
        ts, "input_tma_full_armer_cta", python_body(cga.input_tma_full_armer_cta)
    )
    for rank in range(ctas):
        _host_masks(monkeypatch, (ctas, 1), rank)
        monkeypatch.setattr(cga.cute.arch, "block_idx", lambda: (rank, 0, 0))
        actual = resource.create_pipeline(resource.pipeline_config)
        assert actual.sync_object_full is full and actual.sync_object_empty is empty
        assert actual.num_stages == base.num_stages
        assert actual.consumer_mask == (3 if ctas == 2 else None)
        assert actual.producer_mask == (1 if ctas == 2 else None)
        assert actual.is_leader_cta == (rank == 0)
        assert actual.cta_group == (
            cga.cute.nvgpu.tcgen05.CtaGroup.TWO
            if ctas == 2
            else cga.cute.nvgpu.tcgen05.CtaGroup.ONE
        )
        assert initialized[-1].consumer_group.size == 1


@pytest.mark.parametrize("local_address", [False, True])
@pytest.mark.parametrize("operand", ["a", "b", "sfa", "sfb"])
def test_tiled_and_gather_completion_address(monkeypatch, local_address, operand):
    from dataclasses import replace

    cfg = replace(_independent_pair(), dense_local_barrier=int(local_address))
    barrier, mapped, converted = object(), object(), object()
    calls = []
    conversions = []
    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(cga.cute.arch, "make_warp_uniform", lambda x: x)
    monkeypatch.setattr(
        cga.prims, "mapa", lambda ptr, rank: (calls.append((ptr, rank)), mapped)[1]
    )
    monkeypatch.setattr(
        cga.prims,
        "cvta_to",
        lambda ptr, space: (conversions.append((ptr, space)), converted)[1],
    )
    for rank in range(8):
        monkeypatch.setattr(cga.cute.arch, "block_idx_in_cluster", lambda: rank)
        for gather in (False, True):
            calls.clear()
            conversions.clear()
            result = python_body(cga.input_tma_barrier_address)(
                cfg, operand, barrier, gather
            )
            if gather or not local_address:
                assert calls == [(barrier, rank & ~1)]
                assert conversions == [(mapped, cga.prims.CvtaSpace.SHARED)]
                assert result is converted
            else:
                assert not calls and not conversions
                assert result is barrier


@pytest.mark.parametrize(
    "instruction,locality",
    [("cta", True), ("cluster", True), ("umma", True), ("umma", False), ("umma", None)],
)
@pytest.mark.parametrize("operand", ["a", "sfa"])
@pytest.mark.parametrize("explicit_hint", [False, True])
def test_input_tma_cache_policy_survives_dispatch(
    monkeypatch, instruction, locality, operand, explicit_hint
):
    """Per-expert hints reach the actual emitter; native defaults still apply."""
    from dataclasses import replace

    cfg = replace(_independent_pair(), dense_retain_a=1)
    _host_masks(monkeypatch, (4, 2), 0)
    weight_hint, activation_hint = object(), object()
    hint = weight_hint if explicit_hint else None
    calls = []
    monkeypatch.setattr(cga, "input_tma_instruction_kind", lambda *args: instruction)
    monkeypatch.setattr(cga, "tma_payload_is_local", lambda *args: locality)
    monkeypatch.setattr(cga, "input_tma_barrier_address", lambda cfg, op, ptr: ptr)
    monkeypatch.setattr(cga, "cga_policy_selected", lambda *args: True)
    monkeypatch.setattr(cga, "_retained_a_cache_policy", lambda: activation_hint)
    for name in ("issue_tma_load", "_issue_tma_load_impl"):
        monkeypatch.setattr(cga, name, python_body(getattr(cga, name)))

    def record(*args, **kwargs):
        calls.append(kwargs)

    for name in (
        "cp_async_bulk_tensor_shared_cta_global",
        "cp_async_bulk_tensor_shared_cluster_global",
    ):
        monkeypatch.setattr(cga.prims, name, record)
    python_body(cga.load_input_tma)(
        cfg, operand, object(), object(), (), object(), l2_cache_hint=hint
    )
    expected = hint
    if not explicit_hint and instruction == "umma":
        expected = activation_hint
    assert len(calls) == 1
    assert calls[0]["l2_cache_hint"] is expected
