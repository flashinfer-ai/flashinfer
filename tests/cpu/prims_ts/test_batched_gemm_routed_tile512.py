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

"""Routed tile512 completion, reuse, and cross-CTA relay contracts."""

from types import SimpleNamespace

import pytest
import cutlass.pipeline as pipeline
from cutlass.experimental.task_scheduling.enums import (
    PipelineType,
    ScheduleStage,
    SignalingThreads,
)

from flashinfer.prims_ts.batched_gemm import smem_ab_resources as ab_resources
from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga
from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    RouteImpl,
    make_config,
    validate_config,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
    _BatchedGemmTaskManager,
    _build_schedule_validate,
    _make_pipeline_configs,
)
from flashinfer.prims_ts.batched_gemm.batched_gemm_tasks import _dense_split_k_unroll
from tests.prims_ts.python_body import python_body
from tests.prims_ts.test_batched_gemm_routed_e32 import (
    routed_e32_tile512_config,
    routed_e32_dense_config,
)

pytestmark = pytest.mark.cpu_only


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize("wide_cga", [False, True])
@pytest.mark.parametrize("equal_depths", [False, True])
def test_routed_tile512_relay_pipeline_contract(route_a, wide_cga, equal_depths):
    options = routed_e32_tile512_config(route_a, wide_cga=wide_cga)
    if not equal_depths:
        options.update(num_stages_smem_sfa=3, num_stages_smem_sfb=4)
    cfg = make_config(**options)
    validate_config(cfg)
    assert cfg.uses_routed_tile512
    assert cfg.tile_m * cfg.umma_cta_count == 512
    assert cfg.tmem_required_cols == 576
    assert not cfg.aliases_c_scratch_with_ab
    tasks, graph, smem, tmem = _build_schedule_validate(cfg, num_k_tiles=7)
    manager = _BatchedGemmTaskManager(
        tasks=tasks,
        resource_dependency_graph=graph,
        smem_allocator=smem,
        tmem_allocator=tmem,
        smem_capacity_bytes=cfg.smem_capacity_bytes,
        tmem_capacity_columns=cfg.tmem_capacity_columns,
        exhaustive_deadlock_race_check=False,
        verbose=False,
    )
    manager.print_and_verify()
    tasks = {task.name: task for task in tasks}
    loaders = [
        tasks["GatherTask" if route_a == RouteImpl.LDGSTS else "LoadATask"],
        tasks["LoadBTask"],
        tasks["LoadSfATask"],
        tasks["LoadSfBTask"],
    ]
    warp_sets = [set(range(t.warp_idx, t.warp_idx + t.num_warps)) for t in loaders]
    assert [len(w) for w in warp_sets] == [4, 1, 4, 1]
    assert len(set.union(*warp_sets)) == sum(map(len, warp_sets))
    a, b, sa, sb = [t.dst_resources[0] for t in loaders]
    shared = route_a == RouteImpl.LDGSTS and equal_depths
    assert cfg.shares_ldgsts_a_sfa_pipeline == shared
    if shared:
        group = a.pipeline_group
        assert group is sa.pipeline_group
        assert group.pipeline_config.producer_group.size == 256  # Per CTA.
        assert group.pipeline_config.consumer_group.size == 1
        assert group.pipeline_config.num_bytes == 0
        assert group.pipeline_config.num_stages == 5
    else:
        assert a.pipeline_group is sa.pipeline_group is None
    assert b.pipeline_group is sb.pipeline_group is None

    pc = _make_pipeline_configs(cfg)
    sfa = pc["smem_sfa"]
    assert sfa.pipeline_type == PipelineType.AsyncUmma
    assert sfa.umma_consumer_producer_op == pipeline.PipelineOp.AsyncLoad
    assert sfa.producer_group.size == 128  # Four warps, local to each CTA.
    assert sfa.num_bytes == 0
    assert sfa.consumer_signaling_threads == SignalingThreads.CtaLeader
    assert sfa.consumer_wait_signaling_threads == SignalingThreads.All
    assert cfg.num_bytes_sfa_per_stage == 4096
    assert pc["smem_b"].num_bytes == 32768
    assert pc["smem_sfb"].num_bytes == 8192
    assert pc["tmem_c"].consumer_group.size == 512
    assert pc["proxy"].producer_group.size == 64
    assert "SyncTask" in tasks
    if route_a == RouteImpl.LDGSTS:
        assert pc["smem_gather"].producer_group.size == 128
        assert pc["smem_gather"].consumer_wait_signaling_threads == SignalingThreads.All
        assert pc["proxy"].num_stages == cfg.num_stages_a == cfg.num_stages_b
    else:
        assert pc["proxy"].num_stages == cfg.num_stages_smem_sfa
        assert pc["smem_a"].num_bytes == 65536

    def event_index(task, resource, stage, method=None):
        matches = [
            i
            for i, entry in enumerate(task.loop_schedule_list)
            if entry[0].name == resource
            and entry[1] == stage
            and (method is None or entry[-1] == method)
        ]
        assert len(matches) == 1, (task.name, resource, stage, matches)
        return matches[0]

    sf_loader = tasks["LoadSfATask"]
    assert event_index(
        sf_loader, "SmemSfA", ScheduleStage.ProducerWork, "load_sfa_tile"
    ) < event_index(sf_loader, "SmemSfA", ScheduleStage.ProducerCommit)
    assert not any(
        e[-1] in ("drain_loop", "drain_tail") for e in sf_loader.loop_schedule_list
    )
    mma = tasks["MmaTask0"]
    work = event_index(
        mma, "TmemC", ScheduleStage.ProducerWork, "mma_fused_sf_interleaved"
    )
    assert event_index(mma, "ProxyCluster", ScheduleStage.ConsumerWait) < work
    assert not any(
        e[1] in (ScheduleStage.ConsumerWait, ScheduleStage.ConsumerTryWait)
        and e[0].name in ("SmemSfA", "LdgstsActivations")
        for e in mma.loop_schedule_list
    )
    # Descriptor work uses the consumer's release state, independent of
    # the relay's wait cursor and of A/B's stage count.
    assert sa.pipeline_config.advance_on_wait
    releases = ["SmemB", "SmemSfB", "ProxyCluster"]
    releases += ["LdgstsActivations"] if shared else [a.name, "SmemSfA"]
    for name in releases:
        assert work < event_index(mma, name, ScheduleStage.ConsumerRelease)
    sync = tasks["SyncTask"]
    waits = [
        e[0].name for e in sync.loop_schedule_list if e[1] == ScheduleStage.ConsumerWait
    ]
    expected = (
        ["LdgstsActivations", "SmemB"]
        if shared
        else ["SmemGather", "SmemB", "SmemSfA"]
        if route_a == RouteImpl.LDGSTS
        else ["SmemSfA"]
    )
    assert waits == expected
    publish = event_index(sync, "ProxyCluster", ScheduleStage.ProducerCommit)
    for name in expected:
        assert event_index(sync, name, ScheduleStage.ConsumerWait) < publish
    assert not any(
        e[1] == ScheduleStage.ConsumerRelease for e in sync.loop_schedule_list
    )


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize(
    "override",
    [
        dict(dense_fuse_tma_pipeline=1),
        dict(dense_fuse_tma_loads=1),
        dict(dense_split_tma_pipeline=1),
        dict(dense_partition_tma_loads=1),
        dict(route_sfs_act=int(RouteImpl.TMA)),
        dict(fuse_sf_copy_to_mma=0),
        dict(sm_arch=100),
    ],
)
def test_routed_tile512_rejects_incompatible_pipelines(route_a, override):
    options = routed_e32_tile512_config(route_a)
    options.update(override)
    with pytest.raises(ValueError):
        validate_config(make_config(**options))


def test_routed_tile512_ldgsts_proxy_rejects_unequal_ab_depths():
    options = routed_e32_tile512_config(RouteImpl.LDGSTS)
    options.update(num_stages_b=6, num_stages_smem_sfb=6)
    with pytest.raises(ValueError, match="num_stages_a == num_stages_b"):
        validate_config(make_config(**options))


@pytest.mark.parametrize("scheduler", ["static", "clc"])
@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize(
    "paired,dense,producer",
    [
        pytest.param(1, 0, "PairedBSfBLoad", id="paired"),
        pytest.param(0, 1, "DenseWeightLoad", id="tma_a"),
        pytest.param(1, 1, None, id="conflicting"),
    ],
)
def test_routed_weight_ring_has_one_producer(
    scheduler, tile_m, paired, dense, producer
):
    from flashinfer.prims_ts.batched_gemm.routed_config import routed_options

    options = routed_options(
        collective_tile_m=tile_m,
        scheduler=scheduler,
        weight_pipeline="shared",
        max_active_clusters=2,
    )
    options.update(
        fuse_operand_sf_loads=paired,
        fused_a_single_commit=paired,
        dense_fuse_tma_loads=dense,
        route_act=int(RouteImpl.LDGSTS if paired else RouteImpl.TMA),
    )
    cfg = make_config(**options)
    assert cfg.uses_routed_weight_pipeline and cfg.dense_partition_tma_loads
    if producer is None:
        with pytest.raises(
            ValueError, match="dense_fuse_tma_loads.*fuse_operand_sf_loads"
        ):
            validate_config(cfg)
        return

    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=9)
    for resource in ("SmemB", "SmemSfB"):
        assert [
            task.name
            for task in tasks
            if any(dst.name == resource for dst in task.dst_resources)
        ] == [producer]


@pytest.mark.parametrize("route_a", [RouteImpl.LDGSTS, RouteImpl.TMA])
@pytest.mark.parametrize("wide_cga", [False, True])
@pytest.mark.parametrize("warps,sfa_stages", [(2, 5), (4, 5), (4, 3)])
def test_routed_dense_weight_pipeline_contract(route_a, wide_cga, warps, sfa_stages):
    options = routed_e32_dense_config(route_a, warps, wide_cga=wide_cga)
    options["num_stages_smem_sfa"] = sfa_stages
    cfg = make_config(**options)
    validate_config(cfg)
    assert cfg.num_stages_a == 5 and cfg.num_stages_b == 6
    assert _dense_split_k_unroll(cfg) == 6
    assert cfg.warp_layout.threads_per_cta == (16 if warps == 2 else 20) * 32
    tasks, graph, smem, tmem = _build_schedule_validate(cfg, num_k_tiles=7)
    _BatchedGemmTaskManager(
        tasks=tasks,
        resource_dependency_graph=graph,
        smem_allocator=smem,
        tmem_allocator=tmem,
        smem_capacity_bytes=cfg.smem_capacity_bytes,
        tmem_capacity_columns=cfg.tmem_capacity_columns,
        exhaustive_deadlock_race_check=False,
        verbose=False,
    ).print_and_verify()
    tasks = {t.name: t for t in tasks}
    weights = tasks["DenseWeightLoad"]
    assert weights.num_warps == 1
    assert "LoadBTask" not in tasks and "LoadSfBTask" not in tasks
    b, sb = weights.dst_resources
    assert b.pipeline_group is sb.pipeline_group
    assert b.pipeline_group.pipeline_config.num_stages == 6
    assert b.pipeline_group.pipeline_config.num_bytes == 40960
    # The single fused producer advances B/SFB once, after both loads.
    commits = [
        e for e in weights.loop_schedule_list if e[1] == ScheduleStage.ProducerCommit
    ]
    assert len(commits) == 1 and commits[0][0] is b
    weight_events = weights.loop_schedule_list
    commit_index = next(
        i for i, e in enumerate(weight_events) if e[1] == ScheduleStage.ProducerCommit
    )
    for method in ("load_b_tile", "load_sfb_tile"):
        load_index = next(
            i
            for i, e in enumerate(weight_events)
            if e[1] == ScheduleStage.ProducerWork and e[-1] == method
        )
        assert load_index < commit_index
    mma = tasks["MmaTask0"]
    waits = [
        e[0].name for e in mma.loop_schedule_list if e[1] == ScheduleStage.ConsumerWait
    ]
    assert "TmaWeights" in waits
    assert "SmemB" not in waits and "SmemSfB" not in waits
    mma_events = mma.loop_schedule_list
    wait_index = next(
        i
        for i, e in enumerate(mma_events)
        if e[0].name == "TmaWeights" and e[1] == ScheduleStage.ConsumerWait
    )
    use_index = next(
        i
        for i, e in enumerate(mma_events)
        if e[1] == ScheduleStage.ProducerWork and e[-1] == "mma_fused_sf_interleaved"
    )
    release_index = next(
        i
        for i, e in enumerate(mma_events)
        if e[0].name == "TmaWeights" and e[1] == ScheduleStage.ConsumerRelease
    )
    assert wait_index < use_index < release_index
    relay_waits = [
        e[0].name
        for e in tasks["SyncTask"].loop_schedule_list
        if e[1] == ScheduleStage.ConsumerWait
    ]
    assert not any(name in relay_waits for name in ("SmemB", "SmemSfB", "TmaWeights"))
    if route_a == RouteImpl.LDGSTS:
        assert cfg.shares_ldgsts_a_sfa_pipeline == (sfa_stages == 5)
    else:
        assert cfg.partitions_routed_tma_a
        assert cfg.routed_a_y_partitions == (4 if wide_cga else 1)


def test_routed_dense_rejects_mismatched_weight_ring_and_dynamic_partition():
    options = routed_e32_dense_config(RouteImpl.TMA, wide_cga=True)
    with pytest.raises(ValueError, match="matching B/SFB depths"):
        validate_config(make_config(**{**options, "num_stages_smem_sfb": 5}))
    with pytest.raises(ValueError, match="static CGA specialization"):
        validate_config(
            make_config(
                **{**options, "dense_specialize_tma_cga": 0, "dense_local_barrier": 0}
            )
        )


@pytest.mark.parametrize("shape", [(2, 1, 1), (2, 2, 1), (2, 4, 1), (4, 4, 1)])
@pytest.mark.parametrize("warps", [2, 4])
@pytest.mark.parametrize("valid_rows", [0, 37, 256])
def test_routed_dense_gather_partitions_cover_rows_once(
    monkeypatch, shape, warps, valid_rows
):
    # Execute the real route preparation, including padding predicates. Across
    # Y issuers each X slice must load every row once, with no cross-slice reads.
    options = routed_e32_dense_config(RouteImpl.TMA, warps, wide_cga=True)
    options.update(cluster_dim_x=shape[0], cluster_dim_y=shape[1])
    if shape == (2, 1, 1):
        # Exercise the specialized fallback of a preferred 4x4 launch.
        options.update(cluster_dim_x=4, cluster_dim_y=4, _active_cga_shape=shape)
    cfg = make_config(**options)
    validate_config(cfg)
    current = SimpleNamespace(warp=0, y=0, x=0)
    monkeypatch.setattr(ab_resources, "Int32", int)
    monkeypatch.setattr(ab_resources.cutlass, "const_expr", lambda x: x)
    monkeypatch.setattr(ab_resources.cutlass, "range_constexpr", range)
    monkeypatch.setattr(ab_resources.cute.arch, "make_warp_uniform", lambda x: x)
    monkeypatch.setattr(ab_resources.cute.arch, "warp_idx", lambda: current.warp)
    monkeypatch.setattr(
        ab_resources,
        "active_cga_coords",
        lambda _: (current.x + shape[0] * current.y, *shape, current.x, current.y),
    )
    prepare = python_body(ab_resources.SmemTmaGatherResource.prepare_gather_tile)
    load_row = python_body(ab_resources.SmemTmaGatherResource._load_routed_row_or_oob)
    local_limit = python_body(ab_resources.SmemTmaGatherResource._local_tile_limit)

    class GatherState(SimpleNamespace):
        skips_padding_gathers = ab_resources.SmemTmaGatherResource.skips_padding_gathers

    for x in range(shape[0]):
        current.x = x
        tile_m = 8 + x
        base_row = tile_m * 256
        addressed, loaded = [], []

        def load_route(*, idx, vector_size):
            assert vector_size == 1
            assert base_row <= idx < base_row + valid_rows
            loaded.append(idx)
            return [100000 + idx]

        resource = GatherState(
            cfg=cfg,
            _operand="a",
            routed_rows=[None] * 256,
            route_map=SimpleNamespace(load=load_route),
            mn_limit=SimpleNamespace(load=lambda **_: [base_row + valid_rows]),
        )
        resource._local_tile_limit = lambda *args: local_limit(resource, *args)

        def record_row(idx, row, limit):
            addressed.append(row)
            result = load_row(resource, idx, row, limit)
            assert result == (100000 + idx if row < valid_rows else -1)
            return result

        resource._load_routed_row_or_oob = record_row
        stage = SimpleNamespace(work_tile=SimpleNamespace(tile_idx=(tile_m, 0, 0)))
        for y in range(shape[1]):
            current.y = y
            for warp in range(warps):
                current.warp = cfg.warp_layout.load_a_warp_idx + warp
                prepare(resource, stage)
        assert sorted(addressed) == list(range(256))
        assert sorted(loaded) == list(range(base_row, base_row + valid_rows))


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("retain", [False, True])
@pytest.mark.parametrize("cta_rank", [1, 7])
def test_routed_dense_gather_multicast_completion_and_cache_hint(
    monkeypatch, batched, retain, cta_rank
):
    # Gather4 must name its own atom leader, including the second X atom.
    # The optional cache policy must not change the completion address/mask.
    class Pointer:
        def __init__(self, value):
            self.value = value

        def data_ptr(self):
            return self.value

    calls = []
    cfg = make_config(**routed_e32_tile512_config(RouteImpl.TMA, wide_cga=True))
    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(
        ab_resources,
        "input_tma_barrier_address",
        python_body(cga.input_tma_barrier_address),
    )
    monkeypatch.setattr(ab_resources, "Int32", int)
    monkeypatch.setattr(ab_resources, "Uint16", int)
    monkeypatch.setattr(ab_resources.cutlass, "const_expr", lambda x: x)
    monkeypatch.setattr(
        ab_resources.cute.arch, "block_idx_in_cluster", lambda: cta_rank
    )
    monkeypatch.setattr(ab_resources.cute.arch, "make_warp_uniform", lambda x: x)
    monkeypatch.setattr(
        ab_resources.prims, "mapa", lambda ptr, rank: Pointer((ptr.value, rank))
    )
    monkeypatch.setattr(ab_resources.prims, "cvta_to", lambda ptr, _: ptr)
    monkeypatch.setattr(ab_resources, "_retained_a_cache_policy", lambda: 12345)
    monkeypatch.setattr(
        ab_resources.prims,
        "inline_ptx_hl",
        lambda code, *, read_only_args: calls.append((code, read_only_args)),
    )
    if batched:
        body = python_body(ab_resources._tma_gather4_cluster_x4)
        body(
            *[Pointer(i) for i in range(4)],
            Pointer("tma"),
            256,
            *range(16),
            Pointer("barrier"),
            0xCCCC,
            cfg,
            "a",
            retain_a=retain,
        )
    else:
        body = python_body(ab_resources._tma_gather4_cluster)
        body(
            Pointer(0),
            Pointer("tma"),
            256,
            *range(4),
            Pointer("barrier"),
            0xCCCC,
            cfg,
            "a",
            retain_a=retain,
        )
    assert len(calls) == 1
    code, operands = calls[0]
    assert code.count("tile::gather4") == (4 if batched else 1)
    assert code.count("multicast::cluster.cta_group::2") == (4 if batched else 1)
    assert ("L2::cache_hint" in code) == retain
    completion_index = 22 if batched else 7
    assert operands[completion_index] == ("barrier", cta_rank & ~1)
    assert operands[completion_index + 1] == 0xCCCC
    if retain:
        assert operands[-1] == 12345
