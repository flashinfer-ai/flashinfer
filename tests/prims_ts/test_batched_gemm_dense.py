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

"""Dense NVFP4 shared-pipeline topology and configuration regressions."""

from dataclasses import replace

import pytest

from tests.prims_ts.python_body import python_body


def _dense_config(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        SfLayout,
        TileScheduler,
        compute_warp_layout,
        make_config,
    )

    values = dict(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_M),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.FP16),
        route_act=0,
        transpose_mma_output=0,
        act_kind=0,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.R128c4),
        umma_cta_count=2,
        cluster_dim_x=2,
        cluster_dim_y=4,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
        tile_n=256,
        tile_k=256,
        mma_m=256,
        mma_n=256,
        mma_k=128,
        epi_tile_n=64,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_a=8,
        num_stages_b=8,
        num_stages_smem_sfa=8,
        num_stages_smem_sfb=8,
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=1,
        num_stages_tmem_sfb=1,
        num_stages_c_smem=1,
        dense_fuse_tma_pipeline=1,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_unroll_loop_2x_for_mma=0,
        use_tma_store=1,
        use_pdl=0,
    )
    values.update(overrides)
    if (values["cluster_dim_x"], values["cluster_dim_y"]) == (2, 1):
        # Presets select fixed 2x1; explicit invalid fallback is tested through
        # validate_config without this fixture's default selection.
        values.update(
            fallback_cluster_dim_x=0,
            dense_specialize_tma_cga=0,
            dense_local_barrier=0,
        )
    cfg = make_config(**values)
    compute_warp_layout(cfg)
    return cfg


def test_dense_cli_architecture_and_shared_pipeline():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    assert "sm_arch" not in _parse_overrides(parser.parse_args([]))
    dense_options = (
        "dense_fuse_tma_pipeline",
        "dense_partition_tma_loads",
        "dense_specialize_tma_cga",
        "dense_fuse_tma_loads",
    )
    flags = ["--sm-arch", "107"]
    for name in dense_options:
        flags.extend(["--" + name.replace("_", "-"), "1"])
    overrides = _parse_overrides(parser.parse_args(flags))
    assert overrides["sm_arch"] == 107
    for name in dense_options:
        assert overrides[name] == 1
        assert name.removeprefix("dense_") not in overrides


@pytest.mark.parametrize("tile_m", [256, 512])
def test_dense_pair_schedule_printer_preserves_member_commits(tile_m):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import make_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _BatchedGemmTaskManager,
        _build_schedule_validate,
    )
    from cutlass.experimental.task_scheduling.enums import ScheduleStage
    from tests.prims_ts.routing_matrix_utils import matrix_options

    cfg = make_config(**matrix_options(tile_m, (4, 4), routed=False))
    tasks, resources, _, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    manager = object.__new__(_BatchedGemmTaskManager)
    manager.tasks = tasks
    manager.resources = resources
    manager._verbose = False
    before = [list(task.loop_schedule_list) for task in tasks]
    manager._print_task_schedules()
    assert manager.tasks is tasks
    assert [task.loop_schedule_list for task in tasks] == before

    producer = next(task for task in tasks if task.name == "DenseActivationLoad")
    commit = next(
        entry
        for entry in producer.loop_schedule_list
        if entry[1] == ScheduleStage.ProducerCommit
    )
    producer.loop_schedule_list.insert(
        producer.loop_schedule_list.index(commit), commit
    )
    with pytest.raises(ValueError, match="Duplicate fused member commit"):
        manager._print_task_schedules()
    assert manager.tasks is tasks


@pytest.mark.parametrize("cluster_y", [1, 4])
def test_dense_shared_pipeline_schedule_and_storage(cluster_y, monkeypatch):
    from cutlass.experimental.task_scheduling.pipeline_group import PipelineGroup
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import DenseTmaPipelineGroup
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        needs_extended_tma_multicast,
        host_clc_consumer_threads,
    )

    cfg = _dense_config(cluster_dim_y=cluster_y)
    validate_config(cfg)  # 2x1 uses a fixed launch; 2x4 retains the fallback.
    tasks, _, allocator, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    tasks = {task.name: task for task in tasks}
    assert needs_extended_tma_multicast(cfg, "sfb")
    assert tasks["LoadSfBTask"].cga_policy == int(CgaPolicy.MCAST_SFB)
    if cluster_y == 1:
        # The default FI path elects one full-slice SFB issuer. Its CLC
        # barrier excludes the skipped warp; partitioned DKG uses both X CTAs.
        separate = replace(cfg, dense_fuse_tma_pipeline=0)
        assert (
            host_clc_consumer_threads(separate, 2, 1)
            - host_clc_consumer_threads(cfg, 2, 1)
            == 32
        )
    members = [
        tasks[name].dst_resources[0]
        for name in ("LoadATask", "LoadBTask", "LoadSfATask", "LoadSfBTask")
    ]
    group = members[0].pipeline_group
    assert isinstance(group, DenseTmaPipelineGroup)
    assert all(member.pipeline_group is group for member in members)
    assert all(member.pipeline_config.num_stages == 8 for member in members)
    assert all(member.pipeline_config.mcast_mode_mn == (1, 1) for member in members)
    assert group.pipeline_config.producer_group.size == 4
    # The logical merge still has four producers. The device full barrier
    # uses one local arrival for multi-atom CGAs, arming their combined bytes.
    monkeypatch.setattr(
        PipelineGroup, "_make_full_sync_object", lambda self, cfg, ptr: cfg
    )
    full_cfg = group._make_full_sync_object(group.pipeline_config, None)
    assert full_cfg.producer_group.size == (1 if cluster_y > 1 else 4)
    assert full_cfg.num_bytes == group.pipeline_config.num_bytes
    assert group.pipeline_config.num_bytes == sum(
        member.pipeline_config.num_bytes for member in members
    )
    _, _, separate_allocator, _ = _build_schedule_validate(
        replace(cfg, dense_fuse_tma_pipeline=0), num_k_tiles=9
    )
    # The allocator reserves 16 bytes per member stage and conservatively
    # reserves the same unit per group barrier. Check its allocation contract
    # so forgetting to register the shared group cannot silently pass.
    group_bytes = group.num_barriers_per_stage * 8 * 16
    assert separate_allocator._barrier_bytes - allocator._barrier_bytes == (
        4 * 8 * 16 - group_bytes
    )
    assert {member.name for member in group.members} == {
        "SmemA",
        "SmemB",
        "SmemSfA",
        "SmemSfB",
    }


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"dense_fuse_tma_pipeline": 2}, "dense_fuse_tma_pipeline"),
        ({"num_stages_b": 7}, "equal A/B/SFA/SFB stages"),
        ({"num_stages_smem_sfa": 7}, "equal A/B/SFA/SFB stages"),
        ({"fuse_sf_copy_to_mma": 0}, "dense_fuse_tma_pipeline requires"),
        ({"cluster_dim_x": 4}, "dense_fuse_tma_pipeline requires"),
        ({"use_pdl": 1}, "dense_fuse_tma_pipeline requires"),
    ],
)
def test_dense_shared_pipeline_rejects_unsupported_combinations(overrides, match):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    cfg = replace(_dense_config(), **overrides)
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


@pytest.mark.parametrize("dim_x, dim_y", [(2, 1), (2, 4), (4, 2), (4, 4)])
def test_shared_release_matches_dkg_union_and_arrival_count(monkeypatch, dim_x, dim_y):
    """Evaluate device mask arithmetic on integers; count every receiver's arrivals.

    DKG's 2-CTA PipelineTmaUmma uses A|B|A_peer|B_peer and NX+NY-1.
    Wider X is checked here to prevent accidentally generalizing 2xY's all-CTA
    mask when the fused-loader configuration uses X=4.
    """
    from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga

    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(cga.cutlass, "range_constexpr", range)
    for name in (
        "_rank_bit",
        "_column_mask",
        "_row_mask",
        "_atom_mask",
        "_project_payload_to_barrier_owners",
        "cga_policy_mask",
    ):
        monkeypatch.setattr(cga, name, python_body(getattr(cga, name)))
    arrivals = [0] * (dim_x * dim_y)
    for y in range(dim_y):
        for x in range(0, dim_x, 2):
            monkeypatch.setattr(
                cga,
                "active_cga_coords",
                lambda cfg, x=x, y=y: (dim_x, dim_y, 0, 0, x, y),
            )
            mask, count = python_body(cga.shared_tma_release)(
                _preloaded_epilogue_config(cluster_dim_x=dim_x, cluster_dim_y=dim_y)
            )
            # Independent set representation of DKG's four operand masks.
            a = {x + dim_x * yy for yy in range(dim_y)}
            a_peer = {rank + 1 for rank in a}
            b = {xx + dim_x * y for xx in range(0, dim_x, 2)}
            b_peer = {rank + 1 for rank in b}
            expected = a | a_peer | b | b_peer
            assert mask == sum(1 << rank for rank in expected)
            assert count == dim_y + dim_x // 2 - 1
            for rank in range(len(arrivals)):
                arrivals[rank] += bool(mask & (1 << rank))
    assert arrivals == [count] * len(arrivals)


@pytest.mark.parametrize("work_stages", [1, 3])
def test_dense_c_scratch_stays_disjoint_from_operand_ring(work_stages):
    """CLC queue depth does not bound the lifetime of pipelined A/B loads."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = _dense_config(
        num_stages_workid=work_stages,
        use_tma_oob_opt=0,
        use_max_tmem_overlap=0,
        num_stages_tmem_acc=2,
        num_stages_c_smem=2,
        epi_tile_n=32,
        dense_partition_tma_loads=1,
        dense_fuse_tma_loads=1,
        dense_rolled_epilogue=1,
        dense_rolled_mma=1,
    )
    validate_config(cfg)
    tasks, _, _, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    resources = {
        resource.name: resource
        for task in tasks
        for resource in task.src_resources + task.dst_resources
    }
    scratch = resources["GmemC"]._alloc_sc
    for name, field in [("SmemA", "_alloc_a"), ("SmemB", "_alloc_b")]:
        operand = getattr(resources[name], field)
        assert (
            scratch.offset + scratch.size_bytes <= operand.offset
            or operand.offset + operand.size_bytes <= scratch.offset
        )


def _tile512_config(**overrides):
    values = dict(
        dense_m_repeats=2,
        dense_b_reuse=1,
        dense_async_sf=1,
        dense_single_expert=1,
        dense_rolled_epilogue=1,
        dense_rolled_mma=1,
        dense_fuse_tma_loads=1,
        dense_partition_tma_loads=1,
        use_tma_oob_opt=0,
        use_max_tmem_overlap=0,
        num_stages_a=5,
        num_stages_b=5,
        num_stages_smem_sfa=5,
        num_stages_smem_sfb=5,
        num_stages_tmem_acc=1,
        num_stages_c_smem=2,
        tmem_ldst_max_num_regs=64,
    )
    values.update(overrides)
    return _dense_config(**values)


@pytest.mark.parametrize("single_expert", [0, 1])
def test_tile512_resource_contract(single_expert):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _make_pipeline_configs,
    )

    cfg = _tile512_config(dense_single_expert=single_expert)
    validate_config(cfg)
    assert cfg.tile_m * cfg.umma_cta_count == 512
    assert cfg.warp_layout.threads_per_cta == 256
    assert cfg.warp_layout.num_padding_warps == 1
    assert cfg.tmem_c_cols_per_stage == 512
    assert cfg.tmem_sfa_cols == cfg.tmem_sfb_cols == 32
    assert cfg.tmem_required_cols == cfg.tmem_capacity_columns == 576
    assert cfg.non_swap_epilogue_subtile_cnt == 8
    assert cfg.num_bytes_c_per_stage == 16384
    assert not cfg.aliases_c_scratch_with_ab
    pipes = _make_pipeline_configs(cfg)
    assert {
        name: pipes["smem_" + name].num_bytes for name in ("a", "b", "sfa", "sfb")
    } == dict(a=65536, b=32768, sfa=8192, sfb=8192)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(mma_m=128),
        dict(mma_n=128),
        dict(dense_m_repeats=3),
        dict(dense_m_repeats=1),
        dict(num_stages_tmem_acc=2),
        dict(use_max_tmem_overlap=1),
        dict(dense_rolled_epilogue=0),
        dict(use_tma_store=0),
        dict(use_tma_store=0, num_stages_c_smem=0),
        dict(
            num_stages_a=6, num_stages_b=6, num_stages_smem_sfa=6, num_stages_smem_sfb=6
        ),
    ],
)
def test_tile512_rejects_unsupported_geometry_and_resources(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError):
        validate_config(_tile512_config(**overrides))


@pytest.mark.parametrize("preload", [0, 1])
def test_tile512_generic_experts_require_aligned_weight_scale_stride(preload):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import _runtime_config

    cfg = _preloaded_epilogue_config(
        dense_single_expert=0, dense_preload_epilogue=preload
    )
    validate_config(cfg, problem_mnk=(2560, 1280, 768))
    assert _runtime_config(cfg, 768, out_hidden=1280).dense_single_expert == 0
    with pytest.raises(ValueError, match="weight scale indexing"):
        validate_config(cfg, problem_mnk=(2560, 1104, 768))
    with pytest.raises(ValueError, match="weight scale indexing"):
        _runtime_config(cfg, 768, out_hidden=1104)


def test_tile512_scratch_and_operand_rings_are_disjoint():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    tasks, _, _, _ = _build_schedule_validate(_tile512_config(), num_k_tiles=11)
    resources = {
        resource.name: resource
        for task in tasks
        for resource in task.src_resources + task.dst_resources
    }
    allocations = [
        getattr(resources[name], field)
        for name, field in (
            ("GmemC", "_alloc_sc"),
            ("SmemA", "_alloc_a"),
            ("SmemB", "_alloc_b"),
            ("SmemSfA", "_alloc_sfa"),
            ("SmemSfB", "_alloc_sfb"),
        )
    ]
    regions = sorted(
        (allocation.offset, allocation.offset + allocation.size_bytes)
        for allocation in allocations
    )
    assert all(
        left[1] <= right[0] for left, right in zip(regions, regions[1:], strict=False)
    )


@pytest.mark.parametrize("cluster_y", [1, 4])
def test_shared_sfb_payload_never_becomes_unicast(cluster_y):
    from dataclasses import replace
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import tma_payload_is_local

    cfg = _dense_config(cluster_dim_y=cluster_y)
    assert tma_payload_is_local(cfg, "sfb") is False
    assert (
        tma_payload_is_local(replace(cfg, _active_cga_shape=(2, 1, 1)), "sfb") is False
    )


def _eight_warp_config(tile_m=512, epi_n=32, c_stages=2, **overrides):
    values = dict(
        dense_epilogue_warps=8,
        epilogue_regs=128,
        dense_single_expert=1,
        dense_rolled_epilogue=1,
        dense_dkg_store_order=0,
        dense_rolled_mma=1,
        dense_async_sf=1,
        dense_fuse_tma_loads=1,
        dense_partition_tma_loads=1,
        dense_specialize_tma_cga=1,
        use_max_tmem_overlap=0,
        use_tma_oob_opt=0,
        use_global_scales=0,
        use_early_exit=0,
        use_work_throttle=0,
        num_stages_workid=1,
        epi_tile_n=epi_n,
        tmem_ldst_max_num_regs=epi_n,
        num_stages_c_smem=c_stages,
        mma_regs=80,
        load_regs=80,
        load_sf_regs=80,
        copy_sf_regs=80,
        workid_regs=80,
        padding_regs=80,
    )
    if tile_m == 256:
        values["num_stages_tmem_acc"] = 2
        # Make space for two C rings without changing their address layout.
        values.update(
            num_stages_a=7,
            num_stages_b=7,
            num_stages_smem_sfa=7,
            num_stages_smem_sfb=7,
        )
    elif epi_n == 64 and c_stages == 2:
        values.update(
            num_stages_a=4,
            num_stages_b=4,
            num_stages_smem_sfa=4,
            num_stages_smem_sfb=4,
        )
    values.update(overrides)
    return (_tile512_config if tile_m == 512 else _dense_config)(**values)


@pytest.mark.parametrize("tile_m", [256, 512])
@pytest.mark.parametrize("epi_n,c_stages", [(32, 2), (64, 1), (64, 2)])
def test_eight_warp_epilogue_resource_and_participant_contract(tile_m, epi_n, c_stages):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        host_clc_consumer_threads,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _make_pipeline_configs,
    )

    cfg = _eight_warp_config(tile_m, epi_n, c_stages)
    validate_config(cfg)
    four_warps = replace(cfg, dense_epilogue_warps=4)
    assert cfg.warp_layout.num_epilogue_warps == 8
    assert cfg.warp_layout.load_a_warp_idx == 8
    assert cfg.warp_layout.threads_per_cta == 384
    assert (
        cfg.non_swap_epilogue_subtile_cnt * 2
        == four_warps.non_swap_epilogue_subtile_cnt
    )
    assert cfg.num_bytes_c_smem_scratch == 2 * four_warps.num_bytes_c_smem_scratch
    pipes = _make_pipeline_configs(cfg)
    assert pipes["tmem_c"].consumer_group.size == 512
    for y in (1, 4):
        assert (
            host_clc_consumer_threads(cfg, 2, y)
            - host_clc_consumer_threads(four_warps, 2, y)
            == 128 * 2 * y
        )

    tasks, _, _, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    resources = {
        resource.name: resource
        for task in tasks
        for resource in task.src_resources + task.dst_resources
    }
    scratch = resources["GmemC"]._alloc_sc
    assert scratch.size_bytes == 2 * 128 * epi_n * 2 * c_stages
    for name, field in (
        ("SmemA", "_alloc_a"),
        ("SmemB", "_alloc_b"),
        ("SmemSfA", "_alloc_sfa"),
        ("SmemSfB", "_alloc_sfb"),
    ):
        operand = getattr(resources[name], field)
        assert (
            scratch.offset + scratch.size_bytes <= operand.offset
            or operand.offset + operand.size_bytes <= scratch.offset
        )


@pytest.mark.parametrize(
    "overrides",
    [
        dict(dense_epilogue_warps=0),
        dict(dense_epilogue_warps=6),
        dict(epilogue_regs=176),
        dict(dense_rolled_epilogue=0),
        dict(dense_single_expert=0),
        dict(epi_tile_m=64),
        dict(epi_tile_n=64, tmem_ldst_max_num_regs=32),
        dict(num_stages_c_smem=3),
        dict(use_tma_store=0),
        dict(use_max_tmem_overlap=1),
        dict(use_per_token_sf_a=1),
    ],
)
def test_eight_warp_epilogue_rejects_unsupported_configs(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError):
        validate_config(_eight_warp_config(**overrides))


def test_eight_warp_epilogue_accounts_for_both_scratch_rings():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    cfg = _eight_warp_config(
        epi_n=64,
        c_stages=2,
        num_stages_a=5,
        num_stages_b=5,
        num_stages_smem_sfa=5,
        num_stages_smem_sfb=5,
    )
    with pytest.raises(ValueError, match="exceed SMEM capacity"):
        validate_config(cfg)


def test_dense_epilogue_warps_cli():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_epilogue_warps"] == 4
    args = parser.parse_args(["--dense-epilogue-warps", "8", "--epilogue-regs", "128"])
    assert _parse_overrides(args)["dense_epilogue_warps"] == 8


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "tile_m,epi_n,c_stages", [(256, 32, 2), (512, 32, 2), (512, 64, 1), (512, 64, 2)]
)
@pytest.mark.parametrize(
    "cluster_y,dtype,mixed", [(1, "fp16", False), (4, "bf16", False), (4, "fp16", True)]
)
def test_eight_warp_epilogue_gpu(
    monkeypatch, tile_m, epi_n, c_stages, cluster_y, dtype, mixed
):
    """Partial M/N, C-ring reuse, and repeated nonzero launches with guards."""
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _eight_warp_config(
        tile_m,
        epi_n,
        c_stages,
        cluster_dim_y=cluster_y,
        dtype_c=int(DType.FP16 if dtype == "fp16" else DType.BF16),
    )

    # Keep random nonzero FP4 products within FP16 range while retaining
    # independent A/B scales. The comparison checks every logical output.
    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=1,
        num_tokens=4096 if mixed else 672,
        top_k=1,
        problem_n=4352 if mixed else 1104,
        problem_k=2304,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=mixed,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


def _preloaded_epilogue_config(**overrides):
    values = dict(
        dense_preload_epilogue=1,
        dense_dkg_store_order=1,
        dense_specialize_tma_cga=1,
        epilogue_regs=248,
        use_global_scales=0,
        use_early_exit=0,
        use_work_throttle=0,
        num_stages_workid=1,
        mma_regs=80,
        load_regs=80,
        load_sf_regs=80,
        copy_sf_regs=80,
        workid_regs=80,
        padding_regs=80,
    )
    values.update(overrides)
    return _tile512_config(**values)


@pytest.mark.parametrize("registers", [248, 256])
@pytest.mark.parametrize("dtype", ["FP16", "BF16"])
@pytest.mark.parametrize("global_scales", [0, 1])
def test_preloaded_epilogue_uses_original_accumulator_and_scratch(
    registers, dtype, global_scales
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = _preloaded_epilogue_config(
        epilogue_regs=registers,
        dtype_c=int(DType[dtype]),
        use_global_scales=global_scales,
    )
    validate_config(cfg)
    assert cfg.warp_layout.threads_per_cta == 256
    assert cfg.num_stages_tmem_acc == 1
    assert cfg.tmem_required_cols == 576
    assert cfg.num_bytes_c_smem_scratch == 32768
    _build_schedule_validate(cfg, num_k_tiles=9)


@pytest.mark.parametrize(
    "overrides",
    [
        dict(dense_preload_epilogue=2),
        dict(transpose_mma_output=1),
        dict(tile_n=128),
        dict(dense_m_repeats=1, dense_b_reuse=0),
        dict(dense_epilogue_warps=8),
        dict(dtype_c=1),  # FP32
        dict(epi_tile_n=32),
        dict(tmem_ldst_max_num_regs=32),
        dict(num_stages_c_smem=1),
        dict(epilogue_regs=176),
        dict(dense_dkg_store_order=0),
    ],
)
def test_preloaded_epilogue_rejects_unsupported_configs(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError):
        validate_config(_preloaded_epilogue_config(**overrides))


def test_preloaded_epilogue_is_opt_in():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_preload_epilogue"] == 0
    assert (
        _parse_overrides(parser.parse_args(["--dense-preload-epilogue", "1"]))[
            "dense_preload_epilogue"
        ]
        == 1
    )


def test_preloaded_store_prefix_cli_is_opt_in():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    name = "dense_preload_store_prefix_fragments"
    assert _parse_overrides(parser.parse_args([]))[name] == 0
    for count in (0, 2):
        args = parser.parse_args(["--dense-preload-store-prefix-fragments", str(count)])
        assert _parse_overrides(args)[name] == count
    for count in (-1, 1, 4, 8):
        with pytest.raises(SystemExit):
            parser.parse_args(["--dense-preload-store-prefix-fragments", str(count)])


@pytest.mark.parametrize(
    "overrides",
    [{"dense_preload_store_prefix_fragments": count} for count in (-1, 1, 4, 8)]
    + [{"dense_preload_epilogue": 0}, {"dense_epilogue_warps": 4}],
)
def test_preloaded_store_prefix_rejects_unsupported_configs(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    options = dict(
        dense_preload_store_prefix_fragments=2,
        dense_epilogue_warps=8,
        epilogue_regs=144,
    )
    options.update(overrides)
    with pytest.raises(ValueError, match="dense_preload_store_prefix_fragments"):
        validate_config(_preloaded_epilogue_config(**options))


def test_preloaded_store_prefix_has_distinct_compile_cache_key():
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    cfg = _preloaded_epilogue_config(dense_epilogue_warps=8, epilogue_regs=144)
    prefix_cfg = replace(cfg, dense_preload_store_prefix_fragments=2)
    assert stable_config_hash(cfg) != stable_config_hash(prefix_cfg)


def test_streaming_c_cli_is_opt_in():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_streaming_c"] == 0
    args = parser.parse_args(["--dense-streaming-c", "1"])
    assert _parse_overrides(args)["dense_streaming_c"] == 1
    with pytest.raises(SystemExit):
        parser.parse_args(["--dense-streaming-c", "2"])


@pytest.mark.parametrize("dtype", ["FP16", "BF16"])
def test_streaming_c_accepts_packed_output(dtype):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        validate_config,
    )

    cfg = _preloaded_epilogue_config(dense_streaming_c=1, dtype_c=int(DType[dtype]))
    validate_config(cfg)


@pytest.mark.parametrize(
    "overrides",
    [dict(dense_streaming_c=2), dict(dense_preload_epilogue=0)],
)
def test_streaming_c_rejects_unsupported_configs(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    options = dict(dense_streaming_c=1)
    options.update(overrides)
    with pytest.raises(ValueError, match="dense_streaming_c"):
        validate_config(_preloaded_epilogue_config(**options))


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize("cluster_y,mixed", [(1, False), (4, False), (4, True)])
@pytest.mark.parametrize("dtype", ["FP16", "BF16"])
def test_preloaded_epilogue_gpu(monkeypatch, cluster_y, mixed, dtype):
    """Repeated nonzero launches exercise release-before-store and C-ring reuse."""
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(cluster_dim_y=cluster_y, dtype_c=int(DType[dtype]))

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=1,
        num_tokens=4096 if mixed else 672,
        top_k=1,
        problem_n=4352 if mixed else 1104,
        problem_k=2304,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=mixed,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize("preload", [0, 1])
@pytest.mark.parametrize("raster", ["ALONG_M", "ALONG_N"])
@pytest.mark.parametrize(
    "cluster_y,experts,tokens,top_k,mixed",
    [(1, 3, 769, 2, False), (4, 3, 769, 2, False), (4, 32, 129, 8, True)],
)
def test_tile512_multi_expert_gpu(
    monkeypatch, preload, raster, cluster_y, experts, tokens, top_k, mixed
):
    """Exercise expert changes, padded token tails and repeated C-ring reuse."""
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        DType,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(
        dense_single_expert=0,
        dense_preload_epilogue=preload,
        epilogue_regs=248 if preload else 176,
        cluster_dim_y=cluster_y,
        dtype_c=int(DType.BF16),
        cta_raster_order=int(CtaRasterOrder[raster]),
    )

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=experts,
        num_tokens=tokens,
        top_k=top_k,
        problem_n=1280,
        problem_k=768,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=mixed,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


@pytest.mark.parametrize("group_m", [1, 2, 4, 8])
def test_grouped_n_raster_cli_and_config(group_m):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )
    from flashinfer.tllm_enums import WeightLayout

    parser = build_arg_parser()
    defaults = _parse_overrides(parser.parse_args([]))
    assert defaults["cta_raster_group_m"] == 1
    assert defaults["weight_layout"] == int(WeightLayout.MajorK)
    overrides = _parse_overrides(
        parser.parse_args(
            [
                "--cta-swizzle-type",
                "rasterizeAlongN",
                "--cta-raster-group-m",
                str(group_m),
                "--weight-layout",
                "block_major_k",
            ]
        )
    )
    cfg = _preloaded_epilogue_config(
        **{
            name: overrides[name]
            for name in ("cta_raster_order", "cta_raster_group_m", "weight_layout")
        }
    )
    validate_config(cfg)
    assert cfg.cta_raster_group_m == group_m
    assert cfg.uses_block_major_k_weight_b


@pytest.mark.parametrize(
    "overrides",
    [
        dict(cta_raster_group_m=0),
        dict(cta_raster_group_m=3),
        dict(cta_raster_group_m=2),  # Default M rasterization.
    ],
)
def test_grouped_n_raster_rejects_invalid_configs(overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    with pytest.raises(ValueError, match="cta_raster_group_m"):
        validate_config(_preloaded_epilogue_config(**overrides))


@pytest.mark.parametrize("group_m", [1, 2, 4, 8])
def test_grouped_n_raster_early_exit_preserves_valid_tiles(monkeypatch, group_m):
    from types import SimpleNamespace
    import cutlass
    from cutlass.experimental.task_scheduling.enums import TileSchedulerType
    from cutlass.experimental.task_scheduling.resources import TileSchedulerConfig
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.smem_misc_resources import (
        BatchedGemmWorkQueue,
        MixedCgaClcDynamicPersistentTileScheduler,
    )

    cfg = _dense_config(
        cluster_dim_y=1,
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        cta_raster_group_m=group_m,
        use_early_exit=1,
        use_clc_fast_drain=1,
    )
    validate_config(cfg)
    queue = BatchedGemmWorkQueue(
        name="WorkQueue",
        cfg=cfg,
        tile_scheduler_config=TileSchedulerConfig(
            tile_scheduler_type=TileSchedulerType.ClcDynamicPersistent,
            tile_scheduler_params=None,
        ),
        num_non_exiting_ctas_value=1,
    )
    count_m, count_n = 2 * group_m + 1, 3
    scheduler = SimpleNamespace(
        raster_group_m=group_m,
        params=SimpleNamespace(
            swizzle_size=1,
            _raster_along_m=False,
            cluster_shape_major_fdd=count_n,
            cluster_shape_minor_fdd=count_m,
            problem_layout_ncluster_mnl=SimpleNamespace(shape=(count_m, count_n, 1)),
            cluster_shape_mn=cfg.preferred_cluster_shape[:2],
        ),
    )
    rasterize = python_body(
        MixedCgaClcDynamicPersistentTileScheduler._swizzle_and_rasterize
    )
    should_skip = python_body(BatchedGemmWorkQueue.should_skip_work_tile)
    maybe_drain = python_body(BatchedGemmWorkQueue._maybe_fast_drain_skipped_tile)
    monkeypatch.setattr(cutlass, "Int32", int)
    skipped = False
    drained = []
    monkeypatch.setattr(queue, "_get_consumer_var_from_ts", lambda _: skipped)
    monkeypatch.setattr(queue, "_clc_fast_drain_all", lambda: drained.append(True))

    visited = []
    for z in range(count_m * count_n):
        tile_idx = rasterize(scheduler, 0, 0, z)
        skipped = should_skip(queue, SimpleNamespace(tile_idx=tile_idx))
        if not skipped:
            visited.append(tile_idx)
        maybe_drain(queue)
        if drained:
            break

    # With grouping, the first inactive M cluster precedes valid N tiles.
    # Ordinary early exit must skip that cluster without cancelling later work.
    assert visited == [(0, n, 0) for n in range(count_n)]
    assert bool(drained) == (group_m == 1)


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize("weight_layout", ["MajorK", "BlockMajorK"])
@pytest.mark.parametrize("early_exit", [0, 1])
@pytest.mark.parametrize(
    "cluster_y,group_m,experts,tokens,top_k,mixed",
    [
        (1, 2, 3, 513, 1, False),  # Three M clusters: incomplete final pair.
        (2, 4, 3, 769, 2, False),  # Six M clusters: incomplete group of four.
        (4, 2, 32, 129, 8, True),  # Persistent mixed-cluster reuse and N tails.
    ],
)
def test_tile512_grouped_n_raster_gpu(
    monkeypatch,
    weight_layout,
    early_exit,
    cluster_y,
    group_m,
    experts,
    tokens,
    top_k,
    mixed,
):
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.tllm_enums import WeightLayout
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        DType,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(
        dense_single_expert=0,
        cluster_dim_y=cluster_y,
        dtype_c=int(DType.BF16),
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        cta_raster_group_m=group_m,
        weight_layout=int(WeightLayout[weight_layout]),
        use_early_exit=early_exit,
        use_clc_fast_drain=early_exit,
    )

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=experts,
        num_tokens=tokens,
        top_k=top_k,
        problem_n=2304,
        problem_k=768,
        early_exit_max_token_ctas=128 if early_exit else 0,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=mixed,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


@pytest.mark.parametrize("dim_x,dim_y", [(2, 1), (4, 1), (4, 2), (4, 4)])
def test_dense_fused_cga4_elects_weight_payload_issuers(monkeypatch, dim_x, dim_y):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga

    cfg = _preloaded_epilogue_config(cluster_dim_x=4, cluster_dim_y=dim_y)
    monkeypatch.setattr(cga, "Int32", int)
    issued = []
    monkeypatch.setattr(
        cga, "_issue_tma_load_impl", lambda *args, **kwargs: issued.append(args)
    )
    for operand in ("a", "b", "sfa", "sfb"):
        issued.clear()
        for y in range(dim_y):
            for x in range(dim_x):
                rank = x + dim_x * y
                monkeypatch.setattr(
                    cga,
                    "active_cga_coords",
                    lambda cfg, x=x, y=y, rank=rank: (dim_x, dim_y, 1, rank, x, y),
                )
                python_body(cga.issue_tma_load)(
                    cfg, operand, rank, None, None, None, None
                )
        expected = [
            x + dim_x * y
            for y in range(dim_y)
            for x in range(dim_x)
            if operand in ("a", "sfa") or x < 2
        ]
        assert [args[2] for args in issued] == expected


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize("weight_layout", ["MajorK", "BlockMajorK"])
@pytest.mark.parametrize("cluster_y", [1, 2, 4])
@pytest.mark.parametrize("raster", ["ALONG_M", "ALONG_N"])
@pytest.mark.parametrize("streaming_c", [0, 1])
def test_tile512_cga4_gpu(monkeypatch, weight_layout, cluster_y, raster, streaming_c):
    """Require both physical shapes, including 4x1/2x1 with identical Y."""
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.tllm_enums import WeightLayout
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        DType,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(
        dense_single_expert=0,
        cluster_dim_x=4,
        cluster_dim_y=cluster_y,
        dtype_c=int(DType.BF16),
        cta_raster_order=int(CtaRasterOrder[raster]),
        weight_layout=int(WeightLayout[weight_layout]),
        dense_streaming_c=streaming_c,
    )

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=32,
        num_tokens=129,
        top_k=8,
        problem_n=2304,
        problem_k=768,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=True,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


def test_split_tma_cli_is_opt_in():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_split_tma_pipeline"] == 0
    assert (
        _parse_overrides(parser.parse_args(["--dense-split-tma-pipeline", "1"]))[
            "dense_split_tma_pipeline"
        ]
        == 1
    )


@pytest.mark.parametrize("stages_a,stages_b", [(5, 5), (4, 7)])
def test_split_tma_resources_and_participants(stages_a, stages_b):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        HostCgaTopology,
        host_clc_consumer_threads,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = _preloaded_epilogue_config(
        dense_split_tma_pipeline=1,
        cluster_dim_x=4,
        dense_single_expert=0,
        num_stages_a=stages_a,
        num_stages_smem_sfa=stages_a,
        num_stages_b=stages_b,
        num_stages_smem_sfb=stages_b,
    )
    validate_config(cfg)
    w = cfg.warp_layout
    assert (
        w.load_a_warp_idx,
        w.load_b_warp_idx,
        w.mma_warp_idx,
        w.workid_warp_idx,
    ) == (4, 5, 6, 7)
    assert w.load_sfa_warp_idx == w.load_a_warp_idx
    assert w.load_sfb_warp_idx == w.load_b_warp_idx
    assert w.num_padding_warps == 0 and w.threads_per_cta == 256
    shared = replace(cfg, dense_split_tma_pipeline=0)
    for x, y in [(2, 1), (4, 1), (4, 2), (4, 4)]:
        # Split B/SFB excludes odd X>=2; the fully fused task also issues
        # activation payloads on those CTAs and must retain them.
        assert host_clc_consumer_threads(cfg, x, y) == (
            host_clc_consumer_threads(shared, x, y) - (x // 2 - 1) * y * 32
        )
    tasks, _, allocator, _ = _build_schedule_validate(cfg, num_k_tiles=9)
    for x, y in [(2, 1), (4, 1), (4, 2), (4, 4)]:
        topo = HostCgaTopology(x, y)
        actual_consumers = sum(
            task.num_warps * 32
            for task in tasks
            for yy in range(y)
            for xx in range(x)
            if topo.selected(CgaPolicy(getattr(task, "cga_policy", 0)), xx, yy)
            and task.run_only_on_cta_id in (None, xx + x * yy)
        )
        assert host_clc_consumer_threads(cfg, x, y) == actual_consumers
    tasks = {task.name: task for task in tasks}
    a, sa = tasks["DenseActivationLoad"].dst_resources
    b, sb = tasks["DenseWeightLoad"].dst_resources
    assert a.pipeline_group is sa.pipeline_group
    assert b.pipeline_group is sb.pipeline_group
    assert a.pipeline_group is not b.pipeline_group
    assert a.pipeline_group.pipeline_config.num_stages == stages_a
    assert b.pipeline_group.pipeline_config.num_stages == stages_b
    assert a.pipeline_group.pipeline_config.num_bytes == 2 * (
        cfg.num_bytes_a_per_stage + cfg.num_bytes_sfa_per_stage
    )
    assert b.pipeline_group.pipeline_config.num_bytes == 2 * (
        cfg.num_bytes_b_tma_per_stage + cfg.num_bytes_sfb_per_stage
    )
    assert {a.pipeline_group.split_operand, b.pipeline_group.split_operand} == {
        "a",
        "b",
    }
    assert allocator._barrier_bytes >= 16 * (stages_a + stages_b)
    allocations = [a._alloc_a, b._alloc_b, sa._alloc_sfa, sb._alloc_sfb]
    for i, first in enumerate(allocations):
        for second in allocations[i + 1 :]:
            assert (
                first.offset + first.size_bytes <= second.offset
                or second.offset + second.size_bytes <= first.offset
            )


@pytest.mark.parametrize(
    "options",
    [
        dict(dense_split_tma_pipeline=2),
        dict(dense_fuse_tma_loads=0),
        dict(tile_scheduler=0),
        dict(use_work_throttle=1),
        dict(num_stages_smem_sfa=4),
        dict(num_stages_b=4),
    ],
)
def test_split_tma_rejects_invalid_configs(options):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    settings = dict(dense_split_tma_pipeline=1)
    settings.update(options)
    with pytest.raises(ValueError):
        validate_config(_preloaded_epilogue_config(**settings))


@pytest.mark.parametrize("dim_x,dim_y", [(2, 1), (2, 4), (4, 1), (4, 2), (4, 4)])
@pytest.mark.parametrize("weights", [False, True])
def test_split_tma_releases_all_and_only_pair_users(monkeypatch, dim_x, dim_y, weights):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga

    monkeypatch.setattr(cga, "Int32", int)
    monkeypatch.setattr(cga.cutlass, "range_constexpr", range)
    for name in (
        "_rank_bit",
        "_column_mask",
        "_row_mask",
        "_atom_mask",
        "_project_payload_to_barrier_owners",
        "cga_policy_mask",
    ):
        monkeypatch.setattr(cga, name, python_body(getattr(cga, name)))
    arrivals = [0] * (dim_x * dim_y)
    for y in range(dim_y):
        for x in range(0, dim_x, 2):
            monkeypatch.setattr(
                cga,
                "active_cga_coords",
                lambda cfg, x=x, y=y: (dim_x, dim_y, 1, x + y * dim_x, x, y),
            )
            mask, count = python_body(cga.split_tma_release)(
                _preloaded_epilogue_config(
                    cluster_dim_x=dim_x, cluster_dim_y=dim_y, dense_split_tma_pipeline=1
                ),
                weights,
            )
            # B/SFB share an N tile across X atoms. A/SFA share an M tile
            # across Y; both CTAs of each consumer atom protect their loaders.
            receivers = {
                xx + yy * dim_x
                for yy in range(dim_y)
                for xx in range(dim_x)
                if (
                    (yy == y and (xx < 2 or xx % 2 == 0))
                    if weights
                    else xx // 2 == x // 2
                )
            }
            assert mask == sum(1 << rank for rank in receivers)
            for rank in receivers:
                arrivals[rank] += 1
    assert arrivals == [
        count if not weights or x < 2 or x % 2 == 0 else 0
        for y in range(dim_y)
        for x in range(dim_x)
    ]
    assert count == (dim_x // 2 if weights else dim_y)


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "x,y,stages_a,stages_b,dtype,layout,raster,specialized,preload,shape",
    [
        (4, 4, 5, 5, "BF16", "BlockMajorK", "ALONG_N", 1, 1, (129, 2304, 2304)),
        (4, 1, 5, 5, "FP16", "MajorK", "ALONG_M", 1, 1, (129, 2304, 2304)),
        (4, 2, 4, 7, "BF16", "MajorK", "ALONG_N", 1, 1, (129, 2304, 2304)),
        (2, 4, 4, 7, "FP16", "BlockMajorK", "ALONG_M", 1, 1, (129, 2304, 2304)),
        (2, 1, 5, 5, "BF16", "BlockMajorK", "ALONG_N", 1, 1, (129, 2304, 2304)),
        (4, 4, 5, 5, "BF16", "BlockMajorK", "ALONG_N", 0, 1, (129, 2304, 2304)),
        (4, 4, 5, 5, "BF16", "BlockMajorK", "ALONG_N", 1, 0, (129, 2304, 2304)),
        # Full 1024-row experts exercise both M fragments. Cover the first
        # stage and wraparound of equal and unequal operand-pair rings.
        (4, 4, 5, 5, "BF16", "BlockMajorK", "ALONG_N", 1, 1, (4096, 1280, 2816)),
        (4, 4, 4, 7, "BF16", "MajorK", "ALONG_M", 1, 1, (4096, 1280, 2816)),
        (4, 4, 5, 5, "FP16", "BlockMajorK", "ALONG_N", 1, 1, (4096, 1280, 256)),
    ],
)
def test_split_tma_gpu(
    monkeypatch,
    x,
    y,
    stages_a,
    stages_b,
    dtype,
    layout,
    raster,
    specialized,
    preload,
    shape,
):
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.tllm_enums import WeightLayout
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        CtaRasterOrder,
        DType,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(
        dense_split_tma_pipeline=1,
        dense_single_expert=0,
        cluster_dim_x=x,
        cluster_dim_y=y,
        num_stages_a=stages_a,
        num_stages_smem_sfa=stages_a,
        num_stages_b=stages_b,
        num_stages_smem_sfb=stages_b,
        dtype_c=int(DType[dtype]),
        weight_layout=int(WeightLayout[layout]),
        cta_raster_order=int(CtaRasterOrder[raster]),
        dense_specialize_tma_cga=specialized,
        dense_preload_epilogue=preload,
        epilogue_regs=248 if preload else 176,
        dense_streaming_c=int(preload and dtype == "BF16"),
    )

    def bounded_sf(shape, *, cfg, device, layout_kind):
        return runner._random_uint8(shape, low=0x28, high=0x41, device=device)

    monkeypatch.setattr(runner, "_random_sf_bytes", bounded_sf)
    assert runner.reference_check(
        num_experts=32,
        num_tokens=shape[0],
        top_k=8,
        problem_n=shape[1],
        problem_k=shape[2],
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=(x, y) != (2, 1),
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


@pytest.mark.parametrize("registers", [144, 160, 168, 176])
@pytest.mark.parametrize("split", [0, 1])
@pytest.mark.parametrize("c_stages", [1, 2])
@pytest.mark.parametrize("global_scales", [0, 1])
@pytest.mark.parametrize("store_prefix", [0, 2])
def test_cooperative_preloaded_epilogue_resources(
    registers, split, c_stages, global_scales, store_prefix
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _make_pipeline_configs,
    )

    cfg = _preloaded_epilogue_config(
        dense_epilogue_warps=8,
        num_stages_c_smem=c_stages,
        epilogue_regs=registers,
        dense_split_tma_pipeline=split,
        dense_single_expert=0,
        cluster_dim_x=4,
        use_global_scales=global_scales,
        dense_preload_store_prefix_fragments=store_prefix,
    )
    validate_config(cfg)
    assert cfg.warp_layout.threads_per_cta == 384
    assert cfg.num_bytes_c_smem_scratch == 16384 * c_stages
    assert cfg.num_stages_a == cfg.num_stages_b == 5
    assert cfg.num_stages_tmem_acc == 1
    assert cfg.tmem_required_cols == 576
    assert _make_pipeline_configs(cfg)["tmem_c"].consumer_group.size == 512
    tasks, _, _, _ = _build_schedule_validate(cfg, num_k_tiles=11)
    resources = {
        resource.name: resource
        for task in tasks
        for resource in task.src_resources + task.dst_resources
    }
    scratch = resources["GmemC"]._alloc_sc
    assert scratch.size_bytes == 16384 * c_stages
    for name, field in (
        ("SmemA", "_alloc_a"),
        ("SmemB", "_alloc_b"),
        ("SmemSfA", "_alloc_sfa"),
        ("SmemSfB", "_alloc_sfb"),
    ):
        operand = getattr(resources[name], field)
        assert (
            scratch.offset + scratch.size_bytes <= operand.offset
            or operand.offset + operand.size_bytes <= scratch.offset
        )


@pytest.mark.gpu
@pytest.mark.arch_rubin
@pytest.mark.parametrize(
    "split,dtype,experts,tokens,top_k,n,k,x,raster,stages_a,stages_b,registers,c_stages",
    [
        (1, "BF16", 32, 4096, 8, 1280, 2816, 4, "ALONG_N", 5, 5, 144, 2),
        (1, "FP16", 32, 4096, 8, 1280, 256, 4, "ALONG_N", 5, 5, 160, 2),
        (1, "BF16", 32, 129, 8, 1280, 2816, 4, "ALONG_N", 5, 5, 168, 2),
        (1, "FP16", 32, 4096, 8, 1280, 2816, 4, "ALONG_M", 4, 7, 144, 2),
        (0, "BF16", 1, 4096, 1, 4352, 2304, 2, "ALONG_N", 5, 5, 144, 2),
        (0, "FP16", 3, 769, 2, 1280, 2304, 2, "ALONG_M", 5, 5, 144, 2),
        (1, "BF16", 32, 4096, 8, 1280, 2816, 4, "ALONG_N", 5, 5, 144, 1),
        (1, "FP16", 32, 4096, 8, 1280, 256, 4, "ALONG_N", 5, 6, 160, 1),
        (1, "FP16", 32, 129, 8, 1280, 2816, 4, "ALONG_M", 5, 6, 168, 1),
        # Six-way MMA: an exact body, both-ring wraparound (LCM=30), and
        # a one-tile remainder on the single-expert path.
        (1, "BF16", 32, 4096, 8, 1280, 1536, 4, "ALONG_N", 5, 6, 144, 1),
        (1, "BF16", 32, 4096, 8, 1280, 7936, 4, "ALONG_N", 5, 6, 144, 1),
        (1, "BF16", 1, 4096, 1, 4352, 1792, 2, "ALONG_N", 5, 6, 144, 1),
        (0, "BF16", 1, 4096, 1, 4352, 2304, 2, "ALONG_N", 5, 5, 144, 1),
    ],
)
def test_cooperative_preloaded_epilogue_gpu(
    monkeypatch,
    split,
    dtype,
    experts,
    tokens,
    top_k,
    n,
    k,
    x,
    raster,
    stages_a,
    stages_b,
    registers,
    c_stages,
):
    import torch
    from dataclasses import fields
    from flashinfer.utils import get_compute_capability
    from flashinfer.tllm_enums import WeightLayout
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run as runner
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        CtaRasterOrder,
    )

    if not torch.cuda.is_available() or get_compute_capability(
        torch.device("cuda")
    ) != (10, 7):
        pytest.skip("Requires SM107")
    monkeypatch.setenv("FLASHINFER_PRIMS_TS_DEBUG_CHECKS", "0")
    cfg = _preloaded_epilogue_config(
        dense_epilogue_warps=8,
        num_stages_c_smem=c_stages,
        epilogue_regs=registers,
        dense_split_tma_pipeline=split,
        dense_retain_a=int(dtype == "BF16"),
        dense_a_l2_promotion=256 if dtype == "BF16" else 128,
        dense_single_expert=int(experts == 1),
        cluster_dim_x=x,
        cluster_dim_y=4,
        dtype_c=int(DType[dtype]),
        cta_raster_order=int(CtaRasterOrder[raster]),
        weight_layout=int(
            WeightLayout.BlockMajorK if raster == "ALONG_N" else WeightLayout.MajorK
        ),
        num_stages_a=stages_a,
        num_stages_smem_sfa=stages_a,
        num_stages_b=stages_b,
        num_stages_smem_sfb=stages_b,
        dense_streaming_c=int(dtype == "BF16"),
    )
    monkeypatch.setattr(
        runner,
        "_random_sf_bytes",
        lambda shape, *, cfg, device, layout_kind: runner._random_uint8(
            shape, low=0x28, high=0x41, device=device
        ),
    )
    assert runner.reference_check(
        num_experts=experts,
        num_tokens=tokens,
        top_k=top_k,
        problem_n=n,
        problem_k=k,
        repeat_launches=3,
        output_guard_elements=256,
        require_mixed_cga_runtime=experts != 3,
        **{field.name: getattr(cfg, field.name) for field in fields(cfg) if field.init},
    )


@pytest.mark.parametrize("enabled", [0, 1])
@pytest.mark.parametrize("local", [False, True])
def test_retain_a_cache_policy_preserves_payload_and_barriers(
    monkeypatch, enabled, local
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_cga as cga

    cfg = _preloaded_epilogue_config(dense_retain_a=enabled)
    policy = object()
    monkeypatch.setattr(cga, "_retained_a_cache_policy", lambda: policy)
    monkeypatch.setattr(cga, "tma_payload_is_local", lambda cfg, operand: local)
    calls = []
    monkeypatch.setattr(
        cga.prims,
        "cp_async_bulk_tensor_shared_cluster_global",
        lambda *args, **kwargs: calls.append((args, kwargs)),
    )
    for operand in ("a", "sfa", "b", "sfb"):
        calls.clear()
        python_body(cga._issue_tma_load_impl)(
            cfg, operand, "dst", "desc", "coords", "barrier", 0x5
        )
        assert len(calls) == 1
        args, kw = calls[0]
        assert args == ("dst", "desc", "coords", "barrier", [])
        assert kw["multicast_mask"] == (None if local else 0x5)
        assert kw["group"] == cfg.cta_group
        assert kw["l2_cache_hint"] is (
            policy if enabled and operand in ("a", "sfa") else None
        )


def test_retain_a_cli_and_validation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        build_arg_parser,
        _parse_overrides,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_retain_a"] == 0
    assert (
        _parse_overrides(parser.parse_args(["--dense-retain-a", "1"]))["dense_retain_a"]
        == 1
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--dense-retain-a", "2"])
    validate_config(_preloaded_epilogue_config(dense_retain_a=1))
    with pytest.raises(ValueError, match="dense_retain_a"):
        validate_config(_preloaded_epilogue_config(dense_retain_a=2))
    with pytest.raises(ValueError, match="dense_retain_a"):
        validate_config(
            _preloaded_epilogue_config(
                dense_retain_a=1,
                dense_fuse_tma_pipeline=0,
                dense_m_repeats=1,
                dense_b_reuse=0,
            )
        )


@pytest.mark.parametrize("promotion_bytes", [0, 128, 256])
@pytest.mark.parametrize("slot", [0, 5])
@pytest.mark.parametrize("prepare_only", [False, True])
def test_a_promotion_preserves_descriptor_arguments(
    monkeypatch, promotion_bytes, slot, prepare_only
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_kernel as kernel

    cfg = _preloaded_epilogue_config(dense_a_l2_promotion=promotion_bytes)
    calls = []
    result = object()

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    monkeypatch.setattr(kernel, "_prepared_tensor_map", capture)
    options = dict(tensor=object(), box_dims=(256, 64, 1), stride_order=(0, 1, 2))
    actual = python_body(kernel._prepared_a_tensor_map)(
        cfg, "storage", slot, prepare_only, loc="loc", ip="ip", **options
    )
    expected_policy = {
        0: kernel.cuda.TensorMapL2Promotion.none,
        128: kernel.cuda.TensorMapL2Promotion.l2_128b,
        256: kernel.cuda.TensorMapL2Promotion.l2_256b,
    }[promotion_bytes]
    assert actual is result
    assert calls == [
        (
            ("storage", slot, prepare_only),
            dict(options, l2_promotion=expected_policy, loc="loc", ip="ip"),
        )
    ]


def test_a_promotion_cli_and_validation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        build_arg_parser,
        _parse_overrides,
    )

    parser = build_arg_parser()
    assert _parse_overrides(parser.parse_args([]))["dense_a_l2_promotion"] == 0
    for value in (0, 128, 256):
        args = parser.parse_args(["--dense-a-l2-promotion", str(value)])
        assert _parse_overrides(args)["dense_a_l2_promotion"] == value
        validate_config(_preloaded_epilogue_config(dense_a_l2_promotion=value))
    for value in (-1, 1, 64, 512):
        with pytest.raises(SystemExit):
            parser.parse_args(["--dense-a-l2-promotion", str(value)])
        with pytest.raises(ValueError, match="dense_a_l2_promotion"):
            validate_config(_preloaded_epilogue_config(dense_a_l2_promotion=value))
    with pytest.raises(ValueError, match="dense_a_l2_promotion"):
        validate_config(
            _preloaded_epilogue_config(
                dense_a_l2_promotion=256,
                dense_fuse_tma_pipeline=0,
                dense_preload_epilogue=0,
                dense_m_repeats=1,
                dense_b_reuse=0,
            )
        )


@pytest.mark.parametrize("stages_b,unroll", [(5, 5), (6, 6)])
def test_cooperative_split_mma_schedule_and_six_stage_storage(stages_b, unroll):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = _preloaded_epilogue_config(
        dense_split_tma_pipeline=1,
        dense_epilogue_warps=8,
        epilogue_regs=144,
        num_stages_c_smem=1,
        num_stages_b=stages_b,
        num_stages_smem_sfb=stages_b,
        dense_single_expert=0,
        cluster_dim_x=4,
    )
    validate_config(cfg)
    tasks, _, _, _ = _build_schedule_validate(cfg, num_k_tiles=31)
    tasks = {task.name: task for task in tasks}
    assert tasks["MmaTask0"].unroll == unroll
    assert tasks["DenseActivationLoad"].unroll == 1
    assert tasks["DenseWeightLoad"].unroll == 1
    a, sa = tasks["DenseActivationLoad"].dst_resources
    b, sb = tasks["DenseWeightLoad"].dst_resources
    assert a.pipeline_group.pipeline_config.num_stages == 5
    assert b.pipeline_group.pipeline_config.num_stages == stages_b
    assert sa.pipeline_group is a.pipeline_group
    assert sb.pipeline_group is b.pipeline_group
    resources = {
        resource.name: resource
        for task in tasks.values()
        for resource in task.src_resources + task.dst_resources
    }
    scratch = resources["GmemC"]._alloc_sc
    assert scratch.size_bytes == 16384
    allocations = [a._alloc_a, b._alloc_b, sa._alloc_sfa, sb._alloc_sfb, scratch]
    for i, first in enumerate(allocations):
        for second in allocations[i + 1 :]:
            assert (
                first.offset + first.size_bytes <= second.offset
                or second.offset + second.size_bytes <= first.offset
            )


@pytest.mark.parametrize("warps,preload", [(4, 0), (4, 1), (8, 0), (8, 1)])
def test_dkg_store_order_requires_supported_store_protocol(warps, preload):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import validate_config

    cfg = _preloaded_epilogue_config(
        dense_epilogue_warps=warps,
        dense_preload_epilogue=preload,
        epilogue_regs=(144 if preload else 128) if warps == 8 else 248,
        num_stages_a=4,
        num_stages_b=4,
        num_stages_smem_sfa=4,
        num_stages_smem_sfb=4,
    )
    if warps == 8 and not preload:
        with pytest.raises(ValueError, match="eight warps requires preload"):
            validate_config(cfg)
    else:
        validate_config(cfg)
