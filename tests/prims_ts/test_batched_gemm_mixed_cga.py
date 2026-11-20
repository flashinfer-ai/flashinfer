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

"""Host-side regressions for wider and flexible BatchedGemm CGAs."""

import pytest


def _bf16_mixed_config(**overrides):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        compute_warp_layout,
        make_config,
        validate_config,
    )

    values = {
        "umma_cta_count": 2,
        "cluster_dim_x": 4,
        "cluster_dim_y": 2,
        "fallback_cluster_dim_x": 2,
        "fallback_cluster_dim_y": 1,
        "tile_scheduler": int(TileScheduler.PERSISTENT),
        "dtype_a": int(DType.BF16),
        "dtype_b": int(DType.BF16),
        "dtype_c": int(DType.BF16),
        "tile_k": 64,
        "mma_m": 256,
        "mma_k": 16,
        "mma_n": 16,
        "tile_n": 16,
        "use_tma_store": 1,
    }
    values.update(overrides)
    cfg = make_config(**values)
    compute_warp_layout(cfg)
    validate_config(cfg)
    return cfg


def _fp4_fc2_mixed_config(**overrides):
    """Exact plain-FC2 configuration used by the mixed-CGA GPU reproducer."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        TileScheduler,
        compute_warp_layout,
        make_config,
        uniform_pipeline_stage_overrides,
        validate_config,
    )

    values = {
        "route_act": 0,
        "act_kind": 0,
        "sf_layout_b": 1,
        "umma_cta_count": 2,
        "cluster_dim_x": 4,
        "cluster_dim_y": 2,
        "fallback_cluster_dim_x": 2,
        "fallback_cluster_dim_y": 1,
        "tile_n": 16,
        "tile_k": 512,
        "mma_m": 256,
        "mma_n": 16,
        "mma_k": 64,
        "dtype_a": int(DType.E2M1),
        "dtype_b": int(DType.E2M1),
        "dtype_c": int(DType.BF16),
        "epilogue_regs": 128,
        "mma_regs": 48,
        "load_regs": 48,
        "load_sf_regs": 48,
        "copy_sf_regs": 48,
        "workid_regs": 48,
        "padding_regs": 48,
        "use_unroll_loop_2x_for_mma": 0,
        "use_max_tmem_overlap": 0,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
        "tile_scheduler": int(TileScheduler.PERSISTENT),
        "num_stages_tmem_acc": 2,
        **uniform_pipeline_stage_overrides(5),
    }
    values.update(overrides)
    cfg = make_config(**values)
    compute_warp_layout(cfg)
    validate_config(cfg)
    return cfg


def _class_a_hang_config():
    """Return the SM107 2-CTA register split that exposed the peer-CTA hang."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        TileScheduler,
        compute_warp_layout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        sm_arch=107,
        batch_mode=int(BatchMode.BATCH_N),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        act_kind=int(ActKind.SWIGLU),
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.AUTO),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        umma_cta_count=2,
        cluster_dim_x=2,
        tile_n=64,
        tile_k=256,
        epi_tile_n=64,
        mma_m=256,
        mma_n=64,
        mma_k=128,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        transpose_mma_output=1,
        num_load_b_warps=0,
        num_load_sfb_warps=4,
        num_stages_a=9,
        num_stages_b=9,
        num_stages_smem_sfa=10,
        num_stages_smem_sfb=10,
        num_stages_tmem_acc=5,
        num_stages_tmem_sfa=6,
        num_stages_tmem_sfb=7,
        num_stages_c_smem=2,
        use_tma_store=0,
        use_early_exit=1,
        use_clc_fast_drain=1,
        use_work_throttle=1,
        use_global_scales=1,
        use_tma_oob_opt=1,
        epilogue_regs=192,
        mma_regs=32,
        load_regs=32,
        load_sf_regs=48,
        load_a_regs=32,
        load_b_regs=48,
        load_sfa_regs=32,
        copy_sf_regs=32,
        padding_regs=48,
        workid_regs=48,
    )
    compute_warp_layout(cfg)
    validate_config(cfg)
    return cfg


def test_4x4_reference_masks_are_consistent():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import HostCgaTopology

    topo = HostCgaTopology(4, 4)

    assert topo.payload_mask_a(0) == 0x1111
    assert topo.payload_mask_a(1) == 0x2222
    assert topo.full_barrier_mask(topo.payload_mask_a(0)) == 0x1111
    assert topo.full_barrier_mask(topo.payload_mask_a(1)) == 0x1111
    assert topo.release_mask_a(0) == 0x0003
    assert topo.release_mask_a(1) == 0x0003

    assert topo.payload_mask_b(0, 1) == 0x0050
    assert topo.payload_mask_b(1, 1) == 0x00A0
    assert topo.full_barrier_mask(topo.payload_mask_b(0, 1)) == 0x0050
    assert topo.full_barrier_mask(topo.payload_mask_b(1, 1)) == 0x0050
    assert topo.release_mask_b(1) == 0x0030

    assert topo.payload_mask_sfb(1) == 0x00F0
    assert topo.full_barrier_mask(topo.payload_mask_sfb(1)) == 0x0050
    assert topo.release_mask_sfb(1) == 0x0010


def test_exact_fc2_4x2_mask_triples_are_consistent():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import HostCgaTopology

    topo = HostCgaTopology(4, 2)

    assert topo.payload_mask_a(0) == 0x11
    assert topo.payload_mask_a(1) == 0x22
    assert topo.payload_mask_a(2) == 0x44
    assert topo.payload_mask_a(3) == 0x88
    assert topo.full_barrier_mask(topo.payload_mask_a(0)) == 0x11
    assert topo.full_barrier_mask(topo.payload_mask_a(1)) == 0x11
    assert topo.full_barrier_mask(topo.payload_mask_a(2)) == 0x44
    assert topo.full_barrier_mask(topo.payload_mask_a(3)) == 0x44
    assert topo.release_mask_a(0) == 0x03
    assert topo.release_mask_a(2) == 0x0C

    assert topo.payload_mask_b(0, 0) == 0x05
    assert topo.payload_mask_b(1, 0) == 0x0A
    assert topo.payload_mask_b(0, 1) == 0x50
    assert topo.payload_mask_b(1, 1) == 0xA0
    assert topo.full_barrier_mask(topo.payload_mask_b(0, 0)) == 0x05
    assert topo.full_barrier_mask(topo.payload_mask_b(1, 1)) == 0x50
    assert topo.release_mask_b(0) == 0x03
    assert topo.release_mask_b(1) == 0x30


def test_4x4_reference_task_participant_counts():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        HostCgaTopology,
    )

    topo = HostCgaTopology(4, 4)
    assert topo.participant_count(CgaPolicy.ALL) == 16
    assert topo.participant_count(CgaPolicy.ATOM_LEADER) == 8
    assert topo.participant_count(CgaPolicy.MCAST_A) == 4
    assert topo.participant_count(CgaPolicy.MCAST_B) == 8
    assert topo.participant_count(CgaPolicy.MCAST_SFB) == 4
    assert topo.participant_count(CgaPolicy.ONE_X_COLUMN) == 4


@pytest.mark.parametrize(
    ("policy", "expected_4x2", "expected_2x1"),
    (
        ("ALL", 0xFF, 0x3),
        ("ATOM_LEADER", 0x55, 0x1),
        ("ONE_Y_ROW", 0x0F, 0x3),
        ("FIRST_X_ATOM_PER_Y", 0x33, 0x3),
        ("ONE_X_COLUMN", 0x11, 0x1),
        ("CTA_ZERO", 0x01, 0x1),
        ("MCAST_A", 0x0F, 0x3),
        ("MCAST_B", 0x33, 0x3),
        ("MCAST_SFB", 0x11, 0x1),
    ),
)
def test_policy_selection_matches_documented_rank_diagram(
    policy, expected_4x2, expected_2x1
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        HostCgaTopology,
    )

    def selected_mask(topo, selected_policy):
        return sum(
            topo.rank_bit(x, y)
            for y in range(topo.dim_y)
            for x in range(topo.dim_x)
            if topo.selected(selected_policy, x, y)
        )

    selected_policy = CgaPolicy[policy]
    assert selected_mask(HostCgaTopology(4, 2), selected_policy) == expected_4x2
    assert selected_mask(HostCgaTopology(2, 1), selected_policy) == expected_2x1


def test_operand_multicast_directions_match_trtllm_mapping():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaMcastDirection,
        tma_mcast_direction,
    )

    assert tma_mcast_direction("a") == CgaMcastDirection.ALONG_Y
    assert tma_mcast_direction("sfa") == CgaMcastDirection.ALONG_Y
    assert tma_mcast_direction("b") == CgaMcastDirection.ACROSS_X_ATOMS
    assert tma_mcast_direction("sfb") == CgaMcastDirection.ACROSS_X_ATOMS
    assert set(CgaMcastDirection) == {
        CgaMcastDirection.ALONG_Y,
        CgaMcastDirection.ACROSS_X_ATOMS,
    }
    with pytest.raises(ValueError, match="Unknown CGA operand"):
        tma_mcast_direction("ab")


def test_mixed_cga_config_and_cli_mapping():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    args = build_arg_parser().parse_args(
        [
            "--umma-cta-count",
            "2",
            "--cluster-dim-x",
            "4",
            "--cluster-dim-y",
            "2",
            "--fallback-cluster-dim-x",
            "2",
            "--fallback-cluster-dim-y",
            "1",
        ]
    )
    overrides = _parse_overrides(args)
    assert overrides["umma_cta_count"] == 2
    assert overrides["cluster_dim_x"] == 4
    assert overrides["cluster_dim_y"] == 2
    assert overrides["fallback_cluster_dim_x"] == 2
    assert overrides["fallback_cluster_dim_y"] == 1

    cfg = _bf16_mixed_config()
    assert cfg.preferred_cluster_shape == (4, 2, 1)
    assert cfg.fallback_cluster_shape == (2, 1, 1)
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        has_multiple_umma_atoms,
        is_flexible_cga,
    )

    assert is_flexible_cga(cfg)
    assert has_multiple_umma_atoms(cfg)
    for removed_property in (
        "preferred_cluster_dim_x",
        "physical_cluster_size",
        "has_wider_cluster",
        "has_flexible_cluster",
        "uses_runtime_cga_topology",
        "uses_mixed_cga_multicast_a",
        "uses_mixed_cga_multicast_b",
        "uses_mixed_cga_multicast_sfa",
        "uses_mixed_cga_multicast_sfb",
    ):
        assert not hasattr(cfg, removed_property)


@pytest.mark.parametrize(
    ("cluster_dim_x", "cluster_dim_y", "fallback_x", "expected_a", "expected_b"),
    (
        (0, 1, 0, False, False),  # fixed legacy 2x1
        (4, 1, 0, False, True),  # fixed 4x1
        (2, 2, 0, True, False),  # fixed 2x2
        (4, 2, 0, True, True),  # fixed 4x2
        (4, 1, 2, False, True),  # flexible 4x1 -> 2x1
        (2, 2, 2, True, False),  # flexible 2x2 -> 2x1
        (4, 2, 2, True, True),  # flexible 4x2 -> 2x1
    ),
)
def test_extended_multicast_is_shape_and_operand_specific(
    cluster_dim_x, cluster_dim_y, fallback_x, expected_a, expected_b
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        is_flexible_cga,
        needs_extended_tma_multicast,
    )

    cfg = _bf16_mixed_config(
        cluster_dim_x=cluster_dim_x,
        cluster_dim_y=cluster_dim_y,
        fallback_cluster_dim_x=fallback_x,
        fallback_cluster_dim_y=1,
    )
    assert is_flexible_cga(cfg) is (fallback_x > 0)
    assert needs_extended_tma_multicast(cfg, "a") is expected_a
    assert needs_extended_tma_multicast(cfg, "b") is expected_b
    assert not needs_extended_tma_multicast(cfg, "sfa")
    assert not needs_extended_tma_multicast(cfg, "sfb")


def test_local_a_payload_keeps_multi_atom_tma_umma_barrier_routing(monkeypatch):
    """A 4x1 CGA has local A payloads but two independent UMMA atoms."""
    from cutlass.experimental.task_scheduling.enums import SignalingThreads
    from cutlass.experimental.task_scheduling.resources import MemoryResource
    from cutlass.pipeline.sm100 import PipelineTmaUmma

    from flashinfer.prims_ts.batched_gemm import smem_ab_resources
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        has_multiple_umma_atoms,
        needs_extended_tma_multicast,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _make_pipeline_configs,
    )

    cfg = _bf16_mixed_config(
        cluster_dim_x=4,
        cluster_dim_y=1,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
    )
    assert has_multiple_umma_atoms(cfg)
    assert not needs_extended_tma_multicast(cfg, "a")

    signaling = _make_pipeline_configs(cfg)["smem_a"].producer_signaling_threads
    assert signaling == (SignalingThreads.CtaLeader | SignalingThreads.TaskWarpLeader)

    # Isolate resource construction from device code: a multi-atom A resource
    # must wrap an actual TMA-to-UMMA base pipeline even though its payload is
    # local. This is the host-side gate whose regression stranded ranks 2/3 on
    # SmemA.empty[0].
    base = object.__new__(PipelineTmaUmma)
    wrapped = object()
    calls = []
    monkeypatch.setattr(
        MemoryResource,
        "create_pipeline",
        lambda _self, _pipeline_config: base,
    )
    monkeypatch.setattr(
        smem_ab_resources,
        "override_tma_umma_pipeline",
        lambda actual_cfg, actual_base, operand: (
            calls.append((actual_cfg, actual_base, operand)),
            wrapped,
        )[1],
    )
    resource = object.__new__(smem_ab_resources.SmemAResource)
    object.__setattr__(resource, "cfg", cfg)
    assert resource.create_pipeline(None) is wrapped
    assert calls == [(cfg, base, "a")]


def test_fixed_legacy_and_wider_task_types_are_narrowly_gated():
    from cutlass.experimental.task_scheduling.task import Task

    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import CgaPolicy
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import CgaTask

    legacy_tasks, *_ = _build_schedule_validate(
        _bf16_mixed_config(
            cluster_dim_x=0,
            cluster_dim_y=1,
            fallback_cluster_dim_x=0,
            use_work_throttle=0,
        ),
        num_k_tiles=1,
    )
    legacy_by_name = {task.name: task for task in legacy_tasks}
    assert type(legacy_by_name["LoadATask"]) is Task
    assert type(legacy_by_name["LoadBTask"]) is Task
    assert isinstance(legacy_by_name["MmaTask0"], CgaTask)
    assert legacy_by_name["MmaTask0"].cga_policy == int(CgaPolicy.ALL)
    assert legacy_by_name["MmaTask0"].run_only_on_cta_id == 0

    fixed_4x1_tasks, *_ = _build_schedule_validate(
        _bf16_mixed_config(
            cluster_dim_x=4,
            cluster_dim_y=1,
            fallback_cluster_dim_x=0,
            use_work_throttle=0,
        ),
        num_k_tiles=1,
    )
    fixed_4x1_by_name = {task.name: task for task in fixed_4x1_tasks}
    assert type(fixed_4x1_by_name["LoadATask"]) is Task
    assert isinstance(fixed_4x1_by_name["LoadBTask"], CgaTask)
    assert fixed_4x1_by_name["LoadBTask"].cga_policy == int(CgaPolicy.MCAST_B)
    assert isinstance(fixed_4x1_by_name["MmaTask0"], CgaTask)
    assert fixed_4x1_by_name["MmaTask0"].cga_policy == int(CgaPolicy.ATOM_LEADER)

    fixed_2x2_tasks, *_ = _build_schedule_validate(
        _bf16_mixed_config(
            cluster_dim_x=2,
            cluster_dim_y=2,
            fallback_cluster_dim_x=0,
            use_work_throttle=0,
        ),
        num_k_tiles=1,
    )
    fixed_2x2_by_name = {task.name: task for task in fixed_2x2_tasks}
    assert isinstance(fixed_2x2_by_name["LoadATask"], CgaTask)
    assert fixed_2x2_by_name["LoadATask"].cga_policy == int(CgaPolicy.MCAST_A)
    assert type(fixed_2x2_by_name["LoadBTask"]) is Task


def test_class_a_rank_gated_tasks_redistribute_registers_on_peer_cta():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import CgaPolicy
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _verify_rank_gated_register_donation,
    )
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import CgaTask

    tasks, *_ = _build_schedule_validate(_class_a_hang_config(), num_k_tiles=4)
    tasks_by_name = {task.name: task for task in tasks}

    expected = {
        "CopySfATask": (18, 1, 32),
        "MmaTask0": (19, 1, 32),
        "WorkScheduleTask": (20, 1, 48),
    }
    for name, (warp_idx, num_warps, num_registers) in expected.items():
        task = tasks_by_name[name]
        assert isinstance(task, CgaTask)
        assert task.cga_policy == int(CgaPolicy.ALL)
        assert task.run_only_on_cta_id == 0
        assert (task.warp_idx, task.num_warps, task.num_registers) == (
            warp_idx,
            num_warps,
            num_registers,
        )
    _verify_rank_gated_register_donation(tasks)


def test_rank_gated_register_verifier_rejects_stock_task():
    from types import SimpleNamespace

    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _verify_rank_gated_register_donation,
    )

    stock_rank_gated_task = SimpleNamespace(
        name="UnsafeTask",
        run_only_on_cta_id=0,
        num_registers=32,
    )

    with pytest.raises(ValueError, match="idle peer-CTA.*UnsafeTask"):
        _verify_rank_gated_register_donation([stock_rank_gated_task])


@pytest.mark.parametrize(
    ("overrides", "match"),
    (
        ({"cluster_dim_x": 3}, "must each be 1, 2, or 4"),
        (
            {"umma_cta_count": 1, "cluster_dim_x": 2},
            "require umma_cta_count=2",
        ),
        (
            {"fallback_cluster_dim_x": 4, "fallback_cluster_dim_y": 2},
            "must differ|must divide",
        ),
    ),
)
def test_mixed_cga_validation_rejects_invalid_shapes(overrides, match):
    with pytest.raises(ValueError, match=match):
        _bf16_mixed_config(**overrides)


def test_mixed_cga_deepseek_fp8_is_explicitly_unsupported():
    """DeepSeek FP8 cannot use the umma_cta_count=2 required by mixed CGA."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_n=16,
        tile_k=128,
        mma_m=256,
        mma_n=16,
        mma_k=32,
        epi_tile_n=16,
        umma_cta_count=2,
        cluster_dim_x=4,
        cluster_dim_y=2,
        fallback_cluster_dim_x=2,
        fallback_cluster_dim_y=1,
        use_deepseek_fp8=1,
    )
    with pytest.raises(ValueError, match="incompatible with cluster m > 1"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("preferred_shape", "fallback_shape"),
    [
        ((4, 1, 1), (2, 1, 1)),
        ((4, 2, 1), (2, 1, 1)),
        ((2, 2, 1), (2, 1, 1)),
        ((4, 4, 1), (2, 2, 1)),
    ],
)
def test_mixed_cga_multiwave_counts_both_physical_shapes(
    preferred_shape, fallback_shape
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _mixed_cga_wave_counts,
    )

    # A 148-CTA resident grid has 74 logical 2-CTA UMMA-atom slots for
    # fallback 2x1.  Exactly 74 logical CGAs is one wave; 75 requires every
    # supported preferred/fallback pair to enter a second wave.
    one_wave = _mixed_cga_wave_counts(
        logical_work_cgas=74,
        physical_grid_ctas=148,
        umma_cta_count=2,
        preferred_shape=preferred_shape,
        fallback_shape=fallback_shape,
    )
    multiwave = _mixed_cga_wave_counts(
        logical_work_cgas=75,
        physical_grid_ctas=148,
        umma_cta_count=2,
        preferred_shape=preferred_shape,
        fallback_shape=fallback_shape,
    )

    assert any(work <= wave for work, wave in one_wave)
    assert all(work > wave for work, wave in multiwave)


def test_clc_count_uses_runtime_cga_shape():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        host_clc_consumer_threads,
    )

    cfg = _bf16_mixed_config()

    def expected(dim_x, dim_y):
        all_ctas = dim_x * dim_y
        load_a_ctas = dim_x if dim_y > 1 else all_ctas
        load_b_ctas = min(dim_x, 2) * dim_y if dim_x > 2 else all_ctas
        atom_leaders = (dim_x // 2) * dim_y
        warps = cfg.warp_layout.num_load_a_warps * load_a_ctas
        warps += cfg.warp_layout.num_load_b_warps * load_b_ctas
        warps += cfg.warp_layout.num_mma_warps * atom_leaders
        warps += cfg.warp_layout.num_workid_warps
        warps += all_ctas * (
            cfg.warp_layout.num_epilogue_warps
            + cfg.warp_layout.num_padding_warps
            + cfg.warp_layout.num_sync_warps
            + cfg.warp_layout.num_cast_a_warps
        )
        return warps * 32

    assert host_clc_consumer_threads(cfg, 4, 2) == expected(4, 2)
    assert host_clc_consumer_threads(cfg, 2, 1) == expected(2, 1)


def test_exact_fc2_barrier_and_task_census():
    from cutlass.experimental.task_scheduling.task import Task

    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import (
        CgaPolicy,
        host_clc_consumer_threads,
        needs_extended_tma_multicast,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
        _make_pipeline_configs,
    )

    cfg = _fp4_fc2_mixed_config()
    assert needs_extended_tma_multicast(cfg, "a")
    assert needs_extended_tma_multicast(cfg, "b")
    assert needs_extended_tma_multicast(cfg, "sfa")
    assert not needs_extended_tma_multicast(cfg, "sfb")
    assert host_clc_consumer_threads(cfg, 4, 2) == 3232
    assert host_clc_consumer_threads(cfg, 2, 1) == 928

    pcfgs = _make_pipeline_configs(cfg)
    expected = {
        # resource: stages, bytes, producer arrivals, consumer arrivals
        "smem_a": (5, 65536, 1, 1),
        "smem_b": (5, 4096, 1, 1),
        "smem_sfa": (5, 8192, 1, 1),
        # Mainline expresses the same four-warp consumer as 128 threads.
        "smem_sfb": (5, 512, 1, 128),
        "tmem_sfa": (5, 0, 1, 1),
        "tmem_sfb": (5, 0, 256, 1),
        "tmem_c": (2, 0, 1, 256),
        "work_throttle": (3, 0, 32, 32),
    }
    for name, (stages, num_bytes, producer_count, consumer_count) in expected.items():
        pcfg = pcfgs[name]
        assert pcfg.num_stages == stages
        assert pcfg.num_bytes == num_bytes
        assert pcfg.producer_group.size == producer_count
        assert pcfg.consumer_group.size == consumer_count

    # The static 2x1 WorkQueue count is verifier-only. Device construction
    # replaces it with runtime 3232/928, asserted above.
    assert pcfgs["workid"].num_stages == 3
    assert pcfgs["workid"].num_bytes == 16
    assert pcfgs["workid"].consumer_group.size == 992

    tasks, *_ = _build_schedule_validate(cfg, num_k_tiles=1)
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import CgaTask

    tasks_by_name = {task.name: task for task in tasks}
    expected_tasks = {
        "LoadATask": (1, CgaPolicy.MCAST_A),
        "LoadBTask": (1, CgaPolicy.MCAST_B),
        "LoadSfATask": (1, CgaPolicy.MCAST_A),
        "LoadSfBTask": (1, CgaPolicy.ALL),
        "CopySfATask": (1, CgaPolicy.ATOM_LEADER),
        "CopySfBTask": (4, CgaPolicy.ALL),
        "MmaTask0": (1, CgaPolicy.ATOM_LEADER),
    }
    for name, (num_warps, policy) in expected_tasks.items():
        task = tasks_by_name[name]
        assert task.num_warps == num_warps
        if policy == CgaPolicy.ALL:
            assert type(task) is Task
        else:
            assert isinstance(task, CgaTask)
            assert task.cga_policy == int(policy)

    assert tasks_by_name["EpilogueTask0"].num_warps == 4
    assert tasks_by_name["PaddingTask"].num_warps == 1
    assert tasks_by_name["WorkScheduleTask"].num_warps == 1
    assert tasks_by_name["WorkScheduleTask"].run_only_on_cta_id == 0


@pytest.mark.parametrize(
    ("swap_ab", "cluster_dim_x", "cluster_dim_y", "expected"),
    (
        # Batch-M token tiles span physical CGA X: ceil(65 / (128 * 4)) * 4.
        (0, 4, 2, 4),
        # Batch-N token tiles span physical CGA Y: ceil(65 / (16 * 2)) * 2.
        (1, 4, 2, 6),
    ),
)
def test_early_exit_token_ctas_include_physical_cga_dimension(
    swap_ab, cluster_dim_x, cluster_dim_y, expected
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        resolve_early_exit_max_token_ctas,
    )

    cfg = _bf16_mixed_config(
        batch_mode=int(BatchMode.BATCH_N if swap_ab else BatchMode.BATCH_M),
        transpose_mma_output=swap_ab,
        cluster_dim_x=cluster_dim_x,
        cluster_dim_y=cluster_dim_y,
        use_early_exit=1,
    )
    overrides = {
        field.name: getattr(cfg, field.name)
        for field in __import__("dataclasses").fields(cfg)
        if field.init
    }
    assert (
        resolve_early_exit_max_token_ctas(
            num_tokens=65,
            num_experts=1,
            top_k=1,
            cfg_overrides=overrides,
        )
        == expected
    )


def test_validation_schedule_uses_active_shape_task_policies():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_cga import CgaPolicy
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )
    from flashinfer.prims_ts.batched_gemm.mixed_cga_ts import CgaTask

    tasks, *_ = _build_schedule_validate(_bf16_mixed_config(), num_k_tiles=1)
    tasks_by_name = {task.name: task for task in tasks}

    assert isinstance(tasks_by_name["LoadATask"], CgaTask)
    assert tasks_by_name["LoadATask"].cga_policy == int(CgaPolicy.MCAST_A)
    assert tasks_by_name["LoadBTask"].cga_policy == int(CgaPolicy.MCAST_B)
    assert tasks_by_name["MmaTask0"].cga_policy == int(CgaPolicy.ATOM_LEADER)
    assert tasks_by_name["WorkScheduleTask"].run_only_on_cta_id == 0
