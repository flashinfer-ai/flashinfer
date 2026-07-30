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

"""Schedule-verification regressions for BatchedGemm TS schedules."""

import sys

import pytest


@pytest.mark.timeout(240)
def test_schedule_checker_reports_no_persistent_c_scratch_ab_alias_race():
    """Persistent multi-stage work IDs keep C scratch separate from A/B.

    This uses a reduced BF16 K=tileK schedule to keep the checker tractable:
    five tasks and two persistent work-tile iterations. The production config
    disables C-scratch/A-B aliasing for multi-stage work-id scheduling, and the
    checker should not report the aliasing race in that schedule.
    """
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )
    from cutlass.experimental.task_scheduling.exhaustive_checker import (
        build_alias_info,
        check_all_interleavings,
    )
    from cutlass.experimental.task_scheduling.resources import WorkQueue

    cfg = make_config(
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=3,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_tma_store=1,
        use_early_exit=1,
    )
    assert not cfg.aliases_c_scratch_with_ab

    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    assert len(tasks) == 5

    resources = []
    seen = set()
    for task in tasks:
        for resource in task.src_resources + task.dst_resources:
            if id(resource) not in seen:
                seen.add(id(resource))
                resources.append(resource)

    alias_map, prod_map, cons_map, overlap_descs = build_alias_info(resources)
    resource_by_id = {id(resource): resource.name for resource in resources}
    assert not any(
        resource_by_id.get(resource_id) == "SmemA"
        and any(resource_by_id.get(alias_id) == "GmemC" for alias_id in aliases)
        for resource_id, aliases in alias_map.items()
    )

    num_tiles = (
        2 if any(isinstance(resource, WorkQueue) for resource in resources) else 1
    )
    result = check_all_interleavings(
        tasks,
        alias_map=alias_map,
        prod_alias_map=prod_map,
        cons_alias_map=cons_map,
        overlap_descs=overlap_descs,
        num_tiles=num_tiles,
        max_states=1_000_000,
        verbose=False,
    )

    assert result.is_safe
    assert not any(
        race.writer_task == "LoadATask"
        and race.writer_resource == "SmemA"
        and race.victim_task == "EpilogueTask0"
        and race.victim_resource == "GmemC"
        for race in result.race_states
    )


def test_validation_accepts_mma_unroll():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_unroll_loop_2x_for_mma=1,
    )
    validate_config(cfg)


@pytest.mark.parametrize(
    ("dtype_c_name", "match"),
    (
        ("FP32", "Unsupported dtype_c output store"),
        ("E4M3", "dtype_c=e4m3 plain FP8 output requires dtype_a=dtype_b=e4m3"),
    ),
)
def test_validation_rejects_unsupported_plain_output_store_dtype(dtype_c_name, match):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(getattr(DType, dtype_c_name)),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    with pytest.raises(ValueError, match=match):
        validate_config(cfg)


def test_runner_uses_fp16_for_fp16_plain_output_store_dtype():
    import cutlass
    import torch

    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.FP16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )

    validate_config(cfg)
    assert batched_gemm_run._plain_output_torch_dtype(cfg) == torch.float16
    assert batched_gemm_run._plain_output_cutlass_dtype(cfg) == cutlass.Float16


@pytest.mark.parametrize(
    "act_kind_name, expected",
    (
        ("NONE", False),
        ("SWIGLU", True),
        ("GEGLU", True),
        ("RELU2", False),
        ("SILU", True),
    ),
)
def test_runner_gated_activation_classification_matches_kernel(act_kind_name, expected):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import ActKind

    assert (
        batched_gemm_run._is_gated_act_kind(int(getattr(ActKind, act_kind_name)))
        is expected
    )


@pytest.mark.parametrize(
    "batch_mode_name, act_kind_name, expected_shape",
    (
        ("BATCH_N", "NONE", (128, 64)),
        ("BATCH_M", "NONE", (128, 64)),
        ("BATCH_N", "SWIGLU", (64, 64)),
        ("BATCH_M", "SWIGLU", (128, 32)),
    ),
)
def test_runner_logical_output_shape_for_gated_activation_abi(
    batch_mode_name, act_kind_name, expected_shape
):
    from flashinfer.prims_ts.batched_gemm import batched_gemm_run
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        make_config,
    )

    batch_mode = getattr(BatchMode, batch_mode_name)
    cfg = make_config(
        batch_mode=int(batch_mode),
        transpose_mma_output=int(batch_mode == BatchMode.BATCH_N),
        act_kind=int(getattr(ActKind, act_kind_name)),
    )
    is_gated = batched_gemm_run._is_gated_act_kind(cfg.act_kind)
    logical_shape = batched_gemm_run._logical_output_shape(cfg, 128, 64, is_gated)

    assert logical_shape == expected_shape


@pytest.mark.parametrize(
    "stage_name",
    (
        "num_stages_a",
        "num_stages_b",
        "num_stages_tmem_acc",
    ),
)
def test_validation_rejects_zero_required_pipeline_stage_counts(stage_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    with pytest.raises(ValueError, match=f"{stage_name} must be positive"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("num_stages_a", "num_stages_b"),
    ((3, 4), (4, 3)),
)
def test_validation_rejects_unequal_2cta_gather_pipeline_stages(
    num_stages_a, num_stages_b
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        cluster_m=2,
        route_act=2,
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=16,
        tile_n=64,
        mma_n=64,
        mma_m=256,
        num_stages_a=num_stages_a,
        num_stages_b=num_stages_b,
    )

    with pytest.raises(
        ValueError, match="2-CTA gather requires num_stages_a == num_stages_b"
    ):
        validate_config(cfg)


@pytest.mark.parametrize(
    "stage_name",
    (
        "num_stages_c_smem",
        "num_stages_smem_sfa",
        "num_stages_smem_sfb",
        "num_stages_tmem_sfa",
        "num_stages_tmem_sfb",
        "num_stages_workid",
    ),
)
def test_validation_accepts_zero_inactive_pipeline_stage_counts(stage_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
    )
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    validate_config(cfg)


@pytest.mark.parametrize(
    ("stage_name", "overrides"),
    (
        ("num_stages_c_smem", {"use_tma_store": 1}),
        ("num_stages_smem_sfa", {}),
        ("num_stages_smem_sfb", {}),
        ("num_stages_tmem_sfa", {}),
        ("num_stages_tmem_sfb", {}),
    ),
)
def test_validation_rejects_zero_active_sf_and_store_stage_counts(
    stage_name, overrides
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        SfLayout,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_k=512,
        mma_k=64,
        tile_n=8,
        epi_tile_n=8,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
    )
    kwargs.update(overrides)
    kwargs[stage_name] = 0
    cfg = make_config(**kwargs)

    with pytest.raises(ValueError, match=f"{stage_name} must be positive"):
        validate_config(cfg)


def test_validation_rejects_zero_persistent_workid_stages():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_workid=0,
    )

    with pytest.raises(ValueError, match="num_stages_workid must be positive"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("epi_tile_n", "expected"),
    (
        (0, "positive"),
        (12, "multiple of 8"),
        (24, "divide tile_n"),
    ),
)
def test_validation_rejects_invalid_epi_tile_n(epi_tile_n, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_n=64,
        epi_tile_n=epi_tile_n,
    )
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"mma_m": 96}, "mma_m must be 64/128/256"),
        ({"mma_n": 12}, "mma_n must be 8/16/32/64/128/256"),
    ),
)
def test_validation_rejects_invalid_mma_shape(overrides, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_m=128,
        mma_n=64,
        epi_tile_n=64,
    )
    kwargs.update(overrides)
    cfg = make_config(**kwargs)
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    (
        ({"tile_n": 16, "mma_n": 32}, "tile_n must be a multiple of mma_n"),
        ({"mma_m": 256}, "cluster-wide tile_m must be a multiple of mma_m"),
        (
            {"cluster_m": 2, "mma_n": 8},
            "cluster_m=2 requires mma_n >= 16",
        ),
    ),
)
def test_validation_rejects_incompatible_tile_and_mma_shapes(overrides, expected):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    kwargs = dict(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=64,
        mma_m=128,
        mma_n=64,
        epi_tile_n=16,
    )
    kwargs.update(overrides)
    cfg = make_config(**kwargs)
    with pytest.raises(ValueError, match=expected):
        validate_config(cfg)


def test_validation_rejects_nvfp4_mma_m64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_k=256,
        mma_k=64,
        tile_n=16,
        mma_m=64,
        mma_n=16,
        epi_tile_n=16,
    )
    with pytest.raises(ValueError, match="NVFP4 MMA does not support mma_m=64"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("batch_mode_name", "transpose_mma_output"),
    (("BATCH_N", 0), ("BATCH_M", 1)),
)
def test_validation_rejects_transpose_incompatible_with_batch_mode(
    batch_mode_name, transpose_mma_output
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(getattr(BatchMode, batch_mode_name)),
        transpose_mma_output=transpose_mma_output,
    )
    with pytest.raises(ValueError, match="transpose_mma_output must match.*swap_ab"):
        validate_config(cfg)


def test_validation_rejects_non_binary_transpose_mma_output():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        make_config,
        validate_config,
    )

    cfg = make_config(transpose_mma_output=2)
    with pytest.raises(ValueError, match="transpose_mma_output must be 0 or 1"):
        validate_config(cfg)


def test_validation_rejects_tile_k_not_multiple_of_mma_k():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=20,
        mma_k=16,
        tile_n=16,
    )
    with pytest.raises(ValueError, match="tile_k must be a multiple of mma_k"):
        validate_config(cfg)


def test_validation_rejects_bf16_kbox_tma_a_tile_k_not_multiple_of_64():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=16,
        mma_k=16,
        tile_n=16,
    )
    assert cfg.use_bf16_kbox_tma_a
    with pytest.raises(ValueError, match="BF16 k-box TMA A path"):
        validate_config(cfg)


def test_validation_rejects_casta_tile_k_not_256():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=16,
        tile_n=16,
    )
    assert cfg.has_cast_a
    with pytest.raises(ValueError, match="CastA MXFP4 input requires tile_k=256"):
        validate_config(cfg)


@pytest.mark.parametrize("dtype_c_name", ("BF16", "FP16"))
def test_validation_accepts_casta_plain_output_store(dtype_c_name):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(getattr(DType, dtype_c_name)),
        tile_k=256,
        mma_k=16,
        tile_n=16,
        num_stages_a=1,
        num_stages_tmem_acc=1,
    )

    assert cfg.has_cast_a
    validate_config(cfg)


def test_validation_rejects_casta_output_quantization():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.MXE2M1),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.E2M1),
        tile_k=256,
        mma_k=16,
        tile_n=16,
    )

    assert cfg.has_cast_a
    with pytest.raises(
        ValueError, match="CastA MXFP4 input supports dtype_c=bf16/fp16"
    ):
        validate_config(cfg)


def test_validation_accepts_silu_activation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        act_kind=int(ActKind.SILU),
    )
    validate_config(cfg)


def test_clustered_swap_ab_ldgsts_splits_b_smem_across_ctas():
    """Each CTA stages its routed B rows and uses the matching K-group stride."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
    )

    cfg = make_config(
        cluster_m=2,
        route_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        tile_n=128,
        mma_n=128,
        epi_tile_n=32,
        tile_k=512,
        mma_k=64,
    )

    assert cfg.is_swap_ab
    assert cfg.has_gather
    assert cfg.split_b_across_ctas
    assert cfg.num_bytes_b_per_stage == 32_768
    assert cfg.num_bytes_b_smem_per_stage == 16_384


def test_r128c4_sfb_s2t_descriptors_advance_in_k_group_order():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import SfLayout
    from flashinfer.prims_ts.batched_gemm.tmem_c_resources import (
        _sfb_s2t_desc_increment,
    )

    assert [
        _sfb_s2t_desc_increment(int(SfLayout.R128c4), copy_idx) for copy_idx in range(8)
    ] == [0, 32, 64, 96, 128, 160, 192, 224]


def test_ldgsts_routed_sf_does_not_use_routed_tma_descriptors():
    """LDGSTS-routed scale factors must not build compact TMA TensorMaps."""
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        uses_routed_sfa_tma_desc,
        uses_routed_sfb_tma_desc,
    )

    common = dict(
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        dtype_a=int(DType.MXE4M3),
        dtype_b=int(DType.MXE4M3),
        dtype_c=int(DType.MXE4M3),
        sf_block_size_a=32,
        sf_block_size_b=32,
        sf_block_size_c=32,
        tile_k=128,
        mma_k=32,
    )

    swap_ab_cfg = make_config(**common)
    assert swap_ab_cfg.has_routed_sfs
    assert swap_ab_cfg.uses_ldgsts_routed_sfs
    assert not uses_routed_sfa_tma_desc(swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(swap_ab_cfg)

    non_swap_ab_cfg = make_config(
        **common,
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
    )
    assert non_swap_ab_cfg.has_routed_sfs
    assert non_swap_ab_cfg.uses_ldgsts_routed_sfs
    assert not uses_routed_sfa_tma_desc(non_swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(non_swap_ab_cfg)

    tma_swap_ab_cfg = make_config(
        **{**common, "route_sfs_act": int(RouteImpl.TMA), "tile_k": 512},
    )
    assert uses_routed_sfb_tma_desc(tma_swap_ab_cfg)
    assert not uses_routed_sfa_tma_desc(tma_swap_ab_cfg)

    tma_non_swap_ab_cfg = make_config(
        **{**common, "route_sfs_act": int(RouteImpl.TMA), "tile_k": 512},
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
    )
    assert uses_routed_sfa_tma_desc(tma_non_swap_ab_cfg)
    assert not uses_routed_sfb_tma_desc(tma_non_swap_ab_cfg)


@pytest.mark.timeout(180)
def test_deepseek_fp8_uses_two_epilogue_warpgroups():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_m=128,
        tile_n=8,
        tile_k=128,
        mma_m=64,
        mma_n=8,
        mma_k=32,
        epi_tile_n=8,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
    )
    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    by_name = {task.name: task for task in tasks}

    assert by_name["EpilogueTask0DsFp8"].warp_idx == 0
    assert by_name["EpilogueTask0DsFp8"].num_warps == 8
    assert by_name["MmaTask0"].warp_idx == 8
    assert by_name["LoadATask"].warp_idx == 9
    assert by_name["LoadBTask"].warp_idx == 10
    assert by_name["LoadSfAbTask"].warp_idx == 11
    assert cfg.threads_per_cta == 384


@pytest.mark.timeout(180)
def test_persistent_deepseek_fp8_schedule_builds():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    task_manager = build_batched_gemm_task_manager(
        num_experts=2,
        num_tokens=256,
        top_k=1,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_m=128,
        tile_n=8,
        tile_k=128,
        mma_m=64,
        mma_n=8,
        mma_k=32,
        epi_tile_n=8,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
    )

    assert {task.name for task in task_manager.tasks} >= {
        "LoadSfAbTask",
        "EpilogueTask0DsFp8",
        "WorkScheduleTask",
    }


@pytest.mark.timeout(180)
@pytest.mark.parametrize(
    ("tile_n", "num_load_b_warps", "load_sfab_warp_idx", "padding_warp_idx"),
    (
        (32, 2, 12, 13),
        (128, 4, 14, 15),
    ),
)
def test_deepseek_fp8_tma_route_uses_generated_load_b_warp_count(
    tile_n,
    num_load_b_warps,
    load_sfab_warp_idx,
    padding_warp_idx,
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        route_act=int(RouteImpl.TMA),
        tile_m=128,
        tile_n=tile_n,
        tile_k=128,
        mma_m=64,
        mma_n=tile_n,
        mma_k=32,
        epi_tile_n=tile_n,
        num_stages_a=6,
        num_stages_b=6,
        num_stages_smem_sfa=6,
        num_stages_smem_sfb=6,
        num_stages_tmem_acc=1,
        use_deepseek_fp8=1,
        num_load_b_warps=num_load_b_warps,
        num_load_sfab_warps=1,
    )
    tasks, _dep_graph, _smem_allocator, _tmem_allocator = _build_schedule_validate(
        cfg,
        num_k_tiles=1,
    )
    by_name = {task.name: task for task in tasks}

    assert by_name["EpilogueTask0DsFp8"].warp_idx == 0
    assert by_name["EpilogueTask0DsFp8"].num_warps == 8
    assert by_name["MmaTask0"].warp_idx == 8
    assert by_name["LoadATask"].warp_idx == 9
    assert by_name["LoadBTask"].warp_idx == 10
    assert by_name["LoadBTask"].num_warps == num_load_b_warps
    assert by_name["LoadSfAbTask"].warp_idx == load_sfab_warp_idx
    assert by_name["PaddingTask"].warp_idx == padding_warp_idx
    assert cfg.threads_per_cta == 512

def test_validation_accepts_clc_fast_drain_for_persistent_early_exit():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        use_early_exit=1,
        use_clc_fast_drain=1,
    )
    validate_config(cfg)


@pytest.mark.parametrize(
    ("tile_scheduler", "use_early_exit"),
    ((0, 1), (1, 0)),
)
def test_validation_rejects_clc_fast_drain_without_persistent_early_exit(
    tile_scheduler, use_early_exit
):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        TileScheduler,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        tile_scheduler=tile_scheduler,
        use_early_exit=use_early_exit,
        use_clc_fast_drain=1,
    )
    with pytest.raises(ValueError, match="requires persistent scheduling"):
        validate_config(cfg)


@pytest.mark.parametrize(
    "removed_option",
    (
        "skip_tmem_dealloc",
        "skip_tmem_dealloc_barrier",
        "skip_tmem_dealloc_instruction",
        "use_delayed_ldgsts_sfb_commit",
        "use_predicated_ldgsts_sfb",
        "skip_ldgsts_sfb_payload",
        "skip_copy_sfb_payload",
        "debug_print",
        "num_experts",
        "top_k",
        "static_num_k_tiles",
        "max_num_ctas_in_token_dim",
        "early_exit_max_token_ctas",
        "acc_dtype_bytes",
    ),
)
def test_config_rejects_removed_non_config_options(removed_option):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
    )

    with pytest.raises(TypeError, match=removed_option):
        make_config(
            dtype_a=int(DType.BF16),
            dtype_b=int(DType.BF16),
            dtype_c=int(DType.BF16),
            tile_k=64,
            mma_k=16,
            tile_n=16,
            **{removed_option: 1},
        )


def test_validation_rejects_ldg_plus_sts_routes():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        RouteImpl,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        route_act=int(RouteImpl.LDG_PLUS_STS),
    )
    with pytest.raises(ValueError, match="LDG_PLUS_STS"):
        validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        route_sfs_act=int(RouteImpl.LDG_PLUS_STS),
    )
    with pytest.raises(ValueError, match="LDG_PLUS_STS"):
        validate_config(cfg)


def test_validation_accepts_minimal_plain_fp8_per_token_sfb():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_a=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        use_per_token_sf_a=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.BF16),
    )
    validate_config(cfg)


def test_validation_accepts_nvfp4_per_token_sfb_only():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        BatchMode,
        DType,
        RouteImpl,
        SfLayout,
        make_config,
        validate_config,
    )

    nvfp4_kwargs = dict(
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        tile_m=128,
        tile_n=8,
        tile_k=512,
        epi_tile_n=8,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        use_global_scales=1,
    )

    cfg = make_config(
        **nvfp4_kwargs,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    assert cfg.has_per_token_sf_b
    validate_config(cfg)

    cfg = make_config(
        **nvfp4_kwargs,
        bias_type=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        **{
            **nvfp4_kwargs,
            "dtype_c": int(DType.E2M1),
            "route_act": int(RouteImpl.TMA),
            "act_kind": int(ActKind.SWIGLU),
        },
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    assert cfg.has_per_token_sf_b
    assert cfg.has_epilogue_quant
    validate_config(cfg)

    cfg = make_config(
        **nvfp4_kwargs,
        use_per_token_sf_a=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    with pytest.raises(ValueError, match="Per-token sf_a is only valid"):
        validate_config(cfg)

    no_swap_ab_kwargs = {
        **nvfp4_kwargs,
        "batch_mode": int(BatchMode.BATCH_M),
        "transpose_mma_output": 0,
    }
    cfg = make_config(
        **no_swap_ab_kwargs,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )
    validate_config(cfg)

    cfg = make_config(
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        tile_k=64,
        mma_k=16,
        tile_n=16,
        use_per_token_sf_b=1,
    )
    with pytest.raises(ValueError, match="Per-token sf_b"):
        validate_config(cfg)


@pytest.mark.timeout(180)
def test_nvfp4_per_token_sfb_schedule_validates():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        RouteImpl,
        SfLayout,
    )
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        build_batched_gemm_task_manager,
    )

    build_batched_gemm_task_manager(
        num_experts=64,
        num_tokens=1,
        top_k=8,
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        route_act=int(RouteImpl.TMA),
        act_kind=int(ActKind.SWIGLU),
        tile_m=128,
        tile_n=8,
        tile_k=512,
        epi_tile_n=8,
        mma_m=128,
        mma_n=8,
        mma_k=64,
        sf_layout_a=int(SfLayout.R128c4),
        sf_layout_b=int(SfLayout.LINEAR),
        sf_layout_c=int(SfLayout.R8c4),
        use_global_scales=1,
        use_per_token_sf_b=1,
        per_token_sf_dtype=int(DType.FP32),
    )


def test_fp4_json_variant_forwards_per_token_sfb_without_skip():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        BatchMode,
        BiasType,
        DType,
        make_config,
        validate_config,
    )
    from flashinfer.prims_ts.batched_gemm.tools import bench

    options = {
        "tileN": 8,
        "tileK": 512,
        "tileScheduler": "static",
        "numStagesMma": 1,
        "numStages": 5,
        "epilogueTileN": 8,
        "mmaM": 128,
        "mmaN": 8,
        "sfLayoutA": "128x4",
        "sfLayoutB": "linear",
        "sfLayoutC": "8x4",
        "dtypeC": "e2m1",
        "transpose_mma_output": True,
        "usePerTokenSfB": True,
        "perTokenSfDtype": "fp32",
        "biasType": "none",
    }
    variant = bench._with_ts_skip_reason(
        bench._fp4_json_variant(
            config_index=0,
            config_comment="FC1_LL",
            combo_index=0,
            options=options,
        )
    )

    assert variant.ts_skip_reason is None
    assert variant.kwargs["dtype_a"] == int(DType.E2M1)
    assert variant.kwargs["dtype_b"] == int(DType.E2M1)
    assert variant.kwargs["dtype_c"] == int(DType.E2M1)
    assert variant.kwargs["batch_mode"] == int(BatchMode.BATCH_N)
    assert variant.kwargs["transpose_mma_output"] == 1
    assert variant.kwargs["use_per_token_sf_a"] == 0
    assert variant.kwargs["use_per_token_sf_b"] == 1
    assert variant.kwargs["per_token_sf_dtype"] == int(DType.FP32)
    assert variant.kwargs["bias_type"] == int(BiasType.NONE)

    non_transposed = bench._fp4_json_variant(
        config_index=0,
        config_comment="FC1_LL",
        combo_index=1,
        options={**options, "transpose_mma_output": False},
    )
    assert non_transposed.kwargs["batch_mode"] == int(BatchMode.BATCH_N)
    assert non_transposed.kwargs["transpose_mma_output"] == 0
    with pytest.raises(ValueError, match="must match.*swap_ab"):
        validate_config(make_config(**non_transposed.kwargs))


def test_trtllm_gen_json_transpose_option_is_normalized(tmp_path):
    import json

    from flashinfer.prims_ts.batched_gemm.tools import bench

    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "templates": {},
                "configs": [
                    {
                        "_comment": "FC1_LL",
                        bench.TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY: False,
                    }
                ],
            }
        )
    )

    [(_config_index, _comment, _combo_index, options)] = bench._expanded_json_options(
        config_path
    )
    assert options["transpose_mma_output"] is False
    assert bench.TRTLLM_GEN_TRANSPOSE_MMA_OUTPUT_KEY not in options


def test_fp4_json_variant_forwards_fp16_dtype_c():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._fp4_json_variant(
        config_index=0,
        config_comment="FC2_LL",
        combo_index=0,
        options={
            "tileN": 8,
            "tileK": 512,
            "tileScheduler": "static",
            "numStagesMma": 1,
            "numStages": 5,
            "epilogueTileN": 8,
            "mmaM": 128,
            "mmaN": 8,
            "dtypeC": "fp16",
        },
    )

    assert variant.kwargs["dtype_c"] == int(DType.FP16)


def test_bf16_json_variant_forwards_fp16_dtype_c():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import DType
    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._bf16_json_variant(
        config_index=0,
        config_comment="FC2_HT",
        combo_index=0,
        options={
            "tileN": 16,
            "tileK": 128,
            "tileScheduler": "static",
            "numStages": 4,
            "epilogueTileN": 16,
            "mmaM": 128,
            "mmaN": 16,
            "dtypeC": "fp16",
        },
    )

    assert variant.kwargs["dtype_a"] == int(DType.BF16)
    assert variant.kwargs["dtype_b"] == int(DType.BF16)
    assert variant.kwargs["dtype_c"] == int(DType.FP16)


def test_generated_command_treats_fp16_dtype_c_as_plain_output():
    from pathlib import Path

    from flashinfer.prims_ts.batched_gemm.tools import bench

    variant = bench._bf16_json_variant(
        config_index=0,
        config_comment="FC2_HT",
        combo_index=0,
        options={
            "tileN": 16,
            "tileK": 128,
            "tileScheduler": "static",
            "numStages": 4,
            "epilogueTileN": 16,
            "mmaM": 128,
            "mmaN": 16,
            "dtypeC": "fp16",
        },
    )

    cmd = bench._trtllm_gen_command(
        binary=Path("batched_gemm"),
        variant=variant,
        num_tokens=128,
        num_experts=1,
        top_k=1,
        warmup_iters=1,
        bench_iters=1,
        num_rotated_buffers=0,
        use_ccache=False,
        use_cuda_graph=False,
    )

    assert "DeepSeekR1_TP1_EP1_MoE_output_tokens128" in cmd
    assert cmd[cmd.index("-dtypeC") + 1] == "fp16"
    assert "-eltwiseActType" not in cmd


def test_runner_cli_accepts_batch_mode(monkeypatch):
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import BatchMode

    import flashinfer.prims_ts.batched_gemm.batched_gemm_run as batched_gemm_run

    captured = {}

    def fake_validate_schedule(**kwargs):
        captured.update(kwargs)
        return True

    monkeypatch.setattr(
        batched_gemm_run,
        "validate_schedule",
        fake_validate_schedule,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "batched_gemm_run.py",
            "--validate-only",
            "--batch-mode",
            "m",
            "--transpose-mma-output",
            "0",
        ],
    )

    batched_gemm_run.main()

    assert captured["batch_mode"] == int(BatchMode.BATCH_M)
    assert captured["transpose_mma_output"] == 0


def test_validation_rejects_deepseek_fp8_fused_gated_activation():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        ActKind,
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=32,
        tile_n=16,
        act_kind=int(ActKind.SWIGLU),
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
    )
    with pytest.raises(ValueError, match="fused gated activation"):
        validate_config(cfg)


def test_validation_rejects_deepseek_fp8_split_tma_c_scratch_stages():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_k=128,
        mma_k=32,
        mma_m=64,
        use_deepseek_fp8=1,
        use_tma_store=1,
        num_load_sfab_warps=1,
        num_stages_c_smem=2,
    )
    with pytest.raises(ValueError, match="num_stages_c_smem=1"):
        validate_config(cfg)


def test_validation_rejects_deepseek_fp8_multi_epilogue_subtiles():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_n=128,
        epi_tile_n=64,
        tile_k=128,
        mma_k=32,
        mma_m=64,
        num_epilogue_warps=8,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
    )
    with pytest.raises(ValueError, match="epi_subtile_cnt=1"):
        validate_config(cfg)


def test_deepseek_fp8_epi_subtile_validation_uses_computed_warp_layout():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        compute_warp_layout,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.BF16),
        tile_n=256,
        epi_tile_n=128,
        tile_k=128,
        mma_k=32,
        mma_m=128,
        mma_n=256,
        use_deepseek_fp8=1,
        num_load_sfab_warps=1,
    )
    compute_warp_layout(cfg)
    assert cfg.num_epilogue_warps == 8
    validate_config(cfg)


def test_validation_rejects_plain_fp8_output_without_activation_side():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
        DType,
        make_config,
        validate_config,
    )

    cfg = make_config(
        dtype_a=int(DType.E4M3),
        dtype_b=int(DType.E4M3),
        dtype_c=int(DType.E4M3),
        tile_k=32,
        mma_k=32,
        tile_n=16,
    )
    with pytest.raises(ValueError, match="Plain FP8 C output"):
        validate_config(cfg)
