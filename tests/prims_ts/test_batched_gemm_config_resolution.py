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

"""Configuration resolution and resource contracts, without GPU execution."""

from dataclasses import FrozenInstanceError, replace

import pytest

from flashinfer.prims_ts.batched_gemm.batched_gemm_config import (
    ActKind,
    BatchMode,
    CtaRasterOrder,
    DType,
    RouteImpl,
    SfLayout,
    TileScheduler,
    compute_launch_grid,
    compute_max_num_ctas_in_token_dim_for_moe,
    compute_warp_layout,
    make_config,
    resolve_problem_config,
    validate_launch_grid,
    validate_config,
)


def _clustered(**overrides):
    values = dict(
        umma_cta_count=2,
        tile_n=128,
        mma_n=128,
        mma_m=256,
        tile_k=256,
        sf_layout_b=int(SfLayout.R128c4),
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=1,
        num_stages_tmem_sfa=2,
        num_stages_tmem_sfb=2,
    )
    values.update(overrides)
    return make_config(**values)


def _bf16_batch_m(**overrides):
    values = dict(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.BF16),
        dtype_b=int(DType.BF16),
        dtype_c=int(DType.BF16),
        umma_cta_count=2,
        mma_m=256,
        mma_n=128,
        mma_k=16,
        tile_n=128,
        tile_k=64,
        epi_tile_n=128,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cta_raster_order=int(CtaRasterOrder.ALONG_N),
        num_stages_tmem_acc=2,
        use_tma_store=1,
        use_shuffled_matrix=1,
    )
    values.update(overrides)
    return make_config(**values)


def _kimi_fc2_token_ctas(*, num_tokens: int, tile_n: int) -> int:
    return compute_max_num_ctas_in_token_dim_for_moe(
        num_tokens=num_tokens,
        num_experts=896,
        top_k=16,
        token_tile_size=tile_n,
        cluster_dim_in_token=1,
    )


def _persistent_grid_config(
    *,
    tile_n: int = 32,
    raster_order: CtaRasterOrder = CtaRasterOrder.ALONG_N,
    umma_cta_count: int = 1,
):
    return make_config(
        tile_n=tile_n,
        mma_n=tile_n,
        mma_m=128 if umma_cta_count == 1 else 256,
        umma_cta_count=umma_cta_count,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        cta_raster_order=int(raster_order),
        use_early_exit=1,
    )


def test_launch_grid_rejects_along_n_cluster_count_overflow():
    token_ctas = _kimi_fc2_token_ctas(num_tokens=3072, tile_n=32)
    cfg = _persistent_grid_config()

    assert token_ctas == 2404
    assert compute_launch_grid(
        cfg,
        problem_m=3584,
        problem_n=3072 * 16,
        early_exit_max_token_ctas=token_ctas,
    ) == (1, 1, 67312)
    with pytest.raises(ValueError, match=r"gridDim\.z=67312.*limit 65535"):
        validate_launch_grid(
            cfg,
            problem_m=3584,
            problem_n=3072 * 16,
            early_exit_max_token_ctas=token_ctas,
        )


def test_launch_grid_accepts_along_m_and_under_limit_along_n():
    over_limit_token_ctas = _kimi_fc2_token_ctas(num_tokens=3072, tile_n=32)
    along_m = _persistent_grid_config(raster_order=CtaRasterOrder.ALONG_M)
    assert compute_launch_grid(
        along_m,
        problem_m=3584,
        problem_n=3072 * 16,
        early_exit_max_token_ctas=over_limit_token_ctas,
    ) == (28, 2404, 1)
    validate_launch_grid(
        along_m,
        problem_m=3584,
        problem_n=3072 * 16,
        early_exit_max_token_ctas=over_limit_token_ctas,
    )

    under_limit_token_ctas = _kimi_fc2_token_ctas(num_tokens=2048, tile_n=32)
    along_n = _persistent_grid_config()
    assert under_limit_token_ctas == 1892
    assert compute_launch_grid(
        along_n,
        problem_m=3584,
        problem_n=2048 * 16,
        early_exit_max_token_ctas=under_limit_token_ctas,
    ) == (1, 1, 52976)
    validate_launch_grid(
        along_n,
        problem_m=3584,
        problem_n=2048 * 16,
        early_exit_max_token_ctas=under_limit_token_ctas,
    )


def test_launch_grid_tile_16_boundary_and_cluster_count():
    tile_16 = _persistent_grid_config(tile_n=16)
    over_limit_token_ctas = _kimi_fc2_token_ctas(num_tokens=1536, tile_n=16)
    assert over_limit_token_ctas == 2376
    with pytest.raises(ValueError, match=r"gridDim\.z=66528.*limit 65535"):
        validate_launch_grid(
            tile_16,
            problem_m=3584,
            problem_n=1536 * 16,
            early_exit_max_token_ctas=over_limit_token_ctas,
        )

    under_limit_token_ctas = _kimi_fc2_token_ctas(num_tokens=1024, tile_n=16)
    assert under_limit_token_ctas == 1864
    validate_launch_grid(
        tile_16,
        problem_m=3584,
        problem_n=1024 * 16,
        early_exit_max_token_ctas=under_limit_token_ctas,
    )

    two_cta = _persistent_grid_config(umma_cta_count=2)
    assert compute_launch_grid(
        two_cta,
        problem_m=3584,
        problem_n=3072 * 16,
        early_exit_max_token_ctas=2404,
    ) == (2, 1, 33656)
    validate_launch_grid(
        two_cta,
        problem_m=3584,
        problem_n=3072 * 16,
        early_exit_max_token_ctas=2404,
    )


def test_static_launch_grid_uses_xy_dimensions():
    cfg = make_config(
        tile_n=32,
        mma_n=32,
        tile_scheduler=int(TileScheduler.STATIC),
        use_early_exit=1,
    )
    assert compute_launch_grid(
        cfg,
        problem_m=3584,
        problem_n=2048 * 16,
        early_exit_max_token_ctas=1892,
    ) == (28, 1892, 1)


def test_resolution_does_not_turn_automatic_counts_into_overrides():
    cfg = make_config(route_act=int(RouteImpl.TMA), tile_n=64, mma_n=64)
    assert cfg.num_load_sfb_warps == 0
    assert compute_warp_layout(cfg).num_load_sfb_warps == 8
    assert cfg.num_load_sfb_warps == 0
    changed = replace(cfg, route_act=int(RouteImpl.NONE))
    fresh = make_config(tile_n=64, mma_n=64)
    assert changed.warp_layout == fresh.warp_layout
    assert fresh.warp_layout.num_load_sfb_warps == 1
    with pytest.raises(FrozenInstanceError):
        cfg.num_load_sfb_warps = 4


@pytest.mark.parametrize("route", [RouteImpl.TMA, RouteImpl.LDGSTS])
@pytest.mark.parametrize("batch", [BatchMode.BATCH_M, BatchMode.BATCH_N])
def test_explicit_operand_warp_count_applies_to_every_transport(route, batch):
    operand = "a" if batch == BatchMode.BATCH_M else "b"
    cfg = make_config(
        batch_mode=int(batch),
        transpose_mma_output=int(batch == BatchMode.BATCH_N),
        route_act=int(route),
        **{f"num_load_{operand}_warps": 8},
    )
    validate_config(cfg)
    assert getattr(cfg.warp_layout, f"num_load_{operand}_warps") == 8


@pytest.mark.parametrize(
    "option",
    [
        "num_gather_warps",
        "gather_warp_idx",
        "gather_regs",
        "num_epilogue_warps",
        "mma_warp_idx",
        "threads_per_cta",
        "num_load_sfab_warps",
        "sf_bits",
        "use_two_tma_load_warps",
        "dtype_acc",
        "do_pdl_wait_for_num_non_exiting_ctas",
        "ldgsts_sfb_producer_commit_prefetch_depth",
    ],
)
def test_removed_or_derived_options_are_not_constructor_inputs(option):
    with pytest.raises(TypeError, match=option):
        make_config(**{option: 1})


@pytest.mark.parametrize("depths", [(2, 3), (3, 2)])
def test_combined_sf_cursor_cannot_exceed_either_operand_allocation(depths):
    cfg = _clustered(num_stages_tmem_sfa=depths[0], num_stages_tmem_sfb=depths[1])
    assert cfg.use_combined_sfab_copy
    with pytest.raises(ValueError, match="equal TMEM SFA/SFB"):
        validate_config(cfg)
    validate_config(_clustered())


@pytest.mark.parametrize("operand", ["a", "b"])
def test_nvfp4_block_length_must_agree_with_descriptors(operand):
    with pytest.raises(TypeError, match="sf_block_size"):
        make_config(**{f"sf_block_size_{operand}": 32})
    cfg = make_config()
    assert cfg.input_sf_block_size_a == cfg.input_sf_block_size_b == 16


def test_overlap_requires_explicit_sf_copy_fusion():
    cfg = _clustered(tile_n=256, mma_n=256, epi_tile_n=64, use_max_tmem_overlap=1)
    with pytest.raises(ValueError, match="requires fuse_sf_copy_to_mma=1"):
        validate_config(cfg)
    validate_config(replace(cfg, fuse_sf_copy_to_mma=1))


@pytest.mark.parametrize(
    ("epi_tile_n", "expected_steps", "expected_tma_passes"),
    ((64, 1, 1), (128, 2, 1), (256, 4, 2)),
)
def test_gated_batch_m_streams_one_epilogue_tile_in_register_steps(
    epi_tile_n, expected_steps, expected_tma_passes
):
    cfg = _bf16_batch_m(
        tile_n=epi_tile_n,
        mma_n=epi_tile_n,
        epi_tile_n=epi_tile_n,
    )
    validate_config(cfg)
    assert cfg.non_swap_epilogue_fragments_per_group == 2
    assert cfg.non_swap_gated_accumulator_cols_per_step == 64
    assert cfg.non_swap_gated_output_cols_per_call == 32
    assert cfg.non_swap_gated_steps_per_epi_tile == expected_steps
    assert cfg.non_swap_epilogue_subtile_cnt == expected_steps
    assert cfg.tma_store_passes == expected_tma_passes


def test_gated_batch_m_rejects_partial_register_step():
    cfg = _bf16_batch_m(
        tile_n=192,
        mma_n=192,
        epi_tile_n=96,
        use_tma_store=0,
        use_shuffled_matrix=0,
    )
    with pytest.raises(ValueError, match="divisible by the accumulator columns"):
        validate_config(cfg)


def test_gated_batch_m_tma_rejects_partial_output_box():
    cfg = _bf16_batch_m(
        tile_n=192,
        mma_n=192,
        epi_tile_n=192,
        use_shuffled_matrix=0,
    )
    with pytest.raises(ValueError, match="divisible by the TMA box width"):
        validate_config(cfg)


@pytest.mark.parametrize(
    ("tile_n", "epi_tile_n"),
    ((128, 128), (256, 128), (256, 256)),
)
def test_non_swap_gated_tma_ring_uses_physical_box_bytes(tile_n, epi_tile_n):
    cfg = _bf16_batch_m(
        tile_n=tile_n,
        mma_n=tile_n,
        epi_tile_n=epi_tile_n,
        num_stages_c_smem=2,
    )
    validate_config(cfg)

    assert cfg.uses_non_swap_c_tma_ring
    assert cfg.non_swap_gated_tma_box_bytes == 16 * 1024
    assert cfg.num_bytes_c_smem_scratch == (
        cfg.non_swap_gated_tma_box_bytes
        * cfg.num_stages_c_smem
        * cfg.num_epilogue_warpgroups
    )
    assert not replace(cfg, use_deepseek_fp8=1).uses_non_swap_c_tma_ring
    assert not replace(cfg, dtype_c=int(DType.E2M1)).uses_non_swap_c_tma_ring
    assert not replace(cfg, dtype_c=int(DType.E4M3)).uses_non_swap_c_tma_ring


def test_swap_gated_tma_scratch_size_is_unchanged():
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        act_kind=int(ActKind.SWIGLU),
        dtype_c=int(DType.BF16),
        tile_n=64,
        epi_tile_n=64,
        num_stages_c_smem=2,
        use_tma_store=1,
    )
    validate_config(cfg)

    assert cfg.is_swap_ab
    assert cfg.num_bytes_c_smem_scratch == (
        cfg.num_bytes_c_tma_store_per_group
        * cfg.num_stages_c_smem
        * cfg.num_epilogue_warpgroups
    )


def test_streamed_gated_batch_m_rejects_overlap_and_nvfp4_tma():
    overlap = _bf16_batch_m(
        tile_n=256,
        mma_n=256,
        epi_tile_n=128,
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
    )
    with pytest.raises(ValueError, match="requires one register step"):
        validate_config(overlap)

    nvfp4_tma = replace(
        _bf16_batch_m(),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        mma_k=64,
        tile_k=256,
    )
    with pytest.raises(ValueError, match="block-scaled-output"):
        validate_config(nvfp4_tma)


@pytest.mark.parametrize(
    ("dtype", "tile_k", "mma_k"),
    ((DType.BF16, 64, 16), (DType.E4M3, 128, 32)),
)
def test_unscaled_batch_m_overlap_uses_accumulator_only_window(dtype, tile_k, mma_k):
    cfg = _bf16_batch_m(
        dtype_a=int(dtype),
        dtype_b=int(dtype),
        tile_n=256,
        mma_n=256,
        tile_k=tile_k,
        mma_k=mma_k,
        epi_tile_n=64,
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
        use_shuffled_matrix=0,
    )
    validate_config(cfg)
    assert cfg.has_unscaled_batch_m
    assert cfg.uses_tmem_overlap_window
    assert cfg.tmem_overlap_window_cols == 448
    assert cfg.tmem_required_cols == 448
    assert cfg.fuse_sf_copy_to_mma == 0
    assert cfg.warp_layout.num_load_sfa_warps == 0
    assert cfg.warp_layout.num_load_sfb_warps == 0


def test_unscaled_bf16_overlap_requires_tile_k_64():
    cfg = _bf16_batch_m(
        tile_n=256,
        mma_n=256,
        tile_k=128,
        epi_tile_n=64,
        num_stages_tmem_acc=1,
        use_max_tmem_overlap=1,
    )
    with pytest.raises(ValueError, match="tile_k=128"):
        validate_config(cfg)


def test_batch_m_overlap_loader_warps_follow_transport():
    base = dict(
        batch_mode=int(BatchMode.BATCH_M),
        transpose_mma_output=0,
        act_kind=int(ActKind.SWIGLU),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.BF16),
        sf_layout_b=int(SfLayout.R128c4),
        umma_cta_count=2,
        mma_m=256,
        mma_n=256,
        mma_k=64,
        tile_n=256,
        tile_k=256,
        epi_tile_n=64,
        tile_scheduler=int(TileScheduler.PERSISTENT),
        num_stages_tmem_acc=1,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_tma_store=1,
    )
    plain = make_config(
        **base,
        route_act=int(RouteImpl.NONE),
        route_sfs_act=int(RouteImpl.NONE),
    )
    validate_config(plain)
    assert plain.warp_layout.num_load_a_warps == 1
    assert plain.warp_layout.num_load_sfa_warps == 1

    routed = make_config(
        **base,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.LDGSTS),
        num_load_a_warps=8,
        num_load_sfa_warps=4,
    )
    validate_config(routed)
    assert routed.warp_layout.num_load_a_warps == 8
    assert routed.warp_layout.num_load_sfa_warps == 4

    tma_routed_sfs = make_config(
        **base,
        route_act=int(RouteImpl.TMA),
        route_sfs_act=int(RouteImpl.TMA),
        num_load_a_warps=8,
        num_load_sfa_warps=4,
    )
    # TMA routing uses a linear SFA staging layout and is independently
    # rejected by fused-copy validation. Warp layout still must provision the
    # gather4 producer correctly before validation reports that combination.
    assert tma_routed_sfs.warp_layout.num_load_a_warps == 8
    assert tma_routed_sfs.warp_layout.num_load_sfa_warps == 4


@pytest.mark.parametrize("sm_arch", (100, 107))
def test_tile256_overlap_capacity_by_arch(sm_arch):
    cfg = _clustered(
        sm_arch=sm_arch,
        tile_n=256,
        mma_n=256,
        epi_tile_n=32,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
    )
    assert cfg.uses_tmem_overlap_window
    if sm_arch == 100:
        with pytest.raises(
            ValueError,
            match="TMEM footprint requires 528 columns",
        ):
            validate_config(cfg)
        validate_config(replace(cfg, epi_tile_n=64))
    else:
        validate_config(cfg)
        assert cfg.tmem_required_cols == 528
        assert cfg.tmem_alloc_cols == 544


@pytest.mark.parametrize(
    (
        "dtype_a",
        "dtype_b",
        "tile_k",
        "mma_k",
        "epi_tile_n",
        "expected_required_cols",
    ),
    (
        (DType.E2M1, DType.E2M1, 256, 64, 64, 496),
        (DType.MXE2M1, DType.MXE4M3, 128, 32, 32, 492),
    ),
)
def test_tile256_overlap_scale_factor_layout(
    dtype_a, dtype_b, tile_k, mma_k, epi_tile_n, expected_required_cols
):
    cfg = _clustered(
        tile_n=256,
        mma_n=256,
        epi_tile_n=epi_tile_n,
        tile_k=tile_k,
        mma_k=mma_k,
        dtype_a=int(dtype_a),
        dtype_b=int(dtype_b),
        sf_layout_b=int(SfLayout.R128c4),
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
    )
    validate_config(cfg)
    assert (
        cfg.tmem_overlap_window_cols + cfg.tmem_sfa_cols + cfg.tmem_sfb_cols
        == cfg.tmem_required_cols
    )
    assert cfg.tmem_required_cols == expected_required_cols
    assert cfg.tmem_required_cols <= cfg.tmem_capacity_columns
    assert cfg.tmem_overlap_window_cols % 4 == 0


def test_tile256_overlap_fused_sf_allocation_matches_footprint():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_kernel import (
        _build_schedule_validate,
    )

    cfg = _clustered(
        tile_n=256,
        mma_n=256,
        epi_tile_n=64,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
    )
    _, _, _, tmem_allocator = _build_schedule_validate(cfg)
    assert tmem_allocator.total_tmem_columns == cfg.tmem_required_cols


def test_host_k_specialization_does_not_mutate_a_reusable_request():
    cfg = _clustered(num_stages_smem_sfb=6)
    short = resolve_problem_config(cfg, 256)
    long = resolve_problem_config(cfg, 2048)
    assert short.ldgsts_sfb_producer_commit_prefetch_depth == 1
    assert long.ldgsts_sfb_producer_commit_prefetch_depth == 3
    assert cfg.ldgsts_sfb_producer_commit_prefetch_depth == 3


def test_transpose_option_is_retained_and_checked():
    cfg = make_config(batch_mode=int(BatchMode.BATCH_M), transpose_mma_output=0)
    validate_config(cfg)
    with pytest.raises(ValueError, match="transpose_mma_output must match"):
        validate_config(replace(cfg, transpose_mma_output=1))


@pytest.mark.parametrize("act_kind", (ActKind.RELU2, ActKind.SILU))
def test_swap_eltwise_output_block_scaling_is_valid(act_kind):
    cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        act_kind=int(act_kind),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        epi_tile_m=128,
        use_shuffled_matrix=1,
    )
    validate_config(cfg)
    assert cfg.has_eltwise_act
    assert not cfg.has_gated_epilogue


def test_non_gated_output_block_scaling_rejects_non_swap_and_mismatched_epi_m():
    swap_cfg = make_config(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        act_kind=int(ActKind.RELU2),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        epi_tile_m=128,
        use_shuffled_matrix=1,
    )
    with pytest.raises(ValueError, match="requires swapAB"):
        validate_config(
            replace(
                swap_cfg,
                batch_mode=int(BatchMode.BATCH_M),
                transpose_mma_output=0,
                use_shuffled_matrix=0,
            )
        )
    with pytest.raises(ValueError, match="act_kind=RELU2 or SILU"):
        validate_config(replace(swap_cfg, act_kind=int(ActKind.NONE)))
    with pytest.raises(ValueError, match="epi_tile_m == tile_m"):
        validate_config(replace(swap_cfg, epi_tile_m=64))
    with pytest.raises(ValueError, match="epi_tile_m == tile_m"):
        validate_config(replace(swap_cfg, epi_tile_m=256))
    with pytest.raises(ValueError, match="whole output scale blocks"):
        validate_config(swap_cfg, problem_mnk=(130, 128, 128))


def test_non_gated_output_block_scaling_reserves_each_warpgroup_tma_stage():
    cfg = _clustered(
        batch_mode=int(BatchMode.BATCH_N),
        transpose_mma_output=1,
        act_kind=int(ActKind.RELU2),
        dtype_a=int(DType.E2M1),
        dtype_b=int(DType.E2M1),
        dtype_c=int(DType.E2M1),
        tile_n=256,
        mma_n=256,
        epi_tile_n=64,
        num_stages_c_smem=2,
        fuse_sf_copy_to_mma=1,
        use_max_tmem_overlap=1,
        use_tma_store=1,
        use_shuffled_matrix=1,
    )
    validate_config(cfg)
    assert cfg.num_epilogue_warpgroups == 2
    assert cfg.num_swap_ab_nongated_quant_tma_stages == 2
    assert cfg.num_bytes_c_smem_scratch >= (
        cfg.num_bytes_c_tma_store_per_group
        * cfg.num_epilogue_warpgroups
        * cfg.num_swap_ab_nongated_quant_tma_stages
    )
    assert replace(cfg, num_stages_c_smem=3).num_swap_ab_nongated_quant_tma_stages == 1


def test_compile_identity_uses_effective_assignments_and_shape_specialization():
    from flashinfer.prims_ts.moe.compile_cache import stable_config_hash

    cfg = _clustered()
    explicit = replace(cfg, num_load_a_warps=cfg.warp_layout.num_load_a_warps)
    assert stable_config_hash(cfg) == stable_config_hash(explicit)
    before = stable_config_hash(cfg)
    compute_warp_layout(cfg)
    assert stable_config_hash(cfg) == before
    assert stable_config_hash(resolve_problem_config(cfg, 256)) != stable_config_hash(
        resolve_problem_config(cfg, 2048)
    )


@pytest.mark.parametrize("batch", [BatchMode.BATCH_M, BatchMode.BATCH_N])
def test_auto_sf_route_is_distinct_from_explicit_preexpanded_storage(batch):
    auto = make_config(
        batch_mode=int(batch),
        transpose_mma_output=int(batch == BatchMode.BATCH_N),
        route_act=int(RouteImpl.TMA),
    )
    preexpanded = replace(auto, route_sfs_act=int(RouteImpl.NONE))
    validate_config(auto)
    validate_config(preexpanded)
    assert auto.sf_route == int(RouteImpl.LDGSTS)
    assert auto.has_routed_sfs
    assert preexpanded.sf_route == int(RouteImpl.NONE)
    assert not preexpanded.has_routed_sfs
    assert auto.compile_signature() != preexpanded.compile_signature()


def test_cli_exposes_canonical_fusion_and_routing_requests():
    from flashinfer.prims_ts.batched_gemm.batched_gemm_run import (
        _parse_overrides,
        build_arg_parser,
    )

    parser = build_arg_parser()
    defaults = _parse_overrides(parser.parse_args([]))
    assert defaults["route_sfs_act"] == int(RouteImpl.AUTO)
    explicit = _parse_overrides(
        parser.parse_args(
            [
                "--fuse-sf-copy-to-mma",
                "1",
                "--fuse-operand-sf-loads",
                "1",
                "--route-sfs-act",
                "none",
            ]
        )
    )
    assert explicit["fuse_sf_copy_to_mma"] == 1
    assert explicit["fuse_operand_sf_loads"] == 1
    assert explicit["route_sfs_act"] == int(RouteImpl.NONE)


@pytest.mark.parametrize("operand", ["a", "b", "sfa", "sfb"])
def test_non_routed_tma_warp_override_cannot_duplicate_full_tile_transactions(operand):
    cfg = make_config(**{f"num_load_{operand}_warps": 4})
    with pytest.raises(ValueError, match="one warp for a non-routed TMA loader"):
        validate_config(cfg)
