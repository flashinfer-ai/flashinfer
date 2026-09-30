"""Unit tests for the DeepGEMM-style expected_m regime hint in dense_gemm's
default tile selector (_select_default_mma_tiler_mn).

These are pure CPU/logic tests (no kernel launch): they pin the per-regime tile
mapping and the M-independence-within-regime contract that lets one compiled
kernel per (N,K,expected_m) be reused for all live M under frozen resolution.
"""

from __future__ import annotations

import cutlass
import pytest
import torch

import b12x._lib.dense_gemm as dense_module
from b12x._lib.dense_gemm import (
    _DenseGemmFusedQuantALaunch,
    _DenseGemmLaunch,
    _DenseGemmPolicy,
    _dense_gemm_policy_for,
    _dense_gemm_target_occupancy,
    _use_low_sm_dense_tactics,
    _tile_major_cluster_limit,
    _select_default_dense_gemm_plan,
    _select_default_mma_tiler_mn,
    _select_block_fp8_decode_slices,
    _select_fp4_tile_k,
    _select_mxfp8_tile_k,
    _validate_mxfp8_bk64_plan,
    dense_gemm,
)

HIGH_SM = 188
LOW_SM = 48
WIDE_N = 4096  # n > 1536 -> the MXFP8 wide-N regime that the hint tunes


@pytest.mark.parametrize("expected_m", (1, 8, 16, 129, 2048, 8192))
@pytest.mark.parametrize("sm_count", (LOW_SM, HIGH_SM))
@pytest.mark.parametrize("c_dtype", (cutlass.BFloat16, cutlass.Float16))
def test_expected_m_bounds_the_complete_scheduling_specialization(
    expected_m: int, sm_count: int, c_dtype,
) -> None:
    n = k = 4096
    plan = _select_default_dense_gemm_plan(
        expected_m, n, k, sm_count, is_mxfp8=True, expected_m=expected_m
    )
    tile_k = _select_mxfp8_tile_k(expected_m, n, k, expected_m, sm_count)
    policies = {
        _dense_gemm_policy_for(
            m=live_m,
            n=n,
            k=k,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=c_dtype,
            mma_tiler_mn=plan.mma_tiler_mn,
            cluster_shape_mn=(1, 1),
            sm_count=sm_count,
            tile_k=tile_k,
            expected_m=expected_m,
            generalize_mxfp8_split_k=True,
        )
        for live_m in (1, 2, 8, 9, 15, 16, 128, 129, expected_m)
        if live_m <= expected_m
    }
    assert len(policies) == 1


@pytest.mark.parametrize("m", (1, 2, 4, 6, 8, 16, 32, 64, 128))
@pytest.mark.parametrize(
    "n,k,expected_tile_k",
    (
        (17408, 5120, 256),
        (8704, 5120, 256),
        (4352, 5120, 256),
        (2176, 5120, 128),
        (5120, 17408, 256),
        (5120, 8704, 256),
        (5120, 4352, 256),
        (5120, 2176, 128),
    ),
)
def test_qwen38_tp_fp4_decode_follows_bk256_cutoff(
    m: int,
    n: int,
    k: int,
    expected_tile_k: int,
) -> None:
    assert (
        _select_fp4_tile_k(m, n, k, None, 48, (64, 128)) == expected_tile_k
    )


def test_fp4_bk256_uses_tile_and_half_sm_wave_cutoffs() -> None:
    assert _select_fp4_tile_k(1, 1024, 4096, None, 48, (64, 64)) == 256
    assert _select_fp4_tile_k(1, 3072, 4096, None, 48, (64, 128)) == 256
    assert _select_fp4_tile_k(1, 2944, 4096, None, 48, (64, 128)) == 128


def test_fp4_bk256_policy_preserves_unqualified_regimes() -> None:
    assert _select_fp4_tile_k(129, 17408, 5120, None, 48, (128, 128)) == 128
    assert _select_fp4_tile_k(1, 4096, 3072, None, 48, (64, 128)) == 128
    assert _select_fp4_tile_k(1, 4096, 5376, None, 188, (64, 128)) == 128
    assert _select_fp4_tile_k(1, 17408, 5120, 129, 48, (64, 128)) == 128


@pytest.mark.parametrize("plan_m", (48, 64))
def test_fp4_medium_m_deep_k_plan_uses_swapped_narrow_tile(plan_m: int) -> None:
    plan = _select_default_dense_gemm_plan(
        plan_m,
        4096,
        5376,
        48,
        is_mxfp8=False,
        expected_m=plan_m,
    )

    assert plan.mma_tiler_mn == (64, 32)
    assert plan.load_path == "tma"
    assert plan.swap_ab


@pytest.mark.parametrize(
    "plan_m,n,k,sm_count",
    (
        (32, 4096, 5376, 48),
        (96, 4096, 5376, 48),
        (64, 3968, 5376, 48),
        (64, 4096, 4992, 48),
        (64, 4096, 5376, HIGH_SM),
    ),
)
def test_fp4_swapped_medium_m_plan_preserves_unqualified_boundaries(
    plan_m: int,
    n: int,
    k: int,
    sm_count: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        plan_m,
        n,
        k,
        sm_count,
        is_mxfp8=False,
        expected_m=plan_m,
    )

    assert plan.mma_tiler_mn != (64, 32)
    assert not plan.swap_ab


@pytest.mark.parametrize(
    "plan_m,n,k",
    (
        (1, 512, 5376),
        (6, 2048, 5376),
        (64, 5120, 17408),
        (128, 5120, 8704),
    ),
)
def test_high_sm_fp4_small_batch_uses_square_bk256_plan(
    plan_m: int,
    n: int,
    k: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        plan_m,
        n,
        k,
        HIGH_SM,
        is_mxfp8=False,
        expected_m=plan_m,
    )

    assert plan.mma_tiler_mn == (64, 64)
    assert not plan.swap_ab
    assert _select_fp4_tile_k(
        plan_m, n, k, plan_m, HIGH_SM, plan.mma_tiler_mn
    ) == 256


def test_high_sm_fp4_wide_output_uses_grid_cutoff_for_bk256() -> None:
    assert _select_fp4_tile_k(
        128, 8704, 5120, 128, HIGH_SM, (64, 128)
    ) == 256
    assert _select_fp4_tile_k(
        128, 7936, 5120, 128, HIGH_SM, (64, 128)
    ) == 128


def test_high_sm_fp4_underfilled_prefill_uses_64x128() -> None:
    plan = _select_default_dense_gemm_plan(
        2048,
        512,
        5376,
        HIGH_SM,
        is_mxfp8=False,
        expected_m=2048,
    )

    assert plan.mma_tiler_mn == (64, 128)


def test_mxfp8_split_k_policy_uses_sm_and_grid_capacity() -> None:
    kwargs = dict(
        m=8,
        n=4096,
        k=4096,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
    )

    assert (
        _dense_gemm_policy_for(
            sm_count=48, generalize_mxfp8_split_k=True, **kwargs
        ).split_k_slices
        == 1
    )
    assert (
        _dense_gemm_policy_for(
            sm_count=HIGH_SM, generalize_mxfp8_split_k=True, **kwargs
        ).split_k_slices
        == 4
    )
    assert (
        _dense_gemm_policy_for(
            sm_count=HIGH_SM,
            generalize_mxfp8_split_k=True,
            **(kwargs | {"n": 6144}),
        ).split_k_slices
        == 2
    )
    assert (
        _dense_gemm_policy_for(
            sm_count=HIGH_SM,
            generalize_mxfp8_split_k=True,
            **(kwargs | {"m": 4}),
        ).split_k_slices
        == 1
    )


def test_high_sm_mxfp8_decode_uses_four_slices_for_bounded_grid() -> None:
    policy = _dense_gemm_policy_for(
        m=6,
        n=4352,
        k=5120,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        sm_count=HIGH_SM,
        expected_m=6,
        generalize_mxfp8_split_k=True,
    )

    assert policy.split_k_slices == 4


@pytest.mark.parametrize("m", (1, 6))
def test_low_sm_mxfp8_decode_plan_uses_grid_based_four_way_split(m: int) -> None:
    plan = _select_default_dense_gemm_plan(
        m, 3072, 5120, 48, is_mxfp8=True, expected_m=m
    )
    policy = _dense_gemm_policy_for(
        m=m,
        n=3072,
        k=5120,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=48,
        tile_k=128,
        expected_m=m,
        generalize_mxfp8_split_k=True,
    )

    assert plan.mma_tiler_mn == (32, 64)
    assert policy.split_k_slices == 4


@pytest.mark.parametrize(
    "m,n,k",
    (
        (1, 3008, 5120),
        (6, 2176, 5120),
        (8, 3072, 5120),
        (6, 3072, 8192),
        (6, 3072, 5376),
    ),
)
def test_low_sm_mxfp8_decode_split_preserves_unqualified_boundaries(
    m: int,
    n: int,
    k: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        m, n, k, 48, is_mxfp8=True, expected_m=m
    )

    assert plan.mma_tiler_mn != (32, 64)


@pytest.mark.parametrize("expected_m", (1536, 2048))
def test_low_sm_mxfp8_medium_prefill_uses_bk64_plan(expected_m: int) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        4096,
        4096,
        48,
        is_mxfp8=True,
        expected_m=expected_m,
    )
    tile_k = _select_mxfp8_tile_k(
        expected_m, 4096, 4096, expected_m, 48
    )
    policy = _dense_gemm_policy_for(
        m=expected_m,
        n=4096,
        k=4096,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=48,
        tile_k=tile_k,
        expected_m=expected_m,
        generalize_mxfp8_split_k=True,
    )

    assert plan.mma_tiler_mn == (128, 128)
    assert tile_k == 64
    assert policy.large_m_unroll


def test_low_sm_mxfp8_bk64_large_prefill_retains_swizzle() -> None:
    policy = _dense_gemm_policy_for(
        m=8192,
        n=4096,
        k=4096,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(128, 128),
        cluster_shape_mn=(1, 1),
        sm_count=48,
        tile_k=64,
        expected_m=8192,
        generalize_mxfp8_split_k=True,
    )

    assert not policy.large_m_unroll


@pytest.mark.parametrize("expected_m", (512, 1536, 2048, 3072))
def test_low_sm_narrow_wide_mxfp8_prefill_uses_128x64(
    expected_m: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        2176,
        5120,
        48,
        is_mxfp8=True,
        expected_m=expected_m,
    )

    assert plan.mma_tiler_mn == (128, 64)
    assert _select_mxfp8_tile_k(expected_m, 2176, 5120, expected_m, 48) == 128


@pytest.mark.parametrize(
    "expected_m,n",
    ((256, 2176), (4096, 2176), (1536, 1536), (1536, 4096)),
)
def test_low_sm_narrow_wide_mxfp8_prefill_preserves_boundaries(
    expected_m: int,
    n: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        n,
        5120,
        48,
        is_mxfp8=True,
        expected_m=expected_m,
    )

    assert plan.mma_tiler_mn != (128, 64)


@pytest.mark.parametrize("n", (4096, 8192, 16384, 32768))
@pytest.mark.parametrize("expected_m", (1, 6))
def test_low_sm_short_k_decode_uses_grid_filling_tile(
    n: int,
    expected_m: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        n,
        1024,
        48,
        is_mxfp8=True,
        expected_m=expected_m,
    )

    assert plan.mma_tiler_mn == (16, 64)
    assert _select_mxfp8_tile_k(expected_m, n, 1024, expected_m, 48) == 128


@pytest.mark.parametrize(
    "n,expected_tile,expected_tile_k",
    (
        (4096, (128, 128), 64),
        (8192, (128, 128), 64),
        (16384, (64, 128), 128),
        (32768, (64, 128), 128),
    ),
)
def test_low_sm_short_k_medium_prefill_uses_output_grid_cutoff(
    n: int,
    expected_tile: tuple[int, int],
    expected_tile_k: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        2048, n, 1024, 48, is_mxfp8=True, expected_m=2048
    )

    assert plan.mma_tiler_mn == expected_tile
    assert _select_mxfp8_tile_k(2048, n, 1024, 2048, 48) == expected_tile_k


@pytest.mark.parametrize("n", (4096, 8192, 16384, 32768))
def test_low_sm_short_k_large_prefill_returns_bk128(n: int) -> None:
    plan = _select_default_dense_gemm_plan(
        4096, n, 1024, 48, is_mxfp8=True, expected_m=4096
    )

    assert plan.mma_tiler_mn == (64, 128)
    assert _select_mxfp8_tile_k(4096, n, 1024, 4096, 48) == 128


@pytest.mark.parametrize(
    "expected_m,n,k",
    (
        (1024, 4096, 4096),
        (3072, 4352, 4096),
        (2048, 3968, 4096),
        (2048, 4096, 1984),
    ),
)
def test_low_sm_mxfp8_medium_prefill_preserves_boundaries(
    expected_m: int,
    n: int,
    k: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        n,
        k,
        48,
        is_mxfp8=True,
        expected_m=expected_m,
    )
    tile_k = _select_mxfp8_tile_k(expected_m, n, k, expected_m, 48)

    assert not (plan.mma_tiler_mn == (128, 128) and tile_k == 64)


@pytest.mark.parametrize("expected_m", (2048, 4096))
@pytest.mark.parametrize(
    "n,k,expected_tile,expected_tile_k",
    (
        (32768, 1024, (128, 128), 64),
        (8192, 1024, (128, 128), 64),
        (17408, 5120, (128, 128), 64),
        (8704, 5120, (128, 128), 64),
        (4096, 5376, (128, 128), 64),
        (6144, 1536, (128, 128), 64),
        (5120, 17408, (128, 64), 128),
        (4352, 5120, (128, 64), 128),
        (2176, 5120, (128, 64), 128),
        (1536, 4096, (128, 64), 128),
        (1024, 4096, (128, 64), 128),
    ),
)
def test_high_sm_mxfp8_prefill_plan_covers_tp_and_common_shapes(
    expected_m: int,
    n: int,
    k: int,
    expected_tile: tuple[int, int],
    expected_tile_k: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        1,
        n,
        k,
        HIGH_SM,
        is_mxfp8=True,
        expected_m=expected_m,
    )

    assert plan.mma_tiler_mn == expected_tile
    assert _select_mxfp8_tile_k(
        1, n, k, expected_m, HIGH_SM
    ) == expected_tile_k


def test_non_mxfp8_split_k_policy_is_unchanged() -> None:
    policy = _dense_gemm_policy_for(
        m=8,
        n=4096,
        k=4096,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        sm_count=48,
    )

    assert policy.split_k_slices == 4


@pytest.mark.parametrize(
    "k,expected_slices",
    ((4096, 4), (4352, 2), (5120, 4), (5376, 1), (8704, 4), (17408, 4)),
)
@pytest.mark.parametrize("m", (1, 6))
def test_low_sm_block_fp8_decode_uses_divisibility_based_slices(
    m: int,
    k: int,
    expected_slices: int,
) -> None:
    assert _select_block_fp8_decode_slices(m, 2048, k, 48) == expected_slices
    plan = _select_default_dense_gemm_plan(
        m,
        2048,
        k,
        48,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=m,
    )

    if expected_slices > 1:
        assert plan.mma_tiler_mn == (32, 64)


@pytest.mark.parametrize(
    "m,n,k,sm_count",
    (
        (8, 2048, 5120, 48),
        (6, 1984, 5120, 48),
        (6, 2048, 5376, 48),
    ),
)
def test_low_sm_block_fp8_decode_preserves_unqualified_boundaries(
    m: int,
    n: int,
    k: int,
    sm_count: int,
) -> None:
    assert _select_block_fp8_decode_slices(m, n, k, sm_count) == 1


@pytest.mark.parametrize(
    "m,k,expected_slices",
    (
        (1, 5120, 1),
        (2, 5120, 2),
        (6, 4096, 2),
        (6, 5376, 2),
        (6, 2176, 1),
        (8, 5120, 1),
    ),
)
def test_high_sm_block_fp8_decode_uses_bounded_two_way_split(
    m: int,
    k: int,
    expected_slices: int,
) -> None:
    assert (
        _select_block_fp8_decode_slices(m, 4096, k, HIGH_SM)
        == expected_slices
    )
    plan = _select_default_dense_gemm_plan(
        m,
        4096,
        k,
        HIGH_SM,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=m,
    )
    policy = _dense_gemm_policy_for(
        m=m,
        n=4096,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=HIGH_SM,
        expected_m=m,
        generalize_block_fp8_split_k=True,
    )

    assert policy.split_k_slices == expected_slices
    if expected_slices > 1:
        assert plan.mma_tiler_mn == (32, 64)


@pytest.mark.parametrize(
    "n,k",
    (
        (17408, 5120),
        (5120, 17408),
        (4096, 4096),
        (16384, 1024),
        (6144, 1536),
    ),
)
@pytest.mark.parametrize("expected_m", (2048, 4096))
def test_high_sm_block_fp8_prefill_uses_64x128(
    n: int,
    k: int,
    expected_m: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        n,
        k,
        HIGH_SM,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=expected_m,
    )

    assert plan.mma_tiler_mn == (64, 128)


def test_high_sm_block_fp8_m128_uses_output_grid_window() -> None:
    assert _select_default_dense_gemm_plan(
        128,
        8704,
        5120,
        HIGH_SM,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=128,
    ).mma_tiler_mn == (64, 128)
    assert _select_default_dense_gemm_plan(
        128,
        17408,
        5120,
        HIGH_SM,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=128,
    ).mma_tiler_mn == (32, 128)
    assert _select_default_dense_gemm_plan(
        128,
        4352,
        5120,
        HIGH_SM,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=128,
    ).mma_tiler_mn == (32, 128)


@pytest.mark.parametrize("k", (10240, 12288, 17408))
@pytest.mark.parametrize("expected_m", (2048, 4096))
def test_low_sm_block_fp8_deep_k_prefill_uses_128_tile_and_small_unroll(
    k: int,
    expected_m: int,
) -> None:
    plan = _select_default_dense_gemm_plan(
        expected_m,
        5120,
        k,
        48,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=expected_m,
    )
    policy = _dense_gemm_policy_for(
        m=expected_m,
        n=5120,
        k=k,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        mma_tiler_mn=plan.mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        sm_count=48,
        tile_k=128,
        expected_m=expected_m,
        generalize_block_fp8_split_k=True,
    )

    assert plan.mma_tiler_mn == (128, 128)
    assert not policy.large_m_unroll


@pytest.mark.parametrize("k", (8192, 8704))
def test_low_sm_block_fp8_prefill_preserves_shorter_k_boundary(k: int) -> None:
    plan = _select_default_dense_gemm_plan(
        4096,
        5120,
        k,
        48,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=4096,
    )

    assert plan.mma_tiler_mn == (64, 128)


@pytest.mark.parametrize("k", (8704, 17408))
def test_low_sm_block_fp8_deep_k_m128_uses_128_tile(k: int) -> None:
    plan = _select_default_dense_gemm_plan(
        128,
        5120,
        k,
        48,
        is_mxfp8=True,
        block_fp8=True,
        expected_m=128,
    )

    assert plan.mma_tiler_mn == (128, 128)


def test_tile_major_cluster_cap_uses_output_grid_geometry() -> None:
    assert _tile_major_cluster_limit(48, n=1024, l=4, tile_n=64) == 40
    assert _tile_major_cluster_limit(48, n=4096, l=1, tile_n=64) == 40
    assert _tile_major_cluster_limit(20, n=4096, l=1, tile_n=64) == 20
    assert _tile_major_cluster_limit(48, n=2048, l=1, tile_n=64) == 48


def _unaligned_dense_operands(
    rows: int,
    columns: int,
    width: int,
    groups: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    lhs = (
        torch.empty((rows, width, groups), dtype=torch.float8_e4m3fn),
        torch.empty((1,), dtype=torch.float8_e8m0fnu),
    )
    rhs = (
        torch.empty((columns, width, groups), dtype=torch.float8_e4m3fn),
        torch.empty((1,), dtype=torch.float8_e8m0fnu),
    )
    return lhs, rhs


def _packed_fp4_operands(
    rows: int,
    columns: int,
    width: int,
) -> tuple[tuple[torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]]:
    lhs = (
        torch.empty((rows, width // 2, 1), dtype=torch.uint8),
        torch.empty((1,), dtype=torch.float8_e4m3fn),
    )
    rhs = (
        torch.empty((columns, width // 2, 1), dtype=torch.uint8),
        torch.empty((1,), dtype=torch.float8_e4m3fn),
    )
    return lhs, rhs


def test_fp4_tile_k_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs, rhs = _packed_fp4_operands(1, 128, 512)
    out = torch.empty((1, 128, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        sm_count=LOW_SM,
        mma_tiler_mn=(64, 128),
        load_path="tma",
        swap_ab=False,
        _tile_k_override=512,
    )

    assert result is out
    assert recorded["tile_k"] == 512


def test_fp4_tile_k_override_rejects_invalid_depth() -> None:
    lhs, rhs = _packed_fp4_operands(1, 128, 512)

    with pytest.raises(ValueError, match="must be 128, 256, or 512"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            sm_count=LOW_SM,
            _tile_k_override=384,
        )


def test_fp4_target_occupancy_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs, rhs = _packed_fp4_operands(1, 128, 512)
    out = torch.empty((1, 128, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float4_e2m1fn",
        sf_dtype="float8_e4m3fn",
        c_dtype="bfloat16",
        sf_vec_size=16,
        sm_count=LOW_SM,
        _target_occupancy_override=3,
    )

    assert result is out
    assert recorded["target_occupancy_override"] == 3


@pytest.mark.parametrize("occupancy", (0, 5))
def test_fp4_target_occupancy_override_rejects_invalid_value(
    occupancy: int,
) -> None:
    lhs, rhs = _packed_fp4_operands(1, 128, 512)
    out = torch.empty((1, 128, 1), dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="must be 1, 2, 3, or 4"):
        dense_gemm(
            lhs,
            rhs,
            out=out,
            ab_dtype="float4_e2m1fn",
            sf_dtype="float8_e4m3fn",
            c_dtype="bfloat16",
            sf_vec_size=16,
            sm_count=LOW_SM,
            _target_occupancy_override=occupancy,
        )


def test_mxfp8_tile_k_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs, rhs = _unaligned_dense_operands(4, 4096, 4096, 1)
    out = torch.empty((4, 4096, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        sm_count=LOW_SM,
        mma_tiler_mn=(128, 128),
        load_path="tma",
        swap_ab=False,
        _tile_k_override=64,
    )

    assert result is out
    assert recorded["tile_k"] == 64


def test_mxfp8_tile_k_override_rejects_invalid_depth() -> None:
    lhs, rhs = _unaligned_dense_operands(4, 4096, 4096, 1)

    with pytest.raises(ValueError, match=r"MXFP8.*one of \(64, 128\)"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=LOW_SM,
            mma_tiler_mn=(128, 128),
            _tile_k_override=256,
        )


@pytest.mark.parametrize("large_m_unroll", (False, True))
def test_mxfp8_large_m_unroll_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
    large_m_unroll: bool,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs, rhs = _unaligned_dense_operands(128, 4096, 4096, 1)
    out = torch.empty((128, 4096, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        sm_count=LOW_SM,
        mma_tiler_mn=(64, 128),
        _large_m_unroll_override=large_m_unroll,
    )

    assert result is out
    assert recorded["policy"].large_m_unroll is large_m_unroll


def test_large_m_unroll_override_rejects_non_bool() -> None:
    lhs, rhs = _unaligned_dense_operands(128, 4096, 4096, 1)

    with pytest.raises(ValueError, match="must be a bool"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=LOW_SM,
            _large_m_unroll_override=1,
        )


def test_mxfp8_split_k_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs, rhs = _unaligned_dense_operands(4, 4096, 4096, 1)
    out = torch.empty((4, 4096, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        sm_count=LOW_SM,
        mma_tiler_mn=(16, 128),
        load_path="tma",
        swap_ab=False,
        _split_k_slices_override=4,
    )

    assert result is out
    assert recorded["policy"].split_k_slices == 4
    assert recorded["policy"].split_k_atomic_bf16


def test_block_fp8_split_k_override_reaches_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(dense_module, "_get_compiled_dense_gemm", compile_dense_gemm)
    lhs = (
        torch.empty((4, 4096, 1), dtype=torch.float8_e4m3fn),
        torch.empty((4, 32), dtype=torch.float32),
    )
    rhs = (
        torch.empty((4096, 4096, 1), dtype=torch.float8_e4m3fn),
        torch.empty((32, 32), dtype=torch.float32),
    )
    out = torch.empty((4, 4096, 1), dtype=torch.bfloat16)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float32",
        c_dtype="bfloat16",
        sf_vec_size=128,
        block_fp8=True,
        sm_count=LOW_SM,
        mma_tiler_mn=(16, 128),
        _split_k_slices_override=4,
    )

    assert result is out
    assert recorded["policy"].split_k_slices == 4
    assert recorded["policy"].split_k_atomic_bf16


@pytest.mark.parametrize("slices", (0, 3, 8))
def test_mxfp8_split_k_override_rejects_invalid_slice_count(slices: int) -> None:
    lhs, rhs = _unaligned_dense_operands(4, 4096, 4096, 1)

    with pytest.raises(ValueError, match="must be 1, 2, or 4"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=LOW_SM,
            _split_k_slices_override=slices,
        )


def test_mxfp8_split_k_override_rejects_uneven_k_tiles() -> None:
    lhs, rhs = _unaligned_dense_operands(4, 4096, 5376, 1)

    with pytest.raises(ValueError, match="divide evenly across slices"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=LOW_SM,
            mma_tiler_mn=(16, 128),
            load_path="tma",
            swap_ab=False,
            _split_k_slices_override=4,
        )


def test_unswapped_tma_epilogue_rejects_unaligned_output_row_stride() -> None:
    lhs, rhs = _unaligned_dense_operands(2, 132, 128, 1)

    with pytest.raises(ValueError, match="16-byte-aligned C row stride"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=HIGH_SM,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=False,
        )


def test_grouped_unaligned_output_requires_padding() -> None:
    lhs, rhs = _unaligned_dense_operands(1, 132, 128, 2)

    with pytest.raises(
        ValueError,
        match="pad N; swapped output storage is unsupported when L > 1",
    ):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=HIGH_SM,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=False,
        )


def test_grouped_output_rejects_swapped_storage() -> None:
    lhs, rhs = _unaligned_dense_operands(1, 132, 128, 2)

    with pytest.raises(ValueError, match=r"swapped dense_gemm.*supports L=1 only"):
        dense_gemm(
            lhs,
            rhs,
            ab_dtype="float8_e4m3fn",
            sf_dtype="float8_e8m0fnu",
            c_dtype="bfloat16",
            sf_vec_size=32,
            sm_count=HIGH_SM,
            mma_tiler_mn=(64, 64),
            load_path="tma",
            swap_ab=True,
        )


@pytest.mark.parametrize(
    (
        "rows",
        "columns",
        "width",
        "expected_m",
        "select_swapped_output_storage",
        "expected_tile",
        "expected_swap",
    ),
    (
        (1, 132, 7168, 1, False, (16, 64), False),
        (2, 132, 7168, 2, True, (64, 32), True),
        (8, 132, 7168, 8, True, (64, 32), True),
        (512, 132, 7168, 512, True, (128, 64), True),
        (2, 128, 7168, 2, False, (64, 64), False),
        (1, 32, 7168, 1, False, (16, 64), False),
        (1, 32, 7168, 1, True, (64, 32), True),
        (2, 16386, 1024, 2048, True, (128, 128), True),
    ),
)
def test_default_plan_adapts_to_output_storage(
    rows: int,
    columns: int,
    width: int,
    expected_m: int,
    select_swapped_output_storage: bool,
    expected_tile: tuple[int, int],
    expected_swap: bool,
) -> None:
    plan = _select_default_dense_gemm_plan(
        rows,
        columns,
        width,
        HIGH_SM,
        is_mxfp8=True,
        expected_m=expected_m,
        select_swapped_output_storage=select_swapped_output_storage,
    )

    assert plan.mma_tiler_mn == expected_tile
    assert plan.swap_ab is expected_swap


def test_wide_unaligned_output_uses_supported_bk128_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorded: dict[str, object] = {}

    def compile_dense_gemm(**kwargs):
        recorded.update(kwargs)

        def launch(**_kwargs) -> None:
            return None

        return launch

    monkeypatch.setattr(
        dense_module,
        "_get_compiled_dense_gemm",
        compile_dense_gemm,
    )
    lhs, rhs = _unaligned_dense_operands(2, 16386, 1024, 1)
    out = torch.empty((2, 16386, 1), dtype=torch.bfloat16)
    alpha = torch.ones(1, dtype=torch.float32)

    result = dense_gemm(
        lhs,
        rhs,
        out=out,
        ab_dtype="float8_e4m3fn",
        sf_dtype="float8_e8m0fnu",
        c_dtype="bfloat16",
        sf_vec_size=32,
        sm_count=HIGH_SM,
        expected_m=2048,
        alpha=alpha,
        plain_fp8=True,
    )

    assert result is out
    assert recorded["tile_k"] == 128
    assert recorded["mma_tiler_mn"] == (128, 128)
    assert recorded["swap_ab"] is True
    assert dense_module.DenseGemmKernel.can_implement(
        recorded["ab_dtype"],
        recorded["sf_dtype"],
        recorded["sf_vec_size"],
        recorded["c_dtype"],
        recorded["mma_tiler_mn"],
        recorded["cluster_shape_mn"],
        16386,
        1024,
        1,
        "k",
        "k",
        "n",
        load_path=recorded["load_path"],
        swap_ab=recorded["swap_ab"],
    )


def _tile(m, *, expected_m=None, n=WIDE_N, k=None, sm_count=HIGH_SM):
    return _select_default_mma_tiler_mn(
        m, n, sm_count, is_mxfp8=True, expected_m=expected_m, k=k
    )


def _bk64_launch(
    *,
    launch_type=_DenseGemmLaunch,
    policy: _DenseGemmPolicy | None = None,
    sfb_k_reuse: bool = True,
    n: int = 16384,
    k: int = 1024,
    l: int = 1,
    mma_tiler_mn: tuple[int, int] = (128, 128),
    b_tile_major: bool = False,
) -> _DenseGemmLaunch:
    if policy is None:
        policy = _DenseGemmPolicy(
            single_work_tile_per_cta=False,
            direct_one_m_tile_scheduler=False,
            use_m1_non_tma=False,
            split_k_slices=1,
            split_k_atomic_bf16=False,
            large_m_unroll=True,
        )
    return launch_type(
        n=n,
        k=k,
        l=l,
        c_l=1,
        a_major="k",
        b_major="k",
        c_major="n",
        ab_dtype=cutlass.Float8E4M3FN,
        sf_dtype=cutlass.Float8E8M0FNU,
        c_dtype=cutlass.BFloat16,
        alpha_dtype=cutlass.Float32,
        sf_vec_size=32,
        mma_k=32,
        tile_k=64,
        mma_tiler_mn=mma_tiler_mn,
        cluster_shape_mn=(1, 1),
        policy=policy,
        sm_count=HIGH_SM,
        sm_version="sm_120",
        load_path="tma",
        swap_ab=False,
        sfb_k_reuse=sfb_k_reuse,
        b_tile_major=b_tile_major,
    )


def _compile_key_differences(
    lhs: _DenseGemmLaunch, rhs: _DenseGemmLaunch
) -> list[tuple[object, object]]:
    return [
        (lhs_value, rhs_value)
        for lhs_value, rhs_value in zip(
            lhs.compile_key(), rhs.compile_key(), strict=True
        )
        if lhs_value != rhs_value
    ]


def test_expected_m_decode_regime_selects_32x128():
    # expected_m in the small-batch regime (9..128) -> 32x128 (probe optimum,
    # ~25% faster than 64x128 at M=32..128).
    for em in (16, 32, 64, 128):
        assert _tile(64, expected_m=em) == (32, 128), em


def test_expected_m_tiny_m_decode_selects_probe_tiles():
    # Exact single-token decode uses the flushed common-shape winner (16x64).
    # The broader tiny-M regime keeps the prior 16x128 specialization.
    assert _tile(64, expected_m=1) == (16, 64)
    for em in (2, 4, 8):
        assert _tile(64, expected_m=em) == (16, 128), em


def test_expected_m_prefill_regime_selects_64x128():
    for em in (129, 256, 512, 2048, 4096):
        assert _tile(64, expected_m=em) == (64, 128), em


def test_expected_m_is_independent_of_live_m():
    # The whole point: the tile is a function of (N,K,expected_m), NOT live M, so
    # one warmed kernel serves every live M in the regime. For a fixed
    # expected_m, the selected tile must be identical across wildly different
    # live M (16, 512, 4096).
    for em, want in ((64, (32, 128)), (1, (16, 64)), (2048, (64, 128))):
        tiles = {_tile(live_m, expected_m=em) for live_m in (1, 16, 128, 512, 4096)}
        assert tiles == {want}, (em, tiles)


def test_dense_compile_key_separates_replicated_sfb_reuse():
    generic_scales = _bk64_launch(sfb_k_reuse=False)
    replicated_scales = _bk64_launch()

    assert _compile_key_differences(generic_scales, replicated_scales) == [
        (False, True)
    ]


def test_dense_compile_key_covers_atom_shape_environment(monkeypatch):
    monkeypatch.setattr(dense_module, "_B12X_DENSE_ATOM_24", False)
    atom_42 = _bk64_launch()
    monkeypatch.setattr(dense_module, "_B12X_DENSE_ATOM_24", True)
    atom_24 = _bk64_launch()

    assert _compile_key_differences(atom_42, atom_24) == [(False, True)]


def test_fused_quant_compile_key_is_distinct_and_exhaustive():
    ordinary = _bk64_launch()
    fused = _bk64_launch(launch_type=_DenseGemmFusedQuantALaunch)

    assert fused.compile_key()[0] == "fused_quant_a"
    # Indexes 1-2 are the fused-only A inner-span and wide-M1 layout fields.
    assert fused.compile_key()[1] == 0
    assert fused.compile_key()[2] is False
    assert fused.compile_key()[3:] == ordinary.compile_key()


def test_high_sm_short_k_large_n_uses_bk64_plan():
    for em in (2048, 4096, 8192):
        for live_m in (1, 64, 4096):
            assert _tile(
                live_m,
                expected_m=em,
                n=16384,
                k=1024,
            ) == (128, 128)
            assert _select_mxfp8_tile_k(live_m, 16384, 1024, em, HIGH_SM) == 64
        policy = _dense_gemm_policy_for(
            m=64,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(128, 128),
            cluster_shape_mn=(1, 1),
            sm_count=HIGH_SM,
            expected_m=em,
        )
        assert policy.large_m_unroll == (em >= 8192)


def test_low_sm_short_k_large_n_keeps_bk128_plan():
    for em in (2048, 4096, 8192):
        for live_m in (1, 64, 4096):
            assert _tile(
                live_m,
                expected_m=em,
                n=16384,
                k=1024,
                sm_count=LOW_SM,
            ) == (64, 128)
            assert _select_mxfp8_tile_k(live_m, 16384, 1024, em, LOW_SM) == 128
        policy = _dense_gemm_policy_for(
            m=64,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(64, 128),
            cluster_shape_mn=(1, 1),
            sm_count=LOW_SM,
            expected_m=em,
        )
        assert policy.large_m_unroll == (em >= 4096)


def test_short_k_two_cta_occupancy_is_low_sm_only():
    kwargs = dict(
        n=16384,
        k=1024,
        l=1,
        ab_dtype=cutlass.Float8E4M3FN,
        c_dtype=cutlass.BFloat16,
        tile_k=128,
        mma_tiler_mn=(16, 128),
        cluster_shape_mn=(1, 1),
        load_path="tma",
        swap_ab=False,
        b_tile_major=False,
    )
    assert _dense_gemm_target_occupancy(sm_count=LOW_SM, **kwargs) == 2
    assert _dense_gemm_target_occupancy(sm_count=HIGH_SM, **kwargs) == 1
    assert _use_low_sm_dense_tactics(LOW_SM)
    assert not _use_low_sm_dense_tactics(HIGH_SM)


def test_wo_b_prefill_switches_to_bm128_bk64_at_2k():
    # Match the specialized DeepGEMM O-projection schedule without changing
    # the 1K schedule that already wins end to end.
    n, k = 4096, 4096
    assert _tile(1024, expected_m=1024, n=n, k=k) == (64, 128)
    assert _select_mxfp8_tile_k(1024, n, k, 1024, HIGH_SM) == 128
    for em in (2048, 4096, 8192):
        tiles = {
            _tile(live_m, expected_m=em, n=n, k=k) for live_m in (1, 64, 2048, 8192)
        }
        assert tiles == {(128, 128)}, (n, k, em, tiles)
        assert _select_mxfp8_tile_k(1, n, k, em, HIGH_SM) == 64


def test_grouped_wo_a_prefill_keeps_bm64_bk128():
    # The DeepGEMM schedule loses on b12x's grouped WO-A kernel at every probed
    # prefill size, so keep the established narrow-N schedule.
    n, k = 1024, 512
    for em in (2048, 4096, 8192):
        assert _tile(1, expected_m=em, n=n, k=k) == (64, 128)
        assert _select_mxfp8_tile_k(1, n, k, em, HIGH_SM) == 128


def test_wo_bk64_override_is_exact_shape_only():
    for n, k in ((1024, 640), (1152, 512), (4096, 3968), (4224, 4096)):
        assert _select_mxfp8_tile_k(2048, n, k, 2048, HIGH_SM) == 128


def test_short_k_1024_and_2048_hints_have_stable_distinct_keys():
    def specialization(live_m: int, expected_m: int | None):
        tile = _tile(live_m, expected_m=expected_m, n=16384, k=1024)
        tile_k = _select_mxfp8_tile_k(live_m, 16384, 1024, expected_m, HIGH_SM)
        policy = _dense_gemm_policy_for(
            m=live_m,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=(1, 1),
            sm_count=HIGH_SM,
            expected_m=expected_m,
        )
        return tile, tile_k, policy

    # Persistent live shapes share one specialization for a fixed hint.
    hint_1024 = {specialization(m, 1024) for m in (16, 1024, 2048, 8192)}
    hint_2048 = {specialization(m, 2048) for m in (16, 1024, 2048, 8192)}
    assert len(hint_1024) == 1
    assert len(hint_2048) == 1
    assert next(iter(hint_1024))[:2] == ((64, 128), 128)
    assert next(iter(hint_2048))[:2] == ((128, 128), 64)
    assert hint_1024 != hint_2048


def test_short_k_no_hint_does_not_cross_bk64_cache_boundary():
    # The no-hint API promises one prefill kernel across live M. BK64 is only
    # selected by an explicit regime hint, so crossing live M=2048 cannot cause
    # a new tile, tile-K, or policy key under frozen resolution.
    specializations = set()
    for live_m in (16, 1024, 2048, 4096, 8192):
        tile = _tile(live_m, expected_m=None, n=16384, k=1024)
        tile_k = _select_mxfp8_tile_k(live_m, 16384, 1024, None, HIGH_SM)
        policy = _dense_gemm_policy_for(
            m=live_m,
            n=16384,
            k=1024,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=tile,
            cluster_shape_mn=(1, 1),
            sm_count=HIGH_SM,
            expected_m=None,
        )
        specializations.add((tile, tile_k, policy))
    assert specializations == {
        (
            (64, 128),
            128,
            _DenseGemmPolicy(False, False, False, 1, True, True),
        )
    }


def test_bk64_rejects_unvalidated_short_row_and_swapped_tiles():
    _validate_mxfp8_bk64_plan(64, (128, 128), False)
    _validate_mxfp8_bk64_plan(64, (128, 64), False)
    _validate_mxfp8_bk64_plan(128, (64, 128), False)
    with pytest.raises(ValueError, match="requires an unswapped 128-row tile"):
        _validate_mxfp8_bk64_plan(64, (64, 128), False)
    with pytest.raises(ValueError, match="requires an unswapped 128-row tile"):
        _validate_mxfp8_bk64_plan(64, (128, 64), True)


def test_no_hint_persistent_policy_keeps_unroll_m_independent():
    # Public expected_m=None warms one persistent-policy kernel and reuses it
    # across live prefill sizes. In particular, crossing M=4096 must not change
    # the compile key solely to toggle mainloop unrolling.
    policies = {
        _dense_gemm_policy_for(
            m=live_m,
            n=1536,
            k=128,
            l=1,
            ab_dtype=cutlass.Float8E4M3FN,
            c_dtype=cutlass.BFloat16,
            mma_tiler_mn=(64, 64),
            cluster_shape_mn=(1, 1),
            sm_count=HIGH_SM,
            expected_m=None,
        )
        for live_m in (16, 512, 1824, 4096, 8192)
    }
    assert len(policies) == 1
    assert next(iter(policies)).large_m_unroll


def test_no_hint_preserves_graft_a_default():
    # expected_m=None preserves the M-independent Graft A behavior outside the
    # tiny standalone decode range: m=1 -> 16x64, m=2..8 -> 16x128,
    # and m>=16 -> 64x128.
    assert _tile(1, expected_m=None) == (16, 64)
    for m in (2, 4, 8):
        assert _tile(m, expected_m=None) == (16, 128), m
    for m in (16, 32, 64, 128, 256, 4096):
        assert _tile(m, expected_m=None) == (64, 128), m


def test_expected_m_prefill_hint_for_narrow_n():
    # Narrow-N has its own occupancy heuristic. Exact M=1 uses the common-shape
    # decode winner, while declared prefill still moves to the prefill tile.
    narrow = 1024
    base = _select_default_mma_tiler_mn(64, narrow, HIGH_SM, is_mxfp8=True)
    decode = _select_default_mma_tiler_mn(64, narrow, HIGH_SM, is_mxfp8=True, expected_m=1)
    small = _select_default_mma_tiler_mn(64, narrow, HIGH_SM, is_mxfp8=True, expected_m=64)
    prefill = _select_default_mma_tiler_mn(
        64, narrow, HIGH_SM, is_mxfp8=True, expected_m=512
    )
    no_hint_decode = _select_default_mma_tiler_mn(1, narrow, HIGH_SM, is_mxfp8=True)
    assert base == (64, 64)
    assert decode == (16, 64)
    assert small == base
    assert prefill == (64, 128)
    assert no_hint_decode == (16, 64)
