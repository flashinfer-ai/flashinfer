import pytest

from flashinfer.gemm.kernels import utils as gemm_utils
from flashinfer.gemm.kernels.utils import (
    _MM_FP4_SWAP_PENALTY_MAX_M,
    _M_BUCKETS,
    _rank_mm_fp4_autotune_tactics,
    _select_sm107_mm_fp4_cute_dsl_tactic,
    _score_mm_fp4_autotune_tactic,
    _score_sm100_mm_fp4_tactic,
)


M = 8192
N = 4096
K = 8192
SM_COUNT = 212
TILE = (256, 256)
CLUSTER = (2, 1)


@pytest.mark.parametrize("kernel_type", ["sm100", "sm107"])
def test_autotune_does_not_penalize_swap_ab_above_large_m_boundary(
    kernel_type,
) -> None:
    m = _MM_FP4_SWAP_PENALTY_MAX_M * 2
    tactics = [
        (TILE, CLUSTER, swap, False, kernel_type, None)
        for swap in (False, True)
    ]

    selected = _rank_mm_fp4_autotune_tactics(
        tactics, m, N, K, SM_COUNT, max_tactics=2
    )

    # Equal analytical scores preserve enumeration order, leaving the measured
    # autotuner to decide between orientations on both kernel families.
    assert selected == tactics


def test_fallback_does_not_penalize_either_orientation_above_large_m_boundary(
) -> None:
    m = _MM_FP4_SWAP_PENALTY_MAX_M * 2
    swap0 = _score_sm100_mm_fp4_tactic(m, N, K, SM_COUNT, TILE, CLUSTER, False)
    swap1 = _score_sm100_mm_fp4_tactic(m, N, K, SM_COUNT, TILE, CLUSTER, True)

    assert swap0 == pytest.approx(swap1)
    assert _score_mm_fp4_autotune_tactic(
        m, N, K, SM_COUNT, TILE, CLUSTER, False
    ) == pytest.approx(swap1)


def test_swap_penalty_is_retained_at_large_m_boundary() -> None:
    m = _MM_FP4_SWAP_PENALTY_MAX_M
    swap0 = _score_sm100_mm_fp4_tactic(m, N, K, SM_COUNT, TILE, CLUSTER, False)
    swap1 = _score_sm100_mm_fp4_tactic(m, N, K, SM_COUNT, TILE, CLUSTER, True)

    assert swap1 == pytest.approx(swap0 * 0.95)


def test_autotune_budget_counts_actual_sm107_tactics() -> None:
    tactics = [
        (
            TILE,
            CLUSTER,
            False,
            False,
            "sm107",
            (128, 256, 128, 256, prefetch_dist),
        )
        for prefetch_dist in range(40)
    ]

    selected = _rank_mm_fp4_autotune_tactics(tactics, M, N, K, SM_COUNT, max_tactics=32)

    assert len(selected) == 32
    assert selected == tactics[:32]


def test_equal_sm107_scores_preserve_enumeration_order() -> None:
    tactic_swap0 = (TILE, CLUSTER, False, False, "sm107", (128, 256, 128, 256, 0))
    tactic_swap1 = (TILE, CLUSTER, True, False, "sm107", (128, 256, 128, 256, 0))
    m = _MM_FP4_SWAP_PENALTY_MAX_M * 2

    selected = _rank_mm_fp4_autotune_tactics(
        [tactic_swap1, tactic_swap0], m, N, K, SM_COUNT, max_tactics=2
    )

    # Equal analytical scores preserve enumeration order, allowing both
    # orientations to reach measured autotuning.
    assert selected == [tactic_swap1, tactic_swap0]


def test_sm107_fallback_adds_large_m_bucket_lazily(monkeypatch) -> None:
    calls = []

    def fake_compute(rep_m, n, real_k, sm_count, sf_vec_size):
        calls.append(rep_m)
        return ("tactic", rep_m)

    monkeypatch.setattr(
        "flashinfer.gemm.kernels.utils._compute_sm107_tactic_for_m", fake_compute
    )
    monkeypatch.setattr(gemm_utils, "_SM107_MM_FP4_TACTIC_CACHE", {})

    tactic = _select_sm107_mm_fp4_cute_dsl_tactic(
        _MM_FP4_SWAP_PENALTY_MAX_M + 1, N, K, SM_COUNT, 16
    )

    assert tactic == ("tactic", 16384)
    assert calls == [*_M_BUCKETS, 16384]
