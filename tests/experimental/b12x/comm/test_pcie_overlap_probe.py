from __future__ import annotations

import pytest

from b12x.comm.pcie.overlap_probe import (
    BackendDecision,
    ProbeConfig,
    decide_query_split,
    decide_overlap,
    decide_tp_backend,
    dma_crossover_bytes,
    maximum_contiguous_enabled_context,
    query_split_crossover_tokens,
    recommend_prefetch_depth,
    summarize,
)


def test_glm52_payload_geometry() -> None:
    config = ProbeConfig()

    assert config.tp_payload_bytes == 100_663_296
    assert config.ckv_local_tokens(65_536) == 16_384
    assert config.ckv_local_bytes(65_536) == 10_747_904
    assert config.ckv_wire_bytes_per_rank(65_536) == 32_243_712
    assert config.query_split_size == 4
    assert config.query_split_rows == 2048
    assert config.indexer_local_topk(8192) == 1024
    assert config.indexer_local_topk(65_536) == 2048
    assert config.indexer_local_tokens(65_536) == 8192
    assert config.baseline_candidate_wire_bytes_per_rank(65_536) == 134_217_728
    assert config.owner_candidate_wire_bytes_per_rank(65_536) == 16_777_216
    assert config.query_split_final_wire_bytes_per_rank() == 58_720_256
    assert config.query_split_wire_bytes_per_rank(65_536) == 75_497_472
    assert config.baseline_candidate_wire_bytes_per_rank(8192) == 67_108_864
    assert config.owner_candidate_wire_bytes_per_rank(8192) == 8_388_608
    assert config.query_split_wire_bytes_per_rank(8192) == 67_108_864


def test_config_rejects_non_divisible_dcp() -> None:
    with pytest.raises(ValueError, match="dcp_size must divide"):
        ProbeConfig(tp_size=8, dcp_size=3).validate()


def test_config_accepts_dcp1_for_tp_and_query_calibration() -> None:
    config = ProbeConfig(tp_size=8, dcp_size=1, indexer_shards=1)

    config.validate()
    assert config.ckv_wire_bytes_per_rank(65_536) == 0
    assert config.query_split_size == 8


def test_config_rejects_automatic_lossy_dma() -> None:
    with pytest.raises(ValueError, match="lossless BF16 DMA"):
        ProbeConfig(dma_wire_mode="ring").validate()


def test_config_rejects_invalid_owner_merge_geometry() -> None:
    with pytest.raises(ValueError, match="tp_size for owner merge"):
        ProbeConfig(tp_rows=12, allreduce_rows=(1,)).validate()


def test_config_rejects_too_few_global_index_candidates() -> None:
    with pytest.raises(ValueError, match="global candidates"):
        ProbeConfig(context_tokens=(4096,)).validate()


def test_summarize_uses_median_and_extrema() -> None:
    summary = summarize([5.0, 1.0, 3.0])

    assert summary.median_ms == 3.0
    assert summary.minimum_ms == 1.0
    assert summary.maximum_ms == 5.0


def test_decision_accepts_useful_overlap() -> None:
    decision = decide_overlap(
        tp_isolated_ms=10.0,
        ckv_isolated_ms=8.0,
        concurrent_tp_ms=11.0,
        concurrent_ckv_ms=9.0,
        concurrent_wall_ms=11.5,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )

    assert decision.enable_overlap
    assert decision.reason == "measured-overlap-benefit"
    assert decision.overlap_gain == pytest.approx((18.0 - 11.5) / 18.0)


def test_decision_rejects_serialization_loss() -> None:
    decision = decide_overlap(
        tp_isolated_ms=10.0,
        ckv_isolated_ms=8.0,
        concurrent_tp_ms=17.0,
        concurrent_ckv_ms=22.0,
        concurrent_wall_ms=22.5,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )

    assert not decision.enable_overlap
    assert decision.reason == "insufficient-overlap-gain"


def test_decision_rejects_pathological_collective_slowdown() -> None:
    decision = decide_overlap(
        tp_isolated_ms=100.0,
        ckv_isolated_ms=1.0,
        concurrent_tp_ms=90.0,
        concurrent_ckv_ms=5.0,
        concurrent_wall_ms=92.0,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )

    assert not decision.enable_overlap
    assert decision.reason == "collective-contention"


def test_prefetch_policy_accepts_small_context_noise_but_rejects_collapse() -> None:
    useful = decide_overlap(
        tp_isolated_ms=10.0,
        ckv_isolated_ms=8.0,
        concurrent_tp_ms=10.2,
        concurrent_ckv_ms=8.2,
        concurrent_wall_ms=17.5,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )
    neutral = decide_overlap(
        tp_isolated_ms=10.0,
        ckv_isolated_ms=8.0,
        concurrent_tp_ms=10.1,
        concurrent_ckv_ms=8.1,
        concurrent_wall_ms=17.9,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )
    collapsed = decide_overlap(
        tp_isolated_ms=10.0,
        ckv_isolated_ms=8.0,
        concurrent_tp_ms=18.0,
        concurrent_ckv_ms=18.5,
        concurrent_wall_ms=19.0,
        ckv_wire_bytes_per_rank=32_000_000,
        minimum_overlap_gain=0.02,
        maximum_collective_slowdown=4.0,
    )

    assert (
        recommend_prefetch_depth([useful, neutral], maximum_overlap_regression=0.01)
        == 1
    )
    assert (
        recommend_prefetch_depth([useful, collapsed], maximum_overlap_regression=0.01)
        == 0
    )


def test_maximum_safe_context_requires_contiguous_enabled_prefix() -> None:
    assert (
        maximum_contiguous_enabled_context(
            [(131_072, True), (8192, True), (65_536, False)]
        )
        == 8192
    )
    assert maximum_contiguous_enabled_context([(8192, False), (65_536, True)]) == 0


def test_tp_backend_requires_material_dma_gain() -> None:
    win = decide_tp_backend(
        nccl_ms=10.0,
        dma_ms=8.0,
        minimum_backend_gain=0.02,
    )
    noise = decide_tp_backend(
        nccl_ms=10.0,
        dma_ms=9.9,
        minimum_backend_gain=0.02,
    )

    assert win.backend == "b12x-dma"
    assert win.dma_gain == pytest.approx(0.2)
    assert noise.backend == "nccl"


def _backend(backend: str) -> BackendDecision:
    return BackendDecision(
        backend=backend,
        reason="test",
        nccl_ms=1.0,
        dma_ms=1.0,
        dma_gain=0.0,
    )


def test_dma_crossover_requires_winning_tail() -> None:
    points = [
        (1024, _backend("nccl")),
        (2048, _backend("b12x-dma")),
        (4096, _backend("b12x-dma")),
        (8192, _backend("b12x-dma")),
    ]
    late_loss = [*points, (16384, _backend("nccl"))]

    assert dma_crossover_bytes(points) == 2048
    assert dma_crossover_bytes(late_loss) == 0


def test_query_split_decision_uses_full_phase() -> None:
    win = decide_query_split(
        full_ms=12.0,
        split_ms=7.0,
        wire_bytes_per_rank=50_000_000,
        minimum_query_split_gain=0.02,
    )
    loss = decide_query_split(
        full_ms=12.0,
        split_ms=12.1,
        wire_bytes_per_rank=50_000_000,
        minimum_query_split_gain=0.02,
    )

    assert win.enable_query_split
    assert win.reason == "measured-full-phase-benefit"
    assert not loss.enable_query_split


def test_query_split_crossover_requires_winning_tail() -> None:
    loss = decide_query_split(
        full_ms=10.0,
        split_ms=11.0,
        wire_bytes_per_rank=1,
        minimum_query_split_gain=0.02,
    )
    win = decide_query_split(
        full_ms=10.0,
        split_ms=8.0,
        wire_bytes_per_rank=1,
        minimum_query_split_gain=0.02,
    )

    assert (
        query_split_crossover_tokens([(8192, loss), (65536, win), (131072, win)])
        == 65536
    )
    assert (
        query_split_crossover_tokens([(8192, win), (65536, loss), (131072, win)]) == 0
    )
