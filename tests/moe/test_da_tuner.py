"""Host-only tests for distribution-aware MoE tuning policy."""

import math
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from flashinfer.fused_moe.da_moe import _local_load_spectrum
from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic
from flashinfer.fused_moe.da_tuner import (
    DAPlanCompiler,
    DAProfileSelection,
    FullOpMeasurementCache,
    RoutingRealizationKey,
)
from flashinfer.fused_moe.tactic_search import FactorizedTactic


def test_da_runtime_falls_back_when_topk_exceeds_local_shard() -> None:
    """A valid EP shape outside the synthetic workload model stays on NoDA."""
    baseline_tactic = (16, 7)
    calls: list[object] = []

    result = run_dist_aware_tactic(
        backend="prims_ts",
        custom_op="test::moe",
        tuner=cast(Any, None),
        config=cast(Any, None),
        runner=cast(Any, None),
        runtime=SimpleNamespace(backend="prims_ts"),
        tuning_config=cast(Any, None),
        inputs=[],
        runner_kwargs={},
        baseline_tactic=baseline_tactic,
        routing_input_mode=0,
        routing_id_index=0,
        routing_weight_index=1,
        num_experts=1024,
        local_expert_offset=0,
        num_local_experts=16,
        top_k=32,
        routing_method_type=0,
        routed_scaling_factor=1.0,
        run_fixed_tactic=lambda tactic: calls.append(tactic) or "ordinary",
        finish_switch=lambda: pytest.fail("ineligible DA shape entered capture"),
    )

    assert result == "ordinary"
    assert calls == [baseline_tactic]


def _selection(
    distribution: str,
    expert_ids: list[list[int]],
    tactic: FactorizedTactic,
) -> DAProfileSelection:
    return DAProfileSelection(
        realization_key=RoutingRealizationKey(
            device=torch.device("cpu"),
            num_tokens=1,
            distribution=distribution,
            sample_index=0,
            local_expert_offset=0,
            num_local_experts=4,
            top_k=2,
            routing_rule_fingerprint="test",
            routed_scaling_factor=1.0,
        ),
        expert_ids=torch.tensor(expert_ids, dtype=torch.int32),
        selected_tactic=tactic,
        candidate_latency_ms=0.9,
        baseline_latency_ms=1.0,
    )


def test_full_op_measurement_pair_uses_abba_order_and_reuses_cache():
    measurements = FullOpMeasurementCache()
    order = []
    first_values = iter((1.0, 3.0))
    second_values = iter((2.0, 4.0))

    def measure_first():
        order.append("a")
        return next(first_values)

    def measure_second():
        order.append("b")
        return next(second_values)

    result = measurements.measure_counterbalanced_pair(
        ("shape", "da"),
        measure_first,
        ("shape", "noda"),
        measure_second,
    )

    assert order == ["a", "b", "b", "a"]
    assert result == (2.0, 3.0)
    assert measurements.count == 2
    assert measurements.measure_counterbalanced_pair(
        ("shape", "da"),
        measure_first,
        ("shape", "noda"),
        measure_second,
    ) == (2.0, 3.0)
    assert order == ["a", "b", "b", "a"]


def test_full_op_measurement_pair_does_not_publish_partial_results():
    measurements = FullOpMeasurementCache()
    first_values = iter((1.0, 3.0, 5.0, 7.0))
    second_values = iter((2.0, math.inf, 6.0, 8.0))

    with pytest.raises(RuntimeError, match="Non-finite full MoE timing"):
        measurements.measure_counterbalanced_pair(
            ("shape", "da"),
            lambda: next(first_values),
            ("shape", "noda"),
            lambda: next(second_values),
        )

    assert measurements.count == 0
    assert measurements.measure_counterbalanced_pair(
        ("shape", "da"),
        lambda: next(first_values),
        ("shape", "noda"),
        lambda: next(second_values),
    ) == (6.0, 7.0)
    assert measurements.count == 2


def test_singleton_pruning_preserves_preferred_distribution_eager_winner():
    compiler = DAPlanCompiler(num_experts=4, control_overhead_us=12.0)
    uniform_tactic = FactorizedTactic((16, 4), tile_n=16, fc1=0, fc2=0)
    preferred_tactic = FactorizedTactic((32, 11), tile_n=32, fc1=1, fc2=1)
    original = (
        _selection("uniform", [[0, 1]], uniform_tactic),
        _selection("ddist:1.1", [[0, 0]], preferred_tactic),
    )
    candidate_latencies = {
        (original[0].realization_key, uniform_tactic.tactic): 0.900,
        (original[0].realization_key, preferred_tactic.tactic): 0.905,
        (original[1].realization_key, uniform_tactic.tactic): 0.904,
        (original[1].realization_key, preferred_tactic.tactic): 0.900,
    }

    capture_selections = compiler.prefer_control_aware_singleton(
        original, candidate_latencies
    )
    compiled = compiler.compile(
        capture_selections,
        baseline_tactic=(64, 0),
        eager_selections=original,
    )

    assert compiled.selections == capture_selections
    assert compiled.bodies == (uniform_tactic,)
    assert compiled.eager_distribution == "ddist:1.1"
    assert compiled.eager_tactic == preferred_tactic


def test_nonlocal_assignment_changes_do_not_change_local_spectrum():
    first = torch.tensor([[4, 0], [4, 1], [5, 2], [6, 3]], dtype=torch.int32)
    second = torch.tensor([[4, 8], [4, 9], [5, 10], [6, 11]], dtype=torch.int32)

    spectra = [
        _local_load_spectrum(
            ids,
            num_experts=16,
            local_expert_offset=4,
            num_local_experts=4,
            normalize=True,
        )
        for ids in (first, second)
    ]

    assert torch.equal(*spectra)
