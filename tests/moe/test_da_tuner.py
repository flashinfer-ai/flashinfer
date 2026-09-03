"""Host-only tests for distribution-aware MoE tuning policy."""

import torch

from flashinfer.fused_moe.da_tuner import (
    DAPlanCompiler,
    DAProfileSelection,
    FullOpMeasurementCache,
    RoutingRealizationKey,
)
from flashinfer.fused_moe.tactic_search import FactorizedTactic


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
