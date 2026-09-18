"""Host-only tests for distribution-aware MoE tuning policy."""

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from flashinfer.autotuner import autotune
from flashinfer.fused_moe.da_moe import DAPlanMode, _local_load_spectrum
from flashinfer.fused_moe.da_runtime import run_dist_aware_tactic
from flashinfer.fused_moe.da_tuner import (
    get_workload,
    moe_workload,
    BalancedEPWorkload,
    FullWorkload,
    DAPlanCompiler,
    DAProfileSelection,
    FullOpMeasurementCache,
    RoutingRealizationFactory,
    RoutingRealizationKey,
)
from flashinfer.fused_moe.tactic_search import FactorizedTactic
from flashinfer.fused_moe.shared.inputs import MoeRunnerInputs, RoutingInputMode


def test_workload_hints_resolve_against_moe_geometry() -> None:
    geometry = {
        "num_experts": 256,
        "num_local_experts": 32,
        "local_expert_offset": 96,
    }

    assert (
        BalancedEPWorkload().local_assignments(capacity_tokens=32, top_k=8, **geometry)
        == 64
    )
    assert (
        BalancedEPWorkload(ep_size=8, ep_rank=3).local_assignments(
            capacity_tokens=1024, top_k=8, **geometry
        )
        == 2048
    )
    assert (
        BalancedEPWorkload(
            ep_size=8, ep_rank=3, assignment_multiplier=1
        ).local_assignments(capacity_tokens=1024, top_k=8, **geometry)
        == 1024
    )
    assert (
        FullWorkload().local_assignments(capacity_tokens=32, top_k=8, **geometry) == 256
    )
    assert (
        BalancedEPWorkload().local_assignments(
            capacity_tokens=32,
            top_k=8,
            num_experts=32,
            num_local_experts=32,
            local_expert_offset=0,
        )
        == 256
    )
    with pytest.raises(ValueError, match="assignment_multiplier must be positive"):
        BalancedEPWorkload(assignment_multiplier=0)


def test_autotune_context_workload_is_nested_and_thread_local() -> None:
    assert get_workload() == BalancedEPWorkload()

    balanced = BalancedEPWorkload(ep_size=8, ep_rank=3)
    with moe_workload(balanced):
        assert get_workload() == balanced
        with autotune(False):
            assert get_workload() == balanced
        with moe_workload(FullWorkload()):
            assert get_workload() == FullWorkload()
        assert get_workload() == balanced

    assert get_workload() == BalancedEPWorkload()


def test_workload_context_restores_after_exception() -> None:
    outer = BalancedEPWorkload(assignment_multiplier=3)
    with moe_workload(outer):
        with (
            pytest.raises(RuntimeError, match="body failed"),
            moe_workload(FullWorkload()),
        ):
            raise RuntimeError("body failed")
        assert get_workload() is outer
    assert get_workload() == BalancedEPWorkload()


def test_workload_context_is_isolated_between_threads() -> None:
    barrier = Barrier(2)
    workloads = (FullWorkload(), BalancedEPWorkload(assignment_multiplier=1))

    def assignments(workload):
        with moe_workload(workload):
            barrier.wait(timeout=10)
            result = get_workload().local_assignments(32, 8, 256, 32, 96)
            barrier.wait(timeout=10)
        assert get_workload() == BalancedEPWorkload()
        return result

    with ThreadPoolExecutor(max_workers=2) as executor:
        assert list(executor.map(assignments, workloads)) == [256, 32]


@pytest.mark.parametrize("offset", [-32, 225, 256])
def test_balanced_workload_rejects_out_of_range_shards(offset) -> None:
    with pytest.raises(ValueError, match="local expert range"):
        BalancedEPWorkload().local_assignments(32, 8, 256, 32, offset)


def test_balanced_workload_accepts_unaligned_range_when_work_divides_evenly() -> None:
    assert BalancedEPWorkload().local_assignments(64, 32, 1024, 64, 480) == 256


def test_unaligned_remainder_requires_explicit_ep_rank() -> None:
    with pytest.raises(ValueError, match="requires explicit ep_rank"):
        BalancedEPWorkload(
            require_equal=False, assignment_multiplier=1
        ).local_assignments(1, 2, 6, 2, 1)

    assert (
        BalancedEPWorkload(
            ep_size=3,
            ep_rank=1,
            require_equal=False,
            assignment_multiplier=1,
        ).local_assignments(1, 2, 6, 2, 1)
        == 1
    )


def test_balanced_workload_requires_equal_division_by_default() -> None:
    with pytest.raises(ValueError, match="must be divisible"):
        BalancedEPWorkload(assignment_multiplier=1).local_assignments(3, 2, 4, 1)

    targets = [
        BalancedEPWorkload(
            require_equal=False, assignment_multiplier=1
        ).local_assignments(3, 2, 4, 1, rank)
        for rank in range(4)
    ]
    assert targets == [2, 2, 1, 1]
    assert sum(targets) == 6


@pytest.mark.parametrize("num_local_assignments_hint", [None, 0, 4])
@pytest.mark.parametrize("distribution", ["uniform", "ddist:2"])
def test_routing_realization_matches_local_assignment_hint(
    num_local_assignments_hint: int | None,
    distribution: str,
) -> None:
    key = RoutingRealizationKey(
        device=torch.device("cpu"),
        num_tokens=8,
        distribution=distribution,
        sample_index=0,
        local_expert_offset=4,
        num_experts=16,
        num_local_experts=2,
        top_k=4,
        num_local_assignments_hint=num_local_assignments_hint,
        routing_rule_fingerprint="test",
        routed_scaling_factor=1.0,
    )

    # Replay defaults stay balanced-1x even under a full-local DA tuning policy.
    with moe_workload(FullWorkload()):
        realization = RoutingRealizationFactory().get_or_create(key)
    ids = realization.expert_ids
    local = (ids >= 4) & (ids < 6)

    expected = 4 if num_local_assignments_hint is None else num_local_assignments_hint
    assert int(local.sum()) == expected
    assert all(row.unique().numel() == 4 for row in ids)
    assert torch.all((ids >= 0) & (ids < 16))
    assert torch.allclose(
        realization.routing_weights.float().sum(dim=1),
        torch.ones(8),
        atol=0.01,
    )


def test_da_runtime_falls_back_when_topk_exceeds_local_shard() -> None:
    """Legacy FullWorkload keeps the ordinary fallback for an undersized shard."""
    baseline_tactic = (16, 7)
    calls: list[object] = []

    with moe_workload(FullWorkload()):
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


@pytest.mark.parametrize("backend", ["prims_ts", "trtllm"])
@pytest.mark.parametrize(
    "workload,expected",
    [
        (BalancedEPWorkload(), 64),
        (BalancedEPWorkload(assignment_multiplier=1), 32),
        (FullWorkload(), 256),
    ],
)
def test_da_runtime_derives_assignment_hint_from_context(
    monkeypatch, backend, workload, expected
):
    """The shared runtime resolves workload metadata before partitioning its plan cache."""
    from flashinfer.fused_moe import da_runtime

    class KeyReached(Exception):
        pass

    captured = {}

    def capture_key(*args, **kwargs):
        captured.update(kwargs)
        raise KeyReached

    monkeypatch.setattr(da_runtime, "make_da_moe_operation_key", capture_key)
    inputs: list[Any] = [None] * len(MoeRunnerInputs._FIELDS)
    inputs[MoeRunnerInputs.idx("hidden_states")] = torch.empty(32, 8)
    with moe_workload(workload), pytest.raises(KeyReached):
        run_dist_aware_tactic(
            backend=backend,
            custom_op="test::moe",
            tuner=cast(Any, None),
            config=cast(Any, None),
            runner=cast(Any, None),
            runtime=SimpleNamespace(backend=backend),
            tuning_config=cast(Any, None),
            inputs=inputs,
            runner_kwargs={},
            baseline_tactic=(16, 7),
            routing_input_mode=RoutingInputMode.UnpackedPrecomputed,
            routing_id_index=MoeRunnerInputs.idx("topk_ids"),
            routing_weight_index=MoeRunnerInputs.idx("expert_weights"),
            num_experts=256,
            local_expert_offset=96,
            num_local_experts=32,
            top_k=8,
            routing_method_type=0,
            routed_scaling_factor=1.0,
            run_fixed_tactic=lambda tactic: pytest.fail("unexpected fallback"),
            finish_switch=lambda: pytest.fail("unexpected capture"),
        )
    assert captured["num_local_assignments_hint"] == expected


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
            num_experts=4,
            num_local_experts=4,
            top_k=2,
            num_local_assignments_hint=2,
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


@pytest.mark.parametrize(
    "candidate_times,baseline_times,margin,admitted",
    [
        ((0.150, 0.180), (0.100, 0.200), 0.0, False),
        ((0.090, 0.150), (0.200, 0.100), 0.0, False),
        ((0.100, 0.200), (0.100, 0.200), 0.0, True),
        ((0.750, 1.500), (1.000, 2.000), 0.25, True),
        ((0.751, 1.400), (1.000, 2.000), 0.25, False),
    ],
)
def test_singleton_guard_requires_each_matched_exemplar(
    candidate_times, baseline_times, margin, admitted
):
    compiler = DAPlanCompiler(num_experts=4, margin=margin)
    tactic = FactorizedTactic((16, 4), tile_n=16, fc1=0, fc2=0)
    selections = tuple(
        replace(
            selection,
            candidate_latency_ms=candidate,
            baseline_latency_ms=baseline,
        )
        for selection, candidate, baseline in zip(
            (
                _selection("uniform", [[0, 1]], tactic),
                _selection("ddist:1.1", [[0, 0]], tactic),
            ),
            candidate_times,
            baseline_times,
            strict=True,
        )
    )

    compiled = compiler.compile(selections, baseline_tactic=(64, 0))

    assert compiled.candidate_policy is DAPlanMode.DA_SINGLE_BODY
    assert compiled.policy is (
        DAPlanMode.DA_SINGLE_BODY if admitted else DAPlanMode.DA_FALLBACK
    )
    assert compiled.guard_reason == (
        "admitted" if admitted else "singleton_guard_rejected"
    )


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
