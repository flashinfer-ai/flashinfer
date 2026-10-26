# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency tuning must preserve replay semantics and measurement provenance."""

from dataclasses import replace
import math

import pytest
import torch

from flashinfer.autotuner import AutoTuner, TuningConfig, autotune
from flashinfer.autotuner._graph_timing import profile_individual_graph_calls
from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2

from .utils import DummyRunner, reset_autotuner


def test_individual_timing_provenance_and_overrides():
    tuner = reset_autotuner()
    sustained = TuningConfig(use_cuda_graph=True)
    individual = replace(sustained, use_cuda_graph_individual_timing=True)
    assert tuner._profiling_policy(sustained) == tuner._default_profiling_policy(
        sustained
    )
    assert tuner._profiling_policy(individual) != tuner._profiling_policy(sustained)
    # The winner remains consumable by ordinary lookup. Its provenance, rather
    # than tensor compatibility, determines whether tuning must measure again.
    keys = [
        tuner._get_cache_key("op", DummyRunner(), ((1,),), config).file_key
        for config in (sustained, individual)
    ]
    assert keys[0] == keys[1]
    with autotune(False, cuda_graph_profile_replays=3):
        overridden = tuner._apply_tuning_overrides(individual)
    assert overridden.use_cuda_graph_individual_timing
    assert overridden.cuda_graph_profile_replays == 3


@pytest.mark.parametrize("graph,cold", [(False, False), (False, True), (True, True)])
def test_inactive_individual_timing_keeps_provenance(graph, cold):
    tuner = reset_autotuner()
    config = TuningConfig(use_cuda_graph=graph, use_cold_l2_cache=cold)
    assert tuner._profiling_policy(config) == tuner._profiling_policy(
        replace(config, use_cuda_graph_individual_timing=True)
    )


def test_measurement_policy_can_enable_and_disable_individual_timing():
    tuner = reset_autotuner()
    config = TuningConfig(use_cuda_graph=True)
    individual = tuner._apply_measure_policy(
        config, MeasurementPolicy(graph_timing="individual")
    )
    assert individual.use_cuda_graph_individual_timing
    assert not config.use_cuda_graph_individual_timing
    assert individual is tuner._apply_measure_policy(
        config, MeasurementPolicy(graph_timing="individual")
    )
    assert not tuner._apply_measure_policy(
        individual, MeasurementPolicy(graph_timing="amortized")
    ).use_cuda_graph_individual_timing
    assert not tuner._apply_measure_policy(
        individual, MeasurementPolicy(_timer="cupti")
    ).use_cuda_graph_individual_timing


@pytest.mark.parametrize(
    "kwargs",
    [
        {"graph_timing": "unknown"},
        {"graph_timing": "individual", "execution_mode": "eager"},
        {"graph_timing": "individual", "_timer": "cupti"},
    ],
)
def test_measurement_policy_rejects_incompatible_graph_timing(kwargs):
    with pytest.raises(ValueError, match="graph_timing"):
        MeasurementPolicy(**kwargs)


@pytest.mark.parametrize(
    "policy,initial_graph,initial_individual,expected",
    [
        (MeasurementPolicy(graph_timing="individual"), True, False, (True, True)),
        (MeasurementPolicy(graph_timing="amortized"), True, True, (True, False)),
        (MeasurementPolicy(execution_mode="cuda_graph"), False, False, (True, False)),
        (MeasurementPolicy(execution_mode="eager"), True, False, (False, False)),
        (MeasurementPolicy(_timer="cupti"), True, True, (True, False)),
    ],
)
def test_explicit_tactic_profile_matches_selection_policy(
    monkeypatch, policy, initial_graph, initial_individual, expected
):
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    inputs = [torch.ones(1)]
    runner = DummyRunner(valid_tactics=(0,))
    config = TuningConfig(
        use_cuda_graph=initial_graph,
        use_cuda_graph_individual_timing=initial_individual,
    )
    measured = []

    def profile(runner, inputs, tactic, effective_config, **kwargs):
        measured.append(effective_config)
        return 1.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    with (
        autotune_v2(persistent_cache=False, measurement_policy=policy),
        autotune(cuda_graph_profile_replays=3),
    ):
        tuner.choose_one("explicit_profile_policy", [runner], config, inputs)
        ordinary = measured[-1]
        effective, batches = tuner.prepare_tactic_profile(inputs, config)
        assert tuner.profile_tactic(runner, inputs, 0, effective, batches) == 1.0

    assert measured[-1] is ordinary
    assert (effective.use_cuda_graph, effective.use_cuda_graph_individual_timing) == (
        expected
    )
    assert effective.cuda_graph_profile_replays == 3
    assert config.cuda_graph_profile_replays == 1
    assert config.use_cuda_graph_individual_timing == initial_individual


def test_explicit_profile_applies_hot_policy_before_arena_preparation(monkeypatch):
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    inputs = [torch.ones(1)]
    config = TuningConfig(use_cold_l2_cache=True, profile_arena_input_indices=(0,))
    # A hot policy must reuse the input without trying to create a CUDA arena.
    with autotune_v2(
        persistent_cache=False, measurement_policy=MeasurementPolicy(cold_l2=False)
    ):
        effective, batches = tuner.prepare_tactic_profile(inputs, config)
    assert not effective.use_cold_l2_cache
    assert batches == [inputs]
    assert config.use_cold_l2_cache


def test_individual_timing_preserves_globaltimer_selection(monkeypatch):
    tuner = reset_autotuner()
    monkeypatch.setattr(tuner, "warmup", 0)
    monkeypatch.setattr(tuner, "_use_global_timer", True)
    stamp = lambda tensor: None
    monkeypatch.setattr(tuner, "_record_global_timer", stamp, raising=False)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: None)
    seen = []

    def profile(run_once, **kwargs):
        seen.append(kwargs["record_global_timer"])
        return 1.0

    monkeypatch.setattr(
        "flashinfer.autotuner._graph_timing.profile_individual_graph_calls", profile
    )
    inputs = [torch.ones(1)]
    config = TuningConfig(use_cuda_graph=True, use_cuda_graph_individual_timing=True)
    assert tuner.profile_tactic(DummyRunner(), inputs, 0, config, [inputs]) == 1.0
    assert seen == [stamp]


def test_individual_timing_reprofiles_sustained_winner(monkeypatch):
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1))
    inputs = [torch.ones(1)]
    sustained = TuningConfig(use_cuda_graph=True)
    individual = replace(sustained, use_cuda_graph_individual_timing=True)
    seen = []

    def profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        mode = tuning_config.use_cuda_graph_individual_timing
        seen.append((mode, tactic))
        return 1.0 if tactic == int(mode) else 2.0

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
    # This is a CPU selection/cache test; device execution is covered below.
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    with autotune(True):
        _, first = tuner.choose_one("latency_policy", [runner], sustained, inputs)
        _, second = tuner.choose_one("latency_policy", [runner], individual, inputs)
        _, cached = tuner.choose_one("latency_policy", [runner], individual, inputs)
    _, serving = tuner.choose_one("latency_policy", [runner], sustained, inputs)
    assert (first, second, cached, serving) == (0, 1, 1, 1)
    assert sorted(seen) == [(False, 0), (False, 1), (True, 0), (True, 1)]


@pytest.mark.parametrize("repeat,replays", [(0, 1), (1, 0)])
def test_individual_timing_rejects_empty_samples(repeat, replays):
    with pytest.raises(ValueError, match="must be positive"):
        profile_individual_graph_calls(lambda _: None, repeat=repeat, replays=replays)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("globaltimer", [False, True])
def test_individual_graph_replay_executes_each_invocation(globaltimer):
    from flashinfer.utils import get_globaltimer_kernel

    stamp = get_globaltimer_kernel() if globaltimer else None
    if globaltimer:
        assert stamp is not None
    value = torch.zeros(1, device="cuda")
    captured = []

    def run_once(index):
        captured.append(index)
        value.add_(1)

    elapsed = profile_individual_graph_calls(
        run_once, repeat=5, replays=3, record_global_timer=stamp
    )
    assert captured == list(range(5))
    assert value.item() == 20  # Five calls in warmup plus three timed replays.
    assert math.isfinite(elapsed) and elapsed > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_autotuner_individual_timing_runs_and_replays(monkeypatch):
    class CountingRunner(DummyRunner):
        def forward(self, inputs, tactic=-1, **kwargs):
            inputs[0].add_(1)
            return inputs[0]

    tuner = reset_autotuner()
    monkeypatch.setattr(tuner, "warmup", 0)
    # Pass explicit batches so this verifies the actual caller-owned tensor.
    inputs = [torch.zeros(1, device="cuda")]
    config = TuningConfig(
        use_cuda_graph=True,
        use_cuda_graph_individual_timing=True,
        profiling_repeat=5,
        cuda_graph_profile_replays=2,
    )
    elapsed = tuner._profile_single_kernel(
        CountingRunner(), inputs, 0, config, input_tensor_batches=[inputs]
    )
    assert inputs[0].item() == 15
    assert math.isfinite(elapsed) and elapsed > 0
