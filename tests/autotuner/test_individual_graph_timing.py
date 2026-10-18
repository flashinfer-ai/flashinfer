# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Latency tuning must preserve replay semantics and measurement provenance."""

from dataclasses import replace
import math

import pytest
import torch

from flashinfer.autotuner import AutoTuner, TuningConfig, autotune
from flashinfer.autotuner._graph_timing import profile_individual_graph_calls

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
def test_individual_graph_replay_executes_each_invocation():
    value = torch.zeros(1, device="cuda")
    captured = []

    def run_once(index):
        captured.append(index)
        value.add_(1)

    elapsed = profile_individual_graph_calls(run_once, repeat=5, replays=3)
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


@pytest.mark.parametrize("policy_kind", ["default", "cold", "eager"])
def test_moe_candidates_use_one_latency_objective(policy_kind):
    from flashinfer.fused_moe.layer import MoELayer
    from flashinfer.autotune_cache import MeasurementPolicy

    class Runner(DummyRunner):
        def __init__(self, name, individual):
            super().__init__()
            self.backend_key = name
            self.config = TuningConfig(
                use_cuda_graph=True,
                use_cuda_graph_individual_timing=individual,
                # TRT runners use cold L2 by default; Frost uses warm L2.
                use_cold_l2_cache=not individual,
            )
            if individual:
                self.tuning_config = self.config

        def pack_inputs(self, act, weights):
            return [act]

        def launch_kwargs_for(self, inputs):
            return {}

        def tuning_config_for(self, inputs):
            return self.config

    tuned = []
    measured = []
    durations = iter((2.0, 1.0))
    actual_tuner = reset_autotuner()
    policy = {
        "default": None,
        "cold": MeasurementPolicy(cold_l2=True),
        "eager": MeasurementPolicy(execution_mode="eager"),
    }[policy_kind]

    class Tuner:
        _effective_measure_policy = policy

        def _apply_measure_policy(self, config, policy):
            return actual_tuner._apply_measure_policy(config, policy)

        def choose_one(self, **kwargs):
            tuned.append(kwargs["tuning_config"])
            return kwargs["runners"][0], 0

        def profile_tactic(self, runner, inputs, tactic, config, batches):
            measured.append(config)
            runner(inputs, tactic=tactic)
            return next(durations)

    layer = MoELayer.__new__(MoELayer)
    layer.tuner = Tuner()
    original, frost = Runner("original", False), Runner("frost", True)
    winner, tactic = layer._select_winner(torch.zeros(1), None, [original, frost])
    assert winner is frost and tactic == 0
    assert len(tuned) == 2 and all(c.use_cuda_graph_individual_timing for c in tuned)
    assert not any(c.use_cold_l2_cache for c in tuned)
    assert len(measured) == 2 and all(
        c.use_cuda_graph_individual_timing for c in measured
    )
    assert all(c.use_cold_l2_cache == (policy_kind == "cold") for c in measured)
    assert all(c.use_cuda_graph == (policy_kind != "eager") for c in measured)
    assert not original.config.use_cuda_graph_individual_timing
    assert original.config.use_cold_l2_cache
