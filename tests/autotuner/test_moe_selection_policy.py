# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MoE selection lifecycle with CPU operands and synthetic profiling times."""

from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import flashinfer.autotuner as tuner_package
from flashinfer import MeasurementPolicy, autotune, autotune_v2, autotune_v2_reload
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from flashinfer.fused_moe import MoEActivationPack, MoELayer
from flashinfer.fused_moe.layer import _BackendChoiceRunner

from .utils import reset_autotuner


class PackedInputs(list):
    pass


class SelectionRunner(TunableRunner):
    def __init__(self, key, mode, version=0):
        self.backend_key = key
        self.supported_routing_modes = (mode,)
        self.version = version

    def __hash__(self):
        return hash((self.backend_key, self.version))

    def get_cache_key_extras(self, inputs):
        return self.backend_key, self.version

    def get_valid_tactics(self, inputs, profile):
        return [7]

    def precompile_tactics(self, inputs, tactics, profile, **kwargs):
        return True

    def pack_inputs(self, activation, weights):
        result = PackedInputs(
            [activation.hidden_states_q, activation.topk_ids, activation.topk_weights]
        )
        result.launch_state = (weights, activation, self.backend_key)
        return result

    def tuning_config_for(self, inputs):
        return TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True)

    def launch_kwargs_for(self, inputs):
        return dict(launch_state=inputs.launch_state)

    def forward(self, inputs, tactic=-1, do_preparation=False, *, launch_state):
        weights, activation, key = launch_state
        assert key == self.backend_key and tactic in [-1, 7]
        assert inputs[0] is activation.hidden_states_q
        assert inputs[1] is activation.topk_ids
        assert inputs[2] is activation.topk_weights
        return inputs[0] + weights.value


@pytest.fixture
def selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(AutoTuner, "_get_l2_cache_size_in_bytes", lambda *_: 32768)
    tuner = reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()
    state = SimpleNamespace(
        tuner=tuner,
        act=MoEActivationPack(
            torch.arange(32).reshape(8, 4).to(torch.bfloat16),
            None,
            torch.zeros((8, 2), dtype=torch.int32),
            torch.ones((8, 2)),
        ),
        weights=SimpleNamespace(value=torch.tensor(11.0)),
        preferred=None,
        calls=[],
        profile_errors=[],
    )

    def layer(keys=("a", "b"), version=0):
        # Exercise the real selection methods without constructing GPU runners.
        result = MoELayer.__new__(MoELayer)
        result.tuner = tuner
        result.config = SimpleNamespace(
            execution=SimpleNamespace(tune_max_num_tokens=64)
        )
        result.runners = [
            SelectionRunner(key, state.act.routing_input_mode, version) for key in keys
        ]
        result._winners = {}
        result._winner_partition = None
        result._winner_context = None
        result._last_winner_backend = None
        return result

    def batches(inputs, config):
        if config.use_cold_l2_cache:
            return [inputs, [x.clone() for x in inputs]]
        return [inputs]

    def profile(runner, inputs, tactic, config, *, input_tensor_batches, **kwargs):
        cross = isinstance(runner, _BackendChoiceRunner)
        if cross:
            assert not kwargs and runner.tactic == 7
            for batch in input_tensor_batches:
                torch.testing.assert_close(
                    runner.forward(batch), batch[0] + runner.weights.value
                )
            assert not config.dynamic_tensor_specs
            preferred = state.preferred or (
                "b"
                if config.cuda_graph_profile_replays == 3
                else ("a" if config.use_cuda_graph != config.use_cold_l2_cache else "b")
            )
            measured = 1.0 if runner.runner.backend_key == preferred else 2.0
        else:
            torch.testing.assert_close(
                runner.forward(inputs, tactic=tactic, **kwargs),
                inputs[0] + kwargs["launch_state"][0].value,
            )
            # v2 also races tactic=-1. Give the explicit tactic a distinct
            # latency so the wrapper checks actual selected-tactic propagation.
            measured = 1.0 if tactic == 7 else 2.0
        state.calls.append(
            dict(
                cross=cross,
                graph=config.use_cuda_graph,
                cold=config.use_cold_l2_cache,
                batches=len(input_tensor_batches),
            )
        )
        return measured

    def checked_profile(*args, **kwargs):
        try:
            return profile(*args, **kwargs)
        except Exception as exc:
            # choose_one treats ordinary profiling exceptions as unsupported
            # tactics. Do not let that hide assertions in this CPU fixture.
            state.profile_errors.append(repr(exc))
            raise

    monkeypatch.setattr(tuner, "_prepare_input_tensors_with_batches", batches)
    monkeypatch.setattr(tuner, "_profile_single_kernel", checked_profile)
    monkeypatch.setattr(
        tuner_package,
        "_collect_metadata",
        lambda: {
            "flashinfer_version": "cpu-selection-contract",
            "gpu_name": "synthetic-cpu-only",
            "cuda_version": "synthetic",
        },
    )
    state.layer = layer
    yield state
    reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()
    assert not state.profile_errors


def check_choice(state, layer, expected, activation=None):
    activation = activation or state.act
    torch.testing.assert_close(
        layer(activation, state.weights),
        activation.hidden_states_q + state.weights.value,
    )
    assert layer.winner_backend == expected


def test_layer_switches_policies_and_replays_without_profiling(selection):
    state = selection
    layer = state.layer()
    for mode in ["cuda_graph", "eager"]:
        for cold in [False, True]:
            policy = MeasurementPolicy(execution_mode=mode, cold_l2=cold)
            expected = "a" if (mode == "cuda_graph") != cold else "b"
            start = len(state.calls)
            with autotune_v2(
                mode="tune", persistent_cache=False, measurement_policy=policy
            ):
                check_choice(state, layer, expected)
                count = len(state.calls)
                check_choice(state, layer, expected)
                assert len(state.calls) == count
            calls = state.calls[start:]
            assert sum(c["cross"] for c in calls) == 2
            assert all(c["graph"] == (mode == "cuda_graph") for c in calls)
            assert all(c["cold"] == cold for c in calls)
            layer.reset_winner()
            with autotune_v2(
                mode="replay", persistent_cache=False, measurement_policy=policy
            ):
                check_choice(state, layer, expected)
                check_choice(state, state.layer(), expected)
            assert len(state.calls) == count


def test_layer_opt_out_does_not_profile_or_prevent_later_tuning(selection):
    state = selection
    layer = state.layer(("b", "a"))
    check_choice(state, layer, "b")
    assert not state.calls
    with autotune(True):
        check_choice(state, layer, "a")
    assert sum(c["cross"] for c in state.calls) == 2


def test_layer_retunes_different_shapes_and_candidate_rosters(selection):
    state = selection
    policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=False)
    layer = state.layer()
    with autotune_v2(mode="tune", persistent_cache=False, measurement_policy=policy):
        check_choice(state, layer, "a")
        smaller = replace(
            state.act,
            hidden_states_q=state.act.hidden_states_q[:7],
            topk_ids=state.act.topk_ids[:7],
            topk_weights=state.act.topk_weights[:7],
        )
        start = len(state.calls)
        check_choice(state, layer, "a", smaller)
        assert sum(c["cross"] for c in state.calls[start:]) == 2
        for other in [
            state.layer(("a", "b", "c")),
            state.layer(version=1),
            state.layer(("b", "a")),
        ]:
            start = len(state.calls)
            check_choice(state, other, "a")
            assert sum(c["cross"] for c in state.calls[start:]) == len(other.runners)


def test_layer_observes_cache_clear_and_nested_contexts(selection):
    state = selection
    layer = state.layer(("b", "a"))
    with autotune(True):
        check_choice(state, layer, "a")
        with autotune(True, cuda_graph_profile_replays=3):
            check_choice(state, layer, "b")
        check_choice(state, layer, "a")
        count = len(state.calls)
        with autotune(True, skip_ops={"moe_backend_choice"}):
            check_choice(state, layer, "b")
        check_choice(state, layer, "a")
        assert len(state.calls) == count
        state.tuner.clear_cache()
        state.preferred = "b"
        check_choice(state, layer, "b")
        assert len(state.calls) > count


def test_layer_replays_separate_persistent_stores(selection, tmp_path):
    state = selection
    layer = state.layer(("b", "a"))
    policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=False)
    roots = [tmp_path / "a", tmp_path / "b"]
    for root, expected in zip(roots, ["a", "b"], strict=True):
        state.preferred = expected
        with autotune_v2(mode="tune", cache_root=root, measurement_policy=policy):
            check_choice(state, layer, expected)
    count = len(state.calls)
    for root, expected in [(roots[0], "a"), (roots[1], "b"), (roots[0], "a")]:
        with autotune_v2(mode="replay", cache_root=root, measurement_policy=policy):
            check_choice(state, layer, expected)
            check_choice(state, state.layer(("b", "a")), expected)
    autotune_v2_reload()
    check_choice(state, layer, "a")
    assert len(state.calls) == count


def test_selection_wrapper_keeps_each_activation_and_weight_pack(selection):
    state = selection
    runner = SelectionRunner("a", state.act.routing_input_mode)
    other = replace(state.act, hidden_states_q=-state.act.hidden_states_q)
    weights = SimpleNamespace(value=torch.tensor(23.0))
    names = ("hidden_states_q", "topk_ids", "topk_weights")
    first = _BackendChoiceRunner(
        runner, 7, state.act, state.weights, names, ("test", 1)
    )
    second = _BackendChoiceRunner(runner, 7, other, weights, names, ("test", 2))
    for wrapped in [first, second, first, second]:
        inputs = [getattr(wrapped.activation, name).clone() for name in names]
        torch.testing.assert_close(
            wrapped.forward(inputs), inputs[0] + wrapped.weights.value
        )
