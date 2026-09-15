# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-runner selection retains its tactic while avoiding a pointless timer."""

from types import SimpleNamespace
from unittest.mock import Mock

from flashinfer.fused_moe import MoELayer
import flashinfer.testing.utils as timing


def _runner(name, elapsed):
    inputs = object()
    config = object()
    state = object()
    return SimpleNamespace(
        backend_key=name,
        inputs=inputs,
        config=config,
        state=state,
        tactic=(name, 17),
        pack_inputs=Mock(return_value=inputs),
        launch_kwargs_for=Mock(return_value={"launch_state": state}),
        tuning_config_for=Mock(return_value=config),
        forward=Mock(return_value=elapsed),
    )


def _layer():
    layer = object.__new__(MoELayer)
    layer.tuner = SimpleNamespace(
        choose_one=Mock(
            side_effect=lambda **kw: (kw["runners"][0], kw["runners"][0].tactic)
        )
    )
    return layer


def test_single_candidate_keeps_internal_tactic_selection_without_cross_runner_timing(
    monkeypatch,
):
    timer = Mock(
        side_effect=AssertionError(
            "A unique candidate needs no cross-runner measurement"
        )
    )
    monkeypatch.setattr(timing, "bench_gpu_time", timer)
    layer = _layer()
    runner = _runner("single", 0.25)
    act, weights = object(), object()
    selected, tactic = layer._select_winner(act, weights, [runner])
    assert selected is runner
    assert tactic == runner.tactic
    runner.pack_inputs.assert_called_once_with(act, weights)
    layer.tuner.choose_one.assert_called_once_with(
        custom_op="moe_single",
        runners=[runner],
        tuning_config=runner.config,
        inputs=runner.inputs,
        launch_state=runner.state,
    )
    timer.assert_not_called()
    runner.forward.assert_not_called()


def test_multiple_candidates_still_compare_their_selected_tactics(monkeypatch):
    timer = Mock(side_effect=lambda fn, **kw: [fn()])
    monkeypatch.setattr(timing, "bench_gpu_time", timer)
    layer = _layer()
    first, second = _runner("first", 0.25), _runner("second", 0.125)
    selected, tactic = layer._select_winner(object(), object(), [first, second])
    assert selected is second
    assert tactic == second.tactic
    assert layer.tuner.choose_one.call_count == timer.call_count == 2
    for runner in (first, second):
        runner.forward.assert_called_once_with(
            runner.inputs, tactic=runner.tactic, launch_state=runner.state
        )
