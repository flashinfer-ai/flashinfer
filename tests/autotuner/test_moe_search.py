# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
import threading

import pytest
import torch

from flashinfer.autotuner.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    _tactic_to_json_hashable,
    autotune,
)
from flashinfer.autotuner.moe_search import iter_moe_tactic_results
from flashinfer.fused_moe.da_tuner import (
    FactorizedTactic,
    FactorizedTacticSpace,
)


class ForeignPair:
    def __init__(self, tile, config):
        self.parts = (tile, config)

    def __iter__(self):
        return iter(self.parts)


class Runner(TunableRunner):
    def __init__(self, *, failure=None):
        self.failure = failure

    def get_valid_tactics(self, inputs, profile):
        return [(8, i) for i in range(9)]

    def get_factorized_tactic_space(self, inputs):
        if self.failure is not None:
            raise self.failure
        return FactorizedTacticSpace(
            [FactorizedTactic((8, i), 8, i // 3, i % 3) for i in range(9)],
            {8: (8, 0)},
        )

    def forward(self, inputs, tactic=-1):
        return inputs[0]


def collect(runner, values, tactics=None):
    calls = []

    def profile(tactic):
        key = _tactic_to_json_hashable(tactic)
        calls.append(key)
        return values[key]

    results = list(
        iter_moe_tactic_results(
            runner,
            [],
            runner.get_valid_tactics([], None) if tactics is None else tactics,
            profile,
            normalize=_tactic_to_json_hashable,
        )
    )
    return results, calls


def test_search_reuses_native_ids_and_measures_fewer_pairs():
    runner = Runner()
    values = {(8, i): 1 + i // 3 + i % 3 for i in range(9)}
    results, calls = collect(runner, values, [ForeignPair(8, i) for i in range(9)])
    assert results == [((8, 0), 1)]
    assert len(calls) < 9
    assert len(calls) == len(set(calls))


def test_missing_metadata_preserves_ordinary_order():
    values = {(8, i): i + 1 for i in range(9)}
    results, calls = collect(
        Runner(failure=RuntimeError("metadata unavailable")), values
    )
    assert calls == list(values)
    assert results == list(values.items())


def test_filtered_space_keeps_only_legal_tactics():
    values = {(8, i): i + 1 for i in range(9)}
    legal = [(8, i) for i in (7, 3, 1)]
    results, calls = collect(Runner(), values, legal)
    assert calls == legal
    assert [t for t, _ in results] == legal


def test_default_candidate_is_not_removed():
    values = {-1: 0.5, **{(8, i): i + 1 for i in range(9)}}
    results, calls = collect(Runner(), values, list(values))
    assert calls == list(values)
    assert min(results, key=lambda pair: pair[1])[0] == -1


def test_failed_profile_is_not_repeated_during_fallback():
    values = {(8, i): i + 1 for i in range(9)}
    values[(8, 3)] = float("inf")
    results, calls = collect(Runner(), values)
    assert len(calls) == len(set(calls)) == 9
    assert [t for t, _ in results] == list(values)
    assert dict(results)[(8, 3)] == float("inf")


def test_unexpected_profile_error_is_not_retried_as_search_fallback():
    calls = []

    def profile(tactic):
        calls.append(tactic)
        if tactic == (8, 3):
            raise RuntimeError("injected failure in profiling bookkeeping")
        return 1.0

    with pytest.raises(RuntimeError, match="bookkeeping"):
        list(
            iter_moe_tactic_results(
                Runner(),
                [],
                [(8, i) for i in range(9)],
                profile,
                normalize=_tactic_to_json_hashable,
            )
        )
    assert calls == [(8, 0), (8, 3)]


def test_metadata_oom_prevents_profiling():
    with pytest.raises(MemoryError, match="admission"):
        collect(Runner(failure=MemoryError("injected")), {})


def test_unserializable_ids_keep_single_rank_exhaustive_fallback():
    legal = [(8, i) for i in range(9)]
    calls = []

    def profile(tactic):
        calls.append(tactic)
        return 1.0

    results = list(
        iter_moe_tactic_results(
            Runner(), [], legal, profile, normalize=lambda tactic: object()
        )
    )
    assert calls == legal
    assert results == [(tactic, 1.0) for tactic in legal]


def test_coupled_local_minimum_is_not_a_global_guarantee():
    values = {(8, i): 3 for i in range(9)}
    values[(8, 0)], values[(8, 8)] = 2, 1
    results, _ = collect(Runner(), values)
    assert results == [((8, 0), 2)]
    assert min(values.values()) == 1


@pytest.fixture
def tuner(monkeypatch):
    instance = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", instance)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    return instance


def test_strategy_inherits_restores_and_stays_thread_local(tuner):
    assert tuner._effective_moe_search_strategy == "exhaustive"
    with autotune(moe_search_strategy="factorized"):
        assert tuner._effective_moe_search_strategy == "factorized"
        with autotune(False):
            assert tuner._effective_moe_search_strategy == "factorized"
        with pytest.raises(RuntimeError), autotune(moe_search_strategy="exhaustive"):
            assert tuner._effective_moe_search_strategy == "exhaustive"
            raise RuntimeError("body failure")
        seen = []
        thread = threading.Thread(
            target=lambda: seen.append(tuner._effective_moe_search_strategy)
        )
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert seen == ["exhaustive"]
        assert tuner._effective_moe_search_strategy == "factorized"
    assert tuner._effective_moe_search_strategy == "exhaustive"


def test_invalid_strategy_does_not_get_or_mutate_tuner(monkeypatch):
    def forbidden():
        pytest.fail("Invalid strategy must fail before accessing the tuner")

    monkeypatch.setattr(AutoTuner, "get", forbidden)
    with (
        pytest.raises(ValueError, match="moe_search_strategy"),
        autotune(cache="must-not-be-opened.json", moe_search_strategy="unknown"),
    ):
        pass


def test_strategy_restores_when_mode_setup_fails(tuner, monkeypatch):
    class BrokenLock:
        def __enter__(self):
            raise RuntimeError("injected mode setup failure")

        def __exit__(self, *args):
            return False

    monkeypatch.setattr(tuner, "_lock", BrokenLock())
    with (
        pytest.raises(RuntimeError, match="mode setup"),
        autotune(moe_search_strategy="factorized"),
    ):
        pass
    assert tuner._effective_moe_search_strategy == "exhaustive"


def test_public_chooser_caches_selected_tactic_without_changing_policy(
    tuner, monkeypatch
):
    calls = []
    runner = Runner()
    config = TuningConfig()
    inputs = [torch.ones(4)]

    def profile(runner, tensors, tactic, tuning_config, **kwargs):
        assert tuning_config == config
        calls.append(_tactic_to_json_hashable(tactic))
        return 1 + tactic[1] // 3 + tactic[1] % 3

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    with autotune(moe_search_strategy="factorized"):
        selected_runner, selected = tuner.choose_one(
            "test_moe", [runner], config, inputs
        )
    assert selected_runner is runner and selected == (8, 0)
    assert len(calls) < 9
    measured = list(calls)
    with autotune(moe_search_strategy="exhaustive"):
        assert tuner.choose_one("test_moe", [runner], config, inputs)[1] == selected
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with autotune(moe_search_strategy="factorized"):
        assert tuner.choose_one("test_moe", [runner], config, inputs)[1] == selected
    assert calls == measured
    assert not tuner.is_tuning_mode


def test_default_chooser_retains_order_and_failure_accounting(tuner, monkeypatch):
    calls = []
    runner = Runner()

    def profile(runner, tensors, tactic, config, **kwargs):
        calls.append(tactic)
        if tactic == (8, 3):
            raise ValueError("injected candidate failure")
        return 1 + tactic[1]

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    with autotune():
        assert tuner.choose_one("test_moe", [runner], TuningConfig(), [torch.ones(4)])[
            1
        ] == (8, 0)
    assert calls == [(8, i) for i in range(9)]
    assert tuner.stats.failed_tactics["test_moe::Runner"] == {(8, 3)}
