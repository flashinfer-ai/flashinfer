# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stage-ranking contracts use CPU tensors and synthetic timing only."""

from types import SimpleNamespace
from unittest.mock import MagicMock
import json
import pytest
import torch
import flashinfer.autotuner as tuner_package
from flashinfer import MeasurementPolicy, autotune, autotune_v2, autotune_v2_reload
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig
from .utils import reset_autotuner


class RankingRunner(TunableRunner):
    def __hash__(self):
        return 913

    def get_valid_tactics(self, inputs, profile):
        return [1, 2]

    def forward(self, inputs, tactic=-1, do_preparation=False):
        return inputs[0]


@pytest.fixture
def ranking(monkeypatch):
    tuner = reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()
    state = SimpleNamespace(
        tuner=tuner,
        runner=RankingRunner(),
        inputs=[torch.arange(32, dtype=torch.float32).reshape(4, 8)],
        config=TuningConfig(use_cuda_graph=True, use_cold_l2_cache=True),
        calls=[],
        preferred=None,
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(AutoTuner, "_get_l2_cache_size_in_bytes", lambda *_: 32768)
    monkeypatch.setattr(
        tuner_package,
        "_collect_metadata",
        lambda: {
            "flashinfer_version": "cpu-ranking-contract",
            "gpu_name": "synthetic-cpu-only",
            "cuda_version": "synthetic",
        },
    )

    def profile(runner, inputs, tactic, config, **kwargs):
        assert runner is state.runner
        batches = kwargs.get("input_tensor_batches")
        state.calls.append(
            dict(
                tactic=tactic,
                graph=config.use_cuda_graph,
                cold=config.use_cold_l2_cache,
                replays=config.cuda_graph_profile_replays,
                batches=batches,
            )
        )
        policy = tuner._effective_measure_policy
        cold = policy.cold_l2 if policy is not None else False
        preferred = state.preferred or (
            2 if cold or config.cuda_graph_profile_replays == 3 else 1
        )
        return 1.0 if tactic == preferred else 2.0

    monkeypatch.setattr(tuner, "_profile_single_kernel", profile)
    yield state
    reset_autotuner()
    tuner._managed_cache = None
    tuner._managed_stores.clear()
    tuner._managed_decoded.clear()


def rank(state):
    return state.tuner.rank_tactics(
        "ranking_contract", [state.runner], state.config, state.inputs, k=2
    )


def choose(state):
    return state.tuner.choose_one(
        "ranking_contract", [state.runner], state.config, state.inputs
    )[1]


@pytest.mark.parametrize("mode", ["cuda_graph", "eager"])
@pytest.mark.parametrize("cold", [False, True])
def test_ranking_applies_policy_and_reuses_prepared_batches(ranking, mode, cold):
    policy = MeasurementPolicy(execution_mode=mode, cold_l2=cold)
    with autotune_v2(mode="tune", persistent_cache=False, measurement_policy=policy):
        assert rank(ranking) == ([2, 1] if cold else [1, 2])
        count = len(ranking.calls)
        assert rank(ranking) == ([2, 1] if cold else [1, 2])
        assert len(ranking.calls) == count == 2
    for call in ranking.calls:
        assert call["graph"] == (mode == "cuda_graph")
        assert call["cold"] == cold
        assert call["batches"] is ranking.calls[0]["batches"]
        assert len(call["batches"]) > 1 if cold else len(call["batches"]) == 1
        for batch in call["batches"]:
            torch.testing.assert_close(batch[0], ranking.inputs[0])


def test_ranking_switches_policy_and_restores_each_shortlist(ranking):
    for cold, expected, count in [
        (False, [1, 2], 2),
        (True, [2, 1], 4),
        (False, [1, 2], 4),
    ]:
        with autotune_v2(
            mode="tune",
            persistent_cache=False,
            measurement_policy=MeasurementPolicy(cold_l2=cold),
        ):
            assert rank(ranking) == expected
            assert len(ranking.calls) == count


def test_ranking_winner_is_replayed_from_active_policy_partition(ranking):
    policy = MeasurementPolicy(cold_l2=False)
    with autotune_v2(mode="tune", persistent_cache=False, measurement_policy=policy):
        expected = rank(ranking)[0]
        assert len(ranking.tuner._winner_cache()) == 1
        assert not ranking.tuner.profiling_cache
    count = len(ranking.calls)
    with autotune_v2(mode="replay", persistent_cache=False, measurement_policy=policy):
        assert choose(ranking) == expected
    assert len(ranking.calls) == count


def test_ranking_publication_invalidates_product_selection_context(ranking):
    with autotune(True):
        before = ranking.tuner._selection_context()
        rank(ranking)
        after = ranking.tuner._selection_context()
        assert after[0] > before[0]
        rank(ranking)
        assert ranking.tuner._selection_context() == after


def test_ranking_replay_override_has_its_own_shortlist(ranking):
    with autotune(True):
        assert rank(ranking) == [1, 2]
        with autotune(True, cuda_graph_profile_replays=3):
            assert rank(ranking) == [2, 1]
        # The winner in this legacy partition now differs. Rebuild the old
        # shortlist and publish its winner, rather than returning stale cache.
        assert rank(ranking) == [1, 2]
    assert [c["replays"] for c in ranking.calls] == [1, 1, 3, 3, 1, 1]


def test_ranking_capture_miss_rejects_before_input_preparation(ranking, monkeypatch):
    prepare = MagicMock(side_effect=AssertionError("prepared during capture"))
    monkeypatch.setattr(ranking.tuner, "_prepare_input_tensors", prepare)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with autotune(True), pytest.raises(RuntimeError, match="active outer CUDA Graph"):
        rank(ranking)
    prepare.assert_not_called()
    assert not ranking.calls


def test_ranking_capture_hit_neither_prepares_nor_profiles(ranking, monkeypatch):
    with autotune(True):
        expected = rank(ranking)
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
        prepare = MagicMock(side_effect=AssertionError("unexpected preparation"))
        monkeypatch.setattr(ranking.tuner, "_prepare_input_tensors", prepare)
        assert rank(ranking) == expected
    prepare.assert_not_called()
    assert len(ranking.calls) == 2


def test_ranking_uses_precompile_contract(ranking, monkeypatch):
    precompile = MagicMock(return_value=True)
    monkeypatch.setattr(ranking.runner, "precompile_tactics", precompile)
    forward = MagicMock(side_effect=AssertionError("fallback preparation ran"))

    # Preserve the forward signature while checking that handled preparation
    # never invokes the fallback route.
    def wrapped(inputs, tactic=-1, do_preparation=False):
        return forward(inputs, tactic=tactic, do_preparation=do_preparation)

    monkeypatch.setattr(ranking.runner, "forward", wrapped)
    with autotune(True):
        assert rank(ranking) == [1, 2]
    precompile.assert_called_once()
    forward.assert_not_called()


def test_ranking_publishes_to_each_store_and_reload_discards_shortlists(
    ranking, tmp_path
):
    policy = MeasurementPolicy(cold_l2=False)
    roots = [tmp_path / "a", tmp_path / "b"]
    for cache, preferred in zip(roots, [1, 2], strict=True):
        ranking.preferred = preferred
        with autotune_v2(mode="tune", cache_root=cache, measurement_policy=policy):
            assert rank(ranking)[0] == preferred
        assert choose(ranking) == preferred
        entries = list(cache.glob("v2/*/entries/*.json"))
        assert len(entries) == 1
    count = len(ranking.calls)
    for cache, expected in [(roots[0], 1), (roots[1], 2), (roots[0], 1)]:
        with autotune_v2(mode="replay", cache_root=cache, measurement_policy=policy):
            assert choose(ranking) == expected
        assert choose(ranking) == expected
    assert len(ranking.calls) == count
    assert ranking.tuner._ranked_tactics_cache
    autotune_v2_reload()
    assert not ranking.tuner._ranked_tactics_cache
    assert choose(ranking) == 1
    assert len(ranking.calls) == count


def test_ranking_loaded_configs_discard_shortlists(ranking, tmp_path):
    with autotune(True):
        assert rank(ranking) == [1, 2]
        key = next(iter(ranking.tuner.profiling_cache))
        cache = tmp_path / "legacy.json"
        cache.write_text(json.dumps({key.file_key: [key.runner_class_name, 2]}))
        assert ranking.tuner.load_configs(str(cache))
        assert not ranking.tuner._ranked_tactics_cache
        ranking.preferred = 2
        assert rank(ranking) == [2, 1]
