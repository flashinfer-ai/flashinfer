# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Coverage must distinguish selected defaults from untuned fallback."""

import json
from dataclasses import replace

import pytest
import torch

from flashinfer import ReplayCoverage
from flashinfer.autotune_cache import autotune_v2
from flashinfer.autotuner import AutoTuner, autotune

from .test_autotune_cache_scope import (
    _CONFIG,
    _GRAPH,
    _OP,
    _choose,
    _load_legacy,
)
from tests.cpu.autotuner.test_tuning_protocol import RecordingProtocol
from .utils import DummyRunner, make_scoped_tuner


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    return make_scoped_tuner(tmp_path, monkeypatch)


def _strict(**kwargs):
    return autotune_v2(
        mode="replay",
        cache_scope="context",
        measurement_policy=_GRAPH,
        strict_replay=True,
        **kwargs,
    )


def _tune(tuner):
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        return _choose(tuner)


def test_strict_hit_reports_managed_then_memory(scoped):
    tuner, calls = scoped
    assert _tune(tuner) == 2
    tuner.clear_cache()
    calls.clear()
    with _strict() as report:
        assert _choose(tuner) == 2
        assert _choose(tuner) == 2
    assert (report.hits, report.misses, report.skipped) == (2, 0, 0)
    assert [r["source"] for r in report.records] == ["managed", "memory"]
    assert all(r["selected_tactic"] == "2" for r in report.records)
    assert all(r["requested_policy"] == r["stored_policy"] for r in report.records)
    assert not calls
    json.dumps(report.records)


def test_cached_default_is_a_hit_not_fallback(scoped, monkeypatch):
    tuner, _ = scoped
    monkeypatch.setattr(
        tuner,
        "_profile_single_kernel",
        lambda runner, inputs, tactic, config, **kwargs: 1.0 if tactic == -1 else 3.0,
    )
    assert _tune(tuner) == -1
    tuner.clear_cache()
    with _strict() as report:
        assert _choose(tuner) == -1
    assert report.hits == 1
    assert report.misses == 0
    assert report.records[0]["selected_tactic"] == "-1"


def test_strict_miss_does_not_profile_or_leak_context(scoped):
    tuner, calls = scoped
    report = ReplayCoverage()
    with (
        pytest.raises(RuntimeError, match="strict replay miss"),
        _strict(replay_report=report),
    ):
        _choose(tuner)
    assert not calls
    assert report.misses == 1
    assert report.records[0]["reasons"] == ("not_found",)
    assert not tuner._in_v2_context
    assert tuner._v2_local.replay_report is None
    assert _choose(tuner) == -1
    assert report.misses == 1


def test_policy_mismatch_is_reported_without_replacing_winner(scoped, tmp_path):
    tuner, calls = scoped
    _tune(tuner)
    entry = next(tmp_path.glob("v2/*/entries/*.json"))
    before = entry.read_bytes()
    report = ReplayCoverage()
    calls.clear()
    with (
        pytest.raises(RuntimeError, match="policy_mismatch"),
        _strict(replay_report=report),
    ):
        _choose(tuner, replace(_CONFIG, cuda_graph_profile_replays=3))
    assert report.records[0]["reasons"] == ("policy_mismatch",)
    assert report.records[0]["requested_policy"]
    assert entry.read_bytes() == before
    with _strict():
        assert _choose(tuner) == 2
    assert not calls


def test_unknown_disk_policy_remains_usable_only_without_strict(scoped, tmp_path):
    tuner, calls = scoped
    _tune(tuner)
    entry = next(tmp_path.glob("v2/*/entries/*.json"))
    payload = json.loads(entry.read_text())
    payload.pop("profiling_policy")
    entry.write_text(json.dumps(payload))
    tuner.clear_cache()
    calls.clear()
    with autotune_v2(mode="replay", cache_scope="context", measurement_policy=_GRAPH):
        assert _choose(tuner) == 2
    with pytest.raises(RuntimeError, match="policy_mismatch"), _strict():
        _choose(tuner)
    assert not calls


def test_rejected_tactic_is_not_a_strict_hit(scoped, monkeypatch):
    tuner, calls = scoped
    _tune(tuner)
    runner = DummyRunner()
    monkeypatch.setattr(
        runner, "validate_tactic", lambda inputs, tactic: False, raising=False
    )
    calls.clear()
    with pytest.raises(RuntimeError, match="invalid_tactic"), _strict():
        tuner.choose_one(_OP, [runner], _CONFIG, [torch.ones(2)])
    assert not calls


@pytest.mark.parametrize("limit", [0, 2])
def test_report_is_bounded_but_counts_all_lookups(scoped, limit):
    tuner, calls = scoped
    report = ReplayCoverage(max_records=limit)
    with autotune_v2(mode="replay", persistent_cache=False, replay_report=report):
        for _ in range(5):
            assert _choose(tuner) == -1
    assert report.misses == 5
    assert report.hits == 0
    assert len(report.records) == limit
    assert report.dropped_records == 5 - limit
    assert not calls


def test_explicit_skip_is_reported_separately(scoped):
    tuner, calls = scoped
    with _strict(skip_ops={_OP}) as report:
        assert _choose(tuner) == -1
    assert (report.hits, report.misses, report.skipped) == (0, 0, 1)
    assert report.records[0]["reasons"] == ("skip_ops",)
    assert not calls


def test_observation_preserves_legacy_precedence(scoped, tmp_path):
    tuner, calls = scoped
    _load_legacy(tuner, tmp_path)
    report = ReplayCoverage()
    with autotune_v2(mode="replay", measurement_policy=_GRAPH, replay_report=report):
        assert _choose(tuner) == 0
    assert report.records[0]["source"] == "legacy"
    assert report.hits == 1
    assert not calls


def test_observation_reports_bundled_source(scoped, monkeypatch):
    tuner, _ = scoped
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    monkeypatch.setattr(
        "flashinfer.autotuner.autotuner.load_from_file",
        lambda key: (True, "DummyRunner", 1, None),
    )
    report = ReplayCoverage()
    with autotune_v2(mode="replay", persistent_cache=False, replay_report=report):
        assert _choose(tuner) == 1
    assert report.records[0]["source"] == "bundled"


def test_reporting_refuses_nested_and_explicit_profiling(scoped):
    tuner, calls = scoped
    with _strict():
        with pytest.raises(RuntimeError, match="forbid tuning"), autotune(True):
            _choose(tuner)
        with pytest.raises(RuntimeError, match="forbid tuning"), autotune(True):
            tuner.rank_tactics(_OP, [DummyRunner()], _CONFIG, [torch.ones(2)], k=2)
        with pytest.raises(RuntimeError, match="forbid profiling"):
            AutoTuner._profile_single_kernel(
                tuner, DummyRunner(), [torch.ones(2)], 0, _CONFIG
            )
    assert not calls


def test_protocol_reports_hits_misses_and_skips(scoped):
    tuner, _ = scoped
    protocol = RecordingProtocol()
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        assert tuner.choose_with_protocol("coverage.protocol", protocol) == "tuned"
    with _strict() as report:
        assert tuner.choose_with_protocol("coverage.protocol", protocol)["tile"] == [
            128,
            256,
        ]
        with pytest.raises(RuntimeError, match="forbid protocol tuning"):
            tuner.choose_with_protocol("coverage.protocol", protocol, force_tune=True)
        with pytest.raises(RuntimeError, match="strict replay miss for protocol"):
            tuner.choose_with_protocol("coverage.absent", protocol)
    assert (report.hits, report.misses) == (1, 1)
    assert report.records[0]["source"] == "protocol"
    with _strict(skip_ops={"coverage.protocol"}) as skipped:
        assert tuner.choose_with_protocol("coverage.protocol", protocol) == "deferred"
    assert skipped.skipped == 1


@pytest.mark.parametrize("configured_group", [False, True])
def test_strict_replay_rejects_uncoordinated_multirank(
    scoped, monkeypatch, configured_group
):
    tuner, _ = scoped
    if configured_group:
        monkeypatch.setattr(
            "flashinfer.autotuner.get_autotune_process_group", lambda: object()
        )
    else:
        monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
        monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    with pytest.raises(RuntimeError, match="single-rank"), _strict():
        pytest.fail("strict multi-rank replay accepted")
    assert not tuner._managed_stores
    assert not tuner.is_tuning_mode


@pytest.mark.parametrize(
    "kwargs,error",
    [
        ({"mode": "tune", "replay_report": ReplayCoverage()}, ValueError),
        ({"mode": "replay", "strict_replay": 1}, TypeError),
        ({"mode": "replay", "replay_report": {}}, TypeError),
        ({"mode": "replay", "strict_replay": True}, ValueError),
        (
            {
                "mode": "replay",
                "cache_scope": "context",
                "persistent_cache": False,
                "strict_replay": True,
            },
            ValueError,
        ),
    ],
)
def test_invalid_diagnostics_arguments_have_no_side_effects(scoped, kwargs, error):
    tuner, _ = scoped
    with pytest.raises(error), autotune_v2(**kwargs):
        pytest.fail("invalid arguments accepted")
    assert not tuner._managed_stores
    assert not tuner.is_tuning_mode


def test_reporting_cannot_enter_an_existing_tuning_context(scoped):
    tuner, _ = scoped
    with (
        autotune(True),
        pytest.raises(RuntimeError, match="while tuning is enabled"),
        _strict(),
    ):
        pytest.fail("strict replay inherited tuning")
    assert not tuner._managed_stores


@pytest.mark.parametrize("value", [-1, 1.5, True])
def test_invalid_report_limit(value):
    with pytest.raises(ValueError, match="max_records"):
        ReplayCoverage(max_records=value)


def test_replay_shortlist_uses_the_reported_winner(scoped):
    tuner, calls = scoped
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        assert (
            tuner.rank_tactics(_OP, [DummyRunner()], _CONFIG, [torch.ones(2)], k=2)[0]
            == 2
        )
    calls.clear()
    with _strict() as report:
        assert tuner.rank_tactics(
            _OP, [DummyRunner()], _CONFIG, [torch.ones(2)], k=2
        ) == [2]
    assert report.hits == 1
    assert not calls
