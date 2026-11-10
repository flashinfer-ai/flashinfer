"""The record of how each tuned tactic was chosen, saved with its entry in the
cache file so that a table can be audited (issue #7884)."""

import json

import pytest
import torch

import flashinfer.autotuner.autotuner as autotuner_module
from flashinfer import autotune
from flashinfer.autotuner import AutoTuner, TuningConfig

from .utils import DummyRunner, reset_autotuner

INPUTS = [torch.empty((16, 32), dtype=torch.float32)]


@pytest.fixture
def same_environment(monkeypatch):
    monkeypatch.setattr(
        autotuner_module, "_collect_metadata", lambda: {"gpu": "test-gpu"}
    )


def _entries(path):
    saved = json.loads(path.read_text())
    return {k: v for k, v in saved.items() if not k.startswith("_")}


@pytest.mark.parametrize("tune_with", ["choose_one", "rank_tactics"])
def test_saved_entry_records_how_the_winner_was_chosen(
    monkeypatch, tmp_path, same_environment, tune_with
):
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2, 3))
    # Tactic 3 fails to profile and is not a candidate.
    times = {0: 2.0, 1: 1.0, 2: 1.004, 3: float("inf")}
    monkeypatch.setattr(
        AutoTuner, "_profile_single_kernel", lambda self, r, i, tac, *a, **k: times[tac]
    )
    path = tmp_path / "configs.json"

    with autotune(True, cache=str(path)):
        if tune_with == "choose_one":
            _, tactic = tuner.choose_one("audit_op", [runner], TuningConfig(), INPUTS)
            assert tactic == 1
        else:
            ranked = tuner.rank_tactics(
                "audit_op", [runner], TuningConfig(), INPUTS, k=3
            )
            assert ranked == [1, 2, 0]

    (entry,) = _entries(path).values()
    assert entry[:2] == ["DummyRunner", 1]
    assert tuple(entry[2]) == tuner._profiling_policy(TuningConfig())
    assert entry[3] == {
        "time_ms": 1.0,
        "runner_up_ms": 1.004,
        "margin": 0.004,
        "candidates": 3,
    }


def test_selection_record_edge_cases():
    record = AutoTuner._selection_record
    inf = float("inf")
    assert record([1.5, inf], 0) == {
        "time_ms": 1.5,
        "runner_up_ms": None,
        "margin": None,
        "candidates": 1,
    }
    assert record([0.0, 0.5], 0)["margin"] is None
    # A near-tie settled for an earlier, slower tactic.
    assert record([1.004, 1.0], 0)["margin"] == round(1.0 / 1.004 - 1.0, 6)


def test_records_and_older_entries_survive_load_and_save(tmp_path, same_environment):
    policy = ["cuda_graph_profile_replays", None, "l2_cache_policy", "hot"]
    record = {"time_ms": 1.0, "runner_up_ms": 1.25, "margin": 0.25, "candidates": 2}
    entries = {
        # Older versions wrote no record, or no policy either.
        "('legacy_op', 'DummyRunner', ((16, 32),))": ["DummyRunner", 0],
        "('policy_op', 'DummyRunner', ((16, 32),))": ["DummyRunner", 2, policy],
        "('audit_op', 'DummyRunner', ((16, 32),))": ["DummyRunner", 1, policy, record],
    }
    path = tmp_path / "configs.json"
    path.write_text(json.dumps({"_metadata": {"gpu": "test-gpu"}, **entries}))

    tuner = reset_autotuner()
    assert tuner.load_configs(str(path))
    out = tmp_path / "out.json"
    tuner.save_configs(str(out))

    assert _entries(out) == entries
