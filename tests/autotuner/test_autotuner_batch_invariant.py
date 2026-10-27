"""Tests for batch-invariant autotuning (set_autotune_batch_invariant).

No GPU is required: profiling is faked and tensors live on the CPU.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

import flashinfer.autotuner.autotuner as autotuner_module
from flashinfer import ReplayCoverage
from flashinfer.autotune_cache import autotune_v2
from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    autotune,
    get_autotune_batch_invariant,
    set_autotune_batch_invariant,
)

from .test_autotune_cache_scope import _CONFIG, _GRAPH, _OP
from .utils import DummyRunner, make_scoped_tuner, reset_autotuner

# Tactic 1 is the fastest but splits the reduction (e.g. split-K).
TIMES = {0: 5.0, 1: 1.0, 2: 3.0}


class SplitRunner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def is_batch_invariant_tactic(self, tactic):
        return tactic != 1

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def _mode(enabled):
    saved = get_autotune_batch_invariant()
    set_autotune_batch_invariant(enabled)
    try:
        yield
    finally:
        set_autotune_batch_invariant(saved)


@pytest.fixture
def batch_invariant():
    yield from _mode(True)


@pytest.fixture
def default_mode():
    yield from _mode(False)


@pytest.fixture
def fake_profile(monkeypatch):
    calls = []

    def profile(self, runner_obj, prof_inputs, tactic, tuning_config=None, **kwargs):
        calls.append(tactic)
        return TIMES[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
    return calls


def _inputs():
    return [torch.empty((16, 32), dtype=torch.float32)]


def test_default_runner_trusts_every_tactic():
    class Plain(TunableRunner):
        def get_valid_tactics(self, inputs, profile):
            return [0]

        def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
            return inputs[0]

    assert Plain().is_batch_invariant_tactic((128, 32, 4))
    assert Plain().is_batch_invariant_tactic(-1)


def test_tuning_picks_fastest_tactic_by_default(fake_profile, default_mode):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        _, tactic = tuner.choose_one(
            "bi_off", [SplitRunner()], TuningConfig(), _inputs()
        )
    assert tactic == 1


def test_tuning_skips_non_invariant_tactics(fake_profile, batch_invariant):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        _, tactic = tuner.choose_one(
            "bi_on", [SplitRunner()], TuningConfig(), _inputs()
        )
    assert tactic == 2
    assert 1 not in fake_profile


def test_cached_non_invariant_tactic_is_a_miss(batch_invariant):
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    key = AutoTuner._get_cache_key("bi_mem", runner, (inputs[0].shape,), config)
    tuner.profiling_cache[key] = (1, None)
    # Inference: the cached split tactic is ignored, the fallback runs.
    assert tuner.choose_one("bi_mem", [runner], config, inputs)[1] == -1
    tuner.profiling_cache[key] = (2, None)
    assert tuner.choose_one("bi_mem", [runner], config, inputs)[1] == 2


def test_cached_non_invariant_tactic_is_reprofiled(fake_profile, batch_invariant):
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    key = AutoTuner._get_cache_key("bi_retune", runner, (inputs[0].shape,), config)
    tuner.profiling_cache[key] = (1, None)
    with autotune(tune_mode=True):
        assert tuner.choose_one("bi_retune", [runner], config, inputs)[1] == 2


def test_file_config_with_non_invariant_tactic_is_ignored(
    monkeypatch, tmp_path, fake_profile, default_mode
):
    monkeypatch.setattr(
        autotuner_module, "_collect_metadata", lambda: {"gpu": "test-gpu"}
    )
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    with autotune(tune_mode=True):
        assert tuner.choose_one("bi_file", [runner], config, inputs)[1] == 1
    path = str(tmp_path / "tuned.json")
    tuner.save_configs(path)
    assert any(
        not k.startswith("_") for k in json.loads((tmp_path / "tuned.json").read_text())
    )

    shapes = (tuple(inputs[0].shape),)
    fresh = reset_autotuner()
    assert fresh.load_configs(path)
    assert fresh.search_cache("bi_file", [runner], shapes, config, inputs)[:3] == (
        True,
        0,
        1,
    )
    saved = get_autotune_batch_invariant()
    set_autotune_batch_invariant(True)
    try:
        assert not fresh.search_cache("bi_file", [runner], shapes, config, inputs)[0]
        assert fresh.choose_one("bi_file", [runner], config, inputs)[1] == -1
    finally:
        set_autotune_batch_invariant(saved)


def test_ranked_shortlist_cached_before_mode_is_reranked(fake_profile, default_mode):
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [1, 2]
    fake_profile.clear()
    set_autotune_batch_invariant(True)
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [2, 0]
    assert sorted(fake_profile) == [0, 2]
    # The re-ranked shortlist is invariant, so it is served from the cache.
    fake_profile.clear()
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [2, 0]
    assert not fake_profile


# --- autotune_v2 integration: managed cache, replay coverage, default race ---


class _NoTwo(DummyRunner):
    """Graph-timed winner of make_scoped_tuner is tactic 2; call it split-K."""

    def is_batch_invariant_tactic(self, tactic):
        return tactic != 2


class _NoTwoNoDefault(_NoTwo):
    """The untuned default (-1) is not batch-invariant either."""

    def is_batch_invariant_tactic(self, tactic):
        return tactic not in (-1, 2)


def _v2_choose(tuner, runner):
    return tuner.choose_one(_OP, [runner], _CONFIG, [torch.ones(2)])[1]


def _v2_tune(tuner, runner):
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        return _v2_choose(tuner, runner)


def _v2_replay(**kwargs):
    return autotune_v2(
        mode="replay", cache_scope="context", measurement_policy=_GRAPH, **kwargs
    )


def test_managed_non_invariant_winner_is_a_reported_miss(
    tmp_path, monkeypatch, default_mode
):
    tuner, calls = make_scoped_tuner(tmp_path, monkeypatch)
    runner = _NoTwo()
    assert _v2_tune(tuner, runner) == 2
    tuner.clear_cache()
    calls.clear()
    set_autotune_batch_invariant(True)
    report = ReplayCoverage()
    with _v2_replay(replay_report=report):
        assert _v2_choose(tuner, runner) == -1
    assert (report.hits, report.misses) == (0, 1)
    assert report.records[0]["reasons"] == ("not_batch_invariant",)
    # Strict replay rejects it like any other miss, without profiling.
    with (
        pytest.raises(RuntimeError, match="not_batch_invariant"),
        _v2_replay(strict_replay=True),
    ):
        _v2_choose(tuner, runner)
    assert not calls
    # Without the mode the same store entry is still a hit.
    set_autotune_batch_invariant(False)
    tuner.clear_cache()
    with _v2_replay(strict_replay=True) as report:
        assert _v2_choose(tuner, runner) == 2
    assert report.hits == 1


def test_retune_under_mode_publishes_invariant_winner(
    tmp_path, monkeypatch, default_mode
):
    tuner, calls = make_scoped_tuner(tmp_path, monkeypatch)
    runner = _NoTwo()
    assert _v2_tune(tuner, runner) == 2
    tuner.clear_cache()
    calls.clear()
    set_autotune_batch_invariant(True)
    assert _v2_tune(tuner, runner) == 0
    assert all(t != 2 for _, t in calls)
    tuner.clear_cache()
    with _v2_replay(strict_replay=True) as report:
        assert _v2_choose(tuner, runner) == 0
    assert report.hits == 1


def test_v2_default_race_skips_non_invariant_default(
    tmp_path, monkeypatch, batch_invariant
):
    tuner, calls = make_scoped_tuner(tmp_path, monkeypatch)
    assert _v2_tune(tuner, _NoTwo()) == 0
    assert -1 in [t for _, t in calls]
    tuner.clear_cache()
    calls.clear()
    tuner2, calls2 = make_scoped_tuner(tmp_path / "b", monkeypatch)
    assert _v2_tune(tuner2, _NoTwoNoDefault()) == 0
    assert sorted(t for _, t in calls2) == [0, 1]


# --- distributed: every rank must use the same setting ---

_DIST_WORKER = r"""
import datetime, os
import torch
import torch.distributed as dist
from flashinfer.autotuner import (
    AutoTuner, TunableRunner, TuningConfig, autotune,
    set_autotune_batch_invariant, set_autotune_process_group,
)
from flashinfer.autotuner.autotuner import _TuningGroupError

rank = int(os.environ["RANK"])
dist.init_process_group("gloo", init_method="file://" + os.environ["STORE"],
    rank=rank, world_size=2, timeout=datetime.timedelta(seconds=25))
torch.cuda.is_current_stream_capturing = lambda: False
torch.cuda.empty_cache = lambda: None
set_autotune_process_group(dist.group.WORLD)

class Split(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]
    def is_batch_invariant_tactic(self, tactic):
        return tactic != 1
    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]

def profile(self, runner, inputs, tactic, config, **kwargs):
    t = torch.tensor([{0: 5.0, 1: 1.0, 2: 3.0}[tactic]], dtype=torch.float64)
    dist.all_reduce(t)
    return t.item() / 2

AutoTuner._profile_single_kernel = profile
config = TuningConfig(use_cold_l2_cache=False)
try:
    for case, modes in (("same", (True, True)), ("mixed", (False, True))):
        set_autotune_batch_invariant(modes[rank])
        tuner = AutoTuner()
        AutoTuner._instance = tuner
        failed = False
        try:
            with autotune(True):
                tactic = tuner.choose_one("op", [Split()], config, [torch.ones(2, 4)])[1]
            assert tactic == 2, tactic
        except _TuningGroupError as exc:
            failed = True
            assert "mismatch" in str(exc), str(exc)
        assert failed == (case == "mixed"), (rank, case)
        dist.barrier()
        print("OK", rank, case, flush=True)
finally:
    set_autotune_process_group(None)
    dist.destroy_process_group()
"""


def test_distributed_ranks_must_agree_on_mode(tmp_path):
    procs = [
        subprocess.Popen(
            [sys.executable, "-c", _DIST_WORKER],
            cwd=Path(__file__).resolve().parents[2],
            env={
                **os.environ,
                "CUDA_VISIBLE_DEVICES": "",
                "RANK": str(rank),
                "STORE": str(tmp_path / "rendezvous"),
                "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": "0",
            },
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        for rank in range(2)
    ]
    try:
        for rank, proc in enumerate(procs):
            out, _ = proc.communicate(timeout=120)
            assert proc.returncode == 0, f"rank {rank}:\n{out}"
            assert f"OK {rank} same" in out and f"OK {rank} mixed" in out, out
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
