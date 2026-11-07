"""Tests for exclude-split-K autotuning (set_autotune_exclude_split_k).

No GPU is required: profiling is faked and tensors live on the CPU.
"""

import json
import logging
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
    get_autotune_exclude_split_k,
    set_autotune_exclude_split_k,
)

from .test_autotune_cache_scope import _CONFIG, _GRAPH, _OP
from .utils import DummyRunner, make_scoped_tuner, reset_autotuner

# Tactic 1 is the fastest but splits the reduction (e.g. split-K).
TIMES = {0: 5.0, 1: 1.0, 2: 3.0}


class SplitRunner(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def is_split_free_tactic(self, tactic):
        return tactic != 1

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


def _mode(enabled):
    saved = get_autotune_exclude_split_k()
    set_autotune_exclude_split_k(enabled)
    try:
        yield
    finally:
        set_autotune_exclude_split_k(saved)


@pytest.fixture
def exclude_split_k():
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

    assert Plain().is_split_free_tactic((128, 32, 4))
    assert Plain().is_split_free_tactic(-1)


class Unclassified(TunableRunner):
    """Keeps the base-class default: its tactics are not constrained."""

    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(record)


@pytest.fixture
def tuner_warnings():
    """WARNING records of the autotuner logger (a standalone Logger that does
    not propagate to the root, so caplog does not see it)."""
    handler = _RecordingHandler()
    handler.setLevel(logging.WARNING)
    autotuner_module.logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        autotuner_module.logger.removeHandler(handler)


def _unclassified_warnings(records):
    return [
        r.getMessage()
        for r in records
        if "does not classify its tactics" in r.getMessage()
    ]


def test_unclassified_runner_warns_once_per_class(
    fake_profile, exclude_split_k, tuner_warnings
):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        # Two ops, two instances of the same class: one warning. The runner's
        # tactics are unconstrained, so the split tactic 1 still wins.
        for op in ("uncl_a", "uncl_b"):
            _, tactic = tuner.choose_one(
                op, [Unclassified()], TuningConfig(), _inputs()
            )
            assert tactic == 1
        # A classified runner never warns.
        tuner.choose_one("uncl_c", [SplitRunner()], TuningConfig(), _inputs())
    warnings = _unclassified_warnings(tuner_warnings)
    assert len(warnings) == 1, [r.getMessage() for r in tuner_warnings]
    assert "Unclassified" in warnings[0]
    assert "uncl_a" in warnings[0]
    assert "SplitRunner" not in warnings[0]
    # clear_cache resets the once-per-class memory.
    tuner_warnings.clear()
    tuner.clear_cache()
    with autotune(tune_mode=True):
        tuner.choose_one("uncl_d", [Unclassified()], TuningConfig(), _inputs())
    assert len(_unclassified_warnings(tuner_warnings)) == 1


def test_unclassified_runner_does_not_warn_without_mode(
    fake_profile, default_mode, tuner_warnings
):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        tuner.choose_one("uncl_off", [Unclassified()], TuningConfig(), _inputs())
    assert not _unclassified_warnings(tuner_warnings)


def test_tuning_picks_fastest_tactic_by_default(fake_profile, default_mode):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        _, tactic = tuner.choose_one(
            "bi_off", [SplitRunner()], TuningConfig(), _inputs()
        )
    assert tactic == 1


def test_tuning_skips_split_k_tactics(fake_profile, exclude_split_k):
    tuner = reset_autotuner()
    with autotune(tune_mode=True):
        _, tactic = tuner.choose_one(
            "bi_on", [SplitRunner()], TuningConfig(), _inputs()
        )
    assert tactic == 2
    assert 1 not in fake_profile


def test_cached_split_k_tactic_is_a_miss(exclude_split_k):
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


class ShapedRunner(SplitRunner):
    """Classifies by a tuple slot, like the MXFP8 runner, so a tactic of the
    wrong shape (an edited cache file, a stale format) raises."""

    def is_split_free_tactic(self, tactic):
        if tactic == -1:
            return True
        return int(tactic[4]) == 1


def test_malformed_cached_tactic_is_a_miss(exclude_split_k):
    tuner = reset_autotuner()
    runner = ShapedRunner()
    inputs = _inputs()
    config = TuningConfig()
    key = AutoTuner._get_cache_key("bi_malformed", runner, (inputs[0].shape,), config)
    for cached in ((1, 2), 7, None):  # IndexError, TypeError, TypeError
        tuner.profiling_cache[key] = (cached, None)
        assert tuner.choose_one("bi_malformed", [runner], config, inputs)[1] == -1


def test_cached_split_k_tactic_is_reprofiled(fake_profile, exclude_split_k):
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    key = AutoTuner._get_cache_key("bi_retune", runner, (inputs[0].shape,), config)
    tuner.profiling_cache[key] = (1, None)
    with autotune(tune_mode=True):
        assert tuner.choose_one("bi_retune", [runner], config, inputs)[1] == 2


def test_file_config_with_split_k_tactic_is_ignored(
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
    saved = get_autotune_exclude_split_k()
    set_autotune_exclude_split_k(True)
    try:
        assert not fresh.search_cache("bi_file", [runner], shapes, config, inputs)[0]
        assert fresh.choose_one("bi_file", [runner], config, inputs)[1] == -1
    finally:
        set_autotune_exclude_split_k(saved)


def test_ranked_shortlist_cached_before_mode_is_reranked(fake_profile, default_mode):
    tuner = reset_autotuner()
    runner = SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [1, 2]
    fake_profile.clear()
    set_autotune_exclude_split_k(True)
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [2, 0]
    assert sorted(fake_profile) == [0, 2]
    # The re-ranked shortlist is split-free, so it is served from the cache.
    fake_profile.clear()
    with autotune(tune_mode=True):
        assert tuner.rank_tactics("bi_rank", [runner], config, inputs, k=2) == [2, 0]
    assert not fake_profile


class SplitDefault(SplitRunner):
    """Its untuned default splits the reduction (e.g. SM107 warp split-K)."""

    def is_split_free_tactic(self, tactic):
        return tactic not in (-1, 1)


def test_untuned_call_uses_first_split_free_fallback_runner(default_mode):
    tuner = reset_autotuner()
    split, plain = SplitDefault(), SplitRunner()
    inputs = _inputs()
    config = TuningConfig()
    assert tuner.choose_one("bi_fb", [split, plain], config, inputs) == (split, -1)
    set_autotune_exclude_split_k(True)
    assert tuner.choose_one("bi_fb", [split, plain], config, inputs) == (plain, -1)
    with autotune(tune_mode=False, skip_ops={"bi_fb"}):
        assert tuner.choose_one("bi_fb", [split, plain], config, inputs) == (
            plain,
            -1,
        )
    with pytest.raises(RuntimeError, match="split-free fallback"):
        tuner.choose_one("bi_fb", [split], config, inputs)


# --- autotune_v2 integration: managed cache, replay coverage, default race ---


class _NoTwo(DummyRunner):
    """Graph-timed winner of make_scoped_tuner is tactic 2; call it split-K."""

    def is_split_free_tactic(self, tactic):
        return tactic != 2


class _NoTwoNoDefault(_NoTwo):
    """The untuned default (-1) splits the reduction too."""

    def is_split_free_tactic(self, tactic):
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


def test_managed_split_k_winner_is_a_reported_miss(tmp_path, monkeypatch, default_mode):
    tuner, calls = make_scoped_tuner(tmp_path, monkeypatch)
    runner = _NoTwo()
    assert _v2_tune(tuner, runner) == 2
    tuner.clear_cache()
    calls.clear()
    set_autotune_exclude_split_k(True)
    report = ReplayCoverage()
    with _v2_replay(replay_report=report):
        assert _v2_choose(tuner, runner) == -1
    assert (report.hits, report.misses) == (0, 1)
    assert report.records[0]["reasons"] == ("excluded_split_k",)
    # Strict replay rejects it like any other miss, without profiling.
    with (
        pytest.raises(RuntimeError, match="excluded_split_k"),
        _v2_replay(strict_replay=True),
    ):
        _v2_choose(tuner, runner)
    assert not calls
    # Without the mode the same store entry is still a hit.
    set_autotune_exclude_split_k(False)
    tuner.clear_cache()
    with _v2_replay(strict_replay=True) as report:
        assert _v2_choose(tuner, runner) == 2
    assert report.hits == 1


def test_retune_under_mode_publishes_split_free_winner(
    tmp_path, monkeypatch, default_mode
):
    tuner, calls = make_scoped_tuner(tmp_path, monkeypatch)
    runner = _NoTwo()
    assert _v2_tune(tuner, runner) == 2
    tuner.clear_cache()
    calls.clear()
    set_autotune_exclude_split_k(True)
    assert _v2_tune(tuner, runner) == 0
    assert all(t != 2 for _, t in calls)
    tuner.clear_cache()
    with _v2_replay(strict_replay=True) as report:
        assert _v2_choose(tuner, runner) == 0
    assert report.hits == 1


def test_v2_default_race_skips_split_k_default(tmp_path, monkeypatch, exclude_split_k):
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
    set_autotune_exclude_split_k, set_autotune_process_group,
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
    def is_split_free_tactic(self, tactic):
        return tactic != 1
    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]

class Plain(Split):
    # Every candidate is split-free: the mode leaves the plan unchanged.
    def get_valid_tactics(self, inputs, profile):
        return [0, 2]

def profile(self, runner, inputs, tactic, config, **kwargs):
    t = torch.tensor([{0: 5.0, 1: 1.0, 2: 3.0}[tactic]], dtype=torch.float64)
    dist.all_reduce(t)
    return t.item() / 2

AutoTuner._profile_single_kernel = profile
config = TuningConfig(use_cold_l2_cache=False)
try:
    for case, modes, runner in (
        ("same", (True, True), Split),
        ("mixed", (False, True), Split),
        ("mixed_plain", (False, True), Plain),
    ):
        set_autotune_exclude_split_k(modes[rank])
        tuner = AutoTuner()
        AutoTuner._instance = tuner
        failed = False
        try:
            with autotune(True):
                tactic = tuner.choose_one(case, [runner()], config, [torch.ones(2, 4)])[1]
            assert tactic == 2, tactic
        except _TuningGroupError as exc:
            failed = True
            assert "mismatch" in str(exc), str(exc)
        assert failed == case.startswith("mixed"), (rank, case)
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
            for case in ("same", "mixed", "mixed_plain"):
                assert f"OK {rank} {case}\n" in out, out
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.communicate()
