"""Tactic-pool changes through real tuner/cache paths with controlled timings."""

import contextlib
import json

import pytest
import torch

from flashinfer.autotune_cache import autotune_v2
from flashinfer.autotuner import AutoTuner, TunableRunner, TuningConfig, autotune

pytestmark = pytest.mark.cpu_only


class PoolRunner(TunableRunner):
    def __init__(self, tactics):
        self.tactics = tactics

    def __hash__(self):
        # Enumeration changes without changing the runner's cache identity.
        return 0

    def get_tactic_cache_key(self, inputs):
        return tuple(self.tactics)

    def get_valid_tactics(self, inputs, profile):
        return list(self.tactics)

    def forward(self, inputs, tactic=-1):
        return inputs[0]


class OtherRunner(PoolRunner):
    def get_tactic_cache_key(self, inputs):
        return None


@pytest.fixture
def setup(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    for module in ("flashinfer.autotuner", "flashinfer.autotuner.autotuner"):
        monkeypatch.setattr(f"{module}._collect_metadata", lambda: {"test": "tactics"})
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path / "managed"))
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    calls = []

    def profile(self, runner, inputs, tactic, config, **kwargs):
        calls.append((type(runner), tactic))
        return {-1: 10.0, 1: 3.0, 2: 1.0, 3: 2.0}[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", profile)
    return tuner, calls


@contextlib.contextmanager
def tuning(mode):
    if mode == "managed":
        with autotune_v2():
            yield
    else:
        with autotune():
            if mode == "v2":
                with autotune_v2(persistent_cache=False):
                    yield
            else:
                yield


@pytest.mark.parametrize("mode", ["v1", "v2", "managed"])
@pytest.mark.parametrize("reload", [False, True])
@pytest.mark.parametrize("multiple", [False, True])
def test_tactic_expansion_retunes_and_preserves_warm_serving(
    setup, monkeypatch, tmp_path, mode, reload, multiple
):
    tuner, calls = setup
    first = PoolRunner([1])
    runners = [first, OtherRunner([3])] if multiple else [first]
    inputs, config = [torch.zeros(4, 8)], TuningConfig()
    with tuning(mode):
        assert tuner.choose_one("pool", runners, config, inputs)[1] == (
            3 if multiple else 1
        )

    path = tmp_path / "tactics.json"
    if reload:
        if mode != "managed":
            tuner.save_configs(str(path))
        tuner = AutoTuner()
        monkeypatch.setattr(AutoTuner, "_instance", tuner)
        if mode != "managed":
            assert tuner.load_configs(str(path))

    first.tactics.append(2)
    calls.clear()
    with tuning(mode):
        assert tuner.choose_one("pool", runners, config, inputs) == (first, 2)
        assert (PoolRunner, 2) in calls
        calls.clear()
        monkeypatch.setattr(
            tuner, "_prepare_input_tensors", lambda *args: pytest.fail("warm synthesis")
        )
        monkeypatch.setattr(
            PoolRunner,
            "get_valid_tactics",
            lambda *args: pytest.fail("warm enumeration"),
        )
        assert tuner.choose_one("pool", runners, config, inputs) == (first, 2)
        assert calls == []

    if reload:
        if mode != "managed":
            tuner.save_configs(str(path))
        tuner = AutoTuner()
        monkeypatch.setattr(AutoTuner, "_instance", tuner)
        if mode != "managed":
            assert tuner.load_configs(str(path))
        with tuning(mode):
            assert tuner.choose_one("pool", runners, config, inputs) == (first, 2)
        assert calls == []

    monkeypatch.setattr(
        PoolRunner,
        "get_tactic_cache_key",
        lambda *args: pytest.fail("serving signature"),
    )
    assert tuner.choose_one("pool", runners, config, inputs) == (first, 2)
    assert calls == []


@pytest.mark.parametrize("mode", ["v1", "managed"])
def test_legacy_coverage_replayed_then_refreshed(setup, monkeypatch, tmp_path, mode):
    tuner, calls = setup
    runner = PoolRunner([1])
    inputs, config = [torch.zeros(4, 8)], TuningConfig()
    with tuning(mode):
        assert tuner.choose_one("legacy", [runner], config, inputs) == (runner, 1)
    path = tmp_path / "legacy.json"
    if mode == "v1":
        tuner.save_configs(str(path))
        data = json.loads(path.read_text())
        for key, value in data.items():
            if not key.startswith("_"):
                data[key] = value[:2]
        path.write_text(json.dumps(data))
    else:
        for entry in (tmp_path / "managed").glob("v2/*/entries/*.json"):
            data = json.loads(entry.read_text())
            data.pop("candidate_keys", None)
            entry.write_text(json.dumps(data))
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    if mode == "v1":
        assert tuner.load_configs(str(path))
    else:
        with autotune_v2(mode="replay"):
            pass
    calls.clear()
    assert tuner.choose_one("legacy", [runner], config, inputs) == (runner, 1)
    assert calls == []
    with tuning(mode):
        assert tuner.choose_one("legacy", [runner], config, inputs) == (runner, 1)
    assert (PoolRunner, 1) in calls


def test_persisted_signature_text_is_canonical():
    from flashinfer.autotuner.autotuner import _ProfilingCandidates

    dtypes = ["fp16", "bf16", "fp8", "fp4", "int8", "mxfp8"]

    class SetRunner(PoolRunner):
        def get_tactic_cache_key(self, inputs):
            return (frozenset(dtypes), ("rev", 2, frozenset({3, 1})))

    keys = _ProfilingCandidates.from_runners([SetRunner([1])], [torch.zeros(1)])
    expected = (
        "('SetRunner', (), (frozenset({"
        + ", ".join(map(repr, sorted(dtypes)))
        + "}), ('rev', 2, frozenset({1, 3}))))"
    )
    # str(frozenset) follows the per-process hash seed; the persisted key must not.
    assert keys.file_keys == (expected,)


@pytest.mark.parametrize("mode", ["v1", "managed"])
def test_frozenset_and_tuple_signatures_stay_distinct_after_reload(
    setup, monkeypatch, tmp_path, mode
):
    tuner, calls = setup

    class TypedRunner(PoolRunner):
        def get_tactic_cache_key(self, inputs):
            return (self.signature,)

    runner = TypedRunner([1])
    runner.signature = (1, 2)
    inputs, config = [torch.zeros(4, 8)], TuningConfig()
    with tuning(mode):
        assert tuner.choose_one("typed", [runner], config, inputs) == (runner, 1)
    path = tmp_path / "typed.json"
    if mode == "v1":
        tuner.save_configs(str(path))
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    if mode == "v1":
        assert tuner.load_configs(str(path))

    # Equal items, different type: the persisted coverage must not match.
    runner.tactics.append(2)
    runner.signature = frozenset({1, 2})
    calls.clear()
    with tuning(mode):
        assert tuner.choose_one("typed", [runner], config, inputs) == (runner, 2)
    assert (TypedRunner, 2) in calls


@pytest.mark.parametrize("mode", ["v1", "managed"])
def test_signature_changed_by_preparation_retunes(setup, mode):
    tuner, calls = setup

    class GrowingRunner(PoolRunner):
        def forward(self, inputs, tactic=-1, do_preparation=False):
            if do_preparation and 2 not in self.tactics:
                # The pool grows after enumeration, so tactic 2 was not compared.
                self.tactics.append(2)
            return inputs[0]

    runner = GrowingRunner([1])
    inputs, config = [torch.zeros(4, 8)], TuningConfig()
    with tuning(mode):
        assert tuner.choose_one("growing", [runner], config, inputs) == (runner, 1)
    assert (GrowingRunner, 2) not in calls
    calls.clear()
    with tuning(mode):
        assert tuner.choose_one("growing", [runner], config, inputs) == (runner, 2)
    assert (GrowingRunner, 2) in calls
