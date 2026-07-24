"""
Tests for the managed v2 autotune cache (``autotune_v2()``).

These tests exercise the per-entry on-disk persistence backend in
``flashinfer/autotune_cache.py`` and its wiring inside the AutoTuner:
publish-on-tune, lazy per-entry lookup, environment-hash isolation, and
corrupt-entry tolerance.

No GPU is required — profiling is monkeypatched and inputs are CPU tensors.
"""

import json

import pytest
import torch

import flashinfer.autotune_cache as autotune_cache_module
from flashinfer.autotune_cache import (
    ManagedAutotuneCache,
    MeasurementPolicy,
    autotune_v2,
)
from flashinfer.autotuner import (
    AutoTuner,
    TuningConfig,
    autotune,
)

from .utils import DummyRunner, reset_autotuner


_OP = "test::autotune_cache_v2"
_CONFIG = TuningConfig()  # no dynamic specs -> a single static profile


def _fresh_process():
    """Simulate a new process: wipe all in-memory state INCLUDING the
    attached managed store (which by design survives reset_autotuner)."""
    tuner = reset_autotuner()
    tuner._managed_cache = None
    return tuner


@pytest.fixture
def cache_root(tmp_path, monkeypatch):
    """Point the managed cache at a fresh temp root and reset the tuner."""
    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    _fresh_process()
    yield tmp_path
    _fresh_process()


def _install_fake_profile(monkeypatch, times, record_configs=None):
    """Replace GPU profiling with a lookup table; returns the call log.
    Pass a list as *record_configs* to also capture each TuningConfig."""
    calls = []

    def fake_profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        calls.append(tactic)
        if record_configs is not None:
            record_configs.append(tuning_config)
        return float(times[tactic])

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    return calls


def _tune_once(monkeypatch, times, tactics=(0, 1, 2)):
    """Run one tuning pass under the managed cache; return (runner, tactic, calls)."""
    calls = _install_fake_profile(monkeypatch, times)
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        runner, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner(tactics)], _CONFIG, inputs
        )
    return runner, tactic, calls


def _entry_files(cache_root):
    return list(cache_root.glob("v2/*/entries/*.json"))


def test_tuning_publishes_one_entry_per_op(cache_root, monkeypatch):
    _, tactic, calls = _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    assert tactic == 1
    assert sorted(t for t in calls if t != -1) == [0, 1, 2]  # each tactic profiled once

    entries = _entry_files(cache_root)
    assert len(entries) == 1
    entry = json.loads(entries[0].read_text())
    assert entry["runner"] == "DummyRunner"
    assert entry["tactic"] == 1
    assert entry["key"].startswith(f"('{_OP}', 'DummyRunner', ")

    # The environment manifest is human-readable alongside the entries.
    manifest_files = list(cache_root.glob("v2/*/manifest.json"))
    assert len(manifest_files) == 1
    manifest = json.loads(manifest_files[0].read_text())
    assert manifest["cache_schema"] == "v2"
    assert "flashinfer_version" in manifest

    # Atomic publication leaves no temp files behind.
    assert list(cache_root.glob("v2/*/entries/*.tmp")) == []


def test_fresh_process_reuses_entry_without_profiling(cache_root, monkeypatch):
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    # Simulate a new process: wipe all in-memory state.
    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert calls == []  # served from disk, no profiling


def test_tuning_mode_also_skips_already_tuned_entries(cache_root, monkeypatch):
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert calls == []  # disk hit short-circuits re-profiling during tuning


def test_corrupt_entry_is_a_miss_and_retuned(cache_root, monkeypatch):
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    (entry_file,) = _entry_files(cache_root)
    entry_file.write_text("this is not json")

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    # Lookup-only mode: corrupt entry -> fallback tactic, no crash.
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1
    assert calls == []

    # Tuning mode: corrupt entry -> retune and republish a valid entry.
    _fresh_process()
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert sorted(t for t in calls if t != -1) == [0, 1, 2]
    entry = json.loads(entry_file.read_text())
    assert entry["tactic"] == 1


def test_embedded_key_mismatch_is_a_miss(cache_root, monkeypatch):
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    (entry_file,) = _entry_files(cache_root)
    entry = json.loads(entry_file.read_text())
    entry["key"] = "('someone::else', 'DummyRunner', ((8, 16),), ())"
    entry_file.write_text(json.dumps(entry))

    _fresh_process()
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1


def test_environment_hash_isolates_entries(cache_root, monkeypatch):
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    # Same op, "different machine": entries must not be visible.
    monkeypatch.setattr(
        "flashinfer.autotuner._collect_metadata",
        lambda: {"gpu": "NVIDIA Different GPU", "flashinfer_version": "x"},
    )
    _fresh_process()
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1  # old env's entry is invisible here

    # Both environment directories coexist; neither was deleted.
    with autotune_v2():
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    assert len(list(cache_root.glob("v2/*"))) == 2
    assert len(_entry_files(cache_root)) == 2


def test_compound_tuple_tactic_roundtrip(cache_root, monkeypatch):
    tactics = ((128, (1, 2)), (256, (2, 1)))
    times = {tactics[0]: 2.0, tactics[1]: 1.0}
    _, tactic, _ = _tune_once(monkeypatch, times, tactics=tactics)
    assert tactic == (256, (2, 1))

    _fresh_process()
    _install_fake_profile(monkeypatch, times)
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner(tactics)], _CONFIG, [torch.zeros(8, 16)]
        )
    # Tuples survive the JSON round-trip as tuples (not lists).
    assert tactic == (256, (2, 1))
    assert isinstance(tactic, tuple)


def test_publish_failure_does_not_break_tuning(cache_root, monkeypatch):
    def broken_replace(src, dst):
        raise OSError("simulated disk failure")

    monkeypatch.setattr(autotune_cache_module.os, "replace", broken_replace)
    _, tactic, _ = _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    assert tactic == 1  # tuning result still usable in-process
    assert _entry_files(cache_root) == []


def test_legacy_v1_path_does_not_create_v2_dirs(cache_root, monkeypatch, tmp_path):
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    v1_path = tmp_path / "legacy_configs.json"  # PathLike: works as before
    with autotune(True, cache=v1_path):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert v1_path.is_file()  # v1 behavior unchanged
    assert list(cache_root.glob("v2")) == []  # no managed dirs created


def test_lookup_negative_memo_and_publish_clears_it(cache_root):
    cache = ManagedAutotuneCache(manifest={"gpu": "test"})
    key = "('op', 'Runner', ((1,),), ())"
    assert cache.lookup(key) is None
    assert key in cache._missing
    cache.publish(key, "Runner", 7)
    assert cache.lookup(key) == ("Runner", 7)


def test_explicit_root_directory(cache_root, monkeypatch, tmp_path):
    """root=<dir> relocates the store; entries land below it, not the default."""
    custom_root = tmp_path / "explicit" / "fi-autotune"
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(cache_root=str(custom_root)):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert len(list(custom_root.glob("v2/*/entries/*.json"))) == 1
    assert _entry_files(cache_root) == []  # default root untouched

    # A fresh process pointing at the same root reuses the entry.
    _fresh_process()
    calls.clear()
    with autotune_v2(mode="replay", cache_root=custom_root):  # os.PathLike form
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert calls == []


def test_persist_false_disables_disk(cache_root, monkeypatch):
    """persistent_cache=False tunes normally but touches no filesystem."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(persistent_cache=False):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert list(cache_root.glob("v2")) == []


def test_v2_disk_entries_cannot_leak_into_v1_file(cache_root, monkeypatch, tmp_path):
    """No mixing by construction: winners tuned inside a v2 context belong
    to the v2 identity (its store partition + its store on disk), so a
    nested v1 save_configs can never write them — or any v2-store entry —
    into the user's v1 JSON file."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})  # _OP -> v2 disk

    # Fresh "process": _OP is served from the v2 store, then a nested v1
    # context tunes a different op and saves its own file.
    _fresh_process()
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    v1_path = tmp_path / "user.json"
    other_op = "test::v1_only_op"
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
        assert tactic == 1  # served from the v2 store
        with autotune(True, cache=str(v1_path)):
            AutoTuner.get().choose_one(
                other_op, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
            )
    keys = [k for k in json.loads(v1_path.read_text()) if not k.startswith("_")]
    # The nested tuning ran under the v2 context's identity (its policy
    # governed the measurement), so its winner persists in the v2 store,
    # not the v1 file.
    assert not any(other_op in k for k in keys)
    assert not any(_OP in k for k in keys)  # v2-store entry never leaked in
    assert any(
        other_op in json.loads(e.read_text())["key"] for e in _entry_files(cache_root)
    )


def test_serving_after_context_exit_reuses_entries(cache_root, monkeypatch):
    """Attach semantics, sglang pattern: a fresh process re-enters the
    tuning context with a warm cache (nothing to profile), then serves
    OUTSIDE any context — lookups must still hit the on-disk entries."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2():  # warm cache: disk hit, no profiling
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    # Serving outside any context (how vLLM/sglang actually run).
    _, tactic = AutoTuner.get().choose_one(
        _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
    )
    assert tactic == 1
    assert calls == []


def test_hydrate_only_context_enables_reuse(cache_root, monkeypatch):
    """vLLM pattern: mode="replay" + persistent_cache=True is a pure
    hydrate step; serving outside the context reuses tuned winners."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        pass  # attach only; no calls inside
    _, tactic = AutoTuner.get().choose_one(
        _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
    )
    assert tactic == 1
    assert calls == []


def test_attach_is_idempotent_and_root_switch_replaces(cache_root, tmp_path):
    tuner = AutoTuner.get()
    with autotune_v2():
        first = tuner._managed_cache
        assert first is not None
    assert tuner._managed_cache is first  # stays attached after exit
    with autotune_v2():  # same (default) root: same store, memos kept
        assert tuner._managed_cache is first
    other_root = tmp_path / "other"
    with autotune_v2(cache_root=other_root):
        assert tuner._managed_cache is not first
        assert tuner._managed_cache.root == other_root


def test_entry_failure_does_not_leak_tuning_mode(cache_root):
    """An exception while entering the delegated autotune() context must
    not leave tuning mode active (the store attach is ambient by design)."""
    tuner = AutoTuner.get()
    # Mixed int/str buckets raise TypeError inside the context setup.
    with (
        pytest.raises(TypeError),
        autotune_v2(tuning_buckets=(64, "128")),
    ):
        pass
    assert tuner.is_tuning_mode is False


def test_measure_policy_overrides_profiling_config(cache_root, monkeypatch):
    """MeasurementPolicy forces cuda_graph/cold_l2 during profiling; None
    fields inherit the op's TuningConfig."""
    seen = []
    _install_fake_profile(
        monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0}, record_configs=seen
    )
    # The op's own config has both flags False (TuningConfig defaults).
    with autotune_v2(measure=MeasurementPolicy(execution_mode="cuda_graph")):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert seen and all(c.use_cuda_graph is True for c in seen)
    assert all(c.use_cold_l2_cache is False for c in seen)  # inherited

    _fresh_process()
    seen2 = []
    _install_fake_profile(
        monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0}, record_configs=seen2
    )
    with autotune_v2(persistent_cache=False, measure=MeasurementPolicy(cold_l2=True)):
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    assert seen2 and all(c.use_cold_l2_cache is True for c in seen2)
    assert all(c.use_cuda_graph is False for c in seen2)  # inherited

    # Outside any policy context the original config is used untouched.
    _fresh_process()
    seen3 = []
    _install_fake_profile(
        monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0}, record_configs=seen3
    )
    with autotune_v2(persistent_cache=False):
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    assert seen3 and all(c is _CONFIG for c in seen3)


def test_measure_policy_isolates_store_identity(cache_root, monkeypatch):
    """Entries tuned under different measurement policies live in different
    environment directories and never overwrite each other."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})  # default policy
    assert len(list(cache_root.glob("v2/*"))) == 1

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    policy = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=True)
    with autotune_v2(measure=policy):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert sorted(t for t in calls if t != -1) == [
        0,
        1,
        2,
    ]  # default-policy entry was NOT visible here
    assert len(list(cache_root.glob("v2/*"))) == 2  # two env dirs coexist
    assert len(_entry_files(cache_root)) == 2

    # The policy is recorded, human-readable, in the manifest.
    manifests = [
        json.loads(p.read_text()) for p in cache_root.glob("v2/*/manifest.json")
    ]
    assert any(m.get("measure_execution_mode") == "cuda_graph" for m in manifests)
    assert any("measure_execution_mode" not in m for m in manifests)


def test_measure_policy_attach_carries_to_serving(cache_root, monkeypatch):
    """A consuming process attaches with the same policy and reuses the
    policy-tuned entries after the context exits."""
    policy = MeasurementPolicy(execution_mode="cuda_graph")
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(measure=policy):
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay", measure=policy):
        pass  # hydrate with the matching policy
    _, tactic = AutoTuner.get().choose_one(
        _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
    )
    assert tactic == 1
    assert calls == []

    # Hydrating WITHOUT the policy points at the default-policy directory:
    # the policy-tuned entry is invisible there.
    _fresh_process()
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1


def test_measure_policy_timer_validation(cache_root):
    with pytest.raises(ValueError, match="timer"):
        MeasurementPolicy(_timer="bogus")
    # All supported timers construct, and explicit ones are identity-bearing.
    for timer in ("auto", "cupti", "events", "events_no_delay"):
        policy = MeasurementPolicy(_timer=timer)
        fields = policy.manifest_fields()
        assert ("measure_timer" in fields) == (timer != "auto")


def test_measure_policy_execution_mode_resolution(cache_root):
    """execution_mode is the primary axis; cuda_graph is derived, not a field."""
    g = MeasurementPolicy(execution_mode="cuda_graph")
    assert g.cuda_graph is True  # derived property
    assert g.timer == "events"  # host-excluded implementation, auto-selected
    assert g.manifest_fields()["measure_execution_mode"] == "cuda_graph"

    e = MeasurementPolicy(execution_mode="eager")
    assert e.cuda_graph is False
    assert e.timer == "events_no_delay"
    assert e.manifest_fields()["measure_execution_mode"] == "eager"

    unset = MeasurementPolicy()
    assert unset.execution_mode == "auto"  # explicit library-decides default
    assert unset.cuda_graph is None  # auto: inherit each op's TuningConfig
    assert unset.timer == "events"  # auto: today's host-excluded resolution

    # Expert timer override within an execution mode, when not contradictory.
    assert MeasurementPolicy(execution_mode="eager", _timer="events").timer == "events"
    assert (
        MeasurementPolicy(execution_mode="cuda_graph", _timer="cupti").timer == "cupti"
    )

    # Contradictions / removed knobs are loud errors, never silent.
    with pytest.raises(ValueError, match="cupti"):
        MeasurementPolicy(execution_mode="eager", _timer="cupti")
    with pytest.raises(ValueError, match="execution_mode"):
        MeasurementPolicy(execution_mode="bogus")


def test_measure_policy_timer_in_manifest(cache_root, monkeypatch):
    """An explicit timer is part of the store's environment identity."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(measure=MeasurementPolicy(_timer="cupti")):
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    manifests = [
        json.loads(p.read_text()) for p in cache_root.glob("v2/*/manifest.json")
    ]
    assert any(m.get("measure_timer") == "cupti" for m in manifests)


def test_measure_policy_all_none_is_default_identity(cache_root, monkeypatch):
    """A MeasurementPolicy with all-None fields does not fragment the store."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay", measure=MeasurementPolicy()):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1  # same env dir as the default-policy tuning
    assert calls == []


def test_reattach_clears_decoded_memo(cache_root, monkeypatch):
    """Entries decoded under one store must not be served under a different
    store's identity: re-attaching (e.g. a new measurement policy) drops the
    decode memo, and the new (empty) env misses instead of replaying store
    A's winner."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    tuner = _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    assert tuner._managed_decoded  # memo hydrated from store A

    with autotune_v2(mode="replay", measure=MeasurementPolicy(execution_mode="eager")):
        assert not tuner._managed_decoded  # re-attach dropped store A's memo
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1  # store B (eager env) is empty: miss, not A's winner
    assert calls == []  # mode="replay": no profiling either


def test_policy_switch_reprofiles_in_process(cache_root, monkeypatch):
    """P1 (codex review): winners tuned under one measurement policy must not
    short-circuit tuning under another in the same process — the in-memory
    winner cache is partitioned by measurement identity, not just the disk."""
    calls = []

    def fake_profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        policy = self._effective_measure_policy
        mode = policy.execution_mode if policy is not None else "auto"
        calls.append((mode, tactic))
        # Under eager measurement tactic 1 wins; otherwise tactic 0 wins.
        times = {0: 1.0, 1: 2.0, 2: 3.0}
        if mode == "eager":
            times = {0: 2.0, 1: 1.0, 2: 3.0}
        return times[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    inputs = [torch.zeros(8, 16)]

    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 0

    with autotune_v2(measure=MeasurementPolicy(execution_mode="eager")):
        _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 1  # re-profiled under eager, not served tactic 0 from memory
    assert ("eager", 0) in calls  # eager tuning really ran

    # Both identities published to their own env dirs.
    entries = _entry_files(cache_root)
    assert len(entries) == 2
    assert len({e.parent.parent.name for e in entries}) == 2


def test_persistent_false_context_never_touches_disk(cache_root, monkeypatch):
    """P1 (codex review): persistent_cache=False forbids disk for the context
    even when an ambient store was attached earlier in the process."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})  # attaches ambient
    before = {p: p.read_bytes() for p in _entry_files(cache_root)}

    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    other_op = _OP + "::inmem"
    with autotune_v2(persistent_cache=False):
        _, tactic = AutoTuner.get().choose_one(
            other_op, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1 and sorted(t for t in calls if t != -1) == [
        0,
        1,
        2,
    ]  # tuned (ambient disk hit forbidden too)

    after = {p: p.read_bytes() for p in _entry_files(cache_root)}
    assert after == before  # no new entries, no rewrites

    # The ambient store still serves outside the context.
    tuner = AutoTuner.get()
    assert tuner._managed_cache is not None
    _, tactic = tuner.choose_one(_OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)])
    assert tactic == 1


def test_nested_contexts_publish_to_their_own_store(cache_root, monkeypatch):
    """P1 (codex review): after an inner context with a different identity
    exits, the outer context's publishes go to the OUTER store, not the
    inner one's manifest."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    inner_pol = MeasurementPolicy(execution_mode="eager")
    with autotune_v2():
        with autotune_v2(measure=inner_pol):
            pass  # inner attaches its own env, then exits
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == 1
    entries = _entry_files(cache_root)
    assert len(entries) == 1
    manifest = json.loads((entries[0].parent.parent / "manifest.json").read_text())
    # Published under the outer (default-policy) identity: no eager marker.
    assert "measure_execution_mode" not in manifest


def test_default_path_races_and_can_win(cache_root, monkeypatch):
    """Regression guard: (runners[0], -1) always races in v2 contexts, so a
    persisted selection can never be slower than not tuning (#3537/#3622)."""
    calls = _install_fake_profile(monkeypatch, times={-1: 0.5, 0: 1.0, 1: 2.0, 2: 3.0})
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1  # the default beat every tuned tactic
    assert -1 in calls  # and it was actually measured
    (entry,) = _entry_files(cache_root)
    assert json.loads(entry.read_text())["tactic"] == -1

    # Fresh process: the persisted -1 replays with zero profiling.
    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={-1: 0.5, 0: 1.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1 and calls == []


def test_v1_selection_does_not_race_default(monkeypatch):
    """Plain autotune() selection stays byte-identical: no -1 candidate."""
    calls = _install_fake_profile(monkeypatch, times={-1: 0.5, 0: 1.0, 1: 2.0, 2: 3.0})
    reset_autotuner()
    with autotune(True):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert -1 not in calls
    assert tactic == 0
    reset_autotuner()


def test_reload_converges_ranks_on_store_state(cache_root, monkeypatch):
    """autotune_v2_reload(): after tuning, dropping in-memory winners makes
    this rank serve the store's canonical entry (simulating another rank's
    later publish winning last-write-wins)."""
    from flashinfer.autotune_cache import autotune_v2_reload

    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})  # local winner: 1
    (entry_file,) = _entry_files(cache_root)
    entry = json.loads(entry_file.read_text())
    entry["tactic"] = 2  # another rank's publish landed last
    entry_file.write_text(json.dumps(entry))

    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    _, tactic = AutoTuner.get().choose_one(
        _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
    )
    assert tactic == 1  # still the locally-measured winner (memoized)

    autotune_v2_reload()
    _, tactic = AutoTuner.get().choose_one(
        _OP, [DummyRunner()], _CONFIG, [torch.zeros(8, 16)]
    )
    assert tactic == 2  # converged on the store's canonical entry
    assert calls == []  # reload never re-profiles


class DummyRunnerB(DummyRunner):
    pass


def test_runner_reorder_replays_correct_runner(cache_root, monkeypatch):
    """Persisted entries key on the runner class, so reordering the runner
    list across processes replays the winner on the RIGHT runner."""
    times = {0: 2.0, 1: 1.0, 2: 3.0}
    _install_fake_profile(monkeypatch, times)
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        runner, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunner((0,)), DummyRunnerB((0, 1, 2))], _CONFIG, inputs
        )
    assert isinstance(runner, DummyRunnerB) and tactic == 1

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times)
    with autotune_v2(mode="replay"):
        runner, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunnerB((0, 1, 2)), DummyRunner((0,))], _CONFIG, inputs
        )
    assert isinstance(runner, DummyRunnerB) and tactic == 1
    assert calls == []


class DummyRunnerWithExtras(DummyRunner):
    def __init__(self, extras, valid_tactics=(0, 1, 2)):
        super().__init__(valid_tactics)
        self._extras = extras

    def get_cache_key_extras(self, inputs):
        return self._extras


def test_extras_distinguish_entries(cache_root, monkeypatch):
    """Runner extras are part of the persisted key: same class, different
    extras -> distinct entries (the vllm#43119 key-completeness class)."""
    _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        AutoTuner.get().choose_one(
            _OP, [DummyRunnerWithExtras(("layout_a",))], _CONFIG, inputs
        )
        AutoTuner.get().choose_one(
            _OP, [DummyRunnerWithExtras(("layout_b",))], _CONFIG, inputs
        )
    entries = _entry_files(cache_root)
    assert len(entries) == 2
    keys = {json.loads(e.read_text())["key"] for e in entries}
    assert any("layout_a" in k for k in keys) and any("layout_b" in k for k in keys)


class DummyRunnerRejecting(DummyRunner):
    def validate_tactic(self, inputs, tactic):
        return False


def test_invalid_cached_tactic_is_a_loud_miss(cache_root, monkeypatch):
    """Runner-contract revalidation: a stored tactic the runner rejects is a
    cache miss (retune or fallback), never a blind replay (#3566 class)."""
    _tune_once(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})

    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2(mode="replay"):
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunnerRejecting()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert tactic == -1  # loud fallback, not the invalid stored tactic
    assert calls == []

    # With tuning enabled the op is re-profiled instead.
    _fresh_process()
    calls = _install_fake_profile(monkeypatch, times={0: 3.0, 1: 1.0, 2: 2.0})
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(
            _OP, [DummyRunnerRejecting()], _CONFIG, [torch.zeros(8, 16)]
        )
    assert len(calls) > 0


def _policy_dependent_profile(monkeypatch):
    """Fake profiler where the eager-measured winner differs from default."""
    calls = []

    def fake_profile(self, runner, inputs, tactic, tuning_config, **kwargs):
        policy = self._effective_measure_policy
        mode = policy.execution_mode if policy is not None else "auto"
        calls.append((mode, tactic))
        times = {-1: 5.0, 0: 1.0, 1: 2.0, 2: 3.0}
        if mode == "eager":
            times = {-1: 5.0, 0: 2.0, 1: 1.0, 2: 3.0}
        return times[tactic]

    monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)
    return calls


def test_bare_serving_after_policy_context_uses_ambient_identity(
    cache_root, monkeypatch
):
    """P1 (codex round 3): after an eager-policy context exits, bare serving
    must serve the ambient (eager) identity's winner — never a winner tuned
    under a different identity that sits in legacy memory."""
    _policy_dependent_profile(monkeypatch)
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 0  # default-identity winner

    with autotune_v2(measure=MeasurementPolicy(execution_mode="eager")):
        _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 1  # eager-identity winner

    # Bare call: ambient store is the eager one -> must serve 1, not 0.
    _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 1


def test_cache_root_switch_repopulates(cache_root, monkeypatch, tmp_path):
    """P1 (codex round 3): switching cache_root must not silently reuse the
    old root's in-memory winners — the new root gets populated."""
    _install_fake_profile(monkeypatch, times={-1: 5.0, 0: 3.0, 1: 1.0, 2: 2.0})
    inputs = [torch.zeros(8, 16)]
    root_b = tmp_path / "root_b"
    with autotune_v2():
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert len(_entry_files(cache_root)) == 1

    with autotune_v2(cache_root=root_b):
        _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 1
    assert len(list(root_b.glob("v2/*/entries/*.json"))) == 1  # B populated


class DummyRunnerFlippable(DummyRunner):
    def __init__(self, valid_tactics=(0, 1, 2)):
        super().__init__(valid_tactics)
        self.reject = False

    def validate_tactic(self, inputs, tactic):
        return not self.reject


def test_in_memory_winner_is_revalidated_same_process(cache_root, monkeypatch):
    """P1 (codex round 3): a winner tuned earlier in THIS process is also
    revalidated — a runtime shape the runner now rejects becomes a loud
    fallback, not a blind replay (#3566 without a restart)."""
    _install_fake_profile(monkeypatch, times={-1: 5.0, 0: 3.0, 1: 1.0, 2: 2.0})
    runner = DummyRunnerFlippable()
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        _, tactic = AutoTuner.get().choose_one(_OP, [runner], _CONFIG, inputs)
    assert tactic == 1

    runner.reject = True  # e.g. a runtime shape this plan cannot serve
    _, tactic = AutoTuner.get().choose_one(_OP, [runner], _CONFIG, inputs)
    assert tactic == -1  # memory AND disk hits rejected -> clean fallback


def test_invalid_mode_raises(cache_root):
    # common mistake -> clear error
    with (
        pytest.raises(ValueError, match="mode must be 'tune' or 'replay'"),
        autotune_v2(mode="serve"),
    ):
        pass
    # the old boolean no longer works
    with pytest.raises(ValueError), autotune_v2(mode=False):
        pass


def test_nested_context_does_not_clobber_ambient(cache_root, monkeypatch):
    """Scope-guard hygiene: a nested (scoped) context is an override for its
    region only — it must NOT rebind the process ambient set by the outer
    (warmup) context, so bare calls after the inner exits keep the outer
    identity."""
    tuner = _fresh_process()
    _policy_dependent_profile(monkeypatch)
    inputs = [torch.zeros(8, 16)]

    # Outer top-level context tunes under the default identity -> ambient.
    with autotune_v2():
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
        outer_ambient = tuner._managed_cache
        # Inner nested context with a DIFFERENT (eager) identity.
        with autotune_v2(measure=MeasurementPolicy(execution_mode="eager")):
            AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
            # In-region lookup targets the inner store (top of stack)...
            assert tuner._active_managed_store.env_hash != outer_ambient.env_hash
        # ...but the ambient default is unchanged by the scoped override.
        assert tuner._managed_cache is outer_ambient

    # After everything exits, bare serving uses the outer (default) identity.
    _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 0  # default-identity winner, not the eager one


def test_sequential_top_level_contexts_last_wins_ambient(cache_root, monkeypatch):
    """Sequential top-level contexts each re-attach: the ambient is last-wins
    (an explicit re-attach IS allowed; only *nested* overrides are scoped)."""
    _policy_dependent_profile(monkeypatch)
    inputs = [torch.zeros(8, 16)]
    with autotune_v2():
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    with autotune_v2(measure=MeasurementPolicy(execution_mode="eager")):
        AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    _, tactic = AutoTuner.get().choose_one(_OP, [DummyRunner()], _CONFIG, inputs)
    assert tactic == 1  # eager winner: last top-level attach won the ambient
