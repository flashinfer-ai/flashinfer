# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Mixed eager/graph ownership without changing process attachment."""

import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch

from flashinfer.autotune_cache import MeasurementPolicy, autotune_v2, autotune_v2_reload
from flashinfer.autotuner import AutoTuner, TuningConfig

from .utils import DummyRunner, make_scoped_tuner


_OP = "test::scoped_cache"
_CONFIG = TuningConfig()
_GRAPH = MeasurementPolicy(execution_mode="cuda_graph", cold_l2=False)
_EAGER = MeasurementPolicy(execution_mode="eager", cold_l2=False)


@pytest.fixture
def scoped(tmp_path, monkeypatch):
    return make_scoped_tuner(tmp_path, monkeypatch)


def _choose(tuner, config=_CONFIG):
    return tuner.choose_one(_OP, [DummyRunner()], config, [torch.ones(2)])[1]


def _load_legacy(tuner, tmp_path):
    """Use public save/load to install a distinct legacy winner."""
    runner = DummyRunner()
    key = tuner._get_cache_key(_OP, runner, ((2,),), _CONFIG)
    tuner.profiling_cache[key] = (0, None)
    path = tmp_path / "legacy.json"
    tuner.save_configs(str(path))
    tuner.clear_cache()
    assert tuner.load_configs(str(path))
    assert _choose(tuner) == 0
    return path


@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("mode", ["tune", "replay"])
def test_identified_scope_excludes_legacy_without_mutation(
    scoped, tmp_path, persistent, mode
):
    tuner, calls = scoped
    path = _load_legacy(tuner, tmp_path)
    before = path.read_bytes()
    with autotune_v2(
        mode=mode,
        persistent_cache=persistent,
        cache_scope="context",
        measurement_policy=_GRAPH,
    ):
        assert _choose(tuner) == (2 if mode == "tune" else -1)
        assert tuner._managed_cache is None
    assert bool(calls) == (mode == "tune")
    calls.clear()
    with autotune_v2(
        mode="replay",
        persistent_cache=persistent,
        cache_scope="context",
        measurement_policy=_GRAPH,
    ):
        assert _choose(tuner) == (2 if mode == "tune" else -1)
    assert not calls
    assert _choose(tuner) == 0
    tuner.save_configs(str(path))
    assert path.read_bytes() == before


def test_scoped_store_does_not_replace_ambient_on_error(scoped):
    tuner, calls = scoped
    with autotune_v2(measurement_policy=_EAGER):
        assert _choose(tuner) == 1
    ambient = tuner._managed_cache
    with (
        pytest.raises(RuntimeError, match="user failure"),
        autotune_v2(cache_scope="context", measurement_policy=_GRAPH),
    ):
        assert _choose(tuner) == 2
        assert tuner._managed_cache is ambient
        raise RuntimeError("user failure")
    calls.clear()
    assert _choose(tuner) == 1
    with autotune_v2(mode="replay", cache_scope="context", measurement_policy=_GRAPH):
        assert _choose(tuner) == 2
    assert tuner._managed_cache is ambient
    assert not calls
    assert not tuner._in_v2_context


@pytest.mark.parametrize("cache_scope", ["process", "context"])
def test_scope_controls_bundled_precedence(scoped, monkeypatch, cache_scope):
    tuner, _ = scoped
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "1")
    monkeypatch.setattr(
        "flashinfer.autotuner.autotuner.load_from_file",
        lambda key: (True, "DummyRunner", 0, None),
    )
    with autotune_v2(mode="replay", cache_scope=cache_scope, measurement_policy=_GRAPH):
        assert _choose(tuner) == (0 if cache_scope == "process" else -1)


@pytest.mark.parametrize("policy", [None, MeasurementPolicy()])
def test_unidentified_memory_scope_preserves_legacy(scoped, tmp_path, policy):
    tuner, calls = scoped
    _load_legacy(tuner, tmp_path)
    with autotune_v2(
        mode="replay",
        persistent_cache=False,
        cache_scope="context",
        measurement_policy=policy,
    ):
        assert _choose(tuner) == 0
    assert not calls


def test_default_policy_persistent_scope_is_isolated(scoped, tmp_path):
    tuner, _ = scoped
    _load_legacy(tuner, tmp_path)
    with autotune_v2(mode="replay", cache_scope="context"):
        assert _choose(tuner) == -1
    assert _choose(tuner) == 0


def test_scoped_shortlist_uses_common_ownership(scoped, tmp_path):
    tuner, calls = scoped
    _load_legacy(tuner, tmp_path)
    runner = DummyRunner()
    inputs = [torch.ones(2)]
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        ranked = tuner.rank_tactics(_OP, [runner], _CONFIG, inputs, k=2)
        assert ranked[0] == 2
        calls.clear()
        assert tuner.rank_tactics(_OP, [runner], _CONFIG, inputs, k=2) == ranked
        assert _choose(tuner) == 2
        assert not calls
    assert _choose(tuner) == 0
    autotune_v2_reload()
    with autotune_v2(mode="replay", cache_scope="context", measurement_policy=_GRAPH):
        assert tuner.rank_tactics(_OP, [runner], _CONFIG, inputs, k=2) == [2]
    assert not calls
    assert _choose(tuner) == 0


def test_protocol_records_follow_scoped_store(scoped):
    tuner, _ = scoped
    with autotune_v2(measurement_policy=_EAGER):
        context = tuner.protocol_context("scoped.protocol")
        context.publish("shape", {"winner": 1})
    ambient = tuner._managed_cache
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        context = tuner.protocol_context("scoped.protocol")
        assert context.lookup("shape") is None
        context.publish("shape", {"winner": 2})
    autotune_v2_reload()
    with autotune_v2(mode="replay", cache_scope="context", measurement_policy=_GRAPH):
        context = tuner.protocol_context("scoped.protocol")
        assert context.measurement_policy == _GRAPH
        assert context.lookup("shape") == {"winner": 2}
    assert tuner._managed_cache is ambient
    assert tuner.protocol_context("scoped.protocol").lookup("shape") == {"winner": 1}


def test_overlapping_replay_does_not_restore_stale_ambient(scoped, tmp_path):
    tuner, _ = scoped
    with autotune_v2(measurement_policy=_EAGER):
        assert _choose(tuner) == 1
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        assert _choose(tuner) == 2
    entered = threading.Barrier(3, timeout=15)
    attached = threading.Barrier(3, timeout=15)

    def replay(policy, expected):
        with autotune_v2(
            mode="replay", cache_scope="context", measurement_policy=policy
        ):
            entered.wait()
            attached.wait()
            assert _choose(tuner) == expected

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(replay, _EAGER, 1)
        b = pool.submit(replay, _GRAPH, 2)
        entered.wait()
        with autotune_v2(mode="replay", cache_root=tmp_path / "new-ambient"):
            new_ambient = tuner._managed_cache
        attached.wait()
        a.result(timeout=15)
        b.result(timeout=15)
    assert tuner._managed_cache is new_ambient
    assert _choose(tuner) == -1


def test_scope_validation_and_reload_do_not_mutate_state(scoped):
    tuner, _ = scoped
    with (
        pytest.raises(ValueError, match="cache_scope"),
        autotune_v2(cache_scope="typo"),
    ):
        pytest.fail("invalid scope accepted")
    assert tuner._managed_cache is None
    assert not tuner._managed_stores
    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        assert _choose(tuner) == 2
        with pytest.raises(RuntimeError, match="reload inside"):
            autotune_v2_reload()
        with (
            pytest.raises(RuntimeError, match="nested"),
            autotune_v2(cache_scope="context"),
        ):
            pytest.fail("nested scope accepted")
        assert _choose(tuner) == 2
    assert not tuner.is_tuning_mode


def _restart_worker(root, mode):
    """Actual fresh processes, with controlled CPU timing and metadata only."""
    import flashinfer.autotuner as package

    package._collect_metadata = lambda: {"test": "scope-restart"}
    torch.cuda.is_current_stream_capturing = lambda: False
    tuner = AutoTuner.get()
    calls = []

    def profile(runner, inputs, tactic, config, **kwargs):
        calls.append(tactic)
        assert mode == "cold", "warm restart attempted profiling"
        winner = 2 if config.use_cuda_graph else 1
        return 1.0 if tactic == winner else 3.0

    tuner._profile_single_kernel = profile
    for policy, winner in [(_EAGER, 1), (_GRAPH, 2)]:
        with autotune_v2(
            cache_root=root, cache_scope="context", measurement_policy=policy
        ):
            assert _choose(tuner) == winner
        assert tuner._managed_cache is None
        assert _choose(tuner) == -1
    print(json.dumps({"profile_calls": len(calls)}))


def test_scoped_cache_reused_in_fresh_process(tmp_path):
    script = (
        "from tests.autotuner.test_autotune_cache_scope import _restart_worker; "
        "import sys; _restart_worker(sys.argv[1], sys.argv[2])"
    )
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "FLASHINFER_AUTOTUNER_LOAD_FROM_FILE": "0",
    }
    counts = []
    for mode in ("cold", "warm"):
        result = subprocess.run(
            [sys.executable, "-c", script, str(tmp_path), mode],
            cwd=Path(__file__).resolve().parents[2],
            env=env,
            text=True,
            capture_output=True,
            timeout=90,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        counts.append(json.loads(result.stdout.splitlines()[-1])["profile_calls"])
    assert counts[0] > 0
    assert counts[1] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("strict_replay", [False, True])
def test_scoped_gpu_tuning_restart_and_capture(tmp_path, monkeypatch, strict_replay):
    class AddRunner(DummyRunner):
        def forward(self, inputs, tactic=-1, **kwargs):
            x = inputs[0].clone() if tactic == 1 else inputs[0]
            return x + 1

    monkeypatch.setenv("FLASHINFER_AUTOTUNE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("FLASHINFER_AUTOTUNER_LOAD_FROM_FILE", "0")
    tuner = AutoTuner(warmup=2, repeat=5)
    monkeypatch.setattr(AutoTuner, "_instance", tuner)
    inputs = [torch.randn(64, device="cuda")]
    runner = AddRunner((0, 1))

    def forward():
        selected, tactic = AutoTuner.get().choose_one(_OP, [runner], _CONFIG, inputs)
        return selected(inputs, tactic=tactic)

    with autotune_v2(cache_scope="context", measurement_policy=_GRAPH):
        torch.testing.assert_close(forward(), inputs[0] + 1)
    # New tuner discards all memory state, so the next scope must hydrate disk.
    tuner = AutoTuner()
    monkeypatch.setattr(AutoTuner, "_instance", tuner)

    def no_profile(*args, **kwargs):
        pytest.fail("warm graph scope attempted profiling")

    monkeypatch.setattr(tuner, "_profile_single_kernel", no_profile)
    with autotune_v2(
        mode="replay",
        cache_scope="context",
        measurement_policy=_GRAPH,
        strict_replay=strict_replay,
    ):
        assert tuner.search_cache(
            _OP, [runner], ((64,),), replace(_CONFIG, use_cuda_graph=True), inputs
        )[0]
        forward()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = forward()
        graph.replay()
        torch.testing.assert_close(out, inputs[0] + 1)
    assert tuner._managed_cache is None
    assert not tuner.search_cache(_OP, [runner], ((64,),), _CONFIG, inputs)[0]
