"""
Tests for ``autotune(cache=...)`` with several cache files in one process.

A process that keeps one cache file per model (for example a serving engine
tuning a target model and its speculative draft model, whose config hashes
differ) enters ``autotune(cache=...)`` once per file.  Entering the second
context must add that file's configs to the ones already loaded instead of
replacing them.

No GPU is required -- profiling is monkeypatched and inputs are CPU tensors.
"""

import json

import torch

import flashinfer.autotuner.autotuner as autotuner_module
from flashinfer.autotuner import AutoTuner, TuningConfig, autotune

from .utils import DummyRunner, reset_autotuner

_CONFIG = TuningConfig()  # no dynamic specs -> a single static profile
_METADATA = {"gpu": "test-gpu"}


def _file_key(custom_op, runner, shapes):
    return AutoTuner._get_cache_key(custom_op, runner, shapes, _CONFIG).file_key


def _write_cache(path, entries):
    payload = {autotuner_module._METADATA_KEY: _METADATA}
    payload.update({key: ["DummyRunner", tactic] for key, tactic in entries.items()})
    path.write_text(json.dumps(payload))


def _entries(path):
    return {
        key: value[1]
        for key, value in json.loads(path.read_text()).items()
        if not key.startswith("_")
    }


def test_entering_second_cache_file_keeps_first_files_configs(monkeypatch, tmp_path):
    monkeypatch.setattr(autotuner_module, "_collect_metadata", lambda: dict(_METADATA))
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        AutoTuner,
        "_profile_single_kernel",
        lambda self, runner, inputs, tactic, tuning_config, **kw: (
            1.0 if tactic == 2 else 2.0
        ),
    )
    tuner = reset_autotuner()
    runner = DummyRunner(valid_tactics=(0, 1, 2))
    shapes_a = (torch.Size([8, 16]),)
    shapes_b = (torch.Size([64, 16]),)
    shapes_shared = (torch.Size([16, 16]),)
    shapes_new = (torch.Size([32, 16]),)

    key_a = _file_key("merge_target", runner, shapes_a)
    key_b = _file_key("merge_draft", runner, shapes_b)
    key_shared = _file_key("merge_shared", runner, shapes_shared)
    file_a = tmp_path / "target.json"
    file_b = tmp_path / "draft.json"
    _write_cache(file_a, {key_a: 0, key_shared: 0})
    _write_cache(file_b, {key_b: 1, key_shared: 1})

    def served(custom_op, shapes):
        hit, _, tactic, _ = tuner.search_cache(custom_op, [runner], shapes, _CONFIG)
        return tactic if hit else None

    with autotune(True, cache=str(file_a)):
        assert served("merge_target", shapes_a) == 0
        assert served("merge_shared", shapes_shared) == 0
        assert served("merge_draft", shapes_b) is None

    with autotune(True, cache=str(file_b)):
        # The first file's configs are still served after the second file
        # is loaded, and the file loaded last wins for a key in both.
        assert served("merge_target", shapes_a) == 0
        assert served("merge_draft", shapes_b) == 1
        assert served("merge_shared", shapes_shared) == 1
        # Tune something new so the context saves the file on exit.
        inputs = [torch.zeros(*shapes_new[0])]
        assert tuner.choose_one("merge_new", [runner], _CONFIG, inputs)[1] == 2

    # The first file is untouched; the second file keeps its own entries
    # and gains the new result.  save_configs snapshots every loaded config
    # as well, so the first file's entries are also written to the second.
    assert _entries(file_a) == {key_a: 0, key_shared: 0}
    saved_b = _entries(file_b)
    assert saved_b[key_b] == 1
    assert saved_b[key_shared] == 1
    assert saved_b[key_a] == 0
    assert saved_b[_file_key("merge_new", runner, shapes_new)] == 2

    # Re-entering the first file makes it the last loaded one again.
    with autotune(False, cache=str(file_a)):
        assert served("merge_shared", shapes_shared) == 0
        assert served("merge_draft", shapes_b) == 1


def test_entering_missing_cache_file_keeps_loaded_configs(monkeypatch, tmp_path):
    monkeypatch.setattr(autotuner_module, "_collect_metadata", lambda: dict(_METADATA))
    tuner = reset_autotuner()
    runner = DummyRunner()
    shapes = (torch.Size([8, 16]),)
    key = _file_key("merge_missing", runner, shapes)
    file_a = tmp_path / "target.json"
    _write_cache(file_a, {key: 1})

    with autotune(False, cache=str(file_a)):
        pass
    with autotune(False, cache=str(tmp_path / "does-not-exist.json")):
        hit, _, tactic, _ = tuner.search_cache(
            "merge_missing", [runner], shapes, _CONFIG
        )
        assert (hit, tactic) == (True, 1)
