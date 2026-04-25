"""Tests for the TacticsWhitelist class.

No GPU is required — all tests use mock data and temporary files.
"""

import json
import os
import tempfile

import pytest

from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    _METADATA_KEY,
    _tactic_to_json_hashable,
    autotune,
)
from flashinfer.tactics_whitelist import TacticsWhitelist


# ---------------------------------------------------------------------------
# Minimal TunableRunner stubs
# ---------------------------------------------------------------------------


class StubRunnerA(TunableRunner):
    def __init__(self, valid_tactics=(0, 1, 2, 3)):
        self._valid = tuple(valid_tactics)

    def get_valid_tactics(self, inputs, profile):
        return list(self._valid)

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


class StubRunnerB(TunableRunner):
    def get_valid_tactics(self, inputs, profile):
        return [[32, 0], [32, 1], [64, 0], [64, 1], [128, 0], [128, 1]]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return inputs[0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_whitelist_json(
    invalid_tactics: dict,
    gpu: str = "NVIDIA B200",
    **extra_meta,
) -> dict:
    """Build a whitelist JSON structure for testing."""
    meta = {"gpu": gpu, "generator_version": "1.0"}
    meta.update(extra_meta)
    return {
        _METADATA_KEY: meta,
        "invalid_tactics": invalid_tactics,
    }


def _write_json(data: dict) -> str:
    """Write *data* to a temp file and return its path."""
    fd, path = tempfile.mkstemp(suffix=".json")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    return path


# ---------------------------------------------------------------------------
# Tests: _tactic_to_json_hashable
# ---------------------------------------------------------------------------


class TestTacticToJsonHashable:
    def test_int(self):
        assert _tactic_to_json_hashable(5) == 5

    def test_list_to_tuple(self):
        assert _tactic_to_json_hashable([32, 0]) == (32, 0)

    def test_nested_list(self):
        assert _tactic_to_json_hashable([128, [64, 64], True]) == (128, (64, 64), True)

    def test_tuple_passthrough(self):
        assert _tactic_to_json_hashable((32, 0)) == (32, 0)

    def test_bool_preserved(self):
        assert _tactic_to_json_hashable(True) is True

    def test_empty_list(self):
        assert _tactic_to_json_hashable([]) == ()

    def test_string_passthrough(self):
        assert _tactic_to_json_hashable("relu") == "relu"

    def test_float_passthrough(self):
        assert _tactic_to_json_hashable(3.14) == 3.14

    def test_none_passthrough(self):
        assert _tactic_to_json_hashable(None) is None

    def test_list_and_tuple_normalize_equal(self):
        """Lists and tuples with the same values should normalize identically."""
        assert _tactic_to_json_hashable([128, 0]) == _tactic_to_json_hashable((128, 0))


# ---------------------------------------------------------------------------
# Tests: TacticsWhitelist.load
# ---------------------------------------------------------------------------


class TestWhitelistLoad:
    def test_load_basic(self, monkeypatch):
        """Load a whitelist and verify internal state."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_whitelist_json(
            {"op::RunnerA": [[128, 0], [128, 1]]},
            gpu="NVIDIA B200",
        )
        path = _write_json(data)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path) is True
            assert wl.is_loaded
            assert wl.summary() == {"op::RunnerA": 2}
        finally:
            os.unlink(path)

    def test_load_gpu_mismatch_skips(self, monkeypatch):
        """Whitelist generated for a different GPU should be skipped."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA H100"},
        )
        data = _make_whitelist_json(
            {"op::RunnerA": [1, 2]},
            gpu="NVIDIA B200",
        )
        path = _write_json(data)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path) is False
            assert not wl.is_loaded
        finally:
            os.unlink(path)

    def test_load_wildcard_gpu_matches_any(self, monkeypatch):
        """gpu='*' should match any current GPU."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA H100"},
        )
        data = _make_whitelist_json({"op::R": [1]}, gpu="*")
        path = _write_json(data)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path) is True
        finally:
            os.unlink(path)

    def test_load_no_metadata_succeeds(self):
        """A whitelist without _metadata should load without GPU checks."""
        data = {"invalid_tactics": {"op::R": [42]}}
        path = _write_json(data)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path) is True
            assert wl.summary() == {"op::R": 1}
        finally:
            os.unlink(path)

    def test_load_empty_invalid_tactics(self, monkeypatch):
        """An empty invalid_tactics dict should load but not mark anything."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_whitelist_json({}, gpu="NVIDIA B200")
        path = _write_json(data)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path) is True
            assert wl.summary() == {}
            assert wl.filter("op", StubRunnerA(), [0, 1, 2]) == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_load_file_not_found(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        wl = TacticsWhitelist()
        with pytest.raises(FileNotFoundError):
            wl.load("/tmp/nonexistent_whitelist_file_12345.json")

    def test_load_corrupt_json(self):
        """Loading a corrupt JSON file should raise an error."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("NOT VALID JSON {{{")
        try:
            wl = TacticsWhitelist()
            with pytest.raises(json.JSONDecodeError):
                wl.load(path)
        finally:
            os.unlink(path)

    def test_load_multiple_files_accumulates(self, monkeypatch):
        """Loading two whitelist files should merge their entries."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data1 = _make_whitelist_json({"opA::RunnerA": [1, 2]}, gpu="NVIDIA B200")
        data2 = _make_whitelist_json({"opB::RunnerB": [3, 4]}, gpu="NVIDIA B200")
        path1 = _write_json(data1)
        path2 = _write_json(data2)
        try:
            wl = TacticsWhitelist()
            assert wl.load(path1) is True
            assert wl.load(path2) is True
            assert wl.summary() == {"opA::RunnerA": 2, "opB::RunnerB": 2}
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ---------------------------------------------------------------------------
# Tests: TacticsWhitelist.filter
# ---------------------------------------------------------------------------


class TestWhitelistFilter:
    def _loaded_whitelist(self, invalid_tactics, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_whitelist_json(invalid_tactics, gpu="NVIDIA B200")
        path = _write_json(data)
        wl = TacticsWhitelist()
        wl.load(path)
        os.unlink(path)
        return wl

    def test_filter_removes_invalid_int_tactics(self, monkeypatch):
        wl = self._loaded_whitelist({"my_op::StubRunnerA": [1, 3]}, monkeypatch)
        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        result = wl.filter("my_op", runner, [0, 1, 2, 3])
        assert result == [0, 2]

    def test_filter_removes_invalid_list_tactics(self, monkeypatch):
        wl = self._loaded_whitelist(
            {"my_op::StubRunnerB": [[128, 0], [128, 1]]}, monkeypatch
        )
        runner = StubRunnerB()
        tactics = [[32, 0], [32, 1], [64, 0], [64, 1], [128, 0], [128, 1]]
        result = wl.filter("my_op", runner, tactics)
        assert result == [[32, 0], [32, 1], [64, 0], [64, 1]]

    def test_filter_no_whitelist_passthrough(self):
        """When no whitelist is loaded, all tactics pass through."""
        wl = TacticsWhitelist()
        runner = StubRunnerA()
        tactics = [0, 1, 2, 3]
        assert wl.filter("op", runner, tactics) == tactics

    def test_filter_no_matching_key_passthrough(self, monkeypatch):
        """When the op::runner key doesn't exist, all tactics pass through."""
        wl = self._loaded_whitelist({"other_op::OtherRunner": [1]}, monkeypatch)
        runner = StubRunnerA()
        assert wl.filter("my_op", runner, [0, 1, 2]) == [0, 1, 2]

    def test_filter_all_invalid_returns_empty(self, monkeypatch):
        wl = self._loaded_whitelist({"my_op::StubRunnerA": [0, 1, 2]}, monkeypatch)
        runner = StubRunnerA(valid_tactics=[0, 1, 2])
        result = wl.filter("my_op", runner, [0, 1, 2])
        assert result == []

    def test_filter_scope_isolation(self, monkeypatch):
        """Same runner class under different custom_ops should filter independently."""
        wl = self._loaded_whitelist(
            {
                "opA::StubRunnerA": [1],
                "opB::StubRunnerA": [2],
            },
            monkeypatch,
        )
        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        assert wl.filter("opA", runner, [0, 1, 2, 3]) == [0, 2, 3]
        assert wl.filter("opB", runner, [0, 1, 2, 3]) == [0, 1, 3]

    def test_filter_json_list_matches_runtime_tuple(self, monkeypatch):
        """Whitelist stores lists (from JSON), runtime may provide tuples — should match."""
        wl = self._loaded_whitelist({"my_op::StubRunnerB": [[128, 0]]}, monkeypatch)
        runner = StubRunnerB()
        result = wl.filter("my_op", runner, [(128, 0), (64, 0)])
        assert result == [(64, 0)]

    def test_filter_empty_tactics_list(self, monkeypatch):
        """Filtering an empty tactics list should return empty."""
        wl = self._loaded_whitelist({"my_op::StubRunnerA": [1]}, monkeypatch)
        runner = StubRunnerA()
        assert wl.filter("my_op", runner, []) == []


# ---------------------------------------------------------------------------
# Tests: TacticsWhitelist.save
# ---------------------------------------------------------------------------


class TestWhitelistSave:
    def test_save_roundtrip(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsWhitelist.save(
                path,
                {"op::RunnerA": [[128, 0], (64, 1)]},
                metadata={"gpu": "NVIDIA B200"},
            )
            wl = TacticsWhitelist()
            assert wl.load(path) is True
            assert wl.summary() == {"op::RunnerA": 2}

            runner = StubRunnerA()
            # (128, 0) normalized from [128, 0] should match
            result = wl.filter("op", runner, [[128, 0], [32, 0]])

            class _FakeRunner:
                __class__ = type("RunnerA", (), {})

            # Use a runner whose __class__.__name__ is "RunnerA"
            fake = _FakeRunner()
            fake.__class__ = type("RunnerA", (), {})
            result = wl.filter("op", fake, [[128, 0], [32, 0]])
            assert result == [[32, 0]]
        finally:
            os.unlink(path)

    def test_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sub", "dir", "whitelist.json")
            TacticsWhitelist.save(path, {"op::R": [1]}, metadata={"gpu": "*"})
            assert os.path.isfile(path)

    def test_save_includes_metadata(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsWhitelist.save(path, {"op::R": [1]}, metadata={"gpu": "TestGPU"})
            with open(path) as f:
                data = json.load(f)
            meta = data[_METADATA_KEY]
            assert meta["gpu"] == "TestGPU"
            assert "generated_at" in meta
            assert meta["generator_version"] == "1.0"
        finally:
            os.unlink(path)

    def test_save_auto_collects_metadata(self, monkeypatch):
        """When metadata=None, save() should auto-collect via _collect_metadata."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "AutoGPU", "cuda_version": "13.0"},
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsWhitelist.save(path, {"op::R": [1]})
            with open(path) as f:
                data = json.load(f)
            meta = data[_METADATA_KEY]
            assert meta["gpu"] == "AutoGPU"
            assert meta["cuda_version"] == "13.0"
            assert "generated_at" in meta
        finally:
            os.unlink(path)

    def test_save_serializes_tuples_as_lists(self):
        """Tuples should be serialized as lists in JSON output."""
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsWhitelist.save(
                path,
                {"op::R": [(128, 0), (64, 1)]},
                metadata={"gpu": "*"},
            )
            with open(path) as f:
                data = json.load(f)
            tactics = data["invalid_tactics"]["op::R"]
            assert tactics == [[128, 0], [64, 1]]
            assert all(isinstance(t, list) for t in tactics)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Tests: integration with AutoTuner.choose_one
# ---------------------------------------------------------------------------


class TestAutotunerIntegration:
    """Verify the whitelist filters tactics inside the autotuner loop."""

    def setup_method(self):
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_choose_one_skips_whitelisted_tactics(self, monkeypatch, tmp_path):
        """Tactics in the whitelist should never be profiled."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )
        monkeypatch.setattr(
            "flashinfer.autotuner._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )

        profiled_tactics = []

        def fake_profile(self_at, runner, inputs, tactic, tuning_config=None, **kw):
            profiled_tactics.append(tactic)
            return {0: 5.0, 2: 1.0}.get(tactic, 10.0)

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

        # Create whitelist that blocks tactics 1 and 3
        wl_path = str(tmp_path / "whitelist.json")
        TacticsWhitelist.save(
            wl_path,
            {"test_op::StubRunnerA": [1, 3]},
            metadata={"gpu": "TestGPU"},
        )

        # Load into the autotuner's whitelist
        self.tuner._whitelist.load(wl_path)

        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        import torch

        inputs = [torch.empty((4, 8), dtype=torch.float32)]
        config = TuningConfig()

        with autotune(tune_mode=True):
            chosen_runner, tactic = self.tuner.choose_one(
                "test_op", [runner], config, inputs
            )

        # Tactics 1 and 3 should NOT have been profiled
        assert 1 not in profiled_tactics
        assert 3 not in profiled_tactics
        # Tactics 0 and 2 should have been profiled
        assert 0 in profiled_tactics
        assert 2 in profiled_tactics
        # Best tactic should be 2 (lowest time=1.0)
        assert tactic == 2

    def test_no_whitelist_profiles_all(self, monkeypatch):
        """Without a whitelist, all tactics should be profiled as usual."""
        profiled_tactics = []

        def fake_profile(self_at, runner, inputs, tactic, tuning_config=None, **kw):
            profiled_tactics.append(tactic)
            return {0: 5.0, 1: 1.0, 2: 3.0}.get(tactic, 10.0)

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

        runner = StubRunnerA(valid_tactics=[0, 1, 2])
        import torch

        inputs = [torch.empty((4, 8), dtype=torch.float32)]
        config = TuningConfig()

        with autotune(tune_mode=True):
            chosen_runner, tactic = self.tuner.choose_one(
                "all_op", [runner], config, inputs
            )

        assert set(profiled_tactics) == {0, 1, 2}
        assert tactic == 1

    def test_failed_tactics_recorded_in_stats(self, monkeypatch):
        """When profiling raises, the tactic should appear in stats.failed_tactics."""

        def fake_profile(self_at, runner, inputs, tactic, tuning_config=None, **kw):
            if tactic in (1, 3):
                raise RuntimeError(f"tactic {tactic} fails")
            return {0: 5.0, 2: 1.0}.get(tactic, 10.0)

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        import torch

        inputs = [torch.empty((4, 8), dtype=torch.float32)]
        config = TuningConfig()

        with autotune(tune_mode=True):
            self.tuner.choose_one("fail_op", [runner], config, inputs)

        key = "fail_op::StubRunnerA"
        assert key in self.tuner.stats.failed_tactics
        assert self.tuner.stats.failed_tactics[key] == {1, 3}

    def test_env_var_auto_loads_whitelist(self, monkeypatch, tmp_path):
        """Setting FLASHINFER_TACTICS_WHITELIST should auto-load on init."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )

        wl_path = str(tmp_path / "env_wl.json")
        TacticsWhitelist.save(
            wl_path,
            {"env_op::StubRunnerA": [99]},
            metadata={"gpu": "TestGPU"},
        )

        monkeypatch.setenv("FLASHINFER_TACTICS_WHITELIST", wl_path)

        AutoTuner._instance = None
        tuner = AutoTuner.get()
        try:
            assert tuner._whitelist.is_loaded
            assert tuner._whitelist.summary() == {"env_op::StubRunnerA": 1}
        finally:
            AutoTuner._instance = None

    def test_choose_one_with_all_tactics_filtered_still_succeeds(
        self, monkeypatch, tmp_path
    ):
        """If whitelist filters ALL tactics for a runner, choose_one should
        gracefully handle it (no crash, returns None or next best)."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )
        monkeypatch.setattr(
            "flashinfer.autotuner._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )

        profiled_tactics = []

        def fake_profile(self_at, runner, inputs, tactic, tuning_config=None, **kw):
            profiled_tactics.append(tactic)
            return 1.0

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

        wl_path = str(tmp_path / "all_blocked.json")
        TacticsWhitelist.save(
            wl_path,
            {"block_op::StubRunnerA": [0, 1, 2]},
            metadata={"gpu": "TestGPU"},
        )
        self.tuner._whitelist.load(wl_path)

        runner = StubRunnerA(valid_tactics=[0, 1, 2])
        import torch

        inputs = [torch.empty((4, 8), dtype=torch.float32)]
        config = TuningConfig()

        with autotune(tune_mode=True):
            chosen_runner, tactic = self.tuner.choose_one(
                "block_op", [runner], config, inputs
            )

        assert profiled_tactics == []
        assert tactic == -1

    def test_whitelist_does_not_affect_cache_hits(self, monkeypatch, tmp_path):
        """Cached results should be returned directly, bypassing whitelist filtering."""
        monkeypatch.setattr(
            "flashinfer.tactics_whitelist._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )
        monkeypatch.setattr(
            "flashinfer.autotuner._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )

        call_count = [0]

        def fake_profile(self_at, runner, inputs, tactic, tuning_config=None, **kw):
            call_count[0] += 1
            return {0: 5.0, 2: 1.0}.get(tactic, 10.0)

        monkeypatch.setattr(AutoTuner, "_profile_single_kernel", fake_profile)

        wl_path = str(tmp_path / "cache_test.json")
        TacticsWhitelist.save(
            wl_path,
            {"cache_op::StubRunnerA": [1, 3]},
            metadata={"gpu": "TestGPU"},
        )
        self.tuner._whitelist.load(wl_path)

        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        import torch

        inputs = [torch.empty((4, 8), dtype=torch.float32)]
        config = TuningConfig()

        with autotune(tune_mode=True):
            _, tactic1 = self.tuner.choose_one("cache_op", [runner], config, inputs)
        first_call_count = call_count[0]
        assert tactic1 == 2

        with autotune(tune_mode=True):
            _, tactic2 = self.tuner.choose_one("cache_op", [runner], config, inputs)

        assert tactic2 == 2
        assert call_count[0] == first_call_count
