"""Tests for the TacticsBlocklist class.

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
from flashinfer.tactics_blocklist import TacticsBlocklist


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


def _make_blocklist_json(
    invalid_tactics: dict,
    gpu: str = "NVIDIA B200",
    **extra_meta,
) -> dict:
    """Build a blocklist JSON structure for testing."""
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
# Tests: TacticsBlocklist.load
# ---------------------------------------------------------------------------


class TestBlocklistLoad:
    def test_load_basic(self, monkeypatch):
        """Load a blocklist and verify internal state."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_blocklist_json(
            {"op::RunnerA": [[128, 0], [128, 1]]},
            gpu="NVIDIA B200",
        )
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is True
            assert bl.is_loaded
            assert bl.summary() == {"op::RunnerA": 2}
        finally:
            os.unlink(path)

    def test_load_gpu_mismatch_skips(self, monkeypatch):
        """Blocklist generated for a different GPU should be skipped."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA H100"},
        )
        data = _make_blocklist_json(
            {"op::RunnerA": [1, 2]},
            gpu="NVIDIA B200",
        )
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is False
            assert not bl.is_loaded
        finally:
            os.unlink(path)

    def test_load_version_mismatch_skips(self, monkeypatch):
        """Blocklist generated with different software version should be skipped."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200", "flashinfer_version": "0.7.0"},
        )
        data = _make_blocklist_json(
            {"op::RunnerA": [1, 2]},
            gpu="NVIDIA B200",
            flashinfer_version="0.6.9",
        )
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is False
            assert not bl.is_loaded
        finally:
            os.unlink(path)

    def test_load_wildcard_gpu_matches_any(self, monkeypatch):
        """gpu='*' should match any current GPU."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA H100"},
        )
        data = _make_blocklist_json({"op::R": [1]}, gpu="*")
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is True
        finally:
            os.unlink(path)

    def test_load_no_metadata_succeeds(self):
        """A blocklist without _metadata should load without GPU checks."""
        data = {"invalid_tactics": {"op::R": [42]}}
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is True
            assert bl.summary() == {"op::R": 1}
        finally:
            os.unlink(path)

    def test_load_empty_invalid_tactics(self, monkeypatch):
        """An empty invalid_tactics dict should load but not mark anything."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_blocklist_json({}, gpu="NVIDIA B200")
        path = _write_json(data)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path) is True
            assert bl.summary() == {}
            assert bl.filter("op", StubRunnerA(), [0, 1, 2]) == [0, 1, 2]
        finally:
            os.unlink(path)

    def test_load_file_not_found(self):
        """Loading a non-existent file should raise FileNotFoundError."""
        bl = TacticsBlocklist()
        with pytest.raises(FileNotFoundError):
            bl.load("/tmp/nonexistent_blocklist_file_12345.json")

    def test_load_corrupt_json(self):
        """Loading a corrupt JSON file should raise an error."""
        fd, path = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w") as f:
            f.write("NOT VALID JSON {{{")
        try:
            bl = TacticsBlocklist()
            with pytest.raises(json.JSONDecodeError):
                bl.load(path)
        finally:
            os.unlink(path)

    def test_load_multiple_files_accumulates(self, monkeypatch):
        """Loading two blocklist files should merge their entries."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data1 = _make_blocklist_json({"opA::RunnerA": [1, 2]}, gpu="NVIDIA B200")
        data2 = _make_blocklist_json({"opB::RunnerB": [3, 4]}, gpu="NVIDIA B200")
        path1 = _write_json(data1)
        path2 = _write_json(data2)
        try:
            bl = TacticsBlocklist()
            assert bl.load(path1) is True
            assert bl.load(path2) is True
            assert bl.summary() == {"opA::RunnerA": 2, "opB::RunnerB": 2}
        finally:
            os.unlink(path1)
            os.unlink(path2)


# ---------------------------------------------------------------------------
# Tests: TacticsBlocklist.filter
# ---------------------------------------------------------------------------


class TestBlocklistFilter:
    def _loaded_blocklist(self, invalid_tactics, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        data = _make_blocklist_json(invalid_tactics, gpu="NVIDIA B200")
        path = _write_json(data)
        bl = TacticsBlocklist()
        bl.load(path)
        os.unlink(path)
        return bl

    def test_filter_removes_invalid_int_tactics(self, monkeypatch):
        bl = self._loaded_blocklist({"my_op::StubRunnerA": [1, 3]}, monkeypatch)
        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        result = bl.filter("my_op", runner, [0, 1, 2, 3])
        assert result == [0, 2]

    def test_filter_removes_invalid_list_tactics(self, monkeypatch):
        bl = self._loaded_blocklist(
            {"my_op::StubRunnerB": [[128, 0], [128, 1]]}, monkeypatch
        )
        runner = StubRunnerB()
        tactics = [[32, 0], [32, 1], [64, 0], [64, 1], [128, 0], [128, 1]]
        result = bl.filter("my_op", runner, tactics)
        assert result == [[32, 0], [32, 1], [64, 0], [64, 1]]

    def test_filter_no_blocklist_passthrough(self):
        """When no blocklist is loaded, all tactics pass through."""
        bl = TacticsBlocklist()
        runner = StubRunnerA()
        tactics = [0, 1, 2, 3]
        assert bl.filter("op", runner, tactics) == tactics

    def test_filter_no_matching_key_passthrough(self, monkeypatch):
        """When the op::runner key doesn't exist, all tactics pass through."""
        bl = self._loaded_blocklist({"other_op::OtherRunner": [1]}, monkeypatch)
        runner = StubRunnerA()
        assert bl.filter("my_op", runner, [0, 1, 2]) == [0, 1, 2]

    def test_filter_all_invalid_returns_empty(self, monkeypatch):
        bl = self._loaded_blocklist({"my_op::StubRunnerA": [0, 1, 2]}, monkeypatch)
        runner = StubRunnerA(valid_tactics=[0, 1, 2])
        result = bl.filter("my_op", runner, [0, 1, 2])
        assert result == []

    def test_filter_scope_isolation(self, monkeypatch):
        """Same runner class under different custom_ops should filter independently."""
        bl = self._loaded_blocklist(
            {
                "opA::StubRunnerA": [1],
                "opB::StubRunnerA": [2],
            },
            monkeypatch,
        )
        runner = StubRunnerA(valid_tactics=[0, 1, 2, 3])
        assert bl.filter("opA", runner, [0, 1, 2, 3]) == [0, 2, 3]
        assert bl.filter("opB", runner, [0, 1, 2, 3]) == [0, 1, 3]

    def test_filter_json_list_matches_runtime_tuple(self, monkeypatch):
        """Blocklist stores lists (from JSON), runtime may provide tuples — should match."""
        bl = self._loaded_blocklist({"my_op::StubRunnerB": [[128, 0]]}, monkeypatch)
        runner = StubRunnerB()
        result = bl.filter("my_op", runner, [(128, 0), (64, 0)])
        assert result == [(64, 0)]

    def test_filter_empty_tactics_list(self, monkeypatch):
        """Filtering an empty tactics list should return empty."""
        bl = self._loaded_blocklist({"my_op::StubRunnerA": [1]}, monkeypatch)
        runner = StubRunnerA()
        assert bl.filter("my_op", runner, []) == []


# ---------------------------------------------------------------------------
# Tests: TacticsBlocklist.save
# ---------------------------------------------------------------------------


class TestBlocklistSave:
    def test_save_roundtrip(self, monkeypatch):
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "NVIDIA B200"},
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsBlocklist.save(
                path,
                {"op::RunnerA": [[128, 0], (64, 1)]},
                metadata={"gpu": "NVIDIA B200"},
            )
            bl = TacticsBlocklist()
            assert bl.load(path) is True
            assert bl.summary() == {"op::RunnerA": 2}

            runner = StubRunnerA()
            # (128, 0) normalized from [128, 0] should match
            result = bl.filter("op", runner, [[128, 0], [32, 0]])

            class _FakeRunner:
                __class__ = type("RunnerA", (), {})

            # Use a runner whose __class__.__name__ is "RunnerA"
            fake = _FakeRunner()
            fake.__class__ = type("RunnerA", (), {})
            result = bl.filter("op", fake, [[128, 0], [32, 0]])
            assert result == [[32, 0]]
        finally:
            os.unlink(path)

    def test_save_creates_directories(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "sub", "dir", "blocklist.json")
            TacticsBlocklist.save(path, {"op::R": [1]}, metadata={"gpu": "*"})
            assert os.path.isfile(path)

    def test_save_includes_metadata(self):
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsBlocklist.save(path, {"op::R": [1]}, metadata={"gpu": "TestGPU"})
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
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "AutoGPU", "cuda_version": "13.0"},
        )
        fd, path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        try:
            TacticsBlocklist.save(path, {"op::R": [1]})
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
            TacticsBlocklist.save(
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
    """Verify the blocklist filters tactics inside the autotuner loop."""

    def setup_method(self):
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_choose_one_skips_blocklisted_tactics(self, monkeypatch, tmp_path):
        """Tactics in the blocklist should never be profiled."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
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

        # Create blocklist that blocks tactics 1 and 3
        bl_path = str(tmp_path / "blocklist.json")
        TacticsBlocklist.save(
            bl_path,
            {"test_op::StubRunnerA": [1, 3]},
            metadata={"gpu": "TestGPU"},
        )

        # Load into the autotuner's blocklist
        self.tuner._blocklist.load(bl_path)

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

    def test_no_blocklist_profiles_all(self, monkeypatch):
        """Without a blocklist, all tactics should be profiled as usual."""
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

    def test_env_var_auto_loads_blocklist(self, monkeypatch, tmp_path):
        """Setting FLASHINFER_TACTICS_BLOCKLIST should auto-load on init."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
            lambda: {"gpu": "TestGPU"},
        )

        bl_path = str(tmp_path / "env_bl.json")
        TacticsBlocklist.save(
            bl_path,
            {"env_op::StubRunnerA": [99]},
            metadata={"gpu": "TestGPU"},
        )

        monkeypatch.setenv("FLASHINFER_TACTICS_BLOCKLIST", bl_path)

        AutoTuner._instance = None
        tuner = AutoTuner.get()
        try:
            assert tuner._blocklist.is_loaded
            assert tuner._blocklist.summary() == {"env_op::StubRunnerA": 1}
        finally:
            AutoTuner._instance = None

    def test_choose_one_with_all_tactics_filtered_still_succeeds(
        self, monkeypatch, tmp_path
    ):
        """If blocklist filters ALL tactics for a runner, choose_one should
        gracefully handle it (no crash, returns None or next best)."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
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

        bl_path = str(tmp_path / "all_blocked.json")
        TacticsBlocklist.save(
            bl_path,
            {"block_op::StubRunnerA": [0, 1, 2]},
            metadata={"gpu": "TestGPU"},
        )
        self.tuner._blocklist.load(bl_path)

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

    def test_blocklist_does_not_affect_cache_hits(self, monkeypatch, tmp_path):
        """Cached results should be returned directly, bypassing blocklist filtering."""
        monkeypatch.setattr(
            "flashinfer.tactics_blocklist._collect_metadata",
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

        bl_path = str(tmp_path / "cache_test.json")
        TacticsBlocklist.save(
            bl_path,
            {"cache_op::StubRunnerA": [1, 3]},
            metadata={"gpu": "TestGPU"},
        )
        self.tuner._blocklist.load(bl_path)

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


# ---------------------------------------------------------------------------
# Offline generator (flashinfer generate-tactics-blocklist)
# ---------------------------------------------------------------------------


class TestGeneratorModule:
    """Generator surface that does not require a GPU to exercise."""

    def test_probe_functions_cover_expected_modes(self):
        from flashinfer.tactics_blocklist_gen import PROBE_FUNCTIONS

        assert set(PROBE_FUNCTIONS) == {
            "NvFP4xNvFP4",
            "Fp8-Block",
            "NvFP4-CUTLASS",
            "Fp8-PerTensor-CUTLASS",
            "BF16-CUTLASS",
            "BF16-Relu2-CUTLASS",
        }

    def test_generate_rejects_unknown_quant_mode(self):
        # Validation happens before any CUDA access, so this needs no GPU.
        from flashinfer.tactics_blocklist_gen import generate

        with pytest.raises(ValueError, match="Unknown quant mode"):
            generate(quant_modes=["does-not-exist"])


class TestGeneratorCli:
    """The generator is exposed through the `flashinfer` CLI entry point."""

    def test_subcommand_registered(self):
        from click.testing import CliRunner

        from flashinfer.__main__ import cli

        result = CliRunner().invoke(cli, ["generate-tactics-blocklist", "--help"])
        assert result.exit_code == 0
        assert "--quant-modes" in result.output

    def test_unknown_quant_mode_is_bad_parameter(self):
        from click.testing import CliRunner

        from flashinfer.__main__ import cli

        result = CliRunner().invoke(
            cli, ["generate-tactics-blocklist", "--quant-modes", "does-not-exist"]
        )
        assert result.exit_code == 2  # click usage error
