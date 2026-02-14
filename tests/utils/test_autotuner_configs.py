"""
Tests for AutoTuner save_configs / load_configs functionality.

These tests exercise the JSON-based config save/load round-trip,
autotune(cache=...) context manager, and the fallback chain in search_cache().

No GPU is required for these tests — they use mock data to populate
the profiling cache and verify serialization behavior.
"""

import json
import os
import tempfile

import pytest

from flashinfer.autotuner import (
    AutoTuner,
    TunableRunner,
    TuningConfig,
    _json_to_tactic,
    _tactic_to_json,
    autotune,
)


# ---------------------------------------------------------------------------
# Minimal TunableRunner stubs for testing
# ---------------------------------------------------------------------------


class FakeRunnerA(TunableRunner):
    """A minimal TunableRunner for testing."""

    def __init__(self, value=0):
        self.value = value

    def get_valid_tactics(self, inputs, profile):
        return [0, 1, 2]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return None


class FakeRunnerB(TunableRunner):
    """A second TunableRunner with a different class name."""

    def __init__(self, value=0):
        self.value = value

    def get_valid_tactics(self, inputs, profile):
        return [0, 1]

    def forward(self, inputs, tactic=-1, do_preparation=False, **kwargs):
        return None


# ---------------------------------------------------------------------------
# Helper to populate the profiling cache with fake entries
# ---------------------------------------------------------------------------

_TUNING_CONFIG = TuningConfig()


def _populate_cache(tuner, runner, custom_op, profile, tactic, runner_id=0):
    """Insert a fake entry into the profiling cache."""
    cache_key = AutoTuner._get_cache_key(custom_op, runner, profile, _TUNING_CONFIG)
    tuner.profiling_cache[cache_key] = (runner_id, tactic, None)


# ---------------------------------------------------------------------------
# Tests: tactic serialization helpers
# ---------------------------------------------------------------------------


class TestTacticConversion:
    def test_int_tactic_roundtrip(self):
        assert _tactic_to_json(5) == 5
        assert _json_to_tactic(5) == 5

    def test_tuple_tactic_to_json(self):
        tactic = (128, (64, 64), (2, 1), True)
        result = _tactic_to_json(tactic)
        assert result == [128, [64, 64], [2, 1], True]

    def test_json_to_tuple_tactic(self):
        json_val = [128, [64, 64], [2, 1], True]
        result = _json_to_tactic(json_val)
        assert result == (128, (64, 64), (2, 1), True)

    def test_nested_roundtrip(self):
        tactic = (256, (128, 128), (1, 2), False)
        assert _json_to_tactic(_tactic_to_json(tactic)) == tactic

    def test_simple_int_list_roundtrip(self):
        """A tactic that is a tuple of ints."""
        tactic = (1, 2, 3)
        json_val = _tactic_to_json(tactic)
        assert json_val == [1, 2, 3]
        assert _json_to_tactic(json_val) == tactic


# ---------------------------------------------------------------------------
# Tests: save_configs / load_configs round-trip
# ---------------------------------------------------------------------------


class TestSaveLoadRoundTrip:
    def setup_method(self):
        """Reset the singleton AutoTuner for each test."""
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_save_and_load_basic(self):
        runner = FakeRunnerA(value=42)
        profile = (
            (1, 3584),
            (256, 512, 448),
        )
        _populate_cache(self.tuner, runner, "test::op1", profile, tactic=5)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            # Read and verify JSON structure (flat dict, no metadata wrapper)
            with open(tmp_path, "r") as f:
                data = json.load(f)

            assert isinstance(data, dict)
            assert len(data) == 1

            # Verify config entry uses runner class name, not runner_id
            config_values = list(data.values())
            runner_name, tactic = config_values[0]
            assert runner_name == "FakeRunnerA"
            assert tactic == 5

            # Load into a fresh tuner and verify
            self.tuner.clear_cache()
            assert len(self.tuner._file_configs) == 0

            self.tuner.load_configs(tmp_path)
            assert len(self.tuner._file_configs) == 1

            # Verify the loaded entry
            loaded_key = list(self.tuner._file_configs.keys())[0]
            loaded_runner_name, loaded_tactic = self.tuner._file_configs[loaded_key]
            assert loaded_runner_name == "FakeRunnerA"
            assert loaded_tactic == 5
        finally:
            os.unlink(tmp_path)

    def test_save_and_load_tuple_tactic(self):
        """Round-trip with a compound tuple tactic (CuteDSL-style)."""
        runner = FakeRunnerA(value=1)
        profile = ((64, 1024),)
        tactic = (128, (64, 64), (2, 1), True)
        _populate_cache(self.tuner, runner, "cute_dsl::moe", profile, tactic=tactic)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            self.tuner.clear_cache()
            self.tuner.load_configs(tmp_path)

            loaded_runner_name, loaded_tactic = list(self.tuner._file_configs.values())[
                0
            ]
            assert loaded_runner_name == "FakeRunnerA"
            assert loaded_tactic == tactic
            assert isinstance(loaded_tactic, tuple)
            assert isinstance(loaded_tactic[1], tuple)
        finally:
            os.unlink(tmp_path)

    def test_save_multiple_entries(self):
        runner_a = FakeRunnerA(value=1)
        runner_b = FakeRunnerB(value=2)
        _populate_cache(self.tuner, runner_a, "op1", ((10, 20),), tactic=3)
        _populate_cache(self.tuner, runner_b, "op2", ((30, 40),), tactic=7)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)
            assert len(data) == 2

            self.tuner.clear_cache()
            self.tuner.load_configs(tmp_path)
            assert len(self.tuner._file_configs) == 2
        finally:
            os.unlink(tmp_path)

    def test_save_empty_cache(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)
            assert data == {}
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: load_configs error handling
# ---------------------------------------------------------------------------


class TestLoadConfigsErrors:
    def setup_method(self):
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_load_nonexistent_file_raises(self):
        with pytest.raises(FileNotFoundError):
            self.tuner.load_configs("/nonexistent/path/config.json")

    def test_load_invalid_json_raises(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not valid json {{{")
            tmp_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                self.tuner.load_configs(tmp_path)
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: search_cache fallback chain
# ---------------------------------------------------------------------------


class TestSearchCacheFallbackChain:
    def setup_method(self):
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_in_memory_cache_has_priority(self):
        """In-memory cache should be checked before file configs."""
        runner = FakeRunnerA(value=1)
        profile = ((10, 20),)

        # Populate both in-memory cache and file configs
        _populate_cache(self.tuner, runner, "op1", profile, tactic=99)

        cache_key = AutoTuner._get_cache_key("op1", runner, profile, _TUNING_CONFIG)
        file_key = str((cache_key[0], cache_key[1], cache_key[3]))
        self.tuner._file_configs[file_key] = ("FakeRunnerA", 55)

        is_hit, runner_id, tactic, _ = self.tuner.search_cache(
            "op1", [runner], profile, _TUNING_CONFIG
        )
        assert is_hit
        assert tactic == 99  # in-memory, not file's 55

    def test_file_configs_used_on_memory_miss(self):
        """File configs should be used when in-memory cache misses."""
        runner = FakeRunnerA(value=1)
        profile = ((10, 20),)

        cache_key = AutoTuner._get_cache_key("op1", runner, profile, _TUNING_CONFIG)
        file_key = str((cache_key[0], cache_key[1], cache_key[3]))
        self.tuner._file_configs[file_key] = ("FakeRunnerA", 42)

        is_hit, runner_id, tactic, _ = self.tuner.search_cache(
            "op1", [runner], profile, _TUNING_CONFIG
        )
        assert is_hit
        assert tactic == 42
        assert runner_id == 0

    def test_runner_name_resolution(self):
        """File config runner name should resolve to correct index in runners list."""
        runner_a = FakeRunnerA(value=1)
        runner_b = FakeRunnerB(value=2)
        profile = ((10, 20),)

        # File config says FakeRunnerB won
        cache_key = AutoTuner._get_cache_key("op1", runner_b, profile, _TUNING_CONFIG)
        file_key = str((cache_key[0], cache_key[1], cache_key[3]))
        self.tuner._file_configs[file_key] = ("FakeRunnerB", 11)

        # Pass runners in order [A, B] — B is at index 1
        is_hit, runner_id, tactic, _ = self.tuner.search_cache(
            "op1", [runner_a, runner_b], profile, _TUNING_CONFIG
        )
        assert is_hit
        assert runner_id == 1  # FakeRunnerB is at index 1
        assert tactic == 11

    def test_fallback_when_nothing_cached(self):
        """Should return fallback when no source has a match."""
        runner = FakeRunnerA(value=1)
        is_hit, runner_id, tactic, _ = self.tuner.search_cache(
            "op_not_cached", [runner], ((999, 999),), _TUNING_CONFIG
        )
        assert not is_hit
        assert runner_id == 0
        assert tactic == -1

    def test_clear_cache_resets_all(self):
        """clear_cache should reset in-memory cache and file configs."""
        runner = FakeRunnerA(value=1)
        _populate_cache(self.tuner, runner, "op1", ((1, 2),), tactic=5)
        self.tuner._file_configs["some_key"] = ("FakeRunnerA", 3)

        self.tuner.clear_cache()

        assert len(self.tuner.profiling_cache) == 0
        assert len(self.tuner._file_configs) == 0


# ---------------------------------------------------------------------------
# Tests: end-to-end save -> load -> search_cache
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def setup_method(self):
        AutoTuner._instance = None
        self.tuner = AutoTuner.get()

    def teardown_method(self):
        AutoTuner._instance = None

    def test_save_then_load_then_search(self):
        """Full workflow: tune -> save -> clear -> load -> search hits."""
        runner = FakeRunnerA(value=10)
        profile = ((32, 7168),)
        _populate_cache(self.tuner, runner, "my_op", profile, tactic=3)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            # Save
            self.tuner.save_configs(tmp_path)

            # Clear everything
            self.tuner.clear_cache()
            assert len(self.tuner.profiling_cache) == 0
            assert len(self.tuner._file_configs) == 0

            # Load
            self.tuner.load_configs(tmp_path)

            # Search should hit
            is_hit, runner_id, tactic, _ = self.tuner.search_cache(
                "my_op", [runner], profile, _TUNING_CONFIG
            )
            assert is_hit
            assert runner_id == 0
            assert tactic == 3
        finally:
            os.unlink(tmp_path)

    def test_save_merges_loaded_and_profiled_configs(self):
        """save_configs should include both loaded file configs and in-memory profiled configs."""
        runner_b = FakeRunnerB(value=2)

        # Simulate loaded file configs
        self.tuner._file_configs["('old_op', 'FakeRunnerA', ((1, 2),))"] = (
            "FakeRunnerA",
            10,
        )

        # Add new in-memory profiled result
        _populate_cache(self.tuner, runner_b, "new_op", ((3, 4),), tactic=20)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)

            # Should contain both entries
            assert len(data) == 2

            # Verify old loaded config is included
            assert "('old_op', 'FakeRunnerA', ((1, 2),))" in data
            old_entry = data["('old_op', 'FakeRunnerA', ((1, 2),))"]
            assert old_entry == ["FakeRunnerA", 10]
        finally:
            os.unlink(tmp_path)

    def test_save_profiled_overrides_loaded_for_same_key(self):
        """In-memory profiled results should override loaded file configs for the same key."""
        runner = FakeRunnerA(value=1)
        profile = ((10, 20),)

        # Simulate a loaded config for this key
        cache_key = AutoTuner._get_cache_key("op1", runner, profile, _TUNING_CONFIG)
        file_key = str((cache_key[0], cache_key[1], cache_key[3]))
        self.tuner._file_configs[file_key] = ("FakeRunnerA", 99)

        # Add a newer in-memory result for the same op/runner/profile
        _populate_cache(self.tuner, runner, "op1", profile, tactic=42)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            self.tuner.save_configs(tmp_path)

            with open(tmp_path, "r") as f:
                data = json.load(f)

            # Only one entry (merged), with the in-memory value
            assert len(data) == 1
            entry = list(data.values())[0]
            assert entry[1] == 42  # in-memory wins over loaded 99
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Tests: autotune(cache=...) context manager
# ---------------------------------------------------------------------------


class TestAutotuneCache:
    def setup_method(self):
        AutoTuner._instance = None

    def teardown_method(self):
        AutoTuner._instance = None

    def test_autotune_cache_loads_on_entry(self):
        """autotune(cache=path) should load configs from the file on entry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump({"('op1', 'FakeRunnerA', ((1, 2),))": ["FakeRunnerA", 7]}, f)
            tmp_path = f.name

        try:
            tuner = AutoTuner.get()
            assert len(tuner._file_configs) == 0

            with autotune(False, cache=tmp_path):
                assert len(tuner._file_configs) == 1
        finally:
            os.unlink(tmp_path)

    def test_autotune_cache_saves_on_exit_when_tuning(self):
        """autotune(True, cache=path) should save configs on exit."""
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "cache.json")

        try:
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            _populate_cache(tuner, runner, "op1", ((1, 2),), tactic=5)

            with autotune(True, cache=tmp_path):
                pass  # configs already in cache from _populate_cache

            # Verify file was written
            with open(tmp_path, "r") as f:
                data = json.load(f)
            assert len(data) == 1
            entry = list(data.values())[0]
            assert entry == ["FakeRunnerA", 5]
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os.rmdir(tmp_dir)

    def test_autotune_cache_no_save_when_not_tuning(self):
        """autotune(False, cache=path) should NOT save on exit."""
        # Write a valid JSON file with a known marker config
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "cache.json")
        marker_data = {"marker_key": ["MarkerRunner", 99]}
        with open(tmp_path, "w") as f:
            json.dump(marker_data, f)

        try:
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            _populate_cache(tuner, runner, "op1", ((1, 2),), tactic=5)

            with autotune(False, cache=tmp_path):
                pass

            # File should NOT have been overwritten (tune_mode=False)
            # It should still contain only the original marker data
            with open(tmp_path, "r") as f:
                data = json.load(f)
            assert data == marker_data
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os.rmdir(tmp_dir)

    def test_autotune_cache_nonexistent_file_creates_on_exit(self):
        """First run: cache file doesn't exist yet, should be created on exit."""
        tmp_dir = tempfile.mkdtemp()
        cache_path = os.path.join(tmp_dir, "new_cache.json")

        try:
            assert not os.path.exists(cache_path)

            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            _populate_cache(tuner, runner, "op1", ((1, 2),), tactic=9)

            with autotune(True, cache=cache_path):
                pass

            # File should now exist with the config
            assert os.path.exists(cache_path)
            with open(cache_path, "r") as f:
                data = json.load(f)
            assert len(data) == 1
        finally:
            if os.path.exists(cache_path):
                os.unlink(cache_path)
            os.rmdir(tmp_dir)

    def test_autotune_cache_no_cache_param(self):
        """autotune() without cache= should behave as before (no file I/O)."""
        tuner = AutoTuner.get()
        assert len(tuner._file_configs) == 0

        with autotune(True):
            pass

        # No file configs loaded, no errors
        assert len(tuner._file_configs) == 0

    def test_full_workflow_tune_then_inference(self):
        """End-to-end: autotune(True, cache=) then autotune(False, cache=)."""
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "cache.json")

        try:
            # Phase 1: Tune and save
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=42)
            profile = ((128, 4096),)
            _populate_cache(tuner, runner, "e2e_op", profile, tactic=13)

            with autotune(True, cache=tmp_path):
                pass

            # Phase 2: Fresh tuner, load and use cached configs
            AutoTuner._instance = None
            tuner2 = AutoTuner.get()

            with autotune(False, cache=tmp_path):
                is_hit, runner_id, tactic, _ = tuner2.search_cache(
                    "e2e_op", [runner], profile, _TUNING_CONFIG
                )
                assert is_hit
                assert tactic == 13
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os.rmdir(tmp_dir)

    def test_incremental_cache_across_blocks(self):
        """Multiple autotune(True, cache=) blocks should accumulate configs."""
        tmp_dir = tempfile.mkdtemp()
        tmp_path = os.path.join(tmp_dir, "cache.json")

        try:
            runner_a = FakeRunnerA(value=1)
            runner_b = FakeRunnerB(value=2)

            # Block 1: tune op1
            tuner = AutoTuner.get()
            _populate_cache(tuner, runner_a, "op1", ((1, 2),), tactic=10)
            with autotune(True, cache=tmp_path):
                pass

            with open(tmp_path, "r") as f:
                data = json.load(f)
            assert len(data) == 1

            # Block 2: tune op2 (should merge with op1)
            _populate_cache(tuner, runner_b, "op2", ((3, 4),), tactic=20)
            with autotune(True, cache=tmp_path):
                pass

            with open(tmp_path, "r") as f:
                data = json.load(f)
            # Should have both op1 (from loaded file) and op2 (from profiling)
            assert len(data) == 2
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            os.rmdir(tmp_dir)

    def test_cache_hit_skips_profiling_during_tune(self):
        """Loaded configs should be used even during autotune(True), skipping profiling."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name

        try:
            # Create a cache file with a known config
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            profile = ((10, 20),)
            _populate_cache(tuner, runner, "op1", profile, tactic=42)
            tuner.save_configs(tmp_path)

            # Reset tuner
            AutoTuner._instance = None
            tuner2 = AutoTuner.get()

            # Use autotune(True, cache=) — should load and hit, not re-profile
            with autotune(True, cache=tmp_path):
                is_hit, runner_id, tactic, _ = tuner2.search_cache(
                    "op1", [runner], profile, _TUNING_CONFIG
                )
                assert is_hit
                assert tactic == 42
                # Should be in _file_configs, not profiling_cache
                assert len(tuner2.profiling_cache) == 0
                assert len(tuner2._file_configs) == 1
        finally:
            os.unlink(tmp_path)


# ---------------------------------------------------------------------------
# Atomic write tests
# ---------------------------------------------------------------------------


class TestAtomicWrite:
    """Tests that save_configs uses atomic writes (tempfile + rename)."""

    def setup_method(self):
        AutoTuner._instance = None

    def teardown_method(self):
        AutoTuner._instance = None

    def test_save_does_not_leave_partial_file_on_success(self):
        """After a successful save, only the target file should exist (no temp files)."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = os.path.join(tmp_dir, "configs.json")
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            profile = ((4, 8),)
            _populate_cache(tuner, runner, "op1", profile, tactic=5)
            tuner.save_configs(target)

            # Target file should exist and be valid JSON
            assert os.path.isfile(target)
            with open(target, "r") as f:
                data = json.load(f)
            assert len(data) == 1

            # No temp files should remain in the directory
            files = os.listdir(tmp_dir)
            assert files == ["configs.json"], f"Unexpected files: {files}"

    def test_save_creates_parent_directories(self):
        """save_configs should create intermediate directories."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = os.path.join(tmp_dir, "sub", "dir", "configs.json")
            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            profile = ((4, 8),)
            _populate_cache(tuner, runner, "op1", profile, tactic=5)
            tuner.save_configs(target)
            assert os.path.isfile(target)

    def test_save_is_atomic_for_concurrent_readers(self):
        """A reader should never see a partially-written file.

        We verify this by checking that the file is always valid JSON,
        even when re-saved multiple times.
        """
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = os.path.join(tmp_dir, "configs.json")
            tuner = AutoTuner.get()

            # First save
            runner = FakeRunnerA(value=1)
            _populate_cache(tuner, runner, "op1", ((4, 8),), tactic=5)
            tuner.save_configs(target)

            # Second save (overwrites atomically)
            _populate_cache(tuner, runner, "op2", ((16, 32),), tactic=7)
            tuner.save_configs(target)

            # File should be valid JSON with 2 entries
            with open(target, "r") as f:
                data = json.load(f)
            assert len(data) == 2

    def test_save_replaces_existing_file(self):
        """os.replace should overwrite the previous file atomically."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            target = os.path.join(tmp_dir, "configs.json")

            # Write initial content
            with open(target, "w") as f:
                json.dump({"old": "data"}, f)

            tuner = AutoTuner.get()
            runner = FakeRunnerA(value=1)
            _populate_cache(tuner, runner, "op1", ((4, 8),), tactic=5)
            tuner.save_configs(target)

            with open(target, "r") as f:
                data = json.load(f)
            # Should only contain the new config, not "old"
            assert "old" not in data
            assert len(data) == 1


# ---------------------------------------------------------------------------
# Thread safety tests
# ---------------------------------------------------------------------------


class TestThreadSafety:
    """Tests for thread-safe AutoTuner operations."""

    def setup_method(self):
        AutoTuner._instance = None

    def teardown_method(self):
        AutoTuner._instance = None

    def test_singleton_is_thread_safe(self):
        """Multiple threads calling AutoTuner.get() should get the same instance."""
        import threading

        results = []
        barrier = threading.Barrier(10)

        def get_instance():
            barrier.wait()
            results.append(id(AutoTuner.get()))

        threads = [threading.Thread(target=get_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All threads should get the same singleton instance
        assert len(set(results)) == 1

    def test_concurrent_search_cache(self):
        """Multiple threads reading from search_cache should not crash."""
        import threading

        tuner = AutoTuner.get()
        runner = FakeRunnerA(value=1)
        profile = ((4, 8),)
        _populate_cache(tuner, runner, "op1", profile, tactic=5)

        errors = []
        barrier = threading.Barrier(10)

        def do_search():
            try:
                barrier.wait()
                for _ in range(100):
                    is_hit, _, tactic, _ = tuner.search_cache(
                        "op1", [runner], profile, _TUNING_CONFIG
                    )
                    assert is_hit
                    assert tactic == 5
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=do_search) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors in threads: {errors}"

    def test_concurrent_load_and_search(self):
        """One thread loading configs while others search should not crash."""
        import threading

        tuner = AutoTuner.get()
        runner = FakeRunnerA(value=1)
        profile = ((4, 8),)

        # Create a cache file
        _populate_cache(tuner, runner, "op1", profile, tactic=5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            tmp_path = f.name
        try:
            tuner.save_configs(tmp_path)

            errors = []
            barrier = threading.Barrier(6)

            def do_load():
                try:
                    barrier.wait()
                    for _ in range(20):
                        tuner.load_configs(tmp_path)
                except Exception as e:
                    errors.append(e)

            def do_search():
                try:
                    barrier.wait()
                    for _ in range(100):
                        tuner.search_cache("op1", [runner], profile, _TUNING_CONFIG)
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=do_load)] + [
                threading.Thread(target=do_search) for _ in range(5)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0, f"Errors in threads: {errors}"
        finally:
            os.unlink(tmp_path)

    def test_autotune_mode_flag_thread_safety(self):
        """Concurrent autotune() contexts should not corrupt is_tuning_mode."""
        import threading

        tuner = AutoTuner.get()
        errors = []
        barrier = threading.Barrier(5)

        def run_autotune():
            try:
                barrier.wait()
                for _ in range(50):
                    with autotune(True):
                        # Brief work inside tuning mode
                        pass
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=run_autotune) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # After all threads complete, tuning mode should be restored to False
        assert not tuner.is_tuning_mode, "is_tuning_mode was not properly restored"
        assert len(errors) == 0, f"Errors in threads: {errors}"
