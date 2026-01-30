"""
Test that the CLI commands work as expected.

In general there can be two types of tests for each command:
- Real tests (with suffix `_real`) that invoke the commands without any mocking
- Mocked tests (with suffix `_mocked`) that use monkeypatch to mock logic that would
  otherwise be slow (e.g. downloading cubins, filesystem calls, etc), and also to
  create deterministic state so we can check for expected output (e.g. number of cubins)

These tests don't require a GPU. CLI tests that require a GPU are in test_cli_cmds_gpu.py.

Note: The `replay` command is tested in tests/utils/test_logging_replay.py alongside
the other logging/replay functionality tests, since it's tightly coupled with that feature.
"""

from .cli_cmd_helpers import (
    _test_cmd_helper,
    _assert_output_contains_all,
    _assert_output_contains_any,
)
from flashinfer.artifacts import ArtifactPath


def test_show_config_cmd_real():
    """
    Test that show-config command works as expected
    """
    out = _test_cmd_helper(["show-config"])

    # Basic sections present
    _assert_output_contains_all(
        out,
        "=== Torch Version Info ===",
        "=== Environment Variables ===",
        "=== Artifact Path ===",
        "=== Downloaded Cubins ===",
    )


def test_show_config_cmd_mocked(monkeypatch):
    """
    Test that show-config command works as but with mocked cubin status
    """
    # Don't check filesystem for cubins
    monkeypatch.setattr(
        "flashinfer.__main__.get_artifacts_status",
        lambda: (("foo.cubin", True), ("bar.cubin", False)),
    )
    # Avoid module registration/inspection
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered",
        lambda: [],
    )

    out = _test_cmd_helper(["show-config"])

    # Uses our monkeypatched data
    assert "Downloaded 1/2 cubins" in out


def test_cli_group_help_real():
    """
    Test that the CLI group runs without error and sanity checks the output
    """
    out = _test_cmd_helper([])
    _assert_output_contains_any(out, "FlashInfer CLI", "Usage")


def test_download_cubin_flag_mocked(monkeypatch):
    # This just tests that the flag is parsed correctly, so we can monkeypatch
    # download_artifacts to avoid the latency of downloading cubins
    monkeypatch.setattr("flashinfer.__main__.download_artifacts", lambda: None)

    out = _test_cmd_helper(["--download-cubin"])
    assert "All cubin download tasks completed successfully" in out


def test_download_cubin_cmd_mocked(monkeypatch):
    """
    Test that download-cubin can download a single cubin using a mocked cubin path
    """
    # Return a real cubin path relative to the repository so it can be downloaded
    fmha_cubin = "fmhaSm100aKernel_QE4m3KvE2m1OE4m3H128PagedKvCausalP32VarSeqQ128Kv128PersistentContext.cubin"

    # Mock get_subdir_file_list to return a list with (filename, checksum) tuples
    def mock_get_subdir_file_list():
        return [(f"{ArtifactPath.TRTLLM_GEN_FMHA}/{fmha_cubin}", "fake_checksum_12345")]

    monkeypatch.setattr(
        "flashinfer.artifacts.get_subdir_file_list", mock_get_subdir_file_list
    )

    # Mock download_file to avoid actual network calls
    monkeypatch.setattr(
        "flashinfer.artifacts.download_file", lambda *_args, **_kwargs: True
    )

    # Mock verify_cubin to always return True
    monkeypatch.setattr("flashinfer.artifacts.verify_cubin", lambda *_args: True)

    out = _test_cmd_helper(["--download-cubin"])
    assert "All cubin download tasks completed successfully" in out


def test_list_cubins_cmd_real():
    out = _test_cmd_helper(["list-cubins"])

    _assert_output_contains_all(out, "Cubin", "Status")


def test_list_cubins_cmd_mocked(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.__main__.get_artifacts_status",
        lambda: (("foo.cubin", True), ("bar.cubin", False)),
    )

    out = _test_cmd_helper(["list-cubins"])
    _assert_output_contains_all(out, "foo.cubin", "bar.cubin")


def test_clear_cache_cmd_mocked(monkeypatch):
    """
    Test that clear-cache command works without actually clearing the cache.

    This doesn't test much, just a basic sanity check.
    """
    monkeypatch.setattr("flashinfer.__main__.clear_cache_dir", lambda: None)

    out = _test_cmd_helper(["clear-cache"])
    assert "Cache cleared successfully" in out


def test_clear_cache_cmd_real(monkeypatch, tmp_path):
    """
    Test that clear-cache command actually clears the cache directory.

    Uses a temporary directory to avoid side effects on the real cache.
    """
    # Create a temporary JIT directory with some dummy cache files
    temp_jit_dir = tmp_path / "cached_ops"
    temp_jit_dir.mkdir(parents=True, exist_ok=True)

    # Create some dummy cached files to simulate a real cache
    dummy_module_dir = temp_jit_dir / "test_module_abc123"
    dummy_module_dir.mkdir(parents=True, exist_ok=True)
    (dummy_module_dir / "test_module.so").write_text("dummy shared library")
    (dummy_module_dir / "build.ninja").write_text("dummy build file")

    # Monkeypatch the FLASHINFER_JIT_DIR to point to our temp directory
    monkeypatch.setattr("flashinfer.jit.core.jit_env.FLASHINFER_JIT_DIR", temp_jit_dir)

    # Verify the cache directory exists before clearing
    assert temp_jit_dir.exists()
    assert (dummy_module_dir / "test_module.so").exists()

    # Run the clear-cache command
    out = _test_cmd_helper(["clear-cache"])
    assert "Cache cleared successfully" in out

    # Verify the cache directory has been removed
    assert not temp_jit_dir.exists()


def test_clear_cubin_cmd_mocked(monkeypatch):
    """
    Test that clear-cubin command works without actually clearing the cubin.

    This doesn't test much, just a basic sanity check.
    """
    monkeypatch.setattr("flashinfer.__main__.clear_cubin", lambda: None)

    out = _test_cmd_helper(["clear-cubin"])
    assert "Cubin cleared successfully" in out


def test_clear_cubin_cmd_real(monkeypatch, tmp_path):
    """
    Test that clear-cubin command actually clears the cubin directory.

    Uses a temporary directory to avoid side effects on the real cubins.
    """
    # Create a temporary cubin directory with some dummy cubin files
    temp_cubin_dir = tmp_path / "cubins"
    temp_cubin_dir.mkdir(parents=True, exist_ok=True)

    # Create some dummy cubin files to simulate real cubins
    dummy_cubin_subdir = temp_cubin_dir / "trtllm_gen_fmha"
    dummy_cubin_subdir.mkdir(parents=True, exist_ok=True)
    (dummy_cubin_subdir / "test_kernel.cubin").write_text("dummy cubin data")
    (dummy_cubin_subdir / "checksums.txt").write_text("abc123 test_kernel.cubin")

    # Monkeypatch FLASHINFER_CUBIN_DIR to point to our temp directory
    # Need to patch it in multiple places where it's imported
    monkeypatch.setattr("flashinfer.artifacts.FLASHINFER_CUBIN_DIR", temp_cubin_dir)
    monkeypatch.setattr(
        "flashinfer.jit.cubin_loader.FLASHINFER_CUBIN_DIR", temp_cubin_dir
    )

    # Verify the cubin directory exists before clearing
    assert temp_cubin_dir.exists()
    assert (dummy_cubin_subdir / "test_kernel.cubin").exists()

    # Run the clear-cubin command
    out = _test_cmd_helper(["clear-cubin"])
    assert "Cubin cleared successfully" in out

    # Verify the cubin directory has been removed
    assert not temp_cubin_dir.exists()


class MockJitSpec:
    """Mock JitSpec for testing export-compile-commands."""

    def __init__(self, name, compile_commands):
        self.name = name
        self._compile_commands = compile_commands

    def get_compile_commands(self):
        return self._compile_commands


def test_export_compile_commands_mocked(monkeypatch, tmp_path):
    """
    Test that export-compile-commands writes correct JSON output.
    """
    # Create mock specs with compile commands
    mock_specs = {
        "module_a": MockJitSpec(
            "module_a",
            [
                {
                    "directory": "/path/to/build",
                    "command": "nvcc -c kernel_a.cu",
                    "file": "kernel_a.cu",
                }
            ],
        ),
        "module_b": MockJitSpec(
            "module_b",
            [
                {
                    "directory": "/path/to/build",
                    "command": "nvcc -c kernel_b.cu",
                    "file": "kernel_b.cu",
                }
            ],
        ),
    }

    monkeypatch.setattr("flashinfer.__main__._ensure_modules_registered", lambda: [])
    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_all_specs", lambda: mock_specs
    )

    # Use tmp_path to write output file
    output_file = tmp_path / "compile_commands.json"
    out = _test_cmd_helper(["export-compile-commands", str(output_file)])

    assert "Successfully exported 2 compile commands" in out
    assert output_file.exists()

    # Verify JSON content
    import json

    with open(output_file) as f:
        commands = json.load(f)

    assert len(commands) == 2
    assert commands[0]["file"] == "kernel_a.cu"
    assert commands[1]["file"] == "kernel_b.cu"


def test_export_compile_commands_output_option(monkeypatch, tmp_path):
    """
    Test that --output option overrides PATH argument.
    """
    mock_specs = {
        "module_a": MockJitSpec(
            "module_a",
            [{"directory": "/build", "command": "nvcc -c a.cu", "file": "a.cu"}],
        ),
    }

    monkeypatch.setattr("flashinfer.__main__._ensure_modules_registered", lambda: [])
    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_all_specs", lambda: mock_specs
    )

    # PATH argument should be ignored when --output is specified
    output_file = tmp_path / "custom_output.json"
    ignored_file = tmp_path / "ignored.json"
    out = _test_cmd_helper(
        ["export-compile-commands", str(ignored_file), "--output", str(output_file)]
    )

    assert "Successfully exported 1 compile commands" in out
    assert output_file.exists()
    assert not ignored_file.exists()


def test_export_compile_commands_no_modules(monkeypatch, tmp_path):
    """
    Test that export-compile-commands handles empty module registry.
    """
    monkeypatch.setattr("flashinfer.__main__._ensure_modules_registered", lambda: [])
    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_all_specs", lambda: {}
    )

    output_file = tmp_path / "compile_commands.json"
    out = _test_cmd_helper(["export-compile-commands", str(output_file)])

    assert "No modules found" in out
    # File should not be created when no modules exist
    assert not output_file.exists()
