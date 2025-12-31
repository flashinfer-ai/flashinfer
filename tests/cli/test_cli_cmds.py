"""
Test that the CLI commands work as expected.

In general there can be two types of tests for each command:
- Real tests (with suffix `_real`) that invoke the commands without any mocking
- Mocked tests (with suffix `_mocked`) that use monkeypatch to mock logic that would
  otherwise be slow (e.g. downloading cubins, filesystem calls, etc), and also to
  create deterministic state so we can check for expected output (e.g. number of cubins)

These tests don't require a GPU. CLI tests that require a GPU are in test_cli_cmds_gpu.py.
"""

from cli_cmd_helpers import _test_cmd_helper
from flashinfer.artifacts import ArtifactPath


def test_show_config_cmd_real():
    """
    Test that show-config command works as expected
    """
    out = _test_cmd_helper(["show-config"])

    # Basic sections present
    assert "=== Torch Version Info ===" in out
    assert "=== Environment Variables ===" in out
    assert "=== Artifact Path ===" in out
    assert "=== Downloaded Cubins ===" in out


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
    assert "FlashInfer CLI" in out or "Usage" in out


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
    monkeypatch.setattr(
        "flashinfer.artifacts.get_cubin_file_list",
        lambda: [f"{ArtifactPath.TRTLLM_GEN_FMHA}/{fmha_cubin}"],
    )

    out = _test_cmd_helper(["--download-cubin"])
    assert "All cubin download tasks completed successfully" in out


def test_list_cubins_cmd_real(monkeypatch):
    out = _test_cmd_helper(["list-cubins"])
    assert "Cubin" in out and "Status" in out


def test_list_cubins_cmd_mocked(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.__main__.get_artifacts_status",
        lambda: (("foo.cubin", True), ("bar.cubin", False)),
    )

    out = _test_cmd_helper(["list-cubins"])
    assert "foo.cubin" in out and "bar.cubin" in out


def test_clear_cache_cmd_mocked(monkeypatch):
    """
    Test that clear-cache command works without actually clearing the cache.

    This doesn't test much, just a basic sanity check.
    """
    monkeypatch.setattr("flashinfer.__main__.clear_cache_dir", lambda: None)

    out = _test_cmd_helper(["clear-cache"])
    assert "Cache cleared successfully" in out


# TODO: add test that actually clears the cache
# need to check that there aren't side effects if we do this


def test_clear_cubin_cmd_mocked(monkeypatch):
    """
    Test that clear-cubin command works without actually clearing the cubin.

    This doesn't test much, just a basic sanity check.
    """
    monkeypatch.setattr("flashinfer.__main__.clear_cubin", lambda: None)

    out = _test_cmd_helper(["clear-cubin"])
    assert "Cubin cleared successfully" in out


# TODO: add test that actually clears the cubins
# need to check that there aren't side effects if we do this
