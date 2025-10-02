from click.testing import CliRunner

from flashinfer.__main__ import cli
from flashinfer.artifacts import ArtifactPath

"""
Test that the CLI commands work as expected.

In general there can be two types of tests for each command:
- Real tests (with suffix `_real`) that invoke the commands without any mocking
- Mocked tests (with suffix `_mocked`) that use monkeypatch to mock logic that would
  otherwise be slow (e.g. downloading cubins, filesystem calls, etc), and also to
  create deterministic state so we can check for expected output (e.g. number of cubins)
"""


def _test_show_config_cmd_helper():
    """
    Helper for show-config tests

    Returns the output of the command
    """
    runner = CliRunner()
    result = runner.invoke(cli, ["show-config"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Basic sections present
    assert "=== Torch Version Info ===" in out
    assert "=== Environment Variables ===" in out
    assert "=== Artifact Path ===" in out
    assert "=== Downloaded Cubins ===" in out

    return out


def test_show_config_cmd_real():
    """
    Test that show-config command works as expected
    """
    _ = _test_show_config_cmd_helper()


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

    out = _test_show_config_cmd_helper()

    # Uses our monkeypatched data
    assert "Downloaded 1/2 cubins" in out


def test_cli_group_help_real():
    """
    Test that the CLI group runs without error and sanity checks the output
    """
    runner = CliRunner()
    result = runner.invoke(cli, [])
    assert result.exit_code == 0, result.output
    assert "FlashInfer CLI" in result.output or "Usage" in result.output


def test_download_cubin_flag_mocked(monkeypatch):
    # This just tests that the flag is parsed correctly, so we can monkeypatch
    # download_artifacts to avoid the latency of downloading cubins
    monkeypatch.setattr("flashinfer.__main__.download_artifacts", lambda: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["--download-cubin"])
    assert result.exit_code == 0, result.output
    assert "All cubin download tasks completed successfully" in result.output


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

    runner = CliRunner()
    result = runner.invoke(cli, ["download-cubin"])
    assert result.exit_code == 0, result.output
    assert "All cubin download tasks completed successfully" in result.output


def test_list_cubins_cmd_mocked(monkeypatch):
    monkeypatch.setattr(
        "flashinfer.__main__.get_artifacts_status",
        lambda: (("foo.cubin", True), ("bar.cubin", False)),
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["list-cubins"])
    assert result.exit_code == 0, result.output
    out = result.output
    assert "foo.cubin" in out and "bar.cubin" in out


def test_clear_cache_cmd_mocked(monkeypatch):
    monkeypatch.setattr("flashinfer.__main__.clear_cache_dir", lambda: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["clear-cache"])
    assert result.exit_code == 0, result.output
    assert "Cache cleared successfully" in result.output


def test_clear_cubin_cmd_mocked(monkeypatch):
    monkeypatch.setattr("flashinfer.__main__.clear_cubin", lambda: None)

    runner = CliRunner()
    result = runner.invoke(cli, ["clear-cubin"])
    assert result.exit_code == 0, result.output
    assert "Cubin cleared successfully" in result.output


def test_module_status_cmd_mocked(monkeypatch):
    # Avoid module registration/inspection
    monkeypatch.setattr("flashinfer.__main__._ensure_modules_registered", lambda: [])

    runner = CliRunner()
    result = runner.invoke(cli, ["module-status"])
    assert result.exit_code == 0, result.output


def test_list_modules_cmd_mocked(monkeypatch):
    # Avoid module registration/inspection
    monkeypatch.setattr("flashinfer.__main__._ensure_modules_registered", lambda: [])

    runner = CliRunner()
    result = runner.invoke(cli, ["list-modules"])
    assert result.exit_code == 0, result.output
