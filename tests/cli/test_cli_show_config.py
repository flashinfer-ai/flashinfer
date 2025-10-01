from click.testing import CliRunner

from flashinfer.__main__ import cli


def test_show_config_cmd_smoke(monkeypatch):
    # Avoid network/FS/GPU dependencies by monkeypatching helpers
    monkeypatch.setattr(
        "flashinfer.__main__.get_artifacts_status",
        lambda: (("foo.cubin", True), ("bar.cubin", False)),
    )
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered",
        lambda: [],
    )

    runner = CliRunner()
    result = runner.invoke(cli, ["show-config"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Basic sections present
    assert "=== Torch Version Info ===" in out
    assert "=== Environment Variables ===" in out
    assert "=== Artifact Path ===" in out
    assert "=== Downloaded Cubins ===" in out

    # Uses our monkeypatched data
    assert "Downloaded 1/2 cubins" in out
