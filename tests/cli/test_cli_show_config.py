from click.testing import CliRunner

from flashinfer.__main__ import cli


def test_show_config_cmd_smoke(monkeypatch):
    runner = CliRunner()
    result = runner.invoke(cli, ["show-config"])
    assert result.exit_code == 0, result.output
    out = result.output

    # Basic sections present
    assert "=== Torch Version Info ===" in out
    assert "=== Environment Variables ===" in out
    assert "=== Artifact Path ===" in out
    assert "=== Downloaded Cubins ===" in out
