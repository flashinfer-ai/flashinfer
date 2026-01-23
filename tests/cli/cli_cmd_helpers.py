from click.testing import CliRunner

from flashinfer.__main__ import cli


def _test_cmd_helper(cmd: list[str]):
    """
    Helper for command tests
    """
    runner = CliRunner()
    result = runner.invoke(cli, cmd)
    assert result.exit_code == 0, result.output
    return result.output
