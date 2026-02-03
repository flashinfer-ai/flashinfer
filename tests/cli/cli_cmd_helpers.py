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


def _assert_output_contains_all(output, *expected_strings):
    """Assert that output contains all expected strings."""
    missing = [s for s in expected_strings if s not in output]
    assert not missing, (
        f"Missing strings in output: {missing}\n\nActual output:\n{output}"
    )


def _assert_output_contains_any(output, *expected_strings):
    """Assert that output contains at least one of the expected strings."""
    found = any(s in output for s in expected_strings)
    assert found, (
        f"None of the expected strings were found in output: {expected_strings}\n\nActual output:\n{output}"
    )
