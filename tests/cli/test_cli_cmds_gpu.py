"""
Tests the module-status and list-modules commands

This is factored out from test_cli_cmds.py because these tests require a GPU.
"""

from .cli_cmd_helpers import _test_cmd_helper


_MOCKED_CUDA_ARCH_LIST = "7.5 8.0 8.9 9.0a 10.0a"


def test_module_status_cmd_mocked(monkeypatch):
    """
    Test that module-status command runs without error and sanity checks the output

    The only mock is to set the CUDA architecture list via monkeypatch, for isolation.
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)
    out = _test_cmd_helper(["module-status"])
    assert "=== Summary ===" in out
    assert "Total modules:" in out
    assert "AOT compiled:" in out
    assert "JIT compiled:" in out
    assert "Not compiled:" in out


# TODO: test module-status command with different filters
# TODO: test module-status command with detailed output


def test_list_modules_cmd_mocked(monkeypatch):
    """
    Test that list-modules command runs without error and sanity checks the output

    The only mock is to set the CUDA architecture list via monkeypatch, for isolation.
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)
    out = _test_cmd_helper(["list-modules"])
    assert "Available compilation modules:" in out


# TODO: test list-modules command with module name
