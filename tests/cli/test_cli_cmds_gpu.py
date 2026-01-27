"""
Tests the module-status and list-modules commands

This is factored out from test_cli_cmds.py because these tests require a GPU.
"""

from .cli_cmd_helpers import _assert_output_contains_all

from dataclasses import dataclass
from datetime import datetime

import pytest
import torch

# Skip the entire module if no CUDA GPU is available
if not torch.cuda.is_available():
    pytest.skip("Requires CUDA GPU for CLI GPU tests.", allow_module_level=True)

from .cli_cmd_helpers import _test_cmd_helper


_MOCKED_CUDA_ARCH_LIST = "7.5 8.0 8.9 9.0a 10.0a"


@dataclass
class MockModuleStatus:
    """Mock module status for testing CLI commands.

    Simulates the structure returned by jit_spec_registry.get_all_statuses()
    with realistic field values for testing different module states.
    """

    name: str
    status: str
    is_compiled: bool
    library_path: str = ""
    sources: list = None
    needs_device_linking: bool = False
    created_at: datetime = None

    def __post_init__(self):
        if self.sources is None:
            self.sources = ["/path/to/source1.cu", "/path/to/source2.cu"]
        if self.created_at is None:
            self.created_at = datetime.now()


def _create_mock_statuses():
    """Create a list of mock module statuses for testing.

    Returns a mix of AOT compiled, JIT compiled, and not compiled modules
    to test different filter and display scenarios.
    """
    return [
        MockModuleStatus(
            name="batch_decode_fp16",
            status="AOT compiled",
            is_compiled=True,
            library_path="/path/to/batch_decode.so",
        ),
        MockModuleStatus(
            name="single_prefill_fp16",
            status="JIT compiled",
            is_compiled=True,
            library_path="/path/to/single_prefill.so",
        ),
        MockModuleStatus(name="norm_fp16", status="Not compiled", is_compiled=False),
    ]


@pytest.fixture
def mock_module_registry(monkeypatch):
    """Fixture that sets up common mocks for module registry tests.

    Mocks:
    - FLASHINFER_CUDA_ARCH_LIST environment variable
    - _ensure_modules_registered to return test statuses
    - jit_spec_registry.get_stats to return predictable statistics
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
    )

    def mock_get_stats():
        return {"total": 3, "compiled": 2, "not_compiled": 1}

    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_stats", mock_get_stats
    )


@pytest.mark.usefixtures("mock_module_registry")
def test_module_status_cmd_mocked():
    """
    Test that module-status command runs without error and sanity checks the output.
    """
    out = _test_cmd_helper(["module-status"])

    _assert_output_contains_all(
        out, "=== Summary ===", "Total modules:", "Compiled:", "Not compiled:"
    )


@pytest.mark.usefixtures("mock_module_registry")
def test_module_status_cmd_with_filters():
    """
    Test that module-status command works with different filter options.
    """
    # Test with compiled filter
    out = _test_cmd_helper(["module-status", "--filter", "compiled"])
    assert "=== Summary ===" in out

    # Test with not-compiled filter
    out = _test_cmd_helper(["module-status", "--filter", "not-compiled"])
    assert "=== Summary ===" in out

    # Test with all filter (default)
    out = _test_cmd_helper(["module-status", "--filter", "all"])
    assert "=== Summary ===" in out


@pytest.mark.usefixtures("mock_module_registry")
def test_module_status_cmd_detailed():
    """
    Test that module-status command works with detailed output.
    """
    out = _test_cmd_helper(["module-status", "--detailed"])

    # Detailed view should show module-specific information
    _assert_output_contains_all(
        out, "Module:", "Status:", "Sources:", "Created:", "=== Summary ==="
    )


@pytest.mark.usefixtures("mock_module_registry")
def test_list_modules_cmd_mocked():
    """
    Test that list-modules command runs without error and sanity checks the output.
    """
    out = _test_cmd_helper(["list-modules"])
    assert "Available compilation modules:" in out


@pytest.mark.usefixtures("mock_module_registry")
def test_list_modules_cmd_with_module_name(monkeypatch):
    """
    Test that list-modules command can inspect a specific module.
    """
    mock_status = MockModuleStatus(
        name="test_module_fp16",
        status="JIT compiled",
        is_compiled=True,
        library_path="/path/to/test_module.so",
    )

    # Mock the get_spec_status to return our test module
    def mock_get_spec_status(module_name):
        if module_name == "test_module_fp16":
            return mock_status
        return None

    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_spec_status", mock_get_spec_status
    )

    # Test with existing module name
    out = _test_cmd_helper(["list-modules", "test_module_fp16"])
    _assert_output_contains_all(
        out,
        "Module: test_module_fp16",
        "Status:",
        "Library Path:",
        "Source Files:",
        "Device Linking:",
    )

    # Test with non-existent module name
    out = _test_cmd_helper(["list-modules", "nonexistent_module"])
    assert "not found" in out
