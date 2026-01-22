"""
Tests the module-status and list-modules commands

This is factored out from test_cli_cmds.py because these tests require a GPU.
"""

from cli_cmd_helpers import _test_cmd_helper
from datetime import datetime
from dataclasses import dataclass


_MOCKED_CUDA_ARCH_LIST = "7.5 8.0 8.9 9.0a 10.0a"


# Create mock module statuses for testing
@dataclass
class MockModuleStatus:
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
    """Create a list of mock module statuses for testing"""
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


def test_module_status_cmd_mocked(monkeypatch):
    """
    Test that module-status command runs without error and sanity checks the output

    Mocks the module registration to avoid GPU architecture issues.
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    # Mock _ensure_modules_registered to return our test statuses
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
    )

    # Mock get_stats to return predictable statistics
    def mock_get_stats():
        return {
            "total": 3,
            "compiled": 2,
            "not_compiled": 1,
        }

    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_stats", mock_get_stats
    )

    out = _test_cmd_helper(["module-status"])
    assert "=== Summary ===" in out
    assert "Total modules:" in out
    assert "Compiled:" in out
    assert "Not compiled:" in out


def test_module_status_cmd_with_filters(monkeypatch):
    """
    Test that module-status command works with different filter options
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    # Mock _ensure_modules_registered
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
    )

    # Mock get_stats
    def mock_get_stats():
        return {"total": 3, "compiled": 2, "not_compiled": 1}

    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_stats", mock_get_stats
    )

    # Test with compiled filter
    out = _test_cmd_helper(["module-status", "--filter", "compiled"])
    assert "=== Summary ===" in out

    # Test with not-compiled filter
    out = _test_cmd_helper(["module-status", "--filter", "not-compiled"])
    assert "=== Summary ===" in out

    # Test with all filter (default)
    out = _test_cmd_helper(["module-status", "--filter", "all"])
    assert "=== Summary ===" in out


def test_module_status_cmd_detailed(monkeypatch):
    """
    Test that module-status command works with detailed output
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    # Mock _ensure_modules_registered
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
    )

    # Mock get_stats
    def mock_get_stats():
        return {"total": 3, "compiled": 2, "not_compiled": 1}

    monkeypatch.setattr(
        "flashinfer.__main__.jit_spec_registry.get_stats", mock_get_stats
    )

    out = _test_cmd_helper(["module-status", "--detailed"])

    # Detailed view should show module-specific information
    assert "Module:" in out
    assert "Status:" in out
    assert "Sources:" in out
    assert "Created:" in out
    assert "=== Summary ===" in out


def test_list_modules_cmd_mocked(monkeypatch):
    """
    Test that list-modules command runs without error and sanity checks the output

    Mocks the module registration to avoid GPU architecture issues.
    """
    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    # Mock _ensure_modules_registered
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
    )

    out = _test_cmd_helper(["list-modules"])
    assert "Available compilation modules:" in out


def test_list_modules_cmd_with_module_name(monkeypatch):
    """
    Test that list-modules command can inspect a specific module
    """
    mock_status = MockModuleStatus(
        name="test_module_fp16",
        status="JIT compiled",
        is_compiled=True,
        library_path="/path/to/test_module.so",
    )

    monkeypatch.setenv("FLASHINFER_CUDA_ARCH_LIST", _MOCKED_CUDA_ARCH_LIST)

    # Mock _ensure_modules_registered to return our statuses
    monkeypatch.setattr(
        "flashinfer.__main__._ensure_modules_registered", _create_mock_statuses
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
    assert "Module: test_module_fp16" in out
    assert "Status:" in out
    assert "Library Path:" in out
    assert "Source Files:" in out
    assert "Device Linking:" in out

    # Test with non-existent module name
    out = _test_cmd_helper(["list-modules", "nonexistent_module"])
    assert "not found" in out
