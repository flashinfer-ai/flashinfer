import pytest
import torch

from flashinfer.utils import (
    supported_compute_capability,
    backend_requirement,
    BackendSupportedError,
)


def test_supported_compute_capability():
    """Test the supported_compute_capability decorator."""

    @supported_compute_capability([80, 86, 89, 90])
    def my_function(x, y):
        return x + y

    # Check attributes
    assert hasattr(my_function, "_supported_ccs"), "Missing _supported_ccs attribute"
    assert my_function._supported_ccs == {80, 86, 89, 90}, "Incorrect _supported_ccs"

    # Check method
    assert hasattr(my_function, "is_compute_capability_supported"), "Missing method"
    assert my_function.is_compute_capability_supported(80) is True
    assert my_function.is_compute_capability_supported(75) is False

    # Check function still works
    result = my_function(5, 10)
    assert result == 15, "Function doesn't work correctly"


def test_input_validation():
    """Test that the decorator validates input correctly."""

    # Test rejection of non-iterable
    with pytest.raises(TypeError, match="must be an iterable"):

        @supported_compute_capability(80)
        def func1():
            pass

    # Test rejection of string values
    with pytest.raises(TypeError, match="must be an integer"):

        @supported_compute_capability(["80", "86"])
        def func2():
            pass

    # Test rejection of float values
    with pytest.raises(TypeError, match="must be an integer"):

        @supported_compute_capability([80.0, 86])
        def func3():
            pass

    # Test rejection of bool values
    with pytest.raises(TypeError, match="got bool"):

        @supported_compute_capability([True, False])
        def func4():
            pass

    # Test acceptance of valid integers
    @supported_compute_capability([75, 80, 86, 89, 90, 100, 103, 110, 120])
    def func5():
        pass

    assert func5._supported_ccs == {75, 80, 86, 89, 90, 100, 103, 110, 120}


def test_backend_requirement_support_checks():
    """Test the backend_requirement decorator support checks."""

    @supported_compute_capability([80, 86, 89, 90])
    def _cudnn_check_my_kernel(x, backend):
        return True

    @supported_compute_capability([75, 80, 86, 89, 90])
    def _cutlass_check_my_kernel(x, backend):
        return True

    def _common_check(x, backend):
        # Common requirement: must be 2D
        return x.dim() == 2

    @backend_requirement(
        {"cudnn": _cudnn_check_my_kernel, "cutlass": _cutlass_check_my_kernel},
        common_check=_common_check,
    )
    def my_kernel(x, backend="cudnn"):
        return x * 2

    # Check methods added
    assert hasattr(my_kernel, "is_backend_supported"), "Missing is_backend_supported"
    assert hasattr(my_kernel, "is_compute_capability_supported"), (
        "Missing is_compute_capability_supported"
    )

    # Check backend support
    assert my_kernel.is_backend_supported("cutlass") is True
    assert my_kernel.is_backend_supported("cudnn") is True
    assert my_kernel.is_backend_supported("trtllm") is False

    # Check compute capability support
    assert my_kernel.is_backend_supported("cutlass", 80) is True
    assert my_kernel.is_backend_supported("cutlass", 75) is True  # cutlass supports 75
    assert (
        my_kernel.is_backend_supported("cudnn", 75) is False
    )  # cudnn does NOT support 75
    assert my_kernel.is_backend_supported("cudnn", 80) is True

    # Check cross-backend compute capability
    assert my_kernel.is_compute_capability_supported(75) is True  # cutlass has it
    assert my_kernel.is_compute_capability_supported(80) is True  # both have it
    assert my_kernel.is_compute_capability_supported(70) is False  # neither has it


def test_backend_requirement_wrapped_function():
    """Test the backend_requirement decorator's wrapped function."""
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA tests (no GPU available)")

    # Get actual device capability
    x = torch.randn(1, 1, device="cuda")
    major, minor = torch.cuda.get_device_capability(x.device)
    actual_capability = major * 10 + minor

    @supported_compute_capability([80, 86, 89, 90, actual_capability])
    def _cutlass_check(x, backend):
        return x.shape[0] > 0

    @supported_compute_capability([75, 80, 86, 89, 90, actual_capability])
    def _cudnn_check(x, backend):
        return x.shape[0] > 0

    @backend_requirement({"cutlass": _cutlass_check, "cudnn": _cudnn_check})
    def my_kernel(x, backend="cutlass"):
        return x * 2

    x = torch.randn(10, 10, device="cuda")

    # Test unsupported backend raises error
    # The error message may include capability info, so use a flexible pattern
    with pytest.raises(
        BackendSupportedError, match="does not support backend 'trtllm'"
    ):
        my_kernel(x, backend="trtllm")

    # Test supported backend works
    result = my_kernel(x, backend="cutlass")
    assert result.shape == x.shape


def test_common_check():
    """Test common_check parameter."""
    if not torch.cuda.is_available():
        pytest.skip("Skipping CUDA tests (no GPU available)")

    x = torch.randn(1, 1, device="cuda")
    major, minor = torch.cuda.get_device_capability(x.device)
    actual_capability = major * 10 + minor

    @supported_compute_capability([80, 86, 89, 90, actual_capability])
    def _cudnn_check_my_kernel(x, backend):
        return True

    @supported_compute_capability([75, 80, 86, 89, 90, actual_capability])
    def _cutlass_check_my_kernel(x, backend):
        return True

    def _common_check(x, backend):
        # Common requirement: must be 2D
        return x.dim() == 2

    @backend_requirement(
        {"cudnn": _cudnn_check_my_kernel, "cutlass": _cutlass_check_my_kernel},
        common_check=_common_check,
    )
    def my_kernel(x, backend="cudnn"):
        return x * 2

    x_2d = torch.randn(10, 10, device="cuda")
    x_3d = torch.randn(10, 10, 10, device="cuda")

    # 2D should work with skip_check
    result = my_kernel(x_2d, backend="cudnn", skip_check=True)
    assert result.shape == x_2d.shape

    # 3D should fail validation
    with pytest.raises(ValueError, match="Problem size is not supported"):
        my_kernel(x_3d, backend="cudnn")


def test_functools_wraps_preserves_metadata():
    """Test that backend_requirement preserves function metadata with functools.wraps."""

    @supported_compute_capability([80, 86, 89, 90])
    def _check(x, backend):
        return True

    @backend_requirement({"backend": _check})
    def my_documented_function(x, backend="backend"):
        """This is my function's docstring."""
        return x * 2

    # Verify that function metadata is preserved
    assert my_documented_function.__name__ == "my_documented_function"
    assert my_documented_function.__doc__ == "This is my function's docstring."

    # Verify that added methods still exist
    assert hasattr(my_documented_function, "is_backend_supported")
    assert hasattr(my_documented_function, "is_compute_capability_supported")
