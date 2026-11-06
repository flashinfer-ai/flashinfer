"""Host-only contracts for backend and capability decorators."""

import pytest

import flashinfer.api_logging as api_logging
from flashinfer.utils import (
    supported_compute_capability,
    backend_requirement,
    experimental_backend,
)


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_has_backend():
    """Test the has_backend method."""

    @backend_requirement({"cudnn": lambda x: True, "cutlass": lambda x: True})
    def my_kernel(x, backend="cudnn"):
        return x * 2

    assert my_kernel.has_backend("cudnn") is True
    assert my_kernel.has_backend("cutlass") is True
    assert my_kernel.has_backend("random_backend") is False


@pytest.mark.cpu_only
def test_has_backend_choices():
    """Test the has_backend_choices method."""

    @backend_requirement({"cudnn": lambda x: True, "cutlass": lambda x: True})
    def my_kernel(x, backend="cudnn"):
        return x * 2

    @backend_requirement({})
    def my_kernel_no_backend(x):
        return x * 2

    assert my_kernel.has_backend_choices() is True
    assert my_kernel_no_backend.has_backend_choices() is False


@pytest.mark.cpu_only
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


@pytest.mark.cpu_only
def test_experimental_backend_excluded_from_auto_unless_allowed(monkeypatch):
    """@experimental_backend checkers stay out of backend="auto" without the opt-in."""
    monkeypatch.delenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", raising=False)
    # The once-per-(api, backend) registry is module state that outlives a test.
    # pytest.warns below needs the pair unseen, so start from an empty set rather
    # than depending on this module never being collected twice in one process.
    monkeypatch.setattr(api_logging, "_WARNED_EXPERIMENTAL_BACKENDS", set())

    def _check_stable(x, backend="auto"):
        return True

    @experimental_backend
    def _check_risky(x, backend="auto"):
        return True

    # No tensor inputs here, so capability is None; accept it explicitly.
    for checker in (_check_stable, _check_risky):
        checker.is_compute_capability_supported = lambda cc: True

    def _prefer_risky(backends, *args, **kwargs):
        return [b for b in ("risky", "stable") if b in backends]

    @backend_requirement(
        {"stable": _check_stable, "risky": _check_risky}, heuristic_func=_prefer_risky
    )
    def my_kernel(x, backend="auto"):
        return my_kernel.suitable_auto_backends[0] if backend == "auto" else backend

    assert my_kernel.experimental_backends == frozenset({"risky"})
    assert my_kernel(1) == "stable"  # gate off: experimental skipped

    # Explicit selection is always allowed; the first use of the pair warns once.
    with pytest.warns(UserWarning, match="risky"):
        assert my_kernel(1, backend="risky") == "risky"

    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    assert my_kernel(1) == "risky"  # gate on: heuristic may pick it
