"""Host-only contracts for backend and capability decorators."""

import inspect

import pytest
import torch

import flashinfer.api_logging as api_logging
import flashinfer.utils as utils
from flashinfer.utils import (
    BackendSupportedError,
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
def test_validate_call_uses_execution_validation_without_executing():
    checks = []
    executions = []

    def check(x, backend="native", *, scale=3):
        checks.append((x, backend, scale))
        return x > 0

    check.is_compute_capability_supported = lambda cc: cc is None

    @backend_requirement({"native": check})
    def kernel(x, backend="native", *, scale=3):
        executions.append(x)
        return x * scale

    assert inspect.signature(kernel.validate_call) == inspect.signature(kernel)
    assert kernel.validate_call(2) is None
    assert checks == [(2, "native", 3)]
    assert executions == []
    assert kernel.validate_call(2, "native", scale=4) is None
    assert checks[-1] == (2, "native", 4)

    for skip_check in (False, True):
        with pytest.raises(ValueError, match="Problem size is not supported"):
            kernel.validate_call(0, skip_check=skip_check)
    with pytest.raises(BackendSupportedError, match="does not support backend"):
        kernel.validate_call(2, backend="missing")
    with pytest.raises(TypeError):
        kernel.validate_call()
    with pytest.raises(TypeError):
        kernel.validate_call(2, x=3)
    assert executions == []

    with pytest.raises(ValueError, match="Problem size is not supported"):
        kernel(0)
    assert executions == []
    assert kernel(2) == 6
    before = len(checks)
    assert kernel(0, skip_check=True) == 0
    assert len(checks) == before


@pytest.mark.cpu_only
@pytest.mark.parametrize("validate_on_call", [False, True])
def test_complete_call_validator_owns_cpu_routing(monkeypatch, validate_on_call):
    def unexpected(*args, **kwargs):
        pytest.fail("complete call validation must not use generic CUDA/auto checks")

    monkeypatch.setattr(utils, "get_compute_capability", unexpected)
    checks = []
    executions = []

    # Metadata-only checkers need not accept the operation's call signature.
    def metadata_only():
        unexpected()

    metadata_only.is_compute_capability_supported = lambda cc: cc >= 100

    def call_check(input, backend="auto", *, alignment=32):
        checks.append((input, backend, alignment))
        if input.shape[-1] != alignment:
            raise ValueError("invalid input width")

    def original(input, backend="auto", *, alignment=32):
        executions.append(input)
        return "executed"

    kernel = backend_requirement(
        {"native": metadata_only},
        common_check=unexpected,
        heuristic_func=unexpected,
        call_check=call_check,
        validate_on_call=validate_on_call,
    )(original)
    assert (kernel is original) is (not validate_on_call)
    assert inspect.signature(kernel.validate_call) == inspect.signature(original)
    assert kernel.is_backend_supported("native", 130) is True
    assert kernel.is_backend_supported("native", 90) is False

    valid = torch.empty((2, 32))
    invalid = torch.empty((2, 16))
    assert kernel.validate_call(valid) is None
    assert checks == [(valid, "auto", 32)]
    assert kernel.validate_call(invalid, "native", alignment=16) is None
    assert checks[-1] == (invalid, "native", 16)
    with pytest.raises(ValueError, match="invalid input width"):
        kernel.validate_call(invalid)
    if validate_on_call:
        for skip_check in (False, True):
            with pytest.raises(ValueError, match="invalid input width"):
                kernel.validate_call(invalid, skip_check=skip_check)
    else:
        with pytest.raises(TypeError, match="skip_check"):
            kernel.validate_call(valid, skip_check=True)
    with pytest.raises(TypeError):
        kernel.validate_call(valid, unexpected_argument=True)
    assert executions == []

    before = len(checks)
    assert kernel(valid) == "executed"
    assert len(checks) == before + int(validate_on_call)
    if validate_on_call:
        with pytest.raises(ValueError, match="invalid input width"):
            kernel(invalid)
    else:
        before = len(checks)
        assert kernel(invalid) == "executed"
        assert len(checks) == before
        with pytest.raises(TypeError, match="skip_check"):
            kernel(valid, skip_check=True)


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
        executions.append(x)
        return my_kernel.suitable_auto_backends[0] if backend == "auto" else backend

    executions = []
    assert my_kernel.experimental_backends == frozenset({"risky"})
    assert my_kernel.validate_call(1) is None
    assert executions == []
    assert my_kernel.suitable_auto_backends == ["stable"]
    assert my_kernel(1) == "stable"  # gate off: experimental skipped
    assert my_kernel(1, backend="auto", skip_check=True) == "stable"

    # Explicit selection is always allowed; the first use of the pair warns once.
    with pytest.warns(UserWarning, match="risky"):
        assert my_kernel(1, backend="risky") == "risky"

    monkeypatch.setenv("FLASHINFER_ALLOW_EXPERIMENTAL_AUTO_BACKENDS", "1")
    assert my_kernel(1) == "risky"  # gate on: heuristic may pick it

    before = len(executions)
    assert my_kernel.validate_call(1) is None
    assert len(executions) == before
    assert my_kernel(1, backend="auto", skip_check=True) == "risky"


@pytest.mark.cpu_only
def test_bypass_skips_backend_selection_and_checks():
    """A call the ``bypass`` predicate claims runs the function directly: no requirement check, no heuristic, no
    capability lookup. Every other call keeps the full selection."""
    calls = []

    def _check_a(x, backend="auto", own_mode=False):
        calls.append("check_a")
        return True

    def _common(x, backend="auto", own_mode=False):
        calls.append("common")
        return True

    _check_a.is_compute_capability_supported = lambda cc: True

    def _heuristic(backends, *args, **kwargs):
        calls.append("heuristic")
        return backends

    @backend_requirement(
        {"a": _check_a},
        common_check=_common,
        heuristic_func=_heuristic,
        bypass=lambda *args, **kwargs: kwargs.get("own_mode", False),
    )
    def my_kernel(x, backend="auto", *, own_mode=False):
        return "own" if own_mode else my_kernel.suitable_auto_backends[0]

    assert my_kernel(1, own_mode=True) == "own"
    assert calls == []
    # also with an explicit backend and skip_check: the predicate decides before either is looked at
    assert my_kernel(1, backend="b", own_mode=True, skip_check=True) == "own"
    assert calls == []

    assert my_kernel(1) == "a"
    assert calls == ["common", "check_a", "heuristic"]
