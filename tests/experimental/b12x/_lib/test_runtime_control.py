"""Independent capture/session scopes may not unfreeze one another."""
import pytest

from b12x._lib import runtime_control as rc


def test_outer_guard_survives_inner_scope_and_exception():
    with rc.kernel_resolution_guard("serving"):
        with pytest.raises(ValueError):
            with rc.kernel_resolution_guard("temporary capture"):
                raise ValueError("capture failed")
        with pytest.raises(rc.KernelResolutionFrozenError):
            rc.raise_if_kernel_resolution_frozen("uncached launch")
    rc.raise_if_kernel_resolution_frozen("new preparation")


def test_independent_guards_can_exit_out_of_order():
    first = rc.kernel_resolution_guard("first session")
    second = rc.kernel_resolution_guard("second session")
    first.__enter__()
    second.__enter__()
    try:
        first.__exit__(None, None, None)
        with pytest.raises(rc.KernelResolutionFrozenError):
            rc.raise_if_kernel_resolution_frozen("uncached launch")
    finally:
        second.__exit__(None, None, None)
    rc.raise_if_kernel_resolution_frozen("new preparation")
