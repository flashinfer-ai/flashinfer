"""Shared fixtures for moe_ep multirank tests."""

from __future__ import annotations

import os
import sys

import pytest

# When pytest is launched under torchrun, avoid shadowing the native
# ``nccl_ep`` / ``nixl_ep`` extension modules with the mock test subpackages
# under ``tests/moe_ep/{nccl,nixl}_ep/``.
_MOE_EP_TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [
    p for p in sys.path if os.path.abspath(p or os.getcwd()) != _MOE_EP_TEST_ROOT
]


@pytest.fixture
def dist_not_initialized():
    """Hide a prior torch.distributed init from bootstrap validation.

    Earlier unit tests may leave a world_size=1 process group active via
    ``auto_bootstrap=True`` layer construction; init tests that assert on
    config fields other than bootstrap sizing need an isolated view of dist.
    """
    from unittest import mock

    with mock.patch("torch.distributed.is_initialized", return_value=False):
        yield


# NOTE: ``pytest_addoption`` (--backend), ``pytest_configure`` (nvep/gpu_*/
# arch_blackwell markers), and ``pytest_collection_modifyitems`` (env/GPU/arch
# auto-skips) are intentionally defined ONLY in the root ``tests/conftest.py``.
# Re-declaring them here triggers a duplicate-option error
# ("option names {'--backend'} already added") because pytest loads both the
# parent and child conftests. Keep the shared fixtures below in this file.


@pytest.fixture
def stubbed_fleet_registry():
    """Inject a stub Fleet class that records dispatch/combine/destroy calls."""
    from unittest import mock

    from flashinfer.moe_ep.core.comm.fleet import _BACKEND_REGISTRY

    log: list[str] = []

    class _StubHandle:
        def dispatch(self, params):
            log.append("dispatch")
            from flashinfer.moe_ep import DispatchOutput

            return DispatchOutput(
                expert_tensors=params.x[0], num_tokens=params.x[0].size(0)
            )

        def combine(self, params):
            log.append("combine")
            from flashinfer.moe_ep import CombineOutput

            return CombineOutput(x=params.x[0] if params.out is None else params.out)

        def complete(self):
            log.append("complete")

        def destroy(self):
            pass

    class _StubFleet:
        def __init__(self, bootstrap, params, algo_knobs):
            log.append("fleet_init")
            self.params = params

        def create_handle(self, params, algo_knobs=()):
            log.append("create_handle")
            return _StubHandle()

        def update_topology(self, bootstrap, algo_knobs=()):
            pass

        def destroy(self):
            log.append("destroy")

    saved_nccl = _BACKEND_REGISTRY.get("nccl_ep")
    saved_nixl = _BACKEND_REGISTRY.get("nixl_ep")
    _BACKEND_REGISTRY["nccl_ep"] = _StubFleet
    _BACKEND_REGISTRY["nixl_ep"] = _StubFleet
    with mock.patch("flashinfer.moe_ep.modes.split_layer.validate_arch_for_backend"):
        yield log
    if saved_nccl is not None:
        _BACKEND_REGISTRY["nccl_ep"] = saved_nccl
    else:
        _BACKEND_REGISTRY.pop("nccl_ep", None)
    if saved_nixl is not None:
        _BACKEND_REGISTRY["nixl_ep"] = saved_nixl
    else:
        _BACKEND_REGISTRY.pop("nixl_ep", None)


def pytest_sessionfinish(session, exitstatus):
    """Tear down torch.distributed once per torchrun session.

    NCCL cannot be re-initialized after destroy_process_group() within the
    same torchrun worker processes, so we must not destroy the process group
    between individual tests (e.g. mega multirank runs two gpu_4 tests).
    """
    import torch.distributed as dist

    if dist.is_initialized():
        dist.destroy_process_group()
