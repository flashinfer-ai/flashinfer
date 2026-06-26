"""Shared fixtures for moe_ep multirank tests."""

from __future__ import annotations

import os
import sys

import pytest
import torch

# When pytest is launched under torchrun, avoid shadowing the native
# ``nccl_ep`` / ``nixl_ep`` extension modules with the mock test subpackages
# under ``tests/moe_ep/{nccl,nixl}_ep/``.
_MOE_EP_TEST_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [
    p
    for p in sys.path
    if os.path.abspath(p or os.getcwd()) != _MOE_EP_TEST_ROOT
]


def pytest_addoption(parser):
    parser.addoption(
        "--backend",
        action="store",
        default=None,
        help="moe_ep backend to exercise (nccl_ep / nixl_ep / both); "
        "consumed by tests/moe_ep/test_moe_ep_layer_multirank.py",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "nvep: requires BUILD_NVEP=1 install")
    config.addinivalue_line("markers", "gpu_2: requires >=2 GPUs")
    config.addinivalue_line("markers", "gpu_4: requires >=4 GPUs")
    config.addinivalue_line("markers", "gpu_8: requires >=8 GPUs")
    config.addinivalue_line("markers", "arch_blackwell: requires sm_100 or sm_103")


def pytest_collection_modifyitems(config, items):
    """Skip moe_ep tests on hosts that lack the requisite env / GPUs / arch."""
    nvep_built = False
    try:
        from importlib import import_module

        nvep_built = bool(import_module("flashinfer.moe_ep").available_backends())
    except ImportError:
        pass

    ngpu = torch.cuda.device_count() if torch.cuda.is_available() else 0
    cc = (
        torch.cuda.get_device_capability(0)
        if torch.cuda.is_available() and ngpu > 0
        else (0, 0)
    )

    for item in items:
        if "nvep" in item.keywords and not nvep_built:
            item.add_marker(
                pytest.mark.skip(
                    reason="needs BUILD_NCCL_EP=1 / BUILD_NIXL_EP=1 install"
                )
            )
        for mk, req in (("gpu_2", 2), ("gpu_4", 4), ("gpu_8", 8)):
            if mk in item.keywords and ngpu < req:
                item.add_marker(pytest.mark.skip(reason=f"needs >= {req} GPUs"))
        if "arch_blackwell" in item.keywords and cc < (10, 0):
            item.add_marker(pytest.mark.skip(reason="needs sm_100+"))


@pytest.fixture
def stubbed_fleet_registry():
    """Inject a stub Fleet class that records dispatch/combine/destroy calls."""
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
