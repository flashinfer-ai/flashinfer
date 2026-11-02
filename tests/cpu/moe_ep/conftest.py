"""Host-side Fleet fixtures for the relocated CPU tests."""

import pytest


@pytest.fixture
def stubbed_fleet_registry():
    """Inject a stub Fleet class that records dispatch/combine/destroy calls."""
    from unittest import mock

    from flashinfer.moe_ep.core.comm.fleet import _FLEET_REGISTRY

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

    saved_nccl = _FLEET_REGISTRY.get("nccl_ep")
    saved_nixl = _FLEET_REGISTRY.get("nixl_ep")
    _FLEET_REGISTRY["nccl_ep"] = _StubFleet
    _FLEET_REGISTRY["nixl_ep"] = _StubFleet
    with mock.patch("flashinfer.moe_ep.modes.split_layer.validate_arch_for_backend"):
        yield log
    if saved_nccl is not None:
        _FLEET_REGISTRY["nccl_ep"] = saved_nccl
    else:
        _FLEET_REGISTRY.pop("nccl_ep", None)
    if saved_nixl is not None:
        _FLEET_REGISTRY["nixl_ep"] = saved_nixl
    else:
        _FLEET_REGISTRY.pop("nixl_ep", None)
