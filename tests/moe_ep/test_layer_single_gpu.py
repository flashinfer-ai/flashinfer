"""B6 — MoEEpLayer forward sequencing test (stubbed Fleet)."""

from __future__ import annotations

import pytest


@pytest.fixture
def stubbed_fleet_registry():
    """Inject a stub Fleet class that records the dispatch/combine/complete
    sequence into a list."""
    from flashinfer.moe_ep.fleet import _BACKEND_REGISTRY

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

    saved = _BACKEND_REGISTRY.get("nccl_ep")
    _BACKEND_REGISTRY["nccl_ep"] = _StubFleet
    yield log
    if saved is not None:
        _BACKEND_REGISTRY["nccl_ep"] = saved


def test_forward_call_order(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
    )

    log = stubbed_fleet_registry
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
        ),
        backend="nccl_ep",
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = layer.forward(t)
    # Fleet only constructed on first forward.
    assert log == ["fleet_init", "create_handle", "dispatch", "combine", "complete"]


def test_backend_config_object_routing(stubbed_fleet_registry):
    """MoEEpLayer accepts NcclEpConfig() / NvepConfig() instead of strings."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        MoEEpLayer,
        MoEEpTensors,
        NcclEpConfig,
    )

    log = stubbed_fleet_registry
    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
        ),
        backend=NcclEpConfig(),
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = layer.forward(t)
    # The config-routed path still ran the same sequence on the stub fleet.
    assert "fleet_init" in log
