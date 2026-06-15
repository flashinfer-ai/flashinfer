"""B6 — MoEEpSplitLayer forward sequencing test (stubbed Fleet)."""

from __future__ import annotations

import pytest


@pytest.fixture
def stubbed_fleet_registry():
    """Inject a stub Fleet class that records the dispatch/combine/complete
    sequence into a list."""
    from flashinfer.moe_ep_v2.fleet import _BACKEND_REGISTRY

    log: list[str] = []

    class _StubHandle:
        def dispatch(self, params):
            log.append("dispatch")
            from flashinfer.moe_ep_v2 import DispatchOutput

            return DispatchOutput(
                expert_tensors=params.x[0], num_tokens=params.x[0].size(0)
            )

        def combine(self, params):
            log.append("combine")
            from flashinfer.moe_ep_v2 import CombineOutput

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

    saved = _BACKEND_REGISTRY.get("nccl_ep")
    _BACKEND_REGISTRY["nccl_ep"] = _StubFleet
    yield log
    if saved is not None:
        _BACKEND_REGISTRY["nccl_ep"] = saved


def test_forward_call_order(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        IdentityConfig,
        MoEEpSplitLayer,
        MoEEpTensors,
        NCCLEPConfig,
        SplitConfig,
    )

    log = stubbed_fleet_registry
    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
        ),
        backend=SplitConfig(comm=NCCLEPConfig(), kernel=IdentityConfig()),
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = split.forward(t)
    assert log == ["fleet_init", "create_handle", "dispatch", "combine", "complete"]


def test_factory_routes_split_backend(stubbed_fleet_registry):
    """MoEEpLayer factory returns MoEEpSplitLayer for SplitConfig / strings."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        MoEEpLayer,
        MoEEpSplitLayer,
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
    assert isinstance(layer, MoEEpSplitLayer)
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = layer.forward(t)
    assert "fleet_init" in log


def test_split_layer_requires_weights_for_non_identity_kernel():
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpSplitLayer,
        SplitConfig,
    )

    with pytest.raises(ValueError, match="FleetParams.weights"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
            ),
            backend=SplitConfig(kernel=FusedMoeKernelConfig()),
        )


def test_split_layer_preprocess_requires_weights():
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        IdentityConfig,
        MoEEpSplitLayer,
        SplitConfig,
    )

    with pytest.raises(ValueError, match="require_weights"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
            ),
            backend=SplitConfig(
                kernel=IdentityConfig(require_weights=True),
            ),
        )


def test_split_layer_rejects_fused_moe_kernel_at_init():
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpSplitLayer,
        MoEWeightPack,
        SplitConfig,
    )
    import torch

    with pytest.raises(NotImplementedError, match="FusedMoeKernelConfig"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=2,
                max_tokens_per_rank=4,
                token_hidden_size=8,
                weights=MoEWeightPack(
                    w13=torch.zeros(1, 4, 8),
                    w2=torch.zeros(1, 8, 4),
                ),
            ),
            backend=SplitConfig(kernel=FusedMoeKernelConfig()),
        )


def test_split_layer_rejects_nixl_oversized_tokens_at_init():
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        MoEEpConfigError,
        MoEEpSplitLayer,
        NvepConfig,
    )

    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=8, max_tokens_per_rank=2048, token_hidden_size=4096
            ),
            backend=NvepConfig(),
        )


def test_split_layer_rejects_nixl_topology_capacity_mismatch_at_init():
    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetAlgoKnobTopologyCapacity,
        FleetParams,
        MoEEpConfigError,
        MoEEpSplitLayer,
        NvepConfig,
    )

    with pytest.raises(MoEEpConfigError, match="topology capacity"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=2, rank=0),
            fleet_params=FleetParams(
                num_experts=8, max_tokens_per_rank=128, token_hidden_size=4096
            ),
            fleet_knobs=[FleetAlgoKnobTopologyCapacity(n=3)],
            backend=NvepConfig(),
        )


def test_split_layer_forward_rejects_token_overflow(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep_v2 import (
        BootstrapConfig,
        FleetParams,
        MoEEpConfigError,
        MoEEpSplitLayer,
        MoEEpTensors,
    )

    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=FleetParams(
            num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
        ),
        backend="nccl_ep",
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(5, 8, device="cuda"),
        topk_ids=torch.zeros(5, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(5, 2, device="cuda") * 0.5,
    )
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        split.forward(t)
