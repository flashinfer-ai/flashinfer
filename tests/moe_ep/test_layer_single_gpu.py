"""B6 — MoEEpSplitLayer forward sequencing test (stubbed Fleet)."""

from __future__ import annotations

import pytest


def test_forward_call_order(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
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

    from flashinfer.moe_ep import (
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


def _minimal_bf16_moe_config():
    from flashinfer.fused_moe.api import (
        BackendOptions,
        ExecutionConfig,
        ExpertConfig,
        MoEConfig,
        QuantConfig,
        QuantVariant,
        RoutingConfig,
        TrtllmBf16Config,
    )

    return MoEConfig(
        routing=RoutingConfig(num_experts=2, top_k=1),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=4,
            local_expert_offset=0,
            local_num_experts=1,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=4),
    )


def test_split_layer_requires_weights_for_non_identity_kernel():
    from flashinfer.moe_ep import (
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
            backend=SplitConfig(
                kernel=FusedMoeKernelConfig(moe_config=_minimal_bf16_moe_config()),
            ),
        )


def test_split_layer_preprocess_requires_weights():
    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        IdentityConfig,
        MoEEpSplitLayer,
        SplitConfig,
    )

    with pytest.raises(ValueError, match="FleetParams.weights"):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=2, max_tokens_per_rank=4, token_hidden_size=8
            ),
            backend=SplitConfig(
                kernel=IdentityConfig(require_weights=True),
            ),
        )


def test_split_layer_accepts_fused_moe_kernel_with_weights(stubbed_fleet_registry):
    """FusedMoe kernel initializes when weights + moe_config are supplied."""
    from unittest import mock

    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetParams,
        FusedMoeKernelConfig,
        MoEEpSplitLayer,
        MoEWeightPack,
        SplitConfig,
    )

    with mock.patch(
        "flashinfer.moe_ep.modes.split_layer.validate_arch_for_backend"
    ):
        split = MoEEpSplitLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=FleetParams(
                num_experts=2,
                max_tokens_per_rank=4,
                token_hidden_size=8,
                weights=MoEWeightPack(
                    w13=torch.zeros(1, 4, 8, device="cuda"),
                    w2=torch.zeros(1, 8, 4, device="cuda"),
                ),
            ),
            backend=SplitConfig(
                kernel=FusedMoeKernelConfig(moe_config=_minimal_bf16_moe_config()),
            ),
        )
    assert split._kernel.kernel_name() == "fused_moe"


def test_split_layer_rejects_nixl_oversized_tokens_at_init():
    from flashinfer.moe_ep import (
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
    from flashinfer.moe_ep import (
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

    from flashinfer.moe_ep import (
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
