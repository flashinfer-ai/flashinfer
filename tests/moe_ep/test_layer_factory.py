"""MoEEpLayer factory routing and lifecycle tests (stubbed fleet, no comm libs)."""

from __future__ import annotations

import pytest


def _split_fleet_params():
    from flashinfer.moe_ep import FleetParams

    return FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8)


def _nvep_fleet_params():
    from flashinfer.moe_ep import FleetParams

    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=4096,
    )


def _mega_fleet_params():
    import torch

    from flashinfer.moe_ep import FleetParams, MoEWeightPack

    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=64,
        token_hidden_size=128,
        weights=MoEWeightPack(
            w13=torch.zeros(1, 256, 128),
            w2=torch.zeros(1, 128, 128),
        ),
    )


def _mega_config(*, preprocess_weights: bool = False):
    from flashinfer.moe_ep import DeepGemmMegaMoeConfig, MegaConfig

    return MegaConfig(
        megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
        preprocess_weights=preprocess_weights,
    )


def test_factory_returns_split_for_string_backend():
    from flashinfer.moe_ep import BootstrapConfig, MoEEpLayer, MoEEpSplitLayer

    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=_split_fleet_params(),
        backend="nccl_ep",
    )
    assert isinstance(layer, MoEEpSplitLayer)


def test_factory_returns_split_for_nvep_config():
    from flashinfer.moe_ep import BootstrapConfig, MoEEpLayer, MoEEpSplitLayer, NvepConfig

    layer = MoEEpLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=_nvep_fleet_params(),
        backend=NvepConfig(),
    )
    assert isinstance(layer, MoEEpSplitLayer)


def test_factory_returns_mega_for_mega_config():
    from unittest import mock

    from flashinfer.moe_ep import BootstrapConfig, MoEEpLayer, MoEEpMegaLayer

    with mock.patch("flashinfer.moe_ep.core.validation.common.validate_mega_arch"):
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=_mega_fleet_params(),
            backend=_mega_config(),
        )
    assert isinstance(layer, MoEEpMegaLayer)


def test_factory_mega_ignores_fleet_knobs_warns():
    from unittest import mock

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetAlgoKnobNumChannelsPerRank,
        MoEEpLayer,
        MoEEpMegaLayer,
    )

    with mock.patch("flashinfer.moe_ep.core.validation.common.validate_mega_arch"):
        with pytest.warns(UserWarning, match="fleet_knobs are ignored"):
            layer = MoEEpLayer(
                bootstrap=BootstrapConfig(world_size=1, rank=0),
                fleet_params=_mega_fleet_params(),
                fleet_knobs=[FleetAlgoKnobNumChannelsPerRank(n=4)],
                backend=_mega_config(),
            )
    assert isinstance(layer, MoEEpMegaLayer)


def test_split_destroy_calls_fleet_destroy(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpSplitLayer,
        MoEEpTensors,
    )

    log = stubbed_fleet_registry
    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=_split_fleet_params(),
        backend="nccl_ep",
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = split.forward(t)
    split.destroy()
    assert log == [
        "fleet_init",
        "create_handle",
        "dispatch",
        "combine",
        "complete",
        "destroy",
    ]


def test_split_destroy_is_idempotent(stubbed_fleet_registry):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    from flashinfer.moe_ep import BootstrapConfig, MoEEpSplitLayer, MoEEpTensors

    log = stubbed_fleet_registry
    split = MoEEpSplitLayer(
        bootstrap=BootstrapConfig(world_size=1, rank=0),
        fleet_params=_split_fleet_params(),
        backend="nccl_ep",
    )
    t = MoEEpTensors(
        hidden_states=torch.zeros(4, 8, device="cuda"),
        topk_ids=torch.zeros(4, 2, dtype=torch.int64, device="cuda"),
        topk_weights=torch.ones(4, 2, device="cuda") * 0.5,
    )
    _ = split.forward(t)
    split.destroy()
    split.destroy()
    assert log.count("destroy") == 1
