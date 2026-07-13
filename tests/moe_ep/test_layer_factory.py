"""MoEEpLayer factory routing and lifecycle tests (stubbed fleet, no comm libs)."""

from __future__ import annotations

import pytest


def _split_fleet_params():
    from flashinfer.moe_ep import FleetParams

    return FleetParams(
        num_experts=2,
        max_tokens_per_rank=4,
        token_hidden_size=8,
    )


def _split_weights():
    from flashinfer.moe_ep import dummy_moe_weights

    return dummy_moe_weights(num_local_experts=2, hidden=8)


def _nvep_fleet_params():
    from flashinfer.moe_ep import FleetParams

    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=4096,
    )


def _nvep_weights():
    from flashinfer.moe_ep import dummy_moe_weights

    return dummy_moe_weights(num_local_experts=8, hidden=4096)


def _mega_fleet_params():
    from flashinfer.moe_ep import FleetParams

    return FleetParams(
        num_experts=1,
        max_tokens_per_rank=64,
        token_hidden_size=128,
    )


def _mega_weights():
    import torch

    from flashinfer.moe_ep import MoEWeightPack

    return MoEWeightPack(
        w13=torch.zeros(1, 256, 128),
        w2=torch.zeros(1, 128, 128),
    )


def _fake_deep_gemm_transformed(
    *,
    num_experts: int = 1,
    intermediate: int = 128,
    hidden: int = 128,
):
    import torch

    fc1_out = 2 * intermediate
    w1 = torch.zeros(num_experts, fc1_out, hidden // 2, dtype=torch.int8)
    sf1 = torch.zeros(num_experts, fc1_out, hidden // 32)
    w2 = torch.zeros(num_experts, hidden, intermediate // 2, dtype=torch.int8)
    sf2 = torch.zeros(num_experts, hidden, intermediate // 32)
    return ((w1, sf1), (w2, sf2))


def _mega_config(*, preprocess_weights: bool = False):
    from flashinfer.moe_ep import DeepGemmMegaMoeConfig, MegaConfig

    if preprocess_weights:
        return MegaConfig(
            megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
            preprocess_weights=True,
        )
    return MegaConfig(
        megakernel=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
        preprocess_weights=False,
        transformed_weights=_fake_deep_gemm_transformed(),
    )


def test_factory_returns_split_for_string_backend():
    import torch
    import torch.distributed

    if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("EP runtime wheels require CUDA 13+")

    from flashinfer.moe_ep import BootstrapConfig, MoEEpLayer, MoEEpSplitLayer

    try:
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=_split_fleet_params(),
            weights=_split_weights(),
            backend="nccl_ep",
        )
    except torch.distributed.DistNetworkError as e:
        pytest.skip(f"No usable network address in this Slurm environment: {e}")
    assert isinstance(layer, MoEEpSplitLayer)


def test_factory_returns_split_for_nvep_config():
    import torch
    import torch.distributed

    if torch.version.cuda and int(torch.version.cuda.split(".")[0]) < 13:
        pytest.skip("EP runtime wheels require CUDA 13+")

    from flashinfer.moe_ep import (
        BootstrapConfig,
        MoEEpLayer,
        MoEEpSplitLayer,
        NvepConfig,
    )

    try:
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, tcp_store=object()),
            fleet_params=_nvep_fleet_params(),
            weights=_nvep_weights(),
            backend=NvepConfig(),
        )
    except torch.distributed.DistNetworkError as e:
        pytest.skip(f"No usable network address in this Slurm environment: {e}")
    assert isinstance(layer, MoEEpSplitLayer)


def test_factory_returns_mega_for_mega_config(dist_not_initialized):
    from unittest import mock

    from flashinfer.moe_ep import BootstrapConfig, MoEEpLayer, MoEEpMegaLayer

    with mock.patch(
        "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend.validate_mega_arch"
    ):
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=_mega_fleet_params(),
            weights=_mega_weights(),
            backend=_mega_config(),
        )
    assert isinstance(layer, MoEEpMegaLayer)


def test_factory_mega_ignores_fleet_knobs_warns(dist_not_initialized):
    from unittest import mock

    from flashinfer.moe_ep import (
        BootstrapConfig,
        FleetAlgoKnobNumChannelsPerRank,
        MoEEpLayer,
        MoEEpMegaLayer,
    )

    with (
        mock.patch(
            "flashinfer.moe_ep.backends.mega.kernel.deep_gemm_mega.backend.validate_mega_arch"
        ),
        pytest.warns(UserWarning, match="fleet_knobs are ignored"),
    ):
        layer = MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
            fleet_params=_mega_fleet_params(),
            weights=_mega_weights(),
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
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=_split_fleet_params(),
        weights=_split_weights(),
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
        bootstrap=BootstrapConfig(world_size=1, rank=0, auto_bootstrap=False),
        fleet_params=_split_fleet_params(),
        weights=_split_weights(),
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


def test_split_layer_init_rejects_process_group_without_dist():
    from unittest import mock

    from flashinfer.moe_ep import BootstrapConfig, MoEEpConfigError, MoEEpSplitLayer

    pg = mock.MagicMock()
    with (
        mock.patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(MoEEpConfigError, match="process_group is set"),
    ):
        MoEEpSplitLayer(
            bootstrap=BootstrapConfig(
                world_size=1,
                rank=0,
                process_group=pg,
                auto_bootstrap=False,
            ),
            fleet_params=_split_fleet_params(),
            weights=_split_weights(),
            backend="nccl_ep",
        )


def test_factory_rejects_raw_mega_kernel_config():
    from flashinfer.moe_ep import BootstrapConfig, DeepGemmMegaMoeConfig, MoEEpLayer

    with pytest.raises(TypeError, match="MegaConfig"):
        MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=_split_fleet_params(),
            weights=_split_weights(),
            backend=DeepGemmMegaMoeConfig(intermediate_size=128, top_k=2),
        )


def test_factory_rejects_raw_split_kernel_config():
    from flashinfer.moe_ep import BootstrapConfig, IdentityConfig, MoEEpLayer

    with pytest.raises(TypeError, match="SplitConfig"):
        MoEEpLayer(
            bootstrap=BootstrapConfig(world_size=1, rank=0),
            fleet_params=_split_fleet_params(),
            weights=_split_weights(),
            backend=IdentityConfig(),
        )
