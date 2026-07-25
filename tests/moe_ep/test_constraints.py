"""B9 — constraint validator unit tests."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep import (
    BootstrapConfig,
    dummy_moe_weights,
    FleetAlgoKnobQuantization,
    FleetParams,
    MoEEpConfigError,
    QuantType,
    ensure_bootstrap_dist_validated,
    validate_bootstrap_process_group_ready,
    validate_bootstrap_world_size,
    validate_fleet_params,
    validate_fleet_weights,
    validate_mega_fleet_params,
    validate_mega_forward_inputs,
    validate_split_forward_inputs,
)


def _split(
    *,
    num_experts: int = 8,
    world_size: int = 4,
    max_tokens_per_rank: int = 128,
    token_hidden_size: int = 4096,
) -> FleetParams:
    return FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        token_hidden_size=token_hidden_size,
    )


def test_num_experts_must_be_divisible_by_world_size():
    p = _split(num_experts=7, world_size=2, max_tokens_per_rank=64)
    pack = dummy_moe_weights(num_local_experts=1, hidden=p.token_hidden_size)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_weights(pack, p, world_size=2)
    # nixl_ep also checks divisibility against topology capacity (defaults to world_size).
    p2 = _split(num_experts=7, world_size=2, max_tokens_per_rank=64)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_params(p2, backend="nixl_ep", world_size=2)


def test_fleet_weights_local_expert_count():
    p = _split(num_experts=8, world_size=4, max_tokens_per_rank=64)
    pack = dummy_moe_weights(num_local_experts=1, hidden=p.token_hidden_size)
    with pytest.raises(MoEEpConfigError, match="num_experts // world_size"):
        validate_fleet_weights(pack, p, world_size=4)


def test_nixl_ep_max_tokens_cap():
    p = _split(max_tokens_per_rank=2048)
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4)
    # nccl_ep has no equivalent cap.
    validate_fleet_params(p, backend="nccl_ep", world_size=4)


def test_nixl_ep_hidden_size_set():
    p = _split(token_hidden_size=1024)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4)
    # nccl_ep allows arbitrary hidden sizes.
    validate_fleet_params(p, backend="nccl_ep", world_size=4)


def test_nixl_ep_supported_hidden_sizes_pass():
    for h in (2048, 4096, 7168, 8192):
        p = _split(token_hidden_size=h)
        validate_fleet_params(p, backend="nixl_ep", world_size=4)


def test_mega_fleet_params_rejects_indivisible_experts():
    p = _split(num_experts=7, world_size=2, max_tokens_per_rank=64)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_mega_fleet_params(p, world_size=2, intermediate_size=2048, top_k=4)


def test_mega_fleet_params_requires_128_aligned_sizes():
    p = _split(max_tokens_per_rank=64, token_hidden_size=1000)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_mega_fleet_params(p, world_size=4, intermediate_size=2048, top_k=4)
    p2 = _split(max_tokens_per_rank=64)
    with pytest.raises(MoEEpConfigError, match="intermediate_size"):
        validate_mega_fleet_params(p2, world_size=4, intermediate_size=1000, top_k=4)


def test_mega_fleet_params_happy_path():
    p = _split(max_tokens_per_rank=64)
    validate_mega_fleet_params(p, world_size=4, intermediate_size=2048, top_k=4)


def test_bootstrap_world_size_must_match_dist_when_initialized():
    from unittest import mock

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    mock_pg = mock.MagicMock()
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_comm_group",
            return_value=mock_pg,
        ),
        mock.patch("torch.distributed.get_world_size", return_value=8),
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_ep_rank_world",
            return_value=(0, 8),
        ),
        pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"),
    ):
        validate_bootstrap_world_size(bootstrap)


def test_bootstrap_world_size_skipped_when_dist_not_initialized():
    from unittest import mock

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        validate_bootstrap_world_size(bootstrap)


def test_bootstrap_process_group_requires_dist_at_init():
    from unittest import mock

    pg = mock.MagicMock()
    bootstrap = BootstrapConfig(world_size=4, rank=0, process_group=pg)
    with (
        mock.patch("torch.distributed.is_initialized", return_value=False),
        pytest.raises(MoEEpConfigError, match="process_group is set"),
    ):
        validate_bootstrap_process_group_ready(bootstrap)


def test_ensure_bootstrap_dist_validated_deferred_world_size_check():
    from unittest import mock

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        ensure_bootstrap_dist_validated(bootstrap)

    mock_pg = mock.MagicMock()
    with (
        mock.patch("torch.distributed.is_initialized", return_value=True),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_comm_group",
            return_value=mock_pg,
        ),
        mock.patch("torch.distributed.get_world_size", return_value=8),
        mock.patch("torch.distributed.get_rank", return_value=0),
        mock.patch(
            "flashinfer.moe_ep.core.bootstrap_utils.bootstrap_ep_rank_world",
            return_value=(0, 8),
        ),
        pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"),
    ):
        ensure_bootstrap_dist_validated(bootstrap)


def test_split_forward_inputs_rejects_token_overflow():
    import torch

    p = _split(max_tokens_per_rank=4, token_hidden_size=8, world_size=1, num_experts=8)
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        validate_split_forward_inputs(
            torch.zeros(5, 8),
            torch.zeros(5, 2, dtype=torch.int64),
            torch.zeros(5, 2),
            p,
        )


def test_split_forward_inputs_rejects_hidden_mismatch():
    import torch

    p = _split(max_tokens_per_rank=4, token_hidden_size=8, world_size=1, num_experts=8)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_split_forward_inputs(
            torch.zeros(4, 16),
            torch.zeros(4, 2, dtype=torch.int64),
            torch.zeros(4, 2),
            p,
        )


def test_nixl_ep_topology_capacity_requires_expert_divisibility():
    p = _split(world_size=2, num_experts=8)
    with pytest.raises(MoEEpConfigError, match="topology capacity"):
        validate_fleet_params(
            p,
            backend="nixl_ep",
            world_size=2,
            topology_capacity=3,
        )
    validate_fleet_params(
        p,
        backend="nixl_ep",
        world_size=2,
        topology_capacity=4,
    )


def test_nixl_ep_topology_capacity_defaults_to_world_size():
    p = _split()
    validate_fleet_params(p, backend="nixl_ep", world_size=4)


def test_mega_forward_inputs_rejects_hidden_mismatch():
    import torch

    p = _split(max_tokens_per_rank=4, token_hidden_size=8, world_size=1, num_experts=8)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_mega_forward_inputs(
            torch.zeros(4, 16),
            torch.zeros(4, 2, dtype=torch.int64),
            torch.zeros(4, 2),
            p,
            top_k=2,
            quantize_input=True,
        )


def test_mega_forward_inputs_happy_path():
    import torch

    p = _split(max_tokens_per_rank=4, token_hidden_size=8, world_size=1, num_experts=8)
    validate_mega_forward_inputs(
        torch.zeros(4, 8),
        torch.zeros(4, 2, dtype=torch.int64),
        torch.zeros(4, 2),
        p,
        top_k=2,
        quantize_input=True,
    )


def test_validate_compute_consistency_requires_do_finalize():
    import dataclasses

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
    from flashinfer.moe_ep.backends.split.kernel.fused_moe.validate import (
        validate_compute_consistency,
    )

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    fleet = _split(num_experts=8, world_size=4)
    moe_config = MoEConfig(
        routing=RoutingConfig(num_experts=8, top_k=4),
        quant=QuantConfig(variant=QuantVariant.BF16),
        experts=ExpertConfig(
            intermediate_size=2048,
            local_expert_offset=0,
            local_num_experts=2,
        ),
        backend=BackendOptions(candidates=(TrtllmBf16Config(),)),
        execution=ExecutionConfig(tune_max_num_tokens=128, do_finalize=False),
    )
    with pytest.raises(MoEEpConfigError, match="do_finalize"):
        validate_compute_consistency(fleet, bootstrap, moe_config)

    ok_config = dataclasses.replace(
        moe_config,
        execution=dataclasses.replace(moe_config.execution, do_finalize=True),
    )
    validate_compute_consistency(fleet, bootstrap, ok_config)


def test_ue8m0_quant_rejected_on_pre_blackwell_for_nixl():
    """If we're on sm_90, nixl_ep + UE8M0 should error.

    Skip when running on a host without CUDA or on Blackwell; the
    constraint only applies to sm_90 hardware.
    """
    import torch

    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")
    cc = torch.cuda.get_device_capability(0)
    if cc >= (10, 0):
        pytest.skip("Blackwell host; UE8M0 is allowed here")
    p = _split()
    q = FleetAlgoKnobQuantization(quants=frozenset({QuantType.UE8M0}))
    with pytest.raises(MoEEpConfigError, match="UE8M0"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4, quant=q)
