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
    num_local_experts: int | None = None,
) -> FleetParams:
    local = num_local_experts if num_local_experts is not None else num_experts // world_size
    return FleetParams(
        num_experts=num_experts,
        max_tokens_per_rank=max_tokens_per_rank,
        token_hidden_size=token_hidden_size,
        weights=dummy_moe_weights(num_local_experts=local, hidden=token_hidden_size),
    )


def test_num_experts_must_be_divisible_by_world_size():
    p = _split(num_experts=7, world_size=2, max_tokens_per_rank=64, num_local_experts=1)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_params(p, backend="nccl_ep", world_size=2)
    # Also fails for nixl_ep.
    p2 = _split(num_experts=7, world_size=2, max_tokens_per_rank=64, num_local_experts=1)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_params(p2, backend="nixl_ep", world_size=2)


def test_fleet_weights_local_expert_count():
    p = _split(num_experts=8, world_size=4, max_tokens_per_rank=64, num_local_experts=1)
    with pytest.raises(MoEEpConfigError, match="num_experts // world_size"):
        validate_fleet_weights(p, world_size=4)


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
    p = _split(num_experts=7, world_size=2, max_tokens_per_rank=64, num_local_experts=1)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_mega_fleet_params(
            p, world_size=2, intermediate_size=2048, top_k=4
        )


def test_mega_fleet_params_requires_128_aligned_sizes():
    p = _split(max_tokens_per_rank=64, token_hidden_size=1000)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_mega_fleet_params(
            p, world_size=4, intermediate_size=2048, top_k=4
        )
    p2 = _split(max_tokens_per_rank=64)
    with pytest.raises(MoEEpConfigError, match="intermediate_size"):
        validate_mega_fleet_params(
            p2, world_size=4, intermediate_size=1000, top_k=4
        )


def test_mega_fleet_params_happy_path():
    p = _split(max_tokens_per_rank=64)
    validate_mega_fleet_params(p, world_size=4, intermediate_size=2048, top_k=4)


def test_bootstrap_world_size_must_match_dist_when_initialized():
    from unittest import mock

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    with mock.patch("torch.distributed.is_initialized", return_value=True):
        with mock.patch("torch.distributed.get_world_size", return_value=8):
            with pytest.raises(MoEEpConfigError, match="BootstrapConfig.world_size"):
                validate_bootstrap_world_size(bootstrap)


def test_bootstrap_world_size_skipped_when_dist_not_initialized():
    from unittest import mock

    bootstrap = BootstrapConfig(world_size=4, rank=0)
    with mock.patch("torch.distributed.is_initialized", return_value=False):
        validate_bootstrap_world_size(bootstrap)


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
