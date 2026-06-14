"""B9 — constraint validator unit tests."""

from __future__ import annotations

import pytest

from flashinfer.moe_ep import (
    FleetAlgoKnobQuantization,
    FleetParams,
    MoEEpConfigError,
    QuantType,
    validate_fleet_params,
)


def test_num_experts_must_be_divisible_by_world_size():
    p = FleetParams(num_experts=7, max_tokens_per_rank=64, token_hidden_size=4096)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_params(p, backend="nccl_ep", world_size=2)
    # Also fails for nixl_ep.
    p2 = FleetParams(num_experts=7, max_tokens_per_rank=64, token_hidden_size=4096)
    with pytest.raises(MoEEpConfigError, match="num_experts"):
        validate_fleet_params(p2, backend="nixl_ep", world_size=2)


def test_nixl_ep_max_tokens_cap():
    p = FleetParams(num_experts=8, max_tokens_per_rank=2048, token_hidden_size=4096)
    with pytest.raises(MoEEpConfigError, match="max_tokens_per_rank"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4)
    # nccl_ep has no equivalent cap.
    validate_fleet_params(p, backend="nccl_ep", world_size=4)


def test_nixl_ep_hidden_size_set():
    p = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=1024)
    with pytest.raises(MoEEpConfigError, match="token_hidden_size"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4)
    # nccl_ep allows arbitrary hidden sizes.
    validate_fleet_params(p, backend="nccl_ep", world_size=4)


def test_nixl_ep_supported_hidden_sizes_pass():
    for h in (2048, 4096, 7168, 8192):
        p = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=h)
        validate_fleet_params(p, backend="nixl_ep", world_size=4)


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
    p = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=4096)
    q = FleetAlgoKnobQuantization(quants=frozenset({QuantType.UE8M0}))
    with pytest.raises(MoEEpConfigError, match="UE8M0"):
        validate_fleet_params(p, backend="nixl_ep", world_size=4, quant=q)
