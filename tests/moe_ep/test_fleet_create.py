"""create_fleet routing and error-path tests."""

from __future__ import annotations

import pytest


def test_create_fleet_unknown_backend_raises_key_error():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet

    with pytest.raises(KeyError, match="unknown backend"):
        create_fleet(
            BootstrapConfig(world_size=1, rank=0),
            FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8),
            backend="not_registered",
        )


def test_create_fleet_rejects_non_string_backend_name():
    from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet

    class _BadBackend:
        backend_name = 42

    with pytest.raises(TypeError, match="backend_name"):
        create_fleet(
            BootstrapConfig(world_size=1, rank=0),
            FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8),
            backend=_BadBackend(),
        )


def test_create_fleet_accepts_config_object_backend_name(stubbed_fleet_registry):
    from flashinfer.moe_ep import BootstrapConfig, NvepConfig, FleetParams, create_fleet

    fleet = create_fleet(
        BootstrapConfig(world_size=1, rank=0),
        FleetParams(num_experts=2, max_tokens_per_rank=4, token_hidden_size=8),
        backend=NvepConfig(),
    )
    assert stubbed_fleet_registry == ["fleet_init"]
    fleet.destroy()
    assert stubbed_fleet_registry == ["fleet_init", "destroy"]
