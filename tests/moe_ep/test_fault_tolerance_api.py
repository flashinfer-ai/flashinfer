"""FT API surface tests — knob, Fleet ABC defaults, capability probe.

No CUDA, no comms, no transport: everything here is host-side dataclass and
ABC behaviour.
"""

from __future__ import annotations

import dataclasses

import pytest

from flashinfer.moe_ep import (
    FleetAlgoKnobFaultTolerance,
    MoEEpFaultToleranceUnsupportedError,
    MoEEpTransportError,
    supports_fault_tolerance,
)
from flashinfer.moe_ep.algo_knobs import _index_knobs
from flashinfer.moe_ep.core.comm.fleet import Fleet


class _BareFleet(Fleet):
    """Minimal concrete Fleet that implements only the required abstracts."""

    def __init__(self, bootstrap=None, params=None, algo_knobs=()) -> None:
        pass

    def create_handle(self, params, algo_knobs=()):
        raise NotImplementedError

    def update_topology(self, bootstrap, algo_knobs=()) -> None:
        raise NotImplementedError

    def destroy(self) -> None:
        pass


class TestFaultToleranceKnob:
    def test_defaults(self) -> None:
        k = FleetAlgoKnobFaultTolerance()
        assert k.enabled is True
        assert k.timeout_ms == 0  # 0 = transport default
        assert k.reconcile_timeout_s > 0
        assert k.coordinator_takeover_s > 0

    def test_frozen_and_hashable(self) -> None:
        k = FleetAlgoKnobFaultTolerance(timeout_ms=5000)
        with pytest.raises(dataclasses.FrozenInstanceError):
            k.timeout_ms = 1  # type: ignore[misc]
        assert hash(k) == hash(FleetAlgoKnobFaultTolerance(timeout_ms=5000))

    def test_rejects_negative_timeout(self) -> None:
        with pytest.raises(ValueError, match="timeout_ms"):
            FleetAlgoKnobFaultTolerance(timeout_ms=-1)

    @pytest.mark.parametrize(
        "kwargs",
        [{"reconcile_timeout_s": 0.0}, {"coordinator_takeover_s": -1.0}],
    )
    def test_rejects_nonpositive_budgets(self, kwargs) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            FleetAlgoKnobFaultTolerance(**kwargs)

    def test_index_knobs_round_trip(self) -> None:
        k = FleetAlgoKnobFaultTolerance(timeout_ms=1234)
        idx = _index_knobs([k])
        assert idx[FleetAlgoKnobFaultTolerance] is k


class TestFleetAbcDefaults:
    """The FT methods are concrete raising defaults, not @abstractmethod."""

    def test_bare_fleet_is_instantiable(self) -> None:
        # Regression guard: if any FT method were made abstract, this would
        # raise TypeError and break every out-of-tree Fleet implementation.
        _BareFleet()

    def test_capability_defaults_false(self) -> None:
        assert _BareFleet().supports_fault_tolerance is False
        assert _BareFleet().active_mask_epoch == 0

    @pytest.mark.parametrize(
        "call",
        [
            lambda f: f.query_active_mask(),
            lambda f: f.query_fault(),
            lambda f: f.set_active_mask([1, 1]),
            lambda f: f.reconcile_active_mask(),
            lambda f: f.clear_faults(),
            lambda f: f.clear_faults(readmit=True),
        ],
    )
    def test_ft_methods_raise_unsupported(self, call) -> None:
        with pytest.raises(MoEEpFaultToleranceUnsupportedError) as ei:
            call(_BareFleet())
        # The message must name the knob so the fix is obvious.
        assert "FleetAlgoKnobFaultTolerance" in str(ei.value)

    def test_stub_fleet_still_works(self, stubbed_fleet_registry) -> None:
        # The duck-typed _StubFleet in conftest does not implement the FT
        # methods at all; adding them to the ABC must not break it.
        from flashinfer.moe_ep import BootstrapConfig, FleetParams, create_fleet

        fleet = create_fleet(
            BootstrapConfig(world_size=1, rank=0),
            FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=4096),
            backend="nccl_ep",
        )
        fleet.destroy()
        assert "fleet_init" in stubbed_fleet_registry


class TestSupportsFaultTolerance:
    def test_unknown_backend_is_false(self) -> None:
        assert supports_fault_tolerance("bogus") is False

    def test_never_raises(self) -> None:
        # Safe to call on a host with no transport built at all.
        for backend in ("nccl_ep", "nixl_ep", ""):
            assert isinstance(supports_fault_tolerance(backend), bool)


class TestTransportError:
    def test_message_includes_decoded_name(self) -> None:
        err = MoEEpTransportError(
            "ncclEpMaskQuery", 5, "  (hint)", code_name="ncclInvalidUsage"
        )
        assert err.fn == "ncclEpMaskQuery"
        assert err.code == 5
        assert "ncclInvalidUsage (5)" in str(err)
        assert "(hint)" in str(err)

    def test_message_without_name(self) -> None:
        assert "failed: 3" in str(MoEEpTransportError("ncclEpMaskClean", 3))
