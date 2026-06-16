"""Host-only unit tests for NcclEpFleet / NcclEpHandle (mocked ``nccl.ep``).

Rewritten for the nccl-ep-v0.1.0 ``nccl.ep`` API. These tests never touch a real
GPU comm or the nccl4py native lib. They inject a fake ``nccl.ep`` module (with
recording stand-ins for ``Group`` / ``Tensor`` / the config dataclasses / the
enums) plus a fake ``nccl.ep.interop.torch.get_nccl_comm_from_group``, and patch
the package build/arch checks.

What they verify is **marshaling and call sequencing**, not numerics:
``GroupConfig`` receives the expected field values, ``Group.create`` /
``create_handle`` are called with the right layout + int64 topk_idx.  Real
end-to-end correctness is covered by the on-cluster smoke + multirank tests.
"""

from __future__ import annotations

import enum
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fake nccl.ep module
# ---------------------------------------------------------------------------


class _RecordingConfig:
    """Base for fake config dataclasses — stores kwargs for assertions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_nccl_ep():
    """Build a fake ``nccl.ep`` module object recording all interactions."""
    ep = types.ModuleType("nccl.ep")
    log: dict = {"handles": [], "groups": []}

    class Algorithm(enum.IntEnum):
        LOW_LATENCY = 0
        HIGH_THROUGHPUT = 1

    class Layout(enum.IntEnum):
        UNSET = 0
        EXPERT_MAJOR = 1
        RANK_MAJOR = 2
        FLAT = 3

    class Tensor:
        def __init__(self, buffer, **kw):
            self.buffer = buffer
            self.shape = tuple(getattr(buffer, "shape", ()))
            self.dtype = getattr(buffer, "dtype", None)

    class GroupConfig(_RecordingConfig):
        pass

    class DispatchConfig(_RecordingConfig):
        pass

    class CombineConfig(_RecordingConfig):
        pass

    class LayoutInfo(_RecordingConfig):
        pass

    class DispatchInputs(_RecordingConfig):
        pass

    class DispatchOutputs(_RecordingConfig):
        pass

    class CombineInputs(_RecordingConfig):
        pass

    class CombineOutputs(_RecordingConfig):
        pass

    class FakeHandle:
        def __init__(self, layout, topk_idx, **kw):
            self.layout = layout
            self.topk_idx = topk_idx
            self.calls: list = []

        def dispatch(self, inputs, outputs, **kw):
            self.calls.append(("dispatch", inputs, outputs, kw))

        def combine(self, inputs, outputs, **kw):
            self.calls.append(("combine", inputs, outputs, kw))

        def complete(self, **kw):
            self.calls.append(("complete", kw))

        def destroy(self):
            self.calls.append(("destroy",))

    class FakeGroup:
        def __init__(self, comm, config):
            self.comm = comm
            self.config = config

        @classmethod
        def create(cls, comm, config):
            g = cls(comm, config)
            log["groups"].append(g)
            return g

        def create_handle(self, layout, topk_idx, **kw):
            h = FakeHandle(layout, topk_idx, **kw)
            log["handles"].append(h)
            return h

        def destroy(self):
            pass

    ep.Algorithm = Algorithm
    ep.Layout = Layout
    ep.Tensor = Tensor
    ep.GroupConfig = GroupConfig
    ep.DispatchConfig = DispatchConfig
    ep.CombineConfig = CombineConfig
    ep.LayoutInfo = LayoutInfo
    ep.DispatchInputs = DispatchInputs
    ep.DispatchOutputs = DispatchOutputs
    ep.CombineInputs = CombineInputs
    ep.CombineOutputs = CombineOutputs
    ep.Group = FakeGroup
    ep._log = log
    return ep


@pytest.fixture
def fake_nccl_ep():
    """Inject a fake ``nccl`` / ``nccl.ep`` / ``nccl.ep.interop.torch`` tree."""
    ep = _make_fake_nccl_ep()

    nccl_pkg = types.ModuleType("nccl")
    interop = types.ModuleType("nccl.ep.interop")
    interop_torch = types.ModuleType("nccl.ep.interop.torch")
    interop_torch.get_nccl_comm_from_group = lambda group=None: object()

    names = ("nccl", "nccl.ep", "nccl.ep.interop", "nccl.ep.interop.torch")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["nccl"] = nccl_pkg
    sys.modules["nccl.ep"] = ep
    sys.modules["nccl.ep.interop"] = interop
    sys.modules["nccl.ep.interop.torch"] = interop_torch
    try:
        yield ep
    finally:
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture
def bypass_build_checks():
    """Bypass _require_built + validate_arch_for_backend in the fleet module."""
    from flashinfer.moe_ep.nccl_ep import fleet as nccl_fleet

    with (
        mock.patch.object(nccl_fleet, "_require_built", return_value=None),
        mock.patch.object(nccl_fleet, "validate_arch_for_backend", return_value=None),
    ):
        yield


# ---------------------------------------------------------------------------
# Fleet config marshaling (host-only)
# ---------------------------------------------------------------------------


def test_fleet_builds_group_config(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig, EpAlgorithm, FleetParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )
    bootstrap = BootstrapConfig(world_size=4, rank=0)

    fleet = NcclEpFleet(bootstrap, params)

    assert len(fake_nccl_ep._log["groups"]) == 1
    cfg = fake_nccl_ep._log["groups"][0].config
    assert cfg.algorithm == fake_nccl_ep.Algorithm.LOW_LATENCY
    assert cfg.num_experts == 8
    assert cfg.max_dispatch_tokens_per_rank == 128
    assert cfg.max_token_bytes == 7168 * 2
    assert fleet.group is fake_nccl_ep._log["groups"][0]


def test_fleet_rejects_non_divisible_experts(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep._validators import MoEEpConfigError
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    params = FleetParams(
        num_experts=7,  # not divisible by world_size=4
        max_tokens_per_rank=128,
        token_hidden_size=7168,
    )
    with pytest.raises(MoEEpConfigError):
        NcclEpFleet(BootstrapConfig(world_size=4, rank=0), params)


# ---------------------------------------------------------------------------
# Handle create-time sequencing (needs a CUDA device for buffer allocation)
# ---------------------------------------------------------------------------


def test_handle_create_uses_expert_major_and_int64_topk(
    fake_nccl_ep, bypass_build_checks
):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("handle alloc needs a CUDA device")

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import BootstrapConfig, FleetParams, HandleParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    params = FleetParams(num_experts=8, max_tokens_per_rank=128, token_hidden_size=7168)
    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), params)

    topk_ids = torch.zeros(16, 2, dtype=torch.int32, device="cuda")
    weights = torch.ones(16, 2, dtype=torch.float32, device="cuda")
    fleet.create_handle(
        HandleParams(topk_ids=topk_ids),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )

    fake_handle = fake_nccl_ep._log["handles"][-1]
    assert fake_handle.layout == fake_nccl_ep.Layout.EXPERT_MAJOR
    # topk_idx is wrapped in a fake Tensor; the underlying buffer is int64.
    assert fake_handle.topk_idx.buffer.dtype == torch.int64


def test_handle_create_uses_rank_major_layout(fake_nccl_ep, bypass_build_checks):
    import torch

    if not torch.cuda.is_available():
        pytest.skip("handle alloc needs a CUDA device")

    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import (
        BootstrapConfig,
        EpAlgorithm,
        EpLayout,
        FleetParams,
        HandleParams,
    )
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    params = FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        algorithm=EpAlgorithm.LOW_LATENCY,
        layout=EpLayout.RANK_MAJOR,
    )
    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), params)

    topk_ids = torch.zeros(16, 2, dtype=torch.int32, device="cuda")
    weights = torch.ones(16, 2, dtype=torch.float32, device="cuda")
    fleet.create_handle(
        HandleParams(topk_ids=topk_ids),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )

    fake_handle = fake_nccl_ep._log["handles"][-1]
    assert fake_handle.layout == fake_nccl_ep.Layout.RANK_MAJOR
    assert fake_handle.topk_idx.buffer.dtype == torch.int64


def test_fleet_params_rejects_rank_major_under_ht():
    from flashinfer.moe_ep.config import EpAlgorithm, EpLayout, FleetParams

    with pytest.raises(ValueError):
        FleetParams(
            num_experts=8,
            max_tokens_per_rank=128,
            token_hidden_size=7168,
            algorithm=EpAlgorithm.HIGH_THROUGHPUT,
            layout=EpLayout.RANK_MAJOR,
        )
