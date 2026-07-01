"""Host-only unit tests for the vLLM-integration API extensions (mocked ``nccl``).

Covers the three ``flashinfer.moe_ep`` gaps that the vLLM all2all backend needs:

* GAP 1 — :func:`NcclEpFleet._resolve_comm` *adopts* an existing ``ncclComm_t``
  (``BootstrapConfig.nccl_comm``) instead of always bootstrapping a fresh
  communicator, and mirrors a specific ``process_group`` when asked.
* GAP 2 — :class:`FleetAlgoKnobAllocator` routes NCCL-EP buffers through a custom
  (or torch-caching) allocator via ``GroupConfig.alloc``.
* GAP 3 — :class:`HandleAlgoKnobNumReceivedTokens` binds an HT ``recv_total_counter``
  at handle-create time and :class:`DispatchOutput` surfaces the recv counts.

Like ``test_fleet_mock.py`` these verify marshaling / call sequencing against a
fake ``nccl`` tree — no GPU comm, no nccl4py native lib. On-cluster tests cover
real numerics.
"""

from __future__ import annotations

import ctypes
import enum
import sys
import types
from unittest import mock

import pytest


# ---------------------------------------------------------------------------
# Fake nccl / nccl.core / nccl.ep tree (superset of test_fleet_mock's fake)
# ---------------------------------------------------------------------------


class _RecordingConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


class _FakeCommunicator:
    """Stand-in for ``nccl.core.Communicator``. Records the adopted ptr and
    asserts (like the real one) that no destroy/abort is ever called on an
    adopted comm."""

    def __init__(self, ptr: int = 0):
        self.ptr = ptr
        self.destroyed = False

    def destroy(self):  # pragma: no cover - must NOT be called on adopted comm
        self.destroyed = True


def _make_fake_nccl_ep():
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

    class AllocConfig(_RecordingConfig):
        # Mirror the real defaults so a caller can leave fields at 0.
        def __init__(self, alloc_fn=0, free_fn=0, context=0):
            super().__init__(alloc_fn=alloc_fn, free_fn=free_fn, context=context)

    class FakeHandle:
        def __init__(self, layout, topk_idx, **kw):
            self.layout = layout
            self.topk_idx = topk_idx
            self.create_kwargs = kw
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
    ep.AllocConfig = AllocConfig
    # Real ctypes function types so the torch-caching trampolines are castable
    # to integer addresses (the callbacks are never invoked in these tests).
    ep.AllocFn = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p
    )
    ep.FreeFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    ep.Group = FakeGroup
    ep._log = log
    return ep


@pytest.fixture
def fake_nccl_ep():
    ep = _make_fake_nccl_ep()

    nccl_pkg = types.ModuleType("nccl")
    core = types.ModuleType("nccl.core")
    core.Communicator = _FakeCommunicator
    interop = types.ModuleType("nccl.ep.interop")
    interop_torch = types.ModuleType("nccl.ep.interop.torch")

    calls: dict = {"get_comm_group": []}

    def _get_comm(group=None):
        calls["get_comm_group"].append(group)
        return _FakeCommunicator(ptr=0xF00D)  # a "fresh" comm sentinel

    interop_torch.get_nccl_comm_from_group = _get_comm
    ep._interop_calls = calls

    names = (
        "nccl",
        "nccl.core",
        "nccl.ep",
        "nccl.ep.interop",
        "nccl.ep.interop.torch",
    )
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["nccl"] = nccl_pkg
    sys.modules["nccl.core"] = core
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
    from flashinfer.moe_ep.nccl_ep import fleet as nccl_fleet

    with (
        mock.patch.object(nccl_fleet, "_require_built", return_value=None),
        mock.patch.object(nccl_fleet, "validate_arch_for_backend", return_value=None),
    ):
        yield


def _ll_params():
    from flashinfer.moe_ep.config import EpAlgorithm, FleetParams

    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
        algorithm=EpAlgorithm.LOW_LATENCY,
    )


def _ht_params():
    from flashinfer.moe_ep.config import EpAlgorithm, FleetParams

    return FleetParams(
        num_experts=8,
        max_tokens_per_rank=128,
        token_hidden_size=7168,
        dtype_bytes=2,
        algorithm=EpAlgorithm.HIGH_THROUGHPUT,
    )


# ---------------------------------------------------------------------------
# GAP 1 — communicator adoption / group mirroring
# ---------------------------------------------------------------------------


def test_gap1_adopts_existing_nccl_comm(fake_nccl_ep, bypass_build_checks):
    """nccl_comm set → wrap it (Communicator(ptr=...)), never bootstrap fresh."""
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0, nccl_comm=0xABCD), _ll_params()
    )
    # The adopted comm carries our ptr and the fresh-bootstrap path was NOT used.
    assert isinstance(fleet._comm, _FakeCommunicator)
    assert fleet._comm.ptr == 0xABCD
    assert fake_nccl_ep._interop_calls["get_comm_group"] == []
    # Fleet.destroy() must never destroy the adopted comm (only the group).
    fleet.destroy()
    assert fleet._comm.destroyed is False


def test_gap1_mirrors_process_group_when_no_comm(fake_nccl_ep, bypass_build_checks):
    """No nccl_comm but a process_group → mirror THAT group (not WORLD)."""
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    sentinel_pg = object()
    NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0, process_group=sentinel_pg), _ll_params()
    )
    assert fake_nccl_ep._interop_calls["get_comm_group"] == [sentinel_pg]


def test_gap1_default_group_when_nothing_set(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _ll_params())
    assert fake_nccl_ep._interop_calls["get_comm_group"] == [None]


# ---------------------------------------------------------------------------
# GAP 2 — allocator plumbing into GroupConfig.alloc
# ---------------------------------------------------------------------------


def test_gap2_no_knob_leaves_alloc_default(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _ll_params())
    cfg = fake_nccl_ep._log["groups"][-1].config
    # No allocator knob → we do not set `alloc`; GroupConfig uses its default.
    assert "alloc" not in cfg.kwargs


def test_gap2_raw_addresses_forwarded(fake_nccl_ep, bypass_build_checks):
    from flashinfer.moe_ep.algo_knobs import FleetAlgoKnobAllocator
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _ll_params(),
        algo_knobs=[FleetAlgoKnobAllocator(alloc_fn=0x111, free_fn=0x222, context=0x7)],
    )
    cfg = fake_nccl_ep._log["groups"][-1].config
    assert cfg.alloc.alloc_fn == 0x111
    assert cfg.alloc.free_fn == 0x222
    assert cfg.alloc.context == 0x7


def test_gap2_torch_caching_installs_trampolines(fake_nccl_ep, bypass_build_checks):
    pytest.importorskip("torch")
    from flashinfer.moe_ep.algo_knobs import FleetAlgoKnobAllocator
    from flashinfer.moe_ep.config import BootstrapConfig
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(
        BootstrapConfig(world_size=4, rank=0),
        _ll_params(),
        algo_knobs=[FleetAlgoKnobAllocator(torch_caching=True)],
    )
    cfg = fake_nccl_ep._log["groups"][-1].config
    # Non-zero C addresses were handed to the library...
    assert cfg.alloc.alloc_fn not in (0, None)
    assert cfg.alloc.free_fn not in (0, None)
    # ...and the trampolines are anchored on the fleet (lifetime rule).
    assert hasattr(fleet, "_alloc_trampolines")
    assert len(fleet._alloc_trampolines) == 2


# ---------------------------------------------------------------------------
# GAP 3 — HT recv-count exposure (host-only; CPU tensors are fine with the fake)
# ---------------------------------------------------------------------------


def test_gap3_ht_binds_recv_total_counter_when_knob_set(
    fake_nccl_ep, bypass_build_checks
):
    torch = pytest.importorskip("torch")
    from flashinfer.moe_ep.algo_knobs import (
        HandleAlgoKnobNumReceivedTokens,
        HandleAlgoKnobTopKWeights,
    )
    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _ht_params())

    topk = torch.zeros(16, 2, dtype=torch.int32)  # CPU
    weights = torch.ones(16, 2, dtype=torch.float32)
    recv_total = torch.zeros(1, dtype=torch.int32)

    handle = fleet.create_handle(
        HandleParams(topk_ids=topk),
        algo_knobs=[
            HandleAlgoKnobTopKWeights(weights=weights),
            HandleAlgoKnobNumReceivedTokens(target=recv_total),
        ],
    )

    fake_handle = fake_nccl_ep._log["handles"][-1]
    # HT layout + a handle-time LayoutInfo carrying recv_total_counter was bound.
    assert fake_handle.layout == fake_nccl_ep.Layout.FLAT
    li = fake_handle.create_kwargs["layout_info"]
    assert li is not None
    assert li.recv_total_counter.buffer is recv_total

    # dispatch surfaces the same counter tensor for the consumer to trim on.
    out = handle.dispatch(_dispatch_params([torch.zeros(16, 7168)]))
    assert out.recv_total_counter is recv_total


def test_gap3_ht_no_knob_keeps_layout_info_none(fake_nccl_ep, bypass_build_checks):
    torch = pytest.importorskip("torch")
    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _ht_params())
    topk = torch.zeros(16, 2, dtype=torch.int32)
    weights = torch.ones(16, 2, dtype=torch.float32)

    handle = fleet.create_handle(
        HandleParams(topk_ids=topk),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )
    fake_handle = fake_nccl_ep._log["handles"][-1]
    # Verified default HT path is unchanged: no handle-time layout_info.
    assert fake_handle.create_kwargs["layout_info"] is None
    out = handle.dispatch(_dispatch_params([torch.zeros(16, 7168)]))
    assert out.recv_total_counter is None


def test_gap3_ll_surfaces_expert_counts(fake_nccl_ep, bypass_build_checks):
    torch = pytest.importorskip("torch")
    from flashinfer.moe_ep.algo_knobs import HandleAlgoKnobTopKWeights
    from flashinfer.moe_ep.config import BootstrapConfig, HandleParams
    from flashinfer.moe_ep.nccl_ep.fleet import NcclEpFleet

    fleet = NcclEpFleet(BootstrapConfig(world_size=4, rank=0), _ll_params())
    topk = torch.zeros(16, 2, dtype=torch.int32)
    weights = torch.ones(16, 2, dtype=torch.float32)

    handle = fleet.create_handle(
        HandleParams(topk_ids=topk),
        algo_knobs=[HandleAlgoKnobTopKWeights(weights=weights)],
    )
    out = handle.dispatch(_dispatch_params([torch.zeros(16, 7168)]))
    # LL EXPERT_MAJOR surfaces the library-written per-expert counts
    # ([num_local_experts] = 8 // 4 = 2), and no HT recv_total_counter.
    assert out.expert_counts is not None
    assert out.expert_counts.shape[0] == 2
    assert out.recv_total_counter is None


def _dispatch_params(tensors):
    from flashinfer.moe_ep.config import DispatchInputParams

    return DispatchInputParams(x=tensors)
