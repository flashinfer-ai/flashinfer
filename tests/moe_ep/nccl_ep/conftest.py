"""Shared fake ``nccl.ep`` fixtures for host-only NcclEpFleet/Handle tests.

``fake_nccl_ep`` injects a recording stand-in for the whole ``nccl`` package
tree (``nccl.ep``, ``nccl.core``, ``nccl.ep.interop.torch``) into
``sys.modules`` so fleet/handle marshaling and the host-path caching layer can
be exercised without a GPU, RDMA fabric, or the nccl4py wheel.
"""

from __future__ import annotations

import ctypes
import enum
import sys
import types
from unittest import mock

import pytest


class _RecordingConfig:
    """Base for fake config dataclasses — stores kwargs for assertions."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_fake_nccl_ep():
    """Build a fake ``nccl.ep`` module object recording all interactions."""
    ep = types.ModuleType("nccl.ep")
    log: dict = {"handles": [], "groups": [], "comm_from_group": []}

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
        pass

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
    # Real CFUNCTYPEs so NcclEpFleet._install_torch_allocator's @AllocFn
    # decoration and ctypes.cast(...)-to-address both work against the fake.
    ep.AllocFn = ctypes.CFUNCTYPE(
        ctypes.c_int, ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_void_p
    )
    ep.FreeFn = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p)
    ep.Group = FakeGroup
    ep._log = log
    return ep


@pytest.fixture
def fake_nccl_ep():
    """Inject a fake ``nccl`` / ``nccl.ep`` / ``nccl.core`` / interop tree."""
    ep = _make_fake_nccl_ep()

    nccl_pkg = types.ModuleType("nccl")
    core = types.ModuleType("nccl.core")

    class Communicator:
        instances: list = []  # noqa: RUF012

        def __init__(self, ptr=None):
            self.ptr = ptr
            type(self).instances.append(self)

    core.Communicator = Communicator
    ep._core = core

    interop = types.ModuleType("nccl.ep.interop")
    interop_torch = types.ModuleType("nccl.ep.interop.torch")

    def _get_nccl_comm_from_group(group=None):
        ep._log["comm_from_group"].append(group)
        return object()

    interop_torch.get_nccl_comm_from_group = _get_nccl_comm_from_group

    names = ("nccl", "nccl.ep", "nccl.core", "nccl.ep.interop", "nccl.ep.interop.torch")
    saved = {name: sys.modules.get(name) for name in names}
    sys.modules["nccl"] = nccl_pkg
    sys.modules["nccl.ep"] = ep
    sys.modules["nccl.core"] = core
    sys.modules["nccl.ep.interop"] = interop
    sys.modules["nccl.ep.interop.torch"] = interop_torch
    try:
        yield ep
    finally:
        Communicator.instances.clear()
        for name, mod in saved.items():
            if mod is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = mod


@pytest.fixture
def bypass_build_checks():
    """Bypass build/arch checks and distributed comm resolution."""
    from flashinfer.moe_ep.backends.split.comm.nccl_ep import fleet as nccl_fleet

    with (
        mock.patch.object(nccl_fleet, "_require_built", return_value=None),
        mock.patch.object(nccl_fleet, "validate_arch_for_backend", return_value=None),
        mock.patch.object(
            nccl_fleet, "validate_bootstrap_world_size", return_value=None
        ),
        mock.patch.object(nccl_fleet, "_resolve_comm", return_value=object()),
    ):
        yield
