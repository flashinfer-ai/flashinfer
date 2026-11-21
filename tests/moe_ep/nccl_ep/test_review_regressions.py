# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Graph storage, stream selection and failed reconfiguration regressions."""

import contextlib
import gc
from types import SimpleNamespace
from unittest.mock import Mock
import weakref

import pytest
import torch

from flashinfer.moe_ep import (
    BootstrapConfig,
    DispatchInputParams,
    EpAlgorithm,
    FleetAlgoKnobFaultTolerance,
    FleetAlgoKnobNcclEpConfig,
    HandleAlgoKnobNumReceivedTokens,
    HandleParams,
)
from flashinfer.moe_ep.backends.split.comm.nccl_ep import _runtime, fleet as module
from tests.moe_ep.nccl_ep.test_handle_mock import _make_fleet, _make_handle


@pytest.mark.parametrize("field", ["tokens", "scales"])
def test_forwarded_geometry_cannot_release_captured_storage(make_native_handle, field):
    fleet, handle = make_native_handle(dispatch_quantization="forward")
    values = torch.zeros(3, 2048, dtype=torch.float8_e4m3fn)
    scales = torch.ones(3, 64, dtype=torch.uint8)
    original = handle.dispatch(DispatchInputParams(x=[values, scales]))
    reference = weakref.ref(
        original.expert_tensors if field == "tokens" else original.expert_scales
    )
    pointer = reference().data_ptr()
    del original
    if field == "tokens":
        values = torch.zeros(3, 2016, dtype=values.dtype)
    else:
        scales = torch.ones(3, 80, dtype=scales.dtype)
    calls = len(handle._handle.calls)
    with pytest.raises(ValueError, match="receive geometry changed"):
        handle.dispatch(DispatchInputParams(x=[values, scales]))
    gc.collect()
    assert reference() is not None and reference().data_ptr() == pointer
    assert len(handle._handle.calls) == calls
    handle.destroy()
    fleet.destroy()


def test_eager_sizing_counter_is_private(fake_nccl_ep, bypass_build_checks):
    fake_nccl_ep.get_lib_version = lambda: "0.3.0"
    fleet = _make_fleet(
        fake_nccl_ep,
        algorithm=EpAlgorithm.HIGH_THROUGHPUT,
        knobs=[FleetAlgoKnobNcclEpConfig(eager_recv=True)],
    )
    shared = torch.zeros(1, dtype=torch.int32)
    knobs = [HandleAlgoKnobNumReceivedTokens(shared)]
    first, second = [_make_handle(fleet, extra_knobs=knobs) for _ in range(2)]
    first._recv_total_target.fill_(7)
    second._recv_total_target.fill_(19)
    second._publish_recv_count(second._stream)
    assert shared.item() == 19
    result = first.dispatch(
        DispatchInputParams(x=[torch.zeros(16, 2048, dtype=torch.bfloat16)])
    )
    assert result.num_tokens == 7 and result.expert_tensors.shape[1] == 7
    assert first._recv_total_target.data_ptr() != second._recv_total_target.data_ptr()
    first._publish_recv_count(first._stream)
    assert shared.item() == 7
    first.destroy()
    second.destroy()
    fleet.destroy()


@pytest.mark.parametrize("adopted", [False, True])
def test_native_replacement_failure_rejects_use_and_allows_retry(
    fake_nccl_ep, bypass_build_checks, monkeypatch, adopted
):
    from tests.moe_ep.nccl_ep.test_communicator_ownership_mock import make_fleet

    fleet = make_fleet(123 if adopted else None)
    old = fleet.group
    monkeypatch.setattr(old, "destroy", Mock())
    create = fake_nccl_ep.Group.create
    monkeypatch.setattr(
        fake_nccl_ep.Group, "create", Mock(side_effect=RuntimeError("native failure"))
    )
    bootstrap = BootstrapConfig(
        rank=0, world_size=4, nccl_comm=123 if adopted else None
    )
    with pytest.raises(RuntimeError, match="native failure"):
        fleet.update_topology(bootstrap)
    assert fleet._group is None and fleet._destroyed
    for operation in (
        lambda: fleet.group,
        lambda: fleet.create_handle(HandleParams(torch.zeros(1, 1, dtype=torch.int64))),
    ):
        with pytest.raises(ValueError, match="no live group"):
            operation()
    monkeypatch.setattr(fake_nccl_ep.Group, "create", create)
    fleet.update_topology(bootstrap)
    assert fleet.group is not old and not fleet._destroyed
    fleet.destroy()


@pytest.mark.parametrize("initial,replacement", [(None, 100), (100, None), (100, 200)])
def test_fault_tolerance_changes_reject_before_teardown(
    fake_nccl_ep, bypass_build_checks, monkeypatch, initial, replacement
):
    monkeypatch.setattr(module, "mask_ffi", lambda: SimpleNamespace(available=True))
    base = [] if initial is None else [FleetAlgoKnobFaultTolerance(timeout_ms=initial)]
    fleet = _make_fleet(fake_nccl_ep, knobs=base)
    old = fleet.group
    destroy = Mock()
    monkeypatch.setattr(old, "destroy", destroy)
    knobs = [FleetAlgoKnobNcclEpConfig()]
    if replacement is not None:
        knobs.append(FleetAlgoKnobFaultTolerance(timeout_ms=replacement))
    with pytest.raises(ValueError, match="fault-tolerance settings"):
        fleet.update_topology(BootstrapConfig(rank=0, world_size=4), knobs)
    assert fleet.group is old
    destroy.assert_not_called()
    fleet.destroy()


def test_zero_stream_is_default_and_explicit_capture_is_detected(monkeypatch):
    default, explicit = object(), object()
    external = Mock(return_value=explicit)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device=None: default)
    monkeypatch.setattr(torch.cuda, "ExternalStream", external)
    assert _runtime.torch_stream(0) is default
    external.assert_not_called()
    assert _runtime.torch_stream(7) is explicit
    current = [default]

    @contextlib.contextmanager
    def stream(value):
        old, current[0] = current[0], value
        try:
            yield
        finally:
            current[0] = old

    monkeypatch.setattr(torch.cuda, "stream", stream)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "is_current_stream_capturing", lambda: current[0] is explicit
    )
    assert _runtime.stream_is_capturing(7)
    assert current[0] is default
    assert not _runtime.stream_is_capturing(0)
    with pytest.raises(RuntimeError, match="query_fault"):
        module.NcclEpFleet._reject_graph_capture(
            SimpleNamespace(_current_stream=lambda: 7), "query_fault"
        )
    assert current[0] is default


@pytest.mark.parametrize("hidden,valid", [(2048, True), (2304, False)])
def test_forwarding_scale_alignment_checked_at_init(hidden, valid):
    from flashinfer.fused_moe.api import QuantFormat
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    layer = SimpleNamespace(
        _comm_backend_name=lambda: "nccl_ep",
        _fleet_knobs=[FleetAlgoKnobNcclEpConfig(dispatch_quantization="forward")],
        _forward_scales=True,
        _kernel=SimpleNamespace(dispatch_format=lambda: QuantFormat.MXFP8),
        _fleet_params=SimpleNamespace(token_hidden_size=hidden),
    )
    if valid:
        MoEEpSplitLayer._validate_nccl_config(layer)
    else:
        with pytest.raises(ValueError, match="scale rows"):
            MoEEpSplitLayer._validate_nccl_config(layer)
