# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Configuration contracts without initializing a GPU or communicator."""

from types import SimpleNamespace

import pytest
import torch

from flashinfer.comm import RdmaDispatchLayout, RdmaMoeAlltoAll, RdmaMoeAlltoAllConfig
from flashinfer.comm import rdma_moe_alltoall as api


@pytest.fixture
def native_calls(monkeypatch):
    calls = []

    def create(name):
        def record(*args):
            calls.append((name, args))
            return object()

        return record

    def gather(records, value, group=None):
        records[:] = [value] * len(records)

    monkeypatch.setattr(api.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(api.dist, "get_rank", lambda group: 0)
    monkeypatch.setattr(api.dist, "get_world_size", lambda group: 4)
    monkeypatch.setattr(api.dist, "all_gather_object", gather)
    monkeypatch.setattr(api.torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(api.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        api.torch.cuda,
        "get_device_properties",
        lambda device: SimpleNamespace(
            name="Unprofiled SM120", major=12, minor=0, multi_processor_count=110
        ),
    )
    monkeypatch.setattr(api, "_call", lambda function, *args: function(*args))
    monkeypatch.setattr(
        api,
        "get_gin_module",
        lambda: SimpleNamespace(
            gin_get_unique_id=lambda: [0] * 128,
            gin_create=create("legacy"),
            gin_create_with_dispatch=create("independent"),
        ),
    )
    return calls


def test_default_configuration_preserves_native_entry(native_calls):
    buffer = RdmaMoeAlltoAll(None, 8, 5120, 384)
    assert buffer.dispatch_dtype == buffer.communication_dtype == "nvfp4"
    assert buffer.dispatch_layout is None
    assert len(native_calls) == 1
    entry, args = native_calls[0]
    assert entry == "legacy"
    assert len(args) == 19


@pytest.mark.parametrize(
    "dispatch",
    [{"dispatch_dtype": "fp8"}, {"dispatch_layout": RdmaDispatchLayout(513, 7)}],
)
def test_dispatch_codec_does_not_change_combine_contract(native_calls, dispatch):
    buffer = RdmaMoeAlltoAll(
        None,
        8,
        5120,
        384,
        config=RdmaMoeAlltoAllConfig(communication_dtype="bf16", **dispatch),
    )
    assert buffer.communication_dtype == "bf16"
    assert buffer.combine_reduction == "fp32"
    assert buffer.dispatch_dtype == ("raw" if "dispatch_layout" in dispatch else "fp8")
    assert len(native_calls) == 1
    entry, args = native_calls[0]
    assert entry == "independent"
    # The original combine precision parameter remains BF16; only the new
    # constructor's dispatch descriptor changes.
    assert args[11] == 2
    if "dispatch_layout" in dispatch:
        assert args[12] == "coalesced"
        assert args[13] == "legacy"
        assert args[19:] == (3, 513, 7)
    else:
        assert args[19:] == (0, 0, 0)


@pytest.mark.parametrize(
    "config,match",
    [
        (RdmaMoeAlltoAllConfig(dispatch_dtype="mxfp8"), "dispatch_dtype"),
        (RdmaMoeAlltoAllConfig(dispatch_layout=RdmaDispatchLayout(0)), "row bytes"),
        (RdmaMoeAlltoAllConfig(dispatch_layout=RdmaDispatchLayout(1, -1)), "row bytes"),
        (
            RdmaMoeAlltoAllConfig(dispatch_layout=RdmaDispatchLayout(2**31 - 96)),
            "row bytes",
        ),
        (RdmaMoeAlltoAllConfig(dispatch_layout=RdmaDispatchLayout(True)), "row bytes"),
        (
            RdmaMoeAlltoAllConfig(
                dispatch_dtype="fp8", dispatch_layout=RdmaDispatchLayout(513)
            ),
            "mutually exclusive",
        ),
        (
            RdmaMoeAlltoAllConfig(
                dispatch_layout=RdmaDispatchLayout(513), communication_mode="legacy"
            ),
            "capacity > 32",
        ),
        (
            RdmaMoeAlltoAllConfig(
                communication_dtype="bf16",
                dispatch_dtype="nvfp4",
                combine_reduction="bf16_local_partial_planned",
            ),
            "typed BF16",
        ),
    ],
)
def test_invalid_dispatch_configuration_fails_before_native_resources(
    native_calls, config, match
):
    with pytest.raises(RuntimeError, match=match):
        RdmaMoeAlltoAll(None, 8, 5120, 384, config=config)
    assert native_calls == []


def test_fp8_receive_ignores_nvfp4_only_scale(native_calls, monkeypatch):
    buffer = RdmaMoeAlltoAll(
        None,
        1,
        4096,
        4,
        config=RdmaMoeAlltoAllConfig(communication_dtype="bf16", dispatch_dtype="fp8"),
    )
    # Exercise the Python/FFI boundary with CPU storage and stubbed launches;
    # this does not initialize CUDA or simulate numerical communication.
    buffer.device = torch.device("cpu")
    monkeypatch.setattr(buffer, "_check_call", lambda: 0)
    monkeypatch.setattr(
        buffer._module, "gin_dispatch", lambda *args: None, raising=False
    )
    received_scales = []

    def dequantize(precision, data, scales, counts, global_scale, output):
        assert precision == 0
        received_scales.append(global_scale)

    monkeypatch.setattr(
        api,
        "get_quantization_module",
        lambda: SimpleNamespace(dequantize_dispatch_recv=dequantize),
    )
    result = buffer.dispatch(
        torch.empty((1, 4096), dtype=torch.uint8),
        torch.zeros((1, 1), dtype=torch.int32),
        x_scales=torch.empty((1, 32), dtype=torch.float32),
        input_dtype="prequantized",
        output_dtype="bf16",
        nvfp4_global_scale=object(),
    )
    assert received_scales == [1.0]
    assert result.data.dtype == torch.bfloat16
    assert result.scales.numel() == 0


@pytest.mark.parametrize(
    "capacity,mode,expected_dispatch",
    [
        (32, "auto", "coalesced"),
        (33, "auto", "legacy"),
        (33, "coalesced", "coalesced"),
        (33, "legacy", "legacy"),
    ],
)
def test_opaque_rank_packet_capacity_boundary_and_explicit_control(
    native_calls, capacity, mode, expected_dispatch
):
    buffer = RdmaMoeAlltoAll(
        None,
        capacity,
        5120,
        384,
        config=RdmaMoeAlltoAllConfig(
            communication_dtype="bf16",
            communication_mode=mode,
            dispatch_layout=RdmaDispatchLayout(513, 7),
        ),
    )
    entry, args = native_calls[0]
    assert entry == "independent"
    assert (args[12] or args[8]) == expected_dispatch
    assert args[19:] == (3, 513, 7)
    assert buffer.communication_dtype == "bf16"
    if mode == "auto":
        # The opaque threshold changes only dispatch, not combine policy.
        assert args[13] == "legacy"
        assert (
            "rank-packet dispatch"
            in buffer._strategy_selection["dispatch_selection_reason"]
        ) == (capacity > 32)


def test_opaque_legacy_rejects_capacity32_before_resources(native_calls):
    with pytest.raises(RuntimeError, match="capacity > 32"):
        RdmaMoeAlltoAll(
            None,
            32,
            5120,
            384,
            config=RdmaMoeAlltoAllConfig(
                communication_dtype="bf16",
                communication_mode="legacy",
                dispatch_layout=RdmaDispatchLayout(513, 7),
            ),
        )
    assert native_calls == []
