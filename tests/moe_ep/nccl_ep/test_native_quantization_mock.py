# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate NCCL forwarding and NVFP4 payload contracts without GPUs."""

from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize(
    "algorithm,layout,shape",
    [
        ("LOW_LATENCY", "RANK_MAJOR", (4, 4, 2048)),
        ("LOW_LATENCY", "EXPERT_MAJOR", (2, 16, 2048)),
        ("HIGH_THROUGHPUT", "EXPERT_MAJOR", (4, 4, 2048)),
    ],
)
def test_forwarded_scales_reach_provider_and_consumer(
    modern_ep, make_native_handle, algorithm, layout, shape
):
    from flashinfer.moe_ep import DispatchInputParams

    fleet, handle = make_native_handle(
        algorithm=algorithm, layout=layout, dispatch_quantization="forward"
    )
    x = torch.zeros(3, 2048, dtype=torch.float8_e4m3fn)
    scales = torch.ones(3, 64, dtype=torch.uint8)
    received = handle.dispatch(DispatchInputParams(x=[x, scales]))
    call = next(c for c in handle._handle.calls if c[0] == "dispatch")
    assert call[1].scales.buffer is scales
    assert (
        call[3]["config"].quantization_recipe
        == modern_ep.DispatchQuantizationRecipe.FWD
    )
    assert received.expert_tensors.shape == shape
    assert received.expert_scales.shape == (*shape[:-1], 64)
    assert call[2].scales.buffer is received.expert_scales or (
        call[2].scales.buffer.data_ptr() == received.expert_scales.data_ptr()
    )
    repeated = handle.dispatch(DispatchInputParams(x=[x, scales]))
    assert repeated.expert_scales.data_ptr() == received.expert_scales.data_ptr()
    handle.destroy()
    fleet.destroy()


@pytest.mark.parametrize("problem", ["missing", "rows", "alignment", "budget", "dtype"])
def test_invalid_forwarded_payload_stops_before_dispatch(make_native_handle, problem):
    from flashinfer.moe_ep import DispatchInputParams

    fleet, handle = make_native_handle(dispatch_quantization="forward")
    x = torch.zeros(3, 2048, dtype=torch.float8_e4m3fn)
    scales = torch.ones(3, 64, dtype=torch.uint8)
    if problem == "rows":
        scales = scales[:2]
    elif problem == "alignment":
        scales = torch.ones(3, 63, dtype=torch.uint8)
    elif problem == "budget":
        scales = torch.ones(3, 4096, dtype=torch.uint8)
    elif problem == "dtype":
        x = torch.zeros(3, 2048, dtype=torch.int64)
    payload = [x] if problem == "missing" else [x, scales]
    with pytest.raises(ValueError):
        handle.dispatch(DispatchInputParams(x=payload))
    assert not any(c[0] == "dispatch" for c in handle._handle.calls)
    handle.destroy()
    fleet.destroy()


def test_nvfp4_scales_zero_rows_and_recipe(modern_ep, make_native_handle):
    from flashinfer.moe_ep import CombineInputParams

    fleet, handle = make_native_handle(combine_quantization="nvfp4")
    x = torch.zeros(4, 4, 2048, dtype=torch.bfloat16)
    x[0, 0, 0] = -2
    handle.combine(CombineInputParams(x=[x]))
    call = next(c for c in handle._handle.calls if c[0] == "combine")
    scales = call[1].scales.buffer
    assert scales.shape == (4, 4, 1) and scales.dtype == torch.float32
    assert scales[0, 0, 0].item() == 1344.0
    assert scales[1:].eq(0).all() and torch.isfinite(scales).all()
    assert (
        call[3]["config"].quantization_recipe
        == modern_ep.CombineQuantizationRecipe.NVFP4
    )
    handle.destroy()
    fleet.destroy()


@pytest.mark.parametrize(
    "options",
    [
        dict(dispatch_quantization="forward"),
        dict(dispatch_zero_copy=True),
        dict(combine_quantization="nvfp4"),
    ],
)
def test_native_options_reject_older_provider(modern_ep, make_native_handle, options):
    modern_ep.get_lib_version = lambda: "0.2.0"
    with pytest.raises(ValueError, match="EP 0.3"):
        make_native_handle(**options)


@pytest.mark.parametrize(
    "algorithm,layout",
    [
        ("HIGH_THROUGHPUT", "EXPERT_MAJOR"),
        ("LOW_LATENCY", "EXPERT_MAJOR"),
    ],
)
def test_nvfp4_rejects_unsupported_layouts(make_native_handle, algorithm, layout):
    with pytest.raises(ValueError, match="LOW_LATENCY|RANK_MAJOR"):
        make_native_handle(
            algorithm=algorithm, layout=layout, combine_quantization="nvfp4"
        )


@pytest.mark.parametrize("forward", [False, True])
def test_split_layer_payload_and_scale_context(forward):
    import contextlib
    from types import MethodType
    from unittest.mock import Mock

    from flashinfer.moe_ep import DispatchOutput, FleetParams, MoEEpTensors
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    x = torch.ones(3, 2048, dtype=torch.bfloat16)
    values = torch.zeros(3, 2048, dtype=torch.uint8)
    scales = torch.ones(3, 64, dtype=torch.uint8)
    packed = Mock(return_value=x)
    quantize = Mock(return_value=(values, scales))
    compute = Mock(return_value=x)
    dispatch = Mock(
        return_value=DispatchOutput(
            expert_tensors=x, expert_scales=scales, num_tokens=3
        )
    )
    handle = SimpleNamespace(
        dispatch=dispatch, combine=lambda p: SimpleNamespace(x=p.x[0])
    )
    layer = SimpleNamespace(
        _forward_scales=forward,
        _fleet_params=FleetParams(
            num_experts=8, max_tokens_per_rank=4, token_hidden_size=2048
        ),
        _kernel=SimpleNamespace(
            pack_dispatch_payload=packed,
            quantize_for_dispatch=quantize,
            compute=compute,
        ),
        _stage=lambda *args: contextlib.nullcontext(),
        _publish_timings=lambda *args: None,
    )
    layer._fleet_kernel_context = MethodType(
        MoEEpSplitLayer._fleet_kernel_context, layer
    )
    inputs = MoEEpTensors(x, torch.zeros(3, 2, dtype=torch.int64), torch.ones(3, 2))
    MoEEpSplitLayer._fleet_dispatch_compute_combine(
        layer, handle, inputs, out=None, out_is_stable=False
    )
    payload = dispatch.call_args[0][0].x
    assert len(payload) == (2 if forward else 1)
    assert compute.call_args[0][0].expert_tensors_scale is (scales if forward else None)
    if forward:
        quantize.assert_called_once_with(x)
        packed.assert_not_called()
        assert payload[0].dtype == torch.float8_e4m3fn and payload[1] is scales
    else:
        packed.assert_called_once_with(x)
        quantize.assert_not_called()
        assert payload[0] is x


@pytest.mark.parametrize("option", ["dispatch_quantization", "combine_quantization"])
@pytest.mark.parametrize("backend", ["nixl_ep", "nvlink_two_sided"])
def test_layer_rejects_quantization_on_other_transport(option, backend):
    from flashinfer.moe_ep import FleetAlgoKnobNcclEpConfig
    from flashinfer.moe_ep.modes.split_layer import MoEEpSplitLayer

    value = "forward" if option == "dispatch_quantization" else "nvfp4"
    layer = MoEEpSplitLayer.__new__(MoEEpSplitLayer)
    torch.nn.Module.__init__(layer)
    layer._comm_backend = backend
    layer._fleet_knobs = [FleetAlgoKnobNcclEpConfig(**{option: value})]
    with pytest.raises(ValueError, match="requires nccl_ep"):
        layer._validate_at_init()


@pytest.mark.parametrize("invalid", ["dtype", "dimensions", "stride"])
def test_nvfp4_input_contract(make_native_handle, invalid):
    from flashinfer.moe_ep import CombineInputParams

    fleet, handle = make_native_handle(combine_quantization="nvfp4")
    x = torch.zeros(4, 4, 2048, dtype=torch.bfloat16)
    if invalid == "dtype":
        x = x.float()
    elif invalid == "dimensions":
        x = x.reshape(-1, 2048)
    else:
        x = x.transpose(0, 1)
    with pytest.raises(ValueError, match="NVFP4 combine requires"):
        handle.combine(CombineInputParams(x=[x]))
    assert not any(call[0] == "combine" for call in handle._handle.calls)
    handle.destroy()
    fleet.destroy()
