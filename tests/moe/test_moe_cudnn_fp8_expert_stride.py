# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed expert-stride views retain their storage and bind live captured weights."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFp8PerTensorConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingInputMode,
    SwiGLU,
)
from tests.moe.test_moe_cudnn_fp8 import KEY, _check, _poison, _reference, _tactic
from tests.moe.test_unified_moe_cudnn import _case

LAYOUT = "blocked_128x128_v1"


def _prepare(w1, w2, **kwargs):
    return CudnnFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        hidden_states_scale_global=64.0,
        intermediate_scale_global=128.0,
        **kwargs,
    )


def test_packed_fc1_preparation_retains_shared_storage():
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Packed MoE validation covers SM100")
    _, _, _, w1, w2 = _case(RoutingInputMode.UnpackedPrecomputed)
    separate = _prepare(w1, w2, weight_layout=LAYOUT)
    paired = _prepare(w1, w2, weight_layout=LAYOUT, copy_fc1_weights=False)
    up, gate = paired["up"], paired["gate"]
    assert up.untyped_storage().data_ptr() == gate.untyped_storage().data_ptr()
    assert gate.data_ptr() - up.data_ptr() == w2.shape[2] * w1.shape[2]
    assert up.stride(0) == gate.stride(0) == 2 * w2.shape[2] * w1.shape[2]
    for name in separate:
        assert torch.equal(
            separate[name].reshape(-1).view(torch.uint8),
            paired[name].reshape(-1).view(torch.uint8),
        )


@pytest.mark.parametrize("storage", ["paired", "padded"])
@pytest.mark.parametrize(
    "tokens,hidden", [(17, 256), (257, 256), (17, 4096), (257, 4096)]
)
def test_packed_expert_stride_capture(storage, tokens, hidden, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Packed MoE validation covers SM100")
    from cudnn.gemm.frost.compiler import _blocked_moe_weight_view

    mode = (
        RoutingInputMode.PackedPrecomputed
        if tokens == 17
        else RoutingInputMode.UnpackedPrecomputed
    )
    config, act, _, w1, w2 = _case(mode, tokens=tokens, hidden=hidden)
    activation = SwiGLU()
    config = replace(
        config,
        activation=activation,
        quant=QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
        backend=BackendOptions(
            (
                CudnnFp8PerTensorConfig(
                    weight_layout=LAYOUT,
                    fc1_tactic=_tactic(True),
                    fc2_tactic=_tactic(True),
                ),
            )
        ),
    )
    x, scale = CudnnFp8PerTensorConfig.prepare_activations(
        act.hidden_states_q, hidden_states_scale_global=64.0
    )
    act = replace(act, hidden_states_q=x, hidden_states_scale=scale)
    reference = _prepare(w1, w2)
    packed = _prepare(
        w1, w2, weight_layout=LAYOUT, copy_fc1_weights=storage != "paired"
    )
    padding = []
    if storage == "padded":
        for name in ("up", "gate", "down"):
            tensor = packed[name]
            expert_elements = tensor.numel() // tensor.shape[0]
            # Every expert is independently padded; NaNs reveal a lost pitch.
            backing = torch.full(
                (tensor.shape[0], expert_elements + 16),
                127,
                dtype=torch.uint8,
                device=tensor.device,
            )
            view = backing.view(torch.float8_e4m3fn).as_strided(
                tensor.shape, (expert_elements + 16, *tensor.stride()[1:])
            )
            view.copy_(tensor)
            packed[name] = view
            padding.append(backing[:, expert_elements:])
    for name in ("up", "gate", "down"):
        tensor = packed[name]
        flat = _blocked_moe_weight_view(tensor)
        assert flat.data_ptr() == tensor.data_ptr()
        assert flat.untyped_storage().data_ptr() == tensor.untyped_storage().data_ptr()
        assert flat.stride(0) == tensor.stride(0)
    weights = MoEWeightPack()
    weights.prepare_for(KEY, packed)
    layer = MoELayer(config)
    layer(act, weights)
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, weights)
    state = runner._resources(inputs)
    expected = _reference(act, reference, activation)
    _check(runner.forward(inputs), expected)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Prepared expert-stride execution must not copy or allocate"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(torch.Tensor, "contiguous", forbidden)
        patch.setattr(runner, "_prepare_stages", forbidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs)
    for pad in padding:
        pad.zero_()
    _poison(state)
    graph.replay()
    _check(output, expected)
    for name in ("up", "gate", "down"):
        previous = expected
        packed[name].view(torch.uint8).bitwise_xor_(128)
        reference[name].view(torch.uint8).bitwise_xor_(128)
        expected = _reference(act, reference, activation)
        assert not torch.equal(previous, expected)
        _poison(state)
        graph.replay()
        _check(output, expected)
    act.topk_ids.add_(1).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    x.view(torch.uint8).bitwise_xor_(128)
    _poison(state)
    graph.replay()
    _check(output, _reference(act, reference, activation))
    torch.cuda.set_sync_debug_mode("error")
    try:
        runner.forward(inputs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        runner.forward(inputs)
        torch.cuda.synchronize()
    names = [event.name for event in prof.events()]
    kernels = [name for name in names if "cudnn_kernel_frost_sm100_moe" in name]
    assert len(kernels) == 2 and all("packed_b_128x128_v1" in name for name in kernels)


@pytest.mark.parametrize("pitch", [0, 65520, 65537, -65536])
def test_packed_expert_stride_rejects_invalid_pitch(pitch):
    from cudnn.gemm.frost.compiler import _blocked_moe_weight_view

    class BadView:
        shape = (8, 2, 2, 128, 128)

        def stride(self):
            return (pitch, 32768, 16384, 128, 1)

        def reshape(self, *shape):
            raise AssertionError("Invalid expert pitch reached reshape")

    with pytest.raises(ValueError, match="expert stride"):
        _blocked_moe_weight_view(BadView())


@pytest.mark.parametrize(
    "pitch,accepted",
    [
        (65536, True),
        (65552, True),
        (131072, True),
        (0, False),
        (65520, False),
        (65537, False),
    ],
)
def test_packed_expert_stride_graph_contract(pitch, accepted):
    import cudnn
    from cudnn.gemm.frost.graph_analyzer import analyze_with_binding

    g = cudnn.pygraph(
        io_data_type=cudnn.data_type.FP8_E4M3,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    a = g.tensor(
        name="a",
        dim=[1, 17, 256],
        stride=[4352, 256, 1],
        data_type=cudnn.data_type.FP8_E4M3,
    )
    b = g.tensor(
        name="b",
        dim=[8, 2, 2, 128, 128],
        stride=[pitch, 32768, 16384, 128, 1],
        data_type=cudnn.data_type.FP8_E4M3,
    )
    offsets = g.tensor(
        name="offsets",
        dim=[8, 1, 1],
        stride=[1, 1, 1],
        data_type=cudnn.data_type.INT32,
    )
    y = g.moe_grouped_matmul(
        a, b, offsets, mode=cudnn.moe_grouped_matmul_mode.NONE, weight_layout=LAYOUT
    )
    y.set_output(True).set_data_type(cudnn.data_type.BFLOAT16)
    if accepted:
        chain, binding = analyze_with_binding(g)
        assert (chain.matmul.M, chain.matmul.N, chain.matmul.K) == (17, 256, 256)
        assert chain.moe.weight_layout == LAYOUT
        assert tuple(binding.b_operands[0].get_stride()) == (
            pitch,
            32768,
            16384,
            128,
            1,
        )
    else:
        with pytest.raises(NotImplementedError, match="expert stride"):
            analyze_with_binding(g)
