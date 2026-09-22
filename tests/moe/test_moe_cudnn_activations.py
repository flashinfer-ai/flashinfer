# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Typed gated activations retain their semantics through Frost fusion/capture."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F

from flashinfer.fused_moe import (
    CudnnMoeConfig,
    GeGLU,
    GeGLUTanh,
    MoELayer,
    MoEWeightPack,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_unified_moe_cudnn import _case, _require_b200


ACTIVATIONS = [
    SiTU(),
    SiTU(gate_scale=0.7, linear_scale=1.2),
    SiTU(linear_scale=None),
    SiTU(gate_scale=0.7, linear_scale=None, clamp_limit=0.25),
    SiTU(clamp_limit=0.25),
    SwiGLU(),
    SwiGLU(alpha=1.7, beta=0.3, limit=0.25),
    SwiGLU(alpha=0.0, beta=-0.1, limit=0.5),
    SwiGLU(alpha=-0.7, beta=0.2, limit=0.5),
    GeGLU(),
    GeGLUTanh(),
    SwiGLUStep(limit=0.25),
]


def _activation(up, gate, activation):
    # Independent torch definitions, preserving the public operation order.
    if isinstance(activation, SwiGLU):
        up = up.clamp(-activation.limit, activation.limit)
        gate = gate.clamp(max=activation.limit)
        return gate * torch.sigmoid(activation.alpha * gate) * (up + activation.beta)
    if isinstance(activation, SiTU):
        if activation.clamp_limit is not None:
            up = up.clamp(-activation.clamp_limit, activation.clamp_limit)
            gate = gate.clamp(max=activation.clamp_limit)
        if activation.linear_scale is not None:
            up = activation.linear_scale * torch.tanh(up / activation.linear_scale)
        return (
            up
            * activation.gate_scale
            * torch.tanh(gate / activation.gate_scale)
            * torch.sigmoid(gate)
        )
    if isinstance(activation, GeGLU):
        return up * F.gelu(gate, approximate="none")
    if isinstance(activation, GeGLUTanh):
        return up * F.gelu(gate, approximate="tanh")
    if isinstance(activation, SwiGLUStep):
        return up.clamp(-activation.limit, activation.limit) * F.silu(gate).clamp(
            max=activation.limit
        )
    raise AssertionError(activation)


def _reference(act, w1, w2, activation):
    x = act.hidden_states_q.float()
    scales = act.topk_weights
    if act.routing_input_mode == RoutingInputMode.PackedPrecomputed:
        scales = scales.to(torch.bfloat16)
    result = torch.zeros_like(x)
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(act.topk_ids == expert)
        if tokens.numel() == 0:
            continue
        up, gate = (x[tokens] @ w1[expert].float().T).chunk(2, -1)
        intermediate = _activation(up, gate, activation).to(torch.bfloat16).float()
        y = (intermediate @ w2[expert].float().T).to(torch.bfloat16).float()
        result.index_add_(0, tokens, y * scales[tokens, slots, None].float())
    return result.to(torch.bfloat16)


def _prepared_case(activation, *, wide=True):
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    tile_name = (
        "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma"
        if wide
        else "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
    )
    knobs = replace(
        GemmKnobs.from_config(by_name(tile_name)), moe_sched_policy=int(wide)
    )
    tactic = (
        20400,
        tuple(sorted((int(k), int(v)) for k, v in knobs.to_public().items())),
    )
    backend = CudnnMoeConfig(
        use_native_routing=True, fc1_tactic=tactic, fc2_tactic=tactic
    )
    mode = (
        RoutingInputMode.PackedPrecomputed
        if wide
        else RoutingInputMode.UnpackedPrecomputed
    )
    config, act, _, w1, w2 = _case(mode, tokens=257, backend=backend)
    act.hidden_states_q.mul_(4)
    config = replace(config, activation=activation)
    weights = MoEWeightPack()
    weights.prepare_for(
        "cudnn",
        CudnnMoeConfig.prepare_weights(
            w1,
            w2,
            num_local_experts=8,
            hidden_size=256,
            intermediate_size=256,
            activation=activation,
        ),
    )
    return config, act, weights, w1, w2


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=repr)
@pytest.mark.parametrize(
    "wide", [True, False], ids=["wide-static-packed", "2cta-dynamic-unpacked"]
)
def test_typed_gated_activation_fused_and_live_capture(activation, wide, monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, w1, w2 = _prepared_case(activation, wide=wide)
    layer = MoELayer(config)
    actual = layer(act, weights)
    torch.testing.assert_close(
        actual, _reference(act, w1, w2, activation), rtol=0.02, atol=0.02
    )
    runner = layer.runners[0]
    state = runner._resources(runner.pack_inputs(act, weights))
    assert state["fused"], "Activation unexpectedly escaped the FC1 fusion"
    assert state["fc1"].graph.get_engine_and_knobs_at_index(0)[0] == 20400
    assert state["fc2"].graph.get_engine_and_knobs_at_index(0)[0] == 20400
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture, stream=side):
        actual = layer(act, weights)
    torch.cuda.current_stream().wait_stream(side)
    previous = actual.clone()
    act.hidden_states_q.neg_()
    act.topk_ids.add_(2).remainder_(8)
    act.topk_weights.mul_(0.5)
    for name in ("routed", "intermediate", "projected", "output"):
        state[name].fill_(float("nan"))
    actual.fill_(float("nan"))
    capture.replay()
    torch.testing.assert_close(
        actual, _reference(act, w1, w2, activation), rtol=0.02, atol=0.02
    )
    assert not torch.equal(actual, previous)


@pytest.mark.parametrize(
    "activation",
    [
        SiTU(linear_scale=None, clamp_limit=0.25),
        SwiGLU(alpha=1.7, beta=0.3, limit=0.25),
        GeGLU(),
        SwiGLUStep(limit=0.25),
    ],
    ids=repr,
)
def test_typed_activation_fp32_fallback(activation, monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    from flashinfer.fused_moe import cudnn_backend

    original = cudnn_backend._Stage

    def without_fusion(*args, **kwargs):
        if kwargs.get("fused"):
            raise NotImplementedError("Exercise the separate FP32 activation graph")
        return original(*args, **kwargs)

    monkeypatch.setattr(cudnn_backend, "_Stage", without_fusion)
    config, act, weights, w1, w2 = _prepared_case(activation)
    layer = MoELayer(config)
    actual = layer(act, weights)
    state = next(iter(layer.runners[0]._resources_cache.values()))
    assert not state["fused"]
    assert state["fc1_output"].dtype == torch.float32
    torch.testing.assert_close(
        actual, _reference(act, w1, w2, activation), rtol=0.02, atol=0.02
    )
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        actual = layer(act, weights)
    act.hidden_states_q.neg_()
    state["fc1_output"].fill_(float("nan"))
    state["intermediate"].fill_(float("nan"))
    actual.fill_(float("nan"))
    capture.replay()
    torch.testing.assert_close(
        actual, _reference(act, w1, w2, activation), rtol=0.02, atol=0.02
    )


def test_activation_parameters_remain_plan_owned_across_interleaved_captures(
    monkeypatch,
):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    first = SiTU(gate_scale=0.7, linear_scale=1.2, clamp_limit=0.25)
    second = SiTU(gate_scale=3.0, linear_scale=5.0, clamp_limit=0.75)
    config, act, weights, w1, w2 = _prepared_case(first)
    layers = [MoELayer(config), MoELayer(replace(config, activation=second))]
    graphs, outputs, states = [], [], []
    for layer in layers:
        layer(act, weights)
        state = next(iter(layer.runners[0]._resources_cache.values()))
        assert state["fused"]
        states.append(state)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outputs.append(layer(act, weights))
        graphs.append(graph)
    assert (
        layers[0].runners[0]._cache_key_extras()
        != layers[1].runners[0]._cache_key_extras()
    )
    pointers = [
        {tensor.data_ptr() for tensor in state["fc1"].scalar_bindings.values()}
        for state in states
    ]
    assert pointers[0] and pointers[0].isdisjoint(pointers[1])
    act.hidden_states_q.neg_()
    for index in (1, 0, 1, 0):
        outputs[index].fill_(float("nan"))
        graphs[index].replay()
        activation = (first, second)[index]
        torch.testing.assert_close(
            outputs[index], _reference(act, w1, w2, activation), rtol=0.02, atol=0.02
        )
    assert not torch.equal(outputs[0], outputs[1])
