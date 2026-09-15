# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Public FP8 MoE quantization, live pack binding, and capture contracts."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFp8PerTensorConfig,
    CudnnFp8PerTensorRunner,
    GeGLU,
    GeGLUTanh,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantVariant,
    SiTU,
    SwiGLU,
    SwiGLUStep,
)
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_moe_cudnn_activations import _activation
from tests.moe.test_unified_moe_cudnn import _case, _require_b200

KEY = "cudnn_fp8_per_tensor"
ACTIVATIONS = [
    SwiGLU(),
    SiTU(),
    SiTU(linear_scale=None, clamp_limit=0.25),
    SwiGLU(alpha=1.7, beta=0.3, limit=0.5),
    GeGLU(),
    GeGLUTanh(),
    SwiGLUStep(limit=0.5),
]


def _tactic(wide):
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    name = (
        "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma"
        if wide
        else "CONFIG_sm100_128x256x128_128x256x32_cluster2x1_2ctamma"
    )
    knobs = replace(GemmKnobs.from_config(by_name(name)), moe_sched_policy=int(wide))
    return 20400, tuple(sorted((int(k), int(v)) for k, v in knobs.to_public().items()))


def _prepare_weights(w1, w2, activation, input_scale=64.0, intermediate_scale=128.0):
    pack = MoEWeightPack()
    view = CudnnFp8PerTensorConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=activation,
        hidden_states_scale_global=input_scale,
        intermediate_scale_global=intermediate_scale,
    )
    pack.prepare_for(KEY, view)
    return pack


def _prepared(activation, wide, tokens=257, default_tactics=False):
    mode = (
        RoutingInputMode.PackedPrecomputed
        if wide
        else RoutingInputMode.UnpackedPrecomputed
    )
    config, act, _, w1, w2 = _case(mode, tokens=tokens)
    tactic = None if default_tactics else _tactic(wide)
    backend = CudnnFp8PerTensorConfig(fc1_tactic=tactic, fc2_tactic=tactic)
    config = replace(
        config,
        activation=activation,
        quant=QuantConfig(variant=QuantVariant.FP8PerTensor),
        backend=BackendOptions((backend,)),
    )
    x, scale = CudnnFp8PerTensorConfig.prepare_activations(
        act.hidden_states_q, hidden_states_scale_global=64.0
    )
    act = replace(act, hidden_states_q=x, hidden_states_scale=scale)
    return config, act, _prepare_weights(w1, w2, activation), w1, w2


def _reference(act, view, activation):
    x = act.hidden_states_q.float()
    result = torch.zeros_like(x)
    weights = act.topk_weights
    if act.routing_input_mode == RoutingInputMode.PackedPrecomputed:
        weights = weights.to(torch.bfloat16).float()
    for expert in range(view["up"].shape[0]):
        tokens, slots = torch.where(act.topk_ids == expert)
        if tokens.numel() == 0:
            continue
        up = (x[tokens] @ view["up"][expert].float().T) * view["fc1_scale"][expert]
        gate = (x[tokens] @ view["gate"][expert].float().T) * view["fc1_scale"][expert]
        h = (
            (
                _activation(up, gate, activation)
                * view["intermediate_multiplier"].view(())
            )
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
            .float()
        )
        y = (
            ((h @ view["down"][expert].float().T) * view["fc2_scale"][expert])
            .to(torch.bfloat16)
            .float()
        )
        result.index_add_(0, tokens, y * weights[tokens, slots, None])
    return result.to(torch.bfloat16)


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm().clamp_min(1e-12)
    assert error <= 0.01, error
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)


def _poison(state):
    for name in ("routed", "intermediate"):
        state[name].view(torch.uint8).fill_(127)
    for name in ("projected", "output"):
        state[name].fill_(float("nan"))


def test_fp8_registration_and_preparation_contract():
    from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

    assert _BACKEND_RUNNERS[CudnnFp8PerTensorConfig] is CudnnFp8PerTensorRunner
    assert CudnnFp8PerTensorConfig.supported(100)
    assert not CudnnFp8PerTensorConfig.supported(120)
    w1 = torch.linspace(-1, 1, 2 * 64 * 32).reshape(2, 64, 32).to(torch.bfloat16)
    w2 = torch.linspace(-0.5, 0.5, 2 * 32 * 32).reshape(2, 32, 32).to(torch.bfloat16)
    view = _prepare_weights(w1, w2, SwiGLU()).get_view(KEY)
    multiplier1 = 448.0 / w1.float().abs().amax(dim=(-1, -2))
    multiplier2 = 448.0 / w2.float().abs().amax(dim=(-1, -2))
    expected = (
        (w1.float() * multiplier1[:, None, None])
        .clamp(-448, 448)
        .to(torch.float8_e4m3fn)
    )
    assert torch.equal(
        view["up"].view(torch.uint8), expected[:, :32].contiguous().view(torch.uint8)
    )
    assert torch.equal(
        view["gate"].view(torch.uint8), expected[:, 32:].contiguous().view(torch.uint8)
    )
    torch.testing.assert_close(
        view["fc1_scale"].flatten(), 1 / (64 * multiplier1), rtol=0, atol=0
    )
    torch.testing.assert_close(
        view["fc2_scale"].flatten(), 1 / (128 * multiplier2), rtol=0, atol=0
    )
    x = torch.tensor([[-100.0, 0.0, 1.0, 100.0]], dtype=torch.bfloat16)
    quantized, scale = CudnnFp8PerTensorConfig.prepare_activations(
        x, hidden_states_scale_global=64.0
    )
    assert scale is None
    assert torch.equal(quantized.float(), torch.tensor([[-448.0, 0.0, 64.0, 448.0]]))


@pytest.mark.parametrize("scale", [0.0, -1.0, float("nan"), float("inf"), 1e38])
def test_fp8_rejects_invalid_calibration(scale):
    w1 = torch.ones(2, 64, 32, dtype=torch.bfloat16)
    w2 = torch.ones(2, 32, 32, dtype=torch.bfloat16)
    with pytest.raises(ValueError):
        _prepare_weights(w1, w2, SwiGLU(), input_scale=scale)


@pytest.mark.parametrize("activation", ACTIVATIONS, ids=repr)
@pytest.mark.parametrize(
    "wide", [True, False], ids=["1cta-static-packed", "2cta-dynamic-unpacked"]
)
def test_fp8_public_moe_live_capture(activation, wide, monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, _, _ = _prepared(activation, wide)
    layer = MoELayer(config)
    view = weights.get_view(KEY)
    actual = layer(act, weights)
    _check(actual, _reference(act, view, activation))
    runner = layer.runners[0]
    state = runner._resources(runner.pack_inputs(act, weights))
    assert layer.winner_backend == KEY and state["fused"]
    for stage in ("fc1", "fc2"):
        assert state[stage].graph.get_engine_and_knobs_at_index(0)[0] == 20400
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        actual = layer(act, weights)
    torch.cuda.current_stream().wait_stream(side)
    previous = actual.clone()
    act.hidden_states_q.copy_((-act.hidden_states_q.float()).to(torch.float8_e4m3fn))
    act.topk_ids.add_(2).remainder_(8)
    act.topk_weights.mul_(0.5)
    view["fc1_scale"].mul_(0.5)
    view["fc2_scale"].mul_(1.5)
    view["intermediate_multiplier"].mul_(0.5)
    _poison(state)
    graph.replay()
    _check(actual, _reference(act, view, activation))
    assert not torch.equal(actual, previous)


def test_fp8_interleaved_weight_packs_and_captures(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, first, w1, w2 = _prepared(SiTU(), True)
    second = _prepare_weights(w1 * -0.75, w2 * 0.5, SiTU(), intermediate_scale=32.0)
    layer = MoELayer(config)
    layer(act, first)
    runner = layer.runners[0]
    inp1, inp2 = runner.pack_inputs(act, first), runner.pack_inputs(act, second)
    state = runner._resources(inp1)
    assert state is runner._resources(inp2)
    for inputs, pack in ((inp1, first), (inp2, second), (inp1, first)):
        _check(runner.forward(inputs), _reference(act, pack.get_view(KEY), SiTU()))
    captures = []
    for inputs in (inp1, inp2):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(inputs)
        captures.append(graph)
    first.get_view(KEY)["fc1_scale"].mul_(0.5)
    second.get_view(KEY)["fc2_scale"].mul_(0.5)
    for index in (0, 1, 0):
        _poison(state)
        captures[index].replay()
        _check(
            state["output"],
            _reference(act, (first, second)[index].get_view(KEY), SiTU()),
        )


def test_fp8_default_plan_and_large_router(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, _, _ = _prepared(
        SwiGLU(), True, tokens=1025, default_tactics=True
    )
    layer = MoELayer(config)
    _check(layer(act, weights), _reference(act, weights.get_view(KEY), SwiGLU()))
    state = next(iter(layer.runners[0]._resources_cache.values()))
    assert state["expert_counts"].numel() == 16
    for stage in ("fc1", "fc2"):
        assert state[stage].graph.get_engine_and_knobs_at_index(0)[0] == 20400


def test_fp8_exact_tactics_and_input_metadata(monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, _, _ = _prepared(SwiGLU(), True)
    layer = MoELayer(config)
    layer(act, weights)
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, weights)
    selected = runner.get_valid_tactics(inputs, None)[0]
    assert selected == (
        config.backend.candidates[0].fc1_tactic,
        config.backend.candidates[0].fc2_tactic,
    )
    _check(
        runner.forward(inputs, tactic=selected),
        _reference(act, weights.get_view(KEY), SwiGLU()),
    )
    bad = replace(act, hidden_states_q=act.hidden_states_q.float())
    with pytest.raises(ValueError, match="E4M3"):
        runner.pack_inputs(bad, weights)
    view = dict(weights.get_view(KEY))
    view["fc1_scale"] = view["fc1_scale"].flatten()
    malformed = MoEWeightPack()
    malformed.prepare_for(KEY, view)
    with pytest.raises(ValueError, match="fc1_scale"):
        runner.pack_inputs(act, malformed)
