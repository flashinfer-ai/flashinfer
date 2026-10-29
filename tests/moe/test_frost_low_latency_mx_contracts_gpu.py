# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Real MX buffers exercise alignment admission and duplicate-route fallback."""

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _assert_near, _quant, _require_gpu
from tests.moe.test_frost_low_latency_mx_activations import reference


def _case(fmt, top_k, packed=True, seed=8163):
    _require_gpu()
    torch.manual_seed(seed)
    e, t, h, i = 4, 2, 256, 96
    device = torch.device("cuda", torch.cuda.current_device())

    def weight(rows, cols):
        if fmt is QuantFormat.MXFP4:
            return (
                torch.randint(
                    0, 256, (e, rows, cols // 2), device=device, dtype=torch.uint8
                ),
                torch.full(
                    (e, rows, cols // 32), 120, device=device, dtype=torch.uint8
                ),
            )
        return _quant((torch.randn(e, rows, cols, device=device) * 0.08).bfloat16())[1:]

    w1, s1 = weight(2 * i, h)
    w2, s2 = weight(h, i)
    dx, q, sf = _quant(torch.randn(t, h, device=device).bfloat16())
    ids = torch.arange(t * top_k, device=device, dtype=torch.int32).view(t, top_k) % e
    scores = torch.rand(t, top_k, device=device) - 0.2
    config = MoEConfig(
        RoutingConfig(e, top_k),
        QuantConfig(weight=fmt, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        activation=SwiGLU(),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    view = FrostLowLatencyConfig.prepare_weights(
        w1, s1, w2, s2, activation=config.activation
    )
    pack = MoEWeightPack({"frost_low_latency": view})
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    act = MoEActivationPack(q, sf, ids, scores, routing_input_mode=mode)
    expected = lambda: reference(
        dx, w1, s1, w2, s2, ids, scores, config.activation, packed
    )[1]
    return config, act, pack, expected


@pytest.mark.parametrize("fmt", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize("packed", [False, True])
def test_mx_duplicate_routes_default_and_live_graph(fmt, packed):
    config, act, pack, expected = _case(fmt, top_k=6, packed=packed)
    layer = MoELayer(config)
    _assert_near(layer(act, pack), expected())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(act, pack).clone()
    for _ in range(3):
        act.topk_ids.add_(1).remainder_(config.routing.num_experts)
        act.topk_weights.copy_(torch.rand_like(act.topk_weights) - 0.2)
        graph.replay()
        _assert_near(output, expected())


@pytest.mark.parametrize("name", ["w1_weight", "w2_weight"])
def test_mxfp4_rejects_half_aligned_weight_before_gpu_launch(name):
    config, act, pack, expected = _case(QuantFormat.MXFP4, top_k=2)
    runner = FrostLowLatencyRunner(config, act.hidden_states_q.device)
    runner.check_support()
    runner.build()
    # Normal preparation remains admitted and the explicit input-device TC
    # compilation path must still produce correct outputs on this architecture.
    _assert_near(runner.forward(runner.pack_inputs(act, pack)), expected())
    view = pack.get_view("frost_low_latency")
    weight = view[name]
    storage = torch.empty(weight.numel() + 16, dtype=weight.dtype, device=weight.device)
    shifted = storage[16:].view_as(weight)
    shifted.copy_(weight)
    assert shifted.is_contiguous() and shifted.data_ptr() % 32 == 16
    view[name] = shifted
    with pytest.raises(ValueError, match=f"{name} must be 32-byte aligned"):
        runner.pack_inputs(act, pack)
