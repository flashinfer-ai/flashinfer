# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded packed NVFP4 grouped MMA: live scales, routing, graphs and storage."""

from dataclasses import replace

import pytest
import torch
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_w4a16 import _reference


TACTICS = [
    ("warp_mma", 4, 2, 128),
    ("warp_mma", 8, 4, 128),
    ("warp_mma", 4, 2, 256),
    ("warp_mma", 8, 2, 256),
]


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    delta = actual.double() - expected.double()
    fp16 = expected.dtype is torch.float16
    assert delta.norm() / expected.double().norm().clamp_min(1e-30) < (
        0.003 if fp16 else 0.015
    )
    assert delta.abs().max() / expected.double().abs().max().clamp_min(1e-30) < (
        0.01 if fp16 else 0.06
    )


def _case(packed, dtype=torch.bfloat16):
    _require_gpu(allow_sm120=True)
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 3),
        (12, 0),
    ):
        pytest.skip("SM103/SM120 packed NVFP4 grouped qualification")
    if torch.cuda.get_device_capability() == (12, 0) and dtype is torch.float16:
        pytest.skip("SM120 FP16 grouped NVFP4 is not admitted")
    torch.manual_seed(7340 + int(packed))
    t, h, i, e, k = 8, 2048, 512, 32, 10
    x = torch.randn(t, h, device="cuda", dtype=dtype) * 0.25
    w1 = torch.randint(256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = (torch.rand(e, 2 * i, h // 16, device="cuda") * 0.02 + 0.005).to(
        torch.float8_e4m3fn
    )
    s2 = (torch.rand(e, h, i // 16, device="cuda") * 0.02 + 0.005).to(
        torch.float8_e4m3fn
    )
    a1 = torch.rand(e, 2, device="cuda") * 0.8 + 0.6
    a2 = torch.rand(e, device="cuda") * 0.8 + 0.6
    view = FrostLowLatencyConfig.prepare_nvfp4_weight_only_weights(
        w1, s1, a1, w2, s2, a2
    )
    ids = torch.randperm(e, device="cuda")[:k].to(torch.int32).repeat(t, 1)
    scores = torch.rand(t, k, device="cuda")
    scores /= scores.sum(-1, keepdim=True)
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(
            weight=QuantFormat.NVFP4,
            activation=QuantFormat.FP16 if dtype is torch.float16 else QuantFormat.BF16,
            output=QuantFormat.FP16 if dtype is torch.float16 else QuantFormat.BF16,
        ),
        ExpertConfig(i),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), pack
    )
    assert [
        v for v in runner.get_valid_tactics(inputs, None) if v[0] == "warp_mma"
    ] == (TACTICS if torch.cuda.get_device_capability() == (12, 0) else TACTICS[2:])
    assert (
        len([v for v in runner.get_valid_tactics(inputs, None) if v[0] == "ffma"]) == 64
    )
    return runner, inputs, view


def _expected(runner, inputs, view, packed):
    return _reference(
        inputs[0], inputs[7], inputs[8], view, runner.config.activation, packed
    )[1]


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("tactic", TACTICS)
def test_grouped_nvfp4_live_graph_and_fresh_storage(dtype, packed, tactic, monkeypatch):
    runner, inputs, view = _case(packed, dtype)
    if tactic not in runner.get_valid_tactics(inputs, None):
        pytest.skip("Tactic is outside this architecture qualification")
    _check(runner.forward(inputs, tactic), _expected(runner, inputs, view, packed))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic)
    for step in range(3):
        inputs[0].mul_(0.875)
        inputs[3].mul_(1.125)
        inputs[6].mul_(0.875)
        for index in (2, 5):
            scale = inputs[index].view(torch.float8_e4m3fn)
            scale.copy_((scale.float() * 0.875).to(torch.float8_e4m3fn))
        inputs[1].bitwise_xor_(1)
        inputs[4].bitwise_xor_(1)
        inputs[7].copy_((torch.arange(80, device="cuda").view(8, 10) + step) % 32)
        if step == 2:
            inputs[7][0, 0] = -1
            inputs[7][-1, -1] = 32
            inputs[8][0, 0] = float("nan")
            inputs[8][-1, -1] = float("inf")
        expected = _expected(runner, inputs, view, packed)
        for buf in inputs[9:12]:
            buf.fill_(float("nan"))
        graph.replay()
        _check(output, expected)

    old = inputs
    inputs = [v.clone() if isinstance(v, torch.Tensor) else v for v in old]
    view = dict(view)
    for key, index in (
        ("fc1_expert_weights", 1),
        ("fc1_block_scales", 2),
        ("fc1_dequant_scale", 3),
        ("fc2_expert_weights", 4),
        ("fc2_block_scales", 5),
        ("fc2_dequant_scale", 6),
    ):
        view[key] = inputs[index]
    expected = _expected(runner, inputs, view, packed)
    for tensor in old:
        if isinstance(tensor, torch.Tensor):
            tensor.view(torch.uint8).zero_()
    from cutlass import cute

    def no_compile(*args, **kwargs):
        raise AssertionError("Fresh storage must reuse executable metadata")

    monkeypatch.setattr(cute, "compile", no_compile)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fresh = runner.forward(inputs, tactic)
        second_graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(second_graph, stream=stream):
            runner.forward(inputs, tactic)
        for buf in inputs[9:12]:
            buf.fill_(float("nan"))
        second_graph.replay()
    torch.cuda.current_stream().wait_stream(stream)
    _check(fresh, expected)


def test_grouped_nvfp4_bounds():
    runner, inputs, _ = _case(False)
    cfg = runner.config
    excluded = [
        replace(cfg, activation=GeGLU()),
        replace(cfg, experts=ExpertConfig(1024)),
        replace(cfg, routing=RoutingConfig(32, 8)),
        replace(cfg, finalize=MoEFinalizeConfig(do_finalize=False)),
        replace(cfg, quant=replace(cfg.quant, activation=QuantFormat.FP16)),
        replace(cfg, quant=replace(cfg.quant, output=QuantFormat.FP16)),
        replace(cfg, quant=replace(cfg.quant, weight=QuantFormat.MXFP4)),
    ]
    for config in excluded:
        other = FrostLowLatencyRunner(config, inputs[0].device)
        other.check_support()
        other.build()
        assert all(t[0] == "ffma" for t in other.get_valid_tactics(inputs, None))
    for t, h in ((1, 2048), (2, 2048), (4, 2048), (8, 4096)):
        changed = [torch.empty(t, h, device="cuda", dtype=torch.bfloat16), *inputs[1:]]
        assert all(v[0] == "ffma" for v in runner.get_valid_tactics(changed, None))
