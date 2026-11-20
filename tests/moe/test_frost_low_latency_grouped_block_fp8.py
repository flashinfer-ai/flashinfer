# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped DeepSeek FP8: block scales, dynamic quantization and graph bindings."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import MoEActivationPack, SwiGLU, ReLU2, SiLU
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_deepseek_fp8 import _inputs, _reference, _quant
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check


def _setup(activation, prequantized):
    _require_gpu(allow_sm90=True, allow_sm120=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        pytest.skip("Grouped DeepSeek FP8 requires SM90 or SM120")
    cfg, x, ids, scores, view, pack = _inputs(activation, 3)
    x, sf = _quant(x) if prequantized else (x, None)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, sf, ids, scores), pack)
    return cfg, x, sf, ids, scores, view, runner, inputs


@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2(), SiLU()])
@pytest.mark.parametrize("prequantized", [False, True])
@pytest.mark.parametrize("group,stages,tile_k", [(2, 4, 64), (4, 4, 128)])
def test_grouped_block_fp8_live_graph_scales(
    activation, prequantized, group, stages, tile_k
):
    cfg, x, sf, ids, scores, view, runner, inputs = _setup(activation, prequantized)
    kind = "wgmma" if torch.cuda.get_device_capability() == (9, 0) else "warp_mma"
    tactic = (kind, group, stages, tile_k)
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.forward(inputs, tactic).clone()
    for step, magnitude in enumerate((0.0, 1e-7, 0.3)):
        if sf is None:
            x.copy_(torch.randn_like(x) * magnitude)
        else:
            x.copy_(
                (torch.randn_like(x, dtype=torch.float32) * (magnitude * 20)).to(
                    x.dtype
                )
            )
            sf.mul_(0.8)
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        # Invalid slots can become valid on the next replay: restore finite
        # scores first so padding NaN/Inf cannot contaminate the test reference.
        scores.copy_(torch.randn_like(scores) * 0.2)
        view["fc1_block_scales"].mul_(1.2)
        view["fc2_block_scales"].mul_(0.9)
        if step == 2:
            ids[0, 0], ids[-1, -1] = -1, cfg.routing.num_experts
            scores[0, 0], scores[-1, -1] = float("nan"), float("inf")
        _, expected, _ = _reference(x, sf, ids, scores, view, activation, True)
        assert torch.isfinite(expected).all()
        for buffer in inputs[9:14]:
            buffer.fill_(float("nan"))
        for _ in range(2):
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("prequantized", [False, True])
def test_grouped_block_fp8_independent_graph_bindings(prequantized):
    a = _setup(SwiGLU(), prequantized)
    b = _setup(SwiGLU(), prequantized)
    from cutlass import cute

    runner_a, inputs_a = a[-2:]
    runner_b, inputs_b = b[-2:]
    assert all(
        v.data_ptr() != w.data_ptr()
        for v, w in zip(inputs_a[:14], inputs_b[:14], strict=True)
    )
    kind = "wgmma" if torch.cuda.get_device_capability() == (9, 0) else "warp_mma"
    tactic = (kind, 4, 4, 128)
    runner_a.forward(inputs_a, tactic)
    original = cute.compile

    def unexpected(*args, **kwargs):
        raise AssertionError("Different live pointers recompiled identical metadata")

    cute.compile = unexpected
    try:
        runner_b.forward(inputs_b, tactic)
    finally:
        cute.compile = original
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    stream_b.wait_stream(torch.cuda.current_stream())
    graph_a, graph_b = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a, stream=stream_a):
        out_a = runner_a.forward(inputs_a, tactic).clone()
    with torch.cuda.graph(graph_b, stream=stream_b):
        out_b = runner_b.forward(inputs_b, tactic).clone()
    torch.cuda.synchronize()
    for epoch in range(3):
        for cfg, x, sf, ids, scores, view, _runner, inputs in (a, b):
            ids.add_(epoch + 1).remainder_(cfg.routing.num_experts)
            scores.neg_()
            view["fc1_block_scales"].mul_(0.75)
            view["fc2_block_scales"].mul_(1.1)
            if sf is not None:
                sf.mul_(1.25)
            else:
                x.mul_(0.5)
            for buffer in inputs[9:14]:
                buffer.fill_(float("nan"))
        stream_a.wait_stream(torch.cuda.current_stream())
        stream_b.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_a):
            graph_a.replay()
        with torch.cuda.stream(stream_b):
            graph_b.replay()
        torch.cuda.synchronize()
        for values, out in ((a, out_a), (b, out_b)):
            cfg, x, sf, ids, scores, view, runner, inputs = values
            _, expected, _ = _reference(x, sf, ids, scores, view, SwiGLU(), True)
            _check(out, expected)


def _admitted_setup(
    tokens=4,
    activation=None,
    prequantized=False,
    hidden=2048,
    intermediate=512,
    topk=10,
    finalize=True,
):
    from flashinfer.fused_moe import (
        FrostLowLatencyConfig,
        MoEWeightPack,
        MoEFinalizeConfig,
        QuantConfig,
        QuantFormat,
    )
    from flashinfer.fused_moe.prepare import _deepseek_fp8_quantize_weights
    from tests.moe.test_frost_low_latency_bf16 import _inputs as dense_inputs

    _require_gpu(allow_sm90=True, allow_sm120=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        pytest.skip("Grouped DeepSeek FP8 requires SM90 or SM120")
    activation = SwiGLU() if activation is None else activation
    cfg, x, w1, w2, ids, scores, _ = dense_inputs(
        activation,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=17,
        topk=topk,
    )
    w1, s1 = _deepseek_fp8_quantize_weights(w1)
    w2, s2 = _deepseek_fp8_quantize_weights(w2)
    view = FrostLowLatencyConfig.prepare_deepseek_fp8_weights(
        w1, s1, w2, s2, activation=activation
    )
    cfg = replace(
        cfg,
        quant=QuantConfig(QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
        finalize=MoEFinalizeConfig(do_finalize=finalize),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    x, sf = _quant(x) if prequantized else (x, None)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, sf, ids, scores), pack)
    return cfg, x, sf, ids, scores, view, runner, inputs


@pytest.mark.parametrize(
    "tokens,activation", [(4, SwiGLU()), (8, SwiGLU()), (8, ReLU2()), (8, SiLU())]
)
@pytest.mark.parametrize("prequantized", [False, True])
def test_grouped_block_fp8_admitted_candidates(tokens, activation, prequantized):
    cfg, x, sf, ids, scores, view, runner, inputs = _admitted_setup(
        tokens, activation, prequantized
    )
    choices = runner.get_valid_tactics(inputs, None)
    assert len([v for v in choices if v[0] == "ffma"]) == 16
    hopper = torch.cuda.get_device_capability() == (9, 0)
    expected = (
        [("wgmma", g, 4, 128) for g in ((2, 4) if activation == SwiGLU() else (2,))]
        if hopper
        else [("warp_mma", g, 2, 128) for g in (2, 4)]
        + ([("warp_mma", 4, 4, 128)] if tokens == 4 else [])
    )
    assert [v for v in choices if v[0] != "ffma"] == expected
    graphs = []
    for tactic in expected:
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        graphs.append((graph, out))
    for epoch in range(2):
        ids.copy_(ids.roll(1, dims=1))
        scores.neg_()
        view["fc1_block_scales"].mul_(1.1)
        view["fc2_block_scales"].mul_(0.9)
        if sf is None:
            x.mul_(-0.7)
        else:
            sf.mul_(0.75)
        if epoch:
            ids[0, 0], ids[-1, -1] = -1, cfg.routing.num_experts
            scores[0, 0], scores[-1, -1] = float("nan"), float("inf")
        _, reference, _ = _reference(x, sf, ids, scores, view, activation, True)
        for graph, out in graphs:
            for buffer in inputs[9:14]:
                buffer.fill_(float("nan"))
            graph.replay()
            _check(out, reference)


@pytest.mark.parametrize(
    "case",
    [
        "tokens2",
        "hidden",
        "intermediate",
        "topk",
        "relu2_t4",
        "silu_t4",
        "custom_swiglu",
        "raw",
    ],
)
def test_grouped_block_fp8_unmeasured_cases_keep_fma(case):
    args = {}
    if case == "tokens2":
        args["tokens"] = 2
    elif case == "hidden":
        args["hidden"] = 256
    elif case == "intermediate":
        args["intermediate"] = 128
    elif case == "topk":
        args["topk"] = 8
    elif case == "relu2_t4":
        args["activation"] = ReLU2()
    elif case == "silu_t4":
        args["activation"] = SiLU()
    elif case == "custom_swiglu":
        args["activation"] = SwiGLU(alpha=1.702)
    elif case == "raw":
        args["finalize"] = False
    *_, runner, inputs = _admitted_setup(**args)
    choices = runner.get_valid_tactics(inputs, None)
    assert len(choices) == 16 and all(t[0] == "ffma" for t in choices)
