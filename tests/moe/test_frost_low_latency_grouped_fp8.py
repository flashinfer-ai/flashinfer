# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Grouped SM90/SM120 FP8: per-tensor scales, typed boundaries and live graph inputs."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEWeightPack,
    MoEFinalizeConfig,
    FrostLowLatencyConfig,
    QuantConfig,
    QuantFormat,
    ReLU2,
    SiTU,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check, _inputs, _reference


def _require_grouped_fp8():
    _require_gpu(allow_sm90=True, allow_sm120=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        pytest.skip("Grouped E4M3 projection requires SM90 or SM120")


@pytest.mark.parametrize("index", [2, 3])
@pytest.mark.parametrize("activation", [SwiGLU(), ReLU2(), SiTU()])
@pytest.mark.parametrize("tile_k", [64, 128])
def test_grouped_fp8_changing_graph_inputs_and_scales(index, activation, tile_k):
    _require_grouped_fp8()
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import execute

    cfg, x, ids, scores, pack, view = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    kind = "wgmma" if torch.cuda.get_device_capability() == (9, 0) else "warp_mma"
    tactic = (kind, 2, 4, tile_k)
    execute(inputs, cfg, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = execute(inputs, cfg, tactic).clone()
    for step in range(2):
        x.copy_((torch.randn_like(x, dtype=torch.float32) * 12.8).to(x.dtype))
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        scores.copy_(torch.randn_like(scores) * 0.2)
        if step:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        view["fc1_dequant_scale"].mul_(1.1)
        view["fc2_dequant_scale"].mul_(0.9)
        view["fc2_act_quant_scale"].fill_(128.0 if step == 0 else 32.0)
        valid_scores = scores.masked_fill(
            (ids < 0) | (ids >= cfg.routing.num_experts), 0
        )
        _, expected, _ = _reference(x, ids, valid_scores, view, activation, True)
        for _ in range(2):
            for buffer in inputs[5:8]:
                buffer.fill_(float("nan"))
            graph.replay()
            _check(output, expected)


def _admitted_inputs(tokens=4, activation=None, hidden=2048, intermediate=512, topk=10):
    _require_grouped_fp8()
    from tests.moe.test_frost_low_latency_bf16 import _inputs as dense_inputs

    activation = SwiGLU() if activation is None else activation
    cfg, x, w1, w2, ids, scores, _ = dense_inputs(
        activation,
        tokens=tokens,
        hidden=hidden,
        intermediate=intermediate,
        experts=17,
        topk=topk,
    )
    view = FrostLowLatencyConfig.prepare_fp8_per_tensor_weights(
        w1,
        w2,
        hidden_states_scale_global=32.0,
        intermediate_scale_global=128.0,
        num_local_experts=17,
        hidden_size=hidden,
        intermediate_size=intermediate,
        activation=activation,
    )
    x, _ = FrostLowLatencyConfig.prepare_fp8_per_tensor_activations(
        x, hidden_states_scale_global=32.0
    )
    cfg = replace(
        cfg,
        quant=QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    return cfg, x, ids, scores, pack, view


@pytest.mark.parametrize(
    "tokens,activation",
    [
        (4, SwiGLU()),
        (8, SwiGLU()),
        (4, SiTU()),
        (8, SiTU()),
        (4, ReLU2()),
        (8, ReLU2()),
    ],
)
@pytest.mark.parametrize("candidate", [0, 1])
def test_grouped_fp8_admitted_tactic_and_graph(tokens, activation, candidate):
    _require_grouped_fp8()
    hopper = torch.cuda.get_device_capability() == (9, 0)
    if hopper and (candidate == 1 or (tokens == 4 and activation == ReLU2())):
        pytest.skip("Hopper has one FP8 candidate and retains FMA-only T4 ReLU2")
    cfg, x, ids, scores, pack, view = _admitted_inputs(
        tokens=tokens, activation=activation
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    choices = runner.get_valid_tactics(inputs, None)
    assert len([v for v in choices if v[0] == "ffma"]) == 16
    expected_choices = (
        [("wgmma", 2, 4, 128)]
        if hopper
        else [("warp_mma", group, 2, 128) for group in (2, 4)]
    )
    if not hopper and (
        activation in (SwiGLU(), SiTU()) or (tokens == 8 and activation == ReLU2())
    ):
        expected_choices += [("warp_mma", group, 2, 128, True) for group in (2, 4)]
    assert [v for v in choices if v[0] != "ffma"] == expected_choices
    tactic = expected_choices[candidate]
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic).clone()
    x.copy_((-x.float()).to(x.dtype))
    ids.copy_(ids.roll(1, dims=1))
    scores.copy_(scores.flip(1))
    view["fc1_dequant_scale"].mul_(1.1)
    view["fc2_dequant_scale"].mul_(0.9)
    view["fc2_act_quant_scale"].fill_(32.0)
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    for _ in range(2):
        for buffer in inputs[5:8]:
            buffer.fill_(float("nan"))
        graph.replay()
        _check(output, expected)


@pytest.mark.parametrize(
    "case",
    ["tokens2", "hidden", "intermediate", "topk", "activation", "custom_swiglu", "raw"],
)
def test_grouped_fp8_unmeasured_cases_keep_fma(case):
    args = {}
    if case == "tokens2":
        args["tokens"] = 2
    elif case == "hidden":
        args["hidden"] = 256
    elif case == "intermediate":
        args["intermediate"] = 128
    elif case == "topk":
        args["topk"] = 8
    elif case == "activation":
        _require_grouped_fp8()
        if torch.cuda.get_device_capability() == (9, 0):
            args["activation"] = ReLU2()
        else:
            from flashinfer.fused_moe import SiLU

            args["activation"] = SiLU()
    elif case == "custom_swiglu":
        args["activation"] = SwiGLU(alpha=1.702)
    cfg, x, ids, scores, pack, _ = _admitted_inputs(**args)
    if case == "raw":
        cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=False))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    assert all(v[0] == "ffma" for v in runner.get_valid_tactics(inputs, None))


@pytest.mark.parametrize(
    "tokens,activation",
    [(4, SwiGLU()), (8, SwiGLU()), (4, SiTU()), (8, SiTU()), (8, ReLU2())],
)
@pytest.mark.parametrize("group", [2, 4])
def test_sm120_fp8_weight_first_cache_and_live_graphs(
    tokens, activation, group, monkeypatch
):
    _require_grouped_fp8()
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("FP8 weight-first choices are qualified on SM120")
    from cutlass import cute
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    cache = {}
    monkeypatch.setattr(runtime, "_COMPILED", cache)
    cases = []
    for _ in range(2):
        cfg, x, ids, scores, pack, view = _admitted_inputs(
            tokens=tokens, activation=activation
        )
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        # Reference and mutate the buffers actually bound by execution.
        live_view = dict(view)
        for key, index in (
            ("fc1_expert_weights", 3),
            ("fc2_expert_weights", 4),
            ("fc1_dequant_scale", 9),
            ("fc2_act_quant_scale", 10),
            ("fc2_dequant_scale", 11),
        ):
            live_view[key] = inputs[index]
        cases.append((runner, inputs, live_view))
    old = ("warp_mma", group, 2, 128)
    overlap = (*old, True)
    runner, inputs, _ = cases[0]
    choices = runner.get_valid_tactics(inputs, None)
    assert old in choices and overlap in choices
    a = runner.forward(inputs, old).clone()
    b = runner.forward(inputs, overlap).clone()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert len(cache) == 2 and {key[4] for key in cache} == {old, overlap}

    def unexpected(*args, **kwargs):
        raise AssertionError("New buffers with identical metadata recompiled")

    monkeypatch.setattr(cute, "compile", unexpected)
    graphs = []
    for runner, inputs, _ in cases:
        for tactic in (old, overlap):
            runner.forward(inputs, tactic)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner.forward(inputs, tactic)
            graphs.append(graph)
    for epoch in range(2):
        for index, (runner, inputs, view) in enumerate(cases):
            x, ids, scores, w1, w2 = inputs[:5]
            x.copy_((-x.float() * (0.5 if index else 0.75)).to(x.dtype))
            w1.copy_((-w1.float()).to(w1.dtype))
            w2.copy_((w2.float() * 0.5).to(w2.dtype))
            ids.add_(1).remainder_(17)
            scores.copy_(scores.flip(1))
            view["fc1_dequant_scale"].mul_(1.1)
            view["fc2_dequant_scale"].mul_(0.9)
            view["fc2_act_quant_scale"].fill_(32.0 if epoch else 64.0)
            if epoch:
                ids[0, 0], ids[-1, -1] = -1, 17
                scores[0, 0], scores[-1, -1] = float("nan"), float("inf")
            valid_scores = scores.masked_fill((ids < 0) | (ids >= 17), 0)
            _, expected, _ = _reference(
                x, ids, valid_scores, view, runner.config.activation, True
            )
            for graph in graphs[2 * index : 2 * index + 2]:
                for buf in inputs[5:8]:
                    buf.fill_(float("nan"))
                graph.replay()
                _check(inputs[7], expected)


@pytest.mark.parametrize(
    "case",
    [
        "tokens2",
        "hidden",
        "intermediate",
        "topk",
        "custom_swiglu",
        "raw",
        "tile",
        "relu2_t4",
    ],
)
def test_sm120_fp8_weight_first_rejects_unqualified_manual_tactics(case):
    _require_grouped_fp8()
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("FP8 weight-first choices are qualified on SM120")
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
        args.update(tokens=4, activation=ReLU2())
    elif case == "custom_swiglu":
        args["activation"] = SwiGLU(alpha=1.702)
    cfg, x, ids, scores, pack, _ = _admitted_inputs(**args)
    if case == "raw":
        cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=False))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactic = ("warp_mma", 8 if case == "tile" else 2, 2, 128, True)
    assert tactic not in runner.get_valid_tactics(inputs, None)
    with pytest.raises(ValueError, match="Weight-first FP8 decode requires"):
        runner.forward(inputs, tactic)
