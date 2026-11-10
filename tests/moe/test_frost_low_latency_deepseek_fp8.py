# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Block-scale, activation and dynamic-binding contracts for the routed FMA path."""

from dataclasses import replace

import pytest
import torch
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _activation
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check


def _quant(value):
    groups = value.float().reshape(value.shape[0], -1, 128)
    scales = groups.abs().amax(-1).clamp_min(1e-4) * (1.0 / 448.0)
    q = (groups / scales[..., None]).clamp(-448, 448).to(torch.float8_e4m3fn)
    return q.reshape_as(value), scales.t().contiguous()


def _inputs(activation, index=0, top_k=3):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    torch.manual_seed(6400 + index)
    t, h, i, e = (
        (1, 2, 4, 8)[index % 4],
        (128, 256)[index % 2],
        (128, 256)[index % 2],
        max(8, top_k),
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * 0.3
    branches = 2 if activation.is_gated else 1
    w1 = (torch.randn(e, branches * i, h, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    w2 = (torch.randn(e, h, i, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    s1 = torch.rand(e, branches * i // 128, h // 128, device="cuda") * 0.2 + 0.05
    s2 = torch.rand(e, h // 128, i // 128, device="cuda") * 0.2 + 0.05
    view = FrostLowLatencyConfig.prepare_deepseek_fp8_weights(
        w1, s1, w2, s2, activation=activation
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    ids = (
        torch.arange(t * top_k, device="cuda", dtype=torch.int32).reshape(t, top_k) % e
    )
    scores = torch.randn(t, top_k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, top_k),
        QuantConfig(QuantFormat.DeepSeekFp8, QuantFormat.DeepSeekFp8),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    return cfg, x, ids, scores, view, pack


def _reference(x, sf, ids, scores, view, activation, packed):
    if x.dtype is torch.bfloat16:
        x, sf = _quant(x)
    dx = x.double() * sf.t().double().repeat_interleave(128, 1)
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    s1, s2 = view["fc1_block_scales"], view["fc2_block_scales"]
    effective = scores.bfloat16().float() if packed else scores.float()
    rows = []
    result = torch.zeros_like(dx, dtype=torch.float32)
    for token in range(x.shape[0]):
        for slot, expert in enumerate(ids[token].tolist()):
            if not 0 <= expert < w1.shape[0]:
                rows.append(
                    torch.zeros(x.shape[1], device=x.device, dtype=torch.bfloat16)
                )
                continue
            first = w1[expert].double() * s1[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            pre = (first @ dx[token]).float()
            up, gate = pre.chunk(2) if activation.is_gated else (pre, None)
            mid = _activation(up, gate, activation)[None]
            qm, sm = _quant(mid)
            dm = qm.double()[0] * sm[:, 0].double().repeat_interleave(128)
            second = w2[expert].double() * s2[expert].double().repeat_interleave(
                128, 0
            ).repeat_interleave(128, 1)
            value = (second @ dm).float().bfloat16()
            rows.append(value)
            result[token] = result[token] + value.float() * effective[token, slot]
    return torch.stack(rows), result.bfloat16(), effective


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_deepseek_fp8_activations_and_dynamic_graph(index, activation):
    cfg, x, ids, scores, view, pack = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    graphs = []
    for tactic in (("ffma", 1, 4), ("ffma", 4, 8)):
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        graphs.append((graph, out))
    for step, magnitude in enumerate((0.0, 1e-7, 0.3)):
        x.copy_(torch.randn_like(x) * magnitude)
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        scores.mul_(-0.8)
        view["fc1_block_scales"].mul_(1.2)
        view["fc2_block_scales"].mul_(0.9)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, None, ids, scores, view, activation, True)
        for graph, out in graphs:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("prequantized", [False, True])
def test_deepseek_fp8_public_contract(packed, finalize, prequantized):
    cfg, x, ids, scores, view, pack = _inputs(SwiGLU(), 1)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    x, sf = _quant(x) if prequantized else (x, None)
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    layer = MoELayer(cfg)
    act = MoEActivationPack(x, sf, ids, scores, routing_input_mode=mode)
    out = layer(act, pack)
    for _ in range(2):
        if sf is None:
            x.mul_(0.8)
        else:
            sf.mul_(0.8)
        out = layer(act, pack)
        raw, expected, effective = _reference(
            x, sf, ids, scores, view, cfg.activation, packed
        )
        if finalize:
            _check(out, expected)
        else:
            _check(out[0], raw)
            torch.testing.assert_close(out[1].float(), effective, atol=0, rtol=0)
            torch.testing.assert_close(
                out[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
            )


@pytest.mark.parametrize("prequantized", [False, True])
def test_deepseek_fp8_warmed_binding_identity(monkeypatch, prequantized):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import deepseek_fp8

    cfg, x, ids, scores, view, pack = _inputs(SwiGLU(), 1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    for value, sf in ((x, None), _quant(x)):
        inputs = runner.pack_inputs(MoEActivationPack(value, sf, ids, scores), pack)
        runner.forward(inputs, ("ffma", 2, 4))
    if prequantized:
        x, sf = value.clone(), sf.clone()
    else:
        x, sf = x.clone(), None
    view = {name: value.clone() for name, value in view.items()}
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    ids, scores = ids.clone(), scores.clone()
    inputs = runner.pack_inputs(MoEActivationPack(x, sf, ids, scores), pack)

    def forbidden(*args, **kwargs):
        raise AssertionError("warmed execution must not reconstruct compiled metadata")

    monkeypatch.setattr(deepseek_fp8.cute, "compile", forbidden)
    monkeypatch.setattr(deepseek_fp8, "from_dlpack", forbidden)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.forward(inputs, ("ffma", 2, 4))
    if sf is None:
        x.mul_(1.3)
    else:
        sf.mul_(1.3)
    graph.replay()
    _, expected, _ = _reference(x, sf, ids, scores, view, cfg.activation, True)
    _check(out, expected)


@pytest.mark.parametrize("first,second", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_deepseek_fp8_max_tokens_routes(first, second):
    cfg, x, ids, scores, view, pack = _inputs(SwiGLU(), 3, top_k=32)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    out = runner.forward(inputs, ("ffma", first, second))
    _, expected, _ = _reference(x, None, ids, scores, view, cfg.activation, True)
    _check(out, expected)


def test_deepseek_fp8_preserves_weights_and_rejects_bad_metadata():
    cfg, x, ids, scores, view, pack = _inputs(SwiGLU())
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match="must not supply activation scales"):
        runner.pack_inputs(
            MoEActivationPack(x, torch.ones(1, device=x.device), ids, scores), pack
        )
    qx, sf = _quant(x)
    with pytest.raises(ValueError, match="FP32"):
        runner.pack_inputs(MoEActivationPack(qx, sf.double(), ids, scores), pack)
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    prepared = FrostLowLatencyConfig.prepare_deepseek_fp8_weights(
        w1, view["fc1_block_scales"], w2, view["fc2_block_scales"]
    )
    for name in ("fc1_expert_weights", "fc2_expert_weights"):
        torch.testing.assert_close(
            prepared[name].view(torch.uint8),
            view[name].view(torch.uint8),
            atol=0,
            rtol=0,
        )
    with pytest.raises(ValueError, match="block scales"):
        FrostLowLatencyConfig.prepare_deepseek_fp8_weights(
            w1, view["fc1_block_scales"].double(), w2, view["fc2_block_scales"]
        )
