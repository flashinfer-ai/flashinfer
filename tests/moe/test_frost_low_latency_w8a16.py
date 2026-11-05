# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Weight-only E4M3 with BF16 activations, live scales and shared FMA bindings."""

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


def _inputs(activation, index=0, top_k=3):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    torch.manual_seed(4800 + index)
    t, h, i, e, k = (
        (1, 2, 4, 8)[index % 4],
        (128, 256, 384)[index % 3],
        (32, 96, 160)[index % 3],
        max(8, top_k),
        top_k,
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * 0.3
    w1 = (
        torch.randn(e, (2 if activation.is_gated else 1) * i, h, device="cuda") * 0.3
    ).to(torch.float8_e4m3fn)
    w2 = (torch.randn(e, h, i, device="cuda") * 0.3).to(torch.float8_e4m3fn)
    s1, s2 = (torch.rand(e, device="cuda") * 0.1 + 0.05 for _ in range(2))
    view = FrostLowLatencyConfig.prepare_fp8_weight_only_weights(
        w1, s1, w2, s2, activation=activation
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
    scores = torch.randn(t, k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.FP8PerTensor, activation=QuantFormat.BF16),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    return cfg, x, ids, scores, pack, view


def _reference(x, ids, scores, view, activation, packed):
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    rows = []
    for token in range(x.shape[0]):
        for expert in ids[token].tolist():
            if not 0 <= expert < w1.shape[0]:
                rows.append(
                    torch.zeros(x.shape[1], device=x.device, dtype=torch.bfloat16)
                )
                continue
            values = (
                (
                    (w1[expert].double() @ x[token].double()).float()
                    * view["fc1_dequant_scale"][expert]
                )
                .bfloat16()
                .float()
            )
            up, gate = values.chunk(2) if activation.is_gated else (values, None)
            mid = _activation(up, gate, activation).bfloat16()
            rows.append(
                (
                    (w2[expert].double() @ mid.double()).float()
                    * view["fc2_dequant_scale"][expert]
                ).bfloat16()
            )
    raw = torch.stack(rows)
    effective = scores.bfloat16().float() if packed else scores.float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        result = (
            result
            + raw.view(*ids.shape, -1)[:, slot].float() * effective[:, slot, None]
        )
    return raw, result.bfloat16(), effective


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_w8a16_activations_and_live_graph_scales(index, activation):
    cfg, x, ids, scores, pack, view = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    captures = []
    for tactic in (("ffma", 1, 4), ("ffma", 4, 8)):
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        captures.append((graph, out))
    for step in range(3):
        x.copy_(torch.randn_like(x) * 0.3)
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
        scores.copy_(torch.randn_like(scores) * 0.2)
        view["fc1_dequant_scale"].mul_(1.2)
        view["fc2_dequant_scale"].mul_(0.85)
        # A W8A8 multiplier must have no effect on the weight-only path.
        view["fc2_act_quant_scale"].fill_(float("nan"))
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        for graph, out in captures:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_w8a16_public_output_contract(packed, finalize):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 1)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    out = MoELayer(cfg)(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), pack
    )
    raw, expected, effective = _reference(x, ids, scores, view, cfg.activation, packed)
    if finalize:
        _check(out, expected)
    else:
        _check(out[0], raw)
        torch.testing.assert_close(out[1].float(), effective, atol=0, rtol=0)
        torch.testing.assert_close(
            out[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
        )


def test_w8a16_preparation_preserves_bytes_and_padding():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    e, h, i = 2, 128, 32
    w1 = torch.randn(e, 2 * i, h, device="cuda").to(torch.float8_e4m3fn)
    w2 = torch.randn(e, h, i, device="cuda").to(torch.float8_e4m3fn)
    s1 = torch.tensor([0.05, 0.1], device="cuda")
    s2 = s1 * 2
    saved_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        view = FrostLowLatencyConfig.prepare_fp8_weight_only_weights(w1, s1, w2, s2)
    finally:
        torch.set_default_dtype(saved_dtype)
    assert view["fc2_act_quant_scale"].dtype is torch.float32
    padded = view["fc1_expert_weights"].view(torch.uint8).reshape(e, 2, 128, h)
    torch.testing.assert_close(
        padded[:, :, :i], w1.view(torch.uint8).reshape(e, 2, i, h), atol=0, rtol=0
    )
    assert not padded[:, :, i:].any()
    torch.testing.assert_close(
        view["fc2_expert_weights"][:, :, :i].view(torch.uint8),
        w2.view(torch.uint8),
        atol=0,
        rtol=0,
    )
    assert not view["fc2_expert_weights"][:, :, i:].view(torch.uint8).any()
    torch.testing.assert_close(view["fc1_dequant_scale"], s1, atol=0, rtol=0)
    torch.testing.assert_close(view["fc2_dequant_scale"], s2, atol=0, rtol=0)
    with pytest.raises(ValueError, match="weight scales"):
        FrostLowLatencyConfig.prepare_fp8_weight_only_weights(w1, s1.double(), w2, s2)


def test_w8a16_rejects_wrong_input_dtype_and_scale():
    cfg, x, ids, scores, pack, _ = _inputs(SwiGLU())
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match="aligned contiguous"):
        runner.pack_inputs(
            MoEActivationPack(x.to(torch.float8_e4m3fn), None, ids, scores), pack
        )
    with pytest.raises(ValueError, match="input scale"):
        runner.pack_inputs(
            MoEActivationPack(x, torch.ones(1, device=x.device), ids, scores), pack
        )


def test_w8a16_cache_distinguishes_w8a8_and_fresh_storage(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp8_per_tensor
    from tests.moe.test_frost_low_latency_fp8_per_tensor import _inputs as fp8_inputs

    # Same geometry and tactic must not alias W8A8's compiled input ABI.
    for setup in (fp8_inputs, _inputs):
        cfg, x, ids, scores, pack, view = setup(SwiGLU(), 1)
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        runner.forward(inputs, ("ffma", 2, 4))
    cloned = {name: value.clone() for name, value in view.items()}
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", cloned)
    x, ids, scores = x.clone(), ids.clone(), scores.clone()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)

    def forbidden(*args, **kwargs):
        raise AssertionError("warmed W8A16 must reuse executable metadata")

    monkeypatch.setattr(fp8_per_tensor.cute, "compile", forbidden)
    monkeypatch.setattr(fp8_per_tensor, "from_dlpack", forbidden)
    result = runner.forward(inputs, ("ffma", 2, 4))
    _, expected, _ = _reference(x, ids, scores, cloned, cfg.activation, True)
    _check(result, expected)


@pytest.mark.parametrize("first,second", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_w8a16_max_tokens_and_routes(first, second):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 3, top_k=32)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    output = runner.forward(inputs, ("ffma", first, second))
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    _check(output, expected)
