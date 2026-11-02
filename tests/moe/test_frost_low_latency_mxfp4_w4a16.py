# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXFP4/BF16 preserves E8M0 K32 scales across the shared FMA pipeline."""

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
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check
from tests.moe.test_frost_low_latency_w4a16 import _reference as _fp4_reference


def _reference(x, ids, scores, view, activation, packed):
    return _fp4_reference(
        x, ids, scores, view, activation, packed, torch.float8_e8m0fnu, 32
    )


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
    branches = 2 if activation.is_gated else 1
    w1 = torch.randint(
        0, 256, (e, branches * i, h // 2), device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = (
        torch.rand(e, branches * i, h // 32, device="cuda") * 0.015625 + 0.00390625
    ).to(torch.float8_e8m0fnu)
    s2 = (torch.rand(e, h, i // 32, device="cuda") * 0.015625 + 0.00390625).to(
        torch.float8_e8m0fnu
    )
    view = FrostLowLatencyConfig.prepare_mxfp4_weight_only_weights(
        w1, s1, w2, s2, activation=activation
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
    scores = torch.randn(t, k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.BF16),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    return cfg, x, ids, scores, pack, view


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_mxfp4_w4a16_activations_and_live_graph_scales(index, activation):
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
        # Updating one matrix's live scale must not change the other matrix.
        fc2_before = view["fc2_dequant_scale"].clone()
        view["fc1_dequant_scale"].mul_(1.2)
        torch.testing.assert_close(
            view["fc2_dequant_scale"], fc2_before, atol=0, rtol=0
        )
        fc1_before = view["fc1_dequant_scale"].clone()
        view["fc2_dequant_scale"].mul_(0.85)
        torch.testing.assert_close(
            view["fc1_dequant_scale"], fc1_before, atol=0, rtol=0
        )
        for name in ("fc1_block_scales", "fc2_block_scales"):
            sf = view[name].view(torch.float8_e8m0fnu)
            sf.copy_((sf.float() * 2.0).to(torch.float8_e8m0fnu))
        if step == 2:
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        for graph, out in captures:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_mxfp4_w4a16_public_output_contract(packed, finalize):
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


@pytest.mark.parametrize("first,second", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_mxfp4_w4a16_max_tokens_and_routes(first, second):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 3, top_k=32)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    output = runner.forward(inputs, ("ffma", first, second))
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    _check(output, expected)


@pytest.mark.parametrize("exponent", [-40, 16])
def test_mxfp4_w4a16_scale_range_outside_fp16(exponent):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU())
    view["fc1_block_scales"].view(torch.float8_e8m0fnu).copy_(
        torch.full_like(
            view["fc1_block_scales"], 2.0**exponent, dtype=torch.float32
        ).to(torch.float8_e8m0fnu)
    )
    x.mul_(2.0 ** (-exponent))
    out = MoELayer(cfg)(MoEActivationPack(x, None, ids, scores), pack)
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    _check(out, expected)


@pytest.mark.parametrize("scale_byte,input_exponent", [(0, 120), (254, -120)])
def test_mxfp4_w4a16_extreme_scale_finite_dot(scale_byte, input_exponent):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU())
    # 0.5 weights avoid overflowing the largest scale, while the smallest
    # scale creates FP32 subnormal products. Both complete dots remain finite.
    view["fc1_expert_weights"].fill_(0x11)
    view["fc1_block_scales"].fill_(scale_byte)
    x.fill_(2.0**input_exponent)
    output = MoELayer(cfg)(MoEActivationPack(x, None, ids, scores), pack)
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    assert torch.isfinite(expected).all() and expected.abs().max() > 0
    _check(output, expected)


def test_mxfp4_w4a16_preparation_preserves_payloads():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    e, h, i = 2, 128, 32
    w1 = torch.randint(0, 256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.randint(
        80, 160, (e, 2 * i, h // 32), device="cuda", dtype=torch.uint8
    ).view(torch.float8_e8m0fnu)
    s2 = torch.randint(80, 160, (e, h, i // 32), device="cuda", dtype=torch.uint8).view(
        torch.float8_e8m0fnu
    )
    view = FrostLowLatencyConfig.prepare_mxfp4_weight_only_weights(w1, s1, w2, s2)
    torch.testing.assert_close(
        view["fc1_expert_weights"].reshape(e, 2, 128, h // 2)[:, :, :i],
        w1.reshape(e, 2, i, h // 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        view["fc1_block_scales"].reshape(e, 2, 128, h // 32)[:, :, :i],
        s1.view(torch.uint8).reshape(e, 2, i, h // 32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        view["fc2_expert_weights"][:, :, : i // 2], w2, atol=0, rtol=0
    )
    torch.testing.assert_close(
        view["fc2_block_scales"][:, :, : i // 32], s2.view(torch.uint8), atol=0, rtol=0
    )
    assert not view["fc1_expert_weights"].reshape(e, 2, 128, h // 2)[:, :, i:].any()
    assert not view["fc2_expert_weights"][:, :, i // 2 :].any()
    with pytest.raises(ValueError):
        FrostLowLatencyConfig.prepare_mxfp4_weight_only_weights(
            w1, s1.view(torch.float8_e4m3fn), w2, s2
        )


def test_mxfp4_and_nvfp4_compiled_metadata_separation(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only
    from tests.moe.test_frost_low_latency_w4a16 import _inputs as nv_inputs

    calls = []
    for factory in (nv_inputs, _inputs):
        cfg, x, ids, scores, pack, view = factory(SwiGLU())
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        expected = runner.forward(args, ("ffma", 2, 4)).clone()
        cloned = {key: value.clone() for key, value in view.items()}
        fresh = MoEWeightPack()
        fresh.prepare_for("frost_low_latency", cloned)
        args = runner.pack_inputs(
            MoEActivationPack(x.clone(), None, ids.clone(), scores.clone()), fresh
        )
        calls.append((runner, args, expected))

    def forbidden(*args, **kwargs):
        raise AssertionError("warm metadata must not compile or retain tensor views")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    for runner, args, expected in reversed(calls):
        torch.testing.assert_close(
            runner.forward(args, ("ffma", 2, 4)), expected, atol=0, rtol=0
        )
