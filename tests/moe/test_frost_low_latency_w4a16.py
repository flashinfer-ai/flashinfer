# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVFP4 weight-only BF16 paths: exact payloads, live scales and shared FMA bindings."""

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
    branches = 2 if activation.is_gated else 1
    w1 = torch.randint(
        0, 256, (e, branches * i, h // 2), device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = (
        torch.rand(e, branches * i, h // 16, device="cuda") * 0.015625 + 0.00390625
    ).to(torch.float8_e4m3fn)
    s2 = (torch.rand(e, h, i // 16, device="cuda") * 0.015625 + 0.00390625).to(
        torch.float8_e4m3fn
    )
    a1 = torch.rand((e, branches) if index % 2 else (e,), device="cuda") + 0.5
    a2 = torch.rand(e, device="cuda") + 0.5
    view = FrostLowLatencyConfig.prepare_nvfp4_weight_only_weights(
        w1, s1, a1, w2, s2, a2, activation=activation
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
    scores = torch.randn(t, k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.NVFP4, activation=QuantFormat.BF16),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    return cfg, x, ids, scores, pack, view


def _decode(q, sf, scale_dtype=torch.float8_e4m3fn, scale_block=16):
    code = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
    lut = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=q.device)
    return (
        lut[code & 7]
        * (1 - 2 * ((code & 8) != 0).float())
        * sf.view(scale_dtype).float().repeat_interleave(scale_block, -1)
    ).double()


def _reference(
    x,
    ids,
    scores,
    view,
    activation,
    packed,
    scale_dtype=torch.float8_e4m3fn,
    scale_block=16,
):
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
                _decode(
                    w1[expert],
                    view["fc1_block_scales"][expert],
                    scale_dtype,
                    scale_block,
                )
                @ x[token].double()
            ).float()
            branches = 2 if activation.is_gated else 1
            values = (
                (
                    values.reshape(branches, -1)
                    * view["fc1_dequant_scale"][expert, :, None]
                )
                .reshape(-1)
                .bfloat16()
                .float()
            )
            up, gate = values.chunk(2) if activation.is_gated else (values, None)
            mid = _activation(up, gate, activation).bfloat16()
            rows.append(
                (
                    (
                        _decode(
                            w2[expert],
                            view["fc2_block_scales"][expert],
                            scale_dtype,
                            scale_block,
                        )
                        @ mid.double()
                    ).float()
                    * view["fc2_dequant_scale"][expert]
                ).bfloat16()
            )
    raw = torch.stack(rows)
    effective = scores.bfloat16().float() if packed else scores.float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        valid = (ids[:, slot] >= 0) & (ids[:, slot] < w1.shape[0])
        weight = torch.where(valid, effective[:, slot], 0.0)
        result = result + raw.view(*ids.shape, -1)[:, slot].float() * weight[:, None]
    return raw, result.bfloat16(), effective


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_w4a16_activations_and_live_graph_scales(index, activation):
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
        for name in ("fc1_block_scales", "fc2_block_scales"):
            sf = view[name].view(torch.float8_e4m3fn)
            sf.copy_((sf.float() * 1.125).to(torch.float8_e4m3fn))
        if step == 2:
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        for graph, out in captures:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_w4a16_public_output_contract(packed, finalize):
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


def test_w4a16_preparation_preserves_bytes_and_padding():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    e, h, i = 2, 128, 32
    w1 = torch.randint(0, 256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = (torch.rand(e, 2 * i, h // 16, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    s2 = (torch.rand(e, h, i // 16, device="cuda") * 0.125).to(torch.float8_e4m3fn)
    a1 = torch.tensor([[0.5, 1.5], [2.0, 0.75]], device="cuda")
    a2 = torch.tensor([1.25, 0.5], device="cuda")
    view = FrostLowLatencyConfig.prepare_nvfp4_weight_only_weights(
        w1, s1, a1, w2, s2, a2
    )
    first = view["fc1_expert_weights"].reshape(e, 2, 128, h // 2)
    scale = view["fc1_block_scales"].reshape(e, 2, 128, h // 16)
    torch.testing.assert_close(
        first[:, :, :i], w1.reshape(e, 2, i, h // 2), atol=0, rtol=0
    )
    torch.testing.assert_close(
        scale[:, :, :i], s1.view(torch.uint8).reshape(e, 2, i, h // 16), atol=0, rtol=0
    )
    assert not first[:, :, i:].any() and not scale[:, :, i:].any()
    torch.testing.assert_close(
        view["fc2_expert_weights"][:, :, : i // 2], w2, atol=0, rtol=0
    )
    torch.testing.assert_close(
        view["fc2_block_scales"][:, :, : i // 16], s2.view(torch.uint8), atol=0, rtol=0
    )
    assert not view["fc2_expert_weights"][:, :, i // 2 :].any()
    assert not view["fc2_block_scales"][:, :, i // 16 :].any()
    torch.testing.assert_close(view["fc1_dequant_scale"], a1, atol=0, rtol=0)
    torch.testing.assert_close(view["fc2_dequant_scale"], a2, atol=0, rtol=0)
    with pytest.raises(ValueError, match="FC1 global scales"):
        FrostLowLatencyConfig.prepare_nvfp4_weight_only_weights(
            w1, s1, a1.double(), w2, s2, a2
        )


def test_w4a16_rejects_wrong_input_dtype_and_scale():
    cfg, x, ids, scores, pack, _ = _inputs(SwiGLU())
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match="aligned contiguous"):
        runner.pack_inputs(
            MoEActivationPack(x.to(torch.float8_e4m3fn), None, ids, scores), pack
        )
    with pytest.raises(ValueError, match="activation scale"):
        runner.pack_inputs(
            MoEActivationPack(x, torch.ones(1, device=x.device), ids, scores), pack
        )


def test_w4a16_fresh_storage_reuses_metadata(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    runner.forward(inputs, ("ffma", 2, 4))
    cloned = {name: value.clone() for name, value in view.items()}
    cloned["fc1_dequant_scale"].mul_(0.9)
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", cloned)
    x, ids, scores = x.clone(), ids.clone(), scores.clone()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)

    def forbidden(*args, **kwargs):
        raise AssertionError("warm W4A16 must reuse executable metadata")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    result = runner.forward(inputs, ("ffma", 2, 4))
    _, expected, _ = _reference(x, ids, scores, cloned, cfg.activation, True)
    _check(result, expected)


@pytest.mark.parametrize("e4m3_max", [256, 448])
def test_w4a16_activation_recipe_env_preserves_checkpoint(monkeypatch, e4m3_max):
    cfg, x, ids, scores, pack, _ = _inputs(SwiGLU())
    act = MoEActivationPack(x, None, ids, scores)
    monkeypatch.delenv("FLASHINFER_NVFP4_4OVER6", raising=False)
    expected = MoELayer(cfg)(act, pack).clone()
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6", "1")
    monkeypatch.setenv("FLASHINFER_NVFP4_4OVER6_ERR_MODE", "MAE")
    monkeypatch.setenv(
        "FLASHINFER_NVFP4_4OVER6_E4M3_USE_256", "1" if e4m3_max == 256 else "0"
    )
    # This mode consumes checkpoint bytes and never invokes the activation
    # quantizer. A recipe must not rescale or requantize those weights.
    actual = MoELayer(cfg)(act, pack)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@pytest.mark.parametrize("first,second", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_w4a16_max_tokens_and_routes(first, second):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 3, top_k=32)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    output = runner.forward(inputs, ("ffma", first, second))
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    _check(output, expected)
