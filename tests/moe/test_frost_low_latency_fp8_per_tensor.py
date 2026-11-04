# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Static FP8 intermediate scaling, shared FMA bodies and graph rebinding."""

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


def _inputs(activation, index=0):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    torch.manual_seed(3700 + index)
    t, h, i, e, k = (
        (1, 2, 4, 8)[index % 4],
        (128, 256, 384)[index % 3],
        (32, 96, 160)[index % 3],
        8,
        3,
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * 0.3
    w1 = (
        torch.randn(
            e,
            (2 if activation.is_gated else 1) * i,
            h,
            device="cuda",
            dtype=torch.bfloat16,
        )
        * 0.05
    )
    w2 = torch.randn(e, h, i, device="cuda", dtype=torch.bfloat16) * 0.05
    view = FrostLowLatencyConfig.prepare_fp8_per_tensor_weights(
        w1,
        w2,
        hidden_states_scale_global=32.0,
        intermediate_scale_global=128.0,
        num_local_experts=e,
        hidden_size=h,
        intermediate_size=i,
        activation=activation,
    )
    x, _ = FrostLowLatencyConfig.prepare_fp8_per_tensor_activations(
        x, hidden_states_scale_global=32.0
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
    scores = torch.randn(t, k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
        ),
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
            values = (w1[expert].double() @ x[token].double()).float() * view[
                "fc1_dequant_scale"
            ][expert]
            values = values.bfloat16().float()
            if activation.is_gated:
                up, gate = values.chunk(2)
            else:
                up, gate = values, None
            mid = (
                (_activation(up, gate, activation) * view["fc2_act_quant_scale"])
                .clamp(-448, 448)
                .to(torch.float8_e4m3fn)
            )
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


def _check(actual, expected):
    assert torch.isfinite(actual).all()
    error = actual.double() - expected.double()
    assert float(error.norm() / expected.double().norm().clamp_min(1e-30)) < 0.015
    assert (
        float(error.abs().max() / expected.double().abs().max().clamp_min(1e-30)) < 0.06
    )


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_fp8_per_tensor_activation_and_dynamic_scales(index, activation):
    cfg, x, ids, scores, pack, view = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    captured = []
    for tactic in (("ffma", 1, 4), ("ffma", 4, 8)):
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        captured.append((graph, out))
    for step, scale in enumerate((128.0, 8.0, 1024.0)):
        x.copy_(
            (torch.randn_like(x, dtype=torch.float32) * 12.8 * (step + 1)).to(x.dtype)
        )
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
        scores.copy_(torch.randn_like(scores) * 0.2)
        view["fc1_dequant_scale"].mul_(1.2)
        view["fc2_dequant_scale"].mul_(0.85)
        view["fc2_act_quant_scale"].fill_(scale)
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        for graph, out in captured:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_fp8_per_tensor_public_output_contract(packed, finalize):
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


def test_fp8_per_tensor_rejects_scale_metadata():
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU())
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match="input scale"):
        runner.pack_inputs(
            MoEActivationPack(x, torch.ones(1, device=x.device), ids, scores), pack
        )
    view["fc1_dequant_scale"] = view["fc1_dequant_scale"].double()
    with pytest.raises(ValueError, match="fc1_dequant_scale"):
        runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)


def test_fp8_per_tensor_padding_honors_destination_device():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    torch.manual_seed(3799)
    w1 = torch.randn(2, 64, 128, dtype=torch.bfloat16)
    w2 = torch.randn(2, 128, 32, dtype=torch.bfloat16)
    kwargs = dict(
        hidden_states_scale_global=32.0,
        intermediate_scale_global=128.0,
        num_local_experts=2,
        hidden_size=128,
        intermediate_size=32,
        device=torch.device("cuda", torch.cuda.current_device()),
    )
    prepare = FrostLowLatencyConfig.prepare_fp8_per_tensor_weights
    moved = prepare(w1, w2, **kwargs)
    resident = prepare(w1.cuda(), w2.cuda(), **kwargs)
    for name, value in moved.items():
        assert value.device == kwargs["device"], name
        if value.dtype is torch.float8_e4m3fn:
            torch.testing.assert_close(
                value.view(torch.uint8),
                resident[name].view(torch.uint8),
                atol=0,
                rtol=0,
            )
        else:
            torch.testing.assert_close(value, resident[name], atol=0, rtol=0)
