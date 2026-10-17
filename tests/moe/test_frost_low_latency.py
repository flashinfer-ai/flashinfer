# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-token MX MoE: independent quantized reference and changing graph inputs."""

from dataclasses import replace
from importlib.metadata import version
from packaging.version import Version

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
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


def _quant(value):
    grouped = value.float().reshape(*value.shape[:-1], -1, 32)
    exponent = (
        torch.ceil(
            torch.log2(
                grouped.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
                / 448.0
            )
        )
        + 127
    ).clamp(0, 254)
    scale = torch.exp2(exponent - 127)
    q = (grouped / scale[..., None]).to(torch.float8_e4m3fn)
    return (
        (q.float() * scale[..., None]).reshape(value.shape),
        q.reshape(value.shape),
        exponent.byte(),
    )


def _decode(w, sf):
    if w.dtype == torch.uint8:
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=w.device,
            dtype=torch.float64,
        )
        values = torch.stack(
            (lut[(w & 15).long()], lut[(w >> 4).long()]), dim=-1
        ).reshape(*w.shape[:-1], -1)
    else:
        values = w.double()
    return (
        values.reshape(*values.shape[:-1], -1, 32)
        * torch.exp2(sf.double() - 127)[..., None]
    ).reshape(values.shape)


def _reference(x, w1, s1, w2, s2, ids, weights, activation):
    routes = []
    for token in range(ids.shape[0]):
        for expert in ids[token].tolist():
            gu = (x[token].double() @ _decode(w1[expert], s1[expert]).T).float()
            up, gate = gu.chunk(2)
            gate = gate.clamp_max(activation.limit)
            up = up.clamp(-activation.limit, activation.limit)
            mid = (
                gate
                / (1 + torch.exp(-gate * activation.alpha))
                * (up + activation.beta)
            )
            dq, _, _ = _quant(mid[None, :])
            routes.append((dq.double() @ _decode(w2[expert], s2[expert]).T).bfloat16())
    raw = torch.cat(routes)
    value = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        # FP64 models one correctly rounded FP32 FMA, in slot order.
        value = (
            raw[slot :: ids.shape[1]].double() * weights[:, slot, None].double()
            + value.double()
        ).float()
    return raw, value.bfloat16()


def _assert_near(actual, expected):
    diff = actual.double() - expected.double()
    assert torch.isfinite(actual).all()
    assert float(diff.norm() / expected.double().norm().clamp_min(1e-30)) < 0.01
    assert (
        float(diff.abs().max() / expected.double().abs().max().clamp_min(1e-30)) < 0.03
    )


def _require_gpu():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    pytest.importorskip("cutlass")
    if Version(version("nvidia-cutlass-dsl")) < Version("4.8"):
        pytest.skip("requires nvidia-cutlass-dsl >= 4.8")


@pytest.mark.parametrize("weight_format", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize(
    "tokens,hidden,intermediate,topk",
    [(1, 1024, 512, 4), (2, 2048, 576, 6), (8, 1024, 256, 4)],
)
def test_frost_graph_reference(weight_format, tokens, hidden, intermediate, topk):
    _require_gpu()
    torch.manual_seed(317)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = 8

    def weight(rows, cols):
        if weight_format == QuantFormat.MXFP4:
            return (
                torch.randint(
                    0, 256, (experts, rows, cols // 2), device=device, dtype=torch.uint8
                ),
                torch.randint(
                    118,
                    123,
                    (experts, rows, cols // 32),
                    device=device,
                    dtype=torch.uint8,
                ),
            )
        _, q, s = _quant(
            (torch.randn(experts, rows, cols, device=device) / (cols**0.5)).bfloat16()
        )
        return q, s

    w1, s1 = weight(2 * intermediate, hidden)
    w2, s2 = weight(hidden, intermediate)
    view = FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    config = MoEConfig(
        routing=RoutingConfig(experts, topk),
        quant=QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate),
        activation=SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        execution=ExecutionConfig(enable_pdl=tokens != 2),
    )
    x = torch.randn(tokens, hidden, device=device).bfloat16()
    _, q, sf = _quant(x)
    ids = (
        torch.arange(tokens * topk, device=device, dtype=torch.int32) % experts
    ).reshape(tokens, topk)
    rw = torch.rand(tokens, topk, device=device)
    # Exercise both routing-weight contracts, raw output and padded I=576.
    for mode in (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ):
        act = MoEActivationPack(q, sf, ids, rw, routing_input_mode=mode)
        for finalize in (True, False):
            cfg = replace(config, finalize=MoEFinalizeConfig(do_finalize=finalize))
            runner = FrostLowLatencyRunner(cfg, device)
            runner.check_support()
            runner.build()
            inputs = runner.pack_inputs(act, pack)
            for tile in (128, 256):
                runner.forward(inputs, tactic=tile)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = runner.forward(inputs, tactic=tile)
                for step in range(3):
                    dq, qn, sn = _quant(
                        (torch.randn_like(x.float()) * (1 + step)).bfloat16()
                    )
                    q.copy_(qn)
                    sf.copy_(sn)
                    ids.copy_(
                        (
                            (
                                torch.arange(
                                    tokens * topk, device=device, dtype=torch.int32
                                )
                                + step * 3
                                + experts
                                - 1
                            )
                            % experts
                        ).reshape_as(ids)
                    )
                    # The per-call packed weight tensor is the captured input;
                    # a caller mutating the original pack should call pack_inputs again.
                    inputs[3].copy_(torch.rand_like(inputs[3]))
                    graph.replay()
                    torch.cuda.synchronize()
                    raw, ref = _reference(
                        dq, w1, s1, w2, s2, ids, inputs[3], cfg.activation
                    )
                    _assert_near(
                        output if finalize else output[0], ref if finalize else raw
                    )
                    if not finalize:
                        torch.testing.assert_close(
                            output[2],
                            torch.arange(
                                tokens * topk, device=device, dtype=torch.int32
                            ),
                        )
    # Actual unified dispatcher, not just the direct runner.
    layer = MoELayer(config, device=device)
    act = MoEActivationPack(q, sf, ids, rw)
    result = layer(act, pack)
    _, expected = _reference(dq, w1, s1, w2, s2, ids, rw.bfloat16(), config.activation)
    _assert_near(result, expected)
    # Reuse the compiled geometry with different activation and weight storage.
    # Negating only Down must negate the output; a cached pointer fails this.
    negative = {name: value.clone() for name, value in view.items()}
    negative["w2_weight"].view(torch.uint8).bitwise_xor_(
        0x88 if weight_format == QuantFormat.MXFP4 else 0x80
    )
    other_weights = MoEWeightPack()
    other_weights.prepare_for("frost_low_latency", negative)
    other_act = MoEActivationPack(q.clone(), sf.clone(), ids.clone(), rw.clone())
    negated = layer(other_act, other_weights)
    torch.testing.assert_close(negated, -result, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        original_output = layer(act, pack)
        negative_output = layer(other_act, other_weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(negative_output, -original_output, rtol=0, atol=0)


def test_frost_capability_and_metadata_rejections():
    with pytest.raises(ValueError):
        FrostLowLatencyConfig(max_num_tokens=9)
    _require_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    config = MoEConfig(
        RoutingConfig(8, 2),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(256),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(config, device)
    runner.check_support()
    runner.build()
    act = MoEActivationPack(
        torch.zeros(3, 1024, device=device, dtype=torch.float8_e4m3fn),
        torch.zeros(3, 32, device=device, dtype=torch.uint8),
        torch.zeros(3, 2, device=device, dtype=torch.int32),
        torch.ones(3, 2, device=device),
    )
    assert not runner._accepts_shape(act)
    ep = FrostLowLatencyRunner(
        replace(config, experts=ExpertConfig(256, local_num_experts=4)), device
    )
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        ep.check_support()


def test_frost_declines_old_compiler(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency import runner as module

    monkeypatch.setattr(module, "version", lambda name: "4.7.1")
    config = MoEConfig(
        RoutingConfig(8, 2),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(256),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8"):
        runner.check_support()


def test_frost_binds_noncurrent_device():
    _require_gpu()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs to check a noncurrent device")
    previous = torch.cuda.current_device()
    other = (previous + 1) % torch.cuda.device_count()
    if torch.cuda.get_device_capability(other) not in ((10, 0), (10, 3)):
        pytest.skip("second device must be SM100/SM103")
    device = torch.device("cuda", other)
    e, h, i, k = 2, 128, 128, 2
    w1 = torch.full((e, 2 * i, h // 2), 0x11, device=device, dtype=torch.uint8)
    w2 = torch.full((e, h, i // 2), 0x11, device=device, dtype=torch.uint8)
    s1 = torch.full((e, 2 * i, h // 32), 123, device=device, dtype=torch.uint8)
    s2 = torch.full((e, h, i // 32), 123, device=device, dtype=torch.uint8)
    dq, q, sf = _quant(torch.randn(1, h, device=device).bfloat16())
    ids = torch.arange(k, device=device, dtype=torch.int32)[None, :]
    rw = torch.full((1, k), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "frost_low_latency", FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    )
    config = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        activation=SwiGLU(alpha=1.0, beta=0.0, limit=10.0),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    layer = MoELayer(config, device=device)
    result = layer(MoEActivationPack(q, sf, ids, rw), weights)
    assert torch.cuda.current_device() == previous
    _, reference = _reference(dq, w1, s1, w2, s2, ids, rw.bfloat16(), config.activation)
    _assert_near(result, reference)
