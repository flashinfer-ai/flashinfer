# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""BF16 generalized bindings: activation semantics, padding and graph ownership."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    GELU,
    GeGLU,
    GeGLUTanh,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    PowLU,
    ReLU,
    ReLU2,
    RoutingConfig,
    SiLU,
    SiTU,
    SwiGLU,
    SwiGLUStep,
    TrtllmBf16Config,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _assert_near, _require_gpu

ACTIVATIONS = [
    SwiGLU(),
    SwiGLU(alpha=1.702, beta=1.0, limit=3.0),
    SiTU(),
    GeGLU(),
    GeGLUTanh(),
    SwiGLUStep(),
    SwiGLUStep(limit=2.0),
    GELU(),
    ReLU(),
    ReLU2(),
    SiLU(),
    Identity(),
    PowLU(),
    PowLU(m=1.5, limit=2.0),
    PowLU(m=0.125),
    PowLU(m=3.0, limit=1.0),
]


def _activation(up, gate, activation):
    if isinstance(activation, PowLU):
        from tests.moe.utils import powlu_ref

        activated = powlu_ref(gate, activation.m)
        if activation.limit is not None:
            activated = activated.clamp_max(activation.limit)
            up = up.clamp(-activation.limit, activation.limit)
        return activated * up
    if isinstance(activation, SwiGLU):
        gate = gate.clamp_max(activation.limit)
        up = up.clamp(-activation.limit, activation.limit)
        return gate * torch.sigmoid(activation.alpha * gate) * (up + activation.beta)
    if isinstance(activation, SiTU):
        return (4 * torch.tanh(gate / 4) * torch.sigmoid(gate)) * (
            25 * torch.tanh(up / 25)
        )
    if isinstance(activation, SwiGLUStep):
        return torch.nn.functional.silu(gate).clamp_max(activation.limit) * up.clamp(
            -activation.limit, activation.limit
        )
    if isinstance(activation, (GeGLU, GeGLUTanh)):
        return up * torch.nn.functional.gelu(
            gate, approximate="tanh" if isinstance(activation, GeGLUTanh) else "none"
        )
    if isinstance(activation, GELU):
        return torch.nn.functional.gelu(up)
    if isinstance(activation, (ReLU, ReLU2)):
        return up.relu().square() if isinstance(activation, ReLU2) else up.relu()
    if isinstance(activation, SiLU):
        return torch.nn.functional.silu(up)
    assert isinstance(activation, Identity)
    return up


def _reference(x, w1, w2, ids, scores, activation, packed, output_dtype=torch.bfloat16):
    rows = []
    for token in range(ids.shape[0]):
        for expert in ids[token].tolist():
            value = (w1[expert].double() @ x[token].double()).float()
            if activation.is_gated:
                up, gate = value.chunk(2)
            else:
                up, gate = value, torch.zeros_like(value)
            mid = _activation(up, gate, activation).to(x.dtype)
            rows.append((w2[expert].double() @ mid.double()).to(output_dtype))
    raw = torch.stack(rows)
    effective_scores = scores.bfloat16().float() if packed else scores.float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        weighted = raw[slot :: ids.shape[1]].float() * effective_scores[:, slot, None]
        result = result + weighted  # Separate FP32 multiply/add, slot order.
    return raw, result.to(output_dtype), effective_scores


def _inputs(
    activation,
    tokens=2,
    hidden=256,
    intermediate=96,
    topk=3,
    experts=None,
    dtype=torch.bfloat16,
    output_dtype=torch.bfloat16,
):
    _require_gpu(allow_sm120=True, allow_sm90=True, allow_sm80=True)
    torch.manual_seed(5103 + tokens)
    e = max(5, topk) if experts is None else experts
    x = torch.randn(tokens, hidden, device="cuda").to(dtype)
    w1 = (
        torch.randn(
            e, (2 if activation.is_gated else 1) * intermediate, hidden, device="cuda"
        )
        * 0.08
    ).to(dtype)
    w2 = (torch.randn(e, hidden, intermediate, device="cuda") * 0.08).to(dtype)
    ids = (
        torch.arange(tokens * topk, device="cuda", dtype=torch.int32).reshape(
            tokens, topk
        )
        % e
    )
    ids[:, -1] = ids[:, 0]  # Repeated experts are legal, scores can be signed.
    scores = torch.randn(tokens, topk, device="cuda") * 0.3
    pack = MoEWeightPack()
    prepare = (
        FrostLowLatencyConfig.prepare_bf16_weights
        if dtype == torch.bfloat16
        else FrostLowLatencyConfig.prepare_fp16_weights
    )
    pack.prepare_for(
        "frost_low_latency",
        prepare(w1, w2, activation=activation),
    )
    cfg = MoEConfig(
        RoutingConfig(e, topk),
        QuantConfig(
            weight=QuantFormat.BF16 if dtype == torch.bfloat16 else QuantFormat.FP16,
            activation=(
                QuantFormat.BF16 if dtype == torch.bfloat16 else QuantFormat.FP16
            ),
            output=(
                QuantFormat.BF16 if output_dtype == torch.bfloat16 else QuantFormat.FP16
            ),
        ),
        ExpertConfig(intermediate),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        execution=ExecutionConfig(tune_max_num_tokens=tokens),
    )
    return cfg, x, w1, w2, ids, scores, pack


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_bf16_activation_tactics_and_changing_graph_inputs(index, activation):
    tokens = (1, 2, 4, 8)[index % 4]
    cfg, x, w1, w2, ids, scores, pack = _inputs(
        activation,
        tokens=tokens,
        hidden=(128, 256, 384)[index % 3],
        intermediate=(32, 96, 160, 224)[index % 4],
        topk=1 if index % 3 == 0 else 3,
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    act = MoEActivationPack(x, None, ids, scores)
    buffers = runner.pack_inputs(act, pack)
    graphs = []
    for tactic in runner.get_valid_tactics(buffers, None):
        runner.forward(buffers, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(buffers, tactic).clone()
        graphs.append((graph, out))
    for step in range(3):
        x.copy_(torch.randn_like(x) * (1 + step))
        ids.copy_((ids + 1) % w1.shape[0])
        scores.copy_(torch.randn_like(scores) * 0.3)
        _, expected, _ = _reference(x, w1, w2, ids, scores, activation, True)
        for graph, out in graphs:
            graph.replay()
            _assert_near(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("score_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("down_columns", [1, 2, 4, 8])
def test_bf16_output_contract_and_independent_call_buffers(
    packed, finalize, score_dtype, down_columns, monkeypatch
):
    cfg, x, w1, w2, ids, scores, pack = _inputs(SwiGLU(), topk=32)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    scores = scores.to(score_dtype)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    one = MoEActivationPack(x, None, ids, scores, routing_input_mode=mode)
    other_x = -x.clone()
    two = MoEActivationPack(other_x, None, ids, scores, routing_input_mode=mode)
    a, b = runner.pack_inputs(one, pack), runner.pack_inputs(two, pack)
    assert all(a[i].data_ptr() != b[i].data_ptr() for i in (5, 6, 7))
    tactic = ("ffma", 1, down_columns)
    runner.forward(a, tactic)
    from cutlass import cute

    def unexpected_compile(*args, **kwargs):
        raise AssertionError("same metadata with different pointers recompiled")

    monkeypatch.setattr(cute, "compile", unexpected_compile)
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    stream_a.wait_stream(torch.cuda.current_stream())
    stream_b.wait_stream(torch.cuda.current_stream())
    graph_a, graph_b = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph_a, stream=stream_a):
        first = runner.forward(a, tactic)
    with torch.cuda.graph(graph_b, stream=stream_b):
        second = runner.forward(b, tactic)
    with torch.cuda.stream(stream_a):
        graph_a.replay()
    with torch.cuda.stream(stream_b):
        graph_b.replay()
    torch.cuda.current_stream().wait_stream(stream_a)
    torch.cuda.current_stream().wait_stream(stream_b)
    for value, inp in ((first, x), (second, other_x)):
        raw, expected, effective = _reference(
            inp, w1, w2, ids, scores, cfg.activation, packed
        )
        if finalize:
            _assert_near(value, expected)
        else:
            _assert_near(value[0], raw)
            torch.testing.assert_close(value[1].float(), effective, rtol=0, atol=0)
            torch.testing.assert_close(
                value[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
            )


def test_bf16_public_layer_and_metadata_rejections():
    cfg, x, _, _, ids, scores, pack = _inputs(ReLU2())
    layer = MoELayer(cfg)
    output = layer(MoEActivationPack(x, None, ids, scores), pack)
    assert output.shape == x.shape and torch.isfinite(output).all()
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match="scales"):
        runner.pack_inputs(MoEActivationPack(x, torch.ones_like(x), ids, scores), pack)
    with pytest.raises(ValueError, match="BF16"):
        runner.pack_inputs(MoEActivationPack(x.float(), None, ids, scores), pack)
    unsupported = replace(cfg, activation=SiTU(gate_scale=2.0))
    with pytest.raises(NotImplementedError, match="default parameters"):
        FrostLowLatencyRunner(unsupported, x.device).check_support()
    with pytest.raises(NotImplementedError, match="top_k"):
        FrostLowLatencyRunner(
            replace(cfg, routing=RoutingConfig(64, 33)), x.device
        ).check_support()


def test_bf16_real_autotuning_and_larger_input_fallback():
    _require_gpu()  # This test's TRT-LLM fallback is SM100-family-only.
    cfg, x, w1, w2, ids, scores, pack = _inputs(
        SwiGLU(), tokens=1, hidden=2048, intermediate=1408, topk=6, experts=64
    )
    # Native TRT routing stores one slot per (token, expert); duplicated slots
    # overwrite each other. Cross-backend inputs must satisfy this unique-ID
    # contract. Frost's broader repeated-slot behavior is tested separately.
    ids.copy_(torch.arange(6, device=x.device, dtype=torch.int32)[None, :])
    cfg = replace(cfg, backend=BackendOptions((FrostLowLatencyConfig(),)))
    layer = MoELayer(cfg)
    with autotune(True):
        tuned = layer(MoEActivationPack(x, None, ids, scores), pack)
    assert next(iter(layer._winners.values()))[1] in [
        ("ffma", first, second) for first in (1, 2, 4, 8) for second in (1, 2, 4, 8)
    ]
    _, expected, _ = _reference(x, w1, w2, ids, scores, cfg.activation, True)
    _assert_near(tuned, expected)

    pack.prepare_for(
        "trtllm_bf16_routed",
        TrtllmBf16Config.prepare_weights(
            w1,
            w2,
            num_local_experts=64,
            hidden_size=2048,
            intermediate_size=1408,
            activation=cfg.activation,
        ),
    )
    fallback = MoELayer(
        replace(
            cfg,
            backend=BackendOptions((FrostLowLatencyConfig(), TrtllmBf16Config())),
            execution=replace(cfg.execution, tune_max_num_tokens=4),
        )
    )
    x, ids, scores = x.repeat(4, 1), ids.repeat(4, 1), scores.repeat(4, 1)
    act = MoEActivationPack(x, None, ids, scores)
    fallback(act, pack)
    assert next(iter(fallback._winners.values()))[0].backend_key == "trtllm_bf16_routed"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = fallback(act, pack)
    x.mul_(0.5)
    ids.add_(7)
    graph.replay()
    _, expected, _ = _reference(x, w1, w2, ids, scores, cfg.activation, True)
    _assert_near(result, expected)
