# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent BF16 cuDNN MoE semantics and live-routing capture replay."""

import math

import pytest
import torch
import torch.nn.functional as F

from flashinfer.autotuner import autotune
from flashinfer.fused_moe import (
    BackendOptions,
    CudnnMoeConfig,
    CudnnMoeRunner,
    ExecutionConfig,
    ExpertConfig,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    RoutingInputMode,
)
from flashinfer.grouped_mm import grouped_mm_bf16
from flashinfer.grouped_mm.cudnn.autotune import CudnnGroupedMmRunner


def _case(mode, tokens=17, experts=8, hidden=256, inter=256, backend=None):
    torch.manual_seed(821)
    x = torch.randn(tokens, hidden, device="cuda", dtype=torch.bfloat16) * 0.5
    w1 = torch.randn(
        experts, 2 * inter, hidden, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(hidden)
    w2 = torch.randn(
        experts, hidden, inter, device="cuda", dtype=torch.bfloat16
    ) / math.sqrt(inter)
    # Skewed, non-aligned group lengths with trailing empty experts.
    ids = torch.stack(
        (
            torch.zeros(tokens, device="cuda", dtype=torch.int32),
            torch.arange(tokens, device="cuda", dtype=torch.int32) % 3 + 1,
        ),
        dim=1,
    )
    scales = torch.softmax(torch.randn(tokens, 2, device="cuda"), -1)
    act = MoEActivationPack(x, None, ids, scales, routing_input_mode=mode)
    config = MoEConfig(
        routing=RoutingConfig(num_experts=experts, top_k=2),
        quant=QuantConfig(weight=QuantFormat.BF16, activation=QuantFormat.BF16),
        experts=ExpertConfig(intermediate_size=inter),
        backend=BackendOptions((backend or CudnnMoeConfig(),)),
        execution=ExecutionConfig(enable_pdl=False, tune_max_num_tokens=tokens),
    )
    weights = MoEWeightPack()
    weights.prepare_for(
        "cudnn",
        CudnnMoeConfig.prepare_weights(
            w1,
            w2,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size=inter,
        ),
    )
    return config, act, weights, w1, w2


def _reference(act, w1, w2):
    x = act.hidden_states_q.float()
    weights = act.topk_weights
    if act.routing_input_mode == RoutingInputMode.PackedPrecomputed:
        weights = weights.to(torch.bfloat16)
    result = torch.zeros_like(x)
    # Independent token/expert reference, not the adapter's permutation maps.
    for expert in range(w1.shape[0]):
        tokens, slots = torch.where(act.topk_ids == expert)
        if tokens.numel() == 0:
            continue
        up, gate = (x[tokens] @ w1[expert].float().T).chunk(2, -1)
        h = (F.silu(gate) * up).to(torch.bfloat16).float()
        y = (h @ w2[expert].float().T).to(torch.bfloat16).float()
        result.index_add_(0, tokens, y * weights[tokens, slots, None].float())
    return result.to(torch.bfloat16)


def _require_b200():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("SM100-family GPU required")


def test_cudnn_registration_and_support():
    from flashinfer.fused_moe.layer import _BACKEND_RUNNERS

    assert _BACKEND_RUNNERS[CudnnMoeConfig] is CudnnMoeRunner
    assert CudnnMoeConfig.supported(100)
    assert not CudnnMoeConfig.supported(120)


def test_cudnn_quantization_axes_are_checked_independently():
    from itertools import product

    from flashinfer.fused_moe.cudnn_fp8_backend import CudnnFp8PerTensorRunner

    for runner, supported in (
        (CudnnMoeRunner, QuantFormat.BF16),
        (CudnnFp8PerTensorRunner, QuantFormat.FP8PerTensor),
    ):
        for weight, activation, output in product(QuantFormat, repeat=3):
            config = QuantConfig(weight=weight, activation=activation, output=output)
            expected = weight == activation == supported and output == QuantFormat.BF16
            assert runner.supports_quant(config) == expected, (runner, config)


def test_packed_unpacked_inputs_keep_explicit_rounding_contract():
    _require_b200()
    from dataclasses import replace

    config, packed, weights, w1, w2 = _case(
        RoutingInputMode.PackedPrecomputed,
        backend=CudnnMoeConfig(use_native_routing=True),
    )
    layer = MoELayer(config)
    layer(packed, weights)
    runner = layer.runners[0]
    unpacked = replace(packed, routing_input_mode=RoutingInputMode.UnpackedPrecomputed)
    packed_inputs = runner.pack_inputs(packed, weights)
    unpacked_inputs = runner.pack_inputs(unpacked, weights)
    assert packed_inputs[2] is packed.topk_weights
    assert unpacked_inputs[2] is unpacked.topk_weights
    assert runner.get_cache_key_extras(packed_inputs) != runner.get_cache_key_extras(
        unpacked_inputs
    )
    for inputs, act in [
        (packed_inputs, packed),
        (unpacked_inputs, unpacked),
        (packed_inputs, packed),
    ]:
        actual = runner.forward(inputs)
        torch.testing.assert_close(
            actual, _reference(act, w1, w2), rtol=0.02, atol=0.02
        )


@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
def test_moe_live_inputs_and_routing_capture(mode):
    _require_b200()
    config, act, weights, w1, w2 = _case(mode)
    layer = MoELayer(config)
    actual = layer(act, weights)
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    assert layer.winner_backend == "cudnn"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    previous = actual.clone()
    act.hidden_states_q.neg_()
    act.topk_ids.add_(2).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    for state in layer.runners[0]._resources_cache.values():
        for name in ("routed", "intermediate", "projected", "output"):
            state[name].fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    assert not torch.equal(actual, previous)


def test_moe_autotune_and_weight_pack_interleave():
    _require_b200()
    config, act, weights, w1, w2 = _case(RoutingInputMode.UnpackedPrecomputed)
    layer = MoELayer(config)
    with autotune():
        actual = layer(act, weights)
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    runner = layer.runners[0]
    first = runner.pack_inputs(act, weights)
    second = MoEWeightPack()
    second.prepare_for(
        "cudnn",
        CudnnMoeConfig.prepare_weights(
            -w1, w2, num_local_experts=8, hidden_size=256, intermediate_size=256
        ),
    )
    other = runner.pack_inputs(act, second)
    torch.testing.assert_close(
        runner.forward(first), _reference(act, w1, w2), rtol=0.02, atol=0.02
    )
    torch.testing.assert_close(
        runner.forward(other), _reference(act, -w1, w2), rtol=0.02, atol=0.02
    )


def test_grouped_tactic_replay_autotune_and_offset_changes():
    _require_b200()
    torch.manual_seed(819)
    a = torch.randn(37, 256, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(8, 256, 256, device="cuda", dtype=torch.bfloat16)
    offsets = torch.tensor(
        [0, 5, 5, 12, 12, 37, 37, 37, 37], device="cuda", dtype=torch.int32
    )
    out = torch.empty(37, 256, device="cuda", dtype=torch.bfloat16)
    runner = CudnnGroupedMmRunner()
    tactics = runner.get_valid_tactics([a, b, offsets, out, None], None)
    assert tactics and all(isinstance(t, tuple) for t in tactics)
    import os

    if os.environ.get("CUDNN_FRONTEND_ENABLE_FROST_ENGINES") == "1":
        assert tactics[0][0] == 20400

    def reference():
        bounds = offsets.tolist()
        return torch.cat(
            [
                a[s:e].float() @ b[j].float().T
                for j, (s, e) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
            ]
        ).to(out.dtype)

    for tactic in tactics[:2]:
        grouped_mm_bf16(a, b, offsets, out=out, tactic=tactic)
        torch.testing.assert_close(out, reference(), rtol=0.02, atol=0.125)
    with autotune():
        grouped_mm_bf16(a, b, offsets, out=out)
    torch.testing.assert_close(out, reference(), rtol=0.02, atol=0.125)
    with pytest.raises(ValueError, match="unavailable"):
        grouped_mm_bf16(a, b, offsets, out=out, tactic=(999999, ()))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_mm_bf16(a, b, offsets, out=out, tactic=tactics[0])
    a.neg_()
    offsets.copy_(
        torch.tensor([0, 0, 1, 8, 14, 15, 16, 18, 37], device="cuda", dtype=torch.int32)
    )
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=0.02, atol=0.125)


@pytest.mark.parametrize("decline_kind", ["not_implemented", "graph_not_supported"])
def test_moe_plain_fc1_fallback_preserves_float_accumulation(decline_kind, monkeypatch):
    _require_b200()
    from flashinfer.fused_moe import cudnn_backend

    import cudnn

    original = cudnn_backend._Stage
    decline = (
        NotImplementedError
        if decline_kind == "not_implemented"
        else cudnn.cudnnGraphNotSupportedError
    )

    def without_fusion(*args, **kwargs):
        if kwargs.get("fused"):
            raise decline("test the fallback graph")
        return original(*args, **kwargs)

    monkeypatch.setattr(cudnn_backend, "_Stage", without_fusion)
    config, act, weights, w1, w2 = _case(RoutingInputMode.UnpackedPrecomputed)
    layer = MoELayer(config)
    actual = layer(act, weights)
    assert all(not s["fused"] for s in layer.runners[0]._resources_cache.values())
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    act.hidden_states_q.neg_()
    for state in layer.runners[0]._resources_cache.values():
        for name in ("fc1_output", "intermediate", "projected", "output"):
            state[name].fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)


@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
@pytest.mark.parametrize("tokens,experts", [(17, 8), (4096, 32)])
def test_native_routing_metadata_replay(mode, tokens, experts):
    _require_b200()
    config, act, weights, w1, w2 = _case(
        mode,
        tokens=tokens,
        experts=experts,
        backend=CudnnMoeConfig(use_native_routing=True),
    )
    layer = MoELayer(config)
    before_ids, before_weights = act.topk_ids.clone(), act.topk_weights.clone()
    actual = layer(act, weights)
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    torch.testing.assert_close(act.topk_ids, before_ids, rtol=0, atol=0)
    torch.testing.assert_close(act.topk_weights, before_weights, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = layer(act, weights)
    previous = actual.clone()
    act.hidden_states_q.neg_()
    act.topk_ids.add_(3).remainder_(experts)
    act.topk_weights.mul_(0.5)
    for state in layer.runners[0]._resources_cache.values():
        for tensor in state["native_sort"].values():
            tensor.fill_(-777)
        for name in ("routed", "intermediate", "projected", "output"):
            state[name].fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    assert not torch.equal(actual, previous)


def test_explicit_fc1_fc2_tactics_prepare_and_replay():
    _require_b200()
    from dataclasses import replace

    config, act, weights, w1, w2 = _case(RoutingInputMode.UnpackedPrecomputed)
    layer = MoELayer(config)
    layer(act, weights)
    runner = layer.runners[0]
    state = runner._resources(runner.pack_inputs(act, weights))
    explicit = CudnnMoeConfig(
        use_native_routing=True,
        fc1_tactic=state["fc1"].tactics[0],
        fc2_tactic=state["fc2"].tactics[0],
    )
    pinned = MoELayer(replace(config, backend=BackendOptions((explicit,))))
    with autotune():
        actual = pinned(act, weights)
    torch.testing.assert_close(actual, _reference(act, w1, w2), rtol=0.02, atol=0.02)
    for state in pinned.runners[0]._resources_cache.values():
        assert state["fc1"].tactics == (explicit.fc1_tactic,)
        assert state["fc2"].tactics == (explicit.fc2_tactic,)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
def test_grouped_fp8_autotune_alpha_and_tactic_capture(dtype):
    _require_b200()
    from flashinfer.grouped_mm import grouped_mm_fp8

    torch.manual_seed(908)
    a = torch.randn(37, 256, device="cuda").clamp(-1, 1).to(dtype)
    b = torch.randn(8, 256, 256, device="cuda").clamp(-1, 1).to(dtype)
    offsets = torch.tensor(
        [0, 5, 5, 12, 12, 37, 37, 37, 37], device="cuda", dtype=torch.int32
    )
    original_offsets = offsets.clone()
    alpha = torch.tensor([0.75], dtype=torch.float32, device="cuda")
    out = torch.empty(37, 256, device="cuda", dtype=torch.bfloat16)

    def reference():
        bounds = offsets.tolist()
        return torch.cat(
            [
                (a[s:e].float() @ b[j].float().T) * alpha
                for j, (s, e) in enumerate(zip(bounds[:-1], bounds[1:], strict=True))
            ]
        ).to(out.dtype)

    with autotune():
        grouped_mm_fp8(a, b, offsets, alpha=alpha, out=out)
    torch.testing.assert_close(out, reference(), rtol=0.02, atol=0.125)
    torch.testing.assert_close(offsets, original_offsets, rtol=0, atol=0)
    runner = CudnnGroupedMmRunner()
    tactic = runner.get_valid_tactics([a, b, offsets, out, alpha], None)[0]
    grouped_mm_fp8(a, b, offsets, alpha=alpha, out=out, tactic=tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_mm_fp8(a, b, offsets, alpha=alpha, out=out, tactic=tactic)
    previous = out.clone()
    a.copy_(-a.float())
    alpha.fill_(0.5)
    offsets.copy_(
        torch.tensor([0, 0, 1, 8, 14, 15, 16, 18, 37], device="cuda", dtype=torch.int32)
    )
    out.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(out, reference(), rtol=0.02, atol=0.125)
    assert not torch.equal(previous, out)
