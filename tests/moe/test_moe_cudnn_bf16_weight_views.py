# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared FC1 storage, legacy packs and live captured weight updates."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import CudnnMoeConfig, MoELayer, MoEWeightPack
from flashinfer.fused_moe.cudnn_backend import CudnnMoeRunner
from flashinfer.tllm_enums import RoutingInputMode
from tests.moe.test_moe_cudnn_bf16_joint import _kernel_names, _poison
from tests.moe.test_unified_moe_cudnn import _case, _reference


def _prepare(w1, w2):
    return CudnnMoeConfig.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w1.shape[1] // 2,
    )


def _storage_bytes(view):
    return sum(
        {
            (t.device, t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
            for t in view.values()
        }.values()
    )


@pytest.mark.parametrize("experts", [1, 8])
@pytest.mark.parametrize("strided", [False, True])
def test_bf16_preparation_uses_one_fc1_allocation(experts, strided):
    w1 = torch.arange(experts * 32 * 32).reshape(experts, 32, 32).to(torch.bfloat16)
    w2 = torch.arange(experts * 32 * 16).reshape(experts, 32, 16).to(torch.bfloat16)
    if strided:
        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2).contiguous().transpose(1, 2)
    view = _prepare(w1, w2)
    torch.testing.assert_close(view["up"], w1[:, :16], rtol=0, atol=0)
    torch.testing.assert_close(view["gate"], w1[:, 16:], rtol=0, atol=0)
    torch.testing.assert_close(view["down"], w2, rtol=0, atol=0)
    assert _storage_bytes(view) == (w1.numel() + w2.numel()) * w1.element_size()
    original = w1.clone()
    view["up"].neg_()
    view["gate"].mul_(0.5)
    torch.testing.assert_close(view["gate_up"][:, :16], original[:, 16:] * 0.5)
    torch.testing.assert_close(view["gate_up"][:, 16:], -original[:, :16])
    torch.testing.assert_close(w1, original, rtol=0, atol=0)


def test_bf16_and_fp8_weight_layout_cache_keys(monkeypatch):
    from flashinfer.fused_moe import CudnnFp8PerTensorConfig
    from flashinfer.fused_moe.cudnn_fp8_backend import CudnnFp8PerTensorRunner

    monkeypatch.setattr(CudnnMoeRunner, "_cache_key_extras", lambda self: ("base",))
    view = _prepare(
        torch.zeros(8, 32, 32, dtype=torch.bfloat16),
        torch.zeros(8, 32, 16, dtype=torch.bfloat16),
    )
    weights = [view[n] for n in ("up", "gate", "down", "gate_up")]
    inputs = [None, None, torch.zeros(1), *weights, None, False]
    legacy = [*inputs[:3], *(t.contiguous() for t in weights), *inputs[7:]]
    runner = object.__new__(CudnnMoeRunner)
    assert runner.get_cache_key_extras(inputs) != runner.get_cache_key_extras(legacy)
    # Independent storage with the same declared layout must reuse the key.
    second = _prepare(
        torch.zeros(8, 32, 32, dtype=torch.bfloat16),
        torch.zeros(8, 32, 16, dtype=torch.bfloat16),
    )
    other = [
        *inputs[:3],
        *(second[n] for n in ("up", "gate", "down", "gate_up")),
        *inputs[7:],
    ]
    assert runner.get_cache_key_extras(inputs) == runner.get_cache_key_extras(other)
    fp8 = object.__new__(CudnnFp8PerTensorRunner)
    fp8.backend_config = CudnnFp8PerTensorConfig()
    # Moving layout inclusion to the base class preserves the FP8 key exactly.
    assert fp8.get_cache_key_extras(inputs) == (
        "base",
        str(inputs[2].dtype),
        False,
        fp8._weight_layout_key(inputs),
    )


def _check(actual, expected):
    # Keep reference reductions off the device under Compute Sanitizer.
    actual, expected = actual.cpu(), expected.cpu()
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)
    assert (
        float(
            (actual.float() - expected.float()).norm()
            / expected.float().norm().clamp_min(1e-12)
        )
        <= 0.01
    )


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize(
    "mode", [RoutingInputMode.PackedPrecomputed, RoutingInputMode.UnpackedPrecomputed]
)
def test_bf16_shared_and_legacy_weights_capture(fused, mode, monkeypatch):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (12, 0),
    ):
        pytest.skip("SM100 or SM120 GPU required")
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    from cudnn.gemm.frost.knobs import GemmKnobs
    from cudnn.gemm.frost.tile_config import by_name

    arch = torch.cuda.get_device_capability()[0]
    name = (
        "CONFIG_sm100_64x64x128_64x64x32_cluster1x4_1ctamma"
        if arch == 10
        else "CONFIG_sm120_32x64x128_16x16x32_cluster1x1_warps2x4"
    )
    knobs = replace(GemmKnobs.from_config(by_name(name)), moe_sched_policy=1)
    record = (
        20400,
        tuple(sorted((int(k), int(v)) for k, v in knobs.to_public().items())),
    )
    backend = CudnnMoeConfig(
        use_native_routing=True, fc1_fusion=fused, fc1_tactic=record, fc2_tactic=record
    )
    config, act, _, w1, w2 = _case(mode, backend=backend)
    canonical = [(w1.clone(), w2.clone()), (w1.clone(), w2.clone()), (-w1, w2 * 0.5)]
    views = [_prepare(a, b) for a, b in canonical]
    views[1] = {
        name: value.clone(memory_format=torch.contiguous_format)
        for name, value in views[1].items()
    }
    # Keep the independent reference weights separate from the prepared down view.
    canonical = [(a.clone(), b.clone()) for a, b in canonical]
    layer = MoELayer(config)
    runner = layer.runners[0]
    inputs = []
    for view in views:
        pack = MoEWeightPack()
        pack.prepare_for("cudnn", view)
        inputs.append(runner.pack_inputs(act, pack))
    states = [runner._resources(x) for x in inputs]
    assert states[0] is states[2] and states[0] is not states[1]
    assert runner.get_cache_key_extras(inputs[0]) == runner.get_cache_key_extras(
        inputs[2]
    )
    assert runner.get_cache_key_extras(inputs[0]) != runner.get_cache_key_extras(
        inputs[1]
    )
    assert _storage_bytes(views[0]) < _storage_bytes(views[1])
    captures = []
    for inp, (a, b), state in zip(inputs, canonical, states, strict=True):
        _check(runner.forward(inp), _reference(act, a, b))
        assert state["fused"] is fused
        plans = state["fc1"].graph._compiled_plans.values()
        assert all(p._compiled.chain.num_gemms == (2 if fused else 1) for p in plans)
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=side):
            output = runner.forward(inp)
        torch.cuda.current_stream().wait_stream(side)
        names = _kernel_names(graph)
        assert sum("frost_sm" in n and "moe_grouped_matmul" in n for n in names) == 2
        captures.append((graph, output))
    act.hidden_states_q.neg_()
    act.topk_ids.add_(2).remainder_(8)
    act.topk_weights.mul_(0.5)
    for index, (view, (a, b), state, (graph, output)) in enumerate(
        zip(views, canonical, states, captures, strict=True)
    ):
        view["up"].neg_()
        view["gate"].mul_(0.5)
        if index == 1:  # Legacy weights have independent fused/unfused storage.
            view["gate_up"][:, :256].mul_(0.5)
            view["gate_up"][:, 256:].neg_()
        view["down"].mul_(0.75)
        changed = torch.cat((-a[:, :256], a[:, 256:] * 0.5), dim=1)
        expected = _reference(act, changed, b * 0.75)
        _poison(state)
        for stage in ("fc1", "fc2"):
            state[stage].workspace.fill_(0xA5)
        with pytest.raises(AssertionError):
            _check(output, expected)
        graph.replay()
        _check(output, expected)
    assert len(runner._resources_cache) == 2
