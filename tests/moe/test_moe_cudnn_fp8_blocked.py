# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Versioned physical weight layout, live binding and independent CUDA captures."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFp8PerTensorConfig,
    MoELayer,
    MoEWeightPack,
    SwiGLU,
)
from tests.moe.test_moe_cudnn_fp8 import KEY, _check, _poison, _prepared, _reference

LAYOUT = "blocked_128x128_v1"


def _weight_pack(view):
    weights = MoEWeightPack()
    weights.prepare_for(KEY, view)
    return weights


def _block(view):
    out = dict(view)
    for name in ("up", "gate", "down"):
        w = view[name]
        e, n, k = w.shape
        out[name] = (
            w.reshape(e, n // 128, 128, k // 128, 128)
            .permute(0, 1, 3, 2, 4)
            .contiguous()
        )
    return out


@pytest.mark.parametrize("tokens", [17, 257])
def test_blocked_weight_captures_bind_current_pack(tokens, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Initial blocked layout validation covers SM100")
    config, act, weights, _, _ = _prepared(SwiGLU(), True, tokens=tokens)
    plain = weights.get_view(KEY)
    second = {name: tensor.clone() for name, tensor in plain.items()}
    second["up"].view(torch.uint8).bitwise_xor_(128)
    blocked = [_block(view) for view in (plain, second)]
    packs = [_weight_pack(view) for view in blocked]
    backend = replace(config.backend.candidates[0], weight_layout=LAYOUT)
    packed_config = replace(config, backend=BackendOptions((backend,)))
    layer = MoELayer(packed_config)
    layer(act, packs[0])
    runner = layer.runners[0]
    with pytest.raises(ValueError, match="up"):
        runner.pack_inputs(act, weights)
    normal_layer = MoELayer(config)
    normal_layer(act, weights)
    normal_runner = normal_layer.runners[0]
    with pytest.raises(ValueError, match="up"):
        normal_runner.pack_inputs(act, packs[0])
    inputs = [runner.pack_inputs(act, pack) for pack in packs]
    state = runner._resources(inputs[0])
    assert state is runner._resources(inputs[1])
    assert runner.get_cache_key_extras(inputs[0]) != normal_runner.get_cache_key_extras(
        normal_runner.pack_inputs(act, weights)
    )
    for stage in ("fc1", "fc2"):
        graph = state[stage].graph
        assert state[stage].weight_layout == LAYOUT
        assert all(
            graph.get_engine_and_knobs_at_index(j)[0] == 20400
            for j in range(graph.get_execution_plan_count())
        )
    expected = [_reference(act, view, config.activation) for view in (plain, second)]
    graphs = []

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Prepared packed-weight execution must not allocate or compile"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(runner, "_prepare_stages", forbidden)
        for inp in inputs:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner.forward(inp)
            graphs.append(graph)
    # Distinct captures sharing a plan must keep each pack's own pointers.
    for j in (0, 1, 0):
        _poison(state)
        graphs[j].replay()
        _check(inputs[j][7], expected[j])
    assert not torch.equal(expected[0], expected[1])
    # Mutate every packed weight and corresponding independent reference once.
    for name in ("up", "gate", "down"):
        plain[name].view(torch.uint8).bitwise_xor_(128)
        blocked[0][name].view(torch.uint8).bitwise_xor_(128)
        _poison(state)
        graphs[0].replay()
        _check(inputs[0][7], _reference(act, plain, config.activation))
    act.topk_ids.add_(1).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    act.hidden_states_q.view(torch.uint8).bitwise_xor_(128)
    _poison(state)
    graphs[0].replay()
    _check(inputs[0][7], _reference(act, plain, config.activation))
    # Prepared forward must remain asynchronous on the current stream.
    torch.cuda.set_sync_debug_mode("error")
    try:
        runner.forward(inputs[0])
    finally:
        torch.cuda.set_sync_debug_mode("default")


def test_blocked_preparation_matches_declared_physical_order():
    _, _, weights, w1, w2 = _prepared(SwiGLU(), True, tokens=17)
    kwargs = dict(
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        hidden_states_scale_global=64.0,
        intermediate_scale_global=128.0,
    )
    prepared = CudnnFp8PerTensorConfig.prepare_weights(
        w1, w2, **kwargs, weight_layout=LAYOUT
    )
    expected = _block(weights.get_view(KEY))
    for name in expected:
        assert torch.equal(
            prepared[name].reshape(-1).view(torch.uint8),
            expected[name].reshape(-1).view(torch.uint8),
        )
    with pytest.raises(ValueError, match="weight_layout"):
        CudnnFp8PerTensorConfig(weight_layout="future_layout")


def test_default_layout_preserves_older_frontend_signature(monkeypatch):
    import cudnn
    from flashinfer.fused_moe.cudnn_fp8_backend import _Fp8Stage

    class AcceptedLegacyCall(Exception):
        pass

    class LegacyGraph:
        def __init__(self, **kwargs):
            pass

        def tensor(self, **kwargs):
            return object()

        def moe_grouped_matmul(self, token, weight, first_token_offset, *, mode):
            raise AcceptedLegacyCall

    monkeypatch.setattr(cudnn, "create_handle", lambda: None)
    monkeypatch.setattr(cudnn, "pygraph", LegacyGraph)
    with pytest.raises(AcceptedLegacyCall):
        _Fp8Stage(
            torch.empty(17, 256),
            [torch.empty(8, 256, 256)],
            torch.empty(9, dtype=torch.int32),
            torch.empty(8, 1, 1),
            torch.empty(17, 256),
        )
