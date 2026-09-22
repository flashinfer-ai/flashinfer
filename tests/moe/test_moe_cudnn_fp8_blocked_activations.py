# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed weights preserve the configurable gated-activation graph interface."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import BackendOptions, MoELayer, MoEWeightPack
from tests.moe.test_moe_cudnn_fp8 import (
    ACTIVATIONS,
    KEY,
    _check,
    _poison,
    _prepared,
    _reference,
)


@pytest.mark.parametrize("activation", ACTIVATIONS)
def test_public_blocked_gated_activation_capture(activation):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Initial blocked weight support covers SM100")
    config, act, ordinary, w1, w2 = _prepared(activation, True, tokens=33)
    backend = replace(config.backend.candidates[0], weight_layout="blocked_128x128_v1")
    view = backend.prepare_weights(
        w1,
        w2,
        num_local_experts=w1.shape[0],
        hidden_size=w1.shape[2],
        intermediate_size=w2.shape[2],
        activation=activation,
        hidden_states_scale_global=64.0,
        intermediate_scale_global=128.0,
        weight_layout=backend.weight_layout,
    )
    packed = MoEWeightPack()
    packed.prepare_for(KEY, view)
    layer = MoELayer(replace(config, backend=BackendOptions((backend,))))
    reference = ordinary.get_view(KEY)
    _check(layer(act, packed), _reference(act, reference, activation))
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, packed)
    state = runner._resources(inputs)
    for name in ("fc1", "fc2"):
        stage = state[name]
        assert stage.graph.get_engine_and_knobs_at_index(0)[0] == 20400
        compiled = stage.graph._compiled_plans[0]._compiled
        assert compiled.chain.moe.weight_layout == "blocked_128x128_v1"
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.forward(inputs)
    previous = out.clone()
    act.hidden_states_q.view(torch.uint8).bitwise_xor_(128)
    act.topk_ids.add_(2).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    _poison(state)
    graph.replay()
    expected = _reference(act, reference, activation)
    _check(out, expected)
    assert not torch.equal(previous, expected)
    for name in ("fc1_scale", "intermediate_multiplier", "fc2_scale"):
        view[name].mul_(0.75)
        reference[name].mul_(0.75)
        _poison(state)
        graph.replay()
        _check(out, _reference(act, reference, activation))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        runner.forward(inputs)
    kernels = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    frost = [name for name in kernels if "cudnn_kernel_frost_sm100_moe" in name]
    assert len(frost) == 2 and all("packed_b_128x128_v1" in name for name in frost), (
        kernels
    )
