# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Normal cuDNN runner dispatches the native tiled finalizer and captures live data."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    CudnnFp8PerTensorConfig,
    CudnnMoeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingInputMode,
    SwiGLU,
)
from tests.moe.test_moe_cudnn_fp8 import (
    KEY,
    _check,
    _poison,
    _reference as fp8_reference,
    _tactic,
)
from tests.moe.test_unified_moe_cudnn import _case, _reference as bf16_reference


@pytest.mark.parametrize("tokens", [17, 257])
@pytest.mark.parametrize("hidden", [4096, 8192])
@pytest.mark.parametrize("quant", ["bf16", "fp8_packed"])
def test_native_tiled_runner_capture(tokens, hidden, quant, monkeypatch):
    if torch.cuda.get_device_capability() != (10, 0):
        pytest.skip("Native tiled cuDNN dispatch validation covers SM100")
    torch.backends.cuda.matmul.allow_tf32 = False
    mode = (
        RoutingInputMode.PackedPrecomputed
        if tokens == 17
        else RoutingInputMode.UnpackedPrecomputed
    )
    config, act, weights, w1, w2 = _case(mode, tokens=tokens, hidden=hidden)
    tactic = _tactic(True)
    if quant == "bf16":
        backend = CudnnMoeConfig(fc1_tactic=tactic, fc2_tactic=tactic)

        def reference():
            return bf16_reference(act, w1, w2)

    else:
        layout = "blocked_128x128_v1"
        backend = CudnnFp8PerTensorConfig(
            fc1_tactic=tactic, fc2_tactic=tactic, weight_layout=layout
        )
        kwargs = dict(
            num_local_experts=w1.shape[0],
            hidden_size=hidden,
            intermediate_size=w2.shape[2],
            activation=SwiGLU(),
            hidden_states_scale_global=64.0,
            intermediate_scale_global=128.0,
        )
        normal = backend.prepare_weights(w1, w2, **kwargs)
        view = backend.prepare_weights(w1, w2, **kwargs, weight_layout=layout)
        weights = MoEWeightPack()
        weights.prepare_for(KEY, view)
        x, scale = backend.prepare_activations(
            act.hidden_states_q, hidden_states_scale_global=64.0
        )
        act = replace(act, hidden_states_q=x, hidden_states_scale=scale)
        config = replace(
            config,
            quant=QuantConfig(
                weight=QuantFormat.FP8PerTensor, activation=QuantFormat.FP8PerTensor
            ),
        )

        def reference():
            return fp8_reference(act, normal, config.activation)

    config = replace(config, backend=BackendOptions((backend,)))
    layer = MoELayer(config)
    _check(layer(act, weights), reference())
    runner = layer.runners[0]
    inputs = runner.pack_inputs(act, weights)
    state = runner._resources(inputs)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Prepared native finalization must not allocate or compile"
        )

    with monkeypatch.context() as patch:
        patch.setattr(torch, "empty", forbidden)
        patch.setattr(runner, "_prepare_stages", forbidden)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs)
    previous = output.clone()
    if quant == "bf16":
        act.hidden_states_q.neg_()
    else:
        act.hidden_states_q.view(torch.uint8).bitwise_xor_(128)
    act.topk_ids.add_(2).remainder_(config.routing.num_experts)
    act.topk_weights.mul_(0.5)
    expected = reference()
    assert not torch.equal(previous, expected)
    _poison(state)
    graph.replay()
    _check(output, expected)
    torch.cuda.set_sync_debug_mode("error")
    try:
        runner.forward(inputs)
    finally:
        torch.cuda.set_sync_debug_mode("default")
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        graph.replay()
    kernels = [
        event.name
        for event in prof.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    assert sum("moeUnpermuteTiledBf16Kernel" in name for name in kernels) == 1, kernels
    assert not any("fi_experiment_finalize" in name for name in kernels), kernels
    assert any("cudnn_kernel_frost_sm100_moe" in name for name in kernels), kernels
