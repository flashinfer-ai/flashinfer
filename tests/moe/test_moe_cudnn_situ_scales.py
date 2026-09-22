# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large SiTU scales exercise tanh near zero; relative error rejects zero output."""

import pytest
import torch
from flashinfer.fused_moe import MoELayer, SiTU
from tests.moe.test_moe_cudnn_activations import _prepared_case, _reference
from tests.moe.test_unified_moe_cudnn import _require_b200


@pytest.mark.parametrize(
    "activation",
    [
        SiTU(gate_scale=1e8),
        SiTU(linear_scale=1e8),
        SiTU(gate_scale=1e8, linear_scale=1e8),
        SiTU(gate_scale=0.13, linear_scale=37.7),
        SiTU(gate_scale=1e-4, linear_scale=1e3),
    ],
    ids=repr,
)
@pytest.mark.parametrize("wide", [True, False], ids=["wide-static", "2cta-dynamic"])
def test_situ_scales_relative_error_and_live_capture(activation, wide, monkeypatch):
    _require_b200()
    monkeypatch.setenv("CUDNN_FRONTEND_ENABLE_FROST_ENGINES", "1")
    config, act, weights, w1, w2 = _prepared_case(activation, wide=wide)
    layer = MoELayer(config)
    output = layer(act, weights)
    state = layer.runners[0]._resources(layer.runners[0].pack_inputs(act, weights))
    assert state["fused"]
    assert all(
        state[name].graph.get_engine_and_knobs_at_index(0)[0] == 20400
        for name in ["fc1", "fc2"]
    )

    def check():
        reference = _reference(act, w1, w2, activation).float()
        actual = output.float()
        relative_l2 = float(
            (actual - reference).norm() / reference.norm().clamp_min(1e-30)
        )
        assert torch.isfinite(actual).all()
        assert relative_l2 <= 0.01, (activation, wide, relative_l2)
        torch.testing.assert_close(actual, reference, rtol=0.02, atol=0.02)

    check()
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        output = layer(act, weights)
    torch.cuda.current_stream().wait_stream(side)
    previous = output.clone()
    act.hidden_states_q.neg_()
    act.topk_ids.add_(2).remainder_(8)
    act.topk_weights.mul_(0.5)
    for name in ["routed", "intermediate", "projected", "output"]:
        state[name].fill_(float("nan"))
    graph.replay()
    check()
    assert not torch.equal(previous, output)


@pytest.mark.parametrize(
    "activation",
    [
        SiTU(gate_scale=1e-39, linear_scale=None),
        SiTU(linear_scale=1e-39),
        SiTU(gate_scale=1e38),
        SiTU(linear_scale=1e38),
    ],
    ids=repr,
)
def test_situ_reciprocal_guard_keeps_extreme_scale_graph_preparable(activation):
    # Preparation-only contract: an extra reciprocal must not overflow FP32.
    # This is not a claim about executing BF16 subnormal intermediates.
    import cudnn
    from flashinfer.fused_moe.cudnn_backend import _gated_activation

    graph = cudnn.pygraph(
        io_data_type=cudnn.data_type.FLOAT,
        intermediate_data_type=cudnn.data_type.FLOAT,
        compute_data_type=cudnn.data_type.FLOAT,
    )
    up = graph.tensor(name="up", dim=[1, 1, 16], stride=[16, 16, 1])
    gate = graph.tensor(name="gate", dim=[1, 1, 16], stride=[16, 16, 1])
    bindings = {}
    result = _gated_activation(
        graph, up, gate, activation, torch.device("cpu"), bindings
    )
    result.set_output(True)
    graph.validate()
    assert bindings and all(torch.isfinite(value).all() for value in bindings.values())
