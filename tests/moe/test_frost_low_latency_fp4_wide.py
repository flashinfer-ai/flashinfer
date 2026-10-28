# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wide FP4 loads preserve typed activations, tail masks and live graph operands."""

from dataclasses import replace

import pytest
import torch
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEFinalizeConfig,
    MoEWeightPack,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS
from tests.moe.test_frost_low_latency_w4a16 import (
    _inputs as nv_inputs,
    _reference as nv_ref,
)
from tests.moe.test_frost_low_latency_mxfp4_w4a16 import (
    _inputs as mx_inputs,
    _reference as mx_ref,
)
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check


def _check_case(factory, reference, index, activation, widths, finalize, top_k=3):
    cfg, x, ids, scores, pack, view = factory(activation, index, top_k=top_k)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactic = ("ffma", 2, 4, *widths)
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = runner.forward(inputs, tactic)
        out = (result if finalize else result[0]).clone()
    for step in range(3):
        x.mul_(-0.875)
        ids.add_(1).remainder_(cfg.routing.num_experts)
        inputs[8].copy_(torch.randn_like(inputs[8]) * 0.2)
        view["fc1_dequant_scale"].mul_(1.125)
        view["fc2_dequant_scale"].mul_(0.875)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            inputs[8][0, 0] = float("nan")
            inputs[8][-1, -1] = float("inf")
        raw, expected, _ = reference(x, ids, inputs[8], view, activation, True)
        out.fill_(float("nan"))
        inputs[9].fill_(float("nan"))
        graph.replay()
        _check(out, expected if finalize else raw)


@pytest.mark.parametrize(
    "factory,reference",
    [(nv_inputs, nv_ref), (mx_inputs, mx_ref)],
    ids=["nvfp4", "mxfp4"],
)
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
@pytest.mark.parametrize("widths", [(4, 8), (8, 4), (8, 8)])
def test_fp4_wide_activations_tail_graph(factory, reference, index, activation, widths):
    _check_case(factory, reference, index, activation, widths, index % 2 == 0)


@pytest.mark.parametrize(
    "factory,reference",
    [(nv_inputs, nv_ref), (mx_inputs, mx_ref)],
    ids=["nvfp4", "mxfp4"],
)
@pytest.mark.parametrize("widths", [(4, 8), (8, 4), (8, 8)])
@pytest.mark.parametrize("finalize", [False, True])
def test_fp4_wide_max_routes(factory, reference, widths, finalize):
    _check_case(factory, reference, 3, SwiGLU(), widths, finalize, top_k=32)


@pytest.mark.parametrize(
    "factory,reference",
    [(nv_inputs, nv_ref), (mx_inputs, mx_ref)],
    ids=["nvfp4", "mxfp4"],
)
def test_fp4_wide_cache_binds_fresh_storage(factory, reference, monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = factory(SwiGLU(), 1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactics = [("ffma", 2, 4), *(("ffma", 2, 4, *w) for w in ((4, 8), (8, 4), (8, 8)))]
    for tactic in tactics:
        runner.forward(inputs, tactic)
    fresh = {key: value.clone() for key, value in view.items()}
    fresh["fc1_dequant_scale"].mul_(0.75)
    fresh_pack = MoEWeightPack()
    fresh_pack.prepare_for("frost_low_latency", fresh)
    fresh_x, fresh_ids, fresh_scores = x * 0.5, ids.roll(1, dims=1), scores * 1.2
    inputs = runner.pack_inputs(
        MoEActivationPack(fresh_x, None, fresh_ids, fresh_scores), fresh_pack
    )
    _, expected, _ = reference(
        fresh_x, fresh_ids, fresh_scores, fresh, cfg.activation, True
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("warm FP4 load-width metadata must bind fresh storage")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    for tactic in reversed(tactics):
        _check(runner.forward(inputs, tactic), expected)


@pytest.mark.parametrize(
    "factory,reference",
    [(nv_inputs, nv_ref), (mx_inputs, mx_ref)],
    ids=["nvfp4", "mxfp4"],
)
@pytest.mark.parametrize("index", [0, 1], ids=["t1", "t2"])
def test_fp4_hopper_advertised_tactics_live_graph(factory, reference, index):
    cfg, x, ids, scores, pack, view = factory(SwiGLU(), index)
    if torch.cuda.get_device_capability(x.device) != (9, 0):
        pytest.skip("Hopper software-decoder tactic qualification")
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactics = runner.get_valid_tactics(inputs, None)
    assert ("ffma", 1, 2, 8, 8) in tactics
    graphs = []
    for tactic in tactics:
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            result = runner.forward(inputs, tactic).clone()
        graphs.append((graph, result))
    for invalid in (False, True):
        x.mul_(-0.75)
        ids.add_(1).remainder_(cfg.routing.num_experts)
        inputs[8].copy_(torch.randn_like(inputs[8]) * 0.2)
        view["fc1_dequant_scale"].mul_(1.125)
        if invalid:
            ids[0, 0] = -1
            inputs[8][0, 0] = float("nan")
        _, expected, _ = reference(x, ids, inputs[8], view, cfg.activation, True)
        for graph, result in graphs:
            result.fill_(float("nan"))
            graph.replay()
            _check(result, expected)
