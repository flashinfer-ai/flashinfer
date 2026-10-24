# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-slot FC1 grouping preserves typed FP4 activation and routing contracts."""

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
from tests.moe.test_frost_low_latency_w4a16 import _inputs as nv_inputs
from tests.moe.test_frost_low_latency_mxfp4_w4a16 import _inputs as mx_inputs


def _check_case(factory, index, activation, group, finalize, top_k=3):
    cfg, x, ids, scores, pack, view = factory(activation, index, top_k=top_k)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactic = ("ffma", 2, 4, group)
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = runner.forward(inputs, tactic)
        output = (result if finalize else result[0]).clone()
    canonical = ids.clone()
    k = ids.shape[1]
    for mode in ("disjoint", "shared", "partial", "permuted", "invalid"):
        ids.copy_(canonical)
        if mode in ("shared", "permuted", "invalid"):
            ids.remainder_(k)
        if mode == "partial":
            ids[1::2, : k // 2].copy_(ids[: ids.shape[0] - 1 : 2, : k // 2])
        if mode == "permuted":
            for row in range(ids.shape[0]):
                ids[row].copy_(torch.roll(ids[row], row))
        x.mul_(-0.875)
        view["fc1_dequant_scale"].mul_(1.125)
        view["fc2_dequant_scale"].mul_(0.875)
        inputs[8].copy_(torch.randn_like(inputs[8]) * 0.2)
        if mode == "invalid":
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            inputs[8][0, 0] = float("nan")
            inputs[8][-1, -1] = float("inf")
        expected = runner.forward(inputs, ("ffma", 2, 4))
        expected = (expected if finalize else expected[0]).clone()
        inputs[9].fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize("factory", [nv_inputs, mx_inputs], ids=["nvfp4", "mxfp4"])
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
@pytest.mark.parametrize("group,finalize", [(2, True), (4, False)])
def test_fp4_route_reuse_activations_graph(factory, index, activation, group, finalize):
    _check_case(factory, index, activation, group, finalize)


@pytest.mark.parametrize("factory", [nv_inputs, mx_inputs], ids=["nvfp4", "mxfp4"])
@pytest.mark.parametrize("group", [2, 4])
@pytest.mark.parametrize("finalize", [False, True])
def test_fp4_route_reuse_max_routes(factory, group, finalize):
    _check_case(factory, 3, SwiGLU(), group, finalize, top_k=32)


@pytest.mark.parametrize("factory", [nv_inputs, mx_inputs], ids=["nvfp4", "mxfp4"])
def test_fp4_route_reuse_cache_binds_fresh_storage(factory, monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = factory(SwiGLU(), 1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactics = [("ffma", 2, 4), ("ffma", 2, 4, 2), ("ffma", 2, 4, 4)]
    for tactic in tactics:
        runner.forward(args, tactic)
    fresh = {key: value.clone() for key, value in view.items()}
    fresh["fc1_dequant_scale"].mul_(0.75)
    fresh_pack = MoEWeightPack()
    fresh_pack.prepare_for("frost_low_latency", fresh)
    args = runner.pack_inputs(
        MoEActivationPack(x * 0.5, None, ids.roll(1, dims=1), scores * 1.2), fresh_pack
    )
    expected = runner.forward(args, tactics[0]).clone()

    def forbidden(*args, **kwargs):
        raise AssertionError("warm route grouping must bind fresh runtime storage")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    for tactic in reversed(tactics):
        torch.testing.assert_close(
            runner.forward(args, tactic), expected, atol=0, rtol=0
        )
