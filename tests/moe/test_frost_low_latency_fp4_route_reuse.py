# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Cross-slot FC1 grouping preserves typed FP4/E4M3 and routing contracts."""

from dataclasses import replace

import pytest
import torch
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantFormat,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS
from tests.moe.test_frost_low_latency_w4a16 import _inputs as nv_inputs
from tests.moe.test_frost_low_latency_mxfp4_w4a16 import _inputs as mx_inputs
from tests.moe.test_frost_low_latency_w8a16 import _inputs as w8_inputs


def _check_case(
    factory,
    index,
    activation,
    group,
    finalize,
    top_k=3,
    dtype=torch.bfloat16,
    output_dtype=None,
    tokens=None,
):
    cfg, x, ids, scores, pack, view = factory(activation, index, top_k=top_k)
    if tokens is not None:
        assert tokens <= x.shape[0]
        x, ids, scores = x[:tokens], ids[:tokens], scores[:tokens]
    output_dtype = dtype if output_dtype is None else output_dtype
    qtype = QuantFormat.FP16 if dtype is torch.float16 else QuantFormat.BF16
    output_qtype = (
        QuantFormat.FP16 if output_dtype is torch.float16 else QuantFormat.BF16
    )
    cfg = replace(
        cfg,
        finalize=MoEFinalizeConfig(do_finalize=finalize),
        quant=replace(cfg.quant, activation=qtype, output=output_qtype),
    )
    x = x.to(dtype)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    w8 = cfg.quant.weight is QuantFormat.FP8PerTensor
    score_slot, mid_slot = (2, 5) if w8 else (8, 9)
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
        inputs[score_slot].copy_(torch.randn_like(inputs[score_slot]) * 0.2)
        if mode == "invalid":
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            inputs[score_slot][0, 0] = float("nan")
            inputs[score_slot][-1, -1] = float("inf")
        expected = runner.forward(inputs, ("ffma", 2, 4))
        expected = (expected if finalize else expected[0]).clone()
        inputs[mid_slot].fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
        assert output.dtype is output_dtype
        torch.testing.assert_close(output, expected, atol=0, rtol=0)
        if w8:
            from tests.moe.test_frost_low_latency_weight_only_fp16 import _reference
            from tests.moe.test_frost_low_latency_fp16 import _check as check_fp16
            from tests.moe.test_frost_low_latency_w8a16 import _check as check_bf16

            raw, final, _ = _reference(x, ids, inputs[score_slot], view, cfg, True)
            (check_fp16 if dtype is torch.float16 else check_bf16)(
                output, final if finalize else raw
            )


@pytest.mark.parametrize(
    "factory", [nv_inputs, mx_inputs, w8_inputs], ids=["nvfp4", "mxfp4", "w8a16"]
)
@pytest.mark.parametrize("tokens", [3, 5, 7])
@pytest.mark.parametrize("group,finalize", [(2, True), (4, False)])
@pytest.mark.parametrize(
    "dtype,output_dtype",
    [(torch.bfloat16, torch.float16), (torch.float16, torch.bfloat16)],
)
def test_route_reuse_partial_groups_mixed_output(
    factory, tokens, group, finalize, dtype, output_dtype
):
    _check_case(
        factory,
        3,
        SwiGLU(),
        group,
        finalize,
        dtype=dtype,
        output_dtype=output_dtype,
        tokens=tokens,
    )


@pytest.mark.parametrize("arch", [(9, 0), (10, 0), (10, 3), (12, 0)])
@pytest.mark.parametrize("index", [0, 1, 2, 3])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_w8_route_reuse_admission(arch, index, dtype, monkeypatch):
    cfg, x, ids, scores, pack, _ = w8_inputs(SwiGLU(), index)
    qtype = QuantFormat.FP16 if dtype is torch.float16 else QuantFormat.BF16
    cfg = replace(cfg, quant=replace(cfg.quant, activation=qtype, output=qtype))
    x = x.to(dtype)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda device: arch)
    tactics = runner.get_valid_tactics(inputs, None)
    eligible = arch == (12, 0) and x.shape[0] >= 4
    assert len(tactics) == (40 if eligible else 16)
    assert sum(len(tactic) == 4 for tactic in tactics) == (24 if eligible else 0)


@pytest.mark.parametrize(
    "factory", [nv_inputs, mx_inputs, w8_inputs], ids=["nvfp4", "mxfp4", "w8a16"]
)
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
@pytest.mark.parametrize("group,finalize", [(2, True), (4, False)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp4_route_reuse_activations_graph(
    factory, index, activation, group, finalize, dtype
):
    _check_case(factory, index, activation, group, finalize, dtype=dtype)


@pytest.mark.parametrize(
    "factory", [nv_inputs, mx_inputs, w8_inputs], ids=["nvfp4", "mxfp4", "w8a16"]
)
@pytest.mark.parametrize("group", [2, 4])
@pytest.mark.parametrize("finalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_fp4_route_reuse_max_routes(factory, group, finalize, dtype):
    _check_case(factory, 3, SwiGLU(), group, finalize, top_k=32, dtype=dtype)


@pytest.mark.parametrize(
    "factory", [nv_inputs, mx_inputs, w8_inputs], ids=["nvfp4", "mxfp4", "w8a16"]
)
def test_fp4_route_reuse_cache_binds_fresh_storage(factory, monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = factory(SwiGLU(), 1)
    if cfg.quant.weight is QuantFormat.FP8PerTensor:
        from flashinfer.fused_moe.backends.frost_low_latency.fma import (
            fp8_per_tensor as fp4_weight_only,
        )
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
