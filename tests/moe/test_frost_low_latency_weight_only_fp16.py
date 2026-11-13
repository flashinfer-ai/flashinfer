# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""FP16 weight-only activations: typed rounding, live graphs and cache separation."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantFormat,
    RoutingInputMode,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.utils import get_compute_capability
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _activation
from tests.moe.test_frost_low_latency_fp16 import _check
from tests.moe.test_frost_low_latency_w4a16 import (
    _inputs as nv_inputs,
    _decode as fp4_decode,
)
from tests.moe.test_frost_low_latency_mxfp4_w4a16 import _inputs as mx_inputs
from tests.moe.test_frost_low_latency_mxint4_w4a16 import (
    _inputs as int_inputs,
    _decode as int_decode,
)
from tests.moe.test_frost_low_latency_w8a16 import _inputs as w8_inputs

FACTORIES = [nv_inputs, mx_inputs, int_inputs, w8_inputs]


def _reference(x, ids, scores, view, config, packed):
    w8 = config.quant.weight is QuantFormat.FP8PerTensor
    integer = config.quant.weight is QuantFormat.MXINT4
    mx = config.quant.weight is QuantFormat.MXFP4
    output_dtype = (
        torch.float16 if config.quant.output is QuantFormat.FP16 else torch.bfloat16
    )
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]

    def decode(q, scale):
        if integer:
            return int_decode(q, scale)
        return fp4_decode(
            q,
            scale,
            torch.float8_e8m0fnu if mx else torch.float8_e4m3fn,
            32 if mx else 16,
        )

    rows = []
    for token in range(x.shape[0]):
        for expert in ids[token].tolist():
            if not 0 <= expert < w1.shape[0]:
                rows.append(
                    torch.zeros(x.shape[1], device=x.device, dtype=output_dtype)
                )
                continue
            pre = (
                (
                    w1[expert].double()
                    if w8
                    else decode(w1[expert], view["fc1_block_scales"][expert])
                )
                @ x[token].double()
            ).float()
            branches = 2 if config.activation.is_gated else 1
            pre = (
                pre.reshape(branches, -1)
                * (
                    view["fc1_dequant_scale"][expert]
                    if w8
                    else view["fc1_dequant_scale"][expert, :, None]
                )
            ).reshape(-1)
            if not integer:
                pre = pre.to(x.dtype).float()
            up, gate = pre.chunk(2) if config.activation.is_gated else (pre, None)
            mid = _activation(up, gate, config.activation).to(x.dtype)
            value = (
                (
                    w2[expert].double()
                    if w8
                    else decode(w2[expert], view["fc2_block_scales"][expert])
                )
                @ mid.double()
            ).float()
            rows.append((value * view["fc2_dequant_scale"][expert]).to(output_dtype))
    raw = torch.stack(rows)
    effective = scores.bfloat16().float() if packed else scores.float()
    out = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        valid = (ids[:, slot] >= 0) & (ids[:, slot] < w1.shape[0])
        weight = torch.where(valid, effective[:, slot], 0.0)
        out = out + raw.view(*ids.shape, -1)[:, slot].float() * weight[:, None]
    return raw, out.to(output_dtype), effective


@pytest.mark.parametrize(
    "factory", FACTORIES, ids=["nvfp4", "mxfp4", "mxint4", "w8a16"]
)
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_weight_only_fp16_activations_live_graph(factory, index, activation):
    cfg, x, ids, scores, pack, view = factory(activation, index)
    cfg = replace(
        cfg,
        quant=replace(cfg.quant, activation=QuantFormat.FP16, output=QuantFormat.FP16),
    )
    x = x.half()
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    assert (
        inputs[5 if cfg.quant.weight is QuantFormat.FP8PerTensor else 9].dtype
        is torch.float16
    )
    graphs = []
    for tactic in (("ffma", 1, 2), ("ffma", 4, 8)):
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        graphs.append((graph, out))
    for step in range(3):
        x.mul_(-0.875)
        ids.add_(1).remainder_(cfg.routing.num_experts)
        live_scores = inputs[2 if cfg.quant.weight is QuantFormat.FP8PerTensor else 8]
        live_scores.copy_(torch.randn_like(live_scores) * 0.2)
        if cfg.quant.weight is QuantFormat.FP8PerTensor:
            view["fc2_act_quant_scale"].fill_(float("nan"))
        view["fc1_dequant_scale"].mul_(1.125)
        view["fc2_dequant_scale"].mul_(0.875)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            live_scores[0, 0] = float("nan")
            live_scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, ids, live_scores, view, cfg, True)
        for graph, out in graphs:
            out.fill_(float("nan"))
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize(
    "factory", FACTORIES, ids=["nvfp4", "mxfp4", "mxint4", "w8a16"]
)
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_weight_only_float16_dtype_cache_and_outputs(factory, packed, finalize):
    base, x, ids, scores, pack, view = factory(SwiGLU(), 1)
    for dtype, output_dtype in (
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float16),
    ):
        cfg = replace(
            base,
            quant=replace(
                base.quant,
                activation=QuantFormat.FP16
                if dtype is torch.float16
                else QuantFormat.BF16,
                output=QuantFormat.FP16
                if output_dtype is torch.float16
                else QuantFormat.BF16,
            ),
            finalize=MoEFinalizeConfig(do_finalize=finalize),
        )
        act = MoEActivationPack(
            x.to(dtype),
            None,
            ids,
            scores,
            routing_input_mode=RoutingInputMode.PackedPrecomputed
            if packed
            else RoutingInputMode.UnpackedPrecomputed,
        )
        result = MoELayer(cfg)(act, pack)
        raw, expected, effective = _reference(
            act.hidden_states_q, ids, scores, view, cfg, packed
        )
        if finalize:
            _check(result, expected)
        else:
            _check(result[0], raw)
            torch.testing.assert_close(result[1].float(), effective, atol=0, rtol=0)
            torch.testing.assert_close(
                result[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
            )


@pytest.mark.parametrize(
    "factory", FACTORIES, ids=["nvfp4", "mxfp4", "mxint4", "w8a16"]
)
def test_weight_only_fp16_fresh_storage_metadata(factory, monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = factory(SwiGLU(), 3, top_k=32)
    cfg = replace(
        cfg,
        quant=replace(cfg.quant, activation=QuantFormat.FP16, output=QuantFormat.FP16),
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x.half(), None, ids, scores), pack)
    runner.forward(inputs, ("ffma", 4, 8))
    if cfg.quant.weight is QuantFormat.FP8PerTensor:
        from flashinfer.fused_moe.backends.frost_low_latency.fma import (
            fp8_per_tensor as fp4_weight_only,
        )
    fresh = {name: value.clone() for name, value in view.items()}
    fresh["fc2_dequant_scale"].mul_(0.75)
    new_pack = MoEWeightPack()
    new_pack.prepare_for("frost_low_latency", fresh)
    new_x = x.half() * 0.5
    inputs = runner.pack_inputs(
        MoEActivationPack(new_x, None, ids.clone(), scores.clone()), new_pack
    )
    _, expected, _ = _reference(new_x, ids, scores, fresh, cfg, True)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "warm metadata must bind fresh FP16 storage without recompilation"
        )

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    _check(runner.forward(inputs, ("ffma", 4, 8)), expected)


@pytest.mark.parametrize("factory", [nv_inputs, mx_inputs], ids=["nvfp4", "mxfp4"])
@pytest.mark.parametrize("index", [0, 1], ids=["t1-bf16-out", "t2-fp16-out"])
def test_weight_only_fp16_advertised_widths_live_graph(factory, index):
    cfg, x, ids, scores, pack, view = factory(SwiGLU(), index)
    capability = get_compute_capability(x.device)
    if capability not in (
        (9, 0),
        (10, 0),
        (10, 3),
        (12, 0),
    ):
        pytest.skip("FP16 wider-load performance qualified on SM90/SM100/SM103/SM120")
    cfg = replace(
        cfg,
        quant=replace(
            cfg.quant,
            activation=QuantFormat.FP16,
            output=QuantFormat.BF16 if index == 0 else QuantFormat.FP16,
        ),
    )
    x = x.half()
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactics = runner.get_valid_tactics(inputs, None)
    grouped = capability == (9, 0) and x.shape[0] >= 2
    assert len(tactics) == (88 if grouped else 64)
    assert ("ffma", 1, 2, 8, 8) in tactics
    graphs = []
    for tactic in tactics:
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs, tactic).clone()
        graphs.append((graph, output))
    for invalid in (False, True):
        x.mul_(-0.75)
        ids.add_(1).remainder_(cfg.routing.num_experts)
        inputs[8].copy_(torch.randn_like(inputs[8]) * 0.2)
        view["fc1_dequant_scale"].mul_(1.125)
        if invalid:
            ids[0, 0] = -1
            inputs[8][0, 0] = float("nan")
        _, expected, _ = _reference(x, ids, inputs[8], view, cfg, True)
        for graph, output in graphs:
            output.fill_(float("nan"))
            graph.replay()
            _check(output, expected)
