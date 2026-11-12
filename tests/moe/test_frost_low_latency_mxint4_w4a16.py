# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MXINT4/BF16 preserves BF16 K32 scales across the shared FMA pipeline."""

from dataclasses import replace
import pytest
import torch
from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _activation
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check


def test_mxint4_packed_decoder_all_codes():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.fused_moe.backends.frost_low_latency.fma.fp4_weight import (
        scaled_quad,
    )

    @cute.kernel
    def kernel(words, scales, out):
        bid, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        row = bid * 256 + tid
        if row < cute.size(words):
            values = scaled_quad(words[row], scales[row], True, 32, True)
            for col in cutlass.range_constexpr(4):
                out[row, col] = values[col]

    @cute.jit
    def launch(words, scales, out, stream):
        kernel(words, scales, out).launch(
            grid=((cute.size(words) + 255) // 256, 1, 1),
            block=(256, 1, 1),
            stream=stream,
        )

    words = torch.arange(65536, device="cuda", dtype=torch.int32)
    scales = torch.ones(65536, device="cuda")
    out = torch.empty(65536, 4, device="cuda")
    major, minor = torch.cuda.get_device_capability()
    compiled = cute.compile(
        launch,
        *(from_dlpack(v) for v in (words, scales, out)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch sm_{major}{minor}a --enable-tvm-ffi",
    )
    codes = (words[:, None] >> torch.tensor([0, 4, 8, 12], device="cuda")) & 15
    expected = torch.where(codes < 8, codes, codes - 16).float()
    compiled(words, scales, out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(words, scales, out)
    for exponent in (0, -120, -17, 19, 120):
        scales.fill_(2.0**exponent)
        out.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(out, expected * scales[:, None], atol=0, rtol=0)


def _decode(q, scale):
    code = torch.stack((q & 15, q >> 4), dim=-1).flatten(-2).long()
    signed = torch.where(code < 8, code, code - 16).float()
    return (signed * scale.float().repeat_interleave(32, -1)).double()


def _reference(x, ids, scores, view, activation, packed):
    w1, w2 = view["fc1_expert_weights"], view["fc2_expert_weights"]
    branches = 2 if activation.is_gated else 1
    rows = []
    for token in range(x.shape[0]):
        for expert in ids[token].tolist():
            if not 0 <= expert < w1.shape[0]:
                rows.append(
                    torch.zeros(x.shape[1], device=x.device, dtype=torch.bfloat16)
                )
                continue
            pre = (
                _decode(w1[expert], view["fc1_block_scales"][expert])
                @ x[token].double()
            ).float()
            pre = (
                pre.reshape(branches, -1) * view["fc1_dequant_scale"][expert, :, None]
            ).reshape(-1)
            up, gate = pre.chunk(2) if activation.is_gated else (pre, None)
            mid = _activation(up, gate, activation).bfloat16()
            value = (
                _decode(w2[expert], view["fc2_block_scales"][expert]) @ mid.double()
            ).float()
            rows.append((value * view["fc2_dequant_scale"][expert]).bfloat16())
    raw = torch.stack(rows)
    effective = scores.bfloat16().float() if packed else scores.float()
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        valid = (ids[:, slot] >= 0) & (ids[:, slot] < w1.shape[0])
        weight = torch.where(valid, effective[:, slot], 0.0)
        result = result + raw.view(*ids.shape, -1)[:, slot].float() * weight[:, None]
    return raw, result.bfloat16(), effective


def _inputs(activation, index=0, top_k=3):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    torch.manual_seed(4800 + index)
    t, h, i, e, k = (
        (1, 2, 4, 8)[index % 4],
        (128, 256, 384)[index % 3],
        (32, 96, 160)[index % 3],
        max(8, top_k),
        top_k,
    )
    x = torch.randn(t, h, device="cuda", dtype=torch.bfloat16) * 0.3
    branches = 2 if activation.is_gated else 1
    w1 = torch.randint(
        0, 256, (e, branches * i, h // 2), device="cuda", dtype=torch.uint8
    )
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = (
        torch.rand(e, branches * i, h // 32, device="cuda") * 0.015625 + 0.00390625
    ).to(torch.bfloat16)
    s2 = (torch.rand(e, h, i // 32, device="cuda") * 0.015625 + 0.00390625).to(
        torch.bfloat16
    )
    view = FrostLowLatencyConfig.prepare_mxint4_weight_only_weights(
        w1, s1, w2, s2, activation=activation
    )
    ids = torch.arange(t * k, device="cuda", dtype=torch.int32).reshape(t, k) % e
    scores = torch.randn(t, k, device="cuda") * 0.2
    cfg = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.MXINT4, activation=QuantFormat.BF16),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    return cfg, x, ids, scores, pack, view


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_mxint4_w4a16_activations_and_live_graph_scales(index, activation):
    cfg, x, ids, scores, pack, view = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    captures = []
    for tactic in (("ffma", 1, 4), ("ffma", 4, 8)):
        runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(inputs, tactic).clone()
        captures.append((graph, out))
    for step in range(3):
        x.copy_(torch.randn_like(x) * 0.3)
        ids.copy_((ids + 1) % cfg.routing.num_experts)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
        scores.copy_(torch.randn_like(scores) * 0.2)
        # Updating one matrix's live scale must not change the other matrix.
        fc2_before = view["fc2_dequant_scale"].clone()
        view["fc1_dequant_scale"].mul_(1.2)
        torch.testing.assert_close(
            view["fc2_dequant_scale"], fc2_before, atol=0, rtol=0
        )
        fc1_before = view["fc1_dequant_scale"].clone()
        view["fc2_dequant_scale"].mul_(0.85)
        torch.testing.assert_close(
            view["fc1_dequant_scale"], fc1_before, atol=0, rtol=0
        )
        for name in ("fc1_block_scales", "fc2_block_scales"):
            sf = view[name].view(torch.bfloat16)
            sf.copy_((sf.float() * 2.0).to(torch.bfloat16))
        if step == 2:
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        for graph, out in captures:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_mxint4_w4a16_public_output_contract(packed, finalize):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 1)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    out = MoELayer(cfg)(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), pack
    )
    raw, expected, effective = _reference(x, ids, scores, view, cfg.activation, packed)
    if finalize:
        _check(out, expected)
    else:
        _check(out[0], raw)
        torch.testing.assert_close(out[1].float(), effective, atol=0, rtol=0)
        torch.testing.assert_close(
            out[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
        )


@pytest.mark.parametrize("first,second", [(1, 1), (1, 8), (8, 1), (8, 8)])
def test_mxint4_w4a16_max_tokens_and_routes(first, second):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 3, top_k=32)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    output = runner.forward(inputs, ("ffma", first, second))
    _, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    _check(output, expected)


def test_mxint4_w4a16_preparation_preserves_payloads():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    e, h, i = 2, 128, 32
    w1 = torch.randint(0, 256, (e, 2 * i, h // 2), device="cuda", dtype=torch.uint8)
    w2 = torch.randint(0, 256, (e, h, i // 2), device="cuda", dtype=torch.uint8)
    s1 = torch.randn(e, 2 * i, h // 32, device="cuda", dtype=torch.bfloat16)
    s2 = torch.randn(e, h, i // 32, device="cuda", dtype=torch.bfloat16)
    view = FrostLowLatencyConfig.prepare_mxint4_weight_only_weights(w1, s1, w2, s2)
    torch.testing.assert_close(
        view["fc1_expert_weights"].reshape(e, 2, 128, h // 2)[:, :, :i],
        w1.reshape(e, 2, i, h // 2),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        view["fc1_block_scales"].reshape(e, 2, 128, h // 32)[:, :, :i],
        s1.reshape(e, 2, i, h // 32),
        atol=0,
        rtol=0,
    )
    torch.testing.assert_close(
        view["fc2_expert_weights"][:, :, : i // 2], w2, atol=0, rtol=0
    )
    torch.testing.assert_close(
        view["fc2_block_scales"][:, :, : i // 32], s2, atol=0, rtol=0
    )
    assert not view["fc1_expert_weights"].reshape(e, 2, 128, h // 2)[:, :, i:].any()
    assert not view["fc2_expert_weights"][:, :, i // 2 :].any()
    with pytest.raises(ValueError):
        FrostLowLatencyConfig.prepare_mxint4_weight_only_weights(w1, s1.float(), w2, s2)


def test_mxint4_and_nvfp4_compiled_metadata_separation(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only
    from tests.moe.test_frost_low_latency_w4a16 import _inputs as nv_inputs

    calls = []
    from tests.moe.test_frost_low_latency_mxfp4_w4a16 import _inputs as mx_inputs

    for factory in (nv_inputs, mx_inputs, _inputs):
        cfg, x, ids, scores, pack, view = factory(SwiGLU())
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        expected = runner.forward(args, ("ffma", 2, 4)).clone()
        cloned = {key: value.clone() for key, value in view.items()}
        fresh = MoEWeightPack()
        fresh.prepare_for("frost_low_latency", cloned)
        args = runner.pack_inputs(
            MoEActivationPack(x.clone(), None, ids.clone(), scores.clone()), fresh
        )
        calls.append((runner, args, expected))

    def forbidden(*args, **kwargs):
        raise AssertionError("warm metadata must not compile or retain tensor views")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    for runner, args, expected in reversed(calls):
        torch.testing.assert_close(
            runner.forward(args, ("ffma", 2, 4)), expected, atol=0, rtol=0
        )


@pytest.mark.parametrize("widths", [(4, 8), (8, 4), (8, 8)])
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_mxint4_wide_load_activations_and_tail_graph(index, activation, widths):
    cfg, x, ids, scores, pack, view = _inputs(activation, index)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactic = ("ffma", 2, 4, *widths)
    runner.forward(args, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = runner.forward(args, tactic).clone()
    for step in range(3):
        x.copy_(torch.randn_like(x) * 0.3)
        ids.add_(1).remainder_(cfg.routing.num_experts)
        scores.copy_(torch.randn_like(scores) * 0.2)
        view["fc1_block_scales"].mul_(1.125)
        view["fc2_block_scales"].mul_(0.875)
        if step == 2:
            ids[0, 0] = -1
            ids[-1, -1] = cfg.routing.num_experts
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        _, expected, _ = _reference(x, ids, scores, view, activation, True)
        graph.replay()
        _check(out, expected)


@pytest.mark.parametrize("widths", [(4, 8), (8, 4), (8, 8)])
@pytest.mark.parametrize("finalize", [False, True])
def test_mxint4_wide_load_max_routes(widths, finalize):
    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 3, top_k=32)
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    raw, expected, _ = _reference(x, ids, scores, view, cfg.activation, True)
    result = runner.forward(args, ("ffma", 8, 8, *widths))
    _check(result if finalize else result[0], expected if finalize else raw)


def test_mxint4_load_width_cache_rebinds_fresh_storage(monkeypatch):
    from flashinfer.fused_moe.backends.frost_low_latency.fma import fp4_weight_only

    cfg, x, ids, scores, pack, view = _inputs(SwiGLU(), 1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    args = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    tactics = [("ffma", 2, 4), *(("ffma", 2, 4, *w) for w in ((4, 8), (8, 4), (8, 8)))]
    for tactic in tactics:
        runner.forward(args, tactic)
    fresh_view = {key: value.clone() for key, value in view.items()}
    fresh_view["fc1_dequant_scale"].mul_(0.75)
    fresh_pack = MoEWeightPack()
    fresh_pack.prepare_for("frost_low_latency", fresh_view)
    fresh_x, fresh_ids, fresh_scores = x * 0.5, ids.roll(1, dims=1), scores * 1.2
    fresh_args = runner.pack_inputs(
        MoEActivationPack(fresh_x, None, fresh_ids, fresh_scores), fresh_pack
    )
    _, expected, _ = _reference(
        fresh_x, fresh_ids, fresh_scores, fresh_view, cfg.activation, True
    )

    def forbidden(*args, **kwargs):
        raise AssertionError("warm load-width metadata must bind fresh runtime storage")

    monkeypatch.setattr(fp4_weight_only.cute, "compile", forbidden)
    monkeypatch.setattr(fp4_weight_only, "from_dlpack", forbidden)
    for tactic in reversed(tactics):
        _check(runner.forward(fresh_args, tactic), expected)
