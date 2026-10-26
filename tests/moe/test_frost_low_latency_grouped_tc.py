# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hopper/SM120 grouped decode: numerical boundaries, route changes and graph bindings."""

from dataclasses import replace
import pytest
import torch
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEWeightPack,
    FrostLowLatencyConfig,
    MoEFinalizeConfig,
    SwiGLU,
    PowLU,
    ReLU2,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from flashinfer.fused_moe.backends.frost_low_latency.fma.activation_config import (
    lower_activation,
)
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_bf16 import (
    ACTIVATIONS,
    _activation,
    _inputs,
    _reference,
)

_DTYPES = [
    (torch.bfloat16, torch.bfloat16),
    (torch.float16, torch.float16),
    (torch.bfloat16, torch.float16),
    (torch.float16, torch.bfloat16),
]


def _require_grouped_tc():
    _require_gpu(allow_sm120=True, allow_sm90=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (12, 0)):
        pytest.skip("Requires Hopper SM90 or SM120")


def _run_projection(
    activation, dtype, outdtype, t, k, g, stages, tile_k, narrow, finalize
):
    _require_grouped_tc()
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import launch

    h, i, e = 256, 128, max(17, k)
    _, x, w1, w2, ids, scores, _ = _inputs(
        activation,
        tokens=t,
        hidden=h,
        intermediate=i,
        experts=e,
        topk=k,
        dtype=dtype,
        output_dtype=outdtype,
    )
    x.mul_(0.2)
    mid = torch.empty(t * k, i, device="cuda", dtype=dtype)
    routes = torch.empty(t * k, h, device="cuda", dtype=outdtype)
    output = torch.empty(t, h, device="cuda", dtype=outdtype)
    args = (x, w1, ids, mid, w2, routes, scores, output)
    name, parameters = lower_activation(activation)
    fn = cute.compile(
        launch,
        *(from_dlpack(v, assumed_align=16) for v in args),
        g,
        stages,
        tile_k,
        name,
        activation.is_gated,
        parameters,
        narrow,
        finalize,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        90 if torch.cuda.get_device_capability() == (9, 0) else 120,
        options=(
            "--gpu-arch sm_90a --enable-tvm-ffi"
            if torch.cuda.get_device_capability() == (9, 0)
            else "--gpu-arch sm_120a --enable-tvm-ffi"
        ),
    )

    def call():
        fn(*args)

    def close(actual, expected, label):
        delta = actual.double() - expected.double()
        err = (
            float(delta.norm() / expected.double().norm().clamp_min(1e-20)),
            float(delta.abs().max() / expected.double().abs().max().clamp_min(1e-20)),
        )
        assert torch.isfinite(actual).all() and err[0] < 0.01 and err[1] < 0.03, (
            label,
            repr(activation),
            str(dtype),
            str(outdtype),
            err,
        )

    def check():
        refmid = []
        refraw = []
        for token in range(t):
            for expert in ids[token].tolist():
                if expert < 0 or expert >= e:
                    refmid.append(torch.zeros(i, device="cuda", dtype=dtype))
                    refraw.append(torch.zeros(h, device="cuda", dtype=outdtype))
                else:
                    value = (w1[expert].double() @ x[token].double()).float()
                    up, gate = (
                        value.chunk(2)
                        if activation.is_gated
                        else (value, torch.zeros_like(value))
                    )
                    m = _activation(up, gate, activation).to(dtype)
                    refmid.append(m)
                    refraw.append((w2[expert].double() @ m.double()).to(outdtype))
        expected_mid = torch.stack(refmid)
        expected_raw = torch.stack(refraw)
        close(mid, expected_mid, "mid")
        close(routes, expected_raw, "routes")
        if finalize:
            effective = scores.bfloat16().float() if narrow else scores.float()
            expected = torch.zeros(t, h, device="cuda", dtype=torch.float32)
            for slot in range(k):
                expected = (
                    expected + expected_raw[slot::k].float() * effective[:, slot, None]
                )
            close(output, expected.to(outdtype), "output")

    call()
    torch.cuda.synchronize()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call()
    for epoch in range(3):
        x.mul_(0.75)
        scores.neg_()
        if epoch == 0:
            ids.copy_(ids.flip(0))
            ids[0, 0] = -1
            ids[-1, -1] = e
        elif epoch == 1:
            ids.fill_(3)
        else:
            ids.fill_(-1)
        mid.fill_(float("nan"))
        routes.fill_(float("nan"))
        output.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check()


def _run_binding(dtype, outdtype, activation, finalize, packed):
    _require_grouped_tc()
    from cutlass import cute

    cfg, x, w1, w2, ids, scores, weights = _inputs(
        activation,
        tokens=4,
        hidden=256,
        intermediate=96,
        topk=10,
        experts=17,
        dtype=dtype,
        output_dtype=outdtype,
    )
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    x2 = -x.clone()
    ids2 = ids.flip(0).clone()
    scores2 = scores.neg().clone()
    w1b, w2b = w1.clone().mul_(0.5), w2.clone().neg_()
    prepare = (
        FrostLowLatencyConfig.prepare_bf16_weights
        if dtype == torch.bfloat16
        else FrostLowLatencyConfig.prepare_fp16_weights
    )
    weights2 = MoEWeightPack()
    weights2.prepare_for("frost_low_latency", prepare(w1b, w2b, activation=activation))
    a = runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), weights
    )
    b = runner.pack_inputs(
        MoEActivationPack(x2, None, ids2, scores2, routing_input_mode=mode), weights2
    )
    assert all(a[j].data_ptr() != b[j].data_ptr() for j in range(8))
    kind = "wgmma" if torch.cuda.get_device_capability() == (9, 0) else "warp_mma"
    tactic = (kind, 4, 4, 64)
    runner.forward(a, tactic)
    original = cute.compile

    def unexpected(*args, **kwargs):
        raise AssertionError("Different live pointers recompiled identical metadata")

    cute.compile = unexpected
    try:
        # Compilation metadata can be shared, buffer/stream bindings cannot.
        stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
        stream_a.wait_stream(torch.cuda.current_stream())
        stream_b.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream_b):
            runner.forward(b, tactic)
        ga, gb = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(ga, stream=stream_a):
            runner.forward(a, tactic)
        with torch.cuda.graph(gb, stream=stream_b):
            runner.forward(b, tactic)
        torch.cuda.synchronize()
        for _epoch in range(3):
            x.mul_(0.75)
            x2.mul_(-0.5)
            ids.add_(1).remainder_(17)
            ids2.add_(3).remainder_(17)
            scores.neg_()
            scores2.mul_(0.5)
            # Both graphs must read new values and overwrite their own live buffers.
            for buffers in [a, b]:
                for j in (5, 6, 7):
                    buffers[j].fill_(float("nan"))
            stream_a.wait_stream(torch.cuda.current_stream())
            stream_b.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream_a):
                ga.replay()
            with torch.cuda.stream(stream_b):
                gb.replay()
            torch.cuda.synchronize()
            for xx, ii, ss, ww1, ww2, buffers in [
                (x, ids, scores, w1, w2, a),
                (x2, ids2, scores2, w1b, w2b, b),
            ]:
                # Packed non-finalized score conversion occurs in pack_inputs.
                effective_scores = buffers[2] if not finalize else ss
                raw, out, _ = _reference(
                    xx,
                    ww1,
                    ww2,
                    ii,
                    effective_scores,
                    activation,
                    packed and finalize,
                    output_dtype=outdtype,
                )
                actual = buffers[7] if finalize else buffers[6]
                ref = out if finalize else raw
                delta = actual.double() - ref.double()
                err = (
                    float(delta.norm() / ref.double().norm().clamp_min(1e-20)),
                    float(
                        delta.abs().max() / ref.double().abs().max().clamp_min(1e-20)
                    ),
                )
                assert (
                    torch.isfinite(actual).all() and err[0] < 0.01 and err[1] < 0.03
                ), err
    finally:
        cute.compile = original


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
@pytest.mark.parametrize("dtype,outdtype", _DTYPES)
def test_grouped_projection_live_graph(index, activation, dtype, outdtype):
    _run_projection(
        activation,
        dtype,
        outdtype,
        (1, 2, 4, 8)[index % 4],
        (1, 3, 10, 32)[index % 4],
        (2, 4, 8)[index % 3],
        (2, 4)[index % 2],
        (64, 128)[index % 2],
        index % 2 == 0,
        index % 3 != 1,
    )


@pytest.mark.parametrize("dtype,outdtype", _DTYPES)
@pytest.mark.parametrize(
    "finalize,packed,activation",
    [
        (True, True, SwiGLU()),
        (True, False, PowLU(m=1.5, limit=2.0)),
        (False, True, ReLU2()),
        (False, False, SwiGLU()),
    ],
)
def test_grouped_projection_independent_buffers_and_streams(
    dtype, outdtype, finalize, packed, activation
):
    _run_binding(dtype, outdtype, activation, finalize, packed)


@pytest.mark.parametrize("tokens", [4, 8])
def test_bounded_tactics_share_original_weights(tokens):
    _require_grouped_tc()
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        SwiGLU(), tokens=tokens, hidden=2048, intermediate=512, experts=17, topk=10
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    tactics = runner.get_valid_tactics(inputs, None)
    grouped = [v for v in tactics if v[0] in ("wgmma", "warp_mma")]
    if torch.cuda.get_device_capability() == (9, 0):
        expected_tactics = [
            ("wgmma", 2, stages, 128 if tokens == 4 else 64) for stages in (2, 4, 8)
        ]
    else:
        choices = (
            ((4, 4, 64), (2, 2, 128))
            if tokens == 4
            else ((8, 2, 128), (4, 2, 64), (2, 2, 64), (2, 2, 128))
        )
        expected_tactics = [("warp_mma", *choice) for choice in choices]
    assert grouped == expected_tactics
    assert len(tactics) == 16 + len(expected_tactics)
    # Repeated routes share weights; values change after each compiled choice.
    for tactic in grouped:
        ids.remainder_(10)
        got = runner.forward(inputs, tactic)
        _, expected, _ = _reference(x, w1, w2, ids, scores, SwiGLU(), True)
        error = (got.double() - expected.double()).norm() / expected.double().norm()
        assert torch.isfinite(got).all() and error < 0.01
    # Untuned selection stays on FMA, even where TC choices are offered.
    from unittest.mock import patch

    with patch(
        "flashinfer.fused_moe.backends.frost_low_latency.runtime.execute"
    ) as call:
        runner.forward(inputs, -1)
    assert call.call_args.args[2] == ("ffma", 1)


@pytest.mark.parametrize(
    "case",
    [
        "fp16",
        "mixed_output",
        "other_activation",
        "tokens2",
        "large_i",
        "large_h",
        "other_topk",
        "no_finalize",
    ],
)
def test_unmeasured_workloads_keep_fma_choices(case):
    _require_grouped_tc()
    args = dict(tokens=4, hidden=2048, intermediate=512, topk=10, experts=17)
    activation = SwiGLU()
    if case == "fp16":
        args["dtype"] = args["output_dtype"] = torch.float16
    elif case == "mixed_output":
        args["output_dtype"] = torch.float16
    elif case == "other_activation":
        activation = ReLU2()
    elif case == "tokens2":
        args["tokens"] = 2
    elif case == "large_i":
        args["intermediate"] = 1024
    elif case == "large_h":
        args["hidden"] = 4096
    elif case == "other_topk":
        args["topk"] = 8
    cfg, x, _, _, ids, scores, weights = _inputs(activation, **args)
    if case == "no_finalize":
        cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=False))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    assert all(v[0] == "ffma" for v in runner.get_valid_tactics(inputs, None))


@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("padding_score", [float("nan"), float("inf"), -float("inf")])
def test_invalid_route_scores_are_ignored_after_graph_capture(
    tokens, packed, padding_score
):
    _require_grouped_tc()
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        SwiGLU(), tokens=tokens, hidden=2048, intermediate=512, experts=17, topk=10
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    inputs = runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), weights
    )
    tactic = next(
        t
        for t in runner.get_valid_tactics(inputs, None)
        if t[0] in ("wgmma", "warp_mma")
    )
    got = runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs, tactic)
    ids[0, 0] = -1
    ids[1, 1] = w1.shape[0]
    scores[0, 0] = padding_score
    scores[1, 1] = padding_score
    valid = (ids >= 0) & (ids < w1.shape[0])
    _, expected, _ = _reference(
        x,
        w1,
        w2,
        ids.clamp(0, w1.shape[0] - 1),
        scores.masked_fill(~valid, 0),
        SwiGLU(),
        packed,
    )
    for _ in range(2):
        for buf in inputs[5:8]:
            buf.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        delta = got.double() - expected.double()
        assert torch.isfinite(got).all()
        assert delta.norm() / expected.double().norm() < 0.01
        assert delta.abs().max() / expected.double().abs().max() < 0.03
