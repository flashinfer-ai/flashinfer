# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Bounded W8A16 Tensor Core admission and live scale/storage contracts."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    FrostLowLatencyConfig,
    GeGLU,
    MoEActivationPack,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    RoutingInputMode,
    SiLU,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _require_gpu
from tests.moe.test_frost_low_latency_bf16 import _inputs as typed_inputs
from tests.moe.test_frost_low_latency_fp8_per_tensor import _check as _check_bf16
from tests.moe.test_frost_low_latency_fp16 import _check as _check_fp16
from tests.moe.test_frost_low_latency_weight_only_fp16 import _reference


def _check(actual, expected):
    assert actual.dtype == expected.dtype
    checker = _check_fp16 if actual.dtype is torch.float16 else _check_bf16
    checker(actual, expected)


def _setup(packed=True, tokens=8, activation=None, dtype=torch.bfloat16, **shape):
    activation = SwiGLU() if activation is None else activation
    _require_gpu(allow_sm120=True, allow_sm90=True)
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 3), (12, 0)):
        pytest.skip("SM90/SM103/SM120 grouped W8A16")
    args = dict(tokens=tokens, hidden=2048, intermediate=512, experts=17, topk=10)
    args.update(shape)
    cfg, x, w1, w2, ids, scores, _ = typed_inputs(
        activation, dtype=dtype, output_dtype=dtype, **args
    )
    cfg = replace(
        cfg,
        quant=QuantConfig(
            weight=QuantFormat.FP8PerTensor,
            activation=QuantFormat.BF16
            if dtype is torch.bfloat16
            else QuantFormat.FP16,
            output=QuantFormat.BF16 if dtype is torch.bfloat16 else QuantFormat.FP16,
        ),
    )
    scales = torch.full((args["experts"],), 0.1, device=x.device)
    view = FrostLowLatencyConfig.prepare_fp8_weight_only_weights(
        w1.to(torch.float8_e4m3fn),
        scales,
        w2.to(torch.float8_e4m3fn),
        scales.clone(),
        activation=activation,
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    inputs = runner.pack_inputs(
        MoEActivationPack(x, None, ids, scores, routing_input_mode=mode), pack
    )
    return runner, inputs, view


def _expected(inputs, view, packed, config):
    x, ids, scores = inputs[:3]
    valid = (ids >= 0) & (ids < view["fc1_expert_weights"].shape[0])
    return _reference(x, ids, scores.masked_fill(~valid, 0), view, config, packed)[1]


CASES = [
    (4, torch.bfloat16, 4, 2),
    (4, torch.bfloat16, 4, 4),
    (4, torch.float16, 4, 2),
    (8, torch.bfloat16, 8, 2),
    (8, torch.float16, 8, 2),
    (8, torch.float16, 4, 2),
]


def _candidate_setup(packed, tokens, dtype, group, stages, activation, **shape):
    runner, inputs, view = _setup(
        packed, tokens=tokens, dtype=dtype, activation=activation, **shape
    )
    sm103 = torch.cuda.get_device_capability() == (10, 3)
    sm90 = torch.cuda.get_device_capability() == (9, 0)
    if sm90:
        admitted_activation = activation in (SwiGLU(), GeGLU()) or (
            dtype is torch.bfloat16 and activation in (ReLU2(), SiLU())
        )
        if tokens != 8 or not admitted_activation:
            pytest.skip("Outside SM90 W8A16 T8 activation admission")
        choices = runner.get_valid_tactics(inputs, None)
        expected = [
            ("warp_mma", g, 2, 128)
            for g in ((4, 8) if dtype is torch.float16 else (8,))
        ]
        if dtype is torch.bfloat16 and activation == GeGLU():
            expected.insert(0, ("warp_mma", 8, 2, 256))
        assert [t for t in choices if t[0] != "ffma"] == expected
        assert len([t for t in choices if t[0] == "ffma"]) == 16
        return runner, inputs, view, ("warp_mma", group, stages, 128)
    if tokens == 8 and group == 4:
        pytest.skip("T8 group4 is only admitted on SM90")
    sm103_activation = activation in (SwiGLU(), GeGLU()) or (
        dtype is torch.float16 and tokens == 8 and activation in (ReLU2(), SiLU())
    )
    if (sm103 and not sm103_activation) or (
        not sm103
        and (
            tokens != 8 or (dtype is torch.bfloat16 and activation in (ReLU2(), SiLU()))
        )
    ):
        pytest.skip("Outside this architecture's W8A16 TC admission")
    choices = runner.get_valid_tactics(inputs, None)
    expected = (
        [("warp_mma", 4, s, 128) for s in ((2, 4) if dtype is torch.bfloat16 else (2,))]
        if tokens == 4
        else [("warp_mma", 8, 2, 128)]
    )
    assert [t for t in choices if t[0] != "ffma"] == expected
    assert len([t for t in choices if t[0] == "ffma"]) == (16 if sm103 else 40)
    return runner, inputs, view, ("warp_mma", group, stages, 128)


@pytest.mark.parametrize("activation", [SwiGLU(), GeGLU(), ReLU2(), SiLU()])
@pytest.mark.parametrize("tokens,dtype,group,stages", CASES)
@pytest.mark.parametrize("packed", [False, True])
def test_grouped_w8a16_admitted_live_graph(
    packed, tokens, dtype, group, stages, activation
):
    runner, inputs, view, tactic = _candidate_setup(
        packed, tokens, dtype, group, stages, activation
    )
    _check_live_graph(runner, inputs, view, packed, tactic)


def _check_live_graph(runner, inputs, view, packed, tactic):
    actual = runner.forward(inputs, tactic)
    _check(actual, _expected(inputs, view, packed, runner.config))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs, tactic)
    x, ids, scores = inputs[:3]
    for epoch in range(3):
        x.mul_(-0.75)
        scores.copy_(scores.flip(1))
        view["fc1_dequant_scale"].mul_(1.1)
        view["fc2_dequant_scale"].mul_(0.9)
        view["fc2_act_quant_scale"].fill_(float("nan"))
        if epoch == 0:
            ids.fill_(3)
        elif epoch == 1:
            ids[0, 0] = -1
            ids[-1, -1] = runner.config.routing.num_experts
            scores[0, 0] = float("nan")
            scores[-1, -1] = float("inf")
        else:
            ids.fill_(-1)
        for buf in inputs[5:8]:
            buf.fill_(float("nan"))
        graph.replay()
        _check(actual, _expected(inputs, view, packed, runner.config))


@pytest.mark.parametrize("activation", [SwiGLU(), GeGLU(), ReLU2(), SiLU()])
@pytest.mark.parametrize("tokens,dtype,group,stages", CASES)
@pytest.mark.parametrize("packed", [False, True])
def test_grouped_w8a16_fresh_storage_and_streams(
    packed, tokens, dtype, group, stages, activation
):
    runner, inputs, _, tactic = _candidate_setup(
        packed, tokens, dtype, group, stages, activation
    )
    _check_fresh_storage(runner, inputs, packed, tactic)


def _check_fresh_storage(runner, inputs, packed, tactic):
    from flashinfer.fused_moe.backends.frost_low_latency.runtime import _COMPILED

    runner.forward(inputs, tactic)
    before = len(_COMPILED)
    pairs = []
    for index in range(2):
        fresh = [v.clone() if isinstance(v, torch.Tensor) else v for v in inputs]
        fresh[0].mul_(index + 0.5)
        fresh[2].mul_(-1 - index)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            runner.forward(fresh, tactic)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = runner.forward(fresh, tactic).clone()
        pairs.append((fresh, stream, graph, out))
    assert len(_COMPILED) == before
    for fresh, stream, graph, out in pairs:
        fresh[0].mul_(-0.8)
        fresh[9].mul_(0.75)
        fresh[11].mul_(1.1)
        fresh[10].fill_(float("nan"))
        view = dict(
            fc1_expert_weights=fresh[3],
            fc2_expert_weights=fresh[4],
            fc1_dequant_scale=fresh[9],
            fc2_dequant_scale=fresh[11],
        )
        expected = _expected(fresh, view, packed, runner.config)
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            graph.replay()
        stream.synchronize()
        _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
def test_grouped_w8a16_large_bf16_sm103(packed):
    _require_gpu(allow_sm120=True)
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("Larger BF16 W8A16 TC admission requires SM103")
    runner, inputs, view, tactic = _candidate_setup(
        packed,
        8,
        torch.bfloat16,
        8,
        2,
        SwiGLU(),
        hidden=7168,
        intermediate=2048,
        topk=8,
    )
    _check_fresh_storage(runner, inputs, packed, tactic)
    _check_live_graph(runner, inputs, view, packed, tactic)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_w8a16_large_sm120(packed, dtype):
    _require_gpu(allow_sm120=True)
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Larger BF16/FP16 W8A16 TC admission requires SM120")
    runner, inputs, view, tactic = _candidate_setup(
        packed,
        8,
        dtype,
        8,
        2,
        SwiGLU(),
        hidden=7168,
        intermediate=2048,
        topk=8,
    )
    _check_fresh_storage(runner, inputs, packed, tactic)
    _check_live_graph(runner, inputs, view, packed, tactic)


def test_grouped_w8a16_bounded_admission():
    cases = [
        dict(tokens=2),
        dict(activation=SwiGLU(alpha=1.702)),
        dict(hidden=4096),
        dict(intermediate=1024),
        dict(topk=8),
    ]
    _setup()
    if torch.cuda.get_device_capability() != (9, 0):
        cases.extend(dict(activation=a) for a in (ReLU2(), SiLU()))
    if torch.cuda.get_device_capability() == (10, 3):
        cases.extend(
            [
                dict(tokens=2, activation=GeGLU()),
                dict(dtype=torch.float16, tokens=2),
                dict(dtype=torch.float16, hidden=7168, intermediate=2048, topk=8),
                dict(tokens=4, hidden=7168, intermediate=2048, topk=8),
                dict(hidden=7040, intermediate=2048, topk=8),
                dict(hidden=7168, intermediate=2048, topk=10),
            ]
        )
        cases.extend(
            dict(dtype=torch.float16, tokens=4, activation=activation)
            for activation in (ReLU2(), SiLU())
        )
        cases.append(dict(dtype=torch.float16, activation=SwiGLU(alpha=1.702)))
        cases.extend(
            dict(dtype=torch.float16, tokens=2, activation=activation)
            for activation in (GeGLU(), ReLU2(), SiLU())
        )
    elif torch.cuda.get_device_capability() == (9, 0):
        cases.extend(
            dict(dtype=torch.bfloat16, tokens=t, activation=a)
            for t in (1, 2, 4)
            for a in (ReLU2(), SiLU())
        )
        cases.extend(dict(tokens=t) for t in (1, 4))
        cases.extend(dict(dtype=torch.float16, tokens=t) for t in (1, 2, 4))
        cases.extend(
            dict(dtype=torch.float16, activation=activation)
            for activation in (ReLU2(), SiLU(), SwiGLU(alpha=1.702))
        )
        cases.extend(
            dict(dtype=torch.float16, tokens=t, activation=GeGLU()) for t in (1, 2, 4)
        )
        cases.extend(
            dict(dtype=torch.bfloat16, tokens=t, activation=GeGLU()) for t in (1, 2)
        )
        cases.extend(
            [
                dict(dtype=torch.float16, hidden=4096),
                dict(dtype=torch.float16, intermediate=1024),
                dict(dtype=torch.float16, topk=8),
            ]
        )
    else:
        cases.append(dict(tokens=4))
        large = dict(tokens=8, hidden=7168, intermediate=2048, topk=8)
        cases.extend(
            {**large, "dtype": dtype, **change}
            for dtype in (torch.bfloat16, torch.float16)
            for change in (
                dict(tokens=4),
                dict(hidden=7040),
                dict(intermediate=1024),
                dict(topk=10),
                dict(activation=GeGLU()),
                dict(activation=ReLU2()),
                dict(activation=SiLU()),
                dict(activation=SwiGLU(alpha=1.702)),
            )
        )
    for case in cases:
        runner, inputs, _ = _setup(**case)
        assert all(t[0] == "ffma" for t in runner.get_valid_tactics(inputs, None))

    for dtype in (torch.bfloat16, torch.float16):
        runner, inputs, _ = _setup(dtype=dtype)
        other = QuantFormat.FP16 if dtype is torch.bfloat16 else QuantFormat.BF16
        runner.config = replace(
            runner.config, quant=replace(runner.config.quant, output=other)
        )
        assert all(t[0] == "ffma" for t in runner.get_valid_tactics(inputs, None))
    for dtype in (torch.bfloat16, torch.float16):
        runner, inputs, _ = _setup(dtype=dtype)
        runner.config = replace(
            runner.config, finalize=replace(runner.config.finalize, do_finalize=False)
        )
        assert all(t[0] == "ffma" for t in runner.get_valid_tactics(inputs, None))


@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("fresh_storage", [False, True])
def test_grouped_w8a16_hopper_k256_contract(tokens, packed, fresh_storage):
    runner, inputs, view = _setup(packed, tokens=tokens, activation=GeGLU())
    if torch.cuda.get_device_capability() != (9, 0):
        pytest.skip("K256 W8A16 requires SM90")
    tactic = ("warp_mma", tokens, 2, 256)
    choices = runner.get_valid_tactics(inputs, None)
    expected = [tactic] + ([("warp_mma", 8, 2, 128)] if tokens == 8 else [])
    assert [t for t in choices if t[0] != "ffma"] == expected
    assert len([t for t in choices if t[0] == "ffma"]) == 16
    if fresh_storage:
        _check_fresh_storage(runner, inputs, packed, tactic)
    else:
        _check_live_graph(runner, inputs, view, packed, tactic)


def test_grouped_w8a16_hopper_k256_rejects_unqualified_calls():
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import execute

    for change in (
        dict(dtype=torch.float16),
        dict(activation=SwiGLU()),
        dict(tokens=2),
        dict(hidden=4096),
        dict(intermediate=1024),
        dict(topk=8),
    ):
        args = dict(tokens=4, activation=GeGLU())
        args.update(change)
        runner, inputs, _ = _setup(**args)
        assert all(
            t[0] == "ffma" or t[-1] != 256
            for t in runner.get_valid_tactics(inputs, None)
        )
        with pytest.raises(ValueError, match="K256 W8A16 requires"):
            execute(inputs, runner.config, ("warp_mma", 4, 2, 256))
