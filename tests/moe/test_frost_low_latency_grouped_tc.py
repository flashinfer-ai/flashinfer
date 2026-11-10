# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SM80/Hopper/SM120 grouped decode: numerical boundaries, route changes and graph bindings."""

from dataclasses import replace
import pytest
import torch
from flashinfer.fused_moe import (
    MoEActivationPack,
    MoEWeightPack,
    FrostLowLatencyConfig,
    MoEFinalizeConfig,
    QuantConfig,
    QuantFormat,
    SwiGLU,
    PowLU,
    ReLU2,
    GeGLU,
    GeGLUTanh,
    SiTU,
    SwiGLUStep,
    GELU,
    SiLU,
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
    _require_gpu(allow_sm120=True, allow_sm90=True, allow_sm80=True)
    if torch.cuda.get_device_capability() not in ((8, 0), (9, 0), (12, 0)):
        pytest.skip("Requires SM80, Hopper SM90 or SM120")


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("length", [1, 31, 32, 33, 65, 80, 129])
@pytest.mark.parametrize("group", [2, 4, 8])
def test_route_members_order_and_live_graph_inputs(dtype, length, group):
    _require_grouped_tc()
    import random

    import cutlass
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from flashinfer.fused_moe.backends.frost_low_latency.tc.route_members import (
        route_members,
    )

    @cute.kernel
    def members_kernel(ids, output, group_size: cutlass.Constexpr):
        leader, _, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        members = route_members(ids, leader, group_size)
        for slot in cutlass.range_constexpr(group_size):
            output[leader, tid, slot] = members[slot]

    @cute.jit
    def launch(ids, output, group_size: cutlass.Constexpr, stream):
        members_kernel(ids, output, group_size).launch(
            grid=(cute.size(ids), 1, 1), block=(128, 1, 1), stream=stream
        )

    ids = torch.zeros(length, device="cuda", dtype=dtype)
    output = torch.empty((length, 128, group), device="cuda", dtype=torch.int32)
    major, minor = torch.cuda.get_device_capability()
    arch = f"sm_{major}{minor}" + ("" if major == 8 else "a")
    compiled = cute.compile(
        launch,
        from_dlpack(ids),
        from_dlpack(output),
        group,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options=f"--gpu-arch {arch} --enable-tvm-ffi",
    )
    compiled(ids, output)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(ids, output)

    rng = random.Random(20261047)
    choices = (-1, 0, 1, 7, 511, 512, 2**33 if dtype == torch.int64 else 2**30)
    patterns = [
        [0] * length,
        list(range(length)),
        [i % 7 - 2 for i in range(length)],
        [rng.choice(choices) for _ in range(length)],
    ]
    for values in patterns:
        ids.copy_(torch.tensor(values, dtype=dtype, device="cuda"))
        expected = []
        for leader, expert in enumerate(values):
            rows = [i for i, value in enumerate(values) if value == expert]
            rank = rows.index(leader)
            picked = rows[rank : rank + group] if rank % group == 0 else []
            expected.append(picked + [-1] * (group - len(picked)))
        expected = torch.tensor(expected, device="cuda", dtype=torch.int32)
        expected = expected[:, None, :].expand_as(output)
        # Check every lane and warp, including empty groups and final chunks.
        output.fill_(-99)
        compiled(ids, output)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        output.fill_(-99)
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0, atol=0)


def _run_projection(
    activation,
    dtype,
    outdtype,
    t,
    k,
    g,
    stages,
    tile_k,
    narrow,
    finalize,
    weight_first=False,
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
        int("".join(map(str, torch.cuda.get_device_capability()))),
        weight_first,
        options=(
            f"--gpu-arch sm_{torch.cuda.get_device_capability()[0]}0"
            f"{'' if torch.cuda.get_device_capability() == (8, 0) else 'a'} --enable-tvm-ffi"
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
    kind = {(8, 0): "ampere_mma", (9, 0): "wgmma", (12, 0): "warp_mma"}[
        torch.cuda.get_device_capability()
    ]
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
@pytest.mark.parametrize(
    "dtype,activation",
    [
        (torch.bfloat16, SwiGLU()),
        (torch.bfloat16, ReLU2()),
        (torch.float16, SwiGLU()),
        (torch.float16, ReLU2()),
        (torch.bfloat16, GeGLU()),
        (torch.float16, GeGLU()),
        (torch.bfloat16, SiLU()),
        (torch.float16, SiLU()),
        (torch.bfloat16, GeGLUTanh()),
        (torch.float16, GeGLUTanh()),
        (torch.bfloat16, SiTU()),
        (torch.float16, SiTU()),
        (torch.bfloat16, SwiGLUStep()),
        (torch.float16, SwiGLUStep()),
    ],
)
def test_bounded_tactics_share_original_weights(tokens, dtype, activation):
    _require_grouped_tc()
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        activation,
        tokens=tokens,
        hidden=2048,
        intermediate=512,
        experts=17,
        topk=10,
        dtype=dtype,
        output_dtype=dtype,
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    tactics = runner.get_valid_tactics(inputs, None)
    grouped = [v for v in tactics if v[0] in ("wgmma", "warp_mma", "ampere_mma")]
    if torch.cuda.get_device_capability() == (8, 0):
        expected_tactics = (
            [("ampere_mma", *((4, 4, 128) if tokens == 4 else (4, 2, 64)))]
            if activation in (SwiGLU(), ReLU2())
            else []
        )
    elif torch.cuda.get_device_capability() == (9, 0):
        if activation in (GeGLUTanh(), SiTU(), SwiGLUStep()):
            expected_tactics = []
        elif activation == GeGLU():
            expected_tactics = [("wgmma", 2, 4, 128 if tokens == 4 else 64)]
        elif dtype == torch.float16 or activation in (ReLU2(), SiLU()):
            if activation in (ReLU2(), SiLU()):
                expected_tactics = [("wgmma", 2, 4, 128)] if tokens == 8 else []
            else:
                tiles = (64, 128) if tokens == 4 else (64,)
                expected_tactics = [("wgmma", 2, 4, tile) for tile in tiles]
        else:
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
        if tokens == 8 and activation == SwiGLU():
            expected_tactics += [("warp_mma", *choice, True) for choice in choices]
    assert grouped == expected_tactics
    assert len(tactics) == 16 + len(expected_tactics)
    # Repeated routes share weights; values change after each compiled choice.
    for tactic in grouped:
        ids.remainder_(10)
        got = runner.forward(inputs, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            runner.forward(inputs, tactic)
        x.neg_()
        ids.copy_(ids.roll(1, dims=1))
        scores.copy_(scores.flip(1))
        for buf in inputs[5:8]:
            buf.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        _, expected, _ = _reference(
            x, w1, w2, ids, scores, activation, True, output_dtype=dtype
        )
        error = (got.double() - expected.double()).norm() / expected.double().norm()
        assert torch.isfinite(got).all() and error < 0.01
        assert (
            got.double() - expected.double()
        ).abs().max() / expected.double().abs().max() < 0.03
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
        "fp16_large",
        "fp16_other_activation",
        "fp16_mixed_output",
        "fp16_custom_swiglu",
        "custom_swiglu",
        "custom_swiglu_step",
        "fp16_custom_swiglu_step",
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
    if case.startswith("fp16"):
        args["dtype"] = args["output_dtype"] = torch.float16
        if case == "fp16_large":
            args.update(hidden=7168, intermediate=2048, topk=8)
        elif case == "fp16_other_activation":
            activation = GELU()
        elif case == "fp16_custom_swiglu_step":
            activation = SwiGLUStep(limit=3.0)
        elif case == "fp16_mixed_output":
            args["output_dtype"] = torch.bfloat16
        else:
            activation = SwiGLU(alpha=1.702)
    elif case == "custom_swiglu_step":
        activation = SwiGLUStep(limit=3.0)
    elif case == "custom_swiglu":
        activation = SwiGLU(alpha=1.702)
    elif case == "mixed_output":
        args["output_dtype"] = torch.float16
    elif case == "other_activation":
        activation = GELU()
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_custom_situ_remains_unsupported(dtype):
    _require_grouped_tc()
    with pytest.raises(NotImplementedError, match="default parameters for SiTU"):
        _inputs(SiTU(gate_scale=2.0), dtype=dtype, output_dtype=dtype)


@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize(
    "dtype,activation",
    [
        (torch.float16, SwiGLU()),
        (torch.bfloat16, ReLU2()),
        (torch.float16, ReLU2()),
    ],
)
def test_sm80_typed_admitted_tactics_live_graph(tokens, dtype, activation):
    _require_grouped_tc()
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 typed grouped admission")
    config, x, w1, w2, ids, scores, weights = _inputs(
        activation,
        tokens=tokens,
        hidden=2048,
        intermediate=512,
        experts=17,
        topk=10,
        dtype=dtype,
        output_dtype=dtype,
    )
    runner = FrostLowLatencyRunner(config, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    choices = [
        v for v in runner.get_valid_tactics(inputs, None) if v[0] == "ampere_mma"
    ]
    assert len(choices) == 1
    got = runner.forward(inputs, choices[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs, choices[0])
    x.neg_()
    ids.copy_(ids.roll(1, dims=1))
    scores.copy_(scores.flip(1))
    _, expected, _ = _reference(
        x, w1, w2, ids, scores, activation, True, output_dtype=dtype
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


@pytest.mark.parametrize(
    "weight",
    [
        QuantFormat.FP8PerTensor,
        QuantFormat.DeepSeekFp8,
        QuantFormat.NVFP4,
        QuantFormat.MXFP4,
        QuantFormat.MXINT4,
    ],
)
def test_sm80_rejects_unqualified_quantized_weights(weight):
    _require_grouped_tc()
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 format-admission boundary")
    config, x, *_ = _inputs(SwiGLU(), tokens=1, hidden=256, intermediate=128)
    config = replace(
        config,
        quant=QuantConfig(
            weight=weight,
            activation=(
                QuantFormat.DeepSeekFp8
                if weight is QuantFormat.DeepSeekFp8
                else QuantFormat.BF16
            ),
            output=QuantFormat.BF16,
        ),
    )
    runner = FrostLowLatencyRunner(config, x.device)
    with pytest.raises(NotImplementedError, match="SM80 currently supports BF16/FP16"):
        runner.check_support()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("name", ["w1_weight", "w2_weight"])
def test_grouped_weight_alignment_is_checked_before_compile(dtype, name):
    _require_grouped_tc()
    config, x, _, _, ids, scores, weights = _inputs(
        SwiGLU(),
        tokens=4,
        hidden=256,
        intermediate=128,
        dtype=dtype,
        output_dtype=dtype,
    )
    view = dict(weights.get_view("frost_low_latency"))
    original = view[name]
    storage = torch.empty(original.numel() + 1, dtype=dtype, device=x.device)
    unaligned = storage[1:].view_as(original)
    unaligned.copy_(original)
    assert unaligned.is_contiguous() and unaligned.data_ptr() % 16
    view[name] = unaligned
    bad_weights = MoEWeightPack()
    bad_weights.prepare_for("frost_low_latency", view)
    runner = FrostLowLatencyRunner(config, x.device)
    runner.check_support()
    runner.build()
    with pytest.raises(ValueError, match=name + " must be aligned contiguous"):
        runner.pack_inputs(MoEActivationPack(x, None, ids, scores), bad_weights)


@pytest.mark.parametrize("tokens", [4, 8])
def test_sm80_large_admitted_tactics_live_graph(tokens):
    _require_grouped_tc()
    if torch.cuda.get_device_capability() != (8, 0):
        pytest.skip("SM80 large-shape qualification")
    cfg, x, w1, w2, ids, scores, weights = _inputs(
        SwiGLU(), tokens=tokens, hidden=7168, intermediate=2048, experts=17, topk=8
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    tactics = runner.get_valid_tactics(inputs, None)
    grouped = [v for v in tactics if v[0] == "ampere_mma"]
    assert grouped == [("ampere_mma", 8, 2, 64)] and len(tactics) == 17
    ids.remainder_(8)
    got = runner.forward(inputs, grouped[0])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runner.forward(inputs, grouped[0])
    for epoch in range(2):
        x.neg_()
        ids.copy_(ids.roll(1, dims=1))
        scores.copy_(scores.flip(1))
        if epoch:
            ids[0, 0] = -1
            ids[-1, -1] = w1.shape[0]
        for buf in inputs[5:8]:
            buf.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        valid = (ids >= 0) & (ids < w1.shape[0])
        _, expected, _ = _reference(
            x,
            w1,
            w2,
            ids.clamp(0, w1.shape[0] - 1),
            scores.masked_fill(~valid, 0),
            SwiGLU(),
            True,
        )
        delta = got.double() - expected.double()
        assert torch.isfinite(got).all()
        assert delta.norm() / expected.double().norm() < 0.01
        assert delta.abs().max() / expected.double().abs().max() < 0.03


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
        if t[0] in ("wgmma", "warp_mma", "ampere_mma")
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


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sm120_weight_first_cache_and_live_graphs(dtype, monkeypatch):
    _require_grouped_tc()
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Weight-first scheduling is qualified on SM120")
    from cutlass import cute
    from flashinfer.fused_moe.backends.frost_low_latency import runtime

    # Isolate this test's metadata cache without clearing other tests' entries.
    cache = {}
    monkeypatch.setattr(runtime, "_COMPILED", cache)
    cases = []
    for _ in range(2):
        cfg, x, w1, w2, ids, scores, weights = _inputs(
            SwiGLU(),
            tokens=8,
            hidden=2048,
            intermediate=512,
            experts=19,
            topk=10,
            dtype=dtype,
            output_dtype=dtype,
        )
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
        # Preparation owns copies of the checkpoint weights. Mutate the live
        # prepared buffers, not the original pre-preparation tensors.
        assert inputs[3].data_ptr() != w1.data_ptr()
        assert inputs[4].data_ptr() != w2.data_ptr()
        cases.append((runner, inputs, x, inputs[3], inputs[4], ids, scores))
    old = ("warp_mma", 4, 2, 64)
    overlap = (*old, True)
    runner, inputs, *_ = cases[0]
    a = runner.forward(inputs, old).clone()
    b = runner.forward(inputs, overlap).clone()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    # Both scheduling choices must occupy distinct compile-cache entries.
    assert len(cache) == 2
    assert {key[4] for key in cache} == {old, overlap}

    def unexpected(*args, **kwargs):
        raise AssertionError("Live buffers with identical metadata recompiled")

    monkeypatch.setattr(cute, "compile", unexpected)
    graphs = []
    for runner, inputs, *_ in cases:
        for tactic in (old, overlap):
            runner.forward(inputs, tactic)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                runner.forward(inputs, tactic)
            graphs.append(graph)
    for _epoch in range(2):
        for index, (_runner, inputs, x, w1, w2, ids, scores) in enumerate(cases):
            x.mul_(-0.75 if index else 0.5)
            w1.mul_(0.75)
            w2.neg_()
            ids.add_(1).remainder_(19)
            scores.copy_(scores.flip(1))
            _, expected, _ = _reference(
                x, w1, w2, ids, scores, SwiGLU(), True, output_dtype=dtype
            )
            for graph in graphs[2 * index : 2 * index + 2]:
                for buf in inputs[5:8]:
                    buf.fill_(float("nan"))
                graph.replay()
                torch.cuda.synchronize()
                got = inputs[7]
                error = (
                    got.double() - expected.double()
                ).norm() / expected.double().norm()
                max_error = (
                    got.double() - expected.double()
                ).abs().max() / expected.double().abs().max()
                assert torch.isfinite(got).all() and error < 0.01 and max_error < 0.03


@pytest.mark.parametrize(
    "case",
    [
        "tokens4",
        "hidden",
        "intermediate",
        "topk",
        "activation",
        "output_dtype",
        "finalize",
    ],
)
def test_sm120_weight_first_rejects_unqualified_manual_tactics(case):
    _require_grouped_tc()
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("Weight-first scheduling is qualified on SM120")
    args = dict(tokens=8, hidden=2048, intermediate=512, experts=17, topk=10)
    activation = SwiGLU()
    if case == "tokens4":
        args["tokens"] = 4
    elif case == "hidden":
        args["hidden"] = 1024
    elif case == "intermediate":
        args["intermediate"] = 256
    elif case == "topk":
        args["topk"] = 8
    elif case == "activation":
        activation = GeGLU()
    elif case == "output_dtype":
        args["output_dtype"] = torch.float16
    cfg, x, _, _, ids, scores, weights = _inputs(activation, **args)
    if case == "finalize":
        cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=False))
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), weights)
    tactic = ("warp_mma", 4, 2, 64, True)
    assert tactic not in runner.get_valid_tactics(inputs, None)
    with pytest.raises(ValueError, match="Weight-first grouped decode requires"):
        runner.forward(inputs, tactic)
