# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host regressions for exact-shape Frost winners and block-scale tactic bounds."""

from collections import OrderedDict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    GeGLU,
    MoEActivationPack,
    MoEConfig,
    MoELayer,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    ReLU2,
    SiTU,
    RoutingConfig,
    RoutingInputMode,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner


@pytest.mark.parametrize("first,following", [(8, (5, 6, 7)), (4, (3,))])
@pytest.mark.parametrize("packed", [False, True])
def test_layer_reselects_shape_bounded_frost_tactic(
    first, following, packed, monkeypatch
):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (9, 0))
    monkeypatch.setattr(
        "flashinfer.fused_moe.backends.frost_low_latency.runner.version",
        lambda _: "4.8.0",
    )
    config = MoEConfig(
        RoutingConfig(16, 10),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor,
            activation=QuantFormat.BF16,
            output=QuantFormat.BF16,
        ),
        ExpertConfig(512),
        activation=GeGLU(),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    runner.check_support()
    runner.build()
    # Keep the actual candidate enumeration and layer cache; no GPU work is needed
    # to reject a winner that belongs to another token count.
    monkeypatch.setattr(
        runner, "pack_inputs", lambda act, weights: [act.hidden_states_q]
    )
    monkeypatch.setattr(runner, "launch_kwargs_for", lambda _: {})
    launched, selected = [], []

    def forward(inputs, tactic, **kwargs):
        assert tactic in runner.get_valid_tactics(inputs, None), (
            inputs[0].shape,
            tactic,
        )
        launched.append((inputs[0].shape[0], tactic))
        return inputs[0]

    monkeypatch.setattr(runner, "forward", forward)
    layer = MoELayer.__new__(MoELayer)
    layer.config, layer.runners, layer._winners = config, [runner], OrderedDict()

    def select(act, weights, runners):
        assert runners == [runner]
        choices = runner.get_valid_tactics([act.hidden_states_q], None)
        tactic = next((t for t in choices if len(t) == 4 and t[-1] == 256), choices[0])
        selected.append(act.num_tokens)
        return runner, tactic

    monkeypatch.setattr(layer, "_select_winner", select)
    weights = MoEWeightPack({"frost_low_latency": {}})
    mode = (
        RoutingInputMode.PackedPrecomputed
        if packed
        else RoutingInputMode.UnpackedPrecomputed
    )
    for tokens in (first, *following, first, *following):
        act = MoEActivationPack(
            torch.empty(tokens, 2048, dtype=torch.bfloat16),
            None,
            torch.empty(tokens, 10, dtype=torch.int32),
            torch.empty(tokens, 10),
            routing_input_mode=mode,
        )
        assert layer(act, weights) is act.hidden_states_q
    assert selected == [first, *following]
    assert launched[0][1] == ("warp_mma", first, 2, 256)
    assert all(tactic[0] == "ffma" for tokens, tactic in launched if tokens != first)


@pytest.mark.parametrize("group", [4, 8])
def test_deepseek_rejects_k256_before_tensor_access(group):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency.tc.deepseek_fp8 import execute

    # A disallowed block-scale tile must fail before touching inputs or CUDA.
    with pytest.raises(ValueError, match="K64/K128"):
        execute([None] * 15, SimpleNamespace(), ("warp_mma", group, 2, 256))


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("capability", [(9, 0), (10, 3), (12, 0)])
@pytest.mark.parametrize("choice", [(8, 2, 128), (4, 2, 64), (2, 2, 64), (2, 2, 128)])
def test_weight_first_rejects_w8a16_before_compile(
    dtype, capability, choice, monkeypatch
):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import execute

    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: capability)

    # Metadata alone suffices: rejecting a dense-only tactic must precede CUDA,
    # DLPack conversion and dispatch to the scaled launch signature.
    def tensor(shape, element_type):
        return SimpleNamespace(shape=shape, dtype=element_type, device="cuda:0")

    inputs = [
        tensor((8, 2048), dtype),
        tensor((8, 10), torch.int32),
        tensor((8, 10), torch.float32),
        tensor((19, 1024, 2048), torch.float8_e4m3fn),
        tensor((19, 2048, 512), torch.float8_e4m3fn),
        tensor((80, 512), dtype),
        tensor((80, 2048), dtype),
        tensor((8, 2048), dtype),
        True,
        *[tensor((1,), torch.float32) for _ in range(3)],
    ]
    config = SimpleNamespace(
        activation=SwiGLU(), finalize=SimpleNamespace(do_finalize=True)
    )
    with pytest.raises(ValueError, match="requires dense BF16/FP16"):
        execute(inputs, config, ("warp_mma", *choice, True))


@pytest.mark.parametrize("dtype", [QuantFormat.BF16, QuantFormat.FP16])
@pytest.mark.parametrize("tokens", [4, 8])
@pytest.mark.parametrize("activation", [SwiGLU(), GeGLU()])
@pytest.mark.parametrize("capability", [(8, 0), (9, 0), (12, 0)])
def test_weight_first_admission_and_tuning_identity(
    dtype, tokens, activation, capability, monkeypatch
):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: capability)
    monkeypatch.setattr(
        "flashinfer.fused_moe.backends.frost_low_latency.runner.version",
        lambda _: "4.8.0",
    )
    config = MoEConfig(
        RoutingConfig(19, 10),
        QuantConfig(weight=dtype, activation=dtype, output=dtype),
        ExpertConfig(512),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    runner.check_support()
    runner.build()
    choices = runner.get_valid_tactics([torch.empty(tokens, 2048)], None)
    extra = [t for t in choices if len(t) == 5]
    eligible = capability == (12, 0) and tokens == 8 and activation == SwiGLU()
    assert bool(extra) == eligible
    if eligible:
        assert len(extra) == 4
        assert all(t[-1] is True and t[:-1] in choices for t in extra)
    assert choices[0][0] == "ffma"
    # Winners persisted by either the earlier PR or current main must be tuned
    # again when the additional serial weight-first choices become available.
    old = (
        {"bf16-grouped-tc-v6", "bf16-grouped-tc-v7", "bf16-grouped-tc-w8-batch-v15"}
        if dtype is QuantFormat.BF16
        else {
            "fp16-grouped-tc-v4",
            "fp16-grouped-tc-v5",
            "fp16-grouped-tc-w8-batch-v13",
        }
    )
    assert not old.intersection(runner._cache_key_extras())


@pytest.mark.parametrize("tokens", [2, 4, 8])
@pytest.mark.parametrize("activation", [SwiGLU(), SiTU(), ReLU2(), GeGLU()])
@pytest.mark.parametrize("capability", [(9, 0), (10, 3), (12, 0)])
def test_fp8_weight_first_admission_and_tuning_identity(
    tokens, activation, capability, monkeypatch
):
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: capability)
    monkeypatch.setattr(
        "flashinfer.fused_moe.backends.frost_low_latency.runner.version",
        lambda _: "4.8.0",
    )
    config = MoEConfig(
        RoutingConfig(19, 10),
        QuantConfig(
            weight=QuantFormat.FP8PerTensor,
            activation=QuantFormat.FP8PerTensor,
            output=QuantFormat.BF16,
        ),
        ExpertConfig(512),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    runner.check_support()
    runner.build()
    choices = runner.get_valid_tactics([torch.empty(tokens, 2048)], None)
    extra = [t for t in choices if len(t) == 5]
    eligible = (
        capability == (12, 0)
        and tokens in (4, 8)
        and (
            activation in (SwiGLU(), SiTU()) or (tokens == 8 and activation == ReLU2())
        )
    )
    assert bool(extra) == eligible
    if eligible:
        assert len(extra) == 2
        assert all(t[-1] is True and t[:-1] in choices for t in extra)
    assert choices[0][0] == "ffma"
    # Invalidate persisted winners from both the old child and the newer main
    # cooperative schedule; neither measured this integrated candidate set.
    assert not {
        "fp8-grouped-tc-v3",
        "fp8-grouped-tc-v4",
        "fp8-grouped-tc-w8-batch-v5",
    }.intersection(runner._cache_key_extras())


def _fp8_metadata(tokens, activation):
    def tensor(shape, dtype):
        stride, strides = 1, []
        for dim in reversed(shape):
            strides.insert(0, stride)
            stride *= dim
        return SimpleNamespace(
            shape=shape, dtype=dtype, device="cuda:0", stride=lambda: tuple(strides)
        )

    gated = 2 if activation.is_gated else 1
    fp8, bf16 = torch.float8_e4m3fn, torch.bfloat16
    return [
        tensor((tokens, 2048), fp8),
        tensor((tokens, 10), torch.int32),
        tensor((tokens, 10), torch.float32),
        tensor((19, gated * 512, 2048), fp8),
        tensor((19, 2048, 512), fp8),
        tensor((tokens * 10, 512), fp8),
        tensor((tokens * 10, 2048), bf16),
        tensor((tokens, 2048), bf16),
        True,
        *[tensor((19,), torch.float32) for _ in range(3)],
    ]


@pytest.mark.parametrize(
    "tokens,activation",
    [(4, SwiGLU()), (8, SwiGLU()), (4, SiTU()), (8, SiTU()), (8, ReLU2())],
)
@pytest.mark.parametrize("group", [2, 4])
def test_fp8_weight_first_execute_binds_live_buffers(
    tokens, activation, group, monkeypatch
):
    pytest.importorskip("cutlass", minversion="4.8")
    from cutlass import cute
    from flashinfer.fused_moe.backends.frost_low_latency import runtime
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import (
        execute,
        launch_fp8,
    )

    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda _: (12, 0))
    monkeypatch.setattr(torch.cuda, "device", lambda _: nullcontext())
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(cute.runtime, "from_dlpack", lambda tensor: tensor)
    monkeypatch.setattr(cute.runtime, "make_fake_stream", lambda **_: None)
    cache, compiled, launched = {}, [], []
    monkeypatch.setattr(runtime, "_COMPILED", cache)

    def compile_pair(function, *args, **kwargs):
        assert function is launch_fp8
        compiled.append((args, kwargs))
        return lambda *live: launched.append(live)

    monkeypatch.setattr(cute, "compile", compile_pair)
    config = SimpleNamespace(
        activation=activation, finalize=SimpleNamespace(do_finalize=True)
    )
    old = ("warp_mma", group, 2, 128)
    for inputs in (
        _fp8_metadata(tokens, activation),
        _fp8_metadata(tokens, activation),
    ):
        expected = tuple(inputs[i] for i in (0, 3, 1, 5, 4, 6, 2, 7, 9, 10, 11))
        for tactic in (old, (*old, True)):
            assert execute(inputs, config, tactic) is inputs[7]
            assert len(launched[-1]) == 11
            assert all(a is b for a, b in zip(launched[-1], expected, strict=True))
    assert len(cache) == len(compiled) == 2
    assert {key[4] for key in cache} == {old, (*old, True)}
    assert not compiled[0][1].get("weight_first", False)
    assert compiled[1][1]["weight_first"] is True
    assert all(compiled[0][0][i] is launched[0][i] for i in range(11))
    assert all(launched[0][i] is not launched[2][i] for i in range(11))


@pytest.mark.parametrize(
    "case",
    [
        "tokens2",
        "hidden",
        "intermediate",
        "topk",
        "activation",
        "raw",
        "output",
        "tile",
        "relu2_t4",
        "sm90",
    ],
)
def test_fp8_weight_first_rejects_before_cuda(case, monkeypatch):
    pytest.importorskip("cutlass", minversion="4.8")
    from flashinfer.fused_moe.backends.frost_low_latency.tc.runtime import execute

    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability",
        lambda _: (9, 0) if case == "sm90" else (12, 0),
    )
    activation = ReLU2() if case == "relu2_t4" else SwiGLU()
    inputs = _fp8_metadata(2 if case == "tokens2" else 4, activation)
    if case == "hidden":
        inputs[0].shape = (4, 256)
    elif case == "intermediate":
        inputs[5].shape = (40, 128)
    elif case == "topk":
        inputs[1].shape = (4, 8)
    elif case == "activation":
        activation = SwiGLU(alpha=1.702)
    elif case == "output":
        inputs[7].dtype = torch.float16
    config = SimpleNamespace(
        activation=activation, finalize=SimpleNamespace(do_finalize=case != "raw")
    )
    tactic = ("warp_mma", 8 if case == "tile" else 2, 2, 128, True)
    # Rejection must occur before entering torch.cuda.device on this CPU test.
    with pytest.raises(ValueError, match="requires"):
        execute(inputs, config, tactic)
