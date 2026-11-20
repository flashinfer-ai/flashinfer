# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Low-token MX MoE: independent quantized reference and changing graph inputs."""

from dataclasses import replace
from importlib.metadata import version
from packaging.version import Version

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
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
    TrtllmFp8BlockConfig,
)
from flashinfer.autotuner import autotune
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner


def _quant(value):
    grouped = value.float().reshape(*value.shape[:-1], -1, 32)
    exponent = (
        torch.ceil(
            torch.log2(
                grouped.abs().amax(-1).clamp_min(torch.finfo(torch.float32).tiny)
                / 448.0
            )
        )
        + 127
    ).clamp(0, 254)
    scale = torch.exp2(exponent - 127)
    q = (grouped / scale[..., None]).to(torch.float8_e4m3fn)
    return (
        (q.float() * scale[..., None]).reshape(value.shape),
        q.reshape(value.shape),
        exponent.byte(),
    )


def _decode(w, sf):
    if w.dtype == torch.uint8:
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device=w.device,
            dtype=torch.float64,
        )
        values = torch.stack(
            (lut[(w & 15).long()], lut[(w >> 4).long()]), dim=-1
        ).reshape(*w.shape[:-1], -1)
    else:
        values = w.double()
    return (
        values.reshape(*values.shape[:-1], -1, 32)
        * torch.exp2(sf.double() - 127)[..., None]
    ).reshape(values.shape)


def _reference(x, w1, s1, w2, s2, ids, weights, activation):
    routes = []
    for token in range(ids.shape[0]):
        for expert in ids[token].tolist():
            gu = (x[token].double() @ _decode(w1[expert], s1[expert]).T).float()
            up, gate = gu.chunk(2)
            gate = gate.clamp_max(activation.limit)
            up = up.clamp(-activation.limit, activation.limit)
            mid = (
                gate
                / (1 + torch.exp(-gate * activation.alpha))
                * (up + activation.beta)
            )
            dq, _, _ = _quant(mid[None, :])
            routes.append((dq.double() @ _decode(w2[expert], s2[expert]).T).bfloat16())
    raw = torch.cat(routes)
    value = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        # FP64 models one correctly rounded FP32 FMA, in slot order.
        value = (
            raw[slot :: ids.shape[1]].double() * weights[:, slot, None].double()
            + value.double()
        ).float()
    return raw, value.bfloat16()


def _assert_near(actual, expected):
    diff = actual.double() - expected.double()
    assert torch.isfinite(actual).all()
    assert float(diff.norm() / expected.double().norm().clamp_min(1e-30)) < 0.01
    assert (
        float(diff.abs().max() / expected.double().abs().max().clamp_min(1e-30)) < 0.03
    )


def _require_gpu(*, allow_sm120=False, allow_sm90=False, allow_sm80=False):
    architectures = (
        (
            (10, 0),
            (10, 3),
            (10, 7),
        )
        + (((12, 0),) if allow_sm120 else ())
        + (((9, 0),) if allow_sm90 else ())
        + (((8, 0),) if allow_sm80 else ())
    )
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability() not in architectures
    ):
        pytest.skip(f"requires one of {architectures}")
    pytest.importorskip("cutlass")
    if Version(version("nvidia-cutlass-dsl")) < Version("4.8"):
        pytest.skip("requires nvidia-cutlass-dsl >= 4.8")


def test_frost_cross_backend_autotuning():
    """Exercise real packing, native/Frost tuning, and the resulting graph."""
    _require_gpu()
    torch.manual_seed(318)
    e, h, i, k = 8, 1024, 256, 2
    device = torch.device("cuda", torch.cuda.current_device())
    w1 = (torch.randn(e, 2 * i, h, device=device) / h**0.5).bfloat16()
    w2 = (torch.randn(e, h, i, device=device) / i**0.5).bfloat16()
    _, q1, s1 = _quant(w1)
    _, q2, s2 = _quant(w2)
    quant = QuantConfig(weight=QuantFormat.MXFP8, activation=QuantFormat.MXFP8)
    activation = SwiGLU()
    weights = MoEWeightPack()
    weights.prepare_for(
        "frost_low_latency", FrostLowLatencyConfig.prepare_weights(q1, s1, q2, s2)
    )
    weights.prepare_for(
        "trtllm_fp8_block",
        TrtllmFp8BlockConfig.prepare_weights(
            w1,
            w2,
            quant=quant,
            num_local_experts=e,
            hidden_size=h,
            intermediate_size=i,
            activation=activation,
        ),
    )
    config = MoEConfig(
        RoutingConfig(e, k),
        quant,
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((TrtllmFp8BlockConfig(), FrostLowLatencyConfig())),
    )
    layer = MoELayer(config, device=device)
    assert len(layer.runners) == 2
    _, q, sf = _quant(torch.randn(1, h, device=device).bfloat16())
    ids = torch.arange(k, device=device, dtype=torch.int32)[None, :]
    rw = torch.softmax(torch.randn(1, k, device=device), -1)
    act = MoEActivationPack(q, sf, ids, rw)
    with autotune(True):
        layer(act, weights)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = layer(act, weights)
    for step in range(3):
        dq, qn, sn = _quant(torch.randn(1, h, device=device).bfloat16())
        q.copy_(qn)
        sf.copy_(sn)
        ids.copy_(
            (torch.arange(k, device=device, dtype=torch.int32)[None, :] + step) % e
        )
        graph.replay()
        _, expected = _reference(dq, q1, s1, q2, s2, ids, rw.bfloat16(), activation)
        _assert_near(output, expected)


@pytest.mark.parametrize("weight_format", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize(
    "tokens,hidden,intermediate,topk",
    [(1, 1024, 512, 4), (2, 2048, 576, 6), (4, 1024, 256, 4), (8, 1024, 256, 4)],
)
def test_frost_graph_reference(weight_format, tokens, hidden, intermediate, topk):
    _require_gpu()
    torch.manual_seed(317)
    device = torch.device("cuda", torch.cuda.current_device())
    experts = 8

    def weight(rows, cols):
        if weight_format == QuantFormat.MXFP4:
            return (
                torch.randint(
                    0, 256, (experts, rows, cols // 2), device=device, dtype=torch.uint8
                ),
                torch.randint(
                    118,
                    123,
                    (experts, rows, cols // 32),
                    device=device,
                    dtype=torch.uint8,
                ),
            )
        _, q, s = _quant(
            (torch.randn(experts, rows, cols, device=device) / (cols**0.5)).bfloat16()
        )
        return q, s

    w1, s1 = weight(2 * intermediate, hidden)
    w2, s2 = weight(hidden, intermediate)
    view = FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    config = MoEConfig(
        routing=RoutingConfig(experts, topk),
        quant=QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        experts=ExpertConfig(intermediate),
        activation=SwiGLU(alpha=1.7, beta=1.0, limit=7.0),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        execution=ExecutionConfig(enable_pdl=tokens != 2),
    )
    x = torch.randn(tokens, hidden, device=device).bfloat16()
    _, q, sf = _quant(x)
    ids = (
        torch.arange(tokens * topk, device=device, dtype=torch.int32) % experts
    ).reshape(tokens, topk)
    rw = torch.rand(tokens, topk, device=device)
    # Exercise both routing-weight contracts, raw output and padded I=576.
    for mode in (
        RoutingInputMode.PackedPrecomputed,
        RoutingInputMode.UnpackedPrecomputed,
    ):
        act = MoEActivationPack(q, sf, ids, rw, routing_input_mode=mode)
        for finalize in (True, False):
            cfg = replace(config, finalize=MoEFinalizeConfig(do_finalize=finalize))
            runner = FrostLowLatencyRunner(cfg, device)
            runner.check_support()
            runner.build()
            inputs = runner.pack_inputs(act, pack)
            for tile in runner.get_valid_tactics(inputs, None):
                runner.forward(inputs, tactic=tile)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    output = runner.forward(inputs, tactic=tile)
                for step in range(3):
                    dq, qn, sn = _quant(
                        (torch.randn_like(x.float()) * (1 + step)).bfloat16()
                    )
                    q.copy_(qn)
                    sf.copy_(sn)
                    ids.copy_(
                        (
                            (
                                torch.arange(
                                    tokens * topk, device=device, dtype=torch.int32
                                )
                                + step * 3
                                + experts
                                - 1
                            )
                            % experts
                        ).reshape_as(ids)
                    )
                    # The per-call packed weight tensor is the captured input;
                    # a caller mutating the original pack should call pack_inputs again.
                    inputs[3].copy_(torch.rand_like(inputs[3]))
                    graph.replay()
                    torch.cuda.synchronize()
                    raw, ref = _reference(
                        dq,
                        w1,
                        s1,
                        w2,
                        s2,
                        ids,
                        (
                            inputs[3].bfloat16()
                            if mode == RoutingInputMode.PackedPrecomputed
                            else inputs[3]
                        ),
                        cfg.activation,
                    )
                    _assert_near(
                        output if finalize else output[0], ref if finalize else raw
                    )
                    if not finalize:
                        if mode == RoutingInputMode.PackedPrecomputed:
                            assert output[1].dtype == torch.bfloat16
                        torch.testing.assert_close(
                            output[2],
                            torch.arange(
                                tokens * topk, device=device, dtype=torch.int32
                            ),
                        )
    # Actual unified dispatcher, not just the direct runner.
    layer = MoELayer(config, device=device)
    act = MoEActivationPack(q, sf, ids, rw)
    result = layer(act, pack)
    _, expected = _reference(dq, w1, s1, w2, s2, ids, rw.bfloat16(), config.activation)
    _assert_near(result, expected)
    # Packed FP32 and explicitly narrowed BF16 routing weights must produce
    # identical outputs, even when the conversion moves into combine.
    narrowed = MoEActivationPack(q, sf, ids, rw.bfloat16())
    torch.testing.assert_close(layer(narrowed, pack), result, rtol=0, atol=0)
    # Reuse the compiled geometry with different activation and weight storage.
    # Negating only Down must negate the output; a cached pointer fails this.
    negative = {name: value.clone() for name, value in view.items()}
    negative["w2_weight"].view(torch.uint8).bitwise_xor_(
        0x88 if weight_format == QuantFormat.MXFP4 else 0x80
    )
    other_weights = MoEWeightPack()
    other_weights.prepare_for("frost_low_latency", negative)
    other_act = MoEActivationPack(q.clone(), sf.clone(), ids.clone(), rw.clone())
    negated = layer(other_act, other_weights)
    torch.testing.assert_close(negated, -result, rtol=0, atol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        original_output = layer(act, pack)
        negative_output = layer(other_act, other_weights)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(negative_output, -original_output, rtol=0, atol=0)

    # Rebinding must also hold for the shared FMA executable cache.
    fma_runner = FrostLowLatencyRunner(config, device)
    fma_runner.check_support()
    fma_runner.build()
    original_inputs = fma_runner.pack_inputs(act, pack)
    negative_inputs = fma_runner.pack_inputs(other_act, other_weights)
    fma_runner.forward(original_inputs, tactic=("ffma", 1))
    fma_runner.forward(negative_inputs, tactic=("ffma", 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        original_output = fma_runner.forward(original_inputs, tactic=("ffma", 1))
        negative_output = fma_runner.forward(negative_inputs, tactic=("ffma", 1))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(negative_output, -original_output, rtol=0, atol=0)


def test_frost_capability_and_metadata_rejections():
    with pytest.raises(ValueError):
        FrostLowLatencyConfig(max_num_tokens=9)
    _require_gpu()
    device = torch.device("cuda", torch.cuda.current_device())
    config = MoEConfig(
        RoutingConfig(8, 2),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(256),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(config, device)
    runner.check_support()
    runner.build()
    act = MoEActivationPack(
        torch.zeros(3, 1024, device=device, dtype=torch.float8_e4m3fn),
        torch.zeros(3, 32, device=device, dtype=torch.uint8),
        torch.zeros(3, 2, device=device, dtype=torch.int32),
        torch.ones(3, 2, device=device),
    )
    assert not runner._accepts_shape(act)
    ep = FrostLowLatencyRunner(
        replace(config, experts=ExpertConfig(256, local_num_experts=4)), device
    )
    with pytest.raises(NotImplementedError, match="expert parallelism"):
        ep.check_support()


@pytest.mark.parametrize("weight_format", [QuantFormat.MXFP4, QuantFormat.MXFP8])
def test_frost_fma_large_expert_weight_offsets(weight_format):
    """Both expert strides cross signed 32-bit word indexing at the last expert."""
    _require_gpu()
    if torch.cuda.mem_get_info()[0] < 100 * 1024**3:
        pytest.skip("large expert-offset regression requires 100 GiB free memory")
    mixed = weight_format == QuantFormat.MXFP4
    e, h, i = (896, 7168, 3072) if mixed else (513, 4096, 4096)
    packing = 2 if mixed else 1
    dtype = torch.uint8 if mixed else torch.float8_e4m3fn
    w1 = torch.zeros((e, 2 * i, h // packing), device="cuda", dtype=torch.uint8)
    w2 = torch.zeros((e, h, i // packing), device="cuda", dtype=torch.uint8)
    w1[-1].fill_(0x22 if mixed else 0x38)  # E2M1/E4M3 unit values.
    w2[-1].fill_(0x22 if mixed else 0x38)
    w1, w2 = w1.view(dtype), w2.view(dtype)
    assert w2.numel() * w2.element_size() > 2**33
    s1 = torch.full((e, 2 * i, h // 32), 114, device="cuda", dtype=torch.uint8)
    s2 = torch.full((e, h, i // 32), 114, device="cuda", dtype=torch.uint8)
    weights = MoEWeightPack()
    weights.prepare_for(
        "frost_low_latency", FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    )
    dq, q, sf = _quant(torch.ones((1, h), device="cuda", dtype=torch.bfloat16))
    ids = torch.full((1, 1), e - 1, device="cuda", dtype=torch.int32)
    rw = torch.ones((1, 1), device="cuda")
    cfg = MoEConfig(
        RoutingConfig(e, 1),
        QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(cfg, q.device)
    runner.check_support()
    runner.build()
    inputs = runner.pack_inputs(MoEActivationPack(q, sf, ids, rw), weights)
    runner.forward(inputs, tactic=("ffma", 1))
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic=("ffma", 1))
    _, expected = _reference(dq, w1, s1, w2, s2, ids, rw, cfg.activation)
    assert expected.float().norm() > 0
    for expert in (e - 1, 0, e - 1):
        ids.fill_(expert)
        graph.replay()
        torch.cuda.synchronize()
        if expert == 0:
            assert torch.count_nonzero(output) == 0
        else:
            _assert_near(output, expected)


@pytest.mark.parametrize("stage", ["fc1", "fc2"])
def test_shared_bf16_fma_large_expert_weight_offsets(stage):
    """Exercise shared device bodies without admitting a new public dtype."""
    _require_gpu()
    if torch.cuda.mem_get_info()[0] < 60 * 1024**3:
        pytest.skip("BF16 expert-offset regression requires 60 GiB free memory")
    from cutlass import cute
    from cutlass.cute.runtime import from_dlpack
    from cuda.bindings import driver
    from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc1 import (
        _kernel as fc1,
    )
    from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16.fma_fc2 import (
        _kernel as fc2,
    )

    @cute.jit
    def first(x, w, ids, mid, stream):
        fc1(x, w, ids, mid, 2, "swiglu", True).launch(
            grid=(cute.size(mid) // 16, 1, 1), block=(256, 1, 1), stream=stream
        )

    @cute.jit
    def second(mid, w, ids, scores, out, stream):
        fc2(mid, w, ids, scores, out).launch(
            grid=(cute.size(out) // 4, 1, 1), block=(32, 1, 1), stream=stream
        )

    e, h, i = 513, 4096, 4096
    ids = torch.full((1, 1), e - 1, device="cuda", dtype=torch.int32)
    x = torch.ones((1, h), device="cuda", dtype=torch.bfloat16)
    shape = (e, 2 * i, h) if stage == "fc1" else (e, h, i)
    w = torch.zeros(shape, device="cuda", dtype=torch.bfloat16)
    w[-1].fill_(1 / h)
    assert (e - 1) * w[0].numel() // 4 >= 2**31
    out = torch.empty_like(x)
    scores = torch.ones((1, 1), device="cuda")
    tensors = (x, w, ids, out) if stage == "fc1" else (x, w, ids, scores, out)
    args = tuple(from_dlpack(v) for v in tensors)
    major, minor = torch.cuda.get_device_capability()
    compiled = cute.compile(
        first if stage == "fc1" else second,
        *args,
        driver.CUstream(torch.cuda.current_stream().cuda_stream),
        options=f"--gpu-arch sm_{major}{minor}a",
    )

    def run():
        compiled(*args, driver.CUstream(torch.cuda.current_stream().cuda_stream))

    run()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for expert, scale in ((e - 1, 1), (0, 1), (e - 1, 2)):
        ids.fill_(expert)
        x.fill_(scale)
        graph.replay()
        expected = torch.tensor(float(scale), dtype=torch.float64)
        if stage == "fc1":
            expected = expected * torch.nn.functional.silu(expected)
        if expert == 0:
            expected.zero_()
        torch.testing.assert_close(
            out, torch.full_like(out, expected.item()), rtol=0, atol=0
        )


@pytest.mark.parametrize("installed", ["4.7.1", "4.8.0.dev0", None])
def test_frost_declines_unsupported_compiler(monkeypatch, installed):
    from flashinfer.fused_moe.backends.frost_low_latency import runner as module

    def probe(name):
        if installed is None:
            raise module.PackageNotFoundError(name)
        return installed

    monkeypatch.setattr(module, "version", probe)
    config = MoEConfig(
        RoutingConfig(8, 2),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(256),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    runner = FrostLowLatencyRunner(config, torch.device("cpu"))
    with pytest.raises(NotImplementedError, match="requires nvidia-cutlass-dsl >= 4.8"):
        runner.check_support()


@pytest.mark.parametrize("tactic", [128, ("ffma", 1)])
def test_frost_binds_noncurrent_device(tactic):
    _require_gpu()
    if torch.cuda.device_count() < 2:
        pytest.skip("requires two GPUs to check a noncurrent device")
    previous = torch.cuda.current_device()
    other = (previous + 1) % torch.cuda.device_count()
    if torch.cuda.get_device_capability(other) not in ((10, 0), (10, 3)):
        pytest.skip("second device must be SM100/SM103")
    device = torch.device("cuda", other)
    e, h, i, k = 2, 128, 128, 2
    w1 = torch.full((e, 2 * i, h // 2), 0x11, device=device, dtype=torch.uint8)
    w2 = torch.full((e, h, i // 2), 0x11, device=device, dtype=torch.uint8)
    s1 = torch.full((e, 2 * i, h // 32), 123, device=device, dtype=torch.uint8)
    s2 = torch.full((e, h, i // 32), 123, device=device, dtype=torch.uint8)
    dq, q, sf = _quant(torch.randn(1, h, device=device).bfloat16())
    ids = torch.arange(k, device=device, dtype=torch.int32)[None, :]
    rw = torch.full((1, k), 0.5, device=device)
    weights = MoEWeightPack()
    weights.prepare_for(
        "frost_low_latency", FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    )
    config = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=QuantFormat.MXFP4, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        activation=SwiGLU(alpha=1.0, beta=0.0, limit=10.0),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
    )
    layer = MoELayer(config, device=device)
    act = MoEActivationPack(q, sf, ids, rw)
    layer(act, weights)
    for key, (runner, _) in list(layer._winners.items()):
        layer._winners[key] = (runner, tactic)
    result = layer(act, weights)
    assert torch.cuda.current_device() == previous
    _, reference = _reference(dq, w1, s1, w2, s2, ids, rw.bfloat16(), config.activation)
    _assert_near(result, reference)
