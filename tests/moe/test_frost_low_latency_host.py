# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm FFI binding, ownership and stream isolation across Frost pipelines."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import weakref

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExpertConfig,
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SwiGLU,
    TrtllmFp4Config,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _assert_near, _quant, _require_gpu
from tests.moe.test_frost_low_latency_bf16 import _inputs, _reference as bf16_reference
from tests.moe.test_frost_low_latency_mx_activations import reference as mx_reference
from tests.moe.test_frost_low_latency_nvfp4 import reference as nvfp4_reference
from tests.moe.test_frost_low_latency_fp8_per_tensor import _reference as fp8_reference


def make_case(fmt, tokens, finalize, seed):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    if torch.cuda.get_device_capability() == (9, 0) and fmt not in (
        QuantFormat.BF16,
        QuantFormat.FP8PerTensor,
    ):
        pytest.skip("SM90 Frost admits dense16 and per-tensor FP8 FMA only")
    torch.manual_seed(seed)
    activation = SwiGLU()
    if fmt == QuantFormat.BF16:
        cfg, q, w1, w2, ids, scores, wp = _inputs(
            activation, tokens=tokens, hidden=512, intermediate=128
        )
        # _inputs has a reproducible seed; alter storage contents independently.
        q.mul_(1 + seed % 3)
        prepared_w1 = wp.get_view("frost_low_latency")["w1_weight"]
        prepared_w1.mul_(1 + seed % 2)
        if prepared_w1.data_ptr() != w1.data_ptr():
            w1.mul_(1 + seed % 2)
        act = MoEActivationPack(q, None, ids, scores)
        reference = lambda: bf16_reference(q, w1, w2, ids, scores, activation, True)
    else:
        h, i, e, k = 512, 128, 5, 3
        w1 = (torch.randn(e, 2 * i, h, device="cuda") * 0.08).bfloat16()
        w2 = (torch.randn(e, h, i, device="cuda") * 0.08).bfloat16()
        x = torch.randn(tokens, h, device="cuda").bfloat16()
        ids = (
            torch.arange(tokens * k, device="cuda", dtype=torch.int32).reshape(
                tokens, k
            )
            % e
        )
        scores = torch.rand(tokens, k, device="cuda")
        quant = QuantConfig(
            weight=fmt,
            activation=fmt
            if fmt in (QuantFormat.NVFP4, QuantFormat.FP8PerTensor)
            else QuantFormat.MXFP8,
            **({"nvfp4_4over6": None} if fmt == QuantFormat.NVFP4 else {}),
        )
        cfg = MoEConfig(
            RoutingConfig(e, k),
            quant,
            ExpertConfig(i),
            activation=activation,
            backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        )
        if fmt == QuantFormat.NVFP4:
            view = FrostLowLatencyConfig.prepare_nvfp4_weights(
                w1, w2, activation=activation
            )
            q, sf = TrtllmFp4Config.prepare_activations(x, quant=quant)
            reference = lambda: nvfp4_reference(
                q, sf, view, ids, scores, activation, True
            )
        elif fmt == QuantFormat.FP8PerTensor:
            view = FrostLowLatencyConfig.prepare_fp8_per_tensor_weights(
                w1,
                w2,
                hidden_states_scale_global=32.0,
                intermediate_scale_global=128.0,
                num_local_experts=e,
                hidden_size=h,
                intermediate_size=i,
                activation=activation,
            )
            q, sf = FrostLowLatencyConfig.prepare_fp8_per_tensor_activations(
                x, hidden_states_scale_global=32.0
            )
            reference = lambda: fp8_reference(q, ids, scores, view, activation, True)
        else:

            def weight(w):
                if fmt == QuantFormat.MXFP4:
                    return (
                        torch.randint(
                            0,
                            256,
                            (*w.shape[:-1], w.shape[-1] // 2),
                            device="cuda",
                            dtype=torch.uint8,
                        ),
                        torch.full(
                            (*w.shape[:-1], w.shape[-1] // 32),
                            120,
                            device="cuda",
                            dtype=torch.uint8,
                        ),
                    )
                return _quant(w)[1:]

            w1, s1 = weight(w1)
            w2, s2 = weight(w2)
            view = FrostLowLatencyConfig.prepare_weights(
                w1, s1, w2, s2, activation=activation
            )
            dx, q, sf = _quant(x)
            reference = lambda: mx_reference(
                dx, w1, s1, w2, s2, ids, scores, activation, True
            )
        act = MoEActivationPack(q, sf, ids, scores)
        wp = MoEWeightPack({"frost_low_latency": view})
    cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
    runner = FrostLowLatencyRunner(cfg, q.device)
    runner.check_support()
    runner.build()
    return runner, runner.pack_inputs(act, wp), reference


PATHS = [
    (QuantFormat.FP8PerTensor, ("ffma", 1, 4)),
    (QuantFormat.FP8PerTensor, ("ffma", 2, 8)),
    (QuantFormat.BF16, ("ffma", 1, 4)),
    (QuantFormat.NVFP4, ("nvfp4_ffma", 2)),
    (QuantFormat.NVFP4, ("nvfp4_ffma", 4)),
    *(
        (fmt, tactic)
        for fmt in (QuantFormat.MXFP4, QuantFormat.MXFP8)
        for tactic in (128, 256, ("ffma", 1))
    ),
]


@pytest.mark.parametrize("fmt,tactic", PATHS)
@pytest.mark.parametrize("tokens", [1, 2, 4])
@pytest.mark.parametrize("finalize", [False, True])
def test_warm_ffi_fresh_bindings_and_concurrent_streams(
    fmt, tactic, tokens, finalize, monkeypatch
):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    if torch.cuda.get_device_capability() == (12, 0) and not isinstance(tactic, tuple):
        pytest.skip("SM120 supports Frost FMA tactics only")
    first, a, reference_a = make_case(fmt, tokens, finalize, 9301)
    second, b, reference_b = make_case(fmt, tokens, finalize, 9302)
    first.forward(a, tactic)
    expected = [
        reference_a()[1 if finalize else 0],
        reference_b()[1 if finalize else 0],
    ]
    torch.cuda.synchronize()
    from cutlass import cute
    from flashinfer.fused_moe.backends.frost_low_latency import runtime
    from flashinfer.fused_moe.backends.frost_low_latency.fma import runtime as mx
    from flashinfer.fused_moe.backends.frost_low_latency.fma.bf16 import runtime as bf16
    from flashinfer.fused_moe.backends.frost_low_latency.fma.nvfp4 import (
        runtime as nvfp4,
    )

    from flashinfer.fused_moe.backends.frost_low_latency.fma import (
        fp8_per_tensor as fp8,
    )

    def unexpected(*args, **kwargs):
        raise AssertionError("warm calls must not compile or construct DLPack wrappers")

    monkeypatch.setattr(cute, "compile", unexpected)
    for module in (runtime, mx, bf16, nvfp4, fp8):
        monkeypatch.setattr(module, "from_dlpack", unexpected)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())

    def run(runner, inputs, stream):
        with torch.cuda.stream(stream):
            for _ in range(8):
                output = runner.forward(inputs, tactic)
        return output

    # Both runners share one immutable compiled entry, but different weights,
    # inputs, output/workspace pointers and thread-local current CUDA streams.
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(run, r, inp, s)
            for r, inp, s in zip((first, second), (a, b), streams, strict=True)
        ]
        outputs = [f.result(timeout=30) for f in futures]
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for out, ref in zip(outputs, expected, strict=True):
        _assert_near(out if finalize else out[0], ref)
    graphs = [torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()]
    for graph, runner, inputs, stream in zip(
        graphs, (first, second), (a, b), streams, strict=True
    ):
        with torch.cuda.graph(graph, stream=stream):
            runner.forward(inputs, tactic)
    # Update existing input storage after capture; only route IDs and scores
    # change, so all three independent numerical references remain applicable.
    id_index, score_index = (
        (1, 2) if fmt in (QuantFormat.BF16, QuantFormat.FP8PerTensor) else (2, 3)
    )
    for inputs in (a, b):
        inputs[id_index].add_(1).remainder_(5)
        if finalize:
            inputs[score_index].mul_(0.5)
    torch.cuda.synchronize()
    for graph, stream in zip(graphs, streams, strict=True):
        with torch.cuda.stream(stream):
            graph.replay()
    for stream in streams:
        torch.cuda.current_stream().wait_stream(stream)
    for out, ref in zip(outputs, (reference_a, reference_b), strict=True):
        _assert_near(out if finalize else out[0], ref()[1 if finalize else 0])


@pytest.mark.parametrize("fmt,tactic", PATHS)
def test_compiled_entry_does_not_retain_call_storage(fmt, tactic):
    _require_gpu(allow_sm120=True, allow_sm90=True)
    if torch.cuda.get_device_capability() == (12, 0) and not isinstance(tactic, tuple):
        pytest.skip("SM120 supports Frost FMA tactics only")
    runner, inputs, reference = make_case(fmt, 1, True, 9401)
    refs = [weakref.ref(v) for v in inputs if isinstance(v, torch.Tensor)]
    runner.forward(inputs, tactic)
    torch.cuda.synchronize()
    del reference, inputs, runner
    gc.collect()
    assert all(ref() is None for ref in refs)
