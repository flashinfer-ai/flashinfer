# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dense 16-bit FMA format contracts, graph replay and SM120 admission."""

from dataclasses import replace

import pytest
import torch

from flashinfer.fused_moe import (
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEFinalizeConfig,
    MoELayer,
    QuantConfig,
    QuantFormat,
    SwiGLU,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _inputs, _reference


def _check(actual, expected):
    assert actual.dtype == expected.dtype
    assert torch.isfinite(actual).all()
    delta = actual.double() - expected.double()
    assert float(delta.norm() / expected.double().norm().clamp_min(1e-30)) < 0.003
    assert (
        float(delta.abs().max() / expected.double().abs().max().clamp_min(1e-30)) < 0.01
    )


@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_fp16_activation_and_changing_graph_inputs(index, activation):
    cfg, x, w1, w2, ids, scores, pack = _inputs(
        activation,
        tokens=(1, 2, 4, 8)[index % 4],
        hidden=(128, 256, 384)[index % 3],
        intermediate=(32, 96, 160)[index % 3],
        dtype=torch.float16,
        output_dtype=torch.float16,
    )
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    buffers = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    captured = []
    for tactic in (("ffma", 1, 4), ("ffma", 4, 8)):
        runner.forward(buffers, tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = runner.forward(buffers, tactic).clone()
        captured.append((graph, out))
    for step in range(3):
        x.copy_(torch.randn_like(x) * (1 + step))
        ids.copy_((ids + 1) % w1.shape[0])
        scores.copy_(torch.randn_like(scores) * 0.3)
        _, expected, _ = _reference(
            x, w1, w2, ids, scores, activation, True, torch.float16
        )
        for graph, out in captured:
            graph.replay()
            _check(out, expected)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_float16_formats_keep_separate_compiled_entries(packed, finalize):
    # Same geometry, activation and tactic in one retained cache: input/output
    # dtypes must distinguish both the executable and its numerical contract.
    for dtype, output_dtype in (
        (torch.bfloat16, torch.bfloat16),
        (torch.float16, torch.float16),
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float16),
    ):
        cfg, x, w1, w2, ids, scores, pack = _inputs(
            SwiGLU(),
            dtype=dtype,
            output_dtype=output_dtype,
        )
        cfg = replace(cfg, finalize=MoEFinalizeConfig(do_finalize=finalize))
        mode = (
            RoutingInputMode.PackedPrecomputed
            if packed
            else RoutingInputMode.UnpackedPrecomputed
        )
        act = MoEActivationPack(x, None, ids, scores, routing_input_mode=mode)
        result = MoELayer(cfg)(act, pack)
        raw, expected, effective = _reference(
            x, w1, w2, ids, scores, cfg.activation, packed, output_dtype
        )
        if finalize:
            _check(result, expected)
        else:
            _check(result[0], raw)
            torch.testing.assert_close(result[1].float(), effective, rtol=0, atol=0)
            torch.testing.assert_close(
                result[2], torch.arange(ids.numel(), device=x.device, dtype=torch.int32)
            )


def test_fp16_output_requires_dense_formats():
    for weight, activation in (
        (QuantFormat.NVFP4, QuantFormat.NVFP4),
        (QuantFormat.MXFP4, QuantFormat.MXFP8),
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
        (QuantFormat.FP8PerTensor, QuantFormat.FP8PerTensor),
    ):
        assert not FrostLowLatencyRunner.supports_quant(
            QuantConfig(
                weight=weight,
                activation=activation,
                output=QuantFormat.FP16,
            )
        )
    assert FrostLowLatencyConfig.supported(120)


@pytest.mark.parametrize("arch", [(9, 0), (12, 0)])
def test_fma_only_arch_does_not_admit_tensor_core_tactics(monkeypatch, arch):
    cfg, x, _, _, ids, scores, pack = _inputs(SwiGLU(), dtype=torch.float16)
    monkeypatch.setattr("flashinfer.utils.get_compute_capability", lambda device: arch)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    buffers = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    assert all(
        isinstance(tactic, tuple) for tactic in runner.get_valid_tactics(buffers, None)
    )
    with pytest.raises(ValueError, match="Tensor Core"):
        runner.forward(buffers, 128)


def test_sm90_rejects_unqualified_blockscaled_formats(monkeypatch):
    cfg, x, _, _, _, _, _ = _inputs(SwiGLU())
    monkeypatch.setattr(
        "flashinfer.utils.get_compute_capability", lambda device: (9, 0)
    )
    assert FrostLowLatencyConfig.supported(90)
    for weight, activation in (
        (QuantFormat.MXFP4, QuantFormat.MXFP8),
        (QuantFormat.MXFP8, QuantFormat.MXFP8),
        (QuantFormat.NVFP4, QuantFormat.NVFP4),
    ):
        runner = FrostLowLatencyRunner(
            replace(cfg, quant=QuantConfig(weight=weight, activation=activation)),
            x.device,
        )
        with pytest.raises(NotImplementedError, match="SM90 supports"):
            runner.check_support()
