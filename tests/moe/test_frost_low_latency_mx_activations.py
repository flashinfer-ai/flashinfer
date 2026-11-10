# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""MX activation extensions: FP32 activation, microscaling, and graph replay."""

from dataclasses import replace

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
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
    SiTU,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import _assert_near, _decode, _quant, _require_gpu
from tests.moe.test_frost_low_latency_bf16 import ACTIVATIONS, _activation


def reference(x, w1, s1, w2, s2, ids, scores, activation, packed=True):
    rows = []
    for token in range(ids.shape[0]):
        for expert in ids[token].tolist():
            value = (x[token].double() @ _decode(w1[expert], s1[expert]).T).float()
            up, gate = (
                value.chunk(2)
                if activation.is_gated
                else (value, torch.zeros_like(value))
            )
            mid = _activation(up, gate, activation)
            dq, _, _ = _quant(mid[None, :])
            rows.append((dq.double() @ _decode(w2[expert], s2[expert]).T).bfloat16())
    raw = torch.cat(rows)
    effective = scores.bfloat16() if packed else scores
    result = torch.zeros_like(x, dtype=torch.float32)
    for slot in range(ids.shape[1]):
        result = (
            raw[slot :: ids.shape[1]].double() * effective[:, slot, None].double()
            + result.double()
        ).float()
    return raw, result.bfloat16()


@pytest.mark.parametrize("weight_format", [QuantFormat.MXFP4, QuantFormat.MXFP8])
@pytest.mark.parametrize("index,activation", list(enumerate(ACTIVATIONS)))
def test_mx_activation_tactics_graph_and_padding(weight_format, index, activation):
    _require_gpu(allow_sm120=True)
    torch.manual_seed(6100 + index)
    t, h, i, e, k = (1, 2, 4, 8)[index % 4], 256, 96, 5, 3
    device = torch.device("cuda", torch.cuda.current_device())

    def weight(rows, cols):
        if weight_format == QuantFormat.MXFP4:
            return (
                torch.randint(
                    0, 256, (e, rows, cols // 2), dtype=torch.uint8, device=device
                ),
                torch.full(
                    (e, rows, cols // 32), 120, dtype=torch.uint8, device=device
                ),
            )
        _, q, sf = _quant((torch.randn(e, rows, cols, device=device) * 0.08).bfloat16())
        return q, sf

    w1, s1 = weight((2 if activation.is_gated else 1) * i, h)
    w2, s2 = weight(h, i)
    wp = MoEWeightPack()
    wp.prepare_for(
        "frost_low_latency",
        FrostLowLatencyConfig.prepare_weights(
            w1,
            s1,
            w2,
            s2,
            activation=activation,
        ),
    )
    _, q, sf = _quant(torch.randn(t, h, device=device).bfloat16())
    ids = torch.arange(t * k, device=device, dtype=torch.int32).view(t, k) % e
    scores = torch.rand(t, k, device=device)
    config = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        activation=activation,
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=8),)),
        execution=ExecutionConfig(enable_pdl=index % 2 == 0),
    )
    for finalize in (True, False):
        cfg = replace(config, finalize=MoEFinalizeConfig(do_finalize=finalize))
        packed = finalize or index % 2 == 0
        act = MoEActivationPack(
            q,
            sf,
            ids,
            scores,
            routing_input_mode=(
                RoutingInputMode.PackedPrecomputed
                if packed
                else RoutingInputMode.UnpackedPrecomputed
            ),
        )
        runner = FrostLowLatencyRunner(cfg, device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(act, wp)
        for tactic in runner.get_valid_tactics(inputs, None):
            runner.forward(inputs, tactic=tactic)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = runner.forward(inputs, tactic=tactic)
            for step in range(3):
                dx, qn, sn = _quant(
                    (torch.randn(t, h, device=device) * (step + 1)).bfloat16()
                )
                q.copy_(qn)
                sf.copy_(sn)
                ids.copy_(
                    (
                        torch.arange(t * k, device=device, dtype=torch.int32).view(t, k)
                        + step
                    )
                    % e
                )
                scores.copy_(torch.rand_like(scores) - 0.2)
                # Raw packed output owns a BF16 score conversion made in pack_inputs.
                inputs[3].copy_(scores)
                graph.replay()
                raw, expected = reference(
                    dx, w1, s1, w2, s2, ids, scores, activation, packed
                )
                _assert_near(
                    output if finalize else output[0], expected if finalize else raw
                )
                assert torch.count_nonzero(inputs[10][:, i:].float()) == 0
                if not finalize:
                    torch.testing.assert_close(
                        output[2], torch.arange(t * k, device=device, dtype=torch.int32)
                    )


def test_mx_activation_scalar_contract():
    from flashinfer.fused_moe.backends.frost_low_latency.fma.activation_config import (
        lower_activation,
    )

    with pytest.raises(NotImplementedError, match="default parameters"):
        lower_activation(SiTU(gate_scale=2.0))
