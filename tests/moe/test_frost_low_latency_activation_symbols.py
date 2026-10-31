# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Large activation constants must compile without aliasing cached kernels."""

import pytest
import torch

from flashinfer.fused_moe import MoEActivationPack, PowLU, SwiGLU, SwiGLUStep
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe import test_frost_low_latency_bf16 as dense
from tests.moe import test_frost_low_latency_fp8_per_tensor as w8a8
from tests.moe import test_frost_low_latency_w8a16 as w8a16
from tests.moe import test_frost_low_latency_mxfp4_w4a16 as mxw4
from tests.moe import test_frost_low_latency_w4a16 as nvw4


@pytest.mark.parametrize(
    "family", ["bf16", "fp16", "w8a8", "w8a16", "mxfp4_w4a16", "nvfp4_w4a16"]
)
def test_large_activation_constants_compile_and_keep_graph_specializations(family):
    captures = []
    for activation in (
        SwiGLU(alpha=1.702, beta=1.0),  # Default limit is FLT_MAX: e+38 in a name.
        SwiGLU(alpha=1.702, beta=-1.0),
        PowLU(m=1.5, limit=1e20),
        SwiGLUStep(limit=1e20),
    ):
        if family in ("bf16", "fp16"):
            dtype = torch.bfloat16 if family == "bf16" else torch.float16
            cfg, x, w1, w2, ids, scores, pack = dense._inputs(activation, dtype=dtype)
            expected = dense._reference(x, w1, w2, ids, scores, activation, True)[1]
        else:
            module = {
                "w8a8": w8a8,
                "w8a16": w8a16,
                "mxfp4_w4a16": mxw4,
                "nvfp4_w4a16": nvw4,
            }[family]
            cfg, x, ids, scores, pack, view = module._inputs(activation, index=1)
            expected = module._reference(x, ids, scores, view, activation, True)[1]
        runner = FrostLowLatencyRunner(cfg, x.device)
        runner.check_support()
        runner.build()
        inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
        tactics = [("ffma", 1, 4)]
        # Exercise the second shared FC1 implementation where route reuse is
        # admitted; other architectures still exercise the single-route body.
        grouped = ("ffma", 1, 4, 2)
        if grouped in runner.get_valid_tactics(inputs, None):
            tactics.append(grouped)
        for tactic in tactics:
            actual = runner.forward(inputs, tactic)
            w8a8._check(actual, expected)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = runner.forward(inputs, tactic).clone()
            # Retain each call's tensors, and replay earlier parameter choices
            # after compiling later ones to detect accidental kernel aliasing.
            captures.append((graph, output, expected, inputs))
    for graph, output, expected, _inputs in captures:
        graph.replay()
        w8a8._check(output, expected)
