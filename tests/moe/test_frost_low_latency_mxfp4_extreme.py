# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finite MXFP4 dots must not overflow while expanding extreme E8M0 weights."""

import pytest
import torch

from flashinfer.fused_moe import (
    FrostLowLatencyConfig,
    MoEActivationPack,
    MoEWeightPack,
    SiLU,
    SwiGLU,
)
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency_bf16 import _activation
from tests.moe.test_frost_low_latency_mxfp4_w4a16 import _inputs


@pytest.mark.parametrize("activation", [SiLU(), SwiGLU()])
@pytest.mark.parametrize("stage", ["fc1", "fc2"])
@pytest.mark.parametrize("word,mean_weight", [(0x77, 6.0), (0x07, 3.0), (0xF7, 0.0)])
@pytest.mark.parametrize(
    "tactic", [("ffma", 2, 4), ("ffma", 2, 4, 8, 8), ("ffma", 2, 4, 2)]
)
def test_mxfp4_extreme_scale_finite_output(
    stage, word, mean_weight, tactic, activation
):
    cfg, x, ids, scores, pack, view = _inputs(activation, index=1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    ids.zero_()  # Exercise the route-group body rather than its single-route fallback.
    scores.fill_(1.0)
    logical_i = cfg.experts.intermediate_size
    branches = 2 if activation.is_gated else 1
    physical_i = view["fc1_expert_weights"].shape[1] // branches
    for prefix in ("fc1", "fc2"):
        weights = view[prefix + "_expert_weights"]
        weights.zero_()
        payload = word if prefix == stage else 0x22
        if prefix == "fc1":
            for branch in range(branches):
                weights[:, branch * physical_i : branch * physical_i + logical_i].fill_(
                    payload
                )
        else:
            weights[..., : logical_i // 2].fill_(payload)
        view[prefix + "_block_scales"].fill_(127)
        view[prefix + "_dequant_scale"].fill_(1.0)
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    if tactic not in runner.get_valid_tactics(inputs, None):
        pytest.skip(
            "the current architecture does not advertise this load/group tactic"
        )
    x.fill_(2.0**-120)
    view[stage + "_block_scales"].fill_(254)
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic).clone()
    # The same captured kernels must read each new scale and input value.
    for scale_byte, input_exponent in (
        (254, -120),
        (253, -119),
        (254, -133),
        (254, None),
        (127, -8),
    ):
        input_value = 0.0 if input_exponent is None else 2.0**input_exponent
        x.fill_(input_value)
        view[stage + "_block_scales"].fill_(scale_byte)
        # A scalar FP64 reference avoids materializing the overflowing FP32
        # weight; preserve the production BF16 preactivation and expert casts.
        weight = mean_weight * 2.0 ** (scale_byte - 127)
        preactivation = x.shape[1] * input_value * (weight if stage == "fc1" else 1.0)
        typed = torch.tensor(
            preactivation, dtype=torch.float64, device=x.device
        ).bfloat16()
        mid = torch.nn.functional.silu(typed.float())
        if activation.is_gated:
            mid = mid * typed.float()
        mid = mid.bfloat16()
        expert = (
            (
                mid.double()
                * cfg.experts.intermediate_size
                * (weight if stage == "fc2" else 1.0)
            )
            .float()
            .bfloat16()
        )
        expected = (expert.float() * cfg.routing.top_k).bfloat16().expand_as(output)
        assert torch.isfinite(expected).all()
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0)


@pytest.mark.parametrize(
    "tactic", [("ffma", 2, 4), ("ffma", 2, 4, 8, 8), ("ffma", 2, 4, 2)]
)
def test_mxfp4_fc2_mixed_scale_rows_live_graph(tactic):
    cfg, x, ids, scores, pack, view = _inputs(SiLU(), index=1)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    ids.zero_()
    scores.fill_(1.0)
    logical_i = cfg.experts.intermediate_size
    x.fill_(2.0**-120)
    for prefix in ("fc1", "fc2"):
        view[prefix + "_expert_weights"].zero_()
        view[prefix + "_block_scales"].fill_(127)
        view[prefix + "_dequant_scale"].fill_(1.0)
    view["fc1_expert_weights"][:, :logical_i].fill_(0x22)
    weights = view["fc2_expert_weights"]
    scales = view["fc2_block_scales"]
    weights[..., : logical_i // 2].fill_(0x22)
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    if tactic not in runner.get_valid_tactics(inputs, None):
        pytest.skip("the current architecture does not advertise this tactic")
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic).clone()
    typed = torch.tensor(
        x.shape[1] * 2.0**-120, dtype=torch.float64, device=x.device
    ).bfloat16()
    mid = torch.nn.functional.silu(typed.float()).bfloat16()
    last_block = logical_i // 32 - 1
    # Move the exceptional scale between output rows and K blocks without
    # recapturing. Include normal, NaN and E8M0's minimum-scale transitions.
    for row, block, byte in (
        (0, 0, 254),
        (3, last_block, 253),
        (x.shape[1] - 1, 0, 254),
        (x.shape[1] - 1, last_block, 127),
        (3, last_block, 255),
        (0, 0, 0),
    ):
        scales.fill_(127)
        weights[..., : logical_i // 2].fill_(0x22)
        scales[0, row, block] = byte
        weights[0, row, block * 16 : (block + 1) * 16].fill_(0x77)
        expert = (mid.double() * logical_i).expand(x.shape[1]).clone()
        scale = float("nan") if byte == 255 else 2.0 ** (byte - 127)
        expert[row] = mid.double() * (logical_i - 32 + 32 * 6.0 * scale)
        expected = (
            (expert.float().bfloat16().float() * cfg.routing.top_k)
            .bfloat16()
            .expand_as(output)
        )
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(output, expected, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("activation", [SiLU(), SwiGLU()])
@pytest.mark.parametrize("hidden", [256, 2048])
@pytest.mark.parametrize(
    "tactic", [("ffma", 2, 4), ("ffma", 2, 4, 8, 8), ("ffma", 2, 4, 2)]
)
def test_mxfp4_fc1_mixed_scale_rows_live_graph(tactic, activation, hidden):
    cfg, x, ids, scores, pack, view = _inputs(activation, index=1)
    logical_i = cfg.experts.intermediate_size
    branches = 2 if activation.is_gated else 1
    experts = cfg.routing.num_experts
    x = torch.empty(x.shape[0], hidden, dtype=x.dtype, device=x.device)
    view = FrostLowLatencyConfig.prepare_mxfp4_weight_only_weights(
        torch.zeros(
            experts,
            branches * logical_i,
            hidden // 2,
            dtype=torch.uint8,
            device=x.device,
        ),
        torch.ones(experts, branches * logical_i, hidden // 32, device=x.device).to(
            torch.float8_e8m0fnu
        ),
        torch.zeros(
            experts, hidden, logical_i // 2, dtype=torch.uint8, device=x.device
        ),
        torch.ones(experts, hidden, logical_i // 32, device=x.device).to(
            torch.float8_e8m0fnu
        ),
        activation=activation,
    )
    pack = MoEWeightPack()
    pack.prepare_for("frost_low_latency", view)
    runner = FrostLowLatencyRunner(cfg, x.device)
    runner.check_support()
    runner.build()
    # Unique experts also exercise the grouped tactic's single-route fallback.
    assert ids.unique().numel() == ids.numel()
    scores.fill_(1.0)
    physical_i = view["fc1_expert_weights"].shape[1] // branches
    input_value = 2.0**-120
    x.fill_(input_value)
    for prefix in ("fc1", "fc2"):
        view[prefix + "_expert_weights"].zero_()
        view[prefix + "_block_scales"].fill_(127)
        view[prefix + "_dequant_scale"].fill_(1.0)
    weights = view["fc1_expert_weights"]
    scales = view["fc1_block_scales"]
    for branch in range(branches):
        weights[:, branch * physical_i : branch * physical_i + logical_i].fill_(0x22)
    view["fc2_expert_weights"][..., : logical_i // 2].fill_(0x22)
    inputs = runner.pack_inputs(MoEActivationPack(x, None, ids, scores), pack)
    if tactic not in runner.get_valid_tactics(inputs, None):
        pytest.skip("the current architecture does not advertise this tactic")
    runner.forward(inputs, tactic)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = runner.forward(inputs, tactic).clone()
    last_block = x.shape[1] // 32 - 1
    patterns = [
        [(0, 0, 0, 254)],
        [(0, logical_i - 1, last_block, 253)],
        [(0, 3, last_block, 255)],
        [(0, 0, 0, 0)],
        [],
    ]
    if activation.is_gated:
        patterns[2:2] = [
            [(1, 3, last_block, 254)],
            [(0, 0, 0, 253), (1, 0, last_block, 254)],
            [(0, 0, 0, 254), (1, logical_i - 1, 0, 253)],
        ]
    for changes in patterns:
        scales.fill_(127)
        for branch in range(branches):
            weights[:, branch * physical_i : branch * physical_i + logical_i].fill_(
                0x22
            )
        # FP64 forms the finite dot without first expanding the huge FP32 weight.
        preactivation = torch.full(
            (branches, logical_i),
            x.shape[1] * input_value,
            dtype=torch.float64,
            device=x.device,
        )
        for branch, row, block, byte in changes:
            physical_row = branch * physical_i + row
            scales[:, physical_row, block] = byte
            weights[:, physical_row, block * 16 : (block + 1) * 16].fill_(0x77)
            scale = float("nan") if byte == 255 else 2.0 ** (byte - 127)
            preactivation[branch, row] += input_value * 32 * (6.0 * scale - 1.0)
        typed = preactivation.bfloat16().float()
        # The first FC1 half is linear/up; SwiGLU applies SiLU to the second
        # (gate) half. Unequal branch scales make this ordering observable.
        mid = _activation(
            typed[0], typed[1] if activation.is_gated else None, activation
        )
        expert = mid.bfloat16().double().sum().float().bfloat16()
        expected = (expert.float() * cfg.routing.top_k).bfloat16().expand_as(output)
        output.fill_(float("nan"))
        graph.replay()
        torch.testing.assert_close(
            output,
            expected,
            atol=0,
            rtol=0,
            equal_nan=True,
            msg=lambda message: f"FC1 scale changes {changes}: {message}",
        )
