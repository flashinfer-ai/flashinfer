# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unrouted MX/NVFP4 slots, including live routes in a captured graph."""

import pytest
import torch

from flashinfer.fused_moe import (
    BackendOptions,
    ExecutionConfig,
    ExpertConfig,
    FrostLowLatencyConfig,
    Identity,
    MoEActivationPack,
    MoEConfig,
    MoEFinalizeConfig,
    MoEWeightPack,
    QuantConfig,
    QuantFormat,
    RoutingConfig,
)
from flashinfer.fused_moe.api import RoutingInputMode
from flashinfer.fused_moe.backends.frost_low_latency.runner import FrostLowLatencyRunner
from tests.moe.test_frost_low_latency import (
    _assert_near,
    _quant,
    _reference,
    _require_gpu,
)
from tests.moe.test_frost_low_latency_nvfp4 import reference as nvfp4_reference


def _runner_inputs(config, q, sf, ids, scores, view, packed):
    weights = MoEWeightPack()
    weights.prepare_for("frost_low_latency", view)
    runner = FrostLowLatencyRunner(config, q.device)
    runner.check_support()
    runner.build()
    act = MoEActivationPack(
        q,
        sf,
        ids,
        scores.clone(),
        routing_input_mode=(
            RoutingInputMode.PackedPrecomputed
            if packed
            else RoutingInputMode.UnpackedPrecomputed
        ),
    )
    return runner, runner.pack_inputs(act, weights)


def _expected(valid_raw, ids, scores, experts, packed, fused):
    """Reduce independently computed expert rows; exclude unrouted slots."""
    valid = (ids >= 0) & (ids < experts)
    raw = torch.where(valid.reshape(-1, 1), valid_raw, 0)
    # Mask the score before multiplication: zero times NaN/Inf is not zero.
    weights = torch.where(valid, scores, 0)
    weights = weights.bfloat16().float() if packed else weights.float()
    result = torch.zeros(
        (ids.shape[0], raw.shape[1]), device=raw.device, dtype=torch.float32
    )
    for slot in range(ids.shape[1]):
        row = raw[slot :: ids.shape[1]]
        if fused:
            result = (
                row.double() * weights[:, slot, None].double() + result.double()
            ).float()
        else:
            result = result + row.float() * weights[:, slot, None]
    return raw, result.bfloat16()


def _assert_raw_contract(output, ids, scores, experts, packed):
    invalid = ((ids < 0) | (ids >= experts)).reshape(-1)
    assert torch.count_nonzero(output[0][invalid]) == 0
    assert output[1].dtype == (torch.bfloat16 if packed else scores.dtype)
    torch.testing.assert_close(output[1], scores, rtol=0, atol=0, equal_nan=True)
    torch.testing.assert_close(
        output[2],
        torch.arange(ids.numel(), device=ids.device, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("weight_format", [QuantFormat.MXFP8, QuantFormat.MXFP4])
@pytest.mark.parametrize("tokens", [1, 2])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_mx_invalid_scores_after_graph_capture(weight_format, tokens, packed, finalize):
    _require_gpu()
    torch.manual_seed(70960)
    e, h, i, k = 2, 256, 128, 2
    dq, q, sf = _quant(torch.randn(tokens, h, device="cuda") * 0.2)

    def weight(rows, cols):
        if weight_format == QuantFormat.MXFP4:
            return (
                torch.randint(
                    0, 256, (e, rows, cols // 2), device=q.device, dtype=torch.uint8
                ),
                torch.full(
                    (e, rows, cols // 32), 120, device=q.device, dtype=torch.uint8
                ),
            )
        return _quant(torch.randn(e, rows, cols, device=q.device) * 0.05)[1:]

    w1, s1 = weight(2 * i, h)
    w2, s2 = weight(h, i)
    view = FrostLowLatencyConfig.prepare_weights(w1, s1, w2, s2)
    config = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(weight=weight_format, activation=QuantFormat.MXFP8),
        ExpertConfig(i),
        backend=BackendOptions((FrostLowLatencyConfig(),)),
        execution=ExecutionConfig(enable_pdl=tokens == 1),
        finalize=MoEFinalizeConfig(do_finalize=finalize),
    )
    valid_ids = (
        torch.arange(tokens * k, device=q.device, dtype=torch.int32).view(tokens, k) % e
    )
    ids = valid_ids.clone()
    scores = torch.tensor([0.7, 0.3], device=q.device).repeat(tokens, 1)
    valid_raw, _ = _reference(dq, w1, s1, w2, s2, ids, scores, config.activation)
    assert torch.count_nonzero(valid_raw) > 0
    runner, inputs = _runner_inputs(config, q, sf, ids, scores, view, packed)
    tactics = runner.get_valid_tactics(inputs, None)
    assert 128 in tactics and 256 in tactics and ("ffma", 1) in tactics
    for tactic in tactics:
        ids.copy_(valid_ids)
        inputs[3].copy_(scores)
        runner.forward(inputs, tactic=tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs, tactic=tactic)
        result = output if finalize else output[0]
        # Each placement becomes invalid after capture, then valid again. Both
        # bounds and all-invalid rows are exercised with every padding score.
        for slot in (0, 1, None):
            for bad_score in (0.0, float("nan"), float("inf"), -float("inf")):
                ids.copy_(valid_ids)
                inputs[3].copy_(scores)
                if slot is None:
                    ids[:, 0] = -1
                    ids[:, 1] = e
                    inputs[3].fill_(bad_score)
                else:
                    ids[:, slot] = -1 if slot == 0 else e
                    inputs[3][:, slot] = bad_score
                expected_scores = inputs[3].clone()
                raw, expected = _expected(
                    valid_raw, ids, expected_scores, e, packed, True
                )
                for replay in (False, True):
                    # Poison route scratch and output so stale values cannot
                    # accidentally satisfy the all-invalid case.
                    inputs[9].fill_(float("nan"))
                    inputs[12].fill_(float("nan"))
                    result.fill_(float("nan"))
                    if replay:
                        graph.replay()
                        current_output = output
                    else:
                        current_output = runner.forward(inputs, tactic=tactic)
                    # Capture records the mapping's arange; it is materialized
                    # only by replay. Check the output from the executed call.
                    _assert_near(
                        current_output if finalize else current_output[0],
                        expected if finalize else raw,
                    )
                    if not finalize:
                        _assert_raw_contract(
                            current_output, ids, expected_scores, e, packed
                        )
            ids.copy_(valid_ids)
            inputs[3].copy_(scores)
            expected_scores = inputs[3].clone()
            graph.replay()
            _, expected = _expected(valid_raw, ids, expected_scores, e, packed, True)
            _assert_near(result, expected if finalize else valid_raw)
            if not finalize:
                _assert_raw_contract(output, ids, expected_scores, e, packed)


@pytest.mark.parametrize("tokens", [1, 4])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("finalize", [False, True])
def test_nvfp4_invalid_rows_after_graph_capture(tokens, packed, finalize):
    _require_gpu(allow_sm120=True)
    e, h, i, k = 2, 512, 128, 2
    device = torch.device("cuda", torch.cuda.current_device())
    # E2M1 code 2 is one, E4M3 scale 0x38 is one. This prepared constant
    # fixture does not depend on a native quantizer or its scale layout helper.
    view = {
        "fc1_expert_weights": torch.full(
            (e, i, h // 2), 0x22, dtype=torch.uint8, device=device
        ),
        "fc2_expert_weights": torch.full(
            (e, h, i // 2), 0x22, dtype=torch.uint8, device=device
        ),
        "fc1_weight_block_scale": torch.full(
            (e, i, h // 16), 0x38, dtype=torch.uint8, device=device
        ),
        "fc2_weight_block_scale": torch.full(
            (e, h, i // 16), 0x38, dtype=torch.uint8, device=device
        ),
        "fc1_dequant_scale": torch.full((e,), 1 / h, device=device),
        "fc2_dequant_scale": torch.full((e,), 1 / i, device=device),
        "fc1_act_global_scale": torch.ones((), device=device),
        "fc2_act_global_scale": torch.ones((), device=device),
    }
    q = torch.full((tokens, h // 2), 0x22, dtype=torch.uint8, device=device)
    sf = torch.full((tokens, h // 16), 0x38, dtype=torch.uint8, device=device)
    valid_ids = (
        torch.arange(tokens * k, device=device, dtype=torch.int32).view(tokens, k) % e
    )
    ids = valid_ids.clone()
    scores = torch.tensor([0.7, 0.3], device=device).repeat(tokens, 1)
    config = MoEConfig(
        RoutingConfig(e, k),
        QuantConfig(QuantFormat.NVFP4, QuantFormat.NVFP4, nvfp4_4over6=None),
        ExpertConfig(i),
        activation=Identity(),
        backend=BackendOptions((FrostLowLatencyConfig(max_num_tokens=4),)),
        finalize=MoEFinalizeConfig(do_finalize=finalize),
    )
    valid_raw, _ = nvfp4_reference(q, sf, view, ids, scores, config.activation, packed)
    assert torch.count_nonzero(valid_raw) > 0
    runner, inputs = _runner_inputs(config, q, sf, ids, scores, view, packed)
    tactics = runner.get_valid_tactics(inputs, None)
    assert ("nvfp4_ffma", 2) in tactics and ("nvfp4_ffma", 4) in tactics
    for tactic in tactics:
        ids.copy_(valid_ids)
        inputs[3].copy_(scores)
        runner.forward(inputs, tactic=tactic)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = runner.forward(inputs, tactic=tactic)
        result = output if finalize else output[0]
        for slot in (0, 1, None):
            ids.copy_(valid_ids)
            inputs[3].copy_(scores)
            if slot is None:
                ids[:, 0] = -1
                ids[:, 1] = e
                inputs[3].fill_(float("nan"))
            else:
                ids[:, slot] = -1 if slot == 0 else e
                inputs[3][:, slot] = float("inf")
            expected_scores = inputs[3].clone()
            raw, expected = _expected(valid_raw, ids, expected_scores, e, packed, False)
            for sentinel in (123.0, float("nan")):
                for replay in (False, True):
                    result.fill_(sentinel)
                    if replay:
                        graph.replay()
                        current_output = output
                    else:
                        current_output = runner.forward(inputs, tactic=tactic)
                    # Capture records the mapping's arange; it is materialized
                    # only by replay. Check the output from the executed call.
                    _assert_near(
                        current_output if finalize else current_output[0],
                        expected if finalize else raw,
                    )
                    if not finalize:
                        _assert_raw_contract(
                            current_output, ids, expected_scores, e, packed
                        )
            ids.copy_(valid_ids)
            inputs[3].copy_(scores)
            expected_scores = inputs[3].clone()
            graph.replay()
            _, expected = _expected(valid_raw, ids, expected_scores, e, packed, False)
            _assert_near(result, expected if finalize else valid_raw)
            if not finalize:
                _assert_raw_contract(output, ids, expected_scores, e, packed)
