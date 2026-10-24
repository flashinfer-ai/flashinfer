# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Host-focused tests for the independent MXFP8 x MXFP4 MegaMoE oracle."""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
    compute_reference,
    decode_e8m0,
    dequantize_mxfp4,
    interleave_gate_up_8,
    quantize_mxfp4,
    quantize_mxfp8,
)


def _problem() -> dict[str, torch.Tensor]:
    hidden = intermediate = 32
    experts = 2
    x = torch.ones((1, 1, hidden), dtype=torch.float32)
    input_q, input_scale = quantize_mxfp8(x)

    canonical_w13 = torch.zeros(
        (experts, 2 * intermediate, hidden), dtype=torch.float32
    )
    eye = torch.eye(hidden, dtype=torch.float32)
    canonical_w13[:, :intermediate] = eye
    canonical_w13[:, intermediate:] = eye
    w13_q, w13_scale = quantize_mxfp4(interleave_gate_up_8(canonical_w13))

    w2 = torch.stack((eye, 2.0 * eye))
    w2_q, w2_scale = quantize_mxfp4(w2)
    return {
        "input_q": input_q,
        "input_scale": input_scale,
        "topk_ids": torch.tensor([[[0, 1]]], dtype=torch.int64),
        "topk_weights": torch.tensor([[[0.25, 0.75]]], dtype=torch.float32),
        "w13_packed": w13_q,
        "w13_scale": w13_scale,
        "w2_packed": w2_q,
        "w2_scale": w2_scale,
    }


def test_mxfp4_known_codes_and_rtne_midpoints() -> None:
    values = torch.zeros((1, 32), dtype=torch.float32)
    values[0, :8] = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0, 6.0])
    packed, scales = quantize_mxfp4(values)
    # Scale is one, and exact midpoint ties select an even-mantissa code.
    assert scales.tolist() == [[127]]
    codes = torch.empty(32, dtype=torch.uint8)
    codes[0::2] = packed[0] & 0x0F
    codes[1::2] = packed[0] >> 4
    assert codes[:8].tolist() == [0, 2, 2, 4, 4, 6, 6, 7]
    dequant = dequantize_mxfp4(packed, scales)
    assert dequant[0, :8].tolist() == [0.0, 1.0, 1.0, 2.0, 2.0, 4.0, 4.0, 6.0]


def test_e8m0_and_mxfp8_scale_recipe() -> None:
    decoded = decode_e8m0(torch.tensor([126, 127, 128], dtype=torch.uint8))
    torch.testing.assert_close(decoded, torch.tensor([0.5, 1.0, 2.0]))
    values = torch.zeros((1, 32), dtype=torch.float32)
    values[0, 0] = 449.0
    _, scales = quantize_mxfp8(values)
    assert scales.item() == 128  # ceil_pow2(449 / 448) == 2


def test_routing_is_post_fc2_and_reduces_in_slot_order() -> None:
    result = compute_reference(**_problem())
    assert result.route_terms.shape == (1, 1, 2, 32)
    # Route terms are explicitly unweighted. Expert 1 has a 2x FC2 weight.
    torch.testing.assert_close(
        result.route_terms[0, 0, 1].float(),
        2.0 * result.route_terms[0, 0, 0].float(),
    )
    expected = (
        result.route_terms[:, :, 0].float() * 0.25
        + result.route_terms[:, :, 1].float() * 0.75
    ).bfloat16()
    torch.testing.assert_close(result.output, expected, rtol=0, atol=0)


def test_minus_one_route_is_zero_and_score_is_ignored() -> None:
    problem = _problem()
    problem["topk_ids"][0, 0, 1] = -1
    problem["topk_weights"][0, 0, 1] = 12345.0
    result = compute_reference(**problem)
    assert torch.count_nonzero(result.route_terms[0, 0, 1]) == 0
    expected = (result.route_terms[:, :, 0].float() * 0.25).bfloat16()
    torch.testing.assert_close(result.output, expected, rtol=0, atol=0)


@pytest.mark.parametrize("bad_id", [-2, 2, 99])
def test_other_invalid_expert_ids_are_rejected(bad_id: int) -> None:
    problem = _problem()
    problem["topk_ids"][0, 0, 0] = bad_id
    with pytest.raises(ValueError, match="expert id"):
        compute_reference(**problem)


def test_scale_shape_mismatch_is_rejected() -> None:
    problem = _problem()
    problem["input_scale"] = torch.zeros((1, 1, 2), dtype=torch.uint8)
    with pytest.raises(ValueError, match="input_scale"):
        compute_reference(**problem)


def test_zero_tokens_produce_empty_outputs() -> None:
    problem = _problem()
    input_q, input_scale = quantize_mxfp8(torch.empty((1, 0, 32), dtype=torch.float32))
    problem.update(
        input_q=input_q,
        input_scale=input_scale,
        topk_ids=torch.empty((1, 0, 2), dtype=torch.int64),
        topk_weights=torch.empty((1, 0, 2), dtype=torch.float32),
    )
    result = compute_reference(**problem)
    assert result.route_terms.shape == (1, 0, 2, 32)
    assert result.output.shape == (1, 0, 32)
