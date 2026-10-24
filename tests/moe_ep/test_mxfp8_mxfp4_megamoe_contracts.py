# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Single-rank, host-runnable correctness contracts for MXFP8 x MXFP4 MegaMoE."""

from __future__ import annotations

import pytest
import torch

from flashinfer.moe_ep.cute_dsl.megamoe.mxfp8_mxfp4.reference import (
    compute_reference,
    interleave_gate_up_8,
    quantize_mxfp4,
    quantize_mxfp8,
)


def _oracle_problem(
    *, tokens: int, hidden: int = 32, intermediate: int = 32, experts: int = 4
) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260922)
    x = torch.randn((1, tokens, hidden), generator=generator)
    input_q, input_scale = quantize_mxfp8(x)
    w13 = (
        torch.randn((experts, 2 * intermediate, hidden), generator=generator)
        / hidden**0.5
    )
    w2 = torch.randn((experts, hidden, intermediate), generator=generator) / (
        intermediate**0.5
    )
    w13_packed, w13_scale = quantize_mxfp4(interleave_gate_up_8(w13))
    w2_packed, w2_scale = quantize_mxfp4(w2)
    slots = 4
    token_idx = torch.arange(tokens).view(1, tokens, 1)
    slot_idx = torch.arange(slots).view(1, 1, slots)
    topk_ids = ((token_idx + slot_idx) % experts).to(torch.int64)
    topk_weights = torch.randn(
        (1, tokens, slots), generator=generator, dtype=torch.float32
    )
    return {
        "input_q": input_q,
        "input_scale": input_scale,
        "topk_ids": topk_ids,
        "topk_weights": topk_weights,
        "w13_packed": w13_packed,
        "w13_scale": w13_scale,
        "w2_packed": w2_packed,
        "w2_scale": w2_scale,
    }


@pytest.mark.parametrize("tokens", [0, 1, 31, 32, 33])
def test_mxfp8_mxfp4_megamoe_oracle_boundary_token_counts(tokens: int) -> None:
    problem = _oracle_problem(tokens=tokens)
    result = compute_reference(**problem)
    assert result.route_terms.shape == (1, tokens, 4, 32)
    assert result.output.shape == (1, tokens, 32)
    assert result.route_terms.dtype == torch.bfloat16
    assert result.output.dtype == torch.bfloat16


def test_mxfp8_mxfp4_megamoe_zero_token_experts_and_extreme_skew() -> None:
    problem = _oracle_problem(tokens=33)
    # Every live edge targets expert 3.  Experts 0..2 are deliberately empty.
    problem["topk_ids"].fill_(3)
    first = compute_reference(**problem)
    second = compute_reference(**problem)
    assert torch.equal(first.route_terms, second.route_terms)
    assert torch.equal(first.output, second.output)
    assert torch.count_nonzero(first.route_terms) > 0


def test_mxfp8_mxfp4_megamoe_masked_routes_ignore_nonfinite_like_magnitudes() -> None:
    problem = _oracle_problem(tokens=3)
    problem["topk_ids"][:, :, 1::2] = -1
    problem["topk_weights"][:, :, 1::2] = torch.finfo(torch.float32).max
    result = compute_reference(**problem)
    assert torch.count_nonzero(result.route_terms[:, :, 1::2]) == 0

    expected = torch.zeros_like(result.output, dtype=torch.float32)
    for slot in (0, 2):
        expected += result.route_terms[:, :, slot].float() * problem["topk_weights"][
            :, :, slot
        ].unsqueeze(-1)
    assert torch.equal(result.output, expected.bfloat16())


@pytest.mark.parametrize("bad_id", [-2, 4, 2**31 - 1])
def test_mxfp8_mxfp4_megamoe_rejects_invalid_routes(bad_id: int) -> None:
    problem = _oracle_problem(tokens=1)
    problem["topk_ids"][0, 0, 0] = bad_id
    with pytest.raises(ValueError, match="expert id"):
        compute_reference(**problem)


def test_mxfp8_mxfp4_megamoe_reduction_is_fixed_slot_order() -> None:
    problem = _oracle_problem(tokens=5)
    result = compute_reference(**problem)
    manual = torch.zeros((1, 5, 32), dtype=torch.float32)
    for slot in range(problem["topk_ids"].shape[-1]):
        manual += result.route_terms[:, :, slot].float() * problem["topk_weights"][
            :, :, slot
        ].unsqueeze(-1)
    assert torch.equal(result.output, manual.bfloat16())
