# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""CPU check that the add_rmsnorm_fp4quant trace reference honours weight_bias."""

import pytest
import torch

from flashinfer.trace.templates.norm import add_rmsnorm_fp4quant_trace


def _run(
    weight: torch.Tensor, weight_bias: float, seed: int
) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    x = torch.randn(4, 64, generator=gen).to(torch.bfloat16)
    r = torch.randn(4, 64, generator=gen).to(torch.bfloat16)
    reference = add_rmsnorm_fp4quant_trace.reference
    assert reference is not None
    out = reference(x, r, weight, weight_bias=weight_bias)
    return tuple(out), r


@pytest.mark.cpu_only
@pytest.mark.parametrize("weight_bias", [0.0, 1.0])
def test_weight_bias_matches_shifted_weight(weight_bias: float) -> None:
    weight = (0.1 * torch.randn(64, generator=torch.Generator().manual_seed(1))).to(
        torch.bfloat16
    )
    got, got_r = _run(weight, weight_bias, seed=0)
    shifted = weight.to(torch.float32) + weight_bias
    want, want_r = _run(shifted, 0.0, seed=0)
    assert torch.equal(got_r, want_r)
    for g, w in zip(got, want, strict=True):
        assert torch.equal(g, w), f"weight_bias={weight_bias}"
