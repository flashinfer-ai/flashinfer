# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Packed-scale rounding matches the existing two-launch path bit for bit."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import moe_unpermute


@pytest.mark.parametrize("tokens,hidden,top_k", [(17, 256, 2), (4096, 2048, 4)])
@pytest.mark.parametrize("expanded", [False, True])
@pytest.mark.parametrize("pdl", [False, True])
def test_inline_scale_rounding_live_capture(tokens, hidden, top_k, expanded, pdl):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("Hopper or newer GPU required")
    torch.manual_seed(937)
    x = torch.randn(tokens * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.randperm(tokens * top_k, device="cuda", dtype=torch.int32)
    indices = indices.view(tokens, top_k)
    indices.view(-1)[::11] = -1
    indices[0].fill_(-1)
    scales = torch.randn(tokens, top_k, device="cuda")
    # Include both sides and exact ties at a BF16 rounding midpoint.
    scales.view(-1)[1:4] = torch.tensor(
        [0.5 + 2**-9 - 2**-24, 0.5 + 2**-9, 0.5 + 2**-9 + 2**-24],
        device="cuda",
    )
    actual = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    expected = torch.empty_like(actual)
    unrounded = torch.empty_like(actual)

    def run_inline():
        moe_unpermute(x, actual, indices, scales, tokens, top_k, pdl, expanded, True)

    def check():
        moe_unpermute(
            x,
            expected,
            indices,
            scales.to(torch.bfloat16),
            tokens,
            top_k,
            pdl,
            expanded,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        # Omitting the BF16 boundary must be detectable with these inputs.
        moe_unpermute(x, unrounded, indices, scales, tokens, top_k, pdl, expanded)
        assert not torch.equal(actual, unrounded)
        assert torch.count_nonzero(actual[0]) == 0

    run_inline()
    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run_inline()
    previous = actual.clone()
    for _ in range(3):
        x.neg_()
        scales.add_(0.03125)
        indices[1:].copy_(indices[1:].roll(1, 0))
        actual.fill_(float("nan"))
        graph.replay()
        check()
        assert not torch.equal(actual, previous)
        previous.copy_(actual)


def test_inline_scale_rounding_rejects_wrong_dtypes():
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    x = torch.ones(2, 256, device="cuda", dtype=torch.float16)
    out = torch.empty(1, 256, device="cuda", dtype=torch.float16)
    indices = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
    scales = torch.ones(1, 2, device="cuda")
    with pytest.raises(ValueError, match="requires BF16"):
        moe_unpermute(x, out, indices, scales, 1, 2, round_scales_to_bf16=True)
