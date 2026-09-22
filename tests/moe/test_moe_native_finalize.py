# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""NVIDIA native finalizer reuse preserves FI's live routing-scale contract."""

import pytest
import torch

from flashinfer.fused_moe.cute_dsl.moe_utils import moe_unpermute
from tests.moe.test_moe_cudnn_bf16_joint import _kernel_names


def _require_blackwell():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] not in (
        10,
        12,
    ):
        pytest.skip("SM100 or SM120 family required")


def _reference(x, indices, scales, rounded):
    x = x.cpu().double()
    indices = indices.cpu().long()
    scales = (scales.to(torch.bfloat16) if rounded else scales).cpu().double()
    result = torch.zeros(indices.shape[0], x.shape[1], dtype=torch.float64)
    for k in range(indices.shape[1]):
        active = indices[:, k] >= 0
        result[active] += x[indices[active, k]] * scales[active, k, None]
    return result.to(torch.bfloat16).cuda()


@pytest.mark.parametrize(
    "tokens,hidden,top_k,enable_pdl",
    [
        (1, 7, 3, False),
        (3, 65, 3, False),
        (7, 257, 8, False),
        (1, 2048, 8, False),
        (64, 2048, 8, False),
        (147, 2048, 8, False),
        (148, 2048, 8, False),
        (149, 2056, 3, False),
        (148, 2048, 64, False),
        (8193, 8, 3, False),
        (1, 2048, 8, True),
        (8, 2048, 8, True),
        (64, 2048, 8, True),
    ],
)
@pytest.mark.parametrize("rounded", [False, True])
def test_native_finalize_live_capture(tokens, hidden, top_k, enable_pdl, rounded):
    _require_blackwell()
    rows = tokens * top_k + 7
    # Exact binary fractions keep the independent FP64 reference bit-exact
    # after BF16 output rounding, including cancellation and masked experts.
    x = (
        ((torch.arange(rows * hidden, device="cuda") % 31 - 15).float() / 32)
        .to(torch.bfloat16)
        .reshape(rows, hidden)
    )
    indices = (
        torch.arange(tokens * top_k, device="cuda", dtype=torch.int32) * 3 % rows
    ).reshape(tokens, top_k)
    indices.view(-1)[::5] = -1
    if tokens > 1:
        indices[0].fill_(-1)
    scales = (
        (torch.arange(tokens * top_k, device="cuda") % 61 + 513).float() / 1024
    ).reshape(tokens, top_k)
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)

    def run():
        moe_unpermute(
            x,
            output,
            indices,
            scales,
            tokens,
            top_k,
            round_scales_to_bf16=rounded,
            use_native_finalize=True,
            enable_pdl=enable_pdl,
        )

    if enable_pdl and torch.cuda.get_device_capability()[0] != 10:
        with pytest.raises(ValueError, match="PDL prototype requires SM100"):
            run()
        return

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        run()  # Build before capture.
        expected = _reference(x, indices, scales, rounded)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        if rounded:
            # These inputs must catch a missing FP32 -> BF16 scale boundary.
            assert not torch.equal(expected, _reference(x, indices, scales, False))
        graph = torch.cuda.CUDAGraph(keep_graph=True)
        with torch.cuda.graph(graph, stream=side):
            run()
        names = _kernel_names(graph)
        assert len(names) == 1 and "finalizeKernel" in names[0]
        unroll = 4 if top_k % 4 == 0 else 2 if top_k % 2 == 0 else 1
        assert f"Li{unroll}ELb{int(enable_pdl)}EEELb{int(rounded)}E" in names[0]
        assert ("finalizeKernelTop8" in names[0]) == (
            enable_pdl and rounded and tokens == 1
        )
        previous = output.clone()
        for _ in range(3):
            x.neg_()
            scales.add_(0.03125)
            indices.copy_(indices.roll(1, 1))
            expected = _reference(x, indices, scales, rounded)
            assert not torch.equal(previous, expected)
            output.fill_(float("nan"))
            graph.replay()
            torch.testing.assert_close(output, expected, rtol=0, atol=0)
            previous.copy_(output)
        indices.fill_(-1)
        output.fill_(float("nan"))
        graph.replay()
        assert torch.count_nonzero(output) == 0
    torch.cuda.current_stream().wait_stream(side)
    graph.reset()


@pytest.mark.parametrize(
    "bad",
    [
        "pdl_tokens",
        "pdl_hidden",
        "pdl_topk",
        "expanded",
        "wide",
        "weights",
        "indices",
        "alignment",
        "vector_tail",
    ],
)
def test_native_finalize_rejects_invalid_metadata(bad):
    _require_blackwell()
    tokens, hidden, top_k = {
        "vector_tail": (148, 2057, 8),
        "pdl_tokens": (65, 2048, 8),
        "pdl_hidden": (1, 1024, 8),
        "pdl_topk": (1, 2048, 4),
    }.get(bad, (1, 2048, 8))
    x = torch.ones(tokens * top_k, hidden, device="cuda", dtype=torch.bfloat16)
    output = torch.empty(tokens, hidden, device="cuda", dtype=torch.bfloat16)
    indices = torch.zeros(tokens, top_k, device="cuda", dtype=torch.int32)
    scales = torch.ones(tokens, top_k, device="cuda")
    kwargs = dict(use_native_finalize=True)
    if bad.startswith("pdl_"):
        kwargs["enable_pdl"] = True
    elif bad == "expanded":
        kwargs["input_is_expanded"] = True
    elif bad == "wide":
        kwargs["use_wide_tiling"] = True
    elif bad == "weights":
        scales = scales.to(torch.bfloat16)
    elif bad == "indices":
        indices = indices.to(torch.int64)
    elif bad == "alignment":
        x = torch.empty(x.numel() + 1, device="cuda", dtype=x.dtype)[1:].view_as(x)
    with pytest.raises(ValueError):
        moe_unpermute(x, output, indices, scales, tokens, top_k, **kwargs)
