# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch

from flashinfer.deepseek_v41 import (
    deepseek_v41_index_scores_fp32,
    deepseek_v41_quantize,
    deepseek_v41_quantize_index_cache,
)
from .test_deepseek_v41 import gate
from .test_deepseek_v41_small_indexer import dequant


@pytest.mark.parametrize(
    "batch,context,page,candidate_count",
    [
        (1, 65, 32, None),
        (2, 2049, 64, None),
        (4, 32769, 128, None),
        (2, 32769, 64, 2048),
        (3, 127, 32, 17),
    ],
)
def test_tiled_indexer_fp64_paging_candidates_padding_and_changed_graph(
    batch, context, page, candidate_count
):
    gate()
    torch.manual_seed(42501 + batch + context + page)
    pages = (context + page - 1) // page
    q = torch.randn(batch * 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages * page, 128, device="cuda", dtype=torch.bfloat16)
    qd, qs = deepseek_v41_quantize(q, format="mxfp4")
    kd, ks = deepseek_v41_quantize(k, format="mxfp4")
    cache = deepseek_v41_quantize_index_cache(k, page_size=page)
    qd, qs = qd.view(batch, 32, 64), qs.view(batch, 32, 4)
    weights = (
        ((torch.arange(32, device="cuda") - 16) / 32).bfloat16()[None].repeat(batch, 1)
    )
    visible = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    if batch > 1:
        visible[0] = 0
    table = torch.randperm(batch * pages, device="cuda").int().view(batch, pages)
    candidates = None
    if candidate_count is not None:
        candidates = torch.randint(
            (context + 7) // 8 + 8,
            (batch, candidate_count),
            device="cuda",
            dtype=torch.int32,
        )
        candidates[:, -2:] = -1
    width = context if candidates is None else candidate_count * 8
    stride = (width + 511) // 512 * 512
    storage = torch.full((batch, stride), 17, device="cuda", dtype=torch.bfloat16)
    out = storage[:, :width]

    def run():
        return deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights,
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=out,
        )

    def verify(exact):
        logical = torch.arange(width, device="cuda")[None].expand(batch, -1)
        if candidates is not None:
            logical = (
                candidates.long()[..., None] * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
        valid = (logical >= 0) & (logical < visible[:, None]) & (logical < context)
        physical = (
            table.gather(1, logical.clamp(0, context - 1).long() // page) * page
            + logical.clamp(0, context - 1) % page
        )
        query = dequant(qd, qs)
        keys = dequant(kd, ks)[physical.long()]
        expected = (
            torch.einsum("bhd,btd->bht", query, keys).relu()
            * weights.double()[..., None]
        ).sum(1)
        assert torch.isneginf(out[~valid]).all()
        actual = out.double().masked_fill(~valid, 0)
        expected = expected.masked_fill(~valid, 0)
        relative = (actual - expected).norm() / expected.norm().clamp_min(1e-20)
        maximum = (actual - expected).abs().max() / expected.abs().max().clamp_min(
            1e-20
        )
        assert relative < 0.0021 and maximum < 0.0042, (float(relative), float(maximum))
        if exact:
            # Dyadic weights bound the full sum's significand: all reduction
            # schedules must agree before the required BF16 result boundary.
            torch.testing.assert_close(
                out[valid], expected.bfloat16()[valid], atol=0, rtol=0
            )
        torch.testing.assert_close(
            storage[:, width:], torch.full_like(storage[:, width:], 17), atol=0, rtol=0
        )

    run()
    verify(True)
    allocated = deepseek_v41_index_scores_fp32(
        qd,
        qs,
        cache,
        weights,
        visible,
        table,
        max_context_len=context,
        candidates=candidates,
    )
    torch.testing.assert_close(allocated, out, atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    for step in range(2):
        qd.bitwise_xor_(0x88)
        qs.add_(1)
        weights.copy_(torch.randn_like(weights))
        k.copy_(torch.randn_like(k) * 0.75)
        deepseek_v41_quantize(k, format="mxfp4", data=kd, scales=ks)
        deepseek_v41_quantize_index_cache(k, page_size=page, out=cache)
        table.copy_(table.flip(1))
        visible.copy_(
            torch.randint(context + 1, (batch,), device="cuda", dtype=torch.int32)
        )
        visible[-1] = context - step
        if candidates is not None:
            candidates.copy_(
                torch.randint(
                    (context + 7) // 8 + 8,
                    candidates.shape,
                    device="cuda",
                    dtype=torch.int32,
                )
            )
            candidates[:, 0] = -1
        graph.replay()
        verify(False)
    torch.cuda.set_sync_debug_mode("error")
    try:
        run()
    finally:
        torch.cuda.set_sync_debug_mode("default")
    alias = (
        cache.as_strided((batch * stride * 2,), (1,))
        .view(torch.bfloat16)
        .view(batch, stride)[:, :width]
    )
    with pytest.raises(ValueError, match="overlap"):
        deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights,
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=alias,
        )
    with pytest.raises(ValueError, match="inference"):
        deepseek_v41_index_scores_fp32(
            qd,
            qs,
            cache,
            weights.requires_grad_(),
            visible,
            table,
            max_context_len=context,
            candidates=candidates,
            out=out,
        )


@pytest.mark.parametrize("query_scale,key_scale", [(0, 254), (254, 0)])
def test_native_mxfp4_extreme_e8m0_scales(query_scale, key_scale):
    gate()
    # Both dequantized operands are finite BF16, including 2**-128.
    # Every channel product is exactly 1/4; 128 channels and 32 equally
    # weighted heads give a score of 32, in either scale orientation.
    qd = torch.full((1, 32, 64), 0x11, device="cuda", dtype=torch.uint8)
    qs = torch.full((1, 32, 4), query_scale, device="cuda", dtype=torch.uint8)
    storage = torch.zeros(2, 2560, device="cuda", dtype=torch.uint8)
    cache = storage.as_strided((2, 32, 1, 68), (2560, 68, 68, 1))
    storage[:, : 32 * 64].fill_(0x11)
    storage[:, 32 * 64 : 32 * 68].fill_(key_scale)
    weights = torch.full((1, 32), 1 / 32, device="cuda", dtype=torch.bfloat16)
    visible = torch.tensor([63], device="cuda", dtype=torch.int32)
    table = torch.tensor([[1, 0]], device="cuda", dtype=torch.int32)
    out = deepseek_v41_index_scores_fp32(
        qd, qs, cache, weights, visible, table, max_context_len=64
    )
    torch.testing.assert_close(
        out[:, :63], torch.full_like(out[:, :63], 32), atol=0, rtol=0
    )
    assert torch.isneginf(out[:, 63]).all()
