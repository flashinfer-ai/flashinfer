# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import pytest
import torch
from flashinfer.deepseek_v41 import (
    deepseek_v41_quantize,
    deepseek_v41_quantize_index_cache,
    deepseek_v41_small_index_scores,
)


def dequant(data, scales):
    lut = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=data.device,
        dtype=torch.float64,
    )
    codes = torch.stack((data & 15, data >> 4), -1).flatten(-2).long()
    return lut[codes] * torch.exp2(scales.double() - 127).repeat_interleave(32, -1)


@pytest.mark.parametrize("batch,context", [(1, 4), (3, 127), (4, 128)])
@pytest.mark.parametrize("sparse", [False, True])
def test_short_paged_mxfp4_fp64_and_changed_graph(batch, context, sparse):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() not in (
        (10, 0),
        (10, 3),
    ):
        pytest.skip("requires SM100/SM103")
    torch.manual_seed(4141 + batch + context)
    page = 64
    pages = (context + 63) // 64
    q = torch.randn(batch * 32, 128, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(batch * pages * page, 128, device="cuda", dtype=torch.bfloat16)
    qd, qs = deepseek_v41_quantize(q, format="mxfp4")
    kd, ks = deepseek_v41_quantize(k, format="mxfp4")
    cache = deepseek_v41_quantize_index_cache(k)
    qd, qs = qd.view(batch, 32, 64), qs.view(batch, 32, 4)
    table = torch.randperm(batch * pages, device="cuda").int().reshape(batch, pages)
    visible = torch.full((batch,), context, device="cuda", dtype=torch.int32)
    if batch > 1:
        visible[0] = 0
    weights = torch.randn(batch, 32, device="cuda", dtype=torch.bfloat16)
    candidates = (
        torch.arange(16, device="cuda", dtype=torch.int32)[None, :].repeat(batch, 1)
        if sparse
        else None
    )
    if sparse:
        candidates[:, -3:] = -1
    out = deepseek_v41_small_index_scores(
        qd,
        qs,
        cache,
        weights,
        visible,
        table,
        max_context_len=context,
        candidates=candidates,
    )

    def check():
        logical = torch.arange(out.shape[1], device="cuda").expand(batch, -1)
        if sparse:
            logical = (
                candidates.long()[..., None] * 8 + torch.arange(8, device="cuda")
            ).flatten(1)
        valid = (logical >= 0) & (logical < visible[:, None])
        physical = (
            table.gather(1, (logical.clamp(0, context - 1) // page).long()) * page
            + logical % page
        )
        query = dequant(qd, qs)
        keys = dequant(kd, ks)[physical.clamp_min(0).long()]
        expected = (
            torch.einsum("bhd,btd->bht", query, keys).relu()
            * weights.double()[..., None]
        ).sum(1)
        assert torch.isneginf(out[~valid]).all()
        actual = out.double().masked_fill(~valid, 0)
        expected = expected.masked_fill(~valid, 0)
        relative = float((actual - expected).norm() / expected.norm().clamp_min(1e-20))
        maximum = float(
            (actual - expected).abs().max() / expected.abs().max().clamp_min(1e-20)
        )
        assert relative < 0.0021 and maximum < 0.0042, (relative, maximum)

    check()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        deepseek_v41_small_index_scores(
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
    qd.bitwise_xor_(0x88)
    weights.mul_(-0.5)
    table.copy_(table.flip(1))
    visible.fill_(context - 1)
    if sparse:
        candidates[:, 0] = -1
    graph.replay()
    check()
    torch.cuda.set_sync_debug_mode("error")
    try:
        deepseek_v41_small_index_scores(
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
    finally:
        torch.cuda.set_sync_debug_mode("default")
