# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

"""Candidate scoring with already packed MXFP4 inputs (SM100/SM103).

The constant values make the layout easy to inspect:0x11 packs two FP4
values of0.5 and scale127 encodes1. Real applications supply quantized Q
and their existing index cache; this example does not quantize or select TopK.
"""

import torch
from flashinfer.deepseek_v41 import (
    deepseek_v41_candidate_scores_fp32,
    deepseek_v41_index_scores_fp32,
    prepare_deepseek_v41_candidate_metadata,
)

batch, page, context, count = 64, 64, 8192, 1024
pages = context // page
qd = torch.full((batch, 32, 64), 0x11, device="cuda", dtype=torch.uint8)
qs = torch.full((batch, 32, 4), 127, device="cuda", dtype=torch.uint8)
weights = torch.full((batch, 32), 1 / 32, device="cuda", dtype=torch.bfloat16)
stride = (page * 68 + 511) // 512 * 512
storage = torch.zeros((batch * pages, stride), device="cuda", dtype=torch.uint8)
# A page holds all packed data first, then all four scale bytes per token.
storage[:, : page * 64].fill_(0x11)
storage[:, page * 64 : page * 68].fill_(127)
cache = storage.as_strided((batch * pages, page, 1, 68), (stride, 68, 68, 1))
table = torch.arange(batch * pages, device="cuda", dtype=torch.int32).view(batch, pages)
visible = torch.full((batch,), context - 1, device="cuda", dtype=torch.int32)
candidates = torch.arange(count, device="cuda", dtype=torch.int32)[None].repeat(
    batch, 1
)
# For small batches, use backend="triton". CuTe candidate scoring requires
# CUTLASS DSL>=4.7. Reuse out and workspace for allocation-free graph replay;
# workspace sizing and ownership are documented on the API.
scores = deepseek_v41_index_scores_fp32(
    qd,
    qs,
    cache,
    weights,
    visible,
    table,
    max_context_len=context,
    candidates=candidates,
    backend="cute_dsl",
)
torch.testing.assert_close(
    scores[:, :-1], torch.full_like(scores[:, :-1], 32), atol=0, rtol=0
)
assert torch.isneginf(scores[:, -1]).all()
print("MXFP4 candidate scores:", scores.shape)

# When layers share candidates, mapping and visibility, publish them once.
# Each consumer supplies its own Q, cache and weights. Publish again when
# metadata changes, and order publication before readers on other streams.
metadata = prepare_deepseek_v41_candidate_metadata(
    visible,
    table,
    candidates,
    page_size=page,
    num_physical_pages=cache.shape[0],
    max_context_len=context,
)
prepared_scores = deepseek_v41_candidate_scores_fp32(qd, qs, cache, weights, metadata)
torch.testing.assert_close(prepared_scores, scores, atol=0, rtol=0)
weights.mul_(2)
deepseek_v41_candidate_scores_fp32(
    qd, qs, cache, weights, metadata, out=prepared_scores
)
torch.testing.assert_close(prepared_scores, scores * 2, atol=0, rtol=0)
print("Reused candidate metadata:", prepared_scores.shape)
