# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import torch
from flashinfer import deepseek_v41 as ds

q = torch.randn(32, 128, device="cuda", dtype=torch.bfloat16)
qd, qs = ds.deepseek_v41_quantize(q, format="mxfp4")
k = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
cache = ds.deepseek_v41_quantize_index_cache(k)
weights = torch.ones(1, 32, device="cuda", dtype=torch.bfloat16)
visible = torch.tensor([128], device="cuda", dtype=torch.int32)
pages = torch.tensor([[0, 1]], device="cuda", dtype=torch.int32)
scores = ds.deepseek_v41_index_scores_fp32(
    qd.view(1, 32, 64),
    qs.view(1, 32, 4),
    cache,
    weights,
    visible,
    pages,
    max_context_len=128,
)
print("Frost MXFP4 score shape:", scores.shape)
