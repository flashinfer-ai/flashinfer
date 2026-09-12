# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import torch
from flashinfer import deepseek_v41 as ds

kv = torch.randn(1, 512, device="cuda")
score = torch.randn_like(kv)
kv_state = torch.zeros(1, 2, 512, device="cuda")
score_state = torch.zeros_like(kv_state)
weight = torch.ones(512, device="cuda", dtype=torch.bfloat16)
starts = torch.tensor([1], device="cuda", dtype=torch.int32)
out, position = ds.deepseek_v41_compressor_decode(
    kv, score, kv_state, score_state, weight, starts
)
print("CSA2 completed pair:", out.shape, position)
