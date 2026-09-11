# Copyright (c) 2026 by FlashInfer team.
# Licensed under the Apache License, Version 2.0.

import torch
from flashinfer import deepseek_v41 as ds

q = torch.randn(1, 1, 64, 512, device="cuda", dtype=torch.bfloat16)
x = torch.randn(128, 512, device="cuda", dtype=torch.bfloat16)
swa = ds.deepseek_v41_quantize_cache(x, format="swa_mxfp8")
main = ds.deepseek_v41_quantize_cache(x, format="main_kv_fp4")
ids = torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
sink = torch.zeros(64, device="cuda")
out, lse, plan = ds.deepseek_v41_decode(q, swa, main, ids, ids, sink, backend="frost")
q.normal_()
out, lse, _ = ds.deepseek_v41_decode(
    q, swa, main, ids, ids, sink, backend="frost", plan=plan
)
print("Frost decode:", out.shape, lse.shape, plan.backend, plan.arithmetic)
