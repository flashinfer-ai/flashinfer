"""Run a prepared experimental FP8 dense MQA lightning-indexer plan."""

import torch
from flashinfer.dense_mqa import prepare_dense_mqa_logits

q = torch.randn(16, 32, 128, device="cuda").to(torch.float8_e4m3fn)
kv = torch.randn(4096, 128, device="cuda").to(torch.float8_e4m3fn)
scales = torch.ones(4096, device="cuda", dtype=torch.float32)
weights = torch.randn(16, 32, device="cuda", dtype=torch.float32)
starts = torch.zeros(16, device="cuda", dtype=torch.int32)
ends = torch.full_like(starts, 4096)
plan = prepare_dense_mqa_logits("fp8", q, kv, weights, starts, ends, kv_scales=scales)
logits = plan.run()  # FP32 [16, 4096]; -inf outside each row's [start, end) window.
print(logits.shape)
