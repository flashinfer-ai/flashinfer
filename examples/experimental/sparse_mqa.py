"""Prepared MXFP4 contiguous sparse MQA; repeated calls own no allocations."""
import torch
from flashinfer.sparse_mqa import prepare_sparse_mqa_metadata, prepare_sparse_mqa_logits

# Packed E2M1 value +1 and packed UE8M0 scale1,32 heads,128 channels.
q = torch.full((16, 32, 64), 0x22, dtype=torch.uint8, device="cuda")
sf_q = torch.full((16, 32), 0x7f7f7f7f, dtype=torch.int32, device="cuda")
kv = torch.full((512, 64), 0x22, dtype=torch.uint8, device="cuda")
sf_kv = torch.full((512,), 0x7f7f7f7f, dtype=torch.int32, device="cuda")
weights = torch.full((16, 32), 1/32, dtype=torch.bfloat16, device="cuda")
sparse = torch.arange(2048, device="cuda", dtype=torch.int32).clamp_max(63).repeat(16, 1)
starts = torch.zeros(16, device="cuda", dtype=torch.int32)
ends = torch.full_like(starts, 512)
metadata = prepare_sparse_mqa_metadata(sparse, fmt="mxfp4", starts=starts, ends=ends, num_kv_tokens=512)
plan = prepare_sparse_mqa_logits(q, sf_q, kv, sf_kv, weights, metadata)
logits = plan.run()
print(logits[:, :512].shape)  # Remaining compressed slots are not written.
