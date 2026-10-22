"""FP8 DSA lightning indexer top-k on SM120: ragged (prefill) and paged (decode)."""

import torch
from flashinfer.dsa_indexer import dsa_indexer_topk

dev, H, D, top_k = "cuda", 64, 128, 2048

# Ragged: two segments; each query sees the keys of its segment up to its causal position.
seg_q, seg_k = [256, 512], [6000, 9000]
q = torch.randn(sum(seg_q), H, D, device=dev).to(torch.float8_e4m3fn)
k = torch.randn(sum(seg_k), D, device=dev).to(torch.float8_e4m3fn)
k_scale = torch.rand(sum(seg_k), device=dev) + 0.5
w = torch.randn(sum(seg_q), H, device=dev) * D**-0.5
cu_q, cu_k = (
    torch.tensor(
        [0, *torch.tensor(s).cumsum(0).tolist()], dtype=torch.int32, device=dev
    )
    for s in (seg_q, seg_k)
)
idx, scores = dsa_indexer_topk(
    q, k, w, cu_q, cu_k, top_k=top_k, backend="sm120", k_scale=k_scale
)

# Reference for the last query of segment 0: it sees all of that segment's keys.
t = seg_q[0] - 1
ref = (q[t].float() @ k[: seg_k[0]].float().T).relu()
ref = (w[t, :, None] * ref).sum(0) * k_scale[: seg_k[0]]
print(
    "ragged k-th score:",
    scores[t].min().item(),
    "ref:",
    ref.topk(top_k).values[-1].item(),
)

# Paged: B requests, one token each, 64-key pages of 128 e4m3 bytes plus 4 scale bytes per key.
B, P, blocks = 4, 64, 256
pages = torch.zeros(B * blocks, P * (D + 4), dtype=torch.uint8, device=dev)
pages[:, : P * D] = torch.randint(
    0, 0x7F, (B * blocks, P * D), device=dev, dtype=torch.uint8
)
pages[:, P * D :] = (
    torch.ones(B * blocks, P, device=dev).view(torch.uint8).view(-1, P * 4)
)
cache = pages.view(B * blocks, P, D + 4)
block_table = torch.randperm(B * blocks, device=dev, dtype=torch.int32).view(B, blocks)
context_lens = torch.tensor(
    [[1000], [5000], [12000], [16384]], dtype=torch.int32, device=dev
)
q_dec = torch.randn(B, 1, H, D, device=dev).to(torch.float8_e4m3fn)
w_dec = torch.randn(B, H, device=dev) * D**-0.5
idx, scores = dsa_indexer_topk(
    q_dec,
    cache,
    w_dec,
    None,
    None,
    top_k=top_k,
    backend="sm120",
    block_table=block_table,
    context_lens=context_lens,
)
print("paged selected per row:", (idx >= 0).sum(1).tolist())  # min(top_k, context_len)
