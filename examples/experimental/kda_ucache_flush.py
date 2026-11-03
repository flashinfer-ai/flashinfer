"""Minimal run of the experimental KDA ReplaySSM verify+flush kernel.

Five speculative steps of T=4 drafts for B=8 requests with every draft
accepted; the ring folds into the state when it reaches flush_min.

    python examples/experimental/kda_ucache_flush.py
"""

import torch

from flashinfer.kda_decode import kda_delta_rule_mtp_ucache_flush

B, T, H, D, RING, POOL = 8, 4, 12, 128, 32, 16
dev, bf16 = "cuda", torch.bfloat16

pool = torch.randn(POOL, H, D, D, device=dev).mul_(0.1).to(bf16)
k_cache = torch.zeros(POOL, H, RING, D, dtype=bf16, device=dev)
u_cache = torch.zeros(POOL, H, RING, D, dtype=bf16, device=dev)
g_cache = torch.zeros(POOL, H, RING, D, dtype=torch.float32, device=dev)
slots = torch.randperm(POOL, device=dev)[:B].to(torch.int32)
hist_len = torch.zeros(B, dtype=torch.int32, device=dev)
cache_base = torch.zeros(B, dtype=torch.int32, device=dev)
A_log = torch.zeros(H, device=dev)
dt_bias = torch.zeros(H, D, device=dev)

for step in range(5):
    q, k, g = (torch.randn(B, T, H, D, device=dev, dtype=bf16) for _ in range(3))
    v = torch.randn(B, T, H, D, device=dev, dtype=bf16)
    beta_logit = torch.randn(B, T, H, device=dev, dtype=bf16)
    y = kda_delta_rule_mtp_ucache_flush(
        A_log,
        g,
        dt_bias,
        lower_bound=-5.0,
        q=q,
        k=k,
        v=v,
        b=beta_logit,
        initial_state_source=pool,
        initial_state_indices=slots,
        k_cache=k_cache,
        u_cache=u_cache,
        g_cache=g_cache,
        hist_len=hist_len,
        cache_base=cache_base,
        restart_hist_on_flush=True,  # commits cursors of folded rows in place
    )
    hist_len += T  # all T drafts accepted
    print(f"step {step}: y {tuple(y.shape)}, hist_len {hist_len.tolist()[:2]}")
