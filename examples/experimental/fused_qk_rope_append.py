"""Minimal fused QK RMSNorm, RoPE, and paged KV append example."""

import torch

from flashinfer.rope import fused_qk_rmsnorm_rope_append_paged_kv_cache


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda")
    batch_size, qo_len = 2, 1
    q_heads, kv_heads, head_dim = 8, 1, 128
    page_size, context_len = 16, 31
    final_len = context_len + qo_len
    pages_per_request = (final_len + page_size - 1) // page_size

    q_indptr = torch.arange(batch_size + 1, device=device, dtype=torch.int32) * qo_len
    seq_lens = torch.full((batch_size,), final_len, device=device, dtype=torch.int32)
    page_indices = torch.arange(
        batch_size * pages_per_request, device=device, dtype=torch.int32
    ).view(batch_size, pages_per_request)

    width = (q_heads + 2 * kv_heads) * head_dim
    qkv = torch.randn(batch_size * qo_len, width, device=device, dtype=torch.bfloat16)
    inv_freq = 1.0 / (
        1e4
        ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim)
    )
    freqs = torch.outer(
        torch.arange(final_len, device=device, dtype=torch.float32), inv_freq
    )
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
    q_weight = torch.ones(head_dim, device=device)
    k_weight = torch.ones(head_dim, device=device)
    cache = tuple(
        torch.empty(
            batch_size * pages_per_request,
            page_size,
            kv_heads,
            head_dim,
            device=device,
            dtype=torch.bfloat16,
        )
        for _ in range(2)
    )

    q = fused_qk_rmsnorm_rope_append_paged_kv_cache(
        qkv,
        cos_sin,
        seq_lens,
        q_indptr,
        page_indices,
        cache,
        False,
        q_weight,
        k_weight,
        2,
    )
    print(f"Q: {tuple(q.shape)}, K cache: {tuple(cache[0].shape)}")


if __name__ == "__main__":
    main()
