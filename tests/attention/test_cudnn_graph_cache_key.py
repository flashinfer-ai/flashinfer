"""Regression tests: the cuDNN graph-cache keys must include attn scale.

The cuDNN SDPA graph bakes ``attn_scale`` in as a compile-time constant, but
``_sdpa_prefill_key_fn`` did not include it in the process-global graph-cache
key.  A second call with identical shapes and a *different* ``scale``
therefore silently replayed the first call's graph and computed attention
with the stale scale (observed on H100: output error vs the correct
reference 2.56, vs the stale-scale reference 0.0037 — an exact stale
replay).  Same-shape different-scale calls are routine in serving (per-layer
logit scaling, muP, model switching), so this is a silent-wrong-results bug,
not a perf detail.

This test runs the same shape twice with two scales and checks each result
against an independent torch reference; without the key fix the second
iteration fails.
"""

import math

import pytest
import torch

from flashinfer.cudnn import (
    cudnn_batch_decode_with_kv_cache,
    cudnn_batch_prefill_with_kv_cache,
)
from flashinfer.cudnn import prefill as cudnn_prefill
from flashinfer.utils import get_compute_capability


def _skip_if_unsupported(device):
    if not cudnn_prefill.CUDNN_AVAILABLE:
        pytest.skip("cudnn-frontend python package not available")
    major, _ = get_compute_capability(torch.device(device))
    if major < 8:
        pytest.skip("cuDNN SDPA requires SM80+")


def _reference(q, k, v, q_lens, kv_lens, scale, causal):
    outs = []
    qo_off = 0
    kv_off = 0
    for lq, lkv in zip(q_lens.tolist(), kv_lens.tolist(), strict=True):
        q_i = q[qo_off : qo_off + lq].float()  # (lq, H, D)
        k_i = k[kv_off : kv_off + lkv].float()
        v_i = v[kv_off : kv_off + lkv].float()
        scores = torch.einsum("qhd,khd->hqk", q_i, k_i) * scale
        if causal:
            qpos = torch.arange(lq, device=q.device).unsqueeze(1)
            kpos = torch.arange(lkv, device=q.device).unsqueeze(0)
            allowed = kpos <= (lkv - lq) + qpos
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        p = torch.softmax(scores, dim=-1)
        outs.append(torch.einsum("hqk,khd->qhd", p, v_i))
        qo_off += lq
        kv_off += lkv
    return torch.cat(outs)


def test_cudnn_prefill_scale_in_graph_cache_key():
    device = "cuda:0"
    _skip_if_unsupported(device)

    torch.manual_seed(0)
    batch_size, num_qo_heads, num_kv_heads, head_dim = 2, 4, 4, 128
    q_lens = torch.tensor([32, 32], dtype=torch.int32, device=device)
    kv_lens = torch.tensor([48, 48], dtype=torch.int32, device=device)
    zero = torch.zeros(1, dtype=torch.int32, device=device)
    qo_indptr = torch.cat([zero, torch.cumsum(q_lens, 0)]).int()
    kv_indptr = torch.cat([zero, torch.cumsum(kv_lens, 0)]).int()

    q = torch.randn(
        int(q_lens.sum()), num_qo_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k = torch.randn(
        int(kv_lens.sum()), num_kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v = torch.randn_like(k)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    # Same shapes twice, different scales: without `scale` in the cache key
    # the second iteration silently reuses the first graph (stale scale).
    for scale_mult in (1.0, 3.0):
        scale = scale_mult / math.sqrt(head_dim)
        out, _ = cudnn_batch_prefill_with_kv_cache(
            q,
            k,
            v,
            scale,
            workspace,
            max_token_per_sequence=32,
            max_sequence_kv=48,
            actual_seq_lens_q=q_lens.view(batch_size, 1, 1, 1),
            actual_seq_lens_kv=kv_lens.view(batch_size, 1, 1, 1),
            causal=True,
            return_lse=True,
            batch_offsets_q=qo_indptr,
            batch_offsets_k=kv_indptr,
            batch_offsets_units="tokens",
        )
        ref = _reference(q, k, v, q_lens, kv_lens, scale, causal=True)
        torch.testing.assert_close(
            out.float(),
            ref,
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, s=scale, sm=scale_mult: (
                f"scale={s} (mult {sm}): stale-scale graph replay?\n{m}"
            ),
        )


def test_cudnn_decode_scale_in_graph_cache_key():
    """Decode analog: the decode key omitted scale (and keyed only on
    (max_sequence_kv, q.shape, k_cache.shape)); same shapes with a different
    scale silently replayed the stale graph."""
    device = "cuda:0"
    _skip_if_unsupported(device)

    torch.manual_seed(0)
    batch_size, num_heads, head_dim, page_size = 2, 4, 128, 16
    kv_len = 48
    pages_per_seq = (kv_len + page_size - 1) // page_size
    total_pages = batch_size * pages_per_seq

    q = torch.randn(
        batch_size, num_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    k_cache = torch.randn(
        total_pages, num_heads, page_size, head_dim, dtype=torch.bfloat16, device=device
    )
    v_cache = torch.randn_like(k_cache)
    block_tables = torch.arange(total_pages, dtype=torch.int32, device=device).view(
        batch_size, pages_per_seq
    )
    seq_lens_kv = torch.full(
        (batch_size, 1, 1, 1), kv_len, dtype=torch.int32, device=device
    )
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)

    def decode_reference(scale):
        outs = []
        for i in range(batch_size):
            pages = block_tables[i].to(torch.int64)
            k_i = (
                k_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(num_heads, -1, head_dim)[:, :kv_len]
                .float()
            )
            v_i = (
                v_cache[pages]
                .permute(1, 0, 2, 3)
                .reshape(num_heads, -1, head_dim)[:, :kv_len]
                .float()
            )
            scores = torch.einsum("hd,hkd->hk", q[i].float(), k_i) * scale
            p = torch.softmax(scores, dim=-1)
            outs.append(torch.einsum("hk,hkd->hd", p, v_i))
        return torch.stack(outs)

    for scale_mult in (1.0, 3.0):
        scale = scale_mult / math.sqrt(head_dim)
        out = cudnn_batch_decode_with_kv_cache(
            q,
            k_cache,
            v_cache,
            scale,
            workspace,
            max_sequence_kv=kv_len,
            actual_seq_lens_kv=seq_lens_kv,
            block_tables=block_tables,
        )
        torch.testing.assert_close(
            out.float(),
            decode_reference(scale),
            atol=2e-2,
            rtol=2e-2,
            msg=lambda m, s=scale, sm=scale_mult: (
                f"decode scale={s} (mult {sm}): stale-scale graph replay?\n{m}"
            ),
        )
