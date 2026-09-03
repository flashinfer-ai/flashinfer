"""Reference oracle for the unified paged-prefill prototype tests/fuzzer.

Pure-torch fp32 implementation of paged batch prefill with bottom-right
causal masking, returning both the output and the contract-form LSE
(base-2, packed ``(total_q_tokens, num_qo_heads)``).  Deliberately written
request-by-request from first principles — slow and obvious beats fast and
kernel-shaped for an oracle.
"""

import math
from typing import Optional, Tuple

import torch


def reference_paged_prefill(
    q: torch.Tensor,  # (total_q, Hq, Dqk) any float dtype
    k_cache: torch.Tensor,  # (pages, Hk, page_size, Dqk) HND / (pages, ps, Hk, D) NHD
    v_cache: torch.Tensor,
    qo_indptr_cpu: torch.Tensor,  # (b+1,) host
    kv_seq_lens_cpu: torch.Tensor,  # (b,) host
    block_tables: torch.Tensor,  # (b, max_pages) device, or None with kv_page_indices
    page_size: int,
    causal: bool,
    sm_scale: Optional[float] = None,
    window_left: int = -1,
    kv_layout: str = "HND",
    kv_page_indices: Optional[torch.Tensor] = None,  # flat CSR page ids
) -> Tuple[torch.Tensor, torch.Tensor]:
    if kv_layout == "NHD":
        k_cache = k_cache.permute(0, 2, 1, 3)
        v_cache = v_cache.permute(0, 2, 1, 3)
    total_q, num_qo_heads, head_dim_qk = q.shape
    num_kv_heads = k_cache.shape[1]
    head_dim_vo = v_cache.shape[3]
    group = num_qo_heads // num_kv_heads
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim_qk)
    b = kv_seq_lens_cpu.shape[0]

    out = torch.empty(
        total_q, num_qo_heads, head_dim_vo, device=q.device, dtype=torch.float32
    )
    lse = torch.empty(total_q, num_qo_heads, device=q.device, dtype=torch.float32)

    for i in range(b):
        s, e = int(qo_indptr_cpu[i]), int(qo_indptr_cpu[i + 1])
        lq, lkv = e - s, int(kv_seq_lens_cpu[i])
        if lq == 0:
            continue
        q_i = q[s:e].float()  # (lq, Hq, Dqk)
        n_pages = (lkv + page_size - 1) // page_size
        if block_tables is not None:
            page_ids = block_tables[i, :n_pages].to(torch.int64)
        else:
            start = int(torch.sum((kv_seq_lens_cpu[:i] + page_size - 1) // page_size))
            page_ids = kv_page_indices[start : start + n_pages].to(torch.int64)
        # (n_pages, Hk, ps, D) -> (Hk, n_pages*ps, D) -> trim to lkv
        k_i = (
            k_cache[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim_qk)[:, :lkv]
            .float()
        )
        v_i = (
            v_cache[page_ids]
            .permute(1, 0, 2, 3)
            .reshape(num_kv_heads, -1, head_dim_vo)[:, :lkv]
            .float()
        )
        # GQA head expansion
        k_i = k_i.repeat_interleave(group, dim=0)  # (Hq, lkv, Dqk)
        v_i = v_i.repeat_interleave(group, dim=0)
        # scores: (Hq, lq, lkv)
        scores = torch.einsum("qhd,hkd->hqk", q_i, k_i) * sm_scale
        qpos = torch.arange(lq, device=q.device).unsqueeze(1)
        kpos = torch.arange(lkv, device=q.device).unsqueeze(0)
        allowed = torch.ones(lq, lkv, dtype=torch.bool, device=q.device)
        if causal:
            # bottom-right causal: query at position p (0-based within its
            # sequence) may attend kv index j iff j <= (lkv - lq) + p
            allowed &= kpos <= (lkv - lq) + qpos  # (lq, lkv)
        if window_left >= 0:
            # sliding window: kv index j visible iff p - j <= window_left
            # where p = (lkv - lq) + qpos is the query's absolute kv position
            allowed &= kpos >= (lkv - lq) + qpos - window_left
        if causal or window_left >= 0:
            scores = scores.masked_fill(~allowed.unsqueeze(0), float("-inf"))
        lse_i = torch.logsumexp(scores, dim=-1) / math.log(2)  # (Hq, lq), base-2
        p = torch.softmax(scores, dim=-1)
        o_i = torch.einsum("hqk,hkd->qhd", p, v_i)  # (lq, Hq, Dvo)
        out[s:e] = o_i
        lse[s:e] = lse_i.transpose(0, 1)
    return out, lse
