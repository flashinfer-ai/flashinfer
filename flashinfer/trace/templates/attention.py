# Copyright (c) 2025 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TraceTemplates for attention operations."""

import math

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


# ── GQA paged decode ─────────────────────────────────────────────────────────


@torch.no_grad()
def _gqa_paged_decode_reference(q, k_cache, v_cache, kv_indptr, kv_indices, sm_scale):
    batch_size, num_qo_heads, head_dim = q.shape
    _, page_size, num_kv_heads, _ = k_cache.shape

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.full(
        (batch_size, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
    )

    gqa_ratio = num_qo_heads // num_kv_heads
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim).to(torch.float32)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim).to(torch.float32)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_start >= page_end:
            output[b].zero_()
            continue
        token_ids = kv_indices[page_start:page_end].to(torch.long)
        k_b = k_flat[token_ids]  # [T, num_kv_heads, head_dim]
        v_b = v_flat[token_ids]
        q_b = q[b].to(torch.float32)  # [num_qo_heads, head_dim]
        for h in range(num_qo_heads):
            kv_h = h // gqa_ratio
            logits = torch.matmul(q_b[h], k_b[:, kv_h].T) * sm_scale
            lse[b, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1)
            output[b, h] = torch.matmul(attn, v_b[:, kv_h]).to(torch.bfloat16)

    return output, lse


gqa_paged_decode_trace = TraceTemplate(
    op_type="gqa_paged",
    name_prefix="gqa_paged_decode",
    description="Batched Grouped Query Attention decode with a paged KV cache.",
    axes={
        "batch_size": Var(description="Total number of query tokens."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "len_indptr": Var(description="Length of kv_indptr array."),
        "num_kv_indices": Var(description="Total number of KV page indices."),
    },
    inputs={
        "q": Tensor(["batch_size", "num_qo_heads", "head_dim"]),
        # k_cache / v_cache come from paged_kv_cache=(k, v)
        "k_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=0,
        ),
        "v_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=1,
        ),
        "kv_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="KV page offsets for each sequence. Set during plan(), not run().",
        ),
        "kv_indices": Tensor(
            ["num_kv_indices"],
            optional=True,
            description="Page IDs for KV cache lookups. Set during plan(), not run().",
        ),
        "sm_scale": Scalar(
            "float32",
            optional=True,
            description="Softmax scale. Default is (1/sqrt(head_dim)). Set during plan(), not run().",
        ),
    },
    outputs={
        "output": Tensor(["batch_size", "num_qo_heads", "head_dim"], dtype_from="q"),
        "lse": Tensor(
            ["batch_size", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "len_indptr == batch_size + 1",
        "num_kv_indices == kv_indptr[-1].item()",
    ],
    tags=["stage:decode", "status:verified"],
    reference=_gqa_paged_decode_reference,
)

# ── GQA paged prefill ────────────────────────────────────────────────────────


@torch.no_grad()
def _gqa_paged_prefill_reference(
    q, k_cache, v_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
):
    total_q, num_qo_heads, head_dim = q.shape
    num_pages, page_size, num_kv_heads, _ = k_cache.shape
    len_indptr = qo_indptr.shape[0]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
    )

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_flat = k_cache.reshape(-1, num_kv_heads, head_dim).to(torch.float32)
    v_flat = v_cache.reshape(-1, num_kv_heads, head_dim).to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if q_start >= q_end or kv_start >= kv_end:
            continue
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        k_b = k_flat[page_ids]
        v_b = v_flat[page_ids]
        num_kv_tokens = page_ids.shape[0]
        q_b = q_f32[q_start:q_end]
        delta = num_kv_tokens - q_b.shape[0]
        for q_idx in range(q_b.shape[0]):
            max_kv = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv <= 0:
                continue
            global_q = q_start + q_idx
            for h in range(num_qo_heads):
                kv_h = h // gqa_ratio
                logits = torch.matmul(q_b[q_idx, h], k_b[:max_kv, kv_h].T) * sm_scale
                lse[global_q, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits, dim=-1)
                output[global_q, h] = torch.matmul(attn, v_b[:max_kv, kv_h]).to(
                    torch.bfloat16
                )

    return output, lse


gqa_paged_prefill_trace = TraceTemplate(
    op_type="gqa_paged",
    name_prefix="gqa_paged_prefill",
    description=(
        "Batched Grouped Query Attention prefill with a paged KV cache. "
        "Causal mask is applied."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "page_size": Const(abbrev="ps"),
        "len_indptr": Var(description="Length of indptr arrays (batch_size + 1)."),
        "total_q": Var(description="Total number of query tokens."),
        "num_kv_indices": Var(description="Total number of KV page indices."),
        "num_pages": Var(),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=0,
        ),
        "v_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            param="paged_kv_cache",
            tuple_idx=1,
        ),
        "qo_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="Query offsets for each sequence. Set during plan(), not run().",
        ),
        "kv_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="KV page offsets for each sequence. Set during plan(), not run().",
        ),
        "kv_indices": Tensor(
            ["num_kv_indices"],
            optional=True,
            description="Page IDs for KV cache lookups. Set during plan(), not run().",
        ),
        "sm_scale": Scalar(
            "float32",
            optional=True,
            description="Softmax scale. Default is (1/sqrt(head_dim)). Set during plan(), not run().",
        ),
    },
    outputs={
        "output": Tensor(["total_q", "num_qo_heads", "head_dim"], dtype_from="q"),
        "lse": Tensor(
            ["total_q", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "total_q == qo_indptr[-1].item()",
        "num_kv_indices == kv_indptr[-1].item()",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_gqa_paged_prefill_reference,
)

# ── GQA ragged prefill ───────────────────────────────────────────────────────


@torch.no_grad()
def _gqa_ragged_prefill_reference(q, k, v, qo_indptr, kv_indptr, sm_scale):
    total_q, num_qo_heads, head_dim = q.shape
    total_kv, num_kv_heads, _ = k.shape
    len_indptr = qo_indptr.shape[0]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim), dtype=torch.bfloat16, device=q.device
    )
    lse = torch.full(
        (total_q, num_qo_heads), -float("inf"), dtype=torch.float32, device=q.device
    )

    gqa_ratio = num_qo_heads // num_kv_heads
    q_f32 = q.to(torch.float32)
    k_f32 = k.to(torch.float32)
    v_f32 = v.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if q_start >= q_end or kv_start >= kv_end:
            continue
        q_b = q_f32[q_start:q_end]  # [S, num_qo_heads, head_dim]
        k_b = k_f32[kv_start:kv_end]  # [T, num_kv_heads, head_dim]
        v_b = v_f32[kv_start:kv_end]
        num_q_tokens = q_b.shape[0]
        num_kv_tokens = k_b.shape[0]
        delta = num_kv_tokens - num_q_tokens
        for q_idx in range(num_q_tokens):
            max_kv = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv <= 0:
                continue
            global_q = q_start + q_idx
            for h in range(num_qo_heads):
                kv_h = h // gqa_ratio
                logits = torch.matmul(q_b[q_idx, h], k_b[:max_kv, kv_h].T) * sm_scale
                lse[global_q, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
                attn = torch.softmax(logits, dim=-1)
                output[global_q, h] = torch.matmul(attn, v_b[:max_kv, kv_h]).to(
                    torch.bfloat16
                )

    return output, lse


gqa_ragged_prefill_trace = TraceTemplate(
    op_type="gqa_ragged",
    name_prefix="gqa_ragged",
    description=(
        "Batched Grouped Query Attention prefill with ragged (variable-length) inputs. "
        "Causal mask is applied."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "len_indptr": Var(description="Length of indptr arrays (batch_size + 1)."),
        "total_q": Var(description="Total number of query tokens."),
        "total_kv": Var(description="Total key-value tokens across all sequences."),
    },
    inputs={
        "q": Tensor(["total_q", "num_qo_heads", "head_dim"]),
        "k": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "v": Tensor(["total_kv", "num_kv_heads", "head_dim"]),
        "qo_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="Query offsets for each sequence. Set during plan(), not run().",
        ),
        "kv_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="Key-value offsets for each sequence. Set during plan(), not run().",
        ),
        "sm_scale": Scalar(
            "float32",
            optional=True,
            description="Softmax scale. Default is (1/sqrt(head_dim)). Set during plan(), not run().",
        ),
    },
    outputs={
        "output": Tensor(
            ["total_q", "num_qo_heads", "head_dim"],
            dtype_from="q",
            description="Attention output tensor.",
        ),
        "lse": Tensor(
            ["total_q", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "total_q == qo_indptr[-1].item()",
        "total_kv == kv_indptr[-1].item()",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_gqa_ragged_prefill_reference,
)

# ── MLA paged decode (DeepSeek-V3 style) ─────────────────────────────────────


@torch.no_grad()
def _mla_paged_decode_reference(
    q_nope, q_pe, ckv_cache, kpe_cache, kv_indptr, kv_indices, sm_scale
):
    batch_size, num_qo_heads, head_dim_ckv = q_nope.shape
    len_indptr = kv_indptr.shape[0]

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (batch_size, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16,
        device=q_nope.device,
    )
    lse = torch.full(
        (batch_size, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=q_nope.device,
    )

    for b in range(batch_size):
        page_beg = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_beg >= page_end:
            output[b].zero_()
            continue
        tok_idx = kv_indices[page_beg:page_end].to(torch.long)
        Kc = Kc_all[tok_idx]  # [L, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L, head_dim_kpe]
        qn = q_nope[b].to(torch.float32)  # [num_qo_heads, head_dim_ckv]
        qp = q_pe[b].to(torch.float32)  # [num_qo_heads, head_dim_kpe]
        logits = ((qn @ Kc.T) + (qp @ Kp.T)) * sm_scale  # [num_qo_heads, L]
        lse[b] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        output[b] = (torch.softmax(logits, dim=-1) @ Kc).to(torch.bfloat16)

    return output, lse


mla_paged_decode_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="mla_paged_decode",
    description=(
        "Batched Multi-head Latent Attention decode with a paged KV cache. "
        "Used for DeepSeek-V3/R1 style models."
    ),
    axes={
        "batch_size": Var(),
        "num_qo_heads": Const(
            description="Number of query heads after tensor parallel split.",
            abbrev="h",
        ),
        "head_dim_ckv": Const(abbrev="ckv"),
        "head_dim_kpe": Const(abbrev="kpe"),
        "page_size": Const(abbrev="ps"),
        "num_pages": Var(
            description="Total number of allocated pages in the KV cache."
        ),
        "len_indptr": Var(description="Length of kv_indptr array."),
        "num_kv_indices": Var(description="Total number of KV page indices."),
    },
    inputs={
        "q_nope": Tensor(
            ["batch_size", "num_qo_heads", "head_dim_ckv"],
            description="Query tensor without positional encoding component.",
        ),
        "q_pe": Tensor(
            ["batch_size", "num_qo_heads", "head_dim_kpe"],
            description="Query positional encoding component.",
        ),
        "ckv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_ckv"],
            description="Compressed key-value cache.",
        ),
        "kpe_cache": Tensor(
            ["num_pages", "page_size", "head_dim_kpe"],
            description="Key positional encoding cache.",
        ),
        "kv_indptr": Tensor(
            ["len_indptr"],
            optional=True,
            description="KV page offsets for each sequence. Set during plan(), not run().",
        ),
        "kv_indices": Tensor(
            ["num_kv_indices"],
            optional=True,
            description="Page indices for KV cache lookups. Set during plan(), not run().",
        ),
        "sm_scale": Scalar(
            "float32",
            optional=True,
            description=(
                "Softmax scale. Default is (1/sqrt(128 + 64) = 1/sqrt(192)), "
                "based on head dimensions before matrix absorption. Set during plan(), not run()."
            ),
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "num_qo_heads", "head_dim_ckv"], dtype_from="q_nope"
        ),
        "lse": Tensor(
            ["batch_size", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "len_indptr == batch_size + 1",
        "num_kv_indices == kv_indptr[-1].item()",
    ],
    tags=["stage:decode", "status:verified"],
    reference=_mla_paged_decode_reference,
)

# ── MLA paged prefill (DeepSeek-V3 style, causal) ────────────────────────────


@torch.no_grad()
def _mla_paged_prefill_reference(
    q_nope, q_pe, ckv_cache, kpe_cache, qo_indptr, kv_indptr, kv_indices, sm_scale
):
    total_q, num_qo_heads, head_dim_ckv = q_nope.shape
    len_indptr = qo_indptr.shape[0]

    Kc_all = ckv_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_ckv]
    Kp_all = kpe_cache.squeeze(1).to(torch.float32)  # [num_pages, head_dim_kpe]

    output = torch.zeros(
        (total_q, num_qo_heads, head_dim_ckv),
        dtype=torch.bfloat16,
        device=q_nope.device,
    )
    lse = torch.full(
        (total_q, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=q_nope.device,
    )

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if q_start >= q_end or kv_start >= kv_end:
            continue
        tok_idx = kv_indices[kv_start:kv_end].to(torch.long)
        Kc = Kc_all[tok_idx]  # [L, head_dim_ckv]
        Kp = Kp_all[tok_idx]  # [L, head_dim_kpe]
        num_kv_tokens = tok_idx.shape[0]
        qn_b = q_nope[q_start:q_end].to(
            torch.float32
        )  # [S, num_qo_heads, head_dim_ckv]
        qp_b = q_pe[q_start:q_end].to(torch.float32)  # [S, num_qo_heads, head_dim_kpe]
        seq_len = q_end - q_start
        delta = num_kv_tokens - seq_len
        for q_idx in range(seq_len):
            max_kv = min(q_idx + 1 + delta, num_kv_tokens)
            if max_kv <= 0:
                continue
            global_q = q_start + q_idx
            qn = qn_b[q_idx]  # [num_qo_heads, head_dim_ckv]
            qp = qp_b[q_idx]  # [num_qo_heads, head_dim_kpe]
            logits = ((qn @ Kc[:max_kv].T) + (qp @ Kp[:max_kv].T)) * sm_scale
            lse[global_q] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            output[global_q] = (torch.softmax(logits, dim=-1) @ Kc[:max_kv]).to(
                torch.bfloat16
            )

    return output, lse


mla_paged_prefill_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="mla_paged_prefill",
    description=(
        "Batched Multi-head Latent Attention prefill with a paged KV cache. "
        "Causal mask is applied. Used for DeepSeek-V3/R1 style models."
    ),
    axes={
        "num_qo_heads": Const(
            description="Number of query heads after tensor parallel split.",
            abbrev="h",
        ),
        "head_dim_ckv": Const(abbrev="ckv"),
        "head_dim_kpe": Const(abbrev="kpe"),
        "page_size": Const(abbrev="ps"),
        "total_q": Var(description="Total number of query tokens."),
        "num_pages": Var(
            description="Total number of allocated pages in the KV cache."
        ),
        "len_indptr": Var(description="Length of indptr arrays (batch_size + 1)."),
        "num_kv_indices": Var(description="Total number of KV page indices."),
    },
    inputs={
        "q_nope": Tensor(
            ["total_q", "num_qo_heads", "head_dim_ckv"],
            description="Query tensor without positional encoding component.",
        ),
        "q_pe": Tensor(
            ["total_q", "num_qo_heads", "head_dim_kpe"],
            description="Query positional encoding component.",
        ),
        "ckv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_ckv"],
            description="Compressed key-value cache.",
        ),
        "kpe_cache": Tensor(
            ["num_pages", "page_size", "head_dim_kpe"],
            description="Key positional encoding cache.",
        ),
        "qo_indptr": Tensor(
            ["len_indptr"],
            description="Query token offsets for each sequence.",
        ),
        "kv_indptr": Tensor(
            ["len_indptr"],
            description="KV page offsets for each sequence.",
        ),
        "kv_indices": Tensor(
            ["num_kv_indices"],
            description="Page indices for KV cache lookups.",
        ),
        "sm_scale": Scalar(
            "float32",
            description=(
                "Softmax scale. Default is (1/sqrt(128 + 64) = 1/sqrt(192)), "
                "based on head dimensions before matrix absorption."
            ),
        ),
    },
    outputs={
        "output": Tensor(
            ["total_q", "num_qo_heads", "head_dim_ckv"], dtype_from="q_nope"
        ),
        "lse": Tensor(
            ["total_q", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "total_q == qo_indptr[-1].item()",
        "num_kv_indices == kv_indptr[-1].item()",
    ],
    tags=["stage:prefill", "status:verified"],
    reference=_mla_paged_prefill_reference,
)

# ── DSA (Dense Sparse Attention) paged ────────────────────────────────────────


@torch.no_grad()
def _dsa_paged_reference(q_nope, q_pe, ckv_cache, kpe_cache, sparse_indices, sm_scale):
    """
    Batched Native Sparse Attention (DSA) reference implementation.

    Uses sparse_indices to select top-K KV cache entries per token.
    Values of -1 in sparse_indices indicate padding (ignored).
    """
    num_tokens, num_qo_heads, head_dim_ckv = q_nope.shape
    head_dim_kpe = q_pe.shape[-1]
    page_size = ckv_cache.shape[1]
    device = q_nope.device

    # Squeeze page dimension when page_size=1; otherwise flatten pages.
    Kc_all = ckv_cache.reshape(-1, head_dim_ckv).to(torch.float32)
    Kp_all = kpe_cache.reshape(-1, head_dim_kpe).to(torch.float32)

    output = torch.zeros(
        (num_tokens, num_qo_heads, head_dim_ckv), dtype=torch.bfloat16, device=device
    )
    lse = torch.full(
        (num_tokens, num_qo_heads), -float("inf"), dtype=torch.float32, device=device
    )

    for t in range(num_tokens):
        indices = sparse_indices[t]
        valid_mask = indices != -1
        valid_indices = indices[valid_mask]
        if valid_indices.numel() == 0:
            output[t].zero_()
            continue
        tok_idx = valid_indices.to(torch.long)
        Kc = Kc_all[tok_idx]
        Kp = Kp_all[tok_idx]
        qn = q_nope[t].to(torch.float32)
        qp = q_pe[t].to(torch.float32)
        logits = (qn @ Kc.T) + (qp @ Kp.T)
        logits_scaled = logits * sm_scale
        lse[t] = torch.logsumexp(logits_scaled, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits_scaled, dim=-1)
        output[t] = (attn @ Kc).to(torch.bfloat16)

    return output, lse


dsa_paged_trace = TraceTemplate(
    op_type="dsa_paged",
    name_prefix="dsa_sparse_attention",
    description=(
        "Batched Native Sparse Attention (DSA) with sparse TopK KV cache selection. "
        "Uses sparse_indices to select only top-K KV cache entries per token. "
        "Supports both decode and prefill stages."
    ),
    axes={
        "num_tokens": Var(
            description="Number of tokens (batch_size for decode, total_num_tokens for prefill)."
        ),
        "num_qo_heads": Const(
            description="Number of query heads after tensor parallel split.",
            abbrev="h",
        ),
        "head_dim_ckv": Const(
            description="Compressed KV head dimension.",
            abbrev="ckv",
        ),
        "head_dim_kpe": Const(
            description="Key positional encoding dimension.",
            abbrev="kpe",
        ),
        "topk": Const(
            description="Number of top-K KV cache entries selected for sparse attention.",
            abbrev="topk",
        ),
        "page_size": Const(
            description="Page size for KV cache.",
            abbrev="ps",
        ),
        "num_pages": Var(
            description="Total number of allocated pages in the KV cache."
        ),
    },
    inputs={
        "q_nope": Tensor(
            ["num_tokens", "num_qo_heads", "head_dim_ckv"],
            description="Query tensor without positional encoding component.",
        ),
        "q_pe": Tensor(
            ["num_tokens", "num_qo_heads", "head_dim_kpe"],
            description="Query positional encoding component.",
        ),
        "ckv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_ckv"],
            description="Compressed key-value cache.",
        ),
        "kpe_cache": Tensor(
            ["num_pages", "page_size", "head_dim_kpe"],
            description="Key positional encoding cache.",
        ),
        "sparse_indices": Tensor(
            ["num_tokens", "topk"],
            description="Sparse indices selecting top-K KV cache entries per token. -1 = padding.",
        ),
        "sm_scale": Scalar(
            "float32",
            description=(
                "Softmax scale. For MLA pre-absorption: 1/sqrt(head_dim_qk + head_dim_kpe)."
            ),
        ),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "num_qo_heads", "head_dim_ckv"],
            dtype_from="q_nope",
            description="Attention output tensor.",
        ),
        "lse": Tensor(
            ["num_tokens", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    constraints=[
        "sparse_indices.shape[0] == num_tokens",
        "sparse_indices.shape[-1] == topk",
        "ckv_cache.shape[1] == page_size",
    ],
    tags=["status:verified", "sparse:topk"],
    reference=_dsa_paged_reference,
)
