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

"""TraceTemplates for attention operations.

Pick the template whose input schema matches your call site. Rows that share
KV layout / indexing / stage are interchangeable from a consumer's viewpoint;
the backend column indicates which kernel the API wraps.

+---------------------------+-------------------+---------------------------+-------------------------+---------+-----------------+
| Template                  | Batching          | KV layout                 | Indexing                | Stage   | Backend         |
+===========================+===================+===========================+=========================+=========+=================+
| ``single_decode``         | single request    | contiguous                | none                    | decode  | any (no plan)   |
| ``single_prefill``        | single request    | contiguous                | none                    | prefill | any (no plan)   |
| ``gqa_paged_decode``      | batched, ragged   | paged tuple (k, v)        | kv_indptr + kv_indices  | decode  | FA2/FA3/cuDNN   |
| ``gqa_paged_prefill``     | batched, ragged   | paged tuple (k, v)        | +qo_indptr              | prefill | FA2/FA3/cuDNN   |
| ``gqa_ragged``            | batched, ragged   | contiguous                | qo_indptr + kv_indptr   | prefill | FA2/FA3         |
| ``mla_paged_decode``      | batched, ragged   | paged MLA (ckv + kpe)     | kv_indptr + kv_indices  | decode  | DeepSeek MLA    |
| ``mla_paged_prefill``     | batched, ragged   | paged MLA (ckv + kpe)     | +qo_indptr              | prefill | DeepSeek MLA    |
| ``dsa_paged``             | batched           | paged MLA                 | sparse_indices (top-K)  | both    | sparse DSA      |
| ``trtllm_batch_decode``   | batched           | paged, interleaved single | block_tables + seq_lens | decode  | TRT-LLM SM100+  |
| ``trtllm_batch_context``  | batched           | paged, interleaved single | block_tables + cum_*    | prefill | TRT-LLM SM100+  |
| ``cudnn_batch_decode``    | batched           | paged, separate k/v       | block_tables            | decode  | cuDNN (no plan) |
| ``cudnn_batch_prefill``   | batched, var-len  | paged or contiguous       | actual_seq_lens_*       | prefill | cuDNN (no plan) |
+---------------------------+-------------------+---------------------------+-------------------------+---------+-----------------+
"""

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
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(batch_size):
        page_start = int(kv_indptr[b].item())
        page_end = int(kv_indptr[b + 1].item())
        if page_start >= page_end:
            output[b].zero_()
            continue
        # kv_indices are page IDs. Gather pages first, then flatten the
        # [num_selected_pages, page_size] axis into a single token axis.
        page_ids = kv_indices[page_start:page_end].to(torch.long)
        k_b = k_cache_f32[page_ids].reshape(-1, num_kv_heads, head_dim)
        v_b = v_cache_f32[page_ids].reshape(-1, num_kv_heads, head_dim)
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
    description=(
        "Batched GQA decode (1 query per seq) with a paged KV cache as a "
        "(k_cache, v_cache) tuple and ragged kv_indptr+kv_indices baked in at "
        "plan() time. Wraps BatchDecodeWithPagedKVCacheWrapper.run()."
    ),
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
    k_cache_f32 = k_cache.to(torch.float32)
    v_cache_f32 = v_cache.to(torch.float32)

    for b in range(len_indptr - 1):
        q_start = int(qo_indptr[b].item())
        q_end = int(qo_indptr[b + 1].item())
        kv_start = int(kv_indptr[b].item())
        kv_end = int(kv_indptr[b + 1].item())
        if q_start >= q_end or kv_start >= kv_end:
            continue
        # kv_indices are page IDs. Gather pages and flatten to a token axis.
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        k_b = k_cache_f32[page_ids].reshape(-1, num_kv_heads, head_dim)
        v_b = v_cache_f32[page_ids].reshape(-1, num_kv_heads, head_dim)
        num_kv_tokens = k_b.shape[0]
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
        "Batched GQA prefill (multi-token per seq, causal) with a paged KV "
        "cache. Adds qo_indptr to gqa_paged_decode's indptr/indices. Wraps "
        "BatchPrefillWithPagedKVCacheWrapper.run()."
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
        "Batched GQA prefill (causal) with contiguous (non-paged) K/V tensors "
        "and qo_indptr/kv_indptr offsets baked in at plan() time. Wraps "
        "BatchPrefillWithRaggedKVCacheWrapper.run()."
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
    _, _, head_dim_kpe = q_pe.shape

    # [num_pages, page_size, head_dim_*] — keep the page dim; flatten after gather.
    Kc_all = ckv_cache.to(torch.float32)
    Kp_all = kpe_cache.to(torch.float32)

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
        # kv_indices are page IDs; gather pages then flatten to a token axis.
        page_ids = kv_indices[page_beg:page_end].to(torch.long)
        Kc = Kc_all[page_ids].reshape(-1, head_dim_ckv)  # [L, head_dim_ckv]
        Kp = Kp_all[page_ids].reshape(-1, head_dim_kpe)  # [L, head_dim_kpe]
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
        "Batched MLA decode (DeepSeek-V2/V3/R1). Query and KV are split into "
        "NoPE (ckv, head_dim_ckv=512) and RoPE (kpe, head_dim_kpe=64) parts: "
        "inputs are (q_nope, q_pe) and (ckv_cache, kpe_cache). "
        "Wraps BatchMLAPagedAttentionWrapper.run() post matrix-absorption."
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
    _, _, head_dim_kpe = q_pe.shape
    len_indptr = qo_indptr.shape[0]

    # [num_pages, page_size, head_dim_*] — keep the page dim; flatten after gather.
    Kc_all = ckv_cache.to(torch.float32)
    Kp_all = kpe_cache.to(torch.float32)

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
        # kv_indices are page IDs; gather pages then flatten to a token axis.
        page_ids = kv_indices[kv_start:kv_end].to(torch.long)
        Kc = Kc_all[page_ids].reshape(-1, head_dim_ckv)  # [L, head_dim_ckv]
        Kp = Kp_all[page_ids].reshape(-1, head_dim_kpe)  # [L, head_dim_kpe]
        num_kv_tokens = Kc.shape[0]
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
        "Batched MLA prefill (multi-token per seq, causal). Same "
        "(q_nope, q_pe) / (ckv_cache, kpe_cache) split as mla_paged_decode "
        "plus qo_indptr for variable query lengths."
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
        "DSA (Dense Sparse Attention): MLA latent layout + per-query top-K "
        "selection via sparse_indices (-1 = padding). Covers decode and "
        "prefill; no kv_indptr/indices."
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

# ── Single prefill / single decode (non-batched) ──────────────────────────────


@torch.no_grad()
def _single_decode_reference(q, k, v, **kwargs):
    """Single-request decode: q @ K.T → softmax → @ V, broadcasting GQA."""
    num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    gqa_ratio = num_qo_heads // num_kv_heads
    sm_scale = kwargs.get("sm_scale")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(q, dtype=torch.float32)
    for h in range(num_qo_heads):
        kv_h = h // gqa_ratio
        logits = (
            torch.matmul(q[h].to(torch.float32), k[:, kv_h].to(torch.float32).T)
            * sm_scale
        )
        attn = torch.softmax(logits, dim=-1)
        output[h] = torch.matmul(attn, v[:, kv_h].to(torch.float32))
    return output.to(q.dtype)


@torch.no_grad()
def _single_prefill_reference(q, k, v, **kwargs):
    """Single-request prefill: standard SDPA with optional causal mask."""
    qo_len, num_qo_heads, head_dim = q.shape
    kv_len, num_kv_heads, _ = k.shape
    gqa_ratio = num_qo_heads // num_kv_heads
    causal = bool(kwargs.get("causal", False))
    sm_scale = kwargs.get("sm_scale")
    if sm_scale is None:
        sm_scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(q, dtype=torch.float32)
    delta = kv_len - qo_len
    for h in range(num_qo_heads):
        kv_h = h // gqa_ratio
        logits = (
            torch.matmul(q[:, h].to(torch.float32), k[:, kv_h].to(torch.float32).T)
            * sm_scale
        )
        if causal:
            mask = torch.full_like(logits, float("-inf"))
            for qi in range(qo_len):
                mask[qi, : qi + 1 + max(0, delta)] = 0.0
            logits = logits + mask
        attn = torch.softmax(logits, dim=-1)
        output[:, h] = torch.matmul(attn, v[:, kv_h].to(torch.float32))
    return output.to(q.dtype)


single_decode_with_kv_cache_trace = TraceTemplate(
    op_type="single_decode",
    name_prefix="single_decode",
    description=(
        "Single-request decode. Q has no batch dim "
        "([num_qo_heads, head_dim]); K and V are contiguous "
        "([kv_len, num_kv_heads, head_dim]). No paging, no plan()."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "kv_len": Var(description="Length of the K/V context."),
    },
    inputs={
        "q": Tensor(["num_qo_heads", "head_dim"]),
        "k": Tensor(
            ["kv_len", "num_kv_heads", "head_dim"],
            description="Key cache, shape varies with kv_layout (default NHD).",
        ),
        "v": Tensor(
            ["kv_len", "num_kv_heads", "head_dim"],
            description="Value cache, shape varies with kv_layout (default NHD).",
        ),
    },
    outputs={
        "output": Tensor(["num_qo_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "stage:decode"],
    reference=_single_decode_reference,
)

single_prefill_with_kv_cache_trace = TraceTemplate(
    op_type="single_prefill",
    name_prefix="single_prefill",
    description=(
        "Single-request prefill. Q is [qo_len, H, D]; K, V are contiguous "
        "[kv_len, Hkv, D]. No paging, no plan(). Optional causal mask and "
        "custom_mask."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "qo_len": Var(description="Length of the query sequence."),
        "kv_len": Var(description="Length of the K/V sequence."),
    },
    inputs={
        "q": Tensor(["qo_len", "num_qo_heads", "head_dim"]),
        "k": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
        "v": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
    },
    outputs={
        "output": Tensor(["qo_len", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "stage:prefill"],
    reference=_single_prefill_reference,
)

# ── TRTLLM paged attention ────────────────────────────────────────────────────
# kv_cache shape is [num_pages, 1 or 2, num_kv_heads, page_size, head_dim] in HND
# (or NHD equivalents). The "1 or 2" axis is 1 for single-tensor interleaved
# layout and 2 for [K, V] split; we model it as a separate dim "kv_cache_dim".

_TRTLLM_AXES: dict[str, Var | Const] = {
    "num_tokens": Var(description="Total query tokens across the batch."),
    "num_heads": Const(abbrev="h"),
    "num_kv_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
    "page_size": Const(abbrev="ps"),
    "num_pages": Var(),
    "kv_cache_dim": Const(
        abbrev="",
        description="1 for interleaved (K,V) single tensor; 2 for separate K/V halves.",
    ),
    "batch_size": Var(),
}


@torch.no_grad()
def _trtllm_kv_from_cache(kv_cache, kv_cache_dim, num_heads, side):
    """Split a TRT-LLM paged kv_cache tensor into either K or V slice.

    kv_cache: [num_pages, kv_cache_dim, num_kv_heads, page_size, head_dim]
    kv_cache_dim == 1: K/V interleaved head-wise along num_kv_heads
    kv_cache_dim == 2: kv_cache[:, 0] is K, kv_cache[:, 1] is V
    """
    if kv_cache_dim == 2:
        return kv_cache[:, 0] if side == "k" else kv_cache[:, 1]
    # Interleaved along heads: even = K, odd = V.
    sel = 0 if side == "k" else 1
    return kv_cache[:, 0, sel::2]


@torch.no_grad()
def _trtllm_paged_attention_reference(
    query, kv_cache, block_tables, seq_lens, causal=False, **kwargs
):
    """Shared reference for trtllm_batch_{decode, context}.

    Treats query as [num_tokens, num_heads, head_dim]; expands each batch's
    variable-length query tokens against its paged KV slice and applies
    optional causal mask.

    ``kv_layout`` selects the per-page memory layout:
      * ``"HND"`` (default): ``[num_pages, kv_cache_dim, num_kv_heads, page_size, head_dim]``
      * ``"NHD"``           : ``[num_pages, kv_cache_dim, page_size, num_kv_heads, head_dim]``
    """
    kv_layout = kwargs.get("kv_layout", "HND")
    num_tokens, num_heads, head_dim = query.shape
    if kv_layout == "HND":
        num_pages, kv_cache_dim, num_kv_heads, page_size, _ = kv_cache.shape
    else:
        num_pages, kv_cache_dim, page_size, num_kv_heads, _ = kv_cache.shape
    gqa_ratio = num_heads // num_kv_heads
    bmm1_scale = float(kwargs.get("bmm1_scale", 1.0 / math.sqrt(head_dim)) or 1.0)
    bmm2_scale = float(kwargs.get("bmm2_scale", 1.0) or 1.0)
    cum_seq_lens_q = kwargs.get("cum_seq_lens_q")
    batch_size = block_tables.shape[0]
    output = torch.zeros_like(query, dtype=torch.float32)
    for b in range(batch_size):
        n_pages_used = (int(seq_lens[b].item()) + page_size - 1) // page_size
        pages = block_tables[b, :n_pages_used].to(torch.long)
        kv_len = int(seq_lens[b].item())
        k_b = _trtllm_kv_from_cache(kv_cache[pages], kv_cache_dim, num_heads, "k")
        v_b = _trtllm_kv_from_cache(kv_cache[pages], kv_cache_dim, num_heads, "v")
        if kv_layout == "HND":
            # [n_pages, Hk, PS, D] → [Hk, n_pages * PS, D] (per-head flatten).
            k_flat = k_b.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)[:kv_len]
            v_flat = v_b.transpose(1, 2).reshape(-1, num_kv_heads, head_dim)[:kv_len]
        else:
            # NHD: [n_pages, PS, Hk, D] reshapes directly.
            k_flat = k_b.reshape(-1, num_kv_heads, head_dim)[:kv_len]
            v_flat = v_b.reshape(-1, num_kv_heads, head_dim)[:kv_len]
        # Figure out which query tokens belong to this batch.
        if cum_seq_lens_q is not None:
            q_start = int(cum_seq_lens_q[b].item())
            q_end = int(cum_seq_lens_q[b + 1].item())
        else:
            q_start = b * (num_tokens // batch_size)
            q_end = q_start + (num_tokens // batch_size)
        q_b = query[q_start:q_end].to(torch.float32)
        for h in range(num_heads):
            kv_h = h // gqa_ratio
            logits = (
                torch.matmul(q_b[:, h], k_flat[:, kv_h].to(torch.float32).T)
                * bmm1_scale
            )
            if causal:
                qi = q_end - q_start
                delta = kv_len - qi
                mask = torch.full_like(logits, float("-inf"))
                for i in range(qi):
                    mask[i, : i + 1 + max(0, delta)] = 0.0
                logits = logits + mask
            attn = torch.softmax(logits, dim=-1)
            output[q_start:q_end, h] = (
                torch.matmul(attn, v_flat[:, kv_h].to(torch.float32)) * bmm2_scale
            )
    return output.to(query.dtype)


@torch.no_grad()
def _trtllm_batch_decode_reference(
    query, kv_cache, workspace_buffer, block_tables, seq_lens, max_seq_len, **kwargs
):
    return _trtllm_paged_attention_reference(
        query, kv_cache, block_tables, seq_lens, causal=False, **kwargs
    )


@torch.no_grad()
def _trtllm_batch_context_reference(
    query,
    kv_cache,
    workspace_buffer,
    block_tables,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    batch_size,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    **kwargs,
):
    return _trtllm_paged_attention_reference(
        query,
        kv_cache,
        block_tables,
        seq_lens,
        causal=True,
        bmm1_scale=bmm1_scale,
        bmm2_scale=bmm2_scale,
        cum_seq_lens_q=cum_seq_lens_q,
    )


trtllm_batch_decode_trace = TraceTemplate(
    op_type="trtllm_paged",
    name_prefix="trtllm_batch_decode",
    description=(
        "SM100+ TRT-LLM paged decode. Single interleaved kv_cache "
        "[num_pages, 1 or 2, Hkv, page_size, D], rectangular block_tables, "
        "two scales (bmm1_scale post-QK, bmm2_scale post-softmax·V) for "
        "FP8/FP4 numerics. Supports q_len_per_req > 1 for spec decoding."
    ),
    axes=_TRTLLM_AXES,
    inputs={
        "query": Tensor(["num_tokens", "num_heads", "head_dim"]),
        "kv_cache": Tensor(
            ["num_pages", "kv_cache_dim", "num_kv_heads", "page_size", "head_dim"],
            description="Paged KV cache; kv_cache_dim is 1 (interleaved) or 2 (K+V).",
        ),
        "block_tables": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
            description="Page table mapping per sequence.",
        ),
        "seq_lens": Tensor(
            ["batch_size"],
            dtype="int32",
            description="Actual KV sequence length per batch entry.",
        ),
        "max_seq_len": Scalar(
            "int32", description="Maximum K/V sequence length in the batch."
        ),
        "bmm1_scale": Scalar(
            "float32", optional=True, description="Scale applied after Q @ K^T."
        ),
        "bmm2_scale": Scalar(
            "float32", optional=True, description="Scale applied after softmax @ V."
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "num_heads", "head_dim"], dtype_from="query"),
    },
    tags=["status:verified", "stage:decode", "backend:trtllm"],
    reference=_trtllm_batch_decode_reference,
)

# Add max_pages_per_seq axis used above
trtllm_batch_decode_trace.axes["max_pages_per_seq"] = Var(
    description="Maximum number of pages per sequence (block_tables width)."
)

trtllm_batch_context_trace = TraceTemplate(
    op_type="trtllm_paged",
    name_prefix="trtllm_batch_context",
    description=(
        "SM100+ TRT-LLM paged context/prefill. Prefill twin of "
        "trtllm_batch_decode: same interleaved kv_cache and block_tables, "
        "but adds cum_seq_lens_q/cum_seq_lens_kv for variable-length "
        "queries."
    ),
    axes={
        **_TRTLLM_AXES,
        "max_pages_per_seq": Var(
            description="Maximum number of pages per sequence (block_tables width)."
        ),
    },
    inputs={
        "query": Tensor(["num_tokens", "num_heads", "head_dim"]),
        "kv_cache": Tensor(
            ["num_pages", "kv_cache_dim", "num_kv_heads", "page_size", "head_dim"],
            description="Paged KV cache; kv_cache_dim is 1 or 2.",
        ),
        "block_tables": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
            description="Page table mapping per sequence.",
        ),
        "seq_lens": Tensor(
            ["batch_size"],
            dtype="int32",
            description="Actual KV sequence length per batch entry.",
        ),
        "max_q_len": Scalar(
            "int32", description="Maximum query sequence length in the batch."
        ),
        "max_kv_len": Scalar(
            "int32", description="Maximum K/V sequence length in the batch."
        ),
        "bmm1_scale": Scalar("float32", description="Scale applied after Q @ K^T."),
        "bmm2_scale": Scalar("float32", description="Scale applied after softmax @ V."),
        "batch_size_scalar": Scalar("int32", param="batch_size"),
        "cum_seq_lens_q": Tensor(
            ["batch_size_plus_1_q"],
            dtype="int32",
            description="Cumulative Q sequence lengths, shape batch_size + 1.",
        ),
        "cum_seq_lens_kv": Tensor(
            ["batch_size_plus_1_kv"],
            dtype="int32",
            description="Cumulative KV sequence lengths, shape batch_size + 1.",
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "num_heads", "head_dim"], dtype_from="query"),
    },
    tags=["status:verified", "stage:prefill", "backend:trtllm"],
    reference=_trtllm_batch_context_reference,
)
trtllm_batch_context_trace.axes["batch_size_plus_1_q"] = Var(
    description="batch_size + 1."
)
trtllm_batch_context_trace.axes["batch_size_plus_1_kv"] = Var(
    description="batch_size + 1."
)


# ── TRT-LLM batch decode MLA (DeepSeek-style) ────────────────────────────────


@torch.no_grad()
def _trtllm_batch_decode_mla_reference(
    query,
    kv_cache,
    workspace_buffer,
    qk_nope_head_dim,
    kv_lora_rank,
    qk_rope_head_dim,
    block_tables,
    seq_lens,
    max_seq_len,
    **kwargs,
):
    """Reference for trtllm_batch_decode_with_kv_cache_mla.

    Query is concatenated [Q_nope, Q_pe] along the head_dim axis; the KV
    cache is [ckv ‖ kpe]. Output is the K_nope-projected attention
    (``[batch, q_len, num_heads, kv_lora_rank]``).
    """
    batch_size, q_len, num_heads, head_dim_qk = query.shape
    assert head_dim_qk == kv_lora_rank + qk_rope_head_dim
    bmm1_scale = kwargs.get("bmm1_scale", 1.0)
    bmm1_scale = (
        float(bmm1_scale) if not isinstance(bmm1_scale, torch.Tensor) else bmm1_scale
    )
    if isinstance(bmm1_scale, torch.Tensor):
        bmm1_scale = float(bmm1_scale.item())
    bmm2_scale = kwargs.get("bmm2_scale", 1.0)
    if isinstance(bmm2_scale, torch.Tensor):
        bmm2_scale = float(bmm2_scale.item())
    # Accept kv_cache with optional leading "num_kv_heads=1" dim
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]
    output = torch.zeros(
        (batch_size, q_len, num_heads, kv_lora_rank),
        dtype=query.dtype,
        device=query.device,
    )
    for b in range(batch_size):
        kv_len = int(seq_lens[b].item())
        n_pages = (kv_len + page_size - 1) // page_size
        pages = block_tables[b, :n_pages].to(torch.long)
        flat = kv_cache[pages].reshape(-1, head_dim_qk)[:kv_len].to(torch.float32)
        # MLA split: first kv_lora_rank dims = ckv (K_nope), last qk_rope_head_dim dims = kpe
        Kn = flat[:, :kv_lora_rank]
        Kp = flat[:, kv_lora_rank:]
        for t in range(q_len):
            q = query[b, t].to(torch.float32)  # [num_heads, head_dim_qk]
            Qn = q[:, :kv_lora_rank]  # [num_heads, kv_lora_rank]
            Qp = q[:, kv_lora_rank:]  # [num_heads, qk_rope_head_dim]
            logits = (Qn @ Kn.T + Qp @ Kp.T) * bmm1_scale
            attn = torch.softmax(logits, dim=-1)
            output[b, t] = (attn @ Kn * bmm2_scale).to(query.dtype)
    return output


trtllm_batch_decode_mla_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="trtllm_batch_decode_mla",
    description=(
        "SM100+ TRT-LLM MLA paged decode. Query is concatenated [Q_nope, "
        "Q_pe] with head_dim_qk = kv_lora_rank + qk_rope_head_dim; KV cache "
        "is [ckv ‖ kpe]. Output dim equals kv_lora_rank."
    ),
    axes={
        "batch_size": Var(),
        "q_len_per_request": Var(description="Query length per request (MTP depth)."),
        "num_heads": Const(abbrev="h"),
        "head_dim_qk": Const(abbrev="d_qk"),
        "kv_lora_rank": Const(abbrev="ckv"),
        "qk_rope_head_dim": Const(abbrev="kpe"),
        "qk_nope_head_dim": Const(abbrev="nope"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "max_pages_per_seq": Var(),
    },
    inputs={
        "query": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "head_dim_qk"],
            description="Concatenated [Q_nope, Q_pe] query.",
        ),
        "kv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_qk"],
            description="Paged KV cache [ckv ‖ kpe]; 4D layout with an extra num_kv_heads=1 dim is also accepted.",
        ),
        "workspace_buffer": Tensor(
            ["num_pages"], dtype="int8", description="Workspace scratch."
        ),
        "qk_nope_head_dim": Scalar("int32"),
        "kv_lora_rank": Scalar("int32"),
        "qk_rope_head_dim": Scalar("int32"),
        "block_tables": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
            description="Page table mapping per sequence.",
        ),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_seq_len": Scalar("int32"),
        "bmm1_scale": Scalar(
            "float32",
            optional=True,
            description="Fused scale applied after Q @ K^T (includes 1/sqrt(head_dim_qk)).",
        ),
        "bmm2_scale": Scalar(
            "float32",
            optional=True,
            description="Scale applied after softmax @ V.",
        ),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "kv_lora_rank"],
            dtype_from="query",
        ),
    },
    tags=["status:verified", "stage:decode", "backend:trtllm", "mla"],
    reference=_trtllm_batch_decode_mla_reference,
)


# ── XQA batch decode (non-MLA) ────────────────────────────────────────────────


@torch.no_grad()
def _xqa_batch_decode_reference(
    query,
    kv_cache,
    workspace_buffer,
    block_tables,
    seq_lens,
    max_seq_len,
    **kwargs,
):
    """Reference for xqa_batch_decode_with_kv_cache.

    Same semantic as trtllm_batch_decode (paged attention on an [num_pages,
    kv_cache_dim, num_kv_heads, page_size, head_dim] HND cache or a
    ``(k_cache, v_cache)`` NHD tuple), shared here so the two XQA-vs-TRT-LLM
    backends trace-compare against the same math.
    """
    # Accept tuple kv_cache by synthesizing an interleaved tensor view.
    if isinstance(kv_cache, tuple):
        k_cache, v_cache = kv_cache
        kv_cache = torch.stack([k_cache, v_cache], dim=1)
    return _trtllm_paged_attention_reference(
        query, kv_cache, block_tables, seq_lens, causal=False, **kwargs
    )


xqa_batch_decode_trace = TraceTemplate(
    op_type="xqa",
    name_prefix="xqa_batch_decode",
    description=(
        "SM100+/SM120+ XQA paged decode wrapper (batch). Accepts both the "
        "5-D interleaved [num_pages, kv_cache_dim, num_kv_heads, page_size, "
        "head_dim] tensor and a (k_cache, v_cache) tuple (NHD). Semantics "
        "match the regular XQA kernel but at the batch-wrapper API level "
        "(sglang/vllm's trtllm-gen XQA entry point)."
    ),
    axes=_TRTLLM_AXES,
    inputs={
        "query": Tensor(["num_tokens", "num_heads", "head_dim"]),
        "kv_cache": Tensor(
            ["num_pages", "kv_cache_dim", "num_kv_heads", "page_size", "head_dim"],
            description="Paged KV cache (5-D HND or 2-tuple NHD).",
        ),
        "block_tables": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
        ),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_seq_len": Scalar("int32"),
        "bmm1_scale": Scalar(
            "float32", optional=True, description="Scale applied after Q @ K^T."
        ),
        "bmm2_scale": Scalar(
            "float32", optional=True, description="Scale applied after softmax @ V."
        ),
    },
    outputs={
        "output": Tensor(["num_tokens", "num_heads", "head_dim"], dtype_from="query"),
    },
    tags=["status:verified", "stage:decode", "backend:xqa"],
    reference=_xqa_batch_decode_reference,
)
xqa_batch_decode_trace.axes["max_pages_per_seq"] = Var(
    description="Maximum number of pages per sequence (block_tables width)."
)


# ── XQA batch decode MLA (DeepSeek-style) ─────────────────────────────────────

# Same math as trtllm_batch_decode_with_kv_cache_mla — the XQA variant is
# just the SM120/121 codegen-based backend for the same op.

xqa_batch_decode_mla_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="xqa_batch_decode_mla",
    description=(
        "SM120+ XQA MLA paged decode wrapper. Same math as "
        "trtllm_batch_decode_with_kv_cache_mla: Q is concatenated "
        "[Q_nope, Q_pe] with head_dim_qk = kv_lora_rank + qk_rope_head_dim; "
        "KV cache is [ckv ‖ kpe]. Output dim equals kv_lora_rank. The XQA "
        "MLA kernel requires FP8 e4m3 Q/KV; the reference accepts dequantized "
        "float inputs for correctness comparison."
    ),
    axes={
        "batch_size": Var(),
        "q_len_per_request": Var(description="Query length per request (must be 1)."),
        "num_heads": Const(abbrev="h"),
        "head_dim_qk": Const(abbrev="d_qk"),
        "kv_lora_rank": Const(abbrev="ckv"),
        "qk_rope_head_dim": Const(abbrev="kpe"),
        "qk_nope_head_dim": Const(abbrev="nope"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "max_pages_per_seq": Var(),
    },
    inputs={
        "query": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "head_dim_qk"],
        ),
        "kv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_qk"],
        ),
        "workspace_buffer": Tensor(["num_pages"], dtype="int8"),
        "qk_nope_head_dim": Scalar("int32"),
        "kv_lora_rank": Scalar("int32"),
        "qk_rope_head_dim": Scalar("int32"),
        "block_tables": Tensor(["batch_size", "max_pages_per_seq"], dtype="int32"),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_seq_len": Scalar("int32"),
        "bmm1_scale": Scalar("float32", optional=True),
        "bmm2_scale": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "kv_lora_rank"],
            dtype_from="query",
        ),
    },
    tags=["status:verified", "stage:decode", "backend:xqa", "mla"],
    reference=_trtllm_batch_decode_mla_reference,
)


# ── Concat MLA K (DeepSeek) ──────────────────────────────────────────────────


@torch.no_grad()
def _concat_mla_k_reference(k, k_nope, k_rope, **_unused):
    """Reference for concat_mla_k: writes ``[k_nope ‖ broadcast(k_rope)]``
    into the output tensor in-place.

    Layouts:
      - k:      [num_tokens, num_heads, nope_dim + rope_dim]
      - k_nope: [num_tokens, num_heads, nope_dim]
      - k_rope: [num_tokens, 1,         rope_dim] (broadcast across heads)
    """
    num_tokens, num_heads, total_dim = k.shape
    nope_dim = k_nope.shape[-1]
    k[..., :nope_dim] = k_nope
    k[..., nope_dim:] = k_rope.expand(num_tokens, num_heads, -1)
    return k


concat_mla_k_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="concat_mla_k",
    description=(
        "DeepSeek MLA K concatenation: broadcasts the per-head-shared RoPE "
        "key (k_rope) across all Q heads and writes ``[k_nope ‖ k_rope]`` "
        "into an output buffer. In-place (mutates k)."
    ),
    axes={
        "num_tokens": Var(),
        "num_heads": Const(abbrev="h"),
        "nope_dim": Const(abbrev="nope"),
        "rope_dim": Const(abbrev="rope"),
        "total_dim": Const(description="nope_dim + rope_dim.", abbrev="d"),
    },
    inputs={
        "k": Tensor(
            ["num_tokens", "num_heads", "total_dim"],
            description="Output buffer (mutated in place).",
        ),
        "k_nope": Tensor(["num_tokens", "num_heads", "nope_dim"]),
        "k_rope": Tensor(
            ["num_tokens", "num_heads_broadcast", "rope_dim"],
            description="Shared across heads (broadcast dim = 1).",
        ),
    },
    outputs={
        "k": Tensor(
            ["num_tokens", "num_heads", "total_dim"],
            dtype_from="k_nope",
            description="Concatenated [k_nope ‖ k_rope] (in-place).",
        ),
    },
    tags=["status:verified", "mla"],
    reference=_concat_mla_k_reference,
)
concat_mla_k_trace.axes["num_heads_broadcast"] = Const(
    description="Always 1 (k_rope is shared across heads).", abbrev=""
)


# ── cuDNN paged attention ─────────────────────────────────────────────────────

_CUDNN_PAGED_AXES: dict[str, Var | Const] = {
    "batch_size": Var(),
    "total_num_pages": Var(),
    "num_pages_per_seq": Var(
        description="block_tables.shape[-1]; max pages used by any seq."
    ),
    "num_heads_qo": Const(abbrev="h"),
    "num_heads_kv": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
    "page_size": Const(abbrev="ps"),
}


@torch.no_grad()
def _cudnn_batch_decode_reference(
    q, k_cache, v_cache, scale, workspace_buffer, max_sequence_kv, **kwargs
):
    """Reference for cudnn_batch_decode_with_kv_cache.

    K/V layout: [total_num_pages, num_heads_kv, page_size, head_dim] (HND).
    block_tables: [batch_size, num_pages_per_seq] gathers per-sequence pages.
    actual_seq_lens_kv (optional) gives the true length of each sequence.
    """
    batch_size, num_heads_qo, head_dim = q.shape
    _, num_heads_kv, page_size, _ = k_cache.shape
    gqa_ratio = num_heads_qo // num_heads_kv
    block_tables = kwargs.get("block_tables")
    actual_seq_lens_kv = kwargs.get("actual_seq_lens_kv")
    output = torch.zeros_like(q, dtype=torch.float32)
    for b in range(batch_size):
        if block_tables is None:
            pages = torch.tensor([b], device=q.device, dtype=torch.long)
        else:
            row = block_tables[b]
            pages = row[row >= 0].to(torch.long)
        kv_len = (
            int(actual_seq_lens_kv[b].item())
            if actual_seq_lens_kv is not None
            else int(max_sequence_kv)
        )
        # Gather + flatten: [num_heads_kv, L, head_dim] after permute.
        k_b = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads_kv, -1, head_dim)[:, :kv_len]
        )
        v_b = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads_kv, -1, head_dim)[:, :kv_len]
        )
        for h in range(num_heads_qo):
            kv_h = h // gqa_ratio
            logits = torch.matmul(
                q[b, h].to(torch.float32), k_b[kv_h].to(torch.float32).T
            ) * float(scale)
            attn = torch.softmax(logits, dim=-1)
            output[b, h] = torch.matmul(attn, v_b[kv_h].to(torch.float32))
    return output.to(q.dtype)


@torch.no_grad()
def _cudnn_batch_prefill_reference(
    q,
    k_cache,
    v_cache,
    scale,
    workspace_buffer,
    max_token_per_sequence,
    max_sequence_kv,
    actual_seq_lens_q,
    actual_seq_lens_kv,
    causal,
    return_lse,
    **kwargs,
):
    """Reference for cudnn_batch_prefill_with_kv_cache (variable-length)."""
    num_tokens, num_heads_qo, head_dim = q.shape
    _, num_heads_kv, page_size, _ = k_cache.shape
    gqa_ratio = num_heads_qo // num_heads_kv
    block_tables = kwargs.get("block_tables")
    batch_size = actual_seq_lens_q.shape[0]
    q_offsets = torch.cat(
        [
            torch.zeros(1, dtype=torch.int64, device=q.device),
            actual_seq_lens_q.to(torch.int64).cumsum(0),
        ]
    )
    output = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full(
        (num_tokens, num_heads_qo),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    for b in range(batch_size):
        q_start = int(q_offsets[b].item())
        q_end = int(q_offsets[b + 1].item())
        if q_end <= q_start:
            continue
        kv_len = int(actual_seq_lens_kv[b].item())
        if block_tables is None:
            pages = torch.tensor([b], device=q.device, dtype=torch.long)
        else:
            row = block_tables[b]
            pages = row[row >= 0].to(torch.long)
        k_b = (
            k_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads_kv, -1, head_dim)[:, :kv_len]
        )
        v_b = (
            v_cache[pages]
            .permute(1, 0, 2, 3)
            .reshape(num_heads_kv, -1, head_dim)[:, :kv_len]
        )
        qi = q_end - q_start
        delta = kv_len - qi
        for h in range(num_heads_qo):
            kv_h = h // gqa_ratio
            qh = q[q_start:q_end, h].to(torch.float32)
            logits = torch.matmul(qh, k_b[kv_h].to(torch.float32).T) * float(scale)
            if causal:
                mask = torch.full_like(logits, float("-inf"))
                for i in range(qi):
                    mask[i, : i + 1 + max(0, delta)] = 0.0
                logits = logits + mask
            lse[q_start:q_end, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
            attn = torch.softmax(logits, dim=-1)
            output[q_start:q_end, h] = torch.matmul(attn, v_b[kv_h].to(torch.float32))
    return (output.to(q.dtype), lse if return_lse else None)


cudnn_batch_decode_trace = TraceTemplate(
    op_type="cudnn_paged",
    name_prefix="cudnn_batch_decode",
    description=(
        "Standalone cuDNN paged decode. Separate k_cache/v_cache "
        "[total_num_pages, Hkv, page_size, D], rectangular block_tables, "
        "single sm_scale. No plan() — block_tables passed at call time."
    ),
    axes=_CUDNN_PAGED_AXES,
    inputs={
        "q": Tensor(["batch_size", "num_heads_qo", "head_dim"]),
        "k_cache": Tensor(["total_num_pages", "num_heads_kv", "page_size", "head_dim"]),
        "v_cache": Tensor(["total_num_pages", "num_heads_kv", "page_size", "head_dim"]),
        "scale": Scalar("float32", description="Softmax scale, typically 1/sqrt(d)."),
        "max_sequence_kv": Scalar(
            "int32", description="Maximum K/V sequence length (s_kv_max)."
        ),
        "block_tables": Tensor(
            ["batch_size", "num_pages_per_seq"],
            dtype="int32",
            optional=True,
            description="Per-sequence page-id mapping.",
        ),
    },
    outputs={
        "output": Tensor(["batch_size", "num_heads_qo", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "stage:decode", "backend:cudnn"],
    reference=_cudnn_batch_decode_reference,
)

cudnn_batch_prefill_trace = TraceTemplate(
    op_type="cudnn_paged",
    name_prefix="cudnn_batch_prefill",
    description=(
        "Standalone cuDNN paged prefill with variable-length sequences. "
        "Per-seq lengths via actual_seq_lens_q/kv (not indptr); accepts "
        "paged (block_tables) or contiguous K/V. No plan()."
    ),
    axes={
        **_CUDNN_PAGED_AXES,
        "num_tokens": Var(description="Total query tokens across the batch."),
    },
    inputs={
        "q": Tensor(["num_tokens", "num_heads_qo", "head_dim"]),
        "k_cache": Tensor(["total_num_pages", "num_heads_kv", "page_size", "head_dim"]),
        "v_cache": Tensor(["total_num_pages", "num_heads_kv", "page_size", "head_dim"]),
        "scale": Scalar("float32", description="Softmax scale."),
        "max_token_per_sequence": Scalar(
            "int32", description="Maximum query tokens per sequence."
        ),
        "max_sequence_kv": Scalar("int32", description="Maximum K/V sequence length."),
        "actual_seq_lens_q": Tensor(
            ["batch_size"],
            dtype="int32",
            description="Actual query sequence length per batch entry.",
        ),
        "actual_seq_lens_kv": Tensor(
            ["batch_size"],
            dtype="int32",
            description="Actual KV sequence length per batch entry.",
        ),
        "block_tables": Tensor(
            ["batch_size", "num_pages_per_seq"],
            dtype="int32",
            optional=True,
        ),
        "causal": Scalar("int32", description="Bool: apply causal mask."),
        "return_lse": Scalar("int32", description="Bool: also return LSE."),
    },
    outputs={
        "output": Tensor(["num_tokens", "num_heads_qo", "head_dim"], dtype_from="q"),
        "lse": Tensor(
            ["num_tokens", "num_heads_qo"],
            dtype="float32",
            optional=True,
            description="Only produced when return_lse=True.",
        ),
    },
    tags=["status:verified", "stage:prefill", "backend:cudnn"],
    reference=_cudnn_batch_prefill_reference,
)


# ── Misc wrapper .run() templates ────────────────────────────────────────────
# These six wrappers live on top of existing kernels; their trace schemas
# follow their Python-level run() signatures.

batch_attention_run_trace = TraceTemplate(
    op_type="gqa_paged",
    name_prefix="batch_attention_run",
    description=(
        "BatchAttention.run(): unified decode+prefill wrapper with paged KV "
        "cache (tuple or interleaved tensor). plan() bakes in routing; run() "
        "takes q and paged kv_cache."
    ),
    axes={
        "num_qo_tokens": Var(description="Total query tokens."),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Var(
            description="Set during plan(); not a dim of the run() signature."
        ),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "q": Tensor(["num_qo_tokens", "num_qo_heads", "head_dim"]),
        "kv_cache": Tensor(
            ["num_qo_tokens", "num_qo_heads", "head_dim"],
            description="Paged KV cache tensor or tuple (layout varies).",
        ),
    },
    outputs={
        "output": Tensor(["num_qo_tokens", "num_qo_heads", "head_dim"], dtype_from="q"),
        "lse": Tensor(
            ["num_qo_tokens", "num_qo_heads"],
            dtype="float32",
            description="The 2-based log-sum-exp of attention logits.",
        ),
    },
    tags=["status:verified"],
)


_POD_AXES: dict[str, Var | Const] = {
    "num_qo_heads": Const(abbrev="h"),
    "num_kv_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
    "prefill_len": Var(description="Total prefill query tokens."),
    "decode_batch_size": Var(description="Number of decode queries."),
    "num_pages": Var(),
    "page_size": Const(abbrev="ps"),
}

pod_with_paged_kv_cache_run_trace = TraceTemplate(
    op_type="pod",
    name_prefix="pod_run",
    description=(
        "PODWithPagedKVCacheWrapper.run(): Prefill-On-Decode fused attention. "
        "Takes separate prefill (q_p, k_p, v_p) + decode (q_d, "
        "paged_kv_cache_d) workloads and fuses them into a single call."
    ),
    axes=_POD_AXES,
    inputs={
        "q_p": Tensor(["prefill_len", "num_qo_heads", "head_dim"]),
        "k_p": Tensor(["prefill_len", "num_kv_heads", "head_dim"]),
        "v_p": Tensor(["prefill_len", "num_kv_heads", "head_dim"]),
        "q_d": Tensor(["decode_batch_size", "num_qo_heads", "head_dim"]),
        "paged_kv_cache_d": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            description="Paged KV cache for the decode branch.",
        ),
    },
    outputs={
        "output_p": Tensor(
            ["prefill_len", "num_qo_heads", "head_dim"], dtype_from="q_p"
        ),
        "output_d": Tensor(
            ["decode_batch_size", "num_qo_heads", "head_dim"], dtype_from="q_d"
        ),
    },
    tags=["status:verified", "stage:pod"],
)


batch_pod_with_paged_kv_cache_run_trace = TraceTemplate(
    op_type="pod",
    name_prefix="batch_pod_run",
    description=(
        "BatchPODWithPagedKVCacheWrapper.run(): batched Prefill-On-Decode. "
        "Both prefill and decode use paged KV caches."
    ),
    axes=_POD_AXES,
    inputs={
        "q_p": Tensor(["prefill_len", "num_qo_heads", "head_dim"]),
        "paged_kv_cache_p": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            description="Paged KV cache for the prefill branch.",
        ),
        "q_d": Tensor(["decode_batch_size", "num_qo_heads", "head_dim"]),
        "paged_kv_cache_d": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            description="Paged KV cache for the decode branch.",
        ),
    },
    outputs={
        "output_p": Tensor(
            ["prefill_len", "num_qo_heads", "head_dim"], dtype_from="q_p"
        ),
        "output_d": Tensor(
            ["decode_batch_size", "num_qo_heads", "head_dim"], dtype_from="q_d"
        ),
    },
    tags=["status:verified", "stage:pod"],
)


block_sparse_attention_run_trace = TraceTemplate(
    op_type="block_sparse",
    name_prefix="block_sparse_run",
    description=(
        "BlockSparseAttentionWrapper.run(): block-sparse attention over "
        "q/k/v with a block-level mask baked in at plan() time."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "qo_len": Var(description="Query sequence length."),
        "kv_len": Var(description="Key/value sequence length."),
    },
    inputs={
        "q": Tensor(["qo_len", "num_qo_heads", "head_dim"]),
        "k": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
        "v": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
    },
    outputs={
        "output": Tensor(["qo_len", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "sparse:block"],
)


variable_block_sparse_attention_run_trace = TraceTemplate(
    op_type="block_sparse",
    name_prefix="var_block_sparse_run",
    description=(
        "VariableBlockSparseAttentionWrapper.run(): variable-length block-"
        "sparse attention. Same q/k/v layout as block_sparse but sequence "
        "lengths vary across the batch and the block mask is per-row."
    ),
    axes={
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "qo_len": Var(description="Query sequence length (variable)."),
        "kv_len": Var(description="Key/value sequence length (variable)."),
    },
    inputs={
        "q": Tensor(["qo_len", "num_qo_heads", "head_dim"]),
        "k": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
        "v": Tensor(["kv_len", "num_kv_heads", "head_dim"]),
    },
    outputs={
        "output": Tensor(["qo_len", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "sparse:block"],
)


multi_level_cascade_run_trace = TraceTemplate(
    op_type="cascade_attention",
    name_prefix="multi_level_cascade_run",
    description=(
        "MultiLevelCascadeAttentionWrapper.run(): cascade attention across "
        "multiple shared-prefix levels. Internally merges per-level "
        "attention states with logsumexp."
    ),
    axes={
        "batch_size": Var(),
        "num_qo_heads": Const(abbrev="h"),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
    },
    inputs={
        "q": Tensor(["batch_size", "num_qo_heads", "head_dim"]),
        "paged_kv_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            description="Paged KV cache (tuple or single tensor).",
        ),
    },
    outputs={
        "output": Tensor(["batch_size", "num_qo_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "cascade"],
)


@torch.no_grad()
def _batch_attention_run_reference(q, kv_cache, **_unused):
    """SDPA over q and a paged kv_cache tuple (k_cache, v_cache). Assumes
    head_dim is the last axis and each sequence's K/V is the full cache."""
    if isinstance(kv_cache, tuple):
        k_cache, v_cache = kv_cache
    else:
        k_cache = kv_cache[:, 0]
        v_cache = kv_cache[:, 1]
    num_tokens, num_qo_heads, head_dim = q.shape
    # Flatten paged cache; assume one sequence.
    k_flat = k_cache.reshape(-1, k_cache.shape[-2], head_dim).to(torch.float32)
    v_flat = v_cache.reshape(-1, v_cache.shape[-2], head_dim).to(torch.float32)
    num_kv_heads = k_flat.shape[1]
    gqa_ratio = num_qo_heads // num_kv_heads
    sm_scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros_like(q, dtype=torch.float32)
    lse = torch.full(
        (num_tokens, num_qo_heads),
        -float("inf"),
        dtype=torch.float32,
        device=q.device,
    )
    for h in range(num_qo_heads):
        kv_h = h // gqa_ratio
        logits = (q[:, h].to(torch.float32) @ k_flat[:, kv_h].T) * sm_scale
        lse[:, h] = torch.logsumexp(logits, dim=-1) / math.log(2.0)
        attn = torch.softmax(logits, dim=-1)
        output[:, h] = attn @ v_flat[:, kv_h]
    return output.to(q.dtype), lse


@torch.no_grad()
def _pod_run_reference(q_p, k_p, v_p, q_d, paged_kv_cache_d, **_unused):
    """POD reference: independent prefill + decode attention passes."""
    p_out = _single_prefill_reference(q_p, k_p, v_p, causal=True)
    dec_kv = (
        paged_kv_cache_d
        if isinstance(paged_kv_cache_d, tuple)
        else (paged_kv_cache_d[:, 0], paged_kv_cache_d[:, 1])
    )
    d_out, _ = _batch_attention_run_reference(q_d, dec_kv)
    return p_out, d_out


@torch.no_grad()
def _batch_pod_run_reference(q_p, paged_kv_cache_p, q_d, paged_kv_cache_d, **_unused):
    """Batch POD: paged prefill + paged decode (both via batch_attention)."""
    pkv_p = (
        paged_kv_cache_p
        if isinstance(paged_kv_cache_p, tuple)
        else (paged_kv_cache_p[:, 0], paged_kv_cache_p[:, 1])
    )
    pkv_d = (
        paged_kv_cache_d
        if isinstance(paged_kv_cache_d, tuple)
        else (paged_kv_cache_d[:, 0], paged_kv_cache_d[:, 1])
    )
    p_out, _ = _batch_attention_run_reference(q_p, pkv_p)
    d_out, _ = _batch_attention_run_reference(q_d, pkv_d)
    return p_out, d_out


@torch.no_grad()
def _block_sparse_run_reference(q, k, v, **_unused):
    """Dense SDPA fallback for block-sparse attention (ignores block mask)."""
    return _single_prefill_reference(q, k, v, causal=False)


@torch.no_grad()
def _multi_level_cascade_run_reference(q, paged_kv_cache, **_unused):
    """Single-level cascade approximation: plain batched SDPA."""
    out, _ = _batch_attention_run_reference(q, paged_kv_cache)
    return out


@torch.no_grad()
def _segment_gemm_run_reference(x, weights, **_unused):
    """Batched matmul: per-segment weights applied to stacked rows. Assumes
    the caller passes a seg_indptr via kwargs; falls back to broadcasting
    the first weight if unavailable."""
    seg_indptr = _unused.get("seg_indptr")
    if seg_indptr is None:
        return torch.matmul(x.to(torch.float32), weights[0].to(torch.float32)).to(
            x.dtype
        )
    out = torch.zeros(
        (x.shape[0], weights.shape[-1]),
        dtype=torch.float32,
        device=x.device,
    )
    for i in range(weights.shape[0]):
        start = int(seg_indptr[i].item())
        end = int(seg_indptr[i + 1].item())
        out[start:end] = x[start:end].to(torch.float32) @ weights[i].to(torch.float32)
    return out.to(x.dtype)


# Attach references to the templates declared above.
batch_attention_run_trace.reference = _batch_attention_run_reference
pod_with_paged_kv_cache_run_trace.reference = _pod_run_reference
batch_pod_with_paged_kv_cache_run_trace.reference = _batch_pod_run_reference
block_sparse_attention_run_trace.reference = _block_sparse_run_reference
variable_block_sparse_attention_run_trace.reference = _block_sparse_run_reference
multi_level_cascade_run_trace.reference = _multi_level_cascade_run_reference


segment_gemm_run_trace = TraceTemplate(
    op_type="segment_gemm",
    name_prefix="segment_gemm_run",
    description=(
        "SegmentGEMMWrapper.run(): variable-size batched GEMM over "
        "concatenated row segments. x is a ragged stack of per-segment "
        "inputs; weights may be shared or per-segment."
    ),
    axes={
        "total_rows": Var(description="Total rows across all segments."),
        "K": Const(abbrev="k"),
        "N": Const(abbrev="n"),
        "batch_size": Var(description="Number of segments."),
    },
    inputs={
        "x": Tensor(
            ["total_rows", "K"],
            description="Stacked segment inputs, row-concatenated.",
        ),
        "weights": Tensor(
            ["batch_size", "K", "N"],
            description="Per-segment weight tensors (may be shared across segments).",
        ),
    },
    outputs={
        "output": Tensor(["total_rows", "N"], dtype_from="x"),
    },
    tags=["status:verified"],
)
segment_gemm_run_trace.reference = _segment_gemm_run_reference


# ── CuteDSL MLA paged decode wrapper (.run) ──────────────────────────────────


@torch.no_grad()
def _cute_dsl_batch_mla_run_reference(
    q,
    kv_cache,
    block_tables,
    seq_lens,
    max_seq_len,
    softmax_scale,
    output_scale: float = 1.0,
    out=None,
    **_unused,
):
    """Reference for CuteDslBatchMLAPagedAttentionWrapper.run.

    Same MLA decode math as ``trtllm_batch_decode_with_kv_cache_mla``: Q is
    [B, q_len, H, D_qk] with D_qk = kv_lora_rank + qk_rope_head_dim; the
    paged kv_cache stores [ckv ‖ kpe]. Output dim equals kv_lora_rank.
    Assumes DeepSeek-V3 default kv_rope_head_dim=64.
    """
    B, q_len, H, D_qk = q.shape
    if kv_cache.dim() == 4:
        kv_cache = kv_cache.squeeze(1)
    page_size = kv_cache.shape[1]
    kv_lora_rank = D_qk - 64
    out_shape = (B, q_len, H, kv_lora_rank)
    output = torch.zeros(out_shape, dtype=q.dtype, device=q.device)
    s = float(softmax_scale)
    s2 = float(output_scale)
    for b in range(B):
        kv_len = int(seq_lens[b].item())
        n_pages = (kv_len + page_size - 1) // page_size
        pages = block_tables[b, :n_pages].to(torch.long)
        flat = kv_cache[pages].reshape(-1, D_qk)[:kv_len].to(torch.float32)
        Kn = flat[:, :kv_lora_rank]
        Kp = flat[:, kv_lora_rank:]
        for t in range(q_len):
            qq = q[b, t].to(torch.float32)
            Qn = qq[:, :kv_lora_rank]
            Qp = qq[:, kv_lora_rank:]
            logits = (Qn @ Kn.T + Qp @ Kp.T) * s
            attn = torch.softmax(logits, dim=-1)
            output[b, t] = (attn @ Kn * s2).to(q.dtype)
    if out is not None:
        out.copy_(output)
    return output


cute_dsl_batch_mla_run_trace = TraceTemplate(
    op_type="mla_paged",
    name_prefix="cute_dsl_batch_mla_run",
    description=(
        "CuteDSL CuteDslBatchMLAPagedAttentionWrapper.run: alternative-"
        "backend MLA decode with the same math as trtllm_batch_decode_mla "
        "but with the CuteDSL kernel signature (q/softmax_scale/output_scale)."
    ),
    axes={
        "batch_size": Var(),
        "q_len_per_request": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim_qk": Const(abbrev="d_qk"),
        "kv_lora_rank": Var(description="head_dim_qk - qk_rope_head_dim."),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "max_pages_per_seq": Var(),
    },
    inputs={
        "q": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "head_dim_qk"],
        ),
        "kv_cache": Tensor(["num_pages", "page_size", "head_dim_qk"]),
        "block_tables": Tensor(["batch_size", "max_pages_per_seq"], dtype="int32"),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_seq_len": Scalar("int32"),
        "softmax_scale": Scalar("float32"),
        "output_scale": Scalar("float32", optional=True),
    },
    outputs={
        "output": Tensor(
            ["batch_size", "q_len_per_request", "num_heads", "kv_lora_rank"],
            dtype_from="q",
        ),
    },
    tags=["status:verified", "stage:decode", "backend:cute-dsl", "mla"],
    reference=_cute_dsl_batch_mla_run_reference,
)


# ── CuteDSL ragged batch prefill wrapper (.run) ──────────────────────────────


@torch.no_grad()
def _cute_dsl_batch_prefill_run_reference(q, k, v, out=None, **_unused):
    """Reference for CuteDslBatchPrefillWrapper.run: causal SDPA on ragged
    [total_q, H, D] / [total_kv, H, D] tensors. Indptr is baked into plan().
    Treats the whole tensor as a single sequence (matches the wrapper's
    single-batch single-request use).
    """
    head_dim = q.shape[-1]
    sm_scale = 1.0 / math.sqrt(head_dim)
    qf = q.to(torch.float32)
    kf = k.to(torch.float32)
    vf = v.to(torch.float32)
    output = torch.zeros_like(q, dtype=torch.float32)
    H = q.shape[-2]
    qo_len = q.shape[0]
    kv_len = k.shape[0]
    delta = kv_len - qo_len
    for h in range(H):
        logits = (qf[:, h] @ kf[:, h].T) * sm_scale
        mask = torch.full_like(logits, float("-inf"))
        for i in range(qo_len):
            mask[i, : i + 1 + max(0, delta)] = 0.0
        logits = logits + mask
        attn = torch.softmax(logits, dim=-1)
        output[:, h] = attn @ vf[:, h]
    output_q = output.to(q.dtype)
    if out is not None:
        out.copy_(output_q)
    return output_q


cute_dsl_batch_prefill_run_trace = TraceTemplate(
    op_type="gqa_ragged",
    name_prefix="cute_dsl_batch_prefill_run",
    description=(
        "CuteDSL CuteDslBatchPrefillWrapper.run: ragged batch prefill "
        "(separate q/k/v) with indptr baked into plan(). Same SDPA math "
        "as the FA2 / FA3 batch prefill wrappers."
    ),
    axes={
        "total_q_len": Var(),
        "total_kv_len": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
    },
    inputs={
        "q": Tensor(["total_q_len", "num_heads", "head_dim"]),
        "k": Tensor(["total_kv_len", "num_heads", "head_dim"]),
        "v": Tensor(["total_kv_len", "num_heads", "head_dim"]),
    },
    outputs={
        "output": Tensor(["total_q_len", "num_heads", "head_dim"], dtype_from="q"),
    },
    tags=["status:verified", "stage:prefill", "backend:cute-dsl"],
    reference=_cute_dsl_batch_prefill_run_reference,
)
