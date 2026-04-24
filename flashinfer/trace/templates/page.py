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

"""TraceTemplates for paged-KV cache append operations."""

import math

import torch

from ..template import Const, Scalar, Tensor, TraceTemplate, Var


@torch.no_grad()
def _append_paged_kv_cache_reference(
    append_key,
    append_value,
    batch_indices,
    positions,
    paged_kv_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    kv_layout="NHD",
    **_unused,
):
    """Append (append_key, append_value) into the paged KV cache at the
    specified (batch_indices, positions) offsets.

    Mutates ``paged_kv_cache`` in place. Accepts both tuple ``(k, v)`` and
    single-tensor interleaved layouts. Only the NHD layout is modelled here;
    HND is a permutation of the same data.
    """
    if isinstance(paged_kv_cache, tuple):
        k_cache, v_cache = paged_kv_cache
    else:
        # Single tensor: [num_pages, 2, page_size, num_kv_heads, head_dim] in NHD
        k_cache = paged_kv_cache[:, 0]
        v_cache = paged_kv_cache[:, 1]
    N = int(batch_indices.shape[0])
    page_size = k_cache.shape[1] if kv_layout == "NHD" else k_cache.shape[2]
    for i in range(N):
        b = int(batch_indices[i].item())
        pos = int(positions[i].item())
        page_offset = pos // page_size
        in_page_offset = pos % page_size
        # kv_indices maps to the global page id for this (batch, page_offset).
        idx_base = int(kv_indptr[b].item())
        page_id = int(kv_indices[idx_base + page_offset].item())
        if kv_layout == "NHD":
            k_cache[page_id, in_page_offset] = append_key[i]
            v_cache[page_id, in_page_offset] = append_value[i]
        else:  # HND
            k_cache[page_id, :, in_page_offset] = append_key[i]
            v_cache[page_id, :, in_page_offset] = append_value[i]
    return paged_kv_cache


append_paged_kv_cache_trace = TraceTemplate(
    op_type="page_append",
    name_prefix="append_paged_kv_cache",
    description=(
        "Append a batch of (key, value) rows into a paged KV cache at "
        "positions determined by (batch_indices, positions) and the per-seq "
        "kv_indptr/kv_indices/kv_last_page_len layout."
    ),
    axes={
        "nnz_kv": Var(description="Total K/V tokens to append."),
        "num_kv_heads": Const(abbrev="kv"),
        "head_dim": Const(abbrev="d"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "batch_size": Var(),
        "batch_size_plus_1": Var(description="batch_size + 1."),
        "num_kv_indices": Var(description="Flat length of kv_indices."),
    },
    inputs={
        "append_key": Tensor(["nnz_kv", "num_kv_heads", "head_dim"]),
        "append_value": Tensor(["nnz_kv", "num_kv_heads", "head_dim"]),
        "batch_indices": Tensor(
            ["nnz_kv"],
            dtype="int32",
            description="Per-token batch index.",
        ),
        "positions": Tensor(
            ["nnz_kv"],
            dtype="int32",
            description="Per-token absolute position.",
        ),
        "paged_kv_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            description="Paged KV cache (tuple or single tensor).",
        ),
        "kv_indices": Tensor(["num_kv_indices"], dtype="int32"),
        "kv_indptr": Tensor(["batch_size_plus_1"], dtype="int32"),
        "kv_last_page_len": Tensor(["batch_size"], dtype="int32"),
    },
    outputs={
        "paged_kv_cache": Tensor(
            ["num_pages", "page_size", "num_kv_heads", "head_dim"],
            dtype_from="append_key",
            description="Updated paged KV cache (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
    reference=_append_paged_kv_cache_reference,
)


@torch.no_grad()
def _append_paged_mla_kv_cache_reference(
    append_ckv,
    append_kpe,
    batch_indices,
    positions,
    ckv_cache,
    kpe_cache,
    kv_indices,
    kv_indptr,
    kv_last_page_len,
    **_unused,
):
    """Append (append_ckv, append_kpe) into the MLA paged KV cache."""
    if ckv_cache is None or kpe_cache is None:
        return ckv_cache, kpe_cache
    N = int(batch_indices.shape[0])
    page_size = ckv_cache.shape[1]
    for i in range(N):
        b = int(batch_indices[i].item())
        pos = int(positions[i].item())
        page_offset = pos // page_size
        in_page_offset = pos % page_size
        idx_base = int(kv_indptr[b].item())
        page_id = int(kv_indices[idx_base + page_offset].item())
        ckv_cache[page_id, in_page_offset] = append_ckv[i]
        kpe_cache[page_id, in_page_offset] = append_kpe[i]
    return ckv_cache, kpe_cache


append_paged_mla_kv_cache_trace = TraceTemplate(
    op_type="page_append",
    name_prefix="append_paged_mla_kv_cache",
    description=(
        "Append MLA (ckv, kpe) rows into an MLA paged KV cache. Same "
        "indexing scheme as append_paged_kv_cache but with the MLA latent "
        "split (ckv ~ head_dim_ckv=512, kpe ~ head_dim_kpe=64)."
    ),
    axes={
        "nnz_kv": Var(description="Total K/V tokens to append."),
        "head_dim_ckv": Const(abbrev="ckv"),
        "head_dim_kpe": Const(abbrev="kpe"),
        "num_pages": Var(),
        # page_size is Var because ckv_cache / kpe_cache are optional.
        "page_size": Var(description="Size of each page (from optional cache)."),
        "batch_size": Var(),
        "batch_size_plus_1": Var(description="batch_size + 1."),
        "num_kv_indices": Var(),
    },
    inputs={
        "append_ckv": Tensor(["nnz_kv", "head_dim_ckv"]),
        "append_kpe": Tensor(["nnz_kv", "head_dim_kpe"]),
        "batch_indices": Tensor(["nnz_kv"], dtype="int32"),
        "positions": Tensor(["nnz_kv"], dtype="int32"),
        "ckv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_ckv"],
            optional=True,
        ),
        "kpe_cache": Tensor(
            ["num_pages", "page_size", "head_dim_kpe"],
            optional=True,
        ),
        "kv_indices": Tensor(["num_kv_indices"], dtype="int32"),
        "kv_indptr": Tensor(["batch_size_plus_1"], dtype="int32"),
        "kv_last_page_len": Tensor(["batch_size"], dtype="int32"),
    },
    outputs={
        "ckv_cache": Tensor(
            ["num_pages", "page_size", "head_dim_ckv"],
            dtype_from="append_ckv",
            description="Updated compressed KV cache (in-place).",
        ),
        "kpe_cache": Tensor(
            ["num_pages", "page_size", "head_dim_kpe"],
            dtype_from="append_kpe",
            description="Updated KPE cache (in-place).",
        ),
    },
    constraints=["batch_size_plus_1 == batch_size + 1"],
    tags=["status:verified"],
    reference=_append_paged_mla_kv_cache_reference,
)


# ── XQA attention (paged KV + block-tables) ──────────────────────────────────

_XQA_AXES: dict[str, Var | Const] = {
    "num_tokens": Var(),
    "num_heads_qo": Const(abbrev="h"),
    "num_kv_heads": Const(abbrev="kv"),
    "head_dim": Const(abbrev="d"),
    "num_pages": Var(),
    "page_size": Const(abbrev="ps"),
    "batch_size": Var(),
    "max_pages_per_seq": Var(),
}


@torch.no_grad()
def _xqa_reference(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    output=None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    **_unused,
):
    """Reference XQA decode: page-gather + SDPA per batch item. kv_layout=NHD.

    The regular XQA kernel applies ``q_scale * kv_scale * rsqrtf(head_dim)``
    to the QK product internally (see csrc/xqa/mha.cu:1765 and
    mha_sm90.cu:781), so this reference mirrors the same scaling to stay in
    sync. Note that XQA MLA uses a different convention (no rsqrt) — see
    ``_xqa_mla_reference``.
    """
    _, num_heads_qo, head_dim = (
        q.shape if q.dim() == 3 else q.reshape(-1, q.shape[-2], q.shape[-1]).shape
    )
    q_flat = q.reshape(-1, num_heads_qo, head_dim)
    num_kv_heads = k_cache.shape[-2]
    gqa_ratio = num_heads_qo // num_kv_heads
    batch_size = page_table.shape[0]
    page_size = k_cache.shape[1]
    qk_scale = float(q_scale) * float(kv_scale) / math.sqrt(head_dim)
    out = torch.zeros_like(q_flat, dtype=torch.float32)
    for b in range(batch_size):
        kv_len = int(seq_lens[b].item())
        n_pages_used = (kv_len + page_size - 1) // page_size
        pages = page_table[b, :n_pages_used].to(torch.long)
        k_b = k_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        v_b = v_cache[pages].reshape(-1, num_kv_heads, head_dim)[:kv_len]
        for h in range(num_heads_qo):
            kv_h = h // gqa_ratio
            logits = (
                q_flat[b, h].to(torch.float32) @ k_b[:, kv_h].to(torch.float32).T
            ) * qk_scale
            attn = torch.softmax(logits, dim=-1)
            out[b, h] = attn @ v_b[:, kv_h].to(torch.float32)
    result = out.reshape(*q.shape).to(q.dtype)
    if output is not None:
        output.copy_(result)
    return result


@torch.no_grad()
def _xqa_mla_reference(
    q,
    k_cache,
    v_cache,
    page_table,
    seq_lens,
    output=None,
    output_dtype=None,
    q_scale: float = 1.0,
    kv_scale: float = 1.0,
    **_unused,
):
    """Reference XQA MLA decode: page-gather + SDPA.

    Unlike the regular XQA kernel (which applies ``rsqrtf(head_dim)`` to the
    QK product internally), the MLA kernel leaves that scaling to the
    caller: it computes ``softmax(Q @ K^T * q_scale * kv_scale) @ V`` with
    no implicit ``1/sqrt(head_dim)`` factor (see csrc/xqa/mla_sm120.cu:456
    — ``qkScaleLog2e = qScale * kvScale * log2e``).

    The V read comes from the ``v_cache`` tensor (separate from ``k_cache``);
    only the first ``v_head_dim`` columns are consumed. This matches the
    kernel's behaviour whether V is stored in a dedicated buffer or aliased
    on top of the K latent.

    Output shape: ``[..., num_heads_qo, v_head_dim]``.
    """
    head_dim_qk = q.shape[-1]
    v_head_dim = v_cache.shape[-1]
    batch_size = page_table.shape[0]
    page_size = k_cache.shape[1]
    num_heads_qo = q.shape[-2] if q.dim() >= 3 else 1
    q_flat = q.reshape(-1, num_heads_qo, head_dim_qk)
    qk_scale = float(q_scale) * float(kv_scale)
    out_shape = q.shape[:-1] + (v_head_dim,)
    out = torch.zeros(
        (q_flat.shape[0], num_heads_qo, v_head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    for b in range(batch_size):
        kv_len = int(seq_lens[b].item())
        n_pages_used = (kv_len + page_size - 1) // page_size
        pages = page_table[b, :n_pages_used].to(torch.long)
        k_b = k_cache[pages].reshape(-1, head_dim_qk)[:kv_len].to(torch.float32)
        v_b = v_cache[pages].reshape(-1, v_cache.shape[-1])[:kv_len].to(torch.float32)
        v_b = v_b[:, :v_head_dim]
        for h in range(num_heads_qo):
            logits = q_flat[b, h].to(torch.float32) @ k_b.T * qk_scale
            attn = torch.softmax(logits, dim=-1)
            out[b, h] = attn @ v_b
    dtype = output_dtype or q.dtype
    result = out.reshape(out_shape).to(dtype)
    if output is not None:
        output.copy_(result)
    return result


xqa_trace = TraceTemplate(
    op_type="xqa",
    name_prefix="xqa",
    description=(
        "XQA (Cross-Query Attention) paged decode kernel. Fast decode path "
        "with separate k/v caches and rectangular page_table[batch_size, "
        "num_pages_per_seq]."
    ),
    axes=_XQA_AXES,
    inputs={
        "q": Tensor(["num_tokens", "num_heads_qo", "head_dim"]),
        "k_cache": Tensor(["num_pages", "num_kv_heads", "page_size", "head_dim"]),
        "v_cache": Tensor(["num_pages", "num_kv_heads", "page_size", "head_dim"]),
        "page_table": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
        ),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "num_heads_qo", "head_dim"],
            dtype_from="q",
        ),
    },
    tags=["status:verified", "backend:xqa"],
    reference=_xqa_reference,
)


xqa_mla_trace = TraceTemplate(
    op_type="xqa",
    name_prefix="xqa_mla",
    description=(
        "XQA MLA decode: MLA (ckv + kpe) latent split applied to the XQA "
        "paged decode path."
    ),
    axes={
        "num_tokens": Var(),
        "num_heads_qo": Const(abbrev="h"),
        "head_dim_ckv": Const(abbrev="ckv"),
        "head_dim_kpe": Const(abbrev="kpe"),
        "num_pages": Var(),
        "page_size": Const(abbrev="ps"),
        "batch_size": Var(),
        "max_pages_per_seq": Var(),
    },
    inputs={
        "q": Tensor(["num_tokens", "num_heads_qo", "head_dim_ckv"]),
        "k_cache": Tensor(["num_pages", "page_size", "head_dim_ckv"]),
        "v_cache": Tensor(["num_pages", "page_size", "head_dim_kpe"]),
        "page_table": Tensor(
            ["batch_size", "max_pages_per_seq"],
            dtype="int32",
        ),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "num_heads_qo", "head_dim_ckv"],
            dtype_from="q",
        ),
    },
    tags=["status:verified", "backend:xqa", "mla"],
    reference=_xqa_mla_reference,
)


# ── TRTLLM FMHA v2 prefill ──────────────────────────────────────────────────


@torch.no_grad()
def _trtllm_fmha_v2_prefill_reference(
    qkv,
    seq_lens,
    max_q_len,
    max_kv_len,
    bmm1_scale,
    bmm2_scale,
    batch_size,
    cum_seq_lens_q,
    cum_seq_lens_kv,
    **_unused,
):
    """Reference for TRT-LLM FMHA v2 prefill.

    Assumes qkv is either a single fused tensor [total_tokens, 3, H, D]
    or a tuple (q, k, v). Treats the workload as causal SDPA per batch.
    """
    if isinstance(qkv, tuple):
        q, k, v = qkv[0], qkv[1], qkv[2] if len(qkv) == 3 else qkv[1]
    elif qkv.dim() == 4 and qkv.shape[1] == 3:
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
    else:
        q = qkv
        k = qkv
        v = qkv
    out = torch.zeros_like(q, dtype=torch.float32)
    num_heads = q.shape[-2]
    for b in range(int(batch_size)):
        q_start = int(cum_seq_lens_q[b].item())
        q_end = int(cum_seq_lens_q[b + 1].item())
        kv_start = int(cum_seq_lens_kv[b].item())
        kv_end = int(cum_seq_lens_kv[b + 1].item())
        q_b = q[q_start:q_end].to(torch.float32)
        k_b = k[kv_start:kv_end].to(torch.float32)
        v_b = v[kv_start:kv_end].to(torch.float32)
        qi = q_end - q_start
        kv_len = kv_end - kv_start
        delta = kv_len - qi
        for h in range(num_heads):
            logits = (q_b[:, h] @ k_b[:, h].T) * float(bmm1_scale)
            mask = torch.full_like(logits, float("-inf"))
            for i in range(qi):
                mask[i, : i + 1 + max(0, delta)] = 0.0
            logits = logits + mask
            attn = torch.softmax(logits, dim=-1)
            out[q_start:q_end, h] = (attn @ v_b[:, h]) * float(bmm2_scale)
    return out.to(q.dtype)


@torch.no_grad()
def _tgv_gemm_sm100_reference(a, b, bias, **_unused):
    """TGV GEMM: C = A @ B + bias."""
    return (a.to(torch.float32) @ b.to(torch.float32) + bias.to(torch.float32)).to(
        a.dtype
    )


# ── TRTLLM FMHA v2 prefill (original) ──────────────────────────────────────

trtllm_fmha_v2_prefill_trace = TraceTemplate(
    op_type="trtllm_paged",
    name_prefix="trtllm_fmha_v2_prefill",
    description=(
        "TRT-LLM FMHA v2 prefill. Accepts fused qkv or separate (q, kv), "
        "variable-length sequences with cum_seq_lens_q/kv."
    ),
    axes={
        "num_tokens": Var(),
        "num_heads": Const(abbrev="h"),
        "head_dim": Const(abbrev="d"),
        "batch_size": Var(),
        "batch_size_plus_1_q": Var(description="batch_size + 1 for cum_seq_lens_q."),
        "batch_size_plus_1_kv": Var(description="batch_size + 1 for cum_seq_lens_kv."),
    },
    inputs={
        "qkv": Tensor(
            ["num_tokens", "num_heads", "head_dim"],
            description="Fused qkv or q tensor (layout determined by input_layout).",
        ),
        "seq_lens": Tensor(["batch_size"], dtype="int32"),
        "max_q_len": Scalar("int32"),
        "max_kv_len": Scalar("int32"),
        "bmm1_scale": Scalar("float32"),
        "bmm2_scale": Scalar("float32"),
        "batch_size_scalar": Scalar("int32", param="batch_size"),
        "cum_seq_lens_q": Tensor(["batch_size_plus_1_q"], dtype="int32"),
        "cum_seq_lens_kv": Tensor(["batch_size_plus_1_kv"], dtype="int32"),
    },
    outputs={
        "output": Tensor(
            ["num_tokens", "num_heads", "head_dim"],
            dtype_from="qkv",
        ),
    },
    tags=["status:verified", "stage:prefill", "backend:trtllm"],
    reference=_trtllm_fmha_v2_prefill_reference,
)


# ── TGV GEMM SM100 ──────────────────────────────────────────────────────────

tgv_gemm_sm100_trace = TraceTemplate(
    op_type="gemm_bf16",
    name_prefix="tgv_gemm_sm100",
    description=(
        "TGV GEMM on SM100: C = A @ B + bias. Automatic dtype detection "
        "(bf16/fp16). Intended for the TRT-LLM TGV backend."
    ),
    axes={
        "M": Var(),
        "N": Const(),
        "K": Const(),
    },
    inputs={
        "a": Tensor(["M", "K"]),
        "b": Tensor(
            ["K", "N"],
            description="Weight matrix in column-major layout.",
        ),
        "bias": Tensor(["N"], description="Bias tensor."),
    },
    outputs={
        "output": Tensor(["M", "N"], dtype_from="a"),
    },
    tags=["status:verified", "backend:tgv"],
    reference=_tgv_gemm_sm100_reference,
)
