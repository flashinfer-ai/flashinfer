"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

---

Minimax Sparse Attention decode path for SM120/SM121.

A decode step is a varlen batch with a short, uniform ``seqlen_q`` per
request. Each valid (kv-head, token, t) entry of ``q2k_indices`` becomes one
work item of the KV-major forward kernel: one token (x its GQA query heads)
attending one selected KV block, written to split slot ``t``. Because
:func:`msa_topk_select` tail-pads invalid entries with -1, the slot is
simply the list position and the split count is the valid prefix length,
so the whole schedule is built with a handful of torch ops and no CUDA
scheduler kernels.
"""

from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.msa import msa_sparse_decode_attention_trace
from .sparse_index_utils import MsaAttentionSchedule


@flashinfer_api
def msa_build_decode_schedule(
    q2k_indices: torch.Tensor,  # (Hkv, total_q, topk) int32, -1 tail-padded
    cu_seqlens_q: torch.Tensor,  # (B + 1,) int32, device
    cu_seqlens_k: torch.Tensor,  # (B + 1,) int32, device
) -> MsaAttentionSchedule:
    """Build a KV-major schedule for decode: one work item per valid
    (kv-head, token, selected-block) entry.

    ``q2k_indices`` must be ascending with ``-1`` entries tail-padded (as
    produced by :func:`msa_topk_select`); the split slot of an entry is
    its position ``t`` in the list.
    """
    num_kv_heads, total_q, topk = q2k_indices.shape
    dev = q2k_indices.device

    q_global = torch.arange(total_q, device=dev, dtype=torch.int32)
    # batch of each token and its batch-local index
    batch_of_q = (
        torch.bucketize(q_global.long(), cu_seqlens_q.long(), right=True) - 1
    ).to(torch.int32)
    q_local = q_global - cu_seqlens_q[batch_of_q.long()]

    valid = q2k_indices >= 0  # (Hkv, total_q, topk)
    split_counts = valid.sum(dim=2).t().contiguous().to(torch.int32)  # (total_q, Hkv)

    # qsplit: q_local | (slot << 24), laid out flat at qi * topk + t
    slots = torch.arange(topk, device=dev, dtype=torch.int32)
    qsplit = (q_local.view(1, total_q, 1) | (slots.view(1, 1, topk) << 24)).expand(
        num_kv_heads, total_q, topk
    )
    qsplit_indices = qsplit.reshape(num_kv_heads, total_q * topk).contiguous()

    # identity row_ptr: work item row r covers qsplit entries [r, r + 1)
    total_rows = total_q * topk
    row_ptr = (
        torch.arange(total_rows + 1, device=dev, dtype=torch.int32)
        .view(1, -1)
        .expand(num_kv_heads, -1)
        .contiguous()
    )

    # Static dense work list: one item per (h, token, t) entry, valid or
    # not; q_count = 0 marks padding (the kernel exits immediately). This
    # keeps the build free of device syncs (no nonzero) and makes the whole
    # decode step CUDA-graph capturable.
    n_items = num_kv_heads * total_q * topk
    h_idx = (
        torch.arange(num_kv_heads, device=dev, dtype=torch.int32)
        .view(-1, 1, 1)
        .expand(num_kv_heads, total_q, topk)
    )
    rows = (
        torch.arange(total_q * topk, device=dev, dtype=torch.int32)
        .view(1, total_q, topk)
        .expand(num_kv_heads, -1, -1)
    )
    batches = batch_of_q.view(1, total_q, 1).expand(num_kv_heads, -1, topk)
    meta = torch.stack(
        [
            h_idx.reshape(-1),
            rows.reshape(-1),
            torch.zeros(n_items, device=dev, dtype=torch.int32),
            valid.reshape(-1).to(torch.int32),  # q_count: 1 valid, 0 padding
            batches.reshape(-1),
            q2k_indices.reshape(-1).clamp(min=0),
        ],
        dim=1,
    ).contiguous()
    work_count = torch.full((1,), n_items, dtype=torch.int32, device=dev)

    return MsaAttentionSchedule(
        row_ptr=row_ptr,
        q_indices=qsplit_indices,  # unused by the kernel; kept for shape parity
        qsplit_indices=qsplit_indices,
        split_counts=split_counts,
        scheduler_metadata=meta,
        work_count=work_count,
        work_capacity=meta.shape[0],
        total_rows=total_rows,
        max_kv_blocks=0,  # not consumed on the decode path; avoids a D2H sync
        topk=topk,
    )


@flashinfer_api(trace=msa_sparse_decode_attention_trace)
def msa_sparse_decode_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqlen_q: int = 1,
    causal: bool = True,
    softmax_scale: Optional[float] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    partial_dtype: Optional[torch.dtype] = None,
):
    """Sparse decode attention for SM120/SM121.

    Computes attention for a decode step: each request contributes
    ``seqlen_q`` query tokens (uniform across the batch) attending only the
    KV blocks selected in ``q2k_indices``. Decode tokens are right-aligned:
    token ``i`` of a request sits at position ``seqlen_k - seqlen_q + i``.

    Parameters
    ----------
    q : torch.Tensor
        ``(batch_size * seqlen_q, num_qo_heads, 128)``, bf16 or fp16.
    k, v : torch.Tensor
        Paged: ``(num_pages, num_kv_heads, 128, 128)`` with ``page_table``
        and ``seqused_k``; or flat varlen ``(total_k, num_kv_heads, 128)``
        with ``cu_seqlens_k``. May be fp8 E4M3 (upconverted in-kernel).
    q2k_indices : torch.Tensor
        ``(num_kv_heads, batch_size * seqlen_q, topk)`` int32, ascending,
        ``-1`` tail-padded (the format produced by
        :func:`msa_topk_select`).
    seqlen_q : int
        Uniform query length per request (e.g. 1, or >1 for speculative
        decoding).
    causal : bool
        Right-aligned causal masking (default True for decode).

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        ``(batch_size * seqlen_q, num_qo_heads, 128)`` in q's dtype; plus
        the natural-log LSE if ``return_softmax_lse``.
    """
    import cutlass
    import cutlass.cute as cute

    from ..utils import is_sm12x_supported
    from .cute_dsl.sparse_decode_sm12x import SparseDecodeForwardSm12x
    from .sparse_attention import (
        _combine_partials,
        _compile_cache,
        _cutlass_dtype,
        _fake,
        _q_offset_tensor,
    )

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "msa_sparse_decode_attention requires SM120 or SM121 and CUDA >= 12.8"
        )
    total_q, num_qo_heads, head_dim = q.shape
    if total_q % seqlen_q != 0:
        raise ValueError(
            f"q rows ({total_q}) must be batch_size * seqlen_q ({seqlen_q})"
        )
    q_fp8 = q.dtype == torch.float8_e4m3fn
    compute_dtype = torch.bfloat16 if q_fp8 else q.dtype
    if not q_fp8 and q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16/fp16/fp8_e4m3, got {q.dtype}")
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    batch_size = total_q // seqlen_q
    num_kv_heads = k.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    group_size = num_qo_heads // num_kv_heads
    if group_size > 16:
        raise ValueError(f"GQA group size {group_size} must be <= 16 for decode")
    if q2k_indices.dtype != torch.int32 or q2k_indices.shape[:2] != (
        num_kv_heads,
        total_q,
    ):
        raise ValueError("q2k_indices must be int32 (num_kv_heads, total_q, topk)")
    # Compiled for a compact layout; reject a strided q2k (e.g. a bare permute of
    # msa_topk_select's output) here, matching msa_sparse_attention's guard.
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    topk = q2k_indices.shape[2]
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    kv_nvfp4 = k.dtype == torch.uint8
    if kv_nvfp4:
        if v.dtype != torch.uint8:
            raise ValueError("k and v must both be packed uint8 for NVFP4")
        if k_scale is None or v_scale is None:
            raise ValueError("NVFP4 KV requires k_scale and v_scale")
    elif not kv_fp8 and k.dtype != compute_dtype:
        raise ValueError("k/v dtype must match q (or be fp8/packed NVFP4)")
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5
    if kv_nvfp4 and k_global_scale is not None:
        softmax_scale = softmax_scale * float(k_global_scale)
    dev = q.device

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged decode requires seqused_k")
        if seqused_k.numel() != batch_size:
            raise ValueError(f"seqused_k must have batch_size ({batch_size}) entries")
        kv_last = (head_dim // 2) if kv_nvfp4 else head_dim
        if k.ndim != 4 or k.shape[2] != 128 or k.shape[3] != kv_last:
            raise ValueError(
                f"paged k/v must be (num_pages, num_kv_heads, 128, {kv_last})"
            )
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = seqused_k.to(dev).cumsum(0)
        pt_dev = page_table.contiguous()
    else:
        if cu_seqlens_k is None:
            raise ValueError("flat decode requires cu_seqlens_k")
        if k.ndim != 3:
            raise ValueError("flat k/v must be (total_k, num_kv_heads, head_dim)")
        cu_k = cu_seqlens_k.to(dev)
        pt_dev = torch.zeros((1, 1), dtype=torch.int32, device=dev)

    # v mirrors k (same layout/dtype for bf16/fp16, fp8, and packed NVFP4); the
    # kernel indexes v with k-derived coordinates, so a mismatched v would read
    # out of bounds.
    if v.shape != k.shape or v.dtype != k.dtype:
        raise ValueError("v must have the same shape and dtype as k")

    cu_q_loc = torch.arange(0, total_q + 1, seqlen_q, dtype=torch.int32, device=dev)
    qoff_dev = _q_offset_tensor(q_offset, cu_q_loc, cu_k, dev)

    if partial_dtype is None:
        partial_dtype = compute_dtype
    o_partial = torch.empty(
        (topk, total_q, num_qo_heads, head_dim), dtype=partial_dtype, device=dev
    )
    lse_partial = torch.empty(
        (topk, total_q, num_qo_heads), dtype=torch.float32, device=dev
    )
    split_counts = torch.empty((total_q, num_kv_heads), dtype=torch.int32, device=dev)

    if kv_nvfp4:
        k_pass = k.view(torch.int32)
        v_pass = v.view(torch.int32)
        ksf_dev = k_scale.reshape(-1).contiguous()
        vsf_dev = v_scale.reshape(-1).contiguous()
    else:
        k_pass, v_pass = k, v
        ksf_dev = torch.zeros(1, dtype=torch.uint8, device=dev)
        vsf_dev = ksf_dev

    key = (
        "decode",
        str(q.dtype),
        q_fp8,
        group_size,
        topk,
        causal,
        paged,
        kv_fp8,
        kv_nvfp4,
        str(partial_dtype),
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        q_in_cdt = _cutlass_dtype(q.dtype)
        kv_cdt = _cutlass_dtype(k_pass.dtype)
        i32 = _cutlass_dtype(torch.int32)
        u8 = _cutlass_dtype(torch.uint8)
        kv_word = k_pass.shape[-1]  # 128 (or 16 int32 words for nvfp4)
        s_tq, s_hq, s_tk, s_hkv, s_b1, s_b0 = (cute.sym_int() for _ in range(6))
        s_pb, s_pm, s_ksf, s_vsf = (cute.sym_int() for _ in range(4))
        kv_shape = (s_tk, s_hkv, 128, kv_word) if paged else (s_tk, s_hkv, kv_word)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj = SparseDecodeForwardSm12x(
            head_dim=head_dim,
            group_size=group_size,
            topk=topk,
            num_threads=128,
            is_causal=causal,
            paged=paged,
            kv_fp8=kv_fp8,
            kv_nvfp4=kv_nvfp4,
            q_fp8=q_fp8,
        )
        compiled = cute.compile(
            kernel_obj,
            _fake(q_in_cdt, (s_tq, s_hq, head_dim)),
            _fake(kv_cdt, kv_shape),
            _fake(kv_cdt, kv_shape),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(u8, (s_ksf,), align=4),
            _fake(u8, (s_vsf,), align=4),
            _fake(i32, (s_hkv, s_tq, topk), align=4),
            _fake(_cutlass_dtype(partial_dtype), (topk, s_tq, s_hq, head_dim)),
            _fake(_cutlass_dtype(torch.float32), (topk, s_tq, s_hq), align=4),
            _fake(i32, (s_tq, s_hkv), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    compiled(
        q,
        k_pass,
        v_pass,
        pt_dev,
        ksf_dev,
        vsf_dev,
        q2k_indices,
        o_partial,
        lse_partial,
        split_counts,
        cu_k,
        qoff_dev,
        float(softmax_scale),
        int(seqlen_q),
        int(total_q),
        int(num_kv_heads),
    )

    lse_out = None
    if return_softmax_lse:
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=dev)
    out = _combine_partials(
        o_partial,
        lse_partial,
        split_counts,
        group_size,
        compute_dtype,
        lse_out=lse_out,
        out_scale=float(v_global_scale) if v_global_scale is not None else 1.0,
    )
    if return_softmax_lse:
        return out, lse_out
    return out
