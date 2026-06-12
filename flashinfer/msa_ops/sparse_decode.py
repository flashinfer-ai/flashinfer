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

Minimax Sparse Attention decode path for SM120/SM121 (Phase 4).

A decode step is a varlen batch with a short, uniform ``seqlen_q`` per
request. Each valid (kv-head, token, t) entry of ``q2k_indices`` becomes one
work item of the KV-major forward kernel: one token (x its GQA query heads)
attending one selected KV block, written to split slot ``t``. Because
:func:`sparse_topk_select` tail-pads invalid entries with -1, the slot is
simply the list position and the split count is the valid prefix length —
so the whole schedule is built with a handful of torch ops, with no CUDA
scheduler kernels.
"""

from typing import Optional

import torch

from .sparse_index_utils import SparseAttentionSchedule


def build_decode_schedule(
    q2k_indices: torch.Tensor,  # (Hkv, total_q, topk) int32, -1 tail-padded
    cu_seqlens_q: torch.Tensor,  # (B + 1,) int32, device
    cu_seqlens_k: torch.Tensor,  # (B + 1,) int32, device
) -> SparseAttentionSchedule:
    """Build a KV-major schedule for decode: one work item per valid
    (kv-head, token, selected-block) entry.

    ``q2k_indices`` must be ascending with ``-1`` entries tail-padded (as
    produced by :func:`sparse_topk_select`); the split slot of an entry is
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

    return SparseAttentionSchedule(
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


def sparse_decode_attention(
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
):
    """Sparse decode attention for SM120/SM121 (Phase 4).

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
        :func:`sparse_topk_select`).
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
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from ..utils import is_sm12x_supported
    from .cute_dsl.sparse_decode_sm12x import SparseDecodeForwardSm12x
    from .sparse_attention import _combine_partials, _compile_cache, _to_cute

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "sparse_decode_attention requires SM120 or SM121 and CUDA >= 12.8"
        )
    total_q, num_qo_heads, head_dim = q.shape
    if total_q % seqlen_q != 0:
        raise ValueError(
            f"q rows ({total_q}) must be batch_size * seqlen_q ({seqlen_q})"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16 or fp16, got {q.dtype}")
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
    topk = q2k_indices.shape[2]
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    kv_nvfp4 = k.dtype == torch.uint8
    if kv_nvfp4:
        if v.dtype != torch.uint8:
            raise ValueError("k and v must both be packed uint8 for NVFP4")
        if k_scale is None or v_scale is None:
            raise ValueError("NVFP4 KV requires k_scale and v_scale")
    elif not kv_fp8 and k.dtype != q.dtype:
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

    o_partial = torch.empty(
        (topk, total_q, num_qo_heads, head_dim), dtype=q.dtype, device=dev
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

    q_c = _to_cute(q, 2)
    k_c = _to_cute(k_pass, 3 if paged else 2)
    v_c = _to_cute(v_pass, 3 if paged else 2)
    pt_c = from_dlpack(pt_dev, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    ksf_c = from_dlpack(ksf_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    vsf_c = from_dlpack(vsf_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    idx_c = from_dlpack(q2k_indices, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    op_c = _to_cute(o_partial, 3)
    lse_c = from_dlpack(lse_partial, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    sc_c = from_dlpack(split_counts, assumed_align=4).mark_layout_dynamic(leading_dim=1)
    cuk_c = from_dlpack(cu_k, assumed_align=4).mark_layout_dynamic(leading_dim=0)

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = ("decode", str(q.dtype), group_size, topk, causal, paged, kv_fp8, kv_nvfp4)
    compiled = _compile_cache.get(key)
    if compiled is None:
        kernel_obj = SparseDecodeForwardSm12x(
            head_dim=head_dim,
            group_size=group_size,
            topk=topk,
            num_threads=128,
            is_causal=causal,
            paged=paged,
            kv_fp8=kv_fp8,
            kv_nvfp4=kv_nvfp4,
        )
        compiled = cute.compile(
            kernel_obj,
            q_c,
            k_c,
            v_c,
            pt_c,
            ksf_c,
            vsf_c,
            idx_c,
            op_c,
            lse_c,
            sc_c,
            cuk_c,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(seqlen_q),
            cutlass.Int32(total_q),
            cutlass.Int32(num_kv_heads),
            stream,
        )
        _compile_cache[key] = compiled

    compiled(
        q_c,
        k_c,
        v_c,
        pt_c,
        ksf_c,
        vsf_c,
        idx_c,
        op_c,
        lse_c,
        sc_c,
        cuk_c,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(seqlen_q),
        cutlass.Int32(total_q),
        cutlass.Int32(num_kv_heads),
        stream,
    )

    lse_out = None
    if return_softmax_lse:
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=dev)
    out = _combine_partials(
        o_partial,
        lse_partial,
        split_counts,
        group_size,
        q.dtype,
        lse_out=lse_out,
        out_scale=float(v_global_scale) if v_global_scale is not None else 1.0,
    )
    if return_softmax_lse:
        return out, lse_out
    return out
