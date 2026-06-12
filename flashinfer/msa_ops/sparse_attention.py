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
"""

from typing import Optional

import torch

from ..utils import is_sm12x_supported

_BLK_KV = 128
_MAX_KV_BLOCKS = 4096  # SMEM selection-map capacity: max_seqlen_k <= 4096 * 128

_compile_cache: dict = {}


def _to_cute(t: torch.Tensor, leading_dim: int):
    from cutlass.cute.runtime import from_dlpack

    dtype_width = t.element_size() * 8
    return (
        from_dlpack(t, assumed_align=16)
        .mark_layout_dynamic(leading_dim=leading_dim)
        .mark_compact_shape_dynamic(
            mode=leading_dim,
            stride_order=t.dim_order(),
            divisibility=128 // dtype_width,
        )
    )


def sparse_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: Optional[int] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Minimax Sparse Attention forward pass for SM120/SM121.

    Computes attention where each query token attends only to its top-K
    selected KV blocks (of ``blk_kv = 128`` tokens each), as chosen by
    :func:`sparse_topk_select`.

    This is the Phase-3a q-major kernel: each CTA processes a tile of query
    tokens for one query head and gathers the union of the tile's selected
    KV blocks. (The KV-major CSR kernel consuming :func:`build_k2q_csr`
    output is Phase 3b.)

    Parameters
    ----------
    q : torch.Tensor
        Shape ``(total_q, num_qo_heads, head_dim)``, bf16 or fp16, varlen
        packed. ``head_dim`` must be 128.
    k, v : torch.Tensor
        Shape ``(total_k, num_kv_heads, head_dim)``, same dtype as ``q``.
        ``num_qo_heads`` must be a multiple of ``num_kv_heads`` (GQA).
    q2k_indices : torch.Tensor
        Shape ``(num_kv_heads, total_q, topk)``, int32. Per (kv-head, query
        token): batch-local KV block indices to attend, ascending, ``-1``
        padded. Shared across the q heads in each GQA group.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        Shape ``(batch_size + 1,)``, int32 cumulative sequence lengths.
    max_seqlen_q : int, optional
        Maximum query length over the batch; computed from ``cu_seqlens_q``
        if omitted.
    causal : bool
        Apply right-aligned causal masking within selected blocks.
    softmax_scale : float, optional
        Defaults to ``head_dim ** -0.5``.
    output : torch.Tensor, optional
        Pre-allocated output, same shape/dtype as ``q``.

    Returns
    -------
    torch.Tensor
        Shape ``(total_q, num_qo_heads, head_dim)``, dtype of ``q``. Query
        tokens with no valid selected blocks produce zeros.
    """
    import cuda.bindings.driver as cuda_driver
    import cutlass

    from .cute_dsl import SparseAttentionForwardSm12x

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "sparse_attention requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )

    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16 or fp16, got {q.dtype}")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("q, k, v must share a dtype")
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, v must be 3D (total_tokens, num_heads, head_dim)")
    total_q, num_qo_heads, head_dim = q.shape
    total_k, num_kv_heads, head_dim_k = k.shape
    if head_dim != 128 or head_dim_k != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}/{head_dim_k}")
    if v.shape != k.shape:
        raise ValueError("k and v must have the same shape")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError(
            f"num_qo_heads ({num_qo_heads}) must be a multiple of "
            f"num_kv_heads ({num_kv_heads})"
        )
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must be int32 of shape (Hkv, total_q, topk)")
    if q2k_indices.shape[0] != num_kv_heads or q2k_indices.shape[1] != total_q:
        raise ValueError(
            f"q2k_indices shape {tuple(q2k_indices.shape)} does not match "
            f"(num_kv_heads={num_kv_heads}, total_q={total_q}, topk)"
        )
    topk = q2k_indices.shape[2]
    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_k.dtype != torch.int32:
        raise ValueError("cu_seqlens_q/cu_seqlens_k must be int32")
    for t in (q, k, v, q2k_indices, cu_seqlens_q, cu_seqlens_k):
        if not t.is_contiguous():
            raise ValueError("all input tensors must be contiguous")

    batch_size = cu_seqlens_q.numel() - 1
    cu_q_cpu = cu_seqlens_q.cpu()
    cu_k_cpu = cu_seqlens_k.cpu()
    if max_seqlen_q is None:
        max_seqlen_q = int((cu_q_cpu[1:] - cu_q_cpu[:-1]).max().item())
    max_seqlen_k = int((cu_k_cpu[1:] - cu_k_cpu[:-1]).max().item())
    if (max_seqlen_k + _BLK_KV - 1) // _BLK_KV > _MAX_KV_BLOCKS:
        raise ValueError(
            f"max_seqlen_k {max_seqlen_k} exceeds the supported limit of "
            f"{_MAX_KV_BLOCKS * _BLK_KV} tokens"
        )

    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    if output is None:
        output = torch.empty_like(q)
    else:
        if output.shape != q.shape or output.dtype != q.dtype:
            raise ValueError("output must match q's shape and dtype")
        if not output.is_contiguous():
            raise ValueError("output must be contiguous")

    cu_q_dev = cu_seqlens_q.to(q.device, non_blocking=True)
    cu_k_dev = cu_seqlens_k.to(q.device, non_blocking=True)

    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    q_c = _to_cute(q, 2)
    k_c = _to_cute(k, 2)
    v_c = _to_cute(v, 2)
    o_c = _to_cute(output, 2)
    idx_c = from_dlpack(q2k_indices, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    cuq_c = from_dlpack(cu_q_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    cuk_c = from_dlpack(cu_k_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    dtype_key = str(q.dtype)
    key = (dtype_key, topk, causal)
    compiled = _compile_cache.get(key)
    if compiled is None:
        kernel_obj = SparseAttentionForwardSm12x(
            head_dim=head_dim,
            m_block_size=64,
            n_block_size=_BLK_KV,
            topk=topk,
            num_threads=128,
            is_causal=causal,
            max_kv_blocks=_MAX_KV_BLOCKS,
        )
        compiled = cute.compile(
            kernel_obj,
            q_c,
            k_c,
            v_c,
            o_c,
            idx_c,
            cuq_c,
            cuk_c,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(max_seqlen_q),
            cutlass.Int32(batch_size),
            cutlass.Int32(num_qo_heads),
            stream,
        )
        _compile_cache[key] = compiled

    compiled(
        q_c,
        k_c,
        v_c,
        o_c,
        idx_c,
        cuq_c,
        cuk_c,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(max_seqlen_q),
        cutlass.Int32(batch_size),
        cutlass.Int32(num_qo_heads),
        stream,
    )
    return output


def _get_sparse_combine_module():
    from .jit import gen_sparse_combine_module

    if not hasattr(_get_sparse_combine_module, "_mod"):
        _get_sparse_combine_module._mod = gen_sparse_combine_module().build_and_load()
    return _get_sparse_combine_module._mod


def _combine_partials(
    o_partial: torch.Tensor,  # [topk, total_q, Hq, d]
    lse_partial: torch.Tensor,  # [topk, total_q, Hq] f32, log2 domain
    split_counts: torch.Tensor,  # [total_q, Hkv] int32
    group_size: int,
    out_dtype: torch.dtype,
    lse_out: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
) -> torch.Tensor:
    """Fused CUDA LSE-weighted reduction over each query's split slots."""
    topk, total_q, num_qo_heads, head_dim = o_partial.shape
    out = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=out_dtype, device=o_partial.device
    )
    _get_sparse_combine_module().sparse_combine(
        o_partial, lse_partial, split_counts, out, lse_out, group_size, out_scale
    )
    return out


def _combine_partials_torch(
    o_partial: torch.Tensor,  # [topk, total_q, Hq, d]
    lse_partial: torch.Tensor,  # [topk, total_q, Hq] f32, log2 domain
    split_counts: torch.Tensor,  # [total_q, Hkv] int32
    group_size: int,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    """LSE-weighted reduction over each query's split slots (torch reference)."""
    topk = o_partial.shape[0]
    counts = split_counts.repeat_interleave(group_size, dim=1)  # [total_q, Hq]
    slots = torch.arange(topk, device=o_partial.device).view(topk, 1, 1)
    mask = slots < counts.unsqueeze(0)  # [topk, total_q, Hq]
    neg_inf = torch.finfo(torch.float32).min
    lse = torch.where(mask, lse_partial, neg_inf)
    lse_max = lse.max(dim=0, keepdim=True).values
    w = torch.exp2(lse - lse_max)
    w = torch.where(mask, w, 0.0)
    denom = w.sum(dim=0)
    # mask the partials too: slots >= count hold uninitialized memory, and
    # 0 * inf/NaN would poison the sum
    o_masked = torch.where(mask.unsqueeze(-1), o_partial.float(), 0.0)
    out = (o_masked * w.unsqueeze(-1)).sum(dim=0)
    out = out / denom.unsqueeze(-1)
    out = torch.nan_to_num(out, nan=0.0)  # queries with zero valid splits
    return out.to(out_dtype)


def sparse_attention_kvmajor(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    causal: bool = False,
    softmax_scale: Optional[float] = None,
    schedule=None,
    target_q_per_cta: int = 128,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    m_block_size: int = 64,
    num_threads: int = 128,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
):
    """Minimax Sparse Attention forward (KV-major, Phase 3b) for SM120/SM121.

    Same semantics as :func:`sparse_attention`, but follows MSA's KV-major
    design: work is distributed over (kv-head, KV block) CSR rows built by
    :func:`build_k2q_csr_schedule`, each KV block is loaded once and shared
    by all queries that selected it, and per-block partial outputs are
    combined with a fused LSE-weighted reduction.

    Parameters are as in :func:`sparse_attention`, with these additions:

    schedule : SparseAttentionSchedule, optional
        Pre-computed schedule (from :func:`build_k2q_csr_schedule`) to
        amortize index preprocessing across layers.
    page_table : torch.Tensor, optional
        Enables the paged-KV path: ``k``/``v`` are then
        ``(num_pages, num_kv_heads, 128, head_dim)`` and ``page_table`` is
        ``(batch_size, max_pages)`` int32 mapping batch-local KV block i to
        a page. Requires ``seqused_k``.
    seqused_k : torch.Tensor, optional
        ``(batch_size,)`` int32, valid KV length per sequence (paged path).
        ``cu_seqlens_k`` may be omitted when this is given.
    return_softmax_lse : bool
        Also return the natural-log LSE, shape ``(total_q, num_qo_heads)``
        float32 (``-inf`` for queries with no valid selected blocks).

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Output ``(total_q, num_qo_heads, head_dim)``; plus LSE if
        ``return_softmax_lse``.
    """
    import cuda.bindings.driver as cuda_driver
    import cutlass
    import cutlass.cute as cute
    from cutlass.cute.runtime import from_dlpack

    from .cute_dsl import SparseAttentionForwardKvMajorSm12x
    from .sparse_index_utils import build_k2q_csr_schedule

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "sparse_attention_kvmajor requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16 or fp16, got {q.dtype}")
    total_q, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    group_size = num_qo_heads // num_kv_heads
    if group_size > m_block_size or m_block_size % group_size != 0:
        raise ValueError(
            f"GQA group size {group_size} must divide m_block_size {m_block_size}"
        )
    topk = q2k_indices.shape[2]
    if softmax_scale is None:
        softmax_scale = head_dim**-0.5

    kv_fp8 = k.dtype == torch.float8_e4m3fn
    kv_nvfp4 = k.dtype == torch.uint8
    if kv_fp8:
        if v.dtype != torch.float8_e4m3fn:
            raise ValueError("k and v must both be float8_e4m3fn")
    elif kv_nvfp4:
        if v.dtype != torch.uint8:
            raise ValueError("k and v must both be packed uint8 for NVFP4")
        if k_scale is None or v_scale is None:
            raise ValueError("NVFP4 KV requires k_scale and v_scale")
        if k_scale.dtype != torch.uint8 or v_scale.dtype != torch.uint8:
            raise ValueError("k_scale/v_scale must be uint8 (E4M3 bytes)")
        if k_global_scale is not None:
            softmax_scale = softmax_scale * float(k_global_scale)
    elif k.dtype != q.dtype or v.dtype != q.dtype:
        raise ValueError("k/v dtype must match q (or be fp8/packed NVFP4)")

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged KV requires seqused_k")
        kv_last_dim = head_dim // 2 if kv_nvfp4 else head_dim
        if k.ndim != 4 or k.shape[2] != _BLK_KV or k.shape[3] != kv_last_dim:
            raise ValueError(
                "paged k/v must be (num_pages, num_kv_heads, "
                f"{_BLK_KV}, {kv_last_dim}), got {tuple(k.shape)}"
            )
        if page_table.dtype != torch.int32 or page_table.ndim != 2:
            raise ValueError("page_table must be int32 of shape (batch, max_pages)")
        if seqused_k.dtype != torch.int32:
            raise ValueError("seqused_k must be int32")
        if cu_seqlens_k is None:
            cu_seqlens_k = torch.zeros(
                seqused_k.numel() + 1, dtype=torch.int32, device=seqused_k.device
            )
            cu_seqlens_k[1:] = seqused_k.cumsum(0)
    else:
        if cu_seqlens_k is None:
            raise ValueError("cu_seqlens_k is required for the non-paged path")
        if k.ndim != 3:
            raise ValueError("flat k/v must be (total_k, num_kv_heads, head_dim)")

    if schedule is None:
        schedule = build_k2q_csr_schedule(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv=_BLK_KV,
            target_q_per_cta=target_q_per_cta,
        )

    dev = q.device
    o_partial = torch.empty(
        (topk, total_q, num_qo_heads, head_dim), dtype=q.dtype, device=dev
    )
    lse_partial = torch.empty(
        (topk, total_q, num_qo_heads), dtype=torch.float32, device=dev
    )

    cu_q_dev = cu_seqlens_q.to(dev, non_blocking=True)
    cu_k_dev = cu_seqlens_k.to(dev, non_blocking=True)

    if paged:
        pt_dev = page_table.contiguous()
    else:
        # dummy 1x1 table for the flat path (kernel signature is shared)
        pt_dev = torch.zeros((1, 1), dtype=torch.int32, device=dev)

    if kv_nvfp4:
        # int32 views of the packed bytes: one word = 8 e2m1 values
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
    op_c = _to_cute(o_partial, 3)
    lse_c = from_dlpack(lse_partial, assumed_align=4).mark_layout_dynamic(leading_dim=2)
    rp_c = from_dlpack(schedule.row_ptr, assumed_align=4).mark_layout_dynamic(
        leading_dim=1
    )
    qs_c = from_dlpack(schedule.qsplit_indices, assumed_align=4).mark_layout_dynamic(
        leading_dim=1
    )
    sched_c = from_dlpack(
        schedule.scheduler_metadata, assumed_align=4
    ).mark_layout_dynamic(leading_dim=1)
    wc_c = from_dlpack(schedule.work_count, assumed_align=4).mark_layout_dynamic(
        leading_dim=0
    )
    cuq_c = from_dlpack(cu_q_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)
    cuk_c = from_dlpack(cu_k_dev, assumed_align=4).mark_layout_dynamic(leading_dim=0)

    stream = cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream)

    key = (
        "kvmajor",
        str(q.dtype),
        group_size,
        causal,
        paged,
        kv_fp8,
        kv_nvfp4,
        m_block_size,
        num_threads,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        kernel_obj = SparseAttentionForwardKvMajorSm12x(
            head_dim=head_dim,
            m_block_size=m_block_size,
            n_block_size=_BLK_KV,
            group_size=group_size,
            num_threads=num_threads,
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
            op_c,
            lse_c,
            rp_c,
            qs_c,
            sched_c,
            wc_c,
            cuq_c,
            cuk_c,
            cutlass.Float32(softmax_scale),
            cutlass.Int32(schedule.work_capacity),
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
        op_c,
        lse_c,
        rp_c,
        qs_c,
        sched_c,
        wc_c,
        cuq_c,
        cuk_c,
        cutlass.Float32(softmax_scale),
        cutlass.Int32(schedule.work_capacity),
        stream,
    )

    lse_out = None
    if return_softmax_lse:
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=dev)
    out = _combine_partials(
        o_partial,
        lse_partial,
        schedule.split_counts,
        group_size,
        q.dtype,
        lse_out=lse_out,
        out_scale=float(v_global_scale) if v_global_scale is not None else 1.0,
    )
    if return_softmax_lse:
        return out, lse_out
    return out
