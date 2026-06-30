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

from ..api_logging import flashinfer_api
from ..trace.templates.msa import msa_sparse_attention_trace
from ..utils import is_sm12x_supported

_BLK_KV = 128

_compile_cache: dict = {}


def _q_offset_tensor(
    q_offset,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    device,
) -> torch.Tensor:
    """Per-batch causal offset (MSA q_offset semantics: query global position
    = q_offset[b] + batch-local index). Defaults to right-aligned
    (seqlen_k - seqlen_q)."""
    if q_offset is None:
        return (
            (cu_seqlens_k[1:] - cu_seqlens_k[:-1])
            - (cu_seqlens_q[1:] - cu_seqlens_q[:-1])
        ).to(torch.int32)
    if isinstance(q_offset, int):
        n = cu_seqlens_q.numel() - 1
        return torch.full((n,), q_offset, dtype=torch.int32, device=device)
    if q_offset.dtype != torch.int32:
        raise ValueError("q_offset must be int32")
    return q_offset.to(device)


def _cutlass_dtype(torch_dtype: torch.dtype):
    import cutlass

    return {
        torch.bfloat16: cutlass.BFloat16,
        torch.float16: cutlass.Float16,
        torch.float32: cutlass.Float32,
        torch.float8_e4m3fn: cutlass.Float8E4M3FN,
        torch.uint8: cutlass.Uint8,
        torch.int32: cutlass.Int32,
    }[torch_dtype]


def _fake(dtype, shape, align=16):
    """Fake compact row-major tensor for TVM-FFI compilation (FlashInfer's
    established cute-dsl pattern: compile once against symbolic shapes, then
    pass torch tensors directly at runtime)."""
    import cutlass.cute as cute

    return cute.runtime.make_fake_compact_tensor(
        dtype,
        shape,
        stride_order=tuple(reversed(range(len(shape)))),
        assumed_align=align,
    )


def _get_compiled_combine(
    partial_dtype: torch.dtype,
    out_dtype: torch.dtype,
    topk: int,
    head_dim: int,
    has_lse_out: bool,
    has_lse_t: bool,
):
    """Compile (once, cached) the CuTe-DSL combine kernel for a config.

    All fake tensor dims are independent symbols: the kernel's loop bounds
    (``topk``, ``head_dim``) come from the kernel object, not tensor shapes, so
    the shapes only fix ndim/strides, so unused optional tensors can be
    passed as small dummies at runtime without symbol conflicts."""
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.sparse_combine_sm12x import SparseCombineSm12x

    key = (
        "combine",
        str(partial_dtype),
        str(out_dtype),
        topk,
        head_dim,
        has_lse_out,
        has_lse_t,
    )
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled

    def fsyms(dtype, ndim, align=16):
        return _fake(
            _cutlass_dtype(dtype),
            tuple(cute.sym_int() for _ in range(ndim)),
            align=align,
        )

    kernel_obj = SparseCombineSm12x(
        head_dim=head_dim,
        topk=topk,
        partial_is_fp8=partial_dtype == torch.float8_e4m3fn,
        has_lse_out=has_lse_out,
        has_lse_t=has_lse_t,
        num_threads=128,
    )
    stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    compiled = cute.compile(
        kernel_obj,
        fsyms(partial_dtype, 4),  # o_partial (topk, total_q, Hq, d)
        fsyms(torch.float32, 3, align=4),  # lse_partial
        fsyms(torch.int32, 2, align=4),  # split_counts
        fsyms(out_dtype, 3),  # out
        fsyms(torch.float32, 2, align=4),  # lse_out (or dummy)
        fsyms(torch.float32, 3, align=4),  # lse_t_partial (or dummy)
        fsyms(torch.float32, 2, align=4),  # lse_t_out (or dummy)
        cutlass.Float32(1.0),
        cutlass.Int32(1),
        cutlass.Int32(1),
        cutlass.Int32(1),
        stream_fake,
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    return compiled


def _combine_partials_cudsl(
    o_partial: torch.Tensor,  # [topk, total_q, Hq, d]
    lse_partial: torch.Tensor,  # [topk, total_q, Hq] f32, log2 domain
    split_counts: torch.Tensor,  # [total_q, Hkv] int32
    group_size: int,
    out_dtype: torch.dtype,
    lse_out: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    lse_t_partial: Optional[torch.Tensor] = None,
    lse_t_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused CuTe-DSL LSE-weighted reduction over each query's split slots."""
    topk, total_q, num_qo_heads, head_dim = o_partial.shape
    out = torch.empty(
        (total_q, num_qo_heads, head_dim), dtype=out_dtype, device=o_partial.device
    )
    has_lse_out = lse_out is not None
    has_lse_t = lse_t_out is not None
    dev = o_partial.device
    dummy2 = torch.empty((1, 1), dtype=torch.float32, device=dev)
    dummy3 = torch.empty((1, 1, 1), dtype=torch.float32, device=dev)
    compiled = _get_compiled_combine(
        o_partial.dtype, out_dtype, topk, head_dim, has_lse_out, has_lse_t
    )
    compiled(
        o_partial,
        lse_partial,
        split_counts,
        out,
        lse_out if has_lse_out else dummy2,
        lse_t_partial if has_lse_t else dummy3,
        lse_t_out if has_lse_t else dummy2,
        float(out_scale),
        int(total_q),
        int(num_qo_heads),
        int(group_size),
    )
    return out


def _combine_partials(
    o_partial: torch.Tensor,  # [topk, total_q, Hq, d]
    lse_partial: torch.Tensor,  # [topk, total_q, Hq] f32, log2 domain
    split_counts: torch.Tensor,  # [total_q, Hkv] int32
    group_size: int,
    out_dtype: torch.dtype,
    lse_out: Optional[torch.Tensor] = None,
    out_scale: float = 1.0,
    lse_t_partial: Optional[torch.Tensor] = None,
    lse_t_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused LSE-weighted reduction over each query's split slots (CuTe-DSL)."""
    return _combine_partials_cudsl(
        o_partial,
        lse_partial,
        split_counts,
        group_size,
        out_dtype,
        lse_out=lse_out,
        out_scale=out_scale,
        lse_t_partial=lse_t_partial,
        lse_t_out=lse_t_out,
    )


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


@flashinfer_api(trace=msa_sparse_attention_trace)
def msa_sparse_attention(
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
    union: bool = False,
    m_block_size: int = 64,
    num_threads: int = 128,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    partial_dtype: Optional[torch.dtype] = None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
    qk_dtype: Optional[torch.dtype] = None,
    pv_dtype: Optional[torch.dtype] = None,
):
    """Minimax Sparse Attention forward (prefill) for SM120/SM121.

    Each query attends only the top-K KV blocks selected in ``q2k_indices``.
    Work is distributed over (kv-head, KV block) CSR rows built by
    :func:`msa_build_k2q_csr` so each block is loaded once and shared by all
    queries that selected it.

    ``q``/``k``/``v`` are ``(total_tokens, num_heads, head_dim)`` with varlen
    offsets ``cu_seqlens_q``/``cu_seqlens_k``; ``q2k_indices`` is
    ``(num_kv_heads, total_q, topk)`` int32 (ascending, -1 padded).

    Notable optional parameters:

    schedule : MsaAttentionSchedule, optional
        Pre-computed schedule (from :func:`msa_build_k2q_csr`) to
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
    union : bool
        Use the query-tile **union** kernel instead of the default KV-major
        split-KV + combine. A CTA owns a tile of query tokens and runs an
        in-kernel online softmax over the *union* of the blocks they selected,
        writing the final output directly (no GMEM partials, no combine) -- the
        memory-bound KV-major path's prefill headroom on SM12x. Currently
        bf16/fp16, flat K/V, no ``return_temperature_lse``; ``schedule`` is
        ignored. The metadata is built internally from ``q2k_indices``.

    Returns
    -------
    torch.Tensor or (torch.Tensor, torch.Tensor)
        Output ``(total_q, num_qo_heads, head_dim)``; plus LSE if
        ``return_softmax_lse``.
    """
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl import SparseAttentionForwardSm12x
    from .sparse_index_utils import msa_build_k2q_csr

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "msa_sparse_attention requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )
    q_fp8 = q.dtype == torch.float8_e4m3fn
    compute_dtype = torch.bfloat16 if q_fp8 else q.dtype
    qk_fp8_mma = qk_dtype == torch.float8_e4m3fn
    if qk_fp8_mma and not q_fp8:
        raise ValueError("qk_dtype=float8_e4m3fn requires fp8 Q")
    if qk_fp8_mma and k.dtype != torch.float8_e4m3fn:
        raise ValueError("qk_dtype=float8_e4m3fn requires fp8 K/V")
    pv_fp8_mma = pv_dtype == torch.float8_e4m3fn
    if pv_fp8_mma and v.dtype != torch.float8_e4m3fn:
        raise ValueError("pv_dtype=float8_e4m3fn requires fp8 V")
    if not q_fp8 and q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16/fp16/fp8_e4m3, got {q.dtype}")
    if q.ndim != 3:
        raise ValueError("q must be 3D (total_q, num_qo_heads, head_dim)")
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
    if num_threads % 32 != 0:
        raise ValueError(f"num_threads must be a multiple of 32, got {num_threads}")
    # CSR builder compiles for a compact q2k layout; reject a strided q2k (e.g. a
    # bare permute of msa_topk_select's output).
    if q2k_indices.dtype != torch.int32 or q2k_indices.ndim != 3:
        raise ValueError("q2k_indices must be int32 of shape (Hkv, total_q, topk)")
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
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
    elif (k.dtype != compute_dtype or v.dtype != compute_dtype) and not (
        q_fp8 and k.dtype == torch.float8_e4m3fn
    ):
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

    # v mirrors k (same layout/dtype for bf16/fp16, fp8, and packed NVFP4); the
    # kernel indexes v with k-derived coordinates, so a mismatched v would read
    # out of bounds.
    if v.shape != k.shape or v.dtype != k.dtype:
        raise ValueError("v must have the same shape and dtype as k")

    for name, t in (("k", k), ("v", v), ("q2k_indices", q2k_indices)):
        if t.device != q.device:
            raise ValueError(f"{name} must be on the same device as q ({q.device})")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous")

    if union:
        # query-tile + block-union + in-kernel online softmax: no GMEM partials and
        # no combine kernel (the memory-bound KV-major path's ~6x prefill headroom on
        # SM12x). bf16/fp16 flat K/V only for now.
        if q_fp8 or qk_fp8_mma or pv_fp8_mma:
            raise ValueError(
                "union=True supports bf16/fp16 Q with bf16/fp16, fp8 (E4M3), or "
                "NVFP4 K/V; q_fp8 / native fp8-MMA are not on the union path yet"
            )
        if return_temperature_lse:
            raise ValueError("union=True does not support return_temperature_lse yet")
        # paged is supported (page_size == _BLK_KV, one page per KV block); the
        # paged k/v shape was validated above and cu_seqlens_k was derived from
        # seqused_k. The union metadata builder works on batch-local block ids
        # either way -- the forward remaps them through page_table when paged.
        dev = q.device
        cu_q_dev = cu_seqlens_q.to(dev, non_blocking=True)
        cu_k_dev = cu_seqlens_k.to(dev, non_blocking=True)
        qoff_dev = _q_offset_tensor(q_offset, cu_q_dev, cu_k_dev, dev)
        return _msa_sparse_attention_union(
            q,
            k,
            v,
            q2k_indices,
            cu_q_dev,
            cu_k_dev,
            qoff_dev,
            page_table=page_table if paged else None,
            causal=causal,
            softmax_scale=softmax_scale,
            group_size=group_size,
            topk=topk,
            compute_dtype=compute_dtype,
            return_softmax_lse=return_softmax_lse,
            kv_fp8=kv_fp8,
            kv_nvfp4=kv_nvfp4,
            k_scale=k_scale,
            v_scale=v_scale,
            v_global_scale=v_global_scale,
        )

    if schedule is None:
        schedule = msa_build_k2q_csr(
            q2k_indices,
            cu_seqlens_q,
            cu_seqlens_k,
            blk_kv=_BLK_KV,
            target_q_per_cta=target_q_per_cta,
        )
    elif schedule.topk != topk or schedule.split_counts.shape != (
        total_q,
        num_kv_heads,
    ):
        # A schedule built for a different topk / total_q / head count would index
        # o_partial and the CSR head rows out of range; reject it instead.
        raise ValueError(
            "caller-provided schedule does not match (total_q, num_kv_heads, topk)"
        )

    dev = q.device
    if partial_dtype is None:
        partial_dtype = compute_dtype
    if partial_dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
    ):
        raise ValueError(f"unsupported partial_dtype {partial_dtype}")
    o_partial = torch.empty(
        (topk, total_q, num_qo_heads, head_dim), dtype=partial_dtype, device=dev
    )
    lse_partial = torch.empty(
        (topk, total_q, num_qo_heads), dtype=torch.float32, device=dev
    )
    if return_temperature_lse:
        lse_t_partial = torch.empty(
            (topk, total_q, num_qo_heads), dtype=torch.float32, device=dev
        )
    else:
        lse_t_partial = torch.zeros((1, 1, 1), dtype=torch.float32, device=dev)

    cu_q_dev = cu_seqlens_q.to(dev, non_blocking=True)
    cu_k_dev = cu_seqlens_k.to(dev, non_blocking=True)
    qoff_dev = _q_offset_tensor(q_offset, cu_q_dev, cu_k_dev, dev)

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

    key = (
        "sparse_attn",
        str(q.dtype),
        q_fp8,
        qk_fp8_mma,
        pv_fp8_mma,
        group_size,
        causal,
        paged,
        kv_fp8,
        kv_nvfp4,
        m_block_size,
        num_threads,
        str(partial_dtype),
        return_temperature_lse,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        q_in_cdt = _cutlass_dtype(q.dtype)
        kv_cdt = _cutlass_dtype(k_pass.dtype)
        i32 = _cutlass_dtype(torch.int32)
        u8 = _cutlass_dtype(torch.uint8)
        kv_last = k_pass.shape[-1]  # 128 (or 16 int32 words for nvfp4)
        s_tq, s_hq, s_tk, s_hkv, s_topk, s_b1, s_b0 = (cute.sym_int() for _ in range(7))
        s_lt0, s_lt1, s_lt2 = (cute.sym_int() for _ in range(3))
        s_pb, s_pm, s_ksf, s_vsf, s_tr, s_qs, s_cap = (cute.sym_int() for _ in range(7))
        kv_shape = (s_tk, s_hkv, _BLK_KV, kv_last) if paged else (s_tk, s_hkv, kv_last)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj = SparseAttentionForwardSm12x(
            head_dim=head_dim,
            m_block_size=m_block_size,
            n_block_size=_BLK_KV,
            group_size=group_size,
            num_threads=num_threads,
            is_causal=causal,
            paged=paged,
            kv_fp8=kv_fp8,
            kv_nvfp4=kv_nvfp4,
            return_temperature_lse=return_temperature_lse,
            q_fp8=q_fp8,
            qk_fp8_mma=qk_fp8_mma,
            pv_fp8_mma=pv_fp8_mma,
        )
        compiled = cute.compile(
            kernel_obj,
            _fake(q_in_cdt, (s_tq, s_hq, head_dim)),
            _fake(kv_cdt, kv_shape),
            _fake(kv_cdt, kv_shape),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(u8, (s_ksf,), align=4),
            _fake(u8, (s_vsf,), align=4),
            _fake(_cutlass_dtype(partial_dtype), (s_topk, s_tq, s_hq, head_dim)),
            _fake(_cutlass_dtype(torch.float32), (s_topk, s_tq, s_hq), align=4),
            _fake(_cutlass_dtype(torch.float32), (s_lt0, s_lt1, s_lt2), align=4),
            _fake(i32, (s_hkv, s_tr), align=4),
            _fake(i32, (s_hkv, s_qs), align=4),
            _fake(i32, (s_cap, 6), align=4),
            _fake(i32, (1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
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
        o_partial,
        lse_partial,
        lse_t_partial,
        schedule.row_ptr,
        schedule.qsplit_indices,
        schedule.scheduler_metadata,
        schedule.work_count,
        cu_q_dev,
        cu_k_dev,
        qoff_dev,
        float(softmax_scale),
        float(lse_temperature_scale),
        int(schedule.work_capacity),
    )

    lse_out = None
    if return_softmax_lse or return_temperature_lse:
        lse_out = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=dev)
    lse_t_out = None
    if return_temperature_lse:
        lse_t_out = torch.empty(
            (total_q, num_qo_heads), dtype=torch.float32, device=dev
        )
    out = _combine_partials(
        o_partial,
        lse_partial,
        schedule.split_counts,
        group_size,
        compute_dtype,
        lse_out=lse_out,
        out_scale=float(v_global_scale) if v_global_scale is not None else 1.0,
        lse_t_partial=lse_t_partial if return_temperature_lse else None,
        lse_t_out=lse_t_out,
    )
    if return_temperature_lse:
        return out, lse_out, lse_t_out
    if return_softmax_lse:
        return out, lse_out
    return out


# int32 union membership mask -> at most 32 query tokens may share one tile.
_UNION_MAX_TOKENS_PER_TILE = 32
_LN2 = 0.6931471805599453


def _union_tile_config(group_size: int):
    """Pick (m_block, num_threads) for the union path: an 8-row-per-warp MMA tile
    (m_block=128, 256 threads) when group_size divides it and the resulting
    tokens_per_tile (= m_block // group_size) fits the 32-bit membership mask, else
    the 64-row tile. M3 attention (group_size 16) -> 128 / 256 / tpt 8."""
    for m_block in (128, 64):
        if m_block % group_size == 0 and m_block // group_size <= (
            _UNION_MAX_TOKENS_PER_TILE
        ):
            return m_block, m_block * 2
    raise ValueError(
        f"union path needs group_size dividing 64 or 128 with "
        f"<= {_UNION_MAX_TOKENS_PER_TILE} tokens/tile, got group_size {group_size}"
    )


def _msa_sparse_attention_union(
    q,
    k,
    v,
    q2k_indices,
    cu_q_dev,
    cu_k_dev,
    qoff_dev,
    *,
    page_table=None,
    causal,
    softmax_scale,
    group_size,
    topk,
    compute_dtype,
    return_softmax_lse,
    kv_fp8=False,
    kv_nvfp4=False,
    k_scale=None,
    v_scale=None,
    v_global_scale=None,
):
    """Union-tile prefill path (query-tile + block-union + in-kernel online softmax,
    no GMEM partials / no combine). bf16/fp16 Q; K/V are bf16/fp16, fp8 (E4M3), or
    NVFP4 -- the quantized caches are dequantized to the compute dtype on load. The
    NVFP4 K global scale is pre-folded into ``softmax_scale`` by the caller; the V
    global scale is passed as ``out_scale`` and applied to the output here (the union
    has no combine kernel). flat K/V, or paged ``(num_pages, Hkv, _BLK_KV, d)`` with
    one page per KV block when ``page_table`` is given. See
    :class:`...cute_dsl.sparse_fwd_union_sm12x.SparseAttentionUnionFwdSm12x`."""
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.sparse_fwd_union_sm12x import SparseAttentionUnionFwdSm12x
    from ._union_metadata import build_msa_union_metadata_device

    total_q, num_qo_heads, head_dim = q.shape
    dev = q.device
    paged = page_table is not None
    # page table is read only on the paged path; a (1, 1) dummy keeps the call
    # signature uniform for the flat kernel.
    page_table_arg = (
        page_table if paged else torch.zeros((1, 1), dtype=torch.int32, device=dev)
    )
    # NVFP4: pass int32 views of the packed bytes (one word = 8 e2m1 values) + flat
    # e4m3 block scales. K's global scale was pre-folded into softmax_scale by the
    # caller; V's becomes out_scale (applied to the output, since union has no
    # combine). bf16/fp16/fp8 pass K/V as-is with dummy (1,) scales and out_scale 1.
    if kv_nvfp4:
        k_pass = k.view(torch.int32)
        v_pass = v.view(torch.int32)
        ksf_dev = k_scale.reshape(-1).contiguous()
        vsf_dev = v_scale.reshape(-1).contiguous()
        out_scale = float(v_global_scale) if v_global_scale is not None else 1.0
    else:
        k_pass, v_pass = k, v
        ksf_dev = torch.zeros(1, dtype=torch.uint8, device=dev)
        vsf_dev = ksf_dev
        out_scale = 1.0
    m_block, num_threads = _union_tile_config(group_size)
    tokens_per_tile = m_block // group_size

    # On-device metadata builder: keeps the (Hkv, total_q, topk) selection on the
    # GPU (only cu_seqlens is read host-side). The host reference
    # build_msa_union_metadata is retained as the test oracle.
    ub, um, uc, wm, n = build_msa_union_metadata_device(
        q2k_indices, cu_q_dev, tokens_per_tile, topk
    )
    # outputs default to 0 / -inf so query tiles that emit no work item (or rows a
    # tile masks out entirely) read back as a zero output with -inf LSE.
    out = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=dev)
    lse2 = torch.full(
        (num_qo_heads, total_q), -float("inf"), dtype=torch.float32, device=dev
    )

    key = (
        "sparse_attn_union",
        str(q.dtype),
        str(k_pass.dtype),
        group_size,
        causal,
        m_block,
        num_threads,
        paged,
        kv_fp8,
        kv_nvfp4,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        cdt = _cutlass_dtype(q.dtype)  # compute dtype (Q + output)
        kdt = _cutlass_dtype(k_pass.dtype)  # K/V storage (fp8, or int32 for NVFP4)
        i32 = _cutlass_dtype(torch.int32)
        u8 = _cutlass_dtype(torch.uint8)
        f32 = _cutlass_dtype(torch.float32)
        s = [cute.sym_int() for _ in range(15)]
        kv_last = k_pass.shape[-1]  # head_dim (bf16/fp8) or 16 int32 words (NVFP4)
        # K/V are 3D flat or 4D paged (num_pages, Hkv, _BLK_KV, kv_last).
        kv_shape = (s[9], s[10], _BLK_KV, kv_last) if paged else (s[2], s[3], kv_last)
        kernel_obj = SparseAttentionUnionFwdSm12x(
            head_dim=head_dim,
            m_block_size=m_block,
            n_block_size=_BLK_KV,
            group_size=group_size,
            num_threads=num_threads,
            is_causal=causal,
            return_softmax_lse=True,
            paged=paged,
            kv_fp8=kv_fp8,
            kv_nvfp4=kv_nvfp4,
        )
        compiled = cute.compile(
            kernel_obj,
            _fake(cdt, (s[0], s[1], head_dim)),
            _fake(kdt, kv_shape),
            _fake(kdt, kv_shape),
            _fake(u8, (s[13],), align=4),
            _fake(u8, (s[14],), align=4),
            _fake(cdt, (s[0], s[1], head_dim)),
            _fake(f32, (s[1], s[0]), align=4),
            _fake(i32, (s[6], s[7]), align=4),
            _fake(i32, (s[6], s[7]), align=4),
            _fake(i32, (s[6],), align=4),
            _fake(i32, (s[6], 3), align=4),
            _fake(i32, (s[8],), align=4),
            _fake(i32, (s[4],), align=4),
            _fake(i32, (s[4],), align=4),
            _fake(i32, (s[5],), align=4),
            _fake(i32, (s[11], s[12]), align=4),
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    wc = torch.tensor([n], dtype=torch.int32, device=dev)
    compiled(
        q,
        k_pass,
        v_pass,
        ksf_dev,
        vsf_dev,
        out,
        lse2,
        ub,
        um,
        uc,
        wm,
        wc,
        cu_q_dev,
        cu_k_dev,
        qoff_dev,
        page_table_arg,
        float(softmax_scale),
        float(out_scale),
        int(n),
    )
    if return_softmax_lse:
        # kernel writes log2-domain LSE as (Hq, total_q); the public contract is
        # natural-log (total_q, num_qo_heads).
        lse_out = (lse2 * _LN2).t().contiguous()
        return out, lse_out
    return out
