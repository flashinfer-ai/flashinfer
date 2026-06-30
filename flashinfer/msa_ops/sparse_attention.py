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
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    return_softmax_lse: bool = False,
    k_scale: Optional[torch.Tensor] = None,
    v_scale: Optional[torch.Tensor] = None,
    k_global_scale: Optional[float] = None,
    v_global_scale: Optional[float] = None,
    q_offset=None,
    return_temperature_lse: bool = False,
    lse_temperature_scale: float = 1.0,
):
    """Minimax Sparse Attention forward (prefill) for SM120/SM121.

    Each query attends only the top-K KV blocks selected in ``q2k_indices``.
    Runs the query-tile **union** kernel: a CTA owns a tile of query tokens and
    runs an in-kernel online softmax over the *union* of the blocks they selected,
    writing the final output directly (no GMEM partials, no combine pass). The
    union metadata is built internally from ``q2k_indices``.

    ``q``/``k``/``v`` are ``(total_tokens, num_heads, head_dim)`` with varlen
    offsets ``cu_seqlens_q``/``cu_seqlens_k``; ``q2k_indices`` is
    ``(num_kv_heads, total_q, topk)`` int32 (ascending, -1 padded). ``q`` is
    bf16/fp16; ``k``/``v`` are bf16/fp16, fp8 (E4M3), or packed NVFP4 (uint8 with
    ``k_scale``/``v_scale``), in GQA or MHA layouts, flat or paged.

    Notable optional parameters:

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
    return_temperature_lse : bool
        Also return the MSA temperature LSE (exponent scaled by
        ``lse_temperature_scale``), same shape as the softmax LSE. When set, the
        return is ``(out, lse, lse_t)``.

    Returns
    -------
    torch.Tensor or tuple of torch.Tensor
        Output ``(total_q, num_qo_heads, head_dim)``; plus LSE if
        ``return_softmax_lse``; plus the temperature LSE if
        ``return_temperature_lse``.
    """
    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "msa_sparse_attention requires SM120 or SM121 (Blackwell) and CUDA >= 12.8"
        )
    compute_dtype = q.dtype
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16/fp16, got {q.dtype}")
    if q.ndim != 3:
        raise ValueError("q must be 3D (total_q, num_qo_heads, head_dim)")
    total_q, num_qo_heads, head_dim = q.shape
    num_kv_heads = k.shape[1]
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    group_size = num_qo_heads // num_kv_heads
    # union metadata uses a per-token bit mask -> reject a strided q2k (e.g. a bare
    # permute of msa_topk_select's output).
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
    elif k.dtype != compute_dtype or v.dtype != compute_dtype:
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
        page_table = page_table.contiguous()
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
        return_temperature_lse=return_temperature_lse,
        lse_temperature_scale=lse_temperature_scale,
        kv_fp8=kv_fp8,
        kv_nvfp4=kv_nvfp4,
        k_scale=k_scale,
        v_scale=v_scale,
        v_global_scale=v_global_scale,
    )


# int32 union membership mask -> at most 32 query tokens may share one tile.
_UNION_MAX_TOKENS_PER_TILE = 32
_LN2 = 0.6931471805599453


def _union_tile_config(group_size: int):
    """Largest MMA tile whose tokens_per_tile (m_block // group_size) fits the
    32-bit membership mask. Returns (m_block, num_threads)."""
    for m_block in (128, 64, 32):
        if m_block % group_size == 0 and m_block // group_size <= (
            _UNION_MAX_TOKENS_PER_TILE
        ):
            return m_block, m_block * 2
    raise ValueError(
        f"union path needs group_size dividing 32/64/128 with "
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
    return_temperature_lse=False,
    lse_temperature_scale=1.0,
    kv_fp8=False,
    kv_nvfp4=False,
    k_scale=None,
    v_scale=None,
    v_global_scale=None,
):
    """Union-tile prefill path. See
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
    # e4m3 block scales; bf16/fp16/fp8 pass K/V as-is with dummy (1,) scales.
    if kv_nvfp4:
        k_pass = k.view(torch.int32)
        v_pass = v.view(torch.int32)
        ksf_dev = k_scale.reshape(-1).contiguous()
        vsf_dev = v_scale.reshape(-1).contiguous()
    else:
        k_pass, v_pass = k, v
        ksf_dev = torch.zeros(1, dtype=torch.uint8, device=dev)
        vsf_dev = ksf_dev
    # V's global scale is applied to the output for any KV dtype (the union has no
    # combine kernel); K's was pre-folded into softmax_scale by the caller.
    out_scale = float(v_global_scale) if v_global_scale is not None else 1.0
    m_block, num_threads = _union_tile_config(group_size)
    tokens_per_tile = m_block // group_size

    ub, um, uc, wm, n = build_msa_union_metadata_device(
        q2k_indices, cu_q_dev, tokens_per_tile, topk
    )
    # outputs default to 0 / -inf so query tiles that emit no work item (or rows a
    # tile masks out entirely) read back as a zero output with -inf LSE.
    out = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=dev)
    # LSE buffers: full (Hq, total_q) log2-domain when requested, else a (1, 1) dummy.
    # The temperature LSE forces the plain LSE on (it is returned alongside it).
    need_lse = return_softmax_lse or return_temperature_lse

    def _lse_buf(on):
        if on:
            return torch.full(
                (num_qo_heads, total_q), -float("inf"), dtype=torch.float32, device=dev
            )
        return torch.zeros((1, 1), dtype=torch.float32, device=dev)

    lse2 = _lse_buf(need_lse)
    lse2_t = _lse_buf(return_temperature_lse)

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
        need_lse,
        return_temperature_lse,
    )
    compiled = _compile_cache.get(key)
    if compiled is None:
        cdt = _cutlass_dtype(q.dtype)  # compute dtype (Q + output)
        kdt = _cutlass_dtype(k_pass.dtype)  # K/V storage (fp8, or int32 for NVFP4)
        i32 = _cutlass_dtype(torch.int32)
        u8 = _cutlass_dtype(torch.uint8)
        f32 = _cutlass_dtype(torch.float32)
        s = [cute.sym_int() for _ in range(19)]
        kv_last = k_pass.shape[-1]  # head_dim (bf16/fp8) or 16 int32 words (NVFP4)
        kv_shape = (s[9], s[10], _BLK_KV, kv_last) if paged else (s[2], s[3], kv_last)
        kernel_obj = SparseAttentionUnionFwdSm12x(
            head_dim=head_dim,
            m_block_size=m_block,
            n_block_size=_BLK_KV,
            group_size=group_size,
            num_threads=num_threads,
            is_causal=causal,
            return_softmax_lse=need_lse,
            return_temperature_lse=return_temperature_lse,
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
            # mLse / mLseT: independent dims so the off-path (1, 1) dummy is accepted.
            _fake(f32, (s[17], s[18]), align=4),
            _fake(f32, (s[15], s[16]), align=4),
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
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    # device-native fill (no host->device copy) so the single-call path stays
    # CUDA-graph capturable; n is the static work-item upper bound.
    wc = torch.full((1,), n, dtype=torch.int32, device=dev)
    compiled(
        q,
        k_pass,
        v_pass,
        ksf_dev,
        vsf_dev,
        out,
        lse2,
        lse2_t,
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
        float(lse_temperature_scale),
        int(n),
    )
    # kernel writes log2-domain LSE as (Hq, total_q); the public contract is
    # natural-log (total_q, num_qo_heads).
    if return_temperature_lse:
        lse_out = (lse2 * _LN2).t().contiguous()
        lse_t_out = (lse2_t * _LN2).t().contiguous()
        return out, lse_out, lse_t_out
    if return_softmax_lse:
        lse_out = (lse2 * _LN2).t().contiguous()
        return out, lse_out
    return out
