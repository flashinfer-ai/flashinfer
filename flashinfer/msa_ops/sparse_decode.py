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
work item: one token (x its GQA query heads) attending one selected KV block,
written to split slot ``t``. The kernel addresses that work directly from a
static ``(topk, total_q, Hkv)`` grid (no schedule tensors); because
:func:`msa_topk_select` tail-pads invalid entries with -1, the slot is simply
the list position and the split count is the valid prefix length, computed
in-kernel. The split partials are reduced by the shared combine kernel.
"""

import functools
from typing import Optional

import torch

from ..api_logging import flashinfer_api
from ..trace.templates.msa import msa_sparse_decode_attention_trace


@functools.cache
def _dummy_tensors(device_index: int):
    # signature fillers for paths that never read them; cached so decode
    # calls launch no fill kernels
    dev = torch.device("cuda", device_index)
    return (
        torch.zeros((1, 1), dtype=torch.int32, device=dev),
        torch.zeros(1, dtype=torch.uint8, device=dev),
        torch.zeros(1, dtype=torch.int32, device=dev),
    )


def _decode_num_chunks(base_ctas: int, topk: int, device) -> int:
    """Top-k split factor: fill the SMs at low batch, 1 (fused) when already full."""
    from ..utils import get_device_sm_count

    if base_ctas <= 0 or topk <= 1:
        return 1
    num_sms = get_device_sm_count(device)
    if base_ctas >= num_sms:
        return 1
    chunks = -(-2 * num_sms // base_ctas)  # ceil to ~2 SM-waves
    pow2 = 1
    while pow2 * 2 <= chunks:
        pow2 *= 2
    return max(1, min(pow2, topk))


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
    force_fused: Optional[bool] = None,
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
    force_fused : bool, optional
        Override the adaptive split-K decision. By default each token's selected
        list is split into chunks (one CTA per chunk online-softmaxes its blocks
        and writes a partial; the combine kernel reduces them), with the chunk
        count adapting to fill the SMs — one block per chunk at low batch, a
        single chunk at high batch. A single chunk is the *fused* path: one CTA
        per (token, kv-head) writes the final output directly (no GMEM partials,
        no combine). ``True``/``False`` force fused/split on; ``None`` (default)
        adapts. Fused is unavailable for NVFP4 KV.

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
        _q_offset_explicit,
    )

    if not is_sm12x_supported(q.device):
        raise RuntimeError(
            "msa_sparse_decode_attention requires SM120 or SM121 and CUDA >= 12.8"
        )
    if q.ndim != 3:
        raise ValueError("q must be 3D (total_q, num_qo_heads, head_dim)")
    if seqlen_q <= 0:
        raise ValueError(f"seqlen_q must be positive, got {seqlen_q}")
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
    if (
        q2k_indices.ndim != 3
        or q2k_indices.dtype != torch.int32
        or q2k_indices.shape[:2] != (num_kv_heads, total_q)
    ):
        raise ValueError("q2k_indices must be int32 (num_kv_heads, total_q, topk)")
    # Compiled for a compact layout; reject a strided q2k (e.g. a bare permute of
    # msa_topk_select's output) here, matching msa_sparse_attention's guard.
    if not q2k_indices.is_contiguous():
        raise ValueError("q2k_indices must be contiguous")
    topk = q2k_indices.shape[2]
    if topk <= 0:
        raise ValueError("q2k_indices topk dimension must be positive")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    kv_nvfp4 = k.dtype == torch.uint8
    if kv_nvfp4:
        if v.dtype != torch.uint8:
            raise ValueError("k and v must both be packed uint8 for NVFP4")
        if k_scale is None or v_scale is None:
            raise ValueError("NVFP4 KV requires k_scale and v_scale")
        if k_scale.dtype != torch.uint8 or v_scale.dtype != torch.uint8:
            raise ValueError("k_scale/v_scale must be uint8 (E4M3 bytes)")
        if k_scale.device != q.device or v_scale.device != q.device:
            raise ValueError("k_scale/v_scale must be on the same device as q")
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
        if seqused_k.dtype != torch.int32 or seqused_k.ndim != 1:
            raise ValueError("seqused_k must be 1D int32")
        if seqused_k.numel() != batch_size:
            raise ValueError(f"seqused_k must have batch_size ({batch_size}) entries")
        if page_table.dtype != torch.int32 or page_table.ndim != 2:
            raise ValueError("page_table must be int32 of shape (batch, max_pages)")
        if page_table.shape[0] != batch_size:
            raise ValueError("page_table batch dimension must match q batch_size")
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
        if cu_seqlens_k.dtype != torch.int32 or cu_seqlens_k.ndim != 1:
            raise ValueError("cu_seqlens_k must be 1D int32")
        if cu_seqlens_k.numel() != batch_size + 1:
            raise ValueError("cu_seqlens_k must have batch_size + 1 entries")
        if k.ndim != 3:
            raise ValueError("flat k/v must be (total_k, num_kv_heads, head_dim)")
        cu_k = cu_seqlens_k.to(dev)
        pt_dev = _dummy_tensors(dev.index)[0]

    # v mirrors k (same layout/dtype for bf16/fp16, fp8, and packed NVFP4); the
    # kernel indexes v with k-derived coordinates, so a mismatched v would read
    # out of bounds.
    if v.shape != k.shape or v.dtype != k.dtype:
        raise ValueError("v must have the same shape and dtype as k")

    # the right-aligned default is computed in-kernel: no offset tensor built
    qoff_default = q_offset is None
    if qoff_default:
        qoff_dev = _dummy_tensors(dev.index)[2]
    else:
        qoff_dev = _q_offset_explicit(q_offset, batch_size, dev)

    if partial_dtype is None:
        partial_dtype = compute_dtype
    if partial_dtype not in (
        torch.float32,
        torch.bfloat16,
        torch.float16,
        torch.float8_e4m3fn,
    ):
        raise ValueError(f"unsupported partial_dtype {partial_dtype}")

    # adaptive split-K: num_chunks fills the (chunk x token x kv-head) grid at low
    # batch; num_chunks==1 is the fused path (no GMEM partials/combine) at high
    # batch. NVFP4 keeps the per-block split.
    if force_fused and kv_nvfp4:
        raise ValueError("force_fused is not supported with NVFP4 KV")
    if kv_nvfp4:
        fused, num_chunks = False, topk
    elif force_fused is True:
        fused, num_chunks = True, 1
    elif force_fused is False:
        fused, num_chunks = False, topk
    else:
        num_chunks = _decode_num_chunks(total_q * num_kv_heads, topk, dev)
        fused = num_chunks == 1

    if fused:
        # one CTA per (token, kv-head) writes the final output + LSE directly;
        # the split-path partial buffers collapse to dummies.
        out_buf = torch.empty(
            (total_q, num_qo_heads, head_dim), dtype=compute_dtype, device=dev
        )
        lse_buf = torch.empty((total_q, num_qo_heads), dtype=torch.float32, device=dev)
        # topk (shape[0]) and head_dim (shape[3]) are static in the signature
        o_partial = torch.empty((topk, 1, 1, head_dim), dtype=partial_dtype, device=dev)
        lse_partial = torch.empty((topk, 1, 1), dtype=torch.float32, device=dev)
        split_counts = torch.empty((1, 1), dtype=torch.int32, device=dev)
    else:
        o_partial = torch.empty(
            (topk, total_q, num_qo_heads, head_dim), dtype=partial_dtype, device=dev
        )
        lse_partial = torch.empty(
            (topk, total_q, num_qo_heads), dtype=torch.float32, device=dev
        )
        split_counts = torch.empty(
            (total_q, num_kv_heads), dtype=torch.int32, device=dev
        )
        # last dim must match the kernel's static head_dim in the signature
        out_buf = torch.empty((1, 1, head_dim), dtype=compute_dtype, device=dev)
        lse_buf = torch.empty((1, 1), dtype=torch.float32, device=dev)

    if kv_nvfp4:
        k_pass = k.view(torch.int32)
        v_pass = v.view(torch.int32)
        ksf_dev = k_scale.reshape(-1).contiguous()
        vsf_dev = v_scale.reshape(-1).contiguous()
    else:
        k_pass, v_pass = k, v
        ksf_dev = _dummy_tensors(dev.index)[1]
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
        fused,
        qoff_default,
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
        # Fused and split paths each pass dummies for the other's output tensors, so
        # those get their own symbols (not tied to mQ's total_q/Hq); only the
        # always-real inputs keep shared symbols. topk and head_dim stay static.
        s_ptq, s_phq = cute.sym_int(), cute.sym_int()  # mOp/mLse (split partials)
        s_sc0, s_sc1 = cute.sym_int(), cute.sym_int()  # mSplitCounts
        s_otq, s_ohq = cute.sym_int(), cute.sym_int()  # mOut/mLseOut (fused)
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
            fused=fused,
            qoff_default=qoff_default,
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
            _fake(_cutlass_dtype(partial_dtype), (topk, s_ptq, s_phq, head_dim)),
            _fake(_cutlass_dtype(torch.float32), (topk, s_ptq, s_phq), align=4),
            _fake(i32, (s_sc0, s_sc1), align=4),
            _fake(_cutlass_dtype(compute_dtype), (s_otq, s_ohq, head_dim)),  # mOut
            _fake(_cutlass_dtype(torch.float32), (s_otq, s_ohq), align=4),  # mLseOut
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),  # out_scale
            cutlass.Int32(1),  # seqlen_q
            cutlass.Int32(1),  # total_q
            cutlass.Int32(1),  # num_kv_heads
            cutlass.Int32(1),  # num_chunks
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
        out_buf,
        lse_buf,
        cu_k,
        qoff_dev,
        float(softmax_scale),
        float(v_global_scale) if v_global_scale is not None else 1.0,
        int(seqlen_q),
        int(total_q),
        int(num_kv_heads),
        int(num_chunks),
    )

    if fused:
        # the kernel already wrote the final, scaled output + natural-log LSE.
        if return_softmax_lse:
            return out_buf, lse_buf
        return out_buf

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
