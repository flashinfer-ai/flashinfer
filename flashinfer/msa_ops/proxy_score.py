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
from ..utils import is_sm12x_supported

_BLK_KV = 128


@flashinfer_api
def msa_proxy_score(
    q: torch.Tensor,
    k: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    *,
    page_table: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    causal: bool = True,
    max_seqlen_q: Optional[int] = None,
    max_k_tiles: Optional[int] = None,
    output: Optional[torch.Tensor] = None,
    reduce_heads: bool = False,
    q_offset=None,
) -> torch.Tensor:
    """MSA dense proxy pass for SM120/SM121: per-KV-block max attention
    logits (MSA's ``OnlyScore`` mode).

    Computes ``max_score[h, t, q]``, the maximum of the unscaled,
    causally-masked ``Q K^T`` logits over the 128 tokens of KV block ``t``,
    for every query token and query head. The output feeds directly into
    :func:`msa_topk_select`. KV blocks beyond a sequence's valid range, or
    entirely above the causal limit, yield ``-inf``.

    There is no softmax and no V: this is MSA pipeline stage 1.

    Parameters
    ----------
    q : torch.Tensor
        ``(total_q, num_qo_heads, 128)``, bf16 or fp16 (the cheap proxy Q).
    k : torch.Tensor
        Flat ``(total_k, num_kv_heads, 128)`` with ``cu_seqlens_k``, or
        paged ``(num_pages, num_kv_heads, 128, 128)`` with ``page_table`` +
        ``seqused_k``. May be fp8 E4M3 (upconverted in-kernel).
        ``num_qo_heads`` must be a multiple of ``num_kv_heads``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        ``(batch_size + 1,)`` int32 cumulative lengths.
    causal : bool
        Right-aligned causal masking (mask applied *before* the block max,
        matching MSA).
    max_k_tiles : int, optional
        Number of KV-block columns in the output; defaults to the maximum
        ``ceil(seqlen_k / 128)`` over the batch.
    output : torch.Tensor, optional
        Pre-allocated float32 output. Shape is
        ``(num_qo_heads, max_k_tiles, total_q)`` normally, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    reduce_heads : bool
        If ``True``, max-reduce the per-head ``max_score`` over the query-head
        axis and return a single ``(1, max_k_tiles, total_q)`` score, recovering
        the *one selection per query* that the MiniMax-M3 lightning indexer
        produces (its ``block_scores = scores.amax(-1).amax(over index heads)``).
        Use this when the query heads are an indexer's proxy heads that collapse
        to a shared block selection. Defaults to ``False``, the per-head
        ``max_score`` of MSA's canonical *one-proxy-head-per-KV-head* pipeline,
        where each head selects its own blocks.

        The reduction is currently a post-kernel ``amax`` over the per-head
        buffer (the kernel is one CTA per head, so a cross-head epilogue would
        need cross-CTA float atomics); folding it into the kernel is a possible
        future optimization (saves materializing the per-head buffer).

    Returns
    -------
    torch.Tensor
        Float32 ``max_score`` ready for :func:`msa_topk_select`:
        ``(num_qo_heads, max_k_tiles, total_q)``, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    """
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.proxy_score_sm12x import MsaProxyScoreSm12x
    from .sparse_attention import (
        _compile_cache,
        _cutlass_dtype,
        _fake,
        _q_offset_tensor,
    )

    if not is_sm12x_supported(q.device):
        raise RuntimeError("msa_proxy_score requires SM120 or SM121 and CUDA >= 12.8")
    if q.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError(f"q must be bf16 or fp16, got {q.dtype}")
    total_q, num_qo_heads, head_dim = q.shape
    if head_dim != 128:
        raise ValueError(f"head_dim must be 128, got {head_dim}")
    num_kv_heads = k.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    kv_fp8 = k.dtype == torch.float8_e4m3fn
    if not kv_fp8 and k.dtype != q.dtype:
        raise ValueError("k dtype must match q (or be float8_e4m3fn)")
    dev = q.device

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged proxy requires seqused_k")
        if k.ndim != 4 or k.shape[2] != _BLK_KV or k.shape[3] != head_dim:
            raise ValueError(
                f"paged k must be (num_pages, num_kv_heads, {_BLK_KV}, {head_dim})"
            )
        batch_size = seqused_k.numel()
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = seqused_k.to(dev).cumsum(0)
        pt_dev = page_table.contiguous()
    else:
        if cu_seqlens_k is None:
            raise ValueError("flat proxy requires cu_seqlens_k")
        if k.ndim != 3:
            raise ValueError("flat k must be (total_k, num_kv_heads, head_dim)")
        cu_k = cu_seqlens_k.to(dev)
        pt_dev = torch.zeros((1, 1), dtype=torch.int32, device=dev)
        batch_size = cu_k.numel() - 1

    cu_q_dev = cu_seqlens_q.to(dev)
    qoff_dev = _q_offset_tensor(q_offset, cu_q_dev, cu_k, dev)
    if max_seqlen_q is None:
        cu_q_cpu = cu_seqlens_q.cpu()
        max_seqlen_q = int((cu_q_cpu[1:] - cu_q_cpu[:-1]).max().item())
    if max_k_tiles is None:
        cu_k_cpu = cu_k.cpu()
        seqlens_k = cu_k_cpu[1:] - cu_k_cpu[:-1]
        max_k_tiles = int((seqlens_k.max().item() + _BLK_KV - 1) // _BLK_KV)

    per_head_shape = (num_qo_heads, max_k_tiles, total_q)
    final_shape = (1, max_k_tiles, total_q) if reduce_heads else per_head_shape
    if output is not None:
        if output.shape != final_shape:
            raise ValueError(f"output must be {final_shape}")
        if output.dtype != torch.float32:
            raise ValueError("output must be float32")

    # The kernel always writes the per-head buffer; reduce_heads then collapses
    # it. When not reducing and an output was given, that buffer *is* the output.
    if reduce_heads or output is None:
        per_head = torch.empty(per_head_shape, dtype=torch.float32, device=dev)
    else:
        per_head = output

    key = ("proxy", str(q.dtype), causal, paged, kv_fp8)
    compiled = _compile_cache.get(key)
    if compiled is None:
        cdt = _cutlass_dtype(q.dtype)
        kdt = _cutlass_dtype(k.dtype)
        i32 = _cutlass_dtype(torch.int32)
        f32 = _cutlass_dtype(torch.float32)
        s_tq, s_hq, s_tk, s_hkv, s_b1, s_b0, s_pb, s_pm, s_mt = (
            cute.sym_int() for _ in range(9)
        )
        k_shape = (s_tk, s_hkv, _BLK_KV, head_dim) if paged else (s_tk, s_hkv, head_dim)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj = MsaProxyScoreSm12x(
            head_dim=head_dim,
            m_block_size=64,
            n_block_size=_BLK_KV,
            num_threads=128,
            is_causal=causal,
            paged=paged,
            kv_fp8=kv_fp8,
        )
        compiled = cute.compile(
            kernel_obj,
            _fake(cdt, (s_tq, s_hq, head_dim)),
            _fake(kdt, k_shape),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(f32, (s_hq, s_mt, s_tq), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            stream_fake,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    compiled(
        q,
        k,
        pt_dev,
        per_head,
        cu_q_dev,
        cu_k,
        qoff_dev,
        int(max_seqlen_q),
        int(batch_size),
        int(num_qo_heads),
        int(max_k_tiles),
    )

    if not reduce_heads:
        return per_head
    if output is None:
        output = torch.empty(final_shape, dtype=torch.float32, device=dev)
    # max over the query-head axis -> one (block, query) score, M3-indexer style.
    torch.amax(per_head, dim=0, keepdim=True, out=output)
    return output
