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

from typing import Optional, Tuple

import torch

from ..api_logging import flashinfer_api
from ..utils import is_sm12x_supported

_BLK_KV = 128
_SF_VEC_SIZE = 16


def _proxy_split_k(base_ctas: int, max_k_tiles: int, device) -> int:
    """KV-block split factor that fills the GPU when the base proxy grid is small.

    The proxy output is per-(head, kv_block, query), so the kv-block range can be
    split across CTAs with no cross-split reduction (each split writes disjoint
    columns). At long context with low batch the base grid (e.g. B8 q1 -> 64 CTAs)
    is smaller than the SM count, so the kernel starves; splitting the kv blocks
    across more CTAs fills it. Returns 1 (the unsplit schedule) whenever the base
    grid already covers the SMs, so high-batch / prefill is unchanged.
    """
    if base_ctas <= 0 or max_k_tiles <= 1:
        return 1
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    target = 2 * num_sms  # the kernels run >=2 resident CTAs/SM (min_blocks_per_mp)
    if base_ctas >= target:
        return 1
    splits = -(-target // base_ctas)  # ceil, so we actually reach the target
    return max(1, min(splits, max_k_tiles))


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


def quantize_bf16_qk_to_nvfp4(
    x: torch.Tensor,
    global_scale: Optional[float] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """Quantize a proxy Q or K tensor to NVFP4 for :func:`msa_proxy_score_fp4`.

    Helper for tests / benchmarks / standalone callers; a real deployment
    quantizes the index Q/K once upstream and stores them packed. Produces the
    cuBLAS **128x4 tiled** scale-factor layout shared by the MSA stack
    (``msa_proxy_score_fp4`` / kvmajor / decode), so one NVFP4 cache feeds all.

    Parameters
    ----------
    x : torch.Tensor
        ``(total, num_heads, 128)`` bf16/fp16 proxy Q or K.
    global_scale : float, optional
        Per-tensor NVFP4 global scale (the multiplier applied to ``x`` before
        e2m1 rounding). Defaults to ``448 * 6 / amax(x)``, the standard choice
        that maps the tensor's max magnitude to the e4m3*e2m1 range.

    Returns
    -------
    x_fp4 : torch.Tensor
        ``(total, num_heads, 64)`` uint8, two packed e2m1 nibbles per byte.
    x_scale : torch.Tensor
        Flat uint8 e4m3 block scales (one per 16 elements) in the cuBLAS 128x4
        tiled layout, indexed by logical row ``token*num_heads + head``.
    inv_global_scale : float
        ``1 / global_scale``; pass as ``q_global_scale`` / ``k_global_scale`` to
        :func:`msa_proxy_score_fp4` (the kernel multiplies the dequantized
        logits by ``q_global_scale * k_global_scale``).
    """
    from flashinfer import nvfp4_quantize

    if x.ndim != 3 or x.shape[2] != 128:
        raise ValueError(f"x must be (total, num_heads, 128), got {tuple(x.shape)}")
    total, num_heads, head_dim = x.shape
    x2d = x.reshape(-1, head_dim)
    if global_scale is None:
        global_scale = (448.0 * 6.0) / x2d.float().abs().max().clamp_min(1e-12)
    gsf = torch.as_tensor([float(global_scale)], dtype=torch.float32, device=x.device)
    # default sfLayout is the cuBLAS 128x4 tiled layout; SF row = token*num_heads
    # + head (the natural (total, num_heads, d).reshape(-1, d) row order).
    xq, sf = nvfp4_quantize(x2d, gsf, sf_vec_size=_SF_VEC_SIZE)
    x_fp4 = xq.view(torch.uint8).reshape(total, num_heads, head_dim // 2)
    x_scale = sf.view(torch.uint8).reshape(-1)  # flat, 128x4 tiled
    return x_fp4, x_scale, 1.0 / float(global_scale)


@flashinfer_api
def msa_proxy_score_fp4(
    q_fp4: torch.Tensor,
    k_fp4: torch.Tensor,
    q_scale: torch.Tensor,
    k_scale: torch.Tensor,
    q_global_scale: float,
    k_global_scale: float,
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
    """NVFP4 MSA dense proxy pass for SM120/SM121 (the FP4 counterpart of
    :func:`msa_proxy_score`).

    Same contract and output as :func:`msa_proxy_score` — per-KV-block max of
    the unscaled, causally-masked ``Q K^T`` logits — but Q/K arrive pre-quantized
    as packed NVFP4 (``e2m1`` + per-16 ``e4m3`` block scales + per-tensor global
    scales), so the index K is read from HBM at ~4 bits/elem. The full-KV index
    read is the dominant decode-step cost, so this is the bandwidth path that
    matches MSA's deployed ``fp4_indexer_block_scores``; :func:`msa_proxy_score`
    stays as the bf16 precision reference.

    The kernel computes ``Q K^T`` on the SM120 fp4 tensor cores
    (``MmaMXF4NVF4Op``), so numerics equal a torch dequant of the same packed
    inputs (not the bf16 reference, which differs by fp4 rounding). The two global
    scales are folded into the logits as ``q_global_scale * k_global_scale`` before
    the block-max. The 16-head ``q_len<=8`` decode uses a packed schedule that scores
    all 16 heads of a kv_head from one shared index-K read. Both flat and paged K are
    supported.

    Parameters
    ----------
    q_fp4 : torch.Tensor
        ``(total_q, num_qo_heads, 64)`` uint8, packed e2m1 (2 nibbles/byte).
    k_fp4 : torch.Tensor
        Flat ``(total_k, num_kv_heads, 64)`` with ``cu_seqlens_k``, or paged
        ``(num_pages, num_kv_heads, 128, 64)`` with ``page_table`` +
        ``seqused_k``. ``num_qo_heads`` must be a multiple of ``num_kv_heads``.
    q_scale, k_scale : torch.Tensor
        Flat uint8 e4m3 block scales in the cuBLAS 128x4 tiled layout, indexed by
        logical row ``token*num_heads + head`` (paged K: ``(page*num_kv_heads +
        kv_head)*128 + token_in_page``). Produced by
        :func:`quantize_bf16_qk_to_nvfp4`; shared with kvmajor / decode.
    q_global_scale, k_global_scale : float
        Per-tensor inverse global scales (``1 / global_scale``) from
        :func:`quantize_bf16_qk_to_nvfp4`.
    cu_seqlens_q, cu_seqlens_k, causal, max_seqlen_q, max_k_tiles, output,
    reduce_heads, q_offset
        As in :func:`msa_proxy_score`.

    Returns
    -------
    torch.Tensor
        Float32 ``max_score`` for :func:`msa_topk_select`:
        ``(num_qo_heads, max_k_tiles, total_q)``, or
        ``(1, max_k_tiles, total_q)`` when ``reduce_heads=True``.
    """
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.proxy_score_fp4_sm12x import (
        MsaProxyScoreFp4MmaDecodePackedSm12x,
        MsaProxyScoreFp4MmaSm12x,
    )
    from .sparse_attention import (
        _compile_cache,
        _cutlass_dtype,
        _fake,
        _q_offset_tensor,
    )

    if not is_sm12x_supported(q_fp4.device):
        raise RuntimeError(
            "msa_proxy_score_fp4 requires SM120 or SM121 and CUDA >= 12.8"
        )
    if q_fp4.dtype != torch.uint8 or k_fp4.dtype != torch.uint8:
        raise ValueError("q_fp4/k_fp4 must be packed uint8 (e2m1x2)")
    if q_scale.dtype != torch.uint8 or k_scale.dtype != torch.uint8:
        raise ValueError("q_scale/k_scale must be uint8 (e4m3 bytes)")
    hd2 = _BLK_KV // 2  # 64 packed bytes / row
    q_scale = q_scale.reshape(-1)
    k_scale = k_scale.reshape(-1)
    total_q, num_qo_heads, q_last = q_fp4.shape
    if q_last != hd2:
        raise ValueError(f"q_fp4 last dim must be {hd2} (head_dim/2), got {q_last}")
    num_kv_heads = k_fp4.shape[1]
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be a multiple of num_kv_heads")
    dev = q_fp4.device

    paged = page_table is not None
    if paged:
        if seqused_k is None:
            raise ValueError("paged proxy requires seqused_k")
        if k_fp4.ndim != 4 or k_fp4.shape[2] != _BLK_KV or k_fp4.shape[3] != hd2:
            raise ValueError(
                f"paged k_fp4 must be (num_pages, num_kv_heads, {_BLK_KV}, {hd2})"
            )
        batch_size = seqused_k.numel()
        cu_k = torch.zeros(batch_size + 1, dtype=torch.int32, device=dev)
        cu_k[1:] = seqused_k.to(dev).cumsum(0)
        pt_dev = page_table.contiguous()
    else:
        if cu_seqlens_k is None:
            raise ValueError("flat proxy requires cu_seqlens_k")
        if k_fp4.ndim != 3:
            raise ValueError("flat k_fp4 must be (total_k, num_kv_heads, 64)")
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
    if reduce_heads or output is None:
        per_head = torch.empty(per_head_shape, dtype=torch.float32, device=dev)
    else:
        per_head = output

    # The 16-head decode path (q_len<=8) packs 16 heads x 8 q into one 128-row
    # tile so the shared index-K is read once per (batch, kv_head) instead of once
    # per query head (16x less K traffic); outside that regime use the per-(q-tile,
    # head) schedule.
    group_size = num_qo_heads // num_kv_heads
    use_packed = (
        group_size == MsaProxyScoreFp4MmaDecodePackedSm12x._QHEAD_PER_KV
        and max_seqlen_q <= MsaProxyScoreFp4MmaDecodePackedSm12x._PACK_Q_LEN
    )

    # split-K factor (kv-block axis): fill the GPU when the base grid underfills
    # the SMs. The packed path is one CTA per (batch, kv_head); the general path is
    # one CTA per (q-tile, batch, head) with 128-row q-tiles. num_splits passes
    # through to the kernel as a runtime arg, so the compiled module is reused (it
    # is not part of the cache key).
    if use_packed:
        base_ctas = batch_size * num_kv_heads
    else:
        base_ctas = ((max_seqlen_q + 127) // 128) * batch_size * num_qo_heads
    num_splits = _proxy_split_k(base_ctas, int(max_k_tiles), dev)

    key = ("proxy_fp4", causal, paged, use_packed)
    compiled = _compile_cache.get(key)
    if compiled is None:
        u8 = _cutlass_dtype(torch.uint8)
        i32 = _cutlass_dtype(torch.int32)
        f32 = _cutlass_dtype(torch.float32)
        (
            s_tq,
            s_hq,
            s_tk,
            s_hkv,
            s_b1,
            s_b0,
            s_pb,
            s_pm,
            s_mt,
            s_qsf,
            s_ksf,
        ) = (cute.sym_int() for _ in range(11))
        if paged:
            k_shape: tuple = (s_tk, s_hkv, _BLK_KV, hd2)
        else:
            k_shape = (s_tk, s_hkv, hd2)
        stream_fake = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
        kernel_obj: "MsaProxyScoreFp4MmaSm12x"
        if use_packed:
            kernel_obj = MsaProxyScoreFp4MmaDecodePackedSm12x(
                head_dim=_BLK_KV,
                is_causal=causal,
                paged=paged,
            )
        else:
            kernel_obj = MsaProxyScoreFp4MmaSm12x(
                head_dim=_BLK_KV,
                is_causal=causal,
                paged=paged,
            )
        compiled = cute.compile(
            kernel_obj,
            _fake(u8, (s_tq, s_hq, hd2), align=4),
            _fake(u8, k_shape, align=4),
            _fake(u8, (s_qsf,), align=4),
            _fake(u8, (s_ksf,), align=4),
            _fake(i32, (s_pb, s_pm), align=4),
            _fake(f32, (s_hq, s_mt, s_tq), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b1,), align=4),
            _fake(i32, (s_b0,), align=4),
            cutlass.Float32(1.0),
            cutlass.Float32(1.0),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),
            cutlass.Int32(1),  # num_splits
            stream_fake,
            options="--enable-tvm-ffi",
        )
        _compile_cache[key] = compiled

    compiled(
        q_fp4,
        k_fp4,
        q_scale,
        k_scale,
        pt_dev,
        per_head,
        cu_q_dev,
        cu_k,
        qoff_dev,
        float(q_global_scale),
        float(k_global_scale),
        int(max_seqlen_q),
        int(batch_size),
        int(num_qo_heads),
        int(max_k_tiles),
        int(num_splits),
    )

    if not reduce_heads:
        return per_head
    if output is None:
        output = torch.empty(final_shape, dtype=torch.float32, device=dev)
    torch.amax(per_head, dim=0, keepdim=True, out=output)
    return output
