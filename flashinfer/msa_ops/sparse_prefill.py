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
from ._common import (
    _BLK_KV,
    _compile_cache,
    _cutlass_dtype,
    _fake,
    _q_offset_tensor,
)


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
    Query tokens are processed in tiles: each tile runs one online softmax over
    the *union* of the blocks its tokens selected, writing the final output
    directly. The union metadata is built internally from ``q2k_indices``.

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
    # The union builder indexes q2k_indices as a flat contiguous buffer, so reject
    # a strided view (e.g. a bare permute of msa_topk_select's output).
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
    return _msa_sparse_prefill(
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


# The union membership mask is int32, so at most 32 query tokens may share a tile.
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


def _get_compiled_union(topk: int, tokens_per_tile: int):
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.sparse_prefill_sm12x import BuildUnionMetaSm12x

    key = ("union_meta", topk, tokens_per_tile)
    compiled = _compile_cache.get(key)
    if compiled is not None:
        return compiled

    def fake_i32(ndim):
        return _fake(cutlass.Int32, tuple(cute.sym_int() for _ in range(ndim)), align=4)

    kernel_obj = BuildUnionMetaSm12x(topk=topk, tokens_per_tile=tokens_per_tile)
    compiled = cute.compile(
        kernel_obj,
        fake_i32(3),  # q2k
        fake_i32(1),  # tile_batch
        fake_i32(1),  # tile_t
        fake_i32(1),  # tile_qbase
        fake_i32(1),  # tile_ntok
        fake_i32(2),  # union_blocks (out)
        fake_i32(2),  # union_masks (out)
        fake_i32(1),  # union_count (out)
        fake_i32(2),  # work_meta (out)
        cutlass.Int32(1),  # H
        cutlass.Int32(1),  # total_tiles
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )
    _compile_cache[key] = compiled
    return compiled


def _build_union_metadata_device(
    q2k_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    tokens_per_tile: int,
    topk: int,
):
    """On-device, CUDA-graph-capturable union-metadata builder. Returns
    per-(batch, q-tile, kv-head) ``(union_blocks, union_masks, union_count,
    work_meta, work_items)``; work-item order is arbitrary — the forward
    scatters by ``work_meta``."""
    num_kv_heads, total_q, topk_in = q2k_indices.shape
    if topk_in != topk:
        raise ValueError(f"q2k_indices topk {topk_in} != {topk}")
    dev = q2k_indices.device
    max_union = tokens_per_tile * topk
    tpt = tokens_per_tile
    batch_size = cu_seqlens_q.shape[0] - 1

    # Static upper bound on the (batch, q-tile) count, from shapes only so it is
    # capture-safe: sum_b ceil(seqlen_b / tpt) <= ceil(total_q / tpt) + batch_size.
    # The extra slots stay empty.
    total_tiles = (total_q + tpt - 1) // tpt + batch_size
    work_items = total_tiles * num_kv_heads

    if work_items == 0:
        zeros2 = torch.empty((0, max_union), dtype=torch.int32, device=dev)
        return (
            zeros2,
            zeros2.clone(),
            torch.empty((0,), dtype=torch.int32, device=dev),
            torch.empty((0, 3), dtype=torch.int32, device=dev),
            0,
        )

    # Per-tile geometry, computed on device (no cu_seqlens host copy, so it is
    # capture-safe): map each flat tile index to its batch via searchsorted over
    # the per-batch tile cumsum. Padding indices clamp to the last batch and get
    # tile_ntok = 0.
    cu = cu_seqlens_q.to(dtype=torch.int64)
    seqlens = cu[1:] - cu[:-1]
    ntiles = (seqlens + tpt - 1) // tpt
    offsets = torch.cumsum(ntiles, 0)  # exclusive-end tile index per batch
    t = torch.arange(total_tiles, dtype=torch.int64, device=dev)
    tile_batch = torch.clamp(
        torch.searchsorted(offsets, t, right=True), max=batch_size - 1
    )
    tile_t = t - (offsets - ntiles)[tile_batch]  # within-batch tile index
    tile_qbase = cu[:-1][tile_batch] + tile_t * tpt
    tile_ntok = torch.clamp(seqlens[tile_batch] - tile_t * tpt, min=0, max=tpt)

    def _to_dev(t: torch.Tensor) -> torch.Tensor:
        return t.to(dtype=torch.int32, device=dev)

    union_blocks = torch.empty((work_items, max_union), dtype=torch.int32, device=dev)
    union_masks = torch.empty((work_items, max_union), dtype=torch.int32, device=dev)
    union_count = torch.empty((work_items,), dtype=torch.int32, device=dev)
    work_meta = torch.empty((work_items, 3), dtype=torch.int32, device=dev)

    compiled = _get_compiled_union(topk, tpt)
    compiled(
        q2k_indices,
        _to_dev(tile_batch),
        _to_dev(tile_t),
        _to_dev(tile_qbase),
        _to_dev(tile_ntok),
        union_blocks,
        union_masks,
        union_count,
        work_meta,
        int(num_kv_heads),
        int(total_tiles),
    )
    return union_blocks, union_masks, union_count, work_meta, work_items


def _msa_sparse_prefill(
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
    :class:`...cute_dsl.sparse_prefill_sm12x.SparsePrefillSm12x`."""
    import cutlass
    import cutlass.cute as cute

    from .cute_dsl.sparse_prefill_sm12x import SparsePrefillSm12x

    total_q, num_qo_heads, head_dim = q.shape
    dev = q.device
    paged = page_table is not None
    # The page table is read only on the paged path; a (1, 1) dummy keeps the
    # call signature uniform for the flat kernel.
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
    # V's global scale is applied to the output here, since prefill has no combine
    # pass to fold it into; K's was pre-folded into softmax_scale by the caller.
    out_scale = float(v_global_scale) if v_global_scale is not None else 1.0
    m_block, num_threads = _union_tile_config(group_size)
    tokens_per_tile = m_block // group_size

    ub, um, uc, wm, n = _build_union_metadata_device(
        q2k_indices, cu_q_dev, tokens_per_tile, topk
    )
    # Outputs default to 0 / -inf so query tiles that emit no work item (or rows a
    # tile masks out entirely) read back as a zero output with -inf LSE.
    out = torch.zeros((total_q, num_qo_heads, head_dim), dtype=q.dtype, device=dev)
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
        "sparse_prefill",
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
        kernel_obj = SparsePrefillSm12x(
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

    # Device-native fill (no host->device copy) so the single-call path stays
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
    # The kernel writes the log2-domain LSE as (Hq, total_q); the public contract
    # is natural-log (total_q, num_qo_heads).
    if return_temperature_lse:
        lse_out = (lse2 * _LN2).t().contiguous()
        lse_t_out = (lse2_t * _LN2).t().contiguous()
        return out, lse_out, lse_t_out
    if return_softmax_lse:
        lse_out = (lse2 * _LN2).t().contiguous()
        return out, lse_out
    return out
