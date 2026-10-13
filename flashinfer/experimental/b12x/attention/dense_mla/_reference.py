"""High-precision paged dense-MLA oracle."""

from __future__ import annotations

import math

import torch

K3_ABSORBED_DIM = 576
K3_VALUE_DIM = 512
K3_RAW_QK_DIM = 192
K3_SM_SCALE = 1.0 / math.sqrt(K3_RAW_QK_DIM)


def _cache_rank3(cache: torch.Tensor, *, head_dim: int) -> torch.Tensor:
    if cache.ndim == 4:
        if int(cache.shape[2]) != 1:
            raise ValueError("rank-4 dense MLA cache must have one KV head")
        cache = cache[:, :, 0, :]
    if cache.ndim != 3:
        raise ValueError(
            "cache must be [pages,page_size,physical_record_width] or its "
            "singleton-head rank-4 form"
        )
    if int(cache.shape[-1]) < head_dim:
        raise ValueError(
            f"dense MLA cache record must contain at least {head_dim} elements"
        )
    return cache


def _scalar(value: torch.Tensor | float | None, *, name: str) -> float:
    if value is None:
        return 1.0
    if isinstance(value, torch.Tensor):
        if value.dtype != torch.float32 or value.numel() != 1:
            raise ValueError(f"{name} must be a scalar float32 tensor")
        return float(value.detach().cpu().item())
    return float(value)


def dense_mla_reference(
    q: torch.Tensor,
    kv_cache: torch.Tensor,
    page_table: torch.Tensor,
    cache_seqlens: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    kv_scale: torch.Tensor | float | None = None,
    q_scale: torch.Tensor | float | None = None,
    sm_scale: float = K3_SM_SCALE,
    v_head_dim: int | None = None,
    window_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute right-aligned causal dense MLA with an optional local window.

    The cache already contains the current query chunk.  Query row ``i`` in a
    request of length ``Q`` therefore sees keys through
    ``cache_len - Q + i`` (inclusive).
    """
    if q.ndim != 3:
        raise ValueError("q must have shape [total_q, heads, head_dim]")
    head_dim = int(q.shape[-1])
    default_value_dims = {K3_ABSORBED_DIM: K3_VALUE_DIM, 1088: 1024}
    if head_dim not in default_value_dims:
        raise ValueError(f"unsupported dense MLA query width {head_dim}")
    if v_head_dim is None:
        v_head_dim = default_value_dims[head_dim]
    v_head_dim = int(v_head_dim)
    if not 0 < v_head_dim <= head_dim:
        raise ValueError("v_head_dim must be in [1, head_dim]")
    if window_size is not None and int(window_size) <= 0:
        raise ValueError("window_size must be positive or None")
    cache = _cache_rank3(kv_cache, head_dim=head_dim)
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("q must be BF16 or E4M3")
    if cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn):
        raise TypeError("kv_cache must be BF16 or E4M3")
    if page_table.ndim != 2 or page_table.dtype != torch.int32:
        raise TypeError("page_table must be rank-2 torch.int32")
    if cache_seqlens.ndim != 1 or cache_seqlens.dtype != torch.int32:
        raise TypeError("cache_seqlens must be rank-1 torch.int32")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be rank-1 torch.int32")
    batch = int(page_table.shape[0])
    if tuple(cache_seqlens.shape) != (batch,):
        raise ValueError("cache_seqlens shape must match page_table batch")
    if tuple(cu_seqlens_q.shape) != (batch + 1,):
        raise ValueError("cu_seqlens_q shape must be [batch + 1]")
    if any(
        t.device != q.device for t in (cache, page_table, cache_seqlens, cu_seqlens_q)
    ):
        raise ValueError("dense MLA reference tensors must be on one device")

    kv_mul = _scalar(kv_scale, name="kv_scale")
    q_mul = _scalar(q_scale, name="q_scale")
    if cache.dtype == torch.float8_e4m3fn and kv_scale is None:
        raise ValueError("E4M3 kv_cache requires kv_scale")
    if q.dtype == torch.float8_e4m3fn and q_scale is None:
        raise ValueError("E4M3 q requires q_scale")
    if cache.dtype == torch.bfloat16 and (kv_scale is not None or q_scale is not None):
        raise ValueError("BF16 dense MLA does not accept quantization scales")
    if q.dtype != cache.dtype and not (
        q.dtype == torch.bfloat16 and cache.dtype == torch.float8_e4m3fn
    ):
        raise TypeError(
            "dense MLA reference supports matching query/cache dtypes or a "
            "BF16 query with an E4M3 cache"
        )
    if q.dtype == torch.bfloat16 and cache.dtype == torch.float8_e4m3fn:
        if q_scale is None:
            raise ValueError("BF16 query quantization for E4M3 cache requires q_scale")
        quantized_q = (q.float() / q_mul).to(torch.float8_e4m3fn)
        q_f32 = quantized_q.float() * q_mul
    else:
        q_f32 = q.float() * q_mul

    cu_host = [int(v) for v in cu_seqlens_q.detach().cpu().tolist()]
    lens_host = [int(v) for v in cache_seqlens.detach().cpu().tolist()]
    if cu_host[0] != 0 or cu_host[-1] != int(q.shape[0]):
        raise ValueError("cu_seqlens_q must span exactly q.shape[0] rows")
    page_size = int(cache.shape[1])
    output = torch.empty(
        (int(q.shape[0]), int(q.shape[1]), v_head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    lse = torch.empty(
        (int(q.shape[0]), int(q.shape[1])),
        dtype=torch.float32,
        device=q.device,
    )

    for request in range(batch):
        q_begin, q_end = cu_host[request], cu_host[request + 1]
        q_len = q_end - q_begin
        kv_len = lens_host[request]
        if q_len <= 0:
            raise ValueError("every request must contain at least one query row")
        if kv_len < q_len:
            raise ValueError("cache length must include the complete query chunk")
        pages_needed = (kv_len + page_size - 1) // page_size
        if pages_needed > int(page_table.shape[1]):
            raise ValueError("page_table is too narrow for cache_seqlens")
        physical_pages = page_table[request, :pages_needed].to(torch.long)
        records = cache.index_select(0, physical_pages).reshape(
            -1, int(cache.shape[-1])
        )
        records = records[:, :head_dim]
        records = records[:kv_len].float() * kv_mul

        q_rows = q_f32[q_begin:q_end]
        for local_q in range(q_len):
            visible = kv_len - q_len + local_q + 1
            visible_begin = (
                0 if window_size is None else max(0, visible - int(window_size))
            )
            key = records[visible_begin:visible]
            logits = torch.einsum("hd,kd->hk", q_rows[local_q], key)
            logits = logits * float(sm_scale)
            probs = torch.softmax(logits, dim=-1)
            output[q_begin + local_q] = torch.einsum(
                "hk,kd->hd", probs, key[:, :v_head_dim]
            )
            lse[q_begin + local_q] = torch.logsumexp(logits, dim=-1)

    return output.to(torch.bfloat16), lse


__all__ = [
    "K3_ABSORBED_DIM",
    "K3_RAW_QK_DIM",
    "K3_SM_SCALE",
    "K3_VALUE_DIM",
    "dense_mla_reference",
]
