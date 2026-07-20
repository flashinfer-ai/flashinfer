# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/mla/compressed_reference.py @ 313da89d (2026-05-26) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""Reference helpers for compressed sparse MLA attention.

The compressed sparse MLA KV page stores each token as a 448-dimension FP8 noPE
vector, a 64-dimension BF16 RoPE vector, and seven UE8M0 scale bytes for the
noPE groups.  Within each page the token payloads are packed first, followed by
the per-token scale region:

    [page_size * 576 payload bytes][page_size * 8 scale bytes][padding]

The padding rounds page byte size up to a 576-byte multiple, matching the
SGLang memory-pool contract.
"""

from __future__ import annotations

import math

import torch


COMPRESSED_MLA_NOPE_DIM = 448
COMPRESSED_MLA_ROPE_DIM = 64
COMPRESSED_MLA_HEAD_DIM = COMPRESSED_MLA_NOPE_DIM + COMPRESSED_MLA_ROPE_DIM
COMPRESSED_MLA_NOPE_GROUP_SIZE = 64
COMPRESSED_MLA_NOPE_GROUPS = COMPRESSED_MLA_NOPE_DIM // COMPRESSED_MLA_NOPE_GROUP_SIZE
COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN = (
    COMPRESSED_MLA_NOPE_DIM + COMPRESSED_MLA_ROPE_DIM * 2
)
COMPRESSED_MLA_SCALE_BYTES_PER_TOKEN = 8
COMPRESSED_MLA_BYTES_PER_TOKEN = (
    COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN + COMPRESSED_MLA_SCALE_BYTES_PER_TOKEN
)
COMPRESSED_MLA_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
COMPRESSED_MLA_UE8M0_BIAS = 127

COMPRESSED_MLA_LOCAL_Q_HEADS_TP2 = 32
COMPRESSED_MLA_TOTAL_Q_HEADS = 64
COMPRESSED_MLA_KV_HEADS = 1
# SGLang's DSV4 backend hard-codes a 256-token physical KV page. Keep that
# distinct from the 128-token sliding SWA window.
COMPRESSED_MLA_DSV4_PAGE_SIZE = 256
COMPRESSED_MLA_C4_PAGE_SIZE = COMPRESSED_MLA_DSV4_PAGE_SIZE // 4
COMPRESSED_MLA_C128_PAGE_SIZE = COMPRESSED_MLA_DSV4_PAGE_SIZE // 128
COMPRESSED_MLA_SWA_TOKENS = 128
COMPRESSED_MLA_INDEX_TOPK = 512


def compressed_mla_page_nbytes(page_size: int) -> int:
    """Return the padded byte count per compressed-MLA KV page."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    unpadded = page_size * COMPRESSED_MLA_BYTES_PER_TOKEN
    return (
        math.ceil(unpadded / COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN)
        * COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN
    )


def compressed_mla_scale_region_offset(page_size: int) -> int:
    """Return the byte offset at which UE8M0 scales start inside a page."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    return page_size * COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN


def ue8m0_to_float(scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 scale bytes using the standard exponent bias."""

    exponent = scale_ue8m0.to(torch.int32) - COMPRESSED_MLA_UE8M0_BIAS
    ones = torch.ones_like(exponent, dtype=torch.float32)
    return torch.ldexp(ones, exponent)


def _float_to_ue8m0_scale(max_abs: torch.Tensor) -> torch.Tensor:
    safe = torch.where(
        max_abs > 0, max_abs / COMPRESSED_MLA_FP8_MAX, torch.ones_like(max_abs)
    )
    exponent = torch.ceil(torch.log2(safe)).to(torch.int32)
    exponent = torch.clamp(
        exponent, -COMPRESSED_MLA_UE8M0_BIAS, 255 - COMPRESSED_MLA_UE8M0_BIAS
    )
    exponent = torch.where(max_abs > 0, exponent, torch.zeros_like(exponent))
    return (exponent + COMPRESSED_MLA_UE8M0_BIAS).to(torch.uint8)


def quantize_compressed_mla_nope_reference(
    k_nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize noPE rows into E4M3 FP8 plus seven UE8M0 scale bytes."""

    if k_nope.shape[-1] != COMPRESSED_MLA_NOPE_DIM:
        raise ValueError(
            f"k_nope last dim must be {COMPRESSED_MLA_NOPE_DIM}, got {k_nope.shape[-1]}"
        )
    grouped = k_nope.float().reshape(
        *k_nope.shape[:-1], COMPRESSED_MLA_NOPE_GROUPS, COMPRESSED_MLA_NOPE_GROUP_SIZE
    )
    max_abs = grouped.abs().amax(dim=-1)
    scale_ue8m0 = _float_to_ue8m0_scale(max_abs)
    scale = ue8m0_to_float(scale_ue8m0).unsqueeze(-1)
    scaled = torch.clamp(
        grouped / scale, -COMPRESSED_MLA_FP8_MAX, COMPRESSED_MLA_FP8_MAX
    )
    quantized = scaled.reshape(k_nope.shape).to(torch.float8_e4m3fn)
    return quantized, scale_ue8m0


def pack_compressed_mla_kv_cache_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    page_size: int,
    num_pages: int | None = None,
) -> torch.Tensor:
    """Pack K/V rows into the flat compressed-MLA uint8 page-buffer layout."""

    if k_nope.ndim != 2 or k_nope.shape[-1] != COMPRESSED_MLA_NOPE_DIM:
        raise ValueError(
            f"k_nope must have shape [tokens, {COMPRESSED_MLA_NOPE_DIM}], got {tuple(k_nope.shape)}"
        )
    if k_rope.ndim != 2 or k_rope.shape[-1] != COMPRESSED_MLA_ROPE_DIM:
        raise ValueError(
            f"k_rope must have shape [tokens, {COMPRESSED_MLA_ROPE_DIM}], got {tuple(k_rope.shape)}"
        )
    if k_nope.shape[0] != k_rope.shape[0]:
        raise ValueError("k_nope and k_rope must have the same token count")

    n_tokens = int(k_nope.shape[0])
    min_pages = math.ceil(n_tokens / page_size)
    if num_pages is None:
        num_pages = min_pages
    if num_pages < min_pages:
        raise ValueError(
            f"num_pages {num_pages} cannot hold {n_tokens} tokens at page_size {page_size}"
        )

    page_nbytes = compressed_mla_page_nbytes(page_size)
    scale_offset = compressed_mla_scale_region_offset(page_size)
    cache = torch.zeros(
        (num_pages, page_nbytes), dtype=torch.uint8, device=k_nope.device
    )
    if n_tokens == 0:
        return cache

    k_nope_fp8, scale_ue8m0 = quantize_compressed_mla_nope_reference(k_nope)
    payload = torch.as_strided(
        cache,
        (num_pages, page_size, COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN),
        (page_nbytes, COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN, 1),
    )
    scales = torch.as_strided(
        cache,
        (num_pages, page_size, COMPRESSED_MLA_SCALE_BYTES_PER_TOKEN),
        (page_nbytes, COMPRESSED_MLA_SCALE_BYTES_PER_TOKEN, 1),
        storage_offset=scale_offset,
    )
    token_ids = torch.arange(n_tokens, dtype=torch.int64, device=k_nope.device)
    pages = token_ids // page_size
    token_offsets = token_ids % page_size

    payload[pages, token_offsets, :COMPRESSED_MLA_NOPE_DIM] = k_nope_fp8.view(
        torch.uint8
    ).view(n_tokens, COMPRESSED_MLA_NOPE_DIM)
    payload[
        pages,
        token_offsets,
        COMPRESSED_MLA_NOPE_DIM:COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN,
    ] = (
        k_rope.to(torch.bfloat16)
        .view(torch.uint8)
        .view(n_tokens, COMPRESSED_MLA_ROPE_DIM * 2)
    )
    scales[pages, token_offsets, :COMPRESSED_MLA_NOPE_GROUPS] = scale_ue8m0.reshape(
        n_tokens, COMPRESSED_MLA_NOPE_GROUPS
    )
    return cache


def unpack_compressed_mla_kv_cache_reference(
    kv_cache: torch.Tensor,
    *,
    page_size: int,
    n_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Unpack every token from a flat compressed-MLA KV page buffer."""

    _validate_flat_cache(kv_cache, page_size=page_size)
    num_pages = int(kv_cache.shape[0])
    total_tokens = num_pages * page_size
    if n_tokens is None:
        n_tokens = total_tokens
    if n_tokens < 0 or n_tokens > total_tokens:
        raise ValueError(f"n_tokens must be in [0, {total_tokens}], got {n_tokens}")
    if n_tokens == 0:
        empty_nope = torch.empty(
            (0, COMPRESSED_MLA_NOPE_DIM), dtype=torch.float32, device=kv_cache.device
        )
        empty_rope = torch.empty(
            (0, COMPRESSED_MLA_ROPE_DIM), dtype=torch.float32, device=kv_cache.device
        )
        return empty_nope, empty_rope

    indices = torch.arange(n_tokens, dtype=torch.int64, device=kv_cache.device)
    return gather_compressed_mla_kv_cache_reference(
        kv_cache, indices, page_size=page_size
    )


def gather_compressed_mla_kv_cache_reference(
    kv_cache: torch.Tensor,
    indices: torch.Tensor,
    *,
    page_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Gather full 512-d K/V rows from flat token indices."""

    _validate_flat_cache(kv_cache, page_size=page_size)
    if indices.ndim != 1:
        raise ValueError(f"indices must be rank-1, got {tuple(indices.shape)}")
    if indices.numel() == 0:
        empty = torch.empty(
            (0, COMPRESSED_MLA_HEAD_DIM), dtype=torch.float32, device=kv_cache.device
        )
        return empty, empty

    page_nbytes = compressed_mla_page_nbytes(page_size)
    scale_offset = compressed_mla_scale_region_offset(page_size)
    flat_cache = kv_cache.reshape(-1)
    idx = indices.to(torch.int64)
    if bool((idx < 0).any()):
        raise ValueError("indices must be non-negative")
    pages = idx // page_size
    token_offsets = idx % page_size
    if bool((pages >= kv_cache.shape[0]).any()):
        raise ValueError("indices exceed kv_cache page capacity")

    token_base = (
        pages * page_nbytes + token_offsets * COMPRESSED_MLA_NOPE_ROPE_BYTES_PER_TOKEN
    )
    nope_offsets = token_base[:, None] + torch.arange(
        COMPRESSED_MLA_NOPE_DIM, device=kv_cache.device, dtype=torch.int64
    )
    rope_byte_offsets = (
        token_base[:, None]
        + COMPRESSED_MLA_NOPE_DIM
        + torch.arange(
            COMPRESSED_MLA_ROPE_DIM * 2, device=kv_cache.device, dtype=torch.int64
        )
    )
    scale_offsets = (
        pages[:, None] * page_nbytes
        + scale_offset
        + token_offsets[:, None] * COMPRESSED_MLA_SCALE_BYTES_PER_TOKEN
        + torch.arange(
            COMPRESSED_MLA_NOPE_GROUPS, device=kv_cache.device, dtype=torch.int64
        )
    )

    nope_fp8 = (
        flat_cache[nope_offsets]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .view(-1, COMPRESSED_MLA_NOPE_DIM)
    )
    scale = ue8m0_to_float(flat_cache[scale_offsets]).view(
        -1, COMPRESSED_MLA_NOPE_GROUPS, 1
    )
    nope = (
        nope_fp8.to(torch.float32).view(
            -1, COMPRESSED_MLA_NOPE_GROUPS, COMPRESSED_MLA_NOPE_GROUP_SIZE
        )
        * scale
    ).reshape(-1, COMPRESSED_MLA_NOPE_DIM)
    rope = (
        flat_cache[rope_byte_offsets]
        .contiguous()
        .view(torch.bfloat16)
        .view(-1, COMPRESSED_MLA_ROPE_DIM)
        .to(torch.float32)
    )
    full = torch.cat((nope, rope), dim=-1)
    return full, full


def compressed_sparse_mla_reference(
    q: torch.Tensor,
    swa_k_cache: torch.Tensor,
    swa_indices: torch.Tensor,
    swa_topk_lengths: torch.Tensor,
    *,
    sm_scale: float,
    attn_sink: torch.Tensor | None = None,
    extra_k_cache: torch.Tensor | None = None,
    extra_indices: torch.Tensor | None = None,
    extra_topk_lengths: torch.Tensor | None = None,
    swa_page_size: int = COMPRESSED_MLA_DSV4_PAGE_SIZE,
    extra_page_size: int | None = None,
    return_lse: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Reference sparse MLA attention over SWA plus optional C4/C128 stream."""

    q3 = _normalize_q(q)
    rows, heads, _ = q3.shape
    swa_indices_2d = _normalize_indices(swa_indices, name="swa_indices")
    if swa_indices_2d.shape[0] != rows:
        raise ValueError("swa_indices row count must match q")
    if swa_topk_lengths.shape != (rows,):
        raise ValueError(
            f"swa_topk_lengths must have shape [{rows}], got {tuple(swa_topk_lengths.shape)}"
        )
    _validate_flat_cache(swa_k_cache, page_size=swa_page_size)

    has_extra = (
        extra_k_cache is not None
        or extra_indices is not None
        or extra_topk_lengths is not None
    )
    if has_extra:
        if extra_k_cache is None or extra_indices is None or extra_topk_lengths is None:
            raise ValueError(
                "extra_k_cache, extra_indices, and extra_topk_lengths must be provided together"
            )
        if extra_page_size is None:
            raise ValueError(
                "extra_page_size is required when extra_k_cache is provided"
            )
        extra_indices_2d = _normalize_indices(extra_indices, name="extra_indices")
        if extra_indices_2d.shape[0] != rows:
            raise ValueError("extra_indices row count must match q")
        if extra_topk_lengths.shape != (rows,):
            raise ValueError(
                f"extra_topk_lengths must have shape [{rows}], got {tuple(extra_topk_lengths.shape)}"
            )
        _validate_flat_cache(extra_k_cache, page_size=extra_page_size)
    else:
        extra_indices_2d = None

    if attn_sink is not None and attn_sink.shape != (heads,):
        raise ValueError(
            f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}"
        )

    out = torch.empty(
        (rows, heads, COMPRESSED_MLA_HEAD_DIM), dtype=torch.float32, device=q3.device
    )
    lse = torch.empty((rows, heads), dtype=torch.float32, device=q3.device)
    q_f32 = q3.float()

    for row in range(rows):
        swa_len = _valid_prefix_length(
            swa_topk_lengths[row], swa_indices_2d[row], swa_indices_2d.shape[1]
        )
        if (
            has_extra
            and extra_indices_2d is not None
            and extra_topk_lengths is not None
        ):
            extra_len = _valid_prefix_length(
                extra_topk_lengths[row],
                extra_indices_2d[row],
                extra_indices_2d.shape[1],
            )
            if extra_len:
                assert extra_k_cache is not None
                assert extra_page_size is not None
                extra_k, extra_v = gather_compressed_mla_kv_cache_reference(
                    extra_k_cache,
                    extra_indices_2d[row, :extra_len],
                    page_size=extra_page_size,
                )
            else:
                extra_k = torch.empty(
                    (0, COMPRESSED_MLA_HEAD_DIM), dtype=torch.float32, device=q3.device
                )
                extra_v = extra_k
        else:
            extra_len = 0
            extra_k = torch.empty(
                (0, COMPRESSED_MLA_HEAD_DIM), dtype=torch.float32, device=q3.device
            )
            extra_v = extra_k

        if swa_len:
            swa_k, swa_v = gather_compressed_mla_kv_cache_reference(
                swa_k_cache, swa_indices_2d[row, :swa_len], page_size=swa_page_size
            )
        else:
            swa_k = torch.empty(
                (0, COMPRESSED_MLA_HEAD_DIM), dtype=torch.float32, device=q3.device
            )
            swa_v = swa_k

        if extra_len:
            k = torch.cat((swa_k, extra_k), dim=0)
            v = torch.cat((swa_v, extra_v), dim=0)
        else:
            k = swa_k
            v = swa_v

        if k.shape[0] == 0:
            if attn_sink is None:
                out[row].zero_()
                lse[row].fill_(-float("inf"))
            else:
                out[row].zero_()
                lse[row] = attn_sink.float()
            continue

        scores = torch.matmul(q_f32[row], k.t()) * float(sm_scale)
        if attn_sink is None:
            row_m = scores.max(dim=-1).values
            weights = torch.exp(scores - row_m[:, None])
            denom = weights.sum(dim=-1)
        else:
            sink = attn_sink.float()
            row_m = torch.maximum(scores.max(dim=-1).values, sink)
            weights = torch.exp(scores - row_m[:, None])
            denom = weights.sum(dim=-1) + torch.exp(sink - row_m)
        out[row] = torch.matmul(weights, v) / denom[:, None]
        lse[row] = row_m + torch.log(denom)

    result = out.to(q3.dtype)
    if return_lse:
        return result, lse
    return result


def _bounded_length(length: torch.Tensor, width: int) -> int:
    value = int(length.item())
    if value < 0:
        raise ValueError(f"topk length must be non-negative, got {value}")
    return min(value, width)


def _valid_prefix_length(
    length: torch.Tensor, indices: torch.Tensor, width: int
) -> int:
    limit = _bounded_length(length, width)
    if limit == 0:
        return 0
    active = indices[:limit].to(torch.int64)
    negative = torch.nonzero(active < 0, as_tuple=False)
    if negative.numel() == 0:
        return limit
    return int(negative[0, 0].item())


def _normalize_q(q: torch.Tensor) -> torch.Tensor:
    if q.ndim == 4 and q.shape[1] == 1:
        q = q[:, 0]
    if q.ndim != 3 or q.shape[-1] != COMPRESSED_MLA_HEAD_DIM:
        raise ValueError(
            f"q must have shape [rows, heads, {COMPRESSED_MLA_HEAD_DIM}], got {tuple(q.shape)}"
        )
    return q


def _normalize_indices(indices: torch.Tensor, *, name: str) -> torch.Tensor:
    if indices.ndim == 3 and indices.shape[1] == 1:
        indices = indices[:, 0]
    if indices.ndim != 2:
        raise ValueError(
            f"{name} must have shape [rows, width] or [rows, 1, width], got {tuple(indices.shape)}"
        )
    return indices


def _validate_flat_cache(kv_cache: torch.Tensor, *, page_size: int) -> None:
    if kv_cache.ndim != 2:
        raise ValueError(
            f"kv_cache must be a flat [pages, page_nbytes] uint8 buffer, got {tuple(kv_cache.shape)}"
        )
    if kv_cache.dtype is not torch.uint8:
        raise TypeError(f"kv_cache must be uint8, got {kv_cache.dtype}")
    expected = compressed_mla_page_nbytes(page_size)
    if kv_cache.shape[1] != expected:
        raise ValueError(
            f"kv_cache page byte width must be {expected} for page_size {page_size}, got {kv_cache.shape[1]}"
        )
