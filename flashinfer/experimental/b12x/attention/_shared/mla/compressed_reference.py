"""Reference helpers for compressed sparse MLA attention.

The compressed sparse MLA KV page stores each token as a 448-dimension FP8 noPE
vector, a 64-dimension BF16 RoPE vector, and seven UE8M0 scale bytes for the
noPE groups.  Within each page the token payloads are packed first, followed by
the per-token scale region:

    [page_size * 576 payload bytes][page_size * 8 scale bytes][optional padding]

SGLang rounds the page byte size up to a 576-byte multiple. vLLM may instead
use a contiguous allocation whose page stride ends immediately after the scale
region. Both layouts contain the same compressed-MLA payload.
"""

from __future__ import annotations

import math

import torch


COMPRESSED_SPARSE_MLA_NOPE_DIM = 448
COMPRESSED_SPARSE_MLA_ROPE_DIM = 64
COMPRESSED_SPARSE_MLA_HEAD_DIM = (
    COMPRESSED_SPARSE_MLA_NOPE_DIM + COMPRESSED_SPARSE_MLA_ROPE_DIM
)
COMPRESSED_SPARSE_MLA_NOPE_GROUP_SIZE = 64
COMPRESSED_SPARSE_MLA_NOPE_GROUPS = (
    COMPRESSED_SPARSE_MLA_NOPE_DIM // COMPRESSED_SPARSE_MLA_NOPE_GROUP_SIZE
)
COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN = (
    COMPRESSED_SPARSE_MLA_NOPE_DIM + COMPRESSED_SPARSE_MLA_ROPE_DIM * 2
)
COMPRESSED_SPARSE_MLA_SCALE_BYTES_PER_TOKEN = 8
COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN = (
    COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN
    + COMPRESSED_SPARSE_MLA_SCALE_BYTES_PER_TOKEN
)
COMPRESSED_SPARSE_MLA_FP8_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
COMPRESSED_SPARSE_MLA_UE8M0_BIAS = 127

COMPRESSED_SPARSE_MLA_LOCAL_Q_HEADS_TP2 = 32
COMPRESSED_SPARSE_MLA_TOTAL_Q_HEADS = 64
COMPRESSED_SPARSE_MLA_KV_HEADS = 1
# SGLang's DSV4 backend hard-codes a 256-token physical KV page. Keep that
# distinct from the 128-token sliding SWA window.
COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE = 256
COMPRESSED_SPARSE_MLA_C4_PAGE_SIZE = COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE // 4
COMPRESSED_SPARSE_MLA_C128_PAGE_SIZE = COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE // 128
COMPRESSED_SPARSE_MLA_SWA_TOKENS = 128
COMPRESSED_SPARSE_MLA_INDEX_TOPK = 512


def compressed_sparse_mla_page_nbytes(page_size: int) -> int:
    """Return the padded byte count per compressed-MLA KV page."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    unpadded = page_size * COMPRESSED_SPARSE_MLA_BYTES_PER_TOKEN
    return (
        math.ceil(unpadded / COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN)
        * COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN
    )


def compressed_sparse_mla_scale_region_offset(page_size: int) -> int:
    """Return the byte offset at which UE8M0 scales start inside a page."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    return page_size * COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN


def ue8m0_to_float(scale_ue8m0: torch.Tensor) -> torch.Tensor:
    """Decode UE8M0 scale bytes using the standard exponent bias."""

    exponent = scale_ue8m0.to(torch.int32) - COMPRESSED_SPARSE_MLA_UE8M0_BIAS
    ones = torch.ones_like(exponent, dtype=torch.float32)
    return torch.ldexp(ones, exponent)


def _float_to_ue8m0_scale(max_abs: torch.Tensor) -> torch.Tensor:
    safe = torch.where(
        max_abs > 0, max_abs / COMPRESSED_SPARSE_MLA_FP8_MAX, torch.ones_like(max_abs)
    )
    exponent = torch.ceil(torch.log2(safe)).to(torch.int32)
    exponent = torch.clamp(
        exponent,
        -COMPRESSED_SPARSE_MLA_UE8M0_BIAS,
        255 - COMPRESSED_SPARSE_MLA_UE8M0_BIAS,
    )
    exponent = torch.where(max_abs > 0, exponent, torch.zeros_like(exponent))
    return (exponent + COMPRESSED_SPARSE_MLA_UE8M0_BIAS).to(torch.uint8)


def quantize_compressed_sparse_mla_nope_reference(
    k_nope: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize noPE rows into E4M3 FP8 plus seven UE8M0 scale bytes."""

    if k_nope.shape[-1] != COMPRESSED_SPARSE_MLA_NOPE_DIM:
        raise ValueError(
            f"k_nope last dim must be {COMPRESSED_SPARSE_MLA_NOPE_DIM}, got {k_nope.shape[-1]}"
        )
    grouped = k_nope.float().reshape(
        *k_nope.shape[:-1],
        COMPRESSED_SPARSE_MLA_NOPE_GROUPS,
        COMPRESSED_SPARSE_MLA_NOPE_GROUP_SIZE,
    )
    max_abs = grouped.abs().amax(dim=-1)
    scale_ue8m0 = _float_to_ue8m0_scale(max_abs)
    scale = ue8m0_to_float(scale_ue8m0).unsqueeze(-1)
    scaled = torch.clamp(
        grouped / scale, -COMPRESSED_SPARSE_MLA_FP8_MAX, COMPRESSED_SPARSE_MLA_FP8_MAX
    )
    quantized = scaled.reshape(k_nope.shape).to(torch.float8_e4m3fn)
    return quantized, scale_ue8m0


def pack_compressed_sparse_mla_kv_cache_reference(
    k_nope: torch.Tensor,
    k_rope: torch.Tensor,
    *,
    page_size: int,
    num_pages: int | None = None,
) -> torch.Tensor:
    """Pack K/V rows into the flat compressed-MLA uint8 page-buffer layout."""

    if k_nope.ndim != 2 or k_nope.shape[-1] != COMPRESSED_SPARSE_MLA_NOPE_DIM:
        raise ValueError(
            f"k_nope must have shape [tokens, {COMPRESSED_SPARSE_MLA_NOPE_DIM}], got {tuple(k_nope.shape)}"
        )
    if k_rope.ndim != 2 or k_rope.shape[-1] != COMPRESSED_SPARSE_MLA_ROPE_DIM:
        raise ValueError(
            f"k_rope must have shape [tokens, {COMPRESSED_SPARSE_MLA_ROPE_DIM}], got {tuple(k_rope.shape)}"
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

    page_nbytes = compressed_sparse_mla_page_nbytes(page_size)
    scale_offset = compressed_sparse_mla_scale_region_offset(page_size)
    cache = torch.zeros(
        (num_pages, page_nbytes), dtype=torch.uint8, device=k_nope.device
    )
    if n_tokens == 0:
        return cache

    k_nope_fp8, scale_ue8m0 = quantize_compressed_sparse_mla_nope_reference(k_nope)
    payload = torch.as_strided(
        cache,
        (num_pages, page_size, COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN),
        (page_nbytes, COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN, 1),
    )
    scales = torch.as_strided(
        cache,
        (num_pages, page_size, COMPRESSED_SPARSE_MLA_SCALE_BYTES_PER_TOKEN),
        (page_nbytes, COMPRESSED_SPARSE_MLA_SCALE_BYTES_PER_TOKEN, 1),
        storage_offset=scale_offset,
    )
    token_ids = torch.arange(n_tokens, dtype=torch.int64, device=k_nope.device)
    pages = token_ids // page_size
    token_offsets = token_ids % page_size

    payload[pages, token_offsets, :COMPRESSED_SPARSE_MLA_NOPE_DIM] = k_nope_fp8.view(
        torch.uint8
    ).view(n_tokens, COMPRESSED_SPARSE_MLA_NOPE_DIM)
    payload[
        pages,
        token_offsets,
        COMPRESSED_SPARSE_MLA_NOPE_DIM:COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN,
    ] = (
        k_rope.to(torch.bfloat16)
        .view(torch.uint8)
        .view(n_tokens, COMPRESSED_SPARSE_MLA_ROPE_DIM * 2)
    )
    scales[pages, token_offsets, :COMPRESSED_SPARSE_MLA_NOPE_GROUPS] = (
        scale_ue8m0.reshape(n_tokens, COMPRESSED_SPARSE_MLA_NOPE_GROUPS)
    )
    return cache


def unpack_compressed_sparse_mla_kv_cache_reference(
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
            (0, COMPRESSED_SPARSE_MLA_NOPE_DIM),
            dtype=torch.float32,
            device=kv_cache.device,
        )
        empty_rope = torch.empty(
            (0, COMPRESSED_SPARSE_MLA_ROPE_DIM),
            dtype=torch.float32,
            device=kv_cache.device,
        )
        return empty_nope, empty_rope

    indices = torch.arange(n_tokens, dtype=torch.int64, device=kv_cache.device)
    return gather_compressed_sparse_mla_kv_cache_reference(
        kv_cache, indices, page_size=page_size
    )


def gather_compressed_sparse_mla_kv_cache_reference(
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
            (0, COMPRESSED_SPARSE_MLA_HEAD_DIM),
            dtype=torch.float32,
            device=kv_cache.device,
        )
        return empty, empty

    page_nbytes = compressed_sparse_mla_page_nbytes(page_size)
    scale_offset = compressed_sparse_mla_scale_region_offset(page_size)
    flat_cache = kv_cache.reshape(-1)
    idx = indices.to(torch.int64)
    if bool((idx < 0).any()):
        raise ValueError("indices must be non-negative")
    pages = idx // page_size
    token_offsets = idx % page_size
    if bool((pages >= kv_cache.shape[0]).any()):
        raise ValueError("indices exceed kv_cache page capacity")

    token_base = (
        pages * page_nbytes
        + token_offsets * COMPRESSED_SPARSE_MLA_NOPE_ROPE_BYTES_PER_TOKEN
    )
    nope_offsets = token_base[:, None] + torch.arange(
        COMPRESSED_SPARSE_MLA_NOPE_DIM, device=kv_cache.device, dtype=torch.int64
    )
    rope_byte_offsets = (
        token_base[:, None]
        + COMPRESSED_SPARSE_MLA_NOPE_DIM
        + torch.arange(
            COMPRESSED_SPARSE_MLA_ROPE_DIM * 2,
            device=kv_cache.device,
            dtype=torch.int64,
        )
    )
    scale_offsets = (
        pages[:, None] * page_nbytes
        + scale_offset
        + token_offsets[:, None] * COMPRESSED_SPARSE_MLA_SCALE_BYTES_PER_TOKEN
        + torch.arange(
            COMPRESSED_SPARSE_MLA_NOPE_GROUPS, device=kv_cache.device, dtype=torch.int64
        )
    )

    nope_fp8 = (
        flat_cache[nope_offsets]
        .contiguous()
        .view(torch.float8_e4m3fn)
        .view(-1, COMPRESSED_SPARSE_MLA_NOPE_DIM)
    )
    scale = ue8m0_to_float(flat_cache[scale_offsets]).view(
        -1, COMPRESSED_SPARSE_MLA_NOPE_GROUPS, 1
    )
    nope = (
        nope_fp8.to(torch.float32).view(
            -1, COMPRESSED_SPARSE_MLA_NOPE_GROUPS, COMPRESSED_SPARSE_MLA_NOPE_GROUP_SIZE
        )
        * scale
    ).reshape(-1, COMPRESSED_SPARSE_MLA_NOPE_DIM)
    rope = (
        flat_cache[rope_byte_offsets]
        .contiguous()
        .view(torch.bfloat16)
        .view(-1, COMPRESSED_SPARSE_MLA_ROPE_DIM)
        .to(torch.float32)
    )
    full = torch.cat((nope, rope), dim=-1)
    return full, full


def pack_deepseek_v41_cache_reference(
    kv: torch.Tensor, *, page_size: int, cache_kind: str
) -> torch.Tensor:
    """Independent V4.1 record encoder, including RN-even E2M1 signed zero."""
    if kv.ndim != 2 or kv.shape[1] != 512:
        raise ValueError("V4.1 kv must have shape [tokens, 512]")
    if page_size <= 0 or cache_kind not in ("swa", "indexed"):
        raise ValueError("V4.1 requires a positive page_size and swa/indexed cache_kind")
    tokens = int(kv.shape[0])
    group_size = 32 if cache_kind == "swa" else 16
    grouped = kv.float().reshape(tokens, 512 // group_size, group_size)
    maximum = grouped.abs().amax(dim=-1)
    if cache_kind == "swa":
        scales = _float_to_ue8m0_scale(maximum.clamp_min(1e-4))
        normalized = grouped / ue8m0_to_float(scales).unsqueeze(-1)
        payload = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8)
        payload = payload.reshape(tokens, 512)
    else:
        scales_fp8 = (maximum.clamp_min(6 * 2**-9) / 6.0).clamp(max=448).to(torch.float8_e4m3fn)
        scale_f32 = scales_fp8.float().unsqueeze(-1)
        normalized = grouped * scale_f32.reciprocal()
        magnitude = normalized.abs().contiguous()
        boundaries = torch.tensor(
            [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0],
            dtype=torch.float32, device=kv.device,
        )
        codes = torch.bucketize(magnitude, boundaries)
        midpoint = boundaries[codes.clamp(max=6)]
        codes += ((codes < 7) & (magnitude == midpoint) & ((codes & 1) != 0)).to(codes.dtype)
        codes = codes.to(torch.uint8) | (torch.signbit(normalized).to(torch.uint8) << 3)
        codes = codes.reshape(tokens, 512)
        payload = codes[:, 0::2] | (codes[:, 1::2] << 4)
        scales = scales_fp8.view(torch.uint8)
    records = torch.cat((payload, scales), dim=-1)
    record_bytes = 528 if cache_kind == "swa" else 288
    cache = torch.zeros(
        ((tokens + page_size - 1) // page_size, page_size * record_bytes),
        dtype=torch.uint8, device=kv.device,
    )
    cache.view(-1, record_bytes)[:tokens].copy_(records)
    return cache


def _decode_deepseek_v41_records(records: torch.Tensor, *, cache_kind: str) -> torch.Tensor:
    if cache_kind == "swa":
        payload = records[..., :512].contiguous().view(torch.float8_e4m3fn).float()
        scales = ue8m0_to_float(records[..., 512:])
        return (payload.reshape(-1, 16, 32) * scales.reshape(-1, 16, 1)).reshape(-1, 512)
    if cache_kind != "indexed":
        raise ValueError("cache_kind must be swa or indexed")
    packed = records[..., :256]
    codes = torch.stack((packed & 15, packed >> 4), dim=-1).reshape(-1, 512)
    lut = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=records.device)
    payload = lut[(codes & 7).long()]
    payload = torch.where((codes & 8) != 0, -payload, payload)
    scales = records[..., 256:].contiguous().view(torch.float8_e4m3fn).float()
    return (payload.reshape(-1, 32, 16) * scales.reshape(-1, 32, 1)).reshape(-1, 512)


def unpack_deepseek_v41_cache_reference(
    cache: torch.Tensor, *, page_size: int, cache_kind: str
) -> torch.Tensor:
    """Decode every V4.1 record to FP32 without using the production reader."""
    record_bytes = 528 if cache_kind == "swa" else 288
    if cache.ndim != 2 or cache.shape[1] != page_size * record_bytes:
        raise ValueError("invalid V4.1 page byte width")
    return _decode_deepseek_v41_records(
        cache.view(torch.uint8).reshape(-1, record_bytes), cache_kind=cache_kind
    )


def _gather_cache_reference(
    cache: torch.Tensor, indices: torch.Tensor, *, page_size: int,
    cache_format: str, cache_kind: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cache_format == "deepseek_v4":
        return gather_compressed_sparse_mla_kv_cache_reference(cache, indices, page_size=page_size)
    record_bytes = 528 if cache_kind == "swa" else 288
    slots = indices.long()
    columns = (slots % page_size)[:, None] * record_bytes + torch.arange(record_bytes, device=cache.device)
    records = cache.view(torch.uint8)[(slots // page_size)[:, None], columns]
    values = _decode_deepseek_v41_records(records, cache_kind=cache_kind)
    return values, values


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
    swa_page_size: int = COMPRESSED_SPARSE_MLA_DSV4_PAGE_SIZE,
    extra_page_size: int | None = None,
    return_lse: bool = False,
    cache_format: str = "deepseek_v4",
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
    if cache_format not in ("deepseek_v4", "deepseek_v41"):
        raise ValueError("unsupported compressed MLA reference cache_format")
    if cache_format == "deepseek_v4":
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
        if cache_format == "deepseek_v4":
            _validate_flat_cache(extra_k_cache, page_size=extra_page_size)
    else:
        extra_indices_2d = None

    if attn_sink is not None and attn_sink.shape != (heads,):
        raise ValueError(
            f"attn_sink must have shape [{heads}], got {tuple(attn_sink.shape)}"
        )

    out = torch.empty(
        (rows, heads, COMPRESSED_SPARSE_MLA_HEAD_DIM),
        dtype=torch.float32,
        device=q3.device,
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
                extra_k, extra_v = _gather_cache_reference(
                    extra_k_cache,
                    extra_indices_2d[row, :extra_len],
                    page_size=extra_page_size,
                    cache_format=cache_format,
                    cache_kind="indexed",
                )
            else:
                extra_k = torch.empty(
                    (0, COMPRESSED_SPARSE_MLA_HEAD_DIM),
                    dtype=torch.float32,
                    device=q3.device,
                )
                extra_v = extra_k
        else:
            extra_len = 0
            extra_k = torch.empty(
                (0, COMPRESSED_SPARSE_MLA_HEAD_DIM),
                dtype=torch.float32,
                device=q3.device,
            )
            extra_v = extra_k

        if swa_len:
            swa_k, swa_v = _gather_cache_reference(
                swa_k_cache, swa_indices_2d[row, :swa_len], page_size=swa_page_size,
                cache_format=cache_format, cache_kind="swa",
            )
        else:
            swa_k = torch.empty(
                (0, COMPRESSED_SPARSE_MLA_HEAD_DIM),
                dtype=torch.float32,
                device=q3.device,
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
    if q.ndim != 3 or q.shape[-1] != COMPRESSED_SPARSE_MLA_HEAD_DIM:
        raise ValueError(
            f"q must have shape [rows, heads, {COMPRESSED_SPARSE_MLA_HEAD_DIM}], got {tuple(q.shape)}"
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
    expected = compressed_sparse_mla_page_nbytes(page_size)
    if kv_cache.shape[1] != expected:
        raise ValueError(
            f"kv_cache page byte width must be {expected} for page_size {page_size}, got {kv_cache.shape[1]}"
        )
