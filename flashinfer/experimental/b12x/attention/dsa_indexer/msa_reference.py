"""PyTorch references for MiniMax-M3 sparse-attention indexer contracts."""

from __future__ import annotations

import math

import torch

from .reference import _split_index_k_cache_reference


MSA_BLOCK_TOKENS = 128
MSA_TOPK_BLOCKS = 16
MSA_SM_SCALE = 1.0 / math.sqrt(128.0)

_FP8_E4M3_MAX = float(torch.finfo(torch.float8_e4m3fn).max)
_INDEX_HEAD_DIM = 128
_INT32_MAX = torch.iinfo(torch.int32).max


def _validate_q_fp8_and_scale(
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
) -> tuple[int, int]:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
        )
    if q_fp8.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"q_fp8 must have dtype torch.float8_e4m3fn, got {q_fp8.dtype}"
        )
    if q_scale.ndim != 2 or q_scale.shape != q_fp8.shape[:2]:
        raise ValueError(
            f"q_scale must have shape {tuple(q_fp8.shape[:2])}, got {tuple(q_scale.shape)}"
        )
    if q_scale.dtype != torch.float32:
        raise ValueError(f"q_scale must have dtype torch.float32, got {q_scale.dtype}")
    if q_scale.device != q_fp8.device:
        raise ValueError("q_scale must be on the same device as q_fp8")
    return int(q_fp8.shape[0]), int(q_fp8.shape[1])


def quantize_msa_q_fp8_reference(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MSA index-Q rows with one positive FP32 scale per (token, head)."""

    if q.ndim != 3:
        raise ValueError(f"q must be rank-3, got {tuple(q.shape)}")
    if q.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(f"q head_dim must be {_INDEX_HEAD_DIM}, got {q.shape[2]}")
    if not q.dtype.is_floating_point:
        raise ValueError(f"q must be floating point, got {q.dtype}")
    q_f32 = q.to(torch.float32)
    scale = q_f32.abs().amax(dim=2) / _FP8_E4M3_MAX
    scale = torch.where(scale > 0, scale, torch.ones_like(scale)).to(torch.float32)
    quant = (q_f32 / scale.unsqueeze(2)).clamp(-_FP8_E4M3_MAX, _FP8_E4M3_MAX)
    return quant.to(torch.float8_e4m3fn).contiguous(), scale.contiguous()


def _validate_kv_fp8(
    kv_fp8: tuple[torch.Tensor, torch.Tensor], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    k_quant, k_scale = kv_fp8
    if k_quant.ndim != 2 or k_quant.shape[1] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"k_quant must have shape (K, {_INDEX_HEAD_DIM}), got {tuple(k_quant.shape)}"
        )
    if k_quant.dtype != torch.float8_e4m3fn:
        raise ValueError(
            f"k_quant must have dtype torch.float8_e4m3fn, got {k_quant.dtype}"
        )
    if k_scale.ndim == 2 and k_scale.shape[1] == 1:
        k_scale = k_scale.squeeze(1)
    if k_scale.ndim != 1 or k_scale.shape[0] != k_quant.shape[0]:
        raise ValueError(
            f"k_scale must have shape ({k_quant.shape[0]},), got {tuple(k_scale.shape)}"
        )
    if k_scale.dtype != torch.float32:
        raise ValueError(f"k_scale must have dtype torch.float32, got {k_scale.dtype}")
    if k_quant.device != device or k_scale.device != device:
        raise ValueError("kv_fp8 tensors must be on the same device as q_fp8")
    return k_quant.contiguous(), k_scale.contiguous()


def _validate_row_bounds(
    k_start: torch.Tensor,
    k_end: torch.Tensor,
    *,
    q_rows: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    if k_start.ndim != 1 or k_end.ndim != 1:
        raise ValueError(
            f"k_start and k_end must be rank-1, got {tuple(k_start.shape)} and {tuple(k_end.shape)}"
        )
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got {tuple(k_start.shape)} vs {tuple(k_end.shape)}"
        )
    if k_start.shape[0] > q_rows:
        raise ValueError(
            f"metadata describes {k_start.shape[0]} query rows, but q has {q_rows}"
        )
    if k_start.dtype != torch.int32 or k_end.dtype != torch.int32:
        raise ValueError("k_start and k_end must have dtype torch.int32")
    if k_start.device != device or k_end.device != device:
        raise ValueError("k_start and k_end must be on the same device as q_fp8")
    return k_start.contiguous(), k_end.contiguous()


def _block_max_from_token_scores(
    token_scores: torch.Tensor,
    *,
    num_blocks: int,
) -> torch.Tensor:
    heads = int(token_scores.shape[0])
    out = torch.full(
        (heads, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=token_scores.device,
    )
    for block_idx in range(num_blocks):
        begin = block_idx * MSA_BLOCK_TOKENS
        end = min(begin + MSA_BLOCK_TOKENS, int(token_scores.shape[1]))
        if end > begin:
            out[:, block_idx] = token_scores[:, begin:end].amax(dim=1)
    return out


def msa_contiguous_block_scores_reference(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    k_start: torch.Tensor,
    k_end: torch.Tensor,
) -> torch.Tensor:
    """Return MSA column-max block scores for contiguous packed K rows.

    The block axis is global over the packed K matrix. Token masking is applied
    before the block max, so empty/future/padded rows remain ``-inf``.
    """

    q_rows, num_heads = _validate_q_fp8_and_scale(q_fp8, q_scale)
    k_quant, k_scale = _validate_kv_fp8(kv_fp8, q_fp8.device)
    k_start, k_end = _validate_row_bounds(
        k_start, k_end, q_rows=q_rows, device=q_fp8.device
    )

    k_rows = int(k_quant.shape[0])
    num_blocks = math.ceil(k_rows / MSA_BLOCK_TOKENS)
    out = torch.full(
        (num_heads, q_rows, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    if k_rows == 0 or k_start.numel() == 0:
        return out

    q_f32 = q_fp8.to(torch.float32)
    k_f32 = k_quant.to(torch.float32)
    token_pos = torch.arange(k_rows, dtype=torch.long, device=q_fp8.device)
    neg_inf = torch.full(
        (num_heads, k_rows), float("-inf"), dtype=torch.float32, device=q_fp8.device
    )
    for q_idx in range(int(k_start.numel())):
        ks = k_start[q_idx].to(torch.long).clamp(min=0, max=k_rows)
        ke = k_end[q_idx].to(torch.long).clamp(min=0, max=k_rows)
        valid = (token_pos >= ks) & (token_pos < ke)
        scores = torch.matmul(q_f32[q_idx], k_f32.transpose(0, 1))
        scores = (
            scores * q_scale[q_idx].unsqueeze(1) * k_scale.unsqueeze(0) * MSA_SM_SCALE
        )
        scores = torch.where(valid.unsqueeze(0), scores, neg_inf)
        out[:, q_idx, :] = _block_max_from_token_scores(scores, num_blocks=num_blocks)
    return out


def msa_paged_decode_block_scores_reference(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    index_k_cache: torch.Tensor,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    page_size: int = 64,
) -> torch.Tensor:
    """Return MSA block scores for decode over the paged index-K cache."""

    if page_size <= 0:
        raise ValueError(f"page_size must be positive, got {page_size}")
    q_rows, num_heads = _validate_q_fp8_and_scale(q_fp8, q_scale)
    if real_page_table.ndim != 2:
        raise ValueError(
            f"real_page_table must be rank-2, got {tuple(real_page_table.shape)}"
        )
    if real_page_table.dtype != torch.int32:
        raise ValueError(
            f"real_page_table must have dtype torch.int32, got {real_page_table.dtype}"
        )
    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            f"cache_seqlens_int32 must be rank-1, got {tuple(cache_seqlens_int32.shape)}"
        )
    if cache_seqlens_int32.dtype != torch.int32:
        raise ValueError(
            f"cache_seqlens_int32 must have dtype torch.int32, got {cache_seqlens_int32.dtype}"
        )
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            "real_page_table and cache_seqlens_int32 must describe the same rows"
        )
    if real_page_table.shape[0] > q_rows:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} exceed q rows {q_rows}"
        )
    if (
        real_page_table.device != q_fp8.device
        or cache_seqlens_int32.device != q_fp8.device
    ):
        raise ValueError("paged metadata tensors must be on the same device as q_fp8")

    k_quant, k_scale = _split_index_k_cache_reference(
        index_k_cache, page_size=page_size
    )
    width_tokens = int(real_page_table.shape[1]) * int(page_size)
    num_blocks = math.ceil(width_tokens / MSA_BLOCK_TOKENS)
    out = torch.full(
        (num_heads, q_rows, num_blocks),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    if width_tokens == 0 or real_page_table.shape[0] == 0:
        return out

    max_pages = int(k_quant.shape[0])
    q_f32 = q_fp8.to(torch.float32)
    token_pos = torch.arange(width_tokens, dtype=torch.long, device=q_fp8.device)
    page_col = torch.div(token_pos, page_size, rounding_mode="floor")
    slot_idx = token_pos % page_size
    neg_inf = torch.full(
        (num_heads, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    for q_idx in range(int(real_page_table.shape[0])):
        page_ids = real_page_table[q_idx, page_col].to(torch.long)
        seq_len = (
            cache_seqlens_int32[q_idx].to(torch.long).clamp(min=0, max=width_tokens)
        )
        valid = (token_pos < seq_len) & (page_ids >= 0) & (page_ids < max_pages)
        safe_page_ids = page_ids.clamp(0, max(max_pages - 1, 0))
        k_selected = k_quant[safe_page_ids, slot_idx]
        scale_selected = k_scale[safe_page_ids, slot_idx]
        scores = torch.matmul(q_f32[q_idx], k_selected.transpose(0, 1))
        scores = (
            scores
            * q_scale[q_idx].unsqueeze(1)
            * scale_selected.unsqueeze(0)
            * MSA_SM_SCALE
        )
        scores = torch.where(valid.unsqueeze(0), scores, neg_inf)
        out[:, q_idx, :] = _block_max_from_token_scores(scores, num_blocks=num_blocks)
    return out


def msa_select_blocks_reference(
    *,
    block_scores: torch.Tensor,
    query_positions: torch.Tensor,
    block_base: torch.Tensor | None = None,
    topk: int = MSA_TOPK_BLOCKS,
) -> torch.Tensor:
    """Select ascending block ids from raw MSA block scores with local forcing."""

    if block_scores.ndim != 3:
        raise ValueError(
            f"block_scores must be rank-3, got {tuple(block_scores.shape)}"
        )
    if block_scores.dtype != torch.float32:
        raise ValueError(
            f"block_scores must have dtype torch.float32, got {block_scores.dtype}"
        )
    if query_positions.ndim != 1 or query_positions.shape[0] != block_scores.shape[1]:
        raise ValueError(
            f"query_positions must have shape ({block_scores.shape[1]},), got {tuple(query_positions.shape)}"
        )
    if query_positions.dtype != torch.int32:
        raise ValueError(
            f"query_positions must have dtype torch.int32, got {query_positions.dtype}"
        )
    if query_positions.device != block_scores.device:
        raise ValueError("query_positions must be on the same device as block_scores")
    if block_base is not None:
        if block_base.ndim != 1 or block_base.shape != query_positions.shape:
            raise ValueError(
                f"block_base must have shape {tuple(query_positions.shape)}, got {tuple(block_base.shape)}"
            )
        if block_base.dtype != torch.int32:
            raise ValueError(
                f"block_base must have dtype torch.int32, got {block_base.dtype}"
            )
        if block_base.device != block_scores.device:
            raise ValueError("block_base must be on the same device as block_scores")
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")

    heads, q_rows, num_blocks = map(int, block_scores.shape)
    result = torch.full(
        (heads, q_rows, topk), -1, dtype=torch.int32, device=block_scores.device
    )
    if topk == 0 or num_blocks == 0:
        return result

    scores = block_scores.clone()
    local_blocks = torch.div(query_positions, MSA_BLOCK_TOKENS, rounding_mode="floor")
    valid_local = (local_blocks >= 0) & (local_blocks < num_blocks)
    scatter_idx = (
        local_blocks.clamp(min=0, max=num_blocks - 1)
        .to(torch.long)
        .view(1, q_rows, 1)
        .expand(heads, q_rows, 1)
    )
    current_local = torch.gather(scores, 2, scatter_idx)
    force_values = torch.where(
        valid_local.view(1, q_rows, 1),
        torch.full_like(current_local, float("inf")),
        current_local,
    )
    scores.scatter_(2, scatter_idx, force_values)

    gather_k = min(topk, num_blocks)
    top_values, top_indices = torch.topk(
        scores, k=gather_k, dim=2, largest=True, sorted=False
    )
    base = torch.zeros_like(query_positions) if block_base is None else block_base
    local_indices = top_indices.to(torch.int32) - base.view(1, q_rows, 1)
    valid = top_values > float("-inf")
    valid = valid & (local_indices >= 0)
    masked = torch.where(
        valid,
        local_indices,
        torch.full_like(local_indices, _INT32_MAX),
    )
    sorted_indices = torch.sort(masked, dim=2).values
    sorted_indices = torch.where(
        sorted_indices == _INT32_MAX,
        torch.full_like(sorted_indices, -1),
        sorted_indices,
    )
    result[:, :, :gather_k].copy_(sorted_indices.to(torch.int32))
    return result


def msa_q2k_indices_reference(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    query_positions: torch.Tensor,
    topk: int = MSA_TOPK_BLOCKS,
    block_base: torch.Tensor | None = None,
    kv_fp8: tuple[torch.Tensor, torch.Tensor] | None = None,
    k_start: torch.Tensor | None = None,
    k_end: torch.Tensor | None = None,
    index_k_cache: torch.Tensor | None = None,
    real_page_table: torch.Tensor | None = None,
    cache_seqlens_int32: torch.Tensor | None = None,
    page_size: int = 64,
) -> torch.Tensor:
    """End-to-end MSA reference for either contiguous prefill or paged decode."""

    has_contiguous = kv_fp8 is not None or k_start is not None or k_end is not None
    has_paged = (
        index_k_cache is not None
        or real_page_table is not None
        or cache_seqlens_int32 is not None
    )
    if has_contiguous == has_paged:
        raise ValueError(
            "provide exactly one of contiguous kv_fp8/k_start/k_end or paged cache metadata"
        )
    if has_contiguous:
        if kv_fp8 is None or k_start is None or k_end is None:
            raise ValueError("contiguous reference requires kv_fp8, k_start, and k_end")
        block_scores = msa_contiguous_block_scores_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
    else:
        if (
            index_k_cache is None
            or real_page_table is None
            or cache_seqlens_int32 is None
        ):
            raise ValueError(
                "paged reference requires index_k_cache, real_page_table, and cache_seqlens_int32"
            )
        block_scores = msa_paged_decode_block_scores_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            real_page_table=real_page_table,
            cache_seqlens_int32=cache_seqlens_int32,
            page_size=page_size,
        )
    return msa_select_blocks_reference(
        block_scores=block_scores,
        query_positions=query_positions,
        block_base=block_base,
        topk=topk,
    )
