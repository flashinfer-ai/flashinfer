# SPDX-FileCopyrightText: 2026 FlashInfer team
# SPDX-License-Identifier: Apache-2.0
# Ported from b12x b12x/attention/indexer/api.py @ 118a3b70 (2026-06-12) -- one-time curated port.
# Upstream b12x is a research sandbox; this tree is the canonical home.
"""NSA indexer API for paged and contiguous logits contracts."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from functools import lru_cache

import torch

from .kernel import (
    IndexerScoreMode,
    IndexerOutputMode,
    PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
    _should_use_schedule_multi_row_kernel,
    clear_indexer_kernel_cache,
    _should_use_schedule_single_row_kernel,
    run_paged_logits_kernel,
    supports_paged_logits_kernel,
)
from .contiguous_kernel import (
    _PREFILL512_BLOCK_K,
    _PREFILL512_BLOCK_Q,
    _PREFILL_BLOCK_K,
    _PREFILL_BLOCK_Q,
    build_indexer_contiguous_logits_kernel_binding,
    resolve_contiguous_prefill_block_k,
    run_contiguous_block_scores_kernel,
    run_contiguous_logits_kernel,
    supports_contiguous_logits_kernel,
)
from .msa_reference import (
    MSA_BLOCK_TOKENS,
    MSA_SM_SCALE,
    MSA_TOPK_BLOCKS,
    msa_contiguous_block_scores_reference,
    msa_paged_decode_block_scores_reference,
    quantize_msa_q_fp8_reference,
)
from .reference import contiguous_logits_reference
from .schedule_metadata import (
    build_paged_mqa_schedule_metadata_torch,
    build_paged_mqa_schedule_metadata_triton,
)
from .tiled_topk import (
    _resolve_supertile_k,
    clear_tiled_topk_kernel_cache,
    run_tiled_topk,
)
from .persistent_topk import clear_persistent_topk2048_kernel_cache


_INDEX_HEAD_DIM = 128
_INT32_MAX = torch.iinfo(torch.int32).max
_VALIDATE_PAGE_IDS = bool(
    int(os.getenv("FLASHINFER_EXP_SM12X_NSA_VALIDATE_PAGE_IDS", "0"))
)


def _is_cuda_graph_capture_active(device: torch.device) -> bool:
    return device.type == "cuda" and torch.cuda.is_current_stream_capturing()


@dataclass(frozen=True)
class IndexerPagedDecodeMetadata:
    real_page_table: torch.Tensor
    cache_seqlens_int32: torch.Tensor
    paged_mqa_schedule_metadata: torch.Tensor | None = None


@dataclass(frozen=True)
class IndexerContiguousMetadata:
    k_start: torch.Tensor
    k_end: torch.Tensor


def build_paged_mqa_schedule_metadata(
    context_lens: torch.Tensor,
    block_kv: int,
    num_sms: int | None = None,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build paged-MQA schedule metadata on the input device."""

    if context_lens.ndim not in (1, 2):
        raise ValueError(
            f"context_lens must be rank-1 or rank-2, got {tuple(context_lens.shape)}"
        )
    if context_lens.ndim == 2 and context_lens.shape[1] == 0:
        raise ValueError(
            "context_lens rank-2 input must have a non-empty trailing dimension"
        )
    if context_lens.dtype != torch.int32:
        raise ValueError(
            f"context_lens must have dtype torch.int32, got {context_lens.dtype}"
        )
    if not context_lens.is_contiguous():
        raise ValueError("context_lens must be contiguous")
    if block_kv <= 0:
        raise ValueError(f"block_kv must be positive, got {block_kv}")
    if out is not None:
        if out.ndim != 2 or out.shape[1] != 2:
            raise ValueError(
                f"out must have shape (num_sms + 1, 2), got {tuple(out.shape)}"
            )
        if out.dtype != torch.int32:
            raise ValueError(f"out must have dtype torch.int32, got {out.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        if out.device != context_lens.device:
            raise ValueError(
                f"out device {out.device} does not match context_lens device {context_lens.device}"
            )
        if num_sms is None:
            num_sms = out.shape[0] - 1
    if num_sms is None:
        if context_lens.device.type == "cuda":
            num_sms = torch.cuda.get_device_properties(
                context_lens.device
            ).multi_processor_count
        else:
            num_sms = 1
    if num_sms <= 0:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    if out is not None and out.shape[0] != num_sms + 1:
        raise ValueError(
            f"out leading dimension {out.shape[0]} does not match num_sms + 1 ({num_sms + 1})"
        )

    schedule = out
    if schedule is None:
        schedule = torch.empty(
            (num_sms + 1, 2),
            dtype=torch.int32,
            device=context_lens.device,
        )
    builder = (
        build_paged_mqa_schedule_metadata_triton
        if context_lens.device.type == "cuda"
        else build_paged_mqa_schedule_metadata_torch
    )
    return builder(
        context_lens,
        block_kv=block_kv,
        num_sms=num_sms,
        pages_per_split=PAGED_MQA_LOGITS_SCHEDULE_PAGES_PER_SPLIT,
        out=schedule,
    )


def clear_indexer_caches() -> None:
    """Clear any cached NSA indexer runtime state."""
    clear_indexer_kernel_cache()
    clear_tiled_topk_kernel_cache()
    clear_persistent_topk2048_kernel_cache()
    _cached_width_cap_tensor.cache_clear()


def uses_paged_mqa_schedule(
    *,
    q_rows: int,
    max_pages: int,
) -> bool:
    """Return whether decode should use a schedule-driven scorer path."""
    return _should_use_schedule_single_row_kernel(
        q_rows=q_rows,
        max_pages=max_pages,
    ) or _should_use_schedule_multi_row_kernel(
        q_rows=q_rows,
        max_pages=max_pages,
    )


def _normalize_weights(
    weights: torch.Tensor,
    *,
    q_rows: int,
    num_heads: int,
    require_float32: bool = False,
) -> torch.Tensor:
    if weights.ndim == 3:
        if weights.shape[2] != 1:
            raise ValueError(
                f"weights rank-3 input must have trailing dimension 1, got {tuple(weights.shape)}"
            )
        weights = weights.squeeze(2)
    if weights.ndim != 2:
        raise ValueError(
            f"weights must be rank-2 or rank-3, got {tuple(weights.shape)}"
        )
    if weights.shape != (q_rows, num_heads):
        raise ValueError(
            f"weights shape must be {(q_rows, num_heads)}, got {tuple(weights.shape)}"
        )
    if require_float32 and weights.dtype != torch.float32:
        raise ValueError(
            f"strict indexer contiguous requires torch.float32 weights, got {weights.dtype}"
        )
    return weights.to(torch.float32)


@lru_cache(maxsize=64)
def _cached_width_cap_tensor(
    width: int,
    device_type: str,
    device_index: int | None,
) -> torch.Tensor:
    return torch.tensor(
        [width], dtype=torch.int32, device=torch.device(device_type, device_index)
    )


def _make_active_width_tensor(
    *,
    seqlens_per_query: torch.Tensor,
    width: int,
) -> torch.Tensor:
    if seqlens_per_query.ndim != 1:
        raise ValueError(
            "seqlens_per_query must be rank-1 when computing active width, got "
            f"{tuple(seqlens_per_query.shape)}"
        )
    active_width = seqlens_per_query.amax().reshape(1)
    if _is_cuda_graph_capture_active(seqlens_per_query.device):
        return active_width.clamp_(min=0, max=int(width))
    width_cap = _cached_width_cap_tensor(
        int(width),
        seqlens_per_query.device.type,
        seqlens_per_query.device.index,
    )
    return torch.minimum(active_width, width_cap)


def quantize_msa_q_fp8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize MSA index-Q rows into the production FP8+scale contract."""

    return quantize_msa_q_fp8_reference(q)


def msa_decode_query_positions(cache_seqlens_int32: torch.Tensor) -> torch.Tensor:
    """Return decode query token positions from cache lengths."""

    if cache_seqlens_int32.ndim != 1:
        raise ValueError(
            f"cache_seqlens_int32 must be rank-1, got {tuple(cache_seqlens_int32.shape)}"
        )
    if cache_seqlens_int32.dtype != torch.int32:
        raise ValueError(
            f"cache_seqlens_int32 must have dtype torch.int32, got {cache_seqlens_int32.dtype}"
        )
    return cache_seqlens_int32 - 1


def msa_prefill_query_positions(
    cu_seqlens_q: torch.Tensor,
    total_q: int,
) -> torch.Tensor:
    """Return packed prefill query token positions."""

    if cu_seqlens_q.ndim != 1:
        raise ValueError(
            f"cu_seqlens_q must be rank-1, got {tuple(cu_seqlens_q.shape)}"
        )
    if cu_seqlens_q.dtype != torch.int32:
        raise ValueError(
            f"cu_seqlens_q must have dtype torch.int32, got {cu_seqlens_q.dtype}"
        )
    total_q = int(total_q)
    if total_q < 0:
        raise ValueError(f"total_q must be non-negative, got {total_q}")
    return torch.arange(total_q, dtype=torch.int32, device=cu_seqlens_q.device)


def _validate_msa_block_scores_selection(
    *,
    block_scores: torch.Tensor,
    query_positions: torch.Tensor,
    block_base: torch.Tensor | None,
    topk: int,
    out_indices: torch.Tensor | None,
) -> tuple[int, int, int]:
    if block_scores.ndim != 3:
        raise ValueError(
            f"block_scores must be rank-3, got {tuple(block_scores.shape)}"
        )
    if block_scores.dtype != torch.float32:
        raise ValueError(
            f"block_scores must have dtype torch.float32, got {block_scores.dtype}"
        )
    num_heads, q_rows, num_blocks = map(int, block_scores.shape)
    if query_positions.ndim != 1 or query_positions.shape[0] != q_rows:
        raise ValueError(
            f"query_positions must have shape ({q_rows},), got {tuple(query_positions.shape)}"
        )
    if query_positions.dtype != torch.int32:
        raise ValueError(
            f"query_positions must have dtype torch.int32, got {query_positions.dtype}"
        )
    if query_positions.device != block_scores.device:
        raise ValueError("query_positions device must match block_scores")
    if block_base is not None:
        if block_base.ndim != 1 or block_base.shape[0] != q_rows:
            raise ValueError(
                f"block_base must have shape ({q_rows},), got {tuple(block_base.shape)}"
            )
        if block_base.dtype != torch.int32:
            raise ValueError(
                f"block_base must have dtype torch.int32, got {block_base.dtype}"
            )
        if block_base.device != block_scores.device:
            raise ValueError("block_base device must match block_scores")
    if out_indices is not None:
        if out_indices.dtype != torch.int32:
            raise ValueError(
                f"out_indices must have dtype torch.int32, got {out_indices.dtype}"
            )
        if out_indices.device != block_scores.device:
            raise ValueError("out_indices device must match block_scores")
        if (
            out_indices.ndim != 3
            or out_indices.shape[0] < num_heads
            or out_indices.shape[1] < q_rows
            or out_indices.shape[2] < topk
        ):
            raise ValueError(
                f"out_indices must have shape at least ({num_heads}, {q_rows}, {topk}), "
                f"got {tuple(out_indices.shape)}"
            )
    return num_heads, q_rows, num_blocks


def msa_topk_blocks(
    *,
    block_scores: torch.Tensor,
    query_positions: torch.Tensor,
    block_base: torch.Tensor | None = None,
    topk: int = MSA_TOPK_BLOCKS,
    out_indices: torch.Tensor | None = None,
    score_scratch: torch.Tensor | None = None,
    top_values: torch.Tensor | None = None,
    top_indices: torch.Tensor | None = None,
    sort_values: torch.Tensor | None = None,
    sort_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select MSA block ids from raw block scores.

    Output ids are ascending and batch-local when ``block_base`` is provided.
    Invalid trailing entries are ``-1``.
    """

    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    num_heads, q_rows, num_blocks = _validate_msa_block_scores_selection(
        block_scores=block_scores,
        query_positions=query_positions,
        block_base=block_base,
        topk=topk,
        out_indices=out_indices,
    )
    if out_indices is None:
        result = torch.full(
            (num_heads, q_rows, topk),
            -1,
            dtype=torch.int32,
            device=block_scores.device,
        )
    else:
        result = out_indices[:num_heads, :q_rows, :topk]
        result.fill_(-1)
    if topk == 0 or num_blocks == 0:
        return result

    if score_scratch is not None:
        if (
            score_scratch.dtype != torch.float32
            or score_scratch.device != block_scores.device
        ):
            raise ValueError("score_scratch must be float32 on the block_scores device")
        if (
            score_scratch.ndim != 3
            or score_scratch.shape[0] < num_heads
            or score_scratch.shape[1] < q_rows
            or score_scratch.shape[2] < num_blocks
        ):
            raise ValueError(
                f"score_scratch must have shape at least ({num_heads}, {q_rows}, {num_blocks}), "
                f"got {tuple(score_scratch.shape)}"
            )
        scores = score_scratch[:num_heads, :q_rows, :num_blocks]
        scores.copy_(block_scores)
    else:
        scores = block_scores.clone()
    local_blocks = torch.div(query_positions, MSA_BLOCK_TOKENS, rounding_mode="floor")
    valid_local = (local_blocks >= 0) & (local_blocks < num_blocks)
    scatter_idx = (
        local_blocks.clamp(min=0, max=num_blocks - 1)
        .to(torch.long)
        .view(1, q_rows, 1)
        .expand(num_heads, q_rows, 1)
    )
    current_local = torch.gather(scores, 2, scatter_idx)
    force_values = torch.where(
        valid_local.view(1, q_rows, 1),
        torch.full_like(current_local, float("inf")),
        current_local,
    )
    scores.scatter_(2, scatter_idx, force_values)

    gather_k = min(topk, num_blocks)
    topk_out = None
    if top_values is not None or top_indices is not None:
        if top_values is None or top_indices is None:
            raise ValueError("top_values and top_indices must be provided together")
        if (
            top_values.dtype != torch.float32
            or top_values.device != block_scores.device
        ):
            raise ValueError("top_values must be float32 on the block_scores device")
        if (
            top_indices.dtype != torch.int64
            or top_indices.device != block_scores.device
        ):
            raise ValueError("top_indices must be int64 on the block_scores device")
        if (
            top_values.ndim != 3
            or top_indices.ndim != 3
            or top_values.shape[0] < num_heads
            or top_values.shape[1] < q_rows
            or top_values.shape[2] < gather_k
            or top_indices.shape[0] < num_heads
            or top_indices.shape[1] < q_rows
            or top_indices.shape[2] < gather_k
        ):
            raise ValueError("top-k scratch tensors are too small")
        topk_out = (
            top_values[:num_heads, :q_rows, :gather_k],
            top_indices[:num_heads, :q_rows, :gather_k],
        )
    top_values, top_indices = torch.topk(
        scores,
        k=gather_k,
        dim=2,
        largest=True,
        sorted=False,
        out=topk_out,
    )
    if out_indices is not None:
        local_indices = result[:, :, :gather_k]
        local_indices.copy_(top_indices)
        if block_base is not None:
            local_indices.sub_(block_base.view(1, q_rows, 1))
    elif sort_values is not None:
        if (
            sort_values.dtype != torch.int32
            or sort_values.device != block_scores.device
        ):
            raise ValueError("sort_values must be int32 on the block_scores device")
        if (
            sort_values.ndim != 3
            or sort_values.shape[0] < num_heads
            or sort_values.shape[1] < q_rows
            or sort_values.shape[2] < gather_k
        ):
            raise ValueError("sort_values scratch tensor is too small")
        local_indices = sort_values[:num_heads, :q_rows, :gather_k]
        local_indices.copy_(top_indices)
        if block_base is not None:
            local_indices.sub_(block_base.view(1, q_rows, 1))
    else:
        local_indices = top_indices.to(torch.int32)
        if block_base is not None:
            local_indices = local_indices - block_base.view(1, q_rows, 1)
    valid = (top_values > float("-inf")) & (local_indices >= 0)
    local_indices.masked_fill_(~valid, _INT32_MAX)
    sort_out = None
    if sort_values is not None and sort_indices is not None:
        if (
            sort_indices.dtype != torch.int64
            or sort_indices.device != block_scores.device
        ):
            raise ValueError("sort_indices must be int64 on the block_scores device")
        if (
            sort_indices.ndim != 3
            or sort_indices.shape[0] < num_heads
            or sort_indices.shape[1] < q_rows
            or sort_indices.shape[2] < gather_k
        ):
            raise ValueError("sort_indices scratch tensor is too small")
        sort_out = (
            sort_values[:num_heads, :q_rows, :gather_k],
            sort_indices[:num_heads, :q_rows, :gather_k],
        )
    if sort_out is None:
        sorted_indices = torch.sort(local_indices, dim=2).values
    else:
        sorted_indices = torch.sort(local_indices, dim=2, out=sort_out).values
    sorted_indices.masked_fill_(sorted_indices == _INT32_MAX, -1)
    result[:, :, :gather_k].copy_(sorted_indices)
    return result


def _validate_paged_decode_inputs(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    real_page_table: torch.Tensor,
    cache_seqlens_int32: torch.Tensor,
    paged_mqa_schedule_metadata: torch.Tensor | None,
) -> torch.Tensor:
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
        )
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
            "cache_seqlens_int32 must be rank-1, got "
            f"{tuple(cache_seqlens_int32.shape)}"
        )
    if real_page_table.shape[0] != cache_seqlens_int32.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} do not match "
            f"cache_seqlens rows {cache_seqlens_int32.shape[0]}"
        )
    if real_page_table.shape[0] > q_fp8.shape[0]:
        raise ValueError(
            f"real_page_table rows {real_page_table.shape[0]} exceed q rows {q_fp8.shape[0]}"
        )
    if real_page_table.device != q_fp8.device:
        raise ValueError(
            f"real_page_table device {real_page_table.device} does not match q_fp8 device {q_fp8.device}"
        )
    if cache_seqlens_int32.device != q_fp8.device:
        raise ValueError(
            f"cache_seqlens_int32 device {cache_seqlens_int32.device} does not match "
            f"q_fp8 device {q_fp8.device}"
        )
    if paged_mqa_schedule_metadata is not None:
        if paged_mqa_schedule_metadata.ndim != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must be rank-2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.shape[1] != 2:
            raise ValueError(
                "paged_mqa_schedule_metadata trailing dimension must be 2, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.shape[0] < 2:
            raise ValueError(
                "paged_mqa_schedule_metadata must have at least two rows, got "
                f"{tuple(paged_mqa_schedule_metadata.shape)}"
            )
        if paged_mqa_schedule_metadata.dtype != torch.int32:
            raise ValueError(
                "paged_mqa_schedule_metadata must have dtype torch.int32, got "
                f"{paged_mqa_schedule_metadata.dtype}"
            )
        if not paged_mqa_schedule_metadata.is_contiguous():
            raise ValueError("paged_mqa_schedule_metadata must be contiguous")
        if paged_mqa_schedule_metadata.device != q_fp8.device:
            raise ValueError(
                "paged_mqa_schedule_metadata device "
                f"{paged_mqa_schedule_metadata.device} does not match q_fp8 device {q_fp8.device}"
            )
    return _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])


def paged_decode_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: IndexerPagedDecodeMetadata | None = None,
    page_size: int = 64,
    preinitialize_invalid_logits: bool = True,
    active_width_override: torch.Tensor | None = None,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    binding=None,
) -> torch.Tensor:
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("active_width_override", active_width_override),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "paged indexer binding owns metadata, scratch, and active width; "
                f"do not also pass {', '.join(extras)}"
            )
        metadata = binding.metadata
        active_width_override = binding.active_width
    if metadata is None:
        raise TypeError("paged_decode_logits requires metadata or binding")
    score_mode = int(score_mode)
    if score_mode not in (IndexerScoreMode.NSA_RELU_SUM, IndexerScoreMode.MSA_BILINEAR):
        raise ValueError(f"unsupported indexer score_mode {score_mode}")

    weights_f = _validate_paged_decode_inputs(
        q_fp8=q_fp8,
        weights=weights,
        real_page_table=metadata.real_page_table,
        cache_seqlens_int32=metadata.cache_seqlens_int32,
        paged_mqa_schedule_metadata=metadata.paged_mqa_schedule_metadata,
    )

    valid_q_rows = metadata.real_page_table.shape[0]
    full_q_rows = q_fp8.shape[0]
    width_tokens = metadata.real_page_table.shape[1] * page_size
    if valid_q_rows == 0 or width_tokens == 0:
        return torch.full(
            (full_q_rows, width_tokens),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )

    seqlens_valid = metadata.cache_seqlens_int32.contiguous()
    if active_width_override is None:
        active_width = _make_active_width_tensor(
            seqlens_per_query=seqlens_valid, width=width_tokens
        )
    else:
        if active_width_override.shape != (1,):
            raise ValueError(
                f"active_width_override must have shape (1,), got {tuple(active_width_override.shape)}"
            )
        if active_width_override.dtype != torch.int32:
            raise ValueError(
                "active_width_override must have dtype torch.int32, got "
                f"{active_width_override.dtype}"
            )
        if active_width_override.device != q_fp8.device:
            raise ValueError(
                "active_width_override device "
                f"{active_width_override.device} does not match q_fp8 device {q_fp8.device}"
            )
        active_width = active_width_override

    validate_page_ids = q_fp8.device.type != "cuda" or (
        _VALIDATE_PAGE_IDS and not _is_cuda_graph_capture_active(q_fp8.device)
    )
    if validate_page_ids:
        active_width_host = min(width_tokens, int(active_width.item()))
        if active_width_host > 0:
            max_page_capacity = index_k_cache.shape[0]
            positions = torch.arange(
                active_width_host,
                dtype=torch.int32,
                device=q_fp8.device,
            ).unsqueeze(0)
            page_cols = torch.div(positions, page_size, rounding_mode="floor").to(
                torch.long
            )
            page_cols = page_cols.expand(valid_q_rows, -1)
            candidate_pages = metadata.real_page_table.gather(1, page_cols)
            candidate_valid_mask = (positions < seqlens_valid.unsqueeze(1)) & (
                candidate_pages >= 0
            )
            overflow_mask = candidate_valid_mask & (
                candidate_pages >= max_page_capacity
            )
            if torch.any(overflow_mask):
                bad = int(candidate_pages[overflow_mask].max().item())
                raise ValueError(
                    f"real_page_table page id {bad} exceeds index_k_cache page capacity {max_page_capacity}"
                )

    if not supports_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        page_size=page_size,
    ):
        raise NotImplementedError(
            "SM12X sparse NSA paged logits requires the production CUDA FP8 "
            "kernel contract; refusing to run the reference fallback. "
            f"q_fp8 shape={tuple(q_fp8.shape)} dtype={q_fp8.dtype} "
            f"device={q_fp8.device}, weights shape={tuple(weights_f.shape)} "
            f"dtype={weights_f.dtype}, index_k_cache shape={tuple(index_k_cache.shape)} "
            f"dtype={index_k_cache.dtype}, real_page_table shape="
            f"{tuple(metadata.real_page_table.shape)} dtype="
            f"{metadata.real_page_table.dtype}, cache_seqlens shape="
            f"{tuple(seqlens_valid.shape)} dtype={seqlens_valid.dtype}, "
            f"page_size={page_size}"
        )

    schedule_metadata = None
    if uses_paged_mqa_schedule(
        q_rows=valid_q_rows,
        max_pages=int(metadata.real_page_table.shape[1]),
    ):
        schedule_metadata = metadata.paged_mqa_schedule_metadata
        if schedule_metadata is None:
            if _is_cuda_graph_capture_active(q_fp8.device):
                raise ValueError(
                    "paged_mqa_schedule_metadata must be precomputed before CUDA graph capture "
                    "for the scheduled decode path"
                )
            schedule_metadata = build_paged_mqa_schedule_metadata(
                seqlens_valid, page_size
            )
    logits_valid = run_paged_logits_kernel(
        q_fp8=q_fp8[:valid_q_rows],
        weights=weights_f[:valid_q_rows],
        index_k_cache=index_k_cache,
        real_page_table=metadata.real_page_table,
        seqlens_per_query=seqlens_valid,
        schedule_metadata=schedule_metadata,
        active_width=active_width,
        page_size=page_size,
        preinitialize_invalid_logits=preinitialize_invalid_logits,
        score_mode=score_mode,
    )
    if valid_q_rows == full_q_rows:
        return logits_valid

    logits = torch.full(
        (full_q_rows, width_tokens),
        float("-inf"),
        dtype=torch.float32,
        device=q_fp8.device,
    )
    logits[:valid_q_rows].copy_(logits_valid)
    return logits


def _validate_msa_q_inputs(
    *,
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
        raise ValueError("q_scale device must match q_fp8")
    return int(q_fp8.shape[0]), int(q_fp8.shape[1])


def _validate_msa_block_scores_out(
    *,
    out: torch.Tensor | None,
    num_heads: int,
    q_rows: int,
    num_blocks: int,
    device: torch.device,
) -> torch.Tensor | None:
    if out is None:
        return None
    if out.dtype != torch.float32:
        raise ValueError(f"out must have dtype torch.float32, got {out.dtype}")
    if out.device != device:
        raise ValueError("out device must match q_fp8")
    if (
        out.ndim != 3
        or out.shape[0] < num_heads
        or out.shape[1] < q_rows
        or out.shape[2] < num_blocks
    ):
        raise ValueError(
            f"out must have shape at least ({num_heads}, {q_rows}, {num_blocks}), "
            f"got {tuple(out.shape)}"
        )
    return out[:num_heads, :q_rows, :num_blocks]


def msa_paged_decode_block_scores(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: IndexerPagedDecodeMetadata | None = None,
    page_size: int = 64,
    out: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    """Score MSA decode candidates and max-pool over 128-token KV blocks."""

    active_width_override = None
    page_scores_scratch = None
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("out", out),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "MSA paged binding owns metadata and block-score buffers; do not also pass "
                f"{', '.join(extras)}"
            )
        metadata = binding.metadata
        active_width_override = getattr(binding, "active_width", None)
        page_scores_scratch = getattr(binding, "page_scores", None)
        out = getattr(binding, "block_scores", None)
    if metadata is None:
        raise TypeError("msa_paged_decode_block_scores requires metadata or binding")
    q_rows, num_heads = _validate_msa_q_inputs(q_fp8=q_fp8, q_scale=q_scale)
    if metadata.real_page_table.ndim != 2:
        raise ValueError(
            f"real_page_table must be rank-2, got {tuple(metadata.real_page_table.shape)}"
        )
    if metadata.cache_seqlens_int32.ndim != 1:
        raise ValueError(
            "cache_seqlens_int32 must be rank-1, got "
            f"{tuple(metadata.cache_seqlens_int32.shape)}"
        )
    valid_q_rows = int(metadata.real_page_table.shape[0])
    if metadata.cache_seqlens_int32.shape[0] != valid_q_rows:
        raise ValueError("real_page_table rows must match cache_seqlens_int32")
    if valid_q_rows > q_rows:
        raise ValueError(
            f"metadata describes {valid_q_rows} q rows, but q_fp8 has {q_rows}"
        )
    width_tokens = int(metadata.real_page_table.shape[1]) * int(page_size)
    num_blocks = math.ceil(width_tokens / MSA_BLOCK_TOKENS)
    out_view = _validate_msa_block_scores_out(
        out=out,
        num_heads=num_heads,
        q_rows=q_rows,
        num_blocks=num_blocks,
        device=q_fp8.device,
    )
    if out_view is None:
        block_scores = torch.full(
            (num_heads, q_rows, num_blocks),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    else:
        block_scores = out_view
        block_scores.fill_(float("-inf"))
    if valid_q_rows == 0 or width_tokens == 0:
        return block_scores

    if q_fp8.device.type != "cuda":
        ref = msa_paged_decode_block_scores_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            index_k_cache=index_k_cache,
            real_page_table=metadata.real_page_table,
            cache_seqlens_int32=metadata.cache_seqlens_int32,
            page_size=page_size,
        )
        block_scores.copy_(ref[:, :q_rows, :num_blocks])
        return block_scores

    max_pages = int(metadata.real_page_table.shape[1])
    use_page_max = os.getenv(
        "FLASHINFER_EXP_SM12X_MSA_DECODE_PAGEMAX", "1"
    ) != "0" and uses_paged_mqa_schedule(q_rows=valid_q_rows, max_pages=max_pages)
    if use_page_max:
        weights = q_scale[:valid_q_rows].contiguous() * MSA_SM_SCALE
        seqlens_valid = metadata.cache_seqlens_int32.contiguous()
        if active_width_override is None:
            active_width = _make_active_width_tensor(
                seqlens_per_query=seqlens_valid,
                width=width_tokens,
            )
        else:
            active_width = active_width_override
        schedule_metadata = metadata.paged_mqa_schedule_metadata
        if schedule_metadata is None:
            if _is_cuda_graph_capture_active(q_fp8.device):
                raise ValueError(
                    "paged_mqa_schedule_metadata must be precomputed before CUDA graph capture "
                    "for MSA scheduled page-max decode"
                )
            schedule_metadata = build_paged_mqa_schedule_metadata(
                seqlens_valid, page_size
            )
        page_scores = run_paged_logits_kernel(
            q_fp8=q_fp8[:valid_q_rows],
            weights=weights,
            index_k_cache=index_k_cache,
            real_page_table=metadata.real_page_table,
            seqlens_per_query=seqlens_valid,
            schedule_metadata=schedule_metadata,
            active_width=active_width,
            page_size=page_size,
            preinitialize_invalid_logits=True,
            score_mode=IndexerScoreMode.MSA_BILINEAR,
            output_mode=IndexerOutputMode.PAGE_HEAD_MAX,
            page_scores=page_scores_scratch,
        )
        paired = page_scores.view(num_heads, valid_q_rows, -1, 2).amax(dim=3)
        block_scores[:, :valid_q_rows, :].copy_(paired[:, :, :num_blocks])
        return block_scores

    q_expanded = (
        q_fp8[:valid_q_rows]
        .contiguous()
        .reshape(
            valid_q_rows * num_heads,
            1,
            _INDEX_HEAD_DIM,
        )
    )
    weights = (
        q_scale[:valid_q_rows].contiguous().reshape(valid_q_rows * num_heads, 1)
        * MSA_SM_SCALE
    )
    metadata_expanded = IndexerPagedDecodeMetadata(
        real_page_table=metadata.real_page_table.repeat_interleave(num_heads, dim=0),
        cache_seqlens_int32=metadata.cache_seqlens_int32.repeat_interleave(num_heads),
        paged_mqa_schedule_metadata=None,
    )
    token_logits = paged_decode_logits(
        q_fp8=q_expanded,
        weights=weights,
        index_k_cache=index_k_cache,
        metadata=metadata_expanded,
        page_size=page_size,
        preinitialize_invalid_logits=True,
        score_mode=IndexerScoreMode.MSA_BILINEAR,
    )
    padded_width = num_blocks * MSA_BLOCK_TOKENS
    if padded_width != width_tokens:
        pooled_input = torch.full(
            (valid_q_rows * num_heads, padded_width),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
        pooled_input[:, :width_tokens].copy_(token_logits)
    else:
        pooled_input = token_logits
    pooled = (
        pooled_input.view(valid_q_rows, num_heads, num_blocks, MSA_BLOCK_TOKENS)
        .amax(dim=3)
        .permute(1, 0, 2)
        .contiguous()
    )
    block_scores[:, :valid_q_rows, :].copy_(pooled)
    return block_scores


def msa_q2k_indices_decode(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    index_k_cache: torch.Tensor,
    metadata: IndexerPagedDecodeMetadata | None = None,
    topk: int = MSA_TOPK_BLOCKS,
    out_indices: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    """Run MSA decode score, block max-pool, local forcing, and top-k selection."""

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("out_indices", out_indices),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "MSA paged binding owns metadata and q2k output; do not also pass "
                f"{', '.join(extras)}"
            )
        metadata = binding.metadata
        if getattr(binding, "topk", None) is not None:
            topk = int(binding.topk)
        out_indices = getattr(binding, "q2k_indices", None)
    if metadata is None:
        raise TypeError("msa_q2k_indices_decode requires metadata or binding")
    q_rows, num_heads = _validate_msa_q_inputs(q_fp8=q_fp8, q_scale=q_scale)
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    if out_indices is not None:
        if out_indices.dtype != torch.int32:
            raise ValueError(
                f"out_indices must have dtype torch.int32, got {out_indices.dtype}"
            )
        if out_indices.device != q_fp8.device:
            raise ValueError("out_indices device must match q_fp8")
        if (
            out_indices.ndim != 3
            or out_indices.shape[0] < num_heads
            or out_indices.shape[1] < q_rows
            or out_indices.shape[2] < topk
        ):
            raise ValueError(
                f"out_indices must have shape at least ({num_heads}, {q_rows}, {topk}), "
                f"got {tuple(out_indices.shape)}"
            )
    valid_q_rows = int(metadata.real_page_table.shape[0])
    if valid_q_rows > q_rows:
        raise ValueError(
            f"metadata describes {valid_q_rows} q rows, but q_fp8 has {q_rows}"
        )
    block_scores = msa_paged_decode_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        index_k_cache=index_k_cache,
        metadata=None if binding is not None else metadata,
        binding=binding,
    )
    if out_indices is None:
        result = torch.full(
            (num_heads, q_rows, topk),
            -1,
            dtype=torch.int32,
            device=q_fp8.device,
        )
    else:
        result = out_indices[:num_heads, :q_rows, :topk]
        result.fill_(-1)
    if valid_q_rows == 0:
        return result
    selected = msa_topk_blocks(
        block_scores=block_scores[:, :valid_q_rows, :],
        query_positions=msa_decode_query_positions(metadata.cache_seqlens_int32),
        topk=topk,
        out_indices=result[:, :valid_q_rows, :],
        score_scratch=getattr(binding, "topk_score_scratch", None)
        if binding is not None
        else None,
        top_values=getattr(binding, "topk_values", None)
        if binding is not None
        else None,
        top_indices=getattr(binding, "topk_indices", None)
        if binding is not None
        else None,
        sort_values=getattr(binding, "sort_values", None)
        if binding is not None
        else None,
        sort_indices=getattr(binding, "sort_indices", None)
        if binding is not None
        else None,
    )
    if selected.data_ptr() != result[:, :valid_q_rows, :].data_ptr():
        result[:, :valid_q_rows, :].copy_(selected)
    return result


def contiguous_logits(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    preinitialize_invalid_logits: bool = True,
    tile_logits: torch.Tensor | None = None,
    score_mode: int = IndexerScoreMode.NSA_RELU_SUM,
    binding=None,
) -> torch.Tensor:
    strict_binding = False
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("tile_logits", tile_logits),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "indexer contiguous binding owns metadata, scratch, "
                f"and tile logits; do not also pass {', '.join(extras)}"
            )
        strict_binding = bool(getattr(binding, "strict", False))
        metadata = binding.metadata
        tile_logits = binding.tile_logits
    if metadata is None:
        raise TypeError("contiguous_logits requires metadata or binding")
    score_mode = int(score_mode)
    if score_mode not in (IndexerScoreMode.NSA_RELU_SUM, IndexerScoreMode.MSA_BILINEAR):
        raise ValueError(f"unsupported indexer score_mode {score_mode}")
    k_start = metadata.k_start
    k_end = metadata.k_end
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if q_fp8.shape[2] != _INDEX_HEAD_DIM:
        raise ValueError(
            f"q_fp8 head_dim must be {_INDEX_HEAD_DIM}, got {q_fp8.shape[2]}"
        )
    _normalize_weights(weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1])
    if k_start.ndim != 1 or k_end.ndim != 1:
        raise ValueError(
            f"k_start and k_end must be rank-1, got {tuple(k_start.shape)} and {tuple(k_end.shape)}"
        )
    if k_start.shape != k_end.shape:
        raise ValueError(
            f"k_start and k_end must have the same shape, got {tuple(k_start.shape)} vs {tuple(k_end.shape)}"
        )
    if k_start.device != q_fp8.device or k_end.device != q_fp8.device:
        raise ValueError("k_start and k_end must be on the same device as q_fp8")

    weights_f = _normalize_weights(
        weights,
        q_rows=q_fp8.shape[0],
        num_heads=q_fp8.shape[1],
        require_float32=strict_binding,
    )
    k_quant, k_scale = kv_fp8
    if supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        result = run_contiguous_logits_kernel(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=preinitialize_invalid_logits,
            tile_logits=tile_logits,
            score_mode=score_mode,
        )
        return result

    if score_mode != IndexerScoreMode.NSA_RELU_SUM:
        raise NotImplementedError(
            "contiguous_logits score_mode=MSA_BILINEAR requires the CUDA FP8 scorer contract"
        )

    return contiguous_logits_reference(
        q_fp8=q_fp8,
        weights=weights_f,
        kv_fp8=kv_fp8,
        k_start=k_start,
        k_end=k_end,
    )


def msa_contiguous_block_scores(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    out: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    """Score MSA prefill candidates and max-pool over global 128-token K blocks."""

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("out", out),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "MSA contiguous binding owns metadata and block-score buffers; do not also pass "
                f"{', '.join(extras)}"
            )
        metadata = binding.metadata
        out = getattr(binding, "block_scores", None)
    if metadata is None:
        raise TypeError("msa_contiguous_block_scores requires metadata or binding")
    q_rows, num_heads = _validate_msa_q_inputs(q_fp8=q_fp8, q_scale=q_scale)
    k_start = metadata.k_start
    k_end = metadata.k_end
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        raise ValueError(
            "MSA contiguous metadata requires matching rank-1 k_start/k_end"
        )
    if k_start.shape[0] > q_rows:
        raise ValueError(
            f"metadata describes {k_start.shape[0]} q rows, but q_fp8 has {q_rows}"
        )
    if k_start.dtype != torch.int32 or k_end.dtype != torch.int32:
        raise ValueError("k_start and k_end must have dtype torch.int32")
    if not _is_cuda_graph_capture_active(q_fp8.device):
        misaligned = torch.remainder(k_start, MSA_BLOCK_TOKENS) != 0
        if bool(torch.any(misaligned).item()):
            raise ValueError("MSA contiguous k_start values must be 128-token aligned")
    k_quant, k_scale = kv_fp8
    k_rows = int(k_quant.shape[0])
    num_blocks = math.ceil(k_rows / MSA_BLOCK_TOKENS)
    out_view = _validate_msa_block_scores_out(
        out=out,
        num_heads=num_heads,
        q_rows=q_rows,
        num_blocks=num_blocks,
        device=q_fp8.device,
    )
    if out_view is None:
        block_scores = torch.full(
            (num_heads, q_rows, num_blocks),
            float("-inf"),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    else:
        block_scores = out_view
        block_scores.fill_(float("-inf"))
    valid_q_rows = int(k_start.shape[0])
    if valid_q_rows == 0 or k_rows == 0:
        return block_scores

    if q_fp8.device.type != "cuda":
        ref = msa_contiguous_block_scores_reference(
            q_fp8=q_fp8,
            q_scale=q_scale,
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
        block_scores.copy_(ref[:, :q_rows, :num_blocks])
        return block_scores

    weights = q_scale.contiguous() * MSA_SM_SCALE
    kernel_kwargs = {}
    if binding is not None and bool(getattr(binding, "strict", False)):
        scratch = binding.scratch
        if not hasattr(scratch, "prepare_k_padding"):
            raise RuntimeError(
                "strict MSA contiguous binding requires plan-owned scratch"
            )
        if not q_fp8.is_contiguous():
            raise ValueError("strict MSA contiguous binding requires contiguous q_fp8")
        if not k_quant.is_contiguous() or not k_scale.is_contiguous():
            raise ValueError(
                "strict MSA contiguous binding requires contiguous K tensors"
            )
        scratch.prepare_k_padding(k_rows=k_rows)
        if k_quant.data_ptr() != scratch.k_quant.data_ptr():
            raise ValueError("strict MSA contiguous K values must be a scratch prefix")
        if k_scale.data_ptr() != scratch.k_scale.data_ptr():
            raise ValueError("strict MSA contiguous K scales must be a scratch prefix")
        q_bytes = q_fp8.view(torch.uint8)
        kernel_kwargs.update(
            q_bytes=q_bytes,
            q_u32=q_bytes.view(torch.uint32).view(
                int(q_fp8.shape[0]),
                int(q_fp8.shape[1]),
                _INDEX_HEAD_DIM // 4,
            ),
            weights_kernel=weights.contiguous(),
            k_quant_bytes=scratch.k_quant.view(torch.uint8),
            k_scale_kernel=scratch.k_scale,
            k_start_kernel=k_start,
            k_end_kernel=k_end,
            out_kernel=scratch.dummy_logits,
            tile_logits_kernel=scratch.tile_logits,
            k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
        )
    return run_contiguous_block_scores_kernel(
        q_fp8=q_fp8,
        weights=weights,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
        block_scores=block_scores,
        num_blocks_out=num_blocks,
        **kernel_kwargs,
    )


def msa_q2k_indices_prefill(
    *,
    q_fp8: torch.Tensor,
    q_scale: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    query_positions: torch.Tensor | None = None,
    block_base: torch.Tensor | None = None,
    topk: int = MSA_TOPK_BLOCKS,
    out_indices: torch.Tensor | None = None,
    binding=None,
) -> torch.Tensor:
    """Run MSA prefill score, block max-pool, local forcing, and top-k selection."""

    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("out_indices", out_indices),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "MSA contiguous binding owns metadata and q2k output; do not also pass "
                f"{', '.join(extras)}"
            )
        metadata = binding.metadata
        if getattr(binding, "topk", None) is not None:
            topk = int(binding.topk)
        out_indices = getattr(binding, "q2k_indices", None)
    if metadata is None:
        raise TypeError("msa_q2k_indices_prefill requires metadata or binding")
    q_rows, num_heads = _validate_msa_q_inputs(q_fp8=q_fp8, q_scale=q_scale)
    valid_q_rows = int(metadata.k_start.shape[0])
    if valid_q_rows > q_rows:
        raise ValueError(
            f"metadata describes {valid_q_rows} q rows, but q_fp8 has {q_rows}"
        )
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    if out_indices is not None:
        if out_indices.dtype != torch.int32:
            raise ValueError(
                f"out_indices must have dtype torch.int32, got {out_indices.dtype}"
            )
        if out_indices.device != q_fp8.device:
            raise ValueError("out_indices device must match q_fp8")
        if (
            out_indices.ndim != 3
            or out_indices.shape[0] < num_heads
            or out_indices.shape[1] < q_rows
            or out_indices.shape[2] < topk
        ):
            raise ValueError(
                f"out_indices must have shape at least ({num_heads}, {q_rows}, {topk}), "
                f"got {tuple(out_indices.shape)}"
            )
    block_scores = msa_contiguous_block_scores(
        q_fp8=q_fp8,
        q_scale=q_scale,
        kv_fp8=kv_fp8,
        metadata=None if binding is not None else metadata,
        binding=binding,
    )
    if out_indices is None:
        result = torch.full(
            (num_heads, q_rows, topk),
            -1,
            dtype=torch.int32,
            device=q_fp8.device,
        )
    else:
        result = out_indices[:num_heads, :q_rows, :topk]
        result.fill_(-1)
    if valid_q_rows == 0:
        return result
    if query_positions is None:
        query_positions = metadata.k_end - 1
    else:
        query_positions = query_positions[:valid_q_rows]
    if block_base is None:
        block_base = torch.div(
            metadata.k_start, MSA_BLOCK_TOKENS, rounding_mode="floor"
        )
    else:
        block_base = block_base[:valid_q_rows]
    selected = msa_topk_blocks(
        block_scores=block_scores[:, :valid_q_rows, :],
        query_positions=query_positions,
        block_base=block_base,
        topk=topk,
        out_indices=result[:, :valid_q_rows, :],
        score_scratch=getattr(binding, "topk_score_scratch", None)
        if binding is not None
        else None,
        top_values=getattr(binding, "topk_values", None)
        if binding is not None
        else None,
        top_indices=getattr(binding, "topk_indices", None)
        if binding is not None
        else None,
        sort_values=getattr(binding, "sort_values", None)
        if binding is not None
        else None,
        sort_indices=getattr(binding, "sort_indices", None)
        if binding is not None
        else None,
    )
    if selected.data_ptr() != result[:, :valid_q_rows, :].data_ptr():
        result[:, :valid_q_rows, :].copy_(selected)
    return result


def _reference_topk_indices_from_logits(
    logits: torch.Tensor,
    *,
    topk: int,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
) -> torch.Tensor:
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    num_rows = int(logits.shape[0])
    result = torch.full((num_rows, topk), -1, dtype=torch.int32, device=logits.device)
    values = torch.full(
        (num_rows, topk), float("-inf"), dtype=torch.float32, device=logits.device
    )
    gather_k = min(topk, int(logits.shape[1]))
    if gather_k:
        topk_pos = torch.argsort(logits, dim=1, descending=True, stable=True)[
            :, :gather_k
        ]
        topk_values = torch.gather(logits, 1, topk_pos)
        result[:, :gather_k] = torch.where(
            torch.isfinite(topk_values),
            topk_pos.to(torch.int32),
            torch.full_like(topk_pos, -1, dtype=torch.int32),
        )
        values[:, :gather_k] = topk_values

    if output_indices is not None:
        if output_indices.dtype != torch.int32:
            raise ValueError(
                f"output_indices must have dtype torch.int32, got {output_indices.dtype}"
            )
        if output_indices.device != logits.device:
            raise ValueError("output_indices device must match logits")
        if (
            output_indices.ndim != 2
            or output_indices.shape[0] < num_rows
            or output_indices.shape[1] < topk
        ):
            raise ValueError(
                f"output_indices must have shape at least ({num_rows}, {topk}), got {tuple(output_indices.shape)}"
            )
        output_indices[:num_rows, :topk].copy_(result)
        result = output_indices[:num_rows, :topk]

    if output_values is not None:
        if output_values.dtype != torch.float32:
            raise ValueError(
                f"output_values must have dtype torch.float32, got {output_values.dtype}"
            )
        if output_values.device != logits.device:
            raise ValueError("output_values device must match logits")
        if (
            output_values.ndim != 2
            or output_values.shape[0] < num_rows
            or output_values.shape[1] < topk
        ):
            raise ValueError(
                f"output_values must have shape at least ({num_rows}, {topk}), got {tuple(output_values.shape)}"
            )
        output_values[:num_rows, :topk].copy_(values)

    return result


def contiguous_tiled_topk(
    *,
    q_fp8: torch.Tensor,
    weights: torch.Tensor,
    kv_fp8: tuple[torch.Tensor, torch.Tensor],
    metadata: IndexerContiguousMetadata | None = None,
    topk: int | None = None,
    tile_logits: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
    output_values: torch.Tensor | None = None,
    output_indices: torch.Tensor | None = None,
    candidate_values: torch.Tensor | None = None,
    candidate_indices: torch.Tensor | None = None,
    merge_positions: torch.Tensor | None = None,
    supertile_k: int | None = None,
    binding=None,
) -> torch.Tensor:
    """Run the prefill NSA scorer in K-supertiles and consume each tile with tiled topk."""

    strict_binding = False
    if binding is not None:
        extras = [
            name
            for name, value in (
                ("metadata", metadata),
                ("tile_logits", tile_logits),
                ("lengths", lengths),
                ("output_values", output_values),
                ("output_indices", output_indices),
                ("candidate_values", candidate_values),
                ("candidate_indices", candidate_indices),
                ("merge_positions", merge_positions),
            )
            if value is not None
        ]
        if extras:
            raise ValueError(
                "indexer contiguous binding owns metadata, scratch, "
                "and top-k scratch buffers; do not also pass "
                f"{', '.join(extras)}"
            )
        if topk is None:
            topk = binding.topk
        elif binding.topk is not None and int(topk) != int(binding.topk):
            raise ValueError(
                f"topk {int(topk)} does not match bound topk {int(binding.topk)}"
            )
        strict_binding = bool(getattr(binding, "strict", False))
        metadata = binding.metadata
        tile_logits = binding.tile_logits
        lengths = binding.lengths
        output_values = binding.output_values
        output_indices = binding.output_indices
        candidate_values = binding.candidate_values
        candidate_indices = binding.candidate_indices
        merge_positions = binding.merge_positions
        if binding.supertile_k is not None:
            supertile_k = int(binding.supertile_k)
    if metadata is None:
        raise TypeError("contiguous_tiled_topk requires metadata or binding")
    if topk is None:
        raise TypeError("contiguous_tiled_topk requires topk or a binding with topk")
    topk = int(topk)
    if topk < 0:
        raise ValueError(f"topk must be non-negative, got {topk}")
    k_start = metadata.k_start
    k_end = metadata.k_end
    if q_fp8.ndim != 3:
        raise ValueError(f"q_fp8 must be rank-3, got {tuple(q_fp8.shape)}")
    if k_start.ndim != 1 or k_end.ndim != 1 or k_start.shape != k_end.shape:
        raise ValueError(
            "tiled topk requires matching rank-1 k_start and k_end tensors"
        )
    weights_f = _normalize_weights(
        weights, q_rows=q_fp8.shape[0], num_heads=q_fp8.shape[1]
    )
    k_quant, k_scale = kv_fp8
    if not supports_contiguous_logits_kernel(
        q_fp8=q_fp8,
        weights=weights_f,
        k_quant=k_quant,
        k_scale=k_scale,
        k_start=k_start,
        k_end=k_end,
    ):
        if strict_binding:
            raise RuntimeError(
                "strict indexer contiguous binding requires the CUDA FP8 scorer "
                "contract; reference logits fallback is disabled"
            )
        if lengths is not None:
            if lengths.ndim != 1 or lengths.shape[0] < int(k_start.shape[0]):
                raise ValueError(
                    f"lengths must have shape at least ({int(k_start.shape[0])},), got {tuple(lengths.shape)}"
                )
            if lengths.dtype != torch.int32:
                raise ValueError(
                    f"lengths must have dtype torch.int32, got {lengths.dtype}"
                )
            if lengths.device != q_fp8.device:
                raise ValueError(
                    f"lengths device {lengths.device} does not match q_fp8 device {q_fp8.device}"
                )
            torch.sub(k_end, k_start, out=lengths[: int(k_start.shape[0])])
        logits = contiguous_logits_reference(
            q_fp8=q_fp8,
            weights=weights_f,
            kv_fp8=kv_fp8,
            k_start=k_start,
            k_end=k_end,
        )
        return _reference_topk_indices_from_logits(
            logits[: int(k_start.shape[0])],
            topk=topk,
            output_values=output_values,
            output_indices=output_indices,
        )
    prefill_block_k = (
        int(binding.prefill_block_k)
        if binding is not None and binding.prefill_block_k is not None
        else resolve_contiguous_prefill_block_k(
            valid_q_rows=int(k_start.shape[0]),
            k_rows=int(k_quant.shape[0]),
            num_heads=int(q_fp8.shape[1]),
        )
    )
    if prefill_block_k is None:
        # This API explicitly requests tiled logits for immediate tiled top-k.
        # The decode scorer does not produce that layout, so force the standard
        # prefill scorer for small q batches instead of failing.
        prefill_block_k = _PREFILL_BLOCK_K
    block_q = (
        _PREFILL512_BLOCK_Q
        if prefill_block_k == _PREFILL512_BLOCK_K
        else _PREFILL_BLOCK_Q
    )

    num_q_rows = int(k_start.shape[0])
    num_q_tiles = (num_q_rows + block_q - 1) // block_q
    num_k_tiles = (int(k_quant.shape[0]) + prefill_block_k - 1) // prefill_block_k
    tile_size = block_q * prefill_block_k
    resolved_supertile_k = _resolve_supertile_k(supertile_k, block_k=prefill_block_k)
    supertile_tiles = max(1, resolved_supertile_k // prefill_block_k)
    num_chunks = (num_k_tiles + supertile_tiles - 1) // supertile_tiles
    max_chunk_tiles = min(supertile_tiles, num_k_tiles)
    chunk_tile_elements = num_q_tiles * max_chunk_tiles * tile_size

    if tile_logits is None:
        if strict_binding:
            raise RuntimeError(
                "strict indexer contiguous binding is missing tile_logits"
            )
        tile_logits = torch.empty(
            (chunk_tile_elements,),
            dtype=torch.float32,
            device=q_fp8.device,
        )
    elif int(tile_logits.numel()) < chunk_tile_elements:
        raise ValueError(
            f"tile_logits has {int(tile_logits.numel())} elements, expected at least "
            f"{chunk_tile_elements} for the largest K-supertile"
        )

    if lengths is None:
        if strict_binding:
            raise RuntimeError("strict indexer contiguous binding is missing lengths")
        global_lengths = (k_end - k_start).contiguous()
    else:
        if lengths.ndim != 1 or lengths.shape[0] < num_q_rows:
            raise ValueError(
                f"lengths must have shape at least ({num_q_rows},), got {tuple(lengths.shape)}"
            )
        if lengths.dtype != torch.int32:
            raise ValueError(
                f"lengths must have dtype torch.int32, got {lengths.dtype}"
            )
        if lengths.device != q_fp8.device:
            raise ValueError(
                f"lengths device {lengths.device} does not match q_fp8 device {q_fp8.device}"
            )
        if not lengths.is_contiguous():
            raise ValueError("lengths must be contiguous")
        global_lengths = lengths[:num_q_rows]
        torch.sub(k_end, k_start, out=global_lengths)
    if strict_binding and (output_values is None or output_indices is None):
        raise RuntimeError(
            "strict indexer contiguous binding is missing output top-k buffers"
        )

    def _run_contiguous_scorer(
        *,
        tile_k_offset: int,
        tile_num_k_tiles: int,
    ) -> None:
        if not strict_binding:
            run_contiguous_logits_kernel(
                q_fp8=q_fp8,
                weights=weights_f,
                k_quant=k_quant,
                k_scale=k_scale,
                k_start=k_start,
                k_end=k_end,
                preinitialize_invalid_logits=True,
                tile_logits=tile_logits,
                tile_k_offset=tile_k_offset,
                tile_num_k_tiles=tile_num_k_tiles,
                prefill_block_k=prefill_block_k,
            )
            return

        if binding is None:
            raise RuntimeError("strict indexer contiguous path requires a binding")
        scratch = binding.scratch
        if tile_logits is None:
            raise RuntimeError(
                "strict indexer contiguous binding is missing tile logits"
            )
        if not hasattr(scratch, "prepare_k_padding"):
            raise RuntimeError(
                "strict indexer contiguous binding requires plan-owned scratch"
            )
        if not q_fp8.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous q_fp8")
        if not weights_f.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous weights")
        if not k_quant.is_contiguous() or not k_scale.is_contiguous():
            raise ValueError("strict indexer contiguous requires contiguous K tensors")
        scratch.prepare_k_padding(k_rows=int(k_quant.shape[0]))
        scratch_k_quant = scratch.k_quant
        scratch_k_scale = scratch.k_scale
        if k_quant.data_ptr() != scratch_k_quant.data_ptr():
            raise ValueError(
                "strict indexer contiguous K values must be a scratch prefix"
            )
        if k_scale.data_ptr() != scratch_k_scale.data_ptr():
            raise ValueError(
                "strict indexer contiguous K scales must be a scratch prefix"
            )
        q_bytes = q_fp8.view(torch.uint8)
        q_u32 = q_bytes.view(torch.uint32).view(
            int(q_fp8.shape[0]),
            int(q_fp8.shape[1]),
            _INDEX_HEAD_DIM // 4,
        )
        kernel_binding = build_indexer_contiguous_logits_kernel_binding(
            q_fp8=q_fp8,
            weights=weights_f,
            k_quant=k_quant,
            k_scale=k_scale,
            k_start=k_start,
            k_end=k_end,
            preinitialize_invalid_logits=True,
            tile_logits=tile_logits,
            tile_k_offset=tile_k_offset,
            tile_num_k_tiles=tile_num_k_tiles,
            prefill_block_k=prefill_block_k,
            q_u32=q_u32,
            q_bytes=q_bytes,
            weights_kernel=weights_f,
            k_quant_bytes=scratch_k_quant.view(torch.uint8),
            k_scale_kernel=scratch_k_scale,
            k_start_kernel=k_start,
            k_end_kernel=k_end,
            out_kernel=scratch.dummy_logits,
            out_view=scratch.dummy_logits,
            k_tma_desc_ptrs=scratch.k_tma_desc_ptrs,
            k_tma_prefill_desc_ptrs=scratch.k_tma_prefill_desc_ptrs,
        )
        run_contiguous_logits_kernel(binding=kernel_binding)

    if num_chunks <= 1:
        # Dead tiles (entirely out of causal/length range) are left UNWRITTEN by
        # the tiled-output contiguous kernel (it `pass`es, trusting run_tiled_topk's
        # k_start/k_end mask). Pre-clear to -inf so any stale value in those slots
        # of the (reused) scratch can never win the tiled top-k.
        tile_logits[:chunk_tile_elements].fill_(float("-inf"))
        _run_contiguous_scorer(
            tile_k_offset=0,
            tile_num_k_tiles=num_k_tiles,
        )
        _, topk_indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=output_values,
            output_indices=output_indices,
            num_k_tiles=num_k_tiles,
        )
        return topk_indices

    # Streaming fold over K-supertiles: each chunk folds the previous chunk's running
    # top-k (carry) into its own radix selection, so the final chunk's output is the
    # exact global top-k. The reused candidate_values/candidate_indices buffers serve
    # as a (2, M, topk) carry double-buffer (read prev half, write next half); the
    # final chunk writes the user output. No (num_chunks, ...) slab, no merge.
    if (candidate_values is None) != (candidate_indices is None):
        raise ValueError(
            "candidate_values and candidate_indices must be provided together"
        )
    if candidate_values is None:
        if strict_binding:
            raise RuntimeError(
                "strict indexer contiguous binding is missing carry buffers"
            )
        candidate_values = torch.empty(
            (2, num_q_rows, topk),
            dtype=torch.float32,
            device=q_fp8.device,
        )
        candidate_indices = torch.empty(
            (2, num_q_rows, topk),
            dtype=torch.int32,
            device=q_fp8.device,
        )
    else:
        assert candidate_indices is not None
        if candidate_values.ndim != 3 or candidate_indices.ndim != 3:
            raise ValueError(
                f"carry buffers must have shape at least (2, {num_q_rows}, {topk})"
            )
        if candidate_values.shape[0] < 2 or candidate_values.shape[1] < num_q_rows:
            raise ValueError(
                "candidate_values shape "
                f"{tuple(candidate_values.shape)} is smaller than required "
                f"(2, {num_q_rows}, {topk})"
            )
        if candidate_indices.shape[0] < 2 or candidate_indices.shape[1] < num_q_rows:
            raise ValueError(
                "candidate_indices shape "
                f"{tuple(candidate_indices.shape)} is smaller than required "
                f"(2, {num_q_rows}, {topk})"
            )
        if candidate_values.shape[2] != topk or candidate_indices.shape[2] != topk:
            raise ValueError(
                "carry buffer top-k dimension must match requested topk "
                f"{topk}, got {candidate_values.shape[2]} and {candidate_indices.shape[2]}"
            )
        if candidate_values.dtype != torch.float32:
            raise ValueError(
                f"candidate_values must have dtype torch.float32, got {candidate_values.dtype}"
            )
        if candidate_indices.dtype != torch.int32:
            raise ValueError(
                f"candidate_indices must have dtype torch.int32, got {candidate_indices.dtype}"
            )
        if (
            candidate_values.device != q_fp8.device
            or candidate_indices.device != q_fp8.device
        ):
            raise ValueError("carry buffer devices must match q_fp8")
    carry_buf_values = candidate_values[:2, :num_q_rows, :]
    carry_buf_indices = candidate_indices[:2, :num_q_rows, :]

    topk_indices = output_indices
    for chunk_idx in range(num_chunks):
        chunk_tile_begin = chunk_idx * supertile_tiles
        chunk_tile_end = min(chunk_tile_begin + supertile_tiles, num_k_tiles)
        chunk_tiles = chunk_tile_end - chunk_tile_begin
        chunk_start = chunk_tile_begin * prefill_block_k
        chunk_rows = chunk_tiles * prefill_block_k
        # tile_logits is reused across chunks; dead tiles are not rewritten by the
        # kernel, so a stale logit from a previous chunk at the same offset could
        # otherwise survive into this chunk's tiled top-k. Pre-clear to -inf.
        tile_logits[: num_q_tiles * chunk_tiles * tile_size].fill_(float("-inf"))
        _run_contiguous_scorer(
            tile_k_offset=chunk_tile_begin,
            tile_num_k_tiles=chunk_tiles,
        )
        is_first = chunk_idx == 0
        is_last = chunk_idx == num_chunks - 1
        carry_values = carry_buf_values[(chunk_idx - 1) % 2]
        carry_indices = carry_buf_indices[(chunk_idx - 1) % 2]
        out_v = output_values if is_last else carry_buf_values[chunk_idx % 2]
        out_i = output_indices if is_last else carry_buf_indices[chunk_idx % 2]
        _, topk_indices = run_tiled_topk(
            tile_logits=tile_logits,
            k_start=k_start,
            lengths=global_lengths,
            topk=topk,
            block_q=block_q,
            block_k=prefill_block_k,
            output_values=out_v,
            output_indices=out_i,
            num_k_tiles=chunk_tiles,
            input_index_offset=chunk_start,
            input_extent=chunk_rows,
            output_index_offset=chunk_start,
            carry_values=carry_values,
            carry_indices=carry_indices,
            is_first=is_first,
        )
    return topk_indices
