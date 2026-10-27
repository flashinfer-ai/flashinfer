"""Lightweight dispatch contract for the QSA CuTe sparse GQA core."""

from __future__ import annotations

import torch

from ..paged._selected_forward_config import (
    HEAD_DIM,
    MAX_SPLITS as NUM_SPLITS,
    SELECTION_WIDTH,
)

BLOCK_N = 16
MAX_SPLIT_ROWS = 64


def _is_page_token_head_layout(tensor: torch.Tensor) -> bool:
    """Accept a strided outer page with non-overlapping inner cache rows."""
    if tensor.ndim != 4 or int(tensor.stride(3)) != 1:
        return False
    _, page_size, kv_heads, head_dim = map(int, tensor.shape)
    page_stride, token_stride, head_stride, _ = map(int, tensor.stride())
    if min(page_stride, token_stride, head_stride) <= 0:
        return False
    head_span = head_dim
    token_span = (kv_heads - 1) * head_stride + head_span
    page_span = (page_size - 1) * token_stride + token_span
    return (
        head_stride >= head_span
        and token_stride >= token_span
        and page_stride >= page_span
    )


def is_qwen_geometry(
    *,
    q_heads: int,
    kv_heads: int,
    head_dim: int,
    selection_width: int,
    block_n: int,
    splits: int,
) -> bool:
    """Return whether scalar dimensions select the Qwen sparse-GQA kernel."""
    splits = int(splits)
    return (
        int(q_heads) > 0
        and int(kv_heads) > 0
        and int(q_heads) % int(kv_heads) == 0
        and int(head_dim) == HEAD_DIM
        and int(selection_width) >= SELECTION_WIDTH
        and int(block_n) == BLOCK_N
        and splits > 0
        and splits <= NUM_SPLITS
        and splits & (splits - 1) == 0
    )


def is_candidate(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    block_table: torch.Tensor,
    request_ids: torch.Tensor,
    selected_positions: torch.Tensor,
    query_positions: torch.Tensor,
    partial_output: torch.Tensor,
    partial_lse: torch.Tensor,
    block_n: int,
    splits: int,
) -> bool:
    """Validate the supported geometry without importing the CuTe kernel."""
    if (
        query.ndim != 3
        or key_cache.ndim != 4
        or value_cache.ndim != 4
        or block_table.ndim != 2
        or request_ids.ndim != 1
        or selected_positions.ndim != 2
        or query_positions.ndim != 1
        or partial_output.ndim != 4
        or partial_lse.ndim != 3
    ):
        return False
    rows, q_heads, head_dim = map(int, query.shape)
    head_layout = (q_heads, int(key_cache.shape[2]))
    if not (
        rows > 0
        and is_qwen_geometry(
            q_heads=q_heads,
            kv_heads=head_layout[1],
            head_dim=head_dim,
            selection_width=int(selected_positions.shape[1]),
            block_n=block_n,
            splits=splits,
        )
        and int(key_cache.shape[0]) > 0
        and int(key_cache.shape[3]) == HEAD_DIM
        and int(selected_positions.shape[0]) >= rows
    ):
        return False
    if not query.is_cuda:
        return False
    if (
        query.dtype != torch.bfloat16
        or key_cache.dtype not in (torch.bfloat16, torch.float8_e4m3fn)
        or value_cache.dtype != key_cache.dtype
        or block_table.dtype != torch.int32
        or request_ids.dtype not in (torch.int32, torch.int64)
        or selected_positions.dtype != torch.int32
        or query_positions.dtype != torch.int64
        or partial_output.dtype != torch.float32
        or partial_lse.dtype != torch.float32
    ):
        return False
    contiguous_tensors = (
        query,
        block_table,
        request_ids,
        selected_positions,
        query_positions,
        partial_output,
        partial_lse,
    )
    return (
        all(
            tensor.device == query.device and tensor.is_contiguous()
            for tensor in contiguous_tensors
        )
        and key_cache.device == query.device
        and value_cache.device == query.device
        and tuple(value_cache.shape) == tuple(key_cache.shape)
        and _is_page_token_head_layout(key_cache)
        and _is_page_token_head_layout(value_cache)
        and tuple(block_table.shape)[0] > 0
        and tuple(request_ids.shape) == (rows,)
        and tuple(query_positions.shape) == (rows,)
        and int(partial_output.shape[0]) >= rows
        and tuple(partial_output.shape[1:]) == (int(splits), q_heads, HEAD_DIM)
        and int(partial_lse.shape[0]) >= rows
        and tuple(partial_lse.shape[1:]) == (int(splits), q_heads)
    )


__all__ = [
    "BLOCK_N",
    "HEAD_DIM",
    "MAX_SPLIT_ROWS",
    "NUM_SPLITS",
    "SELECTION_WIDTH",
    "is_candidate",
    "is_qwen_geometry",
]
