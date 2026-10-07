"""Shared internal MLA utilities without backend dependencies."""

from dataclasses import dataclass
from typing import Optional

import torch

from ..utils import _check_block_tables_shape


def _round_to_seq_len_bucket(x: int) -> int:
    """Power-of-2 bucket for max_seq_len in autotune cache keys.

    Collapses small variations across deployments so they can share cache
    entries (e.g., max_seq_len=16385 and 32768 hash to the same bucket).
    """
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


@dataclass(frozen=True)
class MLAHeadDimensions:
    """
    The dimensions of a single MLA head.

    Args:
        qk_nope_head_dim (int): The number of input channels without positional information in non-absorb mode.
        qk_rope_head_dim (int): The number of channels carrying positional information for both absorb and non-absorb modes.
        v_head_dim (int): The number of value channels, which is also the output head dimension in non-absorb mode.
        kv_lora_rank (int): The dimension of the compressed key-value representation across heads.
    """

    qk_nope_head_dim: int
    qk_rope_head_dim: int
    v_head_dim: int
    kv_lora_rank: int


deepseek_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=128,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
)

smaller_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=256,
)

compact_query_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=64,
    qk_rope_head_dim=64,
    v_head_dim=128,
    kv_lora_rank=512,
)

nope_mla_dimensions = MLAHeadDimensions(
    qk_nope_head_dim=256,
    qk_rope_head_dim=0,
    v_head_dim=256,
    kv_lora_rank=512,
)

supported_mla_head_dimensions = [
    deepseek_mla_dimensions,
    smaller_mla_dimensions,
    compact_query_mla_dimensions,
    nope_mla_dimensions,
]

# Preserve the public type identity after moving its implementation.
MLAHeadDimensions.__module__ = "flashinfer.mla._core"


def _normalize_mla_kv_cache(kv_cache: torch.Tensor) -> torch.Tensor:
    """Expose the native KV head axis without copying caller storage."""
    if kv_cache.ndim == 3:
        return kv_cache.unsqueeze(1)
    if kv_cache.ndim != 4:
        raise ValueError(f"Expected kv_cache.ndim == 3 or 4, got {kv_cache.ndim}")
    return kv_cache


def _check_mla_query_kv_shape(
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    kv_lora_rank: int,
    qk_rope_head_dim: int,
    batch_size: Optional[int] = None,
    max_q_len: Optional[int] = None,
) -> torch.Tensor:
    """Validate common query/KV shapes and expose 4D KV storage."""
    if query.ndim == 4:
        qk_head_dim = query.shape[-1]
    elif query.ndim == 3:
        if batch_size is None or max_q_len is None:
            raise ValueError(
                "batch_size and max_q_len are required when query.ndim == 3"
            )
        _, _, qk_head_dim = query.shape
    else:
        raise ValueError(f"Expected query.ndim == 3 or 4, got {query.ndim}")

    kv_cache = _normalize_mla_kv_cache(kv_cache)

    is_deepseek_dimensions = (
        kv_lora_rank == deepseek_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == deepseek_mla_dimensions.qk_rope_head_dim
    )
    is_smaller_mla_dimensions = (
        kv_lora_rank == smaller_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == smaller_mla_dimensions.qk_rope_head_dim
    )
    is_nope_mla_dimensions = (
        kv_lora_rank == nope_mla_dimensions.kv_lora_rank
        and qk_rope_head_dim == nope_mla_dimensions.qk_rope_head_dim
    )
    if not (
        is_deepseek_dimensions or is_smaller_mla_dimensions or is_nope_mla_dimensions
    ):
        raise ValueError(
            f"Unsupported MLA dimensions, got kv_lora_rank={kv_lora_rank} and qk_rope_head_dim={qk_rope_head_dim}, supported dimensions are: {supported_mla_head_dimensions}"
        )

    ckv_dim = kv_cache.shape[3]
    expected_qk_head_dim = kv_lora_rank + qk_rope_head_dim
    if qk_head_dim != expected_qk_head_dim or ckv_dim != expected_qk_head_dim:
        raise ValueError(
            f"Expected head dim {expected_qk_head_dim} for query and kv_cache, got {qk_head_dim} and {ckv_dim}"
        )

    return kv_cache


def _check_mla_dense_page_table_shape(
    page_table: torch.Tensor,
    num_seqs: int,
    page_size: int,
    uses_shared_paged_kv_idx: bool,
    require_aligned_block_table: bool,
) -> None:
    """Validate dense page mappings with the caller's layout/alignment policy."""
    _check_block_tables_shape(page_table, uses_shared_paged_kv_idx)
    B_block_table = page_table.shape[0]
    block_num = page_table.shape[-1]
    block_size = page_size
    if num_seqs != B_block_table:
        raise ValueError(
            f"Expected batch size {num_seqs} for query and block_table, got {num_seqs} and {B_block_table}"
        )
    if require_aligned_block_table and block_num % (128 / block_size) != 0:
        raise ValueError(
            f"Expected block_num % (128 / block_size) == 0, got {block_num=} and {block_size=}"
        )


def _check_mla_sparse_index_shape(
    query: torch.Tensor, page_table: torch.Tensor, sparse_mla_top_k: int
) -> None:
    """Validate sparse indices for an already validated uniform or flat query."""
    page_table_shape = page_table.shape
    expected_page_table_shape = (
        (query.size(0), sparse_mla_top_k)
        if query.ndim == 3
        else (query.size(0), query.size(1), sparse_mla_top_k)
    )
    if page_table_shape != expected_page_table_shape:
        raise ValueError(
            "Expected page_table.shape == "
            f"{expected_page_table_shape}" + f", got {page_table_shape}"
        )
