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

import functools
from typing import Optional

import torch

from .jit.sparse_scores import gen_sparse_scores_module
from .utils import register_custom_op, register_fake_op


# The scorer multiplies with m16n8k16, which arrived with sm_80.
_SPARSE_SCORES_MIN_CAPABILITY = (8, 0)


@functools.cache
def get_sparse_scores_module():
    major, minor = torch.cuda.get_device_capability()
    if (major, minor) < _SPARSE_SCORES_MIN_CAPABILITY:
        raise RuntimeError(
            "sparse_paged_scores needs compute capability "
            f"{_SPARSE_SCORES_MIN_CAPABILITY[0]}.{_SPARSE_SCORES_MIN_CAPABILITY[1]} "
            f"or newer for its tensor-core path, got {major}.{minor}"
        )
    return gen_sparse_scores_module().build_and_load()


@register_custom_op(
    "flashinfer::sparse_paged_scores", mutates_args=("visible_blocks", "logits")
)
def _sparse_paged_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits: torch.Tensor,
    compress_ratio: int,
    divisor: float,
) -> None:
    get_sparse_scores_module().sparse_paged_scores(
        q,
        k_cache,
        page_table,
        token_to_req,
        query_positions,
        sequence_lengths,
        visible_blocks,
        logits,
        compress_ratio,
        divisor,
    )


@register_fake_op("flashinfer::sparse_paged_scores")
def _sparse_paged_scores_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits: torch.Tensor,
    compress_ratio: int,
    divisor: float,
) -> None:
    pass


def sparse_paged_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    page_table: torch.Tensor,
    token_to_req: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    compress_ratio: int,
    divisor: float,
    num_columns: Optional[int] = None,
    logits: Optional[torch.Tensor] = None,
    visible_blocks: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""Score every visible KV entry of a paged cache against a multi-head query.

    The logits a sparse-attention selector ranks by:

    .. math::
        \mathrm{score}(row, col) =
            \frac{1}{d} \sum_h \max\bigl(0, K[col] \cdot Q[row, h]\bigr)

    There is no softmax and no value aggregation -- this is the input to a top-k,
    not an attention output.

    Entries on a page the block table does not map come out as ``-inf`` so a
    top-k never selects them. Columns past what the query can see are left
    untouched instead, and the count of what it can see is returned, so a top-k
    has to bound itself by that count rather than by the column width.

    Parameters
    ----------
    q : torch.Tensor
        Queries, shape ``[rows, num_heads, head_dim]``, float16 or bfloat16.
        ``head_dim`` must be 64, 128, 192 or 256 and ``num_heads`` at most 16.
    k_cache : torch.Tensor
        Paged keys, shape ``[num_pages, page_size, head_dim]``, same dtype as ``q``.
    page_table : torch.Tensor
        Logical page to physical page per request, shape ``[num_requests, table_width]``.
        A negative entry marks an unmapped page.
    token_to_req : torch.Tensor
        Request each row belongs to, shape ``[rows]``. A negative entry empties the row.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    sequence_lengths : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    compress_ratio : int
        Tokens each cache entry stands for. A query sees only the entries whose
        tokens are all behind it.
    divisor : float
        Scale applied to the summed score, typically ``sqrt(head_dim)``.
    num_columns : Optional[int]
        Entries to score. Defaults to what the page table can address.
    logits : Optional[torch.Tensor]
        Output scores, shape ``[rows, num_columns]``, float32. Allocated when omitted.
        Columns past a row's visible count are left untouched.
    visible_blocks : Optional[torch.Tensor]
        Receives the visible entry count per row, shape ``[rows]``. Allocated when
        omitted.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The scores and the per-row visible count.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [rows, heads, head_dim], got {q.ndim}D")
    if k_cache.ndim != 3:
        raise ValueError(
            f"k_cache must be [pages, page_size, head_dim], got {k_cache.ndim}D"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if q.shape[1] > 16:
        raise ValueError(
            f"sparse_paged_scores handles at most 16 query heads, got {q.shape[1]}"
        )
    rows = q.shape[0]
    columns = (
        page_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
    )
    if logits is None:
        logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    if visible_blocks is None:
        visible_blocks = torch.empty(rows, dtype=page_table.dtype, device=q.device)
    if rows and not columns:
        # No column to score, but the caller still reads the visible counts.
        visible_blocks.zero_()
    if rows and columns:
        _sparse_paged_scores(
            q,
            k_cache,
            page_table,
            token_to_req,
            query_positions,
            sequence_lengths,
            visible_blocks,
            logits,
            compress_ratio,
            divisor,
        )
    return logits, visible_blocks
