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

from ..trace.templates.qsa import qsa_paged_scores_trace
from ..api_logging import flashinfer_api
from ..jit.qsa_ops import gen_qsa_ops_module
from ..utils import (
    backend_requirement,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)


@functools.cache
def get_qsa_scores_module():
    return gen_qsa_ops_module().build_and_load()


# The scorer multiplies with m16n8k16, which arrived with SM80.
@supported_compute_capability([80, 86, 87, 89, 90, 100, 103, 107, 110, 120, 121])
def _check_qsa_paged_scores(*args, **kwargs) -> bool:
    return True


@register_custom_op(
    "flashinfer::qsa_paged_scores", mutates_args=("visible_blocks", "logits")
)
def _qsa_paged_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits: torch.Tensor,
    compress_ratio: int,
    divisor: float,
) -> None:
    get_qsa_scores_module().qsa_paged_scores(
        q,
        k_cache,
        block_table,
        token_to_request,
        query_positions,
        seq_lens,
        visible_blocks,
        logits,
        compress_ratio,
        divisor,
    )


@register_fake_op("flashinfer::qsa_paged_scores")
def _qsa_paged_scores_fake(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    visible_blocks: torch.Tensor,
    logits: torch.Tensor,
    compress_ratio: int,
    divisor: float,
) -> None:
    pass


@flashinfer_api(trace=qsa_paged_scores_trace)
@backend_requirement(backend_checks={}, common_check=_check_qsa_paged_scores)
def qsa_paged_scores(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
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
    untouched instead, and the number of entries actually scored is returned --
    what the query can see, capped by the column width -- so a top-k bounds
    itself by that count rather than by the width.

    Parameters
    ----------
    q : torch.Tensor
        Queries, shape ``[rows, num_heads, head_dim]``, float16 or bfloat16.
        ``head_dim`` must be 64, 128, 192 or 256 and ``num_heads`` at most 16.
    k_cache : torch.Tensor
        Paged keys, shape ``[num_pages, page_size, head_dim]``, same dtype as ``q``.
    block_table : torch.Tensor
        Logical page to physical page per request, shape ``[num_requests, table_width]``.
        A negative entry marks an unmapped page.
    token_to_request : torch.Tensor
        Request each row belongs to, shape ``[rows]``. A negative entry empties the row.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    seq_lens : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    compress_ratio : int
        Tokens each cache entry stands for. A query sees only the entries whose
        tokens are all behind it.
    divisor : float
        Scale applied to the summed score, typically ``sqrt(head_dim)``.
    num_columns : Optional[int]
        Entries to score. Defaults to what the block table can address. When
        ``logits`` is given as well its width has to match, since the kernel
        takes the width from the tensor it writes.
    logits : Optional[torch.Tensor]
        Output scores, shape ``[rows, num_columns]``, float32. Allocated when omitted.
        Columns past a row's visible count are left untouched.
    visible_blocks : Optional[torch.Tensor]
        Receives the number of entries actually scored for each row, shape
        ``[rows]`` -- what the query can see, capped by the column width.
        Allocated when omitted.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        The scores, and the per-row count of entries scored.
    """
    if q.ndim != 3:
        raise ValueError(f"q must be [rows, heads, head_dim], got {q.ndim}D")
    if k_cache.ndim != 3:
        raise ValueError(
            f"k_cache must be [pages, page_size, head_dim], got {k_cache.ndim}D"
        )
    if k_cache.shape[1] < 1:
        # A column is divided by this to reach its page. With no pages the
        # kernel has nothing to read and says so, but a page of no entries is
        # a divisor of zero, which it cannot.
        raise ValueError(
            f"k_cache pages hold {k_cache.shape[1]} entries; a page holds at least one"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if divisor <= 0:
        raise ValueError(f"divisor must be positive, got {divisor}")
    if q.shape[1] > 16:
        raise ValueError(
            f"qsa_paged_scores handles at most 16 query heads, got {q.shape[1]}"
        )
    rows = q.shape[0]
    columns = (
        block_table.shape[1] * k_cache.shape[1] if num_columns is None else num_columns
    )
    if logits is None:
        logits = torch.empty((rows, columns), dtype=torch.float32, device=q.device)
    elif logits.shape[1] != columns:
        # The kernel takes the width from the tensor it writes, so a narrower
        # num_columns alongside a wider logits would silently score the wider
        # one and report a count past what the caller asked for.
        raise ValueError(
            f"logits is {logits.shape[1]} columns wide but {columns} were "
            "asked for; pass a view of the width you want scored"
        )
    if visible_blocks is None:
        visible_blocks = torch.empty(rows, dtype=block_table.dtype, device=q.device)
    if rows and not columns:
        # No column to score, but the caller still reads the visible counts.
        visible_blocks.zero_()
    if rows and columns:
        _qsa_paged_scores(
            q,
            k_cache,
            block_table,
            token_to_request,
            query_positions,
            seq_lens,
            visible_blocks,
            logits,
            compress_ratio,
            divisor,
        )
    return logits, visible_blocks
