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

from ..api_logging import flashinfer_api
from ..trace.templates.qsa import (
    qsa_expand_block_route_trace,
    qsa_route_from_blocks_trace,
    qsa_route_from_logical_trace,
)
from ..jit.qsa_ops import gen_qsa_route_module
from ..utils import register_custom_op, register_fake_op


@functools.cache
def get_qsa_route_module():
    return gen_qsa_route_module().build_and_load()


@register_custom_op("flashinfer::qsa_expand_block_route", mutates_args=("out",))
def _qsa_expand_block_route(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    out: torch.Tensor,
    compress_ratio: int,
) -> None:
    get_qsa_route_module().qsa_expand_block_route(
        indexer_block_ids,
        query_positions,
        seq_lens,
        token_to_request,
        out,
        compress_ratio,
    )


@register_fake_op("flashinfer::qsa_expand_block_route")
def _qsa_expand_block_route_fake(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    out: torch.Tensor,
    compress_ratio: int,
) -> None:
    pass


@flashinfer_api(trace=qsa_expand_block_route_trace)
def qsa_expand_block_route(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    compress_ratio: int,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Expand a per-query list of selected blocks into the token route it stands for.

    A block-granular selector picks ``block_topk`` blocks of ``compress_ratio`` tokens
    each. Attention works on tokens, so every selected block becomes its
    ``compress_ratio`` tokens, in selection order.

    The block a query sits in is only partially in the past, so it is never expanded
    whole; its already-seen tokens are appended after the expanded blocks instead. That
    tail is at most ``compress_ratio - 1`` tokens, which fixes the route width at
    ``block_topk * compress_ratio + compress_ratio - 1``.

    The selection is the caller's, so this is enforced rather than assumed: a selected
    block the query has not passed is dropped whole, all ``compress_ratio`` of its
    positions. Keeping only its seen tokens would repeat exactly what the tail appends.

    Positions no token reaches are written as ``-1``, for the consumer to mask.

    Parameters
    ----------
    indexer_block_ids : torch.Tensor
        Selected block ids, shape ``[rows, block_topk]``, int32 or int64.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    seq_lens : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    token_to_request : torch.Tensor
        Request each row belongs to, shape ``[rows]``. A negative entry empties the row.
    compress_ratio : int
        Tokens per block.
    out : Optional[torch.Tensor]
        Route to write, shape ``[rows, block_topk * compress_ratio + compress_ratio - 1]``.
        Allocated when omitted.

    Returns
    -------
    torch.Tensor
        The token route, ``-1`` padded.

    Examples
    --------
    >>> import torch
    >>> import flashinfer
    >>> blocks = torch.tensor([[1, 0]], dtype=torch.int32, device="cuda")
    >>> positions = torch.tensor([9], dtype=torch.int32, device="cuda")
    >>> seq_lens = torch.tensor([16], dtype=torch.int32, device="cuda")
    >>> token_to_request = torch.tensor([0], dtype=torch.int32, device="cuda")
    >>> flashinfer.qsa_ops.qsa_expand_block_route(blocks, positions, seq_lens, token_to_request, 4)
    tensor([[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]], device='cuda:0', dtype=torch.int32)

    Block 2 is the one the query at position 9 sits in, so selecting it drops those
    four positions instead:

    >>> blocks = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    >>> flashinfer.qsa_ops.qsa_expand_block_route(blocks, positions, seq_lens, token_to_request, 4)
    tensor([[-1, -1, -1, -1, 0, 1, 2, 3, 8, 9, -1]], device='cuda:0', dtype=torch.int32)
    """
    if indexer_block_ids.ndim != 2:
        raise ValueError(
            f"indexer_block_ids must be 2D [rows, block_topk], got {indexer_block_ids.ndim}D"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    rows, block_topk = indexer_block_ids.shape
    width = block_topk * compress_ratio + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (rows, width),
            dtype=indexer_block_ids.dtype,
            device=indexer_block_ids.device,
        )
    elif tuple(out.shape) != (rows, width):
        raise ValueError(f"out must have shape {(rows, width)}, got {tuple(out.shape)}")
    if rows:
        _qsa_expand_block_route(
            indexer_block_ids,
            query_positions,
            seq_lens,
            token_to_request,
            out,
            compress_ratio,
        )
    return out


@register_custom_op(
    "flashinfer::qsa_route_from_blocks",
    mutates_args=("out_logical", "out_route", "out_mask"),
)
def _qsa_route_from_blocks(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_logical: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    num_slots: int,
) -> None:
    get_qsa_route_module().qsa_route_from_blocks(
        indexer_block_ids,
        query_positions,
        seq_lens,
        token_to_request,
        block_table,
        out_logical,
        out_route,
        out_mask,
        compress_ratio,
        page_size,
        num_slots,
    )


@register_fake_op("flashinfer::qsa_route_from_blocks")
def _qsa_route_from_blocks_fake(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_logical: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    num_slots: int,
) -> None:
    pass


@flashinfer_api(trace=qsa_route_from_blocks_trace)
def qsa_route_from_blocks(
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    seq_lens: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_logical: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    num_slots: int,
) -> None:
    r"""Turn a per-query block selection straight into a paged attention route.

    Fuses what would otherwise be three passes over the same route: expanding the
    selected blocks into tokens (see :func:`qsa_expand_block_route`), mapping each token
    through the block table into a physical KV slot, and packing per-entry validity
    into the bitmask a block-sparse attention reads.

    An entry is valid when it names a real token: inside the request, on a logical
    page the block table covers, on a page the table maps, and in a slot the cache
    holds. Invalid entries keep their mask bit clear and route to the slot of the
    row's first valid entry: they are read before the mask applies, and a masked entry
    still meets its V row with a zero weight, so the slot has to hold finite values.
    Slot 0 is the caller's padding and need not. A row with no valid entry -- one
    without a request, or one that sees nothing -- is fully masked on slot 0, and its
    attention output is undefined: the caller must not keep it.

    The logical route is written out as well, for callers that reuse a selection
    across steps after the physical route derived from it has been consumed.

    Parameters
    ----------
    indexer_block_ids : torch.Tensor
        Selected block ids, shape ``[rows, block_topk]``, int32 or int64.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    seq_lens : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    token_to_request : torch.Tensor
        Request each row belongs to, shape ``[rows]``. A negative entry empties the row.
    block_table : torch.Tensor
        Logical page to physical page per request, shape ``[num_requests, table_width]``.
        A negative entry marks an unmapped page.
    out_logical : torch.Tensor
        Receives the logical token route, shape ``[>= rows, width]``.
    out_route : torch.Tensor
        Receives the physical slot route, shape ``[rows, width]``, contiguous.
    out_mask : torch.Tensor
        Receives the packed validity, ``ceil(width / 8)`` uint8 per row, contiguous.
    compress_ratio : int
        Tokens per block.
    page_size : int
        KV entries per physical page.
    num_slots : int
        Total KV entries the cache holds.

    Notes
    -----
    ``width`` is ``block_topk * compress_ratio + compress_ratio - 1``: every selected
    block expands to ``compress_ratio`` tokens, and the query's own block contributes
    at most ``compress_ratio - 1`` already-seen tokens.
    """
    if indexer_block_ids.ndim != 2:
        raise ValueError(
            f"indexer_block_ids must be 2D [rows, block_topk], got {indexer_block_ids.ndim}D"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if indexer_block_ids.shape[0] == 0:
        return
    _qsa_route_from_blocks(
        indexer_block_ids,
        query_positions,
        seq_lens,
        token_to_request,
        block_table,
        out_logical,
        out_route,
        out_mask,
        compress_ratio,
        page_size,
        num_slots,
    )


@register_custom_op(
    "flashinfer::qsa_route_from_logical",
    mutates_args=("out_route", "out_mask", "out_indptr"),
)
def _qsa_route_from_logical(
    logical: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
    out_indptr: Optional[torch.Tensor],
) -> None:
    get_qsa_route_module().qsa_route_from_logical(
        logical,
        token_to_request,
        block_table,
        out_route,
        out_mask,
        valid_rows,
        page_size,
        num_slots,
        out_indptr,
    )


@register_fake_op("flashinfer::qsa_route_from_logical")
def _qsa_route_from_logical_fake(
    logical: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
    out_indptr: Optional[torch.Tensor],
) -> None:
    pass


@flashinfer_api(trace=qsa_route_from_logical_trace)
def qsa_route_from_logical(
    logical: torch.Tensor,
    token_to_request: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
    out_indptr: Optional[torch.Tensor] = None,
) -> None:
    r"""Map a logical token route through a block table into physical KV slots.

    The second half of :func:`qsa_route_from_blocks`, for callers whose logical route
    was produced earlier and outlived the physical one -- a speculative decoder reuses
    a selection across its steps.

    An entry is valid when it names a real token: non-negative, on a logical page the
    block table covers, on a page the table maps, and in a slot the cache holds.
    Invalid entries keep their mask bit clear and route to the slot of the row's first
    valid entry, for the reason :func:`qsa_route_from_blocks` gives. A row with no
    valid entry -- padding at or past ``valid_rows``, a row without a request, or one
    whose route names nothing the table maps -- is fully masked on slot 0, and its
    attention output is undefined: the caller must not keep it.

    Parameters
    ----------
    logical : torch.Tensor
        Logical token route, shape ``[>= valid_rows, width]``. Rows past
        ``valid_rows`` are never read.
    token_to_request : torch.Tensor
        Request each live row belongs to, at least ``valid_rows`` entries.
    block_table : torch.Tensor
        Logical page to physical page per request, shape ``[num_requests, table_width]``.
        A negative entry marks an unmapped page.
    out_route : torch.Tensor
        Receives the physical slot route, shape ``[rows, width]``, contiguous.
    out_mask : torch.Tensor
        Receives the packed validity, ``ceil(width / 8)`` uint8 per row, contiguous.
    valid_rows : int
        Rows that carry a real query; the rest are masked off.
    page_size : int
        KV entries per physical page.
    num_slots : int
        Total KV entries the cache holds.
    out_indptr : Optional[torch.Tensor]
        Receives the row pointers of a block-sparse plan over this route, ``rows + 1``
        int32: ``min(r, valid_rows) * width``. The rows past ``valid_rows`` are given
        no entries, so the plan's work for them reads nothing instead of attending
        over a route that is all masked.
    """
    if out_route.ndim != 2:
        raise ValueError(f"out_route must be 2D [rows, width], got {out_route.ndim}D")
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if out_route.shape[0] == 0 and out_indptr is None:
        return
    _qsa_route_from_logical(
        logical,
        token_to_request,
        block_table,
        out_route,
        out_mask,
        valid_rows,
        page_size,
        num_slots,
        out_indptr,
    )
