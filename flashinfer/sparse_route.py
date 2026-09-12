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

from .jit.sparse_route import gen_sparse_route_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_sparse_route_module():
    return gen_sparse_route_module().build_and_load()


@register_custom_op("flashinfer::expand_block_route", mutates_args=("out",))
def _expand_block_route(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    compress_ratio: int,
) -> None:
    get_sparse_route_module().expand_block_route(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        out,
        compress_ratio,
    )


@register_fake_op("flashinfer::expand_block_route")
def _expand_block_route_fake(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    out: torch.Tensor,
    compress_ratio: int,
) -> None:
    pass


def expand_block_route(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
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
    block_indices : torch.Tensor
        Selected block ids, shape ``[rows, block_topk]``, int32 or int64.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    sequence_lengths : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    token_to_req : torch.Tensor
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
    >>> token_to_req = torch.tensor([0], dtype=torch.int32, device="cuda")
    >>> flashinfer.expand_block_route(blocks, positions, seq_lens, token_to_req, 4)
    tensor([[4, 5, 6, 7, 0, 1, 2, 3, 8, 9, -1]], device='cuda:0', dtype=torch.int32)

    Block 2 is the one the query at position 9 sits in, so selecting it drops those
    four positions instead:

    >>> blocks = torch.tensor([[2, 0]], dtype=torch.int32, device="cuda")
    >>> flashinfer.expand_block_route(blocks, positions, seq_lens, token_to_req, 4)
    tensor([[-1, -1, -1, -1, 0, 1, 2, 3, 8, 9, -1]], device='cuda:0', dtype=torch.int32)
    """
    if block_indices.ndim != 2:
        raise ValueError(
            f"block_indices must be 2D [rows, block_topk], got {block_indices.ndim}D"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    rows, block_topk = block_indices.shape
    width = block_topk * compress_ratio + compress_ratio - 1
    if out is None:
        out = torch.empty(
            (rows, width), dtype=block_indices.dtype, device=block_indices.device
        )
    elif tuple(out.shape) != (rows, width):
        raise ValueError(f"out must have shape {(rows, width)}, got {tuple(out.shape)}")
    if rows:
        _expand_block_route(
            block_indices,
            query_positions,
            sequence_lengths,
            token_to_req,
            out,
            compress_ratio,
        )
    return out


@register_custom_op(
    "flashinfer::qsa_route_from_blocks",
    mutates_args=("out_logical", "out_route", "out_mask"),
)
def _qsa_route_from_blocks(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out_logical: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    num_slots: int,
) -> None:
    get_sparse_route_module().qsa_route_from_blocks(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
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
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out_logical: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    compress_ratio: int,
    page_size: int,
    num_slots: int,
) -> None:
    pass


def qsa_route_from_blocks(
    block_indices: torch.Tensor,
    query_positions: torch.Tensor,
    sequence_lengths: torch.Tensor,
    token_to_req: torch.Tensor,
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
    selected blocks into tokens (see :func:`expand_block_route`), mapping each token
    through the block table into a physical KV slot, and packing per-entry validity
    into the bitmask a block-sparse attention reads.

    An entry is valid when it names a real token: inside the request, on a logical
    page the block table covers, on a page the table maps, and in a slot the cache
    holds. Invalid entries route to slot 0 with their mask bit clear -- an
    out-of-range slot would be read before the mask applies.

    The logical route is written out as well, for callers that reuse a selection
    across steps after the physical route derived from it has been consumed.

    Parameters
    ----------
    block_indices : torch.Tensor
        Selected block ids, shape ``[rows, block_topk]``, int32 or int64.
    query_positions : torch.Tensor
        Position of each query inside its request, shape ``[rows]``.
    sequence_lengths : torch.Tensor
        KV length of each request, shape ``[num_requests]``.
    token_to_req : torch.Tensor
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
    if block_indices.ndim != 2:
        raise ValueError(
            f"block_indices must be 2D [rows, block_topk], got {block_indices.ndim}D"
        )
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if block_indices.shape[0] == 0:
        return
    _qsa_route_from_blocks(
        block_indices,
        query_positions,
        sequence_lengths,
        token_to_req,
        block_table,
        out_logical,
        out_route,
        out_mask,
        compress_ratio,
        page_size,
        num_slots,
    )


@register_custom_op(
    "flashinfer::qsa_route_from_logical", mutates_args=("out_route", "out_mask")
)
def _qsa_route_from_logical(
    logical: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
) -> None:
    get_sparse_route_module().qsa_route_from_logical(
        logical,
        token_to_req,
        block_table,
        out_route,
        out_mask,
        valid_rows,
        page_size,
        num_slots,
    )


@register_fake_op("flashinfer::qsa_route_from_logical")
def _qsa_route_from_logical_fake(
    logical: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
) -> None:
    pass


def qsa_route_from_logical(
    logical: torch.Tensor,
    token_to_req: torch.Tensor,
    block_table: torch.Tensor,
    out_route: torch.Tensor,
    out_mask: torch.Tensor,
    valid_rows: int,
    page_size: int,
    num_slots: int,
) -> None:
    r"""Map a logical token route through a block table into physical KV slots.

    The second half of :func:`qsa_route_from_blocks`, for callers whose logical route
    was produced earlier and outlived the physical one -- a speculative decoder reuses
    a selection across its steps.

    An entry is valid when it names a real token: non-negative, on a logical page the
    block table covers, on a page the table maps, and in a slot the cache holds.
    Invalid entries route to slot 0 with their mask bit clear, since an out-of-range
    slot would be read before the mask applies. Route rows at or past ``valid_rows``
    are padding and come out fully masked.

    Parameters
    ----------
    logical : torch.Tensor
        Logical token route, shape ``[>= valid_rows, width]``. Rows past
        ``valid_rows`` are never read.
    token_to_req : torch.Tensor
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
    """
    if out_route.ndim != 2:
        raise ValueError(f"out_route must be 2D [rows, width], got {out_route.ndim}D")
    if page_size < 1:
        raise ValueError(f"page_size must be positive, got {page_size}")
    if out_route.shape[0] == 0:
        return
    _qsa_route_from_logical(
        logical,
        token_to_req,
        block_table,
        out_route,
        out_mask,
        valid_rows,
        page_size,
        num_slots,
    )
