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

import torch

from .jit.sparse_pre_indexer import gen_sparse_pre_indexer_module
from .utils import register_custom_op, register_fake_op


@functools.cache
def get_sparse_pre_indexer_module():
    return gen_sparse_pre_indexer_module().build_and_load()


@register_custom_op(
    "flashinfer::qsa_pre_indexer",
    mutates_args=("q_out", "state_cache", "compressed_cache"),
)
def _qsa_pre_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    work_metadata: torch.Tensor,
    compress_ratio: int,
    mrope_h: int,
    mrope_w: int,
    is_k_mrope: bool,
    cache_has_rope_pos: bool,
) -> None:
    get_sparse_pre_indexer_module().qsa_pre_indexer(
        q,
        k,
        positions,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        state_cache,
        state_slots,
        state_block_table,
        query_start_loc,
        logical_positions,
        compressed_cache,
        compressed_slots,
        work_metadata,
        compress_ratio,
        mrope_h,
        mrope_w,
        is_k_mrope,
        cache_has_rope_pos,
    )


@register_fake_op("flashinfer::qsa_pre_indexer")
def _qsa_pre_indexer_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    work_metadata: torch.Tensor,
    compress_ratio: int,
    mrope_h: int,
    mrope_w: int,
    is_k_mrope: bool,
    cache_has_rope_pos: bool,
) -> None:
    pass


def qsa_pre_indexer(
    q: torch.Tensor,
    k: torch.Tensor,
    positions: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    q_norm_weight: torch.Tensor,
    k_norm_weight: torch.Tensor,
    eps: float,
    q_out: torch.Tensor,
    state_cache: torch.Tensor,
    state_slots: torch.Tensor,
    state_block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    logical_positions: torch.Tensor,
    compressed_cache: torch.Tensor,
    compressed_slots: torch.Tensor,
    work_metadata: torch.Tensor,
    compress_ratio: int,
    mrope_h: int = 0,
    mrope_w: int = 0,
    is_k_mrope: bool = False,
    cache_has_rope_pos: bool = False,
) -> None:
    r"""Build the query and compressed-key rows a sparse route is scored on.

    One launch does two things. Every query row is RMS-normalised with a
    stored-plus-one weight and rotated by a partial NeoX RoPE -- the first half
    of the row turns, pairing each element with the one a quarter-row away.
    Every completed compression group is mean-pooled from its raw keys,
    normalised, rotated the same way and written to the compressed cache; the
    keys a group needs may reach back past the start of this step's chunk, into
    the per-request ring, and the first work item of a request also commits that
    request's raw-key suffix to that ring for the next step to read.

    The result of this is what :func:`sparse_paged_scores` scores.

    Parameters
    ----------
    q : torch.Tensor
        Queries, shape ``[tokens, num_heads * head_dim]``, float16 or bfloat16.
        ``head_dim`` must be 128 or 256.
    k : torch.Tensor
        Raw keys, shape ``[tokens, head_dim]``, same dtype as ``q``.
    positions : torch.Tensor
        Rotary coordinates, int64. Shape ``[tokens]``, or ``[3, tokens]`` for the
        temporal, height and width axes of an interleaved three-axis rope.
    cos_sin_cache : torch.Tensor
        Pair-major rotary table, shape ``[max_position, head_dim // 2]``: a row
        is cosine and sine interleaved so that one pair is a single access.
    q_norm_weight, k_norm_weight : torch.Tensor
        Norm weights of one row, held as the offset from one.
    eps : float
        Added to the mean square before the reciprocal square root.
    q_out : torch.Tensor
        Receives the normalised, rotated queries, shape
        ``[tokens, num_heads, head_dim]``.
    state_cache : torch.Tensor
        The per-request ring of raw keys, shape ``[blocks, ring, 1, width]``.
        ``width`` is ``head_dim``, plus room for three int64 coordinates after it
        when ``cache_has_rope_pos``.
    state_slots : torch.Tensor
        Ring slot each token commits to, int64, shape ``[tokens]``. Negative to skip.
    state_block_table : torch.Tensor
        Ring block of each request, int32, shape ``[num_requests, 1]``.
    query_start_loc : torch.Tensor
        Token each request starts at, int32, shape ``[num_requests + 1]``.
    logical_positions : torch.Tensor
        Position of each token in its request's whole sequence, int64.
    compressed_cache : torch.Tensor
        Receives the pooled keys, shape ``[blocks, page, 1, head_dim]``.
    compressed_slots : torch.Tensor
        Compressed slot each group boundary writes to, int64, shape ``[tokens]``.
    work_metadata : torch.Tensor
        One row per compression group, int32, shape ``[num_work, 2]``: the request
        and the index of the group within it. A negative request skips the row.
    compress_ratio : int
        Raw keys pooled into one compressed entry.
    mrope_h, mrope_w : int
        Section widths of the height and width axes of a three-axis rope.
    is_k_mrope : bool
        Whether the compressed key picks its rotary axis per pair.
    cache_has_rope_pos : bool
        Whether the ring keeps each row's rotary coordinates after its elements,
        which is what lets a group that reaches into it rotate by the right ones.

    Notes
    -----
    The query picks its axis per pair exactly when ``positions`` carries three of
    them, so three-axis positions with ``is_k_mrope=False`` is not a supported
    configuration.
    """
    if q.ndim != 2:
        raise ValueError(f"q must be [tokens, heads * head_dim], got {q.ndim}D")
    if q_out.ndim != 3:
        raise ValueError(f"q_out must be [tokens, heads, head_dim], got {q_out.ndim}D")
    head_dim = q_out.shape[2]
    if head_dim not in (128, 256):
        raise ValueError(f"qsa_pre_indexer builds head_dim 128 or 256, got {head_dim}")
    if compress_ratio < 1:
        raise ValueError(f"compress_ratio must be positive, got {compress_ratio}")
    if positions.ndim == 2 and not is_k_mrope:
        raise ValueError("three-axis positions need a three-axis key")
    if q.shape[0] == 0:
        return
    _qsa_pre_indexer(
        q,
        k,
        positions,
        cos_sin_cache,
        q_norm_weight,
        k_norm_weight,
        eps,
        q_out,
        state_cache,
        state_slots,
        state_block_table,
        query_start_loc,
        logical_positions,
        compressed_cache,
        compressed_slots,
        work_metadata,
        compress_ratio,
        mrope_h,
        mrope_w,
        is_k_mrope,
        cache_has_rope_pos,
    )
