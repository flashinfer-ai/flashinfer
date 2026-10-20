# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 by FlashInfer team.

"""Single-head paged MQA logits, for sparse attention whose indexer scores keys
with one query head and no head weights."""

import torch

from ..api_logging import flashinfer_api
from ..cute_dsl.availability import is_cute_dsl_experimental_available
from ..trace.templates.attn_scores import single_head_paged_mqa_logits_trace
from ..utils import get_compute_capability, get_device_sm_count


@flashinfer_api(trace=single_head_paged_mqa_logits_trace)
def single_head_paged_mqa_logits(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    block_table: torch.Tensor,
    context_lengths: torch.Tensor,
    *,
    requests: torch.Tensor,
    scores: torch.Tensor,
) -> None:
    """Compute FP32 scores of query rows against their request's paged keys.

    This is :func:`fp8_paged_mqa_logits` for an indexer with one query head.
    With one head there is no head weight and no ReLU, so a score is the plain
    dot product of a query row and a key. Query rows come in groups of 1 to 8
    consecutive rows, such as a decode step's token and its draft tokens.
    It runs as one launch without host synchronization, so it can be captured in a
    CUDA graph (run it once before capture to compile it). Queries, keys, pages and lengths can change between replays.

    Parameters
    ----------
    query : torch.Tensor
        BF16 ``[rows, 128]``.
    key_cache : torch.Tensor
        BF16 ``[pages, 64, 128]``. Pages can be any multiple of 16 bytes apart.
    block_table : torch.Tensor
        int32 ``[requests, blocks]``: each request's pages of 64 keys. Only pages
        below a group's length are read.
    context_lengths : torch.Tensor
        int32 ``[groups]``: each group's number of keys.
    requests : torch.Tensor
        int32 ``[groups]``: each group's request, a row of ``block_table``. A
        group of length 0 reads nothing, so its entry can hold any value.
    scores : torch.Tensor
        float32 ``[rows, max_context]``, output: ``scores[q, k]`` is the dot
        product of query row ``q`` with key ``k``, for ``k`` below its group's
        length. Later entries are not written. ``max_context`` sizes the grid,
        and the block table must cover it.
    """
    if not is_cute_dsl_experimental_available():
        raise RuntimeError("single_head_paged_mqa_logits needs CuTe DSL 4.7 or newer")
    from ..cute_dsl.launch import argument_kind, compiled
    from .kernels import single_head_scorer
    from .kernels.single_head_scorer import (
        CTAS_PER_SM,
        DIM,
        MAX_ROWS,
        PAGE,
        TILE,
        TILES_PER_CTA,
    )

    max_context = scores.shape[1]
    batch, width = requests.shape[0], block_table.shape[1]
    if min(batch, max_context, key_cache.shape[0], width) <= 0:
        raise ValueError("queries, keys, scores and the block table must be nonempty")
    rows = scores.shape[0] // batch
    if not 1 <= rows <= MAX_ROWS or rows * batch != scores.shape[0]:
        raise ValueError("each request must serve 1 to 8 query rows")
    if query.shape[0] != scores.shape[0] or context_lengths.shape[0] != batch:
        raise ValueError("Mismatched query rows, scores, lengths or requests")
    if max_context > PAGE * width:
        raise ValueError("The block table must cover the context capacity")
    if {query.dtype, key_cache.dtype} != {torch.bfloat16} or query.shape[1] != DIM:
        raise ValueError("Queries and keys must be BF16 of dimension 128")
    if key_cache.shape[1:] != (PAGE, DIM) or key_cache.stride()[1:] != (DIM, 1):
        raise ValueError("Keys must be [pages, 64, 128], each page's keys contiguous")
    if get_compute_capability(query.device)[0] != 10:
        raise ValueError("single_head_paged_mqa_logits requires SM10x")
    tensors = (query, context_lengths, requests, block_table, scores)
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("Queries, lengths, tables and scores must be contiguous")
    # CTAs per request group: two per SM in all, and enough that a CTA's run over
    # the whole capacity fits the TILES_PER_CTA tiles whose pages it stages (the
    # lengths stay on the device). CTAs past a group's length exit right away.
    key_tiles = (max_context + TILE - 1) // TILE
    runs = -(-key_tiles // TILES_PER_CTA)
    workers = max(CTAS_PER_SM * get_device_sm_count(query.device) // batch, runs, 1)
    args = (query, key_cache, *tensors[1:], max_context, workers)
    kinds = tuple(map(argument_kind, args))
    function = compiled(
        single_head_scorer.scorers(rows),
        "single_head_scorer",
        f"scorer_{rows}",
        (),
        query.device,
        kinds,
        (single_head_scorer.__file__,),
    )
    function(*args)
