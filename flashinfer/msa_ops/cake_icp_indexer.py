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

from typing import Optional

import torch

from ..api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the kernel contract, the host
# planning (split count, pool and merge geometry), the symmetric-memory
# workspace, launch binding and JIT registration live in
# flashinfer.experimental.cake_msa_icp_indexer.

_FEATURE = "MiniMax-M3 ICP decode indexer chain (fused FP8 scorer + symmetric-memory exchange merge, SM100/SM103)"


@flashinfer_experimental_api(feature=_FEATURE)
def prepare_msa_icp_indexer(
    index_q: torch.Tensor,
    index_k: torch.Tensor,
    block_table: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    positions: torch.Tensor,
    active_rows: torch.Tensor,
    forced_column: torch.Tensor,
    forced_block: torch.Tensor,
    n_ordinary: torch.Tensor,
    *,
    workspace,
    query_len: int,
    out: Optional[torch.Tensor] = None,
    candidates: Optional[torch.Tensor] = None,
    timeout_us: float = 50_000.0,
    backend: str = "cake",
):
    r"""Prepare one decode step of the MiniMax-M3 index-context-parallel (ICP)
    indexer chain on this rank, SM100 / SM103 only.

    The index keys of a sequence are sharded by rows across the ``world_size``
    ranks of a group: every 128- or 256-token page holds ``rows_per_rank``
    rows per rank.  For every decode token and index head the chain scores the
    token's FP8 query against this rank's rows of every visible page (FP8
    products, FP32 accumulation, ``-inf`` for rows the token may not see),
    keeps the exact top-16 pages under the canonical key (score descending,
    global block id ascending, ``+0.0 == -0.0``), exchanges the per-rank lists
    over NVLink-mapped symmetric memory and writes, on the rank that owns the
    head, the merged top-16 global block ids ascending with a ``-1`` tail,
    with the forced (newest) block injected once and ``n_ordinary`` ordinary
    slots.  Source and exported programs are bitwise identical; the merged
    output is bit-exact against the CPU oracle.

    Preparation validates and binds the tensors from their metadata (no
    device reads) and returns a runner; ``runner()`` launches the two
    generated programs with no allocation and no host synchronization and can
    be captured into a CUDA Graph.  One launch covers all ``total_tokens``
    tokens of ``index_q`` (``total_tokens <= tokens_cap`` of the workspace);
    every per-token tensor is indexed by that token.  Every rank of the workspace's group must
    launch the chain for the same step; after a step completes, read
    ``workspace.status_value()`` outside any capture -- a non-zero value is
    fatal for the step (see ``MSAIcpIndexerRunner``).

    Parameters
    ----------
    index_q : torch.Tensor
        float8_e4m3fn ``[total_tokens, 4, 128]`` index queries of this rank's
        decode tokens (contiguous).  ``total_tokens == request_count * query_len``.
    index_k : torch.Tensor
        float8_e4m3fn ``[num_pages, rows_per_rank, 128]`` view of this rank's
        row fragments of every physical page; the page pitch (``stride(0)``)
        may exceed the fragment (compound pages), the fragment itself is dense
        and 16-byte aligned.  ``rows_per_rank * world_size`` is the page size
        (128 for 2 or 4 ranks, 256 for 2 ranks).
    block_table : torch.Tensor
        int32 ``[request_count, columns]`` physical page ids per request,
        live prefix first; the column count bounds every request the buffers
        admit and sizes the candidate pool.
    query_start_loc : torch.Tensor
        int32 ``[request_count + 1]`` token offsets of the requests
        (``[0] == 0``, ``[-1] == total_tokens``, at most ``query_len`` tokens
        per request).  Read on device only.
    seq_lens : torch.Tensor
        int32 ``[request_count]`` visible history length of each request
        including its decode tokens.
    positions : torch.Tensor
        int64 ``[total_tokens]`` causal position of each decode token.
    active_rows : torch.Tensor
        uint8 ``[total_tokens]``; inactive tokens (``0``) score nothing and
        produce pad lists.
    forced_column : torch.Tensor
        int32 ``[total_tokens]`` column of the token's forced block inside
        this rank's readable window, ``-1`` when absent.
    forced_block : torch.Tensor
        int32 ``[total_tokens]`` global id of the token's forced (newest)
        block, ``-1`` for none; injected once by the merge.
    n_ordinary : torch.Tensor
        int32 ``[total_tokens]`` ordinary slots of the token (at most 15
        whenever a forced block exists).
    workspace : MSAIcpIndexerWorkspace
        This rank's exchange workspace (``create_msa_icp_indexer_workspace``)
        created collectively on the group; it fixes ``world_size`` and the
        token capacity.
    query_len : int
        Decode tokens per request, ``1 <= query_len <= 8``; selects the
        generated scorer program.
    out : Optional[torch.Tensor]
        int32 ``[total_tokens, 4 // world_size, 16]`` merged global block ids
        of the heads this rank owns, ascending, ``-1`` tail.  Allocated when
        omitted.
    candidates : Optional[torch.Tensor]
        float32 ``[total_tokens, 4, 16, 2]`` rank-local merged lists
        (score, int32 id bits), best first, ``(-inf, -1)`` padded.  Allocated
        when omitted.
    timeout_us : float
        Poll deadline of the exchange from CTA start; past it the status word
        receives 8 and the missing peer words merge as absent.
    backend : str
        Only ``"cake"``.

    Returns
    -------
    MSAIcpIndexerRunner
        Callable runner; ``runner()`` launches the chain and returns ``out``.
    """
    if backend != "cake":
        raise ValueError("MSA ICP indexer currently supports backend='cake'")
    from ..experimental.cake_msa_icp_indexer import cake_backend

    return cake_backend.prepare_msa_icp_indexer(
        index_q,
        index_k,
        block_table,
        query_start_loc,
        seq_lens,
        positions,
        active_rows,
        forced_column,
        forced_block,
        n_ordinary,
        workspace=workspace,
        query_len=query_len,
        out=out,
        candidates=candidates,
        timeout_us=timeout_us,
        backend="cake",
    )


@flashinfer_experimental_api(feature=_FEATURE)
def create_msa_icp_indexer_workspace(
    *,
    world_size: int,
    rank: int,
    tokens_cap: int,
    device: torch.device,
    group=None,
):
    r"""Create this rank's exchange workspace of the ICP indexer chain.

    Collective: every rank of ``group`` (default: the world group, which must
    have exactly ``world_size`` ranks) calls it in the same order.  The
    symmetric pool holds ``2 * tokens_cap * (4 // world_size) * world_size * 16``
    64-bit words (``msa_icp_indexer_workspace_bytes``); ``tokens_cap`` bounds
    ``total_tokens`` of every step prepared against it.  Not allowed inside
    CUDA Graph capture.
    """
    from ..experimental.cake_msa_icp_indexer import cake_workspace

    return cake_workspace.create_workspace(
        world_size=world_size,
        rank=rank,
        tokens_cap=tokens_cap,
        device=device,
        group=group,
    )


@flashinfer_experimental_api(feature=_FEATURE)
def destroy_msa_icp_indexer_workspace(workspace) -> None:
    r"""Release a workspace created by ``create_msa_icp_indexer_workspace``.

    Collective: every rank calls it after its last launch against the
    workspace has completed.  Runners prepared against it must not be
    launched afterwards.
    """
    from ..experimental.cake_msa_icp_indexer import cake_workspace

    cake_workspace.destroy_workspace(workspace)


@flashinfer_experimental_api(feature=_FEATURE)
def msa_icp_indexer_workspace_bytes(world_size: int, tokens_cap: int) -> dict:
    r"""Byte budget of one rank's workspace for ``world_size`` ranks and ``tokens_cap`` tokens."""
    from ..experimental.cake_msa_icp_indexer import cake_workspace

    return cake_workspace.workspace_bytes(world_size, tokens_cap)
