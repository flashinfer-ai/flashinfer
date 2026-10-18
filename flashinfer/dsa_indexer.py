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

from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_experimental_api

# Thin experimental entry points; validation of the backend contract, the
# host dispatch and workspace policy, launch binding and JIT registration live
# in flashinfer.experimental.cake_dsa_indexer.

_FEATURE = "DSA indexer top-k selection (32 heads x 128, SM100/SM103/SM107)"


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    *,
    top_k: int = 2048,
    softmax_scale: Optional[float] = None,
    q_causal_offsets: Optional[torch.Tensor] = None,
    ratio: int = 1,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    workspace_buffer: Optional[torch.Tensor] = None,
    indices: Optional[torch.Tensor] = None,
    scores: Optional[torch.Tensor] = None,
    backend: str = "cake",
) -> Tuple[torch.Tensor, torch.Tensor]:
    r"""Fused DSA indexer scoring and deterministic exact top-k selection
    (DeepSeek Sparse Attention indexer, training), SM100 / SM103 / SM107 only.

    For query ``t`` and key ``j`` of the same packed segment

    ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])``

    with FP32 products, accumulation and score arithmetic (no BF16 intermediate
    rounding; the reduction scheme is fixed and documented in
    ``flashinfer/experimental/cake_dsa_indexer/README.md``).  Every row selects
    the exact global top ``min(top_k, visible)`` keys of its visible prefix,
    ranked by score descending then key id descending (``+0.0 == -0.0`` for
    ranking), and returns them in ascending id order with aligned scores.

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T, 32, 128]`` indexer queries after RoPE (contiguous).
    k : torch.Tensor
        BF16 ``[Tkv, 128]`` packed indexer keys after normalization and RoPE.  A
        row-strided view (for example ``packed[:, :128]`` of a ``[Tkv, 704]``
        tensor) is read in place; the row stride must be a multiple of 8
        elements.
    w : torch.Tensor
        FP32 ``[T, 32]`` signed head weights, already scaled by ``32 ** -0.5``.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[S + 1]`` independent query and key segment boundaries on the
        device (``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] == Tkv``).  They
        are not read on the host.
    top_k : int
        Output slots per query, ``1 <= top_k <= 4096`` (default 2048).
    softmax_scale : Optional[float]
        Positive scale applied to the per-head logits; defaults to ``128 ** -0.5``.
    q_causal_offsets : Optional[torch.Tensor]
        int64 ``[S]`` per-segment query offsets on the device.  Key ``j`` is
        visible to the query at segment-local position ``u`` iff ``j < Lk`` and
        ``j < floor((offset + u + 1) / ratio)``; without offsets ``offset = Lk -
        Lq`` for ``ratio == 1`` (queries are the tail of their key prefix) and
        ``0`` for compressed keys.  Negative offsets, empty segments and rows
        without visible keys are defined (the row is all padding).
    ratio : int
        Key compression ratio, ``>= 1`` (normally 1).
    max_seqlen_q : Optional[int]
        Host mirror of the longest query segment; accepted for signature parity
        and not read by any launch decision.
    max_seqlen_k : Optional[int]
        The caller's bound on every key segment (``max(cu_seqlens_k[1:] -
        cu_seqlens_k[:-1])``).  When given it sizes the rank finalize's bitmap
        pool and so selects the finalize program variant; the results are
        bitwise identical with and without it.  A value above ``Tkv`` or below
        ``ceil(Tkv / S)`` is rejected.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least :func:`dsa_indexer_topk_workspace_size` bytes for
        the call's geometry (uint8, 8-byte aligned); allocated through the
        caching allocator when omitted.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    backend : str
        Only ``"cake"``.

    Returns
    -------
    indices : torch.Tensor
        int32 ``[T, top_k]`` segment-local key ids, ascending within the row,
        unique; unused tail slots hold ``-1``.
    scores : torch.Tensor
        FP32 ``[T, top_k]`` scores aligned with ``indices`` (computed FP32
        bits; a selected ``-0.0`` is not normalized); tail slots hold ``-inf``.

    Identical inputs give identical ids and score bits; the result does not
    depend on the internal query / key partition or on the program the host
    dispatches for the geometry.  No ``[T, Tkv]`` matrix is materialized and
    the call performs no host synchronization, so it can be captured into a
    CUDA graph.  Finite inputs are the normal domain: with NaN / inf /
    overflowing inputs the operator returns with the output structure intact
    and the kernels never hang.
    """
    if backend == "cake":
        from .experimental.cake_dsa_indexer.cake_backend import (
            dsa_indexer_topk as cake_dsa_indexer_topk,
        )

        return cake_dsa_indexer_topk(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            top_k=top_k,
            softmax_scale=softmax_scale,
            q_causal_offsets=q_causal_offsets,
            ratio=ratio,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            workspace_buffer=workspace_buffer,
            indices=indices,
            scores=scores,
        )
    raise ValueError("DSA indexer top-k currently supports backend='cake'")


def dsa_indexer_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = 2048,
    ratio: int = 1,
    device: Optional[torch.device] = None,
) -> int:
    r"""Explicit workspace bound of :func:`dsa_indexer_topk` in bytes for one call geometry.

    The host dispatch picks the scan program (unit geometry, key-range split,
    CTA pair) from ``T = num_queries``, ``Tkv = num_keys``, ``S =
    num_segments``, ``ratio`` and ``top_k`` together with the SM count of
    ``device``, and the bound follows that choice: ``grid x queries_per_unit
    x candidate_capacity(top_k) x 8`` bytes of persistent candidate buffers
    (for example about 37 MiB at ``top_k = 2048`` with four queries per unit
    on a 148-SM device, 74 MiB with eight) plus ``T x n_split x top_k x 8``
    bytes of staging when the dispatch splits the key ranges.  Size a reused
    buffer for the largest bound over the geometries it serves.  Raises
    ``NotImplementedError`` while no generated program is registered for the
    device's architecture.
    """
    from .experimental.cake_dsa_indexer.cake_backend import (
        dsa_indexer_workspace_size as cake_dsa_indexer_workspace_size,
    )

    return cake_dsa_indexer_workspace_size(
        num_queries, num_keys, num_segments, top_k=top_k, ratio=ratio, device=device
    )
