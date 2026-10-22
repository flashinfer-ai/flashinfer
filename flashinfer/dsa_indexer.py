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
# in flashinfer.experimental.cake_dsa_indexer (backend="cake") and
# flashinfer.experimental.sm120_dsa_indexer (backend="sm120").

_FEATURE = (
    "DSA indexer top-k selection (cake: BF16 32x128 on SM100/SM103/SM107; "
    "sm120: FP8 / MXFP4 32/64x128 on SM120)"
)


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_indexer_topk(
    q: torch.Tensor,
    k: torch.Tensor,
    w: torch.Tensor,
    cu_seqlens_q: Optional[torch.Tensor],
    cu_seqlens_k: Optional[torch.Tensor],
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
    return_scores: bool = True,
    backend: str = "cake",
    k_scale: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fused DSA indexer scoring and deterministic exact top-k selection
    (DeepSeek Sparse Attention indexer, training), SM100 / SM103 / SM107 only;
    ``backend="sm120"`` is the FP8 / MXFP4 serving form on SM120 (below).

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
        cu_seqlens_k[:-1])``); the results are bitwise identical with and
        without it.  ``backend="cake"``: it sizes the rank finalize's bitmap
        pool and so selects the finalize program variant; a value above
        ``Tkv`` or below ``ceil(Tkv / S)`` is rejected.  ``backend="sm120"``:
        it must bound every segment's keys, and lets a call of short segments
        skip the long-row launches.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least :func:`dsa_indexer_topk_workspace_size` bytes for
        the call's geometry (uint8, 8-byte aligned); allocated through the
        caching allocator when omitted.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    return_scores : bool
        ``False``: indices only; ``scores`` must be omitted and ``None`` is
        returned in its place.  ``backend="sm120"`` then skips the score
        writes; ``backend="cake"`` still computes them internally.
    backend : str
        ``"cake"`` (BF16 training form above) or ``"sm120"``: the serving form
        on SM120, with FP8 (e4m3) ``q [T, H, 128]`` and ``k [Tkv, 128]``,
        ``H`` in {32, 64}, a per-key FP32 ``k_scale``, and ``w`` carrying the
        query scale and every softmax / head scale (``softmax_scale`` must be
        ``None``); ``s[t, j] = k_scale[j] * sum_h w[t, h] * relu(q[t, h] .
        k[j])``, bit-identical to DeepGEMM's ``fp8_mqa_logits`` on SM120.
        With ``q_scale``, MXFP4 instead: packed e2m1 ``q [T, H, 64]`` and
        ``k [Tkv, 64]`` (uint8 / int8, element ``2i`` in the low nibble) with
        int32 ``q_scale [T, H]`` and ``k_scale [Tkv]`` (four ue8m0 bytes, byte
        ``j`` scaling elements ``[32j, 32j + 32)``), bit-identical to
        DeepGEMM's FP4 ``fp8_fp4_mqa_logits``.
        ``top_k <= 4096``; ``max_seqlen_q`` and ``workspace_buffer`` are not
        used.  A selected ``-0.0`` score is returned as ``+0.0``.  Only
        SM12x GPUs are supported.
    k_scale : Optional[torch.Tensor]
        Per-key dequantization scale (``backend="sm120"`` only): FP32 ``[Tkv]``
        for FP8 keys, int32 ``[Tkv]`` block scales for MXFP4 keys.
    q_scale : Optional[torch.Tensor]
        int32 ``[T, H]`` MXFP4 block scales of ``q`` (``backend="sm120"``
        only); selects the MXFP4 form.
    block_table, context_lens : Optional[torch.Tensor]
        The paged (decode) form of ``backend="sm120"``: ``q`` is
        ``[B, next_n, H, D]`` (``q_scale [B, next_n, H]`` for MXFP4), ``k`` the
        paged indexer cache whose page ``p`` starts at byte
        ``p * k.stride(0) * k.element_size()`` and holds ``k.shape[1]`` key rows
        (128 e4m3 / 64 packed MXFP4 bytes each) followed by their scales (FP32 /
        int32 ue8m0 x4), ``w [B * next_n, H]``, int32 ``block_table
        [B, max_blocks]`` and int32 ``context_lens [B, next_n]`` (row ``j`` of
        request ``b`` sees keys ``[0, context_lens[b, j])``).
        ``cu_seqlens_q``, ``cu_seqlens_k`` and ``k_scale`` are ``None``;
        ``next_n`` (speculative-decoding verify: drafts + 1) is 1 to 16.  The
        outputs are ``[B * next_n, top_k]``
        request-local key positions: each row's exact top ``min(top_k, len)``
        set (the ragged form's ranking) in no particular order, then ``-1``;
        the scores are bit-identical to DeepGEMM's
        ``fp8_fp4_paged_mqa_logits``.  The launches depend only on the batch
        shape and ``block_table.shape[1] * k.shape[1]`` (the longest possible
        row): a CUDA graph captured once replays for any ``context_lens``.  The
        logits go through a per-device ``[B * next_n, that length]`` fp32
        buffer, so calls on concurrent streams of a device must not overlap;
        run one call outside capture before capturing a graph.
        ``q_causal_offsets``, ``ratio``, ``max_seqlen_k`` and
        ``workspace_buffer`` do not apply.
    enable_pdl : Optional[bool]
        Programmatic dependent launch (``backend="sm120"`` only): each kernel's
        launch and setup overlap the previous kernel's end; ``None``: on where
        the device supports it.

    Returns
    -------
    indices : torch.Tensor
        int32 ``[T, top_k]`` segment-local key ids, ascending within the row
        (the paged form: request-local, in no particular order), unique;
        unused tail slots hold ``-1``.
    scores : Optional[torch.Tensor]
        ``None`` with ``return_scores=False``; else FP32 ``[T, top_k]`` scores
        aligned with ``indices`` (``backend="cake"`` keeps a selected ``-0.0``,
        ``"sm120"`` returns ``+0.0``); tail slots hold ``-inf``.

    Identical inputs give identical ids and score bits; the result does not
    depend on the internal query / key partition or on the program the host
    dispatches for the geometry.  The ragged form never materializes a
    ``[T, Tkv]`` matrix, and no call synchronizes with the host, so calls can
    be captured into a CUDA graph.  Finite inputs are the normal domain: with NaN / inf /
    overflowing inputs the operator returns with the output structure intact
    and the kernels never hang.
    """
    paged = block_table is not None or context_lens is not None
    if paged and backend != "sm120":
        raise ValueError(
            "the paged form (block_table, context_lens) needs backend='sm120'"
        )
    if backend == "sm120" and softmax_scale is not None:
        raise ValueError(
            "backend='sm120' takes all scales folded into w; pass softmax_scale=None"
        )
    if backend == "sm120" and paged:
        if block_table is None or context_lens is None:
            raise ValueError("the paged form needs both block_table and context_lens")
        if cu_seqlens_q is not None or cu_seqlens_k is not None or k_scale is not None:
            raise ValueError(
                "the paged form takes cu_seqlens_q=None, cu_seqlens_k=None and k_scale=None"
                " (the scales live in the pages)"
            )
        if q_causal_offsets is not None or ratio != 1 or max_seqlen_k is not None:
            raise ValueError(
                "q_causal_offsets, ratio and max_seqlen_k apply to the ragged form only"
            )
        from .experimental.sm120_dsa_indexer import dsa_indexer_topk_paged

        return dsa_indexer_topk_paged(
            q,
            k,
            w,
            context_lens,
            block_table,
            top_k=top_k,
            max_context_len=block_table.shape[1] * k.shape[1],
            page_rows=k.shape[1],
            q_scale=q_scale,
            indices=indices,
            scores=scores,
            return_scores=return_scores,
            enable_pdl=enable_pdl,
        )
    if backend == "sm120":
        if k_scale is None:
            raise ValueError("backend='sm120' requires k_scale")
        if cu_seqlens_q is None or cu_seqlens_k is None:
            raise ValueError("the ragged form needs cu_seqlens_q and cu_seqlens_k")
        from .experimental.sm120_dsa_indexer import (
            dsa_indexer_topk as sm120_dsa_indexer_topk,
        )

        return sm120_dsa_indexer_topk(
            q,
            k,
            w,
            cu_seqlens_q,
            cu_seqlens_k,
            k_scale=k_scale,
            q_scale=q_scale,
            top_k=top_k,
            q_causal_offsets=q_causal_offsets,
            ratio=ratio,
            max_seqlen_k=max_seqlen_k,
            indices=indices,
            scores=scores,
            return_scores=return_scores,
            enable_pdl=enable_pdl,
        )
    if k_scale is not None or q_scale is not None:
        raise ValueError("k_scale and q_scale are only used by backend='sm120'")
    if backend == "cake":
        if not return_scores and scores is not None:
            raise ValueError("scores given with return_scores=False")
        from .experimental.cake_dsa_indexer.cake_backend import (
            dsa_indexer_topk as cake_dsa_indexer_topk,
        )

        indices, cake_scores = cake_dsa_indexer_topk(
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
        return indices, cake_scores if return_scores else None
    raise ValueError("DSA indexer top-k supports backend='cake' or 'sm120'")


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
