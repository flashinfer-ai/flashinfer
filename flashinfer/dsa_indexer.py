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

# Thin experimental entry point of the SM120 serving form; validation of the
# backend contract, launch binding and JIT registration live in
# flashinfer.experimental.sm120_dsa_indexer (backend="sm120").  The
# deterministic Cake training form (SM100 / SM103 / SM107) is
# flashinfer.cake_dsa_indexer_det_train (flashinfer-ai/flashinfer#8928).

_FEATURE = "DSA indexer top-k selection (sm120: FP8 / MXFP4 32/64x128 on SM120)"

_CAKE_MOVED = (
    "the deterministic Cake DSA indexer (BF16 / FP8 on SM100 / SM103 / SM107) is "
    "flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk; "
    "flashinfer.dsa_indexer.dsa_indexer_det_topk takes backend='sm120' only"
)


@flashinfer_experimental_api(feature=_FEATURE)
def dsa_indexer_det_topk(
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
    backend: str,
    k_scale: Optional[torch.Tensor] = None,
    q_scale: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    context_lens: Optional[torch.Tensor] = None,
    enable_pdl: Optional[bool] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fused DSA indexer scoring and exact top-k selection (DeepSeek Sparse
    Attention indexer), the experimental FP8 / MXFP4 serving form on SM120:
    ``backend="sm120"``, the only backend.

    The deterministic Cake training form (BF16 / FP8 on SM100 / SM103 / SM107)
    is :func:`flashinfer.cake_dsa_indexer_det_train.cake_dsa_indexer_det_train_topk`
    (flashinfer-ai/flashinfer#8928); ``backend="cake"`` raises a ``ValueError``
    that names it.

    Every row selects the exact global top ``min(top_k, visible)`` keys of its
    visible prefix, ranked by score descending then key id descending
    (``+0.0 == -0.0`` for ranking), and returns them in ascending id order with
    aligned scores (the paged form: below).

    Parameters
    ----------
    q : torch.Tensor
        FP8 (e4m3) ``[T, H, 128]`` indexer queries, ``H`` in {32, 64}; packed
        MXFP4 ``[T, H, 64]`` with ``q_scale`` (see ``backend``).
    k : torch.Tensor
        FP8 (e4m3) ``[Tkv, 128]`` packed indexer keys; packed MXFP4
        ``[Tkv, 64]`` with ``q_scale``; the paged form: the paged indexer cache
        (see ``block_table``).
    w : torch.Tensor
        FP32 ``[T, H]`` head weights carrying the query scale and every
        softmax / head scale.
    cu_seqlens_q, cu_seqlens_k : torch.Tensor
        int32 ``[S + 1]`` independent query and key segment boundaries on the
        device (``cu_seqlens_q[-1] == T``, ``cu_seqlens_k[-1] == Tkv``).  They
        are not read on the host.
    top_k : int
        Output slots per query, ``1 <= top_k <= 4096`` (default 2048).
    softmax_scale : Optional[float]
        Must be ``None``: every scale is folded into ``w``.
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
        Accepted for signature parity; not used.
    max_seqlen_k : Optional[int]
        The caller's bound on every key segment (``max(cu_seqlens_k[1:] -
        cu_seqlens_k[:-1])``); the results are bitwise identical with and
        without it.  It must bound every segment's keys, and lets a call of
        short segments skip the long-row launches.
    workspace_buffer : Optional[torch.Tensor]
        Accepted for signature parity; not used.
    indices, scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 / float32 outputs; allocated when
        omitted.
    return_scores : bool
        ``False``: indices only; ``scores`` must be omitted and ``None`` is
        returned in its place; the score writes are skipped.
    backend : str
        Required; ``"sm120"``: the serving form on SM120, with FP8 (e4m3) ``q [T, H, 128]`` and ``k [Tkv, 128]``,
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
        Per-key dequantization scale: FP32 ``[Tkv]`` for FP8 keys
        (contiguous), int32 ``[Tkv]`` block scales for MXFP4 keys.
    q_scale : Optional[torch.Tensor]
        int32 ``[T, H]`` MXFP4 block scales of ``q``; selects the MXFP4 form.
    block_table, context_lens : Optional[torch.Tensor]
        The paged (decode) form: ``q`` is
        ``[B, next_n, H, D]`` (``q_scale [B, next_n, H]`` for MXFP4), ``k`` the
        paged indexer cache whose page ``p`` starts at byte
        ``p * k.stride(0) * k.element_size()`` and holds ``k.shape[1]`` key rows
        (128 e4m3 / 64 packed MXFP4 bytes each) followed by their scales (FP32 /
        int32 ue8m0 x4), ``w [B * next_n, H]``, int32 ``block_table
        [B, max_blocks]`` and int32 ``context_lens [B, next_n]`` (row ``j`` of
        request ``b`` sees keys ``[0, context_lens[b, j])``; every length must lie in
        ``[0, block_table.shape[1] * k.shape[1]]``, which is not checked on the
        device).
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
        Programmatic dependent launch: each kernel's
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
        aligned with ``indices`` (a selected ``-0.0`` is returned as ``+0.0``);
        tail slots hold ``-inf``.

    Identical inputs give identical ids and score bits; the result does not
    depend on the internal query / key partition or on the program the host
    dispatches for the geometry.  The ragged form never materializes a
    ``[T, Tkv]`` matrix, and no call synchronizes with the host, so calls can
    be captured into a CUDA graph.  Finite inputs are the normal domain: with NaN / inf /
    overflowing inputs the operator returns with the output structure intact
    and the kernels never hang.
    """
    if backend == "cake":
        raise ValueError(_CAKE_MOVED)
    if backend != "sm120":
        raise ValueError(
            f"DSA indexer top-k supports backend='sm120' only, got {backend!r}"
        )
    paged = block_table is not None or context_lens is not None
    if softmax_scale is not None:
        raise ValueError(
            "backend='sm120' takes all scales folded into w; pass softmax_scale=None"
        )
    if paged:
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
        from .experimental.sm120_dsa_indexer import dsa_indexer_det_topk_paged

        return dsa_indexer_det_topk_paged(
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
    if k_scale is None:
        raise ValueError("backend='sm120' requires k_scale")
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError("the ragged form needs cu_seqlens_q and cu_seqlens_k")
    from .experimental.sm120_dsa_indexer import (
        dsa_indexer_det_topk as sm120_dsa_indexer_topk,
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


@flashinfer_experimental_api
def dsa_indexer_topk(
    q: torch.Tensor,
    kv: torch.Tensor,
    kv_scales: torch.Tensor,
    weights: torch.Tensor,
    cu_end: torch.Tensor,
    top_k: int = 2048,
    *,
    cu_start: Optional[torch.Tensor] = None,
    cand_cap: int = 12288,
    out: Optional[torch.Tensor] = None,
    workspace: Optional[Tuple[torch.Tensor, ...]] = None,
) -> torch.Tensor:
    r"""Fused FP8 indexer scoring and top-k with LiteTopK on SM100-family GPUs.

    Selects the largest scores
    ``sum_h weights[i, h] * relu(q[i, h] . kv[j]) * kv_scales[j]``
    over each row's visible range without materializing the full logits matrix.
    A sampled threshold filters candidates before exact selection. Rows whose
    candidate buffers overflow or contain too few candidates are rescored and
    selected on the GPU with bounded scratch, without host synchronization.

    Parameters
    ----------
    q : torch.Tensor
        ``[num_q, num_heads, 128]`` float8_e4m3fn queries; 32 or 64 heads.
    kv : torch.Tensor
        ``[seq_kv, 128]`` float8_e4m3fn keys; ``1 <= seq_kv <= 2**20``.
    kv_scales : torch.Tensor
        ``[seq_kv]`` float32 per-key dequantization scales.
    weights : torch.Tensor
        ``[num_q, num_heads]`` float32 head weights, including query scales.
    cu_end : torch.Tensor
        ``[num_q]`` int32 exclusive end of each row's visible key range.
    top_k : int
        Number of selected keys: 512, 1024 or 2048.
    cu_start : Optional[torch.Tensor]
        ``[num_q]`` int32 start of each visible range; defaults to zero. Both
        endpoints are clamped to ``[0, seq_kv]`` on the GPU. An end before its
        start gives an empty row.
    cand_cap : int
        Candidate capacity per row, a multiple of four in ``[4096, 12288]``.
        A smaller buffer saves memory but can trigger more recovery work.
    out : Optional[torch.Tensor]
        Preallocated ``[num_q, top_k]`` int32 output.
    workspace : Optional[Tuple[torch.Tensor, ...]]
        Scratch from :func:`dsa_indexer_topk_workspace`, optionally reused for
        smaller query/key counts. Its capacity must cover this call and its
        candidate width must equal ``cand_cap``. Do not share it between
        concurrent calls, alias it with inputs/outputs, or change its counters.

    Returns
    -------
    torch.Tensor
        ``[num_q, top_k]`` int32 indices relative to the clamped ``cu_start`` of
        each row, unordered, with ``-1`` padding when fewer keys are visible.
        Selection is exact for the kernel's FP32 scores, up to ties at the
        boundary. Its scoring reduction is not bitwise identical to DeepGEMM.

    Notes
    -----
    All tensors must be contiguous on the same CUDA device. Queries, keys,
    scales, weights and output require 16-byte aligned data pointers. Inputs
    and scores must be finite. Requires CUDA 12.9+ and the SM100 headers from
    an installed ``deep_gemm`` package.

    Scratch is bounded by processing at most 4096 query rows per window and a
    recovery score pool of at most 64 MiB. Extreme score distributions can
    trigger expensive recovery. Warm the JIT module before CUDA graph capture;
    keep the captured input, output and workspace storage alive through replay.

    This experimental revision removes ``prefix_logits`` and the ``status``
    return: sampling is internal and failed candidate rows recover on the GPU.
    """
    from .experimental.dsa_indexer import run

    return run(
        q, kv, kv_scales, weights, cu_end, top_k, cu_start, cand_cap, out, workspace
    )


@flashinfer_experimental_api
def dsa_indexer_topk_workspace(
    device: torch.device,
    num_q: int,
    num_heads: int,
    seq_kv: int,
    *,
    cand_cap: int = 12288,
) -> Tuple[torch.Tensor, ...]:
    """Allocate reusable scratch for :func:`dsa_indexer_topk`.

    ``num_q``, ``num_heads`` and ``seq_kv`` describe the maximum intended call;
    the same scratch can serve smaller calls if its capacities suffice.
    Allocation is limited to 4096 query rows and 64 MiB of recovery scores.
    ``num_heads`` must be 32 or 64, ``seq_kv`` in ``[1, 2**20]``, and
    ``cand_cap`` a multiple of four in ``[4096, 12288]``.

    Returns six tensors in launcher order: zeroed block counters, sample
    codes, candidate scores, candidate indices, row state, and recovery scores.
    A successful call leaves the counters zero; other scratch is overwritten.
    Reuse only in stream order, without concurrent calls or input/output
    aliasing. Allocate and warm the API before capturing CUDA graphs.
    """
    from .experimental.dsa_indexer import allocate_workspace

    return allocate_workspace(device, num_q, num_heads, seq_kv, cand_cap)
