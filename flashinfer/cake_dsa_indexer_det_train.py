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

Deterministic Cake DSA indexer top-k for training (SM100 / SM103 / SM107).

Every public name carries the mode word ``det``; no entry point takes a
``deterministic=`` keyword (flashinfer-ai/flashinfer#8928).  The
non-deterministic counterpart is ``flashinfer.cake_dsa_indexer_nondet_train``,
which leaves the slot order of each row unspecified.

Thin entry points: validation of the backend contract, the host dispatch and
workspace policy, the launch binding, JIT registration and the generated
kernel sources live in the backend package
``flashinfer.experimental.cake.cake_dsa_indexer_det_train``, which this module
imports on the first call (never at import time).
"""

import functools
from typing import Optional, Tuple

import torch

from .api_logging import flashinfer_api


@functools.cache
def _backend():
    from .experimental.cake.cake_dsa_indexer_det_train import (  # noqa: PLC0415
        cake_backend,
    )

    return cake_backend


@flashinfer_api
def cake_dsa_indexer_det_train_topk(
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
    return_scores: bool = True,
    k_scale: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    r"""Fused DSA indexer scoring and deterministic exact top-k selection
    (DeepSeek Sparse Attention indexer, training), SM100 / SM103 / SM107 only.

    For query ``t`` and key ``j`` of the same packed segment

    ``s[t, j] = sum_h w[t, h] * relu(softmax_scale * sum_d q[t, h, d] * k[j, d])``

    with FP32 products, accumulation and score arithmetic (no BF16 intermediate
    rounding; the reduction scheme is fixed and documented in
    ``flashinfer/experimental/cake/cake_dsa_indexer_det_train/README.md``).  Every
    row selects the exact global top ``min(top_k, visible)`` keys of its visible
    prefix, ranked by score descending then key id descending (``+0.0 == -0.0``
    for ranking), and returns them in ascending id order with aligned scores.

    The FP8 form is selected by float8_e4m3fn ``q`` and ``k`` with a per-key
    FP32 ``k_scale``:

    ``s[t, j] = k_scale[j] * sum_h w[t, h] * relu(sum_d q[t, h, d] * k[j, d])``

    where ``w`` carries every other scale (``softmax_scale`` must be ``None``).
    The e4m3 products are exact and accumulate in FP32 on the tensor cores:
    four K = 32 steps on SM100 / SM103, in the order of DeepGEMM's
    ``fp8_mqa_logits`` (bit-identical logits), and two K = 64 steps on SM107,
    whose logits can differ from DeepGEMM's in the last bits.  Every FP8
    program of one architecture gives the same ids and score bits; selection,
    visibility and outputs follow the BF16 form.

    Parameters
    ----------
    q : torch.Tensor
        BF16 ``[T, 32, 128]`` indexer queries after RoPE (contiguous);
        float8_e4m3fn for the FP8 form.
    k : torch.Tensor
        BF16 ``[Tkv, 128]`` packed indexer keys after normalization and RoPE.  A
        row-strided view (for example ``packed[:, :128]`` of a ``[Tkv, 704]``
        tensor) is read in place; the row stride must be a multiple of 8
        elements.  The FP8 form: float8_e4m3fn (the same dtype as ``q``), a
        row stride that is a multiple of 16 elements.
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
        The FP8 form takes ``None`` (the scale is folded into ``w``).
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
        without it.  It sizes the rank finalize's bitmap pool and so selects
        the finalize program variant.  On ``sm_100a`` and ``sm_103a`` it also
        shows the host whether every key segment of a packed call has the same
        length (``Tkv == S * max_seqlen_k``): such a call runs its scan program
        without the longest-segment-first unit order (the ``:lsf`` copy).  A
        value above ``Tkv`` or below ``ceil(Tkv / S)`` is rejected.
    workspace_buffer : Optional[torch.Tensor]
        Scratch of at least :func:`cake_dsa_indexer_det_train_topk_workspace_size`
        bytes for the call's geometry (uint8, 8-byte aligned); allocated through
        the caching allocator when omitted.  A caller-owned buffer may be shared
        with other calls and kernels between runs: on the ticket architectures
        the backend then zeroes the buffer's ticket word before every run,
        while its own workspace is zeroed once at prepare.
    indices : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` int32 output; allocated when omitted.
    scores : Optional[torch.Tensor]
        Caller-owned ``[T, top_k]`` float32 output; allocated when omitted.
    return_scores : bool
        ``False``: indices only; ``scores`` must be omitted and ``None`` is
        returned in its place (the scores are still computed internally).
    k_scale : Optional[torch.Tensor]
        FP32 ``[Tkv]`` per-key dequantization scale of the FP8 form
        (contiguous).  BF16 keys take none.

    Returns
    -------
    indices : torch.Tensor
        int32 ``[T, top_k]`` segment-local key ids, ascending within the row,
        unique; unused tail slots hold ``-1``.
    scores : Optional[torch.Tensor]
        ``None`` with ``return_scores=False``; else FP32 ``[T, top_k]`` scores
        aligned with ``indices`` (a selected ``-0.0`` stays ``-0.0``); tail
        slots hold ``-inf``.

    Identical inputs give identical ids and score bits; the result does not
    depend on the internal query / key partition or on the program the host
    dispatches for the geometry.  The call never materializes a ``[T, Tkv]``
    matrix, and no call synchronizes with the host, so calls can be captured
    into a CUDA graph.  Finite inputs are the normal domain: with NaN / inf /
    overflowing inputs the operator returns with the output structure intact
    and the kernels never hang.
    """
    if cu_seqlens_q is None or cu_seqlens_k is None:
        raise ValueError(
            "cake_dsa_indexer_det_train_topk needs cu_seqlens_q and cu_seqlens_k"
        )
    if not return_scores and scores is not None:
        raise ValueError("scores given with return_scores=False")
    indices, cake_scores = _backend().cake_dsa_indexer_det_train_topk(
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
        k_scale=k_scale,
    )
    return indices, cake_scores if return_scores else None


@flashinfer_api
def cake_dsa_indexer_det_train_topk_workspace_size(
    num_queries: int,
    num_keys: int,
    num_segments: int,
    *,
    top_k: int = 2048,
    ratio: int = 1,
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.bfloat16,
) -> int:
    r"""Explicit workspace bound of :func:`cake_dsa_indexer_det_train_topk` in bytes for one call geometry.

    The host dispatch picks the scan program (unit geometry, key-range split,
    CTA pair) from ``T = num_queries``, ``Tkv = num_keys``, ``S =
    num_segments``, ``ratio`` and ``top_k`` together with the SM count of
    ``device``, and the bound follows that choice: ``grid x queries_per_unit
    x candidate_capacity(top_k) x 8`` bytes of persistent candidate buffers
    (for example 74 MiB at ``top_k = 2048`` with four queries per unit on a
    148-SM ``sm_100a`` device, whose capacity floor is 16384 slots at that
    ``top_k``, and 148 MiB with eight) plus ``T x n_split x top_k x 8``
    bytes of staging when the dispatch splits the key ranges.  Size a reused
    buffer for the largest bound over the geometries it serves.  ``dtype`` is
    the q / k dtype: ``torch.float8_e4m3fn`` bounds the FP8 form, whose
    programs differ.  Raises ``NotImplementedError`` while no generated program
    of the form is registered for the device's architecture.

    Parameters
    ----------
    num_queries : int
        ``T``, the number of packed queries of the call.
    num_keys : int
        ``Tkv``, the number of packed keys of the call.
    num_segments : int
        ``S``, the number of segments (``len(cu_seqlens_q) - 1``).
    top_k : int
        Output slots per query, as passed to :func:`cake_dsa_indexer_det_train_topk`
        (default 2048).
    ratio : int
        Key compression ratio, as passed to :func:`cake_dsa_indexer_det_train_topk`
        (default 1).
    device : Optional[torch.device]
        The device whose SM count and architecture select the scan program;
        the current CUDA device when omitted.
    dtype : torch.dtype
        The q / k dtype: ``torch.bfloat16`` (default) or ``torch.float8_e4m3fn``.

    Returns
    -------
    int
        The workspace bound in bytes for this geometry.
    """
    return _backend().cake_dsa_indexer_det_train_topk_workspace_size(
        num_queries,
        num_keys,
        num_segments,
        top_k=top_k,
        ratio=ratio,
        device=device,
        dtype=dtype,
    )


__all__ = [
    "cake_dsa_indexer_det_train_topk",
    "cake_dsa_indexer_det_train_topk_workspace_size",
]
