# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block selection of a block-sparse attention indexer: score every complete 4-token block, keep the top-k.

The sparse-attention indexer of the Qwen3.8 attention family scores a query token against the compressed
keys of the complete blocks it can see and keeps the ``top_k`` best blocks::

    score[t, b] = scale * sum_h relu(qi[t, h] . kbar[b])        for b < floor((pos_t + 1) / block_size)
    block_ids[t] = top_k(score[t, :])                            (local block ids of the token's sequence)

``qi`` holds a few indexer heads per token (4 for Qwen3.8-Flash-Next) and ``kbar`` one compressed key per
block (the block's mean-pooled, normalized, position-encoded raw keys -- produced upstream). The scale is the
constant ``1 / sqrt(head_dim)``: every head carries the same weight.

This is exactly the ratio-causal, ReLU-summed, head-reduced score the DSA indexer forward computes with its
compact-logits + fused top-k path (``cudnn.DSA.indexer_forward_top_k_wrapper``): the compression ratio is
the block size, the per-head weights are the constant scale, and the dense ``[T, n_blocks]`` score tensor is
never materialized -- the kernel writes only the ratio-causal candidates compactly and a radix pass selects
the top-k per token. :func:`qsa_select_forward` is the thin host wrapper that spells the QSA geometry in that
vocabulary.

Contract (prefill, THD-packed queries):

* ``qi``: ``[T, n_heads, head_dim]`` bf16, the indexer queries of all sequences packed along ``T``; the
  token ``t`` of sequence ``s`` (``cu_seqlens_q[s] <= t < cu_seqlens_q[s + 1]``) sits at position
  ``pos_t = q_pos0[s] + (t - cu_seqlens_q[s])`` (``q_pos0 = None`` = every sequence starts at 0).
* ``kbar``: ``[B, n_blocks_max, head_dim]`` bf16, the compressed keys of every sequence, one row per
  complete block, padded to the longest sequence. A token reads rows ``[0, floor((pos_t + 1) / block_size))``
  of its sequence; the caller guarantees those rows are complete blocks (``n_blocks_s >=
  floor((q_pos0[s] + S_q[s]) / block_size)``) -- rows past them are never read.
* ``block_ids``: ``[T, top_k]`` int32, LOCAL block ids of the token's sequence, ``-1`` padded where a token
  sees fewer than ``top_k`` blocks (the first ``block_size * top_k + block_size - 1`` tokens of a sequence
  select every visible block -- the identity region), unsorted. ``scores`` are the selected logits
  (``-inf`` on the padding). With ``deterministic=True`` (the default) a tie at the k-th boundary keeps the
  smaller block id, so the selected SET is reproducible across runs; the slot order within a row is not.
* **The fp8 indexer cache**: ``qi`` and ``kbar`` may both be ``float8_e4m3fn`` -- plain e4m3 of the RMSNormed values,
  no scales (both are O(1) after their norms; the choice of the reference serving stacks). The scorer then runs
  the DSA MXFP8 arm with all-ones E8M0 scale blobs, which is exactly ``sum_h relu(e4m3(q_h) . e4m3(kbar_b))`` in
  fp32 -- the bf16 kernel on the upcast values gives the same scores -- so the selection differs from the bf16
  one only where the e4m3 rounding of the inputs moves a score across the k-th boundary. That is a NUMERICS
  choice the caller makes by the dtype, not a knob: measured on the oracle's indexer activations (random weights,
  one dense sequence), the e4m3 selection agrees with the bf16 one on 99.0 % of the selected blocks at
  S = 4096, 97.4 % at 32768 and 96.5 % at 131072 (sparse rows; every flipped block within the row's own e4m3 score
  noise of the k-th score), against 99.8 % for bf16 vs fp32. A mixed pair (one bf16, one e4m3) is refused.

**Streams (Rule 5).** Every torch operation of a call runs on the LAUNCH stream (``stream``, else torch's current stream), resolved
first: the cached constants' creation (the constant head weights, the packed-K row offsets, the e4m3 arm's unit scale blobs), the
e4m3 arm's per-call padded Q prefix, and the scorer wrapper's own work.  A cached constant filled on one stream and read by a later
call on another is ordered behind its fill through the event recorded when it was created (``wait_event`` on the reading stream).
Inside a stream capture neither the record nor the wait is issued (an event recorded outside a capture cannot be waited on inside
one), so a captured call's constants are warmed up on the capture stream first -- one call outside the capture at the same
``(B, n_blocks_max)``, head count and cache dtype -- the rule every capture of the gated attention block already follows.

The DECODE form is :func:`qsa_select_decode_forward` (one to a few query rows per sequence, the verify rows sharing the
step-0 list): the dense DSA scorer at the smallest MMA tile plus the radix top-k kernel with one length per row.
"""

from __future__ import annotations

import math
from threading import Lock
from typing import Optional, Sequence, Union

import torch

from flashinfer.experimental.qwen_3_5_block._cudnn_frost._torch_stream import _raw_current_stream, stream_context
from flashinfer.experimental.qwen_3_5_block._cudnn_frost._tupledict import TupleDict  # the DSL-free home of the result type (Rule 7: this module's import must not execute the DSL)
from flashinfer.experimental.qwen_3_5_block._cudnn_frost.frost.buffers import cutedsl_arch_requirement_error, cutedsl_requirement_error

__all__ = [
    "qsa_select_forward",
    "QSA_BLOCK_SIZE",
    "QSA_TOP_K",
    "qsa_select_decode_forward",
    "decode_row_positions_and_counts",
    "TOP_K_KERNEL_MAX",
    "INDEXER_SCORER_HEAD_DIM",
    "indexer_scorer_head_groups",
]

QSA_BLOCK_SIZE = 4  # tokens per compressed block (the compression ratio of the indexer)
INDEXER_SCORER_HEAD_DIM = 128  # the head dim the indexer scorer packs (qi and kbar rows); the block's in-block indexer declines any other
QSA_TOP_K = 512  # blocks kept per token (2048 selected tokens / 4)
TOP_K_KERNEL_MAX = 2048  # the radix top-k kernel's bound on top_k (0 < top_k <= 2048); the block's own cap (512) is QsaSpec's
_SCORE_ROW_ALIGN = 8  # fp32 score rows are read by the top-k kernel in 256-bit vectors: n_blocks_max must be a multiple of 8
_E4M3 = torch.float8_e4m3fn  # the fp8 indexer cache's element type (plain e4m3, no scales)
_E8M0 = torch.float8_e8m0fnu  # the MXFP8 scorer's scale type; 1.0 (= 2^0, byte 127) everywhere for the plain-e4m3 scorer
_SF_VEC_SIZE = 32  # elements per E8M0 scale of the MXFP8 scorer (head_dim 128 -> 4 scale groups per row)
_SF_ATOM_ROWS = 128  # packed (token, head) rows per scale atom: every scale span is a multiple of it
_MXFP8_TILE_ROWS = 128  # the MXFP8 scorer's packed-row tile (fixed; the bf16 scorer's m_block_size knob does not apply)

_ones_cache: dict = {}  # (device index, capacity) -> (the bf16 ones buffer, the handle of the stream that filled it, its fill event or None)
_ones_cache_lock = Lock()


def indexer_scorer_head_groups() -> tuple:
    """The indexer head counts per token the bf16 scorer packs on its MMA tile -- the DSA kernel's own contract, read, not
    transcribed.  The module that spells it carries no kernel import of its own, but importing it runs the DSA package's
    ``__init__`` (its kernel modules): a declaration-time caller gates the CuTe DSL floor FIRST
    (``api._check_qsa_indexer_geometry``), so a too-old DSL reads as a version decline, never as an error from inside the DSL."""
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward._support import SUPPORTED_QHEAD_PER_KV_HEAD_BF16

    return tuple(SUPPORTED_QHEAD_PER_KV_HEAD_BF16)


_cu_k_cache: dict = {}  # (device index, B, n_blocks_max) -> (the int32 row offsets, the filling stream's handle, its fill event or None)
_cu_k_cache_lock = Lock()
_unit_scale_cache: dict = {}  # (device index, capacity) -> (the flat E8M0 ones blob of that capacity, the filling stream's handle, its fill event or None)
_unit_scale_cache_lock = Lock()


def _device_index(device: torch.device) -> int:
    return torch.cuda.current_device() if device.index is None else device.index


def _current_stream_handle(index: int) -> int:
    """torch's current stream on device ``index`` as a raw handle -- the identity the constant caches compare (torch's raw getter, ~0.07 us;
    the Stream object only where that getter is absent)."""
    raw = _raw_current_stream(torch, index)
    return int(raw) if raw is not None else torch.cuda.current_stream(index).cuda_stream


def _cached_constant(cache: dict, lock: Lock, key: tuple, index: int, make):
    """A process-lifetime device constant of the selector (never evicted: a CUDA graph may retain the address captured at warm-up), created
    by ``make()`` on the stream torch is CURRENTLY on -- the caller has entered the launch stream's context, so the fill is ordered ahead of
    the scorer that reads it (Rule 5) -- with an event recorded behind the fill.  A later caller whose launch stream is not the one that filled
    the constant first waits on that event (``wait_event`` on its own stream: the read is ordered behind the fill whatever the filling stream
    is doing -- the previous form handed a scorer on a side stream a buffer whose fill sat on torch's parked default stream).  Inside a stream
    capture neither the record nor the wait is issued (an event recorded outside a capture cannot be waited on inside one): the constants of
    a captured call are warmed up on the capture stream beforehand (the module docstring)."""
    with lock:
        entry = cache.get(key)
        if entry is None:
            value = make()
            event = None
            if not torch.cuda.is_current_stream_capturing():
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(index))
            cache[key] = (value, _current_stream_handle(index), event)
            return value
    value, producer, event = entry
    if event is not None and _current_stream_handle(index) != producer and not torch.cuda.is_current_stream_capturing():
        torch.cuda.current_stream(index).wait_event(event)
    return value


def _constant_weights(numel: int, device: torch.device) -> torch.Tensor:
    """A cached bf16 ones buffer (power-of-two capacity buckets, never evicted) -- the constant head weight.

    Mirrors the DSA denominator placeholder: a CUDA graph may retain the address captured at warm-up, so a
    bucket is never replaced; one allocation per (device, capacity) for the life of the process.  Created on the
    launch stream of the call that first needs the bucket and ordered for every other stream (:func:`_cached_constant`).
    """
    index = _device_index(device)
    capacity = 1 << (max(1, numel) - 1).bit_length()
    buf = _cached_constant(
        _ones_cache, _ones_cache_lock, (index, capacity), index, lambda: torch.ones((capacity,), dtype=torch.bfloat16, device=torch.device("cuda", index))
    )
    return buf[:numel]


def _unit_scale_capacity(numel: int) -> int:
    """The power-of-two capacity bucket of a ``numel``-element unit scale blob (the constant weights' rule, :func:`_constant_weights`)."""
    return 1 << (max(1, int(numel)) - 1).bit_length()


def _unit_scale_blob(shape: tuple, device: torch.device) -> torch.Tensor:
    """An all-ones E8M0 scale tensor of ``shape`` (``(L, rows, sf_groups)``): the plain-e4m3 scorer's scale factors.

    Every byte is 127 (= 2^0), so the MXFP8 scorer multiplies by 1.0 and computes the plain e4m3 x e4m3 dot.  The storage is ONE
    flat blob per ``(device, capacity)`` with the capacity rounded up to a power of two (the constant weights' rule), so a process
    that serves many geometries holds O(log n) blobs, never one per distinct shape; a blob is never evicted (a CUDA graph may retain
    the address captured at warm-up), is created on the launch stream of the first call that needs its bucket and is ordered for
    every other stream (:func:`_cached_constant`); the blob is 1/32 of the data.  The result is the bucket's leading ``numel``
    elements viewed in ``shape``: a contiguous E8M0 view at the bucket's base address (the allocator's 16-byte-aligned block start)
    -- the exact packed shape the scorer checks, the dtype, the contiguity and the alignment its CuTe tensor conversion assumes.
    """
    index = _device_index(device)
    dims = tuple(int(x) for x in shape)
    numel = math.prod(dims)
    capacity = _unit_scale_capacity(numel)
    blob = _cached_constant(
        _unit_scale_cache,
        _unit_scale_cache_lock,
        (index, capacity),
        index,
        lambda: torch.ones((capacity,), dtype=torch.float32, device=torch.device("cuda", index)).to(_E8M0),
    )
    return blob[:numel].view(dims)


def _round_up(x: int, m: int) -> int:
    return -(-int(x) // int(m)) * int(m)


def _padded_prefix(cu_seqlens: torch.Tensor, align: int) -> torch.Tensor:
    """``cu_seqlens`` with every sequence length rounded up to ``align`` (device ops, no sync): the MXFP8 scale prefix."""
    lengths = cu_seqlens[1:] - cu_seqlens[:-1]
    out = torch.zeros_like(cu_seqlens)
    torch.cumsum(torch.div(lengths + (align - 1), align, rounding_mode="floor") * align, dim=0, out=out[1:])
    return out


def _fp8_q_scale_prefix(cu_seqlens_q: torch.Tensor, n_heads: int) -> torch.Tensor:
    """The e4m3 arm's ``cu_seqlens_q_scale_padded`` for ``cu_seqlens_q`` (int32 ``[B + 1]``, any device -- a host tensor gives a host
    result): every sequence's length rounded up to the scale span of ``n_heads`` heads, ``128 // gcd(128, n_heads)`` tokens (32 at 4
    heads).  A caller whose ``cu_seqlens_q`` is a plan constant builds it once (the block's indexer stage, at ``compile``) and hands it
    to :func:`qsa_select_forward` instead of paying the five device ops of :func:`_padded_prefix` per call (Rule 1)."""
    return _padded_prefix(cu_seqlens_q, _SF_ATOM_ROWS // math.gcd(_SF_ATOM_ROWS, int(n_heads)))


def _padded_cu_seqlens_k(batch: int, n_blocks_max: int, device: torch.device) -> torch.Tensor:
    """``[0, n, 2n, ..., B n]`` int32: the row offsets of a ``[B, n_blocks_max, D]`` key buffer seen as packed -- built ON DEVICE (two
    small launches on the launch stream, no host upload: the pageable upload of the previous form blocked the host on the stream it
    was issued on) the first time a ``(device, B, n_blocks_max)`` is seen, cached and ordered for every stream (:func:`_cached_constant`)."""
    index = _device_index(device)
    batch, n_blocks_max = int(batch), int(n_blocks_max)
    if batch * n_blocks_max >= 2**31:
        raise ValueError(f"kbar holds {batch} x {n_blocks_max} rows; the packed row offsets must fit int32")
    return _cached_constant(
        _cu_k_cache,
        _cu_k_cache_lock,
        (index, batch, n_blocks_max),
        index,
        lambda: torch.arange(batch + 1, dtype=torch.int32, device=torch.device("cuda", index)) * n_blocks_max,
    )


def _as_host_ints(values, name: str) -> list:
    if isinstance(values, torch.Tensor):
        if values.ndim != 1:
            raise ValueError(f"{name} must be 1-D, got shape {tuple(values.shape)}")
        return [int(v) for v in values.tolist()]
    return [int(v) for v in values]


def qsa_select_forward(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    top_k: int = QSA_TOP_K,
    block_size: int = QSA_BLOCK_SIZE,
    q_pos0: Optional[torch.Tensor] = None,
    n_blocks_per_seq: Optional[Union[Sequence[int], torch.Tensor]] = None,
    max_seqlen_q: Optional[int] = None,
    scale: Optional[float] = None,
    w: Optional[torch.Tensor] = None,
    deterministic: bool = True,
    block_ids_out: Optional[torch.Tensor] = None,
    scores_out: Optional[torch.Tensor] = None,
    cand_buffer: Optional[torch.Tensor] = None,
    cand_batch_offsets: Optional[torch.Tensor] = None,
    stream=None,
    cu_seqlens_q_scale_padded: Optional[torch.Tensor] = None,
) -> TupleDict:
    """Select the ``top_k`` compressed blocks per query token (see the module docstring for the contract).

    Args:
        qi: ``[T, n_heads, head_dim]`` bf16 indexer queries, THD-packed; ``n_heads`` in (4, 8, 16, 32, 64)
            (the groups the DSA unified scorer packs), ``head_dim == 128``. Any strides with a unit last
            stride are read in place (a column slice of a wider slab needs no copy). ``float8_e4m3fn``
            together with an e4m3 ``kbar`` selects the fp8 indexer cache arm (the module docstring).
        kbar: ``[B, n_blocks_max, head_dim]`` bf16 (or e4m3 with an e4m3 ``qi``) compressed keys, contiguous.
        cu_seqlens_q: ``[B + 1]`` int32 CUDA prefix of the query lengths (``T == cu_seqlens_q[-1]``).
        top_k: blocks kept per token (1..2048).
        block_size: tokens per compressed block (the compression ratio), 4 for QSA.
        q_pos0: ``[B]`` int32 CUDA, the position of each sequence's first query token (chunked prefill:
            the tokens already in the cache); ``None`` = 0 for every sequence.
        n_blocks_per_seq: optional HOST-visible block counts per sequence; when given, the geometry
            ``floor((q_pos0 + S_q) / block_size) <= n_blocks_per_seq <= n_blocks_max`` is checked on the
            host (a CUDA tensor here, or a CUDA ``q_pos0`` with it, is read back -- one sync; pass lists
            for a sync-free call). Device values are otherwise consumed as provided.
        max_seqlen_q: the longest query length; required under CUDA-graph capture (deriving it syncs).
        scale: the constant head weight; ``None`` = ``1 / sqrt(head_dim)``.
        w: optional ``[T, n_heads]`` bf16 or fp32 per-head weights replacing the constant (every head then
            weighs ``w[t, h] * scale``); bf16 only with e4m3 inputs.
        deterministic: ties at the k-th boundary keep the smaller block id (reproducible sets).
        block_ids_out / scores_out: optional preallocated ``[T, top_k]`` int32 / fp32 outputs.
        cand_buffer / cand_batch_offsets: the compact-logits scratch from
            ``cudnn.DSA.compress_topk_cand_buffer_size_thd(cu_seqlens_q, cu_seqlens_k, block_size, q_pos0)``
            with ``cu_seqlens_k = arange(B + 1) * n_blocks_max``; both are required under graph capture.
        stream: optional ``cuda.CUstream``; ``None`` = torch's current stream.
        cu_seqlens_q_scale_padded: (appended) the e4m3 arm's padded Q scale prefix -- ``[B + 1]`` int32 CUDA, every
            sequence's length rounded up to the scale span of its heads (``_fp8_q_scale_prefix``) -- precomputed by a
            caller whose ``cu_seqlens_q`` is a plan constant (the block's indexer stage: nothing built per execute,
            Rule 1); ``None`` = derived per call on device (five small ops); refused on the bf16 arm.

    Returns:
        ``TupleDict(block_ids=[T, top_k] int32, scores=[T, top_k] fp32)``.

    Raises:
        NotImplementedError: a head count outside the packed groups, a device below the SM100 family, or a
            CuTe DSL below the floor / without the target (the message names the version or the set).
        ValueError: a malformed input (shape, dtype, device, prefix, geometry).
    """
    if not isinstance(qi, torch.Tensor) or qi.ndim != 3:
        raise ValueError(f"qi must be a [T, n_heads, head_dim] tensor, got {None if not isinstance(qi, torch.Tensor) else tuple(qi.shape)}")
    if not isinstance(kbar, torch.Tensor) or kbar.ndim != 3:
        raise ValueError(f"kbar must be a [B, n_blocks_max, head_dim] tensor, got {None if not isinstance(kbar, torch.Tensor) else tuple(kbar.shape)}")
    total_q, n_heads, head_dim = qi.shape
    batch, n_blocks_max, head_dim_k = kbar.shape
    if head_dim != 128 or head_dim_k != 128:
        raise ValueError(f"qsa_select_forward scores head_dim 128 indexer heads, got qi head_dim {head_dim} and kbar head_dim {head_dim_k}")
    fp8 = _fp8_cache_arm(qi, kbar)
    if not qi.is_cuda or kbar.device != qi.device:
        raise ValueError("qi and kbar must be CUDA tensors on one device")
    if qi.stride(-1) != 1:
        raise ValueError(f"qi needs a unit last stride, got strides {tuple(qi.stride())}")
    if not kbar.is_contiguous():
        raise ValueError(f"kbar must be contiguous, got strides {tuple(kbar.stride())}")
    if not isinstance(cu_seqlens_q, torch.Tensor) or cu_seqlens_q.ndim != 1 or cu_seqlens_q.dtype != torch.int32 or not cu_seqlens_q.is_cuda:
        raise ValueError("cu_seqlens_q must be a 1-D int32 CUDA tensor of length B + 1")
    if cu_seqlens_q.numel() != batch + 1:
        raise ValueError(f"cu_seqlens_q has {cu_seqlens_q.numel()} entries; kbar has B = {batch} sequences, so B + 1 = {batch + 1} are needed")
    if cu_seqlens_q.device != qi.device:
        raise ValueError("cu_seqlens_q must be on qi's device")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not 1 <= top_k <= 2048:
        raise ValueError(f"top_k must be in [1, 2048], got {top_k}")
    if n_blocks_max < 1:
        raise ValueError("kbar must hold at least one block row")
    if q_pos0 is not None:
        if (
            not isinstance(q_pos0, torch.Tensor)
            or q_pos0.ndim != 1
            or q_pos0.numel() != batch
            or q_pos0.dtype != torch.int32
            or not q_pos0.is_cuda
            or q_pos0.device != qi.device
        ):
            raise ValueError(f"q_pos0 must be a [B={batch}] int32 CUDA tensor on qi's device")
    if w is not None:
        if not isinstance(w, torch.Tensor) or tuple(w.shape) != (total_q, n_heads) or w.dtype not in (torch.bfloat16, torch.float32) or w.device != qi.device:
            raise ValueError(f"w must be a [T={total_q}, n_heads={n_heads}] bf16 or fp32 tensor on qi's device")
        if fp8 and w.dtype != torch.bfloat16:
            raise ValueError(f"the e4m3 scorer takes bfloat16 head weights, got w of {w.dtype}")

    if cu_seqlens_q_scale_padded is not None:
        if not fp8:
            raise ValueError("cu_seqlens_q_scale_padded is the e4m3 arm's padded Q scale prefix; the bf16 scorer takes none (a bf16 pair was given)")
        p = cu_seqlens_q_scale_padded
        if not isinstance(p, torch.Tensor) or p.ndim != 1 or p.dtype != torch.int32 or not p.is_cuda or p.device != qi.device or p.numel() != batch + 1:
            raise ValueError(f"cu_seqlens_q_scale_padded must be a [B + 1 = {batch + 1}] int32 CUDA tensor on qi's device")

    if n_blocks_per_seq is not None:
        counts = _as_host_ints(n_blocks_per_seq, "n_blocks_per_seq")
        if len(counts) != batch:
            raise ValueError(f"n_blocks_per_seq has {len(counts)} entries for B = {batch} sequences")
        cu_q = _as_host_ints(cu_seqlens_q, "cu_seqlens_q")
        pos0 = _as_host_ints(q_pos0, "q_pos0") if q_pos0 is not None else [0] * batch
        for s, n_blocks_s in enumerate(counts):
            seq_q = cu_q[s + 1] - cu_q[s]
            needed = (pos0[s] + seq_q) // block_size
            if not needed <= n_blocks_s <= n_blocks_max:
                raise ValueError(
                    f"sequence {s}: {seq_q} query tokens from position {pos0[s]} see floor(({pos0[s]} + {seq_q}) / {block_size}) = {needed} "
                    f"complete blocks, but n_blocks_per_seq = {n_blocks_s} and kbar holds {n_blocks_max} rows per sequence"
                )

    # Rule 7 first: the CuTe DSL floor (device-free) BEFORE the head-group read -- the module that spells the scorer's packed head
    # groups imports the DSA indexer package, whose __init__ imports its kernel modules, so a sub-floor DSL gets the typed version
    # decline here and never an error from inside the DSL (qsa_select_decode_forward orders it the same way); then the head count (a property
    # of the request, ahead of any device query), then the target and the device family, then the wrapper import.
    dsl_problem = cutedsl_requirement_error("qsa_select_forward")
    if dsl_problem is not None:
        raise NotImplementedError(dsl_problem)
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward._support import SUPPORTED_QHEAD_PER_KV_HEAD_BF16

    if n_heads not in SUPPORTED_QHEAD_PER_KV_HEAD_BF16:
        raise NotImplementedError(f"qsa_select_forward packs n_heads in {SUPPORTED_QHEAD_PER_KV_HEAD_BF16} indexer heads per token, got {n_heads}")
    device_cc = tuple(torch.cuda.get_device_capability(qi.device))
    arch_problem = cutedsl_arch_requirement_error(device_cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if device_cc[0] != 10:
        raise NotImplementedError(f"qsa_select_forward runs the SM100-family indexer scorer (cc 10.x); found cc {device_cc[0]}.{device_cc[1]}")

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward import indexer_forward_top_k_wrapper

    if scale is None:
        scale = head_dim**-0.5
    # Rule 5: the LAUNCH stream first.  Every torch operation below -- the cached constants' creation and their cross-stream ordering
    # (_cached_constant), the e4m3 arm's per-call prefix, the scorer wrapper's own work -- runs on it; stream=None is torch's current
    # stream, where the context is a no-op (recipe R1: the one stream helper, never a hand-rolled ExternalStream).
    with stream_context(stream, qi.device):
        weights = w if w is not None else _constant_weights(total_q * n_heads, qi.device).view(total_q, n_heads)
        cu_seqlens_k = _padded_cu_seqlens_k(batch, n_blocks_max, qi.device)
        fp8_kwargs = {}
        if fp8:
            # The plain-e4m3 scorer = the MXFP8 arm with all-ones scales. Its scale storage is per 128 packed rows: the Q
            # prefix rounds every sequence up to 128 // gcd(128, n_heads) tokens (32 for 4 heads) -- the caller's precomputed
            # one (a plan constant of the block's indexer stage, Rule 1) or derived here on device (no sync) --, the blob
            # covers the largest padded extent that prefix can reach; the K prefix is host-known (kbar is padded per
            # sequence to n_blocks_max rows, its scale span to the next multiple of 128).
            sf_groups = head_dim // _SF_VEC_SIZE
            q_align = _SF_ATOM_ROWS // math.gcd(_SF_ATOM_ROWS, n_heads)
            q_rows_bound = _round_up((total_q + batch * (q_align - 1)) * n_heads, _SF_ATOM_ROWS)
            k_rows_per_seq = _round_up(n_blocks_max, _SF_ATOM_ROWS)
            fp8_kwargs = dict(
                precision="mxfp8",
                q_scale=_unit_scale_blob((1, q_rows_bound, sf_groups), qi.device),
                k_scale=_unit_scale_blob((1, batch * k_rows_per_seq, sf_groups), qi.device),
                cu_seqlens_q_scale_padded=cu_seqlens_q_scale_padded if cu_seqlens_q_scale_padded is not None else _padded_prefix(cu_seqlens_q, q_align),
                cu_seqlens_k_scale_padded=_padded_cu_seqlens_k(batch, k_rows_per_seq, qi.device),
                sf_vec_size=_SF_VEC_SIZE,
            )
        result = indexer_forward_top_k_wrapper(
            qi,
            kbar.view(batch * n_blocks_max, 1, head_dim),
            weights,
            top_k=top_k,
            ratio=block_size,
            qhead_per_kv_head=n_heads,
            sm_scale=float(scale),
            stream=stream,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=n_blocks_max,
            q_causal_offsets=q_pos0,
            topk_indices_global=False,
            cand_buffer=cand_buffer,
            out_indices=block_ids_out,
            out_logits=scores_out,
            cand_batch_offsets=cand_batch_offsets,
            return_softmax=False,
            deterministic=deterministic,
            **fp8_kwargs,
        )
    return TupleDict(block_ids=result["indices"], scores=result["logits"])


def _fp8_cache_arm(qi: torch.Tensor, kbar: torch.Tensor) -> bool:
    """True for the e4m3 pair (the fp8 indexer cache), False for the bf16 pair; anything else is refused."""
    if qi.dtype == torch.bfloat16 and kbar.dtype == torch.bfloat16:
        return False
    if qi.dtype == _E4M3 and kbar.dtype == _E4M3:
        return True
    raise ValueError(f"qi and kbar must both be bfloat16, or both float8_e4m3fn for the fp8 indexer cache; got {qi.dtype} and {kbar.dtype}")


# --------------------------------------------------------------------------------------------- the decode form


def decode_row_positions_and_counts(
    kv_lens: torch.Tensor,
    rows_per_seq: int,
    block_size: int,
    n_blocks_max: int,
    *,
    list_per_sequence: bool,
) -> tuple:
    """The decode step's query positions and per-row visible COMPLETE-block counts (any device; no sync).

    ``kv_lens [B]`` is the logical KV length per sequence INCLUDING this step's ``rows_per_seq`` new tokens, so the
    step's first row (the step-0 token) sits at ``pos0 = kv_lens - rows_per_seq`` and row ``j`` at ``pos0 + j``.  A row
    at position ``p`` sees ``n_vis = clamp(floor((p + 1) / block_size), 0, n_blocks_max)`` complete blocks.

    ``list_per_sequence=True`` returns the step-0 row only (``positions [B]``, ``counts [B]``): the list every row of
    the sequence shares (the speculative-verify reuse).  ``False`` returns one row per new token (``[B, rows_per_seq]``
    each).  The counts are block-granular: consecutive rows see the SAME count except at a block boundary, which is
    why the radix top-k is driven with one length per row (``next_n = 1``) rather than its per-key draft stagger
    (``seq_lens[b] - next_n + j + 1``, which adds one KEY per row -- the decode form of a 1:1 indexer).
    """
    pos0 = kv_lens.to(torch.int32) - int(rows_per_seq)
    if list_per_sequence:
        positions = pos0
    else:
        positions = pos0[:, None] + torch.arange(int(rows_per_seq), dtype=torch.int32, device=kv_lens.device)[None, :]
    counts = torch.div(positions + 1, int(block_size), rounding_mode="floor").clamp_(0, int(n_blocks_max)).to(torch.int32)
    return positions, counts


def qsa_select_decode_forward(
    qi: torch.Tensor,
    kbar: torch.Tensor,
    kv_lens: torch.Tensor,
    *,
    top_k: int = QSA_TOP_K,
    block_size: int = QSA_BLOCK_SIZE,
    list_per_sequence: bool = True,
    n_blocks_per_seq: Optional[Union[Sequence[int], torch.Tensor]] = None,
    scale: Optional[float] = None,
    tie_break: int = 1,
    m_block_size: Optional[int] = None,
    stream=None,
) -> TupleDict:
    """Block selection for a DECODE step (one to a few query tokens per sequence; the speculative-verify rows share
    the step-0 list).

    The prefill selector (:func:`qsa_select_forward`) runs the compact-logits + fused top-k path, sized for thousands of
    query rows.  A decode step has ``rows_per_seq`` (1, or ``1 + drafts``) rows per sequence, so the dense
    ``[rows, n_blocks_max]`` score matrix is small (``B x 4 x n_blocks_max x 4 B``: 2 MiB at 128K tokens, ``B = 4``)
    and the two-launch form -- the DSA unified scorer writing dense ratio-causal scores, then the radix top-k
    kernel with one length per row -- serves it with the smallest MMA tile (``m_block_size = max(16, n_heads)``: a
    16-wide N tile packs the four indexer heads of up to four tokens).

    Contract:

    * ``qi``: ``[B, rows_per_seq, n_heads, head_dim]`` bf16, the indexer queries of this step's tokens, row ``j`` of
      sequence ``b`` at position ``pos0[b] + j`` with ``pos0 = kv_lens - rows_per_seq``; ``n_heads`` in
      (4, 8, 16, 32, 64), ``head_dim == 128``. ``float8_e4m3fn`` together with an e4m3 ``kbar`` = the fp8 indexer
      cache arm (the module docstring): the dense MXFP8 scorer at its 128-row tile with all-ones scales.
    * ``kbar``: ``[B, n_blocks_max, head_dim]`` bf16 (or e4m3 with an e4m3 ``qi``), contiguous, the compressed keys of
      every sequence (one row per complete block; rows at or past a row's visible count are never read);
      ``n_blocks_max`` a multiple of 8 (the fp32 score rows are vector-read; size the compressed-key cache to a
      multiple-of-8 row capacity).
    * ``kv_lens``: ``[B]`` int32 CUDA, the logical KV length per sequence INCLUDING the new tokens (the serving
      stack's vocabulary); every value ``>= rows_per_seq``.
    * ``list_per_sequence=True`` (default): ONE list per sequence from the step-0 row -- ``block_ids [B, top_k]``,
      ``block_lens [B]``, ``scores [B, top_k]``; the ``rows_per_seq - 1`` verify rows reuse it (the sparse core's
      per-sequence list form: row ``j`` sees the listed blocks plus every token from the step-0 tail start to its
      own position).  ``False``: one list per row -- ``block_ids [B, rows_per_seq, top_k]``, ``block_lens
      [B, rows_per_seq]``, ``scores [B, rows_per_seq, top_k]`` -- each row at its own position (the exact per-row
      re-selection).
    * ``block_ids`` are LOCAL block ids of the sequence, a valid prefix of ``block_lens = min(top_k, n_vis)`` entries
      then ``-1``; ``scores`` the selected scores (``-inf`` on the padding).  ``tie_break = 1`` (default) keeps the
      smaller block id at an equal k-th score (the selected SET is reproducible; its order is not); ``0`` leaves the
      cutoff tie to the kernel, ``2`` prefers the larger id.
    * ``top_k`` in ``[1, 2048]`` (the radix kernel's bound; a block declaration caps it at 512); ``block_size`` is the
      compression ratio (4); ``scale`` defaults to ``1 / sqrt(head_dim)``; ``m_block_size`` is the bf16 scorer's tile
      knob (the e4m3 scorer's tile is fixed at 128 rows and refuses another value).

    Launches: the position / count prologue (two small elementwise ops), the scorer, the top-k.  Nothing is read
    back to the host unless ``n_blocks_per_seq`` is given (a host consistency check that reads ``kv_lens``).

    Raises:
        NotImplementedError: a CuTe DSL below the floor / without the target, a device outside the SM100 family
            (cc 10.x), or a head count outside the packed groups -- the message names the version or the set.
        ValueError: a malformed input (shape, dtype, device, alignment, geometry, ``top_k`` past 2048).
    """
    if not isinstance(qi, torch.Tensor) or qi.ndim != 4:
        raise ValueError(f"qi must be a [B, rows_per_seq, n_heads, head_dim] tensor, got {None if not isinstance(qi, torch.Tensor) else tuple(qi.shape)}")
    if not isinstance(kbar, torch.Tensor) or kbar.ndim != 3:
        raise ValueError(f"kbar must be a [B, n_blocks_max, head_dim] tensor, got {None if not isinstance(kbar, torch.Tensor) else tuple(kbar.shape)}")
    batch, rows_per_seq, n_heads, head_dim = (int(x) for x in qi.shape)
    batch_k, n_blocks_max, head_dim_k = (int(x) for x in kbar.shape)
    if batch_k != batch:
        raise ValueError(f"qi holds B = {batch} sequences but kbar holds {batch_k}")
    if rows_per_seq < 1:
        raise ValueError("qi must hold at least one query row per sequence")
    if head_dim != 128 or head_dim_k != 128:
        raise ValueError(f"qsa_select_decode_forward scores head_dim 128 indexer heads, got qi head_dim {head_dim} and kbar head_dim {head_dim_k}")
    fp8 = _fp8_cache_arm(qi, kbar)
    if not qi.is_cuda or kbar.device != qi.device:
        raise ValueError("qi and kbar must be CUDA tensors on one device")
    if qi.stride(-1) != 1:
        raise ValueError(f"qi needs a unit last stride, got strides {tuple(qi.stride())}")
    if not kbar.is_contiguous():
        raise ValueError(f"kbar must be contiguous, got strides {tuple(kbar.stride())}")
    if n_blocks_max < 1 or n_blocks_max % _SCORE_ROW_ALIGN != 0:
        raise ValueError(
            f"kbar must hold a positive multiple of {_SCORE_ROW_ALIGN} block rows per sequence (the fp32 score rows are read in 256-bit vectors), "
            f"got n_blocks_max = {n_blocks_max}: allocate the compressed-key cache with a multiple-of-{_SCORE_ROW_ALIGN} row capacity"
        )
    if (
        not isinstance(kv_lens, torch.Tensor)
        or kv_lens.ndim != 1
        or kv_lens.numel() != batch
        or kv_lens.dtype != torch.int32
        or not kv_lens.is_cuda
        or kv_lens.device != qi.device
    ):
        raise ValueError(f"kv_lens must be a [B={batch}] int32 CUDA tensor on qi's device")
    if block_size < 1:
        raise ValueError(f"block_size must be >= 1, got {block_size}")
    if not 1 <= top_k <= TOP_K_KERNEL_MAX:
        raise ValueError(f"top_k must be in [1, {TOP_K_KERNEL_MAX}] (the radix top-k kernel's bound), got {top_k}")
    if tie_break not in (0, 1, 2):
        raise ValueError(f"tie_break must be 0 (kernel order), 1 (smaller block id) or 2 (larger block id), got {tie_break}")
    if m_block_size is not None and (int(m_block_size) < 16 or int(m_block_size) % 16 != 0):
        raise ValueError(f"m_block_size must be a multiple of 16 (the scorer's N tile), got {m_block_size}")
    if fp8 and m_block_size is not None and int(m_block_size) != _MXFP8_TILE_ROWS:
        raise ValueError(
            f"the e4m3 scorer runs the MXFP8 kernel's fixed {_MXFP8_TILE_ROWS}-row tile; m_block_size = {m_block_size} is a knob of the bf16 scorer only"
        )

    if n_blocks_per_seq is not None:
        counts = _as_host_ints(n_blocks_per_seq, "n_blocks_per_seq")
        if len(counts) != batch:
            raise ValueError(f"n_blocks_per_seq has {len(counts)} entries for B = {batch} sequences")
        lens = _as_host_ints(kv_lens, "kv_lens")
        for s, n_blocks_s in enumerate(counts):
            if lens[s] < rows_per_seq:
                raise ValueError(f"sequence {s}: kv_lens = {lens[s]} is shorter than the {rows_per_seq} new rows of this step")
            needed = lens[s] // block_size
            if not needed <= n_blocks_s <= n_blocks_max:
                raise ValueError(
                    f"sequence {s}: a KV length of {lens[s]} holds floor({lens[s]} / {block_size}) = {needed} complete blocks, "
                    f"but n_blocks_per_seq = {n_blocks_s} and kbar holds {n_blocks_max} rows per sequence"
                )

    # Rule 7 first: the DSL floor and target, then the device family -- before any kernel module is imported, so a
    # sub-floor DSL gets the typed message and never a DSL-internal import error.
    dsl_problem = cutedsl_requirement_error("qsa_select_decode_forward")
    if dsl_problem is not None:
        raise NotImplementedError(dsl_problem)
    device_cc = tuple(torch.cuda.get_device_capability(qi.device))
    arch_problem = cutedsl_arch_requirement_error(device_cc)
    if arch_problem is not None:
        raise NotImplementedError(arch_problem)
    if device_cc[0] != 10:
        raise NotImplementedError(f"qsa_select_decode_forward runs the SM100-family indexer scorer (cc 10.x); found cc {device_cc[0]}.{device_cc[1]}")

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward._support import SUPPORTED_QHEAD_PER_KV_HEAD_BF16

    if n_heads not in SUPPORTED_QHEAD_PER_KV_HEAD_BF16:
        raise NotImplementedError(f"qsa_select_decode_forward packs n_heads in {SUPPORTED_QHEAD_PER_KV_HEAD_BF16} indexer heads per token, got {n_heads}")
    if fp8:
        m_eff = _MXFP8_TILE_ROWS
    else:
        m_eff = max(16, n_heads) if m_block_size is None else int(m_block_size)
    if m_eff % n_heads != 0:
        raise ValueError(f"m_block_size must be a multiple of n_heads = {n_heads} (whole tokens per tile), got {m_block_size}")

    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_forward import indexer_forward_wrapper
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.indexer_top_k import indexer_top_k_wrapper
    from flashinfer.experimental.qwen_3_5_block._cudnn_frost.deepseek_sparse_attention.utils.runtime import torch_stream_context

    if scale is None:
        scale = head_dim**-0.5
    rows = 1 if list_per_sequence else rows_per_seq
    # Rule 5: every torch operation on the launch stream (the operands' device named: the context's default is the current device)
    with torch_stream_context(stream, qi.device):
        positions, n_vis = decode_row_positions_and_counts(kv_lens, rows_per_seq, block_size, n_blocks_max, list_per_sequence=list_per_sequence)
        pos0 = positions if list_per_sequence else positions[:, 0].contiguous()
        q_rows = qi if not list_per_sequence else qi[:, :1]
        weights = _constant_weights(batch * rows * n_heads, qi.device).view(batch, rows, n_heads)
        fp8_kwargs = {}
        if fp8:
            # all-ones E8M0 blobs in the dense (BSHD) packing: (B, packed rows rounded up to 128, 4) for Q, (B, blocks
            # rounded up to 128, 4) for K -- exact shapes, each a contiguous view of its capacity bucket (_unit_scale_blob)
            sf_groups = head_dim // _SF_VEC_SIZE
            fp8_kwargs = dict(
                precision="mxfp8",
                q_scale=_unit_scale_blob((batch, _round_up(rows * n_heads, _SF_ATOM_ROWS), sf_groups), qi.device),
                k_scale=_unit_scale_blob((batch, _round_up(n_blocks_max, _SF_ATOM_ROWS), sf_groups), qi.device),
                sf_vec_size=_SF_VEC_SIZE,
            )
        scores = indexer_forward_wrapper(
            q_rows,
            kbar.view(batch, n_blocks_max, 1, head_dim),
            weights,
            ratio=block_size,
            qhead_per_kv_head=n_heads,
            m_block_size=m_eff,
            sm_scale=float(scale),
            stream=stream,
            q_causal_offsets=pos0,
            **fp8_kwargs,
        )["scores"]
        selected = indexer_top_k_wrapper(
            scores.view(batch * rows, n_blocks_max),
            n_vis.reshape(-1).contiguous(),
            top_k,
            next_n=1,
            return_val=True,
            stream=stream,
            tie_break=tie_break,
        )
        block_lens = torch.minimum(n_vis, torch.full_like(n_vis, int(top_k)))
    out_shape = (batch, top_k) if list_per_sequence else (batch, rows_per_seq, top_k)
    return TupleDict(block_ids=selected["indices"].view(out_shape), block_lens=block_lens, scores=selected["values"].view(out_shape))
