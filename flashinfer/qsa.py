# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Packed BF16 query sparse attention through cuDNN Frontend."""

import functools
import importlib
import os
from types import SimpleNamespace
from typing import Optional

import torch
from packaging.version import Version

from .api_logging import flashinfer_api, flashinfer_experimental_api
from .trace.templates.qsa import qsa_forward_trace, qsa_indexed_forward_trace
from .utils import (
    get_compute_capability,
    register_custom_op,
    register_fake_op,
)

__all__ = ["qsa_forward", "qsa_indexed_forward", "qsa_topk_indexer"]

# QSA stage-1 indexer contract constants (fixed by the kernel).
QSA_INDEXER_NUM_HEADS = 4  # index heads summed into the block score
QSA_INDEXER_HEAD_DIM = 128  # per-head indexer vector width
QSA_INDEXER_BLOCK_SIZE = 4  # tokens per compressed block (CR)
QSA_INDEXER_BLOCK_TOPK = 512  # blocks selected per query
QSA_INDEXER_PAGE_BLOCKS = 64  # compressed blocks per cache page
# CR * BLOCK_TOPK token slots + (CR - 1) causal-tail slots + 1 count column.
QSA_INDEXER_PACKED_WIDTH = 2052

# Backends accepted by qsa_topk_indexer. "specialized" is the only fused
# implementation of this op; "auto" routes to it whenever it can serve the
# call. There is no composable / reference backend in FlashInfer for this
# contract, so a call that "specialized" cannot serve raises
# NotImplementedError (with the reason recorded) instead of falling back.
QSA_TOPK_INDEXER_BACKENDS = ("auto", "specialized")

# Kill switch: set FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1 to force the
# non-fused path. For this op that path does not exist, so the call raises
# NotImplementedError; the switch still guarantees the specialized kernel is
# never dispatched.
_SPECIALIZED_KILL_SWITCH = "FLASHINFER_SPECIALIZED_KERNEL_DISABLE"

# Observability: per-backend dispatch counts and the reasons recorded each
# time "specialized" declined a call.
qsa_topk_indexer_dispatch_stats: dict = {"specialized": 0}
qsa_topk_indexer_fallback_reasons: list = []


@functools.cache
def _cudnn_ops():
    frontend = importlib.import_module("cudnn")
    if Version(frontend.__version__) < Version("1.32.0"):
        raise RuntimeError(
            f"QSA requires nvidia-cudnn-frontend>=1.32.0; found {frontend.__version__}"
        )
    return importlib.import_module("cudnn.ops")


@flashinfer_api(trace=qsa_forward_trace)
def qsa_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    sel_blocks: torch.Tensor,
    lo: torch.Tensor,
    hi: torch.Tensor,
    blk_base: torch.Tensor,
    tok_base: torch.Tensor,
    block_first: torch.Tensor,
    block_last: torch.Tensor,
    scale: float = 1 / 16,
    block_size: int = 4,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute packed query sparse attention from a block-selection bitmap.

    Requires cuDNN Frontend 1.32+, BF16, head dimension 256, 12Q/1KV or
    24Q/2KV, and SM100, SM103 or SM107. cuDNN checks geometry, storage and
    compiler support. All tensors must be contiguous on one CUDA device.

    Parameters
    ----------
    q : torch.Tensor
        BF16 queries, shape ``[T, Hq, 256]``.
    k, v : torch.Tensor
        BF16 keys and values, shape ``[T, Hkv, 256]``.
    sel_blocks : torch.Tensor
        uint8 bitmap, shape ``[T, NB]``. Nonzero selects a global block.
    lo, hi : torch.Tensor
        int32 inclusive key-token bounds, each shape ``[T]``.
    blk_base, tok_base : torch.Tensor
        int32 first global block/token of each query's sequence, shape ``[T]``.
    block_first, block_last : torch.Tensor
        int32 inclusive global token span of each block, shape ``[NB]``.
    scale : float
        Finite attention score scale. Defaults to ``1 / sqrt(256)``.
    block_size : int
        Must be 4. Block numbering restarts at each packed sequence boundary.

    Returns
    -------
    out : torch.Tensor
        BF16 attention output, shape ``[T, Hq, 256]``.
    out_fp32 : torch.Tensor
        FP32 output before the BF16 cast, with the same shape.
    lse : torch.Tensor
        FP32 natural-log softmax normalizer, shape ``[Hq, T]``.

    Notes
    -----
    Metadata must describe one consistent packed partition. Values remain on
    device and are a caller contract. No causal tail is added implicitly.
    Empty rows return zero outputs and negative-infinity LSE.

    This explicit forward call does not register autograd. cuDNN allocates
    outputs and scratch on the input device and uses its current CUDA stream.
    Warm up before CUDA graph capture. Bitmap scratch includes int32
    ``[T, NB]`` indices; prefer :func:`qsa_indexed_forward` for long sequences
    when token indices are already available.
    """
    result = _cudnn_ops().qsa_forward(
        q,
        k,
        v,
        sel_blocks,
        lo,
        hi,
        blk_base,
        tok_base,
        block_first,
        block_last,
        scale,
        block_size,
    )
    return result["out"], result["out_fp32"], result["lse"]


@flashinfer_api(trace=qsa_indexed_forward_trace)
def qsa_indexed_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    indices: torch.Tensor,
    scale: float = 1 / 16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute packed query sparse attention from selected token indices.

    Requires cuDNN Frontend 1.32+, SM100/SM103, BF16 12Q/1KV and dimension
    256. All tensors must be contiguous on one CUDA device.

    Parameters
    ----------
    q : torch.Tensor
        BF16 queries, shape ``[T, 12, 256]``.
    k, v : torch.Tensor
        BF16 keys and values, shape ``[T, 1, 256]``.
    indices : torch.Tensor
        int32 absolute packed token IDs, shape ``[T, W]``. ``W`` is a positive
        multiple of four, at most 2052. Each four-slot group is a distinct
        complete block, four ``-1`` padding slots, or a partial causal tail
        followed by ``-1``. Groups may appear in any order but must not overlap.
        Tokens must belong to the query's sequence and cannot exceed its token
        position. Block origins restart at each sequence boundary. These
        device-value rules are a caller contract, not host-side checks.
    scale : float
        Finite attention score scale, default ``1 / sqrt(256)``.

    Returns
    -------
    out, out_fp32, lse : tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        BF16 output, FP32 output before casting (both ``[T, 12, 256]``), and
        FP32 natural-log LSE (``[12, T]``). Empty rows produce zero outputs
        and negative-infinity LSE.

    Notes
    -----
    Consumes existing indices; no scoring or top-k selection is performed.
    Outputs and scratch are allocated by cuDNN on the input device's current
    CUDA stream. Warm up before capture; replay rereads the indices.
    This explicit forward call does not register autograd.

    Results are not bitwise repeatable. cuDNN's parallel index compaction can
    change the accumulation order, including BF16 output values. Sorting
    caller indices does not remove this internal ordering variation.
    """
    result = _cudnn_ops().qsa_indexed_forward(q, k, v, indices, scale)
    return result["out"], result["out_fp32"], result["lse"]


@functools.cache
def _get_qsa_topk_indexer_module():
    from .jit import gen_qsa_topk_indexer_module

    module = gen_qsa_topk_indexer_module().build_and_load()

    @register_custom_op("flashinfer::qsa_topk_indexer", mutates_args=("packed",))
    def _qsa_topk_indexer(
        packed: torch.Tensor,
        qi: torch.Tensor,
        ck_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cpage_base: torch.Tensor,
    ) -> None:
        module.run(qi, ck_cache, cu_seqlens, cpage_base, packed)

    @register_fake_op("flashinfer::qsa_topk_indexer")
    def _fake_qsa_topk_indexer(
        packed: torch.Tensor,
        qi: torch.Tensor,
        ck_cache: torch.Tensor,
        cu_seqlens: torch.Tensor,
        cpage_base: torch.Tensor,
    ) -> None:
        pass

    return SimpleNamespace(run=_qsa_topk_indexer)


def _check_qsa_topk_indexer_inputs(
    qi: torch.Tensor,
    ck_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cpage_base: torch.Tensor,
    out: Optional[torch.Tensor],
) -> None:
    if not (
        qi.is_cuda and ck_cache.is_cuda and cu_seqlens.is_cuda and cpage_base.is_cuda
    ):
        raise ValueError("qsa_topk_indexer requires CUDA tensors.")
    device = qi.device
    for name, t in (
        ("ck_cache", ck_cache),
        ("cu_seqlens", cu_seqlens),
        ("cpage_base", cpage_base),
    ):
        if t.device != device:
            raise ValueError(f"{name} must be on {device}, got {t.device}.")
    if qi.dtype != torch.float8_e4m3fn or ck_cache.dtype != torch.float8_e4m3fn:
        raise ValueError("qi and ck_cache must be float8_e4m3fn.")
    if cu_seqlens.dtype != torch.int32 or cpage_base.dtype != torch.int32:
        raise ValueError("cu_seqlens and cpage_base must be int32.")
    if qi.ndim != 3 or qi.shape[1:] != (QSA_INDEXER_NUM_HEADS, QSA_INDEXER_HEAD_DIM):
        raise ValueError(
            f"qi must have shape [T, {QSA_INDEXER_NUM_HEADS}, "
            f"{QSA_INDEXER_HEAD_DIM}], got {tuple(qi.shape)}."
        )
    if ck_cache.ndim != 4 or ck_cache.shape[1:] != (
        QSA_INDEXER_PAGE_BLOCKS,
        1,
        QSA_INDEXER_HEAD_DIM,
    ):
        raise ValueError(
            f"ck_cache must have shape [num_pages, {QSA_INDEXER_PAGE_BLOCKS}, 1, "
            f"{QSA_INDEXER_HEAD_DIM}], got {tuple(ck_cache.shape)}."
        )
    if cu_seqlens.ndim != 1 or cpage_base.ndim != 1:
        raise ValueError("cu_seqlens and cpage_base must be 1-D.")
    if cu_seqlens.shape[0] != cpage_base.shape[0] + 1:
        raise ValueError(
            "cu_seqlens must have shape [batch + 1] matching cpage_base [batch]; got "
            f"{tuple(cu_seqlens.shape)} and {tuple(cpage_base.shape)}."
        )
    for name, t in (
        ("qi", qi),
        ("ck_cache", ck_cache),
        ("cu_seqlens", cu_seqlens),
        ("cpage_base", cpage_base),
    ):
        if not t.is_contiguous():
            raise ValueError(f"{name} must be contiguous.")
    # Contiguity does not establish pointer alignment: a contiguous view into
    # offset storage keeps the offset address, and the kernels use 8- and
    # 16-byte vector accesses on these buffers. Fresh torch allocations are
    # 256-byte aligned; only offset storage views can fail this.
    for name, t in (("qi", qi), ("ck_cache", ck_cache)):
        if t.data_ptr() % 16:
            raise ValueError(f"{name} must be 16-byte aligned (storage-offset view).")
    if out is not None:
        if out.device != device or out.dtype != torch.int32:
            raise ValueError("out must be an int32 tensor on the input device.")
        if (
            out.shape != (qi.shape[0], QSA_INDEXER_PACKED_WIDTH)
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"out must be contiguous with shape [T, {QSA_INDEXER_PACKED_WIDTH}], "
                f"got {tuple(out.shape)}."
            )
        if out.data_ptr() % 16:
            raise ValueError("out must be 16-byte aligned (storage-offset view).")


def _qsa_topk_indexer_unservable_reason(qi: torch.Tensor) -> Optional[str]:
    """Why the "specialized" backend cannot serve this call, or None.

    GPU-free guards first (the kill switch), then the device guard. Contract
    violations (dtype / geometry / contiguity) are caller errors and raise
    ValueError in _check_qsa_topk_indexer_inputs instead.
    """
    if os.environ.get(_SPECIALIZED_KILL_SWITCH) == "1":
        return f"{_SPECIALIZED_KILL_SWITCH}=1 disables the specialized kernel"
    cc = get_compute_capability(qi.device)
    if cc != (10, 7):
        return f"specialized kernel targets SM107 only; device is sm{cc[0]}{cc[1]}"
    return None


@flashinfer_experimental_api(
    feature="QSA stage-1 token top-k block indexer (FP8 paged compressed keys, SM107)"
)
def qsa_topk_indexer(
    qi: torch.Tensor,
    ck_cache: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cpage_base: torch.Tensor,
    out: Optional[torch.Tensor] = None,
    *,
    backend: str = "auto",
) -> torch.Tensor:
    r"""Select the top-scoring key blocks for each packed query token (QSA
    stage 1) and expand them to the packed request-local token indices the
    SM107 QSA stage-2 sparse-attention path consumes.

    For every query token, each visible compressed block is scored as
    ``sum_h relu(qi[t, h] . ck[block])`` over the ``4`` index heads, the top
    ``512`` blocks are selected with a **deterministic tie-break (the lower
    block index wins)**, selections are expanded to ``4`` token ids per block,
    the partial causal tail of up to ``3`` tokens is appended, and the result
    is written in ascending token-id order with ``-1`` padding and the valid
    count in the last column.

    A token at position ``pos`` within its request sees ``(pos + 1) // 4``
    complete compressed blocks (including its own); rows whose visible blocks
    all fit in the selection budget are emitted directly as the identity
    selection without touching ``qi`` or ``ck_cache``.

    The kernel is bitwise deterministic, allocation-free at run time and safe
    to capture in a CUDA graph (warm up once before capture to build the JIT
    module).

    Backend selection follows the specialized-kernel pattern:
    ``QSA_TOPK_INDEXER_BACKENDS = ("auto", "specialized")``. ``"specialized"``
    (this kernel) is the only fused implementation of the op, and FlashInfer
    has no composable / reference implementation of this contract, so when the
    specialized kernel cannot serve a call — the device is not SM107, or the
    kill switch ``FLASHINFER_SPECIALIZED_KERNEL_DISABLE=1`` is set — both
    ``backend="auto"`` and ``backend="specialized"`` raise
    ``NotImplementedError`` naming the reason rather than silently running a
    different backend; the reason is also appended to
    ``flashinfer.qsa.qsa_topk_indexer_fallback_reasons``. Successful dispatches
    are counted in ``flashinfer.qsa.qsa_topk_indexer_dispatch_stats``. The kill
    switch guarantees the specialized kernel is never dispatched even on
    ``backend="auto"``.

    Parameters
    ----------
    qi : torch.Tensor
        FP8 (e4m3) indexer queries, shape ``[T, 4, 128]``; ``T`` is the packed
        token count over all requests.
    ck_cache : torch.Tensor
        FP8 (e4m3) paged compressed-key cache, shape
        ``[num_pages, 64, 1, 128]``: 64 compressed blocks (of 4 tokens each)
        per page.
    cu_seqlens : torch.Tensor
        int32 ``[batch + 1]`` request boundaries in the packed token dimension
        (``cu_seqlens[-1] == T``). Values are read on the device only.
    cpage_base : torch.Tensor
        int32 ``[batch]`` first page of each request's contiguous page range
        in ``ck_cache``. The range must hold every compressed block the
        request's tokens can see: with ``len_r = cu_seqlens[r + 1] -
        cu_seqlens[r]``, pages ``cpage_base[r]`` through ``cpage_base[r] +
        ceil(len_r / 256) - 1`` must exist in ``ck_cache``. Like the
        ``cu_seqlens`` values themselves (read on the device only), this is a
        device-value caller contract — the host cannot see the lengths to
        check it, and the kernels read keys without re-checking the cache
        extent, so a violating call reads out of bounds.
    out : Optional[torch.Tensor]
        Caller-owned int32 ``[T, 2052]`` output; allocated when omitted.
    backend : str
        ``"auto"`` (default) or ``"specialized"``. Both resolve to the
        specialized SM107 kernel when it can serve the call; see the backend
        selection notes above for the failure behavior.

    Returns
    -------
    packed : torch.Tensor
        int32 ``[T, 2052]``: the selected token ids of each row —
        **request-local** logical positions, ``0`` naming the row's request's
        first token — sorted ascending, then ``-1`` padding; the last column
        holds the row's valid count. This is the packed-selection layout the
        SM107 QSA stage-2 sparse-attention path consumes (and the layout
        vLLM's Qwen QSA serving path exchanges between the stages).
        :func:`qsa_indexed_forward` is NOT a direct consumer: it takes
        absolute packed token ids and no count column, so feeding it requires
        adding each row's request start (``cu_seqlens[req]``) to the valid
        ids and overwriting the final count column with ``-1`` — and it has
        its own support matrix (SM100/SM103 at the supported cuDNN Frontend),
        which does not include this kernel's SM107 target, so that pairing
        also requires moving the indices to a supported device.
    """
    if backend not in QSA_TOPK_INDEXER_BACKENDS:
        raise ValueError(
            f"backend must be one of {QSA_TOPK_INDEXER_BACKENDS}, got {backend!r}."
        )
    _check_qsa_topk_indexer_inputs(qi, ck_cache, cu_seqlens, cpage_base, out)
    reason = _qsa_topk_indexer_unservable_reason(qi)
    if reason is not None:
        qsa_topk_indexer_fallback_reasons.append(reason)
        raise NotImplementedError(
            f"qsa_topk_indexer cannot run the specialized kernel ({reason}), "
            "and FlashInfer has no composable fallback implementation of this "
            "contract; use the serving stack's generic indexer path instead."
        )
    if out is None:
        out = torch.empty(
            (qi.shape[0], QSA_INDEXER_PACKED_WIDTH),
            dtype=torch.int32,
            device=qi.device,
        )
    _get_qsa_topk_indexer_module().run(out, qi, ck_cache, cu_seqlens, cpage_base)
    qsa_topk_indexer_dispatch_stats["specialized"] += 1
    return out
