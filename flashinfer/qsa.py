# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0

"""Packed BF16 query sparse attention through cuDNN Frontend."""

import functools
import importlib

import torch
from packaging.version import Version

from .api_logging import flashinfer_api
from .trace.templates.qsa import qsa_forward_trace, qsa_indexed_forward_trace

__all__ = ["qsa_forward", "qsa_indexed_forward"]


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
