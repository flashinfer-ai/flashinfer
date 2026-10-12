"""
Copyright (c) 2026 by NVIDIA CORPORATION.

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

from __future__ import annotations

import ctypes
import math
from typing import Optional

import torch

from ...jit.cpp_ext import is_cuda_version_at_least
from .jit import get_union_gqa_module

BLOCK_SIZE = 64
_HEAD_DIM = 128
_GQA_GROUP_SIZE = 16
_MAX_TOPK_BLOCKS = 16
_LOG2_E_FLOAT32 = ctypes.c_float(1.4426950408889634).value
_MIN_NORMAL_FLOAT32 = 2.0**-126


def _check_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    ndim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Validate tensor type, rank, dtype, device, and contiguous storage."""
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}D, got shape {tuple(tensor.shape)}")
    if tensor.dtype != dtype:
        raise TypeError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _check_tma_alignment(tensor: torch.Tensor, name: str) -> None:
    """Reject tensors whose data pointers lack the required 16-byte alignment."""
    if tensor.data_ptr() % 16 != 0:
        raise ValueError(f"{name} must be 16-byte aligned")


def _overlaps(lhs: torch.Tensor, rhs: torch.Tensor) -> bool:
    """Check byte ranges; all tensors passed here are contiguous."""

    lhs_start = lhs.data_ptr()
    rhs_start = rhs.data_ptr()
    return lhs_start < rhs_start + rhs.numel() * rhs.element_size() and rhs_start < (
        lhs_start + lhs.numel() * lhs.element_size()
    )


def union_gqa_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    *,
    softmax_scale: Optional[float] = None,
    max_seqlen_q: Optional[int] = None,
    causal: bool = True,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run causal union-GQA sparse prefill on SM100.

    Every value ``i`` in ``block_indices`` denotes the 64-token KV interval
    ``[64 * i, 64 * (i + 1))`` in the same sequence as its query. Negative
    values are padding. Valid block IDs in one query must be unique; their
    order does not affect correctness, although similar ordering across nearby
    queries usually reduces union work.

    This first backend is intentionally narrow: 16-byte-aligned BF16 flat
    self-attention, head dimension 128, GQA group size 16, at most 16 selected
    blocks per query, causal masking, and exact compute capability 10.0.

    Parameters
    ----------
    q : torch.Tensor
        Contiguous, 16-byte-aligned BF16 tensor of shape
        ``(total_q, num_qo_heads, 128)``.
    k, v : torch.Tensor
        Contiguous, 16-byte-aligned BF16 tensors of shape
        ``(total_q, num_kv_heads, 128)``.
    block_indices : torch.Tensor
        Contiguous int32 tensor of shape
        ``(total_q, num_kv_heads, topk_blocks)``. Each index identifies one
        64-token block; negative entries are ignored.
    cu_seqlens_q : torch.Tensor
        Int32 cumulative sequence lengths of shape ``(batch_size + 1,)``.
        Values must start at zero, end at ``total_q``, and be nondecreasing.
        CPU input is accepted and copied to ``q.device`` outside CUDA graph
        capture. CPU values are checked before launch; invalid CUDA values
        raise an asynchronous device error.
    softmax_scale : float, optional
        Defaults to ``1 / sqrt(128)``. Must remain a positive normal float32
        whose product with log2(e) is finite.
    max_seqlen_q : int, optional
        An upper bound on every sequence length represented by
        ``cu_seqlens_q``. A value smaller than the actual maximum is invalid.
        Supplying it avoids a device-to-host synchronization and is required
        during CUDA graph capture. Invalid CUDA bounds raise an asynchronous
        device error.
    causal : bool, default=True
        The initial kernel supports causal self-attention only.
    out : torch.Tensor, optional
        Optional contiguous, 16-byte-aligned BF16 output with the same shape
        as ``q``. It must not overlap any input tensor.

    Returns
    -------
    torch.Tensor
        Sparse-attention output with the same shape and dtype as ``q``.
    """

    if not isinstance(q, torch.Tensor):
        raise TypeError("q must be a torch.Tensor")
    if q.device.type != "cuda":
        raise ValueError(f"q must be a CUDA tensor, got {q.device}")
    if torch.cuda.get_device_capability(q.device) != (10, 0):
        major, minor = torch.cuda.get_device_capability(q.device)
        raise RuntimeError(
            "union_gqa_prefill requires compute capability 10.0 "
            f"(SM100), got {major}.{minor}"
        )
    if not is_cuda_version_at_least("12.8"):
        raise RuntimeError("union_gqa_prefill on SM100 requires CUDA 12.8 or newer")
    if not causal:
        raise ValueError("union_gqa_prefill currently supports causal=True only")

    device = q.device
    _check_tensor(q, "q", ndim=3, dtype=torch.bfloat16, device=device)
    _check_tensor(k, "k", ndim=3, dtype=torch.bfloat16, device=device)
    _check_tensor(v, "v", ndim=3, dtype=torch.bfloat16, device=device)
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        _check_tma_alignment(tensor, name)
    _check_tensor(
        block_indices,
        "block_indices",
        ndim=3,
        dtype=torch.int32,
        device=device,
    )

    total_q, num_qo_heads, head_dim = q.shape
    if total_q <= 0:
        raise ValueError("total_q must be positive")
    if head_dim != _HEAD_DIM:
        raise ValueError(f"q head dimension must be {_HEAD_DIM}, got {head_dim}")
    if k.shape[0] != total_q or v.shape[0] != total_q:
        raise ValueError("k/v total-token dimension must match q")
    if k.shape != v.shape or k.shape[2] != _HEAD_DIM:
        raise ValueError(
            "k and v must have identical shape (total_q, num_kv_heads, 128)"
        )
    num_kv_heads = k.shape[1]
    if num_kv_heads <= 0 or num_qo_heads != _GQA_GROUP_SIZE * num_kv_heads:
        raise ValueError(
            "union_gqa_prefill requires num_qo_heads / num_kv_heads == "
            f"{_GQA_GROUP_SIZE}, got {num_qo_heads} / {num_kv_heads}"
        )
    if block_indices.shape[:2] != (total_q, num_kv_heads):
        raise ValueError(
            "block_indices must have shape "
            f"({total_q}, {num_kv_heads}, topk_blocks), got "
            f"{tuple(block_indices.shape)}"
        )
    topk_blocks = block_indices.shape[2]
    if not 1 <= topk_blocks <= _MAX_TOPK_BLOCKS:
        raise ValueError(
            f"topk_blocks must be in [1, {_MAX_TOPK_BLOCKS}], got {topk_blocks}"
        )

    if not isinstance(cu_seqlens_q, torch.Tensor):
        raise TypeError("cu_seqlens_q must be a torch.Tensor")
    if cu_seqlens_q.ndim != 1 or cu_seqlens_q.dtype != torch.int32:
        raise TypeError("cu_seqlens_q must be a 1D int32 tensor")
    if cu_seqlens_q.numel() < 2:
        raise ValueError("cu_seqlens_q must contain at least two entries")
    cpu_offsets = None
    if cu_seqlens_q.device.type == "cpu":
        cpu_offsets = cu_seqlens_q.tolist()
        if (
            cpu_offsets[0] != 0
            or cpu_offsets[-1] != total_q
            or any(
                end < start
                for start, end in zip(cpu_offsets, cpu_offsets[1:], strict=False)
            )
        ):
            raise ValueError(
                "cu_seqlens_q must be nondecreasing, start at zero, and end at total_q"
            )
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and cu_seqlens_q.device != device:
        raise ValueError(
            "cu_seqlens_q must already be on q.device during CUDA graph capture"
        )
    cu_seqlens_q = cu_seqlens_q.to(device=device, non_blocking=True).contiguous()

    if softmax_scale is None:
        softmax_scale = _HEAD_DIM**-0.5
    softmax_scale = ctypes.c_float(float(softmax_scale)).value
    log2_scale = ctypes.c_float(softmax_scale * _LOG2_E_FLOAT32).value
    if softmax_scale < _MIN_NORMAL_FLOAT32 or not math.isfinite(log2_scale):
        raise ValueError("softmax_scale is outside the supported float32 range")

    if max_seqlen_q is None:
        if capturing:
            raise ValueError("max_seqlen_q must be provided during CUDA graph capture")
        seqlens = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        max_seqlen_q = int(seqlens.max().item())
    max_seqlen_q = int(max_seqlen_q)
    if not 1 <= max_seqlen_q <= total_q:
        raise ValueError(f"max_seqlen_q must be in [1, total_q], got {max_seqlen_q}")
    if cpu_offsets is not None and any(
        end - start > max_seqlen_q
        for start, end in zip(cpu_offsets, cpu_offsets[1:], strict=False)
    ):
        raise ValueError("max_seqlen_q is smaller than a sequence length")

    if out is None:
        out = torch.empty_like(q)
    else:
        _check_tensor(out, "out", ndim=3, dtype=torch.bfloat16, device=device)
        if out.shape != q.shape:
            raise ValueError(
                f"out must have the same shape as q, got {tuple(out.shape)}"
            )
        if any(
            _overlaps(out, tensor) for tensor in (q, k, v, block_indices, cu_seqlens_q)
        ):
            raise ValueError("out must not overlap any input tensor")
        _check_tma_alignment(out, "out")

    get_union_gqa_module().run(
        q,
        k,
        v,
        block_indices,
        cu_seqlens_q,
        out,
        softmax_scale,
        max_seqlen_q,
    )
    return out


__all__ = ["BLOCK_SIZE", "union_gqa_prefill"]
