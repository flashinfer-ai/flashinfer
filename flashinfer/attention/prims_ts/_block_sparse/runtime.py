# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runtime validation and launch adapter for block-sparse attention."""

from collections.abc import Callable
from dataclasses import dataclass
import math

import torch

from .._tensor_aliasing import _validate_out_does_not_overlap_inputs
from ..decode import (
    _validate_16byte_alignment,
    _validate_exact_compact_strides,
    _validate_scale,
)


@dataclass(frozen=True)
class _BlockSparseRunArgs:
    """Validated launch arguments with ``out`` and ``sm_scale`` materialized."""

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    out: torch.Tensor
    sm_scale: float


def _validate_bshd_tensor(
    tensor: torch.Tensor,
    name: str,
    *,
    expected_shape: tuple[int, int, int, int],
    expected_dtype: torch.dtype,
    expected_device: torch.device,
) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if tensor.ndim != 4 or tuple(tensor.shape) != expected_shape:
        raise ValueError(
            f"{name} must have compact BSHD shape {expected_shape}, "
            f"got {tuple(tensor.shape)}"
        )
    if tensor.dtype != expected_dtype:
        raise ValueError(f"{name} must have dtype {expected_dtype}, got {tensor.dtype}")
    if tensor.device != expected_device:
        raise ValueError(
            f"{name} must be on planned device {expected_device}, got {tensor.device}"
        )
    _validate_exact_compact_strides(tensor, name, "BSHD")
    _validate_16byte_alignment(tensor, name)


def validate_block_sparse_run(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor,
    device: torch.device,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    head_dim: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    output_dtype: torch.dtype,
    sm_scale: float | None,
    out: torch.Tensor | None,
) -> _BlockSparseRunArgs:
    """Validate one run and allocate only an omitted output tensor.

    Q/O use compact ``[B, Sq, H, D]`` and K/V use compact
    ``[B, Skv, H, D]`` on the planned device and dtype. An explicit output is
    returned by identity and may not overlap any live launch input.
    ``sm_scale=None`` is materialized as ``1 / sqrt(D)``.
    """

    q_shape = (batch_size, seq_len_q, num_heads, head_dim)
    kv_shape = (batch_size, seq_len_kv, num_heads, head_dim)
    for tensor, name, shape, dtype in (
        (q, "q", q_shape, q_dtype),
        (k, "k", kv_shape, kv_dtype),
        (v, "v", kv_shape, kv_dtype),
    ):
        _validate_bshd_tensor(
            tensor,
            name,
            expected_shape=shape,
            expected_dtype=dtype,
            expected_device=device,
        )
    effective_scale = _validate_scale(
        1.0 / math.sqrt(head_dim) if sm_scale is None else sm_scale,
        "sm_scale",
    )
    if out is None:
        out = torch.empty(q_shape, device=device, dtype=output_dtype)
    else:
        _validate_bshd_tensor(
            out,
            "out",
            expected_shape=q_shape,
            expected_dtype=output_dtype,
            expected_device=device,
        )
        _validate_out_does_not_overlap_inputs(
            out,
            ("q", q),
            ("k", k),
            ("v", v),
            ("block_indptr", block_indptr),
            ("block_indices", block_indices),
            ("kv_valid_bits", kv_valid_bits),
        )
    return _BlockSparseRunArgs(q=q, k=k, v=v, out=out, sm_scale=effective_scale)


def launch_block_sparse(
    run_args: _BlockSparseRunArgs,
    *,
    block_indptr: torch.Tensor,
    block_indices: torch.Tensor,
    kv_valid_bits: torch.Tensor,
    row_route_offsets: torch.Tensor,
    route_workspace: torch.Tensor,
    max_blocks_per_row: int | None,
    compiled: Callable[..., object],
) -> torch.Tensor:
    """Invoke the fused prepare/attention adapter asynchronously.

    ``kv_valid_bits`` is always present in this ABI: it is either the
    effective caller mask or a plan-owned dummy for an unmasked specialization.
    ``max_blocks_per_row=None`` is passed as the ``-1`` sentinel understood by
    prepare; an explicit value is checked as live data and does not specialize
    the compiled adapter.
    The adapter first rewrites ``route_workspace`` from live metadata, then
    launches attention using only the route headers and metadata on the same
    CUDA stream. It writes ``run_args.out`` in place and this function returns
    that same tensor.
    """

    compiled(
        run_args.q,
        run_args.k,
        run_args.v,
        run_args.out,
        block_indptr,
        block_indices,
        kv_valid_bits,
        row_route_offsets,
        route_workspace,
        -1 if max_blocks_per_row is None else max_blocks_per_row,
        run_args.sm_scale,
    )
    return run_args.out


__all__ = [
    "_BlockSparseRunArgs",
    "launch_block_sparse",
    "validate_block_sparse_run",
]
