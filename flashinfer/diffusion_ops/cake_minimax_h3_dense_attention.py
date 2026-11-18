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

import functools
import math
from typing import NamedTuple, Optional

import torch
import tvm_ffi

from ..jit.cake_minimax_h3_dense_attention import ATTENTION
from ..utils import register_custom_op, register_fake_op

MINIMAX_H3_NUM_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_WIDTH = MINIMAX_H3_NUM_HEADS * MINIMAX_H3_HEAD_DIM
MINIMAX_H3_MAX_TOKENS = 131072
# BF16(FP32(1 / sqrt(128))): the query is scaled by this constant and the product rounded to BF16
# before the softmax (softmax scale 1.0), matching the MiniMax-H3 video DiT graph.
MINIMAX_H3_QUERY_SCALE_BF16 = 0.08837890625

# The kernel's geometry: one persistent 256-thread CTA per SM walks the (head, 128-row query tile)
# work items over 64-key K/V tiles; the items of the last wave split over at most 8 K/V ranges.
_TILE_Q = 128
_TILE_KV = 64
_SPLIT_K_MAX = 8
# Per CTA of the persistent grid: one split-KV partial-state slot (64 output accumulators + m[2] +
# d[2] FP32 values per thread) and one INT32 arrival counter.
_PARTIAL_BYTES_PER_CTA = 68 * 2 * _TILE_Q * 4
_COUNTER_BYTES_PER_CTA = 4
# log2(e): the query is pre-scaled by BF16(1 / sqrt(128)), so the softmax scale is 1.0.
_SOFTMAX_SCALE_LOG2 = math.log2(math.e)
# Two BF16 copies of MINIMAX_H3_QUERY_SCALE_BF16 (bits 0x3DB5).
_QUERY_SCALE_BF16X2 = 0x3DB53DB5


class MinimaxH3DenseAttentionPlan(NamedTuple):
    """The launch plan of one call: the persistent grid and the kernel's plan scalars."""

    grid: int
    num_q_tiles: int
    num_kv_tiles: int
    total_work: int
    split_base: int
    split_k: int
    split_tiles: int


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def minimax_h3_dense_attention_launch_plan(
    tokens: int, num_sms: int
) -> MinimaxH3DenseAttentionPlan:
    """The launch plan of ``tokens`` rows on a device of ``num_sms`` SMs.

    ``ceil(tokens / 128) * 56`` (head, query tile) items run on a persistent grid of
    ``max(1, min(items, num_sms))`` CTAs.  The ``items % grid`` items of the last wave are each
    split over ``split_k = min(8, grid // tail)`` K/V ranges of ``split_tiles`` 64-key tiles (lowered
    while a range would be empty); work ids at or above ``split_base`` are those parts, and the
    CTA that completes an item merges its parts in-kernel.
    """

    tokens, num_sms = int(tokens), int(num_sms)
    if not 1 <= tokens <= MINIMAX_H3_MAX_TOKENS:
        raise ValueError(
            f"tokens must lie in [1, {MINIMAX_H3_MAX_TOKENS}], got {tokens}"
        )
    if num_sms < 1:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    num_q_tiles = _ceil_div(tokens, _TILE_Q)
    num_kv_tiles = _ceil_div(tokens, _TILE_KV)
    items = num_q_tiles * MINIMAX_H3_NUM_HEADS
    grid = max(1, min(items, num_sms))
    tail = items % grid
    split_k = 1
    if tail:
        split_k = min(_SPLIT_K_MAX, grid // tail)
        while (
            split_k > 1
            and (split_k - 1) * _ceil_div(num_kv_tiles, split_k) >= num_kv_tiles
        ):
            split_k -= 1
    split_base = items - tail if split_k > 1 else items
    return MinimaxH3DenseAttentionPlan(
        grid=grid,
        num_q_tiles=num_q_tiles,
        num_kv_tiles=num_kv_tiles,
        total_work=split_base + (items - split_base) * split_k,
        split_base=split_base,
        split_k=split_k,
        split_tiles=_ceil_div(num_kv_tiles, split_k),
    )


def minimax_h3_dense_attention_workspace_bytes(num_sms: int) -> int:
    """Bytes of the per-device workspace: one partial-state slot and one arrival counter per CTA
    of the largest persistent grid (``num_sms`` CTAs)."""

    if int(num_sms) < 1:
        raise ValueError(f"num_sms must be positive, got {num_sms}")
    return int(num_sms) * (_PARTIAL_BYTES_PER_CTA + _COUNTER_BYTES_PER_CTA)


def _device_index(device: torch.device) -> int:
    return device.index if device.index is not None else torch.cuda.current_device()


@functools.cache
def _target(index: int) -> str:
    """The exact target the program ships for device ``index`` (``ATTENTION.targets``)."""
    capability = tuple(int(value) for value in torch.cuda.get_device_capability(index))
    target = ATTENTION.target_for((capability[0], capability[1]))
    if target is None:
        raise RuntimeError(
            "MiniMax-H3 dense attention requires compute capability "
            f"9.0, 10.0, 10.3, 12.0 or 12.1, got {capability[0]}.{capability[1]}"
        )
    return target


@functools.cache
def _num_sms(index: int) -> int:
    return int(torch.cuda.get_device_properties(index).multi_processor_count)


_WORKSPACES: dict = {}


def _workspace(device: torch.device) -> torch.Tensor:
    """Per-device zero-initialized workspace for the in-kernel tail split-KV merge.

    The kernel rewinds its arrival counters after every launch, so one buffer per device
    serves every call on that device (stream-ordered launches).
    """

    index = _device_index(device)
    workspace = _WORKSPACES.get(index)
    if workspace is None:
        _target(index)
        workspace = torch.zeros(
            minimax_h3_dense_attention_workspace_bytes(_num_sms(index)),
            dtype=torch.uint8,
            device=torch.device("cuda", index),
        )
        _WORKSPACES[index] = workspace
    return workspace


def _check_rows(
    name: str, rows: torch.Tensor, tokens: int, device: torch.device
) -> None:
    if rows.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {rows.dtype}")
    if rows.ndim != 2 or tuple(rows.shape) != (tokens, MINIMAX_H3_WIDTH):
        raise ValueError(
            f"{name} must have shape [{tokens}, {MINIMAX_H3_WIDTH}], got {tuple(rows.shape)}"
        )
    if not rows.is_cuda or not rows.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")
    if rows.device != device:
        raise ValueError(f"{name} must be on the same CUDA device as q ({device})")
    if rows.data_ptr() % 16 != 0:
        raise ValueError(f"{name} must be 16-byte aligned")


@register_custom_op(
    "flashinfer::minimax_h3_dense_attention", mutates_args=("out", "workspace")
)
def _minimax_h3_dense_attention_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    index = _device_index(q.device)
    target = _target(index)
    num_sms = _num_sms(index)
    nbytes = minimax_h3_dense_attention_workspace_bytes(num_sms)
    if (
        workspace.dtype != torch.uint8
        or workspace.ndim != 1
        or not workspace.is_contiguous()
        or workspace.device != q.device
        or workspace.numel() < nbytes
    ):
        raise ValueError(
            "workspace must be a contiguous 1-D uint8 tensor on the device of q holding "
            f"at least {nbytes} bytes"
        )
    tokens = int(q.shape[0])
    plan = minimax_h3_dense_attention_launch_plan(tokens, num_sms)
    # The partial-state slots, then the arrival counters, of the per-device workspace.
    partial_bytes = num_sms * _PARTIAL_BYTES_PER_CTA
    partial_ws = workspace[:partial_bytes].view(torch.float32)
    split_counters = workspace[
        partial_bytes : partial_bytes + num_sms * _COUNTER_BYTES_PER_CTA
    ].view(torch.int32)
    rows = (tokens, MINIMAX_H3_NUM_HEADS, MINIMAX_H3_HEAD_DIM)
    with tvm_ffi.use_torch_stream():
        ATTENTION.entry(target)(
            q.view(rows),
            k.view(rows),
            v.view(rows),
            out,
            partial_ws,
            split_counters,
            tokens,
            plan.num_q_tiles,
            plan.num_kv_tiles,
            plan.total_work,
            plan.split_base,
            plan.split_k,
            plan.split_tiles,
            _SOFTMAX_SCALE_LOG2,
            _QUERY_SCALE_BF16X2,
            plan.grid,
            1,
            1,
        )


@register_fake_op("flashinfer::minimax_h3_dense_attention")
def _minimax_h3_dense_attention_fake(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    workspace: torch.Tensor,
) -> None:
    pass


def minimax_h3_dense_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    r"""Dense non-causal BF16 self-attention for the MiniMax-H3 video DiT.

    Computes, per head ``h`` of 56 heads with head_dim 128 and batch 1::

        qs = bf16(q * bf16(1 / sqrt(128)))        # product rounded to BF16 before the softmax
        y  = softmax(qs @ k^T) @ v                # softmax scale 1.0, no mask, no dropout

    ``q``, ``k``, ``v`` and ``y`` are contiguous BF16 ``[tokens, 7168]`` rows whose element
    ``[t, h * 128 + d]`` holds channel ``d`` of head ``h``; the row<->head re-layout, the
    query scale and the attention run in one kernel.  One runtime-variable kernel serves any
    ``1 <= tokens <= 131072``: a persistent 256-thread CTA per SM walks the (head, 128-row
    query tile) work items with a two-stage K/V TMA ring and a ping-pong schedule between
    its two warp groups; the items of the last wave are split across the otherwise idle CTAs
    by K/V range and merged in-kernel through a small per-device workspace
    (:func:`minimax_h3_dense_attention_launch_plan`).

    Parameters
    ----------
    q, k, v : torch.Tensor
        Contiguous ``bfloat16`` CUDA tensors of shape ``[tokens, 7168]``.
    out : Optional[torch.Tensor]
        Optional pre-allocated output of the same shape/dtype; allocated when omitted.

    Returns
    -------
    torch.Tensor
        ``bfloat16`` ``[tokens, 7168]`` attention output in the input row layout.
    """

    tokens = int(q.shape[0]) if q.ndim == 2 else -1
    if not 1 <= tokens <= MINIMAX_H3_MAX_TOKENS:
        raise ValueError(
            f"tokens must lie in [1, {MINIMAX_H3_MAX_TOKENS}], got q of shape {tuple(q.shape)}"
        )
    for name, rows in (("q", q), ("k", k), ("v", v)):
        _check_rows(name, rows, tokens, q.device)
    if out is None:
        out = torch.empty_like(q)
    else:
        _check_rows("out", out, tokens, q.device)
    _minimax_h3_dense_attention_impl(q, k, v, out, _workspace(q.device))
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_MAX_TOKENS",
    "MINIMAX_H3_NUM_HEADS",
    "MINIMAX_H3_QUERY_SCALE_BF16",
    "MINIMAX_H3_WIDTH",
    "MinimaxH3DenseAttentionPlan",
    "minimax_h3_dense_attention",
    "minimax_h3_dense_attention_launch_plan",
    "minimax_h3_dense_attention_workspace_bytes",
]
