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
from typing import Optional

import torch

from ..jit.cake_minimax_h3_dense_attention import gen_minimax_h3_dense_attention_module
from ..utils import register_custom_op, register_fake_op

MINIMAX_H3_NUM_HEADS = 56
MINIMAX_H3_HEAD_DIM = 128
MINIMAX_H3_WIDTH = MINIMAX_H3_NUM_HEADS * MINIMAX_H3_HEAD_DIM
MINIMAX_H3_MAX_TOKENS = 131072
# BF16(FP32(1 / sqrt(128))): the query is scaled by this constant and the product rounded to BF16
# before the softmax (softmax scale 1.0), matching the MiniMax-H3 video DiT graph.
MINIMAX_H3_QUERY_SCALE_BF16 = 0.08837890625


@functools.cache
def _get_module():
    return gen_minimax_h3_dense_attention_module().build_and_load()


_WORKSPACES: dict = {}


def _workspace(device: torch.device) -> torch.Tensor:
    """Per-device zero-initialized workspace for the in-kernel tail split-KV merge.

    The kernel rewinds its arrival counters after every launch, so one buffer per device
    serves every call on that device (stream-ordered launches).
    """

    index = device.index if device.index is not None else torch.cuda.current_device()
    workspace = _WORKSPACES.get(index)
    if workspace is None:
        with torch.cuda.device(index):
            nbytes = int(_get_module().minimax_h3_dense_attention_workspace_size())
            workspace = torch.zeros(
                nbytes, dtype=torch.uint8, device=torch.device("cuda", index)
            )
        _WORKSPACES[index] = workspace
    return workspace


def _check_rows(name: str, rows: torch.Tensor, tokens: int) -> None:
    if rows.dtype != torch.bfloat16:
        raise ValueError(f"{name} must be bfloat16, got {rows.dtype}")
    if rows.ndim != 2 or tuple(rows.shape) != (tokens, MINIMAX_H3_WIDTH):
        raise ValueError(
            f"{name} must have shape [{tokens}, {MINIMAX_H3_WIDTH}], got {tuple(rows.shape)}"
        )
    if not rows.is_cuda or not rows.is_contiguous():
        raise ValueError(f"{name} must be a contiguous CUDA tensor")


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
    _get_module().minimax_h3_dense_attention(q, k, v, out, workspace)


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
    r"""Dense non-causal BF16 self-attention for the MiniMax-H3 video DiT on SM120 (GB202).

    Computes, per head ``h`` of 56 heads with head_dim 128 and batch 1::

        qs = bf16(q * bf16(1 / sqrt(128)))        # product rounded to BF16 before the softmax
        y  = softmax(qs @ k^T) @ v                # softmax scale 1.0, no mask, no dropout

    ``q``, ``k``, ``v`` and ``y`` are contiguous BF16 ``[tokens, 7168]`` rows whose element
    ``[t, h * 128 + d]`` holds channel ``d`` of head ``h``; the row<->head re-layout, the
    query scale and the attention run in one kernel.  One runtime-variable kernel serves any
    ``1 <= tokens <= 131072``: a persistent 256-thread CTA per SM walks the (head, 128-row
    query tile) work items with a two-stage K/V TMA ring and a ping-pong schedule between
    its two warp groups; the items of the last wave are split across the otherwise idle CTAs
    by K/V range and merged in-kernel through a small per-device workspace.

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
        _check_rows(name, rows, tokens)
    if out is None:
        out = torch.empty_like(q)
    else:
        _check_rows("out", out, tokens)
    _minimax_h3_dense_attention_impl(q, k, v, out, _workspace(q.device))
    return out


__all__ = [
    "MINIMAX_H3_HEAD_DIM",
    "MINIMAX_H3_MAX_TOKENS",
    "MINIMAX_H3_NUM_HEADS",
    "MINIMAX_H3_QUERY_SCALE_BF16",
    "MINIMAX_H3_WIDTH",
    "minimax_h3_dense_attention",
]
