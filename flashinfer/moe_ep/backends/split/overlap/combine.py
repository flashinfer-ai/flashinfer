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
"""Dest-side overlap combine algorithms.

The tile-ready consumer ships GEMM2 rows into each dest rank's inbox. Combine
turns that inbox into ``[tokens, hidden]`` like NCCL combine. Swap the
callable on :class:`FusedMoeKernelConfig.overlap_combine_fn` (or pass it to
:meth:`SplitKernelBackend.collect_overlap_combine` via the kernel) to try
another algorithm without touching ship or GEMM2.

Inbox layout is ``[world, num_local_experts, tokens_per_rank, hidden]``.
Source rank ``S`` wrote dest ``D`` at ``inbox_D[S, local_expert, slot]``.
Global expert id ``e`` maps to ``src = e // nle``, ``local_expert = e % nle``.
Slot is the original token index on the home GPU. The consumer looks up that
home GPU and index from the dispatch payload; it must not infer them from the
dispatch-buffer column.
"""

from __future__ import annotations

from typing import Protocol

import torch

from .peer_inbox import CombineInboxWorkspace


class OverlapCombineFn(Protocol):
    """Dest-side combine after the tile-ready consumer has shipped.

    Must return ``[tokens, hidden]`` with ``hidden_states`` dtype.
    """

    def __call__(
        self,
        inbox: CombineInboxWorkspace,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> torch.Tensor: ...


def weighted_reduce_inbox(
    inbox: CombineInboxWorkspace,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Weighted sum of EXPERT_MAJOR inbox rows into ``[tokens, hidden]``.

    ``inbox`` is ``[world, num_local_experts, tokens_per_rank, hidden]``.
    ``topk_ids`` are global expert ids. Does not wait for peers; the caller
    must have already established visibility of shipped rows.
    """
    buf = inbox.inbox
    if buf.dim() != 4:
        raise ValueError(
            "inbox must be [world, num_local_experts, tokens_per_rank, hidden], "
            f"got {tuple(buf.shape)}"
        )
    world, nle, tokens_per_rank, hidden = buf.shape
    if topk_ids.dim() != 2:
        raise ValueError(
            f"topk_ids must be [tokens, topk], got {tuple(topk_ids.shape)}"
        )
    tokens = int(topk_ids.shape[0])
    if tokens > tokens_per_rank:
        raise ValueError(
            f"topk_ids tokens={tokens} exceeds inbox tokens_per_rank={tokens_per_rank}"
        )
    if int(hidden_states.shape[-1]) != hidden:
        raise ValueError(
            f"hidden_states hidden={int(hidden_states.shape[-1])} != inbox hidden={hidden}"
        )
    if tuple(topk_weights.shape) != tuple(topk_ids.shape):
        raise ValueError(
            f"topk_weights shape {tuple(topk_weights.shape)} != topk_ids {tuple(topk_ids.shape)}"
        )

    ids = topk_ids.to(dtype=torch.int64)
    weights = topk_weights.to(dtype=torch.float32)
    src = torch.div(ids, nle, rounding_mode="floor").clamp(0, world - 1)
    local_e = (ids % nle).clamp(0, nle - 1)
    slot = torch.arange(tokens, device=ids.device).unsqueeze(1).expand_as(ids)
    contrib = buf[src, local_e, slot].to(dtype=torch.float32) * weights.unsqueeze(-1)
    return contrib.sum(dim=1).to(dtype=hidden_states.dtype)


def basic_overlap_combine(
    inbox: CombineInboxWorkspace,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """Default overlap combine: quorum all-reduce, then weighted inbox reduce.

    The all-reduce is a device-ordered stand-in so dest sees every source's
    stores after the consumer's ``fence.sys``. Replace this callable for
    credit/arrival or a fused reduce kernel.
    """
    inbox.wait_peers()
    return weighted_reduce_inbox(inbox, hidden_states, topk_ids, topk_weights)
