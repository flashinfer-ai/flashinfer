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

from __future__ import annotations

from typing import Any, Optional

import torch

from ..api_logging import flashinfer_experimental_api

# Thin experimental entry points; the backend (dispatch table, launch
# geometry, JIT registration) lives in flashinfer.experimental.kimi_k3_fused_router.


def allocate_kimi_k3_route_plan(num_tokens: int, block_m: int, device: torch.device):
    """Allocate a reusable route plan for :func:`kimi_k3_fused_router` (no launch)."""
    from ..experimental.kimi_k3_fused_router.cake_backend import (
        allocate_kimi_k3_route_plan as allocate,
    )

    return allocate(num_tokens, block_m, device)


@flashinfer_experimental_api(feature="Kimi-K3 fused router")
def prepare_kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
    backend: str = "cake",
):
    r"""Prepare the Kimi-K3 fused MoE router (SM100 / SM103) for repeated launches.

    The experimental generated-program backend routes FP32 gate logits for 896
    experts (top-16 on ``sigmoid(logits) + bias``, weights renormalized over
    the selected sigmoid scores) and writes the ``block_m``-aligned route plan
    (``moe_align_block_size`` layout) in one launch.  Exactly the routed
    shapes ``num_tokens in {1, 2, 4, ..., 8192}`` with ``block_m in {8, 16}``
    are served; see ``flashinfer/experimental/kimi_k3_fused_router/README.md``.

    Parameters
    ----------
    logits : torch.Tensor
        Contiguous float32 ``[num_tokens, 896]`` gate logits.
    bias : torch.Tensor
        Contiguous float32 ``[896]`` selection bias (selection only; it does
        not enter the weights).
    block_m : int
        Route block alignment, 8 or 16.
    plan : Optional[KimiK3RoutePlan]
        Caller-owned outputs from
        :func:`flashinfer.fused_moe.allocate_kimi_k3_route_plan`; allocated
        with worst-case capacity when omitted.
    backend : str
        Only ``"cake"`` is supported.

    Returns
    -------
    KimiK3FusedRouterRunner
        Calling it launches the router on the current stream with no CUDA
        allocation or host synchronization and returns the bound plan.  CUDA
        Graph capture of the runner is supported; prepare outside capture.
    """
    if backend != "cake":
        raise ValueError("the Kimi-K3 fused router currently supports backend='cake'")
    from ..experimental.kimi_k3_fused_router.cake_backend import (
        prepare_kimi_k3_fused_router as prepare,
    )

    return prepare(logits, bias, block_m=block_m, plan=plan)


@flashinfer_experimental_api(feature="Kimi-K3 fused router")
def kimi_k3_fused_router(
    logits: torch.Tensor,
    bias: torch.Tensor,
    *,
    block_m: int = 8,
    plan: Optional[Any] = None,
    backend: str = "cake",
):
    r"""Route Kimi-K3 gate logits and build the aligned plan in one launch.

    Equivalent to :func:`prepare_kimi_k3_fused_router` followed by one launch;
    returns the :class:`KimiK3RoutePlan` (``topk_weights``, ``topk_ids``,
    ``sorted_token_ids``, ``expert_ids``, ``num_tokens_post_padded``,
    ``expert_counts``, ``expert_offsets``, ``expert_scatter_offsets``).
    """
    if backend != "cake":
        raise ValueError("the Kimi-K3 fused router currently supports backend='cake'")
    from ..experimental.kimi_k3_fused_router.cake_backend import (
        prepare_kimi_k3_fused_router as prepare,
    )

    return prepare(logits, bias, block_m=block_m, plan=plan)()


__all__ = [
    "allocate_kimi_k3_route_plan",
    "kimi_k3_fused_router",
    "prepare_kimi_k3_fused_router",
]
