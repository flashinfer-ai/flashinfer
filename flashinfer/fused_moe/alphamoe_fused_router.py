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

import functools
from typing import NamedTuple, Optional

import torch

from ..api_logging import flashinfer_api
from ..jit import gen_alphamoe_fused_router_module
from ..trace.templates.moe import alphamoe_fused_router_trace
from ..utils import (
    backend_requirement,
    register_custom_op,
    register_fake_op,
    supported_compute_capability,
)

_MAX_EXPERTS = 512
_MAX_TOP_K = 16
_MAX_BLOCK_M = 16
_INT32_MAX = 2**31 - 1


class AlphaMoERoutePlan(NamedTuple):
    """Device-resident routing outputs and reusable private workspace.

    The first five fields are the route-plan ABI consumed by AlphaMoE compute
    kernels. ``expert_counts``, ``expert_offsets``, and
    ``expert_scatter_offsets`` are implementation-owned workspace retained in
    the tuple so repeated calls can avoid allocation. Consumers must use only
    the prefix of ``sorted_token_ids`` named by
    ``num_tokens_post_padded[0]`` and the corresponding prefix of
    ``expert_ids``; reading that device scalar on the host is not required to
    launch a compatible compute kernel.
    """

    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor
    expert_counts: torch.Tensor
    expert_offsets: torch.Tensor
    expert_scatter_offsets: torch.Tensor

    @property
    def max_route_blocks(self) -> int:
        """Allocated number of route blocks."""

        return self.expert_ids.numel()

    @property
    def max_padded_pairs(self) -> int:
        """Allocated number of padded token/route entries."""

        return self.sorted_token_ids.numel()


def _require_plain_int(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _validate_problem(
    logits: torch.Tensor,
    top_k: int,
    block_m: int,
    has_shared_expert: bool,
) -> tuple[int, int, int, int]:
    if not isinstance(logits, torch.Tensor):
        raise TypeError(f"logits must be a torch.Tensor, got {type(logits).__name__}")
    if logits.device.type != "cuda":
        raise ValueError("logits must be a CUDA tensor")
    if logits.dtype != torch.float32:
        raise TypeError(f"logits must have dtype torch.float32, got {logits.dtype}")
    if logits.ndim != 2:
        raise ValueError(
            f"logits must have shape [num_tokens, num_experts], got {logits.shape}"
        )
    if not logits.is_contiguous():
        raise ValueError("logits must be contiguous")
    top_k = _require_plain_int("top_k", top_k)
    block_m = _require_plain_int("block_m", block_m)
    if not isinstance(has_shared_expert, bool):
        raise TypeError(
            f"has_shared_expert must be a bool, got {type(has_shared_expert).__name__}"
        )

    num_tokens, num_experts = (int(dim) for dim in logits.shape)
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive")
    if not 1 <= num_experts <= _MAX_EXPERTS:
        raise ValueError(
            f"num_experts must be in [1, {_MAX_EXPERTS}], got {num_experts}"
        )
    if not 1 <= top_k <= min(num_experts, _MAX_TOP_K):
        raise ValueError(
            f"top_k must be in [1, min(num_experts, {_MAX_TOP_K})], got {top_k}"
        )
    if not 1 <= block_m <= _MAX_BLOCK_M:
        raise ValueError(f"block_m must be in [1, {_MAX_BLOCK_M}], got {block_m}")
    if has_shared_expert and top_k < 2:
        raise ValueError("a forced shared expert requires top_k >= 2")
    if num_tokens > _INT32_MAX or num_tokens * top_k > _INT32_MAX:
        raise ValueError("num_tokens and num_tokens * top_k must fit in int32")
    return num_tokens, num_experts, top_k, block_m


def _max_route_blocks(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    block_m: int,
) -> int:
    """Safe closed-form bound for aligned route blocks."""

    pairs = num_tokens * top_k
    nonempty = min(num_experts, pairs)
    return nonempty + (pairs - nonempty) // block_m


def allocate_alphamoe_route_plan(
    logits: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    has_shared_expert: bool = False,
) -> AlphaMoERoutePlan:
    """Allocate a reusable :class:`AlphaMoERoutePlan` for ``logits``.

    This allocation helper does not launch the router. Pass the result back as
    ``plan=`` to :func:`alphamoe_fused_router`; that form is CUDA-graph-safe
    after the JIT module has been loaded once outside capture.
    """

    num_tokens, num_experts, top_k, block_m = _validate_problem(
        logits, top_k, block_m, has_shared_expert
    )
    max_route_blocks = _max_route_blocks(num_tokens, top_k, num_experts, block_m)
    max_padded_pairs = max_route_blocks * block_m
    if max_padded_pairs > _INT32_MAX:
        raise ValueError("maximum padded route count must fit in int32")
    device = logits.device
    return AlphaMoERoutePlan(
        topk_weights=torch.empty(
            (num_tokens, top_k), dtype=torch.float32, device=device
        ),
        topk_ids=torch.empty((num_tokens, top_k), dtype=torch.int32, device=device),
        sorted_token_ids=torch.empty(
            max_padded_pairs, dtype=torch.int32, device=device
        ),
        expert_ids=torch.empty(max_route_blocks, dtype=torch.int32, device=device),
        num_tokens_post_padded=torch.empty(1, dtype=torch.int32, device=device),
        expert_counts=torch.empty(num_experts, dtype=torch.int32, device=device),
        expert_offsets=torch.empty(num_experts + 1, dtype=torch.int32, device=device),
        expert_scatter_offsets=torch.empty(
            num_experts, dtype=torch.int32, device=device
        ),
    )


def _validate_plan(
    plan: AlphaMoERoutePlan,
    *,
    logits: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    top_k: int,
    block_m: int,
) -> None:
    if not isinstance(plan, AlphaMoERoutePlan):
        raise TypeError(f"plan must be an AlphaMoERoutePlan, got {type(plan).__name__}")

    max_route_blocks = _max_route_blocks(num_tokens, top_k, num_experts, block_m)
    expected = {
        "topk_weights": (torch.float32, 2, (num_tokens, top_k), None),
        "topk_ids": (torch.int32, 2, (num_tokens, top_k), None),
        "sorted_token_ids": (torch.int32, 1, None, max_route_blocks * block_m),
        "expert_ids": (torch.int32, 1, None, max_route_blocks),
        "num_tokens_post_padded": (torch.int32, 1, (1,), None),
        "expert_counts": (torch.int32, 1, (num_experts,), None),
        "expert_offsets": (torch.int32, 1, (num_experts + 1,), None),
        "expert_scatter_offsets": (torch.int32, 1, (num_experts,), None),
    }
    for name, (dtype, ndim, shape, min_numel) in expected.items():
        tensor = getattr(plan, name)
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"plan.{name} must be a torch.Tensor")
        if tensor.device != logits.device:
            raise ValueError(
                f"plan.{name} must be on {logits.device}, got {tensor.device}"
            )
        if tensor.dtype != dtype:
            raise TypeError(f"plan.{name} must have dtype {dtype}, got {tensor.dtype}")
        if tensor.ndim != ndim:
            raise ValueError(f"plan.{name} must be rank {ndim}, got rank {tensor.ndim}")
        if not tensor.is_contiguous():
            raise ValueError(f"plan.{name} must be contiguous")
        if shape is not None and tuple(tensor.shape) != shape:
            raise ValueError(f"plan.{name} must have shape {shape}, got {tensor.shape}")
        if min_numel is not None and tensor.numel() < min_numel:
            raise ValueError(
                f"plan.{name} needs capacity {min_numel}, got {tensor.numel()}"
            )


@functools.cache
def get_alphamoe_fused_router_module():
    """Build and cache the exact-SM100/SM103 router extension."""

    return gen_alphamoe_fused_router_module().build_and_load()


@register_custom_op(
    "flashinfer::alphamoe_fused_router",
    mutates_args=(
        "topk_weights",
        "topk_ids",
        "sorted_token_ids",
        "expert_ids",
        "num_tokens_post_padded",
        "expert_counts",
        "expert_offsets",
        "expert_scatter_offsets",
    ),
)
def _alphamoe_fused_router_impl(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_scatter_offsets: torch.Tensor,
    top_k: int,
    block_m: int,
    has_shared_expert: bool,
) -> None:
    get_alphamoe_fused_router_module().fused_router_op(
        logits,
        topk_weights,
        topk_ids,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        expert_counts,
        expert_offsets,
        expert_scatter_offsets,
        top_k,
        block_m,
        has_shared_expert,
    )


@register_fake_op("flashinfer::alphamoe_fused_router")
def _fake_alphamoe_fused_router(
    logits: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    expert_counts: torch.Tensor,
    expert_offsets: torch.Tensor,
    expert_scatter_offsets: torch.Tensor,
    top_k: int,
    block_m: int,
    has_shared_expert: bool,
) -> None:
    pass


@supported_compute_capability([100, 103])
def _check_alphamoe_fused_router_supported(
    logits: torch.Tensor,
    top_k: int,
    block_m: int = 8,
    has_shared_expert: bool = False,
    plan: Optional[AlphaMoERoutePlan] = None,
) -> bool:
    # Detailed shape, dtype, device, capacity, and alias validation is kept in
    # both the Python entrypoint and the C++ binding. This checker contributes
    # the exact architecture metadata used by backend_requirement.
    return True


@backend_requirement({}, common_check=_check_alphamoe_fused_router_supported)
@flashinfer_api(trace=alphamoe_fused_router_trace)
def alphamoe_fused_router(
    logits: torch.Tensor,
    *,
    top_k: int,
    block_m: int = 8,
    has_shared_expert: bool = False,
    plan: Optional[AlphaMoERoutePlan] = None,
) -> AlphaMoERoutePlan:
    r"""Route FP32 logits and build an AlphaMoE aligned plan (SM100/SM103).

    One cooperative generated kernel performs top-k selection, selected-logit
    softmax, per-expert counting, block padding, and grouped route scatter.
    With ``has_shared_expert=True``, the last expert is forced into the final
    route slot and the remaining ``top_k - 1`` experts are selected from
    ``[0, num_experts - 1)``.

    Parameters
    ----------
    logits : torch.Tensor
        Contiguous FP32 CUDA tensor shaped ``(num_tokens, num_experts)``.
        ``num_tokens`` must be positive and ``num_experts`` must be in
        ``[1, 512]``. Logits are expected to be finite.
    top_k : int
        Routes per token, in ``[1, min(num_experts, 16)]``.
    block_m : int
        Per-expert alignment in ``[1, 16]``. Defaults to 8.
    has_shared_expert : bool
        Force the last expert into every token's last route slot. This mode
        requires ``top_k >= 2``.
    plan : Optional[AlphaMoERoutePlan]
        Reusable output/workspace allocation. When omitted, a
        worst-case-capacity plan is allocated. Supply a plan for steady-state
        or CUDA-graph use and warm up the JIT module once before capture.

    Returns
    -------
    AlphaMoERoutePlan
        ``topk_weights`` and ``topk_ids`` have shape
        ``(num_tokens, top_k)``. ``sorted_token_ids`` stores flattened
        ``token * top_k + route`` indices grouped by expert and padded with
        sentinel ``num_tokens * top_k``. ``expert_ids`` stores one expert per
        ``block_m`` entries, and ``num_tokens_post_padded`` is a device-side
        int32 scalar naming the valid extent. Atomic scatter order within an
        expert is deliberately unspecified.

    Notes
    -----
    When routed logits are exactly equal, the selected expert set and its order
    are unspecified; no lower-expert-ID tie break is guaranteed.
    The frozen CUDA device source was generated from Cake/Loom commit
    ``e2aa03274`` and compiled with ``--use_fast_math``; its SHA256 is
    ``ec5bc689e68264a11a56a17fb10f699bc3733a521dea916b71ecda51d4227801``.
    Source-validation head ``def2a9dcb`` retains that exact device source while
    strengthening route-plan coverage checks.
    The operation has no dependency on an AlphaMoE compute kernel and its plan
    can feed either the W8A8 or NVFP4 fused up/down path.
    """

    num_tokens, num_experts, top_k, block_m = _validate_problem(
        logits, top_k, block_m, has_shared_expert
    )
    if plan is None:
        plan = allocate_alphamoe_route_plan(
            logits,
            top_k=top_k,
            block_m=block_m,
            has_shared_expert=has_shared_expert,
        )
    else:
        _validate_plan(
            plan,
            logits=logits,
            num_tokens=num_tokens,
            num_experts=num_experts,
            top_k=top_k,
            block_m=block_m,
        )

    _alphamoe_fused_router_impl(
        logits,
        plan.topk_weights,
        plan.topk_ids,
        plan.sorted_token_ids,
        plan.expert_ids,
        plan.num_tokens_post_padded,
        plan.expert_counts,
        plan.expert_offsets,
        plan.expert_scatter_offsets,
        top_k,
        block_m,
        has_shared_expert,
    )
    return plan


__all__ = [
    "AlphaMoERoutePlan",
    "allocate_alphamoe_route_plan",
    "alphamoe_fused_router",
    "get_alphamoe_fused_router_module",
]
