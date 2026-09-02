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
from types import SimpleNamespace
from typing import Optional, Tuple

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit import gen_alphamoe_router_module
from flashinfer.trace.templates.moe import alphamoe_fused_router_trace
from flashinfer.utils import (
    backend_requirement,
    register_custom_op,
    supported_compute_capability,
)

# alphamoe_fused_router is plain CUDA (warp reductions, shared-memory scans,
# Programmatic Dependent Launch on SM90+ with a regular-launch fallback on
# older architectures), portable across all tensor-core capable GPUs.
_ALPHAMOE_ROUTER_SUPPORTED_CC = [80, 86, 89, 90, 100, 103, 107, 110, 120, 121]

# Canonical output order of the routing bundle (matches ``AlphaMoeRoutePlan``
# iteration order and the ``alphamoe_fused_router`` return tuple).
_PLAN_FIELDS = (
    "topk_weights",
    "topk_ids",
    "sorted_token_ids",
    "expert_ids",
    "num_tokens_post_padded",
    "expert_counts",
    "expert_offsets",
    "expert_scatter_offsets",
)


def _alphamoe_router_geometry(
    num_tokens: int, num_experts: int, top_k: int, block_m: int
) -> Tuple[int, int]:
    """(max_blocks, slots); depends only on the shape configuration."""
    pairs = num_tokens * top_k
    nonempty = min(num_experts, pairs)
    max_blocks = nonempty + (pairs - nonempty) // block_m
    return max_blocks, max_blocks * block_m


class AlphaMoeRoutePlan:
    """Persistent, shape-stable routing-plan buffers for the AlphaMoE router.

    The plan owns every output tensor of the fused routing bundle (plus the
    internal scratch workspace). Allocating the buffers once before warmup
    and reusing them for every routing call keeps tensor addresses fixed, so
    a forward that routes through a plan can be captured into a CUDA graph
    and replayed with new logit *values* in the same (or the captured) input
    buffer; the replay then rewrites buffer contents only and no
    re-allocation, re-planning, or host synchronization is required.

    Iterating or unpacking a plan yields the eight public output tensors in
    the canonical order documented under :func:`alphamoe_fused_router`, and
    the same tensors are also available as named attributes.

    Attributes
    ----------
    num_tokens, num_experts, top_k, block_m : int
        Geometry configuration the plan was allocated for.
    has_shared_expert : bool
        Shared-expert configuration the plan was allocated for.
    """

    def __init__(
        self,
        num_tokens: int,
        num_experts: int,
        top_k: int,
        block_m: int,
        has_shared_expert: bool = False,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_tokens = int(num_tokens)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.block_m = int(block_m)
        self.has_shared_expert = bool(has_shared_expert)
        max_blocks, slots = _alphamoe_router_geometry(
            self.num_tokens, self.num_experts, self.top_k, self.block_m
        )
        scratch_elems = (
            self.num_tokens * ((self.num_experts + 31) // 32)
            if self.num_experts > 1024
            else 0
        )
        f32 = dict(dtype=torch.float32, device=device)
        i32 = dict(dtype=torch.int32, device=device)
        self.topk_weights = torch.empty((self.num_tokens, self.top_k), **f32)
        self.topk_ids = torch.empty((self.num_tokens, self.top_k), **i32)
        self.sorted_token_ids = torch.empty((slots,), **i32)
        self.expert_ids = torch.empty((max_blocks,), **i32)
        self.num_tokens_post_padded = torch.empty((1,), **i32)
        self.expert_counts = torch.empty((self.num_experts,), **i32)
        self.expert_offsets = torch.empty((self.num_experts + 1,), **i32)
        self.expert_scatter_offsets = torch.empty((self.num_experts,), **i32)
        # Internal workspace for the generic large-num_experts path; not part
        # of the public bundle.
        self.scratch = torch.empty((scratch_elems,), **i32)
        self._tensors = tuple(getattr(self, name) for name in _PLAN_FIELDS)

    # Tuple emulation over the eight public outputs, so a plan can be used
    # interchangeably with the return value of ``alphamoe_fused_router``.
    def __iter__(self):
        return iter(self._tensors)

    def __len__(self) -> int:
        return len(self._tensors)

    def __getitem__(self, index):
        return self._tensors[index]


@supported_compute_capability(_ALPHAMOE_ROUTER_SUPPORTED_CC)
def _check_alphamoe_router_vibecuda(
    router_logits: torch.Tensor,
    plan: Optional[AlphaMoeRoutePlan] = None,
    top_k: Optional[int] = None,
    block_m: Optional[int] = None,
    has_shared_expert: bool = False,
    backend: str = "vibecuda",
) -> bool:
    """Validate dtypes, shapes, and configuration for the fused AlphaMoE router.

    Returns ``True`` when all inputs are valid and raises ``ValueError``
    otherwise, so direct FFI callers and the Python API share one contract.
    """
    if router_logits.dim() != 2:
        raise ValueError(
            f"router_logits must be 2D [num_tokens, num_experts], got "
            f"{tuple(router_logits.shape)}"
        )
    if router_logits.dtype != torch.float32:
        raise ValueError(
            f"router_logits must be float32, got {router_logits.dtype}"
        )
    if not router_logits.is_cuda:
        raise ValueError("router_logits must be a CUDA tensor")
    if not router_logits.is_contiguous():
        raise ValueError("router_logits must be contiguous")

    num_tokens, num_experts = router_logits.shape
    if num_tokens < 1:
        raise ValueError(f"num_tokens must be >= 1, got {num_tokens}")
    if num_experts < 2:
        raise ValueError(f"num_experts must be >= 2, got {num_experts}")

    if plan is not None:
        if not isinstance(plan, AlphaMoeRoutePlan):
            raise ValueError(
                f"plan must be an AlphaMoeRoutePlan from "
                f"allocate_alphamoe_route_plan, got {type(plan).__name__}"
            )
        top_k = plan.top_k
        block_m = plan.block_m
        has_shared_expert = plan.has_shared_expert
        if (plan.num_tokens, plan.num_experts) != (num_tokens, num_experts):
            raise ValueError(
                f"plan geometry ({plan.num_tokens} tokens, "
                f"{plan.num_experts} experts) does not match router_logits "
                f"({num_tokens} tokens, {num_experts} experts)"
            )
    elif top_k is None or block_m is None:
        raise ValueError(
            "top_k and block_m are required when no route plan is provided"
        )
    if not 1 <= int(top_k):
        raise ValueError(f"top_k must be >= 1, got {top_k}")
    if not 1 <= int(block_m):
        raise ValueError(f"block_m must be >= 1, got {block_m}")
    has_shared = int(bool(has_shared_expert))
    routed_top_k = int(top_k) - has_shared
    routed_experts = num_experts - has_shared
    if routed_top_k < 0 or routed_top_k > routed_experts:
        raise ValueError(
            f"invalid top_k ({top_k})/num_experts ({num_experts}) for the "
            f"shared-expert configuration ({has_shared_expert})"
        )
    return True


@functools.cache
def get_alphamoe_router_module():
    """Build, load, and cache the AlphaMoE fused-router JIT module as a custom op."""
    module = gen_alphamoe_router_module().build_and_load()

    @register_custom_op(
        "flashinfer::alphamoe_fused_router",
        mutates_args=[
            "topk_weights",
            "topk_ids",
            "sorted_token_ids",
            "expert_ids",
            "num_tokens_post_padded",
            "expert_counts",
            "expert_offsets",
            "expert_scatter_offsets",
            "scratch",
        ],
    )
    def alphamoe_fused_router(
        router_logits: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_padded: torch.Tensor,
        expert_counts: torch.Tensor,
        expert_offsets: torch.Tensor,
        expert_scatter_offsets: torch.Tensor,
        scratch: torch.Tensor,
        top_k: int,
        block_m: int,
        has_shared_expert: bool,
    ) -> None:
        """Custom-op wrapper that writes the routing bundle into the outputs."""
        module.alphamoe_fused_router(
            router_logits,
            topk_weights,
            topk_ids,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            expert_counts,
            expert_offsets,
            expert_scatter_offsets,
            scratch,
            top_k,
            block_m,
            has_shared_expert,
        )

    return SimpleNamespace(alphamoe_fused_router=alphamoe_fused_router)


def allocate_alphamoe_route_plan(
    router_logits: torch.Tensor,
    top_k: int,
    block_m: int,
    has_shared_expert: bool = False,
) -> AlphaMoeRoutePlan:
    """Allocate the persistent routing-plan buffers for one routing geometry.

    All output shapes of the fused AlphaMoE router depend only on
    ``(num_tokens, num_experts, top_k, block_m)``, so the entire routing
    bundle can be pre-allocated from the logits *shape* before any routing
    call runs. Call this once outside the hot path (e.g. before CUDA graph
    warmup/capture) and pass the returned plan to
    :func:`alphamoe_fused_router`; every call then reuses the same buffer
    addresses and only rewrites their contents.

    Parameters
    ----------
    router_logits : torch.Tensor
        Per-token expert logits of shape ``(num_tokens, num_experts)``,
        ``float32``, on CUDA. Only the shape, dtype, and device are used;
        the values are never read here.
    top_k : int
        Number of selected experts per token, including the shared expert
        column when ``has_shared_expert`` is true.
    block_m : int
        Token-block alignment of the block-sparse MoE backend.
    has_shared_expert : bool
        Whether the last expert id is a shared expert that every token also
        routes to. Default ``False``.

    Returns
    -------
    AlphaMoeRoutePlan
        The persistent routing-plan buffers, iterable/unpackable as the
        eight-tuple documented under :func:`alphamoe_fused_router`.
    """
    if not isinstance(router_logits, torch.Tensor) or router_logits.dim() != 2:
        raise ValueError(
            "router_logits must be a 2D [num_tokens, num_experts] tensor"
        )
    if not router_logits.is_cuda:
        raise ValueError("router_logits must be a CUDA tensor")
    num_tokens, num_experts = router_logits.shape
    return AlphaMoeRoutePlan(
        num_tokens,
        num_experts,
        top_k,
        block_m,
        has_shared_expert,
        device=router_logits.device,
    )


@backend_requirement({"vibecuda": _check_alphamoe_router_vibecuda})
@flashinfer_api(trace=alphamoe_fused_router_trace)
def alphamoe_fused_router(
    router_logits: torch.Tensor,
    plan: Optional[AlphaMoeRoutePlan] = None,
    top_k: Optional[int] = None,
    block_m: Optional[int] = None,
    has_shared_expert: bool = False,
    backend: str = "vibecuda",
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    r"""Fused AlphaMoE gating router for a block-sparse MoE backend.

    One fused call consumes the per-token expert logits and emits the complete
    routing metadata bundle: stable top-k selection with an optional shared
    expert, softmax weights, the expert histogram, block-m-aligned padded
    expert offsets with the padded extent, expert-grouped sorted route ids,
    per-block expert ids, and per-expert scatter offsets.

    Selection for each token picks ``routed_top_k = top_k -
    int(has_shared_expert)`` experts from ``router_logits[:, :num_experts -
    int(has_shared_expert)]`` in descending logit order with a stable
    tie-break (equal logits keep the lower expert index first, i.e. the
    result of a stable descending sort over the row). When a shared
    expert is present, expert id ``num_experts - 1`` is appended as the last
    selected column with ``router_logits[:, -1]`` as its value.
    ``topk_weights`` is the max-subtracted fp32 softmax over the ``top_k``
    selected logits per token.

    Route ids in ``sorted_token_ids`` are flat ``token * top_k + slot``
    indices in increasing order within each expert segment; in-segment
    padding slots carry the sentinel ``num_tokens * top_k`` and slots outside
    every expert segment stay zero. ``expert_ids`` is zero outside real
    expert blocks.

    All output shapes depend only on
    ``(num_tokens, num_experts, top_k, block_m)``, every output is fully
    written on each call, and no host synchronization happens on the launch
    path, so the operator is safe under CUDA graph capture: replaying a
    captured graph with updated ``router_logits`` values in the captured
    buffer reproduces the eagerly computed routing bundle for those values.
    Pass a plan from :func:`allocate_alphamoe_route_plan` (allocated before
    warmup/capture) to keep every output address fixed across capture and
    replay.

    Parameters
    ----------
    router_logits : torch.Tensor
        Per-token expert logits of shape ``(num_tokens, num_experts)``,
        ``float32``, contiguous, on CUDA.
    plan : AlphaMoeRoutePlan, optional
        Persistent routing-plan buffers from
        :func:`allocate_alphamoe_route_plan`. When provided, the routing
        geometry is taken from the plan and results are written into the
        plan's buffers; ``top_k``/``block_m``/``has_shared_expert`` must be
        left at their defaults. When omitted, ``top_k`` and ``block_m`` are
        required and fresh output tensors are allocated per call.
    top_k : int, optional
        Number of selected experts per token, including the shared expert
        column when ``has_shared_expert`` is true. Required when no ``plan``
        is provided.
    block_m : int, optional
        Token-block alignment of the block-sparse MoE backend. Required when
        no ``plan`` is provided.
    has_shared_expert : bool
        Whether the last expert id is a shared expert that every token also
        routes to. Default ``False``.
    backend : str
        Backend selector. Only ``"vibecuda"`` (the custom CUDA
        implementation) is available; selecting it on an unsupported
        architecture raises instead of silently rerouting.

    Returns
    -------
    topk_weights : torch.Tensor
        Softmaxed routing weights, ``(num_tokens, top_k)`` ``float32``.
    topk_ids : torch.Tensor
        Selected expert ids, ``(num_tokens, top_k)`` ``int32``.
    sorted_token_ids : torch.Tensor
        Expert-grouped flat route ids, ``(max_blocks * block_m,)`` ``int32``.
    expert_ids : torch.Tensor
        Per-block expert id, ``(max_blocks,)`` ``int32``.
    num_tokens_post_padded : torch.Tensor
        Total padded token extent, ``(1,)`` ``int32``.
    expert_counts : torch.Tensor
        Selected routes per expert, ``(num_experts,)`` ``int32``.
    expert_offsets : torch.Tensor
        Exclusive prefix of the padded counts, ``(num_experts + 1,)``
        ``int32``.
    expert_scatter_offsets : torch.Tensor
        Per-expert scatter offsets, ``(num_experts,)`` ``int32``; identical
        to ``expert_counts`` (the upstream routing plan returns
        ``counts.clone()`` for this output).
    """
    if plan is None:
        if top_k is None or block_m is None:
            raise ValueError(
                "top_k and block_m are required when no plan is provided"
            )
        num_tokens, num_experts = router_logits.shape
        plan = AlphaMoeRoutePlan(
            num_tokens,
            num_experts,
            int(top_k),
            int(block_m),
            has_shared_expert,
            device=router_logits.device,
        )
    get_alphamoe_router_module().alphamoe_fused_router(
        router_logits,
        plan.topk_weights,
        plan.topk_ids,
        plan.sorted_token_ids,
        plan.expert_ids,
        plan.num_tokens_post_padded,
        plan.expert_counts,
        plan.expert_offsets,
        plan.expert_scatter_offsets,
        plan.scratch,
        plan.top_k,
        plan.block_m,
        plan.has_shared_expert,
    )
    return tuple(plan)
