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
from typing import NamedTuple, Optional

import torch

from flashinfer.api_logging import flashinfer_api
from flashinfer.jit.fused_moe import gen_trtllm_gen_routing_module
from flashinfer.tllm_enums import RoutingMethodType
from flashinfer.utils import (
    backend_requirement,
    device_support_pdl,
    register_custom_op,
    supported_compute_capability,
)

# The trtllm_gen_routing module is compiled for SM 10.x and 12.x only (same
# gate as the fused_moe_trtllm_sm100 module the routing kernels ship in).
_TRTLLM_GEN_ROUTING_SUPPORTED_CC = [100, 103, 120, 121]


def _max_num_ctas_in_batch_dim(
    num_tokens: int, top_k: int, num_experts: int, tile_tokens_dim: int
) -> int:
    """Python mirror of Routing::getMaxNumCtasInBatchDim (runner.h)."""
    num_remaining = num_tokens * top_k
    num_filled = min(num_experts, num_remaining)
    max_ctas = num_filled
    num_remaining -= num_filled
    if num_remaining > 0:
        max_ctas += num_remaining // tile_tokens_dim
    return max_ctas


def max_num_padded_tokens(
    num_tokens: int, top_k: int, num_experts: int, tile_tokens_dim: int
) -> int:
    """Python mirror of Routing::getMaxPermutedPaddedCount (runner.h).

    Upper bound on the padded/permuted token count produced by trtllm-gen
    routing for the given problem size; sizes the ``permuted_idx_to_token_idx``
    output.
    """
    return (
        _max_num_ctas_in_batch_dim(num_tokens, top_k, num_experts, tile_tokens_dim)
        * tile_tokens_dim
    )


class TrtllmGenRoutingResult(NamedTuple):
    """Outputs of the trtllm-gen MoE routing stage.

    ``topk_ids``/``topk_weights`` are the per-token expert selection. The
    remaining fields are the permutation/bookkeeping tensors the fused MoE
    kernels consume; entries beyond the actual padded count (see
    ``total_num_padded_tokens``) are undefined.

    In from-logits mode the routing kernels never emit expert ids directly —
    expert identity lives in the permuted layout. ``topk_ids`` is therefore
    reconstructed on the torch side: permuted slot ``p`` belongs to CTA tile
    ``p // tile_tokens_dim``, and ``cta_idx_xy_to_batch_idx`` maps each CTA
    tile to its expert.
    """

    topk_ids: torch.Tensor  # [num_tokens, top_k + nfse] int32, -1 for inactive slots
    topk_weights: torch.Tensor  # [num_tokens, top_k + nfse] bfloat16
    total_num_padded_tokens: torch.Tensor  # [1] int32
    expanded_idx_to_permuted_idx: torch.Tensor  # [num_tokens, top_k + nfse] int32
    permuted_idx_to_token_idx: torch.Tensor  # [max_num_padded_tokens] int32
    cta_idx_xy_to_batch_idx: torch.Tensor  # [max_num_ctas] int32
    cta_idx_xy_to_mn_limit: torch.Tensor  # [max_num_ctas] int32
    num_non_exiting_ctas: torch.Tensor  # [1] int32


@supported_compute_capability(_TRTLLM_GEN_ROUTING_SUPPORTED_CC)
def _check_trtllm_gen_routing_supported(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    routing_method: RoutingMethodType,
    top_k: int,
    *,
    num_fused_shared_experts: int = 0,
    n_group: int = 0,
    topk_group: int = 0,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
    tile_tokens_dim: int = 8,
    norm_topk_prob: bool = True,
    enable_pdl: Optional[bool] = None,
) -> bool:
    if routing_logits.dim() != 2:
        raise ValueError(
            f"routing_logits must be 2D [num_tokens, num_experts], "
            f"got {tuple(routing_logits.shape)}"
        )
    if routing_logits.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(
            f"routing_logits must be float32 or bfloat16, got {routing_logits.dtype}"
        )
    num_experts = routing_logits.shape[1]
    if not 1 <= top_k <= num_experts:
        raise ValueError(f"top_k must be in [1, num_experts], got {top_k}")
    if routing_bias is not None and routing_bias.numel() != num_experts:
        raise ValueError(
            f"routing_bias must have num_experts ({num_experts}) elements, "
            f"got {routing_bias.numel()}"
        )
    if tile_tokens_dim < 1 or tile_tokens_dim & (tile_tokens_dim - 1):
        raise ValueError(
            f"tile_tokens_dim must be a positive power of two, got {tile_tokens_dim}"
        )
    if routing_method == RoutingMethodType.Unspecified:
        raise ValueError("routing_method must be a concrete RoutingMethodType")
    return True


@functools.cache
def get_trtllm_gen_routing_module():
    """Build, load, and cache the trtllm_gen_routing JIT module as a custom op."""
    module = gen_trtllm_gen_routing_module().build_and_load()

    @register_custom_op(
        "flashinfer::trtllm_gen_routing",
        mutates_args=[
            "topk_packed",
            "topk_weights",
            "expert_count_histogram",
            "total_num_padded_tokens",
            "expanded_idx_to_permuted_idx",
            "permuted_idx_to_token_idx",
            "cta_idx_xy_to_batch_idx",
            "cta_idx_xy_to_mn_limit",
            "num_non_exiting_ctas",
        ],
    )
    def trtllm_gen_routing(
        routing_logits: torch.Tensor,
        routing_bias: Optional[torch.Tensor],
        topk_packed: torch.Tensor,
        topk_weights: torch.Tensor,
        expert_count_histogram: torch.Tensor,
        total_num_padded_tokens: torch.Tensor,
        expanded_idx_to_permuted_idx: torch.Tensor,
        permuted_idx_to_token_idx: torch.Tensor,
        cta_idx_xy_to_batch_idx: torch.Tensor,
        cta_idx_xy_to_mn_limit: torch.Tensor,
        num_non_exiting_ctas: torch.Tensor,
        top_k: int,
        num_fused_shared_experts: int,
        n_group: int,
        topk_group: int,
        local_expert_offset: int,
        local_num_experts: int,
        routed_scaling_factor: float,
        tile_tokens_dim: int,
        routing_method_type: int,
        norm_topk_prob: bool,
        enable_pdl: bool,
    ) -> None:
        module.trtllm_gen_routing(
            routing_logits,
            routing_bias,
            topk_packed,
            topk_weights,
            expert_count_histogram,
            total_num_padded_tokens,
            expanded_idx_to_permuted_idx,
            permuted_idx_to_token_idx,
            cta_idx_xy_to_batch_idx,
            cta_idx_xy_to_mn_limit,
            num_non_exiting_ctas,
            top_k,
            num_fused_shared_experts,
            n_group,
            topk_group,
            local_expert_offset,
            local_num_experts,
            routed_scaling_factor,
            tile_tokens_dim,
            routing_method_type,
            norm_topk_prob,
            enable_pdl,
        )

    return SimpleNamespace(trtllm_gen_routing=trtllm_gen_routing)


@backend_requirement({}, common_check=_check_trtllm_gen_routing_supported)
@flashinfer_api
def trtllm_gen_routing(
    routing_logits: torch.Tensor,
    routing_bias: Optional[torch.Tensor],
    routing_method: RoutingMethodType,
    top_k: int,
    *,
    num_fused_shared_experts: int = 0,
    n_group: int = 0,
    topk_group: int = 0,
    local_expert_offset: int = 0,
    local_num_experts: Optional[int] = None,
    routed_scaling_factor: float = 1.0,
    tile_tokens_dim: int = 8,
    norm_topk_prob: bool = True,
    enable_pdl: Optional[bool] = None,
) -> TrtllmGenRoutingResult:
    r"""Standalone trtllm-gen MoE routing (expert selection + permutation).

    Runs the same routing kernels the trtllm-gen fused MoE launchers execute
    before their GEMMs (``Routing::Runner::run``), and returns the routing
    outputs directly instead of feeding them into a GEMM. This makes the
    routing stage unit-testable in isolation from quantization/GEMM axes.

    Parameters
    ----------
    routing_logits : torch.Tensor
        Router logits of shape ``(num_tokens, num_experts)``, ``float32`` or
        ``bfloat16``.
    routing_bias : Optional[torch.Tensor]
        Per-expert routing bias of shape ``(num_experts,)``, ``float32`` or
        ``bfloat16`` (used by DeepSeekV3/MiniMax2-style methods).
    routing_method : RoutingMethodType
        The routing method to run (all methods except ``Unspecified``).
    top_k : int
        Number of experts selected per token.
    num_fused_shared_experts : int
        Extra fused shared-expert slots appended per token (DeepSeekV3 path).
    n_group, topk_group : int
        Expert-group parameters for grouped routing (DeepSeekV3). 0 disables
        grouping.
    local_expert_offset, local_num_experts : int
        Expert-parallel shard description. ``local_num_experts`` defaults to
        ``num_experts``.
    routed_scaling_factor : float
        Output weight scale (DeepSeekV3/MiniMax2-style methods).
    tile_tokens_dim : int
        Token-tile size the downstream grouped GEMM would use; must be a power
        of two. The permutation/padding outputs depend on it.
    norm_topk_prob : bool
        Whether SigmoidRenorm renormalizes the selected probabilities. Only
        consulted for ``RoutingMethodType.SigmoidRenorm``.
    enable_pdl : Optional[bool]
        Whether to launch with programmatic dependent launch. Defaults to
        auto-detection.

    Returns
    -------
    TrtllmGenRoutingResult
        Named tuple with expert selection (``topk_ids``, ``topk_weights``)
        and permutation/bookkeeping tensors
        (``total_num_padded_tokens``, ``expanded_idx_to_permuted_idx``,
        ``permuted_idx_to_token_idx``, ``cta_idx_xy_to_batch_idx``,
        ``cta_idx_xy_to_mn_limit``, ``num_non_exiting_ctas``).

    Notes
    -----
    ``topk_weights`` is always ``bfloat16``: the routing dispatcher hard-codes
    its output dtype regardless of the logits dtype.

    ``topk_ids`` is reconstructed from the permutation (the kernels emit no
    direct id output in from-logits mode): permuted slot ``p`` lies in CTA
    tile ``p // tile_tokens_dim``, whose expert is
    ``cta_idx_xy_to_batch_idx[p // tile_tokens_dim] + local_expert_offset``.
    """
    device = routing_logits.device
    num_tokens, num_experts = routing_logits.shape
    if local_num_experts is None:
        local_num_experts = num_experts
    total_experts_per_token = top_k + num_fused_shared_experts
    total_num_experts = num_experts + num_fused_shared_experts

    max_ctas = _max_num_ctas_in_batch_dim(
        num_tokens, total_experts_per_token, total_num_experts, tile_tokens_dim
    )
    max_padded = max_ctas * tile_tokens_dim
    histogram_size = max(2 * total_num_experts, 2 * 256)

    def empty_i32(*shape):
        return torch.empty(shape, dtype=torch.int32, device=device)

    topk_packed = empty_i32(num_tokens, total_experts_per_token)
    topk_weights = torch.empty(
        (num_tokens, total_experts_per_token), dtype=torch.bfloat16, device=device
    )
    expert_count_histogram = empty_i32(histogram_size)
    total_num_padded_tokens = empty_i32(1)
    expanded_idx_to_permuted_idx = empty_i32(num_tokens, total_experts_per_token)
    permuted_idx_to_token_idx = empty_i32(max_padded)
    cta_idx_xy_to_batch_idx = empty_i32(max_ctas)
    cta_idx_xy_to_mn_limit = empty_i32(max_ctas)
    num_non_exiting_ctas = empty_i32(1)

    if enable_pdl is None:
        enable_pdl = device_support_pdl(device)

    get_trtllm_gen_routing_module().trtllm_gen_routing(
        routing_logits,
        routing_bias,
        topk_packed,
        topk_weights,
        expert_count_histogram,
        total_num_padded_tokens,
        expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx,
        cta_idx_xy_to_batch_idx,
        cta_idx_xy_to_mn_limit,
        num_non_exiting_ctas,
        top_k,
        num_fused_shared_experts,
        n_group,
        topk_group,
        local_expert_offset,
        local_num_experts,
        float(routed_scaling_factor),
        tile_tokens_dim,
        int(routing_method),
        norm_topk_prob,
        enable_pdl,
    )

    # Reconstruct expert ids from the permuted layout: slot p -> CTA tile
    # p // tile_tokens_dim -> expert. cta_idx_xy_to_batch_idx holds the
    # *local* expert index of each CTA tile. Slots with e2p < 0 (experts
    # outside the local shard) get id -1.
    e2p = expanded_idx_to_permuted_idx.long()
    active = e2p >= 0
    experts_of_cta = cta_idx_xy_to_batch_idx.long() + local_expert_offset
    topk_ids = torch.where(
        active,
        experts_of_cta[e2p.clamp(min=0) // tile_tokens_dim],
        torch.full_like(e2p, -1),
    ).to(torch.int32)

    return TrtllmGenRoutingResult(
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        total_num_padded_tokens=total_num_padded_tokens,
        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
        permuted_idx_to_token_idx=permuted_idx_to_token_idx,
        cta_idx_xy_to_batch_idx=cta_idx_xy_to_batch_idx,
        cta_idx_xy_to_mn_limit=cta_idx_xy_to_mn_limit,
        num_non_exiting_ctas=num_non_exiting_ctas,
    )
