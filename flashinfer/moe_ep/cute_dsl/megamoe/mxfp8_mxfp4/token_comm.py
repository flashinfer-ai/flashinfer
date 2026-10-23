# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Binding policy for the shared MegaMoE token communication helper.

Only construction-time constants live here.  Device tensor arguments are
assembled by the integrated kernel from :mod:`workspace` regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from .workspace import WorkspaceConfig


@dataclass(frozen=True)
class TokenCommBinding:
    """Format-specific constants passed to ``TokenInPullTokenBackPush``.

    ``transports_compat_topk_weights`` records a temporary ABI constraint of
    the shared helper.  It does *not* authorize FC1 to apply routing weights;
    the home-rank FP32 scores are authoritative and are applied after FC2.
    """

    world_size: int
    num_topk: int
    num_experts_per_rank: int
    num_total_experts: int
    hidden: int
    sf_uint32_per_token: int
    token_padding_block: int
    sf_padding_block: int
    cluster_tile_tokens: int
    cluster_shape_mn: tuple[int, int]
    dispatch_warp_start: int
    num_other_warps: int
    flag_batch: int
    transports_compat_topk_weights: bool = True
    apply_routing_weight_in_fc1: bool = False
    is_swap_ab: bool = True
    sf_atom_swizzled: bool = True

    def helper_kwargs(
        self, *, fc1_token_dtype: Any, combine_format: Any
    ) -> dict[str, Any]:
        """Return kwargs accepted by the vendored helper constructor."""

        return {
            "world_size": self.world_size,
            "num_topk": self.num_topk,
            "num_experts_per_rank": self.num_experts_per_rank,
            "num_total_experts": self.num_total_experts,
            "hidden": self.hidden,
            "fc1_token_dtype": fc1_token_dtype,
            "combine_format": combine_format,
            # The FC2 epilogue writes route terms home; dispatch warps never
            # push tokens back. Scores are not folded into FC1 or dispatch.
            "token_back_by_dispatch": False,
            "fc2_publishes_per_token_cluster_tile": 0,
            "token_back_reduce_topk": False,
            "token_back_standalone": False,
            "sf_uint32_per_token": self.sf_uint32_per_token,
            "token_padding_block": self.token_padding_block,
            "sf_padding_block": self.sf_padding_block,
            "cluster_tile_tokens": self.cluster_tile_tokens,
            "cluster_shape_mn": self.cluster_shape_mn,
            "dispatch_warp_start": self.dispatch_warp_start,
            "num_other_warps": self.num_other_warps,
            "is_swap_ab": self.is_swap_ab,
            "sf_atom_swizzled": self.sf_atom_swizzled,
            "flag_batch": self.flag_batch,
            "token_back_schedule_mode": "static",
        }

    def instantiate(
        self,
        *,
        helper_type: Callable[..., Any],
        fc1_token_dtype: Any,
        combine_format: Any,
    ) -> Any:
        """Instantiate the existing helper without importing CuTe on the host."""

        return helper_type(
            **self.helper_kwargs(
                fc1_token_dtype=fc1_token_dtype, combine_format=combine_format
            )
        )


def make_token_comm_binding(
    config: WorkspaceConfig,
    *,
    cluster_shape_mn: tuple[int, int],
    dispatch_warp_start: int,
    num_other_warps: int,
    flag_batch: int = 1,
) -> TokenCommBinding:
    if len(cluster_shape_mn) != 2 or any(v <= 0 for v in cluster_shape_mn):
        raise ValueError("cluster_shape_mn must contain two positive dimensions")
    if dispatch_warp_start < 0 or num_other_warps < 0:
        raise ValueError("warp indices/counts must be non-negative")
    if not 1 <= flag_batch <= 32:
        raise ValueError("flag_batch must be in [1, 32]")
    return TokenCommBinding(
        world_size=config.world_size,
        num_topk=config.num_topk,
        num_experts_per_rank=config.num_experts_per_rank,
        num_total_experts=config.num_total_experts,
        hidden=config.hidden,
        sf_uint32_per_token=config.sf_uint32_per_token,
        token_padding_block=config.token_padding_block,
        sf_padding_block=config.sf_padding_block,
        cluster_tile_tokens=config.cluster_tile_tokens,
        cluster_shape_mn=cluster_shape_mn,
        dispatch_warp_start=dispatch_warp_start,
        num_other_warps=num_other_warps,
        flag_batch=flag_batch,
    )
