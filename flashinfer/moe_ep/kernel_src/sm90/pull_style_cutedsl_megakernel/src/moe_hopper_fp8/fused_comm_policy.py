# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Layout and protocol capabilities shared by fused SM90 MegaMoE kernels."""

from __future__ import annotations

from dataclasses import dataclass


FUSED_COMM_OPTIMIZATION_VERSION = "fused_comm_v1"


@dataclass(frozen=True)
class FusedCommOptimizations:
    paired_bf16_stores: bool = False
    skip_zero_counts: bool = False


def resolve_fused_comm_optimizations(
    *,
    mma_tiler_mnk: tuple[int, int, int],
    cluster_shape_mnk: tuple[int, int, int],
    static_expert_shape: tuple[int, int, int] | None,
    swap_ab: bool,
    fused_token_comm: bool,
    bf16_output: bool,
    combine_format: str = "bf16",
    token_back_by_dispatch: bool = False,
    fc2_in_kernel_topk_reduce: bool = False,
    dedup_dispatch: bool = False,
    paired_stores: bool = True,
    skip_zero_counts: bool = True,
) -> FusedCommOptimizations:
    """Resolve layout capability without changing arithmetic or readiness.

    K and input precision do not change the M256/N64 BF16 consumer layout.
    Complete channel clusters make every selected store a four-byte aligned
    pair; token tails are guarded in the epilogue. Kernel geometry is validated
    by callers: M256 already excludes the current M128-only pingpong schedule.
    Other output/return layouts retain scalar stores. The two last arguments
    are independent diagnostic controls, not additional online tuning dimensions.
    """
    for name, value in (("paired_stores", paired_stores),
                        ("skip_zero_counts", skip_zero_counts)):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a bool")
    tile = tuple(mma_tiler_mnk)
    cluster = tuple(cluster_shape_mnk)
    shape = tuple(static_expert_shape) if static_expert_shape is not None else None
    # This first shared domain is swapAB only. Non-swapAB uses another consumer
    # coordinate mapping and has not been qualified for these defaults.
    shared_protocol = fused_token_comm and swap_ab and not dedup_dispatch
    paired_layout = (
        shared_protocol
        and bf16_output
        and combine_format == "bf16"
        and tile[:2] == (256, 64)
        and shape is not None
        and shape[2] > 0
        and cluster[0] > 0
        and shape[2] % (tile[0] * cluster[0]) == 0
        and not token_back_by_dispatch
        and not fc2_in_kernel_topk_reduce
    )
    return FusedCommOptimizations(
        paired_bf16_stores=paired_stores and paired_layout,
        # Every CTA still reaches the grid barrier. SM0 publishes completion
        # for every expert, including zero totals, before the system fence.
        skip_zero_counts=skip_zero_counts and shared_protocol,
    )
