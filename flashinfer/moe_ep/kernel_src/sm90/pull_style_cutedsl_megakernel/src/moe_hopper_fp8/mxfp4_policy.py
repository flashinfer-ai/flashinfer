# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Pure-host eligibility and identities for fused MXFP4 optimizations.

Legality is based on kernel geometry and protocol, never a token-count lookup.
These helpers do not import CUDA, allocate workspaces, or select another dtype.
"""

from __future__ import annotations

from dataclasses import dataclass

from .fused_comm_policy import resolve_fused_comm_optimizations


# Register-A fence ordering is part of the compiled implementation.
MXFP4_OPTIMIZATION_VERSION = "fused_local_v3"
MXFP4_READY_MODES = ("tile", "k256")


@dataclass(frozen=True)
class Mxfp4Optimizations:
    peer32: bool = False
    offset_bulk: bool = False
    skip_zero_counts: bool = False
    fc2_tail_n8: bool = False
    fc1_ready_mode: str = "tile"
    fc1_ready_segments: int = 1
    fc1_ready_bits: int = 0

    def identity(self) -> tuple:
        """Bind every effective choice, including workspace protocol, to JIT."""
        return (
            MXFP4_OPTIMIZATION_VERSION,
            self.peer32,
            self.offset_bulk,
            self.skip_zero_counts,
            self.fc2_tail_n8,
            self.fc1_ready_mode,
            self.fc1_ready_segments,
            self.fc1_ready_bits,
        )


def validate_mxfp4_optional_optimizations(
    *, fc2_tail_n8: bool = False, fc1_ready_mode: str = "tile"
) -> None:
    if type(fc2_tail_n8) is not bool:
        raise ValueError("fc2_tail_n8 must be a bool")
    if not isinstance(fc1_ready_mode, str) or fc1_ready_mode not in MXFP4_READY_MODES:
        raise ValueError("fc1_ready_mode must be 'tile' or 'k256'")


def resolve_mxfp4_optimizations(
    *,
    fp8_scale_mode: str,
    mma_tiler_mnk: tuple[int, int, int],
    cluster_shape_mnk: tuple[int, int, int],
    static_expert_shape: tuple[int, int, int] | None,
    world_size: int,
    pingpong: bool = False,
    token_back_by_dispatch: bool = False,
    fc2_in_kernel_topk_reduce: bool = False,
    fc1_early_done_publish: bool = False,
    fc1_store_offload: bool = False,
    dedup_dispatch: bool = False,
    fc2_tail_n8: bool = False,
    fc1_ready_mode: str = "tile",
    local_optimizations: bool = True,
    skip_zero_counts: bool = True,
) -> Mxfp4Optimizations:
    """Resolve a supported implementation without rejecting legacy geometry.

    Explicit unsupported optional strategies fail before allocation; callers
    retaining their default settings keep a legal original implementation.
    ``local_optimizations`` and ``skip_zero_counts`` are host diagnostic inputs,
    not independent dimensions of the online candidate product. Zero-count
    skipping is derived for fused MXFP4 without dispatch deduplication.
    """
    validate_mxfp4_optional_optimizations(
        fc2_tail_n8=fc2_tail_n8, fc1_ready_mode=fc1_ready_mode
    )
    for name, value in (
        ("local_optimizations", local_optimizations),
        ("skip_zero_counts", skip_zero_counts),
    ):
        if type(value) is not bool:
            raise ValueError(f"{name} must be a bool")

    fused = fp8_scale_mode == "mxfp4_hybrid"
    tile = tuple(mma_tiler_mnk)
    cluster = tuple(cluster_shape_mnk)
    shape = tuple(static_expert_shape) if static_expert_shape is not None else None
    # The optional tail and readiness protocols retain their measured domain.
    peer_layout = (
        fused
        and not pingpong
        and tile == (256, 64, 256)
        and shape is not None
        and shape[2] == 7168
        and not token_back_by_dispatch
        and not fc2_in_kernel_topk_reduce
        and not dedup_dispatch
    )
    shared = resolve_fused_comm_optimizations(
        mma_tiler_mnk=tile,
        cluster_shape_mnk=cluster,
        static_expert_shape=shape,
        swap_ab=True,
        fused_token_comm=fused,
        bf16_output=True,
        token_back_by_dispatch=token_back_by_dispatch,
        fc2_in_kernel_topk_reduce=fc2_in_kernel_topk_reduce,
        dedup_dispatch=dedup_dispatch,
        paired_stores=local_optimizations and tile[2] == 256,
        skip_zero_counts=skip_zero_counts,
    )
    tail_supported = peer_layout
    if fc2_tail_n8 and not tail_supported:
        raise ValueError("fc2_tail_n8 is unsupported by this MXFP4 geometry")

    # This is a protocol capability, not an unconditional constructor assert.
    # Do not widen the bitmap/scheduler domain solely from divisibility checks.
    ready_supported = (
        peer_layout
        and cluster == (2, 1, 1)
        and shape == (96, 6144, 7168)
        and world_size == 4
        and fc1_early_done_publish
        and not fc1_store_offload
    )
    if fc1_ready_mode == "k256" and not ready_supported:
        raise ValueError("fc1_ready_mode='k256' is unsupported by this MXFP4 protocol")

    segments = 1
    bits = 0
    if fc1_ready_mode == "k256":
        intermediate = shape[1] // 2
        segments = intermediate // 256
        bits = intermediate // 64
        assert 0 < bits < 64 and bits == segments * 4

    return Mxfp4Optimizations(
        peer32=shared.paired_bf16_stores,
        offset_bulk=(local_optimizations and fused and not pingpong and tile[2] == 256),
        skip_zero_counts=shared.skip_zero_counts,
        fc2_tail_n8=fc2_tail_n8,
        fc1_ready_mode=fc1_ready_mode,
        fc1_ready_segments=segments,
        fc1_ready_bits=bits,
    )
