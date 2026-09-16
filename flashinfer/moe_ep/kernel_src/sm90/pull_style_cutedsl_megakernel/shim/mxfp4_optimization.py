# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded fused MXFP4 strategy identities, separate from frozen measurements.

This dependency-leaf host module neither compiles kernels nor touches caches.
The legacy geometry tables remain provenance, not new performance results.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
from typing import Any

from ..src.moe_hopper_fp8.mxfp4_policy import (
    MXFP4_OPTIMIZATION_VERSION,
    Mxfp4Optimizations,
    resolve_mxfp4_optimizations,
    validate_mxfp4_optional_optimizations,
)
from .mxfp4_tuner import (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    hopper_mxfp4_ordered_candidates,
    validate_hopper_mxfp4_tactic,
)


MXFP4_STRATEGY_FIELDS = frozenset({"fc2_tail_n8", "fc1_ready_mode"})


def normalize_mxfp4_optimization_tactic(
    tactic: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a complete legacy tactic or validate a complete strategy tactic.

    A legacy 17-field tactic explicitly means tail off / whole-tile ready.
    Partial strategy identities are rejected. Returning both fields even for
    defaults makes replacing an enabled tactic with an old tactic reset the
    protocol, rather than retaining the preceding workspace's settings.
    """
    if not isinstance(tactic, Mapping):
        raise TypeError("MXFP4 tactic must be a mapping")
    present = MXFP4_STRATEGY_FIELDS.intersection(tactic)
    if present and present != MXFP4_STRATEGY_FIELDS:
        raise ValueError(
            "MXFP4 strategy tactic requires both fc2_tail_n8 and fc1_ready_mode"
        )
    core = validate_hopper_mxfp4_tactic(
        {
            key: value
            for key, value in tactic.items()
            if key not in MXFP4_STRATEGY_FIELDS
        },
        execution_mode="fused",
    )
    tail = tactic.get("fc2_tail_n8", False)
    ready = tactic.get("fc1_ready_mode", "tile")
    validate_mxfp4_optional_optimizations(fc2_tail_n8=tail, fc1_ready_mode=ready)
    return dict(core, fc2_tail_n8=tail, fc1_ready_mode=ready)


def resolve_mxfp4_tactic_optimizations(
    tactic: Mapping[str, Any],
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    local_optimizations: bool = True,
    skip_zero_counts: bool = True,
) -> Mxfp4Optimizations:
    """Check model/protocol eligibility before allocating a fused workspace."""
    for name, value in (
        ("hidden", hidden),
        ("intermediate", intermediate),
        ("num_experts", num_experts),
        ("world_size", world_size),
    ):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if num_experts % world_size:
        raise ValueError("num_experts must be divisible by world_size")
    selected = normalize_mxfp4_optimization_tactic(tactic)
    tile = selected["mma_tiler_mnk"]
    if hidden % tile[2] or intermediate % tile[2]:
        raise ValueError("MXFP4 tile K must divide hidden and intermediate")
    # Match the existing fused constructor's effective normalization: folded
    # producers force direct FC1 stores and early publication. This affects
    # protocol eligibility, not the requested tactic recorded by the caller.
    folded = selected["fold_producer_warps"] and selected["active_dispatch_warps"] == 1
    return resolve_mxfp4_optimizations(
        fp8_scale_mode="mxfp4_hybrid",
        mma_tiler_mnk=tile,
        cluster_shape_mnk=selected["cluster_shape_mnk"],
        static_expert_shape=(num_experts // world_size, intermediate * 2, hidden),
        world_size=world_size,
        pingpong=selected["pingpong"],
        token_back_by_dispatch=selected["token_back_mode"] != "epi_warps",
        fc2_in_kernel_topk_reduce=selected["in_kernel_fc2_reduce"],
        fc1_early_done_publish=folded or selected["fc1_early_done_publish"],
        fc1_store_offload=False if folded else selected["fc1_store_offload"],
        dedup_dispatch=selected["dedup_dispatch"],
        fc2_tail_n8=selected["fc2_tail_n8"],
        fc1_ready_mode=selected["fc1_ready_mode"],
        local_optimizations=local_optimizations,
        skip_zero_counts=skip_zero_counts,
    )


def expand_mxfp4_optimization_candidates(
    tactics: Iterable[Mapping[str, Any]],
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
) -> list[dict[str, Any]]:
    """Keep base candidates first, then add at most three eligible variants.

    No new tile, layout or total-token rule is introduced. Ineligible optional
    variants are omitted, while invalid base candidates remain explicit errors.
    This is a bounded strategy union for live measurement, not a winner table.
    """
    base = []
    for tactic in tactics:
        normalized = normalize_mxfp4_optimization_tactic(tactic)
        resolve_mxfp4_tactic_optimizations(
            normalized,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
        )
        if normalized not in base:
            base.append(normalized)
    result = list(base)
    for tactic in base:
        for tail, ready in ((True, "tile"), (False, "k256"), (True, "k256")):
            candidate = dict(tactic, fc2_tail_n8=tail, fc1_ready_mode=ready)
            try:
                resolve_mxfp4_tactic_optimizations(
                    candidate,
                    hidden=hidden,
                    intermediate=intermediate,
                    num_experts=num_experts,
                    world_size=world_size,
                )
            except ValueError:
                continue
            if candidate not in result:
                result.append(candidate)
    return result


def hopper_mxfp4_optimization_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """The single ordered strategy domain for fused offline and online tuning.

    Token capacity only orders the existing geometry candidates. It never
    changes strategy eligibility or selects an alternate source tree.
    """
    base = hopper_mxfp4_ordered_candidates(
        max_tokens,
        execution_mode="fused",
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=routing_profile,
    )
    return expand_mxfp4_optimization_candidates(
        base,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
    )


def mxfp4_optimization_candidate_sha256(tactics: Iterable[Mapping[str, Any]]) -> str:
    """Bind both the implementation version and exact complete candidate set."""
    serialized = {
        json.dumps(
            normalize_mxfp4_optimization_tactic(tactic),
            sort_keys=True,
            separators=(",", ":"),
        )
        for tactic in tactics
    }
    payload = {
        "implementation": MXFP4_OPTIMIZATION_VERSION,
        "candidates": sorted(serialized),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
