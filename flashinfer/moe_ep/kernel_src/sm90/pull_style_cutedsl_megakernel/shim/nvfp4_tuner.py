# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Independent NVFP4 candidate identities and exact-capacity H200 defaults.

Candidates are measured by the collective tuner on the current device. The
portable default profiles contain configuration only, never imported scores.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from typing import Any

from .nvfp4_policy import split_nvfp4_conversion_tactic, validate_nvfp4_kperm
from .mxfp4_optimization import (
    hopper_mxfp4_candidates,
    normalize_mxfp4_optimization_tactic,
    resolve_mxfp4_tactic_optimizations,
)
from .nvfp4_candidate_data import (
    NVFP4_SF0_CANDIDATE_SEEDS,
    NVFP4_DEFAULT_DOMAIN,
    NVFP4_DEFAULT_TACTICS_BY_TOKENS,
)

# Stable cache-schema tag: single-promotion uses one FP8 weight operand.
# The nv23/nv24 suffix counts tactic fields; preserve it for cache compatibility.
NVFP4_TUNING_VERSION = "nvfp4_single_k64_sf0_nv23_candidates_v1"


def normalize_hopper_nvfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the MXFP4 core fields plus compact pull, LUT replicas and lane map."""
    core, conversion = split_nvfp4_conversion_tactic(tactic)
    return dict(normalize_mxfp4_optimization_tactic(core), **conversion)


def nvfp4_candidate_sha256(candidates: Iterable[Mapping[str, Any]]) -> str:
    values = [normalize_hopper_nvfp4_tactic(x) for x in candidates]
    return hashlib.sha256(
        json.dumps(
            {"version": NVFP4_TUNING_VERSION, "candidates": values},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def _sf0_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str,
    kperm: int,
) -> list[dict[str, Any]]:
    """Return the complete sf0 candidate set in stable order."""
    validate_nvfp4_kperm(kperm)
    seeds = list(
        hopper_mxfp4_candidates(
            max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
            routing_profile=routing_profile,
        )
    )
    for tactic in NVFP4_SF0_CANDIDATE_SEEDS:
        try:
            core, _ = split_nvfp4_conversion_tactic(tactic)
            resolve_mxfp4_tactic_optimizations(
                core,
                hidden=hidden,
                intermediate=intermediate,
                num_experts=num_experts,
                world_size=world_size,
            )
        except ValueError:
            continue
        seeds.append(tactic)
    result = []
    for seed in seeds:
        selected = normalize_hopper_nvfp4_tactic(seed)
        if selected not in result:
            result.append(selected)
    return result


def hopper_nvfp4_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = "block_permutation_v1",
    kperm: int = 64,
) -> list[dict[str, Any]]:
    """Return the shape-legal NVFP4 candidate union in stable order."""
    return _sf0_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        kperm=kperm,
    )


def filter_hopper_nvfp4_candidates(candidates, **context) -> list[dict[str, Any]]:
    """Normalize NVFP4 candidates; legality follows the source policy."""
    del context
    return [normalize_hopper_nvfp4_tactic(candidate) for candidate in candidates]


def hopper_nvfp4_default_profile(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    num_topk: int,
    gate_up_clamp: float | None,
    routing_profile: str,
    kperm: int,
    device_name: str | None = None,
    sm_count: int | None = None,
    overlap: bool | None = None,
) -> dict[str, Any] | None:
    """Match a portable capacity default, returning a fresh tactic/config pair.

    Unknown hardware or an unmatched model/capacity uses the caller's existing
    heuristic. The profile's overlap is a default: an explicit caller override
    must be applied after its tactic geometry, and is not replaced here.
    """
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    if any(
        type(value) is not int
        for value in (
            max_tokens,
            hidden,
            intermediate,
            num_experts,
            world_size,
            num_topk,
            sm_count,
        )
    ):
        return None
    context = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        num_topk=num_topk,
        gate_up_clamp=gate_up_clamp,
        routing_profile=routing_profile,
        kperm=kperm,
        device_name=device_name,
        sm_count=sm_count,
    )
    if any(value != NVFP4_DEFAULT_DOMAIN[name] for name, value in context.items()):
        return None
    profile: dict[str, Any] | None = NVFP4_DEFAULT_TACTICS_BY_TOKENS.get(max_tokens)
    if profile is None:
        return None
    return dict(
        requested_tactic=normalize_hopper_nvfp4_tactic(profile["requested_tactic"]),
        requested_config=dict(profile["requested_config"]),
    )


def nvfp4_curated_domain_sha256(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str,
    kperm: int,
) -> str:
    """Bind the exact capacity, candidate set and default policy."""
    candidates = hopper_nvfp4_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        kperm=kperm,
    )
    payload = dict(
        version=NVFP4_TUNING_VERSION,
        max_tokens=max_tokens,
        shape=(hidden, intermediate, num_experts, world_size),
        routing_profile=routing_profile,
        kperm=kperm,
        candidates=candidates,
        default_domain=NVFP4_DEFAULT_DOMAIN,
        default_profile=NVFP4_DEFAULT_TACTICS_BY_TOKENS.get(max_tokens),
    )
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hopper_nvfp4_cache_provenance_sha256(
    *,
    routing_profile: str,
    kperm: int,
    overlap: bool | None,
    max_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
) -> str:
    """Separate NV layout/overlap/default policy from other tuning namespaces."""
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    from .mxfp4_tuner import hopper_mxfp4_cache_provenance_sha256

    payload = dict(
        version=NVFP4_TUNING_VERSION,
        kperm=kperm,
        overlap=overlap,
        domain_sha256=nvfp4_curated_domain_sha256(
            max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
            routing_profile=routing_profile,
            kperm=kperm,
        ),
        geometry_provenance=hopper_mxfp4_cache_provenance_sha256(
            routing_profile=routing_profile
        ),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


nvfp4_cache_provenance = hopper_nvfp4_cache_provenance_sha256
