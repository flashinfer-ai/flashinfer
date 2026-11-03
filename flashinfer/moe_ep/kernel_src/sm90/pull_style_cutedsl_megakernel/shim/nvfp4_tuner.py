# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""NVFP4 candidate validation, capacity defaults and tuning-cache identity."""

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
    _H200_REFERENCE,
    NVFP4_DEFAULT_TACTICS_BY_TOKENS,
    NVFP4_SF2_CANDIDATE_SEEDS,
    _scale_ring_tactic,
)

NVFP4_TUNING_VERSION = "nvfp4_single_k64_scale_ring_candidates_v2"


def normalize_hopper_nvfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate the complete MX core and NV conversion fields."""
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


def _validate_candidate(tactic, *, hidden, intermediate, num_experts, world_size):
    selected = normalize_hopper_nvfp4_tactic(tactic)
    core, _ = split_nvfp4_conversion_tactic(selected)
    resolve_mxfp4_tactic_optimizations(
        core,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
    )
    if selected["nvfp4_sf_stages"] and (
        selected["mma_tiler_mnk"][1:] != (64, 256) or selected["pingpong"]
    ):
        raise ValueError("NVFP4 scale-ring candidates require non-pingpong N64/K256")
    return selected


def expand_nvfp4_sf2_candidates(
    seeds: Iterable[Mapping[str, Any]],
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    kperm: int = 64,
) -> list[dict[str, Any]]:
    """Keep unique legal sf0 seeds and append R1/R2 for N64/K256 cores.

    No device-name, clamp, live-token or overlap gate is applied. The kernel
    constructor checks the real layout budget and requires four AB stages.
    """
    validate_nvfp4_kperm(kperm)
    shape = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
    )
    result = []
    for seed in seeds:
        selected = _validate_candidate(seed, **shape)
        if selected["nvfp4_sf_stages"]:
            raise ValueError("scale-ring expansion requires sf0 seeds")
        if selected not in result:
            result.append(selected)
    for seed in tuple(result):
        if seed["mma_tiler_mnk"][1:] != (64, 256) or seed["pingpong"]:
            continue
        for replicas in (1, 2):
            candidate = _scale_ring_tactic(seed, replicas)
            if candidate not in result:
                result.append(candidate)
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
    """Return shape-legal NV candidates in stable order for live measurement.

    The kernel constructor checks actual shared-memory resources. These
    candidates do not carry measurements from a different format or device.
    """
    validate_nvfp4_kperm(kperm)
    shape = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
    )
    seeds = list(
        hopper_mxfp4_candidates(
            max_tokens,
            routing_profile=routing_profile,
            **shape,
        )
    )
    seeds.extend(NVFP4_SF0_CANDIDATE_SEEDS)
    seeds.extend(NVFP4_SF2_CANDIDATE_SEEDS)
    result = []
    for seed in seeds:
        try:
            selected = _validate_candidate(seed, **shape)
        except ValueError:
            continue
        if selected not in result:
            result.append(selected)
    return result


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
    """Return a reference-model heuristic using the smallest adequate bucket.

    Capacities above the largest bucket use that bucket. Other model shapes
    use the caller's generic fallback. Clamp and SM90 product name do not
    restrict candidate legality; performance on unmeasured cases is unknown.
    The resolver applies explicit caller settings and actual resource checks.
    """
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    if type(max_tokens) is not int or max_tokens <= 0:
        return None
    context = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        num_topk=num_topk,
        routing_profile=routing_profile,
        kperm=kperm,
    )
    if any(
        type(context[name]) is not type(value) or context[name] != value
        for name, value in _H200_REFERENCE.items()
    ):
        return None
    buckets = tuple(NVFP4_DEFAULT_TACTICS_BY_TOKENS)
    bucket = next((value for value in buckets if value >= max_tokens), buckets[-1])
    profile = NVFP4_DEFAULT_TACTICS_BY_TOKENS[bucket]
    return dict(
        requested_tactic=normalize_hopper_nvfp4_tactic(profile["requested_tactic"]),
        requested_config=dict(profile["requested_config"]),
    )


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
    """Bind NV layout/schedule and source policy, excluding capacity buckets.

    Model/hardware and capacity/live tokens are separate knob-cache axes;
    excluding capacity here permits the cache's nearby-bucket lookup.
    """
    validate_nvfp4_kperm(kperm)
    if overlap is not None and type(overlap) is not bool:
        raise ValueError("overlap must be None or bool")
    from .mxfp4_tuner import hopper_mxfp4_cache_provenance_sha256

    payload = dict(
        version=NVFP4_TUNING_VERSION,
        kperm=kperm,
        overlap=overlap,
        geometry_provenance=hopper_mxfp4_cache_provenance_sha256(
            routing_profile=routing_profile,
        ),
    )
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
