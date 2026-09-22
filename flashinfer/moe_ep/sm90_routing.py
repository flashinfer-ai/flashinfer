# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Pure-host routing identities and generators for SM90 MoE benchmarks.

The production kernels consume rank-local route tensors, but tuning and
benchmark provenance must identify the complete global ``[rank, token, topk]``
array.  This module owns that host-side definition so every backend receives
the same route IDs for a given profile.  NumPy is imported lazily to keep this
module free of CUDA and PyTorch initialization side effects.

This is deliberately a dependency leaf: it must not import any ``moe_ep``
backend or kernel-drop module.  Keeping that boundary lets both production
shims and host-only benchmarks share routing identities without bootstrapping
vendored CuTe DSL packages.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal


SM90_ROUTING_PROFILE_BLOCK_PERMUTATION: Literal["block_permutation_v1"] = (
    "block_permutation_v1"
)
SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED: Literal[
    "published_exact_balanced_v1"
] = "published_exact_balanced_v1"

SM90_BENCHMARK_ROUTING_MODE_BLOCK_PERMUTATION = "block_permutation"
SM90_BENCHMARK_ROUTING_MODE_PUBLISHED_EXACT_BALANCED = "published_exact_balanced"

_SM90_ROUTING_PROFILES = frozenset(
    {
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    }
)
_PROFILE_BY_BENCHMARK_MODE = {
    SM90_BENCHMARK_ROUTING_MODE_BLOCK_PERMUTATION: (
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
    ),
    SM90_BENCHMARK_ROUTING_MODE_PUBLISHED_EXACT_BALANCED: (
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
    ),
}
_BENCHMARK_MODE_BY_PROFILE = {
    profile: mode for mode, profile in _PROFILE_BY_BENCHMARK_MODE.items()
}


def normalize_sm90_routing_profile(routing_profile: object) -> str:
    """Return one canonical SM90 routing profile, rejecting aliases strictly."""

    if not isinstance(routing_profile, str) or routing_profile not in (
        _SM90_ROUTING_PROFILES
    ):
        raise ValueError(
            "routing_profile must be exactly one of "
            f"{sorted(_SM90_ROUTING_PROFILES)!r}; got {routing_profile!r}"
        )
    return routing_profile


def sm90_routing_profile_from_benchmark_mode(mode: object) -> str:
    """Map the benchmark's stable legacy CLI spelling to a canonical profile."""

    if not isinstance(mode, str) or mode not in _PROFILE_BY_BENCHMARK_MODE:
        raise ValueError(
            "unsupported SM90 benchmark routing mode; expected one of "
            f"{sorted(_PROFILE_BY_BENCHMARK_MODE)!r}, got {mode!r}"
        )
    return _PROFILE_BY_BENCHMARK_MODE[mode]


def sm90_benchmark_mode_from_routing_profile(routing_profile: object) -> str:
    """Map a canonical routing profile to the benchmark's legacy CLI spelling."""

    normalized = normalize_sm90_routing_profile(routing_profile)
    return _BENCHMARK_MODE_BY_PROFILE[normalized]


def _validate_geometry(
    *, world_size: int, tokens: int, topk: int, total_experts: int
) -> None:
    for name, value, minimum in (
        ("world_size", world_size, 1),
        ("tokens", tokens, 0),
        ("topk", topk, 1),
        ("total_experts", total_experts, 1),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer, got {value!r}")
    if topk > total_experts:
        raise ValueError("topk cannot exceed total_experts")
    if total_experts % world_size:
        raise ValueError("total_experts must be divisible by world_size")


def generate_sm90_block_permutation_routes_numpy(
    *, world_size: int, tokens: int, topk: int, total_experts: int, seed: int
) -> Any:
    """Return the historical padded block-permutation global route array."""

    import numpy as np

    _validate_geometry(
        world_size=world_size,
        tokens=tokens,
        topk=topk,
        total_experts=total_experts,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if tokens == 0:
        return np.empty((world_size, 0, topk), dtype=np.int64)

    rng = np.random.default_rng(seed)
    padded = (tokens + total_experts - 1) // total_experts * total_experts
    num_blocks = padded // total_experts
    expert_permutations = rng.random((world_size, num_blocks, total_experts)).argsort(
        axis=-1
    )
    topk_offsets = rng.random((world_size, num_blocks, total_experts)).argsort(axis=-1)[
        ..., :topk
    ]
    token_offsets = np.arange(total_experts)[None, None, :, None]
    expert_indices = (token_offsets + topk_offsets[:, :, None, :]) % total_experts
    topk_blocks = np.take_along_axis(
        expert_permutations[..., None], expert_indices, axis=2
    )
    return topk_blocks.reshape(world_size, padded, topk)[:, :tokens, :]


def _generate_sm90_published_exact_balanced_routes_cyclic_numpy(
    *,
    np: Any,
    world_size: int,
    tokens: int,
    topk: int,
    total_experts: int,
    seed: int,
) -> Any:
    """Construct exact-balanced routes when the historical greedy path fails."""

    routed_rows = tokens * topk
    local_experts = total_experts // world_size
    rng = np.random.default_rng(seed)
    owner_permutation = rng.permutation(world_size)
    local_permutations = np.stack(
        [rng.permutation(local_experts) for _ in range(world_size)]
    )
    phase = int(rng.integers(total_experts))

    # Each total_experts consecutive flat slots form a permutation of all
    # experts. Each source owns routed_rows slots, which is divisible by
    # world_size, so its owner counts are exact. A token consumes at most one
    # period, so its consecutive top-k slots cannot duplicate an expert.
    flat = np.arange(world_size * routed_rows, dtype=np.int64) + phase
    owner_slots = flat % world_size
    owners = owner_permutation[owner_slots]
    local_slots = (flat // world_size) % local_experts
    local = local_permutations[owners, local_slots]
    routes = owners * local_experts + local
    return routes.astype(np.int32, copy=False).reshape(world_size, tokens, topk)


def generate_sm90_published_exact_balanced_routes_numpy(
    *, world_size: int, tokens: int, topk: int, total_experts: int, seed: int
) -> Any:
    """Return the published Hopper exact-balanced global route array."""

    import numpy as np

    _validate_geometry(
        world_size=world_size,
        tokens=tokens,
        topk=topk,
        total_experts=total_experts,
    )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"seed must be an integer, got {seed!r}")
    if tokens == 0:
        return np.empty((world_size, 0, topk), dtype=np.int32)

    routed_rows = tokens * topk
    if routed_rows % world_size:
        raise ValueError("tokens * topk must be divisible by world_size")
    local_experts = total_experts // world_size
    rng = np.random.default_rng(seed)

    if routed_rows % total_experts == 0 and total_experts % topk == 0:
        permutations_per_rank = routed_rows // total_experts
        routes = np.empty(
            (world_size, permutations_per_rank, total_experts), dtype=np.int32
        )
        for source_rank in range(world_size):
            for permutation_idx in range(permutations_per_rank):
                routes[source_rank, permutation_idx] = rng.permutation(total_experts)
        return routes.reshape(world_size, tokens, topk)

    def cyclic_fallback() -> Any:
        return _generate_sm90_published_exact_balanced_routes_cyclic_numpy(
            np=np,
            world_size=world_size,
            tokens=tokens,
            topk=topk,
            total_experts=total_experts,
            seed=seed,
        )

    rows_per_owner = routed_rows // world_size
    owners = np.empty((world_size, routed_rows), dtype=np.int32)
    owner_template = np.repeat(np.arange(world_size), rows_per_owner)
    for source_rank in range(world_size):
        owners[source_rank] = rng.permutation(owner_template)

    routes = np.full_like(owners, -1)
    for owner in range(world_size):
        positions = np.argwhere(owners == owner)
        groups: dict[tuple[int, int], list[int]] = {}
        for source_rank, flat_row in positions:
            key = (int(source_rank), int(flat_row) // topk)
            groups.setdefault(key, []).append(int(flat_row))
        group_items = list(groups.items())
        rng.shuffle(group_items)
        group_items.sort(key=lambda item: len(item[1]), reverse=True)
        base_rows, extra_rows = divmod(routed_rows, local_experts)
        target = np.full(local_experts, base_rows, dtype=np.int32)
        if extra_rows:
            target[rng.permutation(local_experts)[:extra_rows]] += 1
        remaining = target.copy()
        for (source_rank, _), flat_rows in group_items:
            order = np.lexsort((rng.random(local_experts), -remaining))
            chosen = order[: len(flat_rows)]
            if np.any(remaining[chosen] <= 0):
                return cyclic_fallback()
            rng.shuffle(flat_rows)
            for flat_row, local_expert in zip(flat_rows, chosen, strict=False):
                routes[source_rank, flat_row] = owner * local_experts + int(
                    local_expert
                )
                remaining[local_expert] -= 1
        if np.any(remaining):
            return cyclic_fallback()

    routes = routes.reshape(world_size, tokens, topk)
    if np.any(routes < 0):
        return cyclic_fallback()
    if np.any(np.diff(np.sort(routes, axis=2), axis=2) == 0):
        return cyclic_fallback()
    route_owners = routes.reshape(world_size, -1) // local_experts
    for source_rank in range(world_size):
        owner_counts = np.bincount(route_owners[source_rank], minlength=world_size)
        if not np.all(owner_counts == rows_per_owner):
            return cyclic_fallback()
    return routes


def generate_sm90_routing_numpy(
    *,
    routing_profile: object,
    world_size: int,
    tokens: int,
    topk: int,
    total_experts: int,
    seed: int,
) -> Any:
    """Return a canonical profile's global ``[rank, token, topk]`` routes."""

    profile = normalize_sm90_routing_profile(routing_profile)
    generator = {
        SM90_ROUTING_PROFILE_BLOCK_PERMUTATION: (
            generate_sm90_block_permutation_routes_numpy
        ),
        SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED: (
            generate_sm90_published_exact_balanced_routes_numpy
        ),
    }[profile]
    return generator(
        world_size=world_size,
        tokens=tokens,
        topk=topk,
        total_experts=total_experts,
        seed=seed,
    )


def sm90_route_ids_sha256(routes: Any) -> str:
    """Hash global route IDs as contiguous little-endian signed int64 bytes."""

    import numpy as np

    canonical = np.ascontiguousarray(routes, dtype="<i8")
    if canonical.ndim != 3:
        raise ValueError(
            "global routes must have shape [world_size, tokens, topk], got "
            f"{canonical.shape!r}"
        )
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def sm90_routing_audit_payload(
    routes: Any,
    *,
    routing_profile: object,
    seed: int,
    total_experts: int,
    world_size: int,
) -> dict[str, object]:
    """Return the official global route hash, balance, and tile-task audit."""

    import numpy as np

    profile = normalize_sm90_routing_profile(routing_profile)
    _validate_geometry(
        world_size=world_size,
        tokens=0,
        topk=1,
        total_experts=total_experts,
    )
    canonical = np.ascontiguousarray(routes, dtype="<i8")
    if canonical.ndim != 3 or canonical.shape[0] != world_size:
        raise ValueError(
            "global routes must have shape [world_size, tokens, topk] with "
            f"world_size={world_size}, got {canonical.shape!r}"
        )
    if canonical.shape[2] <= 0:
        raise ValueError("global routes must have a positive topk dimension")
    if canonical.size and (
        int(canonical.min()) < 0 or int(canonical.max()) >= total_experts
    ):
        raise ValueError("global routes contain an out-of-range expert ID")

    counts = np.bincount(canonical.reshape(-1), minlength=total_experts)
    local_experts = total_experts // world_size
    owners = []
    for owner in range(world_size):
        owner_counts = counts[owner * local_experts : (owner + 1) * local_experts]
        owners.append(
            {
                "owner": owner,
                "rows": int(owner_counts.sum()),
                "expert_count_min": int(owner_counts.min()),
                "expert_count_max": int(owner_counts.max()),
                "n32_tile_tasks": int(((owner_counts + 31) // 32).sum()),
                "n64_tile_tasks": int(((owner_counts + 63) // 64).sum()),
                "n128_tile_tasks": int(((owner_counts + 127) // 128).sum()),
            }
        )
    return {
        "mode": sm90_benchmark_mode_from_routing_profile(profile),
        "routing_profile": profile,
        "seed": seed,
        "world_size": world_size,
        "tokens_per_rank": int(canonical.shape[1]),
        "top_k": int(canonical.shape[2]),
        "num_experts": total_experts,
        "route_ids_sha256": sm90_route_ids_sha256(canonical),
        "expert_count_min": int(counts.min()),
        "expert_count_max": int(counts.max()),
        "owners": owners,
    }


__all__ = [
    "SM90_BENCHMARK_ROUTING_MODE_BLOCK_PERMUTATION",
    "SM90_BENCHMARK_ROUTING_MODE_PUBLISHED_EXACT_BALANCED",
    "SM90_ROUTING_PROFILE_BLOCK_PERMUTATION",
    "SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED",
    "generate_sm90_block_permutation_routes_numpy",
    "generate_sm90_published_exact_balanced_routes_numpy",
    "generate_sm90_routing_numpy",
    "normalize_sm90_routing_profile",
    "sm90_benchmark_mode_from_routing_profile",
    "sm90_route_ids_sha256",
    "sm90_routing_audit_payload",
    "sm90_routing_profile_from_benchmark_mode",
]
