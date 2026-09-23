# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Hopper MXFP4 base tactics, default buckets, and persistent cache identity.

The complete shape-filtered online/offline tuning list is exposed by
``hopper_mxfp4_optimization_candidates`` in ``mxfp4_optimization``.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, Literal

from flashinfer.moe_ep.sm90_routing import (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION,
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED,
    normalize_sm90_routing_profile,
)

Mxfp4RoutingProfile = Literal["block_permutation_v1", "published_exact_balanced_v1"]
MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_BLOCK_PERMUTATION
)
MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE: Mxfp4RoutingProfile = (
    SM90_ROUTING_PROFILE_PUBLISHED_EXACT_BALANCED
)
MXFP4_TUNING_ROUTING_PROFILES = (
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE,
)
MXFP4_TUNING_TOKEN_BUCKETS = (8, 32, 64, 128, 256, 512, 1024, 2048)
_IMPLEMENTATION = "mxfp4_fused"


def _tactic(
    tile,
    *,
    group_hint,
    stages,
    cluster=(2, 1, 1),
    pingpong=False,
    load_balance="atomic_counter",
    token_back="epi_warps",
    dedup_dispatch=False,
    active_dispatch_warps=4,
    fc1_store_offload=False,
    fc1_early_done_publish=False,
    fold_producer_warps=False,
):
    return {
        "mma_tiler_mnk": tile,
        "cluster_shape_mnk": cluster,
        "group_hint": group_hint,
        "num_sched_stages": stages,
        "pingpong": pingpong,
        "swap_ab": True,
        "fp8_accum_mode": "1xacc",
        "load_balance_mode": load_balance,
        "token_back_mode": token_back,
        "in_kernel_fc2_reduce": False,
        "combine_format": "bf16",
        "grouped_token_back": False,
        "dedup_dispatch": dedup_dispatch,
        "active_dispatch_warps": active_dispatch_warps,
        "fc1_store_offload": fc1_store_offload,
        "fc1_early_done_publish": fc1_early_done_publish,
        "fold_producer_warps": fold_producer_warps,
    }


# Keep the measured union's stable order. Defaults below only reorder it;
# additional strategies are admitted by the shape/protocol policy.
_BASE_TACTICS = (
    _tactic(
        (256, 16, 256),
        group_hint=512,
        stages=1,
        load_balance="static",
        dedup_dispatch=True,
        active_dispatch_warps=2,
        fc1_store_offload=True,
    ),
    _tactic(
        (256, 32, 128),
        group_hint=330,
        stages=1,
        cluster=(1, 1, 1),
        active_dispatch_warps=2,
        fc1_store_offload=True,
    ),
    _tactic(
        (128, 16, 256), group_hint=78, stages=1, pingpong=True, load_balance="static"
    ),
    _tactic((256, 32, 128), group_hint=396, stages=2),
    _tactic((256, 16, 256), group_hint=396, stages=1),
    _tactic(
        (128, 64, 256),
        group_hint=528,
        stages=2,
        cluster=(1, 1, 1),
        token_back="reuse_dispatch_warps",
    ),
    _tactic(
        (256, 64, 256),
        group_hint=512,
        stages=1,
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
    ),
    _tactic((256, 16, 256), group_hint=396, stages=2, load_balance="static"),
    _tactic(
        (256, 64, 256),
        group_hint=528,
        stages=2,
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
    ),
    _tactic(
        (128, 16, 256), group_hint=78, stages=2, pingpong=True, load_balance="static"
    ),
    _tactic((256, 16, 256), group_hint=512, stages=2),
    _tactic(
        (256, 16, 256),
        group_hint=132,
        stages=1,
        load_balance="static",
        active_dispatch_warps=1,
        fc1_store_offload=True,
    ),
    _tactic(
        (128, 64, 256), group_hint=330, stages=2, token_back="reuse_dispatch_warps"
    ),
    _tactic(
        (256, 16, 256),
        group_hint=132,
        stages=1,
        load_balance="static",
        fc1_store_offload=True,
    ),
    _tactic(
        (256, 16, 256),
        group_hint=132,
        stages=1,
        load_balance="static",
        dedup_dispatch=True,
        fc1_store_offload=True,
    ),
    _tactic(
        (256, 64, 256),
        group_hint=512,
        stages=1,
        dedup_dispatch=True,
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
    ),
    _tactic((256, 16, 256), group_hint=512, stages=1),
)

# Indices into _BASE_TACTICS for the token buckets above.
_DEFAULT_INDICES = {
    "block_permutation_v1": (13, 14, 11, 0, 1, 15, 8, 6),
    "published_exact_balanced_v1": (4, 16, 16, 10, 7, 3, 12, 5),
}

_FUSED_FIELDS = frozenset(
    {
        "active_dispatch_warps",
        "cluster_shape_mnk",
        "combine_format",
        "dedup_dispatch",
        "fc1_early_done_publish",
        "fc1_store_offload",
        "fold_producer_warps",
        "fp8_accum_mode",
        "group_hint",
        "grouped_token_back",
        "in_kernel_fc2_reduce",
        "load_balance_mode",
        "mma_tiler_mnk",
        "num_sched_stages",
        "pingpong",
        "swap_ab",
        "token_back_mode",
    }
)

# Opaque inputs to the existing persistent cache key. Keep these values so
# this source-only cleanup can replay already measured winners. Historical
# reports are not loaded or validated; executable candidates and the current
# optimization policy are hashed separately below.
_CACHE_PROVENANCE = {
    "block_permutation_v1": {
        "artifact_manifest_sha256": "852cbb019a17a9e76991b4f82b5ae64fb3df3245f6e72ec43a20bb407cbbd477",
        "candidate_union_sha256": "dce1c6e5f22ac482928ebc2b07ba682665a41464ae44f367c306d2065c3b1926",
        "domain_sha256": "bff57fb6a658b968e40a8f0c0ec7b4d3e52392070905dae574de808cdeccc6ba",
        "external_schema_version": 1,
        "input_recipe_sha256": "ca6258b91b1b64a7953d9c4e1376f6503085df0af856845b0844f6e4a7829de8",
        "policy_sha256": "d3349c7ca1976b02ab076dcd87617de7552164d640481d7d614256d70da9a4cf",
        "routing_identity_sha256": "bedd230fa8afa768cc841438a91c0156de8fa6954f1af4ebbcb471478805190b",
        "runtime_manifest_sha256": "94ae92677c24630f28192d5df57704ff3ca7d6d94e9a24ebf46b761c198d9bf8",
        "source_manifest_sha256": "1081ba172754b107a9ab2360a6fac8d2e854fd5c1fc45305b571c935885d5277",
        "workload_recipe_sha256": "f4381c345df95b7da21bbe27766427aa5723aefa64f842c04c95fafae7f53352",
    },
    "published_exact_balanced_v1": {
        "artifact_manifest_sha256": "62733c7605f7233ac81c341084e0d589f4a91ca3f1aaaf1fac0660f7d1842a61",
        "domain_sha256": "86b846926d510fb013132462cf88122e36479db5b374645cfa3951021becf1a6",
        "external_schema_version": 1,
        "policy_sha256": "88503f1be444226eed1cf59d1083de5ffa92491d9bb5c626dcf7f96d22013137",
        "runtime_manifest_sha256": "f4112c7d0d7ead640239c1df3d7f4af74e2a1fb35cf5e821edd1beba9bba3e99",
        "source_manifest_sha256": "bcc0448df03348eb82addb5513152f7c7db42673769a40f9d67600d72c08a689",
        "workload_recipe_sha256": "e53d38f2f6fc708fb93a6405ad2129b49a3ca5499e34083e1aa2c706c9eacbfd",
    },
}


def _current_hopper_device_identity() -> tuple[str, tuple[int, int], int]:
    """Return ``(product name, compute capability, SM count)`` for CUDA."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Hopper MXFP4 cache/heuristic/autotune requires a CUDA device"
        )
    device = torch.cuda.current_device()
    name = str(torch.cuda.get_device_name(device))
    raw_capability = torch.cuda.get_device_capability(device)
    capability = (int(raw_capability[0]), int(raw_capability[1]))
    properties = torch.cuda.get_device_properties(device)
    sm_count = int(properties.multi_processor_count)
    return name, capability, sm_count


def require_hopper_mxfp4_fused_tuning_device() -> None:
    """Require a live SM90 device for fused cache/heuristic/autotune.

    Fused tactics have no fixed SM partition. Their residency is resolved
    from the live device, while ``group_hint`` remains a performance hint, so
    a product-name or SM-count restriction would reject otherwise legal
    Hopper configurations such as H20.
    """

    actual = _current_hopper_device_identity()
    if actual[1] != (9, 0):
        raise RuntimeError(
            "Hopper MXFP4 fused cache/heuristic/autotune requires SM90 "
            f"(CC 9.0); got {actual!r}"
        )


def normalize_hopper_mxfp4_routing_profile(
    routing_profile: str,
) -> Mxfp4RoutingProfile:
    """Compatibility wrapper around the canonical SM90 profile normalizer."""

    return normalize_sm90_routing_profile(routing_profile)  # type: ignore[return-value]


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive non-bool integer")
    return value


def _optional_positive_int(value: Any, name: str) -> int | None:
    return None if value is None else _positive_int(value, name)


def _triple(value: Any, name: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{name} must be an M,N,K triple")
    return tuple(_positive_int(item, name) for item in value)  # type: ignore[return-value]


def _require_exact_fields(
    tactic: Mapping[str, Any], expected: frozenset[str], implementation: str
) -> None:
    actual = set(tactic)
    if actual != expected:
        raise ValueError(
            f"{implementation} tactic fields differ: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def validate_hopper_mxfp4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and normalize one tactic from the bounded Hopper search domain.

    The returned mapping is a fresh copy whose tile and cluster fields are
    tuples. No defaults are inserted: callers must preserve the complete
    current fused tactic identity.
    """

    if not isinstance(tactic, Mapping):
        raise TypeError(f"tactic must be a mapping, got {type(tactic).__name__}")

    if any(
        name in tactic for name in ("fc2_tail_n8", "fc1_ready_mode", "tail_split_pairs")
    ):
        from .mxfp4_optimization import normalize_mxfp4_optimization_tactic

        return normalize_mxfp4_optimization_tactic(tactic)
    _require_exact_fields(tactic, _FUSED_FIELDS, _IMPLEMENTATION)
    if tactic["swap_ab"] is not True:
        raise ValueError("MXFP4 fused requires swap_ab=true")
    if not isinstance(tactic["pingpong"], bool):
        raise ValueError("MXFP4 fused pingpong must be bool")
    tile = _triple(tactic["mma_tiler_mnk"], "mma_tiler_mnk")
    m, n, k = tile
    if (
        m not in (128, 256)
        or n not in (16, 32, 64, 128)
        or k
        not in (
            128,
            256,
        )
    ):
        raise ValueError("illegal MXFP4 fused tile")
    if tactic["pingpong"] and m != 128:
        raise ValueError("MXFP4 fused pingpong requires M128")
    cluster = _triple(tactic["cluster_shape_mnk"], "cluster_shape_mnk")
    if cluster not in ((1, 1, 1), (2, 1, 1), (1, 2, 1), (2, 2, 1)):
        raise ValueError("illegal MXFP4 fused cluster")
    if tactic["fp8_accum_mode"] != "1xacc":
        raise ValueError("MXFP4 fused fixes 1xacc")
    if tactic["load_balance_mode"] not in ("static", "atomic_counter"):
        raise ValueError("illegal MXFP4 fused load balance")
    if tactic["token_back_mode"] not in (
        "epi_warps",
        "reuse_dispatch_warps",
        "standalone_warps",
    ):
        raise ValueError("illegal MXFP4 fused token-back")
    group_hint = _optional_positive_int(tactic["group_hint"], "group_hint")
    stages = _optional_positive_int(tactic["num_sched_stages"], "num_sched_stages")
    if tactic["in_kernel_fc2_reduce"] is not False:
        raise ValueError("MXFP4 fused fixes in-kernel reduce false")
    for name in (
        "dedup_dispatch",
        "grouped_token_back",
        "fc1_store_offload",
        "fc1_early_done_publish",
        "fold_producer_warps",
    ):
        if not isinstance(tactic[name], bool):
            raise ValueError(f"MXFP4 fused {name} must be bool")
    active_dispatch_warps = tactic["active_dispatch_warps"]
    if isinstance(active_dispatch_warps, bool) or active_dispatch_warps not in (
        1,
        2,
        4,
    ):
        raise ValueError("MXFP4 fused active_dispatch_warps must be 1, 2, or 4")
    if tactic["fold_producer_warps"] and active_dispatch_warps != 1:
        raise ValueError(
            "MXFP4 fused fold_producer_warps=True requires active_dispatch_warps=1"
        )
    if tactic["grouped_token_back"] is not False:
        raise ValueError(
            "MXFP4 fused production BF16-combine tactics fix grouped_token_back=false"
        )
    if tactic["combine_format"] != "bf16":
        raise ValueError("MXFP4 fused production tactics fix combine_format='bf16'")
    return {
        "active_dispatch_warps": active_dispatch_warps,
        "cluster_shape_mnk": cluster,
        "combine_format": "bf16",
        "dedup_dispatch": tactic["dedup_dispatch"],
        "fc1_early_done_publish": tactic["fc1_early_done_publish"],
        "fc1_store_offload": tactic["fc1_store_offload"],
        "fold_producer_warps": tactic["fold_producer_warps"],
        "fp8_accum_mode": "1xacc",
        "group_hint": group_hint,
        "grouped_token_back": False,
        "in_kernel_fc2_reduce": False,
        "load_balance_mode": tactic["load_balance_mode"],
        "mma_tiler_mnk": tile,
        "num_sched_stages": stages,
        "pingpong": tactic["pingpong"],
        "swap_ab": True,
        "token_back_mode": tactic["token_back_mode"],
    }


def is_valid_hopper_mxfp4_tactic(tactic: Mapping[str, Any]) -> bool:
    """Return whether ``tactic`` is a complete legal fused tactic."""

    try:
        validate_hopper_mxfp4_tactic(tactic)
    except (TypeError, ValueError):
        return False
    return True


def _candidate_id(implementation: str, tactic: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {"implementation": implementation, "tactic": tactic},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _candidate_union_sha256(records: tuple[dict[str, Any], ...]) -> str:
    canonical = json.dumps(
        [
            {
                "candidate_id": record["candidate_id"],
                "tactic": record["tactic"],
            }
            for record in records
        ],
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _base_candidates() -> list[dict[str, Any]]:
    return [dict(tactic) for tactic in _BASE_TACTICS]


def _base_candidate_union_sha256() -> str:
    return _candidate_union_sha256(
        tuple(
            {"candidate_id": _candidate_id(_IMPLEMENTATION, tactic), "tactic": tactic}
            for tactic in _BASE_TACTICS
        )
    )


def hopper_mxfp4_cache_provenance_sha256(
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> str:
    """Return the exact cache identity for the shipped tuning domain."""

    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    candidate_union_sha256 = _base_candidate_union_sha256()

    payload = {
        "execution_mode": "fused",
        "routing_profile": profile,
        "runtime_candidate_union_sha256": candidate_union_sha256,
        "tuning_provenance": _CACHE_PROVENANCE[profile],
    }
    from ..src.moe_hopper_fp8.mxfp4_policy import MXFP4_OPTIMIZATION_VERSION
    from .mxfp4_optimization import mxfp4_tail_candidate_provenance

    # Shape and world size are already separate knob-cache key fields.
    # This domain version binds eligibility/code changes without changing the
    # retained cache namespace.
    payload["optimization_domain"] = {
        "implementation": MXFP4_OPTIMIZATION_VERSION,
        "fc2_tail_n8": [False, True],
        "fc1_ready_mode": ["tile", "k256"],
        "tail_split_pairs": [False, True],
        "tail_candidate_extension": mxfp4_tail_candidate_provenance(),
    }
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def is_hopper_mxfp4_tactic_shape_compatible(
    tactic: Mapping[str, Any],
    *,
    hidden: int,
    intermediate: int,
) -> bool:
    """Whether a validated tactic's GEMM K axes divide the model shape."""

    _positive_int(hidden, "hidden")
    _positive_int(intermediate, "intermediate")
    normalized = validate_hopper_mxfp4_tactic(tactic)
    tile_k = int(normalized["mma_tiler_mnk"][2])
    return hidden % tile_k == 0 and intermediate % tile_k == 0


def _token_bucket(max_tokens: int) -> int:
    _positive_int(max_tokens, "max_tokens")
    for bucket in MXFP4_TUNING_TOKEN_BUCKETS:
        if max_tokens <= bucket:
            return bucket
    return MXFP4_TUNING_TOKEN_BUCKETS[-1]


def hopper_mxfp4_default_tactic(
    max_tokens: int,
    *,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> dict[str, Any]:
    """Return a copy of the ceil-bucket default, clamping above 2048 tokens."""
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    bucket_index = MXFP4_TUNING_TOKEN_BUCKETS.index(_token_bucket(max_tokens))
    return dict(_BASE_TACTICS[_DEFAULT_INDICES[profile][bucket_index]])


def _ordered_base_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
) -> list[dict[str, Any]]:
    """Put a legal default first, otherwise keep the base union's order."""
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    legal = [
        tactic
        for tactic in _base_candidates()
        if is_hopper_mxfp4_tactic_shape_compatible(
            tactic, hidden=hidden, intermediate=intermediate
        )
    ]
    if not legal:
        raise ValueError(
            f"no runtime MXFP4 fused tactic supports hidden={hidden}, intermediate={intermediate}"
        )
    default = hopper_mxfp4_default_tactic(max_tokens, routing_profile=profile)
    if default not in legal:
        return legal
    return [default, *(candidate for candidate in legal if candidate != default)]


__all__ = [
    "MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE",
    "MXFP4_PUBLISHED_EXACT_ROUTING_PROFILE",
    "MXFP4_TUNING_ROUTING_PROFILES",
    "MXFP4_TUNING_TOKEN_BUCKETS",
    "Mxfp4RoutingProfile",
    "hopper_mxfp4_cache_provenance_sha256",
    "hopper_mxfp4_default_tactic",
    "is_hopper_mxfp4_tactic_shape_compatible",
    "is_valid_hopper_mxfp4_tactic",
    "normalize_hopper_mxfp4_routing_profile",
    "require_hopper_mxfp4_fused_tuning_device",
    "validate_hopper_mxfp4_tactic",
]
