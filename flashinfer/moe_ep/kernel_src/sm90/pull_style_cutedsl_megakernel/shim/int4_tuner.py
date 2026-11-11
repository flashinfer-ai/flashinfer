# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Bounded tactics for signed INT4/K128-BF16 weights with FP8 activations."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

from .mxfp4_optimization import (
    hopper_mxfp4_candidates,
    mxfp4_optimization_candidate_sha256,
    normalize_mxfp4_optimization_tactic,
)

INT4_TUNING_VERSION = "int4_k128_bf16_xor_reglut_linear_m64_v1"


def normalize_hopper_int4_tactic(tactic: Mapping[str, Any]) -> dict[str, Any]:
    selected = normalize_mxfp4_optimization_tactic(tactic)
    if selected["fc1_ready_mode"] != "tile" or selected["fc2_tail_n8"]:
        raise ValueError("INT4 requires fc1_ready_mode='tile' and fc2_tail_n8=False")
    return selected


def hopper_int4_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = "block_permutation_v1",
) -> list[dict[str, Any]]:
    """Reuse legal packed-weight schedules, measuring INT4 independently."""
    candidates = hopper_mxfp4_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
        routing_profile=routing_profile,
        include_fused_defaults=False,
    )
    return [
        normalize_hopper_int4_tactic(tactic)
        for tactic in candidates
        if tactic["fc1_ready_mode"] == "tile" and not tactic["fc2_tail_n8"]
    ]


def int4_candidate_sha256(candidates) -> str:
    identity = {
        "version": INT4_TUNING_VERSION,
        "schedules": mxfp4_optimization_candidate_sha256(
            normalize_hopper_int4_tactic(c) for c in candidates
        ),
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def hopper_int4_cache_provenance_sha256(
    *,
    max_tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = "block_permutation_v1",
) -> str:
    return int4_candidate_sha256(
        hopper_int4_candidates(
            max_tokens,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
            routing_profile=routing_profile,
        )
    )
