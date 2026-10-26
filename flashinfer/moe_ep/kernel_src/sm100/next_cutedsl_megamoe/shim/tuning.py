# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Host-only candidate generation for Blackwell GenPhase configuration tuning.

The estimate ranks the first candidate; it is not a measured performance
guarantee. Collective tuning must validate and time candidates on the actual
workload before recording a winner. No routing tensors are inspected here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

KNOB_KEYS = ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")
_GEOMETRIES = (
    ((256, 128, 128), (4, 1)),
    ((256, 64, 128), (4, 1)),
    ((256, 64, 128), (2, 1)),
)


def _effective_sms(config: Any, sm_count: int) -> int:
    """Validate the device SM count and apply the optional workspace SM limit."""
    if type(sm_count) is not int or sm_count < 2:
        raise ValueError("GenPhase tuning requires at least two SMs")
    return min(sm_count, config.max_sm_count or sm_count)


def _live_tokens(config: Any, live_tokens: int | None) -> int:
    """Validate the host workload count, defaulting to workspace capacity."""
    count = config.max_tokens_per_rank if live_tokens is None else live_tokens
    if type(count) is not int or not 0 <= count <= config.max_tokens_per_rank:
        raise ValueError("live_tokens must be an integer between zero and capacity")
    return count


def is_valid_sm100_genphase_knobs(knobs: dict[str, Any], config: Any) -> bool:
    """Validate the tuning dialect without allocating or compiling a kernel."""
    unknown = set(knobs) - set(KNOB_KEYS)
    if unknown:
        raise ValueError(f"unknown Blackwell GenPhase knob keys: {sorted(unknown)}")
    if "schedule_policy" in knobs:
        policy = knobs["schedule_policy"]
        if (
            not isinstance(policy, tuple)
            or len(policy) != 2
            or policy[0] != "grouped"
            or (
                policy[1] is not None and (type(policy[1]) is not int or policy[1] <= 0)
            )
        ):
            return False
    try:
        dataclasses.replace(config, **knobs)
    except (TypeError, ValueError):
        return False
    return True


def default_sm100_genphase_knobs(
    config: Any, *, sm_count: int, live_tokens: int | None = None
) -> dict[str, Any]:
    """Estimate a starting tile/cluster/hint from geometry and host token count.

    Sparse expert batches favor N64; dense batches amortize N128's wider
    token tile. CGA2 supplies more independent workers when the estimated
    FC1 work cannot fill one CGA4 wave. These estimates assume spread routing;
    skew and masked routes can change the best choice and should be tuned.
    """
    sms = _effective_sms(config, sm_count)
    tokens = _live_tokens(config, live_tokens)
    if tokens == 0:
        n, cluster = (128, 4) if sms >= 4 else (64, 2)
        hint = None
    else:
        expert_rows = (
            tokens * config.world_size * config.num_topk / config.num_total_experts
        )
        n = 64 if expert_rows < 64 or sms < 4 else 128
        active_experts = min(
            config.num_total_experts // config.world_size, tokens * config.num_topk
        )
        # A four-CTA cluster covers 512 FC1 output columns with two-CTA MMA.
        cga4_tiles = active_experts * ((2 * config.intermediate + 511) // 512)
        cluster = 2 if n == 64 and (sms < 4 or cga4_tiles < sms // 4) else 4
        hint = 4 * (sms // cluster)
    return {
        "mma_tiler_mnk": (256, n, 128),
        "cluster_shape_mn": (cluster, 1),
        "schedule_policy": ("grouped", hint),
    }


def sm100_genphase_candidates(
    config: Any,
    *,
    sm_count: int,
    live_tokens: int | None = None,
    constraints: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Enumerate all three admitted geometries and original/two/four/eight-wave hints.

    The estimate is first; the original N128/CGA4 configuration
    remains in the candidate set. Explicit caller constraints are respected;
    they can introduce a hint outside the standard grid. Numerical operation
    attributes are never tuning axes.
    """
    sms = _effective_sms(config, sm_count)
    preferred = default_sm100_genphase_knobs(
        config, sm_count=sms, live_tokens=live_tokens
    )
    constraints = {} if constraints is None else dict(constraints)
    unknown = set(constraints) - set(KNOB_KEYS)
    if unknown:
        raise ValueError(f"unknown Blackwell GenPhase knob keys: {sorted(unknown)}")
    proposed = [preferred]
    for tile, cluster in _GEOMETRIES:
        if sms < cluster[0]:
            continue
        for hint in (
            4 * (sms // cluster[0]),
            None,
            2 * (sms // cluster[0]),
            8 * (sms // cluster[0]),
        ):
            proposed.append(
                dict(
                    mma_tiler_mnk=tile,
                    cluster_shape_mn=cluster,
                    schedule_policy=("grouped", hint),
                )
            )
    candidates = []
    for knobs in proposed:
        # Filter geometry constraints; overlaying one dimension can silently
        # turn an admitted pair into an invalid N128/CGA2 pair.
        if any(knobs[k] != v for k, v in constraints.items() if k != "schedule_policy"):
            continue
        if "schedule_policy" in constraints:
            knobs = {**knobs, "schedule_policy": constraints["schedule_policy"]}
        if knobs not in candidates and is_valid_sm100_genphase_knobs(knobs, config):
            candidates.append(knobs)
    if not candidates:
        raise ValueError(
            "no admitted Blackwell GenPhase candidates satisfy the constraints"
        )
    return candidates
