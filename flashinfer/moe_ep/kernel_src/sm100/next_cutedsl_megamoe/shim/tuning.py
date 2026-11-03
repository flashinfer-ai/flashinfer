# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Candidate generation for Blackwell standard and GenPhase kernels.

The estimate ranks the first candidate; it is not a measured performance
guarantee. Collective tuning must validate and time candidates on the actual
workload before recording a winner. No routing tensors are inspected here.
"""

from __future__ import annotations

import dataclasses
from typing import Any

KNOB_KEYS = ("mma_tiler_mnk", "cluster_shape_mn", "schedule_policy")


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
    if config.quant_kind == "nvfp4":
        n, cluster = 128, 4
    elif config.quant_kind == "mxfp4_mxfp8":
        cluster = 4
    if sms < cluster:
        raise ValueError(
            f"{config.quant_kind} GenPhase requires at least {cluster} SMs"
        )
    if tokens:
        hint = 4 * (sms // cluster)
    return {
        "mma_tiler_mnk": (256, n, config.gather_tile_k),
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
    for tile, cluster in config.admitted_geometries:
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


STANDARD_KNOB_KEYS = (
    "mma_tiler_mnk",
    "cluster_shape_mn",
    "fallback_cluster_shape_mn",
    "schedule_policy",
    "work_id_mode",
    "fc2_use_bulk",
    "fc2_tma_stages",
    "epi_flag_batches",
    "token_in_flag_batch",
    "token_back_mode",
    "reduce_topk_in_kernel",
)


def knob_keys(config):
    return KNOB_KEYS if config.kernel_variant == "genphase" else STANDARD_KNOB_KEYS


def is_valid_sm100_knobs(knobs, config):
    if config.kernel_variant == "genphase":
        return is_valid_sm100_genphase_knobs(knobs, config)
    if unknown := set(knobs) - set(STANDARD_KNOB_KEYS):
        raise ValueError(f"unknown Blackwell standard knob keys: {sorted(unknown)}")
    try:
        dataclasses.replace(config, **knobs)
    except (TypeError, ValueError):
        return False
    return True


def sm100_standard_candidates(config, *, allow_nondeterministic=False):
    """Sweep tile, launch, epilogue batching and FC2 stores at fixed numerics.

    Keep the caller's initial configuration in the grid. In-kernel reduction
    is opt-in and cannot replace a custom or persistent finalizer.
    """
    base = {key: getattr(config, key) for key in STANDARD_KNOB_KEYS}
    base["mma_tiler_mnk"] = config.resolved_mma_tiler_mnk
    proposed = [base]
    reduction = (
        (False, True)
        if (
            allow_nondeterministic
            and config.combine_dtype == "bf16"
            and config.apply_routing_weights_before_fc2
            and not config.use_custom_finalize
            and not config.use_persistent_finalize_kernel
        )
        else (False,)
    )
    for tile_n in (128, 256):
        for cluster, fallback, schedule, work in (
            ((2, 1), None, ("grouped", None), "grid_stride"),
            ((4, 1), (2, 1), ("phase_interleave", None), "atomic_counter"),
        ):
            for epi in ((1, 4), (2, 4)):
                for bulk in (False, True):
                    for reduce in reduction:
                        proposed.append(
                            dict(
                                # K128 admits both N tiles on Blackwell. Keep
                                # the caller's original K depth in base above.
                                mma_tiler_mnk=(256, tile_n, 128),
                                cluster_shape_mn=cluster,
                                fallback_cluster_shape_mn=fallback,
                                schedule_policy=schedule,
                                work_id_mode=work,
                                fc2_use_bulk=bulk,
                                fc2_tma_stages=2 if bulk else None,
                                epi_flag_batches=epi,
                                token_in_flag_batch=1,
                                token_back_mode="epi_warps",
                                reduce_topk_in_kernel=reduce,
                            )
                        )
    candidates = []
    for knobs in proposed:
        if knobs["reduce_topk_in_kernel"] and not allow_nondeterministic:
            continue
        if knobs not in candidates and is_valid_sm100_knobs(knobs, config):
            candidates.append(knobs)
    if not candidates:
        raise ValueError("no valid Blackwell standard tuning candidates")
    return candidates


def sm100_schedule_candidates(base, config):
    """Sweep phase-interleave hints and token-input batching at fixed geometry."""
    return [
        knobs
        for hint in (None, 3, 4, 6)
        for batch in (1, 4)
        if is_valid_sm100_knobs(
            knobs := {
                **base,
                "schedule_policy": ("phase_interleave", hint),
                "work_id_mode": "atomic_counter",
                "token_in_flag_batch": batch,
            },
            config,
        )
    ]
