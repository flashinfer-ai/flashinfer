# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Candidate generation and strategy validation for fused Hopper MXFP4."""

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
    MXFP4_TUNING_TOKEN_BUCKETS,
    hopper_mxfp4_default_tactic,
    normalize_hopper_mxfp4_routing_profile,
    _ordered_base_candidates,
    _tactic,
    validate_hopper_mxfp4_tactic,
)


MXFP4_STRATEGY_FIELDS = frozenset({"fc2_tail_n8", "fc1_ready_mode", "tail_split_pairs"})


# Pull-buffer tactic fields: the compact (one slot per active warp) buffer,
# rows in flight per active dispatch warp, and the per-pass rotation of the
# warp -> row assignment.  A normalized tactic carries them only when they
# differ from these defaults, so tactics, candidate sets and cache entries that
# do not use them keep their identity.  Applying a tactic resets omitted fields
# to the defaults.
MXFP4_PULL_DEFAULTS: dict[str, Any] = {
    "compact_pull_buffer": False,
    "dispatch_pull_batch": 1,
    "dispatch_pull_rotate": False,
}


def _normalize_pull_fields(
    tactic: Mapping[str, Any], core: Mapping[str, Any]
) -> dict[str, Any]:
    compact = tactic.get("compact_pull_buffer", False)
    batch = tactic.get("dispatch_pull_batch", 1)
    rotate = tactic.get("dispatch_pull_rotate", False)
    if type(compact) is not bool:
        raise ValueError("compact_pull_buffer must be a bool")
    if type(batch) is not int or not 1 <= batch <= 4:
        raise ValueError("dispatch_pull_batch must be an int in 1..4")
    if type(rotate) is not bool:
        raise ValueError("dispatch_pull_rotate must be a bool")
    if batch > 1:
        # The batch keeps rows in the idle slots of the 4-slot pull buffer of
        # the serial epi_warps dispatch path.
        if compact:
            raise ValueError(
                "dispatch_pull_batch > 1 requires compact_pull_buffer=False"
            )
        if (
            core["token_back_mode"] != "epi_warps"
            or core["dedup_dispatch"]
            or core["grouped_token_back"]
        ):
            raise ValueError(
                "dispatch_pull_batch > 1 requires token_back_mode='epi_warps' "
                "without dedup_dispatch / grouped_token_back"
            )
        if batch * core["active_dispatch_warps"] > 4:
            raise ValueError(
                "dispatch_pull_batch * active_dispatch_warps must not exceed the 4 pull slots"
            )
    selected = {
        "compact_pull_buffer": compact,
        "dispatch_pull_batch": batch,
        "dispatch_pull_rotate": rotate,
    }
    return {
        name: value
        for name, value in selected.items()
        if value != MXFP4_PULL_DEFAULTS[name]
    }


def normalize_mxfp4_optimization_tactic(
    tactic: Mapping[str, Any],
) -> dict[str, Any]:
    """Expand a complete legacy tactic or validate a complete strategy tactic.

    Legacy 17/19-field tactics keep tail-pair scheduling off. Partial FC2-tail
    and readiness identities are rejected. Returning every strategy field makes
    an old tactic reset the protocol instead of retaining a previous setting.
    """
    if not isinstance(tactic, Mapping):
        raise TypeError("MXFP4 tactic must be a mapping")
    pull_fields = {k: v for k, v in tactic.items() if k in MXFP4_PULL_DEFAULTS}
    tactic = {k: v for k, v in tactic.items() if k not in MXFP4_PULL_DEFAULTS}
    legacy_strategy_fields = MXFP4_STRATEGY_FIELDS - {"tail_split_pairs"}
    present = legacy_strategy_fields.intersection(tactic)
    if present and present != legacy_strategy_fields:
        raise ValueError(
            "MXFP4 strategy tactic requires both fc2_tail_n8 and fc1_ready_mode"
        )
    core = validate_hopper_mxfp4_tactic(
        {
            key: value
            for key, value in tactic.items()
            if key not in MXFP4_STRATEGY_FIELDS
        },
    )
    tail = tactic.get("fc2_tail_n8", False)
    ready = tactic.get("fc1_ready_mode", "tile")
    validate_mxfp4_optional_optimizations(fc2_tail_n8=tail, fc1_ready_mode=ready)
    tail_pairs = tactic.get("tail_split_pairs", False)
    if type(tail_pairs) is not bool:
        raise ValueError("tail_split_pairs must be a bool")
    if tail_pairs:
        if core["cluster_shape_mnk"] != (1, 2, 1):
            raise ValueError(
                "MXFP4 tail_split_pairs requires cluster_shape_mnk=(1, 2, 1)"
            )
        if ready != "tile":
            raise ValueError("MXFP4 tail_split_pairs requires fc1_ready_mode='tile'")
    return dict(
        core,
        fc2_tail_n8=tail,
        fc1_ready_mode=ready,
        tail_split_pairs=tail_pairs,
        **_normalize_pull_fields(pull_fields, core),
    )


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


def _tail_pair_candidate_extensions() -> list[dict[str, Any]]:
    """Retain measured large-token anchors and one tail neighbor per anchor.

    The eight frozen block-permutation winners supply the small-token seeds.
    Four measured large-token strategies reuse the same M256N64K256 layout.
    These are candidates for live measurement, not new heuristic winners.
    """
    seeds = [
        normalize_mxfp4_optimization_tactic(hopper_mxfp4_default_tactic(token))
        for token in MXFP4_TUNING_TOKEN_BUCKETS
    ]
    large = [
        dict(seeds[-1], group_hint=group, fc2_tail_n8=True, fc1_ready_mode=ready)
        for group, ready in (
            (528, "tile"),
            (256, "k256"),
            (512, "k256"),
            (128, "k256"),
        )
    ]
    # Whole-tile readiness is required by independent expert-tail scheduling.
    # Keep N8 where already selected: it is compatible with this task mapping.
    tails = [
        dict(
            tactic,
            cluster_shape_mnk=(1, 2, 1),
            fc1_ready_mode="tile",
            tail_split_pairs=True,
        )
        for tactic in (*seeds, *large)
    ]
    return [*large, *tails]


def mxfp4_tail_candidate_provenance() -> dict[str, Any]:
    """Bind per-candidate eligibility and exact added strategies into the cache."""
    return {
        "eligibility": "per_candidate_v1",
        "candidate_union_sha256": mxfp4_optimization_candidate_sha256(
            _tail_pair_candidate_extensions()
        ),
    }


# The FP8 heuristic's buckets: 8, 16, ..., 32768 tokens per rank.
MXFP4_FUSED_DEFAULT_TOKEN_BUCKETS = tuple(1 << power for power in range(3, 16))


def _strategy(
    core: Mapping[str, Any],
    *,
    n8: bool = False,
    ready: str = "tile",
    tail_pairs: bool = False,
    compact: bool = False,
    batch: int = 1,
    rotate: bool = False,
) -> dict[str, Any]:
    return normalize_mxfp4_optimization_tactic(
        dict(
            core,
            fc2_tail_n8=n8,
            fc1_ready_mode=ready,
            tail_split_pairs=tail_pairs,
            compact_pull_buffer=compact,
            dispatch_pull_batch=batch,
            dispatch_pull_rotate=rotate,
        )
    )


def _folded_n64(tile_k: int, group_hint: int, stages: int, cluster=(1, 1, 1)):
    return _tactic(
        (256, 64, tile_k),
        group_hint=group_hint,
        stages=stages,
        cluster=cluster,
        active_dispatch_warps=1,
        fc1_early_done_publish=True,
        fold_producer_warps=True,
    )


def _k256_fused_defaults() -> dict[int, dict[str, Any]]:
    # H7168/I3072/E384/top-6/EP4. T8 and T32 keep the legacy winners and T16
    # takes a base candidate; the rest add compact pull buffers, K256 FC1
    # readiness, FC2 N8 tails, tail pairs and rotation where they measured faster.
    n16 = dict(load_balance="static", fc1_store_offload=True)
    t8 = _strategy(_tactic((256, 16, 256), group_hint=132, stages=1, **n16))
    t16 = _strategy(_tactic((256, 16, 256), group_hint=512, stages=2))
    t32 = _strategy(
        _tactic((256, 16, 256), group_hint=132, stages=1, dedup_dispatch=True, **n16)
    )
    t64 = _strategy(
        _tactic(
            (256, 16, 256),
            group_hint=330,
            stages=1,
            cluster=(1, 1, 1),
            dedup_dispatch=True,
            fc1_store_offload=True,
        )
    )
    t256 = _strategy(
        _tactic(
            (256, 32, 256),
            group_hint=330,
            stages=2,
            cluster=(1, 1, 1),
            active_dispatch_warps=2,
            fc1_store_offload=True,
        ),
        compact=True,
    )
    t512 = _strategy(
        _folded_n64(256, 512, 1, cluster=(2, 1, 1)), ready="k256", compact=True
    )
    t1024 = _strategy(
        _folded_n64(256, 528, 1, cluster=(2, 1, 1)),
        n8=True,
        ready="k256",
        compact=True,
    )
    t4096 = _strategy(
        _folded_n64(256, 528, 1, cluster=(1, 2, 1)),
        n8=True,
        tail_pairs=True,
        compact=True,
    )
    t32768 = _strategy(
        _folded_n64(256, 330, 1, cluster=(1, 2, 1)),
        tail_pairs=True,
        compact=True,
        rotate=True,
    )
    return {
        8: t8,
        16: t16,
        32: t32,
        64: t64,
        128: t64,
        256: t256,
        512: t512,
        1024: t1024,
        2048: t1024,
        4096: t4096,
        8192: t4096,
        16384: t4096,
        32768: t32768,
    }


def _k128_fused_defaults() -> dict[int, dict[str, Any]]:
    # H2560/I640/E512/top-10/EP2: unfolded N16 up to T256, N32 at T512, then
    # folded N64 with batched, rotated dispatch pulls.
    def unfolded(tile_n, group_hint, stages, warps, load_balance="atomic_counter"):
        return _strategy(
            _tactic(
                (256, tile_n, 128),
                group_hint=group_hint,
                stages=stages,
                cluster=(1, 1, 1),
                load_balance=load_balance,
                active_dispatch_warps=warps,
                fc1_store_offload=True,
            )
        )

    t8 = unfolded(16, 132, 1, 2, "static")
    t16 = unfolded(16, 330, 2, 2)
    t32 = unfolded(16, 330, 2, 4)
    t64 = unfolded(16, 330, 2, 4, "static")
    t512 = unfolded(32, 330, 2, 4)
    t1024 = _strategy(_folded_n64(128, 264, 1), batch=2, rotate=True)
    t4096 = _strategy(_folded_n64(128, 330, 3), batch=3, rotate=True)
    t8192 = _strategy(_folded_n64(128, 264, 3), batch=3, rotate=True)
    t32768 = _strategy(_folded_n64(128, 330, 1), batch=3, rotate=True)
    return {
        8: t8,
        16: t16,
        32: t32,
        64: t64,
        128: t16,
        256: t32,
        512: t512,
        1024: t1024,
        2048: t1024,
        4096: t4096,
        8192: t8192,
        16384: t8192,
        32768: t32768,
    }


# Per-bucket defaults measured on H200 with block-permutation routing
# (October 2026 sweeps), as complete tactics including strategy and pull fields.
# A model uses the table of the largest tile K that divides its hidden and
# intermediate sizes. The legacy 8-bucket table in mxfp4_tuner still orders the
# geometry candidates, seeds the tail-pair candidates and backs the NV fallback.
_FUSED_DEFAULTS = {
    MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE: (
        (256, _k256_fused_defaults()),
        (128, _k128_fused_defaults()),
    ),
}


def _fused_default_table(
    hidden: int, intermediate: int, routing_profile: str
) -> dict[int, dict[str, Any]]:
    profile = normalize_hopper_mxfp4_routing_profile(routing_profile)
    for tile_k, table in _FUSED_DEFAULTS.get(profile, ()):
        if hidden % tile_k == 0 and intermediate % tile_k == 0:
            return table
    return {}


# Token-tile widths for the folded M256 N64 defaults. One N72/N80 task holds an
# expert's 65-72/65-80 rows where N64 needs a second, nearly empty tile (4-38%
# faster on H200); at equal tile counts the wider tiles measured 3-15% slower,
# so ties keep N64.
_ROW_TILE_WIDTHS = (64, 72, 80)


def _row_tile_variants(tactic: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    m, n, k = tactic["mma_tiler_mnk"]
    if (m, n) != (256, 64) or not tactic["fold_producer_warps"]:
        return {}
    # FC2 N8 tails and K256 FC1 readiness are N64-only protocols.
    return {
        width: dict(
            tactic,
            mma_tiler_mnk=(m, width, k),
            fc2_tail_n8=False,
            fc1_ready_mode="tile",
        )
        for width in _ROW_TILE_WIDTHS[1:]
    }


def _row_tiled(
    tactic: dict[str, Any], routed_rows: int, num_experts: int
) -> dict[str, Any]:
    """Re-tile for the fewest tasks per expert at the balanced row count."""
    variants = _row_tile_variants(tactic)
    if not variants:
        return tactic
    width = min(
        _ROW_TILE_WIDTHS,
        key=lambda w: (-(-routed_rows // (num_experts * w)), w),
    )
    return variants.get(width, tactic)


def hopper_mxfp4_fused_default_tactic(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    num_topk: int | None = None,
) -> dict[str, Any] | None:
    """Return the measured per-bucket fused default, clamping above 32768 tokens.

    With ``num_topk``, a folded M256 N64 default takes the token-tile width of
    64/72/80 that needs the fewest tiles for the balanced rows per expert,
    ``max_tokens * num_topk * world_size / num_experts``.

    Returns None when no table covers the routing profile and tile alignment,
    or when the bucket's tactic is not eligible for this model/EP shape; the
    caller then keeps the legacy heuristic.
    """
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive integer")
    table = _fused_default_table(hidden, intermediate, routing_profile)
    if not table:
        return None
    bucket = next(
        (b for b in MXFP4_FUSED_DEFAULT_TOKEN_BUCKETS if max_tokens <= b),
        MXFP4_FUSED_DEFAULT_TOKEN_BUCKETS[-1],
    )
    tactic = dict(table[bucket])
    if num_topk is not None:
        if type(num_topk) is not int or num_topk <= 0:
            raise ValueError("num_topk must be a positive integer")
        tactic = _row_tiled(tactic, max_tokens * num_topk * world_size, num_experts)
    try:
        resolve_mxfp4_tactic_optimizations(
            tactic,
            hidden=hidden,
            intermediate=intermediate,
            num_experts=num_experts,
            world_size=world_size,
        )
    except ValueError:
        return None
    return tactic


def mxfp4_fused_default_provenance(routing_profile: str) -> dict[str, Any] | None:
    """Bind the per-bucket defaults that join the candidates into the cache."""
    tables = _FUSED_DEFAULTS.get(
        normalize_hopper_mxfp4_routing_profile(routing_profile)
    )
    if tables is None:
        return None
    tactics = [tactic for _, table in tables for tactic in table.values()]
    return {
        "eligibility": "largest_tile_k_table_n72_n80_tilings_v1",
        "candidate_union_sha256": mxfp4_optimization_candidate_sha256(
            [
                *tactics,
                *(v for t in tactics for v in _row_tile_variants(t).values()),
            ]
        ),
    }


def hopper_mxfp4_candidates(
    max_tokens: int,
    *,
    hidden: int,
    intermediate: int,
    num_experts: int,
    world_size: int,
    routing_profile: str = MXFP4_BLOCK_PERMUTATION_ROUTING_PROFILE,
    include_fused_defaults: bool = True,
) -> list[dict[str, Any]]:
    """The single ordered strategy domain for fused offline and online tuning.

    Token capacity only orders the existing geometry candidates. It never
    changes strategy eligibility or selects an alternate source tree. The
    eligible per-bucket defaults of the shape's table follow, then the N72 and
    N80 tilings of this capacity's bucket default, so tuning always measures
    the untuned default (the row-tiled one included). NV seeds its own domain
    with ``include_fused_defaults=False``: those defaults were measured on
    MXFP4 only.
    """
    base = _ordered_base_candidates(
        max_tokens,
        hidden=hidden,
        intermediate=intermediate,
        routing_profile=routing_profile,
    )
    shape = dict(
        hidden=hidden,
        intermediate=intermediate,
        num_experts=num_experts,
        world_size=world_size,
    )
    result = expand_mxfp4_optimization_candidates(base, **shape)
    extensions = list(_tail_pair_candidate_extensions())
    if include_fused_defaults:
        extensions += _fused_default_table(
            hidden, intermediate, routing_profile
        ).values()
        default = hopper_mxfp4_fused_default_tactic(
            max_tokens, routing_profile=routing_profile, **shape
        )
        if default is not None:
            extensions += _row_tile_variants(default).values()
    for candidate in extensions:
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
            result.append(dict(candidate))
    return result


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
