# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded stage candidates for shapes without offline Frost measurements.

These estimates order already legal artifacts; they do not establish kernel
legality or predict latency. Stage autotuning and complete-pipeline timing make
the final decision. No routing tensor is read back to the host.
"""

from __future__ import annotations

import math
import re

POLICY_VERSION = "frost-stage-heuristic-v12"
STAGE_BUDGET = 4
FUSED_BUDGET = 2
_ELEMENT_BYTES = {"bfloat16": 2, "float8_e4m3fn": 1, "float4_e2m1fn_x2": 0.5}


def _tile(kernel):
    meta = kernel.tactic_metadata
    tile = meta["cta_tile"]
    cluster = re.search(r"_cluster(\d+)x(\d+)(?:_|$)", meta["tile"])
    if cluster is None:
        raise ValueError(f"Missing Frost cluster geometry: {meta['tile']}")
    cm, cn = map(int, cluster.groups())
    return tile["m"], tile["n"], tile["k_bytes"], cm, cn


def stage_candidates(
    kernels,
    *,
    rows,
    n,
    k,
    experts,
    sm_count,
    cluster_capacities=(),
    budget=STAGE_BUDGET,
):
    """Rank a bounded, diverse pool for one GEMM using its actual N and K.

    Estimate balanced expert occupancy, tile padding, reduction iterations and
    cluster waves. Keep alternate orientations, stores and CTA groups to hedge
    against skewed routing and errors in this simple model. Store variants share
    a tile family and cannot crowd every other family out of the pool.
    """
    if not kernels or min(rows, n, k, experts, sm_count, budget) <= 0:
        return ()
    active = min(rows, experts)
    per_expert = math.ceil(rows / active)

    def score(kernel):
        m, tn, kb, cm, cn = _tile(kernel)
        meta = kernel.tactic_metadata
        rm, rn = (n, per_expert) if kernel.swap_ab else (per_expert, n)
        tm, nn = math.ceil(rm / m), math.ceil(rn / tn)
        clusters = active * math.ceil(tm / cm) * math.ceil(nn / cn)
        cluster_size = cm * cn
        # GPC packing limits can leave SMs idle for larger clusters. Use the
        # device's launch capacity, captured during preparation, when available.
        capacity = (
            cluster_capacities[cluster_size - 1]
            if cluster_size <= len(cluster_capacities)
            else max(1, sm_count // cluster_size)
        )
        if not capacity:
            return (math.inf, math.inf, cluster_size, kernel.artifact_id)
        waves = math.ceil(clusters / capacity)
        # K is determined by the wider operand. Mixed FP8/FP4 still uses
        # FP8's reduction tile; two FP4 operands double the element count.
        token_bytes = _ELEMENT_BYTES[kernel.contract["token_dtype"]]
        weight_bytes = _ELEMENT_BYTES[kernel.contract["weight_dtype"]]
        iterations = math.ceil(k * max(token_bytes, weight_bytes) / kb)
        padding = (math.ceil(tm / cm) * cm * m * math.ceil(nn / cn) * cn * tn) / (
            rm * rn
        )
        # A larger tile amortizes loads when there is enough parallel work.
        weight_bytes *= 2 if getattr(kernel, "gated", False) else 1
        am, bn = (
            (weight_bytes, token_bytes)
            if kernel.swap_ab
            else (token_bytes, weight_bytes)
        )
        traffic = (m * am + tn * bn) / (m * tn)
        estimate = waves * iterations * m * tn * padding * traffic
        # Sparse expert tiles favor masked vector stores; dense tiles can
        # amortize TMA's staging and synchronization. Keep the alternate store
        # below so this ordering never substitutes for pipeline measurement.
        token_tile = tn if kernel.swap_ab else m
        prefer_tma = per_expert >= token_tile
        store_mismatch = (meta["store_mode"] == "tma") != prefer_tma
        return (estimate, store_mismatch, cluster_size, kernel.artifact_id)

    ordered = sorted(kernels, key=score)
    selected = [ordered[0]]

    def axis(kernel, field):
        return (
            getattr(kernel, field, False)
            if field == "combines_output"
            else kernel.tactic_metadata.get(field, False)
        )

    m, tn, *_ = _tile(selected[0])
    token_tile = tn if selected[0].swap_ab else m
    # The two-slot fused pool needs different CTA groups for dense experts;
    # sparse experts benefit more from comparing masked STG with staged TMA.
    axes = (
        ("swap_ab", "cta_group", "store_mode", "combines_output")
        if per_expert >= token_tile
        else ("swap_ab", "store_mode", "cta_group", "combines_output")
    )
    for field in axes:
        other = next(
            (a for a in ordered if axis(a, field) != axis(selected[0], field)),
            None,
        )
        if other is not None and other not in selected and len(selected) < budget:
            selected.append(other)
    families = {a.tactic_metadata["tile"] for a in selected}
    for candidate in ordered:
        if len(selected) == budget:
            break
        if candidate.tactic_metadata["tile"] not in families:
            selected.append(candidate)
            families.add(candidate.tactic_metadata["tile"])
    for candidate in ordered:
        if len(selected) == budget:
            break
        if candidate not in selected:
            selected.append(candidate)
    return tuple(selected)


def select_stages(
    first,
    second,
    fused,
    *,
    tokens,
    hidden,
    intermediate,
    experts,
    topk,
    sm_count,
    cluster_capacities=(),
):
    """Bounded grouped-input/GATHER stages and fused FC1 alternatives."""
    common = dict(
        rows=tokens * topk,
        experts=experts,
        sm_count=sm_count,
        cluster_capacities=cluster_capacities,
    )
    gather = stage_candidates(
        tuple(a for a in first if getattr(a, "gathers_input", False)),
        n=intermediate,
        k=hidden,
        budget=STAGE_BUDGET,
        **common,
    )
    first = (
        stage_candidates(
            tuple(a for a in first if not getattr(a, "gathers_input", False)),
            n=intermediate,
            k=hidden,
            budget=2 if gather else STAGE_BUDGET,
            **common,
        )
        + gather
    )
    grouped = tuple(b for b in second if not getattr(b, "combines_output", False))
    combined = tuple(b for b in second if getattr(b, "combines_output", False))
    # Reserve a representative of both orientations within each reduction
    # layout. A single COMBINE slot could otherwise discard the swapped path
    # before complete-pipeline tuning can compare it with FC2 + finalize.
    second = tuple(
        candidate
        for pool in (grouped, combined)
        for candidate in stage_candidates(
            pool,
            n=hidden,
            k=intermediate,
            budget=STAGE_BUDGET // 2 if grouped and combined else STAGE_BUDGET,
            **common,
        )
    )
    if not first or not second:
        return (), ()
    fused_budget = FUSED_BUDGET
    if fused and all(
        "float4" in fused[0].contract[d] for d in ("token_dtype", "weight_dtype")
    ):
        token_tile = min(_tile(a)[1 if a.swap_ab else 0] for a in fused)
        if tokens * topk >= experts * token_tile:
            # Once a full token tile per expert is available, keep another
            # cluster family alongside the one-/two-CTA representatives.
            # NVFP4 fused epilogues can favor larger clusters even when
            # the padding/wave estimate orders the smaller cluster first.
            fused_budget += 1
    fused = tuple(
        candidate
        for gather in (False, True)
        for candidate in stage_candidates(
            tuple(a for a in fused if getattr(a, "gathers_input", False) == gather),
            n=intermediate,
            k=hidden,
            budget=fused_budget,
            **common,
        )
    )
    return first + fused, second
