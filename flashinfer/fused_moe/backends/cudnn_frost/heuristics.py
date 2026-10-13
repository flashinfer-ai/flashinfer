# Copyright (c) 2026 by FlashInfer team. Licensed under Apache-2.0.
"""Bounded stage candidates for shapes without offline Frost measurements.

These estimates order already legal artifacts; they do not establish kernel
legality or predict latency. Stage autotuning and complete-pipeline timing make
the final decision. No routing tensor is read back to the host.
"""

from __future__ import annotations

import math
import re

POLICY_VERSION = "frost-stage-heuristic-v2"
STAGE_BUDGET = 4
FUSED_BUDGET = 2


def _tile(kernel):
    meta = kernel.tactic_metadata
    tile = meta["cta_tile"]
    cluster = re.search(r"_cluster(\d+)x(\d+)_", meta["tile"])
    if cluster is None:
        raise ValueError(f"Missing Frost cluster geometry: {meta['tile']}")
    cm, cn = map(int, cluster.groups())
    return tile["m"], tile["n"], tile["k_bytes"], cm, cn


def stage_candidates(kernels, *, rows, n, k, experts, sm_count, budget=STAGE_BUDGET):
    """Rank a bounded, diverse pool for one GEMM using its actual N and K.

    Estimate balanced expert occupancy, tile padding, reduction iterations and
    cluster waves. Keep an alternate orientation and CTA group to hedge against
    skewed routing and errors in this simple model. Store variants share a tile
    family and cannot crowd every other family out of the pool.
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
        capacity = max(1, sm_count // cluster_size)
        waves = math.ceil(clusters / capacity)
        # K is determined by the wider operand. Mixed FP8/FP4 still uses
        # FP8's reduction tile; two FP4 operands double the element count.
        elements = (
            2
            if all(
                "float4" in kernel.contract[d] for d in ("token_dtype", "weight_dtype")
            )
            else 1
        )
        iterations = math.ceil(k / (kb * elements))
        padding = (math.ceil(tm / cm) * cm * m * math.ceil(nn / cn) * cn * tn) / (
            rm * rn
        )
        # A larger tile amortizes loads when there is enough parallel work.
        token_bytes = 0.5 if "float4" in kernel.contract["token_dtype"] else 1
        weight_bytes = 0.5 if "float4" in kernel.contract["weight_dtype"] else 1
        weight_bytes *= 2 if kernel.gated else 1
        am, bn = (
            (weight_bytes, token_bytes)
            if kernel.swap_ab
            else (token_bytes, weight_bytes)
        )
        traffic = (m * am + tn * bn) / (m * tn)
        estimate = waves * iterations * m * tn * padding * traffic
        return (estimate, cluster_size, meta["store_mode"] != "tma", kernel.artifact_id)

    ordered = sorted(kernels, key=score)
    selected = [ordered[0]]
    for field in ("swap_ab", "cta_group"):
        other = next(
            (
                a
                for a in ordered
                if a.tactic_metadata.get(field, False)
                != selected[0].tactic_metadata.get(field, False)
            ),
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
    first, second, fused, *, tokens, hidden, intermediate, experts, topk, sm_count
):
    """Four unfused choices per stage and a bounded fused FC1 pool."""
    common = dict(rows=tokens * topk, experts=experts, sm_count=sm_count)
    first = stage_candidates(first, n=intermediate, k=hidden, **common)
    second = stage_candidates(second, n=hidden, k=intermediate, **common)
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
    fused = stage_candidates(
        fused, n=intermediate, k=hidden, budget=fused_budget, **common
    )
    return first + fused, second
