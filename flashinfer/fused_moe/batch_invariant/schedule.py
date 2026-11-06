# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMMs' dispatcher: the dense GEMM class's ranking heuristic at the grouped launch's shape.

A grouped GEMM of the MoE class is the dense class under a different scheduler (one segment of rows per expert), so
its records are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.schedule.rank_topk`: the
part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and operand bytes
are the grouped launch's: ``M`` = the rows of one active expert (the per-group tail of the row tile), ``N`` = the
active experts' weight rows side by side (every group's tiles count, and the weights, the dominant operand of the
op in every regime, are read once per group). The ranking is filtered by the grouped body's acceptance and the
grouped space (swizzle 1: the group scheduler has no raster swizzle). Numerics-neutral: every record of the space is
canonical (the same bits), so the dispatcher may depend on anything; the ranked list is what the autotuner profiles
under ``batch_invariant``.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import CLUSTERS, TILES_N, Knobs, knob_product
from ...gemm.batch_invariant.schedule import rank_topk

#: How many ranked records per GEMM the autotuner profiles (the Cartesian product over the two GEMMs is the tactic space).
TOP_K = 3


def active_experts(T: int, k: int, E: int) -> int:
    """The dispatcher's group count: every expert active, at most one per expanded row."""
    return max(1, min(E, T * k))


def rows_per_expert(T: int, k: int, E: int) -> int:
    """The dispatcher's size proxy: average rows per active expert."""
    return max(1, -(-(T * k) // active_experts(T, k, E)))


def grouped_knobs(clusters=None) -> List[Knobs]:
    """Every well-formed dense record at swizzle 1 (the grouped space before the body's acceptance)."""
    return knob_product(
        swizzles=(1,), clusters=CLUSTERS if clusters is None else clusters
    )


def _grouped(cfg: Knobs) -> Knobs:
    """A dense record as a grouped one: swizzle 1, no grid cap, the N tile clamped into the grouped space (the dense
    dispatcher's narrow decode tiles have no staged-epilogue layout: `blocks.compile.can_implement_grouped`)."""
    return Knobs(
        (cfg.mma_tiler_mn[0], max(cfg.mma_tiler_mn[1], TILES_N[0])),
        cfg.cluster_shape_mn,
        cfg.use_2cta_instrs,
        True,
        cfg.raster_along,
        1,
    )


def all_valid_configs(
    N: int, K: int, E: int, chain: str = "plain", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped guarantee is claimed over, for a weight shape and a chain."""
    from .blocks.compile import can_implement_grouped

    return [
        k for k in grouped_knobs(clusters) if can_implement_grouped(k, N, K, E, chain)
    ]


def rank_grouped(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: str,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
) -> List[Knobs]:
    """The ranked grouped records for one GEMM: the dense ranking at ``(M = rows, N = n_active * N)`` (``n_active``
    the active experts, ``E`` by default) mapped onto the grouped space, in rank order, de-duplicated, at least ``k``
    records when the space has them (the ranking is extended by the rest of the space in its enumeration order, so
    the list is never empty for an implementable shape)."""
    from .blocks.compile import can_implement_grouped

    out: List[Knobs] = []
    seen = set()
    ranked = rank_topk(
        rows,
        (n_active or E) * N,
        K,
        1,
        "k",
        sm_count,
        max(k, 10),
        family,
        "bf16",
        nominal,
    )
    for cand in [_grouped(c) for c in ranked] + grouped_knobs():
        if cand in seen or not can_implement_grouped(cand, N, K, E, chain):
            continue
        seen.add(cand)
        out.append(cand)
        if len(out) >= k:
            break
    return out


def default_configs(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    gated: bool = True,
    nominal: bool = False,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I for the ungated class), config for GEMM2 (N = H, K = I))``: the first
    record of each ranked list."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1, chain1 = (2 * I, "swiglu") if gated else (I, "relu2")
    c1 = rank_grouped(rows, n1, H, E, chain1, sm_count, family, 1, nominal, n_act)
    c2 = rank_grouped(rows, H, I, E, "plain", sm_count, family, 1, nominal, n_act)
    if not c1 or not c2:
        raise ValueError(
            f"no grouped record implements (H={H}, I={I}, E={E}, gated={gated})"
        )
    return c1[0], c2[0]
