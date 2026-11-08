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

from ...gemm.batch_invariant.generator import CLUSTERS, Knobs, knob_product

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


def all_valid_configs(
    N: int, K: int, E: int, chain: str = "plain", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped guarantee is claimed over, for a weight shape and a chain."""
    from .blocks.compile import can_implement_grouped

    return [
        k for k in grouped_knobs(clusters) if can_implement_grouped(k, N, K, E, chain)
    ]


def raster_for(cfg: Knobs, rows: int) -> str:
    """The grouped scheduler's raster as a structural rule, not an estimate term: along N once an expert's rows fill the
    record's CTA row tile (its groups then span several m-blocks and consecutive CTAs share the expert's A block while its
    weight tiles stream), along M below it (the weight tile is the larger operand of a few rows)."""
    return "n" if rows >= cfg.cta_tile_mn[0] else "m"


def pair_term(cfg: Knobs, rows: int) -> float:
    """The expected dummy share of a cluster of ``cm`` CTA tiles along M: every expert's rows are padded to the record's M
    extent (``cm`` CTA row tiles), and with the experts' row counts phased uniformly over that extent the expected padding
    is half an extent, so the cluster's records carry (rows + cm x tile / 2) / (rows + tile / 2) of the single-tile
    records' rows; the dense estimate at the average rows sees only the average's own tail."""
    cm = cfg.cluster_shape_mn[0]
    half = cfg.cta_tile_mn[0] / 2.0
    return (rows + half * cm) / (rows + half)


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
    """The ranked grouped records for one GEMM: the grouped space the body accepts for the weight shape and the chain,
    ordered by the dense class's estimate at ``(M = rows, N = n_active * N, K)`` (``n_active`` the active experts, ``E`` by
    default) times the pair term (the expected dummy half of the two-tile clusters along M), each tile family priced once
    (at the enumeration's raster: the dense estimate's tile-order term is the dense scheduler's, not the grouped one's) and
    the raster the structural rule gives the grouped scheduler (`raster_for`) first among its twins, best first, ties in
    the space's enumeration order; at a decode-sized ``rows`` the dense dispatcher's structural guard
    (`heuristic.decode_sane`: 1-CTA tiles in an N-cluster) selects the candidates. As in the dense ``rank_topk``, each of
    the best ceil(k/2) records is followed by its raster twin, then the rank order fills the list: distinct records, at most
    ``k``, never empty for an implementable shape."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import decode_sane, estimate_cycles
    from .blocks.compile import can_implement_grouped

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    space = [c for c in grouped_knobs() if can_implement_grouped(c, N, K, E, chain)]
    guarded = [c for c in space if decode_sane(c, rows, False)]

    def key(i, c):
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", "bf16", fam, sm_count, nominal=nominal
        )
        return (
            est * pair_term(c, rows),
            0 if c.raster_along == raster_for(c, rows) else 1,
            i,
        )

    ranked = [
        c for _k, c in sorted((key(i, c), c) for i, c in enumerate(guarded or space))
    ]
    members = set(space)
    out: List[Knobs] = []
    seen = set()
    for rec in ranked[: -(-k // 2)]:
        twin = dataclasses.replace(
            rec, raster_along="n" if rec.raster_along == "m" else "m"
        )
        for cand in (rec, twin):
            if cand not in seen and (cand is rec or cand in members):
                out.append(cand)
                seen.add(cand)
    for rec in ranked:
        if len(out) >= k:
            break
        if rec not in seen:
            out.append(rec)
            seen.add(rec)
    return out[:k]


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
