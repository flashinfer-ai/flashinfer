# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The grouped GEMMs' dispatcher: the dense GEMM class's ranking heuristic at the grouped launch's shape.

A grouped GEMM of the MoE class is the dense class under a different scheduler (one segment of rows per expert), so
its records are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles`:
the part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and operand
bytes are the grouped launch's: ``M`` = the rows of one active expert (the per-group tail of the row tile), ``N`` =
the active experts' weight rows side by side (every group's tiles count, and the weights, the dominant operand of the
op in every regime, are read once per group). Two record spaces are ranked together: the grouped records (the dense
records at swizzle 1: the group scheduler has no raster swizzle) and the swapped records (:mod:`.blocks.swap_bf16`:
the weights on the MMA's M side, 128 rows per CTA, the group's tokens a narrow N tile of 8 / 16 / 32 rows), which the
dense model prices as it prices a swapped decode record -- the weights' rows on the M side with one CTA per 128 of
them, the tokens' rows on the N side in token tiles -- so the CTA count and the operand bytes per CTA of each body
enter one estimate and the pick follows the shape, never a rows-per-expert switch. The ranking is filtered by each
body's acceptance. Numerics-neutral: every record of either space is canonical (the same bits), so the dispatcher may
depend on anything; the ranked list is what the autotuner profiles under ``batch_invariant``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING, List, Optional, Tuple, Union

from ...gemm.batch_invariant.generator import CLUSTERS, Knobs, knob_product

if TYPE_CHECKING:
    from .blocks.epilogue import GroupedChain

#: How many ranked records per GEMM the autotuner profiles (the Cartesian product over the two GEMMs is the tactic space).
TOP_K = 3
#: Cycles per node of a chain's term in the exposed-epilogue term (empirical: a thread's dependent fp32 chain runs about one
#: node of the spec's S-expression per cycle; the term is the same for every part).
CHAIN_NODE_CYCLES = 1.0
#: Epilogue warpgroups of the grouped body (its subtiles are split between them): `blocks.scheduler.EPI_WARPGROUPS_DENSE`.
_EPI_WARPGROUPS = 2


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
    """The grouped record space the guarantee is claimed over, for a weight shape and a chain (the swapped records are the
    second space: :func:`all_valid_configs_swapped`; the guarantee covers both)."""
    from .blocks.compile import can_implement_grouped

    return [
        k for k in grouped_knobs(clusters) if can_implement_grouped(k, N, K, E, chain)
    ]


def swapped_knobs() -> List[Knobs]:
    """The swapped record space of the bf16 classes (:mod:`.blocks.swap_bf16`): the swapped 1-CTA records at the narrow token
    tiles (128 weight rows x 8 / 16 / 32 tokens), one per tile, no cluster, swizzle 1. Disjoint from the grouped space (the
    swapped flag); the body reads the same dense rows through a TMA box."""
    from .blocks.swap_bf16 import swapped_knobs as _swapped

    return _swapped()


def all_valid_configs_swapped(
    N: int, K: int, E: int, chain: str = "plain"
) -> List[Knobs]:
    """The swapped records the swapped body accepts for a weight shape and a chain."""
    from .blocks.compile import can_implement_swapped

    return [k for k in swapped_knobs() if can_implement_swapped(k, N, K, E, chain)]


def all_valid_configs_bf16(
    N: int, K: int, E: int, chain: str = "plain", clusters=None
) -> List[Knobs]:
    """Both record spaces for a weight shape and a chain: the grouped records, then the swapped ones."""
    return all_valid_configs(N, K, E, chain, clusters) + all_valid_configs_swapped(
        N, K, E, chain
    )


@functools.lru_cache(maxsize=None)
def chain_nodes(chain: Union[str, "GroupedChain"]) -> int:
    """The node count of the chain's term, the S-expression :mod:`.spec` builds for one output element: the gated term on the
    (gate, up) accumulator pair, the ungated term on one accumulator, or the plain conversion (one node). A legacy name maps
    onto the class's default chain of that kind (``swiglu`` = the gated SiLU, ``relu2``, ``plain``)."""
    from . import mterm as term
    from . import spec as M
    from .blocks.epilogue import GroupedChain

    if isinstance(chain, str):
        chain = GroupedChain.legacy(chain, "bf16")
    alg = term.TermAlgebra()
    if chain.pair:
        t = M.glu_elem(
            alg,
            alg.leaf("g", 0, 0),
            alg.leaf("u", 0, 0),
            chain.act,
            chain.out_fmt,
            chain.params,
        )
    elif chain.kind == "act":
        t = M.act_elem(alg, alg.leaf("x", 0), chain.act, chain.out_fmt, chain.params)
    else:
        return 1
    return term.sexpr(t).count("(")


def expected_token_tiles(rows: float, tile: int) -> float:
    """``E[ceil(X / tile) | X >= 1]`` for ``X ~ Poisson(rows)``: the token tiles an active expert needs when the expanded rows
    are routed uniformly at random, as a function of the average rows per active expert and the tile. The ranking prices the
    swapped records at this count rather than at the average's own ``ceil(rows / tile)``: a group whose rows spill past the tile
    streams its weight tile once more, and at a few rows per tile the spill is common."""
    lam = max(float(rows), 1e-6)
    sigma = math.sqrt(lam)
    lo, hi = max(1, int(lam - 10 * sigma) - 1), int(lam + 10 * sigma) + 2
    total = expect = 0.0
    for x in range(lo, hi):
        p = math.exp(
            x * math.log(lam) - lam - math.lgamma(x + 1)
        )  # the Poisson mass in log space: no underflow at a large mean
        total += p
        expect += p * math.ceil(x / tile)
    return expect / total if total > 0 else float(math.ceil(lam / tile))


def tiles_term(cfg: Knobs, rows: int) -> float:
    """The load's spill factor of a swapped record: the expected token tiles per active expert over the average's own tile count
    (:func:`expected_token_tiles`); 1 for the grouped records, whose cluster tile along M the pair term already prices."""
    if not cfg.swap_ab:
        return 1.0
    return expected_token_tiles(rows, cfg.tile_n) / max(1, math.ceil(rows / cfg.tile_n))


def exposed_epilogue_cycles(cfg: Knobs, rows: int, nodes: int) -> float:
    """The chain work of a CTA's last tile, which no mainloop overlaps (the earlier tiles' epilogues run behind the next tile's
    mainloop through the accumulator stages): a thread's chains times the chain's node cycles. The grouped body computes a
    row's outputs per thread, the tile's columns split between its epilogue warpgroups (the gated chain pairs two accumulator
    columns per output); the swapped body computes a weight row's live token columns per thread, ``min(rows, tile)`` of them
    (the chain of a dead column is skipped). At a few rows per expert this term separates the bodies: the grouped body's
    epilogue is the tile's width wide whatever the live rows."""
    if cfg.swap_ab:
        chains = min(rows, cfg.tile_n)
    else:
        chains = cfg.tile_n / _EPI_WARPGROUPS
    return chains * nodes * CHAIN_NODE_CYCLES


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
    chain: Union[str, "GroupedChain"],
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked records for one GEMM over both record spaces: the grouped records the grouped body accepts for the weight
    shape and the chain (at a decode-sized ``rows`` the dense dispatcher's structural guard, `heuristic.decode_sane`: 1-CTA
    tiles in an N-cluster, selects the candidates) and the swapped records the swapped body accepts (``swap``: None ranks them
    with the rest, True keeps only them, False excludes them), ordered by the dense class's estimate at ``(M = rows, N =
    n_active * N, K)`` (``n_active`` the active experts, ``E`` by default; a swapped record is priced as the dense model
    prices a swapped decode record, the weights on the MMA's M side and the tokens on its N side) times the pair term (the
    expected dummy half of the two-tile clusters along M) and the swapped records' spill factor (`tiles_term`: the token
    tiles an expert needs under a uniformly routed load), plus the last tile's exposed epilogue chain
    (`exposed_epilogue_cycles`: the grouped body's epilogue is the tile's width wide whatever the live rows, the swapped
    body's the live token columns), each tile family priced once (at the enumeration's raster: the dense estimate's
    tile-order term is the dense scheduler's, not the grouped one's) and the raster the structural rule gives the grouped
    scheduler (`raster_for`) first among its twins, best first, ties in the spaces' enumeration order. ``chain`` is a legacy
    name or the GEMM's :class:`.blocks.epilogue.GroupedChain` (its term prices the epilogue). As in the dense
    ``rank_topk``, each of the best ceil(k/2) records is followed by its raster twin, then the rank order fills the list:
    distinct records, at most ``k``, never empty for an implementable shape."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import decode_sane, estimate_cycles
    from .blocks.compile import can_implement_grouped, can_implement_swapped

    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    dense = (
        []
        if swap is True
        else [c for c in grouped_knobs() if can_implement_grouped(c, N, K, E, chain)]
    )
    swapped = (
        []
        if swap is False
        else [c for c in swapped_knobs() if can_implement_swapped(c, N, K, E, chain)]
    )
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    nodes = chain_nodes(chain if not isinstance(chain, str) else chain)

    def key(i, c):
        canon = dataclasses.replace(c, raster_along="m")
        est = estimate_cycles(
            canon, rows, dense_n, K, 1, "k", "bf16", fam, sm_count, nominal=nominal
        )
        # the grouped launch's own terms on the dense estimate: the cluster tile's dummy share (pair term), the swapped records'
        # spill over their token tile under the load, and the last tile's exposed epilogue chain
        est = est * pair_term(c, rows) * tiles_term(c, rows) + exposed_epilogue_cycles(
            c, rows, nodes
        )
        # three significant digits: records the estimate cannot separate (at the HBM floor every token tile reads the same
        # weights) fall through to the structural raster and the enumeration order -- the grouped records before the swapped
        # ones, the narrower token tile before the wider
        return (
            float(f"{est:.3g}"),
            0 if (c.swap_ab or c.raster_along == raster_for(c, rows)) else 1,
            i,
        )

    ranked = [
        c for _k, c in sorted((key(i, c), c) for i, c in enumerate(guarded + swapped))
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
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I for the ungated class), config for GEMM2 (N = H, K = I))``: the first
    record of each ranked list over both record spaces (``swap``: None, or True / False to keep only / exclude the
    swapped records)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1, chain1 = (2 * I, "swiglu") if gated else (I, "relu2")
    c1 = rank_grouped(rows, n1, H, E, chain1, sm_count, family, 1, nominal, n_act, swap)
    c2 = rank_grouped(rows, H, I, E, "plain", sm_count, family, 1, nominal, n_act, swap)
    if not c1 or not c2:
        raise ValueError(
            f"no grouped record implements (H={H}, I={I}, E={E}, gated={gated})"
        )
    return c1[0], c2[0]
