# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The fp8 grouped GEMMs' dispatcher: the dense fp8 class's ranking heuristic at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`): a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model at the dense shape whose tiles, M tail and operand bytes are the grouped
launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by side), for the fp8 format
(``in_fmt="e4m3"``: the format's MMA rate and operand bytes), mapped onto the grouped space and filtered by the grouped fp8
body's acceptance. Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import Knobs
from .blocks.fp8_epilogue import GEMM2_CHAIN, Fp8Chain
from .schedule import (
    record_cost,
    TOP_K,
    active_experts,
    grouped_knobs,
    raster_for,
    rows_per_expert,
)

IN_FMT = "e4m3"


def gemm1_chain(act: str = "silu", params=(), gated: bool = True) -> Fp8Chain:
    """GEMM1 of the W8A8 class: the gated or the ungated activation's fused quantization."""
    return Fp8Chain("glu" if gated else "act", act, "e4m3", tuple(params), True)


def all_valid_configs_fp8(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN, clusters=None
) -> List[Knobs]:
    """The configuration space the grouped fp8 guarantee is claimed over, for a weight shape and a chain."""
    from .blocks.fp8_compile import can_implement_grouped_fp8

    return [
        k
        for k in grouped_knobs(clusters)
        if can_implement_grouped_fp8(k, N, K, E, chain)
    ]


def swapped_knobs_fp8() -> List[Knobs]:
    """The swapped record space of the fp8 classes (:mod:`.blocks.fp8_swap_kernel`): the swapped 1-CTA records at 128 or 64
    weight rows x 16 / 32 tokens (``kind::f8f6f4`` takes N >= 16), no cluster, swizzle 1. Disjoint from the grouped space (the
    swapped flag); the body reads the same dense rows through a TMA box."""
    from .blocks.fp8_swap_kernel import swapped_knobs

    return swapped_knobs()


def all_valid_configs_swapped_fp8(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN
) -> List[Knobs]:
    """The swapped records the fp8 swapped body accepts for a weight shape and a chain."""
    from .blocks.fp8_compile import can_implement_swapped_fp8

    return [
        k for k in swapped_knobs_fp8() if can_implement_swapped_fp8(k, N, K, E, chain)
    ]


def all_valid_configs_fp8_both(
    N: int, K: int, E: int, chain: Fp8Chain = GEMM2_CHAIN, clusters=None
) -> List[Knobs]:
    """Both record spaces of the fp8 classes for a weight shape and a chain: the grouped records, then the swapped ones."""
    return all_valid_configs_fp8(
        N, K, E, chain, clusters
    ) + all_valid_configs_swapped_fp8(N, K, E, chain)


def chain_nodes_fp8(chain: Fp8Chain) -> int:
    """The node count of a W8A8 chain's term (the S-expression :mod:`..spec_fp8` builds for one output element: GEMM1's fused
    e4m3 quantization of the gated or the ungated activation on the alpha-scaled accumulators, GEMM2's scaled conversion): the
    exposed-epilogue term's cost unit (:func:`.schedule.exposed_epilogue_cycles`)."""
    from . import mterm as term
    from . import spec_fp8 as M8

    alg = term.TermAlgebra()
    alpha, inv = alg.sym("alpha"), alg.sym("inv")
    if chain.pair:
        t = M8.glu_quant_e4m3_elem(
            alg,
            alg.leaf("g", 0, 0),
            alg.leaf("u", 0, 0),
            alpha,
            inv,
            chain.act,
            chain.params,
        )
    elif chain.kind == "act":
        t = M8.act_quant_e4m3_elem(
            alg, alg.leaf("x", 0), alpha, inv, chain.act, chain.params
        )
    elif chain.scaled:
        t = M8.scaled_out_elem(alg, alg.leaf("x", 0), alpha, chain.out_fmt)
    else:
        return 1
    return term.sexpr(t).count("(")


def rank_grouped_fp8(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Fp8Chain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """The ranked records for one fp8 GEMM over both record spaces, as the bf16 backend's ``schedule.rank_grouped``: the grouped
    records the fp8 body accepts for the weight shape and the chain (at a decode-sized ``rows`` the dense dispatcher's structural
    guard selects the candidates) and the swapped records the fp8 swapped body accepts (``swap``: None ranks them with the rest,
    True keeps only them, False excludes them), ordered by the dense fp8 class's estimate at ``(M = rows, N = n_active * N, K)``
    (the fp8 instruction rate and operand bytes; the e4m3 output of GEMM1's chain is one byte) times the pair term and the swapped
    records' spill factor, plus the last tile's exposed epilogue chain (the W8A8 chain's node count) and the token box's zero-fill
    at the format's element width, rounded to three significant digits; each tile family priced once at the enumeration's raster
    and the grouped scheduler's structural raster first among its twins; each of the best ceil(k/2) records is followed by its
    raster twin, then the rank order fills the list: distinct records, at most ``k``, never empty for an implementable shape.
    Numerics-neutral: every record of either space computes the class."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import decode_sane
    from .blocks.fp8_compile import can_implement_grouped_fp8, can_implement_swapped_fp8

    fam = family or "sm_107"
    dense = (
        []
        if swap is True
        else [
            c for c in grouped_knobs() if can_implement_grouped_fp8(c, N, K, E, chain)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_fp8()
            if can_implement_swapped_fp8(c, N, K, E, chain)
        ]
    )
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    nodes = chain_nodes_fp8(chain)

    def key(i, c):
        est = record_cost(
            c,
            rows,
            n_active or E,
            N,
            K,
            nodes,
            chain.pair,
            fam,
            sm_count,
            IN_FMT,
            nominal,
        )
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


def default_configs_fp8(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    gated: bool = True,
    nominal: bool = False,
    chain1: Optional[Fp8Chain] = None,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I ungated), config for GEMM2 (N = H, K = I))``: the first record of each ranked
    list over both record spaces (``swap``: None, or True / False to keep only / exclude the swapped records). ``chain1`` names
    GEMM1's chain (the activation's; its acceptance is the chain kind's)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    c1chain = (
        chain1
        if chain1 is not None
        else gemm1_chain("silu" if gated else "relu2", (), gated)
    )
    n1 = 2 * I if gated else I
    c1 = rank_grouped_fp8(
        rows, n1, H, E, c1chain, sm_count, family, 1, nominal, n_act, swap
    )
    c2 = rank_grouped_fp8(
        rows, H, I, E, GEMM2_CHAIN, sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped fp8 record implements (H={H}, I={I}, E={E}, gated={gated})"
        )
    return c1[0], c2[0]
