# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The block-scale fp8 grouped GEMMs' dispatcher: the dense block-scale class's ranking heuristic at the grouped launch's shape,
and the placement rule of the intermediate's requantization.

As for the bf16 classes (:mod:`.schedule`): a grouped GEMM is the dense class under a different scheduler, so its records are
ranked by the dense dispatcher's model at the dense shape whose tiles, M tail and operand bytes are the grouped launch's
(``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by side), for the block-scale format
(``in_fmt="e4m3_bs"``: the fp8 instruction rate, the promotion hand-off chain, no 2-CTA issue gain for the lockstep pair),
with the dense dispatcher's decode guard and its few-token N-cluster rule, mapped onto the grouped space and filtered by the
grouped body's acceptance. The requantization's two spellings (GEMM1's epilogue on the whole-block tiles, or the standalone
pass on the bf16 intermediate) are placements of one term: ``fused_quant_default`` picks the epilogue for the chain whose block
is formed inside one promotion warpgroup (the ungated chain) from a row count per active expert, the standalone pass otherwise.
Numerics-neutral: every record of the space computes the class.
"""

from __future__ import annotations

import functools

from typing import List, Optional, Sequence, Tuple

from ...gemm.batch_invariant.generator import CLUSTERS, Knobs, knob_product
from ...gemm.batch_invariant.heuristic import on_rank_cache_clear
from .blocks.epilogue_fp8bs import (
    GEMM2_CHAIN,
    OUT_FMT_QUANT,
    TILES_N_GROUPED,
    Fp8BsChain,
    gemm1_chain,
    quant_twin,
)
from .spec_fp8bs import BLOCK as QBLOCK
from .spec_fp8bs import STAGE_K, check_block
from .schedule import (
    TOP_K,
    active_experts,
    chain_nodes,
    raster_for,
    record_cost,
    rows_per_expert,
)

IN_FMT = "e4m3_bs"


def swapped_knobs_fp8bs() -> List[Knobs]:
    """The swapped decode space of the block-scale classes (:mod:`.blocks.swap` at the format: one record per (weight tile 64 /
    128, token tile 8 / 16 / 32 / 64))."""
    from .blocks.swap import swapped_knobs

    return swapped_knobs(IN_FMT)


def can_implement_grouped_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """Acceptance of the grouped body for the block-scale classes (`blocks.compile.can_implement_grouped` at the chain taken at
    ``block`` and ``interleaved``): a record of the dense block-scale class's space at swizzle 1 with a tile of two promotion
    warpgroups (N 64 / 128 / 256), K in whole smem stages, the weight's rows whole scale blocks (the interleaved weight: of
    both halves), the fused requantization only for the 128-wide class where the tile's outputs are whole 128-column blocks."""
    from .blocks.compile import can_implement_grouped

    if not isinstance(chain, Fp8BsChain):
        return False
    if chain.quant and check_block(block) != QBLOCK:
        return False  # the fused requantization is the 128-wide class's
    return can_implement_grouped(knobs, N, K, E, chain.at(block, interleaved))


def can_implement_swapped_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """Acceptance of the swapped body for the block-scale classes (`blocks.compile.can_implement_swapped` at the chain taken at
    ``block`` and ``interleaved``): a swapped record of the class (a 64- / 128-row weight tile, an 8 / 16 / 32 / 64 token tile,
    1-CTA, no cluster), the bf16 / f32 chains (the fused requantization is not a spelling of this body), the operand rule."""
    from .blocks.compile import can_implement_swapped

    if not isinstance(chain, Fp8BsChain) or chain.quant:
        return False  # the swapped body runs the bf16 / f32 chains
    return can_implement_swapped(
        knobs, N, K, E, chain.at(check_block(block), interleaved)
    )


def can_implement_fp8bs(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> bool:
    """A record either body takes for the block-scale classes: the grouped body (the dense-derived records) or the swapped body
    (the swapped records); the two spaces are disjoint."""
    return can_implement_grouped_fp8bs(
        knobs, N, K, E, chain, interleaved, block
    ) or can_implement_swapped_fp8bs(knobs, N, K, E, chain, interleaved, block)


#: The fused requantization (GEMM1's output blocks quantized in its epilogue, the 256-wide tile) from this many rows per active
#: expert; below it the standalone pass and GEMM1's free tile choice (empirical; a placement).
FUSED_QUANT_MIN_ROWS = 128


def grouped_knobs_fp8bs(clusters=None) -> List[Knobs]:
    """The grouped block-scale space: the dense block-scale class's records (128-row CTA tiles, the six clusters, both rasters)
    at swizzle 1 with a tile of two promotion warpgroups (N 64 / 128 / 256), or the subset with the given clusters."""
    return [
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt=IN_FMT,
        )
        if k.tile_n in TILES_N_GROUPED
    ]


def all_valid_configs_fp8bs(
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain = GEMM2_CHAIN,
    clusters=None,
    interleaved: Optional[bool] = None,
    block: int = QBLOCK,
) -> List[Knobs]:
    """The configuration space the grouped block-scale guarantee is claimed over, for a weight shape, a chain and a block (the
    fused requantization's chain admits the 128-wide class's tiles whose outputs are whole 128-column blocks)."""
    return [
        k
        for k in grouped_knobs_fp8bs(clusters)
        if can_implement_grouped_fp8bs(k, N, K, E, chain, interleaved, block)
    ]


def all_valid_configs_fp8bs_swapped(
    N: int, K: int, E: int, chain: Fp8BsChain = GEMM2_CHAIN, block: int = QBLOCK
) -> List[Knobs]:
    """The swapped decode records the swapped body accepts for a weight shape, a chain and a block (the bf16 / f32 chains)."""
    return [
        k
        for k in swapped_knobs_fp8bs()
        if can_implement_swapped_fp8bs(k, N, K, E, chain, None, block)
    ]


def chain_nodes_fp8bs(chain: Fp8BsChain) -> int:
    """The node count of a block-scale GEMM1 / GEMM2 chain's term: the bf16 chain's (the fused requantization: its bf16 twin's plus
    the quantizer's per-element multiply and conversion)."""
    if chain.out_fmt == OUT_FMT_QUANT:
        return chain_nodes(quant_twin(chain)) + 2
    return chain_nodes(chain)


def promotion_overrides_fp8bs(c: Knobs, block: int = QBLOCK) -> dict:
    """The ranking's promotion terms for one record of the class: blocks per smem stage (the 32-wide class promotes every
    instruction step) and the swapped body's one promotion warpgroup (half the padded body's two), layered on the dense model's
    constants (``schedule.record_cost`` overrides; the offline grader `bench/moe_rank_eval.py` prices records with the same)."""
    from ...gemm.batch_invariant.heuristic import CONSTANTS

    spb = STAGE_K // check_block(block)
    groups = 1.0 if c.swap_ab else 2.0
    return dict(
        promote_handoff_clk=CONSTANTS["promote_handoff_clk"] * spb,
        promote_pair_clk=CONSTANTS["promote_pair_clk"] * spb,
        promote_bytes_per_clk=CONSTANTS["promote_bytes_per_clk"] * groups / 2.0 / spb,
    )


def rank_grouped_fp8bs(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
    block: int = QBLOCK,
) -> List[Knobs]:
    """:func:`_rank_grouped_fp8bs` memoised on its arguments (all hashable scalars; the chain a frozen dataclass): the runner
    ranks at every forward without a tactic, once per token count."""
    return list(
        _rank_grouped_fp8bs(
            rows, N, K, E, chain, sm_count, family, k, nominal, n_active, swap, block
        )
    )


@functools.lru_cache(maxsize=16384)
def _rank_grouped_fp8bs(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: Fp8BsChain,
    sm_count: Optional[int],
    family: Optional[str],
    k: int,
    nominal: bool,
    n_active: Optional[int],
    swap: Optional[bool],
    block: int,
) -> Tuple[Knobs, ...]:
    """The ranked records for one block-scale GEMM over both record spaces, as the bf16 backend's ``schedule.rank_grouped``: the
    grouped records the padded body accepts (at a decode-sized ``rows`` the dense dispatcher's structural guard and its few-token
    N-cluster rule select the candidates) and the swapped decode records the swapped body accepts (``swap``: None ranks them with
    the rest, True keeps only them, False excludes them), ordered by the dense block-scale class's estimate at ``(M = rows, N =
    n_active * N, K)`` under the shared ranking contract (`schedule.record_cost`: the dense estimate with the MoE ranking's
    constant overrides, a swapped record priced in the swapped orientation, times the pair term and the swapped records' spill
    factor, plus the exposed epilogue chain on this chain's term, the token box's exposed zero-fill at the e4m3 element width and
    the body's per-launch fixed cost), to three significant digits; each tile family priced once at the
    enumeration's raster and the grouped scheduler's structural raster first among its twins; each of the best ceil(k/2) records
    followed by its raster twin, then the rank order fills the list: distinct records, at most ``k``, never empty for an
    implementable shape. Numerics-neutral: every record of either space computes the class. ``block``: the class -- the
    estimate's promotion term is charged per instruction step at the 32-wide class (four promotions per smem stage), and the
    swapped body's single promotion warpgroup at half the padded body's rate, through the dense model's constants."""
    import dataclasses

    from ...gemm.batch_invariant.heuristic import (
        PART_TABLE,
        TOKEN_BOX_ROWS,
        capacity_for,
        decode_sane,
    )

    check_block(block)
    fam = family or "sm_107"
    dense_n = (n_active or E) * N
    dense = (
        []
        if swap is True
        else [
            c
            for c in grouped_knobs_fp8bs()
            if can_implement_grouped_fp8bs(c, N, K, E, chain, None, block)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_fp8bs()
            if can_implement_swapped_fp8bs(c, N, K, E, chain, None, block)
        ]
    )

    def promotion_overrides(c: Knobs) -> dict:
        return promotion_overrides_fp8bs(c, block)

    space = dense + swapped
    if not space:
        return ()
    guarded = [c for c in dense if decode_sane(c, rows, False, True)] or dense
    if rows <= TOKEN_BOX_ROWS // 4:
        # the dense block-scale dispatcher's few-token rule: the 128-row token box is at least three quarters zero-fill, a TMA
        # cost per block per CTA that the N-cluster's multicast issues once per cluster; the widest N-cluster shares it most
        # while its clusters still fit the part in one wave (every active expert's tiles count)
        cap4 = capacity_for(fam, sm_count, nominal).get(4) or max(
            1, (sm_count or PART_TABLE[fam].sm_count) // 4
        )
        wide = [
            c
            for c in guarded
            if c.cluster_n == 4
            and ((dense_n + c.tile_n - 1) // c.tile_n + 3) // 4 <= cap4
        ]
        guarded = wide or guarded
    nodes = chain_nodes_fp8bs(chain)

    def key(i, c):
        # the shared contract's cost with the class's promotion terms layered on the ranking's overrides
        est = record_cost(
            c,
            rows,
            n_active or E,
            N,
            K,
            nodes,
            False,
            fam,
            sm_count,
            IN_FMT,
            nominal,
            overrides=promotion_overrides(c),
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
    return tuple(out[:k])


on_rank_cache_clear(
    _rank_grouped_fp8bs.cache_clear
)  # the memo follows the dispatcher's rankings (a device's registered numbers)


def fused_quant_default(
    T: int, k: int, E: int, block: int = QBLOCK, gated: bool = True
) -> bool:
    """Whether the block-scale op requantizes GEMM1's output in the epilogue (the wide-tile records) or by the standalone quantizer
    pass at this (T, k, E) by default, both spellings the same bits: the epilogue for the ungated chain, whose 128-output block
    is formed inside one promotion warpgroup, from a row count per active expert; the standalone pass for the gated chain, whose
    block spans the two warpgroups' halves and whose epilogue joins them once per tile (the fused spelling stays a tactic). The
    32-wide class requantizes by the standalone pass only (the fused epilogue is the 128-wide class's)."""
    return (
        block == QBLOCK
        and not gated
        and rows_per_expert(T, k, E) >= FUSED_QUANT_MIN_ROWS
    )


def default_configs_fp8bs(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    act: str = "silu",
    gated: bool = True,
    params: Sequence[Optional[float]] = (),
    fused: bool = False,
    nominal: bool = False,
    swap: Optional[bool] = None,
    block: int = QBLOCK,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I, K = H; N = I ungated), config for GEMM2 (N = H, K = I))`` of a block-scale class (``block``
    128 / 32): the first record of each ranked list over the grouped and the swapped records (``swap`` None / True / False ranks /
    keeps / excludes the swapped records); ``fused`` ranks GEMM1 over the fused requantization's records (which the swapped
    body does not serve and the 32-wide class does not take)."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if gated else I
    c1 = rank_grouped_fp8bs(
        rows,
        n1,
        H,
        E,
        gemm1_chain(act, params, gated, fused),
        sm_count,
        family,
        1,
        nominal,
        n_act,
        swap,
        block,
    )
    c2 = rank_grouped_fp8bs(
        rows, H, I, E, GEMM2_CHAIN, sm_count, family, 1, nominal, n_act, swap, block
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped block-scale record implements (H={H}, I={I}, E={E}, gated={gated}, fused={fused}, block={block})"
        )
    return c1[0], c2[0]


__all__ = [
    "FUSED_QUANT_MIN_ROWS",
    "IN_FMT",
    "all_valid_configs_fp8bs",
    "all_valid_configs_fp8bs_swapped",
    "default_configs_fp8bs",
    "fused_quant_default",
    "gemm1_chain",
    "grouped_knobs_fp8bs",
    "promotion_overrides_fp8bs",
    "rank_grouped_fp8bs",
    "chain_nodes_fp8bs",
    "swapped_knobs_fp8bs",
]
