# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The NVFP4 MoE classes' grouped-GEMM dispatcher: the dense 4-bit GEMM class's ranking heuristic at the grouped launch's
shape, as the bf16 classes' (:mod:`.schedule`, whose terms this module reuses).

The grouped block-scaled GEMM is the dense NVFP4 class under a different scheduler (one segment of rows per expert, whole
128-row scale-factor atoms per segment), so its records are ranked by the dense dispatcher's model
(:func:`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` at the instruction class ``in_fmt``: the part's roofline
with nominal constants, no measured tables) at the dense shape whose tiles, M tail and operand bytes are the grouped
launch's: ``M`` = the rows of one active expert, ``N`` = the active experts' weight rows side by side (every group's tiles
count, and the weights are read once per group). Two record spaces are ranked together, as the bf16 classes' dispatcher
ranks them: the grouped records (the dense records at swizzle 1: no in-cluster split, no operand swap, no B-reuse tile, no
in-CTA chunk fold) and the swapped records (:mod:`.blocks.swap`: the weights on the MMA's M side, 128 rows per CTA, the
group's tokens a narrow N tile of 8 / 16 / 32 rows), which the dense model prices as a swapped decode record. The grouped
launch's own terms on the dense estimate are the bf16 dispatcher's: the pair term (the cluster tile's dummy share), the
swapped records' spill factor (the Poisson expectation of token tiles per active expert), the last tile's exposed epilogue
chain (this class's chain term: the W4A4 gated chain, or the static class's chain plus its share of the block quantization;
four epilogue warpgroups) and the token box's zero-fill past the group's rows at the packed fp4 element width, rounded to three
significant digits so records the estimate cannot separate fall through to the structural raster and the enumeration order.
Every record of either space is filtered by its body's acceptance. Numerics-neutral: every record of either space is
canonical (the same bits), so the dispatcher may depend on anything; the ranked list is what the autotuner profiles under
``batch_invariant``.
"""

from __future__ import annotations

import dataclasses
import functools
from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_M_BLOCKSCALED,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from .blocks.epilogue import GroupedEpilogueBlock, W4A4Chain
from .blocks.swap import swapped_knobs
from .schedule import (  # noqa: F401  (re-exported)
    TOP_K,
    active_experts,
    exposed_epilogue_cycles,
    pair_term,
    raster_for,
    rows_per_expert,
    tiles_term,
    zero_fill_cycles,
)

#: Epilogue warpgroups of the grouped block-scaled body (:data:`.blocks.operands.EPI_WARPGROUPS_BLOCKSCALED`): the exposed
#: epilogue term splits a tile's chains between them.

GROUP_ALIGN = (
    128  # the padded group alignment the dispatcher ranks at (one scale-factor atom)
)


def grouped_knobs_blockscaled(clusters=None) -> List[Knobs]:
    """The grouped block-scaled space: the dense records of the nvfp4 knob product at swizzle 1 (no in-cluster split, no
    operand swap, no B-reuse tile, no in-CTA chunk fold: the group scheduler has none of them), the same records for both
    instruction classes (the step is the class)."""
    return list(
        _grouped_knobs_blockscaled(None if clusters is None else tuple(clusters))
    )


@functools.lru_cache(maxsize=None)
def _grouped_knobs_blockscaled(clusters) -> Tuple[Knobs, ...]:
    return tuple(
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt="nvfp4",
        )
        if k.cluster_k == 1 and not k.swap_ab and not k.b_reuse and not k.cta_chunks
    )


def swapped_knobs_blockscaled() -> List[Knobs]:
    """The swapped decode space of the grouped block-scaled classes (:mod:`.blocks.swap`): the dense class's swapped 1-CTA
    records at the narrow token tiles (128 weight rows x 8 / 16 / 32 tokens), one per tile, no cluster, swizzle 1. Disjoint from
    the grouped space; the body reads the dense packing."""
    return swapped_knobs("nvfp4")


#: Records the fp4 runners hand the autotuner per GEMM: the ranked list's head (its raster twins included).
TOP_K_FP4 = 6
#: The average rows per active expert fill at most three quarters of the swapped token tile (an empirical margin).
SWAP_TILE_FILL = (3, 4)
#: The narrowest swapped tile: the structural rule's pick (the bench's reference point; the ranking decides the dispatch).
SWAP_TILES_DEFAULT = (8,)
#: The gather form's token tile: the swapped record whose one-warp scale producer keeps a stage's row copies in flight; the
#: wider tiles serialize them (the campaign's lever record).
GATHER_TILE_N = 8


def gather_form_rule(config1: Knobs, scatter_launch: bool) -> Knobs:
    """The fused ops' default form of GEMM1's record: the gather form (``gather_a``) of the swapped record at the
    ``GATHER_TILE_N`` token tile when the box form would scatter the quantized rows in a launch of its own
    (``scatter_launch``: a pre-quantized input, the MXFP4 pair), the record as ranked otherwise (a bf16 NVFP4 input: its
    quantizer scatters in its own pass; the grouped records; the wider swapped tiles)."""
    if scatter_launch and config1.swap_ab and config1.tile_n == GATHER_TILE_N:
        return dataclasses.replace(config1, gather_a=True)
    return config1


def token_tile_default(rows: int) -> Optional[int]:
    """The structural rule of the first campaign (kept as the benches' reference point, not the dispatch): the narrowest swapped
    tile of ``SWAP_TILES_DEFAULT`` the average rows per active expert fill to at most ``SWAP_TILE_FILL``, ``None`` otherwise. The
    dispatch ranks the swapped records with the grouped ones (:func:`rank_grouped_nvfp4`)."""
    num, den = SWAP_TILE_FILL
    for tn in SWAP_TILES_DEFAULT:
        if rows * den <= tn * num:
            return tn
    return None


def chain_nodes_nvfp4(chain) -> int:
    """The node count of the chain's term per output element (the exposed epilogue's cost unit): the W4A4 gated / ungated chain on the
    row-scaled accumulators (``glu_scaled_elem`` / ``act_scaled_elem``), GEMM2's scaled conversion, or the static class's GEMM1 -- the
    fp32 gated value plus its share of the block's quantization (the block's scale and codes over its 16 outputs); the MXFP4
    classes' chains by :func:`.schedule_mxfp4.chain_nodes_mx`."""
    from . import mterm as term
    from . import spec_nvfp4 as Q
    from .blocks.epilogue import Mxfp4Chain

    if isinstance(chain, Mxfp4Chain):
        from .schedule_mxfp4 import chain_nodes_mx

        return chain_nodes_mx(chain)
    alg = term.TermAlgebra()
    s = Q.w4a4_row_scale(alg, alg.leaf("alpha", 0), alg.leaf("pts"))
    g, u = alg.leaf("g", 0, 0), alg.leaf("u", 0, 0)
    if chain.static_quant:
        # the block's DAG (the fp32 values, the amax tree, the scale, the codes) sized once and split over its outputs
        gs = alg.leaf("gs2", 0)
        if chain.pair:
            vals = [
                Q.glu_scaled_pre(
                    alg,
                    alg.leaf("g", 0, j),
                    alg.leaf("u", 0, j),
                    s,
                    chain.act,
                    chain.params,
                )
                for j in range(Q.NVFP4_BLOCK)
            ]
        else:
            vals = [
                Q.act_scaled_pre(alg, alg.leaf("u", 0, j), s, chain.act, chain.params)
                for j in range(Q.NVFP4_BLOCK)
            ]
        sf_d, codes = Q.static_block(alg, vals, gs)
        return -(-_dag_size([sf_d, *codes]) // Q.NVFP4_BLOCK)
    if chain.pair:
        t = Q.glu_scaled_elem(alg, g, u, s, chain.act, chain.out_fmt, chain.params)
    elif chain.kind == "act":
        t = Q.act_scaled_elem(alg, u, s, chain.act, chain.out_fmt, chain.params)
    else:
        t = Q.scaled_out_elem(alg, u, s, chain.out_fmt)
    return term.size(t)


def _dag_size(roots) -> int:
    """The distinct nodes reachable from several roots (``mterm.size`` over one): a block's terms share their sub-terms."""
    seen: set = set()
    stack = list(roots)
    while stack:
        c = stack.pop()
        if c.uid in seen:
            continue
        seen.add(c.uid)
        stack.extend(a for a in c.args if hasattr(a, "uid"))
    return len(seen)


def _grouped(cfg: Knobs) -> Knobs:
    """A dense block-scaled record as a grouped one: swizzle 1, no grid cap, the tile clamped into the grouped space (a
    swapped decode record's narrow token tile -> the narrowest grouped N tile; a B-reuse tile -> the 2-CTA tile; an
    in-cluster split or an in-CTA chunk fold -> the plain record of the same tile and cluster)."""
    tm = (
        cfg.tile_m
        if cfg.tile_m in TILES_M_BLOCKSCALED[cfg.use_2cta_instrs]
        else TILES_M_BLOCKSCALED[cfg.use_2cta_instrs][0]
    )
    tn = cfg.tile_n if cfg.tile_n in TILES_N_BLOCKSCALED else TILES_N_BLOCKSCALED[0]
    return Knobs(
        (tm, tn), cfg.cluster_shape_mn, cfg.use_2cta_instrs, True, cfg.raster_along, 1
    )


def all_valid_configs_nvfp4(
    N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4", clusters=None
) -> List[Knobs]:
    """The configuration space the grouped guarantee is claimed over, for an expert weight shape, a chain and an
    instruction class."""
    from .blocks.compile import (
        can_implement_grouped_blockscaled as can_implement_grouped_nvfp4,
    )

    return [
        k
        for k in grouped_knobs_blockscaled(clusters)
        if can_implement_grouped_nvfp4(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_nvfp4_swapped(
    N: int, K: int, E: int, chain: W4A4Chain, in_fmt: str = "nvfp4"
) -> List[Knobs]:
    """The swapped decode records the swapped body accepts for an expert weight shape, a chain and an instruction class (the
    dense packing); the guarantee is claimed over this space and the grouped space together."""
    from .blocks.compile import (
        can_implement_swapped_blockscaled as can_implement_swapped_nvfp4,
    )

    return [
        k
        for k in swapped_knobs_blockscaled()
        if can_implement_swapped_nvfp4(k, N, K, E, chain, in_fmt)
    ]


def padded_rows(rows: int) -> int:
    """The rows per active expert rounded up to whole scale-factor atoms (the grouped kernel's tile fit)."""
    return -(-max(1, rows) // GROUP_ALIGN) * GROUP_ALIGN


def gather_twins(swapped: List[Knobs]) -> List[Knobs]:
    """The gather-form twins of the swapped records (:class:`.blocks.swap.SwappedGroupedGemm`'s gather record): the same tile,
    the token rows and their scale factors read by the kernel through the plan's map instead of from the permuted rows (GEMM1
    only: GEMM2's rows are the class's own intermediates, already dense)."""
    return [dataclasses.replace(c, gather_a=True) for c in swapped if c.swap_ab]


def rank_grouped_nvfp4(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
    gather: Optional[bool] = False,
) -> List[Knobs]:
    """The ranked records (:func:`_rank_grouped_nvfp4`), memoised on the arguments: the ranking runs on the host at every call
    of the op and of the layer, and enumerating and costing the space is milliseconds of Python per GEMM. ``gather``: the
    swapped records' gather form (GEMM1) -- False excludes it (GEMM2, the default), True replaces the swapped records by their
    gather twins, None ranks both forms (the twin right after its record: the same cost, the form a placement)."""
    return list(
        _rank_grouped_nvfp4(
            rows,
            N,
            K,
            E,
            chain,
            in_fmt,
            sm_count,
            family,
            k,
            nominal,
            n_active,
            swap,
            gather,
        )
    )


@functools.lru_cache(maxsize=4096)
def _rank_grouped_nvfp4(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str,
    sm_count: Optional[int],
    family: Optional[str],
    k: int,
    nominal: bool,
    n_active: Optional[int],
    swap: Optional[bool],
    gather: Optional[bool] = False,
) -> Tuple[Knobs, ...]:
    """The ranked records for one GEMM over both record spaces (the bf16 dispatcher's form, :func:`.schedule.rank_grouped`): the
    grouped records the grouped body accepts for the weight shape, the chain and the instruction class (at a decode-sized ``rows``
    the dense dispatcher's structural guard, `heuristic.decode_sane`, selects the candidates; a record whose M tile is taller than
    the group's padded rows is excluded: the group scheduler's tiles never span groups) and the swapped records the swapped body
    accepts (``swap``: None ranks them with the rest, True keeps only them, False excludes them), ordered by the shared contract's
    cost (:func:`.schedule.record_cost` at this class's instruction class, with this class's chain term in nodes: the dense estimate
    under the ranking's overrides times the pair and token-tile terms, the exposed epilogue chain, the token box's exposed zero fill
    at the packed fp4 width and the body's per-launch fixed cost), rounded to three significant digits; the grouped scheduler's
    structural raster first among twins, then the wider N tile among tied grouped records (fewer tiles: fewer per-tile costs),
    ties in the spaces' enumeration order (the grouped records before the swapped ones, the narrower token tile first). Each of the best ceil(k/2) records is followed by its raster twin, then the rank order fills the
    list: distinct records, at most ``k``, never empty for an implementable shape."""
    from ...gemm.batch_invariant.heuristic import decode_sane
    from .blocks.compile import (
        can_implement_grouped_blockscaled as can_implement_grouped_nvfp4,
    )
    from .blocks.compile import (
        can_implement_swapped_blockscaled as can_implement_swapped_nvfp4,
    )
    from .schedule import record_cost

    fam = family or "sm_107"
    m_fit = padded_rows(rows)
    dense = (
        []
        if swap is True
        else [
            c
            for c in grouped_knobs_blockscaled()
            if c.tile_m <= m_fit
            and can_implement_grouped_nvfp4(c, N, K, E, chain, in_fmt)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_blockscaled()
            if can_implement_swapped_nvfp4(c, N, K, E, chain, in_fmt)
        ]
    )
    if gather is not False:
        twins = [
            c
            for c in gather_twins(swapped)
            if can_implement_swapped_nvfp4(c, N, K, E, chain, in_fmt)
        ]
        swapped = twins if gather else swapped + twins
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    nodes = chain_nodes_nvfp4(chain)

    def key(i, c):
        # the shared contract's cost (`schedule.record_cost`) at this class's instruction class and chain term (a gather twin
        # at its record's cost: the form moves the same bytes)
        est = record_cost(
            dataclasses.replace(c, gather_a=False),
            rows,
            n_active or E,
            N,
            K,
            nodes,
            chain.pair,
            fam,
            sm_count,
            in_fmt,
            nominal,
        )
        return (
            float(f"{est:.3g}"),
            0 if (c.swap_ab or c.raster_along == raster_for(c, rows)) else 1,
            0
            if c.swap_ab
            else -c.tile_n,  # tied grouped records: the wider N tile first (fewer tiles, fewer per-tile costs)
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
    # the autotuner's structural slot: the best-ranked 2-CTA record joins the list once the rows per expert fill at least half
    # its cluster tile (the pair's share of the stream is then real), whatever the model's order (its pair terms are the
    # coarsest)
    pair = next(
        (
            c
            for c in ranked
            if c.use_2cta_instrs and not c.swap_ab and 2 * rows >= c.mma_tiler_mn[0]
        ),
        None,
    )
    if pair is not None and pair not in seen and k > 1:
        out = out[: k - 1] + [pair] if len(out) >= k else out + [pair]
    return tuple(out[:k])


def default_configs_nvfp4(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    in_fmts: Tuple[str, str],
    chain1: W4A4Chain,
    chain2: W4A4Chain,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
    gather: Optional[bool] = False,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list
    (``swap``: the swapped decode body by the structural rule (None), forced (True) or forbidden (False); ``gather``: GEMM1's
    swapped records in their gather form -- excluded (False), forced (True) or ranked with the box form (None))."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    c1 = rank_grouped_nvfp4(
        rows,
        n1,
        H,
        E,
        chain1,
        in_fmts[0],
        sm_count,
        family,
        1,
        nominal,
        n_act,
        swap,
        gather,
    )
    c2 = rank_grouped_nvfp4(
        rows, H, I, E, chain2, in_fmts[1], sm_count, family, 1, nominal, n_act, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped block-scaled record implements (H={H}, I={I}, E={E}, {in_fmts})"
        )
    return c1[0], c2[0]
