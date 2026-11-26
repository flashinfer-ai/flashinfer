# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The MXFP8 grouped GEMMs' dispatcher: the dense MX class's estimate over both record spaces at the grouped launch's shape.

As for the bf16 classes (:mod:`.schedule`), a grouped GEMM is the dense class under a different scheduler, so its records
are ranked by the dense dispatcher's model (`flashinfer.gemm.batch_invariant.heuristic.estimate_cycles` for the MX class
of the part: the part's roofline with nominal constants, no measured tables) at the dense shape whose tiles, M tail and
operand bytes are the grouped launch's (``M`` = the rows of one active expert, ``N`` = the active experts' weight rows
side by side). Two record spaces are ranked together: the grouped MX space (the dense block-scaled records at swizzle 1
without in-cluster split, operand swap or B-reuse tile, filtered by the grouped block-scaled body's acceptance) and the
swapped records (:mod:`.blocks.swap`: the weights on the MMA's M side, 128 rows per CTA, the group's tokens a
narrow N tile of 8 / 16 / 32 rows on the dense packing), which the dense model prices as it prices a swapped decode
record, so the CTA count and the operand bytes per CTA of each body enter one estimate and the pick follows the shape,
never a rows-per-expert switch. The grouped launch's own terms on the estimate are the bf16 dispatcher's (`schedule`):
the raster rule and the pair term, the swapped records' spill over their token tile under the load, the last tile's
exposed epilogue chain (the MX requantization's term for the fused chain, four epilogue warpgroups) and the token box's
zero-fill at the e4m3 width. Numerics-neutral: every record of either space computes the class.
"""

from __future__ import annotations

import functools
from typing import List, Optional, Tuple

from ...gemm.batch_invariant.generator import (
    CLUSTERS,
    TILES_N_BLOCKSCALED,
    Knobs,
    knob_product,
)
from ...gemm.batch_invariant.heuristic import decode_sane
from ...gemm.batch_invariant.heuristic import on_rank_cache_clear
from .blocks.epilogue import GroupedEpilogueBlock
from .blocks.epilogue_mx import OUT_FMT, MxChain, bf16_twin
from .blocks.operands import EPI_WARPGROUPS_BLOCKSCALED
from .spec_mxfp4w8a import MX_CLASSES as MX_MIXED_CLASSES
from .spec_mxfp8 import MX_CLASSES
from .schedule import (
    TOP_K,
    active_experts,
    chain_nodes,
    record_cost,
    rows_per_expert,
)

#: GEMM2 of the MXFP8 classes: the dequantised accumulator converted to bf16 (the instruction class is the launch's).
GEMM2_CHAIN = MxChain("plain", "silu", "bf16")
#: The instruction classes the record spaces and their acceptance serve: the MXFP8 classes and the mixed MXFP4 x MXFP8 ones
#: (the mixed dispatcher, :mod:`.schedule_mxfp4w8a`, ranks with its own terms).
MX_SERVED = MX_CLASSES + MX_MIXED_CLASSES
#: K elements per smem stage the swapped body admits for the MXFP8 classes: the format's one 128-byte atom of K and two atoms.
SWAP_K_BLOCKS = (128, 256)


def swap_k_block(in_fmt: str, K: int) -> int:
    """The swapped body's k-block for a GEMM of reduction length ``K``: two atoms of K per stage when they divide ``K``, else
    the one-atom default -- a function of the fixed dims alone, so a placement of the same fold (the barrier round trips over
    K halve; the MMA sequence is the class's). The grouped body and the dense class keep the default."""
    kbs = [kb for kb in SWAP_K_BLOCKS if K % kb == 0] or [SWAP_K_BLOCKS[0]]
    return max(kbs)


def swap_k_block_arg(in_fmt: str, K: int, k_block: Optional[int] = None) -> int:
    """The k-block as the kernel spells it: ``k_block`` (default `swap_k_block`) with the format's default as 0 (one object
    per k-block, `blocks.compile.swapped_kernel_name`)."""
    kb = k_block or swap_k_block(in_fmt, K)
    return 0 if kb == SWAP_K_BLOCKS[0] else kb


def can_implement_grouped_mxfp8(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """Acceptance of the grouped body for the MX classes (`blocks.compile.can_implement_grouped_blockscaled` at the chain
    taken as an MX chain of the instruction class ``in_fmt``, an MXFP8 or a mixed one): a well-formed dense record of the
    class's space at swizzle 1 (no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold: the group
    scheduler has none of them) with a block-scaled tile N, a chain with a bf16 / f16 / f32 output or the fused
    requantization, K a whole number of 32-element scale blocks (hence 16-byte e4m3 rows; whole granules of a mixed class),
    16-byte-aligned output rows (whole 32-element blocks for the fused requantization), whole ``2 x GROUP`` column blocks
    for the gated chain, at most ``MAX_E`` experts."""
    from .blocks.compile import can_implement_grouped_blockscaled

    if in_fmt not in MX_SERVED or not isinstance(chain, GroupedEpilogueBlock):
        return False
    return can_implement_grouped_blockscaled(
        knobs, N, K, E, MxChain.of(chain, in_fmt), in_fmt
    )


def can_implement_swapped_mxfp8(
    knobs: Knobs,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str,
    k_block: int = 0,
) -> bool:
    """Acceptance of the swapped body for the MX classes (`blocks.compile.can_implement_swapped_blockscaled` at the chain
    taken as an MX chain): a swapped record of the dense MX space at a narrow token tile (128 weight rows x 8 / 16 / 32
    tokens, 1-CTA, no cluster, swizzle 1) under the grouped body's operand rules; ``k_block`` (as `swap_k_block_arg` spells
    it) one of `SWAP_K_BLOCKS` dividing ``K`` for the MXFP8 classes (a mixed class's record carries its own stage width and
    takes none). The body reads the dense packing only: the dense rows with the scale factors at the padded positions."""
    from .blocks.compile import can_implement_swapped_blockscaled

    if in_fmt not in MX_SERVED or not isinstance(chain, GroupedEpilogueBlock):
        return False
    if k_block and (
        in_fmt not in MX_CLASSES or k_block not in SWAP_K_BLOCKS or K % k_block
    ):
        return False
    return can_implement_swapped_blockscaled(
        knobs, N, K, E, MxChain.of(chain, in_fmt), in_fmt
    )


def can_implement_mxfp8(
    knobs: Knobs, N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str
) -> bool:
    """A record either body takes for the MXFP8 classes: the grouped body (a dense record, either packing) or the swapped body
    (a narrow swapped record); the two spaces are disjoint."""
    return can_implement_grouped_mxfp8(
        knobs, N, K, E, chain, in_fmt
    ) or can_implement_swapped_mxfp8(knobs, N, K, E, chain, in_fmt)


#: SM cycles of one smem-stage round trip of the block-scaled mainloop (the stage's barrier wait, its scale-factor copies to TMEM,
#: the commit) that the stream does not hide at a few rows per expert: empirical, one value for every part.
STAGE_ROUND_TRIP_CYCLES = 76.0


def stage_round_trip_cycles(cfg: Knobs, K: int, in_fmt: str) -> float:
    """The mainloop's stage round trips over K at the record's k-block (the swapped body's two-atom stage halves them,
    `swap_k_block`; the grouped body stages one atom) times the round trip's cycles: the term that prices the k-block."""
    kb = swap_k_block(in_fmt, K) if cfg.swap_ab else SWAP_K_BLOCKS[0]
    return (K / kb) * STAGE_ROUND_TRIP_CYCLES


RASTER_N_ROWS = 128  # an expert spanning a whole 128-row block or more: the grouped scheduler rasters along N
EPI_WARPGROUPS = (
    EPI_WARPGROUPS_BLOCKSCALED  # the block-scaled grouped body's epilogue warpgroups
)


def raster_for(rows: int) -> str:
    """The grouped scheduler's raster as a structural rule, not an estimate term: along N once an expert's rows fill a
    128-row block (consecutive CTAs then share the expert's A block while its weight tiles stream), along M below it
    (the weight tile is the larger operand of a few rows)."""
    return "n" if rows >= RASTER_N_ROWS else "m"


def pair_term(k: Knobs, rows: int) -> float:
    """The expected dummy half of a cluster that pairs two 128-row tiles along M (``cluster_shape_mn[0] == 2``): every
    expert's rows are padded to the record's M extent (128 x cm), and with the experts' row counts phased uniformly over
    that extent the expected padding is half an extent, so the pair's records carry (rows + 128) / (rows + 64) of the
    single-tile records' rows; the dense estimate at the average rows sees only the average's own tail."""
    cm = k.cluster_shape_mn[0]
    return (rows + 64.0 * cm) / (rows + 64.0)


def grouped_knobs_mx(in_fmt: str = "mxfp8", clusters=None) -> List[Knobs]:
    """The grouped MX space before the body's shape acceptance: every well-formed dense record of the class at swizzle 1
    with a block-scaled tile N, no in-cluster split, no operand swap, no B-reuse tile, no in-CTA chunk fold (the group
    scheduler has none of them). One record list for both instruction classes."""
    return [
        k
        for k in knob_product(
            swizzles=(1,),
            clusters=CLUSTERS if clusters is None else clusters,
            in_fmt=in_fmt,
        )
        if k.cluster_k == 1
        and not k.swap_ab
        and not k.b_reuse
        and not k.cta_chunks
        and k.tile_n in TILES_N_BLOCKSCALED
    ]


#: The gather form's token tile: GEMM1's swapped record reads the token rows itself (through the plan's map) when its token tile
#: is at most this wide; the one-warp scale producer keeps a narrow tile's row copies in flight, a wider tile exposes them
#: (declared empirical: the in-op A/B of the two forms per token tile).
GATHER_TOKEN_TILE_MAX = 8
#: Rows per expert at most for the gather form (an empirical bound): the gather reads every distinct token row of a tile
#: separately at each k-stage while the launches the form saves are fixed, so the form pays once the tiles hold several live
#: rows.
GATHER_ROWS_PER_EXPERT_MAX = 2


def gather_form_rule(config1: Knobs, rows: Optional[int] = None) -> Knobs:
    """The fused op's default form of GEMM1's record on the dense packing: the gather form (``gather_a``) of a swapped record at
    most `GATHER_TOKEN_TILE_MAX` tokens wide at a routing of at most `GATHER_ROWS_PER_EXPERT_MAX` rows per expert (``rows``:
    `schedule.rows_per_expert`; None: the record's rule alone) -- the standalone scatter launch the box form needs is then the
    one launch the gather saves -- else the record as given. A placement: the same bits either way."""
    import dataclasses

    if (
        config1.swap_ab
        and config1.mma_tiler_mn[1] <= GATHER_TOKEN_TILE_MAX
        and (rows is None or rows <= GATHER_ROWS_PER_EXPERT_MAX)
    ):
        return dataclasses.replace(config1, gather_a=True)
    return config1


def swapped_knobs_mx(in_fmt: str = "mxfp8") -> List[Knobs]:
    """The swapped record space of the MX classes (:mod:`.blocks.swap`): the dense MX class's swapped 1-CTA records at the
    narrow token tiles (128 weight rows x 8 / 16 / 32 tokens), one per tile -- the same list for both MXFP8 instruction
    classes; a mixed class's records carry the stage width the dense class gives it (``SWAP_K_BLOCK``: the sm_107 class
    streams its e2m1 weight at a 256-element stage, a 128-byte TMA row; the sm_100 class the format's default), the record's
    own field (the lowering's inner radix, a placement of the same fold). Disjoint from the grouped space (the swapped flag);
    the body reads the dense packing."""
    import dataclasses

    from ...gemm.batch_invariant.generator import SWAP_K_BLOCK
    from .blocks.swap import swapped_knobs

    assert in_fmt in MX_SERVED, in_fmt
    if in_fmt in MX_CLASSES:
        return swapped_knobs(MX_CLASSES[0])
    kb = SWAP_K_BLOCK.get(in_fmt, 0)
    return [dataclasses.replace(k, k_block=kb) for k in swapped_knobs(in_fmt)]


def all_valid_configs_mx(
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp8",
    clusters=None,
) -> List[Knobs]:
    """The grouped record space the guarantee is claimed over, for a weight shape, a chain and a class (the swapped records
    are the second space: :func:`all_valid_configs_mx_swapped`; the guarantee covers both)."""
    return [
        k
        for k in grouped_knobs_mx(in_fmt, clusters)
        if can_implement_grouped_mxfp8(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mx_swapped(
    N: int, K: int, E: int, chain: GroupedEpilogueBlock, in_fmt: str = "mxfp8"
) -> List[Knobs]:
    """The swapped records the swapped body accepts for a weight shape, a chain and a class (the dense packing)."""
    return [
        k
        for k in swapped_knobs_mx(in_fmt)
        if can_implement_swapped_mxfp8(k, N, K, E, chain, in_fmt)
    ]


def all_valid_configs_mxfp8(
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp8",
    clusters=None,
) -> List[Knobs]:
    """Both record spaces for a weight shape, a chain and a class: the grouped records, then the swapped ones."""
    return all_valid_configs_mx(
        N, K, E, chain, in_fmt, clusters
    ) + all_valid_configs_mx_swapped(N, K, E, chain, in_fmt)


@functools.lru_cache(maxsize=None)
def chain_nodes_mx(chain: GroupedEpilogueBlock) -> float:
    """The node count of the chain's term per output element, the exposed epilogue's cost unit (:func:`.schedule.chain_nodes`):
    the bf16 chains' own; the fused requantization's (``out_fmt == "mxfp8"``) the bf16-rounded chain's plus the quantizer
    class's element chain (the inverse scale applied, the e4m3 conversion) and its share of the block's scale chain and amax
    tree (one scale per 32 outputs: the amax's ``fabs`` leaves and ``fmaxc`` combines, the scale's normalisation, UE8M0
    conversion and exact inverse)."""
    from ...batch_invariant.quantization import mx_spec
    from . import mterm as term

    if chain.out_fmt != OUT_FMT:
        # the bf16 chain's term (an MX chain's twin: the shared counter knows the bf16 and fp8 chain objects)
        return float(chain_nodes(chain.twin if isinstance(chain, MxChain) else chain))
    alg = term.TermAlgebra()
    fmt, rule = mx_spec.MXFP8, mx_spec.DEFAULT_RULE
    amax = alg.leaf("amax", 0)
    sf_d, inv = mx_spec.scale_chain(alg, amax, fmt, rule)
    elem = mx_spec.quant_elem(alg, alg.leaf("y", 0), alg.leaf("inv", 0), fmt, rule)
    scale_nodes = term.size(inv)  # the DAG's nodes, as `chain_nodes` counts the chain's
    amax_nodes = (
        2 * mx_spec.BLOCK - 1
    )  # the fabs leaves and the pairwise fmaxc combines of one block
    return (
        float(chain_nodes(bf16_twin(chain)))
        + term.size(elem)
        + (scale_nodes + amax_nodes) / float(mx_spec.BLOCK)
    )


def rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: GroupedEpilogueBlock,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    k: int = TOP_K,
    nominal: bool = False,
    n_active: Optional[int] = None,
    swap: Optional[bool] = None,
) -> List[Knobs]:
    """:func:`_rank_grouped_mx` memoised on its arguments (all hashable scalars; the chain a frozen dataclass, taken at
    ``in_fmt``): the runner ranks at every forward without a tactic, once per token count."""
    chain = MxChain.of(chain, in_fmt)
    return list(
        _rank_grouped_mx(
            rows, N, K, E, chain, in_fmt, sm_count, family, k, nominal, n_active, swap
        )
    )


@functools.lru_cache(maxsize=16384)
def _rank_grouped_mx(
    rows: int,
    N: int,
    K: int,
    E: int,
    chain: MxChain,
    in_fmt: str,
    sm_count: Optional[int],
    family: Optional[str],
    k: int,
    nominal: bool,
    n_active: Optional[int],
    swap: Optional[bool],
) -> Tuple[Knobs, ...]:
    """The ranked records for one GEMM over both record spaces: the grouped MX records the body accepts for the weight shape
    and the chain (at a decode-sized ``rows`` the dense dispatcher's structural guard, `heuristic.decode_sane`: 1-CTA tiles
    in an N-cluster, selects the candidates) and the swapped records the swapped body accepts (``swap``: None ranks them
    with the rest, True keeps only them, False excludes them), ordered by the dense MX class's estimate at ``(M = rows,
    N = n_active * N, K)`` (``n_active`` the active experts, ``E`` by default; a swapped record is priced as the dense model
    prices a swapped decode record) times the pair term (the expected dummy half of the two-tile clusters along M) and the
    swapped records' spill factor (`schedule.tiles_term`), plus the last tile's exposed epilogue chain
    (`schedule.exposed_epilogue_cycles` with the chain's term, :func:`chain_nodes_mx`, over the block-scaled body's four
    epilogue warpgroups) and the token box's zero-fill at the e4m3 width (`schedule.zero_fill_cycles`), each tile family
    priced once (at the enumeration's raster) and the raster the structural rule gives the grouped scheduler
    (`raster_for`) first among its twins, best first, ties in the spaces' enumeration order (the grouped records before the
    swapped ones, the narrower token tile first). As in the dense ``rank_topk``, each of the best ceil(k/2) records is
    followed by its raster twin (the one knob the estimate does not price), then the rank order fills the list: distinct
    records, at most ``k``, never empty for an implementable shape."""
    import dataclasses

    fam = family or "sm_107"
    dense = (
        []
        if swap is True
        else [
            c
            for c in grouped_knobs_mx(in_fmt)
            if can_implement_grouped_mxfp8(c, N, K, E, chain, in_fmt)
        ]
    )
    swapped = (
        []
        if swap is False
        else [
            c
            for c in swapped_knobs_mx()
            if can_implement_swapped_mxfp8(c, N, K, E, chain, in_fmt)
        ]
    )
    space = dense + swapped
    guarded = [c for c in dense if decode_sane(c, rows, False)] or dense
    raster = raster_for(rows)
    nodes = chain_nodes_mx(chain)

    def key(i, c):
        # the shared contract's cost (`schedule.record_cost`: the dense estimate under the MoE ranking's overrides at the
        # grouped shape, the pair and spill terms, the fused requantization chain's exposed epilogue over the block-scaled
        # body's four warpgroups, the token box's exposed fill, the body's fixed cost) plus this class's stage round trips
        # at the record's k-block; three significant digits, so records the estimate cannot separate fall through to the
        # structural raster and the enumeration order
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
            in_fmt,
            nominal,
            epi_warpgroups=EPI_WARPGROUPS,
        ) + stage_round_trip_cycles(c, K, in_fmt)
        return (
            float(f"{est:.3g}"),
            0 if (c.swap_ab or c.raster_along == raster) else 1,
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
    _rank_grouped_mx.cache_clear
)  # the memo follows the dispatcher's rankings (a device's registered numbers)


def ranked_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedEpilogueBlock,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    top_k: int = TOP_K,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[List[Knobs], List[Knobs]]:
    """The ranked lists of both GEMMs (GEMM1 on (N = 2I or I, K = H), GEMM2 on (N = H, K = I)) at the grouped launch's shape."""
    rows, n_act = rows_per_expert(T, k, E), active_experts(T, k, E)
    n1 = 2 * I if chain1.pair else I
    return (
        rank_grouped_mx(
            rows,
            n1,
            H,
            E,
            chain1,
            in_fmt,
            sm_count,
            family,
            top_k,
            nominal,
            n_act,
            swap,
        ),
        rank_grouped_mx(
            rows,
            H,
            I,
            E,
            GEMM2_CHAIN,
            in_fmt,
            sm_count,
            family,
            top_k,
            nominal,
            n_act,
            swap,
        ),
    )


def default_configs_mx(
    T: int,
    k: int,
    H: int,
    I: int,
    E: int,
    chain1: GroupedEpilogueBlock,
    in_fmt: str = "mxfp8",
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    nominal: bool = False,
    swap: Optional[bool] = None,
) -> Tuple[Knobs, Knobs]:
    """``(config for GEMM1 (N = 2I or I, K = H), config for GEMM2 (N = H, K = I))``: the first record of each ranked list over
    both record spaces (``swap``: None, or True / False to keep only / exclude the swapped records)."""
    c1, c2 = ranked_mx(
        T, k, H, I, E, chain1, in_fmt, sm_count, family, 1, nominal, swap
    )
    if not c1 or not c2:
        raise ValueError(
            f"no grouped MX record implements (H={H}, I={I}, E={E}, chain={chain1.describe()}, class={in_fmt})"
        )
    return c1[0], c2[0]


def packing_for(c1: Knobs, c2: Knobs) -> str:
    """The packing of the expanded rows the two records run on: the dense packing (the rows back to back, the scale factors
    at the padded positions) when either is a swapped record (its body reads nothing else; the padded body reads it too),
    the padded packing (whole 128-row atoms per group) otherwise. A placement: the same bits."""
    return "dense" if (c1.swap_ab or c2.swap_ab) else "padded"
