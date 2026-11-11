# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dispatcher of the batch-invariant GEMM classes: a ranking heuristic over the class's configuration space.

For a shape, layout, format and part it orders the records of the enumerated space by a per-tile roofline estimate
(SM-cycles of the launch) and `schedule.default_config` takes the first; `schedule.rank_topk` hands the first ``k`` to
FlashInfer's autotuner. Nothing here reaches the bits: every record of the space computes the class (R3), the heuristic
only orders them, so it may be as rough as it is.

The estimate uses the part's numbers (SM count, co-resident cluster capacity per cluster size, L2 size, the dense bf16
MMA rate per SM, the L2 -> SMEM and HBM bandwidths per SM; the capacity and the L2 size are the device's own once the
API has queried them in the process, else the family's nominal table, which ``nominal=True`` selects regardless: the
generated Lean cases are the nominal parts' dispatch) and a few shape constants (`CONSTANTS`: per-CTA row efficiency,
the transaction efficiency of a row-major weight's narrow boxes, the multicast benefit, the shared-L2 fraction, the
usable L2 fraction of the HBM traffic model, per-step and fixup floors, raster / swizzle / 2-CTA adjustments). Every
constant is empirical and shared by all parts and formats; none is fitted per part. The format enters through its
structure only: element width, MMA K step and the 8-bit kind's doubled instruction rate.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .generator import Knobs, enumerate_space
from .spec import BLOCK_K, ELEM_BITS_OF, FP8_BLOCKSCALE_FMTS, MMA_K_OF, MMA_RATE_OF

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- the part


@dataclass(frozen=True)
class Part:
    """A part's nominal numbers. ``family`` is the compute capability's family (sm_107 / sm_100 / sm_103); ``capacity`` the
    co-resident cluster count per cluster size at the nominal SM count (the driver's GPC-aware occupancy); the rates are per
    SM per cycle, dense bf16 for the MMA; ``l2_bytes`` the L2 size."""

    family: str
    sm_count: int
    capacity: Tuple[Tuple[int, int], ...]
    mma_flop_per_clk: float = 8192.0
    load_bytes_per_clk: float = 64.0
    hbm_bytes_per_clk: float = 32.0
    l2_bytes: float = 112 * 2**20


_RUBIN_CAPACITY = ((1, 204), (2, 102), (4, 42), (8, 20))
_B200_CAPACITY = ((1, 148), (2, 74), (4, 33), (8, 15))

PART_TABLE: Dict[str, Part] = {
    "sm_107": Part("sm_107", 204, _RUBIN_CAPACITY, hbm_bytes_per_clk=32.0),
    "sm_100": Part(
        "sm_100", 148, _B200_CAPACITY, hbm_bytes_per_clk=28.0, l2_bytes=126.5 * 2**20
    ),
    "sm_103": Part(
        "sm_103", 148, _B200_CAPACITY, hbm_bytes_per_clk=28.0, l2_bytes=126.5 * 2**20
    ),
}

#: The parts as (family, SM count): what the dispatcher serves, what the generated Lean cases and the tests enumerate.
PARTS: Tuple[Tuple[str, int], ...] = (
    ("sm_107", 204),
    ("sm_107", 224),
    ("sm_100", 148),
    ("sm_100", 152),
    ("sm_103", 148),
    ("sm_103", 152),
)
FAMILY_OF_CC = {(10, 0): "sm_100", (10, 3): "sm_103", (10, 7): "sm_107"}


def family_of(cc: Tuple[int, int]) -> str:
    """The architecture family of a compute capability; the API passes the operand device's. The family selects the
    block-scaled classes' instruction (`schedule.mx_class_for`), hence the bits, so an unlisted capability raises instead
    of borrowing a family."""
    from ...batch_invariant.support import BatchInvariantUnsupportedError

    key = (int(cc[0]), int(cc[1]))
    if key not in FAMILY_OF_CC:
        raise BatchInvariantUnsupportedError(
            f"batch_invariant: compute capability {key} is not a part of the mode; the parts are {sorted(FAMILY_OF_CC)}"
        )
    return FAMILY_OF_CC[key]


# the co-resident cluster counts a device reported at run time, keyed (family, SM count); device-free callers use the table
_QUERIED_CAPACITY: Dict[Tuple[str, int], Dict[int, int]] = {}


def capacity_for(
    family: str, sm_count: Optional[int], nominal: bool = False
) -> Dict[int, int]:
    """Co-resident clusters per cluster size for the part: the device's own numbers when the API has queried them in this
    process (unless ``nominal``), else the family's nominal table scaled to ``sm_count``."""
    part = PART_TABLE.get(family, PART_TABLE["sm_107"])
    sms = sm_count or part.sm_count
    q = None if nominal else _QUERIED_CAPACITY.get((family, sms))
    if q:
        return q
    return {
        size: max(1, round(cap * sms / part.sm_count)) for size, cap in part.capacity
    }


def register_capacity(family: str, sm_count: int, size: int, clusters: int) -> None:
    """Record a device's co-resident cluster count for one cluster size (from the API's occupancy query)."""
    cap = _QUERIED_CAPACITY.setdefault(
        (family, sm_count), dict(capacity_for(family, sm_count))
    )
    if cap.get(size) == clusters:
        return  # nothing the ranking depends on changed: the cached rankings stay
    cap[size] = clusters
    rank.cache_clear()


# the L2 sizes devices reported at run time, keyed like the capacities
_QUERIED_L2: Dict[Tuple[str, int], float] = {}


def l2_for(family: str, sm_count: Optional[int], nominal: bool = False) -> float:
    """The part's L2 size in bytes: the device's own once the API has recorded it (unless ``nominal``), else the
    family's."""
    part = PART_TABLE.get(family, PART_TABLE["sm_107"])
    if not nominal:
        q = _QUERIED_L2.get((family, sm_count or part.sm_count))
        if q:
            return q
    return part.l2_bytes


def register_l2(family: str, sm_count: int, l2_bytes: float) -> None:
    """Record a device's L2 size (from the API's device properties)."""
    if _QUERIED_L2.get((family, sm_count)) == float(l2_bytes):
        return  # unchanged: the cached rankings stay
    _QUERIED_L2[(family, sm_count)] = float(l2_bytes)
    rank.cache_clear()


# ----------------------------------------------------------------------------- the estimate

#: Shape constants of the estimate (empirical, shared by every part and format).
CONSTANTS: Dict[str, float] = dict(
    rows64_eff=0.7,  # MMA efficiency of a 64-row CTA tile relative to 128 rows
    eff16b=0.15,  # transaction efficiency of a B box 16 bytes wide ...
    eff32b=0.2,  # ... 32 bytes ...
    eff64b=0.6,  # ... 64 bytes (128 and wider: 1)
    mcast_pow=1.25,  # multicast divides an operand's traffic by cluster_dim ** mcast_pow
    l2_frac=1.0,  # the shared L2 fabric delivers this fraction of SMs x load bandwidth in aggregate
    l2_resident=0.6,  # the fraction of the L2 the HBM traffic model may keep resident between a panel's uses
    refetch=1.3,  # the refetched bytes' HBM time beyond the absorbed share adds this many times over
    absorb=0.2,  # the fraction of the body's time during which refetched bytes stream without a stall
    step_floor=32.0,  # cycles per k-step no tile can beat
    fixup_floor=8000.0,  # cycles of the workspace split's fixup launch
    raster_pow=0.05,  # the wrong raster: (tile-count imbalance) ** raster_pow
    raster_min=2.0,  # ... once the tile-count imbalance reaches this ratio
    overrun_pow=0.65,  # a swizzle group larger than its axis: (swizzle / tiles) ** overrun_pow
    sw_pen=1.1,  # swizzle > 1 where it cannot help
    sw_gain=1.0,  # swizzle > 1 with many tiles in flight on both axes
    sw_partial=1.1,  # a swizzle that does not divide its axis (a partial last group)
    two_cta=0.9,  # the 2-CTA pair's shared B and halved per-CTA issue
    promote_bytes_per_clk=256.0,  # the block-scale class's promotion pass: the CTA tile's fp32 partial through the promotion warps' TMEM
    # loads and packed fmas, per block
    promote_handoff_clk=256.0,  # the block-scale class's per-block hand-off: the dependent chain of the accumulator stage's mbarrier wake-up, the
    # TMEM load round trip and the release the MMA warp observes; nominally two round trips of 128 cycles
    promote_pair_clk=16.0,  # the 2-CTA pair's hand-off waits for the peer CTA's release across the cluster: the remote arrive's extra latency
)

DECODE_M = 64  # up to this many rows the candidates are the `decode_sane` subset
TOKEN_BOX_ROWS = 128  # the rows of the token (A) TMA box: the CTA tile's M


def decode_sane(
    k: Knobs, M: int, swapped: bool = False, narrow_tiles: bool = False
) -> bool:
    """The candidate guard at M <= DECODE_M (structural): 1-CTA tiles of N >= 32 in an N-cluster at swizzle 1; where the
    space has swapped decode records (``swapped``: the block-scaled classes), those records at swizzle 1 -- the narrow
    token tiles on the 1-CTA instruction, the 64-wide one on the 2-CTA pair once the tokens fill at least half of it --
    and no dense record (one token tile holds every token, the weights stream through the 128-row operand). The dense
    guard leaves out the 16-wide tile because its 16-byte B box wastes the TMA's transactions; ``narrow_tiles`` admits it
    for the block-scale class, whose decode block is bounded by the per-block accumulator hand-off (`estimate_cycles`),
    not by the box, so the narrowest tile that still fits the part in one wave streams the weight through the most CTAs.
    Records outside the guard stay members of the space (the autotuner may still reach them)."""
    if M > DECODE_M:
        return True
    if swapped:
        return (
            k.swap_ab
            and k.swizzle_size == 1
            and (not k.use_2cta_instrs or (k.tile_n == 64 and M > 32))
        )
    return (
        (not k.use_2cta_instrs)
        and k.tile_n >= (16 if narrow_tiles else 32)
        and k.cluster_m == 1
        and k.swizzle_size == 1
    )


def hbm_traffic(
    M: int,
    N: int,
    K: int,
    eb: float,
    out_bytes: int,
    cl_rows: int,
    cl_cols: int,
    ncl_m: int,
    ncl_n: int,
    items: int,
    cap: int,
    sw: int,
    rast: str,
    l2: float,
) -> float:
    """HBM bytes of the launch under the persistent scheduler's tile order: raster along m walks M within a band of ``sw``
    cluster tiles along N (along n: mirrored), ``cap`` clusters per wave, ``items`` work items (the workspace split) each
    over K / items. Within a wave every CTA runs the same k-step, so one wave's operand slabs are shared through the L2;
    across waves a band's panels stay resident when they fit in ``l2`` beside one wave's share of the other operand, else
    they are fetched again every wave of the band, and the other operand is fetched once when the whole of it fits beside
    one band, else once per band (never more than once per wave). ``l2`` is the usable L2 in bytes; the cluster tile is
    ``cl_rows`` x ``cl_cols`` and the grid ``ncl_m`` x ``ncl_n`` cluster tiles."""
    kk = K / items
    a_panel = cl_rows * kk * eb
    b_panel = cl_cols * kk * eb
    a_all = (
        M * kk * eb
    )  # the operands themselves (out-of-bounds boxes cost no HBM traffic)
    b_all = N * kk * eb
    waves_total = max(1, -(-(ncl_m * ncl_n) // cap))
    if rast == "m":
        s_eff = max(1, min(sw, ncl_n))
        bands = -(-ncl_n // s_eff)
        band = s_eff * b_panel  # the resident B band
        wave_other = min(ncl_m, -(-cap // s_eff)) * a_panel  # one wave's A rows
        waves_per_band = max(1, -(-(ncl_m * s_eff) // cap))
        a_traffic = a_all * (1 if a_all + band <= l2 else min(bands, waves_total))
        b_traffic = b_all * (
            1 if (waves_per_band == 1 or band + wave_other <= l2) else waves_per_band
        )
    else:
        s_eff = max(1, min(sw, ncl_m))
        bands = -(-ncl_m // s_eff)
        band = s_eff * a_panel
        wave_other = min(ncl_n, -(-cap // s_eff)) * b_panel
        waves_per_band = max(1, -(-(ncl_n * s_eff) // cap))
        b_traffic = b_all * (1 if b_all + band <= l2 else min(bands, waves_total))
        a_traffic = a_all * (
            1 if (waves_per_band == 1 or band + wave_other <= l2) else waves_per_band
        )
    return items * (a_traffic + b_traffic) + M * N * out_bytes


def estimate_cycles(
    k: Knobs,
    M: int,
    N: int,
    K: int,
    chunks: int,
    b_major: str,
    in_fmt: str,
    family: str,
    sm_count: Optional[int],
    out_bytes: int = 2,
    c: Optional[Dict[str, float]] = None,
    nominal: bool = False,
) -> float:
    """SM-cycles of the launch for record ``k`` at the shape: waves of CTA tiles over the part's co-resident cluster
    capacity, each the slower of its MMA and its L2 -> SMEM traffic per k-step plus a floor, bounded below by the shared
    L2 fabric and by the operands' HBM bytes, plus the stall of the bytes the scheduler's tile order fetches again
    (`hbm_traffic`) beyond what the body's duration streams unnoticed, the epilogue and (workspace split) the fixup;
    raster, swizzle and 2-CTA adjustments multiply. ``b_major`` does not enter the estimate (the narrow-tile cost is
    measured alike in both layouts); the layout selects the space. ``nominal`` takes the family's table for the
    capacity and the L2 size instead of a device's queried numbers. Lower is better; ``inf`` when the record cannot
    tile the shape."""
    p = c or CONSTANTS
    part = PART_TABLE.get(family, PART_TABLE["sm_107"])
    sms = sm_count or part.sm_count
    cap = capacity_for(family, sms, nominal)
    tm, tn = k.mma_tiler_mn
    cm, cn = k.cluster_shape_mn
    group = 2 if k.use_2cta_instrs else 1
    rows = tm // group
    eb = ELEM_BITS_OF[in_fmt] / 8.0
    kk = MMA_K_OF[in_fmt]
    rate = (
        part.mma_flop_per_clk * MMA_RATE_OF[in_fmt]
    )  # the 8-bit kinds: twice the dense bf16 rate
    # the swapped decode record computes C^T: the kernel's rows are the weights' N, its columns the tokens' M
    pm, pn = (N, M) if k.swap_ab else (M, N)
    m_tiles = -(-pm // tm)
    n_tiles = -(-pn // tn)
    if m_tiles == 0 or n_tiles == 0:
        return float("inf")
    ws = (
        chunks > 1 and not k.cta_chunks and k.cluster_k == 1
    )  # workspace split: `chunks` work items per tile and a fixup kernel (the in-CTA and in-cluster folds store directly)
    items = chunks if ws else 1
    ctas_m = m_tiles * group
    pad_m = -(-ctas_m // cm) * cm
    pad_n = -(-n_tiles // cn) * cn
    clusters = pad_m * pad_n // (cm * cn) * items
    size = cm * cn
    capacity = cap.get(size) or max(1, sms // size)
    waves = -(-clusters // capacity)
    ksteps = max(1.0, (K / (chunks if (ws or k.cluster_k > 1) else 1)) / kk)
    flops = 2.0 * rows * tn * kk
    mma_eff = 1.0 if rows >= 128 else p["rows64_eff"]
    mma_cycles = flops / (rate * mma_eff)
    # a narrow N tile (the CTA's share tn / group, in bytes of a one-byte element): its B box wastes transactions for a
    # row-major weight and each MMA's A-operand read is amortised over fewer MACs in either layout; the cost is measured
    # alike in both layouts, per MMA column: a 4-bit element is a column like an 8-bit one
    wb = tn / group * max(eb, 1.0)
    b_eff = (
        p["eff16b"]
        if wb <= 16
        else p["eff32b"]
        if wb <= 32
        else p["eff64b"]
        if wb <= 64
        else 1.0
    )
    if k.swap_ab:
        b_eff = 1.0  # the swapped record's narrow tile is the token tile, one per CTA for the whole K: no such cost is measured for it
    a_bytes = rows * kk * eb / (cn ** p["mcast_pow"])
    b_bytes = tn * kk * eb / (cm ** p["mcast_pow"]) / b_eff
    step = (
        max(mma_cycles, (a_bytes + b_bytes) / part.load_bytes_per_clk) + p["step_floor"]
    )
    if in_fmt in FP8_BLOCKSCALE_FMTS:
        # the block-scale class promotes every accumulator element of the CTA tile once per K block (two promotion warpgroups:
        # a TMEM read and a packed fma per element pair) and ends every block in an accumulator hand-off: the promotion warps
        # wait for the stage, load it, fold, fence and release it before the MMA warp may reuse it. The pass is one more
        # per-k-step term next to the MMA and the operand traffic: a fixed hand-off chain (its latency does not shrink with
        # the tile, so at decode every tile of one wave costs the same per block and the narrowest one-wave tile wins) plus
        # the tile's bytes at the pass rate. In the 2-CTA pair the stage is released by both CTAs' promotion warps and the
        # leader's MMA waits for the peer's arrive across the cluster, so the pair's chain is longer by that remote arrive.
        handoff = p["promote_handoff_clk"] + (
            p["promote_pair_clk"] if k.use_2cta_instrs else 0.0
        )
        promote = (handoff + rows * tn * 4.0 / p["promote_bytes_per_clk"]) / (
            BLOCK_K // kk
        )
        step = (
            max(
                mma_cycles,
                (a_bytes + b_bytes) / part.load_bytes_per_clk,
                promote,
            )
            + p["step_floor"]
        )
    body = waves * ksteps * step
    active = ctas_m * n_tiles * items
    l2_total = (
        active
        * ksteps
        * (a_bytes + b_bytes)
        / (part.load_bytes_per_clk * sms * p["l2_frac"])
    )
    body = max(body, l2_total)
    epi = waves * rows * tn * (4.0 if ws else out_bytes) / part.load_bytes_per_clk
    fix = (
        (M * N * (chunks * 4.0 + out_bytes)) / (part.hbm_bytes_per_clk * sms)
        + p["fixup_floor"]
        if ws
        else 0.0
    )
    hbm_bytes = hbm_traffic(
        pm,
        pn,
        K,
        eb,
        out_bytes,
        cm * rows,
        cn * tn,
        pad_m // cm,
        pad_n // cn,
        items,
        capacity,
        k.swizzle_size,
        k.raster_along,
        p["l2_resident"] * l2_for(family, sms, nominal),
    )
    unique = (pm * K + K * pn) * eb + pm * pn * out_bytes
    bw = part.hbm_bytes_per_clk * sms
    # the operands' own bytes bound the launch from below (overlapped with the body). The bytes the tile order fetches
    # again are hidden while the HBM is lightly loaded: a fraction `absorb` of the body's time streams them at no cost
    # (less what the operands themselves already take); beyond that the pipeline stalls and `refetch` x the remainder
    # adds to the launch
    slack = max(0.0, p["absorb"] * body - unique / bw)
    t = (
        max(body + epi, unique / bw)
        + p["refetch"] * max(0.0, (hbm_bytes - unique) / bw - slack)
        + fix
    )
    # L2 locality. Raster: along the axis with fewer tiles (consecutive CTAs then share the larger operand's panel); the
    # wrong choice costs more the more unbalanced the tile grid. Swizzle: groups of ``swizzle_size`` tiles along the slow
    # axis; a group larger than that axis leaves CTAs idle (the overrun), a valid group helps only when many tiles are in
    # flight on both axes.
    imbalance = max(m_tiles, n_tiles) / min(m_tiles, n_tiles)
    if m_tiles > 1 and n_tiles > 1 and imbalance >= p["raster_min"]:
        good_raster = (k.raster_along == "m") == (m_tiles < n_tiles)
        if not good_raster:
            t *= imbalance ** p["raster_pow"]
    sw = k.swizzle_size
    if sw > 1:
        axis = n_tiles if k.raster_along == "m" else m_tiles
        if sw > axis:
            t *= (sw / axis) ** p["overrun_pow"]
        else:
            t *= (
                p["sw_gain"]
                if (waves > 1 and min(m_tiles, n_tiles) >= 8)
                else p["sw_pen"]
            )
            if axis % sw:
                t *= p["sw_partial"]
    if k.use_2cta_instrs and in_fmt not in FP8_BLOCKSCALE_FMTS:
        # the pair's halved per-CTA issue; not for the block-scale class, whose pair runs its promotion in lockstep (the
        # accumulator hand-off above is pair-wide: the block cannot be shorter than either CTA's promotion pass)
        t *= p["two_cta"]
    return t


@functools.lru_cache(maxsize=4096)
def rank(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    arch_name: Optional[str] = None,
    topk: Optional[int] = None,
    in_fmt: str = "bf16",
    nominal: bool = False,
) -> Tuple[Tuple[Knobs, float], ...]:
    """The records of the class's enumerated space for the shape, layout and format (at M <= DECODE_M the `decode_sane`
    subset) ordered by the estimate, best first; ``topk`` truncates. ``arch_name`` is accepted for the former signature and
    ignored; ``nominal`` ranks at the family's nominal table (the generated Lean cases), never a device's queried numbers.
    Cached per argument tuple."""
    fam = family or "sm_107"
    space = enumerate_space(M, N, K, chunks=chunks, b_major=b_major, in_fmt=in_fmt)
    swapped = any(k.swap_ab for k in space)
    blockscale = in_fmt in FP8_BLOCKSCALE_FMTS
    guarded = [k for k in space if decode_sane(k, M, swapped, blockscale)]
    if blockscale and M <= TOKEN_BOX_ROWS // 4:
        # the block-scale class at a few tokens: the 128-row token box is at least three quarters zero-fill, a TMA cost per
        # block per CTA that the N-cluster's multicast issues once per cluster; the widest N-cluster shares it most, as long
        # as its clusters still fit the part in one wave (the estimate's wave count would otherwise double)
        cap4 = capacity_for(fam, sm_count, nominal).get(4) or max(
            1, (sm_count or PART_TABLE[fam].sm_count) // 4
        )
        wide = [
            k
            for k in guarded
            if k.cluster_n == 4 and ((N + k.tile_n - 1) // k.tile_n + 3) // 4 <= cap4
        ]
        guarded = wide or guarded
    space = guarded or space
    if not space:
        return ()
    scored = [
        (
            estimate_cycles(
                k,
                M,
                N,
                K,
                chunks,
                b_major,
                in_fmt,
                fam,
                sm_count,
                out_bytes,
                nominal=nominal,
            ),
            i,
            k,
        )
        for i, k in enumerate(space)
    ]
    scored.sort(key=lambda t: (t[0], t[1]))
    out = tuple((k, float(t)) for t, _, k in scored)
    return out[:topk] if topk else out


def best_record(
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    sm_count: Optional[int] = None,
    family: Optional[str] = None,
    in_fmt: str = "bf16",
    nominal: bool = False,
) -> Knobs:
    """The dispatcher's pick: the first of `rank` (a member of the enumerated space by construction)."""
    return rank(
        M,
        N,
        K,
        chunks,
        b_major,
        out_bytes,
        sm_count,
        family,
        in_fmt=in_fmt,
        nominal=nominal,
    )[0][0]
