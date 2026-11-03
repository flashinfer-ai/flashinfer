# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The dispatcher of the batch-invariant GEMM classes: a ranking heuristic over the class's configuration space.

For a shape, layout, format and part it orders the records of the enumerated space by a per-tile roofline estimate
(SM-cycles of the launch) and `schedule.default_config` takes the first; `schedule.rank_topk` hands the first ``k`` to
FlashInfer's autotuner. Nothing here reaches the bits: every record of the space computes the class (R3), the heuristic
only orders them, so it may be as rough as it is.

The estimate uses the part's nominal numbers (SM count, co-resident cluster capacity per cluster size, the dense bf16
MMA rate per SM, the L2 -> SMEM and HBM bandwidths per SM) and a few shape constants (`CONSTANTS`: per-CTA row
efficiency, the transaction efficiency of narrow B boxes, the multicast benefit, the shared-L2 fraction, per-step and
fixup floors, raster / swizzle / 2-CTA adjustments). Every number is empirical and shared by all parts and formats;
none is fitted per part. The format enters through its structure only: element width, MMA K step and the 8-bit kind's
doubled instruction rate.
"""

from __future__ import annotations

import functools
import logging
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from .generator import Knobs, enumerate_space
from .spec import ELEM_BITS_OF, MMA_K_OF

logger = logging.getLogger(__name__)

# ----------------------------------------------------------------------------- the part


@dataclass(frozen=True)
class Part:
    """A part's nominal numbers. ``family`` is the compute capability's family (sm_107 / sm_100 / sm_103); ``capacity`` the
    co-resident cluster count per cluster size at the nominal SM count (the driver's GPC-aware occupancy); the rates are per
    SM per cycle, dense bf16 for the MMA."""

    family: str
    sm_count: int
    capacity: Tuple[Tuple[int, int], ...]
    mma_flop_per_clk: float = 8192.0
    load_bytes_per_clk: float = 64.0
    hbm_bytes_per_clk: float = 32.0


_RUBIN_CAPACITY = ((1, 204), (2, 102), (4, 42), (8, 20))
_B200_CAPACITY = ((1, 148), (2, 74), (4, 33), (8, 15))

PART_TABLE: Dict[str, Part] = {
    "sm_107": Part("sm_107", 204, _RUBIN_CAPACITY, hbm_bytes_per_clk=32.0),
    "sm_100": Part("sm_100", 148, _B200_CAPACITY, hbm_bytes_per_clk=28.0),
    "sm_103": Part("sm_103", 148, _B200_CAPACITY, hbm_bytes_per_clk=28.0),
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
    """The architecture family of a compute capability; the API passes the operand device's."""
    return FAMILY_OF_CC.get((int(cc[0]), int(cc[1])), "sm_107")


# the co-resident cluster counts a device reported at run time, keyed (family, SM count); device-free callers use the table
_QUERIED_CAPACITY: Dict[Tuple[str, int], Dict[int, int]] = {}


def capacity_for(family: str, sm_count: Optional[int]) -> Dict[int, int]:
    """Co-resident clusters per cluster size for the part: the device's own numbers when the API has queried them in this
    process, else the family's nominal table scaled to ``sm_count``."""
    part = PART_TABLE.get(family, PART_TABLE["sm_107"])
    sms = sm_count or part.sm_count
    q = _QUERIED_CAPACITY.get((family, sms))
    if q:
        return q
    return {
        size: max(1, round(cap * sms / part.sm_count)) for size, cap in part.capacity
    }


def register_capacity(family: str, sm_count: int, size: int, clusters: int) -> None:
    """Record a device's co-resident cluster count for one cluster size (from `api.co_resident_clusters`)."""
    cap = _QUERIED_CAPACITY.setdefault(
        (family, sm_count), dict(capacity_for(family, sm_count))
    )
    cap[size] = clusters
    rank.cache_clear()


# ----------------------------------------------------------------------------- the estimate

#: Shape constants of the estimate (empirical, shared by every part and format).
CONSTANTS: Dict[str, float] = dict(
    rows64_eff=0.8,  # MMA efficiency of a 64-row CTA tile relative to 128 rows
    eff16b=0.15,  # transaction efficiency of a B box 16 bytes wide ...
    eff32b=0.2,  # ... 32 bytes ...
    eff64b=0.5,  # ... 64 bytes (128 and wider: 1)
    mcast_pow=1.25,  # multicast divides an operand's traffic by cluster_dim ** mcast_pow
    l2_frac=1.0,  # the shared L2 fabric delivers this fraction of SMs x load bandwidth in aggregate
    step_floor=32.0,  # cycles per k-step no tile can beat
    fixup_floor=8000.0,  # cycles of the workspace split's fixup launch
    raster_pow=0.1,  # the wrong raster: (tile-count imbalance) ** raster_pow
    raster_min=4.0,  # ... once the tile-count imbalance reaches this ratio
    overrun_pow=0.65,  # a swizzle group larger than its axis: (swizzle / tiles) ** overrun_pow
    sw_pen=1.1,  # swizzle > 1 where it cannot help
    sw_gain=1.0,  # swizzle > 1 with many tiles in flight on both axes
    sw_partial=1.05,  # a swizzle that does not divide its axis (a partial last group)
    two_cta=0.9,  # the 2-CTA pair's shared B and halved per-CTA issue
)

DECODE_M = 64  # up to this many rows the candidates are the `decode_sane` subset


def decode_sane(k: Knobs, M: int) -> bool:
    """The candidate guard at M <= DECODE_M (structural): 1-CTA tiles of N >= 32 in an N-cluster at swizzle 1. Records
    outside the guard stay members of the space (the autotuner may still reach them)."""
    if M > DECODE_M:
        return True
    return (
        (not k.use_2cta_instrs)
        and k.tile_n >= 32
        and k.cluster_m == 1
        and k.swizzle_size == 1
    )


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
) -> float:
    """SM-cycles of the launch for record ``k`` at the shape: waves of CTA tiles over the part's co-resident cluster
    capacity, each the slower of its MMA and its L2 -> SMEM traffic per k-step plus a floor, bounded below by the shared
    L2 fabric and by the unique HBM bytes, plus the epilogue and (workspace split) the fixup; raster, swizzle and 2-CTA
    adjustments multiply. Lower is better; ``inf`` when the record cannot tile the shape."""
    p = c or CONSTANTS
    part = PART_TABLE.get(family, PART_TABLE["sm_107"])
    sms = sm_count or part.sm_count
    cap = capacity_for(family, sms)
    tm, tn = k.mma_tiler_mn
    cm, cn = k.cluster_shape_mn
    group = 2 if k.use_2cta_instrs else 1
    rows = tm // group
    eb = ELEM_BITS_OF[in_fmt] / 8.0
    kk = MMA_K_OF[in_fmt]
    rate = part.mma_flop_per_clk * (
        kk / 16.0
    )  # the 8-bit kind (K step 32): twice the dense bf16 rate
    m_tiles = -(-M // tm)
    n_tiles = -(-N // tn)
    if m_tiles == 0 or n_tiles == 0:
        return float("inf")
    ws = (
        chunks > 1 and not k.cta_chunks
    )  # workspace split: `chunks` work items per tile and a fixup kernel
    items = chunks if ws else 1
    ctas_m = m_tiles * group
    pad_m = -(-ctas_m // cm) * cm
    pad_n = -(-n_tiles // cn) * cn
    clusters = pad_m * pad_n // (cm * cn) * items
    size = cm * cn
    capacity = cap.get(size) or max(1, sms // size)
    waves = -(-clusters // capacity)
    ksteps = max(1.0, (K / (chunks if ws else 1)) / kk)
    flops = 2.0 * rows * tn * kk
    mma_eff = 1.0 if rows >= 128 else p["rows64_eff"]
    mma_cycles = flops / (rate * mma_eff)
    wb = tn / group * eb
    b_eff = (
        p["eff16b"]
        if wb <= 16
        else p["eff32b"]
        if wb <= 32
        else p["eff64b"]
        if wb <= 64
        else 1.0
    )
    a_bytes = rows * kk * eb / (cn ** p["mcast_pow"])
    b_bytes = tn * kk * eb / (cm ** p["mcast_pow"]) / b_eff
    step = (
        max(mma_cycles, (a_bytes + b_bytes) / part.load_bytes_per_clk) + p["step_floor"]
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
    unique = (M * K + K * N) * eb + M * N * out_bytes
    t = max(body + epi, unique / (part.hbm_bytes_per_clk * sms)) + fix
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
    if k.use_2cta_instrs:
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
) -> Tuple[Tuple[Knobs, float], ...]:
    """The records of the class's enumerated space for the shape, layout and format (at M <= DECODE_M the `decode_sane`
    subset) ordered by the estimate, best first; ``topk`` truncates. ``arch_name`` is accepted for the former signature and
    ignored. Cached per argument tuple."""
    fam = family or "sm_107"
    space = enumerate_space(M, N, K, chunks=chunks, b_major=b_major, in_fmt=in_fmt)
    guarded = [k for k in space if decode_sane(k, M)]
    space = guarded or space
    if not space:
        return ()
    scored = [
        (
            estimate_cycles(
                k, M, N, K, chunks, b_major, in_fmt, fam, sm_count, out_bytes
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
) -> Knobs:
    """The dispatcher's pick: the first of `rank` (a member of the enumerated space by construction)."""
    return rank(M, N, K, chunks, b_major, out_bytes, sm_count, family, in_fmt=in_fmt)[
        0
    ][0]
