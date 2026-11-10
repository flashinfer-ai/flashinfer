# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Cost model of the bf16 GEMM records: the dispatcher behind `mm_bf16(batch_invariant=True)`.

For one knob record at one shape the model predicts the device time (us) from the part's structure (`Arch`) and fitted
constants (`Fit`); `rank` evaluates a whole configuration space at once and `schedule.default_config` takes the argmin.
Nothing here reaches the bits: every record of the space computes the class (R3), the model only orders them.

The time is a soft maximum of four bounds -- the per-CTA tile time (waves of tiles, each the slowest of the MMA, TMA,
smem and latency paths per k-tile, plus the epilogue), the HBM traffic under the persistent raster's L2 footprint rule,
the L2 -> SM traffic, and the small-M hot-line bound -- plus the launch floor and, for the workspace split, the fixup node.

Every numeric constant in this file (the `Arch` entries, the `Fit` defaults, `DECODE_M`) and every value in
`cost_model_fits.json` is empirical: chosen or fitted from timings of these kernels on the parts, not derived from first
principles. The fits are stored per architecture family; the SM count is a parameter within a family.
"""

from __future__ import annotations

import functools
import json
import os
import warnings
from dataclasses import asdict, dataclass, replace
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

from .generator import Knobs, inner_radix
from .spec import ELEM_BITS, MMA_K

# ----------------------------------------------------------------------------- the part


@dataclass(frozen=True)
class Arch:
    """A part's numbers (empirical). ``family`` keys the fitted constants in `cost_model_fits.json`; ``placeholder`` marks a
    part whose numbers are public figures only."""

    name: str
    sm_count: int
    clock_ghz: float  # sustained SM clock under load
    flop_per_clk_sm: int = 8192  # dense bf16 FLOP / clk / SM
    hbm_tbps: float = 9.1  # copy bandwidth, TB/s
    l2_bytes: int = 112 << 20
    smem_bytes: int = 334848  # shared memory per CTA the kernel plans against (`cutlass.memory.get_smem_capacity_in_bytes`)
    tmem_cols: int = 512  # TMEM columns the kernel plans against
    placeholder: bool = False
    family: str = "sm_107"  # the fit's key (sm_107 / sm_100 / sm_103)


# The parts, as (family, SM count); the SM count is the only difference the model sees within a family.


def _sm100_part(
    name: str, sm_count: int, family: str, placeholder: bool = False
) -> Arch:
    """A part with the SM100-generation memory system (the sm_103 parts are placeholders: public figures only)."""
    return Arch(
        name,
        sm_count,
        1.55,
        hbm_tbps=7.4,
        l2_bytes=126 << 20,
        smem_bytes=232448,
        placeholder=placeholder,
        family=family,
    )


ARCHS: Dict[str, Arch] = {
    "sm_107": Arch("sm_107", 204, 2.3),
    "sm_107_224": Arch("sm_107_224", 224, 2.3),
    "sm_100": _sm100_part("sm_100", 148, "sm_100"),
    "sm_100_152": _sm100_part("sm_100_152", 152, "sm_100"),
    "sm_103": _sm100_part("sm_103", 148, "sm_103", placeholder=True),
    "sm_103_152": _sm100_part("sm_103_152", 152, "sm_103", placeholder=True),
}
#: The parts as (family, SM count): what the dispatcher serves, what the generated Lean cases and the tests enumerate.
PARTS = tuple((a.family, a.sm_count) for a in ARCHS.values())
FAMILY_OF_CC = {(10, 0): "sm_100", (10, 3): "sm_103", (10, 7): "sm_107"}
FIT_FALLBACK = {
    "sm_103": "sm_100",
    "sm_100": "sm_107",
}  # a family without its own fit takes the nearest family's


def family_of(cc: Tuple[int, int]) -> str:
    """The architecture family of a compute capability (the fits' key); the API passes the operand device's."""
    return FAMILY_OF_CC.get((int(cc[0]), int(cc[1])), "sm_107")


def arch_for(
    sm_count: Optional[int] = None,
    name: Optional[str] = None,
    family: Optional[str] = None,
) -> Arch:
    """The part the dispatcher models: by name; else the ``family`` (the operand device's, `family_of`; sm_107 when not given) at
    ``sm_count`` SMs (204 when not given). A (family, SM count) with an entry in `ARCHS` is that part; any other count is the
    family's canonical part with the SM count replaced (an emulated SKU: `name = <family>_<count>`, e.g. ``sm_107_148`` for a
    148-SM grid cap)."""
    if name:
        return ARCHS[name]
    family = family or "sm_107"
    if sm_count is None:
        sm_count = 204
    for a in ARCHS.values():
        if a.family == family and a.sm_count == sm_count:
            return a
    base = next((a for a in ARCHS.values() if a.family == family), ARCHS["sm_107"])
    return replace(base, name=f"{family}_{sm_count}", sm_count=sm_count)


# ----------------------------------------------------------------------------- the fitted constants


@dataclass(frozen=True)
class Fit:
    """The fitted constants (all empirical; the defaults are placeholders that the entries of `cost_model_fits.json`
    replace). Per-SM constants carry across the parts of a family; part-level ones (the aggregate L2, the HBM efficiency,
    the launch and fixup floors, `softmax_p`, `cl4_ns`, `oob_gbps`) are fitted per family."""

    mma_eff: float = 0.92  # sustained fraction of the MMA instruction rate
    inst_min_clk: float = 18.0  # minimum time of one tcgen05.mma, in SM clocks
    ktile_ns: float = 60.0  # per k-tile per CTA: TMA issue + mbarrier hand-shakes
    ktile_2cta_ns: float = 60.0  # the same for the 2-CTA instruction
    fetch_gbps: float = 140.0  # per-SM TMA fetch from L2 (this CTA's multicast share)
    hot_gbps: float = 120.0  # L2 bandwidth of one line's home slice (the small-M bound)
    oob_gbps: float = 70.0  # TMA zero-fill rate of out-of-bounds A rows
    cl4_ns: float = 70.0  # per k-tile cost of a cluster of four or more CTAs
    miss_cost: float = (
        2.0  # L2 bandwidth per byte that misses to HBM, on top of the hit traffic
    )
    softmax_p: float = 3.0  # the bounds combine as a p-norm (p -> inf: a pure max)
    ingress_gbps: float = 220.0  # per-SM smem ingress of the stage
    smem_gbps: float = 300.0  # per-SM smem bandwidth shared by the stage's writes and the MMA's operand reads
    l2_tbps: float = 28.0  # aggregate L2 -> SM bandwidth, TB/s
    hbm_eff: float = 0.85  # fraction of the copy bandwidth a streaming GEMM reaches
    latency_us: float = 1.3  # memory latency behind the ring
    launch_us: float = 3.0  # kernel floor: launch, prologue, tail
    fill_us: float = 0.7  # the first stage's latency per persistent CTA
    tile_us: float = (
        0.25  # per tile: scheduler step, accumulator hand-off, store commit
    )
    epi_gbps: float = (
        120.0  # epilogue bandwidth per SM (TMEM -> registers -> smem -> TMA store)
    )
    fixup_us: float = 2.5  # the split class's fixup node: launch gap + its own floor
    cluster_pen: float = 0.02  # coupling penalty per extra CTA in the cluster
    l2_frac: float = 0.8  # usable fraction of the L2 for the residency rule
    nmajor_pen: float = 1.2  # multiplier on the line-granular B traffic of the `n` layout when the B box row is under 128 bytes
    two_cta_gain: float = 1.0  # per-k-tile time multiplier of the 2-CTA instruction


FITS_PATH = os.path.join(os.path.dirname(__file__), "cost_model_fits.json")


def _load_fits() -> Dict[str, Fit]:
    out: Dict[str, Fit] = {}
    if not os.path.exists(FITS_PATH):
        # the dispatcher then runs on the first-principles defaults: the same bits (R3), another speed profile
        warnings.warn(
            f"{FITS_PATH} is missing: the batch-invariant GEMM dispatcher falls back to its default constants "
            "(correct bits, untuned configuration choices); the file ships with the package",
            RuntimeWarning,
            stacklevel=2,
        )
        return out
    with open(FITS_PATH) as f:
        for name, d in json.load(f).items():
            # a family's entry: its constants plus a `_meta` provenance block (part, date, driver, method)
            out[name] = Fit(
                **{k: v for k, v in d.items() if k in Fit.__dataclass_fields__}
            )
    return out


FITS: Dict[str, Fit] = _load_fits()


def fit_for(arch: Arch) -> Fit:
    """The family's entry of `cost_model_fits.json`, else the nearest family's (`FIT_FALLBACK`), else the defaults."""
    fam: Optional[str] = arch.family
    while fam:
        if fam in FITS:
            return FITS[fam]
        fam = FIT_FALLBACK.get(fam)
    return FITS.get("sm_107") or Fit()


# ----------------------------------------------------------------------------- the model


@functools.lru_cache(maxsize=None)
def _epi_tile(ctm: int, ctn: int, two: bool, out_bytes: int) -> Tuple[int, int]:
    """The epilogue subtile the body stores through (the library's rule, pure Python), (rows, cols)."""
    try:
        from cutlass.utils.blackwell_helpers import compute_epilogue_tile_size

        rows, cols = compute_epilogue_tile_size(
            ctm, ctn, two, out_bytes * 8, None, False, True
        )
        return (int(rows), int(cols))
    except Exception:  # noqa: BLE001  (no DSL at hand)
        return (ctm, min(ctn, 32 if ctm == 128 else 64))


def knob_arrays(recs: Sequence[Knobs]) -> Dict[str, np.ndarray]:
    """The record fields the model reads, as arrays (one entry per record)."""
    return dict(
        tm=np.array([k.tile_m for k in recs], dtype=np.float64),
        tn=np.array([k.tile_n for k in recs], dtype=np.float64),
        cm=np.array([k.cluster_m for k in recs], dtype=np.float64),
        cn=np.array([k.cluster_n for k in recs], dtype=np.float64),
        two=np.array([k.use_2cta_instrs for k in recs], dtype=bool),
        rm=np.array([k.raster_along == "m" for k in recs], dtype=bool),
        sw=np.array([k.swizzle_size for k in recs], dtype=np.float64),
        cc=np.array([k.cta_chunks for k in recs], dtype=bool),
    )


def predict_arrays(
    a: Dict[str, np.ndarray],
    M,
    N,
    K,
    chunks,
    b_major: str,
    out_bytes: int,
    arch: Arch,
    fit: Fit,
    parts: bool = False,
):
    """The model on arrays: one entry per (record, shape); ``M``, ``N``, ``K``, ``chunks`` scalars or arrays matching the records."""
    M = np.asarray(M, dtype=np.float64)
    N = np.asarray(N, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    chunks = np.asarray(chunks, dtype=np.float64)
    elem = ELEM_BITS / 8.0
    inst_k = MMA_K
    inner = inner_radix()
    kb = inst_k * inner  # K elements per 128-byte stage
    fpc = arch.flop_per_clk_sm
    two = a["two"]
    cta_div = np.where(two, 2.0, 1.0)
    ctm = a["tm"] / cta_div  # the CTA's rows
    ctn = a["tn"]
    cm, cn = a["cm"], a["cn"]
    cl = cm * cn
    pairs_m = (
        cm / cta_div
    )  # MMA tiles (CTA pairs) along M in the cluster: B is multicast across them
    split = chunks > 1
    ws = (
        split & ~a["cc"]
    )  # the workspace split (C units per output tile) vs the in-CTA fold
    nchunk_cta = np.where(a["cc"] & split, chunks, 1.0)
    Kc = K / chunks

    # tiles, cluster units, waves: the scheduler rounds the cluster grid up to the swizzle along the raster's minor axis, so
    # ghost units (zero-filled A) cost a tile time like real ones
    m_tiles = np.ceil(M / a["tm"])
    n_tiles = np.ceil(N / ctn)
    m_cl = np.ceil(m_tiles / pairs_m)
    n_cl = np.ceil(n_tiles / cn)
    s = a["sw"]
    rm = a["rm"]
    n_cl = np.where(rm, np.ceil(n_cl / s) * s, n_cl)
    m_cl = np.where(rm, m_cl, np.ceil(m_cl / s) * s)
    units = m_cl * n_cl * np.where(ws, chunks, 1.0)
    P = np.floor(arch.sm_count / cl)
    waves = np.ceil(units / P)
    k_tiles = np.ceil(Kc / kb) * nchunk_cta  # k-tiles per tile per CTA

    # per k-tile time
    mma_eff = fit.mma_eff
    t_inst = np.maximum(
        2.0 * ctm * ctn * inst_k / (fpc * arch.clock_ghz) / mma_eff,
        fit.inst_min_clk / arch.clock_ghz,
    )  # ns
    t_mma_kt = inner * t_inst * np.where(two, fit.two_cta_gain, 1.0)
    a_stage = ctm * kb * elem
    b_stage = ctn * kb * elem / cta_div
    # the `n` layout's B box: a row narrower than a 128-byte line still costs the line (times a fitted penalty)
    b_row = ctn * elem / cta_div
    b_eff = np.where(
        (b_major == "n") & (b_row < 128), kb * 128.0 * fit.nmajor_pen, b_stage
    )
    out_b = np.where(ws, 4.0, float(out_bytes))
    ep = np.array(
        [
            _epi_tile(int(m_), int(n_), bool(t_), int(o_))
            for m_, n_, t_, o_ in zip(ctm, ctn, two, out_b, strict=False)
        ],
        dtype=np.float64,
    )
    c_stage = ep[:, 0] * ep[:, 1] * out_b
    stages = np.maximum(
        1.0, np.floor((arch.smem_bytes - 1024 - 2 * c_stage) / (a_stage + b_stage))
    )
    t_ingress_kt = (
        a_stage + b_eff
    ) / fit.ingress_gbps  # bytes / (GB/s) = ns: every byte landing in the ring
    # smem bandwidth shared by the stage's writes and the MMA's operand reads (`inner` instructions per k-tile)
    mma_reads = inner * inst_k * elem * (ctm + ctn / cta_div)
    t_smem_kt = (a_stage + b_eff + mma_reads) / fit.smem_gbps
    t_fetch_kt = (
        a_stage / cn + b_eff / pairs_m
    ) / fit.fetch_gbps  # the bytes this CTA's TMA unit requests (multicast shares)
    t_lat_kt = fit.latency_us * 1e3 / stages
    ktile = np.where(two, fit.ktile_2cta_ns, fit.ktile_ns)
    # zero-fill of the A box's rows past M, split among an N-cluster's CTAs
    oob_rows = np.where(
        two,
        np.where(ctm >= M, ctm, np.maximum(0.0, 2 * ctm - M)),
        np.maximum(0.0, ctm - M),
    )
    oob_rows = np.minimum(oob_rows, ctm)
    t_oob_kt = oob_rows * 128.0 / cn / fit.oob_gbps
    # the TMA path pays its per-k-tile costs serially with its fetch; clusters of four or more pay their fan-in on top
    t_kt = np.maximum.reduce(
        [t_mma_kt, t_fetch_kt + t_oob_kt + ktile, t_ingress_kt, t_smem_kt, t_lat_kt]
    ) + np.where(cl >= 4, fit.cl4_ns, 0.0)
    t_main = k_tiles * t_kt / 1e3  # us per tile per CTA

    # epilogue, accumulator stages, the in-CTA fold's passes
    t_epi = ctm * ctn * out_b / fit.epi_gbps / 1e3 + fit.tile_us
    one_stage = a["cc"] & split & ((3 * ctn > arch.tmem_cols) | (ctm == 64))
    t_fold = np.where(
        a["cc"] & split, nchunk_cta * ctm * ctn * 8.0 / fit.epi_gbps / 1e3, 0.0
    )
    t_tile = np.where(
        one_stage, t_main + t_epi + t_fold, np.maximum(t_main, t_epi + t_fold)
    )
    t_cta = fit.fill_us + waves * t_tile + t_epi

    # aggregate traffic: L2 -> SM (multicast shares the reads) and HBM (the raster's footprint rule)
    l2_bytes = units * cl * k_tiles * (a_stage / cn + b_stage / pairs_m)
    rows_cl = a["tm"] * pairs_m
    cols_cl = ctn * cn
    A_total = M * K * elem
    B_total = N * K * elem
    l2_eff = fit.l2_frac * arch.l2_bytes
    K_unit = (
        Kc * nchunk_cta
    )  # the K extent one unit streams (all chunks for the in-CTA fold, one for the workspace split)
    maj_cl = np.where(
        rm, m_cl, n_cl
    )  # the long axis walked inside a group (all of it per group)
    min_cl = np.where(rm, n_cl, m_cl)  # the minor axis: s columns of it form a group
    walk_dim = np.where(
        rm, rows_cl, cols_cl
    )  # rows / columns of one cluster tile along each axis
    fix_dim = np.where(rm, cols_cl, rows_cl)
    G = s * maj_cl  # units per group
    n_groups = np.ceil(min_cl / s)
    waves_per_group = np.maximum(1.0, G / P)
    walked_total = np.where(
        rm, A_total, B_total
    )  # the operand walked along the major axis (A for raster m)
    fixed_total = np.where(
        rm, B_total, A_total
    )  # the operand a group holds fixed (B for raster m)
    walk_fp = (
        np.minimum(maj_cl, np.ceil(P / s)) * walk_dim * K_unit * elem
    )  # the walked operand's footprint of one wave
    group_fp = (
        np.minimum(s, min_cl) * fix_dim * K_unit * elem
    )  # one group's fixed operand
    # the walked operand streams once per group pass; a pass after the first misses the L2 for what does not fit
    passes = np.minimum(n_groups, waves)
    miss_frac = np.clip(
        (walked_total + group_fp - l2_eff) / np.maximum(walked_total, 1.0), 0.0, 1.0
    )
    walked_hbm = walked_total * (1.0 + (passes - 1.0) * miss_frac)
    # a group's fixed operand stays resident across the group's waves when it fits next to the walked footprint
    fixed_hbm = fixed_total * np.where(
        group_fp + walk_fp <= l2_eff, 1.0, waves_per_group
    )
    # the workspace split walks one chunk's tiles after another: the per-chunk footprint rule times C
    units_c = (
        m_cl * n_cl
    )  # one chunk's units: the passes and groups of one chunk's walk
    waves_c = np.ceil(units_c / P)
    passes_c = np.minimum(n_groups, waves_c)
    walked_c = walked_total / chunks  # one chunk's slice of the walked operand
    fixed_c = fixed_total / chunks
    miss_c = np.clip(
        (walked_c + group_fp - l2_eff) / np.maximum(walked_c, 1.0), 0.0, 1.0
    )
    walked_ws = chunks * walked_c * (1.0 + (passes_c - 1.0) * miss_c)
    fixed_ws = (
        chunks
        * fixed_c
        * np.where(group_fp + walk_fp <= l2_eff, 1.0, np.maximum(1.0, G / P))
    )
    walked_hbm = np.where(ws, walked_ws, walked_hbm)
    fixed_hbm = np.where(ws, fixed_ws, fixed_hbm)
    out_hbm = M * N * out_bytes
    ws_hbm = np.where(ws, 2.0 * chunks * M * N * 4.0, 0.0)  # partials out and back
    hbm_bytes = walked_hbm + fixed_hbm + out_hbm + ws_hbm
    hbm_bw = fit.hbm_eff * arch.hbm_tbps * 1e6  # bytes / us
    t_hbm = hbm_bytes / hbm_bw
    t_l2 = (l2_bytes + fit.miss_cost * hbm_bytes) / (
        fit.l2_tbps * 1e6
    )  # misses cost the L2 their fill and their read
    # the hot-line bound: with M valid rows the machine's A requests land on M L2 home slices per k-tile
    a_rows_valid = np.minimum(M, ctm)
    a_req_bytes = units * (cl / cn) * k_tiles * a_rows_valid * 128.0
    hot_rate = np.minimum(
        fit.hot_gbps * a_rows_valid * np.where(ws, chunks, 1.0), fit.l2_tbps * 1e3
    )  # GB/s
    t_hot = a_req_bytes / (hot_rate * 1e3)  # us
    bounds = np.stack([t_cta, t_hbm, t_l2, t_hot])
    t_body = np.power(
        np.sum(np.power(bounds, fit.softmax_p), axis=0), 1.0 / fit.softmax_p
    )  # a soft max: close bounds interfere
    t_kernel = (fit.launch_us + t_body) * (1.0 + fit.cluster_pen * (cl - 1.0))
    t_fixup = np.where(
        ws, fit.fixup_us + (chunks * M * N * 4.0 + out_hbm) / hbm_bw, 0.0
    )
    t = t_kernel + t_fixup
    if not parts:
        return t
    return t, dict(
        units=units,
        waves=waves,
        stages=stages,
        k_tiles=k_tiles,
        t_mma_kt=t_mma_kt,
        t_ingress_kt=t_ingress_kt,
        t_smem_kt=t_smem_kt,
        t_fetch_kt=t_fetch_kt,
        t_oob_kt=t_oob_kt,
        t_lat_kt=t_lat_kt,
        t_kt=t_kt,
        t_main=t_main,
        t_epi=t_epi,
        t_cta=t_cta,
        t_hbm=t_hbm,
        t_l2=t_l2,
        t_hot=t_hot,
        hbm_bytes=hbm_bytes,
        l2_bytes=l2_bytes,
        t_fixup=t_fixup,
    )


def predict_many(
    recs: Sequence[Knobs],
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
    parts: bool = False,
):
    """Predicted device time (us) of every record of ``recs`` at (M, N, K); ``parts=True`` also returns the terms."""
    arch = arch or arch_for()
    fit = fit or fit_for(arch)
    return predict_arrays(
        knob_arrays(recs), M, N, K, chunks, b_major, out_bytes, arch, fit, parts
    )


def predict(
    k: Knobs,
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
) -> float:
    """Predicted device time (us) of one record at one shape."""
    return float(predict_many([k], M, N, K, chunks, b_major, out_bytes, arch, fit)[0])


def explain(
    k: Knobs,
    M: int,
    N: int,
    K: int,
    chunks: int = 1,
    b_major: str = "n",
    out_bytes: int = 2,
    arch: Optional[Arch] = None,
    fit: Optional[Fit] = None,
) -> Dict[str, float]:
    """The predicted time and every term of one record (for the docs and the tests)."""
    t, p = predict_many([k], M, N, K, chunks, b_major, out_bytes, arch, fit, parts=True)
    return dict(us=float(t[0]), **{n: float(v[0]) for n, v in p.items()})


# ----------------------------------------------------------------------------- the dispatcher's entry

DECODE_M = 64  # up to this many rows the candidates are the `decode_sane` subset


def decode_sane(k: Knobs, M: int) -> bool:
    """The candidate guard at M <= DECODE_M (a heuristic, not a space change): 1-CTA tiles of N >= 32 in an N-cluster at
    swizzle 1. Records outside the guard stay members of the space."""
    if M > DECODE_M:
        return True
    return (
        (not k.use_2cta_instrs)
        and k.tile_n >= 32
        and k.cluster_m == 1
        and k.swizzle_size == 1
    )


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
) -> Tuple[Tuple[Knobs, float], ...]:
    """The records of the class's enumerated space for the shape and layout (at M <= DECODE_M the `decode_sane` subset) ordered
    by predicted time, best first; ``topk`` truncates. Cached per argument tuple."""
    from .generator import enumerate_space

    arch = arch_for(sm_count, arch_name, family)
    space = enumerate_space(M, N, K, chunks=chunks, b_major=b_major)
    guarded = [k for k in space if decode_sane(k, M)]
    space = guarded or space
    if not space:
        return ()  # nothing implements the shape in this layout / class: the caller decides
    t = predict_many(space, M, N, K, chunks, b_major, out_bytes, arch)
    order = np.argsort(t, kind="stable")
    out = tuple((space[i], float(t[i])) for i in order)
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
) -> Knobs:
    """The model's pick: the argmin of `rank` (a member of the enumerated space by construction)."""
    return rank(M, N, K, chunks, b_major, out_bytes, sm_count, family)[0][0]


def fit_to_dict(fit: Fit) -> Dict[str, float]:
    return asdict(fit)
