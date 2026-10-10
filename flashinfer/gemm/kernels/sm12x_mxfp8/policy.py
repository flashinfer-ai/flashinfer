# SPDX-FileCopyrightText: Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Tactics of the SM12x MXFP8 ``cute-dsl`` backend.

A tactic is a flat tuple ``(family, schedule, *config)``:

- ``("gemv", "rows", mb, lpr, warps, ctas_per_sm, k_split)``: CUDA-core
  GEMV for M <= mb (gemv.py).
- ``("skinny", "streamk", nt, rt, ku, p, warps, ctas_per_sm, min_units,
  cta_wide)``: stream-K over the weight with warp-level block-scaled MMA
  (skinny.py).
- ``("persistent", "dp" | "streamk", bm, bn, wm, wn, ks, kw, sa)``:
  warp-specialized persistent kernel (persistent.py); ``streamk`` splits the
  K iterations of the last partial wave over the grid.
- ``("pingpong", "dp", tile_n, tile_k, epi_n, epi_stages, group, coop, early)``:
  persistent ping-pong kernel (pingpong.py).

Every schedule is deterministic. M is a runtime value for all kernels; each
candidate list is built for one autotuner bucket and stays valid for every M
that maps to it. There are no per-shape tables: candidates and the default
tactic are derived from (M, N, K) and the device (SM count, L2 size).
"""

from typing import NamedTuple

from .common import SMEM_BYTES, ceil_div
from .persistent import max_b_stages

VERSION = "sm12x_mxfp8_v1"

# Largest M served by the skinny family (the GEMV serves M <= its mb).
SKINNY_MAX_M = 32


class Device(NamedTuple):
    sms: int
    l2_bytes: int


# ------------------------------------------------------------------ gemv
def _gemv_lanes(k):
    """Widest lane group that still gives every lane one 16-byte step."""
    lpr = 32
    while lpr > 8 and k < 16 * lpr:
        lpr //= 2
    return lpr


def gemv(mb, k, warps=8, ctas_per_sm=4, k_split=1):
    return ("gemv", "rows", mb, _gemv_lanes(k), warps, ctas_per_sm, k_split)


def gemv_valid(tactic, n, k):
    _, _, mb, lpr, warps, _, ks = tactic
    if warps % ks or k % (16 * ks):
        return False
    steps = ceil_div(k // ks, 16 * lpr)
    return steps <= 16 and mb * k * 2 <= 96 * 1024


def gemv_grid(tactic, n, dev):
    _, _, _, lpr, warps, cps, ks = tactic
    row_groups = ceil_div(n, 32 // lpr)
    return max(1, min(dev.sms * cps, ceil_div(row_groups, warps // ks)))


def _gemv_default(mb, n, k):
    """Split one row's K over warps when there are few rows per SM and enough
    K per lane: small N, larger M. Never below 320 K elements per warp."""
    ks = 1
    if k >= 1024:
        cap = 12288 if mb >= 4 else (1280 if mb == 2 else 256)
        while ks < 8 and n * ks * 2 <= cap and k // (ks * 2) >= 320:
            ks *= 2
    if ks >= 8:
        return gemv(mb, k, 16, 2, ks)
    return gemv(mb, k, 8, 4, ks)


# ---------------------------------------------------------------- skinny
def skinny(tm, rt, warps=4, ctas_per_sm=4, min_units=1, cta_wide=False, ku=4, p=1):
    nt = max(1, tm // 8)
    return ("skinny", "streamk", nt, rt, ku, p, warps, ctas_per_sm, min_units, cta_wide)


def skinny_smem(tactic):
    _, _, nt, rt, ku, _, warps, _, _, cta_wide = tactic
    if cta_wide:
        return 2 * 8 * nt * (ku * 33 + 16) + 16
    return warps * 2 * rt * nt * 512


def skinny_valid(tactic, n, k):
    _, _, nt, _, ku, _, warps, _, _, cta_wide = tactic
    if cta_wide and warps * 32 < 8 * nt * ku:
        return False
    return skinny_smem(tactic) <= SMEM_BYTES


def skinny_grid(tactic, n, k, dev):
    _, _, _, rt, ku, _, warps, cps, min_units, cta_wide = tactic
    rows = 16 * rt * (warps if cta_wide else 1)
    units = ceil_div(n, rows) * ceil_div(k // 32, ku)
    per_cta = min_units if cta_wide else warps * min_units
    return min(dev.sms * cps, max(1, units // per_cta), units)


def _bucket(m):
    return 1 << max(0, (m - 1).bit_length())


def _decode_default(m, n, k):
    """GEMV / skinny default for M <= SKINNY_MAX_M.

    M <= 4 is a pure weight stream: the CUDA-core GEMV. M 5..8 on large
    weights with K <= 2560: the 8-token GEMV. Otherwise the stream-K MMA
    kernel, with 32-token tiles for large weights and M 17..32.
    """
    b = _bucket(m)
    big = n * k >= 8 * 1024 * 1024
    if b <= 4:
        for t in (_gemv_default(b, n, k), gemv(b, k)):
            if gemv_valid(t, n, k):
                return t
        return skinny(8, 2)
    if b == 8:
        if big and k <= 2560:
            t = gemv(8, k)
            if gemv_valid(t, n, k):
                return t
        return skinny(8, 1) if big else skinny(8, 2 if n >= 512 else 1)
    if b == 16:
        if big:
            return skinny(32, 2, min_units=2)
        if (k // 32) % 4 and n >= 512:
            # K % 128 != 0: two-block units avoid a half-empty last unit per row tile.
            return skinny(16, 2, min_units=2, ku=2, p=4)
        return skinny(16, 2 if n >= 512 else 1)
    if n <= 256:
        return skinny(32, 1)
    if k <= 1024:
        t = skinny(32, 1, warps=8, ctas_per_sm=1)
        if skinny_valid(t, n, k):
            return t
    if big:
        return skinny(32, 2, min_units=2)
    return skinny(32, 1, ctas_per_sm=2, min_units=2)


def _decode_candidates(m, n, k):
    b = _bucket(m)
    out = [_decode_default(m, n, k)]
    if b <= 4:
        out += [
            gemv(b, k),
            gemv(b, k, 4, 8),
            gemv(b, k, 16, 2, 2),
            gemv(b, k, 16, 2, 4),
        ]
        if k >= 4096:
            # Deep K: an 8-way split keeps enough rows in flight per SM.
            out += [gemv(b, k, 8, 4, 8), gemv(b, k, 16, 2, 8)]
    elif b <= 16:
        out += [skinny(b, 1), skinny(b, 2)]
        if (k // 32) % 4:
            out.append(skinny(b, 2, min_units=2, ku=2, p=4))
    else:
        out += [
            skinny(32, 2, min_units=2),
            skinny(32, 1, ctas_per_sm=2, min_units=2),
            skinny(32, 1),
        ]
    return out


# ------------------------------------------------------------ persistent
def persistent(bm, bn, wm, wn, ks=1, kw=128, sa=2, sched="dp"):
    return ("persistent", sched, bm, bn, wm, wn, ks, kw, sa)


def persistent_stages(tactic):
    _, _, bm, bn, _, _, ks, kw, sa = tactic
    return min(8, max_b_stages(bm, bn, ks, kw, sa))


def persistent_schedule(tactic, m, n, k, dev):
    """(grid, stream-K tiles) of a persistent tactic for this M."""
    _, sched, bm, bn, _, _, ks, kw, _ = tactic
    tiles = ceil_div(m, bm) * ceil_div(n, bn)
    if sched == "dp":
        return min(tiles, dev.sms), 0
    k_iters = ceil_div(ceil_div(k, kw), ks)
    if tiles >= dev.sms:
        return dev.sms, tiles % dev.sms
    return min(dev.sms, tiles * k_iters), tiles


def _stream_tile(m, n):
    """Persistent tile for weights that are streamed once at M <= 64.

    A cold weight streams fastest from about 8 to 20 CTAs that each read long
    contiguous runs of a row: the widest N tile that still leaves about 8
    tiles, with a K stage of 256 to 512 bytes per row (about 32 KB, double
    buffered). More CTAs or shorter runs per row lose DRAM locality.
    """
    if m > 32:
        return persistent(64, 128, 2, 4) if n >= 1024 else persistent(64, 64, 2, 2)
    if n >= 1024:
        return persistent(32, 128, 1, 4, 2)
    if n >= 384:
        return persistent(32, 64, 2, 2, 3)
    return persistent(32, 32, 2, 1, 4)


def _stream_tile_candidates(m, n):
    out = [_stream_tile(m, n), persistent(32, 128, 1, 4, 2)]
    out += [persistent(32, 64, 2, 2, 3), persistent(32, 32, 2, 1, 4)]
    if m > 32:
        out += [persistent(64, 128, 2, 4), persistent(64, 64, 2, 2)]
    return out


def _persistent_candidates(m, n, k, dev):
    p, sk = persistent, "streamk"
    out = [p(32, 64, 1, 4, 2)]
    if m <= 128:
        out += [
            p(64, 64, 2, 2),
            p(128, 64, 4, 2),
            p(32, 64, 1, 4, 2, sched=sk),
            p(32, 64, 2, 4, sched=sk),
        ]
    elif m <= 256:
        out += [p(64, 128, 1, 4), p(32, 64, 1, 4, 2, sched=sk)]
    else:
        out += [p(128, 64, 2, 2), p(128, 128, 2, 4, sched=sk)]
        if m > 512:
            out.append(p(64, 128, 2, 4))
    return out


# --------------------------------------------------------------- pingpong
def _group(k, dev, frac):
    """Largest M group whose A panels (group x 128 x K bytes) fit a fraction of L2."""
    return max(1, int(dev.l2_bytes * frac) // (128 * k))


def _wave_tile_n(m, n, dev):
    """N tile from the tile count: rounds of tiles over the SMs times the
    per-tile cost (a 128x64 tile costs about 0.575 of a 128x128 one)."""

    def cost(tn):
        tiles = ceil_div(m, 128) * ceil_div(n, tn)
        return ceil_div(tiles, dev.sms) * (1.0 if tn == 128 else 0.575)

    return min((128, 64), key=cost)


def pingpong(tile_n, tile_k, epi_n, epi_stages, group, coop=False, early=False):
    return ("pingpong", "dp", tile_n, tile_k, epi_n, epi_stages, group, coop, early)


def _pingpong_default(m, n, k, dev):
    return pingpong(_wave_tile_n(m, n, dev), 128, 32, 0, _group(k, dev, 0.5))


def _pingpong_candidates(m, n, k, dev):
    direct = (n * 2) % 16 != 0
    tile_n = _wave_tile_n(m, n, dev)
    groups = sorted({_group(k, dev, 0.5), _group(k, dev, 0.125)})
    out = [_pingpong_default(m, n, k, dev)]
    for g in groups:
        out.append(pingpong(tile_n, 128, 32, 0, g))
        if not direct:
            # Wider epilogue sub-tile: half the barriers per tile, fewer stages.
            out.append(pingpong(tile_n, 128, 64, 2, g))
    if not direct:
        out.append(pingpong(tile_n, 64, 64, 2, groups[-1]))
        # Cooperative 256x128 tile: both warp groups share one B tile (less
        # L2-to-SM traffic per MAC), epilogue not overlapped.
        for early in (False, True):
            out.append(pingpong(128, 64, 64, 1, groups[-1], coop=True, early=early))
    return out


# ------------------------------------------------------------------ policy
def check_shape(m, n, k):
    if m < 0 or n <= 0 or k <= 0:
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires positive N and K")
    if k % 32:
        raise ValueError(f"SM12x cute-dsl mm_mxfp8 requires K % 32 == 0, got K={k}")
    if max(m * k, n * k, m * n) >= 2**31:
        raise ValueError("SM12x cute-dsl mm_mxfp8 requires element offsets below 2**31")


def _dedup(tactics, n, k):
    seen, out = set(), []
    for t in tactics:
        if t in seen:
            continue
        if t[0] == "gemv" and not gemv_valid(t, n, k):
            continue
        if t[0] == "skinny" and not skinny_valid(t, n, k):
            continue
        seen.add(t)
        out.append(t)
    return out


def valid_tactics(m, n, k, dev):
    """Candidates for the autotuner bucket whose representative M is ``m``.

    Every candidate is compiled before the bucket is profiled, so the lists
    only keep configurations that won or tied a bucket in sweeps of all
    configurations on SM120 and SM121.
    """
    check_shape(m, n, k)
    out = [default_tactic(m, n, k, dev)]
    if m <= SKINNY_MAX_M:
        out += _decode_candidates(m, n, k)
    if 2 < m <= 64:
        out += _stream_tile_candidates(m, n)
    if 4 < m <= 16 or 32 < m <= 64:
        out.append(persistent(32, 64, 1, 4, 2))
    if 16 < m <= 32:
        out.append(persistent(32, 64, 2, 2))
    if 32 < m <= 64:
        out.append(persistent(32, 128, 1, 4))
    if m > 64:
        out += _persistent_candidates(m, n, k, dev)
        out += _pingpong_candidates(m, n, k, dev)
    return _dedup(out, n, k)


def default_tactic(m, n, k, dev):
    """Feature-based tactic used when the autotuner has no entry for this M.

    Chosen from CUPTI kernel timings (CUDA graphs, cold L2) on DGX Spark (GB10)
    over dense layers of recent LLMs (N 96..16384, K 384..8192):

    - Tiny or very large weights at M <= 32: GEMV / stream-K kernels.
    - Weights under 4M elements: the streaming persistent tile up to M = 64,
      except the GEMV at M <= 4 for rows up to 2048 elements.
    - M <= 4 otherwise: GEMV. M 5..32: the streaming tile below 8M elements,
      then the stream-K kernel.
    - Up to M = 64 (M = 256 for tiny weights): 32- or 64-row persistent tiles.
    - Larger M: the 128-row ping-pong kernel.
    """
    nk = n * k
    tiny = nk < 1 << 20
    small = not tiny and nk < 4 << 20
    if m <= SKINNY_MAX_M:
        if tiny or nk > 64 << 20:
            return _decode_default(m, n, k)
        if small and (m > 4 or k > 2048):
            return _stream_tile(m, n)
        if m <= 4:
            return _decode_default(m, n, k)
        if nk < 8 << 20:
            return _stream_tile(m, n)
        if m <= 8:
            return _decode_default(m, n, k)
        t = skinny(32, 1, warps=8, ctas_per_sm=1)
        if skinny_valid(t, n, k):
            return t
        return persistent(32, 64, 1, 4, 2)
    if tiny and m <= 256:
        return persistent(32, 64, 2, 2, 2)
    if m <= 64:
        if small:
            return _stream_tile(m, n)
        if 2048 <= n <= 8192:
            return persistent(64, 64, 2, 2)
        return persistent(32, 64, 1, 4, 2)
    return _pingpong_default(m, n, k, dev)


def resolve(tactic, m):
    """Runtime adjustment of a tuned tactic to the actual M.

    A cooperative 256-row tile whose last tile would be mostly padding (a
    tactic tuned at the top of its bucket, run at the bottom) falls back to
    the 128-row ping-pong tile.
    """
    if tactic[0] == "pingpong" and tactic[7] and (-m) % 256 * 8 > m:
        return pingpong(128, 128, 64, 2, tactic[6])
    return tactic


FAMILIES = ("gemv", "skinny", "persistent", "pingpong")


def supports_m(tactic, m):
    """Whether ``tactic`` is a tactic of this backend that can serve this M.

    Anything else (for example a tactic cached by another backend or an older
    version) makes the runner fall back to the default tactic.
    """
    if not isinstance(tactic, tuple) or not tactic or tactic[0] not in FAMILIES:
        return False
    if tactic[0] == "gemv":
        return m <= tactic[2]
    if tactic[0] == "skinny":
        return m <= SKINNY_MAX_M
    return True
