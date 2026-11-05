"""MX quantization: config space, ownership maps and the closed-form dispatcher.

Pure integer logic imported by the CuTe-DSL kernel (at trace time), by the symbolic tracer and by the CPU reference,
so the consumers cannot disagree about *which* elements a thread owns. Canonical constants: ``BLOCK = 32`` elements
per UE8M0 scale, 32 lanes, ``Kbp = pow2ceil(K / 32)`` the padded block count of a row (the compile-time bucket; M, K
and the strides are runtime arguments of every kernel).

The dispatcher :func:`default_config` is a **closed-form rule** of ``(M, K, element bytes, SM count, layout)`` -- no
measured table, no autotuning -- and every config it can return is a member of
:func:`all_valid_configs`, so the choice never changes bits (R3).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List

from .layout import (
    SF_COL_TILE,
    SF_ROW_TILE,
    layout_name,
    num_blocks as _num_blocks,
    padded_blocks as _padded_blocks,
)

BLOCK = 32
LANES = 32
MAX_BLOCKS_PER_THREAD = (
    32  # bound on the unrolled per-thread block loop (code size), a validity rule
)
MAX_FRAG_BYTES = (
    256  # a thread's whole row span is one register fragment: <= 64 registers
)
DEFAULT_SM_COUNT = (
    204  # sm_107 as shipped here; the dispatcher takes the real count from the caller
)


def num_blocks(K: int) -> int:
    return _num_blocks(K, BLOCK)


def padded_blocks(K: int) -> int:
    return _padded_blocks(K, BLOCK)


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


@dataclass(frozen=True)
class MXQuantConfig:
    """One kernel configuration (a placement of the block amax plus free structure). All valid configs compute the
    canonical dataflow.

    ``threads_per_row`` T: a power of two from 1 to 512 (T < 32: a warp handles 32 / T rows). ``elems_per_thread`` V:
    32 (one lane per block), 16 or 8 (2 / 4 lanes share a block: the ``elem`` digit is factored ``[(V, reg), (32/V,
    lane)]`` and the block amax is finished by 1 / 2 shuffles). ``blk_order``: ``"lane"``: consecutive lanes own
    consecutive blocks (coalesced loads); ``"reg"``: a thread owns consecutive blocks. ``mode = "tile"`` (128x4 layout
    only): a CTA covers a 128-row x ``tile_blocks``-block tile, each of its ``tile_warps`` warps streams ``128 /
    tile_warps`` rows one at a time (a row pass = the ``threads_per_row = 32`` placement), the scale bytes are staged
    in shared memory in the 128x4 order and the tile's scale region is written with one 16-byte store per thread."""

    threads_per_row: int
    rows_per_cta: int
    elems_per_thread: int = 32
    blk_order: str = "lane"
    mode: str = "row"
    tile_warps: int = 8
    tile_bpt: int = 1

    # -- derived ------------------------------------------------------------------
    @property
    def lanes_per_block(self) -> int:
        return BLOCK // self.elems_per_thread

    @property
    def warps_per_row(self) -> int:
        return max(self.threads_per_row // LANES, 1)

    @property
    def lanes_per_row(self) -> int:
        return min(self.threads_per_row, LANES)

    @property
    def block_lanes(self) -> int:
        """Lane-level factor of the ``blk`` digit: how many blocks the lanes of one row cover at once."""
        return self.lanes_per_row // self.lanes_per_block

    @property
    def rows_per_warp(self) -> int:
        return LANES // self.threads_per_row if self.threads_per_row < LANES else 1

    @property
    def num_threads(self) -> int:
        if self.mode == "tile":
            return LANES * self.tile_warps
        return self.threads_per_row * self.rows_per_cta

    @property
    def tile_passes(self) -> int:
        return SF_ROW_TILE // self.tile_warps

    @property
    def tile_blocks(self) -> int:
        """Tile mode: blocks per column tile (per warp row pass)."""
        return self.block_lanes * self.tile_bpt

    def blocks_per_thread(self, K: int) -> int:
        if self.mode == "tile":
            return self.tile_bpt
        return padded_blocks(K) // (self.block_lanes * self.warps_per_row)

    def bucket(self, K: int) -> int:
        """The compiled kernel's Kbp bucket (row mode) or the blocks of a column tile (tile mode, independent of K)."""
        return self.tile_blocks if self.mode == "tile" else padded_blocks(K)

    def frag_bytes(self, K: int, elem_bytes: int) -> int:
        return self.elems_per_thread * self.blocks_per_thread(K) * elem_bytes

    # -- validity -----------------------------------------------------------------
    def validate(self, K: int, elem_bytes: int = 2, swizzled=True) -> None:
        """``swizzled``: the scale-factor layout (``True`` / ``"128x4"``, ``"8x4"``, ``False`` / ``"linear"``); the tile mode
        stages 128x4 tiles and exists for that layout only."""
        layout = layout_name(swizzled)
        T, V = self.threads_per_row, self.elems_per_thread
        if self.mode not in ("row", "tile"):
            raise ValueError(f"mode={self.mode}")
        if V not in (32, 16, 8):
            raise ValueError(f"elems_per_thread={V}")
        if elem_bytes not in (2, 4):
            raise ValueError(f"elem_bytes={elem_bytes}")
        if self.mode == "tile":
            num_blocks(K)
            if layout != "128x4":
                raise ValueError(
                    "tile mode stages the 128x4 swizzled scale tiles; the 8x4 and linear layouts use the row kernel"
                )
            if T != LANES or self.blk_order != "lane":
                raise ValueError("tile mode: threads_per_row=32, lane order")
            if self.rows_per_cta != SF_ROW_TILE:
                raise ValueError(
                    f"tile mode: rows_per_cta={self.rows_per_cta} must be {SF_ROW_TILE}"
                )
            if not (1 <= self.tile_warps <= 32 and SF_ROW_TILE % self.tile_warps == 0):
                raise ValueError(
                    f"tile mode: tile_warps={self.tile_warps} must divide {SF_ROW_TILE} (1..32)"
                )
            if self.tile_bpt not in (1, 2):
                raise ValueError(f"tile mode: tile_bpt={self.tile_bpt}")
            if self.tile_blocks % SF_COL_TILE:
                raise ValueError(
                    "tile mode: a column tile holds whole 4-column scale tiles"
                )
            if self.frag_bytes(K, elem_bytes) > MAX_FRAG_BYTES:
                raise ValueError(
                    f"frag_bytes={self.frag_bytes(K, elem_bytes)} > {MAX_FRAG_BYTES}"
                )
            return
        if not (is_pow2(T) and 1 <= T <= 512):
            raise ValueError(f"threads_per_row={T}")
        if self.lanes_per_row % self.lanes_per_block:
            raise ValueError(f"T={T} has fewer lanes than a block needs with V={V}")
        if padded_blocks(K) % (self.block_lanes * self.warps_per_row):
            raise ValueError(
                f"block_lanes*W={self.block_lanes * self.warps_per_row} does not divide Kbp={padded_blocks(K)}"
            )
        if self.blocks_per_thread(K) > MAX_BLOCKS_PER_THREAD:
            raise ValueError(
                f"blocks_per_thread={self.blocks_per_thread(K)} > {MAX_BLOCKS_PER_THREAD}"
            )
        if self.rows_per_cta < 1 or self.rows_per_cta % self.rows_per_warp:
            raise ValueError(
                "rows_per_cta must be a positive multiple of rows_per_warp"
            )
        if self.num_threads > 1024 or self.num_threads % LANES:
            raise ValueError(f"num_threads={self.num_threads}")
        if self.blk_order not in ("lane", "reg"):
            raise ValueError(f"blk_order={self.blk_order}")
        if self.frag_bytes(K, elem_bytes) > MAX_FRAG_BYTES:
            raise ValueError(
                f"frag_bytes={self.frag_bytes(K, elem_bytes)} > {MAX_FRAG_BYTES}: use more threads per row"
            )

    def is_valid(self, K: int, elem_bytes: int = 2, swizzled=True) -> bool:
        try:
            self.validate(K, elem_bytes, swizzled)
            return True
        except ValueError:
            return False

    @staticmethod
    def parse(desc: str) -> "MXQuantConfig":
        m = re.fullmatch(r"TILE_V(8|16|32)_W(\d+)(_B2)?", desc)
        if m:
            return MXQuantConfig(
                LANES,
                SF_ROW_TILE,
                int(m.group(1)),
                mode="tile",
                tile_warps=int(m.group(2)),
                tile_bpt=2 if m.group(3) else 1,
            )
        m = re.fullmatch(r"T(\d+)_R(\d+)_V(8|16|32)(_RM)?", desc)
        if not m:
            raise ValueError(desc)
        return MXQuantConfig(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            "reg" if m.group(4) else "lane",
        )

    def describe(self) -> str:
        if self.mode == "tile":
            return f"TILE_V{self.elems_per_thread}_W{self.tile_warps}" + (
                f"_B{self.tile_bpt}" if self.tile_bpt > 1 else ""
            )
        return (
            f"T{self.threads_per_row}_R{self.rows_per_cta}_V{self.elems_per_thread}"
            + ("_RM" if self.blk_order == "reg" else "")
        )

    # -- canon placement -------------------------------------------------------------
    def placement(self, K: int):
        """This config as a :class:`flashinfer.batch_invariant.canon.Placement` of the kernels' reduction (digits ``elem``, ``blk``):
        ``elem = [(V, reg), (32/V, lane)]``; ``blk`` = the block-lane factor, the per-thread register factor (in the
        order ``blk_order`` says) and the warp factor, outermost."""
        from ..canon import Placement

        self.validate(K)
        if self.mode == "tile":
            return Placement.make(
                {
                    "elem": [
                        (self.elems_per_thread, "reg"),
                        (self.lanes_per_block, "lane"),
                    ],
                    "blk": [(self.block_lanes, "lane"), (self.tile_bpt, "reg")],
                },
                rows_per_cta=self.rows_per_cta,
                elems_per_thread=self.elems_per_thread,
                blk_order=self.blk_order,
                mode=self.mode,
                tile_warps=self.tile_warps,
                tile_bpt=self.tile_bpt,
            )
        bpt, bl, W = self.blocks_per_thread(K), self.block_lanes, self.warps_per_row
        blk = (
            [(bl, "lane"), (bpt, "reg")]
            if self.blk_order == "lane"
            else [(bpt, "reg"), (bl, "lane")]
        )
        return Placement.make(
            {
                "elem": [
                    (self.elems_per_thread, "reg"),
                    (self.lanes_per_block, "lane"),
                ],
                "blk": blk + [(W, "warp")],
            },
            rows_per_cta=self.rows_per_cta,
            elems_per_thread=self.elems_per_thread,
            blk_order=self.blk_order,
            mode=self.mode,
        )

    # -- ownership (the aligned-subrange rule, for the tests) --------------------------
    def owned_blocks(self, K: int, w: int, lane: int) -> List[int]:
        bpt, bl = self.blocks_per_thread(K), self.block_lanes
        bi = lane // self.lanes_per_block
        if self.blk_order == "lane":
            return [bi + bl * kk + bl * bpt * w for kk in range(bpt)]
        return [kk + bpt * bi + bpt * bl * w for kk in range(bpt)]

    def owned_part(self, lane: int) -> int:
        """Which ``V``-element part of its blocks the lane owns (0 when V = 32)."""
        return lane % self.lanes_per_block


def all_valid_configs(
    K: int, elem_bytes: int = 2, swizzled=True, max_threads: int = 512
) -> List[MXQuantConfig]:
    """Every config the test-suite checks for config agreement (R3 is claimed over this set); ``swizzled`` names the layout
    (``True`` / ``"128x4"``, ``"8x4"``, ``False`` / ``"linear"``: the tile placements exist for 128x4 only)."""
    out, seen = [], set()
    if layout_name(swizzled) == "128x4":
        for V in (32, 16, 8):
            for warps in (4, 8, 16):
                for bpt in (1, 2):
                    cfg = MXQuantConfig(
                        LANES,
                        SF_ROW_TILE,
                        V,
                        mode="tile",
                        tile_warps=warps,
                        tile_bpt=bpt,
                    )
                    if cfg.num_threads <= max_threads and cfg.is_valid(
                        K, elem_bytes, swizzled
                    ):
                        out.append(cfg)
    for T in (1 << i for i in range(10)):
        base_rows = LANES // T if T < LANES else 1
        for mult in (1, 2, 4):
            rows = base_rows * mult
            for V in (32, 16, 8):
                for order in ("lane", "reg"):
                    cfg = MXQuantConfig(T, rows, V, order)
                    if cfg.num_threads > max_threads or not cfg.is_valid(
                        K, elem_bytes, swizzled
                    ):
                        continue
                    key = cfg.placement(
                        K
                    )  # the two orders coincide when a factor is trivial
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(cfg)
    return out


# ----------------------------------------------------------------------------- the closed-form dispatcher
# The constants below are the physics the rule reasons about; every one was set from a sweep that timed
# every valid configuration at every grid point (CUPTI kernel time, cold L2), and is a property of a
# streaming row kernel on this architecture, not of a shape.
TILE_MAX_BYTES_PER_ROW = 2048  # the widest tile that ever measured best: 1024 bf16 / 512 fp32 elements per row per CTA (256 KB per CTA)
TILE_MIN_CTAS_PER_SM = 1  # the 32-block, 16-warp tile once its grid holds one CTA per SM and the row is at least a column tile wide
#                           (K >= 1024; a wider tile idles lanes in every pass). Measured under cold-L2 kernel time: on SM107 the tile beats
#                           the best row configuration by 10-22 % at M >= 16384 x K >= 1024 and M >= 4096 x K >= 6144, on SM100 it is within
#                           -7..+4 % there; below one CTA per SM it loses 1.2-5x on both (its serial row passes starve memory parallelism).
TILE_W16_MIN_CTAS_PER_SM = 0  # always 16 warps per tile CTA (the 8-warp tile never measured better by more than a few percent)
TILE_TIERS = (
    (16, 2),
)  # (V, blocks per lane per pass): one tile shape, 32 blocks per pass; V >= LANE_BYTES_MIN / elem_bytes, so fp32
#                          inputs (4 KB of input per row per CTA > TILE_MAX_BYTES_PER_ROW) have no tile tier
ROW_LARGE_CTAS_PER_SM = 2  # a row grid with fewer CTAs per SM than this is latency-bound: the most lanes per row
LANE_BYTES = 64  # bytes of input per lane per block in the row tier: V = 64 / elem_bytes (32 for bf16 / f16, 16 for fp32)
LANE_BYTES_MIN = 32  # never below two 16-byte vectors per lane per block (V >= 16 for 16-bit inputs, >= 8 for fp32)
FRAG_BYTES_TARGET = 128  # the per-lane input fragment the best row configs keep: blocks per thread <= 128 / (V * elem_bytes)
ROW_LANES_LARGE = (
    64,
    128,
)  # bandwidth-bound rows (enough CTAs for the SMs): threads per row in this window, V = 64 B per lane
#                              (128 lanes with two blocks per thread beat 256 lanes with one by 11-15 % at K >= 6144, M >= 4096)
ROW_LANES_MAX = 512  # latency-bound rows (fewer CTAs than SMs): the most lanes per row the kernel allows
CTA_THREADS_TARGET = (
    128  # rows per CTA fill a 128-thread CTA where the row has fewer lanes
)


def _row_cfg(K: int, elem_bytes: int, V: int, bpt: int, swizzled: bool):
    """The row config with ``V`` elements per lane and ``bpt`` blocks per thread, or None when no valid one exists."""
    Kbp = padded_blocks(K)
    lanes = Kbp * (BLOCK // V)
    if lanes % bpt or not is_pow2(lanes // bpt) or lanes // bpt > ROW_LANES_MAX:
        return None
    T = lanes // bpt
    rows_per_warp = LANES // T if T < LANES else 1
    rows = max(rows_per_warp, CTA_THREADS_TARGET // T)
    rows -= rows % rows_per_warp
    cfg = MXQuantConfig(T, max(rows, rows_per_warp), V, "lane")
    return cfg if cfg.is_valid(K, elem_bytes, swizzled) else None


def _row_candidates(K: int, elem_bytes: int, swizzled: bool):
    """Row configs with 32..64 bytes per lane per block and a fragment of at most ``FRAG_BYTES_TARGET`` bytes, the widest lanes
    first: ``[(V, bpt, cfg)]``."""
    out = []
    V = min(32, LANE_BYTES // elem_bytes)
    while max(8, LANE_BYTES_MIN // elem_bytes) <= V:
        for bpt in (1, 2, 4, 8):
            if bpt * V * elem_bytes > FRAG_BYTES_TARGET:
                break
            cfg = _row_cfg(K, elem_bytes, V, bpt, swizzled)
            if cfg is not None:
                out.append((V, bpt, cfg))
        V //= 2
    if not out:  # the fragment target cannot be met (never on the supported K): the widest valid config
        for V in (32, 16, 8):
            for bpt in (1, 2, 4, 8, 16, 32):
                cfg = _row_cfg(K, elem_bytes, V, bpt, swizzled)
                if cfg is not None:
                    out.append((V, bpt, cfg))
    return out


@functools.lru_cache(maxsize=65536)
def default_config(
    M: int, K: int, elem_bytes: int = 2, sm_count: int = DEFAULT_SM_COUNT, swizzled=True
) -> MXQuantConfig:
    """The closed-form dispatcher: the expected-best member of
    :func:`all_valid_configs` from the problem sizes, the element width, the SM count and the layout.

    1. **Tile tier** (128x4 layout, 16-bit inputs): the 128-row x 32-block tile placement (two lanes per block, two blocks
       per lane, 16 warps) when the row is at least a column tile wide (``K >= 1024``) and its grid
       ``ceil(M / 128) x ceil(K / 32 / 32)`` holds ``TILE_MIN_CTAS_PER_SM x sm_count`` CTAs; fitted on cold-L2 kernel-time
       sweeps of SM107 and SM100 parts.
    2. **Row tier**: ``V = 64 / elem_bytes`` elements per lane (64 B per block per lane; 32 B when more lanes are needed) and
       a per-lane fragment of at most ``FRAG_BYTES_TARGET`` = 128 B (one or two blocks per thread for 64-byte lanes).
       Bandwidth-bound rows (the grid holds >= ``ROW_LARGE_CTAS_PER_SM x sm_count`` CTAs): the fewest blocks per thread that
       keeps the threads per row within ``ROW_LANES_LARGE`` = [64, 128]. Latency-bound rows (a shallower grid): the most lanes per row
       up to ``ROW_LANES_MAX`` = 512 (one block or half a block per lane: warps hide the latency, not per-thread ILP).
       ``rows_per_cta`` fills a 128-thread CTA where the row has fewer lanes.
    The SM count enters only through the CTA counts of 1 and 2 (no table: 148, 152, 204, 224 or any count); every pick is a
    member of the space, so it never touches bits; more SMs never pick fewer CTAs (`test_dispatcher_is_monotone_in_the_sm_count`).
    ``swizzled`` names the scale-factor layout (``True`` / ``"128x4"``, ``"8x4"``, ``False`` / ``"linear"``): the 8x4 and linear
    layouts have no tile tier."""
    nb = num_blocks(K)
    swizzled = layout_name(swizzled)
    if swizzled == "128x4" and TILE_MIN_CTAS_PER_SM is not None:
        row_tiles = -(-M // SF_ROW_TILE)
        for V, bpt in TILE_TIERS:
            tb = V * bpt
            if (
                tb * BLOCK * elem_bytes > TILE_MAX_BYTES_PER_ROW
                or V * elem_bytes < LANE_BYTES_MIN
                or nb < tb  # a column tile wider than the row idles lanes in every pass
            ):
                continue
            ctas = row_tiles * -(-nb // tb)
            if ctas >= TILE_MIN_CTAS_PER_SM * sm_count:
                warps = 16 if ctas >= TILE_W16_MIN_CTAS_PER_SM * sm_count else 8
                cfg = MXQuantConfig(
                    LANES, SF_ROW_TILE, V, mode="tile", tile_warps=warps, tile_bpt=bpt
                )
                if cfg.is_valid(K, elem_bytes, swizzled):
                    return cfg
    cands = _row_candidates(K, elem_bytes, swizzled)
    assert cands, (M, K, elem_bytes)
    lo, hi = ROW_LANES_LARGE
    V0 = cands[0][0]
    # bandwidth-bound pick: the widest lanes (V0), the fewest blocks per thread with T in [lo, hi]; else the T closest to the window
    wide = [(V, bpt, c) for V, bpt, c in cands if V == V0] or cands
    inside = [t for t in wide if lo <= t[2].threads_per_row <= hi]
    large = (
        min(inside, key=lambda t: t[1])
        if inside
        else min(
            wide,
            key=lambda t: (
                max(lo - t[2].threads_per_row, t[2].threads_per_row - hi),
                t[1],
            ),
        )
    )
    if -(-M // large[2].rows_per_cta) >= ROW_LARGE_CTAS_PER_SM * sm_count:
        return large[2]
    # latency-bound pick (fewer CTAs than SMs): the most lanes per row, the widest lanes on ties
    small = max(cands, key=lambda t: (t[2].threads_per_row, t[0]))
    return small[2]
