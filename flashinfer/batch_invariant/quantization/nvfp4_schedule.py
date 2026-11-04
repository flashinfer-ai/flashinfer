"""NVFP4 quantization: config space, ownership maps and the scale-factor layout (docs/nvfp4_quant_scoping.md
section 3).

Pure integer logic imported by the CuTe-DSL kernel (at trace time), by the symbolic tracer and by the CPU
reference, and mirrored by the Lean model, so the consumers cannot disagree about *which* elements a
thread owns or *where* a scale byte goes.

Canonical constants: ``BLOCK = 16`` elements per scale factor, 32 lanes, ``Kbp = pow2ceil(K / 16)`` the
padded block count of a row. Scale factors use FlashInfer's 128x4 swizzled layout only (rows padded to a
multiple of 128, scale columns to a multiple of 4, padding bytes zero).
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List

BLOCK = 16
LANES = 32
SF_ROW_TILE = 128
SF_COL_TILE = 4
MAX_BLOCKS_PER_THREAD = (
    32  # bound on the unrolled per-thread block loop (code size), a validity rule
)
MAX_FRAG_BYTES = 256  # a thread's whole row span (its blocks' halves) is one register fragment: <= 64 registers
REREAD_MIN_FRAG_BYTES = 128  # per-token "reread" staging is enumerated only where the fragment is this large


def pow2ceil(n: int) -> int:
    return 1 if n <= 1 else 1 << (n - 1).bit_length()


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def num_blocks(K: int) -> int:
    if K <= 0 or K % BLOCK:
        raise ValueError(f"K={K} must be a positive multiple of {BLOCK}")
    return K // BLOCK


def padded_blocks(K: int) -> int:
    """``Kbp``: the padded block count a kernel is compiled for (one kernel per Kbp bucket)."""
    return pow2ceil(num_blocks(K))


# ---------------------------------------------------------------------------- the 128x4 scale layout
def padded_sf_rows(M: int) -> int:
    return -(-M // SF_ROW_TILE) * SF_ROW_TILE


def padded_sf_cols(K: int) -> int:
    return -(-num_blocks(K) // SF_COL_TILE) * SF_COL_TILE


def sf_numel(M: int, K: int) -> int:
    return padded_sf_rows(M) * padded_sf_cols(K)


def sf_offset_128x4(row: int, col: int, padded_cols: int) -> int:
    """Byte offset of the scale of block ``col`` of row ``row`` in FlashInfer's 128x4 swizzled layout
    (``compute_sf_index_swizzled_128x4_gpu`` / ``get_sf_out_offset_128x4``): tiles of 128 rows x 4 scale
    columns are 512 contiguous bytes laid out as ``[row % 32][(row % 128) // 32][col % 4]``; column
    tiles are consecutive within a row tile; row tiles are ``128 * padded_cols`` bytes apart."""
    return (
        (col % 4)
        + (col // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + (row // 128) * (128 * padded_cols)
    )


# ---------------------------------------------------------------------------- the config space
@dataclass(frozen=True)
class QuantConfig:
    """One kernel configuration (a placement of the amax reduction plus free structure). All valid
    configs compute the canonical dataflow.

    ``threads_per_row`` T: a power of two from 1 to 512 (T < 32: a warp handles 32 / T rows).
    ``elems_per_thread`` V: 16 (one lane per 16-element block) or 8 (two lanes share a block: the
    ``elem`` digit is factored ``[(8, reg), (2, lane)]`` and the block amax is finished by one shuffle).
    ``blk_order``: which factor of the ``blk`` digit is innermost. ``"lane"``: consecutive lanes own
    consecutive blocks (coalesced 512 B per warp-load; the row reduction then shuffles one partial per
    owned block). ``"reg"``: a thread owns consecutive blocks (one shuffle per lane step, strided loads).
    ``stage`` (per-token only; the per-tensor class has a single pass): ``"regs"`` keeps the row's
    input fragment between the amax pass and the quantisation pass, ``"reread"`` reloads it (L2-hot).
    A thread's span (``V * blocks_per_thread`` elements) is one register fragment of at most
    ``MAX_FRAG_BYTES``.
    """

    threads_per_row: int
    rows_per_cta: int
    elems_per_thread: int = 16
    blk_order: str = "lane"
    stage: str = "regs"
    # "row": the row kernel above. "tile" (per-tensor class only): a CTA covers a 128-row x 32-block
    # (V = 16; 16-block for V = 8) tile; each of its ``tile_warps`` warps streams 128 / tile_warps rows
    # of the tile one at a time (a row pass = 32 lanes x one block or half-block: the same placement as
    # ``threads_per_row = 32``), the scale bytes are staged in shared memory in the 128x4 layout and the
    # tile's scale region is written out with one 16-byte store per thread. Loads and both stores stream.
    mode: str = "row"
    tile_warps: int = 8
    tile_bpt: int = 1  # tile mode: blocks per lane per row pass (1 or 2: 32- or 64-block column tiles)

    # -- derived ------------------------------------------------------------------
    @property
    def lanes_per_block(self) -> int:
        return BLOCK // self.elems_per_thread

    @property
    def warps_per_row(self) -> int:  # W
        return max(self.threads_per_row // LANES, 1)

    @property
    def lanes_per_row(self) -> int:  # Lr
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
        """Tile mode: row passes per warp."""
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
        """The compiled kernel's Kbp bucket: the padded block count (row mode) or the blocks of a column
        tile (tile mode, independent of K)."""
        return self.tile_blocks if self.mode == "tile" else padded_blocks(K)

    def frag_bytes(self, K: int, elem_bytes: int) -> int:
        """Input bytes a thread holds for one row (its blocks' halves)."""
        return self.elems_per_thread * self.blocks_per_thread(K) * elem_bytes

    # -- validity -----------------------------------------------------------------
    def validate(self, K: int, elem_bytes: int = 2) -> None:
        T, V = self.threads_per_row, self.elems_per_thread
        if self.mode not in ("row", "tile"):
            raise ValueError(f"mode={self.mode}")
        if self.mode == "tile":
            num_blocks(K)
            if (
                T != LANES
                or V not in (8, 16)
                or self.blk_order != "lane"
                or self.stage != "regs"
            ):
                raise ValueError(
                    "tile mode: threads_per_row=32, elems_per_thread in (8, 16), lane order, register staging"
                )
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
            return
        if not (is_pow2(T) and 1 <= T <= 512):
            raise ValueError(f"threads_per_row={T}")
        if V not in (8, 16):
            raise ValueError(f"elems_per_thread={V}")
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
        if self.stage not in ("regs", "reread"):
            raise ValueError(f"stage={self.stage}")
        if self.frag_bytes(K, elem_bytes) > MAX_FRAG_BYTES:
            raise ValueError(
                f"frag_bytes={self.frag_bytes(K, elem_bytes)} > {MAX_FRAG_BYTES}: use more threads per row"
            )

    def is_valid(self, K: int, elem_bytes: int = 2) -> bool:
        try:
            self.validate(K, elem_bytes)
            return True
        except ValueError:
            return False

    @staticmethod
    def parse(desc: str) -> "QuantConfig":
        """Inverse of :meth:`describe` (the dispatch table stores descriptions)."""
        m = re.fullmatch(r"TILE_V(8|16)_W(\d+)(_B2)?", desc)
        if m:
            return QuantConfig(
                LANES,
                SF_ROW_TILE,
                int(m.group(1)),
                mode="tile",
                tile_warps=int(m.group(2)),
                tile_bpt=2 if m.group(3) else 1,
            )
        m = re.fullmatch(r"T(\d+)_R(\d+)_V(8|16)(_RM)?(_RR)?", desc)
        if not m:
            raise ValueError(desc)
        return QuantConfig(
            int(m.group(1)),
            int(m.group(2)),
            int(m.group(3)),
            "reg" if m.group(4) else "lane",
            "reread" if m.group(5) else "regs",
        )

    def describe(self) -> str:
        if self.mode == "tile":
            return f"TILE_V{self.elems_per_thread}_W{self.tile_warps}" + (
                f"_B{self.tile_bpt}" if self.tile_bpt > 1 else ""
            )
        s = f"T{self.threads_per_row}_R{self.rows_per_cta}_V{self.elems_per_thread}"
        return (
            s
            + ("_RM" if self.blk_order == "reg" else "")
            + ("_RR" if self.stage == "reread" else "")
        )

    # -- canon placement -------------------------------------------------------------
    def placement(self, K: int):
        """This config as a :class:`flashinfer.batch_invariant.canon.Placement` of the row amax reduction (digits ``elem``,
        ``blk``): ``elem = [(V, reg), (16/V, lane)]``; ``blk`` = the block-lane factor, the per-thread
        register factor (in the order ``blk_order`` says) and the warp factor, outermost."""
        from ..canon import Placement

        self.validate(K)
        if self.mode == "tile":
            # a row pass: the ``threads_per_row = 32`` placement over one column tile of block_lanes
            # blocks (the per-tensor class never combines the block digit; it fixes the ownership)
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
                stage=self.stage,
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
            stage=self.stage,
            mode=self.mode,
        )

    # -- ownership (the aligned-subrange rule, for the tests) --------------------------
    def owned_blocks(self, K: int, w: int, lane: int) -> List[int]:
        """Blocks of the row owned by lane ``lane`` of warp ``w`` (lane < lanes_per_row)."""
        bpt, bl = self.blocks_per_thread(K), self.block_lanes
        bi = lane // self.lanes_per_block
        if self.blk_order == "lane":
            return [bi + bl * kk + bl * bpt * w for kk in range(bpt)]
        return [kk + bpt * bi + bpt * bl * w for kk in range(bpt)]

    def owned_half(self, lane: int) -> int:
        """Which ``V``-element half of its blocks the lane owns (0 when V = 16)."""
        return lane % self.lanes_per_block


def all_valid_configs(
    K: int, elem_bytes: int = 2, per_token: bool = False, max_threads: int = 512
) -> List[QuantConfig]:
    """Every config the test-suite checks for config agreement (R3 is claimed over this set)."""
    out, seen = [], set()
    if not per_token:
        for V in (16, 8):
            for warps in (4, 8, 16):
                for bpt in (1, 2):
                    cfg = QuantConfig(
                        LANES,
                        SF_ROW_TILE,
                        V,
                        mode="tile",
                        tile_warps=warps,
                        tile_bpt=bpt,
                    )
                    if cfg.num_threads <= max_threads and cfg.is_valid(K, elem_bytes):
                        out.append(cfg)
    for T in (1 << i for i in range(10)):
        base_rows = LANES // T if T < LANES else 1
        for mult in (1, 2, 4):
            rows = base_rows * mult
            for V in (16, 8):
                for order in ("lane", "reg"):
                    cfg = QuantConfig(T, rows, V, order)
                    if cfg.num_threads > max_threads or not cfg.is_valid(K, elem_bytes):
                        continue
                    key = cfg.placement(
                        K
                    )  # the two orders coincide when a factor is trivial
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append(cfg)
                    if (
                        per_token
                        and cfg.frag_bytes(K, elem_bytes) >= REREAD_MIN_FRAG_BYTES
                    ):
                        out.append(QuantConfig(T, rows, V, order, "reread"))
    return out


# -- the closed-form dispatcher (2026-10-30; fitted and graded against the CUPTI regret sweeps of both parts, Rubin NVL8 (SM107)
# and B200 (SM100), by bench/bench_quant_regret.py + bench/grade_quant_dispatch.py; the measured table it replaces lived here). One function of (M, K, elem bytes, per_token, sm_count); the SM count enters only through CTA counts.
#   Tile tier (per-tensor class only): the 128-row tile placement (16 warps streaming the tile's rows, scales staged in smem)
#   with two blocks per lane when its grid holds TILE_MIN_CTAS_PER_SM CTAs per SM, else with one, and only when the row is a
#   whole number of column tiles (the real block count, not the padded bucket: a partial tile idles lanes in every pass -- the
#   K <= 512 x M = 32768 tail of rule v2, the K = 384 / 768 cells of the first v3 pass on the B200);
#   the sweeps show it beating every row config by 1.2-1.4x once its grid is at least one CTA per SM and losing below that.
#   Row tier: one 16-element block per lane (V = 16, 32 B of bf16: FlashInfer's mechanism), as many lanes as blocks (T = Kbp,
#   capped at ROW_LANES_MAX) when the grid is shallow -- the decode / latency-bound regime -- and two blocks per lane (half the
#   lanes, twice the bytes in flight per thread) once the halved grid still holds ROW_BPT2_MIN_CTAS_PER_SM CTAs per SM; the CTA
#   is filled to CTA_THREADS_TARGET threads with rows when a row has fewer lanes.
DEFAULT_SM_COUNT = 204
CTA_THREADS_TARGET = 128
ROW_LANES_MAX = 512  # per-tensor: lanes per row are capped by the kernel (16 warps); per-token: 256 (the row-wide amax
ROW_LANES_MAX_PER_TOKEN = 256  # reduction spans the warps: 8 warps per row beat 16 at every K >= 6144 in the sweeps)
FRAG_BYTES_TARGET = 128  # blocks per thread double only while the per-thread fragment stays within 128 B of input ...
ROW_BPT2_MIN_CTAS_PER_SM = (
    4  # ... and while the halved grid still holds this many CTAs per SM ...
)
ROW_DOUBLINGS = 1  # ... at most once for the per-tensor class (one pass: more lanes beat more work per thread) ...
ROW_DOUBLINGS_PER_TOKEN = 2  # ... twice for the per-token class (two passes over the row: the fragment stays in registers)
TILE_MIN_CTAS_PER_SM = 1
TILE_WARPS = 16


def _row_cfg(
    K: int,
    elem_bytes: int,
    V: int,
    bpt: int,
    stage: str = "regs",
    lanes_max: int = ROW_LANES_MAX,
):
    """The row config with ``V`` elements per lane and ``bpt`` blocks per thread, or None when no valid one exists."""
    Kbp = padded_blocks(K)
    lanes = Kbp * (BLOCK // V)
    if lanes % bpt or not is_pow2(lanes // bpt) or lanes // bpt > lanes_max:
        return None
    T = lanes // bpt
    rows_per_warp = LANES // T if T < LANES else 1
    rows = max(rows_per_warp, CTA_THREADS_TARGET // T)
    rows -= rows % rows_per_warp
    cfg = QuantConfig(T, max(rows, rows_per_warp), V, "lane", stage)
    return cfg if cfg.is_valid(K, elem_bytes) else None


def _ctas(M: int, cfg: QuantConfig) -> int:
    return -(-M // cfg.rows_per_cta)


@functools.lru_cache(maxsize=65536)
def closed_form_config(
    K: int,
    M: int,
    elem_bytes: int = 2,
    per_token: bool = False,
    sm_count: int = DEFAULT_SM_COUNT,
    layout: str = "128x4",
) -> QuantConfig:
    """The closed form (the constants above); every pick is a member of ``all_valid_configs``; more SMs never pick fewer
    CTAs. The tile tier writes the 128x4 layout: the other layouts take the row tier."""
    Kbp = padded_blocks(K)
    if not per_token and layout == "128x4":
        row_tiles = -(-M // SF_ROW_TILE)
        for bpt in (2, 1):
            cfg = QuantConfig(
                LANES, SF_ROW_TILE, 16, mode="tile", tile_warps=TILE_WARPS, tile_bpt=bpt
            )
            if not cfg.is_valid(K, elem_bytes) or num_blocks(K) % cfg.tile_blocks:
                continue  # only exact column tiles: a partial tile (K = 384, 768, 1536 at 64 blocks) idles lanes
            if (
                row_tiles * (num_blocks(K) // cfg.tile_blocks)
                >= TILE_MIN_CTAS_PER_SM * sm_count
            ):
                return cfg
    # row tier: one block per lane with as many lanes as the row has blocks (the bytes in flight of the latency-bound regime),
    # then blocks per thread doubled while the halved grid still fills the SMs (and the fragment stays small)
    lanes_max = ROW_LANES_MAX_PER_TOKEN if per_token else ROW_LANES_MAX
    bpt = max(1, Kbp // lanes_max)
    cfg = _row_cfg(K, elem_bytes, 16, bpt, lanes_max=lanes_max)
    for _ in range(ROW_DOUBLINGS_PER_TOKEN if per_token else ROW_DOUBLINGS):
        nxt = _row_cfg(K, elem_bytes, 16, bpt * 2, lanes_max=lanes_max)
        if (
            nxt is None
            or nxt.frag_bytes(K, elem_bytes) > FRAG_BYTES_TARGET
            or _ctas(M, nxt) < ROW_BPT2_MIN_CTAS_PER_SM * sm_count
        ):
            break
        cfg, bpt = nxt, bpt * 2
    if cfg is not None:
        return cfg
    for V in (
        16,
        8,
    ):  # the fragment or lane bounds leave no one-block config (never on the supported K)
        for bpt in (1, 2, 4, 8, 16, 32):
            cfg = _row_cfg(K, elem_bytes, V, bpt)
            if cfg is not None:
                return cfg
    raise ValueError(f"no valid config for K={K}")


def default_config(
    K: int,
    M: int,
    elem_bytes: int = 2,
    per_token: bool = False,
    sm_count=None,
    layout: str = "128x4",
) -> QuantConfig:
    """The dispatcher: the closed form above (no table, no autotuning), for the device's SM count (the caller passes it;
    ``None`` = DEFAULT_SM_COUNT) and the scale-factor layout. It may depend on anything: every config it can return
    computes the canonical dataflow, so the choice never changes bits (R3 over ``all_valid_configs``)."""
    return closed_form_config(
        K, M, elem_bytes, per_token, sm_count or DEFAULT_SM_COUNT, layout
    )
