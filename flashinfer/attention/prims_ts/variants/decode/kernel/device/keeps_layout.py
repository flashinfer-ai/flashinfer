# Copyright (c) 2026 by FlashInfer team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keeps (non-swapped MMA) score layout: TMEM rows, columns and tcgen05 accesses."""

import cutlass.cute as cute
import cutlass
from cutlass import Int32
from cutlass.experimental import primitives as prims
from ..config import FmhaDecodeConfig
from .common import Constexpr


class EvenOddColumns:
    """Logical-order view of O columns stored even-then-odd per group of 8.

    Physical position 8g + j holds logical column 8g + 2j (j < 4) or
    8g + 2(j - 4) + 1. Indexing is compile-time only, so the reorder is free
    register renaming, and elements are read where the caller reads them
    (after its tcgen05 wait).
    """

    def __init__(self, vals, num: int) -> None:
        assert num % 8 == 0
        self.vals = vals

    def __getitem__(self, col: int):
        group, r = divmod(col, 8)
        return self.vals[group * 8 + r // 2 + 4 * (r % 2)]


@cute.jit
def keeps_col_base(
    cfg: Constexpr[FmhaDecodeConfig], lane_idx: Int32, half_cols: int
) -> Int32:
    """Return the lane's keepsMmaAb output-column base."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        return Int32(0)
    if cutlass.const_expr(cfg.tile_size_q == 128):
        return Int32(0)
    return keeps_q64_col_base(lane_idx, half_cols)


@cute.jit
def keeps_o_tail_tcgen05_ld(
    cfg: Constexpr[FmhaDecodeConfig],
    tmem_addr,
    *,
    num: Constexpr[int],
    offset: Constexpr[int],
):
    """Load final Keeps O columns owned by a correction lane.

    Contiguous O uses the tileSizeQ-specific shape with the headdim/2 half
    split. Lane-panel D512 O keeps logical columns [256, 512) in lanes 16-31
    at the physical columns of [0, 256), so a 32x32b load gives threads 0-15
    and 16-31 the same logical column ownership as that half split.
    """
    if cutlass.const_expr(cfg.keeps_o_uses_lane_panels):
        vals = prims.tcgen05_ld("32x32b", tmem_addr, num=num)
        if cutlass.const_expr(cfg.uses_permuted_packed_v_order):
            return EvenOddColumns(vals, num)
        return vals
    return keeps_tcgen05_ld(cfg, tmem_addr, num=num, offset=offset)


@cute.jit
def keeps_q64_col_base(lane_idx: Int32, half_cols: int) -> Int32:
    """Return the head-dim column base for one q64 keeps lane group."""
    return (lane_idx >> Int32(4)) * Int32(half_cols)


@cute.jit
def keeps_q64_row_idx(warp_grp_thread_idx: Int32) -> Int32:
    """Map a correction lane to its q-row for tileSizeQ <= 64 keeps paths."""
    return (warp_grp_thread_idx >> Int32(5)) * Int32(16) + (
        warp_grp_thread_idx & Int32(0xF)
    )


@cute.jit
def keeps_route_atom(cfg: Constexpr[FmhaDecodeConfig], half: Int32, position) -> Int32:
    """Return the route atom a Keeps spatial half owns at ``position``.

    The halves interleave over the route's atoms: half ``h`` owns atoms ``h``,
    ``h + halves``, ... in order, so its ``position``-th atom is ``position *
    halves + h`` (``FmhaDecodeConfig.keeps_route_atom_owner`` is the inverse).
    """
    return position * Int32(cfg.keeps_spatial_halves) + half


@cute.jit
def keeps_row_idx(cfg: Constexpr[FmhaDecodeConfig], warp_grp_thread_idx: Int32):
    """Map a correction lane to the logical output row it owns."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # KV256's 2x2 datapath exposes two spatial KV128 partials for each
        # logical Q row. Threads [0, 64) and [64, 128) therefore share rows.
        return warp_grp_thread_idx & Int32(63)
    if cutlass.const_expr(cfg.tile_size_q == 128):
        return warp_grp_thread_idx
    return keeps_q64_row_idx(warp_grp_thread_idx)


@cute.jit
def keeps_score_col(
    cfg: Constexpr[FmhaDecodeConfig],
    warp_grp_thread_idx: Int32,
    reg_idx: Constexpr[int],
    col_base: Int32,
) -> Int32:
    """Return the semantic KV column represented by one Keeps score register."""
    if cutlass.const_expr(cfg.tile_size_kv == 256):
        # A physical thread owns four K32 fragments: ``keeps_fragments_per_atom``
        # per layout atom of its spatial half (``_keeps_route_atom``), each
        # fragment one K32 sub-block of that atom.
        fragment_regs = cfg.softmax_score_fragment_regs
        fragments_per_atom = cfg.keeps_fragments_per_atom
        spatial = keeps_spatial_half(cfg, warp_grp_thread_idx)
        fragment = reg_idx // fragment_regs
        semantic_block = keeps_route_atom(cfg, spatial, fragment // fragments_per_atom)
        return (
            semantic_block * Int32(cfg.keeps_atom_tokens)
            + Int32((fragment % fragments_per_atom) * fragment_regs)
            + Int32(reg_idx % fragment_regs)
        )
    return col_base + Int32(reg_idx)


@cute.jit
def keeps_spatial_half(
    cfg: Constexpr[FmhaDecodeConfig], warp_grp_thread_idx: Int32
) -> Int32:
    """Return the spatial half a Keeps warp-group thread belongs to.

    KV256 threads ``[0, 64)`` and ``[64, 128)`` own the two spatial KV128
    partials of one logical Q row (``FmhaDecodeConfig.keeps_spatial_halves``);
    every other Keeps profile has a single half.
    """
    if cutlass.const_expr(cfg.keeps_spatial_halves == 2):
        return warp_grp_thread_idx >> Int32(6)
    return Int32(0)


@cute.jit
def keeps_tcgen05_ld(
    cfg: Constexpr[FmhaDecodeConfig],
    tmem_addr,
    *,
    num: Constexpr[int],
    offset: Constexpr[int],
):
    """Load keepsMmaAb TMEM fragments using the tileSizeQ-specific shape."""
    if cutlass.const_expr(cfg.tile_size_kv == 256 or cfg.tile_size_q == 128):
        return prims.tcgen05_ld(
            "32x32b",
            tmem_addr,
            num=num,
        )
    # The 16x32bx2 variant has a required half-split offset operand.  Route
    # through the public primitive wrapper so Python constants are materialized
    # as MLIR values before reaching the low-level operation.
    return prims.tcgen05_ld(
        "16x32bx2",
        tmem_addr,
        num=num,
        offset=offset,
    )


@cute.jit
def keeps_tcgen05_st(
    cfg: Constexpr[FmhaDecodeConfig],
    tmem_addr,
    val,
    *,
    offset: Constexpr[int],
) -> None:
    """Store keepsMmaAb TMEM fragments using the tileSizeQ-specific shape."""
    if cutlass.const_expr(cfg.tile_size_kv == 256 or cfg.tile_size_q == 128):
        prims.tcgen05_st(
            "32x32b",
            tmem_addr,
            val,
        )
    else:
        prims.tcgen05_st(
            "16x32bx2",
            tmem_addr,
            val,
            offset=offset,
        )


@cute.jit
def swaps_routed_coordinate(
    cfg: Constexpr[FmhaDecodeConfig],
    lane_k_offset: Int32,
    origin0: Int32,
    origin1: Int32,
    origin2: Int32,
    origin3: Int32,
    *,
    token_group_idx: Constexpr[int],
) -> tuple[Int32, Int32]:
    """Map one SWAP register group to its staged atom and logical coordinate."""

    atom_size = min(cfg.kv_block_size, 32)
    groups_per_atom = atom_size // 8
    origin_idx = token_group_idx // groups_per_atom
    atom_origin = origin0
    if cutlass.const_expr(origin_idx == 1):
        atom_origin = origin1
    elif cutlass.const_expr(origin_idx == 2):
        atom_origin = origin2
    elif cutlass.const_expr(origin_idx == 3):
        atom_origin = origin3
    token_offset = (token_group_idx % groups_per_atom) * 8
    return atom_origin, atom_origin + Int32(token_offset) + lane_k_offset
