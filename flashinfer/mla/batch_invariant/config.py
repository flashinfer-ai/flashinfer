# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""The configuration space of the sparse MLA class. Every member computes the class (R3); the knobs place work only.

* ``head_block``: head rows per CTA (one MMA warp per 16); heads past a block go to another CTA;
* ``gather_warps``: warps that copy the tile's key rows into shared memory;
* ``fold``: who folds a row's tiles: ``"fixup"`` (a second kernel over per-tile partial rows) or ``"inkernel"`` (the
  row's CTA walks every tile in order);
* ``tiles_per_unit``: tiles a CTA computes before it exits (``0``: all of the row's; required by ``inkernel``);
* ``pv_pass``: output columns per PV pass of a warp;
* ``mma_split``: warps a 16-head group's tile is spread over (``4`` needs ``head_block == 16``): its positions for
  the scores and its output columns for the PV fold.

The record is emitted to Lean as ``BIK.SparseMLA.Config`` and checked against ``ConfigOK``, the mirror of
``well_formed``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

HEAD_BLOCKS = (16, 32, 64)
GATHER_WARPS = (4, 8)
FOLDS = ("fixup", "inkernel")
TILES_PER_UNIT = (0, 1, 2, 4, 8)
PV_PASSES = (64, 128)
MMA_SPLITS = (1, 4)
# the dispatcher's rules, see ``default_config``
BUSY_UNITS_PER_SM = 16
WS_ROW_FLOATS = 516
FIXUP_WORKSPACE_FLOATS = 1 << 28


@dataclass(frozen=True)
class SparseMlaConfig:
    head_block: int = 16
    gather_warps: int = 8
    fold: str = "fixup"
    tiles_per_unit: int = 1
    pv_pass: int = 128
    mma_split: int = 1

    def well_formed(self) -> bool:
        return (
            self.head_block in HEAD_BLOCKS
            and self.gather_warps in GATHER_WARPS
            and self.gather_warps * 16 >= self.head_block
            and self.fold in FOLDS
            and self.tiles_per_unit in TILES_PER_UNIT
            and (self.fold == "inkernel") == (self.tiles_per_unit == 0)
            and self.pv_pass in PV_PASSES
            and self.mma_split in MMA_SPLITS
            and (self.mma_split == 1 or self.head_block == 16)
        )

    def describe(self) -> str:
        return (
            f"h{self.head_block}_g{self.gather_warps}_{self.fold}"
            f"_t{self.tiles_per_unit}_p{self.pv_pass}_w{self.mma_split}"
        )

    def lean(self) -> str:
        """The record ``BIK.SparseMLA.Config`` (headBlock, gatherWarps, fold, tilesPerUnit, pvPass, mmaSplit)."""
        return (
            f"⟨{self.head_block}, {self.gather_warps}, {FOLDS.index(self.fold)}, "
            f"{self.tiles_per_unit}, {self.pv_pass}, {self.mma_split}⟩"
        )


def all_valid_configs() -> List[SparseMlaConfig]:
    """The enumerated space (R3 is claimed over it)."""
    out = [
        SparseMlaConfig(h, g, f, t, p, w)
        for h in HEAD_BLOCKS
        for g in GATHER_WARPS
        for f in FOLDS
        for t in TILES_PER_UNIT
        for p in PV_PASSES
        for w in MMA_SPLITS
    ]
    return [c for c in out if c.well_formed()]


def default_config(
    rows: int, heads: int, width: int, sm_count: int = 148
) -> SparseMlaConfig:
    """The dispatcher. It may read anything about the call; it only chooses among members of the space.

    One tile per CTA with the fixup fold while the tiles leave SMs idle (a 2,048-key row is then 16 to 17 parallel
    units); four tiles per CTA once every SM has ``BUSY_UNITS_PER_SM`` tiles queued (a CTA's start and its query
    rows are then paid once per four tiles); the in-kernel fold, which keeps one partial row per head instead of
    one per tile, when the partial rows would pass ``FIXUP_WORKSPACE_FLOATS`` (prefill-sized calls)."""
    head_block = 16 if heads <= 16 else 32 if heads <= 32 else 64
    split = 4 if head_block == 16 else 1
    tiles = -(-width // 128)
    if rows * heads * tiles * WS_ROW_FLOATS > FIXUP_WORKSPACE_FLOATS:
        return SparseMlaConfig(head_block, 8, "inkernel", 0, 128, split)
    units = rows * -(-heads // head_block) * tiles
    per_unit = 4 if units >= BUSY_UNITS_PER_SM * sm_count else 1
    return SparseMlaConfig(head_block, 8, "fixup", per_unit, 128, split)
