# Copyright (c) 2026 by FlashInfer team.
# SPDX-License-Identifier: Apache-2.0
"""Configuration space and dispatcher of the tensor-parallel partials kernel of ``lse.row`` (``README.md``, Sampling: the
tensor-parallel protocol).

A :class:`TPConfig` is a placement of the shard's blocks on threads: ``threads_per_row`` (``W = T / 32`` warps per row per
CTA), rows per CTA, the vector width and the cluster size ``C``; the blocks per thread ``kpw`` and the cover's base follow
from the shard (:meth:`TPConfig.structure`, :class:`shards.TPStructure`). Every member stores the same
partials (the wire format is a function of the shard alone: ``shards.ShardLayout``); the dispatcher only picks. A partials
launch makes ONE pass over the shard (the two rounds are two launches), so there is no staging knob.
"""

from __future__ import annotations

import functools
import re
from dataclasses import dataclass
from typing import List, Optional

from .schedule import LANES, THREADS, V, max_vec_elems, pow2ceil
from .shards import ShardLayout, TPStructure

KPW_MAX = 32  # blocks per thread (256 elements): the programs unroll a thread's chunk
COVER_SLACK = 2  # the cover may be at most this many times the power-of-two ceiling of the shard's block span
CLUSTERS = (1, 2, 4, 8, 16)


@dataclass(frozen=True)
class TPConfig:
    threads_per_row: int  # 32 .. 1024 per CTA: W = T / 32 warps per row
    rows_per_cta: int = 1
    vec_elems: int = V
    cluster: int = 1  # CTAs per row (the cover is split across them; partials cross through distributed smem)

    # the row loader's expectations (``kernel._RowLoader``): fragments are kept, there is no smem staging
    stage: str = "regs"

    @property
    def keep_regs(self) -> bool:
        return True

    @property
    def warps_per_row(self) -> int:
        return self.threads_per_row // LANES

    @property
    def num_threads(self) -> int:
        return self.threads_per_row * self.rows_per_cta

    def structure(self, lay: ShardLayout) -> TPStructure:
        self.validate(lay)
        return TPStructure.resolve(lay, self.warps_per_row, self.cluster)

    def validate(self, lay: ShardLayout) -> None:
        if self.threads_per_row not in THREADS:
            raise ValueError(f"threads_per_row={self.threads_per_row}")
        if self.vec_elems not in (1, 2, 4, 8):
            raise ValueError(f"vec_elems={self.vec_elems}")
        if self.rows_per_cta < 1 or self.num_threads > 1024:
            raise ValueError(
                f"rows_per_cta={self.rows_per_cta} (num_threads={self.num_threads})"
            )
        if self.cluster not in CLUSTERS:
            raise ValueError(f"cluster={self.cluster}")
        if self.cluster > 1 and self.rows_per_cta != 1:
            raise ValueError("a cluster member holds one row per CTA")
        if lay.lo == lay.hi:
            raise ValueError("an empty shard has no kernel")
        st = TPStructure.resolve(lay, self.warps_per_row, self.cluster)
        B_lo, B_hi = lay.block_span
        if st.kpw > KPW_MAX:
            raise ValueError(f"kpw={st.kpw} blocks per thread exceeds {KPW_MAX}")
        if st.Kv > COVER_SLACK * pow2ceil(B_hi - B_lo):
            raise ValueError(f"cover of {st.Kv} blocks for a span of {B_hi - B_lo}")

    def is_valid(self, lay: ShardLayout) -> bool:
        try:
            self.validate(lay)
            return True
        except ValueError:
            return False

    def describe(self) -> str:
        return f"T{self.threads_per_row}_R{self.rows_per_cta}_V{self.vec_elems}" + (
            f"_C{self.cluster}" if self.cluster > 1 else ""
        )

    @staticmethod
    def parse(desc: str) -> "TPConfig":
        m = re.fullmatch(r"T(\d+)_R(\d+)_V(\d+)(_C(\d+))?", desc)
        if not m:
            raise ValueError(desc)
        return TPConfig(
            int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(5) or 1)
        )


def all_valid_tp_configs(
    lay: ShardLayout, elem_bytes: int = 2, vec_max: Optional[int] = None
) -> List[TPConfig]:
    """Every configuration the suites check for agreement on this shard."""
    out = []
    vmax = (
        max_vec_elems(lay.hi - lay.lo, elem_bytes)
        if vec_max is None
        else min(vec_max, max_vec_elems(lay.hi - lay.lo, elem_bytes))
    )
    vecs = [
        v for v in (1, 2, 4, 8) if v <= vmax
    ]  # every width the shard admits (a view of a wider tensor may force a narrow one)
    for T in THREADS:
        for rows in (1, 2, 4):
            for vec in vecs:
                cfg = TPConfig(T, rows, vec)
                if cfg.is_valid(lay):
                    out.append(cfg)
        for C in CLUSTERS[1:]:
            cfg = TPConfig(T, 1, vmax, C)
            if cfg.is_valid(lay):
                out.append(cfg)
    return out


# ---------------------------------------------------------------------------------------------------- the dispatcher
BYTES_PER_THREAD_LAT = 256  # the row class's latency-regime constant: ~256 input bytes per thread of a CTA's share
BYTES_PER_THREAD_TP = 512  # the row class's throughput-regime constant
CLUSTER_ROWS = (
    512  # fewer rows than SMs: C = min(16, pow2floor(CLUSTER_ROWS / M)) CTAs per row
)
CLUSTER_MIN_SHARE = 4096  # a CTA's share of the shard is at least this many bytes
THREADS_PER_CTA_TARGET = 128  # throughput regime: rows per CTA = max(1, this / T)


def _pow2floor(n: int) -> int:
    return 1 << (n.bit_length() - 1) if n >= 1 else 1


def _clamp_T(T: int) -> int:
    return min(1024, max(32, pow2ceil(T)))


def tp_config(
    lay: ShardLayout,
    M: int,
    elem_bytes: int,
    sm_count: int,
    vec_max: Optional[int] = None,
) -> TPConfig:
    """A closed-form pick (no table), the row class's rule applied to the shard: fewer rows than SMs -> a cluster of
    ``min(16, pow2floor(512 / M))`` CTAs per row with ~256 shard bytes per thread; otherwise one CTA per row (up to four
    rows per CTA of ~128 threads) with ~512 shard bytes per thread. Every pick is a member of :func:`all_valid_tp_configs`,
    so it never changes bits (the wire format does not depend on it at all)."""
    sm = int(sm_count)
    return _rule(lay, int(M), int(elem_bytes), sm, vec_max)


@functools.lru_cache(maxsize=65536)
def _rule(
    lay: ShardLayout, M: int, elem_bytes: int, sm: int, vec_max: Optional[int]
) -> TPConfig:
    vec = (
        max_vec_elems(lay.hi - lay.lo, elem_bytes)
        if vec_max is None
        else min(vec_max, max_vec_elems(lay.hi - lay.lo, elem_bytes))
    )
    sb = (lay.hi - lay.lo) * elem_bytes
    space = all_valid_tp_configs(lay, elem_bytes, vec)
    if sm > M:
        C = min(
            16,
            max(1, _pow2floor(CLUSTER_ROWS // max(M, 1))),
            max(1, _pow2floor(sb // CLUSTER_MIN_SHARE)),
        )
        while C >= 1:
            T = _clamp_T(-(-(sb // C) // BYTES_PER_THREAD_LAT))
            while T >= 32:
                cand = TPConfig(T, 1, vec, C)
                if cand in space:
                    return cand
                T //= 2
            C //= 2
    T = _clamp_T(-(-sb // BYTES_PER_THREAD_TP))
    rows = 1 if sm > M else max(1, min(4, THREADS_PER_CTA_TARGET // T))
    while T >= 32:
        for r in (rows, 1):
            cand = TPConfig(T, r, vec)
            if cand in space:
                return cand
        T //= 2
    return space[0]


# ----------------------------------------------------------------------------------- the merge and the sharded outputs
MERGE_SMEM_BUDGET = (
    160 * 1024
)  # bytes of staged partials per CTA of the merge kernel (32 warps: warp = tree lane, lane = row)
MERGE_MAX_ROWS = 8  # rows per CTA of the warp-mapped merge kernel (<= 32 lanes; 8 keeps two CTAs per SM resident and fills the grid at M = 4096)
MERGE_LANE_MAPPED_ROWS = (
    4  # rows per CTA of the lane-mapped merge kernel (one warp per row)
)


def merge_rows_per_cta(row_stride_floats: int) -> int:
    """Rows per CTA of the merge kernel: the largest power of two <= 32 whose staged partials fit the budget."""
    rows = MERGE_MAX_ROWS
    while rows > 1 and rows * row_stride_floats * 4 > MERGE_SMEM_BUDGET:
        rows //= 2
    return rows


OUT_ROWS_PER_CTA = 4  # the row-output kernel: a CTA covers OUT_ROWS_PER_CTA rows x one 256-column block tile
