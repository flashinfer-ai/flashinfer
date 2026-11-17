"""
Copyright (c) 2026 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Host-side tile plan of the Cake fused QKV projection + norm + RoPE + head
exchange.

The producer kernel is a persistent GEMM (``128 x block_n x 64`` tiles) whose
epilogue normalises, rotates and stores every tile straight into the
destination rank's symmetric receive slot.  Everything the kernel needs to know
about *where* a tile goes travels in a per-rank ``int64 [num_tiles, 8]`` tile
table (:data:`TILE_FIELDS`); the kernel walks the table statically (CTA ``b``
runs rows ``b, b + grid, ...``).  This module builds that table and the scalars
the launcher binds next to it, and resolves the per-shape plan parameters (tile
width, tile order, 2-CTA pair tiles) from the row geometry alone.

Row geometry (per rank ``r`` of ``W`` ranks): ``x`` is ``[B, S_local, hidden]``,
``M = B * S_local`` producer rows, ``K = hidden``, ``N = 3 * H * D``; destination
``p`` receives head band ``[p * H_local, (p + 1) * H_local)`` of every rank's
tokens (``H_local = H / W``), in layout ``"bshd"`` (three ``[B, S_global,
H_local, D]`` planes for Q, K, V) or ``"fused"`` (one ``[B, S_global, H_local,
3 * D]`` payload).
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import torch

HEAD_DIM = 128
BLOCK_M = 128
BLOCK_K = 64
BLOCK_N_CHOICES: tuple[int, ...] = (256, 128)
# Readiness words per destination are capped: ``ceil(m_tiles / cap)`` m-tiles
# share one word when a rank produces more than ``cap`` m-tiles.
DEFAULT_WORDS_CAPACITY_PER_DEST = 512
NORM_KIND_V = 0
NORM_KIND_Q = 1
NORM_KIND_K = 2
# GEMM output group ``g`` (Q, K, V order of the fused weight) -> norm kind.
GEMM_GROUP_NORM_KIND: tuple[int, ...] = (NORM_KIND_Q, NORM_KIND_K, NORM_KIND_V)
TILE_FIELDS: tuple[str, ...] = (
    "mt",
    "n0",
    "dest",
    "norm_kind",
    "base_elem",
    "row_stride",
    "word",
    "last_ticket",
)
TILE_DEST = TILE_FIELDS.index("dest")
TILE_WORD = TILE_FIELDS.index("word")
TILE_LAST_TICKET = TILE_FIELDS.index("last_ticket")
# Column order of a cross-destination-band (``xband``) table: the eight fields above describe head 0 of a
# 256-column tile, ``dest1`` / ``base_elem1`` its head 1 (``word`` / ``last_ticket`` are shared; both bases hold
# the head's own position, so no head stride is added to either).
TILE_FIELDS_XB: tuple[str, ...] = (*TILE_FIELDS, "dest1", "base_elem1")
TILE_DEST1 = TILE_FIELDS_XB.index("dest1")
TILE_BASE_ELEM1 = TILE_FIELDS_XB.index("base_elem1")
# Tile order within a destination: ``mt_outer`` = token tiles outer;
# ``nt_outer`` = the destination's ``3 * n_sub`` column tiles outer, token tiles
# inner (the CTAs of one column tile share its weight box through L2);
# ``grouped`` = ``group_m`` token tiles per group, column tiles outer inside the
# group.  ``mt_outer`` is ``grouped`` with ``group_m = 1``, ``nt_outer`` is
# ``grouped`` with ``group_m = m_tiles``.
TILE_ORDERS: tuple[str, ...] = ("mt_outer", "nt_outer", "grouped")
# Rows per lockstep unit of the 2-CTA pair-tile producer: aligned row pairs
# share ``n0``; an odd run is completed with a pad row (``dest = -1``) whose CTA
# runs the GEMM in lockstep, stores nothing and takes no ticket.
PAIR_LOCKSTEP = 2
# ``nt_outer`` re-streams the activation once per column tile, free only while
# ``2 * M * K`` bytes stay L2-resident; above this the plan is ``grouped``.
NT_OUTER_A_BYTES_MAX = 32 << 20
# A grouped plan takes the largest ``group_m`` whose activation slice
# ``2 * group_m * 128 * K`` stays within this budget.
GROUP_A_SLICE_BYTES_MAX = 16 << 20
# 2-CTA pair tiles: every ``block_n = 256`` row and every row with at least
# ``PAIR_MIN_M`` producer rows pairs on the architectures that build the pair
# producer, unless the pair-tile gate selects the single-CTA producer for that
# row (``pair_off_large_m``); pair rows with ``M >= PAIR_MIN_M`` are grouped
# with at least ``PAIR_MIN_GROUPS`` token-tile groups per rank.
PAIR_BN = 256
PAIR_MIN_M = 4096
PAIR_MIN_GROUPS = 4
PAIR_ARCHES: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
# The smallest cross-destination-band threshold (``xband_min_m``): a cross-destination-band row below
# ``PAIR_MIN_M`` (a 128-column band row the band rule keeps single-CTA) runs the ``block_n = 256`` pair tile with
# the grouped order and ``m_tiles // PAIR_MIN_GROUPS`` token tiles per group.
XBAND_MIN_M_FLOOR = 2048
# Group sizes the fused exchange is delivered for; every per-(architecture, world size) table below has exactly
# these keys per architecture.
SUPPORTED_WORLD_SIZES: tuple[int, ...] = (2, 4, 8)
# Group scaling: ``group_m_scale`` multiplies the ``grouped`` group count of a row whose ``hidden >=
# GROUP_M_SCALE_MIN_K`` (and raises the activation-slice budget by the same factor); every other row keeps the
# baseline plan.  ``GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD`` is the shipped scale per architecture and world size
# (1 = the baseline plan): measured per (architecture, world size) -- SM100 groups of 4 and 2 ranks and SM103
# every group size double the group, SM100 groups of 8 ranks, SM90 and SM107 keep the baseline plan.  The scale changes
# the tile ORDER only (every destination element is still written exactly once; the readiness words are
# unchanged); ``FLASHINFER_QKV_EXCHANGE_GROUP_M_SCALE`` (see the Cake backend) overrides a session's scale.
GROUP_M_SCALES: tuple[int, ...] = (1, 2, 4)
GROUP_M_SCALE_MIN_K = 6144
GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD: dict[str, dict[int, int]] = {
    "sm_90a": {2: 1, 4: 1, 8: 1},
    "sm_100a": {2: 2, 4: 2, 8: 1},
    "sm_103a": {2: 2, 4: 2, 8: 2},
    # SM107 keeps the baseline plan on every group size (the 2-rank doubling was not adopted); the 8-rank entry
    # exists because every row covers SUPPORTED_WORLD_SIZES -- SM107 sessions admit 2 or 4 ranks only.
    "sm_107a": {2: 1, 4: 1, 8: 1},
}
LAYOUTS: tuple[str, ...] = ("bshd", "fused")
# SMs the persistent producer grid leaves free: one for the wait kernel, one
# slack (``producer_grid``); a fired ``reserved_sms_large_m`` gate selects 0.
RESERVED_SMS = 2
# Row-count gates of the host plan (``rows = batch * s_local`` of a call: the token
# rows this rank projects), in the order of ``SHAPE_GATE_NAMES``.
# ``words_capacity_small_m``: a call with ``rows <= T`` publishes one readiness
# word per destination (``words_capacity_per_dest = 1``: the wait kernel scans
# ``world_size`` words instead of one per 128-row tile and destination);
# ``reserved_sms_large_m``: a call with ``rows >= T`` runs the persistent producer
# grid over every SM (``reserved_sms = 0`` instead of ``RESERVED_SMS``);
# ``per_dest_large_m``: a call with ``rows >= T`` launches the producer once per
# destination run of its tile table (the first launch carries the folded
# rendezvous, the later ones are the plain producer, stream-ordered behind it)
# instead of one persistent grid; ``pair_off_large_m``: a ``block_n = 256`` call
# with ``rows >= T`` runs the single-CTA producer instead of the 2-CTA pair tile
# (the pair policy's tile order and ``group_m`` are kept; ``SHAPE_GATE_BN`` names
# the tile width the gate needs, so ``gate_rows`` takes the call's ``bn`` while
# the gate is on; a 128-column call never fires it).  ``0`` = the gate never
# fires; ``SHAPE_GATE_ALL_ROWS`` (spelled ``all`` in the environment variable and
# in ``describe()``) = the small-row word gate fires on every row count.  Every
# gate is inclusive at its threshold; the first three change neither the tile
# table nor the stored bits, the pair-tile gate changes the producer program of
# the rows it selects and needs that program delivered.  On the cells of
# ``SHAPE_GATE_K_TABLE`` the per-destination launch gate reads its threshold by
# the call's hidden size (K of the projection) instead of one value; the
# cross-destination-band gate (``xband_min_m``, ``XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD``)
# sits beside the four gates.  Measured per
# architecture (``SHAPE_GATE_DEFAULT_BY_ARCH``): SM100 / SM103 ship the small-row
# word gate and the per-destination launch gate, SM90 ships the small-row word
# gate and the large-row reserved-SM gate, SM107 ships no per-architecture gate;
# three (architecture, world size) cells differ from those tables
# (``SHAPE_GATE_DEFAULT_BY_ARCH_WORLD``, merged by ``default_shape_gates``): a
# group of 8 ranks on SM90 and groups of 2 and 4 ranks on SM107 ship the
# small-row word gate for every row.  Elsewhere the pair-tile gate ships nowhere
# (an opt-in on the pair architectures; SM90 builds no pair tile and refuses
# it); ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` (see the
# Cake backend) overrides a session's gates in either direction.
SHAPE_GATE_NAMES: tuple[str, ...] = (
    "words_capacity_small_m",
    "reserved_sms_large_m",
    "per_dest_large_m",
    "pair_off_large_m",
)
SHAPE_GATE_BN: dict[str, int] = {"pair_off_large_m": PAIR_BN}
SHAPE_GATE_OFF = 0
# The "every row" threshold of the small-row word gate (``rows <= SHAPE_GATE_ALL_ROWS`` holds for every row count a
# session can allocate); a large-row gate never takes it (``0`` is its "never fires" spelling).
SHAPE_GATE_ALL_ROWS = 2**31 - 1
SHAPE_GATE_ALL_ROWS_NAMES: tuple[str, ...] = ("words_capacity_small_m",)
SHAPE_GATE_DEFAULT_BY_ARCH: dict[str, dict[str, int]] = {
    # SM90: the small-row word gate (4096; every row for a group of 8 ranks, see SHAPE_GATE_DEFAULT_BY_ARCH_WORLD)
    # and the large-row reserved-SM gate (16384) on every group size; the per-destination launch gate off.
    "sm_90a": {
        "words_capacity_small_m": 4096,
        "reserved_sms_large_m": 16384,
        "per_dest_large_m": SHAPE_GATE_OFF,
        "pair_off_large_m": SHAPE_GATE_OFF,
    },
    "sm_100a": {
        "words_capacity_small_m": 4096,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": 16384,
        "pair_off_large_m": SHAPE_GATE_OFF,
    },
    "sm_103a": {
        "words_capacity_small_m": 4096,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": 16384,
        "pair_off_large_m": SHAPE_GATE_OFF,
    },
    # SM107: no gate in the per-architecture row; its 2- and 4-rank cells in SHAPE_GATE_DEFAULT_BY_ARCH_WORLD ship
    # the small-row word gate for every row.
    "sm_107a": {
        "words_capacity_small_m": SHAPE_GATE_OFF,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": SHAPE_GATE_OFF,
        "pair_off_large_m": SHAPE_GATE_OFF,
    },
}
# The (architecture, world size) cells whose shipped gates differ from ``SHAPE_GATE_DEFAULT_BY_ARCH``; only differing
# cells are listed (a redundant cell is refused below), every other cell is the per-architecture table.
SHAPE_GATE_DEFAULT_BY_ARCH_WORLD: dict[tuple[str, int], dict[str, int]] = {
    # SM90, 8 ranks: one readiness word per destination for every row count.
    ("sm_90a", 8): {"words_capacity_small_m": SHAPE_GATE_ALL_ROWS},
    # SM107, 2 and 4 ranks (the sizes it admits): the same.
    ("sm_107a", 2): {"words_capacity_small_m": SHAPE_GATE_ALL_ROWS},
    ("sm_107a", 4): {"words_capacity_small_m": SHAPE_GATE_ALL_ROWS},
}


def ceil_div(a: int, b: int) -> int:
    return -(-a // b)


def heads_divisibility_error(heads: int, world_size: int) -> str:
    return (
        f"head count {heads} must be divisible by world_size={world_size}: the fused "
        f"QKV exchange sends head bands of {heads} / {world_size} heads to every rank"
    )


@dataclass(frozen=True)
class Geometry:
    """One call's row geometry under the sequence-parallel split."""

    world_size: int
    batch: int
    s_local: int
    hidden: int
    head_dim: int = HEAD_DIM

    def __post_init__(self) -> None:
        for name in ("world_size", "batch", "s_local", "hidden", "head_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive int, got {value!r}")
        if self.head_dim != HEAD_DIM:
            raise ValueError(
                f"the producer fixes head_dim={HEAD_DIM}, got {self.head_dim}"
            )
        if self.hidden % self.head_dim:
            raise ValueError(
                f"hidden={self.hidden} must be divisible by head_dim={self.head_dim}"
            )
        if self.hidden % BLOCK_K:
            raise ValueError(f"hidden={self.hidden} must be a multiple of {BLOCK_K}")
        if self.heads % self.world_size:
            raise ValueError(heads_divisibility_error(self.heads, self.world_size))

    @property
    def heads(self) -> int:
        return self.hidden // self.head_dim

    @property
    def s_global(self) -> int:
        return self.s_local * self.world_size

    @property
    def h_local(self) -> int:
        return self.heads // self.world_size

    @property
    def rows(self) -> int:
        """Producer rows per rank (``M = B * S_local``)."""

        return self.batch * self.s_local

    @property
    def c_band(self) -> int:
        """Columns of one destination's head band."""

        return self.h_local * self.head_dim

    @property
    def slot_elems(self) -> int:
        """Elements of one receive slot: ``B * S_global * H_local * 3 * D``."""

        return self.batch * self.s_global * self.h_local * 3 * self.head_dim


def block_n_for(geometry: Geometry) -> int:
    """256 unless the destination band ``H_local * 128`` is not a multiple of 256 (then 128)."""

    return 256 if geometry.c_band % 256 == 0 else 128


@dataclass(frozen=True)
class Policy:
    """The per-row plan parameters resolved from the geometry (with the arithmetic behind them)."""

    bn: int
    order: str
    group_m: int
    pair: bool
    a_bytes: int
    m_tiles: int
    reasons: tuple[str, ...]
    group_m_scale: int = 1

    def summary(self) -> dict[str, Any]:
        return {
            "bn": self.bn,
            "order": self.order,
            "group_m": self.group_m,
            "pair": self.pair,
            "a_bytes": self.a_bytes,
            "m_tiles": self.m_tiles,
            "group_m_scale": self.group_m_scale,
            "reasons": list(self.reasons),
        }


def pair_geometry(geometry: Geometry) -> bool:
    """``bn == PAIR_BN or M >= PAIR_MIN_M`` with ``bn`` from :func:`block_n_for`."""

    return block_n_for(geometry) == PAIR_BN or geometry.rows >= PAIR_MIN_M


def plan_policy(
    geometry: Geometry,
    *,
    arch: str,
    pair_available: bool = True,
    group_m_scale: int | None = None,
    xband: bool = False,
) -> Policy:
    """Resolve tile width, tile order and the pair tile from ``(M, K, H_local)``.

    1. ``bn``: :func:`block_n_for`.
    2. Tile order of 1-CTA rows: ``2 * M * K <= NT_OUTER_A_BYTES_MAX`` selects
       ``nt_outer``; above it ``grouped`` with the largest ``group_m`` whose
       activation slice stays within ``GROUP_A_SLICE_BYTES_MAX``.
    3. Pair tile: :func:`pair_geometry` on an architecture of ``PAIR_ARCHES``
       whose loader delivers the pair producer (``pair_available``).  Pair rows
       with ``M >= PAIR_MIN_M`` (or above the L2 budget) are ``grouped`` with the
       largest even ``group_m`` within the slice budget and at most
       ``m_tiles // PAIR_MIN_GROUPS``; smaller pair rows keep ``nt_outer``.
    4. ``group_m_scale`` (``None`` = the architecture's default for the
       geometry's world size, ``GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD[arch]
       [world_size]``): a ``grouped`` row with ``hidden >= GROUP_M_SCALE_MIN_K``
       takes ``group_m`` times the scale (capped at ``m_tiles``, kept even under
       the pair) within the slice budget raised by the same factor; other rows
       are unchanged.
    5. ``xband`` (a fired cross-destination-band gate): a row below
       ``PAIR_MIN_M`` with at least ``XBAND_MIN_M_FLOOR`` rows takes the pair
       geometry and the grouped pair order of rule 3.
    """

    if group_m_scale is None:
        group_m_scale = default_group_m_scale(arch, geometry.world_size)
    if isinstance(group_m_scale, bool) or group_m_scale not in GROUP_M_SCALES:
        raise ValueError(
            f"group_m_scale must be one of {GROUP_M_SCALES}, got {group_m_scale!r}"
        )
    bn = block_n_for(geometry)
    M, K = geometry.rows, geometry.hidden
    m_tiles = ceil_div(M, BLOCK_M)
    a_bytes = 2 * M * K
    reasons = [f"bn = {bn} (band = {geometry.c_band} columns)"]
    slice_cap = max(1, GROUP_A_SLICE_BYTES_MAX // (2 * BLOCK_M * K))
    if not isinstance(xband, bool):
        raise ValueError(f"xband must be a bool, got {xband!r}")
    shape_pair = pair_geometry(geometry)
    xband_small = xband and not shape_pair and M >= XBAND_MIN_M_FLOOR
    if xband_small:
        shape_pair = True
    pair = shape_pair and arch in PAIR_ARCHES and pair_available
    if shape_pair:
        why = " or ".join(
            ([f"bn {bn} == {PAIR_BN}"] if bn == PAIR_BN else [])
            + ([f"M {M} >= {PAIR_MIN_M}"] if M >= PAIR_MIN_M else [])
            + ([f"xband M {M} >= {XBAND_MIN_M_FLOOR}"] if xband_small else [])
        )
        if arch not in PAIR_ARCHES:
            launcher = f"{arch} has no 2-CTA pair producer"
        elif not pair_available:
            launcher = f"{arch} delivers no pair producer for bn {bn}"
        else:
            launcher = f"{arch} runs the 2-CTA pair producer"
        reasons.append(
            f"{why} -> pair geometry; {launcher} -> pair {'on' if pair else 'off'}"
        )
    else:
        reasons.append(f"bn {bn} != {PAIR_BN} and M {M} < {PAIR_MIN_M} -> 1-CTA tiles")
    if pair and (M >= PAIR_MIN_M or a_bytes > NT_OUTER_A_BYTES_MAX or xband_small):
        order = "grouped"
        cap = min(m_tiles // PAIR_MIN_GROUPS, slice_cap)
        group_m = max(PAIR_LOCKSTEP, cap - cap % PAIR_LOCKSTEP)
        group_m = min(group_m, m_tiles)
        reasons.append(
            f"pair: grouped, group_m = largest even <= min(m_tiles // {PAIR_MIN_GROUPS}, "
            f"{GROUP_A_SLICE_BYTES_MAX >> 20} MiB / (2 * 128 * K) = {slice_cap}) = {group_m}"
        )
    elif pair:
        order, group_m = "nt_outer", m_tiles
        reasons.append(f"pair: A bytes {a_bytes} <= {NT_OUTER_A_BYTES_MAX} -> nt_outer")
    elif a_bytes <= NT_OUTER_A_BYTES_MAX:
        order, group_m = "nt_outer", m_tiles
        reasons.append(f"A bytes {a_bytes} <= {NT_OUTER_A_BYTES_MAX} -> nt_outer")
    else:
        order, group_m = "grouped", max(1, min(m_tiles, slice_cap))
        reasons.append(
            f"A bytes {a_bytes} > {NT_OUTER_A_BYTES_MAX} -> grouped, group_m = {group_m}"
        )
    if group_m_scale != 1 and order == "grouped" and K >= GROUP_M_SCALE_MIN_K:
        scaled = min(m_tiles, group_m * group_m_scale)
        if pair and scaled % PAIR_LOCKSTEP and scaled != m_tiles:
            scaled -= scaled % PAIR_LOCKSTEP
        slice_bytes = 2 * scaled * BLOCK_M * K
        budget = GROUP_A_SLICE_BYTES_MAX * group_m_scale
        if slice_bytes > budget:
            raise ValueError(
                f"group_m_scale={group_m_scale}: A slice 2 * {scaled} * {BLOCK_M} * {K} = "
                f"{slice_bytes} exceeds the raised budget {budget}"
            )
        reasons.append(
            f"group_m_scale={group_m_scale} (K {K} >= {GROUP_M_SCALE_MIN_K}): group_m "
            f"{group_m} -> {scaled} within {budget >> 20} MiB"
        )
        group_m = scaled
    return Policy(
        bn=bn,
        order=order,
        group_m=group_m,
        pair=pair,
        a_bytes=a_bytes,
        m_tiles=m_tiles,
        reasons=tuple(reasons),
        group_m_scale=group_m_scale,
    )


def _resolve_order(
    order: str, group_m: int | None, pair: bool, m_tiles: int
) -> tuple[str, int]:
    if order not in TILE_ORDERS:
        raise ValueError(f"order must be one of {TILE_ORDERS}, got {order!r}")
    if not isinstance(pair, bool):
        raise ValueError(f"pair must be a bool, got {pair!r}")
    if group_m is not None and (
        isinstance(group_m, bool) or not isinstance(group_m, int) or group_m < 1
    ):
        raise ValueError(f"group_m must be a positive int, got {group_m!r}")
    if order == "mt_outer":
        if group_m not in (None, 1):
            raise ValueError(f"order='mt_outer' is group_m = 1, got group_m={group_m}")
        if pair:
            raise ValueError(
                "order='mt_outer' cannot satisfy the pair lockstep contract"
            )
        return order, 1
    if order == "nt_outer":
        if group_m not in (None, m_tiles):
            raise ValueError(
                f"order='nt_outer' is group_m = m_tiles = {m_tiles}, got group_m={group_m}"
            )
        return order, m_tiles
    if group_m is None:
        raise ValueError("order='grouped' needs group_m")
    effective = min(int(group_m), m_tiles)
    if pair and effective % PAIR_LOCKSTEP and effective != m_tiles:
        raise ValueError(
            f"order='grouped' with the pair tile needs group_m % {PAIR_LOCKSTEP} == 0, got {group_m}"
        )
    return order, effective


def _order_dest_runs(
    per_dest: Sequence[Sequence[Sequence[list[list[int]]]]], mode: str, tail_rows: int
) -> list[list[int]]:
    """Concatenate the exchange runs in the production order of ``mode`` (:data:`DEST_INTERLEAVE_MODES`).

    ``per_dest[d][gi][j]`` holds the rows (lockstep pad rows included) of the ``j``-th column tile of token-tile group
    ``gi`` of the ``d``-th destination of ``dest_order`` (the local destination last).  ``off``: destination blocks.
    ``il``: token-tile group outer, column tile, then the destinations in ``dest_order``.  ``lt``: ``il`` with the
    trailing local runs of at least ``tail_rows`` rows moved to the end of the table.  ``ls``: the remote destination
    blocks with the local runs spread evenly between them."""

    if mode == "off":
        return [
            row
            for blocks in per_dest
            for group in blocks
            for run in group
            for row in run
        ]
    if mode in ("il", "lt"):
        seq: list[tuple[int, list[list[int]]]] = []
        for gi in range(len(per_dest[0])):
            width = max(len(blocks[gi]) for blocks in per_dest)
            for j in range(width):
                for d, blocks in enumerate(per_dest):
                    if j < len(blocks[gi]):
                        seq.append((d, blocks[gi][j]))
        if mode == "lt":
            local, tail, rows = len(per_dest) - 1, set(), 0
            for index in range(len(seq) - 1, -1, -1):
                if rows >= tail_rows:
                    break
                if seq[index][0] == local:
                    tail.add(index)
                    rows += len(seq[index][1])
            seq = [s for i, s in enumerate(seq) if i not in tail] + [
                s for i, s in enumerate(seq) if i in tail
            ]
        return [row for _, run in seq for row in run]
    if mode != "ls":
        raise ValueError(
            f"dest_interleave must be one of {DEST_INTERLEAVE_MODES}, got {mode!r}"
        )
    local_runs = [run for group in per_dest[-1] for run in group]
    remote = per_dest[:-1]
    if not remote:
        return [row for run in local_runs for row in run]
    bounds = [len(local_runs) * i // len(remote) for i in range(len(remote) + 1)]
    out: list[list[int]] = []
    for r, blocks in enumerate(remote):
        chunk = local_runs[bounds[r] : bounds[r + 1]]
        runs = [run for group in blocks for run in group]
        placed = 0
        for i, run in enumerate(runs):
            out.extend(run)
            while placed < ((i + 1) * len(chunk)) // len(runs):
                out.extend(chunk[placed])
                placed += 1
        for run in chunk[placed:]:
            out.extend(run)
    return out


def _plan_rows(
    *,
    rank: int,
    dest_order: Sequence[int],
    m_tiles: int,
    n_sub: int,
    block_n: int,
    heads: int,
    head_dim: int,
    c_band: int,
    layout: str,
    plane: int,
    s_local: int,
    row_stride: int,
    group_m: int,
    lockstep: int,
    m_tiles_per_word: int,
    dest_interleave: str = "off",
    interleave_tail_rows: int = 0,
) -> tuple[list[list[int]], int]:
    """Tile rows in production order (:func:`_order_dest_runs` of ``dest_interleave``); returns
    ``(rows, pad_rows)``."""

    G = m_tiles_per_word
    per_dest: list[list[list[list[list[int]]]]] = []
    pads = 0

    def tile(dest: int, mt: int, nt: int) -> list[int]:
        g, sub = divmod(nt, n_sub)
        n0 = g * heads * head_dim + dest * c_band + sub * block_n
        hl0 = sub * (block_n // head_dim)
        if layout == "bshd":
            base_elem = g * plane + rank * s_local * row_stride + hl0 * head_dim
        else:
            base_elem = rank * s_local * row_stride + hl0 * 3 * head_dim + g * head_dim
        word = mt // G
        in_word = min(G, m_tiles - word * G)
        return [
            mt,
            n0,
            dest,
            GEMM_GROUP_NORM_KIND[g],
            base_elem,
            row_stride,
            word,
            3 * n_sub * in_word - 1,
        ]

    for dest in dest_order:
        blocks: list[list[list[list[int]]]] = []
        for g0 in range(0, m_tiles, group_m):
            mts = range(g0, min(m_tiles, g0 + group_m))
            group: list[list[list[int]]] = []
            for nt in range(3 * n_sub):
                run = [tile(int(dest), mt, nt) for mt in mts]
                while lockstep > 1 and len(run) % lockstep:
                    pad = list(run[-1])
                    pad[TILE_DEST] = -1
                    pad[TILE_WORD] = 0
                    pad[TILE_LAST_TICKET] = 0
                    run.append(pad)
                    pads += 1
                group.append(run)
            blocks.append(group)
        per_dest.append(blocks)
    return _order_dest_runs(per_dest, dest_interleave, interleave_tail_rows), pads


def _plan_rows_xband(
    *,
    rank: int,
    dest_order: Sequence[int],
    m_tiles: int,
    heads: int,
    head_dim: int,
    h_local: int,
    layout: str,
    plane: int,
    s_local: int,
    row_stride: int,
    group_m: int,
    lockstep: int,
    m_tiles_per_word: int,
    dest_interleave: str = "off",
    interleave_tail_rows: int = 0,
) -> tuple[list[list[int]], int]:
    """:data:`TILE_FIELDS_XB` rows of the cross-destination-band plan in production order (:func:`_order_dest_runs`
    of ``dest_interleave``); returns ``(rows, pad_rows)``.

    Each GEMM group ``g`` (Q, K, V) is cut into ``H * head_dim / 256`` tiles; tile ``k`` covers the global heads
    ``2k`` and ``2k + 1``, each with its own destination ``head // H_local`` and the slot base of :func:`_plan_rows`
    at local head ``head % H_local``.  A tile belongs to the destination of its first head: destination blocks in
    ``dest_order``, inside a block the ``group_m`` m-tile groups, the block's tiles (``g`` outer, ``k`` inner) outer
    and the group's m-tiles inner -- the order of :func:`_plan_rows`.  ``word = mt // G``; ``last_ticket = 3 *
    H_local * in_word - 1`` counts one ticket per head.  ``lockstep > 1`` pads every same-``n0`` run like
    :func:`_plan_rows` (``dest = dest1 = -1``, ``word = 0``, ``last_ticket = 0``)."""

    G = m_tiles_per_word
    heads_per_tile = XBAND_BN // head_dim
    tiles_per_group = heads * head_dim // XBAND_BN
    per_dest: list[list[list[list[list[int]]]]] = []
    pads = 0

    def base_of(g: int, hl: int) -> int:
        if layout == "bshd":
            return g * plane + rank * s_local * row_stride + hl * head_dim
        return rank * s_local * row_stride + hl * 3 * head_dim + g * head_dim

    def tile(mt: int, g: int, k: int) -> list[int]:
        h0, h1 = heads_per_tile * k, heads_per_tile * k + 1
        word = mt // G
        in_word = min(G, m_tiles - word * G)
        return [
            mt,
            g * heads * head_dim + k * XBAND_BN,
            h0 // h_local,
            GEMM_GROUP_NORM_KIND[g],
            base_of(g, h0 % h_local),
            row_stride,
            word,
            3 * h_local * in_word - 1,
            h1 // h_local,
            base_of(g, h1 % h_local),
        ]

    for dest in dest_order:
        owned = [
            (g, k)
            for g in range(3)
            for k in range(tiles_per_group)
            if (heads_per_tile * k) // h_local == int(dest)
        ]
        blocks: list[list[list[list[int]]]] = []
        for g0 in range(0, m_tiles, group_m):
            mts = range(g0, min(m_tiles, g0 + group_m))
            group: list[list[list[int]]] = []
            for g, k in owned:
                run = [tile(mt, g, k) for mt in mts]
                while lockstep > 1 and len(run) % lockstep:
                    pad = list(run[-1])
                    pad[TILE_DEST] = -1
                    pad[TILE_DEST1] = -1
                    pad[TILE_WORD] = 0
                    pad[TILE_LAST_TICKET] = 0
                    run.append(pad)
                    pads += 1
                group.append(run)
            blocks.append(group)
        per_dest.append(blocks)
    return _order_dest_runs(per_dest, dest_interleave, interleave_tail_rows), pads


@dataclass(frozen=True)
class TilePlan:
    """One rank's tile table (CPU ``int64 [num_tiles, 8]``, ``[num_tiles, 10]`` for a cross-destination-band plan)
    and the scalars bound next to it."""

    tiles: torch.Tensor
    layout: str
    world: int
    rank: int
    batch: int
    s_local: int
    s_global: int
    heads: int
    head_dim: int
    h_local: int
    M: int
    N: int
    K: int
    bn: int
    m_tiles: int
    n_sub: int
    c_band: int
    num_tiles: int
    num_words: int
    m_tiles_per_word: int
    words_capacity_per_dest: int
    dest_order: tuple[int, ...]
    row_stride: int
    head_stride: int
    batch_stride: int
    slot_elems: int
    plane: int
    order: str
    group_m: int
    pair: bool
    num_pad_rows: int
    xband: bool = False
    # Destination order of the table (:data:`DEST_INTERLEAVE_MODES`; ``off`` = destination blocks) and, for ``lt``,
    # the local-tail row budget (``0`` otherwise).  Only the order of the rows changes.
    dest_interleave: str = "off"
    interleave_tail_rows: int = 0

    @property
    def lockstep(self) -> int:
        return PAIR_LOCKSTEP if self.pair else 1

    @property
    def tile_fields(self) -> tuple[str, ...]:
        """The table's columns: :data:`TILE_FIELDS_XB` for a cross-destination-band plan, else
        :data:`TILE_FIELDS`."""

        return TILE_FIELDS_XB if self.xband else TILE_FIELDS

    @property
    def num_real_tiles(self) -> int:
        return self.num_tiles - self.num_pad_rows

    def summary(self) -> dict[str, Any]:
        record = {
            "layout": self.layout,
            "world": self.world,
            "rank": self.rank,
            "M": self.M,
            "N": self.N,
            "K": self.K,
            "block_n": self.bn,
            "m_tiles": self.m_tiles,
            "n_sub": self.n_sub,
            "num_tiles": self.num_tiles,
            "num_words": self.num_words,
            "m_tiles_per_word": self.m_tiles_per_word,
            "words_capacity_per_dest": self.words_capacity_per_dest,
            "dest_order": list(self.dest_order),
            "row_stride": self.row_stride,
            "head_stride": self.head_stride,
            "batch_stride": self.batch_stride,
            "slot_elems": self.slot_elems,
            "plane": self.plane,
            "order": self.order,
            "group_m": self.group_m,
            "pair": self.pair,
            "num_pad_rows": self.num_pad_rows,
        }
        if self.xband:
            record["xband"] = True
            record["tile_fields"] = list(TILE_FIELDS_XB)
        if self.dest_interleave != "off":
            record["dest_interleave"] = self.dest_interleave
            record["interleave_tail_rows"] = self.interleave_tail_rows
        return record


def chunk_plan(
    geometry: Geometry,
    layout: str,
    rank: int,
    *,
    block_n: int,
    words_capacity_per_dest: int = DEFAULT_WORDS_CAPACITY_PER_DEST,
    order: str = "mt_outer",
    group_m: int | None = None,
    pair: bool = False,
    xband: bool = False,
    dest_interleave: str = "off",
    interleave_tail_rows: int = 0,
) -> TilePlan:
    """The exchange plan of source ``rank`` as a tile table.

    Destination order: sender-cyclic remote first ``(r + 1, ..., r + W - 1) mod
    W``, local ``r`` last.  Readiness words: ``G = ceil(m_tiles /
    words_capacity_per_dest)`` token tiles per word, ``word = mt // G``.
    ``xband`` builds the cross-destination-band table of
    :func:`_plan_rows_xband` (``block_n = XBAND_BN`` pair plan of an
    :func:`xband_eligible` geometry; ``n_sub`` = ``H_local`` head tickets per
    destination band and GEMM group).  ``dest_interleave`` reorders the rows
    (:func:`_order_dest_runs`; ``interleave_tail_rows`` is the positive ``lt``
    local-tail budget and ``0`` for every other mode); every row keeps its
    destination, word and ticket.
    """

    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    world = geometry.world_size
    if isinstance(rank, bool) or not isinstance(rank, int) or not 0 <= rank < world:
        raise ValueError(f"rank must be an int in [0, {world}), got {rank!r}")
    if block_n not in BLOCK_N_CHOICES:
        raise ValueError(f"block_n must be one of {BLOCK_N_CHOICES}, got {block_n!r}")
    if (
        isinstance(words_capacity_per_dest, bool)
        or not isinstance(words_capacity_per_dest, int)
        or words_capacity_per_dest <= 0
    ):
        raise ValueError(
            f"words_capacity_per_dest must be a positive int, got {words_capacity_per_dest!r}"
        )
    heads, head_dim, h_local, c_band = (
        geometry.heads,
        geometry.head_dim,
        geometry.h_local,
        geometry.c_band,
    )
    if not isinstance(xband, bool):
        raise ValueError(f"xband must be a bool, got {xband!r}")
    if dest_interleave not in DEST_INTERLEAVE_MODES:
        raise ValueError(
            f"dest_interleave must be one of {DEST_INTERLEAVE_MODES}, got {dest_interleave!r}"
        )
    if (
        isinstance(interleave_tail_rows, bool)
        or not isinstance(interleave_tail_rows, int)
        or (
            interleave_tail_rows <= 0
            if dest_interleave == "lt"
            else interleave_tail_rows != 0
        )
    ):
        raise ValueError(
            "interleave_tail_rows must be a positive int for dest_interleave='lt' and 0 otherwise, got "
            f"{interleave_tail_rows!r} (dest_interleave={dest_interleave!r})"
        )
    if xband:
        if block_n != XBAND_BN or not pair:
            raise ValueError(
                f"a cross-destination-band plan is a block_n={XBAND_BN} pair plan, got block_n={block_n} pair={pair}"
            )
        if not xband_eligible(geometry):
            raise ValueError(
                f"a cross-destination-band plan needs a destination band that is not a multiple of {XBAND_BN} "
                f"columns while H * head_dim is, got H={heads} W={geometry.world_size} band={c_band}"
            )
    elif c_band % block_n:
        raise ValueError(
            f"block_n={block_n} must divide the destination band of {c_band} columns"
        )
    batch, s_local, s_global = geometry.batch, geometry.s_local, geometry.s_global
    M, K, N = geometry.rows, geometry.hidden, 3 * heads * head_dim
    m_tiles = ceil_div(M, BLOCK_M)
    n_sub = h_local if xband else c_band // block_n
    G = ceil_div(m_tiles, words_capacity_per_dest)
    num_words = ceil_div(m_tiles, G)
    dest_order = tuple((rank + k) % world for k in range(1, world)) + (rank,)
    if layout == "bshd":
        row_stride = h_local * head_dim
        head_stride = head_dim
        plane = batch * s_global * h_local * head_dim
    else:
        row_stride = h_local * 3 * head_dim
        head_stride = 3 * head_dim
        plane = 0
    batch_stride = s_global * row_stride
    order, group_m_eff = _resolve_order(order, group_m, pair, m_tiles)
    lockstep = PAIR_LOCKSTEP if pair else 1
    if xband:
        rows, num_pad_rows = _plan_rows_xband(
            rank=rank,
            dest_order=dest_order,
            m_tiles=m_tiles,
            heads=heads,
            head_dim=head_dim,
            h_local=h_local,
            layout=layout,
            plane=plane,
            s_local=s_local,
            row_stride=row_stride,
            group_m=group_m_eff,
            lockstep=lockstep,
            m_tiles_per_word=G,
            dest_interleave=dest_interleave,
            interleave_tail_rows=interleave_tail_rows,
        )
        real_rows = m_tiles * 3 * heads * head_dim // XBAND_BN
        width = len(TILE_FIELDS_XB)
    else:
        rows, num_pad_rows = _plan_rows(
            rank=rank,
            dest_order=dest_order,
            m_tiles=m_tiles,
            n_sub=n_sub,
            block_n=block_n,
            heads=heads,
            head_dim=head_dim,
            c_band=c_band,
            layout=layout,
            plane=plane,
            s_local=s_local,
            row_stride=row_stride,
            group_m=group_m_eff,
            lockstep=lockstep,
            m_tiles_per_word=G,
            dest_interleave=dest_interleave,
            interleave_tail_rows=interleave_tail_rows,
        )
        real_rows = world * m_tiles * 3 * n_sub
        width = len(TILE_FIELDS)
    tiles = torch.tensor(rows, dtype=torch.int64).contiguous()
    num_tiles = len(rows)
    if num_tiles - num_pad_rows != real_rows or tuple(tiles.shape) != (
        num_tiles,
        width,
    ):
        raise AssertionError("tile table inconsistent with the plan")
    return TilePlan(
        tiles=tiles,
        layout=layout,
        world=world,
        rank=rank,
        batch=batch,
        s_local=s_local,
        s_global=s_global,
        heads=heads,
        head_dim=head_dim,
        h_local=h_local,
        M=M,
        N=N,
        K=K,
        bn=block_n,
        m_tiles=m_tiles,
        n_sub=n_sub,
        c_band=c_band,
        num_tiles=num_tiles,
        num_words=num_words,
        m_tiles_per_word=G,
        words_capacity_per_dest=words_capacity_per_dest,
        dest_order=dest_order,
        row_stride=row_stride,
        head_stride=head_stride,
        batch_stride=batch_stride,
        slot_elems=geometry.slot_elems,
        plane=plane,
        order=order,
        group_m=group_m_eff,
        pair=pair,
        num_pad_rows=num_pad_rows,
        xband=xband,
        dest_interleave=dest_interleave,
        interleave_tail_rows=interleave_tail_rows,
    )


def plan_for(
    geometry: Geometry,
    layout: str,
    rank: int,
    *,
    arch: str,
    pair_available: bool = True,
    words_capacity_per_dest: int = DEFAULT_WORDS_CAPACITY_PER_DEST,
    group_m_scale: Optional[int] = None,
    pair: Optional[bool] = None,
    xband: bool = False,
    bn128_max_m: Optional[int] = None,
    dest_interleave: str = "off",
    reserved_sms: int = RESERVED_SMS,
) -> tuple[Policy, TilePlan]:
    """Policy and tile plan of one row for one rank (``group_m_scale`` ``None`` = the architecture's default for
    the geometry's world size).  ``pair`` overrides the policy's pair tile (``False`` = the single-CTA producer with
    the policy's tile order and ``group_m``, what a fired ``pair_off_large_m`` gate selects); ``None`` keeps it.
    ``xband`` (what a fired ``xband_min_m`` gate selects) builds the cross-destination-band plan: the
    ``block_n = XBAND_BN`` pair tiles with the policy's tile order and ``group_m``; the policy (``bn`` of the band)
    must be the pair tile (below ``PAIR_MIN_M`` the policy takes the pair geometry, :func:`plan_policy`).
    ``bn128_max_m`` is the largest row count of the 128-column tile table (:func:`default_bn128_max_m`; ``None`` = the
    table's cell of the geometry, ``0`` = off): a row of a 256-column band at or below it that is not a
    cross-destination-band row takes the 128-column tile with the band policy's tile order, ``group_m`` and pair flag
    (on the pair architectures only a pair row: a single-CTA row keeps its tile).  ``dest_interleave`` (a mode of
    :data:`DEST_INTERLEAVE_MODES` the caller selected for this row, :func:`plan_row`) orders the table; ``lt`` reads
    its local-tail budget from ``reserved_sms`` (:func:`interleave_tail_rows`)."""

    policy = plan_policy(
        geometry,
        arch=arch,
        pair_available=pair_available,
        group_m_scale=group_m_scale,
        xband=bool(xband),
    )
    if pair is not None and bool(pair) != policy.pair:
        policy = dataclasses.replace(
            policy,
            pair=bool(pair),
            reasons=(
                *policy.reasons,
                f"pair override {policy.pair} -> {bool(pair)} (row-count gate): the "
                f"{'pair' if pair else 'single-CTA'} producer with order {policy.order}"
                f" / group_m {policy.group_m}",
            ),
        )
    max_m = (
        default_bn128_max_m(arch, geometry.world_size, geometry.hidden)
        if bn128_max_m is None
        else check_bn128_max_m(bn128_max_m)
    )
    if (
        not xband
        and max_m
        and policy.bn != HOPPER_BN128_BN
        and geometry.rows <= max_m
        and (policy.pair or arch not in PAIR_ARCHES)
    ):
        table = (
            "BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD"
            if arch in PAIR_ARCHES
            else "HOPPER_BN128_MAX_M_BY_ARCH_WORLD"
        )
        policy = dataclasses.replace(
            policy,
            bn=HOPPER_BN128_BN,
            reasons=(
                *policy.reasons,
                f"128-column tile table: ({arch}, W={geometry.world_size}, K {geometry.hidden}) M {geometry.rows} <= "
                f"{max_m} ({table}) -> block_n {HOPPER_BN128_BN} {'pair' if policy.pair else 'single-CTA'} (band "
                f"rule {policy.bn}); order {policy.order} / group_m {policy.group_m} unchanged",
            ),
        )
    if xband:
        if not policy.pair:
            raise ValueError(
                "the cross-destination-band plan is the 2-CTA pair tile, but the row's policy is the single-CTA "
                f"producer (pair_available={pair_available}, pair={pair})"
            )
        policy = dataclasses.replace(
            policy,
            reasons=(
                *policy.reasons,
                f"xband_min_m gate: band {geometry.c_band} % {XBAND_BN} != 0 and H * head_dim % {XBAND_BN} == 0 -> "
                f"block_n {XBAND_BN} cross-destination-band pair tiles ({len(TILE_FIELDS_XB)}-field table) with "
                f"order {policy.order} / group_m {policy.group_m}, one persistent launch",
            ),
        )
    if dest_interleave not in DEST_INTERLEAVE_MODES:
        raise ValueError(
            f"dest_interleave must be one of {DEST_INTERLEAVE_MODES}, got {dest_interleave!r}"
        )
    tail = 0
    if dest_interleave != "off":
        if arch not in DEST_INTERLEAVE_ARCHES:
            raise ValueError(
                f"dest_interleave={dest_interleave} is refused on {arch} (architectures: {DEST_INTERLEAVE_ARCHES})"
            )
        if dest_interleave == "lt":
            tail = interleave_tail_rows(
                arch, pair=bool(policy.pair or xband), reserved_sms=reserved_sms
            )
        policy = dataclasses.replace(
            policy,
            reasons=(
                *policy.reasons,
                f"destination interleave {dest_interleave} (DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD band row "
                f"{dest_interleave_band(arch, geometry.world_size, geometry.hidden)})"
                + (f", local tail {tail} rows" if tail else ""),
            ),
        )
    plan = chunk_plan(
        geometry,
        layout,
        rank,
        block_n=XBAND_BN if xband else policy.bn,
        words_capacity_per_dest=words_capacity_per_dest,
        order=policy.order,
        group_m=policy.group_m if policy.order == "grouped" else None,
        pair=policy.pair,
        xband=bool(xband),
        dest_interleave=dest_interleave,
        interleave_tail_rows=tail,
    )
    return policy, plan


def producer_grid(
    num_tiles: int, *, sm_count: int, reserved_sms: int, lockstep: int
) -> int:
    """Persistent grid: ``min(num_tiles, sm_count - reserved_sms)``, rounded down to the lockstep."""

    grid = max(1, min(int(num_tiles), int(sm_count) - int(reserved_sms)))
    if lockstep > 1:
        grid = max(lockstep, grid - grid % lockstep)
    return grid


def words_capacity_for_rows(max_rows: int) -> int:
    """Per-``(slot, source)`` readiness words a pad must hold for every row count up to ``max_rows``."""

    return min(DEFAULT_WORDS_CAPACITY_PER_DEST, ceil_div(int(max_rows), BLOCK_M))


def default_group_m_scale(arch: str, world_size: int) -> int:
    """The shipped ``group_m_scale`` of ``arch`` for a group of ``world_size`` ranks
    (``GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD``); an unknown architecture or world size is an error."""

    try:
        row = GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD[arch]
    except KeyError:
        raise ValueError(
            f"no group_m_scale default for architecture {arch!r}; known: "
            f"{tuple(GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD)}"
        ) from None
    try:
        return row[world_size]
    except KeyError:
        raise ValueError(
            f"no group_m_scale default for architecture {arch!r} at world_size={world_size!r}; known: {tuple(row)}"
        ) from None


def default_shape_gates(arch: str, *, world_size: int) -> dict[str, int]:
    """The shipped row-count gates of ``arch`` for a group of ``world_size`` ranks
    (:data:`SHAPE_GATE_DEFAULT_BY_ARCH` with the :data:`SHAPE_GATE_DEFAULT_BY_ARCH_WORLD` cell on top; a copy); an
    unknown architecture or world size is an error.  ``world_size`` is a required keyword: no caller gets the
    per-architecture table alone."""

    try:
        base = SHAPE_GATE_DEFAULT_BY_ARCH[arch]
    except KeyError:
        raise ValueError(
            f"no shape gates for architecture {arch!r}; known: {tuple(SHAPE_GATE_DEFAULT_BY_ARCH)}"
        ) from None
    if (
        isinstance(world_size, bool)
        or not isinstance(world_size, int)
        or world_size not in SUPPORTED_WORLD_SIZES
    ):
        raise ValueError(
            f"no shape gates for architecture {arch!r} at world_size={world_size!r}; known: "
            f"{SUPPORTED_WORLD_SIZES}"
        )
    return {**base, **SHAPE_GATE_DEFAULT_BY_ARCH_WORLD.get((arch, world_size), {})}


def shape_gate_spelling(value: Any) -> Any:
    """How a gate threshold reads in ``describe()``: ``"all"`` for :data:`SHAPE_GATE_ALL_ROWS`, else the value."""

    return "all" if value == SHAPE_GATE_ALL_ROWS else value


def spell_shape_gates(gates: Mapping[str, Any]) -> dict[str, Any]:
    """A gate table with every threshold spelled as ``describe()`` reports it (:func:`shape_gate_spelling`)."""

    return {name: shape_gate_spelling(value) for name, value in gates.items()}


def check_shape_gates(gates: Mapping[str, Any]) -> dict[str, int]:
    """Validate a gate table: exactly :data:`SHAPE_GATE_NAMES`, each an int ``>= 0`` (``0`` = off) or, for the
    small-row word gate only, ``"all"`` / :data:`SHAPE_GATE_ALL_ROWS` (every row).  Returns the int table."""

    if set(gates) != set(SHAPE_GATE_NAMES):
        raise ValueError(
            f"shape gates must name exactly {SHAPE_GATE_NAMES}, got {tuple(gates)}"
        )
    checked: dict[str, int] = {}
    for name in SHAPE_GATE_NAMES:
        value = gates[name]
        if isinstance(value, str) and value.lower() == "all":
            value = SHAPE_GATE_ALL_ROWS
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"shape gate {name} must be an int >= 0 (0 = off), got {value!r}"
            )
        if value > SHAPE_GATE_ALL_ROWS:
            raise ValueError(
                f"shape gate {name}={value} is above SHAPE_GATE_ALL_ROWS ({SHAPE_GATE_ALL_ROWS}): "
                "no row count reaches it, so it would be a second spelling of 'all' (every "
                f"row; {SHAPE_GATE_ALL_ROWS_NAMES}) or of 0 (a large-row gate that never fires)"
            )
        if value == SHAPE_GATE_ALL_ROWS and name not in SHAPE_GATE_ALL_ROWS_NAMES:
            raise ValueError(
                f"shape gate {name}: 'all' (every row) is a small-row gate spelling "
                f"({SHAPE_GATE_ALL_ROWS_NAMES}); a large-row gate that never fires is 0"
            )
        checked[name] = int(value)
    return checked


def _check_shape_gate_override_table() -> None:
    """Import-time check of :data:`SHAPE_GATE_DEFAULT_BY_ARCH_WORLD`: known cells, valid thresholds, each differing
    from the per-architecture table (one canonical form), the pair-tile gate on the pair architectures only."""

    for (arch, world_size), cell in SHAPE_GATE_DEFAULT_BY_ARCH_WORLD.items():
        base = SHAPE_GATE_DEFAULT_BY_ARCH[arch]
        if world_size not in SUPPORTED_WORLD_SIZES or not cell:
            raise ValueError(
                f"SHAPE_GATE_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered cell"
            )
        merged = check_shape_gates({**base, **cell})
        for name, value in cell.items():
            if merged[name] == base[name]:
                raise ValueError(
                    f"SHAPE_GATE_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}][{name!r}] equals the {arch} default"
                )
            if name in SHAPE_GATE_BN and value and arch not in PAIR_ARCHES:
                raise ValueError(
                    f"SHAPE_GATE_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}][{name!r}]: {arch} builds no pair tile"
                )


_check_shape_gate_override_table()


# Per-(architecture, world size) threshold of the per-destination launch gate by the call's hidden size ``K`` (the K
# of the projection): ``(arch, world_size) -> ((k_lo, threshold), ...)`` with ``k_lo`` ascending from 0.  A call reads
# the threshold of the last band whose ``k_lo <= K`` and launches the producer once per destination run iff
# ``rows >= threshold`` (inclusive; ``0`` = that band never fires).  A cell without an entry keeps the one-band
# ``per_dest_large_m`` of ``default_shape_gates``.  Measured per cell: on SM100 and SM103 the per-destination launch
# pays at a smaller row count for a large hidden size than for a small one.  The table applies while
# ``per_dest_large_m`` is left at its default: an explicit ``per_dest_large_m`` in
# ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` decides the gate alone, and ``per_dest_k_table=0`` there switches the table
# off (the one-band threshold of the per-architecture table).  It changes the launch structure only -- neither the
# tile table nor the stored bits.
SHAPE_GATE_K_TABLE: dict[tuple[str, int], tuple[tuple[int, int], ...]] = {
    ("sm_100a", 8): ((0, 16384), (4096, 32768), (6144, 16384)),
    ("sm_100a", 4): ((0, 32768), (6144, 16384), (8192, 8192)),
    ("sm_100a", 2): ((0, 16384), (8192, 8192)),
    ("sm_103a", 8): ((0, 32768), (6144, 16384)),
    ("sm_103a", 4): ((0, 32768), (6144, 16384)),
    ("sm_103a", 2): ((0, 32768), (4096, 16384)),
}


def check_shape_gate_k_table(bands: Any) -> tuple[tuple[int, int], ...]:
    """Validate one ``SHAPE_GATE_K_TABLE`` entry: a non-empty tuple of ``(k_lo, threshold)`` int pairs, ``k_lo``
    strictly ascending from 0, thresholds in ``[0, SHAPE_GATE_ALL_ROWS)``, neighbouring bands at different
    thresholds.  Returns the entry as a tuple of tuples."""

    if not isinstance(bands, (tuple, list)) or not bands:
        raise ValueError(
            f"a K table entry must be a non-empty tuple of (k_lo, threshold), got {bands!r}"
        )
    checked: list[tuple[int, int]] = []
    for band in bands:
        if (
            not isinstance(band, (tuple, list))
            or len(band) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in band
            )
        ):
            raise ValueError(
                f"K table band {band!r} must be a (k_lo, threshold) pair of ints >= 0"
            )
        k_lo, threshold = int(band[0]), int(band[1])
        if (not checked and k_lo != 0) or (checked and k_lo <= checked[-1][0]):
            raise ValueError(
                f"K table bands must start at k_lo = 0 and ascend strictly, got {bands!r}"
            )
        if threshold >= SHAPE_GATE_ALL_ROWS:
            raise ValueError(
                f"K table band {band!r}: the threshold must be below SHAPE_GATE_ALL_ROWS (0 = never fires)"
            )
        if checked and threshold == checked[-1][1]:
            raise ValueError(
                f"K table bands {bands!r}: neighbouring bands read the same threshold {threshold}; merge them"
            )
        checked.append((k_lo, threshold))
    return tuple(checked)


def _check_shape_gate_k_table() -> None:
    """Import-time check of :data:`SHAPE_GATE_K_TABLE`: delivered cells of architectures that ship the
    per-destination launch gate, valid entries, each differing from the cell's one-band threshold."""

    for (arch, world_size), bands in SHAPE_GATE_K_TABLE.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"SHAPE_GATE_K_TABLE[{(arch, world_size)!r}] is not a delivered cell"
            )
        if check_shape_gate_k_table(bands) != bands:
            raise ValueError(
                f"SHAPE_GATE_K_TABLE[{(arch, world_size)!r}] is not in canonical form"
            )
        one_band = {
            **SHAPE_GATE_DEFAULT_BY_ARCH[arch],
            **SHAPE_GATE_DEFAULT_BY_ARCH_WORLD.get((arch, world_size), {}),
        }["per_dest_large_m"]
        if not one_band:
            raise ValueError(
                f"SHAPE_GATE_K_TABLE[{(arch, world_size)!r}]: {arch} ships the per-destination launch gate off"
            )
        if all(threshold == one_band for _, threshold in bands):
            raise ValueError(
                f"SHAPE_GATE_K_TABLE[{(arch, world_size)!r}] equals the one-band per_dest_large_m={one_band}"
            )


_check_shape_gate_k_table()


def default_shape_gate_k_table(
    arch: str, *, world_size: int
) -> Optional[tuple[tuple[int, int], ...]]:
    """The shipped K table of the per-destination launch gate of ``arch`` for a group of ``world_size`` ranks
    (:data:`SHAPE_GATE_K_TABLE`; ``None`` = the one-band threshold of :func:`default_shape_gates`); an unknown
    architecture or world size is an error."""

    default_shape_gates(arch, world_size=world_size)
    return SHAPE_GATE_K_TABLE.get((arch, world_size))


def k_band_index(bands: Sequence[tuple[int, int]], hidden: int) -> int:
    """Index of the band a call with ``K = hidden`` reads (the last band whose ``k_lo <= hidden``)."""

    if isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1:
        raise ValueError(f"hidden must be a positive int, got {hidden!r}")
    index = 0
    for position, (k_lo, _threshold) in enumerate(bands):
        if hidden >= k_lo:
            index = position
    return index


def k_band_threshold(bands: Sequence[tuple[int, int]], hidden: int) -> int:
    """The per-destination launch threshold a call with ``K = hidden`` reads from a K table entry."""

    return int(bands[k_band_index(bands, hidden)][1])


# Cross-destination-band plans (``xband``).  On a row whose destination band is not a multiple of 256 columns
# (``block_n_for`` gives 128) while every Q / K / V group is (``H * head_dim % 256 == 0``; e.g. 8 ranks with hidden
# 3072: 24 heads, 3 per destination), the 2-CTA ``block_n = 256`` pair producer still runs: each 256-column tile covers
# two consecutive global heads, each head carries its own destination and slot base (:data:`TILE_FIELDS_XB`) and takes
# its own readiness ticket.  ``xband_min_m`` is a row-count gate beside the four gates: an eligible call with ``rows >=
# xband_min_m`` runs the cross-destination-band plan as one persistent launch (it takes precedence over the
# per-destination launch gate); ``0`` = never.  A threshold is at least ``XBAND_MIN_M_FLOOR`` (a fired row below
# ``PAIR_MIN_M`` runs the pair tile with the grouped pair order, :func:`plan_policy`) and applies on the pair
# architectures only.  ``XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD`` lists the cells that
# ship it (every other cell: 0): SM103 and SM100 groups of 8 ranks.  The cross-band tile stores through the staged
# row path, so on SM100, whose default producer build at 8 ranks stores through bulk row copies, the cross-band rows run
# the staged build of the same operand load width.  The tile table and the producer program change; the stored bits do
# not.
XBAND_BN = 256
XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD: dict[tuple[str, int], int] = {
    ("sm_103a", 8): 2048,
    ("sm_100a", 8): 2048,
}


def xband_eligible(geometry: Geometry) -> bool:
    """Whether a cross-destination-band plan exists for the geometry: the destination band is not a multiple of
    :data:`XBAND_BN` columns while ``H * head_dim`` is (a 256-column tile never straddles a Q / K / V group)."""

    return (
        geometry.c_band % XBAND_BN != 0
        and (geometry.heads * geometry.head_dim) % XBAND_BN == 0
    )


def check_xband_min_m(arch: str, value: Any) -> int:
    """Validate an ``xband_min_m`` threshold for ``arch``: an int ``>= 0`` (``0`` = off); a non-zero threshold is at
    least ``XBAND_MIN_M_FLOOR``, below ``SHAPE_GATE_ALL_ROWS``, and refused off the pair architectures."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"xband_min_m must be an int >= 0 (0 = off), got {value!r}")
    if value and value < XBAND_MIN_M_FLOOR:
        raise ValueError(
            f"xband_min_m={value} is below XBAND_MIN_M_FLOOR={XBAND_MIN_M_FLOOR}: the cross-destination-band plan is "
            f"the 2-CTA pair tile, which a 128-column row runs from {XBAND_MIN_M_FLOOR} rows only"
        )
    if value >= SHAPE_GATE_ALL_ROWS:
        raise ValueError(
            f"xband_min_m={value} is not below SHAPE_GATE_ALL_ROWS ({SHAPE_GATE_ALL_ROWS}): no row count reaches it; "
            "a gate that never fires is 0"
        )
    if value and arch not in PAIR_ARCHES:
        raise ValueError(
            f"xband_min_m={value} is refused on {arch}: the cross-destination-band plan is the 2-CTA pair tile "
            f"(pair architectures: {PAIR_ARCHES})"
        )
    return int(value)


def _check_xband_default_table() -> None:
    for (arch, world_size), value in XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered cell"
            )
        if not check_xband_min_m(arch, value):
            raise ValueError(
                f"XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}] is 0; list only the cells that ship it"
            )


_check_xband_default_table()


def default_xband_min_m(arch: str, *, world_size: int) -> int:
    """The shipped ``xband_min_m`` of ``arch`` for a group of ``world_size`` ranks
    (:data:`XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD`; 0 = off); an unknown architecture or world size is an error."""

    default_shape_gates(arch, world_size=world_size)
    return int(XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD.get((arch, world_size), 0))


# The 128-column tile table: per ``(arch, world_size)``, ``hidden -> largest M`` (inclusive) of the rows whose band
# rule (:func:`block_n_for`) gives 256 columns but which run the 128-column tile (:func:`plan_for`; the tile order,
# ``group_m`` and the pair flag stay the band policy's, the row-count gates read the band width).  One cell ships, SM90
# with 4 ranks at ``hidden = 3072``, ``M <= 1024``: there the last persistent wave of the 256-column plan is underfilled
# (288 tiles on 130 producer SMs, 2.22 waves) while the 128-column tile doubles the tile count (576 tiles, 4.43
# waves).  Not a cell of the pair architectures.
HOPPER_BN128_BN = 128
HOPPER_BN128_MAX_M_BY_ARCH_WORLD: dict[tuple[str, int], dict[int, int]] = {
    ("sm_90a", 4): {3072: 1024},
}


def check_bn128_max_m(value: Any) -> int:
    """Validate a 128-column tile bound: an int ``>= 0`` (``0`` = off) below ``SHAPE_GATE_ALL_ROWS``."""

    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"bn128_max_m must be an int >= 0 (0 = off), got {value!r}")
    if value >= SHAPE_GATE_ALL_ROWS:
        raise ValueError(
            f"bn128_max_m={value} is not below SHAPE_GATE_ALL_ROWS ({SHAPE_GATE_ALL_ROWS})"
        )
    return int(value)


def _check_hopper_bn128_table() -> None:
    for (arch, world_size), cells in HOPPER_BN128_MAX_M_BY_ARCH_WORLD.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or arch in PAIR_ARCHES
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"HOPPER_BN128_MAX_M_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered SM90 cell"
            )
        for hidden, max_m in cells.items():
            if (
                isinstance(hidden, bool)
                or not isinstance(hidden, int)
                or hidden <= 0
                or hidden % HEAD_DIM
                or not check_bn128_max_m(max_m)
            ):
                raise ValueError(
                    f"HOPPER_BN128_MAX_M_BY_ARCH_WORLD[{(arch, world_size)!r}][{hidden!r}] = {max_m!r}: hidden must "
                    f"be a positive multiple of {HEAD_DIM} and the bound a positive row count; list only the cells "
                    "that ship it"
                )


_check_hopper_bn128_table()


def hopper_bn128_max_m(arch: str, world_size: int, hidden: int) -> int:
    """The largest row count (inclusive) of ``(arch, world_size, hidden)`` that takes the 128-column tile
    (:data:`HOPPER_BN128_MAX_M_BY_ARCH_WORLD`); ``0`` = no cell."""

    cells = HOPPER_BN128_MAX_M_BY_ARCH_WORLD.get((arch, int(world_size)), {})
    return int(cells.get(int(hidden), 0))


# Rows that keep the baseline producer build under the default build: per ``(arch, world_size)``, ``(min_hidden,
# min_M)`` -- a row with ``hidden >= min_hidden`` and ``M >= min_M`` launches the baseline build's programs instead of
# the session's default build (the loader's ``PRODUCER_VARIANT_DEFAULT_BY_ARCH_WORLD``).  SM90 with 8 ranks at
# ``hidden >= 6144``, ``M >= 8192``: there the decoupled-storer build (``sw3``) measured 0.3-1.6 % slower than the
# baseline build, or even with it, on 8 x H100, so these rows keep the baseline build and its previous speed instead
# of a small loss.  A build named by the environment variable runs on every row.
PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD: dict[tuple[str, int], tuple[int, int]] = {
    ("sm_90a", 8): (6144, 8192),
}


def _check_producer_shipped_rows_table() -> None:
    for (arch, world_size), cell in PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered cell"
            )
        if (
            not isinstance(cell, tuple)
            or len(cell) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in cell
            )
            or cell[0] % HEAD_DIM
        ):
            raise ValueError(
                f"PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD[{(arch, world_size)!r}] = {cell!r}: (min_hidden, min_M) must be "
                f"positive ints, min_hidden a multiple of {HEAD_DIM}"
            )


_check_producer_shipped_rows_table()


def default_shipped_producer_row(
    arch: str, world_size: int, hidden: int, rows: int
) -> bool:
    """Whether a row of ``M = rows`` at ``(arch, world_size, hidden)`` launches the baseline producer build under the
    default build (:data:`PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD`)."""

    cell = PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD.get((arch, int(world_size)))
    return cell is not None and int(hidden) >= cell[0] and int(rows) >= cell[1]


# The 128-column pair tile table of the pair architectures: per ``(arch, world_size)``, ``hidden -> largest M``
# (inclusive) of the 256-column band rows whose policy is the 2-CTA pair tile but which run the 128-column pair tile
# (:func:`plan_for`; tile order, ``group_m`` and the row-count gates stay the band policy's; a single-CTA row and a
# cross-destination-band row keep their tile).  SM100 and SM103 with 8 ranks at ``hidden = 4096``, ``M <= 512``: the
# 256-column pair plan is 192 tiles on 146 producer SMs (1.32 waves), the 128-column pair plan 384 tiles (2.63 waves).
BLACKWELL_BN128_BN = 128
BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD: dict[tuple[str, int], dict[int, int]] = {
    ("sm_100a", 8): {4096: 512},
    ("sm_103a", 8): {4096: 512},
}


def _check_blackwell_bn128_table() -> None:
    for (arch, world_size), cells in BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or arch not in PAIR_ARCHES
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered pair-architecture "
                "cell"
            )
        for hidden, max_m in cells.items():
            if (
                isinstance(hidden, bool)
                or not isinstance(hidden, int)
                or hidden <= 0
                or hidden % HEAD_DIM
                or not check_bn128_max_m(max_m)
            ):
                raise ValueError(
                    f"BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD[{(arch, world_size)!r}][{hidden!r}] = {max_m!r}: hidden "
                    f"must be a positive multiple of {HEAD_DIM} and the bound a positive row count; list only the "
                    "cells that ship it"
                )


_check_blackwell_bn128_table()


def blackwell_bn128_max_m(arch: str, world_size: int, hidden: int) -> int:
    """The largest row count (inclusive) of ``(arch, world_size, hidden)`` that takes the 128-column pair tile
    (:data:`BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD`); ``0`` = no cell."""

    cells = BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD.get((arch, int(world_size)), {})
    return int(cells.get(int(hidden), 0))


def default_bn128_max_m(arch: str, world_size: int, hidden: int) -> int:
    """The 128-column tile bound of ``(arch, world_size, hidden)``: :data:`HOPPER_BN128_MAX_M_BY_ARCH_WORLD` off the
    pair architectures, :data:`BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD` on them; ``0`` = no cell."""

    if arch in PAIR_ARCHES:
        return blackwell_bn128_max_m(arch, world_size, hidden)
    return hopper_bn128_max_m(arch, world_size, hidden)


# Destination interleave of the tile table.  The shipped table runs destination blocks (remote destinations in
# sender-cyclic order, the local band last).  On the rows of ``DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD`` (per
# ``(arch, world_size)``: ``hidden -> (min M, max M)``, both inclusive) a mode of ``DEST_INTERLEAVE_MODES`` reorders
# the runs: ``il`` = token-tile group outer, column tile, then every destination (the grouped activation reuse is
# kept); ``lt`` = ``il`` with the trailing local runs of about one persistent wave (``DEST_INTERLEAVE_WAVE_SMS`` minus
# the row's reserved SMs) moved to the end; ``ls`` = the remote blocks with the local runs spread evenly into them.
# Every row keeps its destination, readiness word and ticket; the stored bits do not change.
# ``DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD`` lists the cells that ship a mode (every other cell: ``off``).  A shipped
# mode comes with the per-destination launch precedence (``dest_interleave_pd``), except on the cells of
# ``DEST_INTERLEAVE_DEFAULT_WITHOUT_PD`` (SM90 group of 4: ``il`` alone): on its band rows the interleaved
# table is one persistent launch, and a fired ``per_dest_large_m`` gate yields to it (recorded in
# ``ShapeGateVerdict.dest_interleave_suppressed``).  ``DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD`` caps the band of
# a shipped mode (rows above the cap keep the destination blocks and their own gates).  Without the precedence a band
# row whose per-destination launch gate fired keeps the destination blocks.
DEST_INTERLEAVE_MODES: tuple[str, ...] = ("off", "il", "lt", "ls")
DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD: dict[
    tuple[str, int], dict[int, tuple[int, int]]
] = {
    ("sm_100a", 8): {3072: (8192, 16384)},
    ("sm_100a", 4): {3072: (8192, 32768)},
    ("sm_100a", 2): {3072: (8192, 65536)},
    ("sm_103a", 8): {3072: (8192, 16384)},
    ("sm_103a", 4): {3072: (8192, 32768)},
    ("sm_103a", 2): {3072: (8192, 65536)},
    ("sm_90a", 4): {3072: (4096, 16384), 4096: (8192, 16384)},
    ("sm_90a", 2): {3072: (8192, 16384), 4096: (8192, 16384)},
}
DEST_INTERLEAVE_ARCHES: tuple[str, ...] = tuple(
    sorted({arch for arch, _ in DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD})
)
DEST_INTERLEAVE_WAVE_SMS: dict[str, int] = {
    "sm_100a": 148,
    "sm_103a": 148,
    "sm_90a": 132,
}
DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD: dict[tuple[str, int], str] = {
    ("sm_100a", 8): "lt",
    ("sm_100a", 4): "lt",
    ("sm_103a", 8): "lt",
    ("sm_103a", 4): "lt",
    ("sm_103a", 2): "lt",
    ("sm_100a", 2): "lt",
    ("sm_90a", 4): "il",
    ("sm_90a", 2): "il",
}
# Cells whose shipped mode leaves the per-destination launch precedence off (the
# measured SM90 arm is the interleave alone; no SM90 band row fires the split).
DEST_INTERLEAVE_DEFAULT_WITHOUT_PD: frozenset[tuple[str, int]] = frozenset(
    {("sm_90a", 4), ("sm_90a", 2)}
)
DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD: dict[tuple[str, int], int] = {
    ("sm_100a", 2): 16384,
    ("sm_103a", 2): 16384,
}


def _check_dest_interleave_tables() -> None:
    for (arch, world_size), cells in DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD.items():
        if (
            arch not in SHAPE_GATE_DEFAULT_BY_ARCH
            or arch not in DEST_INTERLEAVE_WAVE_SMS
            or world_size not in SUPPORTED_WORLD_SIZES
        ):
            raise ValueError(
                f"DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD[{(arch, world_size)!r}] is not a delivered cell"
            )
        for hidden, band in cells.items():
            if (
                isinstance(hidden, bool)
                or not isinstance(hidden, int)
                or hidden <= 0
                or hidden % HEAD_DIM
                or len(band) != 2
                or not 0 < int(band[0]) <= int(band[1]) < SHAPE_GATE_ALL_ROWS
            ):
                raise ValueError(
                    f"DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD[{(arch, world_size)!r}][{hidden!r}] = {band!r}: hidden "
                    f"must be a positive multiple of {HEAD_DIM} and the band (min M, max M) with 0 < min <= max"
                )
    for (arch, world_size), mode in DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD.items():
        if (
            (arch, world_size) not in DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD
            or mode not in DEST_INTERLEAVE_MODES[1:]
        ):
            raise ValueError(
                f"DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}] = {mode!r}: a cell of "
                f"DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD with a mode of {DEST_INTERLEAVE_MODES[1:]}"
            )
    for (arch, world_size), cap in DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD.items():
        if (
            arch,
            world_size,
        ) not in DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD or not check_bn128_max_m(cap):
            raise ValueError(
                f"DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD[{(arch, world_size)!r}] = {cap!r}: a positive row "
                "count on a cell of DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD"
            )
    for cell in DEST_INTERLEAVE_DEFAULT_WITHOUT_PD:
        if cell not in DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD:
            raise ValueError(
                f"DEST_INTERLEAVE_DEFAULT_WITHOUT_PD cell {cell!r} is not a cell of "
                "DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD"
            )


_check_dest_interleave_tables()


def dest_interleave_band(
    arch: str, world_size: int, hidden: int
) -> Optional[tuple[int, int]]:
    """The ``(min M, max M)`` band of ``(arch, world_size, hidden)`` (:data:`DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD`);
    ``None`` = no cell."""

    band = DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD.get((arch, int(world_size)), {}).get(
        int(hidden)
    )
    return None if band is None else (int(band[0]), int(band[1]))


def resolve_dest_interleave(
    arch: str,
    world_size: int,
    mode: Optional[str] = None,
    *,
    pd: Optional[bool] = None,
    max_m: Optional[int] = None,
) -> tuple[str, bool, Optional[int]]:
    """``(mode, pd, max_m)`` of a session: ``mode`` ``None`` = the ``(arch, world_size)`` default
    (:data:`DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD`, with the per-destination launch precedence -- not on the cells of
    :data:`DEST_INTERLEAVE_DEFAULT_WITHOUT_PD` -- and the cap of
    :data:`DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD` unless ``pd`` / ``max_m`` say otherwise; ``off`` where the cell
    ships none); an explicit mode takes ``pd`` (default ``False``) and ``max_m`` (default no cap) as given.  A mode
    other than ``off`` is refused on an architecture without a band cell."""

    default_shape_gates(arch, world_size=world_size)
    if mode is None:
        mode = DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD.get((arch, world_size), "off")
        if mode != "off":
            pd = (
                (arch, world_size) not in DEST_INTERLEAVE_DEFAULT_WITHOUT_PD
                if pd is None
                else pd
            )
            max_m = (
                DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD.get((arch, world_size))
                if max_m is None
                else max_m
            )
    if mode not in DEST_INTERLEAVE_MODES:
        raise ValueError(
            f"dest_interleave must be one of {DEST_INTERLEAVE_MODES}, got {mode!r}"
        )
    pd = bool(pd) if pd is not None else False
    if pd and mode == "off":
        raise ValueError("dest_interleave_pd needs dest_interleave il, lt or ls")
    if max_m is not None:
        if isinstance(max_m, bool) or not isinstance(max_m, int) or max_m <= 0:
            raise ValueError(
                f"dest_interleave_max_m must be a positive int, got {max_m!r}"
            )
        if mode == "off":
            raise ValueError("dest_interleave_max_m needs dest_interleave il, lt or ls")
    if mode != "off" and arch not in DEST_INTERLEAVE_ARCHES:
        raise ValueError(
            f"dest_interleave={mode} is refused on {arch}: no DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD cell "
            f"(architectures: {DEST_INTERLEAVE_ARCHES})"
        )
    return mode, pd, max_m


def dest_interleave_row(
    mode: str,
    max_m: Optional[int],
    arch: str,
    world_size: int,
    hidden: int,
    rows: int,
) -> bool:
    """Whether a row of ``rows`` rows and ``hidden`` columns is a band row of ``mode``: the mode on, ``rows`` inside
    the cell's :data:`DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD` band and at most ``max_m`` (``None`` = no cap)."""

    if mode == "off":
        return False
    band = dest_interleave_band(arch, world_size, hidden)
    if band is None:
        return False
    high = band[1] if max_m is None else min(band[1], int(max_m))
    return band[0] <= int(rows) <= high


def interleave_tail_rows(arch: str, *, pair: bool, reserved_sms: int) -> int:
    """The ``lt`` local-tail row budget: one persistent wave of :data:`DEST_INTERLEAVE_WAVE_SMS` minus the row's
    reserved SMs, rounded down to the lockstep (at least one lockstep unit)."""

    lockstep = PAIR_LOCKSTEP if pair else 1
    return max(
        lockstep,
        (DEST_INTERLEAVE_WAVE_SMS[arch] - int(reserved_sms)) // lockstep * lockstep,
    )


@dataclass(frozen=True)
class ShapeGateVerdict:
    """The row-count gates of one call (``rows = batch * s_local``): which fired and what they select.
    ``words_capacity_per_dest`` is ``1`` when the small-row gate fired, else ``None`` (the session's capacity
    stands); ``reserved_sms`` is ``0`` when the large-row reserved-SM gate fired, else ``None`` (``RESERVED_SMS``
    stands); ``producer_launches`` is ``"per_dest"`` or ``"persistent"``; ``pair`` is ``False`` when the pair-tile
    gate fired (the single-CTA producer), else ``None`` (the policy's tile stands); ``bn`` is the tile width the
    verdict was read against (``None`` when the caller gave none).  With a K table (``k_table``, the per-destination
    launch threshold by hidden size) the verdict records the call's ``hidden`` and the threshold it read
    (``per_dest_threshold``); with the cross-destination-band gate on (``xband_min_m``), ``xband`` is ``True`` when it
    fired (the cross-destination-band plan, one persistent launch), ``False`` otherwise, and ``xband_suppressed``
    names a per-destination launch gate that fired but yields to it; ``xband`` is ``None`` while the gate is off."""

    rows: int
    gates: dict[str, int]
    words_capacity_per_dest: Optional[int]
    producer_launches: str
    reserved_sms: Optional[int] = None
    pair: Optional[bool] = None
    bn: Optional[int] = None
    hidden: Optional[int] = None
    k_table: Optional[tuple[tuple[int, int], ...]] = None
    per_dest_threshold: Optional[int] = None
    xband_min_m: int = 0
    xband: Optional[bool] = None
    xband_eligible: Optional[bool] = None
    xband_suppressed: tuple[str, ...] = ()
    # A per-destination launch gate that fired on a destination-interleave band row but yields to the interleaved
    # table (``dest_interleave_pd``: one persistent launch); empty on every other row.
    dest_interleave_suppressed: tuple[str, ...] = ()

    @property
    def fired(self) -> tuple[str, ...]:
        """The gates that fired, in ``SHAPE_GATE_NAMES`` order, then ``xband_min_m``."""

        fired = []
        if self.words_capacity_per_dest is not None:
            fired.append("words_capacity_small_m")
        if self.reserved_sms is not None:
            fired.append("reserved_sms_large_m")
        if self.producer_launches == "per_dest":
            fired.append("per_dest_large_m")
        if self.pair is not None:
            fired.append("pair_off_large_m")
        if self.xband:
            fired.append("xband_min_m")
        return tuple(fired)

    def summary(self) -> dict[str, Any]:
        record = {
            "rows": self.rows,
            "bn": self.bn,
            "gates": spell_shape_gates(self.gates),
            "fired": list(self.fired),
            "words_capacity_per_dest": self.words_capacity_per_dest,
            "reserved_sms": self.reserved_sms,
            "producer_launches": self.producer_launches,
            "pair": self.pair,
        }
        if self.k_table is not None:
            record["hidden"] = self.hidden
            record["per_dest_k_table"] = [list(band) for band in self.k_table]
            record["per_dest_threshold"] = self.per_dest_threshold
        if self.xband_min_m:
            record["xband_min_m"] = self.xband_min_m
            record["xband"] = self.xband
            record["xband_eligible"] = self.xband_eligible
            record["xband_suppressed"] = list(self.xband_suppressed)
        if self.dest_interleave_suppressed:
            record["dest_interleave_suppressed"] = list(self.dest_interleave_suppressed)
        return record


def gate_rows(
    gates: Mapping[str, Any],
    rows: int,
    *,
    bn: Optional[int] = None,
    hidden: Optional[int] = None,
    k_table: Optional[Sequence[tuple[int, int]]] = None,
    xband_min_m: int = 0,
    xband_eligible: Optional[bool] = None,
    dest_interleave_row: bool = False,
    dest_interleave_pd: bool = False,
) -> ShapeGateVerdict:
    """Resolve the row-count gates on one call's ``rows`` (inclusive thresholds; a gate at ``0`` never fires).
    ``bn`` is the call's tile width (:func:`block_n_for`), required while a gate of :data:`SHAPE_GATE_BN` is on:
    the pair-tile gate fires on ``bn == PAIR_BN`` calls only.  ``k_table`` (an entry of :data:`SHAPE_GATE_K_TABLE`)
    replaces the ``per_dest_large_m`` threshold by the threshold of the band of the call's ``hidden`` (required with
    it).  ``xband_min_m`` (``0`` = off) is the cross-destination-band gate: it fires on a call with ``rows >=
    xband_min_m`` whose geometry is :func:`xband_eligible` (``xband_eligible``, required unless the verdict cannot
    depend on it: a ``bn == XBAND_BN`` call is never eligible, a call below the threshold never fires) and then
    suppresses a fired per-destination launch gate (the plan is one persistent launch).  ``dest_interleave_row`` (the
    call is a band row of the session's destination interleave, :func:`dest_interleave_row`) with
    ``dest_interleave_pd`` suppresses a fired per-destination launch gate the same way (the interleaved table is one
    persistent launch)."""

    checked = check_shape_gates(gates)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"rows must be a positive int, got {rows!r}")
    if bn is not None and (
        isinstance(bn, bool) or not isinstance(bn, int) or bn not in BLOCK_N_CHOICES
    ):
        raise ValueError(f"bn must be one of {BLOCK_N_CHOICES}, got {bn!r}")
    if hidden is not None and (
        isinstance(hidden, bool) or not isinstance(hidden, int) or hidden < 1
    ):
        raise ValueError(f"hidden must be a positive int, got {hidden!r}")
    small = checked["words_capacity_small_m"]
    reserved = checked["reserved_sms_large_m"]
    large = checked["per_dest_large_m"]
    pair_off = checked["pair_off_large_m"]
    if pair_off and bn is None:
        raise ValueError(
            f"pair_off_large_m={pair_off} fires on bn == {SHAPE_GATE_BN['pair_off_large_m']} "
            "calls only; pass the call's tile width (bn=block_n_for(geometry))"
        )
    pair_fires = (
        bool(pair_off) and bn == SHAPE_GATE_BN["pair_off_large_m"] and rows >= pair_off
    )
    bands = None
    if k_table is not None:
        bands = check_shape_gate_k_table(k_table)
        if hidden is None:
            raise ValueError(
                "the per-destination launch gate reads its threshold from the K table by the call's hidden size; "
                "pass hidden=geometry.hidden"
            )
        large = k_band_threshold(bands, hidden)
    per_dest = bool(large) and rows >= large
    if (
        isinstance(xband_min_m, bool)
        or not isinstance(xband_min_m, int)
        or xband_min_m < 0
    ):
        raise ValueError(
            f"xband_min_m must be an int >= 0 (0 = off), got {xband_min_m!r}"
        )
    xband: Optional[bool] = None
    suppressed: tuple[str, ...] = ()
    if xband_min_m:
        if xband_eligible is None and bn == XBAND_BN:
            xband_eligible = False
        if xband_eligible is None and rows < xband_min_m:
            xband = False
        elif not isinstance(xband_eligible, bool):
            raise ValueError(
                f"xband_min_m={xband_min_m} fires on xband_eligible calls only; pass the call's verdict "
                "(xband_eligible=xband_eligible(geometry))"
            )
        else:
            xband = xband_eligible and rows >= xband_min_m
        if xband and per_dest:
            per_dest = False
            suppressed = ("per_dest_large_m",)
    dsti_suppressed: tuple[str, ...] = ()
    if dest_interleave_row and dest_interleave_pd and per_dest:
        per_dest = False
        dsti_suppressed = ("per_dest_large_m",)
    return ShapeGateVerdict(
        rows=int(rows),
        gates=checked,
        words_capacity_per_dest=1 if small and rows <= small else None,
        producer_launches="per_dest" if per_dest else "persistent",
        reserved_sms=0 if reserved and rows >= reserved else None,
        pair=False if pair_fires else None,
        bn=bn,
        hidden=hidden,
        k_table=bands,
        per_dest_threshold=int(large) if bands is not None else None,
        xband_min_m=int(xband_min_m),
        xband=xband,
        xband_eligible=xband_eligible if xband_min_m else None,
        xband_suppressed=suppressed,
        dest_interleave_suppressed=dsti_suppressed,
    )


def plan_row(
    geometry: Geometry,
    layout: str,
    rank: int,
    *,
    arch: str,
    gates: Mapping[str, Any],
    k_table: Optional[Sequence[tuple[int, int]]] = None,
    xband_min_m: int = 0,
    pair_available: bool = True,
    words_capacity_per_dest: int = DEFAULT_WORDS_CAPACITY_PER_DEST,
    group_m_scale: Optional[int] = None,
    bn128_max_m: Optional[int] = None,
    dest_interleave: Optional[str] = None,
    dest_interleave_pd: Optional[bool] = None,
    dest_interleave_max_m: Optional[int] = None,
) -> tuple[ShapeGateVerdict, Policy, TilePlan]:
    """The host plan of one call: the row-count gate verdict (:func:`gate_rows` on the call's rows, tile width,
    hidden size and cross-destination-band eligibility), then the policy and tile plan of :func:`plan_for` with the
    verdict applied -- one readiness word per destination for a fired small-row gate (otherwise
    ``words_capacity_per_dest``), the single-CTA producer for a fired pair-tile gate, the cross-destination-band plan
    for a fired ``xband_min_m`` gate, the 128-column tile on a row of the 128-column tile table (``bn128_max_m``, see
    :func:`plan_for`; the gates read the band width), the destination interleave of :func:`resolve_dest_interleave`
    (``dest_interleave`` ``None`` = the ``(arch, world_size)`` default) on its band rows (:func:`dest_interleave_row`)
    unless a per-destination launch gate fired there without the precedence (the row keeps the destination blocks).
    ``verdict.producer_launches`` / ``verdict.reserved_sms`` are the launch structure of the call."""

    bn = block_n_for(geometry)
    dsti, dsti_pd, dsti_max_m = resolve_dest_interleave(
        arch,
        geometry.world_size,
        dest_interleave,
        pd=dest_interleave_pd,
        max_m=dest_interleave_max_m,
    )
    dsti_row = dest_interleave_row(
        dsti,
        dsti_max_m,
        arch,
        geometry.world_size,
        int(geometry.hidden),
        int(geometry.rows),
    )
    verdict = gate_rows(
        gates,
        int(geometry.rows),
        bn=bn,
        hidden=int(geometry.hidden),
        k_table=k_table,
        xband_min_m=xband_min_m,
        xband_eligible=xband_eligible(geometry) if xband_min_m else None,
        dest_interleave_row=dsti_row,
        dest_interleave_pd=dsti_pd,
    )
    row_dsti = dsti if dsti_row and verdict.producer_launches != "per_dest" else "off"
    words = (
        verdict.words_capacity_per_dest
        if verdict.words_capacity_per_dest is not None
        else words_capacity_per_dest
    )
    policy, plan = plan_for(
        geometry,
        layout,
        rank,
        arch=arch,
        pair_available=pair_available,
        words_capacity_per_dest=words,
        group_m_scale=group_m_scale,
        pair=verdict.pair,
        xband=bool(verdict.xband),
        bn128_max_m=bn128_max_m,
        dest_interleave=row_dsti,
        reserved_sms=RESERVED_SMS
        if verdict.reserved_sms is None
        else int(verdict.reserved_sms),
    )
    return verdict, policy, plan


def dest_parts(plan: TilePlan) -> tuple[tuple[int, int], ...]:
    """Contiguous ``(start, stop)`` runs of the plan's table that share one destination, in table order; lockstep pad
    rows (``dest = -1``) stay with the run they pad.  Covers the table exactly once -- the launches of a
    ``per_dest`` call.  A cross-destination-band plan has no per-destination runs (its tiles span two destinations;
    it runs as one persistent launch) and is refused."""

    if plan.xband:
        raise ValueError(
            "dest_parts: a cross-destination-band plan has no per-destination runs; it runs as one persistent launch"
        )
    if plan.dest_interleave != "off":
        raise ValueError(
            f"dest_parts: a dest_interleave={plan.dest_interleave!r} plan has no per-destination runs; it runs as one "
            "persistent launch"
        )
    dests = [int(value) for value in plan.tiles[:, TILE_FIELDS.index("dest")].tolist()]
    if not dests:
        raise ValueError("dest_parts: empty tile table")
    parts: list[tuple[int, int]] = []
    start, current = 0, dests[0]
    for index, dest in enumerate(dests):
        if dest == -1 or dest == current:
            continue
        parts.append((start, index))
        start, current = index, dest
    parts.append((start, len(dests)))
    if sum(stop - start for start, stop in parts) != len(dests) or parts[0][0] != 0:
        raise AssertionError(
            f"dest_parts {parts} do not cover the {len(dests)}-row table"
        )
    return tuple(parts)


__all__ = [
    "BLACKWELL_BN128_BN",
    "BLACKWELL_BN128_MAX_M_BY_ARCH_WORLD",
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N_CHOICES",
    "DEFAULT_WORDS_CAPACITY_PER_DEST",
    "DEST_INTERLEAVE_ARCHES",
    "DEST_INTERLEAVE_DEFAULT_BY_ARCH_WORLD",
    "DEST_INTERLEAVE_DEFAULT_WITHOUT_PD",
    "DEST_INTERLEAVE_MAX_M_DEFAULT_BY_ARCH_WORLD",
    "DEST_INTERLEAVE_MIN_M_BY_ARCH_WORLD",
    "DEST_INTERLEAVE_MODES",
    "DEST_INTERLEAVE_WAVE_SMS",
    "GEMM_GROUP_NORM_KIND",
    "GROUP_M_SCALES",
    "GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD",
    "GROUP_M_SCALE_MIN_K",
    "HEAD_DIM",
    "HOPPER_BN128_BN",
    "HOPPER_BN128_MAX_M_BY_ARCH_WORLD",
    "LAYOUTS",
    "PAIR_ARCHES",
    "PAIR_BN",
    "PAIR_LOCKSTEP",
    "PAIR_MIN_GROUPS",
    "PAIR_MIN_M",
    "PRODUCER_SHIPPED_ROWS_BY_ARCH_WORLD",
    "RESERVED_SMS",
    "SHAPE_GATE_ALL_ROWS",
    "SHAPE_GATE_ALL_ROWS_NAMES",
    "SHAPE_GATE_BN",
    "SHAPE_GATE_DEFAULT_BY_ARCH",
    "SHAPE_GATE_DEFAULT_BY_ARCH_WORLD",
    "SHAPE_GATE_K_TABLE",
    "SHAPE_GATE_NAMES",
    "SHAPE_GATE_OFF",
    "SUPPORTED_WORLD_SIZES",
    "TILE_FIELDS",
    "TILE_FIELDS_XB",
    "TILE_ORDERS",
    "XBAND_BN",
    "XBAND_MIN_M_DEFAULT_BY_ARCH_WORLD",
    "XBAND_MIN_M_FLOOR",
    "Geometry",
    "Policy",
    "ShapeGateVerdict",
    "TilePlan",
    "blackwell_bn128_max_m",
    "block_n_for",
    "ceil_div",
    "check_bn128_max_m",
    "check_shape_gate_k_table",
    "check_shape_gates",
    "check_xband_min_m",
    "chunk_plan",
    "default_bn128_max_m",
    "default_group_m_scale",
    "default_shape_gate_k_table",
    "default_shape_gates",
    "default_shipped_producer_row",
    "default_xband_min_m",
    "dest_interleave_band",
    "dest_interleave_row",
    "dest_parts",
    "gate_rows",
    "heads_divisibility_error",
    "hopper_bn128_max_m",
    "interleave_tail_rows",
    "k_band_index",
    "k_band_threshold",
    "pair_geometry",
    "plan_for",
    "plan_policy",
    "plan_row",
    "producer_grid",
    "resolve_dest_interleave",
    "shape_gate_spelling",
    "spell_shape_gates",
    "words_capacity_for_rows",
    "xband_eligible",
]
