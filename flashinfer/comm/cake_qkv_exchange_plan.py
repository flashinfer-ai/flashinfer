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
# producer; pair rows with ``M >= PAIR_MIN_M`` are grouped with at least
# ``PAIR_MIN_GROUPS`` token-tile groups per rank.
PAIR_BN = 256
PAIR_MIN_M = 4096
PAIR_MIN_GROUPS = 4
PAIR_ARCHES: tuple[str, ...] = ("sm_100a", "sm_103a", "sm_107a")
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
    # SM107 keeps the baseline plan on every group size (the scale was not measured there); the 8-rank entry
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
# instead of one persistent grid.  ``0`` = the gate never fires.  Every gate is
# inclusive at its threshold and changes neither the tile table nor the stored
# bits.  Measured per architecture: SM100 / SM103 ship the small-row word gate
# and the per-destination launch gate, SM90 ships the large-row reserved-SM gate,
# SM107 ships none;
# ``FLASHINFER_QKV_EXCHANGE_SHAPE_GATES`` (see the Cake backend) overrides a
# session's gates in either direction.
SHAPE_GATE_NAMES: tuple[str, ...] = (
    "words_capacity_small_m",
    "reserved_sms_large_m",
    "per_dest_large_m",
)
SHAPE_GATE_OFF = 0
SHAPE_GATE_DEFAULT_BY_ARCH: dict[str, dict[str, int]] = {
    "sm_90a": {
        "words_capacity_small_m": SHAPE_GATE_OFF,
        "reserved_sms_large_m": 16384,
        "per_dest_large_m": SHAPE_GATE_OFF,
    },
    "sm_100a": {
        "words_capacity_small_m": 4096,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": 16384,
    },
    "sm_103a": {
        "words_capacity_small_m": 4096,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": 16384,
    },
    # SM107: no gate ships (none was measured there); every call takes the ungated host plan.
    "sm_107a": {
        "words_capacity_small_m": SHAPE_GATE_OFF,
        "reserved_sms_large_m": SHAPE_GATE_OFF,
        "per_dest_large_m": SHAPE_GATE_OFF,
    },
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
    shape_pair = pair_geometry(geometry)
    pair = shape_pair and arch in PAIR_ARCHES and pair_available
    if shape_pair:
        why = " or ".join(
            ([f"bn {bn} == {PAIR_BN}"] if bn == PAIR_BN else [])
            + ([f"M {M} >= {PAIR_MIN_M}"] if M >= PAIR_MIN_M else [])
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
    if pair and (M >= PAIR_MIN_M or a_bytes > NT_OUTER_A_BYTES_MAX):
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
) -> tuple[list[list[int]], int]:
    """Tile rows in production order; returns ``(rows, pad_rows)``."""

    G = m_tiles_per_word
    rows: list[list[int]] = []
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
        for g0 in range(0, m_tiles, group_m):
            mts = range(g0, min(m_tiles, g0 + group_m))
            for nt in range(3 * n_sub):
                run = [tile(int(dest), mt, nt) for mt in mts]
                while lockstep > 1 and len(run) % lockstep:
                    pad = list(run[-1])
                    pad[TILE_DEST] = -1
                    pad[TILE_WORD] = 0
                    pad[TILE_LAST_TICKET] = 0
                    run.append(pad)
                    pads += 1
                rows.extend(run)
    return rows, pads


@dataclass(frozen=True)
class TilePlan:
    """One rank's tile table (CPU ``int64 [num_tiles, 8]``) and the scalars bound next to it."""

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

    @property
    def lockstep(self) -> int:
        return PAIR_LOCKSTEP if self.pair else 1

    @property
    def num_real_tiles(self) -> int:
        return self.num_tiles - self.num_pad_rows

    def summary(self) -> dict[str, Any]:
        return {
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
) -> TilePlan:
    """The exchange plan of source ``rank`` as a tile table.

    Destination order: sender-cyclic remote first ``(r + 1, ..., r + W - 1) mod
    W``, local ``r`` last.  Readiness words: ``G = ceil(m_tiles /
    words_capacity_per_dest)`` token tiles per word, ``word = mt // G``.
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
    if c_band % block_n:
        raise ValueError(
            f"block_n={block_n} must divide the destination band of {c_band} columns"
        )
    batch, s_local, s_global = geometry.batch, geometry.s_local, geometry.s_global
    M, K, N = geometry.rows, geometry.hidden, 3 * heads * head_dim
    m_tiles = ceil_div(M, BLOCK_M)
    n_sub = c_band // block_n
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
    )
    tiles = torch.tensor(rows, dtype=torch.int64).contiguous()
    num_tiles = len(rows)
    if num_tiles - num_pad_rows != world * m_tiles * 3 * n_sub or tuple(
        tiles.shape
    ) != (
        num_tiles,
        len(TILE_FIELDS),
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
) -> tuple[Policy, TilePlan]:
    """Policy and tile plan of one row for one rank (``group_m_scale`` ``None`` = the architecture's default for
    the geometry's world size)."""

    policy = plan_policy(
        geometry, arch=arch, pair_available=pair_available, group_m_scale=group_m_scale
    )
    plan = chunk_plan(
        geometry,
        layout,
        rank,
        block_n=policy.bn,
        words_capacity_per_dest=words_capacity_per_dest,
        order=policy.order,
        group_m=policy.group_m if policy.order == "grouped" else None,
        pair=policy.pair,
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


def default_shape_gates(arch: str) -> dict[str, int]:
    """The shipped row-count gates of ``arch`` (:data:`SHAPE_GATE_DEFAULT_BY_ARCH`; a copy); an unknown architecture
    is an error."""

    try:
        return dict(SHAPE_GATE_DEFAULT_BY_ARCH[arch])
    except KeyError:
        raise ValueError(
            f"no shape gates for architecture {arch!r}; known: {tuple(SHAPE_GATE_DEFAULT_BY_ARCH)}"
        ) from None


def check_shape_gates(gates: Mapping[str, Any]) -> dict[str, int]:
    """Validate a gate table: exactly :data:`SHAPE_GATE_NAMES`, each an int ``>= 0`` (``0`` = off)."""

    if set(gates) != set(SHAPE_GATE_NAMES):
        raise ValueError(
            f"shape gates must name exactly {SHAPE_GATE_NAMES}, got {tuple(gates)}"
        )
    checked: dict[str, int] = {}
    for name in SHAPE_GATE_NAMES:
        value = gates[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(
                f"shape gate {name} must be an int >= 0 (0 = off), got {value!r}"
            )
        checked[name] = int(value)
    return checked


@dataclass(frozen=True)
class ShapeGateVerdict:
    """The row-count gates of one call (``rows = batch * s_local``): which fired and what they select.
    ``words_capacity_per_dest`` is ``1`` when the small-row gate fired, else ``None`` (the session's capacity
    stands); ``reserved_sms`` is ``0`` when the large-row reserved-SM gate fired, else ``None`` (``RESERVED_SMS``
    stands); ``producer_launches`` is ``"per_dest"`` or ``"persistent"``."""

    rows: int
    gates: dict[str, int]
    words_capacity_per_dest: Optional[int]
    producer_launches: str
    reserved_sms: Optional[int] = None

    @property
    def fired(self) -> tuple[str, ...]:
        """The gates that fired, in ``SHAPE_GATE_NAMES`` order."""

        fired = []
        if self.words_capacity_per_dest is not None:
            fired.append("words_capacity_small_m")
        if self.reserved_sms is not None:
            fired.append("reserved_sms_large_m")
        if self.producer_launches == "per_dest":
            fired.append("per_dest_large_m")
        return tuple(fired)

    def summary(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "gates": dict(self.gates),
            "fired": list(self.fired),
            "words_capacity_per_dest": self.words_capacity_per_dest,
            "reserved_sms": self.reserved_sms,
            "producer_launches": self.producer_launches,
        }


def gate_rows(gates: Mapping[str, Any], rows: int) -> ShapeGateVerdict:
    """Resolve the row-count gates on one call's ``rows`` (inclusive thresholds; a gate at ``0`` never fires)."""

    checked = check_shape_gates(gates)
    if isinstance(rows, bool) or not isinstance(rows, int) or rows < 1:
        raise ValueError(f"rows must be a positive int, got {rows!r}")
    small = checked["words_capacity_small_m"]
    reserved = checked["reserved_sms_large_m"]
    large = checked["per_dest_large_m"]
    return ShapeGateVerdict(
        rows=int(rows),
        gates=checked,
        words_capacity_per_dest=1 if small and rows <= small else None,
        producer_launches="per_dest" if large and rows >= large else "persistent",
        reserved_sms=0 if reserved and rows >= reserved else None,
    )


def dest_parts(plan: TilePlan) -> tuple[tuple[int, int], ...]:
    """Contiguous ``(start, stop)`` runs of the plan's table that share one destination, in table order; lockstep pad
    rows (``dest = -1``) stay with the run they pad.  Covers the table exactly once -- the launches of a
    ``per_dest`` call."""

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
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N_CHOICES",
    "DEFAULT_WORDS_CAPACITY_PER_DEST",
    "GEMM_GROUP_NORM_KIND",
    "GROUP_M_SCALES",
    "GROUP_M_SCALE_DEFAULT_BY_ARCH_WORLD",
    "GROUP_M_SCALE_MIN_K",
    "HEAD_DIM",
    "LAYOUTS",
    "PAIR_ARCHES",
    "PAIR_BN",
    "PAIR_LOCKSTEP",
    "PAIR_MIN_GROUPS",
    "PAIR_MIN_M",
    "RESERVED_SMS",
    "SHAPE_GATE_DEFAULT_BY_ARCH",
    "SHAPE_GATE_NAMES",
    "SHAPE_GATE_OFF",
    "SUPPORTED_WORLD_SIZES",
    "TILE_FIELDS",
    "TILE_ORDERS",
    "Geometry",
    "Policy",
    "ShapeGateVerdict",
    "TilePlan",
    "block_n_for",
    "ceil_div",
    "check_shape_gates",
    "chunk_plan",
    "default_group_m_scale",
    "default_shape_gates",
    "dest_parts",
    "gate_rows",
    "heads_divisibility_error",
    "pair_geometry",
    "plan_for",
    "plan_policy",
    "producer_grid",
    "words_capacity_for_rows",
]
