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
from typing import Any, Sequence

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
PAIR_ARCHES: tuple[str, ...] = ("sm_100a", "sm_103a")
LAYOUTS: tuple[str, ...] = ("bshd", "fused")


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

    def summary(self) -> dict[str, Any]:
        return {
            "bn": self.bn,
            "order": self.order,
            "group_m": self.group_m,
            "pair": self.pair,
            "a_bytes": self.a_bytes,
            "m_tiles": self.m_tiles,
            "reasons": list(self.reasons),
        }


def pair_geometry(geometry: Geometry) -> bool:
    """``bn == PAIR_BN or M >= PAIR_MIN_M`` with ``bn`` from :func:`block_n_for`."""

    return block_n_for(geometry) == PAIR_BN or geometry.rows >= PAIR_MIN_M


def plan_policy(
    geometry: Geometry, *, arch: str, pair_available: bool = True
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
    """

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
    return Policy(
        bn=bn,
        order=order,
        group_m=group_m,
        pair=pair,
        a_bytes=a_bytes,
        m_tiles=m_tiles,
        reasons=tuple(reasons),
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
) -> tuple[Policy, TilePlan]:
    """Policy and tile plan of one row for one rank."""

    policy = plan_policy(geometry, arch=arch, pair_available=pair_available)
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


__all__ = [
    "BLOCK_K",
    "BLOCK_M",
    "BLOCK_N_CHOICES",
    "DEFAULT_WORDS_CAPACITY_PER_DEST",
    "GEMM_GROUP_NORM_KIND",
    "HEAD_DIM",
    "LAYOUTS",
    "PAIR_ARCHES",
    "PAIR_BN",
    "PAIR_LOCKSTEP",
    "PAIR_MIN_GROUPS",
    "PAIR_MIN_M",
    "TILE_FIELDS",
    "TILE_ORDERS",
    "Geometry",
    "Policy",
    "TilePlan",
    "block_n_for",
    "ceil_div",
    "chunk_plan",
    "heads_divisibility_error",
    "pair_geometry",
    "plan_for",
    "plan_policy",
    "producer_grid",
    "words_capacity_for_rows",
]
