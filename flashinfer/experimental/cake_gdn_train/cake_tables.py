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

CPU int32 table builders of the GDN KCP route (the kernel ABI table set): the
chunk tables the local programs read (``chunk_bos`` / ``chunk_len`` /
``seq_chunk_start``), the segment tables and compact state pools the walks read
(``seg_chunk_start``, ``walk_seg``, ``h0_index`` / ``out_index`` / ``dht_index``
with the two-integer pool addressing ``((h // hg) * n + slot) * hg + h % hg`` and
``tile_stride``), the merge chain tables (CSR over element offsets, one chain
per (document, owned head)), the head ownership table (``head_groups``, the
padded owner when ``HV % Hg != 0``) and the convolution halo row counts.

Everything is derived from the package-local :mod:`.cake_kcp_plan` context
(``build_kcp_context``) and its head plan ``ctx.heads(HV)``; the segment count
``P`` per fragment comes from the context's plan rule (``kcp_plan_row`` over every
rank's fragment table under the shipped ``"row_cost"`` rule, the per-rank
``kcp_plan`` under a rule selected by the test knobs) -- the state of the GDN route is per value
head, so every head count here is ``HV``.  Tables are built once per ``(HV, sm_count, P)`` key at
first use (``ctx.prepare(H_qk, HV)``; never at ``build_kcp_context``,
never per call) and cached on the context (``ctx._caches``) together with their
int32 / int64 device copies (int32 everywhere except the merge's int64 element
offsets ``INT64_TABLES``): no per-call host-to-device copy, no device read,
capture-safe.  Pure host
metadata; nothing here reads a tensor's contents.

Buffer forms (the kernels' ``tile_stride``): the c1 send buffer and the b1
buffer with re-sent maps are interleaved PAIRS of ``[128, 128]`` FP32 tiles per
(slot, head) -- ``[B_f | M_f^T]`` (halves ``TILE_B`` / ``TILE_M``) and
``[d_f | M_f^T]`` (``TILE_D`` / ``TILE_M``) -- the c2 receive buffer (``S_in``),
the b2 receive buffer (``dS_out``) and the offsets-only b1 buffer are dense
planes.  :class:`PoolAddressing` recovers the kernels' ``(hg, n, tile_stride,
half)`` form of every pool from the head plan and reproduces its explicit
tile tables exactly (checked by the package tests on every step-0 row).
"""

from __future__ import annotations

import dataclasses

import torch

from .cake_kcp_plan import (
    upload_tables,
    CHUNK,
    TILE,
    TILE_B,
    TILE_D,
    TILE_M,
    TILE_STRIDE_PAIR,
    TILE_STRIDE_PLANE,
    KcpContext,
    KcpHeadPlan,
    head_groups,
)

# The pools a walk launch addresses: name -> (split table of the head plan, slot count field of the context,
# tile stride, half of the pair; ``None`` = a dense plane).
POOLS = {
    "s_in": (
        "c2_recv",
        "n_in",
        TILE_STRIDE_PLANE,
        None,
    ),  # S_in = the c2 receive buffer (h0 of the tape walk)
    "c1_b": (
        "c1_send",
        "n_out",
        TILE_STRIDE_PAIR,
        TILE_B,
    ),  # B_f half of the c1 send pairs (out of the offset walk)
    "c1_m": (
        "c1_send",
        "n_out",
        TILE_STRIDE_PAIR,
        TILE_M,
    ),  # M_f^T half of the c1 send pairs (out of the transfer walk)
    "ds_out": (
        "b2_recv",
        "n_out",
        TILE_STRIDE_PLANE,
        None,
    ),  # dS_out = the b2 receive buffer (dht of the adjoint tape walk)
    "b1_d": (
        "b1_send_d",
        "n_in",
        TILE_STRIDE_PLANE,
        None,
    ),  # d_f plane of the offsets-only b1 send buffer
    "b1_d_dm": (
        "b1_send_dm",
        "n_in",
        TILE_STRIDE_PAIR,
        TILE_D,
    ),  # d_f half of the b1 send pairs (maps re-sent)
    "b1_m_dm": (
        "b1_send_dm",
        "n_in",
        TILE_STRIDE_PAIR,
        TILE_M,
    ),  # M_f^T half of the b1 send pairs
}
LOCAL_TABLES = ("chunk_bos", "chunk_len", "seq_chunk_start")
WALK_TABLES = (
    "seg_chunk_start",
    "walk_seg_offset",
    "walk_seg_tape",
    "h0_index",
    "out_index",
    "dht_index",
    "carry_index",
    "dcarry_index",
    "h0_index_fused",
    "dht_index_fused",
)
CHAIN_FAMILIES = ("fwd", "adj_b1", "adj_c1")


def _ceil_div(a: int, b: int) -> int:
    return -(-a // b)


# ----------------------------------------------------------------------------------------------------------------
# Ownership (owner table incl. the padded (HV, W) = (3, 2) case) and halo rows


def owner_table(HV: int, world_size: int) -> dict:
    """The head ownership of ``HV`` value heads over ``W`` ranks: ``Hg = ceil(HV / W)``, ``owner(h) = h // Hg``,
    ``Hg_r = clamp(HV - r Hg, 0, Hg)``; the single partial owner (``0 < Hg_r < Hg``, only when ``HV % Hg != 0``, e.g.
    ``(HV, W) = (3, 2)``: ``Hg = 2``, owners ``(2, 1)``) is padded to ``Hg`` tile slots per fragment slot so that the
    kernels' two-integer addressing is exact (``pad_tiles = n_owners Hg - HV`` per slot, never read)."""
    Hg, n_owners, hg_r = head_groups(HV, world_size)
    padded = [j for j, x in enumerate(hg_r) if 0 < x < Hg]
    return {
        "HV": int(HV),
        "world_size": int(world_size),
        "Hg": Hg,
        "n_owners": n_owners,
        "Hg_r": list(hg_r),
        "Hg_pad_r": [Hg if x > 0 else 0 for x in hg_r],
        "owner_of_head": [h // Hg for h in range(int(HV))],
        "head0": [r * Hg for r in range(int(world_size))],
        "full_groups": n_owners * Hg == int(HV),
        "padded_owner": padded[0] if padded else -1,
        "pad_tiles": n_owners * Hg - int(HV),
    }


def halo_rows(ctx: KcpContext) -> dict:
    """Row counts of the convolution halo plan of this rank (rows of ``C`` channels; BF16 forward history rows,
    FP32 backward ``dpre`` rows): the per-peer split sizes in rows and their sums."""
    h = ctx.halo
    return {
        "halo": h.halo,
        "n_cont": h.n_cont,
        "hist_valid": list(h.hist_valid),
        "n_hist_rows": h.n_hist_rows,
        "n_head_rows": h.n_head_rows,
        "n_tail_rows": h.n_tail_rows,
        "fwd_send_rows": list(h.fwd_send_rows),
        "fwd_recv_rows": list(h.fwd_recv_rows),
        "bwd_send_rows": list(h.bwd_send_rows),
        "bwd_recv_rows": list(h.bwd_recv_rows),
    }


# ----------------------------------------------------------------------------------------------------------------
# Pool addressing


@dataclasses.dataclass(frozen=True)
class PoolAddressing:
    """The kernels' view of one compact state pool: tile ``((h // hg) * n + slot) * hg + h % hg`` of (slot, head),
    ``tile_stride`` tiles apart (``1`` dense plane, ``2`` interleaved pair) and, for a pair, the ``half`` this
    launch reads or writes (the base pointer the host passes is the pool plus ``half * TILE`` elements)."""

    name: str
    hg: int
    n: int
    tile_stride: int
    half: int
    n_tiles: int  # tiles of the whole pool (``n_owners * hg * n``)

    def tile(self, slot: int, h: int) -> int:
        return ((int(h) // self.hg) * self.n + int(slot)) * self.hg + int(h) % self.hg

    def elem(self, slot: int, h: int) -> int:
        """Element offset of the tile's first float inside the flat pool (the half included)."""
        return (self.tile(slot, h) * self.tile_stride + self.half) * TILE

    @property
    def base_elem(self) -> int:
        """Element offset of the base pointer the host passes for this half (``half * TILE``)."""
        return self.half * TILE

    @property
    def numel(self) -> int:
        return self.n_tiles * self.tile_stride * TILE


def pool_addressing(hp: KcpHeadPlan, ctx: KcpContext, name: str) -> PoolAddressing:
    """The :class:`PoolAddressing` of pool ``name`` (:data:`POOLS`) for head plan ``hp`` of ``ctx``."""
    split, slots, stride, half = POOLS[name]
    n = int(getattr(ctx, slots))
    n_tiles = hp.n_owners * hp.Hg * n
    expected = sum(hp.splits[split]) // (stride * TILE) if n else 0
    if n_tiles != expected:
        raise ValueError(
            f"pool {name}: {n_tiles} tiles by the two-integer form, {expected} by the split table"
        )
    return PoolAddressing(
        name=name,
        hg=hp.Hg,
        n=n,
        tile_stride=stride,
        half=0 if half is None else int(half),
        n_tiles=n_tiles,
    )


# ----------------------------------------------------------------------------------------------------------------
# Merge chains (program #4, ``kda_chunk_kcp_affine_merge_fp32_v1``): CSR over element offsets


@dataclasses.dataclass(frozen=True)
class ChainTables:
    """Chain tables of one merge launch: CSR over elements ``(elem_m, elem_c, elem_y)`` in application order
    (element offsets into the ``m`` / ``c`` / ``y`` pools; ``elem_c = -1`` no offset; ``elem_y`` = where the state
    ENTERING the element is published when ``write_states``, ``-1`` none), the initial state ``x0_index`` (``-1``
    zero) and ``y_final`` (the state leaving the chain).  One chain per (document chain, owned head)."""

    family: str
    ncols: int
    chain_start: tuple[int, ...]
    elem_m: tuple[int, ...]
    elem_c: tuple[int, ...]
    elem_y: tuple[int, ...]
    x0_index: tuple[int, ...]
    y_final: tuple[int, ...]

    @property
    def n_chains(self) -> int:
        return len(self.x0_index)

    @property
    def n_elems(self) -> int:
        return len(self.elem_m)

    def check_alignment(self) -> None:
        for name in ("elem_m", "elem_c", "elem_y", "x0_index", "y_final"):
            if any(v % 4 for v in getattr(self, name) if v >= 0):
                raise ValueError(
                    f"ChainTables {self.family}: {name} holds an offset that is not a multiple of 4 floats"
                )
        if any(v < 0 for v in self.elem_m):
            raise ValueError(
                f"ChainTables {self.family}: every element needs an operator tile (elem_m >= 0)"
            )

    def device_tables(self, device) -> dict[str, torch.Tensor]:
        """The six tables on ``device`` in one asynchronous upload (``upload_tables``: no synchronizing call)."""
        i32, i64 = torch.int32, torch.int64
        return upload_tables(
            {
                "chain_start": (self.chain_start, i32, None),
                "elem_m": (self.elem_m, i64, None),
                "elem_c": (self.elem_c, i64, None),
                "elem_y": (self.elem_y, i64, None),
                "x0_index": (self.x0_index, i64, None),
                "y_final": (self.y_final, i64, None),
            },
            device,
        )


def _chains(
    family: str,
    hp: KcpHeadPlan,
    starts,
    m_off,
    c_off,
    out_off,
    sm: int,
    sc: int,
    so: int,
) -> ChainTables:
    chain_start, em, ec, ey, x0, yf = [0], [], [], [], [], []
    starts = [int(x) for x in starts]
    for ci in range(len(starts) - 1):
        e0, e1 = starts[ci], starts[ci + 1]
        for hg in range(hp.Hg_r):
            for e in range(e0 + 1, e1):
                em.append(int(m_off[e]) + hg * sm)
                ec.append(int(c_off[e]) + hg * sc)
                ey.append(int(out_off[e - 1]) + hg * so)
            chain_start.append(len(em))
            x0.append(int(c_off[e0]) + hg * sc)
            yf.append(int(out_off[e1 - 1]) + hg * so)
    tables = ChainTables(
        family=family,
        ncols=128,
        chain_start=tuple(chain_start),
        elem_m=tuple(em),
        elem_c=tuple(ec),
        elem_y=tuple(ey),
        x0_index=tuple(x0),
        y_final=tuple(yf),
    )
    tables.check_alignment()
    return tables


def owner_chains(hp: KcpHeadPlan) -> dict[str, ChainTables]:
    """The owner's merge chains of head plan ``hp`` (empty on a rank owning no heads).

    ``fwd``: per document and owned head, from ``X0 = B_{f1}`` (the start fragment's offset = ``S_in(f2)``) over
    the fragments ``f2 .. f(n-1)`` with ``Mop = M_f`` read from the c1 ``M_f^T`` tiles (``transpose_m = 0``); the
    state entering each element and the final state are the ``S_in`` of the element's fragment and of ``fn``,
    published into the c2 send layout.  ``adj_b1`` / ``adj_c1``: from ``X0 = d_{fn}`` over ``f(n-1) .. f2`` with
    ``Mop = M_f^T`` (``transpose_m = 1``), the maps read from the b1 pairs (``adj_b1``: ``save_transfer=False``) or
    from the saved c1 receive tensor (``adj_c1``: ``save_transfer=True``, or the replay); the entering / final
    states are the ``dS_out`` of the predecessors, published into the b2 send layout."""
    hs = hp.hg_stride
    return {
        "fwd": _chains(
            "fwd",
            hp,
            hp.fwd_chain_start,
            hp.fwd_m_off,
            hp.fwd_b_off,
            hp.fwd_out_off,
            hs["fwd_m_off"],
            hs["fwd_b_off"],
            hs["fwd_out_off"],
        ),
        "adj_b1": _chains(
            "adj_b1",
            hp,
            hp.adj_chain_start,
            hp.adj_m_off_b1,
            hp.adj_d_off_dm,
            hp.adj_out_off,
            hs["adj_m_off_b1"],
            hs["adj_d_off_dm"],
            hs["adj_out_off"],
        ),
        "adj_c1": _chains(
            "adj_c1",
            hp,
            hp.adj_chain_start,
            hp.adj_m_off_c1,
            hp.adj_d_off_d,
            hp.adj_out_off,
            hs["adj_m_off_c1"],
            hs["adj_d_off_d"],
            hs["adj_out_off"],
        ),
    }


# ----------------------------------------------------------------------------------------------------------------
# Table bundles (cached on the context)


@dataclasses.dataclass(frozen=True)
class LocalTables:
    """Head-independent tables of this rank's local programs (#1 prep, #5 output, #6 backward prep, #8 gradients,
    #9 gate epilogue): ``chunk_bos`` / ``chunk_len`` (``NC_pad`` entries, the pad chunks zero), ``seq_chunk_start``
    (``n_frag + 1``), ``NC`` real chunks, the grid bound ``NC_pad`` and the padded row count ``R = 64 NC_pad`` of
    the token-row tapes.  ``device`` holds the int32 device copies (the context's)."""

    T_r: int
    num_seqs: int
    NC: int
    NC_max: int
    NC_pad: int
    R: int
    chunk_bos: tuple[int, ...]
    chunk_len: tuple[int, ...]
    seq_chunk_start: tuple[int, ...]
    device: dict[str, torch.Tensor]

    def chunk_rows(self, c: int) -> range:
        """Padded tape rows of chunk ``c`` (``[64 c, 64 c + chunk_len[c])``)."""
        return range(c * CHUNK, c * CHUNK + self.chunk_len[c])


@dataclasses.dataclass(frozen=True)
class WalkTables:
    """Tables of this rank's walks (#2 forward state walk, #3 transfer walk, #7 adjoint walk) at ``HV`` heads:
    the segment plan (``P`` segments per fragment, ``NSEG = n_frag P`` pseudo-sequences), the
    chunk-aligned segment table ``seg_chunk_start [NSEG + 1]``, the segment lists ``walk_seg_offset`` (the segments
    the offset / transfer walks visit: crossing-out fragments at ``P = 1``, every segment at ``P > 1``) and
    ``walk_seg_tape`` (all), the index tables (``h0_index``: slot in ``S_in`` of segment 0 of a crossing-in
    fragment, else ``-1``; ``out_index``: c1 slot of a crossing-out fragment at ``P = 1`` / staging slot at
    ``P > 1``; ``dht_index``: ``dS_out`` slot of the last segment of a crossing-out fragment; ``carry_index`` /
    ``dcarry_index`` the inter-segment carry slots; the ``*_fused`` one-pool forms), the pool addressing of every
    exchange pool (:data:`POOLS`), the owner's merge chains and the int32 / int64 device copies."""

    HV: int
    Hg: int
    Hg_r: int
    n_owners: int
    P: int
    NSEG: int
    P_occ: int
    P_chain: int
    NC_max_frag: int
    p_rule: str
    seg_chunk_start: tuple[int, ...]
    walk_seg_offset: tuple[int, ...]
    walk_seg_tape: tuple[int, ...]
    h0_index: tuple[int, ...]
    out_index: tuple[int, ...]
    dht_index: tuple[int, ...]
    carry_index: tuple[int, ...]
    dcarry_index: tuple[int, ...]
    h0_index_fused: tuple[int, ...]
    dht_index_fused: tuple[int, ...]
    pools: dict[str, PoolAddressing]
    splits: dict[str, tuple[int, ...]]
    numel: dict[str, int]
    chains: dict[str, ChainTables]
    device: dict[str, torch.Tensor]

    @property
    def n_walk_offset(self) -> int:
        return len(self.walk_seg_offset)

    @property
    def n_walk_tape(self) -> int:
        return len(self.walk_seg_tape)

    def segment_chunks(self, seg: int) -> range:
        """Chunk ids of segment ``seg`` (``[seg_chunk_start[seg], seg_chunk_start[seg + 1])``)."""
        return range(self.seg_chunk_start[seg], self.seg_chunk_start[seg + 1])


def local_tables(ctx: KcpContext) -> LocalTables:
    """The :class:`LocalTables` of ``ctx`` (built once, cached on the context)."""
    key = ("gdn_local_tables",)
    cached = ctx._caches.get(key)
    if cached is None:
        cached = LocalTables(
            T_r=ctx.T_r,
            num_seqs=ctx.n_frag,
            NC=ctx.NC,
            NC_max=ctx.NC_max,
            NC_pad=ctx.NC_pad,
            R=ctx.R,
            chunk_bos=tuple(ctx.chunk_bos),
            chunk_len=tuple(ctx.chunk_len),
            seq_chunk_start=tuple(ctx.seq_chunk_start),
            device={name: ctx.tables[name] for name in LOCAL_TABLES},
        )
        ctx._caches[key] = cached
    return cached


def walk_tables(ctx: KcpContext, HV: int, *, P: int | None = None) -> WalkTables:
    """The :class:`WalkTables` of ``ctx`` at ``HV`` value heads (built once per ``(ctx, HV, P)``, cached on the
    context; the head plan ``ctx.heads(HV, P=P)`` is cached too).  ``P`` overrides the plan rule's segment count
    (the GDN host's segments knob)."""
    HV = int(HV)
    hp = ctx.heads(HV, P=P)
    key = ("gdn_walk_tables", HV, hp.P)
    cached = ctx._caches.get(key)
    if cached is not None:
        return cached
    pools = {name: pool_addressing(hp, ctx, name) for name in POOLS}
    chains = owner_chains(hp)
    device = {
        name: hp.device_tables[name] for name in WALK_TABLES if name in hp.device_tables
    }
    # the fused-walk index tables are CPU tuples on the head plan: uploaded once here, in one asynchronous copy
    device.update(
        upload_tables(
            {
                name: (getattr(hp, name), torch.int32, None)
                for name in WALK_TABLES
                if name not in hp.device_tables
            },
            ctx.device,
        )
    )
    for family, tables in chains.items():
        for name, t in tables.device_tables(ctx.device).items():
            device[f"{family}.{name}"] = t
    cached = WalkTables(
        HV=HV,
        Hg=hp.Hg,
        Hg_r=hp.Hg_r,
        n_owners=hp.n_owners,
        P=hp.P,
        NSEG=hp.NSEG,
        P_occ=hp.segments.P_occ,
        P_chain=hp.segments.P_chain,
        NC_max_frag=hp.segments.NC_max_frag,
        p_rule=hp.segments.p_rule,
        seg_chunk_start=tuple(hp.seg_chunk_start),
        walk_seg_offset=tuple(hp.walk_seg_offset),
        walk_seg_tape=tuple(hp.walk_seg_tape),
        h0_index=tuple(hp.h0_index),
        out_index=tuple(hp.out_index),
        dht_index=tuple(hp.dht_index),
        carry_index=tuple(hp.carry_index),
        dcarry_index=tuple(hp.dcarry_index),
        h0_index_fused=tuple(hp.h0_index_fused),
        dht_index_fused=tuple(hp.dht_index_fused),
        pools=pools,
        splits={k: tuple(v) for k, v in hp.splits.items()},
        numel=dict(hp.numel),
        chains=chains,
        device=device,
    )
    ctx._caches[key] = cached
    return cached


INT64_TABLES = (
    "elem_m",
    "elem_c",
    "elem_y",
    "x0_index",
    "y_final",
)  # the merge kernel's element offsets (verbatim)


@dataclasses.dataclass(frozen=True)
class PreparedTables:
    """The device-resident table bundle of one ``(HV, sm_count, P)`` key of a context (``ctx.prepare(H_qk, HV)``):
    the local chunk tables, the walk / pool / chain tables and the merged device dict (chain tables under
    ``"<family>.<name>"``), every tensor on ``ctx.device`` with the design's dtype (int32; ``INT64_TABLES`` int64)."""

    key: tuple[int, int, int]
    H_qk: int
    HV: int
    local: LocalTables
    walk: WalkTables
    device: dict[str, torch.Tensor]

    @property
    def P(self) -> int:
        return self.walk.P


def table_dtype(name: str) -> torch.dtype:
    """The design dtype of table ``name`` (``"<family>.<name>"`` accepted): int64 for the merge's element offsets,
    int32 otherwise."""
    return torch.int64 if name.rsplit(".", 1)[-1] in INT64_TABLES else torch.int32


def prepare(
    ctx: KcpContext, H_qk: int, HV: int, *, P: int | None = None
) -> PreparedTables:
    """Build (once per ``(HV, sm_count, P)`` key) and return the :class:`PreparedTables` of ``ctx`` for the head pair:
    the first-use upload (``build_kcp_context`` has no ``HV``; the host calls this before the
    first forward and the #7039 test before measuring).  ``HV`` must be a multiple of ``H_qk``; ``P`` is the
    segments override.  Every table dtype is checked against the design here, before any launch."""
    H_qk, HV = int(H_qk), int(HV)
    if H_qk < 1 or HV < 1 or HV % H_qk:
        raise ValueError(
            f"prepare: HV={HV} must be a positive multiple of H_qk={H_qk} (grouped value heads)"
        )
    hp = ctx.heads(HV, P=P)
    key = ("gdn_prepared", HV, int(ctx.sm_count), hp.P)
    cached = ctx._caches.get(key)
    if cached is None:
        lt = local_tables(ctx)
        wt = walk_tables(ctx, HV, P=P)
        device = {**lt.device, **wt.device}
        for name, t in device.items():
            if t.dtype != table_dtype(name):
                raise RuntimeError(
                    f"prepare: table {name} has dtype {t.dtype}, the design requires {table_dtype(name)}"
                )
            if (
                t.device != torch.device(ctx.device)
                and t.device.type != torch.device(ctx.device).type
            ):
                raise RuntimeError(
                    f"prepare: table {name} lives on {t.device}, the context on {ctx.device}"
                )
        cached = PreparedTables(
            key=(HV, int(ctx.sm_count), hp.P),
            H_qk=H_qk,
            HV=HV,
            local=lt,
            walk=wt,
            device=device,
        )
        ctx._caches[key] = cached
    return cached


def describe(ctx: KcpContext, HV: int) -> dict:
    """Plain-int summary of both bundles for reports."""
    lt, wt = local_tables(ctx), walk_tables(ctx, HV)
    return {
        "rank": ctx.rank,
        "T_r": lt.T_r,
        "n_frag": lt.num_seqs,
        "NC": lt.NC,
        "NC_pad": lt.NC_pad,
        "R": lt.R,
        "HV": wt.HV,
        "Hg": wt.Hg,
        "Hg_r": wt.Hg_r,
        "P": wt.P,
        "NSEG": wt.NSEG,
        "n_walk_offset": wt.n_walk_offset,
        "n_walk_tape": wt.n_walk_tape,
        "pools": {k: dataclasses.asdict(v) for k, v in wt.pools.items()},
        "chains": {k: (v.n_chains, v.n_elems) for k, v in wt.chains.items()},
        "halo": halo_rows(ctx),
    }


__all__ = [
    "CHAIN_FAMILIES",
    "INT64_TABLES",
    "LOCAL_TABLES",
    "POOLS",
    "WALK_TABLES",
    "ChainTables",
    "LocalTables",
    "PoolAddressing",
    "PreparedTables",
    "WalkTables",
    "describe",
    "halo_rows",
    "local_tables",
    "owner_chains",
    "owner_table",
    "pool_addressing",
    "prepare",
    "table_dtype",
    "walk_tables",
]
