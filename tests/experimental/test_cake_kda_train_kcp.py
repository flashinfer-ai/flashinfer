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

CPU tests of the KCP metadata layer (``cake_kcp_plan``): partitions against the rows of record, exchange
splits and chain tables, head ownership, the plan rule, the byte queries, the private test knobs; the GPU parity test
of the CPU-built chunk tables against the ``table`` program (``-k gpu``); and the FP64
fake backend against the unpartitioned FP64 reference (``-k fake``).  Rows, reference partitions and operands come
from the reference harness (``kcp_shapes``, ``kcp_inputs``) and the reference module ``reference_kcp`` when they are on
``PYTHONPATH``; the harness-backed tests are skipped otherwise.
"""

import ctypes
import gc
import json
import os
import time
import types
import warnings
import weakref

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

kp = cake_package("cake_kda_train", "cake_kcp_plan")
cake_backend = cake_package("cake_kda_train", "cake_backend")
cake_jit = cake_package("cake_kda_train", "cake_jit")
cake_kcp_jit = cake_package("cake_kda_train", "cake_kcp_jit")

try:  # the reference harness (rows of record and the reference partitions)
    import kcp_shapes
except ImportError:  # pragma: no cover - the harness is an optional environment module
    kcp_shapes = None

HARNESS = pytest.mark.skipif(
    kcp_shapes is None, reason="reference harness kcp_shapes not on PYTHONPATH"
)
SM_COUNT = 152  # GB300; the plan rule's examples of the design note §7.2
EVIDENCE_DIR = os.environ.get("KCP_EVIDENCE_DIR")
LAYOUT_NAMES = {"C": "contiguous", "Z": "zigzag", "D": "zigzag_per_document"}
OWNERSHIP_H = (1, 3, 4, 6, 12, 64)
OWNERSHIP_W = (2, 4, 8, 16)


def _evidence(name: str, record: dict) -> None:
    """Append one JSON record to ``$KCP_EVIDENCE_DIR/<name>.jsonl`` (the generated evidence)."""
    if not EVIDENCE_DIR:
        return
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, f"{name}.jsonl"), "a") as fh:
        fh.write(json.dumps(record) + "\n")


# --------------------------------------------------------------------------- rows of record


def _cases():
    """(row, W, layout letter) of every C row (all its CP and layouts, the explicit ragged tables as 'R'), every
    adversarial row of record and every perf pack at CP {2, 4, 8, 16} x {C, Z, D}."""
    if kcp_shapes is None:
        return []
    cases = []
    for row in kcp_shapes.CORRECT.values():
        for W in row.cp:
            for lay in row.layouts:
                cases.append((row.name, W, lay))
    for row in kcp_shapes.ADVERSARIAL.values():
        for W in row.cp:
            for lay in row.layouts:
                cases.append((row.name, W, lay))
    for row in kcp_shapes.PACKS.values():
        for W in (2, 4, 8, 16):
            for lay in ("C", "Z", "D"):
                cases.append((row.name, W, lay))
    return cases


CASES = _cases()
CASE_IDS = [f"{name}-cp{W}-{lay}" for name, W, lay in CASES]


def harness_partition(row, W: int, layout: str):
    """The harness fragments (reference) and this module's partition of the same (row, W, layout)."""
    frags = kcp_shapes.partition(row, W, layout)
    cu = torch.tensor(row.cu_seqlens, dtype=torch.int64)
    if layout == "R":
        table = [
            [
                (f.doc, f.off, f.length)
                for f in sorted((f for f in frags if f.rank == r), key=lambda f: f.slot)
            ]
            for r in range(W)
        ]
        part = kp.kcp_partition(cu, W, fragments=table)
    elif layout == "D":
        cuts = [kcp_shapes.doc_cuts_D(row, W, d) for d in range(len(row.lengths))]
        part = kp.kcp_partition(cu, W, layout="zigzag_per_document", cut_points=cuts)
    else:
        part = kp.kcp_partition(cu, W, layout=LAYOUT_NAMES[layout])
    return frags, part


def contexts(part, sm_count: int = SM_COUNT):
    return [
        kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=sm_count)
        for r in range(part.world_size)
    ]


@HARNESS
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_partition_matches_the_harness(case) -> None:
    """Gate 1: kcp_partition == kcp_shapes.partition / links / local_cu_seqlens on every row, CP and layout."""
    name, W, layout = case
    row = kcp_shapes.ROWS[name]
    frags, part = harness_partition(row, W, layout)
    links = kcp_shapes.links(frags)
    assert part.world_size == W and part.cu_seqlens == tuple(row.cu_seqlens)
    n_frags = 0
    for r in range(W):
        mine = part.fragments[r]
        theirs = sorted((f for f in frags if f.rank == r), key=lambda f: f.slot)
        assert [(f.doc, f.off, f.length, f.rank, f.slot) for f in mine] == [
            (f.doc, f.off, f.length, f.rank, f.slot) for f in theirs
        ], f"rank {r}: fragment table differs from the harness"
        for f in mine:
            lk = links[(r, f.slot)]
            assert f.prev == lk["prev"] and f.next == lk["next"], (
                f"rank {r} slot {f.slot}: links differ"
            )
            assert (
                f.crossing_in == lk["crossing_in"]
                and f.crossing_out == lk["crossing_out"]
            )
        assert part.local_cu_seqlens(r) == tuple(kcp_shapes.local_cu_seqlens(frags, r))
        n_frags += len(mine)
    assert n_frags == len(frags)
    _evidence(
        "partition",
        {"row": name, "W": W, "layout": layout, "fragments": n_frags, "equal": True},
    )


@HARNESS
@pytest.mark.parametrize("case", CASES, ids=CASE_IDS)
def test_splits_chains_and_bytes(case) -> None:
    """Gate 2: split sizes sum to the §7.4 buffer sizes on every rank and agree pairwise; the chain tables cover every
    document in order exactly once; in_slot / out_slot follow links(); byte totals match kcp_shapes.comm_bytes."""
    name, W, layout = case
    row = kcp_shapes.ROWS[name]
    H = row.h
    frags, part = harness_partition(row, W, layout)
    links = kcp_shapes.links(frags)
    ctxs = contexts(part)
    plans = [c.heads(H) for c in ctxs]
    s = H * kp.STATE_TILE_BYTES
    tile = kp.TILE
    n_in_total = sum(c.n_in for c in ctxs)
    n_out_total = sum(c.n_out for c in ctxs)
    assert n_in_total == n_out_total
    for r, (ctx, hp) in enumerate(zip(ctxs, plans, strict=True)):
        # crossing slots vs links()
        ni = no = 0
        for f in ctx.fragments:
            lk = links[(r, f.slot)]
            assert (ctx.in_slot[f.slot] >= 0) == lk["crossing_in"]
            assert (ctx.out_slot[f.slot] >= 0) == lk["crossing_out"]
            if lk["crossing_in"]:
                assert ctx.in_slot[f.slot] == ni
                ni += 1
            if lk["crossing_out"]:
                assert ctx.out_slot[f.slot] == no
                no += 1
        assert (
            (ctx.n_in, ctx.n_out) == (ni, no)
            and ctx.N_in == n_in_total
            and ctx.N_out == n_out_total
        )
        # §7.4 buffer sizes (elements): c1 send n_out 2s, c1 recv N_out 2s Hg_r/H, c2 send N_in s Hg_r/H, c2 recv n_in s;
        # b1 with maps = pairs [d_f | M_f^T] per crossing-in fragment: send n_in 2s, recv N_in 2s Hg_r/H.
        # No padding on any row of record (H = 4 / 64 with W in {2, 4, 8, 16}).
        assert hp.pad_tiles == 0 and hp.padded_owner == -1 and hp.full_groups
        assert all(v == 0 for v in hp.pad_numel.values())
        sp = hp.splits
        assert sum(sp["c1_send"]) * 4 == ctx.n_out * 2 * s
        assert sum(sp["c1_recv"]) == n_out_total * hp.Hg_r * 2 * tile
        assert sum(sp["c2_send"]) == n_in_total * hp.Hg_r * tile
        assert sum(sp["c2_recv"]) * 4 == ctx.n_in * s
        assert sum(sp["b1_send_d"]) * 4 == ctx.n_in * s
        assert sum(sp["b1_send_dm"]) * 4 == ctx.n_in * 2 * s
        assert sum(sp["b1_recv_d"]) == n_in_total * hp.Hg_r * tile
        assert sum(sp["b1_recv_dm"]) == n_in_total * hp.Hg_r * 2 * tile
        assert sum(sp["b2_send"]) == n_out_total * hp.Hg_r * tile
        assert sum(sp["b2_recv"]) * 4 == ctx.n_out * s
        for key, n in hp.numel.items():
            if key in sp:
                assert n == sum(sp[key])
        assert hp.numel["staging"] == (hp.NSEG * H * kp.SUMMARY_TILE if hp.P > 1 else 0)
        assert hp.tile_stride == {
            "c1": 2,
            "c2": 1,
            "b1_d": 1,
            "b1_dm": 2,
            "b2": 1,
            "staging": 1,
            "carry": 1,
        }
        # the b1 pair tables: d_f half even, M_f^T half = d_f half + 1, both indexed by the crossing-in slot
        assert len(hp.b1_d_tile_dm) == len(hp.b1_m_tile_dm) == ctx.n_in
        for s_ in range(ctx.n_in):
            for h in range(H):
                pair = ((h // hp.Hg) * ctx.n_in + s_) * hp.Hg + h % hp.Hg
                assert hp.b1_d_tile_dm[s_][h] == 2 * pair + kp.TILE_D
                assert hp.b1_m_tile_dm[s_][h] == 2 * pair + kp.TILE_M
                assert hp.b1_d_tile_d[s_][h] == pair
        # every chain-table offset is -1 or a multiple of 4 floats
        for name_ in (
            "fwd_m_off",
            "fwd_b_off",
            "fwd_out_off",
            "adj_d_off_d",
            "adj_d_off_dm",
            "adj_m_off_b1",
            "adj_m_off_c1",
            "adj_out_off",
        ):
            assert all(o == -1 or (o >= 0 and o % 4 == 0) for o in getattr(hp, name_))
            assert name_ in hp.hg_stride
        # pairwise agreement of every collective's split tables
        for send_key, recv_key in (
            ("c1_send", "c1_recv"),
            ("c2_send", "c2_recv"),
            ("b1_send_d", "b1_recv_d"),
            ("b1_send_dm", "b1_recv_dm"),
            ("b2_send", "b2_recv"),
        ):
            for j in range(W):
                assert hp.splits[send_key][j] == plans[j].splits[recv_key][r], (
                    send_key,
                    r,
                    j,
                )
        for j in range(W):
            assert ctx.halo.fwd_send_rows[j] == ctxs[j].halo.fwd_recv_rows[r]
            assert ctx.halo.bwd_send_rows[j] == ctxs[j].halo.bwd_recv_rows[r]
        # halo plan facts
        assert ctx.halo.n_hist_rows == sum(ctx.halo.hist_valid)
        for f in ctx.fragments:
            assert ctx.halo.hist_valid[f.slot] == (
                min(3, f.off) if (f.length > 0 and f.off > 0) else 0
            )
            rows = ctx.halo.hist_row_off[f.slot]
            assert sum(1 for x in rows if x >= 0) == ctx.halo.hist_valid[f.slot]
            if f.crossing_out:
                length = row.lengths[f.doc]
                assert sum(1 for x in ctx.halo.tail_row_off[f.slot] if x >= 0) == min(
                    3, length - f.end
                )
            else:
                assert all(x == -1 for x in ctx.halo.tail_row_off[f.slot])
        assert sorted(x for r_ in ctx.halo.hist_row_off for x in r_ if x >= 0) == list(
            range(ctx.halo.n_hist_rows)
        )
        assert sorted(x for r_ in ctx.halo.tail_row_off for x in r_ if x >= 0) == list(
            range(ctx.halo.n_tail_rows)
        )
        assert sorted(
            x for r_ in ctx.halo.head_row_off for k in r_ for x in k if x >= 0
        ) == list(range(ctx.halo.n_head_rows))
        assert len(ctx.halo.hist_index) == sum(ctx.halo.fwd_send_rows)
        assert all(0 <= i < ctx.T_r for i in ctx.halo.hist_index)
        # chain tables of this owner: every document with >= 2 fragments, in order, every element exactly once
        chains = part.chains()
        if hp.Hg_r > 0:
            docs = [d for d, c in enumerate(chains) if len(c) >= 2]
            assert list(hp.fwd_chain_doc) == docs and list(hp.adj_chain_doc) == docs
            assert len(hp.fwd_chain_start) == len(docs) + 1 and hp.fwd_chain_start[
                -1
            ] == len(hp.fwd_m_off)
            for ci, d in enumerate(docs):
                assert (
                    hp.fwd_chain_start[ci + 1] - hp.fwd_chain_start[ci]
                    == len(chains[d]) - 1
                )
                assert (
                    hp.adj_chain_start[ci + 1] - hp.adj_chain_start[ci]
                    == len(chains[d]) - 1
                )
            assert len(hp.fwd_m_off) == n_out_total == len(hp.adj_d_off_d)
            assert (
                len(set(hp.fwd_m_off)) == n_out_total
                and len(set(hp.fwd_b_off)) == n_out_total
            )
            assert (
                len(set(hp.fwd_out_off)) == n_in_total
                and len(set(hp.adj_out_off)) == n_out_total
            )
            assert all(
                0 <= o < hp.numel["c1_recv"] for o in hp.fwd_m_off + hp.fwd_b_off
            )
            assert all(0 <= o < hp.numel["c2_send"] for o in hp.fwd_out_off)
            assert all(0 <= o < hp.numel["b2_send"] for o in hp.adj_out_off)
            assert all(0 <= o < hp.numel["b1_recv_d"] for o in hp.adj_d_off_d)
            assert all(0 <= o < hp.numel["b1_recv_dm"] for o in hp.adj_d_off_dm)
            assert all(
                o == -1 or (0 <= o < hp.numel["b1_recv_dm"]) for o in hp.adj_m_off_b1
            )
            assert all(
                o == -1 or o == hp.adj_d_off_dm[e] + kp.TILE
                for e, o in enumerate(hp.adj_m_off_b1)
            )  # the M half sits beside the d half of the same pair
            assert sum(1 for o in hp.adj_m_off_b1 if o == -1) == len(
                docs
            )  # one map-less element (document end) per chain
            for ci in range(len(docs)):
                e0 = hp.adj_chain_start[ci]
                assert hp.adj_m_off_b1[e0] == -1 and hp.adj_m_off_c1[e0] == -1
                assert all(
                    hp.adj_m_off_b1[e] >= 0
                    for e in range(e0 + 1, hp.adj_chain_start[ci + 1])
                )
        else:
            assert (
                not hp.fwd_m_off
                and not hp.adj_d_off_d
                and hp.numel["c1_recv"] == 0
                and hp.numel["c2_send"] == 0
            )
    # byte totals over the ranks match the harness
    hb = kcp_shapes.comm_bytes(frags, H)
    totals = {"c1_send": 0, "c2_recv": 0, "b2_recv": 0, "in": 0}
    for ctx in ctxs:
        cb = kp.kcp_comm_bytes(ctx, H=H)
        totals["c1_send"] += cb["c1"]["send"]
        totals["c2_recv"] += cb["c2"]["recv"]
        totals["b2_recv"] += cb["b2"]["recv"]
        totals["in"] += cb["crossing_fragments_in"]
        assert cb["state_bytes"] == hb["state_bytes"]
    assert totals["in"] == hb["crossing_fragments_in"]
    assert totals["c2_recv"] == hb["fwd_state_bytes"]
    assert totals["c1_send"] == hb["fwd_summary_bytes"]
    assert totals["b2_recv"] == hb["bwd_adjoint_bytes"]
    _evidence(
        "splits",
        {
            "row": name,
            "W": W,
            "layout": layout,
            "H": H,
            "N_in": n_in_total,
            "N_out": n_out_total,
            "fwd_state_bytes": totals["c2_recv"],
            "fwd_summary_bytes": totals["c1_send"],
            "bwd_adjoint_bytes": totals["b2_recv"],
            "harness": hb,
            "P": [hp.P for hp in plans],
            "NSEG": [hp.NSEG for hp in plans],
            "pass": True,
        },
    )


# --------------------------------------------------------------------------- ownership, plan rule, limits (no harness)


def _small_partition(W: int, lengths=(700, 1, 1300, 999), layout="zigzag_per_document"):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return kp.kcp_partition(torch.tensor(cu, dtype=torch.int64), W, layout=layout)


@pytest.mark.parametrize("H", OWNERSHIP_H)
@pytest.mark.parametrize("W", OWNERSHIP_W)
def test_head_ownership_rows(H, W) -> None:
    """Gate 3: Hg = ceil(H / W), owner(h) = h // Hg, Hg_r = clamp(H - r Hg, 0, Hg); W not dividing H and H < W give
    owners 0 .. H-1 with empty splits on the rest; every head has exactly one owner; the single partial owner is
    padded to Hg tiles per slot so the two-integer K1 / K2 addressing is exact for every H, and the padding
    is reported separately."""
    Hg, n_owners, hg_r = kp.head_groups(H, W)
    assert Hg == -(-H // W) and n_owners == -(-H // Hg) and sum(hg_r) == H
    assert all(0 <= x <= Hg for x in hg_r) and all(
        x == Hg for x in hg_r[: n_owners - 1]
    )
    if H < W:
        assert Hg == 1 and n_owners == H and hg_r == tuple([1] * H + [0] * (W - H))
    owners = [kp.owner_of_head(h, H, W) for h in range(H)]
    assert owners == [h // Hg for h in range(H)] and max(owners) == n_owners - 1
    part = _small_partition(W)
    ctxs = contexts(part)
    plans = [c.heads(H) for c in ctxs]
    full = n_owners * Hg == H
    partial = [j for j, x in enumerate(hg_r) if 0 < x < Hg]
    assert len(partial) <= 1 and (not partial) == full
    padded_owner = partial[0] if partial else -1
    pad_tiles = n_owners * Hg - H
    N_in = sum(c.n_in for c in ctxs)
    N_out = sum(c.n_out for c in ctxs)
    for r, (ctx, hp) in enumerate(zip(ctxs, plans, strict=True)):
        assert (hp.Hg, hp.n_owners, hp.Hg_r, hp.head0, hp.full_groups) == (
            Hg,
            n_owners,
            hg_r[r],
            r * Hg,
            full,
        )
        # padding: every owner with heads holds Hg tile slots per fragment slot
        assert hp.Hg_pad_by_rank == tuple(Hg if x > 0 else 0 for x in hg_r)
        assert (hp.padded_owner, hp.pad_tiles) == (padded_owner, pad_tiles)
        assert sum(hp.splits["c1_send"]) == ctx.n_out * n_owners * Hg * kp.SUMMARY_TILE
        assert sum(hp.splits["c2_recv"]) == ctx.n_in * n_owners * Hg * kp.TILE
        assert hp.pad_numel["c1_send"] == ctx.n_out * pad_tiles * kp.SUMMARY_TILE
        assert hp.pad_numel["c2_recv"] == ctx.n_in * pad_tiles * kp.TILE
        assert hp.pad_numel["b1_send_d"] == ctx.n_in * pad_tiles * kp.TILE
        assert hp.pad_numel["b1_send_dm"] == ctx.n_in * pad_tiles * kp.SUMMARY_TILE
        assert hp.pad_numel["b2_recv"] == ctx.n_out * pad_tiles * kp.TILE
        own_pad = (Hg - hp.Hg_r) if 0 < hp.Hg_r < Hg else 0
        assert hp.pad_numel["c1_recv"] == N_out * own_pad * kp.SUMMARY_TILE
        assert hp.pad_numel["c2_send"] == N_in * own_pad * kp.TILE
        assert hp.pad_numel["b1_recv_dm"] == N_in * own_pad * kp.SUMMARY_TILE
        assert hp.pad_numel["b2_send"] == N_out * own_pad * kp.TILE
        for i in range(W):
            assert (
                hp.splits["c1_recv"][i]
                == ctxs[i].n_out * hp.Hg_pad_by_rank[r] * kp.SUMMARY_TILE
            )
            assert plans[i].splits["c1_send"][r] == hp.splits["c1_recv"][i]
        cb = kp.kcp_comm_bytes(ctx, H=H)
        assert cb["padding"]["c1"] == {
            "send": hp.pad_numel["c1_send"] * 4,
            "recv": hp.pad_numel["c1_recv"] * 4,
        }
        assert cb["padding"]["padded_owner"] == padded_owner
        assert cb["padding"]["pad_tiles_per_slot"] == pad_tiles
        assert cb["padding"]["s_in_saved"] == hp.pad_numel["c2_recv"] * 4
        assert cb["padding"]["B1"] == {
            "send": (hp.pad_numel["c1_send"] + hp.pad_numel["c2_send"]) * 4,
            "recv": (hp.pad_numel["c1_recv"] + hp.pad_numel["c2_recv"]) * 4,
        }
        assert (cb["padding"]["B1"]["send"] + cb["padding"]["B1"]["recv"] > 0) == (
            pad_tiles > 0
        )
        # saved set: tables + padded S_in + the carry plane of a P > 1 rank (kept
        # by default; saved_carry=False counts the plane-less re-walk form)
        tb_bytes = kp.tables_bytes(ctx.NC_pad, ctx.n_frag, hp.NSEG)
        no_plane = tb_bytes + ctx.n_in * n_owners * Hg * kp.STATE_TILE_BYTES
        plane_slots = ctx.n_frag * (hp.P - 1) + max(ctx.n_in, ctx.n_out)
        plane = plane_slots * H * kp.STATE_TILE_BYTES if hp.P > 1 else 0
        got = kp.kcp_saved_bytes(ctx, H=H, with_conv=False, save_transfer=False)
        got_no_plane = kp.kcp_saved_bytes(
            ctx, H=H, with_conv=False, save_transfer=False, saved_carry=False
        )
        assert got == no_plane + plane
        assert got_no_plane == no_plane
        if (
            hp.Hg_r == 0
        ):  # a rank owning no heads: empty owner-side splits and chains, still a party to every collective
            assert (
                sum(hp.splits["c1_recv"]) == 0
                and sum(hp.splits["c2_send"]) == 0
                and not hp.fwd_m_off
            )
            assert all(
                sum(plans[i].splits["c1_send"][r : r + 1]) == 0 for i in range(W)
            )
        # explicit tile tables == the two-integer form for every H (padding makes it exact)
        for s_, tiles in enumerate(hp.h0_tile):
            for h in range(H):
                two_int = ((h // Hg) * ctx.n_in + s_) * Hg + h % Hg
                assert tiles[h] == two_int
            assert len(set(tiles)) == H
        for s_, tiles in enumerate(hp.out_tile):
            for h in range(H):
                assert tiles[h] == ((h // Hg) * ctx.n_out + s_) * Hg + h % Hg
        for s_, tiles in enumerate(hp.dht_tile):
            for h in range(H):
                assert tiles[h] == ((h // Hg) * ctx.n_out + s_) * Hg + h % Hg
        # the padded owner's chain tables walk hg < Hg_r only; their hg stride spans the padded Hg slots
        if hp.Hg_r > 0 and hp.fwd_m_off:
            assert all(
                o + (hp.Hg - 1) * hp.hg_stride["fwd_m_off"] < hp.numel["c1_recv"]
                for o in hp.fwd_m_off
            )
        all_tiles = [t for tiles in hp.h0_tile for t in tiles]
        assert len(set(all_tiles)) == len(all_tiles) and all(
            0 <= t * kp.TILE < max(1, hp.numel["c2_recv"]) for t in all_tiles
        )
        all_out = [t for tiles in hp.out_tile for t in tiles]
        assert len(set(all_out)) == len(all_out) and all(
            0 <= t * kp.SUMMARY_TILE < max(1, hp.numel["c1_send"]) for t in all_out
        )
    _evidence(
        "ownership",
        {
            "H": H,
            "W": W,
            "Hg": Hg,
            "n_owners": n_owners,
            "Hg_r": list(hg_r),
            "full_groups": full,
            "padded_owner": padded_owner,
            "pad_tiles": pad_tiles,
            "pass": True,
        },
    )


def test_tile_offsets_follow_p1_r1() -> None:
    """Chain-table tile offsets are -1 or non-negative multiples of 4 floats; the checker rejects anything
    else before a launch."""
    kp._check_tile_offsets("ok", (-1, 0, 4, kp.TILE, 3 * kp.SUMMARY_TILE))
    for bad in ((3,), (-2,), (kp.TILE + 2,)):
        with pytest.raises(ValueError, match="multiples of 4 floats"):
            kp._check_tile_offsets("bad", bad)
    part = _small_partition(4)
    for ctx in contexts(part):
        hp = ctx.heads(6)
        for name_, stride in hp.hg_stride.items():
            assert stride in (kp.TILE, kp.SUMMARY_TILE)
            assert all(o == -1 or (o >= 0 and o % 4 == 0) for o in getattr(hp, name_))
            assert hp.device_tables[name_].dtype == torch.int64


def test_zero_token_rank_keeps_chains_and_splits() -> None:
    """Gate 3 (C4b): a rank with T_r = 0 has no local sequences and no token-side tables but still owns heads, holds
    its chain tables and joins every collective with zero-length splits."""
    table = [[(0, 0, 1200)], [], [(0, 1200, 900)], [(0, 2100, 900)]]
    part = kp.kcp_partition(
        torch.tensor([0, 3000], dtype=torch.int64), 4, fragments=table
    )
    assert part.rank_tokens(1) == 0 and part.local_cu_seqlens(1) == (0,)
    ctxs = contexts(part)
    ctx = ctxs[1]
    assert (
        ctx.T_r == 0
        and ctx.n_frag == 0
        and ctx.NC_pad == 0
        and ctx.seq_chunk_start == (0,)
        and ctx.seq_order == ()
    )
    hp = ctx.heads(4)
    assert hp.Hg_r == 1 and hp.P == 1 and hp.NSEG == 0 and hp.seg_chunk_start == (0,)
    assert (
        len(hp.fwd_chain_doc) == 1
        and len(hp.fwd_m_off) == 2
        and len(hp.adj_d_off_d) == 2
    )  # chain 0 -> 2 -> 3
    assert sum(hp.splits["c1_send"]) == 0 and sum(hp.splits["c2_recv"]) == 0
    assert (
        sum(hp.splits["c1_recv"]) == 2 * kp.SUMMARY_TILE
        and sum(hp.splits["c2_send"]) == 2 * kp.TILE
    )
    for other in (ctxs[0], ctxs[2], ctxs[3]):
        assert other.heads(4).splits["c1_send"][1] == other.n_out * kp.SUMMARY_TILE
    assert (
        kp.kcp_workspace_bytes(ctx, H=4) == kp.kcp_arena_layout(ctx, H=4)["arena"]
    )  # no local arena without tokens
    assert kp.kcp_saved_bytes(ctx, H=4, save_transfer=False) == kp.tables_bytes(0, 0, 0)
    assert kp.kcp_saved_bytes(ctx, H=4) == kp.kcp_saved_bytes(
        ctx, H=4, save_transfer=True
    )
    cb = kp.kcp_comm_bytes(ctx, H=4)
    assert (
        cb["c1"]["send"] == 0
        and cb["c1"]["recv"] == 2 * kp.SUMMARY_TILE * 4
        and cb["halo_fwd"] == {"send": 0, "recv": 0}
    )


def test_kcp_plan_examples() -> None:
    """Gate 3: the shipped default rule is wave_fill1_p1tie (the one-wave fill rule plus the P = 1 tie line, on top of the
    earlier wave rule), and the §7.2 examples at 152 SMs hold
    under the earlier occ rule, now selected explicitly."""
    assert kp.P_RULE_DEFAULT == "wave_fill1_p1tie" and kp.P_RULE_DEFAULT in kp.P_RULES
    dflt = kp.kcp_plan(1, 256, 64, SM_COUNT)
    assert dflt == kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="wave_fill1_p1tie")
    fill1 = kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="wave_fill1")
    assert (dflt.P, dflt.NSEG) == (
        fill1.P,
        fill1.NSEG,
    )  # one fragment: the P = 1 tie line does not apply
    # 256 chunks is AT the chain cap, not above it: the wave line does not apply; the earlier wave
    # rule gives the occ P = 3 (192 CTAs on 152 SMs), the shipped one-wave fill the largest one-CTA-per-SM P = 2
    wave = kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="wave")
    assert (wave.p_rule, wave.P, wave.P_occ, wave.P_chain) == ("wave", 3, 3, 1)
    assert wave.P == kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="occ").P
    assert (dflt.p_rule, dflt.P, dflt.P_occ, dflt.P_chain) == (
        "wave_fill1_p1tie",
        2,
        3,
        1,
    )
    # the first NC above the cap: P_chain = 2 and the wave line applies under the shipped rule as under wave
    above = kp.kcp_plan(1, 257, 64, SM_COUNT)
    assert (above.p_rule, above.P_chain) == ("wave_fill1_p1tie", 2) and above.P >= 2
    assert above.P == kp.kcp_plan(1, 257, 64, SM_COUNT, p_rule="wave").P
    assert kp.kcp_plan(12, 64, 64, SM_COUNT, p_rule="occ").P == 1  # D / CP16 Q3
    assert kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="occ").P == 3  # C / CP16
    assert kp.kcp_plan(1, 244, 64, SM_COUNT, p_rule="occ").P == 3
    assert kp.kcp_plan(2, 1950, 64, SM_COUNT, p_rule="occ").P == 8  # C / CP2
    p = kp.kcp_plan(1, 64, 4, SM_COUNT, p_rule="occ")  # H = 4, one 64-chunk fragment
    assert (p.P_occ, p.P_chain, p.P, p.NSEG) == (38, 1, 38, 38)
    p3 = kp.kcp_plan(3, 10, 64, SM_COUNT, p_rule="occ")
    assert (p3.P, p3.NSEG, p3.P_occ, p3.P_chain) == (1, 3, 1, 1)
    assert (p3.P_bal, p3.n_frag, p3.NC_max_frag, p3.p_rule) == (64, 3, 10, "occ")
    assert kp.kcp_plan(1, 5, 1, SM_COUNT, p_rule="occ").P == 5  # capped by NC_max_frag
    assert kp.kcp_plan(1, 100000, 64, SM_COUNT, p_rule="occ").P == 64  # 64-segment cap
    p0 = kp.kcp_plan(0, 0, 64, SM_COUNT, p_rule="occ")
    assert (p0.P, p0.NSEG, p0.P_occ, p0.P_chain) == (1, 0, 1, 1)
    assert (p0.P_bal, p0.n_frag, p0.NC_max_frag, p0.p_rule) == (64, 0, 0, "occ")
    p20 = kp.kcp_plan(2, 0, 64, SM_COUNT, p_rule="occ")
    assert (p20.P, p20.NSEG) == (1, 2)
    assert kp.kcp_plan(1, 64, 4, SM_COUNT, P_bal=8, p_rule="occ").P == 8  # P_bal cap
    with pytest.raises(ValueError):
        kp.kcp_plan(1, 64, 0, SM_COUNT)
    _evidence(
        "plan_examples",
        {
            "D_CP16_Q3": 1,
            "C_CP16": 3,
            "C_CP2": 8,
            "H4_64chunk": 38,
            "default_rule": kp.P_RULE_DEFAULT,
            "default_C_CP16_P": dflt.P,
            "pass": True,
        },
    )


# The plan rule.  Rule of record: on ranks with n_frag <= WAVE_N_FRAG_MAX = 3 the "wave"
# rule is argmin_P over P in [P_chain, min(64, NC_max)] (smallest P on ties) of
#   ceil(n_frag P H / SM) x ceil(NC_max / P) x t_walk + (P - 1) x t_compose,  t_walk = 2.85 + 3.5 + 1.2 us, t_compose = 20.4 us
# (10-ns integers); ranks with more fragments, and ranks whose longest fragment is <= 256 chunks (P_chain = 1), keep the occ P.  Information form (b) (superseded): the same
# with the LPT makespan over the actual per-fragment chunk counts (kp.wave_rule_lpt_P).  GB300: SM = 152.  `_lpt_ref` is an
# independent job-by-job greedy (the oracle of the LPT helper).
WALK_10NS, COMPOSE_10NS = 755, 2040


def _lpt_ref(chains, machines: int) -> int:
    """Reference LPT list scheduling: every job (longest first) to the least-loaded machine (a heap of loads)."""
    import heapq

    heap = [0] * int(machines)
    for L in sorted((L for L, m in chains for _ in range(m)), reverse=True):
        heapq.heapreplace(heap, heap[0] + L)
    return max(heap)


def _wave_ref_cost(nc_frags, H: int, sm: int, P: int) -> int:
    chains = [(-(-nc // P), P * H) for nc in nc_frags if nc > 0]
    return (_lpt_ref(chains, sm) if chains else 0) * WALK_10NS + (P - 1) * COMPOSE_10NS


def _wave_ref_P(nc_frags, H: int, sm: int) -> int:
    nc_max = max(nc_frags, default=0)
    if nc_max == 0:
        return 1
    hi = max(1, min(64, nc_max))
    lo = min(max(1, -(-nc_max // 256)), hi)
    return min(range(lo, hi + 1), key=lambda P: (_wave_ref_cost(nc_frags, H, sm, P), P))


def _argmin_a(n_frag: int, nc_max: int, H: int, sm: int = SM_COUNT) -> int:
    """Form (a) without the guard (the equal-segment argmin for any n_frag; information on guarded ranks)."""
    if nc_max == 0:
        return 1
    hi = max(1, min(64, nc_max))
    lo = min(max(1, -(-nc_max // 256)), hi)
    return min(
        range(lo, hi + 1),
        key=lambda P: (kp.wave_rule_cost_10ns(n_frag, nc_max, H, sm, P), P),
    )


WAVE_EXAMPLES_H64 = {
    # (n_frag <= 3, NC_max_frag): (occ P, wave P of record, wave cost us at its P, (runner-up P, its cost us))
    (1, 488): (
        3,
        7,
        1707.90,
        (2, 1862.60),
    ),  # Q1 CP8 C one-fragment ranks: 3 waves x 70 steps vs 1 wave x 244
    (1, 976): (4, 7, 3293.40, (4, 3745.60)),  # Q1 CP4 C rank 3 (P_chain = 4)
    (1, 244): (3, 7, 915.15, (2, 941.50)),  # Q1 CP16 C one-fragment ranks
    (2, 525): (3, 7, 3519.90, (3, 4004.55)),  # Q1 CP4 C rank 0 (28874 + 33590 tokens)
    (2, 895): (4, 7, 5920.80, (13, 5975.25)),  # Q1 CP4 C rank 1
    (2, 931): (4, 7, 6147.30, (13, 6224.40)),  # Q1 CP4 C rank 2
    (2, 452): (2, 7, 3066.90, (13, 3151.55)),  # Q1 CP8 C rank 0
    (2, 407): (2, 7, 2795.10, (8, 2838.15)),  # Q1 CP8 C rank 2
    (2, 443): (2, 7, 3021.60, (8, 3102.40)),  # Q1 CP8 C rank 4
    (2, 208): (2, 7, 1481.40, (8, 1516.90)),  # Q1 CP16 C rank 1 (13258 + 2358 tokens)
    (2, 163): (2, 7, 1209.60, (1, 1230.65)),  # Q1 CP16 C rank 4
    (2, 199): (2, 7, 1436.10, (8, 1464.05)),  # Q1 CP16 C rank 8
    (3, 488): (
        2,
        7,
        4878.90,
        (2, 5547.00),
    ),  # Q1 CP4 Z ranks 0 / 2 / 3 (3 fragments: the guard still applies)
    (2, 1907): (
        8,
        13,
        12453.15,
        (19, 12568.00),
    ),  # Q1 CP2 C rank 1 (2940 + 121988 tokens): 11 waves x 147 steps
}
# Guarded ranks (n_frag > 3) keep the occ P: (n_frag, NC_max): (occ P == wave P, unguarded form (a) P for information)
WAVE_GUARDED_H64 = {
    (4, 931): (4, 7),  # Q1 CP2 Z rank 1 (82 / 895 / 46 / 931 chunks)
    (8, 251): (
        1,
        5,
    ),  # Q1 CP4 D rank 3 (8 fragments; measured 8-11 % slower at P = 4 / 3 and at P = 5)
    (8, 127): (1, 2),  # Q1 CP8 D rank 2
    (12, 64): (
        1,
        4,
    ),  # D / CP16 Q3 (§7.2 example): unguarded (a) would take 21 waves x 16 steps at P = 4
}
# Information form (b) on the ACTUAL fragment tables, by hand (H = 64, SM = 152): {nc_frags: (lpt P, cost us, (runner-up P, cost))}
WAVE_LPT_H64 = {
    (452, 37): (
        2,
        1726.70,
        (7, 1730.55),
    ),  # Q1 CP8 C rank 0: the near tie the measurement contradicts (P = 7 is ~6 % faster)
    (82, 407): (7, 1730.55, (6, 1747.90)),  # Q1 CP8 C rank 2
    (46, 443): (2, 1696.50, (7, 1730.55)),  # Q1 CP8 C rank 4
    (452, 37, 488): (7, 3270.75, (8, 3283.60)),  # Q1 CP4 Z rank 0
}
# Q1 (lengths 28874 / 38814 / 60180 / 121988, cu 0 / 28874 / 67688 / 127868 / 249856) C fragments per rank by hand:
# CP4 T_r = 62464; CP8 T_r = 31232; CP16 T_r = 15616 (ceil(len / 64) chunks per fragment).
Q1_C_FRAGMENT_SETS = {
    4: [(2, 525), (2, 895), (2, 931), (1, 976)],
    8: [(2, 452), (1, 488), (2, 407), (1, 488), (2, 443), (1, 488), (1, 488), (1, 488)],
    16: [
        (1, 244),
        (2, 208),
        (1, 244),
        (1, 244),
        (2, 163),
        (1, 244),
        (1, 244),
        (1, 244),
        (2, 199),
        (1, 244),
        (1, 244),
        (1, 244),
        (1, 244),
        (1, 244),
        (1, 244),
        (1, 244),
    ],
}
Q1_C_ACTUAL = {  # the actual per-fragment chunk counts of the same ranks
    4: [(452, 525), (82, 895), (46, 931), (976,)],
    8: [(452, 37), (488,), (82, 407), (488,), (46, 443), (488,), (488,), (488,)],
    16: [(244,), (208, 37), (244,), (244,), (82, 163)]
    + [(244,)] * 3
    + [(46, 199)]
    + [(244,)] * 7,
}
# The A/B rows and G1 re-judge rows (row, CP, layout, H): the CPU evidence `plan_wave_lane_rows` records occ / record
# ((a) + guard) / unguarded (a) / information (b) P per rank, so the rows whose program changes are known before any GPU step.
LANE_LC_ROWS = (
    *[("Q1", W, lay, 64) for W in (2, 4, 8, 16) for lay in ("C", "Z", "D")],
    ("C1", 2, "C", 4),
    ("C1", 4, "C", 4),
    ("C2", 4, "C", 4),
    ("C2", 4, "Z", 4),
    ("C2", 4, "D", 4),
    ("C4a", 4, "R", 4),
    ("C4b", 4, "R", 4),
    ("C5b", 4, "R", 4),
    ("C7", 4, "R", 1),
    ("A1v2", 8, "C", 4),
    ("A2v2", 8, "C", 4),
    ("A4v2", 8, "C", 4),
    ("A4v2", 16, "C", 4),
)


def test_lpt_makespan_matches_the_job_by_job_greedy() -> None:
    """The water-filling LPT of the plan module equals the job-by-job greedy on random inputs and on the hand cases."""
    import random

    rng = random.Random(3837)
    for _ in range(400):
        machines = rng.choice([1, 2, 3, 7, 148, 152, 160])
        chains = [
            (rng.randint(1, 300), rng.randint(1, 700)) for _ in range(rng.randint(1, 6))
        ]
        assert kp.lpt_makespan(chains, machines) == _lpt_ref(chains, machines), (
            chains,
            machines,
        )
    assert kp.lpt_makespan([(226, 128), (19, 128)], 152) == 226
    assert kp.lpt_makespan([(65, 448), (6, 448)], 152) == 213
    assert kp.lpt_makespan([(70, 448), (65, 448), (6, 448)], 152) == 417
    assert kp.lpt_makespan([(61, 512), (57, 512), (5, 512)], 152) == 416
    assert kp.lpt_makespan([], 152) == 0
    assert kp.lpt_makespan([(5, 3)], 1) == 15
    assert (
        kp.lpt_makespan([(70, 448)], 152) == 3 * 70
    )  # equal jobs: ceil(jobs / machines) x length


def test_kcp_plan_wave_rule_examples() -> None:
    """The plan rule at 152 SMs, H = 64: the rule of record equals the hand-derived argmin on ranks with <= 3 fragments,
    the occ P on guarded ranks; the information form (b) equals its hand values and the oracle; occ is untouched."""
    walk = sum(kp.WAVE_RULE_10NS[k] for k in ("tape", "xfer", "offset"))
    assert (walk, kp.WAVE_RULE_10NS["compose"], kp.WAVE_N_FRAG_MAX) == (
        WALK_10NS,
        COMPOSE_10NS,
        3,
    )
    for (n_frag, nc), (p_occ, p_wave, cost, (p2, cost2)) in WAVE_EXAMPLES_H64.items():
        occ = kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="occ")
        wave = kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="wave")
        assert (occ.P, occ.p_rule) == (p_occ, "occ"), (n_frag, nc, occ)
        p_rec = (
            p_wave if nc >= kp.WAVE_MIN_NC_MAX_FRAG else p_occ
        )  # short fragments keep the occ P
        assert (wave.P, wave.NSEG, wave.p_rule) == (p_rec, n_frag * p_rec, "wave"), (
            n_frag,
            nc,
            wave,
        )
        assert _argmin_a(n_frag, nc, 64) == p_wave
        assert (wave.P_occ, wave.P_chain, wave.P_bal) == (
            occ.P_occ,
            occ.P_chain,
            occ.P_bal,
        )
        assert kp.wave_rule_cost_us(n_frag, nc, 64, SM_COUNT, p_wave) == pytest.approx(
            cost, abs=1e-9
        ), (n_frag, nc)
        assert kp.wave_rule_cost_us(n_frag, nc, 64, SM_COUNT, p2) == pytest.approx(
            cost2, abs=1e-9
        ), (n_frag, nc)
        # exhaustive argmin over the admissible range (smallest P on ties); equal fragments reproduce it through the LPT form
        hi = min(64, nc)
        costs = [
            (kp.wave_rule_cost_10ns(n_frag, nc, 64, SM_COUNT, P), P)
            for P in range(wave.P_chain, hi + 1)
        ]
        assert min(costs) == (
            kp.wave_rule_cost_10ns(n_frag, nc, 64, SM_COUNT, p_wave),
            p_wave,
        )
        assert kp.wave_rule_lpt_P([nc] * n_frag, 64, SM_COUNT) == p_wave
        assert kp.wave_rule_lpt_cost_10ns(
            [nc] * n_frag, 64, SM_COUNT, p_wave
        ) == kp.wave_rule_cost_10ns(n_frag, nc, 64, SM_COUNT, p_wave)
    for (n_frag, nc), (p_occ, p_a) in WAVE_GUARDED_H64.items():
        occ = kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="occ")
        wave = kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="wave")
        assert (occ.P, wave.P, wave.p_rule) == (p_occ, p_occ, "wave"), (
            n_frag,
            nc,
            wave,
        )
        assert _argmin_a(n_frag, nc, 64) == p_a, (n_frag, nc)
    for nc_frags, (p_lpt, cost, (p2, cost2)) in WAVE_LPT_H64.items():
        nc_frags = list(nc_frags)
        assert (
            kp.wave_rule_lpt_P(nc_frags, 64, SM_COUNT)
            == p_lpt
            == _wave_ref_P(nc_frags, 64, SM_COUNT)
        ), nc_frags
        assert kp.wave_rule_lpt_cost_10ns(
            nc_frags, 64, SM_COUNT, p_lpt
        ) / 100.0 == pytest.approx(cost, abs=1e-9)
        assert kp.wave_rule_lpt_cost_10ns(
            nc_frags, 64, SM_COUNT, p2
        ) / 100.0 == pytest.approx(cost2, abs=1e-9)
        assert kp.wave_rule_lpt_cost_10ns(
            nc_frags, 64, SM_COUNT, p_lpt
        ) == _wave_ref_cost(nc_frags, 64, SM_COUNT, p_lpt)
    # H = 4 rows (one fragment per rank, NC_max <= 256): the chain condition keeps the occ P; the unguarded (a) argmin
    # (the compose term decides, 4 P CTAs never fill a wave) is information
    for n_frag, nc, p_a in (
        (1, 65, 5),
        (1, 79, 5),
        (1, 256, 10),
        (1, 128, 8),
        (1, 64, 5),
    ):
        assert (
            kp.kcp_plan(n_frag, nc, 4, SM_COUNT, p_rule="wave").P
            == kp.kcp_plan(n_frag, nc, 4, SM_COUNT).P
            == 38
        ), (n_frag, nc)
        assert _argmin_a(n_frag, nc, 4) == p_a, (n_frag, nc)
    assert kp.wave_rule_cost_us(1, 65, 4, SM_COUNT, 5) == pytest.approx(179.75)
    assert (
        kp.kcp_plan(1, 257, 4, SM_COUNT, p_rule="wave").P == _argmin_a(1, 257, 4) == 10
    )  # first NC above the chain cap
    assert (
        kp.kcp_plan(4, 10, 4, SM_COUNT, p_rule="wave").P
        == kp.kcp_plan(4, 10, 4, SM_COUNT).P
        == 10
    )  # C1 CP2 rank 0: guarded
    assert kp.WAVE_MIN_NC_MAX_FRAG == 257
    # edge cases and the untouched occ / P_bal paths
    assert kp.kcp_plan(0, 0, 64, SM_COUNT, p_rule="wave") == kp.KcpSegmentPlan(
        P=1,
        NSEG=0,
        P_occ=1,
        P_chain=1,
        P_bal=64,
        n_frag=0,
        NC_max_frag=0,
        p_rule="wave",
    )
    assert kp.kcp_plan(2, 0, 64, SM_COUNT, p_rule="wave").P == 1
    assert kp.kcp_plan(1, 1, 64, SM_COUNT, p_rule="wave").P == 1
    assert (
        kp.kcp_plan(1, 100000, 64, SM_COUNT, p_rule="wave").P == 64
    )  # P_chain 391 > 64: the cap wins
    assert (
        kp.kcp_plan(1, 64, 4, SM_COUNT, P_bal=8, p_rule="occ").P == 8
    )  # P_bal still caps the occ rule only
    assert (
        kp.kcp_plan(1, 64, 4, SM_COUNT, P_bal=8, p_rule="wave").P == 8
    )  # a short fragment keeps the occ rule incl. P_bal
    assert (
        _argmin_a(1, 488, 4)
        == kp.kcp_plan(1, 488, 4, SM_COUNT, P_bal=8, p_rule="wave").P
    )  # P_bal does not enter the wave rule
    assert (
        kp.kcp_plan(8, 64, 4, SM_COUNT, P_bal=8, p_rule="wave").P
        == kp.kcp_plan(8, 64, 4, SM_COUNT, P_bal=8, p_rule="occ").P
    )  # guarded: occ incl. P_bal
    dflt_488 = kp.kcp_plan(1, 488, 64, SM_COUNT)
    wave_488 = kp.kcp_plan(1, 488, 64, SM_COUNT, p_rule="wave")
    assert (dflt_488.P, dflt_488.NSEG, dflt_488.p_rule) == (
        wave_488.P,
        wave_488.NSEG,
        "wave_fill1_p1tie",
    )  # above the chain cap the shipped default (wave_fill1) is the wave rule
    assert kp.kcp_plan(1, 488, 64, SM_COUNT, p_rule="occ").P == 3
    with pytest.raises(ValueError, match="p_rule"):
        kp.kcp_plan(1, 488, 64, SM_COUNT, p_rule="lpt")
    assert kp.wave_rule_lpt_P([0, 0], 64, SM_COUNT) == 1
    _evidence(
        "plan_wave_examples",
        {
            "record": {str(k): v[1] for k, v in WAVE_EXAMPLES_H64.items()},
            "guarded": {str(k): v[0] for k, v in WAVE_GUARDED_H64.items()},
            "lpt_info": {str(k): v[0] for k, v in WAVE_LPT_H64.items()},
            "pass": True,
        },
    )


def test_kcp_plan_wave_rule_is_pure_in_sm_count() -> None:
    """The wave rule is a pure function of (fragment table, H, SM count): repeatable, order-independent, and the
    SM count enters only through the wave quantization (GB200 148 SMs: 576 CTAs = 4 waves -> P = 9 for 488 chunks;
    SM = 1: every admissible P costs n_frag P H ceil(NC / P) walk steps, non-decreasing in P -> P = P_chain = 2)."""
    for sm in (1, 132, 148, 152, 160):
        a = kp.kcp_plan(1, 488, 64, sm, p_rule="wave")
        b = kp.kcp_plan(1, 488, 64, sm, p_rule="wave")
        assert a == b and a.p_rule == "wave"
    assert [kp.kcp_plan(1, 488, 64, sm, p_rule="wave").P for sm in (152, 148, 1)] == [
        7,
        9,
        2,
    ]
    assert kp.wave_rule_cost_us(1, 488, 64, 148, 9) == pytest.approx(1824.20)
    assert (
        kp.kcp_plan(1, 488, 64, 152, p_rule="wave").P == 7
    )  # unchanged after the other SM counts (no hidden state)
    for sm in (
        148,
        152,
    ):  # the information form is order-independent in the fragment table
        assert kp.wave_rule_lpt_P([452, 37], 64, sm) == kp.wave_rule_lpt_P(
            [37, 452], 64, sm
        )


@HARNESS
@pytest.mark.parametrize("W", (4, 8, 16))
def test_kcp_plan_wave_rule_on_q1_c(monkeypatch, W) -> None:
    """The Q1 C fragment sets at CP4 / CP8 / CP16 (hand-derived from Q1's lengths) and the P of the rules per rank
    through the context (knob p_rule via build_kcp_context; describe() records it; the shipped default is wave_fill1 =
    wave above the chain cap + the one-wave fill on one-fragment ranks below it): every rank has <= 2 fragments: at
    CP4 / CP8 (fragments > 256 chunks) the wave rule gives P = 7 everywhere (= the default), at CP16 (<= 244 chunks)
    the chain condition keeps the occ P under wave while the shipped fill plans P = 2 on every rank (one-fragment
    ranks 3 -> 2, two-fragment ranks already 2); the information form (b) is recorded."""
    row = kcp_shapes.PACKS["Q1"]
    assert list(row.lengths) == [28874, 38814, 60180, 121988]
    _frags, part = harness_partition(row, W, "C")
    expect_occ = {
        4: [3, 4, 4, 4],
        8: [2, 3, 2, 3, 2, 3, 3, 3],
        16: [3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3],
    }[W]
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.build_kcp_context(
            part,
            None,
            0,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave"},
        )
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    expect_wave = [7] * W if W in (4, 8) else expect_occ
    expect_dflt = (
        [7] * W if W in (4, 8) else [2] * W
    )  # wave_fill1 (plan diff of record: Q1 CP16 C H64 -> [2] x16)
    got_lpt = []
    for r in range(W):
        mine = part.fragments[r]
        actual = [-(-f.length // 64) for f in mine]
        n_frag, nc = len(mine), max(actual)
        assert (n_frag, nc) == Q1_C_FRAGMENT_SETS[W][r], (
            W,
            r,
            [(f.doc, f.off, f.length) for f in mine],
        )
        assert tuple(actual) == Q1_C_ACTUAL[W][r], (W, r, actual)
        dflt = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        occ = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "occ"},
        )
        wave = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave"},
        )
        assert (occ.p_rule, wave.p_rule, dflt.p_rule) == (
            "occ",
            "wave",
            "wave_fill1_p1tie",
        )
        assert (dflt.heads(64).P, dflt.describe(64)["p_rule"]) == (
            expect_dflt[r],
            "wave_fill1_p1tie",
        )
        assert (
            dflt.heads(64).P
            == kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="wave_fill1").P
        )
        fill1 = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )
        assert (fill1.p_rule, fill1.heads(64).P) == ("wave_fill1", expect_dflt[r])
        p_occ_plan = kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="occ").P
        assert occ.heads(64).P == expect_occ[r] == p_occ_plan
        assert (
            wave.heads(64).P
            == expect_wave[r]
            == kp.kcp_plan(n_frag, nc, 64, SM_COUNT, p_rule="wave").P
        ), (W, r)
        assert (wave.heads(64).NSEG, wave.heads(64).segments.p_rule) == (
            expect_wave[r] * n_frag,
            "wave",
        )
        assert (
            occ.describe(64)["p_rule"],
            wave.describe(64)["p_rule"],
            wave.describe(64)["heads"]["P"],
        ) == ("occ", "wave", expect_wave[r])
        # the byte queries follow the context's rule (staging and carry planes grow with P)
        wb, ob = kp.kcp_workspace_bytes(wave, H=64), kp.kcp_workspace_bytes(occ, H=64)
        assert (wb > ob) if expect_wave[r] > expect_occ[r] else (wb == ob), (
            W,
            r,
            wb,
            ob,
        )
        p_lpt = kp.wave_rule_lpt_P(actual, 64, SM_COUNT)
        assert p_lpt == _wave_ref_P(actual, 64, SM_COUNT), (W, r, actual)
        if tuple(actual) in WAVE_LPT_H64:
            assert p_lpt == WAVE_LPT_H64[tuple(actual)][0], (W, r, actual)
        got_lpt.append(p_lpt)
    if W == 8:
        assert got_lpt == [2, 7, 7, 7, 2, 7, 7, 7]
    _evidence(
        "plan_wave_q1_c",
        {
            "W": W,
            "occ": expect_occ,
            "wave": expect_wave,
            "wave_fill1": expect_dflt,
            "default_rule": kp.P_RULE_DEFAULT,
            "lpt_info": got_lpt,
        },
    )


@HARNESS
@pytest.mark.parametrize(
    "case", LANE_LC_ROWS, ids=[f"{n}-cp{W}-{lay}-H{H}" for n, W, lay, H in LANE_LC_ROWS]
)
def test_kcp_plan_wave_rule_lane_rows(monkeypatch, case) -> None:
    """Every A/B row and G1 re-judge row of the wave-rule study: per rank the actual fragment table and the P of occ / the rule of
    record ((a) + guard) / unguarded (a) / information (b); the record equals (a) when n_frag <= 3 and occ otherwise,
    the context's plan uses it, (b) equals the oracle.  The evidence record `plan_wave_lane_rows` decides which rows
    measured under another form must be re-run on the rule of record."""
    name, W, lay, H = case
    row = kcp_shapes.ROWS[name]
    _frags, part = harness_partition(row, W, lay)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    ranks = []
    for r in range(W):
        mine = part.fragments[r]
        actual = [-(-f.length // 64) for f in mine]
        n_frag, nc = len(mine), max(actual, default=0)
        occ = kp.kcp_plan(n_frag, nc, H, SM_COUNT, p_rule="occ")
        rec = kp.kcp_plan(n_frag, nc, H, SM_COUNT, p_rule="wave")
        p_a = _argmin_a(n_frag, nc, H) if n_frag else 1
        p_b = kp.wave_rule_lpt_P(actual, H, SM_COUNT) if n_frag else 1
        assert p_b == (_wave_ref_P(actual, H, SM_COUNT) if n_frag else 1), (
            case,
            r,
            actual,
        )
        applies = n_frag <= kp.WAVE_N_FRAG_MAX and nc >= kp.WAVE_MIN_NC_MAX_FRAG
        assert (p_a if applies else occ.P) == rec.P, (case, r, actual, rec)
        if n_frag <= 1:
            assert p_a == p_b
        ctx = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave"},
        )
        assert ctx.heads(H).P == rec.P, (case, r, ctx.heads(H))
        ranks.append(
            {
                "rank": r,
                "nc_frags": actual,
                "occ": occ.P,
                "record": rec.P,
                "a": p_a,
                "b": p_b,
                "guarded": n_frag > kp.WAVE_N_FRAG_MAX,
                "short": nc < kp.WAVE_MIN_NC_MAX_FRAG,
            }
        )
    _evidence(
        "plan_wave_lane_rows",
        {
            "row": name,
            "cp": W,
            "layout": lay,
            "H": H,
            "sm": SM_COUNT,
            "n_frag_max": kp.WAVE_N_FRAG_MAX,
            "min_nc_max": kp.WAVE_MIN_NC_MAX_FRAG,
            "occ": [x["occ"] for x in ranks],
            "record": [x["record"] for x in ranks],
            "a": [x["a"] for x in ranks],
            "b": [x["b"] for x in ranks],
            "record_differs_from_occ": any(x["record"] != x["occ"] for x in ranks),
            "record_differs_from_a": any(x["record"] != x["a"] for x in ranks),
            "record_differs_from_b": any(x["record"] != x["b"] for x in ranks),
            "ranks": ranks,
        },
    )


# --------------------------------------------------------------------------- the one-wave fill rules


def _fill_ref(n_frag: int, nc_max: int, H: int, sm: int, n_frag_max: int) -> int:
    """Oracle of the fill rules: below the chain cap, on a rank with at most ``n_frag_max`` fragments, the largest P
    with n_frag P H <= SM clamped to [P_chain (= 1 there), min(64, NC_max)]; otherwise the shipped ``wave`` P."""
    if (
        0 < n_frag <= n_frag_max
        and 0 < nc_max < kp.WAVE_MIN_NC_MAX_FRAG
        and sm // (n_frag * H) >= 1
    ):
        return max(1, min(64, nc_max, sm // (n_frag * H)))
    return kp.kcp_plan(n_frag, nc_max, H, sm, p_rule="wave").P


def _onewave_argmin_ref(n_frag: int, nc_max: int, H: int, sm: int) -> int:
    """Information form F2 of the enumeration: the equal-segment argmin restricted to one-wave P (n_frag P H
    <= SM) below the chain cap; occ when no one-wave P exists; the shipped P elsewhere."""
    if 0 < n_frag <= kp.WAVE_N_FRAG_MAX and 0 < nc_max < kp.WAVE_MIN_NC_MAX_FRAG:
        hi = min(64, nc_max, sm // (n_frag * H))
        if hi >= 1:
            return min(
                range(1, hi + 1),
                key=lambda P: (kp.wave_rule_cost_10ns(n_frag, nc_max, H, sm, P), P),
            )
        return kp.kcp_plan(n_frag, nc_max, H, sm, p_rule="occ").P
    return kp.kcp_plan(n_frag, nc_max, H, sm, p_rule="wave").P


def _a_all_ref(n_frag: int, nc_max: int, H: int, sm: int) -> int:
    """Information form F1: the equal-segment argmin for every rank with n_frag <= 3, no chain-cap condition (the
    equal-segment program measured earlier); occ on guarded ranks."""
    if 0 < n_frag <= kp.WAVE_N_FRAG_MAX and nc_max > 0:
        return _argmin_a(n_frag, nc_max, H, sm)
    return kp.kcp_plan(n_frag, nc_max, H, sm, p_rule="occ").P


def test_kcp_plan_wave_fill_examples() -> None:
    """The one-wave fill rule knob values at 152 SMs: the one-fragment CP16 C ranks move P 3 -> 2 (one wave of 128 CTAs), two-fragment
    H = 64 ranks move 2 -> 1 under wave_fill only, every rank above the chain cap and every H <= 4 rank keeps its program,
    the public default is untouched."""
    SM = SM_COUNT
    assert kp.P_RULES == (
        "occ",
        "wave",
        "wave_fill1",
        "wave_fill",
        "wave_fill1_p1tie",
    )
    assert kp.WAVE_FILL_N_FRAG_MAX == {"wave_fill1": 1, "wave_fill": 3}
    for nc in (222, 244, 256):
        assert kp.kcp_plan(1, nc, 64, SM, p_rule="occ").P == 3
        assert kp.kcp_plan(1, nc, 64, SM, p_rule="wave").P == 3
        assert kp.kcp_plan(1, nc, 64, SM, p_rule="wave_fill1").P == 2
        assert kp.kcp_plan(1, nc, 64, SM, p_rule="wave_fill").P == 2
        assert (
            _argmin_a(1, nc, 64) == 7
        )  # the literal equal-segment argmin (measured slower)
    for nc in (122, 208, 244):
        assert (
            kp.kcp_plan(2, nc, 64, SM, p_rule="wave").P,
            kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill1").P,
            kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill").P,
        ) == (2, 2, 1)
    assert (
        kp.kcp_plan(3, 122, 64, SM, p_rule="wave").P,
        kp.kcp_plan(3, 122, 64, SM, p_rule="wave_fill1").P,
        kp.kcp_plan(3, 122, 64, SM, p_rule="wave_fill").P,
    ) == (1, 1, 1)
    for n_frag, nc in ((1, 488), (2, 525), (3, 488), (1, 976), (2, 1907)):
        w = kp.kcp_plan(n_frag, nc, 64, SM, p_rule="wave").P
        assert w == kp.kcp_plan(n_frag, nc, 64, SM, p_rule="wave_fill1").P
        assert w == kp.kcp_plan(n_frag, nc, 64, SM, p_rule="wave_fill").P
    assert kp.kcp_plan(1, 488, 64, SM, p_rule="wave_fill1").P == 7
    for n_frag, nc, H in (
        (1, 65, 4),
        (1, 79, 4),
        (1, 256, 4),
        (1, 128, 4),
        (1, 12, 4),
        (1, 21, 4),
        (2, 14, 4),
        (1, 79, 1),
        (1, 16, 1),
        (1, 1, 1),
        (3, 4, 4),
    ):
        occ = kp.kcp_plan(n_frag, nc, H, SM, p_rule="occ").P
        assert occ == kp.kcp_plan(n_frag, nc, H, SM, p_rule="wave").P
        assert occ == kp.kcp_plan(n_frag, nc, H, SM, p_rule="wave_fill1").P, (
            n_frag,
            nc,
            H,
        )
        assert occ == kp.kcp_plan(n_frag, nc, H, SM, p_rule="wave_fill").P, (
            n_frag,
            nc,
            H,
        )
    assert (
        kp.kcp_plan(8, 251, 64, SM, p_rule="wave_fill").P
        == kp.kcp_plan(8, 251, 64, SM, p_rule="occ").P
        == 1
    )
    assert (
        kp.kcp_plan(4, 10, 4, SM, p_rule="wave_fill").P
        == kp.kcp_plan(4, 10, 4, SM, p_rule="occ").P
        == 10
    )
    for rule in ("wave_fill1", "wave_fill"):
        assert kp.kcp_plan(0, 0, 64, SM, p_rule=rule) == kp.KcpSegmentPlan(
            P=1,
            NSEG=0,
            P_occ=1,
            P_chain=1,
            P_bal=64,
            n_frag=0,
            NC_max_frag=0,
            p_rule=rule,
        )
        assert kp.kcp_plan(2, 0, 64, SM, p_rule=rule).P == 1
        assert kp.kcp_plan(1, 1, 64, SM, p_rule=rule).P == 1
        assert kp.kcp_plan(1, 100000, 64, SM, p_rule=rule).P == 64
    # the SM count enters only through floor(SM / (n_frag H)): GB200 148 SMs -> 2 as well; SM = 1 -> no one-wave P -> occ
    assert kp.kcp_plan(1, 244, 64, 148, p_rule="wave_fill1").P == 2
    assert (
        kp.kcp_plan(1, 244, 64, 1, p_rule="wave_fill1").P
        == kp.kcp_plan(1, 244, 64, 1, p_rule="occ").P
        == 1
    )
    # an SM count that is not a multiple of H would move H = 4 rows (150 SMs: 37 vs 38); 152 does not
    assert kp.kcp_plan(1, 65, 4, 150, p_rule="wave_fill1").P == 37
    assert kp.kcp_plan(1, 65, 4, 150, p_rule="occ").P == 38
    # P_bal caps the occ rule only; the other plan terms are reported alike; the public default is untouched
    assert kp.kcp_plan(1, 64, 4, SM, P_bal=8, p_rule="wave_fill1").P == 38
    f = kp.kcp_plan(1, 244, 64, SM, p_rule="wave_fill1")
    o = kp.kcp_plan(1, 244, 64, SM, p_rule="occ")
    assert (f.P_occ, f.P_chain, f.P_bal, f.NSEG, f.p_rule) == (
        o.P_occ,
        o.P_chain,
        o.P_bal,
        2,
        "wave_fill1",
    )
    assert (
        kp.kcp_plan(1, 244, 64, SM).P == 2 and kp.P_RULE_DEFAULT == "wave_fill1_p1tie"
    )
    with pytest.raises(ValueError, match="p_rule"):
        kp.kcp_plan(1, 244, 64, SM, p_rule="fill")
    _evidence("plan_wave_fill_examples", {"pass": True})


def test_wave_fill_knobs_are_gated(monkeypatch) -> None:
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.validate_test_knobs({"p_rule": "wave_fill1"})
    assert kp.validate_test_knobs(None)["p_rule"] == "wave_fill1_p1tie"
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    for rule in ("wave_fill1", "wave_fill"):
        assert kp.validate_test_knobs({"p_rule": rule}) == {
            "precision_mode": 0,
            "p_rule": rule,
        }


def _kb_rows():
    """Enumeration set: every C row at its CP / layouts for H in {1, 4, 64} (the H2 gate-2 / gate-4
    rows and their H coverage), the CP16 semantics row C6, every adversarial row at its CP in layouts C / Z (A3v2: D / Z),
    and every perf pack at CP {2, 4, 8, 16} x {C, Z, D} at H = 64."""
    if kcp_shapes is None:
        return []
    rows = []
    for row in kcp_shapes.CORRECT.values():
        for W in row.cp:
            for lay in row.layouts:
                for H in (1, 4, 64):
                    rows.append((row.name, W, lay, H))
    rows.append(("C6", 16, "Z", 4))
    for row in kcp_shapes.ADVERSARIAL.values():
        for W in row.cp:
            for lay in ("C", "Z") if "C" in row.layouts else ("D", "Z"):
                rows.append((row.name, W, lay, 4))
    for row in kcp_shapes.PACKS.values():
        for W in (2, 4, 8, 16):
            for lay in ("C", "Z", "D"):
                rows.append((row.name, W, lay, 64))
    return rows


KB_ROWS = _kb_rows()


@HARNESS
@pytest.mark.parametrize(
    "case", KB_ROWS, ids=[f"{n}-cp{W}-{lay}-H{H}" for n, W, lay, H in KB_ROWS]
)
def test_kcp_plan_wave_fill_lane_rows(monkeypatch, case) -> None:
    """The one-wave fill enumeration: per rank of every row of :func:`_kb_rows` the P under occ / the shipped
    wave rule / wave_fill1 (knob of record) / wave_fill, plus the information forms F1 (equal-segment argmin for every
    n_frag <= 3 rank) and F2 (one-wave argmin) and the 148-SM values; the module equals the oracles, wave_fill1 differs
    from wave on one-fragment ranks below the cap only and never at H <= 4, and a context built with the knob plans with
    it.  The evidence record `plan_wave_fill_rows` defines the re-judge set (the plan-diff tool)."""
    name, W, lay, H = case
    row = kcp_shapes.ROWS[name]
    _frags, part = harness_partition(row, W, lay)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    ranks = []
    for r in range(W):
        mine = part.fragments[r]
        actual = [-(-f.length // 64) for f in mine]
        n_frag, nc = len(mine), max(actual, default=0)
        rec = {"rank": r, "nc_frags": actual, "n_frag": n_frag, "nc_max": nc}
        for rule in kp.P_RULES:
            rec[rule] = kp.kcp_plan(n_frag, nc, H, SM_COUNT, p_rule=rule).P
        rec["a_all"] = _a_all_ref(n_frag, nc, H, SM_COUNT)
        rec["onewave_argmin"] = _onewave_argmin_ref(n_frag, nc, H, SM_COUNT)
        rec["sm148"] = {
            rule: kp.kcp_plan(n_frag, nc, H, 148, p_rule=rule).P
            for rule in ("wave", "wave_fill1", "wave_fill")
        }
        ranks.append(rec)
    _evidence(
        "plan_wave_fill_rows",
        {
            "row": name,
            "cp": W,
            "layout": lay,
            "H": H,
            "sm": SM_COUNT,
            "rules": list(kp.P_RULES),
            "n_frag_max": kp.WAVE_N_FRAG_MAX,
            "min_nc_max": kp.WAVE_MIN_NC_MAX_FRAG,
            "fill_n_frag_max": dict(kp.WAVE_FILL_N_FRAG_MAX),
            **{rule: [x[rule] for x in ranks] for rule in kp.P_RULES},
            "a_all": [x["a_all"] for x in ranks],
            "onewave_argmin": [x["onewave_argmin"] for x in ranks],
            "changed_wave_fill1": [
                x["rank"] for x in ranks if x["wave_fill1"] != x["wave"]
            ],
            "changed_wave_fill": [
                x["rank"] for x in ranks if x["wave_fill"] != x["wave"]
            ],
            "ranks": ranks,
        },
    )
    for rec in ranks:
        n_frag, nc = rec["n_frag"], rec["nc_max"]
        assert rec["wave_fill1"] == _fill_ref(n_frag, nc, H, SM_COUNT, 1), (case, rec)
        assert rec["wave_fill"] == _fill_ref(
            n_frag, nc, H, SM_COUNT, kp.WAVE_N_FRAG_MAX
        ), (case, rec)
        if n_frag != 1:
            assert rec["wave_fill1"] == rec["wave"], (case, rec)
        if nc >= kp.WAVE_MIN_NC_MAX_FRAG or n_frag > kp.WAVE_N_FRAG_MAX:
            assert rec["wave_fill"] == rec["wave"] == rec["wave_fill1"], (case, rec)
        if H <= 4:
            assert rec["wave_fill1"] == rec["wave"], (case, rec)
    changed = [rec["rank"] for rec in ranks if rec["wave_fill1"] != rec["wave"]]
    for r in changed[:1] or [0]:
        ctx = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )
        assert (ctx.p_rule, ctx.heads(H).P, ctx.describe(H)["p_rule"]) == (
            "wave_fill1",
            ranks[r]["wave_fill1"],
            "wave_fill1",
        )


# --------------------------------------------------------------------------- the P = 1 tie rule and the fused carry merges


def _p1tie_ref(n_frag: int, nc_frags, H: int, sm: int) -> int:
    """Oracle of the "wave_fill1_p1tie" rule: the "wave_fill1" P, then 1 on a rank with 2..3 fragments below
    the chain cap whose base P > 1 runs more CTAs than one wave of P = 1 (n_frag H <= SM) when the LPT makespan of one
    equal-segment walk at P = 1 over the fragment chunk counts is <= the base P's (the wave model's waves x steps on equal
    fragments)."""
    nc_frags = [int(x) for x in nc_frags]
    nc_max = max(nc_frags, default=0)
    base = kp.kcp_plan(n_frag, nc_max, H, sm, p_rule="wave_fill1").P
    if (
        2 <= n_frag <= kp.WAVE_N_FRAG_MAX
        and 0 < nc_max < kp.WAVE_MIN_NC_MAX_FRAG
        and base > 1
        and n_frag * H <= sm
    ):

        def steps(P: int) -> int:
            return kp.lpt_makespan(
                [(-(-nc // P), P * H) for nc in nc_frags if nc > 0], sm
            )

        if steps(1) <= steps(base):
            return 1
    return base


def test_kcp_plan_p1tie_examples() -> None:
    """The P = 1 tie rule at 152 SMs: equal two-fragment H = 64 ranks below the chain cap (the CP16 / CP8
    Z ranks of record) move P 2 -> 1; unequal two-fragment ranks (CP16 C ranks cut by a document boundary), three-fragment
    ranks, one-fragment ranks, ranks above the cap and every H <= 4 rank keep the shipped program; nc_frags is validated;
    the public default is untouched."""
    SM = SM_COUNT
    assert kp.P1_TIE_BASE == {"wave_fill1_p1tie": "wave_fill1"}
    assert "wave_fill1_p1tie" in kp.P_RULES and kp.P_RULE_DEFAULT == "wave_fill1_p1tie"
    for nc in (56, 111, 118, 122, 125, 128, 200, 256):
        assert kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill1").P == 2
        assert kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill1_p1tie").P == 1
        assert (
            kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill1_p1tie", nc_frags=[nc, nc]).P
            == 1
        )
        assert kp.p1_tie_steps([nc, nc], 64, SM, 1) == nc
        assert kp.p1_tie_steps([nc, nc], 64, SM, 2) == 2 * -(-nc // 2) >= nc
        assert _p1tie_ref(2, [nc, nc], 64, SM) == 1
    # equal fragments reproduce the wave model's waves x steps exactly
    for n_frag, nc, P in (
        (2, 111, 2),
        (2, 111, 1),
        (3, 122, 1),
        (1, 244, 2),
        (1, 244, 3),
        (2, 170, 2),
        (2, 60, 19),
    ):
        waves = -(-(n_frag * P * 64) // SM)
        assert kp.p1_tie_steps([nc] * n_frag, 64, SM, P) == waves * -(-nc // P)
    # unequal two-fragment ranks: P = 1 walks the long fragment alone -> the base P stays
    for frags in ([170, 59], [200, 30], [60, 62], [130, 100], [111, 50]):
        nc = max(frags)
        assert (
            kp.kcp_plan(2, nc, 64, SM, p_rule="wave_fill1_p1tie", nc_frags=frags).P == 2
        ), frags
        assert kp.p1_tie_steps(frags, 64, SM, 1) > kp.p1_tie_steps(frags, 64, SM, 2)
        assert _p1tie_ref(2, frags, 64, SM) == 2
    assert (
        kp.p1_tie_steps([60, 62], 64, SM, 2) == 61
    )  # LPT: 31 + 30 on 80 of the 152 SMs
    assert kp.p1_tie_steps([170, 59], 64, SM, 2) == 115  # 85 + 30
    assert (
        kp.kcp_plan(2, 100, 64, SM, p_rule="wave_fill1_p1tie", nc_frags=[100, 100]).P
        == 1
    )
    # an empty second fragment: one live chain group of 64 CTAs at P = 1 (111 steps) vs 128 CTAs x 56 at P = 2 -> stays 2
    assert (
        kp.kcp_plan(2, 111, 64, SM, p_rule="wave_fill1_p1tie", nc_frags=[111, 0]).P == 2
    )
    # three-fragment H = 64 ranks (P_occ = 1 already), ranks above the cap (the wave argmin), one-fragment ranks (the
    # fill), many-fragment ranks: unchanged
    for n_frag, nc in (
        (3, 122),
        (3, 80),
        (2, 257),
        (2, 488),
        (3, 300),
        (1, 244),
        (1, 100),
        (1, 488),
        (4, 100),
        (8, 251),
    ):
        assert (
            kp.kcp_plan(n_frag, nc, 64, SM, p_rule="wave_fill1_p1tie").P
            == kp.kcp_plan(n_frag, nc, 64, SM, p_rule="wave_fill1").P
        ), (n_frag, nc)
    # H <= 4 ranks keep their program: the base P fills the SMs with chains of ceil(NC / P) < NC steps
    for n_frag, nc, H in (
        (2, 14, 4),
        (2, 100, 4),
        (2, 256, 4),
        (3, 20, 4),
        (3, 4, 4),
        (3, 128, 4),
        (2, 2, 4),
        (2, 1, 4),
        (2, 79, 1),
        (3, 16, 1),
        (2, 1, 1),
        (3, 256, 1),
    ):
        assert (
            kp.kcp_plan(n_frag, nc, H, SM, p_rule="wave_fill1_p1tie").P
            == kp.kcp_plan(n_frag, nc, H, SM, p_rule="wave_fill1").P
        ), (n_frag, nc, H)
    # 148 SMs (GB200): the same move; SM = 1: no occupancy pressure, base P = 1 already
    assert kp.kcp_plan(2, 122, 64, 148, p_rule="wave_fill1_p1tie").P == 1
    assert (
        kp.kcp_plan(2, 122, 64, 1, p_rule="wave_fill1_p1tie").P
        == kp.kcp_plan(2, 122, 64, 1, p_rule="wave_fill1").P
        == 1
    )
    # the plan terms are reported alike; the rule name travels
    a = kp.kcp_plan(2, 111, 64, SM, p_rule="wave_fill1_p1tie")
    b = kp.kcp_plan(2, 111, 64, SM, p_rule="wave_fill1")
    assert (a.P, a.NSEG, a.P_occ, a.P_chain, a.P_bal, a.n_frag, a.NC_max_frag) == (
        1,
        2,
        b.P_occ,
        b.P_chain,
        b.P_bal,
        2,
        111,
    )
    assert (a.p_rule, b.P, b.NSEG, b.p_rule) == ("wave_fill1_p1tie", 2, 4, "wave_fill1")
    # nc_frags is validated for every rule; the other rules ignore the counts
    for rule in kp.P_RULES:
        with_counts = kp.kcp_plan(2, 111, 64, SM, p_rule=rule, nc_frags=[111, 111]).P
        plain = kp.kcp_plan(2, 111, 64, SM, p_rule=rule).P
        assert with_counts == plain, rule
        assert (
            plain == kp.kcp_plan(2, 111, 64, SM, p_rule=rule, nc_frags=[111, 50]).P
            or rule == "wave_fill1_p1tie"
        )
        for bad in ([111], [111, 112], [111, -1], [111, 50, 10]):
            with pytest.raises(ValueError, match="nc_frags"):
                kp.kcp_plan(2, 111, 64, SM, p_rule=rule, nc_frags=bad)
    assert kp.kcp_plan(0, 0, 64, SM, p_rule="wave_fill1_p1tie").P == 1
    assert kp.kcp_plan(2, 0, 64, SM, p_rule="wave_fill1_p1tie").P == 1
    assert kp.kcp_plan(2, 111, 64, SM).P == 1  # the shipped default
    assert kp.kcp_plan(2, 111, 64, SM, nc_frags=[111, 111]).P == 1
    assert kp.kcp_plan(2, 111, 64, SM, p_rule="wave_fill1").P == 2
    assert kp.kcp_plan(2, 111, 64, SM, p_rule="wave_fill1", nc_frags=[111, 111]).P == 2
    with pytest.raises(ValueError, match="p_rule"):
        kp.kcp_plan(2, 111, 64, SM, p_rule="p1tie")
    _evidence("plan_p1tie_examples", {"pass": True})


def _two_doc_two_rank_partition(chunks: int = 100):
    """Two documents of two equal fragments across two ranks: every rank holds a document-start (crossing-out) fragment
    and a document-last (crossing-in) fragment of ``chunks`` chunks -- the two-fragment Z rank shape of the tie rule."""
    n = chunks * 64
    cu = torch.tensor([0, 2 * n, 4 * n], dtype=torch.int64)
    return kp.kcp_partition(
        cu, 2, fragments=[[(0, 0, n), (1, n, n)], [(0, n, n), (1, 0, n)]]
    )


def test_p1tie_knob_is_gated_and_plans(monkeypatch) -> None:
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.validate_test_knobs({"p_rule": "wave_fill1_p1tie"})
    part = _two_doc_two_rank_partition(100)
    ctx0 = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
    assert (ctx0.p_rule, ctx0.heads(64).P, ctx0.heads(64).NSEG) == (
        "wave_fill1_p1tie",
        1,
        2,
    )
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    ctx_fill1 = kp.build_kcp_context(
        part,
        None,
        0,
        "cpu",
        4,
        sm_count=SM_COUNT,
        _kcp_test_knobs={"p_rule": "wave_fill1"},
    )
    assert (ctx_fill1.p_rule, ctx_fill1.heads(64).P, ctx_fill1.heads(64).NSEG) == (
        "wave_fill1",
        2,
        4,
    )
    assert kp.validate_test_knobs({"p_rule": "wave_fill1_p1tie"}) == {
        "precision_mode": 0,
        "p_rule": "wave_fill1_p1tie",
    }
    for r in range(2):
        ctx = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1_p1tie"},
        )
        assert (ctx.p_rule, ctx.heads(64).P, ctx.heads(64).NSEG) == (
            "wave_fill1_p1tie",
            1,
            2,
        )
        assert ctx.describe(64)["p_rule"] == "wave_fill1_p1tie"
        assert ctx.heads(4).P == ctx0.heads(4).P  # H = 4: unchanged
    # an unequal pair (170 + 59 chunks on one rank) keeps P = 2 under the knob
    n1, n2 = 170 * 64, 59 * 64
    part_u = kp.kcp_partition(
        torch.tensor([0, 2 * n1, 2 * n1 + 2 * n2], dtype=torch.int64),
        2,
        fragments=[[(0, 0, n1), (1, n2, n2)], [(0, n1, n1), (1, 0, n2)]],
    )
    ctx_u = kp.build_kcp_context(
        part_u,
        None,
        0,
        "cpu",
        4,
        sm_count=SM_COUNT,
        _kcp_test_knobs={"p_rule": "wave_fill1_p1tie"},
    )
    assert ctx_u.heads(64).P == 2


_INV_WALKS = ("offset", "all", "live", "in", "m_b1", "start", "end")


def _inventory(ctx, H: int) -> dict:
    """Inventory of one rank's tables under ``ctx.p_rule`` (CPU): P / NSEG, the K4 chain families (chains,
    elements), the walk lists (segments, CTAs = segments x H, LPT step-times at one CTA per SM, longest segment) and
    whether the fused carry launches save a launch (both families non-empty)."""
    tb = cake_kcp._Tables(ctx, H)
    seg_table = [int(x) for x in tb.hp.seg_chunk_start]

    def walk(lst) -> dict:
        steps = [seg_table[s + 1] - seg_table[s] for s in lst]
        groups: dict[int, int] = {}
        for st in steps:
            if st > 0:
                groups[st] = groups.get(st, 0) + H
        return {
            "segs": len(lst),
            "ctas": len(lst) * H,
            "lpt_steps": kp.lpt_makespan(list(groups.items()), ctx.sm_count)
            if groups
            else 0,
            "max_steps": max(steps, default=0),
        }

    chains = {k: int(c.n_chains) for k, c in tb.chains.items()}
    return {
        "P": int(tb.P),
        "NSEG": int(tb.NSEG),
        "n_in": int(tb.n_in),
        "n_out": int(tb.n_out),
        "n_frag": int(tb.n_frag),
        "chains": chains,
        "elems": {k: int(c.n_elems) for k, c in tb.chains.items()},
        "walks": {
            "offset": walk(tb.walk_offset),
            "all": walk(tb.walk_all),
            "live": walk(tb.walk_live),
            "in": walk(tb.walk_in),
            "m_b1": walk(tb.walk_m_b1),
            "start": walk(tb.walk_start),
            "end": walk(tb.walk_end),
        },
        "transpose_tiles": int(sum(len(row) for row in tb.mf_tiles)),
        "fuse_saves_fwd": int(chains["carry_start"] > 0 and chains["carry_in"] > 0),
        "fuse_saves_bwd": int(chains["dcarry_end"] > 0 and chains["dcarry_cross"] > 0),
    }


# the rows of :func:`_kb_rows` on which the P = 1 tie rule changes at least one rank's P at 152 SMs (the enumeration
# of record of the rule's adoption): the two-fragment Z ranks of the perf packs at CP 8 / 16 and four coverage rows, all
# H = 64
P1TIE_CHANGED_ROWS = frozenset(
    [(f"Q{i}", W, "Z", 64) for i in range(1, 7) for W in (8, 16)]
    + [("C2", 4, "D", 64), ("C2", 4, "Z", 64), ("C6", 8, "Z", 64), ("C8", 8, "Z", 64)]
)


@HARNESS
@pytest.mark.parametrize(
    "case", KB_ROWS, ids=[f"{n}-cp{W}-{lay}-H{H}" for n, W, lay, H in KB_ROWS]
)
def test_kcp_plan_p1tie_lane_rows(monkeypatch, case) -> None:
    """Plan enumeration of the P = 1 tie rule: per rank of every row of :func:`_kb_rows` the P under the previous rule
    "wave_fill1" (named explicitly: the default is the tie rule itself) and under the shipped default "wave_fill1_p1tie"
    (with the actual fragment chunk counts), the P = 1 / base-P walk makespans of the tie test, and -- for the H = 64 rows
    at CP 8 / 16 -- the K4 / walk inventory of both programs (the pre-estimate reads it).  The module equals the oracle, the
    rule changes only two- / three-fragment ranks below the cap whose base P exceeds 1, never an H <= 4 rank, the rows it
    changes are exactly :data:`P1TIE_CHANGED_ROWS`, and a context built without a knob plans with it.  Evidence record
    `plan_p1tie_rows`."""
    name, W, lay, H = case
    row = kcp_shapes.ROWS[name]
    _frags, part = harness_partition(row, W, lay)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    assert (
        kp.P_RULE_DEFAULT == "wave_fill1_p1tie"
        and kp.P1_TIE_BASE[kp.P_RULE_DEFAULT] == "wave_fill1"
    )
    want_inv = H == 64 and W >= 8
    ranks = []
    for r in range(W):
        mine = part.fragments[r]
        actual = [-(-f.length // 64) for f in mine]
        n_frag, nc = len(mine), max(actual, default=0)
        base = kp.kcp_plan(
            n_frag, nc, H, SM_COUNT, p_rule="wave_fill1", nc_frags=actual
        ).P
        cand = kp.kcp_plan(n_frag, nc, H, SM_COUNT, nc_frags=actual).P
        assert (
            cand
            == kp.kcp_plan(
                n_frag, nc, H, SM_COUNT, p_rule="wave_fill1_p1tie", nc_frags=actual
            ).P
        ), (case, r)
        rec = {
            "rank": r,
            "nc_frags": actual,
            "n_frag": n_frag,
            "nc_max": nc,
            "frags": [
                (f.doc, f.off, f.length, bool(f.crossing_in), bool(f.crossing_out))
                for f in mine
            ],
            "wave_fill1": base,
            "wave_fill1_p1tie": cand,
            "steps_p1": kp.p1_tie_steps(actual, H, SM_COUNT, 1) if nc else 0,
            "steps_base": kp.p1_tie_steps(actual, H, SM_COUNT, base) if nc else 0,
            "sm148": kp.kcp_plan(
                n_frag, nc, H, 148, p_rule="wave_fill1_p1tie", nc_frags=actual
            ).P,
        }
        assert cand == _p1tie_ref(n_frag, actual, H, SM_COUNT), (case, rec)
        if cand != base:
            assert cand == 1 and base > 1 and 2 <= n_frag <= 3 and 0 < nc < 257, (
                case,
                rec,
            )
            assert rec["steps_p1"] <= rec["steps_base"], (case, rec)
        if H <= 4:
            assert cand == base, (case, rec)
        if want_inv:
            ctx_b = kp.build_kcp_context(
                part,
                None,
                r,
                "cpu",
                4,
                sm_count=SM_COUNT,
                _kcp_test_knobs={"p_rule": "wave_fill1"},
            )
            ctx_c = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
            inv_b, inv_c = _inventory(ctx_b, H), _inventory(ctx_c, H)
            assert (inv_b["P"], inv_c["P"]) == (base, cand), (case, rec)
            if cand == base:
                assert inv_b == inv_c
            rec["inv_wave_fill1"], rec["inv_p1tie"] = inv_b, inv_c
        ranks.append(rec)
    changed = [x["rank"] for x in ranks if x["wave_fill1_p1tie"] != x["wave_fill1"]]
    assert bool(changed) == ((name, W, lay, H) in P1TIE_CHANGED_ROWS), (case, changed)
    _evidence(
        "plan_p1tie_rows",
        {
            "row": name,
            "cp": W,
            "layout": lay,
            "H": H,
            "sm": SM_COUNT,
            "wave_fill1": [x["wave_fill1"] for x in ranks],
            "wave_fill1_p1tie": [x["wave_fill1_p1tie"] for x in ranks],
            "changed": changed,
            "inventory": want_inv,
            "ranks": ranks,
        },
    )
    for r in changed[:1] or [0]:
        ctx = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        assert (ctx.p_rule, ctx.heads(H).P, ctx.describe(H)["p_rule"]) == (
            "wave_fill1_p1tie",
            ranks[r]["wave_fill1_p1tie"],
            "wave_fill1_p1tie",
        )


def test_merge_fuse_knob_and_fused_carry_chains(monkeypatch) -> None:
    """``merge_fuse`` is a gated test knob ("none" shipped | "carry"); the fused chain families
    carry_both / dcarry_both are the concatenation of carry_start + carry_in / dcarry_end + dcarry_cross (same pools; every
    chain keeps its elements, initial state and final slot), non-empty at P > 1 on a rank holding both fragment kinds, empty
    at P = 1; the default options and the other families are untouched."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert cake_kcp.MERGE_FUSES == ("none", "carry")
    assert cake_kcp.MERGE_FUSE_DEFAULT == "none"
    assert cake_kcp.P2_KNOBS["merge_fuse"] == cake_kcp.MERGE_FUSES
    assert cake_kcp._options({}).merge_fuse == "none"
    assert "merge_fuse" not in cake_kcp._OPTION_KEYS
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"merge_fuse": "carry"}})
    with pytest.raises(TypeError):
        cake_kcp._options({"merge_fuse": "carry"})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    o = cake_kcp._options({"_kcp_test_knobs": {"merge_fuse": "carry"}})
    assert (o.merge_fuse, o.schedule, o.fused_walks, o.cols_rule) == (
        "carry",
        "wait_first",
        1,
        "fixed",
    )
    o = cake_kcp._options(
        {
            "_kcp_test_knobs": {
                "merge_fuse": "carry",
                "cols_rule": "by_chains",
                "p_rule": "wave_fill1_p1tie",
            }
        }
    )
    assert (o.merge_fuse, o.cols_rule, o.p_rule) == (
        "carry",
        "by_chains",
        "wave_fill1_p1tie",
    )
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"merge_fuse": "none"}}).merge_fuse
        == "none"
    )
    with pytest.raises(ValueError, match="merge_fuse"):
        cake_kcp._options({"_kcp_test_knobs": {"merge_fuse": "all"}})
    spec = cake_kcp._Tables.FAMILIES
    assert (
        spec["carry_both"] == {**spec["carry_start"], "x0": "S_in"} == spec["carry_in"]
    )
    assert (
        spec["dcarry_both"]
        == {**spec["dcarry_end"], "x0": "b2_recv"}
        == spec["dcarry_cross"]
    )
    part = _two_doc_two_rank_partition(100)
    for r in range(2):
        ctx = kp.build_kcp_context(
            part,
            None,
            r,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )  # P = 2 on the tie shape: the shipped tie rule plans P = 1 here
        tb = cake_kcp._Tables(ctx, 64)
        assert tb.P == 2 and (tb.n_in, tb.n_out) == (1, 1)
        for both, a, b in (
            ("carry_both", "carry_start", "carry_in"),
            ("dcarry_both", "dcarry_end", "dcarry_cross"),
        ):
            ca, cb, cf = tb.chains[a], tb.chains[b], tb.chains[both]
            assert (ca.n_chains, cb.n_chains, cf.n_chains) == (64, 64, 128)
            assert cf.ncols == ca.ncols == cb.ncols == cake_kcp.D
            assert cf.elem_m == ca.elem_m + cb.elem_m
            assert cf.elem_c == ca.elem_c + cb.elem_c
            assert cf.elem_y == ca.elem_y + cb.elem_y
            assert cf.x0_index == ca.x0_index + cb.x0_index
            assert cf.y_final == ca.y_final + cb.y_final
            assert cf.chain_start == ca.chain_start + [
                ca.n_elems + s for s in cb.chain_start[1:]
            ]
            assert all(x == -1 for x in ca.x0_index) and all(
                x >= 0 for x in cb.x0_index
            )
            cf.check_alignment()
            tb._check_chain(both, cf)
        d = tb.describe()["chains"]
        assert d["carry_both"] == {
            "n_chains": 128,
            "n_elems": d["carry_start"]["n_elems"] + d["carry_in"]["n_elems"],
        }
        assert d["dcarry_both"] == {
            "n_chains": 128,
            "n_elems": d["dcarry_end"]["n_elems"] + d["dcarry_cross"]["n_elems"],
        }
        inv = _inventory(ctx, 64)
        assert (inv["fuse_saves_fwd"], inv["fuse_saves_bwd"]) == (1, 1)
    e = cake_kcp.Chains.concat(cake_kcp.Chains(), cake_kcp.Chains())
    assert (e.n_chains, e.n_elems, e.chain_start) == (0, 0, [0])
    with pytest.raises(ValueError, match="ncols"):
        cake_kcp.Chains.concat(cake_kcp.Chains(128), cake_kcp.Chains(256))
    one = cake_kcp.Chains()
    one.add([(0, -1, -1), (16384, 16384, -1)], x0=-1, y_final=32768)
    two = cake_kcp.Chains.concat(one, one)
    assert (
        two.chain_start == [0, 2, 4]
        and two.x0_index == [-1, -1]
        and two.y_final == [32768, 32768]
    )
    # P = 1 (an eight-fragment D rank): the fused families are empty like their parts
    tb1 = cake_kcp._Tables(
        kp.build_kcp_context(_small_partition(4), None, 1, "cpu", 4, sm_count=SM_COUNT),
        64,
    )
    assert tb1.P == 1
    assert all(
        tb1.chains[k].n_chains == 0
        for k in (
            "carry_start",
            "carry_in",
            "carry_both",
            "dcarry_end",
            "dcarry_cross",
            "dcarry_both",
        )
    )
    inv1 = _inventory(
        kp.build_kcp_context(_small_partition(4), None, 1, "cpu", 4, sm_count=SM_COUNT),
        64,
    )
    assert (inv1["fuse_saves_fwd"], inv1["fuse_saves_bwd"]) == (0, 0)


def test_limits_are_rejected_before_launch() -> None:
    """Gate 3: n_frag > 65536 and int32 overflows are refused by the metadata layer."""
    W = 2
    many = [(0, i, 1, 0) for i in range(kp.MAX_FRAGS + 1)] + [
        (0, kp.MAX_FRAGS + 1, 1, 1)
    ]
    part = kp.kcp_partition(
        torch.tensor([0, kp.MAX_FRAGS + 2], dtype=torch.int64), W, fragments=many
    )
    with pytest.raises(ValueError, match="at most 65536"):
        kp.build_kcp_context(part, None, 0, "cpu", sm_count=SM_COUNT)
    big = kp.kcp_partition(
        torch.tensor([0, 2**25], dtype=torch.int64), 2, layout="contiguous"
    )
    ctx = kp.build_kcp_context(big, None, 0, "cpu", sm_count=SM_COUNT)
    with pytest.raises(ValueError, match="int32"):
        ctx.heads(64)  # R * H * 64 >= 2**31
    with pytest.raises(ValueError):
        kp.build_kcp_context(part, None, 2, "cpu")
    # [1 | 1 | 1 | long]: head row 0 of the long fragment has THREE consumers (the fragments ending at 3, 2, 1) -- the
    # C3 row of record at layout D (a 5-token document in 1-token pieces) has this pattern: three slots per head row
    p3 = kp.kcp_partition(
        torch.tensor([0, 500], dtype=torch.int64),
        4,
        fragments=[[(0, 0, 1)], [(0, 1, 1)], [(0, 2, 1)], [(0, 3, 497)]],
    )
    ctx3 = kp.build_kcp_context(p3, None, 3, "cpu", sm_count=SM_COUNT)
    assert (
        ctx3.halo.head_row_off[0] == ((0, 1, 3), (2, 4, -1), (5, -1, -1))
        and ctx3.halo.n_head_rows == 6
    )
    assert tuple(ctx3.halo_tables["head_row_off"].shape) == (1, 3, 3)
    assert kp.MAX_HEAD_CONSUMERS == 3
    # [1 | 2 | long] (the C7 pattern) has two consumers per head row and is accepted
    p2 = kp.kcp_partition(
        torch.tensor([0, 500], dtype=torch.int64),
        3,
        fragments=[[(0, 0, 1)], [(0, 1, 2)], [(0, 3, 497)]],
    )
    ctx2 = kp.build_kcp_context(p2, None, 2, "cpu", sm_count=SM_COUNT)
    assert (
        ctx2.halo.head_row_off[0][0] == (0, 1, -1)
        and ctx2.halo.head_row_off[0][1] == (2, -1, -1)
        and ctx2.halo.head_row_off[0][2] == (3, -1, -1)
    )


def test_partition_validation_and_defaults() -> None:
    """kcp_partition: explicit tables are validated; layouts cut as documented; zero-length fragments are skipped by
    the chains; the default zigzag_per_document cut is near-equal."""
    cu = torch.tensor(
        [0, 100, 100, 350], dtype=torch.int64
    )  # a zero-length document in the middle
    part = kp.kcp_partition(cu, 2, layout="contiguous")
    assert [[(f.doc, f.off, f.length) for f in frs] for frs in part.fragments] == [
        [(0, 0, 100), (2, 0, 75)],
        [(2, 75, 175)],
    ]
    assert (
        part.fragments[0][1].next == (1, 0)
        and part.fragments[1][0].prev == (0, 1)
        and part.fragments[0][0].next is None
    )
    z = kp.kcp_partition(cu, 2, layout="zigzag")
    assert [[(f.doc, f.off, f.length) for f in frs] for frs in z.fragments] == [
        [(0, 0, 88), (2, 163, 87)],
        [(0, 88, 12), (2, 0, 76), (2, 76, 87)],
    ]
    d = kp.kcp_partition(cu, 2, layout="zigzag_per_document")
    assert [(f.doc, f.off, f.length) for f in d.fragments[0]] == [
        (0, 0, 25),
        (0, 75, 25),
        (1, 0, 0),
        (1, 0, 0),
        (2, 0, 63),
        (2, 188, 62),
    ]
    assert [(f.doc, f.off, f.length) for f in d.fragments[1]] == [
        (0, 25, 25),
        (0, 50, 25),
        (1, 0, 0),
        (1, 0, 0),
        (2, 63, 63),
        (2, 126, 62),
    ]
    assert (
        d.fragments[0][2].prev is None and d.fragments[0][2].next is None
    )  # zero-length: not in the chain
    assert d.chains()[2] == ((0, 4), (1, 4), (1, 5), (0, 5))
    with_cuts = kp.kcp_partition(
        cu,
        2,
        layout="zigzag_per_document",
        cut_points=[[0, 1, 2, 3, 100], [0, 0, 0, 0, 0], [0, 10, 10, 250, 250]],
    )
    assert [(f.doc, f.off, f.length) for f in with_cuts.fragments[1]] == [
        (0, 1, 1),
        (0, 2, 1),
        (1, 0, 0),
        (1, 0, 0),
        (2, 10, 0),
        (2, 10, 240),
    ]
    assert with_cuts.fragments[1][4].prev is None and with_cuts.fragments[1][
        5
    ].prev == (0, 4)
    with pytest.raises(ValueError):
        kp.kcp_partition(
            cu, 2, layout="zigzag_per_document", cut_points=[[0, 50, 100]] * 3
        )
    with pytest.raises(ValueError, match="covered"):
        kp.kcp_partition(
            torch.tensor([0, 10], dtype=torch.int64),
            2,
            fragments=[[(0, 0, 4)], [(0, 5, 5)]],
        )
    with pytest.raises(ValueError, match="not covered|covered up to|gap"):
        kp.kcp_partition(
            torch.tensor([0, 10], dtype=torch.int64),
            2,
            fragments=[[(0, 0, 6)], [(0, 5, 5)]],
        )
    with pytest.raises(ValueError, match="order"):
        kp.kcp_partition(
            torch.tensor([0, 10], dtype=torch.int64),
            1,
            fragments=[[(0, 5, 5), (0, 0, 5)]],
        )
    with pytest.raises(ValueError):
        kp.kcp_partition(
            torch.tensor([0, 10], dtype=torch.int64), 2, fragments=[(0, 0, 10, 2)]
        )
    with pytest.raises(ValueError):
        kp.kcp_partition(torch.tensor([0, 10], dtype=torch.int64), 2, layout="ring")
    with pytest.raises(ValueError, match="CPU"):
        kp.kcp_partition(torch.tensor([0, 10], dtype=torch.int64, device="meta"), 2)
    flat = kp.kcp_partition(
        torch.tensor([0, 10], dtype=torch.int64),
        2,
        fragments=[(0, 0, 4, 1), (0, 4, 6, 0)],
    )
    assert (
        flat.layout == "explicit"
        and flat.records(0) == ((0, 4, 6, (1, 0), None),)
        and flat.records(1) == ((0, 0, 4, None, (0, 0)),)
    )
    assert kp.kcp_partition([0, 10], 1).fragments == (
        (kp.KcpFragment(0, 0, 10, 0, 0),),
    )


def test_tables_follow_the_kernel_contract() -> None:
    """chunk_bos / chunk_len / seq_chunk_start / seq_order / seg_chunk_start of the spec formulas; the four chunk tables
    are laid out as the existing backend lays them out, and the boundary kernels' ``[NSEG + 1]`` segment table follows
    them in the same 256 B-aligned form.  The existing backend's own fifth table is a different object since its segmented
    recurrences took a block table (``seg_table``, ``NSEG + 2 * N + 1`` words): the two layouts agree on the first four
    entries, and the KCP's fifth entry keeps the width the K1-K4 ``seq_chunk_start`` argument has."""
    cu = (0, 1, 1, 66, 130, 130, 193)  # 1 token, empty, 65, 64, empty, 63
    t = kp.chunk_tables(cu)
    assert t["seq_chunk_start"] == [0, 1, 1, 3, 4, 4, 5]
    nc_max, nc_pad, R = kp.chunk_geometry(193, 6)
    assert (nc_max, nc_pad, R) == (4 + 6, 10, 640) and len(t["chunk_bos"]) == nc_pad
    assert t["chunk_bos"][:5] == [0, 1, 65, 66, 130] and t["chunk_bos"][5:] == [0] * 5
    assert t["chunk_len"][:5] == [1, 64, 1, 64, 63] and t["chunk_len"][5:] == [0] * 5
    assert t["seq_order"] == [
        2,
        0,
        3,
        5,
        1,
        4,
    ]  # counts 1,0,2,1,0,1: by count desc, index asc
    seg = kp.seg_chunk_start(t["seq_chunk_start"], 3)
    assert (
        seg[:6] == [0, 0, 0, 1, 1, 1]
        and seg[6:9] == [1, 1, 2]
        and seg[-1] == 5
        and len(seg) == 19
    )
    assert (
        seg[9:12] == [3, 3, 3] and seg[12:15] == [4, 4, 4] and seg[15:18] == [4, 4, 4]
    )
    layout = kp.table_layout(nc_pad, 6, 18)
    assert layout[:4] == cake_backend._table_layout(nc_pad, 6, 0)
    assert layout[4] == (
        "seg_chunk_start",
        256,
        19,
    )  # NSEG + 1 words in the slot after seq_order
    assert cake_backend.TABLE_NAMES[:4] == kp.TABLE_NAMES[:4]
    assert (
        cake_backend._table_layout(nc_pad, 6, 18)[4][0] != kp.TABLE_NAMES[4]
    )  # the backend's block table: not this table
    assert (
        kp.tables_bytes(nc_pad, 6, 18) == 4 * 64 * 5
    )  # five 256 B-aligned tables of < 64 words each
    assert (
        kp.table_words(0, 0, 0) == 128
    )  # the two one-word tables each occupy a 256 B slot


def test_byte_queries_and_knobs(monkeypatch) -> None:
    """Gate 4: keyword-only signatures; the KCP contract §3.2 literal forms work with defaults; save_transfer and P > 1 terms
    per §9; validate_test_knobs refuses a non-zero precision_mode without the environment variable."""
    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctxs = contexts(part)
    ctx = ctxs[1]
    hp = ctx.heads(64)
    s = 64 * kp.STATE_TILE_BYTES
    assert kp.SAVE_TRANSFER_DEFAULT is True  # shipped default
    assert kp.kcp_saved_bytes(ctx) == kp.kcp_saved_bytes(ctx, save_transfer=True)
    assert (
        hp.P > 1
    )  # 3000 tokens in one fragment at 152 SMs: P_occ = ceil(152 / 64) = 3
    plane = (
        (ctx.n_frag * (hp.P - 1) + max(ctx.n_in, ctx.n_out)) * s
    )  # the whole carry plane of the P > 1 rank (fused walks: one extra slot per crossing fragment)
    base = kp.kcp_saved_bytes(
        ctx, save_transfer=False
    )  # literal form: H = 64, light, no transfer, conv, C = 8192
    assert (
        base
        == kp.tables_bytes(ctx.NC_pad, ctx.n_frag, hp.NSEG)
        + ctx.n_in * s
        + plane
        + ctx.halo.n_hist_rows * 8192 * 2
    )  # no token-row Aqk / Akk term: the saved set carries none; the carry plane is part of the shipped set
    assert cake_kcp.KCP_SAVED_VERSION == 2
    fields = cake_kcp.KcpSaved.__dataclass_fields__
    assert "Aqk" not in fields and "Akk" not in fields
    assert cake_kcp.KcpSaved._OPTIONAL == ("c1_recv", "M_seg", "carry")
    assert cake_kcp.KcpSaved.__dataclass_fields__["carry"].default is None
    assert kp.kcp_saved_bytes(ctx) == kp.kcp_saved_bytes(
        ctx, saved_carry=True
    )  # the shipped default counts the carry plane
    assert (
        kp.kcp_saved_bytes(ctx) - kp.kcp_saved_bytes(ctx, saved_carry=False) == plane
    )  # saved_carry=False = the re-walk form of the test knob: no plane term
    for c in [*contexts(_small_partition(4)), *ctxs]:
        p_c = c.heads(64).P
        assert kp.kcp_saved_bytes(c) - kp.kcp_saved_bytes(c, saved_carry=False) == (
            (c.n_frag * (p_c - 1) + max(c.n_in, c.n_out)) * s if p_c > 1 else 0
        )  # the plane term exists on P > 1 ranks only
    assert (
        kp.kcp_saved_bytes(ctx, save_transfer=False, with_conv=False)
        == base - ctx.halo.n_hist_rows * 8192 * 2
    )
    assert kp.kcp_saved_bytes(
        ctx, save_transfer=True
    ) - base == ctx.N_out * 2 * hp.Hg_r * kp.STATE_TILE_BYTES + (
        hp.NSEG * s if hp.P > 1 else 0
    )
    assert (
        kp.kcp_saved_bytes(ctx, save_mode="tape", save_transfer=False) - base
        == ctx.NC_pad * 64 * 32768 + ctx.R * 64 * 256
    )
    with pytest.raises(TypeError):
        kp.kcp_saved_bytes(ctx, 64)  # positional H refused
    with pytest.raises(TypeError):
        kp.kcp_comm_bytes(ctx, 64)
    with pytest.raises(TypeError):
        kp.kcp_workspace_bytes(ctx, 64)
    assert kp.kcp_comm_bytes(ctx) == kp.kcp_comm_bytes(ctx, save_transfer=True)
    cb = kp.kcp_comm_bytes(ctx, save_transfer=False)
    assert (
        cb["H"] == 64
        and cb["C"] == 8192
        and cb["c1"]["send"] == ctx.n_out * 2 * s
        and cb["c2"]["recv"] == ctx.n_in * s
    )
    assert cb["b1"]["send"] == ctx.n_in * 2 * s  # saved mode: pairs [d_f | M_f^T]
    assert cb["padding"]["B1"] == {"send": 0, "recv": 0} and cb["padding"]["B2"] == {
        "send": 0,
        "recv": 0,
    }  # H = 64: no padding
    assert (
        cb["padding"]["pad_tiles_per_slot"] == 0 and cb["padding"]["padded_owner"] == -1
    )
    assert (
        hp.tile_stride["c1"] == kp.TILE_STRIDE_PAIR == 2 and hp.tile_stride["c2"] == 1
    )
    assert kp.kcp_comm_bytes(ctx, save_transfer=True)["b1"]["send"] == ctx.n_in * s
    assert kp.kcp_comm_bytes(ctx, boundary_mode="replay")["b1"]["send"] == ctx.n_in * s
    assert (
        cb["B3"]["send"]
        == cb["B1"]["send"]
        + cb["b1_replay"]["send"]
        + cb["b2"]["send"]
        + cb["halo_bwd"]["send"]
    )
    ws_t = kp.kcp_workspace_bytes(ctx, save_transfer=True)
    assert kp.kcp_workspace_bytes(ctx) == ws_t
    ws = kp.kcp_workspace_bytes(ctx, save_transfer=False)
    local = cake_backend.workspace_bytes(
        ctx.T_r, ctx.n_frag, 64, backward=True, save_mode="light", sm_count=SM_COUNT
    )
    # the pools do not depend on save_transfer / C: the kept c1 tensor and the halo rows are per-call tensors (SPEC-4)
    arena = kp.kcp_arena_layout(ctx)
    assert (
        arena
        == kp.kcp_arena_layout(ctx, save_transfer=False)
        == kp.kcp_arena_layout(ctx, C=4096)
    )
    assert ws == ws_t == local + arena["arena"]
    assert arena["pool.c1_recv"] == hp.numel["c1_recv"] * 4
    assert (
        arena["pool.stage_b"] + arena["pool.stage_m"]
        == hp.NSEG * 64 * kp.SUMMARY_TILE * 4
    )
    # one storage per SHARED_POOLS pair (b1_send = c1_send, b2_send = c2_send); the ring pools on first use
    assert kp.SHARED_POOLS == (("c1_send", "b1_send"), ("c2_send", "b2_send"))
    assert kp.LAZY_POOLS == ("ring_out", "ring_in")
    alone = ("c1_recv", "stage_b", "stage_m", "carry", "eye", "b1_recv", "b2_recv")
    assert arena["arena"] == sum(arena[f"pool.{n}"] for n in alone) + sum(
        max(arena[f"pool.{a}"], arena[f"pool.{b}"]) for a, b in kp.SHARED_POOLS
    )
    assert arena["ring"] == arena["pool.ring_out"] + arena["pool.ring_in"] > 0
    fwd = kp.kcp_arena_layout(ctx, backward=False)
    assert "pool.b1_send" not in fwd and fwd["arena"] < arena["arena"]
    assert kp.kcp_workspace_bytes(ctx, backward=False) < ws
    # knobs
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert kp.validate_test_knobs(None) == {
        "precision_mode": 0,
        "p_rule": "wave_fill1_p1tie",
    }
    assert kp.validate_test_knobs(None)["p_rule"] == kp.P_RULE_DEFAULT
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.validate_test_knobs({"precision_mode": 2})
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.validate_test_knobs({"precision_mode": 0})
    with pytest.raises(ValueError, match="enable_test_knobs"):
        kp.validate_test_knobs({"p_rule": "wave"})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    assert kp.validate_test_knobs({"precision_mode": 1}) == {
        "precision_mode": 1,
        "p_rule": "wave_fill1_p1tie",
    }
    assert kp.validate_test_knobs({}) == {
        "precision_mode": 0,
        "p_rule": "wave_fill1_p1tie",
    }
    assert kp.validate_test_knobs({"p_rule": "occ"}) == {
        "precision_mode": 0,
        "p_rule": "occ",
    }
    assert kp.validate_test_knobs({"p_rule": "wave"}) == {
        "precision_mode": 0,
        "p_rule": "wave",
    }
    with pytest.raises(ValueError, match="p_rule"):
        kp.validate_test_knobs({"p_rule": "lpt"})
    with pytest.raises(ValueError, match="precision_mode"):
        kp.validate_test_knobs({"precision_mode": 3})
    with pytest.raises(ValueError, match="unknown"):
        kp.validate_test_knobs({"bf16x3": True})
    with pytest.raises(TypeError):
        kp.validate_test_knobs([2])
    assert (
        ctx.sequence.next("c1") == 1
        and ctx.sequence.next("c2") == 2
        and ctx.sequence.log == ["c1", "c2"]
    )
    ctx.sequence.reset()
    assert ctx.sequence.count == 0
    assert set(ctx.describe(64)) >= {"rank", "T_r", "halo", "heads"}


# --------------------------------------------------------------------------- the schedule / fused-walk / column-rule knobs

cake_kcp = cake_package("cake_kda_train", "cake_kcp")
LEVER_TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")


def test_round1_knobs_are_gated_and_validated(monkeypatch) -> None:
    """``schedule``, ``fused_walks`` and ``cols_rule`` are test knobs: refused
    without :func:`cake_kcp_plan.enable_test_knobs`, never public keywords (``graph`` is the exception, public
    since this revision), validated against their admitted values; the
    shipped defaults since an earlier revision are the pair wait_first + fused_walks=1 (one tape launch per
    phase) with the registry's K4 records (cols_rule fixed); overlap / fused_walks=0 stay selectable."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    o = cake_kcp._options({})
    assert (o.schedule, o.fused_walks, o.cols_rule) == ("wait_first", 1, "fixed")
    assert (
        cake_kcp.SCHEDULE_DEFAULT == "wait_first" and cake_kcp.FUSED_WALKS_DEFAULT == 1
    )
    for knob in (
        {"schedule": "wait_first"},
        {"fused_walks": 1},
        {"cols_rule": "by_chains"},
    ):
        with pytest.raises(ValueError, match="enable_test_knobs"):
            cake_kcp._options({"_kcp_test_knobs": knob})
    for key in cake_kcp.P2_KNOBS:
        if (
            key in cake_kcp._OPTION_KEYS
        ):  # graph is a public keyword -> validated, not refused
            with pytest.raises(ValueError, match=key):
                cake_kcp._options({key: 1})
            continue
        with pytest.raises(TypeError):
            cake_kcp._options({key: 1})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    o = cake_kcp._options(
        {
            "_kcp_test_knobs": {
                "schedule": "wait_first",
                "fused_walks": 1,
                "cols_rule": "by_chains",
            }
        }
    )
    assert (o.schedule, o.fused_walks, o.cols_rule) == ("wait_first", 1, "by_chains")
    o = cake_kcp._options(
        {"_kcp_test_knobs": {"fused_walks": 1, "transfer_walk": "bf16x3"}}
    )
    assert (o.fused_walks, o.transfer_walk, o.precision_mode) == (1, "bf16x3", 0)
    assert cake_kcp._options({"_kcp_test_knobs": {"fused_walks": 0}}).fused_walks == 0
    o = cake_kcp._options(
        {"_kcp_test_knobs": {"schedule": "overlap", "fused_walks": 0}}
    )
    assert (o.schedule, o.fused_walks) == ("overlap", 0)
    for bad in ({"schedule": "eager"}, {"fused_walks": 2}, {"cols_rule": "c32"}):
        with pytest.raises(ValueError):
            cake_kcp._options({"_kcp_test_knobs": bad})
    with pytest.raises(ValueError, match="unknown"):
        cake_kcp._options({"_kcp_test_knobs": {"schedul": "wait_first"}})


def test_ring_exchange_contract() -> None:
    """SPEC-2's ring exchange is explicit only (``auto`` is a2a), needs every document chain within
    ``RING_MAX_HOPS`` hops, and has no form with ``save_transfer=True`` (the ring receives no c1 tensor): the
    documented ``NotImplementedError`` the multi-GPU rows avoid by asking for the recompute form."""
    from types import SimpleNamespace

    near = SimpleNamespace(max_hops=cake_kcp.RING_MAX_HOPS)
    far = SimpleNamespace(max_hops=cake_kcp.RING_MAX_HOPS + 1)
    # the shipped default: an unset save_transfer resolves to True outside the ring
    assert cake_kcp.KcpOptions().check().save_transfer is True
    assert cake_kcp._exchange_scheme(near, cake_kcp.KcpOptions()) == "a2a"
    assert (
        cake_kcp._exchange_scheme(
            near, cake_kcp.KcpOptions(exchange="ring", save_transfer=False)
        )
        == "ring"
    )
    with pytest.raises(NotImplementedError, match="no ring form"):
        cake_kcp._exchange_scheme(
            near, cake_kcp.KcpOptions(exchange="ring", save_transfer=True)
        )
    with pytest.raises(ValueError, match="n_d_max"):
        cake_kcp._exchange_scheme(
            far, cake_kcp.KcpOptions(exchange="ring", save_transfer=False)
        )


def test_host_trim_knob_is_gated_and_validated(monkeypatch) -> None:
    """``host_trim`` is a private test knob: refused without
    :func:`cake_kcp_plan.enable_test_knobs`, never a public keyword, validated against :data:`HOST_TRIMS`; the shipped default is 1
    (the trimmed host path of the eager call), 0 selects the previous host path (the same launches
    in the same order) for the A/B record."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert cake_kcp.HOST_TRIM_DEFAULT == 1 and cake_kcp.HOST_TRIMS == (0, 1)
    assert cake_kcp._options({}).host_trim == 1
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"host_trim": 0}})
    with pytest.raises(TypeError):
        cake_kcp._options({"host_trim": 1})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    assert cake_kcp._options({"_kcp_test_knobs": {"host_trim": 1}}).host_trim == 1
    assert cake_kcp._options({"_kcp_test_knobs": {"host_trim": 0}}).host_trim == 0
    o = cake_kcp._options(
        {
            "_kcp_test_knobs": {
                "host_trim": 1,
                "fused_walks": 1,
                "walk_geometry": "lpt_tape_one_cta_per_sm",
            }
        }
    )
    assert (o.host_trim, o.fused_walks, o.walk_geometry) == (
        1,
        1,
        "lpt_tape_one_cta_per_sm",
    )
    for bad in ({"host_trim": 2}, {"host_trim": -1}):
        with pytest.raises(ValueError):
            cake_kcp._options({"_kcp_test_knobs": bad})
    with pytest.raises(TypeError):
        cake_kcp._options({"host_trim": 0})


def test_saved_carry_knob_is_gated_and_validated(monkeypatch) -> None:
    """``saved_carry`` is a private test knob: refused without :func:`cake_kcp_plan.enable_test_knobs`, never a public keyword,
    validated against :data:`SAVED_CARRIES`; the shipped default is 1 (the forward of a ``P > 1`` rank keeps its
    carry plane in the saved set -- ``KcpSaved.carry``, an optional tensor that travels through ``tensors()`` /
    ``flags()`` / ``from_tensors`` like the other two -- and the saved-mode backward reads it), 0 selects the re-walk
    of the forward carries for the A/B record."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert cake_kcp.SAVED_CARRY_DEFAULT == 1 and cake_kcp.SAVED_CARRIES == (0, 1)
    assert cake_kcp.P2_KNOBS["saved_carry"] == cake_kcp.SAVED_CARRIES
    assert cake_kcp._options({}).saved_carry == 1
    assert cake_kcp.KcpOptions().saved_carry == 1
    assert "saved_carry" not in cake_kcp._OPTION_KEYS
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"saved_carry": 0}})
    with pytest.raises(TypeError):
        cake_kcp._options({"saved_carry": 0})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    assert cake_kcp._options({"_kcp_test_knobs": {"saved_carry": 1}}).saved_carry == 1
    assert cake_kcp._options({"_kcp_test_knobs": {"saved_carry": 0}}).saved_carry == 0
    assert cake_kcp._options({"_kcp_test_knobs": {"saved_carry": "0"}}).saved_carry == 0
    o = cake_kcp._options(
        {"_kcp_test_knobs": {"saved_carry": 0, "host_trim": 1, "fused_walks": 1}}
    )
    assert (o.saved_carry, o.host_trim, o.fused_walks) == (0, 1, 1)
    for bad in ({"saved_carry": 2}, {"saved_carry": -1}):
        with pytest.raises(ValueError):
            cake_kcp._options({"_kcp_test_knobs": bad})
    with pytest.raises(TypeError):
        cake_kcp._options({"saved_carry": 1})
    # the saved set with and without the plane: flags / tensors / from_tensors round trip
    meta = dict(
        version=cake_kcp.KCP_SAVED_VERSION,
        T_r=8,
        H=4,
        n_frag=1,
        P=2,
        Hg=4,
        scale=0.5,
        save_mode="light",
        save_transfer=True,
        offset_class="fla",
        exchange="a2a",
        precision_mode=0,
        transfer_walk="bf16x3_hw64",
    )
    s_in, c1, m_seg, plane = (
        torch.zeros(n, dtype=torch.float32) for n in (3, 5, 7, 11)
    )
    with_plane = cake_kcp.KcpSaved(
        S_in=s_in, c1_recv=c1, M_seg=m_seg, carry=plane, ctx=None, **meta
    )
    without = cake_kcp.KcpSaved(S_in=s_in, c1_recv=c1, M_seg=m_seg, ctx=None, **meta)
    assert without.carry is None and without.flags() == (True, True, False)
    assert with_plane.flags() == (True, True, True)
    assert [t.numel() for t in without.tensors()] == [3, 5, 7]
    assert [t.numel() for t in with_plane.tensors()] == [3, 5, 7, 11]
    assert with_plane.bytes - without.bytes == 11 * 4
    for saved in (without, with_plane):
        back = cake_kcp.KcpSaved.from_tensors(
            saved.tensors(), saved.flags(), saved.meta(), None
        )
        assert back.flags() == saved.flags() and back.meta() == saved.meta()
        assert (back.carry is None) == (saved.carry is None)
        assert back.carry is None or back.carry is plane
    assert (
        "saved_carry" not in with_plane.meta()
    )  # the plane's presence travels as a flag, not as metadata


def test_ring_exchange_resolves_save_transfer() -> None:
    """``exchange="ring"`` has no ``save_transfer=True`` form (the ring path receives no c1 tensor),
    so an unset ``save_transfer`` resolves to ``False`` for the ring and to ``SAVE_TRANSFER_DEFAULT`` (``True``)
    otherwise; an explicit ``False`` is kept everywhere; the byte queries keep their explicit default."""
    assert kp.SAVE_TRANSFER_DEFAULT is True
    assert (
        cake_kcp.KcpOptions.save_transfer is None
    )  # the dataclass default = "resolve"
    assert cake_kcp._options({}).save_transfer is True
    assert cake_kcp._options({"exchange": "auto"}).save_transfer is True
    assert cake_kcp._options({"exchange": "a2a"}).save_transfer is True
    assert cake_kcp._options({"exchange": "ring"}).save_transfer is False
    assert (
        cake_kcp._options({"exchange": "ring", "save_transfer": False}).save_transfer
        is False
    )
    assert cake_kcp._options({"save_transfer": False}).save_transfer is False
    assert cake_kcp._options({"save_transfer": True}).save_transfer is True
    assert cake_kcp.KcpOptions().check().save_transfer is True
    assert cake_kcp.KcpOptions(exchange="ring").check().save_transfer is False
    # the resolved options select the ring scheme on a two-rank partition (one hop) without raising
    part = _small_partition(2, lengths=(3000,), layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=SM_COUNT)
    tb = cake_kcp._Tables(ctx, 4)
    assert tb.max_hops <= cake_kcp.RING_MAX_HOPS
    opts = cake_kcp._options({"exchange": "ring"})
    assert cake_kcp._exchange_scheme(tb, opts) == "ring" and opts.save_transfer is False
    assert cake_kcp._exchange_scheme(tb, cake_kcp._options({})) == "a2a"
    with pytest.raises(
        NotImplementedError, match="ring"
    ):  # the contract guard behind the options level
        cake_kcp._exchange_scheme(
            tb, cake_kcp.KcpOptions(exchange="ring", save_transfer=True)
        )


def test_ring_exchange_with_explicit_save_transfer_true() -> None:
    """An explicit ``save_transfer=True`` together with ``exchange="ring"`` is refused (ValueError) -- the saved set the caller asked for has no ring form."""
    with pytest.raises(ValueError, match="ring"):
        cake_kcp._options({"exchange": "ring", "save_transfer": True})
    with pytest.raises(ValueError, match="ring"):
        cake_kcp.KcpOptions(exchange="ring", save_transfer=True).check()


def test_graph_knob_is_gated_and_validated(monkeypatch) -> None:
    """``graph`` is a public keyword of the entry points since this
    revision (validated against ``GRAPH_MODES``; shipped default ``GRAPH_DEFAULT == "off"``, the eager path)
    and stays reachable through the gated ``_kcp_test_knobs`` for the A/B record (refused without
    :func:`cake_kcp_plan.enable_test_knobs`; a knob value contradicting the keyword is a ``ValueError``).  A call is never
    captured without a caller-owned workspace, under ``exchange="ring"``
    or ``debug_sequence``; the validity key tells operands apart by data pointer, shape, stride and dtype, keeps one
    entry per key, and the first call of a key is eager bookkeeping only."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert cake_kcp.GRAPH_MODES == ("off", "replay")
    assert cake_kcp.P2_KNOBS["graph"] == cake_kcp.GRAPH_MODES
    assert cake_kcp.GRAPH_DEFAULT == "off" and "GRAPH_DEFAULT" in cake_kcp.__all__
    assert cake_kcp._options({}).graph == "off"
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"graph": "replay"}})
    # the public keyword needs no environment gate and is validated like every option
    assert "graph" in cake_kcp._OPTION_KEYS
    assert cake_kcp._options({"graph": "replay"}).graph == "replay"
    assert cake_kcp._options({"graph": "off"}).graph == "off"
    with pytest.raises(ValueError, match="graph"):
        cake_kcp._options({"graph": "capture"})
    with pytest.raises(ValueError, match="graph"):
        cake_kcp.KcpOptions(graph="capture").check()
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    on = cake_kcp._options({"_kcp_test_knobs": {"graph": "replay"}})
    assert on.graph == "replay"
    # the knob form (A/B record) and the keyword may agree, never contradict
    assert (
        cake_kcp._options(
            {"graph": "replay", "_kcp_test_knobs": {"graph": "replay"}}
        ).graph
        == "replay"
    )
    with pytest.raises(ValueError, match="twice"):
        cake_kcp._options({"graph": "off", "_kcp_test_knobs": {"graph": "replay"}})
    assert cake_kcp._options({"_kcp_test_knobs": {"graph": "off"}}).graph == "off"
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"graph": "capture"}})
    # graph_cap (the cache-config form of a captured call) is a gated knob, never a public keyword
    # (plus the kernel-node forms "node_attr" / "node_attr_post"; "node_attr" is
    # GRAPH_CAP_DEFAULT)
    assert cake_kcp.GRAPH_CAPS == (
        "none",
        "capture_pref",
        "node_attr",
        "node_attr_post",
    )
    assert cake_kcp.GRAPH_NODE_CAPS == ("node_attr", "node_attr_post")
    assert cake_kcp.P2_KNOBS["graph_cap"] == cake_kcp.GRAPH_CAPS
    assert cake_kcp.GRAPH_CAP_DEFAULT in cake_kcp.GRAPH_CAPS
    assert cake_kcp.GRAPH_CAP_DEFAULT == "node_attr"
    assert "graph_cap" not in cake_kcp._OPTION_KEYS
    assert cake_kcp._options({}).graph_cap == cake_kcp.GRAPH_CAP_DEFAULT
    for cap in cake_kcp.GRAPH_CAPS:
        o_cap = cake_kcp._options(
            {"graph": "replay", "_kcp_test_knobs": {"graph_cap": cap}}
        )
        assert (o_cap.graph, o_cap.graph_cap) == ("replay", cap)
    with pytest.raises(ValueError, match="graph_cap"):
        cake_kcp._options({"_kcp_test_knobs": {"graph_cap": "always"}})
    with pytest.raises(TypeError):
        cake_kcp._options({"graph_cap": "none"})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"graph_cap": "none"}})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    ctx = object()
    t = [torch.empty(4, 2, 128, dtype=torch.bfloat16) for _ in range(5)]
    flags = (True, "light")
    for (
        bad_call
    ) in (  # the keyword is validated on the caller's frame, before any device work
        lambda: cake_kcp.chunk_kda_kcp(*t, ctx, graph="capture"),
        lambda: cake_kcp.chunk_kda_kcp_fwd(*t, ctx, graph="capture"),
    ):
        with pytest.raises(ValueError, match="graph"):
            bad_call()
    assert cake_kcp._graph_entry("fwd", ctx, None, on, None, t, flags) is None
    ws = cake_kcp.KcpWorkspace(local=False)
    ring = cake_kcp._options(
        {
            "exchange": "ring",
            "save_transfer": False,
            "_kcp_test_knobs": {"graph": "replay"},
        }
    )
    assert cake_kcp._graph_entry("fwd", ctx, ws, ring, None, t, flags) is None
    dbg = cake_kcp._options(
        {"debug_sequence": True, "_kcp_test_knobs": {"graph": "replay"}}
    )
    assert cake_kcp._graph_entry("fwd", ctx, ws, dbg, None, t, flags) is None
    try:
        e = cake_kcp._graph_entry("fwd", ctx, ws, on, None, t, flags)
        assert e is not None and e.seen == 0 and e.graph is None
        assert e.digest.startswith("fwd:")
        assert cake_kcp._graph_entry("fwd", ctx, ws, on, None, t, flags) is e
        assert cake_kcp._graph_entry("bwd", ctx, ws, on, None, t, flags) is not e
        assert (
            cake_kcp._graph_entry(
                "fwd", ctx, ws, on, None, [t[0].clone(), *t[1:]], flags
            )
            is not e
        )
        # the same pointer with another stride / dtype / shape is another key
        assert (
            cake_kcp._graph_entry(
                "fwd", ctx, ws, on, None, [t[0].transpose(0, 1), *t[1:]], flags
            )
            is not e
        )
        assert (
            cake_kcp._graph_entry(
                "fwd", ctx, ws, on, None, [t[0].view(torch.int16), *t[1:]], flags
            )
            is not e
        )
        assert (
            cake_kcp._graph_entry("fwd", ctx, ws, on, None, [t[0][:2], *t[1:]], flags)
            is not e
        )
        assert (
            cake_kcp._graph_entry("fwd", ctx, ws, on, None, t, (False, "light"))
            is not e
        )
        assert cake_kcp._graph_entry("fwd", ctx, ws, on, 0.5, t, flags) is not e
        assert cake_kcp._graph_entry("fwd", object(), ws, on, None, t, flags) is not e
        cake_kcp.LAST_CALL["fwd"] = {
            "graph": "replay",
            "graph_event": "eager",
            "graph_key": None,
            "graph_fallback": None,
        }
        n0 = cake_kcp.GRAPH_STATS["eager"]
        cake_kcp._graph_eager("fwd", e)
        assert e.seen == 1 and cake_kcp.GRAPH_STATS["eager"] == n0 + 1
        assert cake_kcp.LAST_CALL["fwd"]["graph_key"] == e.digest
        assert cake_kcp.LAST_CALL["fwd"]["graph_fallback"] is None
        # graph_cache_clear(): drops every entry (to be called before destroy_process_group), reports the count, keeps the stats
        n_entries = len(cake_kcp._GRAPHS)
        assert n_entries >= 2 and "graph_cache_clear" in cake_kcp.__all__
        assert cake_kcp.graph_cache_clear() == n_entries
        assert cake_kcp._GRAPHS == {} and cake_kcp.graph_cache_clear() == 0
        assert cake_kcp.GRAPH_STATS["eager"] == n0 + 1
    finally:
        cake_kcp._GRAPHS.clear()
        cake_kcp.LAST_CALL.pop("fwd", None)


def test_cols_rule_thresholds() -> None:
    """D1: up to 24 concurrent 128-column state chains keep the registry's c8 record, more take the c16
    record; 256-column map chains stay with the registry's c16; ``fixed`` never asks for a record."""
    f = cake_kcp.merge_cols_per_cta
    assert cake_kcp.COLS_RULE_CHAINS == 24
    assert all(
        f(n, c, "fixed") is None for n in (128, 256) for c in (1, 24, 25, 64, 96)
    )
    assert all(f(128, c, "by_chains") is None for c in (1, 8, 24))
    assert all(f(128, c, "by_chains") == 16 for c in (25, 64, 96))
    assert all(f(256, c, "by_chains") is None for c in (1, 24, 25, 96))


def _check_fused_tables(ctx, H: int) -> dict:
    """The fused-walk tables of one rank against the separate-walk tables (``fused_walks=0``) of ``cake_kcp``."""
    TILE = cake_kcp.TILE
    tb0 = cake_kcp._Tables(ctx, H, fused_walks=0)
    tb1 = cake_kcp._Tables(ctx, H, fused_walks=1)
    P, NSEG = tb1.P, tb1.NSEG
    frags = list(ctx.fragments)
    live = [f for f in frags if f.length > 0]
    live_segs = {f.slot * P + p for f in live for p in range(P)}
    # one launch per phase visits exactly the segments of the separate launches, once each, longest first
    assert set(tb1.walk_live) == live_segs
    assert live_segs == set(tb0.walk_start) | set(tb0.walk_in_p0) | set(tb0.walk_in_p1)
    assert live_segs == set(tb0.walk_end) | set(tb0.walk_cross_last) | set(
        tb0.walk_cross_rest
    )
    assert set(tb1.walk_in_fused) == set(tb0.walk_in_p0) | set(tb0.walk_in_p1)
    assert set(tb1.walk_cross_fused) == set(tb0.walk_cross_last) | set(
        tb0.walk_cross_rest
    )
    assert len(tb1.walk_live) == len(live_segs)
    seg_len = [
        tb1.hp.seg_chunk_start[s + 1] - tb1.hp.seg_chunk_start[s] for s in tb1.walk_live
    ]
    assert seg_len == sorted(seg_len, reverse=True)
    # the default tables and walk lists are unchanged by the knob
    for name in (
        "walk_start",
        "walk_in_p0",
        "walk_in_p1",
        "walk_end",
        "walk_cross_last",
        "walk_cross_rest",
    ):
        assert getattr(tb0, name) == getattr(tb1, name), name
    for name in ("h0_index", "carry_index", "dht_index", "dcarry_index", "out_index"):
        assert (
            getattr(tb0, name) == getattr(tb1, name) == tuple(getattr(tb1.hp, name))
        ), name
    # the carry plane grows by one slot per crossing fragment (shared between the forward and the adjoint)
    extra = tb1.n_frag * (P - 1)
    n_extra = max(tb1.n_in, tb1.n_out) if P > 1 else 0
    assert len(tb1.carry_tiles) == len(tb0.carry_tiles) + n_extra == extra + n_extra
    assert tb1.sizes["carry"] == tb0.sizes["carry"] + n_extra * H * TILE
    assert tb1.sizes["carry"] == tb1.hp.numel["carry_fused"]
    assert tb0.sizes["carry"] == tb1.hp.numel["carry"]
    assert {k: v for k, v in tb1.sizes.items() if k != "carry"} == {
        k: v for k, v in tb0.sizes.items() if k != "carry"
    }
    # every segment reads its initial state (final adjoint) from ONE pool: the carry plane at P > 1 (-1 only for the
    # zero state of a document start / end), S_in / dS_out at P = 1 (the default tables)
    for f in frags:
        for p in range(P):
            s = f.slot * P + p
            h0, dht = tb1.h0_index_fused[s], tb1.dht_index_fused[s]
            if P == 1:
                assert (h0, dht) == (tb0.h0_index[s], tb0.dht_index[s])
                continue
            assert -1 <= h0 < len(tb1.carry_tiles) and -1 <= dht < len(tb1.carry_tiles)
            if p == 0:
                assert h0 == (extra + ctx.in_slot[f.slot] if f.crossing_in else -1)
            else:
                assert h0 == tb0.carry_index[s] == f.slot * (P - 1) + p - 1
            if p == P - 1:
                assert dht == (extra + ctx.out_slot[f.slot] if f.crossing_out else -1)
            else:
                assert dht == tb0.dcarry_index[s] == f.slot * (P - 1) + p
    # chain tables: identical except that the first element of every carry_in (dcarry_cross) chain publishes its
    # entering state -- the S_in (dS_out) tiles -- into the segment's slot of the fused table (write_states)
    ins = [f for f in live if f.crossing_in]
    outs = [f for f in live if f.crossing_out]
    for name, c0 in tb0.chains.items():
        c1 = tb1.chains[name]
        if name in ("carry_both", "dcarry_both"):
            # the fused chain families: in each table the concatenation of their two parts (whose own fused / unfused
            # relation is checked below); the carry_in / dcarry_cross first elements differ by design
            a, b = (
                ("carry_start", "carry_in")
                if name == "carry_both"
                else ("dcarry_end", "dcarry_cross")
            )
            for tb in (tb0, tb1):
                both = cake_kcp.Chains.concat(tb.chains[a], tb.chains[b])
                assert tb.chains[name].as_dict() == both.as_dict(), name
            continue
        d0, d1 = c0.as_dict(), c1.as_dict()
        for key in (
            "ncols",
            "n_chains",
            "chain_start",
            "elem_m",
            "elem_c",
            "x0_index",
            "y_final",
        ):
            assert d0[key] == d1[key], (name, key)
        if P > 1 and name in ("carry_in", "dcarry_cross"):
            fs = ins if name == "carry_in" else outs
            assert c1.n_chains == len(fs) * H
            firsts = set(c1.chain_start[:-1])
            for j, e in enumerate(c1.chain_start[:-1]):
                f, h = fs[j // H], j % H
                seg = f.slot * P if name == "carry_in" else f.slot * P + P - 1
                slot = (
                    tb1.h0_index_fused[seg]
                    if name == "carry_in"
                    else tb1.dht_index_fused[seg]
                )
                assert c0.elem_y[e] == -1
                assert c1.elem_y[e] == tb1.carry_tiles[slot][h] * TILE
                assert slot >= extra
            assert all(
                a == b
                for i, (a, b) in enumerate(zip(c0.elem_y, c1.elem_y, strict=True))
                if i not in firsts
            )
        else:
            assert d0["elem_y"] == d1["elem_y"], name
    return {
        "rank": ctx.rank,
        "P": P,
        "NSEG": NSEG,
        "n_live_segments": len(live_segs),
        "launches_default": [
            len(tb0.walk_start),
            len(tb0.walk_in_p0),
            len(tb0.walk_in_p1),
        ],
        "carry_slots": [len(tb0.carry_tiles), len(tb1.carry_tiles)],
    }


def _fused_cases():
    cases = [
        ("small-D", _small_partition(4)),
        ("small-C3000", _small_partition(4, lengths=(3000,), layout="contiguous")),
        ("small-Z", _small_partition(4, lengths=(3000, 776), layout="zigzag")),
    ]
    table = [[(0, 0, 1200)], [], [(0, 1200, 900)], [(0, 2100, 900)]]
    cases.append(
        (
            "ragged-zero-rank",
            kp.kcp_partition(
                torch.tensor([0, 3000], dtype=torch.int64), 4, fragments=table
            ),
        )
    )
    if kcp_shapes is not None:
        for name, W, lay in (
            ("Q1", 4, "C"),
            ("Q1", 4, "Z"),
            ("Q1", 4, "D"),
            ("Q1", 8, "C"),
            ("Q1", 16, "C"),
            ("Q1", 16, "Z"),
            ("C1", 4, "C"),
            ("C2", 4, "Z"),
            ("C3", 4, "D"),
            ("C4b", 4, "R"),
            ("C5", 4, "R"),
            ("C7", 4, "R"),
        ):
            cases.append(
                (
                    f"{name}-cp{W}-{lay}",
                    harness_partition(kcp_shapes.ROWS[name], W, lay)[1],
                )
            )
    return cases


FUSED_CASES = _fused_cases()


@pytest.mark.parametrize("case", FUSED_CASES, ids=[c[0] for c in FUSED_CASES])
@pytest.mark.parametrize("H", (4, 64))
def test_fused_walk_tables(case, H) -> None:
    """With ``fused_walks=1`` every live segment of every rank has a valid initial-state
    (final-adjoint) slot in one pool, the walk lists have one launch per phase, the carry plane grows by one slot per
    crossing fragment, and the only chain-table change is the publication of the ``S_in`` / ``dS_out`` copies."""
    name, part = case
    for ctx in contexts(part):
        rec = _check_fused_tables(ctx, H)
        _evidence("fused_walks", {"case": name, "H": H, **rec})


def test_fused_walk_tables_are_a_pure_function_of_the_plan() -> None:
    """G3: the fused tables (like the default ones) depend on (fragments, H, SM count) only -- two independently built
    contexts give identical tables; the plan JSON of a rank is identical across builds."""
    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    for r in range(4):
        a = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        b = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        assert a.describe(64) == b.describe(64)
        ta, tb = (
            cake_kcp._Tables(a, 64, fused_walks=1),
            cake_kcp._Tables(b, 64, fused_walks=1),
        )
        assert (
            ta.P > 1
        )  # a 3000-token document at CP4: segmented ranks (the carry plane is in use)
        for name in (
            "walk_live",
            "walk_in_fused",
            "walk_cross_fused",
            "h0_index_fused",
            "dht_index_fused",
        ):
            assert getattr(ta, name) == getattr(tb, name), name
        assert {k: c.as_dict() for k, c in ta.chains.items()} == {
            k: c.as_dict() for k, c in tb.chains.items()
        }
        assert ta.describe() == tb.describe()
        assert (
            ta.describe()["fused_walks"] == 1
            and cake_kcp._Tables(a, 64, fused_walks=0).describe()["fused_walks"] == 0
        )
        assert cake_kcp._Tables(a, 64).describe()["fused_walks"] == 1  # shipped


# --------------------------------------------------------------------------- walk launch geometry

WALK_LISTS = (
    "walk_offset",
    "walk_all",
    "walk_in",
    "walk_m_b1",
)  # the lists lpt_chains re-orders (segment order today)
TAPE_LISTS = (
    "walk_start",
    "walk_in_p0",
    "walk_in_p1",
    "walk_end",
    "walk_cross_last",
    "walk_cross_rest",
    "walk_live",
    "walk_in_fused",
    "walk_cross_fused",
)  # already LPT-ordered by the shipped tables
INDEX_TABLES = (
    "h0_index",
    "carry_index",
    "dht_index",
    "dcarry_index",
    "out_index",
    "h0_index_fused",
    "dht_index_fused",
    "in_index",
    "m_b1_index",
    "seg_index",
)


def test_walk_geometry_knob_is_gated_and_validated(monkeypatch) -> None:
    """``walk_geometry`` is a test knob: shipped default ``"lpt_tape_one_cta_per_sm"`` since this
    revision (``"shipped"`` = the earlier geometry, selectable through the knob), refused
    without :func:`cake_kcp_plan.enable_test_knobs`, never a public keyword, validated against :data:`WALK_GEOMETRIES` in the
    options and in the tables; the knob is recorded by the tables' ``describe()``."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    assert cake_kcp.WALK_GEOMETRY_DEFAULT == "lpt_tape_one_cta_per_sm"
    assert cake_kcp.WALK_GEOMETRIES == (
        "shipped",
        "lpt_chains",
        "one_cta_per_sm",
        "lpt_one_cta_per_sm",
        "tape_one_cta_per_sm",
        "lpt_tape_one_cta_per_sm",
    )
    assert cake_kcp.P2_KNOBS["walk_geometry"] == cake_kcp.WALK_GEOMETRIES
    assert cake_kcp._options({}).walk_geometry == "lpt_tape_one_cta_per_sm"
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"walk_geometry": "shipped"}})
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"walk_geometry": "lpt_chains"}})
    with pytest.raises(TypeError):
        cake_kcp._options({"walk_geometry": "lpt_chains"})
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    for g in cake_kcp.WALK_GEOMETRIES:
        o = cake_kcp._options({"_kcp_test_knobs": {"walk_geometry": g}})
        assert (o.walk_geometry, o.schedule, o.fused_walks) == (g, "wait_first", 1)
    # the earlier geometry stays selectable for the A/B record
    o = cake_kcp._options({"_kcp_test_knobs": {"walk_geometry": "shipped"}})
    assert o.walk_geometry == "shipped"
    o = cake_kcp._options(
        {"_kcp_test_knobs": {"walk_geometry": "lpt_one_cta_per_sm", "fused_walks": 0}}
    )
    assert (o.walk_geometry, o.fused_walks) == ("lpt_one_cta_per_sm", 0)
    for bad in ("lpt", "one_cta", "", 1):
        with pytest.raises(ValueError):
            cake_kcp._options({"_kcp_test_knobs": {"walk_geometry": bad}})
    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    with pytest.raises(ValueError, match="walk_geometry"):
        cake_kcp._Tables(ctx, 64, walk_geometry="lpt")
    assert (
        cake_kcp._Tables(ctx, 64).describe()["walk_geometry"]
        == "lpt_tape_one_cta_per_sm"
    )
    assert (
        cake_kcp._Tables(ctx, 64, walk_geometry="shipped").describe()["walk_geometry"]
        == "shipped"
    )
    assert (
        cake_kcp._Tables(ctx, 64, walk_geometry="lpt_chains").describe()[
            "walk_geometry"
        ]
        == "lpt_chains"
    )
    # cudaFuncCachePreferL1 brackets the capped launches, cudaFuncCachePreferNone restores the runtime's choice
    assert (cake_kcp.WALK_CACHE_CONFIG_CAP, cake_kcp.WALK_CACHE_CONFIG_DEFAULT) == (
        2,
        0,
    )
    assert set(cake_kcp.WALK_CAP_GEOMETRIES) == set(cake_kcp.WALK_CAP_PROGRAMS)
    assert set(cake_kcp.WALK_CAP_PROGRAMS["tape_one_cta_per_sm"]) == {
        "fwdh_tape",
        "dhu_tape",
    }
    assert set(cake_kcp.WALK_CAP_PROGRAMS["one_cta_per_sm"]) == {
        "fwdh_offset",
        "fwdh_tape",
        "dhu_offset",
        "dhu_tape",
    }
    for g in cake_kcp.WALK_CAP_GEOMETRIES:
        assert (
            cake_kcp.WALK_CAP_PROGRAMS[g]
            == cake_kcp.WALK_CAP_PROGRAMS[g.removeprefix("lpt_")]
        )
    assert set(cake_kcp.WALK_LPT_GEOMETRIES) | set(cake_kcp.WALK_CAP_GEOMETRIES) == set(
        cake_kcp.WALK_GEOMETRIES
    ) - {"shipped"}


def _seg_lengths(tb, segs) -> list[int]:
    return [tb.hp.seg_chunk_start[s + 1] - tb.hp.seg_chunk_start[s] for s in segs]


def _check_walk_geometry_tables(ctx, H: int) -> dict:
    """The walk geometry tables of one rank against the earlier ``shipped`` ones: lpt_chains permutes the offset / staging
    walk lists into LPT order (longest segment first, ties by segment number) and changes nothing else;
    one_cta_per_sm and tape_one_cta_per_sm change no table at all (the cap acts at launch time); the lpt_* cap lists
    equal lpt_chains'; the default tables (no geometry given) are the shipped ``lpt_tape_one_cta_per_sm`` ones."""
    tb0 = cake_kcp._Tables(ctx, H, walk_geometry="shipped")  # the earlier geometry
    tb_lpt = cake_kcp._Tables(ctx, H, walk_geometry="lpt_chains")
    tb_cap = cake_kcp._Tables(ctx, H, walk_geometry="one_cta_per_sm")
    tb_both = cake_kcp._Tables(ctx, H, walk_geometry="lpt_one_cta_per_sm")
    tb_tcap = cake_kcp._Tables(ctx, H, walk_geometry="tape_one_cta_per_sm")
    tb_tboth = cake_kcp._Tables(ctx, H, walk_geometry="lpt_tape_one_cta_per_sm")
    tb_def = cake_kcp._Tables(ctx, H)  # the shipped default since this revision
    rec = {
        "P": tb0.P,
        "NSEG": tb0.NSEG,
        "n_frag": tb0.n_frag,
        "reordered": {},
        "unequal": {},
    }
    for name in WALK_LISTS:
        shipped, lpt = list(getattr(tb0, name)), list(getattr(tb_lpt, name))
        assert sorted(shipped) == sorted(lpt), (
            name
        )  # the same launch, every segment once
        assert len(set(lpt)) == len(lpt), name
        lens = _seg_lengths(tb_lpt, lpt)
        assert lens == sorted(lens, reverse=True), (name, lens)
        for a, b in zip(lpt, lpt[1:], strict=False):  # ties by segment number
            la, lb = _seg_lengths(tb_lpt, (a, b))
            assert la > lb or (la == lb and a < b), (name, a, b)
        assert list(getattr(tb_cap, name)) == shipped, name
        assert list(getattr(tb_both, name)) == lpt, name
        assert list(getattr(tb_tcap, name)) == shipped, name
        assert list(getattr(tb_tboth, name)) == lpt, name
        assert list(getattr(tb_def, name)) == lpt, name
        rec["reordered"][name] = shipped != lpt
        # the re-sort changes a list exactly when its shipped order is not already LPT (equal chains: nothing moves)
        key = {s: (-_seg_lengths(tb0, (s,))[0], s) for s in shipped}
        rec["unequal"][name] = shipped != sorted(shipped, key=key.__getitem__)
        assert rec["reordered"][name] == rec["unequal"][name], (name, shipped, lpt)
    for name in TAPE_LISTS:  # already LPT: unchanged under every geometry
        ref = list(getattr(tb0, name))
        lens = _seg_lengths(tb0, ref)
        assert lens == sorted(lens, reverse=True), name
        for tb in (tb_lpt, tb_cap, tb_both, tb_tcap, tb_tboth, tb_def):
            assert list(getattr(tb, name)) == ref, name
    for name in INDEX_TABLES:
        for tb in (tb_lpt, tb_cap, tb_both, tb_tcap, tb_tboth, tb_def):
            assert tuple(getattr(tb, name)) == tuple(getattr(tb0, name)), name
    for tb in (tb_lpt, tb_cap, tb_both, tb_tcap, tb_tboth, tb_def):
        assert tb.sizes == tb0.sizes and tb.splits == tb0.splits
        assert {k: c.as_dict() for k, c in tb.chains.items()} == {
            k: c.as_dict() for k, c in tb0.chains.items()
        }
        d0, d1 = tb0.describe(), tb.describe()
        assert {k: v for k, v in d1.items() if k != "walk_geometry"} == {
            k: v for k, v in d0.items() if k != "walk_geometry"
        }
    assert tb0.describe()["walk_geometry"] == "shipped"
    assert tb_lpt.describe()["walk_geometry"] == "lpt_chains"
    assert tb_cap.describe()["walk_geometry"] == "one_cta_per_sm"
    assert tb_both.describe()["walk_geometry"] == "lpt_one_cta_per_sm"
    assert tb_tcap.describe()["walk_geometry"] == "tape_one_cta_per_sm"
    assert tb_tboth.describe()["walk_geometry"] == "lpt_tape_one_cta_per_sm"
    # the default tables ARE the adopted geometry's
    assert tb_def.describe() == tb_tboth.describe()
    return rec


@pytest.mark.parametrize("case", FUSED_CASES, ids=[c[0] for c in FUSED_CASES])
@pytest.mark.parametrize("H", (4, 64))
def test_walk_geometry_tables(case, H) -> None:
    """Walk geometry: the knob's tables against the shipped ones on every rank of the small partitions."""
    name, part = case
    for ctx in contexts(part):
        rec = _check_walk_geometry_tables(ctx, H)
        _evidence("walk_geometry", {"case": name, "H": H, "rank": ctx.rank, **rec})


@HARNESS
@pytest.mark.parametrize(
    "case",
    [c for c in CASES if c[0].startswith("Q") and c[1] in (8, 16)],
    ids=[
        f"{c[0]}-cp{c[1]}-{c[2]}"
        for c in CASES
        if c[0].startswith("Q") and c[1] in (8, 16)
    ],
)
def test_walk_geometry_tables_on_perf_rows(case) -> None:
    """walk geometry on the CP8 / CP16 perf rows of record (H = 64): on every rank the lpt_chains lists are LPT permutations
    that touch no other table, and a row re-orders some offset walk exactly when some rank's shipped list is not
    already longest-first (rows whose ranks all carry equal chains, e.g. Q2 CP8 Z, re-order nothing)."""
    row_name, W, lay = case
    row = kcp_shapes.ROWS[row_name]
    _, part = harness_partition(row, W, lay)
    reordered = unequal = False
    for ctx in contexts(part):
        rec = _check_walk_geometry_tables(ctx, 64)
        reordered = reordered or any(rec["reordered"].values())
        unequal = unequal or any(rec["unequal"].values())
        _evidence(
            "walk_geometry_perf",
            {"row": row_name, "cp": W, "layout": lay, "rank": ctx.rank, **rec},
        )
    assert reordered == unequal, (row_name, W, lay)


def test_walk_geometry_tables_are_a_pure_function_of_the_plan() -> None:
    """G3: the walk geometry lists depend on (fragments, H, SM count) only -- two independently built contexts give identical
    lists under every geometry."""
    part = _small_partition(4, lengths=(3000, 700), layout="zigzag")
    for r in range(4):
        a = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        b = kp.build_kcp_context(part, None, r, "cpu", 4, sm_count=SM_COUNT)
        for g in cake_kcp.WALK_GEOMETRIES:
            ta, tb = (
                cake_kcp._Tables(a, 64, walk_geometry=g),
                cake_kcp._Tables(b, 64, walk_geometry=g),
            )
            for name in WALK_LISTS + TAPE_LISTS:
                assert getattr(ta, name) == getattr(tb, name), (g, name)
            assert ta.describe() == tb.describe()


# --------------------------------------------------------------------------- GPU table parity (R15; `-k gpu`)


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {tuple(cake_jit.ARCHES)}"
        )
    return torch.device("cuda", index)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


PARITY_CASES = (
    [c for c in CASES if kcp_shapes.ROWS[c[0]].kind != "perf"]
    if kcp_shapes is not None
    else []
)
PARITY_IDS = [f"{name}-cp{W}-{lay}" for name, W, lay in PARITY_CASES]
PARITY_EXTRA_P = (
    1,
    3,
    64,
)  # segment counts checked beside the plan's own P (empty segments, P > chunks, P = 1)


@HARNESS
@pytest.mark.parametrize("case", PARITY_CASES, ids=PARITY_IDS)
def test_gpu_table_parity(device, case) -> None:
    """Gate 5 (R15): the CPU-built chunk_bos / chunk_len / seq_chunk_start / seq_order are bitwise the
    kda_chunk_train_layout_table_v1 output on every rank of every C row, adversarial row and ragged table (one launch
    per rank; pad entries pre-filled with a sentinel), and the boundary kernels' segment table follows its formula from
    that output for the plan's P and the extra segment counts (its own program left the package with the segmented
    recurrences' block table; the kernels that read it are covered by the FP64-referenced rows)."""
    name, W, layout = case
    row = kcp_shapes.ROWS[name]
    _frags, part = harness_partition(row, W, layout)
    sm = torch.cuda.get_device_properties(device).multi_processor_count
    i32 = torch.int32
    failures = []
    checked = 0
    for r in range(W):
        ctx = kp.build_kcp_context(part, None, r, device)
        if ctx.n_frag == 0:
            _evidence(
                "parity",
                {
                    "row": name,
                    "W": W,
                    "layout": layout,
                    "rank": r,
                    "table": "all",
                    "equal": None,
                    "note": "no local sequence (T_r = 0, no fragments): the table program is not launched",
                },
            )
            continue
        N, NC_pad = ctx.n_frag, ctx.NC_pad
        bos = torch.full((NC_pad,), -7, dtype=i32, device=device)
        ln = torch.full((NC_pad,), -7, dtype=i32, device=device)
        scs = torch.full((N + 1,), -7, dtype=i32, device=device)
        order = torch.full((N,), -7, dtype=i32, device=device)
        values = {
            "cu_seqlens": ctx.cu_seqlens_local,
            "num_seqs": N,
            "num_chunks_pad": NC_pad,
            "chunk_bos": bos,
            "chunk_len": ln,
            "seq_chunk_start": scs,
            "seq_order": order,
        }
        grid = cake_jit.launch_grid(
            "table", num_chunks_pad=NC_pad, num_seqs=N, num_heads=row.h, sm_count=sm
        )
        cake_jit.run_module("table", values, grid=grid, device_index=device.index)
        torch.cuda.synchronize(device)
        got = {
            "chunk_bos": bos.cpu().tolist(),
            "chunk_len": ln.cpu().tolist(),
            "seq_chunk_start": scs.cpu().tolist(),
            "seq_order": order.cpu().tolist(),
        }
        want = {
            "chunk_bos": list(ctx.chunk_bos),
            "chunk_len": list(ctx.chunk_len),
            "seq_chunk_start": list(ctx.seq_chunk_start),
            "seq_order": list(ctx.seq_order),
        }
        for table in want:
            equal = got[table] == want[table]
            checked += 1
            _evidence(
                "parity",
                {
                    "row": name,
                    "W": W,
                    "layout": layout,
                    "rank": r,
                    "table": table,
                    "entries": len(want[table]),
                    "equal": equal,
                    "sm_count": sm,
                    "P": None,
                },
            )
            if not equal:
                failures.append((r, table))
        hp = ctx.heads(row.h)
        for P in sorted({hp.P, *PARITY_EXTRA_P}):
            # the segment table from the device-built seq_chunk_start (the table program's output) must equal the
            # CPU formula from the CPU-built one; at the plan's P it is the head plan's table
            want_seg = kp.seg_chunk_start(list(ctx.seq_chunk_start), P)
            if P == hp.P:
                assert tuple(want_seg) == hp.seg_chunk_start
            equal = kp.seg_chunk_start(got["seq_chunk_start"], P) == want_seg
            checked += 1
            _evidence(
                "parity",
                {
                    "row": name,
                    "W": W,
                    "layout": layout,
                    "rank": r,
                    "table": "seg_chunk_start",
                    "entries": len(want_seg),
                    "equal": equal,
                    "sm_count": sm,
                    "P": P,
                    "plan_P": hp.P,
                },
            )
            if not equal:
                failures.append((r, f"seg_chunk_start[P={P}]"))
    assert checked > 0
    assert not failures, f"table parity failures (rank, table): {failures}"


# --------------------------------------------------------------------------- FP64 fake backend (`-k fake`)


FAKE_CASES = [
    ("C1", 2, "C"),
    ("C1", 4, "C"),
    ("C2", 4, "C"),
    ("C2", 4, "Z"),
    ("C2", 4, "D"),
    ("C3", 4, "D"),
    ("C4a", 4, "R"),
    ("C4b", 4, "R"),
    ("C5", 4, "R"),
    ("C5b", 4, "R"),
]
FAKE_IDS = [f"{name}-cp{W}-{lay}" for name, W, lay in FAKE_CASES]
FAKE_REL_RMS = 1e-12  # vs the unpartitioned FP64 reference ("~1e-12", the design note §11.3); declared before the run
FAKE_MAX_ABS_REL = 1e-11  # max |diff| relative to max |ref| of the tensor
FAKE_CHAIN_VS_LOCAL = 1e-9  # owner-chain dS_in vs the local adjoint's dS0 (absolute; FP64 states of unit scale)
_REFS: dict = {}


@pytest.fixture(scope="module")
def fake():
    pytest.importorskip("kcp_inputs")
    pytest.importorskip("reference_kcp")
    from tests.test_helpers import cake_kcp_fake_backend as fb

    return fb


def _progress(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _reference_of(fake, row):
    ref = _REFS.get(row.name)
    if ref is None:
        ref = fake.load_or_compute_reference(
            row, os.environ.get("KCP_REF_DIR"), progress=_progress
        )
        _REFS[row.name] = ref
    return ref


def _assert_close_to_reference(fake, res, ref, row, docs, tag: str) -> dict:
    m = fake.compare_to_reference(res, ref, row.cu_seqlens, docs)
    for tensor, mm in m.items():
        _progress(
            f"{tag} {tensor}: rel_rms {mm['rel_rms']:.3e} max_abs {mm['max_abs']:.3e} (ref max {mm['ref_max']:.3e}, rel {mm['max_abs_rel']:.3e})"
        )
    for tensor, mm in m.items():
        assert mm["rel_rms"] <= FAKE_REL_RMS, (tag, tensor, mm)
        assert mm["max_abs_rel"] <= FAKE_MAX_ABS_REL, (tag, tensor, mm)
    return m


@HARNESS
@pytest.mark.parametrize("case", FAKE_CASES, ids=FAKE_IDS)
def test_fake_backend_matches_fp64_reference(fake, case) -> None:
    """Gate 6: the FP64 fake backend (W emulated ranks, dict collectives, the reference per-fragment compute) equals the
    unpartitioned FP64 reference to ~1e-12 on C1-C5 at CP2 / CP4 in every layout of the row; saved == replay bitwise;
    the collective order is the design's."""
    name, W, layout = case
    row = kcp_shapes.ROWS[name]
    t0 = time.time()
    res = fake.run_fake(row, W, layout, progress=_progress)
    ref = _reference_of(fake, row)
    docs = list(range(len(row.lengths)))
    m = _assert_close_to_reference(
        fake, res, ref, row, docs, f"{name} cp{W} {layout} saved"
    )
    assert fake.replay_equals_saved(res), (
        "replay-mode gradients differ from saved-mode gradients"
    )
    assert res["logs"]["forward"] == ["c1", "c2"]
    assert res["logs"]["backward_saved"] == ["b1", "b2"]
    assert res["logs"]["backward_replay"] == ["c1", "c2", "b1", "b2"]
    for mode in ("saved", "replay"):
        assert res["checks"][mode]["chain_vs_local_dS_in"] <= FAKE_CHAIN_VS_LOCAL, res[
            "checks"
        ]
    ranks = res["ranks"]
    _evidence(
        "fake",
        {
            "row": name,
            "W": W,
            "layout": layout,
            "H": row.h,
            "T_r": [c.T_r for c in ranks.ctxs],
            "n_frag": [c.n_frag for c in ranks.ctxs],
            "N_in": ranks.ctxs[0].N_in,
            "metrics_saved": m,
            "replay_equals_saved": True,
            "checks": res["checks"],
            "max_rel_rms": max(mm["rel_rms"] for mm in m.values()),
            "max_abs_rel": max(mm["max_abs_rel"] for mm in m.values()),
            "elapsed_s": round(time.time() - t0, 1),
            "pass": True,
        },
    )


@HARNESS
def test_fake_backend_save_transfer_is_bitwise(fake) -> None:
    """save_transfer=True (the owner keeps the c1 receive tensor) gives bitwise the save_transfer=False gradients."""
    row = kcp_shapes.ROWS["C2"]
    a = fake.run_fake(row, 4, "D", save_transfer=False, progress=_progress)
    b = fake.run_fake(row, 4, "D", save_transfer=True, progress=_progress)
    for tensor in ("out", "dq", "dk", "dv", "dg", "dbeta"):
        assert torch.equal(a["saved"][tensor], b["saved"][tensor]), tensor
    assert fake.replay_equals_saved(b)
    _evidence(
        "fake",
        {
            "row": "C2",
            "W": 4,
            "layout": "D",
            "check": "save_transfer bitwise",
            "pass": True,
        },
    )


@HARNESS
@pytest.mark.parametrize(
    "case", [("C1", 4, "C"), ("C3", 4, "D")], ids=["C1-cp4-C", "C3-cp4-D"]
)
def test_fake_backend_document_alone_equals_slice(fake, case) -> None:
    """Gate 6: every document run alone (its own partition over the same W) equals the slice of the whole-pack run to
    rounding: no leakage between documents, no dependence of a document's result on the other documents."""
    name, W, layout = case
    row = kcp_shapes.ROWS[name]
    full = fake.run_fake(row, W, layout, progress=_progress)
    cu = row.cu_seqlens
    for d in range(len(row.lengths)):
        alone = fake.run_fake(row, W, layout, docs=[d], progress=_progress)
        for tensor in ("out", "dq", "dk", "dv", "dg", "dbeta"):
            mm = fake.metrics(
                alone["saved"][tensor], full["saved"][tensor][cu[d] : cu[d + 1]]
            )
            _progress(
                f"{name} doc {d} alone vs slice {tensor}: rel_rms {mm['rel_rms']:.3e} max_abs_rel {mm['max_abs_rel']:.3e}"
            )
            assert (
                mm["rel_rms"] <= FAKE_REL_RMS and mm["max_abs_rel"] <= FAKE_MAX_ABS_REL
            ), (d, tensor, mm)
        _evidence(
            "fake",
            {
                "row": name,
                "W": W,
                "layout": layout,
                "check": f"document {d} alone == slice",
                "pass": True,
            },
        )


@HARNESS
def test_fake_backend_transfer_matches_lane_p(fake) -> None:
    """The identity / v = 0 transfer map equals the reference basis-vector ``transfer`` helper (FP64) on a 63-token fragment."""
    row = kcp_shapes.ROWS["C5"]
    ranks = fake.build_ranks(row, 4, "R", progress=_progress)
    err = fake._transfer_matches_lane_p(ranks.frag_ops(0, 0), row.h)
    _progress(
        f"transfer (identity, v = 0) vs the reference transfer(): max |diff| {err:.3e}"
    )
    assert err <= 1e-13
    _evidence(
        "fake",
        {"row": "C5", "check": "transfer vs reference", "max_abs": err, "pass": True},
    )


@pytest.mark.parametrize("walk_form", ["auto", "full"])
@pytest.mark.parametrize(
    "graph_cap", ["none", "capture_pref", "node_attr", "node_attr_post"]
)
def test_gpu_graph_replay_is_bitwise(device, monkeypatch, graph_cap, walk_form) -> None:
    """The graph-replay option, KDA module, one rank (``world == 1``: no collective): under the public keyword
    ``graph="replay"`` the first call of a validity key runs eagerly, the second is captured and replayed once, later
    calls replay the graph; ``out`` and the five gradients are bitwise the eager (``graph="off"``) results on every call.
    The backward's key holds the forward's saved pointers, so its capture follows the forward's by one call.  Another
    input pointer is another key (one eager call, still bitwise); ``graph_cache_clear()`` drops the entries.
    Both cache-config forms of a capture (knob ``graph_cap``: ``"none"`` / ``"capture_pref"``) capture
    and replay under the shipped capped geometry: the per-launch bracket is skipped inside the capture
    (``capture_skips`` grows), the eager calls record ``cache_config="bracket"`` and the captured / replayed calls
    record the arm (with the set / reset evidence of the capture-wide preference under ``"capture_pref"``).  The kernel-node
    forms ``"node_attr"`` / ``"node_attr_post"`` capture and replay bitwise too: no
    launch is skipped (``capture_skips`` unchanged), every captured tape launch gets the carveout attribute (``node_attrs``
    grows by the tape launches of the captured calls, recorded as ``cache_config_nodes``).  The cap applies to the
    full-width launches of the capped programs only (a two-CTA launch runs uncapped), so the expectations follow the
    forms the calls launched (``LAST_CALL[kind]["walk_forms"]``): under ``walk_form="auto"`` the registry's rule
    decides per launch, under ``"full"`` every launch is full-width and the bracket / skip / node-attribute paths are
    exercised."""
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    knobs = {"_kcp_test_knobs": {"graph_cap": graph_cap, "walk_form": walk_form}}
    backend = cake_kcp.get_backend()
    programs = backend.programs() if hasattr(backend, "programs") else None
    cap_programs = cake_kcp.WALK_CAP_PROGRAMS.get(cake_kcp.WALK_GEOMETRY_DEFAULT, ())
    capped = isinstance(programs, cake_kcp._FiPrograms) and bool(cap_programs)

    def n_capped(rec) -> int:
        """The call's launches under the cap: full-width launches of the capped programs (``walk_forms``)."""
        return sum(
            1
            for name, form, _ in rec["walk_forms"]
            if cake_kcp._PROGRAM_OF[name] in cap_programs and form == "full"
        )

    def eager_cc(rec) -> str:
        return "bracket" if capped and n_capped(rec) else "none"

    dt = backend.dtypes
    H = 2
    cu = torch.tensor([0, 300, 1024, 1500], dtype=torch.int64)
    part = kp.kcp_partition(cu, 1)
    ctx = kp.build_kcp_context(part, None, 0, device)
    T = int(ctx.T_r)
    gen = torch.Generator(device=device).manual_seed(0x3837)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(
        x,
    ):  # the delta rule's operand contract (harness kcp_inputs): q / k L2-normalized per head; raw randn keys blow the state up
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = unit(randn(T, H, cake_kcp.D)).to(dt["q"])
    k = unit(randn(T, H, cake_kcp.D)).to(dt["k"])
    v = (0.5 * randn(T, H, cake_kcp.D)).to(dt["v"])
    g = (-5.0 * torch.sigmoid(randn(T, H, cake_kcp.D))).to(
        dt["g"]
    )  # natural-log decay in (-5, 0), the harness's default regime
    beta = torch.sigmoid(randn(T, H)).to(dt["beta"])
    dout = (0.5 * randn(T, H, cake_kcp.D)).to(dt["dout"])
    ws = cake_kcp.KcpWorkspace(state_dtype=dt["state"], local=backend.local_stages)

    def run(**kw):
        out, saved = cake_kcp.chunk_kda_kcp_fwd(
            q, k, v, g, beta, ctx, workspace=ws, **kw
        )
        rec_f = dict(cake_kcp.LAST_CALL["fwd"])
        grads = cake_kcp.chunk_kda_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **kw
        )
        rec_b = dict(cake_kcp.LAST_CALL["bwd"])
        return (
            (out, *grads),
            (rec_f["graph_event"], rec_b["graph_event"]),
            (rec_f, rec_b),
        )

    cake_kcp.graph_cache_clear()
    # the eager reference requests the SPEC-3 probe (off by default): the first saved-mode backward of the process
    # runs it, and the replay-mode calls below (probe off) must still match the reference bitwise
    probing = backend._spec3_pending and backend.local_stages and T > 0
    ref, ev0, recs0 = run(graph="off", spec3_probe="record", **knobs)
    assert not probing or (recs0[1]["spec3"] or {}).get("checked"), recs0[1]["spec3"]
    ref = tuple(
        x.clone() for x in ref
    )  # copies taken before any other call: the comparison stays explicit
    assert all(torch.isfinite(x.float()).all().item() for x in ref), (
        "the eager reference is not finite: the operands violate the delta rule's contract"
    )
    assert ev0 == ("eager", "eager") and recs0[0]["graph"] == "off"
    assert (recs0[0]["cache_config"], recs0[1]["cache_config"]) == (
        eager_cc(recs0[0]),
        eager_cc(recs0[1]),
    ), recs0
    if (
        capped and walk_form == "full"
    ):  # every tape launch full-width: the eager bracket on both calls
        assert n_capped(recs0[0]) >= 1 and n_capped(recs0[1]) >= 1, recs0
    fallback0 = cake_kcp.GRAPH_STATS["fallback"]
    skips0 = programs.capture_skips if capped else None
    nodes0 = programs.node_attrs if capped else None
    events, ccs, brackets, nodes, caps, inv = [], [], [], [], [], []
    try:
        for _ in range(4):
            res, ev, recs = run(graph="replay", **knobs)
            events.append(ev)
            ccs.append((recs[0]["cache_config"], recs[1]["cache_config"]))
            brackets.append(
                (recs[0]["cache_config_bracket"], recs[1]["cache_config_bracket"])
            )
            nodes.append(
                (
                    recs[0].get("cache_config_nodes") or 0,
                    recs[1].get("cache_config_nodes") or 0,
                )
            )
            caps.append((n_capped(recs[0]), n_capped(recs[1])))
            inv.append(recs)
            assert all(torch.equal(a, b) for a, b in zip(res, ref, strict=True)), events
        assert [e[0] for e in events] == ["eager", "capture", "replay", "replay"], (
            events
        )
        assert events[-1][1] == "replay" and "capture" in [e[1] for e in events], events
        assert cake_kcp.GRAPH_STATS["fallback"] == fallback0
        assert cake_kcp.LAST_CALL["fwd"]["graph"] == "replay"
        cap_cc = graph_cap if capped else "none"
        assert [c[0] for c in ccs] == [eager_cc(inv[0][0]), cap_cc, cap_cc, cap_cc], ccs
        # the backward's validity key hashes the workspace pools and the saved tensors: whether the second
        # replay-mode call repeats the first call's key (capture at the second call, like the forward) or sees a
        # key the first calls changed (capture at the third) depends on which pools those calls create -- both
        # were observed (sm_100a two-level-carry rows: second; sm_103a: third) -- so the sequence is pinned by its
        # structure: eager calls, one capture, replays after it
        bwd_events = [e[1] for e in events]
        cap_b = bwd_events.index("capture")
        assert cap_b in (1, 2), events
        assert bwd_events == ["eager"] * cap_b + ["capture"] + ["replay"] * (
            3 - cap_b
        ), events
        assert [c[1] for c in ccs] == [eager_cc(inv[i][1]) for i in range(cap_b)] + [
            cap_cc
        ] * (4 - cap_b), ccs
        n_cap_captured = (
            caps[1][0] + caps[cap_b][1]
        )  # the capped (full-width tape) launches inside the two captures
        if capped and walk_form == "full":
            assert n_cap_captured >= 2, (
                caps
            )  # both captures hold full-width tape launches
        if capped and graph_cap in cake_kcp.GRAPH_NODE_CAPS:
            # kernel-node cap forms: nothing launched bare; every captured capped launch carries the node attribute
            assert programs.capture_skips == skips0, (programs.capture_skips, skips0)
            assert programs.node_attrs == nodes0 + n_cap_captured, (
                programs.node_attrs,
                nodes0,
                caps,
            )
            assert nodes[1][0] == caps[1][0] and nodes[cap_b][1] == caps[cap_b][1], (
                nodes,
                caps,
            )
            assert [n[0] for n in nodes[1:]] == [caps[1][0]] * 3, (
                nodes
            )  # the replays carry the capture's record
            assert brackets[1][0] is None and brackets[cap_b][1] is None, brackets
            if n_cap_captured:
                evidence = programs.node_attr_log
                assert evidence and all(
                    e["rc"] == 0
                    and e["node_type"] == 0
                    and e["readback"] == cake_kcp.GRAPH_NODE_CARVEOUT
                    for log in evidence.values()
                    for e in log
                ), evidence
        elif capped:
            assert programs.capture_skips == skips0 + n_cap_captured, (
                programs.capture_skips,
                skips0,
                caps,
            )  # each capped launch inside a capture launches bare and counts one skip
            if graph_cap == "capture_pref":
                for b in (
                    brackets[1][0],
                    brackets[cap_b][1],
                ):  # the forward's and the backward's capture evidence
                    assert [e["set"] for e in b] == [
                        cake_kcp.WALK_CACHE_CONFIG_CAP,
                        cake_kcp.WALK_CACHE_CONFIG_DEFAULT,
                    ], b
                    assert all(e["rc"] == 0 and e["readback"] == e["set"] for e in b), b
            else:
                assert brackets[1][0] is None and brackets[cap_b][1] is None, brackets
        out2, _ = cake_kcp.chunk_kda_kcp_fwd(
            q.clone(), k, v, g, beta, ctx, workspace=ws, graph="replay", **knobs
        )
        assert cake_kcp.LAST_CALL["fwd"]["graph_event"] == "eager" and torch.equal(
            out2, ref[0]
        )
        n_entries = cake_kcp.graph_cache_clear()
        assert n_entries >= 2 and cake_kcp.graph_cache_clear() == 0
    finally:
        cake_kcp.graph_cache_clear()
    res, ev, _ = run(graph="off", **knobs)
    assert ev == ("eager", "eager") and all(
        torch.equal(a, b) for a, b in zip(res, ref, strict=True)
    )
    _evidence(
        f"gpu_graph_replay_{graph_cap}_{walk_form}",
        {
            "module": "cake_kcp",
            "T": T,
            "H": H,
            "graph_cap": graph_cap,
            "walk_form": walk_form,
            "walk_forms_eager": [list(map(list, r["walk_forms"])) for r in recs0],
            "capped_geometry": capped,
            "events": events,
            "cache_config": ccs,
            "cache_config_bracket": brackets,
            "capture_skips_delta": (programs.capture_skips - skips0)
            if capped
            else None,
            "node_attrs_delta": (programs.node_attrs - nodes0) if capped else None,
            "cache_config_nodes": nodes,
            "capped_launches": caps,
            "node_attr_log": {
                f"{k[0]}:{k[1]}": v for k, v in programs.node_attr_log.items()
            }
            if capped
            else None,
            "bitwise": True,
            "fallbacks": cake_kcp.GRAPH_STATS["fallback"] - fallback0,
        },
    )


class _FakeJit:
    """A program provider's registry double for the CPU tests of :class:`cake_kcp._FiPrograms`: records every launch."""

    def __init__(self) -> None:
        self.launched: list[tuple] = []

    def launch_kcp(
        self, program, abi, *, grid, device_index, precision_mode, transfer_walk
    ) -> None:
        self.launched.append(
            (program, grid, device_index, precision_mode, transfer_walk)
        )


def _fake_call(geometry: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(walk_geometry=geometry, scratch={})


def test_capped_launch_without_cudart_degrades(monkeypatch) -> None:
    """When no soname of ``_CUDART_SONAMES`` loads, the shipped geometry does not raise: one
    ``RuntimeWarning`` naming the sonames tried, the capped programs launch without the bracket (the LPT order is the
    tables', untouched here), nothing is logged as a bracket, and the call records ``cache_config="unavailable"``; the
    failed load is remembered (no second warning) and the capture-wide preference reports ``unavailable`` too."""
    monkeypatch.setattr(cake_kcp, "_CUDART_SONAMES", ("libcudart_bogus_cake3837.so.0",))
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {})
    monkeypatch.setattr(cake_kcp, "_capturing", lambda: False)
    stats0 = dict(cake_kcp.CACHE_CONFIG_STATS)
    jit = _FakeJit()
    programs = cake_kcp._FiPrograms(jit)
    call = _fake_call(cake_kcp.WALK_GEOMETRY_DEFAULT)
    program = cake_kcp.WALK_CAP_PROGRAMS[cake_kcp.WALK_GEOMETRY_DEFAULT][0]
    with pytest.warns(RuntimeWarning, match="libcudart_bogus_cake3837") as rec:
        programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    runtime_warnings = [w for w in rec if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 1 and "unavailable" in str(
        runtime_warnings[0].message
    )
    assert jit.launched == [(program, 4, 0, 0, "bf16x3_hw64")]
    assert (
        programs.carveout_log == {}
        and programs.carveout_state == {}
        and programs.unavailable == 1
    )
    assert call.scratch["cache_config"] == "unavailable" and cake_kcp._CUDART_CACHE == {
        "rt": None
    }
    with warnings.catch_warnings():
        warnings.simplefilter(
            "error"
        )  # the second launch warns no more (the failed load is remembered)
        programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    assert len(jit.launched) == 2 and programs.unavailable == 2
    with pytest.raises(RuntimeError, match="libcudart_bogus_cake3837"):
        cake_kcp.walk_device_cache_config(
            0, None
        )  # the evidence function itself still reports the missing runtime
    cap = types.SimpleNamespace(
        opts=types.SimpleNamespace(graph_cap="capture_pref"),
        walk_geometry=cake_kcp.WALK_GEOMETRY_DEFAULT,
    )
    assert cake_kcp._graph_cap_bracket(cap) == ("unavailable", None)
    # _record counts calls, not launches: nothing recorded here
    assert dict(cake_kcp.CACHE_CONFIG_STATS) == stats0


def test_capped_launch_skips_the_bracket_inside_a_capture(monkeypatch) -> None:
    """Inside a stream capture a capped program launches bare (no ``cudaDeviceSetCacheConfig``, which
    would invalidate the capture) and ``capture_skips`` counts it; outside a capture the bracket is issued (set + reset,
    logged as evidence) and the call records ``cache_config="bracket"``; a geometry without capped programs never
    brackets.  The capture-wide preference is arm (i)'s only with a capped geometry and a runtime."""
    monkeypatch.setattr(
        cake_kcp, "_CUDART_CACHE", {"rt": object()}
    )  # a loadable runtime (never called below)
    sets: list[tuple[int, int]] = []

    def fake_set(device_index, config):
        sets.append((int(device_index), int(config)))
        return {
            "device_index": int(device_index),
            "set": config,
            "rc": 0,
            "readback": config,
        }

    monkeypatch.setattr(cake_kcp, "walk_device_cache_config", fake_set)
    jit = _FakeJit()
    programs = cake_kcp._FiPrograms(jit)
    program = cake_kcp.WALK_CAP_PROGRAMS[cake_kcp.WALK_GEOMETRY_DEFAULT][0]
    stats0 = dict(cake_kcp.CACHE_CONFIG_STATS)
    monkeypatch.setattr(cake_kcp, "_capturing", lambda: True)
    call = _fake_call(cake_kcp.WALK_GEOMETRY_DEFAULT)
    programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    assert (
        jit.launched == [(program, 4, 0, 0, "bf16x3_hw64")]
        and sets == []
        and programs.capture_skips == 1
    )
    assert (
        "cache_config" not in call.scratch
    )  # the captured call's form is _graph_capture's (_graph_cap_bracket)
    assert cake_kcp.CACHE_CONFIG_STATS["capture_skips"] == stats0["capture_skips"] + 1
    monkeypatch.setattr(cake_kcp, "_capturing", lambda: False)
    programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    assert len(jit.launched) == 2
    assert sets == [
        (0, cake_kcp.WALK_CACHE_CONFIG_CAP),
        (0, cake_kcp.WALK_CACHE_CONFIG_DEFAULT),
    ]
    assert call.scratch["cache_config"] == "bracket"
    assert programs.carveout_state == {0: cake_kcp.WALK_CACHE_CONFIG_DEFAULT}
    assert [e["set"] for e in programs.carveout_log[(program, 0)]] == [
        cake_kcp.WALK_CACHE_CONFIG_CAP,
        cake_kcp.WALK_CACHE_CONFIG_DEFAULT,
    ]
    plain = _fake_call("shipped")
    programs.launch(program, [], grid=4, device_index=0, args={"_call": plain})
    assert len(jit.launched) == 3 and len(sets) == 2 and plain.scratch == {}
    cap = types.SimpleNamespace(
        opts=types.SimpleNamespace(graph_cap="capture_pref"),
        walk_geometry=cake_kcp.WALK_GEOMETRY_DEFAULT,
    )
    assert cake_kcp._graph_cap_bracket(cap) == (
        "capture_pref",
        cake_kcp.WALK_CACHE_CONFIG_CAP,
    )
    cap.walk_geometry = "shipped"
    assert cake_kcp._graph_cap_bracket(cap) == ("none", None)
    none_ = types.SimpleNamespace(
        opts=types.SimpleNamespace(graph_cap="none"),
        walk_geometry=cake_kcp.WALK_GEOMETRY_DEFAULT,
    )
    assert cake_kcp._graph_cap_bracket(none_) == ("none", None)


def test_spec3_probe_is_a_keyword_and_the_package_reads_no_environment_switch(
    monkeypatch,
) -> None:
    """The SPEC-3 probe's mode is the entry points' ``spec3_probe`` keyword (``"off"`` by default, ``"record"``,
    ``"warn"``, ``"assert"``); no environment variable selects it.  The package's sources read no environment
    variable: the private ``_kcp_test_knobs`` hook opens only after :func:`cake_kcp_plan.enable_test_knobs`."""
    assert cake_kcp._options({}).spec3_probe == "off"
    for mode in cake_kcp.SPEC3_PROBES:
        assert cake_kcp._options({"spec3_probe": mode}).spec3_probe == mode
    for bad in ("0", "1", "", None):
        with pytest.raises(ValueError, match="spec3_probe"):
            cake_kcp._options({"spec3_probe": bad})
    assert not hasattr(cake_kcp, "SPEC3_ENV")
    monkeypatch.setenv("CAKE_KCP_SPEC3", "0")
    assert cake_kcp.KernelBackend()._spec3_pending
    # the earlier environment opt-in of the test hook opens nothing
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", False)
    monkeypatch.setenv("CAKE_KCP_TEST_KNOBS", "1")
    assert not kp.test_knobs_enabled() and not cake_kcp_jit.test_knobs_enabled()
    with pytest.raises(ValueError, match="enable_test_knobs"):
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}})
    assert kp.enable_test_knobs() is False and kp.test_knobs_enabled()
    assert (
        cake_kcp._options({"_kcp_test_knobs": {"precision_mode": 1}}).precision_mode
        == 1
    )
    assert kp.enable_test_knobs(False) is True and not kp.test_knobs_enabled()
    package = os.path.dirname(cake_kcp.__file__)
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(package, name), encoding="utf-8") as f:
            for number, line in enumerate(f, 1):
                assert "os.environ" not in line and "getenv" not in line, (
                    f"{name}:{number}: {line.strip()}"
                )


def test_host_trim_options_passthrough() -> None:
    """Trimmed host path: ``chunk_kda_kcp`` validates the options once; the autograd node passes the validated ``KcpOptions``
    through ``_kcp_options`` and ``_options`` returns it unchanged (it travels alone)."""
    opts = cake_kcp._options({"exchange": "a2a"})
    assert cake_kcp._options({"_kcp_options": opts}) is opts
    with pytest.raises(TypeError):
        cake_kcp._options({"_kcp_options": opts, "exchange": "a2a"})
    with pytest.raises(TypeError):
        cake_kcp._options({"_kcp_options": {"exchange": "a2a"}})


def test_host_trim_workspace_fast_paths() -> None:
    """Trimmed host path: ``ensure(host_trim=True)`` returns the last tables for the same (ctx, H, device, backward, knobs) and
    takes the full path otherwise; ``pool(cached=True)`` returns the same ``[:n]`` view while the pool is unchanged."""
    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    ws = cake_kcp.KcpWorkspace(local=False)
    tb = ws.ensure(ctx, 64, device="cpu", backward=False, host_trim=True)
    assert ws.ensure(ctx, 64, device="cpu", backward=False, host_trim=True) is tb
    assert ws._last_ensure is not None and ws._last_ensure[0] is ctx
    assert not ws.has("b1_send")
    assert (
        ws.ensure(ctx, 64, device="cpu", backward=True, host_trim=True) is tb
    )  # another signature: the full path
    assert ws.has("b1_send")
    assert (
        ws.ensure(ctx, 64, device="cpu", backward=False) is tb
    )  # the untrimmed path never consults the fast path
    n = tb.sizes["c1_send"]
    v1 = ws.pool("c1_send", n, cached=True)
    assert ws.pool("c1_send", n, cached=True) is v1
    v0 = ws.pool("c1_send", n)
    assert v0 is not v1 and v0.data_ptr() == v1.data_ptr() and v0.shape == v1.shape
    ws._pools["c1_send"] = torch.empty_like(
        ws._pools["c1_send"]
    )  # a re-sized pool invalidates the cached view
    v2 = ws.pool("c1_send", n, cached=True)
    assert (
        v2 is not v1
        and v2.data_ptr() == ws._pools["c1_send"].data_ptr()
        and v2.shape == v0.shape
    )


def test_workspace_pools_shared_lazy_and_predicted() -> None:
    """The workspace's KCP pools against the plan module's byte query on the CPU (no local arena): every ``(ctx, H)``
    of the C / Z / D partitions at W = 2 / 4 / 8 / 16, H = 4 / 64, allocates exactly ``kcp_arena_layout(...)["arena"]``
    bytes for a forward and a backward workspace; each :data:`SHARED_POOLS` pair is one tensor of the larger need,
    whichever ensure came first; the ring pools stay unallocated until a call reads them, then hold their size and
    add ``kcp_arena_layout(...)["ring"]`` bytes; the pool sizes are the plan module's (:func:`kcp_pool_numel`)."""
    seen_p = seen_ring = 0
    for W in (2, 4, 8, 16):
        for lengths, layout in (
            ((3000,), "contiguous"),
            ((40, 40), "zigzag"),
            ((700, 1, 1300, 999), "zigzag_per_document"),
        ):
            part = _small_partition(W, lengths=lengths, layout=layout)
            for r, ctx in enumerate(contexts(part)):
                for H in (4, 64):
                    tb = cake_kcp.KcpWorkspace(local=False).tables(ctx, H)
                    assert tb.sizes == kp.kcp_pool_numel(ctx, H=H)
                    seen_p += tb.P > 1
                    full = kp.kcp_arena_layout(ctx, H=H)
                    fwd = kp.kcp_arena_layout(ctx, H=H, backward=False)
                    # forward first, then backward (the pair grows in place of both names)
                    ws = cake_kcp.KcpWorkspace(local=False)
                    ws.ensure(ctx, H, device="cpu", backward=False)
                    assert ws.bytes == fwd["arena"], (W, layout, r, H)
                    assert not ws.has("b1_send") and not ws.has("ring_out")
                    ws.ensure(ctx, H, device="cpu", backward=True)
                    assert ws.bytes == full["arena"], (W, layout, r, H)
                    for a, b in kp.SHARED_POOLS:
                        assert ws.pool(a) is ws.pool(b)
                        assert ws.pool(a).numel() == max(tb.sizes[a], tb.sizes[b])
                    for name in kp.FWD_POOLS + kp.BWD_POOLS:
                        if name not in kp.LAZY_POOLS:
                            assert ws.pool(name).numel() >= tb.sizes[name]
                    assert not ws.has("ring_out") and not ws.has("ring_in")
                    # a backward workspace from the start: the same storage
                    ws2 = cake_kcp.KcpWorkspace(ctx, H, device="cpu", local=False)
                    assert ws2.bytes == full["arena"]
                    assert ws2.pool("c1_send") is ws2.pool("b1_send")
                    # a ring call's first read allocates the hop pools (a zero need allocates nothing)
                    for name in kp.LAZY_POOLS:
                        v = ws.pool(name, tb.sizes[name])
                        assert v.numel() == tb.sizes[name]
                    assert ws.bytes == full["arena"] + full["ring"]
                    seen_ring += full["ring"] > 0
    assert (
        seen_p > 0 and seen_ring > 0
    )  # the rows cover the P > 1 and the crossing-fragment forms


def test_host_trim_backward_walk_sublists_are_table_owned() -> None:
    """Trimmed host path: the backward's tape-replay walk lists ``s % P == 0`` / ``!= 0`` of ``walk_all`` are the table set's own
    list objects (one device copy per table set on the trimmed path, not one per call), and an identity-keyed cache
    fed with fresh objects stays bounded."""
    for lengths, layout in (
        ((3000,), "contiguous"),
        ((700, 1, 1300, 999), "zigzag_per_document"),
        ((40, 40), "zigzag"),
    ):
        part = _small_partition(4, lengths=lengths, layout=layout)
        ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
        tb = cake_kcp._Tables(ctx, 64, fused_walks=1)
        p0, rest = tb.walk_all_p0, tb.walk_all_rest
        assert tb.walk_all_p0 is p0 and tb.walk_all_rest is rest
        assert p0 == [s for s in tb.walk_all if s % tb.P == 0]
        assert rest == [s for s in tb.walk_all if s % tb.P != 0]
        assert sorted(p0 + rest) == sorted(tb.walk_all)
        if p0:
            d0 = tb.dev_list(p0, "cpu")
            assert tb.dev_list(tb.walk_all_p0, "cpu") is d0 and d0.tolist() == p0
        if tb.P == 1:
            assert rest == [] and p0 == list(tb.walk_all)
    # the bound: fresh list objects miss every time and the cache is emptied at ID_CACHE_MAX entries
    keep = []
    for i in range(cake_kcp._Tables.ID_CACHE_MAX + 5):
        fresh = [i, i + 1]
        keep.append(fresh)
        tb.dev_list(fresh, "cpu")
    assert 0 < len(tb._dev_lists) <= cake_kcp._Tables.ID_CACHE_MAX
    assert tb.dev_list(tb.walk_offset, "cpu") is tb.dev_list(tb.walk_offset, "cpu")


def test_host_trim_table_identity_caches() -> None:
    """Trimmed host path: the table set's identity-keyed device copies (walk lists, the segment table, tile tables, Indexed
    metadata, chain tables) hold the same integers as the content-keyed copies of today's path."""
    from flashinfer.experimental.cake_kda_train import cake_kcp_jit

    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    tb = cake_kcp._Tables(ctx, 64, fused_walks=1)
    assert tb.P > 1
    w1 = tb.dev_list(tb.walk_offset, "cpu")
    assert tb.dev_list(tb.walk_offset, "cpu") is w1
    assert w1.dtype == torch.int32 and w1.tolist() == [int(s) for s in tb.walk_offset]
    copy = list(tb.walk_offset)
    w2 = tb.dev_list(copy, "cpu")
    assert w2 is not w1 and torch.equal(
        w2, w1
    )  # another list object: another entry, the same integers
    seg = tb.dev_list(tb.hp.seg_chunk_start, "cpu")
    assert seg.tolist() == [int(x) for x in tb.hp.seg_chunk_start]
    cpu = torch.device("cpu")
    t_id = tb.device_tiles(tb.mf_tiles, cpu, by_identity=True)
    assert tb.device_tiles(tb.mf_tiles, cpu, by_identity=True) is t_id
    assert torch.equal(t_id, tb.device_tiles(tb.mf_tiles, cpu))
    meta = tb.indexed_meta(tb.h0_index, tb.s_in_tiles, 64, "cpu")
    assert tb.indexed_meta(tb.h0_index, tb.s_in_tiles, 64, "cpu") is meta
    plain = cake_kcp.Indexed(
        torch.empty(0),
        tuple(int(i) for i in tb.h0_index),
        tuple(tuple(int(t) for t in row) for row in tb.s_in_tiles),
    )
    assert (meta.index, meta.tiles, meta.addr) == (
        plain.index,
        plain.tiles,
        plain.addressing(64),
    )
    assert meta.dev_index.dtype == torch.int32 and meta.dev_index.tolist() == list(
        plain.index
    )
    ix = cake_kcp.Indexed(torch.empty(0), meta.index, meta.tiles, meta)
    assert ix.addressing(64) == plain.addressing(64) and ix.meta is meta
    assert cake_kcp.Indexed(torch.empty(0), meta.index, meta.tiles).meta is None
    for name, ch in tb.chains.items():
        mine = ch.device_tables("cpu")
        assert ch.device_tables("cpu") is mine
        ref = cake_kcp_jit._chain_tables(ch, "cpu")
        assert set(mine) == set(ref), name
        for k in ref:
            assert mine[k].dtype == ref[k].dtype and torch.equal(mine[k], ref[k]), (
                name,
                k,
            )
    empty = cake_kcp.Chains(8)
    e_mine, e_ref = empty.device_tables("cpu"), cake_kcp_jit._chain_tables(empty, "cpu")
    for k in e_ref:
        assert e_mine[k].dtype == e_ref[k].dtype and torch.equal(e_mine[k], e_ref[k]), k


def test_host_trim_kernel_backend_run_skips_the_discarded_abi() -> None:
    """Trimmed host path: under host_trim a launch goes to the provider's ``run_program`` without the positional ABI (which that
    form discards), keeping the D1 bookkeeping of the merge (alignment check, ``cols_per_cta`` None, ``merge_log``);
    an explicit cols rule takes today's path (ABI + the ``_c<cols>`` record)."""
    calls: list = []

    class FakeJit:
        def run_program(self, name, args, *, device_index):
            calls.append(("run_program", name, args.get("chains"), device_index))

        def run_module(self, record, values, *, grid, device_index):
            calls.append(("run_module", record, grid, device_index))

    ch = cake_kcp.Chains(cake_kcp.D)
    ch.add([(0, -1, -1)], x0=-1, y_final=0)
    big = cake_kcp.Chains(
        cake_kcp.D
    )  # > COLS_RULE_CHAINS state chains: by_chains asks for the explicit c16 record
    for i in range(cake_kcp.COLS_RULE_CHAINS + 1):
        big.add([(4 * i, -1, -1)], x0=-1, y_final=4 * i)
    y, m = torch.zeros(2 * cake_kcp.TILE), torch.zeros(2 * cake_kcp.TILE)

    def call(
        host_trim: int,
        cols_rule: str = "fixed",
        geometry: str = cake_kcp.WALK_GEOMETRY_DEFAULT,
    ):
        return types.SimpleNamespace(
            host_trim=host_trim,
            cols_rule=cols_rule,
            merge_log=[],
            device=torch.device("cpu"),
            walk_geometry=geometry,
            scratch={},
            precision_mode=0,
            transfer_walk="bf16x3_hw64",
        )

    backend = cake_kcp.KernelBackend(programs=cake_kcp._FiPrograms(FakeJit()))
    c1 = call(1)
    args = {
        "family": "fwd",
        "m": m,
        "c": None,
        "x0": None,
        "y": y,
        "chains": ch,
        "transpose_m": 0,
        "write_states": 1,
        "_call": c1,
    }
    backend.run("kcp_affine_merge", args)
    assert calls == [("run_program", "kcp_affine_merge", ch, None)]
    assert (
        args["cols_per_cta"] is None
        and c1.merge_log == [("fwd", cake_kcp.D, 1, None)]
        and id(ch) in backend._checked_chains
    )
    calls.clear()
    c0 = call(0)
    backend.run("kcp_affine_merge", {**args, "_call": c0})
    assert calls == [
        ("run_program", "kcp_affine_merge", ch, None)
    ] and c0.merge_log == [("fwd", cake_kcp.D, 1, None)]
    calls.clear()
    c2 = call(1, cols_rule="by_chains")
    backend.run("kcp_affine_merge", {**args, "chains": big, "_call": c2})
    n_cta = (cake_kcp.COLS_RULE_CHAINS + 1) * (cake_kcp.D // 16)
    assert calls == [("run_module", "kcp_affine_merge_c16", (n_cta, 1, 1), None)]
    assert (
        c2.merge_log == [("fwd", cake_kcp.D, cake_kcp.COLS_RULE_CHAINS + 1, 16)]
        and id(big) in backend._checked_chains
    )
    calls.clear()
    c3 = call(1, geometry="shipped")
    backend.run("kcp_fwdh_tape", {"walk": [0, 1], "h0": None, "out": None, "_call": c3})
    assert calls == [("run_program", "kcp_fwdh_tape", None, None)]


def test_walk_form_of_decides_once_per_table_set() -> None:
    """``_walk_form_of`` (the record form of a K1 / K2 walk launch) evaluates the registry's rule once per
    ``(program, walk list, H, SM count, knob)`` of the call's table set and looks the form up on every later launch:
    the forms of a per-launch evaluation (the rule's inputs are static per ``(ctx, H)``), a fresh equal list finding
    the same entry, the knob's forced forms passing through; a provider without a form rule gives ``"full"``."""
    part = kp.kcp_partition((0, 64 * 200, 64 * 350), 4, layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=148)
    tb = cake_kcp._Tables(ctx, 64)
    evaluated: list[tuple] = []

    def rule(program, chains, num_heads, sm_count, *, knob="auto"):
        evaluated.append((program, tuple(chains), num_heads, sm_count, knob))
        return cake_kcp_jit.walk_form(program, chains, num_heads, sm_count, knob=knob)

    programs = types.SimpleNamespace(
        jit=types.SimpleNamespace(
            walk_form=rule,
            WALK_FORM_RULES=cake_kcp_jit.WALK_FORM_RULES,
            sm_count=lambda index: 148,
        )
    )
    seg = [int(x) for x in tb.hp.seg_chunk_start]
    lists = {
        "offset": tb.walk_offset,
        "in": tb.walk_in,
        "live": tb.walk_live,
        "end": tb.walk_end,
        "first": list(tb.walk_offset[:1]),  # a list unlike the others: its own entry
    }

    def call(knob: str):
        return types.SimpleNamespace(
            tb=tb, H=64, device=torch.device("cpu"), walk_form=knob
        )

    forms: dict[
        tuple, str
    ] = {}  # (knob, program, walk tuple) -> form: one rule evaluation each
    for knob in ("auto", "full", "split"):
        for name in cake_kcp_jit.WALK_FORM_RULES:
            for tag, lst in lists.items():
                if not lst:
                    continue
                want = cake_kcp_jit.walk_form(
                    name, [seg[s + 1] - seg[s] for s in lst], 64, 148, knob=knob
                )
                forms[(knob, name, tuple(lst))] = want
                for _ in range(3):  # the first launch evaluates, the others look up
                    got = cake_kcp._walk_form_of(
                        call(knob), programs, name, {"walk": lst}
                    )
                    assert got == want, (knob, name, tag, got, want)
                fresh = {
                    "walk": [int(s) for s in lst]
                }  # a fresh equal list: the same entry
                assert cake_kcp._walk_form_of(call(knob), programs, name, fresh) == want
    # every distinct (knob, program, walk list) evaluated exactly once (equal lists share the entry)
    assert len(forms) >= 3 * len(cake_kcp_jit.WALK_FORM_RULES) * 2, len(forms)
    assert sorted(evaluated) == sorted(
        (name, tuple(seg[s + 1] - seg[s] for s in walk), 64, 148, knob)
        for (knob, name, walk) in forms
    ), (len(evaluated), len(forms))
    assert all(form == knob for (knob, _, _), form in forms.items() if knob != "auto")
    assert {
        forms[("auto", name, tuple(tb.walk_offset))]
        for name in cake_kcp_jit.WALK_FORM_RULES
    } >= {"split"}  # the tape programs' rule
    bare = types.SimpleNamespace(jit=types.SimpleNamespace())
    assert (
        cake_kcp._walk_form_of(
            types.SimpleNamespace(), bare, "kcp_fwdh_offset", {"walk": tb.walk_offset}
        )
        == "full"
    )


class _FakeGraphRt:
    """A CUDA-runtime double of the capture-graph entry points of the kernel-node cap forms: one
    active capture whose next operation depends on one kernel node; records every call."""

    def __init__(
        self, *, status: int = 1, n_deps: int = 1, node_type: int = 0, rc_set: int = 0
    ) -> None:
        self.status, self.n_deps, self.node_type, self.rc_set = (
            status,
            n_deps,
            node_type,
            rc_set,
        )
        self.calls: list[tuple] = []
        self.values: dict[int, int] = {}
        self.next_node = 0x1000
        self._keep: list = []

    def cudaRuntimeGetVersion(self, ver):
        ver._obj.value = 13000  # the double speaks CUDA 13: the plain export has the 7-argument (v3) form
        return 0

    def cudaStreamGetCaptureInfo(self, stream, status, cid, graph, deps, edges, n_deps):
        node = self.next_node
        self.next_node += 0x10
        status._obj.value = self.status
        cid._obj.value = 7
        graph._obj.value = 0xBEEF
        handle = ctypes.c_void_p(node)
        self._keep.append(
            handle
        )  # the array the runtime would own: valid until the next capture call
        deps._obj.contents = handle
        n_deps._obj.value = self.n_deps
        self.calls.append(("info", stream.value))
        return 0

    def cudaGraphNodeGetType(self, node, node_type):
        node_type._obj.value = self.node_type
        self.calls.append(("type", node.value))
        return 0

    def cudaGraphKernelNodeSetAttribute(self, node, attr, value):
        self.calls.append(("set", node.value, attr, value._obj[0]))
        if self.rc_set:
            return self.rc_set
        self.values[node.value] = value._obj[0]
        return 0

    def cudaGraphKernelNodeGetAttribute(self, node, attr, value):
        self.calls.append(("get", node.value, attr))
        value._obj[0] = self.values.get(node.value, 0xFFFFFFFF)
        return 0

    def cudaGetLastError(self):
        self.calls.append(("lasterror",))
        return 0


def _fake_node_call(form: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        walk_geometry=cake_kcp.WALK_GEOMETRY_DEFAULT,
        scratch={},
        opts=types.SimpleNamespace(graph_cap=form),
    )


def _fake_stream(monkeypatch) -> None:
    monkeypatch.setattr(
        torch.cuda,
        "current_stream",
        lambda device_index=None: types.SimpleNamespace(cuda_stream=0x5151),
    )


def test_capped_launch_sets_the_node_attribute_inside_a_capture(monkeypatch) -> None:
    """The kernel-node cap ``graph_cap="node_attr"``: inside a stream capture a capped program
    launches bare and the kernel node it just added (``cudaStreamGetCaptureInfo``, ``_v2`` on CUDA 12: one dependency node of an active
    capture, type kernel) gets the preferred shared-memory carveout launch attribute ``GRAPH_NODE_CARVEOUT`` with a
    read-back; nothing is counted as a capture skip; the call records the number of capped launches; the evidence log keeps
    the first records per (program, device).  ``"node_attr_post"`` keeps the node handle for :meth:`apply_node_caps` instead,
    which sets the attributes after the capture.  Without the graph entry points both forms report ``unavailable`` and the
    launch falls back to the bare capture-skip form.  A capture-info reply without exactly one kernel-node dependency degrades
    the same way: uncapped launch, one warning per process, the reason on the call."""
    rt = _FakeGraphRt()
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {"rt": rt, "graph": rt})
    monkeypatch.setattr(cake_kcp, "_capturing", lambda: True)
    _fake_stream(monkeypatch)
    stats0 = dict(cake_kcp.CACHE_CONFIG_STATS)
    jit = _FakeJit()
    programs = cake_kcp._FiPrograms(jit)
    program = cake_kcp.WALK_CAP_PROGRAMS[cake_kcp.WALK_GEOMETRY_DEFAULT][0]
    call = _fake_node_call("node_attr")
    programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    assert jit.launched == [(program, 4, 0, 0, "bf16x3_hw64")] * 2
    assert programs.capture_skips == 0 and programs.node_attrs == 2
    assert (
        call.scratch["node_attr_launches"] == 2
        and "node_attr_pending" not in call.scratch
    )
    kinds = [c[0] for c in rt.calls]
    assert kinds == ["info", "type", "set", "get"] * 2, kinds
    sets = [c for c in rt.calls if c[0] == "set"]
    assert all(
        c[2] == cake_kcp._LAUNCH_ATTR_SHMEM_CARVEOUT
        and c[3] == cake_kcp.GRAPH_NODE_CARVEOUT
        for c in sets
    ), sets
    assert [c[1] for c in sets] == [
        0x1000,
        0x1010,
    ]  # one node per launch, the just-captured one
    log = programs.node_attr_log[(program, 0)]
    assert [e["node"] for e in log] == [0x1000, 0x1010]
    assert all(
        e["rc"] == 0
        and e["readback"] == cake_kcp.GRAPH_NODE_CARVEOUT
        and e["node_type"] == 0
        and e["status"] == 1
        and e["n_deps"] == 1
        and e["stream"] == 0x5151
        and e["immediate"] is True
        for e in log
    ), log
    assert (
        cake_kcp.CACHE_CONFIG_STATS["node_attr_launches"]
        == stats0["node_attr_launches"] + 2
    )
    assert cake_kcp.CACHE_CONFIG_STATS["capture_skips"] == stats0["capture_skips"]
    # the evidence log keeps four records per (program, device); the counters keep counting
    for _ in range(4):
        programs.launch(program, [], grid=4, device_index=0, args={"_call": call})
    assert len(programs.node_attr_log[(program, 0)]) == 4 and programs.node_attrs == 6
    # the bracket form of a captured call: node_attr / node_attr_post, or unavailable without the entry points
    cap = types.SimpleNamespace(
        opts=types.SimpleNamespace(graph_cap="node_attr"),
        walk_geometry=cake_kcp.WALK_GEOMETRY_DEFAULT,
    )
    assert cake_kcp._graph_cap_bracket(cap) == ("node_attr", None)
    cap.opts.graph_cap = "node_attr_post"
    assert cake_kcp._graph_cap_bracket(cap) == ("node_attr_post", None)
    cap.walk_geometry = "shipped"
    assert cake_kcp._graph_cap_bracket(cap) == ("none", None)
    # node_attr_post: the handles are kept on the call and set afterwards
    post = _fake_node_call("node_attr_post")
    programs.launch(program, [], grid=4, device_index=0, args={"_call": post})
    programs.launch(program, [], grid=4, device_index=0, args={"_call": post})
    assert (
        post.scratch["node_attr_launches"] == 0
        and len(post.scratch["node_attr_pending"]) == 2
    )
    n_set = len([c for c in rt.calls if c[0] == "set"])
    assert programs.apply_node_caps(post) == 2
    assert len([c for c in rt.calls if c[0] == "set"]) == n_set + 2
    assert (
        post.scratch["node_attr_launches"] == 2
        and "node_attr_pending" not in post.scratch
    )
    assert programs.node_attrs == 8
    # a failing set inside the capture raises (the capture then falls back to the eager path, recorded by _graph_capture)
    bad = _FakeGraphRt(rc_set=1)
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {"rt": bad, "graph": bad})
    programs2 = cake_kcp._FiPrograms(_FakeJit())
    with pytest.raises(RuntimeError, match="cudaGraphKernelNodeSetAttribute"):
        programs2.launch(
            program,
            [],
            grid=4,
            device_index=0,
            args={"_call": _fake_node_call("node_attr")},
        )
    assert ("lasterror",) in bad.calls
    # a capture whose dependency set is not one kernel node DEGRADES: the launch is
    # issued uncapped, one RuntimeWarning per process, the call records the reason -- never an exception / eager fallback
    odd = _FakeGraphRt(n_deps=2)
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {"rt": odd, "graph": odd})
    odd_call = _fake_node_call("node_attr")
    skips0 = programs2.capture_skips
    degraded0 = cake_kcp.CACHE_CONFIG_STATS["node_attr_degraded"]
    with pytest.warns(RuntimeWarning, match="one dependency node"):
        programs2.launch(
            program,
            [],
            grid=4,
            device_index=0,
            args={"_call": odd_call},
        )
    assert programs2.capture_skips == skips0 + 1
    assert cake_kcp.CACHE_CONFIG_STATS["node_attr_degraded"] == degraded0 + 1
    assert "one dependency node" in odd_call.scratch["node_attr_degraded"]
    assert "node_attr_launches" not in odd_call.scratch
    assert not any(c[0] == "set" for c in odd.calls)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # remembered: no second warning in the process
        programs2.launch(
            program,
            [],
            grid=4,
            device_index=0,
            args={"_call": odd_call},
        )
    assert programs2.capture_skips == skips0 + 2 and programs2.node_attrs == 0
    # without the graph entry points: unavailable, and the launch takes the capture-skip form
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {"rt": object(), "graph": None})
    programs3 = cake_kcp._FiPrograms(_FakeJit())
    plain = _fake_node_call("node_attr")
    programs3.launch(program, [], grid=4, device_index=0, args={"_call": plain})
    assert (
        programs3.capture_skips == 1
        and programs3.node_attrs == 0
        and plain.scratch == {}
    )
    cap.walk_geometry = cake_kcp.WALK_GEOMETRY_DEFAULT
    cap.opts.graph_cap = "node_attr"
    assert cake_kcp._graph_cap_bracket(cap) == ("unavailable", None)
    # the shipped forms are untouched by the kernel-node forms (the capture skip)
    none_ = _fake_node_call("none")
    programs3.launch(program, [], grid=4, device_index=0, args={"_call": none_})
    assert programs3.capture_skips == 2 and none_.scratch == {}


def test_graph_entry_points_missing_degrades(monkeypatch) -> None:
    """A CUDA runtime without the capture-graph entry points: one ``RuntimeWarning`` naming them, ``_cudart_graph_available()``
    is ``None`` (remembered), the shipped forms are unaffected."""
    monkeypatch.setattr(cake_kcp, "_CUDART_CACHE", {"rt": object()})
    monkeypatch.setattr(cake_kcp, "_cudart", lambda: types.SimpleNamespace())
    with pytest.warns(RuntimeWarning, match="cudaStreamGetCaptureInfo") as rec:
        assert cake_kcp._cudart_graph_available() is None
    assert len([w for w in rec if issubclass(w.category, RuntimeWarning)]) == 1
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert (
            cake_kcp._cudart_graph_available() is None
        )  # remembered: no second warning
    assert cake_kcp._CUDART_CACHE["graph"] is None


def test_capture_info_export_name_resolution() -> None:
    """CUDA 12.3+ exports the 7-argument stream-capture-info call as ``cudaStreamGetCaptureInfo_v3``, CUDA 11.3-12.x the
    6-argument form as ``..._v2``, and CUDA 13 (13000+) the 7-argument form under the plain name (observed: libcudart.so.13
    exports neither suffixed name); the module resolves the first name present and its arity, and a plain name on a pre-13 runtime
    (the pre-11.3 form without dependency nodes) is unusable."""
    assert cake_kcp._CUDART_CAPTURE_INFO_NAMES == (
        "cudaStreamGetCaptureInfo_v3",
        "cudaStreamGetCaptureInfo_v2",
        "cudaStreamGetCaptureInfo",
    )
    ns = types.SimpleNamespace
    assert (
        cake_kcp._capture_info_name(
            ns(cudaStreamGetCaptureInfo_v2=1, cudaStreamGetCaptureInfo=2)
        )
        == "cudaStreamGetCaptureInfo_v2"
    )
    assert (
        cake_kcp._capture_info_name(
            ns(cudaStreamGetCaptureInfo_v3=1, cudaStreamGetCaptureInfo_v2=1)
        )
        == "cudaStreamGetCaptureInfo_v3"
    )
    assert (
        cake_kcp._capture_info_name(ns(cudaStreamGetCaptureInfo=2))
        == "cudaStreamGetCaptureInfo"
    )
    assert cake_kcp._capture_info_name(ns()) is None
    assert cake_kcp._capture_info_arity(ns(), "cudaStreamGetCaptureInfo_v3") == 7
    assert cake_kcp._capture_info_arity(ns(), "cudaStreamGetCaptureInfo_v2") == 6

    def version(v):
        def fn(ver):
            ver._obj.value = v
            return 0

        return fn

    assert (
        cake_kcp._capture_info_arity(
            ns(cudaRuntimeGetVersion=version(13000)), "cudaStreamGetCaptureInfo"
        )
        == 7
    )
    assert (
        cake_kcp._capture_info_arity(
            ns(cudaRuntimeGetVersion=version(12090)), "cudaStreamGetCaptureInfo"
        )
        == 0
    )
    assert (
        cake_kcp._capture_info_arity(ns(), "cudaStreamGetCaptureInfo") == 0
    )  # no version call: unusable
    assert cake_kcp._CUDART_GRAPH_ENTRY_POINTS == (
        "cudaGraphNodeGetType",
        "cudaGraphKernelNodeSetAttribute",
        "cudaGraphKernelNodeGetAttribute",
    )
    rt = _FakeGraphRt()  # the double speaks CUDA 13
    assert cake_kcp._runtime_version(rt) == 13000
    assert cake_kcp._capture_info_arity(rt, cake_kcp._capture_info_name(rt)) == 7


def test_host_trim_templates_live_on_the_arena_and_weakly_key_the_table_set() -> None:
    """Template-cache ownership: the trimmed path's ``begin_call`` template (the call-independent values of
    one (arena, table set, direction)) is cached on the local arena whose views it holds, keyed by the table set's
    identity with a WEAK reference to it (a hit is valid only while that table set object is alive and identical);
    the backend keeps no template cache, so a table set or an arena dies when its owner drops it -- a training loop
    with a new context / workspace per step retains nothing.  Dead entries are purged on the next miss."""

    class Arena:  # the local arena's surface used by begin_call: plan geometry and the layout views
        def __init__(self, ctx, H: int) -> None:
            self.plan = types.SimpleNamespace(
                S=int(ctx.T_r),
                N=int(ctx.n_frag),
                H=int(H),
                NC_pad=int(ctx.NC_pad),
                R=int(ctx.R),
                NSEG=4,
                P=2,
                seg_arg=2,
                proportional=0,
            )
            n = max(int(ctx.NC_pad), 1)
            self._views = {
                mode: types.SimpleNamespace(
                    h=torch.zeros(n, 2, cake_kcp.D),
                    dh=torch.zeros(n, 2, cake_kcp.D),
                    w=torch.zeros(4),
                )
                for mode in (cake_kcp.LOCAL_LAYOUT_FWD, cake_kcp.LOCAL_LAYOUT_BWD)
            }

        def views(self, mode: str):
            return self._views[mode]

    def call_for(arena, ctx, tb, H: int, backward: bool, host_trim: int = 1):
        return types.SimpleNamespace(
            local=arena,
            ctx=ctx,
            tb=tb,
            T_r=int(ctx.T_r),
            H=int(H),
            backward=backward,
            host_trim=host_trim,
            ops={},
            scale=1.0,
            scratch={},
            local_values=None,
        )

    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctx = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    H = 64
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    arena = Arena(ctx, H)
    backend = cake_kcp.KernelBackend()
    assert not hasattr(
        backend, "_local_templates"
    )  # no template cache on the long-lived backend
    c1 = call_for(arena, ctx, tb, H, False)
    backend.begin_call(c1)
    c2 = call_for(arena, ctx, tb, H, False)
    backend.begin_call(c2)
    assert (
        c1.local_values["h2d"] is c2.local_values["h2d"]
    )  # the cached template: the same view objects per call
    assert c1.local_values is not c2.local_values and c1.scratch[
        "views"
    ] is arena.views("fwd")
    cache = getattr(arena, cake_kcp.TEMPLATE_CACHE_ATTR)
    assert set(cache) == {(id(tb), False)}
    assert (
        isinstance(cache[(id(tb), False)][0], weakref.ref)
        and cache[(id(tb), False)][0]() is tb
    )
    c3 = call_for(arena, ctx, tb, H, True)
    backend.begin_call(c3)
    assert set(cache) == {(id(tb), False), (id(tb), True)} and "dh2d" in c3.local_values
    # the table set dies although the arena and the backend live on: only a weak reference points at it
    wr_tb = weakref.ref(tb)
    del tb, c1, c2, c3
    gc.collect()
    assert wr_tb() is None
    # a new table set on the same arena misses, purges the dead entries and leaves one live entry
    ctx2 = kp.build_kcp_context(part, None, 1, "cpu", 4, sm_count=SM_COUNT)
    tb2 = cake_kcp._Tables(ctx2, H, fused_walks=1)
    c4 = call_for(arena, ctx2, tb2, H, False)
    backend.begin_call(c4)
    assert set(cache) == {(id(tb2), False)} and cache[(id(tb2), False)][0]() is tb2
    # the arena dies with its templates; the backend holds nothing of it
    wr_arena = weakref.ref(arena)
    del arena, c4, cache
    gc.collect()
    assert wr_arena() is None
    # the previous host path (host_trim = 0) builds the values per call and caches nothing on the arena
    arena_b = Arena(ctx2, H)
    c0 = call_for(arena_b, ctx2, tb2, H, False, host_trim=0)
    backend.begin_call(c0)
    assert not hasattr(arena_b, cake_kcp.TEMPLATE_CACHE_ATTR)
    assert c0.local_values["h2d"].data_ptr() == arena_b.views("fwd").h.data_ptr()


LOCAL_RECORD_CASES = [
    (4, "sm_103a", 152),
    (64, "sm_103a", 152),
    (64, "sm_107a", 216),
    (4, "sm_100a", 148),
]


@pytest.mark.parametrize(
    "H,arch,sm",
    LOCAL_RECORD_CASES,
    ids=[f"h{h}-{a}-{s}" for h, a, s in LOCAL_RECORD_CASES],
)
def test_local_stage_values_cover_the_records(H, arch, sm) -> None:
    """The local stages against the existing backend's registry on the CPU: the kernel backend's value dict of a
    forward (no token-row copies: the saved set carries none), a saved-mode backward and a replay backward binds
    every logical name of every
    record it launches -- the plan's records on the forward path without a saved set and on the replay path, the
    direct WY record where the plan fuses the WY representation into a recurrence the KCP does not run, and the
    KCP-owned ``kcp_fwd_o`` / ``kcp_dav`` on their grids; the two backward modes launch the same records on the same
    grids; the intra record that computes the layout itself gets the arena's scratch table set (and the workspace
    bytes count it) while every other program reads the context's tables."""
    part = _small_partition(2)
    ctx = kp.build_kcp_context(part, None, 0, "cpu", 4, sm_count=sm)
    assert ctx.T_r > 0
    plan = cake_backend.plan(ctx.T_r, ctx.n_frag, H, sm, arch)
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    launches: list[tuple[str, str, tuple[int, int, int], dict]] = []

    class Arena:  # the arena's surface used by begin_call / stage: the plan, the layout views, the launcher
        def __init__(self) -> None:
            self.plan = plan
            self.device = torch.device("cpu")
            self.device_index = 0
            self.arch = arch
            self.launcher = self._launch
            self._views: dict = {}

        def views(self, mode: str):
            ns = self._views.get(mode)
            if ns is None:
                ns = types.SimpleNamespace()
                for r in plan.layout(mode):
                    small = (
                        torch.zeros(2, cake_kcp.D)
                        if r.name in ("h", "dh")
                        else torch.zeros(1)
                    )
                    setattr(ns, r.name, small)
                self._views[mode] = ns
            return ns

        def _launch(self, record, values, *, grid, device_index) -> None:
            launches.append(("backend", record, grid, dict(values)))

    class Provider:
        def run_record(self, record, values, *, grid, device_index) -> None:
            launches.append(("kcp", record, grid, dict(values)))

    arena = Arena()
    backend = cake_kcp.KernelBackend(programs=Provider())

    def call_for(backward: bool, *, ops, out=None, grads=None):
        return types.SimpleNamespace(
            local=arena,
            ctx=ctx,
            tb=tb,
            T_r=int(ctx.T_r),
            H=H,
            backward=backward,
            host_trim=1,
            ops=ops,
            scale=0.5,
            scratch={},
            local_values=None,
            out=out,
            grads=grads,
            device=torch.device("cpu"),
            opts=cake_kcp._options({"spec3_probe": "off"}),
        )

    def run(call, stages) -> list:
        backend.begin_call(call)
        for name, extra in stages:
            backend.stage(name, call, **extra)
        done = list(launches)
        launches.clear()
        return done

    ops = {n: torch.zeros(1) for n in ("q", "k", "v", "g", "beta")}
    fwd_call = call_for(False, ops=ops, out=torch.zeros(1))
    fwd = run(fwd_call, (("fwd_intra", {}), ("wy_fwd", {}), ("fwd_o", {})))
    bwd_ops = {**ops, "dout": torch.zeros(1)}
    grads = {n: torch.zeros(1) for n in ("dq", "dk", "dv", "dg", "dbeta")}
    saved = types.SimpleNamespace(S_in=torch.zeros(1), c1_recv=None, M_seg=None)
    bwd_stages = (
        ("fwd_intra", {}),
        ("wy", {}),
        ("dav", {}),
        ("dqkg", {}),
        ("glm_intra", {}),
        ("gate_epilogue", {}),
    )
    bwd = run(
        call_for(True, ops=bwd_ops, grads=grads),
        (("bwd_prep", {"saved": saved}), *bwd_stages),
    )
    replay = run(
        call_for(True, ops=bwd_ops, grads=grads),
        (("bwd_prep", {"saved": None}), *bwd_stages),
    )
    # the two backward modes: the same records on the same grids
    assert [(k, r, g) for k, r, g, _ in replay] == [(k, r, g) for k, r, g, _ in bwd]
    # spec3_probe="off": the saved-mode backward ran no probe and left it pending for a later call
    assert backend._spec3_pending and backend.spec3 is None
    # every launched record of the existing backend's registry finds its logical names in the value dict
    for kind, record, _grid, values in fwd + bwd:
        if kind == "backend":
            missing = set(cake_jit.MODULES[record]["logical"].values()) - set(values)
            assert not missing, (record, sorted(missing))
    # the records of record and their paths
    f, b = cake_kcp.LOCAL_PATH_FWD, cake_kcp.LOCAL_PATH_BWD

    def expect(stage: str, path: str) -> str:
        if stage in cake_kcp.WY_DIRECT_RECORDS and not plan.dispatched(stage, path):
            return cake_kcp.WY_DIRECT_RECORDS[stage]
        return plan.record(stage, path)

    assert [r for _, r, _, _ in fwd] == [
        expect("fwd_intra", f),
        expect("wy_fwd", f),
        "kcp_fwd_o",
    ]
    assert [r for _, r, _, _ in bwd] == [
        expect("bwd_prep", b),
        expect("fwd_intra", b),
        expect("wy", b),
        "kcp_dav",
        expect("dqkg", b),
        expect("glm_intra", b),
        # the gate epilogue launches only where the plan's GLM intra program does not fuse it (sm_100a: glm_intra;
        # sm_103a runs glm_intra_v14 and sm_107a glm_intra_v18, and the stage call launches nothing)
        *([expect("gate_epilogue", b)] if plan.dispatched("gate_epilogue", b) else []),
    ]
    assert plan.dispatched("gate_epilogue", b) == (arch == "sm_100a")
    assert expect("glm_intra", b) == {
        "sm_103a": "glm_intra_v14",
        "sm_107a": "glm_intra_v18",
    }.get(arch, "glm_intra")
    if (
        H == 64 and arch != "sm_107a"
    ):  # the plan fuses the WY representation into its recurrences there
        assert (
            not plan.dispatched("wy_fwd", f)
            and [r for _, r, _, _ in fwd][1] == "wy_fwd"
        )
    # the write switches (no token-row copies on this route: the saved set carries none), the scratch tables and
    # the grids
    intra, bintra = fwd[0][3], bwd[1][3]
    assert (intra["write_ext"], intra["write_int"], intra["write_prep"]) == (0, 1, 1)
    assert (bintra["write_ext"], bintra["write_int"], bintra["write_prep"]) == (0, 1, 0)
    assert (
        intra["aqk_ext"]
        is intra["akk_ext"]
        is arena.views(cake_kcp.LOCAL_LAYOUT_FWD).akk_int
    )
    assert (
        bintra["aqk_ext"]
        is bintra["akk_ext"]
        is arena.views(cake_kcp.LOCAL_LAYOUT_BWD).akk_int
    )
    assert "Aqk" not in fwd_call.scratch and "Akk" not in fwd_call.scratch
    if kp.intra_writes_tables(plan):
        scratch = getattr(arena, cake_kcp.SCRATCH_TABLES_ATTR)
        assert intra["seg_table"] is scratch["seg_table"]
        assert intra["chunk_bos"] is scratch["chunk_bos"]
        assert intra["cu_seqlens"] is ctx.cu_seqlens_local
        assert kp.local_scratch_table_bytes(plan) == plan.table_words() * 4 > 0
    else:
        assert "seg_table" not in intra and "cu_seqlens" not in intra
        assert kp.local_scratch_table_bytes(plan) == 0
    assert (
        fwd[1][3]["chunk_bos"] is ctx.tables["chunk_bos"]
    )  # the WY program reads the context's tables
    assert fwd[2][2] == (max(1, min(plan.NC_pad * H, sm)), 1, 1)
    assert bwd[3][2] == (plan.NC_pad // 2, H, 1)
    for kind, record, grid, _values in fwd + bwd:
        if kind == "backend":
            stage = cake_jit.MODULES[record]["stage"]
            path = f if record in [r for _, r, _, _ in fwd] else b
            if plan.dispatched(stage, path):
                assert grid == plan.grid(stage, path)
            else:
                assert grid == cake_jit.launch_grid(
                    record,
                    num_chunks_pad=plan.NC_pad,
                    num_seqs=plan.N,
                    num_heads=H,
                    sm_count=sm,
                )
    # the workspace prediction counts the scratch set
    scratch_bytes = (
        cake_backend.tables_bytes(ctx.T_r, ctx.n_frag, H, sm_count=sm)
        if kp.intra_writes_tables(plan)
        else 0
    )
    assert kp.kcp_workspace_bytes(ctx, H=H) == (
        cake_backend.workspace_bytes(
            ctx.T_r, ctx.n_frag, H, backward=True, save_mode="light", sm_count=sm
        )
        + scratch_bytes
        + kp.kcp_arena_layout(ctx, H=H)["arena"]
    )
    # the KCP-owned records (the registry's kcp_fwd_o / kcp_dav) find their logical names too
    for kind, record, _grid, values in fwd + bwd:
        if kind == "kcp":
            missing = set(cake_kcp_jit.MODULES[record]["logical"].values()) - set(
                values
            )
            assert not missing, (record, sorted(missing))


def test_gpu_host_trim_fresh_context_per_call_releases_everything(device) -> None:
    """Template-cache ownership (KDA module, one rank): a training loop that builds a NEW context and
    workspace per step must not grow device memory -- after each step's objects are dropped the workspace, its
    local arena and the context are collected (weak references die) and ``torch.cuda.memory_allocated`` returns to
    the level after the first step; the fresh-context results are bitwise the first step's (same kernels, same
    argument order)."""
    backend = cake_kcp.get_backend()
    dt = backend.dtypes
    H = 2
    cu = torch.tensor([0, 300, 1024, 1500], dtype=torch.int64)
    part = kp.kcp_partition(cu, 1)
    T = int(kp.build_kcp_context(part, None, 0, device).T_r)
    gen = torch.Generator(device=device).manual_seed(0x3837)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(
        x,
    ):  # the delta rule's operand contract (harness kcp_inputs): q / k L2-normalized per head
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = unit(randn(T, H, cake_kcp.D)).to(dt["q"])
    k = unit(randn(T, H, cake_kcp.D)).to(dt["k"])
    v = (0.5 * randn(T, H, cake_kcp.D)).to(dt["v"])
    g = (-5.0 * torch.sigmoid(randn(T, H, cake_kcp.D))).to(dt["g"])
    beta = torch.sigmoid(randn(T, H)).to(dt["beta"])
    dout = (0.5 * randn(T, H, cake_kcp.D)).to(dt["dout"])

    def step():
        ctx = kp.build_kcp_context(part, None, 0, device)
        ws = cake_kcp.KcpWorkspace(state_dtype=dt["state"], local=backend.local_stages)
        out, saved = cake_kcp.chunk_kda_kcp_fwd(q, k, v, g, beta, ctx, workspace=ws)
        grads = cake_kcp.chunk_kda_kcp_bwd(
            q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws
        )
        assert cake_kcp.LAST_CALL["bwd"]["host_trim"] == cake_kcp.HOST_TRIM_DEFAULT
        refs = [weakref.ref(ws), weakref.ref(ctx)] + (
            [weakref.ref(ws.local)] if ws.local is not None else []
        )
        return tuple(x.detach().clone() for x in (out, *grads)), refs

    cake_kcp.graph_cache_clear()
    first, refs = step()
    assert all(torch.isfinite(x.float()).all().item() for x in first)
    gc.collect()
    torch.cuda.synchronize(device)
    assert all(r() is None for r in refs), (
        "the first step's workspace / arena / context are retained"
    )
    base = torch.cuda.memory_allocated(device)
    for i in range(3):
        res, refs = step()
        assert all(torch.equal(a, b) for a, b in zip(first, res, strict=True)), i
        del res
        gc.collect()
        torch.cuda.synchronize(device)
        assert all(r() is None for r in refs), (
            i,
            "a fresh context's workspace / arena / context is retained",
        )
        now = torch.cuda.memory_allocated(device)
        assert now == base, (i, now - base)


def test_gpu_workspace_upgrade_releases_superseded_storage_first(device) -> None:
    """A backward ``ensure`` on a forward workspace releases the forward-only local arena and every pool it outgrows
    before allocating their successors: the old and the new storage never coexist, so the upgrade's allocation
    high-water equals its final growth.  A pool growth alone (the shared c1 / b1 send storage outgrown, no arena
    change, a cached view of the old storage present) frees the old storage and peaks at exactly its final allocation.
    Rows: the contiguous and the per-document zigzag partitions at W = 2 / 4, every rank with tokens, H = 4 / 64.  (The
    weakref checks are leak checks; the high-water assertions are the ones a late release fails.)"""
    seen = pooled = 0
    for W, lengths, layout in (
        (2, (3000,), "contiguous"),
        (4, (700, 1, 1300, 999), "zigzag_per_document"),
    ):
        part = _small_partition(W, lengths=lengths, layout=layout)
        for r in range(W):
            ctx = kp.build_kcp_context(part, None, r, device)
            if ctx.T_r == 0:
                continue
            for H in (4, 64):
                ws = cake_kcp.KcpWorkspace()
                ws.ensure(ctx, H, backward=False)
                fwd_arena = weakref.ref(ws.local)
                torch.cuda.synchronize(device)
                base = torch.cuda.memory_allocated(device)
                torch.cuda.reset_peak_memory_stats(device)
                ws.ensure(ctx, H, backward=True)
                torch.cuda.synchronize(device)
                grown = torch.cuda.memory_allocated(device) - base
                peak = torch.cuda.max_memory_allocated(device) - base
                assert ws.local.backward and fwd_arena() is None, (W, layout, r, H)
                assert grown > 0 and peak <= grown + (1 << 20), (
                    W,
                    layout,
                    r,
                    H,
                    peak,
                    grown,
                )
                seen += 1
                n = ws._pools["b1_send"].numel()
                if n > 0:
                    ws.pool(
                        "c1_send", n, cached=True
                    )  # a cached view of the old storage
                    old = weakref.ref(ws._pools["b1_send"])
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                    assert ws._grow("b1_send", 2 * n, ws.device)
                    torch.cuda.synchronize(device)
                    assert ws._pools["c1_send"] is ws._pools["b1_send"]
                    assert old() is None and not ws._slices, (W, layout, r, H)
                    now = torch.cuda.memory_allocated(device)
                    assert torch.cuda.max_memory_allocated(device) == now, (
                        W,
                        layout,
                        r,
                        H,
                        torch.cuda.max_memory_allocated(device) - now,
                    )
                    pooled += 1
                del ws
    assert seen > 0 and pooled > 0, (seen, pooled)


@pytest.mark.parametrize("graph", ["off", "replay"])
def test_gpu_first_training_step_releases_the_forward_arena(device, graph) -> None:
    """First training step on an empty workspace (KDA module, one rank): the forward sizes the workspace for the
    forward only and the backward upgrades it.  Under both ``graph`` settings the first step's allocation high-water
    is no higher than the second step's (the retained workspace plus the step's own peak; both steps run eagerly, as
    no key of the second step was seen before), and the forward-only local arena is collected by the end of the step
    (under ``"replay"`` the upgrade drops the cache entries that pin it, the upgrading call's own entry included).
    Both steps run the shipped defaults, so the SPEC-3 probe stays off (when requested it runs once per backend, on
    the first saved-mode backward, and adds its row copies to that backward's peak)."""
    backend = cake_kcp.get_backend()
    dt = backend.dtypes
    H = 2
    cu = torch.tensor([0, 300, 1024, 1500], dtype=torch.int64)
    part = kp.kcp_partition(cu, 1)
    ctx = kp.build_kcp_context(part, None, 0, device)
    T = int(ctx.T_r)
    gen = torch.Generator(device=device).manual_seed(0x3837)

    def randn(*shape):
        return torch.randn(*shape, device=device, generator=gen, dtype=torch.float32)

    def unit(x):  # the delta rule's operand contract: q / k L2-normalized per head
        return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)

    q = unit(randn(T, H, cake_kcp.D)).to(dt["q"])
    k = unit(randn(T, H, cake_kcp.D)).to(dt["k"])
    v = (0.5 * randn(T, H, cake_kcp.D)).to(dt["v"])
    g = (-5.0 * torch.sigmoid(randn(T, H, cake_kcp.D))).to(dt["g"])
    beta = torch.sigmoid(randn(T, H)).to(dt["beta"])
    dout = (0.5 * randn(T, H, cake_kcp.D)).to(dt["dout"])
    ws = cake_kcp.KcpWorkspace(state_dtype=dt["state"], local=backend.local_stages)

    def step() -> tuple:
        out, saved = cake_kcp.chunk_kda_kcp_fwd(
            q, k, v, g, beta, ctx, workspace=ws, graph=graph
        )
        arena = weakref.ref(ws.local) if ws.local is not None else None
        grads = cake_kcp.chunk_kda_kcp_bwd(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            dout,
            saved=saved,
            workspace=ws,
            graph=graph,
        )
        assert cake_kcp.LAST_CALL["bwd"].get("spec3") is None, (
            "the default backward ran the SPEC-3 probe"
        )
        return tuple(x.detach().clone() for x in (out, *grads)), arena

    cake_kcp.graph_cache_clear()
    gc.collect()
    torch.cuda.synchronize(device)
    try:
        base0 = torch.cuda.memory_allocated(device)
        torch.cuda.reset_peak_memory_stats(device)
        first, fwd_arena = step()
        torch.cuda.synchronize(device)
        hw_first = torch.cuda.max_memory_allocated(device) - base0
        assert all(torch.isfinite(x.float()).all().item() for x in first)
        gc.collect()
        assert fwd_arena is not None and fwd_arena() is None, (
            graph,
            "the forward-only arena outlives the first step",
        )
        assert ws.local is not None and ws.local.backward
        del first
        gc.collect()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        eager = cake_kcp.GRAPH_STATS["eager"]
        second, _ = step()
        torch.cuda.synchronize(device)
        hw_second = torch.cuda.max_memory_allocated(device) - base0
        if graph != "off":
            assert cake_kcp.GRAPH_STATS["eager"] == eager + 2, (
                "the second step did not run eagerly"
            )
        assert hw_first <= hw_second + (1 << 20), (graph, hw_first, hw_second)
        del second
    finally:  # the cached entries pin ws / ctx: never leak them into later tests
        cake_kcp.graph_cache_clear()
