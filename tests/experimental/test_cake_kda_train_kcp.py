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
of the CPU-built chunk / segment tables against the ``table`` / ``seg_table`` programs (``-k gpu``); and the FP64
fake backend against the unpartitioned FP64 reference (``-k fake``).  Rows, reference partitions and operands come
from the reference harness (``kcp_shapes``, ``kcp_inputs``) and lane P's ``reference_kcp`` when they are on
``PYTHONPATH``; the harness-backed tests are skipped otherwise.
"""

import json
import os
import time

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

kp = cake_package("cake_kda_train", "cake_kcp_plan")
cake_backend = cake_package("cake_kda_train", "cake_backend")
cake_jit = cake_package("cake_kda_train", "cake_jit")

try:  # the reference harness (rows of record and the reference partitions)
    import kcp_shapes
except ImportError:  # pragma: no cover - the harness is a lane-environment module
    kcp_shapes = None

HARNESS = pytest.mark.skipif(
    kcp_shapes is None, reason="reference harness kcp_shapes not on PYTHONPATH"
)
SM_COUNT = 152  # GB300; the plan rule's examples of FINAL_DESIGN §7.2
EVIDENCE_DIR = os.environ.get("KCP_EVIDENCE_DIR")
LAYOUT_NAMES = {"C": "contiguous", "Z": "zigzag", "D": "zigzag_per_document"}
OWNERSHIP_H = (1, 3, 4, 6, 12, 64)
OWNERSHIP_W = (2, 4, 8, 16)


def _evidence(name: str, record: dict) -> None:
    """Append one JSON record to ``$KCP_EVIDENCE_DIR/<name>.jsonl`` (the lane's generated evidence)."""
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
        # b1 with maps = pairs [d_f | M_f^T] per crossing-in fragment (P1-R3): send n_in 2s, recv N_in 2s Hg_r/H.
        # No P1-R4 padding on any row of record (H = 4 / 64 with W in {2, 4, 8, 16}).
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
        # P1-R1: every chain-table offset is -1 or a multiple of 4 floats
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
    padded to Hg tiles per slot (P1-R4) so the two-integer K1 / K2 addressing is exact for every H, and the padding
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
        # P1-R4 padding: every owner with heads holds Hg tile slots per fragment slot
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
        assert (
            kp.kcp_saved_bytes(ctx, H=H, with_conv=False, save_transfer=False)
            == 256 * ctx.T_r * H
            + kp.tables_bytes(ctx.NC_pad, ctx.n_frag, hp.NSEG)
            + ctx.n_in * n_owners * Hg * kp.STATE_TILE_BYTES
        )
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
        # explicit tile tables == the two-integer form for every H (P1-R4 padding makes it exact)
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
    """P1-R1: chain-table tile offsets are -1 or non-negative multiples of 4 floats; the checker rejects anything
    else before a launch."""
    kp._check_tile_offsets("ok", (-1, 0, 4, kp.TILE, 3 * kp.SUMMARY_TILE))
    for bad in ((3,), (-2,), (kp.TILE + 2,)):
        with pytest.raises(ValueError, match="P1-R1"):
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
    """Gate 3: the shipped default rule is wave (round-1 adoption of lever L-C), and the §7.2 examples at 152 SMs
    hold under the round-0 occ rule, now selected explicitly."""
    assert kp.P_RULE_DEFAULT == "wave" and kp.P_RULE_DEFAULT in kp.P_RULES
    dflt = kp.kcp_plan(1, 256, 64, SM_COUNT)
    assert dflt == kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="wave")
    # 256 chunks is AT the chain cap, not above it: the wave line does not apply
    # (ruling P2-R9) and the shipped rule gives the occ P
    assert (dflt.p_rule, dflt.P, dflt.P_occ, dflt.P_chain) == ("wave", 3, 3, 1)
    assert dflt.P == kp.kcp_plan(1, 256, 64, SM_COUNT, p_rule="occ").P
    # the first NC above the cap: P_chain = 2 and the wave line applies
    above = kp.kcp_plan(1, 257, 64, SM_COUNT)
    assert (above.p_rule, above.P_chain) == ("wave", 2) and above.P >= 2
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


# Lever L-C (Phase-2 round 1).  Rule of record (rulings P2-R7 / P2-R7a / P2-R9): on ranks with n_frag <= WAVE_N_FRAG_MAX = 3 the "wave"
# rule is argmin_P over P in [P_chain, min(64, NC_max)] (smallest P on ties) of
#   ceil(n_frag P H / SM) x ceil(NC_max / P) x t_walk + (P - 1) x t_compose,  t_walk = 2.85 + 3.5 + 1.2 us, t_compose = 20.4 us
# (10-ns integers); ranks with more fragments, and ranks whose longest fragment is <= 256 chunks (P_chain = 1), keep the occ P.  Information form (b) (ruling P2-R6, superseded): the same
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
# The lane's A/B rows and G1 re-judge rows (row, CP, layout, H): the CPU evidence `plan_wave_lane_rows` records occ / record
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
    """Lever L-C at 152 SMs, H = 64: the rule of record equals the hand-derived argmin on ranks with <= 3 fragments,
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
        )  # short fragments keep the occ P (P2-R9)
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
    assert kp.kcp_plan(1, 488, 64, SM_COUNT, p_rule="wave") == kp.kcp_plan(
        1, 488, 64, SM_COUNT
    )  # the shipped default is the wave rule (round-1 adoption)
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
    """The Q1 C fragment sets at CP4 / CP8 / CP16 (hand-derived from Q1's lengths) and the P of both rules per rank
    through the context (knob p_rule via build_kcp_context; describe() records it; the shipped default is wave):
    every rank has <= 2 fragments: at CP4 / CP8 (fragments > 256 chunks) the rule gives P = 7 everywhere, at CP16
    (<= 244 chunks) the chain condition keeps the occ P; the information form (b) is recorded."""
    row = kcp_shapes.PACKS["Q1"]
    assert list(row.lengths) == [28874, 38814, 60180, 121988]
    _frags, part = harness_partition(row, W, "C")
    expect_occ = {
        4: [3, 4, 4, 4],
        8: [2, 3, 2, 3, 2, 3, 3, 3],
        16: [3, 2, 3, 3, 2, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3],
    }[W]
    monkeypatch.delenv(kp.TEST_KNOBS_ENV, raising=False)
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        kp.build_kcp_context(
            part,
            None,
            0,
            "cpu",
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave"},
        )
    monkeypatch.setenv(kp.TEST_KNOBS_ENV, "1")
    expect_wave = [7] * W if W in (4, 8) else expect_occ
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
        assert (occ.p_rule, wave.p_rule, dflt.p_rule) == ("occ", "wave", "wave")
        assert (dflt.heads(64).P, dflt.describe(64)["p_rule"]) == (
            expect_wave[r],
            "wave",
        )
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
        {"W": W, "occ": expect_occ, "wave": expect_wave, "lpt_info": got_lpt},
    )


@HARNESS
@pytest.mark.parametrize(
    "case", LANE_LC_ROWS, ids=[f"{n}-cp{W}-{lay}-H{H}" for n, W, lay, H in LANE_LC_ROWS]
)
def test_kcp_plan_wave_rule_lane_rows(monkeypatch, case) -> None:
    """Every A/B row and G1 re-judge row of lane LC: per rank the actual fragment table and the P of occ / the rule of
    record ((a) + guard) / unguarded (a) / information (b); the record equals (a) when n_frag <= 3 and occ otherwise,
    the context's plan uses it, (b) equals the oracle.  The evidence record `plan_wave_lane_rows` decides which rows
    measured under another form must be re-run on the rule of record (rulings P2-R6a / P2-R7a)."""
    name, W, lay, H = case
    row = kcp_shapes.ROWS[name]
    _frags, part = harness_partition(row, W, lay)
    monkeypatch.setenv(kp.TEST_KNOBS_ENV, "1")
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
    """chunk_bos / chunk_len / seq_chunk_start / seq_order / seg_chunk_start of the spec formulas; the table layout
    equals the existing backend's with the segment table present."""
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
    assert layout == cake_backend._table_layout(nc_pad, 6, 18)
    assert (
        kp.tables_bytes(nc_pad, 6, 18) == 4 * 64 * 5
    )  # five 256 B-aligned tables of < 64 words each
    assert (
        kp.table_words(0, 0, 0) == 128
    )  # the two one-word tables each occupy a 256 B slot


def test_byte_queries_and_knobs(monkeypatch) -> None:
    """Gate 4: keyword-only signatures; the GOAL §3.2 literal forms work with defaults; save_transfer and P > 1 terms
    per §9; validate_test_knobs refuses a non-zero precision_mode without the environment variable."""
    part = _small_partition(4, lengths=(3000,), layout="contiguous")
    ctxs = contexts(part)
    ctx = ctxs[1]
    hp = ctx.heads(64)
    s = 64 * kp.STATE_TILE_BYTES
    assert kp.SAVE_TRANSFER_DEFAULT is True  # P2-R3 shipped default
    assert kp.kcp_saved_bytes(ctx) == kp.kcp_saved_bytes(ctx, save_transfer=True)
    base = kp.kcp_saved_bytes(
        ctx, save_transfer=False
    )  # literal form: H = 64, light, no transfer, conv, C = 8192
    assert (
        base
        == 256 * ctx.T_r * 64
        + kp.tables_bytes(ctx.NC_pad, ctx.n_frag, hp.NSEG)
        + ctx.n_in * s
        + ctx.halo.n_hist_rows * 8192 * 2
    )
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
        hp.P > 1
    )  # 3000 tokens in one fragment at 152 SMs: P_occ = ceil(152 / 64) = 3
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
    assert (
        cb["b1"]["send"] == ctx.n_in * 2 * s
    )  # saved mode: pairs [d_f | M_f^T] (P1-R3)
    assert cb["padding"]["B1"] == {"send": 0, "recv": 0} and cb["padding"]["B2"] == {
        "send": 0,
        "recv": 0,
    }  # H = 64: no P1-R4 padding
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
    assert kp.kcp_arena_layout(ctx) == kp.kcp_arena_layout(ctx, save_transfer=True)
    arena = kp.kcp_arena_layout(ctx, save_transfer=False)
    assert ws == local + arena["arena"] and arena["arena"] == max(
        arena["forward"], arena["backward"]
    )
    assert (
        arena["fwd.c1_recv"] == hp.numel["c1_recv"] * 4
        and kp.kcp_arena_layout(ctx, save_transfer=True)["fwd.c1_recv"] == 0
    )
    assert (
        arena["fwd.staging"]
        == arena["bwd.staging"]
        == hp.NSEG * 64 * kp.SUMMARY_TILE * 4
    )
    assert (
        kp.kcp_arena_layout(ctx, save_transfer=True)["fwd.staging"]
        == arena["fwd.staging"]
    )
    assert kp.kcp_workspace_bytes(ctx, backward=False, save_transfer=False) <= ws
    assert ws_t <= ws  # the kept c1 tensor leaves the arena (SPEC-4)
    # knobs
    monkeypatch.delenv(kp.TEST_KNOBS_ENV, raising=False)
    assert kp.validate_test_knobs(None) == {"precision_mode": 0, "p_rule": "wave"}
    assert kp.validate_test_knobs(None)["p_rule"] == kp.P_RULE_DEFAULT
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        kp.validate_test_knobs({"precision_mode": 2})
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        kp.validate_test_knobs({"precision_mode": 0})
    with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
        kp.validate_test_knobs({"p_rule": "wave"})
    monkeypatch.setenv(kp.TEST_KNOBS_ENV, "1")
    assert kp.validate_test_knobs({"precision_mode": 1}) == {
        "precision_mode": 1,
        "p_rule": "wave",
    }
    assert kp.validate_test_knobs({}) == {"precision_mode": 0, "p_rule": "wave"}
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


# --------------------------------------------------------------------------- round-1 levers (P2-R1 / P2-R4; lane LAB)

cake_kcp = cake_package("cake_kda_train", "cake_kcp")
LEVER_TENSORS = ("out", "dq", "dk", "dv", "dg", "dbeta")


def test_round1_knobs_are_gated_and_validated(monkeypatch) -> None:
    """``schedule`` (L-A), ``fused_walks`` (L-B) and ``cols_rule`` (D1) are P1-R15-class test knobs: refused
    without ``CAKE_KCP_TEST_KNOBS=1``, never public keywords, validated against their admitted values; the
    shipped defaults since the round-1 adoption are the pair wait_first + fused_walks=1 (one tape launch per
    phase) with the registry's K4 records (cols_rule fixed); overlap / fused_walks=0 stay selectable."""
    monkeypatch.delenv(cake_kcp.TEST_KNOBS_ENV, raising=False)
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
        with pytest.raises(ValueError, match="CAKE_KCP_TEST_KNOBS"):
            cake_kcp._options({"_kcp_test_knobs": knob})
    for key in cake_kcp.P2_KNOBS:
        with pytest.raises(TypeError):
            cake_kcp._options({key: 1})
    monkeypatch.setenv(cake_kcp.TEST_KNOBS_ENV, "1")
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
    # the shipped default since the round-1 adoption
    assert cake_kcp.KcpOptions().save_transfer is True
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


def test_cols_rule_thresholds() -> None:
    """D1 (P2-R4): up to 24 concurrent 128-column state chains keep the registry's c8 record, more take the c16
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
    """The L-B tables of one rank against the separate-walk tables (``fused_walks=0``) of ``cake_kcp``."""
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
    # entering state -- the S_in (dS_out) tiles -- into the segment's slot of the fused table (P1-R1 write_states)
    ins = [f for f in live if f.crossing_in]
    outs = [f for f in live if f.crossing_out]
    for name, c0 in tb0.chains.items():
        c1 = tb1.chains[name]
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
    """L-B (P2-R1 / P2-R4 bundle): with ``fused_walks=1`` every live segment of every rank has a valid initial-state
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
)  # segment counts launched beside the plan's own P (empty segments, P > chunks, P = 1)


@HARNESS
@pytest.mark.parametrize("case", PARITY_CASES, ids=PARITY_IDS)
def test_gpu_table_parity(device, case) -> None:
    """Gate 5 (R15): the CPU-built chunk_bos / chunk_len / seq_chunk_start / seq_order / seg_chunk_start are bitwise
    the kda_chunk_train_layout_table_v1 / seg_table_v1 outputs on every rank of every C row, adversarial row and
    ragged table (one launch of each program per rank and segment count; pad entries pre-filled with a sentinel)."""
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
            seg = torch.full((N * P + 1,), -7, dtype=i32, device=device)
            cake_jit.run_module(
                "seg_table",
                {
                    "seq_chunk_start": scs,
                    "num_seqs": N,
                    "segments": P,
                    "seg_chunk_start": seg,
                },
                grid=(1, 1, 1),
                device_index=device.index,
            )
            torch.cuda.synchronize(device)
            want_seg = kp.seg_chunk_start(list(ctx.seq_chunk_start), P)
            if P == hp.P:
                assert tuple(want_seg) == hp.seg_chunk_start
            equal = seg.cpu().tolist() == want_seg
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
FAKE_REL_RMS = 1e-12  # vs the unpartitioned FP64 reference ("~1e-12", FINAL_DESIGN §11.3); declared before the run
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
    """Gate 6: the FP64 fake backend (W emulated ranks, dict collectives, lane P's per-fragment compute) equals the
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
    """The identity / v = 0 transfer map equals lane P's basis-vector ``transfer`` helper (FP64) on a 63-token fragment."""
    row = kcp_shapes.ROWS["C5"]
    ranks = fake.build_ranks(row, 4, "R", progress=_progress)
    err = fake._transfer_matches_lane_p(ranks.frag_ops(0, 0), row.h)
    _progress(f"transfer (identity, v = 0) vs lane P transfer(): max |diff| {err:.3e}")
    assert err <= 1e-13
    _evidence(
        "fake",
        {"row": "C5", "check": "transfer vs lane P", "max_abs": err, "pass": True},
    )
