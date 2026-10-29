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

CPU tests of the GDN KCP metadata layer of ``flashinfer.experimental.cake_gdn_train``: the package-local plan
module against ``cake_kda_train.cake_kcp_plan`` (field-identical partitions and contexts on the step-0 rows with
``H := HV``; the KDA package is imported HERE only, never by the package), the per-rank crossing counts and
exchange bytes against the rows of record, the table builders (pool addressing, segment lists, chains), head
ownership incl. the padded ``(HV, W) = (3, 2)`` case, the byte queries, the registry seed, and the fake backend
against the FP64 reference (bitwise repeatable, saved == replay).

Environment of the tests (never of the package): ``GDN_ROWS_DIR`` (the step-0 row JSONs; the row tests skip
without it), ``GDN_EVIDENCE_DIR`` (JSONL evidence records), the FP64 reference ``gdn_reference`` on ``PYTHONPATH``
(the fake-backend tests skip without it).
"""

from __future__ import annotations

import ast
import dataclasses
import glob
import hashlib
import importlib
import json
import math
import os
import pathlib
import random
import re

import pytest
import torch


def _package(submodule: str):
    try:
        from tests.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_train", submodule)
    except KeyError:  # the registry entry is added by the package scaffolding lane
        return importlib.import_module(
            f"flashinfer.experimental.cake_gdn_train.{submodule}"
        )


gp = _package("cake_kcp_plan")
gt = _package("cake_tables")
gb = _package("cake_bytes")
gj = _package("cake_jit")
fake = _package("cake_fake_backend")

try:
    import gdn_reference  # the FP64 reference of the validation harness
except ImportError:  # pragma: no cover - optional environment module
    gdn_reference = None

PACKAGE_DIR = pathlib.Path(gp.__file__).resolve().parent
ROWS_DIR = os.environ.get("GDN_ROWS_DIR")
EVIDENCE_DIR = os.environ.get("GDN_EVIDENCE_DIR")
SM_COUNT = 152  # GB300
C_CONV = 10_240  # the rows' convolution width (2 * 16 + 48) * 128
CPU = torch.device("cpu")
OWNERSHIP = [(HV, W) for HV in (3, 4, 6, 12, 24, 48) for W in (2, 4, 8, 16)]
FAKE_TOL_REL_RMS = 0.05  # FLA-class (BF16 tapes) against FP64
FAKE_CU = [
    0,
    100,
    230,
    307,
    507,
]  # four documents, 507 tokens, boundaries inside chunks


def _evidence(name: str, record: dict) -> None:
    if not EVIDENCE_DIR:
        return
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, f"{name}.jsonl"), "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- rows of record


def _rows() -> list[str]:
    if not ROWS_DIR:
        return []
    return sorted(
        p
        for p in glob.glob(os.path.join(ROWS_DIR, "*.json"))
        if not os.path.basename(p).startswith("rows_")
    )


ROW_FILES = _rows()
ROW_IDS = [os.path.basename(p)[: -len(".json")] for p in ROW_FILES]
ROWS = pytest.mark.skipif(
    not ROW_FILES, reason="GDN_ROWS_DIR with the step-0 row JSONs not set"
)


def _ragged_fragments(
    cu: list[int], t_r: list[int]
) -> list[list[tuple[int, int, int]]]:
    """Explicit per-rank fragment tables of a contiguous ragged split (harness ``gdn_shapes.ragged_fragments``,
    copied: the rows' ``fragments=`` input)."""
    out: list[list[tuple[int, int, int]]] = []
    pos = 0
    num_docs = len(cu) - 1
    for n in t_r:
        gs, ge = pos, pos + n
        frs: list[tuple[int, int, int]] = []
        d = 0
        while d < num_docs and cu[d + 1] <= gs:
            d += 1
        p = gs
        while p < ge and d < num_docs:
            e = min(ge, cu[d + 1])
            if e > p:
                frs.append((d, p - cu[d], e - p))
            p = e
            d += 1
        out.append(frs)
        pos = ge
    return out


def _partition_inputs(row: dict) -> dict:
    """The ``kcp_partition`` call of a row JSON (layout C / Z / D / R)."""
    cu = torch.tensor(row["cu_seqlens"], dtype=torch.int64)
    W = int(row["cp"])
    if row["layout"] == "R":
        return {
            "cu_seqlens_global_cpu": cu,
            "world_size": W,
            "fragments": _ragged_fragments(row["cu_seqlens"], row["ragged_table"]),
        }
    if row["layout"] == "D":
        return {
            "cu_seqlens_global_cpu": cu,
            "world_size": W,
            "layout": "zigzag_per_document",
            "cut_points": row["cut_points"],
        }
    return {"cu_seqlens_global_cpu": cu, "world_size": W, "layout": row["fi_layout"]}


def _plain(x):
    """Dataclass / tensor / tuple -> JSON-like plain values (tensors as lists with their dtype)."""
    if isinstance(x, torch.Tensor):
        return {"dtype": str(x.dtype), "shape": list(x.shape), "values": x.tolist()}
    if hasattr(x, "__dataclass_fields__"):
        return {
            f: _plain(getattr(x, f))
            for f in x.__dataclass_fields__
            if not f.startswith("_") and f != "sequence"
        }
    if isinstance(x, dict):
        return {str(k): _plain(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_plain(v) for v in x]
    return x


def _rename_h(x):
    """The KDA module's ``H`` keys -> ``HV`` (the only naming difference between the two modules)."""
    if isinstance(x, dict):
        return {("HV" if k == "H" else k): _rename_h(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_rename_h(v) for v in x]
    return x


def _context_record(ctx, HV: int) -> dict:
    """Every field of a context and its head plan as plain values (the field-identity record)."""
    rec = _plain(ctx)
    rec.pop("partition", None)
    rec.pop("group", None)
    rec["device"] = str(ctx.device)
    rec["heads"] = _plain(ctx.heads(HV))
    rec["describe"] = ctx.describe(HV)
    return rec


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_plan_parity_with_kda_on_rows(path, monkeypatch) -> None:
    """Result 1: ``cake_gdn_train.cake_kcp_plan`` is field-identical to ``cake_kda_train.cake_kcp_plan`` on every
    step-0 row (partition, every rank's context and head plan at ``H := HV``) and reproduces the row's recorded
    fragment table.  Result 2: ``n_in`` / ``n_out`` / ``max_chain_fragments`` and the exchange bytes of
    ``gdn_kcp_comm_bytes`` equal the row JSON.  Both modules plan under the per-rank rule they share,
    ``"wave_fill1"`` (the GDN default is the row rule, the KDA default its own tie rule; each is selected explicitly
    here: the keyword-only ``p_rule`` on the GDN side, the private test hook on the KDA side)."""
    kp = importlib.import_module(
        "flashinfer.experimental.cake_kda_train.cake_kcp_plan"
    )  # the test only
    # the KDA module accepts its private test hook only after its opt-in (enable_test_knobs)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    with open(path) as fh:
        row = json.load(fh)
    H_qk, HV = (int(x) for x in row["heads"])
    W = int(row["cp"])
    args = _partition_inputs(row)
    part = gp.kcp_partition(**args)
    kpart = kp.kcp_partition(**args)
    assert part.to_dict() == kpart.to_dict(), "partition differs from the KDA module"
    assert _plain(part.to_dict()["fragments"]) == row["fragments"], (
        "partition differs from the row of record"
    )  # JSON round trip: the row stores (prev, next) tuples as lists
    assert [list(part.local_cu_seqlens(r)) for r in range(W)] == row["local_cu_seqlens"]
    n_in, n_out = part.crossing_counts()
    assert (
        list(n_in) == row["crossing_in_per_rank"]
        and list(n_out) == row["crossing_out_per_rank"]
    )
    assert part.max_chain_fragments() == row["max_chain_fragments"]
    assert [len(c) for c in part.chains()] == row["chain_fragments_per_doc"]
    kda_bytes = row.get("kda_plan_comm_bytes_H=HV")
    if isinstance(kda_bytes, dict) and "error" in kda_bytes:
        kda_bytes = None
    per_rank, records = [], []
    for r in range(W):
        ctx = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )  # the per-rank rule both modules share; the GDN default "row_cost" is tested on the rows below
        kctx = kp.build_kcp_context(
            kpart,
            None,
            r,
            CPU,
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )
        mine, theirs = _context_record(ctx, HV), _rename_h(_context_record(kctx, HV))
        records.append(mine)
        assert mine == theirs, (
            f"rank {r}: context / head plan differ from the KDA module (HV={HV})"
        )
        assert (ctx.n_in, ctx.n_out) == (n_in[r], n_out[r])
        cb = gb.gdn_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        entry = {
            "rank": r,
            "n_in": ctx.n_in,
            "n_out": ctx.n_out,
            "T_r": ctx.T_r,
            "n_frag": ctx.n_frag,
            "P": ctx.heads(HV).P,
        }
        entry.update({k: cb[k] for k in ("B1", "B2", "B3", "halo_fwd", "halo_bwd")})
        if kda_bytes is not None:
            # the rows were built with the KDA module's context at sm_count=1; the exchange bytes do not depend on P
            assert {
                k: cb[k] for k in ("B1", "B2", "B3", "halo_fwd", "halo_bwd")
            } == kda_bytes[str(r)], (
                f"rank {r}: exchange bytes differ from the row of record"
            )
            entry["rows_match"] = True
        per_rank.append(entry)
    _evidence(
        "parity",
        {
            "row": row["row"],
            "cp": W,
            "layout": row["layout"],
            "fla_padded": row["fla_padded"],
            "H_qk": H_qk,
            "HV": HV,
            "total": row["padded_total"],
            "fragments": sum(len(frs) for frs in part.fragments),
            "n_in": list(n_in),
            "n_out": list(n_out),
            "max_chain_fragments": part.max_chain_fragments(),
            "bytes_rows_match": kda_bytes is not None,
            "per_rank": per_rank,
            "context_sha256": _sha(records),
            "equal": True,
        },
    )


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_tables_on_rows(path) -> None:
    """The table bundles on every step-0 row and rank: pool addressing == the head plan's explicit tile tables,
    chunk tables cover the local sequences, the segment lists / index tables follow the plan, the chains cover
    every document chain once per owned head, device copies have the kernels' dtypes."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.kcp_partition(**_partition_inputs(row))
    chains = part.chains()
    multi = [c for c in chains if len(c) >= 2]
    summary = []
    for r in range(W):
        ctx = gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT)
        hp = ctx.heads(HV)
        lt, wt = gt.local_tables(ctx), gt.walk_tables(ctx, HV)
        assert (
            gt.local_tables(ctx) is lt and gt.walk_tables(ctx, HV) is wt
        )  # cached on the context
        # chunk tables
        assert (
            len(lt.chunk_bos) == len(lt.chunk_len) == lt.NC_pad
            and 64 * lt.NC_pad == lt.R
        )
        assert (
            lt.seq_chunk_start[-1] == lt.NC <= lt.NC_pad
            and len(lt.seq_chunk_start) == ctx.n_frag + 1
        )
        cu = ctx.cu_seqlens_local_cpu
        for s in range(ctx.n_frag):
            ln = cu[s + 1] - cu[s]
            cs = range(lt.seq_chunk_start[s], lt.seq_chunk_start[s + 1])
            assert len(cs) == -(-ln // 64)
            assert sum(lt.chunk_len[c] for c in cs) == ln
            assert all(lt.chunk_bos[c] == cu[s] + 64 * i for i, c in enumerate(cs))
        assert all(lt.chunk_len[c] == 0 for c in range(lt.NC, lt.NC_pad))
        for name in gt.LOCAL_TABLES:
            assert lt.device[name].dtype == torch.int32 and lt.device[
                name
            ].tolist() == list(getattr(lt, name))
        # segment plan and lists
        plan = gp.kcp_plan_row(
            gp.row_tables(part), HV, SM_COUNT, p_rule=gp.P_RULE_DEFAULT
        )[r]
        assert (wt.P, wt.NSEG) == (plan.P, plan.NSEG) and wt.p_rule == gp.P_RULE_DEFAULT
        assert hp.segments.NC_max_frag == plan.NC_max_frag
        assert len(wt.seg_chunk_start) == wt.NSEG + 1 and len(wt.h0_index) == wt.NSEG
        assert tuple(wt.walk_seg_tape) == tuple(range(wt.NSEG))
        if wt.P == 1:
            assert (
                tuple(wt.seg_chunk_start) == tuple(lt.seq_chunk_start)
                or ctx.n_frag == 0
            )
            assert tuple(wt.walk_seg_offset) == tuple(
                f.slot for f in ctx.fragments if f.crossing_out
            )
            assert tuple(wt.h0_index) == tuple(ctx.in_slot) and tuple(
                wt.dht_index
            ) == tuple(wt.out_index) == tuple(ctx.out_slot)
        else:
            assert tuple(wt.walk_seg_offset) == tuple(range(wt.NSEG))
        # pools == explicit tile tables
        for pool, tiles, half in (
            ("s_in", hp.h0_tile, None),
            ("ds_out", hp.dht_tile, None),
            ("b1_d", hp.b1_d_tile_d, None),
            ("c1_b", hp.out_tile, gp.TILE_B),
            ("c1_m", hp.out_tile, gp.TILE_M),
            ("b1_d_dm", hp.b1_d_tile_dm, "elem"),
            ("b1_m_dm", hp.b1_m_tile_dm, "elem"),
        ):
            pa = wt.pools[pool]
            for slot, row_tiles in enumerate(tiles):
                for h in range(HV):
                    if half is None or half == "elem":
                        assert pa.elem(slot, h) == row_tiles[h] * gp.TILE
                    else:
                        assert pa.elem(slot, h) == (row_tiles[h] * 2 + half) * gp.TILE
            assert pa.numel == sum(hp.splits[gt.POOLS[pool][0]])
        # chains
        fwd = wt.chains["fwd"]
        assert fwd.n_chains == len(multi) * hp.Hg_r
        assert fwd.n_elems == sum(len(c) - 2 for c in multi) * hp.Hg_r
        for fam in ("adj_b1", "adj_c1"):
            assert (
                wt.chains[fam].n_chains == fwd.n_chains
                and wt.chains[fam].n_elems == fwd.n_elems
            )
        for name in gt.WALK_TABLES:
            assert wt.device[name].dtype == torch.int32
        for fam in gt.CHAIN_FAMILIES:
            assert wt.device[f"{fam}.chain_start"].dtype == torch.int32
            for name in ("elem_m", "elem_c", "elem_y", "x0_index", "y_final"):
                assert wt.device[f"{fam}.{name}"].dtype == torch.int64
        summary.append(
            {
                "rank": r,
                "NC": lt.NC,
                "P": wt.P,
                "NSEG": wt.NSEG,
                "n_walk_offset": wt.n_walk_offset,
                "fwd_chains": fwd.n_chains,
                "fwd_elems": fwd.n_elems,
            }
        )
    _evidence(
        "tables",
        {
            "row": row["row"],
            "cp": W,
            "layout": row["layout"],
            "HV": HV,
            "per_rank": summary,
            "pass": True,
        },
    )


# --------------------------------------------------------------------------- ownership, tables, bytes (no rows needed)


def _small_partition(W: int, lengths=(300, 70, 1000, 5), layout="contiguous"):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    return gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), W, layout)


@pytest.mark.parametrize("HV,W", OWNERSHIP, ids=[f"HV{h}-W{w}" for h, w in OWNERSHIP])
def test_owner_table(HV, W) -> None:
    """Head ownership: ``Hg = ceil(HV / W)``, ``owner(h) = h // Hg``, ``Hg_r = clamp(HV - r Hg, 0, Hg)``; the single
    partial owner is padded to ``Hg`` tile slots (``(3, 2)``: ``Hg = 2``, owners ``(2, 1)``, padded owner 1, one pad
    tile per slot); the padding is counted in the splits and reported by the byte query."""
    ot = gt.owner_table(HV, W)
    Hg = -(-HV // W)
    assert ot["Hg"] == Hg and ot["n_owners"] == -(-HV // Hg) and sum(ot["Hg_r"]) == HV
    assert ot["owner_of_head"] == [h // Hg for h in range(HV)]
    assert all(0 <= x <= Hg for x in ot["Hg_r"]) and all(
        x == Hg for x in ot["Hg_r"][: ot["n_owners"] - 1]
    )
    partial = [j for j, x in enumerate(ot["Hg_r"]) if 0 < x < Hg]
    assert (ot["padded_owner"], ot["pad_tiles"]) == (
        (partial[0] if partial else -1),
        ot["n_owners"] * Hg - HV,
    )
    assert ot["full_groups"] == (not partial)
    if (HV, W) == (3, 2):
        assert (
            ot["Hg_r"] == [2, 1]
            and ot["padded_owner"] == 1
            and ot["pad_tiles"] == 1
            and ot["Hg_pad_r"] == [2, 2]
        )
    part = _small_partition(W, layout="zigzag")
    ctxs = [
        gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT) for r in range(W)
    ]
    for r, ctx in enumerate(ctxs):
        hp = ctx.heads(HV)
        assert (hp.Hg, hp.n_owners, hp.Hg_r, hp.padded_owner, hp.pad_tiles) == (
            Hg,
            ot["n_owners"],
            ot["Hg_r"][r],
            ot["padded_owner"],
            ot["pad_tiles"],
        )
        assert (
            sum(hp.splits["c1_send"])
            == ctx.n_out * ot["n_owners"] * Hg * gp.SUMMARY_TILE
        )
        cb = gb.gdn_kcp_comm_bytes(ctx, H_qk=1, HV=HV, C=C_CONV)
        assert (
            cb["padding"]["padded_owner"] == ot["padded_owner"]
            and cb["padding"]["pad_tiles_per_slot"] == ot["pad_tiles"]
        )
        pad_b1 = cb["padding"]["B1"]["send"] + cb["padding"]["B1"]["recv"]
        if ot["pad_tiles"] == 0:
            assert pad_b1 == 0 and cb["padding"]["B2"] == {"send": 0, "recv": 0}
        elif ctx.n_in + ctx.n_out > 0:
            assert pad_b1 > 0
        for i in range(W):
            assert ctxs[i].heads(HV).splits["c1_send"][r] == hp.splits["c1_recv"][i]
    _evidence("ownership", {**ot, "pass": True})


def test_prepare_tables_first_use() -> None:
    """``ctx.prepare(H_qk, HV)`` builds the device-resident bundle once per ``(HV, sm_count, P)``
    key at first use (``build_kcp_context`` builds no head-dependent table; the second call returns the same
    object; a ``P`` override gets its own key and plan), rejects ``HV % H_qk != 0`` and an out-of-range ``P``, and
    every table dtype is int32 except the merge's int64 element offsets."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)  # C1
    part = gp.kcp_partition(cu, 2, "contiguous")
    ctx = gp.build_kcp_context(part, None, 1, CPU, 4, sm_count=SM_COUNT)
    assert ctx._head_plans == {} and not ctx._caches
    pt = ctx.prepare(2, 6)
    assert pt is ctx.prepare(2, 6) and pt.key == (6, SM_COUNT, pt.P) and pt.H_qk == 2
    assert pt.walk is gt.walk_tables(ctx, 6) and pt.local is gt.local_tables(ctx)
    assert set(pt.device) >= set(gt.LOCAL_TABLES) | set(gt.WALK_TABLES)
    assert any(n.endswith(".elem_m") for n in pt.device)
    for name, tensor in pt.device.items():
        assert tensor.dtype == gt.table_dtype(name), name
        assert (tensor.dtype == torch.int64) == (
            name.rsplit(".", 1)[-1] in gt.INT64_TABLES
        ), name
    with pytest.raises(ValueError, match="multiple of H_qk"):
        ctx.prepare(4, 6)
    p2 = ctx.prepare(2, 6, P=2)
    assert p2 is not pt and p2.key == (6, SM_COUNT, 2) and p2.P == 2
    assert 2 * ctx.n_frag == p2.walk.NSEG and p2.walk.p_rule == "override"
    assert ctx.heads(6).P == pt.P  # the rule's plan is untouched by the override
    assert (
        ctx.prepare(2, 6, P=pt.P) is pt
    )  # an override equal to the rule's P is the same key
    with pytest.raises(ValueError, match="override must be in"):
        ctx.prepare(2, 6, P=0)
    with pytest.raises(ValueError, match="override must be in"):
        ctx.prepare(2, 6, P=65)
    _evidence(
        "prepare",
        {
            "key": list(pt.key),
            "P_override": p2.P,
            "n_tables": len(pt.device),
            "int64_tables": sorted(
                n for n, x in pt.device.items() if x.dtype == torch.int64
            ),
            "pass": True,
        },
    )


def test_zero_token_rank_and_small_fragments() -> None:
    """C4b-like explicit table (a rank with 0 tokens) and C5-like 63 / 64 / 65-token fragments: tables and pools."""
    cu = torch.tensor([0, 1700, 3000], dtype=torch.int64)
    part = gp.kcp_partition(
        cu, 4, fragments=_ragged_fragments([0, 1700, 3000], [1200, 0, 900, 900])
    )
    ctxs = [
        gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT) for r in range(4)
    ]
    assert ctxs[1].T_r == 0 and ctxs[1].n_frag == 0 and gt.local_tables(ctxs[1]).NC == 0
    wt1 = gt.walk_tables(ctxs[1], 24)
    assert wt1.NSEG == 0 and wt1.n_walk_offset == 0 and wt1.Hg_r == 6  # still an owner
    assert (
        wt1.chains["fwd"].n_chains == 2 * wt1.Hg_r
    )  # two document chains of 2 fragments (ranks 0 -> 2, 2 -> 3)
    cu5 = torch.tensor([0, 63 + 64 + 65 + 500], dtype=torch.int64)
    part5 = gp.kcp_partition(
        cu5, 4, fragments=_ragged_fragments([0, 692], [63, 64, 65, 500])
    )
    for r, want in enumerate((1, 1, 2, 8)):
        ctx = gp.build_kcp_context(part5, None, r, CPU, 4, sm_count=SM_COUNT)
        lt = gt.local_tables(ctx)
        assert want == lt.NC and lt.chunk_len[0] == min(64, ctx.T_r)
    with pytest.raises(ValueError, match="p_rule"):
        gp.build_kcp_context(part5, None, 0, CPU, 4, sm_count=SM_COUNT, p_rule="nope")


# --------------------------------------------------------------------------- the row rules ("row_cost_lpt" shipped, "row_cost" selectable)

ROW_COST_CHANGED = {  # (row, cp, layout) -> the P vector the "row_cost" row rule picks where it departs from "wave_fill1"
    ("P1", 2, "D"): (1, 1),
    ("P2", 2, "Z"): (4, 4),
    # P3 CP2 C: admitted by the "one" margin of 0 (model ratio 0.980; "occ" keeps its 2 %)
    ("P3", 2, "C"): (1, 1),
    ("P1", 4, "D"): (1, 1, 1, 1),
}

# SM_COUNT = 152 (GB300; the GB200 of record reports 152 too).  Other SM counts have no fixture: the census's model-only
# columns are not pinned.  Source: generated by r13_mod13_census.py 2026-10-28T03:09:15Z from the plan module sha256/16 df8e02283700caa3 (rule row_cost_lpt, HV 48, SM 152); never retyped
ROW_COST_LPT_CHANGED = {  # (row, cp, layout) -> the P vector the shipped "row_cost_lpt" rule picks where it departs from "row_cost"
    ("P1", 4, "Z"): (2, 2, 3, 3),
    ("P3", 4, "C"): (6, 6, 6, 6),
    ("P3", 4, "Z"): (3, 3, 3, 3),
    ("P5", 4, "C"): (3, 6, 6, 6),
    ("P5", 4, "Z"): (3, 3, 3, 3),
}
# fragment tables of record (chunk count, crosses out) per rank, HV = 48 heads, SM_COUNT SMs
TABLE_P1_CP4_D = (
    (
        (258, True),
        (234, False),
        (61, True),
        (62, False),
        (68, True),
        (67, False),
        (30, True),
        (30, False),
        (47, True),
        (45, False),
        (21, True),
        (21, False),
    ),
    (
        (250, True),
        (258, True),
        (67, True),
        (66, True),
        (65, True),
        (69, True),
        (32, True),
        (32, True),
        (45, True),
        (44, True),
        (23, True),
        (24, True),
    ),
    (
        (228, True),
        (243, True),
        (69, True),
        (69, True),
        (69, True),
        (61, True),
        (29, True),
        (28, True),
        (42, True),
        (46, True),
        (22, True),
        (19, True),
    ),
    (
        (261, True),
        (249, True),
        (63, True),
        (59, True),
        (63, True),
        (69, True),
        (31, True),
        (29, True),
        (46, True),
        (48, True),
        (21, True),
        (22, True),
    ),
)
TABLE_P1_CP2_C = (
    ((1890, True),),
    ((89, False), (512, False), (527, False), (236, False), (360, False), (169, False)),
)
TABLE_P3_CP2_C = (
    ((2003, False), (23, True)),
    ((2025, False),),
)


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_row_cost_rule_on_rows_of_record(path) -> None:
    """The ``"row_cost"`` row rule (selectable; the base the shipped ``"row_cost_lpt"`` starts from) on every step-0 row:
    the four rows of ``ROW_COST_CHANGED`` get their recorded vector, every other row keeps the per-rank ``"wave_fill1"``
    vector; every rank's context derives the same vector (same topology, same ``P`` on every rank) and reports
    ``p_rule="row_cost"``."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.kcp_partition(**_partition_inputs(row))
    tables = gp.row_tables(part)
    assert len(tables) == W and all(
        len(t) == len(frs) for t, frs in zip(tables, part.fragments, strict=True)
    )
    plans = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost")
    got = tuple(seg.P for seg in plans)
    shipped = tuple(
        gp.kcp_plan(
            len(t),
            max((nc for nc, _ in t), default=0),
            HV,
            SM_COUNT,
            p_rule="wave_fill1",
        ).P
        for t in tables
    )
    key = (row["row"], W, row["layout"])
    want = ROW_COST_CHANGED.get(key, shipped)
    assert got == want, (key, got, want, shipped)
    assert (got != shipped) == (key in ROW_COST_CHANGED)
    assert all(
        seg.p_rule == "row_cost" and seg.n_frag * seg.P == seg.NSEG for seg in plans
    )
    for r in range(W):
        ctx = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        hp = ctx.heads(HV)
        assert ctx.p_rule == "row_cost" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        # only the segment program differs from the per-rank plan: partition, fragment table, local tables and
        # crossing lists are the same object graph values
        ref = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )
        assert ctx.partition.to_dict() == ref.partition.to_dict() == part.to_dict()
        assert ctx.frag_table.tolist() == ref.frag_table.tolist()
        assert ctx.cu_seqlens_local_cpu == ref.cu_seqlens_local_cpu
        assert (ctx.in_slot, ctx.out_slot, ctx.chains) == (
            ref.in_slot,
            ref.out_slot,
            ref.chains,
        )
        assert (ctx.NC, ctx.NC_max, ctx.NC_pad, ctx.R) == (
            ref.NC,
            ref.NC_max,
            ref.NC_pad,
            ref.R,
        )
        assert hp.segments.NC_max_frag == ref.heads(HV).segments.NC_max_frag
        assert (hp.P == ref.heads(HV).P) == (got[r] == shipped[r])
    _evidence(
        "row_cost",
        {
            "row": row["row"],
            "cp": W,
            "layout": row["layout"],
            "HV": HV,
            "sm_count": SM_COUNT,
            "P_shipped": list(shipped),
            "P_row_cost": list(got),
            "changed": got != shipped,
        },
    )


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_row_cost_lpt_rule_on_rows_of_record(path) -> None:
    """The shipped row rule (``"row_cost_lpt"``, the default of ``build_kcp_context``) on every step-0 row: the five
    rows of ``ROW_COST_LPT_CHANGED`` get their recorded vector, every other row keeps the ``"row_cost"`` vector; a rank
    the ``"row_cost"`` rule plans at ``P = 1`` keeps ``P = 1``; every rank's default context derives the same vector and
    reports ``p_rule="row_cost_lpt"``; only the segment program differs from the ``"row_cost"`` context."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.kcp_partition(**_partition_inputs(row))
    tables = gp.row_tables(part)
    base = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost")
    plans = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost_lpt")
    got = tuple(seg.P for seg in plans)
    row_cost = tuple(seg.P for seg in base)
    key = (row["row"], W, row["layout"])
    want = ROW_COST_LPT_CHANGED.get(key, row_cost)
    assert got == want, (key, got, want, row_cost)
    assert (got != row_cost) == (key in ROW_COST_LPT_CHANGED)
    assert all(
        seg.p_rule == "row_cost_lpt" and seg.n_frag * seg.P == seg.NSEG for seg in plans
    )
    assert all(new.P == 1 for old, new in zip(base, plans, strict=True) if old.P == 1)
    for r in range(W):
        ctx = gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT)
        hp = ctx.heads(HV)
        assert ctx.p_rule == gp.P_RULE_DEFAULT == "row_cost_lpt" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        ref = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        assert ctx.partition.to_dict() == ref.partition.to_dict() == part.to_dict()
        assert ctx.frag_table.tolist() == ref.frag_table.tolist()
        assert ctx.cu_seqlens_local_cpu == ref.cu_seqlens_local_cpu
        assert (ctx.in_slot, ctx.out_slot, ctx.chains) == (
            ref.in_slot,
            ref.out_slot,
            ref.chains,
        )
        assert (ctx.NC, ctx.NC_max, ctx.NC_pad, ctx.R) == (
            ref.NC,
            ref.NC_max,
            ref.NC_pad,
            ref.R,
        )
        assert hp.segments.NC_max_frag == ref.heads(HV).segments.NC_max_frag
        assert (hp.P == ref.heads(HV).P) == (got[r] == row_cost[r])
    _evidence(
        "row_cost_lpt",
        {
            "row": row["row"],
            "cp": W,
            "layout": row["layout"],
            "HV": HV,
            "sm_count": SM_COUNT,
            "P_row_cost": list(row_cost),
            "P_row_cost_lpt": list(got),
            "changed": got != row_cost,
        },
    )


def test_row_cost_rule_is_the_default_and_the_per_rank_rules_stay_selectable() -> None:
    """``P_RULE_DEFAULT`` is the ``"row_cost_lpt"`` row rule (``"row_cost"`` stays selectable); ``kcp_plan`` refuses
    both (they need every rank's table); the per-rank rules
    stay selectable through ``p_rule`` and then drive the head plan; the row rule picks the recorded vectors on three
    tables of record (two flip to ``"one"`` -- one far below any margin, one at a model ratio of 0.980 that only the
    ``"one"`` margin of 0 admits -- and one keeps the base far from any margin)."""
    assert gp.P_RULE_DEFAULT == "row_cost_lpt" and "row_cost_lpt" in gp.P_RULES
    assert "row_cost" in gp.P_RULES
    assert gp.ROW_COST_BASE == "wave_fill1" and gp.ROW_COST_POLICIES == ("occ", "one")
    assert gp.ROW_COST_COMPOSE_K == 2 and gp.ROW_COST_MARGIN_PCT == {"occ": 2, "one": 0}
    with pytest.raises(ValueError, match="row_cost"):
        gp.kcp_plan(2, 500, 48, SM_COUNT, p_rule="row_cost")
    with pytest.raises(ValueError, match="row_cost_lpt"):
        gp.kcp_plan(2, 500, 48, SM_COUNT, p_rule="row_cost_lpt")
    HV = 48

    def base(tables):
        return tuple(
            gp.kcp_plan(
                len(t),
                max((nc for nc, _ in t), default=0),
                HV,
                SM_COUNT,
                p_rule="wave_fill1",
            ).P
            for t in tables
        )

    def row(tables):
        return tuple(
            seg.P for seg in gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost")
        )

    assert row(TABLE_P1_CP4_D) == (1, 1, 1, 1) and base(TABLE_P1_CP4_D) == (2, 2, 1, 2)
    assert row(TABLE_P1_CP2_C) == base(TABLE_P1_CP2_C) == (19, 3)
    assert row(TABLE_P3_CP2_C) == (1, 1) and base(TABLE_P3_CP2_C) == (11, 19)
    # the per-policy margin on the third table: "one" is cheaper than the base by less than the "occ" margin (2 %)
    # and flips only because the "one" margin is 0; "occ" (8, 8) is costlier than the base there and never flips
    c_base = gp.row_cost_10ns(TABLE_P3_CP2_C, HV, SM_COUNT, (11, 19))
    c_one = gp.row_cost_10ns(TABLE_P3_CP2_C, HV, SM_COUNT, (1, 1))
    c_occ = gp.row_cost_10ns(TABLE_P3_CP2_C, HV, SM_COUNT, (8, 8))
    assert isinstance(c_base, int) and isinstance(c_one, int)
    assert 98 * c_base < 100 * c_one < 100 * c_base
    assert 100 * c_one <= (100 - gp.ROW_COST_MARGIN_PCT["one"]) * c_base
    assert 100 * c_one > (100 - gp.ROW_COST_MARGIN_PCT["occ"]) * c_base
    assert c_occ > c_base
    # contexts: the default rule on every rank gives the row's vector; the selectable row rule and a per-rank rule
    # are honoured when asked for
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    part = gp.kcp_partition(cu, 2, "contiguous")
    want = tuple(
        seg.P
        for seg in gp.kcp_plan_row(
            gp.row_tables(part), 6, SM_COUNT, p_rule="row_cost_lpt"
        )
    )
    want_old = tuple(
        seg.P
        for seg in gp.kcp_plan_row(gp.row_tables(part), 6, SM_COUNT, p_rule="row_cost")
    )
    for r in range(2):
        ctx = gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT)
        assert ctx.p_rule == "row_cost_lpt" and want[r] == ctx.heads(6).P
        assert ctx.heads(6).segments.p_rule == "row_cost_lpt"
        old = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        assert old.p_rule == "row_cost" and want_old[r] == old.heads(6).P
        assert old.heads(6).segments.p_rule == "row_cost"
        occ = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="occ"
        )
        seg = occ.heads(6).segments
        nc_max = max((nc for nc, _ in gp.row_tables(part)[r]), default=0)
        assert (
            seg.p_rule == "occ"
            and seg.P == gp.kcp_plan(occ.n_frag, nc_max, 6, SM_COUNT, p_rule="occ").P
        )
    # zero-token rank: an empty table costs nothing and plans P = 1, NSEG = 0
    for rule in ("row_cost", "row_cost_lpt"):
        plans = gp.kcp_plan_row(((), ((300, True),)), HV, SM_COUNT, p_rule=rule)
        assert (plans[0].P, plans[0].NSEG, plans[0].n_frag) == (1, 0, 0)
    assert gp.row_cost_10ns(
        ((), ((300, True),)), HV, SM_COUNT, (1, 1)
    ) == gp.row_cost_rank_10ns(((300, True),), HV, SM_COUNT, 1)


def test_row_cost_model_is_integer_and_crossing_aware() -> None:
    """The cost model: at ``P = 1`` a fragment that does not cross out pays the tape walk only, a crossing-out one the
    three walks; at ``P > 1`` both pay the three walks plus the weighted compose term; costs are exact integers in the
    plan's 10 ns units; the row cost is the slowest rank's."""
    T = gp.WAVE_RULE_10NS
    HV, SM = 48, SM_COUNT
    walk3 = T["tape"] + T["xfer"] + T["offset"]
    assert gp.row_cost_rank_10ns(((100, False),), HV, SM, 1) == 100 * T["tape"]
    assert gp.row_cost_rank_10ns(((100, True),), HV, SM, 1) == 100 * walk3
    two = (
        50 * walk3 + 1 * T["compose"] * 1 * gp.ROW_COST_COMPOSE_K
    )  # 96 chains: one wave
    assert (
        gp.row_cost_rank_10ns(((100, False),), HV, SM, 2)
        == gp.row_cost_rank_10ns(((100, True),), HV, SM, 2)
        == two
    )
    assert (
        gp.row_cost_rank_10ns((), HV, SM, 1)
        == 0
        == gp.row_cost_rank_10ns(((0, True),), HV, SM, 1)
    )
    # an empty fragment still pays the compose term at P > 1 (the plan never picks P > 1 for it)
    assert (
        gp.row_cost_rank_10ns(((0, True),), HV, SM, 3)
        == 2 * T["compose"] * gp.ROW_COST_COMPOSE_K
    )
    # more chains than SMs: the LPT makespan of 4 x 48 = 192 chains of 100 steps on 152 SMs is 200 steps
    assert gp.row_cost_rank_10ns(((100, True),) * 4, HV, SM, 1) == 200 * walk3
    # the row cost is the maximum over the ranks
    tables = (((100, True),), ((100, True),) * 4)
    assert gp.row_cost_10ns(tables, HV, SM, (1, 1)) == 200 * walk3
    for v in (
        gp.row_cost_rank_10ns(((257, True), (30, False)), HV, SM, 3),
        gp.row_cost_10ns(tables, HV, SM, (2, 2)),
    ):
        assert isinstance(v, int)


def _saved_formula(ctx, H_qk, HV, save_mode, boundary_mode, save_transfer, C) -> int:
    hp = ctx.heads(HV)
    s = HV * gp.STATE_TILE_BYTES
    s_pad = hp.n_owners * hp.Hg * gp.STATE_TILE_BYTES
    total = (
        ctx.halo.n_hist_rows * C * 2
    )  # light: S_in / c1 / M_seg / conv history only, no tape
    if boundary_mode == "saved":
        total += ctx.n_in * s_pad
        if save_transfer:
            total += ctx.N_out * 2 * hp.Hg_pad_by_rank[ctx.rank] * gp.STATE_TILE_BYTES
            if hp.P > 1:
                total += (hp.P - 1) * ctx.n_frag * s
    if (
        save_mode == "full"
    ):  # every forward tape: h, v_new / w / u, Aqk / Akk, gamma, gcum / rowscale / qscale
        total += (
            ctx.NC_pad * HV * (32768 + 4) + 3 * ctx.R * HV * 256 + 2 * ctx.R * HV * 128
        )
        total += 3 * ctx.R * HV * 4
    return total


@pytest.mark.parametrize("W", (2, 4))
def test_byte_queries(W) -> None:
    """Result 3: the byte queries follow the design formulas for both ``save_mode`` and ``boundary_mode`` values,
    ``save_transfer`` on / off; the exchange terms depend on ``HV`` only, ``H_qk`` enters through the default
    channel count; the validators reject ``HV % H_qk != 0``."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)  # C1
    part = gp.kcp_partition(cu, W, "contiguous")
    H_qk, HV = 2, 6
    s = HV * gp.STATE_TILE_BYTES
    records = []
    b1_sums = {"design": 0, "buffers": 0}
    for r in range(W):
        ctx = gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT)
        hp = ctx.heads(HV)
        cb = gb.gdn_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        assert (
            cb["C"] == C_CONV
            and gb.gdn_kcp_comm_bytes(ctx, H_qk=16, HV=48)["C"] == (2 * 16 + 48) * 128
        )
        assert (
            cb["c1"]["send"] == 2 * s * ctx.n_out and cb["c2"]["recv"] == s * ctx.n_in
        )  # no padding at (6, W)
        assert (
            cb["b2"]["recv"] == s * ctx.n_out and cb["b1"]["send"] == s * ctx.n_in
        )  # save_transfer: d_f only
        assert cb["c1"]["recv"] == 2 * ctx.N_out * hp.Hg_r * gp.STATE_TILE_BYTES
        assert cb["c2"]["send"] == ctx.N_in * hp.Hg_r * gp.STATE_TILE_BYTES
        assert cb["halo_fwd"]["send"] == sum(ctx.halo.fwd_send_rows) * C_CONV * 2
        assert cb["halo_bwd"]["recv"] == sum(ctx.halo.bwd_recv_rows) * C_CONV * 4
        assert cb["B1"] == {
            "send": cb["c1"]["send"] + cb["c2"]["send"] + cb["halo_fwd"]["send"],
            "recv": cb["c1"]["recv"] + cb["c2"]["recv"] + cb["halo_fwd"]["recv"],
        }
        assert cb["B2"] == {
            "send": cb["b1"]["send"] + cb["b2"]["send"] + cb["halo_bwd"]["send"],
            "recv": cb["b1"]["recv"] + cb["b2"]["recv"] + cb["halo_bwd"]["recv"],
        }
        assert (
            cb["B3"]["send"]
            == cb["B1"]["send"]
            + cb["b1_replay"]["send"]
            + cb["b2"]["send"]
            + cb["halo_bwd"]["send"]
        )
        maps = gb.gdn_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False
        )
        s_owner = hp.Hg_r * gp.STATE_TILE_BYTES
        # lower-bound form ("b1_design"): b1_send = s n_in + s n_out [not save_transfer]; owner
        # b1_recv in d / m halves
        assert maps["b1_design"]["send"] == s * (ctx.n_in + ctx.n_out)
        assert maps["b1_design"]["recv"] == (ctx.N_in + ctx.N_out) * s_owner
        assert (
            maps["b1_d"] == cb["b1"] == cb["b1_replay"] == cb["b1_design"] == cb["b1_d"]
        )
        assert maps["b1_m"] == {"send": s * ctx.n_out, "recv": ctx.N_out * s_owner}
        assert cb["b1_m"] == {"send": 0, "recv": 0}
        assert maps["B3"] == cb["B3"]  # the replay never re-sends maps
        # "b1" = the plan's split tables: the re-sent maps are paired with the crossing-IN slots (2 s per
        # crossing-in fragment); they differ from the design form per rank, the rank sums agree
        assert maps["b1"] == {
            "send": 2 * s * ctx.n_in,
            "recv": 2 * ctx.N_in * s_owner,
        }
        assert cb["padding"]["b1"] == {
            "send": hp.pad_numel["b1_send_d"] * 4,
            "recv": hp.pad_numel["b1_recv_d"] * 4,
        }
        b1_sums["design"] += maps["b1_design"]["send"]
        b1_sums["buffers"] += maps["b1"]["send"]
        replay = gb.gdn_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False, boundary_mode="replay"
        )
        assert replay["b1"] == cb["b1_replay"]
        # the exchange terms are HV-only: H_qk = 1, 2, 3, 6 give the same c1 / c2 / b1 / b2
        for hq in (1, 3, 6):
            other = gb.gdn_kcp_comm_bytes(ctx, H_qk=hq, HV=HV, C=C_CONV)
            assert all(other[k] == cb[k] for k in ("c1", "c2", "b1", "b2"))
        for save_mode in gb.SAVE_MODES:
            for boundary_mode in gb.BOUNDARY_MODES:
                for save_transfer in (True, False):
                    got = gb.gdn_kcp_saved_bytes(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                    )
                    want = _saved_formula(
                        ctx, H_qk, HV, save_mode, boundary_mode, save_transfer, C_CONV
                    )
                    assert got == want, (r, save_mode, boundary_mode, save_transfer)
                    layout = gb.gdn_kcp_saved_layout(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                    )
                    assert sum(layout.values()) == got and all(
                        v >= 0 for v in layout.values()
                    )
                    ws = gb.gdn_kcp_workspace_layout(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                    )
                    assert (
                        ws["workspace"]
                        == max(ws["forward"], ws["backward"])
                        == gb.gdn_kcp_workspace_bytes(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode=save_mode,
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                        )
                    )
                    assert ws["forward"] == sum(
                        v for k, v in ws.items() if k.startswith("fwd.")
                    )
                    # the backward is the largest of its lifetime phases, each below the sum of what the arena
                    # can hold; the kept form lists the hand-off tensors apart and both forms need the same
                    # transient memory besides the saved set
                    phases = {k: v for k, v in ws.items() if k.startswith("bwd.phase.")}
                    parts = sum(
                        v
                        for k, v in ws.items()
                        if k.startswith("bwd.")
                        and not k.startswith("bwd.phase.")
                        and k != "bwd.kept_tapes"
                    )
                    assert ws["backward"] == max(phases.values())
                    assert all(
                        phases[f"bwd.phase.{p}"] <= parts
                        for p in ("exchange", "walk", "end")
                    )
                    kept_ws, replay_ws = (
                        gb.gdn_kcp_workspace_layout(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode=save_mode,
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                            reuse_recompute_tapes=form,
                        )
                        for form in (True, False)
                    )
                    tapes = ctx.NC_pad * HV * (32768 + 4) + ctx.R * HV * (
                        3 * 256 + 2 * 128 + 3 * 4
                    )
                    assert kept_ws["bwd.kept_tapes"] == (
                        0 if save_mode == "full" else tapes
                    )
                    assert replay_ws["bwd.kept_tapes"] == 0
                    assert (
                        kept_ws["backward_transient"]
                        == replay_ws["backward_transient"]
                        == replay_ws["backward"]
                    )
                    assert (
                        ws["bwd.dv_new"] == 0 and ws["bwd.dv_intra"] == ctx.R * HV * 256
                    )  # dv_new is stored in place of dv_intra
                    assert (
                        ws["fwd.c1_send"] == cb["c1"]["send"]
                        and ws["bwd.b2_send"] == cb["b2"]["send"]
                    )
                    if save_mode == "light":
                        assert (
                            ws["fwd.h_tape"] == ctx.NC_pad * HV * 32768
                            and ws["bwd.h_tape"] == ws["fwd.h_tape"]
                        )
                    else:
                        assert (
                            "fwd.h_tape" not in ws
                            and layout["h_tape"] == ctx.NC_pad * HV * 32768
                        )
                    if boundary_mode == "replay":
                        assert (
                            layout["S_in"] == 0
                            and layout["c1_recv"] == 0
                            and ws["bwd.c1_send"] == cb["c1"]["send"]
                        )
        light, full = (
            gb.gdn_kcp_saved_bytes(ctx, H_qk=H_qk, HV=HV, save_mode=m, C=C_CONV)
            for m in ("light", "full")
        )
        assert full - light == ctx.NC_pad * HV * (32768 + 4) + ctx.R * HV * (
            3 * 256 + 2 * 128 + 3 * 4
        )  # h + gamma, v_new / w / u, Aqk / Akk, gcum / rowscale / qscale
        host = gb.gdn_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, group_reduce="host", C=C_CONV
        )
        assert host["bwd.group_partials"] == 2 * ctx.T_r * HV * 128 * 4
        records.append(
            {
                "rank": r,
                "n_in": ctx.n_in,
                "n_out": ctx.n_out,
                "B1": cb["B1"],
                "B2": cb["B2"],
                "B3": cb["B3"],
                "saved_light": light,
                "saved_full": full,
                "workspace_light": gb.gdn_kcp_workspace_bytes(
                    ctx, H_qk=H_qk, HV=HV, C=C_CONV
                ),
            }
        )
    with pytest.raises(ValueError, match="multiple"):
        gb.gdn_kcp_comm_bytes(ctx, H_qk=4, HV=6)
    with pytest.raises(ValueError, match="save_mode"):
        gb.gdn_kcp_saved_bytes(ctx, H_qk=2, HV=6, save_mode="tape")
    with pytest.raises(ValueError, match="boundary_mode"):
        gb.gdn_kcp_saved_bytes(ctx, H_qk=2, HV=6, boundary_mode="none")
    assert b1_sums["design"] == b1_sums["buffers"]  # same bytes over the group

    _evidence(
        "bytes",
        {
            "row": "C1",
            "cp": W,
            "H_qk": H_qk,
            "HV": HV,
            "C": C_CONV,
            "per_rank": records,
        },
    )


# --------------------------------------------------------------------------- registry seed and package hygiene


def test_registry_seed() -> None:
    """The ``cake_jit`` seed: the shipped records of the nine programs (one library per record and exported arch
    in ``build.json``; + the two test-only transfer walks of ``build.test.json``), the knob vocabulary,
    every record's logical names in the vocabulary, the grid / selection rules, literal
    tables (``ast.literal_eval`` of the module source equals the runtime dicts; frozen 20-hex library names; no
    build-system token) and the manifest skeleton."""
    assert set(gj.PROGRAMS) == {rec["program"] for rec in gj.MODULES.values()}
    assert sum(len(gj.records_of_program(p)) for p in gj.PROGRAMS) == len(gj.MODULES)
    assert all(
        len(gj.records_of_program(p)) == 1
        for p in gj.PROGRAMS
        if p
        not in (
            "kcp_fwdh_offset",
            "kcp_fwdh_tape",
            "kcp_dhu_offset",
            "kcp_dhu_tape",
            "kcp_affine_merge",
        )
    )  # the vh=64 forms of the state walks #2 (L8A) / #7 and the c8 / c16 merges are the only multi-record programs
    assert set(gj.TEST_MODULES) == {
        "kcp_transfer_walk_three_mma",
        "kcp_transfer_walk_plain_tf32",
    }  # no bf16x3 module is built
    assert gj.PRECISION_MODES == (0, 1, 2) and gj.PRECISION_MODE_DEFAULT == 0
    assert gj.TRANSFER_WALKS == ("tf32_hilo", "bf16x3")
    assert gj.TRANSFER_WALK_DEFAULT == "tf32_hilo"
    assert gj.TRANSFER_WALKS_BUILT == ("tf32_hilo",)
    assert gj.OFFSET_CLASSES == ("fla", "fp32") and gj.OFFSET_CLASS_DEFAULT == "fla"
    vocab = set(gj.LOGICAL_TENSORS) | set(gj.LOGICAL_SCALARS)
    manifest_rendered = (PACKAGE_DIR / "build.json").exists()
    source = (PACKAGE_DIR / "cake_jit.py").read_text()
    literal = {}
    for node in ast.parse(source).body:
        target = (
            node.target
            if isinstance(node, ast.AnnAssign)
            else (node.targets[0] if isinstance(node, ast.Assign) else None)
        )
        if isinstance(target, ast.Name) and target.id in ("MODULES", "TEST_MODULES"):
            literal[target.id] = ast.literal_eval(node.value)
    assert literal == {"MODULES": gj.MODULES, "TEST_MODULES": gj.TEST_MODULES}
    build_system_token = (
        "t" + "art"
    )  # assembled so that this public file does not carry the token itself
    assert not re.search(
        rf"\b{build_system_token}\b", source
    )  # no build-system token in the public registry
    for name, rec in {**gj.MODULES, **gj.TEST_MODULES}.items():
        assert set(rec["logical"].values()) <= vocab, name
        assert {k for kind, k in rec["arg_plan"] if kind != "grid"} == set(
            rec["logical"]
        ), name
        assert rec["launch"]["cooperative"] is False and rec["ffi_entry"] == "run"
        assert re.fullmatch(r"cake_gdn_train_[0-9a-f]{20}", rec["library"]), name
        assert rec["kernel_symbol"] == f"kernel_{rec['library']}"
        assert rec["arg_plan"][-3:] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        assert rec["select_rule"]["kind"] in ("always", "columns", "ctas_per_head"), (
            name
        )
        assert ("test_knobs" in rec) == (name in gj.TEST_MODULES), name
        if not manifest_rendered:
            with pytest.raises(RuntimeError, match="no rendered library"):
                gj.library_name(name, "sm_103a")
    assert gj.select_record("kcp_affine_merge", ncols=128) == "kcp_affine_merge_c8"
    assert gj.select_record("kcp_affine_merge", ncols=256) == "kcp_affine_merge_c16"
    assert (
        gj.select_record("kcp_transfer_walk") == "kcp_transfer_walk"
    )  # the test-only records are never selected by default
    assert (
        gj.select_record("kcp_transfer_walk", precision_mode=1)
        == "kcp_transfer_walk_three_mma"
    )
    assert (
        gj.select_record("kcp_transfer_walk", precision_mode=2)
        == "kcp_transfer_walk_plain_tf32"
    )
    with pytest.raises(ValueError, match="not built in this package"):
        gj.select_record("kcp_transfer_walk", transfer_walk="bf16x3")
    with pytest.raises(ValueError, match="not built in this package"):
        gj.select_record("kcp_transfer_walk", precision_mode=1, transfer_walk="bf16x3")
    with pytest.raises(ValueError, match="transfer_walk must be one of"):
        gj.select_record("kcp_transfer_walk", transfer_walk="bf16x3_hw64")
    with pytest.raises(ValueError):
        gj.select_record("kcp_transfer_walk", precision_mode=3)
    with pytest.raises(ValueError):
        gj.select_record(
            "fwd_o", precision_mode=1
        )  # no test-only record of another program
    with pytest.raises(ValueError):
        gj.select_record("kcp_affine_merge", ncols=64)
    assert gj.launch_grid("fwd_prep", num_chunks=10, num_qk_heads=16) == (
        10,
        16,
        1,
    )  # chunks_by_qk_heads: one CTA per (chunk, Q/K head) of the registry of record
    assert gj.launch_grid("fwd_o", num_chunks=10, num_heads=48) == (480, 1, 1)
    assert gj.launch_grid("kcp_fwdh_tape", n_walk=3, num_heads=48) == (144, 1, 1)
    assert gj.launch_grid(
        "kcp_fwdh_tape_vh64", n_walk=3, num_heads=48, ctas_per_head=2
    ) == (288, 1, 1)  # the vh = 64 form of the state walks is its own record (L8A)
    assert gj.select_record("kcp_fwdh_tape", ctas_per_head=2) == "kcp_fwdh_tape_vh64"
    with pytest.raises(ValueError, match="ctas_per_head"):
        gj.launch_grid("kcp_fwdh_tape", n_walk=3, num_heads=48, ctas_per_head=3)
    assert gj.launch_grid("kcp_affine_merge_c8", n_chains=5, ncols=128) == (80, 1, 1)
    assert gj.launch_grid("kcp_affine_merge_c16", n_chains=5, ncols=256) == (80, 1, 1)
    with pytest.raises(ValueError):
        gj.launch_grid("kcp_affine_merge_c8", n_chains=5, ncols=100)
    assert gj.TEST_MODULES["kcp_transfer_walk_three_mma"]["test_knobs"] == {
        "precision_mode": 1
    }
    assert gj.TEST_MODULES["kcp_transfer_walk_plain_tf32"]["test_knobs"] == {
        "precision_mode": 2
    }
    assert "kcp_transfer_walk_bf16x3" not in gj.TEST_MODULES
    assert gj.MODULES["kcp_affine_merge_c16"]["grid_rule"] == {
        "kind": "kcp_chains",
        "cols_per_cta": 16,
    }
    assert gj.MODULES["kcp_affine_merge_c8"]["select_rule"] == {
        "kind": "columns",
        "ncols": 128,
    }
    # the rendered manifests: one library per shipped record and exported arch; the test-only walks likewise
    manifest = json.loads((PACKAGE_DIR / "build.json").read_text())
    assert manifest["package"] == "flashinfer.experimental.cake_gdn_train"
    n_libs = len(manifest["libraries"])
    assert n_libs == len(gj.MODULES) * len(gj.ARCHES)
    assert {lib["arch"] for lib in manifest["libraries"]} == set(gj.ARCHES)
    test_manifest_path = PACKAGE_DIR / "build.test.json"
    n_test_libs = (
        len(json.loads(test_manifest_path.read_text())["libraries"])
        if test_manifest_path.exists()
        else None
    )
    assert n_test_libs in (None, len(gj.TEST_MODULES) * len(gj.ARCHES))
    libraries = {
        name: rec["library"] for name, rec in {**gj.MODULES, **gj.TEST_MODULES}.items()
    }
    assert len(set(libraries.values())) == len(libraries)  # distinct identities
    gb = _package("cake_backend")
    assert (
        gb.arch_for_capability((10, 3)) == "sm_103a"
        and gb.arch_for_capability((9, 0)) is None
    )
    _evidence(
        "registry",
        {
            "records": sorted(gj.MODULES),
            "test_records": sorted(gj.TEST_MODULES),
            "libraries": n_libs,
            "test_libraries": n_test_libs,
            "library_by_record": libraries,
            "precision_modes": list(gj.PRECISION_MODES),
            "transfer_walks": list(gj.TRANSFER_WALKS),
            "offset_classes": list(gj.OFFSET_CLASSES),
            "pass": True,
        },
    )


def test_package_imports_no_kda_and_no_environ() -> None:
    """The package never imports ``cake_kda_train`` and reads no environment variable; the plan
    module imports torch, torch.distributed and dataclasses only."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
                assert not any("cake_kda_train" in n for n in names), (
                    f"{path.name} imports cake_kda_train"
                )
                assert "os" not in [
                    a.name for a in node.names if isinstance(node, ast.Import)
                ], f"{path.name} imports os"
            if isinstance(node, ast.Attribute) and node.attr == "environ":
                raise AssertionError(f"{path.name} reads os.environ")
    plan_imports = {
        node.names[0].name if isinstance(node, ast.Import) else node.module
        for node in ast.walk(ast.parse((PACKAGE_DIR / "cake_kcp_plan.py").read_text()))
        if isinstance(node, (ast.Import, ast.ImportFrom))
        and (not isinstance(node, ast.ImportFrom) or node.level == 0)
    }
    assert plan_imports <= {
        "dataclasses",
        "torch",
        "torch.distributed",
        "__future__",
    }, plan_imports
    header = (PACKAGE_DIR / "cake_kcp_plan.py").read_text()
    assert (
        "13d7fcf4041635f5b7f573bb0dbab23570ae1034" in header
        and "0f2d8b06bbea49df37b043fba3026a06799f0a0cba09d697e490d02bb747dfbf" in header
    )


# --------------------------------------------------------------------------- fake backend vs the FP64 reference

REFERENCE = pytest.mark.skipif(
    gdn_reference is None,
    reason="gdn_reference (FP64 reference harness) not on PYTHONPATH",
)


def _inputs(
    cu: list[int], H_qk: int, HV: int, seed: int = 20261023
) -> dict[str, torch.Tensor]:
    """Seeded CPU operands of the normal regime (no batch axis): q / k BF16 L2-normalised rows, v / dout BF16,
    g = -exp(A[hv]) softplus(x) FP32 (A ~ U(0, log 16) per value head), beta = sigmoid(x) FP32."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    T = cu[-1]
    a_log = torch.rand(HV, generator=gen) * math.log(16.0)

    def l2(x):
        return (x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)).to(
            torch.bfloat16
        )

    return {
        "q": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "k": l2(torch.randn(T, H_qk, 128, generator=gen)),
        "v": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
        "g": (
            -torch.exp(a_log)[None, :]
            * torch.nn.functional.softplus(torch.randn(T, HV, generator=gen))
        )
        .float()
        .contiguous(),
        "beta": torch.sigmoid(torch.randn(T, HV, generator=gen)).float().contiguous(),
        "dout": torch.randn(T, HV, 128, generator=gen).to(torch.bfloat16),
    }


def _bitwise(a: dict, b: dict, names) -> bool:
    return all(torch.equal(a[n], b[n]) for n in names)


@REFERENCE
@pytest.mark.parametrize("layout", ("contiguous", "zigzag"))
def test_fake_backend_matches_fp64_reference(layout) -> None:
    """Result 4: the fake backend's forward / backward of a 2-rank C and Z row at (H_qk, HV) = (2, 6), T = 507,
    matches the FP64 reference within FLA-class error (rel-RMS <= 0.05 per tensor, finite), is bitwise repeatable,
    and gives bitwise equal gradients for boundary_mode saved / replay, save_transfer on / off and save_mode
    light / full."""
    H_qk, HV = 2, 6
    cu = FAKE_CU
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 2, layout)
    assert sum(part.crossing_counts()[0]) >= 1  # the row crosses a rank boundary
    ops = _inputs(cu, H_qk, HV)
    res = fake.emulate(part, ops, H_qk=H_qk, HV=HV)
    assert res["logs"]["forward"] == ["c1", "c2"] and res["logs"]["backward_saved"] == [
        "b1",
        "b2",
    ]
    assert res["logs"]["backward_replay"] == ["c1", "c2", "b1", "b2"]
    assert fake.replay_equals_saved(res), "saved and replay gradients differ"
    ref = gdn_reference.gdn_exact_with_grads(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ops["dout"],
        cu,
        scale=fake.SCALE,
    )
    metrics = {}
    for name in ("out",) + fake.GRAD_NAMES:
        m = gdn_reference.error_metrics(res["saved"][name], ref[name])
        metrics[name] = {
            k: m[k]
            for k in (
                "rel_rms",
                "max_abs",
                "ref_rms",
                "ref_max_abs",
                "nonfinite",
                "numel",
            )
        }
        assert m["nonfinite"] == 0, name
        assert m["rel_rms"] <= FAKE_TOL_REL_RMS, (name, m["rel_rms"])
    # repeatability (bitwise) and the other saved-set forms (same programs, same bits)
    again = fake.emulate(part, ops, H_qk=H_qk, HV=HV)
    assert _bitwise(res["saved"], again["saved"], ("out",) + fake.GRAD_NAMES), (
        "a second run differs bitwise"
    )
    no_transfer = fake.emulate(part, ops, H_qk=H_qk, HV=HV, save_transfer=False)
    full = fake.emulate(part, ops, H_qk=H_qk, HV=HV, save_mode="full")
    assert fake.replay_equals_saved(no_transfer) and fake.replay_equals_saved(full)
    assert _bitwise(res["saved"], no_transfer["saved"], ("out",) + fake.GRAD_NAMES), (
        "save_transfer=False differs"
    )
    assert _bitwise(res["saved"], full["saved"], ("out",) + fake.GRAD_NAMES), (
        "save_mode=full differs"
    )
    # the no-CP form (one rank) agrees with the reference too and documents the local error floor
    single = fake.emulate(
        gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous"),
        ops,
        H_qk=H_qk,
        HV=HV,
    )
    local = {
        name: gdn_reference.error_metrics(single["saved"][name], ref[name])["rel_rms"]
        for name in ("out",) + fake.GRAD_NAMES
    }
    _evidence(
        "fake_backend",
        {
            "layout": layout,
            "world_size": 2,
            "H_qk": H_qk,
            "HV": HV,
            "cu_seqlens": cu,
            "T_r": [part.rank_tokens(r) for r in range(2)],
            "n_in": list(part.crossing_counts()[0]),
            "n_out": list(part.crossing_counts()[1]),
            "metrics": metrics,
            "local_rel_rms": local,
            "tolerance_rel_rms": FAKE_TOL_REL_RMS,
            "saved_eq_replay": True,
            "repeat_bitwise": True,
            "save_transfer_false_bitwise": True,
            "save_mode_full_bitwise": True,
            "pass": True,
        },
    )


@REFERENCE
def test_fake_backend_fragment_state_matches_reference() -> None:
    """The offset / transfer walks and the merge: the carried ``S_in`` of every crossing-in fragment equals the FP64
    state at its cut (``cut_states``) within FLA-class error, and ``S_out = M_f S_in + B_f`` holds for the FP32
    summaries (the merge's inputs)."""
    H_qk, HV = 2, 6
    cu = FAKE_CU
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 2, "zigzag")
    ops = _inputs(cu, H_qk, HV)
    ranks = fake.build_ranks(
        part, [fake.slice_local(ops, part, r) for r in range(2)], H_qk=H_qk, HV=HV
    )
    _outs, saved = fake.emulate_forward(ranks)
    cuts = sorted(
        {cu[f.doc] + f.off for frs in part.fragments for f in frs if f.crossing_in}
    )
    states = gdn_reference.cut_states(
        ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], cu, cuts, scale=fake.SCALE
    )
    worst = 0.0
    for rk in ranks:
        pa = rk.wt.pools["s_in"]
        for f in rk.ctx.fragments:
            if not f.crossing_in:
                continue
            slot = rk.ctx.in_slot[f.slot]
            S = torch.stack(
                [
                    saved.s_in[rk.rank][
                        pa.elem(slot, h) : pa.elem(slot, h) + gp.TILE
                    ].view(128, 128)
                    for h in range(HV)
                ]
            )
            m = gdn_reference.error_metrics(S, states[cu[f.doc] + f.off])
            worst = max(worst, m["rel_rms"])
            assert m["rel_rms"] <= FAKE_TOL_REL_RMS, (rk.rank, f.slot, m["rel_rms"])
    _evidence(
        "fake_states",
        {"layout": "zigzag", "cuts": cuts, "worst_rel_rms": worst, "pass": True},
    )


def test_fake_backend_run_module_contract() -> None:
    """``run_module`` takes the kernels' call: an unknown record or a missing logical argument is refused before
    any work; every record of the registry has a fake implementation."""
    with pytest.raises(KeyError):
        fake.run_module("nope", {})
    with pytest.raises(ValueError, match="needs logical argument"):
        fake.run_module("gate_epilogue", {"dg_part": torch.zeros(1)})
    shapes = fake.tape_shapes(T_r=10, NC_pad=2, R=128, H_qk=2, HV=6)
    assert shapes["h_tape"] == ((2, 6, 128, 128), torch.bfloat16) and shapes["dq"] == (
        (10, 2, 128),
        torch.bfloat16,
    )
    for name in list(gj.MODULES) + list(
        gj.TEST_MODULES
    ):  # every record dispatches: the argument check is reached
        with pytest.raises(ValueError, match="needs logical argument"):
            fake.run_module(name, {})


def test_dhu_offset_walk_defers_the_dh_tape_to_the_first_tape_walk() -> None:
    """The #7 offset walk hands the kernel ``h_tape`` as its never-dereferenced ``dh_tape`` pointer
    (``DHU_OFFSET_TAPE_STANDIN``), so the arena allocates ``dh_tape`` at the first tape walk -- after the adjoint
    merge -- and never before the adjoint exchange.  CP = 1 with two segments per fragment (P = 2) runs the offset
    walk over every segment before the segment merges; the fake backend on the CPU records the arena's tapes at
    every #7 launch."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    seen: list[tuple[str, bool, bool]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            if name in ("kcp_dhu_offset", "kcp_dhu_tape"):
                call = args["_call"]
                seen.append(
                    (
                        name,
                        "dh_tape" in call.ws._tapes,
                        args["dh_tape"].data_ptr()
                        == call.tape(kcp.DHU_OFFSET_TAPE_STANDIN).data_ptr(),
                    )
                )
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        out, saved = kcp.chunk_gdn_kcp_fwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.chunk_gdn_kcp_bwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            ops["dout"],
            saved=saved,
            backend="fake",
            segments=2,
        )
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    assert out.shape[0] == cu[-1] and all(
        torch.isfinite(g.float()).all() for g in grads
    )
    names = [s[0] for s in seen]
    assert "kcp_dhu_offset" in names and "kcp_dhu_tape" in names, names
    first_tape = names.index("kcp_dhu_tape")
    offsets_before = [s for s in seen[:first_tape] if s[0] == "kcp_dhu_offset"]
    assert offsets_before, (
        "the offset walk must run before the first tape walk in this form"
    )
    assert all(not present for _, present, _ in offsets_before), (
        "dh_tape sat in the arena at an offset walk"
    )
    assert all(alias for name, _, alias in seen if name == "kcp_dhu_offset"), (
        "the offset walk's dh_tape pointer is not the stand-in tape"
    )
    assert seen[first_tape][1] and not seen[first_tape][2], (
        "the first tape walk must allocate and receive the real dh_tape"
    )
    assert (
        kcp.DHU_OFFSET_TAPE_STANDIN in kcp.FWD_TAPES
    )  # a tape live by then: saved (full) or replayed (light)


def test_persistent_programs_receive_the_workspace_ticket_slots() -> None:
    """#5 / #6 / #8 receive the caller workspace's ticket slot of their site as the logical ``ticket``: a 128-byte
    aligned u32 slot that is zero at every launch (the kernels leave it so; the emulation never touches it), the
    same slot across the forward and the backward of a caller-held workspace and across repeated backwards.  The
    #7 walks launch the record the registry provides for the form rule (the ``_vh64`` record where it ships, else
    the full-head record), and the test knob ``dhu_vh`` forces the full-head form."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU, sm_count=SM_COUNT)
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.KcpWorkspace()
    seen: list[tuple[str, int, bool, bool]] = []
    walks: list[tuple[str, int, str, tuple[int, int, int]]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if name in kcp.TICKET_SITES:
                t = args["ticket"]
                seen.append(
                    (
                        name,
                        t.data_ptr(),
                        t.dtype == torch.uint32
                        and t.numel() == kcp.TICKET_SLOT_WORDS
                        and t.data_ptr() % kcp.TICKET_SLOT_BYTES == 0
                        and t.data_ptr() == call.ws.ticket(name).data_ptr(),
                        int(t.view(torch.int32).abs().sum()) == 0,
                    )
                )
            super().run(name, args)
            if name in kcp.DHU_PROGRAMS:
                record, grid = call.records[-1]
                walks.append((name, int(args["_grid_args"]["n_walk"]), record, grid))

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        out, saved = kcp.chunk_gdn_kcp_fwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
            workspace=ws,
        )
        bwd = dict(saved=saved, backend="fake", segments=2, workspace=ws)
        grads = kcp.chunk_gdn_kcp_bwd(
            ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx, ops["dout"], **bwd
        )
        n_rule = len(walks)
        grads2 = kcp.chunk_gdn_kcp_bwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            ops["dout"],
            test_knobs={"dhu_vh": 128},
            **bwd,
        )
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    assert out.shape[0] == cu[-1]
    assert all(torch.equal(a, b) for a, b in zip(grads, grads2, strict=True))
    assert {s for s, *_ in seen} == set(kcp.TICKET_SITES)
    assert all(slot_ok for _, _, slot_ok, _ in seen), seen
    assert all(zero for _, _, _, zero in seen), "a ticket word was not zero at a launch"
    ptr: dict[str, set[int]] = {}
    for site, p, _, _ in seen:
        ptr.setdefault(site, set()).add(p)
    assert all(len(v) == 1 for v in ptr.values()), (
        ptr
    )  # one slot per site for the workspace's lifetime
    assert len({next(iter(v)) for v in ptr.values()}) == len(kcp.TICKET_SITES)
    assert all(ws.ticket(s).data_ptr() == next(iter(ptr[s])) for s in kcp.TICKET_SITES)
    assert ws.ticket_bytes == len(kcp.TICKET_SITES) * kcp.TICKET_SLOT_BYTES
    assert ws.bytes >= ws.ticket_bytes + ws.tape_bytes
    # the #7 walks: the record follows the registry, the grid the record
    assert n_rule and len(walks) > n_rule
    for name, n_walk, record, grid in walks[:n_rule]:
        if record.endswith(kcp.VH64_SUFFIX):
            assert kcp.dhu_vh(n_walk, HV, SM_COUNT) == 64, (name, n_walk)
            assert grid == (n_walk * HV * 2, 1, 1), (name, n_walk, grid)
            assert gj.MODULES[record]["program"] == name
        else:
            assert record == name and grid == (n_walk * HV, 1, 1), (
                name,
                n_walk,
                record,
                grid,
            )
            assert (
                kcp.dhu_vh(n_walk, HV, SM_COUNT) == 128
                or (name + kcp.VH64_SUFFIX) not in gj.MODULES
            )
    for name, n_walk, record, grid in walks[
        n_rule:
    ]:  # dhu_vh = 128 forced: the full-head form everywhere
        assert record == name and grid == (n_walk * HV, 1, 1), (
            name,
            n_walk,
            record,
            grid,
        )


def test_dead_pools_are_not_sized_for_the_call() -> None:
    """R2-L1 1a: a call's workspace holds no pool the call never touches.  Under the shipped ``exchange="a2a"`` the
    ring pools are not sized; under ``save_transfer`` the forward binds fresh tensors for ``c1_recv`` / ``stage_m``
    (saved set) and the saved-context backward binds the saved ones, so those pools are not sized either.  A
    workspace built from ``(ctx, H_qk, HV)`` still sizes every pool; ``exchange="ring"`` sizes the ring pools."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    assert set(kcp.RING_POOLS) == {"ring_out", "ring_in"}
    assert set(kcp.SAVE_TRANSFER_FRESH_POOLS) == {"c1_recv", "stage_m"}
    assert set(kcp.RING_POOLS) <= set(kcp.FWD_DEAD_POOLS) and set(
        kcp.RING_POOLS
    ) <= set(kcp.BWD_DEAD_BEFORE_TAPES)
    dead = set(kcp.RING_POOLS) | set(kcp.SAVE_TRANSFER_FRESH_POOLS)
    # the full workspace of the public constructor sizes every pool (the caller-held form)
    full = kcp.KcpWorkspace(ctx, H_qk, HV, device="cpu", backward=True)
    assert dead <= set(full._pools)
    # a2a + save_transfer (shipped defaults): the dead pools are absent after the forward and after the backward
    ws = kcp.KcpWorkspace()
    out, saved = kcp.chunk_gdn_kcp_fwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        backend="fake",
        segments=2,
        workspace=ws,
    )
    assert saved is not None and saved.c1_recv is not None
    assert not (dead & set(ws._pools)), sorted(dead & set(ws._pools))
    assert not (dead & set(kcp.LAST_CALL["fwd"]["pool_bytes"]))
    assert (
        "c1_send" not in ws._pools
    )  # released right after wait(c1) in the eager forward
    assert "eye" in ws._pools
    grads = kcp.chunk_gdn_kcp_bwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        ops["dout"],
        saved=saved,
        backend="fake",
        segments=2,
        workspace=ws,
    )
    assert all(torch.isfinite(g.float()).all() for g in grads)
    assert not (dead & set(ws._pools)), sorted(dead & set(ws._pools))
    # the same call without the caller workspace records the same absence (per-call workspace)
    kcp.chunk_gdn_kcp_fwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        backend="fake",
        segments=2,
    )
    assert not (dead & set(kcp.LAST_CALL["fwd"]["pool_bytes"]))
    # a replay backward (saved=None) binds no saved tensor: c1_recv / stage_m are sized, the ring pools are not
    ws2 = kcp.KcpWorkspace()
    kcp.chunk_gdn_kcp_bwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        ops["dout"],
        saved=None,
        backend="fake",
        segments=2,
        workspace=ws2,
        boundary_mode="replay",
    )
    assert not (set(kcp.RING_POOLS) & set(kcp.LAST_CALL["bwd"]["pool_bytes"]))
    assert kcp._saved_fresh_pools(None) == () and kcp._saved_fresh_pools(saved) == (
        "c1_recv",
        "stage_m",
    )
    # the pools held DURING a call (the eager forward drops FWD_DEAD_POOLS before its output, so the end-of-call
    # record shows ``eye`` alone): a2a sizes neither the ring pools nor the fresh-bound ones; exchange="ring"
    # (recompute form, save_transfer=False) sizes the ring pools and c1_recv
    seen_pools: list[set[str]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            if not seen_pools or seen_pools[-1] is None:
                seen_pools.append(set(args["_call"].ws._pools))
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        kcp.chunk_gdn_kcp_fwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        seen_pools.append(None)  # type: ignore[arg-type]
        kcp.chunk_gdn_kcp_fwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
            exchange="ring",
            save_transfer=False,
        )
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    a2a_pools, ring_pools = seen_pools[0], seen_pools[2]
    assert (
        not (dead & a2a_pools) and {"c1_send", "stage_b", "carry", "eye"} <= a2a_pools
    ), sorted(a2a_pools)
    assert (set(kcp.RING_POOLS) | {"c1_recv", "stage_m"}) <= ring_pools, sorted(
        ring_pools
    )
    # ensure(skip=...) semantics on a bare workspace
    ws4 = kcp.KcpWorkspace()
    tb = ws4.ensure(ctx, H_qk, HV, device="cpu", backward=True, skip=kcp.RING_POOLS)
    assert not (set(kcp.RING_POOLS) & set(ws4._pools)) and "c1_send" in ws4._pools
    assert set(tb.pool_sizes(True)) >= set(
        kcp.RING_POOLS
    )  # the size table is unchanged
    with pytest.raises(KeyError, match="not sized"):
        ws4.pool("ring_out", 16)
    ws4.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert set(kcp.RING_POOLS) <= set(ws4._pools)


def test_eager_forward_sizes_c2_send_at_its_first_writer(monkeypatch) -> None:
    """The eager a2a forward leaves ``c2_send`` unsized by ``ensure`` (``_prepare``): the fwd merge binds a fresh plane
    and wait(c2) drops it, so neither the per-call workspace nor the launch record carries it afterwards; the
    caller-held workspace of the public constructor sizes the pool up front (the forward uses it and, like every
    other forward pool, releases it at the end of the eager call); the outputs and the saved set are bitwise the same
    either way.  The receive planes (``S_in``, ``c1_recv``) are not allocated yet when the summaries' walks run (the
    forward's peak moment)."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.KcpWorkspace()
    at_summaries: list[set[str]] = []
    summaries = kcp._summaries

    def recording_summaries(call):
        at_summaries.append({"S_in", "c1_recv"} & set(call.bufs))
        summaries(call)

    monkeypatch.setattr(kcp, "_summaries", recording_summaries)
    out, saved = kcp.chunk_gdn_kcp_fwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        backend="fake",
        segments=2,
        workspace=ws,
    )
    assert "c2_send" not in ws._pools
    assert "c2_send" not in kcp.LAST_CALL["fwd"]["pool_bytes"]
    assert (
        "c2_send" not in kcp.LAST_CALL["fwd"]["fresh_bytes"]
    )  # dropped after wait(c2)
    assert {"S_in", "c1_recv"} <= set(
        kcp.LAST_CALL["fwd"]["fresh_bytes"]
    )  # the saved set stays bound
    # neither receive plane exists at the summaries moment
    assert at_summaries and all(seen == set() for seen in at_summaries)
    full = kcp.KcpWorkspace(ctx, H_qk, HV, device="cpu", backward=False)
    assert "c2_send" in full._pools
    out2, saved2 = kcp.chunk_gdn_kcp_fwd(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        backend="fake",
        segments=2,
        workspace=full,
    )
    # FWD_DEAD_POOLS: every forward pool goes at the end of an eager call
    assert "c2_send" not in full._pools
    assert torch.equal(out, out2) and torch.equal(saved.S_in, saved2.S_in)
    assert torch.equal(saved.c1_recv, saved2.c1_recv)


def test_backward_forgets_its_own_saved_set_at_the_last_reader() -> None:
    """A caller-owned saved set is never mutated by ``chunk_gdn_kcp_bwd`` (two backwards from it are bitwise equal and
    its tensors stay); a set marked ``release_in_backward`` (the autograd Function's own) comes back with ``S_in``,
    ``c1_recv`` and ``M_seg`` forgotten, with the same gradients bitwise; the per-call buffers of a replay backward
    (``saved=None``) are dropped the same way."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    _, saved = kcp.chunk_gdn_kcp_fwd(*args, backend="fake", segments=2)
    assert saved is not None and saved.c1_recv is not None and saved.M_seg is not None
    assert saved.release_in_backward is False
    g1 = kcp.chunk_gdn_kcp_bwd(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    assert (
        saved.S_in is not None and saved.c1_recv is not None and saved.M_seg is not None
    )
    g2 = kcp.chunk_gdn_kcp_bwd(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    assert all(torch.equal(a, b) for a, b in zip(g1, g2, strict=True))
    owned = dataclasses.replace(saved, release_in_backward=True)
    g3 = kcp.chunk_gdn_kcp_bwd(
        *args, ops["dout"], saved=owned, backend="fake", segments=2
    )
    assert all(torch.equal(a, b) for a, b in zip(g1, g3, strict=True))
    assert owned.S_in is None and owned.c1_recv is None and owned.M_seg is None
    assert (
        saved.S_in is not None
    )  # the original set is untouched (replace copied the references)
    assert not (
        {"S_in", "c1_recv", "stage_m"} & set(kcp.LAST_CALL["bwd"]["fresh_bytes"])
    )
    with pytest.raises(KeyError):
        owned.forget("tapes")
    g4 = kcp.chunk_gdn_kcp_bwd(
        *args,
        ops["dout"],
        saved=None,
        backend="fake",
        segments=2,
        boundary_mode="replay",
    )
    assert all(torch.isfinite(g.float()).all() for g in g4)
    assert "S_in" not in kcp.LAST_CALL["bwd"]["fresh_bytes"]


def test_transpose_tiles_scratch_is_bitwise_the_temporary_form() -> None:
    """R2-L1 1a: the ``M_f -> M_f^T`` scatter with the gather in a caller scratch (the carry plane during the
    summaries) writes the same bytes as the fresh-temporary form and leaves the gathered tiles in the scratch."""
    kcp = _package("cake_kcp")
    D, TILE = kcp.D, kcp.TILE
    gen = torch.Generator(device="cpu").manual_seed(7)
    n_tiles = 10
    pool_a = torch.randn(n_tiles * TILE, generator=gen)
    pool_b = pool_a.clone()
    src_tiles = torch.tensor([7, 2, 9], dtype=torch.int64)
    dst_tiles = torch.tensor([0, 4, 1], dtype=torch.int64)
    scratch = torch.full((5 * TILE,), float("nan"))
    kcp.Backend.transpose_tiles(
        src=pool_a, src_tiles=src_tiles, dst=pool_a, dst_tiles=dst_tiles
    )
    kcp.Backend.transpose_tiles(
        src=pool_b,
        src_tiles=src_tiles,
        dst=pool_b,
        dst_tiles=dst_tiles,
        scratch=scratch,
    )
    assert torch.equal(pool_a, pool_b)
    gathered = scratch[: 3 * TILE].view(3, D, D)
    assert torch.equal(gathered.transpose(1, 2), pool_b.view(-1, D, D)[dst_tiles])
    assert torch.isnan(
        scratch[3 * TILE :]
    ).all()  # the scratch beyond the gathered tiles is untouched
    # the call-level form passes the carry plane as scratch only when it has room; nothing when there is no tile
    assert "scratch" in kcp.KcpCall.transpose_tiles.__kwdefaults__


def test_workspace_layout_counts_one_staging_plane_under_save_transfer() -> None:
    """R2-L1 1a: ``gdn_kcp_workspace_layout`` counts the ``stage_m`` plane only when the call sizes it (replay
    boundary or ``save_transfer=False``); under the shipped saved + ``save_transfer`` form the saved ``M_seg^T``
    stands in for it and the workspace holds ``stage_b`` alone (the ring pools are never workspace terms)."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    part = gp.kcp_partition(cu, 2, "contiguous")
    H_qk, HV = 2, 6
    seen_segmented = False
    # Under the shipped row rule this row plans P = 1 on both ranks ("row_cost" plans P = 1 there -- fragments of 2-14
    # chunks: the per-rank programs
    # segment them 7- and 13-fold to fill the SMs, and the compose chain of those segments costs more than the one-wave
    # walks they save), so the staging plane is 0; the plane term is exercised under the per-rank rule below.
    for r in range(2):
        ctx_row = gp.build_kcp_context(part, None, r, CPU, 4, sm_count=SM_COUNT)
        hp_row = ctx_row.heads(HV)
        assert hp_row.segments.p_rule == "row_cost_lpt" and hp_row.P == 1
        one_row = gb.gdn_kcp_workspace_layout(ctx_row, H_qk=H_qk, HV=HV)
        assert one_row["fwd.staging"] == one_row["bwd.staging"] == 0
    for r in range(2):
        ctx = gp.build_kcp_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )
        hp = ctx.heads(
            HV
        )  # the layout's staging term uses the plan's own segment count
        plane = hp.NSEG * HV * gp.STATE_TILE_BYTES if hp.P > 1 else 0
        seen_segmented = seen_segmented or hp.P > 1
        one = gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
        two_a = gb.gdn_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, boundary_mode="replay"
        )
        two_b = gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, save_transfer=False)
        assert one["fwd.staging"] == one["bwd.staging"] == plane
        assert two_a["fwd.staging"] == two_a["bwd.staging"] == 2 * plane
        assert two_b["fwd.staging"] == two_b["bwd.staging"] == 2 * plane
        assert one["fwd.c1_recv"] == 0 and two_b["fwd.c1_recv"] > 0
        assert not any(k.endswith("ring_out") or k.endswith("ring_in") for k in one)
        assert (
            gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, P=1)["fwd.staging"] == 0
        )
    assert seen_segmented  # the row has a segmented rank: the plane term was exercised


def test_backward_allocates_the_gradients_right_before_dqkg() -> None:
    """R2-L1 1b: the five gradient outputs are allocated after the adjoint tape walk (right before #8), so they are
    not live at the backward's peak moment (the dh_tape allocation of the adjoint walk); under fused walks at P > 1
    the b2_recv pool is released before that walk (the dcarry_cross merge copied dS_out into the carry slots)."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    seen: list[
        tuple[str, bool, bool]
    ] = []  # (program, grads allocated?, b2_recv held?)

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if name in (
                "kcp_dhu_offset",
                "kcp_dhu_tape",
                "bwd_dqkg",
                "gate_epilogue",
                "bwd_prep",
            ):
                seen.append(
                    (
                        name,
                        bool(getattr(call, "grads", None)),
                        "b2_recv" in call.ws._pools,
                    )
                )
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        out, saved = kcp.chunk_gdn_kcp_fwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.chunk_gdn_kcp_bwd(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            ops["dout"],
            saved=saved,
            backend="fake",
            segments=2,
        )
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    assert (
        out.shape[0] == cu[-1]
        and len(grads) == 5
        and all(torch.isfinite(g.float()).all() for g in grads)
    )
    names = [n for n, _, _ in seen]
    assert (
        "kcp_dhu_tape" in names
        and "bwd_dqkg" in names
        and names.index("kcp_dhu_tape") < names.index("bwd_dqkg")
    )
    assert all(not g for n, g, _ in seen if n != "bwd_dqkg" and n != "gate_epilogue"), (
        seen
    )
    assert all(g for n, g, _ in seen if n in ("bwd_dqkg", "gate_epilogue")), seen
    assert all(not held for n, _, held in seen if n == "kcp_dhu_tape"), (
        seen
    )  # P = 2, fused walks: b2_recv gone
    assert [n for n in names if n == "bwd_dqkg"] == ["bwd_dqkg"]


# --------------------------------------------------------------------------- reuse_recompute_tapes (CPU, fake backend)
# The checkpoint-aware light backward on the torch emulation: torch.utils.checkpoint(use_reentrant=False) re-runs the
# forward inside .backward() under the engine's graph task; with the switch the recompute forward keeps its ten tapes
# for the backward of that task, which then launches nothing before #6.  CP = 1 (one process); P = 1 and P = 2.

CKPT_GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)


def _ckpt_setup():
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.kcp_partition(torch.tensor(cu, dtype=torch.int64), 1, "contiguous")
    ctx = gp.build_kcp_context(part, None, 0, CPU)
    return kcp, ctx, _inputs(cu, H_qk, HV)


def _ckpt_autograd(
    kcp,
    ctx,
    ops,
    *,
    ckpt: bool,
    twice: bool = False,
    layers: int = 1,
    workspace=None,
    **opts,
) -> dict[str, torch.Tensor]:
    """``out`` and the five gradients of one ``chunk_gdn_kcp`` layer (or ``layers`` chained ones: the next layer's
    ``v`` is the previous output), directly or under a non-reentrant checkpoint; ``twice`` backwards with
    ``retain_graph`` first (accumulation)."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", **opts}

    def fn(q, k, v, g, beta):
        out = kcp.chunk_gdn_kcp(q, k, v, g, beta, ctx, workspace=workspace, **o)
        for _ in range(layers - 1):
            out = kcp.chunk_gdn_kcp(q, k, out, g, beta, ctx, workspace=workspace, **o)
        return out

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, use_reentrant=False)
    else:
        out = fn(*leaves)
    out.backward(ops["dout"], retain_graph=twice)
    if twice:
        out.backward(ops["dout"])
    return {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)},
    }


def _stats_delta(kcp, before: dict[str, int]) -> dict[str, int]:
    return {k: kcp.KEPT_STATS[k] - before[k] for k in before}


def _assert_same(a: dict, b: dict, what: str) -> None:
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(a[n], b[n]), f"{what}: {n} differs"


@pytest.mark.parametrize(
    "segments,save_transfer", [(1, True), (2, True), (2, False), (1, False)]
)
def test_checkpoint_backward_reuses_the_recompute_tapes(
    segments, save_transfer
) -> None:
    """With ``reuse_recompute_tapes`` the checkpoint backward binds the recompute forward's tapes: the gradients are
    bitwise the direct autograd's and the switch-off checkpoint's; the backward's launch inventory starts at #6 (no
    #1, no offset / tape walk, no carry merge); the registry sees one put and one take and is empty afterwards; the
    hand-off forward's tape walk is the replay's launch list (the same ``n_walk`` per #2 tape launch).  ``P = 2``
    without ``save_transfer`` hands the ``stage_m`` plane over as well (the backward merges read it)."""
    kcp, ctx, ops = _ckpt_setup()
    base = dict(segments=segments, save_transfer=save_transfer)
    seen: list[
        tuple[str, bool, bool, int]
    ] = []  # (program, backward?, hand-off forward?, n_walk)

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            n_walk = int((args.get("_grid_args") or {}).get("n_walk", -1))
            seen.append(
                (
                    name,
                    bool(call.backward),
                    bool(call.scratch.get("hand_off", False)),
                    n_walk,
                )
            )
            super().run(name, args)

    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    previous = kcp.set_backend("fake", Recording())
    try:
        light = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, reuse_recompute_tapes=False, **base
        )
        light_seq = list(kcp.LAST_CALL["bwd"]["sequence"])
        light_source = kcp.LAST_CALL["bwd"]["tapes_source"]
        replay_walks = [w for n, bwd, _, w in seen if bwd and n == "kcp_fwdh_tape"]
        seen.clear()
        before = dict(kcp.KEPT_STATS)
        kept = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, reuse_recompute_tapes=True, **base
        )
    finally:
        kcp.set_backend("fake", previous)
    _assert_same(kept, direct, "kept vs direct")
    _assert_same(light, direct, "checkpoint (switch off) vs direct")
    # the switch-off checkpoint ran the replay
    assert light_source == "replay", light_source
    assert kcp.LAST_CALL["bwd"]["tapes_source"] == "kept"
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 1,
        "take": 1,
        "miss": 0,
        "released": 1,
        "callback": 0,
        "evicted": 0,
    }, delta
    assert kcp.kept_registry_size() == 0
    assert (
        kcp.LAST_CALL["fwd"]["hand_off"] is True
    )  # the last forward recorded is the recompute's
    # launch inventory: the switch-off backward replays #1 first; the kept backward starts at #6
    assert light_seq[0] == "fwd_prep", light_seq[:4]
    kept_seq = list(kcp.LAST_CALL["bwd"]["sequence"])
    assert kept_seq[0] == "bwd_prep", kept_seq[:4]
    forward_programs = {"fwd_prep", "kcp_fwdh_offset", "kcp_fwdh_tape", "fwd_o"}
    assert not (forward_programs & set(kept_seq)), kept_seq
    bwd_launches = [n for n, bwd, _, _ in seen if bwd]
    assert (
        bwd_launches
        and bwd_launches[0] == "bwd_prep"
        and not (forward_programs & set(bwd_launches))
    )
    # two forwards ran (the checkpointed one, then torch's recompute = the hand-off forward); on a verified torch
    # the recompute's #5 is left out (recompute_output="auto": this node is the segment's last saver), so fwd_o ran
    # in the first one only; any other torch build launches it in both
    assert [h for n, bwd, h, _ in seen if not bwd and n == "fwd_prep"] == [False, True]
    assert [h for n, bwd, h, _ in seen if not bwd and n == "fwd_o"] == (
        [False] if kcp.torch_release_verified() else [False, True]
    )
    # the hand-off forward's #2 tape launches are the replay's (same list lengths in the same order)
    kept_walks = [
        w for n, bwd, h, w in seen if (not bwd) and h and n == "kcp_fwdh_tape"
    ]
    assert kept_walks == replay_walks and kept_walks, (kept_walks, replay_walks)


@pytest.mark.parametrize("layers", [2, 3])
def test_one_live_kept_set_in_a_multi_layer_checkpoint_segment(
    layers, monkeypatch
) -> None:
    """Several GDN layers inside one checkpoint segment on one caller-held workspace: torch recomputes every layer's
    forward before the first backward, so only ONE kept set is live -- each hand-off forward evicts the previous
    layer's set before its own first launch -- and only the last layer's backward takes the kept path; the earlier
    layers' backwards miss and run the replay.  The gradients of the whole chain are bitwise the direct autograd's
    (the evicted layers included); the registry is empty afterwards."""
    kcp, ctx, ops = _ckpt_setup()
    ws = kcp.KcpWorkspace()
    base = dict(segments=2, layers=layers, workspace=ws)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    sources: list[str] = []
    original_backward = kcp._backward

    def recording_backward(call, saved):
        original_backward(call, saved)
        sources.append(call.scratch.get("tapes_source"))

    monkeypatch.setattr(kcp, "_backward", recording_backward)
    before = dict(kcp.KEPT_STATS)
    kept = _ckpt_autograd(kcp, ctx, ops, ckpt=True, reuse_recompute_tapes=True, **base)
    _assert_same(kept, direct, f"{layers} layers, one live kept set, vs direct")
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": layers,
        "take": 1,
        "miss": layers - 1,
        "released": 1,
        "callback": 0,
        "evicted": layers - 1,
    }, delta
    assert kcp.kept_registry_size() == 0
    # the backwards run last layer first: the kept one, then the replays of the evicted layers
    assert sources == ["kept"] + ["replay"] * (layers - 1), sources


def test_kept_backward_twice_accumulates_with_a_fresh_recompute() -> None:
    """``retain_graph=True`` and a second ``.backward()`` is a second graph task: torch recomputes the forward again,
    a second entry is put and taken; the gradients accumulate to exactly twice the direct gradient."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    twice = _ckpt_autograd(
        kcp, ctx, ops, ckpt=True, twice=True, segments=2, reuse_recompute_tapes=True
    )
    for n in CKPT_GRADS:
        assert torch.equal(twice[n], direct[n] + direct[n]), n
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 2,
        "take": 2,
        "miss": 0,
        "released": 2,
        "callback": 0,
        "evicted": 0,
    }, delta
    assert kcp.kept_registry_size() == 0


def test_kept_backward_falls_back_to_the_replay_without_an_entry(monkeypatch) -> None:
    """Correctness never depends on the registry: a backward that finds no entry runs today's replay (bitwise the
    same gradients); the untaken entry is dropped at the graph task's end by the engine's final callback."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    monkeypatch.setattr(kcp, "_kept_take", lambda key: None)
    res = _ckpt_autograd(
        kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=True
    )
    _assert_same(res, direct, "fallback replay vs direct")
    assert kcp.LAST_CALL["bwd"]["tapes_source"] == "replay"
    assert kcp.LAST_CALL["bwd"]["sequence"][0] == "fwd_prep"
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 1,
        "take": 0,
        "miss": 0,
        "released": 0,
        "callback": 1,
        "evicted": 0,
    }, delta
    assert kcp.kept_registry_size() == 0


def test_kept_entry_is_dropped_when_the_node_is_not_in_the_backward() -> None:
    """A backward that recomputes the segment (another saved tensor of the segment is unpacked) but never runs the
    GDN node leaves an untaken entry: the engine's final callback drops it at the task's end."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    a = torch.randn(3, requires_grad=True)

    def fn(a_, q, k, v, g, beta):
        return a_ * a_, kcp.chunk_gdn_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            backend="fake",
            segments=2,
            reuse_recompute_tapes=True,
        )

    b, out = checkpoint(fn, a, *leaves, use_reentrant=False)
    before = dict(kcp.KEPT_STATS)
    b.sum().backward()
    assert a.grad is not None and all(lf.grad is None for lf in leaves)
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 1,
        "take": 0,
        "miss": 0,
        "released": 0,
        "callback": 1,
        "evicted": 0,
    }, delta
    assert kcp.kept_registry_size() == 0
    del out


def test_kept_entry_of_a_failed_backward_is_evicted_by_the_next_kept_forward() -> None:
    """A backward that raises after the recompute (here: a node after the GDN layer in the segment fails) leaves its
    entry -- the engine runs no final callback after an exception -- and the next hand-off forward of a new graph
    task evicts it; that backward then runs normally and the registry is empty again."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    class Boom(torch.autograd.Function):
        @staticmethod
        def forward(fctx, x):
            fctx.save_for_backward(x)
            return x.clone()

        @staticmethod
        def backward(fctx, grad):
            (x,) = (
                fctx.saved_tensors
            )  # the unpack recomputes the segment (the GDN forward keeps its tapes)
            raise RuntimeError("injected failure after the recompute")

    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]

    def fn(q, k, v, g, beta):
        return Boom.apply(
            kcp.chunk_gdn_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                segments=2,
                reuse_recompute_tapes=True,
            )
        )

    out = checkpoint(fn, *leaves, use_reentrant=False)
    before = dict(kcp.KEPT_STATS)
    with pytest.raises(RuntimeError, match="injected failure"):
        out.backward(ops["dout"])
    assert (
        kcp.kept_registry_size() == 1
    )  # documented: the entry outlives the failed task
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 1,
        "take": 0,
        "miss": 0,
        "released": 0,
        "callback": 0,
        "evicted": 0,
    }, delta
    del out
    # the direct reference names the replay form: under the shipped default a direct backward looks the registry up
    # and counts a "miss" (by design; test_switch_is_a_no_op_...), which would blur the eviction accounting below
    direct = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, segments=2, reuse_recompute_tapes=False
    )
    kept = _ckpt_autograd(
        kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=True
    )
    _assert_same(kept, direct, "kept after a failed task vs direct")
    delta = _stats_delta(kcp, before)
    assert delta == {
        "put": 2,
        "take": 1,
        "miss": 0,
        "released": 1,
        "callback": 0,
        "evicted": 1,
    }, delta
    assert kcp.kept_registry_size() == 0


def test_switch_is_a_no_op_outside_the_light_saved_checkpoint_backward() -> None:
    """The switch changes nothing for the direct autograd path (no graph task at the forward), the replay boundary
    (nothing saved) and ``save_mode="full"`` (the saved tapes): no entry is put, the gradients are the same bits."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    on_direct = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, segments=2, reuse_recompute_tapes=True
    )
    _assert_same(on_direct, direct, "direct autograd with the switch")
    assert (
        kcp.LAST_CALL["bwd"]["tapes_source"] == "replay"
        and kcp.LAST_CALL["fwd"]["hand_off"] is False
    )
    replay = _ckpt_autograd(
        kcp,
        ctx,
        ops,
        ckpt=True,
        segments=2,
        boundary_mode="replay",
        reuse_recompute_tapes=True,
    )
    _assert_same(replay, direct, "replay boundary with the switch")
    assert kcp.LAST_CALL["bwd"]["tapes_source"] == "recompute"
    full = _ckpt_autograd(
        kcp,
        ctx,
        ops,
        ckpt=True,
        segments=2,
        save_mode="full",
        reuse_recompute_tapes=True,
    )
    _assert_same(full, direct, "save_mode=full with the switch")
    assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved"
    delta = _stats_delta(kcp, before)
    assert delta["put"] == 0 and delta["take"] == 0 and delta["released"] == 0, delta
    assert delta["miss"] == 1, (
        delta
    )  # the direct backward with the switch looked and found nothing (by design)
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- recompute_output (CPU, fake backend)
# The recompute's dead output program: torch's non-reentrant checkpoint re-runs the segment inside .backward() only to
# re-save its tensors and discards that run's output.  When the GDN node is the segment's last saver nothing reads the
# recompute's `out`, and recompute_output="auto" (the shipped default) leaves #5 out of that one forward; every state
# the frame proof cannot establish launches it as before.  The proof is read on the verified torch builds only
# (RECOMPUTE_OUTPUT_TORCH_VERSIONS): elsewhere every recompute launches #5 with the reason `torch_version` and the
# counters stay, and the tests below assert that launch path there.


def _fwd_o_launches(kcp):
    """A recording fake backend: ``(program, hand-off forward?, the call's scratch)`` of every forward ``fwd_prep`` /
    ``fwd_o`` launch; the scratch carries that forward's ``fwd_o`` decision and reason once it has run."""
    seen: list[tuple[str, bool, dict]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward and name in ("fwd_prep", "fwd_o"):
                seen.append(
                    (name, bool(call.scratch.get("hand_off", False)), call.scratch)
                )
            super().run(name, args)

    return seen, Recording()


def _launches(seen, name: str) -> list[bool]:
    return [h for n, h, _ in seen if n == name]


def _forwards(seen) -> list[tuple[bool, str, str]]:
    """Per forward in launch order (one ``fwd_prep`` each): ``(hand-off forward?, fwd_o decision, reason)``."""
    return [(h, s["fwd_o"], s["fwd_o_reason"]) for n, h, s in seen if n == "fwd_prep"]


def _verified_torch(kcp) -> bool:
    """Whether this torch build reads the frame proof (else every recompute launches #5, reason ``torch_version``)."""
    return kcp.torch_release_verified()


def _frame_reason(kcp, reason: str) -> str:
    """The reason recorded on this build: ``reason`` on a verified torch, ``torch_version`` elsewhere."""
    return reason if _verified_torch(kcp) else "torch_version"


def _frame_delta(kcp, skipped: int, launched: int) -> dict[str, int]:
    """The counter delta on this build: the frame outcomes on a verified torch, untouched elsewhere."""
    if _verified_torch(kcp):
        return {"skipped": skipped, "launched": launched}
    return {"skipped": 0, "launched": 0}


def _dead_decision(kcp) -> str:
    """What a recompute whose output is dead records on this build: ``skipped`` on a verified torch, else ``launched``."""
    return "skipped" if _verified_torch(kcp) else "launched"


def _fwd_o_after_recompute(kcp) -> list[bool]:
    """``fwd_o`` hand-off flags of one original forward and one recompute whose output is dead: the original's only on
    a verified torch, the recompute's too elsewhere."""
    return [False] if _verified_torch(kcp) else [False, True]


def _skip_delta(kcp, before: dict[str, int]) -> dict[str, int]:
    return {k: kcp.RECOMPUTE_OUTPUT_STATS[k] - before[k] for k in before}


def _ckpt_leaves(ops) -> list[torch.Tensor]:
    return [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]


def test_recompute_output_skipped_once_when_the_node_closes_the_segment() -> None:
    """A segment whose last saver is the GDN call: the recompute leaves #5 out exactly once per checkpoint step --
    fwd_prep runs in both forwards, fwd_o in the original only -- the outputs and the gradients are bitwise the
    ``recompute_output="always"`` form's and the direct autograd's, the kept-tape accounting is unchanged, and the
    skip does not rest on the kept tapes (the replay form skips too).  A later op of the segment that saves nothing
    (``.sum()``) keeps the skip: torch's early stop ends the recompute right after this node's tensors are re-saved,
    before that op runs.  On a torch build outside the verified set every recompute launches #5 instead (reason
    ``torch_version``, counters untouched) and the same bitwise equalities hold."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    fwd_o, decision = _fwd_o_after_recompute(kcp), _dead_decision(kcp)
    reason, delta = _frame_reason(kcp, "dead"), _frame_delta(kcp, 1, 0)

    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)

    def seg_sum(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.chunk_gdn_kcp(
                q, k, v, g, beta, ctx, backend="fake", segments=2
            ).sum()

        total = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        total.backward()
        return [lf.grad for lf in leaves]

    direct_sum = seg_sum(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        always = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, segments=2, recompute_output="always"
        )
        assert _launches(seen, "fwd_prep") == [False, True]
        assert _launches(seen, "fwd_o") == [False, True]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == "always"
        seen.clear()
        before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
        auto = _ckpt_autograd(kcp, ctx, ops, ckpt=True, segments=2)
        assert _launches(seen, "fwd_prep") == [False, True]
        assert _launches(seen, "fwd_o") == fwd_o
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == decision
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == reason
        assert _skip_delta(kcp, before) == delta
        assert _stats_delta(kcp, kept_before) == {
            "put": 1,
            "take": 1,
            "miss": 0,
            "released": 1,
            "callback": 0,
            "evicted": 0,
        }
        seen.clear()
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        replay_form = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=False
        )
        # no hand-off without reuse_recompute_tapes: on a verified torch the recompute still skips #5; elsewhere it
        # launches #5 as a plain (non hand-off) forward
        assert _launches(seen, "fwd_o") == (
            [False] if _verified_torch(kcp) else [False, False]
        )
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "replay"
        assert _skip_delta(kcp, before) == delta
        seen.clear()
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        ckpt_sum = seg_sum(True)
        assert _launches(seen, "fwd_o") == fwd_o
        assert _skip_delta(kcp, before) == delta
    finally:
        kcp.set_backend("fake", previous)
    _assert_same(auto, direct, "auto vs direct")
    _assert_same(auto, always, "auto vs always")
    _assert_same(replay_form, direct, "auto + replay form vs direct")
    for a, b in zip(ckpt_sum, direct_sum, strict=True):
        assert torch.equal(a, b)
    assert kcp.kept_registry_size() == 0


# The recompute_output tests below pin reuse_recompute_tapes=True: they check the output program's gate under the
# kept-tape form (the recompute launches #1 and hands the tapes off); the shipped default "auto" takes the replay form
# in several of these scenarios (a later saver, a failed frame proof), which the reuse_recompute_tapes tests cover.
def test_recompute_output_launched_when_a_later_op_of_the_segment_saves_the_output() -> (
    None
):
    """A downstream op inside the same segment that saves ``o`` (``o * w`` with a learnable ``w``: mul saves both
    operands) keeps ``o`` live in the recompute: more holders wait than this node fills, so #5 is launched (reason
    ``holders_remain``) and the gradients -- ``dw`` included -- are bitwise the always form's and the direct
    autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    HV = ops["v"].shape[1]
    w0 = torch.randn(HV, 128, generator=torch.Generator().manual_seed(7)).to(
        torch.bfloat16
    )

    def run(ckpt: bool, **o) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        w = w0.detach().clone().requires_grad_(True)

        def fn(q, k, v, g, beta, w_):
            out = kcp.chunk_gdn_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                segments=2,
                reuse_recompute_tapes=True,
                **o,
            )
            return out * w_

        y = checkpoint(fn, *leaves, w, use_reentrant=False) if ckpt else fn(*leaves, w)
        y.backward(ops["dout"])
        return [lf.grad for lf in leaves] + [w.grad]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        auto = run(True)
        assert _launches(seen, "fwd_prep") == [False, True]
        assert _launches(seen, "fwd_o") == [False, True]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == _frame_reason(
            kcp, "holders_remain"
        )
        assert _skip_delta(kcp, before) == _frame_delta(kcp, 0, 1)
        always = run(True, recompute_output="always")
    finally:
        kcp.set_backend("fake", previous)
    for a, d, w in zip(auto, direct, always, strict=True):
        assert torch.equal(a, d) and torch.equal(a, w)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_multi_layer_segment_skips_only_the_last_layer() -> None:
    """Three GDN layers in one checkpoint segment (the shipped one-live-kept-set form): the next layer saves this
    layer's output as its ``v``, so in the recompute layers 1-2 launch #5 (holders remain) and only the last layer,
    the segment's last saver, leaves it out -- pinned per forward, in launch order.  The chain's gradients are
    bitwise the direct autograd's and the always form's; the kept-tape accounting is the shipped one (one live set,
    the earlier layers replay)."""
    kcp, ctx, ops = _ckpt_setup()
    layers = 3
    ws = kcp.KcpWorkspace()
    base = dict(segments=2, layers=layers, workspace=ws, reuse_recompute_tapes=True)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
        auto = _ckpt_autograd(kcp, ctx, ops, ckpt=True, **base)
        assert _launches(seen, "fwd_prep") == [False] * layers + [True] * layers
        fwds = _forwards(seen)
        # the original forwards all launch; in the recompute layers 1-2 launch (the next layer holds their output)
        # and layer 3, the last saver, leaves #5 out
        assert [f[:2] for f in fwds[:layers]] == [(False, "launched")] * layers
        assert fwds[layers:] == [
            (True, "launched", _frame_reason(kcp, "holders_remain"))
        ] * (layers - 1) + [(True, _dead_decision(kcp), _frame_reason(kcp, "dead"))]
        assert _skip_delta(kcp, before) == _frame_delta(kcp, 1, layers - 1)
        assert _stats_delta(kcp, kept_before) == {
            "put": layers,
            "take": 1,
            "miss": layers - 1,
            "released": 1,
            "callback": 0,
            "evicted": layers - 1,
        }
        seen.clear()
        always = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, recompute_output="always", **base
        )
        assert _launches(seen, "fwd_o") == [False] * layers + [True] * layers
    finally:
        kcp.set_backend("fake", previous)
    _assert_same(auto, direct, f"{layers} layers, auto vs direct")
    _assert_same(auto, always, f"{layers} layers, auto vs always")
    assert kcp.kept_registry_size() == 0


def test_recompute_output_with_an_untaken_node_and_a_second_backward() -> None:
    """(a) The segment is recomputed for another saved tensor while the GDN node itself is not in the backward: the
    node is still the segment's last saver, so #5 is left out, and the kept-tape accounting is the shipped one (the
    untaken entry goes at the task's end).  (b) ``retain_graph`` and a second ``.backward()``: two recomputes, two
    skips, two kept entries put and taken, gradients exactly twice the direct ones."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    leaves = _ckpt_leaves(ops)
    a = torch.randn(3, requires_grad=True)

    def fn(a_, q, k, v, g, beta):
        return a_ * a_, kcp.chunk_gdn_kcp(
            q, k, v, g, beta, ctx, backend="fake", segments=2
        )

    b, out = checkpoint(fn, a, *leaves, use_reentrant=False)
    before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
    b.sum().backward()
    assert a.grad is not None and all(lf.grad is None for lf in leaves)
    assert kcp.LAST_CALL["fwd"]["fwd_o"] == _dead_decision(kcp)
    assert _skip_delta(kcp, before) == _frame_delta(kcp, 1, 0)
    assert _stats_delta(kcp, kept_before) == {
        "put": 1,
        "take": 0,
        "miss": 0,
        "released": 0,
        "callback": 1,
        "evicted": 0,
    }
    assert kcp.kept_registry_size() == 0
    del out
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
    twice = _ckpt_autograd(kcp, ctx, ops, ckpt=True, twice=True, segments=2)
    for n in CKPT_GRADS:
        assert torch.equal(twice[n], direct[n] + direct[n]), n
    assert _skip_delta(kcp, before) == _frame_delta(kcp, 2, 0)
    assert _stats_delta(kcp, kept_before) == {
        "put": 2,
        "take": 2,
        "miss": 0,
        "released": 2,
        "callback": 0,
        "evicted": 0,
    }
    assert kcp.kept_registry_size() == 0


def test_recompute_output_needs_the_verified_torch_version(monkeypatch) -> None:
    """The frame proof is read on a verified torch only: with the version constant naming another torch the recompute
    launches #5 as before (reason ``torch_version``; the frame is never read, so the counters stay), gradients
    unchanged."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, segments=2, reuse_recompute_tapes=True
    )
    monkeypatch.setattr(kcp, "RECOMPUTE_OUTPUT_TORCH_VERSIONS", ("0.0.0",))
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        res = _ckpt_autograd(
            kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=True
        )
        assert _launches(seen, "fwd_o") == [False, True]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == "torch_version"
        assert _skip_delta(kcp, before) == {"skipped": 0, "launched": 0}
    finally:
        kcp.set_backend("fake", previous)
    _assert_same(res, direct, "unverified torch: launched form vs direct")


def test_recompute_output_launched_under_a_reentrant_checkpoint() -> None:
    """``use_reentrant=True`` re-runs the segment inside the backward as well, but through the reentrant Function: no
    checkpoint frame, no saved-tensors hooks, and that re-run's output IS consumed (by the nested backward).  #5 is
    launched in both forwards (reason ``no_hooks``); gradients bitwise the direct autograd's; registry empty."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    def run(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.chunk_gdn_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                segments=2,
                reuse_recompute_tapes=True,
            )

        out = checkpoint(fn, *leaves, use_reentrant=True) if ckpt else fn(*leaves)
        out.backward(ops["dout"])
        return [lf.grad for lf in leaves]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        res = run(True)
        assert _launches(seen, "fwd_prep") == [False, True]
        assert _launches(seen, "fwd_o") == [False, True]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == _frame_reason(kcp, "no_hooks")
        assert _skip_delta(kcp, before) == {"skipped": 0, "launched": 0}
    finally:
        kcp.set_backend("fake", previous)
    for a, d in zip(res, direct, strict=True):
        assert torch.equal(a, d)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_for_a_non_executable_layer_before_the_node() -> None:
    """Two layers in one segment, the first with operands that require no grad: autograd creates no node for it
    (its ``apply`` is not executable and packs nothing), yet its output is the second layer's ``v``.  The gate read
    before ``apply`` launches its #5 in both forwards with the static reason ``not_executable`` -- the frame is
    never counted for it, so the second layer's waiting holders cannot pass for its own -- while the second layer,
    the segment's last saver, leaves #5 out of the recompute.  Output and gradients (``v`` gets none: it reaches
    the node through a constant) are bitwise the always form's and the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    ws = kcp.KcpWorkspace()

    def run(ckpt: bool, **o) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        opt = dict(
            backend="fake", segments=2, workspace=ws, reuse_recompute_tapes=True, **o
        )

        def fn(q, k, v, g, beta):
            first = kcp.chunk_gdn_kcp(
                q.detach(),
                k.detach(),
                v.detach(),
                g.detach(),
                beta.detach(),
                ctx,
                **opt,
            )
            assert not first.requires_grad
            return kcp.chunk_gdn_kcp(q, k, first, g, beta, ctx, **opt)

        out = checkpoint(fn, *leaves, use_reentrant=False) if ckpt else fn(*leaves)
        out.backward(ops["dout"])
        assert leaves[2].grad is None
        return [out.detach()] + [lf.grad for i, lf in enumerate(leaves) if i != 2]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        auto = run(True)
        fwds = _forwards(seen)
        # layer 1 original, layer 2 original, layer 1 recompute, layer 2 recompute
        assert [f[:2] for f in fwds] == [
            (False, "launched"),
            (False, "launched"),
            (True, "launched"),
            (True, _dead_decision(kcp)),
        ]
        assert fwds[0][2] == fwds[2][2] == "not_executable"
        assert fwds[3][2] == _frame_reason(kcp, "dead")
        assert _skip_delta(kcp, before) == _frame_delta(kcp, 1, 0)
        seen.clear()
        always = run(True, recompute_output="always")
        assert [f[1:] for f in _forwards(seen)] == [("launched", "always")] * 4
    finally:
        kcp.set_backend("fake", previous)
    for a, d, w in zip(auto, direct, always, strict=True):
        assert torch.equal(a, d) and torch.equal(a, w)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_under_no_grad_inside_the_recompute() -> None:
    """The GDN call under ``torch.no_grad()`` inside a segment whose later op (``o * w``, a learnable ``w``) saves
    its output: grad mode is off at ``apply``, autograd creates no node, and the gate launches #5 in the original
    and the recompute forward alike (reason ``not_executable``; the frame is never counted, the counters stay).
    ``dw`` is bitwise the always form's and the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    HV = ops["v"].shape[1]
    w0 = torch.randn(HV, 128, generator=torch.Generator().manual_seed(11)).to(
        torch.bfloat16
    )

    def run(ckpt: bool, **o) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        w = w0.detach().clone().requires_grad_(True)

        def fn(q, k, v, g, beta, w_):
            with torch.no_grad():
                out = kcp.chunk_gdn_kcp(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    backend="fake",
                    segments=2,
                    reuse_recompute_tapes=True,
                    **o,
                )
            return out * w_

        y = checkpoint(fn, *leaves, w, use_reentrant=False) if ckpt else fn(*leaves, w)
        y.backward(ops["dout"])
        assert all(lf.grad is None for lf in leaves)
        return [y.detach(), w.grad]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        auto = run(True)
        assert _forwards(seen) == [
            (False, "launched", "not_executable"),
            (True, "launched", "not_executable"),
        ]
        assert _skip_delta(kcp, before) == {"skipped": 0, "launched": 0}
        seen.clear()
        always = run(True, recompute_output="always")
        assert _forwards(seen) == [
            (False, "launched", "always"),
            (True, "launched", "always"),
        ]
    finally:
        kcp.set_backend("fake", previous)
    for a, d, w in zip(auto, direct, always, strict=True):
        assert torch.equal(a, d) and torch.equal(a, w)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_when_early_stop_is_off() -> None:
    """``set_checkpoint_early_stop(False)`` around the checkpoint: the recompute re-runs the whole segment and
    nothing bounds the waiting holders to this node's, so #5 is launched in the recompute (reason
    ``no_early_stop``); gradients bitwise the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    def run(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.chunk_gdn_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                segments=2,
                reuse_recompute_tapes=True,
            )

        if ckpt:
            with set_checkpoint_early_stop(False):
                out = checkpoint(fn, *leaves, use_reentrant=False)
        else:
            out = fn(*leaves)
        out.backward(ops["dout"])
        return [out.detach()] + [lf.grad for lf in leaves]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        res = run(True)
        assert _launches(seen, "fwd_prep") == [False, True]
        assert _launches(seen, "fwd_o") == [False, True]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == _frame_reason(
            kcp, "no_early_stop"
        )
        assert _skip_delta(kcp, before) == _frame_delta(kcp, 0, 1)
    finally:
        kcp.set_backend("fake", previous)
    for a, d in zip(res, direct, strict=True):
        assert torch.equal(a, d)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_under_a_users_saved_tensors_hooks() -> None:
    """A user's ``saved_tensors_hooks`` around the GDN call inside the segment (its later op ``o * w`` still saves
    through the checkpoint, so the recompute runs): the top of the hook stack is not torch's checkpoint hook, the
    proof is not read, and #5 is launched in both forwards (reason ``user_hooks``, counters untouched); gradients
    bitwise the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    HV = ops["v"].shape[1]
    w0 = torch.randn(HV, 128, generator=torch.Generator().manual_seed(13)).to(
        torch.bfloat16
    )

    def run(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        w = w0.detach().clone().requires_grad_(True)

        def fn(q, k, v, g, beta, w_):
            with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
                out = kcp.chunk_gdn_kcp(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    backend="fake",
                    segments=2,
                    reuse_recompute_tapes=True,
                )
            return out * w_

        y = checkpoint(fn, *leaves, w, use_reentrant=False) if ckpt else fn(*leaves, w)
        y.backward(ops["dout"])
        return [y.detach(), w.grad] + [lf.grad for lf in leaves]

    direct = run(False)
    seen, rec = _fwd_o_launches(kcp)
    previous = kcp.set_backend("fake", rec)
    try:
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        res = run(True)
        reason = _frame_reason(kcp, "user_hooks")
        assert _forwards(seen) == [
            (False, "launched", reason),
            (True, "launched", reason),
        ]
        assert _skip_delta(kcp, before) == {"skipped": 0, "launched": 0}
    finally:
        kcp.set_backend("fake", previous)
    for a, d in zip(res, direct, strict=True):
        assert torch.equal(a, d)
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- issue #7074 (1): sync-free context preparation (CPU)
# The tables of a context and its head plan travel in pinned, asynchronous copies (upload_tables); the values are those of
# torch.tensor(values, dtype).  The no-sync contract itself is measured on a device (test_cake_gdn_train.py, the dist test).


def test_upload_tables_matches_torch_tensor() -> None:
    gp = _package("cake_kcp_plan")
    specs = {
        "a": ([1, 2, 3], torch.int32, None),
        "b": ([[1, 2], [3, 4], [5, 6]], torch.int32, (3, 2)),
        "c": ((7, 8), torch.int64, None),
        "e": ([], torch.int32, (0, 4)),
        "f": ([[[9]]], torch.int32, None),
        "g": ([], torch.int64, None),
    }
    out = gp.upload_tables(specs, "cpu")
    assert set(out) == set(specs)
    for name, (values, dtype, shape) in specs.items():
        ref = torch.tensor(list(values), dtype=dtype)
        ref = ref.reshape(shape) if shape is not None else ref
        assert out[name].dtype == dtype, name
        assert tuple(out[name].shape) == tuple(ref.shape), name
        assert torch.equal(out[name], ref), name
    with pytest.raises(ValueError, match="holds 3 values"):
        gp.upload_tables({"x": ([1, 2, 3], torch.int32, (2, 2))}, "cpu")
    assert gp.UPLOAD_ALIGN_BYTES % 128 == 0


def _lpt_makespan_list(chains, machines):
    """The list form of the greedy's water filling (one load per machine): the reference the multiset form equals."""
    loads = [0] * machines
    for L, m in sorted(chains, reverse=True):
        L, m = int(L), int(m)
        if L <= 0 or m <= 0:
            continue

        def needed(level):
            return sum(-(-(level - load) // L) for load in loads if load < level)

        lo = min(loads)
        hi = lo + (m + 1) * L
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if needed(mid) <= m:
                lo = mid
            else:
                hi = mid - 1
        counts = [(-(-(lo - load) // L) if load < lo else 0) for load in loads]
        rest = m - sum(counts)
        loads = [load + c * L for load, c in zip(loads, counts, strict=True)]
        if rest:
            for i in sorted(range(machines), key=loads.__getitem__)[:rest]:
                loads[i] += L
    return max(loads)


def _lpt_makespan_greedy(chains, machines):
    """Job by job: longest first, each to the least-loaded machine."""
    loads = [0] * machines
    for L, m in sorted(chains, reverse=True):
        for _ in range(int(m)):
            if L > 0:
                loads[loads.index(min(loads))] += int(L)
    return max(loads)


def test_lpt_makespan_equals_the_list_form_and_the_greedy() -> None:
    """The multiset water filling returns the list form's integer on every instance (the plan's ``P`` vectors are
    untouched) and the job-by-job greedy's on small ones."""
    rng = random.Random(7074)
    for trial in range(400):
        machines = rng.choice([1, 2, 3, 7, 16, 132, 148, 152])
        chains = [
            (
                rng.randint(0, 40),
                rng.randint(0, 600) if rng.random() < 0.8 else rng.randint(0, 8),
            )
            for _ in range(rng.randint(0, 9))
        ]
        got = (
            gp.lpt_makespan(chains, machines)
            if any(L > 0 and m > 0 for L, m in chains)
            else None
        )
        if got is None:
            assert gp.lpt_makespan(chains, machines) == 0
            continue
        assert got == _lpt_makespan_list(chains, machines), (trial, machines, chains)
        if sum(m for _, m in chains) <= 300:
            assert got == _lpt_makespan_greedy(chains, machines), (
                trial,
                machines,
                chains,
            )
    # the issue's single-rank layout: two fragments of 2048 chunks, P * HV chains each, 152 SMs
    assert gp.lpt_makespan([(2048, 6), (2048, 6)], 152) == 2048
    assert gp.lpt_makespan([(8, 6 * 256), (8, 6 * 256)], 152) == 8 * -(
        -(2 * 6 * 256) // 152
    )
    with pytest.raises(ValueError):
        gp.lpt_makespan([(1, 1)], 0)


# --------------------------------------------------------------------------- issue #7074 (3): the recompute_output gate is a torch RELEASE check
# Every CUDA build of the verified release (2.14.1+cu130, 2.14.1+cu132, ...) reads the frame proof; other releases launch #5 as before.


@pytest.mark.parametrize(
    "version,verified",
    [
        ("2.14.1+cu130", True),
        ("2.14.1+cu132", True),
        ("2.14.1", True),
        ("2.14.0+cu130", False),
        ("2.15.0+cu132", False),
        ("2.14.1.dev20260901+cu130", False),
    ],
)
def test_recompute_output_gate_is_a_torch_release_check(
    monkeypatch, version, verified
) -> None:
    """The verified set names torch releases; the local ``+cuNNN`` build tag is not part of the identity."""
    kcp = _package("cake_kcp")
    assert kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS == ("2.14.1",)
    assert kcp.torch_release(version) == version.split("+", 1)[0]
    monkeypatch.setattr(torch, "__version__", version)
    assert kcp.torch_release() == version.split("+", 1)[0]
    assert kcp.torch_release_verified() is verified
    dead, reason = kcp._recompute_output_dead(6)  # outside any checkpoint: never dead
    assert dead is False
    assert (reason == "torch_version") is (not verified), reason


def test_recompute_output_auto_resolves_on_every_cuda_build_of_the_release(
    monkeypatch,
) -> None:
    """The resolved mode of ``recompute_output="auto"`` per torch build string: the two CUDA builds of the verified
    release skip the recompute's #5 (reason ``dead``), the neighbouring releases launch it (``torch_version``)."""
    kcp, ctx, ops = _ckpt_setup()
    if kcp.torch_release() not in kcp.RECOMPUTE_OUTPUT_TORCH_VERSIONS:
        pytest.skip(
            "the running torch is not a verified release: the frame proof cannot be exercised"
        )
    direct = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, segments=2, reuse_recompute_tapes=True
    )
    expect = {
        "2.14.1+cu130": ("dead", [False], "skipped"),
        "2.14.1+cu132": ("dead", [False], "skipped"),
        "2.14.0+cu130": ("torch_version", [False, True], "launched"),
        "2.15.0+cu132": ("torch_version", [False, True], "launched"),
    }
    for build, (reason, fwd_o, decision) in expect.items():
        monkeypatch.setattr(torch, "__version__", build)
        seen, rec = _fwd_o_launches(kcp)
        previous = kcp.set_backend("fake", rec)
        try:
            res = _ckpt_autograd(
                kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=True
            )
            assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == reason, build
            assert kcp.LAST_CALL["fwd"]["fwd_o"] == decision, build
            assert _launches(seen, "fwd_o") == fwd_o, build
        finally:
            kcp.set_backend("fake", previous)
        _assert_same(res, direct, build)


# --------------------------------------------------------------------------- issue #7074 (4): gdn_kcp_workspace_layout follows the arena's lifetimes
# What a caller-held workspace binds, per tape form: the forward's tape-walk peak and the backward's phase maximum.


class _PeakWorkspace:
    """A ``KcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.KcpWorkspace):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.peak = 0

            def _upd(self):
                self.peak = max(self.peak, self.bytes)

            def tape(self, name):
                t = super().tape(name)
                self._upd()
                return t

            def pool(self, name, n=None):
                t = super().pool(name, n)
                self._upd()
                return t

            def ensure(self, *a, **k):
                tb = super().ensure(*a, **k)
                self._upd()
                return tb

            def adopt_tape(self, name, t):
                r = super().adopt_tape(name, t)
                self._upd()
                return r

        return Tracked()


def test_workspace_layout_follows_the_arena_lifetimes() -> None:
    """``gdn_kcp_workspace_layout`` is what a caller-held workspace binds: the forward's tape-walk peak, and the
    backward's phase maximum of the form -- the kept-tape form ends with the four re-adopted forward tapes beside the
    backward tapes (the ten kept tapes are hand-off tensors outside the workspace, ``bwd.kept_tapes``), the replay
    form holds the ten light tapes through the backward; both forms need the same transient bytes."""
    kcp, ctx, ops = _ckpt_setup()
    gb = _package("cake_bytes")
    H_qk, HV = 2, 6
    assert (
        tuple(gb.TAPE_NAME_OF.get(n, n) for n in gb.FWD_TAPES_DEAD_AFTER_PREP)
        == kcp.FWD_DEAD_AFTER_PREP
    )
    assert (
        tuple(gb.TAPE_NAME_OF.get(n, n) for n in gb.FWD_TAPES_ARENA_AFTER_FWD)
        == kcp.FWD_TAPES_ARENA_AFTER_FWD
    )
    assert gb.EYE_BYTES == gb.STATE_TILE_BYTES == 128 * 128 * 4
    from torch.utils.checkpoint import checkpoint

    layouts = {}
    for form in (True, False):
        layout = gb.gdn_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
        )
        layouts[form] = layout
        ws = _PeakWorkspace(kcp)
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

        def fn(q, k, v, g, beta, _ws=ws, _form=form):
            return kcp.chunk_gdn_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                workspace=_ws,
                backend="fake",
                reuse_recompute_tapes=_form,
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        assert ws.peak == layout["forward"], (form, ws.peak, layout["forward"])
        ws.peak = 0
        out.backward(ops["dout"])
        assert ws.peak == layout["backward"], (form, ws.peak, layout["backward"])
        assert ws.bytes == layout["bwd.phase.end"], (
            form,
            ws.bytes,
            layout["bwd.phase.end"],
        )
        assert layout["bwd.phase.end"] == layout["bwd.eye"] + layout[
            "bwd.tickets"
        ] + sum(
            layout[f"bwd.{n}"]
            for n in ("do_g", "dv_intra", "dh_tape", "dg_part", "gamma_dot")
        ) + sum(
            layout[f"bwd.{n}"]
            for n in (
                gb.FWD_TAPES_ARENA_AFTER_FWD
                if form
                else tuple(k for k in gb.FWD_TAPES_ARENA_AFTER_FWD)
                + ("w", "u", "akk", "gamma", "gcum", "rowscale")
            )
        )
        assert kcp.kept_registry_size() == 0
    kept, replay = layouts[True], layouts[False]
    assert kept["backward"] < replay["backward"]
    assert kept["bwd.kept_tapes"] == sum(
        replay[f"bwd.{n}"]
        for n in (
            "w",
            "u",
            "aqk",
            "akk",
            "h_tape",
            "v_new",
            "gamma",
            "gcum",
            "rowscale",
            "qscale",
        )
    )
    assert replay["bwd.kept_tapes"] == 0
    assert (
        kept["backward_transient"] == replay["backward_transient"] == replay["backward"]
    )
    assert kept["forward"] == replay["forward"]
    # the keyword resolves through the package default, and the bytes query is the layout's maximum
    default = gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
    opt = _package("cake_options")
    assert default == layouts[opt.REUSE_RECOMPUTE_TAPES_DEFAULT is True]
    assert (
        gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="auto")
        == replay
    )
    for form in (True, False):
        assert (
            gb.gdn_kcp_workspace_bytes(
                ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
            )
            == layouts[form]["workspace"]
        )
    with pytest.raises(ValueError, match="reuse_recompute_tapes must be"):
        gb.gdn_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="yes")


# --------------------------------------------------------------------------- issue #7074 (2): reuse_recompute_tapes="auto"
# The recompute's tapes are kept only when the GDN node closes the checkpointed region (the frame proof); with a later saver the
# kept set would live through that operation's backward and raise the region's peak by the full tape set, so "auto" replays there.


def _ckpt_autograd_region(
    kcp, ctx, ops, *, ckpt: bool, later_saver: bool, **opts
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """``out`` and the gradients of a region that ends with the GDN call (``later_saver=False``) or continues with an
    operation that saves tensors for its backward (a scaling by a leaf, ``later_saver=True``), directly or under a
    non-reentrant checkpoint; the leaf's gradient comes beside the result."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    scale = torch.full((), 1.5, dtype=torch.bfloat16, requires_grad=True)
    o = {"backend": "fake", **opts}

    def fn(q, k, v, g, beta):
        out = kcp.chunk_gdn_kcp(q, k, v, g, beta, ctx, **o)
        return out * scale if later_saver else out

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, use_reentrant=False)
    else:
        out = fn(*leaves)
    out.backward(ops["dout"])
    res = {"out": out.detach()}
    res.update({n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)})
    return res, (scale.grad if later_saver else None)


@pytest.mark.parametrize("segments,save_transfer", [(1, True), (2, True), (2, False)])
def test_reuse_recompute_tapes_auto_keeps_tapes_only_when_the_node_closes_the_region(
    segments, save_transfer
) -> None:
    """``"auto"``: the recompute hands its tapes to the backward when the GDN node is the region's last saver (the
    frame proof ``dead``; the backward of the region then starts with this node) and takes the replay form when a
    later operation of the region saves a tensor (``holders_remain``) -- so the kept set never outlives the GDN
    backward.  ``True`` always hands off, ``False`` never; every form is bitwise the direct autograd result, and
    the registry is empty afterwards.  On an unverified torch release ``"auto"`` is the replay form."""
    kcp, ctx, ops = _ckpt_setup()
    base = {"segments": segments, "save_transfer": save_transfer}
    for later_saver in (False, True):
        direct, direct_scale = _ckpt_autograd_region(
            kcp, ctx, ops, ckpt=False, later_saver=later_saver, **base
        )
        for mode in (True, False, "auto"):
            before = dict(kcp.KEPT_STATS)
            res, scale_grad = _ckpt_autograd_region(
                kcp,
                ctx,
                ops,
                ckpt=True,
                later_saver=later_saver,
                reuse_recompute_tapes=mode,
                **base,
            )
            what = f"later_saver={later_saver} mode={mode!r} {base}"
            _assert_same(res, direct, what)
            if later_saver:
                assert torch.equal(scale_grad, direct_scale), what
            delta = _stats_delta(kcp, before)
            reason = kcp.LAST_CALL["fwd"]["hand_off_reason"]
            source = kcp.LAST_CALL["bwd"]["tapes_source"]
            if mode is True:
                expect, kept = (1, 1, 0), True
                assert reason == "always", what
            elif mode is False:
                expect, kept = (0, 0, 0), False
                assert reason == "off", what
            elif later_saver or not _verified_torch(kcp):
                expect, kept = (0, 0, 0), False
                assert reason == _frame_reason(kcp, "holders_remain"), what
            else:
                expect, kept = (1, 1, 0), True
                assert reason == "dead", what
            assert (delta["put"], delta["take"], delta["miss"]) == expect, (what, delta)
            assert source == ("kept" if kept else "replay"), (what, source)
            assert kcp.kept_registry_size() == 0, what


def test_reuse_recompute_tapes_auto_is_an_option_value() -> None:
    opt = _package("cake_options")
    assert opt.REUSE_RECOMPUTE_TAPES == (True, False, "auto")
    assert opt.REUSE_RECOMPUTE_TAPES_DEFAULT in opt.REUSE_RECOMPUTE_TAPES
    assert (
        opt.KcpOptions(reuse_recompute_tapes="auto").check().reuse_recompute_tapes
        == "auto"
    )
    assert (
        opt.KcpOptions().check().reuse_recompute_tapes
        == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    )
    assert opt.KcpOptions(graph="replay").check().reuse_recompute_tapes is False
    with pytest.raises(ValueError, match="needs graph='off'"):
        opt.KcpOptions(graph="replay", reuse_recompute_tapes="auto").check()
    with pytest.raises(ValueError, match="needs graph='off'"):
        opt.KcpOptions(graph="replay", reuse_recompute_tapes=True).check()
    for bad in ("yes", 1, 0, "AUTO"):
        with pytest.raises(TypeError, match="must be a bool or 'auto'"):
            opt.KcpOptions(reuse_recompute_tapes=bad).check()
