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

CPU tests of the GDN KCP metadata layer of ``flashinfer.experimental.cake.cake_gdn_det_train_kcp``: the package-local plan
module against ``cake_kda_det_train_kcp.cake_kcp_plan`` (field-identical partitions and contexts on the step-0 rows with
``H := HV``, except the b1 buffer with maps, ``B1_DM_FIELDS``, which has this package's own dense layout; the KDA
package is imported HERE only, never by the package), the per-rank crossing counts and
exchange bytes against the rows of record, the table builders (pool addressing, segment lists, chains), head
ownership incl. the padded ``(HV, W) = (3, 2)`` case, the byte queries, the registry seed, and the fake backend
against the FP64 reference (bitwise repeatable, saved == replay).

Environment of the tests (never of the package): ``GDN_ROWS_DIR`` (the step-0 row JSONs; the row tests skip
without it), ``GDN_EVIDENCE_DIR`` (JSONL evidence records), the FP64 reference ``gdn_reference`` on ``PYTHONPATH``
(the fake-backend tests skip without it).
"""

from __future__ import annotations

import ast
import collections.abc
import dataclasses
import glob
import hashlib
import importlib
import itertools
import json
import math
import os
import pathlib
import random
import re
import threading
import types

import pytest
import torch


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU tests run many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (the forward-form suite took about two hours on a shared CI host, against
    11 s on a GPU node), so this module runs with at most 8 threads and oneDNN off, as the non-deterministic KCP suite
    does."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _package(submodule: str):
    try:
        from tests.cake.test_helpers.cake_package import cake_package

        return cake_package("cake_gdn_det_train_kcp", submodule)
    except KeyError:  # the registry entry is added by the package scaffolding lane
        return importlib.import_module(
            f"flashinfer.experimental.cake.cake_gdn_det_train_kcp.{submodule}"
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
    """The ``cake_gdn_det_train_kcp_partition`` call of a row JSON (layout C / Z / D / R)."""
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
    # the contexts are plain classes listing their fields in ``_FIELDS``; the other plan objects are dataclasses
    fields = getattr(type(x), "_FIELDS", None) or getattr(
        x, "__dataclass_fields__", None
    )
    if fields is not None:
        return {
            f: _plain(getattr(x, f))
            for f in fields
            if not f.startswith("_") and f != "sequence"
        }
    if isinstance(
        x, collections.abc.Mapping
    ):  # dicts and the KDA module's lazily uploaded table mapping
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
    # the KDA module keeps the fragment table as host rows plus a lazily uploaded property and records its upload
    # knobs; compare the table itself on both sides
    for name in ("frag_rows", "upload", "table_cache"):
        rec.pop(name, None)
    rec["frag_table"] = _plain(ctx.frag_table)
    rec["device"] = str(ctx.device)
    rec["heads"] = _plain(ctx.heads(HV))
    rec["describe"] = ctx.describe(HV)
    return rec


# The b1 buffer with re-sent maps is this package's own compact layout (the d_f plane of the crossing-in slots, then
# the maps of the fragments crossing in and out); the KDA module keeps one [d_f | M_f^T] pair per crossing-in slot,
# whose map half of a document's last fragment is never written.  The parity record leaves that buffer out.
B1_DM_FIELDS = (
    "b1_d_tile_dm",
    "b1_m_tile_dm",
    "b1_m_index",
    "n_io_by_rank",
    "adj_d_off_dm",
    "adj_m_off_b1",
)
B1_DM_KEYS = (*B1_DM_FIELDS, "b1_send_dm", "b1_recv_dm", "b1_dm")


def _drop_b1_dm(rec: dict) -> dict:
    """``rec`` (a :func:`_context_record`) without the fields and table keys of the b1 buffer with maps."""
    for heads in (rec["heads"], rec["describe"].get("heads", {})):
        for name in B1_DM_FIELDS:
            heads.pop(name, None)
        for table in ("splits", "numel", "pad_numel", "hg_stride", "tile_stride"):
            if isinstance(heads.get(table), dict):
                for key in B1_DM_KEYS:
                    heads[table].pop(key, None)
        if isinstance(heads.get("device_tables"), dict):
            for key in B1_DM_KEYS:
                heads["device_tables"].pop(key, None)
    return rec


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_plan_parity_with_kda_on_rows(path, monkeypatch) -> None:
    """Result 1: ``cake_gdn_det_train_kcp.cake_kcp_plan`` is field-identical to ``cake_kda_det_train_kcp.cake_kcp_plan`` on every
    step-0 row (partition, every rank's context and head plan at ``H := HV``), except the b1 buffer with maps
    (``B1_DM_FIELDS`` and its split / table keys, checked here for tile coverage and offset range instead), and
    reproduces the row's recorded fragment table.  Result 2: ``n_in`` / ``n_out`` / ``max_chain_fragments`` and the
    exchange bytes of ``cake_gdn_det_train_kcp_comm_bytes`` equal the row JSON.  Both modules plan under the per-rank rule they
    share, ``"wave_fill1"`` (the GDN default is the row rule, the KDA default its own tie rule; each is selected
    explicitly here: the keyword-only ``p_rule`` on the GDN side, the private test hook on the KDA side)."""
    kp = importlib.import_module(
        "flashinfer.experimental.cake.cake_kda_det_train_kcp.cake_kcp_plan"
    )  # the test only
    # the KDA module accepts its private test hook only after its opt-in (enable_test_knobs)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    with open(path) as fh:
        row = json.load(fh)
    H_qk, HV = (int(x) for x in row["heads"])
    W = int(row["cp"])
    args = _partition_inputs(row)
    part = gp.cake_gdn_det_train_kcp_partition(**args)
    kpart = kp.cake_kda_det_train_kcp_partition(**args)
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
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )  # the per-rank rule both modules share; the GDN default "row_cost" is tested on the rows below
        kctx = kp.cake_kda_det_train_kcp_build_context(
            kpart,
            None,
            r,
            CPU,
            4,
            sm_count=SM_COUNT,
            _kcp_test_knobs={"p_rule": "wave_fill1"},
        )
        mine = _drop_b1_dm(_context_record(ctx, HV))
        theirs = _drop_b1_dm(_rename_h(_context_record(kctx, HV)))
        records.append(mine)
        assert mine == theirs, (
            f"rank {r}: context / head plan differ from the KDA module (HV={HV})"
        )
        assert (ctx.n_in, ctx.n_out) == (n_in[r], n_out[r])
        # the b1 buffer with maps: every send tile written once (d_f tiles + the maps of the fragments crossing in
        # and out), the owner offsets inside the receive buffer
        hp = ctx.heads(HV)
        n_io = sum(1 for f in part.fragments[r] if f.crossing_in and f.crossing_out)
        dm_tiles = sorted(
            t
            for rows in (hp.b1_d_tile_dm, hp.b1_m_tile_dm)
            for row in rows
            for t in row
        )
        assert dm_tiles == list(range(sum(hp.splits["b1_send_dm"]) // gp.TILE))
        assert len(hp.b1_m_tile_dm) == hp.n_io_by_rank[r] == n_io
        assert all(0 <= o < hp.numel["b1_recv_dm"] for o in hp.adj_d_off_dm)
        assert all(o == -1 or 0 <= o < hp.numel["b1_recv_dm"] for o in hp.adj_m_off_b1)
        cb = gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
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
    part = gp.cake_gdn_det_train_kcp_partition(**_partition_inputs(row))
    chains = part.chains()
    multi = [c for c in chains if len(c) >= 2]
    summary = []
    for r in range(W):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
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
    return gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), W, layout
    )


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
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(W)
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
        cb = gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=1, HV=HV, C=C_CONV)
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


@dataclasses.dataclass(frozen=True)
class _BatchMeta:
    """Per-batch metadata a model hands to every layer (the shape of the FSDP2 input-walk report)."""

    cu_seqlens: torch.Tensor
    kcp: object


def _zigzag_fragments(lengths: list[int], W: int) -> list[list[tuple[int, int, int]]]:
    """Every rank's ``(doc, offset, length)``: pieces ``r`` and ``2W - 1 - r`` of width ``ceil(L / 2W)``."""
    frags: list[list[tuple[int, int, int]]] = [[] for _ in range(W)]
    for doc, length in enumerate(lengths):
        width = math.ceil(length / (2 * W))
        cuts = [min(i * width, length) for i in range(2 * W)] + [length]
        for r in range(W):
            for i in (r, 2 * W - 1 - r):
                frags[r].append((doc, cuts[i], cuts[i + 1] - cuts[i]))
    return frags


def test_context_is_a_leaf_for_input_walks(monkeypatch) -> None:
    """A prepared context, CP or single-rank, is one leaf for the input walks that recurse into dataclass, dict,
    list and tuple containers (FSDP2's ``fully_shard`` pre-forward hook collects and replaces the grad-requiring
    tensors of a module's arguments that way): the walk visits as many nodes as with no context, at any fragment
    or token count, and the replace pass returns the metadata object itself.  The context keeps the frozen
    dataclass behaviour it had: the constructor, frozen attributes, identity equality, the repr."""
    fsdp_utils = pytest.importorskip("torch.distributed.fsdp._common_utils")
    if not hasattr(fsdp_utils, "_collect_grad_tensors"):
        pytest.skip("this torch's FSDP has no recursive input walk")
    visits = [0]
    collect = fsdp_utils._collect_grad_tensors

    def counting(output, out, _depth=0):
        visits[0] += 1
        return collect(output, out, _depth)

    monkeypatch.setattr(fsdp_utils, "_collect_grad_tensors", counting)

    def walk(meta) -> int:
        visits[0] = 0
        assert fsdp_utils.collect_grad_tensors(((meta,), {"meta": meta})) == ()
        assert fsdp_utils.replace_grad_tensors(meta, iter(())) is meta
        return visits[0]

    cu = torch.tensor([0, 4000], dtype=torch.int64)
    baseline = walk(_BatchMeta(cu, None))
    assert walk(_BatchMeta(cu, (None, None))) > baseline  # the walk recurses
    # the mixed-precision input cast (``MixedPrecisionPolicy(param_dtype=...)``) rebuilds the dataclasses it walks
    apply_to_tensors = getattr(
        importlib.import_module("torch.distributed.utils"), "_apply_to_tensors", None
    )
    for n_docs, W in ((2, 8), (64, 8), (64, 16)):
        lengths = [4000] * n_docs
        cu = torch.tensor([0, *itertools.accumulate(lengths)], dtype=torch.int64)
        part = gp.cake_gdn_det_train_kcp_partition(
            cu, W, fragments=_zigzag_fragments(lengths, W)
        )
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, 0, CPU, 4, sm_count=SM_COUNT
        )
        ctx.prepare(1, 48)
        single = gp.cake_gdn_det_train_kcp_build_context(
            gp.cake_gdn_det_train_kcp_partition(cu, 1, "contiguous"),
            None,
            0,
            CPU,
            4,
            sm_count=SM_COUNT,
        )
        single.prepare(1, 48 // W)
        for c in (ctx, single):
            assert not dataclasses.is_dataclass(c)
            assert walk(_BatchMeta(cu, c)) == baseline, (n_docs, W)
            if apply_to_tensors is not None:
                assert apply_to_tensors(lambda t: t, _BatchMeta(cu, c)).kcp is c
            nested = walk(_BatchMeta(cu, {"kcp": [None, (None,)]}))
            assert walk(_BatchMeta(cu, {"kcp": [c, (c,)]})) == nested

    # the frozen-dataclass behaviour, unchanged
    with pytest.raises(dataclasses.FrozenInstanceError):
        ctx.rank = 1
    with pytest.raises(dataclasses.FrozenInstanceError):
        del ctx.T_r
    clone = gp.CakeGDNDetKcpContext(
        **{f: getattr(ctx, f) for f in gp.CakeGDNDetKcpContext._FIELDS}
    )
    assert clone != ctx and clone == clone and hash(clone) != hash(ctx)
    assert all(
        getattr(clone, f) is getattr(ctx, f) for f in gp.CakeGDNDetKcpContext._FIELDS
    )
    assert gp.CakeGDNDetKcpContext._FIELDS[-4:] == (
        "p_rule",
        "sequence",
        "_head_plans",
        "_caches",
    )
    fresh = gp.CakeGDNDetKcpContext(
        *(getattr(ctx, f) for f in gp.CakeGDNDetKcpContext._FIELDS[:-4])
    )
    assert (
        fresh.p_rule == gp.P_RULE_DEFAULT
        and fresh._head_plans == {}
        and fresh._caches == {}
    )
    assert (
        isinstance(fresh.sequence, gp.KcpSequence)
        and fresh.sequence is not ctx.sequence
    )
    assert (
        gp.CakeGDNDetKcpContext(
            *(getattr(ctx, f) for f in gp.CakeGDNDetKcpContext._FIELDS[:-3]), None
        ).sequence
        is None
    )
    text = repr(ctx)
    assert (
        text.startswith("CakeGDNDetKcpContext(partition=")
        and f"p_rule={ctx.p_rule!r})" in text
    )
    assert "sequence=" not in text and "_caches" not in text


def test_prepare_tables_first_use() -> None:
    """``ctx.prepare(H_qk, HV)`` builds the device-resident bundle once per ``(HV, sm_count, P)``
    key at first use (``cake_gdn_det_train_kcp_build_context`` builds no head-dependent table; the second call returns the same
    object; a ``P`` override gets its own key and plan), rejects ``HV % H_qk != 0`` and an out-of-range ``P``, and
    every table dtype is int32 except the merge's int64 element offsets."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)  # C1
    part = gp.cake_gdn_det_train_kcp_partition(cu, 2, "contiguous")
    ctx = gp.cake_gdn_det_train_kcp_build_context(
        part, None, 1, CPU, 4, sm_count=SM_COUNT
    )
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
    part = gp.cake_gdn_det_train_kcp_partition(
        cu, 4, fragments=_ragged_fragments([0, 1700, 3000], [1200, 0, 900, 900])
    )
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(4)
    ]
    assert ctxs[1].T_r == 0 and ctxs[1].n_frag == 0 and gt.local_tables(ctxs[1]).NC == 0
    wt1 = gt.walk_tables(ctxs[1], 24)
    assert wt1.NSEG == 0 and wt1.n_walk_offset == 0 and wt1.Hg_r == 6  # still an owner
    assert (
        wt1.chains["fwd"].n_chains == 2 * wt1.Hg_r
    )  # two document chains of 2 fragments (ranks 0 -> 2, 2 -> 3)
    cu5 = torch.tensor([0, 63 + 64 + 65 + 500], dtype=torch.int64)
    part5 = gp.cake_gdn_det_train_kcp_partition(
        cu5, 4, fragments=_ragged_fragments([0, 692], [63, 64, 65, 500])
    )
    for r, want in enumerate((1, 1, 2, 8)):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part5, None, r, CPU, 4, sm_count=SM_COUNT
        )
        lt = gt.local_tables(ctx)
        assert want == lt.NC and lt.chunk_len[0] == min(64, ctx.T_r)
    with pytest.raises(ValueError, match="p_rule"):
        gp.cake_gdn_det_train_kcp_build_context(
            part5, None, 0, CPU, 4, sm_count=SM_COUNT, p_rule="nope"
        )


# --------------------------------------------------------------------------- the row rules ("row_cost_lpt" shipped, "row_cost" selectable)

# Row keys of ROW_COST_CHANGED / ROW_COST_LPT_CHANGED / SMALL_ONE_CHANGED / OPT_IN_CHANGED (_row_key): (row, cp, layout),
# the layout suffixed "_fla" on an FLA-padded row (the "_fla" row files): its fragment tables differ from those of its
# unpadded twin, and so may its vector.
ROW_COST_CHANGED = {  # (row, cp, layout) -> the P vector the "row_cost" row rule picks where it departs from "wave_fill1"
    ("P1", 2, "D"): (1, 1),
    ("P2", 2, "Z"): (4, 4),
    # P3 CP2 C: admitted by the "one" margin of 0 (model ratio 0.980; "occ" keeps its 2 %)
    ("P3", 2, "C"): (1, 1),
    ("P1", 4, "D"): (1, 1, 1, 1),
    # The four rows above are the departures on the unpadded HV = 48 rows at CP <= 4.  The departures on every other
    # step-0 row of record (HV 3 to 48, CP 2 to 16, the FLA-padded twins) follow, generated by a census of all 107 rows
    # at SM_COUNT from the det plan module sha256/16 aec7abcaa1daae3d, whose "row_cost" vectors equal those of the
    # module that recorded the four rows above; never retyped.  Each switches the whole row to "one", except
    # ("P2", 2, "Z_fla"), which switches to "occ" like its unpadded twin.
    ("A1", 8, "C"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("A2", 8, "C"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("A4", 16, "C"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("C1", 2, "C"): (1, 1),
    ("C1", 4, "C"): (1, 1, 1, 1),
    ("C2", 4, "C"): (1, 1, 1, 1),
    ("C2", 4, "D"): (1, 1, 1, 1),
    ("C2", 4, "Z"): (1, 1, 1, 1),
    ("C3", 4, "D"): (1, 1, 1, 1),
    ("C4a", 4, "R"): (1, 1, 1, 1),
    ("C4b", 4, "R"): (1, 1, 1, 1),
    ("C5a", 4, "R"): (1, 1, 1, 1),
    ("C5b", 4, "R"): (1, 1, 1, 1),
    ("C6", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("C7", 4, "R"): (1, 1, 1, 1),
    ("C7b", 4, "R"): (1, 1, 1, 1),
    ("C8", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P1", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P1", 8, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P1", 16, "Z"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P1", 16, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P2", 2, "Z_fla"): (4, 4),
    ("P2", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P2", 8, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P2", 16, "Z"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P2", 16, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P3", 2, "C_fla"): (1, 1),
    ("P3", 4, "Z_fla"): (1, 1, 1, 1),
    ("P3", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P3", 8, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P3", 16, "Z"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P3", 16, "Z_fla"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P4", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P4", 16, "Z"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    ("P5", 8, "Z"): (1, 1, 1, 1, 1, 1, 1, 1),
    ("P5", 16, "Z"): (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
}

# SM_COUNT = 152 (GB300; the GB200 of record reports 152 too).  Other SM counts have no fixture: the census's model-only
# columns are not pinned.  Source: generated by r13_mod13_census.py 2026-10-28T03:09:15Z from the plan module sha256/16 df8e02283700caa3 (rule row_cost_lpt, HV 48, SM 152); never retyped
ROW_COST_LPT_CHANGED = {  # (row, cp, layout) -> the P vector the shipped "row_cost_lpt" rule picks where it departs from "row_cost"
    ("P1", 4, "Z"): (2, 2, 3, 3),
    ("P3", 4, "C"): (6, 6, 6, 6),
    ("P3", 4, "Z"): (3, 3, 3, 3),
    ("P5", 4, "C"): (3, 6, 6, 6),
    ("P5", 4, "Z"): (3, 3, 3, 3),
    # The FLA-padded twins, generated by the census of all 107 step-0 rows at SM_COUNT from the det plan module
    # sha256/16 aec7abcaa1daae3d; never retyped.  No other row departs.  ("P3", 4, "Z_fla") is not listed: its
    # "row_cost" vector is already P = 1 on every rank, which the rule keeps.
    ("P1", 4, "Z_fla"): (2, 2, 3, 3),
    ("P2", 4, "C_fla"): (3, 6, 3, 6),
    ("P3", 4, "C_fla"): (6, 6, 6, 6),
}
# (row, cp, layout) -> the P vector the default "small_one" rule picks where it departs from "row_cost_lpt" at SM_COUNT.
# "small_one" stays the default; each row it moves off "row_cost_lpt" is a recorded departure.  Generated by the census
# of all 107 step-0 rows from the det plan module sha256/16 aec7abcaa1daae3d; never retyped.  No other row departs.
SMALL_ONE_CHANGED = {
    # A3 CP4 D (HV 6, every fragment <= 68 chunks): measured per rank "small_one" P = 1 / NSEG 4 vs "row_cost_lpt" P = 7 / NSEG 28; row_cost_10ns 51340 vs 39580
    ("A3", 4, "D"): (1, 1, 1, 1),
}
# (row, cp, layout) -> {rule: the P vector} of the opt-in row rules ("fill_down", "small_one", "fill_small") where they
# depart from "row_cost_lpt", the same at SM_COUNT and 148 (test_opt_in_row_rules_keep_the_rows_of_record).  Each is the
# rule as defined: "fill_down" lowers the CP8 / CP16 ranks below the chain cap whose one-wave fill is lower, and the
# HV = 6 row of SMALL_ONE_CHANGED runs P = 1.  Generated by the census of all 107 step-0 rows from the det plan module
# sha256/16 aec7abcaa1daae3d; never retyped.  No other (row, rule) departs.
OPT_IN_CHANGED = {
    # A3 CP4 D: the SMALL_ONE_CHANGED row ("fill_small" is "small_one" on it)
    ("A3", 4, "D"): {
        "small_one": (1, 1, 1, 1),
        "fill_small": (1, 1, 1, 1),
    },
    ("C8", 8, "C"): {
        "fill_down": (3, 3, 3, 1, 3, 1, 3, 3),
        "fill_small": (3, 3, 3, 1, 3, 1, 3, 3),
    },
    ("P1", 8, "C"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 1, 3),
        "fill_small": (3, 3, 3, 3, 3, 3, 1, 3),
    },
    ("P1", 8, "C_fla"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 1, 2),
        "fill_small": (3, 3, 3, 3, 3, 3, 1, 2),
    },
    ("P1", 16, "C"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1),
        "fill_small": (3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1),
    },
    ("P1", 16, "C_fla"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1),
        "fill_small": (3, 3, 3, 3, 3, 3, 3, 3, 1, 3, 1, 3, 1, 1, 3, 1),
    },
    ("P2", 16, "C"): {
        "fill_down": (3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3),
        "fill_small": (3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3),
    },
    ("P2", 16, "C_fla"): {
        "fill_down": (3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1),
        "fill_small": (3, 1, 3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 1),
    },
    ("P3", 16, "C"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3),
        "fill_small": (3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 3),
    },
    ("P3", 16, "C_fla"): {
        "fill_down": (3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1),
        "fill_small": (3, 3, 3, 3, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3, 1),
    },
    ("P4", 8, "D"): {
        "fill_down": (1, 3, 3, 1, 1, 1, 1, 3),
        "fill_small": (1, 3, 3, 1, 1, 1, 1, 3),
    },
    ("P4", 16, "D"): {
        "fill_down": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        "fill_small": (1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
    },
    ("P5", 16, "C"): {
        "fill_down": (3, 1, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3),
        "fill_small": (3, 1, 3, 3, 1, 3, 3, 3, 1, 3, 3, 3, 3, 3, 3, 3),
    },
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


def _row_key(row: dict) -> tuple[str, int, str]:
    """The key of a step-0 row in ``ROW_COST_CHANGED`` / ``ROW_COST_LPT_CHANGED`` / ``SMALL_ONE_CHANGED`` /
    ``OPT_IN_CHANGED``: ``(row, cp, layout)``, the layout suffixed ``"_fla"`` on an FLA-padded row."""
    return (
        row["row"],
        int(row["cp"]),
        row["layout"] + ("_fla" if row["fla_padded"] else ""),
    )


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_row_cost_rule_on_rows_of_record(path) -> None:
    """The ``"row_cost"`` row rule (selectable; the base the shipped ``"row_cost_lpt"`` starts from) on every step-0 row:
    the rows of ``ROW_COST_CHANGED`` get their recorded vector, every other row keeps the per-rank ``"wave_fill1"``
    vector; every rank's context derives the same vector (same topology, same ``P`` on every rank) and reports
    ``p_rule="row_cost"``."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.cake_gdn_det_train_kcp_partition(**_partition_inputs(row))
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
    key = _row_key(row)
    want = ROW_COST_CHANGED.get(key, shipped)
    assert got == want, (key, got, want, shipped)
    assert (got != shipped) == (key in ROW_COST_CHANGED)
    assert all(
        seg.p_rule == "row_cost" and seg.n_frag * seg.P == seg.NSEG for seg in plans
    )
    for r in range(W):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        hp = ctx.heads(HV)
        assert ctx.p_rule == "row_cost" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        # only the segment program differs from the per-rank plan: partition, fragment table, local tables and
        # crossing lists are the same object graph values
        ref = gp.cake_gdn_det_train_kcp_build_context(
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
    """The ``"row_cost_lpt"`` row rule on every step-0 row: the rows of ``ROW_COST_LPT_CHANGED`` get their recorded
    vector, every other row keeps the ``"row_cost"`` vector; a rank the ``"row_cost"`` rule plans at ``P = 1`` keeps
    ``P = 1``; every rank's ``"row_cost_lpt"`` context derives the same vector and reports ``p_rule="row_cost_lpt"``, and
    the default context (``"small_one"``) plans the same ``P`` except on the rows of ``SMALL_ONE_CHANGED``, which get
    their recorded vector; only the segment program differs from the ``"row_cost"`` context."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.cake_gdn_det_train_kcp_partition(**_partition_inputs(row))
    tables = gp.row_tables(part)
    base = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost")
    plans = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="row_cost_lpt")
    got = tuple(seg.P for seg in plans)
    row_cost = tuple(seg.P for seg in base)
    key = _row_key(row)
    want = ROW_COST_LPT_CHANGED.get(key, row_cost)
    assert got == want, (key, got, want, row_cost)
    assert (got != row_cost) == (key in ROW_COST_LPT_CHANGED)
    assert all(
        seg.p_rule == "row_cost_lpt" and seg.n_frag * seg.P == seg.NSEG for seg in plans
    )
    assert all(new.P == 1 for old, new in zip(base, plans, strict=True) if old.P == 1)
    dflt_plans = gp.kcp_plan_row(tables, HV, SM_COUNT, p_rule="small_one")
    dflt_got = tuple(seg.P for seg in dflt_plans)
    assert dflt_got == SMALL_ONE_CHANGED.get(key, got), (key, dflt_got, got)
    assert (dflt_got != got) == (key in SMALL_ONE_CHANGED)
    for r in range(W):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost_lpt"
        )
        hp = ctx.heads(HV)
        assert ctx.p_rule == "row_cost_lpt" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        dflt = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        assert dflt.p_rule == gp.P_RULE_DEFAULT == "small_one"
        dhp = dflt.heads(HV)
        assert (dhp.P, dhp.segments.NSEG) == (dflt_plans[r].P, dflt_plans[r].NSEG)
        if key not in SMALL_ONE_CHANGED:
            assert (dhp.P, dhp.segments.NSEG) == (hp.P, hp.segments.NSEG)
        ref = gp.cake_gdn_det_train_kcp_build_context(
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
    """``P_RULE_DEFAULT`` is the ``"small_one"`` row rule on top of ``"row_cost_lpt"`` (``"row_cost_lpt"`` and
    ``"row_cost"`` stay selectable); ``kcp_plan`` refuses both (they need every rank's table); the per-rank rules
    stay selectable through ``p_rule`` and then drive the head plan; the row rule picks the recorded vectors on three
    tables of record (two flip to ``"one"`` -- one far below any margin, one at a model ratio of 0.980 that only the
    ``"one"`` margin of 0 admits -- and one keeps the base far from any margin)."""
    assert gp.P_RULE_DEFAULT == "small_one" and "row_cost_lpt" in gp.P_RULES
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
    # contexts: the "row_cost_lpt" rule on every rank gives the row's vector, the default ("small_one": HV = 6, fragments
    # of <= 160 chunks) plans P = 1; the selectable row rule and a per-rank rule are honoured when asked for
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    part = gp.cake_gdn_det_train_kcp_partition(cu, 2, "contiguous")
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
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost_lpt"
        )
        assert ctx.p_rule == "row_cost_lpt" and want[r] == ctx.heads(6).P
        assert ctx.heads(6).segments.p_rule == "row_cost_lpt"
        dflt = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        assert dflt.p_rule == "small_one" == dflt.heads(6).segments.p_rule
        assert dflt.heads(6).P == 1
        old = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        assert old.p_rule == "row_cost" and want_old[r] == old.heads(6).P
        assert old.heads(6).segments.p_rule == "row_cost"
        occ = gp.cake_gdn_det_train_kcp_build_context(
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


def test_opt_in_row_rules_only_lower_the_shipped_P() -> None:
    """The row rules ``"fill_down"``, ``"small_one"`` (the default) and ``"fill_small"`` start from the
    ``"row_cost_lpt"`` vector and only lower a rank's ``P``: ``"fill_down"`` takes ``min(P, wave_fill P)`` on
    the ranks below the chain cap and keeps the others, ``"small_one"`` runs ``P = 1`` on every rank exactly when
    ``HV <= SMALL_ONE_HV_MAX`` and no fragment of any rank exceeds ``SMALL_ONE_NC_MAX`` chunks (else the shipped
    vector), ``"fill_small"`` is ``"small_one"`` on those rows and ``"fill_down"`` elsewhere; every plan reports its
    rule and ``NSEG = n_frag * P``; ``kcp_plan`` refuses them (they need every rank's table)."""
    rules = ("fill_down", "small_one", "fill_small")
    assert gp.P_RULE_DEFAULT == "small_one" and set(rules) <= set(gp.ROW_RULES)
    assert set(gp.ROW_RULES) <= set(gp.P_RULES)
    assert (gp.SMALL_ONE_HV_MAX, gp.SMALL_ONE_NC_MAX) == (12, 160)
    for rule in rules:
        with pytest.raises(ValueError, match=rule):
            gp.kcp_plan(2, 500, 48, SM_COUNT, p_rule=rule)
    tables = (
        TABLE_P1_CP4_D,
        TABLE_P1_CP2_C,
        TABLE_P3_CP2_C,
        (((130, True),) * 12,) * 8,  # (4,12)-like CP8 D: 12 short fragments per rank
        (((237, True), (237, True), (100, False)),) * 4,  # Z-like: 237-chunk fragments
        # above the chain cap: fill_down keeps the shipped P
        (((473, True), (473, False)),) * 4,
        # (2,6)-like CP16 Z: two short fragments per rank
        (((119, True), (119, True)),) * 8,
        ((), ((150, True), (20, False))),  # a zero-token rank
    )
    lowered = {"fill_down": 0, "small_one": 0}
    for sm in (SM_COUNT, 148):
        for table in tables:
            for HV in (6, 12, 24, 48):
                ship = gp.kcp_plan_row(table, HV, sm, p_rule="row_cost_lpt")
                for rule in rules:
                    plans = gp.kcp_plan_row(table, HV, sm, p_rule=rule)
                    assert plans == gp.kcp_plan_row(table, HV, sm, p_rule=rule)
                    assert len(plans) == len(ship)
                    for seg, base, t in zip(plans, ship, table, strict=True):
                        assert seg.p_rule == rule and 1 <= seg.P <= base.P
                        assert seg.n_frag * seg.P == seg.NSEG and seg.n_frag == len(t)
                fill = tuple(
                    min(
                        base.P,
                        gp.kcp_plan(
                            len(t),
                            max((nc for nc, _ in t), default=0),
                            HV,
                            sm,
                            p_rule="wave_fill",
                        ).P,
                    )
                    if 0 < max((nc for nc, _ in t), default=0) < gp.WAVE_MIN_NC_MAX_FRAG
                    else base.P
                    for base, t in zip(ship, table, strict=True)
                )
                assert fill == tuple(
                    seg.P for seg in gp.kcp_plan_row(table, HV, sm, p_rule="fill_down")
                )
                small = HV <= gp.SMALL_ONE_HV_MAX and all(
                    nc <= gp.SMALL_ONE_NC_MAX for t in table for nc, _ in t
                )
                got = tuple(
                    seg.P for seg in gp.kcp_plan_row(table, HV, sm, p_rule="small_one")
                )
                assert got == (
                    tuple(1 for _ in table) if small else tuple(seg.P for seg in ship)
                )
                both = tuple(
                    seg.P for seg in gp.kcp_plan_row(table, HV, sm, p_rule="fill_small")
                )
                assert both == (tuple(1 for _ in table) if small else fill)
                lowered["small_one"] += int(small and max(seg.P for seg in ship) > 1)
                lowered["fill_down"] += int(fill != tuple(seg.P for seg in ship))
    # each rule changes some shipped plan (else it would be a no-op)
    assert lowered["small_one"] > 0 and lowered["fill_down"] > 0, lowered


@pytest.mark.parametrize("rule", ["fill_down", "small_one", "fill_small"])
def test_opt_in_row_rules_give_every_rank_the_same_plan(rule) -> None:
    """Determinism of the row rules on top of ``"row_cost_lpt"``: the plan is a pure function of every rank's fragment table, ``HV`` and the
    SM count, so every rank's context derives its ``P`` / ``NSEG`` from the same row vector, a second build of the
    same topology is field-identical, and only the segment program differs from the ``"row_cost_lpt"`` context (partition,
    fragment table, local tables and crossing lists unchanged)."""
    cases = (
        # (2,6) C1-like: small_one -> P = 1
        ([0, 100, 740, 817, 1817, 1946], 2, "contiguous", 6),
        ([0, 3000], 4, "zigzag", 12),  # (4,12) C2-Z-like
        ([0, 3000], 4, "zigzag_per_document", 12),
        # (8,24) two-fragment ranks below the cap
        ([0, 30_000, 47_000], 4, "zigzag", 24),
        # (2,6) A3-like CP4 D: fragments of <= 160 chunks, small_one -> P = 1 on every rank
        ([0, 32_801, 65_570], 4, "zigzag_per_document", 6),
    )
    changed = 0
    for cu, W, layout, HV in cases:
        part = gp.cake_gdn_det_train_kcp_partition(
            torch.tensor(cu, dtype=torch.int64), W, layout
        )
        row = gp.kcp_plan_row(gp.row_tables(part), HV, SM_COUNT, p_rule=rule)
        ship = gp.kcp_plan_row(gp.row_tables(part), HV, SM_COUNT, p_rule="row_cost_lpt")
        changed += int(tuple(s.P for s in row) != tuple(s.P for s in ship))
        for r in range(W):
            ctx = gp.cake_gdn_det_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule=rule
            )
            again = gp.cake_gdn_det_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule=rule
            )
            hp = ctx.heads(HV)
            assert ctx.p_rule == rule == hp.segments.p_rule
            assert (hp.P, hp.segments.NSEG) == (row[r].P, row[r].NSEG)
            assert _sha(_context_record(ctx, HV)) == _sha(_context_record(again, HV))
            ref = gp.cake_gdn_det_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost_lpt"
            )
            assert ref.p_rule == "row_cost_lpt"
            assert ctx.partition.to_dict() == ref.partition.to_dict() == part.to_dict()
            assert ctx.frag_table.tolist() == ref.frag_table.tolist()
            assert ctx.cu_seqlens_local_cpu == ref.cu_seqlens_local_cpu
            assert (ctx.in_slot, ctx.out_slot, ctx.chains) == (
                ref.in_slot,
                ref.out_slot,
                ref.chains,
            )
            assert (hp.P == ref.heads(HV).P) == (row[r].P == ship[r].P)
    assert changed > 0  # the cases exercise a changed plan


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_opt_in_row_rules_keep_the_rows_of_record(path) -> None:
    """The row rules on top of ``"row_cost_lpt"`` return its segment program on every step-0 row (GB300 and B200 SM
    counts), except on the (row, rule) pairs of ``OPT_IN_CHANGED``, which get their recorded vector at both SM counts:
    the CP8 / CP16 ranks below the chain cap whose one-wave fill is lower (``"fill_down"`` / ``"fill_small"``) and the
    ``HV <= SMALL_ONE_HV_MAX`` row of ``SMALL_ONE_CHANGED`` (``"small_one"`` / ``"fill_small"``)."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    part = gp.cake_gdn_det_train_kcp_partition(**_partition_inputs(row))
    tables = gp.row_tables(part)
    key = _row_key(row)
    changed = OPT_IN_CHANGED.get(key, {})
    for sm in (SM_COUNT, 148):
        ship = gp.kcp_plan_row(tables, HV, sm, p_rule="row_cost_lpt")
        for rule in ("fill_down", "small_one", "fill_small"):
            got = gp.kcp_plan_row(tables, HV, sm, p_rule=rule)
            want = changed.get(rule)
            if want is not None:
                # a recorded departure: its vector, a P other than "row_cost_lpt", NSEG = n_frag * P
                p_got = tuple(s.P for s in got)
                assert p_got == want != tuple(s.P for s in ship), (key, sm, rule)
                assert all(s.n_frag * s.P == s.NSEG for s in got)
                continue
            assert [(s.P, s.NSEG) for s in got] == [(s.P, s.NSEG) for s in ship], (
                row["row"],
                row["cp"],
                row["layout"],
                sm,
                rule,
            )


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
        if (save_transfer or save_mode == "full") and hp.P > 1:
            total += hp.P * ctx.n_frag * s
    if (
        save_mode == "full" and boundary_mode == "saved"
    ):  # every forward tape: h, v_new / w / u, Aqk / Akk, gamma, gcum / rowscale / qscale (the replay boundary keeps
        # none: its backward is the light replay)
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
    part = gp.cake_gdn_det_train_kcp_partition(cu, W, "contiguous")
    H_qk, HV = 2, 6
    s = HV * gp.STATE_TILE_BYTES
    records = []
    b1_sums = {"design": 0, "buffers": 0}
    for r in range(W):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        hp = ctx.heads(HV)
        cb = gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        assert (
            cb["C"] == C_CONV
            and gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=16, HV=48)["C"]
            == (2 * 16 + 48) * 128
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
        maps = gb.cake_gdn_det_train_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False
        )
        s_owner = hp.Hg_r * gp.STATE_TILE_BYTES
        # "b1_design": b1_send = s n_in + s n_io [not save_transfer] (n_io = fragments crossing in AND
        # out: the maps the owner's adjoint chain reads; a document's last fragment sends none); owner
        # b1_recv in d / m planes
        n_io = sum(1 for f in part.fragments[r] if f.crossing_in and f.crossing_out)
        N_io = sum(
            1 for frs in part.fragments for f in frs if f.crossing_in and f.crossing_out
        )
        assert hp.n_io_by_rank[r] == n_io and sum(hp.n_io_by_rank) == N_io
        assert maps["b1_design"]["send"] == s * (ctx.n_in + n_io)
        assert maps["b1_design"]["recv"] == (ctx.N_in + N_io) * s_owner
        assert (
            maps["b1_d"] == cb["b1"] == cb["b1_replay"] == cb["b1_design"] == cb["b1_d"]
        )
        assert maps["b1_m"] == {"send": s * n_io, "recv": N_io * s_owner}
        assert cb["b1_m"] == {"send": 0, "recv": 0}
        assert maps["B3"] == cb["B3"]  # the replay never re-sends maps
        # "b1" = the plan's split tables = the design form (both count any padding; none at (6, W)): the d_f
        # plane of the crossing-in slots + the map plane of the fragments crossing in and out
        assert maps["b1"] == maps["b1_design"]
        assert maps["b1"] == {
            "send": s * (ctx.n_in + n_io),
            "recv": (ctx.N_in + N_io) * s_owner,
        }
        # every tile of the b1 send buffer with maps is written: the d_f tiles and the M_f^T tiles cover
        # it exactly once (no never-written element reaches the collective)
        dm_tiles = [
            t
            for rows in (hp.b1_d_tile_dm, hp.b1_m_tile_dm)
            for row in rows
            for t in row
        ]
        assert sorted(dm_tiles) == list(range(sum(hp.splits["b1_send_dm"]) // gp.TILE))
        assert len(hp.b1_m_tile_dm) == n_io and len(hp.b1_d_tile_dm) == ctx.n_in
        assert cb["padding"]["b1"] == {
            "send": hp.pad_numel["b1_send_d"] * 4,
            "recv": hp.pad_numel["b1_recv_d"] * 4,
        }
        b1_sums["design"] += maps["b1_design"]["send"]
        b1_sums["buffers"] += maps["b1"]["send"]
        replay = gb.cake_gdn_det_train_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False, boundary_mode="replay"
        )
        assert replay["b1"] == cb["b1_replay"]
        # the exchange terms are HV-only: H_qk = 1, 2, 3, 6 give the same c1 / c2 / b1 / b2
        for hq in (1, 3, 6):
            other = gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=hq, HV=HV, C=C_CONV)
            assert all(other[k] == cb[k] for k in ("c1", "c2", "b1", "b2"))
        planes = 0  # kept-form rows that hand the M_seg^T plane over
        for save_mode in gb.SAVE_MODES:
            for boundary_mode in gb.BOUNDARY_MODES:
                for save_transfer in (True, False):
                    got = gb.cake_gdn_det_train_kcp_saved_bytes(
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
                    layout = gb.cake_gdn_det_train_kcp_saved_layout(
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
                    ws = gb.cake_gdn_det_train_kcp_workspace_layout(
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
                        == gb.cake_gdn_det_train_kcp_workspace_bytes(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode=save_mode,
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                        )
                    )
                    # the forward is the larger of its two moments (#1, the tape walk), each below the sum of what
                    # the forward binds
                    fwd_phases = {
                        k: v for k, v in ws.items() if k.startswith("fwd.phase.")
                    }
                    fwd_parts = sum(
                        v
                        for k, v in ws.items()
                        if k.startswith("fwd.") and not k.startswith("fwd.phase.")
                    )
                    assert set(fwd_phases) == {"fwd.phase.prep", "fwd.phase.walk"}
                    assert ws["forward"] == max(fwd_phases.values())
                    assert all(v <= fwd_parts for v in fwd_phases.values())
                    # the backward is the largest of its lifetime phases, each below the sum of what the arena
                    # can hold; the kept form lists the hand-off tensors apart (its transient memory: below)
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
                    tapes = ctx.NC_pad * HV * (32768 + 4) + ctx.R * HV * (
                        3 * 256 + 2 * 128 + 3 * 4
                    )
                    # the kept-tape form exists for the light form of a saved boundary only (no hand-off under a
                    # replay boundary); at P > 1 without save_transfer it also hands the M_seg^T plane over (the
                    # plan's P and the override P = 2, where every rank has the plane)
                    light_saved = save_mode == "light" and boundary_mode == "saved"
                    for P_q in (None, 2):
                        kept_ws, replay_ws = (
                            gb.cake_gdn_det_train_kcp_workspace_layout(
                                ctx,
                                H_qk=H_qk,
                                HV=HV,
                                save_mode=save_mode,
                                boundary_mode=boundary_mode,
                                save_transfer=save_transfer,
                                C=C_CONV,
                                P=P_q,
                                reuse_recompute_tapes=form,
                            )
                            for form in (True, False)
                        )
                        P_eff = hp.P if P_q is None else P_q
                        stage_plane = (
                            ctx.n_frag * P_eff * s
                            if (light_saved and P_eff > 1 and not save_transfer)
                            else 0
                        )
                        planes += stage_plane > 0
                        assert kept_ws["bwd.kept_tapes"] == (
                            tapes + stage_plane if light_saved else 0
                        )
                        if not light_saved:
                            assert kept_ws == replay_ws
                        assert replay_ws["bwd.kept_tapes"] == 0
                        assert replay_ws["backward_transient"] == replay_ws["backward"]
                        # both forms need the same transient bytes besides the saved set, but the kept form holds
                        # the handed-over plane through the adjoint tape walk and #8 (the replay drops its staging
                        # before)
                        assert kept_ws["backward_transient"] == (
                            max(
                                replay_ws["bwd.phase.exchange"],
                                max(
                                    replay_ws["bwd.phase.walk"],
                                    replay_ws["bwd.phase.end"],
                                )
                                + stage_plane,
                            )
                            if light_saved
                            else replay_ws["backward"]
                        )
                    assert (
                        ws["bwd.dv_new"] == 0 and ws["bwd.dv_intra"] == ctx.R * HV * 256
                    )  # dv_new is stored in place of dv_intra
                    assert (
                        ws["fwd.c1_send"] == cb["c1"]["send"]
                        and ws["bwd.b2_send"] == cb["b2"]["send"]
                    )
                    if save_mode == "light" and boundary_mode == "saved":
                        # the shipped forward is fused: no h tape, #1's akk still bound at its peak; the split
                        # form's tape walk binds h / v_new once akk / gcum and (with an exchange) the c1 send pool
                        # are gone
                        c1_gone = (
                            ws["fwd.c1_send"] + ws["fwd.c1_send_mf"] if ctx.N_out else 0
                        )
                        assert (
                            "fwd.h_tape" not in ws
                            and ws["fwd.akk"] == ctx.R * HV * 128
                            and ws["bwd.h_tape"] == ctx.NC_pad * HV * 32768
                            and ws["forward"] == ws["fwd.phase.prep"]
                            and ws["forward_split"]
                            == max(
                                ws["forward"],
                                ws["forward"]
                                + ws["bwd.h_tape"]
                                + ws["bwd.v_new"]
                                - ws["fwd.akk"]
                                - ws["fwd.gcum"]
                                - c1_gone,
                            )
                        )
                    elif save_mode == "light" or boundary_mode == "replay":
                        # the replay boundary keeps no tape: "full" there is the light replay's arena
                        assert (
                            ws["fwd.h_tape"] == ctx.NC_pad * HV * 32768
                            and ws["bwd.h_tape"] == ws["fwd.h_tape"]
                            and ws["forward_split"] == ws["forward"]
                            and layout["h_tape"] == 0
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
        assert planes > 0  # the P = 2 kept-form rows without save_transfer
        light, full = (
            gb.cake_gdn_det_train_kcp_saved_bytes(
                ctx, H_qk=H_qk, HV=HV, save_mode=m, C=C_CONV
            )
            for m in ("light", "full")
        )
        assert full - light == ctx.NC_pad * HV * (32768 + 4) + ctx.R * HV * (
            3 * 256 + 2 * 128 + 3 * 4
        )  # h + gamma, v_new / w / u, Aqk / Akk, gcum / rowscale / qscale
        host = gb.cake_gdn_det_train_kcp_workspace_layout(
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
                "workspace_light": gb.cake_gdn_det_train_kcp_workspace_bytes(
                    ctx, H_qk=H_qk, HV=HV, C=C_CONV
                ),
            }
        )
    with pytest.raises(ValueError, match="multiple"):
        gb.cake_gdn_det_train_kcp_comm_bytes(ctx, H_qk=4, HV=6)
    with pytest.raises(ValueError, match="save_mode"):
        gb.cake_gdn_det_train_kcp_saved_bytes(ctx, H_qk=2, HV=6, save_mode="tape")
    with pytest.raises(ValueError, match="boundary_mode"):
        gb.cake_gdn_det_train_kcp_saved_bytes(ctx, H_qk=2, HV=6, boundary_mode="none")
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


# Three ranks, two documents.  Rank 1 holds the last fragment of document 0 (crossing in only: in_slot 0) before a
# middle fragment of document 1 (crossing in and out: in_slot 1, io_slot 0), so there the index of the b1 map plane
# (io_slot) differs from the crossing-in index (in_slot).  Contiguous layouts never give such a rank.
B1_MAP_CU = (0, 200, 500)
B1_MAP_FRAGMENTS = (
    ((0, 0, 100), (1, 0, 100)),
    ((0, 100, 100), (1, 100, 100)),
    ((1, 200, 100),),
)


@pytest.mark.parametrize("HV", (4, 5))
def test_b1_map_plane_index_on_a_middle_fragment(HV) -> None:
    """The b1 buffer with maps (``save_transfer=False``) is indexed by ``io_slot`` in its ``M_f^T`` plane, not by
    ``in_slot``.  Pinned on every rank of :data:`B1_MAP_FRAGMENTS`: ``b1_m_index``, the send / receive split sizes,
    the ``d_f`` and ``M_f^T`` tiles (every send tile written once; only the padding tiles of the padded owner stay
    unwritten), the owner offsets ``adj_m_off_b1`` (where rank 1's map lands after the all-to-all) and the byte
    query (``b1 == b1_design``, padding included).  ``HV = 4``: ``Hg = 2``, owners ``(2, 2, 0)``; ``HV = 5``: owners
    ``(2, 2, 1)``, owner 2 padded by one tile per slot."""
    W = 3
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(B1_MAP_CU, dtype=torch.int64), W, fragments=B1_MAP_FRAGMENTS
    )
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(W)
    ]
    hps = [ctx.heads(HV) for ctx in ctxs]
    assert tuple(ctxs[1].in_slot) == (0, 1)
    assert [tuple(hp.n_io_by_rank) for hp in hps] == [(0, 1, 0)] * W
    # b1_m_index per segment: the io_slot at segment 0 of a fragment crossing in and out, else -1
    per_slot = ((-1, -1), (-1, 0), (-1,))
    for r, hp in enumerate(hps):
        want = [v if p == 0 else -1 for v in per_slot[r] for p in range(hp.P)]
        assert list(hp.b1_m_index) == want, r
    # split sizes in tiles: send [dst owner j] = (n_in + n_io) Hg_pad_j, receive [src i] = (n_in_i + n_io_i) Hg
    send = {
        4: ((0, 0, 0), (6, 6, 0), (2, 2, 0)),
        5: ((0, 0, 0), (6, 6, 6), (2, 2, 2)),
    }[HV]
    recv = {4: ((0, 6, 2), (0, 6, 2), (0, 0, 0)), 5: ((0, 6, 2),) * W}[HV]
    for r, hp in enumerate(hps):
        b1_send, b1_recv = hp.b1_splits(True)
        assert all(x % gp.TILE == 0 for x in (*b1_send, *b1_recv))
        assert tuple(x // gp.TILE for x in b1_send) == send[r], r
        assert tuple(x // gp.TILE for x in b1_recv) == recv[r], r
    assert all(send[i][j] == recv[j][i] for i in range(W) for j in range(W))
    # send tiles: the d_f plane by in_slot, then the M_f^T plane by io_slot, per owner block
    d_tiles = {
        4: ((), ((0, 1, 6, 7), (2, 3, 8, 9)), ((0, 1, 2, 3),)),
        5: ((), ((0, 1, 6, 7, 12), (2, 3, 8, 9, 14)), ((0, 1, 2, 3, 4),)),
    }[HV]
    m_tiles = {
        4: ((), ((4, 5, 10, 11),), ()),
        5: ((), ((4, 5, 10, 11, 16),), ()),
    }[HV]
    never_written = {4: ((), (), ()), 5: ((), (13, 15, 17), (5,))}[HV]
    for r, hp in enumerate(hps):
        assert hp.b1_d_tile_dm == d_tiles[r] and hp.b1_m_tile_dm == m_tiles[r], r
        written = [
            t
            for rows in (hp.b1_d_tile_dm, hp.b1_m_tile_dm)
            for row in rows
            for t in row
        ]
        assert len(written) == len(set(written)), r  # no send tile written twice
        n_tiles = sum(hp.b1_splits(True)[0]) // gp.TILE
        assert sorted((*written, *never_written[r])) == list(range(n_tiles)), r
    # owner side: the adjoint chains (doc 0: rank 1 slot 0; doc 1: rank 2 slot 0, rank 1 slot 1) read the map of
    # rank 1's middle fragment behind rank 1's n_in = 2 d_f slots
    for o, hp in enumerate(hps):
        if hp.Hg_r == 0:
            assert list(hp.adj_m_off_b1) == []
            continue
        assert list(hp.adj_m_off_b1) == [-1, -1, 4 * gp.TILE], o
        sent = hps[1].b1_m_tile_dm[0][o * hp.Hg] * gp.TILE - sum(
            hps[1].b1_splits(True)[0][:o]
        )
        assert hp.adj_m_off_b1[2] - sum(hp.b1_splits(True)[1][:1]) == sent, o
    for r, (ctx, hp) in enumerate(zip(ctxs, hps, strict=True)):
        maps = gb.cake_gdn_det_train_kcp_comm_bytes(
            ctx, H_qk=1, HV=HV, C=C_CONV, save_transfer=False
        )
        b1_send, b1_recv = hp.b1_splits(True)
        assert (
            maps["b1"]
            == maps["b1_design"]
            == {"send": sum(b1_send) * 4, "recv": sum(b1_recv) * 4}
        ), r
        pad = maps["padding"]["b1"]  # inside b1_design, not on top of it
        n_io, N_io = hp.n_io_by_rank[r], sum(hp.n_io_by_rank)
        s_owner = hp.Hg_r * gp.STATE_TILE_BYTES
        assert maps["b1_design"] == {
            "send": (ctx.n_in + n_io) * HV * gp.STATE_TILE_BYTES + pad["send"],
            "recv": (ctx.N_in + N_io) * s_owner + pad["recv"],
        }, r


# --------------------------------------------------------------------------- registry seed and package hygiene


def test_registry_seed() -> None:
    """The ``cake_jit`` seed: the shipped records of the nine programs and the output walk (one library per record and exported arch
    in ``build.json``; + the two test-only transfer walks of ``build.test.json``), the knob vocabulary,
    every record's logical names in the vocabulary, the grid / selection rules, literal
    tables (``ast.literal_eval`` of the module source equals the runtime dicts; frozen 20-hex library names; no
    build-system token) and the manifest skeleton."""
    # every program has a shipped record, except the in-kernel L2 norm programs until the exporter renders them
    # (all three or none: their host plan is L2N_RECORD_PLANS)
    shipped = {rec["program"] for rec in gj.MODULES.values()}
    assert shipped <= set(gj.PROGRAMS)
    assert set(gj.PROGRAMS) - shipped in (set(), set(gj.L2N_PROGRAMS))
    assert set(gj.L2N_RECORD_PLANS) == set(gj.L2N_PROGRAMS)
    assert gj.l2n_records_shipped() == (set(gj.L2N_PROGRAMS) <= shipped)
    assert sum(len(gj.records_of_program(p)) for p in gj.PROGRAMS) == len(gj.MODULES)
    assert all(
        len(gj.records_of_program(p)) == 1
        for p in gj.PROGRAMS
        if (p in shipped)
        and p
        not in (
            "kcp_fwdh_offset",
            "kcp_fwdh_tape",
            "kcp_fwdh_out",
            "kcp_dhu_offset",
            "kcp_dhu_tape",
            "kcp_affine_merge",
            "fwd_prep_l2n",
        )
    )  # the vh=64 forms of the state walks #2 (L8A, the output walk too) / #7, the c8 / c16 merges and the forward
    # form of #1_l2n (fwd_prep_l2n_fwd) are the only multi-record programs
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
        suffix = "_kcp" if rec["program"].startswith("kcp_") else ""
        assert re.fullmatch(
            rf"cake_gdn_det_train_[0-9a-f]{{20}}{suffix}", rec["library"]
        ), name
        assert rec["kernel_symbol"] == f"kernel_{rec['library']}"
        assert rec["arg_plan"][-3:] == [
            ["grid", "grid_x"],
            ["grid", "grid_y"],
            ["grid", "grid_z"],
        ]
        assert rec["select_rule"]["kind"] in (
            "always",
            "columns",
            "ctas_per_head",
            "prep_form",
        ), name
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
    assert manifest["package"] == "flashinfer.experimental.cake.cake_gdn_det_train_kcp"
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


_DYNAMIC_IMPORTS = ("import_module", "__import__")


def _imported_names(node: ast.AST) -> list[str]:
    """The module names one AST node imports: an ``import`` / ``from ... import`` statement, or a call of
    ``importlib.import_module`` / ``import_module`` / ``__import__`` with a literal string module name."""
    if isinstance(node, ast.Import):
        return [a.name for a in node.names]
    if isinstance(node, ast.ImportFrom):
        return [a.name for a in node.names] + ([node.module] if node.module else [])
    if isinstance(node, ast.Call):
        func = node.func
        name = (
            func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        )
        if name in _DYNAMIC_IMPORTS:
            args = list(node.args[:1]) + [
                k.value for k in node.keywords if k.arg == "name"
            ]
            return [
                a.value
                for a in args
                if isinstance(a, ast.Constant) and isinstance(a.value, str)
            ]
    return []


def _kda_imports(source: str) -> list[int]:
    """The lines of ``source`` that import a KDA package (any ``cake_kda*`` module), statically or dynamically."""
    return [
        node.lineno
        for node in ast.walk(ast.parse(source))
        if any("cake_kda" in n for n in _imported_names(node))
    ]


def test_the_kda_import_scan_sees_dynamic_imports() -> None:
    """The scan of :func:`test_package_imports_no_kda_and_no_environ` rejects a literal dynamic import of a KDA package
    (``importlib.import_module``, ``import_module``, ``__import__``) as well as an import statement."""
    kda = "flashinfer.experimental.cake_kda_det_train_kcp.cake_kcp_plan"
    for line in (
        f"import importlib\nm = importlib.import_module({kda!r})\n",
        f"from importlib import import_module\nm = import_module(name={kda!r})\n",
        f"m = __import__({kda!r})\n",
        f"import {kda}\n",
        "from flashinfer.experimental import cake_kda_det_train\n",
    ):
        assert _kda_imports(line), line
    for line in (
        "import importlib\nm = importlib.import_module('flashinfer.cake_gdn_det_train_kcp')\n",
        "m = __import__(name)\n",
    ):
        assert not _kda_imports(line), line


def test_package_imports_no_kda_and_no_environ() -> None:
    """The package never imports a KDA package (any ``cake_kda*`` module, by statement or by a literal
    ``importlib.import_module`` / ``__import__`` call) and reads no environment variable; the plan module imports
    torch, torch.distributed and dataclasses only."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        source = path.read_text()
        assert not _kda_imports(source), f"{path.name} imports a KDA package"
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, layout
    )
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
        gp.cake_gdn_det_train_kcp_partition(
            torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
        ),
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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, "zigzag"
    )
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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
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
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.cake_gdn_det_train_kcp_backward(
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


@pytest.mark.parametrize("fused_o", (0, 1))
def test_fwdh_offset_walk_defers_its_tapes_to_the_first_tape_walk(fused_o: int) -> None:
    """The #2 offset walk hands the kernel ``u`` as its never-dereferenced ``h_tape`` / ``v_new`` pointers
    (``FWDH_OFFSET_TAPE_STANDIN``), so the arena allocates ``h_tape`` / ``v_new`` at the first tape walk and never at an
    offset walk.  CP = 1 with two segments per fragment (P = 2) runs the offset walk over every segment before the
    tape walk, in the forward (the summaries) and in the light backward's replay; the fake backend on the CPU records
    the arena's tapes and the pointers at every #2 launch.  The stand-in has the dtype the offset binding checks for
    both pointers, and #1 writes it before any walk.  The fused forward (``fused_o`` = 1, the shipped form) runs the
    output walk in place of the tape walk: it takes neither pointer, so its forward never allocates the two tapes;
    the backward's replay is the split form either way."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    seen: list[tuple[str, bool, int, bool, bool]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            if name == "kcp_fwdh_out":
                call = args["_call"]
                assert "h_tape" not in args and "v_new" not in args
                bound = sum(n in call.ws._tapes for n in ("h_tape", "v_new"))
                seen.append((name, call.backward, bound, False, False))
            elif name in kcp.FWDH_PROGRAMS:
                call = args["_call"]
                standin = call.tape(kcp.FWDH_OFFSET_TAPE_STANDIN).data_ptr()
                ptrs = (args["h_tape"].data_ptr(), args["v_new"].data_ptr())
                bound = sum(n in call.ws._tapes for n in ("h_tape", "v_new"))
                real = name == "kcp_fwdh_tape" and ptrs == (
                    call.tape("h_tape").data_ptr(),
                    call.tape("v_new").data_ptr(),
                )
                seen.append(
                    (name, call.backward, bound, ptrs == (standin, standin), real)
                )
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
            test_knobs={"fused_o": fused_o},
        )
        grads = kcp.cake_gdn_det_train_kcp_backward(
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
            test_knobs={"fused_o": fused_o},
        )
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    assert out.shape[0] == cu[-1] and all(
        torch.isfinite(g.float()).all() for g in (out, *grads)
    )
    fused = bool(fused_o) and kcp.fused_o_shipped()
    forward = [s for s in seen if not s[1]]
    assert ("kcp_fwdh_out" in [s[0] for s in forward]) == fused
    assert all(s[0] != "kcp_fwdh_out" for s in seen if s[1])  # the replay is split
    if fused:
        assert "kcp_fwdh_tape" not in [s[0] for s in forward]
        assert any(s[0] == "kcp_fwdh_offset" for s in forward)
        assert all(s[3] for s in forward if s[0] == "kcp_fwdh_offset")
        assert all(s[2] == 0 for s in forward), (
            "a fused forward bound h_tape / v_new",
            forward,
        )
    for backward in (True,) if fused else (False, True):
        phase = [s for s in seen if s[1] == backward]
        names = [s[0] for s in phase]
        assert "kcp_fwdh_offset" in names and "kcp_fwdh_tape" in names, (
            backward,
            names,
        )
        first_tape = names.index("kcp_fwdh_tape")
        offsets = [s for s in phase if s[0] == "kcp_fwdh_offset"]
        assert names.index("kcp_fwdh_offset") < first_tape, (
            "the offset walk must run before the first tape walk in this form",
            backward,
        )
        assert all(s[3] for s in offsets), (
            "the offset walk's h_tape / v_new pointers are not the stand-in tape",
            backward,
        )
        assert all(s[2] == 0 for s in phase[:first_tape]), (
            "h_tape / v_new sat in the arena at an offset walk",
            backward,
        )
        assert phase[first_tape][2] == 2 and phase[first_tape][4], (
            "the first tape walk must allocate and receive the real h_tape / v_new",
            backward,
        )
        assert all(s[4] and not s[3] for s in phase if s[0] == "kcp_fwdh_tape"), (
            backward
        )
    standin = kcp.FWDH_OFFSET_TAPE_STANDIN
    assert standin in kcp.FWD_TAPES and standin not in kcp.FWD_DEAD_AFTER_PREP
    dims = dict(R=ctx.R, NC_pad=ctx.NC_pad, HV=HV, T=ctx.T_r)
    assert (
        kcp.tape_spec(standin, **dims)[1]
        == kcp.tape_spec("h_tape", **dims)[1]
        == kcp.tape_spec("v_new", **dims)[1]
        == torch.bfloat16
    )


def test_persistent_programs_receive_the_workspace_ticket_slots() -> None:
    """#5 / #6 / #8 receive the caller workspace's ticket slot of their site as the logical ``ticket``: a 128-byte
    aligned u32 slot that is zero at every launch (the kernels leave it so; the emulation never touches it), the
    same slot across the forward and the backward of a caller-held workspace and across repeated backwards.  The
    #7 walks launch the record the registry provides for the form rule (the ``_vh64`` record where it ships, else
    the full-head record), and the test knob ``dhu_vh`` forces the full-head form.  The forward is the split form
    (``fused_o`` = 0): the fused forward launches no #5 (its output walk takes no ticket)."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU, sm_count=SM_COUNT)
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.CakeGDNDetKcpWorkspace()
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
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
            workspace=ws,
            test_knobs={"fused_o": 0},
        )
        bwd = dict(saved=saved, backend="fake", segments=2, workspace=ws)
        grads = kcp.cake_gdn_det_train_kcp_backward(
            ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx, ops["dout"], **bwd
        )
        n_rule = len(walks)
        grads2 = kcp.cake_gdn_det_train_kcp_backward(
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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    assert set(kcp.RING_POOLS) == {"ring_out", "ring_in"}
    assert set(kcp.SAVE_TRANSFER_FRESH_POOLS) == {"c1_recv", "stage_m"}
    assert set(kcp.RING_POOLS) <= set(kcp.FWD_DEAD_POOLS) and set(
        kcp.RING_POOLS
    ) <= set(kcp.BWD_DEAD_BEFORE_TAPES)
    dead = set(kcp.RING_POOLS) | set(kcp.SAVE_TRANSFER_FRESH_POOLS)
    # the full workspace of the public constructor sizes every pool (the caller-held form)
    full = kcp.CakeGDNDetKcpWorkspace(ctx, H_qk, HV, device="cpu", backward=True)
    assert dead <= set(full._pools)
    # a2a + save_transfer (shipped defaults): the dead pools are absent after the forward and after the backward
    ws = kcp.CakeGDNDetKcpWorkspace()
    out, saved = kcp.cake_gdn_det_train_kcp_forward(
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
    grads = kcp.cake_gdn_det_train_kcp_backward(
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
    kcp.cake_gdn_det_train_kcp_forward(
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
    ws2 = kcp.CakeGDNDetKcpWorkspace()
    kcp.cake_gdn_det_train_kcp_backward(
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
            # every pool the call holds at any of its launches (an eager call sizes its pools after its first launch)
            if not seen_pools or seen_pools[-1] is None:
                seen_pools.append(set())
            seen_pools[-1] |= set(args["_call"].ws._pools)
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        kcp.cake_gdn_det_train_kcp_forward(
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
        kcp.cake_gdn_det_train_kcp_forward(
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
    ws4 = kcp.CakeGDNDetKcpWorkspace()
    tb = ws4.ensure(ctx, H_qk, HV, device="cpu", backward=True, skip=kcp.RING_POOLS)
    assert not (set(kcp.RING_POOLS) & set(ws4._pools)) and "c1_send" in ws4._pools
    assert set(tb.pool_sizes(True)) >= set(
        kcp.RING_POOLS
    )  # the size table is unchanged
    with pytest.raises(KeyError, match="not sized"):
        ws4.pool("ring_out", 16)
    ws4.ensure(ctx, H_qk, HV, device="cpu", backward=True)
    assert set(kcp.RING_POOLS) <= set(ws4._pools)


def _other_mode_twin(cls, name: str):
    """A stand-in for the other mode's class of the same shape: every attribute and method of ``cls`` (dataclass
    fields included) on a class that is not ``cls`` -- what the non-deterministic package's ``CakeGDNNondetKcpSaved`` /
    ``CakeGDNNondetKcpWorkspace`` look like to this package (same fields and version), built without importing that package."""
    body = {k: v for k, v in vars(cls).items() if k not in ("__dict__", "__weakref__")}
    body.update(__module__="other_mode_stand_in", __qualname__=name)
    return type(name, (), body)


def test_backward_refuses_a_saved_set_or_workspace_of_the_other_mode() -> None:
    """The package backward reads the deterministic forward's ``CakeGDNDetKcpSaved`` and a caller-held ``CakeGDNDetKcpWorkspace``
    only.  A saved set of the other mode with every field of this forward's saved set (same version, same context,
    same problem: it passes every field check of ``_check_saved``) and a workspace of the other mode are refused with
    ``ValueError`` naming the deterministic mode and the expected type, as is any other object; the real saved set
    and workspace still run."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    args = [ops[n] for n in ("q", "k", "v", "g", "beta")]
    ws = kcp.CakeGDNDetKcpWorkspace()
    _, saved = kcp.cake_gdn_det_train_kcp_forward(
        *args, ctx, backend="fake", segments=2, workspace=ws
    )
    assert type(saved) is kcp.CakeGDNDetKcpSaved
    saved_twin = _other_mode_twin(kcp.CakeGDNDetKcpSaved, "CakeGDNNondetKcpSaved")(
        **{f.name: getattr(saved, f.name) for f in dataclasses.fields(saved) if f.init}
    )
    assert dataclasses.fields(saved_twin) == dataclasses.fields(saved)
    assert saved_twin.version == saved.version and saved_twin.ctx is ctx
    ws_twin = object.__new__(
        _other_mode_twin(kcp.CakeGDNDetKcpWorkspace, "CakeGDNNondetKcpWorkspace")
    )
    vars(ws_twin).update(vars(kcp.CakeGDNDetKcpWorkspace()))
    for name, kw, expected in (
        ("saved", {"saved": saved_twin}, "CakeGDNDetKcpSaved"),
        ("saved", {"saved": object(), "workspace": ws}, "CakeGDNDetKcpSaved"),
        ("workspace", {"saved": saved, "workspace": ws_twin}, "CakeGDNDetKcpWorkspace"),
        ("workspace", {"workspace": object()}, "CakeGDNDetKcpWorkspace"),
    ):
        with pytest.raises(
            ValueError,
            match=rf"cake_gdn_det_train_kcp_backward: {name} must be a {expected} of the deterministic mode",
        ):
            kcp.cake_gdn_det_train_kcp_backward(
                *args, ctx, ops["dout"], backend="fake", segments=2, **kw
            )
    grads = kcp.cake_gdn_det_train_kcp_backward(
        *args, ctx, ops["dout"], saved=saved, backend="fake", segments=2, workspace=ws
    )
    assert all(torch.isfinite(g.float()).all() for g in grads)


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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.CakeGDNDetKcpWorkspace()
    at_summaries: list[set[str]] = []
    summaries = kcp._summaries

    def recording_summaries(call):
        at_summaries.append({"S_in", "c1_recv"} & set(call.bufs))
        summaries(call)

    monkeypatch.setattr(kcp, "_summaries", recording_summaries)
    out, saved = kcp.cake_gdn_det_train_kcp_forward(
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
    full = kcp.CakeGDNDetKcpWorkspace(ctx, H_qk, HV, device="cpu", backward=False)
    assert "c2_send" in full._pools
    out2, saved2 = kcp.cake_gdn_det_train_kcp_forward(
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
    """A caller-owned saved set is never mutated by ``cake_gdn_det_train_kcp_backward`` (two backwards from it are bitwise equal and
    its tensors stay); a set marked ``release_in_backward`` (the autograd Function's own) comes back with ``S_in``,
    ``c1_recv`` and ``M_seg`` forgotten, with the same gradients bitwise; the per-call buffers of a replay backward
    (``saved=None``) are dropped the same way."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    _, saved = kcp.cake_gdn_det_train_kcp_forward(*args, backend="fake", segments=2)
    assert saved is not None and saved.c1_recv is not None and saved.M_seg is not None
    assert saved.release_in_backward is False
    g1 = kcp.cake_gdn_det_train_kcp_backward(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    assert (
        saved.S_in is not None and saved.c1_recv is not None and saved.M_seg is not None
    )
    g2 = kcp.cake_gdn_det_train_kcp_backward(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    assert all(torch.equal(a, b) for a, b in zip(g1, g2, strict=True))
    owned = dataclasses.replace(saved, release_in_backward=True)
    g3 = kcp.cake_gdn_det_train_kcp_backward(
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
    g4 = kcp.cake_gdn_det_train_kcp_backward(
        *args,
        ops["dout"],
        saved=None,
        backend="fake",
        segments=2,
        boundary_mode="replay",
    )
    assert all(torch.isfinite(g.float()).all() for g in g4)
    assert "S_in" not in kcp.LAST_CALL["bwd"]["fresh_bytes"]


@pytest.mark.parametrize(
    "save_mode,save_transfer",
    [("light", True), ("full", True), ("full", False)],
    ids=["light-st", "full-st", "full-nost"],
)
@pytest.mark.parametrize("P", (2, 3))
def test_saved_byte_query_counts_the_whole_m_seg_plane(
    save_mode, save_transfer, P
) -> None:
    """The saved set keeps ``M_seg`` as the whole ``M_seg^T`` staging plane: ``[NSEG, HV]`` FP32 states with
    ``NSEG = n_frag * P``, under ``save_transfer`` and under ``save_mode="full"``.  The byte query's ``M_seg`` term
    (``P n_frag s``) equals the bytes of that saved tensor."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    _, saved = kcp.cake_gdn_det_train_kcp_forward(
        ops["q"],
        ops["k"],
        ops["v"],
        ops["g"],
        ops["beta"],
        ctx,
        backend="fake",
        segments=P,
        save_mode=save_mode,
        save_transfer=save_transfer,
    )
    assert saved is not None and saved.P == P and saved.M_seg is not None
    plane = saved.M_seg.numel() * saved.M_seg.element_size()
    assert plane == P * ctx.n_frag * HV * gp.STATE_TILE_BYTES
    q = {
        "H_qk": H_qk,
        "HV": HV,
        "save_mode": save_mode,
        "save_transfer": save_transfer,
        "P": P,
    }
    assert gb.cake_gdn_det_train_kcp_saved_layout(ctx, **q)["M_seg"] == plane


# README saved-set table row (by its item cell's start) -> the cake_gdn_det_train_kcp_saved_layout terms it states
README_SAVED_ROWS = {
    "inputs ": (),
    "`S_in`": ("S_in",),
    "received summary pairs": ("c1_recv",),
    "`M_seg`": ("M_seg",),
    "convolution history rows": ("hist",),
    "`gamma`, `gcum`, `Akk`": ("gamma", "gcum", "akk"),
    "`rowscale`, `qscale`": ("rowscale", "qscale"),
    "`h` tape, `v_new`, `w`, `u`, `Aqk`": ("h_tape", "v_new", "w", "u", "aqk"),
    "`qk_rstd`": ("qk_rstd",),
    "`qn`, `kn`": ("qn", "kn"),
}
# the ten forward tapes save_mode="full" keeps (the options table's save_mode row states their bytes per row)
FULL_TAPE_TERMS = (
    "gamma",
    "gcum",
    "akk",
    "rowscale",
    "qscale",
    "h_tape",
    "v_new",
    "w",
    "u",
    "aqk",
)


def test_readme_saved_set_table_matches_the_saved_layout() -> None:
    """The README's saved-set table states what ``cake_gdn_det_train_kcp_saved_layout`` counts: every layout term sits in one
    row; a ``yes`` column is a nonzero term on some rank (under ``save_transfer`` and without it, unless the row is
    conditional on ``save_transfer=True``), a ``**no**`` column a zero term on every rank; the ``M_seg`` row's light
    column holds under ``save_transfer=True`` only (two ranks, a crossing fragment, ``P = 2``, the in-kernel norm
    on).  The per-token figures are the layout's bytes of the row per padded token row at ``HV = 48``, and the
    options table's ``save_mode`` row states the ten full-mode tapes the same way."""
    readme = (PACKAGE_DIR / "README.md").read_text()
    lines = readme[
        readme.index("### Saved set, replay and byte queries") :
    ].splitlines()
    head = lines.index(
        '| item | `save_mode="full"` | `save_mode="light"` | bytes per rank |'
    )
    table = []
    for line in lines[head + 2 :]:
        if not line.startswith("|"):
            break
        table.append([c.strip() for c in line.strip().strip("|").split("|")])
    rows = {}
    for item, full_cell, light_cell, bytes_cell in table:
        (prefix,) = [k for k in README_SAVED_ROWS if item.startswith(k)]
        assert prefix not in rows, ("duplicate row", item)
        rows[prefix] = (item, {"full": full_cell, "light": light_cell}, bytes_cell)
    assert set(rows) == set(README_SAVED_ROWS) and len(table) == len(rows)
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(part, None, r, CPU) for r in range(2)
    ]
    assert sum(c.n_in for c in ctxs) > 0  # a crossing fragment: S_in / hist terms exist

    def terms(save_mode: str, save_transfer: bool) -> list[dict[str, int]]:
        return [
            gb.cake_gdn_det_train_kcp_saved_layout(
                c,
                H_qk=H_qk,
                HV=HV,
                save_mode=save_mode,
                save_transfer=save_transfer,
                P=2,
                use_qk_l2norm_in_kernel=True,
            )
            for c in ctxs
        ]

    keys = [k for ks in README_SAVED_ROWS.values() for k in ks]
    assert sorted(keys) == sorted(terms("full", True)[0])  # each term in one row
    for prefix, (item, cells, _) in rows.items():
        st_only = "`save_transfer=True`" in item
        for mode, cell in cells.items():
            on, off = terms(mode, True), terms(mode, False)
            for k in README_SAVED_ROWS[prefix]:
                if cell == "yes":
                    assert any(t[k] > 0 for t in on), (item, mode, k)
                    assert any(t[k] > 0 for t in off) != st_only, (item, mode, k)
                elif cell.startswith("**no**"):
                    assert all(t[k] == 0 for t in on + off), (item, mode, k)
                else:
                    assert cell.startswith("`save_transfer=True` only"), (item, cell)
                    assert any(t[k] > 0 for t in on), (item, mode, k)
                    assert all(t[k] == 0 for t in off), (item, mode, k)
    # the per-token figures: the layout's bytes per padded token row (R = 64 NC_pad rows) at (16, 48)
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor([0, 4096], dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    assert 64 * ctx.NC_pad == ctx.R
    lay = gb.cake_gdn_det_train_kcp_saved_layout(
        ctx, H_qk=16, HV=48, save_mode="full", P=1
    )
    n_per_token = 0
    for prefix, (item, _, bytes_cell) in rows.items():
        if bytes_cell.endswith("B per token"):
            n_per_token += 1
            row_bytes = sum(lay[k] for k in README_SAVED_ROWS[prefix])
            assert row_bytes % ctx.R == 0, item
            # the whole cell: no digit may stand before the figure
            assert re.fullmatch(
                rf"(about )?{row_bytes // ctx.R:,} B per token", bytes_cell
            ), (item, row_bytes, bytes_cell)
    assert n_per_token == 3
    # the formula cells: qk_rstd = 8 T_r H_qk (saved boundary), M_seg = P n_frag s (P > 1), S_in = n_in s_pad, the
    # owner's pairs = 2 s_owner N_out
    assert rows["`qk_rstd`"][2] == "`8 * T_r * H_qk`"
    # the full set's normalized rows: qn = kn = 256 T_r H_qk (BF16, K = 128), 8,192 B per token for both at H_qk = 16
    assert (
        rows["`qn`, `kn`"][2]
        == "`2 * 256 * T_r * H_qk` (8,192 B per token at `H_qk = 16`)"
    )
    lay_qk = gb.cake_gdn_det_train_kcp_saved_layout(
        ctx, H_qk=16, HV=48, save_mode="full", P=1, use_qk_l2norm_in_kernel=True
    )
    assert lay_qk["qn"] + lay_qk["kn"] == 2 * 256 * ctx.T_r * 16 == 8192 * ctx.T_r
    assert rows["`M_seg`"][2] == "`P * n_frag * s`"
    assert rows["`S_in`"][2] == "`n_in * s_pad`"
    assert rows["received summary pairs"][2] == "`2 * s_owner * N_out`"
    # a padded owner (HV = 48 on five ranks: Hg = 10, five owners, the last holds 8 heads and 2 padding tiles): S_in
    # is n_in s_pad (s_pad = n_owners s_owner, above s) and every owner receives 2 s_owner per crossing-out fragment
    # of the group
    part5 = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor([0, 4096], dtype=torch.int64), 5, "contiguous"
    )
    ctxs5 = [
        gp.cake_gdn_det_train_kcp_build_context(part5, None, r, CPU) for r in range(5)
    ]
    assert all(c.N_out > 0 for c in ctxs5) and any(c.n_in > 0 for c in ctxs5)
    for c in ctxs5:
        hp = c.heads(48)
        assert (hp.Hg, hp.n_owners, hp.pad_tiles) == (10, 5, 2)
        s_owner = hp.Hg * gp.STATE_TILE_BYTES
        s_pad = hp.n_owners * s_owner
        assert s_pad == (48 + 2) * gp.STATE_TILE_BYTES
        lay5 = gb.cake_gdn_det_train_kcp_saved_layout(
            c, H_qk=16, HV=48, save_transfer=True
        )
        assert lay5["S_in"] == c.n_in * s_pad, (c.rank, lay5)
        assert lay5["c1_recv"] == 2 * s_owner * c.N_out, (c.rank, lay5)
    for c in ctxs:
        s = 6 * gp.STATE_TILE_BYTES
        lay2 = gb.cake_gdn_det_train_kcp_saved_layout(
            c, H_qk=H_qk, HV=HV, P=2, use_qk_l2norm_in_kernel=True
        )
        assert lay2["qk_rstd"] == 8 * c.T_r * H_qk
        assert lay2["M_seg"] == 2 * c.n_frag * s and lay2["S_in"] == c.n_in * s
        # the replay boundary keeps no boundary state, M_seg, tape or qk_rstd (only the convolution rows remain)
        for save_mode in ("light", "full"):
            rep = gb.cake_gdn_det_train_kcp_saved_layout(
                c,
                H_qk=H_qk,
                HV=HV,
                P=2,
                save_mode=save_mode,
                boundary_mode="replay",
                use_qk_l2norm_in_kernel=True,
            )
            assert {k for k, v in rep.items() if v} <= {"hist"}, (save_mode, rep)
    # qn / kn (the full set's normalized rows) sit in their own row, not in the save_mode row's ten tapes
    assert set(FULL_TAPE_TERMS) == {
        k
        for k in keys
        if k not in ("S_in", "c1_recv", "M_seg", "hist", "qk_rstd", "qn", "kn")
    }
    tapes = sum(lay[k] for k in FULL_TAPE_TERMS)
    assert tapes % ctx.R == 0
    (save_mode_row,) = [
        ln for ln in readme.splitlines() if ln.startswith("| `save_mode` |")
    ]
    assert re.search(
        rf"(?<![\d,]){tapes // ctx.R:,} B per padded token row at `HV = 48`",
        save_mode_row,
    ), save_mode_row


def _nan_filled_allocations(monkeypatch, kcp) -> None:
    """Fill every tensor the package allocates (``kcp._alloc``: workspace pools, tapes, fresh saved-set tensors) with
    NaN bytes, so a read of a buffer no program of the call wrote shows up as a non-finite result."""
    real_alloc = kcp._alloc

    def nan_alloc(shape, dtype, device):
        t = real_alloc(shape, dtype, device)
        if t.numel():
            t.view(-1).view(torch.uint8).fill_(0xFF)
        return t

    monkeypatch.setattr(kcp, "_alloc", nan_alloc)


@pytest.mark.parametrize("use_qk_l2norm_in_kernel", (False, True))
def test_full_backward_without_save_transfer_reads_no_unwritten_plane(
    monkeypatch, use_qk_l2norm_in_kernel
) -> None:
    """``save_mode="full"`` with ``save_transfer=False`` at ``P > 1`` (also the only form of ``exchange="ring"``): the
    backward's merges read the forward's ``M_seg^T`` plane (``stage_m``), which the full backward does not rebuild (no
    transfer walk), so the saved set carries it as ``M_seg`` and the backward binds it.  Every tensor the package
    allocates is filled with NaN bytes: an unwritten read shows up as a non-finite result.  The full form gives the
    light form's bits on a ``None``, a fresh and a shared workspace (twice) and under a non-reentrant checkpoint; the
    byte query counts ``M_seg``; a full saved set without it is refused."""
    kcp = _package("cake_kcp")
    _nan_filled_allocations(monkeypatch, kcp)
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    base = {
        "segments": 2,
        "save_transfer": False,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
    }
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    _, saved = kcp.cake_gdn_det_train_kcp_forward(
        *args, backend="fake", save_mode="full", **base
    )
    assert saved is not None and saved.P == 2 and saved.c1_recv is None
    assert saved.M_seg is not None and torch.isfinite(saved.M_seg).all()
    m_seg_bytes = gb.cake_gdn_det_train_kcp_saved_layout(
        ctx,
        H_qk=H_qk,
        HV=HV,
        save_mode="full",
        save_transfer=False,
        P=2,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
    )["M_seg"]
    # the query counts the saved plane itself: NSEG = n_frag P states
    assert m_seg_bytes == 2 * ctx.n_frag * HV * gp.STATE_TILE_BYTES
    assert m_seg_bytes == saved.M_seg.numel() * saved.M_seg.element_size()
    # an older full saved set without M_seg would read an unwritten plane: refused
    with pytest.raises(ValueError, match="needs M_seg"):
        kcp.cake_gdn_det_train_kcp_backward(
            *args,
            ops["dout"],
            saved=dataclasses.replace(saved, M_seg=None),
            backend="fake",
            save_mode="full",
            **base,
        )
    light = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    for n, t in light.items():
        assert torch.isfinite(t.float()).all(), ("light", n)
    shared = kcp.CakeGDNDetKcpWorkspace()
    arms = (
        (None, False),
        (kcp.CakeGDNDetKcpWorkspace(), False),
        (shared, False),
        (shared, False),
        (None, True),
    )
    for i, (ws, ckpt) in enumerate(arms):
        full = _ckpt_autograd(
            kcp, ctx, ops, ckpt=ckpt, workspace=ws, save_mode="full", **base
        )
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved", i
        _assert_same(full, light, f"save_mode=full without save_transfer, arm {i}")


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
    assert "scratch" in kcp.CakeGDNDetKcpCall.transpose_tiles.__kwdefaults__


def test_workspace_layout_counts_one_staging_plane_under_save_transfer() -> None:
    """R2-L1 1a: ``cake_gdn_det_train_kcp_workspace_layout`` counts the ``stage_m`` plane only when the call sizes it (the replay
    boundary, or the light saved form without ``save_transfer``); when the saved set carries ``M_seg`` -- the shipped
    saved + ``save_transfer`` form and ``save_mode="full"`` with either ``save_transfer`` -- the saved ``M_seg^T``
    stands in for it and the workspace holds ``stage_b`` alone (the ring pools are never workspace terms).  The full
    form without ``save_transfer`` is also checked against the pools a call at ``P = 2`` actually sizes."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    part = gp.cake_gdn_det_train_kcp_partition(cu, 2, "contiguous")
    H_qk, HV = 2, 6
    seen_segmented = False
    # Under the shipped row rule ("small_one" here; "row_cost_lpt" too) this row plans P = 1 on both ranks ("row_cost" plans P = 1 there -- fragments of 2-14
    # chunks: the per-rank programs
    # segment them 7- and 13-fold to fill the SMs, and the compose chain of those segments costs more than the one-wave
    # walks they save), so the staging plane is 0; the plane term is exercised under the per-rank rule below.
    for r in range(2):
        ctx_row = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        hp_row = ctx_row.heads(HV)
        assert hp_row.segments.p_rule == gp.P_RULE_DEFAULT and hp_row.P == 1
        one_row = gb.cake_gdn_det_train_kcp_workspace_layout(ctx_row, H_qk=H_qk, HV=HV)
        assert one_row["fwd.staging"] == one_row["bwd.staging"] == 0
    for r in range(2):
        ctx = gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )
        hp = ctx.heads(
            HV
        )  # the layout's staging term uses the plan's own segment count
        plane = hp.NSEG * HV * gp.STATE_TILE_BYTES if hp.P > 1 else 0
        seen_segmented = seen_segmented or hp.P > 1
        one = gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
        two_a = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, boundary_mode="replay"
        )
        two_b = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_transfer=False
        )
        full_no_st = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", save_transfer=False
        )
        assert one["fwd.staging"] == one["bwd.staging"] == plane
        assert two_a["fwd.staging"] == two_a["bwd.staging"] == 2 * plane
        assert two_b["fwd.staging"] == two_b["bwd.staging"] == 2 * plane
        assert full_no_st["fwd.staging"] == full_no_st["bwd.staging"] == plane
        assert one["fwd.c1_recv"] == 0 and two_b["fwd.c1_recv"] > 0
        assert not any(k.endswith("ring_out") or k.endswith("ring_in") for k in one)
        assert (
            gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, P=1)[
                "fwd.staging"
            ]
            == 0
        )
    assert seen_segmented  # the row has a segmented rank: the plane term was exercised
    # save_mode="full" without save_transfer at a forced P = 2, against the pools the call sizes: the forward binds a
    # fresh M_seg^T plane and the backward the saved one for stage_m, so both lifetimes size stage_b alone.  The
    # context's own plan is P = 2 (the layout's staging term reads the plan's NSEG).
    kcp = _package("cake_kcp")
    cu2 = [0, 2000, 4096]
    part2 = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu2, dtype=torch.int64), 1, "contiguous"
    )
    ctx2 = gp.cake_gdn_det_train_kcp_build_context(
        part2, None, 0, CPU, 4, sm_count=16, p_rule="wave_fill1"
    )
    assert ctx2.heads(HV).P == 2
    plane2 = ctx2.heads(HV).NSEG * HV * gp.STATE_TILE_BYTES
    ops = _inputs(cu2, H_qk, HV)
    held: dict[str, dict[str, int]] = {"fwd": {}, "bwd": {}}
    phase = ["fwd"]

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            pools = args["_call"].ws.pool_bytes
            for pool in ("stage_b", "stage_m"):
                if pool in pools:
                    got = held[phase[0]]
                    got[pool] = max(got.get(pool, 0), pools[pool])
            super().run(name, args)

    opts = dict(backend="fake", segments=2, save_mode="full", save_transfer=False)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx2)
    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        _, saved = kcp.cake_gdn_det_train_kcp_forward(*args, **opts)
        phase[0] = "bwd"
        kcp.cake_gdn_det_train_kcp_backward(*args, ops["dout"], saved=saved, **opts)
    finally:
        if previous is None:
            kcp._BACKENDS.pop("fake", None)
        else:
            kcp._BACKENDS["fake"] = previous
    assert kcp.LAST_CALL["fwd"]["P"] == kcp.LAST_CALL["bwd"]["P"] == 2
    assert saved is not None and saved.M_seg is not None
    assert saved.M_seg.numel() * saved.M_seg.element_size() == plane2
    assert held["fwd"] == held["bwd"] == {"stage_b": plane2}, held
    full2 = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx2, H_qk=H_qk, HV=HV, save_mode="full", save_transfer=False, P=2
    )
    assert full2["fwd.staging"] == full2["bwd.staging"] == plane2


def _held_pools(kcp, ctx, ops, *, pools=("stage_b", "stage_m", "carry"), **opts):
    """The explicit forward and backward entries (per-call workspaces) under a recording fake backend: the most bytes
    each pool of ``pools`` holds at a launch of the forward and of the backward (a pool the call does not size is
    absent), and the saved set."""
    held: dict[str, dict[str, int]] = {"fwd": {}, "bwd": {}}
    phase = ["fwd"]

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            sizes = args["_call"].ws.pool_bytes
            got = held[phase[0]]
            for pool in pools:
                if pool in sizes:
                    got[pool] = max(got.get(pool, 0), sizes[pool])
            super().run(name, args)

    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    previous = kcp.set_backend("fake", Recording())
    try:
        _, saved = kcp.cake_gdn_det_train_kcp_forward(*args, backend="fake", **opts)
        phase[0] = "bwd"
        kcp.cake_gdn_det_train_kcp_backward(
            *args,
            ops["dout"],
            saved=saved,
            backend="fake",
            **opts,
        )
    finally:
        kcp.set_backend("fake", previous)
    return held, saved


@pytest.mark.parametrize(
    "opts,planes",
    [
        ({}, 1),  # the shipped light saved form with save_transfer: stage_b alone
        ({"save_transfer": False}, 2),
        ({"save_mode": "full", "save_transfer": False}, 1),
        ({"save_mode": "full"}, 1),
        ({"boundary_mode": "replay"}, 2),
    ],
)
def test_workspace_layout_follows_the_segments_override(opts, planes) -> None:
    """``cake_gdn_det_train_kcp_workspace_layout(P=...)`` reports the planes of a call at that ``segments`` override: on the
    checkpoint pack (one rank, five fragments, the plan's own ``P = 1``) a call with ``segments=2`` holds staging
    planes of ``NSEG = 2 n_frag`` states and a carry plane of ``n_frag`` states in both lifetimes, and the layout at
    ``P = 2`` counts the same bytes in every save form (it read the staging plane from the plan's ``NSEG``, half the
    call's here).  An override outside the range the call accepts is refused by both byte queries, as by
    ``CakeGDNDetKcpContext.heads``."""
    kcp, ctx, ops = _ckpt_setup()
    H_qk, HV = CKPT_HEADS
    assert ctx.heads(HV).P == 1 and ctx.n_frag == 5
    s = HV * gp.STATE_TILE_BYTES
    plane = 2 * ctx.n_frag * s
    assert plane == 3_932_160
    held, _ = _held_pools(kcp, ctx, ops, segments=2, **opts)
    assert kcp.LAST_CALL["fwd"]["P"] == kcp.LAST_CALL["bwd"]["P"] == 2
    layout = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, P=2, **opts
    )
    for phase in ("fwd", "bwd"):
        staging = held[phase].get("stage_b", 0) + held[phase].get("stage_m", 0)
        assert staging == planes * plane == layout[f"{phase}.staging"], (
            phase,
            held,
            layout[f"{phase}.staging"],
        )
        assert held[phase]["carry"] == ctx.n_frag * s == layout[f"{phase}.carry"], (
            phase,
            held,
            layout[f"{phase}.carry"],
        )
    rule = gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, **opts)
    assert rule["fwd.staging"] == rule["fwd.carry"] == 0  # the plan's P = 1
    # the range the call accepts: [1, min(MAX_SEGMENTS, chunks of the longest fragment)] = [1, 16] (1000 tokens)
    for P in (0, 17):
        for query in (
            gb.cake_gdn_det_train_kcp_workspace_layout,
            gb.cake_gdn_det_train_kcp_saved_layout,
        ):
            with pytest.raises(ValueError, match=r"override must be in \[1, 16\]"):
                query(ctx, H_qk=H_qk, HV=HV, P=P, **opts)
        with pytest.raises(ValueError, match=r"override must be in \[1, 16\]"):
            ctx.heads(HV, P=P)
    top = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, P=16, **opts
    )
    assert top["fwd.staging"] == planes * 16 * ctx.n_frag * s


def test_backward_allocates_the_gradients_right_before_dqkg() -> None:
    """R2-L1 1b: the five gradient outputs are allocated after the adjoint tape walk (right before #8), so they are
    not live at the backward's peak moment (the dh_tape allocation of the adjoint walk); under fused walks at P > 1
    the b2_recv pool is released before that walk (the dcarry_cross merge copied dS_out into the carry slots)."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
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
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.cake_gdn_det_train_kcp_backward(
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

CKPT_CU = [0, 100, 740, 817, 1817, 1946]
CKPT_HEADS = (2, 6)
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
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_det_train_kcp_build_context(part, None, 0, CPU)
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
    """``out`` and the five gradients of one ``cake_gdn_det_kcp`` layer (or ``layers`` chained ones: the next layer's
    ``v`` is the previous output), directly or under a non-reentrant checkpoint; ``twice`` backwards with
    ``retain_graph`` first (accumulation)."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", **opts}

    def fn(q, k, v, g, beta):
        out = kcp.cake_gdn_det_kcp(q, k, v, g, beta, ctx, workspace=workspace, **o)
        for _ in range(layers - 1):
            out = kcp.cake_gdn_det_kcp(
                q, k, out, g, beta, ctx, workspace=workspace, **o
            )
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
    forward_programs = {
        "fwd_prep",
        "kcp_fwdh_offset",
        "kcp_fwdh_tape",
        "kcp_fwdh_out",
        "fwd_o",
    }
    assert not (forward_programs & set(kept_seq)), kept_seq
    bwd_launches = [n for n, bwd, _, _ in seen if bwd]
    assert (
        bwd_launches
        and bwd_launches[0] == "bwd_prep"
        and not (forward_programs & set(bwd_launches))
    )
    # two forwards ran (the checkpointed one, then torch's recompute = the hand-off forward); the first hands no
    # tape over, so it writes the output with the output walk (fused, where the registry ships it); on a verified
    # torch the recompute's #5 is left out (recompute_output="auto": this node is the segment's last saver); any
    # other torch build launches it after the hand-off forward's tape walk
    assert [h for n, bwd, h, _ in seen if not bwd and n == "fwd_prep"] == [False, True]
    fused = kcp.fused_o_shipped()
    assert [h for n, bwd, h, _ in seen if not bwd and n == "kcp_fwdh_out"][:1] == (
        [False] if fused else []
    )
    assert all(not h for n, bwd, h, _ in seen if not bwd and n == "kcp_fwdh_out")
    assert [h for n, bwd, h, _ in seen if not bwd and n == "fwd_o"] == (
        [] if fused else [False]
    ) + ([] if kcp.torch_release_verified() else [True])
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
    ws = kcp.CakeGDNDetKcpWorkspace()
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
        return a_ * a_, kcp.cake_gdn_det_kcp(
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
            kcp.cake_gdn_det_kcp(
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
    ``fwd_o`` / ``kcp_fwdh_out`` launch; the scratch carries that forward's ``fwd_o`` decision and reason once it has
    run."""
    seen: list[tuple[str, bool, dict]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward and name in ("fwd_prep", "fwd_o", "kcp_fwdh_out"):
                seen.append(
                    (name, bool(call.scratch.get("hand_off", False)), call.scratch)
                )
            super().run(name, args)

    return seen, Recording()


def _launches(seen, name: str) -> list[bool]:
    return [h for n, h, _ in seen if n == name]


def _o_writes(seen) -> list[bool]:
    """The hand-off flags of the forwards that wrote the output, in launch order: by #5 (decision ``launched``) or by
    the output walk (``fused``); each forward's own launches agree with its decision (``fwd_o`` once when launched,
    ``kcp_fwdh_out`` when fused, neither when skipped)."""
    writes = []
    for i, (name, hand_off, scratch) in enumerate(seen):
        if name != "fwd_prep":
            continue
        own = [n for n, _, s in seen[i + 1 :] if s is scratch and n != "fwd_prep"]
        decision = scratch["fwd_o"]
        assert own.count("fwd_o") == int(decision == "launched"), (decision, own)
        assert ("kcp_fwdh_out" in own) == (decision == "fused"), (decision, own)
        if decision != "skipped":
            writes.append(hand_off)
    return writes


def _wrote(kcp, hand_off: bool) -> str:
    """The ``fwd_o`` decision of a forward that writes the output: ``fused`` (the output walk) unless it hands its
    tapes to the backward or the registry ships no output walk (``launched``: the tape walk, then #5)."""
    return "fused" if not hand_off and kcp.fused_o_shipped() else "launched"


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


def _auto_hand_off(kcp) -> bool:
    """Whether the shipped ``reuse_recompute_tapes="auto"`` hands the recompute's tapes to the backward when the node
    closes the segment: the frame proof is read on a verified torch only; elsewhere every recompute takes the replay
    form (no hand-off, no kept entry)."""
    return _verified_torch(kcp)


def _dead_decision(kcp, hand_off: bool = True) -> str:
    """What a recompute whose output is dead records on this build: ``skipped`` on a verified torch; elsewhere the
    recompute writes the output, as :func:`_wrote` gives for a forward with or without the hand-off (``launched``
    with the hand-off of ``reuse_recompute_tapes=True``)."""
    return "skipped" if _verified_torch(kcp) else _wrote(kcp, hand_off)


def _fwd_o_after_recompute(kcp, hand_off: bool = True) -> list[bool]:
    """``fwd_o`` hand-off flags of one original forward and one recompute whose output is dead: the original's only on
    a verified torch, the recompute's too elsewhere (``hand_off``: whether that recompute hands its tapes off)."""
    return [False] if _verified_torch(kcp) else [False, hand_off]


def _skip_delta(kcp, before: dict[str, int]) -> dict[str, int]:
    return {k: kcp.RECOMPUTE_OUTPUT_STATS[k] - before[k] for k in before}


def _ckpt_leaves(ops) -> list[torch.Tensor]:
    return [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]


def test_recompute_output_skipped_once_when_the_node_closes_the_segment() -> None:
    """A segment whose last saver is the GDN call: the recompute leaves #5 out exactly once per checkpoint step --
    fwd_prep runs in both forwards, the output program (the output walk) in the original only -- the outputs and the gradients are bitwise the
    ``recompute_output="always"`` form's and the direct autograd's, the kept-tape accounting is unchanged, and the
    skip does not rest on the kept tapes (the replay form skips too).  A later op of the segment that saves nothing
    (``.sum()``) keeps the skip: torch's early stop ends the recompute right after this node's tensors are re-saved,
    before that op runs.  On a torch build outside the verified set every recompute writes the output instead (reason
    ``torch_version``, counters untouched): the shipped ``reuse_recompute_tapes="auto"`` takes the replay form
    there, so the recompute hands nothing off, puts no kept entry and writes the output as a plain forward; the same
    bitwise equalities hold."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    hand = _auto_hand_off(kcp)
    fwd_o, decision = _fwd_o_after_recompute(kcp, hand), _dead_decision(kcp, hand)
    reason, delta = _frame_reason(kcp, "dead"), _frame_delta(kcp, 1, 0)

    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)

    def seg_sum(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.cake_gdn_det_kcp(
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
        assert _launches(seen, "fwd_prep") == [False, hand]
        assert _o_writes(seen) == [False, hand]
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == _wrote(kcp, hand)
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == "always"
        seen.clear()
        before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
        auto = _ckpt_autograd(kcp, ctx, ops, ckpt=True, segments=2)
        assert _launches(seen, "fwd_prep") == [False, hand]
        assert _o_writes(seen) == fwd_o
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == decision
        assert kcp.LAST_CALL["fwd"]["fwd_o_reason"] == reason
        assert _skip_delta(kcp, before) == delta
        assert _stats_delta(kcp, kept_before) == {
            "put": int(hand),
            "take": int(hand),
            "miss": 0,
            "released": int(hand),
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
        assert _o_writes(seen) == ([False] if _verified_torch(kcp) else [False, False])
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "replay"
        assert _skip_delta(kcp, before) == delta
        seen.clear()
        before = dict(kcp.RECOMPUTE_OUTPUT_STATS)
        ckpt_sum = seg_sum(True)
        assert _o_writes(seen) == fwd_o
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
# The untaken-node test runs the default "auto" (its hand-off holds on a verified torch only: _auto_hand_off).
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
            out = kcp.cake_gdn_det_kcp(
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
        assert _o_writes(seen) == [False, True]
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
    ws = kcp.CakeGDNDetKcpWorkspace()
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
        assert [f[:2] for f in fwds[:layers]] == [(False, _wrote(kcp, False))] * layers
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
        assert _o_writes(seen) == [False] * layers + [True] * layers
    finally:
        kcp.set_backend("fake", previous)
    _assert_same(auto, direct, f"{layers} layers, auto vs direct")
    _assert_same(auto, always, f"{layers} layers, auto vs always")
    assert kcp.kept_registry_size() == 0


def test_recompute_output_with_an_untaken_node_and_a_second_backward() -> None:
    """(a) The segment is recomputed for another saved tensor while the GDN node itself is not in the backward: the
    node is still the segment's last saver, so #5 is left out, and the kept-tape accounting is the shipped one (the
    untaken entry goes at the task's end).  (b) ``retain_graph`` and a second ``.backward()``: two recomputes, two
    skips, two kept entries put and taken, gradients exactly twice the direct ones.  Both parts run the shipped
    ``reuse_recompute_tapes="auto"``: on a torch build outside the verified set it takes the replay form (no kept
    entry is put, taken or dropped), the recompute writes the output as a plain forward and the counters stay."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    hand = int(_auto_hand_off(kcp))  # kept entries per recompute
    leaves = _ckpt_leaves(ops)
    a = torch.randn(3, requires_grad=True)

    def fn(a_, q, k, v, g, beta):
        return a_ * a_, kcp.cake_gdn_det_kcp(
            q, k, v, g, beta, ctx, backend="fake", segments=2
        )

    b, out = checkpoint(fn, a, *leaves, use_reentrant=False)
    before, kept_before = dict(kcp.RECOMPUTE_OUTPUT_STATS), dict(kcp.KEPT_STATS)
    b.sum().backward()
    assert a.grad is not None and all(lf.grad is None for lf in leaves)
    assert kcp.LAST_CALL["fwd"]["fwd_o"] == _dead_decision(kcp, bool(hand))
    assert _skip_delta(kcp, before) == _frame_delta(kcp, 1, 0)
    assert _stats_delta(kcp, kept_before) == {
        "put": hand,
        "take": 0,
        "miss": 0,
        "released": 0,
        "callback": hand,
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
        "put": 2 * hand,
        "take": 2 * hand,
        "miss": 0,
        "released": 2 * hand,
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
        assert _o_writes(seen) == [False, True]
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
            return kcp.cake_gdn_det_kcp(
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
        assert _o_writes(seen) == [False, True]
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

    ws = kcp.CakeGDNDetKcpWorkspace()

    def run(ckpt: bool, **o) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        opt = dict(
            backend="fake", segments=2, workspace=ws, reuse_recompute_tapes=True, **o
        )

        def fn(q, k, v, g, beta):
            first = kcp.cake_gdn_det_kcp(
                q.detach(),
                k.detach(),
                v.detach(),
                g.detach(),
                beta.detach(),
                ctx,
                **opt,
            )
            assert not first.requires_grad
            return kcp.cake_gdn_det_kcp(q, k, first, g, beta, ctx, **opt)

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
            (False, _wrote(kcp, False)),
            (False, _wrote(kcp, False)),
            (True, "launched"),
            (True, _dead_decision(kcp)),
        ]
        assert fwds[0][2] == fwds[2][2] == "not_executable"
        assert fwds[3][2] == _frame_reason(kcp, "dead")
        assert _skip_delta(kcp, before) == _frame_delta(kcp, 1, 0)
        seen.clear()
        always = run(True, recompute_output="always")
        assert [f[1:] for f in _forwards(seen)] == [
            (_wrote(kcp, False), "always")
        ] * 2 + [("launched", "always")] * 2
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
                out = kcp.cake_gdn_det_kcp(
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
            (False, _wrote(kcp, False), "not_executable"),
            (True, "launched", "not_executable"),
        ]
        assert _skip_delta(kcp, before) == {"skipped": 0, "launched": 0}
        seen.clear()
        always = run(True, recompute_output="always")
        assert _forwards(seen) == [
            (False, _wrote(kcp, False), "always"),
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
            return kcp.cake_gdn_det_kcp(
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
        assert _o_writes(seen) == [False, True]
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
                out = kcp.cake_gdn_det_kcp(
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
            (False, _wrote(kcp, False), reason),
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
# torch.tensor(values, dtype).  The no-sync contract itself is measured on a device (test_cake_gdn_det_train_kcp_cp1.py, the dist test).


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
            assert _o_writes(seen) == fwd_o, build
        finally:
            kcp.set_backend("fake", previous)
        _assert_same(res, direct, build)


# --------------------------------------------------------------------------- issue #7074 (4): cake_gdn_det_train_kcp_workspace_layout follows the arena's lifetimes
# What a caller-held workspace binds, per tape form: the forward's tape-walk peak and the backward's phase maximum.


class _PeakWorkspace:
    """A ``CakeGDNDetKcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.CakeGDNDetKcpWorkspace):
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
    """``cake_gdn_det_train_kcp_workspace_layout`` is what a caller-held workspace binds: the forward's peak (the shipped fused
    forward: #1; with test knob ``fused_o=0``: the tape walk, ``forward_split``), and the backward's phase maximum of
    the form -- the kept-tape form ends with the re-adopted forward tapes (the fused set ``aqk`` / ``qscale``; the
    split forward's four with ``fused_o=0``) beside the backward tapes (the ten kept tapes are hand-off tensors outside
    the workspace, ``bwd.kept_tapes``), the replay form holds the ten light tapes through the backward; both forms need
    the same transient bytes here (``save_transfer`` on, no norm)."""
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
    assert (
        tuple(gb.TAPE_NAME_OF.get(n, n) for n in gb.FWD_TAPES_ARENA_AFTER_FUSED)
        == kcp.FWD_TAPES_ARENA_AFTER_FUSED
    )
    assert gb.EYE_BYTES == gb.STATE_TILE_BYTES == 128 * 128 * 4
    from torch.utils.checkpoint import checkpoint

    layouts = {}
    for form, fused in itertools.product((True, False), (0, 1)):
        layout = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
        )
        layouts[form] = layout
        ws = _PeakWorkspace(kcp)
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

        def fn(q, k, v, g, beta, _ws=ws, _form=form, _fused=fused):
            return kcp.cake_gdn_det_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                workspace=_ws,
                backend="fake",
                reuse_recompute_tapes=_form,
                test_knobs={"fused_o": _fused},
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        want = layout["forward"] if fused else layout["forward_split"]
        assert ws.peak == want, (form, fused, ws.peak, want)
        ws.peak = 0
        out.backward(ops["dout"])
        if not fused:
            # the split forward's kept backward re-adopts its four tapes (the layout reports the shipped fused form)
            left = set(ws._tapes) & set(kcp.FWD_TAPES)
            assert left == set(
                kcp.FWD_TAPES_ARENA_AFTER_FWD if form else kcp.FWD_TAPES
            ), (form, sorted(left))
            assert kcp.kept_registry_size() == 0
            continue
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
                gb.FWD_TAPES_ARENA_AFTER_FUSED
                if form
                else tuple(k for k in gb.FWD_TAPES_ARENA_AFTER_FWD)
                + ("w", "u", "akk", "gamma", "gcum", "rowscale")
            )
        )
        assert kcp.kept_registry_size() == 0
    kept, replay = layouts[True], layouts[False]
    # the fused forward binds no h / v_new at any moment (its peak is #1, akk / gcum still bound)
    assert (
        replay["forward_split"] - replay["forward"]
        == (
            replay["bwd.h_tape"]
            + replay["bwd.v_new"]
            - replay["bwd.akk"]
            - replay["bwd.gcum"]
        )
        > 0
    )
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
    default = gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
    opt = _package("cake_options")
    assert default == layouts[opt.REUSE_RECOMPUTE_TAPES_DEFAULT is True]
    assert (
        gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="auto"
        )
        == replay
    )
    for form in (True, False):
        assert (
            gb.cake_gdn_det_train_kcp_workspace_bytes(
                ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
            )
            == layouts[form]["workspace"]
        )
    with pytest.raises(ValueError, match="reuse_recompute_tapes must be"):
        gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="yes"
        )


@pytest.mark.parametrize("segments", (1, 2))
def test_workspace_layout_of_a_forward_without_a_saved_set(segments) -> None:
    """A caller-held workspace and ``cake_gdn_det_train_kcp_forward`` with the norm: without a saved set
    (``save_for_backward=False``) the fused forward's #1 binds the arena ``r`` (``fwd.qk_rstd``, ``8 T_r H_qk``) and the
    workspace pools a saved set replaces (the ``stage_m`` plane at ``P > 1``).  The forward's peak is the ``forward`` of
    ``cake_gdn_det_train_kcp_workspace_layout(..., save_for_backward=False, backward=False)`` exactly (with ``fused_o=0`` its
    ``forward_split``: ``r`` goes before the tape walk); the saved-set forward's peak is the default layout's, which the
    keyword leaves unchanged."""
    kcp, ctx, ops = _ckpt_setup()
    gb = _package("cake_bytes")
    H_qk, HV = 2, 6
    leaves = [ops[n] for n in LEAVES]
    o = dict(use_qk_l2norm_in_kernel=True)
    default = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, P=segments, **o
    )
    for save in (False, True):
        layout = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            P=segments,
            backward=save,
            save_for_backward=save,
            **o,
        )
        for fused in (1, 0):
            ws = _PeakWorkspace(kcp)
            _out, saved = kcp.cake_gdn_det_train_kcp_forward(
                *leaves,
                ctx,
                workspace=ws,
                backend="fake",
                save_for_backward=save,
                segments=segments,
                test_knobs={"fused_o": fused},
                **o,
            )
            assert (saved is not None) == save
            assert kcp.LAST_CALL["fwd"]["fwd_o"] == ("fused" if fused else "launched")
            assert kcp.LAST_CALL["fwd"]["P"] == segments
            want = layout["forward"] if fused else layout["forward_split"]
            assert ws.peak == want, (save, fused, ws.peak, want)
            assert not {*kcp.QK_TAPES, kcp.QK_RSTD} & set(ws._tapes), sorted(ws._tapes)
        assert layout["fwd.qk_rstd"] == (0 if save else 8 * ctx.T_r * H_qk), save
    assert layout == default
    # the replay boundary keeps no saved set either way (a saved boundary without one: its backward is the saved=None
    # replay, test_workspace_layout_option_checks)
    replay = dict(boundary_mode="replay", P=segments, **o)
    assert gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, H_qk=H_qk, HV=HV, save_for_backward=False, **replay
    ) == gb.cake_gdn_det_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, **replay)


@pytest.mark.parametrize(
    "save_transfer,l2n",
    [(False, False), (True, False), (False, True)],
    ids=["no_save_transfer", "save_transfer", "no_save_transfer-norm"],
)
def test_kept_form_at_two_segments_follows_the_arena_and_the_hand_off(
    monkeypatch, save_transfer, l2n
) -> None:
    """The kept-tape form at ``P = 2`` under a non-reentrant checkpoint, measured on a caller-held workspace: without
    ``save_transfer`` the recompute forward hands its ``M_seg^T`` staging plane over with the ten tapes, so the
    backward's workspace sizes ``stage_b`` only (one staging plane, as with the saved ``M_seg``); ``bwd.kept_tapes``
    is every byte the hand-off holds (with the norm also ``qn`` / ``kn``); the backward binds ``backward`` (its
    recompute forward or a later phase); the workspace's most before the re-adoption plus the hand-off is
    ``backward_transient`` -- the replay form's figure, plus the handed-over plane through the adjoint tape walk and
    #8, less the arena ``qk_rstd`` the kept form does not need -- and the end state is ``bwd.phase.end``."""
    from torch.utils.checkpoint import checkpoint

    kcp, ctx, ops = _ckpt_setup()
    H_qk, HV = CKPT_HEADS
    q = dict(
        H_qk=H_qk,
        HV=HV,
        P=2,
        save_transfer=save_transfer,
        with_conv=False,
        use_qk_l2norm_in_kernel=l2n,
    )
    layout = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, reuse_recompute_tapes=True, **q
    )
    replay = gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, reuse_recompute_tapes=False, **q
    )
    handed: list[int] = []

    class Recorded(kcp._KeptTapes):
        def __init__(self, tapes, stage_m, geometry, gid, qk=None):
            super().__init__(tapes, stage_m, geometry, gid, qk)
            held = [*tapes.values(), *(qk or {}).values()]
            held += [stage_m] if stage_m is not None else []
            storages = {t.untyped_storage().data_ptr(): t for t in held}
            handed.append(sum(t.untyped_storage().nbytes() for t in storages.values()))

    peaks: dict[str, int] = {}
    real_backward, real_release = kcp._backward, kcp._kept_release

    def backward(call, saved):
        # the recompute forward and the backward's ensure are done: the backward proper starts here
        peaks["recompute"] = call.ws.peak
        call.ws.peak, call.ws.pool_max = 0, {}
        call.ws._upd()
        real_backward(call, saved)

    def release(call, kept):
        peaks["before_release"] = call.ws.peak
        real_release(call, kept)

    monkeypatch.setattr(kcp, "_KeptTapes", Recorded)
    monkeypatch.setattr(kcp, "_backward", backward)
    monkeypatch.setattr(kcp, "_kept_release", release)
    ws = _pool_peak_workspace(kcp)
    leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]

    def fn(q_, k_, v_, g_, beta_):
        return kcp.cake_gdn_det_kcp(
            q_,
            k_,
            v_,
            g_,
            beta_,
            ctx,
            workspace=ws,
            backend="fake",
            segments=2,
            save_transfer=save_transfer,
            reuse_recompute_tapes=True,
            use_qk_l2norm_in_kernel=l2n,
        )

    out = checkpoint(fn, *leaves, use_reentrant=False)
    assert ws.peak == layout["forward"], (ws.peak, layout["forward"])
    ws.peak = 0
    out.backward(ops["dout"])
    assert kcp.kept_registry_size() == 0
    plane = ctx.n_frag * 2 * HV * gp.STATE_TILE_BYTES
    ten = sum(replay[f"bwd.{n}"] for n in FULL_TAPE_TERMS)
    qk = sum(replay.get(f"bwd.{n}", 0) for n in ("qn", "kn"))
    assert (
        layout["bwd.staging"]
        == plane
        == replay["bwd.staging"] - (0 if save_transfer else plane)
    )
    assert layout["bwd.kept_tapes"] == ten + qk + (0 if save_transfer else plane)
    assert handed == [layout["bwd.kept_tapes"]], (handed, layout["bwd.kept_tapes"])
    assert _recorded_pool_terms(ws.pool_max) == _layout_pool_terms(layout, "bwd")
    assert max(peaks["recompute"], ws.peak) == layout["backward"], (peaks, ws.peak)
    assert peaks["before_release"] + handed[0] == layout["backward_transient"], peaks
    assert ws.bytes == layout["bwd.phase.end"]
    rstd = 8 * ctx.T_r * H_qk if l2n else 0
    assert (
        layout["backward_transient"]
        == max(
            replay["bwd.phase.exchange"],
            max(replay["bwd.phase.walk"], replay["bwd.phase.end"])
            + (0 if save_transfer else plane),
        )
        - rstd
    )


@pytest.mark.parametrize("segments", (1, 2))
def test_workspace_layout_of_a_full_forward_without_a_saved_set(segments) -> None:
    """``save_mode="full"`` saves the forward tapes as fresh tensors only when the forward keeps a saved set.  Without
    one (``save_for_backward=False``; every forward of ``boundary_mode="replay"``) the tapes are the arena's, as in the
    light split forward (a ``"full"`` forward never runs the output walk): the forward's peak is the layout's
    ``forward`` (= ``forward_split``, the light split figure) at either ``fused_o``, the forward leaves the light
    split forward's tapes bound, and the replay boundary's layout and backward peak are the light replay's."""
    kcp, ctx, ops = _ckpt_setup()
    gb = _package("cake_bytes")
    H_qk, HV = 2, 6
    leaves = [ops[n] for n in LEAVES]
    for l2n in (False, True):
        o = dict(P=segments, use_qk_l2norm_in_kernel=l2n)
        nosave = dict(backward=False, save_for_backward=False, **o)
        layout = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", **nosave
        )
        light = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, **nosave
        )
        assert layout["forward"] == layout["forward_split"] == light["forward_split"]
        saved_full = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", **o
        )
        assert layout["forward"] > saved_full["forward"], (layout, saved_full)
        for fused in (1, 0):
            ws = _PeakWorkspace(kcp)
            _out, saved = kcp.cake_gdn_det_train_kcp_forward(
                *leaves,
                ctx,
                workspace=ws,
                backend="fake",
                save_mode="full",
                save_for_backward=False,
                segments=segments,
                test_knobs={"fused_o": fused},
                use_qk_l2norm_in_kernel=l2n,
            )
            assert saved is None
            assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
            assert kcp.LAST_CALL["fwd"]["P"] == segments
            assert ws.peak == layout["forward"], (l2n, fused, ws.peak, layout)
            left = set(ws._tapes) & set(kcp.FWD_TAPES)
            assert left == set(kcp.FWD_TAPES_ARENA_AFTER_FWD), (l2n, fused, left)
        # the replay boundary keeps no saved set: "full" is the light replay, forward and backward
        replay = dict(boundary_mode="replay", **o)
        layout = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", **replay
        )
        assert layout == gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, **replay
        )
        ws = _PeakWorkspace(kcp)
        grads = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]
        out = kcp.cake_gdn_det_kcp(
            *grads,
            ctx,
            workspace=ws,
            backend="fake",
            save_mode="full",
            boundary_mode="replay",
            segments=segments,
            use_qk_l2norm_in_kernel=l2n,
        )
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        assert ws.peak == layout["forward"], (l2n, ws.peak, layout["forward"])
        ws.peak = 0
        out.backward(ops["dout"])
        assert ws.peak == layout["backward"], (l2n, ws.peak, layout["backward"])


# --------------------------------------------------------------------------- issue #7074 (2): reuse_recompute_tapes="auto"
# The recompute's tapes are kept only when the GDN node closes the checkpointed region (the frame proof); with a later saver the
# kept set would live through that operation's backward and raise the region's peak by the full tape set, so "auto" replays there.


def _ckpt_autograd_region(
    kcp, ctx, ops, *, ckpt: bool, later_saver: bool, handle=None, **opts
) -> tuple[dict[str, torch.Tensor], torch.Tensor | None]:
    """``out`` and the gradients of a region that ends with the GDN call (``later_saver=False``) or continues with an
    operation that saves tensors for its backward (a scaling by a leaf, ``later_saver=True``), directly or under a
    non-reentrant checkpoint (``handle``: the GDN call's ``recompute_handle``); the leaf's gradient comes beside the
    result."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    scale = torch.full((), 1.5, dtype=torch.bfloat16, requires_grad=True)
    o = {"backend": "fake", **opts}
    if handle is not None:
        assert ckpt, "a recompute handle needs the checkpoint"
        o["recompute_handle"] = handle

    def fn(q, k, v, g, beta):
        out = kcp.cake_gdn_det_kcp(q, k, v, g, beta, ctx, **o)
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
        opt.CakeGDNDetKcpOptions(reuse_recompute_tapes="auto")
        .check()
        .reuse_recompute_tapes
        == "auto"
    )
    assert (
        opt.CakeGDNDetKcpOptions().check().reuse_recompute_tapes
        == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    )
    assert (
        opt.CakeGDNDetKcpOptions(graph="replay").check().reuse_recompute_tapes is False
    )
    with pytest.raises(ValueError, match="needs graph='off'"):
        opt.CakeGDNDetKcpOptions(graph="replay", reuse_recompute_tapes="auto").check()
    with pytest.raises(ValueError, match="needs graph='off'"):
        opt.CakeGDNDetKcpOptions(graph="replay", reuse_recompute_tapes=True).check()
    for bad in ("yes", 1, 0, "AUTO"):
        with pytest.raises(TypeError, match="must be a bool or 'auto'"):
            opt.CakeGDNDetKcpOptions(reuse_recompute_tapes=bad).check()


# ----------------------------------------------------------------------------------------------------------------
# Recompute handle (issue #7818): the checkpoint recompute reads the original forward's incoming states from the
# handle instead of exchanging them again.  These CPU rows run one rank (no crossing fragment, ``S_in`` empty): they
# check the code path -- the capture / replay decision, the local carries at ``P > 1``, bitwise equality, retention
# and misuse; the multi-GPU rows (test_cake_gdn_det_train_kcp_dist.py) check the exchange count and the retained bytes.


def _handle_autograd(
    kcp, ctx, ops, *, handle, twice: bool = False, **opts
) -> tuple[dict[str, torch.Tensor], list]:
    """One ``cake_gdn_det_kcp`` layer under a non-reentrant checkpoint with ``recompute_handle=handle`` (``None``: the
    plain checkpoint); returns out + gradients and the ``recompute_handle`` mode of the original forward and of the
    recompute of every backward (read from ``LAST_CALL`` after each: torch's early stop ends the recompute inside the
    node's ``apply``, after the forward recorded itself)."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", "save_transfer": False, **opts}

    def fn(q, k, v, g, beta, h):
        return kcp.cake_gdn_det_kcp(q, k, v, g, beta, ctx, recompute_handle=h, **o)

    from torch.utils.checkpoint import checkpoint

    out = checkpoint(fn, *leaves, handle, use_reentrant=False)
    modes = [kcp.LAST_CALL["fwd"]["recompute_handle"]]
    out.backward(ops["dout"], retain_graph=twice)
    modes.append(kcp.LAST_CALL["fwd"]["recompute_handle"])
    if twice:
        out.backward(ops["dout"])
        modes.append(kcp.LAST_CALL["fwd"]["recompute_handle"])
    return {
        "out": out.detach(),
        **{n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)},
    }, modes


@pytest.mark.parametrize(
    "segments,save_mode", [(1, "light"), (2, "light"), (2, "full")]
)
@pytest.mark.parametrize("reuse", [True, False, "auto"])
def test_recompute_handle_is_bitwise_the_checkpoint_and_the_direct_pass(
    monkeypatch, segments, save_mode, reuse
) -> None:
    """The original forward stores ``S_in`` ("capture"), torch's recompute reads it ("replay"): out and the five
    gradients are bitwise the plain checkpoint's and the direct pass's at the same options, at ``P = 1`` and
    ``P = 2`` (the carries rebuilt locally), with and without the kept tapes.  ``save_mode="full"`` at ``P = 2``
    (its backward runs no transfer walk and reads the saved ``M_seg^T`` plane) runs with every package allocation
    filled with NaN bytes and gives the light direct pass's bits."""
    kcp, ctx, ops = _ckpt_setup()
    if save_mode == "full":
        _nan_filled_allocations(monkeypatch, kcp)
    base = dict(segments=segments, reuse_recompute_tapes=reuse, save_mode=save_mode)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_transfer=False, **base)
    if save_mode == "full":
        light = _ckpt_autograd(
            kcp,
            ctx,
            ops,
            ckpt=False,
            save_transfer=False,
            segments=segments,
            reuse_recompute_tapes=reuse,
        )
        _assert_same(direct, light, "full direct vs light direct")
    plain, plain_modes = _handle_autograd(kcp, ctx, ops, handle=None, **base)
    handle = kcp.CakeGDNDetKcpRecomputeHandle()
    got, modes = _handle_autograd(kcp, ctx, ops, handle=handle, **base)
    _assert_same(plain, direct, "plain checkpoint vs direct")
    _assert_same(got, direct, "handle checkpoint vs direct")
    assert plain_modes == [None, None], plain_modes
    assert modes == ["capture", "replay"], modes
    assert kcp.HANDLE_KEY in handle.entries
    assert kcp.kept_registry_size() == 0


def test_recompute_handle_retains_exactly_s_in_and_survives_a_second_backward() -> None:
    """The entry is the saved set's own ``S_in`` (no copy; ``nbytes`` is its size and the light byte query without
    the transfer and the convolution rows); ``retain_graph`` + a second backward reuses it and accumulates exactly
    twice the gradient."""
    kcp, ctx, ops = _ckpt_setup()
    H_qk, HV = ops["q"].shape[-2], ops["v"].shape[-2]
    handle = kcp.CakeGDNDetKcpRecomputeHandle()
    once, _ = _handle_autograd(kcp, ctx, ops, handle=kcp.CakeGDNDetKcpRecomputeHandle())
    twice, modes = _handle_autograd(kcp, ctx, ops, handle=handle, twice=True)
    assert modes == ["capture", "replay", "replay"], modes
    for n in CKPT_GRADS:
        assert torch.equal(twice[n], once[n] + once[n]), n
    s_in = handle.entries[kcp.HANDLE_KEY]["S_in"]
    assert handle.nbytes() == s_in.numel() * s_in.element_size()
    assert handle.nbytes() == kcp.cake_gdn_det_train_kcp_saved_bytes(
        ctx, H_qk=H_qk, HV=HV, save_mode="light", save_transfer=False, with_conv=False
    )
    handle.clear()
    assert handle.nbytes() == 0 and not handle.entries


def test_recompute_handle_entry_rules() -> None:
    """A forward outside the backward that finds an entry raises (a handle reused across micro-steps); a recompute
    whose handle lost its entry is an original forward again (it captures; the result is unchanged);
    ``save_transfer=True`` (keyword or options dict), ``boundary_mode="replay"``, ``graph="replay"`` and
    ``debug_sequence`` are refused; an object without an ``entries`` dict is refused."""
    kcp, ctx, ops = _ckpt_setup()
    q, k, v, g, beta = (ops[n] for n in ("q", "k", "v", "g", "beta"))
    handle = kcp.CakeGDNDetKcpRecomputeHandle()
    kcp.cake_gdn_det_kcp(q, k, v, g, beta, ctx, backend="fake", recompute_handle=handle)
    assert kcp.LAST_CALL["fwd"]["recompute_handle"] == "capture"
    with pytest.raises(
        RuntimeError, match="new CakeGDNDetKcpRecomputeHandle per micro-step"
    ):
        kcp.cake_gdn_det_kcp(
            q, k, v, g, beta, ctx, backend="fake", recompute_handle=handle
        )
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    h2 = kcp.CakeGDNDetKcpRecomputeHandle()
    from torch.utils.checkpoint import checkpoint

    def fn(*a):
        return kcp.cake_gdn_det_kcp(
            *a[:5], ctx, backend="fake", save_transfer=False, recompute_handle=a[5]
        )

    out = checkpoint(fn, *leaves, h2, use_reentrant=False)
    h2.clear()
    out.backward(ops["dout"])
    assert kcp.LAST_CALL["fwd"]["recompute_handle"] == "capture"
    assert kcp.HANDLE_KEY in h2.entries
    ref = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_transfer=False)
    for n, lf in zip(CKPT_GRADS, leaves, strict=True):
        assert torch.equal(lf.grad, ref[n]), n
    for bad in (
        {"save_transfer": True},
        {"options": {"save_transfer": True}},
        {"boundary_mode": "replay"},
        {"graph": "replay", "workspace": kcp.CakeGDNDetKcpWorkspace()},
        {"test_knobs": {"debug_sequence": True}},
    ):
        with pytest.raises(ValueError):
            kcp.cake_gdn_det_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                recompute_handle=kcp.CakeGDNDetKcpRecomputeHandle(),
                **bad,
            )
    with pytest.raises(TypeError, match="entries"):
        kcp.cake_gdn_det_kcp(
            q, k, v, g, beta, ctx, backend="fake", recompute_handle=object()
        )


# ----------------------------------------------------------------------------------------------------------------
# Packed Q/K/V (issue #7817): q / k / v as head views of the convolution's packed row; the backward writes dq / dk / dv
# into one packed row and the split hands it on without a concatenation.


def _packed_autograd(
    kcp, ctx, ops, *, batched: bool = False, ckpt: bool = False, **opts
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """``out`` and the gradients of one layer whose q / k / v come from :func:`cake_gdn_det_train_kcp_packed_qkv_views` of a packed leaf;
    ``dq`` / ``dk`` / ``dv`` are read back from the leaf's gradient, which a hook also captures as it arrives."""
    H_qk, HV = ops["q"].shape[-2], ops["v"].shape[-2]
    T = ops["q"].shape[0]
    qkv = torch.cat(
        [ops[n].reshape(T, -1) for n in ("q", "k", "v")], dim=-1
    ).contiguous()
    if batched:
        qkv = qkv.unsqueeze(0)
    qkv.requires_grad_(True)
    g, beta = (ops[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    seen: list[torch.Tensor] = []
    qkv.register_hook(lambda grad: seen.append(grad))
    o = {"backend": "fake", **opts}

    def fn(qkv, g, beta):
        q, k, v = kcp.cake_gdn_det_train_kcp_packed_qkv_views(qkv, H_qk=H_qk, HV=HV)
        gg, bb = (g.unsqueeze(0), beta.unsqueeze(0)) if batched else (g, beta)
        return kcp.cake_gdn_det_kcp(q, k, v, gg, bb, ctx, **o)

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, qkv, g, beta, use_reentrant=False)
    else:
        out = fn(qkv, g, beta)
    dout = ops["dout"].unsqueeze(0) if batched else ops["dout"]
    out.backward(dout)
    grad = qkv.grad.reshape(T, -1)
    a, b = H_qk * 128, 2 * H_qk * 128
    res = {
        "out": out.detach().reshape(ops["dout"].shape),
        "dq": grad[:, :a].reshape(T, H_qk, 128),
        "dk": grad[:, a:b].reshape(T, H_qk, 128),
        "dv": grad[:, b:].reshape(T, HV, 128),
        "dg": g.grad,
        "dbeta": beta.grad,
    }
    return res, seen


@pytest.mark.parametrize(
    "batched,ckpt,group_reduce,layout",
    [
        (False, False, "fused", "tma"),
        (True, False, "fused", "tma"),
        (False, True, "fused", "tma"),
        (False, False, "host", "tma"),
        (False, False, "fused", "copy"),
    ],
)
def test_packed_qkv_gradients_are_bitwise_and_need_no_concatenation(
    batched, ckpt, group_reduce, layout
) -> None:
    """q / k / v as views of one packed leaf: out and the five gradients are bitwise those of separate contiguous
    leaves; with the views passed through (``layout="tma"``) the backward writes one packed dQKV row
    (``grad_layout="packed"``) and the split returns it as the leaf's gradient (a view, no concatenation); with
    ``layout="copy"`` the operands are copied, the gradients are separate and the split concatenates them."""
    kcp, ctx, ops = _ckpt_setup()
    base = dict(group_reduce=group_reduce, layout=layout)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    before = dict(kcp.PACKED_SPLIT_STATS)
    got, seen = _packed_autograd(kcp, ctx, ops, batched=batched, ckpt=ckpt, **base)
    what = f"batched={batched} ckpt={ckpt} {base}"
    _assert_same(got, direct, what)
    assert direct["dq"].is_contiguous(), what
    packed = layout == "tma"
    assert kcp.LAST_CALL["bwd"]["grad_layout"] == (
        "packed" if packed else "separate"
    ), what
    delta = {k: kcp.PACKED_SPLIT_STATS[k] - before[k] for k in before}
    assert delta == (
        {"aliased": 1, "concatenated": 0}
        if packed
        else {"aliased": 0, "concatenated": 1}
    ), (
        what,
        delta,
    )
    assert len(seen) == 1, what
    assert (seen[0]._base is not None) == packed, what


def test_packed_qkv_split_concatenates_any_other_gradient() -> None:
    """Gradients that are not one packed row (an unused ``k``: zeros; separate tensors) are concatenated: the leaf's
    gradient is the plain split's."""
    kcp = _package("cake_kcp")
    H_qk, HV, T = 2, 6, 37
    gen = torch.Generator(device="cpu").manual_seed(7817)
    x = torch.randn(T, (2 * H_qk + HV) * 128, generator=gen).to(torch.bfloat16)
    a = x.clone().requires_grad_(True)
    b = x.clone().requires_grad_(True)
    before = dict(kcp.PACKED_SPLIT_STATS)
    q, k, v = kcp.cake_gdn_det_train_kcp_packed_qkv_views(a, H_qk=H_qk, HV=HV)
    assert (
        q.shape == (T, H_qk, 128)
        and k.shape == (T, H_qk, 128)
        and v.shape == (T, HV, 128)
    )
    assert q.data_ptr() == a.data_ptr() and q.stride() == (a.shape[-1], 128, 1)
    ((q.float() * 3).sum() + (v.float() ** 2).sum()).backward()
    q2, _, v2 = b.split((H_qk * 128, H_qk * 128, HV * 128), dim=-1)
    ((q2.float() * 3).sum() + (v2.float() ** 2).sum()).backward()
    assert torch.equal(a.grad, b.grad)
    assert kcp.PACKED_SPLIT_STATS["concatenated"] - before["concatenated"] == 1
    assert kcp.PACKED_SPLIT_STATS["aliased"] == before["aliased"]


def test_packed_qkv_views_det_kcp_rejects_a_bad_row() -> None:
    kcp = _package("cake_kcp")
    x = torch.zeros(5, 10 * 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected"):
        kcp.cake_gdn_det_train_kcp_packed_qkv_views(
            x, H_qk=2, HV=4
        )  # 8 * 128 channels expected
    with pytest.raises(ValueError, match="dense"):
        kcp.cake_gdn_det_train_kcp_packed_qkv_views(
            x.t().contiguous().t(), H_qk=2, HV=6
        )
    with pytest.raises(ValueError):
        kcp.cake_gdn_det_train_kcp_packed_qkv_views(
            torch.zeros(5, 9 * 128, dtype=torch.bfloat16), H_qk=2, HV=5
        )


# --------------------------------------------------------------------------- the output walk kcp_fwdh_out (fused_o)
# A forward whose h / v_new tapes have no reader runs kcp_fwdh_out (#2 with #5 folded into each chunk step, no tape
# written) on every tape-walk launch list.  On the fake backend these are host-plumbing checks (launch lists, output
# coverage, arena); the kernels' bitwise check is the GPU suites' (fused_o 1 vs 0 in one build).
FUSED_TO_SPLIT = {
    "kcp_fwdh_out": "kcp_fwdh_tape",
    "kcp_fwdh_out_vh64": "kcp_fwdh_tape_vh64",
}


# The fused output walk on the W = 8 row (the multi-rank harness of the non-deterministic twin's tests).
LEAVES = ("q", "k", "v", "g", "beta")
EXPLICIT_GRADS = ("dq", "dk", "dv", "dbeta", "dg")  # the backward entries' output order

# The exchange (the owner chains of the families fwd, adj_b1, adj_c1) exists only with W > 1: the host orchestration of
# W ranks runs here in one process on W threads, on the fake backend, with an in-process stand-in of the one collective
# the a2a exchange issues (all_to_all_single with explicit split sizes, matched across the ranks by issue order).  The
# stand-in moves bytes only; every rank's arithmetic is the package's own.

MR_CU = [
    0,
    1201,
]  # one long document: 2 W ragged pieces per document under zigzag_per_document
MR_W = 8
MR_HEADS = (2, 8)  # one owned value head per rank (Hg = 1)
# Mild decays for the multi-rank row: ``_inputs``' gate (-exp(A) softplus(x), mean about -0.7 .. -11 per token) makes
# the transfer of a ~75-token fragment underflow to ~0.  At 1/200 of it a fragment keeps exp(-0.3 .. -4) of its incoming
# state: the owner chains carry state across all W ranks.
MR_G_SCALE = 1.0 / 200.0


class _ThreadWork:
    """The pending receive of one rank in one collective: ``wait()`` copies the peers' chunks once all have issued."""

    def __init__(self, comm, seq: int, rank: int, output, recv_splits) -> None:
        self.comm, self.seq, self.rank = comm, seq, rank
        self.output, self.recv_splits = output, recv_splits

    def wait(self) -> None:
        comm = self.comm
        with comm.cond:
            done = comm.cond.wait_for(
                lambda: comm.failure is not None
                or len(comm.deposits.get(self.seq, {})) == comm.world,
                timeout=comm.timeout,
            )
            if comm.failure is not None:
                raise RuntimeError(
                    f"rank {self.rank}: another rank failed"
                ) from comm.failure
            if not done:
                raise TimeoutError(
                    f"rank {self.rank}: collective {self.seq} incomplete after {comm.timeout} s"
                )
            got = [comm.deposits[self.seq][src][self.rank] for src in range(comm.world)]
            comm.reads[self.seq] = comm.reads.get(self.seq, 0) + 1
            if comm.reads[self.seq] == comm.world:
                del comm.deposits[self.seq]
        offset = 0
        for src, (n, chunk) in enumerate(zip(self.recv_splits, got, strict=True)):
            assert chunk.shape[0] == n, (self.rank, src, n, tuple(chunk.shape))
            self.output.narrow(0, offset, n).copy_(chunk)
            offset += n


class _ThreadDist:
    """The ``torch.distributed`` surface the host module reads during an a2a call (``is_available``,
    ``is_initialized``, ``all_to_all_single``) for ``world`` ranks on as many threads of this process; the rank of a
    call is its thread's.  ``exchange="ring"`` and ``debug_sequence`` are not emulated."""

    def __init__(self, world: int, timeout: float = 900.0) -> None:
        self.world, self.timeout = int(world), float(timeout)
        self.cond = threading.Condition()
        self.deposits: dict[int, dict[int, list[torch.Tensor]]] = {}
        self.reads: dict[int, int] = {}
        self.issued = [0] * self.world
        self.local = threading.local()
        self.failure: BaseException | None = None

    @staticmethod
    def is_available() -> bool:
        return True

    @staticmethod
    def is_initialized() -> bool:
        return True

    def all_to_all_single(
        self,
        output,
        input,  # noqa: A002 - torch.distributed's keyword
        output_split_sizes=None,
        input_split_sizes=None,
        group=None,
        async_op=False,
    ):
        rank = self.local.rank
        seq = self.issued[rank]
        self.issued[rank] += 1
        send = [int(x) for x in input_split_sizes]
        recv = [int(x) for x in output_split_sizes]
        assert len(send) == len(recv) == self.world, (rank, send, recv)
        chunks = [c.clone() for c in torch.split(input, send)]
        with self.cond:
            self.deposits.setdefault(seq, {})[rank] = chunks
            self.cond.notify_all()
        work = _ThreadWork(self, seq, rank, output, recv)
        if async_op:
            return work
        work.wait()
        return None

    def fail(self, exc: BaseException) -> None:
        with self.cond:
            if self.failure is None:
                self.failure = exc
            self.cond.notify_all()


def _run_threaded(kcp, world: int, fn, monkeypatch) -> list:
    """``fn(rank, comm)`` on ``world`` threads with ``cake_kcp.dist`` replaced by a :class:`_ThreadDist`; the per-rank
    results, or the first rank's failure."""
    comm = _ThreadDist(world)
    monkeypatch.setattr(kcp, "dist", comm)
    results: list = [None] * world
    errors: list = []

    def worker(rank: int) -> None:
        comm.local.rank = rank
        try:
            results[rank] = fn(rank, comm)
        except BaseException as exc:  # noqa: BLE001 - re-raised in the main thread
            errors.append((rank, exc))
            comm.fail(exc)

    threads = [
        threading.Thread(target=worker, args=(r,), name=f"kcp-rank-{r}")
        for r in range(world)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(comm.timeout + 60)
    assert not any(t.is_alive() for t in threads), "a rank thread did not finish"
    if errors:
        rank, exc = errors[0]
        raise AssertionError(f"rank {rank} failed: {exc!r}") from exc
    return results


def _call_recorder(kcp, comm_of=lambda: None):
    """A recording fake backend: per rank (the calling thread's, 0 without threads) the :class:`CakeGDNDetKcpCall` objects in
    first-launch order and, per call, the program names it launched."""
    calls: dict[int, list] = {}
    launched: dict[int, list[str]] = {}
    lock = threading.Lock()

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            comm = comm_of()
            rank = getattr(comm.local, "rank", 0) if comm is not None else 0
            with lock:
                seen = calls.setdefault(rank, [])
                if all(c is not call for c in seen):
                    seen.append(call)
                launched.setdefault(id(call), []).append(name)
            super().run(name, args)

    return calls, launched, Recording()


_FULL_NO_ST = {"save_mode": "full", "save_transfer": False}
_L2N = {"use_qk_l2norm_in_kernel": True}
# the layout terms of each runtime pool a call sizes (the c1 send pool holds the M_f^T tiles at P > 1)
_POOL_TERMS = {
    "c1_send": ("c1_send", "c1_send_mf"),
    "c1_recv": ("c1_recv",),
    "c2_send": ("c2_send",),
    "stage_b": ("staging",),
    "stage_m": ("staging",),
    "carry": ("carry",),
    "eye": ("eye",),
    "b1_send": ("b1_send",),
    "b1_recv": ("b1_recv",),
    "b2_send": ("b2_send",),
    "b2_recv": ("b2_recv",),
}


def _pool_peak_workspace(kcp):
    """A :class:`_PeakWorkspace` that also records the largest size of every pool (``pool_max``), the bytes it
    binds when the adjoint tape walk first binds ``dh_tape`` (``walk_bytes``: the backward's walk phase) and the
    bytes it binds just before ``c1_send`` is first released (``c1_release_bytes``: in a replay backward the replayed
    forward's moment before its c1 release, the backward's ``replay_forward`` phase)."""

    class Pools(type(_PeakWorkspace(kcp))):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.pool_max: dict[str, int] = {}
            self.walk_bytes: int | None = None
            self.c1_release_bytes: int | None = None

        def release_pool(self, name):
            if name == "c1_send" and self.c1_release_bytes is None:
                self.c1_release_bytes = self.bytes
            return super().release_pool(name)

        def _upd(self):
            super()._upd()
            for name, b in self.pool_bytes.items():
                self.pool_max[name] = max(self.pool_max.get(name, 0), b)

        def tape(self, name):
            first = name == "dh_tape" and name not in self._tapes
            t = super().tape(name)
            if first:
                self.walk_bytes = self.bytes
            return t

    return Pools()


def _recorded_pool_terms(pool_max: dict[str, int]) -> dict[str, int]:
    """The recorded pool sizes summed per layout term group (``c1_send`` + ``c1_send_mf``; the two staging planes)."""
    out: dict[str, int] = {}
    for name, terms in _POOL_TERMS.items():
        out["+".join(terms)] = out.get("+".join(terms), 0) + pool_max.get(name, 0)
    return out


def _layout_pool_terms(lay: dict[str, int], phase: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for terms in dict.fromkeys(_POOL_TERMS.values()):
        out["+".join(terms)] = sum(lay.get(f"{phase}.{t}", 0) for t in terms)
    return out


_ARENA_ROWS = (
    [
        (2, P, opts, 1)
        for P in (1, 2)
        for opts in ({}, {"save_transfer": False}, _FULL_NO_ST)
    ]
    + [(2, 2, {}, 0), (2, 2, _FULL_NO_ST, 0)]
    # the replay boundary on one rank and on two ranks with a crossing fragment
    + [
        (W, P, {"boundary_mode": "replay", **o}, 1)
        for W, P, o in (
            (1, 2, {}),
            (1, 2, {"save_mode": "full"}),
            (2, 1, {}),
            (2, 2, {}),
            (2, 2, {"save_mode": "full"}),
        )
    ]
    # graph="replay": the first call of a key runs eagerly and releases nothing
    + [
        (2, 2, {"graph": "replay", **o}, 1)
        for o in ({}, _FULL_NO_ST, {"boundary_mode": "replay"})
    ]
    # calls without a saved set (the backward: the saved=None replay) and the in-kernel norm (the arena r of a
    # forward without a saved set)
    + [
        (2, 2, {"save_for_backward": False}, 1),
        (2, 2, {"save_for_backward": False, "save_mode": "full"}, 1),
        (1, 2, {"save_for_backward": False, **_L2N}, 1),
        (1, 2, {"boundary_mode": "replay", **_L2N}, 1),
        (1, 2, _L2N, 1),
    ]
    # graph="replay" with the norm: the arena r held through the forward, the norm's rows in the graph backward
    + [
        (1, 2, {"graph": "replay", **o, **_L2N}, 1)
        for o in ({}, {"boundary_mode": "replay"}, {"save_for_backward": False})
    ]
    # four ranks: the middle ranks' fragments cross in and out (carry: max(n_in, n_out) slots, the b1 split forms)
    + [(4, 2, o, fw) for o in ({}, _FULL_NO_ST) for fw in (1, 0)]
)
# a short pack (rank 0: a 2-chunk document and one chunk crossing out, NC_pad = 4): there the split forward peaks at
# #1 (the c1 send pool outweighs h / v_new) and the replay boundary's backward at its replayed forward
_SHORT_CU = (0, 100, 228)


@pytest.mark.parametrize(
    "W,segments,opts,fused_walks,cu",
    [(*row, CKPT_CU) for row in _ARENA_ROWS]
    + [
        (2, 2, {"boundary_mode": "replay"}, 1, _SHORT_CU),
        (2, 2, {"save_for_backward": False, "save_mode": "full"}, 1, _SHORT_CU),
        # the replay backward after a fused forward (light, saved boundary, no saved set): its replayed forward
        # binds #1's eight tapes only before the c1 release (measured at that release; the backward peaks at the
        # adjoint exchange here)
        (2, 2, {"save_for_backward": False}, 1, _SHORT_CU),
    ],
)
def test_workspace_layout_follows_the_arena_of_every_rank(
    monkeypatch, W, segments, opts, fused_walks, cu
) -> None:
    """``cake_gdn_det_train_kcp_workspace_layout(with_conv=False)`` is the most a caller-held workspace binds on each rank in the
    forward on a fresh arena and in the backward that follows on the same workspace (the explicit entries; the
    backward's replay form), every pool term is the size the call gives that pool, and ``cake_gdn_det_train_kcp_saved_bytes`` is
    what the saved set holds: on two ranks with a crossing fragment -- at ``P = 2`` the c1 send pool's ``M_f^T`` tiles
    and, with the fused walks, the carry slots of the crossing fragments; the b1 pools at their larger split form; the
    forward exchange pools the backward sizes again; the eager forward's fresh c2 send plane and its c1 send pool
    released before the tape walk -- in the light form with and without ``save_transfer`` and the full form,
    ``fused_walks`` 1 and 0, on two ranks and on four (whose middle ranks hold fragments that cross in and out: the
    carry plane's ``max(n_in, n_out)`` slots); the replay boundary on one and two ranks (``save_mode="full"`` keeps
    no tape and binds the light form's arena; ``reuse_recompute_tapes=True`` reports the same figures);
    ``graph="replay"`` (the first call of a key: no buffer is released), also with the norm; calls without a saved
    set (``save_for_backward=False``) and the ``saved=None`` replay backward after them; the in-kernel norm with and
    without a saved set; a short pack where the split forward peaks at #1 and the replay boundary's backward at its
    replayed forward.  On every rank of a group that exchanges, a replay backward under ``graph="off"`` binds
    ``bwd.phase.replay_forward`` just before its replayed forward releases the c1 send pool (after a fused forward
    without a saved set: #1's eight tapes, no ``h`` / ``v_new``)."""
    kcp = _package("cake_kcp")
    H_qk, HV = CKPT_HEADS
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(list(cu), dtype=torch.int64), W, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(W)
    ]
    assert W == 1 or sum(c.n_in for c in ctxs) == sum(c.n_out for c in ctxs) > 0
    # four ranks: some rank's fragments cross in and out, so max(n_in, n_out) and n_in + n_out differ there
    assert W < 4 or any(min(c.n_in, c.n_out) > 0 for c in ctxs)
    ops = _inputs(list(cu), H_qk, HV)
    knobs = {"test_knobs": {"fused_walks": 0}} if not fused_walks else {}
    call_opts = {k: v for k, v in opts.items() if k != "save_for_backward"}
    with_set = opts.get("save_for_backward", True)
    wss = [_pool_peak_workspace(kcp) for _ in range(W)]
    got: dict[int, dict] = {}

    def rank_fn(r, comm):
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        kw = dict(
            backend="fake",
            segments=segments,
            workspace=wss[r],
            **call_opts,
            **knobs,
        )
        _, saved = kcp.cake_gdn_det_train_kcp_forward(
            *args, ctxs[r], save_for_backward=with_set, **kw
        )
        ws = wss[r]
        got[r] = {
            "forward": ws.peak,
            "fwd_pools": dict(ws.pool_max),
            "saved": 0
            if saved is None
            else sum(t.numel() * t.element_size() for t in saved.tensors()),
        }
        # without a saved set (saved=None) the backward is the replay
        ws.peak, ws.pool_max, ws.walk_bytes, ws.c1_release_bytes = 0, {}, None, None
        kcp.cake_gdn_det_train_kcp_backward(
            *args, ctxs[r], loc["dout"], saved=saved, **kw
        )
        got[r].update(
            backward=ws.peak,
            bwd_pools=dict(ws.pool_max),
            walk=ws.walk_bytes,
            c1_release=ws.c1_release_bytes,
            end=ws.bytes,
        )

    try:
        if W == 1:
            rank_fn(0, None)
        else:
            _run_threaded(kcp, W, rank_fn, monkeypatch)
    finally:  # the graph entries of these workspaces (pinned by the cache) go with the test
        for key in [
            k for k, e in kcp._GRAPHS.items() if any(e.pins[1] is ws for ws in wss)
        ]:
            del kcp._GRAPHS[key]
    q_opts = {
        k: v
        for k, v in opts.items()
        if k not in ("graph", "save_for_backward", "use_qk_l2norm_in_kernel")
    }
    # some fragment crosses: every rank runs the c1 exchange
    exchanges = sum(c.n_out for c in ctxs) > 0
    replay_c1 = 0
    for r, c in enumerate(ctxs):
        q = dict(H_qk=H_qk, HV=HV, P=segments, with_conv=False, **q_opts)
        l2n = dict(use_qk_l2norm_in_kernel=opts.get("use_qk_l2norm_in_kernel", False))
        lay = gb.cake_gdn_det_train_kcp_workspace_layout(
            c,
            reuse_recompute_tapes=False,
            fused_walks=fused_walks,
            graph=opts.get("graph", "off"),
            save_for_backward=with_set,
            **q,
            **l2n,
        )
        assert got[r]["forward"] == lay["forward"], (r, got[r]["forward"], lay)
        rec, want = (
            _recorded_pool_terms(got[r]["fwd_pools"]),
            _layout_pool_terms(lay, "fwd"),
        )
        assert {k: v for k, v in rec.items() if not k.startswith("b")} == {
            k: v for k, v in want.items() if not k.startswith("b")
        }, (r, "fwd")
        assert got[r]["backward"] == lay["backward"], (r, got[r]["backward"], lay)
        assert _recorded_pool_terms(got[r]["bwd_pools"]) == _layout_pool_terms(
            lay, "bwd"
        ), (r, "bwd")
        if opts.get("graph", "off") == "off":
            # the adjoint tape walk's moment and the end state (the norm's rows go after #8)
            norm = sum(lay.get(f"bwd.{k}", 0) for k in ("qn", "kn", "qk_rstd"))
            assert got[r]["walk"] == lay["bwd.phase.walk"], (r, got[r]["walk"])
            assert got[r]["end"] == lay["bwd.phase.end"] - norm, (r, "end")
            if exchanges and "bwd.phase.replay_forward" in lay:
                # the replayed forward's moment just before its c1 send pool goes (every rank of the group)
                assert got[r]["c1_release"] == lay["bwd.phase.replay_forward"], (
                    r,
                    got[r]["c1_release"],
                )
                replay_c1 += 1
        if with_set:
            assert got[r]["saved"] == gb.cake_gdn_det_train_kcp_saved_bytes(
                c, **q, **l2n
            ), (
                r,
                got[r],
            )
        else:
            assert got[r]["saved"] == 0
        if segments > 1:
            assert lay["fwd.c1_send_mf"] == c.n_out * HV * gp.STATE_TILE_BYTES
        if opts.get("boundary_mode") == "replay":
            assert got[r]["saved"] == 0
            light = gb.cake_gdn_det_train_kcp_workspace_layout(
                c,
                reuse_recompute_tapes=False,
                H_qk=H_qk,
                HV=HV,
                P=segments,
                with_conv=False,
                boundary_mode="replay",
                graph=opts.get("graph", "off"),
                **l2n,
            )
            assert lay == light  # the replay boundary: "full" is the light form's
            if opts.get("graph", "off") == "off":
                # the runtime never hands tapes over under a replay boundary: True reports the replay form
                assert lay == gb.cake_gdn_det_train_kcp_workspace_layout(
                    c,
                    reuse_recompute_tapes=True,
                    H_qk=H_qk,
                    HV=HV,
                    P=segments,
                    with_conv=False,
                    **q_opts,
                    **l2n,
                )
    # every replay backward (no saved set kept) of an exchanging group under graph="off" measured that moment
    replays = not with_set or opts.get("boundary_mode") == "replay"
    graph_off = opts.get("graph", "off") == "off"
    assert replay_c1 == (W if exchanges and replays and graph_off else 0), replay_c1


@pytest.mark.parametrize(
    "opts",
    [{}, _FULL_NO_ST, {"boundary_mode": "replay"}, {"save_for_backward": False}],
    ids=["light", "full", "replay", "no_saved_set"],
)
def test_workspace_reused_across_steps_binds_the_graph_replay_figure(
    monkeypatch, opts
) -> None:
    """Three forward + backward steps of the explicit entries on one caller-held workspace per rank (two ranks, a
    crossing fragment, ``P = 2``): the first step binds the ``graph="off"`` figures; the arena keeps the tapes the
    backward bound, so every later backward binds the ``graph="replay"`` figure exactly and every later forward more
    than the first forward and less than that figure.  A call without a saved set (light, saved boundary) is the
    exception: its fused forward releases ``h`` / ``v_new``, which the replayed forward binds again only after its
    c1 release, so its later backwards bind that figure less the c1 send pool and its ``M_f^T`` tiles (less on the
    rank whose fragment crosses out) and at least the first backward; its later forwards peak when they size their
    pools, the previous backward's ``h`` / ``v_new`` still bound: that figure less the backward-only pools
    (``c2_send``, ``b1_send`` / ``b1_recv``, ``b2_send`` / ``b2_recv``).  Which of the two is larger is a relation of
    pool sizes, not a rule."""
    kcp = _package("cake_kcp")
    H_qk, HV = CKPT_HEADS
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 2, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(2)
    ]
    ops = _inputs(CKPT_CU, H_qk, HV)
    wss = [_PeakWorkspace(kcp) for _ in range(2)]
    steps: dict[int, list[tuple[int, int]]] = {0: [], 1: []}
    call = {k: v for k, v in opts.items() if k != "save_for_backward"}
    with_set = opts.get("save_for_backward", True)

    def rank_fn(r, comm):
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        kw = dict(backend="fake", segments=2, workspace=wss[r], **call)
        for _ in range(3):
            wss[r].peak = 0
            _, saved = kcp.cake_gdn_det_train_kcp_forward(
                *args, ctxs[r], save_for_backward=with_set, **kw
            )
            fwd = wss[r].peak
            wss[r].peak = 0
            kcp.cake_gdn_det_train_kcp_backward(
                *args, ctxs[r], loc["dout"], saved=saved, **kw
            )
            steps[r].append((fwd, wss[r].peak))

    _run_threaded(kcp, 2, rank_fn, monkeypatch)
    for r, c in enumerate(ctxs):
        q = dict(H_qk=H_qk, HV=HV, P=2, with_conv=False, reuse_recompute_tapes=False)
        eager = gb.cake_gdn_det_train_kcp_workspace_layout(c, **q, **opts)
        graph = gb.cake_gdn_det_train_kcp_workspace_layout(
            c, graph="replay", **q, **opts
        )
        assert steps[r][0] == (eager["forward"], eager["backward"]), (r, steps[r])
        assert graph["backward"] > eager["backward"], r
        # without a saved set: the c1 send pool the replayed forward releases before it binds h / v_new, and the
        # pools only a backward sizes (the forward's ensure skips them)
        c1 = graph["bwd.c1_send"] + graph["bwd.c1_send_mf"] if not with_set else 0
        bwd_only = sum(
            graph[f"bwd.{k}"]
            for k in ("c2_send", "b1_send", "b1_recv", "b2_send", "b2_recv")
        )
        assert (c1 > 0) == (not with_set and c.n_out > 0), (r, c1)
        for fwd, bwd in steps[r][1:]:
            if with_set:
                assert bwd == graph["backward"] == graph["workspace"], (r, steps[r])
                assert eager["forward"] < fwd < bwd, (r, steps[r])
            else:
                # the replayed forward never holds the c1 send pool and h / v_new together
                assert bwd == graph["backward"] - c1, (r, steps[r], graph)
                assert bwd >= eager["backward"], (r, steps[r])
                # the forward peaks at its ensure, the previous backward's h / v_new still bound
                assert fwd == graph["backward"] - bwd_only, (r, steps[r], graph)
                assert fwd > eager["forward"], (r, steps[r])


def test_workspace_layout_option_checks() -> None:
    """The workspace queries refuse a ``graph`` value outside ``("off", "replay")``, the kept-tape form under
    ``graph="replay"`` (as the call's options do; the default there is the replay form) and a non-bool
    ``save_for_backward``.  ``save_for_backward=False`` with a backward reports the replay a ``saved=None`` backward
    runs: the replay boundary's backward terms, in either ``save_mode`` and ``reuse_recompute_tapes`` form, except
    the replayed forward after the fused (light, saved boundary) forward, which finds no ``h`` / ``v_new``; the
    forward is the call's."""
    _, ctx, _ = _ckpt_setup()
    q = dict(H_qk=2, HV=6)
    assert gb.GRAPH_MODES == _package("cake_options").GRAPH_MODES
    with pytest.raises(ValueError, match="graph"):
        gb.cake_gdn_det_train_kcp_workspace_layout(ctx, graph="on", **q)
    for reuse in (True, "auto"):
        with pytest.raises(ValueError, match="needs graph='off'"):
            gb.cake_gdn_det_train_kcp_workspace_bytes(
                ctx, graph="replay", reuse_recompute_tapes=reuse, **q
            )
    assert gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, graph="replay", **q
    ) == gb.cake_gdn_det_train_kcp_workspace_layout(
        ctx, graph="replay", reuse_recompute_tapes=False, **q
    )
    for save_mode, reuse in itertools.product(("light", "full"), (True, False)):
        lay = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx,
            save_for_backward=False,
            save_mode=save_mode,
            reuse_recompute_tapes=reuse,
            P=2,
            **q,
        )
        rep = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, boundary_mode="replay", P=2, **q
        )
        only_fwd = gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, save_for_backward=False, save_mode=save_mode, backward=False, P=2, **q
        )
        # after the fused forward of a light saved boundary the replayed forward finds no h / v_new before its c1
        # release; every other backward term is the replay boundary's
        fused = "fwd.h_tape" not in only_fwd  # the forward's own fused flag
        assert fused == (save_mode == "light")
        h_v = rep["bwd.h_tape"] + rep["bwd.v_new"] if fused else 0
        assert lay["bwd.phase.replay_forward"] == rep["bwd.phase.replay_forward"] - h_v
        bwd_terms = {
            k: v
            for k, v in lay.items()
            if k.startswith("bwd.") and k != "bwd.phase.replay_forward"
        }
        assert bwd_terms == {
            k: v
            for k, v in rep.items()
            if k.startswith("bwd.") and k != "bwd.phase.replay_forward"
        }
        assert lay["bwd.kept_tapes"] == 0
        phases = [v for k, v in lay.items() if k.startswith("bwd.phase.")]
        assert lay["backward"] == max(phases) == lay["backward_transient"]
        assert lay["backward"] <= rep["backward"]
        assert lay["forward"] == only_fwd["forward"] and only_fwd["backward"] == 0
    with pytest.raises(TypeError, match="save_for_backward"):
        gb.cake_gdn_det_train_kcp_workspace_layout(
            ctx, save_for_backward=0, backward=False, **q
        )


def _mr_setup():
    kcp = _package("cake_kcp")
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(MR_CU, dtype=torch.int64), MR_W, "zigzag_per_document"
    )
    return kcp, part, _mr_inputs()


def _mr_inputs() -> dict[str, torch.Tensor]:
    ops = dict(_inputs(MR_CU, *MR_HEADS))
    ops["g"] = (ops["g"] * MR_G_SCALE).contiguous()
    return ops


def _mr_run(kcp, part, ops, monkeypatch, *, sm_count: int = SM_COUNT, knobs=None):
    """Forward + saved-set backward of every rank of the multi-rank row on the shipped options (plus ``knobs`` on both
    entries): the global-order results, the per-rank calls (forward, backward) and the launched programs per call."""
    knobs = dict(knobs or {})
    holder: dict = {}
    calls, launched, rec = _call_recorder(kcp, lambda: holder.get("comm"))
    # group None: the stand-in collective ignores the group (the rank is the calling thread's)
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=sm_count
        )
        for r in range(MR_W)
    ]

    def rank_fn(r, comm):
        holder["comm"] = comm
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        out, saved = kcp.cake_gdn_det_train_kcp_forward(
            *args, ctxs[r], backend="fake", **knobs
        )
        grads = kcp.cake_gdn_det_train_kcp_backward(
            *args,
            ctxs[r],
            loc["dout"],
            saved=saved,
            backend="fake",
            **knobs,
        )
        return out, grads

    previous = kcp.set_backend("fake", rec)
    try:
        per_rank = _run_threaded(kcp, MR_W, rank_fn, monkeypatch)
    finally:
        kcp.set_backend("fake", previous)
        monkeypatch.undo()
    res = {"out": fake.gather_global(part, [o for o, _ in per_rank])}
    for i, n in enumerate(EXPLICIT_GRADS):
        res[n] = fake.gather_global(part, [g[i] for _, g in per_rank])
    return res, ctxs, calls, launched


def _nan_output(kcp, monkeypatch) -> None:
    """``CakeGDNDetKcpCall.output`` pre-fills a new output with NaN, so a row that neither an output walk nor #5 writes shows."""
    original = kcp.CakeGDNDetKcpCall.output

    def output(self):
        fresh = self.out is None
        t = original(self)
        if fresh:
            t.fill_(float("nan"))
        return t

    monkeypatch.setattr(kcp.CakeGDNDetKcpCall, "output", output)


def _split_view(call) -> tuple[list, list]:
    """A forward call's program sequence and launch records with every output-walk launch named as the tape walk it
    replaces and #5 left out: a fused forward's view equals its split twin's."""
    seq = [FUSED_TO_SPLIT.get(n, n) for n in call.sequence if n != "fwd_o"]
    recs = [
        (FUSED_TO_SPLIT.get(r, r), tuple(g)) for r, g in call.records if r != "fwd_o"
    ]
    return seq, recs


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"use_qk_l2norm_in_kernel": True},
        {"test_knobs": {"schedule": "wait_first"}},
        {"test_knobs": {"fused_walks": 0}},
        {"test_knobs": {"fwdh_vh": 64}},
        {"test_knobs": {"fwdh_vh": 128}},
        {"save_transfer": False},
    ],
    ids=(
        "shipped",
        "l2n",
        "wait_first",
        "fused_walks0",
        "vh64",
        "vh128",
        "no_save_transfer",
    ),
)
def test_fused_output_walk_reproduces_the_split_forward(monkeypatch, knobs) -> None:
    """The W = 8 row with the shipped fused forward and with test knob ``fused_o=0``: out (pre-filled with NaN when it
    is allocated) and the five gradients equal bitwise; each rank's fused forward runs ``kcp_fwdh_out`` on exactly the
    launch lists (records modulo the program, grids) the split forward runs ``kcp_fwdh_tape`` on, and no #5; its
    ``fwd_o`` evidence is ``"fused"``, and its arena ends without ``h_tape`` / ``v_new``.  The backward calls are
    the same launches either way."""
    kcp, part, ops = _mr_setup()
    runs = {}
    for fused in (1, 0):
        k = {**knobs, "test_knobs": {**knobs.get("test_knobs", {}), "fused_o": fused}}
        _nan_output(kcp, monkeypatch)
        runs[fused] = _mr_run(kcp, part, ops, monkeypatch, knobs=k)
    (res1, _, calls1, _), (res0, _, calls0, _) = runs[1], runs[0]
    assert not torch.isnan(res1["out"].float()).any(), (
        "a row of the fused output was never written"
    )
    for n in ("out",) + EXPLICIT_GRADS:
        assert torch.equal(res1[n], res0[n]), n
    for r in range(MR_W):
        (f1,) = [c for c in calls1[r] if not c.backward]
        (f0,) = [c for c in calls0[r] if not c.backward]
        assert f1.scratch["fwd_o"] == "fused" and f0.scratch["fwd_o"] == "launched", r
        assert "fwd_o" not in f1.sequence and "kcp_fwdh_tape" not in f1.sequence, r
        assert "kcp_fwdh_out" in f1.sequence and "fwd_o" in f0.sequence, r
        assert _split_view(f1) == _split_view(f0), r
        assert not ({"h_tape", "v_new"} & set(f1.ws._tapes)), (r, sorted(f1.ws._tapes))
        assert {"h_tape", "v_new"} <= set(f0.ws._tapes), (r, sorted(f0.ws._tapes))
        b1 = [(c.sequence, c.records) for c in calls1[r] if c.backward]
        b0 = [(c.sequence, c.records) for c in calls0[r] if c.backward]
        assert b1 == b0, r


@pytest.mark.parametrize("segments", (1, 2))
def test_fused_output_walk_under_checkpoints_and_handles(segments) -> None:
    """One rank (no exchange), P = 1 / 2: direct autograd, non-reentrant checkpoints of a region the GDN node closes and
    of one with a later saver under every ``reuse_recompute_tapes`` form (the mixed case: a fused first forward, a
    split hand-off recompute with ``True``), the recompute handle with a later saver (its recompute writes the output
    the saver reads: the ``S_in`` branch's tape walks run fused), ``save_mode="full"`` and the replay boundary: out
    and every gradient equal bitwise with the shipped options and with ``fused_o=0`` (the handle form: and the direct
    pass); the first forward of a light region is fused, a recompute is fused exactly when it launches #5 without a
    hand-off (the frame proof is read on a verified torch only), a hand-off or a full / replay-boundary forward never
    is, and no kept entry is left."""
    kcp, ctx, ops = _ckpt_setup()
    opt = _package("cake_options")
    base = {"segments": segments}
    forms = [
        ("direct", dict(ckpt=False, later_saver=False)),
        ("closing_auto", dict(ckpt=True, later_saver=False)),
        ("later_auto", dict(ckpt=True, later_saver=True)),
        ("later_true", dict(ckpt=True, later_saver=True, reuse_recompute_tapes=True)),
        ("later_false", dict(ckpt=True, later_saver=True, reuse_recompute_tapes=False)),
        (
            "closing_true",
            dict(ckpt=True, later_saver=False, reuse_recompute_tapes=True),
        ),
        ("full", dict(ckpt=True, later_saver=True, save_mode="full")),
        (
            "replay_boundary",
            dict(ckpt=False, later_saver=False, boundary_mode="replay"),
        ),
        ("handle_later", dict(ckpt=True, later_saver=True, handle=True)),
    ]
    for name, form in forms:
        got = {}
        for fused in (1, 0):
            run = dict(form)
            if run.pop("handle", False):
                run["handle"] = kcp.CakeGDNDetKcpRecomputeHandle()
            seen: list[str] = []
            previous = kcp._BACKENDS.get("fake")

            class Recording(kcp.FakeBackend):
                def run(self, prog, args, _seen=seen):
                    if not args["_call"].backward:
                        _seen.append(prog)
                    super().run(prog, args)

            kcp._BACKENDS["fake"] = Recording()
            try:
                res, scale_grad = _ckpt_autograd_region(
                    kcp, ctx, ops, **run, **base, test_knobs={"fused_o": fused}
                )
            finally:
                kcp._BACKENDS["fake"] = previous
            # the last forward's record: the recompute's under a checkpoint
            got[fused] = (res, scale_grad, seen, dict(kcp.LAST_CALL["fwd"]))
        (r1, s1, seen1, rec1), (r0, s0, seen0, rec0) = got[1], got[0]
        for n in ("out",) + CKPT_GRADS:
            assert torch.equal(r1[n], r0[n]), (name, n)
        assert (s1 is None and s0 is None) or torch.equal(s1, s0), name
        light = (
            form.get("save_mode", "light") == "light"
            and form.get("boundary_mode", "saved") == "saved"
        )
        assert ("kcp_fwdh_out" in seen1) == light, (name, seen1)
        assert "kcp_fwdh_out" not in seen0, name
        if light:
            # the first forward is fused (one #5 fewer); a recompute that launches #5 without a hand-off is fused too
            # (one more): it launches #5 unless the frame proof finds its output dead (a closing node, verified torch),
            # and it hands its tapes off with True, or with "auto" when the frame proof keeps them (_auto_hand_off)
            reuse = form.get("reuse_recompute_tapes", opt.REUSE_RECOMPUTE_TAPES_DEFAULT)
            later = form["later_saver"]
            hand_off = reuse is True or (
                reuse == "auto" and not later and _auto_hand_off(kcp)
            )
            recompute_fused = (
                form["ckpt"] and (later or not _verified_torch(kcp)) and not hand_off
            )
            assert seen0.count("fwd_o") - seen1.count("fwd_o") == (
                2 if recompute_fused else 1
            ), (name, recompute_fused, seen0, seen1)
            assert seen1.count("fwd_o") + seen1.count("kcp_fwdh_out") >= 1, name
        else:
            assert seen1 == seen0, name  # no forward of these forms fuses
        if "handle" in form:
            # the recompute read S_in from the handle and its walks wrote the output the later saver consumes
            direct, direct_scale = _ckpt_autograd_region(
                kcp, ctx, ops, ckpt=False, later_saver=True, save_transfer=False, **base
            )
            for n in ("out",) + CKPT_GRADS:
                assert torch.equal(r1[n], direct[n]), (name, n)
            assert torch.equal(s1, direct_scale), name
            assert rec1["recompute_handle"] == rec0["recompute_handle"] == "replay"
            assert (rec1["fwd_o"], rec0["fwd_o"]) == ("fused", "launched"), name
            assert rec1["P"] == segments, name
        assert kcp.kept_registry_size() == 0, name


@pytest.mark.parametrize("segments", (1, 2))
def test_recompute_handle_fused_recompute_under_a_later_saver(
    monkeypatch, segments
) -> None:
    """A recompute handle in a checkpointed region with a later saver (a scaling by a leaf), under every
    ``reuse_recompute_tapes`` form: torch's recompute launches the output (the frame proof sees the later saver), so
    without a hand-off (``False`` / ``"auto"``) it runs the output walk ``kcp_fwdh_out`` from the handle's ``S_in`` --
    at ``P = 2`` one fused walk over every live segment from the carry plane -- and the later saver reads that output;
    ``True`` hands the tapes off (a split recompute).  With every package allocation filled with NaN bytes, out, the
    five gradients and the scale's gradient are finite and the bits of the split form (``fused_o=0``) and of the direct
    pass."""
    kcp, ctx, ops = _ckpt_setup()
    _nan_filled_allocations(monkeypatch, kcp)
    direct, direct_scale = _ckpt_autograd_region(
        kcp,
        ctx,
        ops,
        ckpt=False,
        later_saver=True,
        save_transfer=False,
        segments=segments,
    )
    for reuse in (False, "auto", True):
        arms = {}
        for fused in (1, 0):
            res, scale_grad = _ckpt_autograd_region(
                kcp,
                ctx,
                ops,
                ckpt=True,
                later_saver=True,
                handle=kcp.CakeGDNDetKcpRecomputeHandle(),
                segments=segments,
                reuse_recompute_tapes=reuse,
                test_knobs={"fused_o": fused},
            )
            rec = kcp.LAST_CALL["fwd"]  # torch's recompute: the last forward
            assert rec["recompute_handle"] == "replay", (reuse, fused, rec)
            assert rec["P"] == segments, (reuse, fused, rec["P"])
            arms[fused] = (res, scale_grad, rec["fwd_o"])
        (r1, s1, o1), (r0, s0, o0) = arms[1], arms[0]
        assert o1 == ("launched" if reuse is True else "fused"), (reuse, o1)
        assert o0 == "launched", (reuse, o0)
        for n in ("out",) + CKPT_GRADS:
            assert torch.isfinite(r1[n].float()).all(), (reuse, n)
            assert torch.equal(r1[n], r0[n]), (reuse, n)
            assert torch.equal(r1[n], direct[n]), (reuse, n)
        assert torch.isfinite(s1.float()).all(), reuse
        assert torch.equal(s1, s0) and torch.equal(s1, direct_scale), reuse
        assert kcp.kept_registry_size() == 0, reuse


def test_fused_forward_releases_a_caller_arenas_tapes_and_kept_backward_leaves_the_fused_set(
    monkeypatch,
) -> None:
    """A caller-held workspace: a split forward (``fused_o=0``) leaves ``aqk`` / ``qscale`` / ``h_tape`` / ``v_new``
    bound; the next fused forward drops ``h_tape`` / ``v_new`` before #1 and leaves ``aqk`` / ``qscale``
    (:data:`FWD_TAPES_ARENA_AFTER_FUSED`); a kept-tape checkpoint backward re-adopts that fused set only, and the
    split one re-adopts the four.  Every arm's out and gradients are the same bits."""
    kcp, ctx, ops = _ckpt_setup()
    assert kcp.FWD_TAPES_ARENA_AFTER_FUSED == ("aqk", "qscale")
    ws = kcp.CakeGDNDetKcpWorkspace()
    leaves = [ops[n] for n in LEAVES]
    o0, _ = kcp.cake_gdn_det_train_kcp_forward(
        *leaves,
        ctx,
        workspace=ws,
        backend="fake",
        test_knobs={"fused_o": 0},
    )
    assert set(ws._tapes) & set(kcp.FWD_TAPES) == set(kcp.FWD_TAPES_ARENA_AFTER_FWD)
    o1, _ = kcp.cake_gdn_det_train_kcp_forward(
        *leaves, ctx, workspace=ws, backend="fake"
    )
    assert kcp.LAST_CALL["fwd"]["fwd_o"] == "fused"
    assert set(ws._tapes) & set(kcp.FWD_TAPES) == set(kcp.FWD_TAPES_ARENA_AFTER_FUSED)
    assert torch.equal(o0, o1)
    arms = {}
    for fused in (1, 0):
        wsk = kcp.CakeGDNDetKcpWorkspace()
        arms[fused] = _ckpt_autograd(
            kcp,
            ctx,
            ops,
            ckpt=True,
            workspace=wsk,
            reuse_recompute_tapes=True,
            test_knobs={"fused_o": fused},
        )
        want = (
            kcp.FWD_TAPES_ARENA_AFTER_FUSED if fused else kcp.FWD_TAPES_ARENA_AFTER_FWD
        )
        assert set(wsk._tapes) & set(kcp.FWD_TAPES) == set(want), (
            fused,
            sorted(wsk._tapes),
        )
        assert kcp.kept_registry_size() == 0
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(arms[1][n], arms[0][n]), n


# --------------------------------------------------------------------------- host launch path: stream binding, registry answers, chunk tokens
# Host data only: what a launch binds for tvm-ffi and what the forward re-derives per call.  The launches, their
# records, grids and arguments are not changed by any of them.


def test_launch_binds_the_current_stream_through_use_raw_stream(monkeypatch) -> None:
    """``cake_backend.run_module`` with a counting stand-in for ``tvm_ffi`` (only ``use_raw_stream``) and a set current
    stream (CPU): every launch enters one ``use_raw_stream(device, raw)`` context on the current stream's device and
    raw handle and launches inside it; a stream switch (another device's stream) binds that stream; a launch on
    another thread binds its own; an exception from the launch leaves no binding behind."""
    be = _package("cake_backend")
    state = {"stream": (0, 11), "enters": 0, "bound": [], "launched": []}

    class _Context:
        def __init__(self, device, raw):
            self.stream = (device[1], raw)  # the stand-in device is ("cuda", index)

        def __enter__(self):
            state["enters"] += 1
            state["bound"].append(self.stream)
            return self

        def __exit__(self, *exc):
            state["bound"].pop()

    monkeypatch.setattr(be, "tvm_ffi", types.SimpleNamespace(use_raw_stream=_Context))
    monkeypatch.setattr(be, "_ffi_device", lambda index: ("cuda", index))
    monkeypatch.setattr(be, "_current_stream", lambda: state["stream"])
    name = "fwd_prep"
    values = {n: n for n, _axis in be._argument_plan(name) if n is not None}

    def entry(*a):
        state["launched"].append(state["bound"][-1] if state["bound"] else None)
        if state.get("raise"):
            raise RuntimeError("inside the launch")

    module = types.SimpleNamespace(**{be.record(name)["ffi_entry"]: entry})

    def launch() -> None:
        be.run_module(name, values, grid=(1, 1, 1), device_index=0, module=module)

    def enters_after(fn) -> int:
        n = state["enters"]
        fn()
        return state["enters"] - n

    assert enters_after(launch) == 1 and state["launched"][-1] == (0, 11)
    assert enters_after(launch) == 1 and state["bound"] == []
    state["stream"] = (1, 22)
    assert enters_after(launch) == 1 and state["launched"][-1] == (1, 22)
    state["stream"] = (0, 11)
    worker = threading.Thread(target=launch)
    assert enters_after(lambda: (worker.start(), worker.join())) == 1
    assert state["launched"][-1] == (0, 11)
    state["raise"] = True
    with pytest.raises(RuntimeError, match="inside the launch"):
        launch()
    assert state["bound"] == [] and state["launched"][-1] == (0, 11)


def test_stream_context_sets_the_ffi_stream_of_the_device() -> None:
    """``cake_backend.stream_context((index, raw))`` with the real ``tvm_ffi``: inside, tvm-ffi's current stream of CUDA
    device ``index`` is ``raw``, another device's stream is untouched, nested contexts restore in order and the
    previous stream is back after the outermost one; ``_ffi_device`` builds each device once.  With CUDA: for torch's
    current stream (the default one and a side stream) the stream it binds is the one ``tvm_ffi.use_torch_stream()``
    binds."""
    import tvm_ffi

    be = _package("cake_backend")
    dev0, dev1 = tvm_ffi.device("cuda", 0), tvm_ffi.device("cuda", 1)
    before0, before1 = tvm_ffi.get_raw_stream(dev0), tvm_ffi.get_raw_stream(dev1)
    with be.stream_context((0, 0x1234)):
        assert tvm_ffi.get_raw_stream(dev0) == 0x1234
        assert tvm_ffi.get_raw_stream(dev1) == before1
        with be.stream_context((1, 0x5678)):
            assert tvm_ffi.get_raw_stream(dev1) == 0x5678
            assert tvm_ffi.get_raw_stream(dev0) == 0x1234
        with be.stream_context((0, 0x9ABC)):
            assert tvm_ffi.get_raw_stream(dev0) == 0x9ABC
        assert tvm_ffi.get_raw_stream(dev0) == 0x1234
        assert tvm_ffi.get_raw_stream(dev1) == before1
    assert tvm_ffi.get_raw_stream(dev0) == before0
    assert be._ffi_device(0) is be._ffi_device(0)
    d1 = be._ffi_device(1)
    assert (d1.dlpack_device_type(), d1.index) == (dev1.dlpack_device_type(), 1)
    if not torch.cuda.is_available():
        return
    index = torch.cuda.current_device()
    dev = tvm_ffi.device("cuda", index)
    side = torch.cuda.Stream()
    for stream in (torch.cuda.current_stream(), side):
        with torch.cuda.stream(stream):
            with tvm_ffi.use_torch_stream():
                want = tvm_ffi.get_raw_stream(dev)
            with be.stream_context(be._current_stream()):
                got = tvm_ffi.get_raw_stream(dev)
        assert got == want == stream.cuda_stream, (got, want, stream)


def test_registry_answers_are_derived_once_per_registry_identity(monkeypatch) -> None:
    """``l2n_records_shipped`` / ``fused_o_shipped(arch)`` answer as their derivation from the registry and derive it
    once per registry identity: a repeated question reads no record; another ``records_of_program``, another record
    table (a copy of the same records), a table of another size, another registry (a stand-in) or another
    architecture is derived again; a registry without a record table is derived on every call."""
    kcp = _package("cake_kcp")
    jit = _package("cake_jit")
    real = jit.records_of_program
    calls: list[str] = []
    # the real registry's answers, kept before records_of_program is replaced: another lookup is another key
    kcp.l2n_records_shipped()
    kcp.fused_o_shipped(None)

    def counting(program):
        calls.append(program)
        return real(program)

    def want_l2n(lookup) -> bool:
        try:
            return all(lookup(p) for p in kcp.L2N_PROGRAMS)
        except ValueError:
            return False

    def want_fused(modules, lookup, arch) -> bool:
        try:
            names = lookup("kcp_fwdh_out")
        except ValueError:
            return False
        forms = {
            int(modules[n].get("grid_rule", {}).get("ctas_per_head", 1))
            for n in names
            if arch is None or arch in modules[n].get("arches", ())
        }
        return forms >= {1, 2}

    def derives(fn) -> int:
        n = len(calls)
        fn()
        return len(calls) - n

    monkeypatch.setattr(jit, "records_of_program", counting)
    assert derives(lambda: kcp.l2n_records_shipped()) > 0
    assert derives(lambda: kcp.l2n_records_shipped()) == 0
    assert kcp.l2n_records_shipped() == want_l2n(real)
    for arch in (None, *jit.ARCHES, "sm_00"):
        assert derives(lambda a=arch: kcp.fused_o_shipped(a)) == 1, arch
        assert derives(lambda a=arch: kcp.fused_o_shipped(a)) == 0, arch
        assert kcp.fused_o_shipped(arch) == want_fused(jit.MODULES, real, arch), arch
    # the same records in another table
    monkeypatch.setattr(jit, "MODULES", dict(jit.MODULES))
    assert derives(lambda: kcp.l2n_records_shipped()) > 0
    assert derives(lambda: kcp.fused_o_shipped(None)) == 1
    del jit.MODULES[next(iter(jit.MODULES))]  # a table of another size
    assert derives(lambda: kcp.l2n_records_shipped()) > 0
    assert derives(lambda: kcp.l2n_records_shipped()) == 0

    # stand-in registries: the answer is the stand-in's
    def form(ctas: int) -> dict:
        return {
            "program": "kcp_fwdh_out",
            "grid_rule": {"ctas_per_head": ctas},
            "arches": ["sm_x"],
        }

    def registry(table) -> types.SimpleNamespace:
        def lookup(program):
            calls.append(program)
            if program not in kcp.PROGRAMS:
                raise ValueError(program)
            return tuple(n for n, r in table.items() if r["program"] == program)

        return types.SimpleNamespace(MODULES=table, records_of_program=lookup)

    both = registry({"a": form(1), "b": form(2)})
    for stand_in, fused in ((both, True), (registry({"a": form(1)}), False)):
        monkeypatch.setattr(kcp, "_registry", lambda s=stand_in: s)
        assert derives(lambda: kcp.fused_o_shipped("sm_x")) == 1
        assert kcp.fused_o_shipped("sm_x") is fused
        assert kcp.fused_o_shipped("sm_y") is False
        assert derives(lambda: kcp.l2n_records_shipped()) > 0
        assert kcp.l2n_records_shipped() is False
    both.MODULES["c"] = form(1)  # the same table, grown
    monkeypatch.setattr(kcp, "_registry", lambda: both)
    assert derives(lambda: kcp.fused_o_shipped("sm_x")) == 1
    tableless = types.SimpleNamespace(
        MODULES=None, records_of_program=both.records_of_program
    )
    monkeypatch.setattr(kcp, "_registry", lambda: tableless)
    assert derives(lambda: kcp.l2n_records_shipped()) > 0
    assert derives(lambda: kcp.l2n_records_shipped()) > 0


def test_chunk_tokens_follow_the_contexts_chunk_table(monkeypatch) -> None:
    """``_chunk_tokens`` (the in-kernel L2 norm's coverage check) is the table's sum, kept in the context's cache with
    the table it was taken over: another table object is summed again, a list or a context without a cache dict is
    summed per call and never kept; ``_prepare`` refuses a coverage other than the call's tokens."""
    kcp, ctx, ops = _ckpt_setup()
    key = ("kcp_chunk_tokens",)
    stub = types.SimpleNamespace(_caches={})
    t1, t2 = (64, 64, 10), (64, 64, 11)
    assert kcp._chunk_tokens(stub, t1) == 138 and stub._caches[key] == (t1, 138)
    assert kcp._chunk_tokens(stub, t1) == 138
    assert kcp._chunk_tokens(stub, t2) == 139 and stub._caches[key][0] is t2
    assert kcp._chunk_tokens(stub, [64, 1]) == 65 and stub._caches[key][0] is t2
    for other in (types.SimpleNamespace(), types.SimpleNamespace(_caches=None)):
        assert kcp._chunk_tokens(other, t1) == 138
    # the real context: its own table, kept once; a wrong coverage is refused
    out = _ckpt_autograd(kcp, ctx, ops, ckpt=False, use_qk_l2norm_in_kernel=True)
    assert ctx._caches[key] == (ctx.chunk_len, sum(ctx.chunk_len))
    assert ctx._caches[key][0] is ctx.chunk_len
    again = _ckpt_autograd(kcp, ctx, ops, ckpt=False, use_qk_l2norm_in_kernel=True)
    assert all(torch.equal(out[k], again[k]) for k in out)
    monkeypatch.setattr(kcp, "_chunk_tokens", lambda c, table: sum(table) - 1)
    with pytest.raises(ValueError, match="chunks cover"):
        _ckpt_autograd(kcp, ctx, ops, ckpt=False, use_qk_l2norm_in_kernel=True)


# --------------------------------------------------------------------------- eager pool / ticket sizing after launch #1
# An eager call sizes its workspace pools and ticket slots after its first launch is issued (``ensure(defer=True)``,
# :meth:`CakeGDNDetKcpWorkspace.settle`): no pool or ticket byte exists at that launch, and the launch log, the
# results and the workspace's bytes are the package's with the sizing done in ``ensure`` (``defer=False``).

DEFER_ROWS = {
    # one rank (no exchange)
    "one_rank": (
        [0, 100, 740, 817, 1817, 1946],
        1,
        {"layout": "contiguous"},
        (2, 6),
        1.0,
    ),
    # the W = 8 multi-rank row (per-document zigzag, every rank in the exchange)
    "w8_zigzag": (MR_CU, MR_W, {"layout": "zigzag_per_document"}, MR_HEADS, MR_G_SCALE),
    # a rank without tokens (it still owns heads and joins the exchange)
    "zero_token_rank": (
        [0, 1700, 3000],
        4,
        {"fragments": _ragged_fragments([0, 1700, 3000], [1200, 0, 900, 900])},
        (2, 6),
        MR_G_SCALE,
    ),
}


def _defer_setup(row: str):
    cu, W, part_kw, heads, g_scale = DEFER_ROWS[row]
    kcp = _package("cake_kcp")
    part = gp.cake_gdn_det_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), W, **part_kw
    )
    ops = dict(_inputs(cu, *heads))
    ops["g"] = (ops["g"] * g_scale).contiguous()
    return kcp, part, W, ops


class _RawPeakWorkspace:
    """A ``CakeGDNDetKcpWorkspace`` recording the most bytes it holds (pools + tapes + ticket slots, read from its
    buffers directly: the byte properties would complete a deferred sizing) after every allocating / releasing
    step."""

    def __new__(cls, kcp):
        class Raw(kcp.CakeGDNDetKcpWorkspace):
            def __init__(self, *a, **k):
                super().__init__(*a, **k)
                self.peak = 0

            def _held(self) -> int:
                n = sum(t.numel() * t.element_size() for t in self._pools.values())
                n += sum(t.numel() * t.element_size() for t in self._tapes.values())
                t = self._tickets
                return n + (0 if t is None else t.numel() * t.element_size())

            def _upd(self):
                self.peak = max(self.peak, self._held())

            def ensure(self, *a, **k):
                tb = super().ensure(*a, **k)
                self._upd()
                return tb

            def settle(self):
                super().settle()
                self._upd()

            def tape(self, name):
                t = super().tape(name)
                self._upd()
                return t

            def pool(self, name, n=None):
                t = super().pool(name, n)
                self._upd()
                return t

            def adopt_tape(self, name, t):
                r = super().adopt_tape(name, t)
                self._upd()
                return r

        return Raw()


def _defer_run(kcp, part, W, ops, monkeypatch, *, defer: bool, knobs: dict, peak: bool):
    """Forward + saved-set backward of every rank (fake backend, ``W`` threads when ``W > 1``) with the deferred
    sizing as shipped (``defer``) or with ``ensure`` sizing at once: the global-order results, per rank the calls in
    order (every eager call reaches ``_record``), the workspace state each call's first launch saw (``id(call)`` ->
    pools held, ticket slots bound, sizing pending) and the peaks.  ``peak``: a caller-held
    :class:`_RawPeakWorkspace` per rank (its forward and backward peaks per rank); else ``workspace=None`` (a fresh
    workspace per call)."""
    holder: dict = {}
    first: dict = {}
    lock = threading.Lock()

    def rank_of() -> int:
        comm = holder.get("comm")
        return getattr(comm.local, "rank", 0) if comm is not None else 0

    class FirstLaunch(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            ws = call.ws
            with lock:
                if id(call) not in first:
                    first[id(call)] = (
                        name,
                        sorted(ws._pools),
                        ws._tickets is not None,
                        ws._pending is not None,
                    )
            super().run(name, args)

    calls: dict[int, list] = {r: [] for r in range(W)}
    real_record = kcp._record

    def record(kind, call):
        with lock:
            calls[rank_of()].append(call)
        return real_record(kind, call)

    monkeypatch.setattr(kcp, "_record", record)
    if not defer:
        original = kcp.CakeGDNDetKcpWorkspace.ensure

        def ensure(self, *a, defer=False, **k):  # noqa: ARG001 - the undeferred sizing
            return original(self, *a, defer=False, **k)

        monkeypatch.setattr(kcp.CakeGDNDetKcpWorkspace, "ensure", ensure)
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(W)
    ]
    wss = [_RawPeakWorkspace(kcp) if peak else None for _ in range(W)]
    peaks: dict[int, dict] = {}

    def rank_fn(r, comm):
        holder["comm"] = comm
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        kw = dict(backend="fake", workspace=wss[r], **knobs)
        out, saved = kcp.cake_gdn_det_train_kcp_forward(*args, ctxs[r], **kw)
        fwd_peak = wss[r].peak if peak else None
        if peak:
            wss[r].peak = 0
        grads = kcp.cake_gdn_det_train_kcp_backward(
            *args, ctxs[r], loc["dout"], saved=saved, **kw
        )
        if peak:
            peaks[r] = {"forward": fwd_peak, "backward": wss[r].peak}
            assert wss[r]._pending is None, r
        return out, grads

    previous = kcp.set_backend("fake", FirstLaunch())
    try:
        if W == 1:
            per_rank = [rank_fn(0, None)]
        else:
            per_rank = _run_threaded(kcp, W, rank_fn, monkeypatch)
    finally:
        kcp.set_backend("fake", previous)
        monkeypatch.undo()
    res = {"out": fake.gather_global(part, [o for o, _ in per_rank])}
    for i, n in enumerate(EXPLICIT_GRADS):
        res[n] = fake.gather_global(part, [g[i] for _, g in per_rank])
    return res, calls, first, peaks


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"save_mode": "full"},
        {"use_qk_l2norm_in_kernel": True},
        {"use_qk_l2norm_in_kernel": True, "test_knobs": {"prep_form": 0}},
        {"segments": 2},
        {"use_qk_l2norm_in_kernel": True, "segments": 2},
    ],
    ids=("light", "full", "l2n", "l2n_pf0", "P2", "l2n_P2"),
)
@pytest.mark.parametrize("row", tuple(DEFER_ROWS))
def test_eager_call_sizes_pools_and_tickets_after_the_first_launch(
    monkeypatch, row, knobs
) -> None:
    """Every rank's forward and saved-set backward (light, ``save_mode="full"``, the in-kernel L2 norm, P = 2) with the
    deferred sizing against ``ensure`` sizing at once: out and the five gradients are bitwise equal; each call issues
    the same programs with the same records and grids; at the first launch of a call on a rank with tokens the
    deferred workspace holds no pool (and no ticket slot unless that program is persistent) and its sizing is pending
    (the undeferred one holds both);
    after each call nothing is pending and every pool, tape and ticket byte equals the undeferred call's, also on a
    rank without tokens; with a caller-held workspace the forward and backward peaks are the same bytes."""
    if row != "one_rank" and knobs.get("segments") == 2:
        pytest.skip("P = 2 needs fragments of more than one chunk (the one-rank row)")
    kcp, part, W, ops = _defer_setup(row)
    ref, ref_calls, ref_first, _ = _defer_run(
        kcp, part, W, ops, monkeypatch, defer=False, knobs=knobs, peak=False
    )
    got, got_calls, got_first, _ = _defer_run(
        kcp, part, W, ops, monkeypatch, defer=True, knobs=knobs, peak=False
    )
    for n in ("out",) + EXPLICIT_GRADS:
        assert torch.equal(got[n], ref[n]), n
    deferred = tokenless = 0
    for r in range(W):
        assert [c.backward for c in got_calls[r]] == [False, True], r
        assert [c.backward for c in ref_calls[r]] == [False, True], r
        for c1, c0 in zip(got_calls[r], ref_calls[r], strict=True):
            what = (r, "bwd" if c1.backward else "fwd")
            assert c1.sequence == c0.sequence, what
            assert c1.records == c0.records, what
            assert c1.ws._pending is None and c0.ws._pending is None, what
            assert c1.ws.pool_bytes == c0.ws.pool_bytes, what
            assert c1.ws.tape_bytes == c0.ws.tape_bytes, what
            assert c1.ws.ticket_bytes == c0.ws.ticket_bytes > 0, what
            assert (id(c1) in got_first) == (id(c0) in ref_first), what
            if id(c0) in ref_first:
                _, _, tk0, pend0 = ref_first[id(c0)]
                assert tk0 and not pend0, what  # the reference sized at ensure
            if c1.T_r == 0:
                tokenless += 1
                continue
            # the call's first launch (#1, or the backward's first program) found no pool sized yet; the ticket
            # slots exist then only when that program is persistent (#6 on saved or kept tapes takes its slot)
            p1, pools1, tk1, pend1 = got_first[id(c1)]
            assert pend1 and pools1 == [], (*what, p1, pools1)
            assert tk1 == (p1 in kcp.TICKET_SITES), (*what, p1, tk1)
            deferred += 1
    assert deferred >= 2 and tokenless == (2 if row == "zero_token_rank" else 0)
    # caller-held workspaces: the same peak bytes per rank, forward and backward
    *_, peak0 = _defer_run(
        kcp, part, W, ops, monkeypatch, defer=False, knobs=knobs, peak=True
    )
    *_, peak1 = _defer_run(
        kcp, part, W, ops, monkeypatch, defer=True, knobs=knobs, peak=True
    )
    assert peak1 == peak0, (peak1, peak0)


@pytest.mark.parametrize("reuse", (False, True, "auto"))
@pytest.mark.parametrize("ckpt", (False, True), ids=("direct", "checkpoint"))
def test_deferred_sizing_under_autograd_and_checkpoints(
    monkeypatch, ckpt, reuse
) -> None:
    """One rank, direct autograd and a non-reentrant checkpoint under every ``reuse_recompute_tapes`` form, with a
    fresh and with a caller-held workspace: out and the five gradients are bitwise the undeferred package's, no
    sizing is left pending and no kept entry is left."""
    kcp, ctx, ops = _ckpt_setup()
    for held in (False, True):
        runs = {}
        for defer in (False, True):
            if not defer:
                original = kcp.CakeGDNDetKcpWorkspace.ensure

                def ensure(self, *a, defer=False, _o=original, **k):  # noqa: ARG001
                    return _o(self, *a, defer=False, **k)

                monkeypatch.setattr(kcp.CakeGDNDetKcpWorkspace, "ensure", ensure)
            ws = kcp.CakeGDNDetKcpWorkspace() if held else None
            runs[defer] = _ckpt_autograd(
                kcp, ctx, ops, ckpt=ckpt, workspace=ws, reuse_recompute_tapes=reuse
            )
            if ws is not None:
                assert ws._pending is None
            monkeypatch.undo()
        _assert_same(runs[True], runs[False], f"held={held}")
    assert kcp.kept_registry_size() == 0


def test_settle_completes_a_pending_sizing_before_any_read() -> None:
    """``ensure(defer=True)`` sizes nothing; a pool read, ``has``, ``release_pool``, the byte properties and the next
    ``ensure`` each complete the pending sizing first, and the result is the undeferred workspace's.  A ticket read
    sizes the ticket slots alone (a persistent first launch issues before the pools); the pools stay pending."""
    kcp, ctx, _ = _ckpt_setup()
    want = kcp.CakeGDNDetKcpWorkspace()
    want.ensure(ctx, 2, 6)
    reads = {
        "pool": lambda ws: ws.pool("eye"),
        "ticket": lambda ws: ws.ticket(kcp.TICKET_SITES[0]),
        "has": lambda ws: ws.has("eye"),
        "bytes": lambda ws: ws.bytes,
        "pool_bytes": lambda ws: ws.pool_bytes,
        "ticket_bytes": lambda ws: ws.ticket_bytes,
        "settle": lambda ws: ws.settle(),
        "ensure": lambda ws: ws.ensure(ctx, 2, 6, defer=True),
    }
    for what, read in reads.items():
        ws = kcp.CakeGDNDetKcpWorkspace()
        ws.ensure(ctx, 2, 6, defer=True)
        assert ws._pools == {} and ws._tickets is None and ws._pending is not None, what
        assert ws.geometry == want.geometry and ws.qk_heads == want.qk_heads, what
        read(ws)
        if what == "ticket":
            assert (
                ws._tickets is not None and ws._pools == {} and ws._pending is not None
            ), what
        elif what != "ensure":
            assert ws._pending is None, what
        assert ws.pool_bytes == want.pool_bytes, what
        assert ws.ticket_bytes == want.ticket_bytes, what
        assert torch.equal(ws.pool("eye"), want.pool("eye")), what
    ws = kcp.CakeGDNDetKcpWorkspace()
    ws.ensure(ctx, 2, 6, defer=True)
    assert ws.release_pool("eye") and ws._pending is None and not ws.has("eye")


# --------------------------------------------------------------------------- host launch-state memos (launch cache, plan memos)
# The launch cache (_Tables.launches: record, grid, test flag, merge columns per launch key) and the plan's memos of the
# launches' static pieces (normalised pool tables, kernel addressing, walk / merge-chain / zero-block device tables,
# split lists, pool sizes) change no launch: every launch receives the very tensors and values it receives without them.


def _arg_print(values: dict) -> tuple:
    """A launch's logical arguments as comparable values: an int32 / int64 tensor (a plan device table, cached on the
    shared context) by its identity (data pointer), shape, stride and dtype; any other tensor (operands, outputs,
    tapes, pools: per call) by shape, stride, storage offset and dtype; a ticket slot (u32, per call) by its
    128-byte alignment in place of its storage offset, which carries the allocator's skew (``_new_ticket_slots``:
    the CPU allocator aligns a buffer to 64 bytes only, so the slots start 0 or 16 words into it, call by call);
    scalars as they are."""
    slot_bytes = _package("cake_kcp").TICKET_SLOT_BYTES
    out = []
    for k in sorted(values):
        v = values[k]
        if isinstance(v, torch.Tensor):
            ptr = v.data_ptr() if v.dtype in (torch.int32, torch.int64) else None
            if v.dtype == torch.uint32:
                offset = ("aligned", v.data_ptr() % slot_bytes)
            else:
                offset = v.storage_offset()
            v = (
                "T",
                ptr,
                tuple(v.shape),
                tuple(v.stride()),
                offset,
                str(v.dtype),
            )
        out.append((k, v))
    return tuple(out)


def _memos_off(kcp, monkeypatch) -> None:
    """The launch path without the launch cache and the plan memos (each piece derived per launch as before)."""
    T = kcp._Tables
    monkeypatch.setattr(
        kcp,
        "_resolve_cached",
        lambda name, args, cols_per_cta=8: kcp.resolve_launch(
            name, args, cols_per_cta=cols_per_cta
        ),
    )
    monkeypatch.setattr(T, "indexed_spec", lambda self, index, tiles: None)
    monkeypatch.setattr(
        T,
        "walk_table",
        lambda self, walk, device: self.dev("walk", [int(s) for s in walk], device),
    )
    monkeypatch.setattr(
        T,
        "neg_index",
        lambda self, device: self.dev("neg", [-1] * int(self.NSEG), device),
    )
    monkeypatch.setattr(
        T,
        "chain_tables",
        lambda self, family, device: kcp._chain_device_tables(
            self, self.chains[family], device
        ),
    )
    monkeypatch.setattr(T, "split_lists", lambda self, coll: self.split_elems(coll))
    monkeypatch.setattr(
        T,
        "pool_size_items",
        lambda self, backward: tuple(self.pool_sizes(backward).items()),
    )


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"save_mode": "full"},
        {"use_qk_l2norm_in_kernel": True},
        {"use_qk_l2norm_in_kernel": True, "test_knobs": {"prep_form": 0}},
        {"segments": 2},
        {"use_qk_l2norm_in_kernel": True, "segments": 2},
    ],
    ids=("light", "full", "l2n", "l2n_pf0", "P2", "l2n_P2"),
)
@pytest.mark.parametrize("row", tuple(DEFER_ROWS))
def test_launch_memos_hand_every_launch_the_same_arguments(
    monkeypatch, row, knobs
) -> None:
    """Every rank's forward and saved-set backward, twice on the same contexts (the second pass hits every memo), with
    the launch cache and the plan memos and without them: each launch gets the same record, grid and logical
    arguments (the same device tensors: data pointer, shape, stride, dtype); out and the five gradients are bitwise
    equal; the cache holds one entry per distinct launch key and the second pass adds none."""
    if row != "one_rank" and knobs.get("segments") == 2:
        pytest.skip("P = 2 needs fragments of more than one chunk (the one-rank row)")
    kcp, part, W, ops = _defer_setup(row)
    ctxs = [
        gp.cake_gdn_det_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(W)
    ]
    holder: dict = {}
    seen: dict = {}
    lock = threading.Lock()

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            comm = holder.get("comm")
            rank = getattr(comm.local, "rank", 0) if comm is not None else 0
            n0 = len(args["_call"].records)
            record, values, grid, _ = kcp._resolve_cached(name, dict(args))
            del args["_call"].records[
                n0:
            ]  # the backend's own resolution below appends the record
            with lock:
                seen.setdefault(holder["tag"], {}).setdefault(rank, []).append(
                    (name, record, tuple(grid), _arg_print(values))
                )
            super().run(name, args)

    def one_pass(tag: str):
        holder["tag"] = tag

        def rank_fn(r, comm):
            holder["comm"] = comm
            loc = fake.slice_local(ops, part, r)
            args = [loc[n] for n in LEAVES]
            out, saved = kcp.cake_gdn_det_train_kcp_forward(
                *args, ctxs[r], backend="fake", **knobs
            )
            grads = kcp.cake_gdn_det_train_kcp_backward(
                *args, ctxs[r], loc["dout"], saved=saved, backend="fake", **knobs
            )
            return out, grads

        if W == 1:
            per_rank = [rank_fn(0, None)]
        else:
            per_rank = _run_threaded(kcp, W, rank_fn, monkeypatch)
        res = {"out": fake.gather_global(part, [o for o, _ in per_rank])}
        for i, n in enumerate(EXPLICIT_GRADS):
            res[n] = fake.gather_global(part, [g[i] for _, g in per_rank])
        return res

    previous = kcp.set_backend("fake", Recording())
    try:
        on1 = one_pass("on1")
        tables = [
            tb
            for c in ctxs
            for k, tb in c._caches.items()
            if isinstance(k, tuple) and k[:1] == ("kcp_tables",)
        ]
        assert tables and all(isinstance(tb.launches, dict) for tb in tables)
        sizes1 = [len(tb.launches) for tb in tables]
        on2 = one_pass("on2")
        assert [len(tb.launches) for tb in tables] == sizes1  # warm: no new key
        assert sum(sizes1) > 0
        _memos_off(kcp, monkeypatch)
        off = one_pass("off")
    finally:
        kcp.set_backend("fake", previous)
        monkeypatch.undo()
    for n in ("out",) + EXPLICIT_GRADS:
        assert torch.equal(on1[n], off[n]) and torch.equal(on2[n], off[n]), n
    for r in range(W):
        assert seen["on1"][r] == seen["off"][r], r
        # the second pass reuses the device tables, so the very same launches repeat
        assert seen["on2"][r] == seen["off"][r], r
    if knobs.get("use_qk_l2norm_in_kernel") and kcp.prep_form_shipped():
        # the light save's forward runs the forward form of #1_l2n and its backward's replay the full record: one
        # pass resolves both forms of the same launch, which the launch key (``_backward_tapes``) keeps apart
        preps = {
            rec
            for per_rank in seen["on1"].values()
            for _n, rec, _g, _v in per_rank
            if rec.startswith("fwd_prep")
        }
        pf = (knobs.get("test_knobs") or {}).get("prep_form", 1)
        assert preps == (
            {"fwd_prep_l2n_fwd", "fwd_prep_l2n"} if pf == 1 else {"fwd_prep_l2n"}
        ), preps


def test_keyword_options_are_validated_once_per_keyword_set(monkeypatch) -> None:
    """``cake_gdn_det_kcp`` without ``options``: the same keywords (values and types) share one validated options
    object, other values build another, an unhashable value (``test_knobs``) validates per call, and an ``options``
    instance is validated per call as before; the results do not change."""
    kcp, ctx, ops = _ckpt_setup()
    built: dict[str, int] = {"keywords": 0, "instance": 0}
    real = kcp.make_options

    def counting(*a, **k):
        built["keywords" if not a or a[0] is None else "instance"] += 1
        return real(*a, **k)

    monkeypatch.setattr(kcp, "make_options", counting)
    kcp._KEYWORD_OPTIONS.clear()
    a = kcp._keyword_options({"save_mode": "full"}, {"backend": "fake"})
    b = kcp._keyword_options({"save_mode": "full"}, {"backend": "fake"})
    assert a is b and built["keywords"] == 1
    c = kcp._keyword_options({"save_mode": "light"}, {"backend": "fake"})
    assert c is not a and built["keywords"] == 2
    d1 = kcp._keyword_options({}, {"backend": "fake", "test_knobs": {"fused_o": 0}})
    d2 = kcp._keyword_options({}, {"backend": "fake", "test_knobs": {"fused_o": 0}})
    assert d1 is not d2 and built["keywords"] == 4
    # the autograd entry: one validated object across forwards of the same keywords
    built.update(keywords=0, instance=0)
    r1 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_mode="full")
    r2 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_mode="full")
    assert built["keywords"] == 1
    _assert_same(r1, r2, "memoised options")
    r0 = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, save_mode="full", test_knobs={"fused_o": 1}
    )
    _assert_same(r0, r1, "per-call options")
    opts = real(None, backend="fake", save_mode="full")
    built.update(keywords=0, instance=0)
    _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=opts)
    _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=opts)
    assert (
        built["keywords"] == 0 and built["instance"] >= 2
    )  # an options instance: validated per call


def test_packed_qkv_views_are_the_sliced_views() -> None:
    """The packed split's views (the channel axis unflattened to heads and split) are the ``[..., :a]`` / ``[a:2a]`` / ``[2a:]``
    slices' views: the same storage offset, shape and strides, with and without a batch axis."""
    kcp = _package("cake_kcp")
    for lead in ((37,), (1, 37)):
        qkv = torch.randn(*lead, (2 * 2 + 6) * 128).to(torch.bfloat16)
        got = kcp._qkv_views(qkv, 2, 6)
        a = 2 * 128
        want = (
            qkv[..., :a].view(*lead, 2, 128),
            qkv[..., a : 2 * a].view(*lead, 2, 128),
            qkv[..., 2 * a :].view(*lead, 6, 128),
        )
        for x, y in zip(got, want, strict=True):
            assert (x.data_ptr(), x.shape, x.stride(), x.storage_offset()) == (
                y.data_ptr(),
                y.shape,
                y.stride(),
                y.storage_offset(),
            )


# --------------------------------------------------------------------------- the deferred #5 decision, the registry read, the packed views
# A forward takes its #5 decision (_fuse_o; the frame proof of recompute_output="auto") right after #1 is issued.

DECISION_FORMS = {
    "light": {},
    "split": {"test_knobs": {"fused_o": 0}},
    "l2n": {"use_qk_l2norm_in_kernel": True},
    "P2": {"segments": 2},
}


@pytest.mark.parametrize("ckpt", (False, True), ids=("direct", "checkpoint"))
@pytest.mark.parametrize("form", ("light", "split", "l2n", "P2"))
def test_output_decision_follows_the_first_launch(monkeypatch, form, ckpt) -> None:
    """A forward takes its #5 decision (``_fuse_o``; under a checkpoint also the frame proof of
    ``recompute_output="auto"``) right after #1 is issued, on a fresh and on a caller-held workspace, and before #1
    when the caller-held workspace still binds the ``h`` / ``v_new`` tapes of an earlier split forward: out and the
    five gradients, the launch logs (sequence, records, ``fwd_o`` / ``fwd_o_reason``) and RECOMPUTE_OUTPUT_STATS are
    those of the decision taken before #1."""
    kcp, ctx, ops = _ckpt_setup()
    knobs = DECISION_FORMS[form]
    prep = "fwd_prep_l2n" if knobs.get("use_qk_l2norm_in_kernel") else "fwd_prep"
    seen: list = []
    real_fuse, real_forward = kcp._fuse_o, kcp._forward

    def fuse(call, *, launch, hand_off):
        held_hv = bool({"h_tape", "v_new"} & set(call.ws._tapes))
        seen.append((tuple(call.sequence), held_hv))
        return real_fuse(call, launch=launch, hand_off=hand_off)

    def early(call, *a, fuse_o=False, **k):
        if callable(fuse_o):
            fuse_o = bool(fuse_o())
        return real_forward(call, *a, fuse_o=fuse_o, **k)

    def arm(deferred: bool, held):
        ws = None if held is False else kcp.CakeGDNDetKcpWorkspace()
        if held == "split":
            # a split forward leaves h / v_new bound in the caller-held workspace
            kcp.cake_gdn_det_train_kcp_forward(
                *[ops[n] for n in LEAVES],
                ctx,
                backend="fake",
                workspace=ws,
                test_knobs={"fused_o": 0},
            )
            assert {"h_tape", "v_new"} <= set(ws._tapes)
        with monkeypatch.context() as m:
            m.setattr(kcp, "_fuse_o", fuse)
            if not deferred:
                m.setattr(kcp, "_forward", early)
            seen.clear()
            stats0 = dict(kcp.RECOMPUTE_OUTPUT_STATS)
            res = _ckpt_autograd(kcp, ctx, ops, ckpt=ckpt, workspace=ws, **knobs)
            stats = {k: v - stats0[k] for k, v in kcp.RECOMPUTE_OUTPUT_STATS.items()}
            log = {kind: dict(kcp.LAST_CALL[kind]) for kind in ("fwd", "bwd")}
            return res, log, stats, list(seen)

    for held in (False, True, "split"):
        ref, ref_log, ref_stats, ref_seen = arm(False, held)
        got, got_log, got_stats, got_seen = arm(True, held)
        _assert_same(got, ref, f"held={held}")
        assert got_log == ref_log, held
        assert got_stats == ref_stats, held
        assert ref_seen and all(s == () for s, _ in ref_seen), (held, ref_seen)
        assert [hv for _, hv in got_seen] == [hv for _, hv in ref_seen], held
        # every forward of this setup holds tokens: #1 is its first launch
        for s, hv in got_seen:
            assert s == (() if hv else (prep,)), (held, s, hv)
        if held == "split":  # the first forward found them bound
            assert got_seen[0][1], got_seen
        elif held is False:
            # a fresh workspace per call binds none at a forward's start
            assert not any(hv for _, hv in got_seen), got_seen
    assert kcp.kept_registry_size() == 0


def test_registry_is_the_imported_cake_jit() -> None:
    """``_registry()`` returns the module ``import`` returns for the package's ``cake_jit`` (imported once); the
    facade's backend package is the package module."""
    kcp = _package("cake_kcp")
    jit = importlib.import_module(f"{kcp.__package__}.cake_jit")
    assert kcp._registry() is jit
    assert kcp._registry() is jit
    api = importlib.import_module("flashinfer.cake_gdn_det_train_kcp")
    pkg = importlib.import_module(kcp.__package__)
    assert api._pkg() is pkg and api._pkg() is pkg


def test_packed_qkv_views_refuse_an_in_place_change() -> None:
    """The split's views are outputs of one custom autograd node: while the packed row requires grad, autograd
    refuses an in-place change to any of them (its custom-function view rule)."""
    kcp = _package("cake_kcp")
    qkv = torch.randn(37, (2 * 2 + 6) * 128).to(torch.bfloat16).requires_grad_(True)
    for i in range(3):
        views = kcp.cake_gdn_det_train_kcp_packed_qkv_views(qkv, H_qk=2, HV=6)
        with pytest.raises(RuntimeError):
            views[i].mul_(2)
