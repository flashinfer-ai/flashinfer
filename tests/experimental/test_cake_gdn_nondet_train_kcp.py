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

CPU tests of the GDN KCP metadata layer of the NON-DETERMINISTIC package ``flashinfer.experimental.cake.cake_gdn_nondet_train_kcp``
(the counterpart of ``cake_gdn_det_train_kcp`` with the same contract and accuracy budgets, not its bitwise
reproducibility):
the package-local plan module against ``cake_kda_det_train_kcp.cake_kcp_plan`` (field-identical partitions and contexts on
the step-0 rows with ``H := HV``; the KDA package is imported HERE only, never by the package), the per-rank crossing
counts and exchange bytes against the rows of record, the table builders (pool addressing, segment lists, chains),
head ownership incl. the padded ``(HV, W) = (3, 2)`` case, the byte queries, the registry seed, an emulated multi-rank
row (W ranks on threads), and the package's host orchestration on the fake backend
(the CPU emulation of the programs) against the FP64 reference.  The mode is in the package name: no entry takes
a ``deterministic`` keyword.

Plan / table / byte / registry / launch-order checks are exact.  Numerical results are judged against the FP64
reference of ``tests/experimental/test_helpers/cake_gdn_det_train_kcp_reference.py`` with the deterministic route's budget for the
fake backend (``FAKE_TOL_REL_RMS``), and two arms that the deterministic tests compare bitwise (repeat, saved /
replay, kept / replayed tapes, checkpoint / direct, forced forms) are each judged against that reference and against
each other with the same budget (``_within_budget``); no tolerance is derived from a non-deterministic output.  The one
bitwise check left is labelled: a property of the fake backend's own emulation, not of the kernels.

Environment of the tests (never of the package): ``GDN_ROWS_DIR`` (the step-0 row JSONs; the row tests skip
without it), ``GDN_EVIDENCE_DIR`` (JSONL evidence records), the FP64 reference ``gdn_reference`` on ``PYTHONPATH``
(the two harness-reference tests skip without it).
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

# torch._dynamo (and sympy) imported once here, single-threaded: the FP64 reference's torch._compile wrappers import it
# lazily, and a first import racing the rank threads of another test can see a partially initialised sympy (Python 3.14)
import torch._dynamo  # noqa: F401

from tests.experimental.test_helpers import cake_gdn_det_train_kcp_reference as REF


def _package(submodule: str):
    """A module of the package through the test-helper registry: a stale entry fails here, never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_nondet_train_kcp", submodule)


def _det_package(submodule: str | None = None):
    """The deterministic package (or one of its modules) through the test-helper registry: a stale entry fails here
    and never skips."""
    from tests.cake.test_helpers.cake_package import cake_package

    return cake_package("cake_gdn_det_train_kcp", submodule)


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


@pytest.fixture(autouse=True, scope="module")
def _few_cpu_threads():
    """The CPU emulation runs many small matmuls and elementwise ops.  On a host with hundreds of CPUs, torch's default
    intra-op pool and the oneDNN matmul path (its own pool, which torch.set_num_threads does not resize) spend far more
    time synchronizing threads than computing (one test of this suite on a 352-CPU host: 272 s by default, 6 s with at
    most 8 threads and oneDNN off), so this module runs that way."""
    threads, onednn = torch.get_num_threads(), torch.backends.mkldnn.enabled
    torch.set_num_threads(min(threads, 8))
    torch.backends.mkldnn.enabled = False
    try:
        yield
    finally:
        torch.backends.mkldnn.enabled = onednn
        torch.set_num_threads(threads)


def _size_rule_at(kcp, monkeypatch, rows: int) -> None:
    """The size rule of ``reuse_recompute_tapes="auto"`` under ``graph="replay"`` at ``rows`` on every architecture
    (``cake_kcp.GRAPH_RECOMPUTE_MIN_ROWS`` and its default), with an empty architecture cache."""
    table = {arch: rows for arch in kcp.GRAPH_RECOMPUTE_MIN_ROWS}
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS", table)
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT", rows)
    monkeypatch.setattr(kcp, "_GRAPH_RECOMPUTE_ARCH", {})


@pytest.fixture(autouse=True)
def _graph_recompute_at_any_size(monkeypatch):
    """The test problems hold far fewer rows than the size rule of ``reuse_recompute_tapes="auto"`` under
    ``graph="replay"``, so every test runs with the rule at 0 rows (the recompute decides at any size); the rule's own
    test sets it around its problem."""
    _size_rule_at(_package("cake_kcp"), monkeypatch, 0)


def _evidence(name: str, record: dict) -> None:
    if not EVIDENCE_DIR:
        return
    os.makedirs(EVIDENCE_DIR, exist_ok=True)
    with open(os.path.join(EVIDENCE_DIR, f"{name}.jsonl"), "a") as fh:
        fh.write(json.dumps(record, sort_keys=True) + "\n")


# --------------------------------------------------------------------------- accuracy budgets (never derived from a run)
# The fake backend's budget is the deterministic route's FAKE_TOL_REL_RMS (FLA-class: BF16 tapes against FP64).  Every
# numerical result of this package is judged against the FP64 reference with it, and two arms that the deterministic
# twin compares bitwise are each judged against the reference AND against each other with the same constant.


def _within_budget(
    cand: torch.Tensor | None,
    ref: torch.Tensor | None,
    what: str,
    *,
    rel_rms: float = FAKE_TOL_REL_RMS,
) -> dict:
    """``cand`` within the budget of ``ref`` (the FP64 reference, or another arm for a cross-arm check): the same
    shape, finite, rel-RMS <= ``rel_rms`` (computed in FP64 by the reference helper's ``error_metrics``).  Two ``None``
    (a leaf the output does not reach) agree."""
    assert (cand is None) == (ref is None), (what, cand is None, ref is None)
    if cand is None:
        return {}
    assert tuple(cand.shape) == tuple(ref.shape), (
        what,
        tuple(cand.shape),
        tuple(ref.shape),
    )
    m = REF.error_metrics(cand, ref)
    assert m["nonfinite"] == 0, f"{what}: {m['nonfinite']} non-finite values"
    assert m["rel_rms"] <= rel_rms, (
        f"{what}: rel-RMS {m['rel_rms']:.3e} > {rel_rms} (max-abs {m['max_abs']:.3e}, worst {m['worst']})"
    )
    return m


def _arms_within_budget(
    arms: dict[str, dict],
    ref: dict,
    names,
    what: str,
    *,
    rel_rms: float = FAKE_TOL_REL_RMS,
) -> None:
    """Every arm against the FP64 reference and every pair of arms against each other, tensor by tensor (``names``),
    with the same budget: the non-deterministic form of the twin's bitwise equality of the arms."""
    keys = list(arms)
    for key in keys:
        for n in names:
            _within_budget(
                arms[key][n], ref[n], f"{what}: {key} {n} vs FP64", rel_rms=rel_rms
            )
    for i, a in enumerate(keys):
        for b in keys[i + 1 :]:
            for n in names:
                _within_budget(
                    arms[a][n], arms[b][n], f"{what}: {a} vs {b} {n}", rel_rms=rel_rms
                )


def _fp64(region, tensors, cotangent=None) -> tuple[torch.Tensor, list]:
    """FP64 autograd of ``region(*leaves)`` over FP64 copies of ``tensors``: ``(output, gradients)`` (``None`` for a
    leaf the output does not reach).  ``cotangent=None`` for a scalar output."""
    leaves = [t.detach().to(torch.float64).requires_grad_(True) for t in tensors]
    y = region(*leaves)
    cot = None if cotangent is None else cotangent.detach().to(torch.float64)
    grads = torch.autograd.grad(y, leaves, cot, allow_unused=True)
    return y.detach(), [None if gr is None else gr.detach() for gr in grads]


def _ref_layer(cu: list[int]):
    """The exact FP64 GDN layer over the pack ``cu`` (no batch axis, ``scale = 128 ** -0.5``), differentiable."""

    def layer(q, k, v, g, beta):
        return REF.gdn_exact_forward(q, k, v, g, beta, cu, scale=REF.SCALE)[0]

    return layer


# The one-rank pack of the host-orchestration tests (the C1 documents) and its FP64 references.
CKPT_CU = [0, 100, 740, 817, 1817, 1946]
CKPT_HEADS = (2, 6)
LEAVES = ("q", "k", "v", "g", "beta")
CKPT_GRADS = (
    "dq",
    "dk",
    "dv",
    "dg",
    "dbeta",
)  # autograd order of the leaves (q, k, v, g, beta)
EXPLICIT_GRADS = ("dq", "dk", "dv", "dbeta", "dg")  # the backward entries' output order
_REFERENCES: dict = {}


def _ckpt_reference(*, layers: int = 1) -> dict[str, torch.Tensor]:
    """FP64 ``out`` and the five gradients of ``layers`` chained exact layers (the next layer's ``v`` is the previous
    output, exact) on the seeded operands of :data:`CKPT_CU` with the seeded ``dout``; cached per layer count."""
    key = ("ckpt", layers)
    if key not in _REFERENCES:
        ops = _inputs(CKPT_CU, *CKPT_HEADS)
        layer = _ref_layer(CKPT_CU)

        def region(q, k, v, g, beta):
            out = layer(q, k, v, g, beta)
            for _ in range(layers - 1):
                out = layer(q, k, out, g, beta)
            return out

        out, grads = _fp64(region, [ops[n] for n in LEAVES], ops["dout"])
        _REFERENCES[key] = {"out": out, **dict(zip(CKPT_GRADS, grads, strict=True))}
    return _REFERENCES[key]


def _grads_dict(grads, order=EXPLICIT_GRADS) -> dict[str, torch.Tensor]:
    """The five gradients of an explicit backward entry (``dq, dk, dv, dbeta, dg``) keyed by name."""
    return dict(zip(order, grads, strict=True))


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
    """The ``cake_gdn_nondet_train_kcp_partition`` call of a row JSON (layout C / Z / D / R)."""
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


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


@ROWS
@pytest.mark.parametrize("path", ROW_FILES, ids=ROW_IDS)
def test_plan_parity_with_kda_on_rows(path, monkeypatch) -> None:
    """Result 1: ``cake_gdn_nondet_train_kcp.cake_kcp_plan`` is field-identical to ``cake_kda_det_train_kcp.cake_kcp_plan`` on
    every step-0 row (partition, every rank's context and head plan at ``H := HV``) and reproduces the row's recorded
    fragment table.  Result 2: ``n_in`` / ``n_out`` / ``max_chain_fragments`` and the exchange bytes of
    ``cake_gdn_nondet_train_kcp_comm_bytes`` equal the row JSON.  Both modules plan under the per-rank rule they share,
    ``"wave_fill1"`` (the GDN default is the row rule, the KDA default its own tie rule; each is selected explicitly
    here: the keyword-only ``p_rule`` on the GDN side, the private test hook on the KDA side)."""
    from tests.cake.test_helpers.cake_package import cake_package

    kp = cake_package("cake.cake_kda_det_train_kcp", "cake_kcp_plan")  # the test only
    # the KDA module accepts its private test hook only after its opt-in (enable_test_knobs)
    monkeypatch.setattr(kp, "_TEST_KNOBS_ENABLED", True)
    with open(path) as fh:
        row = json.load(fh)
    H_qk, HV = (int(x) for x in row["heads"])
    W = int(row["cp"])
    args = _partition_inputs(row)
    part = gp.cake_gdn_nondet_train_kcp_partition(**args)
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
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
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
        mine, theirs = _context_record(ctx, HV), _rename_h(_context_record(kctx, HV))
        records.append(mine)
        assert mine == theirs, (
            f"rank {r}: context / head plan differ from the KDA module (HV={HV})"
        )
        assert (ctx.n_in, ctx.n_out) == (n_in[r], n_out[r])
        cb = gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
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
    part = gp.cake_gdn_nondet_train_kcp_partition(**_partition_inputs(row))
    chains = part.chains()
    multi = [c for c in chains if len(c) >= 2]
    summary = []
    for r in range(W):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
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
    return gp.cake_gdn_nondet_train_kcp_partition(
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
        gp.cake_gdn_nondet_train_kcp_build_context(
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
        cb = gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=1, HV=HV, C=C_CONV)
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
        part = gp.cake_gdn_nondet_train_kcp_partition(
            cu, W, fragments=_zigzag_fragments(lengths, W)
        )
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, 0, CPU, 4, sm_count=SM_COUNT
        )
        ctx.prepare(1, 48)
        single = gp.cake_gdn_nondet_train_kcp_build_context(
            gp.cake_gdn_nondet_train_kcp_partition(cu, 1, "contiguous"),
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
    clone = gp.CakeGDNNondetKcpContext(
        **{f: getattr(ctx, f) for f in gp.CakeGDNNondetKcpContext._FIELDS}
    )
    assert clone != ctx and clone == clone and hash(clone) != hash(ctx)
    assert all(
        getattr(clone, f) is getattr(ctx, f) for f in gp.CakeGDNNondetKcpContext._FIELDS
    )
    assert gp.CakeGDNNondetKcpContext._FIELDS[-4:] == (
        "p_rule",
        "sequence",
        "_head_plans",
        "_caches",
    )
    fresh = gp.CakeGDNNondetKcpContext(
        *(getattr(ctx, f) for f in gp.CakeGDNNondetKcpContext._FIELDS[:-4])
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
        gp.CakeGDNNondetKcpContext(
            *(getattr(ctx, f) for f in gp.CakeGDNNondetKcpContext._FIELDS[:-3]), None
        ).sequence
        is None
    )
    text = repr(ctx)
    assert (
        text.startswith("CakeGDNNondetKcpContext(partition=")
        and f"p_rule={ctx.p_rule!r})" in text
    )
    assert "sequence=" not in text and "_caches" not in text


def test_prepare_tables_first_use() -> None:
    """``ctx.prepare(H_qk, HV)`` builds the device-resident bundle once per ``(HV, sm_count, P)``
    key at first use (``cake_gdn_nondet_train_kcp_build_context`` builds no head-dependent table; the second call returns the same
    object; a ``P`` override gets its own key and plan), rejects ``HV % H_qk != 0`` and an out-of-range ``P``, and
    every table dtype is int32 except the merge's int64 element offsets."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)  # C1
    part = gp.cake_gdn_nondet_train_kcp_partition(cu, 2, "contiguous")
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        cu, 4, fragments=_ragged_fragments([0, 1700, 3000], [1200, 0, 900, 900])
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
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
    part5 = gp.cake_gdn_nondet_train_kcp_partition(
        cu5, 4, fragments=_ragged_fragments([0, 692], [63, 64, 65, 500])
    )
    for r, want in enumerate((1, 1, 2, 8)):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part5, None, r, CPU, 4, sm_count=SM_COUNT
        )
        lt = gt.local_tables(ctx)
        assert want == lt.NC and lt.chunk_len[0] == min(64, ctx.T_r)
    with pytest.raises(ValueError, match="p_rule"):
        gp.cake_gdn_nondet_train_kcp_build_context(
            part5, None, 0, CPU, 4, sm_count=SM_COUNT, p_rule="nope"
        )


# --------------------------------------------------------------------------- the row rules ("row_cost_lpt" shipped, "row_cost" selectable)

# Row keys of ROW_COST_CHANGED / ROW_COST_LPT_CHANGED (_row_key): (row, cp, layout), the layout suffixed "_fla" on an
# FLA-padded row (the "_fla" row files): its fragment tables differ from those of its unpadded twin, and so may its vector.
ROW_COST_CHANGED = {  # (row, cp, layout) -> the P vector the "row_cost" row rule picks where it departs from "wave_fill1"
    ("P1", 2, "D"): (1, 1),
    ("P2", 2, "Z"): (4, 4),
    # P3 CP2 C: admitted by the "one" margin of 0 (model ratio 0.980; "occ" keeps its 2 %)
    ("P3", 2, "C"): (1, 1),
    ("P1", 4, "D"): (1, 1, 1, 1),
    # The four rows above are the departures on the unpadded HV = 48 rows at CP <= 4.  The departures on every other
    # step-0 row of record (HV 3 to 48, CP 2 to 16, the FLA-padded twins) follow, generated by a census of all 107 rows
    # at SM_COUNT from the plan module sha256/16 9a80bc5b7c6e525d, whose "row_cost" vectors equal those of the module
    # that recorded the four rows above; never retyped.  Each switches the whole row to "one", except
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
    # The FLA-padded twins, generated by the census of all 107 step-0 rows at SM_COUNT from the plan module sha256/16
    # 9a80bc5b7c6e525d; never retyped.  No other row departs.  ("P3", 4, "Z_fla") is not listed: its "row_cost" vector
    # is already P = 1 on every rank, which the rule keeps.
    ("P1", 4, "Z_fla"): (2, 2, 3, 3),
    ("P2", 4, "C_fla"): (3, 6, 3, 6),
    ("P3", 4, "C_fla"): (6, 6, 6, 6),
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
    """The key of a step-0 row in ``ROW_COST_CHANGED`` / ``ROW_COST_LPT_CHANGED``: ``(row, cp, layout)``, the layout
    suffixed ``"_fla"`` on an FLA-padded row."""
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
    part = gp.cake_gdn_nondet_train_kcp_partition(**_partition_inputs(row))
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
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        hp = ctx.heads(HV)
        assert ctx.p_rule == "row_cost" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        # only the segment program differs from the per-rank plan: partition, fragment table, local tables and
        # crossing lists are the same object graph values
        ref = gp.cake_gdn_nondet_train_kcp_build_context(
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
    """The shipped row rule (``"row_cost_lpt"``, the default of ``cake_gdn_nondet_train_kcp_build_context``) on every step-0 row: the
    rows of ``ROW_COST_LPT_CHANGED`` get their recorded vector, every other row keeps the ``"row_cost"`` vector; a rank
    the ``"row_cost"`` rule plans at ``P = 1`` keeps ``P = 1``; every rank's default context derives the same vector and
    reports ``p_rule="row_cost_lpt"``; only the segment program differs from the ``"row_cost"`` context."""
    with open(path) as fh:
        row = json.load(fh)
    HV = int(row["heads"][1])
    W = int(row["cp"])
    part = gp.cake_gdn_nondet_train_kcp_partition(**_partition_inputs(row))
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
    for r in range(W):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost_lpt"
        )
        hp = ctx.heads(HV)
        assert ctx.p_rule == "row_cost_lpt" == hp.segments.p_rule
        assert (got[r], plans[r].NSEG) == (hp.P, hp.segments.NSEG)
        ref = gp.cake_gdn_nondet_train_kcp_build_context(
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
    assert gp.P_RULE_DEFAULT == "fill_small_guard" and "row_cost_lpt" in gp.P_RULES
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
    part = gp.cake_gdn_nondet_train_kcp_partition(cu, 2, "contiguous")
    want = tuple(
        seg.P
        for seg in gp.kcp_plan_row(
            gp.row_tables(part), 6, SM_COUNT, p_rule=gp.P_RULE_DEFAULT
        )
    )
    want_old = tuple(
        seg.P
        for seg in gp.kcp_plan_row(gp.row_tables(part), 6, SM_COUNT, p_rule="row_cost")
    )
    for r in range(2):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        assert ctx.p_rule == gp.P_RULE_DEFAULT and want[r] == ctx.heads(6).P
        assert ctx.heads(6).segments.p_rule == gp.P_RULE_DEFAULT
        old = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="row_cost"
        )
        assert old.p_rule == "row_cost" and want_old[r] == old.heads(6).P
        assert old.heads(6).segments.p_rule == "row_cost"
        occ = gp.cake_gdn_nondet_train_kcp_build_context(
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


def test_graph_replay_context_takes_the_fill_down_plan() -> None:
    """``cake_gdn_nondet_train_kcp_build_context(..., graph_replay=True)`` plans with :data:`P_RULE_GRAPH` (``"fill_down_guard"``: the default
    ``"fill_small_guard"`` without its ``P = 1`` rows) on every rank; an explicit ``p_rule`` still wins; the default context
    keeps :data:`P_RULE_DEFAULT`; a non-bool hint is a ``TypeError``."""
    assert gp.P_RULE_GRAPH == "fill_down_guard" and gp.P_RULE_GRAPH in gp.ROW_RULES
    assert gp.P_RULE_GRAPH in gp.P_RULES
    # a small-head row of short fragments: the default walks every fragment as one chain, the graph plan keeps segments
    small = tuple(((128, True), (128, True), (128, True)) for _ in range(8))
    fs = [
        seg.P for seg in gp.kcp_plan_row(small, 6, SM_COUNT, p_rule=gp.P_RULE_DEFAULT)
    ]
    fd = [seg.P for seg in gp.kcp_plan_row(small, 6, SM_COUNT, p_rule=gp.P_RULE_GRAPH)]
    assert set(fs) == {1} and max(fd) > 1, (fs, fd)
    # every other row: the graph plan is the default plan
    for table, hv in (
        (small, 48),
        (tuple(((300, True), (40, False)) for _ in range(4)), 6),
    ):
        fs = [
            seg.P
            for seg in gp.kcp_plan_row(table, hv, SM_COUNT, p_rule=gp.P_RULE_DEFAULT)
        ]
        fd = [
            seg.P
            for seg in gp.kcp_plan_row(table, hv, SM_COUNT, p_rule=gp.P_RULE_GRAPH)
        ]
        assert fs == fd, (hv, fs, fd)
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    for W, layout in ((2, "contiguous"), (4, "zigzag_per_document")):
        part = gp.cake_gdn_nondet_train_kcp_partition(cu, W, layout)
        want = tuple(
            seg.P
            for seg in gp.kcp_plan_row(
                gp.row_tables(part), 6, SM_COUNT, p_rule=gp.P_RULE_GRAPH
            )
        )
        for r in range(W):
            ctx = gp.cake_gdn_nondet_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT, graph_replay=True
            )
            assert ctx.p_rule == gp.P_RULE_GRAPH
            assert ctx.heads(6).segments.p_rule == gp.P_RULE_GRAPH
            assert want[r] == ctx.heads(6).P, (W, layout, r)
            dflt = gp.cake_gdn_nondet_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT
            )
            assert dflt.p_rule == gp.P_RULE_DEFAULT
            mine = gp.cake_gdn_nondet_train_kcp_build_context(
                part,
                None,
                r,
                CPU,
                4,
                sm_count=SM_COUNT,
                p_rule="fill_small",
                graph_replay=True,
            )
            assert mine.p_rule == "fill_small"
    for bad in (1, None, "True"):
        with pytest.raises(TypeError, match="graph_replay must be a bool"):
            gp.cake_gdn_nondet_train_kcp_build_context(
                part, None, 0, CPU, 4, sm_count=SM_COUNT, graph_replay=bad
            )


def test_faster_schedule_rules_never_raise_a_ranks_P() -> None:
    """The non-deterministic package's faster-schedule row rules (``"rank_lpt"``, ``"fill_down"``, ``"small_one"``,
    ``"fill_small"``) start from the shipped ``"row_cost_lpt"`` vector and only lower a rank's ``P``: ``"fill_down"``
    takes ``min(row_cost_lpt, wave_fill)`` on the ranks below the chain cap and keeps the others, ``"small_one"`` runs
    ``P = 1`` on every rank exactly when ``HV <= SMALL_ONE_HV_MAX`` and no fragment of any rank exceeds
    ``SMALL_ONE_NC_MAX`` chunks (else the shipped vector), ``"fill_small"`` is ``"small_one"`` on those rows and
    ``"fill_down"`` elsewhere, and every plan reports its rule and ``NSEG = n_frag * P``."""
    rules = ("rank_lpt", "fill_down", "small_one", "fill_small")
    assert set(rules) <= set(gp.ROW_RULES) <= set(gp.P_RULES)
    tables = (
        TABLE_P1_CP4_D,
        TABLE_P1_CP2_C,
        TABLE_P3_CP2_C,
        (((130, True),) * 12,) * 8,  # (4,12)-like CP8 D: 12 short fragments per rank
        (((237, True), (237, True), (100, False)),) * 4,  # Z-like: 237-chunk fragments
        (((473, True), (473, False)),)
        * 4,  # above the chain cap: fill_down keeps the shipped P
        (((119, True), (119, True)),)
        * 8,  # (2,6)-like CP16 Z: two short fragments per rank
        ((), ((150, True), (20, False))),  # a zero-token rank
    )
    lowered = 0  # (table, HV) cases where small_one moves a shipped P > 1 to 1
    for table in tables:
        for HV in (6, 12, 24, 48):
            ship = gp.kcp_plan_row(table, HV, SM_COUNT, p_rule="row_cost_lpt")
            for rule in rules:
                plans = gp.kcp_plan_row(table, HV, SM_COUNT, p_rule=rule)
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
                        SM_COUNT,
                        p_rule="wave_fill",
                    ).P,
                )
                if 0 < max((nc for nc, _ in t), default=0) < gp.WAVE_MIN_NC_MAX_FRAG
                else base.P
                for base, t in zip(ship, table, strict=True)
            )
            assert fill == tuple(
                seg.P
                for seg in gp.kcp_plan_row(table, HV, SM_COUNT, p_rule="fill_down")
            )
            small = HV <= gp.SMALL_ONE_HV_MAX and all(
                nc <= gp.SMALL_ONE_NC_MAX for t in table for nc, _ in t
            )
            got = tuple(
                seg.P
                for seg in gp.kcp_plan_row(table, HV, SM_COUNT, p_rule="small_one")
            )
            assert got == (
                tuple(1 for _ in table) if small else tuple(seg.P for seg in ship)
            )
            both = tuple(
                seg.P
                for seg in gp.kcp_plan_row(table, HV, SM_COUNT, p_rule="fill_small")
            )
            assert both == (tuple(1 for _ in table) if small else fill)
            lowered += int(small and max(seg.P for seg in ship) > 1)
    assert lowered > 0  # the rule changes some shipped plan (else it would be a no-op)


def test_fill_guard_keeps_row_cost_lpt_for_a_long_one_chain_fragment() -> None:
    """The shipped rules ``"fill_small_guard"`` (eager) and ``"fill_down_guard"`` (graph replay) are ``"fill_small"`` /
    ``"fill_down"`` plus a row guard on SM counts of at least ``FILL_GUARD_SM_MIN``: on a row whose every rank holds at
    least two fragments, when the fill sends a rank to P = 1 below its ``"row_cost_lpt"`` P, that rank's longest
    fragment has at least ``FILL_GUARD_NC_MIN`` chunks and the fill's slowest-rank B3 walk steps exceed those of the
    ``"row_cost_lpt"`` vector, the row keeps the ``"row_cost_lpt"`` vector.  Below the guard's SM range (B200 148,
    GB300 152) every plan equals the unguarded rule's; on R200 (212 / 216 SMs) the K3E zigzag CP8 row (third rank:
    three 256-chunk fragments) and the quarter / half K3E zigzag rows (CP4 / CP2) keep ``"row_cost_lpt"``, while the DET
    64k zigzag CP4 row (128-chunk fragments), the single-document zigzag row (no P = 1 rank) and the half K3E contiguous
    CP2 row (the fill walks fewer steps) keep the one-wave fill."""
    pairs = (("fill_small_guard", "fill_small"), ("fill_down_guard", "fill_down"))
    assert (
        gp.P_RULE_DEFAULT == "fill_small_guard" and gp.P_RULE_GRAPH == "fill_down_guard"
    )
    assert {r for pair in pairs for r in pair} <= set(gp.ROW_RULES) <= set(gp.P_RULES)
    assert 152 < gp.FILL_GUARD_SM_MIN <= 212 and 128 < gp.FILL_GUARD_NC_MIN <= 237
    k3e = torch.tensor([0, 98304, 163840, 221184, 262144], dtype=torch.int64)
    k3e_z = gp.row_tables(gp.cake_gdn_nondet_train_kcp_partition(k3e, 8, "zigzag"))
    assert [len(t) for t in k3e_z] == [2, 2, 3, 2, 2, 2, 2, 2]
    det3 = torch.tensor([0, 15945, 37790, 65536], dtype=torch.int64)
    det3_z4 = gp.row_tables(gp.cake_gdn_nondet_train_kcp_partition(det3, 4, "zigzag"))
    assert [len(t) for t in det3_z4] == [2, 3, 2, 3]
    one_doc_z = gp.row_tables(
        gp.cake_gdn_nondet_train_kcp_partition(torch.tensor([0, 254976]), 8, "zigzag")
    )
    k3e4 = torch.tensor([0, 49152, 81920, 110592, 131072], dtype=torch.int64)
    k3e4_z4 = gp.row_tables(gp.cake_gdn_nondet_train_kcp_partition(k3e4, 4, "zigzag"))
    k3e2 = torch.tensor([0, 24576, 40960, 55296, 65536], dtype=torch.int64)
    k3e2_z2 = gp.row_tables(gp.cake_gdn_nondet_train_kcp_partition(k3e2, 2, "zigzag"))
    k3e2_c2 = gp.row_tables(
        gp.cake_gdn_nondet_train_kcp_partition(k3e2, 2, "contiguous")
    )
    assert [len(t) for t in k3e2_c2] == [2, 3]
    tables = (
        k3e_z,
        det3_z4,
        one_doc_z,
        k3e4_z4,
        k3e2_z2,
        k3e2_c2,
        TABLE_P1_CP4_D,
        TABLE_P1_CP2_C,
        TABLE_P3_CP2_C,
        (((130, True),) * 12,) * 8,
        (((119, True), (119, True)),) * 8,
        ((), ((150, True), (20, False))),
    )

    def slowest(table, HV, sm, P):
        return max(
            (
                gp.walk_steps_rank_b3(t, HV, sm, p)
                for t, p in zip(table, P, strict=True)
                if t
            ),
            default=0,
        )

    def Ps(table, HV, sm, rule):
        plans = gp.kcp_plan_row(table, HV, sm, p_rule=rule)
        assert all(
            seg.p_rule == rule and seg.n_frag * seg.P == seg.NSEG for seg in plans
        )
        return tuple(seg.P for seg in plans)

    guarded_rows = 0
    for table in tables:
        for HV in (6, 12, 24, 48):
            for sm in (148, 152, 212, 216):
                ship = Ps(table, HV, sm, "row_cost_lpt")
                one = HV <= gp.SMALL_ONE_HV_MAX and all(
                    nc <= gp.SMALL_ONE_NC_MAX for t in table for nc, _ in t
                )
                for guarded, plain in pairs:
                    got, fill = Ps(table, HV, sm, guarded), Ps(table, HV, sm, plain)
                    assert all(1 <= p <= q for p, q in zip(got, ship, strict=True))
                    # the small_one rows (the det twin's default as well) are not guarded
                    keep = (
                        sm >= gp.FILL_GUARD_SM_MIN
                        and not (guarded == "fill_small_guard" and one)
                        and all(len(t) >= 2 for t in table)
                        and any(
                            f == 1 < q
                            and max(nc for nc, _ in t) >= gp.FILL_GUARD_NC_MIN
                            for f, q, t in zip(fill, ship, table, strict=True)
                        )
                        and slowest(table, HV, sm, fill) > slowest(table, HV, sm, ship)
                    )
                    assert got == (ship if keep else fill), (guarded, sm, HV)
                    guarded_rows += keep and fill != ship
    assert guarded_rows > 0  # the guard changes some plan (else it would be a no-op)
    for sm in (212, 216):
        assert Ps(k3e_z, 48, sm, "fill_small") == (2, 2, 1, 2, 2, 2, 2, 2)
        assert Ps(k3e_z, 48, sm, "row_cost_lpt") == (3, 3, 2, 3, 3, 3, 3, 3)
        for guarded, plain in pairs:
            assert Ps(k3e_z, 48, sm, guarded) == (3, 3, 2, 3, 3, 3, 3, 3)
            assert Ps(det3_z4, 48, sm, guarded) == Ps(det3_z4, 48, sm, plain)
            assert Ps(k3e4_z4, 48, sm, plain) == (2, 1, 2, 2)
            assert Ps(k3e4_z4, 48, sm, guarded) == (3, 2, 3, 3)
            assert Ps(k3e2_z2, 48, sm, plain) == (1, 2)
            assert Ps(k3e2_z2, 48, sm, guarded) == (2, 2)
            # a P = 1 rank with a 224-chunk fragment, but the fill walks fewer steps than "row_cost_lpt" (3, 2)
            assert Ps(k3e2_c2, 48, sm, "row_cost_lpt") == (3, 2)
            assert Ps(k3e2_c2, 48, sm, guarded) == (3, 1) == Ps(k3e2_c2, 48, sm, plain)
            assert (
                Ps(one_doc_z, 48, sm, guarded)
                == (2,) * 8
                == Ps(one_doc_z, 48, sm, "fill_down")
            )
            assert Ps(one_doc_z, 48, sm, "row_cost_lpt") == (3,) * 8
    for sm in (148, 152):
        assert (
            Ps(k3e_z, 48, sm, "fill_small_guard")
            == (1,) * 8
            == Ps(k3e_z, 48, sm, "row_cost_lpt")
        )
    want = (3, 3, 2, 3, 3, 3, 3, 3)
    for r in range(8):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            gp.cake_gdn_nondet_train_kcp_partition(k3e, 8, "zigzag"),
            None,
            r,
            CPU,
            4,
            sm_count=212,
        )
        assert ctx.p_rule == "fill_small_guard" and want[r] == ctx.heads(48).P


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
    """Result 3: the byte queries follow the design formulas for every ``save_mode`` and ``boundary_mode`` value,
    ``save_transfer`` on / off (``"light_akk"`` with the in-kernel norm, which it needs: the light figures of the
    norm plus #1's ``Akk`` tape in the saved set of a saved boundary and out of the arenas); the exchange terms
    depend on ``HV`` only, ``H_qk`` enters through the default channel count; the validators reject
    ``HV % H_qk != 0``."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)  # C1
    part = gp.cake_gdn_nondet_train_kcp_partition(cu, W, "contiguous")
    H_qk, HV = 2, 6
    s = HV * gp.STATE_TILE_BYTES
    records = []
    b1_sums = {"design": 0, "buffers": 0}
    for r in range(W):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        hp = ctx.heads(HV)
        cb = gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=H_qk, HV=HV, C=C_CONV)
        assert (
            cb["C"] == C_CONV
            and gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=16, HV=48)["C"]
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
        maps = gb.cake_gdn_nondet_train_kcp_comm_bytes(
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
        replay = gb.cake_gdn_nondet_train_kcp_comm_bytes(
            ctx, H_qk=H_qk, HV=HV, C=C_CONV, save_transfer=False, boundary_mode="replay"
        )
        assert replay["b1"] == cb["b1_replay"]
        # the exchange terms are HV-only: H_qk = 1, 2, 3, 6 give the same c1 / c2 / b1 / b2
        for hq in (1, 3, 6):
            other = gb.cake_gdn_nondet_train_kcp_comm_bytes(
                ctx, H_qk=hq, HV=HV, C=C_CONV
            )
            assert all(other[k] == cb[k] for k in ("c1", "c2", "b1", "b2"))
        planes = 0  # kept-form rows that hand the M_seg^T plane over
        akk = ctx.R * HV * 128  # #1's BF16 Akk tape
        for save_mode in gb.SAVE_MODES:
            # the in-kernel-norm-only modes ("light_akk") refuse a norm-off call: their rows run with the norm on
            norm = {}
            if save_mode in gb.AKK_SAVE_MODES:
                with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
                    gb.cake_gdn_nondet_train_kcp_saved_bytes(
                        ctx, H_qk=H_qk, HV=HV, save_mode=save_mode, C=C_CONV
                    )
                with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel"):
                    gb.cake_gdn_nondet_train_kcp_workspace_layout(
                        ctx, H_qk=H_qk, HV=HV, save_mode=save_mode, C=C_CONV
                    )
                norm = {"use_qk_l2norm_in_kernel": True}
            for boundary_mode in gb.BOUNDARY_MODES:
                for save_transfer in (True, False):
                    got = gb.cake_gdn_nondet_train_kcp_saved_bytes(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                        **norm,
                    )
                    if norm:
                        # the light set of the norm plus the Akk tape (a saved boundary; the replay keeps none)
                        want = gb.cake_gdn_nondet_train_kcp_saved_bytes(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode="light",
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                            **norm,
                        ) + (akk if boundary_mode == "saved" else 0)
                    else:
                        want = _saved_formula(
                            ctx,
                            H_qk,
                            HV,
                            save_mode,
                            boundary_mode,
                            save_transfer,
                            C_CONV,
                        )
                    assert got == want, (r, save_mode, boundary_mode, save_transfer)
                    layout = gb.cake_gdn_nondet_train_kcp_saved_layout(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                        **norm,
                    )
                    assert sum(layout.values()) == got and all(
                        v >= 0 for v in layout.values()
                    )
                    ws = gb.cake_gdn_nondet_train_kcp_workspace_layout(
                        ctx,
                        H_qk=H_qk,
                        HV=HV,
                        save_mode=save_mode,
                        boundary_mode=boundary_mode,
                        save_transfer=save_transfer,
                        C=C_CONV,
                        **norm,
                    )
                    assert (
                        ws["workspace"]
                        == max(ws["forward"], ws["backward"])
                        == gb.cake_gdn_nondet_train_kcp_workspace_bytes(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode=save_mode,
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                            **norm,
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
                            gb.cake_gdn_nondet_train_kcp_workspace_layout(
                                ctx,
                                H_qk=H_qk,
                                HV=HV,
                                save_mode=save_mode,
                                boundary_mode=boundary_mode,
                                save_transfer=save_transfer,
                                C=C_CONV,
                                P=P_q,
                                reuse_recompute_tapes=form,
                                **norm,
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
                    elif save_mode in gb.AKK_SAVE_MODES and boundary_mode == "saved":
                        # the light form of the norm, but #1's Akk goes to the saved set: no fwd.akk / bwd.akk
                        # term, no h tape, #1's moment and every backward phase lower by the tape
                        light_ws = gb.cake_gdn_nondet_train_kcp_workspace_layout(
                            ctx,
                            H_qk=H_qk,
                            HV=HV,
                            save_mode="light",
                            boundary_mode=boundary_mode,
                            save_transfer=save_transfer,
                            C=C_CONV,
                            **norm,
                        )
                        assert "fwd.akk" not in ws and "bwd.akk" not in ws
                        assert light_ws["fwd.akk"] == light_ws["bwd.akk"] == akk
                        assert "fwd.h_tape" not in ws and layout["h_tape"] == 0
                        assert layout["akk"] == akk
                        assert light_ws["fwd.phase.prep"] - ws["fwd.phase.prep"] == akk
                        akk_phases = [k for k in light_ws if k.startswith("bwd.phase.")]
                        assert akk_phases == [
                            k for k in ws if k.startswith("bwd.phase.")
                        ]
                        for k in akk_phases:
                            assert light_ws[k] - ws[k] == akk, (r, k, save_transfer)
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
            gb.cake_gdn_nondet_train_kcp_saved_bytes(
                ctx, H_qk=H_qk, HV=HV, save_mode=m, C=C_CONV
            )
            for m in ("light", "full")
        )
        assert full - light == ctx.NC_pad * HV * (32768 + 4) + ctx.R * HV * (
            3 * 256 + 2 * 128 + 3 * 4
        )  # h + gamma, v_new / w / u, Aqk / Akk, gcum / rowscale / qscale
        host = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
                "workspace_light": gb.cake_gdn_nondet_train_kcp_workspace_bytes(
                    ctx, H_qk=H_qk, HV=HV, C=C_CONV
                ),
            }
        )
    with pytest.raises(ValueError, match="multiple"):
        gb.cake_gdn_nondet_train_kcp_comm_bytes(ctx, H_qk=4, HV=6)
    with pytest.raises(ValueError, match="save_mode"):
        gb.cake_gdn_nondet_train_kcp_saved_bytes(ctx, H_qk=2, HV=6, save_mode="tape")
    with pytest.raises(ValueError, match="boundary_mode"):
        gb.cake_gdn_nondet_train_kcp_saved_bytes(
            ctx, H_qk=2, HV=6, boundary_mode="none"
        )
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
    """The ``cake_jit`` seed: the shipped records of the nine programs and the two output walks (one library per record and exported arch
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
            "kcp_fwdh_out_tape",
            "kcp_dhu_offset",
            "kcp_dhu_tape",
            "kcp_affine_merge",
        )
    )  # the vh=64 forms of the state walks #2 (L8A, the output walks too) / #7 and the c8 / c16 merges are the only
    # multi-record programs
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
            rf"cake_gdn_train_[0-9a-f]{{20}}{suffix}", rec["library"]
        ), name
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
    # the library names are this package's own: never a name of the deterministic twin's registry
    assert not any(
        rec["library"].startswith("cake_gdn_det_train")
        for rec in {**gj.MODULES, **gj.TEST_MODULES}.values()
    )
    # the rendered manifests: one library per shipped record and exported arch; the test-only walks likewise
    if not manifest_rendered or not (PACKAGE_DIR / "cake_build.py").is_file():
        pytest.skip(
            "build.json / cake_build.py are not rendered into the package yet (the registry checks above ran)"
        )
    manifest = json.loads((PACKAGE_DIR / "build.json").read_text())
    assert (
        manifest["package"] == "flashinfer.experimental.cake.cake_gdn_nondet_train_kcp"
    )
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
    """The package never imports a KDA package (no module path or name containing ``cake_kda``) and reads no
    environment variable; the plan module imports torch, torch.distributed and dataclasses only."""
    for path in sorted(PACKAGE_DIR.glob("*.py")):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = [a.name for a in node.names] + (
                    [node.module]
                    if isinstance(node, ast.ImportFrom) and node.module
                    else []
                )
                assert not any("cake_kda" in n for n in names), (
                    f"{path.name} imports a KDA package"
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


@REFERENCE
@pytest.mark.parametrize("layout", ("contiguous", "zigzag"))
def test_fake_backend_matches_fp64_reference(layout) -> None:
    """Result 4: the fake backend's forward / backward of a 2-rank C and Z row at (H_qk, HV) = (2, 6), T = 507,
    matches the FP64 reference within FLA-class error (rel-RMS <= 0.05 per tensor, finite); a second run, the replay
    boundary, ``save_transfer=False`` and ``save_mode="full"`` are each within the same budget of the reference and of
    the first run.  The emulation's own saved == replay identity (a property of the CPU fake backend, not of the
    kernels) is the one bitwise check kept."""
    H_qk, HV = 2, 6
    cu = FAKE_CU
    part = gp.cake_gdn_nondet_train_kcp_partition(
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
    # FAKE-BACKEND BITWISE PROPERTY (the CPU emulation itself, not the kernels): one device and thread count replay
    # the same programs on the same inputs, so its saved and replay gradients are the same bits.
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
    # a second run and the other saved-set forms: each within the budget of the reference and of the first run
    again = fake.emulate(part, ops, H_qk=H_qk, HV=HV)
    no_transfer = fake.emulate(part, ops, H_qk=H_qk, HV=HV, save_transfer=False)
    full = fake.emulate(part, ops, H_qk=H_qk, HV=HV, save_mode="full")
    names = ("out",) + fake.GRAD_NAMES
    _arms_within_budget(
        {
            "first": res["saved"],
            "repeat": again["saved"],
            "replay": {"out": res["saved"]["out"], **res["replay"]},
            "save_transfer=False": no_transfer["saved"],
            "save_transfer=False replay": {
                "out": no_transfer["saved"]["out"],
                **no_transfer["replay"],
            },
            "save_mode=full": full["saved"],
            "save_mode=full replay": {"out": full["saved"]["out"], **full["replay"]},
        },
        {n: ref[n] for n in names},
        names,
        f"fake backend {layout}",
    )
    # the no-CP form (one rank) agrees with the reference too and documents the local error floor
    single = fake.emulate(
        gp.cake_gdn_nondet_train_kcp_partition(
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
            "saved_eq_replay_fake_backend_bitwise": True,
            "repeat_within_budget": True,
            "save_transfer_false_within_budget": True,
            "save_mode_full_within_budget": True,
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
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
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
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


def test_dhu_offset_walk_reads_the_cut_table_its_start_launch_writes() -> None:
    """Every #7 offset walk is issued right after one ``kcp_dhu_cut_start`` launch over the same walk list and reads
    the ``c_start`` table that launch wrote (each entry in ``-1 .. nc - 1``); the gradients equal those of the same
    backward with every offset walk given a null table (every chunk walked).  CP = 1 with two segments per fragment
    (P = 2) in the normal gate regime of ``_inputs`` (some chains cut), fake backend on the CPU."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    seen: list[dict] = []

    class Recording(kcp.FakeBackend):
        whole = False

        def run(self, name, args):
            if name == "kcp_dhu_offset" and self.whole:
                args = {**args, "c_start": None}
            super().run(name, args)
            if name in ("kcp_dhu_cut_start", "kcp_dhu_offset"):
                table = args.get("c_start")
                seen.append(
                    {
                        "name": name,
                        "walk": args["walk_seg"].tolist(),
                        "seg": args["seg_chunk_start"].tolist(),
                        "ptr": None if table is None else table.data_ptr(),
                        "table": None if table is None else table.tolist(),
                    }
                )

    def backward(whole: bool):
        backend = Recording()
        backend.whole = whole
        previous = kcp._BACKENDS.get("fake")
        kcp._BACKENDS["fake"] = backend
        try:
            _, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                ops["q"],
                ops["k"],
                ops["v"],
                ops["g"],
                ops["beta"],
                ctx,
                backend="fake",
                segments=2,
            )
            return kcp.cake_gdn_nondet_train_kcp_backward(
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

    cut_grads = backward(whole=False)
    cut_seen = list(seen)
    seen.clear()
    whole_grads = backward(whole=True)
    offsets = [i for i, s in enumerate(cut_seen) if s["name"] == "kcp_dhu_offset"]
    assert offsets, [s["name"] for s in cut_seen]
    cut = 0
    for i in offsets:
        start, walk = cut_seen[i - 1], cut_seen[i]
        assert i > 0 and start["name"] == "kcp_dhu_cut_start", i
        assert start["walk"] == walk["walk"], (start["walk"], walk["walk"])
        assert walk["ptr"] is not None and start["ptr"] == walk["ptr"], (
            "the walk reads another table"
        )
        for s in walk["walk"]:
            nc = walk["seg"][s + 1] - walk["seg"][s]
            for h in range(HV):
                c_start = walk["table"][s * HV + h]
                assert -1 <= c_start <= nc - 1, (s, h, c_start, nc)
                cut += int(c_start + 1 < nc)
    assert cut > 0, (
        "no chain was cut: the seeded normal regime of _inputs should cut some on the fake"
    )
    for a, b in zip(cut_grads, whole_grads, strict=True):
        assert torch.equal(a, b), (
            "the cut walk's gradients differ from the whole walk's"
        )


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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
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
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
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
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, CPU, sm_count=SM_COUNT
    )
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.CakeGDNNondetKcpWorkspace()
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
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
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
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            ops["dout"],
            **bwd,
        )
        n_rule = len(walks)
        grads2 = kcp.cake_gdn_nondet_train_kcp_backward(
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
    # the rule's #7 form and the forced full-head form: each within the budget of the reference and of each other
    assert cu == CKPT_CU and (H_qk, HV) == CKPT_HEADS
    ref = _ckpt_reference()
    _within_budget(out, ref["out"], "ticket row: out vs FP64")
    _arms_within_budget(
        {"rule form": _grads_dict(grads), "dhu_vh=128": _grads_dict(grads2)},
        ref,
        EXPLICIT_GRADS,
        "#7 forms",
    )
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    assert set(kcp.RING_POOLS) == {"ring_out", "ring_in"}
    assert set(kcp.SAVE_TRANSFER_FRESH_POOLS) == {"c1_recv", "stage_m"}
    assert set(kcp.RING_POOLS) <= set(kcp.FWD_DEAD_POOLS) and set(
        kcp.RING_POOLS
    ) <= set(kcp.BWD_DEAD_BEFORE_TAPES)
    dead = set(kcp.RING_POOLS) | set(kcp.SAVE_TRANSFER_FRESH_POOLS)
    # the full workspace of the public constructor sizes every pool (the caller-held form)
    full = kcp.CakeGDNNondetKcpWorkspace(ctx, H_qk, HV, device="cpu", backward=True)
    assert dead <= set(full._pools)
    # a2a + save_transfer (shipped defaults): the dead pools are absent after the forward and after the backward
    ws = kcp.CakeGDNNondetKcpWorkspace()
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
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
    grads = kcp.cake_gdn_nondet_train_kcp_backward(
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
    kcp.cake_gdn_nondet_train_kcp_forward(
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
    ws2 = kcp.CakeGDNNondetKcpWorkspace()
    kcp.cake_gdn_nondet_train_kcp_backward(
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
    seen_sizes: list[dict[str, int]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            if not seen_pools or seen_pools[-1] is None:
                call = args["_call"]
                seen_pools.append(set(call.ws._pools))
                seen_sizes.append(dict(call.tb.pool_sizes(call.backward)))
            super().run(name, args)

    previous = kcp._BACKENDS.get("fake")
    kcp._BACKENDS["fake"] = Recording()
    try:
        kcp.cake_gdn_nondet_train_kcp_forward(
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
        kcp.cake_gdn_nondet_train_kcp_forward(
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
    # the per-call workspace sizes no empty pool (one rank exchanges nothing: c1_send, and under the ring c1_recv and
    # the ring pools, are empty; pool() makes an empty one on demand)
    sizes = seen_sizes[0]
    a2a_live = {n for n in ("c1_send", "stage_b", "carry", "eye") if sizes.get(n, 1)}
    assert not (dead & a2a_pools) and a2a_live <= a2a_pools, sorted(a2a_pools)
    assert {"stage_b", "carry", "eye"} <= a2a_live
    empty = {n for n in a2a_pools if sizes.get(n, 1) == 0}
    assert not empty, sorted(empty)
    ring_sizes = seen_sizes[1]  # seen_pools[1] is the separator (no sizes)
    ring_live = {
        n for n in (*kcp.RING_POOLS, "c1_recv", "stage_m") if ring_sizes.get(n, 1)
    }
    assert ring_live <= ring_pools and "stage_m" in ring_live, sorted(ring_pools)
    empty = {n for n in ring_pools if ring_sizes.get(n, 1) == 0}
    assert not empty, sorted(empty)
    # ensure(skip=...) semantics on a bare workspace
    ws4 = kcp.CakeGDNNondetKcpWorkspace()
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
    other forward pool, releases it at the end of the eager call); the outputs are within the accuracy budget of the
    FP64 reference and of each other either way, and the saved planes agree within the same budget.  The receive planes (``S_in``, ``c1_recv``) are not allocated yet when the summaries' walks run (the
    forward's peak moment)."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    at_summaries: list[set[str]] = []
    summaries = kcp._summaries

    def recording_summaries(call):
        at_summaries.append({"S_in", "c1_recv"} & set(call.bufs))
        summaries(call)

    monkeypatch.setattr(kcp, "_summaries", recording_summaries)
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
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
    full = kcp.CakeGDNNondetKcpWorkspace(ctx, H_qk, HV, device="cpu", backward=False)
    assert "c2_send" in full._pools
    out2, saved2 = kcp.cake_gdn_nondet_train_kcp_forward(
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
    assert cu == CKPT_CU and (H_qk, HV) == CKPT_HEADS
    _arms_within_budget(
        {"per-call workspace": {"out": out}, "caller workspace": {"out": out2}},
        _ckpt_reference(),
        ("out",),
        "eager forward",
    )
    for name in ("S_in", "c1_recv"):
        a, b = getattr(saved, name), getattr(saved2, name)
        assert a.dtype == b.dtype, name
        _within_budget(b, a, f"saved {name}: caller vs per-call workspace")


def test_backward_forgets_its_own_saved_set_at_the_last_reader() -> None:
    """A caller-owned saved set is never mutated by ``cake_gdn_nondet_train_kcp_backward`` (two backwards from it keep its tensors and
    are each within the accuracy budget of the FP64 reference and of each other); a set marked ``release_in_backward``
    (the autograd Function's own) comes back with ``S_in``, ``c1_recv`` and ``M_seg`` forgotten, its gradients within
    the same budget; the per-call buffers of a replay backward (``saved=None``) are dropped the same way."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    _, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, backend="fake", segments=2)
    assert saved is not None and saved.c1_recv is not None and saved.M_seg is not None
    assert saved.release_in_backward is False
    g1 = kcp.cake_gdn_nondet_train_kcp_backward(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    assert (
        saved.S_in is not None and saved.c1_recv is not None and saved.M_seg is not None
    )
    g2 = kcp.cake_gdn_nondet_train_kcp_backward(
        *args, ops["dout"], saved=saved, backend="fake", segments=2
    )
    owned = dataclasses.replace(saved, release_in_backward=True)
    g3 = kcp.cake_gdn_nondet_train_kcp_backward(
        *args, ops["dout"], saved=owned, backend="fake", segments=2
    )
    assert owned.S_in is None and owned.c1_recv is None and owned.M_seg is None
    assert (
        saved.S_in is not None
    )  # the original set is untouched (replace copied the references)
    assert not (
        {"S_in", "c1_recv", "stage_m"} & set(kcp.LAST_CALL["bwd"]["fresh_bytes"])
    )
    with pytest.raises(KeyError):
        owned.forget("tapes")
    g4 = kcp.cake_gdn_nondet_train_kcp_backward(
        *args,
        ops["dout"],
        saved=None,
        backend="fake",
        segments=2,
        boundary_mode="replay",
    )
    assert all(torch.isfinite(g.float()).all() for g in g4)
    assert "S_in" not in kcp.LAST_CALL["bwd"]["fresh_bytes"]
    assert cu == CKPT_CU and (H_qk, HV) == CKPT_HEADS
    _arms_within_budget(
        {
            "first": _grads_dict(g1),
            "second": _grads_dict(g2),
            "released": _grads_dict(g3),
            "replay": _grads_dict(g4),
        },
        _ckpt_reference(),
        EXPLICIT_GRADS,
        "backwards from one saved set",
    )


# README saved-set table row (by its item cell's start) -> the cake_gdn_nondet_train_kcp_saved_layout terms it states
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
    """The README's saved-set table states what ``cake_gdn_nondet_train_kcp_saved_layout`` counts: every layout term sits in one
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(part, None, r, CPU) for r in range(2)
    ]
    assert sum(c.n_in for c in ctxs) > 0  # a crossing fragment: S_in / hist terms exist

    def terms(save_mode: str, save_transfer: bool) -> list[dict[str, int]]:
        return [
            gb.cake_gdn_nondet_train_kcp_saved_layout(
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor([0, 4096], dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    assert 64 * ctx.NC_pad == ctx.R
    lay = gb.cake_gdn_nondet_train_kcp_saved_layout(
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
    assert rows["`M_seg`"][2] == "`P * n_frag * s`"
    assert rows["`S_in`"][2] == "`n_in * s_pad`"
    assert rows["received summary pairs"][2] == "`2 * s_owner * N_out`"
    # a padded owner (HV = 48 on five ranks: Hg = 10, five owners, the last holds 8 heads and 2 padding tiles): S_in
    # is n_in s_pad (s_pad = n_owners s_owner, above s) and every owner receives 2 s_owner per crossing-out fragment
    # of the group
    part5 = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor([0, 4096], dtype=torch.int64), 5, "contiguous"
    )
    ctxs5 = [
        gp.cake_gdn_nondet_train_kcp_build_context(part5, None, r, CPU)
        for r in range(5)
    ]
    assert all(c.N_out > 0 for c in ctxs5) and any(c.n_in > 0 for c in ctxs5)
    for c in ctxs5:
        hp = c.heads(48)
        assert (hp.Hg, hp.n_owners, hp.pad_tiles) == (10, 5, 2)
        s_owner = hp.Hg * gp.STATE_TILE_BYTES
        s_pad = hp.n_owners * s_owner
        assert s_pad == (48 + 2) * gp.STATE_TILE_BYTES
        lay5 = gb.cake_gdn_nondet_train_kcp_saved_layout(
            c, H_qk=16, HV=48, save_transfer=True
        )
        assert lay5["S_in"] == c.n_in * s_pad, (c.rank, lay5)
        assert lay5["c1_recv"] == 2 * s_owner * c.N_out, (c.rank, lay5)
    for c in ctxs:
        s = 6 * gp.STATE_TILE_BYTES
        lay2 = gb.cake_gdn_nondet_train_kcp_saved_layout(
            c, H_qk=H_qk, HV=HV, P=2, use_qk_l2norm_in_kernel=True
        )
        assert lay2["qk_rstd"] == 8 * c.T_r * H_qk
        assert lay2["M_seg"] == 2 * c.n_frag * s and lay2["S_in"] == c.n_in * s
        # the replay boundary keeps no boundary state, M_seg, tape or qk_rstd (only the convolution rows remain)
        for save_mode in ("light", "full"):
            rep = gb.cake_gdn_nondet_train_kcp_saved_layout(
                c,
                H_qk=H_qk,
                HV=HV,
                P=2,
                save_mode=save_mode,
                boundary_mode="replay",
                use_qk_l2norm_in_kernel=True,
            )
            assert {k for k, v in rep.items() if v} <= {"hist"}, (save_mode, rep)
    assert set(FULL_TAPE_TERMS) == {
        k for k in keys if k not in ("S_in", "c1_recv", "M_seg", "hist", "qk_rstd")
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
    light form's bits (the fake backend is deterministic) on a ``None``, a fresh and a shared workspace (twice) and
    under a non-reentrant checkpoint; the byte query counts ``M_seg``; a full saved set without it is refused."""
    kcp = _package("cake_kcp")
    _nan_filled_allocations(monkeypatch, kcp)
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ops = _inputs(cu, H_qk, HV)
    base = {
        "segments": 2,
        "save_transfer": False,
        "use_qk_l2norm_in_kernel": use_qk_l2norm_in_kernel,
    }
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    _, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        *args, backend="fake", save_mode="full", **base
    )
    assert saved is not None and saved.P == 2 and saved.c1_recv is None
    assert saved.M_seg is not None and torch.isfinite(saved.M_seg).all()
    m_seg_bytes = gb.cake_gdn_nondet_train_kcp_saved_layout(
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
        kcp.cake_gdn_nondet_train_kcp_backward(
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
    shared = kcp.CakeGDNNondetKcpWorkspace()
    arms = (
        (None, False),
        (kcp.CakeGDNNondetKcpWorkspace(), False),
        (shared, False),
        (shared, False),
        (None, True),
    )
    for i, (ws, ckpt) in enumerate(arms):
        full = _ckpt_autograd(
            kcp, ctx, ops, ckpt=ckpt, workspace=ws, save_mode="full", **base
        )
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved", i
        for n in light:
            assert torch.equal(full[n], light[n]), (i, n)


def test_transpose_tiles_scratch_is_bitwise_the_temporary_form() -> None:
    """R2-L1 1a: the ``M_f -> M_f^T`` scatter with the gather in a caller scratch (the carry plane during the
    summaries) writes the same bytes as the fresh-temporary form and leaves the gathered tiles in the scratch.  Exact:
    a tile transpose moves bytes (no arithmetic, no reduction), so it has no non-deterministic form."""
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
    assert "scratch" in kcp.CakeGDNNondetKcpCall.transpose_tiles.__kwdefaults__


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
        _, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, backend="fake", **opts)
        phase[0] = "bwd"
        kcp.cake_gdn_nondet_train_kcp_backward(
            *args,
            ops["dout"],
            saved=saved,
            backend="fake",
            **opts,
        )
    finally:
        kcp.set_backend("fake", previous)
    return held, saved


def test_workspace_layout_counts_one_staging_plane_under_save_transfer() -> None:
    """R2-L1 1a: ``cake_gdn_nondet_train_kcp_workspace_layout`` counts the ``stage_m`` plane only when the call sizes it (the replay
    boundary, or the light saved form without ``save_transfer``); when the saved set carries ``M_seg`` -- the shipped
    saved + ``save_transfer`` form and ``save_mode="full"`` with either ``save_transfer`` -- the saved ``M_seg^T``
    stands in for it and the workspace holds ``stage_b`` alone (the ring pools are never workspace terms).  The full
    form without ``save_transfer`` is also checked against the pools a call at ``P = 2`` actually sizes."""
    cu = torch.tensor([0, 100, 740, 817, 1817, 1946], dtype=torch.int64)
    part = gp.cake_gdn_nondet_train_kcp_partition(cu, 2, "contiguous")
    H_qk, HV = 2, 6
    seen_segmented = False
    # Under the shipped row rule this row plans P = 1 on both ranks ("row_cost" plans P = 1 there -- fragments of 2-14
    # chunks: the per-rank programs
    # segment them 7- and 13-fold to fill the SMs, and the compose chain of those segments costs more than the one-wave
    # walks they save), so the staging plane is 0; the plane term is exercised under the per-rank rule below.
    for r in range(2):
        ctx_row = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        hp_row = ctx_row.heads(HV)
        assert hp_row.segments.p_rule == gp.P_RULE_DEFAULT and hp_row.P == 1
        one_row = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx_row, H_qk=H_qk, HV=HV
        )
        assert one_row["fwd.staging"] == one_row["bwd.staging"] == 0
    for r in range(2):
        ctx = gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT, p_rule="wave_fill1"
        )
        hp = ctx.heads(
            HV
        )  # the layout's staging term uses the plan's own segment count
        plane = hp.NSEG * HV * gp.STATE_TILE_BYTES if hp.P > 1 else 0
        seen_segmented = seen_segmented or hp.P > 1
        one = gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
        two_a = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, boundary_mode="replay"
        )
        two_b = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_transfer=False
        )
        full_no_st = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, save_mode="full", save_transfer=False
        )
        assert one["fwd.staging"] == one["bwd.staging"] == plane
        assert two_a["fwd.staging"] == two_a["bwd.staging"] == 2 * plane
        assert two_b["fwd.staging"] == two_b["bwd.staging"] == 2 * plane
        assert full_no_st["fwd.staging"] == full_no_st["bwd.staging"] == plane
        assert one["fwd.c1_recv"] == 0 and two_b["fwd.c1_recv"] > 0
        assert not any(k.endswith("ring_out") or k.endswith("ring_in") for k in one)
        assert (
            gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, P=1)[
                "fwd.staging"
            ]
            == 0
        )
    assert seen_segmented  # the row has a segmented rank: the plane term was exercised
    # save_mode="full" without save_transfer at P = 2, against the pools the call sizes: the forward binds a fresh
    # M_seg^T plane and the backward the saved one for stage_m, so both lifetimes size stage_b alone (the context's own
    # plan is P = 2; the segments override is checked by test_workspace_layout_follows_the_segments_override)
    kcp = _package("cake_kcp")
    cu2 = [0, 2000, 4096]
    part2 = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu2, dtype=torch.int64), 1, "contiguous"
    )
    ctx2 = gp.cake_gdn_nondet_train_kcp_build_context(
        part2, None, 0, CPU, 4, sm_count=16, p_rule="wave_fill1"
    )
    assert ctx2.heads(HV).P == 2
    plane2 = ctx2.heads(HV).NSEG * HV * gp.STATE_TILE_BYTES
    held, saved = _held_pools(
        kcp,
        ctx2,
        _inputs(cu2, H_qk, HV),
        pools=("stage_b", "stage_m"),
        segments=2,
        save_mode="full",
        save_transfer=False,
    )
    assert kcp.LAST_CALL["fwd"]["P"] == kcp.LAST_CALL["bwd"]["P"] == 2
    assert saved is not None and saved.M_seg is not None
    assert saved.M_seg.numel() * saved.M_seg.element_size() == plane2
    assert held["fwd"] == held["bwd"] == {"stage_b": plane2}, held
    full2 = gb.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx2, H_qk=H_qk, HV=HV, save_mode="full", save_transfer=False, P=2
    )
    assert full2["fwd.staging"] == full2["bwd.staging"] == plane2


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
    """``cake_gdn_nondet_train_kcp_workspace_layout(P=...)`` reports the planes of a call at that ``segments`` override: on the
    checkpoint pack (one rank, five fragments, the plan's own ``P = 1``) a call with ``segments=2`` holds staging
    planes of ``NSEG = 2 n_frag`` states and a carry plane of ``n_frag`` states in both lifetimes, and the layout at
    ``P = 2`` counts the same bytes in every save form (it read the staging plane from the plan's ``NSEG``, half the
    call's here).  An override outside the range the call accepts is refused by both byte queries, as by
    ``CakeGDNNondetKcpContext.heads``."""
    kcp, ctx, ops = _ckpt_setup()
    H_qk, HV = CKPT_HEADS
    assert ctx.heads(HV).P == 1 and ctx.n_frag == 5
    s = HV * gp.STATE_TILE_BYTES
    plane = 2 * ctx.n_frag * s
    assert plane == 3_932_160
    held, _ = _held_pools(kcp, ctx, ops, segments=2, **opts)
    assert kcp.LAST_CALL["fwd"]["P"] == kcp.LAST_CALL["bwd"]["P"] == 2
    layout = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
    rule = gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV, **opts)
    assert rule["fwd.staging"] == rule["fwd.carry"] == 0  # the plan's P = 1
    # the range the call accepts: [1, min(MAX_SEGMENTS, chunks of the longest fragment)] = [1, 16] (1000 tokens)
    for P in (0, 17):
        for query in (
            gb.cake_gdn_nondet_train_kcp_workspace_layout,
            gb.cake_gdn_nondet_train_kcp_saved_layout,
        ):
            with pytest.raises(ValueError, match=r"override must be in \[1, 16\]"):
                query(ctx, H_qk=H_qk, HV=HV, P=P, **opts)
        with pytest.raises(ValueError, match=r"override must be in \[1, 16\]"):
            ctx.heads(HV, P=P)
    top = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
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
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            ops["q"],
            ops["k"],
            ops["v"],
            ops["g"],
            ops["beta"],
            ctx,
            backend="fake",
            segments=2,
        )
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
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


def _ckpt_setup():
    kcp = _package("cake_kcp")
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    return kcp, ctx, _inputs(CKPT_CU, *CKPT_HEADS)


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
    """``out`` and the five gradients of one ``cake_gdn_nondet_kcp`` layer (or ``layers`` chained ones: the next layer's
    ``v`` is the previous output), directly or under a non-reentrant checkpoint; ``twice`` backwards with
    ``retain_graph`` first (accumulation)."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", **opts}

    def fn(q, k, v, g, beta):
        out = kcp.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, workspace=workspace, **o)
        for _ in range(layers - 1):
            out = kcp.cake_gdn_nondet_kcp(
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


def _assert_close_arms(a: dict, b: dict, what: str, ref: dict | None = None) -> None:
    """Two arms (``out`` and the five gradients) each within the budget of the FP64 reference ``ref`` (default: the
    one-layer reference of the checkpoint pack) and of each other -- the twin asserts they are the same bits."""
    _arms_within_budget(
        {"first arm": a, "second arm": b},
        _ckpt_reference() if ref is None else ref,
        ("out",) + CKPT_GRADS,
        what,
    )


def _assert_accumulated(twice: dict, once: dict, what: str, ref: dict) -> None:
    """Two ``.backward()`` calls: every gradient within the budget of TWICE the FP64 gradient (not exactly twice the
    one-backward result), and of twice that result."""
    for n in CKPT_GRADS:
        _within_budget(twice[n], 2 * ref[n], f"{what}: {n} vs 2 x FP64")
        _within_budget(
            twice[n], (once[n].double() * 2), f"{what}: {n} vs 2 x one backward"
        )


def _list_arms(arms: dict[str, list], ref: list, names, what: str) -> None:
    """:func:`_arms_within_budget` for arms given as lists in the order of ``names``."""
    names = tuple(names)
    _arms_within_budget(
        {key: dict(zip(names, arm, strict=True)) for key, arm in arms.items()},
        dict(zip(names, ref, strict=True)),
        names,
        what,
    )


def _scaled_reference(w0: torch.Tensor) -> tuple[torch.Tensor, list]:
    """FP64 ``(y, [dq, dk, dv, dg, dbeta, dw])`` of the segment ``y = gdn(q, k, v, g, beta) * w`` on the checkpoint
    pack with the seeded ``dout``."""
    ops = _inputs(CKPT_CU, *CKPT_HEADS)
    layer = _ref_layer(CKPT_CU)
    return _fp64(
        lambda q, k, v, g, beta, w_: layer(q, k, v, g, beta) * w_,
        [*(ops[n] for n in LEAVES), w0],
        ops["dout"],
    )


@pytest.mark.parametrize(
    "segments,save_transfer", [(1, True), (2, True), (2, False), (1, False)]
)
def test_checkpoint_backward_reuses_the_recompute_tapes(
    segments, save_transfer
) -> None:
    """With ``reuse_recompute_tapes`` the checkpoint backward binds the recompute forward's tapes: the gradients are
    within the accuracy budget of the FP64 reference and of the direct autograd's and the switch-off checkpoint's; the
    backward's launch inventory starts at #6 (no
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
    _assert_close_arms(kept, direct, "kept vs direct")
    _assert_close_arms(light, direct, "checkpoint (switch off) vs direct")
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
        "kcp_fwdh_out_tape",
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
    # a light forward (the hand-off one included) never runs the tape-and-output walk of save_mode="full"
    assert not any(n == "kcp_fwdh_out_tape" for n, *_ in seen), seen
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
    layers' backwards miss and run the replay.  The gradients of the whole chain are within the accuracy budget of the
    chained FP64 reference and of the direct autograd's (the evicted layers included); the registry is empty
    afterwards."""
    kcp, ctx, ops = _ckpt_setup()
    ws = kcp.CakeGDNNondetKcpWorkspace()
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
    _assert_close_arms(
        kept,
        direct,
        f"{layers} layers, one live kept set, vs direct",
        _ckpt_reference(layers=layers),
    )
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
    a second entry is put and taken; the gradients accumulate to within the budget of twice the FP64 gradient."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    twice = _ckpt_autograd(
        kcp, ctx, ops, ckpt=True, twice=True, segments=2, reuse_recompute_tapes=True
    )
    _assert_accumulated(twice, direct, "kept backward twice", _ckpt_reference())
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
    """Correctness never depends on the registry: a backward that finds no entry runs today's replay (gradients within
    the accuracy budget of the reference and of the direct pass); the untaken entry is dropped at the graph task's end by the engine's final callback."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    monkeypatch.setattr(kcp, "_kept_take", lambda key: None)
    res = _ckpt_autograd(
        kcp, ctx, ops, ckpt=True, segments=2, reuse_recompute_tapes=True
    )
    _assert_close_arms(res, direct, "fallback replay vs direct")
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
        return a_ * a_, kcp.cake_gdn_nondet_kcp(
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
            kcp.cake_gdn_nondet_kcp(
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
    _assert_close_arms(kept, direct, "kept after a failed task vs direct")
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
    (nothing saved) and ``save_mode="full"`` (the saved tapes): no entry is put, the gradients are within the accuracy
    budget of the FP64 reference and of the direct pass."""
    kcp, ctx, ops = _ckpt_setup()
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
    before = dict(kcp.KEPT_STATS)
    on_direct = _ckpt_autograd(
        kcp, ctx, ops, ckpt=False, segments=2, reuse_recompute_tapes=True
    )
    _assert_close_arms(on_direct, direct, "direct autograd with the switch")
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
    _assert_close_arms(replay, direct, "replay boundary with the switch")
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
    _assert_close_arms(full, direct, "save_mode=full with the switch")
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
    ``fwd_o`` / ``kcp_fwdh_out`` / ``kcp_fwdh_out_tape`` launch; the scratch carries that forward's ``fwd_o``
    decision and reason once it has run."""
    seen: list[tuple[str, bool, dict]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward and name in (
                "fwd_prep",
                "fwd_o",
                "kcp_fwdh_out",
                "kcp_fwdh_out_tape",
            ):
                seen.append(
                    (name, bool(call.scratch.get("hand_off", False)), call.scratch)
                )
            super().run(name, args)

    return seen, Recording()


def _launches(seen, name: str) -> list[bool]:
    return [h for n, h, _ in seen if n == name]


def _o_writes(seen) -> list[bool]:
    """The hand-off flags of the forwards that wrote the output, in launch order: by #5 (decision ``launched``), by
    the output walk (``fused``) or by the tape-and-output walk (``fused_tape``, a ``save_mode="full"`` forward); each
    forward's own launches agree with its decision (``fwd_o`` once when launched, ``kcp_fwdh_out`` when fused,
    ``kcp_fwdh_out_tape`` when fused_tape, none of them when skipped)."""
    writes = []
    for i, (name, hand_off, scratch) in enumerate(seen):
        if name != "fwd_prep":
            continue
        own = [n for n, _, s in seen[i + 1 :] if s is scratch and n != "fwd_prep"]
        decision = scratch["fwd_o"]
        assert own.count("fwd_o") == int(decision == "launched"), (decision, own)
        assert ("kcp_fwdh_out" in own) == (decision == "fused"), (decision, own)
        assert ("kcp_fwdh_out_tape" in own) == (decision == "fused_tape"), (
            decision,
            own,
        )
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
    fwd_prep runs in both forwards, the output program (the output walk) in the original only -- the outputs and the gradients are within the
    accuracy budget of the FP64 reference and of the ``recompute_output="always"`` form's and the direct autograd's,
    the kept-tape accounting is unchanged, and the
    skip does not rest on the kept tapes (the replay form skips too).  A later op of the segment that saves nothing
    (``.sum()``) keeps the skip: torch's early stop ends the recompute right after this node's tensors are re-saved,
    before that op runs.  On a torch build outside the verified set every recompute writes the output instead (reason
    ``torch_version``, counters untouched): the shipped ``reuse_recompute_tapes="auto"`` takes the replay form
    there, so the recompute hands nothing off, puts no kept entry and writes the output as a plain forward; the same
    budget checks hold."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    hand = _auto_hand_off(kcp)
    fwd_o, decision = _fwd_o_after_recompute(kcp, hand), _dead_decision(kcp, hand)
    reason, delta = _frame_reason(kcp, "dead"), _frame_delta(kcp, 1, 0)

    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)

    def seg_sum(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.cake_gdn_nondet_kcp(
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
    _assert_close_arms(auto, direct, "auto vs direct")
    _assert_close_arms(auto, always, "auto vs always")
    _assert_close_arms(replay_form, direct, "auto + replay form vs direct")
    layer = _ref_layer(CKPT_CU)
    _, ref_sum = _fp64(lambda *x: layer(*x).sum(), [ops[n] for n in LEAVES])
    _list_arms(
        {"checkpoint": ckpt_sum, "direct": direct_sum},
        ref_sum,
        CKPT_GRADS,
        "segment .sum()",
    )
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
    ``holders_remain``) and the gradients -- ``dw`` included -- are within the accuracy budget of the FP64 reference
    and of the always form's and the direct autograd's."""
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
            out = kcp.cake_gdn_nondet_kcp(
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
    _, ref_w = _scaled_reference(w0)
    _list_arms(
        {"auto": auto, "direct": direct, "always": always},
        ref_w,
        CKPT_GRADS + ("dw",),
        "o * w segment",
    )
    assert kcp.kept_registry_size() == 0


def test_recompute_output_multi_layer_segment_skips_only_the_last_layer() -> None:
    """Three GDN layers in one checkpoint segment (the shipped one-live-kept-set form): the next layer saves this
    layer's output as its ``v``, so in the recompute layers 1-2 launch #5 (holders remain) and only the last layer,
    the segment's last saver, leaves it out -- pinned per forward, in launch order.  The chain's gradients are within
    the accuracy budget of the chained FP64 reference and of the direct autograd's and the always form's; the kept-tape
    accounting is the shipped one (one live set, the earlier layers replay)."""
    kcp, ctx, ops = _ckpt_setup()
    layers = 3
    ws = kcp.CakeGDNNondetKcpWorkspace()
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
    chain_ref = _ckpt_reference(layers=layers)
    _assert_close_arms(auto, direct, f"{layers} layers, auto vs direct", chain_ref)
    _assert_close_arms(auto, always, f"{layers} layers, auto vs always", chain_ref)
    assert kcp.kept_registry_size() == 0


def test_recompute_output_with_an_untaken_node_and_a_second_backward() -> None:
    """(a) The segment is recomputed for another saved tensor while the GDN node itself is not in the backward: the
    node is still the segment's last saver, so #5 is left out, and the kept-tape accounting is the shipped one (the
    untaken entry goes at the task's end).  (b) ``retain_graph`` and a second ``.backward()``: two recomputes, two
    skips, two kept entries put and taken, gradients within the budget of twice the FP64 gradient.  Both parts run
    the shipped ``reuse_recompute_tapes="auto"``: on a torch build outside the verified set it takes the replay form
    (no kept entry is put, taken or dropped), the recompute writes the output as a plain forward and the counters
    stay."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    hand = int(_auto_hand_off(kcp))  # kept entries per recompute
    leaves = _ckpt_leaves(ops)
    a = torch.randn(3, requires_grad=True)

    def fn(a_, q, k, v, g, beta):
        return a_ * a_, kcp.cake_gdn_nondet_kcp(
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
    _assert_accumulated(twice, direct, "two recomputes", _ckpt_reference())
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
    _assert_close_arms(res, direct, "unverified torch: launched form vs direct")


def test_recompute_output_launched_under_a_reentrant_checkpoint() -> None:
    """``use_reentrant=True`` re-runs the segment inside the backward as well, but through the reentrant Function: no
    checkpoint frame, no saved-tensors hooks, and that re-run's output IS consumed (by the nested backward).  #5 is
    launched in both forwards (reason ``no_hooks``); gradients within the accuracy budget of the FP64 reference and
    of the direct autograd's; registry empty."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    def run(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.cake_gdn_nondet_kcp(
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
    ref = _ckpt_reference()
    _list_arms(
        {"reentrant checkpoint": res, "direct": direct},
        [ref[n] for n in CKPT_GRADS],
        CKPT_GRADS,
        "reentrant checkpoint",
    )
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_for_a_non_executable_layer_before_the_node() -> None:
    """Two layers in one segment, the first with operands that require no grad: autograd creates no node for it
    (its ``apply`` is not executable and packs nothing), yet its output is the second layer's ``v``.  The gate read
    before ``apply`` launches its #5 in both forwards with the static reason ``not_executable`` -- the frame is
    never counted for it, so the second layer's waiting holders cannot pass for its own -- while the second layer,
    the segment's last saver, leaves #5 out of the recompute.  Output and gradients (``v`` gets none: it reaches
    the node through a constant) are within the accuracy budget of the FP64 reference and of the always form's and
    the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint

    ws = kcp.CakeGDNNondetKcpWorkspace()

    def run(ckpt: bool, **o) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)
        opt = dict(
            backend="fake", segments=2, workspace=ws, reuse_recompute_tapes=True, **o
        )

        def fn(q, k, v, g, beta):
            first = kcp.cake_gdn_nondet_kcp(
                q.detach(),
                k.detach(),
                v.detach(),
                g.detach(),
                beta.detach(),
                ctx,
                **opt,
            )
            assert not first.requires_grad
            return kcp.cake_gdn_nondet_kcp(q, k, first, g, beta, ctx, **opt)

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
    layer = _ref_layer(CKPT_CU)

    def chain(q, k, v, g, beta):
        first = layer(q.detach(), k.detach(), v.detach(), g.detach(), beta.detach())
        return layer(q, k, first, g, beta)

    y_ref, g_ref = _fp64(chain, [ops[n] for n in LEAVES], ops["dout"])
    assert g_ref[2] is None  # v reaches the node through a constant
    _list_arms(
        {"auto": auto, "direct": direct, "always": always},
        [y_ref] + [gr for i, gr in enumerate(g_ref) if i != 2],
        ("out", "dq", "dk", "dg", "dbeta"),
        "non-executable first layer",
    )
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_under_no_grad_inside_the_recompute() -> None:
    """The GDN call under ``torch.no_grad()`` inside a segment whose later op (``o * w``, a learnable ``w``) saves
    its output: grad mode is off at ``apply``, autograd creates no node, and the gate launches #5 in the original
    and the recompute forward alike (reason ``not_executable``; the frame is never counted, the counters stay).
    ``dw`` is within the accuracy budget of the FP64 reference and of the always form's and the direct autograd's."""
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
                out = kcp.cake_gdn_nondet_kcp(
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
    layer = _ref_layer(CKPT_CU)

    def no_grad_layer(q, k, v, g, beta, w_):
        with torch.no_grad():
            out = layer(q, k, v, g, beta)
        return out * w_

    y_ref, g_ref = _fp64(no_grad_layer, [*(ops[n] for n in LEAVES), w0], ops["dout"])
    _list_arms(
        {"auto": auto, "direct": direct, "always": always},
        [y_ref, g_ref[5]],
        ("y", "dw"),
        "GDN call under no_grad",
    )
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_when_early_stop_is_off() -> None:
    """``set_checkpoint_early_stop(False)`` around the checkpoint: the recompute re-runs the whole segment and
    nothing bounds the waiting holders to this node's, so #5 is launched in the recompute (reason
    ``no_early_stop``); gradients within the accuracy budget of the FP64 reference and of the direct autograd's."""
    kcp, ctx, ops = _ckpt_setup()
    from torch.utils.checkpoint import checkpoint, set_checkpoint_early_stop

    def run(ckpt: bool) -> list[torch.Tensor]:
        leaves = _ckpt_leaves(ops)

        def fn(q, k, v, g, beta):
            return kcp.cake_gdn_nondet_kcp(
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
    ref = _ckpt_reference()
    _list_arms(
        {"checkpoint, early stop off": res, "direct": direct},
        [ref[n] for n in ("out",) + CKPT_GRADS],
        ("out",) + CKPT_GRADS,
        "early stop off",
    )
    assert kcp.kept_registry_size() == 0


def test_recompute_output_launched_under_a_users_saved_tensors_hooks() -> None:
    """A user's ``saved_tensors_hooks`` around the GDN call inside the segment (its later op ``o * w`` still saves
    through the checkpoint, so the recompute runs): the top of the hook stack is not torch's checkpoint hook, the
    proof is not read, and #5 is launched in both forwards (reason ``user_hooks``, counters untouched); gradients
    within the accuracy budget of the FP64 reference and of the direct autograd's."""
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
                out = kcp.cake_gdn_nondet_kcp(
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
    y_ref, g_ref = _scaled_reference(w0)
    _list_arms(
        {"checkpoint": res, "direct": direct},
        [y_ref, g_ref[5]] + g_ref[:5],
        ("y", "dw") + CKPT_GRADS,
        "user saved-tensors hooks",
    )
    assert kcp.kept_registry_size() == 0


# --------------------------------------------------------------------------- issue #7074 (1): sync-free context preparation (CPU)
# The tables of a context and its head plan travel in pinned, asynchronous copies (upload_tables); the values are those of
# torch.tensor(values, dtype).  The no-sync contract itself is measured on a device (test_cake_gdn_nondet_train_kcp_cp1.py, the dist test).


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
        _assert_close_arms(res, direct, build)


# --------------------------------------------------------------------------- issue #7074 (4): cake_gdn_nondet_train_kcp_workspace_layout follows the arena's lifetimes
# What a caller-held workspace binds, per tape form: the forward's tape-walk peak and the backward's phase maximum.


class _PeakWorkspace:
    """A ``CakeGDNNondetKcpWorkspace`` recording the most bytes it binds (every allocation / release path)."""

    def __new__(cls, kcp):
        class Tracked(kcp.CakeGDNNondetKcpWorkspace):
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
    """``cake_gdn_nondet_train_kcp_workspace_layout`` is what a caller-held workspace binds: the forward's peak (the shipped fused
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
        layout = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
        )
        layouts[form] = layout
        ws = _PeakWorkspace(kcp)
        leaves = [
            ops[n].detach().clone().requires_grad_(True)
            for n in ("q", "k", "v", "g", "beta")
        ]

        def fn(q, k, v, g, beta, _ws=ws, _form=form, _fused=fused):
            return kcp.cake_gdn_nondet_kcp(
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
    default = gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, H_qk=H_qk, HV=HV)
    opt = _package("cake_options")
    assert default == layouts[opt.REUSE_RECOMPUTE_TAPES_DEFAULT is True]
    assert (
        gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="auto"
        )
        == replay
    )
    for form in (True, False):
        assert (
            gb.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes=form
            )
            == layouts[form]["workspace"]
        )
    with pytest.raises(ValueError, match="reuse_recompute_tapes must be"):
        gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, H_qk=H_qk, HV=HV, reuse_recompute_tapes="yes"
        )


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
    # a checkpointed call saves as "light" (the default with the norm, "light_akk",
    # too), so the queries size it with save_mode="light"
    q = dict(
        H_qk=H_qk,
        HV=HV,
        P=2,
        save_transfer=save_transfer,
        with_conv=False,
        use_qk_l2norm_in_kernel=l2n,
        save_mode="light",
    )
    layout = gb.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, reuse_recompute_tapes=True, **q
    )
    replay = gb.cake_gdn_nondet_train_kcp_workspace_layout(
        ctx, reuse_recompute_tapes=False, **q
    )
    handed: list[int] = []

    class Recorded(kcp._KeptTapes):
        def __init__(self, tapes, stage_m, geometry, gid, qk=None, **kw):
            super().__init__(tapes, stage_m, geometry, gid, qk, **kw)
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
        return kcp.cake_gdn_nondet_kcp(
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
        out = kcp.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, **o)
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


def _region_reference(later_saver: bool) -> tuple[dict, torch.Tensor | None]:
    """FP64 reference of :func:`_ckpt_autograd_region`: ``out`` (scaled by the leaf ``1.5`` with a later saver) and the
    five gradients, and the scale's gradient (``None`` without a later saver); cached."""
    key = ("region", later_saver)
    if key not in _REFERENCES:
        if not later_saver:
            _REFERENCES[key] = (_ckpt_reference(), None)
        else:
            ops = _inputs(CKPT_CU, *CKPT_HEADS)
            layer = _ref_layer(CKPT_CU)
            scale = torch.full((), 1.5, dtype=torch.bfloat16)
            y, grads = _fp64(
                lambda q, k, v, g, beta, s_: layer(q, k, v, g, beta) * s_,
                [*(ops[n] for n in LEAVES), scale],
                ops["dout"],
            )
            _REFERENCES[key] = (
                {"out": y, **dict(zip(CKPT_GRADS, grads[:5], strict=True))},
                grads[5],
            )
    return _REFERENCES[key]


@pytest.mark.parametrize("segments,save_transfer", [(1, True), (2, True), (2, False)])
def test_reuse_recompute_tapes_auto_keeps_tapes_only_when_the_node_closes_the_region(
    segments, save_transfer
) -> None:
    """``"auto"``: the recompute hands its tapes to the backward when the GDN node is the region's last saver (the
    frame proof ``dead``; the backward of the region then starts with this node) and takes the replay form when a
    later operation of the region saves a tensor (``holders_remain``) -- so the kept set never outlives the GDN
    backward.  ``True`` always hands off, ``False`` never; every form is within the accuracy budget of the FP64
    reference and of the direct autograd result, and the registry is empty afterwards.  On an unverified torch release ``"auto"`` is the replay form."""
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
            ref, ref_scale = _region_reference(later_saver)
            _assert_close_arms(res, direct, what, ref)
            if later_saver:
                _within_budget(scale_grad, ref_scale, f"{what}: dscale vs FP64")
                _within_budget(scale_grad, direct_scale, f"{what}: dscale vs direct")
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
        opt.CakeGDNNondetKcpOptions(reuse_recompute_tapes="auto")
        .check()
        .reuse_recompute_tapes
        == "auto"
    )
    assert (
        opt.CakeGDNNondetKcpOptions().check().reuse_recompute_tapes
        == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    )
    assert (
        opt.CakeGDNNondetKcpOptions(graph="replay").check().reuse_recompute_tapes
        == opt.REUSE_RECOMPUTE_TAPES_DEFAULT
    )
    for value in (True, False, "auto"):
        o = opt.CakeGDNNondetKcpOptions(
            graph="replay", reuse_recompute_tapes=value
        ).check()
        assert o.reuse_recompute_tapes == value
    for bad in ("yes", 1, 0, "AUTO"):
        with pytest.raises(TypeError, match="must be a bool or 'auto'"):
            opt.CakeGDNNondetKcpOptions(reuse_recompute_tapes=bad).check()


# --------------------------------------------------------------------------- reuse_recompute_tapes under graph="replay" (CPU)
# The checkpointed step under graph="replay" on a caller-held workspace.  The CPU node refuses every CUDA-graph capture
# (a stub of _graph_capture; the refused entry falls back to eager, a code path of the package), so each call runs the
# graph mode's eager form: the recompute decides its hand-off and #5 before the graph lookup, the hand-off keeps the
# arena's tapes bound (nothing is unbound: a captured graph bakes their addresses) and the backward binds them only
# while the workspace has made no call in between.  The captured / replayed form runs on a GPU
# (test_cake_gdn_nondet_train_kcp_cp1.py).


def _refuse_captures(kcp, monkeypatch) -> None:
    def refuse(kind, entry, call, body):
        entry.disabled = "RuntimeError: no CUDA-graph capture on the CPU test node"
        kcp.GRAPH_STATS["fallback"] += 1
        return None

    monkeypatch.setattr(kcp, "_graph_capture", refuse)


def _recorded_step(kcp, ctx, ops, **opts) -> tuple[dict, list[tuple[str, bool, bool]]]:
    """Out and the five gradients of one :func:`_ckpt_autograd` step and its launches in issue order, as
    ``(program, backward?, hand-off forward?)``."""
    seen: list[tuple[str, bool, bool]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            seen.append(
                (name, bool(call.backward), bool(call.scratch.get("hand_off", False)))
            )
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        res = _ckpt_autograd(kcp, ctx, ops, **opts)
    finally:
        kcp.set_backend("fake", previous)
    return res, seen


def _ckpt_reference_l2n() -> dict[str, torch.Tensor]:
    """:func:`_ckpt_reference` of the in-kernel Q / K L2 norm: the exact layer on the raw rows normalized in FP64
    (``x / sqrt(sum x^2 + 1e-6)`` per token and head), gradients carried back to the raw rows."""
    key = ("ckpt_l2n", 1)
    if key not in _REFERENCES:
        ops = _inputs(CKPT_CU, *CKPT_HEADS)
        layer = _ref_layer(CKPT_CU)

        def norm(x):
            return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)

        def region(q, k, v, g, beta):
            return layer(norm(q), norm(k), v, g, beta)

        out, grads = _fp64(region, [ops[n] for n in LEAVES], ops["dout"])
        _REFERENCES[key] = {"out": out, **dict(zip(CKPT_GRADS, grads, strict=True))}
    return _REFERENCES[key]


@pytest.mark.parametrize("l2n", [False, True])
@pytest.mark.parametrize("segments,save_transfer", [(1, True), (2, True), (2, False)])
def test_graph_replay_checkpoint_runs_the_eager_checkpoint_programs(
    segments, save_transfer, l2n, monkeypatch
) -> None:
    """``graph="replay"`` on a caller-held workspace, ``torch.utils.checkpoint(use_reentrant=False)``, the default
    ``reuse_recompute_tapes`` / ``recompute_output``: the step launches exactly the eager step's programs in the same
    order (the recompute hands its tapes over and, on a verified torch, leaves #5 out; the backward starts at #6),
    the kept tapes are the arena's own tensors (never unbound), the registry sees one put / take / release and no
    stale entry; ``reuse_recompute_tapes=False`` (the previous graph form) replays #1 in the backward.  Out and the
    five gradients are the same bits in the three arms and within the budget of the FP64 reference and the direct
    autograd."""
    kcp, ctx, ops = _ckpt_setup()
    _refuse_captures(kcp, monkeypatch)
    base = dict(segments=segments, save_transfer=save_transfer)
    if l2n:
        base["use_qk_l2norm_in_kernel"] = True
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    kcp.graph_cache_clear()
    eager, eager_seen = _recorded_step(
        kcp, ctx, ops, ckpt=True, workspace=kcp.CakeGDNNondetKcpWorkspace(), **base
    )
    bound: dict[str, bool] = {}
    original_backward = kcp._backward

    def checking_backward(call, saved):
        if saved is not None and saved.kept is not None:
            # the arena hand-off: the kept tensors are the workspace's own tapes at the backward
            for n, t in (*saved.kept.tapes.items(), *saved.kept.qk.items()):
                bound[n] = call.ws._tapes.get(n) is t
        original_backward(call, saved)

    monkeypatch.setattr(kcp, "_backward", checking_backward)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    before, stale = dict(kcp.KEPT_STATS), dict(kcp.KEPT_STALE_STATS)
    graph, graph_seen = _recorded_step(
        kcp, ctx, ops, ckpt=True, workspace=ws, graph="replay", **base
    )
    graph_source = kcp.LAST_CALL["bwd"]["tapes_source"]
    recompute = dict(kcp.LAST_CALL["fwd"])
    delta = _stats_delta(kcp, before)
    assert dict(kcp.KEPT_STALE_STATS) == stale
    previous, previous_seen = _recorded_step(
        kcp,
        ctx,
        ops,
        ckpt=True,
        workspace=kcp.CakeGDNNondetKcpWorkspace(),
        graph="replay",
        reuse_recompute_tapes=False,
        **base,
    )
    assert graph_seen == eager_seen, (graph_seen, eager_seen)
    prep = "fwd_prep_l2n" if l2n else "fwd_prep"
    n_prep = sum(1 for n, _, _ in graph_seen if n == prep)
    assert sum(1 for n, _, _ in previous_seen if n == prep) == 3
    assert kcp.kept_registry_size() == 0
    if _verified_torch(kcp):
        assert n_prep == 2, graph_seen
        assert graph_source == "kept"
        assert recompute["hand_off"] is True and recompute["fwd_o"] == "skipped"
        assert delta == {
            "put": 1,
            "take": 1,
            "miss": 0,
            "released": 1,
            "callback": 0,
            "evicted": 0,
        }, delta
        names = set(kcp.FWD_TAPES) | (set(kcp.QK_TAPES) if l2n else set())
        assert set(bound) == names and all(bound.values()), bound
        # the backward launches nothing of the forward
        bwd = [n for n, b, _ in graph_seen if b]
        assert bwd[0] == "bwd_prep", bwd[:3]
    else:  # "auto" keeps no tape without the frame proof: the replay form, as eager
        assert n_prep == 3 and graph_source == "replay"
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(graph[n], eager[n]), n
        assert torch.equal(graph[n], previous[n]), n
    _assert_close_arms(
        graph,
        direct,
        "graph checkpoint vs direct",
        _ckpt_reference_l2n() if l2n else None,
    )


def test_graph_replay_auto_size_rule_is_one_table_per_architecture(monkeypatch) -> None:
    """The size rule's rows per architecture (``GRAPH_RECOMPUTE_MIN_ROWS``, the shipped values: the module's own,
    without this suite's rule at 0): the B200 crossover on sm_100a, none on sm_103a and sm_107a (GB300 and R200 gain
    at the smallest measured rank), sm_100a's value on any other device; the rule reads the kernel backend's
    architecture only."""
    monkeypatch.undo()
    kcp = _package("cake_kcp")
    assert kcp.GRAPH_RECOMPUTE_MIN_ROWS == {
        "sm_100a": 61440,
        "sm_103a": 0,
        "sm_107a": 0,
    }
    assert kcp.GRAPH_RECOMPUTE_MIN_ROWS_DEFAULT == 61440
    assert (
        kcp._graph_recompute_arch(kcp.get_backend("fake"), torch.device("cpu")) is None
    )


@pytest.mark.parametrize(
    "arch,l2n",
    [("sm_100a", False), ("sm_103a", False), ("sm_107a", False), ("sm_100a", True)],
)
def test_graph_replay_auto_size_rule_keys_a_small_recompute_as_the_original_forward(
    arch, l2n, monkeypatch
) -> None:
    """The size rule of ``reuse_recompute_tapes="auto"`` under ``graph="replay"`` (``GRAPH_RECOMPUTE_MIN_ROWS`` of the
    device's architecture): a checkpoint recompute whose rank holds fewer token x value-head rows (``T_r * HV``)
    reads no checkpoint frame, looks up the original forward's key (flags ``(save,)``) and runs its form -- no
    hand-off, #5 launched, reasons ``"small"`` -- so its backward replays #1 (three prep launches) and no entry is
    put; at exactly that many rows the recompute decides as before (on a verified torch its own key, the hand-off and
    two prep launches).  The rule reads the row of the architecture (the other rows and the default at 0).  ``True``
    still hands off below the rule and ``False`` still reads the frame for #5.  Out and the five gradients are the
    same bits in every arm."""
    kcp, ctx, ops = _ckpt_setup()
    _refuse_captures(kcp, monkeypatch)
    monkeypatch.setattr(kcp, "_graph_recompute_arch", lambda backend, device: arch)
    base = {"use_qk_l2norm_in_kernel": True} if l2n else {}
    rows = int(ctx.T_r) * CKPT_HEADS[1]
    keys: list[tuple] = []
    reads: list[int] = []
    original_entry, original_dead = kcp._graph_entry, kcp._recompute_output_dead

    def spy_entry(kind, ctx_, ws, backend, opts, scale, tensors, flags):
        if kind == "fwd":
            keys.append(tuple(flags))
        return original_entry(kind, ctx_, ws, backend, opts, scale, tensors, flags)

    def spy_dead(n_saved):
        reads.append(n_saved)
        return original_dead(n_saved)

    monkeypatch.setattr(kcp, "_graph_entry", spy_entry)
    monkeypatch.setattr(kcp, "_recompute_output_dead", spy_dead)
    prep = "fwd_prep_l2n" if l2n else "fwd_prep"

    def step(min_rows: int, **kw):
        _size_rule_at(kcp, monkeypatch, 0)
        monkeypatch.setitem(kcp.GRAPH_RECOMPUTE_MIN_ROWS, arch, min_rows)
        keys.clear()
        reads.clear()
        before = dict(kcp.KEPT_STATS)
        kcp.graph_cache_clear()
        res, seen = _recorded_step(
            kcp,
            ctx,
            ops,
            ckpt=True,
            workspace=kcp.CakeGDNNondetKcpWorkspace(),
            graph="replay",
            **base,
            **kw,
        )
        return {
            "res": res,
            "prep": sum(1 for n, _, _ in seen if n == prep),
            "keys": list(keys),
            "reads": len(reads),
            "recompute": dict(kcp.LAST_CALL["fwd"]),
            "source": kcp.LAST_CALL["bwd"]["tapes_source"],
            "delta": _stats_delta(kcp, before),
        }

    small = step(rows + 1)
    assert small["keys"] == [(True,), (True,)], small["keys"]
    assert small["reads"] == 0
    rc = small["recompute"]
    assert rc["hand_off"] is False and rc["fwd_o"] != "skipped", rc
    assert rc["hand_off_reason"] == rc["fwd_o_reason"] == "small", rc
    assert small["prep"] == 3 and small["source"] == "replay"
    assert small["delta"]["put"] == small["delta"]["take"] == 0, small["delta"]
    at = step(rows)
    forced = step(rows + 1, reuse_recompute_tapes=True)
    off = step(rows + 1, reuse_recompute_tapes=False)
    assert at["reads"] > 0 and off["reads"] > 0
    assert off["recompute"]["hand_off"] is False and off["prep"] == 3
    assert kcp.kept_registry_size() == 0
    if _verified_torch(kcp):
        assert at["keys"] == [(True,), (True, True, False)], at["keys"]
        assert at["recompute"]["hand_off"] is True and at["prep"] == 2
        assert at["source"] == "kept" and at["delta"]["put"] == 1, at["delta"]
        assert forced["recompute"]["hand_off"] is True and forced["prep"] == 2
        assert forced["recompute"]["hand_off_reason"] == "always"
        assert off["recompute"]["fwd_o"] == "skipped"
    else:  # no frame proof on this torch: "auto" decides the replay form, True still hands off
        assert at["keys"] == [(True,), (True,)] and at["prep"] == 3
        assert forced["recompute"]["hand_off"] is True
    for arm in (at, forced, off):
        for n in ("out",) + CKPT_GRADS:
            assert torch.equal(arm["res"][n], small["res"][n]), n


def test_graph_replay_auto_size_rule_is_read_once_per_recompute(monkeypatch) -> None:
    """The node's forward reads the size rule once, and only where a hand-off can follow: torch's recompute inside the
    backward pass (one read per checkpointed step, below and above the rule); the original forward and a step without
    a checkpoint read none.  The read sees the rank's rows (``T_r * HV``)."""
    kcp, ctx, ops = _ckpt_setup()
    _refuse_captures(kcp, monkeypatch)
    rows = int(ctx.T_r) * CKPT_HEADS[1]
    reads: list[int] = []
    original = kcp._graph_recompute_small

    def spy(opts, v, T_r, HV):
        reads.append(T_r * HV)
        return original(opts, v, T_r, HV)

    monkeypatch.setattr(kcp, "_graph_recompute_small", spy)
    for min_rows in (0, rows + 1):
        _size_rule_at(kcp, monkeypatch, min_rows)
        for ckpt in (True, False):
            reads.clear()
            kcp.graph_cache_clear()
            _ckpt_autograd(
                kcp,
                ctx,
                ops,
                ckpt=ckpt,
                workspace=kcp.CakeGDNNondetKcpWorkspace(),
                graph="replay",
            )
            assert reads == ([rows] if ckpt else []), (min_rows, ckpt, reads)
    assert kcp.kept_registry_size() == 0


def test_graph_replay_auto_size_rule_reads_each_device_architecture_once(
    monkeypatch,
) -> None:
    """The size rule caches the architecture per (backend name, device index): two devices whose architectures
    differ never share an entry and give their own answers, each device is read once, and a request other than
    ``"auto"`` reads nothing."""
    kcp = _package("cake_kcp")
    monkeypatch.setattr(kcp, "GRAPH_RECOMPUTE_MIN_ROWS", {"sm_100a": 100, "sm_103a": 0})
    archs = {0: "sm_100a", 1: "sm_103a"}
    read: list[int] = []

    def arch_of(backend, device):
        read.append(device.index)
        return archs[device.index]

    monkeypatch.setattr(kcp, "_graph_recompute_arch", arch_of)

    class Operand:  # what the rule reads of v: its device index and its device
        def __init__(self, index: int) -> None:
            self.device = torch.device("cuda", index)

        def get_device(self) -> int:
            return self.device.index

    auto = types.SimpleNamespace(reuse_recompute_tapes="auto", backend="fake")
    for _ in range(3):
        assert kcp._graph_recompute_small(auto, Operand(0), 10, 5) is True
        assert kcp._graph_recompute_small(auto, Operand(1), 10, 5) is False
    assert read == [0, 1], read
    assert kcp._GRAPH_RECOMPUTE_ARCH == {("fake", 0): "sm_100a", ("fake", 1): "sm_103a"}
    for reuse in (True, False):
        req = types.SimpleNamespace(reuse_recompute_tapes=reuse, backend="fake")
        assert kcp._graph_recompute_small(req, Operand(2), 10, 5) is False
    assert read == [0, 1], read


@pytest.mark.parametrize("l2n", [False, True])
@pytest.mark.parametrize("segments,save_transfer", [(1, True), (2, False)])
def test_graph_replay_kept_backward_keys_the_arena_hand_off_by_its_layout(
    segments, save_transfer, l2n, monkeypatch
) -> None:
    """The kept backward's CUDA-graph key under ``graph="replay"`` holds the arena hand-off's layout (flag, geometry,
    ``stage_m`` and ``qn`` / ``kn`` presence), not its tensors: the key's tensor part is the replay form's (operands,
    ``dout``, saved set) and its flags end with :func:`_kept_layout` of the entry the backward took.  A kept set that
    is not the hand-off just taken on this workspace (a stamp from an earlier call) keys by its tensors."""
    kcp, ctx, ops = _ckpt_setup()
    if not _verified_torch(kcp):
        pytest.skip("no hand-off without the frame proof on this torch")
    _refuse_captures(kcp, monkeypatch)
    base = dict(segments=segments, save_transfer=save_transfer)
    if l2n:
        base["use_qk_l2norm_in_kernel"] = True
    keys: list[tuple[int, tuple]] = []
    taken: list = []
    original_entry, original_take = kcp._graph_entry, kcp._kept_take

    def spy_entry(kind, ctx_, ws, backend, opts, scale, tensors, flags):
        if kind == "bwd":
            keys.append((len(tuple(tensors)), flags))
        return original_entry(kind, ctx_, ws, backend, opts, scale, tensors, flags)

    def spy_take(key):
        entry = original_take(key)
        taken.append(None if entry is None else kcp._kept_layout(entry))
        return entry

    monkeypatch.setattr(kcp, "_graph_entry", spy_entry)
    monkeypatch.setattr(kcp, "_kept_take", spy_take)
    for reuse in (False, "auto"):
        kcp.graph_cache_clear()
        _recorded_step(
            kcp,
            ctx,
            ops,
            ckpt=True,
            workspace=kcp.CakeGDNNondetKcpWorkspace(),
            graph="replay",
            reuse_recompute_tapes=reuse,
            **base,
        )
    (n_replay, f_replay), (n_kept, f_kept) = keys
    assert n_kept == n_replay, keys
    layout = taken[-1]
    assert layout is not None and f_kept == f_replay + layout, (f_kept, layout)
    assert layout[2] == (segments > 1 and not save_transfer), layout
    assert layout[3] == (tuple(kcp.QK_TAPES) if l2n else ()), layout


def test_kept_key_part_keys_only_the_hand_off_just_taken_by_its_layout() -> None:
    """:func:`_kept_key_part` (no frame proof needed, so on every torch): the arena hand-off just taken on this
    workspace keys by its layout only; an arena set of another workspace, a stamp from an earlier call and the eager
    hand-off's own tensors key by their tensors; no kept set adds nothing (the backward counts its own call before
    it)."""
    kcp, _ctx, ops = _ckpt_setup()
    ws, other = kcp.CakeGDNNondetKcpWorkspace(), kcp.CakeGDNNondetKcpWorkspace()
    t = ops["q"]
    arena = kcp._KeptTapes({"w": t}, None, (1, 2, 3, 4), 0, ws=ws)
    arena.calls = ws.calls
    ws.calls += 1  # the backward's own call: the stamp is the previous one
    assert kcp._kept_key_part(arena, ws) == ((), (True, (1, 2, 3, 4), False, ()))
    # another workspace's arena
    assert kcp._kept_key_part(arena, other) == ((t,), (True,))
    ws.calls += 1  # a call in between: not the hand-off just taken
    assert kcp._kept_key_part(arena, ws) == ((t,), (True,))
    # the eager hand-off's own tensors
    fresh = kcp._KeptTapes({"w": t}, None, (1, 2, 3, 4), 0)
    assert kcp._kept_key_part(fresh, None) == ((t,), (True,))
    assert kcp._kept_key_part(None, ws) == ((), ())


class _LaterSaver(torch.autograd.Function):
    """A later saver of the checkpointed region (it saves its input, so torch's recompute runs in its backward); its
    backward first runs ``hook()`` -- a call on the GDN layer's workspace between the recompute and the GDN backward."""

    hook = None

    @staticmethod
    def forward(fctx, x):
        fctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(fctx, gy):
        (_,) = fctx.saved_tensors
        if _LaterSaver.hook is not None:
            _LaterSaver.hook()
        return gy


def test_graph_replay_kept_tapes_are_refused_after_another_call_on_the_workspace(
    monkeypatch,
) -> None:
    """``graph="replay"``, ``reuse_recompute_tapes=True`` and a later saver in the region: the recompute hands over
    the arena's own tapes, and the later saver's backward then runs an explicit forward of OTHER operands on the same
    workspace (it rewrites those tapes) before the GDN backward.  That backward must not bind them: the entry is stale
    (``KEPT_STALE_STATS["calls"]``), counted as a miss, and the backward replays #1 and the tape walk; the gradients
    are the bits of the same step without the intervening call (which binds the kept tapes) and within the budget of
    the FP64 reference."""
    kcp, ctx, ops = _ckpt_setup()
    _refuse_captures(kcp, monkeypatch)
    other = _inputs(CKPT_CU, *CKPT_HEADS, seed=7)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    o = dict(backend="fake", graph="replay", reuse_recompute_tapes=True)

    def step(intervene: bool) -> tuple[dict, str]:
        leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]

        def fn(*a):
            return _LaterSaver.apply(
                kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **o)
            )

        def hook():
            kcp.cake_gdn_nondet_train_kcp_forward(
                *(other[n] for n in LEAVES),
                ctx,
                workspace=ws,
                **o,
            )

        _LaterSaver.hook = hook if intervene else None
        try:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
            out.backward(ops["dout"])
        finally:
            _LaterSaver.hook = None
        source = kcp.LAST_CALL["bwd"]["tapes_source"]
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)},
        }, source

    kcp.graph_cache_clear()
    clean, clean_source = step(False)
    before, stale = dict(kcp.KEPT_STATS), dict(kcp.KEPT_STALE_STATS)
    refused, refused_source = step(True)
    delta = _stats_delta(kcp, before)
    assert clean_source == "kept" and refused_source == "replay"
    assert kcp.KEPT_STALE_STATS["calls"] == stale["calls"] + 1
    assert {k: kcp.KEPT_STALE_STATS[k] - stale[k] for k in ("arena", "stream")} == {
        "arena": 0,
        "stream": 0,
    }
    assert (delta["put"], delta["take"], delta["miss"], delta["released"]) == (
        1,
        0,
        1,
        0,
    ), delta
    assert kcp.kept_registry_size() == 0
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(refused[n], clean[n]), n
    _assert_close_arms(refused, clean, "stale hand-off (replay) vs kept")


def test_kept_stale_names_why_an_arena_hand_off_no_longer_holds() -> None:
    """:func:`_kept_stale` of an arena hand-off: ``None`` right after the stamp; ``"calls"`` once the workspace made
    another call; ``"arena"`` when the arena no longer binds a kept tensor (released, or the ``stage_m`` pool
    re-allocated); ``"stream"`` for a backward on another stream; an eager hand-off (``ws`` ``None``) is never stale."""
    kcp = _package("cake_kcp")
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 1, "contiguous"
    )
    ctx = gp.cake_gdn_nondet_train_kcp_build_context(part, None, 0, CPU)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    ws.ensure(ctx, 2, 6, device="cpu", backward=True, P=2)
    tapes = {n: ws.tape(n) for n in kcp.FWD_TAPES}
    stage_m = ws.pool("stage_m")
    template = kcp._KeptTapes(tapes, stage_m, ws.geometry, -1, {}, ws=ws)

    def fresh():
        return kcp._kept_stamped(template)

    assert kcp._kept_stale(fresh()) is None
    entry = fresh()
    ws.calls += 1
    assert kcp._kept_stale(entry) == "calls"
    entry = fresh()
    entry.stream = entry.stream + 1
    assert kcp._kept_stale(entry) == "stream"
    entry = fresh()
    ws.release_tape("u")
    assert kcp._kept_stale(entry) == "arena"
    ws.adopt_tape("u", tapes["u"])
    assert kcp._kept_stale(fresh()) is None
    entry = fresh()
    ws.release_pool("stage_m")
    ws.ensure(ctx, 2, 6, device="cpu", backward=True, P=2)  # a new stage_m pool
    assert kcp._kept_stale(entry) == "arena"
    eager = kcp._KeptTapes(dict(tapes), None, ws.geometry, -1, {})
    ws.calls += 5
    assert kcp._kept_stale(eager) is None


class _FakeGraph:
    """CPU stand-in for a captured ``torch.cuda.CUDAGraph``: counts its replays."""

    def __init__(self) -> None:
        self.replays = 0

    def replay(self) -> None:
        self.replays += 1


def _fake_captures(kcp, monkeypatch) -> list:
    """CPU stand-in for the CUDA-graph capture: the body runs eagerly once (the capture's own replay) and the entry
    keeps a :class:`_FakeGraph` and the arena signature as :func:`_graph_capture` does; a later replay runs nothing
    and returns the capture's static results (the CUDA-graph contract), so a test sees which call replays which
    graph.  Returns the ``(kind, entry)`` pairs captured."""
    captured: list = []

    def capture(kind, entry, call, body):
        result = body()
        kcp._record(kind, call)
        kcp.LAST_CALL[kind].update(graph_event="capture", graph_key=entry.digest)
        entry.record = dict(kcp.LAST_CALL[kind])
        entry.graph = _FakeGraph()
        entry.arena = kcp._arena_sig(call.ws)
        entry.seen += 1
        kcp.GRAPH_STATS["capture"] += 1
        call.backend.end_call(call)
        captured.append((kind, entry))
        return result

    monkeypatch.setattr(kcp, "_graph_capture", capture)
    return captured


@pytest.mark.parametrize(
    "case", ["other_call", "data_ptr", "shape", "stride", "arena_reused"]
)
def test_graph_replay_kept_backward_never_replays_on_stale_memory(
    case, monkeypatch
) -> None:
    """The kept backward's graph is keyed by the arena hand-off's layout (:func:`_kept_key_part`), so its safety rests
    on the take's stale check and the arena signature of the capture.  CPU captures (:func:`_fake_captures`): three
    clean checkpoint steps on one workspace end with the kept backward's graph replayed; then one step where
    (``other_call``) another call on the workspace runs between the recompute and the backward, (``data_ptr`` /
    ``shape`` / ``stride``) the registered kept set carries a tape with another data pointer, shape or stride than the
    arena's, or (``arena_reused``) a tape of the arena was released and bound anew at another address.  That step
    replays no graph on stale memory: the kept backward's graph is not replayed (the take refuses the entry and the
    replay form runs, or the arena signature resets the graphs), and out and the gradients are the clean step's
    bits.  The leaves are the same tensors in every step (a graph key holds their addresses); the forward graphs are
    captured in the second step, the kept backward's once the replayed recompute hands it the static saved set."""
    kcp, ctx, ops = _ckpt_setup()
    captured = _fake_captures(kcp, monkeypatch)
    other = _inputs(CKPT_CU, *CKPT_HEADS, seed=7)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    o = dict(backend="fake", graph="replay", reuse_recompute_tapes=True)
    leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]

    def step(hook=None) -> tuple[dict, dict]:
        for lf in leaves:
            lf.grad = None

        def fn(*a):
            return _LaterSaver.apply(
                kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **o)
            )

        _LaterSaver.hook = hook
        try:
            from torch.utils.checkpoint import checkpoint

            out = checkpoint(fn, *leaves, use_reentrant=False)
            out.backward(ops["dout"])
        finally:
            _LaterSaver.hook = None
        # copies: a replay returns the capture's static results, which a later call rewrites
        return {
            "out": out.detach().clone(),
            **{n: lf.grad.clone() for n, lf in zip(CKPT_GRADS, leaves, strict=True)},
        }, dict(kcp.LAST_CALL["bwd"])

    kcp.graph_cache_clear()
    events = []
    for _ in range(6):
        clean, bwd = step()
        events.append(bwd["graph_event"])
        if bwd["graph_event"] == "replay":
            break
    assert bwd["graph_event"] == "replay" and bwd["tapes_source"] == "kept", events
    # the kept backward's key flags: (saved, packed_grads) + the hand-off layout
    kept_graphs = [e for kind, e in captured if kind == "bwd" and len(e.key[6]) > 2]
    assert len(kept_graphs) == 1 and kept_graphs[0].graph.replays == 1, (
        captured,
        events,
    )
    kept_graph = kept_graphs[0]
    hook, held = None, None
    if case == "other_call":

        def hook():
            kcp.cake_gdn_nondet_train_kcp_forward(
                *(other[n] for n in LEAVES), ctx, workspace=ws, **o
            )

    elif case in ("data_ptr", "shape", "stride"):
        original_put = kcp._kept_put

        def put(key, entry):
            t = entry.tapes["u"]
            entry.tapes["u"] = {
                "data_ptr": t.clone(),
                "shape": t.reshape(-1),
                "stride": t.as_strided(t.shape, (0,) * t.dim()),
            }[case]
            original_put(key, entry)

        monkeypatch.setattr(kcp, "_kept_put", put)
    else:  # the arena's "u" released and bound anew at another address (the old buffer is held)
        held = ws.tape("u")
        ws.release_tape("u")
    stale, graph0 = dict(kcp.KEPT_STALE_STATS), dict(kcp.GRAPH_STATS)
    result, bwd = step(hook)
    assert kept_graph.graph is None or kept_graph.graph.replays == 1, bwd
    assert bwd["graph_event"] != "replay", bwd
    stale_delta = {k: kcp.KEPT_STALE_STATS[k] - stale[k] for k in stale}
    if case == "other_call":
        assert stale_delta == {"calls": 1, "arena": 0, "stream": 0}, stale_delta
        assert bwd["tapes_source"] == "replay", bwd
    elif case == "arena_reused":
        assert held is not None and held.data_ptr() != ws.tape("u").data_ptr()
        assert kcp.GRAPH_STATS["replay"] == graph0["replay"], "a graph replayed"
        assert kcp.GRAPH_STATS["reset"] > graph0["reset"]
    else:
        assert stale_delta == {"calls": 0, "arena": 1, "stream": 0}, stale_delta
        assert bwd["tapes_source"] == "replay", bwd
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(result[n], clean[n]), n


def _same_leaf_steps(kcp, ctx, ops, ws, o):
    """A step function of :func:`torch.utils.checkpoint` steps of one layer on ``ws`` with the same leaf tensors in
    every step (a graph key holds their addresses); each step returns copies of out and the five gradients (a replay
    returns the capture's static results) and the backward's record."""
    leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]

    def step() -> tuple[dict, dict]:
        from torch.utils.checkpoint import checkpoint

        for lf in leaves:
            lf.grad = None
        out = checkpoint(
            lambda *a: kcp.cake_gdn_nondet_kcp(*a, ctx, workspace=ws, **o),
            *leaves,
            use_reentrant=False,
        )
        out.backward(ops["dout"])
        return {
            "out": out.detach().clone(),
            **{n: lf.grad.clone() for n, lf in zip(CKPT_GRADS, leaves, strict=True)},
        }, dict(kcp.LAST_CALL["bwd"])

    return step


def test_graph_replay_kept_backward_capture_failing_after_its_body_runs_the_kept_eager_form(
    monkeypatch,
) -> None:
    """A backward capture that fails after its body ran (the body's #8 released the kept set) falls back to the eager
    form on the very kept set: the fallback binds the kept tapes (``tapes_source`` ``"kept"``, one release counted),
    and out and the gradients are the bits of the eager step.  CPU captures: the forwards' succeed
    (:func:`_fake_captures`), every backward capture runs its body and then fails."""
    kcp, ctx, ops = _ckpt_setup()
    captured = _fake_captures(kcp, monkeypatch)
    fake_capture = kcp._graph_capture
    failed: list = []

    def capture(kind, entry, call, body):
        if kind == "fwd":
            return fake_capture(kind, entry, call, body)
        # the captured body ran (its #8 released the kept set), then the capture failed
        body()
        entry.disabled = "RuntimeError: capture failed after its body (test)"
        kcp.GRAPH_STATS["fallback"] += 1
        failed.append(entry)
        return None

    monkeypatch.setattr(kcp, "_graph_capture", capture)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    step = _same_leaf_steps(
        kcp,
        ctx,
        ops,
        ws,
        dict(backend="fake", graph="replay", reuse_recompute_tapes=True),
    )
    kcp.graph_cache_clear()
    reference, _ = step()  # the eager step (no entry seen yet)
    for _ in range(5):
        before = dict(kcp.KEPT_STATS)
        result, bwd = step()
        if failed:
            break
    # the kept backward's capture failed (its key flags end with the hand-off layout)
    assert failed and len(failed[-1].key[6]) > 2, (failed, captured)
    assert bwd["tapes_source"] == "kept" and bwd["graph_event"] == "eager", bwd
    assert str(bwd["graph_fallback"]).startswith("RuntimeError: capture failed"), bwd
    delta = _stats_delta(kcp, before)
    assert (delta["take"], delta["miss"], delta["released"]) == (1, 0, 1), delta
    for n in ("out",) + CKPT_GRADS:
        assert torch.equal(result[n], reference[n]), n
    kcp.graph_cache_clear()


def test_graph_replay_cached_hand_off_holds_no_arena_tensor(monkeypatch) -> None:
    """A captured hand-off forward caches its saved set as the template of its replays with the kept set's layout
    only: once the workspace drops its tapes and pools, no tensor of the arena stays alive through the graph cache
    (the cached graphs are still there).  CPU captures (:func:`_fake_captures`)."""
    import gc
    import weakref

    kcp, ctx, ops = _ckpt_setup()
    captured = _fake_captures(kcp, monkeypatch)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    step = _same_leaf_steps(
        kcp,
        ctx,
        ops,
        ws,
        dict(backend="fake", graph="replay", reuse_recompute_tapes=True),
    )
    kcp.graph_cache_clear()
    events = []
    for _ in range(6):
        _, bwd = step()
        events.append(bwd["graph_event"])
        if bwd["graph_event"] == "replay" and bwd["tapes_source"] == "kept":
            break
    assert events[-1] == "replay", events
    hand_off = []
    for kind, e in captured:
        cached = None if e.result is None else e.result[1]
        if kind == "fwd" and cached is not None and cached.kept is not None:
            hand_off.append(e)
    assert hand_off, captured
    # the context's own constants (the identity plane ``eye``) are bound into the pools but owned by the context,
    # which the graph keys pin; every other pool and every tape is the workspace's arena
    constants = [c for c in ctx._caches.values() if isinstance(c, torch.Tensor)]
    arena = [
        (n, weakref.ref(t))
        for n, t in (*ws._tapes.items(), *ws._pools.items())
        if not any(t is c for c in constants)
    ]
    assert {n for n, _ in arena} >= {*ws._tapes, "stage_m"}, sorted(ws._pools)
    for n in list(ws._tapes):
        ws.release_tape(n)
    for n in list(ws._pools):
        ws.release_pool(n)
    gc.collect()
    assert all(e.key in kcp._GRAPHS for e in hand_off)
    alive = [(n, tuple(r().shape)) for n, r in arena if r() is not None]
    assert not alive, alive
    kcp.graph_cache_clear()


def test_graph_cache_keeps_the_keys_of_24_checkpointed_layers_and_drops_the_least_recently_used(
    monkeypatch,
) -> None:
    """The graph cache keeps the three keys of each of 24 checkpointed layers (72 keys, cycled every step): after the
    eager and the capture step every lookup replays (CPU stand-in captures: the entry gets a :class:`_FakeGraph` as
    the forward / backward entries do).  Beyond its bound the least recently used key goes first (a hit refreshes)."""
    kcp, ctx, _ = _ckpt_setup()
    opt = _package("cake_options")
    ws = kcp.CakeGDNNondetKcpWorkspace()
    backend = kcp.get_backend("fake")
    opts = opt.CakeGDNNondetKcpOptions(backend="fake", graph="replay").check()
    kinds = (
        ("fwd", (True,)),
        ("fwd", (True, True, False)),
        ("bwd", (True, True, True)),
    )
    layers = [tuple(torch.empty(1) for _ in range(5)) for _ in range(24)]
    kcp.graph_cache_clear()
    replays = []
    for _ in range(4):
        n = 0
        for tensors in layers:
            for kind, flags in kinds:
                e = kcp._graph_entry(kind, ctx, ws, backend, opts, None, tensors, flags)
                if e.replayable(ws):
                    n += 1
                elif e.seen > 0:  # the capture
                    e.graph, e.arena = _FakeGraph(), kcp._arena_sig(ws)
                    e.seen += 1
                else:  # the eager call
                    e.seen += 1
        replays.append(n)
    assert replays == [0, 0, 72, 72], replays
    monkeypatch.setattr(kcp, "GRAPH_CACHE_MAX", 3)
    kcp.graph_cache_clear()
    keep = [torch.empty(1) for _ in range(4)]

    def entry(i):
        return kcp._graph_entry("fwd", ctx, ws, backend, opts, None, (keep[i],), (i,))

    a, b, _c = entry(0), entry(1), entry(2)
    assert entry(0) is a  # a hit: key 0 becomes the most recently used
    entry(3)  # beyond the bound: key 1 (least recently used) goes, key 0 stays
    assert a.key in kcp._GRAPHS and b.key not in kcp._GRAPHS, list(kcp._GRAPHS)
    assert len(kcp._GRAPHS) == 3
    kcp.graph_cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# Recompute handle (issue #7818): the checkpoint recompute reads the original forward's incoming states from the
# handle instead of exchanging them again.  These CPU rows run one rank (no crossing fragment, ``S_in`` empty): they
# check the code path -- the capture / replay decision, the local carries at ``P > 1``, the accuracy budget, retention
# and misuse; the multi-GPU rows (test_cake_gdn_nondet_train_kcp_dist.py) check the exchange count and the retained bytes.


def _handle_autograd(
    kcp, ctx, ops, *, handle, twice: bool = False, **opts
) -> tuple[dict[str, torch.Tensor], list]:
    """One ``cake_gdn_nondet_kcp`` layer under a non-reentrant checkpoint with ``recompute_handle=handle`` (``None``: the
    plain checkpoint); returns out + gradients and the ``recompute_handle`` mode of the original forward and of the
    recompute of every backward (read from ``LAST_CALL`` after each: torch's early stop ends the recompute inside the
    node's ``apply``, after the forward recorded itself)."""
    leaves = [
        ops[n].detach().clone().requires_grad_(True)
        for n in ("q", "k", "v", "g", "beta")
    ]
    o = {"backend": "fake", "save_transfer": False, **opts}

    def fn(q, k, v, g, beta, h):
        return kcp.cake_gdn_nondet_kcp(q, k, v, g, beta, ctx, recompute_handle=h, **o)

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
def test_recompute_handle_is_within_budget_of_the_checkpoint_and_the_direct_pass(
    monkeypatch, segments, save_mode, reuse
) -> None:
    """The original forward stores ``S_in`` ("capture"), torch's recompute reads it ("replay"): out and the five
    gradients are within the accuracy budget of the FP64 reference and of the plain checkpoint's and the direct
    pass's at the same options, at ``P = 1`` and ``P = 2`` (the carries rebuilt locally), with and without the kept
    tapes.  ``save_mode="full"`` at ``P = 2`` (the handle's recompute writes the fresh saved ``M_seg^T`` plane with
    its own transfer walk; the full backward runs none and reads that plane) runs with every package allocation
    filled with NaN bytes and gives the light direct pass's bits in the direct, the plain checkpoint and the handle
    arms (the fake backend is deterministic)."""
    kcp, ctx, ops = _ckpt_setup()
    if save_mode == "full":
        _nan_filled_allocations(monkeypatch, kcp)
    base = dict(segments=segments, reuse_recompute_tapes=reuse, save_mode=save_mode)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_transfer=False, **base)
    plain, plain_modes = _handle_autograd(kcp, ctx, ops, handle=None, **base)
    handle = kcp.CakeGDNNondetKcpRecomputeHandle()
    got, modes = _handle_autograd(kcp, ctx, ops, handle=handle, **base)
    if save_mode == "full":
        assert kcp.LAST_CALL["fwd"]["P"] == kcp.LAST_CALL["bwd"]["P"] == 2
        assert kcp.LAST_CALL["bwd"]["tapes_source"] == "saved"
        light = _ckpt_autograd(
            kcp,
            ctx,
            ops,
            ckpt=False,
            save_transfer=False,
            segments=segments,
            reuse_recompute_tapes=reuse,
        )
        for what, arm in (
            ("full direct", direct),
            ("full plain checkpoint", plain),
            ("full handle checkpoint", got),
        ):
            for n in light:
                assert torch.equal(arm[n], light[n]), (what, n)
    _assert_close_arms(plain, direct, "plain checkpoint vs direct")
    _assert_close_arms(got, direct, "handle checkpoint vs direct")
    assert plain_modes == [None, None], plain_modes
    assert modes == ["capture", "replay"], modes
    assert kcp.HANDLE_KEY in handle.entries
    assert kcp.kept_registry_size() == 0


@pytest.mark.parametrize("segments", (1, 2))
def test_recompute_handle_fused_recompute_under_a_later_saver(
    monkeypatch, segments
) -> None:
    """A recompute handle in a checkpointed region with a later saver (a scaling by a leaf): torch's recompute
    launches the output (the frame proof sees the later saver), so without a hand-off (``reuse_recompute_tapes``
    ``False`` / ``"auto"``) it runs the output walk ``kcp_fwdh_out`` from the handle's ``S_in`` -- at ``P = 2`` one
    fused walk over every live segment from the carry plane -- and the later saver reads that output.  With every
    package allocation filled with NaN bytes, out, the five gradients and the scale's gradient are the bits of the
    split form (``fused_o=0``) and within the budget of the FP64 reference; ``True`` hands the tapes off (a split
    recompute)."""
    from torch.utils.checkpoint import checkpoint

    kcp, ctx, ops = _ckpt_setup()
    _nan_filled_allocations(monkeypatch, kcp)
    ref, ref_scale = _region_reference(True)
    for reuse in (False, "auto", True):
        arms = {}
        for fused in (1, 0):
            leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]
            scale = torch.full((), 1.5, dtype=torch.bfloat16, requires_grad=True)
            o = dict(
                backend="fake",
                save_transfer=False,
                segments=segments,
                reuse_recompute_tapes=reuse,
                test_knobs={"fused_o": fused},
            )

            def fn(q, k, v, g, beta, h, _o=o, _scale=scale):
                out = kcp.cake_gdn_nondet_kcp(
                    q, k, v, g, beta, ctx, recompute_handle=h, **_o
                )
                return out * _scale

            handle = kcp.CakeGDNNondetKcpRecomputeHandle()
            out = checkpoint(fn, *leaves, handle, use_reentrant=False)
            modes = [kcp.LAST_CALL["fwd"]["recompute_handle"]]
            out.backward(ops["dout"])
            rec = kcp.LAST_CALL["fwd"]  # torch's recompute: the last forward
            modes.append(rec["recompute_handle"])
            assert modes == ["capture", "replay"], (reuse, fused, modes)
            assert rec["P"] == segments, (reuse, fused, rec["P"])
            res = {"out": out.detach()}
            res.update({n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)})
            arms[fused] = (res, scale.grad, rec["fwd_o"])
        (r1, s1, o1), (r0, s0, o0) = arms[1], arms[0]
        assert o1 == ("launched" if reuse is True else "fused"), (reuse, o1)
        assert o0 == "launched", (reuse, o0)
        what = f"handle, later saver, P={segments}, reuse={reuse!r}"
        for n in ("out",) + CKPT_GRADS:
            assert torch.isfinite(r1[n].float()).all(), (what, n)
            assert torch.equal(r1[n], r0[n]), (what, n)
        assert torch.isfinite(s1.float()).all() and torch.equal(s1, s0), what
        _assert_close_arms(r1, r0, what, ref)
        _within_budget(s1, ref_scale, f"{what}: dscale vs FP64")
        assert kcp.kept_registry_size() == 0, what


def test_recompute_handle_retains_exactly_s_in_and_survives_a_second_backward() -> None:
    """The entry is the saved set's own ``S_in`` (no copy; ``nbytes`` is its size and the light byte query without
    the transfer and the convolution rows); ``retain_graph`` + a second backward reuses it and accumulates the
    gradient to within the budget of twice the FP64 gradient."""
    kcp, ctx, ops = _ckpt_setup()
    H_qk, HV = ops["q"].shape[-2], ops["v"].shape[-2]
    handle = kcp.CakeGDNNondetKcpRecomputeHandle()
    once, _ = _handle_autograd(
        kcp, ctx, ops, handle=kcp.CakeGDNNondetKcpRecomputeHandle()
    )
    twice, modes = _handle_autograd(kcp, ctx, ops, handle=handle, twice=True)
    assert modes == ["capture", "replay", "replay"], modes
    _assert_accumulated(twice, once, "handle, second backward", _ckpt_reference())
    s_in = handle.entries[kcp.HANDLE_KEY]["S_in"]
    assert handle.nbytes() == s_in.numel() * s_in.element_size()
    assert handle.nbytes() == kcp.cake_gdn_nondet_train_kcp_saved_bytes(
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
    handle = kcp.CakeGDNNondetKcpRecomputeHandle()
    kcp.cake_gdn_nondet_kcp(
        q,
        k,
        v,
        g,
        beta,
        ctx,
        backend="fake",
        recompute_handle=handle,
    )
    assert kcp.LAST_CALL["fwd"]["recompute_handle"] == "capture"
    with pytest.raises(
        RuntimeError, match="new CakeGDNNondetKcpRecomputeHandle per micro-step"
    ):
        kcp.cake_gdn_nondet_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            backend="fake",
            recompute_handle=handle,
        )
    leaves = [x.detach().clone().requires_grad_(True) for x in (q, k, v, g, beta)]
    h2 = kcp.CakeGDNNondetKcpRecomputeHandle()
    from torch.utils.checkpoint import checkpoint

    def fn(*a):
        return kcp.cake_gdn_nondet_kcp(
            *a[:5],
            ctx,
            backend="fake",
            save_transfer=False,
            recompute_handle=a[5],
        )

    out = checkpoint(fn, *leaves, h2, use_reentrant=False)
    h2.clear()
    out.backward(ops["dout"])
    assert kcp.LAST_CALL["fwd"]["recompute_handle"] == "capture"
    assert kcp.HANDLE_KEY in h2.entries
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, save_transfer=False)
    _arms_within_budget(
        {
            "recompute without its entry": {
                n: lf.grad for n, lf in zip(CKPT_GRADS, leaves, strict=True)
            },
            "direct": {n: direct[n] for n in CKPT_GRADS},
        },
        _ckpt_reference(),
        CKPT_GRADS,
        "handle that lost its entry",
    )
    for bad in (
        {"save_transfer": True},
        {"options": {"save_transfer": True}},
        {"boundary_mode": "replay"},
        {"graph": "replay", "workspace": kcp.CakeGDNNondetKcpWorkspace()},
        {"test_knobs": {"debug_sequence": True}},
    ):
        with pytest.raises(ValueError):
            kcp.cake_gdn_nondet_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                backend="fake",
                recompute_handle=kcp.CakeGDNNondetKcpRecomputeHandle(),
                **bad,
            )
    with pytest.raises(TypeError, match="entries"):
        kcp.cake_gdn_nondet_kcp(
            q,
            k,
            v,
            g,
            beta,
            ctx,
            backend="fake",
            recompute_handle=object(),
        )


# --------------------------------------------------------------------------- the multi-rank row (W ranks on threads)
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
                lambda: (
                    comm.failure is not None
                    or len(comm.deposits.get(self.seq, {})) == comm.world
                ),
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
        recording = getattr(self.local, "recording", None)
        # a capture of _fake_captures_threaded: its replays re-issue this collective
        if recording is not None:
            recording.append((output, input.clone(), recv, send))
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
    """A recording fake backend: per rank (the calling thread's, 0 without threads) the :class:`CakeGDNNondetKcpCall` objects in
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
    """``cake_gdn_nondet_train_kcp_workspace_layout(with_conv=False)`` is the most a caller-held workspace binds on each rank in the
    forward on a fresh arena and in the backward that follows on the same workspace (the explicit entries; the
    backward's replay form), every pool term is the size the call gives that pool, and ``cake_gdn_nondet_train_kcp_saved_bytes`` is
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(list(cu), dtype=torch.int64), W, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
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
        _, saved = kcp.cake_gdn_nondet_train_kcp_forward(
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
        kcp.cake_gdn_nondet_train_kcp_backward(
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
        lay = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
            assert got[r]["saved"] == gb.cake_gdn_nondet_train_kcp_saved_bytes(
                c, **q, **l2n
            ), (r, got[r])
        else:
            assert got[r]["saved"] == 0
        if segments > 1:
            assert lay["fwd.c1_send_mf"] == c.n_out * HV * gp.STATE_TILE_BYTES
        if opts.get("boundary_mode") == "replay":
            assert got[r]["saved"] == 0
            light = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
                assert lay == gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
    [
        {},
        _FULL_NO_ST,
        {"boundary_mode": "replay"},
        {"save_for_backward": False},
        {"save_mode": "light_akk", "use_qk_l2norm_in_kernel": True},
    ],
    ids=["light", "full", "replay", "no_saved_set", "light_akk"],
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
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 2, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
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
            _, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                *args, ctxs[r], save_for_backward=with_set, **kw
            )
            fwd = wss[r].peak
            wss[r].peak = 0
            kcp.cake_gdn_nondet_train_kcp_backward(
                *args, ctxs[r], loc["dout"], saved=saved, **kw
            )
            steps[r].append((fwd, wss[r].peak))

    _run_threaded(kcp, 2, rank_fn, monkeypatch)
    for r, c in enumerate(ctxs):
        q = dict(H_qk=H_qk, HV=HV, P=2, with_conv=False, reuse_recompute_tapes=False)
        eager = gb.cake_gdn_nondet_train_kcp_workspace_layout(c, **q, **opts)
        graph = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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


@pytest.mark.parametrize("save_transfer", [True, False])
def test_two_rank_light_akk_is_bitwise_light(monkeypatch, save_transfer) -> None:
    """``save_mode="light_akk"`` on two ranks (a crossing fragment, ``P = 2``, the in-kernel norm, fake backend):
    out and the five gradients of every rank are the bits of ``"light"`` through the explicit entries and through
    autograd (the saved set through ``save_for_backward`` / ``CakeGDNNondetKcpSaved.from_tensors``), eager and in four
    ``graph="replay"`` steps on the same caller-held workspace per rank (CPU captures whose replays re-issue their
    collectives; the last step replays both graphs).  The backward's launch census is light's with ``fwd_prep_l2n``
    swapped for ``fwd_prep_replay_l2n`` once per light replay (the forward's is light's), and each rank's saved set
    gains its ``Akk`` tape, ``128 R HV`` bytes, the byte query's difference."""
    kcp = _package("cake_kcp")
    H_qk, HV = CKPT_HEADS
    replay = kcp.AKK_REPLAY_OF["fwd_prep_l2n"]
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 2, "contiguous"
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(2)
    ]
    assert sum(c.n_out for c in ctxs) > 0 and all(int(c.T_r) > 0 for c in ctxs)
    # the kernels write every element of a per-call buffer; the fake backend writes each Akk block's valid rows and
    # columns only, so zero-filled buffers keep the unwritten entries equal across the arms
    monkeypatch.setattr(
        kcp,
        "_alloc",
        lambda shape, dtype, device: torch.zeros(
            tuple(shape), dtype=dtype, device=device
        ),
    )
    monkeypatch.setattr(kcp, "LAST_CALL", _ThreadLastCall())
    _fake_captures_threaded(kcp, monkeypatch)
    ops = _inputs(CKPT_CU, H_qk, HV)
    base = dict(
        backend="fake",
        segments=2,
        save_transfer=save_transfer,
        use_qk_l2norm_in_kernel=True,
    )
    graphs = ("off",) + ("replay",) * 4

    def run(**o) -> list:
        kcp.graph_cache_clear()

        def rank_fn(r, comm):
            loc = fake.slice_local(ops, part, r)
            args = [loc[n] for n in LEAVES]
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                *args, ctxs[r], **base, **o
            )
            fwd = dict(kcp.LAST_CALL["fwd"])
            grads = kcp.cake_gdn_nondet_train_kcp_backward(
                *args,
                ctxs[r],
                loc["dout"],
                saved=saved,
                **base,
                **o,
            )
            explicit = {
                "out": out,
                **dict(zip(EXPLICIT_GRADS, grads, strict=True)),
                "saved_bytes": int(saved.bytes),
                "fwd": fwd,
                "bwd": dict(kcp.LAST_CALL["bwd"]),
            }
            leaves = [loc[n].detach().clone().requires_grad_(True) for n in LEAVES]
            ws = kcp.CakeGDNNondetKcpWorkspace()
            steps = []
            for graph in graphs:
                for lf in leaves:
                    lf.grad = None
                y = kcp.cake_gdn_nondet_kcp(
                    *leaves,
                    ctxs[r],
                    workspace=ws,
                    graph=graph,
                    **base,
                    **o,
                )
                y.backward(loc["dout"])
                res = {
                    "out": y.detach().clone(),
                    **{
                        n: lf.grad.clone()
                        for n, lf in zip(CKPT_GRADS, leaves, strict=True)
                    },
                }
                steps.append(
                    (res, dict(kcp.LAST_CALL["fwd"]), dict(kcp.LAST_CALL["bwd"]))
                )
            return explicit, steps

        return _run_threaded(kcp, 2, rank_fn, monkeypatch)

    light = run(save_mode="light")
    akk = run(save_mode="light_akk")
    default = run()  # the norm's default save mode: "light_akk"
    kcp.graph_cache_clear()
    names = ("out",) + CKPT_GRADS
    for r, c in enumerate(ctxs):
        (le, lsteps), (ae, asteps) = light[r], akk[r]
        for n in names:
            assert torch.equal(ae[n], le[n]), (r, "explicit", n)
        lb, ab = le["bwd"], ae["bwd"]
        assert ab["akk_source"] == "saved" and lb["akk_source"] is None, r
        assert ab["tapes_source"] == lb["tapes_source"] == "replay", r
        assert ae["fwd"]["sequence"] == le["fwd"]["sequence"], r
        assert lb["sequence"].count("fwd_prep_l2n") >= 1, r
        assert ab["sequence"] == [
            replay if p == "fwd_prep_l2n" else p for p in lb["sequence"]
        ], r
        assert "fwd_prep_l2n" not in ab["sequence"], r
        q = dict(
            H_qk=H_qk,
            HV=HV,
            P=2,
            save_transfer=save_transfer,
            with_conv=False,
            use_qk_l2norm_in_kernel=True,
        )
        tape = int(c.R) * HV * 128
        assert ae["saved_bytes"] - le["saved_bytes"] == tape, r
        assert (
            gb.cake_gdn_nondet_train_kcp_saved_bytes(c, save_mode="light_akk", **q)
            - gb.cake_gdn_nondet_train_kcp_saved_bytes(c, save_mode="light", **q)
            == tape
        ), r
        assert gb.cake_gdn_nondet_train_kcp_saved_bytes(
            c, **q
        ) == gb.cake_gdn_nondet_train_kcp_saved_bytes(c, save_mode="light_akk", **q)
        de, dsteps = default[r]
        for n in names:
            assert torch.equal(de[n], ae[n]), (r, "default explicit", n)
        assert de["saved_bytes"] == ae["saved_bytes"], r
        assert de["bwd"]["akk_source"] == "saved", r
        assert de["bwd"]["sequence"] == ab["sequence"], r
        for i, ((dres, _df, dbw), (ares, _af, abw)) in enumerate(
            zip(dsteps, asteps, strict=True)
        ):
            for n in names:
                assert torch.equal(dres[n], ares[n]), (r, "default step", i, n)
            assert dbw["sequence"] == abw["sequence"], (r, i)
        for i, ((lres, _lf, lbw), (ares, _af, abw)) in enumerate(
            zip(lsteps, asteps, strict=True)
        ):
            for n in names:
                assert torch.equal(ares[n], lres[n]), (r, i, n)
                assert torch.equal(ares[n], ae[n]), (r, i, n)
            assert abw["akk_source"] == "saved", (r, i, abw["akk_source"])
            assert abw["sequence"] == [
                replay if p == "fwd_prep_l2n" else p for p in lbw["sequence"]
            ], (r, i)
        _res, af, abw = asteps[-1]
        assert af["graph_event"] == abw["graph_event"] == "replay", (r, af, abw)


class _ThreadLastCall(dict):
    """``cake_kcp.LAST_CALL`` per rank thread: the package only subscripts the module dict, which the rank threads of
    :func:`_run_threaded` would otherwise share, so each thread reads and writes its own records."""

    def __init__(self) -> None:
        super().__init__()
        self._local = threading.local()

    def _own(self) -> dict:
        own = getattr(self._local, "records", None)
        if own is None:
            own = self._local.records = {}
        return own

    def __getitem__(self, kind):
        return self._own()[kind]

    def __setitem__(self, kind, record) -> None:
        self._own()[kind] = record


class _CollectiveGraph(_FakeGraph):
    """CPU stand-in for a captured graph of one rank thread.  NCCL collectives are captured per rank, so its replay
    re-issues, in order, the collectives its capture issued -- the capture's payloads into the same receive buffers
    (every step of a test using it runs the same operands) -- and runs nothing else."""

    def __init__(self, comm, issued: list) -> None:
        super().__init__()
        self.comm, self.issued = comm, issued

    def replay(self) -> None:
        super().replay()
        for output, payload, recv, send in self.issued:
            self.comm.all_to_all_single(
                output, payload, output_split_sizes=recv, input_split_sizes=send
            )


def _fake_captures_threaded(kcp, monkeypatch) -> list:
    """:func:`_fake_captures` for the rank threads of :func:`_run_threaded`: the capture records the collectives its
    body issued on the calling rank's thread and the entry keeps a :class:`_CollectiveGraph` of them, so a rank that
    replays a graph meets its peers' eager or captured collectives in the same order.  Returns the
    ``(rank, kind, entry)`` triples captured."""
    captured: list = []
    lock = threading.Lock()

    def capture(kind, entry, call, body):
        comm = kcp.dist
        comm.local.recording = []
        try:
            result = body()
        finally:
            issued, comm.local.recording = comm.local.recording, None
        kcp._record(kind, call)
        kcp.LAST_CALL[kind].update(graph_event="capture", graph_key=entry.digest)
        entry.record = dict(kcp.LAST_CALL[kind])
        entry.graph = _CollectiveGraph(comm, issued)
        entry.arena = kcp._arena_sig(call.ws)
        entry.seen += 1
        kcp.GRAPH_STATS["capture"] += 1
        call.backend.end_call(call)
        with lock:
            captured.append((comm.local.rank, kind, entry))
        return result

    monkeypatch.setattr(kcp, "_graph_capture", capture)
    return captured


# two ranks, document 3 crossing: rank 0 holds 1517 tokens (9102 token x value-head rows at HV = 6), rank 1 holds 429
# (2574 rows); a size rule between them puts rank 1 under it
XR_FRAGMENTS = [
    [(0, 0, 100), (1, 0, 640), (2, 0, 77), (3, 0, 700)],
    [(3, 700, 300), (4, 0, 129)],
]
XR_RULE = 6000


@pytest.mark.parametrize("l2n", [False, True])
def test_graph_replay_checkpoint_group_with_one_rank_under_the_size_rule(
    l2n, monkeypatch
) -> None:
    """``graph="replay"``, ``torch.utils.checkpoint(use_reentrant=False)``, the default ``reuse_recompute_tapes="auto"``
    on two ranks (one crossing fragment, ``P = 2``, ``save_transfer=False``) with the size rule between the ranks'
    token x value-head rows: per rank and step the recompute of rank 0 decides (on a verified torch the hand-off,
    reason ``"dead"``, and the kept backward; elsewhere the replay form) and captures its own form, while rank 1 falls
    under the rule (no hand-off, reasons ``"small"``, the replay-form backward) and replays the original forward's
    graph -- one group mixing the two forms.  Five checkpointed steps on caller-held workspaces with the same leaves
    (CPU captures whose replays re-issue their collectives, :func:`_fake_captures_threaded`); the last step replays
    every graph on both ranks.  Out and the five gradients of every rank and step are the bits of
    ``reuse_recompute_tapes=False`` (every rank the replay form) and of ``graph="off"``."""
    kcp = _package("cake_kcp")
    H_qk, HV = CKPT_HEADS
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 2, fragments=XR_FRAGMENTS
    )
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(2)
    ]
    rows = [int(c.T_r) * HV for c in ctxs]
    assert rows[1] < XR_RULE <= rows[0], rows
    assert part.fragment(0, 3).length + part.fragment(1, 0).length == 1000
    _size_rule_at(kcp, monkeypatch, XR_RULE)
    monkeypatch.setattr(kcp, "LAST_CALL", _ThreadLastCall())
    captured = _fake_captures_threaded(kcp, monkeypatch)
    ops = _inputs(CKPT_CU, H_qk, HV)
    base = dict(backend="fake", segments=2, save_transfer=False)
    if l2n:
        base["use_qk_l2norm_in_kernel"] = True
    n_steps = 5

    def run(**o) -> tuple[list, list]:
        kcp.graph_cache_clear()
        first = len(captured)

        def rank_fn(r, comm):
            from torch.utils.checkpoint import checkpoint

            loc = fake.slice_local(ops, part, r)
            leaves = [loc[n].detach().clone().requires_grad_(True) for n in LEAVES]
            ws = kcp.CakeGDNNondetKcpWorkspace()
            steps = []
            for _ in range(n_steps):
                for lf in leaves:
                    lf.grad = None
                out = checkpoint(
                    lambda *a: kcp.cake_gdn_nondet_kcp(
                        *a, ctxs[r], workspace=ws, **base, **o
                    ),
                    *leaves,
                    use_reentrant=False,
                )
                out.backward(loc["dout"])
                res = {
                    "out": out.detach().clone(),
                    **{
                        n: lf.grad.clone()
                        for n, lf in zip(CKPT_GRADS, leaves, strict=True)
                    },
                }
                steps.append(
                    (res, dict(kcp.LAST_CALL["fwd"]), dict(kcp.LAST_CALL["bwd"]))
                )
            return steps

        per_rank = _run_threaded(kcp, 2, rank_fn, monkeypatch)
        return per_rank, captured[first:]

    auto, auto_captured = run(graph="replay")
    replay_form, _ = run(graph="replay", reuse_recompute_tapes=False)
    eager, _ = run()
    kcp.graph_cache_clear()
    verified = _verified_torch(kcp)
    for i in range(n_steps):
        (_, f0, b0), (_, f1, b1) = auto[0][i], auto[1][i]
        # rank 1 under the size rule: no decision, the original forward's form, the replay-form backward
        assert f1["hand_off"] is False, (i, f1)
        assert f1["hand_off_reason"] == f1["fwd_o_reason"] == "small", (i, f1)
        assert b1["tapes_source"] == "replay", (i, b1)
        # rank 0 decides from the frame: the hand-off and the kept backward on a verified torch
        assert f0["hand_off_reason"] == _frame_reason(kcp, "dead"), (i, f0)
        assert f0["hand_off"] is verified, (i, f0)
        assert b0["tapes_source"] == ("kept" if verified else "replay"), (i, b0)
    for r in range(2):
        _, f, b = auto[r][-1]
        assert f["graph_event"] == b["graph_event"] == "replay", (r, f, b)
    # the captured forward keys: rank 0's recompute its own (hand-off) key on a verified torch, rank 1 only the
    # original forward's
    fwd_flags = [
        {
            tuple(e.key[6])
            for rank, kind, e in auto_captured
            if rank == r and kind == "fwd"
        }
        for r in range(2)
    ]
    assert fwd_flags[1] == {(True,)}, fwd_flags
    assert fwd_flags[0] == (
        {(True,), (True, True, False)} if verified else {(True,)}
    ), fwd_flags
    for r in range(2):
        for i in range(n_steps):
            for n in ("out",) + CKPT_GRADS:
                a = auto[r][i][0][n]
                assert torch.equal(a, replay_form[r][i][0][n]), (r, i, n)
                assert torch.equal(a, eager[r][i][0][n]), (r, i, n)


@pytest.mark.parametrize(
    "segments,save_transfer,l2n",
    [
        (1, True, False),
        (2, True, False),
        (2, False, False),
        (2, True, True),
        (2, False, True),
    ],
)
def test_workspace_layout_follows_the_arena_of_the_graph_replay_kept_form(
    monkeypatch, segments, save_transfer, l2n
) -> None:
    """``graph="replay"`` under ``torch.utils.checkpoint(use_reentrant=False)`` on a caller-held workspace created
    empty (the first call of each key runs eagerly and releases nothing): the forward binds
    ``cake_gdn_nondet_train_kcp_workspace_layout(graph="replay")["forward"]`` exactly and the backward that follows -- the recompute
    forward and the GDN backward -- its ``backward`` exactly, with every pool at its term: ``True`` (the recompute's
    arena hand-off, the backward binds the kept tapes) the kept figure, ``False`` the replay form's, and ``"auto"``
    the kept figure at the rule's 0 rows on a verified torch (the hand-off) and the replay form's above the rank's
    rows (the size rule).  The query's ``"auto"`` (the replay form) bounds the call: equal on the replay side,
    larger by the backward's arena ``qk_rstd`` (0 without the norm) on the kept side."""
    from torch.utils.checkpoint import checkpoint

    kcp, ctx, ops = _ckpt_setup()
    _refuse_captures(kcp, monkeypatch)
    H_qk, HV = CKPT_HEADS
    rows = int(ctx.T_r) * HV
    call = dict(
        segments=segments,
        save_transfer=save_transfer,
        graph="replay",
        use_qk_l2norm_in_kernel=l2n,
    )

    def measure(reuse) -> dict:
        ws = _pool_peak_workspace(kcp)
        kcp.graph_cache_clear()
        leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]

        def fn(*a):
            return kcp.cake_gdn_nondet_kcp(
                *a,
                ctx,
                workspace=ws,
                backend="fake",
                reuse_recompute_tapes=reuse,
                **call,
            )

        out = checkpoint(fn, *leaves, use_reentrant=False)
        got = {"forward": ws.peak, "fwd_pools": dict(ws.pool_max)}
        ws.peak, ws.pool_max = 0, {}
        out.backward(ops["dout"])
        got.update(
            backward=ws.peak,
            bwd_pools=dict(ws.pool_max),
            source=kcp.LAST_CALL["bwd"]["tapes_source"],
        )
        kcp.graph_cache_clear()
        return got

    def layout(reuse) -> dict:
        # a checkpointed call saves as "light" (the default with the norm, "light_akk",
        # too), so the query sizes it with save_mode="light"
        return gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx,
            H_qk=H_qk,
            HV=HV,
            P=segments,
            save_transfer=save_transfer,
            with_conv=False,
            graph="replay",
            reuse_recompute_tapes=reuse,
            use_qk_l2norm_in_kernel=l2n,
            save_mode="light",
        )

    lay = {reuse: layout(reuse) for reuse in (True, False, "auto")}
    assert lay["auto"] == lay[False]
    arms = {True: measure(True), False: measure(False)}
    assert arms[True]["source"] == "kept" and arms[False]["source"] == "replay"
    if _verified_torch(kcp):
        arms["auto"] = measure("auto")
        assert arms["auto"]["source"] == "kept"
    _size_rule_at(kcp, monkeypatch, rows + 1)
    arms["auto_small"] = measure("auto")
    assert arms["auto_small"]["source"] == "replay"
    for arm, got in arms.items():
        form = lay[True] if got["source"] == "kept" else lay[False]
        assert got["forward"] == form["forward"], (arm, got["forward"], form)
        rec, want = (
            _recorded_pool_terms(got["fwd_pools"]),
            _layout_pool_terms(form, "fwd"),
        )
        assert {k: v for k, v in rec.items() if not k.startswith("b")} == {
            k: v for k, v in want.items() if not k.startswith("b")
        }, (arm, "fwd")
        assert got["backward"] == form["backward"], (arm, got["backward"], form)
        assert _recorded_pool_terms(got["bwd_pools"]) == _layout_pool_terms(
            form, "bwd"
        ), (arm, "bwd")
        if arm in ("auto", "auto_small"):
            # "auto" reports the replay form: an upper bound, equal on the replay side, larger by the backward's
            # arena qk_rstd on the kept side
            bound = lay["auto"]
            assert got["forward"] == bound["forward"], arm
            gap = bound["backward"] - got["backward"]
            if got["source"] == "kept":
                assert gap == lay[False].get("bwd.qk_rstd", 0), (arm, gap)
            else:
                assert gap == 0, (arm, gap)


def test_workspace_layout_option_checks() -> None:
    """The workspace queries refuse a ``graph`` value outside ``("off", "replay")`` and a non-bool
    ``save_for_backward``; under ``graph="replay"`` the default and ``"auto"`` report the replay form and ``True`` the
    arena hand-off (the replay form's terms without the backward's arena ``qk_rstd``, no hand-off tensor).  ``save_for_backward=False`` with a backward reports the replay a ``saved=None`` backward
    runs: the replay boundary's backward terms, in either ``save_mode`` and ``reuse_recompute_tapes`` form, except
    the replayed forward after the fused (light, saved boundary) forward, which finds no ``h`` / ``v_new``; the
    forward is the call's."""
    _, ctx, _ = _ckpt_setup()
    q = dict(H_qk=2, HV=6)
    assert gb.GRAPH_MODES == _package("cake_options").GRAPH_MODES
    with pytest.raises(ValueError, match="graph"):
        gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, graph="on", **q)
    for l2n in (False, True):
        # the forms of a checkpointed call, which saves as "light" (the default with
        # the norm, "light_akk", too)
        g = dict(graph="replay", use_qk_l2norm_in_kernel=l2n, save_mode="light", **q)
        replay = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, reuse_recompute_tapes=False, **g
        )
        assert gb.cake_gdn_nondet_train_kcp_workspace_layout(ctx, **g) == replay
        assert (
            gb.cake_gdn_nondet_train_kcp_workspace_layout(
                ctx, reuse_recompute_tapes="auto", **g
            )
            == replay
        )
        kept = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, reuse_recompute_tapes=True, **g
        )
        rstd = replay.get("bwd.qk_rstd", 0)
        assert (rstd > 0) == l2n
        assert kept == {
            **replay,
            **({"bwd.qk_rstd": 0} if l2n else {}),
            "bwd.phase.graph": replay["bwd.phase.graph"] - rstd,
            "backward": replay["backward"] - rstd,
            "backward_transient": replay["backward_transient"] - rstd,
            "workspace": max(replay["forward"], replay["backward"] - rstd),
        }
        assert (
            gb.cake_gdn_nondet_train_kcp_workspace_bytes(
                ctx, reuse_recompute_tapes=True, **g
            )
            == kept["workspace"]
        )
    for save_mode, reuse in itertools.product(("light", "full"), (True, False)):
        lay = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx,
            save_for_backward=False,
            save_mode=save_mode,
            reuse_recompute_tapes=reuse,
            P=2,
            **q,
        )
        rep = gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, boundary_mode="replay", P=2, **q
        )
        only_fwd = gb.cake_gdn_nondet_train_kcp_workspace_layout(
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
        gb.cake_gdn_nondet_train_kcp_workspace_layout(
            ctx, save_for_backward=0, backward=False, **q
        )


def _mr_setup():
    kcp = _package("cake_kcp")
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(MR_CU, dtype=torch.int64), MR_W, "zigzag_per_document"
    )
    return kcp, part, _mr_inputs()


def _mr_inputs() -> dict[str, torch.Tensor]:
    ops = dict(_inputs(MR_CU, *MR_HEADS))
    ops["g"] = (ops["g"] * MR_G_SCALE).contiguous()
    return ops


def _mr_reference() -> dict[str, torch.Tensor]:
    key = ("multi_rank",)
    if key not in _REFERENCES:
        ops = _mr_inputs()
        out, grads = _fp64(_ref_layer(MR_CU), [ops[n] for n in LEAVES], ops["dout"])
        _REFERENCES[key] = {"out": out, **dict(zip(CKPT_GRADS, grads, strict=True))}
    return _REFERENCES[key]


def _mr_run(kcp, part, ops, monkeypatch, *, sm_count: int = SM_COUNT, knobs=None):
    """Forward + saved-set backward of every rank of the multi-rank row on the shipped options (plus ``knobs`` on both
    entries): the global-order results, the per-rank calls (forward, backward) and the launched programs per call."""
    knobs = dict(knobs or {})
    holder: dict = {}
    calls, launched, rec = _call_recorder(kcp, lambda: holder.get("comm"))
    # group None: the stand-in collective ignores the group (the rank is the calling thread's)
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=sm_count
        )
        for r in range(MR_W)
    ]

    def rank_fn(r, comm):
        holder["comm"] = comm
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            *args, ctxs[r], backend="fake", **knobs
        )
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
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


def test_multi_rank_row_is_within_budget(monkeypatch) -> None:
    """The multi-rank row (one document over W = 8 ranks, ``zigzag_per_document``, SM count 152) on the shipped
    options: every rank runs one forward call and its backward, and out and the five gradients are within the budget
    of the FP64 reference."""
    kcp, part, ops = _mr_setup()
    assert sum(part.crossing_counts()[0]) > 0  # the row exchanges
    res, _ctxs, calls, _launched = _mr_run(kcp, part, ops, monkeypatch)
    for r in range(MR_W):
        fwd_calls = [c for c in calls[r] if not c.backward]
        bwd_calls = [c for c in calls[r] if c.backward]
        assert len(fwd_calls) == 1 and bwd_calls, (r, len(calls[r]))
    _arms_within_budget(
        {"shipped": res}, _mr_reference(), ("out",) + EXPLICIT_GRADS, "multi-rank row"
    )


@pytest.mark.parametrize("l2n", (False, True), ids=("norm_off", "l2n"))
def test_launch_recorder_lists_the_programs_of_the_call(monkeypatch, l2n) -> None:
    """Every call's launch log (``CakeGDNNondetKcpCall.sequence``, recorded in ``LAST_CALL``) lists exactly the programs the
    backend ran, in order, on every rank of the W = 8 row and on one rank without an exchange.  With
    ``use_qk_l2norm_in_kernel=True`` (#7817) the same holds; the forward runs #1_l2n and the backward #8_l2n in the
    places of #1 / #8, and every other launch -- this package's plan (``P_RULE_DEFAULT`` on every rank), its walks,
    merges and exchanges -- is the norm-off launch log's: the option replaces those two programs and nothing else of
    this route."""
    # the light route: the default with the norm, "light_akk", replays #1 with
    # fwd_prep_replay_l2n (test_cake_gdn_nondet_train_kcp_light_akk.py)
    knobs = {"use_qk_l2norm_in_kernel": True, "save_mode": "light"} if l2n else {}
    kcp, part, ops = _mr_setup()
    _res, ctxs, calls, launched = _mr_run(kcp, part, ops, monkeypatch, knobs=knobs)
    for r in range(MR_W):
        assert len(calls[r]) >= 2, (
            r,
            len(calls[r]),
        )  # the forward's call and the backward's
        for call in calls[r]:
            ran = launched[id(call)]
            assert ran == [n for n in call.sequence if n in kcp.PROGRAMS], r
    if l2n:
        assert all(ctx.p_rule == gp.P_RULE_DEFAULT for ctx in ctxs)
        _res_off, _ctxs_off, calls_off, launched_off = _mr_run(
            kcp, part, ops, monkeypatch
        )
        sib = kcp.L2N_SIBLING_OF
        for r in range(MR_W):
            on = [launched[id(c)] for c in calls[r]]
            off = [[sib.get(n, n) for n in launched_off[id(c)]] for c in calls_off[r]]
            assert on == off, r
            assert any("fwd_prep_l2n" in seq for seq in on), r
            assert any("bwd_dqkg_l2n" in seq for seq in on), r
            assert not any({"fwd_prep", "bwd_dqkg"} & set(seq) for seq in on), r
    # one rank, no exchange: the recorded launches are the call's sequence too, and LAST_CALL carries it
    kcp1, ctx, ops1 = _ckpt_setup()
    calls1, launched1, rec = _call_recorder(kcp1)
    previous = kcp1.set_backend("fake", rec)
    try:
        out, saved = kcp1.cake_gdn_nondet_train_kcp_forward(
            *(ops1[n] for n in LEAVES),
            ctx,
            backend="fake",
            **knobs,
        )
        fwd_seq = list(kcp1.LAST_CALL["fwd"]["sequence"])
        fwd_l2n = bool(kcp1.LAST_CALL["fwd"]["qk_l2norm"])
        kcp1.cake_gdn_nondet_train_kcp_backward(
            *(ops1[n] for n in LEAVES),
            ctx,
            ops1["dout"],
            saved=saved,
            backend="fake",
            **knobs,
        )
        bwd_seq = list(kcp1.LAST_CALL["bwd"]["sequence"])
    finally:
        kcp1.set_backend("fake", previous)
    assert len(calls1[0]) == 2, len(calls1[0])
    fwd_call, bwd_call = calls1[0]
    assert launched1[id(fwd_call)] == [n for n in fwd_seq if n in kcp1.PROGRAMS]
    assert launched1[id(bwd_call)] == [n for n in bwd_seq if n in kcp1.PROGRAMS]
    prep, dqkg = ("fwd_prep_l2n", "bwd_dqkg_l2n") if l2n else ("fwd_prep", "bwd_dqkg")
    # the light forward writes the output with the output walk where the registry ships it, else with #5
    o_writer = "kcp_fwdh_out" if kcp1.fused_o_shipped() else "fwd_o"
    assert {prep, o_writer} <= set(fwd_seq) and {"bwd_prep", dqkg} <= set(bwd_seq)
    assert ({"fwd_o", "kcp_fwdh_tape"} & set(fwd_seq)) == (
        set() if o_writer == "kcp_fwdh_out" else {"fwd_o", "kcp_fwdh_tape"}
    )
    assert fwd_l2n == l2n


# --------------------------------------------------------------------------- host launch path: launch cache, stream scope
# Host data only (record names, grids, launch order, context entries): exact checks.


def _harvest_launches(kcp, ctx, ops, **knobs) -> list[tuple[str, dict]]:
    """``(program, args)`` of every launch of one forward + backward on the fake backend (``args`` without the call)."""
    seen: list[tuple[str, dict]] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            seen.append((name, {k: v for k, v in args.items() if k != "_call"}))
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    try:
        args = [ops[n] for n in LEAVES]
        _out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            *args, ctx, backend="fake", **knobs
        )
        kcp.cake_gdn_nondet_train_kcp_backward(
            *args,
            ctx,
            ops["dout"],
            saved=saved,
            backend="fake",
            **knobs,
        )
    finally:
        kcp.set_backend("fake", previous)
    return seen


def _same_launch(got: tuple, want: tuple) -> bool:
    """Two ``(record, values, grid, is_test)`` resolutions name the same record, grid and flag and the same values
    (the same tensor objects, equal scalars)."""
    (r1, v1, g1, t1), (r2, v2, g2, t2) = got, want
    if (r1, tuple(g1), t1) != (r2, tuple(g2), t2) or v1.keys() != v2.keys():
        return False
    return all(
        v1[k] is v2[k] if isinstance(v2[k], torch.Tensor) else v1[k] == v2[k]
        for k in v2
    )


LAUNCH_CACHE_KNOBS = (
    {},
    {"segments": 2},
    {
        "segments": 2,
        "test_knobs": {"cols_rule": "by_chains", "fwdh_vh": 64, "dhu_vh": 128},
    },
    {
        "segments": 2,
        "test_knobs": {"schedule": "overlap", "fwdh_vh": 128, "fused_o": 0},
    },
    {"segments": 2, "save_mode": "full"},
)


def test_launch_cache_resolves_every_launch_as_resolve_launch(monkeypatch) -> None:
    """``_resolve_cached`` (the launch cache of the context tables) answers every launch of fake-backend forward +
    backward runs on a context with and without an SM count -- default options, two segments, forced walk forms with
    the by-chains merge columns, the overlap schedule with the split forward (``fused_o`` = 0: the tape walk and #5;
    the other light rows run the output walk), ``save_mode="full"`` (the tape-and-output walk); each persistent
    launch also with the kernels backend's ``arch``
    grid argument, each walk's arguments also as a transfer walk of every precision mode (the test-only records) --
    under both merge column defaults with the record, public values, grid and test flag of ``resolve_launch``, on the
    first launch of its key (the one call of ``resolve_launch``) and on every later one, and appends
    ``(record, grid)`` to the call's records each time."""
    kcp, ctx, ops = _ckpt_setup()
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(CKPT_CU, dtype=torch.int64), 1, "contiguous"
    )
    ctx_sm = gp.cake_gdn_nondet_train_kcp_build_context(
        part, None, 0, CPU, sm_count=SM_COUNT
    )
    launches: list[tuple[str, dict]] = []
    for c in (ctx, ctx_sm):
        for knobs in LAUNCH_CACHE_KNOBS:
            launches += _harvest_launches(kcp, c, ops, **knobs)
    extra = []
    for name, args in launches:
        grid_args = args.get("_grid_args") or {}
        if "sm_count" in grid_args:
            extra.append(
                (name, {**args, "_grid_args": {**grid_args, "arch": "sm_100a"}})
            )
        if name in kcp.FWDH_PROGRAMS + kcp.DHU_PROGRAMS and name not in (
            "kcp_fwdh_out",
            "kcp_fwdh_out_tape",
        ):
            extra += [
                ("kcp_transfer_walk", {**args, "_precision_mode": pm})
                for pm in (0, 1, 2)
            ]
    launches += extra
    names = {n for n, _ in launches}
    assert {"kcp_affine_merge", "kcp_transfer_walk", "fwd_o", "bwd_dqkg"} <= names
    assert ("kcp_fwdh_out" in names) == kcp.fused_o_shipped()
    assert ("kcp_fwdh_out_tape" in names) == kcp.fused_o_tape_shipped()
    assert names & set(kcp.FWDH_PROGRAMS) and names & set(kcp.DHU_PROGRAMS)
    real = kcp.resolve_launch
    assert any(real(n, dict(a))[3] for n, a in launches)  # a test-only record
    misses: list[tuple] = []

    def counting(name, args, *, cols_per_cta=8):
        misses.append(kcp._launch_key(name, args, cols_per_cta))
        return real(name, args, cols_per_cta=cols_per_cta)

    monkeypatch.setattr(kcp, "resolve_launch", counting)
    call = types.SimpleNamespace(tb=types.SimpleNamespace(launches={}), records=[])
    for _round in range(2):
        for cols in (8, 16):
            for name, args in launches:
                want = real(name, dict(args), cols_per_cta=cols)
                got = kcp._resolve_cached(
                    name, {**args, "_call": call}, cols_per_cta=cols
                )
                assert _same_launch(got, want), (name, cols)
                assert call.records[-1] == (want[0], want[2]), (name, cols)
    keys = {kcp._launch_key(n, a, c) for n, a in launches for c in (8, 16)}
    assert len(misses) == len(set(misses)) == len(keys) == len(call.tb.launches)
    assert len(call.records) == 4 * len(launches)
    # no tables, or tables without a cache: resolve_launch itself (nothing cached)
    for stub in (types.SimpleNamespace(records=[]), None):
        name, args = launches[0]
        before = len(misses)
        got = kcp._resolve_cached(name, {**args, "_call": stub})
        assert _same_launch(got, real(name, dict(args))) and len(misses) == before + 1


def test_launch_cache_keeps_every_calls_launch_records(monkeypatch) -> None:
    """One context on the fake backend: a run on cold launch caches, a run on warm ones and a run with the cache
    bypassed (``resolve_launch`` per launch) record the same launches -- records and grids in order, forward and
    backward, direct and checkpointed -- the tables cache one entry per launch key, and every run's results are within
    the budget of the FP64 reference and of each other."""
    kcp, ctx, ops = _ckpt_setup()
    ref = _ckpt_reference()
    tb = kcp._context_tables(ctx, CKPT_HEADS[1], fused_walks=1, P=2)
    assert tb.launches == {}
    arms, records = {}, {}
    for arm in ("cold", "warm", "bypassed"):
        res = {}
        for ckpt in (False, True):
            with monkeypatch.context() as m:
                if arm == "bypassed":
                    m.setattr(
                        kcp,
                        "_resolve_cached",
                        lambda name, args, *, cols_per_cta=8: kcp.resolve_launch(
                            name, args, cols_per_cta=cols_per_cta
                        ),
                    )
                res[ckpt] = _ckpt_autograd(kcp, ctx, ops, ckpt=ckpt, segments=2)
            records[(arm, ckpt)] = {
                k: list(kcp.LAST_CALL[k]["records"]) for k in ("fwd", "bwd")
            }
        arms[arm] = res
        if arm == "cold":
            n_keys = len(tb.launches)
            assert n_keys > 0
    assert len(tb.launches) == n_keys  # warm and bypassed runs add no key
    for ckpt in (False, True):
        assert (
            records[("cold", ckpt)]
            == records[("warm", ckpt)]
            == records[("bypassed", ckpt)]
        ), ckpt
        _arms_within_budget(
            {arm: arms[arm][ckpt] for arm in arms},
            ref,
            ("out",) + CKPT_GRADS,
            f"launch cache arms (checkpoint={ckpt})",
        )
    launched = {
        (r, tuple(g)) for rec in records.values() for k in rec for r, g in rec[k]
    }
    assert launched <= {(r, g) for r, g, _t, _c in tb.launches.values()}


def test_launch_scope_binds_the_stream_once_per_entry_body(monkeypatch) -> None:
    """``cake_backend.launch_scope`` with a counting stand-in for ``tvm_ffi`` and a set current stream (CPU): a launch
    outside every scope enters its own ``use_raw_stream(device, raw)`` context on the current stream's device and raw
    handle; inside a scope it enters none while the current stream is the bound one and its own after a stream switch;
    a nested scope on the same stream enters nothing, one on another stream binds that stream until it exits; another
    thread sees no scope; an exception restores the binding.  The kernels backend hands out its glue's scope; a glue
    without one and the fake backend none."""
    be = _package("cake_backend")
    kcp = _package("cake_kcp")
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

    monkeypatch.setattr(
        be,
        "tvm_ffi",
        types.SimpleNamespace(use_raw_stream=_Context),
    )
    monkeypatch.setattr(be, "_ffi_device", lambda index: ("cuda", index))
    monkeypatch.setattr(be, "_current_stream", lambda: state["stream"])
    name = "fwd_prep"
    values = {n: n for n, _axis in be._argument_plan(name) if n is not None}
    module = types.SimpleNamespace(
        **{
            be.record(name)["ffi_entry"]: lambda *a: state["launched"].append(
                state["bound"][-1] if state["bound"] else None
            )
        }
    )

    def launch() -> None:
        be.run_module(name, values, grid=(1, 1, 1), device_index=0, module=module)

    def enters_after(fn) -> int:
        n = state["enters"]
        fn()
        return state["enters"] - n

    assert enters_after(launch) == 1 and state["launched"][-1] == (0, 11)
    with be.launch_scope():
        assert state["enters"] == 2 and be._SCOPE.stream == (0, 11)
        assert enters_after(launch) == 0 and enters_after(launch) == 0
        assert state["launched"][-1] == (0, 11)
        with be.launch_scope():
            assert enters_after(launch) == 0
        assert be._SCOPE.stream == (0, 11) and state["bound"] == [(0, 11)]
        # a stream switch (to another device's stream) inside the scope
        state["stream"] = (1, 22)
        assert enters_after(launch) == 1 and state["launched"][-1] == (1, 22)
        with be.launch_scope():  # a scope on the other stream binds it
            assert be._SCOPE.stream == (1, 22)
            assert enters_after(launch) == 0 and state["launched"][-1] == (1, 22)
        assert be._SCOPE.stream == (0, 11)
        assert enters_after(launch) == 1
        state["stream"] = (0, 11)
        worker = threading.Thread(target=launch)
        assert enters_after(lambda: (worker.start(), worker.join())) == 1
        assert enters_after(launch) == 0
    assert getattr(be._SCOPE, "stream", None) is None and state["bound"] == []
    with pytest.raises(RuntimeError, match="inside the scope"), be.launch_scope():
        raise RuntimeError("inside the scope")
    assert getattr(be._SCOPE, "stream", None) is None and state["bound"] == []
    assert enters_after(launch) == 1
    # the backends' scopes
    assert kcp.FakeBackend().launch_scope(None) is None
    assert (
        kcp.KernelBackend(launcher=types.SimpleNamespace()).launch_scope(None) is None
    )
    with kcp.KernelBackend(launcher=be).launch_scope(None):
        assert be._SCOPE.stream == (0, 11)
        assert enters_after(launch) == 0
    assert getattr(be._SCOPE, "stream", None) is None


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


def test_cake_gdn_nondet_kcp_options_dicts_are_validated_once_per_content() -> None:
    """``cake_gdn_nondet_kcp`` with an ``options`` dict validates it once per distinct content (taken at each call: the same
    dict changed is another key; with the value types: ``save_transfer=1`` never stands in for ``True``), together
    with the literal keywords and knobs beside it; equal content in another dict shares that CakeGDNNondetKcpOptions; an
    unhashable value (a ``test_knobs`` dict) is validated per call; an invalid dict raises on every call; an CakeGDNNondetKcpOptions
    instance is not memoized."""
    kcp, ctx, ops = _ckpt_setup()
    kcp._KEYWORD_OPTIONS.clear()
    seen = []
    real = kcp._fwd_entry

    def spy(*a, opts, **kw):
        seen.append(opts)
        return real(*a, opts=opts, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(kcp, "_fwd_entry", spy)
    d = {"segments": 2}
    try:
        r1 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=d)
        r2 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=dict(d))
        d["save_transfer"] = False  # the same dict, changed (the default is True)
        r3 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=d)
        _ckpt_autograd(
            kcp, ctx, ops, ckpt=False, options={"segments": 2}, save_mode="full"
        )
        # keywords only: its own key
        _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
        for _ in range(2):
            knobs = {"segments": 2, "test_knobs": {"schedule": "overlap"}}
            _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=knobs)
        for _ in range(2):
            with pytest.raises(TypeError, match="save_transfer must be a bool"):
                _ckpt_autograd(kcp, ctx, ops, ckpt=False, options={"save_transfer": 1})
            with pytest.raises(TypeError, match="unexpected option"):
                _ckpt_autograd(kcp, ctx, ops, ckpt=False, options={"no_such_option": 1})
        inst = kcp.make_options(None, segments=2, backend="fake")
        n_keys = len(kcp._KEYWORD_OPTIONS)
        _ckpt_autograd(kcp, ctx, ops, ckpt=False, options=inst)
    finally:
        mp.undo()
    assert len(seen) == 8
    assert seen[0] is seen[1] and seen[0].save_transfer is True
    assert seen[2] is not seen[0] and seen[2].save_transfer is False
    assert seen[3] is not seen[0] and seen[3].save_mode == "full"
    assert seen[4] is not seen[0] and seen[4] == seen[0]
    assert seen[5] is not seen[6] and seen[5].knob("schedule") == "overlap"
    assert seen[0] == kcp.make_options({"segments": 2}, backend="fake")
    assert n_keys == 4 == len(kcp._KEYWORD_OPTIONS)
    assert seen[7] == inst
    assert set(r3) == set(r1)
    for k in r1:
        assert torch.equal(r1[k], r2[k]), k


def test_launch_key_reads_every_resolve_input() -> None:
    """``_launch_key`` changes with the program, every private scalar ``resolve_launch`` reads, the grid rule's
    arguments (values and order), ``ncols`` / ``cols_per_cta`` / the caller's grid x and the backend's merge columns,
    and with nothing else (another tensor of the same launch is the same key)."""
    kcp = _package("cake_kcp")
    base = {
        "_grid_args": {"n_walk": 2, "num_heads": 6},
        "_precision_mode": 0,
        "_transfer_walk": "plain",
        "_sm_count": 148,
        "_fwdh_vh": 0,
        "_dhu_vh": 0,
        "_vh64_fits": False,
        "_dhu_long_chain": False,
        "_arch": "sm_100a",
        "ncols": 128,
        "cols_per_cta": 8,
        "_grid": 12,
        "q": object(),
        "num_heads": 6,
    }
    k0 = kcp._launch_key("kcp_fwdh_tape", base, 8)
    assert k0 == kcp._launch_key("kcp_fwdh_tape", {**base, "q": object()}, 8)
    assert k0 == kcp._launch_key("kcp_fwdh_tape", dict(base), 8)
    names = ("_precision_mode", "_transfer_walk", *kcp._LAUNCH_KEY_ARGS)
    assert set(names) >= {
        "_sm_count",
        "_fwdh_vh",
        "_dhu_vh",
        "_vh64_fits",
        "_dhu_long_chain",
        "_arch",
        "ncols",
        "cols_per_cta",
        "_grid",
    }
    for name in names:
        assert kcp._launch_key("kcp_fwdh_tape", {**base, name: "other"}, 8) != k0, name
        missing = {k: v for k, v in base.items() if k != name}
        assert kcp._launch_key("kcp_fwdh_tape", missing, 8) != k0, name
    assert kcp._launch_key("kcp_fwdh_offset", base, 8) != k0
    assert kcp._launch_key("kcp_fwdh_tape", base, 16) != k0
    for grid_args in (
        {"n_walk": 3, "num_heads": 6},
        {"num_heads": 6, "n_walk": 2},
        None,
    ):
        changed = {**base, "_grid_args": grid_args}
        assert kcp._launch_key("kcp_fwdh_tape", changed, 8) != k0


# --------------------------------------------------------------------------- host launch path: launch memo, per-call workspace
# The plan's launch memo (``_Tables.memo``) and the per-call workspace (no ``workspace=``): host data and allocations
# only.  The "bypassed" arm replaces every memo method by the per-call derivation it replaced.


def _memo_bypass(kcp, m) -> None:
    tables = kcp._Tables
    m.setattr(tables, "indexed_spec", lambda self, index, tiles: None)
    m.setattr(
        tables,
        "walk_table",
        lambda self, walk, device: (
            len(walk),
            self.dev("walk", [int(x) for x in walk], device),
        ),
    )
    m.setattr(
        tables,
        "neg_index",
        lambda self, device: self.dev("neg", [-1] * int(self.NSEG), device),
    )
    m.setattr(
        tables,
        "chain_tables",
        lambda self, family, device: kcp._chain_device_tables(
            self, self.chains[family], device
        ),
    )
    m.setattr(tables, "split_lists", lambda self, coll: self.split_elems(coll))


def _mr_harvest(kcp, part, ops, ctxs, monkeypatch, *, bypass: bool):
    """Forward + saved-set backward of every rank of the multi-rank row (shipped options, per-call workspaces):
    per rank the launches ``(program, args without the call)`` in order, and the global-order results."""
    holder: dict = {}
    seen: dict[int, list[tuple[str, dict]]] = {}
    lock = threading.Lock()

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            comm = holder.get("comm")
            rank = getattr(comm.local, "rank", 0) if comm is not None else 0
            with lock:
                seen.setdefault(rank, []).append(
                    (name, {k: v for k, v in args.items() if k != "_call"})
                )
            super().run(name, args)

    def rank_fn(r, comm):
        holder["comm"] = comm
        loc = fake.slice_local(ops, part, r)
        args = [loc[n] for n in LEAVES]
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
            *args, ctxs[r], backend="fake"
        )
        grads = kcp.cake_gdn_nondet_train_kcp_backward(
            *args,
            ctxs[r],
            loc["dout"],
            saved=saved,
            backend="fake",
        )
        return out, grads

    previous = kcp.set_backend("fake", Recording())
    try:
        with monkeypatch.context() as m:
            if bypass:
                _memo_bypass(kcp, m)
            per_rank = _run_threaded(kcp, MR_W, rank_fn, m)
    finally:
        kcp.set_backend("fake", previous)
    res = {"out": fake.gather_global(part, [o for o, _ in per_rank])}
    for i, n in enumerate(EXPLICIT_GRADS):
        res[n] = fake.gather_global(part, [g[i] for _, g in per_rank])
    return seen, res


# int tensors a launch writes rather than reads: the #7 cut table ``c_start`` (written by
# ``kcp_dhu_cut_start``, read by its walk; a context-cached buffer whose unwalked entries
# are never written), a buffer and not a plan table
_LAUNCH_WRITTEN_INT_ARGS = frozenset({"c_start"})


def _same_args(a: dict, b: dict, *, same_plan: bool) -> list[str]:
    """The keys whose values differ: plan / context int tables (int32 / int64) must be the very same tensor within
    one plan and equal across plans; every other tensor (operands, pools, tapes, outputs, tickets, the launch-written
    int tables of ``_LAUNCH_WRITTEN_INT_ARGS``) must agree in dtype, shape and stride; every other value must be
    equal."""
    bad = []
    if a.keys() != b.keys():
        return ["<keys>"]
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
            if (
                x.dtype in (torch.int32, torch.int64)
                and k not in _LAUNCH_WRITTEN_INT_ARGS
            ):
                ok = (
                    (x is y)
                    if same_plan
                    else (x.dtype == y.dtype and torch.equal(x, y))
                )
            else:
                ok = (x.dtype, tuple(x.shape), x.stride()) == (
                    y.dtype,
                    tuple(y.shape),
                    y.stride(),
                )
        else:
            ok = type(x) is type(y) and x == y
        if not ok:
            bad.append(k)
    return bad


def test_launch_memo_hands_every_launch_the_per_call_derivation(monkeypatch) -> None:
    """The plan's launch memo (R4-4) on the multi-rank row (W = 8, exchanges and owner merges on every rank): a cold
    run (fresh contexts), a warm run (the same contexts) and a run with every memo method bypassed (fresh contexts,
    the per-call derivation) launch the same programs in the same order with the same arguments -- the plan's int
    tables are the very same tensors within one plan and equal across plans, the addressing scalars and grids equal,
    every buffer of the same dtype, shape and stride -- and give bitwise identical outputs and gradients.  Every
    memoised addressing is ``Indexed.addressing`` of its tables."""
    kcp, part, ops = _mr_setup()
    ctxs = [
        gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(MR_W)
    ]
    cold, res_cold = _mr_harvest(kcp, part, ops, ctxs, monkeypatch, bypass=False)
    warm, res_warm = _mr_harvest(kcp, part, ops, ctxs, monkeypatch, bypass=False)
    ctxs_b = [
        gp.cake_gdn_nondet_train_kcp_build_context(
            part, None, r, CPU, 4, sm_count=SM_COUNT
        )
        for r in range(MR_W)
    ]
    byp, res_byp = _mr_harvest(kcp, part, ops, ctxs_b, monkeypatch, bypass=True)
    names = {n for r in cold for n, _ in cold[r]}
    assert {
        "kcp_affine_merge",
        "kcp_fwdh_offset",
        "kcp_transfer_walk",
        "kcp_fwdh_tape",
    } <= names, names
    assert names & set(kcp.DHU_PROGRAMS)
    for r in range(MR_W):
        assert (
            [n for n, _ in cold[r]] == [n for n, _ in warm[r]] == [n for n, _ in byp[r]]
        ), r
        for i, ((n, a), (_, b), (_, c)) in enumerate(
            zip(cold[r], warm[r], byp[r], strict=True)
        ):
            assert not _same_args(a, b, same_plan=True), (
                r,
                i,
                n,
                _same_args(a, b, same_plan=True),
            )
            assert not _same_args(b, c, same_plan=False), (
                r,
                i,
                n,
                _same_args(b, c, same_plan=False),
            )
    for k in res_cold:
        assert torch.equal(res_cold[k], res_warm[k]) and torch.equal(
            res_cold[k], res_byp[k]
        ), k
    specs = 0
    for ctx in ctxs:
        for key, tb in ctx._caches.items():
            if not (isinstance(key, tuple) and key and key[0] == "kcp_tables"):
                continue
            for mk, v in tb.memo.items():
                if mk[0] == "ix" and v.addr is not None:
                    specs += 1
                    assert (
                        kcp.Indexed(torch.empty(0), v.index, v.tiles).addressing(tb.H)
                        == v.addr
                    )
    for ctx in ctxs_b:  # the bypassed arm memoised no view of a pool
        for key, tb in ctx._caches.items():
            if isinstance(key, tuple) and key and key[0] == "kcp_tables":
                assert not any(
                    mk[0] in ("ix", "ixn", "walk", "chain", "neg", "split")
                    for mk in tb.memo
                ), key
    assert specs > 0


def test_per_call_workspace_shares_its_context_ticket_set() -> None:
    """A call without ``workspace=`` takes its context's ticket set of the current stream (R4-4): the persistent
    launches of every such call on one context see the same three zero slots; another context and a caller-held
    workspace use their own; the slots are 128-byte aligned u32 slots and stay zero at every launch.  The forwards are
    the split form (``fused_o`` = 0), so #5 takes its slot."""
    kcp, ctx, ops = _ckpt_setup()
    ctx2 = gp.cake_gdn_nondet_train_kcp_build_context(
        gp.cake_gdn_nondet_train_kcp_partition(
            torch.tensor(CKPT_CU, dtype=torch.int64), 1, "contiguous"
        ),
        None,
        0,
        CPU,
    )
    seen: list[tuple[str, str, int, bool]] = []
    phase = ["?"]

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            if name in kcp.TICKET_SITES:
                t = args["ticket"]
                seen.append(
                    (
                        phase[0],
                        name,
                        t.data_ptr(),
                        t.dtype == torch.uint32
                        and t.numel() == kcp.TICKET_SLOT_WORDS
                        and t.data_ptr() % kcp.TICKET_SLOT_BYTES == 0
                        and int(t.view(torch.int32).abs().sum()) == 0,
                    )
                )
            super().run(name, args)

    previous = kcp.set_backend("fake", Recording())
    ws = kcp.CakeGDNNondetKcpWorkspace()
    try:
        for tag, c, w in (
            ("a", ctx, None),
            ("a", ctx, None),
            ("b", ctx2, None),
            ("w", ctx, ws),
        ):
            phase[0] = tag
            _ckpt_autograd(
                kcp,
                c,
                ops,
                ckpt=False,
                segments=2,
                workspace=w,
                test_knobs={"fused_o": 0},
            )
    finally:
        kcp.set_backend("fake", previous)
    assert all(ok for *_x, ok in seen), seen
    ptr: dict[tuple[str, str], set[int]] = {}
    for tag, site, p, _ok in seen:
        ptr.setdefault((tag, site), set()).add(p)
    assert {s for _t, s in ptr} == set(kcp.TICKET_SITES)
    assert all(len(v) == 1 for v in ptr.values()), (
        ptr
    )  # one slot per site: the two per-call runs share it
    for site in kcp.TICKET_SITES:
        a, b, w = (next(iter(ptr[(t, site)])) for t in ("a", "b", "w"))
        assert len({a, b, w}) == 3, (site, a, b, w)
        assert w == ws.ticket(site).data_ptr()
    shared = [
        v
        for k, v in ctx._caches.items()
        if isinstance(k, tuple) and k[0] == "kcp_call_tickets"
    ]
    assert len(shared) == 1 and int(shared[0][0].view(torch.int32).abs().sum()) == 0


@pytest.mark.parametrize("segments", [1, 2])
def test_per_call_workspace_sizes_no_dead_or_empty_pool(segments) -> None:
    """The per-call workspace (R4-4) sizes no pool its call never touches: a saved-context backward none of the
    forward exchange's pools (``SAVED_BWD_DEAD_POOLS``), and no call an empty pool (the staging and carry planes at
    ``P = 1``); a caller-held workspace still sizes them (its documented layout).  The end-of-call records name the
    same pools either way, and the results are bitwise equal."""
    kcp = _package("cake_kcp")
    cu = [0, 100, 740, 817, 1817, 1946]
    H_qk, HV = 2, 6
    part = gp.cake_gdn_nondet_train_kcp_partition(
        torch.tensor(cu, dtype=torch.int64), 2, "contiguous"
    )
    ops = _inputs(cu, H_qk, HV)
    assert set(kcp.SAVED_BWD_DEAD_POOLS) == {"c1_send", "c1_recv", "c2_send"}

    def run(workspaces):
        held: dict[int, list[tuple[bool, set[str], dict[str, int]]]] = {}
        lock = threading.Lock()
        holder: dict = {}

        class Recording(kcp.FakeBackend):
            def run(self, name, args):
                call = args["_call"]
                comm = holder.get("comm")
                rank = getattr(comm.local, "rank", 0)
                with lock:
                    calls = held.setdefault(rank, [])
                    first = (
                        not calls or calls[-1][0] is not call
                    )  # what ensure() sized (before any on-demand pool)
                    calls.append(
                        (
                            call,
                            first,
                            set(call.ws._pools),
                            {n: t.numel() for n, t in call.ws._pools.items()},
                        )
                    )
                super().run(name, args)

        ctxs = [
            gp.cake_gdn_nondet_train_kcp_build_context(
                part, None, r, CPU, 4, sm_count=SM_COUNT
            )
            for r in range(2)
        ]
        records: dict[int, dict] = {}

        def rank_fn(r, comm):
            holder["comm"] = comm
            loc = fake.slice_local(ops, part, r)
            args = [loc[n] for n in LEAVES]
            kw = dict(
                backend="fake",
                segments=segments,
                workspace=workspaces[r],
            )
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctxs[r], **kw)
            fwd_pools = dict(kcp.LAST_CALL["fwd"]["pool_bytes"])
            grads = kcp.cake_gdn_nondet_train_kcp_backward(
                *args, ctxs[r], loc["dout"], saved=saved, **kw
            )
            records[r] = {
                "fwd": fwd_pools,
                "bwd": dict(kcp.LAST_CALL["bwd"]["pool_bytes"]),
            }
            return out, grads

        previous = kcp.set_backend("fake", Recording())
        mp = pytest.MonkeyPatch()
        try:
            per_rank = _run_threaded(kcp, 2, rank_fn, mp)
        finally:
            mp.undo()
            kcp.set_backend("fake", previous)
        return held, records, per_rank

    held_pc, rec_pc, res_pc = run([None, None])
    held_ws, rec_ws, res_ws = run(
        [kcp.CakeGDNNondetKcpWorkspace(), kcp.CakeGDNNondetKcpWorkspace()]
    )
    dead = set(kcp.SAVED_BWD_DEAD_POOLS)
    for r in range(2):
        for call, first, names, sizes in held_pc[r]:
            if first:
                assert not any(n == 0 for n in sizes.values()), sizes
            if call.backward:
                assert not (dead & names), sorted(dead & names)
        bwd_ws = [
            names for call, first, names, _s in held_ws[r] if call.backward and first
        ]
        assert bwd_ws and {"c1_send", "c2_send"} <= bwd_ws[0], bwd_ws[0]
        if segments == 1:
            fwd_ws = [
                sizes
                for call, first, _n, sizes in held_ws[r]
                if first and not call.backward
            ]
            assert (
                fwd_ws and fwd_ws[0].get("stage_b") == 0 and fwd_ws[0].get("carry") == 0
            ), fwd_ws[0]
        # LAST_CALL is one module dict both rank threads write: compare the pool names of the snapshots (every eager
        # call ends with its exchange and staging pools released)
        assert {k: set(v) for k, v in rec_pc[r].items()} == {
            k: set(v) for k, v in rec_ws[r].items()
        }, (r, rec_pc[r], rec_ws[r])
        (o1, g1), (o2, g2) = res_pc[r], res_ws[r]
        assert torch.equal(o1, o2) and all(
            torch.equal(a, b) for a, b in zip(g1, g2, strict=True)
        ), r


def test_cake_gdn_nondet_kcp_keyword_options_are_validated_once_per_keyword_set() -> (
    None
):
    """``cake_gdn_nondet_kcp`` without an ``options`` object validates its keyword options once per distinct keyword set
    (with the value types: ``save_transfer=1`` never stands in for ``True``) and reuses that CakeGDNNondetKcpOptions; an unhashable
    value (a ``test_knobs`` dict) is validated per call; an invalid set raises on every call."""
    kcp, ctx, ops = _ckpt_setup()
    kcp._KEYWORD_OPTIONS.clear()
    seen = []
    real = kcp._fwd_entry

    def spy(*a, opts, **kw):
        seen.append(opts)
        return real(*a, opts=opts, **kw)

    mp = pytest.MonkeyPatch()
    mp.setattr(kcp, "_fwd_entry", spy)
    try:
        r1 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
        r2 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2)
        r3 = _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2, save_transfer=True)
        _ckpt_autograd(
            kcp, ctx, ops, ckpt=False, segments=2, test_knobs={"schedule": "overlap"}
        )
        _ckpt_autograd(
            kcp, ctx, ops, ckpt=False, segments=2, test_knobs={"schedule": "overlap"}
        )
        for _ in range(2):
            with pytest.raises(TypeError, match="save_transfer must be a bool"):
                _ckpt_autograd(kcp, ctx, ops, ckpt=False, segments=2, save_transfer=1)
        with pytest.raises(TypeError, match="unexpected option"):
            _ckpt_autograd(kcp, ctx, ops, ckpt=False, no_such_option=1)
    finally:
        mp.undo()
    assert len(seen) == 5
    assert (
        seen[0] is seen[1] and seen[2] is not seen[0] and seen[2].save_transfer is True
    )
    assert seen[3] is not seen[4] and seen[3].knob("schedule") == "overlap"
    assert seen[0] == kcp.make_options(
        None,
        segments=2,
        backend="fake",
        **{"save_mode": "light", "boundary_mode": "saved", "graph": "off"},
    )
    assert len(kcp._KEYWORD_OPTIONS) == 2
    for k in r1:
        assert torch.equal(r1[k], r2[k]) and torch.equal(r1[k], r3[k]), k


# --------------------------------------------------------------------------- numerics harness hooks (fake backend)

NONDET_REPEATS = 8  # the worst case over this many runs is gated per tensor


def _pack_reference(cu: list[int], heads=CKPT_HEADS) -> dict[str, torch.Tensor]:
    key = ("pack", tuple(cu), tuple(heads))
    if key not in _REFERENCES:
        ops = _inputs(cu, *heads)
        out, grads = _fp64(_ref_layer(cu), [ops[n] for n in LEAVES], ops["dout"])
        _REFERENCES[key] = {"out": out, **dict(zip(CKPT_GRADS, grads, strict=True))}
    return _REFERENCES[key]


def _poison(ws) -> list[str]:
    """NaN into every floating pool and tape of a caller-held workspace except its constants (the identity tile
    ``eye``, shared by the context, and the zeroed ticket words of the persistent programs)."""
    poisoned = []
    for kind, table in (("pool", ws._pools), ("tape", ws._tapes)):
        for name, t in table.items():
            if name == "eye" or not t.is_floating_point():
                continue
            t.fill_(float("nan"))
            poisoned.append(f"{kind}:{name}")
    return poisoned


def test_reused_workspace_poisoned_between_changed_shapes_stays_within_budget() -> None:
    """One caller-held workspace across two token layouts of different sizes (1946 tokens in five documents, 1500 in
    two), filled with NaN between every forward + backward step: every step's out and gradients stay finite and
    within the budget of its FP64 reference (no stale scratch is read)."""
    kcp = _package("cake_kcp")
    layouts = (CKPT_CU, [0, 300, 1500])
    ws = kcp.CakeGDNNondetKcpWorkspace()
    poisoned_any = False
    for _round in range(2):
        for cu in layouts:
            part = gp.cake_gdn_nondet_train_kcp_partition(
                torch.tensor(cu, dtype=torch.int64), 1, "contiguous"
            )
            ctx = gp.cake_gdn_nondet_train_kcp_build_context(
                part, None, 0, CPU, sm_count=SM_COUNT
            )
            ops = _inputs(cu, *CKPT_HEADS)
            args = [ops[n] for n in LEAVES]
            common = dict(backend="fake", segments=2, workspace=ws)
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, **common)
            grads = kcp.cake_gdn_nondet_train_kcp_backward(
                *args, ctx, ops["dout"], saved=saved, **common
            )
            ref = _pack_reference(cu)
            what = f"T={cu[-1]} round {_round}"
            _within_budget(out, ref["out"], f"{what}: out")
            for n, t in _grads_dict(grads).items():
                _within_budget(t, ref[n], f"{what}: {n}")
            poisoned_any = bool(_poison(ws)) or poisoned_any
    assert poisoned_any, "the workspace held no pool or tape to poison"


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
def test_worst_case_over_repeated_runs_is_within_budget(ckpt) -> None:
    """NONDET_REPEATS runs of the same autograd step: the worst rel-RMS per tensor against the FP64 reference stays
    within the budget (every run is gated too)."""
    kcp, ctx, ops = _ckpt_setup()
    ref = _ckpt_reference()
    worst = {n: 0.0 for n in ("out",) + CKPT_GRADS}
    for i in range(NONDET_REPEATS):
        res = _ckpt_autograd(kcp, ctx, ops, ckpt=ckpt, segments=2)
        for n in worst:
            m = _within_budget(res[n], ref[n], f"run {i}: {n}")
            worst[n] = max(worst[n], m["rel_rms"])
    assert all(w <= FAKE_TOL_REL_RMS for w in worst.values()), worst
    _evidence(
        "nondet_repeats",
        {"ckpt": ckpt, "runs": NONDET_REPEATS, "worst_rel_rms": worst, "pass": True},
    )


@pytest.mark.parametrize("ckpt", [False, True], ids=["direct", "checkpoint"])
def test_gradients_accumulate_into_a_nonzero_grad(ckpt) -> None:
    """Leaves whose ``.grad`` already holds a nonzero tensor ``g0`` (the scale of the gradient): after the backward
    every ``.grad`` is within the budget of ``g0 + gradient`` in FP64."""
    kcp, ctx, ops = _ckpt_setup()
    ref = _ckpt_reference()
    gen = torch.Generator(device="cpu").manual_seed(8063)
    leaves = _ckpt_leaves(ops)
    pre = {}
    for n, lf in zip(CKPT_GRADS, leaves, strict=True):
        rms = float(ref[n].pow(2).mean().sqrt())
        g0 = (torch.randn(lf.shape, generator=gen, dtype=torch.float64) * rms).to(
            lf.dtype
        )
        lf.grad = g0.clone()
        pre[n] = g0

    def fn(q, k, v, g, beta):
        return kcp.cake_gdn_nondet_kcp(
            q, k, v, g, beta, ctx, backend="fake", segments=2
        )

    if ckpt:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, use_reentrant=False)
    else:
        out = fn(*leaves)
    out.backward(ops["dout"])
    for n, lf in zip(CKPT_GRADS, leaves, strict=True):
        assert lf.grad.dtype == pre[n].dtype, n
        _within_budget(lf.grad, pre[n].double() + ref[n], f"{n}: g0 + gradient")


# ----------------------------------------------------------------------------------------------------------------
# Packed Q/K/V (issue #7817): q / k / v as head views of the convolution's packed row; the backward writes dq / dk / dv
# into one packed row and the split hands it on without a concatenation.


def _packed_autograd(
    kcp, ctx, ops, *, batched: bool = False, ckpt: bool = False, **opts
) -> tuple[dict[str, torch.Tensor], list[torch.Tensor]]:
    """``out`` and the gradients of one layer whose q / k / v come from :func:`cake_gdn_nondet_train_kcp_packed_qkv_views` of a packed leaf;
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
        q, k, v = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(qkv, H_qk=H_qk, HV=HV)
        gg, bb = (g.unsqueeze(0), beta.unsqueeze(0)) if batched else (g, beta)
        return kcp.cake_gdn_nondet_kcp(q, k, v, gg, bb, ctx, **o)

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
def test_packed_qkv_gradients_need_no_concatenation(
    batched, ckpt, group_reduce, layout
) -> None:
    """q / k / v as views of one packed leaf: out and the five gradients are within the budgets of separate contiguous
    leaves (and of FP64); with the views passed through (``layout="tma"``) the backward writes one packed dQKV row
    (``grad_layout="packed"``) and the split returns it as the leaf's gradient (a view, no concatenation); with
    ``layout="copy"`` the operands are copied, the gradients are separate and the split concatenates them."""
    kcp, ctx, ops = _ckpt_setup()
    base = dict(group_reduce=group_reduce, layout=layout)
    direct = _ckpt_autograd(kcp, ctx, ops, ckpt=False, **base)
    before = dict(kcp.PACKED_SPLIT_STATS)
    got, seen = _packed_autograd(kcp, ctx, ops, batched=batched, ckpt=ckpt, **base)
    what = f"batched={batched} ckpt={ckpt} {base}"
    _assert_close_arms(got, direct, what)
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
    q, k, v = kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(a, H_qk=H_qk, HV=HV)
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


def test_cake_gdn_nondet_train_kcp_packed_qkv_views_rejects_a_bad_row() -> None:
    kcp = _package("cake_kcp")
    x = torch.zeros(5, 10 * 128, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="expected"):
        kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
            x, H_qk=2, HV=4
        )  # 8 * 128 channels expected
    with pytest.raises(ValueError, match="dense"):
        kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
            x.t().contiguous().t(), H_qk=2, HV=6
        )
    with pytest.raises(ValueError):
        kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(
            torch.zeros(5, 9 * 128, dtype=torch.bfloat16), H_qk=2, HV=5
        )


# --------------------------------------------------------------------------- the output walk kcp_fwdh_out (fused_o)
# A forward whose h / v_new tapes have no reader runs kcp_fwdh_out (#2 with #5 folded into each chunk step, no tape
# written) on every tape-walk launch list.  On the fake backend these are host-plumbing checks (launch lists, output
# coverage, arena); the kernels' bitwise check is the GPU suites' (fused_o 1 vs 0 in one build).
FUSED_TO_SPLIT = {
    "kcp_fwdh_out": "kcp_fwdh_tape",
    "kcp_fwdh_out_vh64": "kcp_fwdh_tape_vh64",
    "kcp_fwdh_out_tape": "kcp_fwdh_tape",
    "kcp_fwdh_out_tape_vh64": "kcp_fwdh_tape_vh64",
}


def _nan_output(kcp, monkeypatch) -> None:
    """``CakeGDNNondetKcpCall.output`` pre-fills a new output with NaN, so a row that neither an output walk nor #5 writes shows."""
    original = kcp.CakeGDNNondetKcpCall.output

    def output(self):
        fresh = self.out is None
        t = original(self)
        if fresh:
            t.fill_(float("nan"))
        return t

    monkeypatch.setattr(kcp.CakeGDNNondetKcpCall, "output", output)


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
    split hand-off recompute with ``True``), the recompute handle, ``save_mode="full"`` and the replay boundary: out
    and every gradient equal bitwise with the shipped options and with ``fused_o=0``; the first forward of a light
    region is fused, a hand-off or a full / replay-boundary forward never is, and no kept entry is left."""
    kcp, ctx, ops = _ckpt_setup()
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
    ]
    for name, form in forms:
        got = {}
        for fused in (1, 0):
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
                    kcp, ctx, ops, **form, **base, test_knobs={"fused_o": fused}
                )
            finally:
                kcp._BACKENDS["fake"] = previous
            got[fused] = (res, scale_grad, seen)
        (r1, s1, seen1), (r0, s0, seen0) = got[1], got[0]
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
            assert seen0.count("fwd_o") - seen1.count("fwd_o") >= 1, (
                name,
                seen0,
                seen1,
            )
            assert seen1.count("fwd_o") + seen1.count("kcp_fwdh_out") >= 1, name
        else:
            assert seen1 == seen0, name  # no forward of these forms fuses
        assert kcp.kept_registry_size() == 0, name


def test_fused_forward_releases_a_caller_arenas_tapes_and_kept_backward_leaves_the_fused_set(
    monkeypatch,
) -> None:
    """A caller-held workspace: a split forward (``fused_o=0``) leaves ``aqk`` / ``qscale`` / ``h_tape`` / ``v_new``
    bound; the next fused forward drops ``h_tape`` / ``v_new`` before #1 and leaves ``aqk`` / ``qscale``
    (:data:`FWD_TAPES_ARENA_AFTER_FUSED`); a kept-tape checkpoint backward re-adopts that fused set only, and the
    split one re-adopts the four.  Every arm's out and gradients are the same bits."""
    kcp, ctx, ops = _ckpt_setup()
    assert kcp.FWD_TAPES_ARENA_AFTER_FUSED == ("aqk", "qscale")
    ws = kcp.CakeGDNNondetKcpWorkspace()
    leaves = [ops[n] for n in LEAVES]
    o0, _ = kcp.cake_gdn_nondet_train_kcp_forward(
        *leaves,
        ctx,
        workspace=ws,
        backend="fake",
        test_knobs={"fused_o": 0},
    )
    assert set(ws._tapes) & set(kcp.FWD_TAPES) == set(kcp.FWD_TAPES_ARENA_AFTER_FWD)
    o1, _ = kcp.cake_gdn_nondet_train_kcp_forward(
        *leaves, ctx, workspace=ws, backend="fake"
    )
    assert kcp.LAST_CALL["fwd"]["fwd_o"] == "fused"
    assert set(ws._tapes) & set(kcp.FWD_TAPES) == set(kcp.FWD_TAPES_ARENA_AFTER_FUSED)
    assert torch.equal(o0, o1)
    arms = {}
    for fused in (1, 0):
        wsk = kcp.CakeGDNNondetKcpWorkspace()
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


# --------------------------------------------------------------------------- the tape-and-output walk kcp_fwdh_out_tape
# A save_mode="full" forward that keeps its saved set and launches #5 runs kcp_fwdh_out_tape (the output walk that also
# writes the saved h / v_new tapes) on every tape-walk launch list (test knob fused_o_tape, 1 = shipped); every light
# forward keeps its form.  On the fake backend these are host-plumbing checks (launch lists, output and tape coverage,
# saved set, registry fallback, workspace); the kernels' bitwise check is the GPU suites' (fused_o_tape 1 vs 0 in one
# build).
TAPE_WALK_TAPES = ("h_tape", "v_new")


def _bytes(t: torch.Tensor) -> torch.Tensor:
    """The bytes of ``t`` (a NaN compares equal to the same NaN bytes)."""
    return t.detach().contiguous().view(-1).view(torch.uint8)


def _nan_fresh_tapes(kcp, monkeypatch) -> None:
    """``CakeGDNNondetKcpCall.fresh_tapes`` pre-fills the fresh ``h_tape`` / ``v_new`` (the saved tapes of ``save_mode="full"``)
    with NaN, so a row no walk writes shows: the split form writes every real row, so a byte compare against it
    catches a row the fused form leaves out (the pad rows are NaN in both)."""
    original = kcp.CakeGDNNondetKcpCall.fresh_tapes

    def fresh_tapes(self, names):
        original(self, names)
        for n in TAPE_WALK_TAPES:
            if n in names:
                self.tapes[n].fill_(float("nan"))

    monkeypatch.setattr(kcp.CakeGDNNondetKcpCall, "fresh_tapes", fresh_tapes)


def _forward_calls(kcp):
    """A recording fake backend: ``[call, programs]`` per forward call in first-launch order (the call's scratch holds
    its ``fwd_o`` decision once it has run)."""
    forwards: list[list] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, args):
            call = args["_call"]
            if not call.backward:
                entry = next((e for e in forwards if e[0] is call), None)
                if entry is None:
                    entry = [call, []]
                    forwards.append(entry)
                entry[1].append(name)
            super().run(name, args)

    return forwards, Recording()


def _fused_tape_forwards(forwards, *, enabled: bool) -> int:
    """Checks every recorded forward's launches against its ``fwd_o`` decision and returns how many ran the
    tape-and-output walk: a ``fused_tape`` forward is a ``save_mode="full"`` one with the walk ``enabled`` (test knob 1
    and shipped records) and runs ``kcp_fwdh_out_tape``, neither the tape walk nor the output walk, and no #5; with the
    walk enabled no full forward writes the output with #5 (every forward of these forms keeps its saved set); a light
    forward never runs it."""
    n = 0
    for call, progs in forwards:
        decision = call.scratch["fwd_o"]
        full = call.opts.save_mode == "full"
        assert ("kcp_fwdh_out_tape" in progs) == (decision == "fused_tape"), (
            decision,
            progs,
        )
        assert progs.count("fwd_o") == int(decision == "launched"), (decision, progs)
        if decision == "fused_tape":
            assert enabled and full, (decision, progs)
            assert not {"kcp_fwdh_tape", "kcp_fwdh_out"} & set(progs), progs
            n += 1
        elif decision == "launched":
            assert not (enabled and full), (decision, progs)
    return n


@pytest.mark.parametrize(
    "knobs",
    [
        {},
        {"use_qk_l2norm_in_kernel": True},
        {"test_knobs": {"schedule": "overlap"}},
        {"test_knobs": {"fused_walks": 0}},
        {"test_knobs": {"fwdh_vh": 64}},
        {"test_knobs": {"fwdh_vh": 128}},
        {"save_transfer": False},
    ],
    ids=(
        "shipped",
        "l2n",
        "overlap",
        "fused_walks0",
        "vh64",
        "vh128",
        "no_save_transfer",
    ),
)
def test_fused_tape_walk_reproduces_the_split_full_forward(monkeypatch, knobs) -> None:
    """The W = 8 row with ``save_mode="full"``, with the shipped tape-and-output walk and with test knob
    ``fused_o_tape=0``: out (pre-filled with NaN when it is allocated), the saved ``h`` / ``v_new`` tapes (pre-filled
    with NaN) and the five gradients equal bitwise; each rank's forward runs ``kcp_fwdh_out_tape`` on exactly the
    launch lists (records modulo the program, grids) the split forward runs ``kcp_fwdh_tape`` on, and no #5; its
    ``fwd_o`` evidence is ``"fused_tape"``.  The backward calls are the same launches either way."""
    kcp, part, ops = _mr_setup()
    assert kcp.fused_o_tape_shipped()
    runs = {}
    for fused in (1, 0):
        k = {
            **knobs,
            "save_mode": "full",
            "test_knobs": {**knobs.get("test_knobs", {}), "fused_o_tape": fused},
        }
        _nan_output(kcp, monkeypatch)
        _nan_fresh_tapes(kcp, monkeypatch)
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
        assert f1.T_r > 0, r  # every rank of the row holds tokens
        assert f1.scratch["fwd_o"] == "fused_tape", (r, f1.scratch["fwd_o"])
        assert f0.scratch["fwd_o"] == "launched", (r, f0.scratch["fwd_o"])
        assert not {"fwd_o", "kcp_fwdh_tape", "kcp_fwdh_out"} & set(f1.sequence), r
        assert "kcp_fwdh_out_tape" in f1.sequence and "fwd_o" in f0.sequence, r
        assert "kcp_fwdh_out_tape" not in f0.sequence, r
        assert _split_view(f1) == _split_view(f0), r
        for n in TAPE_WALK_TAPES:
            assert torch.equal(_bytes(f1.tapes[n]), _bytes(f0.tapes[n])), (r, n)
        b1 = [(c.sequence, c.records) for c in calls1[r] if c.backward]
        b0 = [(c.sequence, c.records) for c in calls0[r] if c.backward]
        assert b1 == b0, r


@pytest.mark.parametrize("segments", (1, 2))
def test_fused_tape_walk_binds_the_saved_tapes(monkeypatch, segments) -> None:
    """One rank, explicit entries, ``save_mode="full"``: the tape-and-output walk receives the saved set's own
    ``h_tape`` / ``v_new`` tensors and the returned output (no copy), the saved set and out equal the split form's
    bytes (tapes pre-filled with NaN), and the backward of either saved set gives the same gradients; a forward
    without a saved set (``save_for_backward=False``) keeps the split form, and so does a light forward."""
    kcp, ctx, ops = _ckpt_setup()
    args = [ops[n] for n in LEAVES]
    o = {
        "backend": "fake",
        "save_mode": "full",
        "segments": segments,
    }
    bound: list[tuple] = []

    class Recording(kcp.FakeBackend):
        def run(self, name, a):
            if name == "kcp_fwdh_out_tape":
                bound.append((a["h_tape"], a["v_new"], a["o"]))
            super().run(name, a)

    arms = {}
    _nan_fresh_tapes(kcp, monkeypatch)
    previous = kcp.set_backend("fake", Recording())
    try:
        for fused in (1, 0):
            bound.clear()
            out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
                *args, ctx, test_knobs={"fused_o_tape": fused}, **o
            )
            decision = kcp.LAST_CALL["fwd"]["fwd_o"]
            assert decision == ("fused_tape" if fused else "launched"), decision
            assert bool(bound) == bool(fused)
            for h, v, oo in bound:
                assert h.data_ptr() == saved.tape("h_tape").data_ptr()
                assert v.data_ptr() == saved.tape("v_new").data_ptr()
                assert oo.data_ptr() == out.data_ptr()
            grads = kcp.cake_gdn_nondet_train_kcp_backward(
                *args,
                ctx,
                ops["dout"],
                saved=saved,
                test_knobs={"fused_o_tape": fused},
                **o,
            )
            arms[fused] = (out, saved, grads)
        bound.clear()
        plain, none = kcp.cake_gdn_nondet_train_kcp_forward(
            *args, ctx, save_for_backward=False, **o
        )
        assert none is None and not bound
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == "launched"
        light, _ = kcp.cake_gdn_nondet_train_kcp_forward(
            *args, ctx, **{**o, "save_mode": "light"}
        )
        assert not bound
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == (
            "fused" if kcp.fused_o_shipped() else "launched"
        )
    finally:
        kcp.set_backend("fake", previous)
    (o1, s1, g1), (o0, s0, g0) = arms[1], arms[0]
    assert torch.equal(o1, o0) and torch.equal(plain, o0) and torch.equal(light, o0)
    for n in TAPE_WALK_TAPES:
        assert torch.equal(_bytes(s1.tape(n)), _bytes(s0.tape(n))), n
    for a, b in zip(g1, g0, strict=True):
        assert torch.equal(a, b)


@pytest.mark.parametrize("segments", (1, 2))
def test_fused_tape_walk_under_checkpoints_and_handles(segments) -> None:
    """One rank (no exchange), P = 1 / 2: ``save_mode="full"`` directly and under non-reentrant checkpoints (the
    region the node closes, a later saver, ``recompute_output="always"``, the in-kernel L2 norm, a recompute handle),
    and the light hand-off forms whose recompute launches #5 (``reuse_recompute_tapes=True`` with a later saver or
    with ``recompute_output="always"``): out and every gradient equal bitwise with the shipped options and with
    ``fused_o_tape=0``; every full forward that writes the output runs the tape-and-output walk and no #5 (the
    expected count per form), a recompute that leaves #5 out keeps the tape walk, no light forward (the hand-off one
    included) runs it, the split arm never does, and no kept entry is left."""
    kcp, ctx, ops = _ckpt_setup()
    assert kcp.fused_o_tape_shipped()
    base = {"segments": segments}
    full = {"save_mode": "full"}
    always = {"recompute_output": "always"}
    hand = {"reuse_recompute_tapes": True}
    # the recompute of a region the node closes leaves #5 out on a verified torch (its tape walk stays)
    closing = 1 if _verified_torch(kcp) else 2
    # id, entry ("region": _ckpt_autograd_region, "handle": _handle_autograd), options, fused_tape forwards
    forms = [
        ("full_direct", "region", dict(ckpt=False, later_saver=False, **full), 1),
        ("full_closing", "region", dict(ckpt=True, later_saver=False, **full), closing),
        ("full_later", "region", dict(ckpt=True, later_saver=True, **full), 2),
        (
            "full_always",
            "region",
            dict(ckpt=True, later_saver=False, **full, **always),
            2,
        ),
        (
            "full_l2n",
            "region",
            dict(ckpt=True, later_saver=True, use_qk_l2norm_in_kernel=True, **full),
            2,
        ),
        ("full_handle", "handle", dict(**full, **always), 2),
        # light: the first forward is the output walk's, the hand-off recompute keeps the tape walk and #5
        ("hand_off_later", "region", dict(ckpt=True, later_saver=True, **hand), 0),
        (
            "hand_off_always",
            "region",
            dict(ckpt=True, later_saver=False, **hand, **always),
            0,
        ),
    ]
    for name, entry, form, want in forms:
        got = {}
        for fused in (1, 0):
            forwards, rec = _forward_calls(kcp)
            previous = kcp.set_backend("fake", rec)
            try:
                knobs = {"test_knobs": {"fused_o_tape": fused}}
                if entry == "handle":
                    res, _modes = _handle_autograd(
                        kcp,
                        ctx,
                        ops,
                        handle=kcp.CakeGDNNondetKcpRecomputeHandle(),
                        **form,
                        **base,
                        **knobs,
                    )
                    scale_grad = None
                else:
                    res, scale_grad = _ckpt_autograd_region(
                        kcp, ctx, ops, **form, **base, **knobs
                    )
            finally:
                kcp.set_backend("fake", previous)
            n = _fused_tape_forwards(forwards, enabled=bool(fused))
            got[fused] = (res, scale_grad, n)
        (r1, s1, n1), (r0, s0, n0) = got[1], got[0]
        for n in ("out",) + CKPT_GRADS:
            assert torch.equal(r1[n], r0[n]), (name, n)
        assert (s1 is None and s0 is None) or torch.equal(s1, s0), name
        assert n1 == want and n0 == 0, (name, n1, n0, want)
        assert kcp.kept_registry_size() == 0, name


def test_fused_tape_walk_needs_both_records_for_the_architecture(monkeypatch) -> None:
    """``fused_o_tape_shipped`` holds only when the registry ships the tape-and-output walk in both forms for the
    architecture: a registry that does not know the program, one that ships the one-CTA form only and an
    architecture outside the records keep the ``save_mode="full"`` forward split (the tape walk, then #5; decision
    ``launched``), as test knob ``fused_o_tape=0`` does; ``fused_o`` (the light output walk) does not gate it.  Every
    arm's out and saved tapes are the same bytes."""
    kcp, ctx, ops = _ckpt_setup()
    jit = kcp._registry()
    names = jit.records_of_program("kcp_fwdh_out_tape")
    forms = {int(jit.MODULES[n]["grid_rule"].get("ctas_per_head", 1)): n for n in names}
    assert sorted(forms) == [1, 2] and forms[2] == forms[1] + kcp.VH64_SUFFIX
    assert kcp.fused_o_tape_shipped() and not kcp.fused_o_tape_shipped("sm_90a")
    for arch in jit.ARCHES:
        assert kcp.fused_o_tape_shipped(arch), arch
    args = [ops[n] for n in LEAVES]
    o = {"backend": "fake", "save_mode": "full", "segments": 2}
    forwards, rec = _forward_calls(kcp)
    _nan_fresh_tapes(kcp, monkeypatch)

    def run(**kw):
        forwards.clear()
        out, saved = kcp.cake_gdn_nondet_train_kcp_forward(*args, ctx, **o, **kw)
        ((_call, progs),) = forwards
        return out, saved, kcp.LAST_CALL["fwd"]["fwd_o"], progs

    previous = kcp.set_backend("fake", rec)
    arms = {}
    try:
        arms["shipped"] = run()
        arms["knob0"] = run(test_knobs={"fused_o_tape": 0})
        arms["fused_o0"] = run(test_knobs={"fused_o": 0})
        modules = jit.MODULES
        one_form = {n: r for n, r in modules.items() if n != forms[2]}
        monkeypatch.setattr(jit, "MODULES", one_form)
        assert not kcp.fused_o_tape_shipped()
        arms["one_form"] = run()
        monkeypatch.setattr(jit, "MODULES", modules)
        real = jit.records_of_program

        def unknown(program):
            if program == "kcp_fwdh_out_tape":
                raise ValueError(f"unknown GDN KCP program {program!r}")
            return real(program)

        monkeypatch.setattr(jit, "records_of_program", unknown)
        assert not kcp.fused_o_tape_shipped()
        arms["unknown"] = run()
        monkeypatch.setattr(jit, "records_of_program", real)
        assert kcp.fused_o_tape_shipped()
    finally:
        kcp.set_backend("fake", previous)
    for key, (_out, _saved, decision, progs) in arms.items():
        fused = key in ("shipped", "fused_o0")
        assert decision == ("fused_tape" if fused else "launched"), (key, decision)
        assert ("kcp_fwdh_out_tape" in progs) == fused, (key, progs)
        assert ("kcp_fwdh_tape" in progs) == (not fused), (key, progs)
        assert progs.count("fwd_o") == int(not fused), (key, progs)
        assert "kcp_fwdh_out" not in progs, (key, progs)
    out0, saved0 = arms["knob0"][:2]
    for key, (out, saved, _d, _p) in arms.items():
        assert torch.equal(out, out0), key
        for n in TAPE_WALK_TAPES:
            assert torch.equal(_bytes(saved.tape(n)), _bytes(saved0.tape(n))), (key, n)


@pytest.mark.parametrize(
    "form",
    [
        dict(later_saver=False),
        dict(later_saver=True),
        dict(later_saver=False, recompute_output="always"),
    ],
    ids=("closing", "later", "always"),
)
def test_fused_tape_walk_keeps_the_workspace_figures(form) -> None:
    """The tape-and-output walk changes no workspace figure: a ``save_mode="full"`` region under a non-reentrant
    checkpoint on a caller-held workspace binds the same forward peak and the same backward peak (the recompute
    forward included) and leaves the same arena tapes and bytes with ``fused_o_tape`` 1 and 0 (the saved tapes and
    the output are fresh tensors either way); the gradients are the same bits."""
    kcp, ctx, ops = _ckpt_setup()
    opts = {k: v for k, v in form.items() if k != "later_saver"}
    figures = {}
    for fused in (1, 0):
        ws = _PeakWorkspace(kcp)
        leaves = [ops[n].detach().clone().requires_grad_(True) for n in LEAVES]
        scale = torch.full((), 1.5, dtype=torch.bfloat16, requires_grad=True)

        def fn(q, k, v, g, beta, _ws=ws, _fused=fused, _scale=scale):
            out = kcp.cake_gdn_nondet_kcp(
                q,
                k,
                v,
                g,
                beta,
                ctx,
                workspace=_ws,
                backend="fake",
                save_mode="full",
                test_knobs={"fused_o_tape": _fused},
                **opts,
            )
            return out * _scale if form["later_saver"] else out

        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *leaves, use_reentrant=False)
        assert kcp.LAST_CALL["fwd"]["fwd_o"] == ("fused_tape" if fused else "launched")
        fwd_peak = ws.peak
        ws.peak = 0
        out.backward(ops["dout"])
        figures[fused] = (
            fwd_peak,
            ws.peak,
            ws.bytes,
            sorted(ws._tapes),
            [lf.grad for lf in leaves],
        )
        assert kcp.kept_registry_size() == 0
    (f1, b1, e1, t1, g1), (f0, b0, e0, t0, g0) = figures[1], figures[0]
    assert (f1, b1, e1, t1) == (f0, b0, e0, t0), (figures[1][:4], figures[0][:4])
    for a, b in zip(g1, g0, strict=True):
        assert torch.equal(a, b)


# ----------------------------------------------------------------------------------------------------------------
# The gradient layout follows the consumer of q / k / v (issue #8367).  Only a packed Q/K/V split node that hands a
# packed dQKV row on (cake_gdn_nondet_train_kcp_packed_qkv_views's, or the deterministic package's cake_gdn_det_train_kcp_packed_qkv_views's) gets one;
# plain split / view / slice views of a packed row and leaf views get three contiguous gradients, which torch's split
# backward concatenates faster than three column views of one row.  The explicit backward keeps the pointer rule.

ROW_FORMS = ("contiguous", "packed", "det", "plain", "slice", "leaf")
CKPT_FORMS = ("direct", "nonreentrant", "reentrant", "handle")


def _row_views(kcp, form: str, row: torch.Tensor, H_qk: int, HV: int):
    """q / k / v of ``form`` over the packed ``[T, (2 H_qk + HV) * 128]`` row: ``"packed"`` =
    :func:`cake_gdn_nondet_train_kcp_packed_qkv_views`, ``"det"`` = the deterministic package's ``cake_gdn_det_train_kcp_packed_qkv_views``, ``"plain"`` =
    ``split`` + ``view``, ``"slice"`` = column slices + ``view``."""
    T, a = row.shape[0], H_qk * 128
    if form == "packed":
        return kcp.cake_gdn_nondet_train_kcp_packed_qkv_views(row, H_qk=H_qk, HV=HV)
    if form == "det":
        det = _det_package()
        return det.cake_gdn_det_train_kcp_packed_qkv_views(row, H_qk=H_qk, HV=HV)
    if form == "plain":
        parts = row.split((a, a, HV * 128), dim=-1)
        return tuple(
            p.view(T, n, 128) for p, n in zip(parts, (H_qk, H_qk, HV), strict=True)
        )
    assert form == "slice", form
    return (
        row[:, :a].view(T, H_qk, 128),
        row[:, a : 2 * a].view(T, H_qk, 128),
        row[:, 2 * a :].view(T, HV, 128),
    )


def _layout_pass(kcp, ctx, ops, form: str, mode: str, **opts) -> dict:
    """One autograd pass of ``cake_gdn_nondet_kcp`` (fake backend) with q / k / v of ``form`` over one packed leaf row
    holding ``q | k | v``: a :func:`_row_views` form, ``"contiguous"`` (three contiguous leaves holding the same
    values) or ``"leaf"`` (three leaves that are head views of the row).  ``mode``: ``"direct"``, a checkpoint whose
    function makes the views (``"nonreentrant"`` / ``"reentrant"``) or the non-reentrant checkpoint with a
    :class:`CakeGDNNondetKcpRecomputeHandle` (``"handle"``).  Returns ``out``, the row gradient (the leaves' gradients side by side
    for the leaf forms), ``dg``, ``dbeta``, the backward's ``grad_layout``, the ``PACKED_SPLIT_STATS`` deltas of this
    package (``split``) and of the deterministic one (``det_split``, ``"det"`` form only) and the (strides,
    contiguous) of every gradient a hook on the ``q`` that the backward reaches received."""
    T, H_qk, HV = ops["q"].shape[0], ops["q"].shape[1], ops["v"].shape[1]
    a = H_qk * 128
    row = torch.cat([ops[n].reshape(T, -1) for n in ("q", "k", "v")], -1)
    row = row.contiguous().requires_grad_(True)
    g, beta = (ops[n].detach().clone().requires_grad_(True) for n in ("g", "beta"))
    leaves: tuple = ()
    if form == "contiguous":
        leaves = tuple(ops[n].detach().clone().requires_grad_(True) for n in "qkv")
    elif form == "leaf":
        cols = ((0, a), (a, 2 * a), (2 * a, None))
        leaves = tuple(
            row.detach()[:, lo:hi].view(T, n, 128).requires_grad_(True)
            for (lo, hi), n in zip(cols, (H_qk, H_qk, HV), strict=True)
        )
    hooks: list = []
    armed: list = []
    o = {"backend": "fake", **opts}
    if mode == "handle":
        o["save_transfer"] = False

    def fn(row, g, beta, *rest):
        q, k, v = rest[:3] if leaves else _row_views(kcp, form, row, H_qk, HV)
        # the q the backward reaches (a reentrant checkpoint: the re-run's; its first pass runs without grad)
        if torch.is_grad_enabled() and q.requires_grad and not armed:
            armed.append(True)
            q.register_hook(
                lambda gr: hooks.append((tuple(gr.stride()), gr.is_contiguous()))
            )
        handle = rest[-1] if mode == "handle" else None
        return kcp.cake_gdn_nondet_kcp(
            q, k, v, g, beta, ctx, recompute_handle=handle, **o
        )

    args = (row, g, beta, *leaves)
    if mode == "handle":
        args = (*args, kcp.CakeGDNNondetKcpRecomputeHandle())
    det_kcp = _det_package("cake_kcp") if form == "det" else None
    before = dict(kcp.PACKED_SPLIT_STATS)
    det_before = dict(det_kcp.PACKED_SPLIT_STATS) if det_kcp is not None else {}
    if mode == "direct":
        out = fn(*args)
    else:
        from torch.utils.checkpoint import checkpoint

        out = checkpoint(fn, *args, use_reentrant=mode == "reentrant")
    out.backward(ops["dout"])
    return {
        "out": out.detach(),
        "row": torch.cat([x.grad.reshape(T, -1) for x in leaves], -1)
        if leaves
        else row.grad,
        "dg": g.grad,
        "dbeta": beta.grad,
        "grad_layout": kcp.LAST_CALL["bwd"]["grad_layout"],
        "split": {n: kcp.PACKED_SPLIT_STATS[n] - before[n] for n in before},
        "det_split": {
            n: det_kcp.PACKED_SPLIT_STATS[n] - det_before[n] for n in det_before
        },
        "hooks": hooks,
    }


def test_the_deterministic_packed_split_is_named_by_its_module_and_class() -> None:
    """``cake_kcp._DET_PACKED_SPLIT`` names the deterministic package's packed Q/K/V split by module and class name
    (this package never imports it): the pair equals the class's own ``__module__`` and ``__qualname__``, so the
    outputs of either package's split hand their packed gradient row on.  A stale pair costs only speed (a
    concatenation) and no error, so it is checked here on the CPU, with a hard import of the class."""
    kcp = _package("cake_kcp")
    det_cls = _det_package("cake_kcp").CakeGDNDetKcpPackedQKVSplitFunction
    assert (det_cls.__module__, det_cls.__qualname__) == kcp._DET_PACKED_SPLIT
    assert kcp._DET_PACKED_SPLIT == (
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp",
        "CakeGDNDetKcpPackedQKVSplitFunction",
    )
    assert kcp._is_det_packed_split(det_cls)


def test_the_deterministic_packed_split_is_recognized_under_a_vendored_prefix() -> None:
    """A build that vendors the deterministic package under another parent package (or as a top-level package)
    imports its split class under another module prefix: the class is still recognized by the module's last two parts
    and its name.  Another class name or another module tail is not.  Class objects only (no tensor, no device)."""
    kcp = _package("cake_kcp")

    def stand_in(
        module: str, qualname: str = "CakeGDNDetKcpPackedQKVSplitFunction"
    ) -> type:
        cls = type(qualname.rsplit(".", 1)[-1], (), {})
        cls.__module__, cls.__qualname__ = module, qualname
        return cls

    for module in (
        "flashinfer.experimental.cake.cake_gdn_det_train_kcp.cake_kcp",
        "cake_gdn_det_train_kcp.cake_kcp",
        "vendored.kernels.cake_gdn_det_train_kcp.cake_kcp",
    ):
        assert kcp._is_det_packed_split(stand_in(module)), module
    for module, qualname in (
        (
            "vendored.cake_gdn_det_train_kcp.cake_kcp",
            "CakeGDNNondetKcpPackedQKVSplitFunction",
        ),
        (
            "vendored.cake_gdn_nondet_train_kcp.cake_kcp",
            "CakeGDNDetKcpPackedQKVSplitFunction",
        ),
        (
            "vendored.xcake_gdn_det_train_kcp.cake_kcp",
            "CakeGDNDetKcpPackedQKVSplitFunction",
        ),
        ("cake_gdn_det_train_kcp.cake_kcp_plan", "CakeGDNDetKcpPackedQKVSplitFunction"),
    ):
        assert not kcp._is_det_packed_split(stand_in(module, qualname)), (
            module,
            qualname,
        )


@pytest.mark.parametrize("mode", CKPT_FORMS)
@pytest.mark.parametrize("form", ROW_FORMS[1:])
def test_gradient_layout_follows_the_consumer_of_the_views(form, mode) -> None:
    """Head views of one packed row get ``dq`` / ``dk`` / ``dv`` as the same views of one packed dQKV row only when
    they are the outputs of a split node that hands that row on (``cake_gdn_nondet_train_kcp_packed_qkv_views`` or the deterministic
    package's ``cake_gdn_det_train_kcp_packed_qkv_views``: ``grad_layout="packed"``, that split's ``aliased`` count + 1); plain
    ``split`` / ``view`` and slice views and leaf views get three contiguous gradients (``"separate"``; no split
    count moves).  The decision is the original forward's (a non-reentrant checkpoint's recompute sees detached
    tensors), the reentrant re-run's own (it makes the views) and the handle form's.  A hook on ``q`` fires once,
    with a contiguous gradient exactly when the layout is separate.  ``out``, the row gradient, ``dg`` and ``dbeta``
    are bitwise those of contiguous operands (the same programs on the same operands, only the gradient layout
    differs: a property of the fake backend's emulation) and within the budget of the FP64 reference."""
    kcp, ctx, ops = _ckpt_setup()
    ref = _layout_pass(kcp, ctx, ops, "contiguous", mode)
    assert ref["grad_layout"] == "separate" and ref["split"] == {
        "aliased": 0,
        "concatenated": 0,
    }
    got = _layout_pass(kcp, ctx, ops, form, mode)
    what = f"form={form} mode={mode}"
    packed = form in ("packed", "det")
    assert got["grad_layout"] == ("packed" if packed else "separate"), what
    assert got["split"] == {"aliased": int(form == "packed"), "concatenated": 0}, (
        what,
        got["split"],
    )
    if form == "det":
        assert got["det_split"] == {"aliased": 1, "concatenated": 0}, (
            what,
            got["det_split"],
        )
    assert len(got["hooks"]) == 1 and got["hooks"][0][1] == (not packed), (
        what,
        got["hooks"],
    )
    for n in ("out", "row", "dg", "dbeta"):
        assert torch.equal(got[n], ref[n]), (what, n)
    T, H_qk = ops["q"].shape[0], ops["q"].shape[1]
    a = H_qk * 128
    res = {
        "out": got["out"],
        "dq": got["row"][:, :a].reshape(T, H_qk, 128),
        "dk": got["row"][:, a : 2 * a].reshape(T, H_qk, 128),
        "dv": got["row"][:, 2 * a :].reshape(ops["v"].shape),
        "dg": got["dg"],
        "dbeta": got["dbeta"],
    }
    _arms_within_budget({"views": res}, _ckpt_reference(), ("out",) + CKPT_GRADS, what)


def test_explicit_backward_keeps_the_packed_layout_of_any_packed_views() -> None:
    """``cake_gdn_nondet_train_kcp_backward`` has no consumer to look at: head views of one packed row (here plain ``split`` views) get
    ``dq`` / ``dk`` / ``dv`` as the same views of one fresh packed row, whatever made the views."""
    kcp, ctx, ops = _ckpt_setup()
    T, H_qk, HV = ops["q"].shape[0], ops["q"].shape[1], ops["v"].shape[1]
    row = torch.cat([ops[n].reshape(T, -1) for n in ("q", "k", "v")], -1).contiguous()
    q, k, v = _row_views(kcp, "plain", row, H_qk, HV)
    o = {"backend": "fake"}
    out, saved = kcp.cake_gdn_nondet_train_kcp_forward(
        q, k, v, ops["g"], ops["beta"], ctx, **o
    )
    dq, dk, dv, _, _ = kcp.cake_gdn_nondet_train_kcp_backward(
        q, k, v, ops["g"], ops["beta"], ctx, ops["dout"], saved=saved, **o
    )
    assert kcp.LAST_CALL["bwd"]["grad_layout"] == "packed"
    assert dq.stride() == q.stride() and dv.stride() == v.stride()
    step = dq.element_size() * H_qk * 128
    assert dk.data_ptr() == dq.data_ptr() + step
    assert dv.data_ptr() == dq.data_ptr() + 2 * step


def test_backward_graph_key_carries_the_gradient_layout(monkeypatch) -> None:
    """The backward's CUDA-graph key holds the layout decision next to the saved-set flag: a graph captured for
    packed views never replays its packed gradient row for plain views of the same pointers, or the reverse."""
    kcp, ctx, ops = _ckpt_setup()
    flags: list = []

    def spy(kind, *args):
        if kind == "bwd":
            flags.append(args[-1])
        return None  # never captured: every call runs eagerly

    monkeypatch.setattr(kcp, "_graph_entry", spy)
    ws = kcp.CakeGDNNondetKcpWorkspace()
    layouts = [
        _layout_pass(kcp, ctx, ops, form, "direct", graph="replay", workspace=ws)[
            "grad_layout"
        ]
        for form in ("packed", "plain", "packed")
    ]
    assert layouts == ["packed", "separate", "packed"]
    assert [f[-1] for f in flags] == [True, False, True], flags
    assert len({f[:-1] for f in flags}) == 1, flags
