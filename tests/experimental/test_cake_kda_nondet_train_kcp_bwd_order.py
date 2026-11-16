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

The two backward placement knobs of the non-deterministic context-parallel (KCP) KDA training route
(``cake_kcp``, shipped defaults ``"under_b1"`` / ``"side"``; private test knobs that force ``"first"`` / ``"main"``):
``bwd_tape_order="under_b1"`` issues the saved-mode backward's ``P = 1`` K1 tape replay after the b1 issue (on the
backwards that skip ``kcp_dav``, its one earlier reader of ``v_new``: ``cake_kcp_plan.kcp_dav_needed``) and
``bwd_k3_stream="side"`` issues the dm-form K3 on the side stream between the last writer of its operands and the b1
issue; every backward their rules refuse runs ``"first"`` / ``"main"``.

CPU rows: the knobs' gate and values, the dAqk allowlist against the records every plan binds (``H >= 16`` and a davq
record, no stale entry) and ``kcp_dav_needed`` against every plan (true exactly where the plan dispatches its dav
stage, H 9-15, whose fused gradient stores no dAqk), the two rules :func:`cake_kcp._tape_order` /
:func:`cake_kcp._k3_stream` per refusal, the launch order of the real backward driver under each knob pair (stubbed
launches; the defaults the ``"under_b1"`` / ``"side"`` order) and :func:`_bwd_sequence_first_main` mapping each
order back to the ``"first"`` / ``"main"`` one, the side stream's join on an error, and that the GPU rows reach
``P = 1`` with ``H >= 16`` on every target SM count.
GPU rows (``BWD_ORDER_ROWS``: W = 2 and 4, H 12 to 64, both gate modes, the norm off and on): every output and
gradient of each knob pair (the defaults among them) bitwise the forced ``"first"`` / ``"main"`` form's in the plain
node (the saved c1 receive tensor: no dm form), with
``save_transfer=False`` (the dm form), under a recompute handle (``"states"``) with the hand-off off and ``"auto"``,
and with ``saved_rows="wy"`` plain and under the handle; every backward's recorded order / stream and refusal as the
rank's facts decide (hand-off bound, ``P``, tokens, the plan's fused-gradient record, the K3 list and the
``walk_streams`` rule), the launches the ``"first"`` / ``"main"`` form's with the moved launch where the knob puts it
(:func:`_bwd_sequence_first_main` gives back exactly the reference sequence) and the collectives in the same order;
the dAqk rows NaN-filled right after the ``dav`` stage change no bit where the backward skips ``kcp_dav`` (both
orders) and reach the gradients on the H12 row (the negative control: ``kcp_dav`` runs there and its fused gradient
reads ``kcp_dav``'s dAqk); in the activated gate without the norm, ``graph="replay"`` with both knobs (eager, capture,
replays: one capture per kind) bitwise the eager ``"first"`` / ``"main"`` form.
"""

from __future__ import annotations

import json
import os
import tempfile
import traceback
import types

import pytest
import torch
import torch.distributed as dist

from tests.experimental.test_cake_kda_nondet_train_kcp_dist import (
    GRAPH_EVENTS,
    _bound_rank_threads,
    _bwd_sequence_first_main,
    _cpu_context,
    _eager_capture_replay,
    _init_group,
    _local_slices,
    _partition,
    _programs_exported,
    _skip_unless_runnable,
    _start_ranks,
    cake_jit,
    cake_kcp,
    cake_kcp_plan,
)
from tests.experimental.test_cake_kda_nondet_train_kcp_handoff import (
    _mode_inputs,
    _mode_row_id,
    _same,
)
from tests.experimental.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    make_inputs,
)

ARCHES = ("sm_100a", "sm_103a", "sm_107a")
# the knob pairs of the GPU rows (the private test knobs): "first_main" forces the previous order, the reference of
# the bitwise and the launch-sequence checks; "base" = no knob, the shipped defaults (as "both"); "tape_b1" / "k3_side" move one
# launch each; "both_two" forces two streams by the walk_streams rule (also for the forward's summary pairs) where the
# K3 list alone keeps one
ARMS = {
    "first_main": {"bwd_tape_order": "first", "bwd_k3_stream": "main"},
    "base": {},
    "tape_b1": {"bwd_tape_order": "under_b1", "bwd_k3_stream": "main"},
    "k3_side": {"bwd_tape_order": "first", "bwd_k3_stream": "side"},
    "both": {"bwd_tape_order": "under_b1", "bwd_k3_stream": "side"},
    "both_two": {
        "bwd_tape_order": "under_b1",
        "bwd_k3_stream": "side",
        "walk_streams": "two",
    },
}
# the node forms of the GPU rows: (save_transfer, recompute handle, checkpoint_handoff, saved_rows); the dm-form K3
# runs where the saved set keeps no c1 receive tensor (save_transfer=False, the handle's "states" tier)
MODES = {
    "plain": (None, False, "auto", "auto"),
    "plain_nt": (False, False, "auto", "auto"),
    "handle_off": (None, True, "off", "auto"),
    "handle_auto": (None, True, "auto", "auto"),
    "rows_wy": (None, False, "auto", "wy"),
    "handle_wy": (None, True, "auto", "wy"),
}
DM_MODES = ("plain_nt", "handle_off", "handle_auto", "handle_wy")
# (lengths, H, W, layout, gate mode, use_qk_l2norm_in_kernel): two H64 W4 rows (8 in-and-out fragments on the
# interior ranks: P = 1, two streams by the rule on the K3 list), a W2 H32 row with the norm, an H16 row (the dAqk
# allowlist's head bound) and the H12 row whose fused gradient takes its dAqk from kcp_dav (refusal and the poison's
# control)
BWD_ORDER_ROWS = [
    ((6144, 4096, 3584, 2560), 64, 4, "zigzag_per_document", "g", False),
    ((6144, 4096, 3584, 2560), 64, 4, "zigzag_per_document", "raw", True),
    ((3000, 1000), 32, 2, "zigzag_per_document", "g", True),
    ((2048, 5, 3000, 1), 16, 4, "zigzag_per_document", "raw", False),
    ((100, 640, 77, 1000, 129), 12, 4, "zigzag_per_document", "g", False),
]


# --------------------------------------------------------------------------------------------- CPU rows


def test_bwd_order_knobs_are_gated_and_validated(monkeypatch) -> None:
    """``bwd_tape_order`` / ``bwd_k3_stream`` are private test knobs: refused without
    :func:`cake_kcp_plan.enable_test_knobs`, never public keywords, validated against their value tuples; the shipped
    defaults are the moved forms (``"under_b1"`` / ``"side"``, each where its rule admits it), and the knobs force the
    previous ``"first"`` / ``"main"`` for the A/B record."""
    assert cake_kcp.BWD_TAPE_ORDERS == ("first", "under_b1")
    assert cake_kcp.BWD_TAPE_ORDER_DEFAULT == "under_b1"
    assert cake_kcp.BWD_K3_STREAMS == ("main", "side")
    assert cake_kcp.BWD_K3_STREAM_DEFAULT == "side"
    for knob, values, default in (
        ("bwd_tape_order", cake_kcp.BWD_TAPE_ORDERS, "under_b1"),
        ("bwd_k3_stream", cake_kcp.BWD_K3_STREAMS, "side"),
    ):
        monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", False)
        assert cake_kcp.P2_KNOBS[knob] == values
        assert knob not in cake_kcp._OPTION_KEYS
        assert getattr(cake_kcp._options({}), knob) == default
        # the previous form (the A/B record's value) only through the gated hook
        forced = values[0]
        assert forced != default
        with pytest.raises(ValueError, match="enable_test_knobs"):
            cake_kcp._options({"_kcp_test_knobs": {knob: forced}})
        with pytest.raises(TypeError):
            cake_kcp._options({knob: forced})
        monkeypatch.setattr(cake_kcp_plan, "_TEST_KNOBS_ENABLED", True)
        for v in values:
            assert getattr(cake_kcp._options({"_kcp_test_knobs": {knob: v}}), knob) == v
        for bad in ("", "both", values[1].upper()):
            with pytest.raises(ValueError, match=knob):
                cake_kcp._options({"_kcp_test_knobs": {knob: bad}})


def _plans():
    """``(arch, sm, gate, l2n, S, docs, H, record)`` of the replay fused-gradient record of every plan of the grid:
    the three architectures at their SM counts, both gate modes, the norm off and fused, head counts around the dAqk allowlist's
    bound and three token / fragment shapes (``ValueError`` plans -- a mode an architecture has no record class
    for -- are left out)."""
    out = []
    for arch in ARCHES:
        for sm in cake_kcp_plan.QK_L2NORM_SMS[arch]:
            for gate in ("g", "raw"):
                for l2n in ("off", "fused"):
                    for S, docs in ((4096, 8), (1500, 2), (16384, 1)):
                        for H in (1, 4, 8, 9, 12, 15, 16, 24, 32, 48, 64):
                            try:
                                p = cake_kcp.cake_backend.plan(
                                    S,
                                    docs,
                                    H,
                                    sm,
                                    arch,
                                    gate=gate,
                                    **cake_kcp_plan.qk_l2norm_plan_kwargs(l2n),
                                )
                            except ValueError:
                                continue
                            rec = p.record("dqkg", cake_kcp.LOCAL_PATH_BWD)
                            out.append((arch, sm, gate, l2n, S, docs, H, rec, p))
    return out


def test_dav_daqk_allowlist_matches_the_plan_records() -> None:
    """:data:`cake_kcp_plan.DAV_DAQK_COVERED` against the record each plan binds for the fused gradient: below
    :data:`DAV_DAQK_MIN_HEADS` every plan refuses (``"heads"``: the serial route's ``dqkg_int*`` takes its dAqk from
    ``kcp_dav``, and no statement covers a davq record there); from 16 heads every plan binds a davq record and is
    admitted (the allowlist is complete), the serial ``dqkg_int*`` records are never listed, and every listed
    ``(record, arch)`` is bound by some plan (no stale entry).  CPU only: the plans read no tensor."""
    plans = _plans()
    assert {a for a, *_ in plans} == set(ARCHES), sorted({a for a, *_ in plans})
    assert cake_kcp_plan.DAV_DAQK_MIN_HEADS == 16
    for rec in cake_kcp_plan.DAV_DAQK_COVERED:
        assert "davq" in rec, rec
    hit = set()
    for arch, sm, gate, l2n, S, docs, H, rec, _p in plans:
        why = cake_kcp_plan.dav_daqk_refusal(rec, arch, H)
        case = (arch, sm, gate, l2n, S, docs, H, rec)
        if H < 16:
            assert why == "heads", case
            continue
        assert "davq" in rec and why is None, (case, why)
        hit.add((rec, arch))
    listed = {
        (r, a) for r, arches in cake_kcp_plan.DAV_DAQK_COVERED.items() for a in arches
    }
    assert listed == hit, (sorted(listed - hit), sorted(hit - listed))
    assert cake_kcp_plan.dav_daqk_refusal("dqkg_int", "sm_103a", 64) == "record"
    assert cake_kcp_plan.dav_daqk_refusal("dqkg_int_davq", "sm_100a", 64) == "arch"
    assert cake_kcp_plan.dav_daqk_refusal(None, "sm_103a", 64) == "record"


def test_kcp_dav_needed_matches_the_plan_records() -> None:
    """:func:`cake_kcp_plan.kcp_dav_needed` on every plan of the grid: true exactly where the plan dispatches its
    ``dav`` stage on the replay path, which is H 9-15 on every architecture, gate mode and norm mode (the serial route
    below 16 heads); there its fused gradient is a ``dqkg_int*`` form that stores no dAqk and the dAqk allowlist
    refuses it (``"heads"``); everywhere else the fused gradient is a davq form (it stores every dAqk row GLM intra
    reads) and the backward skips ``kcp_dav``.  CPU only."""
    seen = set()
    for arch, sm, gate, l2n, S, docs, H, rec, p in _plans():
        case = (arch, sm, gate, l2n, S, docs, H, rec)
        need = cake_kcp_plan.kcp_dav_needed(p)
        assert need == p.dispatched("dav", cake_kcp.LOCAL_PATH_BWD), case
        assert need == (9 <= H <= 15), case
        if need:
            assert "davq" not in rec and rec.startswith("dqkg_int"), case
            assert cake_kcp_plan.dav_daqk_refusal(rec, arch, H) == "heads", case
        else:
            assert "davq" in rec, case
        seen.add((arch, need))
    assert seen == {(a, n) for a in ARCHES for n in (False, True)}, sorted(seen)


def _order_call(**kw) -> types.SimpleNamespace:
    """A call with the fields :func:`cake_kcp._tape_order` reads (a covered H64 sm_103a plan on a replay arena,
    the a2a exchange, ``wait_first``, the kernel backend, ``P = 1``), overridden by ``kw``."""
    record = kw.pop("record", "dqkg_int_davq")
    dav = kw.pop("dav", False)
    local = types.SimpleNamespace(
        plan=types.SimpleNamespace(
            record=lambda stage, path: record,
            dispatched=lambda stage, path: dav and stage == "dav",
        ),
        arch=kw.pop("arch", "sm_103a"),
        layout_passes=kw.pop("layout_passes", {}),
    )
    base = dict(
        bwd_tape_order="under_b1",
        local=local,
        tb=types.SimpleNamespace(P=1),
        exchange="a2a",
        schedule="wait_first",
        backend=cake_kcp.KernelBackend(),
        T_r=1000,
        H=64,
        scratch={},
        opts=types.SimpleNamespace(kcp_dav=kw.pop("kcp_dav", "auto")),
    )
    base.update(kw)
    return types.SimpleNamespace(**base)


def test_tape_order_rule_per_refusal() -> None:
    """:func:`cake_kcp._tape_order`: ``"under_b1"`` on an admitted backward, else ``"first"`` with the first
    refusal in rule order; both recorded in the call's scratch (no refusal when the knob is ``"first"``)."""
    lean = {cake_kcp.LOCAL_LAYOUT_BWD_LEAN: 2}
    h = {"h": torch.zeros(1)}
    intra = {"aqk_int": torch.zeros(1), "akk_int": torch.zeros(1)}
    wy = {**intra, "w": torch.zeros(1), "u": torch.zeros(1)}
    cases = [
        (dict(bwd_tape_order="first"), {}, "first", None),
        ({}, dict(replay=True), "first", "replay"),
        ({}, dict(tape=h), "first", "handoff"),
        (dict(tb=types.SimpleNamespace(P=2)), {}, "first", "segments"),
        (dict(exchange="ring"), {}, "first", "ring"),
        (dict(schedule="overlap"), {}, "first", "schedule"),
        (dict(backend=cake_kcp.Backend()), {}, "first", "backend"),
        (dict(local=None), {}, "first", "no_local"),
        (dict(T_r=0), {}, "first", "no_local"),
        (dict(layout_passes=lean), {}, "first", "lean"),
        (dict(layout_passes=lean), dict(rows=intra), "first", "lean"),
        (dict(layout_passes=lean), dict(rows=wy), "under_b1", None),
        (dict(H=12, record="dqkg_int", dav=True), {}, "first", "dav"),
        (
            dict(H=15, record="dqkg_int_raw", arch="sm_100a", dav=True),
            {},
            "first",
            "dav",
        ),
        (dict(kcp_dav="always"), {}, "first", "dav"),
        (dict(layout_passes=lean, kcp_dav="always"), {}, "first", "lean"),
        (dict(H=4, record="dqkg_int_davq_v24", arch="sm_100a"), {}, "under_b1", None),
        (dict(H=8, record="dqkg_int_davq_raw", arch="sm_107a"), {}, "under_b1", None),
        ({}, {}, "under_b1", None),
        ({}, dict(rows=intra), "under_b1", None),
        (
            dict(H=16, record="dqkg_int_davq_nohint_raw", arch="sm_107a"),
            {},
            "under_b1",
            None,
        ),
    ]
    for fields, args, want, why in cases:
        call = _order_call(**fields)
        got = cake_kcp._tape_order(
            call,
            replay=args.get("replay", False),
            tape=args.get("tape"),
            rows=args.get("rows"),
        )
        assert got == want, (fields, args, got)
        assert call.scratch == {
            "bwd_tape_order": want,
            "bwd_tape_order_refusal": why,
        }, (fields, args, call.scratch)


def test_k3_stream_rule_per_refusal(monkeypatch) -> None:
    """:func:`cake_kcp._k3_stream`: ``"side"`` on an admitted backward whose table set already ran its K3 over the
    list on the device, else ``"main"`` with the first refusal in rule order and, for the table set's first K3, the
    key the driver marks after issuing it on the call's stream; both recorded in the call's scratch.  The rule reads
    the table set's ``walk_streams`` decision of the K3 list with the call's ``walk_streams`` knob.  CPU only: the
    device's SM count and the side stream are stubbed."""
    monkeypatch.setattr(cake_kcp.cake_jit, "sm_count", lambda index: 152)
    made = []
    monkeypatch.setattr(cake_kcp, "_pair_streams", lambda index: made.append(index))
    asked = []

    def tables(P=1, walk=(3, 5), decided="two"):
        def rule(w, H, sm, knob):
            asked.append((tuple(w), H, sm, knob))
            return decided

        return types.SimpleNamespace(P=P, walk_m_b1=list(walk), walk_streams=rule)

    def call(**kw):
        base = dict(
            bwd_k3_stream="side",
            tb=tables(),
            exchange="a2a",
            device=torch.device("cuda", 0),
            backend=cake_kcp.KernelBackend(),
            host_trim=1,
            H=64,
            walk_streams="auto",
            scratch={},
        )
        base.update(kw)
        return types.SimpleNamespace(**base)

    cases = [
        (dict(bwd_k3_stream="main"), "dm", None),
        ({}, "d", "form"),
        (dict(tb=tables(P=2)), "dm", "segments"),
        (dict(exchange="ring"), "dm", "ring"),
        (dict(tb=tables(walk=())), "dm", "no_walk"),
        (dict(device=torch.device("cpu")), "dm", "backend"),
        (dict(backend=cake_kcp.Backend()), "dm", "backend"),
        (dict(host_trim=0), "dm", "host_trim"),
        (dict(tb=tables(decided="one")), "dm", "rule"),
    ]
    for fields, form, why in cases:
        c = call(**fields)
        stream, mark = cake_kcp._k3_stream(c, form=form)
        assert (stream, mark) == ("main", None), (fields, form, stream, mark)
        assert c.scratch == {"bwd_k3_stream": "main", "bwd_k3_stream_refusal": why}
    assert not made  # no side stream without a two-stream decision
    # the walk_streams knob reaches the rule
    asked.clear()
    c = call(walk_streams="two")
    cake_kcp._k3_stream(c, form="dm")
    assert asked == [((3, 5), 64, 152, "two")], asked
    made.clear()
    # the table set's first K3 over its list: the call's stream and the key to mark; once marked, the side stream
    c = call()
    stream, mark = cake_kcp._k3_stream(c, form="dm")
    assert stream == "main" and mark == (id(c.tb.walk_m_b1), 0), (stream, mark)
    assert c.scratch["bwd_k3_stream_refusal"] == "first_launch"
    assert made == [0]
    c.tb.__dict__.setdefault("_k3_side_seen", set()).add(mark)
    c.scratch = {}
    assert cake_kcp._k3_stream(c, form="dm") == ("side", None)
    assert c.scratch == {"bwd_k3_stream": "side", "bwd_k3_stream_refusal": None}


class _LogCall(types.SimpleNamespace):
    """A call for the real :func:`cake_kcp._backward` whose launches, merges, collectives and waits are logged in
    ``log`` in issue order; every other method does nothing."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        return lambda *a, **k: None

    def walk(self, name, walk, **_):
        if len(walk):
            self.log.append(name)

    def stage(self, name, **_):
        if name == self.fail_at:
            raise RuntimeError(f"stub failure at {name}")
        self.log.append(name)

    def merge(self, family, **_):
        self.log.append("merge:" + family)

    def a2a(self, tag, *_a, **_k):
        self.log.append("a2a:" + tag)
        return tag

    def wait(self, work):
        self.log.append(f"wait:{work}")


def _driver_log(
    monkeypatch, tb, ctx, *, order: str, k3: str, transfer: bool, fail_at=None
):
    """The issue log of the real saved-mode backward driver at ``P = 1`` (an interior rank whose fragment crosses in
    and out) under ``bwd_tape_order=order`` / ``bwd_k3_stream=k3``, the saved set with (``transfer``) or without the
    c1 receive tensor; :func:`cake_kcp._tape_order` runs for real (a covered plan), the K3 stream decision and its
    fork are stubbed (they need a CUDA device: the fork logs ``fork``, its K3 launch and ``join:record``, the call's
    stream logs ``join:wait``)."""

    def k3_stream(call, *, form):
        side = call.bwd_k3_stream == "side" and form == "dm"
        call.scratch["bwd_k3_stream"] = "side" if side else "main"
        return ("side" if side else "main"), None

    def fork(call):
        call.log.append("fork")
        call.walk("kcp_transfer_walk", call.tb.walk_m_b1)
        call.log.append("join:record")
        return types.SimpleNamespace(
            wait_event=lambda ev: call.log.append("join:wait")
        ), "join"

    monkeypatch.setattr(cake_kcp, "_k3_stream", k3_stream)
    monkeypatch.setattr(cake_kcp, "_k3_side_fork", fork)
    monkeypatch.setattr(
        cake_kcp,
        "_rank_check",
        lambda call, boundary: call.log.append("check:" + boundary),
    )
    monkeypatch.setattr(
        cake_kcp, "_adj_tapes", lambda call, **_: call.log.append("adj_tapes")
    )
    local = types.SimpleNamespace(
        plan=types.SimpleNamespace(
            record=lambda stage, path: "dqkg_int_davq",
            # the davq fused gradient stores every dAqk row: the plan dispatches no dav (kcp_dav skipped)
            dispatched=lambda stage, path: stage != "dav",
        ),
        arch="sm_103a",
        layout_passes={},
    )
    call = _LogCall(
        opts=types.SimpleNamespace(kcp_dav=cake_kcp.KCP_DAV_DEFAULT),
        tb=tb,
        ctx=ctx,
        T_r=int(ctx.T_r),
        H=64,
        backward=True,
        exchange="a2a",
        schedule="wait_first",
        merge_fuse="carry",
        host_trim=1,
        l2n=False,
        l2n_mode="off",
        gate="g",
        local=local,
        local_values=None,
        backend=cake_kcp.KernelBackend(),
        bwd_tape_order=order,
        bwd_k3_stream=k3,
        bufs={},
        scratch={},
        log=[],
        fail_at=fail_at,
    )
    saved = types.SimpleNamespace(
        S_in=torch.zeros(1),
        c1_recv=torch.zeros(1) if transfer else None,
        M_seg=None,
        carry=None,
        tape=None,
        rows=None,
    )
    try:
        cake_kcp._backward(call, saved)
    except RuntimeError as exc:
        if fail_at is None or str(exc) != f"stub failure at {fail_at}":
            raise
        call.log.append("raised")
    return call.log, call.scratch


def test_backward_issue_order_per_knob(monkeypatch) -> None:
    """The real backward driver at ``P = 1`` on an interior rank: under ``"first"`` / ``"main"`` the K1 tape replay
    right after the local rows, then ``kcp_dav``, K2 ``_offset`` and (dm form) K3, the check and b1; ``"under_b1"``
    moves the replay between the b1 issue and its wait; ``"side"`` forks K3 right after the WY program and joins
    before the check and b1; both together compose, and the shipped defaults run both; the d form (a saved c1 receive
    tensor) has no K3 to move.  The collectives keep their order in every form, and the join runs even when a launch
    between fork and join raises.  :func:`_bwd_sequence_first_main` maps each form's launches (the logged issue order
    without the stubs' fork / join / check / wait entries) back to the ``"first"`` / ``"main"`` form's, and a record
    whose decisions are not its sequence's onto another sequence.  CPU only: the launches are stubbed."""
    ctx = _cpu_context((3000,), 3, "contiguous", rank=1, sm=1)
    tb = cake_kcp._Tables(ctx, 64, fused_walks=cake_kcp._options({}).fused_walks)
    assert tb.P == 1 and len(tb.walk_m_b1) and len(tb.walk_in) and len(tb.walk_all_p0)
    head = ["bwd_prep", "fwd_intra", "wy"]
    tail = [
        "wait:b1",
        "merge:adj_b1",
        "a2a:b2",
        "wait:b2",
        "adj_tapes",
        *cake_kcp.TAIL_STAGES,
    ]
    want = {
        ("first", "main"): head
        + [
            "kcp_fwdh_tape",
            "dav",
            "kcp_dhu_offset",
            "kcp_transfer_walk",
            "check:saved",
            "a2a:b1",
        ]
        + tail,
        ("under_b1", "main"): head
        + [
            "dav",
            "kcp_dhu_offset",
            "kcp_transfer_walk",
            "check:saved",
            "a2a:b1",
            "kcp_fwdh_tape",
        ]
        + tail,
        ("first", "side"): head
        + [
            "fork",
            "kcp_transfer_walk",
            "join:record",
            "kcp_fwdh_tape",
            "dav",
            "kcp_dhu_offset",
        ]
        + ["join:wait", "check:saved", "a2a:b1"]
        + tail,
        ("under_b1", "side"): head
        + [
            "fork",
            "kcp_transfer_walk",
            "join:record",
            "dav",
            "kcp_dhu_offset",
            "join:wait",
        ]
        + ["check:saved", "a2a:b1", "kcp_fwdh_tape"]
        + tail,
    }
    for (order, k3), expect in want.items():
        log, scratch = _driver_log(
            monkeypatch, tb, ctx, order=order, k3=k3, transfer=False
        )
        assert log == expect, (order, k3, log)
        assert (
            scratch["bwd_tape_order"] == order
            and scratch["bwd_tape_order_refusal"] is None
        )
        assert scratch["bwd_k3_stream"] == k3
    # the shipped defaults: the replay under b1 and K3 on the side stream
    opts = cake_kcp._options({})
    log, _ = _driver_log(
        monkeypatch,
        tb,
        ctx,
        order=opts.bwd_tape_order,
        k3=opts.bwd_k3_stream,
        transfer=False,
    )
    assert log == want[("under_b1", "side")], log

    def record(order: str, k3: str, log: list) -> dict:
        """The LAST_CALL form of a logged backward: its launches and collectives, and its two decisions."""
        stub = ("fork", "join:", "check:", "wait:")
        seq = [x for x in log if not x.startswith(stub)]
        return {"sequence": seq, "bwd_tape_order": order, "bwd_k3_stream": k3}

    ref = record("first", "main", want[("first", "main")])["sequence"]
    for (order, k3), expect in want.items():
        assert _bwd_sequence_first_main(record(order, k3, expect)) == ref, (order, k3)
        for other in want:
            if other != (order, k3):
                got = _bwd_sequence_first_main(record(*other, expect))
                assert got != ref, (order, k3, other, got)
    # the d form: no K3, the adjoint merge reads the saved c1 receive tensor
    d_ref = None
    for order in ("first", "under_b1"):
        log, scratch = _driver_log(
            monkeypatch, tb, ctx, order=order, k3="side", transfer=True
        )
        base = [x for x in want[(order, "main")] if x != "kcp_transfer_walk"]
        assert log == [("merge:adj_c1" if x == "merge:adj_b1" else x) for x in base], (
            order,
            log,
        )
        assert scratch["bwd_k3_stream"] == "main"
        got = _bwd_sequence_first_main(record(order, "main", log))
        d_ref = got if d_ref is None else d_ref
        assert got == d_ref and "kcp_transfer_walk" not in got, (order, got)
    # an error between the fork and the join: the call's stream still waits for the join
    log, _ = _driver_log(
        monkeypatch, tb, ctx, order="first", k3="side", transfer=False, fail_at="dav"
    )
    assert log == head + [
        "fork",
        "kcp_transfer_walk",
        "join:record",
        "kcp_fwdh_tape",
    ] + [
        "join:wait",
        "raised",
    ], log


def test_bwd_order_gpu_rows_reach_both_knobs() -> None:
    """The GPU rows (``BWD_ORDER_ROWS``) admit both knobs somewhere and refuse the tape order on the H12 row (where
    ``kcp_dav`` runs): on every SM count
    of every target architecture some rank with tokens of an ``H >= 16`` row plans ``P = 1`` with a non-empty K3 list,
    some rank of the H64 rows has two streams by the ``walk_streams`` rule on its K3 list, and the H12 row has a rank
    with tokens (its plan's fused gradient reads ``kcp_dav``'s dAqk: the poison's negative control).  CPU only: the context's plan reads no
    device."""
    for arch, sms in cake_kcp_plan.QK_L2NORM_SMS.items():
        for sm in sms:
            p1, rule_two, h12 = [], [], []
            for row in BWD_ORDER_ROWS:
                lengths, H, W, layout = row[:4]
                part, _ = _partition(lengths, W, layout)
                for rank in range(W):
                    ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
                        part, None, rank, "cpu", 4, sm_count=sm
                    )
                    if int(ctx.T_r) == 0:
                        continue
                    if H < 16:
                        h12.append((_mode_row_id(row), rank))
                        continue
                    if ctx.heads(H).P != 1:
                        continue
                    tb = cake_kcp._Tables(ctx, H)
                    if not len(tb.walk_m_b1):
                        continue
                    p1.append((_mode_row_id(row), rank))
                    seg = [int(x) for x in tb.hp.seg_chunk_start]
                    chains = [seg[s + 1] - seg[s] for s in tb.walk_m_b1]
                    if H == 64 and cake_kcp.walk_streams(chains, H, sm) == "two":
                        rule_two.append((_mode_row_id(row), rank))
            assert p1 and rule_two and h12, (arch, sm, p1, rule_two, h12)


# --------------------------------------------------------------------------------------------- GPU rows


def _seq_index(seq: list, name: str, start: int = 0) -> int:
    return seq.index(name, start) if name in seq[start:] else -1


def _collectives(seq: list) -> list:
    return [x for x in seq if x.startswith(("a2a:", "ring:"))]


def _run_rank_order(
    rank: int, W: int, row, rdzv: str | None, result_dir: str | None, local_id: int
) -> dict:
    """One rank of a GPU row (see :func:`test_bwd_order_gpu_rows`)."""
    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout, gate, l2n = row
    raw = gate == "raw"
    device = torch.device("cuda", local_id % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        _init_group(rank, W, device, rdzv)
    prev_knobs = cake_kcp_plan.enable_test_knobs(True)
    try:
        part, cu = _partition(lengths, W, layout)
        ctx = cake_kcp_plan.cake_kda_nondet_train_kcp_build_context(
            part, dist.group.WORLD if W > 1 else None, rank, device
        )
        glob = (
            _mode_inputs(lengths, H, device, gate, l2n)
            if (raw or l2n)
            else make_inputs(lengths, H, device)
        )
        names = ("q", "k", "v", "g", "beta", "dout")
        ops = _local_slices(part, rank, cu, {n: glob[n] for n in names})
        q, k, v, g, beta, dout = (ops[n] for n in names)
        params = (glob["A_log"], glob["dt_bias"]) if raw else ()
        T_r = int(q.shape[0])
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        arch = cake_jit.device_arch(index)
        sm = int(cake_jit.sm_count(index))
        P = int(ctx.heads(H).P)
        record, dav_live = None, False
        if T_r > 0:
            plan = cake_kcp.cake_backend.plan(
                T_r,
                int(ctx.n_frag),
                H,
                sm,
                arch,
                gate=gate,
                **cake_kcp_plan.qk_l2norm_plan_kwargs("fused" if l2n else "off"),
            )
            record = plan.record("dqkg", cake_kcp.LOCAL_PATH_BWD)
            dav_live = cake_kcp_plan.kcp_dav_needed(plan)
        tb = cake_kcp._Tables(ctx, H)
        seg = [int(x) for x in tb.hp.seg_chunk_start]
        chains_m = [seg[s + 1] - seg[s] for s in tb.walk_m_b1] if P == 1 else []
        info: dict = {
            "T_r": T_r,
            "P": P,
            "n_frag": int(ctx.n_frag),
            "arch": arch,
            "sm": sm,
            "record": record,
            "dav_live": dav_live,
            "k3_chains": chains_m,
            "torch": torch.__version__,
        }
        flags: dict[str, bool] = {}
        engaged = {"tape_b1": 0, "k3_side": 0}

        def leaves():
            return [
                x.detach().clone().requires_grad_(True)
                for x in (q, k, v, g, beta, *params)
            ]

        def gate_kw(lv) -> dict:
            if not raw:
                return {}
            return {
                "use_gate_in_kernel": True,
                "A_log": lv[5],
                "dt_bias": lv[6],
                "lower_bound": LOWER_BOUND,
            }

        def step(mode: str, knobs: dict, poison: bool = False):
            """One autograd step of the node form ``mode`` under ``knobs``: outputs, gradients and the backward's
            LAST_CALL record.  ``poison``: the dAqk rows NaN-filled right after the ``dav`` stage (stream-ordered;
            ``kcp_dav`` launches there only where the plan dispatches dav)."""
            save_transfer, handle, handoff, rows = MODES[mode]
            lv = leaves()
            kw = {
                "use_qk_l2norm_in_kernel": l2n,
                "checkpoint_handoff": handoff,
                "saved_rows": rows,
                "_kcp_test_knobs": dict(knobs),
            }
            if save_transfer is not None:
                kw["save_transfer"] = save_transfer
            orig = cake_kcp.KernelBackend.stage

            def poisoned(self, name, call, **extra):
                orig(self, name, call, **extra)
                if name == "dav":
                    views = call.scratch["views"]
                    t = getattr(views, "dAqk_sink", None)
                    t = call.local_values["dAqk"] if t is None else t
                    t.fill_(float("nan"))

            if poison:
                cake_kcp.KernelBackend.stage = poisoned
            try:
                cake_kcp.LAST_CALL.pop("bwd", None)
                if handle:
                    h = cake_kcp.CakeKDANondetKcpRecomputeHandle()

                    def layer(*a):
                        return cake_kcp.cake_kda_nondet_kcp(
                            *a[:5],
                            ctx,
                            **gate_kw(a),
                            **kw,
                            recompute_handle=a[-1],
                            checkpoint_keep="states",
                        )

                    out = checkpoint(layer, *lv, h, use_reentrant=False)
                else:
                    out = cake_kcp.cake_kda_nondet_kcp(
                        *lv[:5], ctx, **gate_kw(lv), **kw
                    )
                out.backward(dout)
                torch.cuda.synchronize(device)
            finally:
                cake_kcp.KernelBackend.stage = orig
            res = {"out": out.detach()}
            for n, x in zip(
                ("q", "k", "v", "g", "beta", "A_log", "dt_bias")[: len(lv)],
                lv,
                strict=True,
            ):
                res["d" + n] = x.grad
            return res, dict(cake_kcp.LAST_CALL.get("bwd") or {})

        k3_first_launch = []

        def check_record(mode: str, arm: str, rec: dict, ref_rec: dict) -> bool:
            """The recorded order / stream and refusal as this rank's facts decide (a knob the arm leaves out: its
            shipped default), and the launches: exactly the forced ``"first"`` / ``"main"`` run's (``ref_rec``) once
            :func:`_bwd_sequence_first_main` puts the moved ones back, each moved one where its knob puts it."""
            knobs = {
                "bwd_tape_order": cake_kcp.BWD_TAPE_ORDER_DEFAULT,
                "bwd_k3_stream": cake_kcp.BWD_K3_STREAM_DEFAULT,
                **ARMS[arm],
            }
            seq, ref_seq = (
                list(rec.get("sequence") or []),
                list(ref_rec.get("sequence") or []),
            )
            ok = _bwd_sequence_first_main(rec) == ref_seq
            ok = ok and _collectives(seq) == _collectives(ref_seq)
            bound = rec.get("checkpoint_handoff") == "bound"
            # bwd_tape_order (mirrors the rule's order for the cases a row can reach)
            if knobs.get("bwd_tape_order") == "under_b1":
                if bound:
                    why = "handoff"
                elif P > 1:
                    why = "segments"
                elif T_r == 0:
                    why = "no_local"
                else:
                    why = "dav" if dav_live else None
            else:
                why = None
            order = (
                "under_b1"
                if knobs.get("bwd_tape_order") == "under_b1" and why is None
                else "first"
            )
            ok = ok and rec.get("bwd_tape_order") == order
            ok = ok and rec.get("bwd_tape_order_refusal") == why
            b1 = _seq_index(seq, "a2a:b1")
            tape = _seq_index(seq, "kcp_fwdh_tape")
            if order == "under_b1":
                engaged["tape_b1"] += 1
                # between the b1 issue and the adjoint merge (the first launch after b1's wait), before b2
                merge = _seq_index(seq, "kcp_affine_merge", max(b1, 0))
                ok = ok and 0 <= b1 < tape < _seq_index(seq, "a2a:b2")
                ok = ok and (merge < 0 or tape < merge)
            elif tape >= 0 and P == 1:
                ok = ok and tape < _seq_index(seq, "dav") < b1
            # bwd_k3_stream
            stream, why3 = "main", None
            if knobs.get("bwd_k3_stream") == "side":
                if mode not in DM_MODES:
                    why3 = "form"
                elif P > 1:
                    why3 = "segments"
                elif not chains_m:
                    why3 = "no_walk"
                elif (
                    cake_kcp.walk_streams(
                        chains_m, H, sm, knob=knobs.get("walk_streams", "auto")
                    )
                    != "two"
                ):
                    why3 = "rule"
                elif (
                    rec.get("bwd_k3_stream_refusal") == "first_launch"
                    and not k3_first_launch
                ):
                    why3 = "first_launch"  # the table set's first K3 over its list on this device: once per rank
                    k3_first_launch.append((mode, arm))
                else:
                    stream = "side"
            ok = ok and rec.get("bwd_k3_stream") == stream
            ok = ok and rec.get("bwd_k3_stream_refusal") == why3
            k3 = _seq_index(seq, "kcp_transfer_walk")
            if stream == "side":
                engaged["k3_side"] += 1
                ok = ok and 0 <= k3 < _seq_index(seq, "dav")
            elif k3 >= 0 and P == 1:
                ok = ok and _seq_index(seq, "kcp_dhu_offset") < k3 < b1
            return ok

        # every node form: each knob pair twice (the shipped defaults among them), bitwise the forced "first" /
        # "main" form's first run, records as the facts decide
        bitwise_fail, record_fail = [], []
        finite = True
        for mode in MODES:
            ref, ref_rec = step(mode, ARMS["first_main"])
            finite = finite and all(
                bool(torch.isfinite(t.float()).all())
                for t in ref.values()
                if t is not None
            )
            if not check_record(mode, "first_main", ref_rec, ref_rec):
                record_fail.append((mode, "first_main", 0))
            for arm in ARMS:
                for r in range(2):
                    if arm == "first_main" and r == 0:
                        continue
                    res, rec = step(mode, ARMS[arm])
                    if not _same(res, ref):
                        bitwise_fail.append((mode, arm, r))
                    if not check_record(mode, arm, rec, ref_rec):
                        record_fail.append(
                            (
                                mode,
                                arm,
                                r,
                                rec.get("bwd_tape_order"),
                                rec.get("bwd_tape_order_refusal"),
                                rec.get("bwd_k3_stream"),
                                rec.get("bwd_k3_stream_refusal"),
                            )
                        )
            del ref
        flags["finite"] = finite
        flags["bitwise"] = not bitwise_fail
        flags["records"] = not record_fail
        info["bitwise_fail"] = bitwise_fail
        info["record_fail"] = record_fail
        info["engaged"] = dict(engaged)
        info["k3_first_launch"] = k3_first_launch
        # the dAqk rows NaN-filled after the dav stage: where the backward skips kcp_dav no bit moves (both orders:
        # "first" forced, and the shipped "under_b1"); where kcp_dav runs (a rank with tokens of the H12 row) it reaches
        # the gradients (the fused gradient reads its dAqk)
        ref, _ = step("plain", ARMS["first_main"])
        poison = {}
        for arm in ("first_main", "base"):
            res, _ = step("plain", ARMS[arm], poison=True)
            poison[arm] = _same(res, ref)
        info["poison_same"] = poison
        if T_r > 0 and not dav_live:
            flags["poison_covered"] = all(poison.values())
        info["poison_uncovered_reached"] = bool(
            T_r > 0 and dav_live and not poison["first_main"]
        )
        # graph replay with both knobs (activated gate, no norm): eager, capture, replays bitwise the eager forced
        # "first" / "main" form
        if not raw and not l2n:

            def saved_pass(knobs: dict, **opts):
                out, saved = cake_kcp.cake_kda_nondet_train_kcp_forward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    save_transfer=False,
                    _kcp_test_knobs=dict(knobs),
                    **opts,
                )
                grads = cake_kcp.cake_kda_nondet_train_kcp_backward(
                    q,
                    k,
                    v,
                    g,
                    beta,
                    ctx,
                    dout,
                    saved=saved,
                    save_transfer=False,
                    _kcp_test_knobs=dict(knobs),
                    **opts,
                )
                return {"out": out, **{f"t{i}": t for i, t in enumerate(grads)}}

            ws0 = cake_kcp.CakeKDANondetKcpWorkspace()
            eager = {
                n: t.clone()
                for n, t in saved_pass(ARMS["first_main"], workspace=ws0).items()
            }
            ws1 = cake_kcp.CakeKDANondetKcpWorkspace()
            cake_kcp.graph_cache_clear()
            events, graph_same, streams = [], True, []
            try:
                # two more rounds than GRAPH_EVENTS: a kind may run eagerly more than once before its capture
                for _ in range(len(GRAPH_EVENTS) + 2):
                    res = {
                        n: t.clone()
                        for n, t in saved_pass(
                            ARMS["both_two"], workspace=ws1, graph="replay"
                        ).items()
                    }
                    graph_same = graph_same and _same(res, eager)
                    events.append(
                        [
                            str(cake_kcp.LAST_CALL["fwd"].get("graph_event")),
                            str(cake_kcp.LAST_CALL["bwd"].get("graph_event")),
                        ]
                    )
                    streams.append(cake_kcp.LAST_CALL["bwd"].get("bwd_k3_stream"))
                torch.cuda.synchronize(device)
            except BaseException as exc:
                traceback.clear_frames(exc.__traceback__)
                raise
            finally:
                cake_kcp.graph_cache_clear()
            flags["graph_replay"] = (
                graph_same
                and all(_eager_capture_replay([e[i] for e in events]) for i in (0, 1))
                and events[-1] == ["replay", "replay"]
            )
            info["graph_events"] = events
            info["graph_k3_streams"] = streams
            del ws0, ws1, eager
        gathered = [None] * W
        mine = (rank, flags, info)
        if W > 1:
            dist.all_gather_object(gathered, mine)
        else:
            gathered = [mine]
    finally:
        cake_kcp_plan.enable_test_knobs(prev_knobs)
    report = {
        "row": _mode_row_id(row),
        "flags": {
            n: all(f.get(n, True) for _, f, _ in gathered)
            for _, f, _ in gathered
            for n in f
        },
        "info_by_rank": [i for _, _, i in gathered],
    }
    expect_engaged = H >= 16 and any(
        i["P"] == 1 and i["k3_chains"] for _, _, i in gathered
    )
    report["tape_b1_engaged"] = sum(i["engaged"]["tape_b1"] for _, _, i in gathered)
    report["k3_side_engaged"] = sum(i["engaged"]["k3_side"] for _, _, i in gathered)
    report["flags"]["knobs_engaged"] = (not expect_engaged) or (
        report["tape_b1_engaged"] > 0 and report["k3_side_engaged"] > 0
    )
    if 9 <= H <= 15:
        # the negative control: the poison reaches the gradients on some rank with tokens (kcp_dav runs there)
        report["flags"]["poison_control"] = any(
            i["poison_uncovered_reached"] for _, _, i in gathered
        )
    report["pass"] = all(report["flags"].values())
    if W > 1:
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "bwd-order-" + _mode_row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(report, fh, indent=1, default=str)
    return report


def _spawn_entry_order(
    rank: int, W: int, row, rdzv: str | None, result_dir: str
) -> None:
    try:
        _bound_rank_threads(W)
        report = _run_rank_order(rank, W, row, rdzv, result_dir, rank)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.parametrize("row", BWD_ORDER_ROWS, ids=_mode_row_id)
def test_bwd_order_gpu_rows(row) -> None:
    """Every knob pair of ``ARMS`` (the shipped defaults, ``"base"``, among them) against the forced ``"first"`` /
    ``"main"`` form (``ARMS["first_main"]``) on every rank of the row, in every node form of ``MODES`` (plain,
    ``save_transfer=False``, a recompute handle with the hand-off off and ``"auto"``, ``saved_rows="wy"`` plain and
    under the handle), each pair twice: ``out`` and every gradient (raw gate mode: of the raw gate, ``A_log`` and
    ``dt_bias``) bitwise the forced form's first run and finite; each backward's ``bwd_tape_order`` /
    ``bwd_k3_stream`` record and refusal as the rank's facts decide (hand-off bound, ``P``, tokens, whether the plan
    dispatches its dav stage (``kcp_dav_needed``), the dm form, the K3 list and the ``walk_streams`` rule; the table
    set's first K3 over its list once on the call's stream); the backward's launches the forced form's in another
    order -- the K1 tape replay between the b1 issue and the adjoint merge under ``"under_b1"``, K3 before ``kcp_dav``
    on the side stream; :func:`_bwd_sequence_first_main` gives back the forced form's sequence exactly -- and its
    collectives in the same order; both knobs engaged somewhere on the rows that can (``H >= 16``, ``P = 1``, a K3
    list).  The dAqk rows NaN-filled after the ``dav`` stage: bitwise the unpoisoned run where the backward skips
    ``kcp_dav`` in both orders, and on the H12 row (``kcp_dav`` runs) the poison reaches the gradients on some rank.
    Activated gate without the norm: ``graph="replay"`` under both knobs (forced two streams) over four functional
    steps (eager, capture, replays) bitwise the eager forced form."""
    W = row[2]
    ntasks = int(os.environ.get("SLURM_NTASKS", "1") or 1)
    if os.environ.get("SLURM_PROCID") is not None and ntasks > 1:
        if ntasks != W:
            pytest.skip(f"a {ntasks}-task Slurm step runs the W={ntasks} rows only")
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank_order(
            int(os.environ["SLURM_PROCID"]),
            W,
            row,
            None,
            os.environ.get("KCP_TEST_RESULT_DIR"),
            int(os.environ.get("SLURM_LOCALID", "0")),
        )
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_bwd_order_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        _start_ranks(_spawn_entry_order, W, row, tmp, result_dir)
        with open(
            os.path.join(result_dir, "bwd-order-" + _mode_row_id(row) + ".json")
        ) as fh:
            report = json.load(fh)
    assert report["pass"], report
