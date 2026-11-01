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

Checkpoint hand-off suite of the Cake non-deterministic packed KDA chunked training package
(``flashinfer.experimental.cake.cake_kda_train``: ``chunk_kda(..., checkpoint_handoff="auto" | "on" | "off")``,
every call with ``deterministic=False``).  On a GPU: the gradients under both checkpoint modes are bitwise those of
``"off"`` on every plan family of the device the packs below reach, with both kinds of record (a tape record: the
tapes; a row record: the recompute launches the intra twin, ``cake_jit.HANDOFF_TWINS``, and the record also keeps the
prep's rows), also with the tape WY twin (``cake_jit.TAPE_TWINS``) on the rows whose record keeps no ``w``, the
record is taken, stale or refused records fall back, the byte budget demotes and refuses, and ``"auto"`` engages
exactly where ``HANDOFF_HEADS``, ``HANDOFF_VERIFIED`` and ``HANDOFF_ROWS_VERIFIED`` say.  Host only: the rules, the
admission, the mode parsing and torch's recompute, capture, compile and dispatch-mode state as the hand-off reads
it.

Like the backend suite it names its package through the shared alias helper and imports nothing else from
flashinfer.
"""

import contextlib
import dataclasses
import json
import os
import re
import sys
import types
from pathlib import Path

import pytest
import torch
from torch.utils.checkpoint import checkpoint

from tests.test_helpers.cake_kda_det_train_reference import make_inputs
from tests.test_helpers.cake_package import cake_package

cake_backend = cake_package("cake.cake_kda_train", "cake_backend")
cake_build = cake_package("cake.cake_kda_train", "cake_build")
cake_jit = cake_package("cake.cake_kda_train", "cake_jit")
PACKAGE_DIR = Path(cake_jit.__file__).resolve().parent
KdaWorkspace = cake_backend.KdaWorkspace
D, SCALE = cake_backend.D, cake_backend.SCALE
ARCHES = tuple(cake_jit.ARCHES)
ENV = cake_backend.HANDOFF_ENV

LEAVES = ("q", "k", "v", "g", "beta")
# the gradient of each leaf, in LEAVES order
GRAD_OF_LEAF = ("dq", "dk", "dv", "dg", "dbeta")
TAIL_STAGES = ("dqkg", "glm_intra", "gate_epilogue")

# A four-document pack of 261k tokens (P8) and its head counts.
P8 = (23496, 88510, 114974, 34413)
P8_HEADS = (1, 2, 4, 8, 16, 32)
# P8 at 64 heads: opt-in, one checkpoint mode per process (run each id in its own pytest process, for memory).
P8_H64_ENV = "CAKE_KDA_HANDOFF_TEST_P8_H64"
# Short documents, unaligned tails, zero-length documents and pad chunks on both routes: the overlap pair's rows
# (148 SMs), the separate dav program (12 heads), the 16-head rows whose light set carries w / u (sm_100a paired
# serial rows; sm_103a / sm_107a above 32 chains), the fused serial forward (32 heads: no forward w on sm_100a /
# sm_103a) and the 64-column fused recompute (64 heads); the rows where 148- / 152-SM sm_100a runs the ``wy_fwd``
# record ((65,) at 4 heads, 256 documents at 16 heads); more than 256 documents (the table / seg_table launches and
# the intra program past its 256-document form) on the segmented route (2 heads), at 16 heads (serial on sm_100a,
# segmented on sm_103a / sm_107a) and on the serial route (32 heads).  The family sweep below adds a row of every
# other plan family the device's plans reach.
SHORT_ROWS = [
    ((1,), 1),
    ((63,), 2),
    ((65,), 4),
    ((1000,), 1),
    ((63, 1, 65), 2),
    ((100, 0, 300), 2),
    ((100, 640, 77, 1000, 129), 2),
    ((63, 1, 65, 30), 2),
    ((20, 20, 20, 20, 20, 20, 20), 8),
    ((300, 77), 12),
    ((129,), 16),
    ((129, 300, 70), 16),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    (tuple(range(1, 257)), 16),
    ((65,) * 257, 2),
    ((65,) * 257, 16),
    ((65,) * 257, 32),
]
# The launch-sequence rows: both routes, the overlap pair, w from the light set, no forward w, the fused recompute
# and a pack that saves the segment states (light_states: the record keeps the set's transfers).
LAUNCH_ROWS = [
    ((100, 640, 77, 1000, 129), 2),
    ((63, 1, 65, 30), 2),
    ((129, 300, 70), 16),
    ((65, 200), 32),
    ((63, 1, 65), 64),
    (P8, 4),
]
LAYER_ROWS = [((100, 640, 77, 1000, 129), 2), ((129, 63), 16), ((65, 200), 32)]
# The rows whose record may keep no forward w (its tape backward launches wy): 32 and 64 heads, P8 at 32 heads among
# them (P8 at 64 heads: opt-in, below)
TAPE_WY_ROWS = [(P8, 32), ((65, 200), 32), ((63, 1, 65), 64), ((65,) * 257, 32)]
# The rows whose record may keep the forward's w, with both forward WY records (``wy_fwd``, ``wy_fwd_rows``)
WY_ROWS = [
    ((65,), 4),
    ((63, 1, 65, 30), 2),
    ((100, 640, 77, 1000, 129), 2),
    ((129, 300, 70), 16),
    (tuple(range(1, 257)), 16),
    ((65,) * 257, 2),
    (P8, 1),
    (P8, 4),
]
# The family sweep: candidate packs across the document-count, chain and segment thresholds, at one head and at
# every head count "auto" engages at (HANDOFF_HEADS); the cheapest candidate (tokens x heads) of each family runs,
# and every family must have a candidate up to P8 at 32 heads (the records read the document count, the heads, the
# SM count and the segments; the token count enters only through the segmented route's saved-states rule).
FAMILY_PACKS = [
    (65,),
    (1000,),
    (63, 1, 65),
    (63, 1, 65, 30),
    (100, 640, 77, 1000, 129),
    (20,) * 7,
    (129, 300, 70),
    (65,) * 33,
    (65,) * 64,
    (300,) * 100,
    tuple(range(1, 257)),
    (65,) * 257,
    (129,) * 300,
    (4000,) * 16,
    (2048,) * 64,
    P8,
]
FAMILY_HEADS = (1, *sorted(cake_backend.HANDOFF_HEADS))
FAMILY_CAP = sum(P8) * 32
# The intra twins (intra record -> twin) and the bytes per internal token-head a row record adds: gk (FP32) + kg
# (BF16) + beta_int (FP32), and kb (BF16) where the record keeps no w; the gk and beta_int arena slots its backward
# layout drops.
TWINS = dict(cake_jit.HANDOFF_TWINS)
ROW_BYTES = D * 4 + D * 2 + 4
ROW_KB_BYTES = D * 2
ROW_ARENA_SAVING = D * 4 + 4


def _shape_id(shape) -> str:
    lengths, heads = shape
    if tuple(lengths) == P8:
        return f"P8-h{heads}"
    if len(lengths) > 8:
        return f"n{len(lengths)}-S{sum(lengths)}-h{heads}"
    return "s" + "_".join(str(n) for n in lengths) + f"-h{heads}"


def _device_or_skip() -> torch.device:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    index = torch.cuda.current_device()
    capability = tuple(torch.cuda.get_device_capability(index))
    if cake_jit.arch_for_capability(capability) is None:
        pytest.skip(
            f"compute capability {capability} is not an exported target {ARCHES}"
        )
    return torch.device("cuda", index)


def _manifest_or_skip() -> dict:
    path = PACKAGE_DIR / "build.json"
    if not path.exists():
        pytest.skip("the package has no build.json yet (not exported)")
    return json.loads(path.read_text())


@pytest.fixture(scope="module")
def device() -> torch.device:
    return _device_or_skip()


@pytest.fixture(scope="module")
def manifest() -> dict:
    return _manifest_or_skip()


@pytest.fixture(params=("tape", "rows"), ids=("kind_tape", "kind_rows"))
def record_kind(request, monkeypatch) -> str:
    """The kind of record the test's hand-offs publish, forced through ``_HANDOFF_ROWS`` (``"rows"`` skips the
    device, workspace and build checks of the rule; the byte budget still applies)."""
    monkeypatch.setattr(cake_backend, "_HANDOFF_ROWS", request.param == "rows")
    return request.param


def _with_rows(p, record_kind: str | None) -> bool:
    """Whether a record of plan ``p`` of the forced kind ``record_kind`` is a row record."""
    return record_kind == "rows" and p.handoff_twin is not None


@pytest.fixture(params=(False, True), ids=("tape_wy_record", "tape_wy_twin"))
def tape_wy(request, monkeypatch) -> bool:
    """Whether the tape backward launches the w-only twin of its WY record (``cake_jit.TAPE_TWINS``), forced
    through ``_TAPE_WY`` (the device, build and load checks of the rule skipped)."""
    monkeypatch.setattr(cake_backend, "_TAPE_WY", request.param)
    return request.param


def _tape_wy_runs(monkeypatch) -> list:
    """Record the ``record`` override of every ``wy`` launch on the tape path through ``KdaWorkspace.run`` (``None``:
    the plan's WY record)."""
    runs: list = []
    real_run = KdaWorkspace.run

    def run(self, stage, values, *, path, **kwargs):
        if stage == "wy" and path == "tape":
            runs.append(kwargs.get("record"))
        return real_run(self, stage, values, path=path, **kwargs)

    monkeypatch.setattr(KdaWorkspace, "run", run)
    return runs


@pytest.fixture(autouse=True)
def _fresh_registry(monkeypatch):
    """Every test starts and ends with an empty hand-off registry, fresh counters and no environment override."""
    monkeypatch.delenv(ENV, raising=False)
    cake_backend._handoff_reset()
    yield
    cake_backend._handoff_reset()


def chunk_kda(*args, **kwargs):
    """The package's autograd entry with the mode it requires."""
    return cake_backend.chunk_kda(*args, deterministic=False, **kwargs)


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bit-pattern equality: ``torch.equal`` on the integer views, so signed zeros and NaN payloads count."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    view = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[
        a.element_size()
    ]
    return torch.equal(a.contiguous().view(view), b.contiguous().view(view))


def _fill_nan(t: torch.Tensor) -> torch.Tensor:
    """NaN in every element of a floating-point tensor (one bit pattern per dtype): an element no launch writes
    stays recognizable, where a zero fill would equal a written zero."""
    if t.is_floating_point():
        t.fill_(float("nan"))
    return t


def _nan_alloc(shape_, dtype, dev) -> torch.Tensor:
    """``_alloc`` with floating-point tensors NaN-filled; integer tensors (the tables, offsets) zero-filled as
    before, so a stray index read cannot fault."""
    if dtype.is_floating_point:
        return torch.full(shape_, float("nan"), dtype=dtype, device=dev)
    return torch.zeros(shape_, dtype=dtype, device=dev)


@contextlib.contextmanager
def _poisoned():
    """The record's per-call tensors NaN-filled before the recompute's launches (``_HANDOFF_POISON``)."""
    old = cake_backend._HANDOFF_POISON
    cake_backend._HANDOFF_POISON = True
    try:
        yield
    finally:
        cake_backend._HANDOFF_POISON = old


def _stats(device: torch.device) -> dict:
    return cake_backend._handoff_stats(device.index)


def _plan(lengths, heads, device):
    return cake_backend.plan(
        sum(lengths),
        len(lengths),
        heads,
        cake_jit.sm_count(device.index),
        cake_jit.device_arch(device.index),
    )


def _family(p) -> tuple:
    """The plan family of a row: the route, the light path, the head count and the records of the saving forward,
    the tape backward and the light backward (the SM count enters through the records it selects)."""
    records = tuple(
        (path, stage, name)
        for path in ("fwd", "tape", p.light_path)
        for stage, name in sorted(p.records[path].items())
        if name is not None
    )
    return (p.recur, p.light_path, p.H, records)


def _leaves(inputs: dict) -> list:
    return [inputs[n].detach().clone().requires_grad_(True) for n in LEAVES]


def _grads(leaves) -> dict:
    return {name: leaf.grad for name, leaf in zip(GRAD_OF_LEAF, leaves, strict=True)}


def _checkpointed_step(inputs: dict, *, reentrant: bool, handoff, workspace=None):
    """One forward of ``chunk_kda`` under ``torch.utils.checkpoint`` and its backward on fresh leaves: ``out``
    (detached) and the leaves' gradients by name.  ``handoff=None`` passes no ``checkpoint_handoff`` (the
    default)."""
    leaves = _leaves(inputs)
    cu = inputs["cu_seqlens"]
    kw = {} if handoff is None else {"checkpoint_handoff": handoff}

    def layer(q, k, v, g, beta):
        return chunk_kda(q, k, v, g, beta, cu, workspace=workspace, **kw)

    out = checkpoint(layer, *leaves, use_reentrant=reentrant)
    out.backward(inputs["dout"])
    return out.detach(), _grads(leaves)


def _check_grads(got: dict, want: dict, tag) -> None:
    for name in want:
        assert _bits_equal(got[name], want[name]), (tag, name)


def _check_handoff_against_off(
    inputs,
    device,
    lengths,
    heads,
    *,
    reentrant: bool,
    record_kind: str | None = None,
    host_reference: bool = False,
) -> None:
    """``"off"`` then ``"on"`` (the record's per-call tensors and the backward's ``dv2`` NaN-filled before the
    launches that write them) under one checkpoint mode: the off arm publishes nothing, the on arm publishes one
    record (of ``record_kind`` where the test forces one) and its backward takes it, the registry is empty after each
    backward, and ``out`` and the five gradients are bitwise equal.  ``host_reference`` keeps the off arm's ``out``
    and gradients on the host while the on arm runs (two sets of 64-head gradients do not fit a 180 GB part)."""
    tag = (
        _shape_id((lengths, heads)),
        "reentrant" if reentrant else "nonreentrant",
        f"kind={record_kind}",
    )
    cake_backend._handoff_reset()
    off_out, off = _checkpointed_step(inputs, reentrant=reentrant, handoff="off")
    if host_reference:
        off_out, off = off_out.cpu(), {name: g.cpu() for name, g in off.items()}
        torch.cuda.empty_cache()
    s = _stats(device)
    assert s["published"] == 0 and s["live_records"] == 0, (tag, "off", s)
    with _poisoned():
        on_out, on = _checkpointed_step(inputs, reentrant=reentrant, handoff="on")
    if host_reference:
        on_out, on = on_out.cpu(), {name: g.cpu() for name, g in on.items()}
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 1 and s["misses"] == 0, (tag, s)
    assert s["live_records"] == 0 and s["live_bytes"] == 0, (tag, s)
    if record_kind is not None:
        with_rows = _with_rows(_plan(lengths, heads, device), record_kind)
        assert s["published_rows"] == int(with_rows) and s["demoted"] == 0, (tag, s)
    assert _bits_equal(on_out, off_out), (tag, "out")
    _check_grads(on, off, tag)


# --------------------------------------------------------------------------- bitwise gate


@pytest.mark.parametrize("heads", P8_HEADS, ids=lambda h: f"P8-h{h}")
def test_handoff_gradients_are_bitwise_off_on_the_p8_pack(
    device, manifest, heads, record_kind
) -> None:
    """The P8 pack with both kinds of record: under non-reentrant and reentrant checkpointing the hand-off's gradients
    are bitwise the light path's (``checkpoint_handoff="off"``), and the record was taken."""
    inputs = make_inputs(P8, heads, device)
    for reentrant in (False, True):
        _check_handoff_against_off(
            inputs, device, P8, heads, reentrant=reentrant, record_kind=record_kind
        )


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_handoff_gradients_are_bitwise_off_on_the_p8_pack_at_64_heads(
    device, manifest, reentrant, record_kind
) -> None:
    """P8 at 64 heads: the same gate, one (checkpoint mode, record kind) per process (each id in its own pytest process,
    for memory).  Opt-in by ``CAKE_KDA_HANDOFF_TEST_P8_H64=1``.  It also measures tape == light at 64 heads."""
    if os.environ.get(P8_H64_ENV) != "1":
        pytest.skip(f"opt-in: {P8_H64_ENV}=1, one id per process")
    inputs = make_inputs(P8, 64, device)
    _check_handoff_against_off(
        inputs,
        device,
        P8,
        64,
        reentrant=reentrant,
        record_kind=record_kind,
        host_reference=True,
    )


@pytest.mark.parametrize("shape", SHORT_ROWS, ids=_shape_id)
def test_handoff_gradients_are_bitwise_off_on_short_packs(
    device, manifest, shape, record_kind
) -> None:
    """Short documents, unaligned tails, zero-length documents, pad chunks and more than 256 documents: the same
    gate on both routes (forward w kept or not, w from the light set, the transfers per call, the table launches,
    and with row records both intra twins and the ``dv2`` pad-row fill kernel)."""
    lengths, heads = shape
    inputs = make_inputs(lengths, heads, device)
    for reentrant in (False, True):
        _check_handoff_against_off(
            inputs, device, lengths, heads, reentrant=reentrant, record_kind=record_kind
        )


def test_handoff_is_bitwise_off_on_every_plan_family_of_the_device(
    device, manifest, record_kind
) -> None:
    """The plan families (:func:`_family`) the device's plans reach over ``FAMILY_PACKS`` x ``FAMILY_HEADS``: each
    has a row up to P8 at 32 heads, and the cheapest row of each family passes the gate under both checkpoint
    modes with the test's kind of record.  Every family is checked and every failure reported together."""
    sms, arch = cake_jit.sm_count(device.index), cake_jit.device_arch(device.index)
    families: dict[tuple, tuple[int, tuple[int, ...], int]] = {}
    for lengths in FAMILY_PACKS:
        for heads in FAMILY_HEADS:
            try:
                p = cake_backend.plan(sum(lengths), len(lengths), heads, sms, arch)
            except ValueError:
                continue
            cost = sum(lengths) * heads
            key = _family(p)
            if key not in families or cost < families[key][0]:
                families[key] = (cost, lengths, heads)
    assert families
    too_big = [_shape_id(row[1:]) for row in families.values() if row[0] > FAMILY_CAP]
    assert not too_big, ("families without a row up to the cap", too_big)
    failures = []
    for _cost, lengths, heads in sorted(families.values()):
        inputs = make_inputs(lengths, heads, device)
        for reentrant in (False, True):
            try:
                _check_handoff_against_off(
                    inputs,
                    device,
                    lengths,
                    heads,
                    reentrant=reentrant,
                    record_kind=record_kind,
                )
            except AssertionError as exc:
                failures.append(
                    (_shape_id((lengths, heads)), reentrant, str(exc)[:200])
                )
        del inputs
        torch.cuda.empty_cache()
    assert not failures, (
        f"{len(failures)} of {2 * len(families)} family checks failed",
        failures,
    )


@pytest.mark.parametrize("shape", TAPE_WY_ROWS, ids=_shape_id)
def test_the_tape_wy_twin_hands_off_bitwise_off(
    device, manifest, monkeypatch, shape, record_kind
) -> None:
    """The rows whose record may keep no forward ``w`` (P8 at 32 heads among them) with the tape WY twin forced
    (``_TAPE_WY = True``), with both kinds of record: under both checkpoint modes the gradients are bitwise the light
    path's (``"off"``), and the backward from the record launches the twin in place of its WY record exactly where
    the record keeps no ``w`` (one launch per backward, as the record's)."""
    lengths, heads = shape
    p = _plan(lengths, heads, device)
    launches = p.dispatched("wy", "tape") and not p.handoff_keeps_w
    assert not launches or p.tape_wy_twin is not None, (p.arch, p.record("wy", "tape"))
    monkeypatch.setattr(cake_backend, "_TAPE_WY", True)
    runs = _tape_wy_runs(monkeypatch)
    inputs = make_inputs(lengths, heads, device)
    for reentrant in (False, True):
        runs.clear()
        _check_handoff_against_off(
            inputs, device, lengths, heads, reentrant=reentrant, record_kind=record_kind
        )
        assert runs == ([p.tape_wy_twin] if launches else []), (reentrant, runs)


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_the_tape_wy_twin_hands_off_bitwise_off_on_p8_h64(
    device, manifest, monkeypatch, reentrant, record_kind
) -> None:
    """P8 at 64 heads with the tape WY twin forced: the same gate, one (checkpoint mode, record kind) per process
    (each id in its own pytest process, for memory).  Opt-in by ``CAKE_KDA_HANDOFF_TEST_P8_H64=1``."""
    if os.environ.get(P8_H64_ENV) != "1":
        pytest.skip(f"opt-in: {P8_H64_ENV}=1, one id per process")
    p = _plan(P8, 64, device)
    launches = p.dispatched("wy", "tape") and not p.handoff_keeps_w
    monkeypatch.setattr(cake_backend, "_TAPE_WY", True)
    runs = _tape_wy_runs(monkeypatch)
    inputs = make_inputs(P8, 64, device)
    _check_handoff_against_off(
        inputs,
        device,
        P8,
        64,
        reentrant=reentrant,
        record_kind=record_kind,
        host_reference=True,
    )
    assert runs == ([p.tape_wy_twin] if launches else []), runs


# --------------------------------------------------------------------------- the recompute's writes


@pytest.mark.parametrize("shape", LAUNCH_ROWS, ids=_shape_id)
def test_the_recompute_writes_change_neither_out_nor_the_saved_set(
    device, manifest, monkeypatch, shape, record_kind
) -> None:
    """The forward with a hand-off request (``write_tapes = 1`` on the ``fwd`` records, the extras in per-call
    tensors; for a row record the intra twin) gives bitwise the light forward's ``out`` and saved set; its ``h`` /
    ``v_new`` (and transfers) are the tape forward's; it keeps ``w`` exactly where the plan says; a row record
    keeps ``gk`` / ``kg`` / ``beta_int`` of the plan's shapes and ``kb`` exactly where it keeps no ``w`` (a tape
    record none of them); its bytes are ``Plan.handoff_bytes(rows)`` and ``memory_report``'s.  Every floating-point
    per-call tensor starts NaN-filled (``_alloc``), so an element one forward writes and the other does not differs
    (a zero fill would hide written zeros)."""
    lengths, heads = shape
    S, N = sum(lengths), len(lengths)
    inputs = make_inputs(lengths, heads, device)
    p = _plan(lengths, heads, device)
    with_rows = _with_rows(p, record_kind)
    monkeypatch.setattr(cake_backend, "_alloc", _nan_alloc)
    ops = [inputs[n] for n in (*LEAVES, "cu_seqlens")]
    common = dict(
        scale=SCALE,
        save_for_backward=True,
        workspace=None,
        arena=None,
        doc_lengths=None,
    )
    out, saved = cake_backend._fwd_call(*ops, save_mode=None, **common)
    request = {"task": 0, "index": device.index, "mode": "on"}
    out_h, saved_h = cake_backend._fwd_call(
        *ops, save_mode=None, handoff=request, **common
    )
    assert "tensors" in request, (lengths, heads)
    assert _bits_equal(out_h, out)
    for field in dataclasses.fields(saved):
        a, b = getattr(saved_h, field.name), getattr(saved, field.name)
        if isinstance(b, torch.Tensor):
            assert _bits_equal(a, b), field.name
        elif isinstance(b, tuple):
            assert len(a) == len(b) and all(
                _bits_equal(x, y) for x, y in zip(a, b, strict=True)
            ), field.name
        else:
            assert a == b, field.name
    tensors = request["tensors"]
    _, tape = cake_backend._fwd_call(*ops, save_mode="tape", **common)
    assert _bits_equal(tensors["h"], tape.h)
    assert _bits_equal(tensors["v_new"], tape.v_new)
    if p.recur == "seg":
        assert _bits_equal(tensors["seg_m"], tape.seg_m)
        if p.seg_save:  # the record keeps the light set's own transfers
            assert tensors["seg_m"].data_ptr() == saved_h.seg_state[1].data_ptr()
    else:
        assert tensors["seg_m"] is None
    assert (tensors["w"] is not None) is p.handoff_keeps_w
    if p.light_saves_wu:  # the light set's own w
        assert tensors["w"].data_ptr() == saved_h.w.data_ptr()
    rows = (p.R, heads, D)
    row_names = ("gk", "kg", "beta_int", "kb")
    if with_rows:
        for name, shape_, dtype in (
            ("gk", rows, torch.float32),
            ("kg", rows, torch.bfloat16),
            ("beta_int", (p.R, heads), torch.float32),
            ("kb", rows, torch.bfloat16),
        ):
            if name in tensors:
                t = tensors[name]
                assert tuple(t.shape) == shape_ and t.dtype == dtype, name
        assert {"gk", "kg", "beta_int"} <= set(tensors)
        assert ("kb" in tensors) is not p.handoff_keeps_w
        # the twin writes every gk row, the pad chunks' too (zeros): no NaN of the fill is left
        assert not torch.isnan(tensors["gk"]).any()
        extra = sum(
            tensors[n].numel() * tensors[n].element_size()
            for n in row_names
            if n in tensors
        )
        assert extra == p.handoff_bytes(rows=True) - p.handoff_bytes()
    else:
        assert not set(row_names) & set(tensors), sorted(tensors)
    report = cake_backend.memory_report(S, N, heads, sm_count=p.sm_count, arch=p.arch)
    assert request["nbytes"] == p.handoff_bytes(rows=with_rows)
    assert (
        request["nbytes"]
        == report["handoff_bytes_rows" if with_rows else "handoff_bytes"]
    )
    assert request["key"] == cake_backend._handoff_key(p, SCALE)
    assert _stats(device)["published"] == 0  # filling a request publishes nothing


@pytest.mark.parametrize("shape", WY_ROWS, ids=_shape_id)
def test_the_kept_forward_w_is_the_tape_wy_w(
    device, manifest, monkeypatch, shape, tape_wy
) -> None:
    """Where the record keeps the forward's ``w`` (its backward launches no ``wy``), that ``w`` is bit for bit the
    ``w`` the tape backward's ``wy`` program writes (its WY record, or the record's w-only twin), pad rows included.
    Both start NaN-filled (the forward's through ``_alloc``, the backward's arena slot right before its ``wy``
    launch), so a row one program writes and the other does not differs."""
    lengths, heads = shape
    p = _plan(lengths, heads, device)
    if not p.handoff_keeps_w or not p.dispatched("wy", "tape"):
        pytest.skip(
            f"{p.arch} at {p.sm_count} SMs: the record keeps no forward w on this row (or the tape path runs no wy)"
        )
    inputs = make_inputs(lengths, heads, device)
    monkeypatch.setattr(cake_backend, "_alloc", _nan_alloc)
    ops = [inputs[n] for n in (*LEAVES, "cu_seqlens")]
    common = dict(scale=SCALE, save_for_backward=True, arena=None, doc_lengths=None)
    request = {"task": 0, "index": device.index, "mode": "on"}
    cake_backend._fwd_call(
        *ops, save_mode=None, workspace=None, handoff=request, **common
    )
    w_fwd = request["tensors"]["w"]
    captured: dict = {}

    def launcher(name, values, *, grid, device_index):
        wy = cake_jit.MODULES[name]["stage"] == "wy"
        if wy:
            _fill_nan(values["w"])
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)
        if wy:
            captured["w"], captured["record"] = values["w"].clone(), name

    # the tape set from a per-call forward; its backward on a tape-mode workspace whose launcher captures wy's w
    _, tape = cake_backend._fwd_call(*ops, save_mode="tape", workspace=None, **common)
    ws = KdaWorkspace(
        p,
        device,
        backward=True,
        save_mode="tape",
        launcher=launcher,
        deterministic=False,
    )
    cake_backend._bwd_call(
        *ops,
        inputs["dout"],
        saved=tape,
        scale=SCALE,
        workspace=ws,
        arena=None,
        doc_lengths=None,
    )
    assert "w" in captured, "the tape backward launched no wy"
    want_record = (p.tape_wy_twin if tape_wy else None) or p.record("wy", "tape")
    assert captured["record"] == want_record, (captured["record"], want_record)
    w_bwd = captured["w"]
    wy_fwd = p.record("wy_fwd", "fwd") if p.dispatched("wy_fwd", "fwd") else None
    info = (wy_fwd, captured["record"], tuple(w_fwd.shape), tuple(w_bwd.shape))
    assert w_bwd.numel() == w_fwd.numel(), info
    a = w_fwd.reshape(p.R, heads, D).contiguous().view(torch.int16)
    b = w_bwd.reshape(p.R, heads, D).contiguous().view(torch.int16)
    rows = (a != b).any(dim=2).any(dim=1).nonzero().flatten()
    assert rows.numel() == 0, (
        *info,
        f"{rows.numel()} of {p.R} rows differ",
        rows[:8].tolist(),
    )


# --------------------------------------------------------------------------- launches


def _expected_fwd(p, intra: str | None = None) -> list[str]:
    """The saving forward's launches (the original forward and a recompute with a tape record run the same records;
    ``intra``: the intra record a recompute with a row record launches instead of the plan's, its twin)."""
    seg = p.recur == "seg"
    seq: list[str] = []
    if p.dispatched("table", "fwd"):
        seq.append(p.record("table", "fwd"))
        if seg and p.dispatched("seg_table", "fwd"):
            seq.append(p.record("seg_table", "fwd"))
    seq.append(p.record("fwd_intra", "fwd") if intra is None else intra)
    if p.dispatched("wy_fwd", "fwd"):
        seq.append(p.record("wy_fwd", "fwd"))
    seq.append(p.record("fwdh", "fwd"))
    if seg:
        seq += [p.record("seg_carry", "fwd"), p.record("fwdh_p3", "fwd")]
    return seq


def _expected_bwd(
    p,
    path: str,
    *,
    prep_path=None,
    skip_wy=False,
    prep: bool = True,
    tape_wy: bool = False,
) -> list[str]:
    """A saved-set backward's launches on ``path`` (the set carries its tables; one tail pass); ``prep_path`` and
    ``skip_wy`` are the hand-off's light prep and its ``wy`` skip, ``prep=False`` a row record's backward (no
    prep launch: the record carries its rows), ``tape_wy`` a tape backward that launches the w-only twin of its WY
    record."""
    seg = p.recur == "seg"
    seq = [p.record("bwd_prep", prep_path or path)] if prep else []
    if p.dispatched("wy", path) and not skip_wy:
        twin = p.tape_wy_twin if (tape_wy and path == "tape") else None
        seq.append(twin or p.record("wy", path))
    if p.dispatched("fwdh", path):
        seq.append(p.record("fwdh", path))
        if seg:
            seq.append(p.record("seg_carry", path))
    if p.dispatched("fwdh_p3", path):
        seq.append(p.record("fwdh_p3", path))
    if p.dispatched("dav", path):
        seq.append(p.record("dav", path))
    dhu = p.record("dhu", path)
    seq += [dhu, p.record("seg_carry", path), dhu] if seg else [dhu]
    seq += [p.record(s, path) for s in TAIL_STAGES if p.dispatched(s, path)]
    return seq


@pytest.mark.parametrize("shape", LAUNCH_ROWS, ids=_shape_id)
def test_the_handoff_backward_launches_the_tape_path_with_the_light_prep(
    device, manifest, shape, record_kind, tape_wy
) -> None:
    """On a caller workspace: the original forward launches the saving forward's records, a recompute with a tape
    record too and one with a row record the intra twin in place of the intra record (on the record's grid); the
    backward from a tape record launches the tape path's records with the light path's prep record (no qg; no kb on
    the rows whose light set carries w), from a row record the same without any prep, and no ``wy`` where the
    record carries the forward's ``w`` (elsewhere its WY record, or with ``tape_wy`` the record's w-only twin, one
    launch either way); ``"off"`` launches the light path; the gradients agree bitwise."""
    lengths, heads = shape
    inputs = make_inputs(lengths, heads, device)
    p = _plan(lengths, heads, device)
    with_rows = _with_rows(p, record_kind)
    launched: list[str] = []
    intra_grids: list[tuple] = []

    def recording(name, values, *, grid, device_index):
        launched.append(name)
        if cake_jit.MODULES[name]["stage"] == "fwd_intra":
            intra_grids.append((name, tuple(grid)))
        cake_jit.run_module(name, values, grid=grid, device_index=device_index)

    ws = KdaWorkspace(p, device, backward=True, launcher=recording, deterministic=False)
    cu = inputs["cu_seqlens"]
    fwd = _expected_fwd(p)
    recompute = _expected_fwd(p, intra=p.handoff_twin if with_rows else None)
    handoff_bwd = _expected_bwd(
        p,
        "tape",
        prep_path=p.light_path,
        skip_wy=p.handoff_keeps_w,
        prep=not with_rows,
        tape_wy=tape_wy,
    )
    preps = [n for n in handoff_bwd if cake_jit.MODULES[n]["stage"] == "bwd_prep"]
    assert preps == ([] if with_rows else [p.record("bwd_prep", p.light_path)])
    intra_grid = tuple(p.grid("fwd_intra", "fwd"))
    grads = {}
    for mode, bwd in (
        ("off", _expected_bwd(p, p.light_path)),
        ("on", handoff_bwd),
    ):
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        launched.clear()
        out = checkpoint(
            lambda *a, m=mode: chunk_kda(*a, cu, workspace=ws, checkpoint_handoff=m),
            *leaves,
            use_reentrant=False,
        )
        assert launched == fwd, mode
        # the original forward publishes nothing
        assert _stats(device)["published"] == 0
        launched.clear()
        intra_grids.clear()
        out.backward(inputs["dout"])
        want = (recompute if mode == "on" else fwd) + bwd
        assert launched == want, (mode, launched)
        # the twin runs on the intra record's grid
        assert [g for _, g in intra_grids] == [intra_grid], (mode, intra_grids)
        s = _stats(device)
        assert s["hits"] == (mode == "on") and s["live_records"] == 0, (mode, s)
        assert s["published_rows"] == int(mode == "on" and with_rows), (mode, s)
        grads[mode] = _grads(leaves)
    _check_grads(grads["on"], grads["off"], (lengths, heads))


# --------------------------------------------------------------------------- stale records and misses


class _AfterRecompute(torch.autograd.Function):
    """Identity inside a checkpointed region whose backward first unpacks its saved tensor (non-reentrant
    checkpointing then re-runs the region: the KDA recompute publishes its record) and then calls ``action``, before
    the KDA node's backward runs."""

    @staticmethod
    def forward(ctx, o, action):
        ctx.save_for_backward(o)
        ctx.action = action
        return o.clone()

    @staticmethod
    def backward(ctx, grad):
        ctx.saved_tensors  # noqa: B018 - the unpack re-runs the region
        if ctx.action is not None:
            ctx.action()
        return grad, None


STALE_ROWS = [((100, 640, 77, 1000, 129), 2), ((129, 300, 70), 16), ((65, 200), 32)]


@pytest.mark.parametrize("shape", STALE_ROWS, ids=_shape_id)
def test_a_stale_record_falls_back_and_stays_correct(
    device, manifest, shape, record_kind
) -> None:
    """An operand changed in place between the recompute (which published the record, of the test's kind) and the
    KDA node's backward: ``cu_seqlens`` bumped to a new version with the same values, or the gate ``g`` halved.
    The record's stamp no longer matches, the backward drops it (one miss, an empty registry) and runs the light
    path: the gradients are bitwise those of ``"off"`` under the same change.  Without a change the record is
    taken.  (Non-reentrant only: under reentrant checkpointing torch's own saved-tensor version check raises
    first; the failed-backward test covers that mode.)"""
    lengths, heads = shape
    inputs = make_inputs(lengths, heads, device)
    with_rows = _with_rows(_plan(lengths, heads, device), record_kind)

    def run(mode: str, change: str | None):
        leaves = _leaves(inputs)
        cu = inputs["cu_seqlens"].clone()
        g_leaf = leaves[3]

        def action():
            with torch.no_grad():
                if change == "cu_version":
                    cu.add_(0)
                elif change == "g_values":
                    g_leaf.mul_(0.5)

        def layer(q, k, v, g, beta):
            o = chunk_kda(q, k, v, g, beta, cu, checkpoint_handoff=mode)
            return _AfterRecompute.apply(o, action if change else None)

        out = checkpoint(layer, *leaves, use_reentrant=False)
        out.backward(inputs["dout"])
        return _grads(leaves)

    for change in (None, "cu_version", "g_values"):
        cake_backend._handoff_reset()
        off = run("off", change)
        assert _stats(device)["published"] == 0
        cake_backend._handoff_reset()
        on = run("on", change)
        s = _stats(device)
        assert s["published"] == 1 and s["live_records"] == 0, (change, s)
        assert s["published_rows"] == int(with_rows), (change, s)
        if change is None:
            assert s["hits"] == 1 and s["misses"] == 0, s
        else:
            assert s["hits"] == 0 and s["misses"] == 1, (change, s)
        _check_grads(on, off, (lengths, heads, change))


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_inputs_changed_in_place_before_the_checkpointed_call_still_hand_off(
    device, manifest, reentrant
) -> None:
    """Inputs whose version counters moved before the checkpointed call (``cu_seqlens`` written in place, ``q``
    scaled in place): the take compares versions through the record's own aliases of the recompute's inputs, not
    through the tensors checkpoint unpacks, so the record is taken and the gradients are bitwise ``"off"``'s."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)

    def run(mode: str):
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        cu = inputs["cu_seqlens"].clone()
        with torch.no_grad():
            cu.add_(0)
            leaves[0].mul_(1)
        assert cu._version > 0 and leaves[0]._version > 0
        out = checkpoint(
            lambda *a: chunk_kda(*a, cu, checkpoint_handoff=mode),
            *leaves,
            use_reentrant=reentrant,
        )
        out.backward(inputs["dout"])
        return _grads(leaves), _stats(device)

    off, _ = run("off")
    on, s = run("on")
    assert s["published"] == 1 and s["hits"] == 1 and s["misses"] == 0, s
    _check_grads(on, off, ("versions before the call", reentrant))


def test_a_record_the_backward_refuses_is_released_and_stays_correct(
    device, manifest, monkeypatch, record_kind
) -> None:
    """A record of the test's kind the node takes but the backward refuses (here: a plan key that differs from the
    forward's): one miss, the record's tensors released before the light path allocates its own (``h`` / ``v_new``
    / ``w`` / ``Aqk`` references cleared, and a row record's ``gk`` / ``kg`` / ``beta_int`` / ``kb``), the
    backward on the light path's own workspace (no hand-off layout: the full backward arena), an empty registry and
    the gradients bitwise ``"off"``'s."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    p = _plan(lengths, heads, device)
    _, off = _checkpointed_step(inputs, reentrant=False, handoff="off")
    real_key, real_take = cake_backend._handoff_key, cake_backend._handoff_take
    real_acquire = cake_backend._entry_acquire
    calls: list[int] = []
    taken: list = []
    backward_arenas: list = []

    def acquire(*args, **kwargs):
        ws, key = real_acquire(*args, **kwargs)
        if kwargs.get("backward"):
            backward_arenas.append((kwargs.get("layout"), type(ws).__name__, ws.bytes))
        return ws, key

    def key(p, scale):  # the recompute's key, then another one in the backward
        calls.append(1)
        k = real_key(p, scale)
        return k if len(calls) == 1 else (*k, "mismatch")

    def take(*args):
        rec = real_take(*args)
        taken.append(rec)
        return rec

    monkeypatch.setattr(cake_backend, "_handoff_key", key)
    monkeypatch.setattr(cake_backend, "_handoff_take", take)
    monkeypatch.setattr(cake_backend, "_entry_acquire", acquire)
    cake_backend._handoff_reset()
    _, on = _checkpointed_step(inputs, reentrant=False, handoff="on")
    s = _stats(device)
    assert len(calls) == 2, calls
    assert s["published"] == 1 and s["hits"] == 0 and s["misses"] == 1, s
    assert s["published_rows"] == int(_with_rows(p, record_kind)), s
    assert s["live_records"] == 0, s
    # taken by the node, refused by the backward's acceptance before its workspace
    assert len(taken) == 1 and taken[0] is not None
    rec = taken[0]
    assert rec.h is None and rec.v_new is None and rec.w is None and rec.aqk is None
    assert rec.seg_m is None and rec.aliases == ()
    assert rec.gk is None and rec.kg is None and rec.beta_int is None and rec.kb is None
    assert backward_arenas == [(None, "KdaWorkspace", p.arena_bytes())], backward_arenas
    _check_grads(on, off, "refused in _backward")


class _HandoffTestFailure(Exception):
    pass


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_records_of_a_failed_backward_are_dropped_by_the_next_forward(
    device, manifest, reentrant
) -> None:
    """A backward that fails after the recompute published its record (an exception in a later node's backward):
    torch may skip the graph task's final callbacks, so the record may outlive the backward; the next forward
    outside any backward drops it (one miss in total), and the next checkpointed step hands off again."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    cu = inputs["cu_seqlens"]

    def fail():
        raise _HandoffTestFailure("the backward fails after the recompute")

    def layer(q, k, v, g, beta):
        o = chunk_kda(q, k, v, g, beta, cu, checkpoint_handoff="on")
        return _AfterRecompute.apply(o, fail)

    leaves = _leaves(inputs)
    out = checkpoint(layer, *leaves, use_reentrant=reentrant)
    with pytest.raises(Exception) as info:
        out.backward(inputs["dout"])
    assert "the backward fails after the recompute" in str(info.value), info.value
    del out, info
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 0, s
    assert s["live_records"] in (0, 1), s  # 1 when torch skipped the final callbacks
    with torch.no_grad():
        chunk_kda(*_leaves(inputs), cu, checkpoint_handoff="on")
    s = _stats(device)
    assert s["live_records"] == 0 and s["live_bytes"] == 0 and s["misses"] == 1, s
    _checkpointed_step(inputs, reentrant=reentrant, handoff="on")
    s = _stats(device)
    assert s["hits"] == 1 and s["live_records"] == 0, s


@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_an_untaken_record_is_dropped_and_repeated_misses_turn_the_handoff_off(
    device, manifest, monkeypatch, reentrant
) -> None:
    """A backward that re-runs a checkpointed region but never reaches its KDA node (the gradient of another
    output only) leaves the record untaken: it is dropped at the end of its graph task (``queue_callback``) as a
    miss.  ``_HANDOFF_MAX_MISSES`` misses in a row turn the hand-off off on the device: the next checkpointed step
    publishes nothing and its gradients are the light path's."""
    lengths, heads = (100, 640, 77), 2
    inputs = make_inputs(lengths, heads, device)
    cu = inputs["cu_seqlens"]
    monkeypatch.setattr(cake_backend, "_HANDOFF_MAX_MISSES", 2)

    def pruned_step():
        leaves = _leaves(inputs)
        x = torch.randn(64, device=device, requires_grad=True)

        def region(q, k, v, g, beta, x_):
            # the KDA output leaves the graph (a reentrant region would otherwise run its node with a zero
            # gradient): the recompute publishes, no backward takes the record
            o = chunk_kda(q, k, v, g, beta, cu, checkpoint_handoff="on")
            return o.detach(), x_ * x_

        _o, y = checkpoint(region, *leaves, x, use_reentrant=reentrant)
        y.sum().backward()
        assert torch.equal(x.grad, 2 * x.detach())
        assert all(leaf.grad is None for leaf in leaves)

    pruned_step()
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 0, s
    assert s["live_records"] == 0 and s["misses"] == 1 and not s["disabled"], s
    pruned_step()
    s = _stats(device)
    assert s["misses"] == 2 and s["disabled"], s
    # turned off: no record, the light path's gradients
    _, on = _checkpointed_step(inputs, reentrant=reentrant, handoff="on")
    s = _stats(device)
    assert s["published"] == 2 and s["hits"] == 0 and s["refused_disabled"] == 1, s
    _, off = _checkpointed_step(inputs, reentrant=reentrant, handoff="off")
    _check_grads(on, off, "disabled")
    cake_backend._handoff_reset()  # back on
    _checkpointed_step(inputs, reentrant=reentrant, handoff="on")
    assert _stats(device)["hits"] == 1


# --------------------------------------------------------------------------- checkpoint variants


@pytest.mark.parametrize(
    "outer_reentrant", (False, True), ids=("outer-nonreentrant", "outer-reentrant")
)
def test_nested_checkpoint_regions_publish_only_the_recompute(
    device, manifest, outer_reentrant
) -> None:
    """Three blocks, each an outer checkpoint region (non-reentrant or reentrant) around an inner non-reentrant
    region with the KDA layer, followed by an op that saves the KDA output (so the outer recompute re-runs the
    inner region's original forward inside the backward).  That forward runs under the inner region's pack hook
    and publishes nothing (``refused_hooks``); each inner recompute publishes one record its node takes: three
    hits, no miss, no budget refusal, never more than one live record, and every gradient bitwise ``"off"``'s."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    first, second = _two_layer_inputs(lengths, heads, device)
    cu = first["cu_seqlens"]
    blocks = 3

    def run(mode: str):
        cake_backend._handoff_reset()
        sets = [first] + [second] * (blocks - 1)
        leaves = [_leaves(s) for s in sets]

        def kda(v_in, q, k, g, beta):
            return chunk_kda(q, k, v_in, g, beta, cu, checkpoint_handoff=mode)

        def block(v_in, q, k, g, beta):
            o = checkpoint(kda, v_in, q, k, g, beta, use_reentrant=False)
            return torch.nn.functional.silu(o)

        x = leaves[0][2]
        for i in range(blocks):
            q, k, _v, g, beta = leaves[i]
            x = checkpoint(block, x, q, k, g, beta, use_reentrant=outer_reentrant)
        x.backward(second["dout"])
        grads = {
            (i, name): leaf.grad
            for i in range(blocks)
            for name, leaf in zip(GRAD_OF_LEAF, leaves[i], strict=True)
            if not (i > 0 and name == "dv")
        }
        return grads, _stats(device)

    off, s_off = run("off")
    assert s_off["published"] == 0, s_off
    on, s = run("on")
    assert s["published"] == blocks and s["hits"] == blocks and s["misses"] == 0, s
    assert s["refused_budget"] == 0 and s["max_live_records"] <= 1, s
    assert s["live_records"] == 0, s
    for key in off:
        assert _bits_equal(on[key], off[key]), (outer_reentrant, key)


def test_handoff_with_checkpoint_early_stop_off(device, manifest) -> None:
    """``torch.utils.checkpoint.set_checkpoint_early_stop(False)``: the recompute runs the whole region; the record
    is taken and the gradients are bitwise ``"off"``'s."""
    set_early_stop = getattr(torch.utils.checkpoint, "set_checkpoint_early_stop", None)
    if set_early_stop is None:
        pytest.skip("this torch has no set_checkpoint_early_stop")
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    with set_early_stop(False):
        _check_handoff_against_off(inputs, device, lengths, heads, reentrant=False)


def test_a_second_backward_with_retain_graph_hands_off_again(device, manifest) -> None:
    """Non-reentrant checkpointing with ``retain_graph=True``: each backward recomputes, publishes and takes its own
    record; the accumulated gradients are bitwise ``"off"``'s."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    cu = inputs["cu_seqlens"]

    def run(mode: str):
        cake_backend._handoff_reset()
        leaves = _leaves(inputs)
        out = checkpoint(
            lambda *a: chunk_kda(*a, cu, checkpoint_handoff=mode),
            *leaves,
            use_reentrant=False,
        )
        out.backward(inputs["dout"], retain_graph=True)
        out.backward(inputs["dout"])
        return _grads(leaves), _stats(device)

    off, _ = run("off")
    on, s = run("on")
    assert s["published"] == 2 and s["hits"] == 2 and s["misses"] == 0, s
    assert s["live_records"] == 0, s
    _check_grads(on, off, "retain_graph")


# --------------------------------------------------------------------------- layers sharing a workspace


def _two_layer_inputs(lengths, heads, device):
    first = make_inputs(lengths, heads, device)
    second = make_inputs(lengths, heads, device, seed=12345)
    return first, second


def _layer_step(first, second, *, ws, handoff, reentrant, regions, layers=2):
    """``layers`` KDA layers on one caller workspace (the output of each is the next one's ``v``), checkpointed
    per layer (``regions="each"``) or as one region (``"one"``); gradients of every leaf by (layer, name)."""
    cu = first["cu_seqlens"]
    sets = [first] + [second] * (layers - 1)
    leaves = [_leaves(s) for s in sets]

    def layer(v_in, q, k, g, beta):
        return chunk_kda(
            q, k, v_in, g, beta, cu, workspace=ws, checkpoint_handoff=handoff
        )

    def args(i):
        q, k, _v, g, beta = leaves[i]
        return q, k, g, beta

    if regions == "each":
        x = leaves[0][2]
        for i in range(layers):
            x = checkpoint(layer, x, *args(i), use_reentrant=reentrant)
    else:

        def stack(v0, *flat):
            x = v0
            for i in range(layers):
                q, k, g, beta = flat[4 * i : 4 * i + 4]
                x = layer(x, q, k, g, beta)
            return x

        flat = [t for i in range(layers) for t in args(i)]
        x = checkpoint(stack, leaves[0][2], *flat, use_reentrant=reentrant)
    x.backward(second["dout"])
    out = {}
    for i in range(layers):
        for name, leaf in zip(GRAD_OF_LEAF, leaves[i], strict=True):
            if i > 0 and name == "dv":
                continue  # the v of a later layer is the previous output, not a leaf
            out[(i, name)] = leaf.grad
    return out


@pytest.mark.parametrize("shape", LAYER_ROWS, ids=_shape_id)
@pytest.mark.parametrize("reentrant", (False, True), ids=("nonreentrant", "reentrant"))
def test_two_checkpointed_layers_share_one_workspace(
    device, manifest, shape, reentrant, record_kind
) -> None:
    """Two KDA layers on one caller-owned workspace, checkpointed per layer or as one region (the recompute then
    publishes both records before either backward runs; the records live outside the arena the second layer's
    backward overwrites): two records published and taken, an empty registry, and every gradient bitwise
    ``"off"``'s.  With row records, per layer: two row records; as one region: the first record is published as a
    row record, the second's admission demotes it (two row records exceed the budget) and is published as a tape
    record, so both backwards take tape records."""
    lengths, heads = shape
    with_rows = int(_with_rows(_plan(lengths, heads, device), record_kind))
    first, second = _two_layer_inputs(lengths, heads, device)
    ws = KdaWorkspace.for_problem(
        sum(lengths), len(lengths), heads, device, deterministic=False
    )
    for regions in ("each", "one"):
        tag = (lengths, heads, reentrant, regions)
        cake_backend._handoff_reset()
        off = _layer_step(
            first, second, ws=ws, handoff="off", reentrant=reentrant, regions=regions
        )
        assert _stats(device)["published"] == 0, tag
        with _poisoned():
            on = _layer_step(
                first, second, ws=ws, handoff="on", reentrant=reentrant, regions=regions
            )
        s = _stats(device)
        assert s["published"] == 2 and s["hits"] == 2 and s["misses"] == 0, (tag, s)
        assert s["live_records"] == 0 and s["refused_budget"] == 0, (tag, s)
        if regions == "each":
            assert s["published_rows"] == 2 * with_rows and s["demoted"] == 0, (tag, s)
        else:
            assert s["published_rows"] == with_rows == s["demoted"], (tag, s)
        for key in off:
            assert _bits_equal(on[key], off[key]), (tag, key)


def test_the_byte_budget_admits_two_records_of_the_shape(
    device, manifest, record_kind
) -> None:
    """Three layers of one shape in one checkpointed region: the recompute publishes the first two records and
    the budget (two tape records of the shape, or one row record where larger) refuses the third before its
    extra writes; that layer's backward runs the light path.  With row records the first record is published as a
    row record and demoted by the second's admission (both then tape records).  Every gradient is bitwise
    ``"off"``'s and the registry is empty after the backward."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    with_rows = int(_with_rows(_plan(lengths, heads, device), record_kind))
    first, second = _two_layer_inputs(lengths, heads, device)
    ws = KdaWorkspace.for_problem(
        sum(lengths), len(lengths), heads, device, deterministic=False
    )
    for reentrant in (False, True):
        cake_backend._handoff_reset()
        kwargs = dict(ws=ws, reentrant=reentrant, regions="one", layers=3)
        off = _layer_step(first, second, handoff="off", **kwargs)
        on = _layer_step(first, second, handoff="on", **kwargs)
        s = _stats(device)
        assert s["published"] == 2 and s["refused_budget"] == 1, (reentrant, s)
        assert s["hits"] == 2 and s["misses"] == 0 and s["live_records"] == 0, s
        assert s["published_rows"] == with_rows == s["demoted"], (reentrant, s)
        for key in off:
            assert _bits_equal(on[key], off[key]), (reentrant, key)


# --------------------------------------------------------------------------- switches and the quiet paths


def test_auto_follows_the_head_counts_and_the_verified_devices(
    device, manifest, monkeypatch
) -> None:
    """The default (``"auto"``, no keyword) with the toolkit and the verified set patched, so the test runs on any
    device: it hands off at a ``HANDOFF_HEADS`` head count on a verified device, and stays off on another device
    (counting ``unverified``) and at a head count outside ``HANDOFF_HEADS`` (no request: nothing counted).  The
    gradients are bitwise ``"off"``'s in every case."""
    lengths = (100, 640, 77, 1000, 129)
    arch, sms = cake_jit.device_arch(device.index), cake_jit.sm_count(device.index)
    monkeypatch.setattr(
        cake_backend, "HANDOFF_VERIFIED", frozenset({(arch, sms, "9.9")})
    )
    # no device listed for row records: every record is a tape record
    monkeypatch.setattr(cake_backend, "HANDOFF_ROWS_VERIFIED", frozenset())
    heads = min(cake_backend.HANDOFF_HEADS)
    # (heads, toolkit, published and hits, unverified)
    cases = ((heads, "9.9", 1, 0), (heads, "9.8", 0, 1), (1, "9.9", 0, 0))
    for H, toolkit, engaged, unverified in cases:
        monkeypatch.setattr(cake_backend, "_handoff_toolkit", lambda a, t=toolkit: t)
        inputs = make_inputs(lengths, H, device)
        _, off = _checkpointed_step(inputs, reentrant=False, handoff="off")
        cake_backend._handoff_reset()
        _, auto = _checkpointed_step(inputs, reentrant=False, handoff=None)
        s = _stats(device)
        assert s["published"] == s["hits"] == engaged, (H, toolkit, s)
        assert s["unverified"] == unverified, (H, toolkit, s)
        assert s["live_records"] == 0, (H, toolkit, s)
        twin = _plan(lengths, H, device).handoff_twin
        assert s["published_rows"] == 0, (H, toolkit, s)
        assert s["rows_unverified"] == engaged * (twin is not None), (H, toolkit, s)
        _check_grads(auto, off, (H, toolkit))


def test_the_shipped_rule_on_this_device(device, manifest) -> None:
    """The default as shipped, nothing patched: this build's toolkit reads as ``major.minor``, and at the smallest
    ``HANDOFF_HEADS`` head count the default hands off exactly when this device (architecture, SM count, toolkit)
    is in ``HANDOFF_VERIFIED``, else it counts ``unverified``.  The toolkit of the one build of the twin, its intra
    record and the light prep reads as ``major.minor.patch``; the record (the call's own workspace) is a row record
    exactly when the device with that toolkit is in ``HANDOFF_ROWS_VERIFIED`` (else ``rows_unverified``) and the
    twin loads (else ``rows_unready``); the gradients are bitwise ``"off"``'s."""
    lengths, heads = (63, 1, 65), min(cake_backend.HANDOFF_HEADS)
    p = _plan(lengths, heads, device)
    key = (p.arch, p.sm_count, cake_backend._handoff_toolkit(p.arch))
    assert re.fullmatch(r"\d+\.\d+", key[2] or ""), key
    engaged = int(key in cake_backend.HANDOFF_VERIFIED)
    twin = p.handoff_twin
    assert twin is not None, (p.arch, p.record("fwd_intra", "fwd"))
    toolkit = cake_backend._handoff_rows_toolkit(
        twin,
        p.record("fwd_intra", "fwd"),
        p.record("bwd_prep", p.light_path),
        p.arch,
    )
    assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit or ""), (key, toolkit)
    assert toolkit.rsplit(".", 1)[0] == key[2], (key, toolkit)
    listed = (p.arch, p.sm_count, toolkit) in cake_backend.HANDOFF_ROWS_VERIFIED
    inputs = make_inputs(lengths, heads, device)
    _, off = _checkpointed_step(inputs, reentrant=False, handoff="off")
    cake_backend._handoff_reset()
    _, auto = _checkpointed_step(inputs, reentrant=False, handoff=None)
    s = _stats(device)
    assert s["published"] == s["hits"] == engaged, (key, s)
    assert s["unverified"] == 1 - engaged, (key, s)
    # a listed device's twin loads here (the suite's launches load it too)
    assert s["published_rows"] == int(engaged and listed), (key, toolkit, s)
    assert s["rows_unverified"] == int(engaged and not listed), (key, toolkit, s)
    assert s["rows_unready"] == 0, (key, toolkit, s)
    _check_grads(auto, off, key)


class _Built:
    """A stand-in for ``cake_build.prebuilt_path(name)``: ``is_file()`` is whether the library counts as prebuilt."""

    def __init__(self, built: bool) -> None:
        self.built = built

    def is_file(self) -> bool:
        return self.built


@pytest.fixture
def _fresh_rows_toolkit():
    """``_handoff_rows_toolkit`` caches its verdict per process: cleared around a test that patches the build
    state."""
    cake_backend._handoff_rows_toolkit.cache_clear()
    yield
    cake_backend._handoff_rows_toolkit.cache_clear()


@pytest.mark.parametrize(
    "case",
    ("unverified", "mixed_build", "load_fails", "caller_workspace", "verified"),
)
def test_a_row_record_falls_back_to_a_tape_record_where_the_rule_refuses_it(
    device, manifest, monkeypatch, _fresh_rows_toolkit, case
) -> None:
    """The rule (``_HANDOFF_ROWS`` unset) with ``"on"`` at a ``HANDOFF_HEADS`` head count, this device listed in
    ``HANDOFF_ROWS_VERIFIED`` with this build's toolkit (another patch version in the ``unverified`` case): a row
    record only on the call's own workspace with the one build and a twin that loads (``verified``: the twin
    launched, no prep); otherwise a tape record (the prep launched) and the counter of the refusal: another
    toolkit listed (``rows_unverified``), one of the three libraries prebuilt and another not (``mixed_build``:
    ``rows_unready``), the twin's load failing (``load_fails``: ``rows_unready``), a caller workspace (no
    counter).  The gradients are bitwise ``"off"``'s in every case."""
    lengths, heads = (100, 640, 77, 1000, 129), min(cake_backend.HANDOFF_HEADS)
    p = _plan(lengths, heads, device)
    twin = p.handoff_twin
    assert twin is not None, (p.arch, p.record("fwd_intra", "fwd"))
    parent, prep = p.record("fwd_intra", "fwd"), p.record("bwd_prep", p.light_path)
    toolkit = cake_backend._handoff_rows_toolkit(twin, parent, prep, p.arch)
    assert toolkit is not None, (
        "this build's twin, intra record and light prep are not one build"
    )
    cake_backend._handoff_rows_toolkit.cache_clear()
    major, minor, patch = toolkit.split(".")
    listed = toolkit if case != "unverified" else f"{major}.{minor}.{int(patch) + 1}"
    monkeypatch.setattr(
        cake_backend,
        "HANDOFF_ROWS_VERIFIED",
        frozenset({(p.arch, p.sm_count, listed)}),
    )
    if case == "mixed_build":
        # one of the three libraries flips its build state (the launches load every library through the real
        # paths): the twin's where nothing is prebuilt here, else the intra record's
        twin_library = cake_jit.library_name(twin, p.arch)
        parent_library = cake_jit.library_name(parent, p.arch)
        real_prebuilt = cake_build.prebuilt_path
        flipped = (
            parent_library if real_prebuilt(twin_library).is_file() else twin_library
        )

        def prebuilt(name):
            if name == flipped:
                return _Built(not real_prebuilt(name).is_file())
            return real_prebuilt(name)

        monkeypatch.setattr(cake_build, "prebuilt_path", prebuilt)
    elif case == "load_fails":
        real_load = cake_jit.load

        def load(name, arch):
            if name == twin:
                raise RuntimeError("the twin does not load")
            return real_load(name, arch)

        monkeypatch.setattr(cake_jit, "load", load)
    workspace = None
    if case == "caller_workspace":
        workspace = KdaWorkspace(p, device, backward=True, deterministic=False)
    runs: list[tuple[str, str | None]] = []
    real_run = KdaWorkspace.run

    def run(self, stage, values, *, path, **kwargs):
        runs.append((stage, kwargs.get("record")))
        return real_run(self, stage, values, path=path, **kwargs)

    inputs = make_inputs(lengths, heads, device)
    _, off = _checkpointed_step(inputs, reentrant=False, handoff="off")
    cake_backend._handoff_reset()
    monkeypatch.setattr(KdaWorkspace, "run", run)
    _, on = _checkpointed_step(
        inputs, reentrant=False, handoff="on", workspace=workspace
    )
    monkeypatch.setattr(KdaWorkspace, "run", real_run)
    s = _stats(device)
    with_rows = case == "verified"
    assert s["published"] == s["hits"] == 1 and s["live_records"] == 0, (case, s)
    assert s["published_rows"] == int(with_rows), (case, s)
    assert s["rows_unverified"] == int(case == "unverified"), (case, s)
    assert s["rows_unready"] == int(case in ("mixed_build", "load_fails")), (case, s)
    # the "on" step: the backward from a tape record runs the light prep, from a row record none; the recompute
    # launches the twin exactly for a row record
    preps = [stage for stage, _ in runs if stage == "bwd_prep"]
    assert len(preps) == int(not with_rows), runs
    assert (("fwd_intra", twin) in runs) is with_rows, runs
    _check_grads(on, off, case)


def test_the_environment_override_end_to_end(device, manifest, monkeypatch) -> None:
    """Through ``chunk_kda``, with ``"auto"`` engaging nowhere (an empty verified set): ``CAKE_KDA_CKPT_HANDOFF=off``
    turns ``"on"`` off and ``=on`` turns ``"auto"`` into ``"on"``; the gradients agree bitwise.  The host tests
    cover the other combinations."""
    monkeypatch.setattr(cake_backend, "HANDOFF_VERIFIED", frozenset())
    inputs = make_inputs((63, 1, 65), 2, device)
    grads = []
    for env, keyword, published in (("off", "on", 0), ("on", "auto", 1)):
        cake_backend._handoff_reset()
        monkeypatch.setenv(ENV, env)
        grads.append(_checkpointed_step(inputs, reentrant=False, handoff=keyword)[1])
        s = _stats(device)
        assert s["published"] == s["hits"] == published, (env, keyword, s)
        assert s["live_records"] == 0, (env, keyword, s)
    _check_grads(grads[1], grads[0], "environment")


def test_plain_and_original_forwards_publish_nothing(device, manifest) -> None:
    """No record outside a checkpoint recompute: a plain forward and backward, a forward under ``no_grad``, the
    original forward of a checkpoint (before its backward), an explicit saved-set call; a plain backward takes
    nothing and the checkpointed forward retains no record."""
    lengths, heads = (100, 640, 77, 1000, 129), 2
    inputs = make_inputs(lengths, heads, device)
    cu = inputs["cu_seqlens"]
    leaves = _leaves(inputs)
    out = chunk_kda(*leaves, cu, checkpoint_handoff="on")
    assert _stats(device)["published"] == 0
    out.backward(inputs["dout"])
    s = _stats(device)
    assert s["published"] == 0 and s["hits"] == 0 and s["misses"] == 0, s
    assert s["live_records"] == 0
    plain = _grads(leaves)
    with torch.no_grad():
        chunk_kda(*_leaves(inputs), cu, checkpoint_handoff="on")
    assert _stats(device)["published"] == 0
    cake_backend.chunk_kda_fwd(
        *(inputs[n] for n in (*LEAVES, "cu_seqlens")), deterministic=False
    )
    assert _stats(device)["published"] == 0
    # the plain call's output: freed before the baseline, not inside the measured window
    del out
    leaves = _leaves(inputs)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    out = checkpoint(
        lambda *a: chunk_kda(*a, cu, checkpoint_handoff="on"),
        *leaves,
        use_reentrant=False,
    )
    torch.cuda.synchronize(device)
    s = _stats(device)
    # the original forward runs outside any backward (graph task -1)
    assert s["published"] == 0 and s["live_records"] == 0 and s["live_bytes"] == 0, s
    # the checkpointed forward keeps its output and nothing more (no record, no tapes); the allocator may also have
    # released earlier per-call temporaries inside the window, so the bound is one-sided
    grown = torch.cuda.memory_allocated(device) - before
    assert grown <= out.untyped_storage().nbytes(), grown
    out.backward(inputs["dout"])
    s = _stats(device)
    assert s["published"] == 1 and s["hits"] == 1 and s["live_records"] == 0, s
    _check_grads(_grads(leaves), plain, "checkpoint vs plain")


def test_capture_blocks_the_handoff(device) -> None:
    """Inside CUDA-graph capture ``_handoff_blocked()`` holds (the request and the take both refuse)."""
    x = torch.zeros(1, device=device)
    graph = torch.cuda.CUDAGraph()
    assert not cake_backend._handoff_blocked()
    with torch.cuda.graph(graph):
        x.add_(1)
        inside = cake_backend._handoff_blocked()
    assert inside
    assert not cake_backend._handoff_blocked()


# --------------------------------------------------------------------------- host only


def test_the_auto_rule(monkeypatch) -> None:
    """Host only: ``"on"`` engages on every plan and ``"off"`` on none; ``"auto"`` at a ``HANDOFF_HEADS`` head count
    on a ``HANDOFF_VERIFIED`` (architecture, SM count, toolkit) device, counting ``unverified`` at those head counts
    on another architecture, SM count or toolkit (also one that cannot be read).  The node makes no ``"auto"``
    request at other head counts, before it reads any torch state.  The shipped devices name exported
    architectures and ``major.minor`` toolkits; the head counts start at two (one head is host-bound)."""
    for arch, sms, toolkit in cake_backend.HANDOFF_VERIFIED:
        assert arch in ARCHES and sms > 0, (arch, sms, toolkit)
        assert re.fullmatch(r"\d+\.\d+", toolkit), (arch, sms, toolkit)
    heads = min(cake_backend.HANDOFF_HEADS)
    assert heads == 2, cake_backend.HANDOFF_HEADS
    gap = next(h for h in range(heads, 1024) if h not in cake_backend.HANDOFF_HEADS)
    arch, other = ARCHES[0], ARCHES[1]
    monkeypatch.setattr(
        cake_backend, "HANDOFF_VERIFIED", frozenset({(arch, 152, "9.9")})
    )
    # (mode, architecture, SM count, toolkit, heads, engages, counts unverified)
    cases = (
        ("on", arch, 148, None, 1, True, False),
        ("off", arch, 152, "9.9", heads, False, False),
        ("auto", arch, 152, "9.9", heads, True, False),
        ("auto", arch, 152, "9.9", max(cake_backend.HANDOFF_HEADS), True, False),
        ("auto", arch, 152, "9.9", 1, False, False),
        ("auto", arch, 152, "9.9", gap, False, False),
        ("auto", arch, 148, "9.9", heads, False, True),
        ("auto", arch, 152, "9.8", heads, False, True),
        ("auto", arch, 152, None, heads, False, True),
        ("auto", other, 152, "9.9", heads, False, True),
    )
    unverified = 0
    for mode, arch_, sms, toolkit, H, want, counted in cases:
        monkeypatch.setattr(cake_backend, "_handoff_toolkit", lambda a, t=toolkit: t)
        p = types.SimpleNamespace(arch=arch_, sm_count=sms, H=H)
        assert cake_backend._handoff_engages(p, mode, 0) is want, (mode, p, toolkit)
        unverified += counted
        assert cake_backend._handoff_stats(0)["unverified"] == unverified, (mode, p)

    def read_torch_state():
        raise RuntimeError("the request read torch state")

    monkeypatch.setattr(cake_backend, "_handoff_task", read_torch_state)
    ctx = types.SimpleNamespace(needs_input_grad=(True,) * 5)

    def v(H):  # an operand on a CUDA device, without one
        return types.SimpleNamespace(
            device=torch.device("cuda", 0), shape=(64, H, D), dim=lambda: 3
        )

    for mode, H in (("auto", 1), ("auto", gap), ("off", heads)):
        assert cake_backend._handoff_request(ctx, mode, v(H)) is None, (mode, H)
    for mode, H in (("auto", heads), ("on", 1)):
        with pytest.raises(RuntimeError, match="torch state"):
            cake_backend._handoff_request(ctx, mode, v(H))


def test_a_full_window_of_misses_and_hits_turns_the_handoff_off(monkeypatch) -> None:
    """Host only: alternating hits and misses never reach ``_HANDOFF_MAX_MISSES`` in a row, but a full window of
    outcomes with as many misses as hits turns the hand-off off."""
    monkeypatch.setattr(cake_backend, "_HANDOFF_WINDOW", 4)
    cake_backend._handoff_reset()
    for hit in (True, False, True):
        cake_backend._handoff_count(0, hit=hit)
    assert not cake_backend._handoff_stats(0)["disabled"]
    cake_backend._handoff_count(0, hit=False)
    s = cake_backend._handoff_stats(0)
    assert s["hits"] == 2 and s["misses"] == 2 and s["disabled"], s


def test_handoff_mode_reads_the_environment_at_each_call(monkeypatch) -> None:
    """Host only: the keyword under each value of the variable (case and spaces ignored; unset, empty or ``auto``:
    no override; an explicit ``"off"`` stays off) and the refusals of either."""
    want = {
        None: ("auto", "on", "off"),
        "": ("auto", "on", "off"),
        "auto": ("auto", "on", "off"),
        " Off ": ("off", "off", "off"),
        "OFF": ("off", "off", "off"),
        "on": ("on", "on", "off"),
        " On ": ("on", "on", "off"),
    }
    for env, modes in want.items():
        if env is None:
            monkeypatch.delenv(ENV, raising=False)
        else:
            monkeypatch.setenv(ENV, env)
        got = tuple(cake_backend._handoff_mode(m) for m in ("auto", "on", "off"))
        assert got == modes, env
    monkeypatch.setenv(ENV, "1")
    with pytest.raises(ValueError, match=ENV):
        cake_backend._handoff_mode("auto")
    monkeypatch.delenv(ENV)
    for bad in ("ON", "force", None, True, 1):
        with pytest.raises(ValueError, match="checkpoint_handoff"):
            cake_backend._handoff_mode(bad)


def test_chunk_kda_passes_the_mode_to_the_node(monkeypatch) -> None:
    """Host only: ``chunk_kda`` hands ``ChunkKDAFunction`` the mode after the environment override, and ``"off"``
    under ``no_grad`` (no node is recorded: no backward would take a record) whatever the variable says."""
    seen: list = []

    class Spy:
        @staticmethod
        def apply(*args):
            seen.append(args[11])

    monkeypatch.setattr(cake_backend, "ChunkKDAFunction", Spy)
    ops = [torch.zeros(1) for _ in range(6)]
    monkeypatch.setenv(ENV, "on")
    cake_backend.chunk_kda(*ops, deterministic=False)
    with torch.no_grad():
        cake_backend.chunk_kda(*ops, checkpoint_handoff="on", deterministic=False)
    assert seen == ["on", "off"], seen


def test_the_apply_slot_refusals() -> None:
    """Host only: ``ChunkKDAFunction.apply`` written for the signature without ``checkpoint_handoff`` (its trailing
    ``bool`` lands in the slot) fails with a ``TypeError`` naming the slot, and an unknown mode with a
    ``ValueError``, before any operand check or launch."""
    ops = [torch.zeros(1) for _ in range(6)]
    for flag in (False, True):
        with pytest.raises(TypeError, match="checkpoint_handoff"):
            cake_backend.ChunkKDAFunction.apply(
                *ops, SCALE, None, None, None, None, flag
            )
    with pytest.raises(ValueError, match="checkpoint_handoff"):
        cake_backend.ChunkKDAFunction.apply(
            *ops, SCALE, None, None, None, None, "force", False
        )


def test_the_recompute_is_told_apart_by_its_saved_tensor_hooks() -> None:
    """Host only (CPU tensors): torch's innermost saved-tensor hooks and graph task as ``_handoff_request`` reads
    them.  Non-reentrant: the original forward sees the region's pack hook outside any backward, the recompute
    the recompute hook inside it.  Reentrant: no hooks, the first pass outside any backward without grad, the
    recompute inside it.  A caller's hooks are ``"other"``.  Nested non-reentrant regions: the outer recompute
    re-runs the inner region's original forward inside the backward under the inner pack hook (refused), and the
    inner recompute runs last under its recompute hook."""
    seen: list = []

    def fn(x):
        seen.append(
            (
                cake_backend._handoff_hooks(),
                cake_backend._handoff_task() not in (None, -1),
                torch.is_grad_enabled(),
            )
        )
        return x.sin()

    assert cake_backend._handoff_hooks() == "none"
    assert cake_backend._handoff_task() == -1
    x = torch.randn(8, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    assert seen == [("original", False, True), ("recompute", True, True)], seen
    seen.clear()
    checkpoint(fn, x, use_reentrant=True).sum().backward()
    assert seen == [("none", False, False), ("none", True, True)], seen
    with torch.autograd.graph.saved_tensors_hooks(lambda t: t, lambda t: t):
        assert cake_backend._handoff_hooks() == "other"
    seen.clear()

    def outer(x_):
        return checkpoint(fn, x_, use_reentrant=False).cos()

    checkpoint(outer, x, use_reentrant=False).sum().backward()
    assert seen[0] == ("original", False, True), seen
    assert ("original", True, True) in seen[1:-1], seen
    assert seen[-1] == ("recompute", True, True), seen


class _ProbeFunction(torch.autograd.Function):
    """Identity whose forward records what ``_handoff_request`` reads there (it runs inside
    ``ChunkKDAFunction.forward``): the hooks, whether a graph task runs, grad mode and ``needs_input_grad``."""

    @staticmethod
    def forward(ctx, x, seen):
        seen.append(
            (
                cake_backend._handoff_hooks(),
                cake_backend._handoff_task() not in (None, -1),
                torch.is_grad_enabled(),
                bool(ctx.needs_input_grad[0]),
            )
        )
        ctx.save_for_backward(x)
        return x.clone()

    @staticmethod
    def backward(ctx, grad):
        return grad, None


def test_a_function_forward_sees_the_recompute_hooks_without_grad_mode() -> None:
    """Host only (CPU tensors): torch runs an autograd Function's forward with grad mode off, also in the checkpoint
    recompute (so ``_handoff_request`` cannot use ``torch.is_grad_enabled()``; ``chunk_kda`` passes ``"off"`` when
    grad mode is off at its call), while the saved-tensor hooks, the graph task and ``needs_input_grad`` there tell
    the recompute apart."""
    seen: list = []

    def fn(x_):
        return _ProbeFunction.apply(x_, seen).sin()

    x = torch.randn(8, requires_grad=True)
    checkpoint(fn, x, use_reentrant=False).sum().backward()
    modes = [s[:3] for s in seen]
    assert modes == [("original", False, False), ("recompute", True, False)], seen
    assert seen[-1][3], seen
    seen.clear()
    checkpoint(fn, x, use_reentrant=True).sum().backward()
    modes = [s[:3] for s in seen]
    assert modes == [("none", False, False), ("none", True, False)], seen
    assert seen[-1][3], seen


def test_compile_compiled_autograd_and_dispatch_modes_block(monkeypatch) -> None:
    """Host only: ``torch.compiler.is_compiling()``, compiled autograd's flags (on, missing or not ``bool``: fail
    closed) and an active Python dispatch mode block the hand-off; nothing blocks it outside them."""
    from torch.utils._python_dispatch import TorchDispatchMode

    monkeypatch.delitem(sys.modules, "torch._dynamo.compiled_autograd", raising=False)
    if not torch.cuda.is_available():
        monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    assert not cake_backend._handoff_blocked()
    assert not cake_backend._handoff_dispatch_modes()
    with monkeypatch.context() as m:
        m.setattr(torch.compiler, "is_compiling", lambda: True, raising=False)
        assert cake_backend._handoff_blocked()
    names = ("in_compiled_autograd_region", "compiled_autograd_enabled")
    cases = (
        ({names[0]: False, names[1]: False}, False),
        ({names[0]: True, names[1]: False}, True),
        ({names[0]: False, names[1]: True}, True),
        ({names[1]: False}, True),  # a flag this module cannot read
        ({names[0]: 0, names[1]: False}, True),  # not a bool
    )
    for flags, want in cases:
        fake = types.ModuleType("torch._dynamo.compiled_autograd")
        for name, value in flags.items():
            setattr(fake, name, value)
        monkeypatch.setitem(sys.modules, "torch._dynamo.compiled_autograd", fake)
        assert cake_backend._handoff_blocked() is want, flags

    class Passthrough(TorchDispatchMode):
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with Passthrough():
        assert cake_backend._handoff_dispatch_modes()
    assert not cake_backend._handoff_dispatch_modes()


def test_the_real_compiled_autograd_flags_block_inside_its_context() -> None:
    """Host only: this torch's ``torch._dynamo.compiled_autograd`` has both flags as ``bool`` (else the hand-off
    fails closed in every process that imports it), off outside its enable context and blocking inside it."""
    ca = pytest.importorskip("torch._dynamo.compiled_autograd")
    for name in ("in_compiled_autograd_region", "compiled_autograd_enabled"):
        value = getattr(ca, name, None)
        assert isinstance(value, bool), (name, value)
    if not torch.cuda.is_available():
        pytest.skip("the capture check needs CUDA")
    assert not cake_backend._handoff_blocked()
    enable = getattr(ca, "_enable", None) or getattr(ca, "enable", None)
    if enable is None:
        pytest.skip("this torch exposes no compiled-autograd enable context")
    try:
        ctx = enable(lambda gm: gm)
        ctx.__enter__()
    except Exception as exc:
        pytest.skip(
            f"the compiled-autograd enable context cannot be entered here: {exc!r}"
        )
    try:
        assert cake_backend._handoff_blocked()
    finally:
        ctx.__exit__(None, None, None)
    assert not cake_backend._handoff_blocked()


def test_memory_report_covers_the_handoff() -> None:
    """Host only: ``memory_report(...)["handoff_bytes"]`` is one tape record of the plan (the tapes; the forward's
    w where the record allocates it; two state blocks on the segmented route when the light set saves none);
    ``handoff_bytes_rows`` one row record (772 B more per internal token-head, 256 B more for ``kb`` where the
    record keeps no ``w``; ``None`` where the twin is not delivered for the architecture); ``handoff_budget_bytes``
    two tape records (the budget of a recompute that asks for a tape record); the per-call arenas of a backward from
    a tape record (``bwd_tape``) and from a row record (``HANDOFF_LAYOUT``: 516 B less per internal token-head, no
    ``gk`` / ``beta_int`` slots)."""
    for lengths in (P8, (100, 640, 77, 1000, 129), (63, 1, 65)):
        S, N = sum(lengths), len(lengths)
        for arch in ARCHES:
            for sm in (148, 152):
                for H in (1, 2, 4, 8, 12, 16, 32, 64):
                    tag = (_shape_id((lengths, H)), arch, sm)
                    p = cake_backend.plan(S, N, H, sm, arch)
                    want = p.NC_pad * H * D * D * 2 + p.R * H * D * 2
                    if p.dispatched("wy_fwd", "fwd") and not p.light_saves_wu:
                        want += p.R * H * D * 2
                    if p.recur == "seg" and not p.seg_save:
                        want += 2 * p.NSEG * H * D * D * 4
                    assert p.handoff_bytes() == want, tag
                    assert p.handoff_twin == TWINS[p.record("fwd_intra", "fwd")], tag
                    want_rows = want + p.R * H * ROW_BYTES
                    if not p.handoff_keeps_w:
                        want_rows += p.R * H * ROW_KB_BYTES
                    assert p.handoff_bytes(rows=True) == want_rows, tag
                    tape = p.workspace_bytes("bwd_tape")
                    lean = p.workspace_bytes(cake_backend.HANDOFF_LAYOUT)
                    assert lean == tape - ROW_ARENA_SAVING * p.R * H, tag
                    for save_mode in ("light", "tape"):
                        report = cake_backend.memory_report(
                            S, N, H, save_mode, sm_count=sm, arch=arch
                        )
                        assert report["handoff_bytes"] == want, tag
                        assert report["handoff_bytes_rows"] == want_rows, tag
                        assert (
                            report["handoff_budget_bytes"]
                            == cake_backend._HANDOFF_BUDGET_RECORDS * want
                        ), tag
                        assert report["arena_bwd_tape"] == tape, tag
                        assert report["arena_bwd_handoff"] == lean, tag


def test_the_row_record_rule(monkeypatch) -> None:
    """Host only (a stand-in plan): an engaged recompute asks for a row record when the plan has an intra twin, the
    call runs on its own workspace, the head count is a ``HANDOFF_HEADS`` one, the device (architecture, SM count)
    has an entry in ``HANDOFF_ROWS_VERIFIED`` (else ``rows_unverified``, the build not read), the toolkit of the one
    build of the twin, its intra record and its light prep completes the key (another toolkit: ``rows_unverified``;
    a mixed build or an unreadable toolkit: ``rows_unready``) and the twin loads (else ``rows_unready``);
    ``_HANDOFF_ROWS`` overrides all of it but the twin.  The shipped entries name exported architectures and
    ``major.minor.patch`` toolkits, each device also in ``HANDOFF_VERIFIED`` with its ``major.minor`` (a row record
    is a hand-off record first)."""
    for key in cake_backend.HANDOFF_ROWS_VERIFIED:
        arch, sms, toolkit = key
        assert arch in ARCHES and sms > 0, key
        assert re.fullmatch(r"\d+\.\d+\.\d+", toolkit), key
        assert (arch, sms, toolkit.rsplit(".", 1)[0]) in cake_backend.HANDOFF_VERIFIED
    heads = min(cake_backend.HANDOFF_HEADS)
    gap = next(h for h in range(heads, 1024) if h not in cake_backend.HANDOFF_HEADS)
    arch = ARCHES[0]
    monkeypatch.setattr(
        cake_backend, "HANDOFF_ROWS_VERIFIED", frozenset({(arch, 152, "9.9.1")})
    )
    asked: list[tuple] = []
    state = {"toolkit": "9.9.1", "loads": True}

    def rows_toolkit(twin, parent, prep, arch_):
        asked.append(("toolkit", twin, parent, prep, arch_))
        return state["toolkit"]

    def rows_loads(twin, arch_):
        asked.append(("load", twin, arch_))
        return state["loads"]

    monkeypatch.setattr(cake_backend, "_handoff_rows_toolkit", rows_toolkit)
    monkeypatch.setattr(cake_backend, "_handoff_rows_loads", rows_loads)

    def plan_(twin="twin", sms=152, H=heads, arch_=arch):
        return types.SimpleNamespace(
            handoff_twin=twin,
            arch=arch_,
            sm_count=sms,
            H=H,
            light_path="light",
            record=lambda stage, path: f"{stage}@{path}",
        )

    other_arch = plan_(arch_=ARCHES[1])
    # (override, plan, own workspace, toolkit, loads, row record, counter, checks asked)
    cases = (
        (None, plan_(), True, "9.9.1", True, True, None, 2),
        (None, plan_(sms=148), True, "9.9.1", True, False, "rows_unverified", 0),
        (None, other_arch, True, "9.9.1", True, False, "rows_unverified", 0),
        (None, plan_(), True, "9.9.2", True, False, "rows_unverified", 1),
        (None, plan_(), True, None, True, False, "rows_unready", 1),
        (None, plan_(), True, "9.9.1", False, False, "rows_unready", 2),
        (None, plan_(), False, "9.9.1", True, False, None, 0),
        (None, plan_(H=gap), True, "9.9.1", True, False, None, 0),
        (None, plan_(H=1), True, "9.9.1", True, False, None, 0),
        (None, plan_(twin=None), True, "9.9.1", True, False, None, 0),
        (False, plan_(), True, "9.9.1", True, False, None, 0),
        (True, plan_(sms=148, H=gap), False, None, False, True, None, 0),
        (True, plan_(twin=None), True, "9.9.1", True, False, None, 0),
    )
    for override, p, own_ws, toolkit, loads, want, counter, checks in cases:
        tag = (override, p.arch, p.sm_count, p.H, p.handoff_twin, own_ws, toolkit)
        cake_backend._handoff_reset()
        asked.clear()
        monkeypatch.setattr(cake_backend, "_HANDOFF_ROWS", override)
        state.update(toolkit=toolkit, loads=loads)
        assert cake_backend._handoff_wants_rows(p, 0, own_ws) is want, tag
        s = cake_backend._handoff_stats(0)
        for name in ("rows_unverified", "rows_unready"):
            assert s[name] == int(name == counter), (tag, s)
        # the build is read last but one, on the twin, the plan's intra record and its light prep; the load last
        expected = [
            ("toolkit", "twin", "fwd_intra@fwd", "bwd_prep@light", p.arch),
            ("load", "twin", p.arch),
        ]
        assert asked == expected[:checks], (tag, asked)


def test_the_row_record_toolkit_reads_one_build(
    monkeypatch, _fresh_rows_toolkit
) -> None:
    """Host only (the build state patched): the toolkit of the twin, the plan's intra record and its light prep is
    the builder's sidecar ``cuda`` when all three are prebuilt and the JIT's ``nvcc`` version when none of them and
    no library of the package is, as ``major.minor.patch``; a mixed state, a version without its patch level, an
    architecture the twin is not delivered for or any other exception gives ``None``.  Cached per arguments."""
    state = {"prebuilt": set(), "sidecar": "13.4.92", "nvcc": "13.4.93"}
    reads: list[str] = []
    others = {"other_sm_100a": {}, "twin_sm_100a": {}, "parent_sm_100a": {}}

    def library_name(name, arch):
        if name == "undelivered":
            raise RuntimeError("not delivered for this architecture")
        return f"{name}_{arch}"

    def prebuilt_info(path, arch):
        reads.append("sidecar")
        return {"cuda": state["sidecar"]}

    def nvcc():
        reads.append("nvcc")
        if state["nvcc"] is None:
            raise FileNotFoundError("nvcc")
        return cake_backend._full_version(state["nvcc"])

    monkeypatch.setattr(cake_jit, "library_name", library_name)
    monkeypatch.setattr(
        cake_build, "prebuilt_path", lambda name: _Built(name in state["prebuilt"])
    )
    monkeypatch.setattr(cake_build, "libraries_by_name", lambda: others)
    monkeypatch.setattr(cake_build, "_prebuilt_info", prebuilt_info)
    monkeypatch.setattr(cake_backend, "_jit_nvcc_version", nvcc)
    three = {"twin_sm_100a", "parent_sm_100a", "prep_sm_100a"}
    # (prebuilt, sidecar cuda, nvcc version, twin, toolkit, the read)
    cases = (
        (three, "13.4.92", "13.4.93", "twin", "13.4.92", ["sidecar"]),
        (set(), "13.4.92", "13.4.93", "twin", "13.4.93", ["nvcc"]),
        (three, "13.4", "13.4.93", "twin", None, ["sidecar"]),
        (set(), "13.4.92", None, "twin", None, ["nvcc"]),
        ({"other_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        ({"twin_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        (three - {"prep_sm_100a"}, "13.4.92", "13.4.93", "twin", None, []),
        (three, "13.4.92", "13.4.93", "undelivered", None, []),
    )
    for prebuilt, sidecar, nvcc_version, twin, want, read in cases:
        tag = (sorted(prebuilt), sidecar, nvcc_version, twin)
        cake_backend._handoff_rows_toolkit.cache_clear()
        reads.clear()
        state.update(prebuilt=set(prebuilt), sidecar=sidecar, nvcc=nvcc_version)
        got = cake_backend._handoff_rows_toolkit(twin, "parent", "prep", "sm_100a")
        assert got == want, (tag, got)
        assert reads == read, (tag, reads)
        # cached: a second call reads nothing again
        state.update(prebuilt=set(), sidecar="0.0.0", nvcc="0.0.0")
        again = cake_backend._handoff_rows_toolkit(twin, "parent", "prep", "sm_100a")
        assert again == want, (tag, again)
        assert reads == read, (tag, reads)
    for text, want in (
        ("13.4.92", "13.4.92"),
        ("13.04.092", "13.4.92"),
        ("13.4.92.1", "13.4.92"),
        ("13.4", None),
        ("13.x.92", None),
    ):
        assert cake_backend._full_version(text) == want, text


def test_a_failed_twin_load_is_retried_a_bounded_number_of_times(monkeypatch) -> None:
    """Host only: a failed load of the twin is counted, not kept (the next recompute loads again) up to
    ``_HANDOFF_ROWS_LOAD_TRIES`` failures per twin and architecture; after those the process stops loading it
    until ``_handoff_reset``.  A load that succeeds is asked every time (``cake_jit.load`` keeps the module)."""
    loads: list[tuple] = []
    state = {"fail": True}

    def load(name, arch):
        loads.append((name, arch))
        if state["fail"]:
            raise RuntimeError("JIT lock timeout")
        return types.SimpleNamespace()

    monkeypatch.setattr(cake_jit, "load", load)
    tries = cake_backend._HANDOFF_ROWS_LOAD_TRIES
    assert tries >= 2
    for _ in range(tries + 2):
        assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    assert loads == [("twin", "sm_100a")] * tries, loads
    # another architecture counts apart; a success is not counted
    state["fail"] = False
    assert cake_backend._handoff_rows_loads("twin", "sm_103a") is True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    cake_backend._handoff_reset()
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True
    assert loads[tries:] == [("twin", "sm_103a")] + [("twin", "sm_100a")] * 2, loads
    # a transient failure then a success: the success counts
    cake_backend._handoff_reset()
    state["fail"] = True
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is False
    state["fail"] = False
    assert cake_backend._handoff_rows_loads("twin", "sm_100a") is True


def _host_record(*, rows: bool, kb: bool = False, task: int = 7):
    """A CPU hand-off record and its byte count: 1000 B of tapes and, for a row record, 640 B of rows (and 256 B
    of ``kb``)."""
    row_tensors = {}
    if rows:
        row_tensors = {
            "gk": torch.zeros(100, dtype=torch.float32),
            "kg": torch.zeros(100, dtype=torch.bfloat16),
            "beta_int": torch.zeros(10, dtype=torch.float32),
        }
        if kb:
            row_tensors["kb"] = torch.zeros(128, dtype=torch.bfloat16)
    nbytes = 1000 + sum(t.numel() * t.element_size() for t in row_tensors.values())
    tensors = {
        "h": torch.zeros(250, dtype=torch.float32),
        "v_new": torch.zeros(1, dtype=torch.bfloat16),
        "w": None,
        "seg_m": None,
        **row_tensors,
    }
    request = {
        "index": 0,
        "task": task,
        "key": ("key",),
        "stream": 0,
        "nbytes": nbytes,
        "tensors": tensors,
    }
    return request, nbytes


def test_the_budget_demotes_row_records_before_it_refuses(monkeypatch) -> None:
    """Host only (CPU records): a call that asks for a tape record is admitted within ``_HANDOFF_BUDGET_RECORDS``
    tape records of its shape, one that asks for a row record within the larger of that and one row record.  A row
    record is admitted when the live records plus it fit; otherwise, and for a call that asks for a tape record, a
    tape record is admitted when the live records plus it fit, every live row record of the device first becoming
    a tape record (its rows and their bytes dropped, ``demoted``) where they do not (``refused_budget`` when it
    still does not fit).  Publishing counts ``published_rows`` for a row record; the end-of-task drop releases the
    records."""
    tape = 1000
    req_a, rows = _host_record(rows=True)
    assert rows == 1640
    aqk_a, aqk_b = torch.zeros(4), torch.zeros(4)
    inputs = [torch.zeros(2) for _ in range(6)]

    def live(aqk):
        return cake_backend._handoff_devices[0].records[aqk.data_ptr()]

    def dropped(rec):
        return all(t is None for t in (rec.gk, rec.kg, rec.beta_int, rec.kb))

    assert cake_backend._handoff_admit(0, 7, tape, rows) == "rows"  # an empty device
    cake_backend._handoff_publish(req_a, aqk_a, inputs)
    s = cake_backend._handoff_stats(0)
    assert s["published"] == s["published_rows"] == 1 and s["live_bytes"] == rows, s
    rec_a = live(aqk_a)
    # a call that asks for a tape record (1640 + 1000 > 2000): the live row record is demoted, the tape record
    # admitted
    assert cake_backend._handoff_admit(0, 7, tape, None) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == tape and rec_a.nbytes == tape, s
    assert dropped(rec_a) and rec_a.h is not None and rec_a.v_new is not None
    req_b, _ = _host_record(rows=False)
    cake_backend._handoff_publish(req_b, aqk_b, inputs)
    s = cake_backend._handoff_stats(0)
    assert s["published"] == 2 and s["published_rows"] == 1, s
    assert s["live_bytes"] == 2 * tape, s
    # two tape records live: neither kind fits, nothing left to demote
    assert cake_backend._handoff_admit(0, 7, tape, rows) is None
    assert cake_backend._handoff_admit(0, 7, tape, None) is None
    s = cake_backend._handoff_stats(0)
    assert s["refused_budget"] == 2 and s["demoted"] == 1, s
    cake_backend._handoff_task_end(0, 7)
    s = cake_backend._handoff_stats(0)
    assert s["live_records"] == 0 and s["live_bytes"] == 0 and s["misses"] == 2, s
    # a second row record does not fit (1640 + 1640 > max(2000, 1640)): the live one is demoted, a tape record
    # admitted (1000 + 1000 <= 2000)
    cake_backend._handoff_reset()
    assert cake_backend._handoff_admit(0, 7, tape, rows) == "rows"
    cake_backend._handoff_publish(_host_record(rows=True)[0], aqk_a, inputs)
    assert cake_backend._handoff_admit(0, 7, tape, rows) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == tape and dropped(live(aqk_a)), s
    # a row record larger than two tape records: the budget of a call that asks for one is one row record
    cake_backend._handoff_reset()
    assert cake_backend._handoff_admit(0, 7, 500, rows) == "rows"
    req_c, _ = _host_record(rows=True, kb=True)
    cake_backend._handoff_publish(req_c, aqk_a, inputs)
    assert cake_backend._handoff_stats(0)["live_bytes"] == rows + 256
    # 1896 live + 1640 > max(1000, 1640); 1896 + 500 too: demoted to its 1000 B of tapes, then 1000 + 500 fits
    assert cake_backend._handoff_admit(0, 7, 500, rows) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 1 and s["live_bytes"] == 1000, s
    # a call that asks for a tape record of that shape: 1000 + 500 > 2 * 500, nothing left to demote
    assert cake_backend._handoff_admit(0, 7, 500, None) is None
    # a live row record that fits as it is stays one: 1640 + 2000 <= 2 * 2000
    cake_backend._handoff_reset()
    cake_backend._handoff_publish(_host_record(rows=True)[0], aqk_a, inputs)
    assert cake_backend._handoff_admit(0, 7, 2000, None) == "tape"
    s = cake_backend._handoff_stats(0)
    assert s["demoted"] == 0 and not dropped(live(aqk_a)), s
    # a turned-off device admits nothing
    cake_backend._handoff_devices[0].disabled = True
    assert cake_backend._handoff_admit(0, 7, 1, 1) is None
