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

The in-kernel Q / K L2 norm of the context-parallel (KCP) KDA training route (``use_qk_l2norm_in_kernel=True`` of
``cake_kcp.chunk_kda_det_kcp`` / ``chunk_kda_det_fwd_kcp`` / ``chunk_kda_det_bwd_kcp``, issue #7817).

CPU rows: the keyword (keyword-only, ``False`` by default, a bool, never an option) and the workspace's form knob;
flag-off identity (the omitted keyword and an explicit ``False`` give the same graph keys, recompute-handle
fingerprints, saved-set tensors / flags / meta, byte queries and existing-backend plan calls) and a hook that compares
that snapshot with one taken on the base of this change; what the flag adds to the keys, the fingerprint and the plan
calls; the saved set's ``qk_rstd`` round trip and the backward's checks of it; the strided-input rule, its metadata
checked alike on every rank; the workspace's mode; the byte queries (the flag-off ones make no CUDA call; the flag-on
ones check the contexts' device and the call's gate mode); the values the call binds for every local stage (the
existing backend's logical names and ``read_rstd`` / ``write_rstd``), the stage order of both forms, the fallback
form's stages never skipped, and the capability check (the existing backend's selection of every stage of the form
over a fixed problem grid, the records' architectures and exports, the first readers, the layouts' slot views); the
refusal before the operand checks; and, while the existing backend lacks the mode's programs, the
``NotImplementedError`` of every flag-on call, workspace and byte query before any workspace sizing, plan or
collective, with the same message on a rank with tokens and on one without, also in spawned CPU groups of 2 and 4
ranks (gloo), a rank with a misaligned view too.  Once it has them: every record the real forward / backward drivers
launch in either form binds all its logical names.  The flag-on CPU rows of the host plumbing run with the
capability check passed (:func:`capable`; nothing launches on the CPU).

GPU rows (one process per rank; ``W = 1`` in the pytest process, ``W > 1`` spawned or one process per Slurm task as in
``test_cake_kda_det_train_kcp_dist.py``): flag-off identity on one device; then every rank runs the flag-on KCP
forward / backward in every mode on raw q / k (normalized rows times ``2^e`` per token and head, ``e`` in
``[-6, 6]``, plus zero and subnormal rows) and must be bitwise repeatable (saved == replay, repeat, autograd ==
explicit, checkpoint == direct, a recompute handle of both tiers == plain checkpoint, save_transfer, ring == a2a,
graph replay == eager, ``workspace=None``, packed strided q / k == contiguous), run the flag-off route's collectives
and local stages, keep the FP32 rstd ``[2, T_r, H]`` (empty on a rank without tokens, which runs no local stage), and
rank 0 compares the gathered result with the single-rank flag-on path and the exact FP64 reference (the normalization
differentiated in FP64): relative L2 error of KCP / single-rank <= 1.10 for ``out, dq, dk, dv, dbeta, dg``.  The
fallback form (the workspace's test knob) runs the same: repeatable, within the same accuracy gate; its bitwise
equality with the fused form is reported, and a fallback form that cannot run yet fails the row with its reason
(``fallback_pending``).  A flag-off launch record carries no key of the mode.  The raw-gate rows run the same with
``use_gate_in_kernel=True``.  The flag-on rows skip with the reason of ``cake_kcp_plan.qk_l2norm_missing()`` until
the existing backend serves the fused form on the device.
"""

from __future__ import annotations

import collections
import dataclasses
import datetime
import gc
import inspect
import json
import os
import re
import tempfile
import traceback
import types

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from tests.experimental.test_cake_kda_det_train_kcp_dist import (
    _free_port,
    _local_slices,
    _partition,
    _programs_exported,
    _to_global,
)
from tests.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    error_metrics,
    kda_exact_forward,
    make_inputs,
)
from tests.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake_kda_det_train_kcp", "cake_kcp")
cake_kcp_plan = cake_package("cake_kda_det_train_kcp", "cake_kcp_plan")
cake_kcp_jit = cake_package("cake_kda_det_train_kcp", "cake_kcp_jit")
cake_backend = cake_package("cake_kda_det_train", "cake_backend")
cake_build = cake_package("cake_kda_det_train", "cake_build")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")

D = cake_kcp.D
SCALE = cake_kcp.SCALE
EPS = 1e-6  # the issue's eps under the square root
L2N = {"use_qk_l2norm_in_kernel": True}
OFF = {"use_qk_l2norm_in_kernel": False}
FALLBACK = {"_qk_l2norm_form": "fallback"}  # the workspace's test knob
TOKEN_GRADS = (
    "dq",
    "dk",
    "dv",
    "dbeta",
    "dg",
)  # the backward's order (dg: the raw gate's in the raw mode)
PARAM_GRADS = ("dA_log", "ddt_bias")  # raw gate mode: this rank's partial sums
ACC_GATE = 1.10  # relative L2 error of KCP / single-rank against the FP64 reference
# Cluster gate hook (flag-off identity against the base of this change): the omitted-keyword snapshot is written to
# $KCP_L2N_FLAG_OFF_CAPTURE (run there on the base commit with this file copied onto it) and compared with
# $KCP_L2N_FLAG_OFF_BASELINE (run here); the GPU row uses the same variables with the suffixes ".gpu.json" / ".gpu.pt".
ENV_CAPTURE = "KCP_L2N_FLAG_OFF_CAPTURE"
ENV_BASELINE = "KCP_L2N_FLAG_OFF_BASELINE"
GRAPH_EVENTS = [
    ["eager", "eager"],
    ["capture", "eager"],
    ["replay", "capture"],
    ["replay", "replay"],
]


def _l2n_missing(form: str = "fused", gate: str = "g") -> str | None:
    """The route's capability check on the current device's architecture (none without CUDA): ``None`` once the
    existing backend serves ``form`` in gate mode ``gate``, else the reason (``cake_kcp_plan.qk_l2norm_missing``)."""
    return cake_kcp_plan.qk_l2norm_missing(form, gate, cake_kcp_plan.qk_l2norm_arch())


@pytest.fixture
def capable(monkeypatch):
    """The capability check passed (``qk_l2norm_missing`` -> ``None``): the CPU rows of the flag-on host plumbing
    run before the existing backend has the mode's programs (nothing launches on the CPU)."""
    monkeypatch.setattr(cake_kcp_plan, "qk_l2norm_missing", lambda *a, **k: None)


def _missing(reason: str | None):
    """A stand-in for ``cake_kcp_plan.qk_l2norm_missing`` that gives ``reason`` for every mode, gate and
    architecture."""
    return lambda *a, **k: reason


# --------------------------------------------------------------------------- CPU helpers


class _Stop(Exception):
    """Raised by the recorders below: the call stops before any device work."""


def _cpu_ctx(W: int = 2, rank: int = 0, lengths=(700, 1, 1300, 999), sm: int = 148):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    part = cake_kcp_plan.partition_det_kcp(cu, W, layout="zigzag_per_document")
    return cake_kcp_plan.build_context_det_kcp(part, None, rank, "cpu", 4, sm_count=sm)


def _cpu_zero_ctx(rank: int = 1):
    """A rank of a ragged CP4 partition; rank 1 holds no token."""
    part, _ = _partition((3000,), 4, "ragged:1200,0,900,900")
    return cake_kcp_plan.build_context_det_kcp(part, None, rank, "cpu", 4, sm_count=148)


def _cpu_ops(ctx, H: int) -> dict:
    T = int(ctx.T_r)
    gen = torch.Generator().manual_seed(7817)
    ops = {n: torch.randn(T, H, D, generator=gen).to(torch.bfloat16) for n in "qkv"}
    ops["g"] = -torch.rand(T, H, D, generator=gen)
    ops["beta"] = torch.rand(T, H, generator=gen).to(torch.bfloat16)
    ops["dout"] = torch.randn(T, H, D, generator=gen).to(torch.bfloat16)
    return ops


def _args(ops: dict, ctx) -> tuple:
    return tuple(ops[n] for n in ("q", "k", "v", "g", "beta")) + (ctx,)


def _packed_qk(q: torch.Tensor, k: torch.Tensor, HV: int | None = None):
    """``q`` / ``k`` as column views of one packed ``[T, (2 H + HV) * 128]`` buffer (a fused projection or convolution
    output: q | k | v), with the same values; and the buffer."""
    T, H, _ = q.shape
    HV = H if HV is None else HV
    buf = torch.zeros(T, (2 * H + HV) * D, dtype=q.dtype, device=q.device)
    buf[:, : H * D] = q.reshape(T, H * D)
    buf[:, H * D : 2 * H * D] = k.reshape(T, H * D)
    return buf[:, : H * D].view(T, H, D), buf[:, H * D : 2 * H * D].view(T, H, D), buf


def _prepared(ctx, ops: dict, H: int, *, backward: bool, workspace=None, **kw):
    """A call prepared on the CPU (operands checked, workspace sized, nothing launched)."""
    call, _ = cake_kcp._prepare(
        *_args(ops, ctx),
        scale=None,
        save_mode="light",
        workspace=workspace,
        options={},
        backward=backward,
        dout=ops["dout"] if backward else None,
        **kw,
    )
    return call


def _cpu_saved(call, **extra):
    """A saved set of ``call``'s problem with the members the forward body gives it (zero tensors; the forward never
    runs on the CPU); ``extra``: further fields (the in-kernel L2 norm's)."""
    tb, opts = call.tb, call.opts

    def t(name: str) -> torch.Tensor:
        return torch.zeros(tb.sizes[name], dtype=call.state_dtype, device=call.device)

    return cake_kcp.SavedDetKcp(
        version=cake_kcp.KCP_SAVED_VERSION,
        S_in=t("S_in"),
        c1_recv=t("c1_recv") if opts.save_transfer else None,
        M_seg=t("stage_m") if (opts.save_transfer and tb.P > 1) else None,
        carry=t("carry") if (tb.P > 1 and call.saved_carry == 1) else None,
        T_r=call.T_r,
        H=call.H,
        n_frag=tb.n_frag,
        P=tb.P,
        Hg=tb.Hg,
        scale=call.scale,
        save_mode="light",
        save_transfer=opts.save_transfer,
        offset_class=opts.offset_class,
        exchange=call.exchange,
        precision_mode=call.precision_mode,
        ctx=call.ctx,
        transfer_walk=call.transfer_walk,
        **extra,
    )


def _sig(t) -> list:
    """A pointer-free tensor signature (shape, strides, dtype), comparable across processes and commits."""
    if not isinstance(t, torch.Tensor):
        return ["none"]
    return [list(t.shape), list(t.stride()), str(t.dtype)]


def _jsonable(x):
    if isinstance(x, torch.Tensor):
        return ["tensor"] + _sig(x)
    if isinstance(x, torch.dtype):
        return str(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if x is None or isinstance(x, (bool, int, float, str)):
        return x
    return repr(x)


_SAVED_ROLES = ("S_in", "c1_recv", "M_seg", "carry", "qk_rstd")


def _saved_summary(saved) -> dict:
    """A saved set's tensor list (each tensor's member name, shape, strides, dtype), flags and meta."""
    roles = []
    for t in saved.tensors():
        role = next(
            (n for n in _SAVED_ROLES if getattr(saved, n, None) is t), "unknown"
        )
        roles.append([role] + _sig(t))
    return {
        "tensors": roles,
        "flags": list(saved.flags()),
        "meta": _jsonable(saved.meta()),
    }


def _plan_recorder(name: str, real, calls: list, results: list | None = None):
    """``real`` (the existing backend's ``plan`` / ``workspace_bytes`` / ``tables_bytes``) recording its calls in
    ``calls`` and, with ``results``, each call's ``(name, args, keywords, result)``; a backend whose function does not
    take ``l2n`` yet is called without it (the flag-off arena: the recorded keywords are the point)."""
    takes_l2n = "l2n" in inspect.signature(real).parameters

    def wrapped(*a, **k):
        calls.append([name, _jsonable(a), _jsonable(sorted(k.items()))])
        if not takes_l2n:
            k = {x: v for x, v in k.items() if x != "l2n"}
        out = real(*a, **k)
        if results is not None:
            results.append((name, _jsonable(a), _jsonable(sorted(k.items())), out))
        return out

    return wrapped


def _recording(m, calls: list, results: list) -> None:
    """Patch (``m``) the existing backend's ``plan`` / ``workspace_bytes`` to record their calls in ``calls`` and,
    with its ``tables_bytes``, their results in ``results`` (:func:`_local_bytes`)."""
    for name in ("plan", "workspace_bytes", "tables_bytes"):
        log = [] if name == "tables_bytes" else calls
        real = getattr(cake_backend, name)
        m.setattr(cake_backend, name, _plan_recorder(name, real, log, results))


def _local_bytes(results: list) -> int:
    """The local part of one workspace byte query from the existing backend's ``results`` behind it
    (:func:`_recording`): the largest arena plus the scratch table set of every distinct problem that has one
    (``cake_kcp_plan.kda_det_workspace_bytes_kcp``; the rest, the KCP pools, does not depend on the l2n mode)."""
    arenas = [int(out) for name, _a, _k, out in results if name == "workspace_bytes"]
    tables = {
        json.dumps([a, k]): int(out)
        for name, a, k, out in results
        if name == "tables_bytes"
    }
    return max(arenas, default=0) + sum(tables.values())


def _snapshot(kw: dict, H: int = 4) -> dict:
    """What a call with the l2n keyword ``kw`` (``{}``: omitted, the only form the base of this change accepts) keys,
    fingerprints, saves and sizes, on the CPU and without device work: the forward and backward graph validity keys
    (pointer-free), the recompute handle's fingerprint of both tiers, the prepared call's operands, a saved set's
    tensors / flags / meta, the saved and workspace byte queries (contexts with and without tokens, and a list) and
    the existing backend's plan calls behind the workspace query."""
    l2n = bool(kw.get("use_qk_l2norm_in_kernel", False))
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    args = _args(ops, ctx)
    snap: dict = {}
    call = _prepared(ctx, ops, H, backward=False, **kw)
    snap["call"] = {
        "T_r": call.T_r,
        "H": call.H,
        "gate": call.gate,
        "ops": {n: _sig(t) for n, t in call.ops.items() if isinstance(t, torch.Tensor)},
    }
    if not kw:  # the dataclass defaults
        saved_kw: dict = {}
    elif l2n:
        saved_kw = {"qk_rstd": torch.zeros(2, ctx.T_r, H), "qk_l2norm": True}
    else:
        saved_kw = {"qk_rstd": None, "qk_l2norm": False}
    saved = _cpu_saved(call, **saved_kw)
    snap["saved"] = _saved_summary(saved)
    keys: list = []

    def entry(kind, ctx_, ws, opts, scale, tensors, flags):
        keys.append(
            {
                "kind": kind,
                "scale": scale,
                "flags": _jsonable(flags),
                "tensors": [_sig(t) for t in tensors],
                "options": [
                    [f.name, getattr(opts, f.name)] for f in dataclasses.fields(opts)
                ],
            }
        )
        raise _Stop

    fps: dict = {}
    real_fp = cake_kcp._handle_fingerprint

    def fingerprint(call_, **kw2):
        out = real_fp(call_, **kw2)
        fps[kw2["keep"]] = _jsonable(out)
        return out

    def stop(*_a, **_k):
        raise _Stop

    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp, "_graph_entry", entry)
        ws = cake_kcp.WorkspaceDetKcp(local=False, **kw)
        with pytest.raises(_Stop):
            cake_kcp.chunk_kda_det_fwd_kcp(*args, workspace=ws, graph="replay", **kw)
        with pytest.raises(_Stop):
            cake_kcp.chunk_kda_det_bwd_kcp(
                *args, ops["dout"], saved=saved, workspace=ws, graph="replay", **kw
            )
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp, "_handle_fingerprint", fingerprint)
        m.setattr(cake_kcp, "_fwd_body", stop)
        for keep in ("states", "transfer"):
            with pytest.raises(_Stop):
                cake_kcp.chunk_kda_det_kcp(
                    *args,
                    recompute_handle=cake_kcp.RecomputeHandleDetKcp(),
                    checkpoint_keep=keep,
                    **kw,
                )
    snap["graph_keys"] = keys
    snap["fingerprints"] = fps
    ctxs = [_cpu_ctx(rank=0), _cpu_ctx(rank=1), _cpu_zero_ctx(1), _cpu_zero_ctx(2)]
    snap["saved_bytes"] = [
        cake_kcp_plan.kda_det_saved_bytes_kcp(c, H=h, **extra, **kw)
        for c in ctxs
        for h in (4, 64)
        for extra in ({}, {"save_transfer": False}, {"checkpoint_keep": "states"})
    ]
    plan_calls: list = []
    snap["workspace_bytes"], snap["local_bytes"] = [], []
    with pytest.MonkeyPatch.context() as m:
        results: list = []
        _recording(m, plan_calls, results)
        for c in (ctxs[0], ctxs[2], ctxs):
            for h in (4, 64):
                for b in (True, False):
                    results.clear()
                    snap["workspace_bytes"].append(
                        cake_kcp_plan.kda_det_workspace_bytes_kcp(
                            c, H=h, backward=b, **kw
                        )
                    )
                    snap["local_bytes"].append(_local_bytes(results))
    snap["plan_calls"] = plan_calls
    return _jsonable(snap)


# The literal flag-off expectations of :func:`_snapshot` ({}), captured once by executing the base of this change on
# the CPU (no CUDA device: the byte queries plan for the first exported architecture) with this file copied onto it:
# ``_golden_view(_snapshot({}))``.  The capture's provenance is under ``"captured"``.
GOLDEN_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "test_cake_kda_det_train_kcp_l2n_golden.json",
)


def _golden_view(snap: dict) -> dict:
    """The parts of a :func:`_snapshot` the golden file holds: the graph keys' kind, flags and key tensor count, the
    fingerprint modes of both tiers, the saved set's tensor roles, flags and meta keys, the saved and workspace bytes
    and the existing backend's plan / byte calls behind the workspace query."""
    return {
        "graph_keys": [
            [k["kind"], k["flags"], len(k["tensors"])] for k in snap["graph_keys"]
        ],
        "fingerprint_modes": {
            keep: dict(fp)["modes"] for keep, fp in sorted(snap["fingerprints"].items())
        },
        "saved_roles": [t[0] for t in snap["saved"]["tensors"]],
        "saved_flags": snap["saved"]["flags"],
        "saved_meta_keys": sorted(snap["saved"]["meta"]),
        "saved_bytes": snap["saved_bytes"],
        "workspace_bytes": snap["workspace_bytes"],
        "plan_calls": snap["plan_calls"],
    }


def _golden() -> dict:
    """The committed golden expectations (:data:`GOLDEN_PATH`) without their provenance; where a CUDA device is
    visible, without the workspace bytes too (the existing backend then plans for the device's architecture, the
    golden for the first exported one)."""
    with open(GOLDEN_PATH) as fh:
        golden = json.load(fh)
    golden.pop("captured", None)
    if torch.cuda.is_available():
        golden.pop("workspace_bytes", None)
    return golden


def _as_golden(snap: dict) -> dict:
    """:func:`_golden_view` of ``snap`` in the golden file's JSON form, with the workspace bytes kept where
    :func:`_golden` keeps them."""
    view = json.loads(json.dumps(_golden_view(snap)))
    if torch.cuda.is_available():
        view.pop("workspace_bytes", None)
    return view


# --------------------------------------------------------------------------- CPU rows: the keyword


L2N_ENTRIES = (
    (cake_kcp, "chunk_kda_det_fwd_kcp"),
    (cake_kcp, "chunk_kda_det_bwd_kcp"),
    (cake_kcp, "chunk_kda_det_kcp"),
    (cake_kcp_plan, "kda_det_workspace_bytes_kcp"),
    (cake_kcp_plan, "kda_det_saved_bytes_kcp"),
)


def test_l2n_keyword_is_keyword_only_and_off_by_default() -> None:
    """``use_qk_l2norm_in_kernel`` is a keyword-only ``bool = False`` of the three entry points, ``WorkspaceDetKcp``
    and the two byte queries, and the autograd Function's last argument; it is an entry keyword, not an option:
    ``OptionsDetKcp`` (every field of which keys the graph cache and the handle fingerprint) has no such field and
    neither ``**options`` nor the private test knobs accept it.  The workspace's form knob is keyword-only, ``None``
    by default."""
    for module, name in L2N_ENTRIES:
        p = inspect.signature(getattr(module, name)).parameters[
            "use_qk_l2norm_in_kernel"
        ]
        assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False, name
    params = inspect.signature(cake_kcp.WorkspaceDetKcp).parameters
    p = params["use_qk_l2norm_in_kernel"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is False
    p = params["_qk_l2norm_form"]
    assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None
    assert cake_kcp.kda_det_saved_bytes_kcp is cake_kcp_plan.kda_det_saved_bytes_kcp
    assert (
        cake_kcp.kda_det_workspace_bytes_kcp
        is cake_kcp_plan.kda_det_workspace_bytes_kcp
    )
    params = inspect.signature(cake_kcp.ChunkKDADetFunctionKcp.forward).parameters
    assert list(params)[-1] == "use_qk_l2norm_in_kernel"
    assert params["use_qk_l2norm_in_kernel"].default is False
    fields = {f.name for f in dataclasses.fields(cake_kcp.OptionsDetKcp)}
    assert not any("l2n" in f or "l2norm" in f for f in fields), fields
    assert "use_qk_l2norm_in_kernel" not in cake_kcp._OPTION_KEYS
    with pytest.raises(TypeError, match="use_qk_l2norm_in_kernel"):
        cake_kcp._options({"use_qk_l2norm_in_kernel": True})
    with pytest.raises(ValueError):
        cake_kcp._options({"_kcp_test_knobs": {"use_qk_l2norm_in_kernel": True}})
    # the collectives do not depend on the mode
    assert "use_qk_l2norm_in_kernel" not in (
        inspect.signature(cake_kcp_plan.kda_det_comm_bytes_kcp).parameters
    )


@pytest.mark.parametrize("bad", (1, 0, None, "true", 1.0), ids=repr)
def test_l2n_keyword_refuses_non_bool(bad) -> None:
    """Every entry refuses a non-bool value with ``ValueError`` before it sizes or launches anything."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    args = _args(ops, ctx)
    kw = {"use_qk_l2norm_in_kernel": bad}
    calls = (
        lambda: cake_kcp.chunk_kda_det_fwd_kcp(*args, **kw),
        lambda: cake_kcp.chunk_kda_det_fwd_kcp(*args, graph="replay", **kw),
        lambda: cake_kcp.chunk_kda_det_bwd_kcp(*args, ops["dout"], **kw),
        lambda: cake_kcp.chunk_kda_det_kcp(*args, **kw),
        lambda: cake_kcp.chunk_kda_det_kcp(
            *args, recompute_handle=cake_kcp.RecomputeHandleDetKcp(), **kw
        ),
        lambda: cake_kcp.WorkspaceDetKcp(**kw),
        lambda: cake_kcp_plan.kda_det_workspace_bytes_kcp(ctx, H=H, **kw),
        lambda: cake_kcp_plan.kda_det_saved_bytes_kcp(ctx, H=H, **kw),
        lambda: cake_kcp_plan.check_qk_l2norm(bad),
    )
    for fn in calls:
        with pytest.raises(ValueError, match="use_qk_l2norm_in_kernel must be a bool"):
            fn()


def test_l2n_form_knob(capable) -> None:
    """The workspace's form (:attr:`WorkspaceDetKcp.l2n`, the existing backend's plan mode): ``"off"`` without the
    flag, ``"fused"`` with it, ``"fallback"`` only through ``_qk_l2norm_form`` with the flag; another form, or a form
    without the flag, is a ``ValueError``.  The plan keywords are none for ``"off"`` and ``l2n=<form>`` otherwise; a
    call takes its form from its workspace, and a recompute handle's fingerprint records it."""
    assert cake_kcp.WorkspaceDetKcp().l2n == "off"
    assert cake_kcp.WorkspaceDetKcp(**L2N).l2n == "fused"
    assert cake_kcp.WorkspaceDetKcp(**L2N, **FALLBACK).l2n == "fallback"
    assert cake_kcp.WorkspaceDetKcp(**L2N, _qk_l2norm_form="fused").l2n == "fused"
    with pytest.raises(ValueError, match="needs use_qk_l2norm_in_kernel=True"):
        cake_kcp.WorkspaceDetKcp(**FALLBACK)
    with pytest.raises(ValueError, match="form must be one of"):
        cake_kcp.WorkspaceDetKcp(**L2N, _qk_l2norm_form="separate")
    assert cake_kcp_plan.QK_L2NORM_FORMS == ("fused", "fallback")
    assert cake_kcp_plan.qk_l2norm_mode(False) == "off"
    assert cake_kcp_plan.qk_l2norm_mode(True) == "fused"
    assert cake_kcp_plan.qk_l2norm_plan_kwargs("off") == {}
    for form in ("fused", "fallback"):
        assert cake_kcp_plan.qk_l2norm_plan_kwargs(form) == {"l2n": form}
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    for ws_kw, call_kw, mode in (
        ({}, {}, "off"),
        (L2N, L2N, "fused"),
        ({**L2N, **FALLBACK}, L2N, "fallback"),
    ):
        ws = cake_kcp.WorkspaceDetKcp(**ws_kw)
        for backward in (False, True):
            call = _prepared(ctx, ops, H, backward=backward, workspace=ws, **call_kw)
            assert (call.l2n, call.l2n_mode) == (mode != "off", mode)
            want = {} if mode == "off" else {"l2n": mode}
            assert cake_kcp._l2n_modes(call) == want


# --------------------------------------------------------------------------- CPU rows: flag-off identity


def test_l2n_flag_off_identity() -> None:
    """The omitted keyword and an explicit ``False`` give the forward / backward graph keys, recompute-handle
    fingerprints (both tiers), saved-set tensors / flags / meta, saved and workspace bytes and existing-backend plan
    calls of the base of this change, as literal expectations (:data:`GOLDEN_PATH`, captured by executing the base),
    and the same operands as each other; nothing in them names the mode."""
    omitted, explicit = _snapshot({}), _snapshot(OFF)
    golden = _golden()
    assert _as_golden(omitted) == golden
    assert _as_golden(explicit) == golden
    assert omitted == explicit
    assert [k["kind"] for k in omitted["graph_keys"]] == ["fwd", "bwd"]
    assert sorted(omitted["fingerprints"]) == ["states", "transfer"]
    flat = json.dumps(omitted)
    for word in ("qk_l2norm", "qk_rstd", '"l2n"'):
        assert word not in flat, word


def test_l2n_flag_off_snapshot_matches_the_base() -> None:
    """Cluster gate hook: flag-off identity against the base of this change.  ``KCP_L2N_FLAG_OFF_CAPTURE=<path>``
    writes the omitted-keyword snapshot (:func:`_snapshot` uses only names the base has: run this test there with this
    file copied onto the base checkout); ``KCP_L2N_FLAG_OFF_BASELINE=<path>`` compares this head's snapshot with it.
    Skipped without either."""
    capture, baseline = os.environ.get(ENV_CAPTURE), os.environ.get(ENV_BASELINE)
    if not capture and not baseline:
        pytest.skip(f"set {ENV_CAPTURE} (on the base) or {ENV_BASELINE} (on this head)")
    snap = json.loads(json.dumps(_snapshot({})))
    if capture:
        with open(capture, "w") as fh:
            json.dump(snap, fh, indent=1, sort_keys=True)
    if baseline:
        with open(baseline) as fh:
            ref = json.load(fh)
        differ = sorted(k for k in set(snap) | set(ref) if snap.get(k) != ref.get(k))
        assert not differ, f"flag-off snapshot differs from the base in {differ}"


def test_l2n_flag_on_keys_and_fingerprint(capable) -> None:
    """The flag on adds exactly: ``"qk_l2norm"`` to both graph keys' flags and the saved ``qk_rstd`` to the backward's
    key tensors (last); ``l2n="fused"`` to the fingerprint's modes (both tiers; every other entry, the operands and
    the options included, is the flag-off one); the rstd (last) and ``"qk_l2norm": True`` to the saved set; ``8 T_r H``
    to the saved bytes (a handle's bytes unchanged); and ``l2n="fused"`` to every plan call of the workspace query
    (the same calls otherwise), whose bytes move by the existing backend's local arena delta of the mode only (zero
    while it has none), against the flag-off run and the literal flag-off expectations alike."""
    H = 4
    off, on = _snapshot(OFF, H), _snapshot(L2N, H)
    assert on["call"] == off["call"]
    for ko, kn in zip(off["graph_keys"], on["graph_keys"], strict=True):
        assert kn["flags"] == ko["flags"] + ["qk_l2norm"], kn["kind"]
        assert (kn["kind"], kn["scale"], kn["options"]) == (
            ko["kind"],
            ko["scale"],
            ko["options"],
        )
    fwd_o, bwd_o = off["graph_keys"]
    fwd_n, bwd_n = on["graph_keys"]
    T = int(_cpu_ctx().T_r)
    rstd = _sig(torch.zeros(2, T, H))
    assert fwd_n["tensors"] == fwd_o["tensors"]
    assert bwd_n["tensors"] == bwd_o["tensors"] + [rstd]
    for keep in ("states", "transfer"):
        fo, fn = dict(off["fingerprints"][keep]), dict(on["fingerprints"][keep])
        assert {k: v for k, v in fn.items() if k != "modes"} == {
            k: v for k, v in fo.items() if k != "modes"
        }
        assert fn["modes"] == sorted(fo["modes"] + [["l2n", "fused"]])
    so, sn = off["saved"], on["saved"]
    assert sn["tensors"] == so["tensors"] + [["qk_rstd"] + rstd]
    assert sn["flags"] == so["flags"]
    assert sn["meta"] == {**so["meta"], "qk_l2norm": True}
    ctxs = [_cpu_ctx(rank=0), _cpu_ctx(rank=1), _cpu_zero_ctx(1), _cpu_zero_ctx(2)]
    deltas = [
        (0 if extra.get("checkpoint_keep") else 8 * int(c.T_r) * h)
        for c in ctxs
        for h in (4, 64)
        for extra in ({}, {"save_transfer": False}, {"checkpoint_keep": "states"})
    ]
    got = [n - o for n, o in zip(on["saved_bytes"], off["saved_bytes"], strict=True)]
    assert got == deltas
    assert off["plan_calls"] and len(on["plan_calls"]) == len(off["plan_calls"])
    for co, cn in zip(off["plan_calls"], on["plan_calls"], strict=True):
        assert cn[:2] == co[:2]
        assert dict(cn[2]) == {**dict(co[2]), "l2n": "fused"}
    # the workspace bytes move by the local arena's delta only (the existing backend's own bytes of the mode, zero
    # while it has none; the KCP pools do not depend on it)
    local = [n - o for n, o in zip(on["local_bytes"], off["local_bytes"], strict=True)]
    pairs = zip(on["workspace_bytes"], off["workspace_bytes"], strict=True)
    assert [n - o for n, o in pairs] == local
    # the same deltas against the literal flag-off expectations of the base of this change
    golden, view = _golden(), _as_golden(on)
    assert [v[:2] for v in view["graph_keys"]] == [
        [kind, flags + ["qk_l2norm"]] for kind, flags, _n in golden["graph_keys"]
    ]
    assert [v[2] for v in view["graph_keys"]] == [
        n + (kind == "bwd") for kind, _f, n in golden["graph_keys"]
    ]
    for keep, modes in golden["fingerprint_modes"].items():
        assert view["fingerprint_modes"][keep] == sorted(modes + [["l2n", "fused"]])
    assert view["saved_roles"] == golden["saved_roles"] + ["qk_rstd"]
    assert view["saved_flags"] == golden["saved_flags"]
    assert view["saved_meta_keys"] == sorted(golden["saved_meta_keys"] + ["qk_l2norm"])
    base_saved = [n - d for n, d in zip(view["saved_bytes"], deltas, strict=True)]
    assert base_saved == golden["saved_bytes"]
    assert len(view["plan_calls"]) == len(golden["plan_calls"])
    for g, c in zip(golden["plan_calls"], view["plan_calls"], strict=True):
        assert c[:2] == g[:2] and dict(c[2]) == {**dict(g[2]), "l2n": "fused"}
    if "workspace_bytes" in golden:  # no CUDA device: the golden's architecture
        base_ws = [n - d for n, d in zip(view["workspace_bytes"], local, strict=True)]
        assert base_ws == golden["workspace_bytes"]


# --------------------------------------------------------------------------- CPU rows: the saved set


def test_l2n_saved_set_round_trip_and_checks(capable) -> None:
    """``SavedDetKcp``: the defaults (no rstd, the mode off, no meta key); with the mode the rstd is the last tensor,
    ``"qk_l2norm": True`` the one extra meta key, the flags and the version (2) unchanged, ``bytes`` ``8 T_r H`` more,
    and ``from_tensors`` reads it back (the autograd node's round trip); the empty ``[2, 0, H]`` rstd of a rank
    without tokens round-trips too.  The backward refuses a set of the other mode, a mode without rstd, an rstd
    without the mode and an rstd of another shape, dtype or layout."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    T = int(ctx.T_r)
    fwd = _prepared(ctx, ops, H, backward=False)
    off = _cpu_saved(fwd)
    assert off.qk_rstd is None and off.qk_l2norm is False
    assert "qk_l2norm" not in off.meta() and cake_kcp.KCP_SAVED_VERSION == 2
    rstd = torch.rand(2, T, H)
    # the flag-off set's own tensors plus the mode's fields: the mode only appends the rstd
    on = dataclasses.replace(off, qk_rstd=rstd, qk_l2norm=True)
    t_on, t_off = on.tensors(), off.tensors()
    assert len(t_on) == len(t_off) + 1 and t_on[-1] is rstd
    assert all(a is b for a, b in zip(t_on[:-1], t_off, strict=True))
    assert on.flags() == off.flags() and on.version == off.version
    assert on.meta() == {**off.meta(), "qk_l2norm": True}
    assert on.bytes == off.bytes + 8 * T * H
    back = cake_kcp.SavedDetKcp.from_tensors(t_on, on.flags(), on.meta(), ctx)
    assert back.qk_rstd is rstd and back.qk_l2norm is True and back.meta() == on.meta()
    assert all(a is b for a, b in zip(back.tensors(), t_on, strict=True))
    back_off = cake_kcp.SavedDetKcp.from_tensors(t_off, off.flags(), off.meta(), ctx)
    assert back_off.qk_rstd is None and back_off.qk_l2norm is False
    # the backward's checks
    bwd_on = _prepared(ctx, ops, H, backward=True, **L2N)
    bwd_off = _prepared(ctx, ops, H, backward=True)
    cake_kcp._check_saved(on, bwd_on)
    cake_kcp._check_saved(off, bwd_off)
    on_off = "use_qk_l2norm_in_kernel=True, the call runs use_qk_l2norm_in_kernel=False"
    off_on = "use_qk_l2norm_in_kernel=False, the call runs use_qk_l2norm_in_kernel=True"
    with pytest.raises(ValueError, match=on_off):
        cake_kcp._check_saved(on, bwd_off)
    with pytest.raises(ValueError, match=off_on):
        cake_kcp._check_saved(off, bwd_on)
    with pytest.raises(ValueError, match="needs the forward's qk_rstd"):
        cake_kcp._check_saved(_cpu_saved(fwd, qk_l2norm=True), bwd_on)
    with pytest.raises(ValueError, match="qk_rstd without"):
        cake_kcp._check_saved(_cpu_saved(fwd, qk_rstd=rstd), bwd_off)
    for bad in (
        torch.rand(2, T, H + 1),
        torch.rand(2, T + 1, H),
        rstd.double(),
        torch.rand(2, H, T).transpose(1, 2),
    ):
        with pytest.raises(ValueError, match="saved.qk_rstd"):
            cake_kcp._check_saved(_cpu_saved(fwd, qk_rstd=bad, qk_l2norm=True), bwd_on)
    # a rank without tokens: the empty rstd
    zctx = _cpu_zero_ctx()
    assert int(zctx.T_r) == 0
    zops = _cpu_ops(zctx, H)
    zfwd = _prepared(zctx, zops, H, backward=False, **L2N)
    empty = torch.empty(2, 0, H, dtype=torch.float32)
    z = _cpu_saved(zfwd, qk_rstd=empty, qk_l2norm=True)
    assert z.bytes == _cpu_saved(zfwd).bytes and z.tensors()[-1] is empty
    zback = cake_kcp.SavedDetKcp.from_tensors(z.tensors(), z.flags(), z.meta(), zctx)
    assert tuple(zback.qk_rstd.shape) == (2, 0, H)
    assert zback.meta()["qk_l2norm"] is True
    cake_kcp._check_saved(z, _prepared(zctx, zops, H, backward=True, **L2N))


# --------------------------------------------------------------------------- CPU rows: the strided-input rule


def _view_cases(T: int = 5, H: int = 3, HV: int = 2):
    """(name, view, T, H, accepted) for the strided-input rule (the single-rank rule)."""
    W = (2 * H + HV) * D
    packed = torch.zeros(T, W, dtype=torch.bfloat16)
    wide = torch.zeros(T, W + 8, dtype=torch.bfloat16)
    flat = torch.zeros(T * (H * D + 8) + 64, dtype=torch.bfloat16)
    one = torch.zeros(T * 2 * D, dtype=torch.bfloat16)
    return [
        ("contiguous", torch.zeros(T, H, D, dtype=torch.bfloat16), T, H, True),
        ("packed_q", packed[:, : H * D].view(T, H, D), T, H, True),
        ("packed_k", packed[:, H * D : 2 * H * D].view(T, H, D), T, H, True),
        ("packed_batch_squeezed", packed[:, : H * D].view(1, T, H, D)[0], T, H, True),
        ("token_stride_plus_8", flat.as_strided((T, H, D), (H * D + 8, D, 1)), T, H, True),
        ("one_head_any_head_stride", one.as_strided((T, 1, D), (2 * D, 7, 1)), T, 1, True),
        ("one_token_packed", packed[:1, : H * D].view(1, H, D), 1, H, True),
        ("empty_packed", packed[:0, : H * D].view(0, H, D), 0, H, True),
        ("empty_contiguous", torch.zeros(0, H, D, dtype=torch.bfloat16), 0, H, True),
        ("one_token_overlapping_rows", flat.as_strided((1, H, D), (8, D, 1)), 1, H, False),
        ("one_token_token_stride_3", flat.as_strided((1, H, D), (3, D, 1)), 1, H, False),
        ("empty_inner_stride_2", flat.as_strided((0, H, D), (H * D * 2, 2 * D, 2)), 0, H, False),
        ("empty_head_stride_256", torch.zeros(0, H, 2 * D, dtype=torch.bfloat16)[..., :D], 0, H, False),
        ("empty_overlapping_rows", flat.as_strided((0, H, D), (8, D, 1)), 0, H, False),
        ("empty_fp32", torch.zeros(0, H, D), 0, H, False),
        ("inner_stride_2", torch.zeros(T, H, 2 * D, dtype=torch.bfloat16)[..., ::2], T, H, False),
        ("head_stride_256", torch.zeros(T, H, 2 * D, dtype=torch.bfloat16)[..., :D], T, H, False),
        ("token_stride_4_mod_8", flat.as_strided((T, H, D), (H * D + 4, D, 1)), T, H, False),
        ("overlapping_rows", flat.as_strided((T, H, D), (8, D, 1)), T, H, False),
        ("base_plus_2_bytes", wide[:, 1 : 1 + H * D].view(T, H, D), T, H, False),
        ("base_plus_8_bytes", wide[:, 4 : 4 + H * D].view(T, H, D), T, H, False),
        ("fp32", torch.zeros(T, H, D), T, H, False),
        ("shape", torch.zeros(T, H + 1, D, dtype=torch.bfloat16), T, H, False),
    ]  # fmt: skip


@pytest.mark.parametrize("case", _view_cases(), ids=lambda c: c[0])
def test_l2n_view_rule(case) -> None:
    """The rule of :func:`cake_kcp._check_qk_view`: BF16 ``[T, H, 128]``, innermost stride 1, head stride exactly 128
    (``H > 1``), token stride a multiple of 8 elements and at least ``H * 128`` -- checked for every ``T``, zero and
    one included, so every rank refuses the same view -- and, where the tensor has elements, a 16-byte aligned base
    (the per-rank part, through the single-rank ``_check_qk_raw`` where the existing backend has it); the head stride
    of ``H == 1`` ignored and canonicalized on the same storage; ``ValueError`` otherwise.  An accepted view comes back
    with the same storage and elements (never a copy)."""
    name, x, T, H, accepted = case
    cpu = torch.device("cpu")
    if not accepted:
        with pytest.raises(ValueError):
            cake_kcp._check_qk_view(x, name, T, H, cpu)
        return
    out = cake_kcp._check_qk_view(x, name, T, H, cpu)
    assert out.data_ptr() == x.data_ptr()
    assert out.shape == x.shape and out.dtype == x.dtype
    if x.numel():
        token, head, inner = out.stride()
        assert (inner, head) == (1, D)
        # the single-rank rule may give a one-token view its canonical token stride
        assert token == x.stride(0) or (T == 1 and token == H * D)
        assert torch.equal(out, x)


def test_l2n_view_refusal_is_rank_independent() -> None:
    """A refused stride pattern is refused with the same message whatever the rank's token count (0, 1 or 5 tokens):
    the metadata checks read no token count, so a rank without tokens or with one token refuses a view exactly as the
    others do (the base-address alignment, a property of each rank's own storage, is the one per-rank check)."""
    H = 3
    flat = torch.zeros(2 * 5 * H * D + 64, dtype=torch.bfloat16)
    for strides in ((H * D, D, 2), (H * D, 2 * D, 1), (8, D, 1), (H * D + 4, D, 1)):
        messages = set()
        for T in (0, 1, 5):
            x = flat.as_strided((T, H, D), strides)
            with pytest.raises(ValueError) as info:
                cake_kcp._check_qk_view(x, "q", T, H, torch.device("cpu"))
            messages.add(str(info.value))
        assert len(messages) == 1, (strides, messages)


def test_l2n_operands_take_packed_views_without_a_copy(capable) -> None:
    """With the flag on, ``_operands`` takes the caller's packed q / k views on their own storage (no copy; also with
    a leading batch axis); without it the same views raise "must be contiguous", and a view that breaks the rule is
    refused by the entry point before anything is sized."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    q, k, buf = _packed_qk(ops["q"], ops["k"])
    dtypes = cake_kcp.get_backend().dtypes
    rest = (ops["v"], ops["g"], ops["beta"], ctx)
    out = cake_kcp._operands(q, k, *rest, dtypes=dtypes, qk_l2norm=True)
    assert out[0].data_ptr() == buf.data_ptr() and out[0].stride() == q.stride()
    assert out[1].data_ptr() == k.data_ptr() and out[1].stride() == k.stride()
    assert torch.equal(out[0], ops["q"]) and torch.equal(out[1], ops["k"])
    outb = cake_kcp._operands(
        q.unsqueeze(0), k.unsqueeze(0), *rest, dtypes=dtypes, qk_l2norm=True
    )
    assert outb[0].data_ptr() == q.data_ptr() and outb[0].stride() == q.stride()
    assert outb[-1] is True
    with pytest.raises(ValueError, match="must be contiguous"):
        cake_kcp._operands(q, k, *rest, dtypes=dtypes)
    bad = torch.zeros(int(ctx.T_r), H, 2 * D, dtype=torch.bfloat16)[..., :D]
    for qq, kk in ((bad, ops["k"]), (ops["q"], bad)):
        with pytest.raises(ValueError):
            cake_kcp.chunk_kda_det_fwd_kcp(qq, kk, *rest, **L2N)
    call = _prepared(ctx, {**ops, "q": q, "k": k}, H, backward=False, **L2N)
    assert call.ops["q"].data_ptr() == q.data_ptr()
    assert call.ops["k"].data_ptr() == k.data_ptr()


def test_l2n_refusal_precedes_the_operand_checks() -> None:
    """With the flag on, the capability refusal and the workspace's checks run before the operand checks, on
    rank-independent inputs only (the context's device, the workspace's modes): a rank whose q view is misaligned
    (the one per-rank view check) raises the others' ``NotImplementedError`` from every entry, not ``ValueError``,
    and a workspace that is not a ``WorkspaceDetKcp`` the others' ``TypeError``; past the check the same view raises
    ``ValueError``.  Without the flag the operand checks come first, as before the mode."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    T = int(ctx.T_r)
    wide = torch.zeros(T, H * D + 8, dtype=torch.bfloat16)
    q = wide[:, 1 : 1 + H * D].view(T, H, D)  # 2 bytes past the 16-byte alignment
    assert T > 0 and q.data_ptr() % 16 == 2
    args = (q, *_args(ops, ctx)[1:])
    dout = ops["dout"]
    entries = {
        "fwd": lambda **kw: cake_kcp.chunk_kda_det_fwd_kcp(*args, **kw),
        "bwd": lambda **kw: cake_kcp.chunk_kda_det_bwd_kcp(*args, dout, **kw),
        "autograd": lambda **kw: cake_kcp.chunk_kda_det_kcp(*args, **kw),
    }
    for entry in entries.values():
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_kcp_plan, "qk_l2norm_missing", _missing("forced"))
            with pytest.raises(NotImplementedError, match="forced"):
                entry(**L2N)
            with pytest.raises(TypeError, match="must be a WorkspaceDetKcp"):
                entry(workspace=object(), **L2N)
        with pytest.MonkeyPatch.context() as m:
            m.setattr(cake_kcp_plan, "qk_l2norm_missing", _missing(None))
            with pytest.raises(ValueError):
                entry(**L2N)
        for kw in ({}, {"workspace": object()}):
            with pytest.raises(ValueError, match="must be contiguous"):
                entry(**kw)


# --------------------------------------------------------------------------- CPU rows: workspace and byte queries


def test_l2n_workspace_mode_is_pinned(capable) -> None:
    """``WorkspaceDetKcp(use_qk_l2norm_in_kernel=)`` fixes the mode (:attr:`qk_l2norm`): a call in the other mode is
    refused in both directions and both entry points; ``workspace=None`` builds one in the call's mode; the mode
    composes with the raw gate mode; ``local_arena="lean"`` is refused with the mode (workspace and byte query) and
    still accepted without it; a backend without the local stages and the SPEC-3 probe refuse the mode."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    args = _args(ops, ctx)
    ws_off, ws_on = cake_kcp.WorkspaceDetKcp(), cake_kcp.WorkspaceDetKcp(**L2N)
    assert (ws_off.qk_l2norm, ws_on.qk_l2norm, ws_on.gate) == (False, True, "g")
    both = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True, **L2N)
    assert (both.gate, both.qk_l2norm, both.l2n) == ("raw", True, "fused")
    on_off = "allocated with use_qk_l2norm_in_kernel=True, the call runs use_qk_l2norm_in_kernel=False"
    off_on = "allocated with use_qk_l2norm_in_kernel=False, the call runs use_qk_l2norm_in_kernel=True"
    with pytest.raises(ValueError, match=off_on):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, workspace=ws_off, **L2N)
    with pytest.raises(ValueError, match=on_off):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, workspace=ws_on)
    with pytest.raises(ValueError, match=off_on):
        cake_kcp.chunk_kda_det_bwd_kcp(*args, ops["dout"], workspace=ws_off, **L2N)
    with pytest.raises(ValueError, match=on_off):
        cake_kcp.chunk_kda_det_bwd_kcp(*args, ops["dout"], workspace=ws_on)
    for kw, want in (({}, False), (OFF, False), (L2N, True)):
        for backward in (False, True):
            call = _prepared(ctx, ops, H, backward=backward, **kw)
            assert call.ws.qk_l2norm is want and call.l2n is want
            assert call.l2n_mode == ("fused" if want else "off")
            assert call.l2n_values is None
    with pytest.raises(NotImplementedError, match='local_arena="lean"'):
        cake_kcp.WorkspaceDetKcp(local_arena="lean", **L2N)
    assert cake_kcp.WorkspaceDetKcp(local_arena="lean").qk_l2norm is False
    for backward in (True, False):
        with pytest.raises(NotImplementedError, match='local_arena="lean"'):
            cake_kcp_plan.kda_det_workspace_bytes_kcp(
                ctx, H=H, backward=backward, local_arena="lean", **L2N
            )
    for probe in ("record", "warn", "assert"):
        with pytest.raises(NotImplementedError, match="spec3_probe"):
            cake_kcp.chunk_kda_det_fwd_kcp(*args, spec3_probe=probe, **L2N)
    stub = type("Stub", (cake_kcp.Backend,), {"name": "stub", "local_stages": False})()
    prev = cake_kcp.set_backend(stub)
    try:
        with pytest.raises(NotImplementedError, match="local kernel stages"):
            cake_kcp.chunk_kda_det_fwd_kcp(*args, **L2N)
    finally:
        cake_kcp.set_backend(prev)


def test_l2n_byte_queries(capable) -> None:
    """``kda_det_saved_bytes_kcp(..., use_qk_l2norm_in_kernel=True)`` is ``8 T_r H`` more than without (zero on a rank
    without tokens; every tier of a recompute handle unchanged); ``kda_det_workspace_bytes_kcp`` plans the local arena
    with ``l2n="fused"`` added to the same calls, for one context and for a list, and its bytes move by the existing
    backend's local arena delta of the mode only (the KCP pools unchanged); the communication query does not take
    the keyword."""
    ctxs = [_cpu_ctx(rank=0), _cpu_ctx(rank=1)] + [_cpu_zero_ctx(r) for r in range(4)]
    assert any(int(c.T_r) == 0 for c in ctxs)
    for c in ctxs:
        for H in (4, 64):
            for kw in ({}, {"save_transfer": False}, {"with_conv": False}):
                off = cake_kcp_plan.kda_det_saved_bytes_kcp(c, H=H, **kw)
                on = cake_kcp_plan.kda_det_saved_bytes_kcp(c, H=H, **kw, **L2N)
                assert on - off == 8 * int(c.T_r) * H, (c.rank, H, kw)
            for keep in ("states", "transfer"):
                assert cake_kcp_plan.kda_det_saved_bytes_kcp(
                    c, H=H, checkpoint_keep=keep, **L2N
                ) == cake_kcp_plan.kda_det_saved_bytes_kcp(c, H=H, checkpoint_keep=keep)
    for target in (ctxs[0], ctxs[1], ctxs):
        for H in (4, 64):
            for backward in (True, False):
                got = {}
                for tag, kw in (("off", {}), ("on", L2N)):
                    calls: list = []
                    results: list = []
                    with pytest.MonkeyPatch.context() as m:
                        _recording(m, calls, results)
                        n = cake_kcp_plan.kda_det_workspace_bytes_kcp(
                            target, H=H, backward=backward, **kw
                        )
                    got[tag] = (n, calls, _local_bytes(results))
                c_off, c_on = got["off"][1], got["on"][1]
                assert c_off and len(c_on) == len(c_off)
                for (n0, a0, k0), (n1, a1, k1) in zip(c_off, c_on, strict=True):
                    assert (n0, a0) == (n1, a1)
                    assert dict(k1) == {**dict(k0), "l2n": "fused"}
                # the local arena's delta only: the pools do not depend on the mode
                delta = got["on"][2] - got["off"][2]
                assert got["on"][0] - got["off"][0] == delta, (H, backward)


def test_l2n_byte_queries_device_and_gate() -> None:
    """With the flag off the byte queries never read a device architecture (``qk_l2norm_arch`` raising) and, where
    the existing backend plans nothing (the saved bytes of every rank, the workspace bytes of ranks without tokens),
    make no CUDA call at all (``torch.cuda.is_available`` raising), with unchanged bytes.  With the flag on, the
    workspace query checks the mode on the architecture of its contexts' device (not the current CUDA device) in its
    gate mode, the saved query on its context's device in the call's gate mode (``use_gate_in_kernel``, a bool)."""

    def no_cuda(*_a, **_k):
        raise AssertionError("a flag-off byte query read the device")

    def saved_bytes(cs) -> list:
        return [
            cake_kcp_plan.kda_det_saved_bytes_kcp(c, H=4, **kw)
            for c in cs
            for kw in ({}, OFF)
        ]

    def workspace_bytes(cs) -> list:
        return [
            cake_kcp_plan.kda_det_workspace_bytes_kcp(t, H=4, backward=b, **kw)
            for t in (*cs, cs)
            for b in (True, False)
            for kw in ({}, OFF)
        ]

    ctxs = [_cpu_ctx(rank=0), _cpu_ctx(rank=1), _cpu_zero_ctx(1), _cpu_zero_ctx(2)]
    zero = [c for c in ctxs if int(c.T_r) == 0]
    assert zero and len(zero) < len(ctxs)
    want = (saved_bytes(ctxs), workspace_bytes(ctxs), workspace_bytes(zero))
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp_plan, "qk_l2norm_arch", no_cuda)
        assert (saved_bytes(ctxs), workspace_bytes(ctxs)) == want[:2]
    with pytest.MonkeyPatch.context() as m:
        for name in ("is_available", "current_device", "get_device_capability"):
            m.setattr(torch.cuda, name, no_cuda)
        assert (saved_bytes(ctxs), workspace_bytes(zero)) == (want[0], want[2])
    checks: list = []
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp_plan, "qk_l2norm_arch", lambda device=None: f"arch:{device}")
        m.setattr(cake_kcp_plan, "require_qk_l2norm", lambda *a: checks.append(a))
        for gate in (False, True):
            cake_kcp_plan.kda_det_workspace_bytes_kcp(
                zero, H=4, use_gate_in_kernel=gate, **L2N
            )
            cake_kcp_plan.kda_det_saved_bytes_kcp(
                ctxs[0], H=4, use_gate_in_kernel=gate, **L2N
            )
    arch = f"arch:{ctxs[0].device}"
    assert checks == [("fused", g, arch) for g in ("g", "g", "raw", "raw")]
    for bad in (1, None, "raw"):
        with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
            cake_kcp_plan.kda_det_saved_bytes_kcp(ctxs[0], H=4, use_gate_in_kernel=bad)


# --------------------------------------------------------------------------- CPU rows: the values and the stage order


def _l2n_fake_call(
    *,
    T: int = 5,
    H: int = 3,
    backward: bool,
    mode: str = "fused",
    local=True,
):
    """A call-like object for :func:`cake_kcp._qk_l2norm_begin`: packed raw q / k and the arena layout's ``qn_int``
    / ``kn_int`` slot views ``[R, H, 128]`` (the forward and replay layouts carry both)."""
    gen = torch.Generator().manual_seed(17)
    q, k, _ = _packed_qk(
        torch.randn(T, H, D, generator=gen).to(torch.bfloat16),
        torch.randn(T, H, D, generator=gen).to(torch.bfloat16),
    )
    R = 64 * max(1, -(-T // 64))
    views = types.SimpleNamespace(
        qn_int=torch.zeros(R, H, D, dtype=torch.bfloat16),
        kn_int=torch.zeros(R, H, D, dtype=torch.bfloat16),
    )
    return types.SimpleNamespace(
        l2n=True, l2n_mode=mode, gate="g", T_r=T, H=H, backward=backward, l2n_values=None,
        backend=types.SimpleNamespace(dtypes=dict(cake_kcp.CONTRACT_DTYPES)),
        local=object() if local else None, device=torch.device("cpu"), ops={"q": q, "k": k},
        scratch={"views": views},
    )  # fmt: skip


def test_l2n_call_values_per_path() -> None:
    """The values a flag-on call binds for every local stage (:func:`cake_kcp._qk_l2norm_begin`, the existing
    backend's logical names): the raw views only under ``q_raw`` / ``k_raw`` with their element strides (head stride
    128), one FP32 ``qk_rstd`` ``[2, T_r, H]``, ``num_tokens``, the dq / dk row strides and the first reader's
    ``(read_rstd, write_rstd)``: saved-mode backward (1, 0) on the saved rstd, replay backward and saving forward
    (0, 1) on a fresh one (kept by the saving forward), fused forward without a saved set (0, 0) on a one-element
    tensor, fallback forward (0, 1) on a transient one.  The normalized rows: ``qn_int`` / ``kn_int`` the arena's
    slot views; a forward writes no token-layout rows (``write_qkn = 0``; ``qn`` / ``kn`` = ``q`` / ``k`` are the
    slot views, in the fallback form their first ``T_r`` rows); a backward binds the slot views as ``qn_int`` /
    ``kn_int`` (never a stand-in: the capability check refuses a layout without them) and its token-layout rows are
    a per-call contiguous BF16 pair (``write_qkn = 1``).  A rank without tokens binds nothing (a saving forward keeps
    the empty ``[2, 0, H]``).  The capability refusal is not here (it runs earlier, in ``_prepare``, on every rank):
    with the check failing the values are bound the same."""
    T, H = 5, 3
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp_plan, "qk_l2norm_missing", lambda *a, **k: "no records")
        saved_rstd = torch.rand(2, T, H)
        cases = (
            ("fwd_save", False, "fused", None, True, (0, 1)),
            ("fwd_nosave", False, "fused", None, False, (0, 0)),
            ("fwd_fallback", False, "fallback", None, False, (0, 1)),
            ("bwd_replay", True, "fused", None, False, (0, 1)),
            ("bwd_saved", True, "fused", saved_rstd, False, (1, 0)),
            ("bwd_fallback_saved", True, "fallback", saved_rstd, False, (1, 0)),
        )
        for name, backward, mode, rstd, save, rw in cases:
            call = _l2n_fake_call(T=T, H=H, backward=backward, mode=mode)
            kept = cake_kcp._qk_l2norm_begin(call, rstd, save=save)
            v = call.l2n_values
            views = call.scratch["views"]
            q, k = call.ops["q"], call.ops["k"]
            assert (v["read_rstd"], v["write_rstd"]) == rw, name
            assert call.scratch["l2n_rstd"] == rw, name
            assert v["q"] is v["qn"] and v["k"] is v["kn"], name
            assert v["q_raw"] is q and v["k_raw"] is k, name
            assert (v["q_raw_row_stride"], v["q_raw_head_stride"]) == (q.stride(0), D)
            assert (v["k_raw_row_stride"], v["k_raw_head_stride"]) == (k.stride(0), D)
            assert q.stride(0) == 3 * H * D  # the packed q | k | v row
            assert v["num_tokens"] == T
            assert v["dq_row_stride"] == v["dk_row_stride"] == H * D
            if backward:
                qn, kn = v["qn"], v["kn"]
                assert v["write_qkn"] == 1, name
                assert qn.shape == kn.shape == (T, H, D) and qn.dtype == torch.bfloat16
                assert qn.is_contiguous() and kn.is_contiguous()
                assert qn.data_ptr() != kn.data_ptr()
                assert v["qn_int"] is views.qn_int and v["kn_int"] is views.kn_int, name
                assert qn.data_ptr() != views.qn_int.data_ptr(), name
            else:
                assert v["write_qkn"] == 0, name
                assert v["qn_int"] is views.qn_int and v["kn_int"] is views.kn_int
                if mode == "fallback":
                    assert v["qn"].shape == (T, H, D)
                    assert v["qn"].data_ptr() == views.qn_int.data_ptr()
                    assert v["kn"].data_ptr() == views.kn_int.data_ptr()
                else:
                    assert v["qn"] is views.qn_int and v["kn"] is views.kn_int
            r = v["qk_rstd"]
            if rstd is not None:
                assert r is rstd, name
            elif rw == (0, 0):
                assert r.numel() == 1 and r.dtype == torch.float32, name
            else:
                assert tuple(r.shape) == (2, T, H) and r.dtype == torch.float32, name
            assert (kept is r) if save else (kept is None), name
        # a rank without tokens / without local stages
        for local, T0 in ((True, 0), (False, T)):
            call = _l2n_fake_call(T=T0, H=H, backward=False, local=local)
            kept = cake_kcp._qk_l2norm_begin(call, save=True)
            assert tuple(kept.shape) == (2, T0, H) and kept.dtype == torch.float32
            assert call.l2n_values is None and "l2n_rstd" not in call.scratch
            call = _l2n_fake_call(T=T0, H=H, backward=True, local=local)
            assert cake_kcp._qk_l2norm_begin(call) is None and call.l2n_values is None


def test_l2n_stage_order() -> None:
    """The backward's rows before its walks (:func:`cake_kcp._bwd_rows`): flag off, the prep, the intra program, the
    WY program (as before the mode); fused, the plan's first q / k reader first (the intra program where its record is
    the ``_l2n`` twin, else the prep); fallback, ``qk_apply`` (from a saved rstd) or ``qk_l2norm`` before the
    unchanged order.  :func:`cake_kcp._qk_l2norm_stage` launches nothing in the fused form."""
    seq: list = []

    class Plan:
        def __init__(self, intra: str) -> None:
            self.intra = intra

        def record(self, stage, path):
            assert (stage, path) == ("fwd_intra", cake_kcp.LOCAL_PATH_BWD)
            return self.intra

    def call(mode: str, intra: str, rw=None):
        c = types.SimpleNamespace(
            l2n=mode != "off",
            l2n_mode=mode,
            local=types.SimpleNamespace(plan=Plan(intra)),
            scratch={} if rw is None else {"l2n_rstd": rw},
        )
        c.stage = lambda name, **extra: seq.append((name, extra.get("saved", "-")))
        return c

    saved = object()
    cases = (
        (("off", "fwd_intra_v14"), saved, ["bwd_prep", "fwd_intra", "wy"]),
        (("fused", "fwd_intra_v14_l2n", (1, 0)), saved, ["fwd_intra", "bwd_prep", "wy"]),
        (("fused", "fwd_intra_l2n", (0, 1)), None, ["fwd_intra", "bwd_prep", "wy"]),
        (("fused", "fwd_intra", (1, 0)), saved, ["bwd_prep", "fwd_intra", "wy"]),
        (("fallback", "fwd_intra_v14", (1, 0)), saved, ["qk_apply", "bwd_prep", "fwd_intra", "wy"]),
        (("fallback", "fwd_intra", (0, 1)), None, ["qk_l2norm", "bwd_prep", "fwd_intra", "wy"]),
    )  # fmt: skip
    for args, sv, want in cases:
        seq.clear()
        cake_kcp._bwd_rows(call(*args), sv)
        assert [n for n, _ in seq] == want, args
        assert [s for n, s in seq if n == "bwd_prep"] == [sv]
    seq.clear()
    cake_kcp._qk_l2norm_stage(call("fused", "fwd_intra_l2n", (0, 1)))
    assert seq == []
    assert cake_kcp_plan.qk_l2norm_intra_first(Plan("fwd_intra_v14_l2n"))
    assert not cake_kcp_plan.qk_l2norm_intra_first(Plan("fwd_intra_v14"))


def _fake_pick(stage: str, path: str, mode: str, docs: int, heads: int, arch: str):
    """The fake registry's selection, shaped as the existing backend's: the intra programs switch above 256
    documents; fused, the forward's intra program is the ``_l2n`` twin and on the replay the intra program (at most
    256 documents) or the prep (above) is; the GLM intra program per architecture (fused, its twin; fallback, its
    capture form); the fallback form's separate programs (none in the fused form)."""
    few = docs <= 256
    twin = "_l2n" if mode == "fused" else ""
    if stage == "fwd_intra":
        first = path == cake_kcp.LOCAL_PATH_FWD or few
        return ("fwd_intra_v14" if few else "fwd_intra") + (twin if first else "")
    if stage == "bwd_prep":
        return "bwd_prep_g8_replay" + ("" if few else twin)
    if stage == "glm_intra":
        glm = {"sm_100a": "glm_intra", "sm_103a": "glm_intra_v14"}
        return glm.get(arch, "glm_intra_v18") + (twin or "_cap")
    return stage if mode == "fallback" else None


def _pick_but(when, out):
    """:func:`_fake_pick`, but ``out`` (a record, ``None`` or an exception to raise) where ``when(stage, path, mode,
    docs, heads, arch)``."""

    def pick(*key):
        return out if when(*key) else _fake_pick(*key)

    return pick


def _fake_registry(
    m,
    *,
    pick=None,
    select_modes=("l2n", "gate"),
    arches=None,
    unexported=(),
    layouts=None,
    seen=None,
):
    """Install a fake existing backend whose entry points and plan take the mode: its ``select_record`` takes the
    keywords ``select_modes`` of ``("l2n", "gate")`` and selects ``pick(stage, path, mode, docs, heads, arch)``
    (default :func:`_fake_pick`; an exception is raised), each call's keywords appended to ``seen``; its records
    are delivered for ``arches`` (record -> architectures; default all) and exported but for the ``unexported``
    libraries (``lib_<record>_<arch>``); its layout tables are ``layouts`` (layout -> buffer names, or an exception
    to raise; default the slot views everywhere)."""

    def fwd(*a, use_qk_l2norm_in_kernel: bool = False, **k):
        raise AssertionError("never called")

    def plan(S, N, H, sm, arch=None, *, gate="g", l2n="off"):
        raise AssertionError("never called")

    def lifetime_table(
        mode="bwd", tail_passes=1, *, plan=None, gate_fused=None, gate=None, l2n=None
    ):
        got = (layouts or {}).get(mode, cake_kcp_plan.QK_L2NORM_SLOTS)
        if isinstance(got, Exception):
            raise got
        return [{"slot": i, "buffer": b} for i, b in enumerate(("u", "w", *got))]

    def select(stage, **kw):
        if seen is not None:
            seen.append({"stage": stage, **kw})
        key = (kw["path"], kw.get("l2n", "off"), kw["num_seqs"], kw["num_heads"])
        got = (pick or _fake_pick)(stage, *key, kw["arch"])
        if isinstance(got, Exception):
            raise got
        return got

    names = ("num_chunks_pad", "num_seqs", "num_heads", "sm_count", "arch", "path")
    kw_only = inspect.Parameter.KEYWORD_ONLY
    params = [inspect.Parameter("stage", inspect.Parameter.POSITIONAL_ONLY)]
    params += [inspect.Parameter(n, kw_only) for n in names + tuple(select_modes)]
    select.__signature__ = inspect.Signature(params)

    class Modules(dict):
        def __missing__(self, name):
            return {"arches": list((arches or {}).get(name, cake_jit.ARCHES))}

    class Libraries:
        def __contains__(self, lib):
            return lib not in unexported

    m.setattr(cake_backend, "chunk_kda_det_fwd", fwd)
    m.setattr(cake_backend, "plan", plan)
    m.setattr(cake_backend, "lifetime_table", lifetime_table)
    m.setattr(cake_jit, "L2N_MODES", ("off", "fused", "fallback"), raising=False)
    m.setattr(cake_jit, "select_record", select)
    m.setattr(cake_jit, "MODULES", Modules())
    m.setattr(cake_jit, "library_name", lambda name, arch: f"lib_{name}_{arch}")
    m.setattr(cake_build, "libraries_by_name", Libraries)


def test_l2n_capability_check() -> None:
    """:func:`cake_kcp_plan.qk_l2norm_missing` (form, gate, arch): rank-independent inputs only (no context, no token
    count, no plan).  It selects through the existing backend's ``select_record(..., l2n=form)`` (and ``gate=`` in
    the raw gate mode) every stage of :data:`cake_kcp_plan.QK_L2NORM_STAGES` of the form on the route's two local
    paths, over the fixed grid :data:`cake_kcp_plan.QK_L2NORM_GRID` at each SM count of the architecture
    (:data:`cake_kcp_plan.QK_L2NORM_SMS`; every exported architecture for ``None``), each problem once; the grid
    holds both document regimes (at most and above 256) on both recurrence routes (segmented and serial) on every
    architecture.  A reason while the existing backend's entry points lack ``use_qk_l2norm_in_kernel``, its plan or
    layout table lacks ``l2n`` or the mode, its selection takes no ``l2n`` (the base of this change) or no ``gate``
    in the raw gate mode, a stage selects nothing (``ValueError`` or ``None``) for any problem of the grid -- also
    one only above or at most 256 documents, or on one architecture --, selects a record not delivered or not
    exported for the architecture, the fused form's readers do not give exactly one normalizing first reader on
    each path (the fallback form's none), or a layout of the route lacks the ``qn_int`` / ``kn_int`` views (named
    in the reason); ``None`` once all hold.  The signature reads are cached per callable."""
    assert list(inspect.signature(cake_kcp_plan.qk_l2norm_missing).parameters) == [
        "mode",
        "gate",
        "arch",
    ]
    F, B = cake_kcp.LOCAL_PATH_FWD, cake_kcp.LOCAL_PATH_BWD
    readers = ((F, "fwd_intra"), (B, "fwd_intra"), (B, "bwd_prep"), (B, "glm_intra"))
    separate = ((F, "qk_l2norm"), (B, "qk_l2norm"), (B, "qk_apply"), (B, "qk_pullback"))
    stages = cake_kcp_plan.QK_L2NORM_STAGES
    assert stages == {"fused": readers, "fallback": readers + separate}
    assert cake_kcp_plan.QK_L2NORM_LAYOUTS == ("fwd", "fwd_kcp", "replay")
    # the grid: both document regimes on both routes, on every architecture at each of its SM counts
    grid, sms = cake_kcp_plan.QK_L2NORM_GRID, cake_kcp_plan.QK_L2NORM_SMS
    assert set(sms) == set(cake_jit.ARCHES)
    both = {(few, route) for few in (True, False) for route in ("seg", "serial")}
    for arch in cake_jit.ARCHES:
        for sm in sms[arch]:
            regimes = {
                (docs <= 256, cake_jit.segment_plan(docs, heads, sm, arch)[0])
                for _nc, docs, heads in grid
            }
            assert regimes == both, (arch, sm, regimes)
    # every stage of the form, every problem, every SM count, once; the mode keywords
    fields = "arch sm_count num_chunks_pad num_seqs num_heads path stage".split()
    for form in ("fused", "fallback"):
        for gate in ("g", "raw"):
            for arch in (None, "sm_103a"):
                seen: list = []
                with pytest.MonkeyPatch.context() as m:
                    _fake_registry(m, seen=seen)
                    assert cake_kcp_plan.qk_l2norm_missing(form, gate, arch) is None
                want = {
                    (a, sm, *problem, path, stage)
                    for a in (cake_jit.ARCHES if arch is None else (arch,))
                    for sm in sms[a]
                    for problem in grid
                    for path, stage in stages[form]
                }
                got = [tuple(k[f] for f in fields) for k in seen]
                assert len(got) == len(want) and set(got) == want, (form, gate, arch)
                assert {k["l2n"] for k in seen} == {form}
                gates = {k.get("gate", "-") for k in seen}
                assert gates == {"-" if gate == "g" else gate}
    where = "sm_100a, 148 SMs, 257 documents, 4 heads, 257 chunks"
    bad = ValueError("no record serves the mode")
    cases = [
        # (the fake registry's keywords, only this form, the reason holds)
        ({}, None, None),
        ({"select_modes": ("gate",)}, None, "selects no record by Q/K L2 norm mode"),
        (
            {"pick": _pick_but(lambda s, *k: s == "glm_intra", bad)},
            None,
            f"selects no glm_intra on {B} (sm_100a, 148 SMs, 1 documents, 4 heads, 1 chunks): {bad}",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, d, h, a: s == "fwd_intra" and d > 256, bad)},
            None,
            f"selects no fwd_intra on {F} ({where}): {bad}",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, d, h, a: s == "bwd_prep" and d <= 256, None)},
            None,
            f"selects no bwd_prep on {B} (sm_100a, 148 SMs, 1 documents, 4 heads, 1 chunks)",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, d, h, a: s == "glm_intra" and h == 16 and a == "sm_107a", bad)},
            None,
            f"selects no glm_intra on {B} (sm_107a, 212 SMs, 1 documents, 16 heads, 1 chunks)",
        ),
        ({"pick": _pick_but(lambda s, *k: s == "qk_pullback", None)}, "fallback", f"selects no qk_pullback on {B}"),
        (
            {"pick": _pick_but(lambda s, p, m, d, h, a: s == "qk_apply" and d > 256, None)},
            "fallback",
            f"selects no qk_apply on {B} ({where})",
        ),
        ({"pick": _pick_but(lambda s, p, *k: s == "qk_l2norm" and p == F, bad)}, "fallback", f"qk_l2norm on {F}"),
        # the fused form's first readers: both / neither on the replay, a plain forward intra program
        (
            {"pick": _pick_but(lambda s, p, m, *k: s == "bwd_prep" and m == "fused", "bwd_prep_g8_replay_l2n")},
            "fused",
            "2 of the replay's records fwd_intra_v14_l2n / bwd_prep_g8_replay_l2n normalize",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, *k: s == "bwd_prep" and m == "fused", "bwd_prep_g8_replay")},
            "fused",
            f"registry ({where}): 0 of the replay's records fwd_intra / bwd_prep_g8_replay normalize",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, d, h, a: s == "fwd_intra" and p == F and d > 256, "fwd_intra")},
            "fused",
            f"registry ({where}): the forward's intra record fwd_intra does not normalize",
        ),
        (
            {"pick": _pick_but(lambda s, p, m, *k: s == "bwd_prep" and m == "fallback", "bwd_prep_g8_replay_l2n")},
            "fallback",
            "the fallback form's readers fwd_intra_v14 / fwd_intra_v14 / bwd_prep_g8_replay_l2n normalize",
        ),
        ({"layouts": {"replay": ("qn_int",)}}, None, "layout 'replay' has no kn_int views"),
        ({"layouts": {"fwd_kcp": ()}}, None, "layout 'fwd_kcp' has no qn_int / kn_int views"),
        ({"layouts": {"fwd": ValueError("bad mode")}}, None, "layout 'fwd' has no l2n="),
    ]  # fmt: skip
    for form in ("fused", "fallback"):
        for kw, only, want in cases:
            with pytest.MonkeyPatch.context() as m:
                _fake_registry(m, **kw)
                got = cake_kcp_plan.qk_l2norm_missing(form, "g")
            if want is None or (only is not None and only != form):
                assert got is None, (form, kw, got)
            else:
                assert got is not None and want in got, (form, kw, got)
    # the architecture: the selected records' "arches" and the build.json libraries there
    what = f"fwd_intra on {F} (sm_103a, 148 SMs, 1 documents, 4 heads, 1 chunks)"
    deliver = {"fwd_intra_v14_l2n": ["sm_100a", "sm_107a"]}
    for name, kw in (
        ("is not delivered", {"arches": deliver}),
        ("has no exported library", {"unexported": ("lib_fwd_intra_v14_l2n_sm_103a",)}),
    ):
        want = f"record fwd_intra_v14_l2n ({what}) {name} for sm_103a"
        with pytest.MonkeyPatch.context() as m:
            _fake_registry(m, **kw)
            for arch in ("sm_100a", "sm_107a"):
                assert cake_kcp_plan.qk_l2norm_missing("fused", "g", arch) is None
            assert cake_kcp_plan.qk_l2norm_missing("fallback", "g", "sm_103a") is None
            for arch in ("sm_103a", None):
                assert cake_kcp_plan.qk_l2norm_missing("fused", "g", arch) == want
    # the gate mode: the raw gate mode needs a selection by gate mode
    raw = "the single-rank registry selects no record of gate mode 'raw'"
    for form in ("fused", "fallback"):
        with pytest.MonkeyPatch.context() as m:
            _fake_registry(m, select_modes=("l2n",))
            assert cake_kcp_plan.qk_l2norm_missing(form, "g") is None
            assert cake_kcp_plan.qk_l2norm_missing(form, "raw") == raw
    # the signatures and the modes
    for form in ("fused", "fallback"):
        for gate in ("g", "raw"):
            with pytest.MonkeyPatch.context() as m:
                _fake_registry(m)
                assert cake_kcp_plan.qk_l2norm_missing(form, gate) is None
                hits = cake_kcp_plan._takes.cache_info().hits
                assert cake_kcp_plan.qk_l2norm_missing(form, gate) is None
                assert cake_kcp_plan._takes.cache_info().hits > hits

                def plain(S, N, H, sm, arch=None, *, gate="g"):
                    raise AssertionError("never called")

                m.setattr(cake_backend, "plan", plain)
                assert "plan" in cake_kcp_plan.qk_l2norm_missing(form, gate)
                _fake_registry(m)
                m.setattr(cake_jit, "L2N_MODES", ("off", "fused"), raising=False)
                if form == "fallback":
                    assert "plan" in cake_kcp_plan.qk_l2norm_missing(form, gate)
                m.setattr(cake_backend, "chunk_kda_det_fwd", plain)
                assert "use_qk_l2norm_in_kernel" in cake_kcp_plan.qk_l2norm_missing(
                    form, gate
                )
    assert cake_kcp_plan.qk_l2norm_missing("off", "g") is None
    cake_kcp_plan.require_qk_l2norm("off")
    with pytest.MonkeyPatch.context() as m:
        m.setattr(cake_kcp_plan, "qk_l2norm_missing", lambda *a, **k: "why")
        with pytest.raises(NotImplementedError) as info:
            cake_kcp_plan.require_qk_l2norm("fused")
        assert str(info.value) == f"{cake_kcp_plan.QK_L2NORM_PENDING} (why)"


def test_l2n_stage_values_merge() -> None:
    """``KernelBackend.stage`` puts the call's values of the mode (``call.l2n_values``: one dict for every stage)
    over the call's values and under the stage's own extras; a call without them (``None``, or an object without the
    attribute) launches exactly the values it launched before the mode existed."""
    launches: list = []

    class Plan:
        def record(self, stage, path):
            return f"rec_{stage}"

        def grid(self, stage, path):
            return (1, 1, 1)

        def dispatched(self, stage, path):
            return True

    class Arena:
        plan = Plan()
        device_index = 0

        def launcher(self, record, values, *, grid, device_index):
            launches.append((record, dict(values)))

    base = {"q": "raw_q", "k": "raw_k", "x": 1}

    def call(**attrs):
        return types.SimpleNamespace(
            local_values=dict(base), scratch={"views": None}, local=Arena(), out=None, grads=None,
            backward=True, host_trim=0, **attrs,
        )  # fmt: skip

    backend = cake_kcp.KernelBackend(programs=object())
    mode = {"q": "qn", "k": "kn", "q_raw": "raw_q", "k_raw": "raw_k", "qk_rstd": "r"}
    backend.stage("dqkg", call())
    backend.stage("dqkg", call(l2n_values=None))
    backend.stage("dqkg", call(l2n_values=mode))
    backend.stage("glm_intra", call(l2n_values=mode))
    backend.stage("dqkg", call(l2n_values={"q": "qn", "x": 2}), x=3)
    assert launches[0] == ("rec_dqkg", base) and launches[1] == ("rec_dqkg", base)
    assert launches[2] == ("rec_dqkg", {**base, **mode})
    assert launches[3] == ("rec_glm_intra", {**base, **mode})
    assert launches[4] == ("rec_dqkg", {**base, "q": "qn", "x": 3})


def test_l2n_fallback_stages_are_never_skipped() -> None:
    """``KernelBackend.stage`` launches a stage of the fallback form (``qk_l2norm`` / ``qk_apply`` / ``qk_pullback``,
    :data:`cake_kcp.L2N_FALLBACK_STAGES`) the plan dispatches and raises ``RuntimeError`` for one it does not, on
    the forward and the backward path, where an optional stage the plan fuses away (the gate epilogue) launches
    nothing."""
    launches: list = []

    class Plan:
        def __init__(self, on: bool) -> None:
            self.on = on

        def record(self, stage, path):
            return f"rec_{stage}"

        def grid(self, stage, path):
            return (1, 1, 1)

        def dispatched(self, stage, path):
            return self.on

    class Arena:
        device_index = 0

        def __init__(self, on: bool) -> None:
            self.plan = Plan(on)

        def launcher(self, record, values, *, grid, device_index):
            launches.append(record)

    def call(on: bool, backward: bool):
        return types.SimpleNamespace(
            local_values={}, scratch={"views": None}, local=Arena(on), out=None, grads=None,
            backward=backward, host_trim=0,
        )  # fmt: skip

    backend = cake_kcp.KernelBackend(programs=object())
    assert cake_kcp.L2N_FALLBACK_STAGES == ("qk_l2norm", "qk_apply", "qk_pullback")
    for name in cake_kcp.L2N_FALLBACK_STAGES:
        for backward in (False, True):
            path = cake_kcp.LOCAL_PATH_BWD if backward else cake_kcp.LOCAL_PATH_FWD
            launches.clear()
            backend.stage(name, call(True, backward))
            assert launches == [f"rec_{name}"], (name, path)
            with pytest.raises(RuntimeError, match=f"dispatches no {name} on '{path}'"):
                backend.stage(name, call(False, backward))
            assert launches == [f"rec_{name}"], (name, path)
    launches.clear()
    backend.stage("gate_epilogue", call(False, True))
    assert launches == []


# PENDING(single-rank in-kernel Q/K L2 norm programs): delete these rows once the existing backend serves the mode
# (they skip then; the W = 2 / 4 refusal rows force a reason and stay); the record-coverage row and the GPU rows below
# take over.
def _never(name: str):
    def fn(*_a, **_k):
        raise AssertionError(f"{name} reached before the in-kernel Q/K L2 norm refusal")

    return fn


def test_l2n_kernel_path_raises_until_bound() -> None:
    """While the existing backend lacks the mode's programs, every flag-on entry raises ``NotImplementedError`` with
    one message (:data:`cake_kcp_plan.QK_L2NORM_PENDING` and the reason) on a rank with tokens and on a rank without,
    before any workspace sizing, local arena plan or bind, existing-backend plan or collective (those raise
    ``AssertionError`` here): a saving and a replay-mode forward, a replay-mode and a saved-mode backward, the
    autograd entry with and without a recompute handle (each with a fresh workspace and with a flag-on workspace
    checked on another architecture), ``WorkspaceDetKcp`` with and without a context, that workspace's ``ensure`` /
    ``reserve`` on the CPU and on a CUDA device (its local arena bind replaced by a raiser), and the unpatched
    workspace and saved byte queries (one context and a list)."""
    missing = _l2n_missing()
    if missing is None:
        pytest.skip(
            "the existing backend has the in-kernel Q/K L2 norm programs: delete this row"
        )
    real = cake_kcp_plan.qk_l2norm_missing
    want = f"{cake_kcp_plan.QK_L2NORM_PENDING} ({real('fused', 'g', None)})"
    want_dev = f"{cake_kcp_plan.QK_L2NORM_PENDING} ({real('fused', 'g', 'sm_100a')})"
    H = 4
    ctxs = (_cpu_ctx(), _cpu_zero_ctx())
    for ctx in ctxs:
        ops = _cpu_ops(ctx, H)
        args = _args(ops, ctx)
        with pytest.MonkeyPatch.context() as m:
            # a flag-on workspace and saved set built where the check passes: on another architecture
            m.setattr(cake_kcp_plan, "qk_l2norm_arch", lambda device=None: "sm_103a")
            m.setattr(
                cake_kcp_plan,
                "qk_l2norm_missing",
                lambda mode="fused", gate="g", arch=None: (
                    None if arch == "sm_103a" else real(mode, gate, arch)
                ),
            )
            ws = cake_kcp.WorkspaceDetKcp(**L2N)
            fwd = _prepared(ctx, ops, H, backward=False, workspace=ws, **L2N)
            rstd = torch.zeros(2, int(ctx.T_r), H)
            saved = _cpu_saved(fwd, qk_rstd=rstd, qk_l2norm=True)
        with pytest.MonkeyPatch.context() as m:
            for name in ("tables", "_size_pools", "_bind_local", "_local_plan"):
                m.setattr(cake_kcp.WorkspaceDetKcp, name, _never(name))
            for name in ("plan", "workspace_bytes", "KdaDetWorkspace"):
                m.setattr(cake_backend, name, _never(f"cake_backend.{name}"))
            # a CUDA device reads as sm_100a (no device is touched: the refusal comes first)
            m.setattr(
                cake_kcp_plan,
                "qk_l2norm_arch",
                lambda device=None: (
                    "sm_100a"
                    if device is not None and torch.device(device).type == "cuda"
                    else None
                ),
            )
            dout = ops["dout"]
            handle = cake_kcp.RecomputeHandleDetKcp
            calls = {
                "fwd": lambda: cake_kcp.chunk_kda_det_fwd_kcp(*args, **L2N),
                "fwd_ws": lambda: cake_kcp.chunk_kda_det_fwd_kcp(
                    *args, workspace=ws, **L2N
                ),
                "fwd_replay": lambda: cake_kcp.chunk_kda_det_fwd_kcp(
                    *args, boundary_mode="replay", **L2N
                ),
                "bwd_replay": lambda: cake_kcp.chunk_kda_det_bwd_kcp(
                    *args, dout, **L2N
                ),
                "bwd_saved_ws": lambda: cake_kcp.chunk_kda_det_bwd_kcp(
                    *args, dout, saved=saved, workspace=ws, **L2N
                ),
                "autograd": lambda: cake_kcp.chunk_kda_det_kcp(*args, **L2N),
                "autograd_ws": lambda: cake_kcp.chunk_kda_det_kcp(
                    *args, workspace=ws, **L2N
                ),
                "autograd_handle": lambda: cake_kcp.chunk_kda_det_kcp(
                    *args, recompute_handle=handle(), **L2N
                ),
                "workspace": lambda: cake_kcp.WorkspaceDetKcp(**L2N),
                "workspace_ctx": lambda: cake_kcp.WorkspaceDetKcp(ctx, H, **L2N),
                "workspace_ctx_lazy": lambda: cake_kcp.WorkspaceDetKcp(
                    ctx, H, allocate=False, **L2N
                ),
                "ensure": lambda: ws.ensure(ctx, H),
                "reserve": lambda: ws.reserve([ctx], H),
                "workspace_bytes": lambda: cake_kcp_plan.kda_det_workspace_bytes_kcp(
                    ctx, H=H, **L2N
                ),
                "workspace_bytes_list": lambda: (
                    cake_kcp_plan.kda_det_workspace_bytes_kcp(list(ctxs), H=H, **L2N)
                ),
                "saved_bytes": lambda: cake_kcp_plan.kda_det_saved_bytes_kcp(
                    ctx, H=H, **L2N
                ),
            }
            for name, fn in calls.items():
                with pytest.raises(NotImplementedError) as info:
                    fn()
                assert str(info.value) == want, (name, int(ctx.T_r), str(info.value))
            cuda = torch.device("cuda", 0)
            for name, fn in (
                ("ensure_cuda", lambda: ws.ensure(ctx, H, device=cuda)),
                ("reserve_cuda", lambda: ws.reserve([ctx], H, device=cuda)),
            ):
                with pytest.raises(NotImplementedError) as info:
                    fn()
                assert str(info.value) == want_dev, (name, int(ctx.T_r))


# W = 2 / 4 CPU process groups (gloo), one rank without tokens in each
REFUSAL_ROWS = [(2, "ragged:3000,0"), (4, "ragged:1200,0,900,900")]


def _refusal_rank(rank: int, W: int, layout: str, port: int, out_dir: str) -> None:
    """One spawned rank of a CPU gloo group: the flag-on entries on this rank's operands, each run to its exception;
    then one all-gather of the outcomes (a rank that had entered a collective alone would hold this gather until
    the group's timeout); rank 0 writes them.  Once the existing backend serves the mode, one rank-independent
    reason stands in for the check (the rows test where the refusal fires, not the reason)."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "gloo", rank=rank, world_size=W, timeout=datetime.timedelta(seconds=120)
    )
    try:
        part, _ = _partition((3000,), W, layout)
        ctx = cake_kcp_plan.build_context_det_kcp(
            part, None, rank, "cpu", 4, sm_count=148
        )
        H = 4
        ops = _cpu_ops(ctx, H)
        args = _args(ops, ctx)
        T = int(ctx.T_r)
        bad = torch.zeros(T, H, 2 * D, dtype=torch.bfloat16)[..., :D]
        # a base address 2 bytes past the 16-byte alignment (the one per-rank view check, where the rank has tokens)
        wide = torch.zeros(T, H * D + 8, dtype=torch.bfloat16)
        misaligned = wide[:, 1 : 1 + H * D].view(T, H, D)
        handle = cake_kcp.RecomputeHandleDetKcp
        calls = {
            "fwd": lambda: cake_kcp.chunk_kda_det_fwd_kcp(*args, **L2N),
            "fwd_replay": lambda: cake_kcp.chunk_kda_det_fwd_kcp(
                *args, boundary_mode="replay", **L2N
            ),
            "bwd_replay": lambda: cake_kcp.chunk_kda_det_bwd_kcp(
                *args, ops["dout"], **L2N
            ),
            "autograd": lambda: cake_kcp.chunk_kda_det_kcp(*args, **L2N),
            "autograd_handle": lambda: cake_kcp.chunk_kda_det_kcp(
                *args, recompute_handle=handle(), **L2N
            ),
            "workspace": lambda: cake_kcp.WorkspaceDetKcp(**L2N),
            "workspace_ctx": lambda: cake_kcp.WorkspaceDetKcp(ctx, H, **L2N),
            "workspace_bytes": lambda: cake_kcp_plan.kda_det_workspace_bytes_kcp(
                ctx, H=H, **L2N
            ),
            "saved_bytes": lambda: cake_kcp_plan.kda_det_saved_bytes_kcp(
                ctx, H=H, **L2N
            ),
            "misaligned_q": lambda: cake_kcp.chunk_kda_det_fwd_kcp(
                misaligned, *args[1:], **L2N
            ),
            "bad_head_stride": lambda: cake_kcp.chunk_kda_det_fwd_kcp(
                bad, *args[1:], **L2N
            ),
        }
        got: dict = {}
        forced = _l2n_missing() is None
        for name, fn in calls.items():
            with pytest.MonkeyPatch.context() as m:
                # past the capability check, the view rule's own refusal; else the check's reason (one stand-in
                # once the existing backend serves the mode)
                if name == "bad_head_stride":
                    m.setattr(cake_kcp_plan, "qk_l2norm_missing", _missing(None))
                elif forced:
                    m.setattr(cake_kcp_plan, "qk_l2norm_missing", _missing("forced"))
                try:
                    fn()
                    got[name] = None
                except Exception as exc:  # the outcome is the row's datum
                    got[name] = [type(exc).__name__, str(exc)]
        gathered: list = [None] * W
        dist.all_gather_object(gathered, {"T_r": int(ctx.T_r), "outcomes": got})
        if rank == 0:
            with open(os.path.join(out_dir, "refusal.json"), "w") as fh:
                json.dump(gathered, fh, indent=1)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "W,layout", REFUSAL_ROWS, ids=[f"w{w}" for w, _l in REFUSAL_ROWS]
)
def test_l2n_refusal_is_rank_symmetric(W, layout) -> None:
    """In a spawned CPU group of ``W`` ranks (gloo) with a rank without tokens, every flag-on entry (forward, replay
    forward, replay backward, the autograd entry with and without a recompute handle, ``WorkspaceDetKcp`` with and
    without a context, the byte queries, a forward on a misaligned q view, misaligned on the ranks with tokens only)
    raises on every rank the same ``NotImplementedError`` with the same message, and past the capability check a
    view with a bad head stride the same ``ValueError``; the ranks then meet in one all-gather, so no rank entered a
    collective alone (no hang)."""
    if not dist.is_available() or not dist.is_gloo_available():
        pytest.skip("torch.distributed with the gloo backend required")
    with tempfile.TemporaryDirectory(prefix="cake_kcp_l2n_refusal_") as tmp:
        mp.start_processes(
            _refusal_rank,
            args=(W, layout, _free_port(), tmp),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(tmp, "refusal.json")) as fh:
            ranks = json.load(fh)
    assert len(ranks) == W
    tokens = [r["T_r"] for r in ranks]
    assert 0 in tokens and any(t > 0 for t in tokens), tokens
    pending = re.escape(cake_kcp_plan.QK_L2NORM_PENDING)
    for name in ranks[0]["outcomes"]:
        outcomes = [r["outcomes"][name] for r in ranks]
        assert all(o is not None for o in outcomes), (name, outcomes)
        assert all(o == outcomes[0] for o in outcomes), (name, outcomes)
        kind, message = outcomes[0]
        if name == "bad_head_stride":
            assert kind == "ValueError" and "head stride" in message, outcomes[0]
        else:
            assert kind == "NotImplementedError", (name, outcomes[0])
            assert re.match(pending, message), (name, message)


class _DriveCall(types.SimpleNamespace):
    """A call for the real :func:`cake_kcp._forward` / :func:`cake_kcp._backward`: its local stages run the kernel
    backend's ``stage`` (the fake arena records their launches); every KCP walk, merge, collective and buffer method
    is recorded by name in ``kcp`` and launches nothing."""

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def record(*_a, **_k) -> None:
            self.kcp.append(name)

        return record

    def fresh(self, name: str) -> torch.Tensor:
        self.kcp.append(f"fresh:{name}")
        self.bufs[name] = torch.zeros(1)
        return self.bufs[name]

    def stage(self, name: str, **extra) -> None:
        self.backend.stage(name, self, **extra)


LOCAL_RECORD_CASES = [
    (4, "sm_100a", 148),
    (64, "sm_100a", 148),
    (4, "sm_103a", 152),
    (64, "sm_103a", 152),
    (64, "sm_107a", 216),
]


@pytest.mark.parametrize("form", ("fused", "fallback"))
@pytest.mark.parametrize("gate", ("g", "raw"))
@pytest.mark.parametrize(
    "H,arch,sm",
    LOCAL_RECORD_CASES,
    ids=[f"h{h}-{a}-{s}" for h, a, s in LOCAL_RECORD_CASES],
)
def test_l2n_local_stage_values_cover_the_records(H, arch, sm, gate, form) -> None:
    """The real drivers (:func:`cake_kcp._qk_l2norm_begin`, then :func:`cake_kcp._forward` /
    :func:`cake_kcp._backward`, their KCP programs and collectives recorded, not run) on the mode's plan
    (``cake_backend.plan(..., l2n=form)``): a saving and a non-saving forward, a saved-mode and a replay backward bind
    every logical name of every record they launch; no record outside the mode's own (``_l2n`` twins, the ``qk_*``
    programs) binds ``q_raw`` / ``k_raw``; the first reader gets the saved rstd with (1, 0), a fresh one with (0, 1)
    or none with (0, 0); no gate epilogue launches; stage for stage the records are the flag-off plan's or the mode's
    twins, the KCP-owned records and the KCP walk / merge / collective sequence unchanged."""
    missing = cake_kcp_plan.qk_l2norm_missing(form, gate, arch)
    if missing is not None:
        pytest.skip(
            f"use_qk_l2norm_in_kernel=True ({form}) cannot run on the KCP route yet: {missing}"
        )
    ctx = _cpu_ctx(sm=sm)
    T = int(ctx.T_r)
    assert T > 0
    plans = {
        mode: cake_backend.plan(
            ctx.T_r, ctx.n_frag, H, sm, arch, gate=gate, **cake_kcp_plan.qk_l2norm_plan_kwargs(mode)
        )
        for mode in ("off", form)
    }  # fmt: skip
    tb = cake_kcp._Tables(ctx, H, fused_walks=cake_kcp._options({}).fused_walks)
    launches: list = []

    class Arena:
        def __init__(self, plan) -> None:
            self.plan = plan
            self.device = torch.device("cpu")
            self.device_index = 0
            self.arch = arch
            self.launcher = self._launch
            self.fwd_layout = cake_kcp.LOCAL_LAYOUT_FWD
            self._views: dict = {}

        def views(self, mode: str):
            ns = self._views.get(mode)
            if ns is None:
                ns = types.SimpleNamespace()
                for r in self.plan.layout(mode):
                    setattr(
                        ns,
                        r.name,
                        torch.zeros(2, D) if r.name in ("h", "dh") else torch.zeros(1),
                    )
                self._views[mode] = ns
            return ns

        def _launch(self, record, values, *, grid, device_index) -> None:
            launches.append(("backend", record, grid, dict(values)))

    class Provider:
        def run_record(self, record, values, *, grid, device_index) -> None:
            launches.append(("kcp", record, grid, dict(values)))

    backend = cake_kcp.KernelBackend(programs=Provider())
    zeros = torch.zeros(T, H, D, dtype=torch.bfloat16)
    q_raw, k_raw, _ = _packed_qk(zeros, zeros)
    ops = {"q": q_raw, "k": k_raw, "v": torch.zeros(1), "beta": torch.zeros(1)}
    if gate == "raw":
        gp = cake_backend._gate_params(
            "raw",
            torch.zeros(H),
            torch.zeros(H * D),
            LOWER_BOUND,
            H,
            torch.device("cpu"),
        )
        ops.update(cake_backend._raw_values(zeros.clone(), gp))
        grad_names = (
            "dq",
            "dk",
            "dv",
            "dbeta",
            "dg_raw",
            "dA_part",
            "db_part",
            "dA_log",
            "ddt_bias",
        )
    else:
        ops["g"] = torch.zeros(1)
        grad_names = ("dq", "dk", "dv", "dg", "dbeta")
    grads = {n: torch.zeros(1) for n in grad_names}
    saved = types.SimpleNamespace(
        S_in=torch.zeros(1), c1_recv=None, M_seg=None, carry=None
    )
    saved_rstd = torch.zeros(2, T, H)

    opts = cake_kcp._options({})

    def run(mode: str, kind: str):
        backward = kind.startswith("bwd")
        call = _DriveCall(
            local=Arena(plans[mode]), ctx=ctx, tb=tb, T_r=T, H=H, backward=backward, host_trim=int(opts.host_trim),
            ops={**ops, **({"dout": torch.zeros(1)} if backward else {})}, scale=0.5, scratch={},
            local_values=None, out=None if backward else torch.zeros(1), grads=grads if backward else None,
            device=torch.device("cpu"), opts=opts, gate=gate, l2n=mode != "off", l2n_mode=mode, l2n_values=None,
            backend=backend, schedule=str(opts.schedule), exchange="a2a", merge_fuse=str(opts.merge_fuse),
            saved_carry=int(opts.saved_carry), bufs={}, kcp=[],
        )  # fmt: skip
        backend.begin_call(call)
        kept = None
        if call.l2n:
            rstd = saved_rstd if kind == "bwd_saved" else None
            kept = cake_kcp._qk_l2norm_begin(call, rstd, save=kind == "fwd_save")
        if backward:
            cake_kcp._backward(call, saved if kind == "bwd_saved" else None)
            # the per-call rows die with the backward
            assert call.l2n_values is None, kind
        else:
            cake_kcp._forward(call, save=kind == "fwd_save", replay=False)
        done = list(launches)
        launches.clear()
        return done, kept, call.kcp

    def mod(src: str, record: str) -> dict:
        return (cake_jit if src == "backend" else cake_kcp_jit).MODULES[record]

    def own(record: str) -> bool:  # the mode's own records
        return "_l2n" in record or record.startswith("qk_")

    expected_rw = {
        "fwd_save": (0, 1),
        "fwd_nosave": (0, 0),
        "bwd_saved": (1, 0),
        "bwd_replay": (0, 1),
    }
    for kind in ("fwd_save", "fwd_nosave", "bwd_saved", "bwd_replay"):
        res, kept, kcp_on = run(form, kind)
        res_off, _, kcp_off = run("off", kind)
        assert kcp_on == kcp_off and kcp_on, kind  # the KCP programs and collectives
        for src, record, _grid, values in res:
            m = mod(src, record)
            absent = set(m["logical"].values()) - set(values)
            assert not absent, (kind, record, sorted(absent))
            assert values["q"] is not q_raw and values["k"] is not k_raw, (kind, record)
            if {"q_raw", "k_raw"} & set(m["logical"].values()):
                assert src == "backend" and own(record), (kind, record)
            if src == "backend":
                assert m["stage"] != "gate_epilogue", (kind, record)
        first = next(v for s, r, _g, v in res if s == "backend" and own(r))
        if form == "fused":
            assert (first["read_rstd"], first["write_rstd"]) == expected_rw[kind], kind
            # the forward writes no token-layout qn / kn, the backward's first reader does
            assert first["write_qkn"] == int(kind.startswith("bwd")), kind
        if kind == "bwd_saved":
            assert first["qk_rstd"] is saved_rstd
        elif kind == "fwd_save":
            assert first["qk_rstd"] is kept and tuple(kept.shape) == (2, T, H)
        # stage for stage: the flag-off records or the mode's twins; the KCP-owned records unchanged
        on_l = [
            (s, r)
            for s, r, _g, _v in res
            if not (s == "backend" and r.startswith("qk_"))
        ]
        off_l = [
            (s, r)
            for s, r, _g, _v in res_off
            if not (s == "backend" and mod(s, r)["stage"] == "gate_epilogue")
        ]
        assert len(on_l) == len(off_l), (kind, on_l, off_l)
        on_stages = collections.Counter(
            mod(s, r)["stage"] if s == "backend" else r for s, r in on_l
        )
        off_stages = collections.Counter(
            mod(s, r)["stage"] if s == "backend" else r for s, r in off_l
        )
        assert on_stages == off_stages, kind
        assert [r for s, r in on_l if s == "kcp"] == [r for s, r in off_l if s == "kcp"]
        for s, r in on_l:
            if s == "backend" and (s, r) not in off_l:
                assert own(r), (kind, r)
        if form == "fallback":
            names = [r for s, r, _g, _v in res if s == "backend"]
            first_qk = "qk_l2norm_apply" if kind == "bwd_saved" else "qk_l2norm"
            assert names[0] == first_qk, (kind, names)
            if kind.startswith("bwd"):
                i = names.index("qk_pullback")
                assert "_l2n_capture" in names[i - 1], names
                assert names[i + 1 :] == (["param_finalize"] if gate == "raw" else [])


# --------------------------------------------------------------------------- GPU helpers


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def _same(x: dict, y: dict, names) -> bool:
    return all(_bits_equal(x[n], y[n]) for n in names)


def _collective_seq(rec: dict) -> list:
    return [s for s in rec["sequence"] if ":" in s]


def _local_stages(rec: dict) -> collections.Counter:
    """The local stage calls of a launch record as a multiset (the fused mode may run the intra program before the
    prep: the same stages in another order)."""
    return collections.Counter(s for s in rec["sequence"] if ":" not in s)


def _graph_events_ok(events: list) -> bool:
    """The forward follows GRAPH_EVENTS exactly; the backward captures once, after one or more eager calls (its key
    holds the saved tensors, whose addresses may repeat one call early), then replays (the raw-gate rows' rule)."""
    if [e[0] for e in events] != [e[0] for e in GRAPH_EVENTS]:
        return False
    bwd = [e[1] for e in events]
    if bwd.count("capture") != 1:
        return False
    i = bwd.index("capture")
    return i >= 1 and set(bwd[:i]) == {"eager"} and set(bwd[i + 1 :]) == {"replay"}


def _skip_unless_runnable(W: int, *, gate: str | None = "g") -> None:
    """Skip without ``W`` devices of a covered architecture, the exported KCP programs or (``gate`` not ``None``)
    the fused form's records of that gate mode."""
    if gate is not None:
        missing = _l2n_missing("fused", gate)
        if missing is not None:
            pytest.skip(
                f"use_qk_l2norm_in_kernel=True cannot run on the KCP route yet: {missing}"
            )
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    if cake_jit.device_arch(torch.cuda.current_device()) not in cake_jit.ARCHES:
        pytest.skip("the current device's architecture is not covered by the package")


def _l2n_inputs(
    lengths, H: int, device, *, raw_gate: bool = False, seed: int | None = None
) -> dict:
    """``make_inputs``' operands with raw q / k: its L2-normalized rows times ``2^e`` per (token, head), ``e`` uniform
    in ``[-6, 6]``, about 1/64 of the rows zero (rstd ``1 / sqrt(1e-6)``) or subnormal (the normalized row times
    ``2^-126``).  ``raw_gate``: ``g`` is a raw BF16 gate with ``A_log`` FP32 ``[H]`` and ``dt_bias`` FP32 ``[H * 128]``."""
    glob = make_inputs(lengths, H, device, seed=seed)
    S = int(glob["q"].shape[0])
    gen = torch.Generator(device="cpu")
    gen.manual_seed(0x7817 if seed is None else int(seed) ^ 0x7817)
    for name in ("q", "k"):
        base = glob[name].float().cpu().reshape(S * H, D)
        e = torch.randint(-6, 7, (S * H, 1), generator=gen).to(torch.float32)
        x = base * torch.exp2(e)
        pick = torch.randperm(S * H, generator=gen)[: max(2, S * H // 64)]
        x[pick[0::2]] = 0.0
        x[pick[1::2]] = base[pick[1::2]] * 2.0**-126
        glob[name] = x.reshape(S, H, D).to(torch.bfloat16).to(device)
    if raw_gate:
        glob["g"] = torch.randn(S, H, D, generator=gen).to(torch.bfloat16).to(device)
        glob["A_log"] = torch.log(torch.rand(H, generator=gen) + 1.0).to(device)
        glob["dt_bias"] = (torch.randn(H * D, generator=gen) * 0.5).to(device)
    return glob


def _exact_l2n(glob: dict, cu_t: torch.Tensor, raw_gate: bool) -> dict:
    """The exact FP64 output and gradients of the raw inputs: ``x_hat = x / sqrt(sum(x * x) + 1e-6)`` per token and
    head for q and k (raw gate mode: ``g = lower_bound * sigmoid(exp(A_log) * (raw + dt_bias))``), the exact
    recurrence, all differentiated in FP64."""
    f64 = torch.float64
    names = ("q", "k", "v", "g", "beta") + (("A_log", "dt_bias") if raw_gate else ())
    leaves = [glob[n].detach().to(f64).requires_grad_(True) for n in names]
    q, k, v, g, beta = leaves[:5]
    qn = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + EPS)
    kn = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + EPS)
    if raw_gate:
        A_log, dt_bias = leaves[5:]
        H = q.shape[1]
        g = LOWER_BOUND * torch.sigmoid(
            torch.exp(A_log).view(1, H, 1) * (g + dt_bias.view(1, H, D))
        )
    o = kda_exact_forward(qn, kn, v, g, beta, cu_t, SCALE, dtype=f64)
    grads = torch.autograd.grad(o, leaves, glob["dout"].to(f64))
    out_names = ("dq", "dk", "dv", "dg", "dbeta") + (PARAM_GRADS if raw_gate else ())
    return {
        "out": o.detach().cpu(),
        **{n: x.detach().cpu() for n, x in zip(out_names, grads, strict=True)},
    }


# --------------------------------------------------------------------------- GPU row: flag-off identity


RECORD_KEYS = ("sequence", "n_launches", "n_collectives", "n_walks", "n_merges", "n_stages",
               "workspace_bytes", "pool_bytes", "fresh_bytes", "exchange", "P", "NSEG")  # fmt: skip


def test_gpu_l2n_flag_off_identity_w1() -> None:
    """On one device, the omitted keyword and an explicit ``False`` give bitwise the same outputs and gradients, the
    same saved-set tensors (member, shape, strides, dtype, in order), flags and meta, and the same launch inventory
    (sequence and counters); with ``KCP_L2N_FLAG_OFF_CAPTURE`` / ``_BASELINE`` the omitted form's record (suffix
    ``.gpu.json``) and its outputs (``.gpu.pt``) are written on the base or compared with it."""
    _skip_unless_runnable(1, gate=None)
    device = torch.device("cuda", torch.cuda.current_device())
    rows = (((100, 640, 77, 1000, 129), 4), ((700, 1, 1300, 999), 64))
    record: dict = {}
    tensors_of: dict = {}
    for lengths, H in rows:
        part, cu = _partition(lengths, 1, "contiguous")
        ctx = cake_kcp_plan.build_context_det_kcp(part, None, 0, device)
        glob = make_inputs(lengths, H, device)
        q, k, v, g, beta, dout = (glob[n] for n in ("q", "k", "v", "g", "beta", "dout"))
        got = {}
        for tag, kw in (("omitted", {}), ("explicit", OFF)):
            ws = cake_kcp.WorkspaceDetKcp(**kw)
            out, saved = cake_kcp.chunk_kda_det_fwd_kcp(
                q, k, v, g, beta, ctx, workspace=ws, **kw
            )
            rec_f = dict(cake_kcp.LAST_CALL["fwd"])
            grads = cake_kcp.chunk_kda_det_bwd_kcp(
                q, k, v, g, beta, ctx, dout, saved=saved, workspace=ws, **kw
            )
            rec_b = dict(cake_kcp.LAST_CALL["bwd"])
            # as on the base of this change: a flag-off launch record carries no key of the mode
            assert not {"l2n", "l2n_rstd"} & (set(rec_f) | set(rec_b)), tag
            got[tag] = {
                "tensors": {"out": out, **dict(zip(TOKEN_GRADS, grads, strict=True))},
                "record": {
                    "saved": _saved_summary(saved),
                    "fwd": _jsonable({x: rec_f.get(x) for x in RECORD_KEYS}),
                    "bwd": _jsonable({x: rec_b.get(x) for x in RECORD_KEYS}),
                },
            }
            del ws, saved
        a, b = got["omitted"], got["explicit"]
        assert _same(a["tensors"], b["tensors"], ("out",) + TOKEN_GRADS), (lengths, H)
        assert a["record"] == b["record"], (lengths, H)
        assert "qk_l2norm" not in json.dumps(a["record"]["saved"])
        # the saved set's meta keys of the base of this change (the golden file, captured by executing the base)
        assert sorted(a["record"]["saved"]["meta"]) == _golden()["saved_meta_keys"]
        key = f"s{'_'.join(map(str, lengths))}-h{H}"
        record[key] = a["record"]
        tensors_of[key] = {n: t.detach().cpu() for n, t in a["tensors"].items()}
    capture, baseline = os.environ.get(ENV_CAPTURE), os.environ.get(ENV_BASELINE)
    snap = json.loads(json.dumps(record))
    if capture:
        with open(capture + ".gpu.json", "w") as fh:
            json.dump(snap, fh, indent=1, sort_keys=True)
        torch.save(tensors_of, capture + ".gpu.pt")
    if baseline:
        with open(baseline + ".gpu.json") as fh:
            ref = json.load(fh)
        assert snap == ref, "the flag-off GPU record differs from the base"
        ref_t = torch.load(baseline + ".gpu.pt")
        assert sorted(ref_t) == sorted(tensors_of)
        for key, ts in tensors_of.items():
            assert _same(ts, ref_t[key], ("out",) + TOKEN_GRADS), key


# --------------------------------------------------------------------------- GPU rows: flag on (one process per rank)

# (lengths, H, W, layout): W = 1 in-process, CP2 / CP4 spawned, CP8 one process per Slurm task; documents crossing
# ranks with boundaries inside chunks (cross-document resets), documents over >= 3 ranks, one-token documents and
# fragments, a zero-token rank at W = 2 / 4 / 8, 4 and 64 heads.
ROWS = [
    ((100, 640, 77, 1000, 129), 4, 1, "contiguous"),
    ((700, 1, 1300, 999), 64, 1, "contiguous"),
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((3000,), 4, 2, "zigzag_per_document"),
    ((2000,), 4, 2, "ragged:2000,0"),
    ((100, 640, 77, 1000, 129), 4, 4, "contiguous"),
    ((3000,), 4, 4, "zigzag"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((3000,), 4, 4, "ragged:1200,0,900,900"),
    ((700, 1, 1300, 999), 64, 4, "zigzag_per_document"),
    ((3000, 2048, 5, 1), 4, 8, "zigzag_per_document"),
    ((4000, 1003), 4, 8, "ragged:1,0,999,1000,1000,1000,1,1002"),
    ((700, 1, 1300, 999), 64, 8, "zigzag"),
]
RAW_GATE_ROWS = [
    ((100, 640, 77, 1000, 129), 4, 1, "contiguous"),
    ((3000,), 4, 2, "zigzag_per_document"),
    ((3000,), 4, 4, "ragged:1200,0,900,900"),
    ((4000, 1003), 4, 8, "ragged:1,0,999,1000,1000,1000,1,1002"),
]


def _row_id(row) -> str:
    lengths, H, W, layout = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}-{layout.split(':')[0]}"


def _gather(x: torch.Tensor, W: int) -> list:
    if W == 1:
        return [x.detach().cpu()]
    objs: list = [None] * W
    dist.all_gather_object(objs, x.detach().cpu())
    return objs


def _run_rank(
    rank: int, W: int, row, port: int, result_dir: str | None, raw_gate: bool = False
) -> dict:
    """One rank: the flag-on KCP passes (both forms) and their bitwise flags, the saved rstd, the flag-off comparison of
    the collectives and stages, the retained bytes; gathered to rank 0 and compared with the single-rank flag-on path
    and the FP64 reference there."""
    from torch.utils.checkpoint import checkpoint

    lengths, H, _w, layout = row
    device = torch.device("cuda", rank % max(torch.cuda.device_count(), 1))
    torch.cuda.set_device(device)
    if W > 1 and not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        dist.init_process_group("nccl", rank=rank, world_size=W, device_id=device)
    group = dist.group.WORLD if W > 1 else None
    part, cu = _partition(lengths, W, layout)
    ctx = cake_kcp_plan.build_context_det_kcp(part, group, rank, device)
    total = cu[-1]
    T = int(ctx.T_r)
    gate = "raw" if raw_gate else "g"
    glob = _l2n_inputs(lengths, H, device, raw_gate=raw_gate)
    names = ("q", "k", "v", "g", "beta", "dout")
    ops = _local_slices(part, rank, cu, {n: glob[n] for n in names})
    q, k, v, g, beta, dout = (ops[n] for n in names)
    GK = (
        dict(use_gate_in_kernel=True, A_log=glob["A_log"], dt_bias=glob["dt_bias"], lower_bound=LOWER_BOUND)
        if raw_gate
        else {}
    )  # fmt: skip
    grad_names = TOKEN_GRADS + (PARAM_GRADS if raw_gate else ())
    tensors = ("out",) + grad_names
    ws = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=raw_gate, **L2N)
    ws_off = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=raw_gate)
    # None once the fallback form can run
    fallback_pending = _l2n_missing("fallback", gate)
    has_fallback = fallback_pending is None
    ws_fb = (
        cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=raw_gate, **L2N, **FALLBACK)
        if has_fallback
        else None
    )

    def saved_pass(qq=q, kk=k, workspace=ws, l2n=True, **opts):
        kw = {**GK, "use_qk_l2norm_in_kernel": l2n}
        out, saved = cake_kcp.chunk_kda_det_fwd_kcp(
            qq, kk, v, g, beta, ctx, workspace=workspace, **kw, **opts
        )
        rec_f = dict(cake_kcp.LAST_CALL["fwd"])
        bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
        grads = cake_kcp.chunk_kda_det_bwd_kcp(
            qq, kk, v, g, beta, ctx, dout, saved=saved, workspace=workspace, **kw, **bwd_opts
        )  # fmt: skip
        rec_b = dict(cake_kcp.LAST_CALL["bwd"])
        res = {"out": out, **dict(zip(grad_names, grads, strict=True))}
        return res, saved, (rec_f, rec_b)

    def autograd_pass(ckpt: bool = False, handle=None, keep="states", workspace=ws):
        srcs = (q, k, v, g, beta) + ((GK["A_log"], GK["dt_bias"]) if raw_gate else ())
        leaves = [x.detach().clone().requires_grad_(True) for x in srcs]
        n = len(srcs)

        def fn(*a):
            gk = {**GK, **({"A_log": a[5], "dt_bias": a[6]} if raw_gate else {})}
            h = a[n] if len(a) > n else None
            return cake_kcp.chunk_kda_det_kcp(
                *a[:5], ctx, None, "light", "saved", workspace, recompute_handle=h,
                checkpoint_keep=keep if h is not None else None, **gk, **L2N,
            )  # fmt: skip

        extra = (handle,) if handle is not None else ()
        out = (
            checkpoint(fn, *leaves, *extra, use_reentrant=False)
            if ckpt
            else fn(*leaves, *extra)
        )
        out.backward(dout)
        leaf_names = ("dq", "dk", "dv", "dg", "dbeta") + (
            PARAM_GRADS if raw_gate else ()
        )
        got = {x: lf.grad for x, lf in zip(leaf_names, leaves, strict=True)}
        return {"out": out.detach(), **got}

    res, saved, (rec_f, rec_b) = saved_pass()
    flags: dict = {}
    reported: dict = {}
    rstd = saved.qk_rstd
    flags["saved_rstd"] = (
        saved.qk_l2norm is True
        and tuple(rstd.shape) == (2, T, H)
        and rstd.dtype == torch.float32
        and rstd.is_contiguous()
        and saved.meta().get("qk_l2norm") is True
        and saved.version == cake_kcp.KCP_SAVED_VERSION
    )
    flags["finite"] = all(bool(torch.isfinite(res[x].float()).all()) for x in tensors)
    flags["records_mark_the_mode"] = (rec_f["l2n"], rec_b["l2n"]) == ("fused", "fused")
    flags["first_reader_rstd"] = T == 0 or (
        tuple(rec_f["l2n_rstd"]) == (0, 1) and tuple(rec_b["l2n_rstd"]) == (1, 0)
    )
    rep, saved_rep, _ = saved_pass()
    flags["rstd_repeat_bitwise"] = _bits_equal(saved_rep.qk_rstd, rstd)
    del saved_rep
    replay, _, (_, rec_br) = saved_pass(boundary_mode="replay")
    flags["replay_rstd_rw"] = T == 0 or tuple(rec_br["l2n_rstd"]) == (0, 1)
    runs = {
        "repeat": rep,
        "saved_eq_replay": replay,
        "save_transfer_eq": saved_pass(save_transfer=False)[0],
        "autograd_eq_saved": autograd_pass(),
        "ckpt_eq_direct": autograd_pass(ckpt=True),
        "handle_states_eq_ckpt": autograd_pass(
            ckpt=True, handle=cake_kcp.RecomputeHandleDetKcp(), keep="states"
        ),
        "handle_transfer_eq_ckpt": autograd_pass(
            ckpt=True, handle=cake_kcp.RecomputeHandleDetKcp(), keep="transfer"
        ),
        "ws_none_eq": autograd_pass(workspace=None),
    }
    qp, kp, _buf = _packed_qk(q, k)
    runs["packed_strided_eq"] = saved_pass(qq=qp, kk=kp)[0]
    flags.update({name: _same(res, r, tensors) for name, r in runs.items()})
    if W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS:
        ring = saved_pass(exchange="ring", save_transfer=False)[0]
        flags["ring_eq_a2a"] = _same(runs["save_transfer_eq"], ring, tensors)
    cake_kcp.graph_cache_clear()
    events = []
    try:
        for _ in range(len(GRAPH_EVENTS)):
            res_graph = {x: t.clone() for x, t in saved_pass(graph="replay")[0].items()}
            events.append(
                [str(cake_kcp.LAST_CALL[x].get("graph_event")) for x in ("fwd", "bwd")]
            )
        torch.cuda.synchronize(device)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        cake_kcp.graph_cache_clear()
    flags["graph_replay"] = _graph_events_ok(events) and _same(res, res_graph, tensors)
    # the fallback form (the workspace's test knob): repeatable and in both modes the same bits; its equality with
    # the fused form is reported (the single-rank tests check it; the accuracy gate below holds either way).  A
    # fallback form that cannot run yet is recorded with its reason and fails the row (never skipped silently)
    fb = None
    flags["fallback_ran"] = has_fallback
    reported["fallback_pending"] = fallback_pending
    if has_fallback:
        fb, saved_fb, (rec_ff, rec_bf) = saved_pass(workspace=ws_fb)
        flags["fallback_records"] = (rec_ff["l2n"], rec_bf["l2n"]) == (
            "fallback",
            "fallback",
        )
        flags["fallback_repeat"] = _same(fb, saved_pass(workspace=ws_fb)[0], tensors)
        flags["fallback_saved_eq_replay"] = _same(
            fb, saved_pass(workspace=ws_fb, boundary_mode="replay")[0], tensors
        )
        reported["fallback_eq_fused"] = _same(fb, res, tensors)
        reported["fallback_rstd_eq_fused"] = _bits_equal(saved_fb.qk_rstd, rstd)
        del saved_fb
    # the flag-off route on the same operands: the same collectives and the same local stage calls (the mode's
    # records replace the q / k readers stage for stage); a rank without tokens runs no local stage
    res_off, saved_off, (rec_fo, rec_bo) = saved_pass(workspace=ws_off, l2n=False)
    flags["collectives_eq_flag_off"] = (
        _collective_seq(rec_f) == _collective_seq(rec_fo)
        and _collective_seq(rec_b) == _collective_seq(rec_bo)
        and rec_f["n_collectives"] == rec_fo["n_collectives"]
        and rec_b["n_collectives"] == rec_bo["n_collectives"]
    )
    flags["stages_eq_flag_off"] = _local_stages(rec_f) == _local_stages(rec_fo) and (
        _local_stages(rec_b) == _local_stages(rec_bo)
    )
    # a flag-off launch record carries no key of the mode
    flags["flag_off_records"] = not ({"l2n", "l2n_rstd"} & (set(rec_fo) | set(rec_bo)))
    flags["zero_token_rank_no_stage"] = T > 0 or (
        rec_f["n_stages"] == 0 and rec_b["n_stages"] == 0
    )
    flags["saved_bytes_eq_query"] = (
        saved.bytes - saved_off.bytes
        == 8 * T * H
        == cake_kcp_plan.kda_det_saved_bytes_kcp(ctx, H=H, with_conv=False, **L2N)
        - cake_kcp_plan.kda_det_saved_bytes_kcp(ctx, H=H, with_conv=False)
    )
    # reported: the flag-off forward on the materialized bf16(x * rstd) (bitwise only where the mode's intra program
    # keeps the flag-off arithmetic after the normalization)
    qn = (q.float() * rstd[0].unsqueeze(-1)).to(torch.bfloat16)
    kn = (k.float() * rstd[1].unsqueeze(-1)).to(torch.bfloat16)
    out_c, _ = cake_kcp.chunk_kda_det_fwd_kcp(
        qn, kn, v, g, beta, ctx, workspace=ws_off, save_for_backward=False, **GK
    )
    reported["out_eq_flag_off_on_normalized"] = _bits_equal(out_c, res["out"])
    # retained after a saving forward: the rstd only (the allocator's 512-byte granule)
    torch.cuda.synchronize(device)
    gc.collect()
    retained = {}
    for tag, workspace, l2n in (("on", ws, True), ("off", ws_off, False)):
        before = torch.cuda.memory_allocated(device)
        kw = {"use_qk_l2norm_in_kernel": l2n, **GK}
        o, s = cake_kcp.chunk_kda_det_fwd_kcp(
            q, k, v, g, beta, ctx, workspace=workspace, **kw
        )
        torch.cuda.synchronize(device)
        retained[tag] = torch.cuda.memory_allocated(device) - before
        del o, s
    granule = -(-8 * T * H // 512) * 512
    flags["retained_fwd_delta_is_rstd"] = retained["on"] - retained["off"] == granule
    red = red_fb = None
    if raw_gate:
        red = cake_kcp.reduce_gate_param_grads_det_kcp(
            ctx, res["dA_log"], res["ddt_bias"]
        )
        red_ok = True
        for r in runs.values():
            ra, rb = cake_kcp.reduce_gate_param_grads_det_kcp(
                ctx, r["dA_log"], r["ddt_bias"]
            )
            red_ok &= _bits_equal(ra, red[0]) and _bits_equal(rb, red[1])
        flags["reduce_eq_modes"] = bool(red_ok)
        if fb is not None:
            red_fb = cake_kcp.reduce_gate_param_grads_det_kcp(
                ctx, fb["dA_log"], fb["ddt_bias"]
            )
    gathered = {x: _gather(res[x], W) for x in ("out",) + TOKEN_GRADS}
    gathered_fb = (
        {x: _gather(fb[x], W) for x in ("out",) + TOKEN_GRADS}
        if fb is not None
        else None
    )
    flag_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
    else:
        flag_list = [flags]
    report: dict = {
        "rank": rank,
        "world": W,
        "row": _row_id(row),
        "raw_gate": raw_gate,
        "pass": True,
        "flags": {},
        "flags_by_rank": flag_list,
        "reported": reported,
        "retained_fwd_bytes": retained,
        "metrics": {},
        "graph_events": events,
        "device": torch.cuda.get_device_name(device),
        "arch": cake_jit.device_arch(device.index),
        "host": os.uname().nodename,
        "job": os.environ.get("SLURM_JOB_ID"),
    }
    for key in sorted({x for f in flag_list for x in f}):
        report["flags"][key] = all(f[key] for f in flag_list if key in f)
        report["pass"] &= report["flags"][key]
    if rank == 0:
        cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
        kw_s = {**GK, **L2N}
        out_s, saved_s = cake_backend.chunk_kda_det_fwd(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, **kw_s
        )
        grads_s = cake_backend.chunk_kda_det_bwd(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, glob["dout"], saved=saved_s, **kw_s
        )  # fmt: skip
        single = {"out": out_s, **dict(zip(grad_names, grads_s, strict=True))}
        del saved_s
        refs = _exact_l2n(glob, cu_t, raw_gate)
        arms = [("", gathered, red)] + (
            [("fallback:", gathered_fb, red_fb)] if gathered_fb else []
        )
        for prefix, gath, red_arm in arms:
            cand = {
                x: _to_global(part, gath[x], cu, total) for x in ("out",) + TOKEN_GRADS
            }
            if raw_gate:
                cand["dA_log"], cand["ddt_bias"] = red_arm[0].cpu(), red_arm[1].cpu()
            for x in tensors:
                ref = refs[x]
                c = cand[x].to(torch.float64).reshape(ref.shape)
                s = single[x].detach().cpu().to(torch.float64).reshape(ref.shape)
                rk = error_metrics(c, ref)["rel_rms"]
                rs = error_metrics(s, ref)["rel_rms"]
                ratio = rk / rs if rs > 0 else (1.0 if rk == 0 else float("inf"))
                # the H-element parameter sums: reported (the raw-gate rows' protocol)
                gated = x not in PARAM_GRADS
                one = single[x].detach().cpu().reshape(cand[x].shape)
                same = _bits_equal(cand[x].cpu(), one)
                report["metrics"][prefix + x] = {
                    "kcp_rel_l2": rk,
                    "single_rel_l2": rs,
                    "ratio": ratio,
                    "bitwise_eq_single": bool(same),
                    "gated": gated,
                    "pass": bool(torch.isfinite(c).all())
                    and (ratio <= ACC_GATE or not gated),
                }
                report["pass"] &= report["metrics"][prefix + x]["pass"]
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"])
        report["pass"] = all(verdict)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        prefix = "l2n-raw-" if raw_gate else "l2n-"
        with open(os.path.join(result_dir, prefix + _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    del runs, res, res_off, saved, saved_off, fb
    gc.collect()
    return report


def _spawn_entry(
    rank: int, W: int, row, port: int, result_dir: str, raw_gate: bool
) -> None:
    try:
        report = _run_rank(rank, W, row, port, result_dir, raw_gate)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _drive(row, raw_gate: bool) -> None:
    lengths, H, W, layout = row
    gate = "raw" if raw_gate else "g"
    prefix = "l2n-raw-" if raw_gate else "l2n-"
    slurm = os.environ.get("SLURM_PROCID") is not None
    if slurm and int(os.environ.get("SLURM_NTASKS", "1")) == W and W > 1:
        _skip_unless_runnable(1, gate=gate)
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]), W, row, int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"), raw_gate,
        )  # fmt: skip
        assert report["pass"], report
        return
    _skip_unless_runnable(W, gate=gate)
    if W == 1:
        report = _run_rank(
            0, 1, row, 0, os.environ.get("KCP_TEST_RESULT_DIR"), raw_gate
        )
        assert report["pass"], report
        return
    with tempfile.TemporaryDirectory(prefix="cake_kcp_l2n_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir, raw_gate),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, prefix + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_l2n_rows_match_single_rank_and_fp64(row) -> None:
    _drive(row, raw_gate=False)


@pytest.mark.parametrize("row", RAW_GATE_ROWS, ids=_row_id)
def test_l2n_raw_gate_rows_match_single_rank_and_fp64(row) -> None:
    """The raw gate mode with the in-kernel Q / K L2 norm: the same checks, the raw gate's gradient among the gated
    tensors, ``dA_log`` / ``ddt_bias`` (reduced in rank order by ``reduce_gate_param_grads_det_kcp``, bitwise the
    same under every mode) reported against FP64."""
    _drive(row, raw_gate=True)
