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

The raw gate mode of the context-parallel (KCP) KDA training route (``use_gate_in_kernel=True`` of
``cake_kcp.chunk_kda_det_kcp`` / ``chunk_kda_det_fwd_kcp`` / ``chunk_kda_det_bwd_kcp``, issue #7816).

CPU rows: the keywords and their refusals, the gate-aware workspace and its byte query (one context or a list of
contexts) and their refusal of the lean local arena, the saved set's metadata, the recompute handle's fingerprint,
the cross-rank reduction helper on one rank, and the local stages' value dicts against the raw records of the
existing backend's registry.

GPU row at W = 1: one reserved raw-mode workspace shared by four packs is bitwise one workspace per pack.

GPU rows (one process per rank; ``W = 1`` in the pytest process, ``W > 1`` spawned or one process per Slurm task as in
``test_cake_kda_det_train_kcp_dist.py``): every rank runs the raw-mode KCP forward / backward in every mode and must be
bitwise repeatable (saved == replay, repeat, autograd == explicit calls, checkpoint == direct, checkpoint with a
recompute handle == plain checkpoint, save_transfer, ring == a2a, graph replay == eager, ``workspace=None``, a
strided raw view == the contiguous one) on every output, the gradient of the raw gate and the per-rank parameter
partial sums; the explicit-gate path on the same activated gate runs the same launches in the same order with the
same collectives (``gate_epilogue`` -> ``param_finalize``); the cross-rank reduction helper gives the same bits on
every rank and under every mode; rank 0 compares the gathered result with the single-rank raw path and the exact
FP64 reference: relative L2 error of KCP / single-rank <= 1.10 for ``out, dq, dk, dv, dbeta`` and the raw gate's
gradient.  ``dA_log`` and ``ddt_bias`` (summed over the ranks by the helper) follow #7816's criterion (d): no less
accurate against FP64 than the existing explicit path, here the explicit KCP route on the same row (its parameter
gradients by the FP64 chain rule from its FP32 activated-gate gradient).  Per row: the single-rank protocol's sanity
rule for parameter gradients, with negative controls that must fail it; seeded: the design's section 5.4 statistic
(``test_raw_gate_kcp_param_grads_no_less_accurate_than_explicit``, env-gated, minutes per row).  The KCP / single-rank
ratio of ``dA_log`` is reported, not gated: an H-element sum of mixed-sign token terms, it spans 0.3-3.7 per seed at
W > 1 on the explicit KCP route as well.
"""

from __future__ import annotations

import gc
import json
import math
import os
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
from tests.experimental.test_cake_kda_det_train_raw_gate_accuracy import (
    AA_SEEDS,
    ALPHA,
    B_BOOT,
    BOOT_SEED,
    ELEMENTWISE_ATOL,
    ELEMENTWISE_RTOL,
    PARAM_SANITY_FACTOR,
    TEST_SEEDS,
    _max_normalized_error,
    bootstrap_critical_value,
    perturbed_torch_gate,
    torch_gate,
    u_stat,
)
from tests.test_helpers.cake_kda_det_train_reference import (
    LOWER_BOUND,
    check_fla_gate,
    error_metrics,
    fla_kda_gate,
    kda_exact_raw_with_grads,
    make_raw_inputs,
    raw_gate,
    t_quantile,
)
from tests.test_helpers.cake_package import cake_package

cake_kcp = cake_package("cake_kda_det_train_kcp", "cake_kcp")
cake_kcp_plan = cake_package("cake_kda_det_train_kcp", "cake_kcp_plan")
cake_backend = cake_package("cake_kda_det_train", "cake_backend")
cake_jit = cake_package("cake_kda_det_train", "cake_jit")
cake_kcp_jit = cake_package("cake_kda_det_train_kcp", "cake_kcp_jit")

D = cake_kcp.D
SCALE = cake_kcp.SCALE
LB = LOWER_BOUND  # -5.0
TOKEN_GRADS = ("dq", "dk", "dv", "dbeta", "dg")  # dg: the raw gate's gradient (BF16)
PARAM_GRADS = ("dA_log", "ddt_bias")  # this rank's partial sums
TENSORS = ("out",) + TOKEN_GRADS + PARAM_GRADS
ACC_GATE = 1.10  # relative L2 error of KCP / single-rank against the FP64 reference
GRAPH_EVENTS = [
    ["eager", "eager"],
    ["capture", "eager"],
    ["replay", "capture"],
    ["replay", "replay"],
]


def _graph_events_ok(events: list) -> bool:
    """The forward follows GRAPH_EVENTS exactly.  The backward key of a rank holds its inputs and its saved tensors
    (``S_in``, ``c1_recv``, ``M_seg``, ``carry``).  When those addresses repeat on the second call, the backward
    captures one call early: at W = 1 the allocator may give the captured forward's ``S_in`` the freed eager
    address; a rank with one token keeps empty saved tensors (address 0).  The explicit mode does the same (the
    W8 ragged row, rank 6).  A capture replays at once, so the collectives stay aligned across ranks.  The rule is
    the single-rank rule of test_cake_kda_det_train_kcp: one capture, eager calls before it, replays after it."""
    if [e[0] for e in events] != [e[0] for e in GRAPH_EVENTS]:
        return False
    bwd = [e[1] for e in events]
    if bwd.count("capture") != 1:
        return False
    i = bwd.index("capture")
    return i >= 1 and set(bwd[:i]) == {"eager"} and set(bwd[i + 1 :]) == {"replay"}


# --------------------------------------------------------------------------- CPU rows


def _cpu_ctx(W: int = 2, rank: int = 0, lengths=(700, 1, 1300, 999), sm: int = 148):
    cu = [0]
    for n in lengths:
        cu.append(cu[-1] + n)
    part = cake_kcp_plan.partition_det_kcp(cu, W, layout="zigzag_per_document")
    return cake_kcp_plan.build_context_det_kcp(part, None, rank, "cpu", 4, sm_count=sm)


def _cpu_ops(ctx, H: int, *, raw: bool = True) -> dict:
    T = int(ctx.T_r)
    gen = torch.Generator().manual_seed(7816)
    ops = {n: torch.randn(T, H, D, generator=gen).to(torch.bfloat16) for n in "qkv"}
    ops["g"] = (
        torch.randn(T, H, D, generator=gen).to(torch.bfloat16)
        if raw
        else -torch.rand(T, H, D, generator=gen)
    )
    ops["beta"] = torch.rand(T, H, generator=gen).to(torch.bfloat16)
    return ops


def _gate_kw(H: int, device="cpu", lb=LB) -> dict:
    return {
        "use_gate_in_kernel": True,
        "A_log": torch.zeros(H, dtype=torch.float32, device=device),
        "dt_bias": torch.zeros(H * D, dtype=torch.float32, device=device),
        "lower_bound": lb,
    }


def test_raw_keywords_are_checked_before_any_launch() -> None:
    """The raw gate keywords follow the single-rank rules (``cake_backend._gate_mode`` / ``_check_raw`` /
    ``_gate_params``) and are checked before the workspace is sized or anything launches: the parameters without
    ``use_gate_in_kernel``, a raw call without ``lower_bound`` or out of range, an FP32 raw gate, a non-bool mode,
    a workspace of the other mode in both directions, and a backend without local stages are refused."""
    H = 4
    ctx = _cpu_ctx()
    ops = _cpu_ops(ctx, H)
    args = (ops["q"], ops["k"], ops["v"], ops["g"], ops["beta"], ctx)
    kw = _gate_kw(H)
    g_ops = _cpu_ops(ctx, H, raw=False)
    g_args = tuple(g_ops[n] for n in ("q", "k", "v", "g", "beta")) + (ctx,)
    with pytest.raises(ValueError, match="use_gate_in_kernel=True only"):
        cake_kcp.chunk_kda_det_fwd_kcp(*g_args, A_log=kw["A_log"])
    with pytest.raises(ValueError, match="use_gate_in_kernel=True only"):
        cake_kcp.chunk_kda_det_bwd_kcp(*g_args, g_ops["q"], lower_bound=LB)
    with pytest.raises(ValueError, match="lower_bound"):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, **{**kw, "lower_bound": None})
    for bad in (0.0, -5.5, float("nan"), True):
        with pytest.raises(ValueError, match="lower_bound"):
            cake_kcp.chunk_kda_det_fwd_kcp(*args, **{**kw, "lower_bound": bad})
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, **{**kw, "use_gate_in_kernel": 1})
    fp32 = (*args[:3], ops["g"].float(), *args[4:])
    with pytest.raises(ValueError, match="torch.bfloat16"):
        cake_kcp.chunk_kda_det_fwd_kcp(*fp32, **kw)
    # an overlapping view is refused (the strided-pointer bindings check no layout of their own)
    over = ops["g"].as_strided(ops["g"].shape, (8, 8, 1))
    with pytest.raises(ValueError, match="strides"):
        cake_kcp.chunk_kda_det_fwd_kcp(*args[:3], over, *args[4:], **kw)
    with pytest.raises(ValueError, match="A_log"):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, **{**kw, "A_log": torch.zeros(H + 1)})
    with pytest.raises(ValueError, match="dt_bias"):
        cake_kcp.chunk_kda_det_fwd_kcp(
            *args, **{**kw, "dt_bias": torch.zeros(H, D - 1)}
        )
    # the workspace's mode: a g workspace refuses a raw call and a raw workspace a g call (both directions)
    ws_g = cake_kcp.WorkspaceDetKcp()
    ws_r = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True)
    assert (ws_g.gate, ws_r.gate) == ("g", "raw")
    with pytest.raises(ValueError, match="use_gate_in_kernel=True"):
        cake_kcp.chunk_kda_det_fwd_kcp(*args, workspace=ws_g, **kw)
    with pytest.raises(ValueError, match="use_gate_in_kernel=False"):
        cake_kcp.chunk_kda_det_fwd_kcp(*g_args, workspace=ws_r)
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp.WorkspaceDetKcp(use_gate_in_kernel="raw")
    # a backend without the local kernel stages (an emulation) has no raw gate
    stub = type("Stub", (cake_kcp.Backend,), {"name": "stub", "local_stages": False})()
    prev = cake_kcp.set_backend(stub)
    try:
        with pytest.raises(NotImplementedError, match="local kernel stages"):
            cake_kcp.chunk_kda_det_fwd_kcp(*args, **kw)
    finally:
        cake_kcp.set_backend(prev)


def test_raw_saved_set_metadata_round_trips() -> None:
    """The saved set records the gate mode and the FP32 lower bound as metadata (same tensors, same version); the
    autograd node's ``meta`` / ``from_tensors`` round trip keeps them; the single-rank mismatch rule refuses the
    other mode and another lower bound."""
    t = torch.zeros(4)
    base = dict(
        version=cake_kcp.KCP_SAVED_VERSION, S_in=t, c1_recv=None, M_seg=None, T_r=1, H=1,
        n_frag=1, P=1, Hg=1, scale=0.5, save_mode="light", save_transfer=False,
        offset_class="fla", exchange="a2a", precision_mode=0, ctx=None,
    )  # fmt: skip
    g = cake_kcp.SavedDetKcp(**base)
    raw = cake_kcp.SavedDetKcp(**base, gate="raw", lower_bound=-5.0)
    assert (g.gate, g.lower_bound) == ("g", None)
    assert raw.tensors() == g.tensors() and raw.flags() == g.flags()
    back = cake_kcp.SavedDetKcp.from_tensors(
        raw.tensors(), raw.flags(), raw.meta(), None
    )
    assert back.meta() == raw.meta() and (back.gate, back.lower_bound) == (
        "raw",
        -5.0,
    )
    gp = cake_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -5.0, 1, torch.device("cpu")
    )
    cake_backend._check_saved_gate(raw, gp)
    cake_backend._check_saved_gate(g, None)
    with pytest.raises(ValueError, match="gate mode"):
        cake_backend._check_saved_gate(raw, None)
    with pytest.raises(ValueError, match="gate mode"):
        cake_backend._check_saved_gate(g, gp)
    other = cake_backend._gate_params(
        "raw", torch.zeros(1), torch.zeros(D), -4.0, 1, torch.device("cpu")
    )
    with pytest.raises(ValueError, match="lower_bound"):
        cake_backend._check_saved_gate(raw, other)


def test_raw_workspace_bytes_and_local_plan() -> None:
    """``kda_det_workspace_bytes_kcp(..., use_gate_in_kernel=True)`` is the raw plan's local arena (the existing backend's
    ``workspace_bytes(gate="raw")`` plus the scratch table set of the raw plan's intra record) plus the same KCP
    pools as the explicit mode; the raw arena is never larger (the gate epilogue's ``dg_int`` / ``db_int`` leave
    the backward layouts); the forward-only workspace does not depend on the mode."""

    def local(ctx, H: int, backward: bool, gate: str) -> int:
        arena = cake_backend.workspace_bytes(
            ctx.T_r, ctx.n_frag, H, backward=backward, save_mode="light", sm_count=ctx.sm_count,
            fwd_layout=cake_kcp_plan.local_fwd_layout(backward, "light"), gate=gate,
        )  # fmt: skip
        plan = cake_backend.plan(ctx.T_r, ctx.n_frag, H, ctx.sm_count, gate=gate)
        if cake_kcp_plan.intra_writes_tables(plan):
            arena += cake_backend.tables_bytes(
                ctx.T_r, ctx.n_frag, H, sm_count=ctx.sm_count
            )
        return int(arena)

    for H in (4, 64):
        for rank in (0, 1):
            ctx = _cpu_ctx(rank=rank)
            for backward in (True, False):
                g = cake_kcp_plan.kda_det_workspace_bytes_kcp(
                    ctx, H=H, backward=backward
                )
                r = cake_kcp_plan.kda_det_workspace_bytes_kcp(
                    ctx, H=H, backward=backward, use_gate_in_kernel=True
                )
                lr, lg = local(ctx, H, backward, "raw"), local(ctx, H, backward, "g")
                assert r - lr == g - lg > 0, (H, rank, backward)  # the same KCP pools
                assert lr <= lg and r <= g, (H, rank, backward, lr, lg)
                if not backward:
                    assert r == g, "the forward layouts do not depend on the gate mode"
    with pytest.raises(ValueError, match="use_gate_in_kernel must be a bool"):
        cake_kcp_plan.kda_det_workspace_bytes_kcp(
            _cpu_ctx(), H=4, use_gate_in_kernel="raw"
        )


def test_raw_workspace_refuses_the_lean_arena() -> None:
    """The raw gate mode runs on the replay local arena: ``WorkspaceDetKcp`` and ``kda_det_workspace_bytes_kcp``
    (backward and forward-only) refuse ``local_arena="lean"`` with ``use_gate_in_kernel=True``."""
    with pytest.raises(NotImplementedError, match='local_arena="lean"'):
        cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True, local_arena="lean")
    ctx = _cpu_ctx()
    for backward in (True, False):
        with pytest.raises(NotImplementedError, match='local_arena="lean"'):
            cake_kcp_plan.kda_det_workspace_bytes_kcp(
                ctx, H=4, backward=backward, use_gate_in_kernel=True, local_arena="lean"
            )


SHARED_PACKS = ((1000, 577), (50, 640, 77, 1000, 129), (1450,), (300, 64, 700))


def test_raw_workspace_bytes_of_a_list_of_contexts() -> None:
    """A list of contexts in raw mode: the largest raw local arena, the raw plans' scratch table sets of every
    distinct ``(T_r, n_frag)`` and the same KCP pools as the explicit mode (the pools do not depend on the mode);
    a list of one context gives the bytes of the context."""

    def local(ctxs, H: int, backward: bool, gate: str) -> int:
        arena, scratch = 0, {}
        for c in ctxs:
            arena = max(arena, int(cake_backend.workspace_bytes(
                c.T_r, c.n_frag, H, backward=backward, save_mode="light", sm_count=c.sm_count,
                fwd_layout=cake_kcp_plan.local_fwd_layout(backward, "light"), gate=gate,
            )))  # fmt: skip
            plan = cake_backend.plan(c.T_r, c.n_frag, H, c.sm_count, gate=gate)
            if cake_kcp_plan.intra_writes_tables(plan):
                scratch[(c.T_r, c.n_frag)] = int(
                    cake_backend.tables_bytes(c.T_r, c.n_frag, H, sm_count=c.sm_count)
                )
        return arena + sum(scratch.values())

    ctxs = [_cpu_ctx(W=1, lengths=lengths) for lengths in SHARED_PACKS]
    for H in (4, 64):
        for backward in (True, False):
            g = cake_kcp_plan.kda_det_workspace_bytes_kcp(ctxs, H=H, backward=backward)
            r = cake_kcp_plan.kda_det_workspace_bytes_kcp(
                ctxs, H=H, backward=backward, use_gate_in_kernel=True
            )
            lr, lg = local(ctxs, H, backward, "raw"), local(ctxs, H, backward, "g")
            assert r - lr == g - lg > 0, (H, backward)  # the same KCP pools
            assert r <= g, (H, backward, r, g)
            for c in ctxs:
                assert cake_kcp_plan.kda_det_workspace_bytes_kcp(
                    [c], H=H, backward=backward, use_gate_in_kernel=True
                ) == cake_kcp_plan.kda_det_workspace_bytes_kcp(
                    c, H=H, backward=backward, use_gate_in_kernel=True
                )


def test_gpu_raw_reserved_shared_workspace_is_bitwise_per_pack() -> None:
    """Four raw-mode packs of one rank (different ``T_r`` / ``n_frag``) at H = 4 and 64 on ONE reserved raw
    workspace, interleaved (saving forwards in pack order, their backwards in reverse order, replay-mode forwards
    and backwards): every output, the raw gate's gradient and the parameter partial sums are bitwise those of one
    raw workspace per pack; ``reserve`` returns ``kda_det_workspace_bytes_kcp(ctxs, H=H, use_gate_in_kernel=True)``, the
    schedule changes no byte count, and an explicit-mode call on the raw workspace is refused."""
    _skip_unless_runnable(1)
    device = torch.device("cuda", torch.cuda.current_device())
    for H in (4, 64):
        ctxs, ops = [], []
        for i, lengths in enumerate(SHARED_PACKS):
            part, cu = _partition(lengths, 1, "contiguous")
            ctxs.append(cake_kcp_plan.build_context_det_kcp(part, None, 0, device))
            glob = make_raw_inputs(lengths, H, device, seed=0x7816 + i)
            o = _local_slices(
                part, 0, cu, {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
            )
            o["GK"] = dict(
                use_gate_in_kernel=True,
                A_log=glob["A_log"],
                dt_bias=glob["dt_bias"],
                lower_bound=LB,
            )
            ops.append(o)  # fmt: skip

        def schedule(wss: list) -> list[dict]:
            n, res, saved = len(ctxs), [{} for _ in ctxs], [None] * len(ctxs)
            for i in range(n):
                o = ops[i]
                out, saved[i] = cake_kcp.chunk_kda_det_fwd_kcp(
                    o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], workspace=wss[i], **o["GK"]
                )  # fmt: skip
                res[i]["out"] = out
            for i in reversed(range(n)):
                o = ops[i]
                res[i]["grads"] = cake_kcp.chunk_kda_det_bwd_kcp(
                    o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], o["dout"], saved=saved[i],
                    workspace=wss[i], **o["GK"],
                )  # fmt: skip
            for i in reversed(range(n)):
                o = ops[i]
                out, none = cake_kcp.chunk_kda_det_fwd_kcp(
                    o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], workspace=wss[i],
                    boundary_mode="replay", **o["GK"],
                )  # fmt: skip
                assert none is None
                res[i]["out_replay"] = out
                res[i]["grads_replay"] = cake_kcp.chunk_kda_det_bwd_kcp(
                    o["q"], o["k"], o["v"], o["g"], o["beta"], ctxs[i], o["dout"], saved=None,
                    workspace=wss[i], **o["GK"],
                )  # fmt: skip
            return res

        ref = schedule(
            [cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True) for _ in ctxs]
        )
        ws = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True)
        got = ws.reserve(ctxs, H)
        assert (
            got
            == ws.bytes
            == cake_kcp_plan.kda_det_workspace_bytes_kcp(
                ctxs, H=H, use_gate_in_kernel=True
            )
        ), H
        for rnd in range(2):
            res = schedule([ws] * len(ctxs))
            for i in range(len(ctxs)):
                for key, val in ref[i].items():
                    a = (val,) if isinstance(val, torch.Tensor) else tuple(val)
                    b = (
                        (res[i][key],)
                        if isinstance(val, torch.Tensor)
                        else tuple(res[i][key])
                    )
                    assert len(a) == len(b) and all(
                        _bits_equal(x, y) for x, y in zip(a, b, strict=True)
                    ), (H, rnd, i, key)
            assert ws.bytes == got, (H, rnd)
        o = ops[0]
        with pytest.raises(ValueError, match="gate mode"):
            cake_kcp.chunk_kda_det_fwd_kcp(
                o["q"], o["k"], o["v"], torch_gate(o["g"], o["GK"]["A_log"], o["GK"]["dt_bias"]),
                o["beta"], ctxs[0], workspace=ws,
            )  # fmt: skip
        del ws, ref, res
        gc.collect()


def test_raw_handle_fingerprint_records_the_gate() -> None:
    """A recompute handle's fingerprint carries the gate mode and the FP32 lower bound (``modes``) and lists only
    tensors under ``operands`` (the raw mode's strides and lower bound are host scalars): a raw entry never binds a
    recompute in the explicit mode or with another lower bound."""
    H = 4
    ctx = _cpu_ctx()
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
    ops = _cpu_ops(ctx, H)

    def call(gate: str, lb: float | None):
        c = types.SimpleNamespace(
            tb=tb, T_r=int(ctx.T_r), H=H, scale=0.5, exchange="a2a",
            opts=cake_kcp._options({}), gate=gate, gate_params=None, ops=dict(ops),
        )  # fmt: skip
        if gate == "raw":
            c.gate_params = cake_backend._gate_params(
                "raw", torch.zeros(H), torch.zeros(H * D), lb, H, torch.device("cpu")
            )
            g = c.ops.pop("g")
            c.ops.update(cake_backend._raw_values(g, c.gate_params))
        return c

    def fp(c):
        return dict(
            cake_kcp._handle_fingerprint(
                c, save_mode="light", keep="states", **cake_kcp._gate_modes(c)
            )
        )

    f_g, f_r, f_r4 = fp(call("g", None)), fp(call("raw", -5.0)), fp(call("raw", -4.0))
    assert dict(f_g["modes"]) == {"use_gate_in_kernel": False, "lower_bound": None}
    assert dict(f_r["modes"]) == {"use_gate_in_kernel": True, "lower_bound": -5.0}
    assert f_r["modes"] != f_r4["modes"] and f_r["modes"] != f_g["modes"]
    names = [n for n, *_ in f_r["operands"]]
    assert names == ["q", "k", "v", "beta", "g_raw", "A_log", "dt_bias"], names
    assert [n for n, *_ in f_g["operands"]] == ["q", "k", "v", "g", "beta"]


def test_reduce_gate_param_grads_one_rank_and_checks() -> None:
    """On one rank the helper returns fresh copies of the inputs in their shapes; it refuses other dtypes and
    shapes, and several ranks without an initialized process group (it never returns a partial sum)."""
    ctx = types.SimpleNamespace(world_size=1, group=None)
    a = torch.arange(4, dtype=torch.float32)
    for b in (torch.ones(4 * D), torch.ones(4, D)):
        ra, rb = cake_kcp.reduce_gate_param_grads_det_kcp(ctx, a, b)
        assert torch.equal(ra, a) and torch.equal(rb, b) and rb.shape == b.shape
        assert ra.data_ptr() != a.data_ptr() and rb.data_ptr() != b.data_ptr()
    with pytest.raises(ValueError, match="float32"):
        cake_kcp.reduce_gate_param_grads_det_kcp(ctx, a.double(), torch.ones(4 * D))
    with pytest.raises(ValueError, match="ddt_bias"):
        cake_kcp.reduce_gate_param_grads_det_kcp(ctx, a, torch.ones(4 * D + 1))
    with pytest.raises(RuntimeError, match="process group"):
        cake_kcp.reduce_gate_param_grads_det_kcp(
            types.SimpleNamespace(world_size=2, group=None), a, torch.ones(4 * D)
        )


LOCAL_RECORD_CASES = [
    (4, "sm_100a", 148),
    (64, "sm_100a", 148),
    (4, "sm_103a", 152),
    (64, "sm_103a", 152),
    (64, "sm_107a", 216),
]


@pytest.mark.parametrize(
    "H,arch,sm",
    LOCAL_RECORD_CASES,
    ids=[f"h{h}-{a}-{s}" for h, a, s in LOCAL_RECORD_CASES],
)
def test_raw_local_stage_values_cover_the_records(H, arch, sm) -> None:
    """The kernel backend's local stages on a raw plan (``cake_backend.plan(..., gate="raw")``): the forward, a
    saved-mode and a replay backward bind every logical name of every record they launch (the raw intra / prep /
    GLM intra records read ``g_raw``, its strides, ``A_log``, ``dt_bias``, ``lower_bound``; the GLM intra program
    writes ``dg_raw`` and the partials ``dA_part`` / ``db_part``; ``param_finalize`` reads ``seq_chunk_start``
    bounded at ``num_seqs`` = the rank's fragments and writes ``dA_log`` / ``ddt_bias`` on its ``(H, 1, 1)`` grid);
    no gate epilogue launches; every stage outside ``cake_jit.GATE_STAGES`` launches exactly the explicit plan's
    record on the same grid, and the raw records replace the explicit ones stage for stage."""
    ctx = _cpu_ctx(sm=sm)
    assert ctx.T_r > 0
    plans = {
        gate: cake_backend.plan(ctx.T_r, ctx.n_frag, H, sm, arch, gate=gate)
        for gate in ("g", "raw")
    }
    tb = cake_kcp._Tables(ctx, H, fused_walks=1)
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
    T = int(ctx.T_r)
    raw_g = torch.zeros(T, H, D, dtype=torch.bfloat16)
    gp = cake_backend._gate_params(
        "raw", torch.zeros(H), torch.zeros(H * D), LB, H, torch.device("cpu")
    )
    ops = {
        "g": {n: torch.zeros(1) for n in ("q", "k", "v", "g", "beta")},
        "raw": {
            **{n: torch.zeros(1) for n in ("q", "k", "v", "beta")},
            **cake_backend._raw_values(raw_g, gp),
        },
    }
    grads = {
        "g": {n: torch.zeros(1) for n in ("dq", "dk", "dv", "dg", "dbeta")},
        "raw": {
            n: torch.zeros(1)
            for n in ("dq", "dk", "dv", "dbeta", "dg_raw", "dA_part", "db_part", "dA_log", "ddt_bias")
        },
    }  # fmt: skip

    def run(gate: str, backward: bool, saved) -> list:
        call = types.SimpleNamespace(
            local=Arena(plans[gate]), ctx=ctx, tb=tb, T_r=T, H=H, backward=backward, host_trim=1,
            ops={**ops[gate], **({"dout": torch.zeros(1)} if backward else {})}, scale=0.5, scratch={},
            local_values=None, out=None if backward else torch.zeros(1), grads=grads[gate] if backward else None,
            device=torch.device("cpu"), opts=cake_kcp._options({"spec3_probe": "off"}),
        )  # fmt: skip
        backend.begin_call(call)
        if backward:
            stages = [("bwd_prep", {"saved": saved}), ("fwd_intra", {}), ("wy", {}), ("dav", {}),
                      ("dqkg", {}), ("glm_intra", {})]  # fmt: skip
            stages.append(
                ("param_finalize", {}) if gate == "raw" else ("gate_epilogue", {})
            )
        else:
            stages = [("fwd_intra", {}), ("wy_fwd", {}), ("fwd_o", {})]
        for name, extra in stages:
            backend.stage(name, call, **extra)
        done = list(launches)
        launches.clear()
        return done

    saved = types.SimpleNamespace(S_in=torch.zeros(1), c1_recv=None, M_seg=None)
    res = {
        (gate, kind): run(gate, kind != "fwd", saved if kind == "bwd" else None)
        for gate in ("g", "raw")
        for kind in ("fwd", "bwd", "replay")
    }
    pr = plans["raw"]
    b = cake_kcp.LOCAL_PATH_BWD
    for kind in ("fwd", "bwd", "replay"):
        for src, record, _grid, values in res[("raw", kind)]:
            mod = (cake_jit if src == "backend" else cake_kcp_jit).MODULES[record]
            missing = set(mod["logical"].values()) - set(values)
            assert not missing, (kind, record, sorted(missing))
            assert "g" not in values, (kind, record)  # no FP32 gate on the raw route
            if src == "backend":
                assert (
                    mod.get("gate", "g") == "raw"
                    or mod["stage"] not in cake_jit.GATE_STAGES
                )
    assert [(k, r, g) for k, r, g, _ in res[("raw", "bwd")]] == [
        (k, r, g) for k, r, g, _ in res[("raw", "replay")]
    ]
    # stage for stage: the same records outside the gate stages, the raw class inside them
    for kind in ("fwd", "bwd"):
        g_l = [
            (k, r, gr)
            for k, r, gr, _ in res[("g", kind)]
            if not (k == "backend" and cake_jit.MODULES[r]["stage"] == "gate_epilogue")
        ]
        r_l = [
            (k, r, gr)
            for k, r, gr, _ in res[("raw", kind)]
            if not (k == "backend" and cake_jit.MODULES[r]["stage"] == "param_finalize")
        ]
        assert len(g_l) == len(r_l), (kind, g_l, r_l)
        for (kg, rg, gg), (kr, rr, gr) in zip(g_l, r_l, strict=True):
            stage = cake_jit.MODULES[rg]["stage"] if kg == "backend" else rg
            if kg == "backend" and stage in cake_jit.GATE_STAGES:
                assert cake_jit.MODULES[rr]["stage"] == stage and gr == gg, (rg, rr)
                assert cake_jit.MODULES[rr].get("gate") == "raw", rr
            else:
                assert (kr, rr, gr) == (kg, rg, gg), (kind, rg, rr)
    bwd = res[("raw", "bwd")]
    records = [r for _, r, _, _ in bwd]
    assert records[-2:] == [pr.record("glm_intra", b), "param_finalize"], records
    assert records[-2] == "glm_intra_raw" and records[0] == pr.record("bwd_prep", b)
    assert records[0] == "bwd_prep_g8_replay_raw"
    assert not any(
        cake_jit.MODULES[r]["stage"] == "gate_epilogue"
        for _, r, _, _ in bwd
        if r in cake_jit.MODULES
    )
    fin = bwd[-1][3]
    assert bwd[-1][2] == pr.grid("param_finalize", b) == (H, 1, 1)
    assert (
        fin["num_seqs"] == ctx.n_frag
        and fin["seq_chunk_start"] is ctx.tables["seq_chunk_start"]
    )
    assert (
        fin["dA_part"] is grads["raw"]["dA_part"]
        and fin["ddt_bias"] is grads["raw"]["ddt_bias"]
    )
    gi = bwd[-2][3]
    assert (
        gi["g_raw"] is raw_g
        and gi["lower_bound"] == LB
        and gi["dg_raw"] is grads["raw"]["dg_raw"]
    )
    assert (gi["g_raw_row_stride"], gi["g_raw_head_stride"]) == (H * D, D)
    fwd = res[("raw", "fwd")]
    assert fwd[0][1] == pr.record("fwd_intra", cake_kcp.LOCAL_PATH_FWD)
    assert cake_jit.MODULES[fwd[0][1]].get("gate") == "raw"
    assert (
        fwd[0][3]["write_ext"],
        fwd[0][3]["write_int"],
        fwd[0][3]["write_prep"],
    ) == (0, 1, 1)


# --------------------------------------------------------------------------- GPU rows (one process per rank)

# (lengths, H, W, layout): W = 1 (in-process), CP2 / CP4 (one node), CP8 (two nodes, Slurm per-task); boundaries
# inside chunks, a document over >= 3 ranks (zigzag, per-document zigzag), one-token documents and fragments,
# explicit ragged tables with a zero-token rank and a one-token rank, a 64-head row.
ROWS = [
    ((100, 640, 77, 1000, 129), 4, 1, "contiguous"),
    ((700, 1, 1300, 999), 64, 1, "contiguous"),
    ((100, 640, 77, 1000, 129), 4, 2, "contiguous"),
    ((3000,), 4, 2, "zigzag_per_document"),
    ((100, 640, 77, 1000, 129), 4, 4, "contiguous"),
    ((3000,), 4, 4, "zigzag"),
    ((3000,), 4, 4, "zigzag_per_document"),
    ((2048, 5, 3000, 1), 4, 4, "zigzag_per_document"),
    ((1200 + 900 + 900,), 4, 4, "ragged:1200,0,900,900"),
    ((5000, 1003), 1, 4, "ragged:1,2,4997,1003"),
    ((700, 1, 1300, 999), 64, 4, "zigzag_per_document"),
    ((3000, 2048, 5, 1), 4, 8, "zigzag_per_document"),
    ((4000, 1003), 4, 8, "ragged:1,0,999,1000,1000,1000,1,1002"),
    ((700, 1, 1300, 999), 64, 8, "zigzag"),
]


def _row_id(row) -> str:
    lengths, H, W, layout = row
    return "s" + "_".join(map(str, lengths)) + f"-h{H}-w{W}-{layout.split(':')[0]}"


def _bits_equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)


def _same(x: dict, y: dict, names=TENSORS) -> bool:
    return all(_bits_equal(x[n], y[n]) for n in names)


def _collective_seq(rec: dict) -> list:
    return [s for s in rec["sequence"] if ":" in s]


def explicit_param_grads(
    dg: torch.Tensor, raw: torch.Tensor, A_log: torch.Tensor, dt_bias: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The explicit route's ``dA_log`` [H] and ``ddt_bias`` (``dt_bias``'s shape) in FP64, by the chain rule from the
    gradient ``dg`` of the activated gate ``g = lb * sigmoid(z)``, ``z = exp(A_log) * (raw + dt_bias)``."""
    H = raw.shape[-2]
    a = torch.exp(A_log.double()).view(1, H, 1)
    z = a * (raw.double() + dt_bias.double().view(1, H, D))
    s = torch.sigmoid(z)
    dz = dg.double().reshape(z.shape) * LB * s * (1 - s)
    return (dz * z).sum((0, 2)), (dz * a).sum(0).reshape(dt_bias.shape)


def _param_check(cand: torch.Tensor, ref: torch.Tensor, bound: float) -> bool:
    return (
        bool(torch.isfinite(cand).all()) and _max_normalized_error(cand, ref) <= bound
    )


def param_vs_explicit(cand, expl, ref, W: int, *, parts_of: str, parts: list) -> dict:
    """Criterion (d) of #7816 for one parameter gradient of one row: the raw route is no less accurate against FP64
    than the explicit route.  On one seed this is the sanity rule of the single-rank protocol
    (``test_cake_kda_det_train_raw_gate_accuracy``): the worst normalized error ``|err| / (atol + rtol |ref|)`` is at most
    ``max(1, PARAM_SANITY_FACTOR x`` the explicit route's ``)``; the seeded statistic is
    :func:`test_raw_gate_kcp_param_grads_no_less_accurate_than_explicit`.  Negative controls must fail the same rule:
    the raw error stretched to 1.5 x the bound and, at W > 1, the rank-order sum without the largest partial."""
    ref = ref.to(torch.float64)
    m_raw, m_exp = _max_normalized_error(cand, ref), _max_normalized_error(expl, ref)
    bound = max(1.0, PARAM_SANITY_FACTOR * m_exp)
    err = cand.to(torch.float64) - ref
    scale = 1.5 * bound
    stretched = ref + (
        err * (scale / m_raw)
        if m_raw > 0
        else scale * (ELEMENTWISE_ATOL + ELEMENTWISE_RTOL * ref.abs())
    )
    controls = {"error_stretched_1.5x_bound": _param_check(stretched, ref, bound)}
    if W > 1:
        k = 0 if parts_of == "dA_log" else 1
        drop = max(range(W), key=lambda r: float(parts[r][k].double().norm()))
        kept = [parts[r][k] for r in range(W) if r != drop]
        total = kept[0].clone()
        for t in kept[1:]:
            total += t
        controls[f"partial_of_rank_{drop}_left_out"] = _param_check(
            total.reshape(ref.shape), ref, bound
        )
    rel_r = error_metrics(cand.to(torch.float64), ref)["rel_rms"]
    rel_e = error_metrics(expl, ref)["rel_rms"]
    ok = _param_check(cand.reshape(ref.shape), ref, bound)
    return {
        "gate": "raw_vs_explicit",
        "max_norm_raw": m_raw,
        "max_norm_explicit": m_exp,
        "bound": bound,
        "explicit_rel_l2": rel_e,
        "raw_over_explicit_rel_l2": rel_r / rel_e if rel_e > 0 else None,
        "negative_controls_pass": controls,
        "pass": ok and not any(controls.values()),
    }


def _run_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """One rank: the raw-mode KCP passes, their bitwise flags, the explicit-mode launch comparison, the reduction
    helper; gathered to rank 0 and compared with the single-rank raw path and the FP64 reference there."""
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
    glob = make_raw_inputs(lengths, H, device)
    A_log, dt_bias = glob["A_log"], glob["dt_bias"]
    tokens = {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
    ops = _local_slices(part, rank, cu, tokens)
    q, k, v, g, beta, dout = (ops[n] for n in ("q", "k", "v", "g", "beta", "dout"))
    GK = dict(use_gate_in_kernel=True, A_log=A_log, dt_bias=dt_bias, lower_bound=LB)
    ws = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True)
    recs: dict = {}

    def saved_pass(raw=g, workspace=ws, tag=None, **opts):
        out, saved = cake_kcp.chunk_kda_det_fwd_kcp(
            q, k, v, raw, beta, ctx, workspace=workspace, **GK, **opts
        )
        rec_f = dict(cake_kcp.LAST_CALL["fwd"])
        bwd_opts = {key: val for key, val in opts.items() if key != "boundary_mode"}
        grads = cake_kcp.chunk_kda_det_bwd_kcp(
            q, k, v, raw, beta, ctx, dout, saved=saved, workspace=workspace, **GK, **bwd_opts
        )  # fmt: skip
        if tag is not None:
            recs[tag] = (rec_f, dict(cake_kcp.LAST_CALL["bwd"]))
        return {"out": out, **dict(zip(TOKEN_GRADS + PARAM_GRADS, grads, strict=True))}

    def autograd_pass(ckpt: bool = False, handle=None, workspace=ws):
        leaves = [
            x.detach().clone().requires_grad_(True)
            for x in (q, k, v, g, beta, A_log, dt_bias)
        ]

        def fn(q_, k_, v_, g_, b_, a_, d_, *h):
            return cake_kcp.chunk_kda_det_kcp(
                q_, k_, v_, g_, b_, ctx, None, "light", "saved", workspace,
                recompute_handle=h[0] if h else None,
                use_gate_in_kernel=True, A_log=a_, dt_bias=d_, lower_bound=LB,
            )  # fmt: skip

        extra = (handle,) if handle is not None else ()
        out = (
            checkpoint(fn, *leaves, *extra, use_reentrant=False)
            if ckpt
            else fn(*leaves, *extra)
        )
        out.backward(dout)
        names = ("dq", "dk", "dv", "dg", "dbeta", "dA_log", "ddt_bias")
        return {
            "out": out.detach(),
            **{n: lf.grad for n, lf in zip(names, leaves, strict=True)},
        }

    def reduced(res: dict) -> tuple[torch.Tensor, torch.Tensor]:
        return cake_kcp.reduce_gate_param_grads_det_kcp(
            ctx, res["dA_log"], res["ddt_bias"]
        )

    pending = bool(getattr(cake_kcp.get_backend(), "_spec3_pending", False))
    res = saved_pass(tag="raw", spec3_probe="record")
    spec3 = (cake_kcp.LAST_CALL.get("bwd") or {}).get("spec3")
    runs = {
        "saved_eq_replay": saved_pass(boundary_mode="replay"),
        "repeat": saved_pass(),
        "autograd_eq_saved": autograd_pass(),
        "ckpt_eq_direct": autograd_pass(ckpt=True),
        "handle_eq_ckpt": autograd_pass(
            ckpt=True, handle=cake_kcp.RecomputeHandleDetKcp()
        ),
        "save_transfer_eq": saved_pass(save_transfer=False),
        "ws_none_eq": autograd_pass(workspace=None),
    }
    # a strided raw view (a column slice of a wider fused projection, the single-rank rule's example) reads the same
    # elements: the same bits, no copy
    T = int(ctx.T_r)
    wide = torch.zeros((T, (H + 2) * D), dtype=torch.bfloat16, device=device)
    wide[:, 8 : 8 + H * D] = g.reshape(T, H * D)
    view = wide[:, 8 : 8 + H * D].view(T, H, D)
    runs["strided_eq"] = saved_pass(raw=view)
    flags = {name: _same(res, r) for name, r in runs.items()}
    if W > 1 and int(ctx.n_d_max) - 1 <= cake_kcp.RING_MAX_HOPS:
        ring = saved_pass(exchange="ring", save_transfer=False)
        flags["ring_eq_a2a"] = _same(runs["save_transfer_eq"], ring)
        runs["ring_eq_a2a"] = ring
    # graph replay: every rank captures and replays the same calls
    cake_kcp.graph_cache_clear()
    events = []
    try:
        for _ in range(len(GRAPH_EVENTS)):
            res_graph = {n: t.clone() for n, t in saved_pass(graph="replay").items()}
            events.append(
                [str(cake_kcp.LAST_CALL[x].get("graph_event")) for x in ("fwd", "bwd")]
            )
        torch.cuda.synchronize(device)
    except BaseException as exc:
        traceback.clear_frames(exc.__traceback__)
        raise
    finally:
        cake_kcp.graph_cache_clear()
    flags["graph_replay"] = _graph_events_ok(events) and _same(res, res_graph)
    runs["graph_replay"] = res_graph
    # the mode is checked against the saved set and the workspace: symmetric on every rank, before any collective
    _, saved_r = cake_kcp.chunk_kda_det_fwd_kcp(
        q, k, v, g, beta, ctx, workspace=ws, **GK
    )
    refusals = []
    for kw, match in (
        ({**GK, "lower_bound": -4.0}, "lower_bound"),
        ({}, "gate mode"),
    ):
        try:
            cake_kcp.chunk_kda_det_bwd_kcp(
                q, k, v, (g if kw else g.float()), beta, ctx, dout, saved=saved_r,
                workspace=ws if kw else cake_kcp.WorkspaceDetKcp(), **kw,
            )  # fmt: skip
            refusals.append(False)
        except ValueError as exc:
            refusals.append(match in str(exc))
    flags["mismatch_refused"] = all(refusals)
    del saved_r
    # the explicit mode on the same activated gate (torch FP32): the same launches in the same order with the same
    # collectives -- gate_epilogue (a stage that launches only on sm_100a) in place of param_finalize
    g_exp = raw_gate(g, A_log, dt_bias, LB, torch.float32).contiguous()
    ws_g = cake_kcp.WorkspaceDetKcp()
    out_e, saved_e = cake_kcp.chunk_kda_det_fwd_kcp(
        q, k, v, g_exp, beta, ctx, workspace=ws_g
    )
    rec_fe = dict(cake_kcp.LAST_CALL["fwd"])
    grads_e = cake_kcp.chunk_kda_det_bwd_kcp(
        q, k, v, g_exp, beta, ctx, dout, saved=saved_e, workspace=ws_g
    )
    rec_be = dict(cake_kcp.LAST_CALL["bwd"])
    rec_fr, rec_br = recs["raw"]
    mapped = [
        "param_finalize" if s == "gate_epilogue" else s for s in rec_be["sequence"]
    ]
    launch_cmp = {
        "fwd_collectives": [rec_fr["n_collectives"], rec_fe["n_collectives"]],
        "bwd_collectives": [rec_br["n_collectives"], rec_be["n_collectives"]],
        "fwd_launches": [rec_fr["n_launches"], rec_fe["n_launches"]],
        "bwd_launches": [rec_br["n_launches"], rec_be["n_launches"]],
        "gates": [rec_fr["gate"], rec_br["gate"], rec_fe["gate"], rec_be["gate"]],
    }
    flags["collectives_eq_explicit"] = (
        _collective_seq(rec_fr) == _collective_seq(rec_fe)
        and _collective_seq(rec_br) == _collective_seq(rec_be)
        and rec_fr["n_collectives"] == rec_fe["n_collectives"]
        and rec_br["n_collectives"] == rec_be["n_collectives"]
    )
    flags["launches_eq_explicit"] = (
        rec_fr["sequence"] == rec_fe["sequence"] and rec_br["sequence"] == mapped
        and launch_cmp["gates"] == ["raw", "raw", "g", "g"]
    )  # fmt: skip
    explicit = {
        "out": out_e,
        **dict(zip(("dq", "dk", "dv", "dbeta", "dg"), grads_e, strict=True)),
    }
    del ws_g, saved_e
    # the reduction helper: the same bits on every rank and under every mode
    red = reduced(res)
    red_ok = True
    for r in runs.values():
        ra, rb = reduced(r)
        red_ok &= _bits_equal(ra, red[0]) and _bits_equal(rb, red[1])
    flags["reduce_eq_modes"] = bool(red_ok)
    if W > 1:
        every: list = [None] * W
        dist.all_gather_object(every, (red[0].cpu(), red[1].cpu()))
        flags["reduce_eq_ranks"] = all(
            torch.equal(e[0], every[0][0]) and torch.equal(e[1], every[0][1])
            for e in every
        )
    else:
        flags["reduce_eq_ranks"] = True
    # a backend with local stages whose probe was pending must have run it on the first pass (reported, not gated
    # beyond 'checked': the forward's intra program writes gk row 63 alone)
    flags["spec3"] = (
        not pending
        or (spec3 is None and T == 0)
        or bool(spec3 and spec3.get("checked"))
    )
    # gather the token tensors and the per-rank partials
    gathered: dict = {}
    for n in ("out",) + TOKEN_GRADS:
        srcs = [(res, n), (explicit, f"explicit_{n}")]
        for src, key in srcs:
            if W > 1:
                objs: list = [None] * W
                dist.all_gather_object(objs, src[n].cpu())
            else:
                objs = [src[n].cpu()]
            gathered[key] = objs
    parts: list = [None] * W
    if W > 1:
        dist.all_gather_object(parts, (res["dA_log"].cpu(), res["ddt_bias"].cpu()))
    else:
        parts = [(res["dA_log"].cpu(), res["ddt_bias"].cpu())]
    flag_list: list = [None] * W
    spec3_list: list = [None] * W
    launch_list: list = [None] * W
    if W > 1:
        dist.all_gather_object(flag_list, flags)
        dist.all_gather_object(spec3_list, spec3)
        dist.all_gather_object(launch_list, launch_cmp)
    else:
        flag_list, spec3_list, launch_list = [flags], [spec3], [launch_cmp]
    report: dict = {
        "rank": rank,
        "world": W,
        "row": _row_id(row),
        "pass": True,
        "flags": {},
        "flags_by_rank": flag_list,
        "metrics": {},
        "spec3_by_rank": spec3_list,
        "launches_by_rank": launch_list,
        "graph_events": events,
        "device": torch.cuda.get_device_name(device),
        "arch": cake_jit.device_arch(device.index),
    }
    for key in flags:
        report["flags"][key] = all(f[key] for f in flag_list)
        report["pass"] &= report["flags"][key]
    if rank == 0:
        cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
        # the single-rank raw path on the whole pack
        out_s, saved_s = cake_backend.chunk_kda_det_fwd(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, **GK
        )
        grads_s = cake_backend.chunk_kda_det_bwd(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], cu_t, glob["dout"],
            saved=saved_s, **GK,
        )  # fmt: skip
        single = {
            "out": out_s,
            **dict(zip(TOKEN_GRADS + PARAM_GRADS, grads_s, strict=True)),
        }
        del saved_s
        # the exact FP64 reference (on this rank's device: the per-token recurrence in FP64)
        ref_out, ref_g = kda_exact_raw_with_grads(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], A_log, dt_bias, glob["dout"], cu_t,
            SCALE, LB,
        )  # fmt: skip
        refs = {n: t.cpu() for n, t in {"out": ref_out, **ref_g}.items()}
        del ref_out, ref_g
        cand = {
            n: _to_global(part, gathered[n], cu, total) for n in ("out",) + TOKEN_GRADS
        }
        sums = [torch.zeros_like(parts[0][0]), torch.zeros_like(parts[0][1])]
        for pa, pb in parts:  # the helper's order: FP32, rank 0 first
            sums[0] += pa
            sums[1] += pb
        cand["dA_log"], cand["ddt_bias"] = red[0].cpu(), red[1].cpu()
        report["reduce_eq_rank_order_sum"] = bool(
            torch.equal(sums[0], cand["dA_log"])
            and torch.equal(sums[1], cand["ddt_bias"])
        )
        report["pass"] &= report["reduce_eq_rank_order_sum"]
        expl = {
            n: _to_global(part, gathered[f"explicit_{n}"], cu, total)
            for n in ("out",) + TOKEN_GRADS
        }
        # the explicit route's parameter gradients: the FP64 chain rule from its FP32 activated-gate gradient
        expl["dA_log"], expl["ddt_bias"] = explicit_param_grads(
            expl["dg"], glob["g"].cpu(), A_log.cpu(), dt_bias.cpu()
        )
        for n in TENSORS:
            ref = refs[n]
            c = cand[n].to(torch.float64)
            s = single[n].cpu().to(torch.float64)
            mk, ms = error_metrics(c, ref), error_metrics(s, ref)
            rk, rs = mk["rel_rms"], ms["rel_rms"]
            ratio = rk / rs if rs > 0 else (1.0 if rk == 0 else float("inf"))
            finite = bool(torch.isfinite(c).all())
            report["metrics"][n] = {
                "kcp_rel_l2": rk,
                "single_rel_l2": rs,
                "ratio": ratio,
                "kcp_max_abs": mk["max_abs"],
                "single_max_abs": ms["max_abs"],
                "bitwise_eq_single": bool(_bits_equal(cand[n].cpu(), single[n].cpu())),
                "finite": finite,
                "pass": finite and ratio <= ACC_GATE,
            }
            # criterion (d): no less accurate than the explicit route (see the module doc)
            if n in PARAM_GRADS:
                report["metrics"][n].update(
                    param_vs_explicit(cand[n], expl[n], ref, W, parts_of=n, parts=parts)
                )
            report["pass"] &= report["metrics"][n]["pass"]
        # report only: the explicit KCP route against the FP64 reference (``dg``: the activated gate's gradient,
        # another quantity than the raw gate's)
        report["explicit_rel_l2"] = {
            n: error_metrics(expl[n].to(torch.float64), refs[n])["rel_rms"]
            for n in ("out", "dq", "dk", "dv", "dbeta", "dA_log", "ddt_bias")
        }
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, report["pass"])
        report["pass"] = all(verdict)
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, "raw-" + _row_id(row) + ".json"), "w") as fh:
            json.dump(report, fh, indent=1, default=str)
    del runs, res, explicit
    gc.collect()
    return report


def _spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        report = _run_rank(rank, W, row, port, result_dir)
        if not report["pass"]:
            raise AssertionError(f"rank {rank}: {report}")
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


def _skip_unless_runnable(W: int) -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA device required")
    if torch.cuda.device_count() < W:
        pytest.skip(f"{W} CUDA devices required, {torch.cuda.device_count()} visible")
    if not _programs_exported():
        pytest.skip("the KCP programs are not exported in the package's build.json yet")
    if cake_jit.device_arch(torch.cuda.current_device()) not in cake_jit.ARCHES:
        pytest.skip("the current device's architecture is not covered by the package")


@pytest.mark.parametrize("row", ROWS, ids=_row_id)
def test_raw_gate_rows_match_single_rank_and_fp64(row) -> None:
    lengths, H, W, layout = row
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
        and W > 1
    ):
        if not _programs_exported():
            pytest.skip(
                "the KCP programs are not exported in the package's build.json yet"
            )
        report = _run_rank(
            int(os.environ["SLURM_PROCID"]), W, row, int(os.environ.get("MASTER_PORT", "29500")),
            os.environ.get("KCP_TEST_RESULT_DIR"),
        )  # fmt: skip
        assert report["pass"], report
        return
    _skip_unless_runnable(W)
    if W == 1:
        report = _run_rank(0, 1, row, 0, os.environ.get("KCP_TEST_RESULT_DIR"))
        assert report["pass"], report
        return
    with tempfile.TemporaryDirectory(prefix="cake_kcp_raw_") as tmp:
        result_dir = os.environ.get("KCP_TEST_RESULT_DIR") or tmp
        mp.start_processes(
            _spawn_entry,
            args=(W, row, _free_port(), result_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(result_dir, "raw-" + _row_id(row) + ".json")) as fh:
            report = json.load(fh)
    assert report["pass"], report


# --------------------------------------------------------------------------- criterion (d) on the KCP route
# #7816: "outputs and gradients, including both gate-parameter gradients, are no less accurate against FP64 than the
# existing explicit gate-plus-KDA path".  For dA_log / ddt_bias on the KCP route the explicit path is the explicit KCP
# route on the same rows (its parameter gradients by the FP64 chain rule from its FP32 activated-gate gradient), and
# the comparison is the seeded protocol of the design of record (7816 design v1.1 section 5.4, implemented by
# test_cake_kda_det_train_raw_gate_accuracy, whose constants are imported): per (tensor, row) the log-ratios
# l_s = log(err_raw / err_explicit) over TEST_SEEDS, U = mean + t_{0.975, N-1} sd / sqrt(N), and the critical value c
# the 1 - ALPHA / m quantile of U over B_BOOT bootstrap resamples (seed BOOT_SEED) of the A/A log-ratios (the explicit
# route against itself with the gate moved by a seeded +-1 ulp, AA_SEEDS).  Explicit arms as in the single-rank
# criterion: E_T (torch FP32 gate) and E_F (flash-linear-attention's gate; left out with the recorded reason when it
# does not import, or required with ..._REQUIRE_FLA=1).  m = 2 tensors x the rows below x the arms present.  Every test
# seed runs the raw route twice: the per-rank partials and the reduced sums must be bitwise identical.
# A single seed per row is not a usable statistic for these H-element sums (the per-seed ratio KCP / single-rank of
# dA_log spans 0.3-3.7 on both routes at W > 1).  Negative controls: the raw arm with its error stretched 1.25x, and
# with the largest per-rank partial left out of the sum, must fail.  Minutes per row: env-gated, run on compute.
ENV_KCP_PROTOCOL = "FLASHINFER_CAKE_KDA_TRAIN_KCP_RAW_GATE_ACCURACY"
ENV_KCP_PROTOCOL_REQUIRE_FLA = ENV_KCP_PROTOCOL + "_REQUIRE_FLA"
PROTOCOL_ROWS = [
    ((3000,), 4, 2, "zigzag_per_document"),
    ((3000,), 4, 4, "zigzag_per_document"),
    ((3000, 2048, 5, 1), 4, 8, "zigzag_per_document"),
    ((4000, 1003), 4, 8, "ragged:1,0,999,1000,1000,1000,1,1002"),
]
CONTROL_STRETCH = 1.25


def _protocol_arms(device) -> tuple[dict, dict]:
    """The explicit arms present (E_T always, E_F when flash-linear-attention's gate imports and gives the same
    values) and the FLA record; ``..._REQUIRE_FLA=1`` fails without E_F or with another ``FLA_COMMIT``."""
    arms = {"E_T": torch_gate}
    fla, info = fla_kda_gate(LB)
    if fla is not None:
        reason = check_fla_gate(fla, make_raw_inputs((65,), 2, device, seed=1), LB)
        if reason is None:
            arms["E_F"] = fla
        else:
            info["unavailable"] = reason
    if os.environ.get(ENV_KCP_PROTOCOL_REQUIRE_FLA, "") in ("1", "true", "yes"):
        if "E_F" not in arms:
            raise RuntimeError(f"E_F required but unavailable: {info}")
        env, found = info.get("commit_env"), info.get("git_commit")
        if found and env and not found.startswith(env):
            raise RuntimeError(
                f"E_F required: FLA_COMMIT={env}, imported checkout {found}"
            )
    return arms, info


def _protocol_rank(rank: int, W: int, row, port: int, result_dir: str | None) -> dict:
    """One rank of the protocol for one row; rank 0 holds the record (errors, log-ratios, U, c, verdicts)."""
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
    cu_t = torch.tensor(cu, dtype=torch.int32, device=device)
    ws_r = cake_kcp.WorkspaceDetKcp(use_gate_in_kernel=True)
    ws_e = cake_kcp.WorkspaceDetKcp()

    def gather(x):
        if W == 1:
            return [x]
        out: list = [None] * W
        dist.all_gather_object(out, x)
        return out

    arms, fla_info = _protocol_arms(device)
    arm_sets = gather(sorted(arms))
    # the same arms on every rank
    assert all(a == arm_sets[0] for a in arm_sets), arm_sets
    m = len(PARAM_GRADS) * len(PROTOCOL_ROWS) * len(arms)

    def inputs(seed):
        glob = make_raw_inputs(lengths, H, device, seed=seed)
        loc = _local_slices(
            part, rank, cu, {n: glob[n] for n in ("q", "k", "v", "g", "beta", "dout")}
        )
        return glob, loc

    def reference(glob):
        _, rg = kda_exact_raw_with_grads(
            glob["q"], glob["k"], glob["v"], glob["g"], glob["beta"], glob["A_log"], glob["dt_bias"], glob["dout"],
            cu_t, SCALE, LB,
        )  # fmt: skip
        return {n: rg[n].cpu() for n in PARAM_GRADS}

    def explicit_arm(glob, loc, gate_fn):
        g = _local_slices(
            part,
            rank,
            cu,
            {"g": gate_fn(glob["g"], glob["A_log"], glob["dt_bias"]).detach()},
        )["g"]
        g = g.contiguous()
        _, saved = cake_kcp.chunk_kda_det_fwd_kcp(
            loc["q"], loc["k"], loc["v"], g, loc["beta"], ctx, workspace=ws_e
        )
        grads = cake_kcp.chunk_kda_det_bwd_kcp(
            loc["q"], loc["k"], loc["v"], g, loc["beta"], ctx, loc["dout"], saved=saved, workspace=ws_e
        )  # fmt: skip
        dgs = gather(grads[4].cpu())
        if rank != 0:
            return None
        dg = _to_global(part, dgs, cu, total)
        a, b = explicit_param_grads(
            dg, glob["g"].cpu(), glob["A_log"].cpu(), glob["dt_bias"].cpu()
        )
        return {"dA_log": a, "ddt_bias": b}

    def raw_arm(glob, loc):
        gk = dict(
            use_gate_in_kernel=True,
            A_log=glob["A_log"],
            dt_bias=glob["dt_bias"],
            lower_bound=LB,
        )
        _, saved = cake_kcp.chunk_kda_det_fwd_kcp(
            loc["q"],
            loc["k"],
            loc["v"],
            loc["g"],
            loc["beta"],
            ctx,
            workspace=ws_r,
            **gk,
        )
        grads = cake_kcp.chunk_kda_det_bwd_kcp(
            loc["q"], loc["k"], loc["v"], loc["g"], loc["beta"], ctx, loc["dout"], saved=saved, workspace=ws_r, **gk
        )  # fmt: skip
        red = cake_kcp.reduce_gate_param_grads_det_kcp(ctx, grads[5], grads[6])
        local = [t.detach().clone() for t in (grads[5], grads[6], red[0], red[1])]
        parts = gather((grads[5].cpu(), grads[6].cpu()))
        if rank != 0:
            return {"local": local}
        res = {"local": local, "dA_log": red[0].cpu(), "ddt_bias": red[1].cpu()}
        # control: the largest per-rank partial left out of the rank-order sum
        for k, n in enumerate(PARAM_GRADS):
            drop = max(range(W), key=lambda r: float(parts[r][k].double().norm()))
            kept = [parts[r][k] for r in range(W) if r != drop]
            t = kept[0].clone() if kept else torch.zeros_like(parts[0][k])
            for x in kept[1:]:
                t += x
            res[f"{n}_drop"] = t.reshape(res[n].shape)
        return res

    def rel(c, r):
        return error_metrics(c.to(torch.float64), r.to(torch.float64))["rel_rms"]

    aa = {n: {"A": [], "B": []} for n in PARAM_GRADS}
    for seed in AA_SEEDS:
        glob, loc = inputs(seed)
        ref = reference(glob) if rank == 0 else None
        ea = explicit_arm(glob, loc, torch_gate)
        eb = explicit_arm(glob, loc, perturbed_torch_gate(seed))
        if rank == 0:
            for n in PARAM_GRADS:
                aa[n]["A"].append(rel(ea[n], ref[n]))
                aa[n]["B"].append(rel(eb[n], ref[n]))
    test = {n: {"R": [], "R_drop": [], **{a: [] for a in arms}} for n in PARAM_GRADS}
    repeat_equal = True
    for seed in TEST_SEEDS:
        glob, loc = inputs(seed)
        ref = reference(glob) if rank == 0 else None
        r = raw_arm(glob, loc)
        # determinism: partials and reduced sums bitwise repeatable
        r2 = raw_arm(glob, loc)
        repeat_equal &= all(
            _bits_equal(x, y) for x, y in zip(r["local"], r2["local"], strict=True)
        )
        e = {a: explicit_arm(glob, loc, fn) for a, fn in arms.items()}
        if rank == 0:
            for n in PARAM_GRADS:
                test[n]["R"].append(rel(r[n], ref[n]))
                test[n]["R_drop"].append(rel(r[f"{n}_drop"], ref[n]))
                for a in arms:
                    test[n][a].append(rel(e[a][n], ref[n]))
    repeat_by_rank = gather(bool(repeat_equal))
    rec: dict = {"row": _row_id(row), "world": W, "pass": all(repeat_by_rank)}
    rec["repeat_bitwise_by_rank"] = repeat_by_rank
    if rank == 0:
        tq_aa = t_quantile(0.975, len(AA_SEEDS) - 1)
        tq = t_quantile(0.975, len(TEST_SEEDS) - 1)
        rec.update(
            protocol={
                "criterion": "7816 (d): raw no less accurate than explicit; design v1.1 section 5.4 on the KCP route",
                "alpha": ALPHA, "m": m, "rows": [_row_id(x) for x in PROTOCOL_ROWS],
                "tensors": list(PARAM_GRADS), "explicit_arms": list(arms),
                "explicit_route": "explicit KCP route, FP64 chain rule from its FP32 activated-gate gradient",
                "fla": fla_info,
                "aa_seeds": list(AA_SEEDS), "test_seeds": list(TEST_SEEDS), "b_boot": B_BOOT, "boot_seed": BOOT_SEED,
                "tq_aa": tq_aa, "tq_test": tq, "control_stretch": CONTROL_STRETCH,
            },
            device=torch.cuda.get_device_name(device), arch=cake_jit.device_arch(device.index),
            host=os.uname().nodename, job=os.environ.get("SLURM_JOB_ID"), aa_err=aa, test_err=test,
            tensors={},
        )  # fmt: skip
        for n in PARAM_GRADS:
            ell_aa = [
                _log_ratio(b, a) for a, b in zip(aa[n]["A"], aa[n]["B"], strict=True)
            ]
            c = bootstrap_critical_value(ell_aa, tq_aa, m)
            rec["tensors"][n] = {
                "ell_aa": ell_aa,
                "U_aa": u_stat(ell_aa, tq_aa),
                "c": c,
                "arms": {},
            }
            for a in arms:
                ell = [
                    _log_ratio(r, x)
                    for r, x in zip(test[n]["R"], test[n][a], strict=True)
                ]
                ell_st = [x + math.log(CONTROL_STRETCH) for x in ell]
                ell_dr = [
                    _log_ratio(r, x)
                    for r, x in zip(test[n]["R_drop"], test[n][a], strict=True)
                ]
                u, u_st, u_dr = u_stat(ell, tq), u_stat(ell_st, tq), u_stat(ell_dr, tq)
                finite = all(math.isfinite(x) for x in test[n]["R"] + test[n][a])
                pooled = math.sqrt(
                    sum(x * x for x in test[n]["R"]) / sum(x * x for x in test[n][a])
                )
                t_rec = {
                    "ell": ell, "U": u, "pass": finite and u <= c,
                    "control_stretched_U": u_st, "control_stretched_fails": u_st > c,
                    "control_drop_U": u_dr, "control_drop_fails": u_dr > c,
                    "rms_ratio_raw_over_explicit": pooled,
                }  # fmt: skip
                rec["tensors"][n]["arms"][a] = t_rec
                rec["pass"] &= (
                    t_rec["pass"]
                    and t_rec["control_stretched_fails"]
                    and t_rec["control_drop_fails"]
                )
    if W > 1:
        verdict: list = [None] * W
        dist.all_gather_object(verdict, rec["pass"])
        rec["pass"] = bool(verdict[0])
        dist.barrier()
    if rank == 0 and result_dir:
        os.makedirs(result_dir, exist_ok=True)
        with open(
            os.path.join(result_dir, "protocol-" + _row_id(row) + ".json"), "w"
        ) as fh:
            json.dump(rec, fh, indent=1, default=str)
    return rec


def _log_ratio(num: float, den: float) -> float:
    return math.log(max(num, 1e-300) / max(den, 1e-300))


def _protocol_spawn_entry(rank: int, W: int, row, port: int, result_dir: str) -> None:
    try:
        rec = _protocol_rank(rank, W, row, port, result_dir)
        if not rec["pass"]:
            raise AssertionError(
                f"rank {rank}: {_row_id(row)} protocol failed (see the record)"
            )
    finally:
        if dist.is_initialized():
            cake_kcp.graph_cache_clear()
            dist.destroy_process_group()


@pytest.mark.skipif(
    os.environ.get(ENV_KCP_PROTOCOL, "") not in ("1", "true", "yes"),
    reason=f"criterion (d) protocol on the KCP route: set {ENV_KCP_PROTOCOL}=1 (minutes per row)",
)
@pytest.mark.parametrize("row", PROTOCOL_ROWS, ids=_row_id)
def test_raw_gate_kcp_param_grads_no_less_accurate_than_explicit(row) -> None:
    """Criterion (d) of #7816 for ``dA_log`` / ``ddt_bias`` on the KCP route (see the section comment): ``U <= c`` per
    tensor and explicit arm, both negative controls fail (``U > c``), and the raw route's partials and reduced sums
    are bitwise repeatable on every rank."""
    lengths, H, W, layout = row
    result_dir = os.environ.get("KCP_TEST_RESULT_DIR")
    if (
        os.environ.get("SLURM_PROCID") is not None
        and int(os.environ.get("SLURM_NTASKS", "1")) == W
        and W > 1
    ):
        rec = _protocol_rank(
            int(os.environ["SLURM_PROCID"]), W, row, int(os.environ.get("MASTER_PORT", "29500")), result_dir
        )  # fmt: skip
        assert rec["pass"], {k: rec.get(k) for k in ("row", "tensors")}
        return
    _skip_unless_runnable(W)
    with tempfile.TemporaryDirectory(prefix="cake_kcp_raw_protocol_") as tmp:
        out_dir = result_dir or tmp
        mp.start_processes(
            _protocol_spawn_entry,
            args=(W, row, _free_port(), out_dir),
            nprocs=W,
            join=True,
            start_method="spawn",
        )
        with open(os.path.join(out_dir, "protocol-" + _row_id(row) + ".json")) as fh:
            rec = json.load(fh)
    assert rec["pass"], {k: rec.get(k) for k in ("row", "tensors")}
