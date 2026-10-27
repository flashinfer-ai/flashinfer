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

Single-GPU tests of ``cake_causal_conv1d_kcp`` (FlashInfer issue #6919): one process, world size 1, no process
group.  The rank holds every fragment of a partitioned problem (the halo exchange of one rank is the identity), so
the history rows, the backward's head rows and the tail fold all run through the host flow.  Criteria:

(a) ``y``, ``dx`` and ``dweight`` against an FP64 oracle of the KCP semantics (the SiLU derivative taken on the FP32
    preactivation, "policy B") within the conv package's contract tolerances: BF16 ``y`` / ``dx`` within
    ``atol 1e-3 + rtol 2^-7`` (one BF16 ulp) with no element outside, FP32 ``dweight`` within ``atol 5e-2 + rtol 1e-4``.
(b) against the shipped zero-history kernels on the unpartitioned documents: ``y`` bitwise equal (the forward runs the
    same instructions), and ``dx`` elementwise within ``1e-2 + 1e-2 |dx_noCP|`` with no mismatch.  The shipped backward
    differentiates the BF16-rounded preactivation ("policy A"), so its ``dx`` is not the KCP ``dx`` bit for bit: the
    two differ by the derivative's sensitivity to that rounding plus one BF16 output rounding (a few elements per
    million sit beyond ``2e-3 + 2^-6 |dx|``, none beyond the clause above; the measured difference is reported as a
    test property).
(c) bitwise: two autograd calls on the same inputs agree on ``y`` / ``dx`` / ``dweight``, the autograd function equals
    the direct forward / backward calls, and the fold of the received ``dpre`` rows reproduces the unpartitioned
    launch's ``dx`` through the same host flow.

A ``KcpContext`` is built with ``build_kcp_context`` when the KCP plan module is importable, else from the same
tables hand-built here (the plan's halo layout for one rank).
"""

from __future__ import annotations

import types

import pytest
import torch

from flashinfer.cake_causal_conv1d_train import (
    cake_causal_conv1d_train_backward,
    cake_causal_conv1d_train_forward,
    is_cake_causal_conv1d_train_supported,
)
from flashinfer.cake_causal_conv1d_train_backend.kcp import (
    cake_causal_conv1d_kcp,
    cake_causal_conv1d_kcp_backward,
    cake_causal_conv1d_kcp_forward,
)

WIDTH = 4
HALO = WIDTH - 1
# The conv package's contract tolerances: y / dx within one BF16 ulp of the oracle (no element outside), dweight
# within atol 5e-2 + rtol 1e-4 of the FP64 one.
Y_ATOL, Y_RTOL = 1e-3, 2**-7
DW_ATOL, DW_RTOL = 5e-2, 1e-4
# The elementwise clause for a context-parallel result against the no-CP kernel of the same package.
NOCP_ATOL, NOCP_RTOL = 1e-2, 1e-2

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_causal_conv1d_train_supported(torch.device("cuda")),
    reason="cake_causal_conv1d_train needs a supported CUDA device (SM100 / SM103 / SM107)",
)

# Rows: (document lengths, fragment lengths over the packed sequence); each fragment is cut at document
# boundaries too.  The rows of record C7 (history rows from two source fragments of one and two tokens, a head row
# with two consumers) and C7b (an empty predecessor), the [1 | 1 | 1 | long] pattern (a head row with three
# consumers), 1-3 token documents inside a fragment, a 1-token continuing fragment, and a wide row.
ROWS = [
    ("c7", (5000, 1003), (1, 2, 4997, 1003), 512),
    ("c7b", (5000, 1003), (1, 2, 0, 6000), 512),
    ("three-consumers", (1000, 0, 3, 3094), (1, 1, 1, 4094), 2048),
    ("short-docs", (1, 2, 3, 57), (3, 30, 30), 384),
    ("one-token-frag", (65,), (64, 1), 256),
    ("wide", (5000, 7000), (1, 2, 4997, 7000), 8192),
]


def _fragments(lens, cuts):
    """``(doc, off, len)`` fragments in document-then-offset order, the cuts split at document boundaries; an empty
    cut is an empty fragment."""
    cu = [0]
    for n in lens:
        cu.append(cu[-1] + n)
    S = cu[-1]
    assert sum(cuts) == S, (cuts, S)
    frags = []
    pos = 0
    for n in cuts:
        end = pos + n
        if n == 0:
            d = (
                max(i for i in range(len(lens)) if cu[i] <= pos)
                if pos < S
                else len(lens) - 1
            )
            frags.append((d, pos - cu[d], 0))
        p = pos
        while p < end:
            d = max(i for i in range(len(lens)) if cu[i] <= p)
            e = min(end, cu[d + 1])
            frags.append((d, p - cu[d], e - p))
            p = e
        pos = end
    return S, tuple(cu), frags


def _halo_tables(lens, frags):
    """One rank's halo plan over ``frags`` (local order = the given order): the kernels' tables and the row pools'
    sizes, laid out as the plan does for a single rank (receive pool rows in consumer order)."""
    n_frag = len(frags)
    local_cu = [0]
    for _, _, ln in frags:
        local_cu.append(local_cu[-1] + ln)
    by_doc = {}
    for slot, (d, off, ln) in enumerate(frags):
        if ln > 0:
            by_doc.setdefault(d, []).append((off, ln, slot))
    owner = {}  # (doc, token) -> (slot, pos)
    for d, chain in by_doc.items():
        for off, ln, slot in chain:
            for t in range(off, off + ln):
                owner[(d, t)] = (slot, t - off)
    hist_row_off = [[-1] * HALO for _ in range(n_frag)]
    hist_index = []
    n = 0
    for slot, (d, off, ln) in enumerate(frags):
        if ln == 0 or off == 0:
            continue
        for k in range(HALO):
            tok = off - HALO + k
            if tok < 0:
                continue
            src_slot, pos = owner[(d, tok)]
            hist_row_off[slot][k] = n
            hist_index.append(local_cu[src_slot] + pos)
            n += 1
    n_hist = n
    tail_row_off = [[-1] * HALO for _ in range(n_frag)]
    head_row_off = [[[-1] * HALO for _ in range(HALO)] for _ in range(n_frag)]
    n = 0
    for slot, (d, off, ln) in enumerate(frags):
        if ln == 0:
            continue
        end = off + ln
        for m in range(HALO):
            tok = end + m
            if (d, tok) not in owner:
                continue
            succ_slot, pos = owner[(d, tok)]
            if pos >= HALO:
                continue  # the successor's token beyond its first three rows is never a head row
            tail_row_off[slot][m] = n
            slots = head_row_off[succ_slot][pos]
            free = [c for c in range(HALO) if slots[c] == -1]
            assert free, "a head token has at most three consuming fragments"
            slots[free[0]] = n
            n += 1
    n_tail = n
    return types.SimpleNamespace(
        n_frag=n_frag,
        local_cu=tuple(local_cu),
        hist_row_off=hist_row_off,
        tail_row_off=tail_row_off,
        head_row_off=head_row_off,
        hist_index=hist_index,
        n_hist_rows=n_hist,
        n_head_rows=n_tail,
        n_tail_rows=n_tail,
    )


def _context(lens, cuts, device):
    """A one-rank ``KcpContext`` for ``cuts`` over documents ``lens``: ``build_kcp_context`` when the plan module
    is importable, else a stand-in with the same fields the conv host module reads."""
    S, cu_global, frags = _fragments(lens, cuts)
    try:
        from flashinfer.experimental.cake_kda_train.cake_kcp_plan import (
            build_kcp_context,
            kcp_partition,
        )

        part = kcp_partition(
            torch.tensor(cu_global, dtype=torch.int64), 1, fragments=[frags]
        )
        ctx = build_kcp_context(part, None, 0, device, conv_kernel_size=WIDTH)
        return ctx, S, cu_global, frags
    except ImportError:
        pass
    t = _halo_tables(lens, frags)
    i32 = torch.int32
    halo = types.SimpleNamespace(
        halo=HALO,
        hist_row_off=tuple(tuple(r) for r in t.hist_row_off),
        tail_row_off=tuple(tuple(r) for r in t.tail_row_off),
        head_row_off=tuple(tuple(tuple(c) for c in r) for r in t.head_row_off),
        hist_index=tuple(t.hist_index),
        fwd_send_rows=(t.n_hist_rows,),
        fwd_recv_rows=(t.n_hist_rows,),
        bwd_send_rows=(t.n_head_rows,),
        bwd_recv_rows=(t.n_tail_rows,),
        n_hist_rows=t.n_hist_rows,
        n_head_rows=t.n_head_rows,
        n_tail_rows=t.n_tail_rows,
    )
    ctx = types.SimpleNamespace(
        rank=0,
        world_size=1,
        group=None,
        device=device,
        conv_kernel_size=WIDTH,
        T_r=S,
        n_frag=t.n_frag,
        cu_seqlens_local=torch.tensor(t.local_cu, dtype=i32, device=device),
        halo=halo,
        halo_tables={
            "hist_row_off": torch.tensor(
                t.hist_row_off, dtype=i32, device=device
            ).reshape(t.n_frag, HALO),
            "tail_row_off": torch.tensor(
                t.tail_row_off, dtype=i32, device=device
            ).reshape(t.n_frag, HALO),
            "head_row_off": torch.tensor(
                t.head_row_off, dtype=i32, device=device
            ).reshape(t.n_frag, HALO, HALO),
            "hist_index": torch.tensor(t.hist_index, dtype=torch.int64, device=device),
        },
    )
    return ctx, S, cu_global, frags


def _problem(lens, cuts, C, device, seed=0):
    """The unpartitioned problem (``x``, ``w``, ``dy``, ``cu``) and the rank's local view of it: every fragment as a
    local sequence in the plan's order (``l2g`` maps local tokens to global ones)."""
    ctx, S, cu_global, frags = _context(lens, cuts, device)
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    w = (torch.randn(C, WIDTH, device=device, generator=g) * 0.5).contiguous()
    dy = torch.randn(S, C, device=device, generator=g, dtype=torch.bfloat16)
    l2g = []
    for d, off, ln in frags:
        l2g.extend(range(cu_global[d] + off, cu_global[d] + off + ln))
    l2g = torch.tensor(l2g, dtype=torch.int64, device=device)
    cu = torch.tensor(cu_global, dtype=torch.int32, device=device)
    return types.SimpleNamespace(
        ctx=ctx,
        S=S,
        C=C,
        x=x,
        w=w,
        dy=dy,
        cu=cu,
        l2g=l2g,
        x_local=x[l2g].contiguous(),
        dy_local=dy[l2g].contiguous(),
    )


def _reference(x, w, cu, dy):
    """FP64 oracle of the unpartitioned problem with the KCP semantics (policy B: the SiLU derivative on the
    unrounded preactivation): BF16 ``y`` and ``dx`` (one rounding each), FP64 ``dweight``."""
    S, C = x.shape
    xf = x.double()
    wd = w.double()
    cu_l = cu.tolist()
    doc = torch.zeros(S, dtype=torch.int64, device=x.device)
    for i in range(len(cu_l) - 1):
        doc[cu_l[i] : cu_l[i + 1]] = i

    def shifted(t, k):  # t[i - k] of the same document, else 0
        out = torch.zeros_like(t)
        if k < S:
            same = (
                doc[k:] == doc[:-k]
                if k
                else torch.ones(S, dtype=torch.bool, device=x.device)
            )
            out[k:] = torch.where(
                same[:, None], t[: S - k], torch.zeros_like(t[: S - k])
            )
        return out

    pre = sum(wd[:, k][None, :] * shifted(xf, HALO - k) for k in range(WIDTH))
    s = torch.sigmoid(pre)
    y = (pre * s).to(torch.bfloat16)
    dpre = dy.double() * s * (1 + pre * (1 - s))

    def shifted_up(t, k):  # t[i + k] of the same document, else 0
        out = torch.zeros_like(t)
        if k < S:
            same = (
                doc[k:] == doc[:-k]
                if k
                else torch.ones(S, dtype=torch.bool, device=x.device)
            )
            out[: S - k] = torch.where(same[:, None], t[k:], torch.zeros_like(t[k:]))
        return out

    dx = sum(wd[:, k][None, :] * shifted_up(dpre, HALO - k) for k in range(WIDTH)).to(
        torch.bfloat16
    )
    dw = torch.stack(
        [(dpre * shifted(xf, HALO - k)).sum(0) for k in range(WIDTH)],
        dim=1,
    )
    return y, dx, dw


def _assert_within(actual, expected, *, atol, rtol, label):
    """No element of ``actual`` outside ``atol + rtol |expected|`` (an exact mismatch count of zero); returns the
    comparison's statistics for the test record."""
    assert tuple(actual.shape) == tuple(expected.shape), label
    a = actual.detach().reshape(-1).double()
    b = expected.detach().reshape(-1).double()
    assert torch.isfinite(a).all(), f"{label} has non-finite values"
    diff = (a - b).abs()
    bound = atol + rtol * b.abs()
    outside = diff > bound
    count = int(outside.sum().item())
    stats = {
        "count_outside": count,
        "max_abs_diff": float(diff.max().item()),
        "max_ratio_to_bound": float((diff / bound).max().item()),
    }
    if count:
        index = int(outside.nonzero()[0].item())
        raise AssertionError(
            f"{label}: {count} of {a.numel()} elements outside atol={atol} rtol={rtol}; first at flat index "
            f"{index}: got {a[index].item()!r}, expected {b[index].item()!r}"
        )
    return stats


def _bits(t):
    return t.detach().view(torch.int16 if t.dtype == torch.bfloat16 else torch.int32)


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
def test_forward_matches_unpartitioned_bitwise(name, lens, cuts, C):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    y_local, hist = cake_causal_conv1d_kcp_forward(p.x_local, p.w, p.ctx)
    y_global = cake_causal_conv1d_train_forward(p.x, p.w, p.cu)
    torch.cuda.synchronize()
    assert hist.shape[0] == max(1, p.ctx.halo.n_hist_rows)
    assert not y_local.isnan().any()
    # (b) the forward runs the zero-history forward's instructions with history rows: bitwise equal.
    assert torch.equal(_bits(y_local), _bits(y_global[p.l2g])), (
        "y differs from the zero-history kernel"
    )
    # (a) y against the FP64 oracle.
    y_ref, _, _ = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(
        y_local, y_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="y vs FP64 oracle"
    )


@pytest.mark.parametrize("name,lens,cuts,C", ROWS, ids=[r[0] for r in ROWS])
def test_backward_matches_reference_and_folds_tails_bitwise(
    name, lens, cuts, C, record_property
):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = cake_causal_conv1d_kcp_forward(p.x_local, p.w, p.ctx)
    dx_local, dw_local = cake_causal_conv1d_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist
    )
    torch.cuda.synchronize()
    assert not dx_local.isnan().any() and not dw_local.isnan().any()
    # (a) dx and dweight against the FP64 oracle of the KCP semantics.
    _, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(
        dx_local, dx_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="dx vs FP64 oracle"
    )
    _assert_within(
        dw_local, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label="dw vs FP64 oracle"
    )
    # (b) dx against the shipped zero-history backward (policy A) on the unpartitioned documents: the elementwise
    # clause for a context-parallel result against the no-CP kernel; the measured difference is recorded.
    dx_nocp, _ = cake_causal_conv1d_train_backward(p.x, p.w, p.cu, p.dy)
    stats = _assert_within(
        dx_local,
        dx_nocp[p.l2g],
        atol=NOCP_ATOL,
        rtol=NOCP_RTOL,
        label="dx vs no-CP kernel",
    )
    for key, value in stats.items():
        record_property(f"dx_vs_nocp_{key}", value)
    # (c) The unpartitioned problem through the same host flow (one fragment per document, no halo rows): the fold
    # of the received dpre rows reproduces its dx bit for bit.
    whole = _problem(lens, tuple(lens), C, device)
    _, hist_w = cake_causal_conv1d_kcp_forward(whole.x_local, whole.w, whole.ctx)
    dx_whole, _ = cake_causal_conv1d_kcp_backward(
        whole.x_local, whole.w, whole.ctx, whole.dy_local, hist_w
    )
    torch.cuda.synchronize()
    assert whole.ctx.halo.n_hist_rows == 0 and whole.ctx.halo.n_tail_rows == 0
    assert torch.equal(_bits(dx_local), _bits(dx_whole[whole.l2g.argsort()][p.l2g])), (
        "dx differs from the unpartitioned launch"
    )


@pytest.mark.parametrize("name,lens,cuts,C", ROWS[:3], ids=[r[0] for r in ROWS[:3]])
def test_autograd_function_and_repeat_are_bitwise(name, lens, cuts, C, record_property):
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    runs = []
    for _ in range(2):
        x = p.x_local.clone().requires_grad_(True)
        w = p.w.clone().requires_grad_(True)
        y = cake_causal_conv1d_kcp(x, w, p.ctx)
        y.backward(p.dy_local)
        runs.append((y.detach(), x.grad, w.grad))
    y_direct, hist = cake_causal_conv1d_kcp_forward(p.x_local, p.w, p.ctx)
    dx_direct, dw_direct = cake_causal_conv1d_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist
    )
    torch.cuda.synchronize()
    (y1, dx1, dw1), (y2, dx2, dw2) = runs
    # (c) two launches on the same inputs, and the autograd function against the direct calls: bitwise.
    assert torch.equal(_bits(y1), _bits(y2))
    assert torch.equal(_bits(dx1), _bits(dx2))
    assert torch.equal(_bits(dw1), _bits(dw2)), "dweight differs between two launches"
    assert torch.equal(_bits(y1), _bits(y_direct))
    assert torch.equal(_bits(dx1), _bits(dx_direct))
    assert torch.equal(_bits(dw1), _bits(dw_direct)), (
        "autograd dweight differs from the direct backward"
    )
    # (a) the autograd outputs against the FP64 oracle of the KCP semantics.
    y_ref, dx_ref, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _assert_within(y1, y_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="y vs FP64 oracle")
    _assert_within(
        dx1, dx_ref[p.l2g], atol=Y_ATOL, rtol=Y_RTOL, label="dx vs FP64 oracle"
    )
    _assert_within(dw1, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label="dw vs FP64 oracle")
    # (b) dx against the shipped zero-history backward (policy A) on the unpartitioned documents under the no-CP
    # clause; the two policies' difference is recorded for the report (it is not a bitwise relation).
    dx_nocp, _ = cake_causal_conv1d_train_backward(p.x, p.w, p.cu, p.dy)
    stats = _assert_within(
        dx1, dx_nocp[p.l2g], atol=NOCP_ATOL, rtol=NOCP_RTOL, label="dx vs no-CP kernel"
    )
    for key, value in stats.items():
        record_property(f"dx_vs_nocp_{key}", value)


def test_empty_rank_returns_empty_outputs():
    device = torch.device("cuda")
    C = 512
    ctx, S, _, _ = _context((0,), (0,), device)
    x = torch.empty(0, C, dtype=torch.bfloat16, device=device)
    w = torch.randn(C, WIDTH, device=device)
    y, hist = cake_causal_conv1d_kcp_forward(x, w, ctx)
    dx, dw = cake_causal_conv1d_kcp_backward(x, w, ctx, torch.empty_like(x), hist)
    assert y.shape == (0, C) and dx.shape == (0, C)
    assert torch.equal(dw, torch.zeros_like(dw))


def test_rejects_mismatched_context():
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError):
        cake_causal_conv1d_kcp_forward(p.x_local[:-1].contiguous(), p.w, p.ctx)


def test_kcp_keeps_fp32_taps_and_no_bias():
    """The KCP path keeps the package's pre-round-6 contract: BF16 taps are refused with the FP32 error text
    (the training routes accept them), and there is no bias keyword."""

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        cake_causal_conv1d_kcp_forward(p.x_local, p.w.to(torch.bfloat16), p.ctx)
    with pytest.raises(ValueError, match="weight must be torch.float32"):
        cake_causal_conv1d_kcp_backward(
            p.x_local,
            p.w.to(torch.bfloat16),
            p.ctx,
            p.dy_local,
            torch.empty(0, p.C, dtype=torch.bfloat16, device=device),
        )
    bias = torch.zeros(p.C, device=device)
    with pytest.raises(TypeError):
        cake_causal_conv1d_kcp_forward(p.x_local, p.w, p.ctx, bias=bias)
    with pytest.raises(TypeError):
        cake_causal_conv1d_kcp(p.x_local, p.w, p.ctx, bias)


def test_kcp_plans_with_the_legacy_tile_rule(monkeypatch):
    """KCP plans its launches with ``cake_jit.plan(..., legacy=True)``: C = 1280 keeps ``(ct_vec, n_sub,
    n_ch_tiles) = (128, 1, 2)`` (the training routes tile it as (32, 4, 5)), and the forward's ``y`` stays
    bitwise the unpartitioned forward's."""

    from flashinfer.cake_causal_conv1d_train_backend import cake_jit
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp

    geometry = {"threads": 128, "vec": 8}
    legacy = cake_jit.plan(5000, 1280, 2, geometry=geometry, legacy=True)
    assert (legacy["ct_vec"], legacy["n_sub"], legacy["n_ch_tiles"]) == (128, 1, 2)
    routed = cake_jit.plan(5000, 1280, 2, geometry=geometry, role="bwd")
    assert (routed["ct_vec"], routed["n_sub"], routed["n_ch_tiles"]) == (32, 4, 5)
    calls = []
    original = conv_kcp.plan

    def recording_plan(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(conv_kcp, "plan", recording_plan)
    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 1280, device)
    y, hist = cake_causal_conv1d_kcp_forward(p.x_local, p.w, p.ctx)
    dx, dw = cake_causal_conv1d_kcp_backward(p.x_local, p.w, p.ctx, p.dy_local, hist)
    assert len(calls) == 2 and all(call.get("legacy") is True for call in calls), calls
    y_full = cake_causal_conv1d_train_forward(p.x, p.w, p.cu)
    assert torch.equal(y, y_full[p.l2g])
    assert torch.isfinite(dw).all() and dx.shape == p.x_local.shape


@pytest.mark.parametrize("form", ["knob", "keyword"])
def test_graph_replay_knob_is_bitwise(monkeypatch, form) -> None:
    """The graph-replay option: under ``_set_test_knobs({"graph": "replay"})`` (form ``knob``) or the public keyword
    ``graph="replay"`` (form ``keyword``) the first call of a key runs eagerly,
    the second is captured and replayed once, later calls replay the graph; ``y`` / ``hist`` / ``dx`` / ``dweight``
    are bitwise the knob-off results on every call.  The backward's key holds the forward's ``hist`` pointer, so
    its capture follows the forward's by one call.  Another input pointer is another key (eager, still bitwise)."""
    from flashinfer.cake_causal_conv1d_train_backend import kcp as conv_kcp

    device = torch.device("cuda")
    p = _problem((5000, 1003), (1, 2, 4997, 1003), 512, device)
    x, w, ctx = p.x_local, p.w, p.ctx
    dy = torch.randn(x.shape, device=device).to(torch.bfloat16)
    y0, hist0 = cake_causal_conv1d_kcp_forward(x, w, ctx)
    dx0, dw0 = cake_causal_conv1d_kcp_backward(x, w, ctx, dy, hist0)
    kw = {}
    if form == "knob":
        monkeypatch.setenv(conv_kcp.TEST_KNOBS_ENV, "1")
        conv_kcp._set_test_knobs({"graph": "replay"})
    else:
        kw = {"graph": "replay"}  # the public keyword, no environment gate
    try:
        events = []
        for _ in range(4):
            y, hist = cake_causal_conv1d_kcp_forward(x, w, ctx, **kw)
            events.append(conv_kcp.LAST_CALL["fwd"]["graph_event"])
            assert torch.equal(y, y0) and torch.equal(hist, hist0)
            dx, dw = cake_causal_conv1d_kcp_backward(x, w, ctx, dy, hist, **kw)
            events.append(conv_kcp.LAST_CALL["bwd"]["graph_event"])
            assert torch.equal(dx, dx0) and torch.equal(dw, dw0)
        assert events == [
            "eager",
            "eager",
            "capture",
            "eager",
            "replay",
            "capture",
            "replay",
            "replay",
        ], events
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
        y, hist = cake_causal_conv1d_kcp_forward(x.clone(), w, ctx, **kw)
        assert conv_kcp.LAST_CALL["fwd"]["graph_event"] == "eager"
        assert torch.equal(y, y0) and torch.equal(hist, hist0)
        if (
            form == "keyword"
        ):  # the autograd entry carries the keyword to the forward and to the backward
            xg = x.clone().requires_grad_(True)
            y = conv_kcp.cake_causal_conv1d_kcp(xg, w, ctx, graph="replay")
            assert conv_kcp.LAST_CALL["fwd"]["graph"] == "replay"
            assert torch.equal(y.detach(), y0)
            y.backward(dy)
            assert conv_kcp.LAST_CALL["bwd"]["graph"] == "replay"
            assert torch.equal(xg.grad, dx0)
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp._GRAPHS.clear()
    y, hist = cake_causal_conv1d_kcp_forward(x, w, ctx)
    assert torch.equal(y, y0) and torch.equal(hist, hist0)
