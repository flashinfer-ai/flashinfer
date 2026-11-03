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

Single-GPU tests of ``cake_causal_conv1d_det_multi_kcp`` (FlashInfer issue #7815): the convolutions of several tensors of
one layer (q / k / v) with one halo exchange per direction.  One process, world size 1 (the rank holds every fragment
of a partitioned problem, so history rows, head rows and the tail fold all run; one rank issues no collective).  The
criterion is bitwise equality with one ``cake_causal_conv1d_det_kcp`` call per tensor: ``y``, ``hist``, ``dx`` and
``dweight`` -- through the explicit forward / backward functions, the autograd entry, the facade, a checkpoint with
and without a recompute handle, and graph replay.  The NCCL rows (one collective per direction) are in
``test_cake_causal_conv1d_det_train_kcp_multi_dist.py``.
"""

from __future__ import annotations

import contextlib

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train import (
    is_cake_causal_conv1d_det_train_supported,
)
from tests.test_cake_causal_conv1d_det_train_kcp import ROWS, WIDTH, _context

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_causal_conv1d_det_train_supported(torch.device("cuda")),
    reason="cake_causal_conv1d_det_train needs a supported CUDA device (SM100 / SM103 / SM107)",
)

# The single-call rows, each with three tensors of the row's C, plus rows whose tensors have different channel counts
# (q / k / v of other head counts) and a one-tensor row.
MROWS = [(name, lens, cuts, (C, C, C)) for name, lens, cuts, C in ROWS] + [
    ("c7-mixed", (5000, 1003), (1, 2, 4997, 1003), (512, 256, 1024)),
    ("short-mixed", (1, 2, 3, 57), (3, 30, 30), (384, 128, 8)),
    ("one-tensor", (5000, 1003), (1, 2, 4997, 1003), (512,)),
]


def _bits(t):
    return t.detach().view(torch.int16 if t.element_size() == 2 else torch.int32)


def _same(a, b) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def _kcp():
    from flashinfer.cake_causal_conv1d_det_train_backend import kcp as conv_kcp

    return conv_kcp


def _problem(lens, cuts, Cs, device, seed=7815):
    ctx, S, _, _ = _context(lens, cuts, device)
    g = torch.Generator(device=device).manual_seed(seed)
    xs = [torch.randn(S, C, device=device, generator=g).to(torch.bfloat16) for C in Cs]
    ws = [
        (torch.randn(C, WIDTH, device=device, generator=g) * 0.5).contiguous()
        for C in Cs
    ]
    dys = [torch.randn(S, C, device=device, generator=g).to(torch.bfloat16) for C in Cs]
    return ctx, xs, ws, dys


def _singles_explicit(conv_kcp, ctx, xs, ws, dys):
    out = []
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = conv_kcp.cake_causal_conv1d_det_forward_kcp(x, w, ctx)
        dx, dw = conv_kcp.cake_causal_conv1d_det_backward_kcp(x, w, ctx, dy, hist)
        out.append({"y": y, "hist": hist, "dx": dx, "dw": dw})
    return out


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
def test_multi_forward_backward_bitwise_single_calls(name, lens, cuts, Cs) -> None:
    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    ref = _singles_explicit(conv_kcp, ctx, xs, ws, dys)
    ys, hists = conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(xs, ws, ctx)
    dxs, dws = conv_kcp.cake_causal_conv1d_det_multi_backward_kcp(
        xs, ws, ctx, dys, hists
    )
    torch.cuda.synchronize()
    assert len(ys) == len(hists) == len(dxs) == len(dws) == len(Cs)
    for i, r in enumerate(ref):
        assert _same(ys[i], r["y"]), (name, i, "y")
        assert _same(hists[i], r["hist"]) and hists[i].is_contiguous(), (name, i)
        assert _same(dxs[i], r["dx"]), (name, i, "dx")
        assert _same(dws[i], r["dw"]), (name, i, "dweight")
    # caller-owned outputs are written in place, same bits
    outs = [torch.empty_like(x) for x in xs]
    ys2, _ = conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(xs, ws, ctx, outs=outs)
    dx_o = [torch.empty_like(x) for x in xs]
    dw_o = [torch.empty_like(w) for w in ws]
    dxs2, dws2 = conv_kcp.cake_causal_conv1d_det_multi_backward_kcp(
        xs, ws, ctx, dys, hists, dxs=dx_o, dweights=dw_o
    )
    for i, r in enumerate(ref):
        assert ys2[i] is outs[i] and dxs2[i] is dx_o[i] and dws2[i] is dw_o[i]
        assert _same(outs[i], r["y"]) and _same(dx_o[i], r["dx"])
        assert _same(dw_o[i], r["dw"])


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
@pytest.mark.parametrize("batched", [False, True], ids=["2d", "3d"])
def test_multi_autograd_and_facade_bitwise_single_calls(
    name, lens, cuts, Cs, batched
) -> None:
    """The autograd entry and the facade against one autograd single call per tensor (and ``[1, T_r, C]`` inputs)."""
    from flashinfer import cake_causal_conv1d_det_train_kcp as facade

    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    if batched:
        xs = [x.unsqueeze(0) for x in xs]
        dys = [d.unsqueeze(0) for d in dys]

    def run(multi: bool):
        xl = [x.clone().requires_grad_(True) for x in xs]
        wl = [w.clone().requires_grad_(True) for w in ws]
        if multi:
            ys = facade.cake_causal_conv1d_det_multi_kcp(xl, wl, ctx)
        else:
            ys = [
                facade.cake_causal_conv1d_det_kcp(x, w, ctx)
                for x, w in zip(xl, wl, strict=True)
            ]
        torch.autograd.backward(list(ys), list(dys))
        return [
            {"y": y.detach(), "dx": x.grad, "dw": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]

    single, multi, again = run(False), run(True), run(True)
    for i in range(len(Cs)):
        for t in ("y", "dx", "dw"):
            assert _same(multi[i][t], single[i][t]), (name, i, t)
            assert _same(again[i][t], multi[i][t]), (name, i, t, "repeat")
        assert multi[i]["y"].shape == xs[i].shape


def _ckpt(fn, args, *, reentrant, counter_cm=None):
    from torch.utils.checkpoint import checkpoint

    if reentrant:
        return checkpoint(fn, *args, use_reentrant=True)
    cm = counter_cm or contextlib.nullcontext
    return checkpoint(
        fn,
        *args,
        use_reentrant=False,
        context_fn=lambda: (contextlib.nullcontext(), cm()),
    )


HROWS = MROWS[:3] + MROWS[-3:-1]


@pytest.mark.parametrize("name,lens,cuts,Cs", HROWS, ids=[r[0] for r in HROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_multi_recompute_handle_bitwise_single_handles(
    name, lens, cuts, Cs, reentrant
) -> None:
    """A checkpoint of the multi call with ONE handle equals a checkpoint of the single calls with one handle each,
    the plain checkpoints and the direct pass, bitwise; the recompute replays the entry ("replay"); the handle holds
    the n pools (the saved set's own tensors, the forward's ``hist`` bits) and retains their bytes."""
    from flashinfer import cake_causal_conv1d_det_train_kcp as facade

    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    n = len(Cs)

    def run(multi: bool, handles, *, ckpt=True):
        xl = [x.clone().requires_grad_(True) for x in xs]
        wl = [w.clone().requires_grad_(True) for w in ws]

        def fn(*args):
            a_x, a_w, a_h = args[:n], args[n : 2 * n], args[2 * n :]
            if multi:
                return facade.cake_causal_conv1d_det_multi_kcp(
                    a_x, a_w, ctx, recompute_handle=a_h[0]
                )
            return tuple(
                facade.cake_causal_conv1d_det_kcp(x, w, ctx, recompute_handle=h)
                for x, w, h in zip(a_x, a_w, a_h, strict=True)
            )

        args = (*xl, *wl, *handles)
        ys = _ckpt(fn, args, reentrant=reentrant) if ckpt else fn(*args)
        torch.autograd.backward(list(ys), list(dys))
        return [
            {"y": y.detach(), "dx": x.grad, "dw": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]

    direct = run(False, [None] * n, ckpt=False)
    singles_h = [conv_kcp.RecomputeHandleDetKcp() for _ in range(n)]
    single_handles = run(False, singles_h)
    plain_multi = run(True, [None])
    h = conv_kcp.RecomputeHandleDetKcp()
    conv_kcp.LAST_CALL.pop("recompute_handle", None)
    multi_handle = run(True, [h])
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay", "n": n}
    for i in range(n):
        for t in ("y", "dx", "dw"):
            for label, got in (
                ("single_handles", single_handles),
                ("plain_multi", plain_multi),
                ("multi_handle", multi_handle),
            ):
                assert _same(got[i][t], direct[i][t]), (name, label, i, t)
    entry = h.entries[conv_kcp.HANDLE_KEY]
    assert len(entry["hists"]) == n
    for i in range(n):
        assert _same(entry["hists"][i], singles_h[i].entries["conv"]["hist"])
    assert h.nbytes() == sum(s.nbytes() for s in singles_h)
    assert h.nbytes() == sum(int(ctx.halo.n_hist_rows) * C * 2 for C in Cs)


def test_multi_recompute_handle_retain_graph_and_nested() -> None:
    """``retain_graph`` + a second backward replays the entry again and accumulates exactly twice; a handle created
    outside nested regions serves both recomputes; the multi handle may also carry another package's entry."""
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_det_train_kcp as facade

    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 256), device
    )

    def fn(xq, xk, xv, wq, wk, wv, h):
        return facade.cake_causal_conv1d_det_multi_kcp(
            (xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h
        )

    def leaves():
        return [x.clone().requires_grad_(True) for x in xs] + [
            w.clone().requires_grad_(True) for w in ws
        ]

    a = leaves()
    ys = fn(*a, None)
    torch.autograd.backward(list(ys), list(dys))
    direct = [t.grad for t in a]
    a = leaves()
    h = conv_kcp.RecomputeHandleDetKcp()
    h.entries["kda"] = {"marker": torch.zeros(1, device=device)}
    ys = checkpoint(fn, *a, h, use_reentrant=False)
    torch.autograd.backward(list(ys), list(dys), retain_graph=True)
    torch.autograd.backward(list(ys), list(dys))
    for g, d in zip((t.grad for t in a), direct, strict=True):
        assert _same(g, d + d)
    assert "kda" in h.entries and len(h.entries["conv"]["hists"]) == 3

    def outer(*args):
        yi = checkpoint(fn, *args, use_reentrant=False)
        return tuple(y * torch.ones_like(y) for y in yi)

    a = leaves()
    h = conv_kcp.RecomputeHandleDetKcp()
    ys = checkpoint(outer, *a, h, use_reentrant=False)
    conv_kcp.LAST_CALL.pop("recompute_handle", None)
    torch.autograd.backward(list(ys), list(dys))
    assert conv_kcp.LAST_CALL["recompute_handle"]["fwd"] == "replay"
    for g, d in zip((t.grad for t in a), direct, strict=True):
        assert _same(g, d)


def test_multi_recompute_handle_misuse_raises() -> None:
    from flashinfer import cake_causal_conv1d_det_train_kcp as facade

    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, _ = _problem((5000, 1003), (1, 2, 4997, 1003), (512, 512, 512), device)
    # a handle used outside the backward a second time (reuse across micro-steps)
    h = conv_kcp.RecomputeHandleDetKcp()
    facade.cake_causal_conv1d_det_multi_kcp(xs, ws, ctx, recompute_handle=h)
    with pytest.raises(RuntimeError, match="recompute_handle"):
        facade.cake_causal_conv1d_det_multi_kcp(xs, ws, ctx, recompute_handle=h)
    # graph replay and a non-handle are refused
    with pytest.raises(ValueError, match="recompute_handle"):
        conv_kcp.cake_causal_conv1d_det_multi_kcp(
            xs,
            ws,
            ctx,
            graph="replay",
            recompute_handle=conv_kcp.RecomputeHandleDetKcp(),
        )
    with pytest.raises(TypeError, match="entries"):
        conv_kcp.cake_causal_conv1d_det_multi_kcp(
            xs, ws, ctx, recompute_handle=object()
        )
    # entries of a single call / another tensor count are not a multi entry, and the reverse
    single = conv_kcp.RecomputeHandleDetKcp()
    facade.cake_causal_conv1d_det_kcp(xs[0], ws[0], ctx, recompute_handle=single)
    with pytest.raises(ValueError, match="multi-call"):
        conv_kcp._handle_hists(single.entries["conv"], xs, ctx)
    with pytest.raises(ValueError, match="multi-call"):
        conv_kcp._handle_hists(h.entries["conv"], xs[:2], ctx)
    with pytest.raises(ValueError, match="hist"):
        conv_kcp._handle_hist(h.entries["conv"], xs[0], ctx)
    with pytest.raises(ValueError, match="KcpContext"):
        other, _, _, _ = _problem((5000, 1003), (1, 2, 4997, 1003), (512,), device)
        conv_kcp._handle_hists(h.entries["conv"], xs, other)
    bound = conv_kcp._handle_hists(h.entries["conv"], xs, ctx)
    assert all(a is b for a, b in zip(bound, h.entries["conv"]["hists"], strict=True))
    # operand shape errors
    with pytest.raises(TypeError, match="sequences"):
        conv_kcp.cake_causal_conv1d_det_multi_kcp(xs[0], ws[0], ctx)
    with pytest.raises(ValueError, match="same number"):
        conv_kcp.cake_causal_conv1d_det_multi_kcp(xs, ws[:2], ctx)
    with pytest.raises(ValueError, match="same number"):
        conv_kcp.cake_causal_conv1d_det_multi_kcp([], [], ctx)
    with pytest.raises(ValueError):
        conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(
            [xs[0], xs[1][:-1].contiguous()], ws[:2], ctx
        )


def test_multi_empty_rank_returns_empty_outputs() -> None:
    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, _, _, _ = _context((0,), (0,), device)
    Cs = (512, 256, 512)
    xs = [torch.empty(0, C, dtype=torch.bfloat16, device=device) for C in Cs]
    ws = [torch.randn(C, WIDTH, device=device) for C in Cs]
    ys, hists = conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(xs, ws, ctx)
    dxs, dws = conv_kcp.cake_causal_conv1d_det_multi_backward_kcp(
        xs, ws, ctx, [torch.empty_like(x) for x in xs], hists
    )
    for y, dx, dw, C in zip(ys, dxs, dws, Cs, strict=True):
        assert y.shape == (0, C) and dx.shape == (0, C)
        assert torch.equal(dw, torch.zeros_like(dw))


def test_multi_graph_replay_is_bitwise() -> None:
    """``graph="replay"``: the first call of a key runs eagerly, the second is captured and replayed once, later calls
    replay; every call's ``ys`` / ``hists`` / ``dxs`` / ``dweights`` are bitwise the eager single calls'.  The
    backward's key holds the forward's pools, so its capture follows the forward's by one call."""
    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 256), device
    )
    ref = _singles_explicit(conv_kcp, ctx, xs, ws, dys)
    try:
        events = []
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(
                xs, ws, ctx, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_det_multi_backward_kcp(
                xs, ws, ctx, dys, hists, graph="replay"
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            for i, r in enumerate(ref):
                assert _same(ys[i], r["y"]) and _same(hists[i], r["hist"])
                assert _same(dxs[i], r["dx"]) and _same(dws[i], r["dw"])
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
        # the autograd entry carries the keyword to the forward and the backward
        xl = [x.clone().requires_grad_(True) for x in xs]
        ys = conv_kcp.cake_causal_conv1d_det_multi_kcp(xl, ws, ctx, graph="replay")
        assert conv_kcp.LAST_CALL["fwd_multi"]["graph"] == "replay"
        torch.autograd.backward(list(ys), list(dys))
        assert conv_kcp.LAST_CALL["bwd_multi"]["graph"] == "replay"
        for i, r in enumerate(ref):
            assert _same(ys[i].detach(), r["y"]) and _same(xl[i].grad, r["dx"])
    finally:
        conv_kcp._GRAPHS.clear()


def test_multi_one_rank_issues_no_exchange(monkeypatch) -> None:
    """One rank: the multi path calls no exchange (each tensor's send rows are its pool), unlike the single call."""
    conv_kcp = _kcp()
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 512), device
    )
    calls = []
    original = conv_kcp._exchange
    monkeypatch.setattr(
        conv_kcp, "_exchange", lambda *a, **kw: calls.append(1) or original(*a, **kw)
    )
    ys, hists = conv_kcp.cake_causal_conv1d_det_multi_forward_kcp(xs, ws, ctx)
    conv_kcp.cake_causal_conv1d_det_multi_backward_kcp(xs, ws, ctx, dys, hists)
    assert calls == []


def test_facade_multi_entry_is_lazy_and_keyword_only() -> None:
    """The KCP facade's ``cake_causal_conv1d_det_multi_kcp`` is exported, takes ``recompute_handle`` keyword-only
    (default ``None``), is the det facade's previous ``cake_causal_conv1d_kcp_multi``, and importing either facade
    still does not import the KCP host modules (fresh interpreter)."""
    import subprocess
    import sys

    code = (
        "import inspect, sys\n"
        "import flashinfer.cake_causal_conv1d_det_train_kcp as f\n"
        "import flashinfer.cake_causal_conv1d_det_train as d\n"
        "kcp = 'flashinfer.cake_causal_conv1d_det_train_backend.kcp'\n"
        "assert kcp not in sys.modules\n"
        "s = inspect.signature(f.cake_causal_conv1d_det_multi_kcp).parameters\n"
        "assert list(s) == ['xs', 'weights', 'ctx', 'recompute_handle'], list(s)\n"
        "p = s['recompute_handle']\n"
        "assert p.kind is inspect.Parameter.KEYWORD_ONLY and p.default is None\n"
        "assert d.cake_causal_conv1d_kcp_multi is f.cake_causal_conv1d_det_multi_kcp\n"
        "assert 'cake_causal_conv1d_det_multi_kcp' in f.__all__ and kcp not in sys.modules\n"
        "print('FACADE_MULTI_OK')\n"
    )
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0 and "FACADE_MULTI_OK" in r.stdout, (r.stdout, r.stderr)
