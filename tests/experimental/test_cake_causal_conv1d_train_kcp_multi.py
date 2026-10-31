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

Single-GPU tests of the non-deterministic ``cake_causal_conv1d_multi_kcp`` (several tensors of one layer with one halo
exchange per direction), on the deterministic multi suite's rows: ``y``, ``hist`` and ``dx`` are bitwise one
non-deterministic single call per tensor and the deterministic multi call; each ``dweight`` is within the package
budget of the single call's and the deterministic one -- through the explicit functions, the autograd entry, the
facade, a checkpoint with and without a recompute handle (2 x forward + backward), gradient accumulation and graph
replay.  The NCCL rows are in ``test_cake_causal_conv1d_train_kcp_multi_dist.py``.  Every test skips until the
exporter has rendered the non-deterministic package.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train_backend import kcp as det_kcp
from flashinfer.experimental.cake import cake_causal_conv1d_train_backend as pkg
from flashinfer.experimental.cake.cake_causal_conv1d_train_backend import (
    kcp as conv_kcp,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    DW_ATOL,
    DW_RTOL,
    WIDTH,
    _context,
)
from tests.test_cake_causal_conv1d_det_train_kcp_multi import MROWS, _problem, _same

NONDET = {"deterministic": False}

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc (TODO-EXPORT until the exporter renders it)"
    ),
)


def _close(a, b, label, *, scale=1.0) -> None:
    assert a.dtype == b.dtype and a.shape == b.shape, label
    torch.testing.assert_close(
        a, b, atol=scale * DW_ATOL, rtol=scale * DW_RTOL, msg=label
    )


def _singles(ctx, xs, ws, dys, *, det=False):
    """One explicit single forward / backward per tensor: the non-deterministic calls, or (``det``) the
    deterministic ones."""

    out = []
    if det:
        forward, backward, kw = (
            det_kcp.cake_causal_conv1d_det_forward_kcp,
            det_kcp.cake_causal_conv1d_det_backward_kcp,
            {},
        )
    else:
        forward, backward, kw = (
            conv_kcp.cake_causal_conv1d_forward_kcp,
            conv_kcp.cake_causal_conv1d_backward_kcp,
            NONDET,
        )
    for x, w, dy in zip(xs, ws, dys, strict=True):
        y, hist = forward(x, w, ctx, **kw)
        dx, dw = backward(x, w, ctx, dy, hist, **kw)
        out.append({"y": y, "hist": hist, "dx": dx, "dw": dw})
    return out


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
def test_multi_matches_single_calls(name, lens, cuts, Cs) -> None:
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    ref = _singles(ctx, xs, ws, dys)
    det = _singles(ctx, xs, ws, dys, det=True)
    ys, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(xs, ws, ctx, **NONDET)
    dxs, dws = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
        xs, ws, ctx, dys, hists, **NONDET
    )
    torch.cuda.synchronize()
    assert len(ys) == len(hists) == len(dxs) == len(dws) == len(Cs)
    for i, (r, d) in enumerate(zip(ref, det, strict=True)):
        for t in ("y", "hist", "dx"):
            got = {"y": ys, "hist": hists, "dx": dxs}[t][i]
            assert _same(got, r[t]) and _same(got, d[t]), (name, i, t)
        _close(dws[i], r["dw"], f"{name} {i} dweight vs single")
        _close(dws[i], d["dw"], f"{name} {i} dweight vs deterministic")
    outs = [torch.empty_like(x) for x in xs]
    ys2, _ = conv_kcp.cake_causal_conv1d_multi_forward_kcp(
        xs, ws, ctx, outs=outs, **NONDET
    )
    dx_o = [torch.full_like(x, float("nan")) for x in xs]
    dw_o = [torch.full_like(w, float("nan")) for w in ws]
    dxs2, dws2 = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
        xs, ws, ctx, dys, hists, dxs=dx_o, dweights=dw_o, **NONDET
    )
    for i, r in enumerate(ref):
        assert ys2[i] is outs[i] and dxs2[i] is dx_o[i] and dws2[i] is dw_o[i]
        assert _same(outs[i], r["y"]) and _same(dx_o[i], r["dx"])
        _close(dw_o[i], r["dw"], f"{name} {i} caller-owned dweight")


@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
@pytest.mark.parametrize("batched", [False, True], ids=["2d", "3d"])
def test_multi_autograd_and_facade_match_single_calls(
    name, lens, cuts, Cs, batched
) -> None:
    from flashinfer import cake_causal_conv1d_train_kcp as facade

    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    if batched:
        xs = [x.unsqueeze(0) for x in xs]
        dys = [d.unsqueeze(0) for d in dys]

    def run(multi: bool):
        xl = [x.clone().requires_grad_(True) for x in xs]
        wl = [w.clone().requires_grad_(True) for w in ws]
        if multi:
            ys = facade.cake_causal_conv1d_multi_kcp(xl, wl, ctx, **NONDET)
        else:
            ys = [
                facade.cake_causal_conv1d_kcp(x, w, ctx, **NONDET)
                for x, w in zip(xl, wl, strict=True)
            ]
        torch.autograd.backward(list(ys), list(dys))
        return [
            {"y": y.detach(), "dx": x.grad, "dw": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]

    single, multi, again = run(False), run(True), run(True)
    for i in range(len(Cs)):
        for t in ("y", "dx"):
            assert _same(multi[i][t], single[i][t]) and _same(
                again[i][t], multi[i][t]
            ), (name, i, t)
        _close(multi[i]["dw"], single[i]["dw"], f"{name} {i} dweight")
        _close(again[i]["dw"], multi[i]["dw"], f"{name} {i} repeated dweight")
        assert multi[i]["y"].shape == xs[i].shape


def _ckpt(fn, args, *, reentrant):
    from torch.utils.checkpoint import checkpoint

    if reentrant:
        return checkpoint(fn, *args, use_reentrant=True)
    return checkpoint(fn, *args, use_reentrant=False)


HROWS = MROWS[:3] + MROWS[-3:-1]


@pytest.mark.parametrize("name,lens,cuts,Cs", HROWS, ids=[r[0] for r in HROWS])
@pytest.mark.parametrize("reentrant", [False, True], ids=["nonreentrant", "reentrant"])
def test_multi_checkpoint_and_recompute_handle(name, lens, cuts, Cs, reentrant) -> None:
    from flashinfer import cake_causal_conv1d_train_kcp as facade

    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(lens, cuts, Cs, device)
    n = len(Cs)

    def run(handle, *, ckpt=True):
        xl = [x.clone().requires_grad_(True) for x in xs]
        wl = [w.clone().requires_grad_(True) for w in ws]

        def fn(*args):
            return facade.cake_causal_conv1d_multi_kcp(
                args[:n], args[n : 2 * n], ctx, recompute_handle=args[2 * n], **NONDET
            )

        args = (*xl, *wl, handle)
        ys = _ckpt(fn, args, reentrant=reentrant) if ckpt else fn(*args)
        torch.autograd.backward(list(ys), list(dys))
        return [
            {"y": y.detach(), "dx": x.grad, "dw": w.grad}
            for y, x, w in zip(ys, xl, wl, strict=True)
        ]

    direct = run(None, ckpt=False)
    plain = run(None)
    h = conv_kcp.RecomputeHandleKcp()
    conv_kcp.LAST_CALL.pop("recompute_handle", None)
    handled = run(h)
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay", "n": n}
    for i in range(n):
        for got in (plain, handled):
            assert _same(got[i]["y"], direct[i]["y"]) and _same(
                got[i]["dx"], direct[i]["dx"]
            ), (name, i)
            _close(got[i]["dw"], direct[i]["dw"], f"{name} {i} checkpointed dweight")
    assert len(h.entries[conv_kcp.HANDLE_KEY]["hists"]) == n
    assert h.nbytes() == sum(int(ctx.halo.n_hist_rows) * C * 2 for C in Cs)


def test_multi_gradient_accumulation_with_retain_graph() -> None:
    from torch.utils.checkpoint import checkpoint

    from flashinfer import cake_causal_conv1d_train_kcp as facade

    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 256), device
    )

    def fn(xq, xk, xv, wq, wk, wv, h):
        return facade.cake_causal_conv1d_multi_kcp(
            (xq, xk, xv), (wq, wk, wv), ctx, recompute_handle=h, **NONDET
        )

    def leaves():
        return [x.clone().requires_grad_(True) for x in xs] + [
            w.clone().requires_grad_(True) for w in ws
        ]

    a = leaves()
    torch.autograd.backward(list(fn(*a, None)), list(dys))
    direct = [t.grad for t in a]
    a = leaves()
    ys = checkpoint(fn, *a, conv_kcp.RecomputeHandleKcp(), use_reentrant=False)
    torch.autograd.backward(list(ys), list(dys), retain_graph=True)
    torch.autograd.backward(list(ys), list(dys))
    for k, (g, d) in enumerate(zip((t.grad for t in a), direct, strict=True)):
        if k < 3:
            assert _same(g, d + d), k  # dx: the same bits doubled exactly
        else:
            _close(g, d + d, f"accumulated dweight {k}", scale=2.0)


def test_multi_empty_rank_returns_empty_outputs() -> None:
    device = torch.device("cuda")
    ctx, _, _, _ = _context((0,), (0,), device)
    Cs = (512, 256, 512)
    xs = [torch.empty(0, C, dtype=torch.bfloat16, device=device) for C in Cs]
    ws = [torch.randn(C, WIDTH, device=device) for C in Cs]
    ys, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(xs, ws, ctx, **NONDET)
    dxs, dws = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
        xs, ws, ctx, [torch.empty_like(x) for x in xs], hists, **NONDET
    )
    for y, dx, dw, C in zip(ys, dxs, dws, Cs, strict=True):
        assert y.shape == (0, C) and dx.shape == (0, C)
        assert torch.equal(dw, torch.zeros_like(dw))


def test_multi_graph_replay() -> None:
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 256), device
    )
    ref = _singles(ctx, xs, ws, dys)
    try:
        events = []
        for _ in range(4):
            ys, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(
                xs, ws, ctx, graph="replay", **NONDET
            )
            events.append(conv_kcp.LAST_CALL["fwd_multi"]["graph_event"])
            dxs, dws = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
                xs, ws, ctx, dys, hists, graph="replay", **NONDET
            )
            events.append(conv_kcp.LAST_CALL["bwd_multi"]["graph_event"])
            torch.cuda.synchronize()
            for i, r in enumerate(ref):
                assert (
                    _same(ys[i], r["y"])
                    and _same(hists[i], r["hist"])
                    and _same(dxs[i], r["dx"])
                )
                _close(dws[i], r["dw"], f"replayed dweight {i}")
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
    finally:
        conv_kcp._GRAPHS.clear()


def test_multi_one_rank_issues_no_exchange(monkeypatch) -> None:
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem(
        (5000, 1003), (1, 2, 4997, 1003), (512, 512, 512), device
    )
    calls = []
    original = conv_kcp._exchange
    monkeypatch.setattr(
        conv_kcp, "_exchange", lambda *a, **kw: calls.append(1) or original(*a, **kw)
    )
    ys, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(xs, ws, ctx, **NONDET)
    conv_kcp.cake_causal_conv1d_multi_backward_kcp(xs, ws, ctx, dys, hists, **NONDET)
    assert calls == []


def test_multi_torch_deterministic_mode_alerts() -> None:
    device = torch.device("cuda")
    ctx, xs, ws, dys = _problem((65,), (64, 1), (256, 128), device)
    _, hists = conv_kcp.cake_causal_conv1d_multi_forward_kcp(xs, ws, ctx, **NONDET)
    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        torch.use_deterministic_algorithms(True)
        with pytest.raises(RuntimeError, match="cake_causal_conv1d_multi_backward_kcp"):
            conv_kcp.cake_causal_conv1d_multi_backward_kcp(
                xs, ws, ctx, dys, hists, **NONDET
            )
        torch.use_deterministic_algorithms(True, warn_only=True)
        with pytest.warns(UserWarning, match="cake_causal_conv1d_multi_backward_kcp"):
            dxs, dws = conv_kcp.cake_causal_conv1d_multi_backward_kcp(
                xs, ws, ctx, dys, hists, **NONDET
            )
        ref = _singles(ctx, xs, ws, dys)
        for i, r in enumerate(ref):
            assert _same(dxs[i], r["dx"]), i
            _close(dws[i], r["dw"], f"warn_only dweight {i}")
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
