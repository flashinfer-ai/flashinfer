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

The opt-in SM reserve of the non-deterministic KCP backward (``bwd_reserve_sms``, FlashInfer issue #8365): one
process, world size 1, the deterministic KCP suite's contexts and FP64 oracle.

(a) The keyword of ``flashinfer.cake_causal_conv1d_nondet_train_kcp``: ``None`` is never forwarded; a given value
    reaches the non-deterministic backend, which raises ``TypeError`` for a ``bool`` or a non-``int`` and
    ``ValueError`` below zero; torch's deterministic mode changes nothing at the call (the value is forwarded,
    nothing warns here); nothing runs the deterministic package.
(b) ``K`` = 1 and 2: the main backward launches the single-wave grid less ``K x ctas_per_sm`` CTAs on both host
    paths, at a width the shipped grid already trims and at one it does not; ``dx`` is bitwise the deterministic
    backward's, ``dweight`` within budget of the FP64 oracle, the workspace reads back zero.
(c) The autograd single and multi entries with ``bwd_reserve_sms=1``: ``y`` / ``dx`` bitwise the default calls'
    (the multi call's bitwise the single calls'), ``dweight`` within budget, the reserved grid launched; activation
    checkpointing with a recompute handle keeps the reserve; graph replay keys a reserved call apart from a default
    one; a reserve at the device's SM count is refused before the forward.
(d) sm_107a (``RESERVE_UNSUPPORTED``): every entry point refuses ``K >= 1`` with a ``ValueError`` that names the
    architecture and the reason, before it launches anything; ``None`` and 0 pass.  (b) and (c) run where the
    reserve is supported; on sm_107a the device itself is checked to refuse it.
"""

from __future__ import annotations

import warnings

import pytest
import torch

from flashinfer import cake_causal_conv1d_nondet_train_kcp as facade
from flashinfer.experimental.cake import cake_causal_conv1d_nondet_train as pkg
from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.experimental.test_cake_causal_conv1d_nondet_train_kcp import (
    _bwd,
    _dw_within,
    _fwd,
    _stream_workspace,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    _bits,
    _problem,
    _reference,
)

# the deterministic package's KCP backend, through the one place the suites name packages
det_kcp = cake_package("cake_causal_conv1d_det_train", "kcp")

LENS, CUTS = (5000, 1003), (1, 2, 4997, 1003)


def _device_arch_or_none():
    if (
        not torch.cuda.is_available()
        or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda"))
    ):
        return None
    return conv_kcp._device_arch(torch.device("cuda"))


ARCH = _device_arch_or_none()

needs_gpu = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)
needs_reserve = pytest.mark.skipif(
    ARCH is None or ARCH in conv_kcp.RESERVE_UNSUPPORTED,
    reason="bwd_reserve_sms >= 1 needs a device whose architecture supports the reserve (not sm_107a)",
)


# ----------------------------------------------------------------------------------------------------------------
# (a) The facade keyword.


class _Recorder:
    """Stands in for an entry point: records the keywords of each call and returns a marker."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, *args, **kwargs):
        self.calls.append(kwargs)
        return "called"


def _route_recorders(monkeypatch):
    from flashinfer import cake_causal_conv1d_det_train_kcp as det_facade

    det_single, det_multi, nd_single, nd_multi = (_Recorder() for _ in range(4))
    monkeypatch.setattr(det_facade, "cake_causal_conv1d_det_kcp", det_single)
    monkeypatch.setattr(det_facade, "cake_causal_conv1d_det_kcp_multi", det_multi)
    monkeypatch.setattr(conv_kcp, "cake_causal_conv1d_nondet_kcp", nd_single)
    monkeypatch.setattr(conv_kcp, "cake_causal_conv1d_nondet_kcp_multi", nd_multi)
    return det_single, det_multi, nd_single, nd_multi


def _entry(multi: bool):
    return (
        facade.cake_causal_conv1d_nondet_kcp_multi
        if multi
        else facade.cake_causal_conv1d_nondet_kcp
    )


@pytest.mark.parametrize("bad", [True, False, 1.0, "1", (1,)])
@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_keyword_type_is_checked_when_given(bad, multi) -> None:
    with pytest.raises(TypeError, match="bwd_reserve_sms must be an int or None"):
        _entry(multi)(None, None, None, bwd_reserve_sms=bad)


@pytest.mark.parametrize("multi", [False, True], ids=["single", "multi"])
def test_negative_reserve_is_refused(multi) -> None:
    with pytest.raises(ValueError, match="bwd_reserve_sms must be >= 0"):
        _entry(multi)(None, None, None, bwd_reserve_sms=-1)


def test_keyword_routing(monkeypatch) -> None:
    """``None`` is not forwarded; a given value reaches the non-deterministic backend unchanged, also under torch's
    deterministic mode (strict), where nothing warns at the call; no ``deterministic`` keyword is forwarded and
    nothing runs the deterministic package."""

    det_single, det_multi, nd_single, nd_multi = _route_recorders(monkeypatch)
    facade.cake_causal_conv1d_nondet_kcp(None, None, None)
    facade.cake_causal_conv1d_nondet_kcp(None, None, None, bwd_reserve_sms=0)
    facade.cake_causal_conv1d_nondet_kcp(None, None, None, bwd_reserve_sms=2)
    facade.cake_causal_conv1d_nondet_kcp_multi(None, None, None, bwd_reserve_sms=1)
    facade.cake_causal_conv1d_nondet_kcp_multi(None, None, None)
    assert [c.get("bwd_reserve_sms", "absent") for c in nd_single.calls] == [
        "absent",
        0,
        2,
    ]
    assert [c.get("bwd_reserve_sms", "absent") for c in nd_multi.calls] == [
        1,
        "absent",
    ]

    prev, prev_warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    torch.use_deterministic_algorithms(True)
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            facade.cake_causal_conv1d_nondet_kcp(None, None, None, bwd_reserve_sms=1)
            facade.cake_causal_conv1d_nondet_kcp_multi(
                None, None, None, bwd_reserve_sms=3
            )
        assert caught == []
    finally:
        torch.use_deterministic_algorithms(prev, warn_only=prev_warn)
    assert nd_single.calls[-1]["bwd_reserve_sms"] == 1
    assert nd_multi.calls[-1]["bwd_reserve_sms"] == 3
    assert all("deterministic" not in c for c in nd_single.calls + nd_multi.calls)
    assert det_single.calls == [] and det_multi.calls == []


# ----------------------------------------------------------------------------------------------------------------
# (b) The reserved grid.


def _record_main_grids(monkeypatch) -> list:
    """Record the main backward's grid at both launch helpers (trimmed and previous host path)."""

    seen: list = []
    launch_plan, run_module = conv_kcp._launch_plan, conv_kcp.run_module

    def main(name: str) -> bool:
        return conv_kcp._loader.MODULES[name]["role"] == conv_kcp.KCP_BWD

    def recording_plan(name, grid, device_index):
        if main(name):
            seen.append(tuple(grid))
        return launch_plan(name, grid, device_index)

    def recording_run(name, values, *, device_index, grid=None):
        if main(name):
            seen.append(None if grid is None else tuple(grid))
        return run_module(name, values, device_index=device_index, grid=grid)

    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_plan)
    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    return seen


def _widths(device: torch.device) -> list[tuple[int, int, int, int]]:
    """``(C, full grid, ctas_per_sm, shipped grid)`` of the first single-wave width whose shipped grid trims one SM
    and of the first whose shipped grid stays whole (multiples of 1024 up to 16384)."""

    arch = conv_kcp._loader.device_arch(device.index)
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    rows: dict[bool, tuple[int, int, int, int]] = {}
    for C in range(1024, 16385, 1024):
        (name,) = conv_kcp.selected_modules(arch, conv_kcp.KCP_BWD, C=C)
        rule = conv_kcp._loader.MODULES[name]["grid_rule"]
        k, mult = int(rule["ctas_per_sm"]), int(rule["grid_mult"][arch])
        if mult != 1:
            continue
        full = k * sms
        n = conv_kcp.plan(1, C, 1, geometry=rule["geometry"], legacy=True)["n_ch_tiles"]
        trims = full % n >= k
        rows.setdefault(trims, (C, full, k, full - k if trims else full))
    return [rows[t] for t in (True, False) if t in rows]


@needs_reserve
@pytest.mark.parametrize("reserve", [1, 2])
def test_backward_launches_the_reserved_grid(monkeypatch, reserve) -> None:
    device = torch.device("cuda")
    rows = _widths(device)
    assert len(rows) == 2, rows
    seen = _record_main_grids(monkeypatch)
    try:
        for C, full, k, shipped in rows:
            p = _problem(LENS, CUTS, C, device)
            _, hist = _fwd(p.x_local, p.w, p.ctx)
            _, hist_det = det_kcp.cake_causal_conv1d_det_train_kcp_forward(
                p.x_local, p.w, p.ctx
            )
            dx_det, _ = det_kcp.cake_causal_conv1d_det_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist_det
            )
            _, _, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
            # The single program's shipped grid at reserve 0 (the ``bwd_route`` knob "single"); the shipped route
            # runs the overlapped pair at reserve 0 (no ``kcp_bwd`` launch) and the reserved grid at reserve K.
            for host_trim, route in (
                (1, "single"),
                (0, "single"),
                (1, "overlap"),
                (0, "overlap"),
            ):
                conv_kcp._set_test_knobs({"host_trim": host_trim, "bwd_route": route})
                conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
                for k_res, expected in ((0, shipped), (reserve, full - reserve * k)):
                    seen.clear()
                    dx, dw = _bwd(
                        p.x_local, p.w, p.ctx, p.dy_local, hist, bwd_reserve_sms=k_res
                    )
                    torch.cuda.synchronize()
                    label = f"C={C} host_trim={host_trim} route={route} reserve={k_res}"
                    want = (
                        [] if route == "overlap" and k_res == 0 else [(expected, 1, 1)]
                    )
                    assert seen == want, (label, seen)
                    assert torch.equal(_bits(dx), _bits(dx_det)), label
                    _dw_within(dw, dw_ref, f"dweight vs FP64 oracle, {label}")
                    held = _stream_workspace()
                    assert held is None or int(torch.count_nonzero(held)) == 0, label
    finally:
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()


# ----------------------------------------------------------------------------------------------------------------
# (c) The autograd entries, checkpointing and graph replay.


def _autograd(p, entry, **kw):
    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)
    y = entry(x, w, p.ctx, **kw)
    y.backward(p.dy_local)
    return y.detach(), x.grad, w.grad


@needs_reserve
def test_autograd_single_and_multi_keep_the_reserve(monkeypatch) -> None:
    device = torch.device("cuda")
    _, (C, full, k, _) = _widths(
        device
    )  # a width whose shipped grid is whole: K = 1 changes it
    p = _problem(LENS, CUTS, C, device)
    seen = _record_main_grids(monkeypatch)
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    y0, dx0, _ = _autograd(p, facade.cake_causal_conv1d_nondet_kcp)
    seen.clear()
    y1, dx1, dw1 = _autograd(p, facade.cake_causal_conv1d_nondet_kcp, bwd_reserve_sms=1)
    assert seen == [(full - k, 1, 1)], seen
    assert torch.equal(_bits(y1), _bits(y0)) and torch.equal(_bits(dx1), _bits(dx0))
    _, _, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    _dw_within(dw1, dw_ref, "single dweight vs FP64 oracle")

    xs = [p.x_local.clone().requires_grad_(True) for _ in range(2)]
    ws = [p.w.clone().requires_grad_(True) for _ in range(2)]
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()  # the single call above cached this key's plan
    seen.clear()
    ys = facade.cake_causal_conv1d_nondet_kcp_multi(xs, ws, p.ctx, bwd_reserve_sms=1)
    torch.autograd.backward(ys, [p.dy_local, p.dy_local])
    # Both tensors share one (ctx, S, C, device, cu_seqlens, reserve) host-cache entry: one plan, on the reserved grid.
    assert seen == [(full - k, 1, 1)], seen
    keys = [key for key in conv_kcp._HOST_CACHE[id(p.ctx)][1] if key[0] == "bwd"]
    assert keys and all(key[-1] == 1 for key in keys), keys
    for y, x, w in zip(ys, xs, ws, strict=True):
        assert torch.equal(_bits(y.detach()), _bits(y1))
        assert torch.equal(_bits(x.grad), _bits(dx1))
        _dw_within(w.grad, dw_ref, "multi dweight vs FP64 oracle")


@needs_reserve
def test_checkpoint_with_a_recompute_handle_keeps_the_reserve(monkeypatch) -> None:
    from torch.utils.checkpoint import checkpoint

    device = torch.device("cuda")
    _, (C, full, k, _) = _widths(
        device
    )  # a width whose shipped grid is whole: K = 1 changes it
    p = _problem(LENS, CUTS, C, device)
    _, dx_direct, _ = _autograd(
        p, facade.cake_causal_conv1d_nondet_kcp, bwd_reserve_sms=1
    )
    seen = _record_main_grids(monkeypatch)
    conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
    handle = conv_kcp.CakeCausalConv1dNondetKcpRecomputeHandle()
    x = p.x_local.clone().requires_grad_(True)
    w = p.w.clone().requires_grad_(True)

    def fn(x_, w_):
        return facade.cake_causal_conv1d_nondet_kcp(
            x_, w_, p.ctx, recompute_handle=handle, bwd_reserve_sms=1
        )

    y = checkpoint(fn, x, w, use_reentrant=False)
    y.backward(p.dy_local)
    assert conv_kcp.LAST_CALL["recompute_handle"] == {"fwd": "replay"}
    assert seen == [(full - k, 1, 1)], seen
    assert torch.equal(_bits(x.grad), _bits(dx_direct))
    handle.clear()


@needs_reserve
def test_graph_replay_keys_the_reserve(monkeypatch) -> None:
    """A reserved backward and a default one on the same operands are two graph keys, each replaying its own grid;
    ``dx`` bitwise the eager result in both."""

    device = torch.device("cuda")
    _, (C, full, k, shipped) = _widths(device)
    p = _problem(LENS, CUTS, C, device)
    x, w, ctx, dy = p.x_local, p.w, p.ctx, p.dy_local
    _, hist = _fwd(x, w, ctx)
    dx0, _ = _bwd(x, w, ctx, dy, hist)
    try:
        conv_kcp._GRAPHS.clear()
        keys = {}
        for _ in range(4):
            for k_res in (0, 1):
                dx, _ = _bwd(x, w, ctx, dy, hist, graph="replay", bwd_reserve_sms=k_res)
                torch.cuda.synchronize()
                assert torch.equal(dx, dx0), k_res
                keys.setdefault(k_res, set()).add(
                    conv_kcp.LAST_CALL["bwd"]["graph_key"]
                )
        assert len(keys[0]) == 1 and len(keys[1]) == 1 and keys[0] != keys[1], keys
        assert conv_kcp.GRAPH_STATS["fallback"] == 0
        assert shipped == full and full - k < shipped
        seen = _record_main_grids(monkeypatch)
        conv_kcp._GRAPHS.clear()
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        _bwd(x, w, ctx, dy, hist, bwd_reserve_sms=1)
        assert seen == [(full - k, 1, 1)], seen
    finally:
        conv_kcp._GRAPHS.clear()


@needs_reserve
def test_reserve_at_the_sm_count_is_refused_before_the_forward() -> None:
    device = torch.device("cuda")
    sms = torch.cuda.get_device_properties(device).multi_processor_count
    p = _problem(LENS, CUTS, 512, device)
    with pytest.raises(ValueError, match="SMs"):
        facade.cake_causal_conv1d_nondet_kcp(p.x_local, p.w, p.ctx, bwd_reserve_sms=sms)


# ----------------------------------------------------------------------------------------------------------------
# (d) Architectures that refuse the reserve.


def test_sm_107a_is_the_refused_architecture() -> None:
    assert set(conv_kcp.RESERVE_UNSUPPORTED) == {"sm_107a"}
    reason = conv_kcp.RESERVE_UNSUPPORTED["sm_107a"]
    assert "1.055x" in reason and "C=1024" in reason and "1.040" in reason
    assert "RESERVE_UNSUPPORTED" in conv_kcp.__all__


def _refused_entries(p):
    """Every entry point that takes ``bwd_reserve_sms``, as ``(name, call(K))``."""
    x, w, ctx, dy = p.x_local, p.w, p.ctx, p.dy_local
    return [
        (
            "cake_causal_conv1d_nondet_kcp",
            lambda k: facade.cake_causal_conv1d_nondet_kcp(
                x, w, ctx, bwd_reserve_sms=k
            ),
        ),
        (
            "cake_causal_conv1d_nondet_kcp_multi",
            lambda k: facade.cake_causal_conv1d_nondet_kcp_multi(
                (x,), (w,), ctx, bwd_reserve_sms=k
            ),
        ),
        (
            "cake_causal_conv1d_nondet_train_kcp_backward",
            lambda k: conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                x, w, ctx, dy, None, bwd_reserve_sms=k
            ),
        ),
        (
            "cake_causal_conv1d_nondet_train_kcp_multi_backward",
            lambda k: conv_kcp.cake_causal_conv1d_nondet_train_kcp_multi_backward(
                [x], [w], ctx, [dy], [None], bwd_reserve_sms=k
            ),
        ),
    ]


def _launch_nothing(monkeypatch):
    def launched(*args, **kwargs):
        raise AssertionError("a refused call reached a forward or backward launch")

    monkeypatch.setattr(conv_kcp.CakeCausalConv1dNondetKcpFunction, "apply", launched)
    monkeypatch.setattr(
        conv_kcp.CakeCausalConv1dNondetKcpMultiFunction, "apply", launched
    )
    monkeypatch.setattr(conv_kcp, "_kcp_backward", launched)
    monkeypatch.setattr(conv_kcp, "_kcp_backward_multi", launched)


@needs_gpu
@pytest.mark.parametrize("k", [1, 2])
def test_reserve_is_refused_on_sm_107a(monkeypatch, k) -> None:
    """With the device reported as sm_107a, all four entry points raise before any launch; the message names the
    architecture and the measured cost."""

    p = _problem(LENS, CUTS, 512, torch.device("cuda"))
    monkeypatch.setattr(conv_kcp, "_device_arch", lambda device: "sm_107a")
    _launch_nothing(monkeypatch)
    for name, call in _refused_entries(p):
        with pytest.raises(ValueError) as err:
            call(k)
        message = str(err.value)
        assert message.startswith(
            f"{name}: bwd_reserve_sms={k} is not supported on sm_107a"
        ), message
        assert "1.055x" in message and "bwd_reserve_sms=None" in message, message
    for reserve in (conv_kcp.RESERVE_DEFAULT, 0):
        conv_kcp._reserve_fits(reserve, p.x_local, "check")  # None / 0 pass on sm_107a


@pytest.mark.skipif(
    ARCH is None or ARCH not in conv_kcp.RESERVE_UNSUPPORTED,
    reason="runs on a device whose architecture refuses the reserve (sm_107a)",
)
def test_this_device_refuses_the_reserve(monkeypatch) -> None:
    p = _problem(LENS, CUTS, 512, torch.device("cuda"))
    _launch_nothing(monkeypatch)
    for _name, call in _refused_entries(p):
        with pytest.raises(ValueError, match=f"not supported on {ARCH}"):
            call(1)
