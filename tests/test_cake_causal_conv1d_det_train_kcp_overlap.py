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

The overlapped KCP backward of the deterministic package (``kcp`` module docstring, "Overlapped backward"): with
``cooperative=False`` and an exchange in flight, the ``kcp_bwd_split`` program and its reduce program run before the
halo exchange's wait with every ``tail_row_off`` entry ``-1``, then the tail-row program ``kcp_bwd_halo`` after it.
One process; the exchange is replaced by one that returns a copy of the rank's receive pool with a work object whose
wait the test records (the multi entry also sees a context of two ranks, so that it exchanges).  Per row of the
single-GPU suites, on both host paths:

- the order of the launches and the wait: head program (when the rank has head rows), split program, reduce program,
  wait, tail-row program (when the rank receives tail rows); with the ``bwd_overlap`` knob ``0`` the wait comes first
  and no tail-row program runs; ``cooperative=True`` waits first as before; the multi entry, with the device's
  architecture in ``MULTI_BWD_OVERLAP_ARCHS`` (shipped: sm_103a and sm_107a): every tensor's head program, the first
  tensor's split and reduce programs, wait, its tail-row program, then the other tensors' split and reduce programs;
  without it, the waiting order (both are run on every architecture);
- ``dx`` and ``dweight`` of each of ``REPEATS`` overlapped calls are bitwise those of the waiting split launch, and
  ``dx`` is bitwise the cooperative launch's;
- the trimmed host path's shared all ``-1`` ``tail_row_off`` rows outlive the host cache, and a new buffer of them is
  filled before any stream can read it.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.cake_causal_conv1d_det_train import (
    is_cake_causal_conv1d_det_train_supported,
)
from flashinfer.experimental.cake.cake_causal_conv1d_det_train import (
    kcp as conv_kcp,
)
from tests.test_cake_causal_conv1d_det_train_kcp import (
    ROWS,
    ROWS_8365,
    _bits,
    _problem,
)
from tests.test_cake_causal_conv1d_det_train_kcp_multi import (
    MROWS,
    _problem as _multi_problem,
)

REPEATS = 3

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_causal_conv1d_det_train_supported(torch.device("cuda")),
    reason="cake_causal_conv1d_det_train needs a supported CUDA device (SM100 / SM103 / SM107)",
)


class _TwoRanks:
    """A context that reports two ranks (so that the multi entry calls the exchange) and is otherwise ``ctx``."""

    world_size = 2

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _recorder(monkeypatch) -> list[str]:
    """The roles of the programs the forward and the backward launch, in launch order, with ``"wait"`` where they wait
    for the exchange (the exchange returns a copy of the rank's receive pool and a work object that records its wait).
    A record of the zero-history registry (the reduce program, the two-launch forward's single-rank main program)
    reports its own role there (``"reduce"``, ``"fwd"``)."""

    order: list[str] = []
    roles: dict[int, str] = {}
    launch_plan, reduce_plan = conv_kcp._launch_plan, conv_kcp._reduce_plan
    zero_history_plan = conv_kcp._zero_history_plan
    launch_trim, launch_with_grid = conv_kcp._launch_trim, conv_kcp._launch_with_grid
    run_module, exchange = conv_kcp.run_module, conv_kcp._exchange

    def role(name: str) -> str:
        record = conv_kcp._loader.MODULES.get(name)
        if record is None:
            record = conv_kcp._zero_history.MODULES[name]
        return record["role"]

    def recording_plan(name, grid, device_index):
        info = launch_plan(name, grid, device_index)
        roles[id(info)] = role(name)
        return info

    def recording_zero_history_plan(name, grid, device_index):
        info = zero_history_plan(name, grid, device_index)
        roles[id(info)] = role(name)
        return info

    def recording_reduce(*args, **kwargs):
        info = reduce_plan(*args, **kwargs)
        if info is not None:
            roles[id(info)] = "reduce"
        return info

    def recording_trim(info, values, device_index):
        order.append(roles.get(id(info), "?"))
        return launch_trim(info, values, device_index)

    def recording_with_grid(name, values, *, grid, device_index):
        order.append(role(name))
        return launch_with_grid(name, values, grid=grid, device_index=device_index)

    def recording_run(name, values, *, device_index, grid=None):
        order.append(role(name))
        return run_module(name, values, device_index=device_index, grid=grid)

    class _Work:
        def wait(self) -> None:
            order.append("wait")

    def recording_exchange(send, **kwargs):
        ctx = kwargs["ctx"]
        inner = ctx._ctx if isinstance(ctx, _TwoRanks) else ctx
        recv, work = exchange(send, **{**kwargs, "ctx": inner})
        assert work is None, "one rank: the package's exchange returns no work"
        return recv.clone(), _Work()

    monkeypatch.setattr(conv_kcp, "_launch_plan", recording_plan)
    monkeypatch.setattr(conv_kcp, "_reduce_plan", recording_reduce)
    monkeypatch.setattr(conv_kcp, "_zero_history_plan", recording_zero_history_plan)
    monkeypatch.setattr(conv_kcp, "_launch_trim", recording_trim)
    monkeypatch.setattr(conv_kcp, "_launch_with_grid", recording_with_grid)
    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    monkeypatch.setattr(conv_kcp, "_exchange", recording_exchange)
    return order


def _want(ctx, *, overlap: bool, cooperative: bool, n: int = 1) -> list[str]:
    head = ["kcp_bwd_head"] * n if int(ctx.T_r) and int(ctx.halo.n_head_rows) else []
    # the cooperative record has no reduce program; the split record's reduce follows it on the stream
    main = ["kcp_bwd"] if cooperative else ["kcp_bwd_split", "reduce"]
    if not overlap:
        return head + ["wait"] + main * n
    # the multi entry overlaps its first tensor only: one tail-row program per call
    tail = ["kcp_bwd_halo"] if int(ctx.halo.n_tail_rows) else []
    return head + main + ["wait"] + tail + main * (n - 1)


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + ROWS_8365, ids=[r[0] for r in ROWS + ROWS_8365]
)
@pytest.mark.parametrize("host_trim", [1, 0])
def test_overlapped_backward_order_and_bits(
    monkeypatch, name, lens, cuts, C, host_trim
) -> None:
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = conv_kcp.cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    ref = {}
    for coop in (True, False):
        ref[coop] = conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
            p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=coop
        )
    torch.cuda.synchronize()
    try:
        order = _recorder(monkeypatch)
        for knob, coop in ((1, False), (0, False), (1, True)):
            conv_kcp._set_test_knobs({"host_trim": host_trim, "bwd_overlap": knob})
            conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()
            want = _want(p.ctx, overlap=knob == 1 and not coop, cooperative=coop)
            for repeat in range(REPEATS):
                order.clear()
                dx, dw = conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
                    p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=coop
                )
                torch.cuda.synchronize()
                label = f"{name} trim={host_trim} knob={knob} coop={coop} #{repeat}"
                assert order == want, (label, order)
                assert torch.equal(_bits(dx), _bits(ref[coop][0])), label
                assert torch.equal(_bits(dw), _bits(ref[coop][1])), label
                assert torch.equal(_bits(dx), _bits(ref[True][0])), label
    finally:
        monkeypatch.undo()
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()


def test_multi_overlap_architectures() -> None:
    assert set(conv_kcp.MULTI_BWD_OVERLAP_ARCHS) == {"sm_103a", "sm_107a"}


@pytest.mark.parametrize("overlap", [True, False])
@pytest.mark.parametrize("name,lens,cuts,Cs", MROWS, ids=[r[0] for r in MROWS])
def test_overlapped_multi_backward_order_and_bits(
    monkeypatch, name, lens, cuts, Cs, overlap
) -> None:
    """The multi entry has one host path (the trimmed one; the ``host_trim`` knob selects the single call's path).
    ``overlap`` puts the device's architecture into ``MULTI_BWD_OVERLAP_ARCHS`` or takes it out."""
    device = torch.device("cuda")
    arch = conv_kcp._device_arch(device)
    archs = conv_kcp.MULTI_BWD_OVERLAP_ARCHS
    archs = archs | {arch} if overlap else archs - {arch}
    ctx, xs, ws, dys = _multi_problem(lens, cuts, Cs, device)
    _, hists = conv_kcp.cake_causal_conv1d_det_train_kcp_multi_forward(xs, ws, ctx)
    ref_dx, ref_dw = conv_kcp.cake_causal_conv1d_det_train_kcp_multi_backward(
        xs, ws, ctx, dys, hists
    )
    torch.cuda.synchronize()
    two = _TwoRanks(ctx)
    try:
        monkeypatch.setattr(conv_kcp, "MULTI_BWD_OVERLAP_ARCHS", frozenset(archs))
        order = _recorder(monkeypatch)
        want = _want(ctx, overlap=overlap, cooperative=False, n=len(Cs))
        for repeat in range(REPEATS):
            order.clear()
            dxs, dws = conv_kcp.cake_causal_conv1d_det_train_kcp_multi_backward(
                xs, ws, two, dys, hists
            )
            torch.cuda.synchronize()
            label = f"{name} overlap={overlap} #{repeat}"
            assert order == want, (label, order)
            for i in range(len(Cs)):
                assert torch.equal(_bits(dxs[i]), _bits(ref_dx[i])), (label, i)
                assert torch.equal(_bits(dws[i]), _bits(ref_dw[i])), (label, i)
    finally:
        monkeypatch.undo()
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()


def test_overlapped_no_tail_rows_outlive_the_host_cache(monkeypatch) -> None:
    """The trimmed host path's all ``-1`` ``tail_row_off`` rows of the overlapped backward live in ``_NO_TAIL``, not
    only in the host-cache entry that names them: after ``cake_causal_conv1d_det_train_kcp_host_cache_clear`` they
    stay allocated and ``-1`` (a graph captured on them keeps a valid pointer), and the next call names the same rows.
    """
    name, lens, cuts, C = ROWS[0]
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = conv_kcp.cake_causal_conv1d_det_train_kcp_forward(p.x_local, p.w, p.ctx)
    ref = conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=False
    )
    torch.cuda.synchronize()

    def tables() -> list[torch.Tensor]:
        return [
            e["no_tail"]
            for _, entries in conv_kcp._HOST_CACHE.values()
            for e in entries.values()
            if "no_tail" in e
        ]

    try:
        conv_kcp._set_test_knobs({"host_trim": 1})
        conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()
        _recorder(monkeypatch)
        ptrs = []
        for repeat in range(2):
            dx, dw = conv_kcp.cake_causal_conv1d_det_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist, cooperative=False
            )
            torch.cuda.synchronize()
            label = f"{name} #{repeat}"
            assert torch.equal(_bits(dx), _bits(ref[0])), label
            assert torch.equal(_bits(dw), _bits(ref[1])), label
            (table,) = tables()
            ptrs.append(table.data_ptr())
            assert conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear() >= 1
            assert not tables(), label
            bufs = conv_kcp._NO_TAIL[torch.cuda.current_device()]
            assert any(
                b.data_ptr() <= ptrs[-1] < b.data_ptr() + b.numel() * b.element_size()
                for b in bufs
            ), label
            assert all(bool((b == -1).all()) for b in bufs), label
        assert ptrs[0] == ptrs[1], ptrs
    finally:
        monkeypatch.undo()
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_det_train_kcp_host_cache_clear()


def test_overlapped_no_tail_rows_are_filled_before_any_stream_reads_them(
    monkeypatch,
) -> None:
    """Every context and stream of a device shares the ``_NO_TAIL`` rows, so a new buffer is ``-1`` when
    ``_no_tail`` returns even while the calling stream is still busy: a call on another stream never reads it
    unfilled.  The calling stream sleeps far past the read on the other stream."""
    device = torch.device("cuda", torch.cuda.current_device())
    monkeypatch.setattr(conv_kcp, "_NO_TAIL", {})
    busy, other = torch.cuda.Stream(device), torch.cuda.Stream(device)
    tables = []
    try:
        for n in (5, 64, 65):  # the first buffer, a fitting entry, a grown buffer
            entry = {"n_frag": n, "device_index": device.index}
            with torch.cuda.stream(busy):
                torch.cuda._sleep(1 << 30)
                table = conv_kcp._no_tail(entry, device)
            with torch.cuda.stream(other):
                filled = bool((table == -1).all())
            assert filled, n
            assert table.shape == (n, conv_kcp.HALO) and entry["no_tail"] is table, n
            tables.append(table)
        assert tables[1].data_ptr() == tables[0].data_ptr()
        assert len(conv_kcp._NO_TAIL[device.index]) == 2
    finally:
        torch.cuda.synchronize(device)
        monkeypatch.undo()
