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

The overlapped KCP backward of the non-deterministic package: the main program ``kcp_bwd_dyn``, launched before the
halo exchange's wait with every ``tail_row_off`` entry ``-1``, then the tail-row program ``kcp_bwd_halo`` after it.
Per row of the deterministic suite and a C = 10240 row, on both host paths, the exchange replaced by one that returns
the rank's receive pool with a wait the test records:

- the backward launches the head program (when the rank has head rows), the main program, waits, then launches the
  tail-row program (when the rank receives tail rows), and never the single backward program;
- ``dx`` is bitwise the deterministic KCP backward's and the single program's (the ``bwd_route`` test knob
  ``"single"``) on each of ``REPEATS`` launches, ``dweight`` within the package budget of the FP64 oracle every time,
  and the accumulator workspace reads back zero after each.

Every test skips until the exporter has rendered the package.
"""

from __future__ import annotations

import pytest
import torch

from flashinfer.experimental.cake import cake_causal_conv1d_nondet_train as pkg
from flashinfer.experimental.cake.cake_causal_conv1d_nondet_train import (
    kcp as conv_kcp,
)
from tests.cake.test_helpers.cake_package import cake_package
from tests.test_cake_causal_conv1d_det_train_kcp import (
    DW_ATOL,
    DW_RTOL,
    ROWS,
    _assert_within,
    _bits,
    _problem,
    _reference,
)

det_kcp = cake_package("cake_causal_conv1d_det_train", "kcp")

REPEATS = 5
WIDE_ROWS = [("wide-qkv", (2000, 1003), (1, 2, 1997, 1003), 10240)]

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_causal_conv1d_nondet_train_supported(torch.device("cuda")),
    reason=(
        "the non-deterministic cake_causal_conv1d_nondet_train package needs a supported CUDA device (SM100 / SM103 / "
        "SM107) and its rendered build.json / csrc"
    ),
)


def _stream_workspace() -> torch.Tensor | None:
    device = torch.device("cuda", torch.cuda.current_device())
    key = (device.index, int(torch.cuda.current_stream(device).cuda_stream))
    return pkg._BACKWARD_WORKSPACES.get(key)


class _TwoRanks:
    """A context that reports two ranks (so that the multi entry calls the exchange) and is otherwise ``ctx``."""

    world_size = 2

    def __init__(self, ctx) -> None:
        self._ctx = ctx

    def __getattr__(self, name):
        return getattr(self._ctx, name)


def _recorder(monkeypatch) -> list[str]:
    """The roles of the programs the forward and the backward launch, in launch order, with ``"wait"`` where they wait
    for the exchange (the exchange returns a clone of the rank's receive pool and a work object that records its
    wait).  A record of the zero-history registry (the two-launch forward's single-rank main program) reports its own
    role there (``"fwd"``)."""

    order: list[str] = []
    roles: dict[int, str] = {}
    launch_plan, launch_trim = conv_kcp._launch_plan, conv_kcp._launch_trim
    zero_history_plan = conv_kcp._zero_history_plan
    launch_with_grid, run_module = conv_kcp._launch_with_grid, conv_kcp.run_module
    exchange = conv_kcp._exchange

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
    monkeypatch.setattr(conv_kcp, "_zero_history_plan", recording_zero_history_plan)
    monkeypatch.setattr(conv_kcp, "_launch_trim", recording_trim)
    monkeypatch.setattr(conv_kcp, "_launch_with_grid", recording_with_grid)
    monkeypatch.setattr(conv_kcp, "run_module", recording_run)
    monkeypatch.setattr(conv_kcp, "_exchange", recording_exchange)
    return order


@pytest.mark.parametrize(
    "name,lens,cuts,C", ROWS + WIDE_ROWS, ids=[r[0] for r in ROWS + WIDE_ROWS]
)
@pytest.mark.parametrize("host_trim", [1, 0])
def test_overlapped_backward_order_and_outputs(
    monkeypatch, name, lens, cuts, C, host_trim
) -> None:
    device = torch.device("cuda")
    p = _problem(lens, cuts, C, device)
    _, hist = conv_kcp.cake_causal_conv1d_nondet_train_kcp_forward(
        p.x_local, p.w, p.ctx
    )
    _, hist_det = det_kcp.cake_causal_conv1d_det_train_kcp_forward(
        p.x_local, p.w, p.ctx
    )
    dx_det, _ = det_kcp.cake_causal_conv1d_det_train_kcp_backward(
        p.x_local, p.w, p.ctx, p.dy_local, hist_det
    )
    _, _, dw_ref = _reference(p.x, p.w, p.cu, p.dy)
    dw_ref = dw_ref.float()
    try:
        conv_kcp._set_test_knobs({"host_trim": host_trim, "bwd_route": "single"})
        dx_single, _ = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
            p.x_local, p.w, p.ctx, p.dy_local, hist
        )
        conv_kcp._set_test_knobs({"host_trim": host_trim})
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
        order = _recorder(monkeypatch)
        want = []
        if p.S and int(p.ctx.halo.n_head_rows):
            want.append(conv_kcp.KCP_BWD_HEAD)
        want += [conv_kcp.KCP_BWD_DYN, "wait"]
        if int(p.ctx.halo.n_tail_rows):
            want.append(conv_kcp.KCP_BWD_HALO)
        for repeat in range(REPEATS):
            order.clear()
            dx, dw = conv_kcp.cake_causal_conv1d_nondet_train_kcp_backward(
                p.x_local, p.w, p.ctx, p.dy_local, hist
            )
            torch.cuda.synchronize()
            label = f"{name} host_trim={host_trim} launch {repeat}"
            assert order == want, (label, order)
            assert torch.equal(_bits(dx), _bits(dx_det)), label
            assert torch.equal(_bits(dx), _bits(dx_single)), label
            _assert_within(
                dw, dw_ref, atol=DW_ATOL, rtol=DW_RTOL, label=f"dweight, {label}"
            )
            held = _stream_workspace()
            assert held is not None and int(torch.count_nonzero(held)) == 0, label
    finally:
        monkeypatch.undo()
        conv_kcp._set_test_knobs(None)
        conv_kcp.cake_causal_conv1d_nondet_train_kcp_host_cache_clear()
