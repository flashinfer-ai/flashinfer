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

Stall regression of the Cake BF16 RMSNorm training backward (issue #7038).

The backward is a main kernel whose persistent CTAs claim row chunks, followed
by a dw reduce kernel under programmatic dependent launch; neither needs the
whole device.  Beside a long-resident one-thread kernel on another stream the
backward must keep its speed: per hidden size (the residual route included)
and per token count of the issue's list, the median of seven event-timed
calls with the side kernel resident is at most ``MAX_RATIO`` times the median
of seven calls alone, or at most ``MAX_EXTRA_MS`` above it.  The alone and
busy calls alternate in pairs (alone, busy, alone, busy, ...), so host launch
gaps and clock drift hit both arms alike.  The bounds sit well above the
slowdown a resident kernel causes (up to 13 % at large token counts, where the
CTA sharing its SM runs slower through the chunk it holds; a few microseconds
at small ones) and well below the stall this test guards against, a wait of
milliseconds behind the resident kernel.  ``dw`` is deterministic:
bitwise equal across the alone and busy calls, a fresh-workspace call and
launch grids other than the production one (a CTA count that is not the slot
count claims the same chunk table), and the workspace counters are zero after
every call.
"""

from __future__ import annotations

import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
cake_rmsnorm_train_backward = pkg.cake_rmsnorm_train_backward
cake_rmsnorm_train_backward_workspace = pkg.cake_rmsnorm_train_backward_workspace
cake_rmsnorm_train_forward = pkg.cake_rmsnorm_train_forward
is_cake_rmsnorm_train_supported = pkg.is_cake_rmsnorm_train_supported

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = ((6144, 1e-5), (4096, 1e-5), (2048, 1e-6), (1536, 1e-6), (512, 1e-6))
EPS = {hidden: eps for hidden, eps in WIDTHS}
# (hidden, residual): every backward route of the package.
ROUTES = tuple((hidden, False) for hidden, _eps in WIDTHS) + ((6144, True),)
# The issue's token counts: the repro's 16384 and the four others of its list.
TOKENS = (256, 1024, 4096, 16384, 65536)
# Side kernel: one thread resident for about 5 ms (the issue's repro); the main
# stream sleeps first so the side kernel is resident when the backward starts.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
# A row passes on either bound (``_keeps_its_speed``): the busy median over the
# alone median, or the busy median minus the alone median in ms.  The bounds sit
# above the slowdown a resident kernel causes (up to 13 % at large token counts,
# a few microseconds at small ones) and below the stall guarded against, which
# is milliseconds long.
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
SEED = 760


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_unless_route(hidden: int, residual: bool) -> None:
    if not is_cake_rmsnorm_train_supported(_device(), hidden):
        pytest.skip(f"hidden={hidden} has no exported route on this device")
    if residual and not loader.route_applies(
        device_capability=_capability(), hidden=hidden, kind="bwd", residual=True
    ):
        pytest.skip(f"hidden={hidden} has no exported residual backward on this device")


def _inputs(rows: int, hidden: int, residual: bool):
    generator = torch.Generator(device=_device()).manual_seed(
        (SEED * 1_000_003 + rows * 7 + hidden) % (2**31 - 1)
    )
    kwargs = dict(dtype=torch.float32, device=_device(), generator=generator)
    x = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(hidden, **kwargs)).to(torch.bfloat16)
    g = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16)
    g_h = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16) if residual else None
    _y, rstd = cake_rmsnorm_train_forward(x, w, EPS[hidden])
    return x, w, g, g_h, rstd


def _backward(x, w, g, g_h, rstd, workspace):
    kwargs = {"workspace": workspace}
    if g_h is not None:
        kwargs["g_residual"] = g_h
    return cake_rmsnorm_train_backward(g, x, w, rstd, **kwargs)


def _timed(call, *, side: torch.cuda.Stream | None) -> tuple[float, torch.Tensor]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    _dx, dw = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), dw.clone()


def _keeps_its_speed(alone_ms: float, busy_ms: float) -> bool:
    """The rule a row passes on: busy over alone within ``MAX_RATIO``, or busy at
    most ``MAX_EXTRA_MS`` above alone."""
    return busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS


@pytest.mark.parametrize("rows", TOKENS)
@pytest.mark.parametrize("hidden,residual", ROUTES, ids=lambda v: str(v))
def test_backward_keeps_its_speed_beside_a_resident_kernel(
    hidden: int, residual: bool, rows: int
) -> None:
    _skip_unless_route(hidden, residual)
    x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
    workspace = cake_rmsnorm_train_backward_workspace(
        rows, hidden, _device(), residual=residual
    )
    side = torch.cuda.Stream()

    def call():
        return _backward(x, w, g, g_h, rstd, workspace)

    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    ratio = busy_ms / alone_ms
    extra_ms = busy_ms - alone_ms
    print(
        f"STALL hidden={hidden} residual={residual} rows={rows} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={ratio:.3f} "
        f"extra_ms={extra_ms:.4f}"
    )
    # R2: dw is a pure function of the inputs, never of the scheduling the side kernel imposes.
    reference = alone[0][1]
    assert torch.isfinite(reference).all()
    for _t, dw in alone[1:] + busy:
        assert torch.equal(dw, reference), (
            f"hidden={hidden} rows={rows}: dw differs across calls"
        )
    fresh = _backward(x, w, g, g_h, rstd, None)[1]
    assert torch.equal(fresh, reference), "a fresh workspace changed dw"
    torch.cuda.synchronize()
    assert torch.count_nonzero(workspace[: loader.COUNTERS_BYTES]) == 0, (
        "the counters are not reset"
    )
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"hidden={hidden} residual={residual} rows={rows}: backward alone {alone_ms:.4f} ms, "
        f"beside a resident one-thread kernel {busy_ms:.4f} ms: ratio {ratio:.3f} "
        f"(at most {MAX_RATIO}) and {extra_ms:.4f} ms more (at most {MAX_EXTRA_MS} ms)"
    )


@pytest.mark.parametrize("rows", (129, 4096, 16384))
@pytest.mark.parametrize("hidden,residual", ROUTES, ids=lambda v: str(v))
def test_dw_is_bitwise_independent_of_the_launch_grid(
    hidden: int, residual: bool, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk table depends on ``(rows, slots, rows_per_issue)`` only: launching fewer CTAs than
    slots (they claim more chunks each) leaves every partial row, and so ``dw``, bitwise unchanged."""

    _skip_unless_route(hidden, residual)
    x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
    workspace = cake_rmsnorm_train_backward_workspace(
        rows, hidden, _device(), residual=residual
    )
    reference = _backward(x, w, g, g_h, rstd, workspace)[1].clone()
    production = loader.launch_grid

    def at(divisor: int):
        def launch_grid(record, rows, *, device_index):
            grid = production(record, rows, device_index=device_index)
            if record["grid_rule"]["kind"] != "claimed_chunks":
                return grid
            return (max(1, grid[0] // divisor), 1, 1)

        return launch_grid

    begin = loader.COUNTERS_BYTES
    for divisor in (2, 3) + (
        (10**9,) if rows <= 4096 else ()
    ):  # half, a third, one CTA
        monkeypatch.setattr(loader, "launch_grid", at(divisor))
        # the first partial row: every chunk read must be rewritten
        workspace[begin : begin + 4 * hidden].view(torch.float32).fill_(float("nan"))
        dw = _backward(x, w, g, g_h, rstd, workspace)[1]
        torch.cuda.synchronize()
        assert torch.equal(dw, reference), (
            f"hidden={hidden} rows={rows}: dw changed at grid / {divisor}"
        )
        assert torch.count_nonzero(workspace[: loader.COUNTERS_BYTES]) == 0
    monkeypatch.setattr(loader, "launch_grid", production)
