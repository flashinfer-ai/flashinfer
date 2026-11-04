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

Stall regression of the Cake BF16 RMSNorm training backward with the
non-deterministic weight gradient (``deterministic=False``; issues #7038,
#7548 item 2, #8066).

The backward runs, per token-count window, one kernel whose persistent CTAs
claim row chunks from a counter and add their per-column ``dw`` sums into FP64
workspace sums (the CTA that finishes last stores ``dw``), or the deterministic
package's pair (main kernel and dw reduce) rendered into this package.  Neither
waits for the whole device: a CTA that becomes resident late finds the chunk
table exhausted.  Beside a
long-resident one-thread kernel on another stream the backward must keep its
speed: per hidden size (the residual route included) and per token count of
the issue's list, the median of seven event-timed calls with the side kernel
resident is at most ``MAX_RATIO`` times the median of seven calls alone, or at
most ``MAX_EXTRA_MS`` above it (the alone and busy calls alternate in pairs).
``dx`` is bitwise equal across the alone and busy calls, a fresh-workspace
call and launch grids other than the production one; ``dw`` stays within the
package's budget against the first call in every one of them; after every
call the workspace is at rest: every counter and FP64 sum zero (a det pair's
partial rows are scratch).

Beside a kernel resident on every SM (#7548 item 2): with the side kernel's
preferred carveout set to the maximum shared memory, the backward starts beside
it and costs at most ``MAX_ALL_SM_EXTRA_MS`` more than alone, far below the side
kernel's ``ALL_SM_SPIN_MS`` lifetime.  Every side CTA records its SM, and a run
counts only when the side kernel holds every SM.  A token count in a
``cake_windows.WHOLE_SM`` range (the deterministic package's whole-SM programs,
which this package runs in some det-pair windows) is not measured
(det-inherited): the test checks that the call runs the deterministic package's
program and that the docstrings list the range with the remedy, then skips.  The
forward programs are the deterministic package's, whose stall test covers them.

Beside a kernel holding shared memory on every SM (review R3, C3): one CTA per
SM with each of ``cake_windows.SMEM_SIDE_KB`` KB of dynamic shared memory
(max-shared carveout).  For every shipped backward window -- every route,
residual and FP32-weight routes included -- the smallest of those sizes beside
which the call waits is at least the package's documented
``cake_windows.SMEM_WAITS`` entry.  The list covers every window; its entry is
the most restrictive level measured on the architecture's boards (the smallest
size beside which any of them waits), so a board waits at that size or only at
a larger one, never at a smaller one.  The deterministic program the window
replaces is measured for reference.
"""

from __future__ import annotations

import ctypes
import functools
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
windows = cake_package("cake_rmsnorm_train", "cake_windows")
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_train_backward_workspace, deterministic=False
)
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
WIDTHS = (
    (6144, 1e-5),
    (5120, 1e-5),
    (4096, 1e-5),
    (2048, 1e-6),
    (1536, 1e-6),
    (512, 1e-6),
    (256, 1e-6),
)
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
# dw of a call against the first alone call's: both are FP32 roundings of FP64
# sums taken in a scheduling-dependent order.
DW_REL_L2 = 1e-6


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and pkg.is_cake_rmsnorm_train_supported(_device(), deterministic=False)
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train(deterministic=False) needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_unless_route(hidden: int, residual: bool) -> None:
    if not pkg.is_cake_rmsnorm_train_supported(_device(), hidden, deterministic=False):
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
    _y, rstd = forward(x, w, EPS[hidden])
    return x, w, g, g_h, rstd


def _backward(x, w, g, g_h, rstd, workspace):
    kwargs = {"workspace": workspace}
    if g_h is not None:
        kwargs["g_residual"] = g_h
    return backward(g, x, w, rstd, **kwargs)


def _rel_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    a, b = actual.double(), reference.double()
    den = b.norm().item()
    return (a - b).norm().item() / den if den > 0 else (a - b).norm().item()


def _workspace_clean(ws: torch.Tensor, hidden: int, residual: bool = False) -> bool:
    """The workspace's documented rest state after a call (any hidden size or
    residual flag): every counter and FP64 sum zero -- every byte before the det
    pair's partial rows (``cake_jit.PAIR_ROWS_OFFSET``), which are scratch."""
    del hidden, residual  # one layout serves every family
    torch.cuda.synchronize()
    return not bool(ws[: loader.PAIR_ROWS_OFFSET].any())


def _timed(
    call, *, side: torch.cuda.Stream | None
) -> tuple[float, tuple[torch.Tensor, torch.Tensor]]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    dx, dw = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), (dx.clone(), dw.clone())


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
    workspace = workspace_for(rows, hidden, _device(), residual=residual)
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
    # dx is a pure function of the inputs; dw stays within its budget whatever the side kernel imposes.
    dx_ref, dw_ref = alone[0][1]
    assert torch.isfinite(dw_ref).all()
    for _t, (dx, dw) in alone[1:] + busy:
        assert torch.equal(dx, dx_ref), (
            f"hidden={hidden} rows={rows}: dx differs across calls"
        )
        assert _rel_l2(dw, dw_ref) <= DW_REL_L2, (
            f"hidden={hidden} rows={rows}: dw rel-L2 {_rel_l2(dw, dw_ref):.3e} across calls"
        )
    dx_fresh, dw_fresh = _backward(x, w, g, g_h, rstd, None)
    assert torch.equal(dx_fresh, dx_ref), "a fresh workspace changed dx"
    assert _rel_l2(dw_fresh, dw_ref) <= DW_REL_L2, "a fresh workspace moved dw"
    assert _workspace_clean(workspace, hidden, residual), (
        "the workspace is not the kernels' state"
    )
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"hidden={hidden} residual={residual} rows={rows}: backward alone {alone_ms:.4f} ms, "
        f"beside a resident one-thread kernel {busy_ms:.4f} ms: ratio {ratio:.3f} "
        f"(at most {MAX_RATIO}) and {extra_ms:.4f} ms more (at most {MAX_EXTRA_MS} ms)"
    )


@pytest.mark.parametrize("rows", (129, 4096, 16384))
@pytest.mark.parametrize("hidden,residual", ROUTES, ids=lambda v: str(v))
def test_outputs_do_not_depend_on_the_launch_grid(
    hidden: int, residual: bool, rows: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The chunk table depends on ``(rows, slots, rows_per_issue)`` only: launching fewer CTAs than
    slots (they claim more chunks each) leaves ``dx`` bitwise unchanged and ``dw`` within its budget,
    and the CTA that finishes last still finds every other CTA counted."""

    _skip_unless_route(hidden, residual)
    x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
    workspace = workspace_for(rows, hidden, _device(), residual=residual)
    dx_ref, dw_ref = (t.clone() for t in _backward(x, w, g, g_h, rstd, workspace))
    production = loader.launch_grid

    def at(divisor: int):
        def launch_grid(record, rows, *, device_index):
            grid = production(record, rows, device_index=device_index)
            if record["grid_rule"]["kind"] != "claimed_chunks":
                return grid
            return (max(1, grid[0] // divisor), 1, 1)

        return launch_grid

    for divisor in (2, 3) + (
        (10**9,) if rows <= 4096 else ()
    ):  # half, a third, one CTA
        monkeypatch.setattr(loader, "launch_grid", at(divisor))
        dx, dw = _backward(x, w, g, g_h, rstd, workspace)
        torch.cuda.synchronize()
        assert torch.equal(dx, dx_ref), (
            f"hidden={hidden} rows={rows}: dx changed at grid / {divisor}"
        )
        assert _rel_l2(dw, dw_ref) <= DW_REL_L2, (
            f"hidden={hidden} rows={rows}: dw moved at grid / {divisor}"
        )
        assert _workspace_clean(workspace, hidden, residual), (
            f"the workspace is not the kernels' state at grid / {divisor}"
        )
    monkeypatch.setattr(loader, "launch_grid", production)


# Item 2 of issue #7548: a side kernel of one 128-thread CTA per SM, each CTA
# spinning ``ALL_SM_SPIN_MS`` of wall time (``%globaltimer``) after recording its
# SM.  A backward that waits for it takes the rest of that lifetime; one that
# starts beside it costs at most ``MAX_ALL_SM_EXTRA_MS`` more than alone.
ALL_SM_SPIN_MS = 50.0
ALL_SM_THREADS = 128
ALL_SM_PRE_SLEEP_CYCLES = 2_000_000
ALL_SM_TOKENS = (256, 1024, 4096, 16384)
ALL_SM_RUNS = 5
ALL_SM_TRIES = 4
MAX_ALL_SM_EXTRA_MS = 1.0
# cudaSharedmemCarveoutMaxShared
CARVEOUT_MAX_SHARED = 100
# Programs that need a whole SM wait beside a kernel resident on every SM
# whatever its carveout: ``cake_windows.WHOLE_SM`` lists, per architecture and
# route, the token-count ranges where this package runs the deterministic
# package's whole-SM programs (its det-pair windows and forward); the backward
# sections of the docstrings list them with the deterministic facade's remedy
# (``cake_windows.WHOLE_SM_REMEDY``).  A call there is not measured
# (det-inherited): the test checks that the call runs the deterministic
# package's program and that the docstrings list the range, then skips.


def _program_id(name: str) -> str:
    """The program part of a module name (the digits after the package prefix,
    without the ``__<above>_<at_most>`` window suffix of a record that splits a
    det window)."""
    return name.split("__", 1)[0].rsplit("_", 1)[1]


def _skip_det_inherited_whole_sm(
    kind: str, hidden: int, residual: bool, rows: int
) -> None:
    """Skip as not measured (det-inherited) when the call's token count is in a
    ``WHOLE_SM`` range of its route, after checking that the call runs the
    deterministic package's program (program-id equality of the main module) and
    that the docstring section of ``kind`` lists the range with the remedy."""
    arch = loader.arch_for_capability(_capability())
    route = loader.route_key(hidden, kind, residual)
    span = next(
        (
            (above, at_most)
            for above, at_most in windows.WHOLE_SM.get(arch, {}).get(route, ())
            if rows > above and (at_most is None or rows <= at_most)
        ),
        None,
    )
    if span is None:
        return
    det = _det_backend()
    assert det is not None, (
        f"{route} at {rows} rows is listed in WHOLE_SM: checking that it runs the "
        "deterministic package's program needs that package"
    )
    mine = [
        n
        for n in loader.selected_modules(arch, hidden, kind, residual, rows=rows)
        if loader.MODULES[n]["role"] != "bwd_reduce"
    ]
    theirs = [
        n
        for n in det.cake_jit.selected_modules(arch, hidden, kind, residual, rows=rows)
        if det.cake_jit.MODULES[n]["role"] != "bwd_reduce"
    ]
    assert [_program_id(n) for n in mine] == [_program_id(n) for n in theirs], (
        f"{route} at {rows} rows is listed in WHOLE_SM but runs {mine}, not the "
        f"deterministic package's {theirs}"
    )
    section = " ".join(windows.SECTIONS[kind].split())
    cells = ", ".join(
        f"({low}, {'inf)' if high is None else f'{high}]'}"
        for low, high in windows.WHOLE_SM[arch][route]
    )
    assert windows.WHOLE_SM_REMEDY in section, "the remedy is not documented"
    assert f"* ``{arch}`` ``{route}``: {cells}" in section, (
        f"the {kind} docstring section does not list {arch} {route} {cells}"
    )
    print(
        f"ALL_SM_NOT_MEASURED det-inherited {arch} {route} rows={rows} "
        f"range={span} program={_program_id(mine[0])}"
    )
    pytest.skip(
        f"not measured (det-inherited): {arch} {route} at {rows} rows runs the "
        f"deterministic package's whole-SM program {_program_id(mine[0])} "
        f"(WHOLE_SM {span})"
    )


_SPIN_KERNEL = "cake_rmsnorm_train_stall_spin"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_train_stall_spin(unsigned long long ns, int* smids) {
    // The launch's dynamic shared memory (possibly none) is held for the CTA's
    // lifetime without being touched.
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids[blockIdx.x] = (int)smid;
    }
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}
"""


def _checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA error {error}")
    return values[0] if values else None


class _AllSmSpin:
    """A side kernel of one CTA per SM (NVRTC + driver API, so the preferred
    shared-memory carveout can be set on it), holding ``smem_bytes`` of dynamic
    shared memory per CTA."""

    def __init__(self, carveout: int, smem_bytes: int = 0) -> None:
        from cuda.bindings import driver, nvrtc

        self._driver = driver
        device = _device()
        torch.empty(1, device=device)  # primary context
        major, minor = _capability()
        prog = _checked(
            nvrtc.nvrtcCreateProgram(_SPIN_SOURCE.encode(), b"spin.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        _checked(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        size = _checked(nvrtc.nvrtcGetCUBINSize(prog))
        cubin = b" " * size
        _checked(nvrtc.nvrtcGetCUBIN(prog, cubin))
        _checked(nvrtc.nvrtcDestroyProgram(prog))
        ctx = _checked(driver.cuDevicePrimaryCtxRetain(device.index or 0))
        _checked(driver.cuCtxSetCurrent(ctx))
        self._module = _checked(driver.cuModuleLoadData(cubin))
        self._func = _checked(
            driver.cuModuleGetFunction(self._module, _SPIN_KERNEL.encode())
        )
        _checked(
            driver.cuFuncSetAttribute(
                self._func,
                driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                carveout,
            )
        )
        self.smem_bytes = int(smem_bytes)
        if self.smem_bytes:
            _checked(
                driver.cuFuncSetAttribute(
                    self._func,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    self.smem_bytes,
                )
            )
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.smids = torch.full((self.sms,), -1, dtype=torch.int32, device=device)

    def launch(self, stream: torch.cuda.Stream) -> None:
        self.smids.fill_(-1)
        params = (
            (int(ALL_SM_SPIN_MS * 1e6), self.smids.data_ptr()),
            (ctypes.c_ulonglong, ctypes.c_void_p),
        )
        _checked(
            self._driver.cuLaunchKernel(
                self._func,
                self.sms,
                1,
                1,
                ALL_SM_THREADS,
                1,
                1,
                self.smem_bytes,
                self._driver.CUstream(stream.cuda_stream),
                params,
                0,
            )
        )

    def coverage(self) -> int:
        """Distinct SMs the last launch held (read after a synchronize)."""
        return int(torch.unique(self.smids[self.smids >= 0]).numel())


def _beside_all_sms(call, spin: _AllSmSpin, side: torch.cuda.Stream):
    """``call`` launched while ``spin`` holds every SM; ``None`` when the side
    kernel did not reach every SM in ``ALL_SM_TRIES`` launches."""
    for _ in range(ALL_SM_TRIES):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize()
        with torch.cuda.stream(side):
            spin.launch(side)
        torch.cuda._sleep(ALL_SM_PRE_SLEEP_CYCLES)
        start.record()
        out = call()
        end.record()
        torch.cuda.synchronize()
        if spin.coverage() == spin.sms:
            return start.elapsed_time(end), out
    return None


def _alone(call):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    torch.cuda._sleep(ALL_SM_PRE_SLEEP_CYCLES)
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _all_sm_medians(call, carveout: int, smem_bytes: int = 0):
    spin = _AllSmSpin(carveout, smem_bytes)
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _alone(call)
    alone, busy = [], []
    for _ in range(ALL_SM_RUNS):
        alone.append(_alone(call))
        timed = _beside_all_sms(call, spin, side)
        if timed is None:
            pytest.skip(
                f"the side kernel did not hold all {spin.sms} SMs in {ALL_SM_TRIES} launches"
            )
        busy.append(timed)
    return (
        statistics.median(t for t, _ in alone),
        statistics.median(t for t, _ in busy),
        alone,
        busy,
    )


@pytest.mark.parametrize("rows", ALL_SM_TOKENS)
@pytest.mark.parametrize(
    "hidden,residual", [r for r in ROUTES if not r[1]], ids=lambda v: str(v)
)
def test_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    hidden: int, residual: bool, rows: int
) -> None:
    _skip_unless_route(hidden, residual)
    _skip_det_inherited_whole_sm("bwd", hidden, residual, rows)
    x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
    workspace = workspace_for(rows, hidden, _device(), residual=residual)

    def call():
        return tuple(t.clone() for t in _backward(x, w, g, g_h, rstd, workspace))

    alone_ms, busy_ms, alone, busy = _all_sm_medians(call, CARVEOUT_MAX_SHARED)
    print(
        f"ALL_SM hidden={hidden} rows={rows} carveout=max_shared "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    dx_ref, dw_ref = alone[0][1]
    for _t, (dx, dw) in alone[1:] + busy:
        assert torch.equal(dx, dx_ref), (
            f"hidden={hidden} rows={rows}: dx differs beside the all-SM kernel"
        )
        assert _rel_l2(dw, dw_ref) <= DW_REL_L2, (
            f"hidden={hidden} rows={rows}: dw moved beside the all-SM kernel"
        )
    assert _workspace_clean(workspace, hidden, residual)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"hidden={hidden} rows={rows}: backward alone {alone_ms:.4f} ms, beside an all-SM "
        f"kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more; the side kernel lives {ALL_SM_SPIN_MS} ms)"
    )


# R3 C3: side kernels holding shared memory on every SM (one CTA per SM, the
# max-shared carveout), KB of dynamic shared memory per CTA: the sizes of the
# package's measured list (``cake_windows.SMEM_SIDE_KB`` / ``SMEM_WAITS``).


def _instances() -> list[tuple[str, int, bool, str, int, int | None, int, str]]:
    """``(route, hidden, residual, w dtype token, above, at_most, rows, form)``
    of every window of every backward route of the device (``rows`` the middle
    token count of the window), ``form`` "single" or "pair"."""
    arch = loader.arch_for_capability(_capability())
    out = []
    for route, names in loader.ROUTES.get(arch, {}).items():
        if not route.startswith("bwd"):
            continue
        hidden = int(route.rsplit("_h", 1)[1])
        residual = "_residual" in route
        token = "f32" if "_w32" in route else "bf16"
        for name in names:
            record = loader.MODULES[name]
            rule = record["workspace_rule"]
            if rule is None:
                continue  # a dw reduce: launched with its main record
            window = record["select_rule"]
            above = 0 if window is None else int(window["above"])
            at_most = None if window is None else window["at_most"]
            rows = (
                max(above + 1, min(2 * above, 65536))
                if at_most is None
                else (above + 1 + int(at_most)) // 2
            )
            form = "single" if rule["kind"] == "claim_sum_f64" else "pair"
            out.append((route, hidden, residual, token, above, at_most, rows, form))
    return sorted(set(out), key=lambda item: (item[0], item[4]))


def _det_backend():
    try:
        return cake_package("cake_rmsnorm_det_train")
    except Exception:  # noqa: BLE001 - a vendored run has no det package
        return None


def _det_serves(det, hidden: int, residual: bool) -> bool:
    return det is not None and det.cake_jit.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind="bwd",
        residual=residual,
    )


def _waits_beside(call, smem_kb: int) -> tuple[bool, float]:
    """Whether ``call`` waits for an all-SM kernel holding ``smem_kb`` KB per
    CTA, and its extra time (ms) beside it over alone."""
    alone_ms, busy_ms, _alone, _busy = _all_sm_medians(
        call, CARVEOUT_MAX_SHARED, smem_kb * 1024
    )
    extra = busy_ms - alone_ms
    return extra > MAX_ALL_SM_EXTRA_MS, extra


def _waits_below(measured: int | None, listed: int | None) -> bool:
    """Whether a window measured to wait from ``measured`` KB waits beside a
    smaller side allocation than its ``listed`` level allows (None: no wait at
    any measured size)."""
    return measured is not None and (listed is None or measured < listed)


def test_backward_beside_an_all_sm_kernel_holding_shared_memory() -> None:
    """R3 C3 (see the module docstring): per shipped backward window, the
    smallest of ``SMEM_SIDE_KB`` beside which the call waits (extra time above
    ``MAX_ALL_SM_EXTRA_MS``; the side kernel lives ``ALL_SM_SPIN_MS``) is not
    below the package's documented ``SMEM_WAITS`` entry (the most restrictive
    level over the architecture's measured boards; a board that waits only at a
    larger size prints ``SMEM_WAITS_LOOSER``), and every window of the device's
    architecture has one.  The deterministic program the window replaces (for an
    FP32-weight twin: its BF16 owner's, called with the BF16 weight; where the
    det package serves the call) is measured beside it for reference."""
    optin = torch.cuda.get_device_properties(_device()).shared_memory_per_block_optin
    sizes = tuple(kb for kb in windows.SMEM_SIDE_KB if kb * 1024 <= optin)
    if not sizes:
        pytest.skip(f"no side size of {windows.SMEM_SIDE_KB} KB fits {optin} B per CTA")
    arch = loader.arch_for_capability(_capability())
    listed = {
        (route, above, at_most): kb
        for route, spans in windows.SMEM_WAITS.get(arch, {}).items()
        for above, at_most, kb in spans
    }
    det = _det_backend()
    instances = _instances()
    assert instances
    report, mismatched, looser, unmeasured, regressions = [], [], [], [], []
    for route, hidden, residual, token, above, at_most, rows, form in instances:
        x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
        if token == "f32":
            w = w.float()
        workspace = workspace_for(rows, hidden, _device(), residual=residual)

        def call():
            return tuple(t.clone() for t in _backward(x, w, g, g_h, rstd, workspace))

        level, extras = None, []
        for kb in sizes:
            waited, extra = _waits_beside(call, kb)
            extras.append(extra)
            if waited and level is None:
                level = kb
        assert _workspace_clean(workspace, hidden, residual)
        det_level, det_extras = None, []
        if _det_serves(det, hidden, residual):
            w_det = w.to(torch.bfloat16)
            det_ws = det.cake_rmsnorm_det_train_backward_workspace(
                rows, hidden, _device(), residual=residual
            )
            kwargs = {"workspace": det_ws}
            if g_h is not None:
                kwargs["g_residual"] = g_h

            def det_call():
                out = det.cake_rmsnorm_det_train_backward(g, x, w_det, rstd, **kwargs)
                return tuple(t.clone() for t in out)

            for kb in sizes:
                waited, extra = _waits_beside(det_call, kb)
                det_extras.append(extra)
                if waited and det_level is None:
                    det_level = kb
        key = (route, above, at_most)
        if key not in listed:
            unmeasured.append((key, level))
        elif _waits_below(level, listed[key]):
            mismatched.append((key, listed[key], level))
        elif listed[key] != level:
            looser.append((key, listed[key], level))
        if (
            level is not None
            and det_extras
            and (det_level is None or det_level > level)
        ):
            regressions.append((key, level, det_level))
        report.append(
            f"SMEM_SIDE {arch} {route} ({above}, {at_most}] rows={rows} form={form} "
            f"waits_from_kb={level} extra_ms="
            + "/".join(f"{e:.4f}" for e in extras)
            + " det_waits_from_kb="
            + (
                f"{det_level} det_extra_ms=" + "/".join(f"{e:.4f}" for e in det_extras)
                if det_extras
                else "n/a"
            )
        )
    # a leading newline: under pytest -s the progress dots precede the first line
    print("\n" + "\n".join(report))
    for key, listed_kb, level in looser:
        print(
            f"SMEM_WAITS_LOOSER {arch} {key[0]} ({key[1]}, {key[2]}) "
            f"listed={listed_kb} measured={level}"
        )
    for key, level in unmeasured:
        print(f"SMEM_WAITS_MEASURED {arch} {key[0]} ({key[1]}, {key[2]}, {level})")
    print(f"SMEM_REGRESSIONS_VS_DET {regressions}")
    assert not mismatched, (
        f"windows that wait beside less shared memory than their SMEM_WAITS entry "
        f"allows (key, listed KB, measured KB): {mismatched}"
    )
    assert not unmeasured, (
        f"{len(unmeasured)} backward windows of {arch} have no SMEM_WAITS entry "
        f"(measured levels printed as SMEM_WAITS_MEASURED): {unmeasured[:6]}"
    )
