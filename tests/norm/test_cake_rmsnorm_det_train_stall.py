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

Calls beside a kernel resident on every SM (issue #7548, item 2; the rules
of :mod:`flashinfer.cake_rmsnorm_det_train`): for every program window of every
route of the device's architecture (forward and backward, BF16 and FP32
weights, residual routes included), at the window's middle token count, the
call is timed alone and beside a side kernel of one CTA per SM launched with
the max-shared carveout.  Beside a 128-thread side CTA without shared memory
exactly the windows of ``WHOLE_SM`` wait (cost at least ``MIN_WAIT_MS`` more,
the side kernel lives ``ALL_SM_SPIN_MS``) and every other window starts (at
most ``MAX_ALL_SM_EXTRA_MS`` more).  Beside a side CTA holding
``RESIDENT_SMEM_BYTES`` of dynamic shared memory every window outside
``WHOLE_SM`` still starts, and beside one KB more some window outside it waits.
Every side CTA records its SM, and a run counts only when the side kernel holds
every SM; the outputs of every call are bitwise those of the first alone call.
"""

from __future__ import annotations

import ctypes
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_det_train")
loader = cake_package("cake_rmsnorm_det_train", "cake_jit")
cake_rmsnorm_det_train_backward = pkg.cake_rmsnorm_det_train_backward
cake_rmsnorm_det_train_backward_workspace = (
    pkg.cake_rmsnorm_det_train_backward_workspace
)
cake_rmsnorm_det_train_forward = pkg.cake_rmsnorm_det_train_forward
is_cake_rmsnorm_det_train_supported = pkg.is_cake_rmsnorm_det_train_supported

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


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_rmsnorm_det_train_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_det_train needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _skip_unless_route(hidden: int, residual: bool) -> None:
    if not is_cake_rmsnorm_det_train_supported(_device(), hidden):
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
    _y, rstd = cake_rmsnorm_det_train_forward(x, w, EPS[hidden])
    return x, w, g, g_h, rstd


def _backward(x, w, g, g_h, rstd, workspace):
    kwargs = {"workspace": workspace}
    if g_h is not None:
        kwargs["g_residual"] = g_h
    return cake_rmsnorm_det_train_backward(g, x, w, rstd, **kwargs)


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
    workspace = cake_rmsnorm_det_train_backward_workspace(
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
    workspace = cake_rmsnorm_det_train_backward_workspace(
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


# Item 2 of issue #7548 and the "Beside kernels resident on other streams" rules of
# :mod:`flashinfer.cake_rmsnorm_det_train`: a side kernel of one CTA per SM, each
# CTA spinning ``ALL_SM_SPIN_MS`` of wall time (``%globaltimer``) after recording
# its SM, launched with the max-shared carveout.  A call that waits for it takes
# the rest of that lifetime (at least ``MIN_WAIT_MS`` more than alone); one that
# starts beside it costs at most ``MAX_ALL_SM_EXTRA_MS`` more than alone.
ALL_SM_SPIN_MS = 50.0
ALL_SM_THREADS = 128
ALL_SM_PRE_SLEEP_CYCLES = 2_000_000
ALL_SM_RUNS = 5
ALL_SM_TRIES = 4
MAX_ALL_SM_EXTRA_MS = 1.0
MIN_WAIT_MS = 0.5 * ALL_SM_SPIN_MS
# cudaSharedmemCarveoutMaxShared
CARVEOUT_MAX_SHARED = 100
# The calls that need a whole SM, with BF16 and FP32 weights alike (the facade's
# per-architecture list): arch -> (kind, hidden, residual, above), every window
# of the route above ``above`` tokens.
WHOLE_SM = {
    "sm_100a": (("bwd", 6144, True, 0),),
    "sm_103a": (("bwd", 6144, True, 0),),
    "sm_107a": (
        ("bwd", 6144, True, 0),
        ("bwd", 1536, False, 1024),
        ("bwd", 2048, False, 256),
        ("bwd", 4096, False, 1024),
        ("fwd", 6144, False, 4096),
    ),
}
# The largest dynamic shared memory (bytes) one resident CTA per SM may hold
# while every call outside ``WHOLE_SM`` still starts beside it (the facade's
# 27 KB / 24 KB); one KB more makes some call wait.
RESIDENT_SMEM_BYTES = {
    "sm_100a": 27 * 1024,
    "sm_103a": 27 * 1024,
    "sm_107a": 24 * 1024,
}


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _needs_whole_sm(
    arch: str, kind: str, hidden: int, residual: bool, above: int
) -> bool:
    """Whether the window starting above ``above`` tokens is in ``WHOLE_SM``."""
    return any(
        (kind, hidden, residual) == entry[:3] and above >= entry[3]
        for entry in WHOLE_SM.get(arch, ())
    )


def _windows(arch: str) -> list[tuple[str, int, bool, str, int, int | None, int]]:
    """``(kind, hidden, residual, w dtype, above, at_most, rows)`` of every
    program window of every route on ``arch``, ``rows`` its middle token count."""
    out = []
    for hidden, _eps in WIDTHS:
        for kind in ("fwd", "bwd"):
            for residual in (False, True):
                for w_dtype in loader.weight_dtypes(arch, hidden, kind, residual):
                    for above, at_most, _library in loader.program_windows(
                        arch, hidden, kind, residual, w_dtype
                    ):
                        rows = (
                            max(above + 1, min(2 * above, 65536))
                            if at_most is None
                            else (above + 1 + at_most) // 2
                        )
                        out.append(
                            (kind, hidden, residual, w_dtype, above, at_most, rows)
                        )
    return out


_SPIN_KERNEL = "cake_rmsnorm_det_train_stall_spin"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_det_train_stall_spin(unsigned long long ns, int* smids) {
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


def _all_sm_extra(call, spin: _AllSmSpin) -> float | None:
    """Median time beside ``spin`` minus median time alone (ms); the outputs of
    every call are bitwise the first alone call's.  ``None`` when the side
    kernel did not hold every SM."""
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _alone(call)
    alone, busy = [], []
    for _ in range(ALL_SM_RUNS):
        alone.append(_alone(call))
        timed = _beside_all_sms(call, spin, side)
        if timed is None:
            return None
        busy.append(timed)
    reference = alone[0][1]
    for _t, out in alone[1:] + busy:
        assert all(torch.equal(a, b) for a, b in zip(out, reference, strict=True))
    return statistics.median(t for t, _ in busy) - statistics.median(
        t for t, _ in alone
    )


def _window_call(kind: str, hidden: int, residual: bool, w_dtype: str, rows: int):
    x, w, g, g_h, rstd = _inputs(rows, hidden, residual)
    if w_dtype == "f32":
        w = w.float()
    if kind == "fwd":
        u = g if residual else None

        def call():
            out = (
                cake_rmsnorm_det_train_forward(x, w, EPS[hidden], residual=u)
                if residual
                else cake_rmsnorm_det_train_forward(x, w, EPS[hidden])
            )
            return tuple(t.clone() for t in out)

        return call
    workspace = cake_rmsnorm_det_train_backward_workspace(
        rows, hidden, _device(), residual=residual
    )

    def call():
        return tuple(t.clone() for t in _backward(x, w, g, g_h, rstd, workspace))

    return call


def _beside(spin: _AllSmSpin) -> tuple[list[tuple], list[tuple]]:
    """Every program window of the device's architecture (``_windows``) split
    into the windows whose call waits for ``spin`` and those whose call starts
    beside it."""
    arch = _arch()
    waits, starts = [], []
    for window in _windows(arch):
        kind, hidden, residual, w_dtype, _above, _at_most, rows = window
        extra = _all_sm_extra(_window_call(kind, hidden, residual, w_dtype, rows), spin)
        print(
            f"ALL_SM {arch} side_smem={spin.smem_bytes} window={window} extra_ms={extra}"
        )
        if extra is None:
            pytest.skip(
                f"the side kernel did not hold all {spin.sms} SMs in {ALL_SM_TRIES} launches"
            )
        assert extra >= MIN_WAIT_MS or extra <= MAX_ALL_SM_EXTRA_MS, (
            f"{window}: {extra:.3f} ms more beside the side kernel, neither a start "
            f"(at most {MAX_ALL_SM_EXTRA_MS} ms) nor a wait (at least {MIN_WAIT_MS} ms)"
        )
        (waits if extra >= MIN_WAIT_MS else starts).append(window)
    return waits, starts


def _listed(window: tuple) -> bool:
    kind, hidden, residual, _w_dtype, above, _at_most, _rows = window
    return _needs_whole_sm(_arch(), kind, hidden, residual, above)


def test_calls_beside_an_all_sm_kernel_with_the_max_shared_carveout() -> None:
    """Beside one resident 128-thread CTA per SM without shared memory and with
    the max-shared carveout, exactly the windows of ``WHOLE_SM`` wait; every
    other window of every route starts beside it."""
    waits, starts = _beside(_AllSmSpin(CARVEOUT_MAX_SHARED))
    assert starts
    listed = [window for window in waits + starts if _listed(window)]
    assert sorted(waits) == sorted(listed), (
        f"wait but not in WHOLE_SM: {sorted(set(waits) - set(listed))}; "
        f"in WHOLE_SM but start: {sorted(set(listed) - set(waits))}"
    )


def test_calls_beside_an_all_sm_kernel_holding_shared_memory() -> None:
    """Beside one resident CTA per SM holding ``RESIDENT_SMEM_BYTES`` of
    dynamic shared memory (max-shared carveout) every window outside
    ``WHOLE_SM`` starts; beside one KB more some other window waits."""
    limit = RESIDENT_SMEM_BYTES[_arch()]
    waits, starts = _beside(_AllSmSpin(CARVEOUT_MAX_SHARED, limit))
    assert starts
    assert all(_listed(window) for window in waits), (
        f"wait beside {limit} B: {[w for w in waits if not _listed(w)]}"
    )
    more, _starts = _beside(_AllSmSpin(CARVEOUT_MAX_SHARED, limit + 1024))
    assert any(not _listed(window) for window in more), (
        f"no window outside WHOLE_SM waits beside {limit + 1024} B"
    )
