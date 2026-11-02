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

Co-residence of the Cake BF16 RMSNorm training kernels with the
non-deterministic weight gradient (``deterministic=False``) beside kernels
resident on other streams, for the routes the backend stall suite
(``test_cake_rmsnorm_train_stall.py``: BF16 weights, the all-SM case without the
residual route) does not cover (issues #7038, #7548 item 2, #8066):

* the backward of every FP32-weight route and of the residual route (both
  weight dtypes) beside a one-thread kernel resident on another stream: the
  median of seven event-timed calls with the side kernel resident is at most
  ``MAX_RATIO`` times, or at most ``MAX_EXTRA_MS`` above, the median alone, in
  at least one of ``ATTEMPTS`` repetitions of both medians (other work on a
  shared host or GPU can slow one repetition; a call that waits for the
  resident kernel is slow in every repetition);
* the same backward routes beside a kernel holding one 128-thread CTA on every
  SM with the maximum shared-memory carveout (#7548 item 2's remedy);
* the forward of every route -- every width including H=256 and H=5120, the
  residual route, both weight dtypes -- beside that all-SM kernel.

Beside the all-SM kernel a call starts when it costs at most
``MAX_ALL_SM_EXTRA_MS`` more than alone (the side kernel lives
``ALL_SM_SPIN_MS``).  A call that waits is not a pass: it
is reported as "not measured (det-inherited)" -- a pytest skip -- only when its
window is in the package's whole-SM list (``cake_windows.WHOLE_SM``, the list
the facade documents) and the call runs the deterministic package's program (a
forward or a deterministic-pair backward window whose program id equals the one
the deterministic package selects at the same token count; an FP32-weight twin
through its BF16 owner route, same form, or -- in a deterministic-pair twin
window where the owner runs a single kernel -- through the owner's pair program
that the twin's program twins); the deterministic package's own call
beside the same side kernel is measured and printed for reference.  Every other
call that waits fails: a single kernel, an unlisted window, or a window whose
program is not the deterministic package's.  Outputs are checked across every call:
``dx`` / ``y`` / ``rstd`` bitwise, ``dw`` within ``DW_REL_L2`` of the first call
alone (bitwise for a deterministic pair), the workspace at rest.
"""

from __future__ import annotations

import ctypes
import functools
import statistics
from typing import Optional

import pytest
import torch

from tests.test_helpers import cake_rmsnorm_train_reference as ref
from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_train")
loader = cake_package("cake_rmsnorm_train", "cake_jit")
windows_table = cake_package("cake_rmsnorm_train", "cake_windows")
forward = functools.partial(pkg.cake_rmsnorm_train_forward, deterministic=False)
backward = functools.partial(pkg.cake_rmsnorm_train_backward, deterministic=False)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_train_backward_workspace, deterministic=False
)


def _det_package():
    try:
        return cake_package("cake_rmsnorm_det_train")
    except Exception:  # noqa: BLE001 - a vendored copy may come without it
        return None


det = _det_package()

SUPPORTED_CAPABILITIES = ((10, 0), (10, 3), (10, 7))
HIDDEN_SIZES = (256, 512, 1536, 2048, 4096, 5120, 6144)
# one-thread side kernel (#7038): as the backend stall suite
TOKENS = (256, 1024, 4096, 16384, 65536)
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
ATTEMPTS = 3
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
DW_REL_L2 = 1e-6
SEED = 7548
# all-SM side kernel (#7548 item 2)
ALL_SM_SPIN_MS = 50.0
ALL_SM_THREADS = 128
ALL_SM_PRE_SLEEP_CYCLES = 2_000_000
ALL_SM_TOKENS = (256, 1024, 4096, 16384)
ALL_SM_RUNS = 5
ALL_SM_TRIES = 4
MAX_ALL_SM_EXTRA_MS = 1.0
CARVEOUT_MAX_SHARED = 100  # cudaSharedmemCarveoutMaxShared


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


def _arch() -> str:
    return loader.arch_for_capability(_capability())


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_rmsnorm_train(deterministic=False) needs an SM100 / SM103 / SM107 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    if _supported():
        for spec in loader.specs_for_arch(_arch()):
            spec.build_and_load()
        if det is not None:
            for spec in det.cake_jit.specs_for_arch(_arch()):
                spec.build_and_load()
    yield


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _skip_unless_route(hidden: int, residual: bool, w: str, kind: str) -> None:
    if not loader.route_applies(
        device_capability=_capability(),
        hidden=hidden,
        kind=kind,
        residual=residual,
        w_dtype=w,
    ):
        pytest.skip(
            f"hidden={hidden} residual={residual} w={w} has no {kind} route here"
        )


# --------------------------------------------------------------------------- the whole-SM list


def _whole_sm_ranges() -> dict[tuple[str, str], tuple[tuple[int, Optional[int]], ...]]:
    """The package's whole-SM list, ``cake_windows.WHOLE_SM``, as ``(arch, route
    key) -> ((above, at_most), ...)``: the token-count ranges whose program needs
    a whole SM (the deterministic package's whole-SM programs in this package's
    det-pair windows and forward, FP32-weight twins included)."""
    return {
        (arch, key): tuple(ranges)
        for arch, routes in windows_table.WHOLE_SM.items()
        for key, ranges in routes.items()
    }


WHOLE_SM = _whole_sm_ranges()


def _listed_whole_sm(kind: str, key: str, rows: int) -> Optional[int]:
    """The lower edge of the ``WHOLE_SM`` range of ``key`` that holds ``rows``,
    else ``None``."""
    for above, at_most in WHOLE_SM.get((_arch(), key), ()):
        if rows > above and (at_most is None or rows <= at_most):
            return above
    return None


# --------------------------------------------------------------------------- inputs and calls


def _inputs(rows: int, hidden: int, residual: bool, w: str):
    generator = torch.Generator(device=_device()).manual_seed(
        (SEED * 1_000_003 + rows * 7 + hidden + int(residual) * 3 + (w == "f32"))
        % (2**31 - 1)
    )
    kwargs = dict(dtype=torch.float32, device=_device(), generator=generator)
    x = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16)
    u = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16) if residual else None
    weight = (1 + 0.1 * torch.randn(hidden, **kwargs)).to(ref.W_DTYPES[w])
    g = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16)
    g_h = torch.randn(rows, hidden, **kwargs).to(torch.bfloat16) if residual else None
    outputs = forward(x, weight, ref.EPS[hidden], residual=u)
    normalized, rstd = (outputs[2] if residual else x), outputs[1]
    return x, u, weight, g, g_h, normalized, rstd


def _backward_call(hidden, residual, w, rows):
    """``(call, det_call or None, form, workspace)`` of one backward case: the
    non-det call on its own workspace and, where the det package serves the
    shape, its call (BF16 weight: an FP32-weight twin's owner)."""
    _x, _u, weight, g, g_h, normalized, rstd = _inputs(rows, hidden, residual, w)
    ws = workspace_for(rows, hidden, _device(), residual=residual)

    def call():
        out = backward(g, normalized, weight, rstd, g_residual=g_h, workspace=ws)
        return tuple(t.clone() for t in out)

    det_call = None
    if det is not None and det.cake_jit.route_applies(
        device_capability=_capability(), hidden=hidden, kind="bwd", residual=residual
    ):
        w_det = weight.to(torch.bfloat16)
        det_ws = det.cake_rmsnorm_det_train_backward_workspace(
            rows, hidden, _device(), residual=residual
        )

        def det_call():
            out = det.cake_rmsnorm_det_train_backward(
                g, normalized, w_det, rstd, g_residual=g_h, workspace=det_ws
            )
            return tuple(t.clone() for t in out)

    records = ref.window_records(loader, _arch(), hidden, "bwd", residual, w, rows)
    return call, det_call, ref.form(loader, records), ws


def _forward_call(hidden, residual, w, rows):
    x, u, weight, _g, _g_h, _n, _r = _inputs(rows, hidden, residual, w)
    eps = ref.EPS[hidden]

    def call():
        return tuple(t.clone() for t in forward(x, weight, eps, residual=u))

    det_call = None
    if det is not None and det.cake_jit.route_applies(
        device_capability=_capability(), hidden=hidden, kind="fwd", residual=residual
    ):
        w_det = weight.to(torch.bfloat16)

        def det_call():
            out = det.cake_rmsnorm_det_train_forward(x, w_det, eps, residual=u)
            return tuple(t.clone() for t in out)

    return call, det_call


def _check_outputs(results, kind: str, form: Optional[str], label: str) -> None:
    """Every call's outputs against the first: forward and ``dx`` bitwise, ``dw``
    bitwise for a det pair and within ``DW_REL_L2`` for the single kernel."""
    first = results[0]
    for out in results[1:]:
        if kind == "fwd":
            assert all(torch.equal(a, b) for a, b in zip(out, first, strict=True)), (
                f"{label}: forward outputs differ across calls"
            )
            continue
        assert torch.equal(out[0], first[0]), f"{label}: dx differs across calls"
        if form == "pair":
            assert torch.equal(out[1], first[1]), (
                f"{label}: det-pair dw differs across calls"
            )
        else:
            assert ref.rel_l2(out[1], first[1]) <= DW_REL_L2, (
                f"{label}: dw moved across calls"
            )


# --------------------------------------------------------------------------- one-thread side kernel


def _timed(call, *, side: Optional[torch.cuda.Stream]):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if side is not None:
        with torch.cuda.stream(side):
            torch.cuda._sleep(SIDE_SLEEP_CYCLES)
    torch.cuda._sleep(PRE_SLEEP_CYCLES)
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), out


def _backward_routes():
    """The FP32-weight routes of every width and the residual route (both dtypes)."""
    for hidden in HIDDEN_SIZES:
        yield (hidden, False, "f32")
    yield (6144, True, "bf16")
    yield (6144, True, "f32")


def _route_id(route) -> str:
    hidden, residual, w = route
    return f"{'res-' if residual else ''}h{hidden}-{w}"


@pytest.mark.parametrize("rows", TOKENS)
@pytest.mark.parametrize("route", list(_backward_routes()), ids=_route_id)
def test_backward_keeps_its_speed_beside_a_resident_one_thread_kernel(route, rows: int):
    hidden, residual, w = route
    _skip_unless_route(hidden, residual, w, "bwd")
    call, _det_call, form, ws = _backward_call(hidden, residual, w, rows)
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    label = f"{_route_id(route)} T={rows}"
    measured = []
    for attempt in range(ATTEMPTS):
        alone, busy = [], []
        for _ in range(RUNS):
            alone.append(_timed(call, side=None))
            busy.append(_timed(call, side=side))
        alone_ms = statistics.median(t for t, _ in alone)
        busy_ms = statistics.median(t for t, _ in busy)
        _check_outputs([o for _, o in alone + busy], "bwd", form, label)
        assert ref.workspace_at_rest(loader, ws), f"{label}: workspace not at rest"
        print(
            f"STALL1 {_arch()} {_route_id(route)} T={rows} form={form} attempt={attempt} "
            f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f} "
            f"extra_ms={busy_ms - alone_ms:.4f}"
        )
        measured.append((alone_ms, busy_ms))
        if busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS:
            break
    else:
        cells = "; ".join(f"alone {a:.4f} ms, beside {b:.4f} ms" for a, b in measured)
        raise AssertionError(
            f"{label}: beside a resident one-thread kernel in every one of {ATTEMPTS} "
            f"repetitions ({cells}): ratio above {MAX_RATIO} and more than "
            f"{MAX_EXTRA_MS} ms above the call alone"
        )


# --------------------------------------------------------------------------- all-SM side kernel

_SPIN_KERNEL = "cake_rmsnorm_train_all_sm_spin"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_train_all_sm_spin(unsigned long long ns, int* smids) {
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
    """One 128-thread CTA per SM (NVRTC + driver API) with the preferred
    shared-memory carveout set to the maximum shared memory."""

    def __init__(self) -> None:
        from cuda.bindings import driver, nvrtc

        self._driver = driver
        device = _device()
        torch.empty(1, device=device)
        major, minor = _capability()
        prog = _checked(
            nvrtc.nvrtcCreateProgram(_SPIN_SOURCE.encode(), b"spin.cu", 0, [], [])
        )
        opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
        _checked(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
        cubin = b" " * _checked(nvrtc.nvrtcGetCUBINSize(prog))
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
                CARVEOUT_MAX_SHARED,
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
                0,
                self._driver.CUstream(stream.cuda_stream),
                params,
                0,
            )
        )

    def coverage(self) -> int:
        return int(torch.unique(self.smids[self.smids >= 0]).numel())


@functools.cache
def _spin() -> _AllSmSpin:
    return _AllSmSpin()


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


def _beside_all_sms(call, side: torch.cuda.Stream):
    spin = _spin()
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


def _all_sm(call) -> tuple[float, float, list]:
    """``(alone median ms, busy median ms, every call's outputs)``."""
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _alone(call)
    alone, busy = [], []
    for _ in range(ALL_SM_RUNS):
        alone.append(_alone(call))
        timed = _beside_all_sms(call, side)
        if timed is None:
            pytest.skip(
                f"the side kernel did not hold all {_spin().sms} SMs in {ALL_SM_TRIES} launches"
            )
        busy.append(timed)
    return (
        statistics.median(t for t, _ in alone),
        statistics.median(t for t, _ in busy),
        [o for _, o in alone + busy],
    )


def _program_id(name: str) -> str:
    """The program part of a module name (the hex digits after the package
    prefix, without a ``__<above>_<at_most>`` window suffix)."""
    return name.split("__", 1)[0].rsplit("_", 1)[1]


def _main_module(names, modules) -> str:
    return next(n for n in names if modules[n]["role"] != "bwd_reduce")


def _owner_pair_program(hidden: int, residual: bool, w: str, twin_main: str):
    """The BF16 owner route's pair main program that the FP32-weight twin
    program ``twin_main`` twins: an FP32-weight route selects among the twins of
    its owner's programs, so in a window where both routes run a pair they run
    one program.  Read from the windows that overlap a twin window of the same
    library; ``None`` unless exactly one owner pair program overlaps."""
    library = loader.MODULES[twin_main]["library"]
    routes = loader.ROUTES[_arch()]

    def span(name: str):
        rule = loader.MODULES[name]["select_rule"] or {}
        return rule.get("above", 0), rule.get("at_most")

    def overlap(a, b) -> bool:
        return (a[1] is None or a[1] > b[0]) and (b[1] is None or b[1] > a[0])

    found = {
        _program_id(o)
        for t in routes[loader.route_key(hidden, "bwd", residual, w)]
        if loader.MODULES[t]["library"] == library
        for o in routes[loader.route_key(hidden, "bwd", residual, "bf16")]
        if loader.MODULES[o]["role"] != "bwd_reduce"
        and (loader.MODULES[o]["workspace_rule"] or {}).get("kind") == "chunk_partials"
        and overlap(span(t), span(o))
    }
    return found.pop() if len(found) == 1 else None


def _runs_det_program(kind: str, hidden: int, residual: bool, w: str, rows: int):
    """``(True / False, note)``: the call runs the program the deterministic
    package selects for the same shape and token count (program id equality of
    the main kernels); an FP32-weight twin through its BF16 owner route (the
    det package serves BF16 weights only), whose form it must share -- or, in a
    pair twin window where the owner runs a single kernel, through the owner's
    pair program that the twin's program twins (:func:`_owner_pair_program`)."""
    if det is None:
        return False, "no det package"
    arch = det.cake_jit.arch_for_capability(_capability())
    if det.cake_jit.route_key(hidden, kind, residual) not in det.cake_jit.ROUTES.get(
        arch, {}
    ):
        return False, "the det package does not serve this shape"
    det_main = _main_module(
        list(det.cake_jit.selected_modules(arch, hidden, kind, residual, rows=rows)),
        det.cake_jit.MODULES,
    )
    owner = ref.window_records(loader, _arch(), hidden, kind, residual, "bf16", rows)
    main = _main_module(owner, loader.MODULES)
    same = _program_id(main) == _program_id(det_main)
    note = f"program={_program_id(main)[:8]} det_program={_program_id(det_main)[:8]}"
    if w != "bf16" and kind == "bwd":
        twin = ref.window_records(loader, _arch(), hidden, kind, residual, w, rows)
        twin_form, owner_form = ref.form(loader, twin), ref.form(loader, owner)
        if twin_form == "pair" and owner_form == "single":
            program = _owner_pair_program(
                hidden, residual, w, _main_module(twin, loader.MODULES)
            )
            same = program is not None and program == _program_id(det_main)
            note += (
                f" (the twin's program twins the owner's pair program "
                f"{(program or 'none')[:8]}; owner form single)"
            )
        else:
            same = same and twin_form == owner_form
            note += f" (via the BF16 owner; twin form {twin_form})"
    return same, note


def _judge(
    kind: str,
    hidden: int,
    residual: bool,
    w: str,
    rows: int,
    form: Optional[str],
    extra: float,
    det_call,
):
    """``(verdict, note)``: ``starts`` passes; ``det_inherited`` (listed and the
    det package's program) is reported as not measured; ``waits`` fails (see
    the module docstring)."""
    if extra <= MAX_ALL_SM_EXTRA_MS:
        return "starts", ""
    key = loader.route_key(hidden, kind, residual, w)
    above = _listed_whole_sm(kind, key, rows)
    if kind == "fwd" or form == "pair":
        det_program, note = _runs_det_program(kind, hidden, residual, w, rows)
    else:
        det_program, note = False, "single kernel"
    if det_call is not None:
        det_alone, det_busy, _outs = _all_sm(det_call)
        note += f" det_extra_ms={det_busy - det_alone:.4f}"
    if above is not None and det_program:
        return "det_inherited", f"whole_sm_above={above} {note}"
    return "waits", f"listed={above is not None} det_program={det_program} {note}"


def _report_waiting(verdict: str, label: str, note: str) -> None:
    if verdict == "det_inherited":
        pytest.skip(
            f"not measured (det-inherited): {label} runs the det package's program, "
            f"which needs a whole SM, and the facade's whole-SM list names it: {note}"
        )


@pytest.mark.parametrize("rows", ALL_SM_TOKENS)
@pytest.mark.parametrize("route", list(_backward_routes()), ids=_route_id)
def test_backward_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    route, rows: int
):
    hidden, residual, w = route
    _skip_unless_route(hidden, residual, w, "bwd")
    call, det_call, form, ws = _backward_call(hidden, residual, w, rows)
    alone_ms, busy_ms, outputs = _all_sm(call)
    key = loader.route_key(hidden, "bwd", residual, w)
    label = f"{key} T={rows}"
    _check_outputs(outputs, "bwd", form, label)
    assert ref.workspace_at_rest(loader, ws), f"{label}: workspace not at rest"
    extra = busy_ms - alone_ms
    verdict, note = _judge("bwd", hidden, residual, w, rows, form, extra, det_call)
    print(
        f"ALL_SM {_arch()} {key} T={rows} form={form} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} extra_ms={extra:.4f} verdict={verdict} {note}"
    )
    _report_waiting(verdict, label, note)
    assert verdict == "starts", (
        f"{label} ({form}): alone {alone_ms:.4f} ms, beside an all-SM kernel with the "
        f"max-shared carveout {busy_ms:.4f} ms (at most {MAX_ALL_SM_EXTRA_MS} ms more; the side "
        f"kernel lives {ALL_SM_SPIN_MS} ms): {note}"
    )


def _forward_routes():
    for hidden in HIDDEN_SIZES:
        for w in ("bf16", "f32"):
            yield (hidden, False, w)
    yield (6144, True, "bf16")
    yield (6144, True, "f32")


@pytest.mark.parametrize("rows", ALL_SM_TOKENS)
@pytest.mark.parametrize("route", list(_forward_routes()), ids=_route_id)
def test_forward_beside_an_all_sm_kernel_with_the_max_shared_carveout(route, rows: int):
    hidden, residual, w = route
    _skip_unless_route(hidden, residual, w, "fwd")
    call, det_call = _forward_call(hidden, residual, w, rows)
    alone_ms, busy_ms, outputs = _all_sm(call)
    key = loader.route_key(hidden, "fwd", residual, w)
    label = f"{key} T={rows}"
    _check_outputs(outputs, "fwd", None, label)
    extra = busy_ms - alone_ms
    verdict, note = _judge("fwd", hidden, residual, w, rows, None, extra, det_call)
    program = ref.window_records(loader, _arch(), hidden, "fwd", residual, w, rows)
    print(
        f"ALL_SM_FWD {_arch()} {key} T={rows} program={program[0].rsplit('_', 1)[1][:8]} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} extra_ms={extra:.4f} verdict={verdict} {note}"
    )
    _report_waiting(verdict, label, note)
    assert verdict == "starts", (
        f"{label}: forward alone {alone_ms:.4f} ms, beside an all-SM kernel with the "
        f"max-shared carveout {busy_ms:.4f} ms (at most {MAX_ALL_SM_EXTRA_MS} ms more): {note}"
    )
