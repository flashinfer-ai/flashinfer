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

Stall regression of the Cake SiLU-gated RMSNorm training forward and backward
beside a kernel resident on another stream (the concurrent-launch contract of
issues #7038 and #7548).

The forward runs one of several fixed-order modules, chosen from the row count
and the SM count: up to 12 rows per SM on SM100 and SM103 (59 on SM107) the
short forward, one 256-thread CTA per row block (fewer CTAs than SMs on SM100
and SM103; on SM107 with 205 SMs or more, as with 212 or 216, at most one wave
of 4 CTAs per SM on all SMs but 16); above that the plain forward, one CTA per
row block; from 976 rows per SM on SM100 and SM103 (800 on SM107) the mid
forward, whole waves of 6 CTAs per SM that each walk at most 2 row blocks; from
2400 rows per SM on SM100 and SM103 the long forward and from 3000 on SM103
(1740 on SM107) the extra-long forward, whole waves of 3 CTAs per SM that each
walk at most 6.  The backward's main kernel launches a grid of independent
CTAs, one per chunk of row blocks, and the weight-gradient reduction a fixed
grid of 128 CTAs.  No kernel waits for another CTA or grid, so none needs the
whole device.  Beside a
long-resident one-thread kernel on another stream each call must keep its
speed: per token count, the median of seven event-timed calls with the side
kernel resident is at most ``MAX_RATIO`` times the median of seven calls
alone, or at most ``MAX_EXTRA_MS`` above it.  The alone and busy calls
alternate in pairs, so host launch gaps and clock drift hit both arms alike.
The bounds sit above the slowdown a resident kernel causes (the CTAs sharing
its SM run a little slower) and below the stall this test guards against, a
wait of milliseconds behind the resident kernel.  ``y``, ``dx``, ``dz`` and
``dw`` are bitwise equal across the alone and busy calls.

The forward beside a kernel resident on every SM (issue #7548, item 2) with
the maximum-shared-memory carveout: the forward at ``ALL_SM_TOKENS`` and the
short forward at its probed rows per SM start beside it and cost at most
``MAX_ALL_SM_EXTRA_MS`` more than alone, far below the side kernel's
``ALL_SM_SPIN_MS`` lifetime.  The forward beside a side kernel that holds ``k``
of the ``N`` SMs (issue #7548, item 1; 1 x 128, 16 x 640 and 32 x 640 side
CTAs): for the long, the extra-long and the mid forward, busy / alone is at
most ``N / (N - k)``, plus the share of one CTA's blocks in the call (a CTA
whose slot the side kernel holds walks all of its blocks once it starts), plus
``PROPORTIONAL_SLACK``, at the token counts that give each CTA slot the fewest
whole row blocks at or above the module's threshold ("first") and the most
below the next module's (or four times "first"; "last").  The short forward at
1 row per SM and at its threshold (on SM107 also at 10 and 30 rows per SM)
keeps its speed beside the one-thread kernel and, beside a side kernel that
holds ``k`` SMs (the item 1 side kernels and 16 CTAs of 1024 threads with the
max-shared carveout), ends no later than the plain forward (the forward module
whatever the rows) beside the same side kernel: the busy median is at most the
plain forward's busy median times ``1 + PROPORTIONAL_SLACK``, or at most
``SHORT_EXTRA_MS`` above it, the two forwards' alone / busy pairs alternating
round by round; a measurement counts only when each forward's alone median is
at most ``SHORT_MAX_ALONE_SPREAD`` times its fastest alone sample, else it is
taken again, at most ``SHORT_EXTRA_ATTEMPTS`` more times.  In these tests every
side CTA records its SM and its start time, a run counts only when the side
kernel held the intended SMs before the call started, and ``y`` is bitwise that
of the first call alone.  The thresholds are the non-deterministic suite's, set
from its measurements of the same forward modules.
"""

from __future__ import annotations

import ctypes
import functools
import statistics

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_det_train")
loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
cake_rmsnorm_silu_gate_det_train_backward = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward
)
cake_rmsnorm_silu_gate_det_train_backward_workspace = (
    pkg.cake_rmsnorm_silu_gate_det_train_backward_workspace
)
cake_rmsnorm_silu_gate_det_train_forward = pkg.cake_rmsnorm_silu_gate_det_train_forward
forward = cake_rmsnorm_silu_gate_det_train_forward
is_cake_rmsnorm_silu_gate_det_train_supported = (
    pkg.is_cake_rmsnorm_silu_gate_det_train_supported
)

D = 128
H = 48
EPS = 1e-6
# The issue's token counts and a small and a large one.
TOKENS = (256, 4096, 30193, 32737, 32388, 65536)
# Side kernel: one thread resident for about 5 ms; the main stream sleeps first
# so the side kernel is resident when the call starts.
SIDE_SLEEP_CYCLES = 10_000_000
PRE_SLEEP_CYCLES = 1_000_000
WARMUP = 3
RUNS = 7
# A row passes on either bound: the busy median over the alone median, or the
# busy median minus the alone median in ms.
MAX_RATIO = 1.5
MAX_EXTRA_MS = 0.020
SEED = 7075

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not is_cake_rmsnorm_silu_gate_det_train_supported(torch.device("cuda")),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


def _inputs(tokens: int):
    generator = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    kwargs = dict(dtype=torch.float32, device="cuda", generator=generator)
    x = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    z = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    dy = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    w = (1 + 0.5 * torch.randn(D, **kwargs)).to(torch.bfloat16)
    return x, z, w, dy


def _timed(call, *, side: torch.cuda.Stream | None):
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
    return start.elapsed_time(end), tuple(t.clone() for t in out)


def _keeps_its_speed(alone_ms: float, busy_ms: float) -> bool:
    return busy_ms / alone_ms <= MAX_RATIO or busy_ms - alone_ms <= MAX_EXTRA_MS


@pytest.mark.parametrize("tokens", TOKENS)
@pytest.mark.parametrize("stage", ["forward", "backward"])
def test_keeps_its_speed_beside_a_resident_kernel(stage: str, tokens: int) -> None:
    x, z, w, dy = _inputs(tokens)
    if stage == "forward":
        y = torch.empty_like(x)

        def call():
            return (cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS, out=y),)

    else:
        workspace = cake_rmsnorm_silu_gate_det_train_backward_workspace(
            tokens * H, x.device
        )

        def call():
            return cake_rmsnorm_silu_gate_det_train_backward(
                dy, x, z, w, EPS, workspace=workspace
            )

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL stage={stage} tokens={tokens} heads={H} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    reference = alone[0][1]
    for _t, out in alone[1:] + busy:
        for got, want in zip(out, reference, strict=True):
            assert torch.equal(got, want), (
                f"{stage} tokens={tokens}: results differ across calls"
            )
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"{stage} tokens={tokens}: {busy_ms:.4f} ms beside a resident kernel vs "
        f"{alone_ms:.4f} ms alone"
    )


# Issue #7548: a side kernel whose CTAs each record their SM and start time
# and spin ``ALL_SM_SPIN_MS`` of wall time (``%globaltimer``).  A call that
# waits for it takes the rest of that lifetime; one that starts beside it costs
# at most ``MAX_ALL_SM_EXTRA_MS`` more than alone.
ALL_SM_SPIN_MS = 50.0
ALL_SM_THREADS = 128
ALL_SM_PRE_SLEEP_CYCLES = 2_000_000
ALL_SM_TOKENS = (256, 4096, 32737)
ALL_SM_RUNS = 5
ALL_SM_TRIES = 4
MAX_ALL_SM_EXTRA_MS = 1.0
# cudaSharedmemCarveoutMaxShared
CARVEOUT_MAX_SHARED = 100
# Item 1: (CTAs, threads) of side kernels that hold some SMs, the issue's
# 1 x 128, 16 x 640 and 32 x 640.
PROPORTIONAL_SIDES = ((1, 128), (16, 640), (32, 640))
PROPORTIONAL_RUNS = 7
# The noise margin of busy / alone over N / (N - k) plus one CTA's share of
# the blocks: the non-deterministic suite's, whose forward tests run the same
# forward modules beside the same side kernels.
PROPORTIONAL_SLACK = 0.02

_SPIN_KERNEL = "cake_rmsnorm_silu_gate_det_train_stall_spin"
_STAMP_KERNEL = "cake_rmsnorm_silu_gate_det_train_stall_stamp"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_silu_gate_det_train_stall_spin(
    unsigned long long ns, int* smids, unsigned long long* starts) {
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    if (threadIdx.x == 0) {
        unsigned int smid;
        asm volatile("mov.u32 %0, %%smid;" : "=r"(smid));
        smids[blockIdx.x] = (int)smid;
        starts[blockIdx.x] = t0;
    }
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    } while (t - t0 < ns);
}

extern "C" __global__ void cake_rmsnorm_silu_gate_det_train_stall_stamp(
    unsigned long long* stamp) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    *stamp = t;
}
"""


def _checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA error {error}")
    return values[0] if values else None


@functools.cache
def _spin_cubin(major: int, minor: int) -> bytes:
    """The side kernels compiled once per architecture (NVRTC)."""
    from cuda.bindings import nvrtc

    prog = _checked(
        nvrtc.nvrtcCreateProgram(_SPIN_SOURCE.encode(), b"spin.cu", 0, [], [])
    )
    opts = [f"--gpu-architecture=sm_{major}{minor}".encode()]
    _checked(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    size = _checked(nvrtc.nvrtcGetCUBINSize(prog))
    cubin = b" " * size
    _checked(nvrtc.nvrtcGetCUBIN(prog, cubin))
    _checked(nvrtc.nvrtcDestroyProgram(prog))
    return cubin


class _Spin:
    """A side kernel of ``ctas`` CTAs (default: one per SM) of ``threads``
    threads (driver API, so the preferred shared-memory carveout can be set on
    it; ``None`` keeps the default), and a one-thread kernel that stamps the
    time on the main stream right before a timed call."""

    def __init__(
        self,
        carveout: int | None,
        *,
        ctas: int | None = None,
        threads: int = ALL_SM_THREADS,
    ) -> None:
        from cuda.bindings import driver

        self._driver = driver
        device = torch.device("cuda", torch.cuda.current_device())
        torch.empty(1, device=device)  # primary context
        cubin = _spin_cubin(*torch.cuda.get_device_capability(device))
        ctx = _checked(driver.cuDevicePrimaryCtxRetain(device.index))
        _checked(driver.cuCtxSetCurrent(ctx))
        self._module = _checked(driver.cuModuleLoadData(cubin))
        self._func = _checked(
            driver.cuModuleGetFunction(self._module, _SPIN_KERNEL.encode())
        )
        self._stamp_func = _checked(
            driver.cuModuleGetFunction(self._module, _STAMP_KERNEL.encode())
        )
        if carveout is not None:
            _checked(
                driver.cuFuncSetAttribute(
                    self._func,
                    driver.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT,
                    carveout,
                )
            )
        self.sms = torch.cuda.get_device_properties(device).multi_processor_count
        self.ctas = self.sms if ctas is None else ctas
        self.threads = threads
        self.smids = torch.full((self.ctas,), -1, dtype=torch.int32, device=device)
        self.starts = torch.zeros(self.ctas, dtype=torch.int64, device=device)
        self.stamp_ns = torch.zeros(1, dtype=torch.int64, device=device)

    def _launch(self, func, ctas, threads, stream, args, types) -> None:
        _checked(
            self._driver.cuLaunchKernel(
                func,
                ctas,
                1,
                1,
                threads,
                1,
                1,
                0,
                self._driver.CUstream(stream.cuda_stream),
                (args, types),
                0,
            )
        )

    def launch(self, stream: torch.cuda.Stream) -> None:
        self.smids.fill_(-1)
        self.starts.fill_(torch.iinfo(torch.int64).max)
        self._launch(
            self._func,
            self.ctas,
            self.threads,
            stream,
            (
                int(ALL_SM_SPIN_MS * 1e6),
                self.smids.data_ptr(),
                self.starts.data_ptr(),
            ),
            (ctypes.c_ulonglong, ctypes.c_void_p, ctypes.c_void_p),
        )

    def stamp(self, stream: torch.cuda.Stream) -> None:
        """Record the time on ``stream`` (enqueued right before the call)."""
        self.stamp_ns.zero_()
        self._launch(
            self._stamp_func,
            1,
            1,
            stream,
            (self.stamp_ns.data_ptr(),),
            (ctypes.c_void_p,),
        )

    def held(self) -> bool:
        """The last launch's CTAs ran on ``ctas`` distinct SMs and had all
        started before the last stamp (read after a synchronize)."""
        coverage = int(torch.unique(self.smids[self.smids >= 0]).numel())
        return coverage == self.ctas and int(self.starts.max()) < int(self.stamp_ns)


# One side kernel per (device, carveout, CTAs, threads), loaded once per process.
_SPINS: dict[tuple[int, int | None, int | None, int], _Spin] = {}


def _spin(
    carveout: int | None, *, ctas: int | None = None, threads: int = ALL_SM_THREADS
) -> _Spin:
    key = (torch.cuda.current_device(), carveout, ctas, threads)
    if key not in _SPINS:
        _SPINS[key] = _Spin(carveout, ctas=ctas, threads=threads)
    return _SPINS[key]


def _spin_timed(call, spin: _Spin | None = None, side=None):
    """``(ms, cloned outputs)`` of ``call`` event-timed on the current stream,
    launched after ``spin`` started on ``side`` when given."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    torch.cuda.synchronize()
    if spin is not None:
        with torch.cuda.stream(side):
            spin.launch(side)
    torch.cuda._sleep(ALL_SM_PRE_SLEEP_CYCLES)
    if spin is not None:
        spin.stamp(torch.cuda.current_stream())
    start.record()
    out = call()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end), tuple(t.clone() for t in out)


def _beside(call, spin: _Spin, side: torch.cuda.Stream):
    """``call`` timed while ``spin`` holds its ``ctas`` distinct SMs from
    before the call; ``None`` when the side kernel did not in
    ``ALL_SM_TRIES`` launches."""
    for _ in range(ALL_SM_TRIES):
        timed = _spin_timed(call, spin, side)
        if spin.held():
            return timed
    return None


def _paired_medians(call, spin: _Spin, runs: int):
    """Medians of ``runs`` alone / beside-``spin`` pairs, and the pairs."""
    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _spin_timed(call)
    alone, busy = [], []
    for _ in range(runs):
        alone.append(_spin_timed(call))
        timed = _beside(call, spin, side)
        if timed is None:
            pytest.skip(
                f"the side kernel did not hold {spin.ctas} distinct SMs before "
                f"the call in {ALL_SM_TRIES} launches"
            )
        busy.append(timed)
    return (
        statistics.median(t for t, _ in alone),
        statistics.median(t for t, _ in busy),
        alone,
        busy,
    )


def _interleaved_paired_medians(calls, spin: _Spin, runs: int):
    """Per call of ``calls``, the medians of ``runs`` alone / beside-``spin``
    pairs and the pairs, the calls' pairs taken round by round (their order
    alternating)."""
    side = torch.cuda.Stream()
    for call in calls:
        for _ in range(WARMUP):
            _spin_timed(call)
    alone = [[] for _ in calls]
    busy = [[] for _ in calls]
    for r in range(runs):
        order = range(len(calls)) if r % 2 == 0 else reversed(range(len(calls)))
        for i in order:
            alone[i].append(_spin_timed(calls[i]))
            timed = _beside(calls[i], spin, side)
            if timed is None:
                pytest.skip(
                    f"the side kernel did not hold {spin.ctas} distinct SMs "
                    f"before the call in {ALL_SM_TRIES} launches"
                )
            busy[i].append(timed)
    return [
        (
            statistics.median(t for t, _ in alone[i]),
            statistics.median(t for t, _ in busy[i]),
            alone[i],
            busy[i],
        )
        for i in range(len(calls))
    ]


def _check_outputs(stage: str, tokens: int, alone, busy) -> None:
    """``y`` of every forward call bitwise that of the first call alone."""
    assert stage == "forward", stage
    first = alone[0][1]
    for _t, out in alone + busy:
        assert torch.equal(out[0], first[0]), f"y tokens={tokens}"


def _plain(arch, role, rows, sm_count):
    return loader.route_module(arch, role)


def _plain_forward(x, z, w, *, out):
    """The forward on the plain forward module whatever the rows (the routing
    before the routed forwards shipped): the short forward's reference."""
    routed = loader.module_for_rows
    loader.module_for_rows = _plain
    try:
        return forward(x, z, w, EPS, out=out)
    finally:
        loader.module_for_rows = routed


@pytest.mark.parametrize("tokens", ALL_SM_TOKENS)
def test_forward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    tokens: int,
) -> None:
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=forward tokens={tokens} heads={H} carveout=max_shared "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: forward alone {alone_ms:.4f} ms, beside an all-SM kernel "
        f"with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


# The long and mid forwards beside the item 1 side kernels: the long forward's
# CTA slots (3 per SM) get row blocks of 32 rows, the mid forward's (6 per SM)
# row blocks of 16 rows, at the fewest whole blocks per slot at or above the
# module's rows per SM on the device ("first": 25 and 11 on SM100 and SM103, 19
# and 9 on SM107) and at four times that for the long forward, the most below
# the long forward's rows per SM for the mid forward ("last": 100 and 24, 76
# and 18).
FORWARD_EDGES = ("first", "last")


def _persistent_forward_tokens(role: str, edge: str) -> tuple[int, int, float]:
    """``(blocks_per_slot, T, cta_share)``: the row blocks per CTA slot of the
    forward module of ``role`` (``fwd_xlong``, ``fwd_long`` or ``fwd_mid``)
    at ``edge`` of its rows, the fewest tokens ``T`` whose ``T * H`` rows give
    each slot that many, and the share of one CTA's blocks in the call alone
    (blocks a CTA walks over blocks a slot runs); skips where the architecture
    routes no such module."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    routes = loader.ROUTES[arch]
    name = routes.get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    # rows per SM that give each slot of the SM one more row block
    step = int(rule["ctas_per_sm"]) * rows_per_block
    blocks_per_slot = -(-loader.min_rows_per_sm(name, arch) // step)
    # the "last" edge: the most below the next tier's start where the
    # architecture routes one (the mid forward below the long one, or on
    # sm_107a, which routes no long forward, below the extra-long one; on
    # sm_103a the long forward below the extra-long one), else four times
    # "first"
    uppers = {"fwd_mid": ("fwd_long", "fwd_xlong"), "fwd_long": ("fwd_xlong",)}
    upper = next((u for u in uppers.get(role, ()) if u in routes), None)
    if edge == "last" and upper is not None:
        upper_start = loader.min_rows_per_sm(routes[upper], arch)
        blocks_per_slot = (upper_start - 1) // step
    elif edge == "last":
        blocks_per_slot *= 4
    tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    assert loader.module_for_rows(arch, "fwd", rows, sms) == name
    (ctas, _, _) = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sms)
    blocks = -(-rows // rows_per_block)
    return blocks_per_slot, tokens, -(-blocks // ctas) * slots / blocks


def _forward_slows_in_proportion(role: str, edge: str, ctas: int, threads: int) -> None:
    blocks_per_slot, tokens, cta_share = _persistent_forward_tokens(role, edge)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    spin = _spin(None, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + cta_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    print(
        f"HELD stage=forward tokens={tokens} heads={H} "
        f"blocks_per_slot={blocks_per_slot} side={ctas}x{threads} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={ratio:.3f} "
        f"ideal={ideal:.3f} cta_share={cta_share:.4f} "
        f"excess={ratio - ideal - cta_share:.4f} role={role}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert ratio <= bound, (
        f"tokens={tokens} ({blocks_per_slot} row blocks per CTA slot): {role} "
        f"alone {alone_ms:.4f} ms, beside {ctas} x {threads} side CTAs "
        f"on {ctas} SMs {busy_ms:.4f} ms: ratio {ratio:.3f} > {bound:.3f} "
        f"(proportional {ideal:.3f} + one CTA's blocks {cta_share:.4f} + "
        f"{PROPORTIONAL_SLACK})"
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
def test_long_forward_slows_in_proportion_to_the_sms_held(
    edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_long", edge, ctas, threads)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
def test_extra_long_forward_slows_in_proportion_to_the_sms_held(
    edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_xlong", edge, ctas, threads)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
def test_mid_forward_slows_in_proportion_to_the_sms_held(
    edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_mid", edge, ctas, threads)


# The short forward: the rows per SM of its token counts on each architecture
# (one row per SM and its threshold, the largest, up to which the architecture
# runs it: 12 on sm_100a and sm_103a, 59 on sm_107a, where 10 and 30 are probed
# as well), and the side kernels of the item 1 test beside it, the issue's three
# with the default carveout and 16 CTAs of 1024 threads with the max-shared
# carveout.
SHORT_FORWARD_ROWS_PER_SM = {
    "sm_100a": (1, 12),
    "sm_103a": (1, 12),
    "sm_107a": (1, 10, 30, 59),
}
SHORT_FORWARD_EDGES = tuple(sorted(set().union(*SHORT_FORWARD_ROWS_PER_SM.values())))
SHORT_FORWARD_SIDES = ((1, 128, None), (16, 640, None), (32, 640, None)) + (
    (16, 1024, CARVEOUT_MAX_SHARED),
)
SHORT_EXTRA_MS = 0.0015
# A held measurement counts when each forward's alone median is at most this
# many times its fastest alone sample in that measurement; otherwise it is taken
# again, at most SHORT_EXTRA_ATTEMPTS more times.  The largest median-to-fastest
# ratio was 1.43 in 170 held measurements on a GB300 node running only this
# test file (20 runs on 2 GPUs), 1.43 in 80 more on that node with pytest,
# pre-commit and compute-sanitizer work on its other GPUs (10 runs), and 1.38
# in 160 on a B200 node (20 runs).  With compile and compute-sanitizer work on
# the other GPUs of a second GB300 node, one measurement had alone medians of
# 0.0103 and 0.0113 ms and failed the bound.
SHORT_MAX_ALONE_SPREAD = 1.7
SHORT_EXTRA_ATTEMPTS = 2


def _short_forward_tokens(rows_per_sm: int) -> int:
    """The most tokens ``T`` (at least one) whose ``T * H`` rows stay within
    ``rows_per_sm`` rows per SM, one of the architecture's
    ``SHORT_FORWARD_ROWS_PER_SM`` (the largest its threshold), where the
    architecture routes the short forward (skips elsewhere)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get("fwd_short")
    if name is None:
        pytest.skip(f"{arch} routes no fwd_short module")
    if rows_per_sm not in SHORT_FORWARD_ROWS_PER_SM[arch]:
        pytest.skip(f"{arch} does not probe {rows_per_sm} rows per SM")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    assert loader.max_rows_per_sm(name, arch) == max(SHORT_FORWARD_ROWS_PER_SM[arch])
    tokens = max(1, rows_per_sm * sms // H)
    assert loader.module_for_rows(arch, "fwd", tokens * H, sms) == name
    return tokens


@pytest.mark.parametrize("rows_per_sm", SHORT_FORWARD_EDGES)
def test_short_forward_keeps_its_speed_beside_a_resident_kernel(
    rows_per_sm: int,
) -> None:
    tokens = _short_forward_tokens(rows_per_sm)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    side = torch.cuda.Stream()
    for _ in range(WARMUP):
        _timed(call, side=None)
    alone, busy = [], []
    for _ in range(RUNS):
        alone.append(_timed(call, side=None))
        busy.append(_timed(call, side=side))
    alone_ms = statistics.median(t for t, _ in alone)
    busy_ms = statistics.median(t for t, _ in busy)
    print(
        f"STALL stage=short_forward tokens={tokens} heads={H} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"short forward tokens={tokens}: {busy_ms:.4f} ms beside a resident "
        f"kernel vs {alone_ms:.4f} ms alone"
    )


@pytest.mark.parametrize("rows_per_sm", SHORT_FORWARD_EDGES)
def test_short_forward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    rows_per_sm: int,
) -> None:
    tokens = _short_forward_tokens(rows_per_sm)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=short_forward tokens={tokens} heads={H} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: short forward alone {alone_ms:.4f} ms, beside an "
        f"all-SM kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


@pytest.mark.parametrize(
    "ctas,threads,carveout",
    SHORT_FORWARD_SIDES,
    ids=["1x128", "16x640", "32x640", "16x1024-max_shared"],
)
@pytest.mark.parametrize("rows_per_sm", SHORT_FORWARD_EDGES)
def test_short_forward_ends_no_later_than_the_forward_beside_held_sms(
    rows_per_sm: int, ctas: int, threads: int, carveout: int | None
) -> None:
    tokens = _short_forward_tokens(rows_per_sm)
    x, z, w, _dy = _inputs(tokens)
    y = torch.empty_like(x)
    y_plain = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    def call_plain():
        return (_plain_forward(x, z, w, out=y_plain),)

    spin = _spin(carveout, ctas=ctas, threads=threads)
    side = f"{ctas}x{threads}" + ("s" if carveout is not None else "")
    attempts = 1 + SHORT_EXTRA_ATTEMPTS
    for attempt in range(1, attempts + 1):
        short_medians, plain_medians = _interleaved_paired_medians(
            (call, call_plain), spin, PROPORTIONAL_RUNS
        )
        alone_ms, busy_ms, alone, busy = short_medians
        plain_alone_ms, plain_busy_ms, plain_alone, plain_busy = plain_medians
        alone_min_ms = min(t for t, _ in alone)
        plain_alone_min_ms = min(t for t, _ in plain_alone)
        spread = max(alone_ms / alone_min_ms, plain_alone_ms / plain_alone_min_ms)
        counts = spread <= SHORT_MAX_ALONE_SPREAD
        print(
            f"HELD stage=short_forward tokens={tokens} heads={H} side={side} "
            f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
            f"plain_alone_ms={plain_alone_ms:.4f} plain_busy_ms={plain_busy_ms:.4f} "
            f"busy_vs_plain={busy_ms / plain_busy_ms:.3f} "
            f"alone_min_ms={alone_min_ms:.4f} plain_alone_min_ms={plain_alone_min_ms:.4f} "
            f"alone_spread={spread:.3f} attempt={attempt}/{attempts} "
            f"counts={int(counts)}"
        )
        _check_outputs("forward", tokens, alone + plain_alone, busy + plain_busy)
        if counts:
            break
    assert (
        busy_ms <= plain_busy_ms * (1 + PROPORTIONAL_SLACK)
        or busy_ms - plain_busy_ms <= SHORT_EXTRA_MS
    ), (
        f"tokens={tokens}: beside {ctas} x {threads} side CTAs ({side}) the short "
        f"forward took {busy_ms:.4f} ms, the plain forward "
        f"{plain_busy_ms:.4f} ms (at most {1 + PROPORTIONAL_SLACK:.2f} times or "
        f"{SHORT_EXTRA_MS} ms more; measurement {attempt} of at most {attempts}, "
        f"alone spread {spread:.3f}, "
        + ("counted)" if counts else f"none counted: above {SHORT_MAX_ALONE_SPREAD})")
    )
