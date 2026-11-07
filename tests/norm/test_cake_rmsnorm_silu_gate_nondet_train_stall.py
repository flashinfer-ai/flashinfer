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

Stall regression of the Cake SiLU-gated RMSNorm training forward and the
non-deterministic (``nondet``) backward beside a kernel resident
on another stream (the concurrent-launch contract of issues #7038 and #7548).

The forward runs the program of its architecture's ``FORWARD_BANDS`` band
of rows per SM.  The plain forward launches a grid of independent CTAs, one
64-thread CTA per block of 8 rows.  Up to 48 rows per SM on SM100 and SM103
(59 on SM107) the short forward launches one 256-thread CTA per block of 16
rows (with 64 SMs or more on SM100 and SM103, as with 148 or 152, and 205 or
more on SM107, as with 212 or 216, at most 4 CTAs per SM on all SMs but 16,
which fewer SMs do not keep at the end of the band).  The mid forward launches
whole waves of 6 CTAs per SM that each walk at most 2 blocks of 16 rows; the
long and the extra-long forward launch whole waves of 3 CTAs per SM that each
walk at most 6 blocks of 32 rows.  Their bands: on SM100 the mid forward up to 140 rows
per SM, the long forward up to 400, the extra-long forward up to 1000 and the
mid forward past it; on SM103 the mid forward up to 250, the extra-long
forward up to 550 and the mid forward past it; on SM107 the mid forward up to
200, the extra-long forward up to 520 and the mid forward past it.

The backward cuts the rows into blocks of 8 rows and the blocks into
``C = ceil(blocks / B)`` chunks of ``B = max(1, ceil(blocks / 8192))``
consecutive blocks, and launches ``G = min(C, 8 * SMs)`` CTAs of 128 threads,
at most one wave (8 CTAs per SM).  While ``C <= 8 * SMs`` the static backward
gives every chunk its own CTA.  Past that the claimed backward runs: its CTAs
claim runs of consecutive chunks from the workspace's claim counter, in
halving rounds and then single chunks, each CTA its next run as the last block
of its current run starts, so a CTA that starts late takes fewer chunks (one
that becomes resident after every run was claimed takes none).  Each CTA adds
the FP32 weight-gradient partials of its chunks exactly into 64-bit integer
windows of the workspace and takes a ticket on the arrival word; whichever CTA
arrives last rounds the exact sum once into ``dw``, and no CTA waits for
another.  In the rows of its ``BACKWARD_BANDS`` bands the weight gradient
comes from a small-row backward: one cluster of 16 CTAs, scheduled together
on the SMs of one GPC, whose CTAs each wait for the other CTAs' FP32 partials
of their columns and finish them, with no workspace.  Beside a
long-resident one-thread kernel on another stream each call must keep its
speed: per token count, the median of seven event-timed calls with the side
kernel resident is at most ``MAX_RATIO`` times the median of seven calls alone,
or at most ``MAX_EXTRA_MS`` above it.  The alone and busy calls alternate in
pairs, so host launch gaps and clock drift hit both arms alike.

The backward beside a kernel resident on every SM (issue #7548, item 2): each
CTA of the static and the claimed backward needs 15616 bytes of dynamic shared
memory (the launch record's ``dynamic_smem_bytes``: the two halves of the
8 x 128 FP32 fold buffer, 8192 bytes, the 7 64-bit windows of each of the 128
columns, 7168 bytes, and the run cell, the arrival flag and the 8-entry claim
table, 160 bytes, 15520 bytes rounded up), so a side kernel holding every SM
with a small carveout can delay its start until the side kernel exits (no test
here asserts that delay; documented in
:mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`).  The documented remedy
is checked here: with the side kernel's preferred carveout set to the maximum
shared memory, the backward (BF16 and FP32 weight gradients) and the forward
start beside it and cost at most ``MAX_ALL_SM_EXTRA_MS`` more than alone, far
below the side kernel's ``ALL_SM_SPIN_MS`` lifetime; so does the small-row
backward of a BF16 weight gradient at 1 and 5 tokens (48 and 240 rows, the
program of each one's band), which on SM100 and SM103 also starts beside it
with the default and the maximum-L1 carveout.  On SM107 it does not start
beside the side kernel with such a small carveout until that kernel exits
(observed on R200 for the small-row backwards): there it must end at
most ``MAX_ALL_SM_EXTRA_MS`` after the backward it replaces (the static
backward at the same rows).

The backward beside a side kernel that holds ``k`` of the ``N`` SMs (issue
#7548, item 1): busy / alone is at most the proportional ``N / (N - k)``, plus
the share in the call of one row block of each CTA of the one-wave grid, plus
``PROPORTIONAL_SLACK``.  The token counts give each CTA of that grid
(``8 * N`` CTAs) 16, 32 and 160 row blocks: ``T = 3157``, ``6314`` and
``31573`` on a 148-SM B200, whose rows have more chunks than the grid has
CTAs, so the claimed backward runs all three (there in chunks of 3, 5 and 24
blocks).  The persistent forwards beside the same side kernels: busy / alone
is at most ``N / (N - k)``, plus the share of one CTA's blocks in the call (a
CTA whose slot the side kernel holds walks all of its blocks once it starts),
plus ``PROPORTIONAL_SLACK``, in each band of the module, at the token counts
that give the module's CTA slots the fewest whole row blocks each past the
previous band's end and the most up to its band's end, or four times the
fewest in the last band, which has no end: the mid forward 1 and 1 blocks of
16 rows (96 rows per SM), then 11 and 44, on SM100, 1 and 2, then 6 and 24, on
SM103 and SM107; the long forward 2 and 4 blocks of 32 rows on SM100; the
extra-long forward 5 and 10 on SM100, 3 and 5 on SM103 and SM107.  The short
forward at 1 row per SM and at the end of its band (48 rows per SM on SM100
and SM103, also at 12; 59 on SM107, also at 10 and 30) keeps
its speed beside the one-thread kernel, starts beside the all-SM kernel with
the max-shared carveout, and beside a side kernel that holds ``k`` SMs (the
item 1 side kernels and 16 CTAs of 1024 threads with the max-shared carveout)
ends no later than the forward it replaces (this package's plain forward, the
role's module, at the same rows) beside the same side kernel: the busy median
is at most the plain forward's busy median times ``1 +
PROPORTIONAL_SLACK``, or at most ``SHORT_EXTRA_MS`` above it: one tick of the
event timer (about 1 microsecond on SM103) and a half, the calls taking a few
microseconds.  The two forwards' alone / busy pairs alternate round by round
over the same stretch of time, which does not make a slow stretch of the host
or the GPU fall on both alike: a measurement counts only when each forward's
alone median is at most ``SHORT_MAX_ALONE_SPREAD`` times its fastest alone
sample of that measurement, else it is taken again, at most
``SHORT_EXTRA_ATTEMPTS`` more times, and the bound applies to the first
measurement that counts (to the last one when none does).  The small-row
backward at 1 and 5 tokens (48 and 240 rows) keeps its speed beside the one-thread
kernel and, beside the short forward's side kernels, ends no later than the
backward it replaces (the role's module, the static backward, at the same
rows) by the short forward's bound and measurement rule: there a measurement
counts only when the short backward's and the replaced backward's alone
medians are each at most ``SHORT_MAX_ALONE_SPREAD`` times their fastest alone
sample of that measurement.

In the item 1 and item 2 tests every side CTA records its SM and its start
time, and a run counts only when the side kernel held the intended SMs before
the call started.  In every test ``y``, ``dx``, ``dz`` and ``dw`` are bitwise
those of the first call alone of the same entry point and module (a chunk's
partial does not depend on the grid or on the CTA that claimed it, and the sum
of the partials is exact), every ``dw`` also lies within the backend suite's
budget of an FP64 reference, the small-row backward's ``dx`` and ``dz`` are
bitwise those of the backward it replaces, and the workspace is zero after
every call.
"""

from __future__ import annotations

import ctypes
import functools
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_nondet_train")
loader = cake_package("cake_rmsnorm_silu_gate_nondet_train", "cake_jit")
forward = pkg.cake_rmsnorm_silu_gate_nondet_train_forward
backward = pkg.cake_rmsnorm_silu_gate_nondet_train_backward
workspace_for = pkg.cake_rmsnorm_silu_gate_nondet_train_backward_workspace

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
SEED = 8067

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_rmsnorm_silu_gate_nondet_train_supported(torch.device("cuda")),
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
    workspace = workspace_for(tokens * H, x.device)
    if stage == "forward":
        y = torch.empty_like(x)

        def call():
            return (forward(x, z, w, EPS, out=y),)

    else:

        def call():
            return backward(dy, x, z, w, EPS, workspace=workspace)

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
    _check_outputs(
        stage,
        tokens,
        alone,
        busy,
        None if stage == "forward" else _dw_reference(dy, x, z),
    )
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
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
# 1 x 128, 16 x 640 and 32 x 640, beside the backward at the token counts that
# give each CTA of its one-wave grid (``ctas_per_sm`` CTAs per SM, the CTAs per
# SM its launch bounds hold at once) this many row blocks.
PROPORTIONAL_SIDES = ((1, 128), (16, 640), (32, 640))
PROPORTIONAL_BLOCKS_PER_CTA = (16, 32, 160)
PROPORTIONAL_RUNS = 7
# The margin for timing noise of busy / alone over N / (N - k) plus one row
# block's share, and of the short forward's and the short backward's busy
# medians over those of the calls they are compared with.
PROPORTIONAL_SLACK = 0.02
# The backend suite's weight-gradient budget:
# |dw - dw64| <= |dw64| * 2^-7 + sum|terms| * 2^-20.
DW_REL_TOL = 2.0**-7
DW_MAG_TOL = 2.0**-20

_SPIN_KERNEL = "cake_rmsnorm_silu_gate_train_stall_spin"
_STAMP_KERNEL = "cake_rmsnorm_silu_gate_train_stall_stamp"
_SPIN_SOURCE = r"""
extern "C" __global__ void cake_rmsnorm_silu_gate_train_stall_spin(
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

extern "C" __global__ void cake_rmsnorm_silu_gate_train_stall_stamp(
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
def _spin_image(major: int, minor: int) -> bytes:
    """The side kernels compiled once per architecture (NVRTC): a cubin for
    ``sm_<major><minor>`` when the loaded NVRTC supports it, else PTX for the
    highest ``compute_XY`` it supports below the device's, which the driver
    compiles for the device when it loads the module (an NVRTC older than the
    device, such as the pip NVRTC 13.0 on SM107).  Skips when the NVRTC
    supports no architecture at or below the device's."""
    from cuda.bindings import nvrtc

    device_arch = 10 * major + minor
    supported = [int(arch) for arch in _checked(nvrtc.nvrtcGetSupportedArchs())]
    if device_arch in supported:
        target = f"sm_{device_arch}"
    else:
        lower = [arch for arch in supported if arch <= device_arch]
        if not lower:
            _, nvrtc_major, nvrtc_minor = nvrtc.nvrtcVersion()
            pytest.skip(
                f"NVRTC {nvrtc_major}.{nvrtc_minor} compiles for no architecture "
                f"at or below sm_{device_arch} (it supports {supported}), so "
                "the side kernels cannot be built"
            )
        target = f"compute_{max(lower)}"
    prog = _checked(
        nvrtc.nvrtcCreateProgram(_SPIN_SOURCE.encode(), b"spin.cu", 0, [], [])
    )
    opts = [f"--gpu-architecture={target}".encode()]
    _checked(nvrtc.nvrtcCompileProgram(prog, len(opts), opts))
    if target.startswith("sm_"):
        size = _checked(nvrtc.nvrtcGetCUBINSize(prog))
        image = b" " * size
        _checked(nvrtc.nvrtcGetCUBIN(prog, image))
    else:
        size = _checked(nvrtc.nvrtcGetPTXSize(prog))
        image = b" " * size
        _checked(nvrtc.nvrtcGetPTX(prog, image))
    _checked(nvrtc.nvrtcDestroyProgram(prog))
    return image


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
        image = _spin_image(*torch.cuda.get_device_capability(device))
        ctx = _checked(driver.cuDevicePrimaryCtxRetain(device.index))
        _checked(driver.cuCtxSetCurrent(ctx))
        self._module = _checked(driver.cuModuleLoadData(image))
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


def _dw_reference(dy, x, z):
    """``(dw64, mag)``: the FP64 weight gradient of the BF16 inputs and
    ``sum |a * n|`` per column, the scale of its rounding (the backend suite's
    reference)."""
    xd = x.double()
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    terms = (dy.double() * torch.nn.functional.silu(z.double()) * n).reshape(-1, D)
    return terms.sum(0), terms.abs().sum(0)


def _check_outputs(stage: str, tokens: int, alone, busy, dw_reference=None) -> None:
    """``y`` (forward) or ``dx``, ``dz`` and ``dw`` (backward) of every call
    of one entry point and module bitwise those of its first call alone; every
    backward ``dw`` also within the backend suite's budget of
    ``dw_reference``."""
    first = alone[0][1]
    for _t, out in alone + busy:
        if stage == "forward":
            assert torch.equal(out[0], first[0]), f"y tokens={tokens}"
            continue
        assert torch.equal(out[0], first[0]), f"dx tokens={tokens}"
        assert torch.equal(out[1], first[1]), f"dz tokens={tokens}"
        assert torch.equal(out[2], first[2]), f"dw tokens={tokens} {out[2].dtype}"
        dw64, mag = dw_reference
        err = (out[2].double() - dw64).abs()
        bound = dw64.abs() * DW_REL_TOL + mag * DW_MAG_TOL
        assert bool((err <= bound).all()), (
            f"dw tokens={tokens} {out[2].dtype}: beyond the budget by "
            f"{(err - bound).max().item():.3e}"
        )


def _check_replaced_rows(tokens: int, alone, replaced_alone) -> None:
    """``dx`` and ``dz`` of the short backward's first call alone bitwise
    those of the backward it replaces (the same row arithmetic; their weight
    gradients sum different partials and are each checked by
    :func:`_check_outputs`)."""
    first, replaced = alone[0][1], replaced_alone[0][1]
    assert torch.equal(replaced[0], first[0]), f"dx tokens={tokens} (replaced)"
    assert torch.equal(replaced[1], first[1]), f"dz tokens={tokens} (replaced)"


@pytest.mark.parametrize("tokens", ALL_SM_TOKENS)
@pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
def test_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    dw_dtype: torch.dtype, tokens: int
) -> None:
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=backward tokens={tokens} heads={H} dw={dw_dtype} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy, _dw_reference(dy, x, z))
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: backward alone {alone_ms:.4f} ms, beside an all-SM kernel "
        f"with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more; the side kernel lives {ALL_SM_SPIN_MS} ms)"
    )


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


def _claim_tokens(blocks_per_cta: int) -> tuple[int, float, int, int]:
    """``(T, block_share, blocks_per_chunk, chunks)``: the most tokens ``T``
    whose ``T * H`` rows give each CTA of the BF16 weight-gradient backward's
    one-wave grid (``backward_ctas``: ``ctas_per_sm`` CTAs per SM, all
    resident at once) at most ``blocks_per_cta`` row blocks, rows with more
    chunks than that grid has CTAs, which the claimed backward runs; the share
    of one row block of each of its CTAs in the call alone; and the chunks of
    those rows (``backward_chunks``)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.route_module(arch, "bwd_bf16_mid")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
    rows_per_block = int(loader.MODULES[name]["grid_rule"]["rows_per_block"])
    tokens = blocks_per_cta * wave * rows_per_block // H
    rows = tokens * H
    assert loader.module_for_rows(arch, "bwd_bf16", rows, sms) == name
    ctas = loader.backward_ctas(name, rows, sms)
    assert ctas == wave
    blocks_per_chunk, chunks = loader.backward_chunks(name, rows)
    blocks = -(-rows // rows_per_block)
    return tokens, ctas / blocks, blocks_per_chunk, chunks


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("blocks_per_cta", PROPORTIONAL_BLOCKS_PER_CTA)
def test_backward_slows_in_proportion_to_the_sms_held(
    blocks_per_cta: int, ctas: int, threads: int
) -> None:
    tokens, block_share, blocks_per_chunk, chunks = _claim_tokens(blocks_per_cta)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)

    def call():
        return backward(dy, x, z, w, EPS, workspace=workspace)

    spin = _spin(None, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + block_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    print(
        f"HELD stage=backward tokens={tokens} heads={H} "
        f"blocks_per_cta={blocks_per_cta} blocks_per_chunk={blocks_per_chunk} "
        f"chunks={chunks} side={ctas}x{threads} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={ratio:.3f} "
        f"ideal={ideal:.3f} block_share={block_share:.4f} "
        f"excess={ratio - ideal - block_share:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy, _dw_reference(dy, x, z))
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert ratio <= bound, (
        f"tokens={tokens} ({blocks_per_cta} row blocks per CTA): backward alone "
        f"{alone_ms:.4f} ms, beside {ctas} x {threads} side CTAs on {ctas} SMs "
        f"{busy_ms:.4f} ms: ratio {ratio:.3f} > {bound:.3f} (proportional "
        f"{ideal:.3f} + one row block {block_share:.4f} + {PROPORTIONAL_SLACK})"
    )


# The persistent forwards beside the item 1 side kernels: the CTA slots of the
# long and the extra-long forward (3 per SM) get row blocks of 32 rows, the mid
# forward's (6 per SM) row blocks of 16 rows, in the module's ``nth`` band of
# the architecture's FORWARD_BANDS, at the fewest whole blocks per slot past
# the previous band's end ("first") and at the most up to the band's end, or
# four times "first" in the last band, which has no end ("last"): the mid
# forward 1 and 1, then 11 and 44, on SM100, 1 and 2, then 6 and 24, on SM103
# and SM107; the long forward 2 and 4 on SM100; the extra-long forward 5 and
# 10 on SM100, 3 and 5 on SM103 and SM107.
FORWARD_EDGES = ("first", "last")
FORWARD_BAND_ORDINALS = (1, 2)


def _forward_band(arch: str, role: str, nth: int) -> tuple[int, int | None]:
    """``(first, last)`` rows per SM of the ``nth`` band (from 1) of ``role``
    in the architecture's ``FORWARD_BANDS``: one more than the previous
    band's end, and the band's end (``None``: no end); skips where the
    architecture has fewer bands of ``role``."""
    first, seen = 1, 0
    for last, band in loader.FORWARD_BANDS[arch]:
        if band == role:
            seen += 1
            if seen == nth:
                return first, last
        if last is not None:
            first = int(last) + 1
    pytest.skip(f"{arch} has {seen} forward band(s) of {role}, not {nth}")


def _persistent_forward_tokens(
    role: str, edge: str, nth: int
) -> tuple[int, int, float]:
    """``(blocks_per_slot, T, cta_share)``: the row blocks per CTA slot of the
    forward module of ``role`` (``fwd_xlong``, ``fwd_long`` or ``fwd_mid``)
    at ``edge`` of its ``nth`` band, the fewest tokens ``T`` whose ``T * H``
    rows give each slot that many, and the share of one CTA's blocks in the
    call alone (blocks a CTA walks over blocks a slot runs); skips where the
    architecture routes no such module or band."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    first, last = _forward_band(arch, role, nth)
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    # rows per SM that give each slot of the SM one more row block
    step = int(rule["ctas_per_sm"]) * rows_per_block
    blocks_per_slot = -(-first // step)
    if edge == "last":
        blocks_per_slot = 4 * blocks_per_slot if last is None else last // step
    assert blocks_per_slot >= 1, (arch, role, nth, edge)
    tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    assert loader.module_for_rows(arch, "fwd", rows, sms) == name
    (ctas, _, _) = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sms)
    blocks = -(-rows // rows_per_block)
    return blocks_per_slot, tokens, -(-blocks // ctas) * slots / blocks


def _forward_slows_in_proportion(
    role: str, edge: str, nth: int, ctas: int, threads: int
) -> None:
    blocks_per_slot, tokens, cta_share = _persistent_forward_tokens(role, edge, nth)
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
@pytest.mark.parametrize("nth", FORWARD_BAND_ORDINALS, ids=["band1", "band2"])
def test_long_forward_slows_in_proportion_to_the_sms_held(
    nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_long", edge, nth, ctas, threads)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("nth", FORWARD_BAND_ORDINALS, ids=["band1", "band2"])
def test_extra_long_forward_slows_in_proportion_to_the_sms_held(
    nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_xlong", edge, nth, ctas, threads)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("nth", FORWARD_BAND_ORDINALS, ids=["band1", "band2"])
def test_mid_forward_slows_in_proportion_to_the_sms_held(
    nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_mid", edge, nth, ctas, threads)


# The short forward: the rows per SM of its token counts on each architecture
# (one row per SM and the end of its band, the largest, up to which the
# architecture runs it: 48 on sm_100a and sm_103a, where 12 is probed as well,
# 59 on sm_107a, where 10 and 30 are probed as well), and the side kernels of the item 1 test beside it, the issue's three
# with the default carveout and 16 CTAs of 1024 threads with the max-shared
# carveout.
SHORT_FORWARD_ROWS_PER_SM = {
    "sm_100a": (1, 12, 48),
    "sm_103a": (1, 12, 48),
    "sm_107a": (1, 10, 30, 59),
}
SHORT_FORWARD_EDGES = tuple(sorted(set().union(*SHORT_FORWARD_ROWS_PER_SM.values())))
SHORT_FORWARD_SIDES = ((1, 128, None), (16, 640, None), (32, 640, None)) + (
    (16, 1024, CARVEOUT_MAX_SHARED),
)
SHORT_EXTRA_MS = 0.0015
# A held measurement counts when each compared call's alone median is at most
# this many times its fastest alone sample in that measurement (a slow stretch
# of the host or the GPU moves the median of calls of a few microseconds);
# otherwise it is taken again, at most SHORT_EXTRA_ATTEMPTS more times.
SHORT_MAX_ALONE_SPREAD = 1.7
SHORT_EXTRA_ATTEMPTS = 2


def _short_forward_tokens(rows_per_sm: int) -> int:
    """The most tokens ``T`` (at least one) whose ``T * H`` rows stay within
    ``rows_per_sm`` rows per SM, one of the architecture's
    ``SHORT_FORWARD_ROWS_PER_SM`` (the largest the end of its band), where the
    architecture routes the short forward (skips elsewhere)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get("fwd_short")
    if name is None:
        pytest.skip(f"{arch} routes no fwd_short module")
    if rows_per_sm not in SHORT_FORWARD_ROWS_PER_SM[arch]:
        pytest.skip(f"{arch} does not probe {rows_per_sm} rows per SM")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    (last, role) = loader.FORWARD_BANDS[arch][0]
    assert role == "fwd_short" and last == max(SHORT_FORWARD_ROWS_PER_SM[arch])
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
    plain = loader.route_module(loader.device_arch(x.device.index), "fwd")

    def call():
        return (forward(x, z, w, EPS, out=y),)

    def call_plain():
        # the module the short forward replaces at these rows: this package's
        # plain forward
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "module_for_rows", lambda *_args: plain)
            return (forward(x, z, w, EPS, out=y_plain),)

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
            f"plain_alone_ms={plain_alone_ms:.4f} "
            f"plain_busy_ms={plain_busy_ms:.4f} "
            f"busy_vs_plain={busy_ms / plain_busy_ms:.3f} "
            f"alone_min_ms={alone_min_ms:.4f} "
            f"plain_alone_min_ms={plain_alone_min_ms:.4f} "
            f"alone_spread={spread:.3f} attempt={attempt}/{attempts} "
            f"counts={int(counts)}"
        )
        # each module's y repeats its own bits (two programs may associate a
        # row's sum of squares differently)
        _check_outputs("forward", tokens, alone, busy)
        _check_outputs("forward", tokens, plain_alone, plain_busy)
        if counts:
            break
    assert (
        busy_ms <= plain_busy_ms * (1 + PROPORTIONAL_SLACK)
        or busy_ms - plain_busy_ms <= SHORT_EXTRA_MS
    ), (
        f"tokens={tokens}: beside {ctas} x {threads} side CTAs ({side}) the short "
        f"forward took {busy_ms:.4f} ms, the plain forward it replaces "
        f"{plain_busy_ms:.4f} ms (at most {1 + PROPORTIONAL_SLACK:.2f} times or "
        f"{SHORT_EXTRA_MS} ms more; measurement {attempt} of at most {attempts}, "
        f"alone spread {spread:.3f}, "
        + ("counted)" if counts else f"none counted: above {SHORT_MAX_ALONE_SPREAD})")
    )


# The small-row backward (BF16 weight gradient): one token and five tokens of H
# heads (48 and 240 rows, each the program of its BACKWARD_BANDS band on the
# architecture), beside the short forward's side kernels.  Its held test takes
# the short forward's bound, slack and SHORT_MAX_ALONE_SPREAD.
SHORT_BACKWARD_TOKENS = (1, 5)
SHORT_BACKWARD_SIDES = SHORT_FORWARD_SIDES
# cudaSharedmemCarveoutMaxL1.  The test below checks that the small-row backward
# starts beside a side kernel holding every SM without
# shared memory with this or the default carveout (at most
# MAX_ALL_SM_EXTRA_MS more than alone), or on the architectures of
# SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT that it ends at most
# MAX_ALL_SM_EXTRA_MS after the backward it replaces (the static backward,
# 15616 bytes a CTA) beside that side kernel.
CARVEOUT_MAX_L1 = 0


def _short_backward(tokens: int) -> tuple[str, str]:
    """``(the small-row backward, the backward it replaces)``: the BF16
    weight-gradient modules of ``tokens * H`` rows (the first the module of
    the architecture's ``BACKWARD_BANDS`` band holding them, the second the
    role's module, the static backward); skips where no band holds them."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    name = loader.module_for_rows(arch, "bwd_bf16", tokens * H, sms)
    if loader.MODULES[name]["role"] not in loader.BAND_ROLES:
        pytest.skip(f"{arch} runs no small-row backward for {tokens * H} rows")
    return name, loader.route_module(arch, "bwd_bf16")


@pytest.mark.parametrize("tokens", SHORT_BACKWARD_TOKENS)
def test_short_backward_keeps_its_speed_beside_a_resident_kernel(tokens: int) -> None:
    _short_backward(tokens)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)

    def call():
        return backward(dy, x, z, w, EPS, workspace=workspace)

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
        f"STALL stage=short_backward tokens={tokens} heads={H} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy, _dw_reference(dy, x, z))
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"short backward tokens={tokens}: {busy_ms:.4f} ms beside a resident "
        f"kernel vs {alone_ms:.4f} ms alone"
    )


@pytest.mark.parametrize("tokens", SHORT_BACKWARD_TOKENS)
def test_short_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    tokens: int,
) -> None:
    _short_backward(tokens)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    dw_dtype = torch.bfloat16
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage=short_backward tokens={tokens} heads={H} dw={dw_dtype} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy, _dw_reference(dy, x, z))
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: short backward alone {alone_ms:.4f} ms, beside an "
        f"all-SM kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more; the side kernel lives {ALL_SM_SPIN_MS} ms)"
    )


# Architectures on which the small-row backward (the exact tiny backward,
# 19584 bytes a CTA at 4 teams) does not start beside a side kernel holding
# every SM with a small carveout (the default or the maximum-L1 one) until that
# kernel exits (observed on R200): there it
# must end at most MAX_ALL_SM_EXTRA_MS after the backward it replaces (the
# static backward, 15616 bytes a CTA) beside it.
SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT = ("sm_107a",)


@pytest.mark.parametrize("tokens", SHORT_BACKWARD_TOKENS)
@pytest.mark.parametrize("carveout", [None, CARVEOUT_MAX_L1], ids=["default", "max_l1"])
def test_short_backward_beside_an_all_sm_kernel_with_a_small_carveout(
    carveout: int | None, tokens: int
) -> None:
    _name, replaced = _short_backward(tokens)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)

    def call():
        return backward(dy, x, z, w, EPS, workspace=workspace)

    alone_ms, busy_ms, alone, busy = _paired_medians(call, _spin(carveout), ALL_SM_RUNS)
    label = "default" if carveout is None else "max_l1"
    print(
        f"ALL_SM stage=short_backward tokens={tokens} heads={H} "
        f"dw=torch.bfloat16 carveout={label} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("backward", tokens, alone, busy, _dw_reference(dy, x, z))
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    if loader.device_arch(x.device.index) in SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT:

        def call_replaced():
            # the module the short backward replaces at these rows
            with pytest.MonkeyPatch.context() as patch:
                patch.setattr(pkg, "module_for_rows", lambda *_args: replaced)
                return backward(dy, x, z, w, EPS, workspace=workspace)

        rep_alone_ms, rep_busy_ms, rep_alone, rep_busy = _paired_medians(
            call_replaced, _spin(carveout), ALL_SM_RUNS
        )
        print(
            f"ALL_SM stage=replaced_backward tokens={tokens} heads={H} "
            f"dw=torch.bfloat16 carveout={label} alone_ms={rep_alone_ms:.4f} "
            f"busy_ms={rep_busy_ms:.4f} extra_ms={rep_busy_ms - rep_alone_ms:.4f}"
        )
        _check_outputs("backward", tokens, rep_alone, rep_busy, _dw_reference(dy, x, z))
        _check_replaced_rows(tokens, alone, rep_alone)
        assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
        assert busy_ms - rep_busy_ms <= MAX_ALL_SM_EXTRA_MS, (
            f"tokens={tokens}: beside an all-SM kernel with the {label} carveout "
            f"the short backward took {busy_ms:.4f} ms, the backward it replaces "
            f"{rep_busy_ms:.4f} ms (at most {MAX_ALL_SM_EXTRA_MS} ms more)"
        )
        return
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: short backward alone {alone_ms:.4f} ms, beside an "
        f"all-SM kernel with the {label} carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more; the side kernel lives {ALL_SM_SPIN_MS} ms)"
    )


@pytest.mark.parametrize(
    "ctas,threads,carveout",
    SHORT_BACKWARD_SIDES,
    ids=["1x128", "16x640", "32x640", "16x1024-max_shared"],
)
@pytest.mark.parametrize("tokens", SHORT_BACKWARD_TOKENS)
def test_short_backward_ends_no_later_than_the_backward_beside_held_sms(
    tokens: int, ctas: int, threads: int, carveout: int | None
) -> None:
    _name, replaced = _short_backward(tokens)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)

    def call():
        return backward(dy, x, z, w, EPS, workspace=workspace)

    def call_replaced():
        # the module the short backward replaces at these rows
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(pkg, "module_for_rows", lambda *_args: replaced)
            return backward(dy, x, z, w, EPS, workspace=workspace)

    dw_reference = _dw_reference(dy, x, z)
    spin = _spin(carveout, ctas=ctas, threads=threads)
    side = f"{ctas}x{threads}" + ("s" if carveout is not None else "")
    attempts = 1 + SHORT_EXTRA_ATTEMPTS
    for attempt in range(1, attempts + 1):
        short_medians, replaced_medians = _interleaved_paired_medians(
            (call, call_replaced), spin, PROPORTIONAL_RUNS
        )
        alone_ms, busy_ms, alone, busy = short_medians
        rep_alone_ms, rep_busy_ms, rep_alone, rep_busy = replaced_medians
        alone_min_ms = min(t for t, _ in alone)
        rep_alone_min_ms = min(t for t, _ in rep_alone)
        spread = max(alone_ms / alone_min_ms, rep_alone_ms / rep_alone_min_ms)
        counts = spread <= SHORT_MAX_ALONE_SPREAD
        print(
            f"HELD stage=short_backward tokens={tokens} heads={H} side={side} "
            f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
            f"replaced_alone_ms={rep_alone_ms:.4f} replaced_busy_ms={rep_busy_ms:.4f} "
            f"busy_vs_replaced={busy_ms / rep_busy_ms:.3f} "
            f"alone_min_ms={alone_min_ms:.4f} replaced_alone_min_ms={rep_alone_min_ms:.4f} "
            f"alone_spread={spread:.3f} attempt={attempt}/{attempts} "
            f"counts={int(counts)}"
        )
        # each module's outputs repeat their own bits; dx and dz match across them
        _check_outputs("backward", tokens, alone, busy, dw_reference)
        _check_outputs("backward", tokens, rep_alone, rep_busy, dw_reference)
        _check_replaced_rows(tokens, alone, rep_alone)
        assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
        if counts:
            break
    assert (
        busy_ms <= rep_busy_ms * (1 + PROPORTIONAL_SLACK)
        or busy_ms - rep_busy_ms <= SHORT_EXTRA_MS
    ), (
        f"tokens={tokens}: beside {ctas} x {threads} side CTAs ({side}) the short "
        f"backward took {busy_ms:.4f} ms, the backward it replaces "
        f"{rep_busy_ms:.4f} ms (at most {1 + PROPORTIONAL_SLACK:.2f} times or "
        f"{SHORT_EXTRA_MS} ms more; measurement {attempt} of at most {attempts}, "
        f"alone spread {spread:.3f}, "
        + ("counted)" if counts else f"none counted: above {SHORT_MAX_ALONE_SPREAD})")
    )
