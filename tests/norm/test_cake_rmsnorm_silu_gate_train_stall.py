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
non-deterministic backward (``deterministic=False``) beside a kernel resident
on another stream (the concurrent-launch contract of issues #7038 and #7548).

The forward launches a plain grid of independent CTAs; on SM100 and SM103
from 976 rows per SM the mid forward launches whole waves of 6 CTAs per SM
that each walk at most 2 row blocks, and from 2400 rows per SM the long
forward whole waves of 3 CTAs per SM that each walk at most 6.  The backward launches
at most one wave of CTAs: when the grid has a CTA for every row block, each
CTA takes its own block; past one wave the CTAs claim every block in chunks of
consecutive blocks from the workspace's claim counter, chunks that shrink to
single blocks at the end, each CTA its next chunk as the last block of its
current one starts, so a CTA that starts late takes fewer blocks (one that
becomes resident after every chunk was claimed takes none).  The CTAs add their weight-gradient
sums into FP64 accumulators and take a ticket on the arrival word; whichever
CTA arrives last finishes the sum, and no CTA waits for another.  Beside a
long-resident one-thread kernel on another stream each call must keep its
speed: per token count, the median of seven event-timed calls with the side
kernel resident is at most ``MAX_RATIO`` times the median of seven calls alone,
or at most ``MAX_EXTRA_MS`` above it.  The alone and busy calls alternate in
pairs, so host launch gaps and clock drift hit both arms alike.

The backward beside a kernel resident on every SM (issue #7548, item 2): each
CTA of the backward needs 20736 bytes of dynamic shared memory, so a side
kernel holding every SM with the default carveout delays its start until the
side kernel exits (documented in :mod:`flashinfer.cake_rmsnorm_silu_gate_train`).
The documented remedy is checked here: with the side kernel's preferred
carveout set to the maximum shared memory, the backward (BF16 and FP32 weight
gradients) and the forward start beside it and cost at most
``MAX_ALL_SM_EXTRA_MS`` more than alone, far below the side kernel's
``ALL_SM_SPIN_MS`` lifetime.

The backward beside a side kernel that holds ``k`` of the ``N`` SMs (issue
#7548, item 1): busy / alone is at most the proportional ``N / (N - k)``, plus
the share of one row block in the call (the last chunks are single blocks, so
the CTAs finish within about one block of each other), plus
``PROPORTIONAL_SLACK``.  The token counts give each CTA of the backward's
one-wave grid 16 and 160 row blocks: ``T = 3157`` and ``31573`` on a
148-SM B200, where the earlier fixed one-wave grid of 16 blocks per CTA ran
1.26-1.35 times slower beside any side kernel.  The long and mid forwards
beside the same side kernels: busy / alone is at most ``N / (N - k)``, plus
the share of one CTA's blocks in the call (a CTA whose slot the side kernel
holds walks all of its blocks once it starts), plus ``PROPORTIONAL_SLACK``, at
the token counts that give the long forward's CTA slots 25 and 100 row blocks
each (its threshold of 2400 rows per SM and four times that) and the mid
forward's 11 and 24 (1056 and 2304 rows per SM: the fewest whole blocks
per slot at or above its threshold of 976 rows per SM, and the most below the
long forward's).

In the item 1 and item 2 tests every side CTA records its SM and its start
time, and a run counts only when the side kernel held the intended SMs before
the call started.  In every test ``y``, ``dx`` and ``dz`` are bitwise those of
the first call alone, every ``dw`` lies within the backend suite's budget of an
FP64 reference (the FP32 runs and the FP64 cross-CTA sum depend on which blocks
each CTA claimed), and the workspace is zero after every call.
"""

from __future__ import annotations

import ctypes
import functools
import statistics

import pytest
import torch

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_train")
loader = cake_package("cake_rmsnorm_silu_gate_train", "cake_jit")
forward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_forward, deterministic=False
)
backward = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward, deterministic=False
)
workspace_for = functools.partial(
    pkg.cake_rmsnorm_silu_gate_train_backward_workspace, deterministic=False
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
SEED = 8067

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or not pkg.is_cake_rmsnorm_silu_gate_train_supported(
        torch.device("cuda"), deterministic=False
    ),
    reason="Cake SiLU-gated RMSNorm training needs an SM100 / SM103 / SM107 GPU",
)


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


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
# give each CTA of its one-wave grid this many row blocks.
PROPORTIONAL_SIDES = ((1, 128), (16, 640), (32, 640))
PROPORTIONAL_BLOCKS_PER_CTA = (16, 160)
PROPORTIONAL_RUNS = 7
# The noise margin of busy / alone over N / (N - k) plus one row block's
# share.  Measured with this backward beside these tests' side kernels (1 x
# 128, 16 x 640 and 32 x 640, default carveout) at 16 and 160 row blocks per
# CTA, the median of 11 to 13 alternating alone / busy pairs: the largest
# excess was -0.0074 on B200 (148 SMs; nsc-svg-slurm-1-gpu-129, job 2451658),
# -0.0065 on GB300 (152 SMs; nvl72d148-T17, job 1459208) and -0.0084 on R200
# (212 SMs; hecate0089, job 920765), and the largest excess of a single pair
# +0.0081 (R200).  0.02 is that worst pair plus a margin for noise.
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


def _dw_reference(dy, x, z):
    """``(dw64, mag)``: the FP64 weight gradient of the BF16 inputs and
    ``sum |a * n|`` per column, the scale of its rounding (the backend suite's
    reference)."""
    xd = x.double()
    n = xd * torch.rsqrt(xd.square().mean(-1, keepdim=True) + EPS)
    terms = (dy.double() * torch.nn.functional.silu(z.double()) * n).reshape(-1, D)
    return terms.sum(0), terms.abs().sum(0)


def _check_outputs(stage: str, tokens: int, alone, busy, dw_reference=None) -> None:
    """``y`` (forward) or ``dx`` and ``dz`` (backward) bitwise those of the
    first call alone; every backward ``dw`` within the backend suite's budget
    of ``dw_reference``."""
    first = alone[0][1]
    for _t, out in alone + busy:
        if stage == "forward":
            assert torch.equal(out[0], first[0]), f"y tokens={tokens}"
            continue
        assert torch.equal(out[0], first[0]), f"dx tokens={tokens}"
        assert torch.equal(out[1], first[1]), f"dz tokens={tokens}"
        dw64, mag = dw_reference
        err = (out[2].double() - dw64).abs()
        bound = dw64.abs() * DW_REL_TOL + mag * DW_MAG_TOL
        assert bool((err <= bound).all()), (
            f"dw tokens={tokens} {out[2].dtype}: beyond the budget by "
            f"{(err - bound).max().item():.3e}"
        )


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


def _claim_tokens(blocks_per_cta: int) -> tuple[int, float]:
    """``(T, block_share)``: the most tokens ``T`` whose ``T * H`` rows give
    each CTA of the BF16 weight-gradient backward's one-wave grid
    (``backward_ctas``) at most ``blocks_per_cta`` row blocks, and the grid over
    the row blocks of the module those rows run (on sm_107a the long backward
    from 1024 rows per SM), which bounds one row block's share of the call
    alone."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.route_module(arch, "bwd_bf16")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
    rows_per_cta = int(loader.MODULES[name]["grid_rule"]["rows_per_cta"])
    tokens = blocks_per_cta * wave * rows_per_cta // H
    name = loader.module_for_rows(arch, "bwd_bf16", tokens * H, sms)
    rows_per_cta = int(loader.MODULES[name]["grid_rule"]["rows_per_cta"])
    blocks = -(-(tokens * H) // rows_per_cta)
    ctas = loader.backward_ctas(name, tokens * H, sms)
    assert ctas == loader.backward_ctas(name, loader.INT32_MAX, sms)
    return tokens, ctas / blocks


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("blocks_per_cta", PROPORTIONAL_BLOCKS_PER_CTA)
def test_backward_slows_in_proportion_to_the_sms_held(
    blocks_per_cta: int, ctas: int, threads: int
) -> None:
    tokens, block_share = _claim_tokens(blocks_per_cta)
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
        f"blocks_per_cta={blocks_per_cta} side={ctas}x{threads} "
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


# The long and mid forwards beside the item 1 side kernels: the long forward's
# CTA slots (3 per SM) get 25 and 100 row blocks of 32 rows each, the mid
# forward's (6 per SM) 11 and 24 row blocks of 16 rows each.
LONG_FORWARD_BLOCKS_PER_SLOT = (25, 100)
MID_FORWARD_BLOCKS_PER_SLOT = (11, 24)


def _persistent_forward_tokens(role: str, blocks_per_slot: int) -> tuple[int, float]:
    """``(T, cta_share)``: the fewest tokens ``T`` whose ``T * H`` rows give
    each CTA slot of the forward module of ``role`` (``fwd_long`` or
    ``fwd_mid``) ``blocks_per_slot`` row blocks, and the share of one CTA's
    blocks in the call alone (blocks a CTA walks over blocks a slot runs);
    skips where the architecture routes no such module."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    assert loader.module_for_rows(arch, "fwd", rows, sms) == name
    (ctas, _, _) = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sms)
    blocks = -(-rows // rows_per_block)
    return tokens, -(-blocks // ctas) * slots / blocks


def _forward_slows_in_proportion(
    role: str, blocks_per_slot: int, ctas: int, threads: int
) -> None:
    tokens, cta_share = _persistent_forward_tokens(role, blocks_per_slot)
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
@pytest.mark.parametrize("blocks_per_slot", LONG_FORWARD_BLOCKS_PER_SLOT)
def test_long_forward_slows_in_proportion_to_the_sms_held(
    blocks_per_slot: int, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_long", blocks_per_slot, ctas, threads)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("blocks_per_slot", MID_FORWARD_BLOCKS_PER_SLOT)
def test_mid_forward_slows_in_proportion_to_the_sms_held(
    blocks_per_slot: int, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_mid", blocks_per_slot, ctas, threads)
