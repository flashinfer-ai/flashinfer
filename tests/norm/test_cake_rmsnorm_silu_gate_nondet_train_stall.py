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

The forward runs the program of its architecture's band of rows per SM:
of ``FORWARD_BANDS`` where ``x`` and ``z`` hold packed rows (row ``n`` at
element ``128 n``, fewer than ``PACKED_ROWS`` rows), else of
``STRIDED_FORWARD_BANDS``.  The plain forward launches a grid of independent
CTAs, one 64-thread CTA per block of 8 rows.  Packed rows up to 38 rows per SM
on SM100, 36 on SM103 and 65 on SM107 run the tiny forward, one 128-thread CTA
per block of 8 rows.  Other layouts up to 12 rows per SM on SM100 and SM103
(59 on SM107) run the short forward, one 256-thread CTA per block of 16 rows
(with 64 SMs or more on SM100 and SM103, as with 148 or 152, fewer CTAs than
the SMs less 16; on SM107 with 205 SMs or more, as with 212 or 216, at most 4
CTAs per SM on all SMs but 16; fewer SMs do not keep these bounds at the end
of the band).  The pipelined forwards (the deterministic package's forward
programs with a gate that does not flush) launch one 128-thread CTA per block
of 16 rows below 8192 rows per SM and from there whole waves of 6 CTAs per SM
that each walk at most 2 blocks; the mid forward launches whole waves of 6
CTAs per SM that each walk at most 2 blocks of 16 rows; the long and the
extra-long forward launch whole waves of 3 CTAs per SM that each walk at most
6 blocks of 32 rows.  Their bands past the tiny or the short forward's: packed
rows on SM100 run the pipelined forward up to 85 rows per SM, the mid forward
up to 160, the long forward up to 400, the extra-long forward up to 520, the
pipelined forward up to 6500 and the mid forward past it; on SM103 the mid
forward up to 240, the extra-long forward up to 600, the pipelined forward up
to 6000, the mid forward up to 8192 and the pipelined forward past it; on
SM107 the mid forward up to 175, the extra-long forward up to 280 and the
pipelined forward of packed rows past it.  Other layouts on SM100 run the mid
forward up to 140 rows per SM, the long forward up to 400, the extra-long
forward up to 700 and the plain forward past it; on SM103 the mid forward up
to 250, the extra-long forward up to 700 and the plain forward past it; on
SM107 the mid forward up to 480 and the plain forward past it.

The backward runs the program of its architecture's ``BACKWARD_BANDS`` band
of the weight gradient's dtype (a call without a weight gradient takes the
BF16 bands).  A small-row backward, up to a count of rows, is one cluster of
16 CTAs, scheduled together on the SMs of one GPC, whose CTAs each wait for
the other CTAs' FP32 partials of their columns and finish them in a fixed
order, with no workspace: the short backward of a BF16 weight gradient
(256-thread CTAs of 16-row blocks, 6144 bytes of dynamic
shared memory each) and the exact tiny backward (512-thread CTAs of 4 teams
of 8-row blocks, 19584 bytes each).  The chunked backwards cut the rows into
blocks of 8 rows and the blocks into ``C = ceil(blocks / B)`` chunks of
``B = max(1, ceil(blocks / 8192))`` consecutive blocks, and launch
``G = min(C, 8 * SMs)`` CTAs of 128 threads, at most one wave (8 CTAs per
SM).  A static backward (the call role) runs the rows up to a few row blocks
a CTA of that grid; its CTA ``b`` runs the chunks ``b``, ``b + G``, ...  The
claimed backward's CTAs claim runs of consecutive chunks from the workspace's
claim counter, in halving rounds
and then single chunks, each CTA its next run as the last block of its
current run starts, so a CTA that starts late takes fewer chunks (one that
becomes resident after every run was claimed takes none).  Each CTA of a
chunked backward adds the FP32 weight-gradient partials of its chunks exactly
into 64-bit integer windows of the workspace and takes a ticket on the
arrival word; whichever CTA arrives last rounds the exact sum once into
``dw``, and no CTA waits for another.  The row-claims backwards of a BF16
weight gradient cut the rows into blocks of 16 rows and launch
``G = min(blocks, ctas_per_sm * SMs)`` CTAs of 256 threads; past one wave the
CTAs claim the blocks from the workspace's claim counter, so a CTA that
starts late takes fewer blocks; each CTA adds its FP32 runs of rows (the
row-claims mid backward: as FP32 two-sum pairs) to FP64 accumulators of the
workspace with atomics and takes a ticket, and the CTA that arrives last
rounds the sum into ``dw``.  The fused backward (``bwd_bf16_fused``, a BF16
weight gradient in a band of rows per SM on every architecture, on sm_107a
past a first repeatable band; grid kind ``row_claims``: the row arithmetic and the
block schedule of the deterministic package's BF16 backward program) cuts
the rows into blocks of 16 rows and launches ``G = min(blocks, 4 * SMs)``
CTAs of 256 threads; where the rows have more blocks than the grid has CTAs,
the CTAs claim runs of blocks from the workspace's claim counter, in halving
rounds and then single blocks, so a CTA that starts late takes fewer blocks;
each CTA adds the FP32 runs of its rows (folded as FP32 two-sum pairs) to
FP64 accumulators of the workspace with atomics and takes a ticket, and the
CTA that arrives last rounds the sum once into ``dw``.  The repeatable
backward (``bwd_bf16_rep``, a BF16 weight gradient past the fused band on
every architecture and on sm_107a also before it; grid kind
``claimed_row_blocks``: the deterministic package's BF16 backward program) has
the fused backward's grid and block
schedule, and adds each claimed chunk's weight-gradient terms exactly into
fixed-point digits of the workspace, so its ``dw`` depends on the rows and
the SM count only.  Packed rows of its last band from
``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on (2560 on SM100 and SM103, 1536
on SM107) run its long role (``bwd_bf16_rep_long``: the deterministic
package's look-ahead program of the same build, the same grid, block
schedule and bits).
Beside a long-resident
one-thread kernel on another stream each call must keep its speed: per token
count, the median of seven event-timed calls with the side kernel resident is
at most ``MAX_RATIO`` times the median of seven calls alone, or at most
``MAX_EXTRA_MS`` above it.  The alone and busy calls alternate in pairs, so
host launch gaps and clock drift hit both arms alike.

The backward beside a kernel resident on every SM (issue #7548, item 2): each
CTA of the chunked backwards needs 15616 bytes of dynamic shared memory (the
launch record's ``dynamic_smem_bytes``: the two halves of the 8 x 128 FP32
fold buffer, 8192 bytes, the 7 64-bit windows of each of the 128 columns,
7168 bytes, and the run cell, the arrival flag and the 8-entry claim table,
160 bytes, 15520 bytes rounded up), each CTA of the row-claims backwards
20736 bytes, each CTA of the fused backward 20736 bytes and each CTA of the
repeatable backward and of its long role 26368 bytes, so a side kernel
holding every SM with a
small carveout can
delay its start until the side kernel exits (no test here asserts that delay;
documented in :mod:`flashinfer.cake_rmsnorm_silu_gate_nondet_train`).  The
documented remedy is checked here: with the side kernel's preferred carveout
set to the maximum shared memory, the backward (BF16 and FP32 weight
gradients) and the forward start beside it and cost at most
``MAX_ALL_SM_EXTRA_MS`` more than alone, far below the side kernel's
``ALL_SM_SPIN_MS`` lifetime; so do the small-row backwards of a BF16 and an
FP32 weight gradient at 1 and 5 tokens (48 and 240 rows, where a small-row
band of the role holds them) and at the first whole tokens of the role's
second small-row band (the exact tiny backward of a BF16 weight gradient on
SM100 and SM107, past the short backward's band), which on SM100 and SM103 also start
beside it with the default and the maximum-L1 carveout, and the backward that
takes the rows past the role's small-row bands, at its first whole tokens
(``SHORT_BACKWARD_PROBES``).  On SM107 they do not start beside the side
kernel with such a small carveout until that kernel exits (observed on R200
for the small-row backwards): there each must end at most
``MAX_ALL_SM_EXTRA_MS`` after the backward it replaces (the module of the
role's first band past its small-row bands, at the same rows).

The backward beside a side kernel that holds ``k`` of the ``N`` SMs (issue
#7548, item 1): busy / alone is at most the proportional ``N / (N - k)``, plus
the share in the call of one row block of each CTA of the module's grid (its
CTAs, ``backward_ctas``, over its row blocks), plus ``PROPORTIONAL_SLACK``.
The token counts give each CTA of a module's one-wave grid (``ctas_per_sm *
N`` CTAs; blocks of 8 rows of a chunked backward, of 16 rows of a row-claims
backward) 16, 32 and 160 row blocks, in the first band of a BF16 weight
gradient whose module ``BACKWARD_BANDS`` routes at those rows (the
repeatable backward at all three: from 1024 rows per SM; its long role where
those rows reach its ``LONG_BACKWARD_ROWS_PER_SM``: at 160 on every
architecture and at 32 too on SM107); their rows have
more chunks or blocks than the grid has CTAs.  So does the long role at the
first whole tokens one row block past its first rows.  The same bound holds at the
edges of the bands, every size derived from
``BACKWARD_BANDS`` and the device's SM count: in each static multi-chunk
band (a static backward's band that holds rows with more chunks than its
grid has CTAs) of either weight-gradient dtype, at its first such whole
tokens and at the most tokens whose rows give each CTA the band end's whole
row blocks a CTA (the band's last whole tokens where that is not past the
first; a weight-gradient dtype with no such band skips); in each row-claims
band (where a band runs a row-claims backward), in the fused band and in each
repeatable band (every architecture, two on sm_107a; their CTAs claim 16-row
blocks at run time where the rows have more blocks than the grid has CTAs), at its first
and its last whole tokens, or, in a band without an end, at its first whole
tokens and the first whole tokens whose rows have more blocks than one wave
of its grid has CTAs, where the band's first tokens are below them (the last
repeatable band's are not: its "last" edge skips).  The
persistent forwards beside the same side kernels: busy / alone is at most
``N / (N - k)``, plus the share of one CTA's blocks in the call (a CTA whose
slot the side kernel
holds walks all of its blocks once it starts), plus ``PROPORTIONAL_SLACK``,
in each band of the module, of packed rows and of a strided layout (``z``
the gate columns of a wider projection), at the band's first and last rows
per SM and at the token counts that give the module's CTA slots the fewest
whole row blocks each past the previous band's end and the most up to its
band's end, or four times the fewest in the last band, which has no end: the
mid forward of the strided layout 1 block of 16 rows on SM100, 1 and 2 on
SM103 and 1 and 5 on SM107 (one band each), of packed rows 1 in its first band
and 68 and 272 in its second on SM100, 1 and 2 in its first band and 63 and 85
in its second on SM103 and 1 on SM107; the long forward 2 and 4 blocks of 32
rows on SM100; the extra-long forward of the strided layout 5 and 7 on SM100
and 3 and 7 on SM103, of packed rows 5 on SM100, 3 and 6 on SM103 and 2 on
SM107.  The pipelined forwards by the same
bound, in each of their bands
at its first and last rows per SM and at the deterministic package's probes
the band holds: 85 blocks of 16 rows per slot (the most below 8192 rows per
SM, where a CTA runs one block), 86 (the fewest whole waves from there) and
344.  The forwards of the first band, the tiny forward of packed rows (at 1,
12, 31 and 38 rows per SM on SM100, 1, 12, 31 and 36 on SM103, 1, 10, 30,
43 and 65 on SM107) and
the short forward of the strided layout (at 1 and 12 rows per SM on SM100 and
SM103, at 1, 10, 30 and 59 on SM107), each up to the end of its band, keep
their speed beside the one-thread kernel, start beside the all-SM kernel with
the max-shared carveout, and beside a side kernel that holds ``k`` SMs (the
item 1 side kernels and 16 CTAs of 1024 threads with the max-shared carveout)
end no later than the deterministic forward (the deterministic package's
forward as it routes the rows, ``cake_jit.forward_module``) beside the same
side kernel: the busy
median is at most the deterministic forward's busy median times ``1 +
PROPORTIONAL_SLACK``, or at most ``SHORT_EXTRA_MS`` above it: one tick of the
event timer (about 1 microsecond on SM103) and a half, the calls taking a few
microseconds.  The two forwards' alone / busy pairs alternate round by round
over the same stretch of time, which does not make a slow stretch of the host
or the GPU fall on both alike: a measurement counts only when each forward's
alone median is at most ``SHORT_MAX_ALONE_SPREAD`` times its fastest alone
sample of that measurement, else it is taken again, at most
``SHORT_EXTRA_ATTEMPTS`` more times, and the bound applies to the first
measurement that counts (to the last one when none does).  The small-row
backwards (``SHORT_BACKWARD_PROBES``, BF16 and FP32 weight gradients) keep
their speed beside the one-thread kernel and, beside the short forward's side
kernels, end no later than the backward they replace (the module of the
role's first band past its small-row bands, at the same rows: for a BF16
weight gradient the fused backward (on sm_107a the repeatable backward), for
an FP32 one the static backward) by
the short
forward's bound and measurement rule: there a measurement
counts only when the small-row backward's and the replaced backward's alone
medians are each at most ``SHORT_MAX_ALONE_SPREAD`` times their fastest alone
sample of that measurement.

In the item 1 and item 2 tests every side CTA records its SM and its start
time, and a run counts only when the side kernel held the intended SMs before
the call started.  In every test ``y``, ``dx`` and ``dz`` are bitwise those of
the first call alone of the same entry point and module, and so is ``dw`` of
every module but the fused and the row-claims backwards (a chunk's partial
does not depend on the grid or on the CTA that claimed it, the sum of the
partials is exact, and a small-row backward sums in a fixed order: the
repeatable backward's ``dw`` among them); every
``dw`` lies within the backend suite's budget of an FP64 reference (the
fused and the row-claims backwards' ``dw`` is checked against that budget
only: its FP32 runs and the order of its FP64 additions follow the blocks
each CTA claimed), the small-row backward's ``dx`` and ``dz`` are bitwise
those of the backward it replaces (where that is the fused backward, whose
row arithmetic is the deterministic package's, those of this package's own
BF16 backward at the same rows instead, the role's static backward where the
architecture routes one, else the row-claims backward, and the fused
backward's ``dx`` and ``dz`` the deterministic package's bits), and the
workspace is zero after every call.  PyTorch's deterministic algorithms are
off in every test (the fused and the row-claims backwards raise PyTorch's
alert under them).
"""

from __future__ import annotations

import ctypes
import functools
import math
import statistics

import pytest
import torch

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_rmsnorm_silu_gate_nondet_train")
loader = cake_package("cake_rmsnorm_silu_gate_nondet_train", "cake_jit")
det_pkg = cake_package("cake_rmsnorm_silu_gate_det_train")
det_loader = cake_package("cake_rmsnorm_silu_gate_det_train", "cake_jit")
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


@pytest.fixture(autouse=True)
def nondeterministic_allowed():
    """PyTorch's deterministic algorithms off in every test: a backward that
    runs the fused or a row-claims module raises PyTorch's alert under them,
    and these tests time the calls."""
    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(False)
    yield
    torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _inputs(tokens: int, layout: str = "packed"):
    """``(x, z, w, dy)`` of ``tokens`` tokens of H heads, packed rows; with
    ``layout="strided"`` ``z`` the gate columns of a wider projection, whose
    rows are not packed (with more than one token)."""
    generator = torch.Generator(device="cuda").manual_seed(SEED + tokens)
    kwargs = dict(dtype=torch.float32, device="cuda", generator=generator)
    x = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    if layout == "packed":
        z = torch.randn(tokens, H, D, **kwargs).to(torch.bfloat16)
    else:
        assert layout == "strided", layout
        wide = torch.randn(tokens, 2 * H * D + 16, **kwargs).to(torch.bfloat16)
        z = wide[:, H * D : 2 * H * D].view(tokens, H, D)
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


def _launched(call, registry=None) -> list[str]:
    """The modules one ``call`` alone launches through ``registry`` (this
    package's ``cake_jit`` unless another package's is given)."""
    registry = loader if registry is None else registry
    names: list[str] = []
    run = registry.run_module

    def spy(name, *args, **kwargs):
        names.append(name)
        return run(name, *args, **kwargs)

    registry.run_module = spy
    try:
        call()
        torch.cuda.synchronize()
    finally:
        registry.run_module = run
    return names


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
    if stage == "forward":
        _check_outputs(stage, tokens, alone, busy)
    else:
        _check_outputs(
            stage,
            tokens,
            alone,
            busy,
            _dw_reference(dy, x, z),
            module=_backward_module(tokens * H),
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
# give each CTA of the one-wave grid of the module that runs them
# (``ctas_per_sm`` CTAs per SM) this many of its row blocks.
PROPORTIONAL_SIDES = ((1, 128), (16, 640), (32, 640))
PROPORTIONAL_BLOCKS_PER_CTA = (16, 32, 160)
PROPORTIONAL_RUNS = 7
# The margin for timing noise of busy / alone over N / (N - k) plus one row
# block's share, and of the tiny, the short forward's and the short backward's
# busy medians over those of the calls they are compared with.
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


def _device() -> tuple[str, int]:
    """``(arch, SMs)`` of the current CUDA device."""
    device_index = torch.cuda.current_device()
    return (
        loader.device_arch(device_index),
        torch.cuda.get_device_properties(device_index).multi_processor_count,
    )


def _backward_role(dw_dtype: torch.dtype) -> str:
    """The backward role of a ``dw_dtype`` weight gradient."""
    return "bwd_f32" if dw_dtype == torch.float32 else "bwd_bf16"


def _backward_module(rows: int, dw_dtype: torch.dtype = torch.bfloat16) -> str:
    """The module that runs a backward of ``rows`` rows into a ``dw_dtype``
    weight gradient on the current device (``module_for_rows``)."""
    arch, sms = _device()
    return loader.module_for_rows(arch, _backward_role(dw_dtype), rows, sms)


def _dw_repeats(name: str) -> bool:
    """Whether the backward module ``name`` repeats its weight gradient bit for
    bit: every module but the fused and the row-claims backwards (grid kind
    ``row_claims``), whose CTAs claim row blocks at run time and add their
    sums with FP64 atomics."""
    return loader.MODULES[name]["grid_rule"]["kind"] != "row_claims"


def _rep(name: str) -> bool:
    """Whether the backward module ``name`` is the repeatable backward
    (``bwd_bf16_rep``) or its long role (``bwd_bf16_rep_long``, packed rows
    from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM on): the deterministic
    package's BF16 backward program or its look-ahead program of the same
    bits, whose ``dx``, ``dz`` and ``dw`` are that package's bits."""
    return loader.MODULES[name]["role"] in ("bwd_bf16_rep", "bwd_bf16_rep_long")


def _fused(name: str) -> bool:
    """Whether the backward module ``name`` is the fused backward
    (``bwd_bf16_fused``), whose row arithmetic is the deterministic package's
    BF16 backward program's, so that its ``dx`` and ``dz`` are that
    package's bits."""
    return loader.MODULES[name]["role"] == "bwd_bf16_fused"


# The name digest of the deterministic package's BF16 backward program whose
# row arithmetic the fused backward runs (the same build of the same row
# operations: ``dx`` and ``dz`` bit for bit).
DET_FUSED_DIGEST = "8a6cb16beff15b6ffbfe"
# The name digest of the deterministic package's look-ahead BF16 backward
# program (its long role, the same build with the same bits), which the
# repeatable backward's long role runs.
DET_LONG_DIGEST = "fef9d1cc2cc267219992"


def _det_routes_same_program() -> bool:
    """Whether the deterministic package's BF16 backward on the current device
    is the program of ``DET_FUSED_DIGEST`` and its long role, where it routes
    one, the program of ``DET_LONG_DIGEST`` (the same bits); once that
    package ships another BF16 backward, the ``dx`` and ``dz`` bits of the
    two packages need not agree."""
    arch, _sms = _device()
    routes = det_loader.ROUTES[arch]
    long = routes.get("bwd_bf16" + det_loader.LONG, f"_{DET_LONG_DIGEST}")
    same = routes["bwd_bf16"].endswith(f"_{DET_FUSED_DIGEST}")
    return same and long.endswith(f"_{DET_LONG_DIGEST}")


def _backward_with(module: str, dy, x, z, w, dw, workspace):
    """One backward into ``dw`` that runs the module ``module`` whatever its
    rows and layout (``module_for_rows`` replaced for the call)."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(pkg, "module_for_rows", lambda *_args, **_kwargs: module)
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)


def _check_outputs(
    stage: str,
    tokens: int,
    alone,
    busy,
    dw_reference=None,
    *,
    module: str | None = None,
) -> None:
    """``y`` (forward) or ``dx`` and ``dz`` (backward) of every call of one
    entry point and module bitwise those of its first call alone, and so its
    ``dw`` where the backward ``module`` repeats it (``_dw_repeats``); every
    backward ``dw`` within the backend suite's budget of ``dw_reference``."""
    assert stage == "forward" or module is not None, "the backward's module"
    first = alone[0][1]
    for _t, out in alone + busy:
        if stage == "forward":
            assert torch.equal(out[0], first[0]), f"y tokens={tokens}"
            continue
        assert torch.equal(out[0], first[0]), f"dx tokens={tokens}"
        assert torch.equal(out[1], first[1]), f"dz tokens={tokens}"
        if _dw_repeats(module):
            assert torch.equal(out[2], first[2]), f"dw tokens={tokens} {out[2].dtype}"
        dw64, mag = dw_reference
        err = (out[2].double() - dw64).abs()
        bound = dw64.abs() * DW_REL_TOL + mag * DW_MAG_TOL
        assert bool((err <= bound).all()), (
            f"dw tokens={tokens} {out[2].dtype}: beyond the budget by "
            f"{(err - bound).max().item():.3e}"
        )


def _check_replaced_rows(
    tokens: int, alone, replaced_alone, replaced: str, inputs, dw_dtype
) -> None:
    """``dx`` and ``dz`` of the small-row backward's first call alone bitwise
    those of the backward it replaces, the module ``replaced`` (the same row
    arithmetic; their weight gradients sum different partials and are each
    checked by :func:`_check_outputs`).  Where ``replaced`` is the fused
    backward (:func:`_fused`) or the repeatable backward (:func:`_rep`, which
    replaces the small-row backwards on sm_107a), whose row arithmetic is the
    deterministic package's, its first call's ``dx`` and ``dz`` are instead
    bitwise those of the deterministic package's backward of ``inputs`` (``x``, ``z``,
    ``w``, ``dy``) into a ``dw_dtype`` weight gradient while that package
    routes that program (:func:`_det_routes_same_program`), and the small-row
    backward's ``dx`` and ``dz`` bitwise those of this package's own backward
    of the role at the same rows (the role's static backward
    ``ROUTES[arch][role]`` where the architecture routes one, else its
    row-claims backward ``<role>_rc``: the row arithmetic the small-row
    backward shares), which leaves its workspace zero."""
    first, rep = alone[0][1], replaced_alone[0][1]
    if not (_fused(replaced) or _rep(replaced)):
        assert torch.equal(rep[0], first[0]), f"dx tokens={tokens} (replaced)"
        assert torch.equal(rep[1], first[1]), f"dz tokens={tokens} (replaced)"
        return
    x, z, w, dy = inputs
    if _det_routes_same_program():
        det = det_pkg.cake_rmsnorm_silu_gate_det_train_backward(
            dy,
            x,
            z,
            w,
            EPS,
            grad_weight=torch.empty(D, dtype=dw_dtype, device=x.device),
        )
        which = "repeatable" if _rep(replaced) else "fused"
        for what, got, want in zip(("dx", "dz"), rep[:2], det[:2], strict=True):
            assert torch.equal(got, want), (
                f"{what} tokens={tokens}: the {which} backward vs the deterministic package"
            )
    arch, _sms = _device()
    role = _backward_role(dw_dtype)
    routes = loader.ROUTES[arch]
    own = routes[role] if role in routes else routes[f"{role}_rc"]
    assert not (_fused(own) or _rep(own)), (arch, own)
    workspace = workspace_for(tokens * H, x.device)
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)
    same = _backward_with(own, dy, x, z, w, dw, workspace)
    assert torch.equal(same[0], first[0]), f"dx tokens={tokens} (own backward)"
    assert torch.equal(same[1], first[1]), f"dz tokens={tokens} (own backward)"
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"


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
    _check_outputs(
        "backward",
        tokens,
        alone,
        busy,
        _dw_reference(dy, x, z),
        module=_backward_module(tokens * H, dw_dtype),
    )
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


def _band_rows(arch: str, role: str, sms: int) -> list[tuple[int, int | None, str]]:
    """``(first, last, module)`` of each ``BACKWARD_BANDS`` band of ``role`` on
    ``arch`` with ``sms`` SMs, in order: its rows from one past the previous
    band's last row (1 in the first band) to its last row (``(ROWS, n)``:
    ``n``; ``r`` rows per SM: ``floor(r * sms)``; ``None``: no end).  A band
    whose last row is below its first holds no rows."""
    bands = []
    first = 1
    for last, band in loader.BACKWARD_BANDS[arch][role]:
        if last is None:
            end = None
        elif isinstance(last, tuple):
            assert last[0] == loader.ROWS, (arch, role, last)
            end = int(last[1])
        else:
            end = math.floor(last * sms)
        bands.append((first, end, loader.route_module(arch, band)))
        if end is None:
            break
        first = max(first, end + 1)
    return bands


def _band_kind(name: str) -> str:
    """The schedule of the backward module ``name``: ``"cluster"`` (a
    small-row backward), ``"static"`` (a chunked backward of a call role: CTA
    ``b`` runs the chunks ``b``, ``b + grid``, ...), ``"claimed"`` (another
    chunked backward: its CTAs claim runs of chunks), ``"row_claims"``,
    ``"fused"`` (``bwd_bf16_fused``, of grid kind ``row_claims`` too: the
    deterministic package's row arithmetic and block schedule, whose CTAs
    claim runs of 16-row blocks where the rows have more blocks than the grid
    has CTAs) or ``"repeatable"`` (``bwd_bf16_rep`` and its long role
    ``bwd_bf16_rep_long``, grid kind ``claimed_row_blocks``: the
    deterministic package's program and its look-ahead program, the same
    schedule)."""
    record = loader.MODULES[name]
    kind = record["grid_rule"]["kind"]
    if _fused(name):
        return "fused"
    if _rep(name):
        return "repeatable"
    if kind != "chunks":
        return kind
    return "static" if record["role"] in loader.ROLES else "claimed"


def _rows_per_block(name: str) -> int:
    """Rows of a row block of the chunked, row-claims, fused or repeatable
    backward module ``name``: ``rows_per_block`` of a chunked or the
    repeatable one, ``rows_per_cta`` of a row-claims or the fused one."""
    rule = loader.MODULES[name]["grid_rule"]
    return int(
        rule["rows_per_cta" if rule["kind"] == "row_claims" else "rows_per_block"]
    )


def _work_units(name: str, rows: int) -> int:
    """What the CTAs of the backward module ``name`` share out at ``rows``
    rows: the chunks of a chunked backward (``backward_chunks``), the row
    blocks of a row-claims, the fused or the repeatable backward."""
    if loader.MODULES[name]["grid_rule"]["kind"] == "chunks":
        return loader.backward_chunks(name, rows)[1]
    return -(-rows // _rows_per_block(name))


def _block_share(name: str, rows: int, sms: int) -> float:
    """The share in a call alone of one row block of each CTA of the backward
    module ``name`` at ``rows`` rows: its CTAs (``backward_ctas``) over its
    row blocks."""
    return loader.backward_ctas(name, rows, sms) / -(-rows // _rows_per_block(name))


def _claim_tokens(blocks_per_cta: int) -> tuple[int, str, float]:
    """``(T, module, block_share)``: the module of the first
    ``BACKWARD_BANDS`` band of a BF16 weight gradient (past the small-row
    bands) that runs the most tokens ``T`` whose ``T * H`` rows give each CTA
    of its own one-wave grid (``backward_ctas``: ``ctas_per_sm`` CTAs per SM)
    at most ``blocks_per_cta`` of its row blocks, rows with more chunks or
    blocks than that grid has CTAs (or of its long role, which runs these
    packed rows in its place from ``LONG_BACKWARD_ROWS_PER_SM`` rows per SM
    on with the same grid), and the share of one row block of each of its
    CTAs in the call alone (``_block_share``); skips where no band's module
    runs its own such rows."""
    arch, sms = _device()
    for _first, _last, name in _band_rows(arch, "bwd_bf16", sms):
        if _band_kind(name) == "cluster":
            continue
        wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
        tokens = blocks_per_cta * wave * _rows_per_block(name) // H
        long = loader.ROUTES[arch].get(loader.MODULES[name]["role"] + loader.LONG)
        got = loader.module_for_rows(arch, "bwd_bf16", tokens * H, sms)
        if got in (name, long):
            name = got
            break
    else:
        pytest.skip(
            f"{arch} ({sms} SMs): no BF16 backward band's module runs "
            f"{blocks_per_cta} row blocks a CTA of its own grid"
        )
    rows = tokens * H
    ctas = loader.backward_ctas(name, rows, sms)
    assert ctas == wave, (arch, rows, name, ctas, wave)
    assert _work_units(name, rows) > ctas, (arch, rows, name, ctas)
    return tokens, name, _block_share(name, rows, sms)


def _backward_slows_in_proportion(
    case: str,
    tokens: int,
    name: str,
    block_share: float,
    ctas: int,
    threads: int,
    dw_dtype: torch.dtype | None = None,
) -> None:
    """Busy / alone of the backward of ``tokens`` tokens (``dw_dtype`` weight
    gradient; ``None``: the default BF16 one) beside ``ctas`` side CTAs of
    ``threads`` threads, at most ``N / (N - ctas) + block_share +
    PROPORTIONAL_SLACK``; the call runs the module ``name``."""
    x, z, w, dy = _inputs(tokens)
    rows = tokens * H
    workspace = workspace_for(rows, x.device)
    dw = None if dw_dtype is None else torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

    assert _launched(call) == [name], (case, tokens, name)
    spin = _spin(None, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + block_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    role = loader.MODULES[name]["role"]
    grid = loader.backward_ctas(name, rows, spin.sms)
    blocks = -(-rows // _rows_per_block(name))
    print(
        f"HELD stage=backward case={case} role={role} kind={_band_kind(name)} "
        f"tokens={tokens} heads={H} rows={rows} dw={dw_dtype} ctas={grid} "
        f"units={_work_units(name, rows)} blocks_per_cta={blocks / grid:.2f} "
        f"side={ctas}x{threads} alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={ratio:.3f} ideal={ideal:.3f} block_share={block_share:.4f} "
        f"excess={ratio - ideal - block_share:.4f}"
    )
    _check_outputs(
        "backward", tokens, alone, busy, _dw_reference(dy, x, z), module=name
    )
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert ratio <= bound, (
        f"tokens={tokens} ({case}): {role} alone {alone_ms:.4f} ms, beside "
        f"{ctas} x {threads} side CTAs on {ctas} SMs {busy_ms:.4f} ms: ratio "
        f"{ratio:.3f} > {bound:.3f} (proportional {ideal:.3f} + one row block "
        f"{block_share:.4f} + {PROPORTIONAL_SLACK})"
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("blocks_per_cta", PROPORTIONAL_BLOCKS_PER_CTA)
def test_backward_slows_in_proportion_to_the_sms_held(
    blocks_per_cta: int, ctas: int, threads: int
) -> None:
    tokens, name, block_share = _claim_tokens(blocks_per_cta)
    _backward_slows_in_proportion(
        f"{blocks_per_cta}_blocks_per_cta", tokens, name, block_share, ctas, threads
    )


# The band edges of the backward beside the item 1 side kernels, by the same
# bound.  A static multi-chunk band: the band of a static backward
# (``_band_kind``) that holds rows with more chunks than its one-wave grid has
# CTAs (so a CTA runs more than one chunk); "first" is its first whole tokens
# of H heads with more chunks than CTAs, "last" the most tokens whose rows
# give each CTA of that grid the band end's whole row blocks a CTA
# (``floor(band end / (wave * rows_per_block))``), or the band's last whole
# tokens where those are not past "first" (a band that ends below 2 whole row
# blocks a CTA).  A row-claims band, the fused band and the repeatable band
# (whose CTAs claim 16-row blocks at run time where the rows have more blocks
# than its grid has CTAs): "first" and "last" are its first and its last whole
# tokens; in a band without an end (the last repeatable band) "last" is the first
# whole tokens whose rows have more blocks than one wave of its grid has
# CTAs, where the band's first tokens are below them.  Every size comes from
# BACKWARD_BANDS and the device's SM count; the ``nth`` such band of the
# weight gradient's role, where the architecture has one.
BACKWARD_EDGES = ("first", "last")


def _edge_band(arch: str, last, band: str, kind: str) -> bool:
    """Whether the ``BACKWARD_BANDS`` band ``(last, band)`` of ``arch`` is a
    band of ``kind`` (``_band_kind`` of its module) the edge tests probe: any
    row-claims, fused or repeatable band; a static band without an end, of a
    count of rows, or
    whose end in rows per SM passes one wave of one-block chunks
    (``ctas_per_sm * rows_per_block``: only past it does a static backward
    run more chunks than CTAs)."""
    name = loader.route_module(arch, band)
    if _band_kind(name) != kind:
        return False
    if kind != "static" or last is None or isinstance(last, tuple):
        return True
    rule = loader.MODULES[name]["grid_rule"]
    return last > int(rule["ctas_per_sm"]) * int(rule["rows_per_block"])


def _backward_band_probes(kind: str) -> list:
    """``(dw_dtype, nth)`` parameters: for each weight-gradient dtype, ``1 ..
    n`` for the most ``BACKWARD_BANDS`` bands of its role that any
    architecture has of ``kind`` (``_edge_band``); for ``"static"`` at least
    ``1``, so a dtype without a static multi-chunk band skips with the
    reason."""
    params = []
    for dw_dtype, label in ((torch.bfloat16, "bf16"), (torch.float32, "f32")):
        role = _backward_role(dw_dtype)
        count = max(
            sum(_edge_band(arch, last, band, kind) for last, band in bands[role])
            for arch, bands in loader.BACKWARD_BANDS.items()
        )
        if kind == "static":
            count = max(count, 1)
        params += [
            pytest.param(dw_dtype, nth, id=f"{label}-band{nth}")
            for nth in range(1, count + 1)
        ]
    return params


def _band_edge(
    dw_dtype: torch.dtype, kind: str, nth: int, edge: str
) -> tuple[int, str, float]:
    """``(T, module, block_share)`` of ``edge`` of the ``nth`` static
    multi-chunk band (``kind`` "static"), row-claims band (``kind``
    "row_claims"), fused band (``kind`` "fused") or repeatable band (``kind``
    "repeatable") of a ``dw_dtype`` weight
    gradient on the current device
    (``BACKWARD_EDGES``), its module and the share of one row block of each
    of its CTAs in the call alone (``_block_share``); skips where the
    architecture has fewer such bands, or where "last" is not past "first"
    ("last" of a band without an end: the first whole tokens past one wave of
    its grid's row blocks)."""
    arch, sms = _device()
    role = _backward_role(dw_dtype)
    bands = []
    for first, last, name in _band_rows(arch, role, sms):
        if _band_kind(name) != kind:
            continue
        if kind == "static":
            # the first rows with more chunks than the one-wave grid has CTAs
            wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
            first = max(first, wave * _rows_per_block(name) + 1)
        lowest = -(-first // H)
        if last is None or lowest * H <= last:
            bands.append((lowest, last, name))
    if len(bands) < nth:
        holds = " with more chunks than CTAs" if kind == "static" else ""
        pytest.skip(
            f"{arch} ({sms} SMs) has {len(bands)} {kind} backward band(s) of "
            f"{role} holding whole tokens of {H} heads{holds}, not {nth}"
        )
    lowest, last, name = bands[nth - 1]
    tokens = lowest
    if edge == "last" and last is None:
        assert kind != "static", f"{arch} {role}: the static band of {name} has no end"
        # past one wave of the grid's row blocks: the CTAs claim blocks
        wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
        tokens = wave * _rows_per_block(name) // H + 1
        if tokens <= lowest:
            pytest.skip(
                f"{arch} {role} {kind} band {nth}: its first tokens ({lowest}) "
                f"are past one wave of its grid ({tokens} tokens)"
            )
    elif edge == "last":
        tokens = last // H
        if kind == "static":
            # the band end's whole row blocks a CTA, where past "first"
            wave = loader.backward_ctas(name, loader.INT32_MAX, sms)
            per_block = wave * _rows_per_block(name)
            whole = last // per_block * per_block // H
            if whole > lowest:
                tokens = whole
        if tokens <= lowest:
            pytest.skip(
                f"{arch} {role} {kind} band {nth}: its last edge ({tokens} "
                f"tokens) is not past its first ({lowest} tokens)"
            )
    rows = tokens * H
    assert loader.module_for_rows(arch, role, rows, sms) == name, (arch, role, rows)
    if kind == "static":
        ctas = loader.backward_ctas(name, rows, sms)
        assert _work_units(name, rows) > ctas, (arch, rows, name, ctas)
    return tokens, name, _block_share(name, rows, sms)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", BACKWARD_EDGES)
@pytest.mark.parametrize(("dw_dtype", "nth"), _backward_band_probes("static"))
def test_static_backward_band_edges_slow_in_proportion_to_the_sms_held(
    dw_dtype: torch.dtype, nth: int, edge: str, ctas: int, threads: int
) -> None:
    tokens, name, block_share = _band_edge(dw_dtype, "static", nth, edge)
    _backward_slows_in_proportion(
        f"static_band{nth}_{edge}", tokens, name, block_share, ctas, threads, dw_dtype
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", BACKWARD_EDGES)
@pytest.mark.parametrize(("dw_dtype", "nth"), _backward_band_probes("row_claims"))
def test_row_claims_backward_band_edges_slow_in_proportion_to_the_sms_held(
    dw_dtype: torch.dtype, nth: int, edge: str, ctas: int, threads: int
) -> None:
    tokens, name, block_share = _band_edge(dw_dtype, "row_claims", nth, edge)
    _backward_slows_in_proportion(
        f"row_claims_band{nth}_{edge}",
        tokens,
        name,
        block_share,
        ctas,
        threads,
        dw_dtype,
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", BACKWARD_EDGES)
@pytest.mark.parametrize(("dw_dtype", "nth"), _backward_band_probes("fused"))
def test_fused_backward_band_edges_slow_in_proportion_to_the_sms_held(
    dw_dtype: torch.dtype, nth: int, edge: str, ctas: int, threads: int
) -> None:
    tokens, name, block_share = _band_edge(dw_dtype, "fused", nth, edge)
    _backward_slows_in_proportion(
        f"fused_band{nth}_{edge}", tokens, name, block_share, ctas, threads, dw_dtype
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", BACKWARD_EDGES)
@pytest.mark.parametrize(("dw_dtype", "nth"), _backward_band_probes("repeatable"))
def test_repeatable_backward_band_edges_slow_in_proportion_to_the_sms_held(
    dw_dtype: torch.dtype, nth: int, edge: str, ctas: int, threads: int
) -> None:
    tokens, name, block_share = _band_edge(dw_dtype, "repeatable", nth, edge)
    _backward_slows_in_proportion(
        f"repeatable_band{nth}_{edge}",
        tokens,
        name,
        block_share,
        ctas,
        threads,
        dw_dtype,
    )


# The repeatable backward's long role (``bwd_bf16_rep_long``) beside the item 1
# side kernels, by the same bound: packed rows one row block past its first
# rows (``LONG_BACKWARD_ROWS_PER_SM * SMs``), as whole tokens of H heads; its
# CTAs claim 16-row blocks at run time, many a CTA of its one-wave grid.
@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
def test_repeatable_backward_long_role_slows_in_proportion_to_the_sms_held(
    ctas: int, threads: int
) -> None:
    arch, sms = _device()
    name = loader.ROUTES[arch]["bwd_bf16_rep_long"]
    start = loader.LONG_BACKWARD_ROWS_PER_SM[arch]["bwd_bf16_rep"] * sms
    tokens = -(-(start + _rows_per_block(name)) // H)
    rows = tokens * H
    assert loader.module_for_rows(arch, "bwd_bf16", rows, sms) == name, rows
    wave = loader.backward_ctas(name, rows, sms)
    assert wave == loader.backward_ctas(name, loader.INT32_MAX, sms), (rows, wave)
    assert _work_units(name, rows) > wave, (arch, rows, name, wave)
    _backward_slows_in_proportion(
        "repeatable_long",
        tokens,
        name,
        _block_share(name, rows, sms),
        ctas,
        threads,
    )


# The persistent forwards beside the item 1 side kernels: the CTA slots of the
# long and the extra-long forward (3 per SM) get row blocks of 32 rows, the mid
# forward's (6 per SM) row blocks of 16 rows, in the module's ``nth`` band of
# the architecture's FORWARD_BANDS (``layout`` "packed": packed rows) or
# STRIDED_FORWARD_BANDS ("strided": ``z`` the gate columns of a wider
# projection), at the band's first and last row per SM ("band_first",
# "band_last"; the last band has no last), at the fewest whole blocks per slot
# past the previous band's end ("first") and at the most up to the band's end,
# or four times "first" in the last band, which has no end ("last"): the mid
# forward of other layouts 1 block per slot on SM100 (its fewest and most), 1
# and 2 on SM103 and 1 and 5 on SM107 (one band each), of packed rows 1 in its
# first band and 68 and 272 in its second on SM100, 1 and 2 in its first band
# and 63 and 85 in its second on SM103 and 1 on SM107; the long forward 2 and
# 4 on SM100; the extra-long forward of other layouts 5 and 7 on SM100 and 3
# and 7 on SM103, of packed rows 5 on SM100, 3 and 6 on SM103 and 2 on SM107.
# A probe outside the band skips, and so does a last probe that is the band's
# first.
FORWARD_EDGES = ("band_first", "first", "last", "band_last")
FORWARD_LAYOUTS = ("packed", "strided")


def _forward_bands(layout: str) -> dict:
    """The forward bands the rows of ``layout`` run by."""
    return loader.FORWARD_BANDS if layout == "packed" else loader.STRIDED_FORWARD_BANDS


def _band_ordinals(role: str) -> tuple[int, ...]:
    """``1 .. n`` for the most ``FORWARD_BANDS`` or ``STRIDED_FORWARD_BANDS``
    bands of ``role`` any architecture routes (so no ordinal is a band no
    architecture has)."""
    count = max(
        sum(band == role for _last, band in bands)
        for layout in FORWARD_LAYOUTS
        for bands in _forward_bands(layout).values()
    )
    return tuple(range(1, count + 1))


def _forward_band(
    arch: str, role: str, nth: int, layout: str
) -> tuple[int, int | None]:
    """``(first, last)`` rows per SM of the ``nth`` band (from 1) of ``role``
    in the architecture's forward bands of ``layout``: one more than the
    previous band's end, and the band's end (``None``: no end); skips where
    the architecture has fewer bands of ``role``."""
    first, seen = 1, 0
    for last, band in _forward_bands(layout)[arch]:
        if band == role:
            seen += 1
            if seen == nth:
                return first, last
        if last is not None:
            first = int(last) + 1
    pytest.skip(f"{arch} has {seen} {layout} forward band(s) of {role}, not {nth}")


def _persistent_forward_tokens(
    role: str, edge: str, nth: int, layout: str
) -> tuple[int, int, float]:
    """``(blocks_per_slot, T, cta_share)``: the row blocks per CTA slot of the
    forward module of ``role`` (``fwd_xlong``, ``fwd_long`` or ``fwd_mid``)
    at ``edge`` of its ``nth`` band of ``layout``, the fewest tokens ``T``
    whose ``T * H`` rows give each slot that many (at "band_first" and
    "band_last" the tokens of the band's first and last row per SM and their
    blocks per slot), and the share of one CTA's blocks in the call alone
    (blocks a CTA walks over blocks a slot runs); skips where the
    architecture routes no such module or band, or the probe is outside the
    band."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    first, last = _forward_band(arch, role, nth, layout)
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    # rows per SM that give each slot of the SM one more row block
    step = int(rule["ctas_per_sm"]) * rows_per_block
    if edge == "band_first":
        tokens = -(-(first * sms) // H)
    elif edge == "band_last":
        if last is None:
            pytest.skip(f"{arch} {layout} {role} band {nth} has no end")
        tokens = last * sms // H
    else:
        blocks_per_slot = -(-first // step)
        if edge == "last":
            fewest = blocks_per_slot
            blocks_per_slot = 4 * blocks_per_slot if last is None else last // step
            if blocks_per_slot == fewest:
                pytest.skip(
                    f"{arch} {layout} {role} band {nth}: its last edge is its "
                    f"first ({fewest} block(s) per slot)"
                )
        assert blocks_per_slot >= 1, (arch, role, nth, edge)
        tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    if not first * sms <= rows or (last is not None and rows > last * sms):
        pytest.skip(
            f"{arch}: {edge} ({rows} rows) is outside the {layout} band {nth} of "
            f"{role} ({first} to {last} rows per SM)"
        )
    packed = layout == "packed"
    assert loader.module_for_rows(arch, "fwd", rows, sms, packed=packed) == name
    (ctas, _, _) = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sms)
    blocks = -(-rows // rows_per_block)
    return blocks / slots, tokens, -(-blocks // ctas) * slots / blocks


def _forward_slows_in_proportion(
    role: str,
    edge: str,
    nth: int,
    ctas: int,
    threads: int,
    layout: str,
    probe: tuple[float, int, float] | None = None,
) -> None:
    blocks_per_slot, tokens, cta_share = (
        _persistent_forward_tokens(role, edge, nth, layout) if probe is None else probe
    )
    x, z, w, _dy = _inputs(tokens, layout)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    name = loader.ROUTES[loader.device_arch(x.device.index)][role]
    assert _launched(call) == [name], (role, layout, tokens)

    spin = _spin(None, ctas=ctas, threads=threads)
    alone_ms, busy_ms, alone, busy = _paired_medians(call, spin, PROPORTIONAL_RUNS)
    ideal = spin.sms / (spin.sms - ctas)
    bound = ideal + cta_share + PROPORTIONAL_SLACK
    ratio = busy_ms / alone_ms
    print(
        f"HELD stage=forward tokens={tokens} heads={H} "
        f"blocks_per_slot={blocks_per_slot:g} side={ctas}x{threads} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} ratio={ratio:.3f} "
        f"ideal={ideal:.3f} cta_share={cta_share:.4f} "
        f"excess={ratio - ideal - cta_share:.4f} role={role} layout={layout}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert ratio <= bound, (
        f"tokens={tokens} ({blocks_per_slot:g} row blocks per CTA slot): {role} "
        f"alone {alone_ms:.4f} ms, beside {ctas} x {threads} side CTAs "
        f"on {ctas} SMs {busy_ms:.4f} ms: ratio {ratio:.3f} > {bound:.3f} "
        f"(proportional {ideal:.3f} + one CTA's blocks {cta_share:.4f} + "
        f"{PROPORTIONAL_SLACK})"
    )


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("nth", _band_ordinals("fwd_long"), ids=lambda n: f"band{n}")
@pytest.mark.parametrize("layout", FORWARD_LAYOUTS)
def test_long_forward_slows_in_proportion_to_the_sms_held(
    layout: str, nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_long", edge, nth, ctas, threads, layout)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("nth", _band_ordinals("fwd_xlong"), ids=lambda n: f"band{n}")
@pytest.mark.parametrize("layout", FORWARD_LAYOUTS)
def test_extra_long_forward_slows_in_proportion_to_the_sms_held(
    layout: str, nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_xlong", edge, nth, ctas, threads, layout)


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", FORWARD_EDGES)
@pytest.mark.parametrize("nth", _band_ordinals("fwd_mid"), ids=lambda n: f"band{n}")
@pytest.mark.parametrize("layout", FORWARD_LAYOUTS)
def test_mid_forward_slows_in_proportion_to_the_sms_held(
    layout: str, nth: int, edge: str, ctas: int, threads: int
) -> None:
    _forward_slows_in_proportion("fwd_mid", edge, nth, ctas, threads, layout)


# The pipelined forwards (the deterministic package's forward programs with
# a gate that does not flush: CTA
# slots of 6 per SM, row blocks of 16 rows, one CTA a block below
# ``persistent_from_rows_per_sm`` rows per SM, then whole waves whose CTAs walk
# at most 2 blocks) in their ``nth`` band of ``layout``: at the band's first
# and last row per SM ("band_first", "band_last"; the last band has no last),
# and at the probes of the deterministic package's own test: the most whole
# blocks per slot below ``persistent_from_rows_per_sm`` ("below"), the fewest
# whole waves at or above it ("first") and four times that ("last").  A probe
# outside the band skips.
PIPELINED_FORWARD_EDGES = ("band_first", "band_last", "below", "first", "last")


def _pipelined_forward_tokens(
    role: str, nth: int, edge: str, layout: str
) -> tuple[float, int, float]:
    """``(blocks_per_slot, T, cta_share)`` of :func:`_persistent_forward_tokens`
    for the pipelined forward ``role`` (``fwd_pipe`` or ``fwd_pipe_packed``)
    at ``edge`` of its ``nth`` band of ``layout``; skips where the
    architecture routes no such module or band, or the probe is outside the
    band."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    first, last = _forward_band(arch, role, nth, layout)
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    rule = loader.MODULES[name]["grid_rule"]
    slots = int(rule["ctas_per_sm"]) * sms
    rows_per_block = int(rule["rows_per_block"])
    most = int(rule["max_blocks_per_cta"])
    start = int(rule["persistent_from_rows_per_sm"])
    # rows per SM that give each slot of the SM one more row block
    step = int(rule["ctas_per_sm"]) * rows_per_block
    if edge == "band_first":
        tokens = -(-(first * sms) // H)
    elif edge == "band_last":
        if last is None:
            pytest.skip(f"{arch} {layout} {role} band {nth} has no end")
        tokens = last * sms // H
    else:
        if edge == "below":
            blocks_per_slot = (start - 1) // step
        else:
            blocks_per_slot = -(-start // (step * most)) * most
            if edge == "last":
                blocks_per_slot *= 4
        tokens = -(-(blocks_per_slot * slots * rows_per_block) // H)
    rows = tokens * H
    if not first * sms <= rows or (last is not None and rows > last * sms):
        pytest.skip(
            f"{arch}: {edge} ({rows} rows) is outside the {layout} band {nth} of "
            f"{role} ({first} to {last} rows per SM)"
        )
    packed = layout == "packed"
    assert loader.module_for_rows(arch, "fwd", rows, sms, packed=packed) == name
    (ctas, _, _) = loader.launch_grid(loader.MODULES[name], rows=rows, sm_count=sms)
    blocks = -(-rows // rows_per_block)
    return blocks / slots, tokens, -(-blocks // ctas) * slots / blocks


@pytest.mark.parametrize(
    "ctas,threads", PROPORTIONAL_SIDES, ids=["1x128", "16x640", "32x640"]
)
@pytest.mark.parametrize("edge", PIPELINED_FORWARD_EDGES)
@pytest.mark.parametrize("nth", (1, 2), ids=lambda n: f"band{n}")
@pytest.mark.parametrize("layout", FORWARD_LAYOUTS)
@pytest.mark.parametrize("role", ["fwd_pipe", "fwd_pipe_packed"])
def test_pipelined_forward_slows_in_proportion_to_the_sms_held(
    role: str, layout: str, nth: int, edge: str, ctas: int, threads: int
) -> None:
    probe = _pipelined_forward_tokens(role, nth, edge, layout)
    _forward_slows_in_proportion(role, edge, nth, ctas, threads, layout, probe)


# The forwards of the first forward band: the tiny forward (packed rows) and
# the short forward (other layouts, here ``z`` the gate columns of a wider
# projection).  The rows per SM of their token counts on each architecture
# (one row per SM, rows between, and the end of the band, the largest, up to
# which the architecture runs the module: the tiny forward 38 on sm_100a, 36
# on sm_103a and 65 on sm_107a; the short forward 12 on sm_100a and sm_103a
# and 59 on sm_107a), and the side kernels of the item 1 test beside
# them, the issue's three with the default carveout and 16 CTAs of 1024
# threads with the max-shared carveout.
FIRST_FORWARD_ROWS_PER_SM = {
    "fwd_tiny": {
        "sm_100a": (1, 12, 31, 38),
        "sm_103a": (1, 12, 31, 36),
        "sm_107a": (1, 10, 30, 43, 65),
    },
    "fwd_short": {
        "sm_100a": (1, 12),
        "sm_103a": (1, 12),
        "sm_107a": (1, 10, 30, 59),
    },
}
FIRST_FORWARD_PROBES = [
    pytest.param(role, rows_per_sm, id=f"{role}-{rows_per_sm}")
    for role, by_arch in FIRST_FORWARD_ROWS_PER_SM.items()
    for rows_per_sm in sorted(set().union(*by_arch.values()))
]
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


def _first_forward(role: str, rows_per_sm: int) -> tuple[int, str, str]:
    """``(T, layout, module)``: the most tokens ``T`` whose ``T * H`` rows stay
    within ``rows_per_sm`` rows per SM, one of the architecture's
    ``FIRST_FORWARD_ROWS_PER_SM`` of ``role`` (the largest the end of its
    band), at least one token of packed rows for the tiny forward and two of
    the strided layout for the short forward (a single token's rows are
    packed whatever the layout), the layout and ``role``'s module, where the
    architecture routes it (skips elsewhere)."""
    device_index = torch.cuda.current_device()
    arch = loader.device_arch(device_index)
    name = loader.ROUTES[arch].get(role)
    if name is None:
        pytest.skip(f"{arch} routes no {role} module")
    probes = FIRST_FORWARD_ROWS_PER_SM[role][arch]
    if rows_per_sm not in probes:
        pytest.skip(f"{arch} does not probe {rows_per_sm} rows per SM of {role}")
    sms = torch.cuda.get_device_properties(device_index).multi_processor_count
    layout = "packed" if role == "fwd_tiny" else "strided"
    (last, first_role) = _forward_bands(layout)[arch][0]
    assert first_role == role and last == max(probes), (arch, role)
    tokens = max(1 if layout == "packed" else 2, rows_per_sm * sms // H)
    packed = layout == "packed"
    assert loader.module_for_rows(arch, "fwd", tokens * H, sms, packed=packed) == name
    return tokens, layout, name


@pytest.mark.parametrize(("role", "rows_per_sm"), FIRST_FORWARD_PROBES)
def test_first_band_forward_keeps_its_speed_beside_a_resident_kernel(
    role: str, rows_per_sm: int
) -> None:
    tokens, layout, name = _first_forward(role, rows_per_sm)
    x, z, w, _dy = _inputs(tokens, layout)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    assert _launched(call) == [name], (role, layout, tokens)
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
        f"STALL stage={role} layout={layout} tokens={tokens} heads={H} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"{role} tokens={tokens}: {busy_ms:.4f} ms beside a resident "
        f"kernel vs {alone_ms:.4f} ms alone"
    )


@pytest.mark.parametrize(("role", "rows_per_sm"), FIRST_FORWARD_PROBES)
def test_first_band_forward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    role: str, rows_per_sm: int
) -> None:
    tokens, layout, name = _first_forward(role, rows_per_sm)
    x, z, w, _dy = _inputs(tokens, layout)
    y = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    assert _launched(call) == [name], (role, layout, tokens)
    alone_ms, busy_ms, alone, busy = _paired_medians(
        call, _spin(CARVEOUT_MAX_SHARED), ALL_SM_RUNS
    )
    print(
        f"ALL_SM stage={role} layout={layout} tokens={tokens} heads={H} "
        f"carveout=max_shared alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs("forward", tokens, alone, busy)
    assert busy_ms - alone_ms <= MAX_ALL_SM_EXTRA_MS, (
        f"tokens={tokens}: {role} alone {alone_ms:.4f} ms, beside an "
        f"all-SM kernel with the max-shared carveout {busy_ms:.4f} ms (at most "
        f"{MAX_ALL_SM_EXTRA_MS} ms more)"
    )


@pytest.mark.parametrize(
    "ctas,threads,carveout",
    SHORT_FORWARD_SIDES,
    ids=["1x128", "16x640", "32x640", "16x1024-max_shared"],
)
@pytest.mark.parametrize(("role", "rows_per_sm"), FIRST_FORWARD_PROBES)
def test_first_band_forward_ends_no_later_than_the_forward_beside_held_sms(
    role: str, rows_per_sm: int, ctas: int, threads: int, carveout: int | None
) -> None:
    tokens, layout, name = _first_forward(role, rows_per_sm)
    x, z, w, _dy = _inputs(tokens, layout)
    y = torch.empty_like(x)
    y_det = torch.empty_like(x)

    def call():
        return (forward(x, z, w, EPS, out=y),)

    def call_det():
        # the deterministic package's forward as it routes these rows
        # (cake_jit.forward_module)
        return (
            det_pkg.cake_rmsnorm_silu_gate_det_train_forward(x, z, w, EPS, out=y_det),
        )

    assert _launched(call) == [name], (role, layout, tokens)
    arch, sms = _device()

    def strides(t):
        # as the deterministic package passes them: a single token's stride
        # as 0, a single head's as 128
        return (t.stride(0) if tokens > 1 else 0, t.stride(1) if H > 1 else D)

    det_name = det_loader.forward_module(
        arch, tokens * H, H, sms, *strides(x), *strides(z)
    )
    assert _launched(call_det, det_loader) == [det_name], (role, layout, tokens)
    spin = _spin(carveout, ctas=ctas, threads=threads)
    side = f"{ctas}x{threads}" + ("s" if carveout is not None else "")
    attempts = 1 + SHORT_EXTRA_ATTEMPTS
    for attempt in range(1, attempts + 1):
        short_medians, det_medians = _interleaved_paired_medians(
            (call, call_det), spin, PROPORTIONAL_RUNS
        )
        alone_ms, busy_ms, alone, busy = short_medians
        det_alone_ms, det_busy_ms, det_alone, det_busy = det_medians
        alone_min_ms = min(t for t, _ in alone)
        det_alone_min_ms = min(t for t, _ in det_alone)
        spread = max(alone_ms / alone_min_ms, det_alone_ms / det_alone_min_ms)
        counts = spread <= SHORT_MAX_ALONE_SPREAD
        print(
            f"HELD stage={role} layout={layout} tokens={tokens} heads={H} side={side} "
            f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
            f"det_alone_ms={det_alone_ms:.4f} det_busy_ms={det_busy_ms:.4f} "
            f"busy_vs_det={busy_ms / det_busy_ms:.3f} "
            f"alone_min_ms={alone_min_ms:.4f} det_alone_min_ms={det_alone_min_ms:.4f} "
            f"alone_spread={spread:.3f} attempt={attempt}/{attempts} "
            f"counts={int(counts)}"
        )
        # each module's and each package's y repeats its own bits: two programs
        # may associate a row's sum of squares differently, and the
        # deterministic forward's approximate gate and rstd need not give this
        # package's bits
        _check_outputs("forward", tokens, alone, busy)
        _check_outputs("forward", tokens, det_alone, det_busy)
        if counts:
            break
    assert (
        busy_ms <= det_busy_ms * (1 + PROPORTIONAL_SLACK)
        or busy_ms - det_busy_ms <= SHORT_EXTRA_MS
    ), (
        f"tokens={tokens}: beside {ctas} x {threads} side CTAs ({side}) the "
        f"{role} took {busy_ms:.4f} ms, the deterministic forward "
        f"{det_busy_ms:.4f} ms (at most {1 + PROPORTIONAL_SLACK:.2f} times or "
        f"{SHORT_EXTRA_MS} ms more; measurement {attempt} of at most {attempts}, "
        f"alone spread {spread:.3f}, "
        + ("counted)" if counts else f"none counted: above {SHORT_MAX_ALONE_SPREAD})")
    )


# The small-row backwards of a BF16 and an FP32 weight gradient: one token and
# five tokens of H heads (48 and 240 rows, in the role's first small-row
# BACKWARD_BANDS band on every architecture: for a BF16 weight gradient the
# short backward), and "second_band": the first whole tokens of H heads in
# the role's second small-row band (the exact tiny backward of a BF16 weight
# gradient on SM100 and SM107, past the short backward's band; skipped where the role has one small-row band), beside the short
# forward's side kernels; their held test takes the short forward's bound,
# slack and SHORT_MAX_ALONE_SPREAD.  And "past": the first whole tokens of H
# heads past the role's last small-row band (by the band ends of
# BACKWARD_BANDS: 6 tokens past 256 rows, 11 tokens past 512), where the
# module of the role's next band takes over from the small-row backward (for
# a BF16 weight gradient the fused backward, on SM107 the repeatable backward,
# for an FP32 one the static backward); it runs in the keeps-speed and the
# max-shared start tests only.
SHORT_BACKWARD_PROBES = ("1x48", "5x48", "second_band", "past")
SHORT_BACKWARD_DW_DTYPES = pytest.mark.parametrize(
    "dw_dtype", [torch.bfloat16, torch.float32], ids=["bf16", "f32"]
)
SHORT_BACKWARD_SIDES = SHORT_FORWARD_SIDES
# cudaSharedmemCarveoutMaxL1.  The test below checks that the small-row backward
# starts beside a side kernel holding every SM without
# shared memory with this or the default carveout (at most
# MAX_ALL_SM_EXTRA_MS more than alone), or on the architectures of
# SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT that it ends at most
# MAX_ALL_SM_EXTRA_MS after the backward it replaces (the module of the role's
# first band past its small-row bands) beside that side kernel.
CARVEOUT_MAX_L1 = 0


# Why the small-carveout and held tests skip the "past" probe: the backward it
# runs is the backward a small-row backward replaces (a held test would
# compare it with itself), and this file asserts no start of the chunked, the
# row-claims, the fused or the repeatable backwards beside a side kernel
# holding every SM with
# a small carveout ("a side kernel holding every SM with a small carveout can delay
# its start until the side kernel exits (no test here asserts that delay;
# documented in flashinfer.cake_rmsnorm_silu_gate_nondet_train)").
PAST_PROBE_SKIP = (
    "the backward past the small-row band: it is the backward a small-row "
    "backward replaces, and a side kernel holding every SM with a small "
    "carveout can delay its start until the side kernel exits (no test here "
    "asserts that delay; documented in "
    "flashinfer.cake_rmsnorm_silu_gate_nondet_train)"
)


def _small_row_bands(arch: str, role: str, sms: int) -> list[tuple[int, int, str]]:
    """``(first, last, module)`` of each small-row ``BACKWARD_BANDS`` band of
    ``role`` on ``arch`` with ``sms`` SMs (``BAND_ROLES``: a count of rows
    each) that holds rows, in order (``_band_rows``)."""
    return [
        (first, last, name)
        for first, last, name in _band_rows(arch, role, sms)
        if loader.MODULES[name]["role"] in loader.BAND_ROLES
        and last is not None
        and first <= last
    ]


def _past_small_row_bands(arch: str, role: str, sms: int) -> tuple[int, str]:
    """``(first row, module)`` of the first ``BACKWARD_BANDS`` band of
    ``role`` on ``arch`` with ``sms`` SMs past its small-row bands
    (``BAND_ROLES``) that holds rows: the backward a small-row backward
    replaces."""
    for first, last, name in _band_rows(arch, role, sms):
        if loader.MODULES[name]["role"] in loader.BAND_ROLES:
            continue
        if last is None or first <= last:
            return first, name
    raise AssertionError(f"{arch} {role}: no band past the small-row bands")


def _short_backward(
    probe: str, dw_dtype: torch.dtype, *, past: bool = False
) -> tuple[int, str, str]:
    """``(tokens, the probed backward, the backward it replaces)`` of the
    ``SHORT_BACKWARD_PROBES`` entry ``probe`` for a ``dw_dtype`` weight
    gradient: the tokens of H heads (for ``second_band`` the first whole
    tokens in the role's second small-row ``BACKWARD_BANDS`` band, for
    ``past`` the first whole tokens past the role's small-row bands), the
    module that runs their rows and the module of the role's first band past
    its small-row bands (``_past_small_row_bands``).  ``past`` probes run that
    module and skip unless ``past`` is set; ``second_band`` probes skip where
    the role has no second small-row band holding whole tokens; the others
    skip where no small-row band holds their rows."""
    arch, sms = _device()
    role = _backward_role(dw_dtype)
    first, replaced = _past_small_row_bands(arch, role, sms)
    if probe == "past":
        if not past:
            pytest.skip(PAST_PROBE_SKIP)
        tokens = -(-first // H)
        name = loader.module_for_rows(arch, role, tokens * H, sms)
        assert name == replaced, (arch, role, tokens, name)
        return tokens, name, replaced
    if probe == "second_band":
        bands = _small_row_bands(arch, role, sms)
        if len(bands) < 2:
            pytest.skip(f"{arch} has {len(bands)} small-row band(s) of {role}")
        band_first, band_last, name = bands[1]
        tokens = -(-band_first // H)
        if tokens * H > band_last:
            pytest.skip(
                f"{arch} {role}: the second small-row band ({band_first}-"
                f"{band_last} rows) holds no whole tokens of {H} heads"
            )
        got = loader.module_for_rows(arch, role, tokens * H, sms)
        assert got == name, (arch, role, tokens, got)
        return tokens, name, replaced
    tokens = int(probe.split("x")[0])
    name = loader.module_for_rows(arch, role, tokens * H, sms)
    if loader.MODULES[name]["role"] not in loader.BAND_ROLES:
        pytest.skip(f"{arch} runs no small-row backward for {tokens * H} rows")
    return tokens, name, replaced


@pytest.mark.parametrize("probe", SHORT_BACKWARD_PROBES)
@SHORT_BACKWARD_DW_DTYPES
def test_short_backward_keeps_its_speed_beside_a_resident_kernel(
    dw_dtype: torch.dtype, probe: str
) -> None:
    tokens, name, _replaced = _short_backward(probe, dw_dtype, past=True)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

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
        f"STALL stage=short_backward tokens={tokens} heads={H} dw={dw_dtype} "
        f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
        f"ratio={busy_ms / alone_ms:.3f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs(
        "backward", tokens, alone, busy, _dw_reference(dy, x, z), module=name
    )
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    assert _keeps_its_speed(alone_ms, busy_ms), (
        f"short backward tokens={tokens}: {busy_ms:.4f} ms beside a resident "
        f"kernel vs {alone_ms:.4f} ms alone"
    )


@pytest.mark.parametrize("probe", SHORT_BACKWARD_PROBES)
@SHORT_BACKWARD_DW_DTYPES
def test_short_backward_starts_beside_an_all_sm_kernel_with_the_max_shared_carveout(
    dw_dtype: torch.dtype, probe: str
) -> None:
    tokens, name, _replaced = _short_backward(probe, dw_dtype, past=True)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
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
    _check_outputs(
        "backward", tokens, alone, busy, _dw_reference(dy, x, z), module=name
    )
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
# module of the role's first band past its small-row bands, on SM107 the
# repeatable backward, 26368 bytes a CTA, for a BF16 weight gradient and
# the static backward, 15616 bytes a CTA, for an FP32 one) beside it.
SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT = ("sm_107a",)


@pytest.mark.parametrize("probe", SHORT_BACKWARD_PROBES)
@SHORT_BACKWARD_DW_DTYPES
@pytest.mark.parametrize("carveout", [None, CARVEOUT_MAX_L1], ids=["default", "max_l1"])
def test_short_backward_beside_an_all_sm_kernel_with_a_small_carveout(
    carveout: int | None, dw_dtype: torch.dtype, probe: str
) -> None:
    tokens, name, replaced = _short_backward(probe, dw_dtype)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

    alone_ms, busy_ms, alone, busy = _paired_medians(call, _spin(carveout), ALL_SM_RUNS)
    label = "default" if carveout is None else "max_l1"
    print(
        f"ALL_SM stage=short_backward tokens={tokens} heads={H} "
        f"dw={dw_dtype} carveout={label} alone_ms={alone_ms:.4f} "
        f"busy_ms={busy_ms:.4f} extra_ms={busy_ms - alone_ms:.4f}"
    )
    _check_outputs(
        "backward", tokens, alone, busy, _dw_reference(dy, x, z), module=name
    )
    assert not bool(workspace.any()), f"tokens={tokens}: the workspace is not zero"
    if loader.device_arch(x.device.index) in SHORT_BACKWARD_WAITS_FOR_SMALL_CARVEOUT:

        def call_replaced():
            # the module the short backward replaces at these rows
            return _backward_with(replaced, dy, x, z, w, dw, workspace)

        rep_alone_ms, rep_busy_ms, rep_alone, rep_busy = _paired_medians(
            call_replaced, _spin(carveout), ALL_SM_RUNS
        )
        print(
            f"ALL_SM stage=replaced_backward tokens={tokens} heads={H} "
            f"dw={dw_dtype} carveout={label} alone_ms={rep_alone_ms:.4f} "
            f"busy_ms={rep_busy_ms:.4f} extra_ms={rep_busy_ms - rep_alone_ms:.4f}"
        )
        _check_outputs(
            "backward",
            tokens,
            rep_alone,
            rep_busy,
            _dw_reference(dy, x, z),
            module=replaced,
        )
        _check_replaced_rows(
            tokens, alone, rep_alone, replaced, (x, z, w, dy), dw_dtype
        )
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
@pytest.mark.parametrize("probe", SHORT_BACKWARD_PROBES)
@SHORT_BACKWARD_DW_DTYPES
def test_short_backward_ends_no_later_than_the_backward_beside_held_sms(
    dw_dtype: torch.dtype, probe: str, ctas: int, threads: int, carveout: int | None
) -> None:
    tokens, name, replaced = _short_backward(probe, dw_dtype)
    x, z, w, dy = _inputs(tokens)
    workspace = workspace_for(tokens * H, x.device)
    dw = torch.empty(D, dtype=dw_dtype, device=x.device)

    def call():
        return backward(dy, x, z, w, EPS, grad_weight=dw, workspace=workspace)

    def call_replaced():
        # the module the short backward replaces at these rows
        return _backward_with(replaced, dy, x, z, w, dw, workspace)

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
            f"HELD stage=short_backward tokens={tokens} heads={H} dw={dw_dtype} "
            f"side={side} "
            f"alone_ms={alone_ms:.4f} busy_ms={busy_ms:.4f} "
            f"replaced_alone_ms={rep_alone_ms:.4f} replaced_busy_ms={rep_busy_ms:.4f} "
            f"busy_vs_replaced={busy_ms / rep_busy_ms:.3f} "
            f"alone_min_ms={alone_min_ms:.4f} replaced_alone_min_ms={rep_alone_min_ms:.4f} "
            f"alone_spread={spread:.3f} attempt={attempt}/{attempts} "
            f"counts={int(counts)}"
        )
        # each module's outputs repeat their own bits; dx and dz match across them
        _check_outputs("backward", tokens, alone, busy, dw_reference, module=name)
        _check_outputs(
            "backward", tokens, rep_alone, rep_busy, dw_reference, module=replaced
        )
        _check_replaced_rows(
            tokens, alone, rep_alone, replaced, (x, z, w, dy), dw_dtype
        )
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
