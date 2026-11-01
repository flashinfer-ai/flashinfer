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

Tests for the Cake Gated DeltaProduct (GDP) decode / MTP-verification kernels
(``K = 128``, ``V = 64``, one to four Householder updates per
token, FP32 state pool; SM90 / SM100 / SM103).

Every output row and every pool slot the call may touch is compared with an
FP64 token-serial reference of the same math (inputs at their BF16 values,
the state rounded to FP32 after each real token) at ``atol = 1e-2``,
``rtol = 5e-3`` with no mismatch allowed; every slot the call must not touch
is checked bitwise.  The matrix covers ``num_householder`` 1 to 4, the head
counts 1 to 80, batches 1 to 257 and token counts 1 to 16 (the 48 decode shapes
of record among them), distinct write slots, padding rows (negative read
slots, whose output rows keep the caller's values), rows without writeback,
per-token state scatter with holes, ``disable_state_update``, the scale
values, a padded pool, strided inputs and outputs, FP16 I/O, int64 indices,
the rejected arguments, batch-row independence, bitwise reproducibility and
one launch of every exported program.

This suite exercises the backend package alone: it names the package once,
through ``tests.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_gdp`` facade and the ``flashinfer`` top-level exposure are
covered by ``test_cake_gdp_decode_api.py``.
"""

from __future__ import annotations

import itertools
import re
from pathlib import Path
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from tests.test_helpers.cake_package import cake_package

pkg = cake_package("cake_gdp")
cake_build = cake_package("cake_gdp", "cake_build")
loader = cake_package("cake_gdp", "cake_jit")
cake_gated_delta_product_mtp = pkg.cake_gated_delta_product_mtp
is_cake_gdp_supported = pkg.is_cake_gdp_supported

SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3))
K = 128
V = 64
ATOL = 1e-2
RTOL = 5e-3
SEED = 760
NUM_HOUSEHOLDER = (1, 2, 3, 4)
HEADS = (1, 2, 3, 6, 10, 12, 20, 24, 40, 80)
BATCHES = (1, 2, 3, 7, 16, 64, 257)
TOKENS = (1, 2, 3, 4, 5, 8, 16)
# The 48 decode shapes of record (num_householder = 3): every performance row is a correctness case.
PERF_HEADS = (80, 20, 10, 24)
PERF_BATCHES = (1, 16, 64, 256)
PERF_TOKENS = (1, 4, 8)
SENTINEL = 7.5


def _device() -> torch.device:
    return torch.device("cuda:0")


def _capability() -> tuple[int, int]:
    return torch.cuda.get_device_capability(_device())


def _supported() -> bool:
    return (
        torch.cuda.is_available()
        and _capability() in SUPPORTED_CAPABILITIES
        and is_cake_gdp_supported(_device())
    )


pytestmark = pytest.mark.skipif(
    not _supported(),
    reason="cake_gdp needs an SM90 / SM100 / SM103 device with exported routes",
)


@pytest.fixture(autouse=True, scope="module")
def warmup_jit():
    # Build the local architecture's libraries that have no prebuilt file before
    # the first test, through the spec objects the package's loader itself builds
    # on first use (``cake_build.load_library``); a prebuilt or already cached
    # library loads without compiling.
    if _supported():
        arch = loader.arch_for_capability(_capability())
        for spec in loader.specs_for_arch(arch):
            spec.build_and_load()
    yield


def _arch() -> str:
    return loader.arch_for_capability(_capability())


def _generator(*keys: int) -> torch.Generator:
    seed = SEED
    for key in keys:
        seed = (seed * 1_000_003 + int(key)) % (2**31 - 1)
    return torch.Generator(device=_device()).manual_seed(seed)


def _skip_unless_route(n: int, T: int, use_qk_l2norm: bool = True) -> None:
    if not loader.route_applies(
        device_capability=_capability(),
        num_householder=n,
        num_tokens=T,
        use_qk_l2norm=use_qk_l2norm,
    ):
        pytest.skip(
            f"num_householder={n}, T={T}, use_qk_l2norm={use_qk_l2norm} has no exported program on this device"
        )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class Case:
    """One decode call: its inputs, the pool before the call and the slot tables."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def arguments(self) -> tuple[tuple, dict]:
        return (
            (
                self.q,
                self.k,
                self.v,
                self.pool,
                self.read,
                self.A_log,
                self.a,
                self.dt_bias,
                self.b,
            ),
            dict(
                scale=self.scale,
                output=self.output,
                ssm_state_indices=self.ssm,
                disable_state_update=self.disable_state_update,
                use_qk_l2norm=self.use_qk_l2norm,
                output_state_indices=self.write,
            ),
        )


def _slots(
    count: int,
    pool_size: int,
    *,
    exclude: torch.Tensor | None,
    generator: torch.Generator,
) -> torch.Tensor:
    """``count`` distinct int32 slots of a ``pool_size`` pool, none in ``exclude``."""

    free = torch.ones(pool_size, dtype=torch.bool, device=_device())
    if exclude is not None:
        free[exclude[exclude >= 0].long()] = False
    candidates = torch.nonzero(free).flatten()
    assert candidates.numel() >= count, (count, pool_size)
    order = torch.randperm(candidates.numel(), device=_device(), generator=generator)
    return candidates[order[:count]].to(torch.int32)


def _make(
    B: int,
    T: int,
    H: int,
    n: int,
    *,
    generator: torch.Generator,
    dtype: torch.dtype = torch.bfloat16,
    pool_size: int | None = None,
    distinct_write: bool = False,
    padded_pool: bool = False,
    strided: bool = False,
    scale: float | None = None,
    disable_state_update: bool = False,
    use_qk_l2norm: bool = True,
    scatter_tokens: bool = False,
) -> Case:
    """Random inputs at the scale of the kernel's own tests: q / k / v ~ N(0, 1), ``a``, ``b``, ``A_log``,
    ``dt_bias`` and the pool at ``0.1 x N(0, 1)``; the pool holds unused slots and the read slots are distinct."""

    device = _device()

    def randn(*shape: int) -> torch.Tensor:
        return torch.randn(
            *shape, device=device, generator=generator, dtype=torch.float32
        )

    HP = H + 2 if strided else H
    q = randn(B, T, HP, K).to(dtype)
    k = randn(B, T, n, HP, K).to(dtype)
    v = randn(B, T, n, HP, V).to(dtype)
    a = randn(B, T, HP).mul_(0.1).to(dtype)
    b = randn(B, T, n, HP).mul_(0.1).to(dtype)
    if strided:
        q, k, v, a, b = (
            q[:, :, :H],
            k[:, :, :, :H],
            v[:, :, :, :H],
            a[..., :H],
            b[..., :H],
        )
    if pool_size is None:
        pool_size = B + 3
        if distinct_write:
            pool_size += B
        if scatter_tokens:
            pool_size += B * T
    if padded_pool:
        backing = randn(pool_size, 2 * H, V, K).mul_(0.1)
        pool = backing[:, ::2]
    else:
        backing = None
        pool = randn(pool_size, H, V, K).mul_(0.1)
    read = _slots(B, pool_size, exclude=None, generator=generator)
    write = (
        _slots(B, pool_size, exclude=read, generator=generator)
        if distinct_write
        else None
    )
    ssm = None
    if scatter_tokens:
        taken = read if write is None else torch.cat([read, write])
        ssm = _slots(B * T, pool_size, exclude=taken, generator=generator).view(B, T)
    return Case(
        B=B,
        T=T,
        H=H,
        n=n,
        q=q,
        k=k,
        v=v,
        a=a,
        b=b,
        A_log=randn(H).mul_(0.1),
        dt_bias=randn(H).mul_(0.1),
        pool=pool,
        backing=backing,
        backing_before=backing.clone() if backing is not None else None,
        pool_before=pool.clone(),
        read=read,
        write=write,
        ssm=ssm,
        scale=scale,
        output=None,
        disable_state_update=disable_state_update,
        use_qk_l2norm=use_qk_l2norm,
    )


# ---------------------------------------------------------------------------
# FP64 reference
# ---------------------------------------------------------------------------


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    return x / torch.sqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def _skipped_rows(read: torch.Tensor) -> torch.Tensor:
    """Batch rows the kernel skips entirely: a negative read slot.

    Such a row computes nothing, stores no output (its output row keeps the caller's value; the reference
    reports zeros for it), writes no state back and scatters nothing, whatever its other slot tables say.
    """

    return read < 0


def _reference(c: Case) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """FP64 token-serial GDP over the case: ``(output, pool_after, touched_slots)``.

    Inputs enter at their BF16 / FP16 values; the state is rounded to FP32 after each real token, as the kernel
    keeps it.  Rows in :func:`_skipped_rows` produce zeros and touch nothing.
    """

    q, k, v, a, b = (t.double() for t in (c.q, c.k, c.v, c.a, c.b))
    A_log, dt_bias = c.A_log.double(), c.dt_bias.double()
    B, T, H, _ = q.shape
    n = k.shape[2]
    scale = K**-0.5 if c.scale is None else float(c.scale)
    read = c.read.long()
    skipped = _skipped_rows(read)
    active = ~skipped
    S = c.pool_before[read.clamp_min(0)].double()  # [B, H, V, K]
    out = torch.zeros(B, T, H, V, dtype=torch.float64, device=q.device)
    after = c.pool_before.double().clone()
    touched = []
    decay = torch.exp(A_log)
    for t in range(T):
        alpha = torch.exp(-decay * F.softplus(a[:, t] + dt_bias))  # [B, H]
        S = S * alpha[..., None, None]
        for j in range(n):
            kq = _l2norm(k[:, t, j]) if c.use_qk_l2norm else k[:, t, j]
            beta = torch.sigmoid(b[:, t, j])
            u = torch.einsum("bhvk,bhk->bhv", S, kq)
            S = S + torch.einsum(
                "bhv,bhk->bhvk", beta[..., None] * (v[:, t, j] - u), kq
            )
        S = S.float().double()
        qn = _l2norm(q[:, t]) if c.use_qk_l2norm else q[:, t]
        out[:, t] = scale * torch.einsum("bhvk,bhk->bhv", S, qn)
        if c.ssm is not None:
            target = c.ssm[:, t].long()
            mask = active & (target >= 0)
            after[target[mask]] = S[mask]
            touched.append(target[mask])
    out[skipped] = 0
    if c.ssm is None and not c.disable_state_update:
        write = read if c.write is None else c.write.long()
        mask = active & (write >= 0)
        after[write[mask]] = S[mask]
        touched.append(write[mask])
    touched_slots = torch.cat(touched) if touched else read.new_empty(0)
    return out, after, torch.unique(touched_slots)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, what: str) -> None:
    torch.testing.assert_close(
        actual.float(),
        expected.float(),
        atol=ATOL,
        rtol=RTOL,
        msg=lambda message: f"{what}: {message}",
    )


def _run(c: Case) -> tuple[torch.Tensor, torch.Tensor]:
    args, kwargs = c.arguments()
    out, pool = cake_gated_delta_product_mtp(*args, **kwargs)
    assert pool is c.pool
    return out, pool


def _check(c: Case, out: torch.Tensor, *, what: str = "") -> None:
    """Compare one finished call with the reference: active output rows, touched slots (tolerance) and the
    untouched slots (bitwise).  Skipped rows must keep the caller's output (``SENTINEL`` when one was given,
    zeros otherwise)."""

    what = what or f"B={c.B} T={c.T} H={c.H} n={c.n}"
    expected, pool_expected, touched = _reference(c)
    active = ~_skipped_rows(c.read.long())
    assert out.shape == expected.shape and out.dtype == (
        c.output.dtype if c.output is not None else c.q.dtype
    )
    _assert_close(out[active], expected[active], f"{what}: output")
    if (~active).any():
        kept = SENTINEL if c.output is not None else 0.0
        assert torch.equal(out[~active], torch.full_like(out[~active], kept)), (
            f"{what}: padding rows"
        )
    pool_size = c.pool.shape[0]
    untouched = torch.ones(pool_size, dtype=torch.bool, device=c.pool.device)
    untouched[touched] = False
    assert torch.equal(c.pool[untouched], c.pool_before[untouched]), (
        f"{what}: untouched slots changed"
    )
    if touched.numel():
        _assert_close(c.pool[touched], pool_expected[touched], f"{what}: written slots")
    if c.backing is not None:
        assert torch.equal(c.backing[:, 1::2], c.backing_before[:, 1::2]), (
            f"{what}: pool padding changed"
        )


def _matrix() -> list:
    pairs = tuple(itertools.product(HEADS, BATCHES))
    cases = []
    for i, (n, T) in enumerate(itertools.product(NUM_HOUSEHOLDER, TOKENS)):
        for j in range(3):
            H, B = pairs[((i * 3 + j) * 11) % len(pairs)]
            cases.append(pytest.param(n, T, H, B, id=f"n{n}-T{T}-H{H}-B{B}"))
    return cases


# ---------------------------------------------------------------------------
# Correctness over the domain
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n,T,H,B", _matrix())
def test_decode_matrix(n: int, T: int, H: int, B: int) -> None:
    """Householder counts x token counts over cycling head / batch pairs (every H and B value appears)."""
    _skip_unless_route(n, T)
    c = _make(B, T, H, n, generator=_generator(1, n, T, H, B))
    out, _ = _run(c)
    _check(c, out)


@pytest.mark.parametrize("H", HEADS)
def test_head_counts(H: int) -> None:
    n = NUM_HOUSEHOLDER[HEADS.index(H) % len(NUM_HOUSEHOLDER)]
    _skip_unless_route(n, 4)
    c = _make(7, 4, H, n, generator=_generator(2, H))
    out, _ = _run(c)
    _check(c, out)


@pytest.mark.parametrize("B", BATCHES)
def test_batch_sizes(B: int) -> None:
    n = NUM_HOUSEHOLDER[BATCHES.index(B) % len(NUM_HOUSEHOLDER)]
    T = (1, 16, 2, 8)[BATCHES.index(B) % 4]
    _skip_unless_route(n, T)
    c = _make(B, T, 24, n, generator=_generator(3, B))
    out, _ = _run(c)
    _check(c, out)


@pytest.mark.parametrize(
    "H,B,T",
    [
        pytest.param(H, B, T, id=f"dec_H{H}_B{B}_T{T}")
        for H in PERF_HEADS
        for B in PERF_BATCHES
        for T in PERF_TOKENS
    ],
)
def test_decode_rows_of_record(H: int, B: int, T: int) -> None:
    """The 48 decode shapes of record (num_householder = 3), each a full correctness case."""
    _skip_unless_route(3, T)
    c = _make(B, T, H, 3, generator=_generator(4, H, B, T))
    out, _ = _run(c)
    _check(c, out)


# ---------------------------------------------------------------------------
# Slot tables
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("T", (1, 8))
@pytest.mark.parametrize("n", (1, 3))
def test_distinct_write_slots(n: int, T: int) -> None:
    """``output_state_indices`` apart from the read slots: the read slots stay bitwise, the write slots receive
    the final states."""
    _skip_unless_route(n, T)
    c = _make(16, T, 12, n, generator=_generator(5, n, T), distinct_write=True)
    out, _ = _run(c)
    _check(c, out)
    assert torch.equal(c.pool[c.read.long()], c.pool_before[c.read.long()])


@pytest.mark.parametrize("provided_output", (False, True))
@pytest.mark.parametrize("T", (1, 4))
def test_padding_rows_are_skipped(T: int, provided_output: bool) -> None:
    """A negative read slot skips the whole row: its output row keeps the caller's value (zeros when the output
    is allocated here), no slot is written for it even when its write slot is valid, and the other rows are
    unaffected."""
    _skip_unless_route(3, T)
    c = _make(
        4, T, 6, 3, generator=_generator(6, T, int(provided_output)), pool_size=12
    )
    c.read = torch.tensor((8, 3, -1, 1), device=_device(), dtype=torch.int32)
    c.write = torch.tensor((2, 4, 5, 6), device=_device(), dtype=torch.int32)
    if provided_output:
        c.output = torch.full(
            (4, T, 6, V), SENTINEL, device=_device(), dtype=torch.bfloat16
        )
    out, _ = _run(c)
    if provided_output:
        assert out is c.output
    _check(c, out)
    assert torch.equal(c.pool[5], c.pool_before[5])
    assert torch.equal(c.pool[8], c.pool_before[8]) and torch.equal(
        c.pool[3], c.pool_before[3]
    )


def test_negative_write_slot_skips_the_writeback_only() -> None:
    """A valid read slot with a negative write slot: the output is computed and stored, no slot is written."""
    _skip_unless_route(2, 3)
    c = _make(5, 3, 10, 2, generator=_generator(7), pool_size=16)
    c.write = c.read.clone()
    c.write[1] = -1
    c.write[4] = -1
    c.output = torch.full(
        (5, 3, 10, V), SENTINEL, device=_device(), dtype=torch.bfloat16
    )
    out, _ = _run(c)
    _check(c, out)
    assert not torch.equal(out[1], torch.full_like(out[1], SENTINEL))
    assert torch.equal(c.pool[c.read[1].long()], c.pool_before[c.read[1].long()])


@pytest.mark.parametrize("B", (1, 3, 16))
@pytest.mark.parametrize("T", (1, 2, 5, 8, 16))
def test_ssm_state_scatter(T: int, B: int) -> None:
    """``ssm_state_indices``: the state after every real token goes to its slot, holes (-1) skip that token's
    write, the last token may target the row's own read slot, and there is no end-of-sequence writeback to the
    write slot (checked through a distinct write table that must stay untouched)."""
    n = (3, 1, 4, 2, 3)[(1, 2, 5, 8, 16).index(T)]
    _skip_unless_route(n, T)
    c = _make(
        B,
        T,
        12,
        n,
        generator=_generator(8, T, B),
        distinct_write=True,
        scatter_tokens=True,
    )
    ssm = c.ssm.clone()
    if T > 1:
        ssm[:, 1] = -1
    ssm[0, T - 1] = c.read[0]
    c.ssm = ssm
    out, _ = _run(c)
    _check(c, out)
    assert torch.equal(c.pool[c.write.long()], c.pool_before[c.write.long()])


def test_ssm_rejects_disable_state_update() -> None:
    _skip_unless_route(3, 2)
    c = _make(
        2,
        2,
        6,
        3,
        generator=_generator(9),
        scatter_tokens=True,
        disable_state_update=True,
    )
    with pytest.raises(ValueError):
        _run(c)


@pytest.mark.parametrize("T", (1, 8))
def test_disable_state_update(T: int) -> None:
    _skip_unless_route(3, T)
    c = _make(7, T, 20, 3, generator=_generator(10, T), disable_state_update=True)
    out, _ = _run(c)
    _check(c, out)
    assert torch.equal(c.pool, c.pool_before)


def test_disable_state_update_defaults_to_false() -> None:
    _skip_unless_route(1, 1)
    c = _make(3, 1, 6, 1, generator=_generator(11))
    c.disable_state_update = None
    out, _ = _run(c)
    _check(c, out)
    assert not torch.equal(c.pool, c.pool_before)


# ---------------------------------------------------------------------------
# Options and layouts
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scale", (None, 1.0, 0.0))
def test_scale(scale: Optional[float]) -> None:
    _skip_unless_route(3, 4)
    c = _make(3, 4, 12, 3, generator=_generator(12), scale=scale)
    out, _ = _run(c)
    _check(c, out)
    if scale == 0.0:
        assert torch.equal(out, torch.zeros_like(out))


@pytest.mark.parametrize("B,H,T", ((7, 12, 4), (16, 80, 1)))
def test_padded_pool(B: int, H: int, T: int) -> None:
    """A pool with padding between heads (a strided view of a wider buffer) is used in place."""
    _skip_unless_route(3, T)
    c = _make(B, T, H, 3, generator=_generator(13, B, H, T), padded_pool=True)
    assert not c.pool.is_contiguous()
    out, pool = _run(c)
    assert pool is c.pool
    _check(c, out)


@pytest.mark.parametrize("T", (1, 5))
def test_strided_inputs_and_output(T: int) -> None:
    """q / k / v / a / b as head slices of wider buffers and the output as a head slice of a wider buffer: read
    and written in place, the padding heads of the output buffer untouched."""
    _skip_unless_route(3, T)
    c = _make(7, T, 10, 3, generator=_generator(14, T), strided=True)
    assert (
        not c.q.is_contiguous() and not c.k.is_contiguous() and not c.a.is_contiguous()
    )
    buffer = torch.full((7, T, 12, V), SENTINEL, device=_device(), dtype=torch.bfloat16)
    c.output = buffer[:, :, :10]
    out, _ = _run(c)
    assert out is c.output
    _check(c, out)
    assert torch.equal(buffer[:, :, 10:], torch.full_like(buffer[:, :, 10:], SENTINEL))


@pytest.mark.parametrize("provided_output", (False, True))
def test_fp16_io(provided_output: bool) -> None:
    """FP16 inputs are converted for the kernel; the output comes back in FP16 (through a bfloat16 round trip
    of a provided buffer, whose padding rows keep their values)."""
    _skip_unless_route(2, 4)
    c = _make(
        4,
        4,
        6,
        2,
        generator=_generator(15, int(provided_output)),
        dtype=torch.float16,
        pool_size=10,
    )
    c.read[2] = -1
    if provided_output:
        c.output = torch.full(
            (4, 4, 6, V), SENTINEL, device=_device(), dtype=torch.float16
        )
    out, _ = _run(c)
    assert out.dtype == torch.float16
    if provided_output:
        assert out is c.output
    _check(c, out)


def test_bf16_output_with_strided_last_dimension_is_copied_back() -> None:
    """A bfloat16 output whose last dimension is not dense receives a copy of the kernel's output."""
    _skip_unless_route(3, 1)
    c = _make(3, 1, 6, 3, generator=_generator(16), pool_size=8)
    c.read[1] = -1
    buffer = torch.full(
        (3, 1, 6, 2 * V), SENTINEL, device=_device(), dtype=torch.bfloat16
    )
    c.output = buffer[..., ::2]
    out, _ = _run(c)
    assert out is c.output
    _check(c, out)
    assert torch.equal(buffer[..., 1::2], torch.full_like(buffer[..., 1::2], SENTINEL))


def test_int64_indices() -> None:
    _skip_unless_route(3, 2)
    c = _make(
        7, 2, 6, 3, generator=_generator(17), distinct_write=True, scatter_tokens=False
    )
    c.read = c.read.long()
    c.write = c.write.long()
    out, _ = _run(c)
    _check(c, out)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16))
def test_low_precision_dt_bias_is_accepted(dtype: torch.dtype) -> None:
    _skip_unless_route(3, 1)
    c = _make(3, 1, 6, 3, generator=_generator(18))
    c.dt_bias = c.dt_bias.to(dtype)
    out, _ = _run(c)
    _check(c, out)


def test_single_launch_with_float32_parameters() -> None:
    """bfloat16 activations, float32 ``A_log`` / ``dt_bias``, int32 slots, a bfloat16 output buffer and no scatter
    table: the call launches the kernel alone -- the stand-in scatter table is not filled and no parameter is
    cast (``torch.profiler`` counts the device events of one warmed-up call)."""
    from torch.profiler import ProfilerActivity, profile

    _skip_unless_route(3, 1)
    c = _make(4, 1, 6, 3, generator=_generator(20))
    assert (
        c.A_log.dtype == torch.float32
        and c.dt_bias.dtype == torch.float32
        and c.ssm is None
    )
    c.output = torch.full(
        (4, 1, 6, V), SENTINEL, device=_device(), dtype=torch.bfloat16
    )
    out, _ = _run(c)  # warm-up: module load, first-call work
    _check(c, out)
    torch.cuda.synchronize(_device())
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            _run(c)
            torch.cuda.synchronize(_device())
    except Exception as ex:  # noqa: BLE001 - profiler unavailable on this host
        pytest.skip(f"torch.profiler unavailable: {ex}")
    device_events = [
        e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA
    ]
    if not device_events:
        pytest.skip("torch.profiler captured no device events")
    names = [e.name for e in device_events]
    assert len(device_events) == 1, names
    assert "cake_gdp" in names[0], names


@pytest.mark.parametrize("T", (1, 4))
def test_use_qk_l2norm_off(T: int) -> None:
    """Without the in-kernel normalization the raw q / k enter the recurrence; a device without an exported
    ``_raw`` route rejects the call with ValueError."""
    if not loader.supported_num_householder(_arch(), False):
        c = _make(2, T, 6, 3, generator=_generator(19, T), use_qk_l2norm=False)
        with pytest.raises(ValueError):
            _run(c)
        pytest.skip("no exported use_qk_l2norm=False route on this device")
    _skip_unless_route(3, T, use_qk_l2norm=False)
    c = _make(3, T, 6, 3, generator=_generator(19, T), use_qk_l2norm=False)
    out, _ = _run(c)
    _check(c, out)


# ---------------------------------------------------------------------------
# Independence, reproducibility, program coverage
# ---------------------------------------------------------------------------


def test_batch_rows_are_independent() -> None:
    """Each row run alone gives the same bits as in the batch (output row and written slot)."""
    _skip_unless_route(3, 4)
    c = _make(5, 4, 12, 3, generator=_generator(20), distinct_write=True)
    c.read[3] = -1
    out, _ = _run(c)
    for row in range(5):
        single = Case(**{**c.__dict__})
        single.B = 1
        single.pool = c.pool_before.clone()
        single.pool_before = single.pool.clone()
        for name in ("q", "k", "v", "a", "b", "read", "write"):
            setattr(single, name, getattr(c, name)[row : row + 1])
        single.output = None
        out_single, _ = _run(single)
        assert torch.equal(out_single[0], out[row]), row
        slot = int(c.write[row])
        if int(c.read[row]) >= 0:
            assert torch.equal(single.pool[slot], c.pool[slot]), row


def test_launches_are_bitwise_reproducible() -> None:
    _skip_unless_route(3, 8)
    c = _make(16, 8, 24, 3, generator=_generator(21))
    results = []
    for _ in range(3):
        c.pool.copy_(c.pool_before)
        out, _ = _run(c)
        results.append((out.clone(), c.pool.clone()))
    for out, pool in results[1:]:
        assert torch.equal(out, results[0][0]) and torch.equal(pool, results[0][1])


def _program_cases() -> list:
    """One (n, T, B, H) selecting each exported program of the local architecture, from the registry alone."""
    if not _supported():
        return []
    arch = _arch()
    sms = loader.sm_count(_device().index or 0)
    cases = []
    for name, record in sorted(loader.MODULES.items()):
        if arch not in record["arches"] or record["role"] != "decode":
            continue  # the prefill records: tests/gdp/test_cake_gdp_prefill.py
        build = record["build"]
        n = build["num_householder"]
        T = build["t_steps"] if build["t_steps"] is not None else build["max_t_steps"]
        use_qk_l2norm = build.get("use_qk_l2norm", True)
        if (
            build["t_steps"] is None
            and loader.max_num_tokens(arch, n, use_qk_l2norm) > T
        ):
            T = loader.max_num_tokens(arch, n, use_qk_l2norm)
        # 27 x 24 = 648 lies in the T = 1 tail-wave rows (4 x sm, 5 x sm] of SM100 / SM103 on
        # 132-160 SMs (l8r4 since the band rule change); 257 x 80 covers the many-wave rows.
        for B, H in itertools.product((1, 3, 7, 16, 64, 257, 27), (6, 12, 20, 24, 80)):
            if build["t_steps"] is None:
                # The runtime program is the fallback: a T no static program owns selects it.
                static = {
                    loader.MODULES[other]["build"]["t_steps"]
                    for other in loader.route_modules(arch, n, use_qk_l2norm)
                }
                T = next(
                    t
                    for t in (T, 16, 2, 3, 5, 6, 7)
                    if t not in static and t <= build["max_t_steps"]
                )
            if (
                loader.selected_module(
                    arch, n, T, B * H, sm_count=sms, use_qk_l2norm=use_qk_l2norm
                )
                == name
            ):
                cases.append(pytest.param(name, n, T, B, H, use_qk_l2norm, id=name))
                break
        else:
            cases.append(pytest.param(name, n, T, 0, 0, use_qk_l2norm, id=name))
    return cases


@pytest.mark.parametrize("name,n,T,B,H,use_qk_l2norm", _program_cases())
def test_every_program_is_launched(
    name: str, n: int, T: int, B: int, H: int, use_qk_l2norm: bool
) -> None:
    """Every exported program of the local architecture is selected by some call of the domain and is correct."""
    assert B > 0, f"no (B, H) of the domain selects {name}"
    c = _make(
        B, T, H, n, generator=_generator(22, n, T, B, H), use_qk_l2norm=use_qk_l2norm
    )
    out, _ = _run(c)
    _check(c, out, what=name)


# ---------------------------------------------------------------------------
# Rejected arguments
# ---------------------------------------------------------------------------


def test_rejects_invalid_arguments() -> None:
    _skip_unless_route(3, 2)
    c = _make(3, 2, 6, 3, generator=_generator(23), pool_size=8)
    args, kwargs = c.arguments()

    def call(**changes):
        a = list(args)
        names = ("q", "k", "v", "pool", "read", "A_log", "a", "dt_bias", "b")
        kw = dict(kwargs)
        for key, value in changes.items():
            if key in names:
                a[names.index(key)] = value
            else:
                kw[key] = value
        return cake_gated_delta_product_mtp(*a, **kw)

    device = _device()
    with pytest.raises(ValueError):  # H != HV
        call(v=c.v[:, :, :, :5])
    with pytest.raises(ValueError):  # num_householder out of range
        call(
            k=torch.cat((c.k, c.k[:, :, :2]), 2),
            v=torch.cat((c.v, c.v[:, :, :2]), 2),
            b=torch.cat((c.b, c.b[:, :, :2]), 2),
        )
    with pytest.raises(ValueError):  # K != 128
        call(q=c.q[..., :64])
    with pytest.raises(ValueError):  # k not in q.dtype
        call(k=c.k.float())
    with pytest.raises(ValueError):  # last dimension not dense
        call(q=c.q.transpose(2, 3).contiguous().transpose(2, 3))
    with pytest.raises(ValueError):  # pool dtype
        call(pool=c.pool.half())
    with pytest.raises(ValueError):  # pool head count
        call(pool=c.pool[:, :5])
    with pytest.raises(ValueError):  # pool rows not dense
        call(pool=torch.empty(8, 6, V, 2 * K, device=device)[..., ::2])
    numel = 8 * 6 * V * K
    misaligned = torch.empty(numel + 4, device=device)[1 : numel + 1].view(8, 6, V, K)
    with pytest.raises(ValueError):  # pool base not 16-byte aligned
        call(pool=misaligned)
    with pytest.raises(ValueError):  # A_log dtype
        call(A_log=c.A_log.to(torch.bfloat16))
    with pytest.raises(ValueError):  # index dtype
        call(read=c.read.to(torch.int16))
    with pytest.raises(ValueError):  # index shape
        call(read=c.read[:2])
    with pytest.raises(ValueError):  # ssm dtype
        call(ssm_state_indices=c.read.view(3, 1).expand(3, 2).long())
    with pytest.raises(ValueError):  # ssm shape
        call(ssm_state_indices=c.read.view(3, 1).contiguous())
    with pytest.raises(ValueError):  # output shape
        call(output=torch.empty(3, 2, 6, 2 * V, device=device, dtype=torch.bfloat16))
    with pytest.raises(ValueError):  # output dtype
        call(output=torch.empty(3, 2, 6, V, device=device, dtype=torch.float32))
    with pytest.raises(ValueError):  # T beyond the exported maximum
        T = loader.max_num_tokens(_arch(), 3) + 1
        big = _make(1, T, 6, 3, generator=_generator(24))
        _run(big)
    with pytest.raises(ValueError):  # CPU tensors
        call(q=c.q.cpu())


def test_is_supported_reports_the_routes() -> None:
    arch = _arch()
    assert is_cake_gdp_supported(_device())
    assert not is_cake_gdp_supported(torch.device("cpu"))
    for n in NUM_HOUSEHOLDER:
        assert is_cake_gdp_supported(_device(), n) == (
            n in loader.supported_num_householder(arch)
        )
        if n in loader.supported_num_householder(arch):
            bound = loader.max_num_tokens(arch, n)
            assert is_cake_gdp_supported(_device(), n, bound)
            assert not is_cake_gdp_supported(_device(), n, bound + 1)
            assert not is_cake_gdp_supported(_device(), n, 0)
    assert is_cake_gdp_supported(_device(), 5) is False


# ---------------------------------------------------------------------------
# Package contract
# ---------------------------------------------------------------------------


def test_registry_records_reference_the_build_manifest() -> None:
    """Every ``MODULES`` record names its ``build.json`` library and nothing to compile.

    The manifest (schema ``cake-build/1``) carries exactly one entry per (library,
    architecture) the registry references, each with the kernel and binding
    sources under ``csrc/`` of this package, their digests and a closure digest;
    the registry itself records the host plan only.
    """

    manifest = cake_build.manifest()
    assert manifest["schema"] == "cake-build/1"
    assert manifest["package"] == pkg.__name__
    libraries = cake_build.libraries_by_name()
    package = Path(cake_build.__file__).parent
    referenced = set()
    for name, record in loader.MODULES.items():
        assert record["library"] == record["kernel"], name
        assert not {"sources", "source_sha256", "compile_flags", "cache_names"} & set(
            record
        ), name
        assert set(record["arches"]) <= set(manifest["arch_flags"]), name
        assert record["role"] in ("decode", "prefill"), name
        if record["role"] == "decode":
            assert record["build"]["num_householder"] in NUM_HOUSEHOLDER, name
        # The launch block is the built kernel's (the exporter writes it): the shape only is checked here.
        block = list(record["launch"]["block"])
        assert len(block) == 3, name
        assert all(isinstance(x, int) and x > 0 for x in block), name
        for arch in record["arches"]:
            library = loader.library_name(name, arch)
            referenced.add(library)
            entry = libraries[library]
            assert entry["arch"] == arch
            assert entry["entry"] == "run"
            assert entry["sources"] == [
                f"csrc/{record['library']}_kernel.cu",
                f"csrc/{record['library']}_binding.cu",
            ]
            assert set(entry["sha256"]) == set(entry["sources"])
            for source in entry["sources"]:
                assert (package / source).is_file(), source
            assert re.fullmatch(r"[0-9a-f]{64}", entry["closure_sha256"])
            # Every decode entry compiles with fast math (the prefill records carry their own flags,
            # checked by tests/gdp/test_cake_gdp_prefill.py).
            if record["role"] == "decode":
                assert "--use_fast_math" in entry["nvcc_flags"]
    assert referenced == set(libraries)
    assert (package / "csrc" / "cake_ffi_checks.h").is_file()


# Band edges and wave bounds of the route tables: B x HV probes for selected_module.
_STATE_HEAD_PROBES = (
    1,
    8,
    160,
    161,
    296,
    304,
    305,
    320,
    592,
    593,
    608,
    609,
    740,
    741,
    760,
    761,
    4096,
    # the two-head edge 32 x sm and the many-wave edge 64 x sm, -1 / 0 / +1, for
    # 132 / 148 / 152 / 160 SMs (every probe is tried on every SM count)
    4223,
    4224,
    4225,
    4735,
    4736,
    4737,
    4863,
    4864,
    4865,
    5119,
    5120,
    5121,
    8447,
    8448,
    8449,
    9471,
    9472,
    9473,
    9727,
    9728,
    9729,
    10239,
    10240,
    10241,
    20480,
)


def test_route_tables_are_consistent() -> None:
    """Every route name lists exported modules of its architecture; static programs of a route own disjoint
    state-head bands per token count; ``ARCH_BY_CAPABILITY`` names the supported capabilities only."""

    # The capability map names the decode capabilities and may name further ones of the prefill role
    # (tests/gdp/test_cake_gdp_prefill.py); exactly the decode capabilities carry a decode route.
    assert set(SUPPORTED_CAPABILITIES) <= set(loader.ARCH_BY_CAPABILITY)
    assert {
        capability
        for capability, arch in loader.ARCH_BY_CAPABILITY.items()
        if any(key.startswith("decode_") for key in loader.ROUTES.get(arch, {}))
    } == set(SUPPORTED_CAPABILITIES)
    assert set(loader.ARCH_BY_CAPABILITY.values()) <= set(loader.ARCHES)
    for arch, routes in loader.ROUTES.items():
        assert arch in loader.ARCHES
        for key, names in routes.items():
            if key == loader.PREFILL_ROUTE:
                continue  # the prefill route: tests/gdp/test_cake_gdp_prefill.py
            assert re.fullmatch(r"decode_nh[1-4](?:_raw)?", key), key
            n = int(key[len("decode_nh")])
            for name in names:
                record = loader.MODULES[name]
                assert arch in record["arches"], (arch, name)
                assert record["build"]["num_householder"] == n, (key, name)
                assert record["build"]["use_qk_l2norm"] == (not key.endswith("_raw")), (
                    key,
                    name,
                )
            for T in range(
                1, loader.max_num_tokens(arch, n, not key.endswith("_raw")) + 1
            ):
                for sms in (132, 148, 152, 160):
                    for state_heads in _STATE_HEAD_PROBES:
                        loader.selected_module(
                            arch,
                            n,
                            T,
                            state_heads,
                            sm_count=sms,
                            use_qk_l2norm=not key.endswith("_raw"),
                        )


@pytest.mark.parametrize("field", ("q", "k"))
def test_misaligned_q_k_views_are_refused(field: str) -> None:
    """q and k are read in 8-byte pieces: a view whose base address or outer strides
    are not multiples of four elements is refused before anything is launched; one
    whose outer strides are multiples of four elements is read in place."""
    _skip_unless_route(2, 1)
    c = _make(3, 1, 5, 2, generator=_generator(31, 1 if field == "q" else 2))
    dense = getattr(c, field)
    flat = torch.empty(dense.numel() + 1, dtype=dense.dtype, device=dense.device)
    shifted = flat[1:].view(dense.shape)  # base offset of one element: 2 bytes
    shifted.copy_(dense)
    with pytest.raises(ValueError, match="aligned"):
        _run(Case(**{**c.__dict__, field: shifted}))
    wide = (*dense.shape[:-1], K + 2)
    odd = torch.empty(wide, dtype=dense.dtype, device=dense.device)[..., :K]
    odd.copy_(dense)  # outer strides of 130 elements: 260 bytes
    with pytest.raises(ValueError, match="aligned"):
        _run(Case(**{**c.__dict__, field: odd}))
    assert torch.equal(c.pool, c.pool_before)
    wide = (*dense.shape[:-1], K + 4)
    even = torch.empty(wide, dtype=dense.dtype, device=dense.device)[..., :K]
    even.copy_(dense)  # outer strides of 132 elements: 264 bytes, read in place
    c = Case(**{**c.__dict__, field: even})
    out, _ = _run(c)
    _check(c, out, what=f"{field} as a view with strides of 132 elements")


def test_misaligned_bf16_output_is_copied_back() -> None:
    """A bfloat16 output whose base address is not 8-byte aligned (here: one element
    into a wider buffer) receives a copy of the kernel's output; the buffer around it
    keeps its values."""
    _skip_unless_route(3, 1)
    c = _make(3, 1, 6, 3, generator=_generator(32), pool_size=8)
    c.read[1] = -1
    buffer = torch.full(
        (3 * 6 * V + 2,), SENTINEL, device=_device(), dtype=torch.bfloat16
    )
    c.output = buffer[1:-1].view(3, 1, 6, V)
    out, _ = _run(c)
    assert out is c.output
    _check(c, out)
    assert buffer[0].item() == SENTINEL and buffer[-1].item() == SENTINEL


def test_alignment_constants_match_the_kernel_sources() -> None:
    """The host checks' alignment constants are the widest vector accesses the shipped
    kernel sources make on q, k and the output (``reinterpret_cast`` on those pointers);
    v, a and b are read element by element (at most one 2-byte element per access)."""
    import flashinfer.cake_gdp_backend as backend

    widths = {
        "uint16_t": 2,
        "unsigned": 4,
        "__nv_bfloat162": 4,
        "int2": 8,
        "uint2": 8,
        "uint4": 16,
        "float4": 16,
    }
    access = re.compile(
        r"reinterpret_cast<(?:const )?(\w+)\s*\*>\("
        r"(?:&\(\(\w+\*\)\()?(q|k|v|a|b|out)\b"
    )
    kernels = sorted((Path(backend.__file__).parent / "csrc").glob("*_kernel.cu"))
    assert kernels
    widest: dict[str, int] = {}
    for kernel in kernels:
        for cast, name in access.findall(kernel.read_text()):
            widest[name] = max(widest.get(name, 0), widths[cast])
    assert widest["q"] == widest["k"] == backend._QK_ALIGNMENT_BYTES == 8
    assert widest["out"] == backend._OUTPUT_ALIGNMENT_BYTES == 8
    assert all(widest.get(name, 0) <= 2 for name in ("v", "a", "b")), widest


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="needs a second GPU")
def test_runs_on_a_non_current_device_and_its_current_stream() -> None:
    """Inputs on cuda:1 whose current stream is a non-default stream, called while cuda:0
    is the current device: the launch binds cuda:1's current stream, so the copy that
    makes q valid (queued on that stream behind a busy loop) is ordered before the
    kernel."""
    _skip_unless_route(3, 2)
    other = torch.device("cuda:1")
    if tuple(torch.cuda.get_device_capability(other)) not in SUPPORTED_CAPABILITIES:
        pytest.skip("cuda:1 is not a supported GPU")
    source = _make(5, 2, 6, 3, generator=_generator(33), pool_size=16)
    c = Case(
        **{
            key: value.to(other) if isinstance(value, torch.Tensor) else value
            for key, value in source.__dict__.items()
        }
    )
    q_values = c.q.clone()
    c.q.zero_()
    stream = torch.cuda.Stream(device=other)
    # Make `stream` current on cuda:1 (this also makes cuda:1 the current device),
    # then step back to cuda:0: cuda:1 keeps `stream` as its current stream.
    torch.cuda.set_stream(stream)
    torch.cuda.set_device(0)
    try:
        assert torch.cuda.current_device() == 0
        assert torch.cuda.current_stream(other) == stream
        busy = torch.randn(4096, 4096, device=other)  # cuda:1 work runs on `stream`
        for _ in range(8):
            busy = busy @ busy / 4096.0
        c.q.copy_(q_values)
        out, _ = _run(c)
        assert torch.cuda.current_device() == 0
        stream.synchronize()
    finally:
        torch.cuda.set_device(other)
        torch.cuda.set_stream(torch.cuda.default_stream(other))
        torch.cuda.set_device(0)
    assert out.device == other and busy.isfinite().all()
    _check(c, out, what="cuda:1 inputs on a non-default stream while cuda:0 is current")


def _static_pick(
    arch: str, n: int, T: int, state_heads: int, sm: int
) -> tuple[str, str | None]:
    """(body, variant) of the static program the band rule selects, or ("runtime", None)."""
    name = loader.selected_module(arch, n, T, state_heads, sm_count=sm)
    record = loader.MODULES[name]
    if record["build"]["t_steps"] != T:
        return ("runtime", None)
    return (record["body"], record["build"]["variant"])


@pytest.mark.parametrize("sm", (148, 152))
def test_t1_band_edges_follow_the_sm_count(sm: int) -> None:
    """T = 1, three Householder steps, on `sm` SMs: the row-partition program up to 160
    state heads; on SM100 / SM103 the one-wave program l8r4p takes (160, 4 x sm], l8r4
    takes (4 x sm, 64 x sm] (the tail-wave rows (4 x sm, 5 x sm] included: the
    one-wave-of-5 band is off) and l8r4w5 takes every count above 64 x sm, while SM90
    ships neither of the two and keeps l8r4 above 160.  The choice depends on
    (arch, T, B x HV, sm_count) only."""
    assert set(loader.ROUTES) == {"sm_90a", "sm_100a", "sm_103a"}
    for arch in loader.ROUTES:
        one_wave = "l8r4" if arch == "sm_90a" else "l8r4p"
        many_wave = "l8r4" if arch == "sm_90a" else "l8r4w5"
        expected = {
            160: ("splitv", "splitv8_tile8"),
            161: ("lanerow", one_wave),
            4 * sm: ("lanerow", one_wave),
            4 * sm + 1: ("lanerow", "l8r4"),
            5 * sm: ("lanerow", "l8r4"),
            5 * sm + 1: ("lanerow", "l8r4"),
            32 * sm - 1: ("lanerow", "l8r4"),
            32 * sm: ("lanerow", "l8r4"),
            64 * sm: ("lanerow", "l8r4"),
            64 * sm + 1: ("lanerow", many_wave),
            20480: ("lanerow", many_wave),
        }
        for state_heads, want in expected.items():
            assert _static_pick(arch, 3, 1, state_heads, sm) == want, (
                arch,
                state_heads,
                sm,
            )
            for n in (1, 2, 4):  # no static program: the runtime one of n
                assert _static_pick(arch, n, 1, state_heads, sm)[0] == "runtime", (
                    arch,
                    n,
                    state_heads,
                )


@pytest.mark.parametrize("sm", (132, 148, 152, 160))
def test_t4_band_follows_the_architecture(sm: int) -> None:
    """T = 4, three Householder steps: SM90 runs the one-CTA-per-SM program l4r2 up to
    `sm` state heads and l8r4 above it; SM100 / SM103 run l8r4 on every state-head count
    and ship no l4r2 program (and SM90 no l8r4p / l8r4w5 program)."""
    for arch in loader.ROUTES:
        for state_heads in (1, 24, sm, sm + 1, 4 * sm, 20480):
            one_cta_wave = arch == "sm_90a" and state_heads <= sm
            want = ("lanerow", "l4r2" if one_cta_wave else "l8r4")
            assert _static_pick(arch, 3, 4, state_heads, sm) == want, (
                arch,
                state_heads,
                sm,
            )
        variants = {
            loader.MODULES[name]["build"]["variant"]
            for key, names in loader.ROUTES[arch].items()
            if key
            != loader.PREFILL_ROUTE  # the prefill route: tests/gdp/test_cake_gdp_prefill.py
            for name in names
        }
        assert ("l4r2" in variants) == (arch == "sm_90a"), (arch, sorted(variants))
        assert ("l8r4w5" in variants) == (arch != "sm_90a"), (arch, sorted(variants))
        assert ("l8r4p" in variants) == (arch != "sm_90a"), (arch, sorted(variants))
        assert "l8r4x2" not in variants, (
            arch,
            sorted(variants),
        )  # the two-heads-per-CTA program ships nowhere


def test_t1_static_programs_partition_every_sm_count() -> None:
    """Exactly one T = 1 static program admits every state-head count on every SM count
    the kernel may meet (`selected_module` raises when two admit), so the runtime program is
    never the T = 1 choice of three Householder steps."""
    for arch in loader.ROUTES:
        for sm in (132, 148, 152, 160):
            for state_heads in (*range(1, 1300), 1536, 4096, 20480):
                body, _variant = _static_pick(arch, 3, 1, state_heads, sm)
                assert body != "runtime", (arch, state_heads, sm)
