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
token, FP32 state pool; SM90 / SM100 / SM103 / SM107; FP16 state pool on SM100 /
SM103 / SM107).

Every output row and every pool slot the call may touch is compared with an
FP64 token-serial reference of the same math (inputs at their BF16 values,
the state rounded to FP32 after each real token) at ``atol = 1e-2``,
``rtol = 5e-3`` with no mismatch allowed; every slot the call must not touch
is checked bitwise.  The matrix covers ``num_householder`` 1 to 4, the head
counts 1 to 80, batches 1 to 257 and token counts 1 to 16 (the 144 decode shapes
of record among them), distinct write slots, padding rows (negative read
slots, whose output rows keep the caller's values), rows without writeback,
per-token state scatter with holes, ``disable_state_update``, the scale
values, a padded pool, strided inputs and outputs, FP16 I/O, int64 indices,
the rejected arguments, batch-row independence, bitwise reproducibility and
one launch of every exported program.

An FP16 pool runs the twin of the FP32 program with the same arithmetic, so each
FP16 case also runs the same call over an FP32 pool holding the FP16 pool's
values: the output must equal it bit for bit, and every stored FP16 element must
be a rounding of the FP32 twin's stored element -- equal to its round to nearest
even without a seed, one of its two FP16 neighbours with one (itself when it is
exact).  The stored elements of at most ``MODEL_MAX_HEADS`` state heads per call
(all of them in smaller calls, a deterministic sample in larger ones) must also
equal the rounding of an independent model: round to nearest even, or
Philox-4x32 random bits added below the FP16 precision and truncated, as PTX
``cvt.rs`` does (key = the seed; one draw per eight elements of a state row,
words 0 / 1 for columns k .. k + 3 and words 2 / 3 for k + 64 .. k + 67, k a
multiple of 4 below 64, 13 bits per element; counter = the element offset, from
the pool view's base, of column k of the state row in the batch row's read
slot).  Every FP16 program runs three call kinds in each rounding, at its own
token count and again at ``T = 1`` when one-token calls select it: a writeback
to the read slot; a writeback to distinct write slots with one row skipping it
(``-1``); and per-token snapshots with a ``-1`` before the last token (from
``T = 2``), a ``-1`` at the last token and a snapshot into the row's own read
slot.  A call kind and rounding form a launch class, and a program runs the
classes it owns: a ``snapshot_noise`` sibling (the same body and variant,
drawing a launch's random bits once) owns the stochastic snapshot calls of its
cell, a ``draw_share`` sibling (the same body and variant, computing each draw
once and sharing it by a lane shuffle) the stochastic calls of its cell, and
their plain program the other classes.  The test records the program
each call launches and the store paths it takes, and fails unless every
program took every path of its classes and every class of a band is owned.  These calls use an FP16 pool with one
guard slot on each side; after each call the slots it must not touch and both
guard slots must be unchanged bit for bit.  The FP16-pool tests skip on SM90,
which has no FP16-pool programs; there
``test_fp16_pool_is_refused_without_fp16_programs`` checks that an FP16 pool is
refused.

This suite exercises the backend package alone: it names the package once,
through ``tests.cake.test_helpers.cake_package``, and imports nothing else from
flashinfer, so a build that vendors the package under another import path
runs it unchanged after pointing that table entry at its package.  The
``flashinfer.cake_gdp`` facade and the ``flashinfer`` top-level exposure are
covered by ``test_cake_gdp_decode_api.py``.
"""

from __future__ import annotations

import hashlib
import itertools
import re
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tests.cake.test_helpers.cake_package import cake_package

pkg = cake_package("cake_gdp")
cake_build = cake_package("cake_gdp", "cake_build")
loader = cake_package("cake_gdp", "cake_jit")
cake_gated_delta_product_mtp = pkg.cake_gated_delta_product_mtp
is_cake_gdp_supported = pkg.is_cake_gdp_supported

SUPPORTED_CAPABILITIES = ((9, 0), (10, 0), (10, 3), (10, 7))
K = 128
V = 64
ATOL = 1e-2
RTOL = 5e-3
SEED = 760
NUM_HOUSEHOLDER = (1, 2, 3, 4)
HEADS = (1, 2, 3, 6, 10, 12, 20, 24, 40, 80)
BATCHES = (1, 2, 3, 7, 16, 64, 257)
TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 16)
# The 144 decode shapes of record (num_householder = 3): every performance row is a correctness case.
PERF_HEADS = (80, 20, 10, 24)
PERF_BATCHES = (1, 16, 64, 256)
PERF_TOKENS = (1, 2, 3, 4, 5, 6, 7, 8, 16)
SENTINEL = 7.5
# Stochastic rounding of an FP16 pool: a seed above 2^32 (both key words count), the default round count.
SR_SEED = 0x1234_5678_9ABC_DEF0
SR_ROUNDS = 10
MASK32 = 0xFFFF_FFFF
PHILOX_M0, PHILOX_M1 = 0xD2511F53, 0xCD9E8D57
PHILOX_W0, PHILOX_W1 = 0x9E3779B9, 0xBB67AE85
# Random123's known answer of Philox-4x32-10 at the zero key and counter.
PHILOX_KAT = (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)
# At most this many stored state heads of one call are compared with the rounding model (a deterministic sample of
# larger calls); every stored element is still checked against the FP32 twin and every untouched slot bitwise.
MODEL_MAX_HEADS = 4096
# State heads per evaluation of the rounding model (bounds its host memory; the sample and the result are unchanged).
MODEL_CHUNK_HEADS = 512
# Stored slots per gather of the all-slot rounding check (bounds its device memory).
CHECK_SLOTS = 64
# The SM counts the band rules are checked on: 132 / 148 / 152 / 160 (SM90 / SM100 / SM103 parts) and
# 212 / 216 (SM107: VR200 / R200).
SM_COUNTS = (132, 148, 152, 160, 212, 216)


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
    reason="cake_gdp needs an SM90 / SM100 / SM103 / SM107 device with exported routes",
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


def _skip_unless_route(
    n: int,
    T: int,
    use_qk_l2norm: bool = True,
    state_dtype: torch.dtype = torch.float32,
) -> None:
    if not loader.route_applies(
        device_capability=_capability(),
        num_householder=n,
        num_tokens=T,
        use_qk_l2norm=use_qk_l2norm,
        state_dtype=state_dtype,
    ):
        pytest.skip(
            f"num_householder={n}, T={T}, use_qk_l2norm={use_qk_l2norm} has no exported program for a "
            f"{state_dtype} state pool on this device"
        )


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


class Case:
    """One decode call: its inputs, the pool before the call and the slot tables."""

    def __init__(self, **fields):
        self.__dict__.update(fields)

    def arguments(self) -> tuple[tuple, dict]:
        kwargs = dict(
            scale=self.scale,
            output=self.output,
            ssm_state_indices=self.ssm,
            disable_state_update=self.disable_state_update,
            use_qk_l2norm=self.use_qk_l2norm,
            output_state_indices=self.write,
        )
        if getattr(self, "rand_seed", None) is not None:
            kwargs.update(rand_seed=self.rand_seed, philox_rounds=self.philox_rounds)
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
            kwargs,
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
    guard_slots: bool = False,
) -> Case:
    """Random inputs at the scale of the kernel's own tests: q / k / v ~ N(0, 1), ``a``, ``b``, ``A_log``,
    ``dt_bias`` and the pool at ``0.1 x N(0, 1)``; the pool holds unused slots and the read slots are distinct.
    ``guard_slots``: the FP16 pool of the call gets a guard slot on each side (:func:`_fp16_twins`)."""

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
        guard_slots=guard_slots,
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


# ---------------------------------------------------------------------------
# FP16 pool: the rounding model and the twin check
# ---------------------------------------------------------------------------


def _philox4x32(seed: int, counters: np.ndarray, rounds: int) -> np.ndarray:
    """``[len(counters), 4]`` uint32 words of Philox-4x32 of ``rounds`` rounds at the counters ``(lo32(c), hi32(c),
    0, 0)`` under the key ``(lo32(seed), hi32(seed))`` of the int64 seed's two's-complement bits."""

    bits = int(seed) & 0xFFFF_FFFF_FFFF_FFFF
    m32 = np.uint64(MASK32)
    k0, k1 = np.uint64(bits & MASK32), np.uint64(bits >> 32)
    c = np.asarray(counters, dtype=np.uint64).reshape(-1)
    c0, c1 = c & m32, c >> np.uint64(32)
    c2, c3 = np.zeros_like(c0), np.zeros_like(c0)
    for _ in range(rounds):
        p0, p1 = c0 * np.uint64(PHILOX_M0), c2 * np.uint64(PHILOX_M1)
        c0, c1, c2, c3 = (
            ((p1 >> np.uint64(32)) ^ c1 ^ k0) & m32,
            p1 & m32,
            ((p0 >> np.uint64(32)) ^ c3 ^ k1) & m32,
            p0 & m32,
        )
        k0 = (k0 + np.uint64(PHILOX_W0)) & m32
        k1 = (k1 + np.uint64(PHILOX_W1)) & m32
    return np.stack([c0, c1, c2, c3], axis=-1).astype(np.uint32)


def _row_noise13(seed: int, rounds: int, row_offsets: np.ndarray) -> np.ndarray:
    """``[len(row_offsets), K]`` 13-bit random values of the stores of whole state rows whose first elements sit at
    the element offsets ``row_offsets`` (from the pool view's base) in the read slot: column ``k`` takes word
    ``(k % 4) // 2 + 2 (k // 64)`` of the draw at ``row_offset + (k % 64 & ~3)``, bits 12:0 for an even ``k`` and
    28:16 for an odd one."""

    rows = np.asarray(row_offsets, dtype=np.uint64).reshape(-1)
    columns = np.arange(0, K // 2, 4, dtype=np.uint64)
    words = _philox4x32(seed, rows[:, None] + columns[None, :], rounds).reshape(
        len(rows), len(columns), 4
    )
    k = np.arange(K)
    picked = words[:, (k % 64) // 4, (k % 4) // 2 + 2 * (k // 64)]
    odd = (k % 2 == 1)[None, :]
    return np.where(odd, picked >> np.uint32(16), picked) & np.uint32(0x1FFF)


def _cvt_rs_f16(x: np.ndarray, r13: np.ndarray) -> np.ndarray:
    """FP16 bits (uint16) of PTX ``cvt.rs.f16x2.f32`` on finite FP32 elements with the 13 random bits ``r13``: the
    magnitude in units of 2^-13 FP16 ulp (the exponent rebased for a normal FP16; for a subnormal the significand
    shifted to the 2^-37 grid, lower bits dropped), plus the random bits, truncated to the FP16 grid; a sum past the
    largest finite value is an infinity."""

    a = np.asarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    sign, a = a & np.uint64(0x8000_0000), a & np.uint64(0x7FFF_FFFF)
    e = a >> np.uint64(23)
    m = (a & np.uint64(0x7F_FFFF)) | np.uint64(0x80_0000)
    shift = np.clip(np.int64(113) - e.astype(np.int64), 0, 63).astype(np.uint64)
    scaled = np.where(
        e >= np.uint64(113),
        a - np.uint64(112 << 23),
        np.where(e >= np.uint64(90), m >> shift, np.uint64(0)),
    )
    h = np.minimum(
        (scaled + np.asarray(r13, dtype=np.uint64)) >> np.uint64(13), np.uint64(0x7C00)
    )
    return ((sign >> np.uint64(16)) | h).astype(np.uint16)


def _fp16_twins(c: Case) -> tuple[Case, Case]:
    """``(c16, c32)``: the call of ``c`` over an FP16 pool (``c``'s own when it is FP16, else ``c``'s pool rounded
    to nearest in the same layout) and the same call over an FP32 pool holding exactly the FP16 pool's values.
    With ``c.guard_slots`` (``_make(guard_slots=True)``, :func:`_holes`) the rounded FP16 pool is the view
    ``[1:-1]`` of a buffer with one guard slot on each side, filled with ``+-SENTINEL``: its base stays 8-byte
    aligned (one slot is a multiple of 8 bytes) and its strides stay those of a contiguous pool (multiples of four
    elements); ``c16.guards`` is the buffer and ``c16.guards_before`` its two guard slots before the call."""

    guarded = getattr(c, "guard_slots", False)
    guards = None
    if c.pool.dtype == torch.float16:
        assert not guarded
        pools = ((c.backing, c.pool), (None, c.pool.float()))
    elif c.backing is not None:
        assert not guarded
        backing16 = c.backing.half()
        backing32 = backing16.float()
        pools = ((backing16, backing16[:, ::2]), (backing32, backing32[:, ::2]))
    else:
        pool16 = c.pool.half()
        if guarded:
            guards = torch.empty(
                (pool16.shape[0] + 2, *pool16.shape[1:]),
                dtype=torch.float16,
                device=pool16.device,
            )
            guards[0].fill_(SENTINEL)
            guards[-1].fill_(-SENTINEL)
            guards[1:-1].copy_(pool16)
            pool16 = guards[1:-1]
            assert pool16.is_contiguous() and pool16.data_ptr() % 8 == 0
        pools = ((None, pool16), (None, pool16.float()))
    twins = []
    for backing, pool in pools:
        twins.append(
            Case(
                **{
                    **c.__dict__,
                    "pool": pool,
                    "pool_before": pool.clone(),
                    "backing": backing,
                    "backing_before": backing.clone() if backing is not None else None,
                    "guards": None,
                    "guards_before": None,
                }
            )
        )
    if guards is not None:
        twins[0].guards = guards
        twins[0].guards_before = guards[[0, -1]].clone()
    return twins[0], twins[1]


def _stores(c: Case) -> list[tuple[int, int]]:
    """``(slot, batch row)`` of every state head row the call stores: the writeback of each active row to its write
    slot, or its per-token snapshots; none with ``disable_state_update``."""

    read = c.read.tolist()
    stores = []
    if c.ssm is not None:
        for b, row in enumerate(c.ssm.tolist()):
            if read[b] >= 0:
                stores.extend((slot, b) for slot in row if slot >= 0)
    elif not c.disable_state_update:
        write = read if c.write is None else c.write.tolist()
        stores.extend(
            (write[b], b) for b in range(len(read)) if read[b] >= 0 and write[b] >= 0
        )
    return stores


def _expected_fp16_bits(
    c16: Case,
    c32: Case,
    slots: list[int],
    rows: list[int],
    seed: Optional[int],
    rounds: int,
) -> np.ndarray:
    """The model of the FP16 bits the call stores in ``slots`` (written by batch ``rows``): the FP32 twin's stored
    state, rounded to nearest even (``seed`` None) or by :func:`_cvt_rs_f16` with the Philox bits of the row's read
    slot."""

    x32 = c32.pool[torch.tensor(slots, device=c32.pool.device)].float()
    if seed is None:
        return x32.half().view(torch.int16).cpu().numpy().view(np.uint16)
    s0, s1, s2, _ = c16.pool.stride()
    H = c16.pool.shape[1]
    read = np.asarray([int(c16.read[b]) for b in rows], dtype=np.int64)
    offsets = (
        read[:, None, None] * s0
        + np.arange(H)[None, :, None] * s1
        + np.arange(V)[None, None, :] * s2
    )
    noise = _row_noise13(seed, rounds, offsets.reshape(-1))
    bits = _cvt_rs_f16(x32.cpu().numpy().reshape(-1, K), noise)
    return bits.reshape(len(slots), H, V, K)


_SAME_WIDTH_INT = {2: torch.int16, 4: torch.int32}


def _same_bits(a: torch.Tensor, b: torch.Tensor) -> bool:
    """Bitwise equality of two tensors of the same dtype and shape (compared as same-width integers): a signed zero
    differs from the other zero, a NaN equals the same NaN bits."""
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    width = _SAME_WIDTH_INT[a.element_size()]
    return torch.equal(a.view(width), b.view(width))


def _check_stored_roundings(
    c16: Case, c32: Case, slots: list[int], *, stochastic: bool, what: str
) -> None:
    """Every element the call stored in ``slots`` is a float16 rounding of the FP32 twin's stored element: its round
    to nearest even (``torch`` ``.half()``) without a seed; with one, one of its two float16 neighbours, or the
    element itself when it is exact (an infinity is the neighbour of 65504 for an element past it)."""

    bad = 0
    for start in range(0, len(slots), CHECK_SLOTS):
        index = torch.tensor(slots[start : start + CHECK_SLOTS], device=_device())
        got = c16.pool[index].view(torch.int16).int() & 0xFFFF
        x32 = c32.pool[index]
        nearest = x32.half()
        near = nearest.view(torch.int16).int() & 0xFFFF
        if not stochastic:
            bad += int((got != near).sum())
            continue
        # The other neighbour is one float16 step from the nearest value towards x32 (sign bit untouched).
        widened = nearest.float()
        toward = (widened.abs() < x32.abs()).int() * 2 - 1
        other = torch.where(widened == x32, near, near + toward)
        bad += int(((got != near) & (got != other)).sum())
    assert bad == 0, (
        f"{what}: {bad} stored FP16 elements are not a rounding of the FP32 twin's element"
    )


def _holes(c: Case) -> list[int]:
    """Put holes into the slot tables of ``c`` (``B >= 2``), name the slots the call must then leave untouched and
    give the call's FP16 pool a guard slot on each side (:func:`_fp16_twins`), so a store through a ``-1`` entry (or
    past the end) is caught.  With ``ssm_state_indices`` and ``T >= 2``: row 0 skips the snapshot of token 0 (a
    ``-1`` before the last token) and stores the snapshot of its last token in its own read slot; row 1 skips the
    snapshot of its last token (a ``-1`` at the last token).  With ``ssm_state_indices`` and ``T = 1``: row 0 stores
    its snapshot in its own read slot and row 1 skips it (``-1``).  Without: row 1 skips its writeback
    (``output_state_indices`` ``-1``)."""

    assert c.B >= 2, "holes in two rows"
    c.guard_slots = True
    if c.ssm is not None:
        ssm = c.ssm.clone()
        last = c.T - 1
        if c.T >= 2:
            skipped = [int(ssm[0, 0]), int(ssm[0, last]), int(ssm[1, last])]
            ssm[0, 0] = -1
        else:
            skipped = [int(ssm[0, 0]), int(ssm[1, 0])]
        ssm[0, last] = c.read[0]
        ssm[1, last] = -1
        c.ssm = ssm
        return skipped
    write = (c.read if c.write is None else c.write).clone()
    skipped = [int(write[1]), int(c.read[1])]
    write[1] = -1
    c.write = write
    return skipped


def _run_fp16(
    c: Case, *, seed: Optional[int], rounds: int = SR_ROUNDS, what: str = ""
) -> tuple[Case, torch.Tensor, Case, torch.Tensor]:
    """Run the call of ``c`` over an FP16 pool (stochastic rounding with ``seed`` and ``rounds``, round to nearest
    without a seed) and over its FP32 twin; check the FP32 twin against the FP64 reference, the FP16 call's output
    against the twin's bit for bit, the FP16 pool's untouched slots, padding and guard slots bitwise, every stored
    FP16 element against the twin's stored element (:func:`_check_stored_roundings`) and the stored elements of at
    most ``MODEL_MAX_HEADS`` state heads (a deterministic sample of larger calls, evaluated ``MODEL_CHUNK_HEADS`` at a
    time) bitwise against the rounding model of the twin's stored state.  Bitwise checks compare same-width integer
    views (:func:`_same_bits`)."""

    what = what or f"B={c.B} T={c.T} H={c.H} n={c.n} seed={seed} rounds={rounds}"
    c16, c32 = _fp16_twins(c)
    if seed is not None:
        c16.rand_seed = torch.tensor([seed], dtype=torch.int64, device=_device())
        c16.philox_rounds = rounds
    out16, _ = _run(c16)
    out32, _ = _run(c32)
    _check(c32, out32, what=f"{what}: FP32 twin")
    assert _same_bits(out16, out32), f"{what}: the output differs from the FP32 twin's"
    stores = _stores(c16)
    stored = sorted({slot for slot, _ in stores})
    untouched = torch.ones(c16.pool.shape[0], dtype=torch.bool, device=_device())
    untouched[stored] = False
    assert _same_bits(c16.pool[untouched], c16.pool_before[untouched]), (
        f"{what}: untouched slots changed"
    )
    if c16.backing is not None:
        assert _same_bits(c16.backing[:, 1::2], c16.backing_before[:, 1::2]), (
            f"{what}: pool padding changed"
        )
    if c16.guards is not None:
        assert _same_bits(c16.guards[[0, -1]], c16.guards_before), (
            f"{what}: a guard slot around the FP16 pool changed"
        )
    _check_stored_roundings(c16, c32, stored, stochastic=seed is not None, what=what)
    if len(stores) > MODEL_MAX_HEADS // c.H:
        step = len(stores) / (MODEL_MAX_HEADS // c.H)
        stores = [stores[int(i * step)] for i in range(MODEL_MAX_HEADS // c.H)]
    mismatched = modelled = 0
    per_chunk = max(1, MODEL_CHUNK_HEADS // c.H)
    for start in range(0, len(stores), per_chunk):
        part = stores[start : start + per_chunk]
        slots, rows = [slot for slot, _ in part], [b for _, b in part]
        expected = _expected_fp16_bits(c16, c32, slots, rows, seed, rounds)
        got = (
            c16.pool[torch.tensor(slots, device=_device())]
            .view(torch.int16)
            .cpu()
            .numpy()
            .view(np.uint16)
        )
        mismatched += int((got != expected).sum())
        modelled += got.size
    assert mismatched == 0, (
        f"{what}: {mismatched} of {modelled} stored FP16 elements differ from the rounding model"
    )
    return c16, out16, c32, out32


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
    """The 144 decode shapes of record (num_householder = 3), each a full correctness case."""
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


_POOL_DTYPES = {"f32": torch.float32, "f16": torch.float16}
# The launch classes of a call (``selected_module``'s ``philox_rounds`` and ``use_ssm_state_indices``): rounding to
# nearest even (0 rounds) or stochastically with every round count the API accepts, with or without snapshots.
ROUND_COUNTS = (0, *range(pkg.PHILOX_ROUNDS_MIN, pkg.PHILOX_ROUNDS_MAX + 1))
LAUNCH_CLASSES = tuple(itertools.product(ROUND_COUNTS, (False, True)))
# The round counts a stochastic call of the program cases tries, in order: the default first.
SR_ROUND_ORDER = (
    SR_ROUNDS,
    *(
        r
        for r in range(pkg.PHILOX_ROUNDS_MIN, pkg.PHILOX_ROUNDS_MAX + 1)
        if r != SR_ROUNDS
    ),
)
# The (B, H) shapes the program cases try, in order.  27 x 24 = 648 lies in the T = 1 tail-wave rows
# (4 x sm, 5 x sm] of SM100 / SM103 on 132-160 SMs (l8r4 since the band rule change); 257 x 80 covers the
# many-wave rows.
PROGRAM_SHAPES = tuple(
    itertools.product((1, 3, 7, 16, 64, 257, 27), (6, 12, 20, 24, 80))
)


def _shape_selecting(
    name: str,
    n: int,
    T: int,
    use_qk_l2norm: bool,
    state_dtype: torch.dtype,
    *,
    min_batch: int = 1,
    philox_rounds: int = 0,
    snapshot: bool = False,
) -> Optional[tuple[int, int]]:
    """The first ``(B, H)`` of :data:`PROGRAM_SHAPES` with ``B >= min_batch`` whose ``T``-token call of the launch
    class (``philox_rounds``, ``snapshot``) selects program ``name`` on this device (``None`` when none does)."""
    sms = loader.sm_count(_device().index or 0)
    for B, H in PROGRAM_SHAPES:
        if min_batch > B:
            continue
        selected = loader.selected_module(
            _arch(),
            n,
            T,
            B * H,
            sm_count=sms,
            use_qk_l2norm=use_qk_l2norm,
            state_dtype=state_dtype,
            philox_rounds=philox_rounds,
            use_ssm_state_indices=snapshot,
        )
        if selected == name:
            return B, H
    return None


def _record_fp16_launches(monkeypatch) -> list[str]:
    """Record the name of every FP16-pool program the package launches from now on (the FP32 launches are not
    recorded)."""
    launched: list[str] = []
    original = loader.run_module

    def recording(name, values, **kwargs):
        if loader.MODULES[name]["build"].get("state_dtype", "f32") == "f16":
            launched.append(name)
        return original(name, values, **kwargs)

    monkeypatch.setattr(loader, "run_module", recording)
    return launched


# The store paths of one FP16 call (:func:`_store_paths`): its writeback to the read slot, to another slot or skipped
# (``-1``), and its snapshots skipped before the last token, skipped at the last token or stored in the row's read slot.
STORE_PATHS = (
    "writeback_read_slot",
    "distinct_write",
    "write_minus1",
    "ssm_minus1_nonlast",
    "ssm_minus1_last",
    "ssm_to_read_slot",
)


def _required_store_paths(T: int) -> set[str]:
    """The store paths every FP16 program must take at ``T`` tokens: all of :data:`STORE_PATHS` (at ``T = 1`` the only
    token is the last, so there is no skipped snapshot before it)."""
    return set(STORE_PATHS) - ({"ssm_minus1_nonlast"} if T == 1 else set())


def _store_paths(c: Case) -> set[str]:
    """The store paths the slot tables of ``c`` take in its active rows (none with ``disable_state_update``)."""
    read = c.read.tolist()
    active = [b for b, slot in enumerate(read) if slot >= 0]
    paths = set()
    if c.ssm is not None:
        for b in active:
            row = c.ssm[b].tolist()
            if any(slot < 0 for slot in row[:-1]):
                paths.add("ssm_minus1_nonlast")
            if row[-1] < 0:
                paths.add("ssm_minus1_last")
            if read[b] in row:
                paths.add("ssm_to_read_slot")
    elif not c.disable_state_update:
        write = read if c.write is None else c.write.tolist()
        for b in active:
            if write[b] == read[b]:
                paths.add("writeback_read_slot")
            elif write[b] < 0:
                paths.add("write_minus1")
            else:
                paths.add("distinct_write")
    return paths


# The store paths each call kind of :func:`_fp16_store_paths` takes in its active rows (:func:`_holes`).
KIND_PATHS = {
    "writeback": ("writeback_read_slot",),
    "distinct write": ("distinct_write", "write_minus1"),
    "snapshots": ("ssm_minus1_nonlast", "ssm_minus1_last", "ssm_to_read_slot"),
}


def _class_owner(
    name: str,
    n: int,
    T: int,
    use_qk_l2norm: bool,
    kind: str,
    rounding: str,
) -> Optional[tuple[int, tuple[int, int]]]:
    """``(philox_rounds, (B, H))`` of the first call of ``kind`` in ``rounding`` (``"rne"``: 0 rounds; ``"sr"``: the
    round counts of :data:`SR_ROUND_ORDER`, the default first) at ``T`` tokens that selects FP16 program ``name``,
    or ``None`` when no such call does: the program does not own that launch class."""
    rounds_tried = (0,) if rounding == "rne" else SR_ROUND_ORDER
    for rounds in rounds_tried:
        shape = _shape_selecting(
            name,
            n,
            T,
            use_qk_l2norm,
            torch.float16,
            min_batch=1 if kind == "writeback" else 2,
            philox_rounds=rounds,
            snapshot=kind == "snapshots",
        )
        if shape is not None:
            return rounds, shape
    return None


def _fp16_store_paths(
    name: str, n: int, T: int, use_qk_l2norm: bool, launched: list[str]
) -> tuple[list[int], list[dict], dict[tuple[str, int], set[str]]]:
    """Run every store path FP16 program ``name`` owns (``launched`` records the FP16 launches, see
    :func:`_record_fp16_launches`): at its ``T``, and at ``T = 1`` when one-token calls select it too, rounding to
    nearest even and stochastically, three call kinds over a guarded FP16 pool: (a) the writeback to the read slot;
    (b) distinct write slots with row 1 skipping its writeback (``B >= 2``, :func:`_holes`); (c) per-token snapshots
    with the holes of :func:`_holes` (``B >= 2``).  A call kind and rounding form a launch class; the program runs
    the classes it owns (:func:`_class_owner`: the first shape of :data:`PROGRAM_SHAPES` and round count that select
    it), and every class it does not own must select a program with its twin key (its sibling, ``snapshot_noise`` or
    ``draw_share``, or the sibling's plain program) on some shape.  Each call is checked by :func:`_run_fp16` (FP32
    twin, rounding model with the read slot's counter, every stored slot, untouched slots, guard slots), must launch
    ``name`` and leave the slots its holes skip unchanged.  Returns the token counts run, one record per call
    (program launched, call kind, ``T``, ``B``, ``H``, rounding, round count and the store paths of its slot tables,
    :func:`_store_paths`) and the store paths of the owned call kinds per (rounding, ``T``)."""
    key = _twin_key(loader.MODULES[name])
    partners = [
        other
        for other in _route_names_of(name)
        if other != name and _twin_key(loader.MODULES[other]) == key
    ]
    keys = {"writeback": 22, "distinct write": 48, "snapshots": 47}
    owners = {}
    token_counts = (T, 1) if T != 1 else (T,)
    for t in token_counts:
        for kind, rounding in itertools.product(keys, ("rne", "sr")):
            owner = _class_owner(name, n, t, use_qk_l2norm, kind, rounding)
            if owner is not None:
                owners[t, kind, rounding] = owner
            elif t == T:
                assert any(
                    _class_owner(other, n, t, use_qk_l2norm, kind, rounding)
                    for other in partners
                ), (
                    f"neither {name} nor a program with its twin key ({partners}) runs {kind} calls in {rounding} "
                    f"at T={t}"
                )
    tokens = sorted({t for t, _, _ in owners}, reverse=True)
    owned: dict[tuple[str, int], set[str]] = {}
    records = []
    for (t, kind, rounding), (rounds, (Bc, Hc)) in owners.items():
        owned.setdefault((rounding, t), set()).update(KIND_PATHS[kind])
        call = _make(
            Bc,
            t,
            Hc,
            n,
            generator=_generator(keys[kind], n, t, Bc, Hc),
            use_qk_l2norm=use_qk_l2norm,
            distinct_write=kind == "distinct write",
            scatter_tokens=kind == "snapshots",
            guard_slots=True,
        )
        skipped = [] if kind == "writeback" else _holes(call)
        seed = None if rounding == "rne" else SR_SEED
        what = f"{name} {kind} T={t} B={Bc} H={Hc} seed={seed} rounds={rounds}"
        launched.clear()
        c16, _, _, _ = _run_fp16(call, seed=seed, rounds=rounds, what=what)
        assert launched == [name], (what, launched)
        assert c16.guards is not None, what
        if skipped:
            assert _same_bits(c16.pool[skipped], c16.pool_before[skipped]), (
                what,
                skipped,
            )
        records.append(
            dict(
                program=launched[0],
                kind=kind,
                T=t,
                B=Bc,
                H=Hc,
                rounding=rounding,
                rounds=rounds,
                paths=_store_paths(c16),
            )
        )
    return tokens, records, owned


def _route_names_of(name: str) -> list[str]:
    """Every FP16 record of the local architecture's routes that hold program ``name``."""
    names = []
    for route in loader.ROUTES[_arch()].values():
        if name in route:
            names.extend(
                other
                for other in route
                if other not in names
                and loader.MODULES[other]["build"].get("state_dtype") == "f16"
            )
    return names


def _missing_store_paths(
    name: str,
    tokens: list[int],
    records: list[dict],
    owned: dict[tuple[str, int], set[str]],
) -> dict[tuple[str, int], list[str]]:
    """The store paths of :func:`_required_store_paths` among those of the call kinds the program owns (``owned``,
    per rounding and token count) that no recorded call launching ``name`` took (empty when the program took every
    path of its classes in both roundings at every token count it ran)."""
    missing = {}
    for rounding, t in itertools.product(("rne", "sr"), tokens):
        key = (name, rounding, t)
        seen = set()
        for record in records:
            if (record["program"], record["rounding"], record["T"]) == key:
                seen |= record["paths"]
        required = _required_store_paths(t) & owned.get((rounding, t), set())
        if required - seen:
            missing[rounding, t] = sorted(required - seen)
    return missing


def _program_cases() -> list:
    """One (n, T, B, H) selecting each exported program of the local architecture, from the registry alone; an
    FP16-pool program is selected and launched over an FP16 pool, in the first launch class that selects it (round
    to nearest without snapshots, else stochastic calls with snapshots: a sibling owns stochastic calls only)."""
    if not _supported():
        return []
    arch = _arch()
    sms = loader.sm_count(_device().index or 0)
    classes = ((0, False), *((rounds, True) for rounds in SR_ROUND_ORDER))
    cases = []
    for name, record in sorted(loader.MODULES.items()):
        if arch not in record["arches"] or record["role"] != "decode":
            continue  # the prefill records: tests/gdp/test_cake_gdp_prefill.py
        build = record["build"]
        dtype = _POOL_DTYPES[build.get("state_dtype", "f32")]
        n = build["num_householder"]
        T = build["t_steps"] if build["t_steps"] is not None else build["max_t_steps"]
        use_qk_l2norm = build.get("use_qk_l2norm", True)
        bound = loader.max_num_tokens(arch, n, use_qk_l2norm, state_dtype=dtype)
        if build["t_steps"] is None and bound > T:
            T = bound
        if build["t_steps"] is None:
            # The runtime program is the fallback: a T no static program owns selects it.
            static = {
                loader.MODULES[other]["build"]["t_steps"]
                for other in loader.route_modules(arch, n, use_qk_l2norm)
            }
            T = next(
                t
                for t in (T, *range(2, build["max_t_steps"] + 1))
                if t not in static and t <= build["max_t_steps"]
            )
        found = next(
            (
                (B, H)
                for (rounds, snapshot), (B, H) in itertools.product(
                    classes if dtype == torch.float16 else classes[:1], PROGRAM_SHAPES
                )
                if loader.selected_module(
                    arch,
                    n,
                    T,
                    B * H,
                    sm_count=sms,
                    use_qk_l2norm=use_qk_l2norm,
                    state_dtype=dtype,
                    philox_rounds=rounds,
                    use_ssm_state_indices=snapshot,
                )
                == name
            ),
            (0, 0),
        )
        cases.append(pytest.param(name, n, T, *found, use_qk_l2norm, dtype, id=name))
    return cases


@pytest.mark.parametrize("name,n,T,B,H,use_qk_l2norm,state_dtype", _program_cases())
def test_every_program_is_launched(
    name: str,
    n: int,
    T: int,
    B: int,
    H: int,
    use_qk_l2norm: bool,
    state_dtype: torch.dtype,
    monkeypatch,
) -> None:
    """Every exported program of the local architecture is selected by some call of the domain and is correct.  An
    FP16-pool program runs every store path of the launch classes it owns (:func:`_fp16_store_paths`), at its ``T``
    and at ``T = 1`` when one-token calls select it too; a class it does not own selects its sibling
    (``snapshot_noise`` or ``draw_share``) or the sibling's plain program, whose own case runs it.  The test records
    the program each call launches and the store paths of its slot tables, prints the coverage and fails unless the
    program took every path of its classes (:func:`_required_store_paths`) in each rounding at each of those token
    counts."""
    assert B > 0, f"no (B, H) of the domain selects {name}"
    if state_dtype != torch.float16:
        c = _make(
            B,
            T,
            H,
            n,
            generator=_generator(22, n, T, B, H),
            use_qk_l2norm=use_qk_l2norm,
        )
        out, _ = _run(c)
        _check(c, out, what=name)
        return
    launched = _record_fp16_launches(monkeypatch)
    tokens, records, owned = _fp16_store_paths(name, n, T, use_qk_l2norm, launched)
    for r in records:
        paths = " ".join(sorted(r["paths"]))
        print(
            f"FP16_STORE_PATHS {r['program']} {r['rounding']}{r['rounds'] or ''} T={r['T']} "
            f"{r['kind']} B={r['B']} H={r['H']}: {paths}"
        )
    assert records, f"{name} owns no launch class"
    missing = _missing_store_paths(name, tokens, records, owned)
    assert not missing, f"{name} missed store paths: {missing}"


def test_fp16_store_paths_are_covered_per_twin_key() -> None:
    """Per twin key on the local architecture (an FP16 program and its sibling, when it has one),
    every call kind of :func:`_fp16_store_paths` in both roundings at the programs' token count selects one of them on
    some shape, so the union of their cases of :func:`test_every_program_is_launched` runs every path of
    :func:`_required_store_paths` in both roundings."""
    arch = _arch()
    if not loader.supported_num_householder(arch, state_dtype=torch.float16):
        pytest.skip("no FP16-pool program on this architecture")
    tokens = {case.values[0]: case.values[2] for case in _program_cases()}
    groups: dict[tuple, set[str]] = {}
    for name in tokens:
        record = loader.MODULES[name]
        if record["build"].get("state_dtype") == "f16":
            groups.setdefault(_twin_key(record), set()).add(name)
    with_sibling = 0
    for key, names in sorted(groups.items()):
        build = loader.MODULES[min(names)]["build"]
        n, l2 = build["num_householder"], build.get("use_qk_l2norm", True)
        (T,) = {tokens[name] for name in names}
        flags = sorted(_is_sibling(loader.MODULES[name]) for name in names)
        assert flags in ([False], [False, True]), (key, sorted(names))
        with_sibling += flags == [False, True]
        for kind, rounding in itertools.product(KIND_PATHS, ("rne", "sr")):
            assert any(
                _class_owner(name, n, T, l2, kind, rounding) for name in sorted(names)
            ), (key, kind, rounding, sorted(names))
    assert with_sibling > 0, "no FP16 program has a sibling here"


# ---------------------------------------------------------------------------
# FP16 state pool
# ---------------------------------------------------------------------------


def test_rounding_model_self_checks() -> None:
    """The test's own model: Philox-4x32-10 gives Random123's known answer at the zero key and counter, and the
    ``cvt.rs`` model equals ``floor(|x| / ulp + r / 8192) * ulp`` (FP64, the binade's FP16 ulp, Inf from 2^16)
    on normal and subnormal FP16 magnitudes, both signs and random 13-bit values."""
    words = _philox4x32(0, np.zeros(1, dtype=np.uint64), 10)
    assert tuple(int(w) for w in words[0]) == PHILOX_KAT
    rng = np.random.default_rng(SEED)
    exponents = rng.integers(-30, 17, size=200_000)
    x = (rng.random(200_000) + 0.5) * np.exp2(exponents)
    x = np.where(rng.random(200_000) < 0.5, -x, x).astype(np.float32)
    r13 = rng.integers(0, 8192, size=x.size).astype(np.uint32)
    v = np.abs(x.astype(np.float64))
    _, exponent = np.frexp(v)
    ulp = np.ldexp(1.0, np.maximum(exponent - 1, -14) - 10)
    magnitude = np.floor(v / ulp + r13 / 8192.0) * ulp
    magnitude = np.where(magnitude >= 65536.0, np.inf, magnitude)
    want = (
        np.where(np.signbit(x), -magnitude, magnitude)
        .astype(np.float16)
        .view(np.uint16)
    )
    assert np.array_equal(_cvt_rs_f16(x, r13), want)


def _fp16_matrix() -> list:
    """Calls over an FP16 pool covering every FP16 body: the T = 1 bands (row partition, one-wave preload, l8r4,
    the many-wave program), static T = 4, 8 and 16 with writeback, distinct write slots, snapshots, holes (skipped
    snapshots, a skipped writeback: :func:`_holes`), the runtime programs of n = 1, 2, 3 and 4, the two
    ``snapshot_noise`` siblings (stochastic calls with snapshots at T = 16 above 2 x SM and, on the runtime program of
    n = 3, from 10 rounds outside (3 x SM, 4 x SM] and from 11 within it; on SM107 from 16 within it and from 11
    above 124 x SM), a padded pool, padding rows and ``disable_state_update``."""
    rows = [
        # (n, T, H, B, options)
        (3, 1, 10, 16, {}),  # 160 state heads: the row partition
        (3, 1, 24, 7, {"padding_row": True}),  # 168: the one-wave preload program
        (3, 1, 20, 64, {"distinct_write": True}),  # 1280: l8r4
        (3, 1, 80, 257, {}),  # 20560 > 64 x SM on up to 320 SMs: the many-wave program
        (3, 4, 12, 16, {"scatter_tokens": True}),
        (3, 4, 12, 16, {"scatter_tokens": True, "holes": True}),
        (3, 8, 10, 7, {"distinct_write": True}),
        (3, 8, 10, 7, {"distinct_write": True, "holes": True}),
        # 480 > 2 x SM on up to 216 SMs: l8r4; a stochastic call stores snapshots there, so it runs the l8r4
        # snapshot_noise sibling (one draw per launch into shared memory), a call that rounds to nearest even runs l8r4
        (3, 16, 24, 20, {"scatter_tokens": True}),
        (3, 16, 24, 20, {"scatter_tokens": True, "holes": True}),
        (3, 16, 10, 3, {"scatter_tokens": True}),  # l8r2: the shared snapshot noise
        (3, 5, 12, 7, {"padded_pool": True}),
        (3, 2, 6, 5, {"disable_state_update": True}),
        (1, 3, 6, 5, {}),
        (2, 9, 6, 3, {"scatter_tokens": True}),
        # the runtime program of n = 3: its snapshot_noise sibling runs the snapshots of 10 rounds or more at
        # these state heads (below 3 x SM)
        (3, 10, 6, 3, {"scatter_tokens": True}),
        (3, 13, 20, 16, {"scatter_tokens": True, "holes": True}),
        (4, 2, 6, 4, {"padding_row": True}),
    ]
    return [
        pytest.param(
            n,
            T,
            H,
            B,
            options,
            id=f"n{n}-T{T}-H{H}-B{B}-{'-'.join(options) or 'plain'}",
        )
        for n, T, H, B, options in rows
    ]


@pytest.mark.parametrize("rounding", ("rne", "sr10", "sr5", "sr16"))
@pytest.mark.parametrize("n,T,H,B,options", _fp16_matrix())
def test_fp16_pool_rounds_the_fp32_state(
    n: int, T: int, H: int, B: int, options: dict, rounding: str
) -> None:
    """An FP16 pool, rounding to nearest even without a seed or stochastically with 10 (the default), 5 or 16
    Philox rounds: the output equals the FP32 twin's bit for bit, every stored element is a rounding of the twin's
    stored FP32 element, and the stored elements of up to ``MODEL_MAX_HEADS`` state heads (all of them below that)
    equal the model's rounding of it; the slots a hole skips stay untouched."""
    _skip_unless_route(n, T, state_dtype=torch.float16)
    options = dict(options)
    padding_row = options.pop("padding_row", False)
    holes = options.pop("holes", False)
    c = _make(B, T, H, n, generator=_generator(40, n, T, H, B), **options)
    if padding_row:
        c.read[1] = -1
    skipped = _holes(c) if holes else []
    seed = None if rounding == "rne" else SR_SEED - 3 * T
    rounds = SR_ROUNDS if rounding in ("rne", "sr10") else int(rounding[2:])
    c16, _, _, _ = _run_fp16(c, seed=seed, rounds=rounds)
    if holes:
        assert _same_bits(c16.pool[skipped], c16.pool_before[skipped]), skipped


@pytest.mark.parametrize("rounding", ("rne", "sr10"))
@pytest.mark.parametrize("n,H,B", ((1, 6, 3), (3, 10, 4)))
def test_fp16_pool_overflow_is_stored_as_infinity(
    n: int, H: int, B: int, rounding: str
) -> None:
    """A state that rounds past the largest float16 (65504) is stored as an infinity of its sign, exactly where the
    rounding model puts one (every element of these calls is modelled), and the output is unaffected: bit for bit
    the FP32 twin's.  ``T = 1``; the first sub-step's values are scaled by 2^20, so the state after it is about
    ``beta v k^T``, past 65504 for a share of the elements; later sub-steps get ``beta = sigmoid(-30)`` and leave
    it.  (n = 1: the runtime program; n = 3 with 40 state heads: the row partition.)"""
    _skip_unless_route(n, 1, state_dtype=torch.float16)
    c = _make(B, 1, H, n, generator=_generator(46, n, H, B))
    c.v[:, :, 0] *= 2.0**20
    c.b[:, :, 1:] = -30.0
    seed = None if rounding == "rne" else SR_SEED
    c16, out16, c32, _ = _run_fp16(c, seed=seed, what=f"overflow n={n} {rounding}")
    assert B * H <= MODEL_MAX_HEADS  # the model compared every stored element
    slots = c16.read.long()
    stored, x32 = c16.pool[slots].float(), c32.pool[slots]
    assert torch.isfinite(x32).all() and torch.isfinite(out16.float()).all()
    big = x32.abs() >= 2.0**16  # past 65504 for either rounding
    assert big.any() and (x32.abs() < 65504).any()
    assert torch.equal(stored[big], torch.where(x32[big] > 0, torch.inf, -torch.inf))
    assert (stored[big] > 0).any() and (stored[big] < 0).any()
    assert torch.isfinite(stored[x32.abs() < 65504]).all()


def test_fp16_seed_contract() -> None:
    """A fixed seed gives bitwise reproducible stores; another seed (a negative one), or 5 rounds instead of 10,
    gives other stores; without a seed the stores round to nearest even, and the stochastic stores differ from
    them in a share of the elements close to the expected one (a rounding is not a no-op)."""
    _skip_unless_route(3, 4, state_dtype=torch.float16)
    c = _make(16, 4, 20, 3, generator=_generator(41))
    pools = {}
    for label, seed, rounds in (
        ("a", SR_SEED, 10),
        ("a_again", SR_SEED, 10),
        ("b", -(2**40) - 7, 10),
        ("a_r5", SR_SEED, 5),
        ("rne", None, 10),
    ):
        c16, _, _, _ = _run_fp16(c, seed=seed, rounds=rounds, what=label)
        pools[label] = c16.pool.clone()
    assert _same_bits(pools["a"], pools["a_again"])
    for other in ("b", "a_r5", "rne"):
        assert not torch.equal(pools["a"], pools[other]), other
    written = c.read.long()
    flips = (pools["a"][written] != pools["rne"][written]).float().mean().item()
    assert 0.15 < flips < 0.35, flips


def test_fp16_graph_replay_reads_the_seed_on_the_device(monkeypatch) -> None:
    """A captured call reads the seed tensor on the device: after an in-place update of the seed, a replay stores
    the rounding of the new seed (and the first seed's bits again after it is restored)."""
    _graph_replay(2, 12, 7, False, monkeypatch)


def test_fp16_graph_replay_on_a_snapshot_noise_sibling(monkeypatch) -> None:
    """As :func:`test_fp16_graph_replay_reads_the_seed_on_the_device` for a stochastic call with snapshots at
    T = 16 on 480 > 2 x SM state heads (on up to 216 SMs), which runs the ``l8r4`` ``snapshot_noise`` sibling:
    every launch, eager and captured, is that program, and each replay stores the eager call's bits for its seed."""
    _graph_replay(16, 24, 20, True, monkeypatch)


def _graph_replay(T: int, H: int, B: int, snapshots: bool, monkeypatch) -> None:
    """Capture one FP16 call (writeback, or snapshots with ``snapshots``) and replay it with two seeds written in
    place (see :func:`test_fp16_graph_replay_reads_the_seed_on_the_device`); the launches must all run one program,
    a ``snapshot_noise`` sibling exactly when the call stores snapshots."""
    _skip_unless_route(3, T, state_dtype=torch.float16)
    c = _make(
        B, T, H, 3, generator=_generator(42 + snapshots), scatter_tokens=snapshots
    )
    c16, _ = _fp16_twins(c)
    c16.rand_seed = torch.tensor([SR_SEED], dtype=torch.int64, device=_device())
    c16.philox_rounds = SR_ROUNDS
    c16.output = torch.zeros(B, T, H, V, device=_device(), dtype=torch.bfloat16)
    launched = _record_fp16_launches(monkeypatch)
    expected = {}
    for seed in (SR_SEED, SR_SEED + 1):
        c16.pool.copy_(c16.pool_before)
        c16.rand_seed.fill_(seed)
        _run(c16)
        torch.cuda.synchronize(_device())
        expected[seed] = c16.pool.clone()
    assert not torch.equal(expected[SR_SEED], expected[SR_SEED + 1])
    c16.rand_seed.fill_(SR_SEED)
    c16.pool.copy_(c16.pool_before)
    stream = torch.cuda.Stream(device=_device())
    stream.wait_stream(torch.cuda.current_stream(_device()))
    with torch.cuda.stream(stream):
        _run(c16)  # warm-up on the capture stream
    torch.cuda.current_stream(_device()).wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _run(c16)
    for seed in (SR_SEED + 1, SR_SEED):
        c16.pool.copy_(c16.pool_before)
        c16.rand_seed.fill_(seed)
        graph.replay()
        torch.cuda.synchronize(_device())
        assert _same_bits(c16.pool, expected[seed]), seed
    assert len(launched) == 4 and len(set(launched)) == 1, launched
    sibling = bool(loader.MODULES[launched[0]]["build"].get("snapshot_noise"))
    assert sibling == snapshots, launched


def test_fp16_pool_layout() -> None:
    """An FP16 pool view whose base is 8-byte aligned and whose strides are multiples of four elements is used in
    place (the Philox counter is an element offset from the view's base, not from its storage); a 4-byte aligned
    base or strides that are not multiples of four elements are refused before anything is launched."""
    _skip_unless_route(3, 4, state_dtype=torch.float16)
    c = _make(5, 4, 6, 3, generator=_generator(43), pool_size=9)
    c16, _ = _fp16_twins(c)
    numel = c16.pool.numel()
    flat = torch.zeros(numel + 8, dtype=torch.float16, device=_device())
    shifted = flat[4 : 4 + numel].view(c16.pool.shape)  # an 8-byte offset
    shifted.copy_(c16.pool_before)
    c.pool, c.pool_before = shifted, shifted.clone()
    _run_fp16(c, seed=SR_SEED, what="8-byte aligned FP16 pool view")
    assert _same_bits(flat[:4], torch.zeros_like(flat[:4]))
    assert _same_bits(flat[4 + numel :], torch.zeros_like(flat[4 + numel :]))
    misaligned = flat[2 : 2 + numel].view(c16.pool.shape)  # a 4-byte offset
    with pytest.raises(ValueError, match="aligned"):
        _run(Case(**{**c16.__dict__, "pool": misaligned}))
    wide = torch.zeros(9, 6, V, K + 2, dtype=torch.float16, device=_device())
    with pytest.raises(ValueError, match="multiples"):
        _run(Case(**{**c16.__dict__, "pool": wide[..., :K]}))


def test_fp16_pool_on_an_unaligned_storage_base() -> None:
    """An FP16 pool whose storage base is 2 bytes past an 8-byte boundary (an ``UntypedStorage`` slice) and whose view
    starts 3 elements into it, so the view's base is 8-byte aligned again: its storage offset admits no int64 view of
    its first elements, so the round-to-nearest call passes a one-element placeholder as the never-read seed (the
    helper checked directly; an ordinary pool keeps the zero-copy view).  The call runs, and its output and stored
    state equal bit for bit those of the same call on an ordinary pool holding the same values; the bytes around
    the view stay untouched."""
    _skip_unless_route(3, 4, state_dtype=torch.float16)
    c = _make(5, 4, 6, 3, generator=_generator(49), pool_size=9)
    c16, _ = _fp16_twins(c)
    shape, numel = c16.pool.shape, c16.pool.numel()
    buffer = torch.zeros(numel + 8, dtype=torch.float16, device=_device())
    storage = buffer.untyped_storage()[2:]  # base 2 bytes past the buffer's
    pool = torch.empty(0, dtype=torch.float16, device=_device())
    pool.set_(storage, 3, shape)
    assert storage.data_ptr() % 8 == 2 and pool.storage_offset() == 3
    assert pool.data_ptr() == buffer[4:].data_ptr() and pool.is_contiguous()
    pool.copy_(c16.pool_before)
    seed, rounds = pkg._rounding(pool, None, SR_ROUNDS)
    assert rounds == 0 and seed.dtype == torch.int64 and seed.numel() == 1
    assert seed.data_ptr() != pool.data_ptr()
    seed, rounds = pkg._rounding(c16.pool, None, SR_ROUNDS)
    assert rounds == 0 and seed.data_ptr() == c16.pool.data_ptr()
    shifted = Case(**{**c16.__dict__, "pool": pool, "pool_before": pool.clone()})
    out, _ = _run(shifted)
    out_ordinary, _ = _run(c16)
    assert not _same_bits(c16.pool, c16.pool_before)
    assert _same_bits(out, out_ordinary) and _same_bits(pool, c16.pool)
    assert _same_bits(buffer[:4], torch.zeros_like(buffer[:4]))
    assert _same_bits(buffer[4 + numel :], torch.zeros_like(buffer[4 + numel :]))


def test_fp16_rejects_invalid_rounding_arguments() -> None:
    """The seed is a one-element int64 tensor on the pool's device and the round count an int (not a bool) from 5
    to 16; anything else is refused before anything is launched."""
    _skip_unless_route(3, 2, state_dtype=torch.float16)
    c = _make(3, 2, 6, 3, generator=_generator(44), pool_size=8)
    c16, _ = _fp16_twins(c)
    seed = torch.tensor([SR_SEED], dtype=torch.int64, device=_device())
    bad = [
        (ValueError, dict(rand_seed=seed.to(torch.int32))),
        (
            ValueError,
            dict(rand_seed=torch.tensor([1, 2], dtype=torch.int64, device=_device())),
        ),
        (ValueError, dict(rand_seed=seed.cpu())),
        (TypeError, dict(rand_seed=5)),
        (ValueError, dict(rand_seed=seed, philox_rounds=4)),
        (ValueError, dict(rand_seed=seed, philox_rounds=17)),
        (ValueError, dict(rand_seed=seed, philox_rounds=0)),
        (TypeError, dict(rand_seed=seed, philox_rounds=True)),
        (TypeError, dict(rand_seed=seed, philox_rounds=10.0)),
    ]
    for error, changes in bad:
        args, kwargs = c16.arguments()
        with pytest.raises(error):
            cake_gated_delta_product_mtp(*args, **{**kwargs, **changes})
        assert _same_bits(c16.pool, c16.pool_before), changes
    args, kwargs = c16.arguments()  # a zero-dimensional seed is one element too
    cake_gated_delta_product_mtp(*args, **kwargs, rand_seed=seed.reshape(()))


def test_fp16_pool_is_refused_without_fp16_programs() -> None:
    """On an architecture without FP16-pool programs (SM90: no hardware ``cvt.rs``) an FP16 pool is refused."""
    if loader.supported_num_householder(_arch(), state_dtype=torch.float16):
        pytest.skip("this device has FP16-pool programs")
    _skip_unless_route(3, 2)
    c = _make(3, 2, 6, 3, generator=_generator(45), pool_size=8)
    c16, _ = _fp16_twins(c)
    with pytest.raises(ValueError, match="float16"):
        _run(c16)
    c16.rand_seed = torch.tensor([SR_SEED], dtype=torch.int64, device=_device())
    c16.philox_rounds = SR_ROUNDS
    with pytest.raises(ValueError, match="float16"):
        _run(c16)
    assert _same_bits(c16.pool, c16.pool_before)


# The build flags of an FP16 program's siblings: the snapshot-only shared noise and the shared draw.
SIBLING_FLAGS = ("snapshot_noise", "draw_share")


def _is_sibling(record: dict) -> bool:
    """Whether a decode record is a sibling of an FP16 program (one of :data:`SIBLING_FLAGS` set)."""
    return any(record["build"].get(flag) for flag in SIBLING_FLAGS)


def _twin_key(record: dict) -> tuple:
    """A decode record's identity without its pool dtype and its sibling flags: an FP16 record, its siblings
    (``snapshot_noise``, ``draw_share``) and their FP32 twin share it."""
    build = {
        k: v
        for k, v in record["build"].items()
        if k not in ("state_dtype", *SIBLING_FLAGS)
    }
    return (record["body"], tuple(sorted(build.items())))


# The launch-class keys a select cell of an FP16 record may carry, the wave keys of a cell (``_selects``), and the
# band a runtime program serves.
CLASS_KEYS = ("rounds_at_least", "rounds_at_most", "snapshot")
WAVE_KEYS = (
    "ctas_per_sm_above",
    "ctas_per_sm_at_most",
    "ctas_per_sm_at_least",
    "ctas_per_sm_below",
)
WHOLE_BAND = {"above": 0, "at_most": None, "kind": "state_heads_between"}


def _cells(rule) -> list:
    """The cells of ``rule``: none for ``None``, else its one cell or its list."""
    return [] if rule is None else rule if isinstance(rule, list) else [rule]


def _band_cells(rule, dropped: tuple = ()) -> list:
    """The cells of ``rule`` without their launch-class keys and the keys ``dropped``, each once, in order; ``None``
    (a runtime program without cells) is the whole band."""
    cells: list = []
    for cell in (
        [WHOLE_BAND] if rule is None else rule if isinstance(rule, list) else [rule]
    ):
        band = {k: v for k, v in cell.items() if k not in (*CLASS_KEYS, *dropped)}
        if band not in cells:
            cells.append(band)
    return cells


def _restricted(rule, arches: list) -> object:
    """``rule`` as it applies on ``arches``: cells naming none of them dropped, ``arches`` leaves narrowed (dropped
    when they name every one), a one-cell list unwrapped."""
    if rule is None:
        return None
    kept = []
    for cell in rule if isinstance(rule, list) else [rule]:
        cell = dict(cell)
        if "arches" in cell:
            named = [arch for arch in cell["arches"] if arch in arches]
            if not named:
                continue
            if sorted(named) == sorted(arches):
                del cell["arches"]
            else:
                cell["arches"] = named
        kept.append(cell)
    return kept[0] if len(kept) == 1 else kept


def test_fp16_records_are_twins_of_the_fp32_records() -> None:
    """Every FP16-pool record ships on SM100 / SM103 / SM107 only, in the routes of its FP32 twin (same body and
    build keywords but ``state_dtype`` and the sibling flags), with the twin's band on each of its architectures
    (its cells there without their launch-class keys and the wave keys the twin's band lacks, at which a program
    and its sibling may part their launch classes), the twin's grid rule and the seed and round-count
    arguments.  Only FP16 cells carry launch-class keys; a ``snapshot_noise`` sibling's cells each admit stochastic
    calls with snapshots only, a ``draw_share`` sibling's stochastic calls only, and a sibling has a plain FP16
    record of the same twin key in the same routes.
    On each of SM100, SM103 and SM107, every FP32 decode record of a route has its plain FP16 twin in that route,
    delivered for that architecture, so an FP16 pool has a route wherever an FP32 pool has one.  For every probe and every launch class (round count 0 or
    5 to 16, with or without snapshots) the FP16 dispatch selects a program with the twin key of the FP32
    dispatch's program: the sibling exactly where one of the sibling's cells admits the call, else the plain twin."""
    fp32 = {
        _twin_key(r): name
        for name, r in loader.MODULES.items()
        if r["role"] == "decode" and "state_dtype" not in r["build"]
    }
    twins: dict[str, str] = {}
    siblings: dict[str, list[str]] = {}
    for name, record in loader.MODULES.items():
        if record["role"] != "decode":
            continue
        rule = record["select_rule"]
        cells = [] if rule is None else rule if isinstance(rule, list) else [rule]
        if "state_dtype" not in record["build"]:
            assert not set(SIBLING_FLAGS) & set(record["build"]), name
            assert not any(set(CLASS_KEYS) & set(cell) for cell in cells), name
            continue
        assert record["build"]["state_dtype"] == "f16", name
        assert set(record["arches"]) <= {"sm_100a", "sm_103a", "sm_107a"}, name
        other = fp32[_twin_key(record)]
        twin = loader.MODULES[other]
        if record["build"].get("snapshot_noise"):
            assert record["build"]["snapshot_noise"] is True and cells, name
            for cell in cells:
                assert cell["snapshot"] is True, (name, cell)
                assert cell["rounds_at_least"] >= pkg.PHILOX_ROUNDS_MIN, (name, cell)
            siblings.setdefault(other, []).append(name)
        elif record["build"].get("draw_share"):
            assert record["build"]["draw_share"] is True and cells, name
            for cell in cells:
                assert cell["rounds_at_least"] >= pkg.PHILOX_ROUNDS_MIN, (name, cell)
            siblings.setdefault(other, []).append(name)
        else:
            assert other not in twins, (name, twins.get(other))
            twins[other] = name
        assert set(record["arches"]) <= set(twin["arches"]), name
        for arch in record["arches"]:
            band = _restricted(twin["select_rule"], [arch])
            spelled = {key for cell in _cells(band) for key in cell}
            dropped = tuple(key for key in WAVE_KEYS if key not in spelled)
            cells = _band_cells(_restricted(record["select_rule"], [arch]), dropped)
            assert cells == _band_cells(band), (name, arch)
        assert record["grid_rule"] == twin["grid_rule"], name
        assert record["logical"]["rand_seed"] == "rand_seed", name
        assert record["logical"]["philox_rounds"] == "philox_rounds", name
        plan = [tuple(entry) for entry in record["arg_plan"]]
        assert ("buffer", "rand_seed") in plan and (
            "parameter",
            "philox_rounds",
        ) in plan, name
        for arch in record["arches"]:
            for key, names in loader.ROUTES[arch].items():
                assert (name in names) == (other in names), (arch, key, name)
    for other, names in siblings.items():
        assert other in twins, (other, names)
        for name in names:
            assert (
                loader.MODULES[name]["arches"] == loader.MODULES[twins[other]]["arches"]
            ), name
    for arch in ("sm_100a", "sm_103a", "sm_107a"):
        for key, names in loader.ROUTES[arch].items():
            if key.startswith("decode_nh"):
                for name in names:
                    if "state_dtype" not in loader.MODULES[name]["build"]:
                        assert name in twins, (arch, key, name)
                        assert twins[name] in names, (arch, key, name)
                        assert arch in loader.MODULES[twins[name]]["arches"], (
                            arch,
                            key,
                            name,
                        )
        for l2 in (True, False):
            assert loader.supported_num_householder(
                arch, l2, state_dtype=torch.float16
            ) == loader.supported_num_householder(arch, l2), (arch, l2)
    for arch, routes in loader.ROUTES.items():
        for key in routes:
            if not key.startswith("decode_nh"):
                continue
            n = int(key[len("decode_nh")])
            l2 = not key.endswith("_raw")
            if not loader.supported_num_householder(
                arch, l2, state_dtype=torch.float16
            ):
                continue
            for T in range(1, loader.max_num_tokens(arch, n, l2) + 1):
                for sms in SM_COUNTS:
                    for state_heads in _STATE_HEAD_PROBES:
                        picked = loader.selected_module(
                            arch, n, T, state_heads, sm_count=sms, use_qk_l2norm=l2
                        )
                        for rounds, snapshot in LAUNCH_CLASSES:
                            admitting = [
                                name
                                for name in siblings.get(picked, [])
                                if loader._selects(
                                    loader.MODULES[name]["select_rule"],
                                    state_heads,
                                    sms,
                                    arch,
                                    philox_rounds=rounds,
                                    snapshot=snapshot,
                                )
                            ]
                            assert len(admitting) <= 1, admitting
                            expected = admitting[0] if admitting else twins[picked]
                            assert expected == loader.selected_module(
                                arch,
                                n,
                                T,
                                state_heads,
                                sm_count=sms,
                                use_qk_l2norm=l2,
                                state_dtype=torch.float16,
                                philox_rounds=rounds,
                                use_ssm_state_indices=snapshot,
                            ), (arch, key, T, state_heads, sms, rounds, snapshot)
    assert siblings, "no sibling (snapshot_noise or draw_share) in the registry"


def test_every_record_binds_the_logical_vocabulary() -> None:
    """Every decode record maps its buffers onto ``LOGICAL_TENSORS`` and its scalars onto ``LOGICAL_SCALARS``."""
    assert set(loader.LOGICAL_TENSORS).isdisjoint(loader.LOGICAL_SCALARS)
    for name, record in loader.MODULES.items():
        if record["role"] != "decode":
            continue
        for kind, key in record["arg_plan"]:
            if kind == "buffer":
                assert record["logical"][key] in loader.LOGICAL_TENSORS, (name, key)
            elif kind == "parameter":
                assert record["logical"][key] in loader.LOGICAL_SCALARS, (name, key)


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
    with pytest.raises(ValueError):  # pool dtype (float16 has its own tests)
        call(pool=c.pool.bfloat16())
    with pytest.raises(ValueError):  # a seed with a float32 pool
        call(rand_seed=torch.tensor([7], dtype=torch.int64, device=device))
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
    f16 = loader.supported_num_householder(arch, state_dtype=torch.float16)
    assert is_cake_gdp_supported(_device(), state_dtype=torch.float16) == bool(f16)
    assert set(f16) <= set(loader.supported_num_householder(arch))
    for n in f16:
        bound = loader.max_num_tokens(arch, n, state_dtype=torch.float16)
        assert bound == loader.max_num_tokens(arch, n)
        assert is_cake_gdp_supported(_device(), n, bound, state_dtype=torch.float16)
        assert not is_cake_gdp_supported(
            _device(), n, bound + 1, state_dtype=torch.float16
        )
    assert not is_cake_gdp_supported(_device(), state_dtype=torch.bfloat16)


def test_route_names_by_state_dtype_are_cached() -> None:
    """The dispatch's records of one route and pool dtype (``_route_names``, cached) equal a fresh filter of
    ``ROUTES`` by ``build["state_dtype"]`` for every architecture (and one without routes), route and pool dtype,
    and a repeated query returns the cached tuple.  A NumPy or torch scalar ``num_householder`` / ``num_tokens`` /
    ``use_qk_l2norm`` (``np.int64(3)``, ``np.array(3)``, ``torch.tensor(3)``) gives the answers of the ``int`` /
    ``bool`` through ``_route_names``, ``route_applies`` and ``is_cake_gdp_supported``, and adds no cache entry."""
    for arch in (*loader.ROUTES, "sm_80"):
        for n, l2 in itertools.product(NUM_HOUSEHOLDER, (True, False)):
            names = loader.ROUTES.get(arch, {}).get(loader.route_key(n, l2), [])
            for dtype, key in loader.STATE_DTYPES.items():
                want = tuple(
                    name
                    for name in names
                    if loader.MODULES[name]["build"].get("state_dtype", "f32") == key
                )
                got = loader._route_names(arch, n, l2, dtype)
                assert got == want, (arch, n, l2, dtype)
                assert loader._route_names(arch, n, l2, dtype) is got
    arch, capability = _arch(), _capability()
    want = {}
    for l2, dtype in itertools.product((True, False), loader.STATE_DTYPES):
        want[l2, dtype] = (
            loader._route_names(arch, 3, l2, dtype),
            loader.route_applies(
                device_capability=capability,
                num_householder=3,
                num_tokens=3,
                use_qk_l2norm=l2,
                state_dtype=dtype,
            ),
            is_cake_gdp_supported(_device(), 3, use_qk_l2norm=l2, state_dtype=dtype),
            is_cake_gdp_supported(_device(), 3, 3, use_qk_l2norm=l2, state_dtype=dtype),
        )
    entries = loader._route_names_cached.cache_info().currsize
    scalars = (np.int64(3), np.array(3), torch.tensor(3))
    flags = {l2: (np.bool_(l2), torch.tensor(l2)) for l2 in (True, False)}
    for value, ((l2, dtype), answers) in itertools.product(scalars, want.items()):
        for flag in (l2, *flags[l2]):
            got = (
                loader._route_names(arch, value, flag, dtype),
                loader.route_applies(
                    device_capability=capability,
                    num_householder=value,
                    num_tokens=value,
                    use_qk_l2norm=flag,
                    state_dtype=dtype,
                ),
                is_cake_gdp_supported(
                    _device(), value, use_qk_l2norm=flag, state_dtype=dtype
                ),
                is_cake_gdp_supported(
                    _device(), value, value, use_qk_l2norm=flag, state_dtype=dtype
                ),
            )
            assert got == answers, (repr(value), repr(flag), dtype)
            assert got[0] is answers[0]
    assert loader._route_names_cached.cache_info().currsize == entries


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


# Band edges and wave bounds of the route tables: B x HV probes for selected_module (those of the SM90 / SM100 /
# SM103 parts, the domain of :data:`DEFAULT_PICKS_OF_RECORD`, then those of SM107 and the wave bounds of the runtime
# n = 3 snapshot_noise cells: 3 x SM, 4 x SM and 124 x SM).
_RECORD_STATE_HEAD_PROBES = (
    1,
    8,
    160,
    161,
    # the two-wave edge 2 x sm and +1 for 132 / 148 / 152 / 160 SMs (the l8r2 / l8r4 edge at T = 16 and the
    # lower edge of the T = 16 snapshot_noise cell)
    264,
    265,
    296,
    297,
    304,
    305,
    320,
    321,
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
_STATE_HEAD_PROBES = (
    *_RECORD_STATE_HEAD_PROBES,
    # the two-wave edge 2 x sm and +1, the one-wave edge 4 x sm and +1 and the tail-wave edge 5 x sm and +1 of
    # T = 1 for 212 / 216 SMs
    424,
    425,
    432,
    433,
    848,
    849,
    864,
    865,
    1060,
    1061,
    1080,
    1081,
    # the two-head edge 32 x sm and the many-wave edge 64 x sm, -1 / 0 / +1, for 212 / 216 SMs
    6783,
    6784,
    6785,
    6911,
    6912,
    6913,
    13567,
    13568,
    13569,
    13823,
    13824,
    13825,
    # the wave bounds of the runtime n = 3 snapshot_noise sibling's cells (within (3 x sm, 4 x sm] it starts at 11
    # rounds on SM100 / SM103 and at 16 on SM107): 3 x sm and +1 for every SM count of :data:`SM_COUNTS`, 4 x sm and
    # +1 for 132 / 160 SMs (those of 148 / 152 / 212 / 216 SMs are above)
    396,
    397,
    444,
    445,
    456,
    457,
    480,
    481,
    636,
    637,
    648,
    649,
    528,
    529,
    640,
    641,
    # the 124 x sm edge and +1 of the sibling's SM107 cells (above it, from 11 rounds instead of 10), for every SM
    # count of :data:`SM_COUNTS`
    16368,
    16369,
    18352,
    18353,
    18848,
    18849,
    19840,
    19841,
    26288,
    26289,
    26784,
    26785,
)


def test_route_tables_are_consistent() -> None:
    """Every route name lists exported modules of its architecture; per pool dtype and launch class (an FP32 call
    with or without snapshots; an FP16 call rounding to nearest or with 5 to 16 rounds, with or without snapshots)
    exactly one program admits every probe (``selected_module`` raises on a tie and on no program), and the classes
    of one call select one body and variant; ``ARCH_BY_CAPABILITY`` names the supported capabilities only."""

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
            dtypes = {
                loader.MODULES[name]["build"].get("state_dtype", "f32")
                for name in names
            }
            assert dtypes <= set(_POOL_DTYPES), (key, dtypes)
            for dtype in (_POOL_DTYPES[d] for d in sorted(dtypes)):
                # An FP32 pool's calls have 0 rounds; an FP16 pool's calls every launch class.
                classes = (
                    LAUNCH_CLASSES
                    if dtype == torch.float16
                    else ((0, False), (0, True))
                )
                bound = loader.max_num_tokens(
                    arch, n, not key.endswith("_raw"), state_dtype=dtype
                )
                for T in range(1, bound + 1):
                    for sms in SM_COUNTS:
                        for state_heads in _STATE_HEAD_PROBES:
                            # selected_module raises unless exactly one program admits the call
                            picks = {
                                loader.selected_module(
                                    arch,
                                    n,
                                    T,
                                    state_heads,
                                    sm_count=sms,
                                    use_qk_l2norm=not key.endswith("_raw"),
                                    state_dtype=dtype,
                                    philox_rounds=rounds,
                                    use_ssm_state_indices=snapshot,
                                )
                                for rounds, snapshot in classes
                            }
                            # one body and variant per call: the classes of a call differ in a
                            # sibling flag only
                            assert (
                                len({_twin_key(loader.MODULES[name]) for name in picks})
                                == 1
                            ), (arch, key, T, state_heads, sms, sorted(picks))


# The number of probes of :func:`_default_pick_table` and the SHA-256 of its lines before the launch classes existed
# (the registry of FlashInfer #9720): a call of the default launch class (no seed, no snapshots) keeps its program.
DEFAULT_PICKS_OF_RECORD = (
    58880,
    "230aba7391a6d99ab133305bf824fda5d9012556aecb713bc382b6ad13acc784",
)
# The domain of that table: the architectures and SM counts of #9720 and :data:`_RECORD_STATE_HEAD_PROBES`
# (sm_107a: :func:`test_sm107a_selects_the_sm100a_programs`).
_RECORD_ARCHES = ("sm_100a", "sm_103a", "sm_90a")
_RECORD_SM_COUNTS = (132, 148, 152, 160)


def _default_pick_table() -> list[str]:
    """One line ``arch route dtype T sm_count state_heads program`` per decode route of :data:`_RECORD_ARCHES`, pool
    dtype, token count, SM count of :data:`_RECORD_SM_COUNTS` and :data:`_RECORD_STATE_HEAD_PROBES` probe: the program
    ``selected_module`` picks with its default keywords."""
    lines = []
    assert set(_RECORD_ARCHES) <= set(loader.ROUTES)
    for arch in sorted(_RECORD_ARCHES):
        for key in sorted(loader.ROUTES[arch]):
            if not key.startswith("decode_nh"):
                continue
            n, l2 = int(key[len("decode_nh")]), not key.endswith("_raw")
            for dtype_key in sorted(_POOL_DTYPES):
                dtype = _POOL_DTYPES[dtype_key]
                bound = loader.max_num_tokens(arch, n, l2, state_dtype=dtype)
                for T in range(1, bound + 1):
                    for sms in _RECORD_SM_COUNTS:
                        for state_heads in _RECORD_STATE_HEAD_PROBES:
                            name = loader.selected_module(
                                arch,
                                n,
                                T,
                                state_heads,
                                sm_count=sms,
                                use_qk_l2norm=l2,
                                state_dtype=dtype,
                            )
                            lines.append(
                                f"{arch} {key} {dtype_key} {T} {sms} {state_heads} {name}"
                            )
    return lines


def test_default_launch_class_keeps_the_picks_of_record() -> None:
    """Every probe of every decode route and pool dtype picks, with ``selected_module``'s default keywords (0 rounds,
    no snapshots), the program it picked before the launch classes existed (:data:`DEFAULT_PICKS_OF_RECORD`)."""
    lines = _default_pick_table()
    digest = hashlib.sha256("\n".join(lines).encode()).hexdigest()
    assert (len(lines), digest) == DEFAULT_PICKS_OF_RECORD


def test_sm107a_selects_the_sm100a_programs() -> None:
    """sm_107a ships the decode programs, routes and bands of sm_100a: its decode routes are sm_100a's, and on every
    route, pool dtype, token count, SM count, probe and launch class it selects the program sm_100a selects, except
    where the sm_107a cells of a ``snapshot_noise`` sibling admit fewer round counts (the runtime n = 3 sibling's
    within (3 x SM, 4 x SM] from 16 rounds, where sm_100a's take 11, and above 124 x SM from 11, where sm_100a's take
    10): there it selects the sibling's plain FP16 program (same body and variant)."""

    def decode_routes(arch: str) -> dict:
        return {
            key: names
            for key, names in loader.ROUTES[arch].items()
            if key.startswith("decode_")
        }

    routes = decode_routes("sm_107a")
    assert routes == decode_routes("sm_100a")
    for key in sorted(routes):
        n, l2 = int(key[len("decode_nh")]), not key.endswith("_raw")
        dtypes = {
            loader.MODULES[name]["build"].get("state_dtype", "f32")
            for name in routes[key]
        }
        for dtype in (_POOL_DTYPES[d] for d in sorted(dtypes)):
            classes = (
                LAUNCH_CLASSES if dtype == torch.float16 else ((0, False), (0, True))
            )
            bound = loader.max_num_tokens("sm_107a", n, l2, state_dtype=dtype)
            assert bound == loader.max_num_tokens("sm_100a", n, l2, state_dtype=dtype)
            for T, sms, state_heads, (rounds, snapshot) in itertools.product(
                range(1, bound + 1), SM_COUNTS, _STATE_HEAD_PROBES, classes
            ):
                p100, p107 = (
                    loader.selected_module(
                        arch,
                        n,
                        T,
                        state_heads,
                        sm_count=sms,
                        use_qk_l2norm=l2,
                        state_dtype=dtype,
                        philox_rounds=rounds,
                        use_ssm_state_indices=snapshot,
                    )
                    for arch in ("sm_100a", "sm_107a")
                )
                if p100 == p107:
                    continue
                r100, r107 = loader.MODULES[p100], loader.MODULES[p107]
                where = (key, dtype, T, sms, state_heads, rounds, snapshot, p100, p107)
                assert r100["build"].get("snapshot_noise") is True, where
                assert not r107["build"].get("snapshot_noise"), where
                assert r107["build"].get("state_dtype") == "f16", where
                assert _twin_key(r100) == _twin_key(r107), where


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
    # FP16-pool kernels: every state access (vector casts and the snapshot stores' inline
    # PTX) is at most F16_STATE_ALIGNMENT_BYTES wide, with no asynchronous copies.
    state_cast = re.compile(r"reinterpret_cast<(?:const )?(\w+)\s*\*>\(state\b")
    state_asm = re.compile(
        r'st\.global\.v(\d)\.b(\d+) \[%0\][^"]*"\s*::\s*"l"\(state\b'
    )
    f16_widest = 0
    f16_kernels = [k for k in kernels if "__half* __restrict__ state" in k.read_text()]
    assert len(f16_kernels) == sum(
        1
        for record in loader.MODULES.values()
        if record["role"] == "decode" and record["build"].get("state_dtype") == "f16"
    )
    for kernel in f16_kernels:
        text = kernel.read_text()
        assert "cp.async" not in text, kernel.name
        sizes = [widths[cast] for cast in state_cast.findall(text)]
        sizes += [int(n) * int(bits) // 8 for n, bits in state_asm.findall(text)]
        assert sizes, kernel.name
        f16_widest = max(f16_widest, *sizes)
    if f16_kernels:
        assert f16_widest == backend.F16_STATE_ALIGNMENT_BYTES == 8


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


@pytest.mark.parametrize("sm", (148, 152, 212, 216))
def test_t1_band_edges_follow_the_sm_count(sm: int) -> None:
    """T = 1, three Householder steps, on `sm` SMs: the row-partition program up to 160
    state heads; on SM100 / SM103 / SM107 the one-wave program l8r4p takes (160, 4 x sm], l8r4
    takes (4 x sm, 64 x sm] (the tail-wave rows (4 x sm, 5 x sm] included: the
    one-wave-of-5 band is off) and l8r4w5 takes every count above 64 x sm, while SM90
    ships neither of the two and keeps l8r4 above 160.  The choice depends on
    (arch, T, B x HV, sm_count) only."""
    assert set(loader.ROUTES) == {"sm_90a", "sm_100a", "sm_103a", "sm_107a"}
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


@pytest.mark.parametrize("sm", SM_COUNTS)
def test_t4_band_follows_the_architecture(sm: int) -> None:
    """T = 4, three Householder steps: SM90 runs the one-CTA-per-SM program l4r2 up to
    `sm` state heads and l8r4 above it; SM100 / SM103 / SM107 run l8r4 on every state-head count
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


# From T = 5 on, l8r2 serves up to k x SM state heads (one wave of k CTAs per SM).
L8R2_CTAS_PER_SM = {
    "sm_90a": {5: 2, 6: 2, 7: 2, 8: 2, 16: 1},
    "sm_100a": {5: 2, 6: 2, 7: 2, 8: 2, 16: 2},
    "sm_103a": {5: 2, 6: 2, 7: 2, 8: 2, 16: 2},
    "sm_107a": {5: 2, 6: 2, 7: 2, 8: 2, 16: 2},
}


@pytest.mark.parametrize("sm", SM_COUNTS)
def test_multi_token_bands_follow_the_architecture(sm: int) -> None:
    """Three Householder steps at T = 2, 3, 5, 6, 7, 8 and 16: l8r4 serves every
    state-head count at T = 2 and 3; from T = 5 on, l8r2 serves up to k x `sm` state
    heads (``L8R2_CTAS_PER_SM``) and l8r4 the counts above."""
    for arch in loader.ROUTES:
        for T in (2, 3, 5, 6, 7, 8, 16):
            k = L8R2_CTAS_PER_SM[arch][T] if T >= 5 else 0
            for state_heads in (1, 24, k * sm, k * sm + 1, 4 * sm, 20480):
                if state_heads < 1:
                    continue
                want = ("lanerow", "l8r2" if state_heads <= k * sm else "l8r4")
                assert _static_pick(arch, 3, T, state_heads, sm) == want, (
                    arch,
                    T,
                    state_heads,
                    sm,
                )


@pytest.mark.parametrize("T", (2, 3, 4, 5, 6, 7, 8, 16))
def test_static_programs_partition_every_token_count(T: int) -> None:
    """For every static token count above one, exactly one static program of three
    Householder steps admits every state-head count on every SM count the kernel may
    meet, so the runtime program never serves these token counts."""
    for arch in loader.ROUTES:
        for sm in SM_COUNTS:
            for state_heads in (*range(1, 1300), 1536, 4096, 20480):
                body, variant = _static_pick(arch, 3, T, state_heads, sm)
                assert body == "lanerow", (arch, T, state_heads, sm, body, variant)


def test_t1_static_programs_partition_every_sm_count() -> None:
    """Exactly one T = 1 static program admits every state-head count on every SM count
    the kernel may meet (`selected_module` raises when two admit), so the runtime program is
    never the T = 1 choice of three Householder steps."""
    for arch in loader.ROUTES:
        for sm in SM_COUNTS:
            for state_heads in (*range(1, 1300), 1536, 4096, 20480):
                body, _variant = _static_pick(arch, 3, 1, state_heads, sm)
                assert body != "runtime", (arch, state_heads, sm)
